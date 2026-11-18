"""
har3d_backbones.py -- the three graph-convolutional backbones used on the
NTU-60 ladder, built on ONE shared trunk so that only the graph layer differs.

    stgcn   ST-GCN (Yan et al., AAAI 2018): a FIXED, hand-defined adjacency.
    agcn    2s-AGCN graph layer (Shi et al., CVPR 2019): fixed adjacency
            + a freely learned graph B + a data-dependent similarity C(x).
    ctrgcn  CTR-GCN graph layer (Chen et al., ICCV 2021): channel-wise
            topology refinement -- a learnable shared topology refined per
            channel from pairwise feature differences.

Shared by all three: four blocks (widths 64, 64, 128, 128; stride 2 in block
3), a 9x1 temporal convolution with batch norm, residual connections, a
batch-norm on the input, mean pooling over time, joints and the two persons,
dropout 0.3, and a linear head. Holding the trunk fixed is deliberate: the
ladder compares GRAPH MECHANISMS at matched depth, width and training recipe,
not published end-to-end systems (CTR-GCN's multi-scale temporal module and
2s-AGCN's two-stream ensembling are intentionally not reproduced).

`graph_parameters(model)` returns the parameters that generate the adaptive
graph, which the adjacency-freezing ablation stops updating mid-training.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from har3d_v2 import adjacency

ARCHS = ("stgcn", "agcn", "agcn_frozen", "ctrgcn")


# ---------------------------------------------------------------------------
# temporal unit shared by all blocks
# ---------------------------------------------------------------------------
def _tconv(cout, stride):
    return nn.Sequential(nn.Conv2d(cout, cout, (9, 1), (stride, 1), (4, 0)),
                         nn.BatchNorm2d(cout))


def _residual(cin, cout, stride, residual):
    if not residual:
        return None
    if cin == cout and stride == 1:
        return nn.Identity()
    return nn.Sequential(nn.Conv2d(cin, cout, 1, (stride, 1)), nn.BatchNorm2d(cout))


# ---------------------------------------------------------------------------
# ST-GCN block: fixed adjacency
# ---------------------------------------------------------------------------
class STGCNBlock(nn.Module):
    def __init__(self, cin, cout, A, stride=1, residual=True):
        super().__init__()
        self.K = A.shape[0]
        self.register_buffer("A", torch.as_tensor(A, dtype=torch.float32))
        self.gconv = nn.Conv2d(cin, cout * self.K, 1)
        self.bn = nn.BatchNorm2d(cout)
        self.tconv = _tconv(cout, stride)
        self.res = _residual(cin, cout, stride, residual)

    def forward(self, x):                              # (N, C, T, J)
        N, _, T, J = x.shape
        r = self.res(x) if self.res is not None else 0
        y = self.gconv(x).view(N, self.K, -1, T, J)
        y = torch.einsum("nkctv,kvw->nctw", y, self.A)
        y = F.relu(self.bn(y))
        return F.relu(self.tconv(y) + r)


# ---------------------------------------------------------------------------
# 2s-AGCN block: A + B + alpha * C(x)
# ---------------------------------------------------------------------------
class AGCNBlock(nn.Module):
    def __init__(self, cin, cout, A, stride=1, residual=True, embed=None):
        super().__init__()
        self.K = A.shape[0]
        self.register_buffer("A", torch.as_tensor(A, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(self.K, A.shape[1], A.shape[2]))
        self.embed = embed or max(cout // 4, 8)
        self.theta = nn.Conv2d(cin, self.embed * self.K, 1)
        self.phi = nn.Conv2d(cin, self.embed * self.K, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.gconv = nn.Conv2d(cin, cout * self.K, 1)
        self.bn = nn.BatchNorm2d(cout)
        self.tconv = _tconv(cout, stride)
        self.res = _residual(cin, cout, stride, residual)

    def forward(self, x):
        N, _, T, J = x.shape
        r = self.res(x) if self.res is not None else 0
        th = self.theta(x).view(N, self.K, self.embed, T, J).mean(3)
        ph = self.phi(x).view(N, self.K, self.embed, T, J).mean(3)
        C = torch.softmax(torch.einsum("nkej,nkel->nkjl", th, ph)
                          / (self.embed ** 0.5), dim=-1)
        Aeff = self.A.unsqueeze(0) + self.B.unsqueeze(0) + self.alpha * C
        y = self.gconv(x).view(N, self.K, -1, T, J)
        y = torch.einsum("nkctv,nkvw->nctw", y, Aeff)
        y = F.relu(self.bn(y))
        return F.relu(self.tconv(y) + r)


# ---------------------------------------------------------------------------
# CTR-GCN block: channel-wise topology refinement
# ---------------------------------------------------------------------------
class CTRGC(nn.Module):
    """One CTR graph convolution for one adjacency partition (Chen et al. 2021)."""

    def __init__(self, cin, cout, rel_reduction=8):
        super().__init__()
        rel = 8 if cin <= 16 else cin // rel_reduction
        self.conv1 = nn.Conv2d(cin, rel, 1)
        self.conv2 = nn.Conv2d(cin, rel, 1)
        self.conv3 = nn.Conv2d(cin, cout, 1)
        self.conv4 = nn.Conv2d(rel, cout, 1)

    def forward(self, x, A, alpha):                    # x (N, C, T, V)
        x1 = self.conv1(x).mean(-2)                    # (N, rel, V)
        x2 = self.conv2(x).mean(-2)
        x3 = self.conv3(x)                             # (N, cout, T, V)
        rel = torch.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))   # (N, rel, V, V)
        topo = self.conv4(rel) * alpha + A.unsqueeze(0).unsqueeze(0)
        return torch.einsum("ncuv,nctv->nctu", topo, x3)


class CTRGCNBlock(nn.Module):
    def __init__(self, cin, cout, A, stride=1, residual=True):
        super().__init__()
        self.K = A.shape[0]
        self.PA = nn.Parameter(torch.as_tensor(A, dtype=torch.float32).clone())
        self.convs = nn.ModuleList([CTRGC(cin, cout) for _ in range(self.K)])
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(cout)
        nn.init.constant_(self.bn.weight, 1e-6)        # start near the residual
        self.down = (nn.Identity() if cin == cout else
                     nn.Sequential(nn.Conv2d(cin, cout, 1), nn.BatchNorm2d(cout)))
        self.tconv = _tconv(cout, stride)
        self.res = _residual(cin, cout, stride, residual)

    def forward(self, x):
        r = self.res(x) if self.res is not None else 0
        y = sum(self.convs[k](x, self.PA[k], self.alpha) for k in range(self.K))
        y = F.relu(self.bn(y) + self.down(x))
        return F.relu(self.tconv(y) + r)


_BLOCKS = {"stgcn": STGCNBlock, "agcn": AGCNBlock, "agcn_frozen": AGCNBlock,
           "ctrgcn": CTRGCNBlock}

# name fragments of the parameters that GENERATE the graph, per architecture
_GRAPH_PARAM_KEYS = {
    "stgcn": (),
    "agcn": (".B", ".alpha", ".theta.", ".phi."),
    "agcn_frozen": (".B", ".alpha", ".theta.", ".phi."),
    "ctrgcn": (".PA", ".alpha", ".conv1.", ".conv2.", ".conv4."),
}


class GraphNet(nn.Module):
    """Shared trunk. Input (N, C, T, J, M); persons folded into the batch."""

    def __init__(self, arch, num_classes, A, in_ch=3, base=64, dropout=0.3):
        super().__init__()
        if arch not in _BLOCKS:
            raise ValueError(f"unknown arch {arch!r}; choose from {ARCHS}")
        self.arch = arch
        Blk = _BLOCKS[arch]
        self.bn = nn.BatchNorm1d(in_ch * A.shape[1])
        self.layers = nn.ModuleList([
            Blk(in_ch, base, A, residual=False),
            Blk(base, base, A),
            Blk(base, base * 2, A, stride=2),
            Blk(base * 2, base * 2, A)])
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(base * 2, num_classes)

    def forward(self, x):
        N, C, T, J, M = x.shape
        x = x.permute(0, 4, 1, 2, 3).reshape(N * M, C, T, J)
        x = self.bn(x.permute(0, 1, 3, 2).reshape(N * M, C * J, T))
        x = x.view(N * M, C, J, T).permute(0, 1, 3, 2)
        for layer in self.layers:
            x = layer(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = x.view(N, M, -1).mean(1)
        return self.fc(self.drop(x))


def build(arch, num_classes, parents, in_ch=3, base=64, dropout=0.3):
    return GraphNet(arch, num_classes, adjacency(parents), in_ch, base, dropout)


def graph_parameters(model):
    """Parameters that generate the (adaptive) graph; empty for ST-GCN."""
    keys = _GRAPH_PARAM_KEYS[model.arch]
    out = []
    for name, prm in model.named_parameters():
        dotted = "." + name
        if any(dotted.endswith(k) or k in dotted for k in keys):
            out.append(prm)
    return out


def freeze_graph(model):
    """Stop updating the adaptive graph. Returns the number of tensors frozen."""
    ps = graph_parameters(model)
    for prm in ps:
        prm.requires_grad_(False)
        prm.grad = None
    return len(ps)


def n_params(model):
    return sum(p.numel() for p in model.parameters())