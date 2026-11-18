"""
har3d_models_v2.py -- ST-GCN + training for the small datasets, on the
corrected topology.

Training protocol (final version, revised after review):

  * EARLY STOPPING on validation NLL, not a fixed epoch count. Each model is
    trained for up to `max_epochs`, the learning rate is halved on a validation
    plateau, and the weights with the lowest validation NLL are restored. Every
    model is therefore evaluated at its own best convergence point, so dataset
    size is not confounded with how over- or under-trained the model is.
    Validation NLL is the selection criterion for every loss (CE, label
    smoothing, focal) because it is a proper scoring rule.
  * MC DROPOUT with the inference dropout rate TUNED on validation. The
    training rate (0.3) is a regulariser; reusing it for MC inference on a few
    hundred samples can starve the network of capacity and flatten
    confidences. The MC rate is chosen from a grid by validation NLL of the
    MC-averaged prediction, and the chosen rate is recorded.
  * loss -- 'ce' | 'ls' (label smoothing) | 'focal'.
  * val+test logits always returned.

Topology is imported from har3d_v2 so there is exactly one definition.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from har3d_v2 import MSR_PARENTS, adjacency


class STGCNBlock(nn.Module):
    def __init__(self, cin, cout, A, stride=1, residual=True):
        super().__init__()
        self.register_buffer("A", torch.as_tensor(A, dtype=torch.float32))
        self.K = A.shape[0]
        self.gconv = nn.Conv2d(cin, cout * self.K, 1)
        self.tconv = nn.Sequential(
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, (9, 1), (stride, 1), (4, 0)),
            nn.BatchNorm2d(cout))
        self.res = (nn.Identity() if (residual and cin == cout and stride == 1)
                    else (nn.Sequential(nn.Conv2d(cin, cout, 1, (stride, 1)),
                                        nn.BatchNorm2d(cout)) if residual else None))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        r = self.res(x) if self.res is not None else 0
        N, _, T, J = x.shape
        y = self.gconv(x).view(N, self.K, -1, T, J)
        y = torch.einsum("nkctv,kvw->nctw", y, self.A)
        return self.relu(self.tconv(y) + r)


class STGCN(nn.Module):
    def __init__(self, num_classes, A, in_ch=3, base=64, dropout=0.3):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_ch * A.shape[1])
        self.layers = nn.ModuleList([
            STGCNBlock(in_ch, base, A, residual=False),
            STGCNBlock(base, base, A),
            STGCNBlock(base, base * 2, A, stride=2),
            STGCNBlock(base * 2, base * 2, A)])
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(base * 2, num_classes)

    def forward(self, x):                       # (N, C, T, J)
        N, C, T, J = x.shape
        x = self.bn(x.permute(0, 1, 3, 2).reshape(N, C * J, T))
        x = x.view(N, C, J, T).permute(0, 1, 3, 2)
        for l in self.layers:
            x = l(x)
        return self.fc(self.drop(F.adaptive_avg_pool2d(x, 1).flatten(1)))


def _loss_fn(kind, smoothing=0.1, gamma=2.0):
    if kind == "ce":
        return lambda o, y: F.cross_entropy(o, y)
    if kind == "ls":
        return lambda o, y: F.cross_entropy(o, y, label_smoothing=smoothing)
    if kind == "focal":
        def focal(o, y):
            logp = F.log_softmax(o, -1)
            lp = logp[torch.arange(len(y)), y]
            return (-(1 - lp.exp()) ** gamma * lp).mean()
        return focal
    raise ValueError(f"unknown loss {kind!r}")


@torch.no_grad()
def _forward_all(model, X, idx, bs=128, device="cpu"):
    return np.concatenate(
        [model(torch.as_tensor(X[idx[b:b + bs]]).to(device)).cpu().numpy()
         for b in range(0, len(idx), bs)], 0).astype(np.float32)


def _state_copy(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def _val_nll(model, X, idx, y, device, bs=256):
    model.eval()
    L = _forward_all(model, X, idx, bs=bs, device=device)
    z = torch.as_tensor(L, dtype=torch.float64)
    return float(F.cross_entropy(z, torch.as_tensor(y[idx])).item())


def _set_dropout(model, p=None, train=True):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            if p is not None:
                m.p = float(p)
            m.train(train)


def mc_tuned(model, X, y, split, passes, p_grid, seed, device):
    """
    MC dropout with the inference rate chosen on validation NLL.
    Returns ({'val_mc': (P,n,C), 'test_mc': ...}, chosen_p, {p: val_nll}).
    Batch-norm stays in eval mode, so only dropout is stochastic.
    """
    model.eval()
    vidx = split["val"]
    scores = {}
    for gi, p in enumerate(p_grid):
        _set_dropout(model, p, train=True)
        torch.manual_seed(seed + 10_000 + gi)
        P = np.stack([_forward_all(model, X, vidx, device=device)
                      for _ in range(passes)], 0)
        z = P - P.max(-1, keepdims=True)
        pr = np.exp(z); pr /= pr.sum(-1, keepdims=True)
        pm = pr.mean(0)
        scores[float(p)] = float(-np.log(np.clip(pm[np.arange(len(vidx)), y[vidx]],
                                                 1e-12, 1)).mean())
    best = min(scores, key=scores.get)
    _set_dropout(model, best, train=True)
    out = {}
    for sub in ("val", "test"):
        idx = split[sub]
        torch.manual_seed(seed + 20_000)
        out[sub + "_mc"] = (np.stack([_forward_all(model, X, idx, device=device)
                                      for _ in range(passes)], 0), y[idx])
    _set_dropout(model, None, train=False)
    return out, best, scores


def train_and_logits(X, y, split, num_classes, A=None, max_epochs=120, bs=32,
                     lr=1e-3, seed=0, device=None, verbose=True, loss="ce",
                     smoothing=0.1, gamma=2.0, mc_passes=0, dropout=0.3,
                     early_stop=True, patience=15, min_epochs=10,
                     mc_p_grid=(0.05, 0.1, 0.2, 0.3), epochs=None):
    """
    Train on split['train'] with early stopping on split['val'] NLL; return
    logits for val and test.

    Returns (out, model): out['val'|'test'] = (logits, labels); with
    mc_passes > 0 also out['val_mc'|'test_mc'] = (logits (P,n,C), labels);
    out['info'] records best_epoch, epochs_run, best_val_nll and mc_p.
    `epochs` (legacy) forces a fixed epoch count with no early stopping.
    Deterministic given `seed`.
    """
    if A is None:
        A = adjacency(MSR_PARENTS)
    if epochs is not None:
        max_epochs, early_stop = int(epochs), False
    if early_stop and len(split.get("val", [])) == 0:
        raise ValueError("early stopping needs a non-empty validation split")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = STGCN(num_classes, A, in_ch=X.shape[1], dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    if early_stop:
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min",
                                                         factor=0.5, patience=5)
    else:
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max_epochs)
    crit = _loss_fn(loss, smoothing, gamma)

    Xtr = torch.as_tensor(X[split["train"]])
    ytr = torch.as_tensor(y[split["train"]])

    best_nll, best_state, best_ep, since = np.inf, None, -1, 0
    ep_run = 0
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        run, tot = 0.0, 0
        for b in range(0, len(perm), bs):
            i = perm[b:b + bs]
            xb, yb = Xtr[i].to(device), ytr[i].to(device)
            opt.zero_grad(set_to_none=True)
            o = model(xb)
            l = crit(o, yb)
            l.backward()
            opt.step()
            run += float(l.detach()) * len(i)
            tot += len(i)
        ep_run = ep + 1
        if early_stop:
            v = _val_nll(model, X, split["val"], y, device)
            sch.step(v)
            if v < best_nll - 1e-6:
                best_nll, best_state, best_ep, since = v, _state_copy(model), ep_run, 0
            else:
                since += 1
            if since >= patience and ep_run >= min_epochs:
                break
        else:
            sch.step()
        if verbose and ep_run % 20 == 0:
            print(f"      ep {ep_run:3d}  train loss {run/tot:.4f}"
                  + (f"  val nll {v:.4f}" if early_stop else ""))

    if early_stop and best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_ep = ep_run
        best_nll = (_val_nll(model, X, split["val"], y, device)
                    if len(split.get("val", [])) else float("nan"))

    model.eval()
    out = {}
    for sub in ("val", "test"):
        idx = split.get(sub, np.array([], int))
        if len(idx) == 0:
            continue
        L = _forward_all(model, X, idx, device=device)
        out[sub] = (L, y[idx])
        if verbose:
            print(f"      {sub:<5} n={len(idx):5d}  top1 {(L.argmax(-1)==y[idx]).mean():.4f}")

    info = {"best_epoch": int(best_ep), "epochs_run": int(ep_run),
            "best_val_nll": float(best_nll), "early_stop": bool(early_stop),
            "mc_p": None, "mc_p_scores": None}
    if mc_passes:
        mc, p_best, scores = mc_tuned(model, X, y, split, mc_passes, mc_p_grid,
                                      seed, device)
        out.update(mc)
        info["mc_p"], info["mc_p_scores"] = p_best, scores
        if verbose:
            print(f"      mc-dropout: {mc_passes} passes, inference p={p_best} "
                  f"(tuned on val NLL)")
    out["info"] = info
    if verbose:
        print(f"      best epoch {best_ep}/{ep_run}  best val NLL {best_nll:.4f}")
    return out, model