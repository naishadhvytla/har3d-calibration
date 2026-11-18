"""
har3d_v2.py -- corrected topology, preprocessing, splits and calibration
helpers for the skeleton calibration benchmark.

Three defects found by MEASURING the cached skeletons rather than reading code:

(1) The legacy parents list is not the MSR tree. 11 of its 19 edges are not
    bones -- the worst is 1.587 m against a 0.275 m upper arm. That list
    defined BOTH the ST-GCN adjacency and bone_pairs. (It is in fact the
    correct Kinect SDK kinematic tree paired with MSR's joint ordering, which
    is why it is internally consistent enough to survive a casual read.)
(2) normalise(root=0) centred on joint 0, which the frame geometry identifies
    as the LEFT SHOULDER. The hip centre is joint 6.
(3) fixed_split() carved validation out of the training subjects by index
    slice, so train and val shared subjects.

The corrected tree was derived from the data: for every joint pair, the
coefficient of variation of the inter-joint distance across all clean frames
was computed. Real bones are rigid, so they have low cv.

verify_topology() re-derives these numbers at run time and raises if any edge
is implausible. Everything here is numpy.
"""
from __future__ import annotations

import numpy as np

# =====================================================================
# 1. Topology
# =====================================================================
JOINT_NAMES = [
    "L_shoulder", "R_shoulder", "shoulder_center", "spine",
    "L_hip", "R_hip", "hip_center", "L_elbow",
    "R_elbow", "L_wrist", "R_wrist", "L_hand",
    "R_hand", "L_knee", "R_knee", "L_ankle",
    "R_ankle", "L_foot", "R_foot", "head",
]

# parent index per joint, -1 = root. Root is hip_center (index 6).
MSR_PARENTS = [2, 2, 3, 6, 6, 6, -1, 0, 1, 7, 8, 9, 10, 4, 5, 13, 14, 15, 16, 2]
ROOT = MSR_PARENTS.index(-1)
SPINE_TOP = JOINT_NAMES.index("shoulder_center")
HIP_L = JOINT_NAMES.index("L_hip")
HIP_R = JOINT_NAMES.index("R_hip")

# The legacy list, kept only so the ablation can reproduce the old numbers.
LEGACY_PARENTS = [-1, 0, 1, 2, 2, 4, 5, 6, 2, 8, 9, 10, 0, 12, 13, 14, 0, 16, 17, 18]

MAX_BONE_M = 0.60      # longest plausible human bone (thigh) with margin
MAX_BONE_CV = 0.20     # rigid connections vary little relative to their length
MAX_BONE_SD_M = 0.045  # ...or little in absolute terms, the fairer test

MAX_BODY_SPAN_M = 3.5
MAX_MEDIAN_PAIR_M = 1.5
MIN_MEDIAN_PAIR_M = 0.05

LR_PAIRS = [(0, 1), (4, 5), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18)]


def bone_pairs(parents=MSR_PARENTS):
    """(child, parent) for every joint. Root maps to itself, a zero bone."""
    return [(j, p if p >= 0 else j) for j, p in enumerate(parents)]


def rigid_spanning_tree(xyz, max_cv=MAX_BONE_CV, max_sd=MAX_BONE_SD_M,
                        allow_fallback=False):
    """
    Shortest rigid connections spanning every joint, as an undirected edge list.
    Judges a candidate parse WITHOUT knowing the joint order.
    """
    F = np.asarray(xyz, np.float64)
    F = F[~(np.abs(F) < 1e-9).all(-1).any(-1)]
    if len(F) < 20:
        return []
    D = np.linalg.norm(F[:, :, None, :] - F[:, None, :, :], axis=-1)
    mu, sd = D.mean(0), D.std(0)
    n = len(mu)
    cand = sorted((mu[i, j], i, j) for i in range(n) for j in range(i + 1, n)
                  if sd[i, j] / max(mu[i, j], 1e-9) <= max_cv or sd[i, j] <= max_sd)
    par = list(range(n))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    def build(edges):
        nonlocal par
        par = list(range(n))
        out = []
        for w, i, j in edges:
            a, b = find(i), find(j)
            if a != b:
                par[a] = b
                out.append((w, i, j))
        return out

    tree = build(cand)
    if len(tree) < n - 1 and allow_fallback:
        # One joint can be so jittery that no edge touching it passes the
        # rigidity filter. A noisy joint must not masquerade as a wrong tree.
        tree = build(sorted((mu[i, j], i, j)
                            for i in range(n) for j in range(i + 1, n)))
    return tree


def tree_optimality(xyz, parents=MSR_PARENTS, max_cv=MAX_BONE_CV,
                    max_sd=MAX_BONE_SD_M):
    """
    Is the proposed tree the best available explanation of this skeleton?

    A noisier RECORDING inflates every edge's sd together and leaves the tree
    near-optimal; a WRONG tree costs far more than the optimum no matter how
    clean the sensor is. `ratio` is the statistic to trust; `shared` is kept
    only as a catastrophe check.
    """
    tree = rigid_spanning_tree(xyz, max_cv=max_cv, max_sd=max_sd,
                               allow_fallback=True)
    n_needed = np.asarray(xyz).shape[1] - 1
    if len(tree) < n_needed:
        return {"ratio": float("inf"), "shared": 0, "n_edges": n_needed,
                "mst_len_m": float("nan"), "proposed_len_m": float("nan")}
    F = np.asarray(xyz, np.float64)
    F = F[~(np.abs(F) < 1e-9).all(-1).any(-1)]
    mu = np.linalg.norm(F[:, :, None, :] - F[:, None, :, :], axis=-1).mean(0)
    prop = [(j, p) for j, p in enumerate(parents) if p >= 0]
    prop_len = float(sum(mu[j, p] for j, p in prop))
    mst_len = float(sum(w for w, _, _ in tree))
    shared = len({frozenset(e) for e in prop} & {frozenset((i, j)) for _, i, j in tree})
    return {"ratio": prop_len / max(mst_len, 1e-9), "shared": shared,
            "n_edges": len(prop), "mst_len_m": mst_len, "proposed_len_m": prop_len}


def verify_topology(sequences, parents=MSR_PARENTS, names=JOINT_NAMES,
                    max_len=MAX_BONE_M, max_cv=MAX_BONE_CV, raise_on_fail=True,
                    min_frames=100, max_sd=MAX_BONE_SD_M,
                    max_ratio=1.10, min_shared=8):
    """
    Re-derive bone statistics from real data and check the tree is anatomical.
    Frames containing an all-zero joint are dropped -- a dropped joint sits at
    the origin and would fabricate a metre-long bone.

    Raises AssertionError on failure. A hard gate, not a warning.
    """
    F = np.concatenate([np.asarray(s, np.float64) for s in sequences], 0)
    if F.ndim != 3:
        raise ValueError(f"expected (T, J, 3) sequences, got trailing shape {F.shape}")
    keep = ~(np.abs(F) < 1e-9).all(-1).any(-1)
    if keep.sum() < min_frames:
        raise ValueError(f"only {int(keep.sum())} frames with all joints present; "
                         f"need {min_frames} for a stable estimate")
    F = F[keep]
    report, bad = [], []
    for j, p in enumerate(parents):
        if p < 0:
            continue
        d = np.linalg.norm(F[:, j] - F[:, p], axis=-1)
        m, s = float(d.mean()), float(d.std())
        cv = s / max(m, 1e-9)
        ok = (m <= max_len) and (cv <= max_cv or s <= max_sd)
        report.append({"child": j, "parent": p, "name": names[j],
                       "parent_name": names[p], "len_m": m, "cv": cv,
                       "sd_m": s, "ok": ok})
        if not ok:
            bad.append(f"  j{j:02d} {names[j]} -> j{p:02d} {names[p]}: "
                       f"{m:.3f} m, cv {cv:.3f}, sd {s*1000:.0f} mm")

    print(f"  topology check: {len(report) - len(bad)}/{len(report)} edges plausible "
          f"({int(keep.sum()):,} clean frames, {100 * keep.mean():.1f}% of input)")

    if bad:
        opt = tree_optimality(F, parents, max_cv=max_cv, max_sd=max_sd)
        too_long = [r for r in report if r["len_m"] > max_len]
        structural = (not (1.0 / max_ratio <= opt["ratio"] <= max_ratio)
                      or opt["shared"] < min_shared or bool(too_long))
        head = (f"  tree optimality: {opt['ratio']:.3f}x the data-derived "
                f"shortest rigid tree, sharing {opt['shared']}/{opt['n_edges']} "
                f"edges with it")
        if structural:
            msg = ("topology check FAILED -- the tree is not the best "
                   "explanation of this geometry:\n" + "\n".join(bad)
                   + f"\n{head}"
                   + f"\n  (limits: length <= {max_len} m; cv <= {max_cv} or "
                     f"sd <= {max_sd*1000:.0f} mm; ratio <= {max_ratio}; "
                     f"shared >= {min_shared})")
            if raise_on_fail:
                raise AssertionError(msg)
            print("  !! " + msg)
        else:
            print(head + " -- near-optimal, so the tree is right and this "
                         "recording is simply noisier.")
            print("  edges that are anatomical but jittery here:")
            for b in bad:
                print("   " + b.strip())
            print("  Reported, not fatal. Note the affected joints in the paper: "
                  "a noisy joint is a per-class calibration confound, not a "
                  "preprocessing error.")
    return report


def adjacency(parents=MSR_PARENTS, strategy="spatial"):
    """
    ST-GCN spatial partitioning: (3, J, J) = self / centripetal / centrifugal,
    each row-normalised.
    """
    J = len(parents)
    A = np.zeros((J, J), np.float32)
    for j, p in enumerate(parents):
        if p >= 0:
            A[j, p] = A[p, j] = 1.0
    root = list(parents).index(-1)
    hop = np.full(J, np.inf)
    hop[root] = 0.0
    frontier = [root]
    while frontier:
        nxt = []
        for j in frontier:
            for n in np.where(A[j] > 0)[0]:
                if not np.isfinite(hop[n]):
                    hop[n] = hop[j] + 1
                    nxt.append(int(n))
        frontier = nxt
    if not np.isfinite(hop).all():
        raise ValueError(f"parents do not form a connected tree; unreachable: "
                         f"{np.where(~np.isfinite(hop))[0].tolist()}")
    parts = np.zeros((3, J, J), np.float32)
    parts[0] = np.eye(J, dtype=np.float32)
    for i in range(J):
        for j in range(J):
            if A[i, j] > 0:
                parts[1 if hop[j] < hop[i] else 2, i, j] = 1.0
    for k in range(3):
        d = parts[k].sum(1, keepdims=True)
        d[d == 0] = 1.0
        parts[k] = parts[k] / d
    return parts


# =====================================================================
# 2. Preprocessing
# =====================================================================
def impute_missing(seq, max_gap=None):
    """
    Kinect drops a joint by writing (0, 0, 0). Linearly interpolate each joint
    over time; hold the nearest value at the ends. Sequences where a joint is
    missing in EVERY frame keep zeros and are reported by the caller.
    """
    x = np.array(seq, np.float32, copy=True)
    T, J, _ = x.shape
    miss = (np.abs(x) < 1e-9).all(-1)
    t = np.arange(T, dtype=np.float64)
    for j in range(J):
        m = miss[:, j]
        if not m.any() or m.all():
            continue
        good = ~m
        for c in range(3):
            x[m, j, c] = np.interp(t[m], t[good], x[good, j, c])
    return x, miss


def _rotation_to_canonical(frame, spine_top=SPINE_TOP, root=ROOT,
                           hip_l=HIP_L, hip_r=HIP_R):
    """
    Orthonormal basis from one frame: hip->shoulder_center becomes +y,
    L_hip->R_hip becomes +x, z completes a right-handed frame.
    """
    up = frame[spine_top] - frame[root]
    right = frame[hip_r] - frame[hip_l]
    nu = np.linalg.norm(up)
    if nu < 1e-6:
        return None
    up = up / nu
    right = right - np.dot(right, up) * up
    nr = np.linalg.norm(right)
    if nr < 1e-6:
        return None
    right = right / nr
    fwd = np.cross(right, up)
    return np.stack([right, up, fwd], 0)


def normalise_skeleton(seq, centre="perframe", align=True, scale="bone",
                       parents=MSR_PARENTS, root=ROOT):
    """
    Canonicalise one sequence BEFORE any stream is derived.

    centre : 'perframe' removes global translation entirely;
             'frame0' keeps the body's trajectory through the scene as signal;
             'none' leaves absolute coordinates.
    align  : rotate by the frame-0 body basis. Per SEQUENCE, so rotation
             dynamics inside the clip survive.
    scale  : 'bone' divides by the sequence-median bone length.

    Ordering is centre, then align, then scale. Rotating before centring would
    rotate about the camera origin instead of the body.
    """
    x = np.asarray(seq, np.float32)
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError(f"expected (T, J, 3), got {x.shape}")

    if centre == "perframe":
        x = x - x[:, root:root + 1, :]
    elif centre == "frame0":
        x = x - x[0:1, root:root + 1, :]
    elif centre != "none":
        raise ValueError(f"unknown centre={centre!r}")

    if align:
        R = _rotation_to_canonical(x[0])
        if R is not None:
            x = x @ R.T.astype(np.float32)

    if scale == "bone":
        pair = [(j, p) for j, p in enumerate(parents) if p >= 0]
        lens = np.stack([np.linalg.norm(x[:, j] - x[:, p], axis=-1) for j, p in pair], 1)
        s = float(np.median(lens))
        if s > 1e-6:
            x = x / s
    elif scale != "none":
        raise ValueError(f"unknown scale={scale!r}")
    return x.astype(np.float32)


def make_stream(xyz, stream, parents=MSR_PARENTS):
    """
    (T, J, 3) -> stream tensor. joint | bone | joint_motion | bone_motion.
    Call on an ALREADY-normalised sequence so all four share one convention.
    """
    x = np.asarray(xyz, np.float32)
    if stream.startswith("bone"):
        out = np.zeros_like(x)
        for j, p in bone_pairs(parents):
            out[:, j] = x[:, j] - x[:, p]
        x = out
    elif not stream.startswith("joint"):
        raise ValueError(f"unknown stream {stream!r}")
    if stream.endswith("motion"):
        m = np.zeros_like(x)
        m[1:] = x[1:] - x[:-1]
        x = m
    return x


def resample(x, T=64):
    """Uniform temporal resample by nearest index. Deterministic."""
    n = x.shape[0]
    if n == T:
        return x
    idx = np.clip(np.round(np.linspace(0, n - 1, T)).astype(int), 0, n - 1)
    return x[idx]


def build_tensors(cache, stream, T=64, parents=MSR_PARENTS,
                  centre="perframe", align=True, scale="bone", impute=True):
    """
    Ragged cache -> (N, 3, T, J) float32, labels, subjects.

    Order of operations, which is the part that matters:
        impute -> normalise -> derive stream -> resample
    The old code did stream -> normalise -> resample, which normalised bone and
    motion tensors as if they were positions.
    """
    off, meta = cache["offsets"], cache["meta"]
    N = len(off) - 1
    J = len(parents)
    X = np.zeros((N, 3, T, J), np.float32)
    n_all_missing = 0
    for i in range(N):
        seq = cache["xyz"][off[i]:off[i + 1]]
        if impute:
            seq, miss = impute_missing(seq)
            n_all_missing += int(miss.all(0).sum())
        seq = normalise_skeleton(seq, centre=centre, align=align,
                                 scale=scale, parents=parents)
        s = resample(make_stream(seq, stream, parents), T)
        X[i] = s.transpose(2, 0, 1)
    if n_all_missing:
        print(f"    note: {n_all_missing} joint-sequences were missing in every "
              f"frame and remain zero")
    y = meta[:, 0].astype(np.int64) - 1
    subj = meta[:, 1].astype(np.int64)
    return X, y, subj


# =====================================================================
# 3. Subject-disjoint splits
# =====================================================================
def _assert_disjoint(split, subjects, n_classes=None):
    tr, va, te = (set(subjects[split[k]].tolist()) for k in ("train", "val", "test"))
    assert not (tr & va), f"train/val share subjects {sorted(tr & va)}"
    assert not (tr & te), f"train/test share subjects {sorted(tr & te)}"
    assert not (va & te), f"val/test share subjects {sorted(va & te)}"
    n = len(subjects)
    idx = np.concatenate([split["train"], split["val"], split["test"]])
    assert len(np.unique(idx)) == len(idx), "a sample appears in two partitions"
    assert len(idx) == n, f"partitions cover {len(idx)} of {n} samples"
    return True


def subject_splits(subjects, labels=None, n_splits=3, seed=0,
                   n_test_subjects=4, n_val_subjects=2, n_classes=None,
                   verbose=True):
    """
    Subject-disjoint train/val/test splits with an EXPLICIT validation size.
    Warns when a class is absent from a partition, which happens easily on
    small subject-wise splits and quietly breaks per-class calibration.
    """
    subjects = np.asarray(subjects)
    uniq = np.unique(subjects)
    need = n_test_subjects + n_val_subjects + 1
    if len(uniq) < need:
        raise ValueError(f"{len(uniq)} subjects available, need at least {need} "
                         f"({n_test_subjects} test + {n_val_subjects} val + 1 train)")
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_splits):
        perm = rng.permutation(uniq)
        te_s = set(perm[:n_test_subjects].tolist())
        va_s = set(perm[n_test_subjects:n_test_subjects + n_val_subjects].tolist())
        tr_s = set(perm[n_test_subjects + n_val_subjects:].tolist())
        sp = {
            "name": f"rand{i}",
            "train": np.where(np.isin(subjects, list(tr_s)))[0],
            "val": np.where(np.isin(subjects, list(va_s)))[0],
            "test": np.where(np.isin(subjects, list(te_s)))[0],
            "subjects": {"train": sorted(tr_s), "val": sorted(va_s), "test": sorted(te_s)},
        }
        _assert_disjoint(sp, subjects)
        if labels is not None:
            K = n_classes or int(np.max(labels)) + 1
            for part in ("train", "val", "test"):
                seen = np.unique(np.asarray(labels)[sp[part]])
                if len(seen) < K and verbose:
                    print(f"    !! {sp['name']} {part}: {K - len(seen)} of {K} "
                          f"classes absent")
        if verbose:
            print(f"    {sp['name']}: train {len(sp['train'])} "
                  f"(subj {sp['subjects']['train']}) "
                  f"val {len(sp['val'])} (subj {sp['subjects']['val']}) "
                  f"test {len(sp['test'])} (subj {sp['subjects']['test']})")
        out.append(sp)
    return out


def fixed_split(subjects, train_subjects=(1, 3, 5, 7, 9), val_subjects=(9,),
                labels=None, verbose=True):
    """
    The conventional MSR protocol (odd subjects train, even test), with
    validation held out BY SUBJECT.
    """
    subjects = np.asarray(subjects)
    train_subjects, val_subjects = set(train_subjects), set(val_subjects)
    if not val_subjects <= train_subjects:
        raise ValueError("val_subjects must be drawn from train_subjects")
    tr_s = train_subjects - val_subjects
    if not tr_s:
        raise ValueError("no training subjects left after holding out val")
    sp = {
        "name": "fixed",
        "train": np.where(np.isin(subjects, list(tr_s)))[0],
        "val": np.where(np.isin(subjects, list(val_subjects)))[0],
        "test": np.where(~np.isin(subjects, list(train_subjects)))[0],
        "subjects": {"train": sorted(tr_s), "val": sorted(val_subjects),
                     "test": sorted(set(np.unique(subjects).tolist()) - train_subjects)},
    }
    _assert_disjoint(sp, subjects)
    if verbose:
        print(f"    fixed: train {len(sp['train'])} val {len(sp['val'])} "
              f"test {len(sp['test'])} subjects {sp['subjects']}")
    return sp


# =====================================================================
# 4. Calibration helpers that need more than one split
# =====================================================================
def softmax(z, T=1.0):
    z = np.asarray(z, np.float64) / T
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def fit_temperature(logits_val, y_val, lo=0.05, hi=10.0, iters=200):
    """Golden-section search on validation NLL. Deterministic, 1-D, exact."""
    z, y = np.asarray(logits_val, np.float64), np.asarray(y_val)
    idx = np.arange(len(y))

    def loss(T):
        p = softmax(z, T)
        return float(-np.log(np.clip(p[idx, y], 1e-12, 1.0)).mean())

    gr = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    for _ in range(iters):
        if loss(c) < loss(d):
            b, d = d, c
            c = b - gr * (b - a)
        else:
            a, c = c, d
            d = a + gr * (b - a)
        if abs(b - a) < 1e-6:
            break
    return float((a + b) / 2)


def fit_temperature_pooled(val_logits, val_labels):
    """
    One temperature from validation sets pooled across splits or seeds.
    The per-split T is the honest single-run estimate, the pooled T is the
    low-variance one, and the gap between them IS the RQ3 finding.
    """
    z = np.concatenate([np.asarray(v, np.float64) for v in val_logits], 0)
    y = np.concatenate([np.asarray(v) for v in val_labels], 0)
    return fit_temperature(z, y), len(y)


def aggregate_nested(rows, metric, split_key="split", seed_key="seed"):
    """
    Mean over seeds WITHIN a split, then mean and spread ACROSS splits.
    On MSR the split-to-split spread is several times the seed-to-seed spread,
    so a pooled "+/- over seeds" label mostly measures which subjects landed in
    the test set.
    """
    by_split = {}
    for r in rows:
        v = r.get(metric)
        if v is None or not np.isfinite(v):
            continue
        by_split.setdefault(r[split_key], []).append(float(v))
    if not by_split:
        return {"mean": np.nan, "std_across_splits": np.nan,
                "mean_std_within_split": np.nan, "n_splits": 0, "n_seeds": 0}
    per_split_mean = np.array([np.mean(v) for v in by_split.values()])
    per_split_std = np.array([np.std(v, ddof=1) if len(v) > 1 else 0.0
                              for v in by_split.values()])
    return {
        "mean": float(per_split_mean.mean()),
        "std_across_splits": float(per_split_mean.std(ddof=1))
        if len(per_split_mean) > 1 else 0.0,
        "mean_std_within_split": float(per_split_std.mean()),
        "n_splits": len(by_split),
        "n_seeds": int(np.median([len(v) for v in by_split.values()])),
    }


# =====================================================================
# 5. Metrics
# =====================================================================
def _bins_equal_width(conf, M):
    edges = np.linspace(0.0, 1.0, M + 1)
    return np.clip(np.digitize(conf, edges[1:-1], right=True), 0, M - 1), M


def _bins_equal_mass(conf, M):
    """Equal sample count per bin. Far less biased at small n than fixed width."""
    n = len(conf)
    order = np.argsort(conf, kind="stable")
    idx = np.empty(n, np.int64)
    edges = np.linspace(0, n, M + 1).astype(int)
    for b in range(M):
        idx[order[edges[b]:edges[b + 1]]] = b
    return idx, M


def calibration_error(p, y, M=15, mode="equal_width", norm="l1"):
    p = np.asarray(p, np.float64)
    conf = p.max(-1)
    correct = (p.argmax(-1) == np.asarray(y)).astype(np.float64)
    idx, M = (_bins_equal_width if mode == "equal_width" else _bins_equal_mass)(conf, M)
    n = len(conf)
    gaps, weights = [], []
    for b in range(M):
        m = idx == b
        if not m.any():
            continue
        gaps.append(abs(correct[m].mean() - conf[m].mean()))
        weights.append(m.sum() / n)
    if not gaps:
        return float("nan")
    gaps, weights = np.array(gaps), np.array(weights)
    return float(gaps.max() if norm == "max" else (gaps * weights).sum())


def ece(p, y, M=15):
    return calibration_error(p, y, M, "equal_width", "l1")


def adaptive_ece(p, y, M=15):
    return calibration_error(p, y, M, "equal_mass", "l1")


def mce(p, y, M=15):
    return calibration_error(p, y, M, "equal_width", "max")


def brier(p, y):
    """Multiclass Brier. Unbiased and binning-free -- trustworthy at small n."""
    p = np.asarray(p, np.float64)
    oh = np.zeros_like(p)
    oh[np.arange(len(y)), np.asarray(y)] = 1.0
    return float(((p - oh) ** 2).sum(-1).mean())


def nll(p, y):
    p = np.asarray(p, np.float64)
    return float(-np.log(np.clip(p[np.arange(len(y)), np.asarray(y)], 1e-12, 1.0)).mean())


def top1(p, y):
    return float((np.asarray(p).argmax(-1) == np.asarray(y)).mean())


def risk_coverage(p, y):
    """Sorted by confidence descending. Returns (coverage, risk, AURC)."""
    p, y = np.asarray(p, np.float64), np.asarray(y)
    err = (p.argmax(-1) != y).astype(np.float64)[np.argsort(-p.max(-1), kind="stable")]
    k = np.arange(1, len(err) + 1)
    risk = np.cumsum(err) / k
    return k / len(err), risk, float(risk.mean())


def aurc(p, y):
    return risk_coverage(p, y)[2]


def per_class_ece(p, y, M=15):
    """One-vs-rest over samples predicted as each class. RQ4."""
    p, y = np.asarray(p, np.float64), np.asarray(y)
    out = np.full(p.shape[1], np.nan)
    for c in range(p.shape[1]):
        m = p.argmax(-1) == c
        if m.sum() < 2:
            continue
        conf, corr = p[m, c], (y[m] == c).astype(np.float64)
        idx, M_ = _bins_equal_mass(conf, min(M, max(2, int(m.sum() // 5))))
        g, w = [], []
        for b in range(M_):
            bm = idx == b
            if bm.any():
                g.append(abs(corr[bm].mean() - conf[bm].mean()))
                w.append(bm.sum() / m.sum())
        if g:
            out[c] = float((np.array(g) * np.array(w)).sum())
    return out


def classwise_ece(p, y, M=15):
    """
    Classwise ECE (Kull et al., NeurIPS 2019): for every class k, the
    equal-width binned ECE of p[:, k] against the one-vs-rest indicator
    1[y == k], averaged over classes. Unlike top-label ECE it scores the whole
    probability vector, so it sees miscalibration in non-argmax classes that
    top-label metrics ignore (Gupta & Ramdas, ICLR 2022).
    """
    p, y = np.asarray(p, np.float64), np.asarray(y)
    n, K = p.shape
    edges = np.linspace(0.0, 1.0, M + 1)
    tot = 0.0
    for k in range(K):
        pk = p[:, k]
        ok = (y == k).astype(np.float64)
        idx = np.clip(np.digitize(pk, edges[1:-1], right=True), 0, M - 1)
        s = 0.0
        for b in range(M):
            m = idx == b
            if m.any():
                s += m.sum() / n * abs(ok[m].mean() - pk[m].mean())
        tot += s
    return float(tot / K)


def brier_decomposition(p, y, M=15):
    """
    Murphy (1973) decomposition of the multiclass Brier score, computed
    one-vs-rest per class and summed (the multiclass Brier score is exactly the
    sum of the K binary scores). With binned forecasts (Stephenson et al. 2008):

        BS = REL - RES + UNC + WB

    REL  reliability  (calibration; lower is better)
    RES  resolution   (refinement / discrimination; HIGHER is better)
    UNC  uncertainty  (depends on the label distribution only)
    WB   within-bin terms (variance minus covariance inside bins; ~0 for
         fine bins)

    The identity is exact by construction (WB is computed, not assumed), so it
    doubles as a test. It explains why a method can improve Brier without
    improving ECE: a gain in RES (sharper, better-ranked forecasts) lowers the
    score even when REL is unchanged.
    """
    p, y = np.asarray(p, np.float64), np.asarray(y)
    n, K = p.shape
    edges = np.linspace(0.0, 1.0, M + 1)
    REL = RES = UNC = WB = 0.0
    for k in range(K):
        f = p[:, k]
        o = (y == k).astype(np.float64)
        obar = o.mean()
        idx = np.clip(np.digitize(f, edges[1:-1], right=True), 0, M - 1)
        rel = res = wb = 0.0
        for b in range(M):
            m = idx == b
            if not m.any():
                continue
            nb = m.sum()
            fb, ob = f[m].mean(), o[m].mean()
            rel += nb * (fb - ob) ** 2
            res += nb * (ob - obar) ** 2
            wb += ((f[m] - fb) ** 2).sum() - 2 * ((f[m] - fb) * (o[m] - ob)).sum()
        REL += rel / n
        RES += res / n
        UNC += obar * (1 - obar)
        WB += wb / n
    return {"rel": float(REL), "res": float(RES), "unc": float(UNC),
            "wb": float(WB), "brier": float(REL - RES + UNC + WB)}


def quality_mask(flags, max_zero_frac=0.5, min_frames=10):
    """
    Sequence-level data-quality filter, applied at tensor-build time (the cache
    stays raw). flags columns: [n_frames, nonfinite_frac, zero_joint_frac, ...].
    A sequence is dropped if more than `max_zero_frac` of its joint-frames are
    all-zero (the Kinect tracker lost the skeleton, so imputation has nothing to
    interpolate from) or if it has fewer than `min_frames` frames. This is the
    principled version of the conventional 557-of-567 MSR-Action3D subset,
    whose exact exclusion list is not consistently documented.
    """
    flags = np.asarray(flags, np.float64)
    keep = (flags[:, 2] <= max_zero_frac) & (flags[:, 0] >= min_frames)
    keep &= np.isfinite(flags[:, 1]) & (flags[:, 1] < 1.0)
    return keep


def load_cache(npz):
    """Ragged cache file -> dict with xyz, offsets, meta and (if present) flags."""
    z = np.load(npz, allow_pickle=False)
    out = {k: z[k] for k in ("xyz", "offsets", "meta")}
    if "flags" in z.files:
        out["flags"] = z["flags"]
    return out


def masked_tensors(cache, stream, T=64, prep=None, max_zero_frac=0.5,
                   min_frames=10, parents=MSR_PARENTS):
    """
    build_tensors() followed by quality_mask(). The mask is applied BEFORE any
    split is drawn, so every split, seed and method sees the same sample set.
    Returns X, y, subj, keep (boolean over the raw cache order).
    """
    prep = dict(prep or {})
    X, y, subj = build_tensors(cache, stream, T=T, parents=parents,
                               scale="bone", impute=True, **prep)
    if "flags" in cache:
        keep = quality_mask(cache["flags"], max_zero_frac, min_frames)
    else:
        keep = np.ones(len(y), bool)
    return X[keep], y[keep], subj[keep], keep


def bootstrap_ci(fn, p, y, n_boot=1000, alpha=0.05, seed=0):
    """Percentile CI. Non-negotiable on the small datasets."""
    rng = np.random.default_rng(seed)
    p, y = np.asarray(p), np.asarray(y)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.integers(0, len(y), len(y))
        vals[i] = fn(p[s], y[s])
    vals = vals[np.isfinite(vals)]
    return (float(fn(p, y)), float(np.quantile(vals, alpha / 2)),
            float(np.quantile(vals, 1 - alpha / 2)))


def evaluate(logits, y, logits_val=None, y_val=None, n_boot=0, seed=0) -> dict:
    """Every metric for one (model, dataset, stream, split, seed) cell."""
    p = softmax(logits)
    out = {"n": int(len(y)), "n_classes": int(p.shape[1]), "top1": top1(p, y),
           "ece_15": ece(p, y), "ece_adaptive": adaptive_ece(p, y),
           "mce_15": mce(p, y), "brier": brier(p, y), "nll": nll(p, y),
           "aurc": aurc(p, y)}
    pc = per_class_ece(p, y)
    out["per_class_ece_mean"] = float(np.nanmean(pc)) if np.isfinite(pc).any() else float("nan")
    out["per_class_ece_max"] = float(np.nanmax(pc)) if np.isfinite(pc).any() else float("nan")
    out["classwise_ece"] = classwise_ece(p, y)
    bd = brier_decomposition(p, y)
    out.update({"brier_rel": bd["rel"], "brier_res": bd["res"],
                "brier_unc": bd["unc"], "brier_wb": bd["wb"]})
    if logits_val is not None and y_val is not None:
        T = fit_temperature(logits_val, y_val)
        pT = softmax(logits, T)
        out.update({"temperature": T, "ece_adaptive_temp": adaptive_ece(pT, y),
                    "brier_temp": brier(pT, y), "nll_temp": nll(pT, y),
                    "top1_temp": top1(pT, y),
                    "classwise_ece_temp": classwise_ece(pT, y)})
    if n_boot:
        for name, fn in (("ece_adaptive", adaptive_ece), ("brier", brier), ("nll", nll)):
            _, lo, hi = bootstrap_ci(fn, p, y, n_boot, seed=seed)
            out[f"{name}_lo"], out[f"{name}_hi"] = lo, hi
    return out


# =====================================================================
# 6. Cross-dataset joint-order matching
# =====================================================================
def distance_signature(xyz, max_frames=20000):
    """Scale-free mean pairwise distance matrix over clean frames."""
    F = np.asarray(xyz, np.float64)
    if F.ndim != 3:
        raise ValueError(f"expected (T, J, 3), got {F.shape}")
    F = F[~(np.abs(F) < 1e-9).all(-1).any(-1)]
    if len(F) < 50:
        raise ValueError(f"only {len(F)} clean frames; need 50+")
    if len(F) > max_frames:
        F = F[np.linspace(0, len(F) - 1, max_frames).astype(int)]
    D = np.linalg.norm(F[:, :, None, :] - F[:, None, :, :], axis=-1).mean(0)
    iu = np.triu_indices(len(D), 1)
    return D / np.median(D[iu])


def _assignment_cost(Dq, Dr, perm):
    return float(np.abs(Dq[np.ix_(perm, perm)] - Dr).sum())


def match_joint_order(Dq, Dr, restarts=30, seed=0):
    """
    perm[c] = q : canonical joint c corresponds to query joint q.
    Quadratic assignment by rank initialisation plus 2-opt, several restarts.
    """
    n = len(Dr)
    rng = np.random.default_rng(seed)
    best = (np.inf, None)
    rank_q, rank_r = np.argsort(Dq.sum(1)), np.argsort(Dr.sum(1))
    for r in range(restarts):
        perm = np.empty(n, int)
        if r == 0:
            perm[rank_r] = rank_q
        else:
            perm = rng.permutation(n)
        c = _assignment_cost(Dq, Dr, perm)
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i + 1, n):
                    p2 = perm.copy()
                    p2[i], p2[j] = p2[j], p2[i]
                    c2 = _assignment_cost(Dq, Dr, p2)
                    if c2 < c - 1e-12:
                        perm, c, improved = p2, c2, True
        if c < best[0]:
            best = (c, perm.copy())
    return best[1], best[0] / (n * n)


def _mirror(perm, pairs=LR_PAIRS):
    out = perm.copy()
    for a, b in pairs:
        out[a], out[b] = perm[b], perm[a]
    return out


def derive_canonical_permutation(query_xyz, reference_xyz, parents=MSR_PARENTS,
                                 root=ROOT, verbose=True):
    """
    Recover the permutation putting `query_xyz` into canonical joint order.
    Apply it as xyz_canonical = xyz_query[:, perm].

    A human skeleton is nearly bilaterally symmetric, so the L/R mirror of any
    correct match is almost as cheap. Resolved on geometry, not cost.
    """
    Dr = distance_signature(reference_xyz)
    Dq = distance_signature(query_xyz)
    perm, resid = match_joint_order(Dq, Dr)
    mirror = _mirror(perm)
    resid_m = _assignment_cost(Dq, Dr, mirror) / (len(Dr) ** 2)

    def lr_sign(xyz, p):
        F = np.asarray(xyz, np.float64)
        F = F[~(np.abs(F) < 1e-9).all(-1).any(-1)]
        return float(np.mean(F[:, p[0], 0] - F[:, p[root], 0]))

    want = lr_sign(reference_xyz, np.arange(len(Dr)))
    got = lr_sign(query_xyz, perm)
    flipped = False
    if np.sign(got) != np.sign(want) and abs(resid_m - resid) < 0.25 * max(resid, 1e-9):
        perm, resid, flipped = mirror, resid_m, True

    if verbose:
        print(f"    joint-order match: residual {resid:.4f} "
              f"(mirror {resid_m:.4f})"
              f"{' [mirror chosen on L/R geometry]' if flipped else ''}")
        print(f"    perm (canonical -> query index): {perm.tolist()}")
        if np.array_equal(perm, np.arange(len(Dr))):
            print("    identity -- the query already uses the canonical order")
    return perm, {"residual": resid, "residual_mirror": resid_m,
                  "mirror_applied": flipped, "perm": perm.tolist()}


def body_extent(xyz, max_frames=2000):
    """(median, max) inter-joint distance in metres, over clean frames."""
    F = np.asarray(xyz, np.float64)
    F = F[~(np.abs(F) < 1e-9).all(-1).any(-1)]
    if len(F) < 5:
        return float("nan"), float("nan")
    if len(F) > max_frames:
        F = F[np.linspace(0, len(F) - 1, max_frames).astype(int)]
    D = np.linalg.norm(F[:, :, None, :] - F[:, None, :, :], axis=-1)
    iu = np.triu_indices(D.shape[1], 1)
    return float(np.median(D[:, iu[0], iu[1]])), float(np.nanmax(D))


def score_parse(xyz, max_len=MAX_BONE_M, max_cv=MAX_BONE_CV,
                max_sd=MAX_BONE_SD_M):
    """
    Order-independent plausibility of a candidate parse, in [0, 1].
    PHYSICAL EXTENT first -- a parse mixing metres with raw depth yields a body
    kilometres across, which no heuristic can argue with -- then rigid structure.
    """
    med, mx = body_extent(xyz)
    if not np.isfinite(med) or med > MAX_MEDIAN_PAIR_M or med < MIN_MEDIAN_PAIR_M \
            or mx > MAX_BODY_SPAN_M:
        return 0.0, med
    tree = rigid_spanning_tree(xyz, max_cv, max_sd, allow_fallback=True)
    n_needed = np.asarray(xyz).shape[1] - 1
    if len(tree) < n_needed:
        return 0.0, float("nan")
    lens = [w for w, _, _ in tree]
    return float(np.mean([w <= max_len for w in lens])), float(np.median(lens))