"""
har3d_scale.py -- the NTU-60 TRAINING-SET LADDER. This is the scale axis.

WHY A LADDER AND NOT FOUR DATASETS
----------------------------------
Comparing calibration on small datasets (MSR, n ~ 500) with a large one
(NTU, n ~ 56,000) confounds two things: small TRAIN n (which may genuinely
change calibration) and small TEST n (which inflates the ECE estimator). The
two predict the same curve, so the scale claim is not identifiable. The ladder
holds architecture, recipe, preprocessing and the TEST SET fixed and varies
only the number of training samples, so the estimator floor is constant across
rungs and any movement in the excess ECE is attributable to the model.

CONVERGENCE CONTROL (revised after review)
------------------------------------------
Every rung is trained with EARLY STOPPING on validation NLL: up to `max_steps`
gradient steps, validation every `eval_every` steps, learning rate halved on a
plateau, and the best-validation-NLL weights restored. Each rung is therefore
measured at its own optimal convergence point. The earlier protocol used a
fixed 6,000-step budget for every rung, which gives small rungs hundreds of
epochs (over-fitting) and large rungs a dozen (under-convergence) -- a
confound between scale and convergence state. That protocol is kept as an
explicitly labelled ABLATION (early_stop=False) so the paper can show whether
the scale conclusion survives both.

Validation is a subject-disjoint slice of the X-Sub TRAIN subjects, identical
for every backbone and every rung.
"""
from __future__ import annotations

import time

import numpy as np

# NTU 25-joint kinematic tree, 0-indexed, root at 20 (spine-shoulder).
NTU_PARENTS = [1, 20, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10, 0, 12, 13, 14,
               0, 16, 17, 18, -1, 22, 7, 24, 11]
NTU_JOINTS = 25
NTU_PERSONS = 2

LADDER_DEFAULT = [500, 1000, 2000, 5000, 10000, 20000, 40091]


# =====================================================================
# 1. Tensors
# =====================================================================
def load_ntu_npz(path):
    """
    CTR-GCN's NTU60_CS.npz -> raw arrays. Layout is (N, 300, 150), NOT the
    (N, 3, 300, 25, 2) usually quoted; 150 = 2 persons x 25 joints x 3 coords.
    """
    z = np.load(path, allow_pickle=False)
    return (z["x_train"], z["y_train"].argmax(-1) if z["y_train"].ndim > 1
            else z["y_train"],
            z["x_test"], z["y_test"].argmax(-1) if z["y_test"].ndim > 1
            else z["y_test"])


def reshape_ntu(x):
    """(N, 300, 150) -> (N, 3, 300, 25, 2), matching CTR-GCN's own feeder."""
    N, T, D = x.shape
    assert D == NTU_PERSONS * NTU_JOINTS * 3, f"unexpected trailing dim {D}"
    return x.reshape(N, T, NTU_PERSONS, NTU_JOINTS, 3).transpose(0, 4, 1, 3, 2)


def valid_length(seq):
    """Frames before the zero padding. seq is (3, T, J, M)."""
    nz = np.abs(seq).sum((0, 2, 3)) > 1e-9
    idx = np.where(nz)[0]
    return (int(idx[0]), int(idx[-1]) + 1) if len(idx) else (0, 1)


def valid_resize(seq, T=64):
    """Crop the zero padding, then uniformly resample to T frames."""
    a, b = valid_length(seq)
    s = seq[:, a:b]
    n = s.shape[1]
    if n == T:
        return s
    idx = np.clip(np.round(np.linspace(0, n - 1, T)).astype(int), 0, max(n - 1, 0))
    return s[:, idx]


def to_stream(x, stream, parents=NTU_PARENTS):
    """
    (C, T, J, M) -> stream tensor of the same shape.
    Applied AFTER valid_resize so all four streams share one convention.
    """
    x = np.asarray(x, np.float32)
    if stream.startswith("bone"):
        out = np.zeros_like(x)
        for j, p in enumerate(parents):
            out[:, :, j] = x[:, :, j] - x[:, :, p if p >= 0 else j]
        x = out
    elif not stream.startswith("joint"):
        raise ValueError(f"unknown stream {stream!r}")
    if stream.endswith("motion"):
        m = np.zeros_like(x)
        m[:, 1:] = x[:, 1:] - x[:, :-1]
        x = m
    return x


def prepare_ntu_tensors(x_raw, stream="joint", T=64, parents=NTU_PARENTS,
                        batch=2000, verbose=True):
    """
    (N, 300, 150) -> (N, 3, T, 25, 2) float32 for one stream.

    Processed in batches because the full x_train is 7.2 GB as float32 and
    materialising a second copy OOMs a 13 GB runtime.
    """
    N = len(x_raw)
    out = np.zeros((N, 3, T, NTU_JOINTS, NTU_PERSONS), np.float32)
    for b0 in range(0, N, batch):
        b1 = min(b0 + batch, N)
        blk = reshape_ntu(np.asarray(x_raw[b0:b1], np.float32))
        for i in range(b1 - b0):
            out[b0 + i] = to_stream(valid_resize(blk[i], T), stream, parents)
        if verbose and (b0 // batch) % 5 == 0:
            print(f"      {b1}/{N} sequences", flush=True)
        del blk
    return out


def subsample_indices(y, n, seed=0):
    """
    Stratified subsample of size n. Every class keeps its share, so a small
    rung is not accidentally a class-imbalance experiment as well.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    K = int(y.max()) + 1
    per = max(1, n // K)
    picked = []
    for c in range(K):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        take = min(per, len(idx))
        picked.append(rng.choice(idx, take, replace=False))
    picked = np.concatenate(picked)
    if len(picked) < n:                        # top up from what is left
        rest = np.setdiff1d(np.arange(len(y)), picked)
        extra = rng.choice(rest, min(n - len(picked), len(rest)), replace=False)
        picked = np.concatenate([picked, extra])
    rng.shuffle(picked)
    return np.sort(picked[:n])


# =====================================================================
# 2. Models -- see har3d_backbones (one shared trunk, three graph layers)
# =====================================================================
def make_model(num_classes, parents=NTU_PARENTS, in_ch=3, base=64,
               dropout=0.3, arch="stgcn"):
    import har3d_backbones as HB
    return HB.build(arch, num_classes, parents, in_ch=in_ch, base=base,
                    dropout=dropout)


# =====================================================================
# 3. One rung
# =====================================================================
def train_rung(Xtr, ytr, Xval, yval, Xte, yte, num_classes, arch="stgcn",
               early_stop=True, max_steps=15000, eval_every=250, patience=8,
               min_steps=500, fixed_steps=6000, bs=64, lr=1e-3, seed=0,
               device=None, parents=NTU_PARENTS, dropout=0.3, loss="ce",
               freeze_graph_after=None, verbose=True, log_every=1000,
               snapshot_steps=None, track_best_acc=False, run_until=0):
    """
    Train one ladder rung and return logits.

    early_stop=True  (main protocol) -- up to `max_steps` steps; validation NLL
                     every `eval_every` steps; LR halved after 3 evaluations
                     without improvement; stop after `patience` evaluations
                     without improvement (never before `min_steps`); restore
                     the best-validation-NLL weights.
    early_stop=False (ablation)      -- exactly `fixed_steps` steps with a
                     cosine schedule, the protocol used before the review.
    freeze_graph_after=S             -- after step S, stop updating the
                     parameters that generate the adaptive graph (the
                     adjacency-freezing ablation for 2s-AGCN).

    snapshot_steps=(s1, s2, ...)     -- also keep the TEST logits of the weights
                     at these steps (multiples of `eval_every`), so backbones
                     can be compared at matched training durations.
    track_best_acc=True              -- also keep the TEST logits of the weights
                     with the best validation ACCURACY (an alternative
                     stopping rule), over every evaluation that was made.
    run_until=S                      -- when early stopping fires before step S,
                     keep training to step S so that the snapshots up to S
                     exist for every backbone. The early-stopping decision
                     and the best-validation-NLL weights are frozen at the
                     moment the rule fires, so nothing in the main output
                     changes; info['steps_run'] stays the step at which the
                     rule fired and info['steps_total'] is the last step.
                     None of the three options touches the optimiser, the
                     scheduler or any random stream before the stopping
                     point: the main output is identical with and without
                     them (tested).

    Returns {'val': (logits, y), 'test': (logits, y), 'info': {...}} and, when
    either option is set, 'extra': {tag: {'logits', 'step', 'val_nll',
    'val_top1'}} with tag 's<step>' or 'acc'.
    Minibatches are drawn with replacement from a seeded generator, so a run
    is deterministic given `seed`.
    """
    import torch
    import torch.nn.functional as F

    import har3d_backbones as HB
    from har3d_models_v2 import _loss_fn

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = make_model(num_classes, parents, in_ch=Xtr.shape[1],
                       dropout=dropout, arch=arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total = max_steps if early_stop else fixed_steps
    if early_stop:
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min",
                                                         factor=0.5, patience=3)
    else:
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, fixed_steps)
    crit = _loss_fn(loss)

    Xt, yt = torch.as_tensor(Xtr), torch.as_tensor(ytr)
    n = len(Xt)
    rng = np.random.default_rng(seed)

    @torch.no_grad()
    def fwd(X, bs_eval=256):
        model.eval()
        return np.concatenate(
            [model(torch.as_tensor(X[b:b + bs_eval]).to(device)).cpu().numpy()
             for b in range(0, len(X), bs_eval)], 0).astype(np.float32)

    def val_stats():
        zv = fwd(Xval)
        z = torch.as_tensor(zv, dtype=torch.float64)
        return (float(F.cross_entropy(z, torch.as_tensor(yval)).item()),
                float((zv.argmax(-1) == np.asarray(yval)).mean()))

    def val_nll():
        return val_stats()[0]

    snapshot_steps = set(int(x) for x in (snapshot_steps or ()))
    extra = {}
    best_acc, best_acc_state, best_acc_step, best_acc_nll = -1.0, None, 0, np.nan
    best, best_state, best_step, since = np.inf, None, 0, 0
    frozen_at, n_frozen, curve = None, 0, []
    stop_step = None                  # step at which the early-stopping rule fired
    t0 = time.time()
    model.train()
    run, seen, corr = 0.0, 0, 0
    step = 0
    for step in range(1, total + 1):
        if freeze_graph_after is not None and step == freeze_graph_after + 1:
            n_frozen = HB.freeze_graph(model)
            frozen_at = freeze_graph_after
            if verbose:
                print(f"      graph frozen after step {frozen_at} "
                      f"({n_frozen} parameter tensors)", flush=True)
        model.train()
        i = torch.as_tensor(rng.integers(0, n, min(bs, n)))
        xb, yb = Xt[i].to(device), yt[i].to(device)
        opt.zero_grad(set_to_none=True)
        out = model(xb)
        l = crit(out, yb)
        l.backward()
        opt.step()
        if not early_stop:
            sch.step()
        run += float(l.detach()) * len(i)
        corr += int((out.argmax(-1) == yb).sum())
        seen += len(i)
        if verbose and step % log_every == 0:
            print(f"      step {step:6d}  loss {run/seen:.4f}  "
                  f"batch acc {corr/seen:.4f}", flush=True)
            run, seen, corr = 0.0, 0, 0
        if early_stop and step % eval_every == 0:
            v, a = val_stats()
            curve.append((step, v))
            sch.step(v)
            if track_best_acc and a > best_acc + 1e-9:
                best_acc, best_acc_step, best_acc_nll = a, step, v
                best_acc_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            if step in snapshot_steps:
                extra[f"s{step}"] = {"logits": fwd(Xte), "step": int(step),
                                     "val_nll": float(v), "val_top1": float(a)}
            if stop_step is None:
                if v < best - 1e-6:
                    best, best_state, best_step, since = (
                        v, {k: t.detach().clone() for k, t in model.state_dict().items()},
                        step, 0)
                else:
                    since += 1
                if since >= patience and step >= min_steps:
                    stop_step = step
            if stop_step is not None and step >= run_until:
                break

    if early_stop and best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_step, best = step, val_nll()

    out = {"val": (fwd(Xval), np.asarray(yval)),
           "test": (fwd(Xte), np.asarray(yte))}
    out["info"] = {"arch": arch, "early_stop": bool(early_stop),
                   "steps_run": int(stop_step if stop_step is not None else step),
                   "steps_total": int(step), "best_step": int(best_step),
                   "best_val_nll": float(best), "frozen_at": frozen_at,
                   "n_frozen": int(n_frozen), "epochs_at_best":
                   float(best_step * min(bs, n) / n), "seconds": time.time() - t0,
                   "val_curve": curve}
    if track_best_acc and best_acc_state is not None:
        model.load_state_dict(best_acc_state)
        extra["acc"] = {"logits": fwd(Xte), "step": int(best_acc_step),
                        "val_nll": float(best_acc_nll), "val_top1": float(best_acc)}
        out["info"]["best_acc_step"] = int(best_acc_step)
    if extra:
        out["extra"] = extra
    if verbose:
        te = out["test"]
        print(f"      [{arch}] best step {best_step}/{out['info']['steps_run']} "
              f"(~{out['info']['epochs_at_best']:.1f} epochs)  val NLL {best:.4f}  "
              f"test top1 {(te[0].argmax(-1) == te[1]).mean():.4f}", flush=True)
    return out


# =====================================================================
# 4. The ladder protocol (shared by Stage 8 and Stage 8b)
# =====================================================================
def ladder_protocol(x_tr_raw, y_tr, perf_file, n_val_subjects=4, seed=0,
                    record=None):
    """
    Subject-disjoint validation carved from the X-Sub TRAIN performers, and the
    training pool that every rung of every backbone subsamples from.

    There is NO fallback: if the performer ids cannot be recovered, this
    raises. (The earlier pipeline fell back to a random 10% slice in a fresh
    runtime, which gave the 2s-AGCN ladder a different pool and a validation
    set that shared subjects with training.) If `record` (a JSON path) exists,
    the recomputed protocol must match it exactly, so every backbone and every
    session uses the same pool and validation set.
    """
    import hashlib
    import json
    from pathlib import Path
    from har3d_ntu import train_performers

    perf_tr = train_performers(perf_file)
    if len(perf_tr) != len(y_tr):
        raise RuntimeError(f"{len(perf_tr)} recovered performers vs {len(y_tr)} "
                           f"training rows -- the performer file does not "
                           f"match this NTU60_CS.npz")
    uniq = np.unique(perf_tr)
    rng = np.random.default_rng(seed)
    val_subj = sorted(int(s) for s in rng.permutation(uniq)[:n_val_subjects])
    val_mask = np.isin(perf_tr, val_subj)
    val_idx, pool_idx = np.where(val_mask)[0], np.where(~val_mask)[0]
    assert not set(perf_tr[val_idx]) & set(perf_tr[pool_idx])
    proto = {"val_subjects": val_subj, "n_val": int(len(val_idx)),
             "n_pool": int(len(pool_idx)),
             "pool_sha1": hashlib.sha1(pool_idx.tobytes()).hexdigest()[:16],
             "val_sha1": hashlib.sha1(val_idx.tobytes()).hexdigest()[:16]}
    if record is not None:
        record = Path(record)
        if record.exists():
            old = json.loads(record.read_text())
            for k in proto:
                if old.get(k) != proto[k]:
                    raise RuntimeError(f"ladder protocol changed ({k}: {old.get(k)} "
                                       f"-> {proto[k]}); refusing to mix pools")
        else:
            record.write_text(json.dumps(proto, indent=2))
    return val_idx, pool_idx, perf_tr, proto


def run_ladder(store, x_tr_raw, y_tr, pool_idx, Xval, yval, Xte, yte, *,
               model_tag, arch, rungs, seeds, stream="joint", T=64,
               early_stop=True, fixed_steps=6000, max_steps=15000,
               eval_every=250, patience=8, min_steps=500,
               freeze_graph_after=None, protocol=None, device=None,
               verbose=True, snapshot_steps=None, track_best_acc=False,
               run_until=0, extras_dir=None, extra_provenance=None):
    """Train every (rung, seed) for one backbone; resumable via the store.
    Rung subsamples depend only on (n, seed), so all backbones see the same
    training samples at a given rung and seed. With `extras_dir`, the extra
    checkpoints of train_rung (snapshots, best validation accuracy) are written
    there as ck__ntu60__<tag>__<stream>__<split>__s<seed>__<ckpt>.npz BEFORE the
    main cell is stored, so an interrupted run is simply repeated."""
    from store import Key
    K = int(y_tr.max()) + 1
    rows = []
    for n_train in rungs:
        n_train = int(min(n_train, len(pool_idx)))
        split = f"ladder_n{n_train}"
        for seed in seeds:
            if store.has(Key(model_tag, "ntu60", stream, split, seed, "test")):
                if verbose:
                    print(f"  [skip] {model_tag} n={n_train} seed{seed}")
                continue
            sel = pool_idx[subsample_indices(y_tr[pool_idx], n_train, seed=seed)]
            if verbose:
                print(f"\n  === {model_tag} ({arch}) n_train={n_train} seed={seed} ===",
                      flush=True)
            Xtr = prepare_ntu_tensors(x_tr_raw[sel], stream, T=T, verbose=False)
            out = train_rung(Xtr, y_tr[sel], Xval, yval, Xte, yte, K, arch=arch,
                             early_stop=early_stop, fixed_steps=fixed_steps,
                             max_steps=max_steps, eval_every=eval_every,
                             patience=patience, min_steps=min_steps, seed=seed,
                             freeze_graph_after=freeze_graph_after,
                             device=device, verbose=verbose,
                             snapshot_steps=snapshot_steps,
                             track_best_acc=track_best_acc, run_until=run_until)
            prov = dict(arch=arch, method=model_tag, stream=stream,
                        n_train=n_train, T=T, train_info=out["info"],
                        protocol=protocol or {},
                        val="subject-disjoint slice of X-Sub train performers",
                        note="test set fixed across every rung and backbone")
            prov.update(extra_provenance or {})
            if extras_dir is not None:
                from pathlib import Path
                for tag, e in out.get("extra", {}).items():
                    np.savez_compressed(
                        Path(extras_dir) / f"ck__ntu60__{model_tag}__{stream}__{split}__s{seed}__{tag}.npz",
                        logits=e["logits"].astype(np.float32), labels=np.asarray(yte),
                        step=np.int64(e["step"]), val_nll=np.float64(e["val_nll"]),
                        val_top1=np.float64(e["val_top1"]))
            for sub in ("val", "test"):
                L, yy = out[sub]
                store.save(Key(model_tag, "ntu60", stream, split, seed, sub),
                           L, yy, provenance=prov)
            rows.append(dict(model=model_tag, n_train=n_train, seed=seed,
                             best_step=out["info"]["best_step"],
                             steps_run=out["info"]["steps_run"],
                             seconds=float(out["info"]["seconds"]),
                             top1=float((out["test"][0].argmax(-1) == yte).mean())))
            del Xtr
    return rows