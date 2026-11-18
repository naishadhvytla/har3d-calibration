"""
har3d_stats.py -- inference at the level of coarse units, not cells.

Cells that share a test set are not independent evidence. On the small
datasets the 20 cells of a split (4 streams x 5 seeds) are scored on the same
test clips, so a test that treats the 240 cells as 240 observations is
anti-conservative. On the NTU ladder every run is scored on one fixed test set,
and a bootstrap over its clips holds the trained models fixed, so it cannot see
the variation between training runs. Every inferential statement in the paper
is therefore made over coarser units:

  small datasets   test sets (dataset x split), 12 units. They are not fully
                   independent either: the four test sets of a dataset share
                   one or two subjects pairwise, so p-values over them are
                   approximate (optimistic) and are reported together with
                   the per-dataset pattern.
  NTU ladder       seed-rung pairs (separately trained runs on one fixed
                   test set), which carry the run-to-run variance

Pure numpy/scipy, no project imports, so it is shared verbatim by the notebook
(Stages 9b and 9c) and by make_numbers.py: one definition of every test.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy import stats

N_BOOT_UNITS = 20000          # cluster-bootstrap resamples over test sets


# =====================================================================
# 1. Elementary tests
# =====================================================================
def holm(pvals):
    """Holm-Bonferroni adjusted p-values (order-preserving, FWER control)."""
    pvals = np.asarray(pvals, float)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, idx in enumerate(np.argsort(pvals, kind="stable")):
        run = max(run, (m - rank) * pvals[idx])
        adj[idx] = min(run, 1.0)
    return adj


def sign_test(deltas):
    """Two-sided exact sign test that positive and negative deltas are equally
    likely (exact zeros dropped). Returns (n_positive, n_nonzero, p)."""
    d = np.asarray(deltas, float)
    d = d[d != 0]
    k = int((d > 0).sum())
    return k, len(d), (float(stats.binomtest(k, len(d), 0.5).pvalue) if len(d) else 1.0)


def cluster_boot_ci(unit_values, n_boot=N_BOOT_UNITS, seed=0, alpha=0.05):
    """Percentile bootstrap interval of the mean of per-unit values. Resampling
    whole units (test sets) keeps every cell of a unit together."""
    d = np.asarray(unit_values, float)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def unit_means(values, units):
    """Mean of `values` within each unit, in order of first appearance."""
    g = defaultdict(list)
    for v, u in zip(values, units):
        g[u].append(v)
    keys = list(g)
    return keys, np.array([np.mean(g[k]) for k in keys])


def ols_loglog(x, y, level=0.95):
    """OLS fit of log y on log x. The t-interval of the slope uses the scatter
    of the points about the line (lack of fit included), unlike a bootstrap
    over seeds within fixed rungs. Returns slope, lo, hi, R^2."""
    lr = stats.linregress(np.log(np.asarray(x, float)), np.log(np.asarray(y, float)))
    t = stats.t.ppf(0.5 + level / 2, len(x) - 2)
    return float(lr.slope), float(lr.slope - t * lr.stderr), float(lr.slope + t * lr.stderr), float(lr.rvalue ** 2)


# =====================================================================
# 2. Small datasets: test sets are the units
# =====================================================================
def pooled_excess(rows):
    """RQ1 in aggregate: excess over the null averaged within each test set
    (dataset x split), then an exact sign test over test sets."""
    keys, u = unit_means([r["null_excess"] for r in rows], [(r["dataset"], r["split"]) for r in rows])
    k, n, p = sign_test(u)
    per_ds = defaultdict(lambda: [0, 0])
    for key, v in zip(keys, u):
        per_ds[key[0]][0] += int(v > 0)
        per_ds[key[0]][1] += 1
    return {"units": keys, "unit_excess": u, "k_pos": k, "n": n, "p": p, "mean": float(u.mean()),
            "per_dataset": {ds: tuple(c) for ds, c in per_ds.items()}}


def method_comparison(per_cell, methods, baseline="raw", n_boot=N_BOOT_UNITS, key="ece"):
    """
    Paired comparison of calibration methods against the baseline at the level
    of test sets. per_cell: dicts with method, dataset, stream, split, seed and
    the score `key` (float; lower is better: 'ece', 'excess', 'brier', 'nll').
    A cell with seed == -1 (the deep ensemble, one per dataset, stream and
    split) is paired with the seed-mean of the baseline.

    Per method: the paired reduction (baseline - method, positive is better) is
    averaged over the cells of each test set; the sign test runs over test
    sets, Holm-corrected over the methods; the interval is a cluster bootstrap
    over test sets. Also returns a cluster-bootstrap interval of each method's
    mean ECE (baseline included).
    """
    base = {(r["dataset"], r["stream"], r["split"], r["seed"]): r[key]
            for r in per_cell if r["method"] == baseline}
    base_mean = defaultdict(list)
    for (ds, st, sp, _), v in base.items():
        base_mean[(ds, st, sp)].append(v)
    res = {}
    for m in methods:
        vals, units = [], []
        for r in per_cell:
            if r["method"] != m:
                continue
            b = (np.mean(base_mean[(r["dataset"], r["stream"], r["split"])]) if int(r["seed"]) == -1
                 else base[(r["dataset"], r["stream"], r["split"], r["seed"])])
            vals.append(b - r[key])
            units.append((r["dataset"], r["split"]))
        keys, d = unit_means(vals, units)
        k, n, p = sign_test(d)
        lo, hi = cluster_boot_ci(d, n_boot=n_boot, seed=0)
        per_ds = defaultdict(lambda: [0, 0])
        for u, v in zip(keys, d):
            per_ds[u[0]][0] += int(v > 0)
            per_ds[u[0]][1] += 1
        res[m] = dict(n=n, k=k, mean=float(d.mean()), lo=lo, hi=hi, p=p,
                      per_dataset={ds: tuple(c) for ds, c in per_ds.items()})
    for m, a in zip(methods, holm([res[m]["p"] for m in methods])):
        res[m]["p_holm"] = float(a)
        res[m]["verdict"] = (("reduces" if res[m]["mean"] > 0 else "increases") if a < 0.05 else "n.s.")
    ece_ci = {}
    for m in [baseline] + list(methods):
        _, d = unit_means([r[key] for r in per_cell if r["method"] == m],
                          [(r["dataset"], r["split"]) for r in per_cell if r["method"] == m])
        ece_ci[m] = cluster_boot_ci(d, n_boot=n_boot, seed=1)
    return res, ece_ci


# =====================================================================
# 3. Ladder: separately trained runs are the units
# =====================================================================
def seed_level_tests(ece, comparisons, regimes, seeds=(0, 1, 2)):
    """
    Exact sign tests over seed-rung pairs. ece: {(model, n_train, seed): aECE}.
    comparisons: [(model_a, model_b)]; regimes: {name: [n_train, ...]}.
    One Holm family over every (comparison, regime) test. Each pair is a
    separate training run of both models on the same subsample, so the
    run-to-run variance is part of the test, unlike a bootstrap over clips.
    """
    out = []
    for a, b in comparisons:
        for name, ns in regimes.items():
            d = [ece[(a, n, s)] - ece[(b, n, s)] for n in ns for s in seeds
                 if (a, n, s) in ece and (b, n, s) in ece]
            k, nz, p = sign_test(d)
            out.append(dict(model_a=a, model_b=b, regime=name, n_pairs=len(d), n_a_higher=k,
                            n_nonzero=nz, mean_delta=float(np.mean(d)), p=p))
    for r, a in zip(out, holm([r["p"] for r in out])):
        r["p_holm"] = float(a)
    return out


def pooled_seed_sd(ece, models, rungs, seeds=(0, 1, 2)):
    """Run-to-run spread: root-mean-square of the across-seed standard
    deviation (ddof=1) over the given models and rungs."""
    sds = [np.std([ece[(m, n, s)] for s in seeds], ddof=1)
           for m in models for n in rungs if all((m, n, s) in ece for s in seeds)]
    return float(np.sqrt(np.mean(np.square(sds)))), len(sds)


def identical_run_gap(rows, a="agcn", b="agcn_frozen", n=2000, seed=0, freeze_step=1000):
    """Run-to-run noise: two runs that restore the same weights from a step at
    or before the freeze are nominally identical models. Returns the gaps in
    aECE and top-1, or None if the condition does not hold."""
    L = {(r["model"], r["n_train"], r["seed"]): r for r in rows if r.get("dataset") == "ntu60"}
    ra, rb = L.get((a, n, seed)), L.get((b, n, seed))
    if ra is None or rb is None or not (ra["best_step"] == rb["best_step"] <= freeze_step):
        return None
    return dict(ece=abs(ra["ece_adaptive"] - rb["ece_adaptive"]), top1=abs(ra["top1"] - rb["top1"]),
                step=int(ra["best_step"]))