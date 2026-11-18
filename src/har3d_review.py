"""
har3d_review.py -- analyses added in the revision.

Everything here answers one question the first version left open: how far do
the conclusions depend on (a) treating overlapping test sets as independent,
(b) the i.i.d. label draw of the null model, (c) the estimator and its bin
count, and (d) the power of the test at these sample sizes?

Top-label calibration error depends on a prediction only through its
confidence c_i and whether it is correct. Under the null model the label is
drawn from the predicted vector, so the prediction is correct with probability
c_i: for every TOP-LABEL statistic the full-vector draw of har3d_calib.ece_null
and a Bernoulli(c_i) draw of the correctness are the same null. All routines
below therefore work on (confidence, correct) and are vectorised over null
draws, which makes 20,000 draws per cell cheap.

  1. Estimators   equal-mass ECE with 5..30 bins, equal-width ECE, the
                  bias-corrected squared calibration error (Kumar et al. 2019)
                  and the binning-free Kolmogorov-Smirnov calibration error
                  (Gupta et al. 2021), each with its own matched null.
  2. Coupled null a null that preserves the dependence between cells that
                  score the same clips: one uniform variate per clip decides,
                  for every model that predicts it, whether the prediction is
                  correct (comonotone coupling), optionally with a random
                  effect per subject. Gives dependence-aware p-values for
                  statistics pooled over cells and test sets.
  3. Subject-level tests  human subjects are the units for paired comparisons
                  of per-clip proper scores.
  4. Subject heterogeneity  between-subject variance of the calibration
                  residual against its null (over-dispersion, tau).
  5. Power        probability of rejecting the null for a known miscalibration
                  (a shift on the logit scale), minimum detectable calibration
                  error, and the size of the test under within-subject
                  dependence.
  6. Per-class null  matched null for the pooled per-class ECE.
  7. Seed-level summaries for the ladder (seeds, not seed-rung pairs, as units).

Pure numpy/scipy. The estimators are checked against har3d_v2 in test_har3d.py.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy import stats

from har3d_stats import holm, sign_test

ESTIMATORS = ("aece5", "aece10", "aece15", "aece20", "aece30", "ece15", "ce2db15", "ks")
EST_LABEL = {"aece5": "adaptive ECE, 5 bins", "aece10": "adaptive ECE, 10 bins",
             "aece15": "adaptive ECE, 15 bins", "aece20": "adaptive ECE, 20 bins",
             "aece30": "adaptive ECE, 30 bins", "ece15": "equal-width ECE, 15 bins",
             "ce2db15": "debiased squared CE, 15 bins", "ks": "KS calibration error"}


# =====================================================================
# 1. Estimators on (confidence, correct), vectorised over draws
# =====================================================================
class Profile:
    """The confidences of one cell, sorted, with the bin layouts cached."""

    def __init__(self, conf):
        conf = np.asarray(conf, np.float64)
        self.n = len(conf)
        self.order = np.argsort(conf, kind="stable")
        self.c = conf[self.order]
        self._bins = {}

    def mass_bins(self, M):
        """Equal-mass bins exactly as har3d_v2._bins_equal_mass."""
        key = ("m", M)
        if key not in self._bins:
            edges = np.linspace(0, self.n, M + 1).astype(int)
            starts = edges[:-1][edges[:-1] < edges[1:]]
            sizes = np.diff(np.append(starts, self.n))
            self._bins[key] = (starts, sizes, np.add.reduceat(self.c, starts) / sizes)
        return self._bins[key]

    def width_bins(self, M):
        """Equal-width bins exactly as har3d_v2._bins_equal_width (empty bins dropped)."""
        key = ("w", M)
        if key not in self._bins:
            edges = np.linspace(0.0, 1.0, M + 1)
            idx = np.clip(np.digitize(self.c, edges[1:-1], right=True), 0, M - 1)
            starts = np.flatnonzero(np.r_[True, idx[1:] != idx[:-1]])
            sizes = np.diff(np.append(starts, self.n))
            self._bins[key] = (starts, sizes, np.add.reduceat(self.c, starts) / sizes)
        return self._bins[key]

    def sort(self, correct):
        """(n,) or (S, n) correctness in cell order -> float32 (S, n) in sorted order."""
        k = np.asarray(correct)
        k = k[None, :] if k.ndim == 1 else k
        return np.ascontiguousarray(k[:, self.order], dtype=np.float32)


def _binned(K, starts, sizes, cbar, n, kind):
    acc = np.add.reduceat(K, starts, axis=1) / sizes
    if kind == "l1":
        return (np.abs(acc - cbar) * (sizes / n)).sum(1)
    # bias-corrected squared calibration error (Kumar, Liang & Ma 2019, Eq. 5):
    # the plug-in squared gap of a bin over-estimates by acc(1-acc)/(n_b-1)
    corr = np.where(sizes > 1, acc * (1 - acc) / np.maximum(sizes - 1, 1), 0.0)
    return (((acc - cbar) ** 2 - corr) * (sizes / n)).sum(1)


def estimates(prof, K, names=ESTIMATORS):
    """Every estimator in `names` for each row of K (sorted correctness, (S, n))."""
    out = {}
    for nm in names:
        if nm.startswith("aece"):
            out[nm] = _binned(K, *prof.mass_bins(int(nm[4:])), prof.n, "l1")
        elif nm == "ece15":
            out[nm] = _binned(K, *prof.width_bins(15), prof.n, "l1")
        elif nm == "ce2db15":
            out[nm] = _binned(K, *prof.mass_bins(15), prof.n, "l2db")
        elif nm == "ks":
            # Kolmogorov-Smirnov calibration error (Gupta et al. 2021): largest
            # gap between cumulative confidence and cumulative accuracy over
            # predictions sorted by confidence; no bins
            out[nm] = np.abs(np.cumsum(prof.c[None, :] - K, axis=1, dtype=np.float64)).max(1) / prof.n
        else:
            raise KeyError(nm)
    return out


def null_correct(prof, S, rng, q=None, chunk=4_000_000):
    """S null draws of the correctness, sorted order: Bernoulli(c_i), or
    Bernoulli(q_i) when the success probabilities q (sorted order) are given."""
    q = (prof.c if q is None else np.asarray(q, np.float64)).astype(np.float32)
    out = np.empty((S, prof.n), np.float32)
    step = max(1, chunk // max(prof.n, 1))
    for a in range(0, S, step):
        b = min(S, a + step)
        out[a:b] = rng.random((b - a, prof.n), dtype=np.float32) < q
    return out


def mc_p(null_vals, obs):
    """Monte-Carlo p-value with the +1 correction (never zero)."""
    return float((1 + (np.asarray(null_vals) >= obs).sum()) / (len(null_vals) + 1))


def cell_tests(conf, correct, S=20000, seed=0, names=ESTIMATORS):
    """Observed value, matched-null mean and 95th percentile, excess and
    Monte-Carlo p-value of every estimator for one cell."""
    prof = Profile(conf)
    obs = estimates(prof, prof.sort(correct), names)
    null = estimates(prof, null_correct(prof, S, np.random.default_rng(seed)), names)
    return {nm: {"obs": float(obs[nm][0]), "null_mean": float(null[nm].mean()),
                 "null_p95": float(np.quantile(null[nm], 0.95)),
                 "null_sd": float(null[nm].std()),
                 "excess": float(obs[nm][0] - null[nm].mean()),
                 "p": mc_p(null[nm], obs[nm][0])} for nm in names}


# =====================================================================
# 2. Coupled null: cells that score the same clips are not independent
# =====================================================================
def coupled_null(cells, n_clips, S=5000, seed=0, names=("aece15",), tau=0.0,
                 subj_of_clip=None, shift=0.0):
    """
    Null draws for a group of cells that share test clips (one dataset).

    cells : list of dicts with 'conf' (n,), 'correct' (n,), 'clip' (n,) global
            clip index in [0, n_clips).
    One uniform variate per clip and draw decides the correctness of every
    prediction of that clip: u < c_i. Each cell's marginal null is unchanged
    (Bernoulli(c_i)), while a clip that comes out 'easy' in a draw is easy for
    every split, stream and seed that predicts it -- the strongest positive
    dependence compatible with the marginals, so pooled p-values built on it
    are conservative with respect to the overlap between cells.
    tau > 0 adds a subject random effect: the success probability of every
    clip of subject g is shifted by d_g ~ N(0, tau^2), redrawn per draw
    (subj_of_clip maps a clip to its subject index).
    shift != 0 draws from a MISCALIBRATED model instead (success probability
    shift_probs(c, shift)); used for the power of pooled statistics.
    Returns {name: {'obs': (ncell,), 'null': (ncell, S)}}.
    """
    rng = np.random.default_rng(seed)
    U = rng.random((S, n_clips), dtype=np.float32)
    D = None
    if tau > 0:
        if subj_of_clip is None:
            raise ValueError("tau > 0 needs subj_of_clip")
        D = rng.normal(0.0, tau, (S, int(np.max(subj_of_clip)) + 1)).astype(np.float32)
    out = {nm: {"obs": np.empty(len(cells)), "null": np.empty((len(cells), S))} for nm in names}
    for i, c in enumerate(cells):
        prof = Profile(c["conf"])
        clip = np.asarray(c["clip"])[prof.order]
        q = (prof.c if shift == 0 else shift_probs(prof.c, shift)).astype(np.float32)[None, :]
        if D is not None:
            q = np.clip(q + D[:, np.asarray(subj_of_clip)[clip]], 0.0, 1.0)
        K = (U[:, clip] < q).astype(np.float32)
        nul = estimates(prof, K, names)
        obs = estimates(prof, prof.sort(c["correct"]), names)
        for nm in names:
            out[nm]["obs"][i] = obs[nm][0]
            out[nm]["null"][i] = nul[nm]
    return out


def pooled_from_null(obs, null, units, datasets=None):
    """
    Pooled excess with a dependence-aware Monte-Carlo p-value.

    obs (ncell,), null (ncell, S) from coupled_null (cells of several datasets
    may be stacked: their draws are independent across datasets). The floor of
    a cell is its null mean; cells are averaged within their unit (test set),
    units are averaged to the grand mean. Returns the observed grand mean, its
    null s.d., the p-value P(null >= observed), the number of units with a
    positive mean excess and the null probability of at least that many, and
    the variance inflation relative to independent cells.
    """
    obs, null = np.asarray(obs, float), np.asarray(null, float)
    floor = null.mean(1)
    ex_obs, ex_null = obs - floor, null - floor[:, None]
    ukeys = list(dict.fromkeys(units))
    W = np.zeros((len(ukeys), len(obs)))
    for i, u in enumerate(units):
        W[ukeys.index(u), i] = 1.0
    W /= W.sum(1, keepdims=True)
    u_obs, u_null = W @ ex_obs, W @ ex_null
    g_obs, g_null = float(u_obs.mean()), u_null.mean(0)
    k_obs = int((u_obs > 0).sum())
    k_null = (u_null > 0).sum(0)
    # variance of the grand mean if the cells were independent
    w_cell = W.sum(0) / len(ukeys)
    var_indep = float((w_cell ** 2 * null.var(1)).sum())
    res = {"n_units": len(ukeys), "n_cells": len(obs), "mean_excess": g_obs,
           "null_sd": float(g_null.std()), "p": mc_p(g_null, g_obs),
           "k_pos": k_obs, "p_k": mc_p(k_null, k_obs),
           "z": float(g_obs / max(g_null.std(), 1e-12)),
           "var_inflation": float(g_null.var() / max(var_indep, 1e-18)),
           "unit_keys": ukeys, "unit_excess": u_obs}
    if datasets is not None:
        per = {}
        for ds in dict.fromkeys(datasets):
            m = np.array([d == ds for d in datasets])
            sub = pooled_from_null(obs[m], null[m], [u for u, k in zip(units, m) if k])
            per[ds] = {k: sub[k] for k in ("n_units", "n_cells", "mean_excess", "null_sd", "p", "k_pos", "z")}
        res["per_dataset"] = per
    return res


def _unit_weights(units):
    ukeys = list(dict.fromkeys(units))
    W = np.zeros((len(ukeys), len(units)))
    for i, u in enumerate(units):
        W[ukeys.index(u), i] = 1.0
    return W / W.sum(1, keepdims=True)


def pooled_power(groups, shifts, S=2000, alpha=0.05, seed=0, name="aece15"):
    """
    Power of the POOLED test (mean excess over test sets, coupled null) against
    a common miscalibration of every cell.

    groups : list of (cells, n_clips, units) -- one entry per dataset; cells as
             in coupled_null, units the test set of every cell.
    For each shift b the labels of every cell are drawn from the shifted model
    (same coupling between cells); the pooled statistic is compared with the
    1 - alpha quantile of its null distribution. Returns rows
    (b, eps = mean true calibration error over cells, power), b = 0 first.
    """
    def pooled(b, sd):
        tot, nu = 0.0, 0
        for gi, (cells, n_clips, units) in enumerate(groups):
            r = coupled_null(cells, n_clips, S=S, seed=sd + gi, names=(name,), shift=b)[name]
            W = _unit_weights(units)
            tot = tot + (W @ (r["null"] - floors[gi][:, None])).sum(0)
            nu += W.shape[0]
        return tot / nu

    # floors from an independent set of null draws
    floors = [coupled_null(c, n, S=S, seed=seed + 1000 + gi, names=(name,))[name]["null"].mean(1)
              for gi, (c, n, _) in enumerate(groups)]
    crit = np.quantile(pooled(0.0, seed + 2000), 1 - alpha)
    rows = []
    for j, b in enumerate((0.0,) + tuple(shifts)):
        eps = np.mean([np.abs(np.asarray(c["conf"], float) - shift_probs(c["conf"], b)).mean()
                       for cells, _, _ in groups for c in cells])
        rows.append({"b": float(b), "eps": float(eps),
                     "power": float((pooled(b, seed + 3000 + 10 * j) > crit).mean())})
    return rows


# =====================================================================
# 3. Subject-level paired tests on per-clip scores
# =====================================================================
class SubjectAccumulator:
    """Sums per-clip values by (dataset, subject) over every cell."""

    def __init__(self):
        self.s = defaultdict(float)
        self.n = defaultdict(int)

    def add(self, dataset, subjects, values):
        subjects, values = np.asarray(subjects), np.asarray(values, float)
        for g in np.unique(subjects):
            m = subjects == g
            self.s[(dataset, int(g))] += float(values[m].sum())
            self.n[(dataset, int(g))] += int(m.sum())

    def means(self):
        keys = sorted(self.s)
        return keys, np.array([self.s[k] / self.n[k] for k in keys])


def subject_test(keys, values, n_boot=20000, seed=0):
    """
    Units are human subjects. Exact sign test over subjects and a percentile
    bootstrap of the mean over subjects, resampled within each dataset.
    keys: list of (dataset, subject); values: per-subject mean difference
    (positive = the method is better).
    """
    values = np.asarray(values, float)
    k, n, p = sign_test(values)
    ds = [d for d, _ in keys]
    rng = np.random.default_rng(seed)
    boot = np.zeros(n_boot)
    for d in dict.fromkeys(ds):
        v = values[np.array([x == d for x in ds])]
        boot += v[rng.integers(0, len(v), (n_boot, len(v)))].sum(1)
    boot /= len(values)
    per = {d: (int((values[np.array([x == d for x in ds])] > 0).sum()), int(sum(x == d for x in ds)))
           for d in dict.fromkeys(ds)}
    return {"mean": float(values.mean()), "lo": float(np.quantile(boot, 0.025)),
            "hi": float(np.quantile(boot, 0.975)), "k": k, "n": n, "p": p, "per_dataset": per}


def subject_method_tests(acc_by_method, methods, n_boot=20000):
    """subject_test for every method and Holm over the methods (one family)."""
    res = {}
    for i, m in enumerate(methods):
        keys, vals = acc_by_method[m].means()
        res[m] = subject_test(keys, vals, n_boot=n_boot, seed=i)
    adj = holm([res[m]["p"] for m in methods])
    for m, a in zip(methods, adj):
        res[m]["p_holm"] = float(a)
        # The sign test speaks about the number of subjects, the mean about the size. When a
        # significant majority and the sign of the mean disagree, neither label is right.
        maj = 1 if 2 * res[m]["k"] > res[m]["n"] else -1
        res[m]["verdict"] = ("n.s." if a >= 0.05 else "mixed" if maj * res[m]["mean"] <= 0
                             else "reduces" if maj > 0 else "increases")
    return res


# =====================================================================
# 4. Subject heterogeneity of the calibration residual
# =====================================================================
def _msb(R, H, ng):
    """Between-subject mean square of the residuals R (S, n); H one-hot (n, G)."""
    gm = (R @ H) / ng
    grand = R.mean(1, keepdims=True)
    return ((gm - grand) ** 2 * ng).sum(1) / (len(ng) - 1)


def subject_heterogeneity(conf, correct, subjects, S=2000, seed=0):
    """
    Is the calibration residual (correct - confidence) more variable between
    subjects than independent labels allow? Returns the over-dispersion factor
    phi = MSB / E_null[MSB], its Monte-Carlo p-value, and the method-of-moments
    variance tau^2 of a subject effect on the success probability (one-way
    random-effects ANOVA; may be negative for a single cell, pool before use).
    """
    conf, correct = np.asarray(conf, np.float64), np.asarray(correct, np.float64)
    _, g = np.unique(subjects, return_inverse=True)
    G = int(g.max()) + 1
    if G < 2:
        return {"phi": float("nan"), "p": float("nan"), "tau2": float("nan"), "G": G}
    H = np.eye(G)[g]
    ng = H.sum(0)
    rng = np.random.default_rng(seed)
    obs = _msb((correct - conf)[None, :], H, ng)[0]
    Rn = (rng.random((S, len(conf))) < conf).astype(np.float64) - conf
    nul = _msb(Rn, H, ng)
    n0 = (len(conf) - (ng ** 2).sum() / len(conf)) / (G - 1)
    return {"phi": float(obs / nul.mean()), "p": mc_p(nul, obs),
            "tau2": float((obs - nul.mean()) / n0), "G": G}


# =====================================================================
# 5. Power of the null test
# =====================================================================
def shift_probs(conf, b):
    """True success probability when the model is miscalibrated by a shift b on
    the logit scale: b > 0 over-confident, b < 0 under-confident."""
    c = np.clip(np.asarray(conf, np.float64), 1e-6, 1 - 1e-6)
    return 1.0 / (1.0 + np.exp(-(np.log(c / (1 - c)) - b)))


B_GRID = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)


def power_curve(conf, bs=B_GRID, S_null=20000, R=300, alphas=(0.05,), seed=0, name="aece15"):
    """
    Power of the matched-null test for one confidence profile.

    For every shift b (and -b) the labels are drawn with success probability
    shift_probs(conf, b); the true top-label calibration error of that model
    is eps = mean |c_i - q_i|. Power is the fraction of R replicates whose
    Monte-Carlo p-value is <= alpha. Returns a list of dicts
    (b, eps, signed gap, power per alpha), b = 0 first (the size of the test).
    """
    prof = Profile(conf)
    rng = np.random.default_rng(seed)
    null = np.sort(estimates(prof, null_correct(prof, S_null, rng), (name,))[name])
    rows = []
    for b in (0.0,) + tuple(bs) + tuple(-x for x in bs):
        q = shift_probs(prof.c, b)
        obs = estimates(prof, null_correct(prof, R, rng, q=q), (name,))[name]
        p = (1 + S_null - np.searchsorted(null, obs, side="left")) / (S_null + 1)
        rows.append({"b": float(b), "eps": float(np.abs(prof.c - q).mean()),
                     "gap": float((prof.c - q).mean()),
                     **{f"power_{a:g}": float((p <= a).mean()) for a in alphas}})
    return rows, float(null.mean()), float(null.std())


def mde(rows, alpha=0.05, side=1, target=0.8):
    """Minimum detectable calibration error: the smallest true eps at which the
    power reaches `target`, by linear interpolation along the b grid for
    over-confidence (side=1) or under-confidence (side=-1). inf if never."""
    key = f"power_{alpha:g}"
    pts = sorted(((r["eps"], r[key]) for r in rows if r["b"] == 0 or np.sign(r["b"]) == side))
    for (e0, p0), (e1, p1) in zip(pts[:-1], pts[1:]):
        if p1 >= target > p0:
            return float(e0 + (target - p0) * (e1 - e0) / max(p1 - p0, 1e-12))
        if p0 >= target:
            return float(e0)
    return float("inf")


def synthetic_conf(n, K, mean_conf, seed=0, pool=20000):
    """A deterministic confidence profile of size n for a K-class softmax model
    whose mean confidence is `mean_conf`: logits ~ N(0, s^2 I) with s set by
    bisection, n evenly spaced quantiles of the resulting confidences."""
    z = np.random.default_rng(seed).normal(size=(pool, K))

    def conf(s):
        e = np.exp(s * (z - z.max(1, keepdims=True)))
        return (e / e.sum(1, keepdims=True)).max(1)
    lo, hi = 0.0, 50.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if conf(mid).mean() < mean_conf else (lo, mid)
    return np.quantile(conf(0.5 * (lo + hi)), (np.arange(n) + 0.5) / n)


def clustered_size(conf, subjects, taus, S_null=2000, R=2000, alpha=0.05, seed=0, name="aece15"):
    """
    Size of the test when labels are NOT independent given the confidences:
    every subject g has a random effect d_g ~ N(0, tau^2) on the success
    probability of all its clips (zero mean, so the model is calibrated on
    average over subjects). Rejection rate at `alpha` of (i) the i.i.d. null
    and (ii) a null that draws the same subject effects (tau known).
    """
    prof = Profile(conf)
    g = np.unique(np.asarray(subjects)[prof.order], return_inverse=True)[1]
    G = int(g.max()) + 1
    rng = np.random.default_rng(seed)

    def draws(tau, n_draw):
        d = rng.normal(0.0, tau, (n_draw, G)) if tau > 0 else np.zeros((n_draw, G))
        q = np.clip(prof.c[None, :] + d[:, g], 0.0, 1.0)
        return (rng.random((n_draw, prof.n)) < q).astype(np.float32)

    null_iid = np.sort(estimates(prof, draws(0.0, S_null), (name,))[name])
    out = []
    for tau in taus:
        obs = estimates(prof, draws(tau, R), (name,))[name]
        null_cl = np.sort(estimates(prof, draws(tau, S_null), (name,))[name])
        p_iid = (1 + S_null - np.searchsorted(null_iid, obs, side="left")) / (S_null + 1)
        p_cl = (1 + S_null - np.searchsorted(null_cl, obs, side="left")) / (S_null + 1)
        out.append({"tau": float(tau), "size_iid": float((p_iid <= alpha).mean()),
                    "size_cluster": float((p_cl <= alpha).mean()),
                    "floor_iid": float(null_iid.mean()), "floor_cluster": float(null_cl.mean())})
    return out


# =====================================================================
# 6. Per-class ECE with a matched null
# =====================================================================
def _pc_layout(conf, pred, K, M=15):
    """Per class: indices of its predictions sorted by confidence and the
    equal-mass bins of har3d_v2.per_class_ece."""
    lay = {}
    for c in range(K):
        idx = np.flatnonzero(pred == c)
        if len(idx) < 2:
            continue
        idx = idx[np.argsort(conf[idx], kind="stable")]
        M_ = min(M, max(2, int(len(idx) // 5)))
        edges = np.linspace(0, len(idx), M_ + 1).astype(int)
        starts = edges[:-1][edges[:-1] < edges[1:]]
        sizes = np.diff(np.append(starts, len(idx)))
        lay[c] = (idx, starts, sizes, np.add.reduceat(conf[idx], starts) / sizes)
    return lay


def per_class_ece_cc(conf, pred, correct, K, M=15, layout=None):
    """har3d_v2.per_class_ece from (confidence, predicted class, correct);
    `correct` may be (n,) or (S, n). Returns (S, K) with NaN where undefined."""
    conf = np.asarray(conf, np.float64)
    k = np.asarray(correct, np.float32)
    k = k[None, :] if k.ndim == 1 else k
    lay = layout or _pc_layout(conf, np.asarray(pred), K, M)
    out = np.full((k.shape[0], K), np.nan)
    for c, (idx, starts, sizes, cbar) in lay.items():
        acc = np.add.reduceat(k[:, idx], starts, axis=1) / sizes
        out[:, c] = (np.abs(acc - cbar) * (sizes / len(idx))).sum(1)
    return out


def per_class_null(conf, pred, correct, clip, n_clips, K, S=5000, seed=0, M=15):
    """
    Matched null of the pooled per-class ECE. Predictions of the same clip by
    different splits and seeds are coupled through one uniform per clip, as in
    coupled_null. Returns per class: n predictions, observed, null mean, null
    95th percentile, p-value.
    """
    conf, pred = np.asarray(conf, np.float64), np.asarray(pred)
    lay = _pc_layout(conf, pred, K, M)
    obs = per_class_ece_cc(conf, pred, correct, K, M, lay)[0]
    U = np.random.default_rng(seed).random((S, n_clips), dtype=np.float32)
    nul = per_class_ece_cc(conf, pred, U[:, np.asarray(clip)] < conf.astype(np.float32)[None, :], K, M, lay)
    rows = []
    for c in range(K):
        if c not in lay:
            continue
        rows.append({"class": c, "n_pred": int(len(lay[c][0])),
                     "n_clips": int(len(np.unique(np.asarray(clip)[lay[c][0]]))),
                     "obs": float(obs[c]), "null_mean": float(nul[:, c].mean()),
                     "null_p95": float(np.quantile(nul[:, c], 0.95)), "p": mc_p(nul[:, c], obs[c])})
    adj = holm([r["p"] for r in rows]) if rows else []
    for r, a in zip(rows, adj):
        r["p_holm"] = float(a)
    return rows


# =====================================================================
# 7. Ladder: seeds as the units
# =====================================================================
def seed_summary(values):
    """
    One value per independently seeded training run (e.g. the difference
    between two backbones averaged over the large rungs of that seed).
    Mean, t-interval and two-sided t-test with n-1 degrees of freedom, the
    exact sign-flip (permutation) p-value and the number of positive seeds.
    """
    v = np.asarray(values, float)
    n = len(v)
    out = {"n": n, "mean": float(v.mean()) if n else float("nan"), "n_pos": int((v > 0).sum())}
    if n >= 2:
        se = v.std(ddof=1) / np.sqrt(n)
        t = stats.t.ppf(0.975, n - 1)
        out.update(sd=float(v.std(ddof=1)), lo=float(v.mean() - t * se), hi=float(v.mean() + t * se),
                   p_t=float(2 * stats.t.sf(abs(v.mean()) / max(se, 1e-300), n - 1)))
        # exact sign-flip test: all 2^n sign assignments
        signs = np.array(np.meshgrid(*[[-1, 1]] * n)).reshape(n, -1).T
        perm = np.abs((signs * np.abs(v)).mean(1))
        out["p_flip"] = float((perm >= abs(v.mean()) - 1e-15).mean())
    else:
        out.update(sd=float("nan"), lo=float("nan"), hi=float("nan"), p_t=float("nan"), p_flip=float("nan"))
    return out
