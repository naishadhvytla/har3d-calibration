"""
har3d_calib.py -- calibration methods and the NULL MODEL.

THE NULL MODEL IS THE POINT OF THIS FILE. A binned calibration error is a
BIASED estimator: a perfectly calibrated model does not report ECE 0, it
reports a positive floor that grows as n shrinks and K grows. On the MSR
datasets that floor is roughly the size of every effect in the paper, so an
ECE quoted without it is uninterpretable.

ece_null() simulates the floor using THE MODEL'S OWN confidence profile rather
than synthetic logits: draw y_sim ~ Categorical(p_i) for each sample. By
construction that simulated dataset is perfectly calibrated AND has exactly the
observed confidence distribution, so any ECE it reports is pure estimator bias.
The excess over the null is the only part that is evidence of miscalibration.

Every metric is imported from har3d_v2 so there is exactly one definition.
"""
from __future__ import annotations

import numpy as np

from har3d_v2 import (adaptive_ece, brier, ece, fit_temperature, nll, softmax,
                      top1)


# =====================================================================
# 1. The null model
# =====================================================================
def ece_null(p, n_sim=500, metric=adaptive_ece, seed=0, M=15):
    """
    Null distribution of `metric` for a PERFECTLY calibrated model with this
    model's confidence profile.

    p : (n, K) predicted probabilities from the real model.
    Returns (values (n_sim,), mean, p95).
    """
    p = np.asarray(p, np.float64)
    p = p / p.sum(-1, keepdims=True)
    rng = np.random.default_rng(seed)
    n, K = p.shape
    cdf = np.cumsum(p, axis=1)
    vals = np.empty(n_sim)
    for i in range(n_sim):
        u = rng.random((n, 1))
        y_sim = (u > cdf).sum(1)              # inverse-CDF categorical draw
        y_sim = np.clip(y_sim, 0, K - 1)
        vals[i] = metric(p, y_sim, M) if metric is not brier else metric(p, y_sim)
    return vals, float(vals.mean()), float(np.quantile(vals, 0.95))


def ece_significance(p, y, n_sim=500, metric=adaptive_ece, seed=0, M=15):
    """
    Observed calibration error against its own null.

    Returns observed, null mean/p95, EXCESS (observed - null mean) and a
    one-sided p-value = P(null >= observed). A p-value above 0.05 means the
    model is statistically indistinguishable from perfectly calibrated at this
    sample size -- which is the honest reading for most MSR cells.
    """
    p = np.asarray(p, np.float64)
    obs = metric(p, y, M) if metric is not brier else metric(p, y)
    vals, mu, p95 = ece_null(p, n_sim=n_sim, metric=metric, seed=seed, M=M)
    # Monte-Carlo p-value with the +1 correction (Phipson & Smyth 2010): it is
    # never 0, so a multiple-comparison correction stays valid. Its smallest
    # value is 1/(n_sim+1), so n_sim must be large enough for the family size.
    pval = float((1 + (vals >= obs).sum()) / (n_sim + 1))
    return {"observed": float(obs), "null_mean": mu, "null_p95": p95,
            "excess": float(obs - mu), "p_value": pval,
            "separable": bool(pval < 0.05), "n": int(len(y)),
            "n_classes": int(p.shape[1])}


def paired_bootstrap(p_a, p_b, y, metric=adaptive_ece, n_boot=2000, seed=0, M=15):
    """
    Is method A better calibrated than method B ON THE SAME SAMPLES? Resampling
    the pair together removes the sample-to-sample variance that dominates an
    unpaired comparison at n = 200.
    """
    p_a, p_b, y = np.asarray(p_a), np.asarray(p_b), np.asarray(y)
    rng = np.random.default_rng(seed)
    n = len(y)
    call = (lambda p, yy: metric(p, yy, M)) if metric is not brier else metric
    obs = call(p_a, y) - call(p_b, y)
    d = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.integers(0, n, n)
        d[i] = call(p_a[s], y[s]) - call(p_b[s], y[s])
    d = d[np.isfinite(d)]
    return {"delta": float(obs), "lo": float(np.quantile(d, 0.025)),
            "hi": float(np.quantile(d, 0.975)),
            "p_value": float(min(1.0, 2 * (1 + min((d >= 0).sum(), (d <= 0).sum())) / (len(d) + 1))),
            "significant": bool(np.quantile(d, 0.025) > 0 or np.quantile(d, 0.975) < 0)}


# =====================================================================
# 2. Post-hoc calibration
# =====================================================================
def fit_vector_scaling(logits_val, y_val, l2=1e-3, iters=300, lr=0.1, seed=0):
    """
    Per-class scale and bias, z' = w * z + b, fitted on validation NLL by plain
    gradient descent with analytic gradients and an L2 pull toward the identity
    (w = 1, b = 0). The regulariser matters: with K = 20 and 60 validation
    samples, unregularised vector scaling memorises the validation set.
    """
    z = np.asarray(logits_val, np.float64)
    y = np.asarray(y_val)
    n, K = z.shape
    w, b = np.ones(K), np.zeros(K)
    oh = np.zeros((n, K))
    oh[np.arange(n), y] = 1.0
    for _ in range(iters):
        p = softmax(z * w + b)
        g = (p - oh) / n                       # dL/d(logit)
        gw = (g * z).sum(0) + 2 * l2 * (w - 1.0)
        gb = g.sum(0) + 2 * l2 * b
        w -= lr * gw
        b -= lr * gb
    return w, b


def apply_vector_scaling(logits, w, b):
    return softmax(np.asarray(logits, np.float64) * w + b)


def sharpen_probs(p, T=1.0):
    """Temperature applied to PROBABILITIES (for ensembles, where the mean is
    taken in probability space and no logits survive)."""
    p = np.asarray(p, np.float64)
    q = np.power(np.clip(p, 1e-12, 1.0), 1.0 / T)
    return q / q.sum(-1, keepdims=True)


def fit_temperature_probs(p_val, y_val, lo=0.05, hi=10.0, iters=200):
    """Golden-section on NLL, for probability-space inputs."""
    p, y = np.asarray(p_val, np.float64), np.asarray(y_val)
    idx = np.arange(len(y))

    def loss(T):
        q = sharpen_probs(p, T)
        return float(-np.log(np.clip(q[idx, y], 1e-12, 1.0)).mean())

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


# =====================================================================
# 3. Uncertainty from stochastic passes
# =====================================================================
def mc_dropout_probs(mc_logits):
    """(P, n, K) logits -> mean predictive probability (n, K)."""
    return softmax(np.asarray(mc_logits, np.float64)).mean(0)


def mc_uncertainty(mc_logits):
    """
    Split predictive entropy into aleatoric and epistemic.
      total     = H[ E_p[y] ]
      aleatoric = E_p[ H[y] ]
      epistemic = total - aleatoric   (mutual information)
    """
    P = softmax(np.asarray(mc_logits, np.float64))
    mean_p = P.mean(0)
    total = -(mean_p * np.log(np.clip(mean_p, 1e-12, 1))).sum(-1)
    aleatoric = -(P * np.log(np.clip(P, 1e-12, 1))).sum(-1).mean(0)
    return {"total": total, "aleatoric": aleatoric,
            "epistemic": total - aleatoric}


def deep_ensemble_probs(logit_list):
    """Mean probability over independently seeded models."""
    return np.mean([softmax(np.asarray(z, np.float64)) for z in logit_list], 0)


def adaptive_ece_cc(conf, correct, M=15):
    """
    Adaptive (equal-mass) ECE from per-sample confidence and correctness only.
    Identical to har3d_v2.adaptive_ece(p, y) when conf = p.max(-1) and
    correct = (p.argmax(-1) == y), but avoids touching the (n, K) matrix, which
    makes bootstrap resampling of large test sets ~20x faster.
    """
    conf = np.asarray(conf, np.float64)
    correct = np.asarray(correct, np.float64)
    n = len(conf)
    order = np.argsort(conf, kind="stable")
    c, k = conf[order], correct[order]
    edges = np.linspace(0, n, M + 1).astype(int)
    starts = edges[:-1][edges[:-1] < edges[1:]]
    sizes = np.diff(np.append(starts, n))
    gap = np.abs(np.add.reduceat(k, starts) / sizes - np.add.reduceat(c, starts) / sizes)
    return float((gap * sizes / n).sum())


# holm and sign_test have one definition, in har3d_stats (shared with
# make_numbers.py); they are re-exported here for the stages that import C.
from har3d_stats import holm, sign_test  # noqa: E402,F401


def boot_mean_ci(x, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI of a mean over cells."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return float(x.mean()), float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


# =====================================================================
# 4. One row per (method, cell)
# =====================================================================
def evaluate_method(p, y, name="raw", n_sim=0, seed=0):
    """Metrics for one already-calibrated probability matrix."""
    out = {"method": name, "n": int(len(y)), "top1": top1(p, y),
           "ece_15": ece(p, y), "ece_adaptive": adaptive_ece(p, y),
           "brier": brier(p, y), "nll": nll(p, y)}
    if n_sim:
        out.update({f"null_{k}": v for k, v in
                    ece_significance(p, y, n_sim=n_sim, seed=seed).items()})
    return out