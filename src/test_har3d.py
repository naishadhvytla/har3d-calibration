"""
test_har3d.py -- correctness tests.

Every assertion here guards something that, if it broke, would SILENTLY corrupt
a calibration number rather than raise. Run before trusting any output.
"""
import numpy as np

import har3d_v2 as V
import har3d_calib as C
import har3d_scale as S

PASS = FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}" + (f"  [{extra}]" if extra else ""))
    else:
        FAIL += 1
        print(f"  FAIL {name}" + (f"  [{extra}]" if extra else ""))


def _depth(j, parents):
    d, p = 0, parents[j]
    while p >= 0:
        d += 1
        p = parents[p]
    return d


def synthetic_skeleton(T=200, seed=0):
    rng = np.random.default_rng(seed)
    rest = np.zeros((20, 3), np.float32)
    L = {0: .22, 1: .22, 2: .276, 3: .218, 4: .221, 5: .221,
         7: .275, 8: .275, 9: .293, 10: .293, 11: .07, 12: .07,
         13: .544, 14: .544, 15: .427, 16: .427, 17: .17, 18: .17, 19: .235}
    D = {0: (-1, .2, 0), 1: (1, .2, 0), 2: (0, 1, 0), 3: (0, 1, 0),
         4: (-1, -.2, 0), 5: (1, -.2, 0), 7: (-.4, -1, 0), 8: (.4, -1, 0),
         9: (0, -1, 0), 10: (0, -1, 0), 11: (0, -1, 0), 12: (0, -1, 0),
         13: (0, -1, 0), 14: (0, -1, 0), 15: (0, -1, 0), 16: (0, -1, 0),
         17: (0, 0, 1), 18: (0, 0, 1), 19: (0, 1, 0)}
    for j in sorted(range(20), key=lambda k: _depth(k, V.MSR_PARENTS)):
        p = V.MSR_PARENTS[j]
        if p < 0:
            continue
        d = np.array(D[j], np.float32)
        rest[j] = rest[p] + d / np.linalg.norm(d) * L[j]
    seq = np.repeat(rest[None], T, 0)
    ang = np.linspace(0, 1.2, T)
    for t in range(T):
        for j in (10, 12):
            v = seq[t, j] - seq[t, 8]
            c, s = np.cos(ang[t]), np.sin(ang[t])
            seq[t, j] = seq[t, 8] + np.array([v[0]*c - v[1]*s, v[0]*s + v[1]*c, v[2]])
    seq += rng.normal(0, 1e-4, seq.shape)
    return seq.astype(np.float32)


print("=" * 70)
print("  har3d test suite")
print("=" * 70)

rng = np.random.default_rng(0)
seq = synthetic_skeleton()

print("\n[1] topology")
rep = V.verify_topology([seq], raise_on_fail=False)
check("corrected tree passes on a synthetic anatomical body",
      all(r["ok"] for r in rep))
try:
    V.verify_topology([seq], parents=V.LEGACY_PARENTS)
    check("legacy tree is rejected", False, "no exception raised")
except AssertionError as e:
    check("legacy tree is rejected", str(e).count("->") >= 5,
          f"{str(e).count('->')} bad edges named")
check("tree is a single rooted tree",
      V.MSR_PARENTS.count(-1) == 1 and len(V.MSR_PARENTS) == 20)
check("no self-parent", all(j != p for j, p in enumerate(V.MSR_PARENTS)))
A = V.adjacency()
check("adjacency shape (3,20,20)", A.shape == (3, 20, 20))
rs = A.sum(-1)
check("adjacency rows sum to 1 or 0",
      np.all(np.isclose(rs, 1.0, atol=1e-6) | np.isclose(rs, 0.0, atol=1e-6)))
check("self partition is the identity", np.allclose(A[0], np.eye(20)))
check("centripetal + centrifugal reproduce the bone graph",
      int(((A[1] + A[2]) > 0).sum()) == 2 * 19)
try:
    V.adjacency(parents=[-1, 0] + [5] * 3 + [-1] * 15)
    check("disconnected tree raises", False)
except ValueError:
    check("disconnected tree raises", True)

print("\n[2] NTU topology (the ladder graph)")
check("NTU tree has 25 joints and one root",
      len(S.NTU_PARENTS) == 25 and S.NTU_PARENTS.count(-1) == 1)
ok = True
for j in range(25):
    p, hops = j, 0
    while S.NTU_PARENTS[p] >= 0:
        p = S.NTU_PARENTS[p]
        hops += 1
        if hops > 25:
            ok = False
            break
check("every NTU joint reaches the root, no cycles", ok)
An = V.adjacency(S.NTU_PARENTS)
check("NTU adjacency builds and is connected", An.shape == (3, 25, 25))

print("\n[3] missing-joint imputation")
s2 = seq.copy()
truth = s2[10, 5].copy()
s2[10, 5] = 0.0
fix, miss = V.impute_missing(s2)
check("gap is detected", bool(miss[10, 5]) and miss.sum() == 1)
check("interpolated value within 1 mm of truth",
      np.linalg.norm(fix[10, 5] - truth) < 1e-3)
s3 = seq.copy()
s3[:, 7] = 0.0
fix3, miss3 = V.impute_missing(s3)
check("always-missing joint stays zero and is flagged",
      np.allclose(fix3[:, 7], 0) and miss3[:, 7].all())

print("\n[4] normalisation invariances")
n = V.normalise_skeleton(seq, centre="perframe", align=True, scale="bone")
check("perframe centring puts the root at the origin every frame",
      np.allclose(n[:, V.ROOT], 0, atol=1e-5))
t = np.array([3.0, -2.0, 7.0], np.float32)
check("translation invariance",
      np.allclose(n, V.normalise_skeleton(seq + t, centre="perframe",
                                          align=True, scale="bone"), atol=1e-4))
Q = np.linalg.qr(rng.normal(size=(3, 3)))[0].astype(np.float32)
if np.linalg.det(Q) < 0:
    Q[:, 0] *= -1
check("rotation invariance (align=True)",
      np.allclose(n, V.normalise_skeleton(seq @ Q.T, centre="perframe",
                                          align=True, scale="bone"), atol=1e-3))
check("scale invariance",
      np.allclose(n, V.normalise_skeleton(seq * 1.7, centre="perframe",
                                          align=True, scale="bone"), atol=1e-4))

print("\n[5] streams")
b_s = V.make_stream(n, "bone")
bm = V.make_stream(n, "bone_motion")
check("root bone is exactly zero", np.allclose(b_s[:, V.ROOT], 0))
check("motion first frame is zero", np.allclose(bm[0], 0))
check("bone_motion is the temporal difference of bone",
      np.allclose(bm[1:], b_s[1:] - b_s[:-1]))
check("bone differs from the legacy tree's bone stream",
      not np.allclose(b_s, V.make_stream(n, "bone", parents=V.LEGACY_PARENTS)))

print("\n[6] subject splits")
subjects = np.repeat(np.arange(1, 11), 20)
labels = np.tile(np.arange(20), 10)
sp = V.subject_splits(subjects, labels=labels, n_splits=3, seed=0, verbose=False)
ok = True
for s in sp:
    trs, vas, tes = (set(subjects[s[k]].tolist()) for k in ("train", "val", "test"))
    ok &= not (trs & vas) and not (trs & tes) and not (vas & tes)
    ok &= len(s["train"]) + len(s["val"]) + len(s["test"]) == len(subjects)
check("train/val/test subject-disjoint and exhaustive", ok)
check("validation is never a single subject",
      all(len(s["subjects"]["val"]) >= 2 for s in sp))
fx = V.fixed_split(subjects, verbose=False)
check("fixed split: train and val are subject-disjoint (the old bug)",
      not (set(subjects[fx["train"]].tolist()) & set(subjects[fx["val"]].tolist())))

print("\n[7] temperature")
K, N = 20, 4000
rs2 = np.random.default_rng(7)
z = rs2.normal(size=(N, K)) * 2.0
y = np.array([rs2.choice(K, p=V.softmax(zi)) for zi in z])
T_hat = V.fit_temperature(z / 2.5, y)
check("recovers a known temperature", abs(T_hat - 1 / 2.5) < 0.05,
      f"got {T_hat:.4f}, expected {1/2.5:.4f}")
p0, pT = V.softmax(z), V.softmax(z, T_hat)
check("temperature NEVER changes accuracy",
      float((p0.argmax(-1) == y).mean()) == float((pT.argmax(-1) == y).mean()))
check("temperature does not reorder any prediction",
      np.array_equal(p0.argmax(-1), pT.argmax(-1)))

print("\n[8] metrics")
K2, N2 = 6, 6000
r2 = np.random.default_rng(11)
zc = r2.normal(size=(N2, K2)) * 1.6
yc = np.array([r2.choice(K2, p=V.softmax(zi)) for zi in zc])
pc = V.softmax(zc)
e_big = V.adaptive_ece(pc, yc)
e_small = V.adaptive_ece(pc[:400], yc[:400])
check("a perfectly calibrated model still reports a nonzero ECE floor",
      0.0 < e_big < 0.035, f"n={N2}: {e_big:.4f}")
check("that floor shrinks as n grows (pure estimator bias)",
      e_small > 2 * e_big, f"n=400: {e_small:.4f}")
po = V.softmax(zc, 0.35)
check("overconfidence is detected", V.adaptive_ece(po, yc) > 6 * e_big)
check("MCE >= ECE by construction", V.mce(po, yc) >= V.ece(po, yc))
p2 = np.array([[0.7, 0.3], [0.2, 0.8]])
y2 = np.array([0, 1])
check("Brier matches the closed form",
      abs(V.brier(p2, y2) - ((0.09 + 0.09) + (0.04 + 0.04)) / 2) < 1e-12)
check("NLL matches the closed form",
      abs(V.nll(p2, y2) - (-(np.log(0.7) + np.log(0.8)) / 2)) < 1e-12)
check("better confidence ranking gives lower AURC",
      V.aurc(V.softmax(zc), yc) < V.aurc(V.softmax(r2.permutation(zc)), yc))

print("\n[9] the null model -- the load-bearing new result")
vals, mu, p95 = C.ece_null(pc, n_sim=120, seed=0)
check("null ECE for a calibrated model is positive, not zero", mu > 0,
      f"n={N2} K={K2}: null mean {mu:.4f}")
small = V.softmax(r2.normal(size=(200, 20)) * 1.6)
_, mu_s, _ = C.ece_null(small, n_sim=120, seed=0)
check("the null floor GROWS as n shrinks and K grows", mu_s > 3 * mu,
      f"n=200 K=20: {mu_s:.4f} vs n={N2} K={K2}: {mu:.4f}")
sig_cal = C.ece_significance(pc, yc, n_sim=120)
check("a calibrated model is NOT flagged as miscalibrated",
      not sig_cal["separable"], f"p={sig_cal['p_value']:.3f}")
yo = np.array([r2.choice(K2, p=pi) for pi in V.softmax(zc, 3.0)])
sig_bad = C.ece_significance(V.softmax(zc, 0.3), yo, n_sim=120)
check("a badly miscalibrated model IS flagged",
      sig_bad["separable"] and sig_bad["excess"] > 0,
      f"excess {sig_bad['excess']:.3f}, p={sig_bad['p_value']:.3f}")

print("\n[10] paired bootstrap and vector scaling")
pb = C.paired_bootstrap(V.softmax(zc, 0.3), pc, yc, n_boot=300)
check("paired bootstrap detects a real calibration difference",
      pb["significant"] and pb["delta"] > 0, f"delta {pb['delta']:.4f}")
pb0 = C.paired_bootstrap(pc, pc, yc, n_boot=300)
check("a method compared with itself shows zero difference",
      abs(pb0["delta"]) < 1e-12)
w, b = C.fit_vector_scaling(zc[:2000], yc[:2000], l2=1e-2)
check("vector scaling stays near the identity under L2",
      0.3 < float(np.mean(w)) < 2.0 and abs(float(np.mean(b))) < 1.0,
      f"mean w {np.mean(w):.3f}, mean b {np.mean(b):.3f}")
pv = C.apply_vector_scaling(zc[2000:], w, b)
check("vector-scaled probabilities are a valid distribution",
      np.allclose(pv.sum(-1), 1.0))

print("\n[11] MC-dropout uncertainty decomposition")
mcl = np.stack([zc + r2.normal(0, 0.4, zc.shape) for _ in range(12)], 0)
u = C.mc_uncertainty(mcl)
check("total = aleatoric + epistemic",
      np.allclose(u["total"], u["aleatoric"] + u["epistemic"], atol=1e-9))
check("epistemic uncertainty is non-negative", (u["epistemic"] > -1e-9).all())
check("mc mean probs are a valid distribution",
      np.allclose(C.mc_dropout_probs(mcl).sum(-1), 1.0))

print("\n[12] ladder plumbing (numpy paths)")
fake = np.zeros((6, 300, 150), np.float32)
fake[:, :40] = np.random.default_rng(1).normal(0, 0.3, (6, 40, 150))
r = S.reshape_ntu(fake)
check("NTU reshape gives (N, 3, T, 25, 2)", r.shape == (6, 3, 300, 25, 2))
a, b_ = S.valid_length(r[0])
check("valid_length finds the padding boundary", (a, b_) == (0, 40), f"{(a, b_)}")
vr = S.valid_resize(r[0], T=64)
check("valid_resize returns T frames", vr.shape == (3, 64, 25, 2))
bs = S.to_stream(vr, "bone")
check("NTU bone stream zeroes the root joint",
      np.allclose(bs[:, :, 20], 0))
bm2 = S.to_stream(vr, "bone_motion")
check("NTU bone_motion is the temporal difference of bone",
      np.allclose(bm2[:, 1:], bs[:, 1:] - bs[:, :-1]))
yl = np.repeat(np.arange(10), 50)
idx = S.subsample_indices(yl, 200, seed=0)
check("stratified subsample has the requested size", len(idx) == 200)
cnt = np.bincount(yl[idx], minlength=10)
check("and keeps every class represented", cnt.min() >= 15, f"min {cnt.min()}")
check("subsample is deterministic for a fixed seed",
      np.array_equal(idx, S.subsample_indices(yl, 200, seed=0)))

print("\n[13] Brier decomposition, classwise ECE, data-quality filter")
rb = np.random.default_rng(21)
zb = rb.normal(size=(800, 7)) * 1.4
pb = V.softmax(zb)
yb = np.array([rb.choice(7, p=pi) for pi in pb])
bd = V.brier_decomposition(pb, yb)
check("Murphy identity is exact: REL - RES + UNC + WB == Brier",
      abs(bd["brier"] - V.brier(pb, yb)) < 1e-10, f"{bd['brier']:.6f}")
check("reliability, resolution, uncertainty are non-negative",
      bd["rel"] >= 0 and bd["res"] >= 0 and bd["unc"] >= 0)
yb_rand = rb.integers(0, 7, 800)
check("a forecast unrelated to the labels has ~zero resolution",
      V.brier_decomposition(pb, yb_rand)["res"] < 0.5 * bd["res"] + 0.01)
cw_cal = V.classwise_ece(pb, yb)
cw_bad = V.classwise_ece(V.softmax(zb, 0.3), yb)
check("classwise ECE separates calibrated from overconfident",
      cw_bad > 2 * cw_cal, f"{cw_cal:.4f} vs {cw_bad:.4f}")
fl = np.array([[40, 0.0, 0.00, 0.0], [38, 0.0, 1.00, 0.0],
               [5, 0.0, 0.00, 0.0], [60, 0.0, 0.40, 0.0]])
check("quality mask drops all-zero and too-short sequences only",
      V.quality_mask(fl, 0.5, 10).tolist() == [True, False, False, True])


print("\n[14] shared helpers: masked tensors, Holm, sign test, ladder protocol")
import har3d_calib as C2
rng15 = np.random.default_rng(15)
seqs = [rng15.normal(0, .3, (30, 20, 3)) + np.array([0, 0, 2.5]) for _ in range(6)]
seqs[2][:25] = 0.0
off15 = np.concatenate([[0], np.cumsum([len(q) for q in seqs])])
cache15 = {"xyz": np.concatenate(seqs), "offsets": off15,
           "meta": np.array([[1 + i % 2, 1 + i // 2, 1] for i in range(6)]),
           "flags": np.array([[len(q), 0.0, float((np.abs(q) < 1e-9).all(-1).mean()), 0.0]
                              for q in seqs])}
X15, y15, s15, k15 = V.masked_tensors(cache15, "joint", 16, dict(centre="frame0", align=False))
check("masked_tensors drops exactly the tracker-loss sequence",
      len(X15) == 5 and not k15[2] and k15.sum() == 5)
check("Holm is monotone and never below raw p",
      np.all(C2.holm([0.01, 0.04, 0.03]) >= np.array([0.01, 0.04, 0.03]))
      and np.allclose(C2.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06]))
check("sign test drops ties", C2.sign_test([1, 1, 0, -1])[:2] == (2, 3))
mu15, lo15, hi15 = C2.boot_mean_ci(rng15.normal(1, 1, 200))
check("bootstrap mean CI brackets the mean", lo15 < mu15 < hi15)
import tempfile, pathlib
from har3d_ntu import XSUB_TRAIN_SUBJECTS
tr_subj = sorted(XSUB_TRAIN_SUBJECTS)
perf15 = np.array([tr_subj[i % len(tr_subj)] for i in range(80)] + [3] * 20)
with tempfile.TemporaryDirectory() as td:
    pf = pathlib.Path(td) / "performer.txt"
    pf.write_text("\n".join(map(str, perf15)))
    y80 = np.arange(80) % 4
    vi, pi, pt, pr = S.ladder_protocol(None, y80, pf, 3, record=pathlib.Path(td) / "p.json")
    check("ladder validation is subject-disjoint from the pool",
          not set(pt[vi]) & set(pt[pi]) and len(vi) + len(pi) == 80)
    vi2, pi2, _, _ = S.ladder_protocol(None, y80, pf, 3, record=pathlib.Path(td) / "p.json")
    check("ladder protocol is reproducible across sessions", np.array_equal(pi, pi2))
    try:
        S.ladder_protocol(None, np.arange(79) % 4, pf, 3)
        check("ladder protocol refuses a performer/row mismatch (no silent fallback)", False)
    except RuntimeError:
        check("ladder protocol refuses a performer/row mismatch (no silent fallback)", True)

print("\n[15] Monte-Carlo p-values are never zero")
_p = np.full((300, 5), 0.2); _p[:, 0] = 0.99; _p /= _p.sum(1, keepdims=True)
_sig = C.ece_significance(_p, np.ones(300, int), n_sim=200, seed=0)
check("null-test p-value has the +1 floor", abs(_sig["p_value"] - 1 / 201) < 1e-12, _sig["p_value"])
_pb = C.paired_bootstrap(_p, np.full_like(_p, 0.2), np.ones(300, int), n_boot=100)
check("paired-bootstrap p-value is never zero", _pb["p_value"] >= 2 / 101 - 1e-12, _pb["p_value"])

_rng = np.random.default_rng(7)
for _n, _K in ((97, 10), (16487, 60)):
    _pp = V.softmax(_rng.normal(0, 2, (_n, _K))); _yy = _rng.integers(0, _K, _n)
    _fast = C.adaptive_ece_cc(_pp.max(-1), (_pp.argmax(-1) == _yy))
    check(f"fast adaptive ECE equals the reference (n={_n})", abs(_fast - V.adaptive_ece(_pp, _yy)) < 1e-12)

print("\n[16] inference over coarse units: test sets and seed-rung pairs (har3d_stats)")
import har3d_stats as HS
check("Holm and the sign test have one definition", C.holm is HS.holm and C.sign_test is HS.sign_test)
_u, _m = HS.unit_means([1.0, 3.0, 10.0, 20.0, 5.0], ["a", "a", "b", "b", "a"])
check("unit means average within units, in order of first appearance",
      _u == ["a", "b"] and np.allclose(_m, [3.0, 15.0]))
_cb = HS.cluster_boot_ci([0.2] * 12, n_boot=500)
check("cluster bootstrap of identical units is degenerate", np.allclose(_cb, (0.2, 0.2)))
_x = np.random.default_rng(4).normal(1, 1, 40)
_lo, _hi = HS.cluster_boot_ci(_x, n_boot=2000)
check("cluster bootstrap interval brackets the mean of the units", _lo < _x.mean() < _hi)
_b, _blo, _bhi, _r2 = HS.ols_loglog([500, 1000, 2000, 5000, 10000], [0.1 * (n / 500) ** -0.4 for n in (500, 1000, 2000, 5000, 10000)])
check("OLS log-log recovers an exact power law", abs(_b + 0.4) < 1e-9 and _r2 > 1 - 1e-12 and _blo <= _b <= _bhi)
_pc = []
for _ds in ("d1", "d2", "d3", "d4"):
    for _sp in ("s0", "s1", "s2"):
        for _st in ("joint", "bone"):
            for _sd in (0, 1):
                _pc.append(dict(method="raw", dataset=_ds, stream=_st, split=_sp, seed=_sd, ece=0.12))
                _pc.append(dict(method="good", dataset=_ds, stream=_st, split=_sp, seed=_sd, ece=0.10))
            _pc.append(dict(method="ens", dataset=_ds, stream=_st, split=_sp, seed=-1, ece=0.13))
_res, _eci = HS.method_comparison(_pc, ["good", "ens"], n_boot=500)
check("method comparison counts test sets, not cells", _res["good"]["n"] == 12 and _res["ens"]["n"] == 12)
check("ensemble is paired with the seed-mean of the baseline", abs(_res["ens"]["mean"] + 0.01) < 1e-12)
check("twelve concordant test sets give the exact two-sided sign-test p",
      abs(_res["good"]["p"] - 2 / 2 ** 12) < 1e-12 and _res["good"]["k"] == 12 and _res["ens"]["k"] == 0)
check("Holm over the methods, verdict from the sign of the mean",
      abs(_res["good"]["p_holm"] - 4 / 2 ** 12) < 1e-12 and _res["good"]["verdict"] == "reduces"
      and _res["ens"]["verdict"] == "increases")
_E = {(m, n, s): (0.05 if m == "a" else 0.02) + 0.001 * s for m in ("a", "b") for n in (1, 2, 3) for s in (0, 1, 2)}
_sl = HS.seed_level_tests(_E, [("a", "b")], {"low": [1], "high": [2, 3]})
check("seed-level tests pool seed-rung pairs per regime",
      [r["n_pairs"] for r in _sl] == [3, 6] and all(r["n_a_higher"] == r["n_pairs"] for r in _sl))
check("seed-level tests are Holm-corrected as one family",
      abs(_sl[1]["p_holm"] - 2 * 2 / 2 ** 6) < 1e-12 and abs(_sl[0]["p_holm"] - 0.25) < 1e-12)
_rows = [dict(dataset="ntu60", model="agcn", n_train=2000, seed=0, best_step=1000, ece_adaptive=0.05, top1=0.50),
         dict(dataset="ntu60", model="agcn_frozen", n_train=2000, seed=0, best_step=1000, ece_adaptive=0.057, top1=0.51)]
_g = HS.identical_run_gap(_rows)
check("identical-run gap is measured when both runs restore the same pre-freeze step",
      _g is not None and abs(_g["ece"] - 0.007) < 1e-12 and abs(_g["top1"] - 0.01) < 1e-12)
_rows[1]["best_step"] = 1250
check("identical-run gap refuses runs that diverged after the freeze", HS.identical_run_gap(_rows) is None)
_pe = HS.pooled_excess([dict(dataset="x", split=s, null_excess=v) for s, v in
                        (("a", 0.02), ("a", -0.01), ("b", 0.03), ("c", 0.01))])
check("pooled excess averages within test sets before the sign test",
      _pe["n"] == 3 and _pe["k_pos"] == 3 and np.allclose(_pe["unit_excess"], [0.005, 0.03, 0.01]))
# methods are tested on the score they are given (Brier, NLL, excess), not on ECE
for _r in _pc:
    _r["brier"] = {"raw": 0.40, "good": 0.42, "ens": 0.38}[_r["method"]]
_resb, _ = HS.method_comparison(_pc, ["good", "ens"], n_boot=500, key="brier")
check("method comparison tests the score it is given",
      _resb["good"]["verdict"] == "increases" and _resb["ens"]["verdict"] == "reduces"
      and _res["good"]["verdict"] == "reduces")
check("method comparison reports the agreement within each dataset",
      _resb["ens"]["per_dataset"] == {d: (3, 3) for d in ("d1", "d2", "d3", "d4")})
_E2 = {("m", 1, s): v for s, v in enumerate((0.10, 0.12, 0.14))}
_E2.update({("m", 2, s): v for s, v in enumerate((0.20, 0.20, 0.20))})
_sd2, _ng2 = HS.pooled_seed_sd(_E2, ["m"], [1, 2])
check("pooled seed s.d. is the root mean square of the per-group s.d.",
      _ng2 == 2 and abs(_sd2 - np.sqrt((0.02 ** 2 + 0.0) / 2)) < 1e-12)
_pf = V.softmax(np.random.default_rng(11).normal(0, 1.5, (300, 10)))
_f_raw = C.ece_null(_pf, n_sim=300, seed=0)[1]
_f_sharp = C.ece_null(C.sharpen_probs(_pf, 0.3), n_sim=300, seed=0)[1]
check("the floor moves with the confidences: sharpening lowers it", _f_sharp < _f_raw, (_f_raw, _f_sharp))

print("\n[18] revision analyses (har3d_review)")
import har3d_review as RV
_rg = np.random.default_rng(21)
_p = V.softmax(_rg.normal(0, 2.0, (400, 12)))
_y = np.array([_rg.choice(12, p=q) for q in _p])
_conf, _corr = _p.max(-1), (_p.argmax(-1) == _y)
_prof = RV.Profile(_conf)
_e = RV.estimates(_prof, _prof.sort(_corr))
check("vectorised equal-mass ECE equals the reference for 5, 15 and 30 bins",
      all(abs(_e[f"aece{M}"][0] - V.adaptive_ece(_p, _y, M)) < 1e-6 for M in (5, 15, 30)))
check("vectorised equal-width ECE equals the reference", abs(_e["ece15"][0] - V.ece(_p, _y)) < 1e-6)
_ks = np.abs(np.cumsum((_conf - _corr)[np.argsort(_conf, kind="stable")])).max() / len(_conf)
check("KS calibration error is the largest cumulative confidence-accuracy gap",
      abs(_e["ks"][0] - _ks) < 1e-6)
_pc_ref = V.per_class_ece(_p, _y)
_pc_new = RV.per_class_ece_cc(_conf, _p.argmax(-1), _corr, 12)[0]
check("per-class ECE from (confidence, prediction, correct) equals the reference",
      np.allclose(np.nan_to_num(_pc_ref, nan=-1), np.nan_to_num(_pc_new, nan=-1), atol=1e-6))
_ct = RV.cell_tests(_conf, _corr, S=2000, seed=0)
_ref_floor = C.ece_null(_p, n_sim=400, seed=0)[1]
check("Bernoulli(confidence) null reproduces the floor of the full-vector label draw",
      abs(_ct["aece15"]["null_mean"] - _ref_floor) < 0.004, (_ct["aece15"]["null_mean"], _ref_floor))
check("Monte-Carlo p-values are never zero and at most one",
      RV.mc_p(np.zeros(99), 1.0) == 0.01 and RV.mc_p(np.ones(99), 0.0) == 1.0)
_nd = RV.estimates(_prof, RV.null_correct(_prof, 4000, np.random.default_rng(1)), ("ce2db15", "aece15"))
check("the debiased squared calibration error is centred on zero under the null",
      abs(_nd["ce2db15"].mean()) < 0.1 * _nd["aece15"].mean() ** 2 + 2e-4, _nd["ce2db15"].mean())
# coupled null: marginals unchanged, cells that share clips move together
_clipA = np.arange(400)
_cells = [dict(conf=_conf, correct=_corr, clip=_clipA),
          dict(conf=_conf, correct=_corr, clip=_clipA),
          dict(conf=_conf, correct=_corr, clip=_clipA + 400)]
_cn = RV.coupled_null(_cells, 800, S=3000, seed=0)["aece15"]
check("coupled null keeps the marginal floor of every cell",
      np.allclose(_cn["null"].mean(1), _ct["aece15"]["null_mean"], atol=0.003))
check("coupled null: cells on the same clips are tied, cells on other clips are not",
      np.corrcoef(_cn["null"][0], _cn["null"][1])[0, 1] > 0.99
      and abs(np.corrcoef(_cn["null"][0], _cn["null"][2])[0, 1]) < 0.08)
_pl = RV.pooled_from_null(_cn["obs"], _cn["null"], ["u1", "u1", "u2"], ["d", "d", "d"])
check("pooled test reports the inflation of the variance by shared clips",
      1.2 < _pl["var_inflation"] < 2.3 and _pl["n_units"] == 2 and 0 < _pl["p"] <= 1,
      _pl["var_inflation"])
# power: size at b = 0, monotone, and the minimum detectable error is finite
_cs = RV.synthetic_conf(300, 20, 0.6)
check("synthetic confidence profile has the requested mean", abs(_cs.mean() - 0.6) < 0.01, _cs.mean())
_rows, _, _ = RV.power_curve(_cs, bs=(0.5, 1.0, 2.0), S_null=4000, R=400, seed=0)
_pw = {r["b"]: r["power_0.05"] for r in _rows}
check("null test has its nominal size", 0.02 < _pw[0.0] < 0.09, _pw[0.0])
check("power grows with the miscalibration", _pw[0.0] < _pw[0.5] < _pw[1.0] <= _pw[2.0] and _pw[2.0] > 0.95, _pw)
_m = RV.mde(_rows)
check("minimum detectable calibration error lies between the tested errors",
      0 < _m < max(r["eps"] for r in _rows), _m)
_sub = np.repeat(np.arange(4), 75)
_cz = RV.clustered_size(_cs, _sub, (0.0, 0.15), S_null=1500, R=600, seed=0)
check("subject effects inflate the size of the i.i.d. null; the cluster-aware null repairs it",
      _cz[0]["size_iid"] < 0.1 and _cz[1]["size_iid"] > 0.3 and _cz[1]["size_cluster"] < 0.1,
      [(r["size_iid"], r["size_cluster"]) for r in _cz])
_het0 = RV.subject_heterogeneity(_cs, np.random.default_rng(3).random(300) < _cs, _sub, S=1000)
_d = np.array([0.25, -0.25, 0.2, -0.2])[_sub]
_het1 = RV.subject_heterogeneity(_cs, np.random.default_rng(3).random(300) < np.clip(_cs + _d, 0, 1), _sub, S=1000)
check("subject heterogeneity: over-dispersion detected only when subjects differ",
      _het0["p"] > 0.01 and _het1["p"] < 0.01 and _het1["tau2"] > 0.01, (_het0["phi"], _het1["phi"]))
_ss = RV.seed_summary([0.03, 0.04, 0.02, 0.05, 0.01])
check("five seeds of one sign give the smallest exact sign-flip p-value, 2/32",
      abs(_ss["p_flip"] - 0.0625) < 1e-12 and _ss["n_pos"] == 5 and _ss["lo"] > 0)
_acc = RV.SubjectAccumulator()
_acc.add("d1", np.array([1, 1, 2, 2]), np.array([1.0, 3.0, -1.0, -3.0]))
_acc.add("d1", np.array([1]), np.array([5.0]))
_k, _v = _acc.means()
check("subject accumulator averages per subject over every cell",
      _k == [("d1", 1), ("d1", 2)] and np.allclose(_v, [3.0, -2.0]))
_st = RV.subject_test([("d", i) for i in range(9)], np.full(9, 0.1) + np.arange(9) * 1e-3, n_boot=500)
check("subject-level test: nine subjects of one sign", _st["k"] == 9 and _st["p"] < 0.01 and _st["lo"] > 0)
_am = RV.SubjectAccumulator()
_am.add("d", np.arange(12), np.r_[np.full(11, 0.01), -1.0])          # better for 11 of 12 subjects, worse on average
_vm = RV.subject_method_tests({"m": _am}, ["m"], n_boot=200)["m"]
check("a significant majority with a mean of the opposite sign is labelled mixed",
      _vm["k"] == 11 and _vm["mean"] < 0 and _vm["verdict"] == "mixed", _vm["verdict"])

print("\n[17] backbones and ladder trainer (torch)")
try:
    import torch
    import har3d_backbones as HB
    torch.manual_seed(0)
    xt = np.random.default_rng(3).normal(0, 0.3, (6, 3, 16, 25, 2)).astype(np.float32)
    yt = np.array([0, 1, 2, 0, 1, 2])
    for arch in ("stgcn", "agcn", "ctrgcn"):
        m = HB.build(arch, 3, S.NTU_PARENTS, base=16)
        with torch.no_grad():
            o = m(torch.as_tensor(xt))
        check(f"{arch} forward gives (N, K)", tuple(o.shape) == (6, 3))
    check("ST-GCN has no adaptive-graph parameters",
          len(HB.graph_parameters(HB.build("stgcn", 3, S.NTU_PARENTS, base=16))) == 0)
    m = HB.build("agcn", 3, S.NTU_PARENTS, base=16)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2, weight_decay=1e-4)
    HB.freeze_graph(m)
    g0 = [q.clone() for q in HB.graph_parameters(m)]
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(m(torch.as_tensor(xt)),
                                          torch.as_tensor(yt)).backward()
        opt.step()
    check("freezing stops every adaptive-graph parameter from updating",
          all(torch.equal(a, b) for a, b in zip(g0, HB.graph_parameters(m))))
    r = S.train_rung(xt, yt, xt, yt, xt, yt, 3, arch="stgcn", max_steps=12,
                     eval_every=4, patience=2, min_steps=4, bs=4, seed=0,
                     verbose=False)
    check("early-stopping rung restores a step no later than it ran",
          1 <= r["info"]["best_step"] <= r["info"]["steps_run"])
    # bit-for-bit comparison of two trainings: on the CPU, where arithmetic is
    # reproducible (two GPU runs of one seed can differ in the last bits)
    r1 = S.train_rung(xt, yt, xt, yt, xt, yt, 3, arch="stgcn", max_steps=12,
                      eval_every=4, patience=2, min_steps=4, bs=4, seed=0,
                      verbose=False, device="cpu")
    r2 = S.train_rung(xt, yt, xt, yt, xt, yt, 3, arch="stgcn", max_steps=12,
                      eval_every=4, patience=2, min_steps=4, bs=4, seed=0,
                      verbose=False, snapshot_steps=(4, 8, 24), track_best_acc=True,
                      run_until=24, device="cpu")
    check("snapshots, the best-accuracy checkpoint and run_until leave the main output unchanged",
          np.array_equal(r1["test"][0], r2["test"][0]) and np.array_equal(r1["val"][0], r2["val"][0])
          and r1["info"]["best_step"] == r2["info"]["best_step"]
          and r1["info"]["steps_run"] == r2["info"]["steps_run"])
    check("matched-step snapshots and the best-accuracy checkpoint are returned",
          {"s4", "s8", "acc"} <= set(r2["extra"]) and r2["extra"]["s4"]["logits"].shape == (6, 3)
          and r2["info"]["steps_total"] >= r2["info"]["steps_run"])
except ImportError:
    print("  (torch not installed -- backbone tests skipped)")

print("\n" + "=" * 70)
print(f"  {PASS} passed, {FAIL} failed")
print("=" * 70)
raise SystemExit(1 if FAIL else 0)