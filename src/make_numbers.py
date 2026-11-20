"""
make_numbers.py RESULTS_DIR OUT_DIR

Writes every number, table and caption fragment the manuscript quotes, from a
results folder of the notebook (runs/final/results):

  OUT_DIR/numbers.tex              \\newcommand macros used in the prose
  OUT_DIR/tab_*.tex                the tables
  OUT_DIR/methods_split_level.json what make_figures.py needs for Fig. 4

Inference is made over coarse units only (har3d_stats.py): test sets
(dataset x split) on the small datasets, which share subjects within a dataset
so their p-values are approximate, and seed-rung pairs on the ladder. The
manuscript contains no typed result that is not produced here; only constants
of the protocol (bin counts, seeds, grid values) are typed.
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import har3d_stats as HS  # noqa: E402

R = Path(sys.argv[1])
O = Path(sys.argv[2])
O.mkdir(parents=True, exist_ok=True)
T = R / "tables"
rows = json.loads((R / "metrics.json").read_text())
# replication and environment re-runs (revision) are reported from their own tables
rows = [r for r in rows if not r["model"].endswith(("_rep", "_env", "_pin")) and "_v8" not in r["model"]
        and r["model"] not in ("agcn_static", "agcn_fixedgraph")]   # v8 runs: Stage 9i
MAC = {}


def mac(name, val):
    assert name.isalpha(), name
    MAC[name] = str(val)


def f3(x):
    return f"{x:.3f}"


def f2(x):
    return f"{x:.2f}"


def sgn(x, d=3):
    if abs(x) < 0.5 * 10 ** (-d):
        return f"{0:.{d}f}"
    return f"{x:+.{d}f}".replace("-", "$-$").replace("+", "$+$")


def neg(s):
    return s.replace("-", "$-$")


def pfmt(p):
    return "$<$0.001" if p < 1e-3 else f"{p:.3f}"


def prel(p):
    """p with its relation, for prose: '\\,=\\,0.012' or '\\,$<$\\,0.001'."""
    return r"\,$<$\,0.001" if p < 1e-3 else r"\,=\," + f"{p:.3f}"


def thou(n):
    return f"{int(n):,}".replace(",", "{,}")


def read_csv(name):
    return list(csv.DictReader(open(T / name)))


DS = ["action3d", "daily", "utkinect"]
DSN = {"action3d": "Action", "daily": "Daily", "utkinect": "UTKinect"}
DSD = {"action3d": "Action3D", "daily": "Daily", "utkinect": "UTKinect"}
ST = ["joint", "bone", "joint_motion", "bone_motion"]
STN = {"joint": "joint", "bone": "bone", "joint_motion": "joint-mot.", "bone_motion": "bone-mot."}
small = [r for r in rows if r["model"] == "stgcn" and r["dataset"] in DS]
assert len(small) == 240, len(small)
ntu = [r for r in rows if r["dataset"] == "ntu60"]
Lr = {(r["model"], r["n_train"], r["seed"]): r for r in ntu}

# ====================================================================== data
filt = json.loads((R / "data_filtering.json").read_text())
mac("nActionKept", filt["action3d"]["n_kept"])
mac("nActionDrop", filt["action3d"]["n_dropped"])
mac("nDailyKept", filt["daily"]["n_kept"])
mac("nUTKept", filt["utkinect"]["n_kept"])
proto = json.loads((R / "ladder_protocol.json").read_text())
mac("nNTUval", thou(proto["n_val"]))
mac("nNTUpool", thou(proto["n_pool"]))
mac("NTUvalSubjects", ", ".join(str(s) for s in proto["val_subjects"]))
_ns_a = sorted({r["n"] for r in small if r["dataset"] == "action3d"})
mac("nActionTestMin", _ns_a[0])
mac("nActionTestMax", _ns_a[-1])

# ====================================================================== RQ1
# per-experiment null tests, Holm over all 240 cells
padj = HS.holm([r["null_p_value"] for r in small])
SEP = {(r["dataset"], r["stream"], r["split"], r["seed"]): (r["null_p_value"], a) for r, a in zip(small, padj)}
nH = sum(a < 0.05 for _, a in SEP.values())
nR = sum(p < 0.05 for p, _ in SEP.values())
mac("nSmallCells", len(SEP))
mac("nSepHolm", nH)
mac("nSepRaw", nR)
mac("pctSepHolm", f"{100 * nH / len(SEP):.1f}")
mac("nExpectedRaw", int(round(0.05 * len(SEP))))
for ds in DS:
    mac(f"nSep{DSN[ds]}", sum(a < 0.05 for k, (_, a) in SEP.items() if k[0] == ds))
mac("nSepActionMotion", sum(a < 0.05 for k, (_, a) in SEP.items() if k[0] == "action3d" and "motion" in k[1]))
mac("nSims", thou(small[0]["null_sims"]))
mac("minHolmP", f"{len(SEP) / (small[0]['null_sims'] + 1):.3f}")

# RQ1 in aggregate: test sets are the units
pe = HS.pooled_excess(small)
mac("nTestSets", pe["n"])
mac("nTestSetsPos", pe["k_pos"])
mac("pTestSetsPos", pfmt(pe["p"]))
mac("meanExcessPooled", f3(pe["mean"]))
# the test sets of a dataset share subjects, so agreement within each dataset is reported too
for ds in DS:
    mac(f"tsPos{DSN[ds]}", "{}/{}".format(*pe["per_dataset"][ds]))


def by_split(sub, key):
    g = defaultdict(list)
    for r in sub:
        g[r["split"]].append(r[key])
    return float(np.mean([np.mean(v) for v in g.values()]))


lines = []
for ds in DS:
    for st in ST:
        sub = [r for r in small if r["dataset"] == ds and r["stream"] == st]
        t, e = by_split(sub, "top1"), by_split(sub, "ece_adaptive")
        fl = np.mean([r["null_null_mean"] for r in sub])
        ex = np.mean([r["null_excess"] for r in sub])
        ns = sum(SEP[(r["dataset"], r["stream"], r["split"], r["seed"])][1] < 0.05 for r in sub)
        lines.append(f"{DSD[ds]} & {STN[st]} & {sub[0]['n_classes']} & {f3(t)} & {f3(e)} & {f3(fl)} & "
                     f"{sgn(ex)} & {ns}/{len(sub)} & {f2(np.mean([r['temperature'] for r in sub]))} & "
                     f"{f3(np.mean([r['brier'] for r in sub]))}\\\\")
        if st == "joint":
            mac(f"floor{DSN[ds]}Joint", f3(fl))
            mac(f"ece{DSN[ds]}Joint", f3(e))
    if ds != DS[-1]:
        lines.append("\\addlinespace[2pt]")
(O / "tab_small.tex").write_text(r"""\begin{table*}[t]
\caption{Small datasets, uncalibrated ST-GCN: mean over the 20 cells of each
dataset and stream (4 subject-disjoint splits $\times$ 5 seeds). Top-1 and
$\aece$ average seeds within a split, then splits. ``Null'' is the matched
estimator floor and ``excess'' the observed $\aece$ minus that floor; ``sep.''
counts cells whose observed $\aece$ is separable from the null after Holm
correction over all """ + str(len(SEP)) + r""" cells. $T$ is the validation-fitted
temperature ($T<1$: under-confident).}
\label{tab:small}
\centering\footnotesize
\begin{tabular}{@{}llcccccccc@{}}
\toprule
Dataset & Stream & $K$ & top-1 & $\aece$ & null & excess & sep. & $T$ & Brier\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# temperatures
mac("nUnderconf", sum(r["temperature"] < 1 for r in small))
mac("meanTSmall", f2(np.mean([r["temperature"] for r in small])))
for m, name in (("stgcn_ls", "LS"), ("stgcn_focal", "Focal")):
    s = [r for r in rows if r["model"] == m]
    mac(f"meanT{name}", f2(np.mean([r["temperature"] for r in s])))
    mac(f"nUnderconf{name}", sum(r["temperature"] < 1 for r in s))
small_all = [r for r in rows if r["dataset"] != "ntu60"]
mac("nTbound", sum(r["temperature"] <= 0.0501 for r in small))
mac("nTboundAll", sum(r["temperature"] <= 0.0501 for r in small_all))
mac("nEpochCap", sum((r["epochs_run"] or 0) >= 120 for r in small_all))
mac("nSmallAll", len(small_all))
_tj = defaultdict(list)
for r in small:
    if r["stream"] == "joint":
        _tj[r["dataset"]].append(r["temperature"])
for ds in DS:
    mac(f"Tj{DSN[ds]}Min", f2(min(_tj[ds])))
    mac(f"Tj{DSN[ds]}Max", f2(max(_tj[ds])))

# worked example: the cells are fixed choices; their values are read from the run
M = {(r["dataset"], r["stream"], r["split"], r["seed"]): r for r in small}
for tag, key in (("WA", ("action3d", "bone_motion", "fixed_frame0", 3)),
                 ("WB", ("daily", "bone", "rand0_frame0", 1))):
    r = M[key]
    mac(f"{tag}obs", f3(r["null_observed"]))
    mac(f"{tag}null", f3(r["null_null_mean"]))
    mac(f"{tag}pninetyfive", f3(r["null_null_p95"]))
    mac(f"{tag}excess", f3(r["null_excess"]))
    mac(f"{tag}p", pfmt(r["null_p_value"]))
    mac(f"{tag}n", r["n"])
    mac(f"{tag}holm", "separable" if SEP[key][1] < 0.05 else "not separable")
_wa = [M[("action3d", "bone_motion", "fixed_frame0", s)]["null_excess"] for s in range(5)]
assert M[("action3d", "bone_motion", "fixed_frame0", 3)]["null_excess"] == max(_wa), "worked example A is described as the largest of its five seeds"

# ====================================================================== RQ4 (per class)
for ds in DS:
    sub = [r for r in small if r["dataset"] == ds and r["stream"] == "joint"]
    mac(f"pc{DSN[ds]}Mean", f3(np.nanmean([r["per_class_ece_mean"] for r in sub])))
    mac(f"pc{DSN[ds]}Max", f3(np.nanmean([r["per_class_ece_max"] for r in sub])))
    mac(f"cw{DSN[ds]}", f3(np.mean([r["classwise_ece"] for r in sub])))
pcp = read_csv("per_class_pooled.csv")
for ds in DS:
    v = [float(x["per_class_ece"]) for x in pcp if x["dataset"] == ds]
    mac(f"pcPool{DSN[ds]}Min", f2(min(v)))
    mac(f"pcPool{DSN[ds]}Max", f2(max(v)))

# ====================================================================== ladder
lad = read_csv("ladder_by_model.csv")
L = {(x["model"], int(x["n_train"])): x for x in lad}
NS = sorted({int(x["n_train"]) for x in lad})
top = NS[-1]
mac("nTop", thou(top))
mac("foldRange", f"{NS[-1] / NS[0]:.0f}")
cols = [("stgcn", "ST-GCN"), ("agcn", "2s-AGCN"), ("ctrgcn", "CTR-GCN"),
        ("agcn_frozen", "AGCN frozen"), ("stgcn_fixed", "ST-GCN 6k")]
body = []
for n in NS:
    cells = []
    for m, _ in cols:
        x = L.get((m, n))
        cells.append("-- & --" if x is None else f"{float(x['top1_mean']):.3f} & {float(x['excess_mean']):.3f}")
    fl = np.mean([float(L[(m, n)]["null_floor_mean"]) for m, _ in cols if (m, n) in L])
    body.append(thou(n) + f" & {fl:.4f} & " + " & ".join(cells) + r"\\")
head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{nm}}}" for _, nm in cols)
cm = "".join(f"\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(cols)))
(O / "tab_ladder.tex").write_text(r"""\begin{table*}[t]
\caption{NTU-60 training-set ladder (fixed X-Sub test set of 16{,}487 clips),
mean over three seeds. Each entry gives top-1 and the excess adaptive ECE
over the matched null; ``floor'' is the mean null $\aece$ at that rung. All
models except the last column are trained to their validation-NLL optimum;
``ST-GCN 6k'' uses the fixed 6{,}000-step budget, and ``AGCN frozen'' stops
updating its graph parameters after 1{,}000 steps.}
\label{tab:ladder}
\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}rc""" + "cc" * len(cols) + r"""@{}}
\toprule
 & & """ + head + r"""\\
""" + cm + r"""
$n_{\mathrm{tr}}$ & floor & """ + " & ".join(["top-1 & exc."] * len(cols)) + r"""\\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


def L_(m, n, k):
    return float(L[(m, n)][k])


for m, nm in (("stgcn", "ST"), ("agcn", "AG"), ("ctrgcn", "CT"), ("agcn_frozen", "FR"), ("stgcn_fixed", "FX")):
    ns = sorted(n for (mm, n) in L if mm == m)
    mac(f"exc{nm}First", f3(L_(m, ns[0], "excess_mean")))
    mac(f"exc{nm}Top", f3(L_(m, top, "excess_mean")))
    mac(f"acc{nm}First", f3(L_(m, ns[0], "top1_mean")))
    mac(f"acc{nm}Top", f3(L_(m, top, "top1_mean")))
    mac(f"T{nm}First", f2(L_(m, ns[0], "temperature_mean")))
    mac(f"T{nm}Top", f2(L_(m, top, "temperature_mean")))
    mac(f"rel{nm}First", f3(L_(m, ns[0], "brier_rel_mean")))
    mac(f"rel{nm}Top", f3(L_(m, top, "brier_rel_mean")))
    mac(f"ts{nm}Top", f3(L_(m, top, "ece_adaptive_ts_mean")))
_tn = [r["temperature"] for r in ntu if r["model"] in ("stgcn", "agcn", "ctrgcn")]
mac("nNTUruns", len(_tn))
mac("nNTUunder", sum(t < 1 for t in _tn))
mac("excSTtwok", f3(L_("stgcn", 2000, "excess_mean")))
mac("excFXfivek", f3(L_("stgcn_fixed", 5000, "excess_mean")))
mac("TFXfivek", f2(L_("stgcn_fixed", 5000, "temperature_mean")))
mac("TSTfivek", f2(L_("stgcn", 5000, "temperature_mean")))
mac("accSTtwentyk", f3(L_("stgcn", 20000, "top1_mean")))
mac("excSTtwentyk", f3(L_("stgcn", 20000, "excess_mean")))
mac("accGapAGTop", f3(L_("stgcn", top, "top1_mean") - L_("agcn", top, "top1_mean")))
mac("accGapCTTop", f3(L_("stgcn", top, "top1_mean") - L_("ctrgcn", top, "top1_mean")))
for m, nm in (("agcn", "AG"), ("ctrgcn", "CT")):
    ex = [L_(m, n, "excess_mean") for n in NS]
    mac(f"exc{nm}Min", f3(min(ex)))
    mac(f"exc{nm}Max", f3(max(ex)))
ts = [L_(m, n, "ece_adaptive_ts_mean") for (m, n) in L if m in ("stgcn", "agcn", "ctrgcn")]
mac("tsLadderMin", f3(min(ts)))
mac("tsLadderMax", f3(max(ts)))
_ts_cells = [r["ece_adaptive_temp"] for r in ntu if r["model"] in ("stgcn", "agcn", "ctrgcn")]
mac("tsCellMin", f3(min(_ts_cells)))
mac("tsCellMax", f3(max(_ts_cells)))
fl = [float(x["null_floor_mean"]) for x in lad]
mac("floorLadderMin", f"{min(fl):.4f}")
mac("floorLadderMax", f"{max(fl):.4f}")
_rat = [L_(m, n, "excess_mean") / L_(m, n, "null_floor_mean") for m in ("stgcn", "agcn", "ctrgcn") for n in NS]
mac("ratioMin", f"{min(_rat):.0f}")
mac("ratioMax", f"{max(_rat):.0f}")

conv = {(x["model"], int(x["n_train"])): x for x in read_csv("ladder_convergence.csv")}
for m, nm in (("stgcn", "ST"), ("agcn", "AG"), ("ctrgcn", "CT")):
    mac(f"step{nm}Top", f"{float(conv[(m, top)]['best_step_mean']):,.0f}".replace(",", "{,}"))
mac("epochsFXFirst", f"{float(conv[('stgcn_fixed', NS[0])]['epochs_at_best_mean']):.0f}")
mac("epochsFXTop", f"{float(conv[('stgcn_fixed', top)]['epochs_at_best_mean']):.0f}")
mac("maxStepsRun", thou(max(r["steps_run"] for r in ntu if r["model"] != "stgcn_fixed")))
_es = [r for r in ntu if r["model"] in ("agcn", "ctrgcn") and r["n_train"] == NS[0]]
mac("firstEvalStops", sum(r["best_step"] == 250 for r in _es))
mac("nFirstEvalCells", len(_es))
_fr2k = [Lr[("agcn_frozen", 2000, s)]["best_step"] for s in (0, 1, 2)]
mac("frTwokStepMin", thou(min(_fr2k)))
mac("frTwokStepMax", thou(max(_fr2k)))

# scaling law: OLS on the ST-GCN rung means (interval includes lack of fit)
_ns = sorted({r["n_train"] for r in ntu if r["model"] == "stgcn"})
_m = [np.mean([r["null_excess"] for r in ntu if r["model"] == "stgcn" and r["n_train"] == n]) for n in _ns]
b, lo, hi, r2 = HS.ols_loglog(_ns, _m)
mac("bST", neg(f"{b:.2f}"))
mac("bSTlo", neg(f"{lo:.2f}"))
mac("bSThi", neg(f"{hi:.2f}"))
mac("rtwoST", f"{r2:.2f}")
mac("nRungsST", len(_ns))
mac("stTopNullP", pfmt(max(r["null_p_value"] for r in ntu if r["model"] == "stgcn" and r["n_train"] == top)))
mac("nSimsNTU", thou(ntu[0]["null_sims"]))

# run-to-run noise: two nominally identical runs (one pair: an illustration)
gap = HS.identical_run_gap(rows)
assert gap is not None, "identical-run check needs both runs restored at or before the freeze step"
mac("noiseECE", f3(gap["ece"]))
mac("noiseAcc", f3(gap["top1"]))
# run-to-run noise estimated from the seeds: spread of aECE across the three
# seeds of each model and large rung, and the standard error it implies for a
# difference of two three-seed means (the quantity in Table pairs)
big = [n for n in NS if n >= 10000]
low = [n for n in NS if n <= 5000]
ECE = {k: r["ece_adaptive"] for k, r in Lr.items()}
_seeds = sorted({s for (_, _, s) in Lr})
_sd, _ng = HS.pooled_seed_sd(ECE, ["stgcn", "agcn", "ctrgcn"], big, seeds=_seeds)
_se = _sd * np.sqrt(2 / len(_seeds))
mac("noiseSeedSD", f"{_sd:.4f}")
mac("nSeedSDGroups", _ng)
mac("seDiff", f"{_se:.4f}")

# paired differences on the fixed test set (bootstrap over clips, Table pairs)
pr = read_csv("ladder_paired_comparisons.csv")
P = {(x["model_a"], x["model_b"], int(x["n_train"])): x for x in pr}
nbt = thou(int(pr[0].get("n_boot") or 500))
fam = [("agcn", "stgcn", "2s-AGCN $-$ ST-GCN"), ("ctrgcn", "stgcn", "CTR-GCN $-$ ST-GCN"),
       ("agcn_frozen", "agcn", "AGCN frozen $-$ 2s-AGCN"), ("stgcn_fixed", "stgcn", "ST-GCN 6k $-$ ST-GCN")]
body = []
for a, b_, nm in fam:
    cells = []
    for n in NS:
        x = P.get((a, b_, n))
        if x is None:
            cells.append("--")
            continue
        d = [Lr[(a, n, s)]["ece_adaptive"] - Lr[(b_, n, s)]["ece_adaptive"] for s in (0, 1, 2)]
        agree = int(sum(np.sign(d) == np.sign(np.mean(d))))
        cells.append(sgn(float(x["delta_ece_adaptive"])) + ("$^{\\dagger}$" if agree < 3 else ""))
    body.append(nm + " & " + " & ".join(cells) + r"\\")
(O / "tab_pairs.tex").write_text(r"""\begin{table*}[t]
\caption{Paired differences in adaptive ECE on the fixed NTU-60 test set
(A $-$ B; negative: A better calibrated), averaged over three seeds; within a
seed, A and B see the same training subsample. Every entry differs from zero
under a bootstrap over test clips (""" + nbt + r""" draws, Holm-corrected within each
family: the two graph rows form one family of 14 tests, the freezing and
budget rows one family each), but that test holds the trained models fixed and
ignores training non-determinism: across seeds, $\aece$ varies with a standard
deviation of about """ + MAC["noiseSeedSD"] + r""" at the large rungs, so a
difference of two three-seed means has a standard error of about """ + MAC["seDiff"] + r""".
$^{\dagger}$: only two of the three seeds agree in sign; all other entries
agree in all three seeds.}
\label{tab:pairs}
\centering\footnotesize
\begin{tabular}{@{}l""" + "c" * len(NS) + r"""@{}}
\toprule
$n_{\mathrm{tr}}$ & """ + " & ".join(thou(n) for n in NS) + r"""\\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")
assert all(x["verdict"] != "n.s." for x in pr), "Table pairs caption says every entry is significant"
for a, b_, nm in (("agcn", "stgcn", "AG"), ("ctrgcn", "stgcn", "CT"), ("agcn_frozen", "agcn", "FR"),
                  ("stgcn_fixed", "stgcn", "FX")):
    first = min(n for (aa, bb, n) in P if (aa, bb) == (a, b_))
    mac(f"d{nm}Top", sgn(float(P[(a, b_, top)]["delta_ece_adaptive"])))
    mac(f"d{nm}First", sgn(float(P[(a, b_, first)]["delta_ece_adaptive"])))
    mac(f"d{nm}TopAbs", f3(abs(float(P[(a, b_, top)]["delta_ece_adaptive"]))))
    mac(f"d{nm}FirstAbs", f3(abs(float(P[(a, b_, first)]["delta_ece_adaptive"]))))
mac("dFXfivekAbs", f3(abs(float(P[("stgcn_fixed", "stgcn", 5000)]["delta_ece_adaptive"]))))


def pd(a, b_, n):
    return float(P[(a, b_, n)]["delta_ece_adaptive"])


mac("dAGBigMin", f3(min(pd("agcn", "stgcn", n) for n in big)))
mac("dAGBigMax", f3(max(pd("agcn", "stgcn", n) for n in big)))
mac("dCTBigMin", f3(min(pd("ctrgcn", "stgcn", n) for n in big)))
mac("dCTBigMax", f3(max(pd("ctrgcn", "stgcn", n) for n in big)))
mac("dAGSmallMin", f3(-max(pd("agcn", "stgcn", n) for n in low)))
mac("dAGSmallMax", f3(-min(pd("agcn", "stgcn", n) for n in low)))
mac("dFRBigMax", f3(max(abs(pd("agcn_frozen", "agcn", n)) for n in big)))
mac("dFRfivek", f3(pd("agcn_frozen", "agcn", 5000)))
mac("dFRfivekAcc", f3(float(P[("agcn_frozen", "agcn", 5000)]["delta_top1"])))
mac("dFXfivekAcc", f3(abs(float(P[("stgcn_fixed", "stgcn", 5000)]["delta_top1"]))))
mac("excFRvsSTBigMin", f3(min(L_("agcn_frozen", n, "excess_mean") - L_("stgcn", n, "excess_mean") for n in big)))
mac("excFRvsSTBigMax", f3(max(L_("agcn_frozen", n, "excess_mean") - L_("stgcn", n, "excess_mean") for n in big)))
_bigd = [abs(pd(a, "stgcn", n)) for a in ("agcn", "ctrgcn") for n in big]
mac("seRatioMin", f"{min(_bigd) / _se:.0f}")
mac("seRatioMax", f"{max(_bigd) / _se:.0f}")
mac("frSERatio", f"{max(abs(pd('agcn_frozen', 'agcn', n)) for n in big) / _se:.1f}")
_frst = [L_("agcn_frozen", n, "excess_mean") - L_("stgcn", n, "excess_mean") for n in big]
mac("frSTseMin", f"{min(_frst) / _se:.0f}")
mac("frSTseMax", f"{max(_frst) / _se:.0f}")

# seed-level tests: separately trained runs are the units (one Holm family)
sl = HS.seed_level_tests(ECE, [("agcn", "stgcn"), ("ctrgcn", "stgcn")], {"low": low, "high": big})
SL = {(r["model_a"], r["regime"]): r for r in sl}
for a, nm in (("agcn", "AG"), ("ctrgcn", "CT")):
    hi_, lo_ = SL[(a, "high")], SL[(a, "low")]
    mac(f"seedPos{nm}", f"{hi_['n_a_higher']}/{hi_['n_pairs']}")
    mac(f"seedNeg{nm}Low", f"{lo_['n_nonzero'] - lo_['n_a_higher']}/{lo_['n_pairs']}")
    mac(f"pSeed{nm}Low", pfmt(lo_["p_holm"]))
    mac(f"pSeed{nm}High", pfmt(hi_["p_holm"]))
mac("seedPHolm", pfmt(max(SL[(a, "high")]["p_holm"] for a in ("agcn", "ctrgcn"))))
mac("nSeedTests", len(sl))
mac("lowRungMax", thou(max(low)))
mac("highRungMin", thou(min(big)))

# ====================================================================== RQ3 (methods)
# Every method changes the confidences and with them the estimator floor, so a
# raw ECE difference between methods mixes calibration with the floor shift.
# The methods are TESTED on the excess over each method's own floor (computed on
# the method's probabilities by Stage 9c of the v6 notebook) and on the
# binning-free Brier score and NLL; the raw adaptive ECE is reported only.
mc = read_csv("methods_comparison.csv")
MN = {"raw": "uncalibrated", "temperature": "temperature", "vector": "vector scaling",
      "label_smooth": "label smoothing", "focal": "focal loss", "mc_dropout": "MC dropout",
      "deep_ensemble": "deep ensemble (5)"}
MK = {"raw": "Raw", "temperature": "TS", "vector": "VS", "label_smooth": "LS", "focal": "FL",
      "mc_dropout": "MC", "deep_ensemble": "DE"}
PN = {"temperature": "temperature scaling", "vector": "vector scaling", "label_smooth": "label smoothing",
      "focal": "focal loss", "mc_dropout": "MC dropout", "deep_ensemble": "the deep ensemble"}
ORDER = ["temperature", "vector", "label_smooth", "focal", "mc_dropout", "deep_ensemble"]
allm = {x["method"]: x for x in mc if x["scope"] == "all"}
pc_rows = read_csv("methods_per_cell.csv")
assert "floor" in pc_rows[0] and pc_rows[0]["floor"] not in ("", None), (
    "methods_per_cell.csv has no per-method floors: these results predate the v6 notebook (Stage 9c)")
for r in pc_rows:
    for c in ("ece", "brier", "nll", "floor", "excess"):
        r[c] = float(r[c])
TESTED = ["excess", "brier", "nll"]
SK = {"ece": "ECE", "excess": "Exc", "brier": "Brier", "nll": "NLL"}
RES = {key: HS.method_comparison(pc_rows, ORDER, key=key) for key in ["ece"] + TESTED}
res_ece, eci = RES["ece"]
NU = res_ece["vector"]["n"]
mac("nUnits", NU)
mac("nBootUnits", thou(HS.N_BOOT_UNITS))
mac("nMethodFamilies", len(TESTED))
for key, (res, _) in RES.items():
    for m in ORDER:
        r, k, s_ = res[m], MK[m], SK[key]
        mac(f"d{s_}{k}", sgn(r["mean"]))
        mac(f"d{s_}{k}Abs", f3(abs(r["mean"])))
        mac(f"imp{s_}{k}", f"{r['k']}/{r['n']}")
        mac(f"imp{s_}{k}PerDS", ", ".join(f"{r['per_dataset'][ds][0]}/{r['per_dataset'][ds][1]}" for ds in DS))
        if key != "ece":
            mac(f"p{s_}{k}Rel", prel(r["p_holm"]))
            mac(f"p{s_}{k}", pfmt(r["p_holm"]))
            mac(f"v{s_}{k}", r["verdict"])
# each method's floor: mean over its cells, and its mean reduction per test set
# (excess = ECE - floor, so the floor reduction is the ECE reduction minus the
# excess reduction, averaged over the same cells and test sets)
FD = {m: RES["ece"][0][m]["mean"] - RES["excess"][0][m]["mean"] for m in ORDER}
for m in ["raw"] + ORDER:
    mac(f"mfloor{MK[m]}", f3(float(allm[m]["floor"])))
for m in ORDER:
    mac(f"dFloor{MK[m]}Abs", f3(abs(FD[m])))

# The abstract, RQ3 and the conclusion state these results in words.
_V = {(key, m): RES[key][0][m]["verdict"] for key in TESTED for m in ORDER}
assert all(_V[(key, m)] == "n.s." for key in ("brier", "nll") for m in ("temperature", "vector", "mc_dropout")), _V
assert all(_V[(key, m)] == "increases" for key in ("brier", "nll") for m in ("label_smooth", "focal")), _V
assert all(_V[(key, "deep_ensemble")] == "reduces" for key in ("brier", "nll")), _V
for m in ("temperature", "vector"):      # "the lower ECE after either rescaling is accounted for by a lower floor"
    assert _V[("excess", m)] == "n.s." and RES["ece"][0][m]["mean"] > 0 and FD[m] > 0.5 * RES["ece"][0][m]["mean"], m
_ls = RES["excess"][0]["label_smooth"]   # "label smoothing raises the excess over its own floor in every test set"
assert _V[("excess", "label_smooth")] == "increases" and _ls["k"] == 0 and _ls["n"] == NU, _ls
assert abs(RES["excess"][0]["vector"]["mean"]) < 0.25 * RES["ece"][0]["vector"]["mean"]   # "its excess barely moves"
_de = RES["ece"][0]["deep_ensemble"]["mean"]   # "higher by X, of which only Y is a higher floor"
assert _de < 0 and _de < FD["deep_ensemble"] < 0 and abs(FD["deep_ensemble"]) < 0.5 * abs(_de), (_de, FD)


def _cell(r, ci=True):
    out = sgn(r["mean"]) + (f" [{sgn(r['lo'])}, {sgn(r['hi'])}]" if ci else "")
    return out


rows_t = []
for m in ORDER:
    cells = [MN[m], sgn(RES["ece"][0][m]["mean"]), f"{RES['ece'][0][m]['k']}/{RES['ece'][0][m]['n']}"]
    for key in TESTED:
        r = RES[key][0][m]
        cells += [_cell(r), f"{r['k']}/{r['n']}", pfmt(r["p_holm"])]
    rows_t.append(" & ".join(cells) + r"\\")
hdr1 = r" & \multicolumn{2}{c}{$\aece$ (descriptive)}"
hdr2 = r"Method & $\Delta$ & $k$"
cmid = r"\cmidrule(lr){2-3}"
for i, key in enumerate(TESTED):
    nm = {"excess": r"excess $\aece$", "brier": "Brier score", "nll": "NLL"}[key]
    hdr1 += r" & \multicolumn{3}{c}{" + nm + "}"
    hdr2 += r" & $\Delta$ [95\% CI] & $k$ & $p_{\mathrm{Holm}}$"
    cmid += r"\cmidrule(lr){" + f"{4 + 3 * i}-{6 + 3 * i}" + "}"
(O / "tab_methods.tex").write_text(r"""\begin{table*}[t]
\caption{Six calibration methods against the uncalibrated ST-GCN on the three
small datasets (four streams, four splits, five seeds; the deep ensemble has
one cell per dataset, stream and split and is compared with the seed-mean of
the uncalibrated cells). Every method changes the confidences and with them the
estimator floor, so a difference in raw $\aece$ mixes calibration with the
shift of the floor; it is shown for reference only. The methods are tested on
the excess adaptive ECE over each method's own matched floor, the Brier score
and the NLL. $\Delta$: mean reduction (positive is better); the
interval is a cluster bootstrap over the """ + MAC["nUnits"] + r""" test sets
(dataset $\times$ split); $k$: test sets in which the method is better; $p$:
exact sign test over test sets, Holm-corrected over the six methods for each
score. The four test sets of a dataset share subjects, so the $p$-values are
approximate.}
\label{tab:methods}
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}l""" + "cc" + "ccc" * len(TESTED) + r"""@{}}
\toprule
""" + hdr1 + r"""\\
""" + cmid + r"""
""" + hdr2 + r"""\\
\midrule
""" + "\n".join(rows_t) + r"""
\bottomrule
\end{tabular}}
\end{table*}
""")
b2 = []
for m in ["raw"] + ORDER:
    x, k = allm[m], MK[m]
    b2.append(f"{MN[m]} & {float(x['ece']):.3f} & {float(x['cw_ece']):.4f} & {float(x['brier']):.3f} & "
              f"{float(x['rel']):.3f} & {float(x['res']):.3f} & {float(x['nll']):.3f} & {float(x['top1']):.3f}\\\\")
    for key, nmk in (("ece", "ece"), ("brier", "brier"), ("nll", "nll"), ("top1", "acc"), ("rel", "rel"), ("res", "res")):
        mac(f"{nmk}{k}", f3(float(x[key])))
    mac(f"cw{k}", f"{float(x['cw_ece']):.4f}")
(O / "tab_methods_metrics.tex").write_text(r"""\begin{table}[t]
\caption{Mean scores per method over the cells of Table~\ref{tab:methods}:
adaptive and classwise ECE \eqref{eq:cwece}, Brier score with its reliability
(REL, lower is better) and resolution (RES, higher is better) terms
\eqref{eq:murphy}, NLL and top-1. The binned scores ($\aece$,
$\ece_{\mathrm{cw}}$, REL) move with each method's floor.}
\label{tab:methods_metrics}
\centering\footnotesize\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{@{}lccccccc@{}}
\toprule
Method & $\aece$ & $\ece_{\mathrm{cw}}$ & Brier & REL & RES & NLL & top-1\\
\midrule
""" + "\n".join(b2) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
grid = {m: {ds: float([y for y in mc if y["scope"] == ds and y["method"] == m][0]["ece"]) for ds in DS}
        for m in ["raw"] + ORDER}
best = {ds: min(grid[m][ds] for m in grid) for ds in DS}
lines = [MN[m] + " & " + " & ".join((f"\\textbf{{{grid[m][ds]:.3f}}}" if abs(grid[m][ds] - best[ds]) < 1e-12
                                     else f"{grid[m][ds]:.3f}") for ds in DS) + r"\\" for m in grid]
(O / "tab_methods_grid.tex").write_text(r"""\begin{table}[t]
\caption{Mean adaptive ECE per dataset and method (all streams); bold: lowest
per dataset. Descriptive only: each dataset has four test sets, and the values
include each method's shift of the floor.}
\label{tab:methods_grid}
\centering\footnotesize
\begin{tabular}{@{}lccc@{}}
\toprule
Method & MSR-Action3D & DailyAct3D & UTKinect\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
for ds in DS:
    x = [y for y in mc if y["scope"] == ds and y["method"] == "deep_ensemble"][0]
    mac(f"dDE{DSN[ds]}", f3(abs(float(x["delta"]))))
# Fig. 4 data: excess, Brier and NLL panels
_fig = {"n_units": NU, "panels": []}
for key in TESTED:
    res, _ = RES[key]
    _fig["panels"].append({"key": key, "tested": True, "methods": {
        m: dict(mean=res[m]["mean"], lo=res[m]["lo"], hi=res[m]["hi"], k=res[m]["k"], n=res[m]["n"],
                p_holm=res[m].get("p_holm"), verdict=res[m]["verdict"])
        for m in ORDER}})
(O / "methods_split_level.json").write_text(json.dumps(_fig, indent=1))


def _holm_txt(p):
    return r"$p_{\mathrm{Holm}}$" + prel(p)


def _join(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _cap(t):
    return t[0].upper() + t[1:]


# The excess, in words: significant changes (sign test, Holm), then changes
# whose cluster-bootstrap interval excludes zero although the sign test does
# not, then the rest.
rs = RES["excess"][0]
_sig = [m for m in ORDER if rs[m]["p_holm"] < 0.05]
_ci = [m for m in ORDER if m not in _sig and (rs[m]["lo"] > 0 or rs[m]["hi"] < 0)]
_rest = [m for m in ORDER if m not in _sig and m not in _ci]
_parts = []
for direction, word in ((1, "lowered"), (-1, "raised")):
    hit = [f"{PN[m]} ({rs[m]['k']}/{rs[m]['n']} test sets better, {_holm_txt(rs[m]['p_holm'])})"
           for m in _sig if np.sign(rs[m]["mean"]) == direction]
    if hit:
        _parts.append(f"{word} by {_join(hit)}")
_txt = "Measured against each method's own floor, the excess adaptive ECE is " + " and ".join(_parts) + "." \
    if _parts else "Measured against each method's own floor, no method changes the excess adaptive ECE significantly."
for direction, word in ((1, "lower"), (-1, "raise")):
    hit = [m for m in _ci if np.sign(rs[m]["mean"]) == direction]
    if hit:
        moved = [str(rs[m]["n"] - rs[m]["k"] if direction < 0 else rs[m]["k"]) + "/" + str(rs[m]["n"]) for m in hit]
        _txt += (f" {_cap(_join([PN[m] for m in hit]))} {word}{'s' if len(hit) == 1 else ''} it on average as well "
                 f"(interval{'s' if len(hit) > 1 else ''} excluding zero), but in only {_join(moved)} test sets, "
                 f"which is not significant after correction.")
if _rest and (_parts or _ci):
    _txt += f" {_cap(_join([PN[m] for m in _rest]))} {'does' if len(_rest) == 1 else 'do'} not change it significantly."
mac("methodsExcessSentence", _txt)

# The floors, in words: how far each method moves its own floor
_fl = {m: float(allm[m]["floor"]) for m in ["raw"] + ORDER}
assert _fl["temperature"] < _fl["raw"] and _fl["vector"] < _fl["raw"], _fl          # the rescalings lower it
assert _fl["label_smooth"] > _fl["raw"] and _fl["focal"] > _fl["raw"], _fl          # the losses raise it
_mv = sorted(ORDER, key=lambda m: abs(_fl[m] - _fl["raw"]))
assert set(_mv[:2]) == {"mc_dropout", "deep_ensemble"}, _mv                         # these move it least
mac("methodsFloorSentence",
    f"The floors move with the confidences: the two rescalings lower the mean floor from {f3(_fl['raw'])} to "
    f"{f3(_fl['temperature'])} (temperature) and {f3(_fl['vector'])} (vector scaling), label smoothing and focal "
    f"loss raise it to {f3(_fl['label_smooth'])} and {f3(_fl['focal'])}, and MC dropout and the ensemble move it "
    f"least ({f3(_fl['mc_dropout'])} and {f3(_fl['deep_ensemble'])}).")

mcp = Counter(r["mc_p"] for r in small)
mac("nMCsmallest", mcp[min(mcp)])
mac("nMCother", sum(v for p_, v in mcp.items() if p_ != min(mcp)))
mac("mcPmin", f"{min(mcp):.2f}")
mac("mcPrestMin", f"{min(p_ for p_ in mcp if p_ != min(mcp)):.1f}")
mac("mcPrestMax", f"{max(mcp):.1f}")

# stream fusion: the single joint stream against the four-stream logit average,
# each against its own floor, before and after a temperature fitted to it
fus = read_csv("stream_fusion.csv")
assert "floor" in fus[0] and fus[0]["floor"] not in ("", None), (
    "stream_fusion.csv has no floors: these results predate the v6 notebook (Stage 9)")
FU = {}
for ds in DS:
    for fu, nm in (("joint", "J"), ("4-stream", "F")):
        s = [x for x in fus if x["dataset"] == ds and x["fusion"] == fu]
        g = {c: float(np.mean([float(x[c]) for x in s])) for c in
             ("top1", "ece_adaptive", "ece_adaptive_ts", "floor", "floor_ts", "brier", "brier_ts", "nll", "nll_ts")}
        g["exc"], g["excts"] = g["ece_adaptive"] - g["floor"], g["ece_adaptive_ts"] - g["floor_ts"]
        FU[(ds, nm)] = g
        mac(f"fus{DSN[ds]}{nm}acc", f3(g["top1"]))
        mac(f"fus{DSN[ds]}{nm}ece", f3(g["ece_adaptive"]))
        mac(f"fus{DSN[ds]}{nm}ts", f3(g["ece_adaptive_ts"]))
        mac(f"fus{DSN[ds]}{nm}exc", neg(f3(g["exc"])))
        mac(f"fus{DSN[ds]}{nm}excts", neg(f3(g["excts"])))
        mac(f"fus{DSN[ds]}{nm}floor", f3(g["floor"]))
        mac(f"fus{DSN[ds]}{nm}floorts", f3(g["floor_ts"]))
        for c in ("brier", "nll"):
            mac(f"fus{DSN[ds]}{nm}{c}", f3(g[c]))
            mac(f"fus{DSN[ds]}{nm}{c}ts", f3(g[c + "_ts"]))
_f4 = [x for x in fus if x["fusion"] == "4-stream"]
mac("nFusCells", len(_f4))
mac("nFusUnder", sum(float(x["temperature"]) < 1 for x in _f4))
# The fusion paragraph states these in words.
assert sum(float(x["temperature"]) < 1 for x in _f4) >= 0.9 * len(_f4)            # fusions are under-confident
for ds in DS:
    J, F = FU[(ds, "J")], FU[(ds, "F")]
    assert F["brier_ts"] < F["brier"] and F["nll_ts"] < F["nll"], ds              # its temperature helps both scores
for ds in ("action3d", "daily"):
    J, F = FU[(ds, "J")], FU[(ds, "F")]
    assert F["exc"] > J["exc"] and F["excts"] < 0.5 * F["exc"], ds                 # fusion raises it, TS removes most
J, F = FU[("utkinect", "J")], FU[("utkinect", "F")]
assert F["exc"] < J["exc"] and F["excts"] > F["exc"], "utkinect"                    # fusion lowers it; TS raises it
assert F["floor_ts"] < F["floor"] and F["ece_adaptive_ts"] < F["ece_adaptive"], "utkinect"

# RQ3: how much of temperature scaling's gain is the floor moving (Stage 9b)
_tf = {x["dataset"]: x for x in read_csv("temperature_floor.csv")}
_xr, _xt, _fd = [], [], []
for ds in DS:
    sub = [r for r in small if r["dataset"] == ds]
    fr, ft = float(_tf[ds]["floor_raw"]), float(_tf[ds]["floor_ts"])
    mac(f"tfRaw{DSN[ds]}", f3(fr))
    mac(f"tfTS{DSN[ds]}", f3(ft))
    _xr.append(np.mean([r["ece_adaptive"] for r in sub]) - fr)
    _xt.append(np.mean([r["ece_adaptive_temp"] for r in sub]) - ft)
    _fd.append(fr - ft)
    mac(f"excRaw{DSN[ds]}", f3(_xr[-1]))
    mac(f"excTS{DSN[ds]}", f3(_xt[-1]))
_nsf = {int(float(x["n_sim"])) for x in _tf.values()}
assert len(_nsf) == 1, _nsf
mac("nSimFloor", thou(_nsf.pop()))      # null draws for the floor of a recalibrated, retrained or fused model
mac("tfDropMean", f3(np.mean(_fd)))
mac("excRawMean", f3(np.mean(_xr)))
mac("excTSMean", f3(np.mean(_xt)))
mac("eceTSDropMean", f3(np.mean(_xr) + np.mean(_fd) - np.mean(_xt)))
assert np.mean(_xt) >= np.mean(_xr), ("the abstract, RQ3 and the conclusion say the lower ECE after temperature "
                                      "scaling is accounted for by the floor; on these results it is not")

# ====================================================================== preprocessing
pc = json.loads((R / "prep_config.json").read_text())
lab = {"A legacy": "A & legacy & joint 0, $t_0$ & --", "B topology only": "B & corrected & joint 0, $t_0$ & --",
       "C frame0": "C & corrected & hip, $t_0$ & off", "D frame0+align": "D & corrected & hip, $t_0$ & on",
       "E perframe+align": "E & corrected & hip, all $t$ & on"}
lines = []
for k, v in pc["summary"].items():
    lead, rest = lab[k].split(" & ", 1)
    lines.append(lead + ("$^{\\star}$" if k == pc["chosen"] else "") + " & " + rest +
                 f" & {v['val_top1']:.3f} & {v['top1']:.3f} & {v['ece']:.3f} & {v['T']:.2f}\\\\")
(O / "tab_prep.tex").write_text(r"""\begin{table}[t]
\caption{Preprocessing ablation on MSR-Action3D (joint stream, ST-GCN, three
splits $\times$ three seeds). Selection used validation top-1 among
corrected-topology settings whose normalization is applied before the streams
are derived (C--E); $^{\star}$: chosen. Test columns are reported only.
$t_0$: first frame.}
\label{tab:prep}
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}llllcccc@{}}
\toprule
 & tree & center & align & val & test & $\aece$ & $T$\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
S_ = pc["summary"]
mac("prepValLegacy", f3(S_["A legacy"]["val_top1"]))
mac("prepValChosen", f3(S_[pc["chosen"]]["val_top1"]))
mac("prepTestLegacy", f3(S_["A legacy"]["top1"]))
mac("prepTestChosen", f3(S_[pc["chosen"]]["top1"]))
mac("prepTmin", f2(min(v["T"] for v in S_.values())))
mac("prepTmax", f2(max(v["T"] for v in S_.values())))

# ====================================================================== environment
# The experiments ran in the first sessions; a later statistics-only session
# (possibly on a CPU runtime) must not change the stack the paper reports.
env = json.loads((R / "environment" / "environment.json").read_text())
hist_f = R / "environment" / "environment_history.json"
hist = json.loads(hist_f.read_text()) if hist_f.exists() else [dict(env, session="current")]
_by = {h.get("session"): h for h in hist}
first = _by.get("v1", hist[0])
second = _by.get("v2", hist[min(1, len(hist) - 1)])
mac("envPython", first["python"])
mac("envTorch", first["packages"]["torch"].split("+")[0])
mac("envNumpy", first["packages"]["numpy"])
mac("envGPU", next((h["cuda"]["device"] for h in hist if h.get("cuda") and h["cuda"].get("device")), "GPU"))
mac("envCudaFirst", (first.get("cuda") or {}).get("version") or "n/a")
mac("envCudaLast", (second.get("cuda") or {}).get("version") or "n/a")
mac("nSessions", len(hist))

# p-values with their relation, for prose (avoids "p = <0.001")
for _nm, _p in (("pTestSetsPos", pe["p"]), ("stTopNullP", max(r["null_p_value"] for r in ntu if r["model"] == "stgcn" and r["n_train"] == top)),
                ("WAp", M[("action3d", "bone_motion", "fixed_frame0", 3)]["null_p_value"]),
                ("WBp", M[("daily", "bone", "rand0_frame0", 1)]["null_p_value"]),
                ("seedPHolm", max(SL[(a, "high")]["p_holm"] for a in ("agcn", "ctrgcn"))),
                ("pSeedAGLow", SL[("agcn", "low")]["p_holm"]), ("pSeedCTLow", SL[("ctrgcn", "low")]["p_holm"]),
                ("pSeedAGHigh", SL[("agcn", "high")]["p_holm"]), ("pSeedCTHigh", SL[("ctrgcn", "high")]["p_holm"])):
    mac(_nm + "Rel", prel(_p))
(O / "numbers.tex").write_text("% generated by make_numbers.py -- do not edit\n" +
                               "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(MAC.items())) + "\n")
print(f"{len(MAC)} macros; tables:", sorted(p.name for p in O.glob("tab_*.tex")))
