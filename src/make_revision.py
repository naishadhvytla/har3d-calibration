#!/usr/bin/env python3
"""
make_revision.py TABLES_DIR GENERATED_DIR FIG_DIR

Tables and figures added in the revision (v7), from the CSV tables that Stages
9d-9h of the notebook write to results/tables:
  generated/tab_estimators.tex      estimator_sensitivity.csv
  generated/tab_dependence.tex      pooled_dependence.csv
  generated/tab_power.tex           power_summary.csv, power_pooled.csv (or its summary)
  generated/tab_prepsens.tex        prep_sensitivity.csv
  generated/tab_replication.tex     replication_cells.csv
  generated/tab_methods_subject.tex methods_subject_level.csv
  fig5_perclass.pdf                 per_class_null.csv   (replaces the v6 figure)
  fig7_power.pdf                    power_curves.csv, power_synthetic.csv, power_summary.csv
  fig8_steps.pdf                    replication_checkpoint_means.csv, replication_cells.csv
"""
import csv, math, sys
from itertools import product
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import har3d_plotstyle as PS  # noqa: E402

T, G, O = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
G.mkdir(parents=True, exist_ok=True); O.mkdir(parents=True, exist_ok=True)
PS.apply()
LABEL = {"action3d": "MSR-Action3D", "daily": "DailyActivity3D", "utkinect": "UTKinect", "all": "all three"}
SMALL = ["action3d", "daily", "utkinect"]
NAME = {"stgcn": "ST-GCN", "agcn": "2s-AGCN", "ctrgcn": "CTR-GCN"}
MCOL = {"stgcn": PS.PALETTE["blue"], "agcn": PS.PALETTE["vermillion"], "ctrgcn": PS.PALETTE["green"]}


def rd(name):
    return list(csv.DictReader(open(T / name)))


def sg(x, d=3):
    """signed number for a table"""
    x = float(x)
    if round(x, d) == 0:
        return f"{0:.{d}f}"
    return (r"$+$" if x > 0 else r"$-$") + f"{abs(x):.{d}f}"


def pv(p):
    p = float(p)
    return r"$<$0.001" if p < 0.001 else f"{p:.3f}"


def write(name, text):
    (G / name).write_text(text)
    print("wrote", G / name)


# ------------------------------------------------------------ estimators
est = rd("estimator_sensitivity.csv")
ELAB = {"aece5": r"adaptive ECE, $M{=}5$", "aece10": r"adaptive ECE, $M{=}10$",
        "aece15": r"adaptive ECE, $M{=}15$ (main)", "aece20": r"adaptive ECE, $M{=}20$",
        "aece30": r"adaptive ECE, $M{=}30$", "ece15": r"equal-width ECE, $M{=}15$",
        "ce2db15": r"debiased squared CE, $M{=}15$", "ks": "KS calibration error"}
body = ""
for r in est:
    body += (f"{ELAB[r['estimator']]} & {float(r['mean_observed']):.3f} & {(0.0 if round(float(r['mean_floor']), 3) == 0 else float(r['mean_floor'])):.3f} & "
             f"{r['cells_p_lt_0.05']} & {r['cells_separable_holm']} & {float(r['pooled_excess']):.3f} & "
             f"{r['test_sets_positive']} & {pv(r['p_coupled'])} & {pv(r['p_clustered'])} & "
             f"{pv(r['p_coupled_action3d'])} & {pv(r['p_coupled_daily'])} & {pv(r['p_coupled_utkinect'])}\\\\\n")
write("tab_estimators.tex", r"""\begin{table*}[t]
\caption{Sensitivity of RQ1 to the calibration estimator and the number of
bins. Every estimator is tested against its own matched null on the same 240
ST-GCN cells. Observed and floor: means over cells. Separable: cells with
$p<0.05$ before and after Holm correction over the 240 cells. Pooled excess:
mean over the 12 test sets of the within-test-set mean excess; $k$: test sets
in which it is positive. $p$-values of the pooled excess are Monte-Carlo
values under the clip-coupled null, all datasets and per dataset, and under
the subject-clustered null (Sec.~\ref{sec:dependence}); $<$0.001 is the
smallest value attainable with $5{,}000$ draws.}
\label{tab:estimators}
\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lccccccccccc@{}}
\toprule
 & \multicolumn{2}{c}{mean over cells} & \multicolumn{2}{c}{cells separable (of 240)} & \multicolumn{4}{c}{pooled over test sets} & \multicolumn{3}{c}{$p$ coupled, per dataset}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-9}\cmidrule(lr){10-12}
Estimator & observed & floor & $p<0.05$ & Holm & excess & $k$ & $p$ coupled & $p$ clustered & Action3D & Daily & UTKinect\\
\midrule
""" + body + r"""\bottomrule
\end{tabular}
\end{table*}
""")

# ------------------------------------------------------------ dependence
dep = [r for r in rd("pooled_dependence.csv") if r["estimator"] == "aece15"]
NULLS = [("independent", "0", "independent cells"), ("coupled", "0", "clip-coupled"),
         ("clustered", "estimated", r"subject-clustered, $\hat\tau$")] + \
        [("clustered, fixed tau", t, rf"subject-clustered, $\tau={t}$") for t in ("0.02", "0.04", "0.06", "0.08", "0.1", "0.15")]
body = ""
for null, tau, lab in NULLS:
    r = [x for x in dep if x["null"] == null and x["tau"] == tau][0]
    body += (f"{lab} & {float(r['mean_excess']):.4f} & {float(r['null_sd']):.4f} & {float(r['variance_inflation']):.1f} & "
             f"{pv(r['p'])} & {r['test_sets_positive']}/12")
    for ds in SMALL:
        body += f" & {float(r[ds + '_mean_excess']):.3f} & {pv(r[ds + '_p'])} & {r[ds + '_test_sets_positive']}"
    body += "\\\\\n"
    if null == "clustered":
        body += "\\midrule\n"
write("tab_dependence.tex", r"""\begin{table*}[t]
\caption{The pooled excess adaptive ECE of the 240 small-dataset cells under
null models of increasing dependence (Sec.~\ref{sec:dependence}). Independent
cells: every cell drawn on its own, the implicit assumption of a test that
treats cells or test sets as independent. Clip-coupled: one draw per clip
decides every prediction of that clip. Subject-clustered: clip-coupled plus a
subject effect of standard deviation $\tau$ on the probability of a correct
prediction; $\hat\tau$ is the per-dataset estimate ($0.087$, $0$ and $0.051$
for MSR-Action3D, DailyActivity3D and UTKinect), the rows below fix $\tau$ for
every dataset. Excess: mean over test sets of the within-test-set mean excess
over the floor of that null; s.d.: standard deviation of that statistic under
the null; var.: its variance relative to independent cells; $k$: test sets
with positive excess. $p$: one-sided Monte-Carlo value ($5{,}000$ draws for
the first three rows, $2{,}000$ for the others).}
\label{tab:dependence}
\centering\footnotesize\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}lccccc@{\hskip 10pt}ccc@{\hskip 10pt}ccc@{\hskip 10pt}ccc@{}}
\toprule
 & \multicolumn{5}{c}{all three datasets (12 test sets)} & \multicolumn{3}{c}{MSR-Action3D} & \multicolumn{3}{c}{DailyActivity3D} & \multicolumn{3}{c}{UTKinect}\\
\cmidrule(r){2-6}\cmidrule(r){7-9}\cmidrule(r){10-12}\cmidrule{13-15}
Null model & excess & s.d. & var. & $p$ & $k$ & excess & $p$ & $k$ & excess & $p$ & $k$ & excess & $p$ & $k$\\
\midrule
""" + body + r"""\bottomrule
\end{tabular}
\end{table*}
""")

# ------------------------------------------------------------ power table
ps = {r["dataset"]: r for r in rd("power_summary.csv")}
pp = {r["scope"]: r for r in rd("power_pooled_summary.csv")} if (T / "power_pooled_summary.csv").exists() else None
if pp is None:                                       # from the full table of the notebook
    pp = {}
    for r in rd("power_pooled.csv"):
        pp.setdefault(r["scope"], {"mde_over": r["mde_over"], "mde_under": r["mde_under"]})
NTEST = {"action3d": "212--270", "daily": "128--160", "utkinect": "79--99"}


def f3(x):
    x = float(x)
    return "--" if not math.isfinite(x) else f"{x:.3f}"


body = ""
for ds in SMALL:
    r = ps[ds]
    body += (f"\\quad {LABEL[ds]} & {NTEST[ds]} & {f3(r['median_mde_over_05'])} & {f3(r['median_mde_under_05'])} & "
             f"{f3(r['median_mde_over_holm'])} & {f3(r['median_mde_over_05_ks'])} & "
             f"{float(r['median_power_05_eps0.05']):.2f} & {float(r['median_power_05_eps0.1']):.2f}\\\\\n")
body = "\\multicolumn{8}{@{}l}{\\emph{One cell}}\\\\\n" + body
body += "\\midrule\n\\multicolumn{8}{@{}l}{\\emph{Pooled over test sets}}\\\\\n"
for sc, lab in (("action3d", "MSR-Action3D (4)"), ("daily", "DailyActivity3D (4)"),
                ("utkinect", "UTKinect (4)"), ("all", "all three (12)")):
    body += f"\\quad {lab} & & {f3(pp[sc]['mde_over'])} & {f3(pp[sc]['mde_under'])} & & & &\\\\\n"
write("tab_power.tex", r"""\begin{table}[t]
\caption{Minimum detectable calibration error (MDE): the true top-label
calibration error $\varepsilon$ at which the matched-null test on adaptive ECE
reaches $80\%$ power, for over- and under-confident miscalibration at
$\alpha{=}0.05$, at the Holm level $0.05/240$ (over-confident), and for the KS
statistic at $\alpha{=}0.05$ (over-confident). Per-cell rows are medians over
the 80 cells of a dataset with their own confidence profiles; the last two
columns give the power at $\varepsilon{=}0.05$ and $0.10$, and $n$ is the
range of test-set sizes. Pooled rows: the test on the mean excess over the
test sets of a dataset (their number in parentheses) under the clip-coupled
null, against a miscalibration common to all cells.}
\label{tab:power}
\centering\footnotesize\setlength{\tabcolsep}{2pt}
\begin{tabular}{@{}lccccccc@{}}
\toprule
 & & \multicolumn{2}{c}{MDE, $\alpha{=}0.05$} & MDE & MDE & \multicolumn{2}{c}{power at $\varepsilon$}\\
\cmidrule(lr){3-4}\cmidrule(l){7-8}
Test & $n$ & over & under & Holm & KS & 0.05 & 0.10\\
\midrule
""" + body + r"""\bottomrule
\end{tabular}
\end{table}
""")

# ------------------------------------------------------------ preprocessing sensitivity
prs = rd("prep_sensitivity.csv")
PLAB = {"frame0 (main)": "C (main)", "frame0_align": "D", "perframe_align": "E", "legacy_centring": "B",
        "legacy": "A"}
SHORT = {"action3d": "Action3D", "daily": "Daily", "utkinect": "UTKinect"}
body, last = "", None
for r in prs:
    if last is not None and r["preprocessing"] != last:
        body += "\\addlinespace[2pt]\n"
    last = r["preprocessing"]
    body += (f"{PLAB[r['preprocessing']]} & {SHORT[r['dataset']]} & {float(r['top1']):.3f} & {float(r['ece_adaptive']):.3f} & "
             f"{float(r['floor']):.3f} & {float(r['pooled_excess']):.3f} & {pv(r['p_coupled'])} & "
             f"{r['cells_separable_holm']}/{r['n_cells']} & {float(r['temperature']):.2f}\\\\\n")
write("tab_prepsens.tex", r"""\begin{table}[t]
\caption{Calibration conclusions under alternative preprocessing (ST-GCN,
joint stream, the four test sets of each dataset, seeds 0--2; 12 cells per
row). Settings as in Table~\ref{tab:prep}: C, hip-centered at frame 0 (the
main experiment, restricted to the same stream and seeds); D, C with rotation
alignment; E, hip-centered per frame, aligned; B, legacy centering; A, legacy
tree and centering (A and B can only be applied to MSR-Action3D). Daily:
MSR-DailyActivity3D.
Excess: pooled over the four test sets, with its $p$-value under the
clip-coupled null. Sep.: cells separable after Holm correction within the row.
$T$: mean fitted temperature.}
\label{tab:prepsens}
\centering\footnotesize\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{@{}llccccccc@{}}
\toprule
Setting & Dataset & top-1 & $\aece$ & floor & excess & $p$ & sep. & $T$\\
\midrule
""" + body + r"""\bottomrule
\end{tabular}
\end{table}
""")


# ------------------------------------------------------------ replication (seed level)
def seed_summary(v):
    v = np.asarray(v, float); n = len(v); m = v.mean()
    sd = v.std(ddof=1) if n > 1 else float("nan")
    half = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
    obs = abs(v.sum())
    flips = [abs(sum(s * x for s, x in zip(sg_, v))) for sg_ in product((1, -1), repeat=n)]
    return dict(n=n, mean=m, lo=m - half, hi=m + half, n_pos=int((v > 0).sum()),
                p_flip=float(np.mean([f >= obs - 1e-15 for f in flips])),
                p_t=float(stats.ttest_1samp(v, 0).pvalue) if n > 1 else float("nan"))


rc = rd("replication_cells.csv")
EX = {(r["backbone"], int(r["n_train"]), int(r["seed"])): float(r["excess"]) for r in rc}
ORIG = {int(r["seed"]): r["origin"] for r in rc}
RUNGS = sorted({k[1] for k in EX}); SEEDS = sorted(ORIG)
body = ""
for s in SEEDS:
    m = {b: np.mean([EX[(b, n, s)] for n in RUNGS]) for b in NAME}
    body += (f"{s} & {'original' if ORIG[s] == 'original' else 'new'} & {m['stgcn']:.3f} & {m['agcn']:.3f} & {m['ctrgcn']:.3f} & "
             f"{sg(m['agcn'] - m['stgcn'])} & {sg(m['ctrgcn'] - m['stgcn'])} & {sg(m['ctrgcn'] - m['agcn'])}\\\\\n")
    if s == max(q for q in SEEDS if ORIG[q] == "original"):
        body += "\\midrule\n"
body += "\\midrule\n"
SUM = {}
for lab, seeds in (("original seeds (exploratory)", [s for s in SEEDS if ORIG[s] == "original"]),
                   ("new seeds (confirmatory)", [s for s in SEEDS if ORIG[s] != "original"]),
                   ("all seeds", SEEDS)):
    cells = []
    for a, b in (("agcn", "stgcn"), ("ctrgcn", "stgcn"), ("ctrgcn", "agcn")):
        ss = seed_summary([np.mean([EX[(a, n, s)] - EX[(b, n, s)] for n in RUNGS]) for s in seeds])
        SUM[(lab, a, b)] = ss
        cells.append(f"{sg(ss['mean'])} [{sg(ss['lo'])}, {sg(ss['hi'])}]")
    body += f"\\multicolumn{{5}}{{@{{}}l}}{{mean over {lab}, 95\\% $t$-interval}} & " + " & ".join(cells) + "\\\\\n"
    cells = [f"{SUM[(lab, a, b)]['n_pos']}/{SUM[(lab, a, b)]['n']}; {SUM[(lab, a, b)]['p_flip']:.3f}"
             for a, b in (("agcn", "stgcn"), ("ctrgcn", "stgcn"), ("ctrgcn", "agcn"))]
    body += "\\multicolumn{5}{@{}l}{\\quad seeds positive; exact sign-flip $p$} & " + " & ".join(cells) + "\\\\\n"
write("tab_replication.tex", r"""\begin{table*}[t]
\caption{Seed-level comparison of the three graph layers at the large rungs.
For each seed, the excess adaptive ECE averaged over the rungs of $10{,}000$,
$20{,}000$ and $32{,}202$ training clips, and the paired differences. Seeds
0--2 are the first-version runs, on which the $10{,}000$-clip boundary was
chosen (exploratory); seeds 3 and 4 were trained afterward in one software
environment under the hypotheses registered before training (confirmatory).
The seed is the unit: a $t$-interval and an exact two-sided sign-flip test
over seeds, whose smallest attainable $p$-value is $0.25$, $0.5$ and $0.0625$
for three, two and five seeds.}
\label{tab:replication}
\centering\footnotesize
\begin{tabular}{@{}llcccccc@{}}
\toprule
 & & \multicolumn{3}{c}{excess $\aece$, mean of three rungs} & \multicolumn{3}{c}{paired difference}\\
\cmidrule(lr){3-5}\cmidrule(l){6-8}
Seed & Runs & ST-GCN & 2s-AGCN & CTR-GCN & 2s-AGCN $-$ ST-GCN & CTR-GCN $-$ ST-GCN & CTR-GCN $-$ 2s-AGCN\\
\midrule
""" + body + r"""\bottomrule
\end{tabular}
\end{table*}
""")
for k, v in SUM.items():
    print(k, {q: (round(w, 4) if isinstance(w, float) else w) for q, w in v.items()})

# ------------------------------------------------------------ methods, subject level
ms = rd("methods_subject_level.csv")
M = {(r["method"], r["score"]): r for r in ms}
MLAB = [("temperature", "temperature"), ("vector", "vector scaling"), ("label_smooth", "label smoothing"),
        ("focal", "focal loss"), ("mc_dropout", "MC dropout"), ("deep_ensemble", "deep ensemble (5)"),
        None,
        ("temperature_xfit", "temperature, cross-fitted"), ("vector_xfit", "vector scaling, cross-fitted"),
        None,
        ("temperature_oracle", "temperature, fitted on test"), ("vector_oracle", "vector scaling, fitted on test")]


def ph(p):
    p = float(p)
    return r"$<$0.001" if p < 0.001 else f"{p:.3f}"


body = ""
for it in MLAB:
    if it is None:
        body += "\\midrule\n"; continue
    m, lab = it
    b, l, e = M[(m, "brier")], M[(m, "nll")], M[(m, "excess")]
    body += (f"{lab} & {sg(b['mean_reduction'], 4)} [{sg(b['ci_lo'], 4)}, {sg(b['ci_hi'], 4)}] & {b['n_improved']}/27 & {ph(b['p_holm'])} & "
             f"{sg(l['mean_reduction'])} [{sg(l['ci_lo'])}, {sg(l['ci_hi'])}] & {l['n_improved']}/27 & {ph(l['p_holm'])} & "
             f"{float(e['mean_score']):.3f} & {e['n_improved']}/12\\\\\n")
raw = {s: float(M[("raw", s)]["mean_score"]) for s in ("brier", "nll", "excess")}
write("tab_methods_subject.tex", r"""\begin{table*}[t]
\caption{Calibration methods with human subjects as the units, and the effect
of the calibration set. For each of the 27 subjects that appear in a test set
(nine per dataset), the paired reduction of the per-clip Brier score and NLL
relative to the uncalibrated model is averaged over every prediction made for
that subject. $\Delta$: mean over subjects (positive is better) with a
$95\%$ bootstrap interval over subjects resampled within datasets; $k$:
subjects for whom the method is better; $p$: exact sign test over subjects,
Holm-corrected within each block. Cross-fitted: for each test subject, the
scaling is fitted on the validation subjects and the other test subjects (five
subjects in all) and applied to the held-out subject. Fitted on test: fitted
on the test set it is evaluated on, an optimistic upper bound for the method
family and not a usable procedure. Excess: mean excess adaptive ECE over the
method's own floor (uncalibrated: """ + f"{raw['excess']:.3f}" + r"""); $k$: test sets in which it is lower than
for the uncalibrated model. Uncalibrated Brier score """ + f"{raw['brier']:.3f}" + r""", NLL """ + f"{raw['nll']:.3f}" + r""".}
\label{tab:methods_subject}
\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lcccccccc@{}}
\toprule
 & \multicolumn{3}{c}{Brier score (27 subjects)} & \multicolumn{3}{c}{NLL (27 subjects)} & \multicolumn{2}{c}{excess $\aece$}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-9}
Method & $\Delta$ [95\% CI] & $k$ & $p_{\mathrm{Holm}}$ & $\Delta$ [95\% CI] & $k$ & $p_{\mathrm{Holm}}$ & mean & $k$\\
\midrule
""" + body + r"""\bottomrule
\end{tabular}
\end{table*}
""")

# ------------------------------------------------------------ fig 5 (per-class, with null)
pc = rd("per_class_null.csv")
fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.5))
for ax, ds in zip(axes, SMALL):
    c = sorted((x for x in pc if x["dataset"] == ds), key=lambda x: -float(x["per_class_ece"]))
    x = np.arange(len(c))
    e = np.array([float(q["per_class_ece"]) for q in c]); fl = np.array([float(q["floor"]) for q in c])
    p95 = np.array([float(q["null_p95"]) for q in c]); r1 = np.array([float(q["one_minus_recall"]) for q in c])
    sig = np.array([float(q["p_holm"]) < 0.05 for q in c])
    ax.bar(x, e, color=PS.ROLE[ds], width=0.8, label="per-class ECE", zorder=2)
    ax.vlines(x, fl, p95, color="0.25", lw=0.8, zorder=3)
    ax.plot(x, fl, "_", color="0.25", ms=5, mew=1.0, zorder=3, label="null: mean to 95th pct.")
    ax.plot(x, r1, "k.", ms=3, zorder=4, label="1 − class recall")
    ax.plot(x[sig], e[sig] + 0.035, "*", color=PS.PALETTE["vermillion"], ms=6, zorder=5,
            label="above null (Holm)")
    ax.set_title(LABEL[ds], fontsize=8); ax.set_xlabel("class (sorted by ECE)")
    ax.set_xticks([]); ax.set_ylim(0, 1)
axes[0].set_ylabel("per-class ECE / 1 − recall"); axes[2].legend(fontsize=6, loc="upper right")
PS.save(fig, str(O / "fig5_perclass")); plt.close(fig)

# ------------------------------------------------------------ fig 7 (power)
pcv = rd("power_curves.csv"); syn = rd("power_synthetic.csv")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.75))
for ds in SMALL:
    r = sorted((x for x in pcv if x["dataset"] == ds and float(x["b"]) >= 0), key=lambda x: float(x["eps"]))
    eps = [float(x["eps"]) for x in r]
    a1.plot(eps, [float(x["power_0.05"]) for x in r], "-", color=PS.ROLE[ds], label=LABEL[ds])
    a1.plot(eps, [float(x["power_holm240"]) for x in r], "--", color=PS.ROLE[ds], lw=1.0)
a1.axhline(0.8, color="0.5", lw=0.6, ls=":"); a1.axhline(0.05, color="0.5", lw=0.6, ls=":")
a1.set_xlim(0, 0.44); a1.set_ylim(0, 1.02)
a1.set_xlabel(r"true calibration error $\varepsilon$ (over-confident)"); a1.set_ylabel("power of the matched-null test")
a1.plot([], [], "-", color="0.3", label=r"$\alpha=0.05$"); a1.plot([], [], "--", color="0.3", lw=1.0, label=r"Holm level, $0.05/240$")
a1.legend(loc="center right", fontsize=6.5); a1.set_title("(a) one experiment, real confidence profiles", fontsize=8)
s20 = [x for x in syn if int(x["K"]) == 20 and x["mde_over_05_ks"] not in ("", None)]
for mc, ls in ((0.5, "-"), (0.7, "--"), (0.9, ":")):
    r = sorted((x for x in s20 if abs(float(x["mean_conf"]) - mc) < 1e-9), key=lambda x: int(x["n"]))
    n = [int(x["n"]) for x in r]
    a2.plot(n, [float(x["mde_over_05"]) for x in r], ls, color=PS.PALETTE["blue"], marker="o", ms=2.5)
    a2.plot(n, [float(x["mde_over_05_ks"]) for x in r], ls, color=PS.PALETTE["vermillion"], marker="s", ms=2.5)
    a2.plot([], [], ls, color="0.3", label=f"mean confidence {mc}")
psum = {r["dataset"]: r for r in rd("power_summary.csv")}
NMID = {"action3d": 250, "daily": 148, "utkinect": 92}
for ds in SMALL:
    a2.plot(NMID[ds], float(psum[ds]["median_mde_over_05"]), "D", color=PS.ROLE[ds], ms=5, mec="k", mew=0.5, zorder=5)
a2.plot([], [], "-o", color=PS.PALETTE["blue"], ms=2.5, label="adaptive ECE, 15 bins")
a2.plot([], [], "-s", color=PS.PALETTE["vermillion"], ms=2.5, label="KS calibration error")
a2.plot([], [], "D", color="0.7", mec="k", mew=0.5, ms=5, label="real cells, adaptive ECE (median)")
a2.set_xscale("log"); a2.set_yscale("log")
a2.set_xlabel(r"test-set size $n$"); a2.set_ylabel(r"minimum detectable error (80% power)")
a2.set_ylim(0.005, 0.9)
a2.set_yticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5]); a2.set_yticklabels(["0.01", "0.02", "0.05", "0.1", "0.2", "0.5"])
a2.set_xticks([50, 100, 250, 500, 1000, 2000, 5000, 16487]); a2.set_xticklabels(["50", "100", "250", "500", "1k", "2k", "5k", "16k"])
a2.minorticks_off()
a2.legend(fontsize=6, ncol=1, loc="upper right"); a2.set_title(r"(b) synthetic profiles, $K=20$, $\alpha=0.05$", fontsize=8)
PS.save(fig, str(O / "fig7_power")); plt.close(fig)

# ------------------------------------------------------------ fig 8 (matched steps)
ck = rd("replication_checkpoint_means.csv")
new = [r for r in rc if r["origin"] != "original"]
fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.6), sharey=True)
for ax, n in zip(axes, RUNGS):
    for b in NAME:
        r = sorted((x for x in ck if x["backbone"] == b and int(x["n_train"]) == n and x["checkpoint"].startswith("step")),
                   key=lambda x: float(x["mean_step"]))
        ax.plot([float(x["mean_step"]) / 1000 for x in r], [float(x["excess"]) for x in r], "-o", color=MCOL[b],
                ms=3, label=NAME[b])
        mine = [x for x in new if x["backbone"] == b and int(x["n_train"]) == n]
        ax.plot(np.mean([float(x["best_step"]) for x in mine]) / 1000, np.mean([float(x["excess"]) for x in mine]),
                "*", color=MCOL[b], ms=10, mec="k", mew=0.5, zorder=5)
        a = [x for x in ck if x["backbone"] == b and int(x["n_train"]) == n and not x["checkpoint"].startswith("step")][0]
        ax.plot(float(a["mean_step"]) / 1000, float(a["excess"]), "D", color=MCOL[b], ms=4.5, mec="k", mew=0.5, zorder=5)
    ax.set_title(f"{n:,} training clips", fontsize=8); ax.set_xlabel("training steps (thousands)")
    ax.set_xlim(0, 24.5); ax.set_ylim(0, 0.145)
axes[0].set_ylabel("excess adaptive ECE")
axes[2].plot([], [], "*", color="0.7", mec="k", mew=0.5, ms=9, label="lowest validation NLL")
axes[2].plot([], [], "D", color="0.7", mec="k", mew=0.5, ms=4.5, label="best validation accuracy")
h, l = axes[2].get_legend_handles_labels()
axes[2].legend(h, l, fontsize=6, loc="upper left")
PS.save(fig, str(O / "fig8_steps")); plt.close(fig)
print("figures 5, 7, 8 written to", O)
