"""
make_figures.py RESULTS_DIR GENERATED_DIR OUT_DIR

Draws all six figures of the manuscript from the result tables of a run
(run make_numbers.py first: it writes GENERATED_DIR/methods_split_level.json).
Nothing here needs the stored predictions: the two figures that pool raw
predictions read the per-bin and per-class tables that Stage 10 of the
notebook writes (tables/reliability_bins.csv, tables/per_class_pooled.csv).
Inference units follow the paper: test sets (dataset x split) on the small
datasets; error bars over seeds on the ladder.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))  # har3d_plotstyle.py ships alongside
import har3d_plotstyle as PS  # noqa: E402

R, G, O = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
O.mkdir(parents=True, exist_ok=True)
T = R / "tables"
PS.apply()
rows = json.loads((R / "metrics.json").read_text())
# replication and environment re-runs (revision) are reported from their own tables
rows = [r for r in rows if not r["model"].endswith(("_rep", "_env"))]
LABEL = {"daily": "DailyActivity3D", "action3d": "MSR-Action3D", "utkinect": "UTKinect", "ntu60": "NTU-60"}
SMALL = ["action3d", "daily", "utkinect"]
MCOL = {"stgcn": PS.PALETTE["blue"], "agcn": PS.PALETTE["vermillion"], "ctrgcn": PS.PALETTE["green"],
        "agcn_frozen": PS.PALETTE["orange"], "stgcn_fixed": PS.PALETTE["skyblue"]}
MNAME = {"stgcn": "ST-GCN (fixed graph)", "agcn": "2s-AGCN (adaptive)", "ctrgcn": "CTR-GCN (channel-wise)",
         "agcn_frozen": "2s-AGCN, graph frozen", "stgcn_fixed": "ST-GCN, fixed step budget"}
lad = [r for r in rows if r["dataset"] == "ntu60"
       and not (r["model"] == "agcn_frozen" and r.get("frozen_at") is None)]


def read_csv(name):
    return list(csv.DictReader(open(T / name)))


# ---------------------------------------------------------------- fig 1
# Equal-mass reliability diagrams, ST-GCN joint stream pooled over splits and
# seeds: bars span each bin's confidence range at the bin's accuracy; the dot
# marks the bin's (mean confidence, accuracy).
rb = read_csv("reliability_bins.csv")
fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.7), sharey=True)
for ax, ds in zip(axes, SMALL):
    b = sorted((x for x in rb if x["dataset"] == ds), key=lambda x: int(x["bin"]))
    lo = np.array([float(x["conf_lo"]) for x in b]); hi = np.array([float(x["conf_hi"]) for x in b])
    acc = np.array([float(x["accuracy"]) for x in b]); mconf = np.array([float(x["mean_conf"]) for x in b])
    ax.plot([0, 1], [0, 1], "--", color="0.45", lw=0.8, zorder=1, label="perfect calibration")
    ax.bar(lo, acc, width=hi - lo, align="edge", color=PS.ROLE["bar"], edgecolor="white", lw=0.4,
           zorder=2, label="accuracy of bin")
    ax.plot(mconf, acc, "o", ms=2.5, color=PS.PALETTE["blue"], zorder=4, label="(mean confidence, accuracy)")
    sub = [r for r in rows if (r["model"], r["dataset"], r["stream"]) == ("stgcn", ds, "joint")]
    e = [r["ece_adaptive"] for r in sub]
    ax.set_title(f"{LABEL[ds]}\naECE {np.mean(e):.3f} ± {np.std(e, ddof=1):.3f}, "
                 f"floor {np.mean([r['null_null_mean'] for r in sub]):.3f}", fontsize=8)
    ax.set_xlabel("confidence (equal-mass bins)"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
axes[0].set_ylabel("accuracy")
axes[0].legend(loc="upper left", fontsize=6, handlelength=1.6)
PS.save(fig, str(O / "fig1_reliability")); plt.close(fig)

# ---------------------------------------------------------------- fig 2
fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(7.4, 2.7))
pts = [r for r in rows if r["model"] == "stgcn" and r["stream"] == "joint"]
for ds in ["action3d", "daily", "utkinect", "ntu60"]:
    sel = [r for r in pts if r["dataset"] == ds]
    x = [np.sqrt(r["n_classes"] / r["n"]) for r in sel]
    axA.scatter(x, [r["ece_adaptive"] for r in sel], s=14, color=PS.ROLE[ds], label=LABEL[ds],
                edgecolor="white", lw=0.3, zorder=3)
    axA.scatter(x, [r["null_null_mean"] for r in sel], s=12, marker="x", color=PS.ROLE["null"], zorder=2,
                label="null floor" if ds == "action3d" else None)
axA.set_xscale("log"); axA.set_yscale("log")
axA.set_xlabel(r"$\sqrt{K/n_{\mathrm{test}}}$"); axA.set_ylabel("adaptive ECE")
axA.set_title("(a) observed vs null floor")
_C = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
_xv = [np.sqrt(r["n_classes"] / r["n"]) for r in pts]
_yv = [r["ece_adaptive"] for r in pts] + [r["null_null_mean"] for r in pts]
axA.set_xlim(min(_xv) / 1.3, max(_xv) * 1.3)
for ax_, vals in ((axA.xaxis, _xv), (axA.yaxis, _yv)):
    lo_, hi_ = min(vals), max(vals)
    ax_.set_major_locator(FixedLocator([c for c in _C if lo_ / 1.3 <= c <= hi_ * 1.3]))
    ax_.set_major_formatter(FormatStrFormatter("%g")); ax_.set_minor_formatter(NullFormatter())
axA.legend(fontsize=6)
for mdl in MCOL:
    by = defaultdict(list)
    for r in lad:
        if r["model"] == mdl:
            by[r["n_train"]].append(r)
    ns = np.array(sorted(by))
    ls = "--" if mdl in ("agcn_frozen", "stgcn_fixed") else "-"
    for ax, key in ((axB, "null_excess"), (axC, "top1")):
        mu = [np.mean([r[key] for r in by[n]]) for n in ns]
        sd = [np.std([r[key] for r in by[n]], ddof=1) for n in ns]
        ax.errorbar(ns, mu, yerr=sd, fmt="o" + ls, ms=3, capsize=2, color=MCOL[mdl], label=MNAME[mdl])
fl = [np.mean([r["null_null_mean"] for r in lad if (r["model"], r["n_train"]) == mn])
      for mn in sorted({(r["model"], r["n_train"]) for r in lad})]       # per variant and rung, as in the text
axB.axhline(0, color="0.7", lw=0.6)
axB.set_xscale("log"); axB.set_xlabel("NTU-60 training clips")
axB.set_ylabel("excess adaptive ECE (observed − floor)")
axB.set_title(f"(b) ladder, fixed test set\n(floor {min(fl):.4f}–{max(fl):.4f})")
axC.set_xscale("log"); axC.set_xlabel("NTU-60 training clips"); axC.set_ylabel("top-1")
axC.set_title("(c) accuracy"); axC.legend(fontsize=5.8, loc="lower right")
PS.save(fig, str(O / "fig2_scale")); plt.close(fig)

# ---------------------------------------------------------------- fig 3
fig, ax = plt.subplots(figsize=(3.6, 2.6))
xs, w = np.arange(len(SMALL)), 0.36
st = [r for r in rows if r["model"] == "stgcn" and r["dataset"] in SMALL]
for off, key, lab, col in ((-w / 2, "ece_adaptive", "raw", PS.ROLE["observed"]),
                           (w / 2, "ece_adaptive_temp", "temperature scaled", PS.ROLE["excess"])):
    mu, lo, hi = [], [], []
    for d in SMALL:
        sub = [r for r in st if r["dataset"] == d]
        per = defaultdict(list)
        for r in sub:
            per[r["split"]].append(r[key])
        tm = [np.mean(v) for v in per.values()]          # one mean per test set
        mu.append(np.mean([r[key] for r in sub])); lo.append(min(tm)); hi.append(max(tm))
    ax.bar(xs + off, mu, w, color=col, label=lab,
           yerr=[np.subtract(mu, lo), np.subtract(hi, mu)], capsize=2, error_kw={"lw": 0.8})
flo = [np.mean([r["null_null_mean"] for r in st if r["dataset"] == d]) for d in SMALL]
ax.scatter(xs - w / 2, flo, marker="_", s=160, lw=1.6, color="k", zorder=5, label="null floor, raw")
if (T / "temperature_floor.csv").exists():
    tf = {x["dataset"]: float(x["floor_ts"]) for x in read_csv("temperature_floor.csv")}
    ax.scatter(xs + w / 2, [tf[d] for d in SMALL], marker="_", s=160, lw=1.6, color="0.45", zorder=5,
               label="null floor, scaled")
ax.set_xticks(xs); ax.set_xticklabels([LABEL[d] for d in SMALL], fontsize=7)
ax.set_ylabel("adaptive ECE"); ax.legend(fontsize=6.5, loc="upper left")
PS.save(fig, str(O / "fig3_tempscaling")); plt.close(fig)

# ---------------------------------------------------------------- fig 4
# Reduction versus the uncalibrated model (positive is better) for each score,
# with cluster-bootstrap intervals over test sets. Tested scores are coloured by
# the Holm-corrected sign test; raw adaptive ECE (when each method's floor is
# not available) is shown in grey because it includes the shift of the floor.
sl = json.loads((G / "methods_split_level.json").read_text())
order = ["temperature", "vector", "label_smooth", "focal", "mc_dropout", "deep_ensemble"]
DISP = {"temperature": "temperature", "vector": "vector scaling", "label_smooth": "label smoothing",
        "focal": "focal loss", "mc_dropout": "MC dropout", "deep_ensemble": "deep ensemble (5)"}
TITLE = {"ece": "adaptive ECE (not tested)", "excess": "excess adaptive ECE", "brier": "Brier score", "nll": "NLL"}
fig, axes = plt.subplots(1, len(sl["panels"]), figsize=(7.2, 2.35), sharey=True)
yb = np.arange(len(order))
for ax, pan in zip(axes, sl["panels"]):
    M = pan["methods"]
    mu = np.array([M[m]["mean"] for m in order])
    lo = np.array([M[m]["lo"] for m in order]); hi = np.array([M[m]["hi"] for m in order])
    col = [PS.ROLE["null"] if not pan["tested"] else
           PS.PALETTE["green"] if M[m]["verdict"] == "reduces" else
           PS.PALETTE["vermillion"] if M[m]["verdict"] == "increases" else PS.PALETTE["skyblue"] for m in order]
    ax.barh(yb, mu, xerr=[mu - lo, hi - mu], color=col, capsize=2, error_kw={"lw": 0.8}, height=0.65)
    ax.axvline(0, color="0.3", lw=0.8)
    ax.set_title(TITLE[pan["key"]], fontsize=8)
    ax.set_xlabel("reduction vs. uncalibrated", fontsize=7)
    span = max(np.abs(np.concatenate([lo, hi])).max(), 1e-3) * 1.15
    ax.set_xlim(-span, span)
axes[0].set_yticks(yb); axes[0].set_yticklabels([DISP[m] for m in order]); axes[0].invert_yaxis()
PS.save(fig, str(O / "fig4_methods")); plt.close(fig)

# ---------------------------------------------------------------- fig 5
pcp = read_csv("per_class_pooled.csv")
fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.4))
for ax, ds in zip(axes, SMALL):
    c = sorted((x for x in pcp if x["dataset"] == ds), key=lambda x: -float(x["per_class_ece"]))
    e = [float(x["per_class_ece"]) for x in c]; r1 = [float(x["one_minus_recall"]) for x in c]
    ax.bar(np.arange(len(c)), e, color=PS.ROLE[ds], width=0.8, label="per-class ECE")
    ax.plot(np.arange(len(c)), r1, "k.", ms=3, label="1 − class recall")
    ax.set_title(LABEL[ds], fontsize=8); ax.set_xlabel("class (sorted by ECE)")
    ax.set_xticks([]); ax.set_ylim(0, 1)
axes[0].set_ylabel("per-class ECE / 1 − recall"); axes[0].legend(fontsize=6, loc="upper right")
PS.save(fig, str(O / "fig5_perclass")); plt.close(fig)

# ---------------------------------------------------------------- fig 6
mrows = {r["method"]: r for r in read_csv("methods_comparison.csv") if r["scope"] == "all"}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.6))
sub = order[1:]
xb = np.arange(len(sub))
a1.bar(xb - 0.2, [float(mrows[m]["rel"]) - float(mrows["raw"]["rel"]) for m in sub], 0.4,
       label="Δ reliability (REL; < 0 better)", color=PS.PALETTE["vermillion"])
a1.bar(xb + 0.2, [float(mrows[m]["res"]) - float(mrows["raw"]["res"]) for m in sub], 0.4,
       label="Δ resolution (RES; > 0 better)", color=PS.PALETTE["blue"])
a1.axhline(0, color="0.4", lw=0.8)
a1.set_xticks(xb); a1.set_xticklabels([DISP[m] for m in sub], rotation=35, ha="right", fontsize=6.5)
a1.set_ylabel("change vs. uncalibrated"); a1.set_title("(a) small datasets, by method")
a1.legend(fontsize=6, loc="best", framealpha=0.8)
for mdl in MCOL:
    by = defaultdict(list)
    for r in lad:
        if r["model"] == mdl:
            by[r["n_train"]].append(r["brier_rel"])
    ns = sorted(by)
    a2.errorbar(ns, [np.mean(by[n]) for n in ns], yerr=[np.std(by[n], ddof=1) for n in ns],
                fmt="o--" if mdl in ("agcn_frozen", "stgcn_fixed") else "o-", ms=3, capsize=2,
                color=MCOL[mdl], label=MNAME[mdl])
a2.set_xscale("log"); a2.set_xlabel("NTU-60 training clips")
a2.set_ylabel("Brier reliability (REL)"); a2.set_title("(b) NTU-60 ladder")
a2.legend(fontsize=5.8, loc="best", framealpha=0.8)
PS.save(fig, str(O / "fig6_brier")); plt.close(fig)
print("figures 1-6 written to", O)
