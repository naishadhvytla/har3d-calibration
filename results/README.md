# results/

Output of the pipeline for the runs reported in the paper. Every model was
trained in notebook sessions v1 and v2; sessions v3–v6 only recomputed
statistics from the stored predictions. The software stack of each session is
in `environment/environment_history.json` (the v6 session ran on the same
stack as v5, so it updated that record's `last_seen` instead of adding one).
Every file here was written by the notebook (v6 for the first-version files, v7 and v8 for the revision).

| File | Content |
|---|---|
| `metrics.json` | one row per test cell (941; see the v8 section for the run tags): accuracy, adaptive and equal-width ECE, classwise ECE, Brier and its Murphy terms, NLL, fitted temperature and post-scaling metrics, per-cell bootstrap intervals (1,000 resamples), and the null-model test (null mean, 95th percentile, excess, Monte-Carlo p-value; 20,000 draws on the small datasets, 500 on NTU-60) |
| `data_filtering.json` | the clip-quality rule and the excluded MSR-Action3D identifiers |
| `ladder_protocol.json` | NTU-60 validation performers, pool size and pool hash (identical for every rung, seed and backbone) |
| `prep_config.json` | the preprocessing ablation and the chosen setting |
| `tables/` | the tables below |
| `environment/` | `environment.json` (last session), `environment_history.json` (every software stack), pip lock files |
| `figures/` | the six figures of the paper (Stage 10) |

Tables:

- `rq1_separability.csv` — per-cell null tests, Holm over the 240 ST-GCN cells
- `rq1_pooled_test_sets.csv` — excess averaged within each of the 12 test sets (dataset × split)
- `ladder_by_model.csv`, `ladder_convergence.csv` — ladder means and stopping points
- `scaling_law_by_model.csv` — OLS fit of log excess on log training size over the rung means, t-interval
- `ladder_paired_comparisons.csv` — paired differences on the fixed NTU-60 test set (bootstrap over test clips, 5,000 draws, Holm within family)
- `ladder_seed_level.csv` — sign tests over seed–rung pairs (separately trained runs), one Holm family
- `methods_per_cell.csv` — every calibration method on every small-dataset cell, with each method's own null floor (1,000 draws on the method's probabilities) and the excess over it
- `methods_comparison.csv` — descriptive means per method, overall and per dataset
- `methods_split_level.csv` — method comparisons over the 12 test sets for the excess, the Brier score and the NLL (tested; sign test, Holm over the six methods for each score, cluster-bootstrap intervals) and the raw adaptive ECE (reference only); the test sets of a dataset share subjects, so the p-values are approximate
- `rq3_temperature_scaling.csv` — temperature scaling per dataset × stream (descriptive)
- `temperature_floor.csv` — the null floor before and after temperature scaling, per dataset (Fig. 3)
- `stream_fusion.csv` — single-stream and fused logits, each with its own floor, Brier score and NLL before and after a temperature fitted to it
- `reliability_bins.csv`, `per_class_pooled.csv` — the data behind Figs. 1 and 5

To regenerate every number, table and figure of the manuscript from this
folder, see the top-level README.

## Added in the revision (v7 notebook)

`replication_protocol.json` (hypotheses, statistic, unit and tests, written
before any replication run), `replication_env.json` (the pinned environment),
`review_dependence.json`, `review_replication.json`, and in `tables/`:

- `estimator_cells.csv`, `estimator_sensitivity.csv` - eight estimators, each against its own matched null
- `subject_heterogeneity.csv`, `pooled_dependence.csv`, `separability_clustered.csv`, `subject_gap.csv` - nulls that keep shared clips and subject effects; subjects as units
- `power_cells.csv`, `power_curves.csv`, `power_summary.csv`, `power_pooled.csv`, `power_ntu.csv`, `power_synthetic.csv`, `size_clustered.csv` - power and size of the test
- `calibration_fits.csv`, `methods_subject_level.csv`, `methods_pooled_null.csv` - calibration methods over subjects; cross-fitted and test-fitted scaling
- `per_class_null.csv` - per-class ECE with class counts and a matched null
- `prep_sensitivity.csv`, `train_only_checks.csv` - alternative preprocessing; tree and joint order from training subjects only
- `replication_cells.csv`, `replication_seed_level.csv`, `replication_checkpoints.csv`, `replication_matched_steps.csv`, `replication_checkpoint_means.csv`, `environment_reruns.csv`, `estimator_ladder_cells.csv`, `estimator_ladder_seed_level.csv`, `ntu_subject_dependence.csv` - replication seeds, matched training steps, environment re-runs, estimators and subject dependence on the ladder

In `methods_subject_level.csv` the row for the NLL of cross-fitted temperature
scaling carries the verdict `increases` written by the notebook as it was run;
the method is better for 20 of 27 subjects with a mean change of about zero,
which the current code labels `mixed`.

## Added in the second round of the revision (v8 notebook)

Run tags in `metrics.json` and in the tables:

- `stgcn_rep`, `agcn_rep`, `ctrgcn_rep`, seed 5: the optional third replication seed listed in `replication_protocol.json`, trained in the pinned environment of `replication_env.json`.
- `agcn_static` (graph A + A_L, no data-dependent term) and `agcn_fixedgraph` (graph frozen at initialization: A only), seeds 0-5 at 10,000 / 20,000 / 32,202 clips, pinned environment: the graph ablations of 2s-AGCN.
- `stgcn_pin`, `agcn_pin`, `ctrgcn_pin`, seeds 0-2 at the same rungs: the original seeds retrained in the pinned environment, so that six seeds of every variant share one environment.
- `*_v8` (`agcn_v8`, `stgcn_v8`, `agcn_static_v8`, `agcn_fixedgraph_v8`, `ctrgcn_v8`; 29 cells of seeds 3 and 5): runs of an interrupted session on an NVIDIA A100, started because the pinned GPU type was temporarily unavailable and stopped when it became available again. They are kept for completeness and are **not used** anywhere in the paper; `src/make_numbers.py`, `src/make_figures.py` and the notebook exclude them.

Files:

- `v8_protocol.json`: written by that interrupted A100 session before any v8 run was trained. It fixes the two graph ablations, the four questions of the ablation chain (ST-GCN, fixed graph, static graph, 2s-AGCN), the statistic and the tests that the paper reports. Its `runs` entry describes the run layout of the A100 session (tags `*_v8`, one GPU type for all six seeds); the runs reported in the paper were then trained in the pinned environment of `replication_env.json` instead, with the tags listed above, and the questions, statistic and tests were unchanged. `v8_env.json` is the software stack of the A100 session.
- `v8_progress.json`: one row per v8 training run, in the order trained (both sessions).
- `review_v8.json`: completeness flags and the main results of Stage 9i.
- `tables/graph_ablation_cells.csv`, `graph_ablation_means.csv`, `graph_ablation_seed_level.csv`: the graph ablations (Table 11 of the paper).
- `tables/v8_seed_level.csv`: the comparison of the variants with six seeds, in the pinned environment and with the original runs of seeds 0-2.
- `tables/pinned_vs_original.csv`: each retrained run of seeds 0-2 against its original.
- `tables/v8_checkpoints_seed_level.csv`: matched training steps and the best-validation-accuracy checkpoint, six seeds in the pinned environment; the ablation chain at the best-accuracy checkpoint.
- `tables/v8_estimator_seed_level.csv`: the ablation chain and the six-seed comparison under all eight estimators.

The tables written by Stage 9h (`replication_*.csv`, `estimator_ladder_*.csv`, `ntu_subject_dependence.csv`) now include seed 5.

The run's stage markers (`.done_*`) and Stage 10's `paper/` folder are not
included; `src/make_numbers.py`, `src/make_figures.py` and
`src/make_revision.py` regenerate every number, table and figure of the paper
from this folder.
