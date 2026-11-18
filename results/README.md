# results/

Output of the pipeline for the runs reported in the paper. Every model was
trained in notebook sessions v1 and v2; sessions v3–v6 only recomputed
statistics from the stored predictions. The software stack of each session is
in `environment/environment_history.json` (the v6 session ran on the same
stack as v5, so it updated that record's `last_seen` instead of adding one).
Every file here was written by the notebook (v6 for the first-version files, v7 for the revision).

| File | Content |
|---|---|
| `metrics.json` | one row per test cell (819): accuracy, adaptive and equal-width ECE, classwise ECE, Brier and its Murphy terms, NLL, fitted temperature and post-scaling metrics, per-cell bootstrap intervals (1,000 resamples), and the null-model test (null mean, 95th percentile, excess, Monte-Carlo p-value; 20,000 draws on the small datasets, 500 on NTU-60) |
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

The run's stage markers (`.done_*`) and Stage 10's `paper/` folder are not
included; `src/make_numbers.py`, `src/make_figures.py` and
`src/make_revision.py` regenerate every number, table and figure of the paper
from this folder.
