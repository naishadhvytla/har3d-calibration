# When Is Calibration Measurable? — code and results

Code, result tables and manuscript generators for *"When Is Calibration
Measurable? A Sample-Size–Controlled Study of Confidence Reliability in 3D
Skeleton Action Recognition"* (Naishadh Vytla, Kiran Kumar Eepuri, R. Sekhar, Pardhu Thottempudi and Avishek Chakraborty).

The study asks when the expected calibration error (ECE) of a skeleton
action-recognition model can be told apart from estimator bias. It tests every
experiment against consistency resampling with the predictions held fixed (a
null model matched to the experiment's own confidences), extends that null to
experiments that share test clips and subjects, measures the power of the
test under controlled miscalibration, runs a fixed-test-set training-size
ladder on NTU RGB+D 60, compares three graph-layer variants (ST-GCN, 2s-AGCN,
CTR-GCN) on one shared trunk with seeds as the units and a pre-specified
replication, and compares calibration methods on binning-free proper scores
and against each method's own floor.

## Layout

| Path | What it is |
|---|---|
| `notebook/har3d_final_v7.ipynb` | the whole pipeline as one resumable Colab notebook (the only file that is executed; modules are written by `%%writefile` cells). Stages 5b, 8c and 9d-9h are the analyses added in the revision |
| `src/` | the same modules as plain files, for reading: preprocessing and topology gate (`har3d_v2.py`, dataset readers), graph-layer variants (`har3d_backbones.py`), ladder and replication training (`har3d_scale.py`), null model and calibration methods (`har3d_calib.py`), dependence-aware nulls, power analysis, estimator sensitivity and seed-level tests (`har3d_review.py`), unit-level inference (`har3d_stats.py`), logit store (`store.py`), the 117-assertion test suite (`test_har3d.py`), and the manuscript generators `make_numbers.py`, `make_figures.py` and `make_revision.py` |
| `results/` | everything the paper is computed from: one row per test cell (`metrics.json`), the result tables, the data filter, the ladder protocol, the registered replication protocol, training logs and the software stack of every session; see `results/README.md` |

## Regenerate every number, table and figure of the paper

```
pip install numpy scipy matplotlib
python src/make_numbers.py  results paper/generated
python src/make_figures.py  results paper/generated paper
python src/make_revision.py results/tables paper/generated paper
```

`paper/generated/` then holds `numbers.tex` and the `tab_*.tex` tables that the
manuscript `\input`s, and `paper/` the eight figures. No GPU and no stored
predictions are needed.

## Re-run the pipeline

Open `notebook/har3d_final_v7.ipynb` in Google Colab, put the raw datasets in
`MyDrive/har3d/` (layout in the notebook's first cell) and *Run all*; every
stage writes a sentinel and is skipped once complete. To run the pipeline from
scratch set `REQUIRE_EXISTING_RUN = False` in Stage 0 (the default continues a
finished run and refuses to start an empty one). Training needs a GPU
(the reported runs used one NVIDIA RTX PRO 6000 Blackwell GPU); evaluation,
statistics and figures run on a CPU runtime. Stage 10 writes the manuscript's
numbers, tables and figures to `results/paper/`.

## Data

The four datasets are public and are not redistributed here: MSR-Action3D and
MSR-DailyActivity3D (Wanqing Li, University of Wollongong), UTKinect-Action3D
(Xia, Chen and Aggarwal, CVPRW 2012) and NTU RGB+D 60 (ROSE Lab, Nanyang
Technological University; access on request). The stored predictions
(logits) from which the result tables were computed are available from the
author on request.

## Citation

See `CITATION.cff`.
