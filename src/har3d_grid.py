"""
har3d_grid.py -- the small-dataset training grid shared by Stage 6 (cross-
entropy + tuned MC dropout) and Stage 6b (label smoothing, focal loss).

One protocol for every method: the same quality-filtered samples, the same
subject-disjoint splits (three random + the conventional fixed split), the same
seeds, early stopping on validation NLL. Only the loss differs, so the six-
method comparison in Stage 9c is like-for-like. Resumable: any cell already in
the logit store is skipped.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import har3d_v2 as V
import har3d_models_v2 as M2
from store import Key, LogitStore

DATASETS = (("action3d", "msr_action3d.npz"),
            ("daily", "msr_daily.npz"),
            ("utkinect", "utkinect.npz"))


def splits_for(subj, y, verbose=False):
    sp = V.subject_splits(subj, labels=y, n_splits=3, seed=0,
                          n_test_subjects=4, n_val_subjects=2, verbose=verbose)
    try:
        sp.append(V.fixed_split(subj, labels=y, verbose=verbose))
    except ValueError as e:
        if verbose:
            print(f"    no fixed split: {e}")
    return sp


def run_grid(cache_dir, logits_dir, prep_cfg, prep_tag, model_tag="stgcn",
             loss_kw=None, streams=("joint",), seeds=(0,), mc_passes=0,
             mc_p_grid=(0.05, 0.1, 0.2, 0.3), T=64, max_epochs=120,
             patience=15, min_epochs=10, max_zero_frac=0.5, min_frames=10,
             datasets=DATASETS, device=None, verbose=True):
    """Train every (dataset, stream, split, seed) cell; returns a log dict."""
    loss_kw = dict(loss_kw or {})
    store = LogitStore(logits_dir)
    logits_dir = Path(logits_dir)
    A = V.adjacency(V.MSR_PARENTS)
    log = {"produced": 0, "skipped": 0, "cells": []}
    for ds, npz in datasets:
        f = Path(cache_dir) / npz
        if not f.exists():
            raise FileNotFoundError(f"{f} missing -- run the ingest stages first")
        cache = V.load_cache(f)
        for stream in streams:
            X, y, subj, keep = V.masked_tensors(cache, stream, T, prep_cfg,
                                                max_zero_frac, min_frames)
            nc = int(y.max()) + 1
            for sp in splits_for(subj, y):
                name = f"{sp['name']}_{prep_tag}"
                for seed in seeds:
                    kt = Key(model_tag, ds, stream, name, seed, "test")
                    if store.has(kt):
                        log["skipped"] += 1
                        continue
                    out, _ = M2.train_and_logits(
                        X, y, sp, nc, A, max_epochs=max_epochs, patience=patience,
                        min_epochs=min_epochs, seed=seed, mc_passes=mc_passes,
                        mc_p_grid=mc_p_grid, device=device, verbose=False,
                        **loss_kw)
                    info = out["info"]
                    prov = dict(arch="stgcn", method=model_tag, loss=loss_kw or "ce",
                                stream=stream, T=T, prep=prep_tag, **prep_cfg,
                                parents="corrected", scale="bone", impute=True,
                                n_after_filter=int(len(y)), n_raw=int(len(keep)),
                                subjects=sp["subjects"], train_info=info,
                                mc_passes=mc_passes)
                    for sub in ("val", "test"):
                        L, yy = out[sub]
                        store.save(Key(model_tag, ds, stream, name, seed, sub),
                                   L, yy, provenance=prov)
                        log["produced"] += 1
                        if sub + "_mc" in out:
                            P, _ = out[sub + "_mc"]
                            np.savez_compressed(
                                logits_dir / f"mc__{ds}__{model_tag}__{stream}__"
                                             f"{name}__s{seed}__{sub}.npz",
                                logits=P.astype(np.float32), labels=yy,
                                mc_p=np.float32(info["mc_p"] or np.nan))
                    row = dict(dataset=ds, stream=stream, split=name, seed=seed,
                               best_epoch=info["best_epoch"],
                               epochs_run=info["epochs_run"],
                               best_val_nll=info["best_val_nll"],
                               mc_p=info["mc_p"],
                               test_top1=float((out["test"][0].argmax(-1)
                                                == out["test"][1]).mean()))
                    log["cells"].append(row)
                    if verbose:
                        print(f"  {model_tag:<12}{ds:<9}{stream:<13}{name:<18}s{seed}"
                              f"  ep {info['best_epoch']:>3}/{info['epochs_run']:<3}"
                              f" top1 {row['test_top1']:.3f}"
                              + (f"  mc p={info['mc_p']}" if info["mc_p"] else ""))
    return log