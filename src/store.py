"""
store.py -- the logit store.

Filename convention:
    {dataset}__{model}__{stream}__{split}__s{seed}__{subset}.npz
e.g. action3d__stgcn__bone_motion__rand0__s2__test.npz

This is the hinge of the project: once (model, dataset, stream, split, seed,
subset) -> logits, labels is on Drive, every table and figure downstream is
CPU-cheap and re-runnable forever. It is the only artifact that costs GPU
hours to rebuild.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Key:
    model: str      # stgcn | ctrgcn | ...
    dataset: str    # daily | action3d | utkinect | ntu60 | ntu120
    stream: str     # joint | bone | joint_motion | bone_motion
    split: str      # xsub | rand0 | fixed | ladder_n5000 ...
    seed: int
    subset: str     # val | test

    def fname(self) -> str:
        return (f"{self.dataset}__{self.model}__{self.stream}__"
                f"{self.split}__s{self.seed}__{self.subset}.npz")


class LogitStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, k: Key) -> Path:
        return self.root / k.fname()

    def has(self, k: Key) -> bool:
        return self.path(k).exists()

    def save(self, k: Key, logits, labels, sample_ids=None, top1=None,
             provenance: dict = None):
        logits = np.asarray(logits, np.float32)
        labels = np.asarray(labels, np.int64)
        assert logits.ndim == 2 and len(logits) == len(labels), \
            f"{k.fname()}: logits {logits.shape} vs labels {labels.shape}"
        prov = dict(provenance or {})
        prov.update({"saved": time.ctime(), "key": asdict(k),
                     "n": int(len(labels)), "C": int(logits.shape[1])})
        if top1 is None:
            top1 = float((logits.argmax(-1) == labels).mean())
        prov["top1"] = float(top1)
        prov["logit_sha1"] = hashlib.sha1(logits.tobytes()).hexdigest()[:16]
        np.savez_compressed(
            self.path(k), logits=logits, labels=labels,
            sample_ids=np.asarray(sample_ids if sample_ids is not None
                                  else np.arange(len(labels))),
            provenance=np.array(json.dumps(prov)))
        return self.path(k)

    def load(self, k: Key) -> dict:
        z = np.load(self.path(k), allow_pickle=False)
        out = {f: z[f] for f in z.files}
        out["provenance"] = json.loads(str(out["provenance"]))
        return out

    def list_keys(self) -> list:
        keys = []
        for p in sorted(self.root.glob("*.npz")):
            parts = p.stem.split("__")
            if len(parts) != 6:
                continue                      # e.g. the mc__ dumps
            ds, mdl, st, sp, sd, sub = parts
            if not sd.startswith("s") or not sd[1:].isdigit():
                continue
            keys.append(Key(mdl, ds, st, sp, int(sd[1:]), sub))
        return keys

    def inventory(self) -> dict:
        keys = self.list_keys()
        inv = {}
        for k in keys:
            inv.setdefault(k.dataset, {}).setdefault(k.model, set()).add(k.stream)
        print(f"  logit store: {len(keys)} files under {self.root}")
        for ds in sorted(inv):
            for m in sorted(inv[ds]):
                print(f"    {ds:<10} {m:<10} streams: {sorted(inv[ds][m])}")
        return inv

    def missing(self, models, datasets, streams, splits, seeds, subsets):
        """What still has to be produced. Run before booking GPU time."""
        want = [Key(m, d, st, sp, sd, sub)
                for d in datasets for m in models for st in streams
                for sp in splits for sd in seeds for sub in subsets]
        miss = [k for k in want if not self.has(k)]
        print(f"  {len(want) - len(miss)}/{len(want)} cells present, "
              f"{len(miss)} missing")
        return miss

    def ensemble(self, base: Key, streams=("joint", "bone",
                                           "joint_motion", "bone_motion")):
        """
        Multi-stream fusion. Raises if members are not aligned -- silently
        averaging misaligned members would corrupt every ensembling result.
        """
        probs, labels, used = [], None, []
        for st in streams:
            k = Key(base.model, base.dataset, st, base.split, base.seed, base.subset)
            if not self.has(k):
                continue
            d = self.load(k)
            if labels is None:
                labels = d["labels"]
            elif not np.array_equal(labels, d["labels"]):
                raise ValueError(f"label mismatch between streams for {k.fname()}")
            probs.append(_softmax(d["logits"]))
            used.append(st)
        if not probs:
            raise FileNotFoundError(f"no streams found for {base.fname()}")
        return np.mean(np.stack(probs, 0), 0), labels, used

    def ensemble_logits(self, base: Key, streams=("joint", "bone",
                                                  "joint_motion", "bone_motion")):
        """
        Mean LOGITS rather than mean probabilities, so a temperature can be
        fitted to the fusion. RQ5 previously fitted T for every individual
        stream and never for the ensemble, which is why "ensembling hurts
        calibration" may have been an artifact of the omitted step.
        """
        zs, labels, used = [], None, []
        for st in streams:
            k = Key(base.model, base.dataset, st, base.split, base.seed, base.subset)
            if not self.has(k):
                continue
            d = self.load(k)
            if labels is None:
                labels = d["labels"]
            elif not np.array_equal(labels, d["labels"]):
                raise ValueError(f"label mismatch between streams for {k.fname()}")
            zs.append(np.asarray(d["logits"], np.float64))
            used.append(st)
        if not zs:
            raise FileNotFoundError(f"no streams found for {base.fname()}")
        return np.mean(np.stack(zs, 0), 0), labels, used


def _softmax(z, T=1.0):
    z = np.asarray(z, np.float64) / T
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)