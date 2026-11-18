"""
har3d_daily.py -- MSR DailyActivity3D ingest, nested-archive aware.

WHY THE ORIGINAL FAILED
-----------------------
MSRDailyActivity3D.zip is a zip OF ZIPS: 9 members, every one of them stored
(compress_type 0) rather than deflated. Stored members copy at raw disk speed,
which is why 3.56 GB "extracted" inside a 101 s run. After extractall there
were nine archives and zero .txt, so find_files() matched nothing.

Fix: expand nested archives wherever they land, recursively.

PARSING
-------
np.loadtxt is not usable here. Skeleton files interleave per-frame header lines
with 4-column joint rows. This parser classifies each line by token count.

Rows per frame is 20 or 40 and divisibility cannot tell them apart. The 40-row
layout interleaves world coordinates (order 1 m) with screen coordinates, so
every candidate layout is tried and scored by bone plausibility.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np

from har3d_io import (expand_archives, extract_any, find_skeleton_files,
                      inspect_methods, read_rows, reclaim_media)

NAME_RE = re.compile(r"a(\d+)[_-]s(\d+)[_-]e(\d+)", re.IGNORECASE)
EXPECTED_SEQUENCES = 320


def _log(m=""):
    print(m, flush=True)


def read_joint_rows(path: Path) -> np.ndarray:
    """4-column joint rows; per-frame markers and the file header discarded."""
    rows, _ = read_rows(path, widths=(4,))
    return rows


CANDIDATES = [(20, "all"),      # 20 rows/frame, one coordinate block
              (40, "first"),    # 40 rows/frame, blocked: world first
              (40, "second"),   # 40 rows/frame, blocked: world second
              (40, "even"),     # 40 rows/frame, interleaved: world on even
              (40, "odd")]      # 40 rows/frame, interleaved: world on odd


def _slice(seq, mode, col=slice(0, 3)):
    """seq is (T, rows_per_frame, 4) -> (T, 20, len(col))."""
    if mode == "all":
        return seq[:, :, col]
    if mode == "first":
        return seq[:, :20, col]
    if mode == "second":
        return seq[:, 20:, col]
    if mode == "even":
        return seq[:, 0::2, col]
    if mode == "odd":
        return seq[:, 1::2, col]
    raise ValueError(f"unknown slice mode {mode!r}")


def score_layout(xyz, parents=None, max_len=0.60, max_cv=0.20):
    """
    Plausibility of a candidate parse, INDEPENDENT of joint order. Scoring
    against a fixed tree would assume Action3D's indexing, which this dataset
    does not share.
    """
    from har3d_v2 import score_parse
    return score_parse(xyz, max_len=max_len, max_cv=max_cv)


def choose_layout(paths, parents=None, n_probe=8):
    """Return (rows_per_frame, mode) for the layout that yields a real skeleton."""
    del parents
    probe = [read_joint_rows(p) for p in list(paths)[:n_probe]]
    probe = [r for r in probe if r.shape[0] > 0]
    if not probe:
        return None, None

    results = []
    for rows, mode in CANDIDATES:
        scores, meds = [], []
        for raw in probe:
            n = raw.shape[0]
            if n % rows or n // rows < 2:
                continue
            xyz = _slice(raw.reshape(n // rows, rows, raw.shape[1]), mode)
            if xyz.shape[1] != 20:
                continue
            f, med = score_layout(xyz)
            scores.append(f)
            meds.append(med)
        if scores:
            results.append((float(np.mean(scores)), rows, mode,
                            float(np.nanmedian(meds)), len(scores)))
    if not results:
        _log("  !! no candidate layout divides the row counts evenly")
        return None, None
    results.sort(reverse=True)

    # A pure 20-row file read as 40 rows/frame pairs two consecutive FRAMES into
    # one, and both halves are still real skeletons -- bone plausibility cannot
    # separate them and would silently halve the frame rate.
    top = results[0][0]
    close = [r for r in results if r[0] >= top - 0.03]
    if len(close) > 1:
        simplest = min(close, key=lambda r: (r[1], r[2] != "all"))
        if simplest is not results[0]:
            _log(f"  scores within 0.03 across {len(close)} candidates; "
                 f"preferring the simpler {simplest[1]} rows/frame layout")
            results.remove(simplest)
            results.insert(0, simplest)

    _log(f"  layout scan over {len(probe)} files "
         f"(fraction of edges that look like bones, median bone length):")
    from har3d_v2 import body_extent
    for f, rows, mode, med, n in results:
        mark = "  <-- chosen" if (f, rows, mode, med, n) == results[0] else ""
        ex = body_extent(_slice(probe[0].reshape(-1, rows, probe[0].shape[1]), mode)) \
            if probe[0].shape[0] % rows == 0 else (float("nan"),) * 2
        _log(f"   {rows:>2} rows/frame, world={mode:<6} score {f:5.2f}  "
             f"median edge {med:9.3f} m  body span {ex[1]:9.2f} m{mark}")

    best_f, rows, mode, med, _ = results[0]
    if best_f < 0.8:
        raise RuntimeError(
            f"no layout produces a plausible skeleton (best {best_f:.2f} of 1.00, "
            f"median edge {med:.3f} m). The file format differs from every "
            f"candidate -- inspect a raw file before going further.")
    return rows, mode


def parse_sequence(path: Path, layout: int, mode: str = "first"):
    """-> (xyz (T,20,3) float32, conf (T,20) float32) or None."""
    raw = read_joint_rows(path)
    n = raw.shape[0]
    if n == 0 or n % layout:
        return None
    seq = raw.reshape(n // layout, layout, raw.shape[1])
    xyz = _slice(seq, mode).astype(np.float32)
    conf = _slice(seq, mode, col=slice(3, 4))[..., 0].astype(np.float32)
    if xyz.shape[1] != 20:
        return None
    return xyz, conf


def ingest(drive_root, cache_dir, work="/content/har3d_work/msr_daily",
           reclaim=True, push_to_drive=True):
    """MSRDailyActivity3D.zip -> msr_daily.npz in the same ragged format."""
    drive_root, cache_dir = Path(drive_root), Path(cache_dir)
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    zp = None
    for pat in ("MSRDailyActivity3D.zip", "*DailyAct*.zip", "*Daily*.zip"):
        hits = sorted(drive_root.glob(pat))
        if hits:
            zp = hits[0]
            break
    if zp is None:
        return {"error": f"no DailyActivity archive under {drive_root}"}

    if not find_skeleton_files(work):
        info = inspect_methods(zp)
        _log(f"  {zp.name}: {info['names']}")
        _log(f"  extracting with {extract_any(zp, work)} ...")
        n = expand_archives(work)
        _log(f"  expanded {n} nested archives")
        if n == 0:
            _log("  !! no nested archive expanded -- see the messages above")

    skel = find_skeleton_files(work)
    _log(f"  skeleton files: {len(skel)} (expected {EXPECTED_SEQUENCES})")
    if not skel:
        present = sorted({p.suffix.lower() for p in work.rglob("*") if p.is_file()})
        return {"error": "no skeleton files after expansion",
                "suffixes_present": present}

    layout, mode = choose_layout(skel)
    if layout is None:
        return {"error": "could not determine rows per frame"}

    perm, perm_info = None, {}
    ref = cache_dir / "msr_action3d.npz"
    if ref.exists():
        import har3d_v2 as V
        rz = np.load(ref, allow_pickle=False)
        ro = rz["offsets"]
        ref_xyz = rz["xyz"][ro[0]:ro[min(120, len(ro) - 1)]]
        probe_xyz = np.concatenate(
            [parse_sequence(p, layout, mode)[0] for p in skel[:60]
             if parse_sequence(p, layout, mode) is not None], 0)
        perm, perm_info = V.derive_canonical_permutation(probe_xyz, ref_xyz)
    else:
        _log("  !! msr_action3d.npz not in the cache -- cannot check joint order; "
             "storing the dataset's own order")

    xyz_all, conf_all, lens, metas, flags, skipped = [], [], [], [], [], []
    for p in skel:
        m = NAME_RE.search(p.name)
        if m is None:
            skipped.append((p.name, "unparseable name"))
            continue
        parsed = parse_sequence(p, layout, mode)
        if parsed is None:
            skipped.append((p.name, "row count not a multiple of layout"))
            continue
        xyz, conf = parsed
        if perm is not None:
            xyz, conf = xyz[:, perm], conf[:, perm]
        xyz_all.append(xyz)
        conf_all.append(conf)
        lens.append(xyz.shape[0])
        metas.append([int(m.group(1)), int(m.group(2)), int(m.group(3))])
        zero = (np.abs(xyz) < 1e-9).all(-1)
        flags.append([xyz.shape[0], float(1.0 - np.isfinite(xyz).all(-1).mean()),
                      float(zero.mean()), float((conf <= 0).mean())])

    if not xyz_all:
        return {"error": "every skeleton file failed to parse",
                "skipped": skipped[:10]}

    out = cache_dir / "msr_daily.npz"
    np.savez_compressed(
        out,
        xyz=np.concatenate(xyz_all, 0).astype(np.float32),
        conf=np.concatenate(conf_all, 0).astype(np.float32),
        offsets=np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
        meta=np.array(metas, np.int64),
        flags=np.array(flags, np.float64),
        rows_per_frame=np.array(layout),
        world_rows=np.array(mode),
        joint_perm=np.array(perm if perm is not None else np.arange(20)))
    _log(f"  cached -> {out.name} ({out.stat().st_size/1e6:.1f} MB, "
         f"{len(metas)} sequences)")

    freed = reclaim_media(work) if reclaim else 0
    if reclaim:
        _log(f"  reclaimed {freed/1e9:.2f} GB of RGB/depth")

    if push_to_drive:
        (drive_root / "cache").mkdir(parents=True, exist_ok=True)
        shutil.copy2(out, drive_root / "cache" / out.name)
        _log(f"  copied to {drive_root / 'cache' / out.name}")

    subj = sorted({m[1] for m in metas})
    acts = sorted({m[0] for m in metas})
    _log(f"  subjects {subj}  actions {len(acts)}  frames {sum(lens):,}")

    res = {"n_sequences": len(metas), "rows_per_frame": layout,
           "world_rows": mode, "joint_order": perm_info,
           "subjects": subj, "n_actions": len(acts), "cache": str(out),
           "freed_bytes": freed}
    if len(metas) != EXPECTED_SEQUENCES:
        res["warning"] = f"{len(metas)} sequences, expected {EXPECTED_SEQUENCES}"
        _log(f"  !! {res['warning']}")
    if skipped:
        res["skipped"] = skipped[:10]
        _log(f"  !! {len(skipped)} files skipped, first few: {skipped[:3]}")
    return res