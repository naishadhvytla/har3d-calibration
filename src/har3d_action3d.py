"""
har3d_action3d.py -- MSR Action3D ingest.

Two archives ship the same 567 sequences in different coordinate systems:
    MSRAction3DSkeletonReal3D.rar      metric (x, y, z) in metres
    MSRAction3DSkeleton(20joints).rar  screen (u, v, d, confidence)

The metric archive is used directly. The screen archive contributes only its
confidence column, which is the per-joint noise proxy RQ4 needs, and the
least-squares intrinsics solve that confirms the two are the same recording.

SETTLED: fitting x/d against u recovers fx = 285.633, cx = 160.002,
fy = -285.574, cy = 120.001, depth scale 0.003998. That is the canonical
Kinect v1 focal length to 0.001%, so the resolution is 320x240 and no
back-projection is needed. The negative fy is the expected image-v-down
against world-y-up flip.

This dataset defines the canonical joint order for the project.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from har3d_io import extract_any, find_skeleton_files, log, read_rows

NAME_RE = re.compile(r"a(\d+)[_-]s(\d+)[_-]e(\d+)", re.IGNORECASE)
N_JOINTS = 20
EXPECTED_FULL = 567
EXPECTED_CONVENTIONAL = 557
KINECT_V1_FX = 285.63


def _resolve(drive_root: Path, patterns):
    for pat in patterns:
        exact = drive_root / pat
        if "*" not in pat and exact.exists():
            return exact
        hits = sorted(drive_root.glob(pat))
        if hits:
            return hits[0]
    return None


def parse_name(path: Path):
    m = NAME_RE.search(path.name)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def load_sequence(path: Path, n_joints=N_JOINTS):
    """(T, J, C) or None. Rows must divide evenly by the joint count."""
    rows, w = read_rows(path, widths=(3, 4))
    if rows.shape[0] == 0 or rows.shape[0] % n_joints:
        return None
    return rows.reshape(rows.shape[0] // n_joints, n_joints, w)


def solve_intrinsics(screen: np.ndarray, real: np.ndarray) -> dict:
    """
    Recover fx, fy, cx, cy by least squares rather than guessing.
    x = (u - cx) d / fx  =>  x/d = (1/fx) u + (-cx/fx), so a line fit of x/d
    against u returns fx and cx exactly.
    """
    u, v, d = screen[:, 0], screen[:, 1], screen[:, 2]
    x, y, z = real[:, 0], real[:, 1], real[:, 2]
    ok = np.isfinite([u, v, d, x, y, z]).all(0) & (np.abs(d) > 1e-6)
    if ok.sum() < 100:
        raise ValueError(f"only {int(ok.sum())} usable joints for the fit")
    u, v, d, x, y, z = (a[ok] for a in (u, v, d, x, y, z))

    def fit(px, ratio):
        A = np.stack([px, np.ones_like(px)], 1)
        (slope, icept), *_ = np.linalg.lstsq(A, ratio, rcond=None)
        pred = A @ np.array([slope, icept])
        ss_res = float(((ratio - pred) ** 2).sum())
        ss_tot = float(((ratio - ratio.mean()) ** 2).sum())
        f = 1.0 / slope if abs(slope) > 1e-12 else float("inf")
        return float(f), float(-icept * f), 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    fx, cx, r2x = fit(u, x / d)
    fy, cy, r2y = fit(v, y / d)
    s = float(np.median(z / d))
    return {"fx_raw": fx, "fy_raw": fy, "cx": cx, "cy": cy,
            "depth_scale": s, "fx": fx * s, "fy": fy * s,
            "r2_x": r2x, "r2_y": r2y, "n_points": int(ok.sum())}


def degeneracy(seq: np.ndarray) -> list:
    xyz = seq[..., :3]
    conf = seq[..., 3] if seq.shape[-1] >= 4 else np.ones(seq.shape[:2])
    return [int(seq.shape[0]),
            float(1.0 - np.isfinite(xyz).all(-1).mean()),
            float((np.abs(xyz) < 1e-9).all(-1).mean()),
            float((conf <= 0).mean())]


def ingest(drive_root, cache_dir, work="/content/har3d_work/msr_action3d",
           push_to_drive=True):
    """Both archives -> msr_action3d.npz (ragged, with per-sequence confidence)."""
    drive_root, cache_dir, work = Path(drive_root), Path(cache_dir), Path(work)
    cache_dir.mkdir(parents=True, exist_ok=True)

    real_arc = _resolve(drive_root, ["MSRAction3DSkeletonReal3D.rar", "*Real3D*"])
    scr_arc = _resolve(drive_root, ["MSRAction3DSkeleton(20joints).rar", "*20joints*"])
    if real_arc is None:
        return {"error": f"Real3D archive not found under {drive_root}"}

    d_real, d_scr = work / "real3d", work / "screen"
    if not find_skeleton_files(d_real):
        log(f"  extracting {real_arc.name} with {extract_any(real_arc, d_real)}")
    if scr_arc is not None and not find_skeleton_files(d_scr):
        log(f"  extracting {scr_arc.name} with {extract_any(scr_arc, d_scr)}")

    def index(root):
        out = {}
        for p in Path(root).rglob("*.txt"):
            k = parse_name(p)
            if k:
                out[k] = p
        return out

    real, scr = index(d_real), index(d_scr) if scr_arc is not None else {}
    log(f"  Real3D sequences: {len(real)}   screen sequences: {len(scr)}")
    if len(real) not in (EXPECTED_FULL, EXPECTED_CONVENTIONAL):
        log(f"  !! {len(real)} sequences; expected 567 (full) or 557 (pre-filtered)")

    intr = {}
    shared = sorted(set(real) & set(scr))
    if shared:
        S, R = [], []
        for k in shared[:40]:
            s_, r_ = load_sequence(scr[k]), load_sequence(real[k])
            if s_ is None or r_ is None:
                continue
            T = min(s_.shape[0], r_.shape[0])
            S.append(s_[:T].reshape(-1, s_.shape[-1]))
            R.append(r_[:T].reshape(-1, r_.shape[-1]))
        if S:
            intr = solve_intrinsics(np.concatenate(S), np.concatenate(R))
            off = abs(abs(intr["fx"]) - KINECT_V1_FX) / KINECT_V1_FX * 100
            log(f"  intrinsics: fx {intr['fx']:.3f}  fy {intr['fy']:.3f}  "
                f"cx {intr['cx']:.3f}  cy {intr['cy']:.3f}  "
                f"depth scale {intr['depth_scale']:.6f}")
            log(f"  fit R2 x {intr['r2_x']:.6f} y {intr['r2_y']:.6f}; "
                f"{off:.3f}% off the canonical Kinect v1 285.63 "
                f"-> 320x{int(round(intr['cy'] * 2))}")
            if min(intr["r2_x"], intr["r2_y"]) < 0.999:
                log("  !! poor pinhole fit -- the two archives may not be the "
                    "same recording; do NOT treat them as interchangeable")

    xyz_all, conf_all, lens, metas, flags, has_conf = [], [], [], [], [], []
    for k in sorted(real):
        seq = load_sequence(real[k])
        if seq is None:
            log(f"  !! unreadable, skipped: {real[k].name}")
            continue
        xyz = seq[..., :3].astype(np.float32)
        c = np.full(xyz.shape[:2], np.nan, np.float32)
        ok = False
        if k in scr:
            s_ = load_sequence(scr[k])
            if s_ is not None and s_.shape[-1] >= 4 and s_.shape[0] == xyz.shape[0]:
                c, ok = s_[..., 3].astype(np.float32), True
        xyz_all.append(xyz)
        conf_all.append(c)
        has_conf.append(ok)
        lens.append(xyz.shape[0])
        metas.append(list(k))
        flags.append(degeneracy(seq))

    if not xyz_all:
        return {"error": "no sequence parsed"}

    out = cache_dir / "msr_action3d.npz"
    np.savez_compressed(
        out,
        xyz=np.concatenate(xyz_all, 0),
        conf=np.concatenate(conf_all, 0),
        has_conf=np.array(has_conf, bool),
        offsets=np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
        meta=np.array(metas, np.int64),
        flags=np.array(flags, np.float64),
        intrinsics=np.array(json.dumps(intr)))
    log(f"  cached -> {out.name} ({out.stat().st_size/1e6:.1f} MB, "
        f"{len(metas)} sequences, {sum(lens):,} frames)")
    log(f"  confidence available for {int(np.sum(has_conf))}/{len(metas)} sequences")

    if push_to_drive:
        (drive_root / "cache").mkdir(parents=True, exist_ok=True)
        import shutil as _sh
        _sh.copy2(out, drive_root / "cache" / out.name)
        log(f"  copied to {drive_root / 'cache' / out.name}")

    fl = np.array(flags)
    return {"n_sequences": len(metas), "n_frames": int(sum(lens)),
            "subjects": sorted({m[1] for m in metas}),
            "n_actions": len({m[0] for m in metas}),
            "intrinsics": intr, "cache": str(out),
            "n_with_confidence": int(np.sum(has_conf)),
            "frac_frames_with_a_dropped_joint": float(fl[:, 2].mean())}


def exclusion_candidates(cache_path, k: int = 10) -> list:
    """
    Rank all 567 by degeneracy to recover the conventional 10 exclusions.
    Reported, never auto-applied: the list goes in the reproducibility
    statement and should be eyeballed before it is fixed.
    """
    z = np.load(cache_path, allow_pickle=False)
    meta, fl = z["meta"], z["flags"]
    n_frames, nonfin, zero_j = fl[:, 0], fl[:, 1], fl[:, 2]
    score = zero_j + nonfin + 0.5 * (n_frames < 20)
    order = np.argsort(-score)
    log(f"  {'sequence':<16}{'frames':>8}{'zero_joints':>14}{'nonfinite':>12}")
    names = []
    for i in order[:k + 5]:
        a, s, e = meta[i]
        nm = f"a{a:02d}_s{s:02d}_e{e:02d}"
        mark = f"  <-- top {k}" if len(names) < k else ""
        log(f"  {nm:<16}{int(n_frames[i]):>8}{zero_j[i]:>14.3f}{nonfin[i]:>12.3f}{mark}")
        if len(names) < k:
            names.append(nm)
    log(f"\n  proposed exclusions ({k}): {', '.join(names)}")
    log("  -> cross-check against a published list before fixing this in the paper")
    return names