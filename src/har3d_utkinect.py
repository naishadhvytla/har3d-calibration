"""
har3d_utkinect.py -- UTKinect-Action3D ingest.

WHAT THIS DATASET IS
--------------------
20 continuous recordings (10 subjects x 2 episodes). Each recording holds all
ten actions performed back to back; actionLabel.txt gives the [start, end]
FRAME NUMBER of every segment. There is no pre-segmented release -- the clips
are cut here. 199 valid clips, not 200: s10_e02 has `carry:  NaN NaN`.

FOUR TRAPS, ALL CONFIRMED AGAINST THE PUBLISHED FILES
-----------------------------------------------------
(1) FRAME NUMBERS ARE NOT ROW INDICES. Only frames in which the skeleton was
    tracked were recorded, so the numbering jumps and the effective rate is
    ~15 fps against a nominal 30. Slicing rows [start:end] instead of masking
    on frame number roughly DOUBLES every clip. Every slice here masks on the
    frame number column.
(2) ~2% OF FRAMES ARE DUPLICATED, carrying two skeleton records with slightly
    different joint values. The dataset page says either may be used; the
    first is kept and the drop count is reported.
(3) SEGMENTS OVERLAP AND RUN OUT OF ORDER. s01_e01 lists push before pull;
    s10_e02's push [1944,1994] and pull [1982,2062] genuinely share 13 frames.
    Reported per sequence, never silently merged or reordered away.
(4) JOINT ORDER IS THE KINECT SDK BETA ENUM (NUI_SKELETON_POSITION_INDEX), not
    MSR Action3D's order. Unlike DailyActivity3D this ordering is DOCUMENTED,
    so the permutation is known before the matcher runs. EXPECTED_SDK_PERM is
    that ground truth and derive_canonical_permutation is checked against it
    rather than trusted -- the only dataset in this project where a
    ground-truth check on the matcher is possible at all.

There is NO confidence / tracking column, so UTKinect cannot serve RQ4's
per-joint noise proxy. Action3D remains the only source for that.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np

# Canonical modules win when har3d.ipynb has run in this session; the bundled
# fallback is used only when they are absent. Never both.
try:
    from har3d_io import extract_any, log
    import har3d_v2 as _G
    GEOM_SOURCE = 'har3d_v2 + har3d_io (canonical)'
except ImportError:
    import har3d_utk_geom as _G
    from har3d_utk_geom import extract_any, log
    GEOM_SOURCE = 'har3d_utk_geom (bundled fallback)'

# ---------------------------------------------------------------- constants
N_JOINTS = 20
EXPECTED_CLIPS = 199          # 200 nominal, minus s10_e02 carry
EXPECTED_RECORDINGS = 20      # 10 subjects x 2 episodes
MIN_CLIP_FRAMES = 5           # below this a clip cannot be resampled meaningfully
LABEL_URL = "https://cvrc.ece.utexas.edu/KinectDatasets/actionLabel.txt"

# File order in actionLabel.txt. Index+1 becomes the label, matching the other
# caches, because build_tensors() does meta[:, 0] - 1.
ACTIONS = ["walk", "sitDown", "standUp", "pickUp", "carry",
           "throw", "push", "pull", "waveHands", "clapHands"]
ACTION_ID = {a.lower(): i + 1 for i, a in enumerate(ACTIONS)}

# Kinect SDK Beta NUI_SKELETON_POSITION_INDEX -> this project's canonical order.
# Derived from the enum on paper, NOT fitted to data. perm[c] = q, applied as
# xyz_canonical = xyz_query[:, perm]. Identical to the permutation the matcher
# recovered for DailyActivity3D, which is what makes that result checkable.
EXPECTED_SDK_PERM = np.array(
    [4, 8, 2, 1, 12, 16, 0, 5, 9, 6, 10, 7, 11, 13, 17, 14, 18, 15, 19, 3])

SEQ_RE = re.compile(r"^\s*s(\d+)[_-]e(\d+)\s*$", re.IGNORECASE)
SEG_RE = re.compile(r"^\s*([A-Za-z]+)\s*:\s*(\S+)\s+(\S+)\s*$")
FILE_RE = re.compile(r"s(\d+)[_-]e(\d+)", re.IGNORECASE)

# Guard against two live definitions of the skeleton.
assert list(_G.MSR_PARENTS) == [2, 2, 3, 6, 6, 6, -1, 0, 1, 7,
                                8, 9, 10, 4, 5, 13, 14, 15, 16, 2], \
    "the geometry module's MSR_PARENTS is not the canonical corrected tree"


# ------------------------------------------------------------ label parsing
def fetch_labels(dest: Path, url: str = LABEL_URL) -> Path | None:
    """Download actionLabel.txt if it is not already beside the archive."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 100:
        return dest
    try:
        import urllib.request
        log(f"  actionLabel.txt absent -- fetching from {url}")
        urllib.request.urlretrieve(url, dest)
        log(f"  fetched {dest.stat().st_size} bytes")
        return dest
    except Exception as e:                       # offline, 404, blocked proxy
        log(f"  !! could not fetch actionLabel.txt: {type(e).__name__}: {e}")
        log("     Download it by hand and place it beside joints.zip.")
        return None


def parse_action_labels(path) -> tuple[list, list]:
    """
    actionLabel.txt -> (segments, skipped).

    Tokenised, never column-sliced: the shipped file has inconsistent spacing
    ('carry:  1868 2122') and trailing whitespace on many lines.
    """
    segments, skipped, cur = [], [], None
    text = Path(path).read_text(errors="ignore")
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        m = SEQ_RE.match(line)
        if m:                                     # sNN_eNN header
            cur = (int(m.group(1)), int(m.group(2)))
            continue
        m = SEG_RE.match(line)
        if not m:
            skipped.append((lineno, line.strip(), "unparseable line"))
            continue
        if cur is None:
            skipped.append((lineno, line.strip(), "segment before any header"))
            continue
        name, a, b = m.group(1), m.group(2), m.group(3)
        aid = ACTION_ID.get(name.lower())
        if aid is None:
            skipped.append((lineno, line.strip(), f"unknown action {name!r}"))
            continue
        try:
            start, end = int(float(a)), int(float(b))
        except ValueError:                        # the s10_e02 'NaN NaN' carry
            skipped.append((lineno, line.strip(), "NaN / non-numeric range"))
            continue
        if end < start:
            start, end = end, start
        segments.append({"subject": cur[0], "episode": cur[1], "action": aid,
                         "name": name, "start": start, "end": end})
    return segments, skipped


def find_overlaps(segments) -> list:
    """Segment pairs sharing at least one frame. Reported, not repaired."""
    by_seq: dict = {}
    for s in segments:
        by_seq.setdefault((s["subject"], s["episode"]), []).append(s)
    out = []
    for key, segs in sorted(by_seq.items()):
        segs = sorted(segs, key=lambda s: s["start"])
        for a, b in zip(segs, segs[1:]):
            if b["start"] <= a["end"]:
                out.append({"seq": key, "a": a["name"], "b": b["name"],
                            "shared": a["end"] - b["start"] + 1})
    return out


# ------------------------------------------------------------ joint parsing
def read_joint_file(path, n_joints=N_JOINTS):
    """
    One recording -> (frame_numbers (T,), xyz (T, J, 3), n_bad_rows, n_dupes).

    Rows are `frame_no x1 y1 z1 ... x20 y20 z20` = 61 tokens. Anything else is
    counted and dropped rather than raised on, so one malformed line cannot
    take down the ingest. Duplicate frame numbers keep their first record.
    """
    want = 1 + 3 * n_joints
    frames, rows, bad = [], [], 0
    for line in Path(path).read_text(errors="ignore").splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != want:
            bad += 1
            continue
        try:
            vals = [float(v) for v in parts]
        except ValueError:
            bad += 1
            continue
        frames.append(int(vals[0]))
        rows.append(vals[1:])

    if not rows:
        return (np.zeros(0, np.int64),
                np.zeros((0, n_joints, 3), np.float32), bad, 0)

    f = np.asarray(frames, np.int64)
    x = np.asarray(rows, np.float32).reshape(-1, n_joints, 3)
    _, first = np.unique(f, return_index=True)    # first record of each frame
    first.sort()
    n_dupes = len(f) - len(first)
    return f[first], x[first], bad, n_dupes


def index_recordings(root: Path) -> dict:
    """(subject, episode) -> path. Layout-agnostic, case-insensitive."""
    out = {}
    for p in Path(root).rglob("*.txt"):
        if "actionlabel" in p.name.lower():
            continue
        m = FILE_RE.search(p.name)
        if m:
            out[(int(m.group(1)), int(m.group(2)))] = p
    return out


# ------------------------------------------------------------------ ingest
def ingest(drive_root, cache_dir, work="/content/har3d_work/utkinect",
           push_to_drive=True, check_perm=True):
    """
    joints.zip + actionLabel.txt -> utkinect.npz, in the same ragged format as
    msr_action3d.npz and msr_daily.npz, so build_tensors() consumes it unchanged.

    Returns a result dict. Any 'error' key makes mark_done() refuse the stage.
    """
    drive_root, cache_dir, work = Path(drive_root), Path(cache_dir), Path(work)
    cache_dir.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    log(f"  geometry from: {GEOM_SOURCE}")

    # -- 1. locate and extract ------------------------------------------------
    zp = None
    for pat in ("joints.zip", "*joint*.zip", "*UTKinect*.zip", "*utkinect*.zip"):
        hits = sorted(drive_root.glob(pat))
        if hits:
            zp = hits[0]
            break
    if zp is None:
        return {"error": f"no joints archive under {drive_root} "
                         f"(expected joints.zip)"}

    if not index_recordings(work):
        try:
            tool = extract_any(zp, work)
            log(f"  extracted {zp.name} with {tool}")
        except Exception as e:
            return {"error": f"extraction failed for {zp.name}: {e}"}

    rec = index_recordings(work)
    log(f"  recordings found: {len(rec)} (expected {EXPECTED_RECORDINGS})")
    if not rec:
        present = sorted({p.suffix.lower() for p in work.rglob('*') if p.is_file()})
        return {"error": "no sNN_eNN .txt recordings after extraction",
                "suffixes_present": present}

    # -- 2. labels ------------------------------------------------------------
    lbl = fetch_labels(drive_root / "actionLabel.txt")
    if lbl is None:
        return {"error": "actionLabel.txt missing and not fetchable; "
                         "without it there are no labelled clips"}
    segments, skipped = parse_action_labels(lbl)
    log(f"  label segments: {len(segments)} usable, {len(skipped)} skipped")
    for s in skipped:
        log(f"    line {s[0]}: {s[1]!r} -- {s[2]}")

    overlaps = find_overlaps(segments)
    if overlaps:
        log(f"  !! {len(overlaps)} segment pairs share frames "
            f"(boundary frames appear in two clips):")
        for o in overlaps:
            log(f"    s{o['seq'][0]:02d}_e{o['seq'][1]:02d}: "
                f"{o['a']} / {o['b']} share {o['shared']} frame(s)")

    # -- 3. cut the clips -----------------------------------------------------
    xyz_all, lens, metas, flags = [], [], [], []
    dropped, tot_dupes, tot_bad = [], 0, 0
    cache_frames: dict = {}

    for seg in sorted(segments, key=lambda s: (s["subject"], s["episode"],
                                               s["action"])):
        key = (seg["subject"], seg["episode"])
        if key not in rec:
            dropped.append((seg["name"], key, "no recording file"))
            continue
        if key not in cache_frames:
            f, x, bad, dup = read_joint_file(rec[key])
            cache_frames[key] = (f, x)
            tot_dupes += dup
            tot_bad += bad
        f, x = cache_frames[key]
        if len(f) == 0:
            dropped.append((seg["name"], key, "recording parsed to 0 frames"))
            continue

        # THE critical line: mask on frame NUMBER, never on row index.
        m = (f >= seg["start"]) & (f <= seg["end"])
        clip = x[m]
        if len(clip) < MIN_CLIP_FRAMES:
            dropped.append((seg["name"], key,
                            f"only {len(clip)} frames in [{seg['start']},"
                            f"{seg['end']}]"))
            continue

        zero = (np.abs(clip) < 1e-9).all(-1)
        xyz_all.append(clip)
        lens.append(len(clip))
        metas.append([seg["action"], seg["subject"], seg["episode"]])
        flags.append([len(clip),
                      float(1.0 - np.isfinite(clip).all(-1).mean()),
                      float(zero.mean()),
                      float("nan")])           # no confidence column exists

    if not xyz_all:
        return {"error": "every segment failed to cut", "dropped": dropped[:10]}

    log(f"  cut {len(xyz_all)} clips "
        f"({tot_dupes} duplicate frames dropped, {tot_bad} malformed rows)")
    for d in dropped:
        log(f"    dropped {d[0]} s{d[1][0]:02d}_e{d[1][1]:02d}: {d[2]}")

    # -- 4. parse plausibility, BEFORE any permutation ------------------------
    probe = np.concatenate(xyz_all[:60], 0)
    score, med = _G.score_parse(probe)
    ext = _G.body_extent(probe)
    log(f"  parse score {score:.2f} of 1.00, median edge {med:.3f} m, "
        f"body span {ext[1]:.2f} m")
    if score < 0.8:
        return {"error": f"parse does not look like a human skeleton "
                         f"(score {score:.2f}, median edge {med:.3f} m). "
                         f"Inspect a raw file before going further."}

    # -- 5. joint order, checked against the documented SDK enum --------------
    perm = EXPECTED_SDK_PERM.copy()
    perm_info = {"source": "SDK enum (assumed, UNCHECKED)"}
    ref = cache_dir / "msr_action3d.npz"
    if ref.exists():
        rz = np.load(ref, allow_pickle=False)
        ro = rz["offsets"]
        ref_xyz = rz["xyz"][ro[0]:ro[min(120, len(ro) - 1)]]
        recovered, perm_info = _G.derive_canonical_permutation(probe, ref_xyz)
        perm_info["source"] = "distance-signature matching"
        agrees = bool(np.array_equal(recovered, EXPECTED_SDK_PERM))
        perm_info["matches_sdk_enum"] = agrees
        if agrees:
            log("  joint order: matcher output EQUALS the documented Kinect SDK "
                "enum permutation -- ground-truth agreement")
        else:
            log(f"  !! matcher returned {recovered.tolist()}")
            log(f"     SDK enum predicts  {EXPECTED_SDK_PERM.tolist()}")
            if check_perm:
                return {"error": "recovered permutation disagrees with the "
                                 "documented Kinect SDK joint order. One of the "
                                 "two is wrong -- do NOT relax this to proceed."}
            log("     check_perm=False, using the matcher's answer")
        perm = recovered
    else:
        log("  !! msr_action3d.npz absent -- applying the SDK enum permutation "
            "UNCHECKED. Run har3d.ipynb Stage 2 for the cross-check.")

    xyz_all = [c[:, perm] for c in xyz_all]

    # -- 6. topology gate on the permuted result ------------------------------
    try:
        _G.verify_topology(xyz_all)
    except AssertionError as e:
        return {"error": f"topology gate failed after permutation: "
                         f"{str(e).splitlines()[0]}"}

    # -- 7. write the cache ---------------------------------------------------
    out = cache_dir / "utkinect.npz"
    n = len(metas)
    np.savez_compressed(
        out,
        xyz=np.concatenate(xyz_all, 0).astype(np.float32),
        conf=np.full((sum(lens), N_JOINTS), np.nan, np.float32),
        has_conf=np.zeros(n, bool),          # no confidence channel in this set
        offsets=np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
        meta=np.array(metas, np.int64),
        flags=np.array(flags, np.float64),
        joint_perm=np.asarray(perm))
    log(f"  cached -> {out.name} ({out.stat().st_size / 1e6:.1f} MB, "
        f"{n} clips, {sum(lens):,} frames)")

    if push_to_drive:
        (drive_root / "cache").mkdir(parents=True, exist_ok=True)
        shutil.copy2(out, drive_root / "cache" / out.name)
        log(f"  copied to {drive_root / 'cache' / out.name}")

    res = {"n_sequences": n, "n_frames": int(sum(lens)),
           "subjects": sorted({m[1] for m in metas}),
           "n_actions": len({m[0] for m in metas}),
           "duplicate_frames_dropped": tot_dupes,
           "malformed_rows": tot_bad,
           "overlapping_segment_pairs": len(overlaps),
           "joint_order": perm_info, "cache": str(out)}
    if n != EXPECTED_CLIPS:
        res["warning"] = f"{n} clips, expected {EXPECTED_CLIPS}"
        log(f"  !! {res['warning']}")
    if dropped:
        res["dropped"] = dropped[:10]
    return res