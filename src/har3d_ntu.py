"""
har3d_ntu.py -- NTU RGB+D preprocessing via the CTR-GCN data_gen scripts.

Known traps, all previously hit:
  * git clone refuses a non-empty directory, so cloning AFTER extracting 56,880
    raw files fails silently. Clone first, then move the raw files in.
  * CTR-GCN is 2021 code: np.int (22 sites) and np.string_ (2) were removed in
    NumPy 2.0. Patched with a negative lookahead so np.int64 and np.float32
    survive. Blunt sed would break them.
  * The exclusion-list files open with two prose lines and a blank line. Parse
    by the SsssCcccPpppRrrrAaaa pattern, never by counting lines.
  * The generated tensor layout is (N, 300, 150), not the (N, 3, 300, 25, 2)
    usually quoted.

The released-checkpoint accuracy gate that used to live here is retired: the
CTR-GCN weights were never obtainable. The scale axis now comes from the
training-set ladder in har3d_scale.py, which is methodologically better anyway
-- it holds architecture, recipe and TEST SET fixed so only train n moves.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

CTRGCN_URL = "https://github.com/Uason-Chen/CTR-GCN.git"
NTU60_FILES, NTU120_FILES = 56_880, 114_480
NTU60_MISSING, NTU120_MISSING = 302, 535
XSUB_TRAIN_SUBJECTS = {1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28,
                       31, 34, 35, 38}


def log(m=""):
    print(m, flush=True)


def clone_repo(repo="/content/CTR-GCN", force=False) -> Path:
    """Clone into a guaranteed-empty path. This must happen BEFORE extraction."""
    repo = Path(repo)
    if (repo / ".git").exists() and not force:
        log(f"  [skip] repo already at {repo}")
        return repo
    if repo.exists():
        log(f"  removing non-empty {repo} so the clone can succeed")
        shutil.rmtree(repo)
    r = subprocess.run(["git", "clone", "--depth", "1", "-q", CTRGCN_URL, str(repo)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"clone failed: {r.stderr.strip()}")
    for s in ("get_raw_skes_data.py", "get_raw_denoised_data.py",
              "seq_transformation.py"):
        assert (repo / "data" / "ntu" / s).exists(), f"missing after clone: {s}"
    log("  clone OK; all three data_gen scripts present")
    return repo


def patch_numpy_aliases(repo="/content/CTR-GCN", subdirs=("ntu", "ntu120")) -> int:
    """
    Fix NumPy 2.0 removals, idempotently. The negative lookahead is the
    important part: a blunt s/np.int/int/ rewrites np.int64 into int64.
    """
    repo = Path(repo)
    pats = [(re.compile(r"\bnp\.int(?![0-9_a-zA-Z])"), "int"),
            (re.compile(r"\bnp\.float(?![0-9_a-zA-Z])"), "float"),
            (re.compile(r"\bnp\.bool(?![0-9_a-zA-Z])"), "bool"),
            (re.compile(r"\bnp\.object(?![0-9_a-zA-Z])"), "object"),
            (re.compile(r"\bnp\.string_"), "np.bytes_")]
    n_files = n_sites = n_scanned = 0
    for sub in subdirs:
        d = repo / "data" / sub
        if not d.exists():
            continue
        for f in d.glob("*.py"):
            n_scanned += 1
            src = f.read_text()
            new = src
            for rx, rep in pats:
                new, k = rx.subn(rep, new)
                n_sites += k
            if new != src:
                f.write_text(new)
                n_files += 1
    log(f"  patched {n_sites} NumPy 2.0 removals across {n_files} of "
        f"{n_scanned} scanned files"
        + ("  (0 is expected on a repo already patched)"
           if n_scanned and not n_sites else ""))
    if not n_scanned:
        raise RuntimeError(
            f"no .py files under {repo}/data -- the clone is incomplete. "
            f"Delete /content/CTR-GCN and re-run.")
    missing = []
    for mod, pkg in (("h5py", "h5py"), ("sklearn", "scikit-learn")):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        log(f"  installing {', '.join(missing)} (the data_gen scripts need them)")
        subprocess.run(f"pip install -q {' '.join(missing)}", shell=True,
                       check=False, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    return n_sites


def _free_gb(path="/content") -> float:
    import os
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        st = os.statvfs(p)
        return st.f_bavail * st.f_frsize / 1e9
    except OSError:
        return float("nan")


def place_raw(drive_root, repo="/content/CTR-GCN", ntu120=False, force=False) -> Path:
    """
    Extract raw .skeleton files into the layout the scripts expect. RESUMABLE:
    a session dying mid-extract leaves a partial directory, and treating "any
    files present" as "done" once failed the count check on 31,734 of 56,880.
    Extraction runs whenever the count is short, overwriting.
    """
    import zipfile
    from har3d_io import extract_any

    repo, drive_root = Path(repo), Path(drive_root)
    dest = repo / "data" / "nturgbd_raw"
    skel = dest / "nturgb+d_skeletons"
    dest.mkdir(parents=True, exist_ok=True)
    exp = NTU120_FILES if ntu120 else NTU60_FILES

    def count():
        return sum(1 for _ in skel.glob("*.skeleton")) if skel.exists() else 0

    have = count()
    if have >= exp and not force:
        log(f"  {have:,} .skeleton already in place")
        return skel
    if have:
        log(f"  {have:,} of {exp:,} present -- a previous extraction was "
            f"interrupted; resuming (existing files are overwritten)")

    archives = ["nturgbd_skeletons_s001_to_s017.zip"]
    if ntu120:
        archives.append("nturgbd_skeletons_s018_to_s032.zip")

    members = 0
    for name in archives:
        z = drive_root / name
        if not z.exists():
            log(f"  !! {name} not found in {drive_root}")
            continue
        with zipfile.ZipFile(z) as zf:
            m = sum(1 for x in zf.namelist() if x.endswith(".skeleton"))
        members += m
        log(f"  {name}: {m:,} .skeleton members, "
            f"{z.stat().st_size/1e9:.1f} GB compressed")
    if members and members != exp:
        raise RuntimeError(
            f"the archives hold {members:,} .skeleton members, not {exp:,}. "
            f"The download is incomplete -- re-download before extracting.")

    log(f"  disk free before: {_free_gb(dest):.1f} GB "
        f"(raw .skeleton needs ~15 GB for NTU-60)")
    for name in archives:
        z = drive_root / name
        if not z.exists():
            continue
        t0 = time.time()
        log(f"  extracting {name} ...")
        extract_any(z, dest)
        log(f"    done in {time.time()-t0:.0f}s, {count():,} files so far")
    log(f"  disk free after: {_free_gb(dest):.1f} GB")

    n = count()
    log(f"  raw .skeleton files: {n:,} (expected {exp:,})  "
        f"{'OK' if n == exp else '<-- MISMATCH'}")
    if n != exp:
        raise RuntimeError(
            f"expected {exp:,} .skeleton files, found {n:,} after a full pass.\n"
            f"  The archives do hold {members:,} members, so this is not a bad "
            f"download.\n  Most likely the session ran out of disk "
            f"({_free_gb(dest):.1f} GB free now) or extraction was interrupted "
            f"again.\n  Free space, then re-run this cell -- it resumes.")
    return skel


def check_statistics(repo="/content/CTR-GCN", ntu120=False) -> int:
    """skes_available_name.txt is what the scripts consume. Verify the arithmetic."""
    sub = "ntu120" if ntu120 else "ntu"
    stat = Path(repo) / "data" / sub / "statistics"
    avail = stat / "skes_available_name.txt"
    n = sum(1 for ln in avail.read_text().splitlines() if ln.strip())
    total = NTU120_FILES if ntu120 else NTU60_FILES
    miss = NTU120_MISSING if ntu120 else NTU60_MISSING
    ok = n == total - miss
    log(f"  arithmetic: {total:,} - {miss} = {total-miss:,}   "
        f"available list says {n:,}   {'OK' if ok else '<-- CHECK'}")
    if not ok:
        raise RuntimeError("exclusion arithmetic does not match the shipped list")
    return n


def run_datagen(repo="/content/CTR-GCN", ntu120=False) -> None:
    """The three official stages, in order, from the directory they expect."""
    wd = Path(repo) / "data" / ("ntu120" if ntu120 else "ntu")
    for script in ("get_raw_skes_data.py", "get_raw_denoised_data.py",
                   "seq_transformation.py"):
        log(f"\n  === {script} ===")
        t0 = time.time()
        r = subprocess.run(["python", script], cwd=wd, text=True, capture_output=True)
        for ln in (r.stdout or "").strip().splitlines()[-5:]:
            log("    " + ln)
        if r.returncode != 0:
            for ln in (r.stderr or "").strip().splitlines()[-15:]:
                log("    ! " + ln)
            raise RuntimeError(f"{script} failed (exit {r.returncode})")
        log(f"  OK in {time.time()-t0:.0f}s")


def verify_outputs(repo="/content/CTR-GCN", ntu120=False) -> dict:
    wd = Path(repo) / "data" / ("ntu120" if ntu120 else "ntu")
    out = {}
    for f in sorted(wd.glob("*.npz")):
        z = np.load(f)
        shapes = {k: list(z[k].shape) for k in z.files}
        log(f"\n  {f.name} ({f.stat().st_size/1e9:.2f} GB)")
        for k, s in shapes.items():
            log(f"    {k:<10} {tuple(s)}")
        out[f.name] = shapes
    if not out:
        raise RuntimeError("no .npz produced -- data_gen did not complete")
    return out


def push_to_drive(drive_root, repo="/content/CTR-GCN", ntu120=False) -> None:
    sub = "ntu120" if ntu120 else "ntu"
    src = Path(repo) / "data" / sub
    dest = Path(drive_root) / "preprocessed" / sub
    dest.mkdir(parents=True, exist_ok=True)
    tot = 0
    for f in list(src.glob("*.npz")) + [src / "statistics" / "skes_available_name.txt",
                                        src / "statistics" / "performer.txt"]:
        if f.exists():
            shutil.copy2(f, dest / f.name)
            tot += f.stat().st_size
    log(f"  copied {tot/1e9:.2f} GB -> {dest}")


def find_performer_file(drive_root=None, repo="/content/CTR-GCN", ntu120=False):
    """
    Locate statistics/performer.txt. The Drive copy (written by
    push_to_drive in Stage 7) is checked FIRST, because /content -- and with it
    the cloned repo -- is wiped whenever the runtime changes. The previous
    pipeline read only the repo copy, so the 2s-AGCN ladder, run in a fresh
    GPU runtime, silently fell back to a random validation slice.
    """
    sub = "ntu120" if ntu120 else "ntu"
    cands = []
    if drive_root is not None:
        cands.append(Path(drive_root) / "preprocessed" / sub / "performer.txt")
    cands.append(Path(repo) / "data" / sub / "statistics" / "performer.txt")
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(
        "performer.txt not found in any of: " + ", ".join(map(str, cands))
        + ". Re-run Stage 7 (it copies the statistics files to Drive).")


def _read_performers(src):
    src = Path(src)
    if src.is_dir():
        src = src / "data" / "ntu" / "statistics" / "performer.txt"
    return np.array([int(ln) for ln in src.read_text().split() if ln.strip()])


def test_performers(src, train_subjects=XSUB_TRAIN_SUBJECTS) -> np.ndarray:
    """
    Performer id per row of x_test, in the order seq_transformation wrote.
    The npz stores no sample ids, so they are recovered from the shipped
    statistics file by applying the same X-Sub criterion. `src` is the
    performer.txt path (see find_performer_file) or the repo root.
    """
    perf = _read_performers(src)
    return perf[~np.isin(perf, list(train_subjects))]


def train_performers(src, train_subjects=XSUB_TRAIN_SUBJECTS) -> np.ndarray:
    """Performer id per row of x_train, same ordering argument."""
    perf = _read_performers(src)
    return perf[np.isin(perf, list(train_subjects))]