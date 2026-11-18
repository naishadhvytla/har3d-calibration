"""
har3d_io.py -- archive extraction and tolerant skeleton-text parsing.

Shared by every ingest stage. Two things live here because both MSR datasets
and NTU tripped over them:

ARCHIVES. MSRDailyActivity3D.zip is a zip OF ZIPS -- 9 stored members, the 8
packs plus sample code -- and the packs themselves use Deflate64 (method 9),
which Python's zipfile cannot read and which Info-ZIP `unzip` 6.00 also cannot
read. 7-Zip can. So extraction tries zipfile, then 7z, then unzip, then unar,
installing each on demand, and nested archives are expanded WHEREVER they land
rather than only where the caller expected them.

TEXT. Skeleton files interleave per-frame marker lines with fixed-width joint
rows, so np.loadtxt raises on inconsistent columns. read_rows() classifies each
line by token count instead and keeps only the modal width.
"""
from __future__ import annotations

import shutil
import subprocess
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

METHOD_NAMES = {0: "stored", 1: "shrunk", 6: "imploded", 8: "deflate",
                9: "DEFLATE64", 12: "bzip2", 14: "lzma", 98: "ppmd"}
ZIPFILE_SUPPORTS = {0, 8, 12, 14}
ARCHIVE_EXT = {".zip", ".rar", ".7z"}
MEDIA_EXT = {".avi", ".mp4", ".mov", ".bin", ".jpg", ".jpeg", ".png",
             ".mat", ".depth", ".wmv"}


def log(m=""):
    print(m, flush=True)


def inspect_methods(archive: Path) -> dict:
    """Compression histogram, and whether zipfile alone can read it."""
    if Path(archive).suffix.lower() != ".zip":
        return {"counts": {}, "names": {"(not a zip)": 1}, "zipfile_can_read": False}
    try:
        with zipfile.ZipFile(archive) as zf:
            counts = Counter(i.compress_type for i in zf.infolist())
    except zipfile.BadZipFile:
        return {"counts": {}, "names": {"(unreadable)": 1}, "zipfile_can_read": False}
    return {"counts": dict(counts),
            "names": {METHOD_NAMES.get(k, f"method {k}"): v for k, v in counts.items()},
            "zipfile_can_read": set(counts) <= ZIPFILE_SUPPORTS}


def ensure_tool(name: str, apt_pkg: str) -> bool:
    if shutil.which(name):
        return True
    subprocess.run(f"apt-get -qq update && apt-get -qq install -y {apt_pkg}",
                   shell=True, check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return shutil.which(name) is not None


def extract_any(archive: Path, dest: Path, quiet: bool = False) -> str:
    """
    Extract into `dest` with whichever tool can read it. Returns the tool used;
    raises listing every attempt. 7z leads the fallbacks because it is the only
    one of them that reads Deflate64 and RAR.
    """
    archive, dest = Path(archive), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    errs = []
    info = inspect_methods(archive)
    if info["zipfile_can_read"]:
        try:
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(dest)
            return "zipfile"
        except (NotImplementedError, RuntimeError, zipfile.BadZipFile) as e:
            errs.append(f"zipfile: {e}")
    else:
        errs.append(f"zipfile: cannot read {info['names']}")
        if not quiet:
            log(f"    {archive.name}: {info['names']} -- using a system tool")
    for name, cmd, pkg in (
            ("7z", ["7z", "x", "-y", f"-o{dest}", str(archive)], "p7zip-full"),
            ("unzip", ["unzip", "-q", "-o", str(archive), "-d", str(dest)], "unzip"),
            ("unar", ["unar", "-quiet", "-force-overwrite", "-o", str(dest),
                      str(archive)], "unar")):
        if not ensure_tool(cmd[0], pkg):
            errs.append(f"{name}: unavailable")
            continue
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            return name
        errs.append(f"{name}: rc={r.returncode} {(r.stderr or r.stdout).strip()[:160]}")
    raise RuntimeError(f"could not extract {archive.name}\n  " + "\n  ".join(errs))


def expand_archives(root: Path, skip=("samplecode",), max_depth=4) -> int:
    """Recursively expand nested archives under `root`, deleting each after."""
    root = Path(root)
    total = 0
    for _ in range(max_depth):
        found = [p for p in root.rglob("*")
                 if p.is_file() and p.suffix.lower() in ARCHIVE_EXT
                 and not any(s in p.name.lower() for s in skip)]
        if not found:
            break
        for p in found:
            try:
                tool = extract_any(p, p.parent, quiet=True)
                p.unlink()
                total += 1
                log(f"    expanded {p.name} with {tool}")
            except (zipfile.BadZipFile, RuntimeError) as e:
                log(f"    !! leaving in place: {p.name}\n       {e}")
    return total


def reclaim_media(root: Path) -> int:
    """Delete RGB/depth so the bulk never has to live anywhere permanent."""
    freed = 0
    for f in Path(root).rglob("*"):
        if f.is_file() and f.suffix.lower() in MEDIA_EXT:
            freed += f.stat().st_size
            f.unlink()
    return freed


def read_rows(path: Path, widths=(3, 4)) -> tuple:
    """
    Numeric rows of the modal token width, marker/header lines discarded.
    Returns (rows, width). Empty array rather than an exception, so one bad file
    cannot take down an ingest of several hundred.
    """
    buckets = {w: [] for w in widths}
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) not in buckets:
                continue
            try:
                buckets[len(parts)].append([float(v) for v in parts])
            except ValueError:
                continue
    w = max(buckets, key=lambda k: len(buckets[k]))
    if not buckets[w]:
        return np.zeros((0, max(widths))), 0
    return np.array(buckets[w], np.float64), w


def find_skeleton_files(root: Path, needle="skeleton") -> list:
    """Case-insensitive, layout-agnostic."""
    return sorted(p for p in Path(root).rglob("*")
                  if p.is_file() and p.suffix.lower() == ".txt"
                  and needle in p.name.lower())