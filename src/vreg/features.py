"""Content features of the master clips (P1, docs/P1-plan.md), used to assign content classes.

* ``si``, ``ti``: spatial and temporal information (ITU-T P.910, ffmpeg ``siti`` filter), as
  P.910 defines them, i.e. the maximum over frames; the means are recorded too.
* ``flat_frac``: fraction of 16x16 luma blocks whose standard deviation is below 2.
* ``dark_frac``: fraction of luma samples below 40.
* ``mean_luma``.

The block and luma statistics use every 3rd frame (30 frames of a 3 s clip).
"""

from __future__ import annotations

import csv
import re
import subprocess
from pathlib import Path

import numpy as np

from . import ff

COLUMNS = ("name", "si", "ti", "si_mean", "ti_mean", "flat_frac", "dark_frac", "mean_luma")
FLAT_STD = 2.0
DARK_LUMA = 40
BLOCK = 16
STEP = 3


def siti(path: Path) -> dict:
    proc = subprocess.run([ff.ffmpeg_bin(), "-hide_banner", "-nostdin", "-i", str(path),
                           "-vf", "siti=print_summary=1", "-f", "null", "-"], capture_output=True, text=True)
    if proc.returncode != 0:
        raise ff.FFmpegError(f"siti failed on {path}: {proc.stderr[-1000:]}")
    out = {}
    for key, name in (("Spatial Information", "si"), ("Temporal Information", "ti")):
        m = re.search(key + r":\s*Average:\s*([\d.]+)\s*Max:\s*([\d.]+)", proc.stderr)
        if not m:
            raise ff.FFmpegError(f"no {key} in the siti summary for {path}")
        out[f"{name}_mean"], out[name] = float(m.group(1)), float(m.group(2))
    return out


def read_luma(path: Path, step: int = STEP) -> np.ndarray:
    """(N, H, W) uint8 luma of every ``step``-th frame."""
    info = ff.probe(path)
    proc = subprocess.run([ff.ffmpeg_bin(), "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(path),
                           "-vf", f"select='not(mod(n\\,{step}))'", "-fps_mode", "passthrough",
                           "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True)
    if proc.returncode != 0:
        raise ff.FFmpegError(f"decode failed on {path}: {proc.stderr.decode()[-1000:]}")
    return np.frombuffer(proc.stdout, np.uint8).reshape(-1, info.height, info.width)


def luma_stats(y: np.ndarray) -> dict:
    n, h, w = y.shape
    hb, wb = h // BLOCK, w // BLOCK
    blocks = y[:, :hb * BLOCK, :wb * BLOCK].astype(np.float32).reshape(n, hb, BLOCK, wb, BLOCK)
    std = blocks.std(axis=(2, 4))
    return {"flat_frac": float((std < FLAT_STD).mean()), "dark_frac": float((y < DARK_LUMA).mean()),
            "mean_luma": float(y.mean())}


def clip_features(name: str, path: Path) -> dict:
    return {"name": name, **siti(path), **luma_stats(read_luma(path))}


def write_csv(rows: list[dict], out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k]) for k in COLUMNS})
    return out


def read_csv(path: Path) -> dict[str, dict]:
    with Path(path).open(newline="") as f:
        return {r["name"]: {k: (v if k == "name" else float(v)) for k, v in r.items()} for r in csv.DictReader(f)}
