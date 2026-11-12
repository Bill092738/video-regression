"""Per-clip closed-loop correction (E24, docs/E24-plan.md): feed the measured playback error back into the upload.

    u_k = clip(X + alpha * sum_{i<k} luma(X - Y_i))        Y_i = playback of an earlier upload of this clip

The forward value is always the black box's own output, so no emulator is involved and there is
nothing to exploit. Only luma is corrected: the same value is added to R, G and B, which leaves
the chroma planes unchanged. Each step costs one upload of the clip being corrected.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import ff, metrics, precomp
from .platform import Platform

LOSSLESS = ["-c:v", "libx264", "-preset", "veryfast", "-qp", "0", "-pix_fmt", "yuv444p"]
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)
CHUNK = 8  # frames converted to float at a time


def correct(x: np.ndarray, playbacks: list[np.ndarray], alpha: float) -> np.ndarray:
    """``x`` and every playback: (N, H, W, 3) uint8 RGB at the source size -> the corrected upload frames."""
    n = min(len(x), *(len(y) for y in playbacks))
    out = np.empty_like(x[:n])
    for i in range(0, n, CHUNK):
        xf = x[i:i + CHUNK].astype(np.float32)
        err = sum((xf - y[i:i + CHUNK].astype(np.float32)) @ LUMA for y in playbacks)
        out[i:i + CHUNK] = np.clip(np.rint(xf + alpha * err[..., None]), 0, 255).astype(np.uint8)
    return out


def feedback_key(front: dict, alpha: float, probes: list[Path]) -> str:
    spec = {"front": precomp.normalize(front), "alpha": round(float(alpha), 4),
            "probes": [f"{Path(p).parent.name}/{Path(p).name}" for p in probes]}
    return "fb_" + hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]


def evaluate_feedback(clip: str, src: Path, probes: list[Path], front: dict, alpha: float, platform: Platform,
                      work_root: Path, judges=(), use_lpips: bool | None = None) -> dict:
    """One feedback upload of ``clip``: correct the source with the earlier playbacks ``probes``, send it
    through ``front`` (a theta: the upscale, optionally a sharpening) and the platform, and score it.

    Returns ``{"metrics", "workdir", "mean_abs_residual", "probes", "seconds", "cached"}``; cached per
    (clip, front, alpha, probes) under ``work_root/clip``. The playback is kept as ``playback.mp4`` so it
    can be the probe of a further step.
    """
    front = precomp.normalize(front)
    workdir = work_root / clip / feedback_key(front, alpha, probes)
    result_file = workdir / "result.json"
    if result_file.exists():
        data = json.loads(result_file.read_text())
        missing = [j for j in judges if j not in data["metrics"] and metrics.judge_available(j)]
        if missing and (workdir / "playback.mp4").exists():
            data["metrics"].update(metrics.compute_judges(workdir / "playback.mp4", src, missing))
            result_file.write_text(json.dumps(data, indent=2))
        return data | {"cached": True}

    for p in probes:
        if not Path(p).exists():
            raise FileNotFoundError(f"{clip}: probe playback {p} is missing (was the cache cleaned?)")
    workdir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    x = ff.read_frames(src)
    size = (x.shape[2], x.shape[1])
    u = correct(x, [ff.read_frames(p, size=size) for p in probes], alpha)
    residual = float(np.abs(u.astype(np.int16) - x[:len(u)].astype(np.int16)).mean())
    u_path = ff.write_frames(u, workdir / "u.mkv", ff.probe(src).fps, LOSSLESS)
    del x, u
    upload = precomp.apply(u_path, front, workdir / "upload.mp4")
    playback = platform.process(upload, workdir)
    m = metrics.compute(playback, src, use_lpips=use_lpips, judges=judges)
    for f in (u_path, upload):
        f.unlink(missing_ok=True)
    data = {"metrics": m, "workdir": str(workdir), "mean_abs_residual": residual, "alpha": alpha, "front": front,
            "probes": [str(p) for p in probes], "seconds": time.perf_counter() - t0}
    result_file.write_text(json.dumps(data, indent=2))
    return data | {"cached": False}
