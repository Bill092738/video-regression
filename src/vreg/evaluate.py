"""One closed-loop sample: X --G(theta)--> upload --platform--> Y --metric(Y, X)--> score."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from . import ff, metrics, precomp
from .platform import Platform


@dataclass
class EvalResult:
    clip: str
    params: dict
    metrics: dict
    upload_kbps: float
    upload_height: int
    seconds: float
    workdir: str
    cached: bool = False

    def to_json(self) -> dict:
        return asdict(self)


def params_key(params: dict) -> str:
    norm = precomp.normalize(params)
    return hashlib.sha1(json.dumps(norm, sort_keys=True).encode()).hexdigest()[:12]


def evaluate(
    clip: str,
    src: Path,
    params: dict,
    platform: Platform,
    work_root: Path,
    use_lpips: bool | None = None,
    keep_media: bool = False,
    judges=(),
) -> EvalResult:
    """``judges``: report-only metrics (``metrics.JUDGE_METRICS``) to compute as well.

    The cache is keyed by theta only.  A cached result that lacks a requested judge is
    back-filled from the cached playback, without running the platform again.
    """
    params = precomp.normalize(params)
    workdir = work_root / clip / params_key(params)
    result_file = workdir / "result.json"
    if result_file.exists():
        data = json.loads(result_file.read_text())
        if backfill(data["metrics"], workdir / "playback.mp4", src, judges):
            result_file.write_text(json.dumps(data, indent=2))
        data["cached"] = True
        return EvalResult(**data)

    workdir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    upload = precomp.apply(src, params, workdir / "upload.mp4")
    up_info = ff.probe(upload)
    playback = platform.process(upload, workdir)
    m = metrics.compute(playback, src, use_lpips=use_lpips, judges=judges)
    res = EvalResult(
        clip=clip,
        params=params,
        metrics=m,
        upload_kbps=up_info.bitrate / 1000,
        upload_height=up_info.height,
        seconds=time.perf_counter() - t0,
        workdir=str(workdir),
    )
    if not keep_media:
        upload.unlink(missing_ok=True)
    result_file.write_text(json.dumps(res.to_json(), indent=2))
    return res


def backfill(m: dict, playback: Path, src: Path, judges) -> bool:
    """Add the requested judges missing from ``m`` (in place); True if anything was added."""
    missing = [j for j in judges if j not in m and metrics.judge_available(j)]
    if not missing or not playback.exists():
        return False
    m.update(metrics.compute_judges(playback, src, missing))
    return True
