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
    keep_media: bool = True,
) -> EvalResult:
    params = precomp.normalize(params)
    workdir = work_root / clip / params_key(params)
    result_file = workdir / "result.json"
    if result_file.exists():
        data = json.loads(result_file.read_text())
        data["cached"] = True
        return EvalResult(**data)

    workdir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    upload = precomp.apply(src, params, workdir / "upload.mp4")
    up_info = ff.probe(upload)
    playback = platform.process(upload, workdir)
    m = metrics.compute(playback, src, use_lpips=use_lpips)
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
