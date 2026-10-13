"""Quality metrics between the original source X and the playback Y.

The playback is rescaled to the source resolution (bicubic, like a player) before
comparison.  Metrics:

* ``vmaf`` / ``vmaf_neg`` -- VMAF; the NEG ("no enhancement gain") model is the
  default target because plain VMAF is known to reward over-sharpening, which
  is exactly the kind of loophole a black-box optimiser will find.
* ``psnr_y``, ``ssim`` -- classic fidelity metrics.
* ``cambi`` -- Netflix's banding detector (lower is better).
* ``lpips`` -- learned perceptual distance (lower is better), GPU via torch.

Report-only judges (:data:`JUDGE_METRICS`), never part of ``composite`` and only
computed when asked for, so a gain on the objective can be checked against
metrics the optimiser did not see:

* ``ssimulacra2`` -- SSIMULACRA2 (higher is better), CPU.
* ``dists`` -- DISTS deep structure/texture distance (lower is better), GPU via piq.
* ``xpsnr`` -- luma XPSNR from ffmpeg's ``xpsnr`` filter (higher is better).
"""

from __future__ import annotations

import functools
import json
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import ff

LPIPS_WEIGHT = 40.0
CAMBI_WEIGHT = 2.0

HIGHER_IS_BETTER = {"vmaf": True, "vmaf_neg": True, "psnr_y": True, "ssim": True,
                    "cambi": False, "lpips": False, "composite": True,
                    "ssimulacra2": True, "dists": False, "xpsnr": True}
JUDGE_METRICS = ("ssimulacra2", "dists", "xpsnr")


def objective_value(m: dict, objective: str) -> float:
    if objective == "composite":
        return composite(m)
    v = m[objective]
    return v if HIGHER_IS_BETTER[objective] else -v


def composite(m: dict) -> float:
    """Single scalar to maximise: VMAF-NEG penalised by perceptual distance and banding."""
    base = m.get("vmaf_neg", m.get("ssim", 0.0) * 100)
    return base - LPIPS_WEIGHT * m.get("lpips", 0.0) - CAMBI_WEIGHT * m.get("cambi", 0.0)


def _vmaf_metrics(dist: Path, ref: Path, w: int, h: int, threads: int) -> dict:
    binary = ff.vmaf_ffmpeg_bin()
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "vmaf.json"
        graph = (
            f"[0:v]scale={w}:{h}:flags=bicubic,setpts=PTS-STARTPTS[d];"
            "[1:v]setpts=PTS-STARTPTS[r];"
            "[d][r]libvmaf=model='version=vmaf_v0.6.1\\:name=vmaf|version=vmaf_v0.6.1neg\\:name=vmaf_neg'"
            ":feature='name=psnr|name=float_ssim|name=cambi'"
            f":log_fmt=json:log_path={log}:n_threads={threads}"
        )
        ff.run(["-i", str(dist), "-i", str(ref), "-lavfi", graph, "-f", "null", "-"], binary=binary)
        pooled = json.loads(log.read_text())["pooled_metrics"]
    names = {"vmaf": "vmaf", "vmaf_neg": "vmaf_neg", "psnr_y": "psnr_y", "ssim": "float_ssim", "cambi": "cambi"}
    # cambi is skipped by libvmaf for tiny frames
    return {k: pooled[v]["mean"] for k, v in names.items() if v in pooled}


def _basic_metrics(dist: Path, ref: Path, w: int, h: int) -> dict:
    """Fallback when no libvmaf-enabled ffmpeg is available."""
    with tempfile.TemporaryDirectory() as td:
        ps, ss = Path(td) / "psnr.log", Path(td) / "ssim.log"
        graph = (
            f"[0:v]scale={w}:{h}:flags=bicubic,setpts=PTS-STARTPTS,split[d1][d2];"
            "[1:v]setpts=PTS-STARTPTS,split[r1][r2];"
            f"[d1][r1]psnr=stats_file={ps}[o1];[d2][r2]ssim=stats_file={ss}[o2]"
        )
        ff.run(["-i", str(dist), "-i", str(ref), "-filter_complex", graph,
                "-map", "[o1]", "-f", "null", "-", "-map", "[o2]", "-f", "null", "-"])
        psnr = [float(x) for x in re.findall(r"psnr_y:([\d.]+|inf)", ps.read_text())]
        ssim = [float(x) for x in re.findall(r" Y:([\d.]+)", ss.read_text())]
    return {"psnr_y": float(np.mean(np.minimum(psnr, 100))), "ssim": float(np.mean(ssim))}


@functools.lru_cache(maxsize=1)
def _lpips_model():
    import warnings

    import lpips
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # torchvision 'pretrained' deprecation
        model = lpips.LPIPS(net="alex", verbose=False)
    return model.to(dev).eval(), dev


@functools.lru_cache(maxsize=8)
def _ref_frames(ref: str, step: int) -> np.ndarray:
    return ff.read_frames(ref, step=step)


def lpips_available() -> bool:
    try:
        import lpips  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def lpips_score(dist: Path, ref: Path, step: int = 3, batch: int = 6) -> float:
    import torch

    model, dev = _lpips_model()
    d, r = _frame_pair(dist, ref, step)
    n = len(d)
    vals = []
    with torch.no_grad():
        for i in range(0, n, batch):
            a = torch.from_numpy(d[i:i + batch]).to(dev).permute(0, 3, 1, 2).float() / 127.5 - 1
            b = torch.from_numpy(r[i:i + batch]).to(dev).permute(0, 3, 1, 2).float() / 127.5 - 1
            vals.append(model(a, b).flatten().cpu())
    return float(torch.cat(vals).mean())


def _frame_pair(dist: Path, ref: Path, step: int) -> tuple[np.ndarray, np.ndarray]:
    r = _ref_frames(str(ref), step)
    d = ff.read_frames(dist, size=(r.shape[2], r.shape[1]), step=step)
    n = min(len(r), len(d))
    return d[:n], r[:n]


def _ssimulacra2_frame(dist: np.ndarray, ref: np.ndarray) -> float:
    """SSIMULACRA2 of one RGB frame pair (same steps as ssimulacra2.compute_ssimulacra2, on arrays)."""
    from ssimulacra2 import ssimulacra2 as s2

    def xyb(lin):
        return s2.make_positive_xyb(s2.linear_rgb_to_xyb(lin))

    lin1, lin2 = s2.srgb_to_linear(ref.astype(np.float64)), s2.srgb_to_linear(dist.astype(np.float64))
    img1, img2 = xyb(lin1), xyb(lin2)
    ms = s2.Msssim()
    for scale in range(s2.kNumScales):
        if img1.shape[0] < 8 or img1.shape[1] < 8:
            break
        mu1, mu2 = s2.blur_image(img1), s2.blur_image(img2)
        sc = s2.MsssimScale()
        sc.avg_ssim = s2.ssim_map(mu1, mu2, s2.blur_image(img1 * img1), s2.blur_image(img2 * img2),
                                  s2.blur_image(img1 * img2))
        sc.avg_edgediff = s2.edge_diff_map(img1, mu1, img2, mu2)
        ms.scales.append(sc)
        if scale < s2.kNumScales - 1:
            lin1, lin2 = s2.downsample(lin1, 2, 2), s2.downsample(lin2, 2, 2)
            img1, img2 = xyb(lin1), xyb(lin2)
    return float(ms.score())


def ssimulacra2_score(dist: Path, ref: Path, step: int = 3) -> float:
    d, r = _frame_pair(dist, ref, step)
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        return float(np.mean(list(pool.map(_ssimulacra2_frame, d, r))))


@functools.lru_cache(maxsize=1)
def _dists_model():
    import warnings

    import piq
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # torchvision 'pretrained' deprecation
        model = piq.DISTS(reduction="none")
    return model.to(dev).eval(), dev


def dists_score(dist: Path, ref: Path, step: int = 3, batch: int = 6) -> float:
    import torch

    model, dev = _dists_model()
    d, r = _frame_pair(dist, ref, step)
    vals = []
    with torch.no_grad():
        for i in range(0, len(d), batch):
            a = torch.from_numpy(d[i:i + batch]).to(dev).permute(0, 3, 1, 2).float() / 255
            b = torch.from_numpy(r[i:i + batch]).to(dev).permute(0, 3, 1, 2).float() / 255
            vals.append(model(a, b).flatten().cpu())
    return float(torch.cat(vals).mean())


def xpsnr_score(dist: Path, ref: Path) -> float:
    info = ff.probe(ref)
    w, h = info.width, info.height
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "xpsnr.log"
        graph = (f"[0:v]scale={w}:{h}:flags=bicubic,format=yuv420p,setpts=PTS-STARTPTS[d];"
                 f"[1:v]format=yuv420p,setpts=PTS-STARTPTS[r];[d][r]xpsnr=stats_file={log}")
        ff.run(["-i", str(dist), "-i", str(ref), "-lavfi", graph, "-f", "null", "-"], binary=ff.xpsnr_ffmpeg_bin())
        m = re.search(r"XPSNR average.*?y:\s*([\d.]+|inf)", log.read_text())
    if not m:
        raise ff.FFmpegError(f"no XPSNR average in the xpsnr stats for {dist}")
    return min(float(m.group(1)), 100.0)


def judge_available(name: str) -> bool:
    if name == "ssimulacra2":
        try:
            import ssimulacra2  # noqa: F401
        except ImportError:
            return False
        return True
    if name == "dists":
        try:
            import piq  # noqa: F401
        except ImportError:
            return False
        return lpips_available()
    if name == "xpsnr":
        return ff.xpsnr_ffmpeg_bin() is not None
    raise KeyError(name)


def compute_judges(dist: Path, ref: Path, names) -> dict:
    """The requested report-only metrics that are available here (unavailable ones are skipped)."""
    fns = {"ssimulacra2": ssimulacra2_score, "dists": dists_score, "xpsnr": xpsnr_score}
    return {n: fns[n](dist, ref) for n in names if judge_available(n)}


def compute(dist: Path, ref: Path, use_lpips: bool | None = None, threads: int = 6, judges=()) -> dict:
    ref_info = ff.probe(ref)
    dist_info = ff.probe(dist)
    w, h = ref_info.width, ref_info.height
    if ff.vmaf_ffmpeg_bin():
        m = _vmaf_metrics(dist, ref, w, h, threads)
    else:
        m = _basic_metrics(dist, ref, w, h)
    if use_lpips is None:
        use_lpips = lpips_available()
    if use_lpips:
        m["lpips"] = lpips_score(dist, ref)
    m["playback_kbps"] = dist_info.bitrate / 1000
    m["playback_codec"] = dist_info.codec
    m["playback_height"] = dist_info.height
    m["composite"] = composite(m)
    m.update(compute_judges(dist, ref, judges))
    return m
