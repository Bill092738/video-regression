"""Parametric pre-compensation generator G(X; theta), implemented as an ffmpeg filter chain.

Every parameter is continuous (discrete choices are snapped inside
:func:`build_filter`), so the same space works for TPE, GP-BO and CMA-ES.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import ff


@dataclass(frozen=True)
class Param:
    name: str
    low: float
    high: float
    identity: float  # value that makes the step a no-op
    doc: str


SPACE: tuple[Param, ...] = (
    Param("denoise", 0.0, 6.0, 0.0, "hqdn3d strength before upload (stop wasting bits on noise)"),
    Param("contrast", 0.90, 1.15, 1.0, "eq contrast (compensate platform flattening)"),
    Param("saturation", 0.85, 1.25, 1.0, "eq saturation"),
    Param("upscale", 1.0, 2.0, 1.0, "upload resolution factor, snapped to {1, 1.5, 2}: probes ladder tiers"),
    Param("sharpen", 0.0, 2.0, 0.0, "unsharp-mask amount on luma (pre-emphasis against low-pass/NR)"),
    Param("sharpen_radius", 1.0, 3.0, 1.0, "unsharp kernel radius -> size 3/5/7"),
    Param("grain", 0.0, 15.0, 0.0, "temporal luma grain / dither strength (anti-banding)"),
)
PARAM_NAMES = tuple(p.name for p in SPACE)
IDENTITY = {p.name: p.identity for p in SPACE}

UPSCALE_LEVELS = (1.0, 1.5, 2.0)
UPLOAD_CODEC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "10", "-pix_fmt", "yuv420p"]


def snap_upscale(x: float) -> float:
    return min(UPSCALE_LEVELS, key=lambda lv: abs(lv - x))


def normalize(params: dict) -> dict:
    """Fill defaults, clip to bounds, snap discrete parameters; canonical form for caching."""
    out = {}
    for p in SPACE:
        v = float(params.get(p.name, p.identity))
        v = min(max(v, p.low), p.high)
        out[p.name] = round(v, 4)
    out["upscale"] = snap_upscale(out["upscale"])
    out["sharpen_radius"] = float(round(out["sharpen_radius"]))
    return out


def build_filter(params: dict, src_height: int) -> str:
    p = normalize(params)
    chain = []
    if p["denoise"] >= 0.1:
        d = p["denoise"]
        chain.append(f"hqdn3d={d:.3f}:{0.75 * d:.3f}:{1.5 * d:.3f}:{1.1 * d:.3f}")
    if abs(p["contrast"] - 1) > 1e-3 or abs(p["saturation"] - 1) > 1e-3:
        chain.append(f"eq=contrast={p['contrast']:.4f}:saturation={p['saturation']:.4f}")
    if p["upscale"] > 1.0:
        h = int(round(src_height * p["upscale"] / 2) * 2)
        chain.append(f"scale=-2:{h}:flags=lanczos")
    if p["sharpen"] >= 0.02:
        size = 2 * int(p["sharpen_radius"]) + 1
        chain.append(f"unsharp=lx={size}:ly={size}:la={p['sharpen']:.4f}:cx=3:cy=3:ca=0")
    if p["grain"] >= 0.2:
        chain.append(f"noise=c0s={p['grain']:.3f}:c0f=t+u")
    chain.append("format=yuv420p")
    return ",".join(chain)


def apply(src: Path, params: dict, out: Path) -> Path:
    """Render the pre-compensated upload file."""
    info = ff.probe(src)
    ff.run(["-i", str(src), "-an", "-vf", build_filter(params, info.height), *UPLOAD_CODEC, str(out)])
    return out
