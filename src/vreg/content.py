"""Test content: synthetic stress clips and real Xiph test sequences.

All clips are normalised to the same "master" format so experiments are
comparable: 1280x720, 30 fps, 3 s (90 frames), near-lossless x264 (qp 0).
"""

from __future__ import annotations

from pathlib import Path

from . import ff

WIDTH, HEIGHT, FPS, SECONDS = 1280, 720, 30, 3
MASTER_CODEC = ["-c:v", "libx264", "-preset", "veryfast", "-qp", "0", "-pix_fmt", "yuv420p"]

XIPH_BASE = "https://media.xiph.org/video/derf/y4m/"
# name -> (file on media.xiph.org, why it is interesting)
XIPH_CLIPS = {
    "park_joy": ("park_joy_420_720p50.y4m", "dense foliage, fast pan: texture loss / denoise"),
    "old_town_cross": ("old_town_cross_420_720p50.y4m", "large smooth sky: banding"),
    "ducks_take_off": ("ducks_take_off_420_720p50.y4m", "water ripples: high-frequency temporal detail"),
    "in_to_tree": ("in_to_tree_420_720p50.y4m", "zoom on tree: mixed smooth/detailed"),
}

# name -> (lavfi source graph, description)
SYNTHETIC_CLIPS = {
    "sky_gradient": (
        f"gradients=s={WIDTH}x{HEIGHT}:r={FPS}:d={SECONDS}:c0=0x203a60:c1=0x8fb3d9:"
        "x0=0:y0=0:x1=0:y1=720:speed=0.004:n=2,format=yuv420p",
        "slowly moving smooth gradient: banding",
    ),
    "mandelbrot": (
        f"mandelbrot=s={WIDTH}x{HEIGHT}:r={FPS}:end_scale=0.05:end_pts={SECONDS * 3},format=yuv420p",
        "zooming fractal: fine detail everywhere",
    ),
    "testsrc": (
        f"testsrc2=s={WIDTH}x{HEIGHT}:r={FPS}:d={SECONDS},format=yuv420p",
        "moving text/edges/colour bars",
    ),
}


def make_synthetic(name: str, out_dir: Path) -> Path:
    graph, _ = SYNTHETIC_CLIPS[name]
    out = out_dir / f"{name}.mkv"
    ff.run(["-f", "lavfi", "-i", graph, "-frames:v", str(FPS * SECONDS), *MASTER_CODEC, str(out)])
    return out


def fetch_xiph(name: str, out_dir: Path) -> Path:
    """Stream just the first frames of a Xiph sequence (no full ~700 MB download)."""
    fname, _ = XIPH_CLIPS[name]
    out = out_dir / f"{name}.mkv"
    src_frames = int(SECONDS * 50)  # the 720p Xiph clips are 50 fps
    ff.run([
        "-i", XIPH_BASE + fname, "-frames:v", str(src_frames),
        "-vf", f"fps={FPS},scale={WIDTH}:{HEIGHT}", "-frames:v", str(FPS * SECONDS),
        *MASTER_CODEC, str(out),
    ])
    return out


def make_tiny(out: Path, seconds: float = 1.0) -> Path:
    """A very small clip for unit tests."""
    ff.run(["-f", "lavfi", "-i", f"testsrc2=s=192x108:r=15:d={seconds},format=yuv420p", *MASTER_CODEC, str(out)])
    return out


def ensure_clips(names: list[str], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in names:
        out = out_dir / f"{name}.mkv"
        if not out.exists():
            if name in SYNTHETIC_CLIPS:
                make_synthetic(name, out_dir)
            elif name in XIPH_CLIPS:
                fetch_xiph(name, out_dir)
            else:
                raise KeyError(f"unknown clip {name!r}; known: {sorted(all_clip_names())}")
        paths[name] = out
    return paths


def all_clip_names() -> list[str]:
    return [*SYNTHETIC_CLIPS, *XIPH_CLIPS]
