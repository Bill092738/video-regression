"""Test content: synthetic stress clips and real Xiph test sequences.

All clips are normalised to the same "master" format so experiments are
comparable: 1280x720, 30 fps, 3 s (90 frames), near-lossless x264 (qp 0).
"""

from __future__ import annotations

import csv
import functools
import os
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from . import ff

CLIPS_DIR = Path("data/clips")
WIDTH, HEIGHT, FPS, SECONDS = 1280, 720, 30, 3
MASTER_CODEC = ["-c:v", "libx264", "-preset", "veryfast", "-qp", "0", "-pix_fmt", "yuv420p"]
# P1 clip manifest (docs/P1-plan.md); VREG_MANIFEST overrides it, e.g. in tests
MANIFEST = Path("docs/p1/clips.csv")
MANIFEST_COLUMNS = ("name", "source_group", "url_or_path", "start_s", "source_type", "license", "license_url")
SOURCE_TYPES = ("natural", "animation", "screen", "synthetic")

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


@dataclass(frozen=True)
class ManifestClip:
    """One row of the clip manifest: a 3 s segment of a source, starting at ``start_s``.

    ``url_or_path`` is a video URL or local path, ``lavfi:<graph>`` for a synthetic source, or a
    printf frame pattern (``.../%08d.png``, first frame 1) for an image sequence, whose frame
    rate is the optional ``src_fps`` column.
    """

    name: str
    source_group: str
    url_or_path: str
    start_s: float
    source_type: str
    license: str
    license_url: str
    src_fps: float | None = None
    note: str = ""


def manifest_path() -> Path:
    return Path(os.environ.get("VREG_MANIFEST") or MANIFEST)


def read_manifest(path: Path | None = None) -> dict[str, ManifestClip]:
    path = Path(path or manifest_path())
    if not path.exists():
        return {}
    out = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            missing = [c for c in MANIFEST_COLUMNS if not (row.get(c) or "").strip()]
            if missing:
                raise ValueError(f"{path}: row {row.get('name')!r} lacks {missing}")
            if row["source_type"] not in SOURCE_TYPES:
                raise ValueError(f"{path}: {row['name']}: source_type must be one of {SOURCE_TYPES}")
            if row["name"] in out:
                raise ValueError(f"{path}: duplicate clip name {row['name']!r}")
            if row["name"] in (*SYNTHETIC_CLIPS, *XIPH_CLIPS) and not _same_as_builtin(row):
                # the P0 clips may be listed, but only with their original recipe (cache keys are names)
                raise ValueError(f"{path}: {row['name']!r} is a P0 clip with a different source")
            fps = (row.get("src_fps") or "").strip()
            out[row["name"]] = ManifestClip(
                name=row["name"], source_group=row["source_group"], url_or_path=row["url_or_path"],
                start_s=float(row["start_s"]), source_type=row["source_type"], license=row["license"],
                license_url=row["license_url"], src_fps=float(fps) if fps else None,
                note=(row.get("note") or "").strip())
    return out


def _same_as_builtin(row: dict) -> bool:
    name, url = row["name"], row["url_or_path"]
    if name in XIPH_CLIPS:
        return url == XIPH_BASE + XIPH_CLIPS[name][0] and float(row["start_s"]) == 0
    return name in SYNTHETIC_CLIPS and url == "lavfi:" + SYNTHETIC_CLIPS[name][0]


@functools.cache
def _manifest_cached(path: str) -> dict[str, ManifestClip]:
    return read_manifest(Path(path))


def manifest() -> dict[str, ManifestClip]:
    return _manifest_cached(str(manifest_path()))


def _time_filters(fps: float) -> tuple[list[str], list[str]]:
    """Filters and output options that turn ``fps`` into the 30 fps master.

    Faster sources drop frames (as for the P0 Xiph clips). Slower ones (24/25 fps) are retimed
    frame for frame instead, because duplicated frames would be unnaturally cheap to encode;
    their motion plays 1.2-1.25x faster.
    """
    if fps >= FPS - 1:
        return [f"fps={FPS}"], []
    return [f"setpts=N/({FPS}*TB)"], ["-r", str(FPS)]


FIT_720P = f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,crop={WIDTH}:{HEIGHT}"


def fetch_segment(clip: ManifestClip, out: Path) -> Path:
    """Cut ``clip`` into the master format: 1280x720 (centre crop if not 16:9), 30 fps, 3 s, qp 0."""
    n = FPS * SECONDS
    src = clip.url_or_path
    if src.startswith("lavfi:"):
        ff.run(["-f", "lavfi", "-i", src[len("lavfi:"):], "-ss", str(clip.start_s), "-frames:v", str(n),
                "-vf", f"{FIT_720P},format=yuv420p", *MASTER_CODEC, str(out)])
        return out
    if "%" in src:
        return _fetch_frames(clip, out, n)
    tf, opts = _time_filters(ff.probe_fps(src))
    ff.run(["-ss", str(clip.start_s), "-i", src, "-an", "-vf", ",".join([*tf, FIT_720P]), *opts,
            "-frames:v", str(n), *MASTER_CODEC, str(out)])
    got = ff.probe(out).frames
    if got != n:
        out.unlink(missing_ok=True)
        raise ff.FFmpegError(f"{clip.name}: source too short at {clip.start_s} s ({got} of {n} frames)")
    return out


def _fetch_frames(clip: ManifestClip, out: Path, n: int) -> Path:
    """An image sequence: download (or read) only the frames the segment needs."""
    if not clip.src_fps:
        raise ValueError(f"{clip.name}: a frame pattern needs src_fps")
    tf, opts = _time_filters(clip.src_fps)
    step = 1 if opts else clip.src_fps / FPS
    first = 1 + round(clip.start_s * clip.src_fps)
    count = int(round(n * step)) + 1
    remote = clip.url_or_path.startswith(("http://", "https://"))
    with tempfile.TemporaryDirectory(dir=out.parent) as td:
        pattern = clip.url_or_path
        if remote:
            ext = Path(pattern).suffix
            local = Path(td) / f"%06d{ext}"

            def get(i: int):
                urllib.request.urlretrieve(pattern % (first + i), str(local) % (i + 1))

            with ThreadPoolExecutor(8) as pool:
                list(pool.map(get, range(count)))
            pattern, start = str(local), 1
        else:
            start = first
        ff.run(["-framerate", str(clip.src_fps), "-start_number", str(start), "-i", pattern,
                "-vf", ",".join([*tf, FIT_720P]), *opts, "-frames:v", str(n), *MASTER_CODEC, str(out)])
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
            elif name in manifest():
                fetch_segment(manifest()[name], out)
            else:
                raise KeyError(f"unknown clip {name!r}; known: {sorted(all_clip_names())}")
        paths[name] = out
    return paths


def all_clip_names() -> list[str]:
    """The P0 clips (``vreg clips all``); manifest clips are selected through ``@`` selectors."""
    return [*SYNTHETIC_CLIPS, *XIPH_CLIPS]


def describe(name: str) -> str:
    builtin = SYNTHETIC_CLIPS.get(name) or XIPH_CLIPS.get(name)
    if builtin:
        return builtin[1]
    c = manifest().get(name)
    return f"{c.source_type}, {c.source_group} @ {c.start_s:g} s, {c.license}" if c else ""
