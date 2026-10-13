"""Thin helpers around the ffmpeg / ffprobe binaries."""

from __future__ import annotations

import functools
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# A static ffmpeg build with libvmaf, installed by scripts/setup_env.sh
_LOCAL_VMAF_FFMPEG = Path.home() / ".local/opt/ffmpeg-vmaf/bin/ffmpeg"


class FFmpegError(RuntimeError):
    pass


@functools.cache
def ffmpeg_bin() -> str:
    path = os.environ.get("VREG_FFMPEG") or shutil.which("ffmpeg")
    if not path:
        raise FFmpegError("ffmpeg not found; install it or set VREG_FFMPEG")
    return path


@functools.cache
def ffprobe_bin() -> str:
    path = os.environ.get("VREG_FFPROBE") or shutil.which("ffprobe")
    if not path:
        sibling = Path(ffmpeg_bin()).with_name("ffprobe")
        if sibling.exists():
            return str(sibling)
        raise FFmpegError("ffprobe not found; install it or set VREG_FFPROBE")
    return path


def _has_filter(binary: str, name: str) -> bool:
    try:
        out = subprocess.run(
            [binary, "-hide_banner", "-filters"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return False
    return any(line.split()[1:2] == [name] for line in out.splitlines() if line.strip())


@functools.cache
def vmaf_ffmpeg_bin() -> str | None:
    """Return an ffmpeg binary that has the libvmaf filter, or None."""
    candidates = [os.environ.get("VREG_FFMPEG_VMAF"), ffmpeg_bin(), str(_LOCAL_VMAF_FFMPEG)]
    for cand in candidates:
        if cand and Path(cand).exists() and _has_filter(cand, "libvmaf"):
            return cand
    return None


@functools.cache
def xpsnr_ffmpeg_bin() -> str | None:
    """Return an ffmpeg binary that has the xpsnr filter (ffmpeg >= 7.1), or None."""
    for cand in (ffmpeg_bin(), vmaf_ffmpeg_bin(), str(_LOCAL_VMAF_FFMPEG)):
        if cand and Path(cand).exists() and _has_filter(cand, "xpsnr"):
            return cand
    return None


def run(args: list[str], binary: str | None = None) -> subprocess.CompletedProcess:
    cmd = [binary or ffmpeg_bin(), "-hide_banner", "-nostdin", "-loglevel", "error", "-y", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FFmpegError(f"ffmpeg failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr[-2000:]}")
    return proc


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int
    duration: float
    bitrate: float  # bits per second of the whole file
    codec: str
    size_bytes: int


def probe(path: str | Path) -> VideoInfo:
    path = Path(path)
    proc = subprocess.run(
        [
            ffprobe_bin(), "-v", "error", "-select_streams", "v:0", "-count_packets",
            "-show_entries", "stream=width,height,r_frame_rate,nb_read_packets,codec_name:format=duration,size",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise FFmpegError(f"ffprobe failed on {path}: {proc.stderr}")
    data = json.loads(proc.stdout)
    st = data["streams"][0]
    num, den = st["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    frames = int(st.get("nb_read_packets", 0))
    duration = float(data["format"].get("duration") or (frames / fps if fps else 0))
    size = int(data["format"].get("size") or path.stat().st_size)
    return VideoInfo(
        width=int(st["width"]),
        height=int(st["height"]),
        fps=fps,
        frames=frames,
        duration=duration,
        bitrate=8 * size / duration if duration else 0.0,
        codec=st.get("codec_name", "?"),
        size_bytes=size,
    )


def probe_fps(src: str | Path) -> float:
    """Frame rate from the stream header only; unlike :func:`probe` it does not read the whole
    file, which matters for remote sources."""
    proc = subprocess.run([ffprobe_bin(), "-v", "error", "-select_streams", "v:0", "-show_entries",
                           "stream=r_frame_rate", "-of", "csv=p=0", str(src)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise FFmpegError(f"ffprobe failed on {src}: {proc.stderr}")
    num, _, den = proc.stdout.strip().partition("/")
    return float(num) / float(den or 1)


def read_frames(
    path: str | Path,
    size: tuple[int, int] | None = None,
    step: int = 1,
    max_frames: int | None = None,
    scale_flags: str = "bicubic",
) -> np.ndarray:
    """Decode a video into an (N, H, W, 3) uint8 RGB array.

    ``size`` = (width, height) rescales every frame (as a viewer's player would);
    ``step`` keeps every n-th frame.
    """
    info = probe(path)
    w, h = size or (info.width, info.height)
    filters = []
    if step > 1:
        filters.append(f"select='not(mod(n\\,{step}))'")
    if (w, h) != (info.width, info.height):
        filters.append(f"scale={w}:{h}:flags={scale_flags}")
    filters.append("format=rgb24")
    args = [ffmpeg_bin(), "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(path),
            "-vf", ",".join(filters), "-fps_mode", "passthrough"]
    if max_frames:
        args += ["-frames:v", str(max_frames)]
    args += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.run(args, capture_output=True)
    if proc.returncode != 0:
        raise FFmpegError(f"decode failed on {path}: {proc.stderr.decode()[-2000:]}")
    buf = np.frombuffer(bytearray(proc.stdout), dtype=np.uint8)
    return buf.reshape(-1, h, w, 3)


def write_frames(
    frames: np.ndarray, path: str | Path, fps: float, codec_args: list[str] | None = None
) -> Path:
    """Encode an (N, H, W, 3) uint8 RGB array to a video file."""
    path = Path(path)
    n, h, w, _ = frames.shape
    codec_args = codec_args or ["-c:v", "libx264", "-preset", "veryfast", "-crf", "8", "-pix_fmt", "yuv420p"]
    cmd = [ffmpeg_bin(), "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
           *codec_args, str(path)]
    proc = subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), capture_output=True)
    if proc.returncode != 0:
        raise FFmpegError(f"encode failed: {proc.stderr.decode()[-2000:]}")
    return path
