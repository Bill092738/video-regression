"""The "platform": a black-box degradation channel f_platform(upload) -> playback.

The optimiser only ever calls :meth:`Platform.process` with an uploaded file and
gets a playback file back.  It never sees the transcoding policy.

``SimulatedPlatform`` mimics behaviour that real UGC platforms are known to have:

* the encode ladder / codec / bit budget depends on the *upload* resolution
  (uploading 1440p gets a better codec and more bits even for the 720p rendition);
* ingest pre-processing (spatio-temporal denoise) whose strength depends on tier;
* capped-CRF rate control (quality target, but a hard maxrate for streaming);
* the viewer is served a fixed playback rendition (720p here).

``CommandPlatform`` wraps a real platform via user supplied shell commands
(upload, then download the processed stream), so the same optimisation loop can
run against e.g. a private/unlisted upload workflow.
"""

from __future__ import annotations

import hashlib
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from . import ff


class Platform(Protocol):
    name: str

    def process(self, upload: Path, workdir: Path) -> Path:
        """Upload ``upload``; return a local path to what a viewer would receive."""
        ...


@dataclass(frozen=True)
class Tier:
    min_upload_height: int
    codec: str          # libx264 | libx265 | libsvtav1
    crf: int
    maxrate_kbps: int
    denoise: str        # hqdn3d params, "" = none
    preset: str = "medium"


@dataclass(frozen=True)
class PlatformProfile:
    name: str
    playback_height: int
    tiers: tuple[Tier, ...]  # sorted by min_upload_height descending
    notes: str = ""

    def tier_for(self, upload_height: int) -> Tier:
        for t in self.tiers:
            if upload_height >= t.min_upload_height:
                return t
        return self.tiers[-1]


PROFILES: dict[str, PlatformProfile] = {
    # Aggressive UGC-style platform: heavy denoise, low bit budget on standard
    # uploads, "HD+" uploads get HEVC and more bits.
    "ugc_strict": PlatformProfile(
        name="ugc_strict",
        playback_height=720,
        tiers=(
            Tier(1440, "libx265", crf=26, maxrate_kbps=2000, denoise="1.5:1.5:4:4", preset="fast"),
            Tier(1080, "libx264", crf=25, maxrate_kbps=1600, denoise="2:2:6:6"),
            Tier(0, "libx264", crf=27, maxrate_kbps=1200, denoise="3:3:8:8"),
        ),
        notes="bilibili-like: strong ingest NR, tight caps",
    ),
    # Milder platform that reserves AV1 for high resolution uploads.
    "ugc_av1": PlatformProfile(
        name="ugc_av1",
        playback_height=720,
        tiers=(
            Tier(1440, "libsvtav1", crf=36, maxrate_kbps=1800, denoise="", preset="10"),
            Tier(0, "libx264", crf=24, maxrate_kbps=1800, denoise="1:1:3:3"),
        ),
        notes="youtube-like: AV1 only for >=1440p uploads",
    ),
}


def _file_hash(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()[:24]


@dataclass
class SimulatedPlatform:
    profile: PlatformProfile
    cache_dir: Path | None = None
    threads: int = 0  # 0 = encoder default
    name: str = field(init=False)

    def __post_init__(self):
        self.name = f"sim:{self.profile.name}"

    @classmethod
    def from_name(cls, name: str, **kw) -> "SimulatedPlatform":
        return cls(PROFILES[name], **kw)

    def _encode_args(self, tier: Tier) -> list[str]:
        rate = ["-maxrate", f"{tier.maxrate_kbps}k", "-bufsize", f"{tier.maxrate_kbps}k"]
        if tier.codec == "libx264":
            return ["-c:v", "libx264", "-preset", tier.preset, "-crf", str(tier.crf), *rate,
                    "-pix_fmt", "yuv420p"]
        if tier.codec == "libx265":
            params = "log-level=error" + (f":pools={self.threads}" if self.threads else "")
            return ["-c:v", "libx265", "-preset", tier.preset, "-crf", str(tier.crf), *rate,
                    "-x265-params", params, "-tag:v", "hvc1", "-pix_fmt", "yuv420p"]
        if tier.codec == "libsvtav1":
            # SVT-AV1 caps bitrate through mbr= in capped-CRF mode
            return ["-c:v", "libsvtav1", "-preset", tier.preset, "-crf", str(tier.crf),
                    "-svtav1-params", f"mbr={tier.maxrate_kbps}", "-pix_fmt", "yuv420p"]
        raise ValueError(tier.codec)

    def process(self, upload: Path, workdir: Path) -> Path:
        upload = Path(upload)
        workdir.mkdir(parents=True, exist_ok=True)
        key = f"{self.profile.name}-{_file_hash(upload)}"
        cached = self.cache_dir / f"{key}.mp4" if self.cache_dir else None
        out = workdir / "playback.mp4"
        if cached and cached.exists():
            shutil.copyfile(cached, out)
            return out

        info = ff.probe(upload)
        tier = self.profile.tier_for(info.height)
        ph = self.profile.playback_height
        filters = []
        if tier.denoise:
            filters.append(f"hqdn3d={tier.denoise}")
        filters.append(f"scale=-2:{ph}:flags=bicubic")
        args = ["-i", str(upload), "-an", "-vf", ",".join(filters), *self._encode_args(tier)]
        if self.threads and tier.codec == "libx264":
            args += ["-threads", str(self.threads)]
        ff.run([*args, str(out)])
        if cached:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(out, cached)
        return out


@dataclass
class CommandPlatform:
    """Adapter for a real platform driven by shell commands.

    ``upload_cmd`` gets ``{file}`` and must print a video id / URL on stdout.
    ``download_cmd`` gets ``{id}`` and ``{out}`` and must write the playback
    stream to ``{out}`` (e.g. ``yt-dlp -f 'bv*[height=720]' -o {out} {id}``);
    it is retried until it succeeds or ``timeout_s`` elapses, because platforms
    transcode asynchronously.

    Only use this against accounts and content you own and in line with the
    platform's terms of service.
    """

    upload_cmd: str
    download_cmd: str
    timeout_s: float = 1800
    poll_s: float = 30
    name: str = "external"

    def process(self, upload: Path, workdir: Path) -> Path:
        workdir.mkdir(parents=True, exist_ok=True)
        up = subprocess.run(
            self.upload_cmd.format(file=shlex.quote(str(upload))),
            shell=True, capture_output=True, text=True, check=True,
        )
        vid = up.stdout.strip().splitlines()[-1]
        (workdir / "platform_id.txt").write_text(vid)
        out = workdir / "playback.mp4"
        deadline = time.monotonic() + self.timeout_s
        while True:
            dl = subprocess.run(
                self.download_cmd.format(id=shlex.quote(vid), out=shlex.quote(str(out))),
                shell=True, capture_output=True, text=True,
            )
            if dl.returncode == 0 and out.exists() and out.stat().st_size > 0:
                return out
            if time.monotonic() > deadline:
                raise TimeoutError(f"platform never produced a playback stream for {vid}: {dl.stderr[-500:]}")
            time.sleep(self.poll_s)


def make_platform(spec: str, cache_dir: Path | None = None) -> Platform:
    """``sim:<profile>`` or ``cmd:<upload_cmd>||<download_cmd>``."""
    kind, _, rest = spec.partition(":")
    if kind == "sim":
        return SimulatedPlatform.from_name(rest or "ugc_strict", cache_dir=cache_dir)
    if kind == "cmd":
        up, _, down = rest.partition("||")
        return CommandPlatform(upload_cmd=up, download_cmd=down)
    raise ValueError(f"unknown platform spec {spec!r}")
