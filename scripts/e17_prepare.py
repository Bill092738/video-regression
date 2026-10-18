"""E17 (docs/E17-plan.md): build the paired-comparison study page from the simulator's playbacks.

    uv run python scripts/e17_prepare.py      # -> runs/e17/index.html, runs/e17/stimuli/*.mp4, runs/e17/trials.json

Open runs/e17/index.html in a browser on the study machine. Nothing is uploaded anywhere.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from vreg import content, ff
from vreg.evaluate import evaluate
from vreg.optimize import cache_root
from vreg.platform import make_platform

OUT = Path("runs/e17")
CLIPS = ("vidyo1_2", "vidyo3_2", "vidyo4_2", "four_people_2", "johnny_2", "kristen_and_sara_2",
         "pedestrian_area_0", "pedestrian_area_8", "tractor_0", "tractor_12")
COMPARISONS = {"recipe": "×2 + mild unsharp vs tier-only", "learned_g": "learned G vs tier-only",
               "bitrate": "16 Mbps vs near-lossless (positive control)"}
STIMULUS_CODEC = ["-c:v", "libx264", "-preset", "slow", "-crf", "4", "-profile:v", "high", "-pix_fmt", "yuv420p",
                  "-movflags", "+faststart", "-an"]
SEED = 17
REPEATS = 3


def playbacks() -> dict[tuple[str, str], Path]:
    """(clip, config) -> playback file; config in {tier, recipe, learned_g, bitrate}."""
    plat = make_platform("sim:ugc_strict")
    work = cache_root(plat)
    srcs = content.ensure_clips(list(CLIPS), content.CLIPS_DIR)
    out = {}
    for c, src in srcs.items():
        for cfg, params, kbps in (("tier", {"upscale": 2.0}, None),
                                  ("recipe", {"upscale": 2.0, "sharpen": 0.5, "sharpen_radius": 2.0}, None),
                                  ("bitrate", {"upscale": 2.0}, 16_000)):
            out[(c, cfg)] = Path(evaluate(c, src, params, plat, work, upload_kbps=kbps).workdir) / "playback.mp4"
        g = Path("runs/p1_g_lpips_s0/round0") / c / "playback.mp4"
        if not g.exists():
            raise FileNotFoundError(f"{g}: run the P1 global G (seed 0) evaluation on {c} first")
        out[(c, "learned_g")] = g
    return out


def main() -> None:
    (OUT / "stimuli").mkdir(parents=True, exist_ok=True)
    files = {}
    for (c, cfg), pb in playbacks().items():
        dst = OUT / "stimuli" / f"{c}__{cfg}.mp4"
        if not dst.exists():
            ff.run(["-i", str(pb), *STIMULUS_CODEC, str(dst)])
        files[(c, cfg)] = dst.name
    rng = random.Random(SEED)
    trials = []
    for c in CLIPS:
        for comp in COMPARISONS:
            ref = "tier"
            treat = comp
            trials.append({"clip": c, "comparison": comp, "treatment": files[(c, treat)], "reference": files[(c, ref)]})
    rng.shuffle(trials)
    for i, t in enumerate(trials):
        t["id"] = i
    repeats = [dict(t, id=len(trials) + k, repeat_of=t["id"]) for k, t in enumerate(rng.sample(trials, REPEATS))]
    all_trials = trials + repeats
    (OUT / "trials.json").write_text(json.dumps(all_trials, indent=2))
    html = (Path(__file__).with_name("e17_page.html").read_text()
            .replace("/*TRIALS*/[]", json.dumps(all_trials)))
    (OUT / "index.html").write_text(html)
    print(f"{len(all_trials)} trials; open {OUT / 'index.html'}")


if __name__ == "__main__":
    main()
