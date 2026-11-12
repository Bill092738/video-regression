"""E24: per-clip closed-loop feedback (vreg.feedback)."""

import numpy as np

from vreg import feedback
from vreg.evaluate import evaluate
from vreg.platform import PlatformProfile, SimulatedPlatform, Tier

TINY = PlatformProfile("tiny", 108, (Tier(0, "libx264", crf=34, maxrate_kbps=400, denoise="2:2:4:4", preset="veryfast"),))


def test_correct_adds_the_luma_error_to_every_channel():
    x = np.full((3, 4, 4, 3), 100, np.uint8)
    y = np.full((3, 4, 4, 3), 90, np.uint8)
    y[..., 1] = 80  # error (10, 20, 10) -> luma error 15.87
    u = feedback.correct(x, [y], 0.5)
    assert u.shape == x.shape and (u == 108).all()
    assert (feedback.correct(x, [y, y], 0.5) == 116).all()  # steps accumulate
    assert (feedback.correct(x, [x], 1.0) == x).all()  # no error, no change
    assert feedback.correct(np.full((1, 2, 2, 3), 250, np.uint8), [np.zeros((1, 2, 2, 3), np.uint8)], 1.0).max() == 255


def test_evaluate_feedback_runs_and_caches(tiny_clip, tmp_path):
    plat = SimulatedPlatform(TINY)
    probe = evaluate("tiny", tiny_clip, {}, plat, tmp_path, use_lpips=False)
    playback = tmp_path / "tiny" / probe.workdir.split("/")[-1] / "playback.mp4"
    r = feedback.evaluate_feedback("tiny", tiny_clip, [playback], {}, 0.5, plat, tmp_path, use_lpips=False)
    assert not r["cached"] and r["mean_abs_residual"] > 0 and "playback_kbps" in r["metrics"]
    again = feedback.evaluate_feedback("tiny", tiny_clip, [playback], {}, 0.5, plat, tmp_path, use_lpips=False)
    assert again["cached"] and again["metrics"] == r["metrics"]
    other = feedback.evaluate_feedback("tiny", tiny_clip, [playback], {}, 1.0, plat, tmp_path, use_lpips=False)
    assert other["workdir"] != r["workdir"]
