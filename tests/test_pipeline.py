
import numpy as np
import pytest

from vreg import ff, metrics, precomp
from vreg.evaluate import evaluate
from vreg.optimize import load_trials, run_study
from vreg.platform import PlatformProfile, SimulatedPlatform, Tier

TINY = PlatformProfile(
    name="tiny",
    playback_height=108,
    tiers=(
        Tier(162, "libx264", crf=24, maxrate_kbps=300, denoise="1:1:2:2", preset="veryfast"),
        Tier(0, "libx264", crf=32, maxrate_kbps=100, denoise="3:3:6:6", preset="veryfast"),
    ),
)


def test_normalize_snaps_and_clips():
    p = precomp.normalize({"upscale": 1.3, "sharpen_radius": 2.6, "grain": 99, "contrast": 0.1})
    assert p["upscale"] == 1.5
    assert p["sharpen_radius"] == 3.0
    assert p["grain"] == 15.0
    assert p["contrast"] == 0.9
    assert set(p) == set(precomp.PARAM_NAMES)


def test_identity_filter_is_noop():
    assert precomp.build_filter(precomp.IDENTITY, 720) == "format=yuv420p"


def test_filter_contains_all_stages():
    f = precomp.build_filter({"denoise": 2, "contrast": 1.1, "upscale": 2, "sharpen": 1, "grain": 5}, 720)
    for stage in ("hqdn3d", "eq=", "scale=-2:1440", "unsharp", "noise="):
        assert stage in f


def test_precomp_upscale_changes_upload_height(tiny_clip, tmp_path):
    out = precomp.apply(tiny_clip, {"upscale": 2.0}, tmp_path / "up.mp4")
    assert ff.probe(out).height == 216


def test_platform_picks_tier_by_upload_height(tiny_clip, tmp_path):
    plat = SimulatedPlatform(TINY)
    assert TINY.tier_for(108).crf == 32
    assert TINY.tier_for(216).crf == 24
    out = plat.process(tiny_clip, tmp_path)
    info = ff.probe(out)
    assert info.height == 108 and info.codec == "h264"


def test_read_frames_shape(tiny_clip):
    frames = ff.read_frames(tiny_clip, size=(96, 54), step=3)
    assert frames.shape == (5, 54, 96, 3) and frames.dtype == np.uint8


def test_metrics_identical_video_is_high(tiny_clip):
    m = metrics.compute(tiny_clip, tiny_clip, use_lpips=False)
    assert m["ssim"] > 0.99
    assert m["psnr_y"] > 50


def test_evaluate_caches(tiny_clip, tmp_path):
    plat = SimulatedPlatform(TINY)
    r1 = evaluate("tiny", tiny_clip, {"sharpen": 0.5}, plat, tmp_path, use_lpips=False)
    r2 = evaluate("tiny", tiny_clip, {"sharpen": 0.5}, plat, tmp_path, use_lpips=False)
    assert not r1.cached and r2.cached
    assert r1.metrics["composite"] == r2.metrics["composite"]


def test_higher_tier_beats_naive_upload(tiny_clip, tmp_path):
    """The black box rewards bigger uploads: the search space must be able to exploit it."""
    plat = SimulatedPlatform(TINY)
    naive = evaluate("tiny", tiny_clip, {}, plat, tmp_path, use_lpips=False)
    up = evaluate("tiny", tiny_clip, {"upscale": 2.0}, plat, tmp_path, use_lpips=False)
    assert up.metrics["ssim"] > naive.metrics["ssim"]


@pytest.mark.slow
def test_study_runs_and_logs(tiny_clip, tmp_path, monkeypatch):
    monkeypatch.setattr("vreg.optimize.cache_root", lambda p: tmp_path / "cache")
    study = run_study(tmp_path / "run", {"tiny": tiny_clip}, SimulatedPlatform(TINY), sampler="random",
                      n_trials=3, objective="ssim", use_lpips=False, log=lambda *_: None)
    trials = load_trials(tmp_path / "run")
    assert len(trials) == 3
    assert trials[0]["params"] == precomp.normalize(precomp.IDENTITY)  # baseline first
    assert study.best_value >= trials[0]["value"]


@pytest.mark.gpu
def test_lpips_runs(tiny_clip):
    pytest.importorskip("lpips")
    assert metrics.lpips_score(tiny_clip, tiny_clip) < 1e-3
