"""P0 additions: judge metrics, cache back-fill, parameter freezing, rescore and seed aggregation."""

import json

import numpy as np
import pytest
from test_pipeline import TINY

from vreg import ff, metrics, precomp, report
from vreg.evaluate import evaluate
from vreg.optimize import load_trials, parse_fixed, run_study
from vreg.platform import SimulatedPlatform
from vreg.rescore import improvement, verdict


def test_judges_are_not_in_composite():
    m = {"vmaf_neg": 80.0, "lpips": 0.1, "cambi": 0.5}
    assert metrics.composite(m) == metrics.composite(m | {"ssimulacra2": -50.0, "dists": 0.9, "xpsnr": 10.0})
    assert metrics.objective_value({"dists": 0.2}, "dists") == -0.2
    assert metrics.objective_value({"ssimulacra2": 70.0}, "ssimulacra2") == 70.0


def test_ssimulacra2_identical_is_100(tiny_clip):
    assert metrics.ssimulacra2_score(tiny_clip, tiny_clip) == pytest.approx(100.0)


def test_xpsnr_identical_is_capped(tiny_clip):
    if not ff.xpsnr_ffmpeg_bin():
        pytest.skip("ffmpeg without the xpsnr filter")
    assert metrics.xpsnr_score(tiny_clip, tiny_clip) == 100.0


@pytest.mark.gpu
def test_dists_identical_is_zero(tiny_clip):
    pytest.importorskip("piq")
    assert metrics.dists_score(tiny_clip, tiny_clip) < 1e-3


def test_cache_hit_backfills_judges_without_platform(tiny_clip, tmp_path):
    plat = SimulatedPlatform(TINY)
    first = evaluate("tiny", tiny_clip, {"sharpen": 0.5}, plat, tmp_path, use_lpips=False)
    assert "ssimulacra2" not in first.metrics

    class NoPlatform:
        name = plat.name

        def process(self, upload, workdir):
            raise AssertionError("the platform must not run again to add a metric")

    second = evaluate("tiny", tiny_clip, {"sharpen": 0.5}, NoPlatform(), tmp_path, use_lpips=False,
                      judges=("ssimulacra2",))
    assert second.cached and "ssimulacra2" in second.metrics
    stored = json.loads((tmp_path / "tiny" / first.workdir.split("/")[-1] / "result.json").read_text())
    assert stored["metrics"]["ssimulacra2"] == second.metrics["ssimulacra2"]
    assert stored["metrics"]["composite"] == first.metrics["composite"]


def test_parse_fixed_normalises_and_validates():
    assert parse_fixed("contrast=1,saturation=1") == {"contrast": 1.0, "saturation": 1.0}
    assert parse_fixed("upscale=1.9") == {"upscale": 2.0}
    assert parse_fixed(None) == {}
    with pytest.raises(ValueError):
        parse_fixed("eq=0")


@pytest.mark.slow
def test_fixed_params_are_constant_and_recorded(tiny_clip, tmp_path, monkeypatch):
    monkeypatch.setattr("vreg.optimize.cache_root", lambda p: tmp_path / "cache")
    study = run_study(tmp_path / "run", {"tiny": tiny_clip}, SimulatedPlatform(TINY), sampler="random",
                      n_trials=3, objective="ssim", use_lpips=False, fixed={"upscale": 2.0, "contrast": 1.0},
                      log=lambda *_: None)
    trials = load_trials(tmp_path / "run")
    assert trials[0]["params"] == precomp.normalize({"upscale": 2.0})  # identity except the fixed values
    assert all(t["params"]["upscale"] == 2.0 and t["params"]["contrast"] == 1.0 for t in trials)
    assert "upscale" not in study.trials[1].params  # not shown to the sampler
    assert json.loads((tmp_path / "run" / "config.json").read_text())["fixed"] == {"upscale": 2.0, "contrast": 1.0}


def test_improvement_sign_and_verdict():
    assert improvement("dists", 0.1, 0.2) == pytest.approx(0.1)
    assert improvement("ssimulacra2", 60, 65) == -5
    summ = {"composite": {"mean": 2.0}, "ssimulacra2": {"mean": 1.0}, "dists": {"mean": -0.01},
            "xpsnr": {"mean": 0.2}}
    assert verdict(summ).startswith("stands (2/3")
    summ["xpsnr"]["mean"] = -0.2
    assert verdict(summ).startswith("not confirmed (1/3")


def _fake_trials(values):
    return [{"trial": i, "value": v} for i, v in enumerate(values)]


def test_seed_groups_and_table():
    base = {"clips": {"a": "a.mkv"}, "platform": "sim:x", "objective": "composite"}
    configs = {"gp_s0": base | {"sampler": "gp", "seed": 0},  # older run without "fixed"
               "gp_s1": base | {"sampler": "gp", "seed": 1, "fixed": {}},
               "rnd": base | {"sampler": "random", "seed": 0, "fixed": {}}}
    groups = report.seed_groups(configs)
    assert groups == {"gp": ["gp_s0", "gp_s1"], "rnd": ["rnd"]}
    runs = {"gp_s0": _fake_trials([0, 5, 9, 10]), "gp_s1": _fake_trials([0, 10, 10, 10, 11]),
            "rnd": _fake_trials([0, 1, 2, 3])}
    assert report.trial_to_fraction(np.array([0, 5, 9.5, 10])) == 2
    rows = {r["label"]: r for r in report.seed_table(runs, groups)}
    assert rows["gp"]["n"] == 2 and rows["gp"]["trials"] == 4
    assert rows["gp"]["final_mean"] == pytest.approx(10.0) and rows["gp"]["final_sd"] == 0.0
    assert rows["gp"]["t95_mean"] == pytest.approx(2.0)  # 3 and 1
    assert report.convergence_plot(runs, groups)[:4] == b"\x89PNG"
