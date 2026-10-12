"""Gradient-free search over pre-compensation parameters theta.

Each trial is expensive (render -> "upload" -> transcode -> decode -> metrics),
so sample-efficient samplers matter: GP Bayesian optimisation and TPE should
beat random search within a few dozen trials; CMA-ES is the population-based
alternative.  Trial 0 is always the identity (naive upload) as the baseline.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import optuna

from . import metrics, precomp
from .evaluate import evaluate
from .platform import Platform

SAMPLERS = ("random", "tpe", "gp", "cmaes")


def make_sampler(name: str, seed: int) -> optuna.samplers.BaseSampler:
    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if name == "tpe":
        return optuna.samplers.TPESampler(seed=seed, multivariate=True, n_startup_trials=8)
    if name == "gp":
        return optuna.samplers.GPSampler(seed=seed, n_startup_trials=8)
    if name == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=seed, n_startup_trials=1, popsize=8)
    raise ValueError(f"unknown sampler {name!r}; choose from {SAMPLERS}")


def cache_root(platform: Platform, root: Path = Path("runs/_cache")) -> Path:
    return root / re.sub(r"[^A-Za-z0-9_.-]+", "_", platform.name)


def suggest(trial: optuna.Trial) -> dict:
    return {p.name: trial.suggest_float(p.name, p.low, p.high) for p in precomp.SPACE}


def run_study(
    out_dir: Path,
    clips: dict[str, Path],
    platform: Platform,
    sampler: str = "gp",
    n_trials: int = 40,
    objective: str = "composite",
    seed: int = 0,
    use_lpips: bool | None = None,
    log=lambda s: print(s, flush=True),
) -> optuna.Study:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"clips": {k: str(v) for k, v in clips.items()}, "platform": platform.name,
            "sampler": sampler, "objective": objective, "seed": seed}
    (out_dir / "config.json").write_text(json.dumps(meta, indent=2))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=out_dir.name,
        storage=f"sqlite:///{out_dir / 'study.db'}",
        sampler=make_sampler(sampler, seed),
        direction="maximize",
        load_if_exists=True,
    )
    if len(study.trials) == 0:
        study.enqueue_trial(dict(precomp.IDENTITY), user_attrs={"baseline": True})

    trials_log = out_dir / "trials.jsonl"
    work = cache_root(platform)

    def objective_fn(trial: optuna.Trial) -> float:
        params = suggest(trial)
        t0 = time.perf_counter()
        per_clip, scores = {}, []
        for clip, src in clips.items():
            r = evaluate(clip, src, params, platform, work, use_lpips=use_lpips)
            per_clip[clip] = r.metrics | {"workdir": r.workdir, "upload_height": r.upload_height}
            scores.append(metrics.objective_value(r.metrics, objective))
        value = float(np.mean(scores))
        trial.set_user_attr("per_clip", per_clip)
        trial.set_user_attr("params_norm", precomp.normalize(params))
        rec = {"trial": trial.number, "value": value, "params": precomp.normalize(params),
               "per_clip": per_clip, "seconds": time.perf_counter() - t0}
        with trials_log.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        done = [t.value for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))]
        best = max([value, *done])
        shown = {k: round(v, 2) for k, v in rec["params"].items()}
        log(f"[{out_dir.name}] trial {trial.number:3d} {objective}={value:7.3f} best={best:7.3f} "
            f"({rec['seconds']:.0f}s) {shown}")
        return value

    remaining = n_trials - len([t for t in study.trials if t.state.is_finished()])
    if remaining > 0:
        study.optimize(objective_fn, n_trials=remaining, catch=(Exception,))
    return study


def load_trials(out_dir: Path) -> list[dict]:
    """Trials from a study directory (deduplicated by trial number, latest wins)."""
    recs = {}
    path = out_dir / "trials.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                recs[r["trial"]] = r
    return [recs[k] for k in sorted(recs)]
