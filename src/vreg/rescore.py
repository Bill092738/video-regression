"""Re-score a run's key configurations with every metric, including the report-only judges.

For an optimisation run the configurations are naive, tier-only and the best theta (the same
decomposition as ``vreg transfer``).  For a surrogate run they are naive, tier-only and each
round's learned ``G``, found through the ``workdir`` entries in ``history.json``.  Everything
reuses the evaluation cache; missing judges are back-filled from the cached playbacks, so the
platform is only run for configurations that were never evaluated.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import content, metrics
from .evaluate import backfill, evaluate
from .optimize import best_params, cache_root, decomposition
from .platform import make_platform

METRICS = ("composite", "vmaf_neg", "lpips", *metrics.JUDGE_METRICS)
NATURAL_CLIPS = tuple(content.XIPH_CLIPS)
REFERENCE = "upscale_only"  # gains are measured over tier-only


def judges_for(g_loss: str | None = None) -> tuple[str, ...]:
    """The judges of a configuration: a learned G is never judged by a metric in its own training
    loss. A G trained with DISTS is judged by SSIMULACRA2, XPSNR and VMAF NEG instead (P1 plan,
    code change 6); VMAF NEG is in no G loss."""
    if g_loss and "dists" in g_loss:
        return ("ssimulacra2", "xpsnr", "vmaf_neg")
    return metrics.JUDGE_METRICS


def improvement(metric: str, a: float, b: float) -> float:
    """``a`` relative to ``b``, signed so that positive means ``a`` is better."""
    return a - b if metrics.HIGHER_IS_BETTER[metric] else b - a


def rescore(run_dir: Path, clips: dict[str, Path], tag: str = "", platform_override: str | None = None,
            log=lambda s: print(s, flush=True)) -> dict:
    """Writes ``rescore.json`` / ``rescore.md`` (``rescore_<tag>.*`` with a tag) into the run.

    ``platform_override`` evaluates the run's configurations on another platform (cross-platform
    transfer, backlog E14); a learned G's own playbacks are only used on the run's platform.
    """
    judges = metrics.JUDGE_METRICS
    hist_file = run_dir / "history.json"
    g_loss = None
    if hist_file.exists():
        hist = json.loads(hist_file.read_text())
        # runs from before the platform was recorded used ugc_strict
        kind, platform, train = "surrogate", hist.get("platform", "sim:ugc_strict"), [
            c for c, r in hist["rounds"][0]["clips"].items() if not r["holdout"]]
        configs = {"naive": {}, "upscale_only": {"upscale": hist["upscale"]}}
        g_loss = hist.get("settings", {}).get("g_loss", "lpips")
    else:
        cfg = json.loads((run_dir / "config.json").read_text())
        kind, platform, train = "optimize", cfg["platform"], list(cfg["clips"])
        configs = decomposition(best_params(run_dir))
    trained_on = platform
    if platform_override and platform_override != platform:
        if kind == "surrogate":
            raise ValueError("a learned G can only be re-scored on its own platform (its playbacks are stored)")
        platform = platform_override
    plat = make_platform(platform)
    work = cache_root(plat)
    out = {"run": run_dir.name, "kind": kind, "platform": platform, "trained_on": trained_on, "train_clips": train,
           "configs": configs, "g_loss": g_loss, "judges": list(judges_for(g_loss)), "clips": {}}
    for clip, src in clips.items():
        row = {label: evaluate(clip, src, params, plat, work, judges=judges).metrics
               for label, params in configs.items()}
        if kind == "surrogate":
            for rec in hist["rounds"]:
                g = rec["clips"].get(clip)
                if g:
                    row[f"G_r{rec['round']}"] = _g_metrics(g, src, judges)
        out["clips"][clip] = row
        log(f"[rescore {run_dir.name}] {clip:16s} " + " ".join(
            f"{label}={m['composite']:.2f}/{m.get('ssimulacra2', float('nan')):.1f}" for label, m in row.items()))
    stem = f"rescore_{tag}" if tag else "rescore"
    (run_dir / f"{stem}.json").write_text(json.dumps(out, indent=2))
    (run_dir / f"{stem}.md").write_text(format_tables(out))
    return out


def _g_metrics(g: dict, src: Path, judges) -> dict:
    """Metrics of a learned G's real playback; judges are stored next to it once computed."""
    wd = Path(g["workdir"])
    cache = wd / "judges.json"
    m = dict(g["metrics"])
    if cache.exists():
        m.update(json.loads(cache.read_text()))
    if backfill(m, wd / "playback.mp4", src, judges):
        cache.write_text(json.dumps({j: m[j] for j in judges if j in m}, indent=2))
    return m


def treatment_labels(res: dict) -> list[str]:
    labels = list(next(iter(res["clips"].values())))
    return [lb for lb in labels if lb == "best" or lb.startswith("G_r")]


def gain_summary(res: dict, clips=NATURAL_CLIPS) -> dict:
    """Per treatment and metric: improvement over tier-only per clip and averaged over ``clips``."""
    ref = REFERENCE
    out = {}
    for t in treatment_labels(res):
        out[t] = {}
        for metric in METRICS:
            per = {c: improvement(metric, row[t][metric], row[ref][metric])
                   for c, row in res["clips"].items() if t in row and metric in row[t] and metric in row[ref]}
            sel = [per[c] for c in clips if c in per]
            out[t][metric] = {"per_clip": per, "mean": float(np.mean(sel)) if sel else float("nan")}
    return out


def verdict(summary: dict, judges=metrics.JUDGE_METRICS) -> str:
    """The P0 E2 rule: the gain stands if >= 2 of the 3 judges agree in sign with composite."""
    c = summary["composite"]["mean"]
    agree = [j for j in judges if j in summary and np.sign(summary[j]["mean"]) == np.sign(c)]
    word = "stands" if len(agree) >= 2 else "not confirmed"
    return f"{word} ({len(agree)}/3 judges agree with composite {c:+.2f}: {', '.join(agree) or 'none'})"


def format_tables(res: dict) -> str:
    labels = list(next(iter(res["clips"].values())))
    lines = [f"# Rescore: {res['run']} ({res['kind']}, {res['platform']})", "",
             f"Training clips: {', '.join(res['train_clips'])}. Natural clips for the means: "
             f"{', '.join(NATURAL_CLIPS)}. Gains are signed so that positive = better.", ""]
    summ = gain_summary(res)
    for metric in METRICS:
        arrow = "higher" if metrics.HIGHER_IS_BETTER[metric] else "lower"
        lines += [f"## {metric} ({arrow} is better)", "",
                  "| clip | " + " | ".join(labels) + " | " + " | ".join(f"{t} − tier" for t in summ) + " |",
                  "|---" * (1 + len(labels) + len(summ)) + "|"]
        for clip, row in res["clips"].items():
            cells = [f"{row[lb][metric]:.3f}" if metric in row.get(lb, {}) else "–" for lb in labels]
            gains = [f"{summ[t][metric]['per_clip'][clip]:+.3f}" if clip in summ[t][metric]["per_clip"] else "–"
                     for t in summ]
            tag = " *(train)*" if clip in res["train_clips"] else ""
            lines.append(f"| {clip}{tag} | " + " | ".join(cells + gains) + " |")
        lines.append("| **natural mean** | " + " | ".join(["" for _ in labels] + [
            f"**{summ[t][metric]['mean']:+.3f}**" for t in summ]) + " |")
        lines.append("")
    lines.append("## Verdict (natural clips, mean gain over tier-only)")
    lines.append("")
    for t in summ:
        lines.append(f"* `{t}`: {verdict(summ[t], res.get('judges', metrics.JUDGE_METRICS))}")
    return "\n".join(lines) + "\n"
