"""Markdown tables and figures for the P0 results (docs/P0-plan.md), from the runs' rescore.json files.

    uv run python scripts/p0_tables.py            # -> runs/p0_tables.md and docs/img/p0/*.png

Sections whose runs are missing are skipped, so this can be run while experiments are in progress.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from vreg import metrics, report  # noqa: E402
from vreg.optimize import load_trials  # noqa: E402
from vreg.rescore import METRICS, NATURAL_CLIPS, gain_summary, verdict  # noqa: E402

RUNS = Path("runs")
IMG = Path("docs/img/p0")
JUDGES = metrics.JUDGE_METRICS
SAMPLERS = ("gp", "tpe", "cmaes", "random")
OBJECTIVES = ("composite", "vmaf_neg", "ssimulacra2", "lpips")
# columns that a row was optimised for (composite contains vmaf_neg and lpips)
OWN = {"composite": {"composite", "vmaf_neg", "lpips"}, "vmaf_neg": {"vmaf_neg"},
       "ssimulacra2": {"ssimulacra2"}, "lpips": {"lpips"}}


def load(name: str) -> dict | None:
    f = RUNS / name / "rescore.json"
    return json.loads(f.read_text()) if f.exists() else None


def fmt(x: float, metric: str) -> str:
    return f"{x:+.4f}" if metric in ("lpips", "dists") else f"{x:+.2f}"


def signs(per_clip: dict) -> str:
    return " ".join(("+" if per_clip[c] > 0 else "−") if c in per_clip else "?" for c in NATURAL_CLIPS)


def ms(xs) -> str:
    xs = np.asarray(xs, float)
    return f"{xs.mean():.2f} ± {xs.std(ddof=1):.2f}" if len(xs) > 1 else f"{xs.mean():.2f}"


def e2(out: list[str]):
    rows = [("strict_gp", "best", "GP-BO θ, ugc_strict"), ("av1_gp", "best", "GP-BO θ, ugc_av1"),
            ("surrogate_trust", "G_r0", "learned G round 0, ugc_strict")]
    rows += [(f"strict_gp_s{s}", "best", f"GP-BO θ seed {s}, ugc_strict") for s in range(1, 5)]
    rows += [(f"surrogate_trust_s{s}", "G_r0", f"learned G round 0 seed {s}") for s in range(3)]
    out += ["## E2: filter gain over tier-only, judged by independent metrics", "",
            "Mean gain over tier-only on the four natural clips; positive = better for every metric. "
            "Signs per clip in the order " + ", ".join(NATURAL_CLIPS) + ".", "",
            "| configuration | " + " | ".join(METRICS) + " | verdict |", "|---" * (len(METRICS) + 2) + "|"]
    for run, t, label in rows:
        res = load(run)
        if not res:
            continue
        s = gain_summary(res)[t]
        cells = [f"{fmt(s[m]['mean'], m)} ({signs(s[m]['per_clip'])})" if m in s else "–" for m in METRICS]
        out.append(f"| {label} | " + " | ".join(cells) + f" | {verdict(s)} |")
    out.append("")
    for run, t, label in rows[:3]:
        res = load(run)
        if res and "testsrc" in res["clips"]:
            s = gain_summary(res)[t]
            out.append(f"* testsrc, {label}: " + ", ".join(
                f"{m} {fmt(s[m]['per_clip']['testsrc'], m)}" for m in METRICS if 'testsrc' in s[m]['per_clip']))
    out.append("")


def _gain(run: str, t: str = "best") -> dict | None:
    res = load(run)
    return gain_summary(res)[t] if res else None


def e3_e5(out: list[str]):
    full = _gain("strict_gp")
    for run, title, desc in (
            ("strict_gp_noeq", "E3: contrast and saturation frozen (`--fix contrast=1,saturation=1`)",
             "filter gain over tier-only with eq frozen, vs the full θ (strict_gp)"),
            ("strict_gp_native", "E5: native tier, filter-only (`--fix upscale=1`)",
             "filter gain at ×1 (best − naive; tier-only = naive here) vs at the full θ's tier (strict_gp)")):
        g = _gain(run)
        if not (full and g):
            continue
        res = load(run)
        out += [f"## {title}", "", f"Mean over the natural clips: {desc}. Best θ: `{json.dumps(res['configs']['best'])}`",
                "", "| metric | full θ gain | this run's gain | fraction kept |", "|---|---|---|---|"]
        for m in METRICS:
            f, x = full[m]["mean"], g[m]["mean"]
            frac = f"{x / f:.0%}" if abs(f) > 1e-9 and np.sign(f) == np.sign(x) else ("n/a (full gain ≤ 0)" if f <= 0 else f"{x / f:.0%}")
            out.append(f"| {m} | {fmt(f, m)} | {fmt(x, m)} | {frac} |")
        out += ["", f"* verdict for this run: {verdict(g)}", ""]


def e4(out: list[str]):
    rows = {o: _gain(f"strict_gp_obj_{o}") for o in OBJECTIVES}
    rows = {o: g for o, g in rows.items() if g}
    if not rows:
        return
    test = {o: gain_summary(load(f"strict_gp_obj_{o}"))["best"] for o in rows}
    mat = np.array([[rows[o][m]["mean"] for m in METRICS] for o in rows])
    # rank per column: 1 = best gain
    ranks = np.array([(-mat[:, j]).argsort().argsort() + 1 for j in range(len(METRICS))]).T
    out += ["## E4: Goodhart matrix", "",
            "Rows: optimised objective (GP-BO, 40 trials, ugc_strict). Cells: gain of the best θ over tier-only, "
            "mean over the natural clips; positive = better. Bold cells are the metrics the row optimised "
            "(composite contains vmaf_neg and lpips).", "",
            "| objective | " + " | ".join(METRICS) + " | negative non-own cells | mean rank |",
            "|---" * (len(METRICS) + 3) + "|"]
    choice = []
    for i, o in enumerate(rows):
        neg = [m for j, m in enumerate(METRICS) if mat[i, j] < 0 and m not in OWN[o]]
        cells = [(f"**{fmt(mat[i, j], m)}**" if m in OWN[o] else fmt(mat[i, j], m)) for j, m in enumerate(METRICS)]
        out.append(f"| {o} | " + " | ".join(cells) + f" | {', '.join(neg) or 'none'} | {ranks[i].mean():.2f} |")
        choice.append((len(neg) > 0, ranks[i].mean(), o))
    out += ["", "testsrc (synthetic graphics/text), same cells:", "",
            "| objective | " + " | ".join(METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]
    for o in rows:
        out.append(f"| {o} | " + " | ".join(
            fmt(test[o][m]["per_clip"].get("testsrc", float("nan")), m) for m in METRICS) + " |")
    best = sorted(choice)[0]
    rule = "no negative non-own cell" if not best[0] else "NO row is free of negative non-own cells; best mean rank"
    out += ["", f"* chosen default objective: **{best[2]}** ({rule}, mean rank {best[1]:.2f})", ""]
    _matrix_png(list(rows), mat)


def _matrix_png(objs: list[str], mat: np.ndarray):
    scale = np.abs(mat).max(0)
    scale[scale == 0] = 1
    fig, ax = plt.subplots(figsize=(1.5 * len(METRICS) + 1.5, 0.55 * len(objs) + 1.2))
    ax.imshow(mat / scale, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    for i in range(len(objs)):
        for j, m in enumerate(METRICS):
            ax.text(j, i, fmt(mat[i, j], m), ha="center", va="center", fontsize=9, color="#1d1d1f")
    ax.set_xticks(range(len(METRICS)), METRICS, fontsize=9)
    ax.set_yticks(range(len(objs)), [f"optimise {o}" for o in objs], fontsize=9)
    ax.set_title("Gain over tier-only (natural clips; blue = better, red = worse, scaled per column)", fontsize=10)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    IMG.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMG / "goodhart_matrix.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def e1(out: list[str]):
    names = {smp: [n for n in [f"strict_{smp}", *(f"strict_{smp}_s{s}" for s in range(1, 5))]
                   if (RUNS / n / "trials.jsonl").exists()] for smp in SAMPLERS}
    if not any(len(v) > 1 for v in names.values()):
        return
    runs = {n: load_trials(RUNS / n) for v in names.values() for n in v}
    rows = {r["label"]: r for r in report.seed_table(runs, {k: v for k, v in names.items() if v})}
    out += ["## E1: samplers across seeds (ugc_strict, composite, 40 trials)", "",
            "| sampler | seeds | final best (mean ± sd) | trial reaching 95 % of final gain (mean ± sd) |",
            "|---|---|---|---|"]
    for r in rows.values():
        out.append(f"| {r['label']} | {r['n']} | {r['final_mean']:.2f} ± {r['final_sd']:.2f} | "
                   f"{r['t95_mean']:.1f} ± {r['t95_sd']:.1f} |")
    out += ["", "Pairwise rule: A beats B only if mean(A) − mean(B) > max(sd(A), sd(B)).", ""]
    labels = list(rows)
    wins = []
    for a in labels:
        for b in labels:
            ra, rb = rows[a], rows[b]
            if a != b and ra["final_mean"] - rb["final_mean"] > max(ra["final_sd"], rb["final_sd"]):
                wins.append(f"{a} beats {b} ({ra['final_mean'] - rb['final_mean']:+.2f})")
    out += [f"* {w}" for w in wins] or ["* no difference at these seeds"]
    out.append("")

    gp = [load(n) for n in names["gp"]]
    gs = [load(f"surrogate_trust_s{s}") for s in range(3)]
    gp, gs = [r for r in gp if r], [r for r in gs if r]
    if gp and gs:
        clips = [c for c in gs[0]["clips"] if "G_r0" in gs[0]["clips"][c]]
        out += [f"Learned G (round 0, {len(gs)} seeds) vs GP-BO θ ({len(gp)} seeds), composite and SSIMULACRA2, "
                "mean ± sd over seeds:", "",
                "| clip | tier-only | GP-BO θ composite | G composite | GP-BO θ ssimulacra2 | G ssimulacra2 |",
                "|---|---|---|---|---|---|"]
        for c in clips:
            tier = gs[0]["clips"][c]["upscale_only"]["composite"]
            th = [r["clips"][c]["best"] for r in gp if c in r["clips"]]
            g = [r["clips"][c]["G_r0"] for r in gs if "G_r0" in r["clips"].get(c, {})]
            tag = " *(holdout)*" if c not in gs[0]["train_clips"] else ""
            out.append(f"| {c}{tag} | {tier:.2f} | {ms([m['composite'] for m in th])} | "
                       f"{ms([m['composite'] for m in g])} | {ms([m.get('ssimulacra2', np.nan) for m in th])} | "
                       f"{ms([m.get('ssimulacra2', np.nan) for m in g])} |")
        out.append("")


def main() -> int:
    out = ["# P0 tables (generated by scripts/p0_tables.py)", ""]
    for fn in (e2, e3_e5, e4, e1):
        fn(out)
    text = "\n".join(out) + "\n"
    (RUNS / "p0_tables.md").write_text(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
