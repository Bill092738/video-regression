"""Tables, decision rules and figures for P1 (docs/P1-plan.md), from the runs' rescore_p1 / rescore_confirm files.

    uv run python scripts/p1_tables.py            # -> runs/p1_tables.md, docs/img/p1/*.png, runs/p1_g_loss.txt
    uv run python scripts/p1_tables.py --g-kinds  # the G kinds that E23 must evaluate (from decisions.json)

Sections whose runs are missing are skipped, so this can be run while experiments are in progress.
All gains are signed so that positive = better. CIs are bootstrap 95 % intervals over clips (10,000
resamples); a clip's value is the mean over the seeds that have it. Baseline tables (E20) use the
train and dev clips only: the confirm split is looked at once, in E23.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from vreg import classify  # noqa: E402
from vreg.rescore import METRICS, improvement, judges_for  # noqa: E402

RUNS = Path("runs")
IMG = Path("docs/img/p1")
DECISIONS = Path("docs/p1/decisions.json")
LOSSES = ("lpips", "dists", "lpips+dists")
G_KINDS = {"g": "global G (@natural:train)", "specA": "specialist A (@A:train)", "specB": "specialist B (@B:train)"}
SEEDS = (0, 1, 2)
N_BOOT = 10_000
JUDGE_SET = ("ssimulacra2", "dists", "xpsnr", "vmaf_neg")


def not_judged(metric: str, loss: str) -> str:
    """Why a metric is not one of a G's judges: it is in G's loss, or simply not in its judge set."""
    return "(own loss) " if metric in loss.split("+") else "(not a judge) "


def tag(loss: str) -> str:
    return loss.replace("+", "_")


def load(run: str, which: str = "p1") -> dict | None:
    f = RUNS / run / f"rescore_{which}.json"
    return json.loads(f.read_text()) if f.exists() else None


def bootstrap(x, alpha: float = 0.05, n: int = N_BOOT, seed: int = 0) -> tuple[float, float, float]:
    """Mean and the bootstrap (1 - alpha) percentile interval of the mean over clips."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, len(x), (n, len(x)))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def fmt(metric: str, m: float, lo: float | None = None, hi: float | None = None) -> str:
    f = (lambda v: f"{v:+.4f}") if metric in ("lpips", "dists") else (lambda v: f"{v:+.2f}")
    if np.isnan(m):
        return "–"
    s = f(m)
    if lo is not None:
        s += f" [{f(lo)}, {f(hi)}]"
        if lo > 0 or hi < 0:
            s = f"**{s}**"
    return s


class Data:
    """Per configuration, per clip: the metrics dict (G configurations: one dict per seed)."""

    def __init__(self):
        self.classes = classify.load()
        self.cls = {n: c["class"] for n, c in self.classes["clips"].items()}
        self.split = {n: c["split"] for n, c in self.classes["clips"].items()}
        self.base: dict[str, dict[str, dict]] = {"naive": {}, "tier": {}, "theta_global": {}, "theta_class": {}}
        glob = load("strict_gp_noeq")
        if glob:
            for clip, row in glob["clips"].items():
                self.base["naive"][clip] = row["naive"]
                self.base["tier"][clip] = row["upscale_only"]
                self.base["theta_global"][clip] = row["best"]
        for k in classify.CLASSES:
            res = load(f"p1_sharp_{k}")
            if res:
                for clip, row in res["clips"].items():
                    if self.cls.get(clip) == k:
                        self.base["theta_class"][clip] = row["best"]
                        self.base["tier"].setdefault(clip, row["upscale_only"])
                        self.base["naive"].setdefault(clip, row["naive"])

    def g(self, kind: str, loss: str, which: str = "p1", seeds=SEEDS) -> dict[str, list[dict]]:
        """clip -> [metrics of each seed]"""
        out: dict[str, list[dict]] = {}
        for s in seeds:
            res = load(f"p1_{kind}_{tag(loss)}_s{s}", which)
            for clip, row in (res or {}).get("clips", {}).items():
                if "G_r0" in row:
                    out.setdefault(clip, []).append(row["G_r0"])
        return out

    def clips(self, k: str, splits) -> list[str]:
        return [n for n in self.cls if (k == "*" or self.cls[n] == k) and self.split[n] in splits]


def gains(a: dict[str, dict | list[dict]], b: dict[str, dict], clips, metric: str) -> list[float]:
    """Per clip improvement of a over b (a G entry is a list over seeds: its gain is the seed mean)."""
    out = []
    for c in clips:
        if c not in a or c not in b or metric not in b[c]:
            continue
        rows = a[c] if isinstance(a[c], list) else [a[c]]
        vals = [improvement(metric, r[metric], b[c][metric]) for r in rows if metric in r]
        if vals:
            out.append(float(np.mean(vals)))
    return out


def seed_gains(a: dict[str, list[dict]], b: dict[str, dict], clips, metric: str) -> list[float]:
    """Mean gain over clips, separately for each seed index."""
    n = max((len(v) for c, v in a.items() if c in clips), default=0)
    return [float(np.mean([improvement(metric, a[c][s][metric], b[c][metric]) for c in clips
                           if c in a and c in b and len(a[c]) > s and metric in a[c][s]] or [np.nan]))
            for s in range(n)]


# ---------------------------------------------------------------------------------------------- sections

def e19(d: Data, out: list[str]):
    r = d.classes["rules"]
    out += ["## E19: content classes and splits", "",
            f"B if SI ≥ {r['si_min_for_B']:.1f} or TI ≥ {r['ti_min_for_B']:.1f} (66th percentiles over the natural "
            f"clips not in C); C if flat_frac ≥ {r['flat_frac_min_for_C']} or dark_frac ≥ {r['dark_frac_min_for_C']}.",
            "", "| class | train | dev | confirm | report only | total | sources |", "|---|---|---|---|---|---|---|"]
    for k, c in d.classes["counts"].items():
        groups = {v["source_group"] for v in d.classes["clips"].values() if v["class"] == k}
        out.append(f"| {k} | {c['train']} | {c['dev']} | {c['confirm']} | {c.get('report', 0)} | {sum(c.values())} "
                   f"| {len(groups)} |")
    if d.classes.get("report_only"):
        out.append("")
        out.append(f"Report-only classes (no split, never trained on or tested, listed only): "
                   f"{', '.join(d.classes['report_only'])}. Pre-registered: {json.dumps(d.classes.get('pre_registered'))}.")
    out.append("")


def e20(d: Data, out: list[str]) -> dict:
    if not d.base["tier"]:
        return {}
    out += ["## E20: baselines per class (train + dev clips; gain over tier-only, mean [95 % CI])", ""]
    summary = {}
    for k in classify.CLASSES:
        if k in d.classes.get("report_only", ()):
            clips = d.clips(k, (classify.REPORT,))
            out += [f"**Class {k}** (report only, {len(clips)} clip{'s' * (len(clips) != 1)}; no CI, not tested).", "",
                    "| clip | configuration | " + " | ".join(METRICS) + " |", "|---" * (len(METRICS) + 2) + "|"]
            for c in clips:
                for label, conf in (("naive", "naive"), ("P0 global sharpening-only θ", "theta_global")):
                    if c in d.base[conf] and c in d.base["tier"]:
                        out.append(f"| {c} | {label} | " + " | ".join(
                            fmt(m, improvement(m, d.base[conf][c][m], d.base["tier"][c][m])) for m in METRICS) + " |")
            out.append("")
            continue
        clips = d.clips(k, ("train", "dev"))
        theta = load(f"p1_sharp_{k}")
        if theta:
            p = theta["configs"]["best"]
            out.append(f"**Class {k}** ({len(clips)} clips). Class-tuned sharpening-only θ: unsharp "
                       f"{p['sharpen']:.2f}, radius {p['sharpen_radius']:.0f} "
                       f"(size {2 * int(p['sharpen_radius']) + 1}).")
        else:
            out.append(f"**Class {k}** ({len(clips)} clips).")
        out += ["", "| configuration | " + " | ".join(METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]
        for label, conf in (("naive", "naive"), ("P0 global sharpening-only θ", "theta_global"),
                            ("class-tuned sharpening-only θ", "theta_class")):
            cells = []
            for m in METRICS:
                g = gains(d.base[conf], d.base["tier"], clips, m)
                cells.append(fmt(m, *bootstrap(g)) + (f" (n={len(g)})" if m == "composite" else ""))
                summary.setdefault(k, {}).setdefault(conf, {})[m] = bootstrap(g)
            out.append(f"| {label} | " + " | ".join(cells) + " |")
        out.append("")
    return summary


def e21(d: Data, out: list[str]) -> str | None:
    clips = d.clips("A", ("dev",))
    rows, score = [], {}
    for loss in LOSSES:
        g = d.g("g", loss, seeds=(0,))  # E21 is defined on seed 0; later seeds belong to E22
        if not any(c in g for c in clips):
            continue
        judges = judges_for(loss)
        cells, pos = [], 0
        for m in JUDGE_SET:
            gv = gains(g, d.base["theta_class"], clips, m)
            mean, lo, hi = bootstrap(gv)
            own = m not in judges
            cells.append((not_judged(m, loss) if own else "") + fmt(m, mean, lo, hi))
            pos += (not own) and mean > 0
        s2 = np.mean(gains(g, d.base["theta_class"], clips, "ssimulacra2") or [np.nan])
        score[loss] = (pos, s2)
        rows.append(f"| {loss} | " + " | ".join(cells) + f" | {pos}/3 |")
    if not rows:
        return None
    out += ["## E21: G loss selection (global G, seed 0, class A dev; gain over class-tuned sharpening-only θ)", "",
            "| G loss | " + " | ".join(JUDGE_SET) + " | judges with positive mean |",
            "|---" * (len(JUDGE_SET) + 2) + "|", *rows, ""]
    chosen = None
    if len(score) == len(LOSSES):
        chosen = max(LOSSES, key=lambda L: (score[L][0], score[L][1]))
        out += [f"Chosen loss (most judges positive, SSIMULACRA2 tie-break): **{chosen}**.", ""]
        f = RUNS / "p1_g_loss.txt"
        if f.exists() and f.read_text().strip() != chosen:
            # the choice is fixed once made: E22 and E23 were run with the recorded loss
            raise SystemExit(f"E21 recomputes {chosen!r} but {f} records {f.read_text().strip()!r}")
        f.write_text(chosen + "\n")
    else:
        out += [f"(only {len(score)} of {len(LOSSES)} losses done; no choice yet)", ""]
    return chosen


def e22(d: Data, out: list[str], loss: str) -> dict:
    judges = judges_for(loss)
    cols = ("composite", *JUDGE_SET)
    summary = {}
    any_rows = False
    for ref_label, ref in (("tier-only", "tier"), ("class-tuned sharpening-only θ", "theta_class")):
        lines = [f"### Gain over {ref_label} (dev clips, mean over seeds per clip, [95 % CI over clips])", "",
                 "| class | configuration | seeds | " + " | ".join(cols) + " |", "|---" * (len(cols) + 3) + "|"]
        for k in classify.CLASSES:
            clips = d.clips(k, ("dev",))
            confs = [(G_KINDS[kind], d.g(kind, loss), kind) for kind in G_KINDS]
            if ref == "tier":
                confs = [("class-tuned θ", d.base["theta_class"], "theta_class"),
                         ("P0 global θ", d.base["theta_global"], "theta_global"), *confs]
            for label, a, key in confs:
                if not any(c in a for c in clips):
                    continue
                any_rows = True
                nseeds = max((len(a[c]) for c in clips if c in a and isinstance(a[c], list)), default="–")
                cells = []
                for m in cols:
                    st = bootstrap(gains(a, d.base[ref], clips, m))
                    summary.setdefault((ref, k, key), {})[m] = st
                    own = m not in judges and m != "composite" and key in G_KINDS
                    cells.append((not_judged(m, loss) if own else "") + fmt(m, *st))
                lines.append(f"| {k} ({len(clips)}) | {label} | {nseeds} | " + " | ".join(cells) + " |")
        lines.append("")
        out += lines
    if not any_rows:
        return {}
    # per-seed signs of G vs class theta, per class
    out += ["### Sign of the mean gain over class-tuned θ, per seed (judges of the chosen loss)", "",
            "| class | configuration | " + " | ".join(judges) + " |", "|---" * (len(judges) + 2) + "|"]
    for k in classify.CLASSES:
        clips = d.clips(k, ("dev",))
        for kind in G_KINDS:
            a = d.g(kind, loss)
            if not any(c in a for c in clips):
                continue
            cells = [" ".join("+" if v > 0 else "−" if v < 0 else "?" for v in seed_gains(a, d.base["theta_class"],
                                                                                    clips, m)) for m in judges]
            out.append(f"| {k} | {G_KINDS[kind]} | " + " | ".join(cells) + " |")
    out.append("")
    return summary


def decisions() -> dict | None:
    return json.loads(DECISIONS.read_text()) if DECISIONS.exists() else None


def gate_kinds(dec: dict) -> list[str]:
    kinds = [dec["g_for_A"]]
    for choice in dec["gate"].values():
        if choice.startswith("G"):
            kinds.append(choice.partition(":")[2] or dec["g_for_A"])
    return list(dict.fromkeys(kinds))


def e23(d: Data, out: list[str], loss: str) -> None:
    dec = decisions()
    if not dec:
        return
    for k, choice in d.classes.get("pre_registered", {}).get("gate", {}).items():
        if dec["gate"].get(k) != choice:
            raise SystemExit(f"decisions.json gate for class {k} must be the pre-registered {choice!r}")
    g = d.g(dec["g_for_A"], loss, "confirm")
    if not g:
        out += ["## E23: confirmation", "", "(decisions recorded; confirm split not evaluated yet)", ""]
        return
    judges = judges_for(loss)
    nominated = dec.get("nominated")
    alpha = 0.025 if nominated else 0.05
    out += [f"## E23: confirmation (confirm split, run once; G = {G_KINDS[dec['g_for_A']]}, loss {loss})", ""]
    for k in ["A", *([nominated] if nominated else [])]:
        clips = d.clips(k, ("confirm",))
        out += [f"### H{'1' if k == 'A' else ' (nominated)'}: class {k}, G vs class-tuned sharpening-only θ "
                f"({len(clips)} clips, {100 * (1 - alpha):g} % CI)", "",
                "| judge | gain [CI] | CI excludes 0 (positive) | positive seeds |", "|---|---|---|---|"]
        passed = []
        for m in judges:
            mean, lo, hi = bootstrap(gains(g, d.base["theta_class"], clips, m), alpha=alpha)
            sg = seed_gains(g, d.base["theta_class"], clips, m)
            ok = lo > 0
            if ok:
                passed.append((m, sum(v > 0 for v in sg)))
            out.append(f"| {m} | {fmt(m, mean, lo, hi)} | {'yes' if ok else 'no'} | {sum(v > 0 for v in sg)}/{len(sg)} |")
        s2 = np.mean(gains(g, d.base["theta_class"], clips, "ssimulacra2") or [np.nan])
        c1 = len(passed) >= 2
        c2 = s2 >= 0
        c3 = c1 and all(n >= 2 for _, n in passed)
        out += ["", f"1. ≥ 2 judges with a CI above 0: **{'yes' if c1 else 'no'}** ({len(passed)})",
                f"2. SSIMULACRA2 mean gain ≥ 0: **{'yes' if c2 else 'no'}** ({s2:+.2f})",
                f"3. sign holds in ≥ 2 of 3 seeds for those judges: **{'yes' if c3 else 'no'}**", "",
                f"**H{'1' if k == 'A' else ''} ({k}): {'confirmed' if c1 and c2 and c3 else 'not confirmed'}.**", ""]
    # gate
    gate, kinds = dec["gate"], {}
    for k, choice in gate.items():
        kinds[k] = choice
    clips_all = d.clips("*", ("confirm",))
    gated = {}
    for c in clips_all:
        choice = kinds.get(d.cls[c], "tier")
        if choice.startswith("G"):
            src = d.g(choice.partition(":")[2] or dec["g_for_A"], loss, "confirm")
            if c in src:
                gated[c] = {m: float(np.mean([r[m] for r in src[c] if m in r])) for m in METRICS if m in src[c][0]}
        elif choice == "theta" and c in d.base["theta_class"]:
            gated[c] = d.base["theta_class"][c]
        elif c in d.base["tier"]:
            gated[c] = d.base["tier"][c]
    out += ["### Gate on the confirm split", "", "Gate table (chosen on dev): " +
            ", ".join(f"{k} → {v}" for k, v in gate.items()) + ".", "",
            "| class | clips | SSIMULACRA2 gate − tier | XPSNR gate − tier | not worse than tier-only |",
            "|---|---|---|---|---|"]
    ok_classes = True
    for k in classify.CLASSES:
        cl = [c for c in d.clips(k, ("confirm",)) if c in gated]
        if k in d.classes.get("report_only", ()) or not cl:
            # report-only classes have no confirm clips and are excluded from every test (E19)
            out.append(f"| {k} | 0 | – | – | not tested (report only) |")
            continue
        s2 = bootstrap(gains(gated, d.base["tier"], cl, "ssimulacra2"))
        xp = bootstrap(gains(gated, d.base["tier"], cl, "xpsnr"))
        ok = s2[0] >= 0 and xp[0] >= 0
        ok_classes &= ok
        out.append(f"| {k} | {len(cl)} | {fmt('ssimulacra2', *s2)} | {fmt('xpsnr', *xp)} | {'yes' if ok else 'no'} |")
    wins = []
    out += ["", "| judge | gate − P0 global sharpening-only θ, all confirm clips |", "|---|---|"]
    for m in judges:
        st = bootstrap(gains(gated, d.base["theta_global"], clips_all, m))
        wins.append(st[0] >= 0)
        out.append(f"| {m} | {fmt(m, *st)} |")
    accepted = ok_classes and sum(wins) >= 2
    explore = ["### Exploratory: every configuration on the confirm split, per class (H2, H3)", "",
               "Gain over tier-only, mean [95 % CI over clips]; G is the mean over its 3 seeds per clip.", "",
               "| class | configuration | " + " | ".join(("composite", *judges)) + " |",
               "|---" * (len(judges) + 3) + "|"]
    for k in classify.CLASSES:
        cl = d.clips(k, ("confirm",))
        if not cl:
            continue
        for label, a in (("class-tuned θ", d.base["theta_class"]), ("P0 global θ", d.base["theta_global"]),
                         (G_KINDS[dec["g_for_A"]], g)):
            explore.append(f"| {k} ({len(cl)}) | {label} | " + " | ".join(
                fmt(m, *bootstrap(gains(a, d.base["tier"], cl, m))) for m in ("composite", *judges)) + " |")
        explore.append(f"| {k} ({len(cl)}) | G − class-tuned θ | " + " | ".join(
            fmt(m, *bootstrap(gains(g, d.base["theta_class"], cl, m))) for m in ("composite", *judges)) + " |")
    explore.append("")
    summary = {k: {key: {m: bootstrap(gains(a, d.base["tier"], d.clips(k, ("confirm",)), m)) for m in judges}
                   for key, a in (("theta_class", d.base["theta_class"]), ("theta_global", d.base["theta_global"]),
                                  ("G", g))}
               for k in classify.CLASSES if d.clips(k, ("confirm",))}
    bar_figure(summary, [("class-tuned θ", "theta_class"), ("P0 global θ", "theta_global"),
                         (G_KINDS[dec["g_for_A"]], "G")], judges,
               "E23: confirm split, gain over tier-only (95 % CI over clips)", IMG / "e23_confirm.png")
    out += ["", f"**Gate: {'accepted' if accepted else 'not accepted'}** (every class ≥ tier-only on SSIMULACRA2 "
            f"and XPSNR: {'yes' if ok_classes else 'no'}; ≥ global θ on {sum(wins)}/{len(judges)} judges).", "",
            *explore]


def e14(d: Data, out: list[str]) -> None:
    """Exploratory addendum: the P1 recipe on sim:ugc_av1 (transferred from ugc_strict vs tuned natively)."""
    glob = load("strict_gp_noeq", "av1")
    if not glob:
        return
    tier = {c: r["upscale_only"] for c, r in glob["clips"].items()}
    confs = {"P0 global θ (ugc_strict)": {c: r["best"] for c, r in glob["clips"].items()},
             "class θ tuned on ugc_strict": {}, "class θ tuned on ugc_av1": {}}
    params = {}
    for k in classify.CLASSES:
        for label, run in (("class θ tuned on ugc_strict", f"p1_sharp_{k}"), ("class θ tuned on ugc_av1", f"p1av1_sharp_{k}")):
            res = load(run, "av1")
            if res:
                params[(label, k)] = res["configs"]["best"]
                confs[label].update({c: r["best"] for c, r in res["clips"].items() if d.cls.get(c) == k})
    cols = ("composite", "ssimulacra2", "dists", "xpsnr")
    out += ["## E14 addendum (exploratory): the P1 recipe on sim:ugc_av1", "",
            "Same frozen classes and splits; AV1 tier at ×2 and no ingest denoise. Gain over ugc_av1 tier-only on "
            "each class's dev + confirm clips (neither θ was tuned on them), mean [95 % CI over clips]. The "
            "confirm split was already used once in E23; nothing here changes a P1 verdict.", ""]
    for (label, k), p in sorted(params.items(), key=lambda x: x[0][1]):
        out.append(f"* {k}, {label}: unsharp {p['sharpen']:.2f}, size {2 * int(p['sharpen_radius']) + 1}")
    out += ["", "| class | configuration | " + " | ".join(cols) + " |", "|---" * (len(cols) + 2) + "|"]
    summary = {}
    for k in classify.CLASSES:
        clips = d.clips(k, ("dev", "confirm"))
        for label, a in confs.items():
            if not any(c in a for c in clips):
                continue
            sts = {m: bootstrap(gains(a, tier, clips, m)) for m in cols}
            summary.setdefault(k, {})[label] = sts
            out.append(f"| {k} ({len(clips)}) | {label} | " + " | ".join(fmt(m, *sts[m]) for m in cols) + " |")
        a, b = confs["class θ tuned on ugc_av1"], confs["class θ tuned on ugc_strict"]
        if any(c in a for c in clips) and any(c in b for c in clips):
            out.append(f"| {k} ({len(clips)}) | ugc_av1-tuned − ugc_strict-tuned θ | " + " | ".join(
                fmt(m, *bootstrap(gains(a, b, clips, m))) for m in cols) + " |")
    out.append("")
    dec = decisions()
    if dec:
        out += ["Gate check on ugc_av1 (confirm clips; the P1 gate table with its θ either transferred or tuned "
                "on ugc_av1): is any class below ugc_av1 tier-only on SSIMULACRA2 or XPSNR?", "",
                "| class | gate | θ source | SSIMULACRA2 gate − tier | XPSNR gate − tier |", "|---|---|---|---|---|"]
        for k in classify.CLASSES:
            cl = d.clips(k, ("confirm",))
            if not cl or dec["gate"].get(k) != "theta":
                if cl:
                    out.append(f"| {k} | tier-only | – | +0.00 | +0.00 |")
                continue
            for label in ("class θ tuned on ugc_strict", "class θ tuned on ugc_av1"):
                if any(c in confs[label] for c in cl):
                    out.append(f"| {k} | θ | {label} | " + " | ".join(
                        fmt(m, *bootstrap(gains(confs[label], tier, cl, m))) for m in ("ssimulacra2", "xpsnr")) + " |")
        out.append("")
    bar_figure(summary, [(lb, lb) for lb in confs], ("ssimulacra2", "dists", "xpsnr"),
               "E14 addendum: sharpening-only θ on ugc_av1 vs its tier-only (dev + confirm, 95 % CI)",
               IMG / "e14_av1.png")


# ---------------------------------------------------------------------------------------------- figures

def bar_figure(summary: dict, rows: list[tuple[str, str]], metrics: tuple[str, ...], title: str, path: Path):
    """summary[(row key)][metric] = (mean, lo, hi); one panel per metric, one bar group per class."""
    if not summary:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.6))
    width = 0.8 / len(rows)
    for ax, m in zip(np.atleast_1d(axes), metrics):
        for i, (label, key) in enumerate(rows):
            xs, ys, err = [], [], []
            for j, k in enumerate(classify.CLASSES):
                st = summary.get(k, {}).get(key, {}).get(m)
                if st and not np.isnan(st[0]):
                    xs.append(j + (i - (len(rows) - 1) / 2) * width)
                    ys.append(st[0])
                    err.append([st[0] - st[1], st[2] - st[0]])
            if xs:
                ax.bar(xs, ys, width, yerr=np.array(err).T, capsize=3, label=label)
        ax.axhline(0, color="0.3", lw=0.8)
        ax.set_xticks(range(len(classify.CLASSES)), [f"class {k}" for k in classify.CLASSES])
        ax.set_title(m + " gain")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> int:
    d = Data()
    if "--g-kinds" in sys.argv:
        dec = decisions()
        if not dec:
            print("docs/p1/decisions.json missing", file=sys.stderr)
            return 1
        print(" ".join(gate_kinds(dec)))
        return 0
    out = ["# P1 tables (generated by scripts/p1_tables.py)", ""]
    e19(d, out)
    s20 = e20(d, out)
    bar_figure(s20, [("P0 global θ", "theta_global"), ("class-tuned θ", "theta_class")],
               ("ssimulacra2", "dists", "xpsnr"), "E20: sharpening-only θ vs tier-only (train + dev clips, 95 % CI)",
               IMG / "e20_baselines.png")
    loss = e21(d, out)
    if not loss and (RUNS / "p1_g_loss.txt").exists():
        loss = (RUNS / "p1_g_loss.txt").read_text().strip()
    if loss:
        s22 = e22(d, out, loss)
        if s22:
            by_class = {}
            for (ref, k, key), v in s22.items():
                if ref == "theta_class":
                    by_class.setdefault(k, {})[key] = v
            bar_figure(by_class, [(G_KINDS[g], g) for g in G_KINDS], judges_for(loss),
                       f"E22: G ({loss}) vs class-tuned sharpening-only θ (dev clips, 95 % CI)", IMG / "e22_dev.png")
        e23(d, out, loss)
    e14(d, out)
    (RUNS / "p1_tables.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
