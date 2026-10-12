"""Plots and a self-contained HTML report for one or more optimisation runs."""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import content, ff, precomp  # noqa: E402
from .optimize import load_trials  # noqa: E402

PALETTE = ["#2a6fdb", "#e0731f", "#2c9f6b", "#c0399f", "#8c6d1f", "#5b5b5b"]
METRIC_COLS = ("composite", "vmaf_neg", "vmaf", "ssim", "lpips", "cambi", "playback_kbps")


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _img_tag(png: bytes, alt: str) -> str:
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{base64.b64encode(png).decode()}">'


def convergence_plot(runs: dict[str, list[dict]]) -> bytes:
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for i, (name, trials) in enumerate(runs.items()):
        vals = np.array([t["value"] for t in trials])
        if not len(vals):
            continue
        c = PALETTE[i % len(PALETTE)]
        ax.plot(np.maximum.accumulate(vals), color=c, lw=2, label=name)
        ax.scatter(range(len(vals)), vals, color=c, s=10, alpha=0.35)
    first = next((t for t in runs.values() if t), None)
    if first:
        ax.axhline(first[0]["value"], color="#888", ls="--", lw=1, label="naive upload (trial 0)")
    ax.set_xlabel("trial (= one upload/transcode/measure cycle)")
    ax.set_ylabel("objective")
    ax.set_title("Best-so-far objective per sampler")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    return _png(fig)


def sensitivity_plot(trials: list[dict]) -> bytes:
    names = precomp.PARAM_NAMES
    cols = 4
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.6 * rows), sharey=True)
    vals = np.array([t["value"] for t in trials])
    best = int(np.argmax(vals))
    for ax, name in zip(axes.flat, names):
        xs = np.array([t["params"][name] for t in trials])
        ax.scatter(xs, vals, s=12, color=PALETTE[0], alpha=0.6)
        ax.scatter([xs[best]], [vals[best]], s=50, color=PALETTE[1], marker="*", zorder=3)
        ax.set_title(name, fontsize=9)
        ax.grid(alpha=0.25)
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    fig.suptitle("Objective vs. each pre-compensation parameter (star = best)", fontsize=10)
    fig.tight_layout()
    return _png(fig)


def crops_figure(src: Path, playbacks: dict[str, Path], frame: int = 45, crop: float = 0.3) -> bytes:
    """Zoomed crops of the same frame: source vs. playback variants."""
    ref = ff.read_frames(src)
    n = min(frame, len(ref) - 1)
    h, w = ref.shape[1:3]
    ch, cw = int(h * crop), int(w * crop)
    # crop around the most detailed region of the reference frame
    gray = ref[n].mean(axis=2)
    grad = np.abs(np.diff(gray, axis=0))[:, :-1] + np.abs(np.diff(gray, axis=1))[:-1, :]
    best, by, bx = -1.0, 0, 0
    for y in range(0, h - ch, max(1, ch // 3)):
        for x in range(0, w - cw, max(1, cw // 3)):
            s = grad[y:y + ch, x:x + cw].mean()
            if s > best:
                best, by, bx = s, y, x
    panels = {"source": ref[n]}
    for label, path in playbacks.items():
        panels[label] = ff.read_frames(path, size=(w, h))[n]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 2.8))
    for ax, (label, img) in zip(np.atleast_1d(axes), panels.items()):
        ax.imshow(img[by:by + ch, bx:bx + cw], interpolation="nearest")
        ax.set_title(label, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    return _png(fig)


def _metrics_table(rows: dict[str, dict]) -> str:
    head = "".join(f"<th>{c}</th>" for c in METRIC_COLS)
    body = []
    for label, m in rows.items():
        cells = "".join(
            f"<td>{m[c]:.3f}</td>" if isinstance(m.get(c), (int, float)) else "<td>-</td>" for c in METRIC_COLS
        )
        body.append(f"<tr><th>{html.escape(label)}</th>{cells}</tr>")
    return f"<table><tr><th></th>{head}</tr>{''.join(body)}</table>"


CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1f;--muted:#6b6b70;--line:#e2e2e0;--card:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--fg:#ececee;--muted:#a0a0a8;--line:#2c2c30;--card:#1c1c1f}}
body{background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,sans-serif;margin:0;padding:24px 16px}
main{max-width:1100px;margin:auto}h1{font-size:1.6rem}h2{margin-top:2.2rem;font-size:1.2rem}
.muted{color:var(--muted)}img{max-width:100%;background:#fff;border-radius:6px}
table{border-collapse:collapse;font-size:13px;margin:.6rem 0;display:block;overflow-x:auto}
td,th{border-bottom:1px solid var(--line);padding:4px 10px;text-align:right}th:first-child{text-align:left}
code{font-size:12px}
"""


def _tier_only(cfg: dict, clip: str, upscale: float) -> dict | None:
    """Naive content at the same upload tier as the best params: isolates the filter gains."""
    from .evaluate import evaluate
    from .optimize import cache_root
    from .platform import make_platform

    try:
        plat = make_platform(cfg["platform"])
        r = evaluate(clip, Path(cfg["clips"][clip]), {"upscale": upscale}, plat, cache_root(plat))
        return r.metrics | {"workdir": r.workdir}
    except Exception:
        return None


def _save(png: bytes, png_dir: Path | None, name: str) -> bytes:
    if png_dir:
        png_dir.mkdir(parents=True, exist_ok=True)
        (png_dir / name).write_bytes(png)
    return png


def build_report(run_dirs: list[Path], out: Path, title: str = "Pre-compensation report",
                 png_dir: Path | None = None) -> Path:
    runs = {d.name: load_trials(d) for d in run_dirs}
    runs = {k: v for k, v in runs.items() if v}
    if not runs:
        raise SystemExit("no trials found in the given run directories")
    parts = [f"<h1>{html.escape(title)}</h1>"]
    cfg0 = json.loads((run_dirs[0] / "config.json").read_text())
    parts.append(f"<p class=muted>platform <code>{html.escape(cfg0['platform'])}</code> · "
                 f"objective <code>{cfg0['objective']}</code> · clips {', '.join(cfg0['clips'])}</p>")

    summary = ["<table><tr><th>run</th><th>trials</th><th>naive</th><th>best</th><th>gain</th>"
               "<th>best at trial</th></tr>"]
    for name, trials in runs.items():
        vals = [t["value"] for t in trials]
        b = int(np.argmax(vals))
        summary.append(f"<tr><th>{html.escape(name)}</th><td>{len(vals)}</td><td>{vals[0]:.2f}</td>"
                       f"<td>{vals[b]:.2f}</td><td>{vals[b] - vals[0]:+.2f}</td><td>{trials[b]['trial']}</td></tr>")
    summary.append("</table>")
    parts += ["<h2>Summary</h2>", *summary,
              _img_tag(_save(convergence_plot(runs), png_dir, "convergence.png"), "convergence")]

    # overall best across runs drives the detailed section
    best_run, best_trial = max(((n, t) for n, ts in runs.items() for t in ts), key=lambda x: x[1]["value"])
    baseline = runs[best_run][0]
    parts.append(f"<h2>Best parameters <span class=muted>(run {html.escape(best_run)}, "
                 f"trial {best_trial['trial']})</span></h2>")
    parts.append("<table><tr><th>param</th><th>value</th><th>identity</th><th>meaning</th></tr>" + "".join(
        f"<tr><th>{p.name}</th><td>{best_trial['params'][p.name]:.3f}</td><td>{p.identity}</td>"
        f"<td style='text-align:left'>{html.escape(p.doc)}</td></tr>" for p in precomp.SPACE) + "</table>")
    parts.append(f"<p>ffmpeg filter: <code>{html.escape(precomp.build_filter(best_trial['params'], 720))}</code></p>")

    for clip, m_best in best_trial["per_clip"].items():
        m_base = baseline["per_clip"][clip]
        parts.append(f"<h2>Clip: {html.escape(clip)}</h2>")
        up = best_trial["params"]["upscale"]
        rows = {"naive upload": m_base}
        tier = _tier_only(cfg0, clip, up) if up > 1 else None
        if tier:
            rows[f"upscale x{up:g} only"] = tier
        rows["pre-compensated (best)"] = m_best
        parts.append(_metrics_table(rows))
        src = Path(cfg0["clips"][clip])
        try:
            panels = {"naive upload": Path(m_base["workdir"]) / "playback.mp4"}
            if tier:
                panels[f"upscale x{up:g} only"] = Path(tier["workdir"]) / "playback.mp4"
            panels["pre-compensated (best)"] = Path(m_best["workdir"]) / "playback.mp4"
            fig = _save(crops_figure(src, panels), png_dir, f"crops_{clip}.png")
            parts.append(_img_tag(fig, f"crops {clip}"))
        except Exception as e:  # media may have been cleaned up
            parts.append(f"<p class=muted>(no frames: {html.escape(str(e))})</p>")

    for d in run_dirs:
        tf = d / "transfer.json"
        if tf.exists():
            t = json.loads(tf.read_text())
            parts.append(f"<h2>Transfer to other clips <span class=muted>(best params of {html.escape(d.name)})"
                         "</span></h2><table><tr><th>clip</th><th>naive</th><th>upscale only</th>"
                         "<th>best params</th><th>gain vs naive</th><th>gain vs upscale only</th></tr>")
            for clip, r in t["clips"].items():
                n, u, b = r["naive"]["composite"], r["upscale_only"]["composite"], r["best"]["composite"]
                tag = " <span class=muted>(train)</span>" if clip in t["train_clips"] else ""
                parts.append(f"<tr><th>{html.escape(clip)}{tag}</th><td>{n:.2f}</td><td>{u:.2f}</td><td>{b:.2f}</td>"
                             f"<td>{b - n:+.2f}</td><td>{b - u:+.2f}</td></tr>")
            parts.append("</table>")

    for name, trials in runs.items():
        if len(trials) >= 5:
            parts.append(f"<h2>Parameter sensitivity: {html.escape(name)}</h2>")
            parts.append(_img_tag(_save(sensitivity_plot(trials), png_dir, f"sensitivity_{name}.png"),
                                  f"sensitivity {name}"))

    return _write(out, title, parts)


def _write(out: Path, title: str, parts: list[str]) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"<!doctype html><html lang=en><meta charset=utf-8>"
                   f"<meta name=viewport content='width=device-width,initial-scale=1'>"
                   f"<title>{html.escape(title)}</title><style>{CSS}</style><main>{''.join(parts)}</main></html>")
    return out


def surrogate_plot(hist: dict) -> bytes:
    rounds = hist["rounds"]
    clips = list(rounds[0]["clips"])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    for i, clip in enumerate(clips):
        c = PALETTE[i % len(PALETTE)]
        ys = [r["clips"][clip]["metrics"]["composite"] - hist["baselines"][clip]["upscale_only"]["composite"]
              for r in rounds]
        hold = rounds[0]["clips"][clip]["holdout"]
        ax.plot(range(len(rounds)), ys, marker="o", color=c, ls="--" if hold else "-",
                label=clip + (" (holdout)" if hold else ""))
        ax2.plot(range(len(rounds)), [r["clips"][clip]["emulator_psnr_on_G"] for r in rounds], marker="o",
                 color=c, ls="--" if hold else "-")
    ax.axhline(0, color="#888", lw=1)
    ax.set_title("Learned G vs. same-tier naive upload (composite gain)")
    ax.set_xlabel("round")
    ax.legend(frameon=False, fontsize=8)
    ax2.set_title("Emulator fidelity on G's outputs (PSNR vs real playback)")
    ax2.set_xlabel("round")
    ax2.set_ylabel("dB")
    for a in (ax, ax2):
        a.grid(alpha=0.25)
    fig.tight_layout()
    return _png(fig)


def build_surrogate_report(run_dir: Path, out: Path, title: str = "Surrogate (emulator) pipeline",
                           png_dir: Path | None = None) -> Path:
    hist = json.loads((run_dir / "history.json").read_text())
    parts = [f"<h1>{html.escape(title)}</h1>",
             f"<p class=muted>run {html.escape(str(run_dir))} · upload upscale x{hist['upscale']:g}</p>",
             _img_tag(_save(surrogate_plot(hist), png_dir, "surrogate.png"), "surrogate rounds")]
    last = hist["rounds"][-1]
    parts.append("<h2>Final round, real black-box results</h2><table><tr><th>clip</th><th>naive</th>"
                 "<th>upscale only</th><th>learned G</th><th>gain vs upscale only</th><th>|G(X)-X|</th></tr>")
    for clip, r in last["clips"].items():
        b = hist["baselines"][clip]
        n, u, g = b["naive"]["composite"], b["upscale_only"]["composite"], r["metrics"]["composite"]
        tag = " <span class=muted>(holdout)</span>" if r["holdout"] else ""
        parts.append(f"<tr><th>{html.escape(clip)}{tag}</th><td>{n:.2f}</td><td>{u:.2f}</td><td>{g:.2f}</td>"
                     f"<td>{g - u:+.2f}</td><td>{r['mean_abs_residual']:.2f}</td></tr>")
    parts.append("</table>")
    for clip, r in last["clips"].items():
        b = hist["baselines"][clip]
        parts.append(f"<h2>{html.escape(clip)}</h2>")
        parts.append(_metrics_table({"naive": b["naive"], "upscale only": b["upscale_only"],
                                     "learned G": r["metrics"]}))
        try:
            src = Path(hist.get("sources", {}).get(clip) or content.CLIPS_DIR / f"{clip}.mkv")
            fig = crops_figure(src, {
                "upscale only": Path(b["upscale_only"]["workdir"]) / "playback.mp4",
                "learned G": Path(r["workdir"]) / "playback.mp4",
            })
            parts.append(_img_tag(_save(fig, png_dir, f"surrogate_crops_{clip}.png"), clip))
        except Exception as e:
            parts.append(f"<p class=muted>(no frames: {html.escape(str(e))})</p>")
    return _write(out, title, parts)
