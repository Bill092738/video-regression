"""E12 (docs/E12-plan.md): a passthrough rule below a bitrate cap, with the upload bitrate as a parameter.

    uv run python scripts/e12.py search --seed 0   # GP-BO, 30 trials (resumable, SQLite)
    uv run python scripts/e12.py eval              # best thetas + fixed configs on A/B dev, with the judges
    uv run python scripts/e12.py tables            # -> runs/e12_tables.md, docs/img/e12/*.png

Black box: sim:ugc_passthrough (ugc_strict + passthrough for H.264 <= 720p and <= 2000 kbps).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402

from vreg import classify, content, metrics, precomp  # noqa: E402
from vreg.evaluate import evaluate  # noqa: E402
from vreg.optimize import cache_root  # noqa: E402
from vreg.platform import make_platform  # noqa: E402

PLATFORM = "sim:ugc_passthrough"
CAP_KBPS = 2000
RUN = Path("runs/e12")
TRIALS = 30
JUDGES = ("ssimulacra2", "dists", "xpsnr")
COLS = ("composite", *JUDGES)
FIXED = {
    "tier": ({"upscale": 2.0}, None),
    "r_mild": ({"upscale": 2.0, "sharpen": 0.5, "sharpen_radius": 2.0}, None),
    "pass": ({}, 1900),
    "pass_mild": ({"sharpen": 0.5, "sharpen_radius": 2.0}, 1900),
    "over": ({}, 2500),  # x264 undershoots on easy clips; 2100 kbps came out at 1888 (plan amendment)
    "tier_16m": ({"upscale": 2.0}, 16_000),
}
LABEL = {"tier": "tier-only ×2, CRF 10", "r_mild": "×2 + mild unsharp, CRF 10", "pass": "×1 at 1900 kbps (under the cap)",
         "pass_mild": "×1 + mild unsharp at 1900 kbps", "over": "×1 at 2500 kbps (over the cap)",
         "tier_16m": "×2 at 16 Mbps"}

_spec = importlib.util.spec_from_file_location("p1_tables", Path(__file__).with_name("p1_tables.py"))
p1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p1)
bootstrap, fmt = p1.bootstrap, p1.fmt


def round2(x: float) -> int:
    """Round to 2 significant digits, so nearby trials share the evaluation cache."""
    e = int(math.floor(math.log10(x))) - 1
    return int(round(x / 10**e) * 10**e)


def train_clips() -> list[str]:
    cls = classify.load()["clips"]
    a = [c for c, v in cls.items() if v["class"] == "A" and v["split"] == "train"][:3]
    b = [c for c, v in cls.items() if v["class"] == "B" and v["split"] == "train"][:3]
    return a + b


def theta(trial_params: dict) -> tuple[dict, int]:
    p = {"upscale": trial_params["upscale"], "sharpen": trial_params["sharpen"], "sharpen_radius": 2.0}
    return precomp.normalize(p), round2(trial_params["upload_kbps"])


def is_passthrough(params: dict, kbps: int) -> bool:
    return params["upscale"] == 1.0 and kbps <= CAP_KBPS


def search(seed: int) -> None:
    plat = make_platform(PLATFORM)
    work = cache_root(plat)
    clips = content.ensure_clips(train_clips(), content.CLIPS_DIR)
    RUN.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(study_name=f"e12_s{seed}", storage=f"sqlite:///{RUN / f'study_s{seed}.db'}",
                                sampler=optuna.samplers.GPSampler(seed=seed, n_startup_trials=8),
                                direction="maximize", load_if_exists=True)
    if not study.trials:
        study.enqueue_trial({"upscale": 2.0, "sharpen": 0.0, "upload_kbps": 80_000})
    log = RUN / f"trials_s{seed}.jsonl"

    def objective(trial: optuna.Trial) -> float:
        tp = {"upscale": trial.suggest_float("upscale", 1.0, 2.0), "sharpen": trial.suggest_float("sharpen", 0.0, 2.0),
              "upload_kbps": trial.suggest_float("upload_kbps", 300, 80_000, log=True)}
        params, kbps = theta(tp)
        vals = [metrics.objective_value(evaluate(c, s, params, plat, work, upload_kbps=kbps).metrics, "composite")
                for c, s in clips.items()]
        v = float(np.mean(vals))
        with log.open("a") as f:
            f.write(json.dumps({"trial": trial.number, "value": v, "params": params, "upload_kbps": kbps,
                                "passthrough": is_passthrough(params, kbps)}) + "\n")
        print(f"[e12 s{seed}] trial {trial.number:2d} composite={v:7.3f} upscale={params['upscale']} "
              f"sharpen={params['sharpen']:.2f} kbps={kbps} {'PASS' if is_passthrough(params, kbps) else ''}", flush=True)
        return v

    left = TRIALS - len([t for t in study.trials if t.state.is_finished()])
    if left > 0:
        study.optimize(objective, n_trials=left, catch=(Exception,))


def trials(seed: int) -> list[dict]:
    f = RUN / f"trials_s{seed}.jsonl"
    recs = {}
    for line in (f.read_text().splitlines() if f.exists() else []):
        r = json.loads(line)
        recs[r["trial"]] = r
    return [recs[k] for k in sorted(recs)]


def configs() -> dict[str, tuple[dict, int | None]]:
    out = dict(FIXED)
    for s in (0, 1):
        t = trials(s)
        if t:
            b = max(t, key=lambda r: r["value"])
            out[f"best_s{s}"] = (b["params"], b["upload_kbps"])
            LABEL[f"best_s{s}"] = (f"GP-BO best, seed {s} (×{b['params']['upscale']:g}, unsharp "
                                   f"{b['params']['sharpen']:.2f}, {b['upload_kbps']} kbps)")
    return out


def eval_dev() -> None:
    plat = make_platform(PLATFORM)
    work = cache_root(plat)
    cls = classify.load()["clips"]
    names = [c for c, v in cls.items() if v["class"] in ("A", "B") and v["split"] == "dev"]
    clips = content.ensure_clips(names, content.CLIPS_DIR)
    f = RUN / "dev.json"
    res = json.loads(f.read_text()) if f.exists() else {}
    for cid, (params, kbps) in configs().items():
        for c, s in clips.items():
            r = evaluate(c, s, params, plat, work, judges=metrics.JUDGE_METRICS, upload_kbps=kbps)
            res.setdefault(c, {})[cid] = r.metrics | {"upload_kbps": r.upload_kbps}
        f.write_text(json.dumps(res, indent=2))
        print(f"[e12 eval] {cid} done", flush=True)


def tables() -> None:
    cls = classify.load()["clips"]
    out = ["# E12 tables (generated by scripts/e12.py)", "",
           f"Black box {PLATFORM}: ugc_strict plus passthrough for H.264 ≤ 720p and ≤ {CAP_KBPS} kbps (invented rule). "
           "Positive = better; mean [95 % CI over clips]; bold = CI excludes 0.", ""]
    out += ["## Q1: discovery by GP-BO (30 trials, composite on 3 A + 3 B train clips)", "",
            "| seed | best composite | best θ | in passthrough region | first passthrough trial within 1 % of the best | "
            "passthrough share of trials |", "|---|---|---|---|---|---|"]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for s in (0, 1):
        t = trials(s)
        if not t:
            continue
        b = max(t, key=lambda r: r["value"])
        near = next((r["trial"] for r in t if r["passthrough"] and r["value"] >= b["value"] - 0.01 * abs(b["value"])), None)
        share = np.mean([r["passthrough"] for r in t])
        out.append(f"| {s} | {b['value']:.2f} | ×{b['params']['upscale']:g}, unsharp {b['params']['sharpen']:.2f}, "
                   f"{b['upload_kbps']} kbps | {'yes' if b['passthrough'] else 'no'} | {near if near is not None else '–'} | "
                   f"{100 * share:.0f} % |")
        best = np.maximum.accumulate([r["value"] for r in t])
        ax.plot([r["trial"] for r in t], best, label=f"seed {s}, best so far")
        ax.scatter([r["trial"] for r in t], [r["value"] for r in t], s=12,
                   c=["tab:green" if r["passthrough"] else "tab:gray" for r in t])
    ax.set_xlabel("trial")
    ax.set_ylabel("composite")
    ax.set_title("E12: GP-BO on ugc_passthrough (green = passthrough trial)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    Path("docs/img/e12").mkdir(parents=True, exist_ok=True)
    fig.savefig("docs/img/e12/e12_search.png", dpi=110)
    plt.close(fig)
    out.append("")

    f = RUN / "dev.json"
    if f.exists():
        res = json.loads(f.read_text())
        cfg = configs()
        out += ["## Q2: every configuration vs tier-only, dev clips", "",
                "| class | configuration | " + " | ".join(COLS) + " | median upload kbps | served |",
                "|---" * (len(COLS) + 4) + "|"]
        for k in ("A", "B"):
            clips = [c for c in res if cls[c]["class"] == k]
            for cid in cfg:
                if cid == "tier" or not all(cid in res[c] for c in clips):
                    continue
                g = {m: bootstrap([res[c][cid][m] - res[c]["tier"][m] if metrics.HIGHER_IS_BETTER[m]
                                   else res[c]["tier"][m] - res[c][cid][m] for c in clips]) for m in COLS}
                params = precomp.normalize(cfg[cid][0])
                n_pass = sum(params["upscale"] == 1.0 and res[c][cid]["upload_kbps"] <= CAP_KBPS for c in clips)
                served = f"{n_pass}/{len(clips)} served unchanged"
                kb = np.median([res[c][cid]["upload_kbps"] for c in clips])
                out.append(f"| {k} ({len(clips)}) | {LABEL[cid]} | " + " | ".join(fmt(m, *g[m]) for m in COLS)
                           + f" | {kb:.0f} | {served} |")
        out += ["", "## Q3: the cliff, ×1 at 1900 kbps (served) − ×1 at 2500 kbps (re-encoded)", "",
                "| class | " + " | ".join(COLS) + " |", "|---" * (len(COLS) + 1) + "|"]
        for k in ("A", "B"):
            clips = [c for c in res if cls[c]["class"] == k]
            g = {m: bootstrap([res[c]["pass"][m] - res[c]["over"][m] if metrics.HIGHER_IS_BETTER[m]
                               else res[c]["over"][m] - res[c]["pass"][m] for c in clips]) for m in COLS}
            out.append(f"| {k} ({len(clips)}) | " + " | ".join(fmt(m, *g[m]) for m in COLS) + " |")
        out.append("")
    Path("runs/e12_tables.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["search", "eval", "tables"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"search": lambda: search(a.seed), "eval": eval_dev, "tables": tables}[a.cmd]()
