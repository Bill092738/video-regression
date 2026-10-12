"""Command line entry point: ``vreg <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RUNS = Path("runs")


def cmd_doctor(_args) -> int:
    from . import ff, metrics

    ok = True
    print(f"ffmpeg         : {ff.ffmpeg_bin()}")
    print(f"ffprobe        : {ff.ffprobe_bin()}")
    vm = ff.vmaf_ffmpeg_bin()
    print(f"libvmaf ffmpeg : {vm or 'MISSING (falls back to PSNR/SSIM only; run scripts/setup_env.sh)'}")
    if metrics.lpips_available():
        import torch

        cuda = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
        print(f"torch / LPIPS  : {torch.__version__} on {cuda}")
    else:
        print("torch / LPIPS  : not installed (uv sync --extra torch)")
    for enc in ("libx264", "libx265", "libsvtav1"):
        import subprocess

        out = subprocess.run([ff.ffmpeg_bin(), "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
        has = f" {enc} " in out
        ok &= has
        print(f"encoder {enc:9s}: {'ok' if has else 'MISSING'}")
    return 0 if ok else 1


def _clips(spec: str) -> dict[str, Path]:
    from . import content

    names = content.all_clip_names() if spec == "all" else [c for c in spec.split(",") if c]
    return content.ensure_clips(names, content.CLIPS_DIR)


def cmd_clips(args) -> int:
    from . import content, ff

    for name, path in _clips(args.names).items():
        i = ff.probe(path)
        desc = (content.SYNTHETIC_CLIPS.get(name) or content.XIPH_CLIPS.get(name))[1]
        print(f"{name:16s} {i.width}x{i.height}@{i.fps:g} {i.frames:3d}f  {path}  -- {desc}")
    return 0


def cmd_eval(args) -> int:
    from .evaluate import evaluate
    from .optimize import cache_root
    from .platform import make_platform

    plat = make_platform(args.platform)
    params = json.loads(args.params)
    for clip, src in _clips(args.clips).items():
        r = evaluate(clip, src, params, plat, cache_root(plat))
        print(json.dumps(r.to_json(), indent=2))
    return 0


def cmd_optimize(args) -> int:
    from .optimize import run_study
    from .platform import make_platform

    plat = make_platform(args.platform)
    clips = _clips(args.clips)
    name = args.name or f"{args.platform.replace(':', '-')}_{args.sampler}_{'+'.join(clips)}"
    study = run_study(RUNS / name, clips, plat, sampler=args.sampler, n_trials=args.trials,
                      objective=args.objective, seed=args.seed)
    b = study.best_trial
    print(f"best value {b.value:.3f} (trial {b.number}); baseline {study.trials[0].value:.3f}")
    print(json.dumps(b.user_attrs.get("params_norm", b.params), indent=2))
    return 0


def cmd_transfer(args) -> int:
    from .optimize import transfer

    transfer(Path(args.run), _clips(args.clips))
    return 0


def cmd_report(args) -> int:
    from .report import build_report, build_surrogate_report

    png_dir = Path(args.png_dir) if args.png_dir else None
    if args.surrogate:
        out = build_surrogate_report(Path(args.runs[0]), Path(args.out), title=args.title, png_dir=png_dir)
    else:
        out = build_report([Path(d) for d in args.runs], Path(args.out), title=args.title, png_dir=png_dir)
    print(out)
    return 0


def cmd_surrogate(args) -> int:
    from .surrogate.loop import run_loop
    from .platform import make_platform

    plat = make_platform(args.platform)
    clips = _clips(args.clips)
    run_loop(RUNS / (args.name or "surrogate"), clips, plat, rounds=args.rounds, upscale=args.upscale,
             n_variants=args.variants, emulator_steps=args.emulator_steps, generator_steps=args.generator_steps,
             finetune_fraction=args.finetune_fraction, trust_weight=args.trust_weight, max_residual=args.max_residual,
             holdout=_clips(args.holdout) if args.holdout else None)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vreg", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="check the environment").set_defaults(fn=cmd_doctor)

    p = sub.add_parser("clips", help="generate / fetch test clips")
    p.add_argument("names", nargs="?", default="all", help="comma separated names or 'all'")
    p.set_defaults(fn=cmd_clips)

    p = sub.add_parser("eval", help="evaluate one parameter set")
    p.add_argument("--clips", default="old_town_cross")
    p.add_argument("--platform", default="sim:ugc_strict")
    p.add_argument("--params", default="{}", help='JSON, e.g. \'{"upscale": 2, "sharpen": 0.5}\'')
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("optimize", help="black-box optimisation of pre-compensation parameters")
    p.add_argument("--clips", default="old_town_cross")
    p.add_argument("--platform", default="sim:ugc_strict", help="sim:<profile> or cmd:<upload>||<download>")
    p.add_argument("--sampler", default="gp", choices=["random", "tpe", "gp", "cmaes"])
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--objective", default="composite",
                   choices=["composite", "vmaf_neg", "vmaf", "ssim", "psnr_y", "lpips", "cambi"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name")
    p.set_defaults(fn=cmd_optimize)

    p = sub.add_parser("report", help="HTML report for one or more runs")
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", default="runs/report.html")
    p.add_argument("--title", default="Pre-compensation report")
    p.add_argument("--png-dir", help="also save the figures as PNGs here (e.g. docs/img)")
    p.add_argument("--surrogate", action="store_true", help="the run is a 'vreg surrogate' run")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("transfer", help="apply a run's best params to other clips")
    p.add_argument("run")
    p.add_argument("--clips", default="all")
    p.set_defaults(fn=cmd_transfer)

    p = sub.add_parser("surrogate", help="differentiable emulator + learned pre-compensation loop")
    p.add_argument("--clips", default="park_joy,old_town_cross,ducks_take_off")
    p.add_argument("--holdout", default="in_to_tree")
    p.add_argument("--platform", default="sim:ugc_strict")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--upscale", type=float, default=2.0, help="fixed upload upscale (tier) for the learned G")
    p.add_argument("--variants", type=int, default=6, help="random pre-comp variants per clip in round 0")
    p.add_argument("--trust-weight", type=float, default=2.0, help="L1 penalty on |G(X)-X| (trust region)")
    p.add_argument("--max-residual", type=float, default=0.06, help="hard cap on G's per-pixel change")
    p.add_argument("--finetune-fraction", type=float, default=0.4, help="steps in rounds >0 relative to round 0")
    p.add_argument("--emulator-steps", type=int, default=3000)
    p.add_argument("--generator-steps", type=int, default=1500)
    p.add_argument("--name")
    p.set_defaults(fn=cmd_surrogate)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
