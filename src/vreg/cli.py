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
    xp = ff.xpsnr_ffmpeg_bin()
    print(f"xpsnr ffmpeg   : {xp or 'MISSING (needs ffmpeg >= 7.1; the xpsnr judge is skipped)'}")
    for j in metrics.JUDGE_METRICS:
        if j != "xpsnr":
            print(f"judge {j:9s}: {'ok' if metrics.judge_available(j) else 'MISSING (skipped; uv sync --extra torch)'}")
    for enc in ("libx264", "libx265", "libsvtav1"):
        import subprocess

        out = subprocess.run([ff.ffmpeg_bin(), "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
        has = f" {enc} " in out
        ok &= has
        print(f"encoder {enc:9s}: {'ok' if has else 'MISSING'}")
    return 0 if ok else 1


def _clips(spec: str) -> dict[str, Path]:
    """Clip names, ``all`` (the P0 clips) or selectors such as ``@A:train`` (see ``vreg.classify``)."""
    from . import classify, content

    names = content.all_clip_names() if spec == "all" else classify.resolve(spec)
    return content.ensure_clips(names, content.CLIPS_DIR)


def cmd_clips(args) -> int:
    import os

    from . import content, ff

    if args.manifest:
        os.environ["VREG_MANIFEST"] = args.manifest
        spec = args.names if args.names != "all" else ",".join(content.read_manifest(Path(args.manifest)))
    else:
        spec = args.names
    failed = []
    for name in (spec.split(",") if args.manifest else [spec]):
        try:
            clips = _clips(name)
        except (ff.FFmpegError, OSError, ValueError) as e:
            print(f"!!! {name}: {e}".splitlines()[0])
            failed.append(name)
            continue
        for n, path in clips.items():
            i = ff.probe(path)
            print(f"{n:22s} {i.width}x{i.height}@{i.fps:g} {i.frames:3d}f  {path}  -- {content.describe(n)}", flush=True)
    if failed:
        print(f"failed: {','.join(failed)}")
    return 1 if failed else 0


def cmd_features(args) -> int:
    import os

    from . import content, features

    os.environ["VREG_MANIFEST"] = args.manifest
    names = list(content.read_manifest(Path(args.manifest)))
    rows = []
    for name, path in content.ensure_clips(names, content.CLIPS_DIR).items():
        rows.append(features.clip_features(name, path))
        r = rows[-1]
        print(f"{name:22s} SI={r['si']:6.1f} TI={r['ti']:6.1f} flat={r['flat_frac']:.2f} "
              f"dark={r['dark_frac']:.2f} luma={r['mean_luma']:.0f}", flush=True)
    print(features.write_csv(rows, Path(args.out)))
    return 0


def cmd_classify(args) -> int:
    from . import classify, content, features

    pre = json.loads(args.pre_registered) if args.pre_registered else None
    res = classify.classify(content.read_manifest(Path(args.manifest)), features.read_csv(Path(args.features)),
                            seed=args.seed, report_only=tuple(c for c in args.report_only.split(",") if c),
                            pre_registered=pre)
    print(classify.write(res, Path(args.out)))
    r = res["rules"]
    print(f"B if SI >= {r['si_min_for_B']:.1f} or TI >= {r['ti_min_for_B']:.1f}")
    for k, c in res["counts"].items():
        print(f"class {k}: {sum(c.values()):3d} clips  " + "  ".join(f"{s} {n}" for s, n in c.items()))
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
    from .optimize import parse_fixed, run_study
    from .platform import make_platform

    plat = make_platform(args.platform)
    clips = _clips(args.clips)
    name = args.name or f"{args.platform.replace(':', '-')}_{args.sampler}_{'+'.join(clips)}"
    study = run_study(RUNS / name, clips, plat, sampler=args.sampler, n_trials=args.trials,
                      objective=args.objective, seed=args.seed, fixed=parse_fixed(args.fix))
    b = study.best_trial
    print(f"best value {b.value:.3f} (trial {b.number}); baseline {study.trials[0].value:.3f}")
    print(json.dumps(b.user_attrs.get("params_norm", b.params), indent=2))
    return 0


def cmd_transfer(args) -> int:
    from .optimize import transfer

    transfer(Path(args.run), _clips(args.clips))
    return 0


def cmd_rescore(args) -> int:
    from .rescore import rescore

    clips = _clips(args.clips)
    for run in args.runs:
        rescore(Path(run), clips, tag=args.tag, platform_override=args.platform)
        print((Path(run) / (f"rescore_{args.tag}.md" if args.tag else "rescore.md")).read_text())
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
    from .surrogate.loop import run_anchored, run_loop
    from .platform import make_platform

    plat = make_platform(args.platform)
    clips = _clips(args.clips)
    if args.estimator == "anchored":
        front = None
        if args.front_from:
            from .optimize import best_params

            front = best_params(Path(args.front_from))
        run_anchored(RUNS / (args.name or "surrogate_anchored"), clips, plat,
                     Path(args.emulator_from) if args.emulator_from else None,
                     anchors=args.anchors, upscale=args.upscale, generator_steps=args.generator_steps,
                     trust_weight=args.trust_weight, max_residual=args.max_residual,
                     holdout=_clips(args.holdout) if args.holdout else None, seed=args.seed, g_loss=args.g_loss,
                     front=front, emulator_steps=args.emulator_steps, n_variants=args.variants,
                     pairs_cache=Path(args.pairs_cache) if args.pairs_cache else None)
        return 0
    run_loop(RUNS / (args.name or "surrogate"), clips, plat, rounds=args.rounds, upscale=args.upscale,
             n_variants=args.variants, emulator_steps=args.emulator_steps, generator_steps=args.generator_steps,
             finetune_fraction=args.finetune_fraction, trust_weight=args.trust_weight, max_residual=args.max_residual,
             holdout=_clips(args.holdout) if args.holdout else None, seed=args.seed, g_loss=args.g_loss,
             pairs_cache=Path(args.pairs_cache) if args.pairs_cache else None)
    return 0


def cmd_surrogate_eval(args) -> int:
    from .surrogate.loop import evaluate_trained

    evaluate_trained(Path(args.run), _clips(args.clips), rnd=args.round, tag=args.tag)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vreg", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="check the environment").set_defaults(fn=cmd_doctor)

    p = sub.add_parser("clips", help="generate / fetch test clips")
    p.add_argument("names", nargs="?", default="all", help="comma separated names or 'all'")
    p.add_argument("--manifest", help="fetch the clips of a manifest (docs/p1/clips.csv); 'all' = every row")
    p.set_defaults(fn=cmd_clips)

    p = sub.add_parser("features", help="SI/TI, flat and dark fractions of every manifest clip")
    p.add_argument("--manifest", default="docs/p1/clips.csv")
    p.add_argument("--out", default="docs/p1/features.csv")
    p.set_defaults(fn=cmd_features)

    p = sub.add_parser("classify", help="content classes and train/dev/confirm splits")
    p.add_argument("--manifest", default="docs/p1/clips.csv")
    p.add_argument("--features", default="docs/p1/features.csv")
    p.add_argument("--out", default="docs/p1/classes.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report-only", default="", help="classes whose clips get no split (listed, never tested)")
    p.add_argument("--pre-registered", help="JSON of decisions fixed before any training, e.g. '{\"gate\": {\"C\": \"tier\"}}'")
    p.set_defaults(fn=cmd_classify)

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
                   choices=["composite", "vmaf_neg", "vmaf", "ssim", "psnr_y", "lpips", "cambi",
                            "ssimulacra2", "dists", "xpsnr"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fix", help="freeze parameters, e.g. contrast=1,saturation=1 or upscale=1")
    p.add_argument("--name")
    p.set_defaults(fn=cmd_optimize)

    p = sub.add_parser("rescore", help="score naive / tier-only / best (or each G round) with all metrics")
    p.add_argument("runs", nargs="+")
    p.add_argument("--clips", default="all")
    p.add_argument("--tag", default="", help="write rescore_<tag>.json / .md instead of rescore.json / .md")
    p.add_argument("--platform", help="evaluate the run's configurations on this platform instead (transfer, E14)")
    p.set_defaults(fn=cmd_rescore)

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
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--g-loss", default="lpips", choices=["lpips", "dists", "lpips+dists"],
                   help="perceptual part of G's training loss (G is never judged by it)")
    p.add_argument("--pairs-cache", help="directory of round-0 pairs shared between runs (default: inside the run)")
    p.add_argument("--estimator", default="emulator", choices=["emulator", "anchored"],
                   help="anchored: real-anchored forward value through a frozen emulator (E7, docs/E7-plan.md)")
    p.add_argument("--anchors", type=int, default=6, help="anchored: black-box anchor phases (one call per clip each)")
    p.add_argument("--emulator-from", help="anchored: checkpoint whose E is reused, e.g. runs/p1_g_lpips_s0/round0.pt (default: train E here)")
    p.add_argument("--front-from", help="anchored: put this optimize run's best theta in front of the black box "
                   "(E9: G on top of theta)")
    p.add_argument("--name")
    p.set_defaults(fn=cmd_surrogate)

    p = sub.add_parser("surrogate-eval", help="evaluate a trained surrogate run's G on further clips, no retraining")
    p.add_argument("run")
    p.add_argument("--clips", required=True)
    p.add_argument("--round", type=int, default=0)
    p.add_argument("--tag", default="eval")
    p.set_defaults(fn=cmd_surrogate_eval)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
