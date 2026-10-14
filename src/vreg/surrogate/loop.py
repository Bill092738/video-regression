"""Surrogate / differentiable-emulator loop (section 3, "代理模型方法", of idea.md).

Round r:
  1. collect pairs (uploaded frame V, playback frame Y) by sending videos through the
     *real* black box.  Round 0 uses random parametric pre-compensations; later rounds
     add the current generator's outputs (domain adaptation: the emulator learns the
     region of input space the generator actually visits);
  2. train / fine-tune the emulator E(V) ~= Y;
  3. train the generator G with E frozen:  min_G  P(E(G(X)), X) + w * L1(E(G(X)), X), where the
     perceptual loss P is LPIPS, DISTS or their sum (``g_loss``);
  4. validate G on the real black box (training clips + held-out clips) and measure the
     sim-to-real gap of E on G's outputs.

Round-0 pairs are cached per (platform, upload tier, clip, variant): the random variant's
parameters depend only on the clip name and the variant index, not on the run seed, so seeds and
specialists trained on overlapping clips reuse the same pairs instead of sending them through the
platform again (``pairs_cache``). The cache keeps the decoded frame pairs as ``.npy`` files, which
are memory-mapped, so a large training set does not have to fit in RAM. Intermediate videos are
deleted once read; only each validation playback is kept, for ``vreg rescore``.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .. import ff, metrics, precomp
from ..evaluate import evaluate
from ..optimize import cache_root
from ..platform import Platform
from .models import apply_full_frames, emulator, generator

FRAME_STEP = 4  # frames kept per clip for training pairs
PATCH = 256
LOSSLESS = ["-c:v", "libx264", "-preset", "veryfast", "-qp", "0", "-pix_fmt", "yuv444p"]
G_LOSSES = ("lpips", "dists", "lpips+dists")


@dataclass
class PairSet:
    inputs: list[np.ndarray] = field(default_factory=list)   # each (N,H,W,3) uint8
    outputs: list[np.ndarray] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def add(self, v: np.ndarray, y: np.ndarray, tag: str):
        n = min(len(v), len(y))
        self.inputs.append(v[:n])
        self.outputs.append(y[:n])
        self.tags.append(tag)

    def sample(self, batch: int, rng: random.Random, device: str, recent_bias: int = 0):
        """Aligned random crops (offsets on a 16 px grid to respect codec block alignment)."""
        xs, ys = [], []
        for _ in range(batch):
            if recent_bias and rng.random() < 0.5:
                k = rng.randrange(max(0, len(self.inputs) - recent_bias), len(self.inputs))
            else:
                k = rng.randrange(len(self.inputs))
            v, y = self.inputs[k], self.outputs[k]
            f = rng.randrange(len(v))
            h, w = v.shape[1:3]
            oy = rng.randrange(0, (h - PATCH) // 16 + 1) * 16
            ox = rng.randrange(0, (w - PATCH) // 16 + 1) * 16
            a, b = v[f, oy:oy + PATCH, ox:ox + PATCH], y[f, oy:oy + PATCH, ox:ox + PATCH]
            if rng.random() < 0.5:
                a, b = a[:, ::-1], b[:, ::-1]
            xs.append(np.ascontiguousarray(a))
            ys.append(np.ascontiguousarray(b))
        to_t = lambda arr: torch.from_numpy(np.stack(arr)).to(device).permute(0, 3, 1, 2).float() / 255  # noqa: E731
        return to_t(xs), to_t(ys)


def channel(front: float | dict) -> dict:
    """The fixed front end in front of the black box: an upscale factor (P1, E7) or a whole theta (E9)."""
    return precomp.normalize(front if isinstance(front, dict) else {"upscale": front})


def send_through_platform(video: Path, front: float | dict, platform: Platform, workdir: Path) -> Path:
    """Same upload path as the parametric pipeline: front end (lanczos upscale, or a theta) -> upload -> platform."""
    upload = precomp.apply(video, channel(front), workdir / "upload.mp4")
    return platform.process(upload, workdir)


def variant_params(clip: str, i: int) -> dict:
    """Random pre-compensation variant ``i`` of ``clip`` (variant 0 = identity); independent of the run seed."""
    if i == 0:
        params = dict(precomp.IDENTITY)
    else:
        rng = random.Random(f"{clip}:{i}")
        params = {p.name: rng.uniform(p.low, p.high) for p in precomp.SPACE}
    params["upscale"] = 1.0  # spatial pre-comp at source res; upscale is part of the channel
    return precomp.normalize(params)


def pairs_dir(cache: Path, platform: Platform, front: float | dict) -> Path:
    name = re.sub(r'[^A-Za-z0-9_.-]+', '_', platform.name)
    if isinstance(front, dict):  # a theta front end: keyed by its parameters
        from ..evaluate import params_key
        return cache / f"{name}_theta_{params_key(front)}"
    return cache / f"{name}_x{front:g}"


def collect_random_pairs(clips: dict[str, Path], platform: Platform, front: float | dict, n_variants: int,
                         cache: Path, log) -> PairSet:
    pairs = PairSet()
    root = pairs_dir(cache, platform, front)
    for clip, src in clips.items():
        for i in range(n_variants):
            params = variant_params(clip, i)
            d = root / f"{clip}_{i}"
            vf, yf, pf = d / "v.npy", d / "y.npy", d / "params.json"
            if not (vf.exists() and yf.exists() and pf.exists() and json.loads(pf.read_text()) == params):
                d.mkdir(parents=True, exist_ok=True)
                v_path = precomp.apply(src, params, d / "v.mkv")
                y_path = send_through_platform(v_path, front, platform, d)
                v = ff.read_frames(v_path, step=FRAME_STEP)
                y = ff.read_frames(y_path, size=(v.shape[2], v.shape[1]), step=FRAME_STEP)
                n = min(len(v), len(y))
                for arr, f in ((v[:n], vf), (y[:n], yf)):
                    tmp = f.with_suffix(".tmp.npy")
                    np.save(tmp, arr)
                    tmp.rename(f)
                pf.write_text(json.dumps(params, indent=2))
                for f in (v_path, d / "upload.mp4", y_path):
                    f.unlink(missing_ok=True)
                log(f"  collected {clip} variant {i}")
            pairs.add(np.load(vf, mmap_mode="r"), np.load(yf, mmap_mode="r"), f"random:{clip}:{i}")
    return pairs


def train_emulator(E, pairs: PairSet, steps: int, device: str, rng: random.Random, lr: float, log,
                   recent_bias: int = 0) -> list[float]:
    opt = torch.optim.AdamW(E.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    E.train()
    losses = []
    for step in range(steps):
        v, y = pairs.sample(12, rng, device, recent_bias=recent_bias)
        loss = F.l1_loss(E(v), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
        if (step + 1) % 500 == 0:
            log(f"  E step {step + 1}/{steps} L1={np.mean(losses[-500:]):.4f}")
    return losses


def perceptual_loss(g_loss: str):
    """The perceptual part of G's loss, as a function of (y_hat, x) in [0, 1]."""
    if g_loss not in G_LOSSES:
        raise ValueError(f"unknown g_loss {g_loss!r}; choose from {G_LOSSES}")
    nets = {}
    if "lpips" in g_loss:
        nets["lpips"] = metrics._lpips_model()[0]
    if "dists" in g_loss:
        nets["dists"] = metrics._dists_model()[0]
    for net in nets.values():
        for p in net.parameters():
            p.requires_grad_(False)

    def loss(y_hat, x):
        total = 0
        if "lpips" in nets:
            total = total + nets["lpips"](y_hat * 2 - 1, x * 2 - 1).mean()
        if "dists" in nets:
            total = total + nets["dists"](y_hat, x).mean()
        return total

    return loss


def train_generator(G, E, sources: list[np.ndarray], steps: int, device: str, rng: random.Random,
                    perceptual, lr: float, l1_weight: float, log, trust_weight: float = 0.0) -> list[float]:
    for p in E.parameters():
        p.requires_grad_(False)
    E.eval()
    opt = torch.optim.Adam(G.parameters(), lr=lr)
    G.train()
    src_pairs = PairSet()
    for s in sources:
        src_pairs.add(s, s, "src")
    losses = []
    for step in range(steps):
        x, _ = src_pairs.sample(8, rng, device)
        gx = G(x).clamp(0, 1)
        y_hat = E(gx)
        loss = perceptual(y_hat, x) + l1_weight * F.l1_loss(y_hat, x)
        if trust_weight:
            # trust region: E is only accurate near data it has seen, so keep G(X) close to X
            loss = loss + trust_weight * F.l1_loss(gx, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if (step + 1) % 500 == 0:
            log(f"  G step {step + 1}/{steps} loss={np.mean(losses[-500:]):.4f}")
    for p in E.parameters():
        p.requires_grad_(True)
    return losses


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return float(10 * np.log10(255**2 / max(mse, 1e-6)))


def validate_clip(G, E, clip: str, src: Path, d: Path, platform: Platform, front: float | dict, device: str,
                  fps: float) -> tuple[dict, np.ndarray, np.ndarray]:
    """G's real playback on one clip -> (record, G's frames at FRAME_STEP, real playback frames)."""
    d.mkdir(parents=True, exist_ok=True)
    x = ff.read_frames(src)
    gx = apply_full_frames(G, x, device)
    gx_path = ff.write_frames(gx, d / "g.mkv", fps, LOSSLESS)
    playback = send_through_platform(gx_path, front, platform, d)
    for f in (gx_path, d / "upload.mp4"):  # never read again; the playback is kept for rescore
        f.unlink(missing_ok=True)
    m = metrics.compute(playback, src)
    # sim-to-real gap: how well E predicted the real playback of G's output
    y_real = ff.read_frames(playback, size=(gx.shape[2], gx.shape[1]), step=FRAME_STEP)
    y_sim = apply_full_frames(E, gx[::FRAME_STEP], device)
    gap = _psnr(y_sim[:len(y_real)], y_real)
    delta = float(np.abs(gx.astype(np.int16) - x.astype(np.int16)).mean())
    rec = {"metrics": m, "emulator_psnr_on_G": gap, "mean_abs_residual": delta, "workdir": str(d)}
    return rec, gx[::FRAME_STEP], y_real


def run_loop(
    out_dir: Path,
    clips: dict[str, Path],
    platform: Platform,
    rounds: int = 3,
    upscale: float = 2.0,
    n_variants: int = 6,
    emulator_steps: int = 3000,
    generator_steps: int = 2000,
    finetune_fraction: float = 0.5,
    l1_weight: float = 1.0,
    trust_weight: float = 0.0,
    max_residual: float = 0.15,
    holdout: dict[str, Path] | None = None,
    seed: int = 0,
    g_loss: str = "lpips",
    pairs_cache: Path | None = None,
    log=lambda s: print(s, flush=True),
) -> list[dict]:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(seed)
    torch.manual_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    holdout = {c: p for c, p in (holdout or {}).items() if c not in clips}
    all_clips = {**clips, **holdout}
    perceptual = perceptual_loss(g_loss)
    pairs_cache = Path(pairs_cache) if pairs_cache else out_dir / "pairs"

    # Baselines on the real black box, via the (cached) parametric evaluator
    work = cache_root(platform)
    baselines = {}
    for clip, src in all_clips.items():
        naive = evaluate(clip, src, {}, platform, work).metrics
        same_tier = evaluate(clip, src, {"upscale": upscale}, platform, work).metrics
        baselines[clip] = {"naive": naive, "upscale_only": same_tier}
        log(f"baseline {clip}: naive={naive['composite']:.2f} upscale_only={same_tier['composite']:.2f}")

    # only the frames G trains on are kept in memory; validation decodes one clip at a time
    train_sources = [ff.read_frames(p, step=FRAME_STEP) for p in clips.values()]
    fps = ff.probe(next(iter(clips.values()))).fps

    log(f"round 0: collecting random pre-compensation pairs from the black box (cache {pairs_cache})")
    pairs = collect_random_pairs(clips, platform, upscale, n_variants, pairs_cache, log)

    E, G = emulator().to(device), generator(max_residual=max_residual).to(device)
    settings = {"trust_weight": trust_weight, "max_residual": max_residual, "emulator_steps": emulator_steps,
                "generator_steps": generator_steps, "g_loss": g_loss, "n_variants": n_variants, "seed": seed,
                "train_clips": list(clips), "pairs_cache": str(pairs_cache)}
    history = []
    for r in range(rounds):
        t0 = time.perf_counter()
        e_steps = emulator_steps if r == 0 else int(emulator_steps * finetune_fraction)
        g_steps = generator_steps if r == 0 else int(generator_steps * finetune_fraction)
        log(f"round {r}: train emulator ({e_steps} steps, {len(pairs.inputs)} videos of pairs)")
        e_loss = train_emulator(E, pairs, e_steps, device, rng, lr=(2e-4 if r == 0 else 1e-4), log=log,
                                recent_bias=len(clips) if r else 0)
        log(f"round {r}: train generator ({g_loss}) through frozen emulator ({g_steps} steps)")
        g_hist = train_generator(G, E, train_sources, g_steps, device, rng, perceptual,
                                 lr=(1e-4 if r == 0 else 5e-5), l1_weight=l1_weight, log=log,
                                 trust_weight=trust_weight)

        rec = {"round": r, "emulator_l1": float(np.mean(e_loss[-200:])), "generator_loss": float(np.mean(g_hist[-200:])),
               "clips": {}}
        for clip, src in all_clips.items():
            c_rec, gx_s, y_real = validate_clip(G, E, clip, src, out_dir / f"round{r}" / clip, platform, upscale,
                                                device, fps)
            c_rec["holdout"] = clip in holdout
            rec["clips"][clip] = c_rec
            b, m = baselines[clip], c_rec["metrics"]
            log(f"  [{'holdout' if clip in holdout else 'train'}] {clip}: G={m['composite']:.2f} "
                f"(naive {b['naive']['composite']:.2f}, upscale-only {b['upscale_only']['composite']:.2f}) "
                f"emulator PSNR on G(X)={c_rec['emulator_psnr_on_G']:.1f} dB  |G(X)-X|={c_rec['mean_abs_residual']:.2f}")
            if clip in clips and r < rounds - 1:  # domain adaptation: real outputs of G become new emulator data
                pairs.add(gx_s, y_real, f"G{r}:{clip}")
        rec["seconds"] = time.perf_counter() - t0
        history.append(rec)
        torch.save({"E": E.state_dict(), "G": G.state_dict()}, out_dir / f"round{r}.pt")
        (out_dir / "history.json").write_text(json.dumps({"baselines": baselines, "rounds": history, "upscale": upscale,
                                                          "platform": platform.name, "settings": settings,
                                                          "sources": {c: str(p) for c, p in all_clips.items()}},
                                                         indent=2))
    return history


def anchor_offsets(G, E, clips: dict[str, Path], platform: Platform, front: float | dict, device: str, fps: float,
                   workdir: Path) -> list[np.ndarray]:
    """One black-box call per clip: delta = y_real(G(X)) - E(G(X)) at the training frames, as int16
    (units of 1/255). Intermediate videos are deleted once read."""
    offsets = []
    for clip, src in clips.items():
        d = workdir / clip
        d.mkdir(parents=True, exist_ok=True)
        gx = apply_full_frames(G, ff.read_frames(src), device)
        gx_path = ff.write_frames(gx, d / "g.mkv", fps, LOSSLESS)
        playback = send_through_platform(gx_path, front, platform, d)
        y_real = ff.read_frames(playback, size=(gx.shape[2], gx.shape[1]), step=FRAME_STEP)
        for f in (gx_path, d / "upload.mp4", playback):
            f.unlink(missing_ok=True)
        y_sim = apply_full_frames(E, gx[::FRAME_STEP], device)
        n = min(len(y_real), len(y_sim))
        offsets.append(y_real[:n].astype(np.int16) - y_sim[:n].astype(np.int16))
    return offsets


def train_generator_anchored(G, E, sources: list[np.ndarray], steps: int, anchors: int, anchor_fn, device: str,
                             rng: random.Random, perceptual, lr: float, l1_weight: float, log,
                             trust_weight: float = 0.0) -> tuple[list[float], list[dict]]:
    """``train_generator`` with the real-anchored estimator (docs/E7-plan.md).

    The steps are split into ``anchors`` phases. At the start of each, ``anchor_fn(G)`` returns
    delta = y_real - E(G(X)) per source (the black box's output at the current G), and the phase
    trains on  y_hat = E(G(X)) + stopgrad(delta): the real output at the anchor, plus the
    emulator's predicted change, with the emulator's gradient. One optimiser runs through all
    phases, as in ``train_generator``.
    """
    for p in E.parameters():
        p.requires_grad_(False)
    E.eval()
    opt = torch.optim.Adam(G.parameters(), lr=lr)
    bounds = [round(i * steps / anchors) for i in range(anchors + 1)]
    losses, anchor_log = [], []
    for k in range(anchors):
        offsets = anchor_fn(G)
        n = [min(len(s), len(o)) for s, o in zip(sources, offsets)]
        anchor_log.append({"step": bounds[k], "mean_abs_offset": float(np.mean([np.abs(o).mean() for o in offsets]))})
        log(f"  anchor {k + 1}/{anchors} at step {bounds[k]}: mean |y_real - E(G(X))| = "
            f"{anchor_log[-1]['mean_abs_offset']:.2f} levels")
        pairs = PairSet()
        for s, o, m in zip(sources, offsets, n):
            pairs.add(s[:m], o[:m], "anchor")
        G.train()
        for step in range(bounds[k], bounds[k + 1]):
            x, delta = pairs.sample(8, rng, device)  # delta: int16 offsets / 255, aligned and flipped with x
            gx = G(x).clamp(0, 1)
            y_hat = (E(gx) + delta).clamp(0, 1)
            loss = perceptual(y_hat, x) + l1_weight * F.l1_loss(y_hat, x)
            if trust_weight:
                loss = loss + trust_weight * F.l1_loss(gx, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if (step + 1) % 250 == 0:
                log(f"  G step {step + 1}/{steps} loss={np.mean(losses[-250:]):.4f}")
    for p in E.parameters():
        p.requires_grad_(True)
    return losses, anchor_log


def run_anchored(
    out_dir: Path,
    clips: dict[str, Path],
    platform: Platform,
    emulator_from: Path | None = None,
    anchors: int = 6,
    upscale: float = 2.0,
    generator_steps: int = 1500,
    l1_weight: float = 1.0,
    trust_weight: float = 2.0,
    max_residual: float = 0.06,
    holdout: dict[str, Path] | None = None,
    seed: int = 0,
    g_loss: str = "lpips",
    front: dict | None = None,
    emulator_steps: int = 3000,
    n_variants: int = 6,
    pairs_cache: Path | None = None,
    log=lambda s: print(s, flush=True),
) -> dict:
    """E7 / E9: train G with the real-anchored estimator through a frozen emulator.

    The emulator is taken from another run (``emulator_from``, E7) or trained here from round-0
    pairs sent through the channel (E9). ``front`` puts a whole theta in front of the black box
    (E9: G on top of theta; at G = identity the output is theta's); by default the front end is
    the ``upscale`` alone. Writes the same ``history.json`` / ``round0.pt`` layout as
    ``run_loop`` (one round), plus the estimator settings and the black-box calls, so
    ``vreg rescore`` works unchanged.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(seed)
    torch.manual_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    holdout = {c: p for c, p in (holdout or {}).items() if c not in clips}
    all_clips = {**clips, **holdout}
    perceptual = perceptual_loss(g_loss)
    front = channel(front) if front else channel(upscale)
    upscale = front["upscale"]
    E = emulator().to(device)
    pair_calls = 0
    if emulator_from:
        E.load_state_dict(torch.load(emulator_from, map_location=device)["E"])
    else:
        pairs_cache = Path(pairs_cache) if pairs_cache else out_dir / "pairs"
        log(f"collecting round-0 pairs through the channel {front} (cache {pairs_cache})")
        pairs = collect_random_pairs(clips, platform, front, n_variants, pairs_cache, log)
        pair_calls = len(clips) * n_variants
        log(f"train emulator ({emulator_steps} steps, {len(pairs.inputs)} videos of pairs)")
        train_emulator(E, pairs, emulator_steps, device, rng, lr=2e-4, log=log)
        del pairs
    G = generator(max_residual=max_residual).to(device)

    work = cache_root(platform)
    baselines = {}
    for clip, src in all_clips.items():
        baselines[clip] = {"naive": evaluate(clip, src, {}, platform, work).metrics,
                           "upscale_only": evaluate(clip, src, {"upscale": upscale}, platform, work).metrics}
    train_sources = [ff.read_frames(p, step=FRAME_STEP) for p in clips.values()]
    fps = ff.probe(next(iter(clips.values()))).fps

    t0 = time.perf_counter()
    log(f"train generator ({g_loss}, {anchors} real anchors, emulator from {emulator_from})")
    g_hist, anchor_log = train_generator_anchored(
        G, E, train_sources, generator_steps, anchors,
        lambda g: anchor_offsets(g, E, clips, platform, front, device, fps, out_dir / "anchor"),
        device, rng, perceptual, lr=1e-4, l1_weight=l1_weight, log=log, trust_weight=trust_weight)
    rec = {"round": 0, "generator_loss": float(np.mean(g_hist[-200:])), "anchors": anchor_log, "clips": {}}
    for clip, src in all_clips.items():
        c_rec, _, _ = validate_clip(G, E, clip, src, out_dir / "round0" / clip, platform, front, device, fps)
        c_rec["holdout"] = clip in holdout
        rec["clips"][clip] = c_rec
        log(f"  [{'holdout' if clip in holdout else 'train'}] {clip}: G={c_rec['metrics']['composite']:.2f} "
            f"(upscale-only {baselines[clip]['upscale_only']['composite']:.2f}) "
            f"emulator PSNR on G(X)={c_rec['emulator_psnr_on_G']:.1f} dB  |G(X)-X|={c_rec['mean_abs_residual']:.2f}")
    rec["seconds"] = time.perf_counter() - t0
    torch.save({"E": E.state_dict(), "G": G.state_dict()}, out_dir / "round0.pt")
    settings = {"estimator": "anchored", "anchors": anchors, "emulator_from": str(emulator_from or "trained here"),
                "front": front,
                "trust_weight": trust_weight, "max_residual": max_residual, "generator_steps": generator_steps,
                "g_loss": g_loss, "seed": seed, "train_clips": list(clips),
                "blackbox_calls": {"anchors": anchors * len(clips), "validation": len(all_clips),
                                   "emulator_pairs": pair_calls if not emulator_from else "see the emulator_from run"}}
    hist = {"baselines": baselines, "rounds": [rec], "upscale": upscale, "platform": platform.name,
            "settings": settings, "sources": {c: str(p) for c, p in all_clips.items()}}
    (out_dir / "history.json").write_text(json.dumps(hist, indent=2))
    return hist


def evaluate_trained(run_dir: Path, clips: dict[str, Path], rnd: int = 0, tag: str = "eval",
                     log=lambda s: print(s, flush=True)) -> dict:
    """Validate a trained run's round-``rnd`` G on further clips, without retraining (P1 E23).

    The results are added to that round's clips in ``history.json`` (``holdout`` true, plus the
    ``tag``), so ``vreg rescore`` picks them up like any other validation clip.
    """
    from ..platform import make_platform

    hist_file = run_dir / "history.json"
    hist = json.loads(hist_file.read_text())
    platform = make_platform(hist["platform"])
    upscale = hist["upscale"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(run_dir / f"round{rnd}.pt", map_location=device)
    E = emulator().to(device)
    G = generator(max_residual=hist["settings"]["max_residual"]).to(device)
    E.load_state_dict(state["E"])
    G.load_state_dict(state["G"])
    rec = next(r for r in hist["rounds"] if r["round"] == rnd)
    work = cache_root(platform)
    for clip, src in clips.items():
        if clip in rec["clips"]:
            log(f"  {clip}: already evaluated")
            continue
        naive = evaluate(clip, src, {}, platform, work).metrics
        same_tier = evaluate(clip, src, {"upscale": upscale}, platform, work).metrics
        hist["baselines"][clip] = {"naive": naive, "upscale_only": same_tier}
        front = hist["settings"].get("front") or upscale  # E9 runs have a theta in front of the black box
        c_rec, _, _ = validate_clip(G, E, clip, src, run_dir / f"round{rnd}" / clip, platform, front, device,
                                    ff.probe(src).fps)
        rec["clips"][clip] = c_rec | {"holdout": True, "tag": tag}
        hist["sources"][clip] = str(src)
        log(f"  [{tag}] {clip}: G={c_rec['metrics']['composite']:.2f} "
            f"(upscale-only {same_tier['composite']:.2f})")
        hist_file.write_text(json.dumps(hist, indent=2))
    return hist
