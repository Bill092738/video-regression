"""Surrogate / differentiable-emulator loop (section 3, "代理模型方法", of idea.md).

Round r:
  1. collect pairs (uploaded frame V, playback frame Y) by sending videos through the
     *real* black box.  Round 0 uses random parametric pre-compensations; later rounds
     add the current generator's outputs (domain adaptation: the emulator learns the
     region of input space the generator actually visits);
  2. train / fine-tune the emulator E(V) ~= Y;
  3. train the generator G with E frozen:  min_G  LPIPS(E(G(X)), X) + w * L1(E(G(X)), X);
  4. validate G on the real black box (training clips + held-out clips) and measure the
     sim-to-real gap of E on G's outputs.
"""

from __future__ import annotations

import json
import random
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


def send_through_platform(video: Path, upscale: float, platform: Platform, workdir: Path) -> Path:
    """Same upload path as the parametric pipeline: (optional lanczos upscale) -> upload -> platform."""
    upload = precomp.apply(video, {"upscale": upscale}, workdir / "upload.mp4")
    return platform.process(upload, workdir)


def collect_random_pairs(clips: dict[str, Path], platform: Platform, upscale: float, n_variants: int,
                         workdir: Path, rng: random.Random, log) -> PairSet:
    pairs = PairSet()
    for clip, src in clips.items():
        for i in range(n_variants):
            if i == 0:
                params = dict(precomp.IDENTITY)
            else:
                params = {p.name: rng.uniform(p.low, p.high) for p in precomp.SPACE}
            params["upscale"] = 1.0  # spatial pre-comp at source res; upscale is part of the channel
            d = workdir / "random" / f"{clip}_{i}"
            d.mkdir(parents=True, exist_ok=True)
            v_path = precomp.apply(src, params, d / "v.mkv")
            y_path = send_through_platform(v_path, upscale, platform, d)
            v = ff.read_frames(v_path, step=FRAME_STEP)
            y = ff.read_frames(y_path, size=(v.shape[2], v.shape[1]), step=FRAME_STEP)
            pairs.add(v, y, f"random:{clip}:{i}")
            log(f"  collected {clip} variant {i}")
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


def train_generator(G, E, sources: list[np.ndarray], steps: int, device: str, rng: random.Random,
                    lpips_net, lr: float, l1_weight: float, log) -> list[float]:
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
        y_hat = E(G(x).clamp(0, 1))
        loss = lpips_net(y_hat * 2 - 1, x * 2 - 1).mean() + l1_weight * F.l1_loss(y_hat, x)
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
    holdout: dict[str, Path] | None = None,
    seed: int = 0,
    log=lambda s: print(s, flush=True),
) -> list[dict]:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(seed)
    torch.manual_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    holdout = holdout or {}
    all_clips = {**clips, **holdout}
    lpips_net, _ = metrics._lpips_model()
    for p in lpips_net.parameters():
        p.requires_grad_(False)

    # Baselines on the real black box, via the (cached) parametric evaluator
    work = cache_root(platform)
    baselines = {}
    for clip, src in all_clips.items():
        naive = evaluate(clip, src, {}, platform, work).metrics
        same_tier = evaluate(clip, src, {"upscale": upscale}, platform, work).metrics
        baselines[clip] = {"naive": naive, "upscale_only": same_tier}
        log(f"baseline {clip}: naive={naive['composite']:.2f} upscale_only={same_tier['composite']:.2f}")

    sources = {c: ff.read_frames(p) for c, p in all_clips.items()}
    fps = ff.probe(next(iter(clips.values()))).fps

    log("round 0: collecting random pre-compensation pairs from the black box")
    pairs = collect_random_pairs(clips, platform, upscale, n_variants, out_dir / "pairs", rng, log)

    E, G = emulator().to(device), generator().to(device)
    history = []
    for r in range(rounds):
        t0 = time.perf_counter()
        e_steps = emulator_steps if r == 0 else int(emulator_steps * finetune_fraction)
        g_steps = generator_steps if r == 0 else int(generator_steps * finetune_fraction)
        log(f"round {r}: train emulator ({e_steps} steps, {len(pairs.inputs)} videos of pairs)")
        e_loss = train_emulator(E, pairs, e_steps, device, rng, lr=(2e-4 if r == 0 else 1e-4), log=log,
                                recent_bias=len(clips) if r else 0)
        log(f"round {r}: train generator through frozen emulator ({g_steps} steps)")
        train_sources = [sources[c][::FRAME_STEP] for c in clips]
        g_loss = train_generator(G, E, train_sources, g_steps, device, rng, lpips_net,
                                 lr=(1e-4 if r == 0 else 5e-5), l1_weight=l1_weight, log=log)

        rec = {"round": r, "emulator_l1": float(np.mean(e_loss[-200:])), "generator_loss": float(np.mean(g_loss[-200:])),
               "clips": {}}
        for clip, src in all_clips.items():
            d = out_dir / f"round{r}" / clip
            d.mkdir(parents=True, exist_ok=True)
            gx = apply_full_frames(G, sources[clip], device)
            gx_path = ff.write_frames(gx, d / "g.mkv", fps, LOSSLESS)
            playback = send_through_platform(gx_path, upscale, platform, d)
            m = metrics.compute(playback, src)
            # sim-to-real gap: how well E predicted the real playback of G's output
            y_real = ff.read_frames(playback, size=(gx.shape[2], gx.shape[1]), step=FRAME_STEP)
            y_sim = apply_full_frames(E, gx[::FRAME_STEP], device)
            gap = _psnr(y_sim[:len(y_real)], y_real)
            delta = float(np.abs(gx.astype(np.int16) - sources[clip].astype(np.int16)).mean())
            rec["clips"][clip] = {"metrics": m, "emulator_psnr_on_G": gap, "mean_abs_residual": delta,
                                  "holdout": clip in holdout, "workdir": str(d)}
            b = baselines[clip]
            log(f"  [{'holdout' if clip in holdout else 'train'}] {clip}: G={m['composite']:.2f} "
                f"(naive {b['naive']['composite']:.2f}, upscale-only {b['upscale_only']['composite']:.2f}) "
                f"emulator PSNR on G(X)={gap:.1f} dB  |G(X)-X|={delta:.2f}")
            if clip in clips:  # domain adaptation: real outputs of G become new emulator data
                pairs.add(gx[::FRAME_STEP], y_real, f"G{r}:{clip}")
        rec["seconds"] = time.perf_counter() - t0
        history.append(rec)
        torch.save({"E": E.state_dict(), "G": G.state_dict()}, out_dir / f"round{r}.pt")
        (out_dir / "history.json").write_text(json.dumps({"baselines": baselines, "rounds": history,
                                                          "upscale": upscale}, indent=2))
    return history
