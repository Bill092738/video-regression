# video-regression: closed-loop black-box pre-compensation for platform video compression

Video platforms re-encode every upload with a policy you cannot see or differentiate through:
the codec, bit budget and ingest filtering depend on the upload's resolution, content and more.
"Upload the best possible master" is therefore **not** the same as "deliver the best possible
playback". This repo is a working demo of the idea in [`idea.md`](idea.md): treat the platform as
a black-box degradation channel `f_platform(·)` and optimise a pre-compensation `G(X; θ)` applied
*before* upload so that the playback `Y = f_platform(G(X; θ))` is as close as possible to the
original `X`.

```
 source X ──► G(X; θ) ──► upload ──► [ platform black box ] ──► playback Y ──► metric(Y, X)
                ▲                                                                  │
                └──────────── θ update (BO / TPE / CMA-ES, or learned G) ◄────────┘
```

Two optimisation routes are implemented, matching section 3 of the idea:

| Route | What is optimised | How | Code |
|---|---|---|---|
| **Black-box search** | 7 interpretable filter parameters (denoise, contrast, saturation, upload upscale / tier, unsharp amount + radius, temporal grain) | Optuna: GP Bayesian optimisation, multivariate TPE, CMA-ES, random baseline | `vreg optimize` |
| **Surrogate / emulator** | A per-pixel residual CNN `G` | Train a differentiable emulator `E ≈ f_platform` on real (upload, playback) pairs, back-prop through it, re-validate on the real black box, add the new pairs, repeat (domain adaptation) | `vreg surrogate` |

## Quick start

```bash
scripts/setup_env.sh          # uv + deps (CUDA torch) + static ffmpeg with libvmaf; no sudo needed
uv run vreg doctor            # check ffmpeg / libvmaf / encoders / GPU
uv run vreg clips all         # 3 synthetic clips + 4 Xiph clips (streams only the first 150 frames)

# black-box search: GP-BO on two clips, 40 trials
uv run vreg optimize --platform sim:ugc_strict --clips old_town_cross,park_joy --sampler gp --trials 40 --name strict_gp
uv run vreg transfer runs/strict_gp --clips all       # do the found params generalise?
uv run vreg report runs/strict_gp --out runs/report.html

# learned pre-compensation through a differentiable emulator
uv run vreg surrogate --clips park_joy,old_town_cross,ducks_take_off --holdout in_to_tree,sky_gradient --rounds 4
uv run vreg report runs/surrogate --surrogate --out runs/surrogate.html

scripts/run_experiments.sh    # the full sampler comparison used for the results below
uv run pytest -q              # fast unit tests on tiny clips (CI runs these without torch)
```

## Components

* **`vreg.platform`**: the black box. `SimulatedPlatform` reproduces behaviour real UGC platforms
  are known to have: the encode ladder (codec, CRF, maxrate cap) is selected by *upload*
  resolution, with ingest spatio-temporal denoising and capped-CRF rate control, and the viewer is
  served a fixed 720p rendition. Two hidden profiles are included:
  * `ugc_strict`: strong ingest denoising and tight caps; ≥1080p uploads get more bits, and ≥1440p
    uploads get HEVC.
  * `ugc_av1`: ≥1440p uploads get AV1.

  The optimiser only ever calls `process(upload) -> playback`.
  `CommandPlatform` is the adapter for a **real** platform: it takes your own upload command and
  download command (e.g. an upload CLI and `yt-dlp -f 'bv*[height=720]'`) and polls until the
  transcode is available. Use it only with your own account and content, within the platform's
  terms of service.
* **`vreg.precomp`**: `G(X; θ)` as an ffmpeg filter chain:
  `hqdn3d → eq → lanczos upscale → unsharp → temporal luma grain`. All parameters are continuous
  (discrete ones are snapped), so every sampler can use the same space.
* **`vreg.metrics`**: the score compares the playback, rescaled to the source resolution, with the
  source X. It uses:
  * **VMAF NEG** (the "no enhancement gain" model), because plain VMAF rewards over-sharpening,
    which is exactly the loophole an optimiser would find;
  * **LPIPS** (on the GPU);
  * **CAMBI** (Netflix's banding detector);
  * PSNR and SSIM for reference.

  The default objective is `composite = VMAF_NEG − 40·LPIPS − 2·CAMBI`.
* **`vreg.optimize`**: Optuna studies backed by SQLite, so they are resumable. Trial 0 is always
  the naive upload. Every (clip, θ) evaluation is cached on disk, so studies, transfers and
  reports reuse each other's work.
* **`vreg.surrogate`**: residual U-Net emulator `E` and generator `G`. The loop is: collect
  real pairs → fit `E` → train `G` with `LPIPS(E(G(X)), X) + L1` → validate on the real black box
  → add `G`'s real pairs → repeat. It logs the emulator's sim-to-real PSNR on `G`'s outputs, which
  is the number that tells you whether `G` is exploiting emulator errors.
* **`vreg.report`**: a self-contained HTML report with:
  * convergence curves per sampler;
  * a decomposition into naive, tier-only (upscale only) and full pre-compensation, so the gain
    from gaming the ladder is separated from the gain from filtering;
  * zoomed frame crops;
  * per-parameter sensitivity plots;
  * the transfer table.

## Results

_Filled in from `scripts/run_experiments.sh` and the surrogate run; see below._

## Repository layout

```
src/vreg/       ff.py content.py platform.py precomp.py metrics.py evaluate.py optimize.py report.py cli.py
src/vreg/surrogate/  models.py loop.py
scripts/        setup_env.sh run_experiments.sh
tests/          tiny-clip unit tests (run in GitHub Actions)
docs/img/       figures used in this README
data/, runs/    generated clips and experiment outputs (git-ignored)
```
