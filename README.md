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

## Status and positioning

* **Status:** the full loop works end to end on **simulated** platforms. The real-platform
  adapter exists but has not been run yet. All results so far use one seed per configuration.
  The checklist, what the results do and do not show, and the prioritised experiment backlog are
  in [`PROGRESS.md`](PROGRESS.md).
* **Not new:** learned pre-processing in front of a standard codec trained through a codec proxy
  (iSIZE DPP, ByteDance RPP), the "upload at 1440p/4K to get a better codec" tier trick (creator
  knowledge), and closed-loop channel matching against real platforms (robust steganography).
  These are cited, not claimed.
* **What this project adds:** optimisation from the **uploader's side** against an **unknown,
  adaptive** platform pipeline: ladder choice, ingest filtering, rate control and the served
  rendition. It runs in a closed loop with a small real-upload budget, and reports how much
  playback quality is recovered **beyond** the tier trick.

  The survey, the overlap check and the license notes for reusable code are in
  [`docs/related-work.md`](docs/related-work.md). The updated direction is in
  [`idea.md`](idea.md).

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
uv run vreg surrogate --clips park_joy,old_town_cross,ducks_take_off --holdout in_to_tree,sky_gradient --rounds 3 --name surrogate_trust
uv run vreg report runs/surrogate_trust --surrogate --out runs/surrogate.html

scripts/run_experiments.sh    # reproduces every result below (~4 h on a 6-core + RTX 4060 box)
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

## Results (simulated platforms, 3 s 720p clips, RTX 4060 + 6 cores)

Objective: `composite` (higher is better), averaged over the training clips `old_town_cross` and
`park_joy`. Each trial is one full render → transcode → measure cycle, about 15–40 s per clip.

### 1. Black-box search on `sim:ugc_strict`

| sampler | naive upload | best after 40 trials | gain | best found at trial |
|---|---|---|---|---|
| GP-BO | 29.45 | **43.80** | +14.35 | 35 |
| TPE | 29.45 | 43.71 | +14.26 | 29 |
| CMA-ES | 29.45 | 43.54 | +14.09 | 38 |
| random | 29.45 | 42.90 | +13.45 | 24 |

![convergence](docs/img/convergence.png)

All samplers converge on the same recipe:
* upload upscaled ×2 (1440p)
* mild unsharp mask (0.4–0.8, 5×5)
* temporal luma grain ≈ 6–7
* +3–6 % contrast
* +14–21 % saturation, which counteracts the colour loss from the platform's ingest denoiser

GP-BO, TPE and random share the same 8 seeded start-up trials, so their curves overlap early.
With one seed per sampler, the gaps between samplers are within noise. The space is only 7-D
and dominated by one discrete choice, so random search is a strong baseline here.

**Where does the gain come from?** The naive → tier-only → full decomposition (from `vreg transfer`):

| clip | naive | upscale ×2 only (tier trick) | best θ | filter gain over tier trick |
|---|---|---|---|---|
| old_town_cross *(train)* | 54.66 | 60.40 | 62.37 | +1.97 |
| park_joy *(train)* | 4.24 | 22.21 | 25.24 | +3.03 |
| ducks_take_off | 9.39 | 25.97 | 29.37 | +3.40 |
| in_to_tree | 49.49 | 58.28 | 61.99 | +3.71 |
| sky_gradient (synthetic) | 92.44 | 93.07 | 93.16 | +0.09 |
| mandelbrot (synthetic) | -18.98 | -10.22 | -9.38 | +0.84 |
| testsrc (synthetic graphics/text) | 68.26 | 69.43 | **57.87** | **−11.56** |

* **Most of the gain (~80 %) comes from gaming the encode ladder**: a bigger upload gets a better
  codec and more bits. This is the "resolution / tier deception" strategy from the idea. It is
  common creator knowledge (e.g. uploading 1440p to YouTube to get VP9/AV1), and in the
  simulator it holds by construction. Treat it as a sanity check that the optimiser finds the
  known trick, not as a finding. The interesting quantity is the gain **on top of** it.
* The **filtering** part adds a further +2–4 on natural footage, and it **transfers to unseen
  natural clips** (ducks, in_to_tree).
* It **hurts synthetic graphics/text badly** (testsrc −11.6), because sharpening, grain and the
  colour boost damage flat colours and hard edges. Pre-compensation has to be **content-adaptive**.

![park_joy crops](docs/img/crops_park_joy.png)

### 2. Different black box, different optimum: `sim:ugc_av1`

GP-BO improves 38.03 → 41.67. The best θ is again ×2 upscale, which here gets AV1 instead of x264.
Unlike on `ugc_strict`, it uses **no grain** and **stronger sharpening** (1.29, 3×3). This
platform does no ingest denoising, so there is nothing for grain to survive. Its AV1 encode is
soft, so pre-emphasis pays off. Compensation is therefore platform-specific, which is the point of
closing the loop against the real target rather than tuning once.

### 3. Learned pre-compensation through a differentiable emulator (`vreg surrogate`)

`G` is a residual U-Net trained through a frozen emulator `E`, which is itself trained on real
(upload, playback) pairs from the black box. Training clips are park_joy, old_town_cross and
ducks_take_off; in_to_tree and sky_gradient are held out. The upload is fixed at ×2.

* **Without a trust region, `G` exploits the emulator.** Round 0 moved pixels by 18–32 levels on
  average. On those outputs the emulator's PSNR against the real playback fell to 19–25 dB, and
  real quality collapsed: old_town_cross went from 60.4 to 39.0, and park_joy and ducks also fell
  far below the upscale-only baseline (`runs/surrogate_naive`). This is the classic failure of
  optimising against a learned simulator.
* **With a trust region** (L1 penalty on `|G(X)−X|`, residual capped at 0.06), round 0 beats
  the same-tier naive upload on every natural clip, including the held-out one:

| clip | upscale only | learned G (round 0) | gain | GP-BO θ (for comparison) |
|---|---|---|---|---|
| old_town_cross | 60.40 | **65.28** | +4.88 | 62.37 |
| park_joy | 22.21 | 22.96 | +0.75 | 25.24 |
| ducks_take_off | 25.97 | 26.55 | +0.58 | 29.37 |
| in_to_tree *(held out)* | 58.28 | **62.26** | +3.98 | 61.99 |
| sky_gradient *(held out)* | 93.07 | 90.81 | −2.26 | 93.16 |

![surrogate rounds](docs/img/surrogate/surrogate.png)
![in_to_tree crops](docs/img/surrogate/surrogate_crops_in_to_tree.png)

* Later "domain adaptation" rounds, which add `G`'s real pairs to `E`'s data and fine-tune both,
  **did not help in this run**:
  * `G`'s residual kept growing.
  * Gains shrank round over round.
  * The emulator's fidelity on `G`'s outputs barely moved (about 25 dB on park_joy and ducks).

  The report therefore selects the round by training-clip gain only, which was round 0. The run
  was stopped after round 2.
* The learned `G` wins on smoother, mid-detail content (old_town_cross, in_to_tree). The 7-knob
  parametric θ wins on very dense texture (park_joy, ducks), where the emulator is weakest
  (~25 dB).

### Caveats

* The platforms are **simulated**. The ladder rules are plausible but invented, so the specific
  numbers (especially the tier-trick gain) do not carry over to a real platform. The tooling does:
  point `CommandPlatform` at a real upload/download workflow.
* There is one seed per configuration, on 3-second clips.
* The objective is itself a model of quality. Grain and saturation gains are partly what LPIPS and
  VMAF-NEG reward. The found recipe (+contrast, +saturation, unsharp) is close to published VMAF
  and VMAF NEG attacks (Zvezdakova 2019; Siniukov 2021: VMAF NEG +23.6 % with no subjective
  gain). Until independent metrics such as SSIMULACRA2 and DISTS and a viewer study confirm them,
  the filter gains are provisional.

### What the results show so far

1. The closed loop works, and every sampler beats a naive upload. With one seed, the samplers
   cannot be told apart.
2. About 80 % of the gain is the known tier trick. Filtering adds +2–4 on natural footage, and
   this transfers to unseen natural clips.
3. The same θ hurts graphics and text, so pre-compensation must be content-adaptive.
4. The optimum depends on the platform: grain on `ugc_strict`, none on `ugc_av1`.
5. A learned `G` exploits its emulator unless it is constrained. With a trust region it beats
   tier-only on all natural clips. Domain-adaptation rounds did not help yet.

### Next steps

The full backlog is in [`PROGRESS.md`](PROGRESS.md#4-experiment-backlog). Each item lists what is
compared and what counts as success. In short:

* **P0, make the claims solid:**
  * several seeds;
  * independent metrics (SSIMULACRA2, DISTS, XPSNR);
  * an ablation with the contrast and saturation knobs frozen;
  * a Goodhart matrix (optimise one metric, judge with the others);
  * filter-only search at the native tier.
* **P1, method improvements:**
  * content-adaptive θ;
  * a real-forward / surrogate-backward estimator for `G`;
  * an emulator ensemble or a conservative emulator;
  * a BO + `G` hybrid;
  * TuRBO and warm-started BO for a small real budget;
  * a common creator recipe as a baseline.
* **P1, simulator realism:**
  * passthrough below a bitrate cap;
  * AV1 film-grain synthesis;
  * several renditions;
  * cross-platform transfer.
* **P2, real platform and human study:**
  * a platform ladder probe;
  * a real closed loop on own content, with private uploads;
  * a small paired-comparison viewer study;
  * longer and more varied content.

## Repository layout

```
src/vreg/       ff.py content.py platform.py precomp.py metrics.py evaluate.py optimize.py report.py cli.py
src/vreg/surrogate/  models.py loop.py
scripts/        setup_env.sh run_experiments.sh
tests/          tiny-clip unit tests (run in GitHub Actions)
docs/img/       figures used in this README
docs/related-work.md  prior work, overlap check, reusable code and licenses
idea.md         project idea and direction (updated after the prior-work review)
PROGRESS.md     what is done, what the results show, experiment backlog
data/, runs/    generated clips and experiment outputs (git-ignored)
```
