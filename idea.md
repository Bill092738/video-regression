# Idea: closed-loop black-box pre-compensation for platform video compression

This note records the project idea and how it has been refined since it was first written. It
also states which parts are already known from prior work, which are left for this project, and
the questions that are still open. What has been built and measured so far is in
[`PROGRESS.md`](PROGRESS.md). The prior-work survey and license notes are in
[`docs/related-work.md`](docs/related-work.md).

## 1. Problem

* **Platform compression is adaptive and hidden.** Video platforms such as YouTube and Bilibili
  re-encode every upload. The codec (AVC/H.264, HEVC/H.265, VP9, AV1), the rate-control mode
  (CRF, capped CRF, CBR, VBR), the bit budget and the ingest filtering can all depend on:
  * the upload's resolution and frame rate;
  * its bitrate;
  * its content;
  * and, reportedly, on popularity or account tier.

  None of this is documented, and none of it can be differentiated through.
* **"Best master in" is not "best playback out".** Uploading a lossless or very high bitrate
  master can still give poor playback. Aggressive ingest denoising, low bit budgets and lowpass
  filtering cause blocking, banding and texture loss.

## 2. Core idea

Treat the platform as a black-box degradation channel `f_platform(·)`. Optimise a
pre-compensation `G(X; θ)`, applied before upload, so that the playback
`Y = f_platform(G(X; θ))` is as close as possible to the source `X` under a perceptual metric.

```
source X ──► G(X; θ) ──► upload ──► [ platform black box f_platform ] ──► playback Y
   │                                                                         │
   └──────────────────────────► metric(Y, X) ◄───────────────────────────────┘
                                       │
                         θ update (black-box search or learned G)
```

## 3. Positioning against prior work (updated)

Prior work already covers several parts of the original idea. The project should not present
those parts as new.

| Part of the idea | Status | Main references |
|---|---|---|
| Learned pre-processing in front of a standard encoder, trained through a differentiable codec proxy, with no decoder change | **Well established**, including deployed systems, but always run by the encoder side with a known codec and settings | iSIZE Deep Video Precoding (TCSVT 2020) and Deep Perceptual Preprocessing (CVPR 2021); ByteDance RPP (2023); Tri-Dynamic Preprocessing for UGC; Google Deep Pre-Editing for JPEG (TIP 2021) |
| Training through a real, non-differentiable codec | **Established** for known codecs: surrogate networks (SelfC, ICCV 2021), or a real codec in the forward pass with a surrogate gradient in the backward pass (SCALED, 2026) | see `docs/related-work.md` |
| "Upload at a higher resolution to get a better codec or more bits" (tier deception) | **Common creator knowledge.** Examples: 1440p/4K on YouTube to get VP9/AV1, and Bilibili's bitrate caps and "no second encode" thresholds | creator blogs and forums |
| Upload, download and adapt repeatedly against a real platform | **Done before, in another field.** Robust steganography uses transport channel matching (TCM), uploading images to social networks repeatedly until recompression stops changing them | Zhao et al.; follow-up work notes that repeated uploads are themselves detectable |
| Sharpening, contrast or saturation raising VMAF without real quality gains | **Known metric vulnerability.** VMAF NEG reduces it but does not remove it | Zvezdakova 2019; Siniukov 2021; Fingscheidt 2026 |
| A learned simulator being exploited by the optimiser | **Known failure mode** of offline model-based optimisation | Conservative Objective Models (ICML 2021); trust-region BO (TuRBO) |

**What remains specific to this project.** We found no published system or repository that does
all of the following together:

* optimises the pre-compensation from the **uploader's side**;
* treats the **whole platform pipeline** as unknown, including:
  * the discrete ladder or tier decision;
  * ingest filtering;
  * rate control;
  * the rendition actually served to the viewer;
* closes the loop against the **real target platform** with a small evaluation budget.

The research question is therefore narrower than "can pre-processing help compression", which is
already known. It is:

> With a budget of tens of real uploads, how much playback quality can an uploader recover on an
> unknown and adaptive platform pipeline, beyond the known tier trick? Does that gain hold up
> under metrics that the optimiser did not optimise, and under human viewing?

## 4. Approach

### 4.1 Black-box search over an interpretable θ

Real uploads are expensive: each one needs an upload, a transcode, a download and a measurement,
which takes minutes to hours. A sample-efficient search is therefore needed.

* **Bayesian optimisation** (GP-BO, or trust-region BO such as TuRBO from the BoTorch tutorial)
  over a low-dimensional θ.
* **Evolution strategies** (CMA-ES) and **TPE**, with random search as the control.
* **Warm start from the simulator**: tune in simulation, then spend the real-platform budget only
  to refine (transfer BO).

### 4.2 Learned pre-compensation through a differentiable emulator

* Fit an emulator `E ≈ f_platform` on real (upload, playback) pairs. Train `G` by
  backpropagating through `E`. Periodically validate on the real platform and add the new pairs
  to `E`'s training data (domain adaptation).
* The emulator **will** be exploited. Planned guards:
  * a trust region on `|G(X) − X|`, already implemented;
  * an ensemble of `E` with a disagreement penalty;
  * a conservative emulator (COMs style);
  * the real-forward / surrogate-backward estimator
    `y = y_real + E(x) − stopgrad(E(x))`, so that `G` only ever sees real outputs in the forward
    pass.

### 4.3 Hybrid

Search the discrete and global knobs (tier, grain level, colour) with BO. Learn `G` for local
detail on top, conditioned on the chosen tier.

## 5. Search space (the "levers")

The optimiser does not edit pixels arbitrarily. It moves levers that change how the platform's
encoder behaves:

* **Dithering and film grain.** Breaks up banding in flat areas and pushes the encoder to spend
  bits on smooth or dark regions. This interacts with the platform's denoiser and, on AV1, with
  film-grain synthesis.
* **Pre-emphasis and unsharp masking.** Counters the lowpass and denoise that platforms apply
  when they lower the bitrate.
* **Resolution, bitrate and format choices that select a tier.** Upscale to 1440p/4K, or set the
  upload bitrate around the platform's "no re-encode" caps.
* **Colour and contrast.** Can compensate for desaturation from the platform's denoiser. This is
  also exactly where metric hacking lives, so every gain here must be confirmed by an independent
  metric.
* **Content adaptivity.** The best θ for natural footage damages graphics and text. θ should
  depend on content features or be chosen per shot.

## 6. Evaluation principles

1. **Baselines in every table:**
   * naive upload;
   * tier trick only (upscale only);
   * a common creator recipe;
   * the found θ or the learned `G`.
2. **Separate the objective from the judge.** Optimise one metric (or composite), and report
   metrics that were not optimised as well: SSIMULACRA2, DISTS, XPSNR. A gain counts only if
   the metrics it was not optimised for also agree.
3. **Held-out clips and held-out content types**, including synthetic graphics and text.
4. **Several seeds**, reported as a mean with a spread.
5. **A small subjective check** (paired comparison) before claiming a perceptual improvement.
6. **Report costs** as well: upload size, upload time, number of real uploads used.

## 7. Ethics, terms of service and licensing

* Run real-platform experiments only with your own account and your own content, uploaded as
  private or unlisted, rate-limited, and within the platform's terms of service. Repeated
  automated uploads can be detected as abnormal behaviour.
* The aim is better playback of one's own content. It is not to evade content moderation or
  detection, and it is not to make the platform spend extra resources.
* Use only third-party code whose license allows the intended use. For example, prefer the MIT
  BoTorch TuRBO tutorial over the non-commercial Uber TuRBO repository, and avoid non-commercial
  IQA toolkits as core dependencies. See `docs/related-work.md`.

## Original notes (summary of the first version)

The first version of this note (in Chinese) described the same pipeline:
* a pre-compensation generator `G(X; θ)`, which adds high-frequency perturbation, micro-structure
  or pseudo-noise;
* a non-differentiable platform `f_platform`;
* playback captured by downloading or stream capture;
* a perceptual metric (VMAF, SSIMULACRA2, LPIPS);
* black-box updates to θ.

It argued that simple regression cannot model a channel that includes quantisation, DCT, motion
estimation and temporal filtering. It proposed two routes: Bayesian optimisation or CMA-ES, and a
differentiable emulator with periodic real-platform calibration. It listed three lever families:
grain or dithering, non-linear unsharp masking, and resolution or chroma-format tricks that
trigger a higher transcoding tier. All of these are implemented in this repository, at least in
simulation.
