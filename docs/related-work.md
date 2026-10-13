# Related work, overlap check and reusable code

This is a survey of prior work, done to check whether this project duplicates existing work and to
find what it can learn from or reuse. It was compiled in 2026-10 from web searches. Some papers
were read only through their abstracts and metadata; patents were not searched systematically.

## Summary

* **No direct duplicate found.** No paper or repository was found that optimises pre-compensation
  from the uploader's side, against an unknown and adaptive platform pipeline (ladder decision,
  ingest filtering, rate control, served rendition), in a closed loop with a small budget of real
  uploads.
* **The building blocks are not new.** Learned pre-processing before a standard codec, codec
  proxies, the tier trick and closed-loop channel matching all have clear precedents. They are
  listed below and should be cited instead of presented as contributions.

## 1. Learned pre-processing in front of a standard codec

| Work | What it does | Relation to this project |
|---|---|---|
| Bourtsoulatze, Chadha, Fadeev, Giotsas, Andreopoulos, **Deep Video Precoding**, IEEE TCSVT 30(12), 2020 ([arXiv 1908.00812](https://arxiv.org/pdf/1908.00812)) | DNN "precoders" before AVC/HEVC/VP9 in adaptive streaming, with no client change. Reports a 15–45 % rate reduction | Same structure as our `G`, but run by the encoder side with known codec settings. The authors are from iSIZE, a commercial company; check for patents before any commercial use |
| Chadha, Andreopoulos, **Deep Perceptual Preprocessing for Video Coding**, CVPR 2021 ([mlanthology](https://mlanthology.org/cvpr/2021/chadha2021cvpr-deep)) | Rate-aware perceptual preprocessing trained through a virtual codec. About 11 % bitrate saving on AVC, AV1 and VVC | Closest to our surrogate route |
| Ma et al., **Rate-Perception Optimized Preprocessing (RPP)**, ByteDance, 2023 ([arXiv 2301.10455](https://arxiv.org/pdf/2301.10455)) | Adaptive DCT loss and a lightweight network; in production; 16 % saving on average | Production evidence that pre-processing helps |
| Zhao et al., **A Tri-Dynamic Preprocessing Framework for UGC Video Compression** (TDP), PKU/ByteDance ([arXiv 2512.16101](https://arxiv.org/html/2512.16101v1)) | Finds that a deep pre-processor makes **up to 26 % of YouTube-UGC clips worse**. Fixes it with an adaptive intensity factor, an adaptive quantisation level for the codec simulator, and an adaptive λ | Matches our testsrc failure. Template for content-adaptive θ (PROGRESS E6) |
| Talebi et al., **Better Compression with Deep Pre-Editing**, IEEE TIP 2021 ([arXiv 2002.00113](https://arxiv.org/pdf/2002.00113)) | CNN pre-editing before JPEG. Loss = fidelity + bit budget + no-reference quality | Image analogue. Its loss design is a reference for `G` |

## 2. Training through a non-differentiable codec

| Work | Technique | Use here |
|---|---|---|
| Tian et al., **SelfC: Self-Conditioned Probabilistic Learning of Video Rescaling**, ICCV 2021 ([arXiv 2107.11639](https://arxiv.org/pdf/2107.11639)) | A DNN surrogate of H.264/H.265 is used during training; the real codec is used at test time | Same as our `E` |
| Pesnel et al., **SCALED: Surrogate-gradient for Codec-Aware Learning of Downsampling in ABR Streaming**, 2026 ([catalyzex](https://www.catalyzex.com/paper/scaled-surrogate-gradient-for-codec-aware)) | **The real codec runs in the forward pass**; the backward pass uses a surrogate gradient derived from the actual compression error. Up to −23.6 % BD-rate on x264 | Fix for `G` exploiting `E` (PROGRESS E7) |
| CATRF, 2026 ([arXiv 2605.18054](https://arxiv.org/pdf/2605.18054)) | Real HEVC/AV1 codec trained through with a straight-through estimator | Same idea |
| Reich et al., **Differentiable JPEG: The Devil is in the Details**, WACV 2024 ([code](https://github.com/necla-ml/diff-jpeg)) | Accurate differentiable JPEG | Only for images; a reference for building a careful proxy |
| Xu, Bajić, **Differentiable Proxy Learning for Adaptive Quantization Control in H.264**, 2026 | Learned H.264 intra proxy | Possible proxy for an H.264 tier |
| Patent: **Codec rate distortion compensating downsampler** ([USPTO 11765360](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11765360)) | Proxy codec network for a specific standard | Patent landscape for learned downsampling and pre-processing |

## 3. Optimising against a learned simulator

| Work | Use here |
|---|---|
| Trabucco et al., **Conservative Objective Models (COMs)**, ICML 2021 ([PMLR](https://proceedings.mlr.press/v139/trabucco21a.html)) | Names our surrogate failure: the optimiser drifts to inputs where the model overestimates. Suggests a conservative `E` (PROGRESS E8) |
| Eriksson et al., **TuRBO: Scalable Global Optimization via Local BO**, NeurIPS 2019 ([paper](https://proceedings.nips.cc/paper_files/paper/2019/file/6c990b7aca7bc7058f5e98ea909e924b-Paper.pdf)) | Trust-region BO for the black-box route. Use the **BoTorch tutorial (MIT)**, not the original repository (non-commercial, see §6) |

## 4. Metric hacking (important for interpreting our results)

| Work | Finding |
|---|---|
| Zvezdakova et al., **Hacking VMAF with Video Color and Contrast Distortion**, 2019 ([arXiv 1907.04807](https://arxiv.org/pdf/1907.04807)) | Colour and contrast changes raise VMAF |
| Siniukov et al., **Hacking VMAF and VMAF NEG**, 2021 ([arXiv 2107.04510](https://arxiv.org/pdf/2107.04510)) | Tuned preprocessing raises VMAF by up to 218.8 % and **VMAF NEG by up to 23.6 %**, while subjective quality falls or stays the same |
| Fingscheidt et al., **On Optimizing Image Codecs for VMAF NEG**, 2026 ([arXiv 2602.21336](https://arxiv.org/html/2602.21336v1)) | VMAF NEG is still exploitable when used as a training target. Proposes a robust MSE + MS-SSIM + VMAF NEG loss |

Our `ugc_strict` optimum (+contrast, +14–21 % saturation, unsharp, grain) is close to these known
attacks. That is why PROGRESS E2–E4 (independent metrics, ablation of the `eq` knobs, a Goodhart
matrix) are P0.

## 5. Closed-loop adaptation to real platforms, and platform behaviour

| Source | Relation |
|---|---|
| Zhao et al., **Transport Channel Matching (TCM)** for robust steganography; follow-ups such as [arXiv 2211.10095](https://arxiv.org/pdf/2211.10095) and [EURASIP 2023](https://jis-eurasipjournals.springeropen.com/articles/10.1186/s13635-023-00148-x) | Repeatedly uploads images to social networks until recompression stops changing them. The methodological precedent for our real-platform loop. [Follow-up work](https://archive.cps-vo.org/node/93191) notes that repeated uploads leave detectable traces and are a behavioural risk |
| Norkin, Birkbeck, **AV1 film grain synthesis** ([DCC 2018](https://norkin.org/pdf/DCC_2018_AV1_film_grain.pdf)); [grav1synth](https://github.com/rust-av/grav1synth) | Denoise, then re-synthesise grain at the decoder. Explains why grain behaves differently on AV1 tiers |
| Meta Engineering, **[How Facebook encodes your videos](https://engineering.fb.com/2021/04/05/video-engineering/how-facebook-encodes-your-videos/)** (2021) | First-party description of how the encode family (H.264 vs VP9) is chosen per video by expected benefit, so the ladder is content- and popularity-dependent |
| Yu et al., **Predicting the Quality of Compressed Videos With Pre-Existing Distortions**, IEEE TIP 2021 ([ResearchGate](https://www.researchgate.net/publication/354232619_Predicting_the_Quality_of_Compressed_Videos_With_Pre-Existing_Distortions)) | The UGC setting: the platform's reference is already distorted |
| YouTube creator practice: [wpreset](https://wpreset.com/why-some-creators-re%E2%80%91encode-their-video-to-4k-the-hidden-trick-to-prevent-blurriness-after-upload-on-youtube/), [filmdaft](https://filmdaft.com/how-to-get-better-image-quality-on-your-youtube-videos/), [OBS forum](https://obsproject.com/forum/threads/lifehack-how-to-improve-image-quality-of-livestreams-uploads-to-youtube-by-upscaling.140425) | "Upload at 1440p/4K to get VP9/AV1". This is the tier trick, so it is common knowledge. Reports on whether 1080p uploads also get VP9 are mixed |
| Bilibili creator posts: [opus 891367754490183699](https://www.bilibili.com/opus/891367754490183699), [cv5778825](https://www.bilibili.com/read/cv5778825) | Community-reported limits: 1080p at about 6000 kbps average and 24000 kbps peak; 4K at about 20000 kbps average. Uploads under the limits reportedly skip the "second encode". Not official; motivates PROGRESS E12 and E15 |

Existing open-source tools ([web-video-optimizer](https://github.com/Aplyca/web-video-optimizer),
[Video-AV1-Optimizer](https://github.com/emike09/Video-AV1-Optimizer), Av1an, ab-av1) search
encoder settings against VMAF for **your own** encode. None of them models a platform's
re-encode. Adversarial work on codecs ([RoVISQ, NDSS](https://www.ndss-symposium.org/ndss-paper/rovisq-reduction-of-video-service-quality-via-adversarial-attacks-on-deep-learning-based-video-compression))
targets learned codecs and aims to degrade quality, which is the opposite goal.

## 6. Reusable code and licenses

Licenses below come from PyPI metadata or the repository's LICENSE file, checked in 2026-10.
Re-check them before vendoring any code. BSD and MIT code must keep its copyright notice.

### Fine to use (permissive)

| Package or repository | License | Possible use |
|---|---|---|
| optuna, cmaes (in use) | MIT | black-box search |
| lpips (in use) | BSD | metric |
| botorch (incl. the TuRBO tutorial) | MIT | trust-region BO, transfer BO (E10) |
| nevergrad | MIT | more black-box optimisers |
| piq | Apache-2.0 | DISTS and other full-reference metrics (E2) |
| DISTS-pytorch | MIT | DISTS (E2) |
| ssimulacra2 (Python, Pacidus/py-ssimulacra2); libjxl `ssimulacra2` tool | BSD-3 | SSIMULACRA2 (E2) |
| Netflix vmaf / libvmaf | BSD-2-Clause-Patent | already used through ffmpeg |
| necla-ml/diff-jpeg | BSD-3 (per repository metadata) | reference differentiable codec |
| torchjpeg | MIT | DCT-domain utilities |
| CompressAI | BSD-3-Clause-Clear (**no patent grant**) | learned-codec baselines |
| yt-dlp | Unlicense | download of one's own uploads (`CommandPlatform`) |
| easyVmaf, ffmpeg-quality-metrics | MIT | metric plumbing |

### Avoid, or use with care

| Package or repository | Issue |
|---|---|
| **pyiqa / IQA-PyTorch** | PyPI metadata says **PolyForm-Noncommercial-1.0.0**, and the pretrained weights carry mixed licenses. Do not make it a core dependency; use piq or the standalone implementations above |
| **uber-research/TuRBO** | Uber **non-commercial** license, which explicitly covers "derivative works, outputs, or results". Use the BoTorch tutorial implementation instead |
| BtbN FFmpeg GPL builds (downloaded by `scripts/setup_env.sh`) | Fine to download and use locally. Bundling or redistributing them requires GPL compliance |
| Xiph `derf` test clips (SVT park_joy, old_town_cross, ducks_take_off, in_to_tree) | Per-clip terms; the SVT sequences fall under the EBU/SVT conditions. Research use and small crops in docs are believed fine but were not verified. Do not redistribute the clips themselves |
| Platform upload tooling (e.g. YouTube Data API, community Bilibili uploaders) | Bound by the platform's terms of service. Use private or unlisted uploads of your own content, rate-limited |
| iSIZE / Sony and other pre-processing patents | Relevant if the project is ever used commercially; not a concern for research |

**This repository has no LICENSE file yet.** Until one is added, others cannot legally reuse
the code. MIT or Apache-2.0 is compatible with every dependency in use.
