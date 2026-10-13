# Progress

Status as of 2026-10. The direction is in [`idea.md`](idea.md). The prior-work and license notes
are in [`docs/related-work.md`](docs/related-work.md). The full result tables are in the
[README](README.md#results-simulated-platforms-3-s-720p-clips-rtx-4060--6-cores).

## 1. What has been done

| Area | State | Where |
|---|---|---|
| Clip set: 3 synthetic clips + 4 Xiph natural clips, 3 s, 720p | done | `vreg clips`, `src/vreg/content.py` |
| Simulated black-box platforms `ugc_strict` and `ugc_av1`: ladder chosen by upload resolution, ingest denoise, capped CRF, fixed 720p rendition | done | `src/vreg/platform.py` |
| Real-platform adapter `CommandPlatform` (user-supplied upload and download commands, polling) | implemented, **not yet run on a real platform** | `src/vreg/platform.py` |
| Parametric pre-compensation θ (7-D): denoise, contrast, saturation, upscale/tier, unsharp amount and radius, temporal grain | done | `src/vreg/precomp.py` |
| Metrics: VMAF NEG, LPIPS, CAMBI, PSNR, SSIM; objective `composite = VMAF_NEG − 40·LPIPS − 2·CAMBI` | done | `src/vreg/metrics.py` |
| Black-box search: GP-BO, TPE, CMA-ES, random (Optuna, resumable, cached evaluations) | done | `vreg optimize` |
| Transfer to unseen clips, and the naive → tier-only → full decomposition | done | `vreg transfer` |
| Learned `G` through a differentiable emulator `E`, with a trust region and real-platform validation rounds | done | `vreg surrogate`, `src/vreg/surrogate/` |
| HTML report: convergence, sensitivity, crops, transfer table, surrogate rounds | done | `vreg report` |
| Unit tests on tiny clips, CI without torch | done | `tests/`, `.github/workflows/ci.yml` |
| Prior-work and license review | done | `docs/related-work.md` |

## 2. What the current results show

All numbers are from **simulated** platforms, with **one seed** per configuration and 3-second
clips. They show that the method and the tooling work. They are not yet evidence about real
platforms.

1. **The closed loop works and all samplers improve on a naive upload.** On `sim:ugc_strict`,
   `composite` rose from 29.45 to about 43–44 within 40 trials. GP-BO scored 43.80 and random
   search 42.90. With one seed, **the samplers cannot be told apart.** The space is small and
   dominated by one discrete choice, so random search is a strong baseline.
2. **About 80 % of the gain is the tier trick.** Uploading at ×2 (1440p) gets a better codec and
   more bits. This trick is common creator knowledge (see `docs/related-work.md`), and the
   simulator contains it by construction. It is a sanity check, not a finding.
3. **Filtering adds a further +2 to +4 on natural footage, and it transfers to unseen natural
   clips** (ducks_take_off +3.4, in_to_tree +3.7 over tier-only).
4. **The same θ damages graphics and text** (testsrc −11.6 compared with tier-only). Pre-
   compensation has to be **content-adaptive**. This matches the published bad-case rate for
   deep pre-processors on UGC (up to 26 % in Tri-Dynamic Preprocessing).
5. **Different platform, different optimum.** On `sim:ugc_av1` the best θ uses no grain and
   stronger sharpening, whereas on `ugc_strict` it uses grain. The loop has to run against the
   actual target platform.
6. **The learned `G` exploits the emulator unless it is constrained.** Without a trust region,
   real quality collapsed (old_town_cross 60.4 → 39.0). With a trust region, round 0 beat
   tier-only on every natural clip, including the held-out in_to_tree (+4.0), and beat GP-BO θ on
   smoother content. Later domain-adaptation rounds **did not help**.

### What the results do **not** yet show

* That any of this transfers to a real platform.
* That the colour, contrast and sharpening gains are perceptual improvements and not metric
  hacking. The found recipe is close to known VMAF and VMAF NEG attacks, and no metric outside
  the objective has been checked yet.
* That differences between samplers, or between `G` and θ, are larger than run-to-run noise.

## 3. Open problems

* **No independent judge yet.** Every reported number is either the objective itself or a
  component of it.
* **Domain-adaptation rounds degrade.** `E`'s fidelity on `G`'s outputs stays at about 25 dB on
  dense texture, and `G`'s residual keeps growing.
* **The simulator lacks some real behaviours:**
  * passthrough or "no second encode" below a bitrate cap, which Bilibili users report (community
    claims, not verified);
  * several renditions served to different viewers;
  * AV1 film-grain synthesis;
  * a delayed or popularity-dependent switch to AV1 or VP9;
  * non-deterministic re-encodes.
* **No LICENSE file** in the repository yet (see `docs/related-work.md`).

## 4. Experiment backlog

Each item names the question it answers, what is compared, and what counts as success.
**P0** items should be done before any claim is made; **P1** items extend the method; **P2** items
are larger studies.

### P0: make the existing claims solid

| # | Experiment | Compare | Success criterion |
|---|---|---|---|
| E1 | **Seeds and variance**: 3–5 seeds per sampler on `ugc_strict`, and 3 seeds for `vreg surrogate` | GP-BO vs TPE vs CMA-ES vs random; learned `G` vs GP-BO θ | Mean ± spread reported. A sampler or method "wins" only if the gap is larger than the spread |
| E2 | **Independent metrics**: add SSIMULACRA2 (BSD) and DISTS (piq, Apache-2.0), plus ffmpeg XPSNR. Re-score the existing best θ and `G` without re-optimising | naive / tier-only / best θ / `G`, on objective metrics vs unoptimised metrics | The filter gain over tier-only keeps the same sign on SSIMULACRA2 and DISTS |
| E3 | **Metric-hacking ablation**: re-run GP-BO with the contrast and saturation (`eq`) knobs frozen at 0 | full θ vs θ without `eq` | Tells how much of the +2–4 filter gain is colour or contrast (suspect) and how much is grain or sharpening |
| E4 | **Goodhart matrix**: optimise for VMAF NEG only, for composite, and for SSIMULACRA2; evaluate each result with all metrics | 3 objectives × all metrics | The objective whose optimum still scores well on the other metrics becomes the default |
| E5 | **Filter-only at the native tier**: search θ with the upscale fixed at ×1 | filter gain at ×1 vs filter gain at ×2 | Separates the filter gain from the tier gain more cleanly than the transfer table |

### P1: method improvements

| # | Experiment | Compare | Success criterion |
|---|---|---|---|
| E6 | **Content-adaptive θ**: gate or scale θ by content features (texture energy, flat-area fraction, text detection), similar to the TDP adaptive intensity factor; or run a per-clip search warm-started from the global θ | global θ vs gated θ vs per-clip θ | testsrc no longer worse than tier-only; natural clips keep at least 80 % of their gain |
| E7 | **Real-forward / surrogate-backward estimator** for `G`: `y = y_real + E(x) − stopgrad(E(x))`, following SelfC and SCALED | current `G` (emulator forward) vs this estimator | No collapse without a trust region; the gain on dense texture (park_joy, ducks) moves closer to GP-BO θ |
| E8 | **Emulator robustness**: an ensemble of `E` with a disagreement penalty; a conservative (COMs-style) `E`; `G`-like perturbations in round-0 data; temporal context (stacked frames) | trust region only vs each option | Emulator PSNR on `G`'s outputs above 30 dB, and domain-adaptation rounds improve instead of degrading |
| E9 | **Hybrid**: BO for tier, grain and colour, then `G` for local detail on top of the chosen θ | θ alone vs `G` alone vs hybrid | Hybrid ≥ the best of the two on every natural clip |
| E10 | **Trust-region BO** (BoTorch TuRBO tutorial, MIT) and BO warm-started from simulator results | GP-BO vs TuRBO vs warm-started BO, at budgets of 10, 20 and 40 trials | Reaches 90 % of the 40-trial gain within about 15 trials (this sets the real-platform budget) |
| E11 | **Common creator recipe as a baseline**: lanczos ×2 + mild unsharp, at a fixed upload bitrate | creator recipe vs best θ | Shows what the optimiser adds over standard advice |

### P1: simulator realism

| # | Experiment | Compare | Success criterion |
|---|---|---|---|
| E12 | Add a **passthrough branch below a bitrate cap** (Bilibili-style "no second encode"); make upload bitrate part of θ | old vs new profile; does the optimiser find the passthrough region? | The optimiser discovers the bitrate threshold when it exists |
| E13 | Add **AV1 film-grain synthesis** (denoise + grain table) to `ugc_av1`, and **several renditions** (480p/720p/1080p, weighted) | single 720p rendition vs multi-rendition objective | θ that is robust across renditions |
| E14 | **Cross-platform transfer**: apply the `ugc_strict` θ to `ugc_av1` and the reverse | transferred vs native θ | Quantifies how much tuning per platform matters |

### P2: real platform and human study

| # | Experiment | Compare | Success criterion |
|---|---|---|---|
| E15 | **Platform probe** (no optimisation). Upload a resolution × bitrate grid of your own test clip (private or unlisted, rate-limited). Record the codecs and bitrates served (`yt-dlp -F`), and re-upload the same file to measure non-determinism | upload resolution and bitrate vs served codec and bitrate | A measured ladder, which then sets the simulator profile |
| E16 | **Real closed loop** through `CommandPlatform` with a budget of about 15–20 uploads, warm-started from simulation | naive vs tier-only vs creator recipe vs found θ | A filter gain over tier-only on unoptimised metrics, on a real platform |
| E17 | **Subjective check**: paired comparison (2AFC) between tier-only and best θ or `G`, with 5–10 viewers, on held-out clips | preference rate | Preference above 50 % with a confidence interval that excludes 50 % |
| E18 | **Longer and more varied content**: 10–30 s clips; UGC, animation, screen recordings, dark scenes with banding risk | per content class | Per-class gain table; tells where pre-compensation should be switched off |

## 5. Engineering backlog

* Add a `LICENSE` file (MIT or Apache-2.0 fit the current dependencies).
* `vreg.metrics`: SSIMULACRA2, DISTS and XPSNR as report-only metrics (E2).
* `vreg optimize`: `--seeds` and aggregation in `vreg report` (E1).
* Freezing parameters from the CLI, e.g. `--fix eq=0,upscale=1` (E3, E5).
* Record upload size and bitrate as a cost in every evaluation.
* A `CommandPlatform` example script for one platform, with rate limiting and private uploads
  (E15, E16).
