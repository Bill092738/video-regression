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
| HTML report: convergence, sensitivity, crops, transfer table, surrogate rounds; mean ± sd across seeds | done | `vreg report` |
| Report-only judges SSIMULACRA2, DISTS, XPSNR (not in `composite`), back-filled from cached playbacks; frozen knobs (`--fix`) | done (P0) | `vreg rescore`, `vreg optimize --fix`, `src/vreg/metrics.py` |
| P0 experiments E1–E5 | done | [§6](#6-p0-results), `scripts/run_p0.sh`, `scripts/p0_tables.py` |
| P1 plan: content-scoped learned `G`, gate, pre-registered classes (E19–E23) | **planned** | [`docs/P1-plan.md`](docs/P1-plan.md) |
| Unit tests on tiny clips, CI without torch | done | `tests/`, `.github/workflows/ci.yml` |
| Prior-work and license review | done | `docs/related-work.md` |

## 2. What the current results show

> **Revised by the P0 results ([§6](#6-p0-results)).** Points 1, 3 and 6 below are the pre-P0
> claims. P0 replaced point 1: GP-BO > TPE > random over five seeds. It did not confirm point 3:
> SSIMULACRA2 and XPSNR are lower than tier-only on every natural clip, on every seed. It
> downgraded point 6: `G` beats tier-only on 3/4 natural clips and beats θ only on
> old_town_cross. Point 2, the tier gain, is confirmed by every independent metric.

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
  hacking. P0 checked them with three independent metrics, which did not confirm them. A viewer
  study is still missing.

## 3. Open problems

* **The objective and the independent judges disagree on the filter gain** (P0, §6). No single
  metric optimised so far is free of regressions on the others, and there is no viewer study
  to decide which side is right.
* **The learned `G`'s advantage is narrow and untested by content type.** It beats GP-BO θ only
  on old_town_cross, and the four natural clips are too few to tell which content it suits. P1
  addresses this with pre-registered content classes and a larger clip set.
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

Execution plan (code changes, commands, decision rules): [`docs/P0-plan.md`](docs/P0-plan.md).
**All five are done.** The results and verdicts are in [§6](#6-p0-results).

| # | Experiment | Compare | Success criterion |
|---|---|---|---|
| E1 | ✅ **Seeds and variance**: 3–5 seeds per sampler on `ugc_strict`, and 3 seeds for `vreg surrogate` | GP-BO vs TPE vs CMA-ES vs random; learned `G` vs GP-BO θ | Mean ± spread reported. A sampler or method "wins" only if the gap is larger than the spread |
| E2 | ✅ **Independent metrics**: add SSIMULACRA2 (BSD) and DISTS (piq, Apache-2.0), plus ffmpeg XPSNR. Re-score the existing best θ and `G` without re-optimising | naive / tier-only / best θ / `G`, on objective metrics vs unoptimised metrics | The filter gain over tier-only keeps the same sign on SSIMULACRA2 and DISTS |
| E3 | ✅ **Metric-hacking ablation**: re-run GP-BO with the contrast and saturation (`eq`) knobs frozen at 0 | full θ vs θ without `eq` | Tells how much of the +2–4 filter gain is colour or contrast (suspect) and how much is grain or sharpening |
| E4 | ✅ **Goodhart matrix**: optimise for VMAF NEG only, for composite, and for SSIMULACRA2; evaluate each result with all metrics | 3 objectives × all metrics | The objective whose optimum still scores well on the other metrics becomes the default |
| E5 | ✅ **Filter-only at the native tier**: search θ with the upscale fixed at ×1 | filter gain at ×1 vs filter gain at ×2 | Separates the filter gain from the tier gain more cleanly than the transfer table |

### P1 (current): content-scoped learned pre-compensation

Execution plan: [`docs/P1-plan.md`](docs/P1-plan.md). P0 confirmed only the tier gain. P1 tests
whether a learned `G` beats the strongest surviving filter, a class-tuned sharpening-only θ, for
a content class defined **before** training. It also adds a gate that switches filters off
elsewhere. Classes, hypotheses, splits and decision rules are pre-registered, and the confirm
split is evaluated once.

| # | Experiment | Compare | Success criterion |
|---|---|---|---|
| E19 | **Content set and classes**: about 60 licensed 3 s clips; SI/TI, flat and dark fractions; classes A (natural, low motion, moderate detail), B (dense texture or high motion), C (flat, gradient or dark), D (graphics, text, screen, animation); train/dev/confirm splits grouped by source | none (set-up) | ≥ 12 clips in A and B, ≥ 8 in C and D; `docs/p1/` committed before any training |
| E20 | **Baselines per class**: naive, tier-only, P0 global sharpening-only θ, class-tuned sharpening-only θ | per class, all metrics | Fixes the baseline `G` must beat |
| E21 | **`G` loss selection**: LPIPS vs DISTS vs LPIPS+DISTS (a `G` is never judged by its own loss) | on class A dev | One loss fixed for E22–E23 |
| E22 | **Specialist vs global `G`**, 3 seeds | global `G`, specialist A, specialist B vs class-tuned sharpening-only θ, on dev | Choose the `G` and the gate table before E23 |
| E23 | **Gate and confirmation** (once, on the confirm split) | gated system vs tier-only vs global sharpening-only θ | H1: ≥ 2 judges with a 95 % CI over clips excluding 0, SSIMULACRA2 not worse, sign holds in ≥ 2/3 seeds. Gate: no class worse than tier-only on SSIMULACRA2 or XPSNR |

E6 (content-adaptive θ) is absorbed into E19–E23: the gate is its content-adaptive part.

### P1-later: method improvements

These items wait until P1 shows whether a learned `G` is worth improving.

| # | Experiment | Compare | Success criterion |
|---|---|---|---|
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
* P1 code changes (manifest-driven clips, `vreg features`, `vreg classify`, clip selectors,
  `--g-loss`, `--pairs-cache`, P1 tables): see [`docs/P1-plan.md`](docs/P1-plan.md).
* Record upload size and bitrate as a cost in every evaluation.
* A `CommandPlatform` example script for one platform, with rate limiting and private uploads
  (E15, E16).

## 6. P0 results

Plan: [`docs/P0-plan.md`](docs/P0-plan.md). Everything below is on the **simulated** platforms. The
tables are generated by `scripts/p0_tables.py` from each run's `rescore.json` (`vreg rescore`).
**All gains are signed so that positive means better**, including LPIPS and DISTS, where lower is
better. "Natural clips" are park_joy and old_town_cross (training), plus ducks_take_off and
in_to_tree (unseen). The per-clip signs are listed in that order.

Judges: SSIMULACRA2 (py-ssimulacra2, mean over every 3rd frame), DISTS (piq, GPU, every 3rd
frame) and luma XPSNR (ffmpeg `xpsnr` filter). None of them is part of `composite`.

### E2: independent metrics on the existing results

Mean gain over tier-only on the natural clips (per-clip signs in brackets):

| configuration | composite | vmaf_neg | lpips | ssimulacra2 | dists | xpsnr | verdict |
|---|---|---|---|---|---|---|---|
| GP-BO θ, ugc_strict (`strict_gp`) | +3.03 (+ + + +) | +1.87 (+ + + +) | +0.0280 (+ + + +) | −2.23 (− − − −) | +0.0003 (+ − − +) | −0.75 (− − − −) | **not confirmed** (1/3) |
| GP-BO θ, ugc_av1 (`av1_gp`) | +4.94 (+ + + +) | +3.13 (+ + − +) | +0.0437 (+ + + +) | +1.90 (+ − − −) | +0.0071 (+ + − +) | +0.21 (+ − − −) | stands (3/3), but carried by park_joy |
| learned G round 0 (`surrogate_trust`) | +2.55 (+ + + +) | −1.03 (− + − −) | +0.0832 (+ + + +) | −5.17 (− − − −) | +0.0101 (− + + +) | +1.24 (+ + + +) | stands (2/3), SSIMULACRA2 disagrees on every clip |
| *reference: tier gain (tier-only − naive), ugc_strict* | +12.27 (+ + + +) | +9.50 (+ + + +) | +0.0657 (+ + + +) | +23.02 (+ + + +) | +0.0267 (+ + + +) | +1.54 (+ + + +) | stands (3/3) |

* **The tier gain is real.** Every judge agrees on every natural clip.
* **The `ugc_strict` filter gain over tier-only is not confirmed.** The full θ (+6 % contrast,
  +15 % saturation, unsharp, grain 6) raises the objective and its components on every natural
  clip. It lowers SSIMULACRA2 and XPSNR on every natural clip, and DISTS is flat. This is the
  pattern expected from metric hacking.
* `ugc_av1` passes the rule on the means, but only because of park_joy. There, the AV1 tier-only
  upload is *worse* than naive on SSIMULACRA2 (−49.9 vs −29.4), and θ recovers +14.3 of that. On
  the other three natural clips, SSIMULACRA2 and XPSNR are slightly lower with θ than with
  tier-only.
* The learned `G` passes on XPSNR (4/4 clips) and DISTS (3/4). SSIMULACRA2 is lower on every clip
  (mean −5.2), and VMAF NEG is lower on 3/4. Later rounds are worse on SSIMULACRA2 (sky_gradient
  −13 → −26).
* testsrc, `ugc_strict` θ vs tier-only: composite −11.6, but SSIMULACRA2 +1.3, DISTS −0.004 and
  XPSNR −7.0. The large composite loss there is mostly CAMBI reacting to the added grain.

### E3: contrast and saturation frozen (`strict_gp_noeq`, `--fix contrast=1,saturation=1`)

Best θ: ×2, unsharp 0.89 (5×5), **no grain**, no denoise. Mean gain over tier-only on the natural
clips:

| metric | full θ (`strict_gp`) | `eq` frozen | ratio |
|---|---|---|---|
| composite | +3.03 | +2.18 | 72 % |
| vmaf_neg | +1.87 | +1.03 | 55 % |
| lpips | +0.0280 | +0.0277 | 99 % |
| ssimulacra2 | −2.23 | −1.74 | (full gain ≤ 0) |
| dists | +0.0003 | +0.0035 | (full gain ≈ 0) |
| xpsnr | −0.75 | +0.52 | (full gain ≤ 0) |

* With `eq` frozen, the optimiser also drops the grain. The result keeps 72 % of the composite
  gain, 55 % of the VMAF NEG gain and all of the LPIPS gain, and it **passes the E2 rule (2/3:
  DISTS and XPSNR)**. SSIMULACRA2 is still lower than tier-only.
* So the colour/contrast boost and the grain it brings along account for about 30 % of the
  composite gain and 45 % of the VMAF NEG gain. They are also what turns XPSNR and DISTS negative.
  This is the suspect part. The plain sharpening part survives two of three judges.

### E5: native tier, filter-only (`strict_gp_native`, `--fix upscale=1`)

Best θ: ×1, **+11 % contrast**, −1 % saturation, grain 3.2, no sharpening. The filter gain at ×1
(best − naive) vs at ×2 (`strict_gp`, best − tier-only), mean over the natural clips:

| metric | ×2 (`strict_gp`) | ×1 (`strict_gp_native`) |
|---|---|---|
| composite | +3.03 | +3.02 |
| vmaf_neg | +1.87 | +2.49 |
| lpips | +0.0280 | +0.0095 |
| ssimulacra2 | −2.23 | −2.48 |
| dists | +0.0003 | −0.0023 |
| xpsnr | −0.75 | −2.13 |

* At the native tier the composite gain is as large as at ×2. It comes almost entirely from
  contrast, and **no judge agrees (0/3)**. At ×1 the filter gain is metric hacking: VMAF NEG
  rewards the contrast boost.

### E4: Goodhart matrix

![Goodhart matrix](docs/img/p0/goodhart_matrix.png)

Rows: optimised objective (GP-BO, 40 trials). Cells: gain of the best θ over tier-only, mean over
the natural clips. Bold cells were optimised by that row (composite contains VMAF NEG and LPIPS).

| objective | composite | vmaf_neg | lpips | ssimulacra2 | dists | xpsnr | negative non-own cells | mean rank |
|---|---|---|---|---|---|---|---|---|
| composite | **+2.99** | **+1.86** | **+0.0273** | −2.03 | +0.0026 | −0.59 | ssimulacra2, xpsnr | 2.00 |
| vmaf_neg | +2.52 | **+1.97** | +0.0126 | −2.01 | −0.0008 | −0.39 | ssimulacra2, dists, xpsnr | 2.17 |
| ssimulacra2 | +1.31 | +1.10 | +0.0047 | **+0.54** | +0.0011 | −0.24 | xpsnr | 2.33 |
| lpips | +0.92 | −1.38 | **+0.0556** | −14.89 | −0.0065 | −1.52 | vmaf_neg, ssimulacra2, dists, xpsnr | 3.50 |

Best θ per row: composite ×2, contrast 1.06, saturation 1.11, unsharp 0.49, grain 9.1;
vmaf_neg ×2, contrast 1.05, saturation 0.94, unsharp 0.62, grain 15; ssimulacra2 ×2, contrast
1.02, saturation 1.04, no unsharp, grain 3.8; lpips ×2, contrast 1.11, saturation 1.25, unsharp
1.68, grain 14.1.

testsrc, same cells: composite row +0.53 / ssimulacra2 +2.12 / xpsnr −6.99; ssimulacra2 row
composite −10.30 / ssimulacra2 −2.37 / xpsnr −4.31; lpips row xpsnr −11.80.

* **No objective satisfies the decision rule.** Every row lowers at least one judge it did not
  optimise. `ssimulacra2` comes closest, with one small negative (XPSNR −0.24), but its gain on
  its own metric is small (+0.54). `composite` has the best mean rank, but that rank is helped by
  the three columns it optimises.
* **Default objective: `composite` is kept**, because the rule gave no winner, and the CLI default
  is unchanged. A filter gain found with any of these objectives must be checked with the judges
  (`vreg rescore`) before it is reported.
* **Optimising LPIPS alone is a textbook Goodhart case.** It maxes out saturation, sharpening and
  grain, and it is the worst row on every other judge (SSIMULACRA2 −14.9).
* Re-running `strict_gp` with the same seed (`strict_gp_obj_composite`) did not replay it bit for
  bit: best 43.72 vs 43.80, with a similar recipe. Optuna's GP sampler is not deterministic, so
  run-to-run noise exists even at a fixed seed.

### E1: seeds and variance

Samplers on `ugc_strict`, composite, 40 trials, two training clips. Five seeds: seed 0 is the
original run (`strict_<sampler>`), seeds 1–4 are `strict_<sampler>_s<n>`.

![convergence across seeds](docs/img/p0/convergence.png)

| sampler | seeds | final best (mean ± sd) | trial reaching 95 % of final gain (mean ± sd) |
|---|---|---|---|
| GP-BO | 5 | **43.79 ± 0.02** | 17.4 ± 8.0 |
| TPE | 5 | 43.43 ± 0.35 | 16.2 ± 5.5 |
| CMA-ES | 5 | 40.71 ± 3.82 | 22.4 ± 6.6 |
| random | 5 | 42.82 ± 0.51 | 15.6 ± 10.3 |

By the rule (A beats B only if mean(A) − mean(B) > max(sd(A), sd(B))): **GP-BO beats TPE (+0.36)
and random (+0.96), and TPE beats random (+0.61).** No sampler beats CMA-ES or loses to it by the
rule, because its sd is so large.

* **GP-BO is the best and by far the most consistent sampler** (sd 0.02 over five seeds). The
  earlier claim "with one seed, the samplers cannot be told apart" is replaced by this ranking:
  GP-BO > TPE > random.
* **CMA-ES is unreliable at this budget.** On two of five seeds (`s2`, `s3`) it never left the
  ×1.5 tier and ended at about 36.5. `upscale` is a continuous knob snapped to {1, 1.5, 2}, and
  CMA-ES starts in the middle of the box with a small step size. With a population of 8, 40
  trials are only five generations, too few to cross into the ×2 tier on some seeds.
* **The best θ is not identifiable, even though the objective is.** The five GP-BO runs reach
  43.77–43.80, but their recipes differ:
  * grain ranges from 0 to 15;
  * unsharp ranges from 0.42 (5×5) to 1.23 (3×3);
  * saturation sits at the upper bound of 1.25 in four of five runs.

  All runs agree on ×2 and +4–6 % contrast. The objective is flat along grain and sharpening,
  so the earlier "all samplers converge on the same recipe" (grain ≈ 6–7, unsharp 0.4–0.8) was
  one point on a ridge, not a stable optimum.

**E2 across seeds.** The four extra GP-BO seeds were re-scored on every clip (`vreg rescore
runs/strict_gp_s<n>`):

| GP-BO seed | composite | ssimulacra2 | dists | xpsnr | verdict |
|---|---|---|---|---|---|
| 0 (`strict_gp`) | +3.03 | −2.23 (− − − −) | +0.0003 | −0.75 | not confirmed (1/3) |
| 1 | +3.22 | −3.97 (− − − −) | −0.0017 | −0.55 | not confirmed (0/3) |
| 2 | +3.19 | −2.88 (− − − −) | −0.0017 | −0.30 | not confirmed (0/3) |
| 3 | +2.98 | −2.64 (− − − −) | −0.0006 | −0.01 | not confirmed (0/3) |
| 4 | +3.08 | −2.70 (− − − −) | −0.0015 | −0.35 | not confirmed (0/3) |

The filter gain over tier-only fails the rule on **every seed**. SSIMULACRA2 is lower than
tier-only on every natural clip in every seed. Seed 0 was the most favourable run for the claim,
not the least.

**Learned `G` across seeds.** `vreg surrogate --rounds 1` with seeds 0, 1 and 2
(`surrogate_trust_s<n>`), the same settings as `surrogate_trust`. Every seed passes the E2 rule
2/3 (DISTS and XPSNR). SSIMULACRA2 is lower than tier-only on every natural clip in every seed
(mean −5.2 to −6.9).

Real composite and SSIMULACRA2 per clip, mean ± sd over seeds (`G`: 3 seeds, round 0; GP-BO θ:
5 seeds):

| clip | tier-only | GP-BO θ composite | `G` composite | GP-BO θ ssimulacra2 | `G` ssimulacra2 |
|---|---|---|---|---|---|
| park_joy *(train)* | 22.21 | **24.95 ± 0.17** | 22.75 ± 0.21 | **−23.57 ± 0.94** | −27.11 ± 0.46 |
| old_town_cross *(train)* | 60.40 | 62.62 ± 0.16 | **65.39 ± 0.87** | 51.48 ± 0.78 | 50.87 ± 0.96 |
| ducks_take_off *(train for `G`)* | 25.97 | **29.09 ± 0.27** | 25.45 ± 0.38 | **−23.29 ± 1.10** | −26.99 ± 0.98 |
| in_to_tree *(held out)* | 58.28 | 62.59 ± 0.40 | 62.32 ± 0.81 | **42.73 ± 0.66** | 38.31 ± 1.54 |
| sky_gradient *(held out)* | 93.07 | **92.87 ± 0.29** | 82.99 ± 1.56 | **86.18 ± 0.95** | 66.91 ± 4.47 |

Bold: the winner by the pairwise rule; no bold means a tie.

* **`G` beats GP-BO θ only on old_town_cross, and only on the objective** (+2.8). On SSIMULACRA2
  the two tie there.
* On the held-out in_to_tree, `G` and θ tie on composite, and θ wins on SSIMULACRA2. The earlier
  claim "`G` beats θ on smoother content (old_town_cross, in_to_tree)" holds only for
  old_town_cross.
* **`G` does not beat tier-only on every natural clip.** On ducks_take_off it is below tier-only
  in all three new seeds (25.45 ± 0.38 vs 25.97), although ducks is one of its training clips.
* `G` damages the held-out synthetic gradient much more than θ does (composite −10.1 vs
  tier-only, SSIMULACRA2 −24).

### Verdicts

| # | Claim | Verdict |
|---|---|---|
| E2 | Tier gain over naive (ugc_strict) | **stands**: all three judges, all natural clips |
| E2 | Filter gain over tier-only, ugc_strict GP-BO θ (+2–4) | **not confirmed**: SSIMULACRA2 and XPSNR lower on every natural clip |
| E2 | Filter gain over tier-only, ugc_av1 GP-BO θ | stands by the rule, but carried by one clip (park_joy) |
| E2 | Learned `G` (trust region) beats tier-only | stands by the rule (DISTS, XPSNR); SSIMULACRA2 disagrees on every clip |
| E3 | Part of the filter gain is colour/contrast hacking | **confirmed**: freezing `eq` costs 28 % of the composite gain and makes the result pass 2/3 judges |
| E5 | Filters help on their own, without the tier | **not confirmed**: at ×1 the gain is contrast only, 0/3 judges |
| E4 | `composite` is a safe default objective | **not confirmed**: no objective is free of judge regressions; default kept, results must be re-scored |
| E1 | Samplers cannot be told apart | **replaced**: over 5 seeds, GP-BO > TPE > random by the rule; CMA-ES misses the ×2 tier on 2/5 seeds |
| E1 | All samplers converge on the same recipe | **downgraded**: tier (×2) and contrast (+4–6 %) are stable; grain (0–15) and sharpening vary freely at the same score |
| E1 + E2 | Filter gain over tier-only, ugc_strict, across seeds | **not confirmed** on all 5 GP-BO seeds (0–1/3 judges); SSIMULACRA2 lower on every clip in every seed |
| E1 | Learned `G` beats tier-only on all natural clips | **downgraded**: 3/4 clips; below tier-only on ducks_take_off in all 3 new seeds |
| E1 | Learned `G` beats GP-BO θ on smoother content | **downgraded**: only old_town_cross, only on composite; tie on in_to_tree, where θ wins on SSIMULACRA2 |
