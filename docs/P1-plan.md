# P1 work plan: content-scoped learned pre-compensation

**Why.** P0 confirmed only the tier gain. The filter gain over tier-only was not confirmed by the
independent judges. The learned `G` beat GP-BO θ only on old_town_cross, and only on the
objective. The only filter that passed two of three judges was sharpening-only θ (P0 E3).

P1 therefore tests a narrower claim. The platform's damage depends systematically on content:
rate control and ingest denoise react to texture, motion and flat areas. A learned `G` may
recover quality beyond the tier trick and beyond simple sharpening **for a defined content
class**, and should be switched off elsewhere.

**Rule that makes this valid.** Classes, hypotheses, splits and decision rules are fixed **before**
any `G` is trained (E19), and the confirmation split is evaluated **once** (E23). Choosing a class
after seeing which clips `G` wins on is not allowed. With the current four natural clips, that
would be selecting a subgroup from one example.

**Scope.** Local workstation, simulated platform `sim:ugc_strict` only, upload tier fixed at ×2
(the tier gain is settled). P1 does not include:
* simulator changes (E12–E14);
* real platforms or the viewer study (P2);
* the "P1-later" method items (E7–E11);
* changes to `composite`.

## Pre-registered hypotheses

| ID | Class | Hypothesis | Role |
|---|---|---|---|
| H1 | **A**: natural, low motion, moderate detail | `G` beats the class-tuned sharpening-only θ | **confirmatory** |
| H2 | **B**: natural, dense texture or high motion | `G` does **not** beat sharpening-only θ, because the emulator is weakest here (~25 dB in P0) | exploratory |
| H3 | **C**: flat, gradient or dark dominated; **D**: graphics, text, screen, animation | Any filter, `G` included, is worse than tier-only, so the gate should switch filters off | exploratory |

The mechanism behind H1: `ugc_strict`'s ingest denoise and capped CRF remove mid-frequency
texture. Where motion is low and detail is moderate, the emulator is most accurate, so `G` can
learn a targeted pre-emphasis that a global unsharp mask cannot.

At most **one** additional class may be nominated as confirmatory after E22, from dev results
only, before E23. If one is nominated, the α of the confirmatory tests is halved (Bonferroni).

## Order

| Step | Item | Rough cost |
|---|---|---|
| 1 | Code changes 1–4 | small–medium |
| 2 | E19: content set, features, classes, splits, then **commit `docs/p1/` before going on** | hours (mostly curation and licenses) |
| 3 | Code changes 5–8 | medium |
| 4 | E20: baselines on every clip | ~3–4 h |
| 5 | E21: `G` loss selection on class A dev | ~3 h |
| 6 | E22: specialist vs global `G`, 3 seeds | ~1–2 nights |
| 7 | E23: gate design on dev, then **one** confirmation run | ~2–3 h |

Run `uv run vreg doctor` first. A step that fails is logged and the driver moves on, as in
`scripts/run_p0.sh`.

## Code changes

Keep each change small and covered by a fast test (CI runs `pytest -m "not gpu"`).

1. **Manifest-driven clips.** Add `docs/p1/clips.csv` (committed) with these columns:
   * `name`;
   * `source_group`: all segments cut from one source share a group;
   * `url_or_path`;
   * `start_s`;
   * `source_type`: `natural`, `animation`, `screen` or `synthetic`;
   * `license`;
   * `license_url`.

   Generalise `content.fetch_xiph` to fetch any URL or local path at an offset into the same
   master format (1280×720, 30 fps, 3 s, qp 0). Add `vreg clips --manifest docs/p1/clips.csv`.
   Masters stay in `data/clips/`, which is git-ignored.
2. **`vreg features`.** For every clip, compute the following and write `docs/p1/features.csv`:
   * SI and TI (ITU-T P.910, ffmpeg `siti` filter);
   * `flat_frac`: the fraction of 16×16 luma blocks with std < 2;
   * `dark_frac`: the fraction of luma below 40;
   * mean luma.
3. **`vreg classify`.** From the manifest and features, write `docs/p1/classes.json` with the
   thresholds, each clip's class and each clip's split. Rules, applied in this order:
   * **D** if `source_type` ≠ `natural`;
   * **C** if `flat_frac ≥ 0.5` or `dark_frac ≥ 0.5`;
   * **B** if SI ≥ the 66th percentile or TI ≥ the 66th percentile, where the percentiles are
     taken over the natural clips not in C;
   * **A** otherwise.

   Splits are made per class, **grouped by `source_group`**: 50 % train, 25 % dev, 25 % confirm,
   with a fixed seed. Do not adjust the thresholds to move particular clips between classes.
4. **Clip selectors.** Wherever `--clips` or `--holdout` is accepted, also accept selectors such
   as `@A:train`, `@B:dev`, `@*:dev` or `@natural:train`, resolved from `classes.json`.
5. **`G` loss option for `vreg surrogate`**: `--g-loss lpips|dists|lpips+dists`. The default
   stays `lpips`. DISTS comes from `piq` and is differentiable. Record the loss in the
   `settings` of `history.json`.
6. **Judges exclude the training loss.** In `vreg rescore` and the P1 tables, a `G` trained with
   DISTS is not judged by DISTS. Its judges are then SSIMULACRA2, XPSNR and VMAF NEG (VMAF NEG is
   not in `G`'s loss).
7. **Reuse round-0 pairs.** Add `--pairs-cache DIR`. Seed the random pre-compensation variants
   by (clip, variant index), not by the run seed, so that seeds and specialists reuse the same
   collected pairs instead of sending them through the platform again.
8. **`scripts/run_p1.sh`** (resumable driver, same style as `run_p0.sh`) and
   **`scripts/p1_tables.py`**. The tables script produces:
   * per-class tables;
   * bootstrap 95 % CIs **over clips** (10,000 resamples), where each clip's value is the mean
     over seeds;
   * the gate evaluation of E23.

## Content set (E19)

Targets:
* at least **12 clips** each in classes A and B;
* at least **8 clips** each in classes C and D.

The class is assigned by the rules above, so collect generously (about 60 segments). Each clip
is a 3 s segment; several segments may come from one source, sharing a `source_group`. Expect a
few GB of masters on disk.

Candidate sources. Check the license **of every clip** and record it in the manifest. Skip a
clip if its terms are unclear.
* Xiph `derf` collection (media.xiph.org/video/derf/): natural sequences, each with its own
  copyright file. The SVT clips fall under the EBU/SVT conditions.
* Blender Foundation open movies (e.g. Big Buck Bunny, Sintel, Tears of Steel; CC BY): animation,
  plus some live action.
* Netflix Open Content: check the license per title.
* The YouTube-UGC dataset: check its license terms before use.
* Your own recordings (phone video, screen captures of your own desktop): natural UGC and
  screen content.
* The existing lavfi synthetic clips.

Never download platform content that you do not own or have a license for.

**Commit `docs/p1/clips.csv`, `features.csv` and `classes.json` before E20.** Any later change to
classes or splits must be noted in PROGRESS. Such a change invalidates the confirm split for the
classes it touches.

## Experiments

Unless stated otherwise:
* platform `sim:ugc_strict`;
* upload ×2;
* every run re-scored with `vreg rescore`;
* judges SSIMULACRA2, DISTS and XPSNR (minus `G`'s own loss, per code change 6).

### E20: baselines on every clip

For every clip in every split, evaluate:
* naive;
* tier-only (×2);
* P0's global sharpening-only θ, taken from the best θ of `strict_gp_noeq`;
* **class-tuned sharpening-only θ**: for each class, GP-BO with 15 trials on that class's
  **train** clips, using `--fix contrast=1,saturation=1,grain=0,denoise=0,upscale=2`, so the
  search is over `sharpen` and `sharpen_radius` only.

Output: per class, the mean gain of each configuration over tier-only on every metric, with CIs.
The class-tuned sharpening-only θ is the **baseline `G` must beat**.

### E21: `G` loss selection (class A dev only)

Train a global `G` on `@natural:train`, with rounds 1, seed 0, and the current trust region and
step settings. Do this for each `--g-loss` (`lpips`, `dists`, `lpips+dists`). Evaluate on `@A:dev`.

Pick the loss that gives the most judges with a positive mean gain over the class-tuned
sharpening-only θ, with SSIMULACRA2 as the tie-break. The loss is fixed from here on.

### E22: specialist vs global `G`

Three seeds each, rounds 1, the loss from E21, pairs reused through `--pairs-cache`:
* **global** `G`: trained on `@natural:train`;
* **specialist A**: trained on `@A:train`;
* **specialist B**: trained on `@B:train` (tests H2).

Evaluate each on `@*:dev`. Report, per class on dev, the gain over tier-only and over the
class-tuned sharpening-only θ, as mean ± CI over clips.

After this step, and before E23, write down:
* which `G` (global or specialist) is used for class A;
* whether one additional class is nominated as confirmatory (dev results only);
* the gate table: for each class, which configuration to use (`G`, sharpening-only θ, or
  tier-only). The table is chosen on dev.

### E23: gate and confirmation (run once)

Evaluate on `@*:confirm`, with the `G` and gate table chosen in E22. No retraining and no
threshold changes after seeing these numbers.

## Decision rules

**H1 (and the nominated class, if any) is confirmed** on the confirm split only if all of the
following hold:
1. at least 2 of the judges show a mean gain of `G` over the class-tuned sharpening-only θ whose
   95 % bootstrap CI over clips excludes 0 (use 97.5 % if a second class was nominated);
2. SSIMULACRA2 mean gain ≥ 0, since it disagreed on every clip in P0;
3. the sign of the result holds for at least 2 of the 3 seeds.

**The gate is accepted** if, on the confirm split:
* every class's mean SSIMULACRA2 and XPSNR with the gate is ≥ tier-only;
* the overall mean over all clips with the gate is ≥ the global sharpening-only θ on at least 2
  judges.

**If H1 fails**, report the negative result: on this black box, the gain an uploader can get
beyond the tier trick comes from simple sharpening, not from a learned `G`. Do not search for a
new class on the confirm split.

## Deliverables

1. Code changes 1–8 with tests; `uv run pytest -q` passes.
2. `docs/p1/clips.csv`, `features.csv` and `classes.json`, committed before E20.
3. A section "P1 results" appended to `PROGRESS.md`, containing:
   * the class sizes;
   * the E20 baseline table;
   * the E21 loss choice;
   * the E22 dev tables and the pre-E23 decisions (written before E23);
   * the E23 confirm table;
   * the gate evaluation;
   * one verdict per hypothesis.
4. README corrections in "Results" and "What the results show so far", only where a verdict
   changes them.
5. Figures in `docs/img/p1/` (PNGs only; `runs/` and media stay local).
6. In the PROGRESS backlog, mark E19–E23 as done.
