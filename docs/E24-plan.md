# E24 plan: per-clip closed-loop feedback vs sharpening

**Why.** The learned-`G` line (P1, E7, E9) closed with a negative result: a `G` trained through an
emulator does not beat class-tuned sharpening. Three properties of that set-up, not of learning as
such, may explain it:

* the emulator is per-frame and trained with L1, so it models the platform as "denoise + blur" and
  cannot see rate or temporal cost; through it, the best `G` is an inverse filter, i.e. sharpening;
* `G`'s loss was perceptual only (the L1 term is numerically negligible), while the judges are
  fidelity metrics;
* `G` changes all three colour channels, while the sharpening θ changes luma only.

E24 tests a correction that avoids all three and needs no emulator and no training: upload the
clip once, measure the playback error, and add part of the **luma** error back to the source.

```
Y0 = platform(front(X))                       probe upload
U  = clip(X + α · luma(X − Y0))               same value added to R, G, B (chroma unchanged)
Y1 = platform(front(U))                       feedback upload
```

The forward value is always the black box's own output. The cost is one extra upload **of the clip
being published**, not hundreds of uploads of training clips.

**Pilot (not evidence).** A scaled check on five pseudo-clips (480×270 pans over still photos with
added noise; ugc_strict ×2 tier scaled by pixel count; PSNR and SSIMULACRA2 only) gave, over
tier-only: mild unsharp +0.97 dB PSNR-Y / +2.06 SSIMULACRA2; luma feedback α = 0.5 +1.67 / +3.28;
both together +2.42 / +4.09. Feedback on RGB instead of luma, α = 1 and further steps lowered
SSIMULACRA2 and raised the playback bitrate by 20–100 %. α = 0.5 and luma-only were chosen on that
pilot and are fixed here. The pilot's rate cap never bound, and its content is not real video, so
E24 is the first real test.

Black box: `sim:ugc_strict` (no online platform). Clips, classes, judges and bootstrap CIs are
those of P1 and E11. Nothing is tuned, so every tested clip is used (train, dev and confirm).

## Conditions (fixed now)

| id | upload | uploads per clip |
|---|---|---|
| `tier` | ×2 lanczos | 1 |
| `r_mild` | ×2 + unsharp luma 5×5, 0.5 (E11's recipe) | 1 |
| `r_default` | ×2 + unsharp luma 5×5, 1.0 (matches the class-tuned θ, E11) | 1 |
| `fb` | feedback α = 0.5 from the `tier` playback, front ×2 | 2 |
| `fb_mild` | feedback α = 0.5 from the `r_mild` playback, front ×2 + mild unsharp | 2 |
| `fb_a1` | as `fb`, α = 1.0 (exploratory: step size) | 2 |
| `fb_it2` | as `fb`, a second step from the `fb` playback (exploratory: iteration) | 3 |

The correction is applied to the 720p source before the front end, as `G` was.

## Hypotheses and rules

Gains are signed so that positive = better, with 95 % bootstrap CIs over clips. "Adds" is E11's
rule: ≥ 2 of the 3 judges (SSIMULACRA2, DISTS, XPSNR) have a CI above 0 and none has a CI below 0.

1. **H1 (confirmatory), class A:** `fb_mild` adds over `r_mild` **and** over `r_default`.
   If so, a closed-loop correction beats sharpening where the learned `G` did not.
2. **H1b (secondary), class A:** `fb` (no sharpening at all) adds over `r_mild`.
3. **H2, rate headroom:** on natural clips whose tier-only playback already uses ≥ 90 % of the
   tier's rate cap, feedback does not beat tier-only (the E14 mechanism: extra detail costs more
   bits than it restores). Reported as feedback − tier-only for capped and uncapped clips.
4. **Exploratory:** classes B and D; `fb_a1` and `fb_it2`; the playback bitrate of every condition
   relative to tier-only (how much of each gain is bought with platform bits).

Outcomes:

* **H1 holds:** the follow-up is E25, distilling the correction into a network trained by plain
  regression on `α · luma(X − platform(X))`. Its training pairs are the cached tier-only
  playbacks, so it needs no new black-box calls and no emulator in the loop, and it costs no extra
  upload at use. The real-platform protocol (E15/E16) gains two uploads per clip for `fb_mild`.
* **H1 fails:** the headroom above sharpening is not reachable by a fidelity-driven per-clip
  correction either, which strengthens the P1 conclusion. No E25.

E24 changes no earlier verdict.

## Deliverables

* `src/vreg/feedback.py`: `correct` and `evaluate_feedback` (cached per clip, front end, α and
  probes), with a fast test.
* `scripts/e24.py`: `run` (resumable; baselines reuse E11's cache) and `tables`
  (→ `runs/e24_tables.md`).
* A PROGRESS section once it has run.

## Running it

```bash
uv run python scripts/e24.py run --clips @A:dev     # 7 clips, a quick look (about 30 platform calls)
uv run python scripts/e24.py run                    # all 77 clips, 4 feedback uploads each
uv run python scripts/e24.py tables
```

DISTS and LPIPS need the GPU; without it those columns are skipped.
