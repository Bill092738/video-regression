# E10 plan (reframed): how many clips does a per-platform validation need?

**Why the reframe.** The backlog's E10 asked how many trials a tuner needs on a small real budget.
E11 showed that on these platforms tuning adds nothing over a fixed recipe: ffmpeg's default
unsharp matches the class-tuned θ, and a mild unsharp (5×5, 0.5) on class A matches the P1 gate.
§8 showed that the recipe can still fail on another platform (rate-starved AV1 clips). So what an
online platform needs is not a tuner but a **validation**: upload a few clips as tier-only and as
recipe, then decide whether to switch the recipe on. E10 asks how few clips that decision needs.
It answers the upload budget of E15/E16 directly.

No online platform is used. The two simulated platforms stand in for "a new platform".

## Data

Per-clip metrics of tier-only (×2, CRF 10), `r_mild` and `r_default` (E11's recipes):
* on `sim:ugc_strict`: from E11;
* on `sim:ugc_av1`: run now with `scripts/e11.py run --platform sim:ugc_av1`, about 2 × 76
  evaluations.

The clips are the tested P1 classes A, B and D, all splits (24–27 clips per class). C is excluded.

## Decision being validated

For a platform, a class and a recipe: **switch the recipe on** iff its mean gain over tier-only on
the validation clips is ≥ 0 on both SSIMULACRA2 and XPSNR (P1's gate criterion).
* The **truth** is the same rule applied to all of the class's clips on that platform.
* For each validation size `n ∈ {2, 3, 4, 6, 8, 10, 12, 16}`, draw 10,000 clip subsets of size
  `n` without replacement. Record:
  * the error rate (decision ≠ truth);
  * the "harm missed" rate (truth off, decision on).

## Selection strategies (fixed now)

* **random:** `n` clips drawn uniformly from the class.
* **hard-half:** `⌈n/2⌉` clips drawn from the class's top third by SI, the rest from the other
  clips. SI is known before any upload, from `docs/p1/features.csv`. This tests whether
  deliberately including high-detail clips catches rate-starvation failures with fewer uploads.

## Output

* Error rate vs `n` per (platform, class, recipe, strategy).
* The smallest `n` with an error ≤ 10 % and ≤ 5 %.
* The implied upload budget per class and platform: `2n`, i.e. tier-only and recipe for each clip.

E10 is descriptive and changes no earlier verdict. Its caveat is that the truth comes from 24–27
clips, so the error rates are relative to that finite set.
