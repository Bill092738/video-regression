# E11 plan: common creator recipes as a baseline

**Why.** P1's accepted gate sharpens class A (natural, low motion) with a class-tuned θ and uses
tier-only elsewhere. That θ came from 15 GP-BO trials on 13 clips: 195 black-box calls, i.e.
uploads on an online platform. Common creator advice needs no tuning at all: upload at 1440p,
add a little sharpening, use the platform's recommended bitrate. E11 measures what the tuning adds
over that advice, and what the recommended upload bitrate costs compared with a near-lossless
upload. If an untuned recipe matches θ, the practical recipe needs **zero** tuning uploads.

Black box: `sim:ugc_strict` (no online platform). Classes, splits, judges and bootstrap CIs are
those of P1.

## Recipes (fixed now, no tuning)

| id | recipe | upload |
|---|---|---|
| `tier` | ×2 lanczos (1440p) | near-lossless x264, CRF 10 (median ≈ 76 Mbps; the P1 default) |
| `r_default` | ×2 + ffmpeg's default unsharp, luma 5×5, amount 1.0 | CRF 10 |
| `r_mild` | ×2 + mild unsharp, luma 5×5, amount 0.5 | CRF 10 |
| `r_mild_16m` | `r_mild` | x264 at 16 Mbps (YouTube's recommended SDR 1440p, 24–30 fps) |
| `tier_16m` | `tier` | 16 Mbps |

The 16 Mbps uploads use x264 veryfast, `-b:v 16M -maxrate 16M -bufsize 32M`.

## Questions and rules

All gains are signed so that positive = better, with 95 % bootstrap CIs over clips.

1. **Q1, does tuning add?** Class-tuned θ − `r_default` and θ − `r_mild`, per class, on the dev
   and confirm clips (θ never saw them). Tuning "adds" in a class if ≥ 2 judges have a CI above 0
   and none has a CI below 0. If neither recipe is beaten in class A, the untuned recipe can
   replace θ in the gate, with no tuning uploads.
2. **Q2, recommended bitrate:** `r_mild_16m` − `r_mild` and `tier_16m` − `tier`, per class, on
   every tested clip. This reports the quality cost of uploading at the recommended rate instead of
   near-lossless.
3. **Gate with a recipe:** the P1 gate table, with the best recipe on class A, re-checked on the
   confirm clips with P1's acceptance criteria. This is exploratory: the confirm clips were used in
   P1.

E11 is descriptive. It changes no P1 verdict.

## Deliverables

* `precomp.apply(…, upload_kbps=…)` and `evaluate(…, upload_kbps=…)`, with their own cache keys,
  and a fast test.
* `scripts/e11.py`: runs the recipes on every tested clip, with the judges, and writes the tables
  to `runs/e11_tables.md` and figures to `docs/img/e11/`.
* A PROGRESS section, and the backlog updated.
