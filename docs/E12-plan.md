# E12 plan: a passthrough rule below a bitrate cap, with the upload bitrate as a parameter

**Why.** E11 found that the upload bitrate matters more than any filter. Some platforms are
reported (community claims, not verified) to skip their own re-encode when an upload is already
small enough, and to serve the uploader's file as is. That turns the upload bitrate into a
parameter with a cliff. E12 tests whether the closed-loop black-box search, the core method of
this project, discovers such a hidden rule by itself, and whether the best strategy depends on
content.

No online platform is used. The rule is invented for the simulator; E12 is about the method's
ability to find discontinuities, not about any real platform.

## Simulated platform `sim:ugc_passthrough`

`ugc_strict`'s ladder, plus one rule:
* an upload that is **H.264, at most 720p** (the served height), with an average bitrate
  **≤ 2000 kbps**, is served unchanged (remuxed), with no ingest denoise and no re-encode;
* everything else goes through the ladder as before (≥ 1440p: HEVC at 2000 kbps after denoise;
  720p: x264 at 1200 kbps after strong denoise).

The cap equals the top tier's bit budget, so the two routes spend the same bits.

## Search

GP-BO (Optuna GP sampler, 8 random start-up trials), **30 trials, seeds 0 and 1**, objective
`composite` (as in P0). Three parameters:
* upscale ∈ [1, 2], snapped to {1, 1.5, 2};
* unsharp amount ∈ [0, 2] (5×5);
* upload bitrate ∈ [300, 80,000] kbps, log scale, rounded to 2 significant digits.

Trial 0 is the P1 default: tier-only, ×2, ≈ near-lossless at 80 Mbps. The training clips are 3
class A and 3 class B train clips (the first three of each in `classes.json`). Upload encoder:
x264 veryfast at the fixed bitrate (as in E11).

## Evaluation (dev clips of A and B, with the judges)

* the best θ of each seed;
* `tier` (×2, CRF 10) and `r_mild` (×2 + unsharp 0.5, CRF 10), from E11;
* `pass` (×1 at 1900 kbps, under the cap) and `pass_mild` (the same + unsharp 0.5);
* `over` (×1 at 2500 kbps, over the cap): shows the cliff (see the amendment below);
* `tier_16m` (×2 at 16 Mbps).

## Questions

1. **Discovery:** does each seed's best θ lie in the passthrough region (×1, ≤ 2000 kbps)? After
   how many trials does a passthrough trial first come within 1 % of the final best?
2. **Content dependence:** `pass` and `pass_mild` vs tier-only per class (A, B), on the judges.
   Expectation: they win on A (low motion fits in 2 Mbps without a second encode), and on B they
   may not.
3. **The cliff:** `pass` − `over`.

Descriptive; it changes no earlier verdict.

## Amendment (before any E12 run)

A smoke test on vidyo1_2 showed that x264 undershoots its target on easy content. A 2100 kbps
upload came out at 1888 kbps, so the platform served it unchanged. Forcing CBR (`nal-hrd=cbr`)
still gave 1989 kbps in MP4. Therefore:
* `over` targets 2500 kbps instead of 2100;
* the tables decide "served unchanged" vs "re-encoded" from each upload's **actual** average
  bitrate (≤ 2000 kbps at ≤ 720p), clip by clip, and report how many clips took each route.

The search is unchanged: the black box always sees the actual bitrate.
