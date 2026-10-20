# E9 plan: a learned `G` on top of the class-tuned sharpening θ

**Why.** E7 ([PROGRESS §9](../PROGRESS.md#9-e7-results-a-real-anchored-estimator-for-g)) showed
that anchoring `G`'s forward value on the black box closes about 80 % of the gap to the class-tuned
sharpening-only θ on class A. What remains is a trade-off: θ wins on XPSNR, E7's `G` on DISTS.
By the E7 plan's rule ("helps"), E9 gets one overnight run with the E7 estimator. E9 asks whether
starting from θ, rather than competing with it, lets `G` keep θ's gain and add its own. The goal
is completeness for the paper; the learned route is not a candidate for the online-platform budget
(E7: 312 black-box calls).

"Black box" means the simulated `sim:ugc_strict`. No online platform is involved.

## Design: θ becomes part of the channel

The class A θ (`runs/p1_sharp_A`: ×2 lanczos upscale, then unsharp 1.06, 5×5) is moved into the
channel `C = platform ∘ θ_A`. `G` works at the source resolution in front of it, so the upload is
`θ_A(G(X))`.

* **At `G = identity` the output is exactly θ_A's.** `G`'s initial state is the baseline, and any
  change is what `G` adds.
* **A new emulator `E_θ ≈ C` is needed.** The P1/E7 emulators model the upscale-only channel, so
  their gradients would ignore the sharpening.
  * It is trained as in P1: 26 `@natural:train` clips × 6 random variants, sent through `C`. That
    is 156 black-box calls, shared by the three seeds through the pairs cache.
  * Then 3000 steps, one emulator per seed.
* **`G` is trained with the E7 estimator,** exactly as E7a:
  * 6 anchors (156 calls) on the frozen `E_θ`;
  * `@natural:train`, loss `lpips`, trust weight 2.0, residual cap 0.06;
  * 1500 steps, seeds 0, 1 and 2.
* **Evaluation:** every dev clip, plus the class A train clips for the train/dev diagnostic. As
  in P1, θ_A is applied in front of `G` on every clip. The gate would use it on class A only.

Budget per the three seeds: 156 (pairs) + 3 × 156 (anchors) black-box calls in training.

## Decision rules (class A dev, before any confirm evaluation)

E9 − θ_A on the 7 class A dev clips, judged by SSIMULACRA2, DISTS and XPSNR. A clip's value is
the mean over the 3 seeds; CIs are 95 % bootstrap over clips.

1. **E9 beats θ** if it meets the H1 criteria of P1:
   * ≥ 2 judges with the CI above 0;
   * SSIMULACRA2 mean ≥ 0;
   * the sign holds in ≥ 2 of 3 seeds.

   Then, and only then, E9 is evaluated **once** on `@A:confirm` with the same rules. The confirm
   clips were used once each for P1's `G` and never for E9.
2. **E9 adds to θ** if it does not beat θ by rule 1, but no judge's CI is below 0 and at least one
   judge's CI is above 0.
3. **E9 does not add** otherwise.

**Consequence, fixed now:**
* "beats": the paper reports the learned route as an improvement over tuned sharpening, with its
  black-box cost.
* "adds": it is reported as a non-inferior refinement that is not confirmed.
* "does not add": the learned-`G` line is closed. P0 + P1 + E7 + E9 give the negative result
  together with its explanation.

## Then, unattended: the rest of E14

After E9, `scripts/run_e9.sh` starts the other half of E14 on its own (`run_p1.sh e14r`):
* the `ugc_av1`-tuned sharpening θ per class (`runs/p1av1_sharp_{A,B,D}`) on `ugc_strict`;
* P0's full GP-BO θ in both directions, on the P1 clips:
  * `strict_gp` on `ugc_av1`;
  * `av1_gp` on `ugc_strict`.

Both use the frozen classes and splits. They are exploratory and change no verdict.
