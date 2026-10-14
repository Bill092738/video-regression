# E7 plan: a real-anchored estimator for the learned `G`

**Why.** P1 found that the learned `G` loses to the class-tuned sharpening-only θ on class A, and
a post-P1 diagnostic showed it loses **even on its own training clips**, by more than on unseen
clips, and even on LPIPS, its own loss ([PROGRESS §7](../PROGRESS.md#7-p1-results)). That is not
overfitting: the gain `G` finds through the emulator `E` does not survive the black box. E7
changes only **where the forward value comes from**, and keeps everything else of P1 fixed.

"Black box" and "real" below mean `f_platform`, which is still the simulated `sim:ugc_strict`.
No online platform is involved.

## Estimator

Training runs in `A` anchor phases. At the start of phase `k`, the current `G_k` is applied to
every training clip, and the result goes once through the black box, giving `y_real_k`. The
offset `δ_k = y_real_k − E(G_k(X))` is stored, at the training frames. During phase `k` the
loss uses

    ŷ = E(G(X)) + stopgrad(δ_k)

* At the anchor (`G = G_k`), the forward value is exactly the black box's output.
* Away from it, it is the anchor's real output plus the emulator's predicted change.
* The gradient is the emulator's.

This is the straight-through form `y_real + E(x) − stopgrad(E(x))` of the backlog item, with the
real output refreshed once per phase instead of every step. A per-step real forward would cost
one black-box call per clip and step, far beyond any upload budget.

## What is held fixed (a clean ablation of the estimator)

* `E` is **loaded from the P1 run of the same seed** (`runs/p1_g_lpips_s{0,1,2}/round0.pt`), frozen.
* Everything else is the same as the P1 global `G`:
  * training clips `@natural:train` (26);
  * loss `lpips` + L1, trust weight 2.0, residual cap 0.06;
  * 1500 generator steps, learning rate 1e-4, seeds 0, 1 and 2;
  * evaluation on all 19 dev clips (and the training clips, for the train/dev diagnostic).
* Classes, splits, judges (SSIMULACRA2, DISTS, XPSNR) and the bootstrap CIs over clips are those
  of P1.

## Runs

| run | anchors `A` | phase length | black-box calls in training |
|---|---|---|---|
| **E7a** (main) | 6 | 250 steps | 6 × 26 = 156 |
| **E7b** (low budget, exploratory) | 1 (at `G = identity`) | 1500 steps | 26 |

Both are inherited `E`'s training on top: 26 clips × 6 random variants = 156 calls, shared with
P1. Every run records its black-box calls (training and evaluation separately). For comparison,
the class-tuned θ for A used 15 trials × 13 clips = 195 calls.

## Decision rules (dev split, before any confirm evaluation)

`G` vs the class-tuned sharpening-only θ on the 7 class A dev clips, judged by SSIMULACRA2,
DISTS and XPSNR. A clip's value is the mean over the 3 seeds.

1. **E7 fixes the gap** if E7a meets the H1 criteria of P1 on A dev:
   * ≥ 2 judges whose 95 % CI excludes 0 (positive);
   * SSIMULACRA2 mean gain ≥ 0;
   * the sign holds in ≥ 2 of 3 seeds.

   Then, and only then, E7a is evaluated **once** on `@A:confirm` with the same rules. The
   confirm clips were used once in P1 for the P1 `G`; E7's `G` is new, but this reuse is noted.
2. **E7 helps** if it does not fix the gap, but E7a beats the P1 global `G` (same `E`, same
   seeds) on ≥ 2 judges, with a paired 95 % CI over A dev clips excluding 0.
3. **E7 does not help** otherwise.

**Consequence for E9** (`G` on top of θ), fixed now:
* "fixes": E9 is worth running, with the E7 estimator.
* "helps": E9 is worth one overnight run, with the E7 estimator, to see whether starting at θ
  closes the rest of the gap.
* "does not help": E9 is not run. The learned-`G` line stops at a negative result, now with the
  emulator-anchoring explanation tested.

**Diagnostic, reported in every case:** E7a and E7b vs θ on the training clips, against the P1
`G`'s train-clip numbers.

## Deliverables

* `vreg surrogate --estimator anchored --anchors A --emulator-from RUN` with a fast test.
* `scripts/run_e7.sh` and `scripts/e7_tables.py`.
* A PROGRESS section "E7 results", with the budget in black-box calls, and the backlog updated.
