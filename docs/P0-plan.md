# P0 work plan: make the current claims solid

**Scope.** Only the five P0 experiments from `PROGRESS.md` §4 (E1–E5). Everything runs on the local
workstation against the **simulated** platforms (`sim:ugc_strict`, and `sim:ugc_av1` where noted).
No real platform, no uploads.

**Not in scope.** P1/P2 items (content-adaptive θ, the surrogate estimator, simulator changes, real
platforms, viewer study), refactors unrelated to P0, and changes to the existing pre-compensation
space or to the simulated platform profiles. Do not change `composite` itself; P0 only measures it.

**Goal.** Decide whether these claims hold:
* the filter gain over tier-only (+2–4 on natural clips) is real, not run-to-run noise, and not
  metric hacking;
* the samplers differ, or they don't.

## Order

| Step | Item | Why this order | Rough cost |
|---|---|---|---|
| 1 | E2: independent metrics + re-score existing configs | Cheapest; answers "is the filter gain hacked?" first | code + minutes of compute |
| 2 | Shared code: `--fix` (parameter freezing) and `--seed` for `surrogate` | Needed by E3, E5, E1 | small |
| 3 | E3: `eq` knobs frozen | | ~20–50 min |
| 4 | E5: native tier, filter-only | | ~20–50 min |
| 5 | E4: Goodhart matrix | Needs E2's metrics as objectives | ~1–3 h |
| 6 | E1: seeds (run overnight) | Most compute | ~7–17 h |

Run `uv run vreg doctor` first, and `uv run vreg clips all` if `data/` is empty.

## Code changes

Keep each change small and covered by a fast test in `tests/` (CI runs `pytest -m "not gpu"`).

1. **New report-only metrics** (`src/vreg/metrics.py`):
   * `ssimulacra2`: higher is better. Use the `ssimulacra2` PyPI package (BSD-3) or libjxl's
     `ssimulacra2` binary. Score a frame subsample, as `lpips_score` does (`step=3`), and pool
     with the mean.
   * `dists`: lower is better, GPU. Use `piq` (Apache-2.0) and add it to the `torch` optional
     extra in `pyproject.toml`.
   * `xpsnr`: higher is better. Use the ffmpeg `xpsnr` filter, which needs ffmpeg ≥ 7.1. If the
     filter is missing, skip it and say so in `vreg doctor`.
   * Add all three to `HIGHER_IS_BETTER` and to the `--objective` choices in `cli.py`.
   * **Do not add them to `composite`.** They are judges, not part of the objective.
   * **Do not use `pyiqa`**: its license is non-commercial.
2. **Back-fill metrics on cache hits** (`src/vreg/evaluate.py`). The cache is keyed by θ only, and
   older `result.json` files lack the new metrics. When a cached result is missing a requested
   metric and `playback.mp4` still exists in the workdir, compute only the missing metrics and
   update `result.json`. Never re-run the platform just to add a metric.
3. **`vreg rescore`** (new CLI command). Given run dirs, re-evaluate a fixed set of configurations
   on all clips with all metrics (back-fill as in change 2) and write
   `<run>/rescore.json`. The configurations are:
   * naive;
   * tier-only (`{"upscale": best.upscale}`);
   * the run's best θ;
   * for a surrogate run, each round's `G` playback, found through the `workdir` entries in
     `history.json`.

   `vreg transfer` already covers the first three for `composite`; reuse its logic.
4. **`--fix name=value,...` for `vreg optimize`** (`optimize.py: suggest`, `cli.py`). Fixed
   parameters are not suggested to the sampler. They are passed as constants, go through
   `precomp.normalize` (so `upscale` still snaps to {1, 1.5, 2}), and are recorded in
   `config.json`. The enqueued identity trial 0 must respect the fixed values.
5. **`--seed` for `vreg surrogate`**. `run_loop` already takes `seed`; only the CLI flag is
   missing.
6. **Multi-seed aggregation in `vreg report`**. When several runs share a sampler and platform and
   differ only in seed, show:
   * the mean ± sd of best-so-far versus trial (a band, not one line per seed);
   * a table: sampler, n seeds, final best mean ± sd, trial at which 95 % of the final gain was
     reached (mean ± sd).

## Experiments

Use distinct `--name`s. An Optuna study is keyed by its directory, so reusing a name resumes the old
run. The evaluation cache under `runs/_cache` is shared and safe to reuse across all runs.

Unless stated otherwise:
* platform `sim:ugc_strict`;
* training clips `old_town_cross,park_joy`;
* GP sampler;
* 40 trials;
* objective `composite`;
* `vreg transfer <run> --clips all` after each run.

### E2: independent metrics (re-score; no new search)

```bash
uv run vreg rescore runs/strict_gp runs/av1_gp runs/surrogate_trust
```

Output: a table per run with clips as rows and naive / tier-only / best θ (/ `G` round 0) as
columns, for each metric: `composite`, `vmaf_neg`, `lpips`, `ssimulacra2`, `dists`, `xpsnr`. Also a
row for the **filter gain**, which is best θ minus tier-only.

**Decision.** On the four natural clips (old_town_cross, park_joy, ducks_take_off, in_to_tree):
* If the filter gain has the same sign as `composite` on at least 2 of the 3 independent metrics
  (`ssimulacra2`, `dists`, `xpsnr`), the claim stands.
* Otherwise it is downgraded to "not confirmed" in the README and PROGRESS.

Do the same check for the learned `G` against tier-only.

### E3: metric-hacking ablation (`eq` frozen)

```bash
uv run vreg optimize --clips old_town_cross,park_joy --sampler gp --trials 40 \
  --fix contrast=1,saturation=1 --name strict_gp_noeq
uv run vreg transfer runs/strict_gp_noeq --clips all
uv run vreg rescore runs/strict_gp_noeq
```

Compare with `strict_gp`. Report what fraction of the filter gain over tier-only survives without
`eq`, on `composite` and on each independent metric. If most of the gain disappears on `composite`
but not on the independent metrics, the colour/contrast part was metric hacking.

### E5: native tier, filter-only

```bash
uv run vreg optimize --clips old_town_cross,park_joy --sampler gp --trials 40 \
  --fix upscale=1 --name strict_gp_native
uv run vreg transfer runs/strict_gp_native --clips all
uv run vreg rescore runs/strict_gp_native
```

Report the filter gain at ×1 (best θ minus naive) next to the filter gain at ×2 (from `strict_gp`).
This tells whether the filters help on their own or only together with the higher tier.

### E4: Goodhart matrix

Optimise each objective, then judge every result with all metrics:

```bash
for obj in composite vmaf_neg ssimulacra2 lpips; do
  uv run vreg optimize --clips old_town_cross,park_joy --sampler gp --trials 40 \
    --objective "$obj" --name "strict_gp_obj_${obj}"
  uv run vreg rescore "runs/strict_gp_obj_${obj}"
done
```

`strict_gp_obj_composite` is the same configuration as `strict_gp`, so it can reuse the cache.

Output: a matrix with the optimised objective as rows and the judge metrics as columns. Each cell
is the gain over tier-only, averaged over the four natural clips; also show testsrc separately.

**Decision.** Pick as the default objective the row that has:
* no negative cell among the judges it was not optimised for;
* the best mean rank across columns.

Write the chosen default into PROGRESS. Change the CLI default only if it differs from
`composite`.

### E1: seeds and variance (overnight)

Seed 0 already exists as `strict_{gp,tpe,cmaes,random}`; add seeds 1–4.

```bash
for s in 1 2 3 4; do
  for smp in gp tpe cmaes random; do
    uv run vreg optimize --clips old_town_cross,park_joy --sampler "$smp" --trials 40 \
      --seed "$s" --name "strict_${smp}_s${s}"
  done
done
uv run vreg report runs/strict_{gp,tpe,cmaes,random} runs/strict_*_s[1-4] \
  --out runs/report_seeds.html --title "Sampler comparison, 5 seeds" --png-dir docs/img/p0
```

If time allows, also run 3 surrogate seeds:

```bash
for s in 0 1 2; do
  uv run vreg surrogate --clips park_joy,old_town_cross,ducks_take_off \
    --holdout in_to_tree,sky_gradient --rounds 1 --seed "$s" --name "surrogate_trust_s${s}"
done
```

**Decision.**
* A sampler "beats" another only if its final-best mean exceeds the other's by more than the
  larger of the two sds. Otherwise report "no difference at 5 seeds".
* Report the learned `G` vs GP-BO θ per clip as mean ± sd over seeds.

If 5 seeds × 4 samplers does not fit in one night, run seeds 1–2 first. The script is resumable:
re-running the same command with the same `--name` continues where it stopped.

## Deliverables

1. Code changes 1–6 with tests; `uv run pytest -q` passes.
2. A results section **appended to `PROGRESS.md`**, named "P0 results". It contains:
   * the E2 table;
   * the E3 and E5 gain fractions;
   * the E4 matrix and the chosen default objective;
   * the E1 sampler table (mean ± sd);
   * one line per experiment saying whether the claim stands, is downgraded, or is not confirmed.
3. README corrections where a claim is downgraded, in the "Results" and "What the results show
   so far" sections. Do not rewrite sections that are unaffected.
4. Figures in `docs/img/p0/`. Commit PNGs only: `runs/` and media are git-ignored and stay
   local.
5. Mark E1–E5 as done in the PROGRESS backlog table.
