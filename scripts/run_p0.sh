#!/usr/bin/env bash
# P0 experiments (docs/P0-plan.md), in the planned order: E2, E3, E5, E4, E1.
# Resumable: optimize/surrogate runs continue under the same --name, and transfer/rescore reuse the
# evaluation cache. A failed step is logged and the script moves on.
# Usage: scripts/run_p0.sh [e2|e3|e5|e4|e1|report|e1s|e1x ...]   (default: all, in order)
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
CLIPS=old_town_cross,park_joy
OPT=(--platform sim:ugc_strict --clips "$CLIPS" --trials 40)

step() { echo "=== $(date '+%F %T') $*"; "$@" || echo "!!! FAILED ($?): $*"; }

e2() { step uv run vreg rescore runs/strict_gp runs/av1_gp runs/surrogate_trust; }

e3() {
  step uv run vreg optimize "${OPT[@]}" --sampler gp --fix contrast=1,saturation=1 --name strict_gp_noeq
  step uv run vreg transfer runs/strict_gp_noeq --clips all
  step uv run vreg rescore runs/strict_gp_noeq
}

e5() {
  step uv run vreg optimize "${OPT[@]}" --sampler gp --fix upscale=1 --name strict_gp_native
  step uv run vreg transfer runs/strict_gp_native --clips all
  step uv run vreg rescore runs/strict_gp_native
}

e4() {
  for obj in composite vmaf_neg ssimulacra2 lpips; do
    step uv run vreg optimize "${OPT[@]}" --sampler gp --objective "$obj" --name "strict_gp_obj_${obj}"
    step uv run vreg transfer "runs/strict_gp_obj_${obj}" --clips all
    step uv run vreg rescore "runs/strict_gp_obj_${obj}"
  done
}

e1() {
  for s in 1 2 3 4; do
    for smp in gp tpe cmaes random; do
      step uv run vreg optimize "${OPT[@]}" --sampler "$smp" --seed "$s" --name "strict_${smp}_s${s}"
    done
  done
}

e1s() {
  for s in 0 1 2; do
    step uv run vreg surrogate --clips park_joy,old_town_cross,ducks_take_off \
      --holdout in_to_tree,sky_gradient --rounds 1 --seed "$s" --name "surrogate_trust_s${s}"
    step uv run vreg rescore "runs/surrogate_trust_s${s}"
  done
}

# GP-BO theta on every clip for seeds 1-4 (seed 0 is strict_gp): G vs theta and E2 across seeds
e1x() {
  for s in 1 2 3 4; do step uv run vreg rescore "runs/strict_gp_s${s}"; done
}

report() {
  step uv run vreg report runs/strict_{gp,tpe,cmaes,random} runs/strict_*_s[1-4] \
    --out runs/report_seeds.html --title "Sampler comparison, 5 seeds" --png-dir docs/img/p0
}

for stage in "${@:-e2 e3 e5 e4 e1 report e1s e1x}"; do
  for s in $stage; do "$s"; done
done
echo "=== $(date '+%F %T') P0 driver done"
