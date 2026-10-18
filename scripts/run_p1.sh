#!/usr/bin/env bash
# P1 experiments (docs/P1-plan.md): E19 set-up, E20 baselines, E21 G loss, E22 global vs specialist G,
# E23 confirmation. Resumable: optimize/surrogate runs continue under the same --name, round-0 pairs are
# shared through runs/p1_pairs, and rescore reuses the evaluation cache. A failed step is logged and the
# script moves on.
# Usage: scripts/run_p1.sh [e19|e20|e21|e22|e23|e14|e14r|tables ...]   (default: e20 e21 tables e22 tables)
# E23 is never part of the default: it needs docs/p1/decisions.json, written and committed after E22.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PLAT=(--platform sim:ugc_strict)
SHARP_ONLY=contrast=1,saturation=1,grain=0,denoise=0,upscale=2
PAIRS=runs/p1_pairs
SEEDS="0 1 2"

step() { echo "=== $(date '+%F %T') $*"; "$@" || echo "!!! FAILED ($?): $*"; }

# the loss chosen in E21 (written by scripts/p1_tables.py)
g_loss() { cat runs/p1_g_loss.txt 2>/dev/null || { echo "no runs/p1_g_loss.txt: run e21 and tables first" >&2; return 1; }; }
loss_tag() { echo "${1//+/_}"; }

e19() {
  step uv run python scripts/make_screen_clips.py
  step uv run vreg clips all --manifest docs/p1/clips.csv
  step uv run vreg features
  step uv run vreg classify --report-only C --pre-registered '{"gate": {"C": "tier"}}'
}

# baselines on every clip: naive, tier-only, P0 global sharpening-only theta, class-tuned sharpening-only theta
e20() {
  step uv run vreg rescore runs/strict_gp_noeq --clips '@*:*' --tag p1
  for k in A B D; do  # class C is report-only (one clip): it gets no class-tuned theta
    step uv run vreg optimize "${PLAT[@]}" --clips "@$k:train" --sampler gp --trials 15 --fix "$SHARP_ONLY" \
      --name "p1_sharp_$k"
    step uv run vreg rescore "runs/p1_sharp_$k" --clips "@$k:*" --tag p1
  done
}

g_run() {  # name train-clips seed loss
  step uv run vreg surrogate "${PLAT[@]}" --clips "$2" --holdout "$5" --rounds 1 --seed "$3" --g-loss "$4" \
    --pairs-cache "$PAIRS" --name "$1"
}

# G loss selection: global G on @natural:train, seed 0, judged on @A:dev
e21() {
  for loss in lpips dists lpips+dists; do
    local run="p1_g_$(loss_tag "$loss")_s0"
    g_run "$run" '@natural:train' 0 "$loss" '@A:dev'
    step uv run vreg rescore "runs/$run" --clips '@A:dev' --tag p1
  done
}

# specialist vs global G, 3 seeds, the E21 loss, evaluated on every dev clip
e22() {
  local loss; loss=$(g_loss) || return
  local t; t=$(loss_tag "$loss")
  for s in $SEEDS; do
    if [ "$s" = 0 ]; then  # the E21 run of the chosen loss is the global G of seed 0
      step uv run vreg surrogate-eval "runs/p1_g_${t}_s0" --clips '@*:dev' --tag dev
    else
      g_run "p1_g_${t}_s$s" '@natural:train' "$s" "$loss" '@*:dev'
    fi
    g_run "p1_specA_${t}_s$s" '@A:train' "$s" "$loss" '@*:dev'
    g_run "p1_specB_${t}_s$s" '@B:train' "$s" "$loss" '@*:dev'
    for g in g specA specB; do step uv run vreg rescore "runs/p1_${g}_${t}_s$s" --clips '@*:dev' --tag p1; done
  done
}

# run once: the G chosen in docs/p1/decisions.json on the confirm split, no retraining
e23() {
  [ -f docs/p1/decisions.json ] || { echo "!!! docs/p1/decisions.json missing: write the pre-E23 decisions first"; return 1; }
  local loss; loss=$(g_loss) || return
  local t; t=$(loss_tag "$loss")
  # every G kind the decisions use: the one for class A and any the gate table names
  local kinds; kinds=$(uv run python scripts/p1_tables.py --g-kinds) || return
  for g in $kinds; do
    for s in $SEEDS; do
      step uv run vreg surrogate-eval "runs/p1_${g}_${t}_s$s" --clips '@*:confirm' --tag confirm
      step uv run vreg rescore "runs/p1_${g}_${t}_s$s" --clips '@*:confirm' --tag confirm
    done
  done
}

# exploratory addendum (backlog E14 applied to the P1 recipe): does the P1 result carry over to sim:ugc_av1?
# Reuses the frozen classes and splits; changes no P1 verdict.
e14() {
  local av1=sim:ugc_av1
  step uv run vreg rescore runs/strict_gp_noeq --clips '@*:*' --tag av1 --platform "$av1"
  for k in A B D; do step uv run vreg rescore "runs/p1_sharp_$k" --clips "@$k:*" --tag av1 --platform "$av1"; done
  for k in A B D; do
    step uv run vreg optimize --platform "$av1" --clips "@$k:train" --sampler gp --trials 15 --fix "$SHARP_ONLY" \
      --name "p1av1_sharp_$k"
    step uv run vreg rescore "runs/p1av1_sharp_$k" --clips "@$k:*" --tag av1
  done
}

# the rest of E14 (exploratory): the ugc_av1-tuned class theta back on ugc_strict, and P0's full GP-BO theta in both
# directions, on the P1 clips
e14r() {
  for k in A B D; do
    step uv run vreg rescore "runs/p1av1_sharp_$k" --clips "@$k:*" --tag strict --platform sim:ugc_strict
  done
  step uv run vreg rescore runs/strict_gp --clips '@*:*' --tag p1
  step uv run vreg rescore runs/av1_gp --clips '@*:*' --tag p1
  step uv run vreg rescore runs/strict_gp --clips '@*:*' --tag av1 --platform sim:ugc_av1
  step uv run vreg rescore runs/av1_gp --clips '@*:*' --tag strict --platform sim:ugc_strict
}

tables() { step uv run python scripts/p1_tables.py; }

for stage in "${@:-e20 e21 tables e22 tables}"; do
  for s in $stage; do "$s"; done
done
echo "=== $(date '+%F %T') P1 driver done"
