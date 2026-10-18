#!/usr/bin/env bash
# E7 (docs/E7-plan.md): the real-anchored estimator for G. Same frozen emulator, clips, loss, trust region,
# steps and seeds as the P1 global G; only the estimator changes. Resumable per run; a failed step is
# logged and the script moves on.
# Usage: scripts/run_e7.sh [e7a|e7b|confirm|tables ...]   (default: e7a e7b tables)
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
SEEDS="0 1 2"

step() { echo "=== $(date '+%F %T') $*"; "$@" || echo "!!! FAILED ($?): $*"; }

run() {  # name anchors seed
  if [ ! -f "runs/$1/history.json" ]; then
    step uv run vreg surrogate --platform sim:ugc_strict --clips '@natural:train' --holdout '@*:dev' --seed "$3" \
      --g-loss lpips --estimator anchored --anchors "$2" --emulator-from "runs/p1_g_lpips_s$3/round0.pt" --name "$1"
  fi
  step uv run vreg rescore "runs/$1" --clips '@A:train,@*:dev' --tag e7
}

e7a() { for s in $SEEDS; do run "e7a_s$s" 6 "$s"; done; }
e7b() { for s in $SEEDS; do run "e7b_s$s" 1 "$s"; done; }

# only if the dev decision is "fixes" (scripts/e7_tables.py says so): E7a once on the class A confirm clips
confirm() {
  uv run python scripts/e7_tables.py --decision | grep -q '^fixes$' || { echo "!!! dev decision is not 'fixes': no confirm run"; return 0; }
  for s in $SEEDS; do
    step uv run vreg surrogate-eval "runs/e7a_s$s" --clips '@A:confirm' --tag confirm
    step uv run vreg rescore "runs/e7a_s$s" --clips '@A:confirm' --tag confirm
  done
}

tables() { step uv run python scripts/e7_tables.py; }

for stage in "${@:-e7a e7b tables}"; do
  for s in $stage; do "$s"; done
done
echo "=== $(date '+%F %T') E7 driver done"
