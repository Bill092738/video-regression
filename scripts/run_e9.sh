#!/usr/bin/env bash
# E9 (docs/E9-plan.md): learned G on top of the class A sharpening theta, trained with the E7 estimator through
# a new emulator of the channel platform∘theta_A. Then, unattended, the rest of E14 (run_p1.sh e14r).
# Resumable per run; a failed step is logged and the script moves on.
# Usage: scripts/run_e9.sh [e9|tables|confirm|e14r ...]   (default: e9 tables confirm e14r)
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
SEEDS="0 1 2"

step() { echo "=== $(date '+%F %T') $*"; "$@" || echo "!!! FAILED ($?): $*"; }

e9() {
  for s in $SEEDS; do
    if [ ! -f "runs/e9_s$s/history.json" ]; then
      step uv run vreg surrogate --platform sim:ugc_strict --clips '@natural:train' --holdout '@*:dev' --seed "$s" \
        --g-loss lpips --estimator anchored --anchors 6 --front-from runs/p1_sharp_A --pairs-cache runs/e9_pairs \
        --name "e9_s$s"
    fi
    step uv run vreg rescore "runs/e9_s$s" --clips '@A:train,@*:dev' --tag e9
  done
}

tables() { step uv run python scripts/e9_tables.py; }

# only if the dev decision is "beats": E9 once on the class A confirm clips
confirm() {
  uv run python scripts/e9_tables.py --decision | grep -q '^beats$' || { echo "=== E9 dev decision is not 'beats': no confirm run"; return 0; }
  for s in $SEEDS; do
    step uv run vreg surrogate-eval "runs/e9_s$s" --clips '@A:confirm' --tag confirm
    step uv run vreg rescore "runs/e9_s$s" --clips '@A:confirm' --tag confirm
  done
  tables
}

e14r() { step scripts/run_p1.sh e14r tables; }

for stage in "${@:-e9 tables confirm e14r}"; do
  for s in $stage; do "$s"; done
done
echo "=== $(date '+%F %T') E9 driver done"
