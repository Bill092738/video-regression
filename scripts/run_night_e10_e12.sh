#!/usr/bin/env bash
# Overnight chain (2026-10-15): E10 (docs/E10-plan.md) then E12 (docs/E12-plan.md). A failed step is logged and
# the chain moves on; every step is resumable (evaluation cache, SQLite studies).
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
step() { echo "=== $(date '+%F %T') $*"; "$@" || echo "!!! FAILED ($?): $*"; }
step uv run python scripts/e11.py run --platform sim:ugc_av1 --recipes tier,r_default,r_mild
step uv run python scripts/e10.py
step uv run python scripts/e12.py search --seed 0
step uv run python scripts/e12.py search --seed 1
step uv run python scripts/e12.py eval
step uv run python scripts/e12.py tables
echo "=== $(date '+%F %T') night chain done"
