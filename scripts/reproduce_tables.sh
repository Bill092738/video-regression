#!/usr/bin/env bash
# Regenerate every results table and figure (P0, P1, E7, E9, E10, E11, E12) from the cached run outputs in
# runs/ (no platform calls), then check that the tables copied into PROGRESS.md still match them.
# Usage: scripts/reproduce_tables.sh        (needs the runs/ directory of the experiments)
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
fail=0
step() { echo "=== $*"; "$@" > /dev/null || { echo "!!! FAILED: $*"; fail=1; }; }
step uv run python scripts/p0_tables.py
step uv run python scripts/p1_tables.py
step uv run python scripts/e7_tables.py
step uv run python scripts/e9_tables.py
step uv run python scripts/e11.py tables
step uv run python scripts/e10.py
step uv run python scripts/e12.py tables
uv run python scripts/check_progress_tables.py || fail=1
exit $fail
