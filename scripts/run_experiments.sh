#!/usr/bin/env bash
# Main demo experiment: compare samplers on the simulated platforms.
set -euo pipefail
cd "$(dirname "$0")/.."
TRIALS="${TRIALS:-40}"
CLIPS="${CLIPS:-old_town_cross,park_joy}"
for s in gp tpe cmaes random; do
  uv run vreg optimize --platform sim:ugc_strict --clips "$CLIPS" --sampler "$s" --trials "$TRIALS" --name "strict_$s"
done
uv run vreg optimize --platform sim:ugc_av1 --clips "$CLIPS" --sampler gp --trials "$TRIALS" --name "av1_gp"
uv run vreg report runs/strict_gp runs/strict_tpe runs/strict_cmaes runs/strict_random \
  --out runs/report_ugc_strict.html --title "Sampler comparison on sim:ugc_strict"
uv run vreg report runs/av1_gp --out runs/report_ugc_av1.html --title "GP-BO on sim:ugc_av1"
