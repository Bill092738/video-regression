#!/usr/bin/env bash
# Main demo experiment: compare samplers on the simulated platforms.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
TRIALS="${TRIALS:-40}"
CLIPS="${CLIPS:-old_town_cross,park_joy}"
for s in gp tpe cmaes random; do
  uv run vreg optimize --platform sim:ugc_strict --clips "$CLIPS" --sampler "$s" --trials "$TRIALS" --name "strict_$s"
done
uv run vreg optimize --platform sim:ugc_av1 --clips "$CLIPS" --sampler gp --trials "$TRIALS" --name "av1_gp"
uv run vreg transfer runs/strict_gp --clips all

# learned pre-compensation: failure case without trust region, then the working configuration
uv run vreg surrogate --clips park_joy,old_town_cross,ducks_take_off --holdout in_to_tree,sky_gradient \
  --rounds 1 --variants 5 --emulator-steps 4000 --generator-steps 3000 --trust-weight 0 --max-residual 0.15 \
  --name surrogate_naive
uv run vreg surrogate --clips park_joy,old_town_cross,ducks_take_off --holdout in_to_tree,sky_gradient \
  --rounds 3 --variants 5 --name surrogate_trust

uv run vreg report runs/strict_gp runs/strict_tpe runs/strict_cmaes runs/strict_random \
  --out runs/report_ugc_strict.html --title "Sampler comparison on sim:ugc_strict"
uv run vreg report runs/av1_gp --out runs/report_ugc_av1.html --title "GP-BO on sim:ugc_av1"
uv run vreg report runs/surrogate_trust --surrogate --out runs/report_surrogate_trust.html
