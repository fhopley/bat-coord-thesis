#!/bin/bash
# Sensitivity runs for Results, Sensitivity Analysis (docs/sensitivity_runs_spec.md).
# Runs from the project root wherever it is started from.
cd "$(dirname "$0")/.."
source venv/bin/activate
# to run file use
#   ./runners/run_sensitivity.sh

set -u

WORKERS=6

# Site subset. Empty runs every site in the scenario (101 for NSW 2024).
# To cut the run time, put a comma-separated site list here; generate one
# that holds the PV size band mix with:
#   python3 analysis/scripts/sensitivity/select_sites.py --fraction 0.6 --quiet
# The processing scripts restrict the reference full runs to whichever sites
# these runs produce, so a subset stays a paired comparison.
SITES=""

if [ -n "$SITES" ]; then
  SITE_ARGS=(--sites "$SITES")
else
  SITE_ARGS=()
fi

run_config () {
  local name="$1"
  local cfg="model/configs/sensitivity/${name}.yaml"
  echo "=== ${name}  $(date '+%Y-%m-%d %H:%M:%S')"
  python3 -u -m model.main "$cfg" --save-flows --workers "$WORKERS" \
    ${SITE_ARGS[@]+"${SITE_ARGS[@]}"} 2>&1 \
    | tee "logs/${name}_$(date +%Y%m%d_%H%M%S).log"
}

# Calibration mode: ./runners/run_sensitivity.sh calibrate
#
# The horizon set's run time turns on how window solve cost scales with window
# length, which is the one quantity the existing runs cannot show. These two
# runs measure it at 20 kWh on CAL_SITES sites, against the production 18 h
# run's own summary.txt. About 20 minutes, and it decides whether the full
# horizon set fits the time available.
CAL_SITES=3

if [ "${1:-}" = "calibrate" ]; then
  for name in sens_h24_cal_rolling_amber sens_h48_cal_rolling_amber; do
    echo "=== ${name}  $(date '+%Y-%m-%d %H:%M:%S')"
    python3 -u -m model.main "model/configs/sensitivity/calibration/${name}.yaml" \
      --workers "$WORKERS" --max-sites "$CAL_SITES" 2>&1 \
      | tee "logs/${name}_$(date +%Y%m%d_%H%M%S).log"
  done
  echo
  echo "Per-site solve times, with the production 18 h run for comparison:"
  python3 analysis/scripts/sensitivity/run_times.py \
    --runs full_run_rolling_amber,sens_h24_cal_rolling_amber,sens_h48_cal_rolling_amber \
    --sizes 20
  exit 0
fi

# Preflight: every scenario checked before anything runs, so a bad config or a
# missing trace file surfaces now rather than part way through the horizon set.
for cfg in model/configs/sensitivity/*.yaml; do
  python3 -u -m model.main "$cfg" --validate-only || exit 3
done

# Order is cheapest first, so results arrive early and the horizon set, which
# dominates the budget, is last and can be stopped part way without losing the
# other three. Within the horizon set the longest window runs last for the
# same reason.

# 1. VPP control window (spec section 3). One scenario.
run_config sens_w15min_optimised_origin_lite

# 2. Degradation cost (spec section 1). Two scenarios.
run_config sens_cdeg034_rolling_amber
run_config sens_cdeg046_rolling_amber

# 3. Impact of nudge (spec section 4). Twenty scenarios.
run_config sens_nudge_rolling_origin_lite_novpp
run_config sens_nudge_rolling_ea_novpp

# 4. Rolling horizon (spec section 2). Forty scenarios, the 48 h set slowest.
run_config sens_h06_rolling_amber
run_config sens_h12_rolling_amber
run_config sens_h24_rolling_amber
run_config sens_h48_rolling_amber

echo "=== all sensitivity runs finished  $(date '+%Y-%m-%d %H:%M:%S')"
