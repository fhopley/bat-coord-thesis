#!/usr/bin/env bash
# Cap hedge battery-size sweep (docs/cap_hedge_battery_size_spec.md).
# Runs from the project root wherever it is started from.
cd "$(dirname "$0")/.."
# run the following if creating new .sh files
#   chmod +x runners/run_cap_hedge_battery_size.sh
# to run file use
#   ./runners/run_cap_hedge_battery_size.sh            check, run every size, collect and chart
#   ./runners/run_cap_hedge_battery_size.sh --check    flows present per run and size only
# Extra arguments go to cap_hedge_battery_size.py (e.g. --force, --sizes 10 20, --no-conservative).
# Sizes with outputs under analysis/output/cap_hedge_battery_size/runs are skipped unless --force,
# so an interrupted sweep resumes where it stopped.
# Once the per-size runs exist, collect and redraw without the data drive:
#   python3 analysis/scripts/cap_hedge_battery_size.py

# Stop on the first failure.
set -euo pipefail

DATA_ROOT="/Volumes/Manual/Thesis Project"
OUT=analysis/output/cap_hedge_battery_size
mkdir -p "$OUT"

if [[ "${1:-}" == "--check" ]]; then
    python3 analysis/scripts/cap_hedge_battery_size.py --data-root "$DATA_ROOT" --check
    exit 0
fi

python3 analysis/scripts/cap_hedge_battery_size.py --data-root "$DATA_ROOT" --run "$@" 2>&1 \
    | tee "$OUT/run_$(date +%Y%m%d_%H%M%S).log"
