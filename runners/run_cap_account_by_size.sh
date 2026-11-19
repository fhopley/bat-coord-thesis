#!/usr/bin/env bash
# Battery-size sweep of the alternative offer analysis
# (docs/cap_account_battery_size_spec.md): the five variants at a $300/MWh cap
# on the coordinated Amber cohort, for every state and year 2021-2025, at each
# battery size.
#
# Runs from the project root wherever it is started from.
cd "$(dirname "$0")/.."
# run the following if creating new .sh files
#   chmod +x runners/run_cap_account_by_size.sh
# to run file use
#   ./runners/run_cap_account_by_size.sh
# a subset of sizes, in kWh:
#   SIZES="5 10" ./runners/run_cap_account_by_size.sh

# Stop on the first failure.
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/Volumes/Manual/Thesis Project}"
SIZES="${SIZES:-5 10 15 20 25 30 35 40 45 50}"
STATES="NSW QLD SA VIC"
YEARS="2021 2022 2023 2024 2025"
RESULTS_ROOT="analysis/output/cap_account/full_run_rolling_amber"
RETAILER_OUT="analysis/output/cap_account_retailer"
DYNAMIC="dynamic_tesla_electric_mixed_2026_modelled"

ANALYSIS="analysis/scripts/cap_account_analysis.py"
RETAILER="analysis/scripts/cap_account_retailer.py"
AGGREGATE="analysis/scripts/cap_account_aggregate.py"

# variant folder name | cap_account_analysis.py arguments
VARIANTS=(
  "cohort_x300_monthly_s0.01|--offer cohort --settlement-period monthly --surcharge-rate-per-kwh 0.01"
  "cohort_x300_daily_s0.01|--offer cohort --settlement-period daily --surcharge-rate-per-kwh 0.01"
  "retailer_x300_monthly_a0.9|--offer retailer --settlement-period monthly --share-fraction 0.9"
  "retailer_x300_daily_a0.9|--offer retailer --settlement-period daily --share-fraction 0.9"
  "${DYNAMIC}|--offer dynamic"
)

for size in ${SIZES}; do
  battery="${size}kWh"
  echo "=== ${battery} ==="

  # 1. Analysis, one run per variant, state and year.
  for entry in "${VARIANTS[@]}"; do
    args="${entry#*|}"
    for state in ${STATES}; do
      for year in ${YEARS}; do
        # shellcheck disable=SC2086
        python3 "${ANALYSIS}" --data-root "${DATA_ROOT}" ${args} \
          --threshold-per-mwh 300 --battery-name "${battery}" \
          --state "${state}" --year "${year}"
      done
    done
  done

  # 2. Retailer outcomes. Offer B's subscription change is solved here, per
  # variant, and written to retailer_summary.csv.
  python3 "${RETAILER}" --battery-name "${battery}" \
    --threshold-per-mwh 300 --subscription-change auto \
    --dynamic "${DYNAMIC}"

  # 3. Household outcomes, with Offer B at the subscription from step 2.
  summary="${RETAILER_OUT}/${battery}/x300/retailer_summary.csv"
  for entry in "${VARIANTS[@]}"; do
    variant="${entry%%|*}"
    sub_args=()
    case "${variant}" in
      retailer_*)
        sub="$(python3 -c "
import sys, pandas as pd
s = pd.read_csv(sys.argv[1]).set_index('variant')
print(s.loc[sys.argv[2], 'subscription_change_vs_amber'])
" "${summary}" "${variant}")"
        sub_args=(--subscription-change-per-month "${sub}")
        ;;
    esac
    python3 "${AGGREGATE}" --data-root "${DATA_ROOT}" \
      --battery-name "${battery}" --results-root "${RESULTS_ROOT}" \
      --variant "${variant}" ${sub_args[@]+"${sub_args[@]}"}
  done
done
