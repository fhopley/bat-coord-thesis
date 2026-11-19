#!/bin/bash
# Runs every demo config (model/configs/demo/) on the sample household H0001,
# with the flags used for the full runs. Output goes to
# data/output/runs/<run name>/, where the analysis scripts read it.
# Runs from the project root wherever it is started from.
#   ./runners/run_demo_runs.sh
#   WORKERS=8 ./runners/run_demo_runs.sh
#   ./runners/run_demo_runs.sh full_run_optimised_amber full_run_solar_only_amber
cd "$(dirname "$0")/.."
[ -f venv/bin/activate ] && source venv/bin/activate
mkdir -p logs
WORKERS=${WORKERS:-4}

if [ $# -gt 0 ]; then
    configs=()
    for name in "$@"; do configs+=("model/configs/demo/${name%.yaml}.yaml"); done
else
    configs=(model/configs/demo/*.yaml)
fi

for cfg in "${configs[@]}"; do
    name=$(basename "$cfg" .yaml)
    echo "=== $name"
    python3 -u -m model.main "$cfg" --save-flows --monthly-bills --workers "$WORKERS" \
        2>&1 | tee "logs/demo_${name}_$(date +%Y%m%d_%H%M%S).log" | grep -E "^Run:|elapsed|ABORTED|ERROR"
done
