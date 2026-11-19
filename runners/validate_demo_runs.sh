#!/bin/bash
# Validates every demo config (model/configs/demo/) without running anything.
# Runs from the project root wherever it is started from.
#   ./runners/validate_demo_runs.sh
cd "$(dirname "$0")/.."
[ -f venv/bin/activate ] && source venv/bin/activate
mkdir -p logs

failed=()
for cfg in model/configs/demo/*.yaml; do
    name=$(basename "$cfg" .yaml)
    if python3 -u -m model.main "$cfg" --validate-only > "logs/validate_demo_${name}.log" 2>&1; then
        echo "OK    $name"
    else
        echo "FAIL  $name  (logs/validate_demo_${name}.log)"
        failed+=("$name")
    fi
done
echo
echo "${#failed[@]} config(s) failed"
[ ${#failed[@]} -eq 0 ]
