#!/bin/bash
# Runs from the project root wherever it is started from.
cd "$(dirname "$0")/../.."
# source venv/bin/activate
# run the following if creating new .sh files 
#   chmod +x energy_masters_pipeline/scripts/estimate_battery_capacity.sh
# to run file use
#   ./energy_masters_pipeline/scripts/estimate_battery_capacity.sh

python3 -m energy_masters_pipeline.scripts.estimate_battery_capacity EM-036
python3 -m energy_masters_pipeline.scripts.estimate_battery_capacity EM-405
python3 -m energy_masters_pipeline.scripts.estimate_battery_capacity EM-414
python3 -m energy_masters_pipeline.scripts.estimate_battery_capacity EM-427
python3 -m energy_masters_pipeline.scripts.estimate_battery_capacity EM-444
python3 -m energy_masters_pipeline.scripts.estimate_battery_capacity EM-449
python3 -m energy_masters_pipeline.scripts.estimate_battery_capacity EM-467
python3 -m energy_masters_pipeline.scripts.estimate_battery_capacity EM-500
python3 -m energy_masters_pipeline.scripts.estimate_battery_capacity EM-520
python3 -m energy_masters_pipeline.scripts.estimate_battery_capacity EM-572
python3 -m energy_masters_pipeline.scripts.estimate_battery_capacity EM-573
python3 -m energy_masters_pipeline.scripts.estimate_battery_capacity EM-606
python3 -m energy_masters_pipeline.scripts.estimate_battery_capacity EM-720
python3 -m energy_masters_pipeline.scripts.estimate_battery_capacity EM-751
