#!/bin/bash
# Runs from the project root wherever it is started from.
cd "$(dirname "$0")/.."
source venv/bin/activate
# run the following if creating new .sh files 
#   chmod +x runners/run_all.sh
# to run file use
#   ./runners/run_all.sh

python3 -u pipeline/run_pipeline.py 2021 | tee logs/pipeline_2021.log
python3 -u pipeline/run_pipeline.py 2022 | tee logs/pipeline_2022.log
python3 -u pipeline/run_pipeline.py 2023 | tee logs/pipeline_2023.log
python3 -u pipeline/run_pipeline.py 2024 | tee logs/pipeline_2024.log
python3 -u pipeline/run_pipeline.py 2025 | tee logs/pipeline_2025.log