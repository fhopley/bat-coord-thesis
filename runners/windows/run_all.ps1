# Runs from the project root wherever it is started from, and returns
# to the starting directory afterwards (also on exit or error).
Push-Location (Join-Path $PSScriptRoot '..\..')
try {

.\venv\Scripts\Activate.ps1
$env:PYTHONUTF8=1

python3 -u pipeline/run_pipeline.py 2021 | tee logs/pipeline_2021.log
python3 -u pipeline/run_pipeline.py 2022 | tee logs/pipeline_2022.log
python3 -u pipeline/run_pipeline.py 2023 | tee logs/pipeline_2023.log
python3 -u pipeline/run_pipeline.py 2024 | tee logs/pipeline_2024.log
python3 -u pipeline/run_pipeline.py 2025 | tee logs/pipeline_2025.log

} finally {
    Pop-Location
}
