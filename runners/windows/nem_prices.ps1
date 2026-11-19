# Runs from the project root wherever it is started from, and returns
# to the starting directory afterwards (also on exit or error).
Push-Location (Join-Path $PSScriptRoot '..\..')
try {

.\venv\Scripts\Activate.ps1
$env:PYTHONUTF8=1

python3 pipeline/nem_prices/A_Download_NEM_Prices.py
python3 pipeline/nem_prices/B_Clean_NEM_Prices.py

} finally {
    Pop-Location
}
