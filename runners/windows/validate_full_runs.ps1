# validate_full_runs.ps1
#
# Preflight-validates every full_run_*.yaml config listed below before
# committing days of compute to them on a separate machine. Does NOT run any
# scenario -- calls model.main with --validate-only, which loads, expands,
# and validates every scenario in each config file exhaustively (does not
# stop at the first failure within a file), then exits.
#
# Static list (as opposed to auto_validate_full_runs.ps1's glob-based
# version): add one line to $configs below as each config file is finished.
# Doubles as a checklist of what's been set up so far -- an incomplete list
# here is as informative as a failed validation.
#
# This script also does not stop at the first FILE that fails -- it checks
# every config file in the list and gives the complete list of what's
# broken in one pass, since the point is to catch everything before walking
# away from the machine, not to fix-and-rerun repeatedly.
#
# Run with:
#   .\runners\windows\validate_full_runs.ps1
#
# Exit code: 0 if every listed config file's every scenario is valid, 1
# otherwise (so this can be used as a CI/automation gate too, not just
# interactively).

# Runs from the project root wherever it is started from, and returns
# to the starting directory afterwards (also on exit or error).
Push-Location (Join-Path $PSScriptRoot '..\..')
try {

.\venv\Scripts\Activate.ps1
$env:PYTHONUTF8=1

# ---------------------------------------------------------------------
# Config files to validate. Add one line per full_run_*.yaml as it's
# finished -- this list is also the checklist of what's been set up.
# ---------------------------------------------------------------------

$configs = @(
     "full_run_optimised_amber.yaml",
     "full_run_optimised_globird.yaml",
     "full_run_optimised_agl.yaml",
     "full_run_optimised_ea.yaml",
     "full_run_optimised_origin_lite.yaml",
     "full_run_optimised_origin_max.yaml"
)

# $configs = @(
#      "full_run_solar_only_amber.yaml",
#      "full_run_solar_only_globird.yaml",
#      "full_run_solar_only_agl.yaml",
#      "full_run_solar_only_ea.yaml",
#      "full_run_solar_only_origin_starter.yaml",
#      "full_run_solar_only_origin_max.yaml",
#      "full_run_self_consumption_amber.yaml",
#      "full_run_self_consumption_globird.yaml",
#      "full_run_self_consumption_agl.yaml",
#      "full_run_self_consumption_ea.yaml",
#      "full_run_self_consumption_origin_lite.yaml",
#      "full_run_self_consumption_origin_max.yaml",
#      "full_run_optimised_amber.yaml",
#      "full_run_optimised_globird.yaml",
#      "full_run_optimised_agl.yaml",
#      "full_run_optimised_ea.yaml",
#      "full_run_optimised_origin_lite.yaml",
#      "full_run_optimised_origin_max.yaml"
# )

$configDir = "model/configs"

Write-Host "Validating $($configs.Count) config file(s)`n"

$failedConfigs = @()
$passedConfigs = @()

foreach ($configName in $configs) {
    $relPath = "$configDir/$configName"
    Write-Host "`n$('=' * 70)"
    Write-Host "  Validating: $relPath" -ForegroundColor Cyan
    Write-Host "$('=' * 70)"

    if (-not (Test-Path $relPath)) {
        Write-Host "  --> $configName NOT FOUND at $relPath" -ForegroundColor Red
        $failedConfigs += $configName
        continue
    }

    python3 -u -m model.main $relPath --validate-only

    if ($LASTEXITCODE -eq 0) {
        $passedConfigs += $configName
    } else {
        $failedConfigs += $configName
        Write-Host "  --> $configName FAILED validation (exit code $LASTEXITCODE)" -ForegroundColor Red
    }
}

Write-Host "`n`n$('#' * 70)"
Write-Host "#  SUMMARY"
Write-Host "$('#' * 70)"
Write-Host "  $($passedConfigs.Count)/$($configs.Count) config file(s) passed validation`n"

if ($passedConfigs.Count -gt 0) {
    Write-Host "  PASSED:" -ForegroundColor Green
    $passedConfigs | ForEach-Object { Write-Host "    - $_" -ForegroundColor Green }
}

if ($failedConfigs.Count -gt 0) {
    Write-Host "`n  FAILED (see output above for the specific scenario/reason, or NOT FOUND):" -ForegroundColor Red
    $failedConfigs | ForEach-Object { Write-Host "    - $_" -ForegroundColor Red }
    Write-Host "`nDo not start the full run until every config file above is fixed." -ForegroundColor Red
    exit 1
}

Write-Host "`nAll config files passed. Safe to start the full run." -ForegroundColor Green
exit 0

} finally {
    Pop-Location
}
