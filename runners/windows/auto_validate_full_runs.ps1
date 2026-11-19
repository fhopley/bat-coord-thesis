# auto_validate_full_runs.ps1
#
# Preflight-validates every full_run_*.yaml config in model/configs/ before
# committing days of compute to them on a separate machine. Does NOT run any
# scenario -- calls model.main with --validate-only, which loads, expands,
# and validates every scenario in each config file exhaustively (does not
# stop at the first failure within a file), then exits.
#
# Unlike a real run, this script also does not stop at the first FILE that
# fails -- it checks every config file and gives you the complete list of
# what's broken in one pass, since the point is to catch everything before
# walking away from the machine, not to fix-and-rerun repeatedly.
#
# Run with:
#   .\runners\windows\auto_validate_full_runs.ps1
#
# Exit code: 0 if every config file's every scenario is valid, 1 otherwise
# (so this can be used as a CI/automation gate too, not just interactively).

# Runs from the project root wherever it is started from, and returns
# to the starting directory afterwards (also on exit or error).
Push-Location (Join-Path $PSScriptRoot '..\..')
try {

.\venv\Scripts\Activate.ps1
$env:PYTHONUTF8=1

$configDir = "model/configs"
$pattern = "full_run_*.yaml"

$configs = Get-ChildItem -Path $configDir -Filter $pattern | Sort-Object Name

if ($configs.Count -eq 0) {
    Write-Host "No config files matching $configDir/$pattern found." -ForegroundColor Yellow
    exit 1
}

Write-Host "Found $($configs.Count) config file(s) matching $configDir/$pattern`n"

$failedConfigs = @()
$passedConfigs = @()

foreach ($config in $configs) {
    $relPath = "$configDir/$($config.Name)"
    Write-Host "`n$('=' * 70)"
    Write-Host "  Validating: $relPath" -ForegroundColor Cyan
    Write-Host "$('=' * 70)"

    python3 -u -m model.main $relPath --validate-only

    if ($LASTEXITCODE -eq 0) {
        $passedConfigs += $config.Name
    } else {
        $failedConfigs += $config.Name
        Write-Host "  --> $($config.Name) FAILED validation (exit code $LASTEXITCODE)" -ForegroundColor Red
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
    Write-Host "`n  FAILED (see output above for the specific scenario/reason):" -ForegroundColor Red
    $failedConfigs | ForEach-Object { Write-Host "    - $_" -ForegroundColor Red }
    Write-Host "`nDo not start the full run until every config file above is fixed." -ForegroundColor Red
    exit 1
}

Write-Host "`nAll config files passed. Safe to start the full run." -ForegroundColor Green
exit 0

} finally {
    Pop-Location
}
