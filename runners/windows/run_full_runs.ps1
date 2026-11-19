# run_full_runs.ps1
#
# Runs every full_run_*.yaml config listed below, one after another, with
# per-config logging. Uses the SAME $configs array format as
# validate_full_runs.ps1 -- copy/paste the finished array across once every
# config has passed validation.
#
# Does NOT abort the batch if one config's run fails partway through:
# continues on to the next config and gives a full pass/fail summary at the
# end. For a multi-day unattended batch, a single bad scenario stopping
# everything downstream would waste the rest of the run; better to get
# everything that CAN complete done, and see the complete picture of
# what needs attention afterward. (If you'd rather stop at the first
# failure instead, say so and this is a one-line change.)
#
# Run with, after validate_full_runs.ps1 has passed:
#   .\runners\windows\run_full_runs.ps1
#
# Exit code: 0 if every run completed successfully, 1 if any failed.

# Runs from the project root wherever it is started from, and returns
# to the starting directory afterwards (also on exit or error).
Push-Location (Join-Path $PSScriptRoot '..\..')
try {

.\venv\Scripts\Activate.ps1
$env:PYTHONUTF8=1

# ---------------------------------------------------------------------
# Config files to run, in order. Copy/paste the finished array from
# validate_full_runs.ps1 once everything in it passes validation.
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

# Applied to every run in the list. Confirmed safe to apply uniformly:
# only perfect_foresight, perfect_foresight_advanced,
# perfect_foresight_custom, vpp_sc_event, and vpp_tou_event actually
# branch on --workers (checked orchestrate.py directly). The
# self_consumption path (no_cer/solar cohorts, or solar_battery with
# coordination_type='self_consumption') never references the workers
# variable at all -- not even to ignore it gracefully, it's simply absent
# from that code path -- so passing --workers there is a genuine no-op:
# no error, no wasted process pool, nothing. Safe to leave at one value
# for every config in the list below.
$Workers = 6

$configDir = "model/configs"

Write-Host "Running $($configs.Count) config file(s), workers=$Workers`n"

$succeeded = @()
$failed = @()
$batchStart = Get-Date

foreach ($configName in $configs) {
    $relPath = "$configDir/$configName"
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($configName)
    $timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $logPath = "logs/${stem}_${timestamp}.log"

    $runIndex = $configs.IndexOf($configName) + 1

    Write-Host "`n$('=' * 70)"
    Write-Host "  [$runIndex/$($configs.Count)] $configName" -ForegroundColor Cyan
    Write-Host "  Log: $logPath"
    Write-Host "$('=' * 70)"

    $runStart = Get-Date
    python3 -u -m model.main $relPath --workers $Workers --save-flows --monthly-bills 2>&1 | tee $logPath
    $runElapsed = (Get-Date) - $runStart

    if ($LASTEXITCODE -eq 0) {
        Write-Host "  --> $configName completed in $($runElapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Green
        $succeeded += $configName
    } else {
        Write-Host "  --> $configName FAILED (exit code $LASTEXITCODE) after $($runElapsed.ToString('hh\:mm\:ss')) -- see $logPath" -ForegroundColor Red
        $failed += $configName
    }
}

$batchElapsed = (Get-Date) - $batchStart

Write-Host "`n`n$('#' * 70)"
Write-Host "#  BATCH SUMMARY"
Write-Host "$('#' * 70)"
Write-Host "  Total elapsed: $($batchElapsed.ToString('dd\:hh\:mm\:ss'))"
Write-Host "  $($succeeded.Count)/$($configs.Count) config file(s) completed successfully`n"

if ($succeeded.Count -gt 0) {
    Write-Host "  SUCCEEDED:" -ForegroundColor Green
    $succeeded | ForEach-Object { Write-Host "    - $_" -ForegroundColor Green }
}

if ($failed.Count -gt 0) {
    Write-Host "`n  FAILED (check the corresponding log file in logs/):" -ForegroundColor Red
    $failed | ForEach-Object { Write-Host "    - $_" -ForegroundColor Red }
    exit 1
}

Write-Host "`nAll runs completed successfully." -ForegroundColor Green
exit 0

} finally {
    Pop-Location
}
