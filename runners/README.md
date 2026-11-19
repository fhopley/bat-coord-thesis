# Runners

Batch scripts that run the pipelines, the model and the analyses. Each one
moves to the project root before running, so it can be started from anywhere.
The runners activate `venv/` in the project root. The runners other than the
demo runners are set up for the EDP households, and the analysis runners read
model outputs from an external drive (`DATA_ROOT`).

| Runner | Runs |
|---|---|
| `validate_demo_runs.sh` | `--validate-only` over every `model/configs/demo/` config |
| `run_demo_runs.sh` | the demo configs on the sample household, all or those named as arguments, with `--save-flows --monthly-bills` |
| `validate_full_runs.sh` | `--validate-only` over the same config list as `run_full_runs.sh`; run it first |
| `run_full_runs.sh` | the production `full_run_*.yaml` configs, with `--save-flows --monthly-bills --workers 6`; one log per config in `logs/`. Commented lines are the other configs, kept to switch in |
| `run_all_cap_analysis.sh` | `cap_account_analysis.py` over every state and year, then `cap_account_retailer.py` and `cap_account_aggregate.py`; stops on the first failure |
| `run_cap_account_by_size.sh` | the alternative offer analysis at every battery size (`docs/cap_account_battery_size_spec.md`); `SIZES` selects a subset |
| `run_cap_hedge_battery_size.sh` | the cap hedge analysis at every battery size (`docs/cap_hedge_battery_size_spec.md`); extra arguments go to `cap_hedge_battery_size.py`, and completed sizes are skipped unless `--force` |
| `run_sensitivity.sh` | the sensitivity runs (`docs/sensitivity_runs_spec.md`); `SITES` restricts them to a site subset |
| `run_all.sh` | the EDP pipeline for 2021–2025 |

```bash
./runners/validate_full_runs.sh
./runners/run_full_runs.sh
```

`windows/` holds the PowerShell equivalents, run as
`.\runners\windows\<name>.ps1`:

| Runner | Runs |
|---|---|
| `validate_full_runs.ps1`, `run_full_runs.ps1` | a listed set of `full_run_*.yaml` configs, continuing past failures and summarising pass/fail at the end |
| `auto_validate_full_runs.ps1` | validates every `model/configs/full_run_*.yaml` |
| `run_all.ps1` | the EDP pipeline for 2021–2025 |
| `nem_prices.ps1` | NEM price download and clean |

The Energy Masters battery estimate batches are next to the scripts they call,
in `energy_masters_pipeline/scripts/`.
