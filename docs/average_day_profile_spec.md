# Average day net import profile — Specification

## Scope

One chart for Results, Societal and Wholesale Value (`thesis/04_results.md`): the cohort's mean net import by local time of day for each cohort and offer.  It shows how differently the offers use the grid over a day.  On the time-of-use offers coordinated imports and exports start and stop at fixed local times, giving sharp steps in the profile.  Amber's dispatch follows each day's prices, giving a smoother profile that differs most from the No CER and Solar only profiles (exporting when they import most, and importing least when solar output is highest).  The Discussion uses it for the network value that is not estimated in this thesis.

A new script, `analysis/scripts/average_day_profile.py`, reusing the loading, weighting, local-time and run-basis machinery of `flow_exposure_analysis.py` (`docs/flow_exposure_spec.md`) by import rather than copying it.

## Series

| Key (`--switch-off`) | Name on chart | Run |
|---|---|---|
| `no_cer` | No CER | `full_run_no_cer_amber` (no battery-size suffix) |
| `solar_only` | Solar only | `full_run_solar_only_amber` (no battery-size suffix) |
| `self_consumption` | Self-consumption | `full_run_self_consumption_amber` |
| `amber` | Amber | Amber optimised run on the basis |
| `globird` | GloBird | GloBird optimised run on the basis |
| `agl` | AGL | AGL optimised run on the basis |
| `energyaustralia` | EnergyAustralia | EnergyAustralia optimised run on the basis |
| `origin` | Origin | Origin Battery Lite optimised run on the basis |

The optimised run for each offer on each basis is the plan's `lp_dir` from `flow_exposure_analysis.plans_for_basis(basis)` (which takes the run names from `financial_analysis.PLAN_RUNS`), so run names have one home.  All series are shown by default.  `--switch-off` takes one or more keys and leaves those series out of the chart (the CSV keeps every series).

On the conservative basis Amber's optimised run is `full_run_rolling_amber_predispatchonly`.  Where an optimised run is a placeholder in `financial_analysis.PLACEHOLDERS`, the placeholder's `run` flows stand in at the battery size where they exist, otherwise the offer is left out with a printed note, as `flow_exposure_analysis.py` does.

## Method

**Flows.**  `flows.parquet` (net import − export per site per 5-minute interval, kW) for each series, each state and year found (`discover_scenarios`), at `--battery-kwh` for the battery series.  Excluded sites dropped as in `load_flows`.

**Site weighting.**  Per interval, the cohort mean across sites with `weighted_row_mean` and that state-year's `site_weight` (`compute_site_weight_for_scenario`), or a plain mean with `--unweighted`.

**Local time of day.**  Each interval is assigned its local wall-clock time of day with `local_wall_clock` (`docs/flow_exposure_spec.md`, Local time), so the profile lines up with the local-time tariff windows in every state, including SA and daylight saving.  Profile bins are the 288 five-minute local times of day.  Each interval is labelled by its start time.

**Per state.**  For each series and state, pool that state's years' per-interval cohort means and take the mean in each time-of-day bin, so every day counts equally (the pooled-interval method of `docs/flow_exposure_spec.md`, Event and window exposure).

**National.**  Combine the state profiles bin by bin with `state_mix`, renormalised over the states that have the series (`_state_mix_weighted_average`).  With `--unweighted`, or when only one state is selected, the chart is for the pooled states without `state_mix`.

**Offer availability.**  An offer's `excluded_states` in `financial_analysis.RETAIL_PLANS` are respected (`series_excluded`).  EnergyAustralia is NSW only, so its flows are not loaded for QLD, SA and VIC, its national profile is its NSW profile, and the national data note carries the line "EnergyAustralia results for NSW only" unless the series is switched off.

**Months.**  `--months` restricts the days used (e.g. `--months 6 7 8` for winter), default all.

## Chart

`docs/thesis_figure_spec.md` format, single panel, height ~3.0 in.

- x axis: local time of day, 00:00 to 24:00, ticks every 3 hours.
- y axis: mean net import (kW per site), with the zero line emphasised.  Negative values are net exports.
- One line per series shown.  No CER solid black, Solar only solid grey, Self-consumption dashed grey, the offers solid in the offer colours and names used in the other Results charts (`OFFER_NAMES` and the plan colours in `flow_exposure_analysis.py`).
- Light shading of the evening peak window (16:00–21:00 local), the window used for the peak-window exposure charts.
- Legend below the panel.  No notes drawn in the figure (basis, battery size, weighting and period go in the caption).

## CLI

```
python3 analysis/scripts/average_day_profile.py
    --data-root PATH            project root with data/output/runs and data/clean/nem_prices
                                (default: cwd)
    --basis BASIS               achievable | conservative | perfect_foresight
                                (default: achievable)
    --battery-kwh KWH           battery size for the battery series (default: 20)
    --switch-off KEY [KEY ...]  series left off the chart, keys from the table above
                                (default: none)
    --states STATE [STATE ...]  restrict states (default: all found)
    --years YEAR [YEAR ...]     restrict years (default: all found)
    --months M [M ...]          restrict months (default: all)
    --per-state                 also write one chart and CSV per state, and per state-year
    --unweighted                plain per-site means, states pooled without state_mix
    --weights-file PATH         default: analysis/config/pv_size_bands.yaml
    --output-dir PATH           default: analysis/output/exposure
```

An unknown `--switch-off` key is an error that lists the valid keys.

## Outputs

Under `{output-dir}/{basis folder}/` (`financial_analysis.RUN_BASES` folder names, e.g. `imperfect_foresight_achievable`):

- `national/average_day_profile_{size}kWh.png` and `.csv`.
- With `--per-state`, the same two files in `{state}_all_years/` and in each `{state}_{year}/`. The state-year profile is the cohort mean over that year's intervals alone, weighted as for the all-years profile.
- When `--switch-off` or `--months` is given, the chart filename gets a suffix naming them (e.g. `average_day_profile_20kWh_excl_no_cer_solar_only.png`, `_months_6-7-8`) so the full chart is not overwritten.

The CSV has one row per time-of-day bin and one column per series (all series, whatever `--switch-off` says), plus a header block or companion rows giving the basis, battery size, weighting, states, years and months.

The thesis copy is `thesis/figures/results/societal_average_day_profile.png`.

## Values needed for the text

Per series, from the CSV (a small summary table printed at the end of the run and written as `average_day_profile_{size}kWh_summary.csv`):

- mean net import in the evening peak window (16:00–21:00) and the midday window (11:00–14:00);
- the largest and smallest values of the profile and their local times;
- the largest change between consecutive five-minute bins and its local time, as the measure of how sharp the transitions are.

## Notes

- On the achievable and perfect foresight bases the AGL, EnergyAustralia, Origin and GloBird runs include VPP event or critical-rate dispatch at maximum dispatch, which is part of their evening profile.  The conservative basis excludes it.
- Load time: one pass reads each `flows.parquet` once (8 runs × 20 state-years at one battery size), the same cost as one basis of `flow_exposure_analysis.py`.
