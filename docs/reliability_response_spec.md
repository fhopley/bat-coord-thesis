# Reliability response analysis — Specification

## Purpose and scope

Measures the physical grid response of each cohort and offer in the
intervals of greatest system stress, for the discussion of coordinated
batteries and system reliability. Stress is taken from the regional spot
price: intervals at or near the market price cap (MPC), or above a static
price.

Net import is counted per interval with equal weight. It is not weighted by
price, since the quantity of interest is the physical flow, not the
retailer's financial exposure (`docs/cap_hedge_spec.md` covers that).

Scenarios are the bars of the `flow_exposure_analysis.py` event exposure
charts in optimised mode on the achievable imperfect-foresight basis
(`fe.set_basis('achievable')`, `fe._bar_order('optimised')`):

| Scenario | Run directory | Battery |
|---|---|---|
| No CER (load only) | `full_run_no_cer_amber` | none |
| Solar Only | `full_run_solar_only_amber` | none |
| Self Consumption (Amber) | `full_run_self_consumption_amber` | 20 kWh |
| Amber | `full_run_rolling_amber` | 20 kWh |
| GloBird | `full_run_rolling_globird` | 20 kWh |
| AGL | `full_run_rolling_agl` | 20 kWh |
| Energy Australia | `full_run_rolling_ea` | 20 kWh, NSW only |
| Origin Battery Lite | `full_run_rolling_origin` | 20 kWh |

With `--with-conservative`, each offer's conservative run is added
(`fe.plans_for_basis('conservative')`, the plan's `lp_dir`):

| Offer | Conservative run directory | Conservative case |
|---|---|---|
| Amber | `full_run_rolling_amber_predispatchonly` | Pre-dispatch price forecasts only |
| GloBird | `full_run_rolling_globird_novpp` | No critical rates |
| AGL | `full_run_rolling_agl_novpp` | No VPP events |
| Energy Australia | `full_run_rolling_ea_novpp` | No VPP events |
| Origin Battery Lite | `full_run_rolling_origin_novpp` | No VPP events |

These are the conservative imperfect-foresight runs of
`financial_analysis.PLAN_RUNS`, labelled `<offer> (conservative)`. The
range from conservative to achievable therefore spans both forecast
quality (Amber) and whether a VPP or critical rate is called during the
stress interval (the other offers).

Labels, colours, hatching and state exclusions come from
`flow_exposure_analysis.py` and `financial_analysis.py`, so they have one
home. The line chart legends name Self Consumption (Amber) as
"Self-consumption", since it is the only self-consumption scenario shown,
as the bar chart tick labels do. The battery size is `--battery-kwh` (default 20).

Sign convention: net import positive, kW per site. Flows are kWh per
5-minute interval, so kW = kWh × 12. Timestamps are interval-end AEST, as
in `docs/model_spec.md`.

## Formulation

### Site weighting

Every per-interval value is a PV-size-band weighted mean across a run's
sites, with the weights from `compute_site_weight_for_scenario` (the
`financial_analysis.py` pipeline and `analysis/config/pv_size_bands.yaml`),
renormalised over the sites present in each flows file. There is no
unweighted mode.

- `net_kw`: weighted mean site net import, kW.
- `share_importing`: weighted share of sites with net import > 0 in the
  interval.

### Selection of stress intervals

A selection is one of:

- **MPC − x** (`--mpc-offsets`, default 0 and 1000): intervals with
  `rrp >= MPC(t) − x − 0.01`. x = 0 is the intervals at the cap.
- **Static T** (`--thresholds`): intervals with `rrp > T`.

`MPC(t)` is the cap in force for the financial year of the interval start
(`MPC_BY_FY`, keyed by the calendar year in which the financial year
starts):

| Financial year | MPC ($/MWh) |
|---|---|
| 2020–21 | 15,000 |
| 2021–22 | 15,100 |
| 2022–23 | 15,500 |
| 2023–24 | 16,600 |
| 2024–25 | 17,500 |
| 2025–26 | 20,300 |

The MPC rises 35% over 2021 to 2025, so a static threshold sits ever
further below it. Selecting relative to the cap holds the depth of
selection constant across years. It also leaves out intervals just below
the cap where a price-following battery charges ahead of higher prices
later in the same event (for example QLD on 1 February 2022 at
$12,594/MWh, with the cap of $15,100 following).

### Event exposure (bar charts)

For each state and scenario, the cached intervals of all years are pooled
and `net_kw` is averaged over the selected intervals. A run with no
selected intervals in a state drops out of that state. The national value
is the `state_mix` weighted average over the states where the run has
selected intervals, renormalised over those states
(`fe._state_mix_weighted_average`). The mean RRP note is formed the same
way. This is the per-state multi-year and national tier of
`flow_exposure_analysis.py`, applied to the new selection.

### Conservative range

With `--with-conservative`, every output is written with a `_range` suffix
beside the achievable-only version, so either can be drawn from the same
cache. The titles end in ", imperfect foresight", the basis being carried
by the bar legend and the line labels.

- **Bars** (`fig_range_bar`): each offer's solid bar runs from zero to its
  conservative value, and a segment in the offer colour tinted towards
  white (`sbf.RANGE_TINT`) runs from the conservative to the achievable
  value, as in `savings_bar_figure.py`. The segment is drawn under the
  solid bar, so a conservative import stays visible when the achievable
  value is an export. Bars are labelled with the achievable value only.
  The legend adds `sbf.RANGE_LABEL`. Scenarios without a conservative run
  (No CER, Solar Only, Self Consumption) are single bars.
- **Flow duration and response**: the core charts add the conservative
  Amber line, dotted in the Amber colour, with the two Amber lines
  labelled "(achievable)" and "(conservative)". On the core flow duration
  chart the legend moves below the axes in two columns, since the
  conservative curve fills the lower left. The `_all` flow duration chart
  stays achievable only.
- **CSVs**: the conservative scenarios are added as rows of
  `interval_summary_range.csv` and as columns of `events_range.csv`.

### Flow duration (equal width)

Every selected (state, interval) of a scenario is one point, weighted
`state_mix[s] / Y_s`, where `Y_s` is the number of years of that run in
state `s`. States therefore count by their mix and not by how often their
price reached the selection. Points are sorted from largest import to
largest export, and the x axis is the cumulative weight as a percentage of
the total. This is the `count` weighting of `cap_hedge_analysis.py`.

The core chart shows No CER, Solar Only, Self Consumption (Amber) and
Amber. The `_all` chart adds every offer present in every state that has
selected intervals, so Energy Australia (NSW only) is not on it.

### Events

An event is a run of selected intervals in one state. Consecutive selected
intervals whose interval ends are more than `EVENT_GAP_MIN + 5` minutes
(35 minutes) apart start a new event. Up to six unselected intervals can
therefore sit inside one event. Events are formed on the state's price
series across all years, from the No CER cache (every run shares the
interval grid).

Per event (`events.csv`):

| Column | Definition |
|---|---|
| `start` | interval start of the first selected interval |
| `end` | interval end of the last selected interval |
| `n_intervals` | selected intervals in the event (gap intervals not counted) |
| `span_h` | `end − start`, hours, gaps included |
| `mean_rrp`, `max_rrp` | over the selected intervals |
| `mpc` | MPC in force at the first selected interval |
| `share_at_mpc` | share of the selected intervals priced at the MPC |
| `<scenario> mean kW`, `<scenario> max kW` | mean and largest `net_kw` over the selected intervals |
| `<scenario> share sites importing` | mean `share_importing` over the selected intervals |

### Response around events

For each event, `net_kw` is taken at offsets of −12 h to +6 h
(`RESPONSE_WINDOW_H`) in 5-minute steps, measured from the first selected
interval. Offset 0 is that interval, plotted at its interval-end time. The
line at each offset is the mean across events, with each event given equal
weight whatever its state or length. The values are PV-size weighted but
not `state_mix` weighted. The line includes gap intervals and post-event
intervals. The core scenarios are shown.

The shaded area, on the right axis, is the share of events still in
progress at each offset: those whose offset lies between 0 and the offset
of their last selected interval, gaps included.

### Interval summary

Per scenario (`interval_summary.csv`), over the selected intervals of all
states and years pooled without `state_mix` weighting except where noted:

| Column | Definition |
|---|---|
| `national_mean_kw` | the national bar value (`state_mix` weighted) |
| `intervals` | selected (state, interval) count |
| `pooled_mean_kw` | mean `net_kw` |
| `share_intervals_importing` | share of intervals with `net_kw > 0` |
| `worst_interval_kw`, `p95_interval_kw` | largest and 95th percentile `net_kw` |
| `mean_share_sites_importing` | mean `share_importing` |
| `events` | events with data for the scenario |
| `events_with_import` | events with any interval with `net_kw > 0` |
| `events_mean_import` | events with mean `net_kw > 0` |

`state_year_counts.csv` gives the selected intervals and events per state
and calendar year of interval start.

## Interface

Script: `analysis/scripts/reliability_response_analysis.py`, run from the
repo root.

```
python3 analysis/scripts/reliability_response_analysis.py --data-root "/Volumes/Manual/Thesis Project"
python3 analysis/scripts/reliability_response_analysis.py --charts-only --mpc-offsets 0 500 1000 2000
```

| Option | Default | Meaning |
|---|---|---|
| `--data-root` | `.` | Root holding `data/output/runs`, `data/clean/nem_prices` and `data/processed/edp` |
| `--output-dir` | `analysis/output/reliability` | Cache and chart output |
| `--mpc-offsets` | `0 1000` | x values, $/MWh |
| `--thresholds` | none | Static thresholds, $/MWh |
| `--states`, `--years` | all | Filter on the discovered state-years |
| `--battery-kwh` | 20 | Battery size of the battery scenarios |
| `--weights-file` | `analysis/config/pv_size_bands.yaml` | Size bands and `state_mix` |
| `--charts-only` | off | Draw from the cache without reading flows; state-years are those in the cache |
| `--extract-only` | off | Build the cache and stop |
| `--no-notes` | notes on | Leave the data notes off the charts, for the thesis |
| `--with-conservative` | off | Add the conservative runs; outputs named `*_range` |

### Stages

**Extract.** One pass over the `flows.parquet` of each scenario and
discovered state-year (`discover_scenarios`, from the No CER run). For each,
the rows within `CONTEXT_HOURS` of any interval with `rrp > EXTRACT_FLOOR`
are written to `<output-dir>/cache/<run_dir>/<state>_<year>.parquet` with
columns `datetime_aest`, `rrp`, `net_kw`, `share_importing`. An existing
cache file is not rebuilt, so the extract can be run in pieces
(`--states`, `--years`), and adding `--with-conservative` later reads only
the conservative runs. To rebuild, delete the cache file.

**Charts.** For each selection, into `<output-dir>/<folder>/`, where
`<folder>` is `MPC` (x = 0), `MPC-<x>` or `T<threshold>`:

| File | Content |
|---|---|
| `event_exposure_national.png` | Event exposure bars, national |
| `event_exposure_<state>.png` | Event exposure bars per state |
| `flow_duration_count.png` | Flow duration, core scenarios |
| `flow_duration_count_all.png` | Flow duration, all offers present in every state |
| `event_response.png` | Response around events |
| `events.csv`, `interval_summary.csv`, `state_year_counts.csv` | As above |

Bar charts are drawn by `fe.fig_scenario_bar`, with the mode left out of
the title so it fits the axes width. The other charts follow
`docs/thesis_figure_spec.md` through `thesis_style`. On the core flow
duration chart the legend is lower left and the note upper right; on the
`_all` chart the legend is below the axes; on the response chart the note
sits directly above the legend, both lower left.

## Configuration

Module constants:

| Constant | Value | Role |
|---|---|---|
| `EXTRACT_FLOOR` | 5000 $/MWh | Lowest price a selection can reach |
| `CONTEXT_HOURS` | 12 | Cached either side of every interval above the floor |
| `EVENT_GAP_MIN` | 30 | Largest gap joined within one event |
| `RESPONSE_WINDOW_H` | (−12, 6) | Response chart window, hours from event start |
| `DEFAULT_MPC_OFFSETS` | (0, 1000) | Default selections |
| `MPC_BY_FY` | table above | Market price cap by financial year |

`CONTEXT_HOURS` must be at least the larger magnitude of
`RESPONSE_WINDOW_H`, so every offset of the response chart is in the cache.

## Failure handling

- A selection whose lowest price is below `EXTRACT_FLOOR` (static T below
  5,000, or x above `min(MPC_BY_FY) − 5,000` = 10,000) stops the script
  with an error.
- A flows file that does not exist is skipped in the extract without a
  message. When the charts are drawn, every scenario and state-year
  missing from the cache is listed as a warning and left out.
- An interval whose financial year is not in `MPC_BY_FY` has an MPC of
  NaN and is never selected by an MPC − x selection.

## Known limitations

- **Price as the measure of stress.** High prices can come from network
  constraints or bidding, not only from scarcity. The deepest supply
  stress of the period, in June 2022, was under administered pricing and
  then market suspension, with prices held at the administered price cap,
  so no MPC − x or high static selection includes it. Fix path: select
  intervals from AEMO lack-of-reserve notices (actual LOR2 and LOR3) from
  the NEMWeb market notices, in place of or alongside the price selection.
- **Price taker.** The flows come from runs that take the historical price
  as given. A fleet large enough to change the price would change both
  the selection and its own response.
- **Empty battery and deliberate charging are not separated.** The cache
  holds net flows only, so an import interval cannot be attributed to a
  depleted battery or to charging. Fix path: add the battery flow from
  `battery_flows.parquet` (and the state of charge built from it) to the
  cache.
- **Response chart weighting.** Events are pooled with equal weight across
  states, not by `state_mix`, so states with more events dominate.

## Validation

QLD 2022 only, measured 19 November 2026 in Claude's environment from the
achievable runs:

- No interval in the 2021 to 2025 price files is priced above
  `MPC_BY_FY`, and each year's highest price equals a cap in the table.
- MPC − 1,000: 51 intervals, 24 events. Event exposure No CER +2.36 kW,
  Solar Only +2.09 kW, Self Consumption (Amber) +1.13 kW, Amber −7.08 kW,
  GloBird −4.09 kW, AGL −3.36 kW, Origin −4.93 kW.
- MPC (x = 0): 48 intervals, 22 events. The Amber cohort imports in one
  interval, QLD 5 July 2022 09:35 at $15,500/MWh (1.04 kW, 77% of sites
  importing). The cohort sits near household load (about +1.25 kW) in the
  three intervals before it, after exporting 8.7 kW at 09:10 to 09:15,
  consistent with an empty battery; not confirmed (see Known limitations).
- MPC − 1,000, conservative runs: Amber −4.87 kW, GloBird −0.53 kW, AGL
  +0.37 kW, Origin +0.83 kW.
