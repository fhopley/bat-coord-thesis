# Flow exposure analysis — Specification

## Purpose and scope

`analysis/scripts/flow_exposure_analysis.py` characterises the grid and
retailer exposure of each coordination strategy from physical flows: the
households' net import and export against the wholesale price and against
time-of-day windows, the VPP forced flows, and the battery throughput. It
covers every state and year the run outputs support, with per-site PV size
weighting and a national tier weighted by state mix.

It reads model outputs written with `--save-flows` and draws every chart for
each run basis. On the achievable basis it also writes the battery cycling
CSVs that `battery_cycling_figure.py` draws the thesis figure from
(`battery_cycling_chart_spec.md`).

```bash
python3 analysis/scripts/flow_exposure_analysis.py --data-root "/Volumes/Manual/Thesis Project"
python3 analysis/scripts/flow_exposure_analysis.py --basis achievable --cycling-sweep
```

---

## Inputs

Under `--data-root`:

| File | Content |
|---|---|
| `data/output/runs/{run}/scenarios/{STATE}_{YEAR}[_{N}kWh]/flows.parquet` | Net grid flow per site (columns) per 5-minute interval, kWh, import positive |
| `.../battery_flows.parquet` | Net battery AC flow per site, kWh, charge positive (`sol_to_bat + grid_to_bat − bat_to_home − bat_to_grid`) |
| `.../vpp_forced_flows.parquet` | Forced flow per site, same layout as `flows.parquet`, zero outside events (perfect-foresight VPP types only) |
| `.../rolling_diagnostics.csv` | Per site: `f_imp_total_kwh`, `f_exp_total_kwh`, `cap_limit_kwh` (rolling types) |
| `.../summary.txt` | `coordination_type`, `retail_tariff` |
| `data/clean/nem_prices/{YEAR}_nem_prices.parquet` | `NSW1_rrp`, `QLD1_rrp`, `SA1_rrp`, `VIC1_rrp` by `datetime_aest` |
| Site metadata | For the PV size weights (`financial_analysis_spec.md`, section 6) |

The VPP caps come from the `vpp` block of the retail tariff named in
`summary.txt` (`model/tariffs/{tariff}.yaml`). Sites in
`site_exclusions.EXCLUDED_SITES` are dropped from every flows file.

**Discovery.** The (state, year) pairs are the `{STATE}_{YEAR}` scenario
folders of `full_run_no_cer_amber`, the one run guaranteed to exist with no
battery-size suffix. `--states` and `--years` narrow the set. A run missing
for a scenario is skipped with a warning.

---

## Run registry and run bases

Three reference scenarios, shown on every chart family that compares
strategies:

| Label | Run | Colour |
|---|---|---|
| No CER (load only) | `full_run_no_cer_amber` | black |
| Solar only | `full_run_solar_only_amber` | `#aaaaaa` |
| Self-consumption (Amber) | `full_run_self_consumption_amber` | outlined grey key style |

Then one entry per offer in `COORDINATION_PLANS`, each with a
self-consumption run (`sc_dir`), an optimised run (`lp_dir`), a colour, a
default mode (the run that is the offer's as-joined behaviour) and an
optimisation type (VPP- or user-dispatched):

| Offer (display name) | Default mode | Optimised by |
|---|---|---|
| Amber | optimised | VPP |
| GloBird | self-consumption | VPP |
| AGL | self-consumption | user |
| EnergyAustralia | optimised | VPP |
| Origin (Battery Lite) | self-consumption | user |

The run basis sets which runs the entries point at. The registry as written
is the perfect-foresight basis (`full_run_self_consumption_*`,
`full_run_optimised_*`). For the conservative and achievable bases each
offer's `(sc_dir, lp_dir)` comes from `financial_analysis.PLAN_RUNS`, so run
names have one home. An offer whose optimised run is a placeholder in
`financial_analysis.PLACEHOLDERS` for that basis is left out, because a
blended saving has no equivalent in flows. `--basis` selects the bases
(default all). Each writes to its own subdirectory of `--output-dir`, and
its title (`financial_analysis.RUN_BASES`) is appended to every chart title.

An offer's availability by state also comes from `financial_analysis`
(`excluded_states` on its `RETAIL_PLANS` entry, matched by `sc_mode`).
EnergyAustralia is NSW only, so its runs are not loaded for QLD, SA and VIC
(`run_excluded`, `plans_in_state`), and the cycling sweep skips them too.
Its national results are then its NSW results, and the national charts
carry the data note line "EnergyAustralia results for NSW only".

---

## Weighting

**Per-site weights.** Each (state, year) has a `site_weight` Series from
the same sizing and size-band pipeline as `financial_analysis.py`
(`load_site_meta`, `compute_dc_ac_ratio`, `assign_direct_size`,
`estimate_missing_sizes`, `assign_bands_and_weights`), imported rather than
duplicated so the two scripts weight sites identically. The DC:AC ratio is
pooled across states per year. The weights are computed once per scenario
and reused for every run in it, since a site's weight does not depend on the
retail plan it is modelled under. Sites without a resolvable size or outside
the band range are excluded from the weighted results and kept in the
unweighted ones.

**Cohort mean.** Per interval, the weighted mean across site columns,
renormalised over the sites present in that file (`weighted_row_mean`); per
site annual totals use the same convention (`weighted_series_mean`). With
`--unweighted`, plain means.

**National tier.** Each state's representative result (its multi-year
result where it has more than one year, otherwise its single year) is
combined by `state_mix`, renormalised over the states that have data for
that run, plan and metric (`combine_states_scenario_stats`,
`combine_states_plan_data`). The same combination is used for every chart
family that has a per-state result. Combining each state's own
event-conditional mean, rather than pooling raw qualifying intervals across
states, keeps population share in charge of representativeness: a state
with more frequent price events gets a more precise number, not a larger
national weight. The national tier is drawn only in weighted mode and only
when more than one state has a representative result.

---

## Local time

The morning, evening and midday windows are local wall-clock time, as in the
tariff logic. Flows and prices are indexed in AEST (UTC+10, no daylight
saving), so the window masks use `local_wall_clock` from
`model/core/localtime.py` with each scenario's state. It is loaded by file
path (`importlib`) so that `model/core/__init__.py`, and the other core
modules it imports, are not executed. The local hour is computed once per
(state, year) and reused for every run, since all runs in a scenario share
the interval grid. The price-threshold metrics are not affected.

---

## Loading

One pass over every needed (run, scenario): each `flows.parquet` is read
once and feeds both the pooled threshold sweeps and the per-scenario window
and event metrics, since the flows files dominate the I/O. Runs that appear
in more than one role (`full_run_self_consumption_amber`) are read once.
Battery runs are read at `--battery-kwh` (default 20). For each interval the
cohort mean net import is converted to kW (÷ interval hours) and joined to
the state's RRP. The offer runs also get import (positive part) and export
(negative part) components.

The pass prints a table of run, scenario, sites present, sites used and
mean net kW, so a missing run or a failed price join is visible directly.

---

## Charts

All charts follow `docs/thesis_figure_spec.md` through `thesis_style`, with
offers named by `OFFER_NAMES` in their registry colours and legends below
the axes. Each carries a data note (state, years, sites, battery size,
weighting, mean price where relevant) unless `--no-notes` is given, for the
thesis, where the note goes in the caption.

### Conditional mean net import by price threshold (pooled)

For each run, every discovered scenario's per-interval `(net kW, RRP)` rows
are concatenated across all states and years. For each threshold p, from
the lower of −$1,000 and the minimum RRP up to the maximum RRP in $100
steps, the value is the mean net import over intervals with RRP ≥ p. These
charts ask how import behaviour depends on price level, irrespective of when
or where, so the intervals are pooled directly rather than averaging
per-scenario curves. The x axis runs to $5,000/MWh, with the $300 and
$1,000/MWh thresholds marked.

- `conditional_import_default_mode`: the three references, and each offer's
  default-mode run, solid for self-consumption and dashed for optimised.
- `conditional_import_optimised_mode`: the three references and each offer's
  optimised run, dashed.

### Per-offer decomposition (pooled)

`decomposition_{offer}`, one per offer: net, import and export threshold
curves for the offer's self-consumption and optimised runs. Component is
line style (net solid, import dashed, export dotted); mode is colour
(self-consumption a fixed dark grey `#404040`, optimised the offer colour).
Each offer's own self-consumption run is used, not a shared reference,
because some offers dispatch the battery for events even in
self-consumption mode.

### Event and window exposure (per scenario, per state, national)

The mean per-site net import (kW) over the intervals selected by each
metric:

| Metric | Selection | Stem |
|---|---|---|
| `ev300`, `ev600`, `ev1000` | RRP > $300, $600, $1,000/MWh | `event_exposure_{300,600,1000}` |
| `morning` | local 6am–9am | `peak_window_morning` |
| `evening` | local 4pm–9pm | `peak_window_evening` |
| `midday` | local 11am–2pm | `midday_export_exposure` |

Each metric has a default-mode and an optimised-mode chart
(`{stem}_{default,optimised}_mode`). Bars: No CER, Solar only,
Self-consumption (Amber) in its outlined style, then each offer. Colour is
offer identity, so the run behind a bar is shown by a white hatch in the
bar's colour: none for VPP-optimised, `///` for self-consumption, `xxx` for
user-optimised. The default-mode chart shows each offer's default run (so
self-consumption or VPP-optimised); the optimised-mode chart each offer's
optimised run (VPP- or user-optimised). Bars are labelled to two decimals,
and the note gives the mean RRP over the selected intervals.

**Per-state multi-year tier.** For a state with more than one year, each
year's per-interval cohort means (computed with that year's own weights and
prices) are concatenated and the metric's selection is applied to the pooled
rows, giving one mean over all qualifying intervals. Averaging each year's
own mean instead would give a calm year with a few events the same say as a
volatile year with many. Weight in the pooled result comes from how many
qualifying intervals a year contributes, which suits a frequency-sensitive
metric; differing site counts between years need no adjustment because each
year's rows already reflect its own sample. The window metrics would give
nearly the same answer either way (every year has the same number of window
intervals), and one method is used for all six.

### Evening peak by battery size (one scenario)

`evening_peak_by_battery_size`: mean net import in the local 4pm–9pm window
for No CER and Solar only, then self-consumption and Amber's optimised run
on the basis at every battery size from 5 to 50 kWh. One (state, year), set
by `--state` / `--year` (default the first found). The Amber bars are left
out on a basis without an Amber optimised run.

### Forced-flow summary

Per offer, per site, as whole-year totals:

```
natural   = SC run total  − SC run forced         (import and export)
forced    = optimised run forced
voluntary = optimised run total − natural − forced
```

`natural` is what the site does under self-consumption without VPP events.
Taking it from the self-consumption run and subtracting it from the
optimised run's total assumes natural flow is a property of the site's
solar, battery and load, not of the plan. The six quantities are computed
per site and then weighted once.

Forced totals come from, in order: `vpp_forced_flows.parquet` (reindexed to
the `flows.parquet` sites, missing sites zero); zero when the coordination
type in `summary.txt` has no VPP events; `f_imp_total_kwh` and
`f_exp_total_kwh` in `rolling_diagnostics.csv` for the rolling VPP types,
which write no forced-flow parquet. A VPP run with neither source, or with
a site missing from its diagnostics, raises an error rather than reading as
zero.

Only the forced component is charted (`forced_flow_summary`): per offer with
any forced flow, a waterfall of forced discharge (export) from zero, forced
charge (import) floating on it, and their total from zero, in kWh/yr per
site. Discharge starts from zero so that a discharge-only cap can be marked
on it. Each offer's annual cap is drawn as a short dashed line on the
column it applies to: the total under `limit_applies_to: import_and_export`,
discharge under `export_only`. The cap is the tariff's `kwh_limit`, checked
against `cap_limit_kwh` in `rolling_diagnostics.csv` with a warning if they
differ. An offer whose tariffs give different caps across the scenarios
drawn gets no cap mark.

The natural and voluntary components are not charted. `voluntary` can be
negative where the optimised run suppresses activity the self-consumption
run would have had (holding charge for a later event instead of exporting
solar), so the three components are an approximation that holds when
optimisation mostly adds activity.

### Battery throughput and discharge summaries

From `battery_flows.parquet`: annual charge (positive part) and discharge
(negative part) per site, weighted. Self-consumption (Amber's
self-consumption run) first, then each offer's optimised run.

- `battery_throughput_summary`: a waterfall per offer of discharge, charge
  floating on it and total, in MWh/yr to one decimal.
- `battery_discharge_summary`: equivalent full cycles per day (discharge ÷
  battery size ÷ 365) as one bar per offer, labelled with cycles per day
  and, beneath, the annual discharge in MWh/yr. Warranty terms are set on
  discharge, so charge is left out.

### Multi-year and national tiers for the annual totals

The forced-flow, throughput and discharge quantities are unconditional
annual sums, so the per-state multi-year value is the plain mean of each
year's (already weighted) result; pooling would give the same number. An
offer appears on a state's multi-year chart only if it has more than one
year there. The national tier combines the per-state results by
`state_mix` as above.

**Site counts in notes.** `n` is shown only for a single (state, year),
where every bar shares one site count. It is omitted on every multi-year
and national chart: a pooled event mean is over intervals, not sites, and a
`state_mix`-weighted national value has no single sample size behind it.

### Battery cycling CSVs (achievable basis)

On the achievable basis the national tier also writes
`battery_cycling_{N}kwh.csv` (per offer: annual discharge, cycles per day,
forced discharge, charge and total, cap and its column), and with
`--cycling-sweep`, `battery_cycling_by_size.csv` (national discharge and
cycles per day at every battery size, reading every size's
`battery_flows.parquet`). See `battery_cycling_chart_spec.md`.

---

## Interface

```
python3 analysis/scripts/flow_exposure_analysis.py
    --data-root PATH          project root; flows, prices and metadata at their
                              usual relative paths underneath (default: cwd)
    --output-dir PATH         (default: analysis/output/exposure)
    --basis BASIS [BASIS...]  perfect_foresight, conservative, achievable or
                              all (default: all)
    --states STATE [STATE...] restrict the discovered states
    --years YEAR [YEAR...]    restrict the discovered years
    --battery-kwh KWH         battery size for every battery chart except the
                              evening-peak size chart (default: 20)
    --state STATE             state for the evening-peak size chart
                              (default: first found)
    --year YEAR               year for the evening-peak size chart
                              (default: first found)
    --unweighted              plain per-site means; no national tier
    --weights-file PATH       (default: analysis/config/pv_size_bands.yaml)
    --no-notes                leave the data notes off, for the thesis
    --cycling-sweep           achievable basis: also write
                              battery_cycling_by_size.csv (slow)
```

Weighted mode needs the weights file and exits with an error without it.
The script also exits with an error if no scenarios are discovered, none
match the filter, or no `flows.parquet` is found for any run.

---

## Output

```
{output-dir}/{basis folder}/
    conditional_import_{default,optimised}_mode.png
    decomposition_{amber,globird,agl,energy_australia,origin_battery_lite}.png
    evening_peak_by_battery_size.png
    {STATE}_{YEAR}/
        event_exposure_{300,600,1000}_{default,optimised}_mode.png
        peak_window_{morning,evening}_{default,optimised}_mode.png
        midday_export_exposure_{default,optimised}_mode.png
        forced_flow_summary.png
        battery_throughput_summary.png
        battery_discharge_summary.png
    {STATE}_all_years/      the same set, for a state with more than one year
    national/               the same set, weighted mode with more than one state;
                            on the achievable basis also battery_cycling_*.csv
```

The basis folders are `perfect_foresight`, `imperfect_foresight_conservative`
and `imperfect_foresight_achievable`.

---

## Known limitations

- **Pooled line charts are not weighted by state mix.** The conditional
  import and decomposition charts pool every state-year's intervals with
  site weighting only, so each state-year counts by its number of intervals,
  not its population share. Fix path: build each state's curve first and
  combine by `state_mix`, as the bar charts do.
- **Net import only.** Every chart except the per-offer decomposition
  measures net import. Where a retailer cannot offset one customer's
  high-price imports against other customers' exports, import alone is the
  hedging quantity. Fix path: an import-only series on the conditional
  import charts.
- **Forced flows by interval for the rolling VPP runs.** These runs write no
  `vpp_forced_flows.parquet` (PB-13 in `docs/review/possible_bugs.md`), so
  only annual forced totals are available for them, and no chart can show
  when forced dispatch happened or at what price.
