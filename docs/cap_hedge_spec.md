# Cap hedge analysis — Specification

## Purpose and scope

Measures the $300 cap cover a retailer needs per site to hedge the above-$300
spot exposure of a solar/battery cohort, and the net position from holding a
given quantity of cover, for these scenarios:

- No CER (load only), Solar Only, Self Consumption (Amber), Amber
- optionally GloBird, AGL, Origin Battery Lite and Energy Australia

Each battery plan uses its rolling, imperfect-foresight run: the plan's
`lp_dir` in `flow_exposure_analysis.py` with `full_run_optimised_` replaced by
`full_run_rolling_` (Amber: `full_run_rolling_amber`). A scenario whose run
directory does not exist is left out with a printed note; `_novpp` runs are
not used.

Labels, colours and the reference run directories are those in
`flow_exposure_analysis.py` (`REFERENCE_SCENARIOS`, `COORDINATION_PLANS`).
The reference greys lighten away from the no-CER baseline, as on that
script's bar charts: No CER black, Solar Only `#aaaaaa`, Self Consumption
`BAR_SC_REFERENCE_COLOR` (`#cccccc`).

The retailer is treated as settling the cohort's net flow at spot in every
scenario, Amber included (Amber customers bear spot themselves in practice).
The result is the hedging need created by each scenario's physical flows, not
by any plan's retail settlement. Exposure below $300/MWh, which retailers
cover with swaps, is out of scope.

---

## ASX $300 cap futures

Product: ASX Australian Electricity Base Load Calendar Quarter $300 Cap
Futures, cash settled, one contract per region per quarter: NSW `GN`,
QLD `GQ`, VIC `GV`, SA `GS`.

| Term | Value |
|---|---|
| Contract unit | 1 MW, base load (00:00–24:00, every day of the quarter): 2,160 / 2,184 / 2,208 MWh for 90 / 91 / 92-day quarters |
| Quotation | $/MWh |
| Last trading day | Last business day of the quarter, 4:00 pm |
| Cash settlement day | Fourth business day after the last trading day |
| Cash settlement price | `CSP = (C − 300·D) / E` |

`C` is the sum of the region's base load spot prices in the quarter that are
greater than $300, `D` the number of those prices, and `E` the number of all
spot prices in the quarter. The spot price is the five-minute price for
quarters starting on or after 1 October 2021, and the thirty-minute price
before that. Sources: ASX 24 Contract Specifications (fetched 1 October 2026);
ASX Australian Electricity fact sheet (2025).

Rewritten per interval, with `P_i` the spot price and `Δt` the interval length
in hours (1/12 or 1/2):

```
CSP_q = (1/E) · Σ_i max(P_i − 300, 0)
payout per MW = CSP_q · H_q = Σ_i max(P_i − 300, 0) · Δt        [$]
```

where `H_q` is the hours in quarter `q` (the contract size in MWh per MW). The
cap therefore pays on every interval priced above $300, base load, whatever
the quarterly average. A buyer at traded price `F_q` pays `F_q · H_q` and
nets `(CSP_q − F_q) · H_q` per MW. Daily margining moves the timing of these
cash flows, not the total; financing cost is ignored.

### Consequences for the price data

- **Pre-5MS quarters.** The clean price files repeat each 30-minute price
  across its six 5-minute slots (`nem_price_spec.md`), so summing excess over
  5-minute slots with `Δt = 1/12` gives the 30-minute settlement exactly. One
  formula serves both periods.
- **Quarter assignment.** `datetime_aest` is interval-end, and each yearly
  file runs from the interval ending 00:05 on 1 January to the one ending
  00:00 on the next 1 January (`nem_price_spec.md`). The quarter is taken from
  the interval start, `datetime_aest − 5 min`, so every interval in a yearly
  file falls in that year's quarters.
- **Strictly above $300.** Intervals at exactly $300/MWh pay nothing and are
  excluded from the above-cap set. In 2021–2025 there are 1,623 such 5-minute
  intervals in NSW, 710 in QLD, 906 in SA and 100 in VIC. The `ev300` and
  `ev1000` metrics in `flow_exposure_analysis.py` use the same strict
  inequality.

---

## Formulation

Per state `s` and 5-minute interval `t`:

| Symbol | Meaning | Units |
|---|---|---|
| `m_t` | Cohort net import per site, positive = import: `weighted_row_mean(flows) / INTERVAL_HOURS` with `site_weight` as in `flow_exposure_spec.md` | kW/site |
| `e_t` | `max(P_t − 300, 0)` | $/MWh |
| `x` | Cap holding per site. Cohort holding is `x · n_sites / 1000` MW. Negative `x` is cap sold. | kW/site |
| `F_q` | ASX traded price for quarter `q` | $/MWh |

Per site, per quarter:

```
payout_q(x)  = x/1000 · Σ_{t∈q} e_t · Δt          = x/1000 · CSP_q · H_q
premium_q(x) = x/1000 · F_q · H_q
L_q          = 1/1000 · Σ_{t∈q} m_t · e_t · Δt     (above-cap cost of the cohort's net flow)
N_q(x)       = payout_q(x) − premium_q(x) − L_q
             = x/1000 · H_q · (CSP_q − F_q) − L_q                          [$/site]
```

`L_q` is negative when the cohort net-exports through the quarter's spikes.

The same holding `x` applies in every quarter: cover is chosen before it is
known which quarter will be bad.

`N_q` is linear in `x`, and the slope `H_q · (CSP_q − F_q) / 1000` does not
depend on the scenario. The mean of `N_q` over quarters is therefore a set of
parallel straight lines, one per scenario, sloping down when caps traded on
average above their realised payout. Scenarios differ in the intercept
(`−L_q`) and in how `N_q` is spread across quarters. The value of holding
cover shows in the spread, so the net-position chart carries the worst
quarter alongside the mean.

`W_k(x)` is the mean of the `k` lowest values of `N_q(x)` over the sample's
quarters; `W_1(x)` is the worst quarter. Each `N_q` is a straight line in `x`,
rising in quarters where `CSP_q > F_q` and falling in the others, so `W_1` is
the lower envelope of those lines. For a cohort that imports through spikes, the
worst quarter at `x = 0` is a spike quarter whose caps paid more than they
cost; adding cover lifts it until a quiet quarter, where the premium buys
little payout, becomes the worst.
`W_k` is concave and piecewise linear, so its maximum can be a flat range.

### Hedge levels reported per scenario

- **`x_0` — exposure-neutral holding.** `x_0 = Σ_t m_t·e_t / Σ_t e_t` over the
  whole sample: the excess-weighted mean net import. At `x_0` total payout
  equals total `L` over the sample, and the summed net position equals minus
  the premium paid. `x_0 < 0` means the cohort is a natural seller of caps.
- **`x_worst` — worst-quarter holding.** The `x` that maximises `W_1(x)`.
  Each `W_k` changes slope only where two quarter lines cross, so it is
  evaluated exactly at every pairwise crossing, and the maximum is reported as
  a range where it is flat. If `W_k` still rises at the edge of the crossings
  (every quarter line slopes the same way), it has no maximum and `x_worst` is
  reported as `+inf` or `-inf`. Where the maximum is flat, its lower end is
  used for markers and for mean `N_q` at `x_worst`. When the mean of
  `N_q` slopes down, holdings above `x_worst` make both the mean and the worst
  quarter worse; below it, less cover raises the mean and lowers the worst
  quarter, and the choice depends on the retailer's risk appetite. The
  rational range of cover is therefore from zero (or from selling, if
  `x_worst < 0`) up to `x_worst`. No holding maximises the mean: with caps
  priced above their payout every kW of cover lowers it, so cover is held only
  to narrow the spread.
- **`x_worst(k)`** — the `x` that maximises `W_k(x)`, for `k` in 1, 2, 4
  and 10. A holding above the largest of these is worse on the mean and on
  every `W_k` reported.

### Chart 1 — above-cap flow duration curve

One line per scenario. The scenario's intervals with `P_t > 300` are sorted
from most import to most export, `y = m_t` (kW/site, import positive, export
negative). `m_t` is the cohort mean, so individual sites can import in
intervals where the cohort net-exports. Two figures with the same y-axis.
Legend entries for battery scenarios (Self Consumption and the battery plans)
carry the battery size, e.g. "Amber, 20 kWh".

- **Count (`flow_duration_300_count.png`).** Title "Cohort net import in
  >$300/MWh intervals, sorted from most import to most export"; x-axis
  "Share of >$300/MWh intervals (%)", 0–100%, each interval equal width.
  The x position where a line crosses zero is the share of above-$300
  intervals in which the cohort net-imports. The mean height of a line is the
  scenario's mean `m_t` over those intervals.
- **Weighted (`flow_duration_300_weighted.png`).** Title "Cohort net import
  in >$300/MWh intervals, each interval's width proportional to its price
  above $300/MWh"; x-axis "Share of >$300/MWh intervals, weighted by price
  above $300/MWh (%)". Each interval's width is proportional to `e_t · Δt`;
  with `Δt` constant this is weighting by `P_t − 300`. The x position where a
  line crosses zero is the price-weighted share of above-$300 intervals in
  which the cohort net-imports. Height times width is the cohort's above-cap
  cost in the interval, so the signed area under a line is proportional to
  `Σ L_q`, and its mean height is `x_0`. Each scenario's `x_0` is drawn as a
  dotted horizontal line in the scenario's colour and given in its legend
  entry in kW/site. For a holding `x` drawn as a horizontal line, the area
  where the curve is above the line is above-cap cost the cap does not cover,
  and the area where the curve is below it is payout in excess of cost; the
  two areas are equal at `x_0`. A mean height above that of the count figure
  means the cohort imports more in the highest-priced intervals. The
  footnote defines `x_0` as the mean net import weighted by price above
  $300/MWh; equivalently, the $300 cap holding whose total payout equals the
  cohort's net above-$300 cost (negative = net earnings) over the sample.
  The cap premium is not included.

Each line is sorted independently, so a given x position is not the same
interval on different lines. Chart 1 carries no cap price and pools all
quarters, so it gives the cover that matches exposure over the sample, not a
holding to buy.

### Chart 2 — net position against cap holding

x-axis: `x` from −6 to +2 kW/site in 0.05 steps, widened to include every
finite `x_0` and `x_worst` with 0.5 kW/site to spare. Two panels sharing the
x-axis, both in $/site/quarter:

- **(a)** mean of `N_q(x)` over quarters, annotated with the common slope:
  the mean cost of each kW/site of cover per quarter, `−mean_q a_q`
- **(b)** worst quarter, `W_1(x)`, peaking at `x_worst`

`x_0` is marked on each scenario's line in both panels, `x_worst` in panel (b).

### Chart 3 — each scenario at its own worst-quarter cover

Each scenario is evaluated at its own `x_worst`, so a change of scenario moves
both the rational cover and the outcomes. Three bar panels, one bar per
scenario:

- **(a)** `x_worst` (kW/site; negative = cap sold)
- **(b)** `W_1(x_worst)` ($/site/quarter)
- **(c)** mean `N_q(x_worst)` × 4 ($/site/year)

Each bar is labelled with its value and, for scenarios other than No CER, the
change from No CER. A scenario whose `x_worst` is infinite is marked instead
of drawn. Panel (a) carries a whisker from the smallest to the largest finite
`x_worst(k)`, showing how far the rational cover moves with the risk measure.

### Chart 4 — state-month outcomes with no cover

The no-cover net position per state and calendar month,
`N_m = −L_m = −1/1000 · Σ_{t∈m} m_t · e_t · Δt` ($/site), with months
assigned from the interval start as for quarters. Each state-month counts
once: `m_t` is the state's PV-size-weighted cohort mean, and there is no
`state_mix` weighting. Four states over 2021–2025 give 240 state-months per
scenario.

Histogram of `N_m` over state-months, titled "Above-$300/MWh net earnings per
site, by state-month, with no cap cover"; x-axis "Net earnings in >$300/MWh
intervals ($/site per month; negative = cost)", y-axis "State-months". A
reference scenario is drawn as a black outline and each battery plan as
filled bars; Solar Only and the other reference are not drawn. The reference
is No CER by default, or Self Consumption with
`--state-month-reference self_consumption`. All bins have one width, the
first of 1, 2, 5, 10, 20, 25, 50, 100, 200, 500 $/site giving at most 50 bins
over the plotted range, with a bin edge at zero so that costs and earnings
never share a bin. A month is nil when it has no >$300 intervals; its
`N_m = 0` falls in the bin starting at zero and is hatched. Otherwise it is
earnings when `N_m > 0` and cost when `N_m < 0`, however small. The legend
(upper right) names the scenarios and the hatching. A table below it,
right-aligned with it and with its first column sized to the longest scenario
name, centred vertically on the midpoint of the y gridline interval nearest
the middle of the axes so its outline falls between gridlines, gives per
scenario the number of state-months under a "State-month net earnings" heading split into "> $0",
"< $0" and "Nil", and the worst cost month ($/site) with its state. Written
under the output directory to `state_months/state_month_outcomes.png` (No CER
reference) or `state_months/state_month_outcomes_vs_sc.png` (Self
Consumption reference).

### Tiers

- **Per state, all years pooled.** Every year of flows found for the state.
- **National.** Chart 1 weights each (state, interval) point by
  `state_mix[s] / Y_s`, with `Y_s` the number of years of data for state `s`,
  multiplied by `e_t · Δt` in the weighted figure. Charts 2 and 3 use
  `N_q,national(x) = Σ_s state_mix[s] · N_{s,q}(x)`, renormalised over the
  states with data in quarter `q`. Cover is bought per region against
  intervals as they occur, so a state with more frequent spikes contributes
  more of the national need; this differs from the national tier in
  `flow_exposure_spec.md` (Weighting, national tier), which combines per-state conditional means.
  Produced only when weighted and more than one state is present.

Charts 2 and 3 and `x_worst` use the quarters common to
every scenario loaded in the tier, so all scenarios are compared over the
same quarters; quarters dropped for this reason are printed. Chart 1 and
`x_0` use every interval loaded.

No per-(state, year) tier: a year gives four quarters, too few for the
worst-quarter measure to separate spike and quiet quarters.

### Summary table

CSV and console, one row per (tier, scenario): above-$300 interval count, mean
`m_t` over those intervals, share of them with `m_t > 0`, `x_0`, `x_worst`,
`x_worst(k)` for each `k`, mean `N_q` at 0, `x_0` and `x_worst`, `W_1` at 0
and `x_worst`, the quarter that is worst at each of those holdings, and the
smallest and largest finite `x_worst(k)`.

`quarter_lines.csv` has one row per (tier, quarter, scenario): `csp`, `price`
(the `F_q` used) and `hours` (state tiers only), `slope_per_kw` (`a_q`),
`above_cap_cost` (`L_q`), `N_q` at 0, `x_0` and `x_worst`, and the quarter's
rank at `x_worst` (1 = worst).

`state_months/state_months.csv` has one row per (scenario, state, year,
month), for every scenario loaded: `n_intervals_gt300`, `above_cap_cost`
(`L_m`) and `N_at_0` (`N_m`).

---

## Inputs

- **Flows.** Same run directories, `--battery-kwh` (default 20), site
  exclusions and site weighting as `flow_exposure_analysis.py`.
- **Spot prices.** `data/clean/nem_prices/{YEAR}_nem_prices.parquet`.
  `CSP_q` is computed from the price file over every interval of the quarter,
  independent of the flows.
- **`F_q`.** From `ASX_CAP_BY_REGION` in `analysis/scripts/asx_caps.py`, the
  traded price at the start of each quarter. With `--premium
  start_of_quarter` (default) each quarter is charged its own price. With
  `--premium sample_mean` every quarter is charged the state's mean of those
  prices over the tier's quarters: cover bought well ahead at a stable price.
  The mean slope of `N_q` is almost unchanged (only the `H_q` weighting
  differs); what moves is which quarters the premium falls on. A quarter with
  flows but no `F_q` is reported and dropped from chart 2.
- **`state_mix`.** From the weights config, as in `flow_exposure_analysis.py`.

---

## Interface

### `analysis/scripts/asx_caps.py`

The single home for cap contract data and settlement. No plotting, no file
I/O.

- `CAP_STRIKE = 300.0` ($/MWh).
- `ASX_NSW_CAP`, `ASX_QLD_CAP`, `ASX_SA_CAP`, `ASX_VIC_CAP`:
  `{(year, quarter): $/MWh}`, start-of-quarter traded prices (NSW also
  carries forward quarters), moved unchanged from `nem_price_analysis.py`.
  `ASX_CAP_BY_REGION` maps `'NSW1'`, `'QLD1'`, `'SA1'`, `'VIC1'` to them.
  `FIRST_FORWARD_QUARTER = (2026, 3)`.
- `contract_quarter(index: pd.DatetimeIndex) -> pd.MultiIndex` — `(year,
  quarter)` of each interval-end timestamp, taken from the interval start
  (`index − 5 min`).
- `cap_settlement(rrp: pd.Series) -> pd.DataFrame` — one row per
  `(year, quarter)` present in `rrp` (5-minute, interval-end, $/MWh), with
  columns `csp` ($/MWh, `Σ max(P − CAP_STRIKE, 0) / E` with `E` the count of
  non-NaN 5-minute prices in the quarter), `hours` (`H_q`, calendar days ×
  24) and `n_nan`. Quarters with `n_nan > 0` are printed by the caller.

`nem_price_analysis.py` imports the four dictionaries, `CAP_STRIKE` and
`FIRST_FORWARD_QUARTER` from `asx_caps.py` by name, so its existing
`ASX_CAP_BY_REGION` assembly is unchanged. Its docstring section on how cap
contracts work is replaced with a reference to this spec.

### `analysis/scripts/cap_hedge_analysis.py`

Run from the project root:

```bash
python3 analysis/scripts/cap_hedge_analysis.py --data-root "/Volumes/Manual/Thesis Project"
```

It imports `REFERENCE_SCENARIOS`, `COORDINATION_PLANS`, `RRP_COL`,
`INTERVAL_HOURS`, `discover_scenarios`, `load_flows`, `get_prices`,
`weighted_row_mean` and `compute_site_weight_for_scenario` from
`flow_exposure_analysis.py`, and `load_weights_config` from
`financial_analysis.py`, so scenario definitions, site exclusions and
weighting stay identical between the two scripts. It makes its own loading
pass over the flows of the scenarios listed under Purpose and scope.

Arguments `--data-root`, `--output-dir`, `--states`, `--years`,
`--battery-kwh`, `--unweighted` and `--weights-file` behave as in
`flow_exposure_analysis.py`. `--state-month-reference` (`no_cer` or
`self_consumption`) selects the reference outline on chart 4. `--core-only` limits the scenarios to No CER,
Solar Only, Self Consumption and Amber. Battery plans use their rolling run
and are drawn dashed; the three references are solid. `--premium` selects the
basis for `F_q` (see Inputs). Outputs go to `{STATE}_all_years/` and
`national/` under `analysis/output/cap_hedge/` (`--premium start_of_quarter`)
or `analysis/output/cap_hedge_sample_mean/` (`--premium sample_mean`) unless
`--output-dir` is given: `flow_duration_300_count.png` and
`flow_duration_300_weighted.png` (chart 1),
`net_position_vs_cover.png` (chart 2), `outcomes_at_x_worst.png` (chart 3),
`hedge_summary.csv` and `quarter_lines.csv`; and under `state_months/`,
`state_month_outcomes.png` (chart 4) and `state_months.csv`.

---

## Verification

`model/tests/test_asx_caps.py` (loading `asx_caps.py` by file path):

- `cap_settlement` on the 2024 price file reproduces AEMO's published Q1 2024
  cap returns (see Validation) within AEMO's $1/MWh rounding.
- Pre-5MS: on a 2021 Q1 fixture of back-filled 30-minute prices, `csp · H_q`
  equals the 30-minute excess × 1/2 summed over half-hours.
- `contract_quarter` assigns the interval ending 00:00 on 1 January to Q4 of
  the prior year.
- An interval at exactly `CAP_STRIKE` contributes zero; one at
  `CAP_STRIKE + 0.01` contributes.

`model/tests/test_cap_hedge.py` (loading `cap_hedge_analysis.py` by file
path, with `analysis/scripts` on `sys.path`):

- `maximise_lowest_k` gives the crossing point of the two-quarter example
  (`x ≈ 0.744`, `W_1 ≈ −34.5`), agrees with a brute-force grid for random
  quarter lines at `k` = 1, 2, 4 and 10, returns `±inf` when every line
  slopes the same way, and returns the full range of a flat maximum.
- `exposure_neutral` balances payout-weighted net import about `x_0`.
- `analyse`: `x_worst`, `W_1` and mean `N_q` at `x_worst`, mean `N_q` at
  zero and the worst quarter at `x_worst` match values computed directly from
  the quarter lines.
- `build_state_tier` with `sample_mean` charges every quarter the mean of the
  start-of-quarter prices and leaves `L_q` unchanged; with
  `start_of_quarter` it charges each quarter its own price.
- `quarter_lines`: the quarter ranked 1 at `x_worst` is the summary's worst
  quarter there, with the same `W_1`.

Analysis-level properties, checked in `cap_hedge_analysis.py`'s diagnostic
output (`CHECK` lines):

- Per-state tier: summed payout at `x_0` equals summed `L` to float
  tolerance, where every flows quarter has an `F_q`.
- Per (scenario, state): each quarter's `L_m` summed over its three months
  equals `L_q` (`CHECK` line giving the largest absolute difference).
- The loading table prints, per scenario and state-year, sites present and
  used, rows with a NaN flow or price, and the count of `P_t > 300`
  intervals (non-vacuity of chart 1).
- Per-state tier: the summary's mean `m_t` over intervals with `P_t > 300`
  matches the `ev300` bar in the matching `{STATE}_all_years` output of
  `flow_exposure_analysis.py` (checked by reading the two outputs).

---

## Known limitations

- **Premium basis.** `F_q` is the price at the start of the quarter. A
  retailer builds cover progressively over the one to three years before
  delivery and pays the average of the prices over that window. Fix path:
  replace `F_q` with the average daily settlement price of each contract over
  a chosen accumulation window, which needs ASX24 daily settlement history.
  `--premium sample_mean` is the bounding case of a long, stable window.
- **Sample size.** About 20 quarters per state for 2021–2025. `x_worst` is set
  by the one spike quarter and one quiet quarter whose lines cross at the
  peak, so it describes the sample's worst quarters; a future worst quarter
  can be larger or differently shaped.
- **Constant holding.** `x` is the same in every quarter. A retailer varying
  cover by quarter (for example with `F_q`) is not modelled.

---

## Validation

Realised `CSP_q` against AEMO Quarterly Energy Dynamics Q1 2024 cap returns,
computed 1 October 2026 from `data/clean/nem_prices/2024_nem_prices.parquet`:

| Region | Computed ($/MWh) | AEMO QED ($/MWh) |
|---|---|---|
| NSW | 16.58 | 17 |
| QLD | 32.21 | 32 |
| VIC | 15.35 | 15 |
| SA | 11.18 | 11 |

Amber cohort means below −10 kW/site in chart 1 come from multi-phase sites,
not from breached export limits. A check of every site's exports against its own
export limit on `full_run_rolling_amber` and `full_run_optimised_amber`, 20 kWh, every
state and year 2021–2025 (1,436 site-years per run), run 1 October 2026:

- No site exports above its own limit,
  `export_limit_kw_per_phase × max(ac_load_net_phases, pv_site_net_phases)`.
- In each state-year's lowest cohort-mean interval (all between 08:00 and
  15:00 AEST), every site exporting above 10 kW is multi-phase (504 of 504,
  rolling run). The weighted cohort mean reaches −12.9 kW/site.
- The largest single-site exports are 26.7 kW (three-phase, limit 30 kW) and
  19.6 kW (two-phase, limit 20 kW). Three sites are recorded with five or six
  phases (S0218, S0305, S0593); their largest exports, 20–23 kW, are within a
  three-phase limit, so the phase count does not change these results.

Mean realised `CSP_q` against mean `F_q` (start-of-quarter), 2021 Q1 – 2025 Q4,
20 quarters per region, computed 1 October 2026:

| Region | Mean `CSP_q` ($/MWh) | Mean `F_q` ($/MWh) |
|---|---|---|
| NSW | 19.64 | 22.67 |
| QLD | 22.96 | 25.27 |
| SA | 21.79 | 23.73 |
| VIC | 8.46 | 10.06 |

