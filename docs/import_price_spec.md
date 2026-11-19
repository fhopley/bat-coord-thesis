# Import price analysis — Specification

## Purpose and scope

Measures the average wholesale price a retailer pays for a cohort's grid
imports, per scenario:

- No CER (load only), Solar Only, Self Consumption (Amber), Amber
- optionally GloBird, AGL, Origin Battery Lite and Energy Australia

Scenario definitions, run directories and colours come from
`cap_hedge_analysis.build_scenarios`, so each battery plan uses its rolling,
imperfect-foresight run and the battery size is set by `--battery-kwh`
(default 20 kWh). `--core-only` limits the scenarios to the first four.

Only imports are priced. The value of exports is outside this script; the
above-$300 part of it is covered by `cap_hedge_spec.md`.

## Formulation

`flows.parquet` holds each site's net import per 5-minute interval (kWh,
import positive). Gross import at site `s` in interval `t` is
`imp_s,t = max(flow_s,t, 0)`; one site's export is not netted against
another site's import.

Per state, year and scenario, with `w_s` the site weight (`site_weight` as in
`flow_exposure_spec.md` (Weighting), renormalised over the sites present in the
flows file; equal weights under `--unweighted`):

```
I_t = Σ_s w_s · imp_s,t / Σ_s w_s              cohort import per site (kWh/site)
C_t = I_t · LF_s · P_t / 1000                   import cost per site ($/site)
```

`P_t` is the state's regional RRP ($/MWh) for the interval. `LF_s` is the
import loss factor of the state's network tariff, `usage_dlf · import_tlf`,
as the model applies it to the wholesale layer of the bill. Intervals with a
missing flow or price are dropped and counted.

Over any set of intervals `T` (a month, a state's whole sample):

```
IWAP_T = 1000 · Σ_{t∈T} C_t / Σ_{t∈T} I_t       import-weighted average price ($/MWh)
```

IWAP is a ratio of sums, so a period with more import counts for more.
Negative prices are included and lower IWAP.

The saving against No CER for a scenario `k` over the same set of intervals is

```
S_k,T = IWAP_NoCER,T − IWAP_k,T                 ($/MWh; positive = cheaper import)
```

IWAP alone can mislead: a scenario that removes low-price imports (Solar
Only removes midday imports) can raise IWAP while lowering import cost. Import
volume and import cost are therefore reported alongside it.

### TOU periods

Every `full_run_*` config uses one network tariff per state for all years:
Ausgrid EA025 (NSW), Energex 6900 (QLD), SAPN RTOU (SA) and United Energy
URSTOU (VIC). The script reads the tariff ids from each scenario's config in
`model/configs/` and stops if the scenarios disagree, then loads the tariffs
from `model/tariffs/` with `load_network_tariff`.

Each hour of the day is assigned a period from the tariff's import
(`usage_tariffs`) rates:

- **Peak**: every hour at the highest rate.
- **Off-peak**: every hour at the lowest rate, except where the lowest rate
  also covers the tariff's export-penalty window (hours with a negative
  `feedin_tariffs` rate); then only the penalty window is off-peak.
- **Shoulder**: every other hour.

The export-penalty exception applies to NSW only. EA025 has one low rate from
21:00 to 15:00, which would put overnight and midday imports in one period;
EA029's 10:00–15:00 penalty window marks the midday period, which several
NSW retail plans price separately. The resulting periods (local time):

| State | Peak | Shoulder | Off-peak |
|---|---|---|---|
| NSW | 15–21 | 21–10 | 10–15 |
| QLD | 16–21 | 21–11 | 11–16 |
| SA | 6–10, 16–24 | 0–6 | 10–16 |
| VIC | 16–21 | 21–11 | 11–16 |

None of the four tariffs has a weekend variant; the script stops if one
does. An interval takes the period of its start instant in the state's local
time (`model.core.localtime.local_wall_clock`), as tariff billing does.

### Period split of the saving

For a set of intervals, let `v_k,p` be the share of scenario `k`'s import in
period `p` and `π_k,p` its IWAP within that period, so
`IWAP_k = Σ_p v_k,p · π_k,p`. The saving of scenario `k` against comparator
`c` splits exactly, with midpoint weights `v̄_p = (v_c,p + v_k,p)/2` and
`π̄_p = (π_c,p + π_k,p)/2`:

```
S = IWAP_c − IWAP_k
  = Σ_p (v_c,p − v_k,p) · π̄_p            between periods: import moved to other periods
  + Σ_p v̄_p · (π_c,p − π_k,p)            within periods: import at cheaper times in a period
```

The between-period term is what a TOU tariff already prices: the customer
pays the cheaper period's rate. The within-period term, and each period's
`π_c,p − π_k,p`, is the reduction in the retailer's cost of supplying a
period's energy, which could be kept or passed on as a lower period rate.
Where one scenario has no import in a period, `π̄_p` is the other scenario's
`π_p` and that period's within term is zero.

Pairs: Solar Only, Self Consumption and each battery plan against No CER;
each battery plan against Self Consumption.

### Observations

A state-month is the unit for distributions: four states over 2021–2025
give 240 per scenario. Months are assigned from the interval start
(`datetime_aest − 5 min`). Each state-month counts once, with no `state_mix`
weighting. A state-month IWAP is a cohort ratio rather than a mean of site
ratios, so sites with very little import in a month do not produce outliers.

The paired saving `S_k,m` for each state-month removes the price level that
all scenarios share in that state and month, so its spread is narrower than
the spread of IWAP itself.

### Tiers

- **Per state, all years pooled.** IWAP and saving over every interval
  loaded for the state. Import and cost per site per year are the sums
  divided by the number of years loaded.
- **National.** `IWAP_national = Σ_s state_mix[s] · IWAP_s`, renormalised
  over the states present; the same weighting applies to the saving, import
  and cost per site per year, and to the per-period IWAP, period shares and
  both terms of the period split. Produced only when weighted and more than one
  state is present.

## Outputs

Under `analysis/output/import_price/` (or `--output-dir`).

### Chart 1 — IWAP by scenario (`{STATE}_all_years/` and `national/`)

`iwap_columns.png`: one column per scenario, height IWAP ($/MWh), in the
scenario's colour. Each column is labelled with its IWAP in $/MWh and, in
brackets, its import (MWh/site/yr), which has no axis on the chart. Battery
scenarios carry the battery size in their tick label, e.g. "Amber, 20 kWh".
The tier, years, battery size and weighting are noted in grey in the top left
of the axes.

### Chart 2 — state-month IWAP distribution (`state_months/`)

`iwap_box.png`: one box per scenario over its state-month IWAPs. Boxes span
the interquartile range with the median marked; whiskers reach the furthest
point within 1.5 × IQR, with points beyond drawn individually; the mean is
marked with a white diamond. Box face in the scenario's colour.

### Chart 3 — state-month saving against No CER (`state_months/`)

`iwap_saving_box.png`: as chart 2, over `S_k,m` for every scenario except
No CER, with a line at zero. y-axis "Saving against No CER ($/MWh)"; the
footnote gives the definition of the saving.

### Chart 4 — IWAP by state and year (`state_year/`)

`iwap_by_state_and_year.png`: a grid with one row per state and one column
per year, each panel a column chart of IWAP per scenario as in chart 1,
labelled with IWAP only (no unit; each row's y-axis label gives $/MWh)
and without tick labels; one legend above the grid
names the scenarios. All panels share the y-axis. Each state-year's IWAP
is computed over that state-year's intervals.

### Chart 5 — IWAP by TOU period (`{STATE}_all_years/` and `national/`)

`iwap_by_period.png`: columns grouped by period (Peak, Shoulder, Off-peak),
one column per scenario in its colour, each labelled with its period IWAP in
$/MWh and, in brackets, the share of the scenario's import in that period.
Same note in the top left as chart 1.

### Chart 6 — period split of the saving (`{STATE}_all_years/` and `national/`)

`iwap_saving_split.png`: one group per pair (e.g. "Amber, 20 kWh vs No CER"),
three columns: between periods (light blue), within periods (dark blue) and
total saving (the scenario's colour), in $/MWh, each labelled with its value.
The note gives the definition of the saving, since the comparator varies. Same note in the top left as chart 1.

### CSV files

- `state_months/state_months.csv`: one row per (scenario, state, year,
  month): `import_mwh_per_site`, `import_cost_per_site` ($),
  `iwap` ($/MWh), `saving_vs_no_cer` ($/MWh), `n_dropped` (intervals
  dropped for a missing flow or price).
- `{STATE}_all_years/iwap_summary.csv` and `national/iwap_summary.csv`: one
  row per scenario: `years`, `import_mwh_per_site_yr`,
  `import_cost_per_site_yr` ($), `iwap`, `saving_vs_no_cer`.
- `state_year/state_years.csv`: one row per (scenario, state, year) with the
  same columns as the summary.
- `{STATE}_all_years/iwap_by_period.csv` and `national/iwap_by_period.csv`:
  one row per (scenario, period): `import_share`, `iwap`.
- `{STATE}_all_years/saving_split.csv` and `national/saving_split.csv`: one
  row per pair: `scenario`, `comparator`, `between`, `within`, `total`
  ($/MWh), and `within_peak`, `within_shoulder`, `within_offpeak` (each
  period's term) and `diff_peak`, `diff_shoulder`, `diff_offpeak`
  (`π_c,p − π_k,p`).

## Interface

Run from the project root:

```bash
python3 analysis/scripts/import_price_analysis.py --data-root "/Volumes/Manual/Thesis Project"
```

Arguments `--data-root`, `--output-dir`, `--states`, `--years`,
`--battery-kwh`, `--core-only`, `--unweighted` and `--weights-file` behave as
in `cap_hedge_analysis.py`. It imports `discover_scenarios`, `load_flows`,
`get_prices`, `compute_site_weight_for_scenario` and `RRP_COL` from
`flow_exposure_analysis.py`, `load_weights_config` from
`financial_analysis.py`, `build_scenarios` from `cap_hedge_analysis.py`, and
`load_network_tariff` and `local_wall_clock` from `model.core`, with the
repository root added to `sys.path`.
Each flows file is read once.

## Verification

Printed as `CHECK` lines on every run:

- Per (scenario, state): IWAP recomputed from the state-month rows
  (`Σ cost / Σ import`) equals the state tier's IWAP.
- No CER's saving is zero in every row.
- Per tier and pair: `between + within` equals the total saving.
- Per tier and scenario: period shares sum to 1.

The period table per state is printed with the tariff id it came from.

The loading table prints, per scenario and state-year, sites present and
used, intervals dropped, import (MWh/site) and IWAP.
