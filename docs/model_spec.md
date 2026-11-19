# Coordination model — Specification

## Purpose and scope

The coordination model takes cleaned household energy data (EDP, 5-minute) and
NEM wholesale prices and models home battery operation under a set of
coordination strategies, producing per-site bills at three settlement layers
(wholesale, network, retail) and the household and retailer views of them.

This document owns everything shared across components: the module layout, sign
conventions and units, the scenario and tariff configuration schema, local
time, price construction, the data context, the output types, the
orchestrator, the CLI and the output format. Each dispatch component's
formulation is in its own specification (see Components).

| Document | Covers |
|---|---|
| `edp_data_spec.md` | EDP site data pipeline and the clean trace format |
| `nem_price_spec.md` | NEM price pipeline and the clean price format |
| `predispatch_data_spec.md` | AEMO pre-dispatch forecast store |
| `perfect_foresight_lp_spec.md` | `perfect_foresight` |
| `perfect_foresight_advanced_spec.md` | `perfect_foresight_advanced` |
| `perfect_foresight_custom_spec.md` | `perfect_foresight_custom` |
| `vpp_sc_event_spec.md` | `vpp_sc_event` and the event classifier |
| `vpp_tou_event_spec.md` | `vpp_tou_event` |
| `rolling_optimisation_spec.md` | the rolling loop, `RollingConfig`, `rolling_perfect_foresight_lp` |
| `rolling_perfect_foresight_lp_advanced_spec.md` | `rolling_perfect_foresight_lp_advanced` |
| `rolling_vpp_event_spec.md` | `rolling_vpp_sc_event`, `rolling_vpp_tou_event` |
| `forecast_providers_spec.md` | price forecast providers |
| `usage_forecast_spec.md`, `solar_forecast_spec.md` | site load and solar forecasts |

---

## Module layout

```
model/
  main.py                 CLI: load and expand YAML, validate, run, write outputs
  orchestrate.py          validation, routing, per-site runners, workers, writers
  core/
    scenario.py           ScenarioConfig, BatterySpec, RollingConfig,
                          UsageForecastConfig, SolarForecastConfig,
                          ROLLING_COORDINATION_TYPES, VPP_COORDINATION_TYPES
    tariff.py             NetworkTariff, RetailTariff, TariffPeriod, BlockTier,
                          BlockTariffWindow, ZeroImportReward, CriticalRate,
                          ChargeWindow, VppConfig, period_rate_series, loaders
    localtime.py          STATE_TIMEZONE, local_wall_clock()
    prices.py             shared price construction for billing and dispatch
    market_prices.py      Market Price Cap by financial year, Market Floor Price
    forecast.py           ForecastProvider contract for the rolling loop
    paths.py              Paths, default_paths()
    data_context.py       DataContext
    site_result.py        SiteTraces, SiteBill, VppTraces
  components/
    ingest.py             the only component that reads input data from disk
    self_consumption.py   rule-based dispatch; also run_sc_with_battery()
    billing.py            compute_bill, compute_monthly_bills
    perfect_foresight_lp.py, perfect_foresight_lp_advanced.py,
    perfect_foresight_custom.py
    vpp_event_classifier.py, vpp_sc_event.py
    rolling_optimisation.py, rolling_advanced.py
    rolling_vpp_event.py, vpp_window_planner.py, vpp_declaration.py,
    vpp_shadow.py, vpp_operator_lp.py, vpp_operator_lp_highs.py
  forecast/               price, usage and solar forecast providers
  configs/                scenario YAML files
  tariffs/                tariff YAML files
  dev/                    single-site inspection, check and fitting scripts
  scripts/                one-off output utilities
  tests/                  pytest suite and fixtures
  outputs/                empty package
```

### Design principles

- Components are functions of their inputs. They receive only the data they
  need, not a global context. `DataContext` organises ingest output; it is not
  passed wholesale to components.
- All configuration and output dataclasses are `frozen=True`. The pandas
  objects inside `DataContext` are read-only by convention; a component that
  needs to modify one takes a copy.
- The orchestrator is the only place that knows which components run, in what
  order, for a coordination type.
- Billing and every optimiser build prices through `core/prices.py`, so an
  optimiser optimises against exactly the rates billing charges.
- One `ScenarioConfig` is one (state, year, cohort, battery) combination.
  Multi-state, multi-year and multi-battery runs are expanded by `main.py`;
  components see one state-year at a time.
- Energy is kWh per interval throughout. Wholesale prices stay in $/MWh until
  `prices.py` converts them.

---

## Sign conventions and units

This section is the canonical statement; component docstrings point here.

### Energy

- `import_kwh` and `export_kwh` are non-negative. Direction is encoded by the
  field, not the sign.
- Flow decomposition fields are non-negative, except `sol_to_home_kwh`, which
  is negative when the inverter draws power overnight (parasitic draw). The
  inverter then acts as additional load; `gross_load = pv_site_net +
  ac_load_net` already reflects it.

### Money

- Costs are stored positive and credits negative, so every total is a straight
  sum of its fields.
- Every feed-in field uses `feedin = −Σ(export_kwh × rate)`, where the export
  rate is the value earned per exported kWh. A positive rate (feed-in tariff,
  reward) gives a negative stored value; a negative rate (an export penalty such
  as Ausgrid EA029, or a negative spot price) gives a positive one.
- Negative wholesale prices flow through unchanged: exporting at a negative
  price costs the customer, importing at one pays them. The exception is a
  `mixed` plan's export credit, which `export_floor_c_per_kwh` can floor.
  The wholesale and network layers are unaffected by the floor.
- `usage_other_c_per_kwh` adds to the import rate and `feedin_other_c_per_kwh`
  to the export rate.

### Loss factors

- Applied to the wholesale and network layers only, never the retail layer.
  The exception is a `mixed` plan's export credit, which is a share of the
  loss-adjusted wholesale export rate, or of the loss-adjusted wholesale and
  network export rates (`export_share_basis`).
- Import factor `= usage_dlf × import_tlf`; export factor
  `= export_dlf × export_tlf`.

### Units

| Quantity | Stored | Conversion |
|---|---|---|
| Wholesale price | $/MWh | `÷ 1000` → $/kWh (`MWH_TO_KWH`) |
| Tariff rates | c/kWh | `÷ 100` → $/kWh (`C_TO_DOLLARS`) |
| Daily charges | c/day | `÷ 100` → $/day |

### Timestamps and period mapping

- Every index is interval-end AEST (`Etc/GMT-10`, fixed UTC+10).
- Tariff periods, windows and weekday/weekend selection are looked up at the
  interval-start instant (timestamp − interval length), converted to the site's
  local time (see Local time).

---

## Scenario configuration

`model/core/scenario.py`. All frozen dataclasses with `__post_init__`
validation.

### ScenarioConfig

| Field | Type | Default | Meaning |
|---|---|---|---|
| `state` | `'NSW' \| 'QLD' \| 'SA' \| 'VIC'` | required | site state; also selects the local timezone and NEM region |
| `year` | `int` | required | calendar year |
| `cohort` | `'no_cer' \| 'solar' \| 'solar_battery'` | required | |
| `network_tariff_id` | `str` | required | file stem in the tariffs directory |
| `retail_tariff_id` | `str` | required | file stem in the tariffs directory |
| `coordination_type` | see below | `None` | required for `solar_battery`, `None` otherwise |
| `battery_spec` | `BatterySpec` | `None` | required for `solar_battery`, `None` otherwise |
| `battery_name` | `str` | `None` | set when a YAML expands a `battery_specs` list; part of `scenario_id` |
| `rolling` | `RollingConfig` | `None` | required for the four rolling types, `None` otherwise |
| `export_limit_kw_per_phase` | `float` | `None` | per-phase export limit, ≥ 0; `None` is unlimited |
| `opt_factor` | `float` | 1.0 | in (0, 1]; `perfect_foresight_custom` only |
| `apply_export_order_nudge` | `bool` | `False` | `perfect_foresight_advanced`, `vpp_tou_event`, `rolling_perfect_foresight_lp_advanced` and `rolling_vpp_tou_event` only (`EXPORT_ORDER_NUDGE_COORDINATION_TYPES`) |
| `op_retail_tariff_id` | `str` | `None` | operator tariff; required for the four VPP types, `None` otherwise |

`__post_init__` enforces the cohort rules above, rejects an unknown
`coordination_type`, the `rolling` and `op_retail_tariff_id` presence rules,
`export_limit_kw_per_phase ≥ 0`, `opt_factor ∈ (0, 1]`, and
`apply_export_order_nudge` under a type that does not apply it.

`scenario_id` is `{state}_{year}`, or `{state}_{year}_{battery_name}` when
`battery_name` is set. It names the output directory.

### Coordination types

| Type | Plan | Dispatch | Spec |
|---|---|---|---|
| `self_consumption` | any | rule-based self-consumption | this document |
| `perfect_foresight` | wholesale-exposed or TOU | full-year LP | `perfect_foresight_lp_spec.md` |
| `perfect_foresight_advanced` | TOU, including block tiers, zero-import reward, critical rates; `mixed` | full-year LP with tier variables | `perfect_foresight_advanced_spec.md` |
| `perfect_foresight_custom` | any | greedy shift heuristic | `perfect_foresight_custom_spec.md` |
| `vpp_sc_event` | plan with a `vpp:` block | operator LP in events, self-consumption between | `vpp_sc_event_spec.md` |
| `vpp_tou_event` | TOU plan with a `vpp:` block | operator LP in events, TOU LP between | `vpp_tou_event_spec.md` |
| `rolling_perfect_foresight_lp` | wholesale-exposed | rolling LP on forecasts | `rolling_optimisation_spec.md` |
| `rolling_perfect_foresight_lp_advanced` | TOU or `mixed` | rolling advanced LP on forecasts | `rolling_perfect_foresight_lp_advanced_spec.md` |
| `rolling_vpp_sc_event` | plan with a `vpp:` block | rolling operator LP, self-consumption between | `rolling_vpp_event_spec.md` |
| `rolling_vpp_tou_event` | TOU plan with a `vpp:` block | rolling operator LP, rolling TOU LP between | `rolling_vpp_event_spec.md` |
| `self_hedging` | | not built (see below) | |

`ROLLING_COORDINATION_TYPES` and `VPP_COORDINATION_TYPES` in `scenario.py`
name the rolling and VPP sets once, so configuration validation, orchestrator
validation and routing test the same sets. The rolling VPP types are in both.
`MIXED_COORDINATION_TYPES` names the types a `mixed` plan may run under, with
`None` standing for the `no_cer` and `solar` cohorts.

### BatterySpec

| Field | Default | Validation | Meaning |
|---|---|---|---|
| `capacity_kwh` | required | > 0 | usable capacity |
| `backup_reserve_pct` | required | 0–100 | minimum SOC as a percentage of capacity |
| `power_kw` | required | > 0 | AC-side charge and discharge limit |
| `charge_efficiency` | required | (0, 1] | AC → DC |
| `discharge_efficiency` | required | (0, 1] | DC → AC |
| `c_deg_per_kwh` | 0.005 | ≥ 0 | throughput cost, $/kWh, on charge and discharge |

`power_kw` is the AC-side limit at the battery inverter. The per-interval AC
cap is `power_kw × interval_hours`, applied to AC flows with no efficiency
factor. DC-side headroom, `(capacity − soc) / η_c` for charging and
`(soc − soc_min) · η_d` for discharging, applies jointly with it. SOC is stored
on the DC side.

`c_deg_per_kwh` appears in optimiser objectives only, never in a bill. Any
cycle earning less than `2 × c_deg` per kWh round trip is not worth doing. The
default of 0.005 does two further jobs:

- With a negative critical import rate, it is profitable to charge from the
  grid and discharge to load in the same interval purely to increase imports.
  0.005 $/kWh makes that unprofitable against the GloBird −5 c/kWh critical
  import rate; a more negative rate may need a larger value.
- The TOU export-order nudge in `perfect_foresight_lp_advanced` relies on
  `c_deg > 0` to prevent degenerate simultaneous import and export at
  zero-spread intervals, so it must not be 0.0 on TOU plans.

### RollingConfig

Specified in `rolling_optimisation_spec.md`, Configuration, with
`UsageForecastConfig` in `usage_forecast_spec.md` and `SolarForecastConfig` in
`solar_forecast_spec.md`. All three live in `scenario.py` because `core` must
not import from the `forecast` package.

---

## Tariffs

`model/core/tariff.py`.

### The settlement model

Every scenario has a `NetworkTariff` and a `RetailTariff`. Billing always
computes three layers over the same physical flows, because the flows incur
wholesale and network costs whether or not the customer sees them:

- **Wholesale** — spot price × loss-adjusted energy.
- **Network** — network tariff × loss-adjusted energy, plus the network daily
  charge.
- **Retail** — what the retailer charges the customer.

`household_total` and `retailer_margin` are views onto the layers (see
SiteBill).

GST: network and retail rates in the tariff files include GST, except feed-in
rates, which are GST-free because households are not GST-registered. The
wholesale layer is spot × loss factor, without GST. A `wholesale_exposed` plan
with `gst_on_wholesale_import: true` charges the household GST on the spot
energy charge of every import, negative prices included: billing adds
`GST_RATE` (0.10, `model/components/billing.py`) × the interval's wholesale
usage cost to `retail_usage_cost`. Exports carry no GST. Dispatch prices
(`build_price_vectors`, `PriceStackOffsets`) leave this GST out, so the
optimiser's import price stays at or above its export price across the market
price range (`validate_price_ordering`).

### NetworkTariff

| Field | Meaning |
|---|---|
| `name`, `state` | identifier; NEM region |
| `usage_tariffs` | import periods (weekday, or all days) |
| `usage_tariffs_weekend` | optional weekend import periods |
| `feedin_tariffs` | export periods; may carry a penalty (EA029) |
| `daily_c_per_day` | sum of the named components under `daily_charges_c_per_day` |
| `usage_dlf`, `export_dlf` | distribution loss factors (import typically > 1, export typically < 1) |
| `import_tlf`, `export_tlf` | transmission loss factors |

Properties `import_loss_factor` and `export_loss_factor` combine them.

Network demand charges and seasonal network tariffs are not modelled.

### RetailTariff

Three types:

| Type | Imports | Exports | Household pays |
|---|---|---|---|
| `tou` | TOU period rates + `usage_other` | feed-in period rates + `feedin_other` | retail layer |
| `wholesale_exposed` | spot (wholesale layer) + network + `usage_other` | spot + network + `feedin_other` | all three layers |
| `mixed` | TOU period rates + `usage_other` | `max(floor, wholesale_export_share × V)` + `feedin_other`, with `V` set by `export_share_basis` (Price construction) | retail layer |

| Field | Meaning |
|---|---|
| `name`, `state`, `type` | |
| `daily_c_per_day` | sum of `daily_charges_c_per_day` components (supply charge, subscription, or a negative participation credit) |
| `usage_tariffs`, `usage_tariffs_weekend` | TOU import periods; empty for `wholesale_exposed` |
| `feedin_tariffs` | TOU export periods; empty for `wholesale_exposed` and `mixed` |
| `usage_other_c_per_kwh`, `feedin_other_c_per_kwh` | sums of named per-kWh adder components |
| `block_import_tiers`, `block_export_tiers` | `tuple[BlockTariffWindow]` or `None` |
| `zero_import_reward` | `ZeroImportReward` or `None` |
| `critical_export_rate`, `critical_import_rate` | `CriticalRate` or `None`; `tou` plans only |
| `vpp` | `VppConfig` or `None` |
| `wholesale_export_share` | in [0, 1]; required on `mixed`, rejected otherwise |
| `export_share_basis` | `'wholesale'` or `'wholesale_and_network'`: the export value `V` the share applies to; required on `mixed`, rejected otherwise |
| `export_floor_c_per_kwh` | floor on `wholesale_export_share × V`, c/kWh, applied every interval, or `None` for no floor; the key is required on `mixed` (its value may be `null`) and rejected otherwise |
| `gst_on_wholesale_import` | bool; billing charges GST on spot imports (The settlement model). Required on `wholesale_exposed` (`true` on the Amber plans, `false` on operator tariffs), rejected otherwise; `False` on other types |

`has_advanced_features` is true when any of the six optional features is
present; `has_vpp` when `vpp` is. A `mixed` tariff may not carry advanced
features: the advanced price construction and billing distinguish only `tou`
and `wholesale_exposed`, so the combination has no defined meaning. An
optimiser dispatches a `mixed` plan against its dispatch prices, which differ
from its billed import rate (see Mixed-plan dispatch prices under Price
construction). `validate_scenario` accepts a `mixed` plan only for the
coordination types listed under Validation.

### Periods

`TariffPeriod(start_hour, end_hour, c_per_kwh)`: `start_hour` in 0–23
inclusive, `end_hour` in 1–24 exclusive, local hours as published. The loader
checks that each period list tiles `[0, 24)` with no gap or overlap.

`period_rate_series(periods, index, interval_hours, state, weekend_periods=None,
lwc=None)` maps periods onto an index at the local interval-start hour,
selecting `weekend_periods` on local Saturdays and Sundays. It serves every
time-of-day lookup (network and retail, import and export) and raises if any
interval matches no period. `lwc` accepts a precomputed `LocalWallClock` to
avoid repeating the timezone conversion.

### Advanced features

| Type | Fields | Rules |
|---|---|---|
| `BlockTier` | `threshold_kwh` (cumulative daily kWh; `inf` for the last), `rate_c_per_kwh` | threshold > 0 |
| `BlockTariffWindow` | `start_hour`, `end_hour`, `tiers`, `applies_weekends = True` | ≥ 2 tiers, ascending thresholds, last `inf`; windows in a list may not overlap; import rates non-decreasing and export rates non-increasing (convexity, so the LP fills tiers in order) |
| `ZeroImportReward` | `start_hour`, `end_hour`, `reward_c_per_day`, `tolerance_kwh_per_hour = 0` | reward > 0; `tolerance_kwh = tolerance_kwh_per_hour × window_hours` |
| `CriticalRate` | `threshold_dollars_per_mwh`, `rate_c_per_kwh` | rate finite, may be negative; the export rate applies at price > threshold, the import rate at price < threshold, replacing that side's TOU or tier rate; export threshold > import threshold; the retired keys `stress_export_amplifier` / `stress_import_amplifier` are rejected |
| `ChargeWindow` | `start_hour`, `end_hour` | integers, `0 ≤ start < end ≤ 24` |

With `applies_weekends: false`, weekend intervals in the window fall back to the
period rates. A block import window overlapping a non-zero TOU usage period
triggers a `UserWarning`: the block rate replaces the TOU rate there. Block
tiers on a `wholesale_exposed` tariff are accepted with a warning and treated as
a retail overlay.

How each feature enters dispatch and billing is in
`perfect_foresight_advanced_spec.md`.

### VppConfig

| Field | Default | Meaning |
|---|---|---|
| `exp_reward_c_per_kwh` | required | household reward per kWh of forced export, ≥ 0 |
| `imp_reward_c_per_kwh` | 0 → 0.01 | household reward per kWh of forced import, ≥ 0 |
| `kwh_limit` | required | annual cap on operator-dispatched energy, > 0 |
| `limit_applies_to` | required | `export_only` or `import_and_export` |
| `default_mode` | `self_consumption` | `tou` or `self_consumption`; validated but not read (between-event behaviour is set by the coordination type) |
| `event_window_hours` | required | operator control window before a spike, > 0 |
| `vpp_threshold_dollars_per_mwh` | required | spike threshold, > 0 |
| `curtail_c_per_kwh` | `None` | event curtailment cost, ≥ 0; `None` disables curtailment |
| `min_export_price_dollars_per_mwh` | `None` | forced-export threshold, ≥ 0 |
| `min_export_price_mode` | `fixed` | `fixed`, `based_on_this_year`, `based_on_last_year` (rejected at validation) |
| `no_grid_charge_windows` | `None` | `tuple[ChargeWindow]`; YAML accepts one `[start, end]` pair or a list of pairs |

The loader replaces an absent or zero `imp_reward_c_per_kwh` with 0.01 c/kWh,
which keeps a strict preference for natural over forced import in the event LP.
Tariffs with no real import reward state a nominal value themselves.

The minimum export price rules (required under `fixed`; a floor under
`based_on_this_year`) apply to all four VPP types and are specified in
`rolling_vpp_event_spec.md`, Minimum export price.

### Operator tariff

VPP scenarios name a second retail tariff, `op_retail_tariff_id`: a
`wholesale_exposed` plan with zero adders, zero daily charges and
`gst_on_wholesale_import: false`. The operator
price vectors (loss-adjusted wholesale plus network) are built from it.

### YAML schema

- Each file declares `tariff_class: network` or `tariff_class: retail`, and a
  retail file declares `type`. The loader rejects the wrong class.
- Daily charges and per-kWh adders are mappings of named components, summed by
  the loader so the breakdown stays in the file.
- Period lists accept `[start_hour, end_hour, c_per_kwh]` tuples or mappings.
- A `wholesale_exposed` file may not define period rates and must define
  `gst_on_wholesale_import`. A `mixed` file may not
  define `feedin_tariffs`, and must define `wholesale_export_share`,
  `export_share_basis` and `export_floor_c_per_kwh` (the last may be `null`).

Tariff files are in `model/tariffs/`. The current set is named
`{STATE}_{plan}_{year}.yaml`, covering for each state a network tariff, Amber
(wholesale-exposed), AGL Battery Rewards (TOU and VPP), Energy Australia
Battery Ease (TOU and VPP), GloBird ZeroHero (advanced TOU), Origin Battery
Starter, Lite and Maximiser, Tesla Electric (`mixed`) and an operator tariff.
`NSW_tesla_electric_mixed_netshare_2026` and
`NSW_tesla_electric_mixed_nofloor_2026` are sensitivity variants of the NSW
Tesla Electric plan with `export_share_basis: wholesale_and_network` and no
export floor respectively. Unprefixed 2024 and 2025 files are the earlier NSW
set. Test fixtures are in
`model/tests/fixtures/tariffs/`.

---

## Local time

`model/core/localtime.py`.

The model runs on one index, AEST. Wholesale prices and physical flows are
market-time quantities and stay in AEST. Tariff clock features (network and
retail periods, block windows, zero-import windows, no-grid-charge windows, the
export-order nudge) are published in local time, which is state-dependent and
observes DST: QLD equals AEST year-round; NSW and VIC share a DST offset; SA is
a half-hour behind AEST with its own DST, so SA local time never aligns to an
AEST hour boundary.

```python
STATE_TIMEZONE = {'NSW': 'Australia/Sydney', 'VIC': 'Australia/Melbourne',
                  'QLD': 'Australia/Brisbane', 'SA': 'Australia/Adelaide'}

local_wall_clock(index, state, interval_hours) -> LocalWallClock
```

`local_wall_clock` converts each interval-start instant to the state's zone and
returns `hour`, `minute`, `dayofweek`, `is_weekend` and `date` (normalised local
dates). It raises on an unknown state. Every clock feature calls it.

The local clock always comes from `scenario.state`, never from a tariff's
declared state, so a cross-state tariff sensitivity run still uses the site's
real clock.

Converting a fixed-offset instant to a local zone is always well defined; there
is no ambiguous or non-existent time in that direction. Grouping by local
`date` gives DST transition days their true 23 or 25 hours (276 or 300
intervals) with no special case. No block, zero-import or no-grid-charge window
is assumed to straddle a DST transition.

The annual bill's daily-charge day count uses AEST interval-start dates. The
monthly bills count local dates, so every month's day count is right, and the
monthly counts sum exactly to the annual one.

---

## Market price limits

`model/core/market_prices.py` holds the Market Price Cap by financial year
(determined values only) and the Market Floor Price (−$1,000/MWh). They are used
to range-check forecast data at ingest and, as the widest bounds across all
years, to prove price ordering in `prices.validate_price_ordering` and
`prices.validate_mixed_price_ordering`. Nothing
clips a price. A year missing from the table raises rather than falling back to
a neighbour.

---

## Price construction

`model/core/prices.py`. Billing and every optimiser build prices here. On
`tou` and `wholesale_exposed` plans an optimiser optimises against exactly the
rates billing charges, apart from the GST billing adds on spot imports under
`gst_on_wholesale_import` (The settlement model); on a `mixed` plan the
dispatch import price differs from the billed rate by design (Mixed-plan
dispatch prices). It owns every
unit conversion and every loss-factor application. `state` is always
`scenario.state`.

`build_layer_rates(index, prices_mwh, network_tariff, retail_tariff, state,
interval_hours) -> LayerRates` returns six $/kWh arrays, value earned per kWh on
the export side:

| Layer | Import | Export |
|---|---|---|
| Wholesale | `RRP/1000 × import_lf` | `RRP/1000 × export_lf` |
| Network | network usage rate × `import_lf` | network feed-in rate × `export_lf` |
| Retail, `tou` | TOU usage rate + `usage_other` | feed-in rate + `feedin_other` |
| Retail, `wholesale_exposed` | `usage_other` | `feedin_other` |
| Retail, `mixed` | TOU usage rate + `usage_other` | `max(floor, share × V)` + `feedin_other` |

On `mixed`, `V` is the wholesale export rate under `export_share_basis:
wholesale`, and the wholesale plus network export rates under
`wholesale_and_network`. `share` is `wholesale_export_share` and `floor` is
`export_floor_c_per_kwh / 100`, or no floor when it is `null`. The floor
applies to each interval separately. `mixed_export_rate` is the single
definition of this export rate; `build_layer_rates` and the mixed dispatch
stack both call it.

`build_price_vectors(...) -> (p_imp, p_exp)` returns the prices an optimiser
dispatches against: all three layers on `wholesale_exposed`, the retail layer
alone on `tou`, and the mixed-plan dispatch prices on `mixed`.

### Mixed-plan dispatch prices

The plan's retailer holds dispatch and optimises from the household's side,
against the export rate the household is credited and an import price that
adds the positive part of the loss-adjusted spot price to the billed rate:

```
p_imp[t] = tou[t] + usage_other + max(0, RRP[t]) / 1000 × import_lf      $/kWh
p_exp[t] = max(floor, share × V[t]) + feedin_other                       $/kWh
```

The spot term raises the bar for grid charging: importing in order to export
later is chosen only when it pays the household at the higher import price.
Dispatch against the billed import rate alone would import at the fixed TOU
rate to export into a price spike, a transfer from the retailer larger than
the household's gain. Only the positive part of spot enters `p_imp`: with
negative spot added in full, `p_imp` would fall below a zero-floored `p_exp`
once `RRP < −1000 × (tou + usage_other) / import_lf`, and the LP would import
and export at once.

Billing uses `build_layer_rates`, so on a `mixed` plan the LP's price cost
(`Σ_t imp[t] × p_imp[t] − exp[t] × p_exp[t]`, its objective less degradation
and curtailment costs) exceeds the bill's dispatch part by
`Σ_t max(0, RRP[t]) / 1000 × import_lf × import_kwh[t]`.

`build_mixed_dispatch_stack(index, network_tariff, retail_tariff, state,
interval_hours, lwc=None, price_floor_mwh=MARKET_PRICE_FLOOR_MWH,
price_cap_mwh=MARKET_PRICE_CAP_MWH) -> MixedDispatchStack` splits the stack into the parts that depend only on the
clock and the parts that depend on RRP, as `build_price_stack_offsets` does
for `wholesale_exposed`. It raises `ValueError` for any other retail type.
`MixedDispatchStack` holds:

| Field | Contents |
|---|---|
| `import_base` | `tou[t] + usage_other`, $/kWh per interval |
| `import_loss_factor`, `export_loss_factor` | from the network tariff |
| `network_export` | loss-adjusted network export rate, $/kWh per interval; `mixed_export_rate` includes it in `V` only under `wholesale_and_network` |
| `retail_tariff` | the plan, which supplies the share, basis, floor and `feedin_other` to `mixed_export_rate` |

`MixedDispatchStack.window_price_vectors(rrp_mwh, start) -> (p_imp, p_exp)` is
the single implementation of the formulas above. `build_price_vectors` calls it
once over the whole period with the settled RRP. The rolling advanced planner
calls it once per window with the forecast RRP.

`validate_mixed_price_ordering(stack, price_floor_mwh, price_cap_mwh)` proves
`p_imp ≥ p_exp` in every interval for any RRP in the market range, and
`build_mixed_dispatch_stack` calls it before returning. The spread
`p_imp − p_exp` is piecewise linear in RRP. `p_imp` is convex with its one bend
at RRP = 0. `p_exp` is also convex, so its bend, where `share × V` meets the
floor, makes the spread concave, and a concave bend is never a minimum. The
check therefore evaluates the spread at the market floor, at 0 and at the cap,
per interval, through `window_price_vectors`. It raises `ValueError` naming the
price, the first offending interval and the two prices there. At the Tesla settings (`wholesale`
basis, floor 0) the ordering holds whenever `tou + usage_other ≥ 0` and
`import_lf ≥ share × export_lf`.

### Price functions

| Function | Used by | Specified in |
|---|---|---|
| `PriceStackOffsets`, `build_price_stack_offsets` | rolling wholesale planner, VPP operator | `rolling_optimisation_spec.md` |
| `mixed_export_rate`, `MixedDispatchStack`, `build_mixed_dispatch_stack`, `validate_mixed_price_ordering` | billing (`mixed_export_rate` only), `build_price_vectors`, rolling advanced planner | this document |
| `MIXED_CURTAIL_COST_PER_KWH`, `dispatch_curtail_cost` | advanced LP and rolling advanced planner, through the orchestrator | `perfect_foresight_advanced_spec.md` |
| `validate_price_ordering` | price-stack construction | `rolling_optimisation_spec.md` |
| `build_base_price_vectors_advanced`, `build_lp_price_vectors_advanced` | advanced LP, rolling advanced planner | `perfect_foresight_advanced_spec.md` |
| `critical_masks`, `apply_critical_rates` | advanced LP and rolling planner (`shift_opposite=True`), billing (`shift_opposite=False`) | `perfect_foresight_advanced_spec.md` |
| `block_tier_masks` | advanced LP, billing, `vpp_sc_event` carry-in | `perfect_foresight_advanced_spec.md` |
| `build_zero_import_penalty` | advanced LP only, never billing | `perfect_foresight_advanced_spec.md` |
| `build_tou_export_order_nudge`, `apply_tou_export_order_nudge` | advanced LP and rolling advanced planner, never billing | `perfect_foresight_advanced_spec.md` |
| `check_export_order_nudge_degradation` | advanced LP, rolling advanced planner, `validate_scenario` | `perfect_foresight_advanced_spec.md` |
| `block_window_mask` | advanced LP, billing | `perfect_foresight_advanced_spec.md` |
| `charge_window_mask` | VPP no-grid-charge masks | `vpp_sc_event_spec.md` |

---

## Input data

### Paths

`model/core/paths.py`. `default_paths(project_root=None)` anchors at the project
root (the working directory by default). Passed to `ingest` and
`validate_scenario` explicitly so tests can point at fixtures.

| Attribute | Location | Contents |
|---|---|---|
| `edp_clean_dir` | `data/clean/edp/` | `{year}/edp_clean_{site}_{year}.parquet` |
| `edp_metadata_dir` | `data/processed/edp/` | `{year}_selected_sites_clean.csv` |
| `nem_prices_dir` | `data/clean/nem_prices/` | `{year}_nem_prices.parquet` |
| `tariffs_dir` | `model/tariffs/` | tariff YAML |

The pre-dispatch store is at `data/clean/predispatch/`
(`orchestrate.DEFAULT_PREDISPATCH_STORE`), outside `Paths`.

### ingest

```python
def ingest(scenario, paths, *, progress_callback=None) -> DataContext
```

The only component that reads input data from disk:

1. Loads site metadata for the year and filters it to the scenario state. Every
   selected site in the state is loaded, including sites with an existing
   battery (see `edp_data_spec.md`).
2. Loads each site's clean trace, converts Wh to kWh and renames `_wh` →
   `_kwh`.
3. Loads the state's RRP column from the year's price file, in $/MWh.
4. Checks that every trace and the prices share one index covering the full
   year.
5. Loads both tariffs.

It raises `FileNotFoundError` for a missing file and `ValueError` for missing
columns or misaligned indices. It warns, rather than raising, when prices carry
`price_flag == 1` intervals (NaN prices are passed through; billing raises on
them and `validate_scenario` rejects them first) and when a tariff's state
differs from the scenario's.

### DataContext

`model/core/data_context.py`, frozen.

| Attribute | Meaning |
|---|---|
| `site_traces` | `Mapping[site_id, DataFrame]`, index `datetime_aest` |
| `prices` | `pd.Series`, RRP in $/MWh, index `datetime_aest` |
| `site_metadata` | `DataFrame`, the scenario state's sites |
| `network_tariff`, `retail_tariff` | loaded tariffs |
| `scenario` | the `ScenarioConfig` |

Site trace columns:

| Column | Meaning |
|---|---|
| `pv_site_net_kwh` | net PV per interval; negative under inverter parasitic draw |
| `gross_load_kwh` | household load |
| `ac_load_net_kwh` | load net of PV; `gross_load = pv_site_net + ac_load_net` |
| `pv_original_kwh`, `ac_load_net_original_kwh` | values before infill |
| `pv_flag`, `load_flag` | EDP quality flags, per stream |
| `pv_infill`, `load_infill` | 0 none, 1 short-gap infill, 2 synthetic or same-day-of-week infill |

Site metadata columns: `edp_site_id`, `state`, `inverter_manufacturer`,
`inverter_model`, `inverter_ac_rating_kw`, `subarray_dc_rating_kw`,
`ac_load_net_phases`, `pv_site_net_phases` (both used for export limits),
`annual_generation_kwh`, `avg_daily_gross_load_kwh`.

Dispatch uses `pv_site_net_kwh` as solar and `gross_load_kwh` as load.

---

## Output types

`model/core/site_result.py`, all frozen.

### SiteTraces

Per-interval flows for one site under one strategy, kWh per interval, all on
the `datetime_aest` index. Produced by every dispatch component, consumed by
billing. The full decomposition is kept because some uses (comparisons against
self-consumption, VPP accounting) cannot be reconstructed from import and export
alone. Flows that do not apply to a cohort are all-zero series, not `None`.

| Field | Meaning |
|---|---|
| `site_id` | |
| `import_kwh`, `export_kwh` | grid → site, site → grid; non-negative |
| `battery_soc_kwh` | end-of-interval SOC (DC); zero with no battery |
| `sol_to_home_kwh` | solar to load; negative under parasitic draw |
| `sol_to_bat_kwh`, `sol_to_grid_kwh` | |
| `bat_to_home_kwh`, `bat_to_grid_kwh` | AC side |
| `grid_to_home_kwh`, `grid_to_bat_kwh` | |
| `curtail_kwh` | curtailed solar; non-negative |

Invariants every producer enforces:

```
import_kwh == grid_to_home + grid_to_bat
export_kwh == sol_to_grid  + bat_to_grid
load       ≈  sol_to_home + bat_to_home + grid_to_home
solar      ≈  sol_to_home + sol_to_bat + sol_to_grid + curtail
```

### SiteBill

Dollars for one site over the billed period. Credits negative.

| Field | Meaning |
|---|---|
| `site_id`, `plan_type` | `plan_type` is the retail tariff type |
| `wholesale_usage_cost`, `wholesale_feedin_credit` | spot × loss-adjusted energy |
| `network_usage_cost`, `network_feedin_credit`, `network_daily_charges` | |
| `retail_usage_cost`, `retail_feedin_credit`, `retail_daily_charges` | `retail_usage_cost` includes the GST on spot imports under `gst_on_wholesale_import` |
| `retail_other_credits` | zero-import rewards and VPP per-kWh rewards; ≤ 0 |
| `household_total` | what the customer pays |
| `retailer_margin` | `household_total − wholesale_net − network_net` |

```
wholesale_exposed:  household_total = wholesale_net + network_net + retail_net + retail_other_credits
tou, mixed:         household_total = retail_net + retail_other_credits
```

`retailer_margin` has one formula for every type. On `wholesale_exposed` the
pass-through cancels and it reduces to the retail layer; on `tou` it is the
spread between retail revenue and wholesale plus network cost. Revenue and
network cost include GST and wholesale cost does not, so `retailer_margin`
includes GST collected from the household; it is not the retailer's margin
excluding GST.

### VppTraces

The natural/forced split from a VPP dispatch, produced by `vpp_sc_event` and
`rolling_vpp_event` alongside `SiteTraces`. Non-negative, same index.

| Field | Meaning |
|---|---|
| `n_imp_kwh` | natural import: what self-consumption would have imported; not capped, not rewarded |
| `f_imp_kwh` | forced import: operator force-charge; counts to the cap under `import_and_export`; earns the import reward |
| `n_exp_kwh` | natural export: solar surplus; not capped, not rewarded |
| `f_exp_kwh` | forced export: battery to grid under operator control; counts to the cap; earns the export reward |

```
n_imp_kwh + f_imp_kwh == SiteTraces.import_kwh
n_exp_kwh + f_exp_kwh == SiteTraces.export_kwh
```

---

## Components

| Component | File | Spec |
|---|---|---|
| `ingest` | `components/ingest.py` | Input data, above |
| `self_consumption`, `run_sc_with_battery` | `components/self_consumption.py` | below |
| `compute_bill`, `compute_monthly_bills`, `compute_bill_and_monthly_bills` | `components/billing.py` | below |
| `perfect_foresight_lp` | `components/perfect_foresight_lp.py` | `perfect_foresight_lp_spec.md` |
| `perfect_foresight_lp_advanced` | `components/perfect_foresight_lp_advanced.py` | `perfect_foresight_advanced_spec.md` |
| `perfect_foresight_custom` | `components/perfect_foresight_custom.py` | `perfect_foresight_custom_spec.md` |
| `classify_vpp_events` | `components/vpp_event_classifier.py` | `vpp_sc_event_spec.md` |
| `vpp_sc_event` | `components/vpp_sc_event.py` | `vpp_sc_event_spec.md`, `vpp_tou_event_spec.md` |
| `rolling_optimisation`, `WindowPlanner` | `components/rolling_optimisation.py` | `rolling_optimisation_spec.md` |
| `AdvancedWindowPlanner` | `components/rolling_advanced.py` | `rolling_perfect_foresight_lp_advanced_spec.md` |
| `rolling_vpp_event`, `VppWindowPlanner` and helpers | `components/rolling_vpp_event.py` and `vpp_*.py` | `rolling_vpp_event_spec.md` |
| Forecast providers | `model/forecast/` | `forecast_providers_spec.md`, `usage_forecast_spec.md`, `solar_forecast_spec.md` |

### self_consumption

```python
def self_consumption(site_id, solar_kwh, load_kwh, battery_spec,
                     interval_hours=5/60, neg_price_aware=False,
                     price_imp=None, price_exp=None) -> SiteTraces
```

Per interval, in priority order:

1. Solar meets load (`sol_to_home`).
2. Surplus solar charges the battery (`sol_to_bat`), limited by the surplus,
   the AC cap and DC headroom.
3. Remaining solar is exported (`sol_to_grid`).
4. Residual load is met by the battery (`bat_to_home`), limited by the residual,
   the AC cap and the energy above the reserve.
5. Remaining load is imported (`grid_to_home`).

The battery never charges from or exports to the grid, and there is no export
limit. SOC starts at the backup reserve.

One function serves all three cohorts: `no_cer` passes zero solar and no
battery, `solar` passes no battery, `solar_battery` passes both. The no-battery
path is vectorised; the battery path is a sequential loop, because each interval
depends on the previous SOC.

`neg_price_aware=True` (requires both price series) adds two corrections after
the normal logic:

- **Negative export price:** export is curtailed.
- **Negative import price:** all remaining solar is curtailed, battery charging
  is switched from solar to grid at the same AC amount (so the SOC trajectory is
  unchanged), and load is served from the grid.

Only the `perfect_foresight_custom` runner uses it, as the heuristic's starting
point and baseline.

Inputs are validated: NaN or negative load, NaN solar and mismatched indices
raise; negative solar is accepted.

`run_sc_with_battery(solar, load, battery_spec, soc_init, interval_hours) ->
(soc_arr, flows)` runs the same battery logic from an arbitrary starting SOC on
numpy arrays, without `neg_price_aware`. The VPP components use it for gaps,
natural-flow bounds and shadow trajectories.

### billing

```python
def compute_bill(traces, prices, network_tariff, retail_tariff, state,
                 interval_hours=5/60, vpp_traces=None) -> SiteBill
```

Builds the three layers from `prices.build_layer_rates`, applies them to the
flows and forms the two views (see SiteBill). Takes `prices` explicitly, so a
dispatch can be billed against any price series. Daily charges are
`c/day ÷ 100 × n_days`.

On a plan with advanced features it also:

- replaces the retail rate on the critical side of each critical interval
  with the critical rate (plus `usage_other` / `feedin_other`);
- replaces the TOU energy rate in block windows with the tiered daily cost,
  cumulative per local calendar day, keeping `usage_other` / `feedin_other`,
  over the window intervals that are not critical on that side;
- credits the zero-import reward for each qualifying local calendar day into
  `retail_other_credits`.

With `vpp_traces`, it adds the per-kWh VPP rewards to `retail_other_credits`:

```
−(Σ f_exp_kwh × exp_reward + Σ f_imp_kwh × imp_reward)       $/kWh from VppConfig
```

A flat VPP participation credit is a negative daily-charge component on the
tariff and flows through `retail_daily_charges`. Operator cashflows are not
billed here; they are derived downstream from `VppTraces` and the operator
price vectors.

Raises `ValueError` on NaN prices, a misaligned index, or `vpp_traces` against
a tariff with no `VppConfig`.

`compute_monthly_bills(...) -> dict[int, SiteBill]` and
`compute_bill_and_monthly_bills(...) -> (SiteBill, dict[int, SiteBill])` split
the period by local calendar month. All three share one settlement build
(`_build_settlement_base`), reduced per month by masking, so monthly bills use
exactly the annual logic and sum to the annual bill. A month absent from the
traces is omitted. Monthly buckets are keyed by month number alone, which
assumes the traces span a single calendar year.

---

## Orchestration

`model/orchestrate.py`.

### Entry points

| Function | Purpose |
|---|---|
| `validate_scenario(scenario, paths)` | pre-flight checks; raises `ScenarioValidationError` |
| `run_scenario(scenario, paths, *, progress_callback, ingest_progress_callback, n_workers=1, max_sites=None, with_monthly_bills=False, only_sites=None)` | ingest, dispatch, bill; returns `ScenarioResult` |
| `run_rolling_perfect_foresight_lp_site`, `run_rolling_perfect_foresight_lp_advanced_site`, `run_rolling_vpp_event_site` | one site through a rolling component; shared by the bulk workers and the viz scripts |
| `build_rolling_price_provider` | the price forecast provider for a rolling scenario |
| `resolve_min_export_price` | the fleet-wide VPP minimum export price |
| `write_*` | output writers (see Outputs) |

### Routing

`run_scenario` ingests, resolves per-site inputs, keeps `only_sites` (if given) then truncates to `max_sites`, then
routes:

| Coordination type | Runner | Baseline | Notes |
|---|---|---|---|
| `no_cer`, `solar` cohorts, `self_consumption` | in `run_scenario` via `_dispatch_for_site` | none | serial |
| `perfect_foresight` | `_run_perfect_foresight` | self-consumption | terminal SOC from the baseline |
| `perfect_foresight_custom` | `_run_perfect_foresight_custom` | `neg_price_aware` self-consumption | baseline also seeds the heuristic |
| `perfect_foresight_advanced` | `_run_perfect_foresight_advanced` | self-consumption | `build_lp_price_vectors_advanced` |
| `vpp_sc_event`, `vpp_tou_event` | `_run_vpp_sc_event` (`gap_strategy` by type) | self-consumption | events from `classify_vpp_events`; `vpp_traces`, `vpp_event` |
| `rolling_perfect_foresight_lp`, `rolling_perfect_foresight_lp_advanced` | `_run_rolling_perfect_foresight_lp` | self-consumption | `rolling_diagnostics` |
| `rolling_vpp_sc_event`, `rolling_vpp_tou_event` | `_run_rolling_vpp_event` | self-consumption | `vpp_traces`, `rolling_diagnostics`; routed before the other rolling types |
| `self_hedging` | none | | raises `NotImplementedError` after ingest |

Every runner except the plain self-consumption path bills the baseline too, so
`baseline_bills` is available for comparison. VCR against a perfect-foresight
run is assembled downstream from the output files, not inside a run.

The export limit for each site is
`export_limit_kw_per_phase × max(ac_load_net_phases, pv_site_net_phases)`,
converted to kWh per interval (`_resolve_per_site_export_limits`). Every
optimising runner applies it; self-consumption never does.

`_resolve_per_site_batteries` and `_resolve_per_site_solar_scaling` currently
return `scenario.battery_spec` and 1.0 for every site. They are the hooks for
per-site battery sizing and solar scaling.

### Parallelism

Every optimising runner takes `workers`. With `workers > 1` sites run in a
`ProcessPoolExecutor` with the `spawn` context and results are collected as they
complete; `lp_seconds` is then cumulative worker time, which exceeds wall-clock.
The rolling runners build the price forecast provider once per worker process in
an initializer (`_init_rolling_worker`) rather than pickling it per site; the
serial path calls the initializer directly. The plain self-consumption path is
always serial.

### Validation

`validate_scenario` runs before any ingest. Any unexpected exception is wrapped
as `ScenarioValidationError`. All types:

- both tariffs load;
- the metadata file exists and has at least one site for the state;
- every site's trace file exists;
- the price file exists and the state's RRP column has no NaN;
- a tariff whose state differs from the scenario's triggers a warning;
- a `mixed` retail tariff is accepted only for the `no_cer` and `solar`
  cohorts and for `self_consumption`, `perfect_foresight_advanced` and
  `rolling_perfect_foresight_lp_advanced` (`MIXED_COORDINATION_TYPES`); under
  `perfect_foresight_advanced` the mixed dispatch stack is built, which runs
  the ordering proof before ingest (the rolling type builds it inside
  `AdvancedWindowPlanner`).

VPP types (`vpp_sc_event`, `vpp_tou_event` and the rolling pair):

- the retail tariff has a `vpp:` block;
- the minimum export price mode and value rules (`rolling_vpp_event_spec.md`);
- for the perfect-foresight VPP types, the operator tariff is
  `wholesale_exposed` (the rolling VPP types check this when the runner builds
  the operator price stack).

Rolling types: see `rolling_optimisation_spec.md`,
`rolling_perfect_foresight_lp_advanced_spec.md` and `rolling_vpp_event_spec.md`.
The component specs cover the remaining type-specific checks.

`ScenarioConfig.__post_init__` and the YAML loader catch structural errors
earlier (see Scenario configuration and Running scenarios).

### ScenarioResult

Frozen; returned by `run_scenario`.

| Field | Populated for | Meaning |
|---|---|---|
| `scenario` | all | |
| `site_traces`, `site_bills` | all | the dispatched strategy's traces and bills |
| `elapsed_seconds`, `ingest_seconds` | all | wall-clock for the whole call and for ingest |
| `lp_seconds` | optimising types | cumulative dispatch time across sites |
| `baseline_traces`, `baseline_bills` | optimising types | self-consumption comparator |
| `n_workers` | all | worker processes used |
| `vpp_traces` | VPP types | per-site `VppTraces` |
| `vpp_event` | `vpp_sc_event`, `vpp_tou_event` | the shared event mask from `classify_vpp_events` |
| `site_monthly_bills` | when `with_monthly_bills` | `{site: {month: SiteBill}}` for the dispatched strategy only |
| `rolling_diagnostics` | rolling types | per-site `RollingDiagnostics` |
| `min_export_price_resolution` | VPP types | how the minimum export price was reached |

---

## Running scenarios

`model/main.py`.

```bash
python3 -u -m model.main model/configs/example_single_scenario.yaml --save-trace S0008 \
    2>&1 | tee logs/run_$(date +%Y%m%d_%H%M%S).log
```

`-u` keeps output unbuffered so progress reaches `tee` as it happens.

### CLI flags

| Flag | Default | Purpose |
|---|---|---|
| `yaml_path` | required | scenario YAML |
| `--workers N` | 1 | worker processes for the optimising types |
| `--sites ID,ID,...` | all | these sites only, applied before `--max-sites` |
| `--max-sites N` | all | first N sites only |
| `--monthly-bills` | off | also write `bills_01.csv` … `bills_12.csv` |
| `--save-flows` | off | write the flow Parquet files |
| `--save-trace SITE_ID` | none | write one site's full trace |
| `--validate-only` | off | validate every scenario, report every failure, run nothing |
| `--project-root PATH` | cwd | root for input and output paths |
| `--output-dir PATH` | `data/output/runs/<run_name>` | output directory |
| `--tariffs-dir PATH` | `model/tariffs` | tariff directory |

A normal run validates every expanded scenario first and aborts on the first
failure. `--validate-only` checks them all.

### Scenario YAML

```yaml
name: example_single                  # optional; default is the file stem
states: [NSW]
years: [2022]
cohort: solar_battery
coordination_type: perfect_foresight  # required for solar_battery

network_tariff_id: NSW_ausgrid_ea025_ea029_2026
retail_tariff_id: NSW_amber_wholesale_2026
op_retail_tariff_id: NSW_operator_wholesale_2026   # VPP types only

export_limit_kw_per_phase: 5.0        # optional
opt_factor: 1.0                       # perfect_foresight_custom only

battery_spec:
  capacity_kwh: 13.5
  backup_reserve_pct: 10.0
  power_kw: 5.0
  charge_efficiency: 0.95
  discharge_efficiency: 0.95
  c_deg_per_kwh: 0.005                # optional; this is the default

rolling:                              # rolling types only
  forecast_source: predispatch
  horizon_hours: 18.0
```

Rules:

- `states` and `years` are non-empty lists.
- **Tariffs.** The singular form (`network_tariff_id`, `retail_tariff_id`) is
  valid only with one state and one year. Otherwise use mappings,
  `network_tariff_ids: {STATE: {YEAR: id}}` and likewise for retail, covering
  every (state, year).
- **Batteries.** `battery_spec` applies one spec to every scenario, with
  `battery_name = None`. `battery_specs` is a list whose entries each carry a
  unique `name`, and each becomes its own scenario.
- The `rolling:` block is parsed by `_parse_rolling`, which rejects unknown keys
  and requires `forecast_source`. `usage_forecast: {}` or `solar_forecast: {}`
  enables that forecast with its fitted defaults; omitting the key means perfect
  knowledge.
- `apply_export_order_nudge` is `true` or `false` (default `false`).
- `use_rolling_window` and `window_days` are rejected with a pointer to the
  rolling types.
- Any other unrecognised top-level key is rejected (`_KNOWN_TOP_LEVEL_KEYS`).
- One coordination type per YAML.

Scenarios expand as the cross-product `states × years × battery entries` and run
in sequence.

### Outputs

```
data/output/runs/{run_name}/
  run_summary.txt                   run header plus every scenario summary
  scenarios/{scenario_id}/
    bills.csv                       one row per site, every SiteBill field
    summary.txt                     cohort totals, per-site averages, baseline comparison,
                                    minimum export price resolution (VPP)
    rolling_diagnostics.csv         rolling types: one row per site
    bills_01.csv .. bills_12.csv    --monthly-bills; months present only
    flows.parquet                   --save-flows: net import − export per site
    battery_flows.parquet           --save-flows, battery cohorts: net battery AC flow
                                    (sol_to_bat + grid_to_bat − bat_to_home − bat_to_grid)
    vpp_forced_flows.parquet        --save-flows, vpp_sc_event / vpp_tou_event:
                                    f_imp − f_exp at event intervals only
    trace_{site_id}.csv             --save-trace: every SiteTraces field
```

The Parquet files are wide (`datetime_aest` plus one `float32` column per site),
snappy-compressed, positive meaning into the site or battery. The rolling VPP
types write no `vpp_forced_flows.parquet`, because their result carries no
`vpp_event` mask.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | every scenario ran, or every scenario validated under `--validate-only` |
| 2 | the YAML failed to load or expand |
| 3 | a scenario failed validation |

---

## Development scripts

`model/dev/` holds single-site tools outside the production path. They run as
`python3 -m model.dev.<name> SITE_ID YEAR [options]`; `--help` lists the
options. `_dev_common.py` holds the shared `DEV_BATTERY` (13.5 kWh, 5 kW, 10%
reserve, 95% / 95%) and trace and price loaders.

| Kind | Scripts |
|---|---|
| Interactive views (matplotlib, day-by-day navigation) | `viz_self_consumption`, `viz_perfect_foresight_lp`, `viz_perfect_foresight_lp_advanced`, `viz_perfect_foresight_custom`, `viz_vpp_sc_event`, `viz_vpp_tou_event`, `viz_rolling_perfect_foresight_lp`, `viz_rolling_perfect_foresight_lp_advanced`, `viz_rolling_vpp_sc_event`, `viz_rolling_vpp_tou_event` |
| Checks (non-interactive) | `check_self_consumption`, `check_perfect_foresight_lp`, `check_perfect_foresight_lp_advanced`, `check_perfect_foresight_custom`, `check_perfect_foresight_custom_twice`, `check_vpp_sc_event`, `check_vpp_tou_event`, `check_usage_forecast`, `check_solar_forecast` |
| Forecast fitting | `fit_usage_forecast`, `fit_solar_forecast` |
| Diagnostics | `check_eov_event` (one `vpp_sc_event` window), `check_min_export_price` (event-window price distribution), `check_simultaneous_flows`, `compare_perfect_foresight` (custom vs LP per interval), `compare_bills` (two runs' bills), `diagnose_shift_window` (one greedy sink search), `diag_import_tou_breakdown`, `export_billing_check` (per-interval billing export for spreadsheet checking) |

`model/scripts/` holds `export_cap_slide_profiles.py` (price profiles for the
cap-mechanism slides) and `reconstruct_battery_flows.py` (rebuilds
`battery_flows.parquet` for runs made before that file existed).

---

## Testing

`pytest` from the project root; `pyproject.toml` sets `testpaths =
["model/tests"]` and `pythonpath = ["."]`.

```bash
python3 -m pytest model/tests/ -v              # full suite
python3 -m pytest model/tests/ -m "not slow"   # quick suite
```

Tests that take more than about 5 s (full-year LP and rolling scenario runs)
carry `@pytest.mark.slow`, registered in `pyproject.toml`. Every other test
runs in well under that, so the quick suite finishes in a minute or two.

Tests are per component and per shared type, on synthetic data with
analytically known answers, so a failure points at one thing. Test files are
named after the module they cover (`test_<module>.py`, with suffixes such as
`_advanced`, `_monthly`, `_vpp`, `_curtail`, `_wiring` for feature subsets).
Each component spec's Verification section lists the properties its tests
check. Pinned fixtures and test tariffs are in `model/tests/fixtures/`.

---

## Known limitations

**Self-consumption ignores the export limit.** The self-consumption baseline,
and the `solar` and `self_consumption` cohorts, export without the per-site
limit that every optimising runner applies. See
`docs/review/possible_bugs.md`, PB-11.

**Uniform battery and solar per scenario.** Every site receives the same
`BatterySpec` and a solar scale of 1.0. The per-site resolution hooks are where
variation would go.

**Linear degradation only.** `c_deg_per_kwh` is a throughput cost in the
objective. Cycle counting, calendar fade and warranty limits are not modelled.

**Network demand and seasonal tariffs are not modelled.**

**LP objective and bill differ on plans with critical rates.** A critical rate
can put the export rate above the import rate in the same interval, which
would make importing and exporting at once profitable. The advanced LP
prevents this by shifting the opposite side of each critical interval by `δ`,
and billing does not, so the LP objective differs from the bill's dispatch part
by `Σ δ[t] × opposite-side flow` (`perfect_foresight_advanced_spec.md`).

**Single-year periods.** Monthly billing and the daily-charge count assume one
calendar year per scenario, which `ScenarioConfig` guarantees.

---

## `self_hedging` (not built)

A cohort-level coordination type: it would receive the full cohort's traces
and return every site's dispatch, because at one site the best behaviour
depends on the others. It is in `CoordinationType` and passes validation, but
the orchestrator raises `NotImplementedError` after ingest (PB-12).
