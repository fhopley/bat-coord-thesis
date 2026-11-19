# perfect_foresight_lp_advanced — Specification

## Purpose and scope

`model/components/perfect_foresight_lp_advanced.py` extends
`perfect_foresight_lp` to retail plans with features that cannot be written as
a per-interval price:

1. **Block import tiers:** stepped import rates on the daily energy imported
   within a time window.
2. **Block export tiers:** stepped export rates on the daily energy exported
   within a time window.
3. **Zero-import reward:** a daily credit for keeping imports within a
   window at or below a tolerance.
4. **Critical export rate:** a fixed export rate that replaces the normal
   export rate while the wholesale price is above a threshold.
5. **Critical import rate:** a fixed import rate that replaces the normal
   import rate while the wholesale price is below a threshold.

Block tiers depend on flows accumulated over the day, and the reward on a
count of qualifying days, so both need structure beyond per-interval prices.
The critical rates are per-interval, but they also take their intervals out
of the block tiers, and they apply differently in the LP and in billing.

It is the `perfect_foresight_advanced` coordination type. It is also the
between-event household optimiser in `vpp_sc_event` when
`gap_strategy='tou_advanced'` (see `vpp_tou_event_spec.md`). For a plan with
`has_advanced_features == False` it gives the same result as
`perfect_foresight_lp`. `rolling_advanced` reuses `_build_block_tier_data`
for its window LP.

It also dispatches `mixed` plans. A `mixed` plan carries no advanced features,
so the LP is the standard formulation run on the mixed-plan dispatch prices
(`model_spec.md`, Mixed-plan dispatch prices): no tier variables, penalty or
critical rates. The billed import rate is not the dispatch import rate on
these plans, so the bill is not the LP objective.

The objective, the core 6T variables, the flow-balance and SOC constraints,
curtailment, simultaneous-flow netting and the flow decomposition are those of
`perfect_foresight_lp` (see `perfect_foresight_lp_spec.md`). Its helpers are
imported, not duplicated. `curtail_cost_per_kwh_dollars` is passed to
`_build_lp` (`perfect_foresight_lp_spec.md`, Curtailment modes). The
orchestrator sets it with `prices.dispatch_curtail_cost(retail_tariff)`: free
curtailment (0.0) on `tou` and `wholesale_exposed` plans, and
`MIXED_CURTAIL_COST_PER_KWH = 1e-5` $/kWh on a `mixed` plan. Whenever the export floor binds,
exporting and curtailing earn the household the same, and the LP's choice
between them is arbitrary. The plan's retailer cannot curtail most sites, so
the cost breaks the tie towards export. Curtailment still occurs where a full
battery and the export limit force it.

---

## Formulation

### Additions to the standard LP

**Tier variables.** For each block window w, each local calendar day d and
each tier k, a variable `v[w,d,k]` holds that day's energy in that tier.
Import and export windows each have their own set. The variables are appended
after `x[5T:6T]`:

| Variable | Bounds | Objective coefficient |
|---|---|---|
| `v_imp[w,d,k]` | [0, `threshold_k − threshold_{k−1}`]; final tier [0, ∞) | `+rate_k / 100` $/kWh |
| `v_exp[w,d,k]` | same | `−rate_k / 100` $/kWh (a reward) |

`threshold_0 = 0`.

**Linking constraints**, one row per window per day:

```
Σ_k v_imp[w,d,k] − Σ_{t ∈ window w, day d} imp[t] = 0
Σ_k v_exp[w,d,k] − Σ_{t ∈ window w, day d} exp[t] = 0
```

Import tier rates are non-decreasing and export tier rates non-increasing
(enforced when the tariff loads). The cost is therefore convex, and the LP
fills the cheapest import tier, or the best-paying export tier, first without
any ordering constraints. With one import and one export window of two tiers,
a year adds 1,460 variables and 730 rows.

**Modified price vectors.** `perfect_foresight_lp_advanced` is given
`price_imp`/`price_exp` from `prices.build_lp_price_vectors_advanced`, then
adds the zero-import penalty and, optionally, the export-order nudge
internally. Neither addition reaches the caller's vectors or billing. On a
`mixed` plan `build_lp_price_vectors_advanced` returns the mixed-plan dispatch
prices from `build_price_vectors` unchanged, since there is nothing to mask or
replace.

### Block tiers

For `tou` plans, `build_lp_price_vectors_advanced` (via
`build_base_price_vectors_advanced`) sets `p_imp` in block import window
intervals to `usage_other_c_per_kwh / 100` only, and `p_exp` in block export
window intervals to `feedin_other_c_per_kwh / 100` only. The TOU energy rate
is carried by the tier variables instead. For `wholesale_exposed` plans the
retail layer is already just `usage_other`/`feedin_other`, so there is nothing
to mask. The tariff loader warns that block tiers on a wholesale plan are
unusual, and treats them as a retail overlay. `mixed` plans may not define
advanced features at all.

Billing (`billing._block_tier_components`) is consistent with the LP. Its
per-interval pass charges the TOU rate in window intervals, so it subtracts
the TOU energy component (keeping `usage_other`/`feedin_other`), groups window
flows by local calendar day, and applies the tier structure.

Tier accounting (the LP's linking rows, billing's tier cost and undo, and the
day-0 carry-in in `vpp_sc_event`) uses the effective window masks, which
leave out critical intervals on the window's side (see Critical rates).
The per-interval block masking in `build_base_price_vectors_advanced` uses
the plain window masks.

A window is a local-time `[start_hour, end_hour)` range. With
`applies_weekends: false` it covers weekdays only, and weekend intervals keep
the standard period rates. Windows on the same side may not overlap. The loader
warns if a block import window overlaps a non-zero `usage_tariffs` period,
because the block rate overrides it.

### Zero-import reward

In the LP, a penalty `π` is added to `p_imp` for reward-window intervals
(`prices.build_zero_import_penalty`):

```
π = (reward_c_per_day / 100) / max(tolerance_kwh, 1.0 + 0.01)      $/kWh
tolerance_kwh = tolerance_kwh_per_hour × (end_hour − start_hour)
```

Floor 1.01 (`_ZERO_IMPORT_PENALTY_SLACK_KWH = 0.01`): the penalty per kWh
is always slightly below the daily reward. When a plan also has a critical
export rate, the LP should still export during a price spike in the reward
window, even if that needs a small import that forfeits the reward. A penalty
above the critical export rate would suppress those exports. The ordering
holds only when `critical_export_rate.rate_c_per_kwh ≥ reward_c_per_day`
(both in cents); this needs checking for each new tariff.

The penalty is not a tariff component. Billing
(`billing._zero_import_day_credits`) counts local days whose window imports
are at or below `tolerance_kwh` (below 1e-9 kWh when the tolerance is zero).
It credits `reward_c_per_day` for each such day to `SiteBill.retail_other_credits`,
where a negative value is a credit. For plans with this feature the LP
objective therefore does not reconcile with the bill.

### Critical rates

An interval is critical when its 5-minute RRP crosses a critical rate's
threshold:

```
C_x = { t : RRP[t] > θ_x }      critical export intervals
C_m = { t : RRP[t] < θ_m }      critical import intervals
```

`θ_x > θ_m` is enforced when the tariff loads, so no interval is in both.
With `ρ_x = (critical_export_rate.rate_c_per_kwh + feedin_other_c_per_kwh) / 100`
and `ρ_m = (critical_import_rate.rate_c_per_kwh + usage_other_c_per_kwh) / 100`
($/kWh):

- **The critical side's rate is replaced.** Export in `C_x` earns `ρ_x` and
  import in `C_m` costs `ρ_m`, instead of the TOU period rate or, inside a
  block window, the tier rate. A negative `ρ_m` pays the household to import.
  `usage_other`/`feedin_other` still apply, because they are not GloBird
  rates. On the ZeroHero plan GloBird confirmed that the rate replaces rather
  than adds.
- **The critical side's flow leaves the block tiers.** Export in `C_x` does
  not count towards any export tier, and import in `C_m` does not count
  towards any import tier. The effective masks for tier accounting are
  `M_w \ C_x` for export windows and `M_w \ C_m` for import windows.
  `prices.block_tier_masks` builds them from a price series.
- **The other side is unchanged.** An import in `C_x` pays its normal rate and
  counts towards any import tier, and an export in `C_m` likewise.

**Billing** (`prices.apply_critical_rates(..., shift_opposite=False)`) sets
`r_exp[t] = ρ_x` for `t ∈ C_x` and `r_imp[t] = ρ_m` for `t ∈ C_m` on the retail
layer rates. The tier undo and tier cost run over the effective masks, so they
never touch the critical side of a critical interval.

**LP** (`prices.apply_critical_rates(..., shift_opposite=True)`), applied to
the block-masked base vectors `b_imp`, `b_exp` from
`build_base_price_vectors_advanced`:

```
t ∈ C_x:   p_exp[t] = ρ_x,   δ[t] = ρ_x − b_exp[t],   p_imp[t] = b_imp[t] + δ[t]
t ∈ C_m:   p_imp[t] = ρ_m,   δ[t] = ρ_m − b_imp[t],   p_exp[t] = b_exp[t] + δ[t]
```

Shifting the opposite side by `δ` keeps the spread `p_imp − p_exp` at its
base value, so `p_imp ≥ p_exp` holds wherever it holds for the base vectors and
the LP never gains from importing and exporting in the same interval. Without
the shift a 100 c/kWh export next to a 59.4 c/kWh import would be free money.
The cost is that the LP overprices an import in `C_x`, and underprices an
export in `C_m`, by `|δ[t]|`, where billing uses the real rates. Both are flows
a household at a price extreme rarely wants. `δ` is measured from the
block-masked base rather than an unmasked one so that the rolling planner,
which holds that base precomputed, builds the same vectors from the same
prices.

The linking rows use the effective masks, so an interval in `C_x` inside an
export window carries `ρ_x` and no tier variable, exactly as billed.

A negative `ρ_m` can make it worthwhile to cycle energy through the battery to
increase import. `c_deg_per_kwh` discourages this, and
`_net_simultaneous_charge_discharge` removes any that remains.

### TOU export-order nudge

On a `tou` plan the export price is flat within a period. On a solar-surplus
day, "export early, then charge from later solar" costs the same as "charge
first, then export the surplus". HiGHS tends to pick the first, which no real
controller does. With `apply_export_order_nudge=True`,
`prices.build_tou_export_order_nudge` adds

```
δ[t] = 1e-8 × minutes since local midnight at interval start      $/kWh
```

to `p_exp`, so exporting later in the day is very slightly more valuable. That
breaks the tie in favour of charging first. The maximum is about 1.44e-5 $/kWh.
It applies to `tou` plans only (it returns `None` otherwise). It is added to
`p_exp` only, so grid charging before a TOU peak stays where the real price
signal puts it.

`prices.apply_tou_export_order_nudge` caps the nudged `p_exp` at `p_imp`
(before the zero-import penalty), so the nudge is dropped where the two are
equal, for example in a free import window with zero feed-in. Uncapped, the LP
would import and export the same kWh in such an interval to earn the nudge,
with no battery involved, up to the export limit.

Charging at one interval and discharging to export at a later one still gains
up to 1.44e-5 $/kWh where the two real prices are equal, against a cycling
cost of `2 × c_deg` per kWh. `prices.check_export_order_nudge_degradation`
raises on a TOU plan unless `c_deg_per_kwh > 7.2e-6` $/kWh. The LP calls it,
and `validate_scenario` calls it before ingest.

The nudge roughly triples HiGHS solve time, so it is off by default. It is set
by the scenario YAML key `apply_export_order_nudge`
(`ScenarioConfig.apply_export_order_nudge`, default `False`) and turned on by
the viz scripts. `ScenarioConfig` rejects it under a coordination type that does
not apply it (`EXPORT_ORDER_NUDGE_COORDINATION_TYPES`). The rolling window
planner applies the same nudge (`rolling_perfect_foresight_lp_advanced_spec.md`).

### Price ordering

`_validate_inputs_advanced` performs the same checks as
`perfect_foresight_lp._validate_inputs` except price ordering. The critical
rates' opposite-side shift preserves the spread and the nudge is capped at
`p_imp`, but callers may supply vectors built another way. On a
`mixed` plan the ordering is proved over the whole market price range when the
dispatch stack is built (`validate_mixed_price_ordering`, `model_spec.md`).

### Local time

Block windows, the reward window, the nudge and the daily reset of tier
accounting all use the site's local time for `state`, through
`core/localtime.local_wall_clock` at the interval-start instant. Day grouping
(`_build_day_map`) uses the actual local calendar date, so 23- and 25-hour DST
days need no special handling. No window is assumed to straddle a DST
transition.

---

## Interface

```python
def perfect_foresight_lp_advanced(
    site_id: str,
    solar_kwh: pd.Series,
    load_kwh: pd.Series,
    price_imp: pd.Series,          # $/kWh, from build_lp_price_vectors_advanced
    price_exp: pd.Series,          # $/kWh, from build_lp_price_vectors_advanced
    battery_spec: BatterySpec,
    soc_init_kwh: float,
    soc_final_kwh: float,
    retail_tariff: RetailTariff,   # block tiers, zero-import reward, critical rates
    prices_mwh: pd.Series,         # $/MWh; defines C_x, C_m for the tier masks
    state: str,                    # site location, for local time
    export_limit_kwh_per_interval: float = np.inf,
    interval_hours: float = 5 / 60,
    first_day_block_import_used: dict[int, float] | None = None,
    first_day_block_export_used: dict[int, float] | None = None,
    apply_export_order_nudge: bool = False,
    curtail_cost_per_kwh_dollars: float | None = 0.0,   # as perfect_foresight_lp
) -> SiteTraces
```

Raises `ValueError` for misaligned indexes, NaN inputs, negative load or
out-of-bounds SOC, and `RuntimeError` if the solver does not return an
optimal solution.

### Horizons that start mid-day

When the horizon starts partway through a local day, as in the `vpp_sc_event`
TOU gap, block energy may already have flowed earlier that day.
`first_day_block_import_used`/`first_day_block_export_used` map window index
(0-based, in tariff order) to kWh already used that day; a missing key means 0.
Only day 0's tier bounds change. The usage fills tiers in order:

```
remaining = used
for each tier k:
    hi[w, 0, k] = max(0, capacity_k − remaining)     (∞ for the final tier)
    remaining   = max(0, remaining − capacity_k)
```

The linking constraints are unchanged, because day 0's row sums only the
horizon's own flows.

The caller (`vpp_sc_event._run_tou_gap`) sums import and export in each
window over the already-assembled trace for the local day of the gap start.

`_build_block_tier_data` also accepts precomputed window masks
(`import_window_masks`, `export_window_masks`). `rolling_advanced` supplies
them because it rebuilds tier data for every 5-minute window. Precomputed
masks are used as given, so a caller must already have removed critical
intervals. Without them, `perfect_foresight_lp_advanced` builds the effective
masks from `prices_mwh` with `block_tier_masks`.

The day-0 carry-in sums flow over the effective masks too, so a critical
export earlier in the day does not use up the day's export tier.

### Orchestrator

`_run_perfect_foresight_advanced` in `model/orchestrate.py` mirrors
`_run_perfect_foresight` (see `perfect_foresight_lp_spec.md`). It builds prices
with `build_lp_price_vectors_advanced`, passes `retail_tariff`, `prices_mwh`,
`state` and `scenario.apply_export_order_nudge`, and bills with
`compute_bill`. The worker passes
`curtail_cost_per_kwh_dollars=dispatch_curtail_cost(retail_tariff)`, derived
from the tariff it already holds, so the worker tuple is unchanged. On a
`mixed` plan the export-order nudge is inert because
`build_tou_export_order_nudge` returns `None` for non-`tou` plans.
`validate_scenario` builds the mixed dispatch stack, so a plan whose ordering
can fail is refused before ingest. Example:
`model/configs/example_perfect_foresight_lp_advanced.yaml`.

---

## Tariff YAML

All fields optional. Reference: `model/tariffs/globird_advanced_tou_2025.yaml`.

```yaml
block_import_tiers:            # list of windows
  - start_hour: 11             # local time, [start_hour, end_hour)
    end_hour: 14
    applies_weekends: true     # default true
    tiers:                     # at least two, ascending threshold
      - threshold_kwh: 50.0
        rate_c_per_kwh: 0.0
      - threshold_kwh: .inf    # final tier must be unbounded
        rate_c_per_kwh: 30.8

block_export_tiers:
  - start_hour: 18
    end_hour: 21
    applies_weekends: true
    tiers:
      - threshold_kwh: 15.0
        rate_c_per_kwh: 14.0
      - threshold_kwh: .inf
        rate_c_per_kwh: 2.0

zero_import_reward:
  start_hour: 18
  end_hour: 21
  reward_c_per_day: 100.0
  tolerance_kwh_per_hour: 0.03   # default 0.0

critical_export_rate:          # tou plans only
  threshold_dollars_per_mwh: 2000.0   # critical while RRP > threshold
  rate_c_per_kwh: 100.0               # replaces the export rate

critical_import_rate:          # tou plans only
  threshold_dollars_per_mwh: -100.0   # critical while RRP < threshold
  rate_c_per_kwh: -5.0                # replaces the import rate; negative pays
```

These load into `BlockTariffWindow`/`BlockTier`, `ZeroImportReward` and
`CriticalRate` on `RetailTariff` (`model/core/tariff.py`).
`RetailTariff.has_advanced_features` is true when any of these fields is set.
The loader rejects `critical_*_rate` on non-`tou` plans, an export threshold
not above the import threshold, a `critical_*_rate` block carrying
`bonus_c_per_kwh`, and the keys `stress_export_amplifier` and
`stress_import_amplifier`. Those keys and field define an additive bonus, and
rejecting them stops such a tariff loading as a replacement rate.

---

## Verification

`model/tests/test_perfect_foresight_lp_advanced.py` checks:

- **Equivalence:** the same result as `perfect_foresight_lp` on standard
  plans, with and without solar.
- **Block imports:** fills the free tier exactly; avoids the expensive tier
  where possible; prefers the free tier to importing outside the window; each
  day gets its own free tier.
- **Block exports:** fills the high-reward tier first; spills to the next tier
  when solar exceeds tier 1.
- **Zero-import reward:** avoids window imports when the battery can cover
  them; allows them when it cannot.
- **Critical rates:** dispatches at spikes and not otherwise; charges at
  negative prices; the opposite-side shift keeps the LP solvable; a critical
  interval inside a block window is priced at the critical rate and left out
  of that window's tier, including in the day-0 carry-in.
- **Export-order nudge:** charges before exporting.
- **Mixed plans:** the LP receives the dispatch prices, not the billed rates;
  it imports at a peak TOU rate to export later only when the export credit
  exceeds the dispatch import price plus round-trip costs; at negative spot,
  with the floor at 0 and the export limit slack, surplus solar is exported,
  not curtailed; the bill uses the billed rates; the LP's price cost less the
  bill's dispatch part equals the spot term on imports; the curtailment cost
  argument reaches the LP (with a negative export credit and no floor, free
  curtailment curtails and a 1 $/kWh cost exports instead).
- **Orchestration** (`test_orchestrate.py`): validation accepts `mixed` for
  the cohorts and types in `MIXED_COORDINATION_TYPES` and rejects it for the
  rest; validation runs the ordering proof; the worker passes the plan's
  curtailment cost.
- **Result:** `SiteTraces` invariants hold with block tiers and the reward;
  the bill is no higher than without a battery.
- **Input validation**, plus unit tests of `_build_day_map` and
  `_build_block_tier_data` (counts, bounds, export cost sign).
- **Precomputed masks:** match internal computation, including with day-0
  carry-in, weekday-only windows over a weekend, a DST transition and
  critical intervals. Mask count and length mismatches raise.

`model/tests/test_billing_advanced.py` checks that billing replaces the rate on
the critical side, including inside a block window, leaves the critical flow
out of the tier, leaves the other side alone, and equals the LP's per-interval
cost plus the tier cost less `Σ δ[t] × opposite-side flow`.

`model/dev/check_perfect_foresight_lp_advanced.py SITE_ID YEAR --retail-tariff ...`
runs a real site non-interactively and exits non-zero on any FAIL.
`model/dev/viz_perfect_foresight_lp_advanced.py` is the interactive view;
its verification output reconciles the LP-price cost plus the tier cost, less
the opposite-side shift, with the bill's dispatch part as a PASS/FAIL check.

---

## Known limitations

- **Reward versus critical export priority** holds only when the critical
  export rate (c/kWh) is at least the daily reward (c). A new tariff that
  breaks this would have its spike exports suppressed in the reward window.
  Fix path: set the penalty from the critical export rate as well as the
  reward.
- **Objective does not reconcile with the bill** for reward plans, because the
  penalty stands in for the real daily credit.

---

## Validation

Measured September 2026, 20 kWh `DEV_BATTERY`, 10 kW export limit, through
`viz_perfect_foresight_lp_advanced` (export-order nudge on):

- **S0008, NSW 2022, GloBird Advanced TOU 2025** (critical rates 100 c/kWh
  above $2000/MWh and −5 c/kWh below −$100/MWh), Ausgrid EA025 network:
  - Block import 11–14: 7.22 kWh/day on average, always within the 50 kWh free tier.
  - Block export 18–21: 0.59 kWh/day on average, always within tier 1.
  - Zero-import reward: 365 of 365 days, $365.
  - Critical export: 69 intervals, 51.6 kWh, 16.7 kWh of it in the export
    window and outside the tier, $51.55.
  - Critical import: 7 intervals, 6.2 kWh, 2.6 kWh of it in the import
    window, −$0.27.
  - LP price reconciliation with the bill's dispatch part within 1.2e-13.
  - HiGHS solve about 31 s, about 632,000 variables.
  - Bill $830 against $2,142 under self-consumption.
- **S0100, NSW 2022, GloBird ZeroHero 2026**, Ausgrid EA025/EA029 network:
  bill $491.47, against $527.29 on the same plan without critical rates;
  critical export 69 intervals, 50.7 kWh; critical import 7 intervals, 7.2 kWh.
- **Equivalence with `perfect_foresight_lp`** on a standard plan across the
  197-site NSW 2022 cohort (June–July 2026).
- **S0008, NSW 2022, Tesla Electric Dynamic 2026** (`mixed`, provisional
  rates, share 0.9 of spot, floor 0), Ausgrid EA025/EA029 network, measured
  9 October 2026 at commit `883cdb3`:
  - Bill $608.78, against $1,254.83 under self-consumption.
  - LP price reconciliation with the bill's dispatch part, after removing the
    $598.62 spot term on imports, within 1.7e-13.
  - Grid to battery 1,195.8 kWh, battery to grid 299.8 kWh, no curtailment,
    no simultaneous import and export.
  - HiGHS solve 35 s, 630,720 variables.
