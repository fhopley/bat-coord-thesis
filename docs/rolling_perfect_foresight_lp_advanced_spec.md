# rolling_perfect_foresight_lp_advanced — Specification

## Purpose and scope

Measures what a household battery loses on an advanced time-of-use plan when
the controller acts on forecasts instead of knowing the future.
`perfect_foresight_lp_advanced` is the upper bound on plans with block tiers,
a zero-import reward and critical rates. This coordination type runs the
same plan through the shared rolling loop, and the difference between the two
bills is the value of foresight on this class of plan.

It is the rolling loop (`rolling_optimisation_spec.md`: decision instants,
causality, providers, terminal pin, fallback, retention, `RollingConfig`) with
`AdvancedWindowPlanner` injected as the window planner. It uses the advanced LP
formulation (`perfect_foresight_advanced_spec.md`: tier variables, the
zero-import penalty, critical rates with the opposite-side shift). This
document covers only what is
new when the two are combined.

| Piece | File |
|---|---|
| `AdvancedWindowPlanner`, `AdvancedWindowSolver`, tier helpers | `model/components/rolling_advanced.py` |
| Base (critical-rate-free) price vectors | `prices.build_base_price_vectors_advanced` |
| Precomputed block masks on `_build_block_tier_data` | `model/components/perfect_foresight_lp_advanced.py` |
| Validation, per-site runner, bulk routing | `model/orchestrate.py` |
| Single-site view | `model/dev/viz_rolling_perfect_foresight_lp_advanced.py` |

**In scope:** TOU plans, from plain time-of-use to the GloBird ZeroHero
reference, which has every feature at once: a block import window, a block
export window, a zero-import reward and both critical rates. The AGL, Origin
and Energy Australia plans are the common case, with plain TOU rates,
sometimes a block tier, and no reward or critical rate. Each absent feature
removes work rather than leaving inert machinery:

- a plan with no block tiers has no tier variables and a fixed constraint
  matrix;
- a plan with no critical rates needs no price forecast.

`mixed` plans are also in scope (`model_spec.md`, Mixed-plan dispatch
prices). They carry no advanced features, so the window LP has no tier
variables, penalty or critical rates. On these plans the price forecast sets
every window price.

The same planner is the household side of `rolling_vpp_tou_event`.

**Out of scope:** wholesale-exposed plans (`rolling_perfect_foresight_lp`); the
planner rejects them.

---

## What is uncertain on this path

- **Retail rates are known exactly on a TOU plan.** `build_price_vectors`
  returns the retail layer only, so the per-interval price depends only on the
  clock.
- **On a `mixed` plan the price forecast sets both window prices**: `p_imp`
  through the positive spot term and `p_exp` through the share and the floor.
  Both are built from the forecast RRP, including at the committed interval
  under `current_interval_price: forecast`. The settled RRP is used only for
  billing.
- **Household load and solar** come from the same providers as on the
  wholesale path (`RollingConfig.usage_forecast`, `solar_forecast`).
- **The wholesale price forecast matters for one thing: which intervals are
  critical.** On the GloBird plan export earns 100 c/kWh in place of its normal
  rate above $2000/MWh, and import costs −5 c/kWh in place of its normal rate
  below −$100/MWh. In both cases the flow leaves the block tier. Under perfect
  foresight the threshold test uses the settled price; here the window LP uses
  the forecast. The resulting error is **discrete**: an interval is critical or
  not, which sets both its rate and whether its flow counts towards the tier,
  and it can flip between successive forecasts. Beyond the fine segment the
  forecast is 30-minute data, so a 5-minute spike cannot appear in most of the
  horizon.

**A rolling run must have at least one genuine source of imperfect
information.** On this path that means a usage forecast, a solar forecast, a
critical rate driven by a price forecast, or a `mixed` plan. A TOU plan with none of these is
perfect foresight over a truncated horizon and is refused.

---

## Precomputed once per site

Calling `perfect_foresight_lp_advanced` per window would repeat period-level
work about 105,000 times per site-year: validation, the penalty, the day map
and the block masks. `AdvancedWindowPlanner` is built once per site and holds:

| Held | Contents |
|---|---|
| `base_imp`, `base_exp` | critical-rate-free, block-masked price vectors, $/kWh, from `build_base_price_vectors_advanced` |
| critical rate parameters | `(θ_x, ρ_x)` and `(θ_m, ρ_m)` for each critical rate present, with `ρ` including `feedin_other`/`usage_other` |
| settled critical masks | `C_x`, `C_m` from `settled_rrp_mwh`, read only at interval k when k commits (see Day state) |
| block masks `M_w` and their contiguous runs | one per import and export window, runs split at local midnight |
| day map `d(t)` | local calendar day per interval (interval-start dates) |
| reward mask `R` and penalty rate `π` | as in `perfect_foresight_advanced_spec.md` |
| tier structure | `BlockWindowSpec` per window, in `_build_block_tier_data`'s order |
| `dispatch_stack` | `mixed` plans only, `None` otherwise: `MixedDispatchStack` from `build_mixed_dispatch_stack` (`model_spec.md`). `base_imp`/`base_exp` are built for every plan but not read on a `mixed` one |

`build_base_price_vectors_advanced` is the first half of
`build_lp_price_vectors_advanced`, which calls it and then applies the
critical rates. The rolling and perfect-foresight paths therefore share one
definition of block masking. The base is **not** derived by calling the full
function with zero prices: a critical import rate with a positive threshold
would fire on the zeros.

---

## Window prices

With `r̂_j` the forecast RRP at window position j, `x_j = 1[r̂_j > θ_x]` and
`m_j = 1[r̂_j < θ_m]`:

```
δ_j      = x_j·(ρ_x − base_exp[k+j]) + m_j·(ρ_m − base_imp[k+j])      $/kWh
p_exp[j] = base_exp[k+j] + δ_j
p_imp[j] = base_imp[k+j] + δ_j + π_j
```

In a forecast-critical export interval this sets `p_exp` to `ρ_x` and shifts
`p_imp` by the same amount, and in a forecast-critical import interval it sets
`p_imp` to `ρ_m` (before the penalty) and shifts `p_exp`. It is
`apply_critical_rates(..., shift_opposite=True)` applied to the forecast, so
with perfect prices the window vectors equal the perfect-foresight vectors.
The spread is unchanged and the LP cannot import and export at once to harvest
a critical rate. Billing replaces the rate on the critical side only, which is
what the retailer pays. The two deliberately differ.

With no critical rates, `requires_price_forecast` is `False` and `δ_j = 0`. The
loop then never queries the price provider, so the October 2022 gap and
`on_degraded_period` do not matter for these plans. They also never fall back
to self-consumption over a missing forecast that could not have changed the
answer. `orchestrate.advanced_plan_needs_price_forecast(retail_tariff)` is the
same rule, used by validation and the bulk runner.

On a `mixed` plan the window prices are
`dispatch_stack.window_price_vectors(r̂, k)`, with no `δ` and no penalty.
`requires_price_forecast` and `advanced_plan_needs_price_forecast` are both
`True`. With perfect prices the window vectors equal `build_price_vectors`
over the same positions. Forecast criticality, and with it tier membership,
follows whether the plan has critical rates, not whether it needs a forecast:
a `mixed` plan needs a forecast and has no critical rates.

**Price ordering.** `p_imp < p_exp` anywhere is free money. The LP imports and
exports in the same interval, and because the committed traces net
simultaneous flows while the LP's variables do not, the day state and the bill
then disagree about tier usage. This is how the guard was found: an
accumulator read 4.0 kWh against traces reading 0.0. The planner checks
`base_imp ≥ base_exp` elementwise at construction and names the first
offending interval. That is sufficient for every forecast, because `δ_j` is
added to both sides and leaves the spread unchanged, and the penalty only
raises `p_imp`. The usual cause is a block import window whose masked price
(`usage_other`) sits below that hour's feed-in rate, which is a tariff
definition error. GloBird passes because its off-peak feed-in is a token
0.00001 c/kWh against a 0.821 c/kWh `usage_other`.

**Export-order nudge.** With `apply_export_order_nudge` on a `tou` plan, the
planner adds `build_tou_export_order_nudge` to `base_exp` at construction,
capped at `base_imp` (`apply_tou_export_order_nudge`), after the price-ordering
check. Every window's `p_exp` is a slice of `base_exp`, so each window breaks
the flat-feed-in tie between "export now, charge from forecast solar later" and
"charge now, export the surplus" in favour of charging first. The cap keeps `base_imp ≥ base_exp`, so the ordering argument above still
holds. The nudge and its degradation-cost guard are those of the
perfect-foresight LP (`perfect_foresight_advanced_spec.md`). It has no effect on
a `mixed` plan.

On a `mixed` plan the window prices depend on the forecast, so the
elementwise check on fixed vectors does not apply. `build_mixed_dispatch_stack`
proves the ordering for every RRP in the market range instead
(`validate_mixed_price_ordering`), so it holds for any forecast the loop can
pass.

---

## Day state

Two quantities reset at local midnight and cannot be recovered from the
window. They are running totals, updated in `observe_commit` from the LP's own
import and export (`x[2n]`, `x[3n]`) or from the fallback's flows.

**Block usage.** For each block window w, `U_w` is the committed import (or
export) inside `M_w` so far today, at intervals that were not critical on w's
side by the settled price. It is passed as day 0's carry-in to the window's
tier bounds: the `first_day_block_*_used` mechanism in
`perfect_foresight_advanced_spec.md`. It is one addition per window per
interval, rather than re-summing assembled traces.

The settled RRP is passed to the planner at construction (`settled_rrp_mwh`)
and read only at entry k, inside `observe_commit(k)`. That is causal: interval
k's price is settled once k has ended, and the accumulator is read only by
later windows. The settled test, not the forecast one, is what billing uses,
so counting by the forecast would let `U_w` and the bill disagree whenever the
forecast missed a spike. `settled_rrp_mwh` may be `None` when no block window
shares a side with a critical rate; otherwise `observe_commit` raises without
it. Validation constructs the planner without prices, which is allowed because
it never commits.

**Reward availability.** `U_R` is the committed import inside `R` so far today.
Today's reward is still available while `U_R ≤ τ`, where `τ = tolerance_kwh`,
or 1e-9 when the tolerance is zero. That threshold is copied from
`billing._zero_import_day_credits` deliberately. A bare `> 0` test would lose
the reward on a 1e-14 residual that billing still pays, and would plan the rest
of the day against an incentive billing disagrees with.

**The zero-import penalty is removed once today's reward is lost:**

```
π_j = π · 1[R[k+j]] · A_j,   A_j = 0 if d(k+j) = d(k) and U_R > τ, else 1
```

Imports later today stop carrying a phantom $0.99/kWh cost, while imports on
the next local day inside the horizon keep it. Without this the evening would
be distorted for reasons unrelated to forecast error.

**The day state is not a running bill.** The objective differs from the bill
by design: it includes the penalty, the critical rates' opposite-side shift
and optionally the nudge. `compute_bill` on the committed flows is the source of truth. The day
state exists only so the objective's incentives reflect what has already
happened today.

---

## The window LP (`AdvancedWindowSolver`)

The window is `perfect_foresight_lp_advanced` restricted to n intervals: the
standard `6n` block first, then one tier variable per (block window, day slot,
tier).

- **Day slots.** Slot 0 is the local date of interval k. The slot count is a
  per-site constant, because the persistent model cannot change its column
  count. It is computed as the most local dates any window touches, from the
  day map in one pass (`_max_slots`): 2 at 18 hours, 3 at 48. It is not derived
  from the horizon, because 23- and 25-hour DST days make any closed form wrong
  or loose. A slot a window does not reach has an empty linking row, which pins
  its tier variables to zero.
- **Linking rows.** For each window and slot, the tier variables minus the
  window's imports (or exports) in that block window on that date, at
  positions not forecast-critical on the window's side, equal 0.
- **Tier bounds.** Tier capacity, with day 0 reduced by `U_w`
  (`tier_upper_bounds`, the same rule as `_build_block_tier_data`).

The objective is the core objective with the window prices above, plus the tier
costs (+ for import tiers, − for export).

**Solving by mutation.** `A_eq` is not invariant here: the linking rows hold a
coefficient for each block interval in the window, and those move as the
window slides. A block window's intervals form a contiguous run in absolute
time, split at local midnight (`_split_runs_by_day`). Each run maps to one
(window, slot). Membership is that run, in window positions, minus the
positions forecast-critical on the window's side: a set of positions per
(window, slot). `set_membership` changes only the symmetric difference between
the previous window's set and this one's. Advancing k by one shifts the run
left by one position, so without critical intervals **only its two ends
change**; a forecast that moves a spike adds or removes a few positions. The
costs, bounds and tier bounds change as on the wholesale path. The split at midnight matters for whole-day block windows such as EA
Battery Ease's 0–24 export tier. Without it, one run would span every day and
post the whole period's flow to one tier variable.

A plan with no block tiers has no tier variables or linking rows and runs on
the plain core model.

Keeping one warm model beats rebuilding per window, and beats a cache of one
model per time of day (see Validation). The value of a persistent model is
mostly the **warm basis** from the immediately preceding, nearly identical
window. Mutating a few coefficients does not meaningfully damage it.

**Structural gate.** An early version of the incremental update used each
linking row's index relative to the linking block instead of its absolute
index. It rewrote flow-balance rows and produced wrong objectives on 466 of
600 benchmark windows, every one of which solved to optimality. The permanent
test therefore mutates a model forward, reads the constraint matrix back out
of HiGHS, and compares it coefficient by coefficient with a fresh build of the
same window, including windows whose forecast puts critical intervals inside a
block window and moves them between steps. On failure it reports which row
types differ.

---

## Terminal SOC, failure handling, diagnostics

**Terminal pin and fallback.** Both are unchanged from the loop.
`observe_commit` receives fallback intervals too, so their flows count against
the tiers and the reward.

**A caveat on the pin that does not arise on the wholesale path.** A window
ending inside a block export window or a reward window is pinned to whatever
self-consumption leaves there, and self-consumption is blind to both. The pin
is always feasible but may be a poorer proxy for stored energy's value. It is
left as-is until a run shows it misbehaving.

`plan_diagnostics()` adds three fields to `RollingDiagnostics`, each `None`
when the plan lacks the feature:

- `reward_days_lost`;
- `block_tier_days_exhausted` (per block window; semicolon-joined in the CSV);
- `windows_with_tier_exhausted` (windows planned with day-0 capacity already
  used).

They are cumulative counts, not dated lists; dated detail for one site comes
from its traces. `plan_diagnostics()` does not mutate: closed days accumulate,
and the open day is added at read time, so reading mid-run cannot corrupt the
count.

---

## Configuration, validation and orchestration

`coordination_type: rolling_perfect_foresight_lp_advanced`, with a
`RollingConfig` and optionally `apply_export_order_nudge: true`. `forecast_source` is still required in the
YAML even when the plan never uses it, because the YAML parser does not load
the tariff. Example: `model/configs/example_rolling_perfect_foresight_lp_advanced.yaml`.

`validate_scenario` requires, for this type:

- the retail tariff is `tou` or `mixed`;
- at least one of a usage forecast, a solar forecast or a critical rate is
  present, or the plan is `mixed`;
- `AdvancedWindowPlanner` constructs, which enforces price ordering, and the
  nudge's degradation-cost guard when the nudge is on, before any ingest.

The shared rolling checks apply. Store coverage is checked only when the plan
has a critical rate or is `mixed`.

`run_rolling_perfect_foresight_lp_advanced_site` passes
`curtail_cost_per_kwh_dollars=dispatch_curtail_cost(retail_tariff)` to the
planner: `MIXED_CURTAIL_COST_PER_KWH` on a `mixed` plan, for the reason given
in `perfect_foresight_advanced_spec.md`, and free curtailment otherwise.

`_run_rolling_perfect_foresight_lp` serves both non-VPP rolling types. The
worker already holds `prices_mwh` for billing and passes it to
`run_rolling_perfect_foresight_lp_advanced_site` as `settled_rrp_mwh`, so the
worker tuple is unchanged. `rolling_vpp_event` passes its settled prices to the
household planner the same way. It branches in two places: it builds price-stack offsets only for the wholesale
type, and skips the price-provider worker initializer when the plan needs no
forecast. The worker's argument tuple carries `coordination_type` explicitly,
not inferred from `offsets is None`, followed by `apply_export_order_nudge`. The tuple is built, unpacked and
documented in three places, and
`test_orchestrate_rolling_advanced_wiring.py` checks all three for arity and
**position**. Two swapped fields keep the arity and unpack silently, so
checking arity alone missed a real swap.

---

## Verification

- **`test_rolling_advanced.py`:**
  - **Tier data:** tier specs follow the canonical order; tier bounds match
    the whole-period builder; surplus slots are pinned to zero.
  - **Solver:** it matches the real builders, including weekday-only windows;
    the structural gate; membership updates touch only the moving ends and
    are validated.
  - **Plans without advanced features:** no block features gives the core
    model; wholesale plans are rejected; the price forecast is required only
    with critical rates.
  - **Prices:** window prices reproduce the perfect-foresight vectors
    (base + critical rates(settled RRP) = `build_lp_price_vectors_advanced`);
    the critical test follows the forecast, not the outcome.
  - **Critical membership:** forecast-critical positions leave the linking row
    on their side only; the structural gate holds as they move; the day state
    ignores flow that was critical by the settled price and counts flow that
    was only forecast-critical; `observe_commit` raises without settled prices
    when they are needed.
  - **Reward:** the penalty stops once the day's reward is lost; the reward
    threshold matches billing; a solver residual does not lose the reward.
  - **Day state:** it resets at local midnight; `day0_used` matches a direct
    sum and feeds the tier bounds; window runs match a direct mask
    computation; the day state advances across a run and accumulates across
    fallback intervals.
  - **The correctness gate:** a full horizon with perfect prices reproduces
    `perfect_foresight_lp_advanced`, including with critical intervals inside
    both block windows.
  - **Loop behaviour:** the loop skips or requires the price provider
    correctly; plans where importing to export pays are rejected.
  - **Diagnostics:** `None` when a feature is absent; counts of reward days
    and tier exhaustion; the summary text.
  - **Mixed plans:** window prices equal `build_price_vectors` over the same
    positions when given the settled RRP; they follow the forecast, not the
    settled price, including at the committed interval; the price forecast is
    required; a full horizon with perfect prices reproduces
    `perfect_foresight_lp_advanced` on the same plan; a plan whose ordering
    can fail in the market range is refused at construction.
- **`test_orchestrate_rolling_advanced.py`** also checks that a `mixed` plan
  validates with no site forecast, that `advanced_plan_needs_price_forecast`
  is `True` for it, and that the site runner passes the plan's curtailment
  cost to the planner.
  - **Whole-day block windows:** run splitting and acceptance.
- **`test_orchestrate_rolling_advanced.py` and `_wiring.py`:** validation,
  routing and the worker tuple.
- **`test_viz_rolling_advanced.py`:** the analysis runs end to end on the real
  GloBird plan; advanced diagnostics reach it; reward credits agree with the
  planner's count; a plan without critical rates needs no price forecast; a
  wholesale plan is refused.
- **`test_prices.py` and `test_perfect_foresight_lp_advanced.py`:** the base
  builder and precomputed masks.

---

## Known limitations

- **Horizon not swept for this path.** 18 hours comes from the wholesale sweep.
  This path has strong daily structure (tier resets at local midnight, a fixed
  evening reward window), so a window ending before the next reset sees a
  different problem from one spanning it.
- **Terminal pin blind to tiers and the reward** (see above).
- **Linear reward penalty.** It is a stand-in for an all-or-nothing daily
  reward. The within-day switch-off above fixes the after-the-loss part, not
  the planned cost within a window. This is inherited from perfect foresight.
- **Critical-interval resolution** is limited by 30-minute coarse data beyond
  the first hour. This is a property of the information, not the component.
- **Non-solve overhead.** Solve-to-wall-clock ratio was measured at 1.94×,
  against 1.7× on the wholesale path. Likely sources are the per-window lists
  and dicts in `tier_upper_bounds` and `window_runs`. Not yet shown to matter.
- **Weekend-only exposure.** A block window with `applies_weekends: false` over
  a zero usage rate would price weekend imports in those hours at the
  underlying period rate, which could be zero. No current plan does this; a
  tariff-loader check would catch it if one ever does.

---

## Validation

Measured September 2026 at commit `498f1cc` (critical rates 100 c/kWh above
$2000/MWh and −5 c/kWh below −$100/MWh), NSW 2022, GloBird ZeroHero 2026,
Ausgrid EA025/EA029 network, 20 kWh, 10 kW export limit, 18 h horizon,
predispatch prices (`price_and_next`, `infill`, staleness 800 h), usage and
solar forecasts on, through `viz_rolling_perfect_foresight_lp_advanced
--compare-perfect-foresight`:

| Site | Self-consumption | Rolling | Perfect foresight | VCR | Reward days lost |
|---|---|---|---|---|---|
| S0008 | $2,048.80 | $965.19 | $795.78 | 86.5% | 0 |
| S0085 | $1,052.11 | $455.95 | $214.36 | 71.2% | 3 |
| S0097 | $4,821.11 | $2,469.01 | $2,325.41 | 94.2% | 4 |
| S0100 | $2,732.75 | $768.75 | $491.47 | 87.6% | 11 |
| S0102 | $1,760.37 | $504.72 | $276.15 | 84.6% | 3 |

- Headroom-weighted VCR 87.2%, mean 84.8%.
- Causality passed on every site. 276 self-consumption fallbacks per site: 275
  from no load forecast in the opening hours of the period, 1 from a missing
  price forecast.
- The planner's lost-reward days agree with billing's credited days on every
  site.
- Median 1.48–2.37 ms per window. These ran alongside the full test suite, so
  they are not comparable with the benchmark below.

Measured 9 October 2026 at commit `883cdb3` on a `mixed` plan: S0008, NSW,
1–5 June 2022, Tesla Electric Dynamic 2026 (provisional rates, share 0.9 of
spot, floor 0), Ausgrid EA025/EA029 network, 20 kWh, 10 kW export limit, 18 h
horizon, predispatch prices (`forecast`, `infill`, staleness 800 h), usage and
solar forecasts on:

| Self-consumption | Rolling | Perfect foresight | VCR |
|---|---|---|---|
| $21.62 | −$0.62 | −$14.38 | 61.8% |

- Causality passed. 289 self-consumption fallbacks: 288 from no load forecast
  on the opening day, 1 from a missing price forecast. The opening-day
  fallbacks cover a fifth of this five-day period.
- Median 1.27 ms per window; 3.8 s wall clock for 1,440 intervals.

Measured during the build (August 2026):

- **Real data (one site-year, GloBird ZeroHero, additive stress bonus, through
  the viz):** VCR 84.7% against `perfect_foresight_lp_advanced`; median
  1.17 ms per window against 1.04 ms for the wholesale path (1.13×).
- **Correctness gate:** full horizon with perfect prices gave an identical bill,
  identical zero-import credits and identical metered totals to
  `perfect_foresight_lp_advanced`.
- **Wholesale path unchanged by planner injection:** the pre- and post-change
  modules, run side by side over 1,200 intervals, agreed on all 11 trace series
  and every diagnostic field.
- **Solver strategy (synthetic benchmark, GloBird-shaped LP, 18 h, 600
  windows):**

  | Strategy | Median | p95 | Peak RSS |
  |---|---|---|---|
  | wholesale core, rebuild | 6.61 ms | 10.07 ms | 61 MB |
  | wholesale core, mutate | 0.77 ms | 1.03 ms | 61 MB |
  | advanced, rebuild | 7.37 ms | 10.52 ms | 61 MB |
  | **advanced, mutate** | **1.15 ms** | **1.42 ms** | 61 MB |
  | advanced, one model per time of day | 2.94 ms | 16.75 ms | 487 MB |

  The benchmark script (`bench_advanced_window.py`) is not in the repository.
