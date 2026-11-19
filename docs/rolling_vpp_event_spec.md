# rolling_vpp_event — Specification

## Purpose and scope

Extends `vpp_sc_event` and `vpp_tou_event` to imperfect foresight on both sides
of the VPP contract. Under perfect foresight the operator declares events from
prices it has already seen, and the household dispatches between events against
a horizon it knows exactly. Here the operator declares events from a forecast,
so it opens windows for spikes that never arrive and misses spikes that appear
late. The household plans into a horizon in which it cannot see events coming,
so the battery the operator takes over is at whatever state of charge the
household's plan left it.

Two coordination types, differing only in what the household does between
events:

| Coordination type | Between events | Perfect-foresight counterpart |
|---|---|---|
| `rolling_vpp_sc_event` | self-consumption | `vpp_sc_event` |
| `rolling_vpp_tou_event` | `AdvancedWindowPlanner` (TOU LP) | `vpp_tou_event` |

It is the rolling loop (`rolling_optimisation_spec.md`: decision instants,
causality, providers, terminal pin, fallback, retention, `RollingConfig`) with
`VppWindowPlanner` injected as the window planner. The operator LP is the
event LP of `vpp_sc_event_spec.md`; the TOU household planner is the one in
`rolling_perfect_foresight_lp_advanced_spec.md`. This document covers only what
is new when they are combined.

| Piece | File |
|---|---|
| Wrapper, phantom-block count, split invariant | `model/components/rolling_vpp_event.py` |
| `VppWindowPlanner` (two-mode planner) | `model/components/vpp_window_planner.py` |
| Declaration rule, `declared_mask`, `CapState` | `model/components/vpp_declaration.py` |
| `ShadowTrajectory` (natural/forced bounds, settlement) | `model/components/vpp_shadow.py` |
| Operator LP reference (`linprog`), packing, residue clamp | `model/components/vpp_operator_lp.py` |
| Operator LP as a persistent HiGHS model (used in runs) | `model/components/vpp_operator_lp_highs.py` |
| Validation, `resolve_min_export_price`, per-site runner, bulk runner | `model/orchestrate.py` |
| Single-site views | `model/dev/viz_rolling_vpp_sc_event.py`, `model/dev/viz_rolling_vpp_tou_event.py` |

Output is `SiteTraces`, a companion `VppTraces` carrying the natural/forced
split, and `RollingDiagnostics`. Billing and settlement downstream are the same
as for the perfect-foresight VPP types.

**Both agents' information degrades.** The operator's event declaration reads
the price forecast. The household's dispatch reads the price forecast (TOU
variant, where the plan has critical rates) and the site forecasts. The
natural/forced counterfactual is projected on site forecasts. Only the
committed interval is settled.

**Critical rates on the household TOU plan are supported.**
`AdvancedWindowPlanner` is reused unchanged and covers plans with and without
them; the household planner is given the settled prices for its day state, and
`ScenarioConfig.apply_export_order_nudge` (worker field
`apply_export_order_nudge`).

**Site forecasts are used when configured.** `run_rolling_vpp_event_site` builds
`UsageForecastProvider` and `SolarForecastProvider` from `RollingConfig` exactly
as `run_rolling_perfect_foresight_lp_site` does, and passes
`commit_actual_load` through. The component itself defaults both providers to
`None` (perfect site knowledge), so a caller that bypasses the site runner gets
perfect site foresight silently; `load_forecast_used` / `solar_forecast_used`
in the diagnostics record which applied. `state` is required whenever a site
forecast is configured, including on the SC variant where nothing else reads
it, because the providers resolve local days from it.

**Both agents share one forecast.** Operator and household read the same price
slice and the same solar and load windows. Where one entity holds both roles
(Energy Australia optimising the battery between events and dispatching it
during them) this is exact. Where they are separate (AGL Battery Rewards, with
between-event optimisation by the battery OEM's software) it is an assumption;
see Known limitations.

---

## Notation

| Symbol | Meaning |
|---|---|
| `t` | interval position in the period, `0 … T−1` |
| `k` | the interval being committed |
| `j` | position in the window, `0 … n−1`; absolute interval `k + j` |
| `n` | window length at `k`: the horizon, shorter near the period end |
| `Δ` | interval length in hours, `5/60` |
| `W` | event window length in intervals, `ceil(event_window_hours / Δ)` |
| `k₀` | first interval of the current event block |
| `t*` | last forecast spike within the declaration lookahead at `k` |
| `n_eff` | operator LP length, `t* − k + 1` |
| `σ[t]` | shadow self-consumption SOC at the end of interval `t` |
| `B_k` | remaining natural import budget at `k` |
| `C` | remaining annual VPP energy cap for the site |

Index labels are interval-end throughout. Local calendar days are local
interval-start dates, matching billing and `_build_day_map`.

---

## Event declaration

### The rule

At the decision instant for `k` the operator holds a price forecast over
`[k, k + n − 1]`, but only the first `W` intervals of it can declare an event.
An event is active at `k` when a forecast price above the threshold lies within
one event window of `k`:

```
lookahead(k) = [k, min(k + W − 1, k + n − 1)]
event(k)     = ∃ t ∈ lookahead(k) : rrp_forecast[t] > vpp_threshold
```

The threshold is strict. `vpp_threshold` is
`VppConfig.vpp_threshold_dollars_per_mwh`, as on the perfect-foresight path.

This is the causal form of the operator's existing rule: the operator takes
control for up to `event_window_hours` before a spike to position the battery,
so at any interval the question is whether a spike lies within that positioning
window ahead. The horizon is normally much longer than the event window, so the
operator routinely sees spikes it cannot yet act on; `W` is the reach of the
declaration and `n` only bounds it near the end of the period.

Spikes closer together than `W` produce one contiguous block, as in the
perfect-foresight classifier.

`declare()` reads only `rrp_forecast[:min(W, n)]`. The price slice may be longer
than `n` when the loop shrank the window after querying the price provider;
reading beyond `n` would let declaration see prices outside the window. A slice
shorter than the lookahead, or a non-finite value inside it, raises.

### Divergence from `classify_vpp_events`

`classify_vpp_events` scans backwards and skips any spike already covered by a
window opened by a later spike. That skip depends on which spikes have already
been processed, which is knowledge of the future, so the forward rule cannot
reproduce it.

The forward rule produces the skip-free union of the backwards windows. It is
strictly more permissive wherever spikes cluster. With `W = 3` and spikes at
`t = 5, 6`, the backwards scan marks `{4, 5, 6}` (the spike at 6 opens the
window and the spike at 5 is skipped); the forward rule marks `{3, 4, 5, 6}`,
because from `t = 3` the spike at 5 is within reach. Where no two spikes are
closer than `W` the two masks are identical.

`classify_vpp_events` is left unchanged, so a rolling run compared against
`vpp_sc_event` or `vpp_tou_event` differs for two reasons at once: forecast
error, and a mask that starts events earlier around spike clusters. Any value
capture ratio for these coordination types carries both, and they are not
separable within the run. The mask difference alone can be measured with
`declared_mask()`, which applies the forward rule to a whole settled series.

### Un-declaration

When the rule stops holding, the event ends immediately and the household
resumes at the next interval.

The alternative, a declared event that runs for a fixed duration, produces
phantom duration: one interval of forecast spike that evaporates on the next
run would commit the operator to a full `event_window_hours` of control with
nothing to dispatch into.

The cost is that a forecast hovering near the threshold inside the lookahead
can toggle control every five minutes, and each toggle re-anchors the shadow
trajectory. No hysteresis or minimum duration is applied; the block-length
diagnostics measure whether this happens.

### Lookahead truncation

The lookahead is `min(W, n)`, so anything that shrinks `n` below `W` shortens
it:

- **End of period.** The last `W − 1` intervals always have a truncated
  lookahead. Structural.
- **Start of period, price side.** `assemble_window` truncates the window when
  its infill history is exhausted.

Site forecast slices do not shorten the window: both providers compute
`n = min(n_intervals, T − k)`, which the loop has already applied, and they
signal unavailability by returning `None`, which the loop resolves as a
self-consumption fallback under `sc_fallback_no_load_forecast` /
`sc_fallback_no_solar_forecast`.

`n_lt_W` counts intervals where the lookahead was truncated, with the first and
last such interval in `n_lt_W_first` / `n_lt_W_last`. Truncation confined to
the leading and trailing edges is structural. Truncation mid-period means the
price provider is failing to fill windows, and the run declares fewer events
for a reason unrelated to forecast quality.

---

## Architecture

### One loop, one planner, two modes

`rolling_optimisation()` runs unchanged, with `VppWindowPlanner` supplied
through its `planner` argument. The planner holds a household sub-planner, the
operator LP, the shadow trajectory, the event state and the cap, and decides per
interval which mode governs.

```
VppWindowPlanner
├── household sub-planner   None (SC, deterministic) | AdvancedWindowPlanner
├── operator LP             HighsOperatorLp, one per site
├── shadow trajectory       ShadowTrajectory
├── event state             mode[t], n_eff[t]
└── cap                     CapState, per site
```

One planner rather than an operator planner and a household planner run
alternately, because of the TOU variant. On `rolling_vpp_tou_event` the
household is billed on total import and export including the flows the
operator forces during events, so the household's block-tier and zero-import
day state has to observe event intervals. Under two planners it would go blind
at exactly the intervals carrying the largest flows. A single planner that
forwards `observe_commit` unconditionally gets this right by construction.

Protocol properties:

- `requires_price_forecast` is always `True`. The operator reads the forecast
  to declare events even when the household plan's prices never move, which
  also satisfies the rule that a rolling run must have at least one genuine
  source of imperfect information.
- `needs_terminal_soc` is `False` on `rolling_vpp_sc_event` and `True` on
  `rolling_vpp_tou_event` (it tests whether a household sub-planner exists).
  On the SC variant nothing reads the loop's terminal SOC pin: the operator LP
  has no terminal constraint and household dispatch is deterministic, so the
  loop skips the self-consumption forward run that produces it.

### Call order

```
window_prices(k)    decide the mode; settle a block that has just ended
solve_window(k)     build the vector; stash the committed split
observe_commit(k)   advance the shadow, decrement the cap, record the split
```

Mode is decided inside `window_prices`, where the forecast slice arrives, and
cached for the `solve_window(k)` that follows. The price vectors returned are
the ones that enter the objective actually solved, so a retained plan is never
annotated with prices from the other mode.

```
window_prices(rrp_mwh, k, n):
    if cap exhausted:                   mode[k] = household   (terminal)
    elif rrp_mwh is None:               mode[k] = household
    elif declare(rrp_mwh, n, W):        mode[k] = operator; t* = last spike
    else:                               mode[k] = household
    if mode[k] = household and a block is open: settle the block
    operator:  return window_price_vectors(rrp[:n], k) on the operator stack
    household: TOU -> AdvancedWindowPlanner.window_prices(...)
               SC  -> zeros(n), zeros(n)
```

A degraded interval (no price forecast) reaches `observe_commit` without a
preceding `window_prices`. Its mode stays household, which is right: an
operator that cannot see prices cannot declare an event. Any open block is
settled there.

`mode` and `n_eff` are length-`T` arrays on the planner. `RetainedWindow` gains
no field. `run_with_planner` returns the planner alongside the outputs, because
the declared mask is not recoverable from the traces: an event interval where
the operator did nothing, or which fell back to self-consumption, carries no
forced flow.

### Solution vector layout

The loop reads the committed interval at fixed offsets on `n`-sized blocks:

```
OC, OD, OI, OE, OS, OQ = 0, n, 2n, 3n, 4n, 5n
                       = charge, discharge, import, export, soc, curtail
```

Both modes return a vector on those offsets. A vector laid out on `n_eff`-sized
blocks would put `x[n]` inside the discharge block and the loop would commit
the wrong numbers without error.

`pack_to_loop_layout` packs the operator LP's eight blocks (`charge, discharge,
n_imp, f_imp, n_exp, f_exp, soc, curtail`) as `import = n_imp + f_imp` and
`export = n_exp + f_exp`, with the four split blocks appended at `x[6n:10n]`.
Positions `n_eff … n−1` of every block are zero except SOC, which holds flat at
`soc[n_eff − 1]`. `n_eff` is recorded per interval so a flat tail is not read
as a plan.

The split is read from the LP's own blocks, not re-derived from
`_reconstruct_flows`: the reconstruction resolves simultaneous charge and
discharge, which matters for the trace and not for the split.

### Household mode

**SC variant.** Dispatch is deterministic and reads neither prices nor the
forecast. The planner runs `run_sc_with_battery` over the committed interval
only and builds:

```
charge[0]    = sol_to_bat + grid_to_bat
discharge[0] = bat_to_home                  (SC never exports from the battery)
import[0]    = grid_to_home + grid_to_bat
export[0]    = sol_to_grid
soc[0 … n−1] = soc after interval 0         (held flat)
curtail[0]   = curtail
```

Every other flow position is zero. Only interval 0 is ever committed and the
dispatch is deterministic, so computing the rest of the window is wasted work;
a retained plan on this path therefore shows only the committed interval.

It returns a vector, never `None`. `None` is the loop's infeasibility signal
and would be counted in `sc_fallback_infeasible`. The price pair returned is
zeros, because no objective is solved; the mode array distinguishes these
intervals.

**TOU variant.** `window_prices` and `solve_window` delegate to
`AdvancedWindowPlanner`. Neither is called in operator mode, which is safe
because `AdvancedWindowPlanner.window_prices` is pure; all of its state lives
in `observe_commit`, which is forwarded every interval. The household plans the
full window with no knowledge that an event may begin inside it, which matches
the contract: the retailer takes control without warning.

`run_with_planner` builds the household planner from the retail tariff, network
tariff and state, with the same horizon and battery as the loop. Its
curtailment cost is the VPP plan's `curtail_c_per_kwh`: absent means
curtailment is pinned off between events as well as during them (see
`docs/review/possible_bugs.md`, PB-8).

### Operator mode

`solve_window` anchors the shadow if this is the first interval of a block,
computes the shadow bounds over `[k, t*]`, and solves the operator LP. The
loop's `soc_final` argument is ignored. On an infeasible solve it returns
`None` and the loop falls back to self-consumption for that interval.

### `observe_commit`

Runs for every committed interval, both modes and all fallbacks:

1. Records the committed split. From the LP when a solution was stashed for
   `k`; otherwise (fallback) every flow is natural: `n_imp = import`,
   `n_exp = export`, `f_imp = f_exp = 0`. A fallback inside an event is counted
   in `event_intervals_fallback`; the event stays active.
2. If the interval is in operator mode, advances the shadow one interval and
   decrements the cap by the committed forced flows.
3. Forwards `(k, import_kwh, export_kwh)` to the household sub-planner, always.

---

## The shadow trajectory

The natural/forced split is defined against what the battery would have done
with no operator involvement. Under perfect foresight that counterfactual is one
self-consumption run across a known event window. Under rolling the boundary is
not known, so there is no fixed span to run across.

### Anchoring and extension

At the interval an event is declared the shadow is anchored at the real
battery's SOC entering that interval, and advanced one interval per committed
interval on the site data the LP saw at `j = 0`:

```
σ[k₀ − 1] = soc_committed[k₀ − 1]
σ[k]      = run_sc_with_battery(solar_window[0], load_window[0], σ[k − 1])
```

Under `commit_actual_load=True` the loop substitutes actuals into position 0, so
the shadow's committed prefix is on actuals and its projection on forecast,
with no branch in the planner. With `commit_actual_load=False` the shadow
degrades exactly as the dispatch does.

The committed prefix is only extended, never recomputed, so there is no bound on
event length.

Anchoring per block, rather than one shadow from the start of the period,
matches `vpp_sc_event` and `vpp_tou_event`: the counterfactual is what this
battery would have done from the state the household handed over. A
period-long shadow would be the truer never-joined counterfactual but would
drift from the real battery, so its bounds would stop describing a reachable
state.

### Projection

Within a solve at `k`, the shadow is projected across `[k, t*]` from its current
SOC using the solar and load windows, and discarded afterwards. `σ_prev[t]` is
the shadow SOC at the start of interval `t`, i.e. `σ[t − 1]`; the projection is
offset by one when building bounds because `run_sc_with_battery` returns
end-of-interval SOC.

### Natural export ceiling

Per interval, the same formula as `vpp_sc_event._compute_bounds`, on the shadow
SOC:

```
n_exp_hi[t] = max(0, solar[t] − load[t]
                     − min((capacity − σ_prev[t]) / η_c, power_kwh))
```

Per-interval rather than a budget, because forced export is the rewarded flow
and the operator wants less of it on the books. With an export budget the LP
could suppress export early (curtail, or charge instead) to bank headroom, then
book a later larger export as natural and under-report what the household is
owed. A per-interval ceiling prevents that. It also penalises bringing exports
forward without increasing them, which the perfect-foresight path does too; it
is kept so the accounting stays comparable.

### Natural import budget

Per interval from the shadow:

```
sc_n_imp[t] = max(0, load[t] − solar[t]
                     − min((σ_prev[t] − soc_min) · η_d, power_kwh))
```

The LP is constrained on the aggregate over the expected event span:

```
full_budget(k) = Σ_{t=k₀}^{k−1} sc_n_imp[t]  (committed, actuals)
               + Σ_{t=k}^{t*}   sc_n_imp[t]  (projected, forecast)
B_k            = max(0, full_budget(k) − Σ_{t=k₀}^{k−1} n_imp_committed[t])
```

Aggregate rather than per-interval, for the reason in `vpp_sc_event_spec.md`
(a per-interval ceiling punishes shifting import in time without increasing
it) and one specific to rolling: the household's counterfactual import
allowance is a property of the event, and the event's intervals are committed
one at a time, so constraining each separately would make the split depend on
when the LP chose to import, which the contract does not distinguish.

`full_budget(k)` moves with the forecast and with `t*`; it is the operator's
current estimate and is not monotone.

### The clamp

`B_k` can go negative before clamping: the LP may front-load natural import
against budget it expects the event to deliver, and the event may then end early
or the forecast shrink. `Σ n_imp ≤ negative` is infeasible against `n_imp ≥ 0`,
so the value is clamped at zero and every further kWh of import in that event
is forced. `n_imp_budget_clamps` counts the solves where this fired. It is a
count, not a volume; the volume is what retrospective settlement corrects.

### Retrospective settlement of the import split

When a block ends (at the first household interval after it, on a degraded
interval, or at the end of the period via `finalize()`), its true span
`[k₀, k_end]` is known and the shadow's committed prefix over it is entirely on
actuals, so the true budget is exact:

```
true_budget   = Σ_{t=k₀}^{k_end} sc_n_imp[t]
imp_total     = Σ_{t=k₀}^{k_end} (n_imp[t] + f_imp[t])
n_imp_settled = min(imp_total, true_budget)
f_imp_settled = imp_total − n_imp_settled
```

The block's `n_imp` and `f_imp` are rescaled to those totals, distributed across
its intervals in proportion to committed import. The correction is symmetric:
volume moves to `f_imp` when the LP over-claimed natural import and to `n_imp`
when it under-claimed.

Settlement is part of the design. It mirrors how the contract settles: the
operator dispatches on a forecast and settles afterwards on what happened.
Without it the household would be under-rewarded by the volume the LP booked as
natural against budget the event never delivered.

`SiteTraces` is untouched, since total import per interval does not change, so
dispatch, SOC and the energy component of the bill are unaffected. Proportional
distribution is exact for billing because the reward rates are flat in $/kWh
and only block totals reach `retail_other_credits`. The annual cap is affected
on `import_and_export` plans; see Annual cap.

Export needs no equivalent: `n_exp_hi[k]` at the committed interval is built
from `σ_prev[k]`, which is already in the committed prefix and on actuals.

`n_imp_resettled_kwh` reports the signed volume moved to `n_imp` (negative when
volume moved to `f_imp`) and `blocks_resettled` the number of blocks where
anything moved.

---

## The operator LP

Over `j = 0 … n_eff − 1`, absolute interval `t = k + j`. Eight blocks,
identical in structure to the event LP in `vpp_sc_event_spec.md`.

### Horizon

```
n_eff = t* − k + 1 ≤ min(W, n)
```

Beyond `t*` the operator has nothing to dispatch into. Whether the event
continues past `t*` is the next interval's question, decided against the next
forecast; planning further would plan against an extension that has not been
declared.

### Objective

```
min  Σ_j [  n_imp[j] ·  op_p_imp[j]
          + f_imp[j] · (op_p_imp[j] + r_imp)
          − n_exp[j] ·  op_p_exp[j]
          − f_exp[j] · (op_p_exp[j] − r_exp) ]
     + c_deg · Σ_j (charge[j] + discharge[j])
     + p_curtail · Σ_j curtail[j]
```

`r_imp` and `r_exp` are the household reward rates in $/kWh. `op_p_imp` and
`op_p_exp` come from the forecast RRP through the operator tariff's
`PriceStackOffsets.window_price_vectors`. Forced import costs the operator more
than natural import and forced export earns it less than natural export, so the
LP fills the natural blocks to their bounds first.

### Constraints

```
flow balance (n_eff rows):
  (n_imp + f_imp) − (n_exp + f_exp) − charge + discharge − curtail = load − solar

SOC continuity (n_eff rows), seeded from the real battery's SOC entering k:
  soc[j] − η_c·charge[j] + discharge[j]/η_d − soc[j−1] = 0

natural import budget (1 row):       Σ_j n_imp[j] ≤ B_k
natural export ceiling (bounds):     0 ≤ n_exp[j] ≤ n_exp_hi[k + j]
remaining annual cap (1 row):        Σ_j f_exp[j] ≤ C                 export_only
                                     Σ_j (f_imp[j] + f_exp[j]) ≤ C    import_and_export
site export limit (n_eff rows):      n_exp[j] + f_exp[j] ≤ export_limit_kwh
minimum export price (bound):        f_exp[j] = 0 where op_p_exp[j] < min_export_price / 1000
no-grid-charge (masked intervals):   charge[j] + curtail[j] ≤ max(0, solar[j] − load[j])
curtail (bound):                     0 ≤ curtail[j] ≤ max(0, solar[j]); pinned to 0 when p_curtail is None
```

The minimum export price is compared against the forecast operator export
price, so the operator may forgo export at an interval that settles above the
threshold, or export at one that settles below. That is an information effect
this path measures. Comparing against `op_p_exp` rather than RRP is shared with
the perfect-foresight path (PB-5).

The curtail upper bound is `max(0, solar)`, never `solar`. EDP solar is
negative at some intervals (overnight inverter parasitic draw; see
`SiteTraces.sol_to_home_kwh`), and a negative upper bound against a zero lower
bound is an infeasible variable that sinks any window containing a night
interval. Randomised test fixtures generate solar on `[−0.01, 0.6]` for this
reason.

### No terminal SOC

The LP does not pin `soc[n_eff − 1]`, and `soc_final` from the loop is ignored.
The operator dispatches on event economics and leaves whatever remains; the
household's next interval starts from that state. The backstop against
emptying the battery at low value is the minimum export price, not a terminal
constraint.

### Feasibility is not guaranteed

On the perfect-foresight path the event LP is always feasible, because its
bounds come from a self-consumption run started at the same SOC, so the SC
dispatch is a valid assignment. Under rolling the shadow and the real battery
diverge (the shadow is anchored at `k₀` and advances on its own SOC), so SC is
not a certificate. A window can be infeasible: battery full, surplus solar,
curtailment pinned off, and export blocked by a finite export limit, an
exhausted cap or the minimum export price. The planner counts it in
`operator_infeasible` and returns `None`; the loop falls back to
self-consumption for the interval. A count that is more than trivial makes the
run suspect.

Any HiGHS status other than optimal is treated as infeasible (the same
behaviour as the wholesale window solver; PB-6).

### Price ordering

`op_p_imp[j] ≥ op_p_exp[j]` at every interval, or the LP can import and export
in the same interval for free. The reward rates cannot cause a crossover: the
effective forced rates `op_p_imp + r_imp` and `op_p_exp − r_exp` only move the
pair further apart. The operator's price stack can, and is checked by
`build_price_stack_offsets` across the market price range when the runner builds
the operator offsets. The household TOU vectors are checked by
`AdvancedWindowPlanner` at construction.

### Solver

The planner uses `HighsOperatorLp`: one `highspy.Highs` model per site, built
once and refreshed between windows with `changeColsCost`, `changeColsBounds` and
`changeRowsBounds`, retaining the basis, as `rolling_optimisation._WindowSolver`
does on the wholesale path. Consecutive windows differ by one interval of shift
and a data refresh, so the previous optimal basis is nearly optimal for the
next problem. The cost of this LP is the solve, not the matrix assembly, so
retaining the basis is the lever; avoiding the rebuild alone does not help.

The model is built at `W` intervals. A window with `n_eff < W` deactivates the
tail: every flow variable bounded to `[0, 0]` and the flow balance RHS set to
zero, so SOC continuity carries `soc[n_eff − 1]` forward and the tail is
feasible, inert and costless. Solving `W` intervals with a dead tail is exactly
equivalent to solving `n_eff`, which is asserted against the from-scratch
builder. Two rows need care to keep the structure fixed: the no-grid-charge
rows are always present (when the plan has windows) and made non-binding by
raising the RHS; the minimum export price is a variable bound, never a row.

Every bound and RHS is rewritten in full on every call, including the dead tail
and unchanged values, so no window inherits a constraint from its predecessor
(`test_repeated_solves_do_not_leak_state`).

`vpp_operator_lp.solve_operator_window` builds the same LP from scratch through
`vpp_sc_event._solve_event_lp` and `linprog`. It is the reference the HiGHS
model is tested against and is not used in runs.

**Solver residue.** HiGHS returns a primal solution to its feasibility
tolerance, so a variable at a zero lower bound can come back fractionally
negative. `pack_to_loop_layout`, shared by both solver paths, clamps flow
residues to zero before the split blocks are summed into the headline import
and export, so the split and the headline stay exactly consistent. A negative
flow beyond `1e-7` (HiGHS's own primal feasibility tolerance) raises, since
anything larger is a bound problem rather than noise. SOC is not clamped: it is
a state with its own bounds.

---

## Annual cap

The cap is per site and exact. `C` starts at `VppConfig.kwh_limit`, and each
committed operator-mode interval decrements it by the committed forced flows
selected by `limit_applies_to`:

```
C ← C − f_exp[k]                    export_only
C ← C − (f_imp[k] + f_exp[k])       import_and_export
```

Both settings are in use across real offerings. Only committed flows count;
planned forced flows in the discarded tail of a window never touch it. A cap
decremented from the plan would exhaust roughly a horizon's worth too fast and
look like plausible under-dispatch.

**Settlement adjusts the cap.** On an `import_and_export` plan the cap counts
forced import, which retrospective settlement rewrites. When a block closes the
cap is decremented by the settlement delta as well, or the constraint is
enforced against one number while `VppTraces`, the bill and the reward report
another. The cap can therefore be exceeded retrospectively, since the
correction lands after the intervals are committed. That is the honest outcome:
the operator dispatched against a projected budget and settlement revised it.
`cap_exhausted_at` then records the block-closing interval. Under `export_only`
no adjustment applies.

**Exhaustion is terminal.** When `C` falls to within `1e-9` kWh of zero (a
solver-tolerance allowance, since `C` is a difference of accumulated LP
outputs) the site stops participating: no further events are declared and the
household strategy runs to the end of the period. `C` only shrinks, so there is
no re-entry and no per-day or per-month reset.

---

## Minimum export price

### What it is for

On plans where the per-kWh reward is zero or near zero, the event LP would
dispatch forced export at any positive price inside an event window, exhausting
the annual cap on the earliest events of the year whether or not later events
would be worth more. The threshold restricts forced export to intervals worth
spending the cap on.

### One value per scenario

VPP operators set one value for the fleet, not one per site. The threshold is
resolved once per scenario by `resolve_min_export_price`, before the worker pool
spawns, and every site's planner reads the same scalar for the whole period.
There is no within-run update: `C` is per-site state inside independent workers
with no channel between them, so a threshold responding to realised depletion
would have to be per site.

### Modes and the floor rule

`VppConfig.min_export_price_mode`:

- **`fixed`** (default): `min_export_price_dollars_per_mwh` is used as stated.
  It must be present.
- **`based_on_this_year`**: derived from the modelled year's own prices and
  cohort load. The configured value is optional and acts as a floor:

  ```
  min_export_price = max(derived, configured or 0)
  ```

  The configured value is the operator's own floor and the derivation can only
  tighten it. This also covers the case where the derivation returns zero
  because the available throughput cannot exhaust `kwh_limit`, which without a
  floor would remove the restriction at exactly the configurations where event
  intervals are scarcest.
- **`based_on_last_year`**: not built; rejected at validation. See below.

These rules apply to all four VPP coordination types. The perfect-foresight
runner resolves through the same `resolve_min_export_price`, so the same YAML
gives the same threshold on both paths and the value capture ratio between them
carries no threshold difference.

Units are $/MWh on this path: `resolve_min_export_price` returns $/MWh and
`VppWindowPlanner` applies its own `/1000` against `op_p_exp`. The
perfect-foresight runner divides by 1000 because `vpp_sc_event` takes $/kWh. A
misplaced conversion here would put the threshold three orders of magnitude too
low, and the only symptom would be `f_exp_blocked_by_price = 0`, which a
legitimately zero threshold also produces.

### Derivation under `based_on_this_year`

`_compute_min_export_price` on the modelled year, restricted to the event mask
from `classify_vpp_events` on the same settled prices:

```
avg_load_kw[t]         = mean over the cohort of min(load_kw[t], power_kw)
effective_export_kw[t] = max(0, power_kw − avg_load_kw[t])
contribution[t]        = effective_export_kw[t] · Δ

Restrict to event-mask intervals. Sort by price descending, accumulate
contribution. The price at which the cumulative sum first reaches kwh_limit is
the estimate; if it never does, the estimate is 0.
```

The cohort is the scenario's own site list, with `gross_load_kwh` unscaled.
`power_kw` is `scenario.battery_spec.power_kw`. Every input is already in memory
at the scenario level.

Restricting to event intervals matters because forced export can only happen
inside events; counting all intervals overstates throughput and sets the
threshold too high. An empty mask resolves directly to the floor with
`n_mask_intervals = 0`, rather than being passed to `_compute_min_export_price`,
whose `vpp_event.any()` test would otherwise fall through to all-year prices.

The mask comes from `classify_vpp_events`, not from this component's forward
rule, so that a rolling run and its perfect-foresight benchmark resolve to the
same number. The two masks differ by at most `W − 1` intervals per spike
cluster, and not at all at a one-interval window.

The estimate assumes every site exports at full effective power at every
selected interval, which overstates throughput and sets the threshold somewhat
high. `cap_used_kwh` against `kwh_limit` across the cohort measures this.

**This is a look-ahead.** Deriving the threshold from the modelled year's own
prices uses information a rolling operator would not have. It is accepted
deliberately: the threshold is a fleet planning parameter chosen before the
period, not a dispatch decision, and `based_on_this_year` stands in for an
operator who guessed the year well. It earns its place over `fixed` because a
YAML sweeping battery sizes, states and years needs a different number in each
cell. Dispatch causality is untouched: no per-interval decision reads anything
the decision instant does not hold.

### What is recorded

`MinExportPriceResolution` is carried on `ScenarioResult` and written to the
scenario summary under MINIMUM EXPORT PRICE: the mode, the operative value, the
configured value, the derived value, which of the two the floor rule selected
(`configured`, `derived` or `equal`), the number of mask intervals, the
throughput available across them, and whether it reached `kwh_limit`. Without
it three different situations give the same number (a positive derived
estimate, a zero because throughput fell short of the cap, and a zero because
the mask was empty), and a `based_on_this_year` run is indistinguishable from a
causal one after the fact.

---

## Output assembly

`SiteTraces` is assembled by the loop's `_reconstruct_flows` from committed
charge, discharge, curtail and SOC, unchanged.

`VppTraces` is assembled by the wrapper from the split the planner recorded per
interval, after `finalize()` has settled any block still open at the period end:

| Interval | `n_imp` | `f_imp` | `n_exp` | `f_exp` |
|---|---|---|---|---|
| Household mode | import | 0 | export | 0 |
| Operator mode | LP `n_imp[0]` (then settled) | LP `f_imp[0]` (then settled) | LP `n_exp[0]` | LP `f_exp[0]` |
| Fallback | import | 0 | export | 0 |

`rolling_optimisation()` keeps its `(SiteTraces, RollingDiagnostics)` return.
The wrapper reads the split off the planner and returns the triple; the loop
knows nothing of VPP.

The wrapper checks that `n_imp + f_imp` and `n_exp + f_exp` reconstruct
`SiteTraces.import_kwh` and `export_kwh` to `1e-6` kWh and raises otherwise.
The split and the headline series come from different places (the LP's own
variables against the loop's flow reconstruction), so a packing error would
leave both individually plausible.

---

## Interface

```python
def rolling_vpp_event(
    site_id: str,
    solar_kwh: pd.Series,
    load_kwh: pd.Series,
    prices_mwh: pd.Series,                   # SETTLED; post-run phantom count only
    op_price_offsets: PriceStackOffsets,     # operator tariff
    forecast_provider,                       # required
    battery_spec: BatterySpec,
    soc_init_kwh: float,
    soc_final_kwh: float,
    config: RollingConfig,
    vpp_threshold_dollars_per_mwh: float,
    event_window_hours: float,
    vpp_exp_reward: float,                   # $/kWh on f_exp
    vpp_imp_reward: float,                   # $/kWh on f_imp
    vpp_kwh_limit: float,
    vpp_limit_applies_to: str,               # 'export_only' | 'import_and_export'
    min_export_price_dollars_per_mwh: float, # resolved fleet-wide scalar
    gap_strategy: str = 'self_consumption',  # | 'tou_advanced'
    export_limit_kwh_per_interval: float = np.inf,
    interval_hours: float = 5 / 60,
    p_curtail: float | None = None,          # $/kWh; None pins curtailment off
    no_grid_charge_mask: np.ndarray | None = None,
    household_planner=None,                  # tou_advanced only; built if None
    retail_tariff=None,                      # } required on tou_advanced
    network_tariff=None,                     # } to build the household planner
    state: str | None = None,                # }
    load_forecast_provider=None,             # None = perfect load knowledge
    solar_forecast_provider=None,            # None = perfect solar knowledge
    commit_actual_load: bool = True,
    progress=None,
) -> tuple[SiteTraces, VppTraces, RollingDiagnostics]
```

`run_with_planner` takes the same arguments and also returns the
`VppWindowPlanner`, for callers that need the `mode` array (the viz shading, a
mask comparison against `classify_vpp_events`).

`gap_strategy` mirrors the parameter on `vpp_sc_event`: `self_consumption` →
`rolling_vpp_sc_event`, `tou_advanced` → `rolling_vpp_tou_event`.

The component raises on: an unknown `gap_strategy`; `tou_advanced` without
either a household planner or all of `retail_tariff`, `network_tariff` and
`state`; a household planner on `self_consumption`; no price forecast provider;
`prices_mwh` of a different length from the site traces; and a horizon shorter
than `W` intervals.

`orchestrate.run_rolling_vpp_event_site` is the per-site entry point shared by
the bulk worker and the viz scripts. It builds the site forecast providers from
`RollingConfig`, sets `soc_init_kwh` and `soc_final_kwh` to the backup reserve,
and returns `(sc_result, SiteTraces, VppTraces, RollingDiagnostics)`, computing
the self-consumption baseline if not supplied.

---

## Configuration

```yaml
coordination_type: rolling_vpp_sc_event      # or rolling_vpp_tou_event
op_retail_tariff_id: <wholesale_exposed operator plan>
retail_tariff_id: <household plan with a vpp: block>
rolling:
  forecast_source: predispatch
  horizon_hours: 18
  ...
```

```yaml
# under vpp: in the household retail tariff
  min_export_price_mode: based_on_this_year   # fixed (default) | based_on_this_year
  min_export_price_dollars_per_mwh: 300.0     # required under fixed; floor otherwise
```

`min_export_price_mode` is a `VppConfig` field, a `Literal` that includes
`'based_on_last_year'` so a YAML naming it parses and then fails validation with
a message about implementation status rather than a bad value.

The rolling VPP types are members of both `ROLLING_COORDINATION_TYPES` and
`VPP_COORDINATION_TYPES` in `scenario.py`.

### Scenario validation

In `ScenarioConfig`:

- `op_retail_tariff_id` required (all `VPP_COORDINATION_TYPES`).

In `validate_scenario`:

- `rolling` required (all `ROLLING_COORDINATION_TYPES`).
- The retail tariff must carry a `vpp:` block.
- `min_export_price_mode='based_on_last_year'` is rejected, so a multi-scenario
  run aborts before any ingest. Applies to all four VPP types.
- `min_export_price_dollars_per_mwh` required under `fixed`. Applies to all four
  VPP types. Because `fixed` is the default, a tariff naming neither a mode nor
  a value fails; it no longer means "derive from this year".
- `horizon_hours / Δ ≥ W`, compared in intervals with the same rounding the
  loop uses. A shorter horizon would truncate every declaration lookahead, so
  the run would declare fewer events for a configuration reason. Also enforced
  in the component.
- `rolling_vpp_tou_event` requires a `tou` household tariff;
  `rolling_vpp_sc_event` places no constraint on it.

The operator tariff's type (`wholesale_exposed`) and its price ordering are not
checked in `validate_scenario` for the rolling VPP types; they are enforced when
the runner calls `build_price_stack_offsets`.

---

## Orchestration

One runner, `_run_rolling_vpp_event`, serves both coordination types. It follows
the `_run_rolling_perfect_foresight_lp` pattern: the price forecast provider is
built once per worker process by `_init_rolling_worker` rather than pickled per
site, and a site with `causality_ok=False` aborts the scenario. Unlike the
advanced rolling path, the price provider is always built, because the operator
declares events from it whatever the household plan reads.

Per-scenario work before the pool spawns:

- Build the operator `PriceStackOffsets` from `op_retail_tariff_id`.
- Convert `exp_reward_c_per_kwh`, `imp_reward_c_per_kwh` and
  `curtail_c_per_kwh` from cents (`_C = 100.0`), as the perfect-foresight VPP
  runner does.
- Resolve the minimum export price with `resolve_min_export_price`.
- Build `no_grid_charge_mask` from `VppConfig.no_grid_charge_windows` against
  `scenario.state`.

The worker `_rolling_vpp_site_worker` selects `gap_strategy` from
`coordination_type`, runs the self-consumption baseline and its bill, calls
`run_rolling_vpp_event_site`, and bills the result with `vpp_traces`. The
household TOU planner is built inside the component, not by the runner.

The worker args tuple is defined by `_ROLLING_VPP_ARG_FIELDS`, built in the
runner and unpacked in the worker. `test_orchestrate_rolling_advanced_wiring.py`
compares the three definitions (field constant, builder, unpacker) by name and
order: arity alone would not catch two swapped fields.

Outputs: `VppTraces` per site on `ScenarioResult.vpp_traces`,
`MinExportPriceResolution` on `ScenarioResult.min_export_price_resolution`, and
the VPP diagnostics as columns of `rolling_diagnostics.csv`.

---

## Diagnostics

Fields on `RollingDiagnostics`, populated through `plan_diagnostics()` (which
does not mutate planner state). All are `None` on non-VPP paths, so a reader can
tell "not applicable" from "zero".

| Field | Meaning |
|---|---|
| `events_anchored` | declared event blocks |
| `event_intervals` | committed intervals under operator control |
| `event_intervals_fallback` | event intervals resolved by self-consumption fallback |
| `event_blocks_no_spike` | blocks in which no settled price exceeded the threshold: phantom events, opened on a forecast spike that never arrived |
| `event_block_len_min` / `_max` / `_mean` | block length distribution |
| `event_blocks_len_1` | single-interval blocks (the flapping measure) |
| `n_lt_W`, `n_lt_W_first`, `n_lt_W_last` | intervals with a truncated declaration lookahead, and the first and last |
| `n_imp_budget_clamps` | operator solves where `B_k` was clamped at zero (count, not volume) |
| `n_imp_resettled_kwh` | signed volume moved to `n_imp` by retrospective settlement |
| `blocks_resettled` | blocks where settlement moved volume |
| `f_imp_total_kwh`, `f_exp_total_kwh` | forced flows after settlement |
| `f_exp_blocked_by_price` | solved event intervals where `op_p_exp` at the committed interval was below the minimum export price, so `f_exp` was pinned to zero |
| `operator_infeasible` | operator windows with no optimal solution |
| `cap_limit_kwh`, `cap_used_kwh`, `cap_remaining_kwh` | the annual cap (remaining floored at zero) |
| `cap_exhausted_at` | interval index at which the cap was exhausted, or `None` |

`event_blocks_no_spike` needs settled prices, which the planner never sees. The
wrapper computes it after the run with `count_blocks_without_a_settled_spike`
over the `mode` array. The wrapper also re-reads `plan_diagnostics()` after
`finalize()`, because the loop builds `RollingDiagnostics` before the last open
block is settled.

---

## Failure handling

| Condition | Handling |
|---|---|
| Operator LP not optimal | `None` to the loop → SC fallback for the interval; `operator_infeasible` and `event_intervals_fallback` counted; event stays active |
| No price forecast at `k` | loop's degraded path; mode household; any open block settled |
| No site forecast at `k` | loop's SC fallback (`sc_fallback_no_load_forecast` / `_no_solar_forecast`); split all natural |
| Flow residue beyond `1e-7` | raises in `pack_to_loop_layout` |
| Split does not reconstruct the headline series | raises in the wrapper |
| Cap exhausted | site stops declaring events for the rest of the period |
| `causality_ok=False` | scenario aborted |
| Shadow anchored at an SOC outside `[soc_min, capacity]` | raises |

---

## Verification

**No exact correctness gate.** On the wholesale path `PerfectForecastProvider`
at full-period horizon reproduces `perfect_foresight_lp` exactly. Here the
declaration rule differs from `classify_vpp_events` by construction, so even a
perfect forecast gives a different event mask. What is checked instead:

- The forward rule's mask is a superset of `classify_vpp_events`, marks the
  same spike intervals, and equals it exactly when no two spikes are closer than
  `W`.
- Spikes closer than `W` form one contiguous block.
- Shadow bounds match `vpp_sc_event._compute_bounds` on the same trajectory;
  the committed prefix is extended, not recomputed; settlement never changes
  total import.
- The HiGHS operator LP matches the `linprog` reference across span lengths, on
  randomised windows, with binding cap, budget and price pin, and with negative
  solar; infeasible windows agree; repeated solves do not leak state.
- On the TOU variant the household's day state includes flows committed during
  events, forced export consumes household tier headroom, and the household
  plan has no visibility of upcoming events.
- The SC variant's committed dispatch matches plain self-consumption exactly
  and never reports an infeasible fallback.
- The split reconstructs the headline import and export; every split series is
  non-negative.
- The cap reconciles with the settled forced flows it counts; `export_only`
  caps are unaffected by import settlement; settlement may push the cap past its
  limit.
- A block open at the period end is settled, and settlement bounds natural
  import by the true budget.
- Phantom blocks are counted per block, and a block whose spike materialises is
  not a phantom.
- The minimum export price pins forced export when above the spike and permits
  it when below or zero.
- The worker args tuple agrees across its three definitions; both rolling VPP
  types route to their own runner.

**Causality.** The loop's causality check (forecast run time at or before the
decision instant, no infill read at or after it) applies unchanged. The
minimum export price is outside it by construction: under `fixed` it is stated,
and under `based_on_this_year` it is derived from the modelled year
deliberately, which is why the resolution record carries the mode.

**Non-vacuous fixtures.** This path has per-event state, a running budget and an
annual cap, so tests assert that their fixture reached the case under test
before the real assertion: that an event was declared, the budget clamp fired,
the cap was reached, a fallback occurred inside an event, an operator window was
infeasible.

**Single-site views.** `viz_rolling_vpp_sc_event.py` and
`viz_rolling_vpp_tou_event.py` run one site for one year through
`run_rolling_vpp_event_site`, the same code as the bulk run, for daily visual
inspection.

---

## Known limitations

**No exact benchmark.** Any value capture ratio against `vpp_sc_event` or
`vpp_tou_event` mixes forecast error with the declaration rule difference. Fix
path: drop the skip from `classify_vpp_events` and re-run the perfect-foresight
components, which makes the two masks agree under a perfect forecast and
restores an exact gate. Not taken, because it invalidates completed runs.

**Run-on blocks.** An event stays declared while any forecast price within `W`
exceeds the threshold. Through a volatile period that is almost always true, so
consecutive events merge into long blocks. Un-declaration prevents phantom
duration after a spike passes; it does nothing about persistently elevated
forecasts. Nearly harmless on `rolling_vpp_sc_event`, where the operator
dispatches close to self-consumption with no spike to serve; costly on
`rolling_vpp_tou_event`, where the household loses TOU optimisation for the
whole block. A shorter event window shrinks it, but the driver is forecast
persistence. No hysteresis or minimum duration is applied either, so a forecast
hovering at the threshold can also flap.

**Forced import dominates under forecast error.** The operator plans `[k, t*]`
and commits only `k`, so on a forecast spike it repeatedly commits the charging
leg and reaches the discharge leg only if the spike survives. With most blocks
phantom, mostly it does not. The behaviour is economically disciplined (forced
charge costs `op_p_imp + r_imp` and forced export earns `op_p_exp − r_exp`, so
the round trip pays only on extreme forecasts) and is kept as a faithful
illustration of what imperfect price forecasts cost.

**The cap guard is one-sided.** The minimum export price conserves the cap
against low-value forced export, but nothing restricts forced import, which on
an `import_and_export` plan can consume most of the cap. An import-side guard
would be an operator-strategy decision, not a mechanical fix.

**Household value capture can exceed 100%.** Forced flows are rewarded per kWh
whether or not the operator profited, so the household bill can fall below the
perfect-foresight bill while operator-side value capture is worse. Any value
capture ratio quoted for these types must name the side it is measured from.

**Shared forecast between agents.** Where operator and household are different
entities, two forecasting stacks would not agree, and the model understates the
friction between them. Fix path: a second provider pair on `RollingConfig`
consumed by the operator side only. Additive; nothing in this design blocks it.

**The threshold carries no realised information.** It is resolved once and
cannot respond to how fast the cap depletes. Fix path: a two-pass calibration,
pass 1 supplying realised cap depletion and pass 2 re-deriving the threshold.
It doubles run time and changes one orchestrator function, touching neither the
planner nor the LP.

**`based_on_this_year` is a look-ahead.** It biases operator-side results
favourably by an unmeasured amount, so operator-side value capture from such a
run is an upper bound on what the threshold mechanism can deliver and is not
comparable with a `fixed` run. Fix path: `based_on_last_year` (not built).
Interim: report the mode with any figure derived under it.

**The estimator's mask is not the mask the run declares.** The threshold is
derived over `classify_vpp_events` intervals, while the run dispatches over the
forward skip-free union and over forecast-declared blocks that never carried a
settled spike. The threshold is therefore mildly permissive relative to the
opportunity the run has. Deliberately not fixed: matching the estimator to the
declaration rule would put a threshold difference inside every value capture
ratio against the perfect-foresight benchmark.

**Horizon not swept for this path.** The production configs use 18 hours,
carried over from the wholesale and advanced rolling paths. This path has
structure those do not: an event window the horizon must contain, and an
operator LP whose length is bounded by it. See Open questions.

**`c_deg` not refit under rolling**, on any path.

**Information timing at the decision instant.** The model assumes the settled
price and refreshed forecast are available at the start of interval `k`; real
publication lands roughly 9 and 30 seconds in. As on every rolling path.

**Decomposition under charge/discharge ties.** `grid_to_home` can go negative
when the LP returns a simultaneous charge/discharge tie, as in
`vpp_sc_event_spec.md`. Billing is unaffected because import and export come
from the LP's own blocks.

---

## `based_on_last_year` (not built)

Rejected at validation. Preliminary design:

- The configured value is used for the first modelled year. For later years the
  estimate comes from `_compute_min_export_price` on year `Y − 1`'s prices and
  cohort load, with the same floor rule.
- The first year is identified by data availability, not a hardcoded year: if
  the prior year's price file or site metadata is absent, the configured value
  is used and the branch recorded. This keeps working as the dataset is
  extended backwards.
- The cohort is whichever sites year `Y − 1` selected, with no attempt to match
  membership against year `Y`. The threshold is fleet-level and the cohort is a
  sample for a load shape; matching membership would mean loading two years of
  traces for one scalar.
- `power_kw`, `kwh_limit`, `limit_applies_to` and `event_window_hours` come
  from the scenario being run; only prices and load shape are historical.
- It reads a full year of prior-year traces, a second ingest-sized read. It
  depends only on `(year − 1, state, power_kw, kwh_limit, limit_applies_to,
  event_window_hours, vpp_threshold)`, so it would be cached in-process on that
  key: a YAML sweeping ten battery sizes over one state and year would load the
  prior year once, since only the `min(load_kw, power_kw)` reduction depends on
  battery size.
- It needs a causality check that `based_on_this_year` does not: every input
  from year `Y − 1` and no read touching year `Y`. The current year's traces
  and prices are both in memory, and a threshold accidentally computed from
  them would look entirely plausible.

---

## Validation

S0008, NSW 2022, AGL Battery Rewards ($1/kWh rewards, $300/MWh minimum export
price, $1,000/MWh declaration threshold, 250 kWh cap), 13.5 kWh battery,
predispatch forecasts, 18-hour horizon, 10-hour event window,
`rolling_vpp_sc_event`:

| Measure | Rolling | Perfect foresight |
|---|---|---|
| Declared blocks | 189 | |
| Blocks with no settled spike | 140 | |
| Single-interval blocks | 50 | |
| Longest block | 3,124 intervals (10.8 days) | |
| Mean block length | 46.6 intervals | |
| Forced import | 227.8 kWh | 7.1 kWh |
| Forced export | 40.5 kWh | 45.1 kWh |
| Intervals with `f_exp` pinned by price | 4,709 | |
| Cap used | 232.3 of 250 kWh | |

The household bill fell below the perfect-foresight bill. With
round-trip efficiency 0.9025 and $1/kWh rewards, forced charge-then-export pays
only above about $2,440/MWh.

These figures predate the cap adjustment on settlement. In that run `f_imp +
f_exp` reached 268.3 kWh against the 250 kWh cap while `cap_used_kwh` reported
232.3; the 36.0 kWh difference was exactly `n_imp_resettled_kwh`.

Solve cost, measured in isolation on the same run's interval counts (about
8,800 event intervals and 96,000 household intervals per site-year):

| `n_eff` | `linprog` | HiGHS warm-start | Speed-up |
|---|---|---|---|
| 10 | 2.68 ms | 0.46 ms | 5.9× |
| 60 | 4.36 ms | 0.49 ms | 8.9× |
| 120 | 6.97 ms | 0.50 ms | 13.8× |

Matrix assembly is 0.25 ms, 3–8% of a `linprog` solve. A fixed-size model
without basis retention was 0.6× to 1.0× the speed of rebuilding. With the
basis retained, cost is nearly flat in span, which is what makes the fixed-size
model viable. Solving only the committed interval on the SC household path
(0.004 ms against 0.290 ms for a 216-interval window) saves about 27.5 s per
site-year, and skipping the terminal SOC pin on the SC variant about 30.5 s.

---

## Open questions

1. **Horizon length for this path.** A run with `PerfectForecastProvider` at 18
   hours, compared against `vpp_sc_event`, bounds it: with prices exact, the
   remaining gap is horizon truncation plus rolling commitment plus the
   declaration rule difference. The mask difference is measurable separately
   (`declared_mask` against `classify_vpp_events` on the same prices), so it
   can be subtracted and the rest attributed to horizon. Cheaper than a sweep,
   and answers whether a sweep is needed.
2. **Whether hysteresis or a minimum event duration is worth adding.** Run-on
   blocks, not flapping, dominated the one measured run; that may differ with a
   shorter event window.
3. **Whether `commit_actual_load=False` changes the split.** Unmeasured on any
   path. Here it would put the shadow's committed prefix on forecast too, so the
   natural/forced split would become forecast-dependent at the committed
   interval.
