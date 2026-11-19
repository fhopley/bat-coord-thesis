# rolling_optimisation — Specification

## Purpose and scope

`model/components/rolling_optimisation.py` computes battery dispatch under
imperfect foresight. At each 5-minute decision instant it plans over a finite
horizon using the forecasts available at that moment, commits the current
interval's action, advances one interval and re-plans. It returns a
`SiteTraces` of the usual shape plus a `RollingDiagnostics`.

It exists to close the gap between the perfect-foresight benchmark and
observed battery behaviour, and to attribute that gap rather than absorb it
into a fitted parameter.

This document owns the **shared rolling mechanism**: the loop, causality, the
horizon and infill, the terminal SOC pin, the window LP, the planner protocol,
failure handling, diagnostics and `RollingConfig`. Four coordination types are
built on it (`ROLLING_COORDINATION_TYPES` in `model/core/scenario.py`):

| Coordination type | Planner | Spec |
|---|---|---|
| `rolling_perfect_foresight_lp` | default wholesale planner (this document) | here |
| `rolling_perfect_foresight_lp_advanced` | `AdvancedWindowPlanner` | `rolling_perfect_foresight_lp_advanced_spec.md` |
| `rolling_vpp_sc_event`, `rolling_vpp_tou_event` | `VppWindowPlanner` | `rolling_vpp_event_spec.md` |

The coordination type names the LP being wrapped, matching the
`perfect_foresight*` family; the module keeps the mechanism's name.

The inputs that can be imperfect are price (`forecast_providers_spec.md`),
household load (`usage_forecast_spec.md`) and solar (`solar_forecast_spec.md`).
Each is independently perfect or forecast, which gives the measurement ladder.

**A rolling run must have at least one genuine source of imperfect
information.** A plan with perfect site knowledge and no price-dependent
element is perfect foresight over a truncated horizon, whatever it is called,
and is refused. For the wholesale type the price stack always depends on RRP,
so this holds automatically. The advanced type enforces it in
`validate_scenario`.

---

## The rolling loop

Intervals are indexed `0 .. T_p−1` by `datetime_aest` (interval-end,
`Etc/GMT-10`). The **decision instant** for interval k is the moment it
begins, `index[k] − interval`. It is the reference for every causality
question.

For each k:

1. Window `[k, k_end]`, `k_end = min(k + H − 1, T_p − 1)`, with
   `H = round(horizon_hours / interval_hours)`.
2. If the planner needs prices, ask the price provider for the forecast at
   the decision instant. `None` → commit self-consumption for k and advance.
   The provider may return a shorter window, and the window shrinks to match.
3. Window load and solar: actual, or from the load and solar providers when
   configured. A provider returning `None` → self-consumption for k. With
   `commit_actual_load` (default `True`), position 0 of both is overwritten
   with the actual value, so forecasts shape the plan, not the execution.
4. Window prices from the planner (`planner.window_prices`).
5. Terminal SOC target (see Terminal SOC).
6. Solve the window with `soc_init` = the last committed SOC. `None`
   (infeasible) → self-consumption for k.
7. Commit interval k: charge, discharge, SOC and curtailment from the
   solution's standard `6n` block. Call `planner.observe_commit(k, import,
   export)`. Retain the window if inside the retention range.

Exactly one interval is committed per solve; committing more would blur the
thing being measured. The period is assumed contiguous.

Every fallback commits one interval of `run_sc_with_battery` from the current
SOC, and also calls `observe_commit`: a fallback is a statement about what the
battery did, not a gap, and per-day planner state must count it.

After the loop, the committed arrays pass through `_clip_nonneg` and
`_net_simultaneous_charge_discharge`, then `_reconstruct_flows`, all shared
with `perfect_foresight_lp`, so the `SiteTraces` invariants hold per interval.

---

## Horizon and infill

**The horizon must contain a complete charge-discharge cycle from any decision
instant.** The terminal SOC pin constrains the endpoint, not the path. A cycle
that fits inside the window can still arrive at the pinned state, so the pin is
harmless. It binds only when a profitable cycle would straddle the window
boundary, charging inside and discharging outside.

The production configs use **18 hours** (216 intervals). This comes from a
real-data sweep (see Validation), where every site turned at the same point
and longer horizons bought nothing. `RollingConfig.horizon_hours` itself
defaults to 48.

Pre-dispatch guarantees only about 15.1 hours measured from the decision
instant (the coarse run can be up to 25 minutes stale; the horizon sawtooths
between 15.1 and 39 hours). Beyond coverage the provider infills at
`infill_lag_hours` (default 24): `price(T) = price(T − lag)`, applied
recursively until it lands in the forecast or the observed past. Infill reads
only prices already observed or already forecast, so it is forecast, not
leakage; a real operator with 15 hours of data and a daily cycle to plan would
extrapolate too. The mechanics, the known-history rule and the `is_infill`
flag are in `forecast_providers_spec.md`.

At 18 hours infill fires for about 1% of window entries, and its cost at that
share is negligible (see Validation). It must exist and be correct, but it does
not carry weight at this horizon. At long horizons it becomes net negative:
more window bought with worse information.

---

## Causality

> The dispatch decision for interval k may use only data available at its
> decision instant.

This is the primary correctness property.

- **Price.** The provider owns causality and publication lags; there is no lag
  parameter in the component. The loop records how far the slice's `run_time`
  is ahead of the decision instant. Anything beyond one interval (the largest
  permitted mode allowance) sets `RollingDiagnostics.causality_ok = False`. It
  also records fine and coarse staleness.
- **Load and solar.** Metered energy for the interval ending at the decision
  instant is observed then, so providers may read labels at or before it
  (`usage_forecast_spec.md`, `solar_forecast_spec.md`).
- **Current-interval price.** The settled dispatch price for interval k is
  published about 9 seconds after k begins, so pricing k at the actual
  (`price_known`, `price_and_next`) is causal. The modes are defined in
  `forecast_providers_spec.md`.

Billing always uses actual prices in every mode.

---

## Price construction (wholesale planner)

For a `wholesale_exposed` plan the customer price is affine in RRP:

```
p_imp[t] = rrp[t]/1000 · import_loss_factor + offset_import[t]
p_exp[t] = rrp[t]/1000 · export_loss_factor + offset_export[t]
```

`prices.build_price_stack_offsets(index, network_tariff, retail_tariff,
state, interval_hours)` returns `PriceStackOffsets`: the two loss factors from
the network tariff, and the full-period offsets from running
`build_layer_rates` against a zero price series. Per window,
`offsets.window_price_vectors(rrp, start=k)` is a multiply and an add, so
tariff period lookups happen once per site rather than about 105,000 times.
Deriving the offsets from `build_layer_rates` guarantees the optimiser and
the bill use the same stack. `test_prices.py` checks that reconstruction
equals `build_price_vectors`.

`build_price_stack_offsets` is the chokepoint for two guards, because the
Energy Masters scripts call the component directly and never pass through the
orchestrator:

1. **The plan must be `wholesale_exposed`.** A TOU customer price does not
   depend on RRP.
2. **Price ordering across the whole market range**
   (`validate_price_ordering`). `perfect_foresight_lp` raises on
   `p_exp > p_imp` for a known series, but inside a rolling loop that would
   kill a run most of the way through. The spread is affine in RRP, so it is
   evaluated at the market price floor and cap (`model/core/market_prices.py`)
   and the run is refused if either end is negative. Loss factors normally
   have import above export, so the floor is the binding case.

No forecast source leaves the market range: the providers do not clip, and the
store validates ranges at ingest. So the in-loop hard error is unreachable.

The trap to watch is export loss factors copied from import ones. They make
the RRP term cancel. For example, `SA_sapn_resele_2025` has
`export_dlf = usage_dlf` and `export_tlf = import_tlf`. At a realistic
`export_dlf = 0.95` the same tariff pair's ordering breaks near −258 $/MWh,
which SA reaches routinely.

---

## Terminal SOC

Two rules, applied by position; there is no policy parameter.

- **A window reaching the end of the period** pins `soc[k_end]` to
  `soc_final_kwh`: the terminal SOC of a whole-period self-consumption run,
  which is also what `perfect_foresight_lp` is given. Rolling, perfect
  foresight and self-consumption therefore finish at the same state.
- **Any other window** pins `soc[k_end]` to the SOC reached by running
  `run_sc_with_battery` forward from the current SOC across the window. It
  uses the same window load and solar the LP sees, so the pin is never
  computed from information the plan does not have.

A planner can set `needs_terminal_soc = False` to skip the forward run
entirely. `VppWindowPlanner` does this, because its operator LP has no
terminal constraint. The forward run costs about 30 seconds per site-year at
18 hours.

Because the forward pin tracks the rolling SOC, a finite-horizon run with a
perfect forecast departs from perfect foresight through horizon truncation and
pin drift together. The two bind at the same boundary and are not separated.
At a horizon reaching the period end both vanish, which is what makes the
correctness gate exact.

---

## The window LP (wholesale planner)

The window subproblem is `perfect_foresight_lp` over n intervals: the same
6n layout, 2n equality rows, bounds and objective (see
`perfect_foresight_lp_spec.md`). `A_eq` depends only on n, η_c and η_d, so
for a fixed window length it is identical for every window of every site.

`_WindowSolver` builds a `highspy` model once per window length and, per
window, changes only:

- the import and export costs;
- the flow-balance right-hand side and the initial SOC row;
- the curtailment ceiling (when not pinned);
- the terminal SOC bounds.

The basis is kept, so consecutive windows start near optimal. The model is
rebuilt only when the window length changes (the final H intervals of the
period). Going through `scipy.optimize.linprog` rebuilds the model every call
and was measured at about 4.4× slower, which is why the component depends on
`highspy` directly.

**The reused model needs one explicit reset.** The terminal pin tightens the
bounds on a single SOC column. Without `_WindowSolver.restore_soc_bounds`, the
next window would inherit the previous window's target as a hard constraint.
That freezes the SOC trajectory while still producing a plausible dispatch.
`test_terminal_pin_does_not_leak_between_windows` guards it.

This invariance is a property of the wholesale subproblem only. The advanced
planner's block-tier linking rows move with the window. It keeps a persistent
model too, and mutates those coefficients (see its spec).

---

## Window planner protocol

```python
class WindowPlanner(Protocol):
    @property
    def requires_price_forecast(self) -> bool: ...
    def window_prices(self, rrp_mwh, k: int, n: int): ...          # (p_imp, p_exp), $/kWh
    def solve_window(self, k, n, solar, load, p_imp, p_exp,
                     soc_init, soc_final): ...                      # primal vector or None
    def observe_commit(self, k: int, import_kwh: float, export_kwh: float) -> None: ...
    def plan_diagnostics(self) -> dict | None: ...
# optional attribute: needs_terminal_soc (default True)
```

The loop owns:

- window geometry;
- all forecast providers, causality and staleness;
- the terminal pin;
- the self-consumption fallback;
- retention.

A planner owns only three things: how window prices are built, what the
window LP is, and any state carried between committed intervals. There is one
loop rather than copies because causality is the property a copy would drift
on.

Constraints on a planner:

- **The solution vector must start with the standard `6n` block** (charge,
  discharge, import, export, SOC, curtail). Extra variables go after it.
- **`observe_commit` receives every committed interval, fallbacks included.**
  Solved intervals pass the LP's own import and export. A fallback passes
  import `grid_to_home + grid_to_bat` and export `sol_to_grid`.
- **`requires_price_forecast = False`** makes the loop skip the price provider
  entirely: no query, no staleness accounting, no fallback over a forecast
  that could not have changed the answer.
- **`plan_diagnostics()`** is read once at the end and must not mutate state.
  Its keys become `RollingDiagnostics` fields.

The default `_WholesalePlanner` is built from `price_offsets` when no planner
is passed. It needs prices, has no state, and `observe_commit` does nothing.

---

## Failure handling

Anything that cannot be solved falls back to self-consumption for that
interval and is counted. There is no tiered recovery: the cases are rare
enough that graded fallbacks would cost more to build and test than they are
worth, and a silent partial relaxation is worse than a visible fallback.

| Cause | Counter |
|---|---|
| No price forecast (`get` returned `None`) | `sc_fallback_no_forecast` |
| No load forecast | `sc_fallback_no_load_forecast` |
| No solar forecast | `sc_fallback_no_solar_forecast` |
| Window not solved to optimality | `sc_fallback_infeasible` |

The first 20 fallback timestamps are kept. Any infeasible window triggers a
warning, and a count that is more than trivial means the run should be treated
as suspect.

When a site forecast is in use and a window is infeasible, the window is
re-solved with actual load and solar. If that succeeds, the failure is counted
in `infeasible_from_site_forecast`. The re-solve's solution is discarded,
because using it would be the leak the forecasts exist to model. With
curtailment pinned, under-forecast load or over-forecast solar leaves surplus
the LP cannot place, which is how site forecasts cause infeasibility.

Degraded forecast periods (October 2022) and store-coverage failures are
handled in the provider (`forecast_providers_spec.md`).

---

## Diagnostics and retention

`RollingDiagnostics` (frozen) is returned rather than logged, so a check
script and the orchestrator can both consume it. `summary()` renders it. Core
fields:

- window and fallback counts by cause, and `fallback_timestamps`;
- `infill_fraction_mean` and `infill_fraction_by_hour`;
- fine and coarse staleness (mean and max);
- `max_run_time_lead_minutes` and `causality_ok`;
- solve time (total, median, p95);
- terminal SOC against its target;
- load- and solar-forecast counters (used, warm-up windows, clipped positions)
  and `infeasible_from_site_forecast`.

Plan-specific fields default to `None` on other paths, so "not applicable"
reads differently from "zero":

- advanced: reward days lost, block-tier exhaustion;
- VPP: event and cap accounting.

Infill fraction is reported by hour of day because the pre-dispatch horizon
sawtooths daily. Without that split, the publication schedule would show up in
any time-of-day analysis as if it were battery behaviour.

**Retention.** Between `retain_plans_from` and `retain_plans_to` (whole
dates, both inclusive; a time of day is rejected), one `RetainedWindow` is kept
per decision instant, including fallbacks, which are marked `solved=False`.
Each holds:

- the forecast vector with `is_infill` and `is_fine`;
- the fine and coarse run times;
- the full plan (battery flows, SOC, planned import, export and curtailment);
- the committed SOC;
- the load and solar forecast vectors when those are in use.

Retaining every instant keeps the sequence contiguous, so a fallback does not
read as missing data. A per-interval post-mortem is the only way to
tell whether the optimiser missed a spike, saw it late, or acted on one that
never arrived. Three days at a 48-hour horizon is about 2.5 MB.

---

## Interface

```python
def rolling_optimisation(
    site_id: str,
    solar_kwh: pd.Series, load_kwh: pd.Series,          # actual, kWh, same index
    forecast_provider: ForecastProvider | None,
    price_offsets: PriceStackOffsets | None,
    battery_spec: BatterySpec,
    soc_init_kwh: float,
    soc_final_kwh: float,                                # whole-period SC terminal SOC
    config: RollingConfig,                               # horizon_hours read here
    export_limit_kwh_per_interval: float = np.inf,
    interval_hours: float = 5 / 60,
    curtail_cost_per_kwh_dollars: float | None = 0.0,    # as perfect_foresight_lp
    progress: Callable[[int, int], None] | None = None,
    progress_every: int = 2000,
    load_forecast_provider: LoadForecastProvider | None = None,
    solar_forecast_provider: SolarForecastProvider | None = None,
    commit_actual_load: bool = True,
    planner: WindowPlanner | None = None,
) -> tuple[SiteTraces, RollingDiagnostics]
```

Raises `ValueError` if:

- the traces are misaligned or contain NaN;
- `price_offsets` covers a different length;
- the SOC is out of bounds;
- neither a planner nor offsets is given;
- the planner needs prices and there is no provider.

The current-interval price mode belongs to the provider and is not read here.

Callers:

- The Energy Masters scripts (`run_cohort_rungs.py`,
  `check_rolling_optimisation.py`, `sweep_horizon.py` and others) call the
  component directly, with curtailment pinned.
- The orchestrator calls it through
  `orchestrate.run_rolling_perfect_foresight_lp_site`, which
  `viz_rolling_perfect_foresight_lp.py` also uses. The single-site diagnostic
  therefore runs the identical code path.

---

## Configuration

`RollingConfig` (frozen, `model/core/scenario.py`) is required for every
rolling coordination type and must be `None` otherwise:

| Field | Default | Meaning |
|---|---|---|
| `forecast_source` | required | `perfect`, `truncated_perfect`, `truncated_perfect_5min`, `predispatch` |
| `horizon_hours` | 48.0 | planning horizon; production configs use 18 |
| `infill_lag_hours` | 24.0 | infill lag |
| `current_interval_price` | `forecast` | `forecast`, `price_known`, `price_and_next`, `next_forecast` |
| `retain_plans_from`, `retain_plans_to` | `None` | retention dates, both or neither |
| `usage_forecast` | `None` | `UsageForecastConfig`; `{}` in YAML for the fitted defaults |
| `solar_forecast` | `None` | `SolarForecastConfig`; `{}` for the fitted defaults |
| `commit_actual_load` | `True` | solve position 0 on actual load and solar |
| `on_degraded_period` | `fail` | `fail`, `self_consumption`, `actuals`, `infill` (predispatch only) |
| `max_staleness_hours` | 6.0 | predispatch store coverage limit |

`commit_actual_load: false` is a diagnostic, not a production setting. The
export limit is enforced inside the LP against the load it is given, and
reconstruction applies no export clip, so forecast load at position 0 can
produce reconstructed export above the limit.

Production configuration (`full_run_rolling_amber.yaml`):

```yaml
coordination_type: rolling_perfect_foresight_lp
rolling:
  forecast_source: predispatch
  horizon_hours: 18.0
  current_interval_price: price_and_next
  on_degraded_period: infill
  max_staleness_hours: 800.0     # spans the October 2022 archive gap
  usage_forecast: {}
  solar_forecast: {}
```

---

## Orchestration and validation

`_run_rolling_perfect_foresight_lp` in `model/orchestrate.py`:

- builds `PriceStackOffsets` and the price provider once per scenario
  (`build_rolling_price_provider`);
- builds each site's load and solar providers when configured;
- runs the self-consumption baseline;
- runs the component with `soc_init = soc_min`, `soc_final` equal to the
  baseline's terminal SOC, and free curtailment;
- bills the result;
- writes `RollingDiagnostics` per site.

The price provider is built once per worker process by an initializer rather
than pickled into every site's arguments. The site forecast providers are
built with no `valid` mask, so every interval (infilled or not) enters the
profile history. A dispatched site needs a forecast at every interval, whereas
the fits used clean data only. VCR for a rolling scenario is assembled
downstream against a separate perfect-foresight run, not computed inline.

`validate_scenario` checks, before any ingest:

- **All rolling types:**
  - `rolling` is present;
  - `retain_plans_from/to` is used only when the scenario resolves to a single
    site;
  - `is_reportable(forecast_source, horizon_hours, period_hours)`: `perfect`
    at a horizon covering the whole period is the correctness gate, not a
    result;
  - for `predispatch`, a store partition exists for every month of the
    scenario year, except known archive gaps.
- **`rolling_perfect_foresight_lp`:** `build_price_stack_offsets` succeeds,
  which covers both the tariff-type guard and the ordering guard.
- The advanced and VPP checks are in their own specs.

---

## Verification

**The correctness gate.** `PerfectForecastProvider` at a horizon covering the
whole period must reproduce `perfect_foresight_lp`. Every window then pins the
same global terminal SOC, so each is the tail subproblem of the first, and by
Bellman optimality the committed sequence equals the full-period optimum.

`model/tests/test_rolling_optimisation.py`:

- **The gate itself**, the `SiteTraces` invariants on the gate run, and the
  period ending at the global self-consumption target.
- **SOC and pin:** SOC stays in bounds; the terminal pin does not leak between
  windows.
- **Horizon costs:** a 48-hour horizon costs almost nothing against full
  foresight, and a 30-minute coarse segment costs more than horizon truncation
  alone.
- **Fallbacks:** fallback on a missing forecast, at the start of the period,
  and on an infeasible window (with a warning).
- **Diagnostics:** infill and staleness are reported, with infill varying by
  hour; coarse staleness is reported separately from fine.
- **Retention:** it captures the decision context and every instant, keeps
  missing-forecast fallbacks as unsolved rows, uses inclusive date bounds,
  rejects a time of day and reversed bounds, and is off by default.
- **Plans:** planned grid flows match the plan's energy balance.
- **Input rejection:** misaligned traces, offsets for a different period, and
  an out-of-bounds SOC.

Providers, site forecasts and planners have their own tests
(`test_forecast_*`, `test_rolling_advanced.py`, `test_rolling_vpp_event.py`,
`test_orchestrate_rolling_advanced*.py`). `test_prices.py` covers
`PriceStackOffsets` and `validate_price_ordering`.

Dev tools:

- `energy_masters_pipeline/scripts/check_rolling_optimisation.py SITE [--forecast predispatch]`
  runs one Energy Masters site against actual, self-consumption and perfect
  foresight, with checks.
- `model/dev/viz_rolling_perfect_foresight_lp.py` is the single-site view on
  the orchestrator path.

---

## Known limitations

- **Every solver failure counts as "infeasible".** Any non-optimal HiGHS status
  falls back to self-consumption and increments `sc_fallback_infeasible`, so a
  numerical failure (for example from a stale warm basis) is
  indistinguishable from a genuinely infeasible window. Fix path: on a
  non-infeasible status, retry once from a cold basis. If it fails again,
  raise, since that is a defect rather than a data condition.
- **Curtailment is pinned or priced across the whole window.** When pinned
  (the Energy Masters default), forecast error can make windows infeasible.
  Curtailment beyond position 0 is never committed, so pinning it there is
  stricter than the physics requires. Fix path: allow priced curtailment from
  position 1 onward and keep position 0 pinned.
- **Horizon generalisation is unproven.** See Open questions.

---

## Validation

Measured during the build (August 2026), with perfect site data and before the
advanced and VPP planners existed. Results for the full configuration come from
the cohort and EDP runs, not from this document.

- **Horizon sweep: five Energy Masters sites.** September 2025 to March 2026,
  11.4–21.3 kWh batteries, $2,187 cohort headroom, pooled VCR, perfect site
  data, pre-dispatch-shaped actual prices:

  | Horizon | 3 h | 6 h | 9 h | 12 h | 15 h | **18 h** | 24 h | 48 h |
  |---|---|---|---|---|---|---|---|---|
  | Cohort VCR | 0.700 | 0.873 | 0.933 | 0.963 | 0.981 | **0.995** | 0.995 | 0.995 |
  | Infill share | 0 % | 0 % | 0 % | 0 % | 0 % | 1.1 % | 7.0 % | 43.2 % |

  - Every site turns at 18 h; per-site loss against its own best is at most
    0.0059.
  - 15 h misses the 0.02 tolerance by 0.014.
  - 48 h is slightly worse than 24 h (0.9945 against 0.9954) as infill rises
    to 43%, which implies about 0.000025 VCR for the 1% infill at 18 h.
- **Correctness gate:** matched `perfect_foresight_lp` to 7.5e-16 on the worst
  flow, with `import_kwh` and `curtail_kwh` bit-identical.
- **Measurement ladder: eight Energy Masters sites, pooled VCR.**

  | Dispatch | VCR |
  |---|---|
  | Rolling 18 h, actual prices | 1.000 |
  | Rolling 18 h, actual prices in pre-dispatch shape | 0.995 |
  | Rolling 18 h, pre-dispatch, `forecast` mode | 0.365 |
  | Actual (Amber) | 0.426 |

  - The rolling machinery (truncation, pin, coarse resolution, infill) costs
    about 0.005; the rest of the gap is the forecast.
  - The gap between any rolling result and Amber's actual dispatch is not
    attributable, because Amber has a different price forecast, a different
    site forecast and its own risk appetite.
- **Performance:** median solve 2.8–4.3 ms per 48-hour window with model
  reuse, about 14 minutes per site-year single-core. The LP is 69–73% of wall
  time.
- **EDP smoke test (one site, `truncated_perfect`):** 99.1% VCR against
  `perfect_foresight_lp`.
- **Horizon against battery capacity, 101 EDP sites (NSW 2024, November 2026).**
  Measured on real pre-dispatch forecasts rather than truncated perfect ones,
  at 6, 12, 18, 24 and 48 h and every capacity from 5 to 50 kWh
  (`docs/sensitivity_runs_spec.md`, section 2). 18 h gives the highest value
  capture ratio at every capacity, on the household bill and the net wholesale
  cost, so nothing beyond it helps: at 50 kWh the household ratio is 0.891 at
  18 h against 0.890 at 24 h, while infill rises from 1.0% to 6.8%. A shorter
  horizon costs little at 5 kWh (0.001) and a great deal at 50 kWh (0.15 at
  6 h). This supersedes the single-site `EM427` capacity sweep, which had
  suggested the required horizon rises with capacity; that result does not
  reproduce across 101 sites.

---

## Open questions

1. **Whether winter moves the answer.** The Energy Masters window is summer,
   while the overnight-charge into morning-peak cycle that breaks short
   horizons is mainly a winter pattern. The EDP adapter in `sweep_horizon.py`
   needs no forecast store and can test this directly.
2. **Site-forecast infill masking in dispatch.** The orchestrator passes no
   `valid` mask, unlike the fits. Should this differ per input, and should it
   be a `RollingConfig` field?

---

## Conventions

- The 30-minute settlement correction is for reconciliation and billing only,
  never the LP. The optimiser always works against 5-minute prices.
- `datetime_aest` is `Etc/GMT-10`, interval-end, no DST; local-clock effects
  go through `core/localtime.py` with an explicit state.
- Costs positive, credits negative; `p_exp` is value earned per exported kWh.
