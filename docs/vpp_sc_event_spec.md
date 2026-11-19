# vpp_sc_event — Specification

## Purpose and scope

`model/components/vpp_sc_event.py` models a home battery enrolled in a
Virtual Power Plant (VPP), with perfect foresight of prices. It produces a
full-year dispatch and a split of every import and export into **natural**
flow (what the household would have done anyway) and **forced** flow (what
the operator added). Forced flow earns the household the VPP reward and counts
against the annual cap.

The two operating regimes are kept separate by construction:

- **Between events** the household strategy runs, starting from whatever SOC
  the previous event left. It is self-consumption by default
  (`gap_strategy='self_consumption'`, the `vpp_sc_event` coordination type),
  or a TOU-optimising LP (`gap_strategy='tou_advanced'`, the
  `vpp_tou_event` coordination type, see `vpp_tou_event_spec.md`).
- **Within each event window** a short LP chooses the operator-optimal
  dispatch against operator prices. It sees only that window and has no
  terminal SOC constraint. The SOC it leaves is the entry SOC for the next
  gap.

The output is a `SiteTraces` of the usual shape plus a `VppTraces` carrying
`n_imp_kwh`, `f_imp_kwh`, `n_exp_kwh` and `f_exp_kwh`. Billing uses
`VppTraces` to pay the reward. `rolling_vpp_event` (see
`rolling_vpp_event_spec.md`) is the imperfect-foresight counterpart and
shares the minimum-export-price resolution.

---

## Event classification

`model/components/vpp_event_classifier.classify_vpp_events` marks the
operator-controlled intervals. It scans prices backwards from the end of the
period. At each interval priced above `vpp_threshold_dollars_per_mwh` that is
not already marked, it marks a window of `event_window_hours` (rounded up to
whole intervals) ending at and including that interval, clamped at the start
of the period. Overlapping windows
merge into one contiguous event. A spike inside a window opened by a later
spike does not open a further window, so earlier intervals are not claimed
twice against the cap. The rule is permissive: it marks every spike an
operator might dispatch into. The dispatcher only consumes the mask, so the
rule can be replaced on its own.

The mask is computed from the scenario's price series and is the same for
every site.

---

## Architecture

Events are contiguous `True` blocks of the mask (`_find_event_windows`),
processed in order. Starting with `soc_entry = soc_init_kwh` (the reserve
floor) and `remaining_cap = vpp_kwh_limit`, each cycle does the following:

1. **Self-consumption over gap + event.** `run_sc_with_battery` (from
   `self_consumption.py`) runs from `soc_entry` over
   `[previous event end, this event end)`.
2. **Gap dispatch.**
   - `self_consumption`: the gap part of that run is the gap dispatch.
   - `tou_advanced`: `_run_tou_gap` produces the gap dispatch instead (see
     `vpp_tou_event_spec.md`).
3. **Bounds.** Natural-flow bounds for the event come from the event part
   of the self-consumption run in step 1 (see Natural-flow bounds).
4. **Event LP.** One solve over the event window, entering at the SOC the gap
   left.
5. **Cap accounting.** `remaining_cap` is reduced by `Σ f_exp` (for
   `export_only`) or `Σ (f_imp + f_exp)` (for `import_and_export`), floored at
   zero.
6. **Advance.** `soc_entry` becomes the LP's final SOC.

After the last event, the tail `[last event end, T)` is filled by the gap
strategy.

An event reached with the cap already exhausted is still solved by the LP.
The cap row then forces `f_imp = f_exp = 0`, but the LP can still re-time
natural flows within the window (see Known limitations).

Under `tou_advanced`, the bounds in step 3 still come from the
self-consumption run in step 1, which starts at the previous event's exit
SOC and follows self-consumption through the gap. The event LP, however,
enters at the TOU gap's exit SOC. The natural/forced split is therefore
always measured against a self-consumption counterfactual, whatever the gap
strategy.

---

## Natural-flow bounds

Import or export that self-consumption would have produced is natural and
earns no reward. Only flow beyond it is forced. The bounds come from the
self-consumption run and are fixed for the solve, so the split does not depend
on the LP's own decisions (`_compute_bounds`). With `sc_soc_prev[t]` the
self-consumption SOC at the start of interval t:

**Natural import: one aggregate budget over the window.**

```
sc_n_imp_total = Σ_t max(0, load[t] − solar[t] − min((sc_soc_prev[t] − soc_min)·η_d, power_kwh))
```

A per-interval ceiling was rejected because it would depend on the LP's own
SOC and create a feedback loop. Forced charging early in the window raises
SOC, which tightens the later ceilings and forces still more import to be
classed as forced. The aggregate budget lets the LP manage the battery
across the window, provided total natural import does not exceed the
self-consumption counterfactual.

**Natural export: a per-interval ceiling.**

```
n_exp_hi[t] = max(0, solar[t] − load[t] − min((capacity − sc_soc_prev[t])/η_c, power_kwh))
```

The ceiling is per interval because an aggregate would let the LP hold back
export early (by charging or curtailing) and claim a later, higher-priced
export as natural. That would disguise forced export as natural and
under-pay the reward.

`n_exp_hi` uses raw solar, not `solar − curtail`, because curtailment is
decided inside the solve. It is therefore a looser, still valid, ceiling. The
curtailment cost discourages curtailing just to create headroom, and the
per-interval form still prevents shifting export between intervals.

---

## Event LP (`_solve_event_lp`)

### Variables

8·T_e continuous variables, where T_e is the event length (120 for a 10-hour
window):

| Variable | Offset | Bounds |
|---|---|---|
| `charge[t]` | `x[0:Te]` | [0, `power_kwh`] |
| `discharge[t]` | `x[Te:2Te]` | [0, `power_kwh`] |
| `n_imp[t]` | `x[2Te:3Te]` | [0, ∞) (aggregate budget row) |
| `f_imp[t]` | `x[3Te:4Te]` | [0, ∞) |
| `n_exp[t]` | `x[4Te:5Te]` | [0, `n_exp_hi[t]`] |
| `f_exp[t]` | `x[5Te:6Te]` | [0, ∞), or [0, 0] below the minimum export price |
| `soc[t]` | `x[6Te:7Te]` | [`soc_min`, `capacity`], no terminal pin |
| `curtail[t]` | `x[7Te:8Te]` | [0, max(0, `solar[t]`)], or [0, 0] when curtailment is disabled |

`power_kwh = power_kw × interval_hours`, `soc_min = backup_reserve_pct / 100 ×
capacity`.

### Objective

```
minimise  Σ_t [ n_imp[t]·op_p_imp[t] + f_imp[t]·(op_p_imp[t] + vpp_imp_reward)
              − n_exp[t]·op_p_exp[t] − f_exp[t]·(op_p_exp[t] − vpp_exp_reward) ]
        + c_deg · Σ_t (charge[t] + discharge[t])
        + p_curtail · Σ_t curtail[t]        (only when p_curtail is not None)
```

Every interval in the window is priced from the **operator's** point of view:
the operator buys and sells at `op_p_imp`/`op_p_exp` (wholesale plus network,
loss-adjusted; see Configuration) and pays the household the reward on forced
flow. There is no household term. The reward makes forced flow slightly
dearer than natural flow, so the LP uses natural flow up to its bounds first.
This is why `VppConfig` floors a zero import reward to 0.01 c/kWh: with an
exactly zero reward the LP could label import either way. `c_deg` and
`p_curtail` shape dispatch only and never appear in the bill.

### Constraints

Equality, 2·T_e rows:

```
(n_imp + f_imp) − (n_exp + f_exp) − charge + discharge − curtail = load − solar        (flow balance)
soc[t] − soc[t−1] − η_c·charge[t] + (1/η_d)·discharge[t] = 0,  soc[−1] = soc_entry        (SOC continuity)
```

Inequality:

| Row | Count | Present when |
|---|---|---|
| `Σ n_imp ≤ sc_n_imp_total` | 1 | always |
| `n_exp[t] + f_exp[t] ≤ export_limit_kwh` | T_e | export limit finite |
| `Σ f_exp ≤ remaining_cap` (`export_only`), or `Σ (f_imp + f_exp) ≤ remaining_cap` (`import_and_export`) | 1 | always |
| `charge[t] + curtail[t] ≤ max(0, solar[t] − load[t])` | one per masked interval | no-grid-charge mask set |

Solved with scipy `linprog(method='highs')`, one solve per event.

`op_p_imp ≥ op_p_exp` is required at every interval and checked up front
(`ValueError`). Pricing all import at no less than the export value makes
simultaneous import and export a loss.

### Curtailment

Controlled by `p_curtail` ($/kWh), set from `VppConfig.curtail_c_per_kwh`:

| `p_curtail` | Behaviour |
|---|---|
| `None` (default) | Curtailment pinned to zero. The LP can be infeasible when the export limit binds against a full battery (the SC fallback then applies). |
| `0` | Allowed at no cost: curtails rather than exporting at a negative price. |
| `> 0` | Curtails only when `op_p_exp < −p_curtail`, or when it is needed for feasibility. |

A negative value is rejected. Curtailment exists only inside events; gap
`curtail_kwh` is zero under self-consumption.

### Minimum export price

`f_exp[t]` is pinned to zero wherever `op_p_exp[t] <
min_export_price_per_kwh`. This keeps the annual cap for high-value
intervals. Without it, on plans with little or no per-kWh reward (for
example EA Battery Ease), the LP would spend the cap on the first events of
the year at any positive price. Natural flows and `f_imp` are unaffected, and
`f_exp = 0` is always feasible.

The value is resolved once per scenario by `orchestrate.resolve_min_export_price`,
which the rolling VPP path also uses. The same YAML therefore gives the same
threshold on both, and the value capture ratio between them isolates
foresight alone. Modes (`VppConfig.min_export_price_mode`):

- `fixed` (default): `min_export_price_dollars_per_mwh` is used as given. If it
  is absent, `validate_scenario` rejects the scenario.
- `based_on_this_year`: `max(derived, configured or 0)`, where the configured
  value is a floor.
- `based_on_last_year`: not implemented; rejected.

The derivation (`_compute_min_export_price`) works as follows:

- Over the intervals marked by `classify_vpp_events`, each interval
  contributes `max(0, power_kw − avg_load_kw[t]) × interval_hours`, where
  `avg_load_kw` is the cohort mean load with each site's load capped at
  `power_kw`.
- Intervals are sorted by price, highest first. The price at which the
  cumulative contribution first reaches `kwh_limit` is the threshold.
- The threshold is 0 if the cap can never be exhausted.

The derivation uses the modelled year's own prices, which is look-ahead. That
is acceptable here because the whole component is a perfect-foresight benchmark.

The threshold is stated in $/MWh of wholesale price and divided by 1000
before being passed to the component. It is then compared with the operator
export price, which also includes the export loss factor and the network
export rate.

### No-grid-charge windows

Some contracts forbid the operator from charging the battery **from the
grid** during a stated daily window, typically the evening peak. Charging
from the household's solar and forced discharge remain allowed. On masked
event intervals the LP carries

```
charge[t] + curtail[t] ≤ max(0, solar[t] − load[t])
```

A bare bound `charge ≤ surplus` is not enough once curtailment is live. The
LP could curtail the surplus S and import S to charge (`charge = S ≤ S`).
The flow decomposition then attributes all of it to `grid_to_bat` because
curtailment drove `sol_to_bat` to zero. That is chosen whenever the operator
import price is below `−p_curtail`, which is the negative-price early
evening before a spike. Coupling `charge` and `curtail` makes that `2S > S`,
so no grid energy can reach the battery.

The restriction is on the charge side rather than on `f_imp`. `n_imp` has
only an aggregate budget, so pinning `f_imp` would leave the LP free to
spend natural-import budget in the window to charge. The row leaves the
natural/forced accounting untouched.

The row keeps the self-consumption assignment feasible, because
self-consumption charges only from surplus and never curtails. It still
allows full curtailment of an unexportable surplus with `charge = 0`.

The mask is built once per scenario by OR-ing `prices.charge_window_mask`
over `VppConfig.no_grid_charge_windows`, using the site state's local time at
interval start. The row applies only inside events.

### Feasibility and fallback

Under `self_consumption` gaps, the self-consumption dispatch is a feasible
point of the event LP. The event LP enters at the same SOC the bounds were
built from, and the self-consumption flows satisfy:

- the flow balance and SOC continuity;
- the natural-import budget (with equality);
- the natural-export ceilings;
- the cap row (with zero forced flow);
- the no-grid-charge rows.

When curtailment is pinned, a binding export limit can still make it
infeasible. Under `tou_advanced` the entry SOC differs from the one the
bounds assume, so this guarantee does not hold.

If HiGHS does not return an optimal solution, the component warns
(`EOV_SKIP:site:win_start:remaining_cap:load−solar:budget:status`). It then
uses the self-consumption flows and SOC for that event and continues. The
orchestrator re-emits these warnings, but suppresses them when
`remaining_cap < 20` kWh.

---

## Output assembly

Event flows are decomposed from `charge`, `discharge` and `curtail` with the
usual priorities (solar to load first, solar charges the battery before the
grid does, battery discharge meets load before exporting), on `solar_eff =
solar − curtail`. `import_kwh` and `export_kwh` are taken directly from the LP
(`n_imp + f_imp`, `n_exp + f_exp`), and `grid_to_home = import_kwh −
grid_to_bat`, so the `VppTraces` identities `n_imp + f_imp = import_kwh` and
`n_exp + f_exp = export_kwh` hold exactly. Because billing uses these LP
totals, a simultaneous charge/discharge tie does not affect the bill.

Gap segments:

- `_append_sc_segment` (self-consumption): `bat_to_grid = 0`,
  `export_kwh = sol_to_grid`.
- `_append_tou_segment` (TOU LP): keeps the LP's `bat_to_grid`, with
  `export_kwh = sol_to_grid + bat_to_grid`. The TOU LP does export from the
  battery.

In gaps `f_imp = f_exp = 0`, `n_imp = import_kwh` and `n_exp = export_kwh`.
Segments are concatenated in time order.

Post-solve checks in the component:

- clipping of solver noise below zero, with a warning above 1e-6;
- warnings for simultaneous charge/discharge or import/export within an event;
- `_check_no_grid_charge`, which warns with interval context if
  `grid_to_bat > 1e-6` on any masked event interval (a breach is a bug).

---

## Interface

```python
def vpp_sc_event(
    site_id: str,
    solar_kwh: pd.Series, load_kwh: pd.Series,     # kWh, full year
    op_price_imp: pd.Series, op_price_exp: pd.Series,   # $/kWh, operator
    vpp_event: np.ndarray,                          # bool, length T
    battery_spec: BatterySpec,
    soc_init_kwh: float,
    vpp_exp_reward: float, vpp_imp_reward: float,   # $/kWh, >= 0
    vpp_kwh_limit: float,
    vpp_limit_applies_to: str,                      # 'export_only' | 'import_and_export'
    export_limit_kwh_per_interval: float = np.inf,
    interval_hours: float = 5 / 60,
    p_curtail: float | None = None,                 # $/kWh
    gap_strategy: str = 'self_consumption',         # | 'tou_advanced'
    retail_tariff: RetailTariff | None = None,      # tou_advanced only
    prices_mwh: pd.Series | None = None,            # tou_advanced only
    price_imp: pd.Series | None = None,             # tou_advanced only (household)
    price_exp: pd.Series | None = None,             # tou_advanced only (household)
    min_export_price_per_kwh: float = 0.0,
    no_grid_charge_mask: np.ndarray | None = None,
    apply_export_order_nudge: bool = False,         # tou_advanced only
    state: str | None = None,                       # tou_advanced only
) -> tuple[SiteTraces, VppTraces]
```

Raises `ValueError` in these cases:

- the indexes are misaligned or the mask length is wrong;
- an input contains NaN, or load is negative;
- `op_p_exp > op_p_imp` at any interval;
- a reward is negative, or `vpp_limit_applies_to` is unknown;
- `soc_init_kwh` is out of bounds, or `p_curtail` is negative;
- `gap_strategy` is unknown, or the `tou_advanced` inputs are missing.

---

## Configuration and orchestration

The coordination types are `vpp_sc_event` and `vpp_tou_event`. Both route to
`_run_vpp_sc_event` in `model/orchestrate.py`, with `gap_strategy` set
accordingly. The scenario needs:

- `retail_tariff_id`: a household plan with a `vpp:` block;
- `op_retail_tariff_id`: the operator tariff, a `wholesale_exposed` plan with
  zero retail adders and zero daily charges.

`op_p_imp`/`op_p_exp` are built from the operator tariff with
`build_price_vectors`. Under `vpp_tou_event` the household vectors come from
`build_lp_price_vectors_advanced`.

Per scenario the orchestrator:

1. converts the rewards and `curtail_c_per_kwh` from c/kWh to $/kWh;
2. resolves the minimum export price;
3. builds the no-grid-charge mask;
4. classifies events.

Per site it runs `self_consumption` as the VCR baseline, then
`vpp_sc_event` from `soc_min` with the site's export limit, and bills with
`compute_bill(..., vpp_traces=...)`.

`vpp:` block fields (`VppConfig` in `model/core/tariff.py`):

| Key | Meaning |
|---|---|
| `exp_reward_c_per_kwh` | Household reward per kWh of forced export, ≥ 0 |
| `imp_reward_c_per_kwh` | Household reward per kWh of forced import; 0 is floored to 0.01 on load |
| `kwh_limit` | Annual cap on forced energy, > 0 |
| `limit_applies_to` | `export_only` or `import_and_export` |
| `event_window_hours` | Operator control window before each spike |
| `vpp_threshold_dollars_per_mwh` | Spike threshold for event classification |
| `curtail_c_per_kwh` | Event curtailment cost; absent disables curtailment |
| `min_export_price_dollars_per_mwh` | Minimum export price (floor under `based_on_this_year`) |
| `min_export_price_mode` | `fixed` (default), `based_on_this_year` |
| `no_grid_charge_windows` | `[17, 21]` or `[[7, 9], [17, 21]]`; local whole hours, `[start, end)`, every day |
| `default_mode` | Parsed and validated, but used by no current component |

---

## Verification

Tests:

- **`test_vpp_sc_event.py`**
  - Window finding and bound construction.
  - With no events the result equals self-consumption and forced flow is zero.
  - Force-discharge; force-charge followed by discharge; forced flow is zero
    outside events.
  - The natural-import and natural-export bounds hold.
  - The cap binds under both `limit_applies_to` settings, is carried across
    events, and gives zero forced flow once exhausted.
  - SOC continuity across events; the post-event gap starts from the event exit SOC.
  - `SiteTraces` invariants and non-negative `VppTraces`; the joint export limit.
  - Input validation.
- **`test_vpp_sc_event_curtail.py`**
  - Disabled curtailment matches the default call.
  - Zero-cost curtailment resolves a binding export limit.
  - Behaviour of a positive cost; negative cost rejected.
  - Invariants hold with curtailment.
  - The aggregate import budget and per-interval export ceiling hold.
  - No SC fallback on standard inputs.
- **`test_vpp_sc_event_min_export.py`**
  - The derivation: a known answer, a cap that is never exhausted, and the
    effect of load.
  - `f_exp` restricted below the threshold; a zero threshold is unrestricted.
  - YAML parsing.
- **`test_vpp_sc_event_no_grid_charge.py`**
  - `None` and an all-false mask are no-ops.
  - `grid_to_bat = 0` on masked intervals, while solar charging and forced
    discharge still occur and `f_imp = 0` at the optimum.
  - Curtail laundering is blocked.
  - Feasible with an event entirely inside a window, including with a
    binding export limit and curtailment.
  - Applies under self-consumption gaps.
  - Window YAML parsing and validation.
- **`test_vpp_event_classifier.py`**
  - Window placement and clamping; merging of overlapping windows.
  - Input validation.

Dev tools:

- `model/dev/check_vpp_sc_event.py SITE_ID YEAR` runs a real site. It reports
  the event summary, the bill against self-consumption, and the
  avoidable-import metric: grid import between events that the battery could
  have served, which should be near zero.
- `model/dev/viz_vpp_sc_event.py` is the interactive view.
- `model/dev/check_eov_event.py SITE_ID YEAR EVENT_DATE` examines one event:
  the split, ending SOC, household and retailer economics, and operator P&L.

---

## Known limitations

- **Greedy cap allocation.** The cap is spent event by event in time order,
  so a later, more valuable event can find it exhausted. The minimum export
  price reduces this but does not solve it. The limitation is immaterial where
  the cap does not bind. Fix path: allocate the cap across events, for
  example with one LP over all event windows linked by the cap row.
- **Natural import is aggregate.** Within a window the LP may re-time natural
  import, including charging the battery from the grid at the cheapest
  intervals, provided total natural import stays within the counterfactual.
  That flow earns no reward and uses no cap, and it continues after the cap is
  exhausted.
- **`grid_to_home` can be negative** at intervals where the LP ties
  simultaneous charge and discharge, because `grid_to_home = import_kwh −
  grid_to_bat` and the tie inflates `grid_to_bat`. Billing is unaffected (it
  uses the LP totals); flow and throughput analysis of `grid_to_home` is not.
  Fix path: net simultaneous charge/discharge before decomposition, as
  `perfect_foresight_lp` does.
- **Feasibility is not guaranteed under `tou_advanced`** or with pinned
  curtailment and a binding export limit (see Feasibility and fallback).
- **The minimum export price is stated as a wholesale price but compared
  with the operator export price** (which includes the loss factor and the
  network export rate).
