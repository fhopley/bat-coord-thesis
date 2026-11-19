# perfect_foresight_lp — Specification

## Purpose and scope

`model/components/perfect_foresight_lp.py` computes the cost-optimal battery
dispatch for one site over a whole period, with perfect knowledge of the
prices the site faces. It is the benchmark every other coordination strategy
is measured against, and the denominator of the Value Capture Ratio (VCR).
It is the `perfect_foresight` coordination type.

It returns a `SiteTraces` of the same shape as `self_consumption`, so billing
and everything downstream treat both identically.

Other components reuse its internals: `perfect_foresight_lp_advanced` builds
on `_build_lp`, and `rolling_optimisation` reuses `_clip_nonneg`,
`_net_simultaneous_charge_discharge` and `_reconstruct_flows`.

---

## Formulation

### Objective

```
minimise   Σ_t [ imp[t]·p_imp[t] − exp[t]·p_exp[t] ]
         + c_deg · Σ_t [ charge[t] + discharge[t] ]
         + c_curtail · Σ_t curtail[t]
```

- `p_imp[t]`, `p_exp[t]`: per-interval import cost and export value, $/kWh.
  `p_exp` may be negative (an export penalty).
- `c_deg`: `BatterySpec.c_deg_per_kwh`, $/kWh of throughput, charged on both
  charge and discharge.
- `c_curtail`: `curtail_cost_per_kwh_dollars` (see Curtailment modes); zero
  by default.

**Daily charges are omitted.** Supply, network and subscription charges do
not depend on dispatch, so they shift every candidate solution by the same
constant. The full `household_total` is produced afterwards by passing the
dispatch through `compute_bill`.

**The price signal is the plan the household actually faces.** The
orchestrator builds `p_imp`/`p_exp` with `prices.build_price_vectors`, the
same rate construction billing uses (`build_layer_rates`), so the LP optimises
against exactly the prices billing charges:

- `wholesale_exposed`: full stack, wholesale + network + retail adders, with
  loss factors on the wholesale and network layers.
- `tou`: the retail layer only.

`validate_scenario` rejects a `mixed` plan for this type. `mixed` plans are
dispatched by `perfect_foresight_advanced` (`perfect_foresight_advanced_spec.md`).

The details are in `model_spec.md` (Tariffs).

**Degradation cost.** `c_deg` makes a round trip worth doing only when the
price spread exceeds `2 × c_deg` plus efficiency losses. It also stops the LP
cycling energy through the battery purely to increase import at negative
prices. It is a modelling cost only and does not appear in the bill. The
`BatterySpec` default is 0.005 $/kWh.

### Decision variables

Per interval t, all continuous, in one flat vector of length 6T:

| Variable | Offset | Bounds | Meaning |
|---|---|---|---|
| `charge[t]` | `x[0:T]` | [0, `power_kw × interval_hours`] | AC energy into the battery |
| `discharge[t]` | `x[T:2T]` | [0, `power_kw × interval_hours`] | AC energy out of the battery |
| `imp[t]` | `x[2T:3T]` | [0, ∞) | Grid import |
| `exp[t]` | `x[3T:4T]` | [0, `export_limit_kwh_per_interval`] | Grid export |
| `soc[t]` | `x[4T:5T]` | [`soc_min`, `capacity_kwh`] | DC state of charge at the end of interval t |
| `curtail[t]` | `x[5T:6T]` | [0, max(0, `solar[t]`)], or [0, 0] when pinned | Curtailed solar |

`soc_min = backup_reserve_pct / 100 × capacity_kwh`. `solar[t]` and `load[t]`
are constants.

Curtailment is a variable because without it the flow balance forces all solar
somewhere. When solar exceeds load plus charge headroom plus the export
limit (a full battery with a binding export limit), the LP would be
infeasible. It also lets the LP refuse to export at a negative price.

### Constraints

2T equality rows, in a sparse CSC matrix of 2T × 6T with about 9T non-zeros
(about 950,000 for a full year).

Flow balance, T rows:

```
imp[t] − exp[t] − charge[t] + discharge[t] − curtail[t] = load[t] − solar[t]
```

SOC continuity, T rows, with `η_c = charge_efficiency` and
`η_d = discharge_efficiency`:

```
soc[t] − soc[t−1] − η_c·charge[t] + (1/η_d)·discharge[t] = 0      t ≥ 1
soc[0]            − η_c·charge[0] + (1/η_d)·discharge[0] = soc_init
```

The terminal SOC is imposed by setting both bounds of `soc[T−1]` to
`soc_final`, which is numerically cleaner than an extra equality row.

### Simultaneous flows

Binary variables would make this a MILP, so simultaneous import/export and
charge/discharge are handled without them:

- **Price ordering (hard error).** `p_exp[t] ≤ p_imp[t]` is checked before the
  solve and a `ValueError` is raised on any violation. If export paid more
  than import cost, the LP would import and export at the same time for
  profit.
- **Post-solve warning.** A warning is issued for any interval with both
  `charge` and `discharge`, or both `imp` and `exp`, above 1e-6 kWh.
- **Netting.** When `c_deg` is too low, the LP can find it worthwhile to
  charge and discharge in the same interval to dump energy as losses, making
  room for more import at a negative price. `import_kwh`/`export_kwh` are
  derived from `charge`/`discharge` in the flow decomposition, so this would
  show up as billable simultaneous flows. After the warning (which reports the
  solver's raw output), both are reduced by `min(charge, discharge)` before the
  decomposition. Net grid flow is unchanged. `soc` is left as solved, so for
  netted intervals it is slightly inconsistent with the netted flows. Raising
  `c_deg` avoids the situation altogether.

---

## Curtailment modes

`curtail_cost_per_kwh_dollars: float | None`, in **dollars** per kWh. The unit
is in the name because a value in cents (for example 5) would silently create a
large phantom cost.

| Value | Bound on `curtail` | Objective |
|---|---|---|
| `0.0` (default) | `max(0, solar)` | zero; the implicit cost is the lost export credit |
| `None` | 0 (pinned) | — |
| `> 0` | `max(0, solar)` | `curtail_cost_per_kwh_dollars` per kWh |

Pinning gives the strictest comparison: only dispatch that is physically
deliverable without curtailment. It can make the LP infeasible if the solar
surplus in some interval cannot be absorbed by load, battery headroom and the
export limit, and the solver then raises `RuntimeError`. A positive cost avoids
that risk while still discouraging curtailment except where it is genuinely
optimal.

Where each caller sits:

- The `perfect_foresight` coordination type (orchestrator) and
  `perfect_foresight_lp_advanced` (which calls `_build_lp` directly) use the
  default `0.0`.
- The Energy Masters scripts use pinned (`None`), through
  `viz_energy_masters.DEFAULT_CURTAIL_COST_PER_KWH_DOLLARS`, because Amber has
  no inverter access and cannot curtail; a benchmark allowed to curtail is not
  like-for-like for that cohort. Each run prints the mode it used.
- The rolling planners (`rolling_optimisation`, `rolling_advanced`) take the
  same parameter with the same three modes. Under the rolling VPP types it is
  set from `VppConfig.curtail_c_per_kwh`, converted to $/kWh.

---

## Interface

```python
def perfect_foresight_lp(
    site_id: str,
    solar_kwh: pd.Series,          # kWh per interval; may be negative (inverter draw)
    load_kwh: pd.Series,           # kWh per interval; must be >= 0
    price_imp: pd.Series,          # $/kWh
    price_exp: pd.Series,          # $/kWh; may be negative
    battery_spec: BatterySpec,
    soc_init_kwh: float,           # within [soc_min, capacity_kwh]
    soc_final_kwh: float,          # within [soc_min, capacity_kwh]
    export_limit_kwh_per_interval: float = np.inf,
    interval_hours: float = 5 / 60,
    curtail_cost_per_kwh_dollars: float | None = 0.0,
) -> SiteTraces
```

Raises `ValueError` if the four series do not share an index, contain NaN,
load is negative, price ordering is violated, or `soc_init_kwh`/`soc_final_kwh`
lie outside the battery bounds. Raises `RuntimeError` if the solver does not
return an optimal solution.

`lp_backend()` returns the solver in use.

### Solver

scipy `linprog(method='highs')`. A Gurobi backend (`_solve_gurobi`) receiving
the same sparse matrix is present but disabled: `_GUROBI_AVAILABLE` is
hard-set to `False` and the `gurobipy` import is commented out.

Small negative solver values in `charge`, `discharge`, `imp`, `exp` and
`curtail` are clipped to zero, with a warning if any exceeds 1e-6 in
magnitude.

### Flow decomposition

The solved `charge`, `discharge` and `curtail` are split into the `SiteTraces`
flows using two priority rules: solar fills battery charge before the grid
does, and battery discharge meets load before exporting.

```
solar_eff     = solar − curtail
sol_to_home   = min(solar_eff, load)          (negative when the inverter draws power)
surplus_solar = solar_eff − sol_to_home
residual_load = load − sol_to_home

sol_to_bat    = min(surplus_solar, charge)
grid_to_bat   = charge − sol_to_bat
sol_to_grid   = surplus_solar − sol_to_bat

bat_to_home   = min(residual_load, discharge)
bat_to_grid   = discharge − bat_to_home
grid_to_home  = residual_load − bat_to_home

import_kwh    = grid_to_home + grid_to_bat
export_kwh    = sol_to_grid + bat_to_grid
curtail_kwh   = curtail
```

The `SiteTraces` invariants, including
`solar = sol_to_home + sol_to_bat + sol_to_grid + curtail_kwh`, hold
algebraically from the flow balance, so no rounding correction is needed.
`import_kwh`/`export_kwh` come from this decomposition, not from the LP's raw
`imp`/`exp`.

### How the orchestrator calls it

For each site, `_run_perfect_foresight` in `model/orchestrate.py`:

1. Runs `self_consumption` and bills it; this is the VCR baseline.
2. Sets `soc_init_kwh = soc_min` and `soc_final_kwh` to the self-consumption
   end-of-period SOC (clamped to the battery bounds), so both strategies
   start and end with the same stored energy and the bill comparison is
   fair.
3. Uses the site's export limit, `export_limit_kw_per_phase ×
   max(ac_load_net_phases, pv_site_net_phases)` from
   `_resolve_per_site_export_limits`, converted to kWh per interval. It is
   unlimited when `export_limit_kw_per_phase` is `None`.
4. Calls `perfect_foresight_lp` and bills the result with `compute_bill`.

Sites run in parallel with `ProcessPoolExecutor` when `workers > 1`.

---

## Verification

`model/tests/test_perfect_foresight_lp.py` checks:

- **Dispatch behaviour:** arbitrage with and without efficiency losses and
  over three periods; no cycling on a flat price or with a high degradation
  cost; a degradation threshold that prevents marginal cycling; solar charging
  the battery before the grid does; solar split between home, battery and
  grid; battery export.
- **Bounds:** SOC limits, power limits, a binding export limit, and the
  terminal SOC.
- **Curtailment:** makes an export-limited case feasible; avoids an export
  penalty. The default equals omitting the argument. Pinned mode gives zero
  curtailment when feasible. A priced mode curtails between free and zero,
  and its cost appears in the objective.
- **Result:** the `SiteTraces` invariants hold, and the LP bill is no higher
  than the no-battery bill.
- **Input validation:** price ordering, NaN solar or price, negative load,
  out-of-bounds SOC and misaligned indexes all raise.

`model/dev/check_perfect_foresight_lp.py SITE_ID YEAR` runs a real site and
checks the flow invariants, SOC and power bounds, the terminal SOC and the
export limit. It reports simultaneous-flow counts for information.
`model/dev/viz_perfect_foresight_lp.py` is the interactive daily view.

---

## Known limitations

- **SOC after netting.** For intervals where simultaneous charge/discharge is
  netted, `battery_soc_kwh` keeps the LP's value, which includes the netted
  round-trip losses. The error is bounded by the netted amount. Fix path:
  recompute SOC from the netted flows.
- **No objective-to-bill reconciliation.** The LP objective without the
  degradation and curtailment terms should equal the dispatch-dependent part
  of `compute_bill`. No test or check script asserts this.

---

## Validation

- **Netting (S0093, SA, 2022, 20 kWh):** 116 intervals with simultaneous
  charge/discharge before netting, 0 after (measured when netting was added).
- **Billing refactor:** after rate construction moved into `prices.py`,
  `bills.csv` matched the previous billing to 4 decimal places.
