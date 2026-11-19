# perfect_foresight_custom — Specification

## Purpose and scope

`model/components/perfect_foresight_custom.py` computes a near-optimal battery
dispatch for one site over a period, with perfect knowledge of the prices the
site faces, using a greedy iterative shift heuristic instead of a linear
program. It is the `perfect_foresight_custom` coordination type, a second
perfect-foresight benchmark alongside `perfect_foresight_lp`
(`perfect_foresight_lp_spec.md`).

Starting from a self-consumption baseline, the heuristic repeatedly moves
stored energy from the cheapest available source interval to a higher-value
sink interval, and stops when no remaining move clears the degradation
threshold. The result is a local optimum: no single source-to-sink exchange
reduces cost. The LP gives the exact dispatch-cost floor; this method builds a
dispatch from individual reallocations, the way an operator would, and the gap
between the two measures what that approach leaves behind.

It takes the same inputs as `perfect_foresight_lp` and returns a `SiteTraces`
of the same shape, so billing and everything downstream treat both
identically. It consumes the `price_imp` / `price_exp` vectors from
`model/core/prices.py`. Tariff structure, loss factors, unit conversion and
local time are resolved upstream, and the component never sees a state or time
zone. Daily and fixed charges do not depend on dispatch and are added
afterwards by `compute_bill`.

---

## Formulation

### Objective

The objective is that of `perfect_foresight_lp`: minimise import cost less
export value against `price_imp` / `price_exp`, plus a cycling cost. Here it
is enforced shift by shift rather than solved globally. A shift is made only
when its margin exceeds `threshold = 2 × c_deg_per_kwh` ($ per DC kWh
shifted), which stands for the degradation cost the LP charges on the charge
and discharge legs. With `c_deg_per_kwh = 0` any strictly positive margin is
taken.

### Conventions and state

As in the LP: SOC is DC, end of interval,
`soc[t] = soc[t-1] + η_c·charge_AC − discharge_AC/η_d`; all flows are AC kWh
per interval; `curtail` is AC solar.
`soc_min = backup_reserve_pct/100 × capacity_kwh`. A shift is measured in DC
kWh (`shift_kwh`), the SOC change it makes.

State is held in flat float64 numpy arrays of length T, indexed by position:
the inputs `solar`, `load`, `p_imp`, `p_exp`, and the mutable `soc`,
`sol_to_home`, `sol_to_bat`, `sol_to_grid`, `bat_to_home`, `bat_to_grid`,
`grid_to_home`, `grid_to_bat` and `curtail`. The arrays hold the full flow
decomposition at every step.

### Baseline

1. The arrays are seeded from a `neg_price_aware` self-consumption run
   (`model_spec.md`, Self-consumption), passed in as `sc_traces` or computed
   internally when `sc_traces` is `None`. Self-consumption never discharges to
   the grid, so `bat_to_grid` starts at zero, and it charges from the grid only
   at negative import prices.
2. Self-consumption ignores the export limit. Solar export above
   `export_limit_kwh_per_interval`, and all solar export at a negative export
   price, is moved from `sol_to_grid` to `curtail`.

### Opportunities

Every interval contributes two sink opportunities: import avoidance (`AVOID`,
seed value `p_imp[t]`) and export (`EXPORT`, seed value `p_exp[t]`). The 2T
opportunities are sorted once in descending seed value and walked in that
order. The seed only sets the order; each sink's value and amount are
re-evaluated from the current state when it is reached. `opt_factor`
truncates the walk to the first `floor(2T × opt_factor)` opportunities.

### Sinks

Each sink is serviced repeatedly, against the cheapest viable source each
time, until it is exhausted, no longer applies, or no viable source remains.

| Sink | Applies when | Amount (AC) | Hurdle on source cost | DC per AC |
|---|---|---|---|---|
| Relocate grid charging (`AVOID`, first) | `grid_to_bat[j] > 0` | `grid_to_bat[j]` | `p_imp[j] − threshold` | `× η_c` |
| Displace import (`AVOID`) | `grid_to_bat[j] = 0` | `min(grid_to_home[j], discharge headroom)` | `p_imp[j]·η_c·η_d − threshold` | `÷ η_d` |
| Redirect solar charge to export (`EXPORT`) | `sol_to_bat[j] > 0`, `grid_to_bat[j] = 0` | `min(sol_to_bat[j], export headroom)` | `p_exp[j] − threshold` | `× η_c` |
| Discharge to grid (`EXPORT`) | `sol_to_bat[j] = 0`, `grid_to_bat[j] = 0` | `min(discharge headroom, export headroom)` | `p_exp[j]·η_c·η_d − threshold` | `÷ η_d` |

Discharge headroom is `power_kw × interval_hours − bat_to_home[j] −
bat_to_grid[j]`. Export headroom is `export_cap − sol_to_grid[j] −
bat_to_grid[j]`.

Relocation and redirect move charging rather than adding a cycle, so their
hurdles carry no round-trip loss. Relocation runs to completion before
displacement starts, and the other sinks are skipped while `grid_to_bat[j] >
0`, so an interval never charges from the grid while discharging. Once a
redirect has used up `sol_to_bat[j]`, the same sink is re-evaluated in
discharge mode.

### Sources

Each candidate interval k is classified as the first applicable source:

| Source | Applies when | Cost compared with the hurdle | DC available |
|---|---|---|---|
| Grid charge at a negative import price | `p_imp[k] < 0` | `p_imp[k]` | charge headroom |
| Curtailed solar | `curtail[k] > 0` | 0 | `min(curtail[k]·η_c, charge headroom)` |
| Exported solar | `sol_to_grid[k] > 0` | `p_exp[k]` | `min(sol_to_grid[k]·η_c, charge headroom)` |
| Cancel discharge to home | `bat_to_home[k] > 0` | `p_imp[k]·η_c·η_d` | `bat_to_home[k]/η_d` |
| Grid charge | otherwise | `p_imp[k]` | charge headroom |

Charge headroom is `(power_kw × interval_hours − sol_to_bat[k] −
grid_to_bat[k]) × η_c`.

Cancelling a discharge keeps energy that is already stored, so only the
discharge loss at the sink applies and the break-even is `p_imp[k] <` sink
value. Reporting its cost as `p_imp[k]·η_c·η_d` makes that condition
comparable with the round-trip hurdles. Its DC is limited to the discharge
being cancelled. Once `bat_to_home[k]` is used up, the interval appears as a
new grid-charge source at the unadjusted cost, so new charging never gets the
lower hurdle.

At a negative import price, grid charging takes precedence over curtailed
solar and over cancelling a discharge: being paid to import is worth more, and
the baseline's negative-price curtailment is kept. Otherwise curtailed solar
has zero cost and is taken first, so a shift that frees battery headroom winds
back curtailment.

### Search window

Sources are searched on both sides of sink j, each side bounded by the SOC
barrier that blocks carrying energy across it:

- **Earlier sources (k < j)** raise SOC on `[k, j)`. The window ends at the
  nearest earlier interval where the battery is full. The carry limit is the
  minimum of `cap − soc` over `[k, j−1]`.
- **Later sources (k > j)** lower SOC on `[j, k)`. The window ends at the
  nearest later interval where SOC is at `soc_min`. The carry limit is the
  minimum of `soc − soc_min` over `[j, k−1]`.

A source is viable if its cost is below the hurdle and the smaller of its DC
available and the carry limit is positive. Across both windows the
lowest-cost viable source is chosen, ties going to the source nearest j.

### Applying a shift

`shift_kwh` is the smaller of the sink amount converted to DC and the source's
DC available after the carry limit. A shift then:

1. adds `shift_kwh` to `soc[k:j]` (earlier source) or subtracts it from
   `soc[j:k]` (later source);
2. at the source, raises stored energy by `shift_kwh`. Grid charge and solar
   capture add `shift_kwh/η_c` AC to `grid_to_bat` or `sol_to_bat`, taking
   captured solar from `curtail` or `sol_to_grid`. A grid-charge source first
   cancels discharge, moving up to `shift_kwh·η_d` AC from `bat_to_home` to
   `grid_to_home`, and charges from the grid only for the remainder;
3. at the sink, lowers stored energy by `shift_kwh`. Relocation removes
   `shift_kwh/η_c` from `grid_to_bat`; displacement moves `shift_kwh·η_d`
   from `grid_to_home` to `bat_to_home`; redirect moves `shift_kwh/η_c` from
   `sol_to_bat` to `sol_to_grid`; discharge to grid adds `shift_kwh·η_d` to
   `bat_to_grid`.

Each shift adds and removes the same DC, so the terminal SOC equals the
baseline's by construction, and the solar, load, import and export identities
hold after every shift.

### Performance

- Positional numpy indexing throughout, with no timestamp arithmetic in the
  loop.
- One O(T log T) sort. Each sink's search is O(W) in the window length W,
  typically under a day of intervals.
- Window ends are found by scanning SOC in 512-interval chunks, and the
  sources in a window are evaluated in one vectorised pass
  (`_source_arrays`).

---

## Interface

```python
def perfect_foresight_custom(
    site_id: str,
    solar_kwh: pd.Series,          # kWh per interval (may be negative: inverter parasitic draw)
    load_kwh: pd.Series,           # kWh per interval, >= 0
    price_imp: pd.Series,          # $/kWh per interval
    price_exp: pd.Series,          # $/kWh per interval (may be negative)
    battery_spec: BatterySpec,
    soc_init_kwh: float,
    soc_final_kwh: float,
    export_limit_kwh_per_interval: float = np.inf,
    interval_hours: float = 5 / 60,
    opt_factor: float = 1.0,       # fraction of sorted opportunities processed
    sc_traces: SiteTraces | None = None,   # baseline; None runs self_consumption internally
) -> SiteTraces
```

When `sc_traces` is supplied, its index must match `solar_kwh`,
`soc_init_kwh` must equal its first SOC and `soc_final_kwh` its last. When it
is `None`, self-consumption runs internally from `soc_min`, so `soc_init_kwh`
must equal `soc_min` and `soc_final_kwh` the self-consumption end SOC.
Passing the output of a previous pass, or of the LP, as `sc_traces` runs the
heuristic from that starting point (`check_perfect_foresight_custom_twice`).

### How the orchestrator calls it

`_run_perfect_foresight_custom` runs `_perfect_foresight_custom_site_worker`
for each site, serially or in a process pool. Each worker runs
`neg_price_aware` self-consumption and bills it as the baseline, then calls
the component with that run as `sc_traces`, `soc_init_kwh = soc_min`,
`soc_final_kwh` set to the baseline's end SOC clamped to
`[soc_min, capacity_kwh]`, the site's resolved export limit and
`scenario.opt_factor`. Self-consumption therefore runs once per site.

---

## Configuration

| Key | Default | Rule |
|---|---|---|
| `opt_factor` | 1.0 | in (0, 1]; used only by `perfect_foresight_custom` |
| `c_deg_per_kwh` (battery) | per battery spec | sets `threshold` |
| `export_limit_kw_per_phase` | none (unlimited) | resolved per site, as for the LP |

---

## Failure handling

`ValueError` is raised for a missing `battery_spec`, misaligned indices, NaN in
any input, negative load, `price_exp > price_imp` in any interval,
`soc_init_kwh` or `soc_final_kwh` outside `[soc_min, capacity_kwh]`,
`opt_factor` outside (0, 1], an `sc_traces` index that does not match, and a
baseline whose boundary SOCs do not match `soc_init_kwh` / `soc_final_kwh`.

After the loop, negative flows are clipped to zero, with a warning if any
exceeds 1e-6 kWh. `sol_to_home` is not clipped: it is negative where solar is
negative, as in the LP and self-consumption. A warning, not an error, is then
raised for a solar-split, load-split or SOC-continuity residual above 1e-6,
SOC outside its bounds, a terminal SOC more than 1e-6 kWh from
`soc_final_kwh`, or any interval that both charges and discharges. A result is
always returned.

---

## Verification

`model/tests/test_perfect_foresight_custom.py` is a script harness, run as
`python3 -m model.tests.test_perfect_foresight_custom`. It covers seven
synthetic cases: arbitrage without solar, solar with an evening peak,
negative-price curtailment wind-back, a binding export limit, flat prices (no
shift), a degradation cost that suppresses marginal cycling, and a random
5-minute day. For each it checks the solar, load, import and export
identities, SOC continuity and bounds, the terminal SOC and non-negative
flows, and that the dispatch cost is no lower than an LP solve of the same
problem (scipy `linprog`) and no higher than the baseline's.

Dev scripts (`model/dev/`): `check_perfect_foresight_custom` runs one real
site and, with `--compare-lp`, reports the gap to `perfect_foresight_lp`;
`check_perfect_foresight_custom_twice` runs the two-pass and seed-from-LP
tests; `compare_perfect_foresight` compares custom and LP per interval;
`diagnose_shift_window` traces one sink's window search;
`viz_perfect_foresight_custom` is the interactive view.

---

## Known limitations

- **Local optimum.** The heuristic stops where no single exchange improves the
  dispatch. A window cannot carry energy across an interval where the battery
  is full (earlier sources) or empty (later sources), so some LP solutions
  cannot be reached from the self-consumption start by single exchanges. The
  two-pass tests (Validation) show the remaining gap is of this kind rather
  than a loose profitability test.
- **Threshold on throughput-neutral shifts.** Relocating grid charging,
  redirecting solar charge and cancelling a discharge move a charge or
  discharge rather than adding one, but their hurdles still subtract the full
  `2 × c_deg_per_kwh`. With a non-zero degradation cost the heuristic is
  stricter on these shifts than the LP objective. Fix path: subtract the
  threshold only for the throughput a shift adds.
- **Synthetic cases not run by pytest.** The harness has no `test_`
  functions, so `pytest model/tests` collects nothing from it. Fix path: wrap
  each case in a test function.

---

## Validation

Measured on 4 June 2026 (commit `4d85f91`), before relocation of grid charging
(`a0915e2`) and the negative-import-price source (`25cd7f6`) were added on 5
June. Not re-measured since.

- **Synthetic cases:** all seven at a 0.00% gap to the LP.
- **S0088, NSW, 2022, Amber wholesale, Ausgrid EA025, 10 kW export limit:**
  custom $193.93, LP (HiGHS) $192.14, gap $1.79 (+0.93%). Of the gap, 46% was
  missed export and 54% excess import, spread across hours of the day. The
  largest cluster was February negative-price overnight charging.
- **Two-pass:** a second pass on its own output, and a pass seeded from the LP
  solution, made no change on the sites tested.
- **Solve time:** 19–22 s per site for a full 5-minute year (T = 105,120),
  against about 20 s for HiGHS.
- **Invariants:** residuals about 1e-16 on a full year; SOC bounds, terminal
  SOC, export and power limits all held.
