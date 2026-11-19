"""perfect_foresight_lp — LP-based perfect-foresight battery dispatch.

Computes the cost-optimal battery dispatch for a single site using a linear
program with complete future knowledge of prices. This is the theoretical-best
benchmark against which other coordination strategies are measured, forming the
denominator of the Value Capture Ratio (VCR).

Objective (dispatch-dependent part of the household bill only; daily charges
are constant across strategies and are excluded):

    minimise  Σ_t [ imp[t]·p_imp[t]  −  exp[t]·p_exp[t] ]
            + c_deg · Σ_t [ charge[t] + discharge[t] ]
            + c_curtail · Σ_t [ curtail[t] ]

subject to flow balance, SOC continuity, SOC bounds, power limits, and an
optional per-site export cap. See docs/perfect_foresight_lp_spec.md for the
full problem statement.

Variable layout (flat vector of length 6T):
    x[0   : T  ] = charge      (AC kWh into battery, per interval)
    x[T   : 2T ] = discharge   (AC kWh out of battery, per interval)
    x[2T  : 3T ] = imp         (grid import kWh, per interval)
    x[3T  : 4T ] = exp         (site export kWh, per interval)
    x[4T  : 5T ] = soc         (DC battery state of charge, end-of-interval)
    x[5T  : 6T ] = curtail     (solar curtailment kWh, per interval)

Solar curtailment is included so the LP remains feasible when solar cannot be
fully absorbed by load, storage, and export (e.g. a full battery plus a binding
export limit, or a negative export rate making curtailment preferable to paying
an export penalty). Its cost, c_curtail (curtail_cost_per_kwh_dollars), has
three modes: 0.0 (default, unchanged from this function's original behaviour)
allows curtailment at zero objective cost, its implicit cost being the
foregone export credit already captured by the reduced exp[t]; None pins
curtailment to zero, at the risk of infeasibility if solar surplus cannot be
absorbed elsewhere; a positive value prices curtailment explicitly, avoiding
that infeasibility risk while still discouraging it. See
docs/perfect_foresight_lp_spec.md, "Curtailment cost modes", for full detail.

Equality constraints (2T rows):
    rows 0..T-1 : flow balance
        imp[t] - exp[t] - charge[t] + discharge[t] - curtail[t] = load[t] - solar[t]
    rows T..2T-1: SOC continuity
        soc[t] - η_c·charge[t] + (1/η_d)·discharge[t] - soc[t-1] = 0
        (RHS = soc_init for t=0, where soc[-1] ≡ soc_init)

Terminal SOC is enforced via tight bounds on soc[T-1] rather than an additional
equality row, which is numerically clean for both HiGHS and Gurobi.

Constraint matrix density: ~9T nonzeros in a 2T × 6T matrix.
For a full non-leap year (T = 105,120): ~950 K nonzeros.

Solver backend: scipy/HiGHS. A Gurobi backend (_solve_gurobi) receiving the
identical sparse constraint matrix is present but disabled: _GUROBI_AVAILABLE
is hard-set to False and the gurobipy import is commented out. Query the
active backend with lp_backend().
"""
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linprog

from model.core.scenario import BatterySpec
from model.core.site_result import SiteTraces


# Numerical tolerances
_WARN_TOL = 1e-6    # warn if a clipped negative has magnitude above this
_SIMUL_TOL = 1e-6   # threshold for simultaneous charge+discharge / imp+exp warning

# Detect Gurobi once at import time so every call pays no overhead.
# try:
#     import gurobipy as _gp
#     from gurobipy import GRB as _GRB
#     _GUROBI_AVAILABLE = True
# except ImportError:
#     _GUROBI_AVAILABLE = False
_GUROBI_AVAILABLE = False # uncomment to force HiGHS for testing/comparison

def lp_backend() -> str:
    """Return the name of the LP solver backend that will be used."""
    return 'gurobi' if _GUROBI_AVAILABLE else 'highs'


def perfect_foresight_lp(
    site_id: str,
    solar_kwh: pd.Series,
    load_kwh: pd.Series,
    price_imp: pd.Series,
    price_exp: pd.Series,
    battery_spec: BatterySpec,
    soc_init_kwh: float,
    soc_final_kwh: float,
    export_limit_kwh_per_interval: float = np.inf,
    interval_hours: float = 5 / 60,
    curtail_cost_per_kwh_dollars: float | None = 0.0,
) -> SiteTraces:
    """Compute the cost-optimal battery dispatch via a linear program.

    Args:
        site_id: Passed through to the output SiteTraces.
        solar_kwh: Gross solar generation per interval, kWh. May be negative
            (inverter parasitic draw); handled correctly by the flow
            decomposition.
        load_kwh: Gross household load per interval, kWh. Must be >= 0.
        price_imp: Per-interval import price, $/kWh. Must satisfy
            price_imp[t] >= price_exp[t] at every t (hard error if violated).
        price_exp: Per-interval export price (value earned per kWh exported),
            $/kWh. May be negative (export penalty).
        battery_spec: Battery specification. c_deg_per_kwh is the throughput-
            proportional degradation cost added to the LP objective.
        soc_init_kwh: Battery DC SOC at the start of the period. Must lie
            within [soc_min, capacity_kwh].
        soc_final_kwh: Required battery DC SOC at the end of the period.
            The orchestrator sets this to the self_consumption end-of-period
            SOC so the LP vs self_consumption bill comparison is fair. Must
            lie within [soc_min, capacity_kwh].
        export_limit_kwh_per_interval: Per-interval export cap in kWh.
            np.inf (default) means unconstrained.
        interval_hours: Interval length in hours. Default 5/60 (5-min).
        curtail_cost_per_kwh_dollars: Cost per kWh of curtailed solar, in
            dollars (not cents). Three modes:
              0.0 (default) -- curtailment allowed at zero cost, identical
                to this function's original behaviour before this parameter
                existed. No existing caller is affected.
              None -- curtailment pinned to zero (hi[curtail] = 0). Carries
                a real, if likely small for sites with real observed export
                limits, infeasibility risk if solar surplus cannot be
                absorbed by load, storage headroom, and the export limit in
                some interval.
              > 0 -- curtailment allowed, priced in the objective at this
                rate. Avoids the infeasibility risk of pinning while still
                discouraging curtailment except where genuinely optimal.

    Returns:
        SiteTraces with LP-optimal import/export and full flow decomposition,
        in exactly the same shape as self_consumption output.

    Raises:
        ValueError: if price_imp < price_exp at any interval; if any input
            series contain NaN or misalign; or if soc_init/soc_final are
            outside battery bounds.
        RuntimeError: if the solver does not find an optimal solution.
    """
    _validate_inputs(
        solar_kwh, load_kwh, price_imp, price_exp,
        battery_spec, soc_init_kwh, soc_final_kwh,
    )

    index = solar_kwh.index
    T = len(index)
    solar = solar_kwh.to_numpy(dtype=float, copy=False)
    load  = load_kwh.to_numpy(dtype=float, copy=False)
    p_imp = price_imp.to_numpy(dtype=float, copy=False)
    p_exp = price_exp.to_numpy(dtype=float, copy=False)

    c_obj, A_eq, b_eq, lo, hi = _build_lp(
        T, solar, load, p_imp, p_exp, battery_spec,
        soc_init_kwh, soc_final_kwh,
        export_limit_kwh_per_interval, interval_hours,
        curtail_cost_per_kwh_dollars,
    )

    if _GUROBI_AVAILABLE:
        x = _solve_gurobi(c_obj, A_eq, b_eq, lo, hi, site_id)
    else:
        x = _solve_highs(c_obj, A_eq, b_eq, lo, hi, site_id)

    OC, OD, OI, OE, OS, OQ = 0, T, 2*T, 3*T, 4*T, 5*T

    charge    = _clip_nonneg(x[OC:OC+T], 'charge',    site_id)
    discharge = _clip_nonneg(x[OD:OD+T], 'discharge', site_id)
    imp       = _clip_nonneg(x[OI:OI+T], 'imp',       site_id)
    exp_      = _clip_nonneg(x[OE:OE+T], 'exp',       site_id)
    soc       = x[OS:OS+T]
    curtail   = _clip_nonneg(x[OQ:OQ+T], 'curtail',   site_id)

    _check_simultaneous_flows(charge, discharge, imp, exp_, site_id)
    charge, discharge = _net_simultaneous_charge_discharge(charge, discharge)

    return _reconstruct_flows(site_id, index, solar, load, charge, discharge, soc, curtail)


# ---------------------------------------------------------------------------
# Solver backends
# ---------------------------------------------------------------------------

def _solve_highs(c_obj, A_eq, b_eq, lo, hi, site_id: str) -> np.ndarray:
    """Solve via scipy/HiGHS (dual simplex by default)."""
    result = linprog(c_obj, A_eq=A_eq, b_eq=b_eq,
                     bounds=np.column_stack([lo, hi]), method='highs')
    if result.status != 0:
        raise RuntimeError(
            f"HiGHS LP did not find an optimal solution for site {site_id}. "
            f"status={result.status}, message={result.message!r}"
        )
    return result.x


def _solve_gurobi(c_obj, A_eq, b_eq, lo, hi, site_id: str) -> np.ndarray:
    """Solve via Gurobi (requires gurobipy and a valid licence).

    Threads is fixed to 1 so that parallel multi-site runs (ProcessPoolExecutor)
    get one core per worker without internal thread contention. The single-thread
    per-site time is slightly higher than Gurobi's multi-threaded default, but
    the parallel speedup across sites far outweighs this cost.
    """
    m = _gp.Model()
    m.setParam('OutputFlag', 0)   # suppress solver stdout/log
    m.setParam('Threads', 1)      # one thread per process; avoids contention
    x = m.addMVar(len(c_obj), lb=lo, ub=hi)
    m.setObjective(c_obj @ x, _GRB.MINIMIZE)
    m.addConstr(A_eq @ x == b_eq)
    m.optimize()
    if m.Status != _GRB.OPTIMAL:
        raise RuntimeError(
            f"Gurobi LP did not reach optimal for site {site_id}. "
            f"status={m.Status}"
        )
    return x.X


# ---------------------------------------------------------------------------
# LP construction
# ---------------------------------------------------------------------------

def _build_lp(
    T: int,
    solar: np.ndarray,
    load: np.ndarray,
    p_imp: np.ndarray,
    p_exp: np.ndarray,
    bspec: BatterySpec,
    soc_init: float,
    soc_final: float,
    export_limit: float,
    interval_hours: float,
    curtail_cost_per_kwh_dollars: float | None = 0.0,
) -> tuple:
    """Build the LP objective, sparse equality constraint matrix, and bounds.

    Returns (c, A_eq, b_eq, lo, hi) ready for scipy.optimize.linprog.
    """
    OC, OD, OI, OE, OS, OQ = 0, T, 2*T, 3*T, 4*T, 5*T
    N = 6 * T

    # ---- Objective ----
    c = np.zeros(N)
    c[OC:OC+T] = bspec.c_deg_per_kwh      # charge throughput degradation
    c[OD:OD+T] = bspec.c_deg_per_kwh      # discharge throughput degradation
    c[OI:OI+T] = p_imp                    # import cost
    c[OE:OE+T] = -p_exp                   # export credit (negative of earned value)
    # curtail: zero coefficient by default (implicit cost is foregone export
    # credit); a positive curtail_cost_per_kwh_dollars adds an explicit price.
    if curtail_cost_per_kwh_dollars:
        c[OQ:OQ+T] = curtail_cost_per_kwh_dollars

    eta_c = bspec.charge_efficiency
    eta_d = bspec.discharge_efficiency
    t  = np.arange(T, dtype=np.intp)
    t1 = np.arange(1, T, dtype=np.intp)  # t >= 1, for soc[t-1] terms

    # ---- Flow-balance rows (0..T-1): 5 nonzeros each ----
    # -charge[t] + discharge[t] + imp[t] - exp[t] - curtail[t] = load[t] - solar[t]
    # fb_cols is grouped by variable type: all OC+t, then OD+t, OI+t, OE+t, OQ+t.
    # fb_rows must match that grouping: np.tile(t,5) = [0,1,...] x5.
    fb_rows = np.tile(t, 5)
    fb_cols = np.concatenate([OC+t, OD+t, OI+t, OE+t, OQ+t])
    fb_data = np.concatenate([-np.ones(T), np.ones(T), np.ones(T),
                               -np.ones(T), -np.ones(T)])
    b_fb    = load - solar

    # ---- SOC-continuity rows (T..2T-1) ----
    # For all t: soc[t] - η_c·charge[t] + (1/η_d)·discharge[t] = 0   (or soc_init for t=0)
    # For t>=1 only: - soc[t-1] term
    sc_rows = np.concatenate([T+t, T+t, T+t])
    sc_cols = np.concatenate([OC+t, OD+t, OS+t])
    sc_data = np.concatenate([
        -np.full(T, eta_c),
        np.full(T, 1.0 / eta_d),
        np.ones(T),
    ])
    # soc[t-1] link (t >= 1 only)
    if T > 1:
        sc_rows = np.concatenate([sc_rows, T + t1])
        sc_cols = np.concatenate([sc_cols, OS + t1 - 1])
        sc_data = np.concatenate([sc_data, -np.ones(T - 1)])

    b_sc        = np.zeros(T)
    b_sc[0]     = soc_init   # soc[-1] moved to RHS for t=0

    # ---- Assemble sparse A_eq ----
    all_rows = np.concatenate([fb_rows, sc_rows])
    all_cols = np.concatenate([fb_cols, sc_cols])
    all_data = np.concatenate([fb_data, sc_data])
    A_eq = sp.csc_matrix((all_data, (all_rows, all_cols)), shape=(2*T, N))
    b_eq = np.concatenate([b_fb, b_sc])

    # ---- Bounds ----
    power_kwh = bspec.power_kw * interval_hours
    soc_min   = (bspec.backup_reserve_pct / 100.0) * bspec.capacity_kwh

    lo = np.zeros(N)
    hi = np.full(N, np.inf)

    hi[OC:OC+T] = power_kwh              # charge <= AC power cap
    hi[OD:OD+T] = power_kwh              # discharge <= AC power cap
    # imp: [0, inf] (already set)
    hi[OE:OE+T] = export_limit           # exp <= export cap (inf if unconstrained)
    lo[OS:OS+T] = soc_min                # soc >= reserve floor
    hi[OS:OS+T] = bspec.capacity_kwh     # soc <= capacity
    # Terminal SOC enforced via tight bounds on soc[T-1]
    lo[OS + T - 1] = soc_final
    hi[OS + T - 1] = soc_final
    # curtail: [0, max(0, solar[t])] — can only curtail non-negative solar,
    # unless curtail_cost_per_kwh_dollars is None, which pins it to zero.
    hi[OQ:OQ+T] = (
        np.maximum(0.0, solar) if curtail_cost_per_kwh_dollars is not None
        else 0.0
    )

    return c, A_eq, b_eq, lo, hi


# ---------------------------------------------------------------------------
# Flow decomposition
# ---------------------------------------------------------------------------

def _reconstruct_flows(
    site_id: str,
    index: pd.DatetimeIndex,
    solar: np.ndarray,
    load: np.ndarray,
    charge: np.ndarray,
    discharge: np.ndarray,
    soc: np.ndarray,
    curtail: np.ndarray,
) -> SiteTraces:
    """Reconstruct the full SiteTraces flow decomposition from LP variables.

    Priority conventions:
        (1) Solar fills battery charge before grid.
        (2) Battery discharge meets load before exporting to grid.

    Curtailment reduces the effective solar that enters the flow decomposition:
    the six split flows are computed on (solar - curtail), not raw solar. This
    means sol_to_home + sol_to_bat + sol_to_grid = solar - curtail, which is
    consistent with the LP flow balance.

    All four SiteTraces invariants hold algebraically from the LP flow balance;
    no rounding correction is needed.
    """
    # Effective solar after curtailment
    solar_eff = solar - curtail          # >= 0 by curtail bounds (curtail <= solar)

    # Step 1: effective solar meets load (negative solar_eff = parasitic draw)
    sol_to_home   = np.minimum(solar_eff, load)
    surplus_solar = solar_eff - sol_to_home     # >= 0
    residual_load = load      - sol_to_home     # >= 0

    # Step 2: charge split — solar first, then grid
    sol_to_bat  = np.minimum(surplus_solar, charge)
    grid_to_bat = charge - sol_to_bat           # >= 0

    # Step 3: remaining surplus exports
    sol_to_grid = surplus_solar - sol_to_bat    # >= 0

    # Step 4: discharge split — covers load first, remainder exports
    bat_to_home = np.minimum(residual_load, discharge)
    bat_to_grid = discharge - bat_to_home       # >= 0

    # Step 5: grid covers remaining load
    grid_to_home = residual_load - bat_to_home  # >= 0

    # Derived import/export (satisfies SiteTraces invariants exactly)
    import_kwh = grid_to_home + grid_to_bat
    export_kwh = sol_to_grid  + bat_to_grid

    def _s(arr: np.ndarray, name: str) -> pd.Series:
        return pd.Series(arr.copy(), index=index, name=name)

    return SiteTraces(
        site_id=site_id,
        import_kwh=_s(import_kwh,   'import_kwh'),
        export_kwh=_s(export_kwh,   'export_kwh'),
        battery_soc_kwh=_s(soc,     'battery_soc_kwh'),
        sol_to_home_kwh=_s(sol_to_home,   'sol_to_home_kwh'),
        sol_to_bat_kwh=_s(sol_to_bat,     'sol_to_bat_kwh'),
        sol_to_grid_kwh=_s(sol_to_grid,   'sol_to_grid_kwh'),
        bat_to_home_kwh=_s(bat_to_home,   'bat_to_home_kwh'),
        grid_to_home_kwh=_s(grid_to_home, 'grid_to_home_kwh'),
        grid_to_bat_kwh=_s(grid_to_bat,   'grid_to_bat_kwh'),
        bat_to_grid_kwh=_s(bat_to_grid,   'bat_to_grid_kwh'),
        curtail_kwh=_s(curtail,           'curtail_kwh'),
    )


# ---------------------------------------------------------------------------
# Validation and post-solve checks
# ---------------------------------------------------------------------------

def _validate_inputs(
    solar_kwh: pd.Series,
    load_kwh: pd.Series,
    price_imp: pd.Series,
    price_exp: pd.Series,
    bspec: BatterySpec,
    soc_init: float,
    soc_final: float,
) -> None:
    """Validate all LP inputs, raising ValueError on any problem."""
    # Index alignment
    if not solar_kwh.index.equals(load_kwh.index):
        raise ValueError(
            "solar_kwh and load_kwh must share the same index "
            f"(solar len={len(solar_kwh)}, load len={len(load_kwh)})."
        )
    for name, s in (('price_imp', price_imp), ('price_exp', price_exp)):
        if not s.index.equals(solar_kwh.index):
            raise ValueError(
                f"{name} index does not match solar_kwh index "
                f"({name} len={len(s)}, solar len={len(solar_kwh)})."
            )

    # NaN checks
    for name, s in (('solar_kwh', solar_kwh), ('load_kwh', load_kwh),
                    ('price_imp', price_imp), ('price_exp', price_exp)):
        if s.isna().any():
            raise ValueError(f"{name} contains NaN values.")

    # Load must be non-negative (solar may be negative — parasitic draw)
    if (load_kwh < 0).any():
        raise ValueError("load_kwh contains negative values.")

    # Price ordering: p_exp[t] <= p_imp[t] everywhere (hard error)
    p_imp_arr = price_imp.to_numpy()
    p_exp_arr = price_exp.to_numpy()
    violation  = p_exp_arr > p_imp_arr
    if violation.any():
        first = int(violation.argmax())
        raise ValueError(
            f"price_exp exceeds price_imp at {violation.sum()} interval(s). "
            f"First violation at index {first}: "
            f"p_imp={p_imp_arr[first]:.6f}, p_exp={p_exp_arr[first]:.6f}. "
            f"This would incentivise simultaneous import and export in the LP."
        )

    # SOC boundary checks
    soc_min = (bspec.backup_reserve_pct / 100.0) * bspec.capacity_kwh
    tol = 1e-9
    for name, val in (('soc_init_kwh', soc_init), ('soc_final_kwh', soc_final)):
        if val < soc_min - tol or val > bspec.capacity_kwh + tol:
            raise ValueError(
                f"{name}={val:.6f} is outside battery bounds "
                f"[{soc_min:.6f}, {bspec.capacity_kwh:.6f}]."
            )


def _clip_nonneg(arr: np.ndarray, name: str, site_id: str) -> np.ndarray:
    """Floor solver numerical noise to zero; warn if magnitude is large."""
    neg_mask = arr < 0
    if neg_mask.any():
        max_neg = float(-arr[neg_mask].min())
        if max_neg > _WARN_TOL:
            warnings.warn(
                f"LP [{site_id}] {name}: {neg_mask.sum()} value(s) below 0 "
                f"with max magnitude {max_neg:.2e} (>{_WARN_TOL:.0e}); clipping.",
                stacklevel=4,
            )
        return np.clip(arr, 0.0, None)
    return arr


def _check_simultaneous_flows(
    charge: np.ndarray,
    discharge: np.ndarray,
    imp: np.ndarray,
    exp: np.ndarray,
    site_id: str,
) -> None:
    """Warn if the solver produced simultaneous charge+discharge or imp+exp.

    With p_imp >= p_exp and c_deg >= 0 a correctly-structured LP should never
    find it strictly beneficial to do this simultaneously. When it occurs it
    is a solver degeneracy artifact: the LP has multiple optimal solutions with
    the same objective value, and the solver landed at one where complementarity
    is not enforced. This is more common with Gurobi's barrier+crossover method
    than with HiGHS's dual simplex.

    Billing does not use the raw imp/exp: import_kwh and export_kwh are
    re-derived by _reconstruct_flows from charge/discharge after
    _net_simultaneous_charge_discharge has removed any overlap. This check
    runs before netting, so it reports the solver's raw output.
    """
    simul_cd = (charge > _SIMUL_TOL) & (discharge > _SIMUL_TOL)
    if simul_cd.any():
        warnings.warn(
            f"LP [{site_id}]: {simul_cd.sum()} interval(s) with simultaneous "
            f"charge and discharge (solver degeneracy; billing unaffected).",
            stacklevel=4,
        )
    simul_ie = (imp > _SIMUL_TOL) & (exp > _SIMUL_TOL)
    if simul_ie.any():
        warnings.warn(
            f"LP [{site_id}]: {simul_ie.sum()} interval(s) with simultaneous "
            f"import and export (solver degeneracy; billing unaffected).",
            stacklevel=4,
        )


def _net_simultaneous_charge_discharge(
    charge: np.ndarray,
    discharge: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove any simultaneous charge+discharge before flow reconstruction.

    If c_deg_per_kwh is insufficiently high, the LP may find it optimal to
    simultaneously charge and discharge in an interval to reduce the battery SOC,
    allowing additional import now or in a later interval at a sufficiently 
    negative price.  Netting here is sufficient to avoid billable simultaneous
    import_kwh/export_kwh, which would otherwise occur in the flow reconstruction.

    Called on the already-clipped charge/discharge, after
    _check_simultaneous_flows has run (so the warning still reports the
    solver's original, pre-correction output).
    """
    overlap = np.minimum(charge, discharge)
    return charge - overlap, discharge - overlap