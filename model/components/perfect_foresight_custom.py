"""perfect_foresight_custom — greedy iterative-shift perfect-foresight dispatch.

A second perfect-foresight benchmark alongside ``perfect_foresight_lp``. Instead
of solving a linear program, it computes a near-optimal battery dispatch with a
greedy iterative shift heuristic built on a sub-component optimality principle:
the dispatch is a set of candidate energy-delivery opportunities, and the
algorithm repeatedly applies the single most valuable charge/discharge
reallocation available until no remaining reallocation improves any opportunity
by more than a degradation threshold (a no-improving-exchange local optimum).

The method seeds from the existing ``self_consumption`` baseline (called
internally, so no self-consumption logic is duplicated), then shifts energy
between intervals. Each shift charges and discharges the same DC amount, so the
terminal SOC is preserved by construction, matching the LP's terminal-SOC pin.

It returns a ``SiteTraces`` in exactly the same shape as ``self_consumption`` and
``perfect_foresight_lp``, so billing and all downstream consumers are unchanged.

See docs/perfect_foresight_custom_spec.md for the full design.

Conventions (identical to perfect_foresight_lp):
    - SOC (battery_soc_kwh) is DC, end-of-interval.
    - eta_c converts AC charge inflow to DC stored; eta_d converts DC drawn to
      AC discharge outflow:  soc[t] = soc[t-1] + eta_c*charge_ac - discharge_ac/eta_d.
    - All flows are AC kWh per interval; curtail is AC solar kWh.
    - price_imp/price_exp are $/kWh per interval from model/core/prices.py.

A "shift" moves shift_kwh (DC) from a source interval to a sink interval:
    - earlier source k < sink j: soc[k:j] += shift_kwh (charge held until j)
    - later   source k > sink j: soc[j:k] -= shift_kwh (discharge covered until k)
with the source and sink flows updated so SOC continuity and the four
SiteTraces invariants hold exactly.
"""
import warnings

import numpy as np
import pandas as pd

from model.core.scenario import BatterySpec
from model.core.site_result import SiteTraces
from model.components.self_consumption import self_consumption


# Numerical tolerances
_EPS = 1e-9          # general "effectively zero" floor for amounts/headroom
_WARN_TOL = 1e-6     # warn if a clipped negative or invariant residual exceeds this
_SIMUL_TOL = 1e-6    # threshold for simultaneous charge+discharge / imp+exp warning
_SOC_TOL = 1e-6      # tolerance on boundary-SOC consistency checks

# Sink kinds
_AVOID = 0           # discharge to displace grid import
_EXPORT = 1          # discharge to grid


def perfect_foresight_custom(
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
    opt_factor: float = 1.0,
    sc_traces: SiteTraces | None = None,
) -> SiteTraces:
    """Compute a near-optimal battery dispatch via greedy iterative shifts.

    Args:
        site_id: Passed through to the output SiteTraces.
        solar_kwh: Gross solar generation per interval, kWh (>= 0 when no
            sc_traces is provided; the internal self_consumption baseline
            requires non-negative solar).
        load_kwh: Gross household load per interval, kWh. Must be >= 0.
        price_imp: Per-interval import price, $/kWh. Must satisfy
            price_imp[t] >= price_exp[t] at every t (hard error if violated).
        price_exp: Per-interval export price (value earned per kWh exported),
            $/kWh. May be negative (export penalty -> economic curtailment).
        battery_spec: Battery specification. c_deg_per_kwh sets the per-kWh
            cycling threshold (threshold = 2 * c_deg_per_kwh).
        soc_init_kwh: Battery DC SOC at the start of the period. When
            sc_traces is None the internal self_consumption baseline starts at
            the reserve floor, so this must equal
            soc_min (= backup_reserve_pct/100 * capacity_kwh). When sc_traces
            is provided this must equal sc_traces.battery_soc_kwh.iloc[0].
        soc_final_kwh: Required battery DC SOC at the end of the period.
            Shifts preserve terminal SOC, so this must equal the baseline's
            end-of-period SOC (sc_traces.battery_soc_kwh.iloc[-1] when
            sc_traces is provided, or the self_consumption end-of-period SOC
            otherwise).
        export_limit_kwh_per_interval: Per-interval export cap in kWh.
            np.inf (default) means unconstrained.
        interval_hours: Interval length in hours. Default 5/60 (5-min).
        opt_factor: Fraction (0, 1] of the sorted opportunity list to process.
            1.0 (default) optimises fully; lower values give a faster approximate
            solution by skipping the lowest-value opportunities.
        sc_traces: Optional pre-computed baseline SiteTraces. When provided,
            its flows and SOC are used as the greedy's starting point and
            self_consumption is not called internally. The index must match
            solar_kwh. Primarily used to run the greedy a second time on its
            own output (two-pass optimisation) or to supply a custom baseline.
            Default None → call self_consumption internally.

    Returns:
        SiteTraces with the greedy-optimised dispatch, in exactly the same shape
        as self_consumption / perfect_foresight_lp output.

    Raises:
        ValueError: on input misalignment, NaN, negative load, price ordering
            violation, out-of-bounds soc_init/soc_final, bad opt_factor, missing
            battery_spec, or a baseline whose boundary SOC is inconsistent with
            soc_init_kwh / soc_final_kwh.
    """
    _validate_inputs(
        solar_kwh, load_kwh, price_imp, price_exp,
        battery_spec, soc_init_kwh, soc_final_kwh, opt_factor,
    )

    index = solar_kwh.index
    n = len(index)

    # Scalars
    cap = float(battery_spec.capacity_kwh)
    soc_min = (battery_spec.backup_reserve_pct / 100.0) * cap
    eta_c = float(battery_spec.charge_efficiency)
    eta_d = float(battery_spec.discharge_efficiency)
    power = float(battery_spec.power_kw) * interval_hours       # AC kWh/interval
    threshold = 2.0 * float(battery_spec.c_deg_per_kwh)         # $/kWh round-trip hurdle
    export_cap = float(export_limit_kwh_per_interval)

    # --- Phase 0: baseline --------------------------------------------------
    # Use a provided SiteTraces when available, otherwise compute self_consumption.
    if sc_traces is not None:
        if not sc_traces.import_kwh.index.equals(index):
            raise ValueError(
                "sc_traces index does not match solar_kwh index "
                f"(sc_traces len={len(sc_traces.import_kwh)}, solar len={n})."
            )
        base = sc_traces
    else:
        base = self_consumption(
            site_id, solar_kwh, load_kwh, battery_spec, interval_hours=interval_hours,
            neg_price_aware=True, price_imp=price_imp, price_exp=price_exp,
        )
    _check_baseline_boundaries(base, soc_min, soc_init_kwh, soc_final_kwh,
                               external=sc_traces is not None)

    # Mutable numpy state (copies; the baseline SiteTraces is not mutated)
    solar = solar_kwh.to_numpy(dtype=float, copy=True)
    load = load_kwh.to_numpy(dtype=float, copy=True)
    p_imp = price_imp.to_numpy(dtype=float, copy=True)
    p_exp = price_exp.to_numpy(dtype=float, copy=True)

    sol_to_home = base.sol_to_home_kwh.to_numpy(dtype=float, copy=True)
    sol_to_bat = base.sol_to_bat_kwh.to_numpy(dtype=float, copy=True)
    sol_to_grid = base.sol_to_grid_kwh.to_numpy(dtype=float, copy=True)
    bat_to_home = base.bat_to_home_kwh.to_numpy(dtype=float, copy=True)
    grid_to_home = base.grid_to_home_kwh.to_numpy(dtype=float, copy=True)
    grid_to_bat = base.grid_to_bat_kwh.to_numpy(dtype=float, copy=True)
    bat_to_grid = base.bat_to_grid_kwh.to_numpy(dtype=float, copy=True)
    soc = base.battery_soc_kwh.to_numpy(dtype=float, copy=True)
    curtail = base.curtail_kwh.to_numpy(dtype=float, copy=True)

    state = _State(
        solar=solar, load=load, p_imp=p_imp, p_exp=p_exp,
        sol_to_home=sol_to_home, sol_to_bat=sol_to_bat, sol_to_grid=sol_to_grid,
        bat_to_home=bat_to_home, grid_to_home=grid_to_home, grid_to_bat=grid_to_bat,
        bat_to_grid=bat_to_grid, soc=soc, curtail=curtail,
        cap=cap, soc_min=soc_min, eta_c=eta_c, eta_d=eta_d, power=power,
        export_cap=export_cap,
    )

    # --- Phase 0b: baseline curtailment pass (this component) --------------
    _baseline_curtail(state)

    # --- Phase 0c: build and sort the opportunity list ---------------------
    # Two opportunities per interval: import-avoidance (seed p_imp) and export
    # (seed p_exp). Seed value is used only for ordering; realised value/amount
    # are recomputed live in Phase 1.
    seed_values = np.concatenate([p_imp, p_exp])
    seed_index = np.concatenate([np.arange(n), np.arange(n)]).astype(np.intp)
    seed_kind = np.concatenate([np.full(n, _AVOID), np.full(n, _EXPORT)]).astype(np.intp)
    order = np.argsort(-seed_values, kind='stable')
    limit = int(np.floor(len(order) * opt_factor))

    # --- Phase 1: greedy shift loop ----------------------------------------
    for pos in range(limit):
        oi = order[pos]
        j = int(seed_index[oi])
        kind = int(seed_kind[oi])
        _service_sink(state, j, kind, threshold)

    # --- Assemble output ---------------------------------------------------
    return _assemble_traces(state, site_id, index, soc_final_kwh)


# ---------------------------------------------------------------------------
# Mutable state container
# ---------------------------------------------------------------------------

class _State:
    """Flat numpy arrays + scalars for the greedy loop. Plain attribute access
    for speed; no validation here."""

    __slots__ = (
        'solar', 'load', 'p_imp', 'p_exp',
        'sol_to_home', 'sol_to_bat', 'sol_to_grid',
        'bat_to_home', 'grid_to_home', 'grid_to_bat', 'bat_to_grid',
        'soc', 'curtail',
        'cap', 'soc_min', 'eta_c', 'eta_d', 'power', 'export_cap',
    )

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# Baseline curtailment
# ---------------------------------------------------------------------------

def _baseline_curtail(st: _State) -> None:
    """Curtail baseline solar exports that breach the export limit or that would
    be exported at a negative price. Moves the curtailed AC solar from
    sol_to_grid into curtail. SOC and battery flows are untouched.

    Vectorised; baseline bat_to_grid is zero so export == sol_to_grid here.
    """
    overflow = np.maximum(0.0, st.sol_to_grid - st.export_cap)
    neg_price_cut = np.where(st.p_exp < 0.0, st.sol_to_grid, 0.0)
    cut = np.maximum(overflow, neg_price_cut)
    st.sol_to_grid -= cut
    st.curtail += cut



# ---------------------------------------------------------------------------
# Greedy servicing of one sink (with retries against successive sources)
# ---------------------------------------------------------------------------

def _service_sink(st: _State, j: int, kind: int, threshold: float) -> None:
    """Service a single sink opportunity at interval j, applying as many shifts
    as remain beneficial. Returns when the sink is exhausted, no longer
    applicable, or no viable source remains.

    AVOID sinks are serviced in two sequential phases:
      A2 phase (if grid_to_bat[j] > 0): move grid charging from j to a cheaper
        interval. No round-trip loss — both source and sink are pure charging
        operations. Hurdle = p_imp[j] − threshold. shift_kwh = amount_ac × η_c.
        Sink application: grid_to_bat[j] -= shift_kwh / η_c. This phase runs to
        completion (or until no cheaper source exists) before the standard phase
        begins, ensuring we never have simultaneous grid charging and discharging.
      Standard AVOID phase (grid_to_bat[j] == 0): discharge battery to displace
        grid_to_home. Full round-trip loss applies.
        Hurdle = p_imp[j] × η_c × η_d − threshold. shift_kwh = amount_ac / η_d.

    EXPORT sinks come in two modes, chosen per iteration:
      Redirect mode (sol_to_bat[j] > 0): cancel solar battery-charging and export
        the solar directly. DC removed from soc = redirect_ac * eta_c (no eta_d
        loss). Hurdle = p_exp − threshold.
      Discharge mode (sol_to_bat[j] == 0): standard bat → grid discharge.
        DC removed = deliver_ac / eta_d. Hurdle = p_exp × η_c × η_d − threshold.
    After redirect exhausts sol_to_bat the loop naturally re-evaluates in
    discharge mode for any remaining export headroom.
    """
    # ── AVOID: A2 pre-phase — relocate grid charging to a cheaper time ────────
    if kind == _AVOID:
        while st.grid_to_bat[j] > _EPS:
            sink_amount_ac = st.grid_to_bat[j]
            shift_kwh = sink_amount_ac * st.eta_c      # DC moved from j to source k
            hurdle = st.p_imp[j] - threshold           # no round-trip loss
            src = _find_best_source(st, j, hurdle)
            if src is None:
                break
            k, _src_cost, src_dc_headroom = src
            shift_kwh = min(shift_kwh, src_dc_headroom)
            if shift_kwh <= _EPS:
                break
            # SOC carry + source: identical to any other shift.
            if k < j:
                st.soc[k:j] += shift_kwh
            else:
                st.soc[j:k] -= shift_kwh
            _apply_source(st, k, shift_kwh)
            # Sink: reduce grid charging at j by the AC amount freed.
            st.grid_to_bat[j] -= shift_kwh / st.eta_c
        # Fall through to standard AVOID phase below.

    # ── Standard loop (AVOID grid_to_home, or EXPORT) ─────────────────────────
    while True:
        sink_value, sink_amount_ac = _evaluate_sink(st, j, kind)
        if sink_amount_ac <= _EPS:
            return

        is_redirect = kind == _EXPORT and st.sol_to_bat[j] > _SIMUL_TOL
        if is_redirect:
            # No round-trip loss at sink: 1 DC from source → 1/eta_c AC exported.
            hurdle = sink_value - threshold
            shift_kwh = min(sink_amount_ac * st.eta_c, np.inf)
        else:
            hurdle = sink_value * st.eta_c * st.eta_d - threshold
            shift_kwh = sink_amount_ac / st.eta_d

        src = _find_best_source(st, j, hurdle)
        if src is None:
            return
        k, _src_cost, src_dc_headroom = src

        shift_kwh = min(shift_kwh, src_dc_headroom)
        if shift_kwh <= _EPS:
            return

        _apply_shift(st, k, j, kind, is_redirect, shift_kwh)
        # Loop: re-evaluate the (reduced) sink against the next-cheapest source.


def _evaluate_sink(st: _State, j: int, kind: int):
    """Return (sink_value, sink_amount_ac) for the live state, or amount 0 if the
    sink no longer applies.

    EXPORT redirect mode (sol_to_bat[j] > 0):
      Does not consume discharge power; it cancels charge power instead.
      Skipped if grid_to_bat[j] > 0 to avoid an interval simultaneously charging
      from grid while being redirected (accounting is undefined).
    EXPORT discharge mode (sol_to_bat[j] == 0):
      Consumes discharge power headroom as usual.
    """
    if kind == _AVOID:
        discharge_headroom = st.power - st.bat_to_home[j] - st.bat_to_grid[j]
        if discharge_headroom <= _EPS:
            return 0.0, 0.0
        # Displace grid import to home. Skip if the interval is charging from
        # grid (would imply simultaneous charge + discharge).
        if st.grid_to_bat[j] > _SIMUL_TOL:
            return 0.0, 0.0
        amount = min(st.grid_to_home[j], discharge_headroom)
        if amount <= _EPS:
            return 0.0, 0.0
        return st.p_imp[j], amount

    # _EXPORT
    export_headroom = st.export_cap - st.sol_to_grid[j] - st.bat_to_grid[j]
    if export_headroom <= _EPS:
        return 0.0, 0.0

    if st.sol_to_bat[j] > _SIMUL_TOL:
        # Redirect mode: cancel solar battery-charge → export directly (no RTE loss).
        # Guard: skip if already grid-charging at j (would create conflicting flows).
        if st.grid_to_bat[j] > _SIMUL_TOL:
            return 0.0, 0.0
        amount = min(st.sol_to_bat[j], export_headroom)
        return st.p_exp[j], amount

    # Discharge mode: bat → grid (standard).
    if st.grid_to_bat[j] > _SIMUL_TOL:
        return 0.0, 0.0
    discharge_headroom = st.power - st.bat_to_home[j] - st.bat_to_grid[j]
    if discharge_headroom <= _EPS:
        return 0.0, 0.0
    amount = min(discharge_headroom, export_headroom)
    if amount <= _EPS:
        return 0.0, 0.0
    return st.p_exp[j], amount


# ---------------------------------------------------------------------------
# Source search
# ---------------------------------------------------------------------------

def _source_arrays(st: _State, k_idx: np.ndarray):
    """Vectorised per-interval charge source classification over the candidate
    indices k_idx. Returns (cost, dc_headroom_uncapped) where the cheapest
    applicable charge option is chosen per interval:

        curtailed solar  -> cost 0
            Wind-back of wasted energy; always worth storing.
        exported solar   -> cost p_exp[k]
            Forego export, store instead. Full round-trip at sink:
            1 AC foregone → η_c DC stored → η_d AC delivered.
            Comparison against hurdle sink_value×η_c×η_d is correct.
        grid charge (cancel existing bat_to_home discharge) -> cost p_imp[k]×η_c×η_d
            The energy is ALREADY IN THE BATTERY. We cancel bat_to_home[k] and
            replace it with grid_to_home[k] at cost p_imp[k] per AC. The
            preserved DC travels to the sink without any further charge loss —
            only the one discharge η_d at the sink applies. The effective cost
            per AC delivered at the sink is p_imp[k], so the correct viability
            condition is p_imp[k] < sink_value. Reporting p_imp[k]×η_c×η_d
            makes this equivalent to the existing hurdle comparison
            (p_imp[k]×η_c×η_d < sink_value×η_c×η_d  ↔  p_imp[k] < sink_value).
        grid charge (new energy, no bat_to_home to cancel) -> cost p_imp[k]
            Full round-trip: 1 AC imported → η_c DC → η_d AC at sink.
            Comparison against hurdle sink_value×η_c×η_d is correct.

    When bat_to_home[k] > 0 the cancel-discharge option is used because it
    is cheaper than new charging. _apply_source handles the mixed regime
    (cancel first, then new-charge for any remainder) correctly.
    """
    cur = st.curtail[k_idx]
    s2g = st.sol_to_grid[k_idx]
    charge_power_ac = np.maximum(
        0.0, st.power - st.sol_to_bat[k_idx] - st.grid_to_bat[k_idx]
    )
    charge_power_dc = charge_power_ac * st.eta_c

    # Negative import price: always classify as grid source (cost = p_imp < 0),
    # even when curtail or sol_to_grid is present. Being paid to import is better
    # than un-curtailing free solar, and _apply_source must add grid_to_bat rather
    # than reversing the neg-price-aware SC baseline curtailment.
    neg_imp = st.p_imp[k_idx] < -_EPS

    cur_mask  = (cur > _EPS) & ~neg_imp
    exp_mask  = (~cur_mask) & (s2g > _EPS) & ~neg_imp
    grid_mask = ~cur_mask & ~exp_mask          # includes all neg_imp intervals

    # Grid cost: adjust for cancel-discharge intervals (bat_to_home > 0).
    # Cancel-discharge has no η_c loss at source, so the effective viability
    # condition is p_imp[k] < sink_value rather than p_imp[k] < sink_value×η_c×η_d.
    # Multiplying by η_c×η_d makes it directly comparable against the existing hurdle.
    # Exception: at negative-price intervals, new grid charge is always preferred over
    # cancel-discharge (adding grid_to_bat at negative cost is more profitable than
    # merely preserving existing DC). So cancel_available excludes neg_imp intervals.
    cancel_available = grid_mask & (st.bat_to_home[k_idx] > _EPS) & ~neg_imp
    grid_cost = np.where(
        cancel_available,
        st.p_imp[k_idx] * st.eta_c * st.eta_d,   # cancel-discharge: adjusted
        st.p_imp[k_idx],                           # new-charge (incl. neg-price)
    )

    cost = np.where(cur_mask, 0.0, np.where(exp_mask, st.p_exp[k_idx], grid_cost))

    # DC headroom per option.
    dc_cur = np.minimum(cur * st.eta_c, charge_power_dc)
    dc_exp = np.minimum(s2g * st.eta_c, charge_power_dc)
    # Grid: two distinct regimes with different hurdle rates, kept separate so
    # the new-charge portion cannot be accessed at the adjusted (lower) hurdle.
    #   Cancel-discharge (bat_to_home > 0, not neg_imp): dc capped at bat_to_home/eta_d
    #     only. Once bat_to_home is exhausted the sink re-evaluates; the interval
    #     then appears as a new-charge source at the unadjusted cost.
    #   New-charge / neg-price: dc = charge_power_dc (full remaining charge power).
    dc_grid = np.where(
        cancel_available,
        st.bat_to_home[k_idx] / st.eta_d,   # cancel portion only
        charge_power_dc,                      # new-charge or neg-price grid
    )

    dc = np.where(cur_mask, dc_cur, np.where(exp_mask, dc_exp, dc_grid))
    return cost, dc


def _find_best_source(st: _State, j: int, hurdle: float):
    """Find the lowest-cost viable charge source within the SOC-bounded windows
    on both sides of sink j. Returns (k, cost, dc_headroom) or None.

    Carry limits:
        backward (k < j): raising soc on [k, j) requires headroom against
            capacity; head_charge[k] = min over [k, j-1] of (cap - soc).
        forward  (k > j): lowering soc on [j, k) requires headroom against the
            reserve floor; head_use[k] = min over [j, k-1] of (soc - soc_min).
    A window ends where the running carry limit reaches ~0 (battery full earlier
    / empty later). The reachable span is discovered with a chunked scan so the
    vectorised evaluation runs only over the (typically sub-daily) window, not
    the whole series.
    """
    n = st.soc.shape[0]
    best_k = -1
    best_cost = np.inf
    best_dc = 0.0
    best_dist = 0

    # ---- Backward window: nearest earlier interval that is full blocks carry ----
    if j > 0:
        ws = _backward_window_start(st, j)            # smallest reachable index
        k_idx = np.arange(ws, j, dtype=np.intp)
        diffs = st.cap - st.soc[ws:j]
        head_charge = np.minimum.accumulate(diffs[::-1])[::-1]   # min over [k, j-1]
        valid = head_charge > _EPS
        if valid.any():
            cost, dc = _source_arrays(st, k_idx)
            dc = np.minimum(dc, head_charge)
            viable = valid & (cost < hurdle) & (dc > _EPS)
            if viable.any():
                rank = np.where(viable, cost + 1e-12 * (j - k_idx), np.inf)
                ii = int(np.argmin(rank))
                best_k, best_cost, best_dc, best_dist = (
                    int(k_idx[ii]), float(cost[ii]), float(dc[ii]), j - int(k_idx[ii])
                )

    # ---- Forward window: nearest later interval that is empty blocks carry ----
    if j < n - 1:
        we = _forward_window_end(st, j)               # largest reachable index
        k_idx_f = np.arange(j + 1, we + 1, dtype=np.intp)
        # head_use[k] = min over [j, k-1] of (soc - soc_min)
        seg = st.soc[j:we] - st.soc_min               # carried intervals [j, we-1]
        head_use = np.minimum.accumulate(seg)         # aligns to k = j+1 .. we
        valid_f = head_use > _EPS
        if valid_f.any():
            cost_f, dc_f = _source_arrays(st, k_idx_f)
            dc_f = np.minimum(dc_f, head_use)
            viable_f = valid_f & (cost_f < hurdle) & (dc_f > _EPS)
            if viable_f.any():
                rank_f = np.where(viable_f, cost_f + 1e-12 * (k_idx_f - j), np.inf)
                tt = int(np.argmin(rank_f))
                ck, cdc, kk = float(cost_f[tt]), float(dc_f[tt]), int(k_idx_f[tt])
                dist = kk - j
                if best_k < 0 or (ck < best_cost - 1e-12) or (
                    abs(ck - best_cost) <= 1e-12 and dist < best_dist
                ):
                    best_k, best_cost, best_dc = kk, ck, cdc

    if best_k < 0:
        return None
    return best_k, best_cost, best_dc


# Chunk size for window discovery. Battery cycling means a full/empty barrier is
# almost always within a day or two; the chunked scan keeps per-sink work bounded
# by the actual window rather than the whole series.
_CHUNK = 512


def _backward_window_start(st: _State, j: int) -> int:
    """Smallest index ws such that the battery is not full anywhere in [ws, j-1].
    The nearest earlier full interval (soc >= cap - eps) blocks carrying charge
    past it, so ws = that index + 1 (or 0 if none)."""
    cap = st.cap
    start = j
    while start > 0:
        new_start = max(0, start - _CHUNK)
        seg = st.soc[new_start:start]
        full = np.nonzero(seg >= cap - _EPS)[0]
        if full.size:
            return new_start + int(full[-1]) + 1
        start = new_start
    return 0


def _forward_window_end(st: _State, j: int) -> int:
    """Largest index we such that the battery is not empty anywhere in [j, we-1]
    ... i.e. the carry of discharge is not blocked before we. The nearest later
    empty interval (soc <= soc_min + eps) blocks carrying discharge past it, so
    we = that index (carry covers up to it) ... we return the last reachable
    source index."""
    n = st.soc.shape[0]
    floor = st.soc_min
    start = j + 1
    while start < n:
        new_end = min(n, start + _CHUNK)
        seg = st.soc[start:new_end]
        empty = np.nonzero(seg <= floor + _EPS)[0]
        if empty.size:
            # first empty at index e; sources up to e are reachable (carry on
            # [j, e-1] still has headroom at e-1), e itself is the barrier.
            return start + int(empty[0])
        start = new_end
    return n - 1


# ---------------------------------------------------------------------------
# Shift application
# ---------------------------------------------------------------------------

def _apply_shift(st: _State, k: int, j: int, sink_kind: int, is_redirect: bool,
                 shift_kwh: float) -> None:
    """Apply a shift of shift_kwh (DC) from source k to sink j.

    SOC continuity and the four SiteTraces invariants are preserved exactly.
    The carry slice handles the intervening SOC; the source raises net DC charge
    at k by shift_kwh; the sink lowers net DC at j by shift_kwh. For redirect
    sinks (is_redirect=True) the DC removed at j comes from cancelling solar
    charge (sol_to_bat -= redirect_ac where redirect_ac*eta_c == shift_kwh), not
    from battery discharge, so no eta_d loss is incurred at the sink.
    """
    if k < j:
        st.soc[k:j] += shift_kwh
    else:
        st.soc[j:k] -= shift_kwh

    _apply_source(st, k, shift_kwh)
    _apply_sink(st, j, sink_kind, is_redirect, shift_kwh)


def _apply_source(st: _State, k: int, shift_kwh: float) -> None:
    """Raise net DC charge at interval k by shift_kwh, classifying the source the
    same way the search did (curtail -> exported solar -> grid).

    Negative import price override: when p_imp[k] < 0, always add grid_to_bat
    regardless of whether curtail or sol_to_grid is present. This preserves the
    neg-price-aware SC baseline (curtailed solar stays curtailed) and correctly
    earns the negative-price revenue by increasing imports.
    """
    eta_c, eta_d = st.eta_c, st.eta_d
    if st.p_imp[k] < -_EPS:
        # Negative import price: add grid charge (don't un-curtail solar).
        st.grid_to_bat[k] += shift_kwh / eta_c
    elif st.curtail[k] > _EPS:
        # Capture curtailed solar (cost 0): move AC solar into the battery.
        ac = shift_kwh / eta_c
        ac = min(ac, st.curtail[k])
        st.curtail[k] -= ac
        st.sol_to_bat[k] += ac
        # DC stored = ac * eta_c == shift_kwh (ac bounded by dc headroom upstream)
    elif st.sol_to_grid[k] > _EPS:
        # Capture exported solar (cost p_exp): redirect export into the battery.
        ac = shift_kwh / eta_c
        st.sol_to_grid[k] -= ac
        st.sol_to_bat[k] += ac
    else:
        # Grid charge (cost p_imp): cancel discharge-to-home before adding charge.
        cancel_ac = min(st.bat_to_home[k], shift_kwh * eta_d)
        st.bat_to_home[k] -= cancel_ac
        st.grid_to_home[k] += cancel_ac
        dc_from_cancel = cancel_ac / eta_d
        dc_remaining = shift_kwh - dc_from_cancel
        st.grid_to_bat[k] += dc_remaining / eta_c


def _apply_sink(st: _State, j: int, sink_kind: int, is_redirect: bool,
                shift_kwh: float) -> None:
    """Lower net DC at interval j by shift_kwh.

    Redirect mode (is_redirect=True): cancel solar charge → export directly.
      redirect_ac = shift_kwh / eta_c AC moved from sol_to_bat to sol_to_grid.
      DC removed = redirect_ac * eta_c = shift_kwh. No battery discharge needed.
    Discharge mode (is_redirect=False): deliver shift_kwh * eta_d AC via battery.
    """
    if sink_kind == _AVOID:
        st.grid_to_home[j] -= shift_kwh * st.eta_d
        st.bat_to_home[j] += shift_kwh * st.eta_d
    elif is_redirect:
        redirect_ac = shift_kwh / st.eta_c
        st.sol_to_bat[j] -= redirect_ac
        st.sol_to_grid[j] += redirect_ac
    else:
        st.bat_to_grid[j] += shift_kwh * st.eta_d


# ---------------------------------------------------------------------------
# Output assembly + checks
# ---------------------------------------------------------------------------

def _assemble_traces(st: _State, site_id: str, index, soc_final_kwh: float) -> SiteTraces:
    # sol_to_home is NOT clipped: it can be legitimately negative when solar_kwh
    # is negative (overnight inverter parasitic draw). The LP and self_consumption
    # both preserve this; clipping here would break the solar and load invariants.
    sol_to_home = st.sol_to_home
    sol_to_bat = _clip_noise(st.sol_to_bat, 'sol_to_bat', site_id)
    sol_to_grid = _clip_noise(st.sol_to_grid, 'sol_to_grid', site_id)
    bat_to_home = _clip_noise(st.bat_to_home, 'bat_to_home', site_id)
    grid_to_home = _clip_noise(st.grid_to_home, 'grid_to_home', site_id)
    grid_to_bat = _clip_noise(st.grid_to_bat, 'grid_to_bat', site_id)
    bat_to_grid = _clip_noise(st.bat_to_grid, 'bat_to_grid', site_id)
    curtail = _clip_noise(st.curtail, 'curtail', site_id)
    soc = st.soc

    _check_invariants(
        st, sol_to_home, sol_to_bat, sol_to_grid, bat_to_home,
        grid_to_home, grid_to_bat, bat_to_grid, curtail, soc,
        site_id, soc_final_kwh,
    )

    import_kwh = grid_to_home + grid_to_bat
    export_kwh = sol_to_grid + bat_to_grid

    def _s(arr, name):
        return pd.Series(arr.copy(), index=index, name=name)

    return SiteTraces(
        site_id=site_id,
        import_kwh=_s(import_kwh, 'import_kwh'),
        export_kwh=_s(export_kwh, 'export_kwh'),
        battery_soc_kwh=_s(soc, 'battery_soc_kwh'),
        sol_to_home_kwh=_s(sol_to_home, 'sol_to_home_kwh'),
        sol_to_bat_kwh=_s(sol_to_bat, 'sol_to_bat_kwh'),
        sol_to_grid_kwh=_s(sol_to_grid, 'sol_to_grid_kwh'),
        bat_to_home_kwh=_s(bat_to_home, 'bat_to_home_kwh'),
        grid_to_home_kwh=_s(grid_to_home, 'grid_to_home_kwh'),
        grid_to_bat_kwh=_s(grid_to_bat, 'grid_to_bat_kwh'),
        bat_to_grid_kwh=_s(bat_to_grid, 'bat_to_grid_kwh'),
        curtail_kwh=_s(curtail, 'curtail_kwh'),
    )


def _clip_noise(arr: np.ndarray, name: str, site_id: str) -> np.ndarray:
    neg = arr < 0
    if neg.any():
        worst = float(-arr[neg].min())
        if worst > _WARN_TOL:
            warnings.warn(
                f"custom [{site_id}] {name}: {int(neg.sum())} value(s) below 0 "
                f"with max magnitude {worst:.2e} (>{_WARN_TOL:.0e}); clipping.",
                stacklevel=3,
            )
        arr = np.clip(arr, 0.0, None)
    return arr


def _check_invariants(
    st, sol_to_home, sol_to_bat, sol_to_grid, bat_to_home,
    grid_to_home, grid_to_bat, bat_to_grid, curtail, soc,
    site_id, soc_final_kwh,
) -> None:
    """Verify the SiteTraces invariants, SOC continuity, bounds, and boundary
    SOC. Warns (rather than raises) on residuals so a usable result is still
    returned, matching the LP's tolerant post-solve checks."""
    eta_c, eta_d = st.eta_c, st.eta_d

    # Solar split: solar = sol_to_home + sol_to_bat + sol_to_grid + curtail
    res_solar = st.solar - (sol_to_home + sol_to_bat + sol_to_grid + curtail)
    # Load split: load = sol_to_home + bat_to_home + grid_to_home
    res_load = st.load - (sol_to_home + bat_to_home + grid_to_home)

    # SOC continuity: soc[t] - soc[t-1] = eta_c*charge_ac - discharge_ac/eta_d
    # The baseline (and hence this dispatch) starts from the reserve floor.
    charge_ac = sol_to_bat + grid_to_bat
    discharge_ac = bat_to_home + bat_to_grid
    delta = eta_c * charge_ac - discharge_ac / eta_d
    cont = np.empty_like(soc)
    cont[0] = soc[0] - (st.soc_min + delta[0])
    cont[1:] = soc[1:] - soc[:-1] - delta[1:]

    for label, res in (('solar split', res_solar), ('load split', res_load),
                       ('soc continuity', cont)):
        worst = float(np.max(np.abs(res))) if res.size else 0.0
        if worst > _WARN_TOL:
            warnings.warn(
                f"custom [{site_id}] {label} residual max {worst:.2e} "
                f"(>{_WARN_TOL:.0e}).", stacklevel=3,
            )

    # SOC bounds
    if (soc < st.soc_min - _WARN_TOL).any() or (soc > st.cap + _WARN_TOL).any():
        lo = float(soc.min()); hi = float(soc.max())
        warnings.warn(
            f"custom [{site_id}] SOC out of bounds: min {lo:.4f}, max {hi:.4f}, "
            f"allowed [{st.soc_min:.4f}, {st.cap:.4f}].", stacklevel=3,
        )

    # Terminal SOC preserved
    if abs(soc[-1] - soc_final_kwh) > _SOC_TOL:
        warnings.warn(
            f"custom [{site_id}] terminal SOC {soc[-1]:.6f} != soc_final "
            f"{soc_final_kwh:.6f}.", stacklevel=3,
        )

    # Simultaneous flows (degeneracy / refinement gaps)
    simul_cd = (charge_ac > _SIMUL_TOL) & (discharge_ac > _SIMUL_TOL)
    if simul_cd.any():
        warnings.warn(
            f"custom [{site_id}]: {int(simul_cd.sum())} interval(s) with "
            f"simultaneous charge and discharge.", stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_inputs(
    solar_kwh, load_kwh, price_imp, price_exp,
    bspec, soc_init, soc_final, opt_factor,
) -> None:
    if bspec is None:
        raise ValueError("perfect_foresight_custom requires a battery_spec.")

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
    for name, s in (('solar_kwh', solar_kwh), ('load_kwh', load_kwh),
                    ('price_imp', price_imp), ('price_exp', price_exp)):
        if s.isna().any():
            raise ValueError(f"{name} contains NaN values.")
    if (load_kwh < 0).any():
        raise ValueError("load_kwh contains negative values.")

    p_imp_arr = price_imp.to_numpy()
    p_exp_arr = price_exp.to_numpy()
    violation = p_exp_arr > p_imp_arr
    if violation.any():
        first = int(violation.argmax())
        raise ValueError(
            f"price_exp exceeds price_imp at {int(violation.sum())} interval(s). "
            f"First at index {first}: p_imp={p_imp_arr[first]:.6f}, "
            f"p_exp={p_exp_arr[first]:.6f}."
        )

    cap = bspec.capacity_kwh
    soc_min = (bspec.backup_reserve_pct / 100.0) * cap
    for name, val in (('soc_init_kwh', soc_init), ('soc_final_kwh', soc_final)):
        if val < soc_min - 1e-9 or val > cap + 1e-9:
            raise ValueError(
                f"{name}={val:.6f} is outside battery bounds "
                f"[{soc_min:.6f}, {cap:.6f}]."
            )

    if not (0.0 < opt_factor <= 1.0):
        raise ValueError(f"opt_factor must be in (0, 1], got {opt_factor}.")


def _check_baseline_boundaries(base, soc_min, soc_init, soc_final,
                               external: bool = False) -> None:
    """Verify the baseline's boundary SOCs are consistent with soc_init/soc_final.

    When external=False (internal self_consumption baseline): soc_init must
    equal soc_min, since self_consumption always starts at the reserve floor.

    When external=True (caller-supplied sc_traces): soc_init must equal the
    first value of the provided baseline, which may differ from soc_min (e.g.
    when the greedy is seeded from a previous pass's output).

    In both cases soc_final must match the baseline's end-of-period SOC.
    """
    soc_arr  = base.battery_soc_kwh.to_numpy()
    base_end = float(soc_arr[-1])

    if external:
        base_start = float(soc_arr[0])
        if abs(soc_init - base_start) > _SOC_TOL:
            raise ValueError(
                f"soc_init_kwh ({soc_init:.6f}) must equal the first SOC of the "
                f"supplied sc_traces ({base_start:.6f})."
            )
    else:
        if abs(soc_init - soc_min) > _SOC_TOL:
            raise ValueError(
                f"soc_init_kwh ({soc_init:.6f}) must equal the reserve floor "
                f"({soc_min:.6f}); the internal self_consumption baseline starts "
                f"there. Pass soc_init = soc_min, or supply a precomputed baseline "
                f"via the sc_traces argument."
            )

    if abs(soc_final - base_end) > _SOC_TOL:
        raise ValueError(
            f"soc_final_kwh ({soc_final:.6f}) must equal the baseline "
            f"end-of-period SOC ({base_end:.6f}). Derive soc_final from the "
            f"same baseline used as input."
        )