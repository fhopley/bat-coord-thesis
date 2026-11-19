"""vpp_sc_event -- VPP battery dispatch with self-consumption between events.

Models home battery dispatch for a site enrolled in a VPP with clean
regime isolation:

  Between events: deterministic self-consumption, computed forward from
      whatever SOC the previous event left behind.

  Within each event window: a short LP finds the operator-optimal forced
      flows (force-charge and force-discharge) given natural-flow bounds
      derived from what self-consumption would have done over the same
      window. The LP has no visibility of any interval outside the current
      event window and no terminal SOC constraint.

The SOC at the end of each event is the fixed entry point for the next
self-consumption segment. The two regimes are isolated by construction:
the LP cannot influence non-event dispatch and the non-event dispatch
cannot influence which event intervals become profitable.

See docs/vpp_sc_event_spec.md for the full problem statement,
bound construction, and invariant proofs.
"""
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linprog

from model.core.scenario import BatterySpec
from model.core.site_result import SiteTraces, VppTraces
from model.core.tariff import RetailTariff
from model.components.self_consumption import run_sc_with_battery


# Numerical tolerances
_CLIP_TOL  = 1e-9   # floor for non-negativity clip
_WARN_TOL  = 1e-6   # threshold for clipping warnings
_SIMUL_TOL = 1e-6   # threshold for simultaneous-flow warnings


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def vpp_sc_event(
    site_id: str,
    solar_kwh: pd.Series,
    load_kwh: pd.Series,
    op_price_imp: pd.Series,
    op_price_exp: pd.Series,
    vpp_event: np.ndarray,
    battery_spec: BatterySpec,
    soc_init_kwh: float,
    vpp_exp_reward: float,
    vpp_imp_reward: float,
    vpp_kwh_limit: float,
    vpp_limit_applies_to: str,
    export_limit_kwh_per_interval: float = np.inf,
    interval_hours: float = 5 / 60,
    p_curtail: float | None = None,
    gap_strategy: str = 'self_consumption',
    retail_tariff: RetailTariff | None = None,
    prices_mwh: pd.Series | None = None,
    price_imp: pd.Series | None = None,
    price_exp: pd.Series | None = None,
    min_export_price_per_kwh: float = 0.0,
    no_grid_charge_mask: np.ndarray | None = None,
    apply_export_order_nudge: bool = False,
    state: str | None = None,
) -> tuple[SiteTraces, VppTraces]:
    """Compute VPP battery dispatch for a single site.

    Args:
        site_id: Passed through to output SiteTraces and VppTraces.
        solar_kwh: Gross solar generation per interval, kWh. Full year.
        load_kwh: Gross household load per interval, kWh. Full year.
        op_price_imp: Operator import price, $/kWh. Full year.
        op_price_exp: Operator export price, $/kWh. Full year.
        vpp_event: Bool array length T; True = operator controls interval.
            Produced by classify_vpp_events.
        battery_spec: Battery specification.
        soc_init_kwh: Battery DC SOC at period start. Typically soc_min
            (the reserve floor), matching self_consumption convention.
        vpp_exp_reward: Household reward per kWh of forced export, $/kWh.
        vpp_imp_reward: Household reward per kWh of forced import, $/kWh.
        vpp_kwh_limit: Annual VPP energy cap (kWh).
        vpp_limit_applies_to: 'export_only' or 'import_and_export'.
        export_limit_kwh_per_interval: Per-interval site export cap, kWh.
            np.inf (default) means unconstrained.
        interval_hours: Interval length in hours. Default 5/60 (5-minute).
        p_curtail: Curtailment penalty for event windows, $/kWh. None
            (default) disables curtailment. 0 allows curtailment at zero
            cost. Positive values price each curtailed kWh in the objective.
            Must be >= 0 when provided. Dispatch-shaping only.
        gap_strategy: Between-event dispatch strategy.
            'self_consumption' (default): gaps filled by SC. Output is
            bit-identical to the original component.
            'tou_advanced': gaps filled by perfect_foresight_lp_advanced
            on a rolling horizon. Requires retail_tariff, prices_mwh,
            price_imp, and price_exp.
        retail_tariff: Household retail tariff. Required for tou_advanced.
        prices_mwh: Full-year wholesale prices, $/MWh. Required for
            tou_advanced (used for zero-import penalty and block tier
            accumulated usage computation).
        price_imp: Full-year household import price, $/kWh (from
            build_lp_price_vectors_advanced). Required for tou_advanced.
        price_exp: Full-year household export price, $/kWh. Required for
            tou_advanced.
        min_export_price_per_kwh: Minimum operator export price ($/kWh) at
            which forced export is permitted within event windows. At
            intervals where op_exp[t] < this value, f_exp[t] is pinned to
            zero. Default 0.0 (no restriction). Resolved once per scenario by
            orchestrate.resolve_min_export_price from VppConfig
            (min_export_price_mode and min_export_price_dollars_per_mwh).
        no_grid_charge_mask: Optional full-year boolean array (length T), True
            at intervals where the operator may not charge the battery from the
            grid. On masked intervals inside an event window the event LP adds a
            row charge[t] + curtail[t] <= max(0, solar[t] - load[t]), keeping
            charging within the solar surplus so no grid energy reaches the
            battery. None (default) disables the restriction and reproduces the
            original behaviour. Built once by the orchestrator from
            VppConfig.no_grid_charge_windows.
        apply_export_order_nudge: When True, pass apply_export_order_nudge=True
            to perfect_foresight_lp_advanced in each TOU gap call, biasing the
            LP toward charging with solar first and exporting the surplus.
            Only meaningful when gap_strategy='tou_advanced'. Default False
            (see perfect_foresight_lp_advanced for performance notes).
        state: The site's actual location (scenario.state). Required when
            gap_strategy='tou_advanced' -- used to resolve local time for
            the TOU gap's block-tier carryover accounting (see
            core/localtime.py). Not needed for gap_strategy='self_consumption'
            since that path has no clock-driven logic of its own.

    Returns:
        (SiteTraces, VppTraces) for the full year. SiteTraces has the same
        12-field shape as self_consumption output. VppTraces carries the
        natural/forced split; gap segments have f_imp = f_exp = 0.

    Raises:
        ValueError: on input validation failure or missing tou_advanced args.
        RuntimeError: if HiGHS does not find an optimal solution for any
            event window (should not occur; treated as a bug).
    """
    _validate_inputs(
        solar_kwh, load_kwh, op_price_imp, op_price_exp,
        vpp_event, battery_spec, soc_init_kwh,
        vpp_exp_reward, vpp_imp_reward, vpp_limit_applies_to,
        p_curtail,
    )

    if gap_strategy not in ('self_consumption', 'tou_advanced'):
        raise ValueError(
            f"gap_strategy must be 'self_consumption' or 'tou_advanced', "
            f"got {gap_strategy!r}"
        )
    if gap_strategy == 'tou_advanced':
        missing = [name for name, val in [
            ('retail_tariff', retail_tariff),
            ('prices_mwh',    prices_mwh),
            ('price_imp',     price_imp),
            ('price_exp',     price_exp),
            ('state',         state),
        ] if val is None]
        if missing:
            raise ValueError(
                f"gap_strategy='tou_advanced' requires: {', '.join(missing)}"
            )

    index    = solar_kwh.index
    T        = len(index)
    solar    = solar_kwh.to_numpy(dtype=float, copy=False)
    load     = load_kwh.to_numpy(dtype=float, copy=False)
    op_imp   = op_price_imp.to_numpy(dtype=float, copy=False)
    op_exp   = op_price_exp.to_numpy(dtype=float, copy=False)
    event    = vpp_event.astype(bool)

    # No-grid-charge mask: normalise to a length-T bool array (all-False when
    # the feature is disabled, so the per-window slice is always valid and adds
    # no LP rows).
    if no_grid_charge_mask is None:
        ngc_mask = np.zeros(T, dtype=bool)
    else:
        ngc_mask = np.asarray(no_grid_charge_mask, dtype=bool)
        if ngc_mask.shape != (T,):
            raise ValueError(
                f"no_grid_charge_mask length ({ngc_mask.shape}) does not match "
                f"solar_kwh length ({T})."
            )

    bspec        = battery_spec
    capacity     = bspec.capacity_kwh
    soc_min      = (bspec.backup_reserve_pct / 100.0) * capacity
    power_kwh    = bspec.power_kw * interval_hours
    eta_c        = bspec.charge_efficiency
    eta_d        = bspec.discharge_efficiency

    T_24h = round(24.0 / interval_hours)   # intervals in 24 hours

    # Identify contiguous event windows
    windows = _find_event_windows(event)

    # Segment accumulators
    all_traces_data: list[dict] = []   # list of flow-array dicts
    all_soc:         list[np.ndarray] = []
    all_n_imp:       list[np.ndarray] = []
    all_f_imp:       list[np.ndarray] = []
    all_n_exp:       list[np.ndarray] = []
    all_f_exp:       list[np.ndarray] = []

    soc_entry        = soc_init_kwh
    remaining_cap    = vpp_kwh_limit
    prev_window_end  = 0   # exclusive end of the previously processed segment

    for win_start, win_end in windows:
        # ------------------------------------------------------------------ #
        # Step 1: Gap + event SC run                                          #
        # Covers gap (prev_window_end..win_start) + event window.            #
        # SC is always run here -- the SC SOC is needed for event LP bounds  #
        # regardless of gap_strategy. For tou_advanced, the gap dispatch is  #
        # replaced by the TOU LP result; only the SC SOC over the event      #
        # window is used from this run.                                       #
        # ------------------------------------------------------------------ #
        sc_soc, sc_flows = run_sc_with_battery(
            solar[prev_window_end:win_end],
            load[prev_window_end:win_end],
            bspec, soc_entry, interval_hours,
        )

        gap_len = win_start - prev_window_end

        if gap_len > 0:
            if gap_strategy == 'tou_advanced':
                # Fill the gap with TOU-optimised dispatch.
                tou_soc, tou_flows = _run_tou_gap(
                    site_id=site_id,
                    gap_start=prev_window_end,
                    gap_end=win_start,
                    win_end=win_end,
                    T=T,
                    T_24h=T_24h,
                    solar_kwh=solar_kwh,
                    load_kwh=load_kwh,
                    soc_entry=soc_entry,
                    bspec=bspec,
                    retail_tariff=retail_tariff,
                    prices_mwh=prices_mwh,
                    price_imp=price_imp,
                    price_exp=price_exp,
                    interval_hours=interval_hours,
                    all_traces_data=all_traces_data,
                    all_soc=all_soc,
                    index=index,
                    apply_export_order_nudge=apply_export_order_nudge,
                    state=state,
                )
                # tou_soc and tou_flows cover [gap_start, gap_end)
                # soc_entry for the event is the TOU LP exit SOC
                lp_soc_entry_for_event = float(tou_soc[-1])
                _append_tou_segment(
                    all_traces_data, all_soc,
                    all_n_imp, all_f_imp, all_n_exp, all_f_exp,
                    tou_flows, tou_soc, gap_len,
                )
            else:
                # Fill the gap with SC dispatch (default).
                lp_soc_entry_for_event = float(sc_soc[gap_len - 1])
                _append_sc_segment(
                    all_traces_data, all_soc,
                    all_n_imp, all_f_imp, all_n_exp, all_f_exp,
                    sc_flows, sc_soc, 0, gap_len,
                )
        else:
            lp_soc_entry_for_event = soc_entry

        # SC SOC over the event window (relative offset within sc arrays)
        ew_offset = gap_len        # start of event window within sc arrays
        ew_len    = win_end - win_start

        # soc_prev[t] for bound construction always comes from SC SOC.
        # The entry SOC for the SC-derived bounds is always sc_soc at the
        # event boundary, not the TOU LP exit SOC.
        sc_soc_ew = sc_soc[ew_offset:ew_offset + ew_len]
        sc_entry_for_bounds = soc_entry if gap_len == 0 else float(sc_soc[ew_offset - 1])
        sc_soc_prev = _make_soc_prev(sc_soc_ew, sc_entry_for_bounds)

        # ------------------------------------------------------------------ #
        # Step 2: Bounds from SC SOC, computed once and never updated        #
        # ------------------------------------------------------------------ #
        solar_ew  = solar[win_start:win_end]
        load_ew   = load[win_start:win_end]
        op_imp_ew = op_imp[win_start:win_end]
        op_exp_ew = op_exp[win_start:win_end]
        ngc_ew    = ngc_mask[win_start:win_end]

        sc_n_imp_total, n_exp_hi = _compute_bounds(
            solar_ew, load_ew, sc_soc_prev, soc_min, capacity, eta_c, eta_d, power_kwh)

        # ------------------------------------------------------------------ #
        # Step 3: LP solve (single solve; no iteration needed)               #
        # ------------------------------------------------------------------ #
        # LP entry SOC comes from the gap strategy (TOU LP or SC).
        lp_soc_entry = lp_soc_entry_for_event

        x = _solve_event_lp(
            solar_ew, load_ew, op_imp_ew, op_exp_ew,
            sc_n_imp_total, n_exp_hi,
            bspec, lp_soc_entry,
            vpp_exp_reward, vpp_imp_reward,
            remaining_cap, vpp_limit_applies_to,
            export_limit_kwh_per_interval, interval_hours,
            site_id, win_start, p_curtail,
            min_export_price_per_kwh,
            ngc_ew,
        )

        # ------------------------------------------------------------------ #
        # Step 4: extract event segment results (or SC fallback if infeasible)#
        # ------------------------------------------------------------------ #
        if x is None:
            # LP infeasible: fall back to SC for this event window.
            # With SC-fixed bounds this should never occur -- the SC dispatch
            # is always a feasible assignment. If this fires, treat as a bug.
            # EOV_SKIP warning already emitted by _solve_event_lp.
            _append_sc_segment(
                all_traces_data, all_soc,
                all_n_imp, all_f_imp, all_n_exp, all_f_exp,
                sc_flows, sc_soc, ew_offset, ew_len,
            )
            soc_entry       = float(sc_soc[ew_offset + ew_len - 1])
            prev_window_end = win_end
            continue
        Te = ew_len
        charge    = _clip(x[0*Te:1*Te],   'charge',    site_id)
        discharge = _clip(x[1*Te:2*Te],   'discharge', site_id)
        n_imp     = _clip(x[2*Te:3*Te],   'n_imp',     site_id)
        f_imp     = _clip(x[3*Te:4*Te],   'f_imp',     site_id)
        n_exp     = _clip(x[4*Te:5*Te],   'n_exp',     site_id)
        f_exp     = _clip(x[5*Te:6*Te],   'f_exp',     site_id)
        lp_soc    = x[6*Te:7*Te]
        curtail   = _clip(x[7*Te:8*Te],   'curtail',   site_id)

        _check_simultaneous_flows(n_imp, f_imp, n_exp, f_exp,
                                  charge, discharge, site_id, win_start)

        ew_flows = _reconstruct_flows(
            solar_ew, load_ew, charge, discharge, curtail,
            n_imp + f_imp, n_exp + f_exp,
        )
        _check_no_grid_charge(
            ew_flows['grid_to_bat'], charge, curtail, solar_ew, load_ew,
            ngc_ew, site_id, win_start,
        )
        all_traces_data.append(ew_flows)
        all_soc.append(lp_soc)
        all_n_imp.append(n_imp)
        all_f_imp.append(f_imp)
        all_n_exp.append(n_exp)
        all_f_exp.append(f_exp)

        # ------------------------------------------------------------------ #
        # Step 5: cap accounting and advance                                  #
        # ------------------------------------------------------------------ #
        if vpp_limit_applies_to == 'export_only':
            used = float(f_exp.sum())
        else:
            used = float(f_imp.sum() + f_exp.sum())
        remaining_cap = max(0.0, remaining_cap - used)

        soc_entry       = float(lp_soc[-1])
        prev_window_end = win_end

    # ---------------------------------------------------------------------- #
    # Final gap: [last_event_end, T)                                          #
    # ---------------------------------------------------------------------- #
    if prev_window_end < T:
        if gap_strategy == 'tou_advanced':
            tou_soc_tail, tou_flows_tail = _run_tou_gap(
                site_id=site_id,
                gap_start=prev_window_end,
                gap_end=T,
                win_end=T,          # no following event; horizon = gap only
                T=T,
                T_24h=T_24h,
                solar_kwh=solar_kwh,
                load_kwh=load_kwh,
                soc_entry=soc_entry,
                bspec=bspec,
                retail_tariff=retail_tariff,
                prices_mwh=prices_mwh,
                price_imp=price_imp,
                price_exp=price_exp,
                interval_hours=interval_hours,
                all_traces_data=all_traces_data,
                all_soc=all_soc,
                index=index,
                apply_export_order_nudge=apply_export_order_nudge,
                state=state,
            )
            _append_tou_segment(
                all_traces_data, all_soc,
                all_n_imp, all_f_imp, all_n_exp, all_f_exp,
                tou_flows_tail, tou_soc_tail, T - prev_window_end,
            )
        else:
            sc_soc_tail, sc_flows_tail = run_sc_with_battery(
                solar[prev_window_end:T],
                load[prev_window_end:T],
                bspec, soc_entry, interval_hours,
            )
            _append_sc_segment(
                all_traces_data, all_soc,
                all_n_imp, all_f_imp, all_n_exp, all_f_exp,
                sc_flows_tail, sc_soc_tail, 0, T - prev_window_end,
            )

    # ---------------------------------------------------------------------- #
    # Assemble full-year SiteTraces and VppTraces                             #
    # ---------------------------------------------------------------------- #
    def _cat(arrays: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(arrays)

    def _s(arr: np.ndarray, name: str) -> pd.Series:
        return pd.Series(arr, index=index, name=name)

    s2h = _cat([d['sol_to_home'] for d in all_traces_data])
    s2b = _cat([d['sol_to_bat']  for d in all_traces_data])
    s2g = _cat([d['sol_to_grid'] for d in all_traces_data])
    b2h = _cat([d['bat_to_home'] for d in all_traces_data])
    g2h = _cat([d['grid_to_home'] for d in all_traces_data])
    g2b = _cat([d['grid_to_bat']  for d in all_traces_data])
    cur = _cat([d['curtail']      for d in all_traces_data])
    b2g = _cat([d['bat_to_grid']  for d in all_traces_data])
    soc = _cat(all_soc)
    n_imp_full = _cat(all_n_imp)
    f_imp_full = _cat(all_f_imp)
    n_exp_full = _cat(all_n_exp)
    f_exp_full = _cat(all_f_exp)

    # Use LP-derived import/export as canonical so VppTraces invariants hold
    # exactly. For event segments these come from the LP variables; for SC
    # (gap) segments they equal g2h+g2b by construction (f_imp=f_exp=0).
    import_kwh = _cat([d['import_kwh'] for d in all_traces_data])
    export_kwh = _cat([d['export_kwh'] for d in all_traces_data])

    traces = SiteTraces(
        site_id=site_id,
        import_kwh=_s(import_kwh,  'import_kwh'),
        export_kwh=_s(export_kwh,  'export_kwh'),
        battery_soc_kwh=_s(soc,    'battery_soc_kwh'),
        sol_to_home_kwh=_s(s2h,    'sol_to_home_kwh'),
        sol_to_bat_kwh=_s(s2b,     'sol_to_bat_kwh'),
        sol_to_grid_kwh=_s(s2g,    'sol_to_grid_kwh'),
        bat_to_home_kwh=_s(b2h,    'bat_to_home_kwh'),
        grid_to_home_kwh=_s(g2h,   'grid_to_home_kwh'),
        grid_to_bat_kwh=_s(g2b,    'grid_to_bat_kwh'),
        bat_to_grid_kwh=_s(b2g,    'bat_to_grid_kwh'),
        curtail_kwh=_s(cur,        'curtail_kwh'),
    )
    vpp_traces = VppTraces(
        site_id=site_id,
        n_imp_kwh=_s(n_imp_full, 'n_imp_kwh'),
        f_imp_kwh=_s(f_imp_full, 'f_imp_kwh'),
        n_exp_kwh=_s(n_exp_full, 'n_exp_kwh'),
        f_exp_kwh=_s(f_exp_full, 'f_exp_kwh'),
    )
    return traces, vpp_traces


# ---------------------------------------------------------------------------
# TOU gap helper
# ---------------------------------------------------------------------------

def _run_tou_gap(
    site_id: str,
    gap_start: int,
    gap_end: int,
    win_end: int,
    T: int,
    T_24h: int,
    solar_kwh: pd.Series,
    load_kwh: pd.Series,
    soc_entry: float,
    bspec: BatterySpec,
    retail_tariff: RetailTariff,
    prices_mwh: pd.Series,
    price_imp: pd.Series,
    price_exp: pd.Series,
    interval_hours: float,
    all_traces_data: list,
    all_soc: list,
    index: pd.DatetimeIndex,
    apply_export_order_nudge: bool = False,
    state: str | None = None,
) -> tuple[np.ndarray, dict]:
    """Fill a between-event gap with TOU-optimised dispatch.

    Runs perfect_foresight_lp_advanced over a rolling horizon that extends
    24 hours past win_end (or to T, whichever comes first) to give the LP
    visibility of the post-event period. Only the gap portion
    [gap_start, gap_end) of the LP solution is returned; the event-window
    and look-ahead portions are discarded.

    The terminal SOC at the horizon end is derived from a SC sub-call over
    the same horizon, giving a physically achievable and feasibility-
    guaranteed terminal constraint.

    Block tier accumulated usage for the first local calendar day of the
    horizon is computed from the already-assembled segments in
    all_traces_data / all_soc, summing flows in each block window that fall
    on the same local calendar day (for `state`) as gap_start.

    Returns:
        tou_soc:   DC SOC array for [gap_start, gap_end), shape (gap_end - gap_start,)
        tou_flows: flow dict for [gap_start, gap_end), same keys as _append_sc_segment
    """
    from model.components.perfect_foresight_lp_advanced import (
        perfect_foresight_lp_advanced,
    )
    from model.core.prices import block_tier_masks
    from model.core.localtime import local_wall_clock

    gap_len     = gap_end - gap_start
    horizon_end = min(win_end + T_24h, T)
    horizon_len = horizon_end - gap_start

    # ------------------------------------------------------------------ #
    # Terminal SOC from SC over the full horizon                          #
    # ------------------------------------------------------------------ #
    sc_soc_horizon, _ = run_sc_with_battery(
        solar_kwh.iloc[gap_start:horizon_end].to_numpy(dtype=float),
        load_kwh.iloc[gap_start:horizon_end].to_numpy(dtype=float),
        bspec, soc_entry, interval_hours,
    )
    soc_final = float(sc_soc_horizon[-1])

    # ------------------------------------------------------------------ #
    # Block tier accumulated usage on the first calendar day              #
    # ------------------------------------------------------------------ #
    first_day_block_import_used: dict[int, float] = {}
    first_day_block_export_used: dict[int, float] = {}

    if all_traces_data and retail_tariff is not None:
        import_windows = retail_tariff.block_import_tiers or ()
        export_windows = retail_tariff.block_export_tiers or ()

        if import_windows or export_windows:
            # Current local calendar day at gap_start (interval-START
            # convention, converted to the site's real local time -- block
            # tier resets are a local-day concept, not an AEST-day one).
            gap_start_day = local_wall_clock(
                index[gap_start:gap_start + 1], state, interval_hours
            ).date[0]

            # Assemble the full-year traces built so far (covers [0, gap_start))
            assembled_len = sum(len(d['import_kwh']) for d in all_traces_data)
            if assembled_len > 0:
                assembled_imp = np.concatenate([d['import_kwh'] for d in all_traces_data])
                assembled_exp = np.concatenate([d['export_kwh'] for d in all_traces_data])
                assembled_idx = index[:assembled_len]

                # Vectorised day mask: derive local calendar dates for every
                # already-assembled interval in one call, then compare with
                # a single == broadcast against gap_start's local date.
                assembled_days = local_wall_clock(
                    assembled_idx, state, interval_hours
                ).date
                day_mask = (assembled_days == gap_start_day)

                # Only worth computing window masks if any interval is on this day.
                # Tier membership leaves out flow that was critical on the
                # window's side by the settled price, as billing does.
                if day_mask.any():
                    imp_masks, exp_masks = block_tier_masks(
                        assembled_idx, retail_tariff, state, interval_hours,
                        None if prices_mwh is None
                        else prices_mwh.iloc[:assembled_len])
                    for w_idx, win_mask in enumerate(imp_masks):
                        used = float(assembled_imp[win_mask & day_mask].sum())
                        if used > 0:
                            first_day_block_import_used[w_idx] = used

                    for w_idx, win_mask in enumerate(exp_masks):
                        used = float(assembled_exp[win_mask & day_mask].sum())
                        if used > 0:
                            first_day_block_export_used[w_idx] = used

    # ------------------------------------------------------------------ #
    # TOU LP over the full horizon                                         #
    # ------------------------------------------------------------------ #
    horizon_traces = perfect_foresight_lp_advanced(
        site_id=site_id,
        solar_kwh=solar_kwh.iloc[gap_start:horizon_end],
        load_kwh=load_kwh.iloc[gap_start:horizon_end],
        price_imp=price_imp.iloc[gap_start:horizon_end],
        price_exp=price_exp.iloc[gap_start:horizon_end],
        battery_spec=bspec,
        soc_init_kwh=soc_entry,
        soc_final_kwh=soc_final,
        retail_tariff=retail_tariff,
        prices_mwh=prices_mwh.iloc[gap_start:horizon_end],
        state=state,
        interval_hours=interval_hours,
        first_day_block_import_used=first_day_block_import_used or None,
        first_day_block_export_used=first_day_block_export_used or None,
        apply_export_order_nudge=apply_export_order_nudge,
    )

    # ------------------------------------------------------------------ #
    # Extract gap portion only [gap_start, gap_end)                       #
    # ------------------------------------------------------------------ #
    tou_soc = horizon_traces.battery_soc_kwh.iloc[:gap_len].to_numpy()

    tou_flows = {
        'sol_to_home':  horizon_traces.sol_to_home_kwh.iloc[:gap_len].to_numpy(),
        'sol_to_bat':   horizon_traces.sol_to_bat_kwh.iloc[:gap_len].to_numpy(),
        'sol_to_grid':  horizon_traces.sol_to_grid_kwh.iloc[:gap_len].to_numpy(),
        'bat_to_home':  horizon_traces.bat_to_home_kwh.iloc[:gap_len].to_numpy(),
        'grid_to_home': horizon_traces.grid_to_home_kwh.iloc[:gap_len].to_numpy(),
        'grid_to_bat':  horizon_traces.grid_to_bat_kwh.iloc[:gap_len].to_numpy(),
        'bat_to_grid':  horizon_traces.bat_to_grid_kwh.iloc[:gap_len].to_numpy(),
        'curtail':      horizon_traces.curtail_kwh.iloc[:gap_len].to_numpy(),
        'import_kwh':   horizon_traces.import_kwh.iloc[:gap_len].to_numpy(),
        'export_kwh':   horizon_traces.export_kwh.iloc[:gap_len].to_numpy(),
    }

    return tou_soc, tou_flows


# ---------------------------------------------------------------------------
# Event window identification
# ---------------------------------------------------------------------------

def _find_event_windows(event: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (start, end) index pairs for contiguous event blocks.

    start is inclusive, end is exclusive (Python slice convention).
    """
    windows = []
    T = len(event)
    in_window = False
    for t in range(T):
        if event[t] and not in_window:
            ws = t
            in_window = True
        elif not event[t] and in_window:
            windows.append((ws, t))
            in_window = False
    if in_window:
        windows.append((ws, T))
    return windows


# ---------------------------------------------------------------------------
# Natural-flow bound construction
# ---------------------------------------------------------------------------

def _make_soc_prev(soc_arr: np.ndarray, soc_entry: float) -> np.ndarray:
    """Return soc_prev[t]: soc_entry at t=0, soc_arr[t-1] thereafter."""
    T = len(soc_arr)
    soc_prev = np.empty(T)
    soc_prev[0] = soc_entry
    if T > 1:
        soc_prev[1:] = soc_arr[:-1]
    return soc_prev


def _compute_bounds(
    solar: np.ndarray,
    load: np.ndarray,
    soc_prev: np.ndarray,
    soc_min: float,
    capacity: float,
    eta_c: float,
    eta_d: float,
    power_kwh: float,
) -> tuple[float, np.ndarray]:
    """Compute natural-flow bounds from the SC SOC trajectory.

    Returns:
        sc_n_imp_total: Aggregate natural import budget for the event window
            (scalar, kWh). The LP's total n_imp across the window must not
            exceed this. Any import above the aggregate is forced import (f_imp)
            and earns the household VPP reward.
        n_exp_hi: Per-interval natural export ceiling (array, kWh/interval).
            n_exp[t] is bounded by n_exp_hi[t] at each interval to prevent the
            LP from laundering forced export as natural across intervals.

    Both are computed once from the SC SOC and never updated. The SC dispatch
    is always a feasible assignment for the LP, so these bounds guarantee
    feasibility (see Section 4 of the spec).
    """
    discharge_avail = np.minimum((soc_prev - soc_min) * eta_d, power_kwh)
    n_imp_hi = np.maximum(0.0, load - solar - discharge_avail)
    sc_n_imp_total = float(n_imp_hi.sum())

    n_exp_hi = np.maximum(
        0.0,
        solar - load - np.minimum((capacity - soc_prev) / eta_c, power_kwh),
    )
    return sc_n_imp_total, n_exp_hi


# ---------------------------------------------------------------------------
# Segment helpers
# ---------------------------------------------------------------------------

def _append_sc_segment(
    all_traces_data, all_soc, all_n_imp, all_f_imp, all_n_exp, all_f_exp,
    sc_flows: dict, sc_soc: np.ndarray,
    start: int, length: int,
) -> None:
    """Append a self-consumption segment (gap) to the accumulators."""
    sl = slice(start, start + length)
    seg = {k: v[sl] for k, v in sc_flows.items()}
    seg['bat_to_grid'] = np.zeros(length)
    # For SC segments: import = grid_to_home + grid_to_bat; export = sol_to_grid.
    # These match g2h+g2b exactly (no forced flows), so canonical import/export
    # is consistent with the flow decomposition.
    seg['import_kwh'] = seg['grid_to_home'] + seg['grid_to_bat']
    seg['export_kwh'] = seg['sol_to_grid']
    seg['curtail'] = seg.get('curtail', np.zeros(length))
    all_traces_data.append(seg)
    all_soc.append(sc_soc[sl])

    # For SC segments: n_imp = total import, n_exp = total export, f_imp = f_exp = 0
    all_n_imp.append(seg['import_kwh'])
    all_f_imp.append(np.zeros(length))
    all_n_exp.append(seg['export_kwh'])
    all_f_exp.append(np.zeros(length))


def _append_tou_segment(
    all_traces_data, all_soc, all_n_imp, all_f_imp, all_n_exp, all_f_exp,
    tou_flows: dict, tou_soc: np.ndarray,
    length: int,
) -> None:
    """Append a TOU-LP gap segment to the accumulators.

    Unlike _append_sc_segment, this preserves bat_to_grid from the TOU LP
    and uses the LP-produced import_kwh and export_kwh directly. The TOU LP
    can and does dispatch bat_to_grid during gap intervals (for example, into
    an evening feedin peak window). Zeroing bat_to_grid here would cause the
    flow traces to under-report exports while the SOC already reflects the
    discharges, making the SOC trajectory inconsistent with the reported flows.

    tou_flows is the dict returned by _run_tou_gap: all arrays are already
    sliced to [gap_start, gap_end) and import_kwh / export_kwh are canonical
    from the LP output (export_kwh includes bat_to_grid).
    """
    seg = {k: v[:length] for k, v in tou_flows.items()}
    all_traces_data.append(seg)
    all_soc.append(tou_soc[:length])

    # For TOU gap segments there are no forced flows; natural = total.
    all_n_imp.append(seg['import_kwh'])
    all_f_imp.append(np.zeros(length))
    all_n_exp.append(seg['export_kwh'])
    all_f_exp.append(np.zeros(length))


def _reconstruct_flows(
    solar: np.ndarray,
    load: np.ndarray,
    charge: np.ndarray,
    discharge: np.ndarray,
    curtail: np.ndarray,
    lp_import: np.ndarray,
    lp_export: np.ndarray,
) -> dict[str, np.ndarray]:
    """Reconstruct flow decomposition from solved LP variables.

    Priority: solar meets load first; solar charges battery before grid;
    battery discharges to load before grid.

    Operates on effective solar (solar - curtail) so that the solar
    invariant solar = sol_to_home + sol_to_bat + sol_to_grid + curtail
    holds. When curtailment is disabled, curtail is all-zero and this
    collapses to the original behaviour.

    lp_import and lp_export (= n_imp + f_imp and n_exp + f_exp from the LP)
    are used as the canonical import_kwh and export_kwh. The heuristic
    decomposition below allocates the flows consistently with those totals:
    grid_to_bat = charge - sol_to_bat and grid_to_home = lp_import - grid_to_bat.
    This guarantees the VppTraces invariants hold exactly, even at intervals
    where the LP has simultaneous charge+discharge (a degenerate LP tie that
    the heuristic would otherwise resolve differently from the LP variables).
    """
    solar_eff     = solar - curtail          # >= 0; curtail bounded by solar
    sol_to_home   = np.minimum(solar_eff, load)
    surplus_solar = solar_eff - sol_to_home

    sol_to_bat  = np.minimum(surplus_solar, charge)
    grid_to_bat = charge - sol_to_bat
    sol_to_grid = surplus_solar - sol_to_bat

    bat_to_home  = np.minimum(load - sol_to_home, discharge)
    bat_to_grid  = discharge - bat_to_home
    # Use LP import directly so grid_to_home + grid_to_bat == lp_import exactly.
    grid_to_home = lp_import - grid_to_bat

    return {
        'sol_to_home': sol_to_home,
        'sol_to_bat':  sol_to_bat,
        'sol_to_grid': sol_to_grid,
        'bat_to_home': bat_to_home,
        'grid_to_home': grid_to_home,
        'grid_to_bat': grid_to_bat,
        'bat_to_grid': bat_to_grid,
        'curtail':     curtail,
        'import_kwh':  lp_import,
        'export_kwh':  lp_export,
    }


# ---------------------------------------------------------------------------
# Per-event LP
# ---------------------------------------------------------------------------

def _solve_event_lp(
    solar: np.ndarray,
    load: np.ndarray,
    op_imp: np.ndarray,
    op_exp: np.ndarray,
    sc_n_imp_total: float,
    n_exp_hi: np.ndarray,
    bspec: BatterySpec,
    soc_entry: float,
    vpp_exp_reward: float,
    vpp_imp_reward: float,
    remaining_cap: float,
    vpp_limit_applies_to: str,
    export_limit: float,
    interval_hours: float,
    site_id: str,
    win_start: int,
    p_curtail: float | None,
    min_export_price_per_kwh: float = 0.0,
    no_grid_charge_ew: np.ndarray | None = None,
) -> np.ndarray | None:
    """Solve the per-event LP and return the raw solution vector x, or None
    if HiGHS cannot find an optimal solution (should not occur with SC-fixed
    bounds; treated as a bug by the caller).

    min_export_price_per_kwh: forced export (f_exp) is restricted to intervals
    where op_exp[t] >= this value. At sub-threshold intervals f_exp is pinned
    to zero, conserving the annual cap for high-value dispatch opportunities.
    """
    Te       = len(solar)
    capacity = bspec.capacity_kwh
    soc_min  = (bspec.backup_reserve_pct / 100.0) * capacity
    power_kwh = bspec.power_kw * interval_hours
    eta_c    = bspec.charge_efficiency
    eta_d    = bspec.discharge_efficiency
    c_deg    = bspec.c_deg_per_kwh

    # Variable offsets: charge, discharge, n_imp, f_imp, n_exp, f_exp, soc, curtail
    OC, OD, ONI, OFI, ONE, OFE, OS, OQ = (
        0, Te, 2*Te, 3*Te, 4*Te, 5*Te, 6*Te, 7*Te)
    N = 8 * Te

    # ------------------------------------------------------------------ #
    # Objective                                                            #
    # ------------------------------------------------------------------ #
    c = np.zeros(N)
    c[OC:OC+Te]   = c_deg
    c[OD:OD+Te]   = c_deg
    c[ONI:ONI+Te] =  op_imp
    c[OFI:OFI+Te] =  op_imp + vpp_imp_reward
    c[ONE:ONE+Te] = -op_exp
    c[OFE:OFE+Te] = -(op_exp - vpp_exp_reward)
    # curtail: penalty when enabled, zero coefficient when disabled
    if p_curtail is not None:
        c[OQ:OQ+Te] = p_curtail

    # ------------------------------------------------------------------ #
    # Equality constraints: flow balance + SOC continuity                 #
    # ------------------------------------------------------------------ #
    t  = np.arange(Te, dtype=np.intp)
    t1 = np.arange(1, Te, dtype=np.intp)

    # Flow balance: (n_imp+f_imp) - (n_exp+f_exp) - charge + discharge - curtail
    #              = load - solar
    fb_rows = np.tile(t, 7)
    fb_cols = np.concatenate([OC+t, OD+t, ONI+t, OFI+t, ONE+t, OFE+t, OQ+t])
    fb_data = np.concatenate([
        -np.ones(Te), np.ones(Te),  # charge, discharge
         np.ones(Te), np.ones(Te),  # n_imp, f_imp
        -np.ones(Te),-np.ones(Te),  # n_exp, f_exp
        -np.ones(Te),               # curtail
    ])
    b_fb = load - solar

    # SOC continuity: soc[t] - eta_c*charge[t] + (1/eta_d)*discharge[t] - soc[t-1] = 0
    sc_rows = np.concatenate([Te+t, Te+t, Te+t])
    sc_cols = np.concatenate([OC+t, OD+t, OS+t])
    sc_data = np.concatenate([
        -np.full(Te, eta_c),
         np.full(Te, 1.0 / eta_d),
         np.ones(Te),
    ])
    if Te > 1:
        sc_rows = np.concatenate([sc_rows, Te+t1])
        sc_cols = np.concatenate([sc_cols, OS+t1-1])
        sc_data = np.concatenate([sc_data, -np.ones(Te-1)])
    b_sc    = np.zeros(Te)
    b_sc[0] = soc_entry

    all_eq_rows = np.concatenate([fb_rows, sc_rows])
    all_eq_cols = np.concatenate([fb_cols, sc_cols])
    all_eq_data = np.concatenate([fb_data, sc_data])
    A_eq = sp.csc_matrix((all_eq_data, (all_eq_rows, all_eq_cols)), shape=(2*Te, N))
    b_eq = np.concatenate([b_fb, b_sc])

    # ------------------------------------------------------------------ #
    # Inequality constraints: n_imp budget + export limit + VPP cap       #
    # ------------------------------------------------------------------ #
    ub_rows_list = []; ub_cols_list = []; ub_data_list = []; b_ub_list = []
    next_row = 0

    # Aggregate natural import budget: sum(n_imp) <= sc_n_imp_total
    ub_rows_list.append(np.full(Te, next_row, dtype=np.intp))
    ub_cols_list.append(ONI + t)
    ub_data_list.append(np.ones(Te))
    b_ub_list.append(np.array([sc_n_imp_total]))
    next_row += 1

    if np.isfinite(export_limit):
        ub_rows_list.append(np.tile(next_row + np.arange(Te), 2))
        ub_cols_list.append(np.concatenate([ONE+t, OFE+t]))
        ub_data_list.append(np.ones(2*Te))
        b_ub_list.append(np.full(Te, export_limit))
        next_row += Te

    if vpp_limit_applies_to == 'export_only':
        cap_cols = OFE + t
    else:
        cap_cols = np.concatenate([OFI+t, OFE+t])
    cap_row = np.full(len(cap_cols), next_row, dtype=np.intp)
    ub_rows_list.append(cap_row)
    ub_cols_list.append(cap_cols)
    ub_data_list.append(np.ones(len(cap_cols)))
    b_ub_list.append(np.array([remaining_cap]))
    next_row += 1

    # No-grid-charge windows (one row per masked interval): keep battery
    # charging within the solar surplus so no grid energy reaches the battery.
    #   charge[i] + curtail[i] <= max(0, solar[i] - load[i])
    # Coupling charge with curtail is required: a bare charge bound can be
    # defeated by curtailing solar to free the cap for grid import (see spec
    # Section 5.y). Reduces to charge[i] <= surplus when curtail is pinned to 0.
    if no_grid_charge_ew is not None and bool(no_grid_charge_ew.any()):
        masked = np.nonzero(no_grid_charge_ew)[0].astype(np.intp)
        m = masked.size
        row_ids = next_row + np.arange(m, dtype=np.intp)
        ub_rows_list.append(np.tile(row_ids, 2))
        ub_cols_list.append(np.concatenate([OC + masked, OQ + masked]))
        ub_data_list.append(np.ones(2 * m))
        b_ub_list.append(np.maximum(0.0, solar[masked] - load[masked]))
        next_row += m

    n_ub = next_row
    A_ub = sp.csc_matrix(
        (np.concatenate(ub_data_list),
         (np.concatenate(ub_rows_list), np.concatenate(ub_cols_list))),
        shape=(n_ub, N),
    )
    b_ub = np.concatenate(b_ub_list)

    # ------------------------------------------------------------------ #
    # Variable bounds                                                      #
    # ------------------------------------------------------------------ #
    lo = np.zeros(N)
    hi = np.full(N, np.inf)

    hi[OC:OC+Te]   = power_kwh
    hi[OD:OD+Te]   = power_kwh
    # n_imp[t]: no per-interval cap; aggregate budget enforced via inequality row
    hi[ONE:ONE+Te] = n_exp_hi       # per-interval natural export ceiling (SC-fixed)
    lo[OS:OS+Te]   = soc_min
    hi[OS:OS+Te]   = capacity
    # No terminal SOC constraint: soc[Te-1] floats within [soc_min, capacity]
    # curtail: [0, max(0, solar)] when enabled; pinned to 0 when disabled
    hi[OQ:OQ+Te] = np.maximum(0.0, solar) if p_curtail is not None else 0.0
    # f_exp: pinned to 0 at intervals below min_export_price threshold.
    # This conserves the annual cap for high-value dispatch opportunities.
    if min_export_price_per_kwh > 0.0:
        hi[OFE:OFE+Te] = np.where(op_exp >= min_export_price_per_kwh, np.inf, 0.0)

    # ------------------------------------------------------------------ #
    # Solve                                                                #
    # ------------------------------------------------------------------ #
    result = linprog(
        c,
        A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        bounds=np.column_stack([lo, hi]),
        method='highs',
    )
    if result.status != 0:
        # Should not occur with SC-fixed bounds (SC is always a feasible point).
        # Emit a structured warning for investigation; caller falls back to SC.
        rhs_total = float((load - solar).sum())
        warnings.warn(
            f"EOV_SKIP:{site_id}:{win_start}:{remaining_cap:.2f}:"
            f"{rhs_total:.2f}:{sc_n_imp_total:.2f}:"
            f"status={result.status}",
            stacklevel=2,
        )
        return None
    return result.x


# ---------------------------------------------------------------------------
# Validation and post-solve checks
# ---------------------------------------------------------------------------

def _validate_inputs(
    solar_kwh, load_kwh, op_price_imp, op_price_exp,
    vpp_event, bspec, soc_init_kwh,
    vpp_exp_reward, vpp_imp_reward, vpp_limit_applies_to,
    p_curtail,
) -> None:
    ref_index = solar_kwh.index
    ref_len   = len(solar_kwh)
    for name, s in (
        ('load_kwh',      load_kwh),
        ('op_price_imp',  op_price_imp),
        ('op_price_exp',  op_price_exp),
    ):
        if not s.index.equals(ref_index):
            raise ValueError(
                f"{name} index does not match solar_kwh index "
                f"({name} len={len(s)}, solar len={ref_len})."
            )
    if len(vpp_event) != ref_len:
        raise ValueError(
            f"vpp_event length ({len(vpp_event)}) does not match "
            f"solar_kwh length ({ref_len})."
        )
    for name, s in (
        ('solar_kwh',    solar_kwh),
        ('load_kwh',     load_kwh),
        ('op_price_imp', op_price_imp),
        ('op_price_exp', op_price_exp),
    ):
        if s.isna().any():
            raise ValueError(f"{name} contains NaN values.")
    if (load_kwh < 0).any():
        raise ValueError("load_kwh contains negative values.")

    op_imp_arr = op_price_imp.to_numpy()
    op_exp_arr = op_price_exp.to_numpy()
    violation  = op_exp_arr > op_imp_arr
    if violation.any():
        first = int(violation.argmax())
        raise ValueError(
            f"op_price_exp exceeds op_price_imp at {violation.sum()} "
            f"interval(s). First at index {first}: "
            f"op_imp={op_imp_arr[first]:.6f}, op_exp={op_exp_arr[first]:.6f}."
        )
    if vpp_exp_reward < 0:
        raise ValueError(f"vpp_exp_reward must be >= 0, got {vpp_exp_reward}")
    if vpp_imp_reward < 0:
        raise ValueError(f"vpp_imp_reward must be >= 0, got {vpp_imp_reward}")
    if vpp_limit_applies_to not in ('export_only', 'import_and_export'):
        raise ValueError(
            f"vpp_limit_applies_to must be 'export_only' or "
            f"'import_and_export', got {vpp_limit_applies_to!r}"
        )
    soc_min = (bspec.backup_reserve_pct / 100.0) * bspec.capacity_kwh
    tol = 1e-9
    if soc_init_kwh < soc_min - tol or soc_init_kwh > bspec.capacity_kwh + tol:
        raise ValueError(
            f"soc_init_kwh={soc_init_kwh:.6f} is outside battery bounds "
            f"[{soc_min:.6f}, {bspec.capacity_kwh:.6f}]."
        )
    if p_curtail is not None and p_curtail < 0:
        raise ValueError(
            f"p_curtail must be >= 0 when provided, got {p_curtail}. "
            f"A negative value would reward curtailment and cause the LP to "
            f"discard all exportable solar."
        )


def _clip(arr: np.ndarray, name: str, site_id: str) -> np.ndarray:
    """Floor solver noise to zero; warn if magnitude is significant."""
    neg_mask = arr < 0
    if neg_mask.any():
        max_neg = float(-arr[neg_mask].min())
        if max_neg > _WARN_TOL:
            warnings.warn(
                f"vpp_sc_event [{site_id}] {name}: {neg_mask.sum()} "
                f"value(s) below 0, max magnitude {max_neg:.2e}; clipping.",
                stacklevel=4,
            )
        return np.clip(arr, 0.0, None)
    return arr


def _check_simultaneous_flows(
    n_imp, f_imp, n_exp, f_exp, charge, discharge, site_id, win_start,
) -> None:
    """Warn on simultaneous charge+discharge or import+export."""
    simul_cd = (charge > _SIMUL_TOL) & (discharge > _SIMUL_TOL)
    if simul_cd.any():
        warnings.warn(
            f"vpp_sc_event [{site_id}]: {simul_cd.sum()} interval(s) "
            f"with simultaneous charge+discharge in event window at "
            f"t={win_start} (solver degeneracy; billing unaffected).",
            stacklevel=3,
        )
    imp = n_imp + f_imp
    exp = n_exp + f_exp
    simul_ie = (imp > _SIMUL_TOL) & (exp > _SIMUL_TOL)
    if simul_ie.any():
        warnings.warn(
            f"vpp_sc_event [{site_id}]: {simul_ie.sum()} interval(s) "
            f"with simultaneous import+export in event window at t={win_start}.",
            stacklevel=3,
        )


def _check_no_grid_charge(
    grid_to_bat, charge, curtail, solar, load, ngc_ew, site_id, win_start,
) -> None:
    """Invariant check: no grid energy charges the battery on masked intervals.

    With the coupling row charge[t] + curtail[t] <= max(0, solar[t] - load[t])
    on masked intervals (spec Section 5.y), grid_to_bat[t] must be zero there.
    Any breach is a bug (a bound/row error or a solver tolerance problem), not
    an accepted artifact. The warning reports the offending interval(s) with the
    context needed to diagnose: charge, curtail, and the solar surplus.
    """
    if ngc_ew is None or not bool(np.asarray(ngc_ew).any()):
        return
    mask = np.asarray(ngc_ew, dtype=bool)
    breach = mask & (grid_to_bat > _SIMUL_TOL)
    if breach.any():
        i = int(np.argmax(breach))
        surplus_i = max(0.0, float(solar[i] - load[i]))
        warnings.warn(
            f"vpp_sc_event [{site_id}]: {int(breach.sum())} masked interval(s) "
            f"with grid_to_bat > 0 in no-grid-charge window at t={win_start}. "
            f"First at offset {i}: grid_to_bat={float(grid_to_bat[i]):.6f}, "
            f"charge={float(charge[i]):.6f}, curtail={float(curtail[i]):.6f}, "
            f"solar_surplus={surplus_i:.6f}.",
            stacklevel=3,
        )