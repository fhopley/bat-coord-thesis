"""perfect_foresight_lp_advanced — LP dispatch with advanced retail tariff features.

Extends `perfect_foresight_lp` to handle:
  - Block import and export tiers (stepped daily pricing)
  - Zero-import reward (daily credit for avoiding window imports)
  - Critical export/import rates (per-interval replacement rate at price
    extremes, with the critical flow outside the block tiers)

Routed via the `perfect_foresight_advanced` coordination type. For a plan
with `retail_tariff.has_advanced_features` False it gives the same result as
`perfect_foresight_lp`.

Design
------
Critical rates and block window masking are applied to the price vectors by
`build_lp_price_vectors_advanced` before this function is called. The
`price_imp` and `price_exp` received here already reflect:
  - Standard rates for non-block intervals
  - Non-block components only (usage_other / feedin_other) for block intervals
  - Critical rates for critical intervals, with the opposite side shifted
Tier membership for critical intervals is decided here, from prices_mwh
(prices.block_tier_masks).

This function adds:
  1. Zero-import reward: per-interval penalty added to p_imp for window
     intervals only (ARTIFICIAL -- excluded from billing). See
     `build_zero_import_penalty` in prices.py.
  2. Block tier variables: new LP variables (v_imp[w,d,k], v_exp[w,d,k])
     representing daily energy allocation to each tier, plus linking
     constraints tying them to the per-interval imp[t] / exp[t] sums.

Variable layout
---------------
The standard 6T block is UNCHANGED (same offsets as perfect_foresight_lp):
  x[0:T]    = charge
  x[T:2T]   = discharge
  x[2T:3T]  = imp
  x[3T:4T]  = exp
  x[4T:5T]  = soc
  x[5T:6T]  = curtail

New tier variables appended after x[5T:6T]:
  For each import block window w, each calendar day d, each tier k:
    v_imp[w, d, k]  -- daily energy volume in that tier (kWh)
  For each export block window w, each calendar day d, each tier k:
    v_exp[w, d, k]

Linking constraints
-------------------
For each import window w and day d:
  Σ_k v_imp[w,d,k] - Σ_{t in w,d} imp[t] = 0

Symmetrically for exports.

Objective contribution from tier variables
------------------------------------------
For import tiers: +tier_rate_k/C ($/kWh) * v_imp[w,d,k]
For export tiers: -tier_rate_k/C ($/kWh) * v_exp[w,d,k]  (reward: negative cost)

Price ordering
--------------
`build_lp_price_vectors_advanced` calls `apply_critical_rates` with
`shift_opposite=True`, which moves the opposite vector by the same amount as
the critical side. The spread (p_imp - p_exp) is preserved, so p_imp >= p_exp
holds wherever it holds for the base vectors. `_validate_inputs_advanced` still skips
the ordering check as a defensive measure for non-standard callers.

Imports from perfect_foresight_lp
----------------------------------
All shared helpers are imported directly from the standard module. No code is
duplicated. See that module's docstring for the full LP formulation.
"""
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

from model.core.scenario import BatterySpec
from model.core.site_result import SiteTraces
from model.core.tariff import RetailTariff
from model.core.localtime import local_wall_clock
from model.core.prices import (
    apply_tou_export_order_nudge,
    build_zero_import_penalty,
    build_tou_export_order_nudge,
    check_export_order_nudge_degradation,
    block_window_mask,
    block_tier_masks,
    C_TO_DOLLARS,
)
from model.components.perfect_foresight_lp import (
    _build_lp,
    _solve_gurobi,
    _solve_highs,
    _reconstruct_flows,
    _clip_nonneg,
    _check_simultaneous_flows,
    _net_simultaneous_charge_discharge,
    _validate_inputs,
    lp_backend,
    _GUROBI_AVAILABLE,
    _WARN_TOL,
    _SIMUL_TOL,
)


def perfect_foresight_lp_advanced(
    site_id: str,
    solar_kwh: pd.Series,
    load_kwh: pd.Series,
    price_imp: pd.Series,
    price_exp: pd.Series,
    battery_spec: BatterySpec,
    soc_init_kwh: float,
    soc_final_kwh: float,
    retail_tariff: RetailTariff,
    prices_mwh: pd.Series,
    state: str,
    export_limit_kwh_per_interval: float = np.inf,
    interval_hours: float = 5 / 60,
    first_day_block_import_used: dict[int, float] | None = None,
    first_day_block_export_used: dict[int, float] | None = None,
    apply_export_order_nudge: bool = False,
    curtail_cost_per_kwh_dollars: float | None = 0.0,
) -> SiteTraces:
    """LP-based perfect-foresight dispatch with advanced retail tariff features.

    Args:
        site_id: Passed through to output SiteTraces.
        solar_kwh: Gross solar generation per interval (kWh). May be negative.
        load_kwh: Gross household load per interval (kWh). Must be >= 0.
        price_imp: Per-interval import price ($/kWh), pre-built by
            `build_lp_price_vectors_advanced`. Block masking and critical
            rates already applied.
        price_exp: Per-interval export price ($/kWh), pre-built by
            `build_lp_price_vectors_advanced`.
        battery_spec: Battery specification.
        soc_init_kwh: Battery SOC at period start.
        soc_final_kwh: Required battery SOC at period end.
        retail_tariff: Retail plan. Used to build block tier variables and
            the zero-import penalty. When has_advanced_features is False,
            produces identical results to perfect_foresight_lp.
        prices_mwh: Raw wholesale prices ($/MWh), same index as solar_kwh.
            Decides which intervals are critical for block tier membership
            (critical flow does not count towards a tier). The critical
            rates themselves are already in price_imp / price_exp.
        state: The site's actual location (scenario.state). Block windows,
            the zero-import reward window, and the export-order nudge are
            all specified in local time, not AEST -- see core/localtime.py.
        export_limit_kwh_per_interval: Per-interval export cap (kWh).
        interval_hours: Interval length in hours.
        first_day_block_import_used: kWh already consumed in each import
            block window on the current calendar day before the horizon
            starts. Keys are window indices (0-based, matching order in
            retail_tariff.block_import_tiers). None means zero for all
            windows. Only affects day-0 tier variable bounds; all subsequent
            days reset normally. Used by vpp_tou_event for mid-day horizons.
        first_day_block_export_used: Same structure for export block windows.
        apply_export_order_nudge: When True, apply a tiny per-interval nudge
            to p_exp_lp that increases linearly through each calendar day,
            biasing the LP toward charging with solar first and exporting the
            surplus -- the physically realistic controller behaviour. Default
            False for bulk runs (adds ~3x solve time overhead). Pass True from
            viz scripts where realistic dispatch profiles are required.
            Raises ValueError on a TOU plan when battery_spec.c_deg_per_kwh
            does not exceed TOU_EXPORT_NUDGE_MAX / 2. The nudge is capped at
            the import price (apply_tou_export_order_nudge). See
            `build_tou_export_order_nudge` in prices.py and
            perfect_foresight_advanced_spec.md for full details.
        curtail_cost_per_kwh_dollars: Curtailment mode, passed to the
            standard LP builder (perfect_foresight_lp_spec.md, Curtailment
            modes). 0.0 (default) is free curtailment. The orchestrator
            passes prices.dispatch_curtail_cost(retail_tariff), which is
            MIXED_CURTAIL_COST_PER_KWH on a mixed plan.

    Returns:
        SiteTraces with LP-optimal dispatch. Identical to perfect_foresight_lp
        output for standard (non-advanced) plans.

    Raises:
        ValueError: for invalid inputs (NaN, misaligned indices, bad SOC bounds).
            Note: the price ordering check (p_exp <= p_imp) is skipped.
            The critical-rate shift preserves the spread, but the export-order
            nudge can push p_exp marginally above p_imp, and callers may
            supply vectors built by other means.
        RuntimeError: if the solver does not find an optimal solution.
    """
    _validate_inputs_advanced(
        solar_kwh, load_kwh, price_imp, price_exp,
        battery_spec, soc_init_kwh, soc_final_kwh,
    )

    index = solar_kwh.index
    T = len(index)
    solar = solar_kwh.to_numpy(dtype=float, copy=False)
    load  = load_kwh.to_numpy(dtype=float, copy=False)

    # Apply zero-import penalty to p_imp (artificial incentive, not in billing).
    p_imp_lp = price_imp.to_numpy(dtype=float, copy=True)
    penalty = build_zero_import_penalty(index, retail_tariff, state, interval_hours)
    if penalty is not None:
        p_imp_lp += penalty.to_numpy()

    # Apply TOU export order nudge to p_exp (artificial incentive, not in billing).
    # Breaks LP degeneracy on solar-surplus days: makes later exports marginally
    # more attractive, so the LP charges first and exports the surplus -- the
    # physically realistic controller behaviour. No-op for wholesale_exposed plans.
    # Disabled by default (adds ~3x solve time); pass apply_export_order_nudge=True
    # from viz scripts where realistic dispatch profiles are required.
    p_exp_lp = price_exp.to_numpy(dtype=float, copy=True)
    if apply_export_order_nudge:
        nudge = build_tou_export_order_nudge(index, retail_tariff, state, interval_hours)
        if nudge is not None:
            check_export_order_nudge_degradation(
                battery_spec.c_deg_per_kwh, retail_tariff.name)
            p_exp_lp = apply_tou_export_order_nudge(
                price_imp.to_numpy(dtype=float), p_exp_lp, nudge.to_numpy())
    # Note: build_lp_price_vectors_advanced applies critical rates with
    # shift_opposite=True, so the spread (p_imp - p_exp) is preserved and
    # p_imp >= p_exp holds wherever it holds for the base vectors. Billing
    # uses shift_opposite=False (the real rates).

    # Build the standard LP components.
    c_std, A_eq_std, b_eq_std, lo_std, hi_std = _build_lp(
        T, solar, load, p_imp_lp, p_exp_lp, battery_spec,
        soc_init_kwh, soc_final_kwh,
        export_limit_kwh_per_interval, interval_hours,
        curtail_cost_per_kwh_dollars=curtail_cost_per_kwh_dollars,
    )

    # Augment with block tier variables and linking constraints.
    n_days, day_of_each_interval = _build_day_map(index, state, interval_hours)
    # Critical flow does not count towards a block tier, so on a plan with
    # critical rates the linking rows use the effective masks (critical
    # intervals on the window's side removed, by the settled price).
    imp_masks = exp_masks = None
    if retail_tariff.has_critical_rates and (retail_tariff.block_import_tiers
                                             or retail_tariff.block_export_tiers):
        imp_masks, exp_masks = block_tier_masks(
            index, retail_tariff, state, interval_hours, prices_mwh)
    (n_tier_vars, tier_lo, tier_hi,
     tier_costs, linking_rows) = _build_block_tier_data(
        T, index, retail_tariff, interval_hours, n_days, day_of_each_interval,
        state,
        first_day_block_import_used=first_day_block_import_used or {},
        first_day_block_export_used=first_day_block_export_used or {},
        import_window_masks=imp_masks or None,
        export_window_masks=exp_masks or None,
    )

    if n_tier_vars == 0:
        # No block features: solve the augmented-but-unmodified standard LP.
        c_obj = c_std
        A_eq  = A_eq_std
        b_eq  = b_eq_std
        lo    = lo_std
        hi    = hi_std
    else:
        c_obj = np.concatenate([c_std, tier_costs])
        lo    = np.concatenate([lo_std, tier_lo])
        hi    = np.concatenate([hi_std, tier_hi])
        # Pad standard constraints with zero columns for tier variables.
        A_eq_pad = sp.hstack(
            [A_eq_std, sp.csc_matrix((2 * T, n_tier_vars))],
            format='csc',
        )
        A_eq = sp.vstack([A_eq_pad, linking_rows], format='csc')
        b_eq = np.concatenate([b_eq_std, np.zeros(linking_rows.shape[0])])


    if _GUROBI_AVAILABLE:
        x = _solve_gurobi(c_obj, A_eq, b_eq, lo, hi, site_id)
    else:
        x = _solve_highs(c_obj, A_eq, b_eq, lo, hi, site_id)


    # Extract core 6T variables (tier variables are not needed for SiteTraces).
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
# Day mapping helper
# ---------------------------------------------------------------------------

def _build_day_map(
    index: pd.DatetimeIndex,
    state: str,
    interval_hours: float,
) -> tuple[int, np.ndarray]:
    """Return (n_days, day_index_per_interval).

    day_index_per_interval[t] is the 0-based calendar day index for interval t,
    using local interval-START dates for `state` (consistent with
    block_window_mask and billing.py's day-grouping). Groups by an actual
    local calendar-date component (via local_wall_clock), so DST transition
    days -- which are genuinely 23 or 25 hours long in local time -- fall
    into their own single day group correctly with no special-casing.
    n_days is the total number of distinct local calendar days in the period.
    """
    lwc = local_wall_clock(index, state, interval_hours)
    # Map each unique local date to a 0-based integer index
    unique_dates, day_index = np.unique(lwc.date, return_inverse=True)
    return len(unique_dates), day_index.astype(np.intp)


# ---------------------------------------------------------------------------
# Block tier LP data builder
# ---------------------------------------------------------------------------

def _build_block_tier_data(
    T: int,
    index: pd.DatetimeIndex,
    retail_tariff: RetailTariff,
    interval_hours: float,
    n_days: int,
    day_of_each_interval: np.ndarray,
    state: str,
    first_day_block_import_used: dict[int, float] | None = None,
    first_day_block_export_used: dict[int, float] | None = None,
    import_window_masks: tuple[np.ndarray, ...] | None = None,
    export_window_masks: tuple[np.ndarray, ...] | None = None,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, sp.csc_matrix]:
    """Build tier variable data for block import and export windows.

    Args:
        first_day_block_import_used: kWh already consumed in each import block
            window on the current calendar day before the horizon starts. Keys
            are window indices (0-based). Missing keys default to zero. Only
            affects day-0 tier variable bounds; all subsequent days are
            unaffected. None or empty dict means zero for all windows.
        first_day_block_export_used: Same for export block windows.
        import_window_masks: Precomputed boolean masks over `index`, one per
            import block window, in the same order as
            retail_tariff.block_import_tiers. When None (the default) each
            mask is computed here by block_window_mask, which is what every
            whole-period caller on a plan without critical rates wants.
            Masks are used as given, so a caller on a plan with critical
            rates must already have removed critical intervals
            (prices.block_tier_masks); perfect_foresight_lp_advanced does.

            Supplied by the rolling advanced path, which calls this function
            once per 5-minute window -- roughly 105,000 times per site-year --
            against masks it computed once for the whole period and slices per
            window. Recomputing them here would mean a local_wall_clock
            conversion per window per block window.

            Each mask must have length len(index).
        export_window_masks: Same for export block windows.

    Returns:
        n_tier_vars: Total number of new tier variables.
        tier_lo: Lower bounds for tier variables (all 0).
        tier_hi: Upper bounds (tier capacity in kWh; inf for final tier).
        tier_costs: Objective coefficients ($/kWh; negative for export reward).
        linking_rows: Sparse matrix of shape (n_linking, 6T + n_tier_vars).
            Each row is one linking constraint:
                Σ_k v[w,d,k] - Σ_{t in w,d} x[2T+t or 3T+t] = 0

    When no block features are present, returns (0, empty, empty, empty, empty).
    """
    import_windows  = retail_tariff.block_import_tiers or ()
    export_windows  = retail_tariff.block_export_tiers or ()
    imp_used = first_day_block_import_used or {}
    exp_used = first_day_block_export_used or {}

    if not import_windows and not export_windows:
        empty = np.empty(0)
        return 0, empty, empty, empty, sp.csc_matrix((0, 6 * T))

    # Enumerate all tier variables in a canonical order:
    # [all import window 0 tiers x days, import window 1 tiers x days, ...,
    #  all export window 0 tiers x days, ...]
    # We build lists of (lo, hi, cost, row_data) then assemble.

    all_lo    = []
    all_hi    = []
    all_costs = []

    # linking_row_data: list of (row_indices_into_x, col_indices_into_tier_vars, data)
    # We'll build COO data for the full linking matrix.
    link_row_list = []   # which linking constraint row
    link_col_list = []   # which column in the FULL extended variable vector
    link_data_list = []
    n_linking = 0

    # Total x variable size is 6T; tier vars start at offset 6T.
    N_CORE = 6 * T
    tier_var_offset = 0   # offset within the tier variable block

    C = C_TO_DOLLARS

    def _process_windows(windows, is_import, used_dict, masks):
        nonlocal n_linking, tier_var_offset
        # Core variable column offset for imp or exp
        core_col_offset = 2 * T if is_import else 3 * T

        if masks is not None and len(masks) != len(windows):
            side = 'import' if is_import else 'export'
            raise ValueError(
                f"_build_block_tier_data: {len(masks)} precomputed {side} "
                f"window mask(s) supplied for {len(windows)} {side} block "
                f"window(s); one mask per window is required, in tariff order"
            )

        for w_idx, window in enumerate(windows):
            if masks is None:
                mask = block_window_mask(index, window, state, interval_hours)
            else:
                mask = masks[w_idx]
                if len(mask) != len(index):
                    side = 'import' if is_import else 'export'
                    raise ValueError(
                        f"_build_block_tier_data: precomputed {side} window "
                        f"mask {w_idx} has length {len(mask)}, expected "
                        f"{len(index)} to match index"
                    )
            n_tiers = len(window.tiers)
            tiers = window.tiers
            # Accumulated usage for this window's first day (0 if not provided)
            day0_used = float(used_dict.get(w_idx, 0.0))

            for d in range(n_days):
                # Intervals in this window on this day
                day_mask = mask & (day_of_each_interval == d)
                window_day_intervals = np.where(day_mask)[0].astype(np.intp)

                # One tier variable per tier per day per window.
                # For day 0 only, accumulated usage reduces available capacity.
                remaining_used = day0_used if d == 0 else 0.0
                prev_threshold = 0.0
                for k, tier in enumerate(tiers):
                    if np.isinf(tier.threshold_kwh):
                        tier_capacity = np.inf
                    else:
                        tier_capacity = tier.threshold_kwh - prev_threshold
                        prev_threshold = tier.threshold_kwh

                    # Reduce capacity by accumulated usage on day 0
                    if d == 0 and remaining_used > 0:
                        if np.isinf(tier_capacity):
                            hi_k = np.inf   # infinite tier: usage never exhausts it
                        else:
                            hi_k = max(0.0, tier_capacity - remaining_used)
                            remaining_used = max(0.0, remaining_used - tier_capacity)
                    else:
                        hi_k = tier_capacity

                    all_lo.append(0.0)
                    all_hi.append(hi_k)

                    cost_k = tier.rate_c_per_kwh / C
                    if not is_import:
                        cost_k = -cost_k   # export: negative = reward
                    all_costs.append(cost_k)

                    # Linking row: Σ_k v[w,d,k] - Σ_t imp/exp[t in w,d] = 0
                    linking_row_idx = n_linking
                    link_row_list.append(np.full(1, linking_row_idx, dtype=np.intp))
                    link_col_list.append(np.array([N_CORE + tier_var_offset], dtype=np.intp))
                    link_data_list.append(np.array([1.0]))

                    tier_var_offset += 1

                # Add -1 entries for each imp/exp[t] in (window, day) to same row
                if len(window_day_intervals) > 0:
                    link_row_list.append(
                        np.full(len(window_day_intervals), linking_row_idx, dtype=np.intp))
                    link_col_list.append(
                        core_col_offset + window_day_intervals)
                    link_data_list.append(
                        np.full(len(window_day_intervals), -1.0))

                n_linking += 1

    _process_windows(import_windows, is_import=True,  used_dict=imp_used,
                     masks=import_window_masks)
    _process_windows(export_windows, is_import=False, used_dict=exp_used,
                     masks=export_window_masks)

    n_tier_vars = tier_var_offset
    tier_lo    = np.array(all_lo)
    tier_hi    = np.array(all_hi)
    tier_costs = np.array(all_costs)

    # Assemble sparse linking matrix: shape (n_linking, N_CORE + n_tier_vars)
    n_cols_total = N_CORE + n_tier_vars
    if link_row_list:
        rows = np.concatenate(link_row_list)
        cols = np.concatenate(link_col_list)
        data = np.concatenate(link_data_list)
    else:
        rows = cols = data = np.empty(0, dtype=np.intp)

    linking_rows = sp.csc_matrix(
        (data, (rows, cols)),
        shape=(n_linking, n_cols_total),
    )

    return n_tier_vars, tier_lo, tier_hi, tier_costs, linking_rows


# ---------------------------------------------------------------------------
# Input validation (relaxed price ordering check)
# ---------------------------------------------------------------------------

def _validate_inputs_advanced(
    solar_kwh: pd.Series,
    load_kwh: pd.Series,
    price_imp: pd.Series,
    price_exp: pd.Series,
    bspec: BatterySpec,
    soc_init: float,
    soc_final: float,
) -> None:
    """Validate LP inputs, skipping the p_exp <= p_imp ordering check.

    When called via the normal path, build_lp_price_vectors_advanced applies
    critical rates with the opposite-side shift, preserving p_imp >= p_exp
    wherever the base vectors satisfy it. The check is still skipped as a defensive measure
    for callers that supply price vectors by other means.

    All other checks (index alignment, NaN, non-negative load, SOC bounds)
    are performed identically to _validate_inputs.
    """
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

    if (load_kwh < 0).any():
        raise ValueError("load_kwh contains negative values.")

    # SOC boundary checks
    soc_min = (bspec.backup_reserve_pct / 100.0) * bspec.capacity_kwh
    tol = 1e-9
    for name, val in (('soc_init_kwh', soc_init), ('soc_final_kwh', soc_final)):
        if val < soc_min - tol or val > bspec.capacity_kwh + tol:
            raise ValueError(
                f"{name}={val:.6f} is outside battery bounds "
                f"[{soc_min:.6f}, {bspec.capacity_kwh:.6f}]."
            )
    # Note: price ordering check (p_exp <= p_imp) deliberately omitted.
    # build_lp_price_vectors_advanced shifts the opposite side of critical
    # intervals, so the spread of the base vectors is preserved.