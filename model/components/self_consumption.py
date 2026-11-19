"""Self-consumption coordination strategy.

Models a single site operating under simple self-consumption logic:
  1. Solar meets load first (sol_to_home).
  2. Surplus solar charges the battery if present (sol_to_bat).
  3. Any remaining solar is exported (sol_to_grid).
  4. Residual load is met by battery discharge if available (bat_to_home).
  5. Any remaining load is imported from the grid (grid_to_home).

Under self-consumption the battery never charges from grid and never
exports to grid: grid_to_bat and bat_to_grid are zero.

Power and capacity constraints
------------------------------
BatterySpec.power_kw is interpreted as the AC-side maximum power flow at
the battery's inverter. The per-interval AC energy cap is therefore
    power_per_interval_kwh = power_kw * interval_hours
and applies directly to sol_to_bat and bat_to_home (no efficiency factor).

DC-side capacity constraints (charge efficiency converts AC inflow to DC
storage; discharge efficiency converts DC storage to AC outflow):
    charge:    sol_to_bat * charge_eff added to soc_dc_kwh
               sol_to_bat <= (capacity_kwh - soc_kwh) / charge_eff
    discharge: bat_to_home / discharge_eff removed from soc_dc_kwh
               bat_to_home <= (soc_kwh - min_soc_kwh) * discharge_eff
               where min_soc_kwh = (backup_reserve_pct/100) * capacity_kwh

These constraints are enforced as min() bounds on each flow at each
interval. The battery state of charge stored is on the DC side - what
is actually in the cells.

Initial state
-------------
Battery SOC at the start of the scenario period is set to the backup
reserve floor (or 0 if no reserve). Self-consumption simulations are
typically annual, so the initial-state choice has negligible effect on
annual totals.

Cohort handling
---------------
This single function handles all three cohorts via the caller passing
appropriate inputs:
  - no_cer:        solar_kwh = zeros, battery_spec = None
  - solar:         solar_kwh = actual,  battery_spec = None
  - solar_battery: solar_kwh = actual,  battery_spec = spec

All flow series are populated regardless of cohort - inapplicable flows
are filled with zeros so downstream code is uniform.
"""
import numpy as np
import pandas as pd

from model.core.scenario import BatterySpec
from model.core.site_result import SiteTraces


def self_consumption(
    site_id: str,
    solar_kwh: pd.Series,
    load_kwh: pd.Series,
    battery_spec: BatterySpec | None,
    interval_hours: float = 5 / 60,
    neg_price_aware: bool = False,
    price_imp: pd.Series | None = None,
    price_exp: pd.Series | None = None,
) -> SiteTraces:
    """Run self-consumption logic for a single site.

    Args:
        site_id: Site identifier (passed through to the output).
        solar_kwh: Gross solar generation per interval, kWh, >= 0.
        load_kwh: Gross household load per interval, kWh, >= 0.
            Both series must share the same index. For the no_cer cohort
            the caller passes solar_kwh as zeros.
        battery_spec: Battery to model, or None for cohorts without a
            battery (no_cer, solar).
        interval_hours: Length of each interval in hours. Default 5/60
            (5-min cadence matching the EDP and NEM clean data). Pass
            0.5 for 30-min cadence.
        neg_price_aware: If True, apply price-aware corrections after the
            normal self-consumption logic at each interval:
              - Negative export price: curtail sol_to_grid (avoid negative
                revenue); price_exp is required.
              - Negative import price (implies negative export price in NEM):
                curtail all remaining solar (sol_to_home and sol_to_bat),
                replace sol_to_bat with grid_to_bat at the same AC amount
                so the SOC trajectory is unchanged, and serve the full load
                from the grid (minus any bat_to_home already calculated).
                This leaves the battery charging schedule and SOC identical
                to the normal baseline, while maximising negative-price
                imports; price_imp is also required.
            Default False preserves existing behaviour exactly.
        price_imp: Per-interval import price series ($/kWh). Required when
            neg_price_aware=True.
        price_exp: Per-interval export price series ($/kWh). Required when
            neg_price_aware=True.

    Returns:
        SiteTraces with import/export and full flow decomposition.

    Raises:
        ValueError: if input series have different indices or different
            lengths, or if any input contains negative values or NaN, or
            if neg_price_aware=True but price series are not provided.
    """
    _validate_inputs(solar_kwh, load_kwh)

    if neg_price_aware:
        if price_imp is None or price_exp is None:
            raise ValueError(
                "price_imp and price_exp must be provided when neg_price_aware=True."
            )
        if not price_imp.index.equals(solar_kwh.index):
            raise ValueError("price_imp index does not match solar_kwh index.")
        if not price_exp.index.equals(solar_kwh.index):
            raise ValueError("price_exp index does not match solar_kwh index.")
        p_imp_arr = price_imp.to_numpy(dtype=float, copy=False)
        p_exp_arr = price_exp.to_numpy(dtype=float, copy=False)
    else:
        p_imp_arr = None
        p_exp_arr = None

    index = solar_kwh.index
    n = len(index)
    solar_arr = solar_kwh.to_numpy(dtype=float, copy=False)
    load_arr = load_kwh.to_numpy(dtype=float, copy=False)

    # Pre-allocate output arrays
    sol_to_home = np.zeros(n)
    sol_to_bat = np.zeros(n)
    sol_to_grid = np.zeros(n)
    bat_to_home = np.zeros(n)
    grid_to_home = np.zeros(n)
    grid_to_bat = np.zeros(n)
    curtail = np.zeros(n)
    soc_kwh = np.zeros(n)

    if battery_spec is None:
        # No battery - solar covers load to the extent it can; rest imports/exports.
        _run_no_battery(
            solar_arr, load_arr,
            sol_to_home, sol_to_grid, grid_to_home,
            curtail, neg_price_aware, p_imp_arr, p_exp_arr,
        )
    else:
        capacity_kwh = battery_spec.capacity_kwh
        min_soc_kwh = (battery_spec.backup_reserve_pct / 100.0) * capacity_kwh
        power_per_interval_kwh = battery_spec.power_kw * interval_hours
        charge_eff = battery_spec.charge_efficiency
        discharge_eff = battery_spec.discharge_efficiency

        # Start the battery at the reserve floor. For an annual sim the
        # initial state has negligible effect on total throughput.
        initial_soc_kwh = min_soc_kwh

        _run_with_battery(
            solar_arr, load_arr,
            sol_to_home, sol_to_bat, sol_to_grid,
            bat_to_home, grid_to_home, grid_to_bat, curtail, soc_kwh,
            capacity_kwh, min_soc_kwh, power_per_interval_kwh,
            charge_eff, discharge_eff, initial_soc_kwh,
            neg_price_aware, p_imp_arr, p_exp_arr,
        )

    bat_to_grid = np.zeros(n)

    import_kwh = grid_to_home + grid_to_bat
    export_kwh = sol_to_grid + bat_to_grid

    def _series(arr: np.ndarray, name: str) -> pd.Series:
        return pd.Series(arr, index=index, name=name)

    return SiteTraces(
        site_id=site_id,
        import_kwh=_series(import_kwh, 'import_kwh'),
        export_kwh=_series(export_kwh, 'export_kwh'),
        battery_soc_kwh=_series(soc_kwh, 'battery_soc_kwh'),
        sol_to_home_kwh=_series(sol_to_home, 'sol_to_home_kwh'),
        sol_to_bat_kwh=_series(sol_to_bat, 'sol_to_bat_kwh'),
        sol_to_grid_kwh=_series(sol_to_grid, 'sol_to_grid_kwh'),
        bat_to_home_kwh=_series(bat_to_home, 'bat_to_home_kwh'),
        grid_to_home_kwh=_series(grid_to_home, 'grid_to_home_kwh'),
        grid_to_bat_kwh=_series(grid_to_bat, 'grid_to_bat_kwh'),
        bat_to_grid_kwh=_series(bat_to_grid, 'bat_to_grid_kwh'),
        curtail_kwh=_series(curtail, 'curtail_kwh'),
    )


def _validate_inputs(solar_kwh: pd.Series, load_kwh: pd.Series) -> None:
    if not solar_kwh.index.equals(load_kwh.index):
        raise ValueError(
            "solar_kwh and load_kwh must share the same index. "
            f"solar length={len(solar_kwh)}, load length={len(load_kwh)}."
        )
    if solar_kwh.isna().any():
        raise ValueError("solar_kwh contains NaN values")
    if load_kwh.isna().any():
        raise ValueError("load_kwh contains NaN values")
    # Note: solar_kwh can be negative — real EDP data shows small overnight
    # inverter parasitic draws (the inverter circuit pulling from grid).
    # The self-consumption logic handles this correctly: sol_to_home goes
    # negative, residual_load increases by |negative_solar|, and grid
    # imports cover the difference. gross_load is unchanged - it already
    # reflects the inverter draw via the EDP convention
    # gross_load = solar + ac_load_net.
    if (load_kwh < 0).any():
        raise ValueError("load_kwh contains negative values")


def _run_no_battery(
    solar: np.ndarray,
    load: np.ndarray,
    sol_to_home_out: np.ndarray,
    sol_to_grid_out: np.ndarray,
    grid_to_home_out: np.ndarray,
    curtail_out: np.ndarray,
    neg_price_aware: bool = False,
    p_imp: np.ndarray | None = None,
    p_exp: np.ndarray | None = None,
) -> None:
    """No-battery self-consumption. Vectorised - no temporal dependence."""
    np.minimum(solar, load, out=sol_to_home_out)
    np.subtract(solar, sol_to_home_out, out=sol_to_grid_out)
    np.subtract(load, sol_to_home_out, out=grid_to_home_out)

    if neg_price_aware:
        # Negative export price: curtail sol_to_grid (avoid negative revenue).
        neg_exp = p_exp < 0.0
        curtail_out[neg_exp] += sol_to_grid_out[neg_exp]
        sol_to_grid_out[neg_exp] = 0.0

        # Negative import price: curtail all solar, grid serves full load.
        neg_imp = p_imp < 0.0
        curtail_out[neg_imp] += sol_to_home_out[neg_imp]
        sol_to_home_out[neg_imp] = 0.0
        grid_to_home_out[neg_imp] = load[neg_imp]


def _run_with_battery(
    solar: np.ndarray,
    load: np.ndarray,
    sol_to_home_out: np.ndarray,
    sol_to_bat_out: np.ndarray,
    sol_to_grid_out: np.ndarray,
    bat_to_home_out: np.ndarray,
    grid_to_home_out: np.ndarray,
    grid_to_bat_out: np.ndarray,
    curtail_out: np.ndarray,
    soc_out: np.ndarray,
    capacity_kwh: float,
    min_soc_kwh: float,
    power_per_interval_kwh: float,
    charge_eff: float,
    discharge_eff: float,
    initial_soc_kwh: float,
    neg_price_aware: bool = False,
    p_imp: np.ndarray | None = None,
    p_exp: np.ndarray | None = None,
) -> None:
    """Sequential per-interval self-consumption with battery.

    Pure Python loop for clarity. Numba-JIT version planned as a follow-up
    once correctness is validated.
    """
    soc = initial_soc_kwh
    n = len(solar)

    for i in range(n):
        s = solar[i]
        l = load[i]

        # 1. Solar to home (free use of available solar to cover load)
        s2h = s if s < l else l  # equivalent to min, faster in CPython

        # 2. Solar to battery: limited by surplus solar, AC power cap, DC headroom
        surplus = s - s2h
        ac_charge_cap = power_per_interval_kwh
        dc_headroom_in_ac = (capacity_kwh - soc) / charge_eff if charge_eff > 0 else 0.0
        s2b = surplus
        if ac_charge_cap < s2b:
            s2b = ac_charge_cap
        if dc_headroom_in_ac < s2b:
            s2b = dc_headroom_in_ac
        if s2b < 0.0:
            s2b = 0.0
        soc = soc + s2b * charge_eff

        # 3. Surplus solar to grid
        s2g = surplus - s2b

        # 4. Battery to home: residual load, AC power cap, available DC above reserve
        residual_load = l - s2h
        ac_discharge_cap = power_per_interval_kwh
        dc_available_in_ac = (soc - min_soc_kwh) * discharge_eff
        if dc_available_in_ac < 0.0:
            dc_available_in_ac = 0.0
        b2h = residual_load
        if ac_discharge_cap < b2h:
            b2h = ac_discharge_cap
        if dc_available_in_ac < b2h:
            b2h = dc_available_in_ac
        if b2h < 0.0:
            b2h = 0.0
        soc = soc - b2h / discharge_eff if discharge_eff > 0 else soc

        # 5. Grid covers what's left
        g2h = residual_load - b2h

        # Neg-price-aware adjustments:
        #
        # Negative export price: curtail sol_to_grid (avoid negative-revenue export).
        #
        # Negative import price (implies negative export price in NEM):
        #   At negative import price it is never economical to discharge the battery
        #   to serve load — grid import earns revenue. So:
        #   • Undo the step-4 battery discharge (b2h → 0, restore SOC).
        #   • Replace sol_to_bat with grid_to_bat at the same AC amount so the
        #     net DC stored is unchanged relative to a plain-import baseline.
        #   • Curtail only non-negative solar portions (sol_to_home and sol_to_bat).
        #     When solar < 0 (inverter parasitic draw at night), s2h is negative and
        #     must NOT be curtailed — it stays as sol_to_home (negative) and
        #     grid_to_home is adjusted upward to cover the extra draw.
        #   • g2h = l − min(s2h, 0) so that grid covers the full load PLUS any
        #     parasitic inverter draw.  For normal daytime solar (s2h ≥ 0) this
        #     reduces to g2h = l, and s2h is zeroed.
        curtail_i = 0.0
        g2b = 0.0
        if neg_price_aware:
            if p_exp[i] < 0.0:
                curtail_i += s2g
                s2g = 0.0
                if p_imp[i] < 0.0:
                    # Undo step-4 discharge so SOC is consistent with b2h = 0.
                    if discharge_eff > 0:
                        soc += b2h / discharge_eff
                    g2b = s2b                      # replace solar bat-charge with grid
                    curtail_i += max(0.0, s2h) + s2b   # only curtail non-negative solar
                    s2b = 0.0
                    b2h = 0.0
                    g2h = l - min(0.0, s2h)        # covers load + any parasitic draw
                    if s2h > 0.0:
                        s2h = 0.0                  # zero positive solar (now curtailed)
                    # s2h < 0 (parasitic) stays — it's the solar balance entry

        sol_to_home_out[i] = s2h
        sol_to_bat_out[i] = s2b
        sol_to_grid_out[i] = s2g
        bat_to_home_out[i] = b2h
        grid_to_home_out[i] = g2h
        grid_to_bat_out[i] = g2b
        curtail_out[i] = curtail_i
        soc_out[i] = soc


def run_sc_with_battery(
    solar: np.ndarray,
    load: np.ndarray,
    battery_spec: BatterySpec,
    soc_init: float,
    interval_hours: float = 5 / 60,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Run self-consumption battery dispatch from an arbitrary starting SOC.

    Unlike `self_consumption()`, this function accepts an explicit starting
    SOC and returns raw numpy arrays rather than a SiteTraces object. It is
    intended for use by coordination components that need to call SC on
    sub-periods (event windows and inter-event gaps) starting from the SOC
    left by a previous dispatch.

    The dispatch logic is identical to `self_consumption()` with a battery:
    solar meets load first, surplus solar charges the battery, residual load
    is met by battery discharge, remainder is grid import. The battery never
    charges from the grid and never exports to the grid. neg_price_aware
    adjustments are not applied.

    Args:
        solar: Gross solar generation per interval, kWh. May be negative
            (inverter parasitic draw). Shape (n,).
        load: Gross household load per interval, kWh. Non-negative. Shape (n,).
        battery_spec: Battery specification.
        soc_init: DC battery SOC at the start of the first interval (kWh).
            Must be within [soc_min, capacity_kwh].
        interval_hours: Interval length in hours. Default 5/60 (5-minute).

    Returns:
        soc_arr: DC battery SOC at the end of each interval, shape (n,).
            Used for natural-flow bound construction over event windows
            and as the SOC handoff at segment boundaries.
        flows: dict of flow arrays, each shape (n,), with keys:
            'sol_to_home', 'sol_to_bat', 'sol_to_grid',
            'bat_to_home', 'grid_to_home', 'grid_to_bat', 'curtail'.
            Used to build SiteTraces segments for non-event gaps.
            'bat_to_grid' is always zero (SC never exports battery to grid)
            and is not included; callers should populate it as zeros.
    """
    n = len(solar)
    sol_to_home = np.zeros(n)
    sol_to_bat  = np.zeros(n)
    sol_to_grid = np.zeros(n)
    bat_to_home = np.zeros(n)
    grid_to_home = np.zeros(n)
    grid_to_bat  = np.zeros(n)
    curtail      = np.zeros(n)
    soc_arr      = np.zeros(n)

    capacity_kwh           = battery_spec.capacity_kwh
    min_soc_kwh            = (battery_spec.backup_reserve_pct / 100.0) * capacity_kwh
    power_per_interval_kwh = battery_spec.power_kw * interval_hours
    charge_eff             = battery_spec.charge_efficiency
    discharge_eff          = battery_spec.discharge_efficiency

    _run_with_battery(
        solar, load,
        sol_to_home, sol_to_bat, sol_to_grid,
        bat_to_home, grid_to_home, grid_to_bat, curtail, soc_arr,
        capacity_kwh, min_soc_kwh, power_per_interval_kwh,
        charge_eff, discharge_eff, soc_init,
    )

    flows = {
        'sol_to_home': sol_to_home,
        'sol_to_bat':  sol_to_bat,
        'sol_to_grid': sol_to_grid,
        'bat_to_home': bat_to_home,
        'grid_to_home': grid_to_home,
        'grid_to_bat': grid_to_bat,
        'curtail':     curtail,
    }
    return soc_arr, flows