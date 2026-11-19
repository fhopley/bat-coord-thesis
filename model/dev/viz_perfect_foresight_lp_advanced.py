"""Interactive visualisation of perfect_foresight_lp_advanced output for a single real site.

Usage (from project root):
    python3 -m model.dev.viz_perfect_foresight_lp_advanced SITE_ID YEAR
    python3 -m model.dev.viz_perfect_foresight_lp_advanced SITE_ID YEAR \\
        --state NSW \\
        --network-tariff ausgrid_ea025_network_2024 \\
        --retail-tariff globird_advanced_tou_2025 \\
        --export-limit-kw 10

Loads the clean EDP trace and NEM price file for one site/year, builds the
LP-specific per-interval price vectors (via build_lp_price_vectors_advanced,
which applies block window masking and critical rates), runs
self_consumption first to obtain a baseline bill and the end-of-period SOC,
then runs perfect_foresight_lp_advanced with:
  - the hardcoded DEV_BATTERY (c_deg defaults to 0.005),
  - SOC initialised at the reserve floor and required to END at the
    self_consumption end-of-period SOC (fair LP-vs-self_consumption comparison),
  - a per-phase export limit (default 10 kW, 1 phase assumed for this dev run).

The visualisation windows, controls, and panel layout are identical to
viz_perfect_foresight_lp. The only visible difference is additional lines in
the verification output for block tiers, the zero-import reward, and critical
rates.

Verification output includes the standard LP checks PLUS:
  - Advanced features present on the loaded tariff
  - Block import/export: daily kWh in each tier (first vs spillover)
  - Zero-import reward: qualifying days vs total days
  - Critical rates: critical intervals, critical-side kWh (and how much of
    it fell in a block window, outside the tier) and its value
  - LP price reconciliation: the per-interval LP-price cost plus the block
    tier cost (effective masks), less the opposite-side shift the LP carries
    but billing does not, must equal the dispatch-dependent part of the bill.
    On a mixed plan the spot term the dispatch import price carries and
    billing does not (max(0, RRP)/1000 x import_lf on imports) is subtracted
    as well.

On a mixed plan the price panels show the dispatch prices the LP optimised
against, not the billed rates: the billed import rate is the plain TOU rate,
and the spot term on top of it is what explains the dispatch.

Two interactive windows: FLOWS and PRICES. See viz_perfect_foresight_lp
docstring for full controls and panel descriptions.
"""
import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model.core.paths import default_paths, Paths
from model.core.scenario import BatterySpec
from model.core.tariff import (
    NetworkTariff, RetailTariff, load_network_tariff, load_retail_tariff,
)
from model.core.prices import (
    build_lp_price_vectors_advanced, build_zero_import_penalty,
    block_window_mask, C_TO_DOLLARS, dispatch_curtail_cost,
)
from model.core.localtime import local_wall_clock
from model.core.site_result import SiteBill
from model.components.self_consumption import self_consumption
from model.components.perfect_foresight_lp import lp_backend
from model.components.perfect_foresight_lp_advanced import perfect_foresight_lp_advanced
from model.components.billing import compute_bill
from model.dev._dev_common import (
    DEV_BATTERY, load_site_trace as _load_site_trace, build_display_prices,
    build_billing_prices,
)

# Re-use all plotting helpers from the standard viz module (identical layout).
from model.dev.viz_perfect_foresight_lp import (
    load_inputs,
    LPAnalysis,
    _recover_curtail,
    _day_slice,
    _reapply_labels,
    _format_time_axis,
    _plot_soc_panel,
    _set_window_title,
    _setup_flows_figure,
    _plot_flows_day,
    _setup_prices_figure,
    _plot_prices_day,
    _price_axis_limits,
    _apply_price_scale,
    _clear_twins,
    _finalise,
    COLOUR_SOLAR, COLOUR_BATTERY, COLOUR_GRID, COLOUR_LOAD, COLOUR_CURTAIL,
    _C_PER_DOLLAR,
)

# Battery coordination mode label shown in chart titles after the retail tariff
# name.  Edit this string to describe the strategy being visualised.
BATTERY_MODE = 'Household Bill Optimised'

# Set True to bias the LP toward charging with solar first and exporting the
# surplus -- produces more realistic per-interval dispatch profiles but adds
# ~3x solve time overhead. Set False to exactly match scenario bulk runs (which
# default to apply_export_order_nudge=False).
APPLY_EXPORT_ORDER_NUDGE = True


# ---------------------------------------------------------------------------
# Display price helpers
# ---------------------------------------------------------------------------
#
# build_display_prices / marginal_tier_rate now live in _dev_common so the
# VPP viz family can share them. _block_tier_band stays here: it is the only
# chart that has room for the shaded tier range.


def _block_tier_band(
    index: pd.DatetimeIndex,
    retail_tariff: RetailTariff,
    interval_hours: float,
    day_start: pd.Timestamp,
    day_end: pd.Timestamp,
    state: str,
) -> dict:
    """Return band data for block windows visible on a given day.

    Returns a dict with keys 'import' and 'export', each a list of
    (start_dt, end_dt, rate_min_c, rate_max_c) tuples for windows that
    overlap the day. The band spans rate_min to rate_max in c/kWh.

    Band edges come from block_window_mask -- the SAME function that places
    the window in the price vectors -- rather than from clock arithmetic on
    day_start. This is not merely tidier: day_start is an AEST timestamp, so
    adding window.start_hour to it yields an AEST wall clock, while the price
    line is placed on LOCAL wall clock. Under DST the two disagree by an hour,
    and the band drifts away from the price step it is supposed to annotate.
    Sharing the mask makes that class of drift impossible by construction,
    and handles the DST-transition days (where a window can be split or
    repeated in local time) without special-casing them.
    """
    result = {'import': [], 'export': []}
    if not retail_tariff.has_advanced_features:
        return result

    delta = pd.Timedelta(hours=interval_hours)
    in_day = (index >= day_start) & (index < day_end)

    def _window_bands(windows, key):
        for window in (windows or []):
            rates = [t.rate_c_per_kwh for t in window.tiers
                     if t.rate_c_per_kwh < 1e6]  # exclude inf placeholder
            if len(rates) < 2:
                continue  # need at least two tiers to show a band
            mask = np.asarray(
                block_window_mask(index, window, state, interval_hours)
            ) & in_day
            pos = np.flatnonzero(mask)
            if pos.size == 0:
                continue
            # Emit one band per contiguous run. Position contiguity implies
            # time contiguity on this regular 5-minute index, and a run break
            # is what a DST-split window looks like from here.
            breaks = np.flatnonzero(np.diff(pos) != 1)
            starts = np.concatenate(([0], breaks + 1))
            ends   = np.concatenate((breaks, [pos.size - 1]))
            for s, e in zip(starts, ends):
                # The price line uses step='post' with interval-END
                # timestamps, so each value is drawn from its timestamp
                # forward. The band therefore runs from the first masked
                # interval-END to one interval past the last.
                result[key].append((
                    index[pos[s]], index[pos[e]] + delta,
                    min(rates), max(rates),
                ))

    _window_bands(retail_tariff.block_import_tiers, 'import')
    _window_bands(retail_tariff.block_export_tiers, 'export')
    return result


# ---------------------------------------------------------------------------
# Data bundle and analysis (advanced variant)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LPAdvancedAnalysis:
    """Equivalent to LPAnalysis from viz_perfect_foresight_lp, plus the
    advanced-tariff price inputs needed for verification."""
    site_id: str
    year: int
    trace: pd.DataFrame
    sc_result: object            # SiteTraces from self_consumption
    lp_result: object            # SiteTraces from perfect_foresight_lp_advanced
    sc_bill: SiteBill
    lp_bill: SiteBill
    p_imp: pd.Series             # $/kWh (LP objective prices; block-masked + critical rates, shifted)
    p_exp: pd.Series             # $/kWh (LP objective prices)
    p_imp_billing: pd.Series     # $/kWh billing prices (critical rates, shift_opposite=False)
    p_exp_billing: pd.Series     # $/kWh billing prices
    p_imp_display: pd.Series     # $/kWh display prices: marginal block tier rate for
                                 #   block window intervals outside critical ones,
                                 #   billing price elsewhere. Used in viz price panels only.
    p_exp_display: pd.Series     # $/kWh display prices (same logic for export windows)
    prices_mwh: pd.Series        # raw $/MWh (for verification reporting)
    curtail: pd.Series           # kWh per interval
    battery_spec: BatterySpec
    network_tariff: NetworkTariff
    retail_tariff: RetailTariff
    export_limit_kw: float
    state: str
    soc_init_kwh: float
    soc_final_kwh: float
    interval_hours: float
    backend: str
    lp_solve_seconds: float


def run_lp_advanced_analysis(
    site_id: str,
    year: int,
    trace: pd.DataFrame,
    prices_mwh: pd.Series,
    network_tariff: NetworkTariff,
    retail_tariff: RetailTariff,
    battery_spec: BatterySpec,
    export_limit_kw: float,
    state: str,
    interval_hours: float = 5 / 60,
    apply_export_order_nudge: bool = True,
) -> LPAdvancedAnalysis:
    """Run self_consumption then perfect_foresight_lp_advanced and bill both.

    Pure computation given inputs (no disk access).

    Args:
        state: the site's actual location (scenario.state), used to resolve
            local time for every clock-driven tariff feature (TOU periods,
            block windows, zero-import reward, export-order nudge). Required
            -- there is no sane default, and passing the wrong state
            silently bills the wrong tariff periods rather than raising, so
            this parameter is deliberately mandatory rather than defaulted
            to 'NSW' or similar.
    """
    solar = trace['pv_site_net_kwh']
    load = trace['gross_load_kwh']

    # Advanced LP price vectors: block masking + critical rates applied.
    # These are the prices the LP optimises against. The critical side takes
    # the critical rate and the opposite side moves by the same delta, so
    # p_imp >= p_exp holds; billing applies the critical rates without the
    # shift.
    p_imp, p_exp = build_lp_price_vectors_advanced(
        solar.index, prices_mwh, network_tariff, retail_tariff, state, interval_hours,
    )

    sc_result = self_consumption(site_id, solar, load, battery_spec, interval_hours)

    soc_min = (battery_spec.backup_reserve_pct / 100.0) * battery_spec.capacity_kwh
    soc_init = soc_min
    soc_final = float(sc_result.battery_soc_kwh.iloc[-1])
    soc_final = min(max(soc_final, soc_min), battery_spec.capacity_kwh)

    export_limit_kwh = export_limit_kw * interval_hours
    _t0 = time.perf_counter()
    lp_result = perfect_foresight_lp_advanced(
        site_id, solar, load, p_imp, p_exp, battery_spec,
        soc_init_kwh=soc_init, soc_final_kwh=soc_final,
        retail_tariff=retail_tariff,
        prices_mwh=prices_mwh,
        state=state,
        export_limit_kwh_per_interval=export_limit_kwh,
        interval_hours=interval_hours,
        apply_export_order_nudge=apply_export_order_nudge,
        curtail_cost_per_kwh_dollars=dispatch_curtail_cost(retail_tariff),
    )
    lp_solve_seconds = time.perf_counter() - _t0

    sc_bill = compute_bill(sc_result, prices_mwh, network_tariff, retail_tariff,
                           state, interval_hours)
    lp_bill = compute_bill(lp_result, prices_mwh, network_tariff, retail_tariff,
                           state, interval_hours)
    curtail = _recover_curtail(solar, lp_result)

    # Billing price vectors: what the customer actually pays/earns -- used
    # for display only.
    p_imp_billing, p_exp_billing = build_billing_prices(
        lp_result.import_kwh.index, prices_mwh, network_tariff, retail_tariff,
        state, interval_hours,
    )

    # A mixed plan displays its dispatch prices (see module docstring).
    p_imp_display, p_exp_display = build_display_prices(
        lp_result.import_kwh.index, lp_result, retail_tariff,
        *((p_imp, p_exp) if retail_tariff.type == 'mixed'
          else (p_imp_billing, p_exp_billing)),
        prices_mwh, state, interval_hours,
    )

    return LPAdvancedAnalysis(
        site_id=site_id, year=year, trace=trace,
        sc_result=sc_result, lp_result=lp_result,
        sc_bill=sc_bill, lp_bill=lp_bill,
        p_imp=p_imp, p_exp=p_exp,
        p_imp_billing=p_imp_billing, p_exp_billing=p_exp_billing,
        p_imp_display=p_imp_display, p_exp_display=p_exp_display,
        prices_mwh=prices_mwh,
        curtail=curtail,
        battery_spec=battery_spec,
        network_tariff=network_tariff,
        retail_tariff=retail_tariff,
        export_limit_kw=export_limit_kw,
        state=state,
        soc_init_kwh=soc_init, soc_final_kwh=soc_final,
        interval_hours=interval_hours,
        backend=lp_backend(),
        lp_solve_seconds=lp_solve_seconds,
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _print_simultaneous_detail(r, charge: np.ndarray, discharge: np.ndarray,
                                tol: float = 1e-6) -> None:
    """Print per-interval detail for simultaneous charge+discharge and
    import+export occurrences, so magnitudes can be assessed (rounding noise
    vs real LP leakage).

    Columns printed:
      timestamp  |  charge kWh  |  discharge kWh  |  import kWh  |  export kWh
    """
    idx = r.import_kwh.index
    imp = r.import_kwh.to_numpy()
    exp = r.export_kwh.to_numpy()

    mask_cd = (charge > tol) & (discharge > tol)
    mask_ie = (imp > tol) & (exp > tol)

    # ---- charge + discharge ------------------------------------------------
    if mask_cd.any():
        rows_cd = []
        for i in np.where(mask_cd)[0]:
            rows_cd.append({
                'timestamp':     idx[i],
                'charge_kwh':    round(float(charge[i]),    9),
                'discharge_kwh': round(float(discharge[i]), 9),
                'import_kwh':    round(float(imp[i]),       9),
                'export_kwh':    round(float(exp[i]),       9),
            })
        df_cd = pd.DataFrame(rows_cd)
        max_c = df_cd['charge_kwh'].max()
        max_d = df_cd['discharge_kwh'].max()
        noise_cd = max(max_c, max_d) < 1e-4
        print(f"\nSimultaneous charge+discharge detail "
              f"({'likely rounding noise' if noise_cd else 'NON-TRIVIAL -- investigate'}):")
        print(f"  max charge={max_c:.3e} kWh  max discharge={max_d:.3e} kWh")
        hdr = (f"  {'Timestamp':<22} {'Charge kWh':>14} {'Discharge kWh':>15} "
               f"{'Import kWh':>12} {'Export kWh':>12}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for row in rows_cd:
            print(f"  {str(row['timestamp']):<22} "
                  f"{row['charge_kwh']:>14.6e} "
                  f"{row['discharge_kwh']:>15.6e} "
                  f"{row['import_kwh']:>12.6e} "
                  f"{row['export_kwh']:>12.6e}")
    else:
        print("  (no simultaneous charge+discharge intervals)")

    # ---- import + export ---------------------------------------------------
    if mask_ie.any():
        rows_ie = []
        for i in np.where(mask_ie)[0]:
            rows_ie.append({
                'timestamp':     idx[i],
                'import_kwh':    round(float(imp[i]),       9),
                'export_kwh':    round(float(exp[i]),       9),
                'charge_kwh':    round(float(charge[i]),    9),
                'discharge_kwh': round(float(discharge[i]), 9),
            })
        df_ie = pd.DataFrame(rows_ie)
        max_i = df_ie['import_kwh'].max()
        max_e = df_ie['export_kwh'].max()
        noise_ie = max(max_i, max_e) < 1e-4
        print(f"\nSimultaneous import+export detail "
              f"({'likely rounding noise' if noise_ie else 'NON-TRIVIAL -- investigate'}):")
        print(f"  max import={max_i:.3e} kWh  max export={max_e:.3e} kWh")
        hdr = (f"  {'Timestamp':<22} {'Import kWh':>14} {'Export kWh':>12} "
               f"{'Charge kWh':>12} {'Discharge kWh':>15}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for row in rows_ie:
            print(f"  {str(row['timestamp']):<22} "
                  f"{row['import_kwh']:>14.6e} "
                  f"{row['export_kwh']:>12.6e} "
                  f"{row['charge_kwh']:>12.6e} "
                  f"{row['discharge_kwh']:>15.6e}")
    else:
        print("  (no simultaneous import+export intervals)")
    print()


def _print_verification_advanced(analysis: LPAdvancedAnalysis) -> None:
    """Print verification stats for the advanced LP run.

    Emits 'FAIL' tokens on any check failure. Extends the standard LP checks
    with advanced-tariff-specific sections.
    """
    a = analysis
    r = a.lp_result
    rt = a.retail_tariff
    solar = a.trace['pv_site_net_kwh']
    load = a.trace['gross_load_kwh']
    cap = a.battery_spec.capacity_kwh
    min_soc = (a.battery_spec.backup_reserve_pct / 100.0) * cap
    tol = 1e-6

    def status(ok: bool) -> str:
        return 'PASS' if ok else 'FAIL'

    print()
    print(f"=== Verification: site {a.site_id} year {a.year} "
          f"(perfect_foresight_lp_advanced) ===")
    T = len(r.import_kwh)
    print(f"--- LP solve timing ---")
    print(f"backend:         {a.backend}")
    print(f"LP solve time:   {a.lp_solve_seconds:.2f}s")
    print(f"T (intervals):   {T:,}")

    # Variable count: 6T standard + tier variables
    from model.components.perfect_foresight_lp_advanced import (
        _build_day_map, _build_block_tier_data,
    )
    n_days, day_map = _build_day_map(r.import_kwh.index, a.state, a.interval_hours)
    n_tv, _, _, _, _ = _build_block_tier_data(
        T, r.import_kwh.index, rt, a.interval_hours, n_days, day_map, a.state)
    print(f"variables:       {6*T + n_tv:,}  (6T + {n_tv} tier vars)")
    print()

    print(f"Battery: {cap} kWh, {a.battery_spec.power_kw} kW, "
          f"reserve {a.battery_spec.backup_reserve_pct}%, "
          f"eff {a.battery_spec.charge_efficiency}/{a.battery_spec.discharge_efficiency}, "
          f"c_deg {a.battery_spec.c_deg_per_kwh}")
    print(f"Export limit: {a.export_limit_kw} kW "
          f"({a.export_limit_kw * a.interval_hours:.4f} kWh/interval)")
    print(f"Network: {a.network_tariff.name} | Retail: {rt.name} ({rt.type})")
    print(f"SOC: init={a.soc_init_kwh:.4f} (floor={min_soc:.4f}), "
          f"final={a.soc_final_kwh:.4f} (= self_consumption end SOC)")
    print()

    # --- Advanced features summary ---
    print("--- Advanced tariff features ---")
    if rt.type == 'mixed':
        print(f"(none -- mixed plan: dispatch import = billed rate + "
              f"max(0, RRP) x import_lf; export credit = "
              f"{rt.wholesale_export_share:g} x {rt.export_share_basis} value, "
              f"floor {rt.export_floor_c_per_kwh} c/kWh)")
    elif not rt.has_advanced_features:
        print("(none -- standard plan; results should match perfect_foresight_lp)")
    else:
        if rt.block_import_tiers:
            for w in rt.block_import_tiers:
                print(f"Block import: [{w.start_hour:02d},{w.end_hour:02d})  "
                      f"{len(w.tiers)} tiers  applies_weekends={w.applies_weekends}")
        if rt.block_export_tiers:
            for w in rt.block_export_tiers:
                print(f"Block export: [{w.start_hour:02d},{w.end_hour:02d})  "
                      f"{len(w.tiers)} tiers  applies_weekends={w.applies_weekends}")
        if rt.zero_import_reward:
            zi = rt.zero_import_reward
            print(f"Zero-import reward: [{zi.start_hour:02d},{zi.end_hour:02d})  "
                  f"${zi.reward_c_per_day/C_TO_DOLLARS:.2f}/day  "
                  f"tol {zi.tolerance_kwh_per_hour} kWh/hr "
                  f"({zi.tolerance_kwh:.4f} kWh/window)")
        if rt.critical_export_rate:
            cr = rt.critical_export_rate
            print(f"Critical export rate: >{cr.threshold_dollars_per_mwh} $/MWh  "
                  f"{cr.rate_c_per_kwh}c/kWh (replaces the export rate)")
        if rt.critical_import_rate:
            cr = rt.critical_import_rate
            print(f"Critical import rate: <{cr.threshold_dollars_per_mwh} $/MWh  "
                  f"{cr.rate_c_per_kwh}c/kWh (replaces the import rate)")
    print()

    # --- Standard flow invariants (identical to the LP version) ---
    print("--- Flow invariants (atol 1e-6) ---")
    load_recon = (r.sol_to_home_kwh + r.bat_to_home_kwh + r.grid_to_home_kwh).to_numpy()
    d = np.abs(load_recon - load.to_numpy()).max()
    print(f"Load balance   max|recon-load|:          {d:.3e}  {status(d < tol)}")

    solar_recon = (r.sol_to_home_kwh + r.sol_to_bat_kwh + r.sol_to_grid_kwh
                   + a.curtail).to_numpy()
    d = np.abs(solar_recon - solar.to_numpy()).max()
    print(f"Solar balance  max|recon+curtail-solar|: {d:.3e}  {status(d < tol)}")

    d = np.abs((r.grid_to_home_kwh + r.grid_to_bat_kwh - r.import_kwh).to_numpy()).max()
    print(f"Import recon   max|g2h+g2b-import|:      {d:.3e}  {status(d < tol)}")

    d = np.abs((r.sol_to_grid_kwh + r.bat_to_grid_kwh - r.export_kwh).to_numpy()).max()
    print(f"Export recon   max|s2g+b2g-export|:      {d:.3e}  {status(d < tol)}")

    neg_ok = True
    for fld in ('sol_to_bat_kwh', 'sol_to_grid_kwh', 'bat_to_home_kwh',
                'bat_to_grid_kwh', 'grid_to_home_kwh', 'grid_to_bat_kwh'):
        mn = getattr(r, fld).min()
        if mn < -tol:
            neg_ok = False
            print(f"  negative flow in {fld}: min={mn:.3e}")
    print(f"Non-negativity (flows except sol_to_home): {status(neg_ok)}")
    print()

    # --- LP constraints ---
    print("--- LP constraints ---")
    soc = r.battery_soc_kwh.to_numpy()
    soc_ok = soc.min() >= min_soc - tol and soc.max() <= cap + tol
    print(f"SOC bounds     min={soc.min():.4f} max={soc.max():.4f} "
          f"[{min_soc:.4f},{cap:.4f}]: {status(soc_ok)}")

    d = abs(float(r.battery_soc_kwh.iloc[-1]) - a.soc_final_kwh)
    print(f"Terminal SOC   |soc[-1]-soc_final|: {d:.3e}  {status(d < tol)}")

    exp_lim = a.export_limit_kw * a.interval_hours
    max_exp = float(r.export_kwh.max())
    print(f"Export limit   max export={max_exp:.4f} <= {exp_lim:.4f}: "
          f"{status(max_exp <= exp_lim + tol)}")

    charge = (r.sol_to_bat_kwh + r.grid_to_bat_kwh).to_numpy()
    discharge = (r.bat_to_home_kwh + r.bat_to_grid_kwh).to_numpy()
    n_simul_cd = int(((charge > tol) & (discharge > tol)).sum())
    n_simul_ie = int(((r.import_kwh.to_numpy() > tol) & (r.export_kwh.to_numpy() > tol)).sum())
    print(f"Simultaneous charge+discharge intervals: {n_simul_cd} (INFO)")
    print(f"Simultaneous import+export intervals:    {n_simul_ie} (INFO)")
    _print_simultaneous_detail(r, charge, discharge, tol)

    pwr = a.battery_spec.power_kw * a.interval_hours
    pwr_ok = charge.max() <= pwr + tol and discharge.max() <= pwr + tol
    print(f"Power limit    max charge={charge.max():.4f} "
          f"max discharge={discharge.max():.4f} <= {pwr:.4f}: {status(pwr_ok)}")
    print()

    # --- Advanced feature verification ---
    if rt.has_advanced_features:
        print("--- Advanced feature verification ---")

        # Block import tiers: report daily window imports per tier
        if rt.block_import_tiers:
            print("Block import tiers:")
            idx = r.import_kwh.index
            # Local wall clock, not the AEST index: the mask below is already
            # state-aware, so grouping on idx.normalize() would sum correctly
            # selected intervals into the wrong daily buckets whenever a local
            # day is not an AEST day. The tier threshold is a per-local-day
            # quantity, so that mis-buckets the tier1/tier2 split at every DST
            # transition, and for SA on every day of the year.
            lwc = local_wall_clock(idx, a.state, a.interval_hours)
            for w in rt.block_import_tiers:
                mask = block_window_mask(idx, w, a.state, a.interval_hours)
                if not mask.any():
                    print(f"  [{w.start_hour:02d},{w.end_hour:02d}): no window intervals found")
                    continue
                window_imp = r.import_kwh.to_numpy()[mask]
                window_dates = lwc.date[mask]
                daily = pd.Series(window_imp).groupby(window_dates).sum()
                t1_cap = w.tiers[0].threshold_kwh if len(w.tiers) > 0 else 0
                in_t1 = daily.clip(upper=t1_cap)
                in_t2 = (daily - t1_cap).clip(lower=0)
                days_with_spillover = int((in_t2 > 1e-4).sum())
                print(f"  [{w.start_hour:02d},{w.end_hour:02d})  "
                      f"daily avg={daily.mean():.3f} kWh  "
                      f"tier1 avg={in_t1.mean():.3f} kWh "
                      f"(cap={t1_cap:.1f})  "
                      f"tier2 spillover on {days_with_spillover}/{len(daily)} days  "
                      f"applies_wknd={w.applies_weekends}")

        # Block export tiers
        if rt.block_export_tiers:
            print("Block export tiers:")
            idx = r.export_kwh.index
            # Local wall clock, not the AEST index -- see the note on the
            # import tiers above.
            lwc = local_wall_clock(idx, a.state, a.interval_hours)
            for w in rt.block_export_tiers:
                mask = block_window_mask(idx, w, a.state, a.interval_hours)
                if not mask.any():
                    print(f"  [{w.start_hour:02d},{w.end_hour:02d}): no window intervals found")
                    continue
                window_exp = r.export_kwh.to_numpy()[mask]
                window_dates = lwc.date[mask]
                daily = pd.Series(window_exp).groupby(window_dates).sum()
                t1_cap = w.tiers[0].threshold_kwh if len(w.tiers) > 0 else 0
                in_t1 = daily.clip(upper=t1_cap)
                in_t2 = (daily - t1_cap).clip(lower=0)
                days_with_spillover = int((in_t2 > 1e-4).sum())
                print(f"  [{w.start_hour:02d},{w.end_hour:02d})  "
                      f"daily avg={daily.mean():.3f} kWh  "
                      f"tier1 avg={in_t1.mean():.3f} kWh "
                      f"(cap={t1_cap:.1f})  "
                      f"tier2 spillover on {days_with_spillover}/{len(daily)} days  "
                      f"applies_wknd={w.applies_weekends}")

        # Zero-import reward: count qualifying days
        if rt.zero_import_reward:
            zi = rt.zero_import_reward
            idx = r.import_kwh.index
            # Local wall clock, not the AEST index. Both the reward window and
            # the day it accrues to are defined in the site's local time, so
            # reading .hour off idx selects the wrong hours on every DST-period
            # day, and for SA year-round (ACST is +9:30). local_wall_clock also
            # does the interval-END -> interval-START shift, and matches the
            # grouping used by billing and by _block_tier_band, so this count
            # cannot drift away from the figure it is printed to cross-check.
            lwc = local_wall_clock(idx, a.state, a.interval_hours)
            local_hour = np.asarray(lwc.hour)
            in_window = (local_hour >= zi.start_hour) & (local_hour < zi.end_hour)
            if in_window.any():
                window_imp = r.import_kwh.to_numpy()[in_window]
                window_dates = lwc.date[in_window]
                daily_win = pd.Series(window_imp).groupby(window_dates).sum()
                threshold = zi.tolerance_kwh if zi.tolerance_kwh > 0 else 1e-9
                qualifying = int((daily_win <= threshold).sum())
                total_days = len(daily_win)
                reward_earned = qualifying * (zi.reward_c_per_day / C_TO_DOLLARS)
                print(f"Zero-import reward [{zi.start_hour:02d},{zi.end_hour:02d}): "
                      f"{qualifying}/{total_days} qualifying days  "
                      f"reward earned ${reward_earned:.2f}  "
                      f"(matches billing: "
                      f"${abs(a.lp_bill.retail_other_credits):.2f})")
                # Verify the reward in the bill matches our count
                bill_reward = abs(a.lp_bill.retail_other_credits)
                reward_ok = abs(reward_earned - bill_reward) < 0.01
                print(f"  Reward reconciliation: {status(reward_ok)}")

        # Critical rates: critical intervals, critical-side kWh, how much of it
        # sat in a block window (billed outside the tier), and its value.
        if rt.has_critical_rates:
            from model.core.prices import critical_masks, critical_rates_dollars
            crit_x, crit_m = critical_masks(a.prices_mwh, rt)
            rho_x, rho_m = critical_rates_dollars(rt)
            idx_r = r.import_kwh.index

            def _in_windows(windows):
                m = np.zeros(len(idx_r), dtype=bool)
                for w in (windows or ()):
                    m |= np.asarray(block_window_mask(idx_r, w, a.state,
                                                      a.interval_hours))
                return m

            for label, crit, rho, flow, win in (
                    ('export', crit_x, rho_x, r.export_kwh.to_numpy(),
                     _in_windows(rt.block_export_tiers)),
                    ('import', crit_m, rho_m, r.import_kwh.to_numpy(),
                     _in_windows(rt.block_import_tiers))):
                if rho is None:
                    continue
                kwh = float(flow[crit].sum())
                kwh_win = float(flow[crit & win].sum())
                print(f"Critical {label}: {int(crit.sum())} intervals  "
                      f"{kwh:.1f} kWh ({kwh_win:.1f} kWh in a block window, "
                      f"outside the tier)  at {rho * C_TO_DOLLARS:.3f}c/kWh "
                      f"= ${kwh * rho:.2f}")

        print()

    # --- Optimisation sanity / VCR direction ---
    print("--- Optimisation sanity ---")
    sc_total = a.sc_bill.household_total
    lp_total = a.lp_bill.household_total
    le_ok = lp_total <= sc_total + 1e-4
    print(f"self_consumption household bill: ${sc_total:,.2f}")
    print(f"advanced LP household bill:      ${lp_total:,.2f}")
    print(f"LP <= self_consumption: {status(le_ok)}  "
          f"(saving ${sc_total - lp_total:,.2f})")

    # LP price reconciliation.
    # The per-interval LP prices (a.p_imp / a.p_exp) differ from billing in two
    # known ways, both added back here so the check is exact:
    #   1. Block windows: the LP price carries only the flat other_rate; the tier
    #      rate is in the LP's tier variables. Billing charges tier + other_rate.
    #      -> add the block tier cost (billing's own tier function, over the
    #      effective masks, so critical intervals are outside the tier).
    #   2. Critical rates: the LP shifts the opposite side of each critical
    #      interval by delta (import during a critical export interval,
    #      export during a critical import interval). Billing does not.
    #      -> subtract sum(imp * delta) over critical export intervals and add
    #      back sum(exp * delta) over critical import intervals.
    # Not included, because they are not in a.p_imp / a.p_exp: the zero-import
    # penalty, the export-order nudge and degradation (LP-only terms), and the
    # zero-import reward (retail_other_credits, excluded from bill_dispatch).
    from model.components.billing import _block_tier_components
    imp_arr = r.import_kwh.to_numpy()
    exp_arr = r.export_kwh.to_numpy()
    lp_price_part = float(np.sum(imp_arr * a.p_imp.to_numpy()
                                 - exp_arr * a.p_exp.to_numpy()))

    from model.core.prices import (
        block_tier_masks, build_base_price_vectors_advanced, critical_masks,
        critical_rates_dollars,
    )
    tier_part = 0.0
    zeros = np.zeros(len(imp_arr))
    imp_masks, exp_masks = block_tier_masks(
        r.import_kwh.index, rt, a.state, a.interval_hours, a.prices_mwh)
    if rt.block_import_tiers:
        _, day_costs = _block_tier_components(
            r.import_kwh, r.import_kwh.index, rt.block_import_tiers, zeros,
            is_import=True, interval_hours=a.interval_hours, state=a.state,
            masks=imp_masks)
        tier_part += float(day_costs.sum())
    if rt.block_export_tiers:
        _, day_costs = _block_tier_components(
            r.export_kwh, r.export_kwh.index, rt.block_export_tiers, zeros,
            is_import=False, interval_hours=a.interval_hours, state=a.state,
            masks=exp_masks)
        tier_part += float(day_costs.sum())

    shift_part = 0.0
    if rt.has_critical_rates:
        b_imp, b_exp = build_base_price_vectors_advanced(
            r.import_kwh.index, a.prices_mwh, a.network_tariff, rt, a.state,
            a.interval_hours)
        crit_x, crit_m = critical_masks(a.prices_mwh, rt)
        rho_x, rho_m = critical_rates_dollars(rt)
        if rho_x is not None:
            delta_x = rho_x - b_exp.to_numpy()[crit_x]
            shift_part += float(np.sum(imp_arr[crit_x] * delta_x))
        if rho_m is not None:
            delta_m = rho_m - b_imp.to_numpy()[crit_m]
            shift_part -= float(np.sum(exp_arr[crit_m] * delta_m))

    # Mixed plan: the dispatch import price carries max(0, RRP)/1000 *
    # import_lf on top of the billed rate (model_spec.md, Mixed-plan dispatch
    # prices).
    spot_part = 0.0
    if rt.type == 'mixed':
        spot_part = float(np.sum(
            np.maximum(0.0, a.prices_mwh.to_numpy()) / 1000.0
            * a.network_tariff.import_loss_factor * imp_arr))

    reconstructed = lp_price_part + tier_part - shift_part - spot_part

    if rt.type == 'wholesale_exposed':
        daily = a.lp_bill.network_daily_charges + a.lp_bill.retail_daily_charges
    else:
        daily = a.lp_bill.retail_daily_charges
    bill_dispatch = lp_total - daily - a.lp_bill.retail_other_credits
    d = abs(reconstructed - bill_dispatch)
    print(f"LP price reconciliation:")
    print(f"  per-interval LP-price cost:   ${lp_price_part:,.4f}")
    print(f"  + block tier cost:            ${tier_part:,.4f}")
    print(f"  - opposite-side shift:        ${shift_part:,.4f}")
    if rt.type == 'mixed':
        print(f"  - mixed spot term on imports: ${spot_part:,.4f}")
    print(f"  = reconstructed:              ${reconstructed:,.4f}")
    print(f"  bill dispatch part:           ${bill_dispatch:,.4f}")
    print(f"  |diff|={d:.3e}  {status(d < 1e-3)}")
    print()

    # --- Annual totals ---
    print("--- Annual totals (kWh) ---")
    print(f"Solar generated:    {solar.sum():>11.1f}")
    print(f"Gross load:         {load.sum():>11.1f}")
    print(f"Solar -> home:      {r.sol_to_home_kwh.sum():>11.1f}")
    print(f"Solar -> battery:   {r.sol_to_bat_kwh.sum():>11.1f}")
    print(f"Solar -> grid:      {r.sol_to_grid_kwh.sum():>11.1f}")
    print(f"Solar curtailed:    {a.curtail.sum():>11.1f}  "
          f"({a.curtail.sum() / solar.sum() * 100 if solar.sum() else 0:.2f}% of solar)")
    print(f"Grid -> battery:    {r.grid_to_bat_kwh.sum():>11.1f}")
    print(f"Battery -> home:    {r.bat_to_home_kwh.sum():>11.1f}")
    print(f"Battery -> grid:    {r.bat_to_grid_kwh.sum():>11.1f}")
    print(f"Grid -> home:       {r.grid_to_home_kwh.sum():>11.1f}")
    print(f"Total import:       {r.import_kwh.sum():>11.1f}")
    print(f"Total export:       {r.export_kwh.sum():>11.1f}")
    self_suff = ((r.sol_to_home_kwh.sum() + r.bat_to_home_kwh.sum())
                 / load.sum() * 100 if load.sum() else 0)
    print(f"Self-sufficiency:   {self_suff:>10.1f}%")
    if rt.zero_import_reward and rt.has_advanced_features:
        zi_cred = a.lp_bill.retail_other_credits
        print(f"Zero-import reward credit: ${zi_cred:.2f} "
              f"(negative = credit to household)")
    print()



# ---------------------------------------------------------------------------
# Advanced prices panel (with block tier bands and marginal display prices)
# ---------------------------------------------------------------------------

def _plot_prices_day_advanced(axes, adv: LPAdvancedAnalysis, day: pd.Timestamp) -> None:
    """Prices panel for the advanced viz.

    Identical to _plot_prices_day except:
      1. Uses p_imp_display / p_exp_display (marginal block tier rates) instead
         of the raw LP objective prices.
      2. Draws a lightly shaded fill_between band spanning the min-to-max tier
         rate range over each block window on the current day. Import band on the
         import panel; export band on the export panel.
    """
    for ax in axes:
        ax.clear()
    _clear_twins(axes[0])
    _clear_twins(axes[1])
    _clear_twins(axes[2])

    # Build a temporary LPAnalysis-shaped object with display prices for _day_slice
    adv_display = LPAnalysis(
        site_id=adv.site_id, year=adv.year, trace=adv.trace,
        sc_result=adv.sc_result, lp_result=adv.lp_result,
        sc_bill=adv.sc_bill, lp_bill=adv.lp_bill,
        p_imp=adv.p_imp_display, p_exp=adv.p_exp_display,
        curtail=adv.curtail,
        battery_spec=adv.battery_spec,
        network_tariff=adv.network_tariff,
        retail_tariff=adv.retail_tariff,
        export_limit_kw=adv.export_limit_kw,
        state=adv.state,
        soc_init_kwh=adv.soc_init_kwh,
        soc_final_kwh=adv.soc_final_kwh,
        interval_hours=adv.interval_hours,
        backend=adv.backend,
        lp_solve_seconds=adv.lp_solve_seconds,
    )

    t, d, soc, p_imp_c, p_exp_c, day_start, day_end, n = _day_slice(adv_display, day)
    if n == 0:
        for ax in axes:
            ax.text(0.5, 0.5, f"No data for {day.normalize().date()}",
                    transform=ax.transAxes, ha='center', va='center')
        _finalise(axes[0].figure, day_start, adv.retail_tariff.name)
        return

    # Compute block tier bands for this day
    bands = _block_tier_band(
        adv.lp_result.import_kwh.index, adv.retail_tariff,
        adv.interval_hours, day_start, day_end, adv.state,
    )

    # Both price axes share one transform and one set of limits, so that a
    # given height means the same c/kWh in both panes. The tier band rates are
    # included alongside the price series: a band is drawn on the price axis in
    # c/kWh, so a tier rate above the day's realised prices would otherwise be
    # clipped out of view -- and a band clipped to invisibility looks identical
    # to a window with no tier structure at all.
    _band_rates = [r for key in ('export', 'import')
                   for (_, _, r_min, r_max) in bands[key]
                   for r in (r_min, r_max)]
    p_lo, p_hi = _price_axis_limits(p_exp_c, p_imp_c, np.asarray(_band_rates))

    def _draw_tier_band(ax_price, day_bands, colour):
        """Draw a lightly shaded rectangle spanning the block window time range
        and the tier rate band (min to max c/kWh). Uses fill_between with explicit
        y bounds so only the intersection rectangle is shaded -- not the full
        panel width or full panel height separately.

        Rates from _block_tier_band are already in c/kWh -- no conversion needed.
        """
        for (b_start, b_end, r_min, r_max) in day_bands:
            if r_min == r_max:
                continue  # single-rate window, no range to shade
            x = [mdates.date2num(b_start), mdates.date2num(b_end)]
            ax_price.fill_between(
                x, r_min, r_max,
                step=None, alpha=0.12, color=colour,
                zorder=0, linewidth=0,
            )

    # Panel 1: Export vs export price
    e1 = d['s2g']
    e2 = d['s2g'] + d['b2g']
    axes[0].fill_between(t, 0, e1, step='post', color=COLOUR_SOLAR, label='sol_to_grid')
    axes[0].fill_between(t, e1, e2, step='post', color=COLOUR_BATTERY, label='bat_to_grid')
    axes[0].step(t, e2, where='post', color=COLOUR_BATTERY, linewidth=1.5,
                 label='total export')
    axes[0].set_ylim(0, max(float(np.max(e2)) if len(e2) else 0, 0.1) * 1.1)
    axes[0].grid(True, alpha=0.3)
    axp0 = axes[0].twinx()
    _draw_tier_band(axp0, bands['export'], COLOUR_GRID)
    axp0.step(t, p_exp_c, where='post', color=COLOUR_GRID, linestyle=':',
              linewidth=1.6, label='export price (marginal)')
    _apply_price_scale(axp0, p_lo, p_hi)
    axp0.set_ylabel('Export price (c/kWh)')
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = axp0.get_legend_handles_labels()
    axp0.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8)

    # Panel 2: Import vs import price
    i1 = d['g2h']
    i2 = d['g2h'] + d['g2b']
    axes[1].fill_between(t, 0, i1, step='post', color=COLOUR_LOAD, label='grid_to_home')
    axes[1].fill_between(t, i1, i2, step='post', color=COLOUR_BATTERY, label='grid_to_bat')
    axes[1].step(t, i2, where='post', color=COLOUR_GRID, linewidth=1.5,
                 label='total import')
    axes[1].set_ylim(0, max(float(np.max(i2)) if len(i2) else 0, 0.1) * 1.1)
    axes[1].grid(True, alpha=0.3)
    axp1 = axes[1].twinx()
    _draw_tier_band(axp1, bands['import'], COLOUR_GRID)
    axp1.step(t, p_imp_c, where='post', color=COLOUR_GRID, linestyle=':',
              linewidth=1.6, label='import price (marginal)')
    _apply_price_scale(axp1, p_lo, p_hi)
    axp1.set_ylabel('Import price (c/kWh)')
    h1, l1 = axes[1].get_legend_handles_labels()
    h2, l2 = axp1.get_legend_handles_labels()
    axp1.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8)

    # Panel 3: SOC + charge/discharge (unchanged)
    _plot_soc_panel(axes[2], t, soc, d, adv.battery_spec, adv.interval_hours)
    axes[2].grid(True, alpha=0.3)
    _format_time_axis(axes[2], t, day_start, day_end)

    _reapply_labels(
        axes,
        ['Export vs Export Price', 'Import vs Import Price',
         'Battery SOC + Charge/Discharge'],
        ['Export (kW)', 'Import (kW)', 'SOC (kWh)'],
    )
    _finalise(axes[0].figure, day_start, adv.retail_tariff.name)


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------

def _interactive_loop_advanced(adv: LPAdvancedAnalysis) -> None:
    """Interactive day navigator for the advanced viz.

    Identical to _interactive_loop except the prices panel calls
    _plot_prices_day_advanced (marginal block tier display prices + tier bands)
    rather than the standard _plot_prices_day. The flows panel is unchanged.
    """
    # Build a standard LPAnalysis for the flows window (uses LP objective prices
    # and the standard _plot_flows_day, which doesn't show price information).
    a_std = LPAnalysis(
        site_id=adv.site_id,
        year=adv.year,
        trace=adv.trace,
        sc_result=adv.sc_result,
        lp_result=adv.lp_result,
        sc_bill=adv.sc_bill,
        lp_bill=adv.lp_bill,
        p_imp=adv.p_imp,
        p_exp=adv.p_exp,
        curtail=adv.curtail,
        battery_spec=adv.battery_spec,
        network_tariff=adv.network_tariff,
        retail_tariff=adv.retail_tariff,
        export_limit_kw=adv.export_limit_kw,
        state=adv.state,
        soc_init_kwh=adv.soc_init_kwh,
        soc_final_kwh=adv.soc_final_kwh,
        interval_hours=adv.interval_hours,
        backend=adv.backend,
        lp_solve_seconds=adv.lp_solve_seconds,
    )
    fig_flows,  axes_flows  = _setup_flows_figure(adv.site_id)
    fig_prices, axes_prices = _setup_prices_figure(adv.site_id)
    plt.ion()

    first_day = adv.trace.index[0].normalize()
    last_day  = adv.trace.index[-1].normalize()
    current   = first_day

    _plot_flows_day(axes_flows, a_std, current)
    _plot_prices_day_advanced(axes_prices, adv, current)
    plt.show()
    plt.pause(0.1)

    print(f"Days: {first_day.date()} to {last_day.date()}")
    print("Controls: Enter=next, b=back, YYYY-MM-DD=jump, q=quit "
          "(two windows update together)")

    while True:
        try:
            raw = input(f"[{current.date()}] > ").strip().lower()
        except EOFError:
            break
        if raw == 'q':
            break
        elif raw == 'b':
            current = max(first_day, current - pd.Timedelta(days=1))
        elif raw in ('', 'n'):
            current = min(last_day, current + pd.Timedelta(days=1))
        else:
            try:
                current = pd.Timestamp(raw).tz_localize(adv.trace.index.tz).normalize()
                if current < first_day or current > last_day:
                    print(f"(out of range {first_day.date()}..{last_day.date()})")
                    current = max(first_day, min(last_day, current))
            except Exception as e:
                print(f"  (unrecognised input: {e})")
                continue

        _plot_flows_day(axes_flows, a_std, current)
        _plot_prices_day_advanced(axes_prices, adv, current)
        fig_flows.canvas.draw_idle()
        fig_prices.canvas.draw_idle()
        plt.pause(0.05)

    plt.close(fig_flows)
    plt.close(fig_prices)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('site_id', help='Site identifier, e.g. S0008')
    parser.add_argument('year', type=int, help='Year, e.g. 2022')
    parser.add_argument('--state', default='NSW',
                        help='NEM region for prices (default NSW)')
    parser.add_argument('--network-tariff', default='ausgrid_ea025_network_2024')
    parser.add_argument('--retail-tariff', default='globird_advanced_tou_2025',
                        help='Retail tariff ID (default: globird_advanced_tou_2025)')
    parser.add_argument('--export-limit-kw', type=float, default=10.0,
                        help='Per-phase export limit kW, 1 phase assumed (default 10)')
    parser.add_argument('--project-root', type=Path, default=None)
    args = parser.parse_args(argv)

    paths = default_paths(args.project_root)
    trace, prices_mwh, network_tariff, retail_tariff = load_inputs(
        args.site_id, args.year, args.state,
        args.network_tariff, args.retail_tariff, paths,
    )
    for col in ('pv_site_net_kwh', 'gross_load_kwh'):
        if col not in trace.columns:
            print(f"ERROR: trace missing {col}", file=sys.stderr)
            return 1

    analysis = run_lp_advanced_analysis(
        args.site_id, args.year, trace, prices_mwh,
        network_tariff, retail_tariff, DEV_BATTERY,
        export_limit_kw=args.export_limit_kw,
        state=args.state,
        apply_export_order_nudge=APPLY_EXPORT_ORDER_NUDGE,
    )
    _print_verification_advanced(analysis)
    _interactive_loop_advanced(analysis)
    return 0


if __name__ == '__main__':
    sys.exit(main())