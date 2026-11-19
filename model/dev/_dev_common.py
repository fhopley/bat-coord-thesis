"""Shared helpers for development/inspection scripts.

Keeps the hardcoded dev battery spec, the single-site trace loader, and the
block-tier display price builder in one place so the self_consumption,
billing, LP and VPP dev tools all agree. No matplotlib dependency here, so
non-visual tools (billing export) don't pull it in -- build_display_prices
returns plain Series and leaves all plotting to its callers.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from model.core.scenario import BatterySpec
from model.core.prices import (
    apply_critical_rates, block_tier_masks, build_layer_rates,
    build_price_vectors, C_TO_DOLLARS,
)
from model.core.localtime import local_wall_clock
from model.core.tariff import RetailTariff


def build_billing_prices(index, prices_mwh, network_tariff, retail_tariff,
                         state, interval_hours):
    """(p_imp, p_exp) the customer is billed, $/kWh. Display only.

    tou and wholesale_exposed: build_price_vectors with critical rates applied
    as billing applies them (no opposite-side shift). mixed: the retail layer
    of build_layer_rates, because build_price_vectors returns a mixed plan's
    dispatch prices, whose import side is not what the household pays.
    """
    if retail_tariff.type == 'mixed':
        lr = build_layer_rates(index, prices_mwh, network_tariff, retail_tariff,
                               state, interval_hours)
        return (pd.Series(lr.retail_import, index=index, name='p_imp'),
                pd.Series(lr.retail_export, index=index, name='p_exp'))
    p_imp, p_exp = build_price_vectors(index, prices_mwh, network_tariff,
                                       retail_tariff, state, interval_hours)
    return apply_critical_rates(p_imp, p_exp, prices_mwh, retail_tariff,
                                shift_opposite=False)


# Hardcoded battery spec for development runs - Powerwall-like defaults.
# Edit here to change the battery used by ALL dev tools at once.

DEV_BATTERY = BatterySpec(
    capacity_kwh=13.5,
    backup_reserve_pct=10.0,
    power_kw=5.0,
    charge_efficiency=0.95,
    discharge_efficiency=0.95,
    c_deg_per_kwh=0.039,  # 0.005 has been proven to avoid simultaneous charge/discharge for globird 2022
)

# DEV_BATTERY = BatterySpec(
#     capacity_kwh=20,
#     backup_reserve_pct=20.0,
#     power_kw=10.0,
#     charge_efficiency=0.95,
#     discharge_efficiency=0.95,
#     c_deg_per_kwh=0.039,  
# )

def load_site_trace(site_id: str, year: int, paths) -> pd.DataFrame:
    """Load and Wh -> kWh convert a single clean EDP trace file.

    Reads the Parquet file written by the EDP cleaner. Returns a DataFrame
    indexed by datetime_aest with the kWh energy columns present in the file
    (pv_site_net_kwh, gross_load_kwh, ac_load_net_kwh).
    """
    path = paths.edp_clean_dir / str(year) / f'edp_clean_{site_id}_{year}.parquet'
    if not path.exists():
        raise FileNotFoundError(f"Clean trace file not found: {path}")
    df = pd.read_parquet(path).set_index('datetime_aest')
    out = pd.DataFrame(index=df.index)
    for col in ['pv_site_net_wh', 'gross_load_wh', 'ac_load_net_wh']:
        if col in df.columns:
            out[col.replace('_wh', '_kwh')] = df[col].values / 1000.0
    return out


# ---------------------------------------------------------------------------
# Block tier display prices
# ---------------------------------------------------------------------------
#
# Moved here from viz_perfect_foresight_lp_advanced so the VPP viz family can
# use the same logic. Display-only: the LP prices block tiers with daily tier
# variables and billing prices them with a per-day tier walk, so neither has a
# per-interval block price to plot. These helpers reconstruct one after the
# fact from the solved flows.


def marginal_tier_rate(cumulative_kwh: float, tiers) -> float:
    """Return the tier rate (c/kWh) that applies when cumulative daily flow
    has reached cumulative_kwh (i.e. the rate for the next kWh)."""
    prev_threshold = 0.0
    for tier in tiers:
        if cumulative_kwh < tier.threshold_kwh:
            return tier.rate_c_per_kwh
        prev_threshold = tier.threshold_kwh
    # Beyond all tiers: use the last tier's rate
    return tiers[-1].rate_c_per_kwh


def build_display_prices(
    index: pd.DatetimeIndex,
    traces,
    retail_tariff: RetailTariff,
    p_imp_billing: pd.Series,
    p_exp_billing: pd.Series,
    prices_mwh: pd.Series,
    state: str,
    interval_hours: float,
) -> tuple[pd.Series, pd.Series]:
    """Build display price series for viz price panels.

    For non-block-window intervals: use the billing price vectors passed in
    unchanged.

    For block window intervals: replace with the marginal tier rate at the
    start of each interval (computed from cumulative daily flow in `traces`).
    The marginal rate shows which tier the next kWh falls in. An interval
    that is critical on the window's side is left out of the tier, as billing
    does (prices.block_tier_masks): it keeps the billing price passed in,
    which is the critical rate, and its flow does not advance the tier.

    Display prices are illustrative (the LP uses tier variables, not
    per-interval block prices) and are ONLY used in the viz panels. Nothing
    here is ever passed to billing or to an optimiser.

    Args:
        traces: SiteTraces from whichever component produced the dispatch
            being displayed (perfect_foresight_lp_advanced, vpp_sc_event,
            ...). Only import_kwh / export_kwh are read, and they are the
            TOTAL flows -- the same quantity billing tiers on -- so on a VPP
            plan operator-forced event flows correctly consume tier headroom.
        p_imp_billing, p_exp_billing: base price vectors, $/kWh. Callers that
            model a plan with critical rates should have applied them with
            shift_opposite=False before calling, so these show what the
            customer actually pays/earns rather than the LP objective prices.
        state: the site's actual location (scenario.state), used to resolve
            local time for block window membership AND for the daily
            cumulative reset -- see core/localtime.py. Both must use the same
            clock as billing's _block_tier_components, which groups by
            local_wall_clock(...).date.
    """
    imp_display = p_imp_billing.copy()
    exp_display = p_exp_billing.copy()

    if not retail_tariff.has_advanced_features:
        return imp_display, exp_display

    # Local calendar date per interval. local_wall_clock already applies the
    # interval-START convention, so no manual shift is needed here -- and
    # must not be applied, or the tier counter resets on the AEST day
    # boundary rather than the local one (an hour out under NSW/VIC DST,
    # 30-90 minutes out for SA year-round).
    lwc = local_wall_clock(index, state, interval_hours)
    start_dates = lwc.date

    # Tier membership as billed: critical intervals on each side removed.
    imp_masks, exp_masks = block_tier_masks(
        index, retail_tariff, state, interval_hours, prices_mwh, lwc=lwc)

    def _fill(display, flow_arr, windows, masks):
        for window, mask in zip(windows, masks):
            if not mask.any():
                continue
            for date in np.unique(start_dates[mask]):
                day_idx = np.where(mask & (start_dates == date))[0]
                cumulative = 0.0
                for i in day_idx:
                    rate_c = marginal_tier_rate(cumulative, window.tiers)
                    display.iloc[i] = rate_c / C_TO_DOLLARS
                    cumulative += flow_arr[i]

    if retail_tariff.block_import_tiers:
        _fill(imp_display, traces.import_kwh.to_numpy(),
              retail_tariff.block_import_tiers, imp_masks)
    if retail_tariff.block_export_tiers:
        _fill(exp_display, traces.export_kwh.to_numpy(),
              retail_tariff.block_export_tiers, exp_masks)

    return imp_display, exp_display
