"""Tests for model.components.perfect_foresight_lp_advanced.

Style mirrors test_perfect_foresight_lp.py: interval_hours=1.0 so kWh/interval
equals kW; battery parameters generous unless a test targets a specific
constraint; all expected values analytically derivable by hand.

Test groups:
  1. Standard plan equivalence: advanced matches standard for non-advanced tariffs.
  2. Block import tiers: LP fills free tier, spills correctly into paid tier.
  3. Block export tiers: LP fills high-reward tier first.
  4. Zero-import reward: penalty steers LP away from window imports when feasible.
  5. Critical export rate: LP exports at critical intervals; critical flow
     does not use up the block export tier.
  6. Critical import rate: LP imports during negative prices.
  7. p_exp clamping: p_exp > p_imp does not raise; LP remains bounded.
  8. SiteTraces invariants: all four invariants hold for advanced dispatch.
  9. Validation error paths.
"""
import math

import numpy as np
import pandas as pd
import pytest

from model.core.scenario import BatterySpec
from model.core.tariff import (
    NetworkTariff, RetailTariff, TariffPeriod,
    BlockTier, BlockTariffWindow, ZeroImportReward, CriticalRate,
)
from model.components.perfect_foresight_lp import perfect_foresight_lp
from model.components.perfect_foresight_lp_advanced import (
    perfect_foresight_lp_advanced,
    _build_day_map,
    _build_block_tier_data,
)
from model.core.prices import build_price_vectors, build_lp_price_vectors_advanced


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

IH = 1.0   # interval_hours -- simplifies kW = kWh/interval


def _idx(n: int) -> pd.DatetimeIndex:
    """n-interval hourly index starting 2022-07-04 01:00 (Monday, winter).

    Deliberately a winter date (no NSW/VIC DST active) so that the hour
    boundaries in this file's tariff/block-window fixtures -- written
    assuming AEST hour-of-day == local hour-of-day -- test LP/tariff logic
    without also being sensitive to the DST offset. DST-specific behaviour
    of local_wall_clock() itself is covered separately in
    test_localtime.py, not re-tested here. This file previously used a
    January (DST-season) date; that made every block-window/TOU-boundary
    test silently depend on NSW's DST offset once tariff period lookups
    switched to real local time, which is not what these tests are
    actually about.
    """
    return pd.date_range('2022-07-04 01:00:00', periods=n, freq='60min',
                         tz='Etc/GMT-10')


def _s(vals, idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(vals, index=idx, dtype=float)


def _bspec(
    capacity: float = 10.0,
    reserve_pct: float = 0.0,
    power: float = 10.0,
    eta_c: float = 1.0,
    eta_d: float = 1.0,
    c_deg: float = 0.0,
) -> BatterySpec:
    return BatterySpec(
        capacity_kwh=capacity,
        backup_reserve_pct=reserve_pct,
        power_kw=power,
        charge_efficiency=eta_c,
        discharge_efficiency=eta_d,
        c_deg_per_kwh=c_deg,
    )


def _network() -> NetworkTariff:
    """Loss-factor-free network: all rates 0, loss factors 1. Keeps price
    vector construction simple so test arithmetic is only about the retail layer."""
    return NetworkTariff(
        name='N', state='NSW',
        usage_tariffs=(TariffPeriod(0, 24, 0.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 0.0),),
        daily_c_per_day=0.0,
        usage_dlf=1.0, export_dlf=1.0, import_tlf=1.0, export_tlf=1.0,
    )


def _flat_rt(usage_c: float = 20.0, feedin_c: float = 5.0) -> RetailTariff:
    """Standard TOU plan with no advanced features."""
    return RetailTariff(
        name='flat', state='NSW', type='tou', daily_c_per_day=0.0,
        usage_tariffs=(TariffPeriod(0, 24, usage_c),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, feedin_c),),
        usage_other_c_per_kwh=0.0,
        feedin_other_c_per_kwh=0.0,
    )


def _run_advanced(
    solar, load, retail_tariff, prices_mwh,
    bspec=None, soc_init=0.0, soc_final=0.0,
    export_limit=np.inf, site_id='S0001',
):
    """Run the advanced LP with price vectors built from the given tariff."""
    n = len(solar)
    idx = _idx(n)
    bspec = bspec or _bspec()
    net = _network()
    p_imp, p_exp = build_lp_price_vectors_advanced(
        idx, _s(prices_mwh, idx), net, retail_tariff, 'NSW', IH,
    )
    return perfect_foresight_lp_advanced(
        site_id,
        _s(solar, idx), _s(load, idx),
        p_imp, p_exp, bspec,
        soc_init_kwh=soc_init,
        soc_final_kwh=soc_final,
        retail_tariff=retail_tariff,
        prices_mwh=_s(prices_mwh, idx),
        export_limit_kwh_per_interval=export_limit,
        interval_hours=IH,
        state='NSW',)


def _run_standard(solar, load, p_imp_vals, p_exp_vals, bspec=None,
                  soc_init=0.0, soc_final=0.0, export_limit=np.inf):
    """Run the standard LP with explicit price vectors."""
    n = len(solar)
    idx = _idx(n)
    bspec = bspec or _bspec()
    return perfect_foresight_lp(
        'S0001',
        _s(solar, idx), _s(load, idx),
        _s(p_imp_vals, idx), _s(p_exp_vals, idx),
        bspec,
        soc_init_kwh=soc_init,
        soc_final_kwh=soc_final,
        export_limit_kwh_per_interval=export_limit,
        interval_hours=IH,
    )


# ---------------------------------------------------------------------------
# 1. Standard plan equivalence
# ---------------------------------------------------------------------------

def test_standard_plan_equivalence_no_solar():
    """Advanced matches standard for a plain TOU plan (no solar)."""
    n = 6
    idx = _idx(n)
    solar = [0.] * n
    load = [1., 1., 1., 1., 1., 1.]
    prices_mwh = [100., 100., 400., 100., 100., 100.]
    rt = _flat_rt(usage_c=20.0, feedin_c=5.0)
    net = _network()
    p_imp, p_exp = build_price_vectors(idx, _s(prices_mwh, idx), net, rt, 'NSW', IH)

    tr_std = perfect_foresight_lp(
        'S', _s(solar, idx), _s(load, idx),
        p_imp, p_exp, _bspec(),
        soc_init_kwh=0., soc_final_kwh=0., interval_hours=IH,
    )
    tr_adv = perfect_foresight_lp_advanced(
        'S', _s(solar, idx), _s(load, idx),
        p_imp, p_exp, _bspec(),
        soc_init_kwh=0., soc_final_kwh=0.,
        retail_tariff=rt, prices_mwh=_s(prices_mwh, idx),
        interval_hours=IH,
        state='NSW',)
    np.testing.assert_allclose(
        tr_adv.import_kwh.to_numpy(), tr_std.import_kwh.to_numpy(), atol=1e-6)
    np.testing.assert_allclose(
        tr_adv.export_kwh.to_numpy(), tr_std.export_kwh.to_numpy(), atol=1e-6)
    np.testing.assert_allclose(
        tr_adv.battery_soc_kwh.to_numpy(), tr_std.battery_soc_kwh.to_numpy(), atol=1e-6)


def test_standard_plan_equivalence_with_solar_and_arbitrage():
    """Advanced matches standard even with solar surplus and price arbitrage."""
    solar =  [3., 0., 0., 2., 0., 0.]
    load  =  [1., 2., 1., 1., 2., 1.]
    prices = [100., 500., 100., 100., 500., 100.]
    rt = _flat_rt()
    idx = _idx(6)
    net = _network()
    p_imp, p_exp = build_price_vectors(idx, _s(prices, idx), net, rt, 'NSW', IH)

    tr_std = perfect_foresight_lp(
        'S', _s(solar, idx), _s(load, idx),
        p_imp, p_exp, _bspec(capacity=4., power=3.),
        soc_init_kwh=0., soc_final_kwh=0., interval_hours=IH,
    )
    tr_adv = perfect_foresight_lp_advanced(
        'S', _s(solar, idx), _s(load, idx),
        p_imp, p_exp, _bspec(capacity=4., power=3.),
        soc_init_kwh=0., soc_final_kwh=0.,
        retail_tariff=rt, prices_mwh=_s(prices, idx),
        interval_hours=IH,
        state='NSW',)
    np.testing.assert_allclose(
        tr_adv.import_kwh.to_numpy(), tr_std.import_kwh.to_numpy(), atol=1e-6)


# ---------------------------------------------------------------------------
# 2. Block import tiers
# ---------------------------------------------------------------------------

def _rt_block_import(start=8, end=10, t1_kwh=2.0, t1_rate=0.0,
                      t2_rate=30.0, outside_c=20.0) -> RetailTariff:
    """TOU plan with a two-tier import window [start, end).
    Usage rate is 0 inside the window (replaced by block tiers).
    """
    periods = []
    if start > 0:
        periods.append(TariffPeriod(0, start, outside_c))
    periods.append(TariffPeriod(start, end, 0.0))
    if end < 24:
        periods.append(TariffPeriod(end, 24, outside_c))
    return RetailTariff(
        name='BlockImp', state='NSW', type='tou', daily_c_per_day=0.0,
        usage_tariffs=tuple(periods),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 0.0),),
        usage_other_c_per_kwh=0.0,
        feedin_other_c_per_kwh=0.0,
        block_import_tiers=(BlockTariffWindow(
            start_hour=start, end_hour=end,
            tiers=(
                BlockTier(threshold_kwh=t1_kwh, rate_c_per_kwh=t1_rate),
                BlockTier(threshold_kwh=math.inf, rate_c_per_kwh=t2_rate),
            ),
        ),),
    )


def test_block_import_fills_free_tier_exactly():
    """LP should import exactly up to the free tier limit (t1_kwh=2) during
    the block window when the alternative is paying the outside rate.

    Setup (4 intervals, 1-hour each):
      t=0: outside window (8c/kWh outside, but we use 0 load so irrelevant)
      t=1: window hour 8 -- first window interval
      t=2: window hour 9 -- second window interval
      t=3: outside window

    Load = 0 everywhere. Battery starts and ends at soc_min=0.
    The LP should charge up to 2 kWh during the window (free tier) to offset
    later discharge. But with no load and no export reward, the LP has no
    incentive to cycle -- it should import nothing and do nothing.

    Reshape to make the incentive clear: add load at t=3 (expensive hour).
    """
    # T=4: hour 7, 8, 9, 10 (index starts at 01:00, so hours are 1,2,3,4 of day)
    # Window [2, 4) in terms of hour-of-day. Build index starting at hour 2.
    n = 4
    idx = pd.date_range('2022-07-04 02:00:00', periods=n, freq='60min',
                        tz='Etc/GMT-10')
    # Hours 2,3 are in window [2,4); hours 4,5 are outside.
    rt = RetailTariff(
        name='BI', state='NSW', type='tou', daily_c_per_day=0.0,
        usage_tariffs=(
            TariffPeriod(0, 2, 20.0),
            TariffPeriod(2, 4, 0.0),    # window: replaced by block tiers
            TariffPeriod(4, 24, 20.0),
        ),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 0.0),),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
        block_import_tiers=(BlockTariffWindow(
            start_hour=2, end_hour=4,
            tiers=(BlockTier(2.0, 0.0), BlockTier(math.inf, 30.0)),
        ),),
    )
    net = _network()
    prices_mwh = pd.Series([100., 100., 100., 100.], index=idx)
    p_imp, p_exp = build_lp_price_vectors_advanced(idx, prices_mwh, net, rt, 'NSW', IH)

    # solar=0, load=[0,0,0,2]: load arrives after window.
    # LP should charge 2 kWh (free) during window, discharge 2 kWh at t=3.
    # Net cost: 0 (free tier) vs 2*20c/100 = 40c (import from grid at t=3).
    # Charge is free in the free tier even though p_imp[window]=0 already;
    # the LP sees p_imp[window]=0 and will import.
    solar = [0., 0., 0., 0.]
    load  = [0., 0., 0., 2.]
    bspec = _bspec(capacity=4., power=4.)
    tr = perfect_foresight_lp_advanced(
        'S', _s(solar, idx), _s(load, idx),
        p_imp, p_exp, bspec,
        soc_init_kwh=0., soc_final_kwh=0.,
        retail_tariff=rt, prices_mwh=prices_mwh,
        interval_hours=IH,
        state='NSW',)
    # Total import in window should be exactly 2 kWh (no incentive to exceed free tier)
    from model.core.prices import block_window_mask
    mask = block_window_mask(idx, rt.block_import_tiers[0], 'NSW', IH)
    window_import = float(tr.import_kwh.to_numpy()[mask].sum())
    assert window_import == pytest.approx(2.0, abs=1e-4)
    # No import at t=3 (battery covers load)
    np.testing.assert_allclose(tr.import_kwh.iloc[-1], 0., atol=1e-4)


def test_block_import_lp_avoids_expensive_tier2_when_possible():
    """LP uses the free tier (tier1) when available and avoids the expensive
    tier2 (30c) by pre-charging from outside the window (20c < 30c).

    2 intervals: t=0 outside window (20c), t=1 inside window [1,2).
    Load=0 everywhere. Battery starts at 0; must end at 1 kWh (soc_final=1).
    Options:
      A) Import 1 kWh at t=0 (20c/kWh) and charge battery.
      B) Import 1 kWh at t=1 (window: 0c for first 2 kWh in free tier).
    Option B is cheaper (0c < 20c), so LP should import in window.
    This verifies the free tier is correctly cheaper than the outside rate.
    """
    n = 2
    idx = pd.date_range('2022-07-04 01:00:00', periods=n, freq='60min',
                        tz='Etc/GMT-10')
    rt = RetailTariff(
        name='BI2', state='NSW', type='tou', daily_c_per_day=0.0,
        usage_tariffs=(
            TariffPeriod(0, 1, 20.0),   # outside window: 20c
            TariffPeriod(1, 2, 0.0),    # window: replaced by block tiers
            TariffPeriod(2, 24, 20.0),
        ),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 0.0),),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
        block_import_tiers=(BlockTariffWindow(
            start_hour=1, end_hour=2,
            tiers=(BlockTier(2.0, 0.0), BlockTier(math.inf, 30.0)),
        ),),
    )
    net = _network()
    prices_mwh = pd.Series([100., 100.], index=idx)
    p_imp, p_exp = build_lp_price_vectors_advanced(idx, prices_mwh, net, rt, 'NSW', IH)

    solar = [0., 0.]
    load  = [0., 0.]
    bspec = _bspec(capacity=2., power=2.)
    tr = perfect_foresight_lp_advanced(
        'S', _s(solar, idx), _s(load, idx),
        p_imp, p_exp, bspec,
        soc_init_kwh=0., soc_final_kwh=1.,   # must charge 1 kWh
        retail_tariff=rt, prices_mwh=prices_mwh,
        interval_hours=IH,
        state='NSW',)
    from model.core.prices import block_window_mask
    mask = block_window_mask(idx, rt.block_import_tiers[0], 'NSW', IH)
    # LP should charge during window (free tier) not before it (20c)
    assert float(tr.import_kwh.to_numpy()[mask].sum()) == pytest.approx(1.0, abs=1e-4)
    assert float(tr.import_kwh.to_numpy()[~mask].sum()) == pytest.approx(0.0, abs=1e-4)


def test_block_import_prefers_free_tier_over_outside_window():
    """When both window (free up to tier1) and outside (20c) can satisfy load,
    LP should prefer to import in the window (cheaper) up to the free tier limit.

    4 intervals at hours [8,9,10,11]. Window [8,10) = intervals 0 and 1.
    Load = 1 kWh at interval 2 (outside window, 20c/kWh).
    Battery can carry energy forward.
    LP should import 1 kWh at t=0 (free) and discharge at t=2 (saving 20c).
    """
    n = 4
    idx = pd.date_range('2022-07-04 08:00:00', periods=n, freq='60min',
                        tz='Etc/GMT-10')
    rt = RetailTariff(
        name='BIPref', state='NSW', type='tou', daily_c_per_day=0.0,
        usage_tariffs=(
            TariffPeriod(0, 8, 20.0), TariffPeriod(8, 10, 0.0),
            TariffPeriod(10, 24, 20.0),
        ),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 0.0),),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
        block_import_tiers=(BlockTariffWindow(
            start_hour=8, end_hour=10,
            tiers=(BlockTier(2.0, 0.0), BlockTier(math.inf, 30.0)),
        ),),
    )
    net = _network()
    prices_mwh = pd.Series([100.] * n, index=idx)
    p_imp, p_exp = build_lp_price_vectors_advanced(idx, prices_mwh, net, rt, 'NSW', IH)

    solar = [0., 0., 0., 0.]
    load  = [0., 0., 1., 0.]   # load after window
    bspec = _bspec(capacity=2., power=2.)
    tr = perfect_foresight_lp_advanced(
        'S', _s(solar, idx), _s(load, idx),
        p_imp, p_exp, bspec,
        soc_init_kwh=0., soc_final_kwh=0.,
        retail_tariff=rt, prices_mwh=prices_mwh,
        interval_hours=IH,
        state='NSW',)
    # LP should import 1 kWh in window (free), discharge at t=2, no outside import
    from model.core.prices import block_window_mask
    mask = block_window_mask(idx, rt.block_import_tiers[0], 'NSW', IH)
    window_import = float(tr.import_kwh.to_numpy()[mask].sum())
    outside_import = float(tr.import_kwh.to_numpy()[~mask].sum())
    assert window_import == pytest.approx(1.0, abs=1e-4)
    assert outside_import == pytest.approx(0.0, abs=1e-4)


def test_block_import_two_days_each_gets_free_tier():
    """LP gets a fresh free tier (2 kWh) on each day. On each day it should
    prefer to import during the free window rather than the 20c outside window.

    4 intervals: 2 days * 2 intervals/day (hours [1,3) each day).
    Day 1: interval at hour 1 (outside), interval at hour 2 (window).
    Day 2: same pattern.
    Load=0; soc_final = 2*1 = 2 kWh (LP must charge 1 kWh per day).
    LP should charge during each day's free window interval.
    """
    n = 4
    # Start at hour 2 so that the window [2,3) captures intervals 0 and 2
    # (alternating out/in for two days).
    idx = pd.date_range('2022-07-04 02:00:00', periods=n, freq='60min',
                        tz='Etc/GMT-10')
    # start_instants for 4 intervals: hours 1,2,3,4 -> window [2,3) = intervals 1,3
    rt = RetailTariff(
        name='BI2D', state='NSW', type='tou', daily_c_per_day=0.0,
        usage_tariffs=(
            TariffPeriod(0, 2, 20.0), TariffPeriod(2, 3, 0.0),
            TariffPeriod(3, 24, 20.0),
        ),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 0.0),),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
        block_import_tiers=(BlockTariffWindow(
            start_hour=2, end_hour=3,
            tiers=(BlockTier(2.0, 0.0), BlockTier(math.inf, 30.0)),
        ),),
    )
    net = _network()
    prices_mwh = pd.Series([100.] * n, index=idx)
    p_imp, p_exp = build_lp_price_vectors_advanced(idx, prices_mwh, net, rt, 'NSW', IH)

    solar = [0., 0., 0., 0.]
    load  = [0., 0., 0., 0.]
    bspec = _bspec(capacity=4., power=4.)
    tr = perfect_foresight_lp_advanced(
        'S', _s(solar, idx), _s(load, idx),
        p_imp, p_exp, bspec,
        soc_init_kwh=0., soc_final_kwh=2.,   # must end at 2 kWh
        retail_tariff=rt, prices_mwh=prices_mwh,
        interval_hours=IH,
        state='NSW',)
    from model.core.prices import block_window_mask
    mask = block_window_mask(idx, rt.block_import_tiers[0], 'NSW', IH)
    # LP should import all 2 kWh in the free window intervals
    assert float(tr.import_kwh.to_numpy()[mask].sum()) == pytest.approx(2.0, abs=1e-4)
    assert float(tr.import_kwh.to_numpy()[~mask].sum()) == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# 3. Block export tiers
# ---------------------------------------------------------------------------

def _rt_block_export(start=14, end=16, t1_kwh=2.0,
                      t1_rate=25.0, t2_rate=5.0,
                      outside_feedin=5.0) -> RetailTariff:
    """TOU plan with a two-tier export window [start, end).
    Feedin rate is 0 inside the window (replaced by block tiers).
    """
    feedin_periods = []
    if start > 0:
        feedin_periods.append(TariffPeriod(0, start, outside_feedin))
    feedin_periods.append(TariffPeriod(start, end, 0.0))
    if end < 24:
        feedin_periods.append(TariffPeriod(end, 24, outside_feedin))
    return RetailTariff(
        name='BlockExp', state='NSW', type='tou', daily_c_per_day=0.0,
        usage_tariffs=(TariffPeriod(0, 24, 20.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=tuple(feedin_periods),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
        block_export_tiers=(BlockTariffWindow(
            start_hour=start, end_hour=end,
            tiers=(
                BlockTier(threshold_kwh=t1_kwh, rate_c_per_kwh=t1_rate),
                BlockTier(threshold_kwh=math.inf, rate_c_per_kwh=t2_rate),
            ),
        ),),
    )


def test_block_export_fills_high_reward_tier_first():
    """LP should export up to the high-reward tier limit (2 kWh) before spilling.

    Window [14,16). 2 intervals. Solar=2 each interval. Load=0. soc_init/final=0.
    Battery charges from solar at t<14, discharges into export during window.
    Tier1: first 2 kWh at 25c (high reward). Tier2: excess at 5c.
    LP should export exactly 2 kWh total in window (fills tier1 exactly),
    not more, since exporting beyond 2 kWh earns only 5c vs outside 5c.
    """
    n = 6
    # Hours 12-17; window [14,16) = intervals at 15:00 and 16:00 (start_instant 14:xx)
    idx = pd.date_range('2022-07-04 13:00:00', periods=n, freq='60min',
                        tz='Etc/GMT-10')
    rt = RetailTariff(
        name='BE', state='NSW', type='tou', daily_c_per_day=0.0,
        usage_tariffs=(TariffPeriod(0, 24, 20.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(
            TariffPeriod(0, 14, 5.0), TariffPeriod(14, 16, 0.0),
            TariffPeriod(16, 24, 5.0),
        ),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
        block_export_tiers=(BlockTariffWindow(
            start_hour=14, end_hour=16,
            tiers=(BlockTier(2.0, 25.0), BlockTier(math.inf, 5.0)),
        ),),
    )
    net = _network()
    prices_mwh = pd.Series([100.] * n, index=idx)
    p_imp, p_exp = build_lp_price_vectors_advanced(idx, prices_mwh, net, rt, 'NSW', IH)

    # Solar fills battery at t=0,1 (before window); window at t=2,3
    solar = [2., 2., 0., 0., 0., 0.]
    load  = [0., 0., 0., 0., 0., 0.]
    bspec = _bspec(capacity=4., power=4.)
    tr = perfect_foresight_lp_advanced(
        'S', _s(solar, idx), _s(load, idx),
        p_imp, p_exp, bspec,
        soc_init_kwh=0., soc_final_kwh=0.,
        retail_tariff=rt, prices_mwh=prices_mwh,
        interval_hours=IH,
        state='NSW',)
    from model.core.prices import block_window_mask
    mask = block_window_mask(idx, rt.block_export_tiers[0], 'NSW', IH)
    window_export = float(tr.export_kwh.to_numpy()[mask].sum())
    # LP should fill the high-reward tier (2 kWh) -- tier2 rate (5c) equals
    # outside rate (5c) so no further incentive beyond tier1
    assert window_export == pytest.approx(2.0, abs=1e-3)


def test_block_export_tier_spills_when_solar_exceeds_tier1():
    """When available solar exceeds the tier1 cap, excess goes to tier2.

    Window [14,16). Solar=3 per interval, load=0, no battery limit.
    Tier1: 2 kWh at 25c. Tier2: excess at 5c. All solar exported.
    """
    n = 2
    idx = pd.date_range('2022-07-04 15:00:00', periods=n, freq='60min',
                        tz='Etc/GMT-10')
    rt = RetailTariff(
        name='BESpill', state='NSW', type='tou', daily_c_per_day=0.0,
        usage_tariffs=(TariffPeriod(0, 24, 20.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(
            TariffPeriod(0, 14, 5.0), TariffPeriod(14, 16, 0.0),
            TariffPeriod(16, 24, 5.0),
        ),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
        block_export_tiers=(BlockTariffWindow(
            start_hour=14, end_hour=16,
            tiers=(BlockTier(2.0, 25.0), BlockTier(math.inf, 5.0)),
        ),),
    )
    net = _network()
    prices_mwh = pd.Series([100.] * n, index=idx)
    p_imp, p_exp = build_lp_price_vectors_advanced(idx, prices_mwh, net, rt, 'NSW', IH)

    # 2 window intervals * 3 kWh = 6 kWh exported. 2 kWh tier1, 4 kWh tier2.
    solar = [3., 3.]
    load  = [0., 0.]
    bspec = _bspec(capacity=10., power=10.)
    tr = perfect_foresight_lp_advanced(
        'S', _s(solar, idx), _s(load, idx),
        p_imp, p_exp, bspec,
        soc_init_kwh=0., soc_final_kwh=0.,
        retail_tariff=rt, prices_mwh=prices_mwh,
        interval_hours=IH,
        state='NSW',)
    from model.core.prices import block_window_mask
    mask = block_window_mask(idx, rt.block_export_tiers[0], 'NSW', IH)
    total_export = float(tr.export_kwh.to_numpy()[mask].sum())
    assert total_export == pytest.approx(6.0, abs=1e-4)


# ---------------------------------------------------------------------------
# 4. Zero-import reward
# ---------------------------------------------------------------------------

def test_zero_import_reward_avoids_window_imports_when_battery_sufficient():
    """With a full battery, LP should avoid imports in the reward window.

    Window [18,20). Load=1 each window interval. Battery can cover load.
    Reward = $1/day. Without penalty, LP might import; with penalty it should
    discharge battery instead.
    """
    n = 4
    idx = pd.date_range('2022-07-04 17:00:00', periods=n, freq='60min',
                        tz='Etc/GMT-10')
    rt = RetailTariff(
        name='ZIR', state='NSW', type='tou', daily_c_per_day=0.0,
        usage_tariffs=(TariffPeriod(0, 24, 20.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 0.0),),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
        zero_import_reward=ZeroImportReward(
            start_hour=18, end_hour=20, reward_c_per_day=100.0,
        ),
    )
    net = _network()
    prices_mwh = pd.Series([100.] * n, index=idx)
    p_imp, p_exp = build_lp_price_vectors_advanced(idx, prices_mwh, net, rt, 'NSW', IH)

    solar = [0., 0., 0., 0.]
    load  = [0., 0., 1., 1.]   # load only in window hours [18,20)
    bspec = _bspec(capacity=4., power=4.)
    # Start with full battery so discharge is always available
    soc_init = 4.0
    soc_final_val = bspec.capacity_kwh * bspec.backup_reserve_pct / 100  # =0
    tr = perfect_foresight_lp_advanced(
        'S', _s(solar, idx), _s(load, idx),
        p_imp, p_exp, bspec,
        soc_init_kwh=soc_init, soc_final_kwh=soc_final_val,
        retail_tariff=rt, prices_mwh=prices_mwh,
        interval_hours=IH,
        state='NSW',)
    from model.core.prices import block_window_mask
    from model.core.tariff import BlockTariffWindow, BlockTier
    mask = block_window_mask(
        idx,
        BlockTariffWindow(start_hour=18, end_hour=20,
                          tiers=(BlockTier(1.0, 0.0), BlockTier(math.inf, 0.0))),
        'NSW',
        IH,
    )
    window_imports = float(tr.import_kwh.to_numpy()[mask].sum())
    assert window_imports == pytest.approx(0., abs=1e-4)


def test_zero_import_reward_allows_imports_when_battery_empty():
    """When the battery is empty and load must be served, LP imports even
    inside the reward window (physical necessity overrides the penalty).
    """
    n = 4
    idx = pd.date_range('2022-07-04 17:00:00', periods=n, freq='60min',
                        tz='Etc/GMT-10')
    rt = RetailTariff(
        name='ZIREmpty', state='NSW', type='tou', daily_c_per_day=0.0,
        usage_tariffs=(TariffPeriod(0, 24, 20.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 0.0),),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
        zero_import_reward=ZeroImportReward(
            start_hour=18, end_hour=20, reward_c_per_day=100.0,
        ),
    )
    net = _network()
    prices_mwh = pd.Series([100.] * n, index=idx)
    p_imp, p_exp = build_lp_price_vectors_advanced(idx, prices_mwh, net, rt, 'NSW', IH)

    # Use only window intervals: no pre-window intervals to pre-charge from.
    # Battery starts and ends empty, forced to import in the window.
    n2 = 2
    idx2 = pd.date_range('2022-07-04 19:00:00', periods=n2, freq='60min',
                         tz='Etc/GMT-10')   # hours 18,19 (start_instants)
    prices2 = pd.Series([100.] * n2, index=idx2)
    net2 = _network()
    p_imp2, p_exp2 = build_lp_price_vectors_advanced(
        idx2, prices2, net2, rt, 'NSW', IH)

    solar2 = [0., 0.]
    load2  = [2., 2.]   # 4 kWh total, all in window, no pre-window to charge from
    bspec2 = _bspec(capacity=4., power=4.)
    tr2 = perfect_foresight_lp_advanced(
        'S', pd.Series(solar2, index=idx2), pd.Series(load2, index=idx2),
        p_imp2, p_exp2, bspec2,
        soc_init_kwh=0., soc_final_kwh=0.,
        retail_tariff=rt, prices_mwh=prices2,
        interval_hours=IH,
        state='NSW',)
    # Must import all 4 kWh (no pre-charged battery, no solar)
    window_imports = float(tr2.import_kwh.to_numpy().sum())
    assert window_imports == pytest.approx(4.0, abs=1e-4)


# ---------------------------------------------------------------------------
# 5. Critical export rate
# ---------------------------------------------------------------------------

def test_critical_export_rate_dispatches_at_spike():
    """At critical intervals, LP should discharge battery to export.

    4 intervals. Battery starts charged. Price spike at t=2,3 (above threshold).
    The critical rate of 100c/kWh replaces the 5c feed-in. LP should export
    during the spike.
    """
    n = 4
    idx = pd.date_range('2022-07-04 01:00:00', periods=n, freq='60min',
                        tz='Etc/GMT-10')
    rt = RetailTariff(
        name='CriticalExp', state='NSW', type='tou', daily_c_per_day=0.0,
        usage_tariffs=(TariffPeriod(0, 24, 20.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 5.0),),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
        critical_export_rate=CriticalRate(
            threshold_dollars_per_mwh=2000.0, rate_c_per_kwh=100.0,
        ),
    )
    net = _network()
    # First two below threshold, last two above
    prices_mwh = pd.Series([100., 100., 5000., 5000.], index=idx)
    p_imp, p_exp = build_lp_price_vectors_advanced(idx, prices_mwh, net, rt, 'NSW', IH)

    solar = [0., 0., 0., 0.]
    load  = [0., 0., 0., 0.]   # no load so battery can export freely
    bspec = _bspec(capacity=4., power=4.)
    tr = perfect_foresight_lp_advanced(
        'S', _s(solar, idx), _s(load, idx),
        p_imp, p_exp, bspec,
        soc_init_kwh=4., soc_final_kwh=0.,   # discharge all
        retail_tariff=rt, prices_mwh=prices_mwh,
        interval_hours=IH,
        state='NSW',)
    # LP should export at spiked intervals (t=2,3)
    spike_export = float(tr.export_kwh.to_numpy()[[2, 3]].sum())
    pre_spike_export = float(tr.export_kwh.to_numpy()[[0, 1]].sum())
    # Spiked export should be >= pre-spike export (incentive to delay)
    assert spike_export >= pre_spike_export - 1e-4


def test_critical_export_in_block_window_does_not_use_the_tier():
    """Hourly day, export window 14-16 with tier 1 = 2 kWh at 25c, then 5c;
    10c feed-in outside the window. Battery holds 6 kWh, power 4 kW, and must
    end empty. Hour 14 is export-critical (100c).

    The LP exports 4 kWh at hour 14. Those 4 kWh are critical and outside the
    tier, so tier 1 is still free at hour 15 and the last 2 kWh earn 25c
    there. If the critical 4 kWh counted towards the tier, hour 15 would pay
    only 5c and the LP would export the 2 kWh at 10c outside the window.
    """
    n = 17
    rt = _rt_block_export(outside_feedin=10.0)
    rt = RetailTariff(**{**rt.__dict__, 'critical_export_rate': CriticalRate(
        threshold_dollars_per_mwh=2000.0, rate_c_per_kwh=100.0)})
    prices = [100.0] * n
    prices[14] = 5000.0
    tr = _run_advanced([0.0] * n, [0.0] * n, rt, prices,
                       bspec=_bspec(capacity=6.0, power=4.0),
                       soc_init=6.0, soc_final=0.0)
    exp = tr.export_kwh.to_numpy()
    assert exp[14] == pytest.approx(4.0, abs=1e-6)
    assert exp[15] == pytest.approx(2.0, abs=1e-6), (
        f"exports by hour: {np.round(exp, 4).tolist()}")


def test_critical_export_in_block_window_billed_as_dispatched():
    """The same dispatch billed: 4 kWh at 100c, 2 kWh in tier 1 at 25c."""
    from model.components.billing import compute_bill
    n = 17
    rt = _rt_block_export(outside_feedin=10.0)
    rt = RetailTariff(**{**rt.__dict__, 'critical_export_rate': CriticalRate(
        threshold_dollars_per_mwh=2000.0, rate_c_per_kwh=100.0)})
    prices = [100.0] * n
    prices[14] = 5000.0
    tr = _run_advanced([0.0] * n, [0.0] * n, rt, prices,
                       bspec=_bspec(capacity=6.0, power=4.0),
                       soc_init=6.0, soc_final=0.0)
    bill = compute_bill(tr, _s(prices, _idx(n)), _network(), rt, 'NSW', IH)
    assert bill.retail_feedin_credit == pytest.approx(-(4.0 * 1.00 + 2.0 * 0.25))


def test_critical_export_rate_no_spike_same_as_standard():
    """When all prices are below threshold, no critical rate: same as standard."""
    n = 4
    idx = _idx(n)
    rt_amp = RetailTariff(
        name='CriticalNoSpike', state='NSW', type='tou', daily_c_per_day=0.0,
        usage_tariffs=(TariffPeriod(0, 24, 20.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 5.0),),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
        critical_export_rate=CriticalRate(
            threshold_dollars_per_mwh=2000.0, rate_c_per_kwh=100.0,
        ),
    )
    rt_plain = _flat_rt(usage_c=20.0, feedin_c=5.0)
    net = _network()
    prices_mwh = [100., 100., 100., 100.]  # all below 2000 threshold

    tr_amp = _run_advanced([0.]*n, [1.]*n, rt_amp, prices_mwh,
                            bspec=_bspec(), soc_init=2., soc_final=0.)

    idx_i = _idx(n)
    p_imp, p_exp = build_price_vectors(idx_i, _s(prices_mwh, idx_i), net, rt_plain, 'NSW', IH)
    tr_std = perfect_foresight_lp(
        'S', _s([0.]*n, idx_i), _s([1.]*n, idx_i),
        p_imp, p_exp, _bspec(),
        soc_init_kwh=2., soc_final_kwh=0., interval_hours=IH,
    )
    np.testing.assert_allclose(
        tr_amp.import_kwh.to_numpy(), tr_std.import_kwh.to_numpy(), atol=1e-5)


# ---------------------------------------------------------------------------
# 6. Critical import rate
# ---------------------------------------------------------------------------

def test_critical_import_rate_charges_at_negative_price():
    """LP should charge battery when the critical import rate applies.

    4 intervals. t=0,1: price below the -100 threshold -> import at -5c/kWh
    in place of 20c. t=2,3: price positive -> 20c. Load=0. LP should charge
    at t=0,1.
    """
    n = 4
    idx = _idx(n)
    rt = RetailTariff(
        name='CriticalImp', state='NSW', type='tou', daily_c_per_day=0.0,
        usage_tariffs=(TariffPeriod(0, 24, 20.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 0.0),),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
        critical_import_rate=CriticalRate(
            threshold_dollars_per_mwh=-100.0, rate_c_per_kwh=-5.0,
        ),
    )
    net = _network()
    prices = [-300., -300., 100., 100.]  # first two are critical
    p_imp, p_exp = build_lp_price_vectors_advanced(idx, _s(prices, idx), net, rt, 'NSW', IH)
    # At t=0,1: p_imp = -5c/kWh; cheaper to import than at t=2,3
    solar = [0.] * n
    load  = [0.] * n
    bspec = _bspec(capacity=4., power=4.)
    tr = perfect_foresight_lp_advanced(
        'S', _s(solar, idx), _s(load, idx),
        p_imp, p_exp, bspec,
        soc_init_kwh=0., soc_final_kwh=2.,   # must end at 2 kWh
        retail_tariff=rt, prices_mwh=_s(prices, idx),
        interval_hours=IH,
        state='NSW',)
    # Battery must reach soc_final=2; LP should prefer cheaper t=0,1
    early_charge = float(tr.import_kwh.to_numpy()[[0, 1]].sum())
    late_charge  = float(tr.import_kwh.to_numpy()[[2, 3]].sum())
    assert early_charge >= late_charge - 1e-4


# ---------------------------------------------------------------------------
# 7. Price ordering under critical rates
# ---------------------------------------------------------------------------

def test_critical_export_rate_shift_keeps_ordering_and_solves():
    """The critical export rate sets p_exp to 100c and shifts p_imp by the same
    delta, preserving p_imp >= p_exp. The LP must solve cleanly with no
    RuntimeError and produce valid SiteTraces.
    """
    n = 4
    idx = _idx(n)
    rt = RetailTariff(
        name='ShiftTest', state='NSW', type='tou', daily_c_per_day=0.0,
        usage_tariffs=(TariffPeriod(0, 24, 5.0),),   # low base p_imp = 5c
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 0.0),),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
        critical_export_rate=CriticalRate(
            threshold_dollars_per_mwh=50.0, rate_c_per_kwh=100.0,
        ),
    )
    net = _network()
    # All prices above threshold: every interval is export-critical.
    # p_exp = 100c; delta = 100c - 0c; p_imp = 5c + 100c = 105c.
    prices = [200., 200., 200., 200.]
    p_imp, p_exp = build_lp_price_vectors_advanced(idx, _s(prices, idx), net, rt, 'NSW', IH)
    assert (p_imp.to_numpy() >= p_exp.to_numpy()).all(), "the shift must give p_imp >= p_exp"
    # Ordering maintained: p_imp should be 1.05, p_exp should be 1.00
    import numpy.testing as npt
    npt.assert_allclose(p_imp.to_numpy(), [1.05]*n, atol=1e-9)
    npt.assert_allclose(p_exp.to_numpy(), [1.00]*n, atol=1e-9)

    solar = [1., 1., 1., 1.]
    load  = [0.5, 0.5, 0.5, 0.5]
    tr = perfect_foresight_lp_advanced(
        'S', _s(solar, idx), _s(load, idx),
        p_imp, p_exp, _bspec(capacity=4., power=4.),
        soc_init_kwh=0., soc_final_kwh=0.,
        retail_tariff=rt, prices_mwh=_s(prices, idx),
        interval_hours=IH,
        state='NSW',)
    assert tr.import_kwh.to_numpy().min() >= -1e-6
    assert tr.export_kwh.to_numpy().min() >= -1e-6


def test_standard_function_still_raises_on_price_ordering_violation():
    """The standard LP (not advanced) should still raise on p_exp > p_imp."""
    idx = _idx(2)
    with pytest.raises(ValueError, match='price_exp exceeds price_imp'):
        perfect_foresight_lp(
            'S',
            _s([0., 0.], idx), _s([0., 0.], idx),
            _s([0.2, 0.2], idx), _s([0.1, 0.3], idx),  # p_exp[1] > p_imp[1]
            _bspec(), 0., 0., interval_hours=IH,
        )


# ---------------------------------------------------------------------------
# 8. SiteTraces invariants
# ---------------------------------------------------------------------------

def test_sitetraces_invariants_with_block_and_reward():
    """All four SiteTraces invariants hold for a fully advanced tariff dispatch."""
    n = 8
    idx = pd.date_range('2022-07-04 08:00:00', periods=n, freq='60min',
                        tz='Etc/GMT-10')
    rt = RetailTariff(
        name='FullAdv', state='NSW', type='tou', daily_c_per_day=0.0,
        usage_tariffs=(
            TariffPeriod(0, 8, 20.0), TariffPeriod(8, 10, 0.0),
            TariffPeriod(10, 16, 20.0), TariffPeriod(16, 24, 40.0),
        ),
        usage_tariffs_weekend=None,
        feedin_tariffs=(
            TariffPeriod(0, 14, 5.0), TariffPeriod(14, 16, 0.0),
            TariffPeriod(16, 24, 5.0),
        ),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
        block_import_tiers=(BlockTariffWindow(
            start_hour=8, end_hour=10,
            tiers=(BlockTier(2.0, 0.0), BlockTier(math.inf, 30.0)),
        ),),
        block_export_tiers=(BlockTariffWindow(
            start_hour=14, end_hour=16,
            tiers=(BlockTier(2.0, 25.0), BlockTier(math.inf, 5.0)),
        ),),
        zero_import_reward=ZeroImportReward(
            start_hour=16, end_hour=18, reward_c_per_day=100.0,
        ),
        critical_export_rate=CriticalRate(
            threshold_dollars_per_mwh=300.0, rate_c_per_kwh=50.0,
        ),
    )
    net = _network()
    prices_mwh = pd.Series([100., 100., 200., 200., 500., 500., 100., 100.],
                            index=idx)
    p_imp, p_exp = build_lp_price_vectors_advanced(idx, prices_mwh, net, rt, 'NSW', IH)
    solar = [2., 2., 1., 0., 0., 0., 1., 1.]
    load  = [1., 1., 1., 1., 1., 1., 1., 1.]
    bspec = _bspec(capacity=4., power=3., eta_c=0.95, eta_d=0.95, c_deg=0.02)
    tr = perfect_foresight_lp_advanced(
        'S', _s(solar, idx), _s(load, idx),
        p_imp, p_exp, bspec,
        soc_init_kwh=0., soc_final_kwh=0.,
        retail_tariff=rt, prices_mwh=prices_mwh,
        interval_hours=IH,
        state='NSW',)

    tol = 1e-6
    imp  = tr.import_kwh.to_numpy()
    exp  = tr.export_kwh.to_numpy()
    s    = np.array(solar)
    l    = np.array(load)

    np.testing.assert_allclose(
        imp, tr.grid_to_home_kwh.to_numpy() + tr.grid_to_bat_kwh.to_numpy(),
        atol=tol, err_msg="import != grid_to_home + grid_to_bat",
    )
    np.testing.assert_allclose(
        exp, tr.sol_to_grid_kwh.to_numpy() + tr.bat_to_grid_kwh.to_numpy(),
        atol=tol, err_msg="export != sol_to_grid + bat_to_grid",
    )
    np.testing.assert_allclose(
        l, tr.sol_to_home_kwh.to_numpy() + tr.bat_to_home_kwh.to_numpy()
        + tr.grid_to_home_kwh.to_numpy(),
        atol=tol, err_msg="load != sol_to_home + bat_to_home + grid_to_home",
    )
    np.testing.assert_allclose(
        s, tr.sol_to_home_kwh.to_numpy() + tr.sol_to_bat_kwh.to_numpy()
        + tr.sol_to_grid_kwh.to_numpy() + tr.curtail_kwh.to_numpy(),
        atol=tol, err_msg="solar != sol_to_home + sol_to_bat + sol_to_grid + curtail",
    )
    for field in ('sol_to_bat_kwh', 'sol_to_grid_kwh', 'bat_to_home_kwh',
                  'bat_to_grid_kwh', 'grid_to_home_kwh', 'grid_to_bat_kwh'):
        arr = getattr(tr, field).to_numpy()
        assert (arr >= -tol).all(), f"{field} has negative values (min={arr.min():.2e})"


def test_lp_advanced_bill_le_no_battery_bill():
    """Advanced LP dispatch cost must not exceed the no-cycling cost."""
    n = 6
    solar =  [0., 0., 0., 0., 0., 0.]
    load  =  [1., 1., 1., 1., 1., 1.]
    prices = [100., 100., 500., 100., 100., 500.]
    rt_adv = RetailTariff(
        name='Adv', state='NSW', type='tou', daily_c_per_day=0.0,
        usage_tariffs=(TariffPeriod(0, 24, 20.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 5.0),),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
        zero_import_reward=ZeroImportReward(
            start_hour=1, end_hour=3, reward_c_per_day=100.0,
        ),
    )
    net = _network()
    idx = _idx(n)
    p_imp, p_exp = build_lp_price_vectors_advanced(
        idx, _s(prices, idx), net, rt_adv, 'NSW', IH)

    # No-cycling: large c_deg
    tr_nc = perfect_foresight_lp_advanced(
        'S', _s(solar, idx), _s(load, idx),
        p_imp, p_exp, _bspec(c_deg=100.),
        soc_init_kwh=0., soc_final_kwh=0.,
        retail_tariff=rt_adv, prices_mwh=_s(prices, idx),
        interval_hours=IH,
        state='NSW',)
    cost_nc = float(np.sum(
        tr_nc.import_kwh.to_numpy() * p_imp.to_numpy()
        - tr_nc.export_kwh.to_numpy() * p_exp.to_numpy()
    ))

    # LP optimal
    tr_lp = perfect_foresight_lp_advanced(
        'S', _s(solar, idx), _s(load, idx),
        p_imp, p_exp, _bspec(c_deg=0.),
        soc_init_kwh=0., soc_final_kwh=0.,
        retail_tariff=rt_adv, prices_mwh=_s(prices, idx),
        interval_hours=IH,
        state='NSW',)
    cost_lp = float(np.sum(
        tr_lp.import_kwh.to_numpy() * p_imp.to_numpy()
        - tr_lp.export_kwh.to_numpy() * p_exp.to_numpy()
    ))
    assert cost_lp <= cost_nc + 1e-6


# ---------------------------------------------------------------------------
# 9. Validation error paths
# ---------------------------------------------------------------------------

def test_nan_solar_raises():
    idx = _idx(2)
    rt = _flat_rt()
    net = _network()
    prices = pd.Series([100., 100.], index=idx)
    p_imp, p_exp = build_lp_price_vectors_advanced(idx, prices, net, rt, 'NSW', IH)
    with pytest.raises(ValueError, match='NaN'):
        perfect_foresight_lp_advanced(
            'S',
            pd.Series([float('nan'), 1.], index=idx),
            pd.Series([0., 0.], index=idx),
            p_imp, p_exp, _bspec(),
            soc_init_kwh=0., soc_final_kwh=0.,
            retail_tariff=rt, prices_mwh=prices,
            interval_hours=IH,
        state='NSW',)


def test_nan_price_imp_raises():
    idx = _idx(2)
    rt = _flat_rt()
    prices = pd.Series([100., 100.], index=idx)
    net = _network()
    p_imp, p_exp = build_lp_price_vectors_advanced(idx, prices, net, rt, 'NSW', IH)
    p_imp_nan = p_imp.copy()
    p_imp_nan.iloc[1] = float('nan')
    with pytest.raises(ValueError, match='NaN'):
        perfect_foresight_lp_advanced(
            'S',
            pd.Series([0., 0.], index=idx),
            pd.Series([0., 0.], index=idx),
            p_imp_nan, p_exp, _bspec(),
            soc_init_kwh=0., soc_final_kwh=0.,
            retail_tariff=rt, prices_mwh=prices,
            interval_hours=IH,
        state='NSW',)


def test_negative_load_raises():
    idx = _idx(2)
    rt = _flat_rt()
    net = _network()
    prices = pd.Series([100., 100.], index=idx)
    p_imp, p_exp = build_lp_price_vectors_advanced(idx, prices, net, rt, 'NSW', IH)
    with pytest.raises(ValueError, match='negative'):
        perfect_foresight_lp_advanced(
            'S',
            pd.Series([0., 0.], index=idx),
            pd.Series([-1., 1.], index=idx),
            p_imp, p_exp, _bspec(),
            soc_init_kwh=0., soc_final_kwh=0.,
            retail_tariff=rt, prices_mwh=prices,
            interval_hours=IH,
        state='NSW',)


def test_soc_init_out_of_bounds_raises():
    idx = _idx(2)
    rt = _flat_rt()
    net = _network()
    prices = pd.Series([100., 100.], index=idx)
    bspec = _bspec(capacity=5., reserve_pct=20.)  # soc_min=1.0
    p_imp, p_exp = build_lp_price_vectors_advanced(idx, prices, net, rt, 'NSW', IH)
    with pytest.raises(ValueError, match='soc_init_kwh'):
        perfect_foresight_lp_advanced(
            'S',
            pd.Series([0., 0.], index=idx),
            pd.Series([0., 0.], index=idx),
            p_imp, p_exp, bspec,
            soc_init_kwh=0.5,   # below soc_min=1.0
            soc_final_kwh=1.0,
            retail_tariff=rt, prices_mwh=prices,
            interval_hours=IH,
        state='NSW',)


def test_misaligned_index_raises():
    idx2 = _idx(2)
    idx3 = _idx(3)
    rt = _flat_rt()
    net = _network()
    prices2 = pd.Series([100., 100.], index=idx2)
    prices3 = pd.Series([100., 100., 100.], index=idx3)
    p_imp, p_exp = build_lp_price_vectors_advanced(idx2, prices2, net, rt, 'NSW', IH)
    # price_imp has 2 intervals, solar/load have 3 -> mismatch
    p_imp3, p_exp3 = build_lp_price_vectors_advanced(idx3, prices3, net, rt, 'NSW', IH)
    with pytest.raises(ValueError, match='index'):
        perfect_foresight_lp_advanced(
            'S',
            pd.Series([0., 0.], index=idx2),
            pd.Series([0., 0.], index=idx2),
            p_imp3, p_exp3, _bspec(),   # mismatched index
            soc_init_kwh=0., soc_final_kwh=0.,
            retail_tariff=rt, prices_mwh=prices2,
            interval_hours=IH,
        state='NSW',)


# ---------------------------------------------------------------------------
# 10. Internal helpers
# ---------------------------------------------------------------------------

def test_build_day_map_single_day():
    """Single day: all intervals map to day 0."""
    idx = _idx(24)
    n_days, day_map = _build_day_map(idx, 'NSW', IH)
    assert n_days == 1
    assert (day_map == 0).all()


def test_build_day_map_two_days():
    """Two days: first 24 intervals map to day 0, next 24 to day 1."""
    idx = _idx(48)
    n_days, day_map = _build_day_map(idx, 'NSW', IH)
    assert n_days == 2
    assert (day_map[:24] == 0).all()
    assert (day_map[24:] == 1).all()


def test_build_block_tier_data_no_features():
    """No block features -> n_tier_vars=0, empty arrays, empty matrix."""
    n = 12
    idx = _idx(n)
    rt = _flat_rt()
    n_days, day_map = _build_day_map(idx, 'NSW', IH)
    n_tv, lo, hi, costs, lr = _build_block_tier_data(n, idx, rt, IH, n_days, day_map, 'NSW')
    assert n_tv == 0
    assert len(lo) == 0
    assert lr.shape == (0, 6 * n)


def test_build_block_tier_data_counts():
    """1 import window, 2 tiers, 2 days -> 4 tier vars, 2 linking rows."""
    n = 48
    idx = _idx(n)
    rt = RetailTariff(
        name='T', state='NSW', type='tou', daily_c_per_day=0.0,
        usage_tariffs=(TariffPeriod(0, 24, 20.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 0.0),),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
        block_import_tiers=(BlockTariffWindow(
            start_hour=8, end_hour=10,
            tiers=(BlockTier(2.0, 0.0), BlockTier(math.inf, 30.0)),
        ),),
    )
    n_days, day_map = _build_day_map(idx, 'NSW', IH)
    assert n_days == 2
    n_tv, lo, hi, costs, lr = _build_block_tier_data(n, idx, rt, IH, n_days, day_map, 'NSW')
    assert n_tv == 4           # 1 window * 2 tiers * 2 days
    assert lr.shape[0] == 2    # 1 window * 2 days = 2 linking rows
    assert lr.shape[1] == 6 * n + 4


def test_build_block_tier_data_tier_bounds():
    """Tier bounds: tier1=[0,2], tier2=[0,inf]."""
    n = 24
    idx = _idx(n)
    rt = RetailTariff(
        name='T', state='NSW', type='tou', daily_c_per_day=0.0,
        usage_tariffs=(TariffPeriod(0, 24, 20.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 0.0),),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
        block_import_tiers=(BlockTariffWindow(
            start_hour=8, end_hour=10,
            tiers=(BlockTier(2.0, 0.0), BlockTier(math.inf, 30.0)),
        ),),
    )
    n_days, day_map = _build_day_map(idx, 'NSW', IH)
    _, lo, hi, costs, _ = _build_block_tier_data(n, idx, rt, IH, n_days, day_map, 'NSW')
    # Tier 1: [0, 2.0]; Tier 2: [0, inf]
    assert lo[0] == pytest.approx(0.0)
    assert hi[0] == pytest.approx(2.0)
    assert lo[1] == pytest.approx(0.0)
    assert math.isinf(hi[1])
    # Costs: tier1=0c/kWh/100=0, tier2=30c/kWh/100=0.30
    assert costs[0] == pytest.approx(0.0)
    assert costs[1] == pytest.approx(0.30)


def test_build_block_tier_data_export_costs_negative():
    """Export tier costs must be negative (reward = reduces bill)."""
    n = 24
    idx = _idx(n)
    rt = RetailTariff(
        name='T', state='NSW', type='tou', daily_c_per_day=0.0,
        usage_tariffs=(TariffPeriod(0, 24, 20.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 0.0),),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
        block_export_tiers=(BlockTariffWindow(
            start_hour=14, end_hour=16,
            tiers=(BlockTier(2.0, 25.0), BlockTier(math.inf, 5.0)),
        ),),
    )
    n_days, day_map = _build_day_map(idx, 'NSW', IH)
    _, _, _, costs, _ = _build_block_tier_data(n, idx, rt, IH, n_days, day_map, 'NSW')
    # Export: costs should be negative (rewards)
    assert costs[0] == pytest.approx(-0.25)   # -25c/100
    assert costs[1] == pytest.approx(-0.05)   # -5c/100


# ---------------------------------------------------------------------------
# apply_export_order_nudge
# ---------------------------------------------------------------------------

def test_export_order_nudge_charges_before_exporting():
    """With apply_export_order_nudge=True, the LP should charge the battery first
    and export the surplus, rather than exporting early and charging later.

    Setup: 4 hourly intervals on a day with excess solar throughout.
    Solar = 3 kWh/interval. Load = 1 kWh/interval. Battery capacity = 4 kWh,
    power = 2 kWh/interval. Flat export rate = 10c/kWh throughout (no price
    variation to distinguish early vs late export). soc_init = soc_min = 0,
    soc_final = 4 (battery must be full at end).

    Net surplus = 2 kWh/interval. Battery must charge 4 kWh total across 4
    intervals at max 2 kWh/interval. Two degenerate solutions exist under a
    flat export rate:
      (A) Export-first: export 2 kWh at t=0 and t=1 (battery empty), then
          charge 2 kWh at t=2 and t=3 (battery fills to 4).
      (B) Charge-first: charge 2 kWh at t=0 and t=1 (battery fills to 4),
          then export 2 kWh at t=2 and t=3 (surplus with full battery).

    Both have identical cost under a flat export price. The nudge makes early
    exports more attractive, so early export produces more revenue -- the LP
    resolves the tie in favour of (B), where early intervals have low nudge
    and export is deferred to later intervals where the nudge is higher.

    We verify: export is zero at t=0 and t=1 (charging) and positive at t=2
    and t=3 (battery full, exporting surplus).
    """
    n = 4
    idx = pd.date_range('2022-07-04 01:00:00', periods=n, freq='60min',
                        tz='Etc/GMT-10')
    rt = RetailTariff(
        name='FlatFeedin', state='NSW', type='tou', daily_c_per_day=0.0,
        usage_tariffs=(TariffPeriod(0, 24, 20.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 10.0),),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
    )
    net = _network()
    prices_mwh = pd.Series([100.] * n, index=idx)
    p_imp, p_exp = build_lp_price_vectors_advanced(idx, prices_mwh, net, rt, 'NSW', IH)

    solar = [3., 3., 3., 3.]
    load  = [1., 1., 1., 1.]
    # soc_final=4: battery must be full at end, so LP must charge 4 kWh total.
    # Power limit 2 kWh/interval means it takes exactly 2 intervals to fill.
    bspec = _bspec(capacity=4., power=2., c_deg=0.005)

    tr = perfect_foresight_lp_advanced(
        'S', _s(solar, idx), _s(load, idx),
        p_imp, p_exp, bspec,
        soc_init_kwh=0., soc_final_kwh=4.,
        retail_tariff=rt, prices_mwh=prices_mwh,
        interval_hours=IH,
        apply_export_order_nudge=True,
        state='NSW',)
    export = tr.export_kwh.to_numpy()
    soc    = tr.battery_soc_kwh.to_numpy()

    # Charge-first (B): no export at t=0, t=1 (solar going to battery).
    # Export of surplus (2 kWh) at t=2 and t=3 once battery is full.
    assert export[0] == pytest.approx(0., abs=1e-4), (
        f"Expected zero export at t=0 (charging), got {export[0]:.4f}"
    )
    assert export[1] == pytest.approx(0., abs=1e-4), (
        f"Expected zero export at t=1 (charging), got {export[1]:.4f}"
    )
    assert export[2] == pytest.approx(2., abs=1e-4), (
        f"Expected 2 kWh export at t=2 (battery full), got {export[2]:.4f}"
    )
    assert export[3] == pytest.approx(2., abs=1e-4), (
        f"Expected 2 kWh export at t=3 (battery full), got {export[3]:.4f}"
    )
    # SOC should be monotonically rising: 2, 4, 4, 4.
    assert soc[0] == pytest.approx(2., abs=1e-4)
    assert soc[1] == pytest.approx(4., abs=1e-4)

# ---------------------------------------------------------------------------
# 12. Precomputed block window masks
# ---------------------------------------------------------------------------
#
# _build_block_tier_data can be handed the block window masks instead of
# computing them itself. The rolling advanced path calls this function once per
# 5-minute window -- roughly 105,000 times per site-year -- and cannot afford a
# local_wall_clock conversion per call. These tests pin the property the whole
# arrangement rests on: supplying the masks must change nothing.

from model.core.prices import block_window_mask  # noqa: E402


def _rt_block_both() -> RetailTariff:
    """TOU plan with both an import block window and an export block window."""
    return RetailTariff(
        name='BlockBoth', state='NSW', type='tou', daily_c_per_day=0.0,
        usage_tariffs=(TariffPeriod(0, 8, 20.0), TariffPeriod(8, 10, 0.0),
                       TariffPeriod(10, 24, 20.0)),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 14, 5.0), TariffPeriod(14, 16, 0.0),
                        TariffPeriod(16, 24, 5.0)),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
        block_import_tiers=(BlockTariffWindow(
            start_hour=8, end_hour=10,
            tiers=(BlockTier(2.0, 0.0), BlockTier(math.inf, 30.0)),
        ),),
        block_export_tiers=(BlockTariffWindow(
            start_hour=14, end_hour=16,
            tiers=(BlockTier(2.0, 12.0), BlockTier(math.inf, 3.0)),
        ),),
    )


def _masks_for(idx, rt, interval_hours=IH, state='NSW'):
    imp = tuple(block_window_mask(idx, w, state, interval_hours)
                for w in (rt.block_import_tiers or ()))
    exp = tuple(block_window_mask(idx, w, state, interval_hours)
                for w in (rt.block_export_tiers or ()))
    return imp, exp


def _assert_same_tier_data(idx, rt, interval_hours=IH, state='NSW', **kw):
    """Both routes through _build_block_tier_data must agree exactly."""
    T = len(idx)
    n_days, day_map = _build_day_map(idx, state, interval_hours)
    imp_masks, exp_masks = _masks_for(idx, rt, interval_hours, state)

    internal = _build_block_tier_data(
        T, idx, rt, interval_hours, n_days, day_map, state, **kw)
    supplied = _build_block_tier_data(
        T, idx, rt, interval_hours, n_days, day_map, state,
        import_window_masks=imp_masks, export_window_masks=exp_masks, **kw)

    assert internal[0] == supplied[0], (
        f"tier variable count differs: {internal[0]} vs {supplied[0]}")
    for name, a, b in zip(('tier_lo', 'tier_hi', 'tier_costs'),
                          internal[1:4], supplied[1:4]):
        np.testing.assert_array_equal(
            a, b, err_msg=f"{name} differs between the two routes")

    la, lb = internal[4], supplied[4]
    assert la.shape == lb.shape, f"linking shape {la.shape} vs {lb.shape}"
    diff = abs(la - lb)
    assert diff.nnz == 0, (
        f"{diff.nnz} differing linking entries; first at "
        f"{list(zip(*diff.nonzero()))[:3]}")
    return internal


def test_precomputed_masks_match_internal_computation():
    """Three days, both a block import and a block export window."""
    idx = _idx(72)
    result = _assert_same_tier_data(idx, _rt_block_both())
    # Sanity: the case is non-trivial -- 2 windows x 2 tiers x 3 days.
    assert result[0] == 12
    assert result[4].shape[0] == 6


def test_precomputed_masks_match_with_day0_carry_in():
    """Day-0 carry-in large enough to exhaust tier 1 and eat into tier 2."""
    idx = _idx(72)
    _assert_same_tier_data(idx, _rt_block_both(),
                           first_day_block_import_used={0: 3.5},
                           first_day_block_export_used={0: 1.0})


def test_precomputed_masks_match_over_a_weekend_for_weekday_only_windows():
    """Weekday-only windows leave empty linking rows on weekend days, which is
    the case most likely to differ between the two routes."""
    rt = RetailTariff(
        name='WeekdayOnly', state='NSW', type='tou', daily_c_per_day=0.0,
        usage_tariffs=(TariffPeriod(0, 8, 20.0), TariffPeriod(8, 10, 0.0),
                       TariffPeriod(10, 24, 20.0)),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 5.0),),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
        block_import_tiers=(BlockTariffWindow(
            start_hour=8, end_hour=10,
            tiers=(BlockTier(2.0, 0.0), BlockTier(math.inf, 30.0)),
            applies_weekends=False,
        ),),
    )
    # _idx starts on a Monday, so 168 hourly intervals covers a full week.
    _assert_same_tier_data(_idx(168), rt)


def test_precomputed_masks_match_across_a_dst_transition():
    """NSW springs forward on 2022-10-02, making that local day 23 hours long.

    This is not a test of DST semantics -- that lives in test_localtime.py.
    It asserts only that the two routes agree, which is exactly where a
    precomputed mask built against a different day boundary would show up.
    """
    idx = pd.date_range('2022-09-30 01:00:00', periods=96, freq='60min',
                        tz='Etc/GMT-10')
    _assert_same_tier_data(idx, _rt_block_both())


def test_supplied_masks_are_actually_used():
    """Guards the tests above from passing on an implementation that accepts
    the masks and then ignores them."""
    idx = _idx(72)
    rt = _rt_block_both()
    T = len(idx)
    n_days, day_map = _build_day_map(idx, 'NSW', IH)
    imp_masks, exp_masks = _masks_for(idx, rt)

    real = _build_block_tier_data(
        T, idx, rt, IH, n_days, day_map, 'NSW',
        import_window_masks=imp_masks, export_window_masks=exp_masks)
    blank = _build_block_tier_data(
        T, idx, rt, IH, n_days, day_map, 'NSW',
        import_window_masks=(np.zeros(T, dtype=bool),),
        export_window_masks=exp_masks)

    assert real[4].nnz > blank[4].nnz, (
        f"an all-false import mask produced {blank[4].nnz} linking nonzeros, "
        f"same or more than the real masks' {real[4].nnz}; the parameter is "
        f"being ignored")


def test_precomputed_mask_count_mismatch_raises():
    idx = _idx(24)
    rt = _rt_block_both()
    n_days, day_map = _build_day_map(idx, 'NSW', IH)
    _, exp_masks = _masks_for(idx, rt)
    with pytest.raises(ValueError, match="one mask per window"):
        _build_block_tier_data(
            len(idx), idx, rt, IH, n_days, day_map, 'NSW',
            import_window_masks=(), export_window_masks=exp_masks)


def test_precomputed_mask_length_mismatch_raises():
    idx = _idx(24)
    rt = _rt_block_both()
    n_days, day_map = _build_day_map(idx, 'NSW', IH)
    imp_masks, exp_masks = _masks_for(idx, rt)
    with pytest.raises(ValueError, match="length"):
        _build_block_tier_data(
            len(idx), idx, rt, IH, n_days, day_map, 'NSW',
            import_window_masks=(imp_masks[0][:-1],),
            export_window_masks=exp_masks)


# ---------------------------------------------------------------------------
# Mixed plans (model_spec.md, Mixed-plan dispatch prices)
# ---------------------------------------------------------------------------

from model.components.billing import compute_bill  # noqa: E402
from model.core.prices import dispatch_curtail_cost  # noqa: E402


def _mixed_rt(usage_c=10.0, floor_c=0.0):
    return RetailTariff(
        name='TestMixed', state='NSW', type='mixed', daily_c_per_day=110.0,
        usage_tariffs=(TariffPeriod(0, 24, usage_c),),
        usage_tariffs_weekend=None, feedin_tariffs=(),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
        wholesale_export_share=0.9, export_share_basis='wholesale',
        export_floor_c_per_kwh=floor_c,
    )


def _run_mixed(rt, prices, solar, load, bspec, soc_init, soc_final,
               export_limit=np.inf, price_vectors=None):
    idx = prices.index
    if price_vectors is None:
        price_vectors = build_lp_price_vectors_advanced(
            idx, prices, _network(), rt, 'NSW', IH)
    p_imp, p_exp = price_vectors
    return perfect_foresight_lp_advanced(
        'S', _s(solar, idx), _s(load, idx), p_imp, p_exp, bspec,
        soc_init_kwh=soc_init, soc_final_kwh=soc_final,
        retail_tariff=rt, prices_mwh=prices, state='NSW',
        export_limit_kwh_per_interval=export_limit, interval_hours=IH,
        curtail_cost_per_kwh_dollars=dispatch_curtail_cost(rt),
    )


def test_mixed_spot_term_stops_grid_charging_into_a_small_spike():
    """Spot at 3000 $/MWh for three hours, then 3200. Billed at the 10 c
    TOU rate, importing to export at 0.9 * 3.20 = 2.88 $/kWh pays; at the
    dispatch import price 0.10 + 3.00 = 3.10 $/kWh it does not. The LP must
    follow the dispatch prices."""
    idx = _idx(6)
    prices = _s([3000.0, 3000.0, 3000.0, 3200.0, 100.0, 100.0], idx)
    rt = _mixed_rt()
    bspec = _bspec(capacity=10.0, power=10.0, c_deg=0.001)
    zeros = [0.0] * 6
    t = _run_mixed(rt, prices, zeros, zeros, bspec, 0.0, 0.0)
    assert t.grid_to_bat_kwh.sum() == pytest.approx(0.0, abs=1e-6), (
        t.grid_to_bat_kwh.to_numpy())
    assert t.export_kwh.sum() == pytest.approx(0.0, abs=1e-6)

    # Non-vacuity: with spot at 1000 $/MWh before the same spike the dispatch
    # import price is 1.10 $/kWh, and the same LP does charge and export.
    # (The billed rates cannot serve as the comparison: a 10 c import against
    # a 2.88 $/kWh export credit makes the LP unbounded.)
    cheaper = _s([1000.0, 1000.0, 1000.0, 3200.0, 100.0, 100.0], idx)
    t_cheaper = _run_mixed(rt, cheaper, zeros, zeros, bspec, 0.0, 0.0)
    assert t_cheaper.export_kwh.iloc[3] == pytest.approx(10.0, abs=1e-6)


def test_mixed_grid_charging_into_a_large_spike_still_pays():
    """The spot term raises the bar but does not remove arbitrage: at 100
    $/MWh the dispatch import price is 0.20 $/kWh against 0.9 * 15.0 at the
    spike."""
    idx = _idx(4)
    prices = _s([100.0, 100.0, 15000.0, 100.0], idx)
    zeros = [0.0] * 4
    t = _run_mixed(_mixed_rt(), prices, zeros, zeros,
                   _bspec(capacity=10.0, power=10.0, c_deg=0.001), 0.0, 0.0)
    assert t.export_kwh.iloc[2] == pytest.approx(10.0, abs=1e-6)
    assert t.grid_to_bat_kwh.sum() == pytest.approx(10.0, abs=1e-6)


def test_mixed_surplus_solar_at_negative_spot_is_exported_not_curtailed():
    """Battery full, no load, spot negative: the floored credit is 0 for
    exporting and for curtailing alike. MIXED_CURTAIL_COST_PER_KWH breaks
    the tie towards export. With the export limit binding, curtailment still
    takes up what cannot be exported."""
    idx = _idx(4)
    prices = _s([-200.0] * 4, idx)
    solar, load = [5.0] * 4, [0.0] * 4
    bspec = _bspec(capacity=1.0, power=1.0, c_deg=0.01)   # no free cycling
    t = _run_mixed(_mixed_rt(), prices, solar, load, bspec, 1.0, 1.0)
    np.testing.assert_allclose(t.export_kwh.to_numpy(), 5.0, atol=1e-6)
    np.testing.assert_allclose(t.curtail_kwh.to_numpy(), 0.0, atol=1e-6)

    t_lim = _run_mixed(_mixed_rt(), prices, solar, load, bspec, 1.0, 1.0,
                       export_limit=2.0)
    np.testing.assert_allclose(t_lim.export_kwh.to_numpy(), 2.0, atol=1e-6)
    np.testing.assert_allclose(t_lim.curtail_kwh.to_numpy(), 3.0, atol=1e-6)


def test_mixed_price_cost_less_bill_is_the_spot_term_on_imports():
    """model_spec.md: the LP's price cost exceeds the bill's dispatch part
    by sum(max(0, RRP)/1000 * import_lf * import_kwh)."""
    idx = _idx(48)
    rng = np.random.default_rng(3)
    prices = _s(rng.normal(150.0, 400.0, 48), idx)
    prices.iloc[[7, 30]] = [9000.0, 12000.0]
    prices.iloc[[12, 13]] = -300.0
    hour = np.arange(48) % 24
    solar = 3.0 * np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None)
    load = 0.5 + 0.8 * (hour >= 17)
    rt = _mixed_rt(usage_c=25.0)
    p_imp, p_exp = build_lp_price_vectors_advanced(
        idx, prices, _network(), rt, 'NSW', IH)
    t = _run_mixed(rt, prices, solar, load, _bspec(c_deg=0.01), 0.0, 0.0,
                   price_vectors=(p_imp, p_exp))
    imp, exp = t.import_kwh.to_numpy(), t.export_kwh.to_numpy()
    assert imp.sum() > 0 and exp.sum() > 0
    assert (imp[prices.to_numpy() > 0] > 0).any()

    price_cost = float((imp * p_imp.to_numpy() - exp * p_exp.to_numpy()).sum())
    bill = compute_bill(t, prices, _network(), rt, 'NSW', IH)
    dispatch_part = bill.retail_usage_cost + bill.retail_feedin_credit
    spot_term = float((np.maximum(0.0, prices.to_numpy()) / 1000.0
                       * _network().import_loss_factor * imp).sum())
    assert price_cost - dispatch_part == pytest.approx(spot_term, abs=1e-9)


@pytest.mark.parametrize('curtail_cost, curtails', [(0.0, True), (1.0, False)])
def test_curtail_cost_reaches_the_lp(curtail_cost, curtails):
    """The curtailment cost argument is honoured, not dropped. With no export
    floor and spot at -500 $/MWh the export credit is -0.45 $/kWh: free
    curtailment strictly beats exporting, and a 1 $/kWh curtailment cost
    makes exporting at -0.45 the cheaper choice."""
    idx = _idx(3)
    prices = _s([-500.0] * 3, idx)
    rt = _mixed_rt(floor_c=None)
    p_imp, p_exp = build_lp_price_vectors_advanced(
        idx, prices, _network(), rt, 'NSW', IH)
    assert (p_exp.to_numpy() < -0.4).all()
    t = perfect_foresight_lp_advanced(
        'S', _s([4.0] * 3, idx), _s([0.0] * 3, idx), p_imp, p_exp,
        _bspec(capacity=1.0, power=1.0, c_deg=0.01),
        soc_init_kwh=1.0, soc_final_kwh=1.0, retail_tariff=rt,
        prices_mwh=prices, state='NSW', interval_hours=IH,
        curtail_cost_per_kwh_dollars=curtail_cost)
    expected_curtail = 4.0 if curtails else 0.0
    np.testing.assert_allclose(t.curtail_kwh.to_numpy(), expected_curtail,
                               atol=1e-6)
    np.testing.assert_allclose(t.export_kwh.to_numpy(), 4.0 - expected_curtail,
                               atol=1e-6)
