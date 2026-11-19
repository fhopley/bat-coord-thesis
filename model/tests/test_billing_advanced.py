"""Tests for advanced billing features in billing.py.

Covers block import tiers, block export tiers, zero-import reward,
critical rates, and combinations thereof. All inputs are hand-constructed
with analytically known answers so expected values are fully visible.

Fixture conventions (mirror test_billing.py):
  - index: 2022-07-04 (Monday), 5-min intervals, Etc/GMT-10
  - network: usage 10c/kWh, feedin 0c/kWh, daily 100c, loss factors 1.0
  - prices: flat 100 $/MWh unless the test needs variation
  - retail daily: 110c/day unless otherwise stated
  - all energy flows uniform per interval unless the test needs variation

The standard billing path (no advanced features) is already covered in
test_billing.py. This file only tests the advanced code paths but includes
one regression check confirming standard plans produce retail_other_credits=0.
"""
import math

import numpy as np
import pandas as pd
import pytest

from model.core.tariff import (
    NetworkTariff, RetailTariff, TariffPeriod,
    BlockTier, BlockTariffWindow, ZeroImportReward, CriticalRate,
)
from model.core.site_result import SiteTraces
from model.components.billing import compute_bill


# -----------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------

INTERVAL = 5 / 60
N_DAY = 288  # intervals per day


def _idx(n_days=1):
    """n_days of 5-min intervals starting 2022-07-04 (Monday)."""
    return pd.date_range(
        '2022-07-04 00:05:00', periods=N_DAY * n_days,
        freq='5min', tz='Etc/GMT-10',
    )


def _traces(imp_kwh, exp_kwh, index):
    """Construct SiteTraces from uniform per-interval values or arrays."""
    n = len(index)
    imp = pd.Series(np.full(n, float(imp_kwh)), index=index)
    exp = pd.Series(np.full(n, float(exp_kwh)), index=index)
    z = pd.Series(np.zeros(n), index=index)
    return SiteTraces(
        site_id='S0001',
        import_kwh=imp, export_kwh=exp, battery_soc_kwh=z,
        sol_to_home_kwh=z, sol_to_bat_kwh=z, sol_to_grid_kwh=z,
        bat_to_home_kwh=z, grid_to_home_kwh=imp, grid_to_bat_kwh=z,
        bat_to_grid_kwh=z, curtail_kwh=z,
    )


def _traces_series(imp_series, exp_series):
    """Construct SiteTraces from pd.Series directly."""
    index = imp_series.index
    z = pd.Series(np.zeros(len(index)), index=index)
    return SiteTraces(
        site_id='S0001',
        import_kwh=imp_series, export_kwh=exp_series, battery_soc_kwh=z,
        sol_to_home_kwh=z, sol_to_bat_kwh=z, sol_to_grid_kwh=z,
        bat_to_home_kwh=z, grid_to_home_kwh=imp_series, grid_to_bat_kwh=z,
        bat_to_grid_kwh=z, curtail_kwh=z,
    )


def _network():
    return NetworkTariff(
        name='TestNet', state='NSW',
        usage_tariffs=(TariffPeriod(0, 24, 10.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 0.0),),
        daily_c_per_day=100.0,
        usage_dlf=1.0, export_dlf=1.0, import_tlf=1.0, export_tlf=1.0,
    )


def _flat_prices(index, mwh=100.0):
    return pd.Series(np.full(len(index), float(mwh)), index=index)


def _two_tier_import_window(start=11, end=14, t1_kwh=50.0,
                             t1_rate=0.0, t2_rate=30.8):
    """GloBird-style: first 50 kWh free, excess at 30.8c."""
    return BlockTariffWindow(
        start_hour=start, end_hour=end,
        tiers=(
            BlockTier(threshold_kwh=t1_kwh, rate_c_per_kwh=t1_rate),
            BlockTier(threshold_kwh=math.inf, rate_c_per_kwh=t2_rate),
        ),
    )


def _two_tier_export_window(start=18, end=21, t1_kwh=15.0,
                             t1_rate=14.0, t2_rate=2.0):
    """GloBird-style: first 15 kWh at 14c, excess at 2c."""
    return BlockTariffWindow(
        start_hour=start, end_hour=end,
        tiers=(
            BlockTier(threshold_kwh=t1_kwh, rate_c_per_kwh=t1_rate),
            BlockTier(threshold_kwh=math.inf, rate_c_per_kwh=t2_rate),
        ),
    )


def _tou_with_block_import(**kwargs):
    """TOU plan with block_import_tiers. Usage_tariff is 0 in the block
    window (standard rate) and 20c elsewhere. No greenpower adder."""
    return RetailTariff(
        name='BlockImport', state='NSW', type='tou', daily_c_per_day=110.0,
        usage_tariffs=(
            TariffPeriod(0, 11, 20.0),
            TariffPeriod(11, 14, 0.0),   # overridden by block tiers
            TariffPeriod(14, 24, 20.0),
        ),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 0.0),),
        usage_other_c_per_kwh=0.0,
        feedin_other_c_per_kwh=0.0,
        block_import_tiers=(_two_tier_import_window(),),
        **kwargs,
    )


def _tou_with_block_export(**kwargs):
    """TOU plan with block_export_tiers. Feedin_tariff is 0 in the block
    window and 5c elsewhere."""
    return RetailTariff(
        name='BlockExport', state='NSW', type='tou', daily_c_per_day=110.0,
        usage_tariffs=(TariffPeriod(0, 24, 20.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(
            TariffPeriod(0, 18, 5.0),
            TariffPeriod(18, 21, 0.0),   # overridden by block tiers
            TariffPeriod(21, 24, 5.0),
        ),
        usage_other_c_per_kwh=0.0,
        feedin_other_c_per_kwh=0.0,
        block_export_tiers=(_two_tier_export_window(),),
        **kwargs,
    )


# -----------------------------------------------------------------------
# Regression: standard plan still produces retail_other_credits=0
# -----------------------------------------------------------------------

def test_standard_plan_retail_other_credits_is_zero():
    """Standard TOU plan: retail_other_credits must be 0.0 and has no effect
    on household_total."""
    idx = _idx(1)
    rt = RetailTariff(
        name='Standard', state='NSW', type='tou', daily_c_per_day=110.0,
        usage_tariffs=(TariffPeriod(0, 24, 20.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 5.0),),
        usage_other_c_per_kwh=0.0,
        feedin_other_c_per_kwh=0.0,
    )
    bill = compute_bill(_traces(1.0, 0.0, idx), _flat_prices(idx), _network(), rt, state='NSW')
    assert bill.retail_other_credits == 0.0
    assert bill.household_total == pytest.approx(
        bill.retail_usage_cost + bill.retail_feedin_credit + bill.retail_daily_charges
    )


# -----------------------------------------------------------------------
# Block import tiers
# -----------------------------------------------------------------------

def test_block_import_tier1_only():
    """36 intervals * 1 kWh = 36 kWh, all within tier 1 (<=50 kWh at 0c).
    Block cost = $0. Outside-window cost = 252 intervals * 1 kWh * 20c/kWh."""
    idx = _idx(1)
    rt = _tou_with_block_import()
    bill = compute_bill(_traces(1.0, 0.0, idx), _flat_prices(idx), _network(), rt, state='NSW')
    # Window: 11-14 = 36 intervals; outside = 252
    # Block: 36 kWh all in tier 1 at 0c = $0
    # Standard credit for window was 0 (TOU rate = 0 there)
    # Outside: 252 * 1.0 * 20/100 = $50.40
    assert bill.retail_usage_cost == pytest.approx(252 * 20.0 / 100)
    assert bill.retail_other_credits == 0.0


def test_block_import_tier1_full_tier2_spillover():
    """60 kWh in window (60/36 per interval): fills tier 1 (50 kWh at 0c),
    10 kWh spills into tier 2 at 30.8c. Outside unchanged."""
    idx = _idx(1)
    rt = _tou_with_block_import()
    # 60/36 kWh per interval * 36 intervals = 60 kWh in window
    imp_per_interval = 60.0 / 36.0
    bill = compute_bill(
        _traces(imp_per_interval, 0.0, idx), _flat_prices(idx), _network(), rt, state='NSW')
    block_cost = 0.0 + 10.0 * 30.8 / 100   # tier 1 free, tier 2 on 10 kWh
    outside_cost = 252 * imp_per_interval * 20.0 / 100
    assert bill.retail_usage_cost == pytest.approx(block_cost + outside_cost)


def test_block_import_exactly_at_tier_boundary():
    """Exactly 50 kWh in window = fills tier 1 exactly, zero tier 2 cost."""
    idx = _idx(1)
    rt = _tou_with_block_import()
    imp_per_interval = 50.0 / 36.0
    bill = compute_bill(
        _traces(imp_per_interval, 0.0, idx), _flat_prices(idx), _network(), rt, state='NSW')
    # Tier 1 exactly full: 50 kWh at 0c = $0 block cost
    outside_cost = 252 * imp_per_interval * 20.0 / 100
    assert bill.retail_usage_cost == pytest.approx(0.0 + outside_cost)


def test_block_import_zero_imports_in_window():
    """Zero imports in window: block cost = $0, standard cost unaffected."""
    idx = _idx(1)
    rt = _tou_with_block_import()
    imp = pd.Series(np.zeros(N_DAY), index=idx)
    # Add 1 kWh only to outside-window intervals
    imp_arr = imp.to_numpy().copy()
    from model.core.prices import block_window_mask
    mask = block_window_mask(idx, rt.block_import_tiers[0], 'NSW')
    imp_arr[~mask] = 1.0
    imp_series = pd.Series(imp_arr, index=idx)
    bill = compute_bill(
        _traces_series(imp_series, pd.Series(np.zeros(N_DAY), index=idx)),
        _flat_prices(idx), _network(), rt, state='NSW')
    outside_n = int((~mask).sum())
    assert bill.retail_usage_cost == pytest.approx(outside_n * 1.0 * 20.0 / 100)


def test_block_import_two_days_independent():
    """Two days: day 1 has 36 kWh (tier 1 only), day 2 has 72 kWh (36 tier 1
    + 22 tier 2). Tiers reset daily."""
    idx = _idx(2)
    # 36 kWh/day on day 1 = 1.0 kWh/interval; 72 kWh/day on day 2 = 2.0 kWh/interval
    from model.core.prices import block_window_mask
    rt = _tou_with_block_import()
    mask = block_window_mask(idx, rt.block_import_tiers[0], 'NSW')
    imp_arr = np.ones(N_DAY * 2)
    # Day 2: double the import in the window
    day2_window = np.zeros(N_DAY * 2, dtype=bool)
    day2_window[N_DAY:] = mask[N_DAY:]
    imp_arr[day2_window] = 2.0
    imp_series = pd.Series(imp_arr, index=idx)
    bill = compute_bill(
        _traces_series(imp_series, pd.Series(np.zeros(N_DAY * 2), index=idx)),
        _flat_prices(idx), _network(), rt, state='NSW')
    # Day 1 block: 36 kWh, all tier 1 at 0c = $0
    # Day 2 block: 72 kWh, 50 tier 1 at 0c + 22 tier 2 at 30.8c
    day2_block = 22.0 * 30.8 / 100
    # Outside window both days: day 1 = 252 * 1.0; day 2 = 252 * 1.0
    outside_both = 252 * 1.0 * 20.0 / 100 * 2
    assert bill.retail_usage_cost == pytest.approx(day2_block + outside_both)


def test_block_import_usage_other_charged_in_window():
    """usage_other_c_per_kwh (e.g. greenpower levy) must be charged for block
    window intervals even though the TOU energy rate is replaced by the tier rate.

    The block tier replaces only the TOU energy component. usage_other is a
    flat per-kWh levy that applies to all imported kWh regardless of window.
    Window intervals should pay: tier_rate + usage_other (not just tier_rate).

    This matches the LP, which sets p_imp[mask] = usage_other_rate for block
    window intervals (the tier variables handle tier_rate on top).

    Setup: 1 day, 1 kWh/interval imported everywhere.
      - Window [11,14): 36 intervals, tier1 rate=0c, usage_other=3c → 3c/kWh
      - Outside window: 252 intervals, TOU=20c, usage_other=3c → 23c/kWh
    """
    idx = _idx(1)
    rt = RetailTariff(
        name='BlockImportGreenpower', state='NSW', type='tou', daily_c_per_day=110.0,
        usage_tariffs=(
            TariffPeriod(0, 11, 20.0), TariffPeriod(11, 14, 0.0),
            TariffPeriod(14, 24, 20.0),
        ),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 0.0),),
        usage_other_c_per_kwh=3.0,   # greenpower levy
        feedin_other_c_per_kwh=0.0,
        block_import_tiers=(_two_tier_import_window(),),
    )
    bill = compute_bill(_traces(1.0, 0.0, idx), _flat_prices(idx), _network(), rt, state='NSW')
    # Window: 36 * 1.0 kWh * (0c tier1 + 3c usage_other) / 100
    window_cost  = 36.0 * 1.0 * 3.0 / 100
    # Outside: 252 * 1.0 kWh * (20c TOU + 3c usage_other) / 100
    outside_cost = 252.0 * 1.0 * 23.0 / 100
    assert bill.retail_usage_cost == pytest.approx(window_cost + outside_cost)


def test_block_import_usage_other_tier2_spillover_with_usage_other():
    """When daily window imports spill into tier2, usage_other is still charged
    on the full window volume (both tiers), not just the outside-window portion.

    Setup: 1 day, imports exceed tier1 threshold in window.
      tier1: 50 kWh at 0c, tier2: excess at 30.8c, usage_other=3c everywhere.
      Window flow = 60 kWh: 50 kWh tier1 + 10 kWh tier2.
      Window cost = 50*(0+3)c + 10*(30.8+3)c = 150c + 338c = 488c = $4.88.
    """
    idx = _idx(1)
    rt = RetailTariff(
        name='BlockImportGreenpowerT2', state='NSW', type='tou', daily_c_per_day=0.0,
        usage_tariffs=(
            TariffPeriod(0, 11, 20.0), TariffPeriod(11, 14, 0.0),
            TariffPeriod(14, 24, 20.0),
        ),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 0.0),),
        usage_other_c_per_kwh=3.0,
        feedin_other_c_per_kwh=0.0,
        block_import_tiers=(_two_tier_import_window(),),  # 50 kWh threshold
    )
    # 60 kWh in window (36 intervals * 60/36 kWh each)
    window_flow = 60.0 / 36.0
    bill = compute_bill(_traces(window_flow, 0.0, idx), _flat_prices(idx),
                        _network(), rt, state='NSW')
    window_cost = (50.0 * 3.0 + 10.0 * (30.8 + 3.0)) / 100
    outside_cost = 252.0 * window_flow * 23.0 / 100
    assert bill.retail_usage_cost == pytest.approx(window_cost + outside_cost)


def test_block_export_feedin_other_charged_in_window():
    """feedin_other_c_per_kwh must be included in the block export window credit.

    The block tier replaces the TOU feedin rate but feedin_other still applies.
    Window credit = tier_rate + feedin_other (both as a reward/credit).

    Setup: 1 day, 0.3 kWh/interval exported everywhere.
      - Window [18,21): 36 intervals, tier1=14c, feedin_other=2c → 16c/kWh
      - Outside: 252 intervals, TOU feedin=5c, feedin_other=2c → 7c/kWh
    """
    idx = _idx(1)
    rt = RetailTariff(
        name='BlockExportFeedinOther', state='NSW', type='tou', daily_c_per_day=0.0,
        usage_tariffs=(TariffPeriod(0, 24, 20.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(
            TariffPeriod(0, 18, 5.0), TariffPeriod(18, 21, 0.0),
            TariffPeriod(21, 24, 5.0),
        ),
        usage_other_c_per_kwh=0.0,
        feedin_other_c_per_kwh=2.0,   # flat feedin bonus
        block_export_tiers=(BlockTariffWindow(
            start_hour=18, end_hour=21,
            tiers=(BlockTier(15.0, 14.0), BlockTier(math.inf, 2.0)),
        ),),
    )
    exp = 0.3
    bill = compute_bill(_traces(0.0, exp, idx), _flat_prices(idx), _network(), rt, state='NSW')
    # Window: 36 * 0.3 = 10.8 kWh < 15 kWh threshold → all tier1
    # credit = -(10.8 * (14c + 2c) / 100) = -(10.8 * 16c / 100)
    window_credit = -(10.8 * 16.0 / 100)
    # Outside: 252 * 0.3 * (5c + 2c) / 100
    outside_credit = -(252 * 0.3 * 7.0 / 100)
    assert bill.retail_feedin_credit == pytest.approx(window_credit + outside_credit)


# -----------------------------------------------------------------------
# Block export tiers
# -----------------------------------------------------------------------

def test_block_export_tier1_only():
    """36 intervals * 0.3 kWh = 10.8 kWh < 15 kWh threshold: all tier 1 at 14c.
    Outside window at 5c/kWh."""
    idx = _idx(1)
    rt = _tou_with_block_export()
    bill = compute_bill(_traces(0.0, 0.3, idx), _flat_prices(idx), _network(), rt, state='NSW')
    # Window (18-21): 36 * 0.3 = 10.8 kWh < 15 kWh, all tier 1 at 14c
    # Standard credit for window was 0 (TOU feedin rate = 0 there)
    block_credit = -(10.8 * 14.0 / 100)   # negative = credit
    # Outside window: (288-36) * 0.3 * 5c/100 = 252 * 0.3 * 0.05
    outside_credit = -(252 * 0.3 * 5.0 / 100)
    assert bill.retail_feedin_credit == pytest.approx(block_credit + outside_credit)


def test_block_export_tier1_full_tier2_spillover():
    """54 kWh in window: 15 kWh tier 1 at 14c, 39 kWh tier 2 at 2c."""
    idx = _idx(1)
    rt = _tou_with_block_export()
    exp_per_interval = 54.0 / 36.0   # 1.5 kWh/interval in window
    bill = compute_bill(_traces(0.0, exp_per_interval, idx), _flat_prices(idx),
                        _network(), rt, state='NSW')
    block_credit = -(15.0 * 14.0 / 100 + 39.0 * 2.0 / 100)
    outside_credit = -(252 * exp_per_interval * 5.0 / 100)
    assert bill.retail_feedin_credit == pytest.approx(block_credit + outside_credit)


def test_block_export_exactly_at_tier_boundary():
    """Exactly 15 kWh in window = fills tier 1 exactly, zero tier 2."""
    idx = _idx(1)
    rt = _tou_with_block_export()
    exp_per_interval = 15.0 / 36.0
    bill = compute_bill(_traces(0.0, exp_per_interval, idx), _flat_prices(idx),
                        _network(), rt, state='NSW')
    block_credit = -(15.0 * 14.0 / 100)
    outside_credit = -(252 * exp_per_interval * 5.0 / 100)
    assert bill.retail_feedin_credit == pytest.approx(block_credit + outside_credit)


def test_block_export_two_days_independent():
    """Two days: tier allocation resets daily."""
    idx = _idx(2)
    from model.core.prices import block_window_mask
    rt = _tou_with_block_export()
    mask = block_window_mask(idx, rt.block_export_tiers[0], 'NSW')
    # Day 1: 10.8 kWh in window (all tier 1); Day 2: 54 kWh in window (tier spillover)
    exp_arr = np.full(N_DAY * 2, 0.3)   # day 1 default: 0.3/interval = 10.8 kWh
    day2_window = np.zeros(N_DAY * 2, dtype=bool)
    day2_window[N_DAY:] = mask[N_DAY:]
    exp_arr[day2_window] = 54.0 / 36.0
    exp_series = pd.Series(exp_arr, index=idx)
    bill = compute_bill(
        _traces_series(pd.Series(np.zeros(N_DAY * 2), index=idx), exp_series),
        _flat_prices(idx), _network(), rt, state='NSW')
    day1_block = -(10.8 * 14.0 / 100)
    day2_block = -(15.0 * 14.0 / 100 + 39.0 * 2.0 / 100)
    # Outside window: day 1 = 252 * 0.3; day 2 = 252 * 0.3 (unchanged outside window)
    outside = -(252 * 0.3 * 5.0 / 100 * 2)
    assert bill.retail_feedin_credit == pytest.approx(day1_block + day2_block + outside)


# -----------------------------------------------------------------------
# Zero-import reward
# -----------------------------------------------------------------------

def test_zero_import_reward_zero_qualifying_days():
    """All days have imports in window: no reward earned."""
    idx = _idx(2)
    rt = RetailTariff(
        name='ZIR', state='NSW', type='tou', daily_c_per_day=110.0,
        usage_tariffs=(TariffPeriod(0, 24, 20.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 0.0),),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
        zero_import_reward=ZeroImportReward(
            start_hour=18, end_hour=21,
            reward_c_per_day=100.0,
            tolerance_kwh_per_hour=0.03,
        ),
    )
    bill = compute_bill(_traces(1.0, 0.0, idx), _flat_prices(idx), _network(), rt, state='NSW')
    assert bill.retail_other_credits == pytest.approx(0.0)


def test_zero_import_reward_all_qualifying_days():
    """Two days of zero imports everywhere: both days earn the reward."""
    idx = _idx(2)
    rt = RetailTariff(
        name='ZIR', state='NSW', type='tou', daily_c_per_day=110.0,
        usage_tariffs=(TariffPeriod(0, 24, 20.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 0.0),),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
        zero_import_reward=ZeroImportReward(
            start_hour=18, end_hour=21,
            reward_c_per_day=100.0,
            tolerance_kwh_per_hour=0.03,
        ),
    )
    bill = compute_bill(_traces(0.0, 0.0, idx), _flat_prices(idx), _network(), rt, state='NSW')
    # $1/day * 2 days = -$2.00 (negative = credit)
    assert bill.retail_other_credits == pytest.approx(-2.0)


def test_zero_import_reward_one_of_two_days():
    """Day 1: zero window imports (qualifies). Day 2: imports in window
    (does not qualify). One day reward."""
    idx = _idx(2)
    rt = RetailTariff(
        name='ZIR', state='NSW', type='tou', daily_c_per_day=110.0,
        usage_tariffs=(TariffPeriod(0, 24, 20.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 0.0),),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
        zero_import_reward=ZeroImportReward(
            start_hour=18, end_hour=21,
            reward_c_per_day=100.0,
            tolerance_kwh_per_hour=0.03,
        ),
    )
    # Day 1 zero imports; day 2 has 1 kWh/interval
    imp_arr = np.zeros(N_DAY * 2)
    imp_arr[N_DAY:] = 1.0
    imp_series = pd.Series(imp_arr, index=idx)
    bill = compute_bill(
        _traces_series(imp_series, pd.Series(np.zeros(N_DAY * 2), index=idx)),
        _flat_prices(idx), _network(), rt, state='NSW')
    assert bill.retail_other_credits == pytest.approx(-1.0)


def test_zero_import_reward_within_tolerance():
    """Imports within tolerance still qualify for the reward."""
    idx = _idx(1)
    rt = RetailTariff(
        name='ZIR', state='NSW', type='tou', daily_c_per_day=110.0,
        usage_tariffs=(TariffPeriod(0, 24, 20.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 0.0),),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
        zero_import_reward=ZeroImportReward(
            start_hour=18, end_hour=21,
            reward_c_per_day=100.0,
            tolerance_kwh_per_hour=0.03,
        ),
    )
    # Window (18-21) = 36 intervals; tolerance = 0.03 * 3 = 0.09 kWh total
    # Spread 0.09 kWh over 36 intervals = 0.0025 kWh/interval (within tolerance)
    start_h = (idx - pd.Timedelta(hours=INTERVAL)).hour.to_numpy()
    mask = (start_h >= 18) & (start_h < 21)
    imp_arr = np.zeros(N_DAY)
    imp_arr[mask] = 0.09 / 36   # exactly at tolerance spread evenly
    imp_series = pd.Series(imp_arr, index=idx)
    bill = compute_bill(
        _traces_series(imp_series, pd.Series(np.zeros(N_DAY), index=idx)),
        _flat_prices(idx), _network(), rt, state='NSW')
    assert bill.retail_other_credits == pytest.approx(-1.0)


def test_zero_import_reward_above_tolerance():
    """Imports above tolerance: reward not earned."""
    idx = _idx(1)
    rt = RetailTariff(
        name='ZIR', state='NSW', type='tou', daily_c_per_day=110.0,
        usage_tariffs=(TariffPeriod(0, 24, 20.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 0.0),),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
        zero_import_reward=ZeroImportReward(
            start_hour=18, end_hour=21,
            reward_c_per_day=100.0,
            tolerance_kwh_per_hour=0.03,
        ),
    )
    # Import 0.1 kWh total in window (just above 0.09 tolerance)
    start_h = (idx - pd.Timedelta(hours=INTERVAL)).hour.to_numpy()
    mask = (start_h >= 18) & (start_h < 21)
    imp_arr = np.zeros(N_DAY)
    imp_arr[mask] = 0.1 / 36   # 0.1 kWh total > 0.09 tolerance
    imp_series = pd.Series(imp_arr, index=idx)
    bill = compute_bill(
        _traces_series(imp_series, pd.Series(np.zeros(N_DAY), index=idx)),
        _flat_prices(idx), _network(), rt, state='NSW')
    assert bill.retail_other_credits == pytest.approx(0.0)


def test_zero_import_reward_zero_tolerance_exact():
    """With zero tolerance, any import (above 1e-9) forfeits reward."""
    idx = _idx(1)
    rt = RetailTariff(
        name='ZIR', state='NSW', type='tou', daily_c_per_day=110.0,
        usage_tariffs=(TariffPeriod(0, 24, 20.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 0.0),),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
        zero_import_reward=ZeroImportReward(
            start_hour=18, end_hour=21,
            reward_c_per_day=100.0,
            # tolerance_kwh_per_hour defaults to 0
        ),
    )
    start_h = (idx - pd.Timedelta(hours=INTERVAL)).hour.to_numpy()
    mask = (start_h >= 18) & (start_h < 21)
    # Tiny import: 1e-8 kWh total (above 1e-9 exact-zero threshold)
    imp_arr = np.zeros(N_DAY)
    imp_arr[mask] = 1e-8 / 36
    imp_series = pd.Series(imp_arr, index=idx)
    bill = compute_bill(
        _traces_series(imp_series, pd.Series(np.zeros(N_DAY), index=idx)),
        _flat_prices(idx), _network(), rt, state='NSW')
    assert bill.retail_other_credits == pytest.approx(0.0)


def test_zero_import_reward_folds_into_household_total():
    """retail_other_credits is included in household_total for TOU plans."""
    idx = _idx(1)
    rt = RetailTariff(
        name='ZIR', state='NSW', type='tou', daily_c_per_day=110.0,
        usage_tariffs=(TariffPeriod(0, 24, 20.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 0.0),),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
        zero_import_reward=ZeroImportReward(
            start_hour=18, end_hour=21, reward_c_per_day=100.0,
        ),
    )
    bill = compute_bill(_traces(0.0, 0.0, idx), _flat_prices(idx), _network(), rt, state='NSW')
    assert bill.retail_other_credits == pytest.approx(-1.0)
    expected_total = (bill.retail_usage_cost + bill.retail_feedin_credit
                      + bill.retail_daily_charges + bill.retail_other_credits)
    assert bill.household_total == pytest.approx(expected_total)


# -----------------------------------------------------------------------
# Critical export rate
# -----------------------------------------------------------------------

def _flat_tou(**kwargs):
    """20c import, 5c feed-in all day, no adders, no block windows."""
    return RetailTariff(
        name='Critical', state='NSW', type='tou', daily_c_per_day=110.0,
        usage_tariffs=(TariffPeriod(0, 24, 20.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 5.0),),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
        **kwargs,
    )


_CX = CriticalRate(threshold_dollars_per_mwh=2000.0, rate_c_per_kwh=100.0)
_CM = CriticalRate(threshold_dollars_per_mwh=-100.0, rate_c_per_kwh=-5.0)


def test_critical_export_rate_no_spike_standard_rate():
    """All prices below the threshold: the standard feed-in applies."""
    idx = _idx(1)
    bill = compute_bill(_traces(0.0, 1.0, idx), _flat_prices(idx, 100.0),
                        _network(), _flat_tou(critical_export_rate=_CX), state='NSW')
    assert bill.retail_feedin_credit == pytest.approx(-(N_DAY * 1.0 * 5.0 / 100))


def test_critical_export_rate_replaces_the_feedin_rate():
    """All prices above the threshold: 100c replaces 5c (not 105c)."""
    idx = _idx(1)
    bill = compute_bill(_traces(0.0, 1.0, idx), _flat_prices(idx, 5000.0),
                        _network(), _flat_tou(critical_export_rate=_CX), state='NSW')
    assert bill.retail_feedin_credit == pytest.approx(-(N_DAY * 1.0 * 100.0 / 100))


def test_critical_export_rate_partial_spike():
    """50 intervals above the threshold, rest below. Only those 50 change."""
    idx = _idx(1)
    n_spiked = 50
    prices = _flat_prices(idx, 100.0)
    prices.iloc[:n_spiked] = 5000.0
    bill = compute_bill(_traces(0.0, 1.0, idx), prices, _network(),
                        _flat_tou(critical_export_rate=_CX), state='NSW')
    expected = -((N_DAY - n_spiked) * 5.0 + n_spiked * 100.0) / 100
    assert bill.retail_feedin_credit == pytest.approx(expected)


def test_critical_export_rate_does_not_affect_imports():
    """The opposite side of a critical export interval keeps its rate."""
    idx = _idx(1)
    bill = compute_bill(_traces(1.0, 0.0, idx), _flat_prices(idx, 5000.0),
                        _network(), _flat_tou(critical_export_rate=_CX), state='NSW')
    assert bill.retail_usage_cost == pytest.approx(N_DAY * 20.0 / 100)


def test_critical_export_rate_keeps_feedin_other():
    """feedin_other still applies on top of the critical rate."""
    idx = _idx(1)
    rt = RetailTariff(
        name='Critical', state='NSW', type='tou', daily_c_per_day=110.0,
        usage_tariffs=(TariffPeriod(0, 24, 20.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 5.0),),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=1.5,
        critical_export_rate=_CX,
    )
    bill = compute_bill(_traces(0.0, 1.0, idx), _flat_prices(idx, 5000.0),
                        _network(), rt, state='NSW')
    assert bill.retail_feedin_credit == pytest.approx(-(N_DAY * 101.5 / 100))


# -----------------------------------------------------------------------
# Critical import rate
# -----------------------------------------------------------------------

def test_critical_import_rate_no_negative_prices():
    """All prices above the threshold: the standard import rate applies."""
    idx = _idx(1)
    bill = compute_bill(_traces(1.0, 0.0, idx), _flat_prices(idx, 100.0),
                        _network(), _flat_tou(critical_import_rate=_CM), state='NSW')
    assert bill.retail_usage_cost == pytest.approx(N_DAY * 20.0 / 100)


def test_critical_import_rate_replaces_the_import_rate():
    """All prices below the threshold: -5c replaces 20c (the household is
    paid to import), not 20c - 5c."""
    idx = _idx(1)
    bill = compute_bill(_traces(1.0, 0.0, idx), _flat_prices(idx, -300.0),
                        _network(), _flat_tou(critical_import_rate=_CM), state='NSW')
    assert bill.retail_usage_cost == pytest.approx(N_DAY * -5.0 / 100)


def test_critical_import_rate_threshold_is_strict():
    """Exactly at the threshold is not critical, on either side."""
    idx = _idx(1)
    rt = _flat_tou(critical_export_rate=_CX, critical_import_rate=_CM)
    at_exp = compute_bill(_traces(0.0, 1.0, idx), _flat_prices(idx, 2000.0),
                          _network(), rt, state='NSW')
    at_imp = compute_bill(_traces(1.0, 0.0, idx), _flat_prices(idx, -100.0),
                          _network(), rt, state='NSW')
    assert at_exp.retail_feedin_credit == pytest.approx(-(N_DAY * 5.0 / 100))
    assert at_imp.retail_usage_cost == pytest.approx(N_DAY * 20.0 / 100)


# -----------------------------------------------------------------------
# Combined features
# -----------------------------------------------------------------------

def test_block_import_and_zero_import_reward_combined():
    """Block import tiers + zero-import reward on same plan.
    Day 1: zero imports -> reward earned, no block cost.
    Day 2: 60 kWh in window -> reward not earned, tier 2 spillover."""
    idx = _idx(2)
    rt = RetailTariff(
        name='Combined', state='NSW', type='tou', daily_c_per_day=110.0,
        usage_tariffs=(
            TariffPeriod(0, 11, 20.0), TariffPeriod(11, 14, 0.0),
            TariffPeriod(14, 24, 20.0),
        ),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 0.0),),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
        block_import_tiers=(_two_tier_import_window(),),
        zero_import_reward=ZeroImportReward(
            start_hour=11, end_hour=14,
            reward_c_per_day=100.0,
            tolerance_kwh_per_hour=0.0,
        ),
    )
    from model.core.prices import block_window_mask
    mask = block_window_mask(idx, rt.block_import_tiers[0], 'NSW')
    imp_arr = np.zeros(N_DAY * 2)
    # Day 1: zero imports everywhere
    # Day 2: 60 kWh in window (60/36 per interval)
    day2_window_mask = np.zeros(N_DAY * 2, dtype=bool)
    day2_window_mask[N_DAY:] = mask[N_DAY:]
    imp_arr[day2_window_mask] = 60.0 / 36.0
    # Also add outside-window imports on day 2
    day2_outside = np.zeros(N_DAY * 2, dtype=bool)
    day2_outside[N_DAY:] = ~mask[N_DAY:]
    imp_arr[day2_outside] = 1.0
    imp_series = pd.Series(imp_arr, index=idx)
    bill = compute_bill(
        _traces_series(imp_series, pd.Series(np.zeros(N_DAY * 2), index=idx)),
        _flat_prices(idx), _network(), rt, state='NSW')
    # Zero-import reward: day 1 qualifies (0 imports), day 2 does not
    assert bill.retail_other_credits == pytest.approx(-1.0)
    # Block cost: day 1 = $0, day 2 = 0 + 10 * 30.8/100 = $3.08
    # Outside window day 2: 252 * 1.0 * 20/100 = $50.40
    assert bill.retail_usage_cost == pytest.approx(10.0 * 30.8 / 100 + 252 * 20.0 / 100)


def _window_prices(idx, windows_and_values, base=100.0):
    """Flat prices with given values over the first `n` intervals of each
    window: windows_and_values is [(window, n, value), ...]."""
    from model.core.prices import block_window_mask
    arr = np.full(len(idx), float(base))
    for window, n, value in windows_and_values:
        pos = np.flatnonzero(block_window_mask(idx, window, 'NSW'))
        assert len(pos) >= n
        arr[pos[:n]] = value
    return pd.Series(arr, index=idx)


def test_critical_export_in_block_window_replaces_tier_and_leaves_tier():
    """Block export 18-21: tier 1 is 15 kWh at 14c, then 2c. Export 0.5 kWh in
    each of the 36 window intervals (18 kWh). The first 12 are critical.

    Critical: 12 * 0.5 = 6 kWh at 100c, outside the tier.
    Tier:     the other 24 * 0.5 = 12 kWh, all within tier 1 at 14c.
    Counting the critical 6 kWh in the tier would push 3 kWh into tier 2, and
    stacking the tier rate on the critical rate would add 6 * 14c.
    """
    idx = _idx(1)
    rt = _tou_with_block_export(critical_export_rate=_CX)
    window = rt.block_export_tiers[0]
    prices = _window_prices(idx, [(window, 12, 5000.0)])
    from model.core.prices import block_window_mask
    in_win = block_window_mask(idx, window, 'NSW')
    exp = pd.Series(np.where(in_win, 0.5, 0.0), index=idx)
    bill = compute_bill(_traces_series(pd.Series(0.0, index=idx), exp), prices,
                        _network(), rt, state='NSW')
    critical = -(6.0 * 100.0 / 100)
    tier = -(12.0 * 14.0 / 100)
    assert bill.retail_feedin_credit == pytest.approx(critical + tier)


def test_critical_import_in_block_window_replaces_tier_and_leaves_tier():
    """Block import 11-14: tier 1 is 50 kWh at 0c, then 30.8c. Import 2 kWh in
    each of the 36 window intervals (72 kWh). The first 18 are critical.

    Critical: 18 * 2 = 36 kWh at -5c, outside the tier.
    Tier:     the other 36 kWh, all within tier 1 at 0c.
    Counting the critical 36 kWh in the tier would put 22 kWh at 30.8c.
    Outside the window: 252 intervals of zero import.
    """
    idx = _idx(1)
    rt = _tou_with_block_import(critical_import_rate=_CM)
    window = rt.block_import_tiers[0]
    prices = _window_prices(idx, [(window, 18, -300.0)])
    from model.core.prices import block_window_mask
    in_win = block_window_mask(idx, window, 'NSW')
    imp = pd.Series(np.where(in_win, 2.0, 0.0), index=idx)
    bill = compute_bill(_traces_series(imp, pd.Series(0.0, index=idx)), prices,
                        _network(), rt, state='NSW')
    assert bill.retail_usage_cost == pytest.approx(36.0 * -5.0 / 100 + 0.0)


def test_opposite_side_critical_interval_stays_in_its_tier():
    """An export-critical interval inside the IMPORT window: the import there
    is unaffected and still counts towards the import tier.

    Import 2 kWh in all 36 window intervals (72 kWh) with the whole window
    export-critical: 50 kWh at 0c + 22 kWh at 30.8c, the same as with no
    critical rate at all.
    """
    idx = _idx(1)
    rt = _tou_with_block_import(critical_export_rate=_CX)
    window = rt.block_import_tiers[0]
    prices = _window_prices(idx, [(window, 36, 5000.0)])
    from model.core.prices import block_window_mask
    in_win = block_window_mask(idx, window, 'NSW')
    imp = pd.Series(np.where(in_win, 2.0, 0.0), index=idx)
    bill = compute_bill(_traces_series(imp, pd.Series(0.0, index=idx)), prices,
                        _network(), rt, state='NSW')
    plain = compute_bill(_traces_series(imp, pd.Series(0.0, index=idx)), prices,
                         _network(), _tou_with_block_import(), state='NSW')
    assert bill.retail_usage_cost == pytest.approx(22.0 * 30.8 / 100)
    assert bill.retail_usage_cost == pytest.approx(plain.retail_usage_cost)


def test_lp_price_and_billing_consistent_with_critical_intervals_in_block_windows():
    """The LP prices a critical interval at its critical rate with no tier
    variable (effective masks), and shifts the opposite side by delta. Billing
    charges the real rates. So, for any flows:

        bill = sum(imp * p_imp_lp - exp * p_exp_lp) + tier cost(effective masks)
               - sum over critical intervals of delta * opposite-side flow

    Flows here are non-zero on BOTH sides in critical intervals inside both
    windows, so the opposite-side term, the tier exclusion and the rate
    replacement are all exercised.
    """
    from model.core.prices import (
        build_lp_price_vectors_advanced, block_tier_masks, critical_masks,
        block_window_mask,
    )
    from model.components.billing import _block_tier_components
    idx = _idx(1)
    net = NetworkTariff(
        name='N', state='NSW',
        usage_tariffs=(TariffPeriod(0, 24, 0.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 0.0),),
        daily_c_per_day=0.0,
        usage_dlf=1.0, export_dlf=1.0, import_tlf=1.0, export_tlf=1.0,
    )
    rt = RetailTariff(
        name='LPBillingCritical', state='NSW', type='tou', daily_c_per_day=0.0,
        usage_tariffs=(
            TariffPeriod(0, 11, 20.0), TariffPeriod(11, 14, 0.0),
            TariffPeriod(14, 24, 20.0),
        ),
        usage_tariffs_weekend=None,
        feedin_tariffs=(
            TariffPeriod(0, 18, 5.0), TariffPeriod(18, 21, 0.0),
            TariffPeriod(21, 24, 5.0),
        ),
        usage_other_c_per_kwh=0.8, feedin_other_c_per_kwh=0.0,
        block_import_tiers=(_two_tier_import_window(),),
        block_export_tiers=(_two_tier_export_window(),),
        critical_export_rate=_CX, critical_import_rate=_CM,
    )
    wi, we = rt.block_import_tiers[0], rt.block_export_tiers[0]
    prices = _window_prices(idx, [(wi, 10, -300.0), (we, 10, 5000.0)])
    arr = prices.to_numpy().copy()
    arr[:6] = 5000.0          # export-critical outside both windows
    arr[100:106] = -300.0     # import-critical outside both windows
    prices = pd.Series(arr, index=idx)

    rng = np.random.default_rng(7)
    imp = pd.Series(rng.uniform(0.0, 1.5, len(idx)), index=idx)
    exp = pd.Series(rng.uniform(0.0, 0.8, len(idx)), index=idx)

    x_mask, m_mask = critical_masks(prices, rt)
    assert (x_mask & block_window_mask(idx, we, 'NSW')).any()
    assert (m_mask & block_window_mask(idx, wi, 'NSW')).any()
    assert (imp.to_numpy()[x_mask] > 0).all() and (exp.to_numpy()[m_mask] > 0).all()

    p_imp, p_exp = build_lp_price_vectors_advanced(idx, prices, net, rt, 'NSW', 5/60)
    per_interval = float(np.sum(imp.to_numpy() * p_imp.to_numpy()
                                - exp.to_numpy() * p_exp.to_numpy()))
    imp_masks, exp_masks = block_tier_masks(idx, rt, 'NSW', 5/60, prices)
    zeros = np.zeros(len(idx))
    _, imp_days = _block_tier_components(imp, idx, rt.block_import_tiers, zeros,
                                         True, 5/60, 'NSW', masks=imp_masks)
    _, exp_days = _block_tier_components(exp, idx, rt.block_export_tiers, zeros,
                                         False, 5/60, 'NSW', masks=exp_masks)
    tier = float(imp_days.sum() + exp_days.sum())
    # delta = LP shift on the opposite side = p_lp - billing rate there.
    from model.core.prices import build_base_price_vectors_advanced
    b_imp, b_exp = build_base_price_vectors_advanced(idx, prices, net, rt, 'NSW', 5/60)
    delta_x = (1.00 + 0.0) - b_exp.to_numpy()
    delta_m = (-0.05 + 0.008) - b_imp.to_numpy()
    opposite = float(np.sum(imp.to_numpy()[x_mask] * delta_x[x_mask])
                     - np.sum(exp.to_numpy()[m_mask] * delta_m[m_mask]))

    bill = compute_bill(_traces_series(imp, exp), prices, net, rt, state='NSW')
    bill_dispatch = bill.retail_usage_cost + bill.retail_feedin_credit
    assert abs(opposite) > 0.1, "fixture must exercise the opposite-side shift"
    assert bill_dispatch == pytest.approx(per_interval + tier - opposite, abs=1e-9)


def test_globird_plan_daily_structure():
    """Smoke test: load the GloBird Advanced TOU 2025 tariff and verify
    compute_bill runs without error and household_total is plausible."""
    from pathlib import Path
    real_dir = Path(__file__).parent.parent / 'tariffs'
    if not (real_dir / 'globird_advanced_tou_2025.yaml').exists():
        pytest.skip("globird_advanced_tou_2025.yaml not present in model/tariffs/")

    from model.core.tariff import load_retail_tariff
    rt = load_retail_tariff('globird_advanced_tou_2025', real_dir)

    idx = _idx(1)
    # Moderate imports and exports
    bill = compute_bill(_traces(0.5, 0.3, idx), _flat_prices(idx, 100.0),
                        _network(), rt, state='NSW')

    # Sanity: household_total equals retail_net + retail_other_credits for TOU
    retail_net = (bill.retail_usage_cost + bill.retail_feedin_credit
                  + bill.retail_daily_charges)
    assert bill.household_total == pytest.approx(retail_net + bill.retail_other_credits)

    # With 0.5 kWh/interval imports and flat 100$/MWh prices (below all thresholds),
    # no critical rates or zero-import reward should fire
    assert bill.retail_other_credits == pytest.approx(0.0)
    # Block import: 36 intervals * 0.5 = 18 kWh in 11-14 window, < 50 kWh tier 1 at 0c
    # so block cost = 0 for that window; no greenpower (zeroed for validation)
    assert bill.retail_usage_cost >= 0.0

# -----------------------------------------------------------------------
# LP price vectors vs billing consistency
# -----------------------------------------------------------------------

def test_lp_price_vectors_block_window_uses_usage_other_not_tou():
    """build_lp_price_vectors_advanced sets p_imp[mask] = usage_other_rate
    (not zero, not TOU rate) for block import window intervals.

    This ensures the LP correctly accounts for the flat levy (greenpower)
    when optimising block window imports, consistent with billing which
    also preserves usage_other for window intervals.
    """
    from model.core.prices import build_lp_price_vectors_advanced
    from model.core.tariff import NetworkTariff
    idx = pd.date_range('2022-07-04 11:05:00', periods=36, freq='5min',
                        tz='Etc/GMT-10')   # interval-END timestamps; start instants 11:00-13:55 all in window [11,14)
    net = NetworkTariff(
        name='N', state='NSW',
        usage_tariffs=(TariffPeriod(0, 24, 0.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 0.0),),
        daily_c_per_day=0.0,
        usage_dlf=1.0, export_dlf=1.0, import_tlf=1.0, export_tlf=1.0,
    )
    rt = RetailTariff(
        name='BlockGreenpower', state='NSW', type='tou', daily_c_per_day=0.0,
        usage_tariffs=(
            TariffPeriod(0, 11, 20.0), TariffPeriod(11, 14, 0.0),
            TariffPeriod(14, 24, 20.0),
        ),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 5.0),),
        usage_other_c_per_kwh=3.0,   # greenpower levy
        feedin_other_c_per_kwh=0.0,
        block_import_tiers=(_two_tier_import_window(),),
    )
    prices_mwh = pd.Series(100.0, index=idx)
    p_imp, p_exp = build_lp_price_vectors_advanced(idx, prices_mwh, net, rt, 'NSW', 5/60)

    usage_other_rate = 3.0 / 100   # $/kWh
    # All 36 intervals are in window: p_imp = usage_other_rate, not 0, not 23c
    import numpy.testing as npt
    npt.assert_allclose(p_imp.to_numpy(), usage_other_rate, atol=1e-9)
    npt.assert_allclose(p_exp.to_numpy(), 5.0 / 100, atol=1e-9)


def test_lp_price_and_billing_consistent_for_block_window_with_usage_other():
    """LP p_imp[mask] = usage_other_rate and billing charges tier_rate +
    usage_other. Both treat usage_other consistently for block window intervals.

    For 1 kWh/interval in window with tier1_rate=0c and usage_other=3c:
      LP price   = 3c/kWh (usage_other; tier captured separately by tier variables)
      Bill cost  = 0c (tier1) + 3c (usage_other) = 3c/kWh  ← must match
    """
    from model.core.prices import build_lp_price_vectors_advanced
    from model.core.tariff import NetworkTariff
    idx = pd.date_range('2022-07-04 11:05:00', periods=36, freq='5min',
                        tz='Etc/GMT-10')   # start instants 11:00-13:55, all in window
    net = NetworkTariff(
        name='N', state='NSW',
        usage_tariffs=(TariffPeriod(0, 24, 0.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 0.0),),
        daily_c_per_day=0.0,
        usage_dlf=1.0, export_dlf=1.0, import_tlf=1.0, export_tlf=1.0,
    )
    rt = RetailTariff(
        name='LPBillingConsistency', state='NSW', type='tou', daily_c_per_day=0.0,
        usage_tariffs=(
            TariffPeriod(0, 11, 20.0), TariffPeriod(11, 14, 0.0),
            TariffPeriod(14, 24, 20.0),
        ),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 0.0),),
        usage_other_c_per_kwh=3.0,
        feedin_other_c_per_kwh=0.0,
        block_import_tiers=(_two_tier_import_window(),),
    )
    prices_mwh = pd.Series(100.0, index=idx)

    # LP price for window intervals
    p_imp, _ = build_lp_price_vectors_advanced(idx, prices_mwh, net, rt, 'NSW', 5/60)
    lp_price_rate = float(p_imp.iloc[0])   # $/kWh; should be 3c/100

    # Billing: 1 kWh/interval, all in window (tier1).
    # _traces passes kWh/interval directly; 36 intervals * 1.0 kWh = 36 kWh total.
    imp_per_interval = 1.0   # kWh/interval
    bill = compute_bill(
        _traces(imp_per_interval, 0.0, idx), prices_mwh, net, rt,
        interval_hours=5/60, state='NSW')
    total_window_kwh = 36.0 * imp_per_interval   # kWh (no interval_hours factor)
    expected_rate = 3.0 / 100   # tier1=0c + usage_other=3c

    assert lp_price_rate == pytest.approx(expected_rate, abs=1e-9)
    # billing charges usage_other (preserved) + tier1 (0c) for all window kWh
    assert bill.retail_usage_cost == pytest.approx(
        total_window_kwh * expected_rate, abs=1e-6)