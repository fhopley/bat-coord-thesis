"""Tests for compute_monthly_bills in billing.py.

Mirrors the fixture conventions of test_billing.py / test_billing_advanced.py
/ test_billing_vpp.py. State is fixed to QLD throughout: QLD's local clock
matches AEST year-round (no DST, no permanent offset -- see
test_localtime.py's "control case"), so local-calendar-month splitting and
AEST-calendar-day splitting agree exactly. This isolates the tests below
from the documented SA/DST daily-charge boundary caveat in
compute_monthly_bills' docstring, which is deliberately out of scope here.

Each test either:
  (a) checks the split itself (which months are present, that traces/
      vpp_traces are sliced correctly, that a single-month period reduces
      to a plain compute_bill call), or
  (b) checks reconciliation: monthly bills, summed, equal the equivalent
      compute_bill() call on the full un-sliced traces.

Since the Level 2 rewrite of billing.py (_build_settlement_base /
_reduce_settlement), compute_monthly_bills no longer just calls compute_bill
per month -- it builds settlement components once and aggregates them
per month, including for block tiers and the zero-import reward (previously
day-grouped scalars, now day-indexed pd.Series filtered by local month).
That refactor is exactly the part most likely to have introduced a subtle
bug, so this file explicitly exercises both advanced features across a
month boundary, not just plain TOU/wholesale plans.
"""
import math

import numpy as np
import pandas as pd
import pytest

from model.core.tariff import (
    NetworkTariff, RetailTariff, TariffPeriod, VppConfig,
    BlockTier, BlockTariffWindow, ZeroImportReward,
)
from model.core.site_result import SiteTraces, VppTraces
from model.components.billing import compute_bill, compute_monthly_bills


STATE = 'QLD'
N_DAY = 288  # 5-min intervals per day


def _idx(start, n_days):
    return pd.date_range(start, periods=N_DAY * n_days, freq='5min',
                         tz='Etc/GMT-10')


def _traces(imp_kwh, exp_kwh, index):
    """Uniform per-interval import/export; rest of SiteTraces zeroed."""
    n = len(index)
    imp = pd.Series(np.full(n, float(imp_kwh)), index=index)
    exp = pd.Series(np.full(n, float(exp_kwh)), index=index)
    z = pd.Series(np.zeros(n), index=index)
    return SiteTraces(
        site_id='S0001',
        import_kwh=imp, export_kwh=exp, battery_soc_kwh=z,
        sol_to_home_kwh=z, sol_to_bat_kwh=z, sol_to_grid_kwh=z,
        bat_to_home_kwh=z, grid_to_home_kwh=imp, grid_to_bat_kwh=z,
        bat_to_grid_kwh=exp, curtail_kwh=z,
    )


def _network(usage_c=10.0, feedin_c=0.0, daily_c=100.0):
    return NetworkTariff(
        name='TestNet', state=STATE,
        usage_tariffs=(TariffPeriod(0, 24, usage_c),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, feedin_c),),
        daily_c_per_day=daily_c,
        usage_dlf=1.0, export_dlf=1.0, import_tlf=1.0, export_tlf=1.0,
    )


def _tou_retail(usage_c=20.0, feedin_c=5.0, daily_c=50.0):
    return RetailTariff(
        name='TestTOU', state=STATE, type='tou', daily_c_per_day=daily_c,
        usage_tariffs=(TariffPeriod(0, 24, usage_c),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, feedin_c),),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
    )


def _flat_prices(index, mwh=100.0):
    return pd.Series(np.full(len(index), float(mwh)), index=index)


# ---------------- Which months are present ----------------

def test_monthly_bills_keys_match_months_spanned():
    """A Dec + Jan period yields exactly the keys {12, 1}."""
    idx = _idx('2021-12-01 00:05:00', 62)  # Dec (31d) + Jan (31d)
    traces = _traces(1.0, 0.0, idx)
    prices = _flat_prices(idx)
    monthly = compute_monthly_bills(traces, prices, _network(), _tou_retail(),
                                    state=STATE)
    assert set(monthly.keys()) == {12, 1}


def test_monthly_bills_omits_months_not_present():
    """A single-month period yields exactly one key."""
    idx = _idx('2022-03-01 00:05:00', 31)  # all of March
    traces = _traces(1.0, 0.0, idx)
    prices = _flat_prices(idx)
    monthly = compute_monthly_bills(traces, prices, _network(), _tou_retail(),
                                    state=STATE)
    assert set(monthly.keys()) == {3}


# ---------------- Single full month reduces to compute_bill ----------------

def test_monthly_bill_matches_compute_bill_for_full_month_period():
    """Splitting a period that IS exactly one calendar month should return
    a bill identical (field-by-field) to compute_bill on the whole thing."""
    idx = _idx('2022-06-01 00:05:00', 30)  # all of June
    traces = _traces(2.5, 1.0, idx)
    prices = _flat_prices(idx, mwh=80.0)
    net = _network()
    retail = _tou_retail()

    annual_equiv = compute_bill(traces, prices, net, retail, state=STATE)
    monthly = compute_monthly_bills(traces, prices, net, retail, state=STATE)

    assert set(monthly.keys()) == {6}
    bill = monthly[6]
    assert bill.site_id == annual_equiv.site_id
    assert bill.household_total == pytest.approx(annual_equiv.household_total)
    assert bill.network_daily_charges == pytest.approx(
        annual_equiv.network_daily_charges)
    assert bill.retail_daily_charges == pytest.approx(
        annual_equiv.retail_daily_charges)
    assert bill.retailer_margin == pytest.approx(annual_equiv.retailer_margin)


# ---------------- Reconciliation: monthly sums equal the annual bill ----------------

def test_monthly_household_totals_sum_to_annual_tou():
    idx = _idx('2021-12-01 00:05:00', 62)  # Dec + Jan
    traces = _traces(3.0, 0.5, idx)
    prices = _flat_prices(idx, mwh=90.0)
    net = _network(usage_c=12.0, daily_c=100.0)
    retail = _tou_retail(usage_c=25.0, feedin_c=4.0, daily_c=60.0)

    annual = compute_bill(traces, prices, net, retail, state=STATE)
    monthly = compute_monthly_bills(traces, prices, net, retail, state=STATE)

    total = sum(b.household_total for b in monthly.values())
    assert total == pytest.approx(annual.household_total)


def test_monthly_daily_charges_sum_to_annual_days():
    """network_daily_charges/retail_daily_charges, summed across months,
    should reconcile with the annual figure (QLD -- no AEST/local mismatch
    at month boundaries, so no partial-day double counting or gaps)."""
    idx = _idx('2021-12-01 00:05:00', 62)  # 31 + 31 = 62 days
    traces = _traces(1.0, 0.0, idx)
    prices = _flat_prices(idx)
    net = _network(daily_c=100.0)
    retail = _tou_retail(daily_c=55.0)

    annual = compute_bill(traces, prices, net, retail, state=STATE)
    monthly = compute_monthly_bills(traces, prices, net, retail, state=STATE)

    assert sum(b.network_daily_charges for b in monthly.values()) == pytest.approx(
        annual.network_daily_charges)
    assert sum(b.retail_daily_charges for b in monthly.values()) == pytest.approx(
        annual.retail_daily_charges)
    # sanity: 62 days at 100c/day = $62.00
    assert annual.network_daily_charges == pytest.approx(62.0)


def test_monthly_bills_reconcile_with_wholesale_exposed_plan():
    idx = _idx('2022-06-01 00:05:00', 61)  # June + July
    traces = _traces(4.0, 2.0, idx)
    prices = _flat_prices(idx, mwh=70.0)
    net = _network(usage_c=10.0, daily_c=100.0)
    retail = RetailTariff(
        name='TestWh', state=STATE, type='wholesale_exposed',
        daily_c_per_day=80.0,
        usage_tariffs=(), usage_tariffs_weekend=None, feedin_tariffs=(),
        usage_other_c_per_kwh=2.0, feedin_other_c_per_kwh=0.0,
    )

    annual = compute_bill(traces, prices, net, retail, state=STATE)
    monthly = compute_monthly_bills(traces, prices, net, retail, state=STATE)

    total = sum(b.household_total for b in monthly.values())
    assert total == pytest.approx(annual.household_total)
    margin_total = sum(b.retailer_margin for b in monthly.values())
    assert margin_total == pytest.approx(annual.retailer_margin)


# ---------------- Advanced features across a month boundary ----------------
# The Level 2 rewrite of billing.py changed _block_tier_components and
# _zero_import_day_credits to return per-local-date pd.Series (filtered by
# month in _reduce_settlement) instead of pre-summed scalars. These tests
# specifically exercise that code path across a real month boundary --
# previously compute_monthly_bills just called compute_bill unchanged, so
# there was nothing month-specific about block tiers/zero-import reward to
# test; that is no longer true.

def _two_tier_import_window(start=11, end=14, t1_kwh=50.0,
                             t1_rate=0.0, t2_rate=30.8):
    """GloBird-style: first 50 kWh free, excess at 30.8c. 3-hour window."""
    return BlockTariffWindow(
        start_hour=start, end_hour=end,
        tiers=(
            BlockTier(threshold_kwh=t1_kwh, rate_c_per_kwh=t1_rate),
            BlockTier(threshold_kwh=math.inf, rate_c_per_kwh=t2_rate),
        ),
    )


def _tou_with_block_import(**kwargs):
    """TOU plan with block_import_tiers, window 11-14. Standard rate is 0
    in the window (overridden by block tiers) and 20c elsewhere."""
    return RetailTariff(
        name='BlockImport', state=STATE, type='tou', daily_c_per_day=110.0,
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


def test_monthly_bills_reconcile_with_block_import_tiers():
    """2 kWh/interval constant import -> 36 intervals x 2 kWh = 72 kWh/day
    inside the 11-14 window, spilling 22 kWh/day into tier 2 (30.8c) every
    day of a Dec+Jan span. household_total, summed across the two monthly
    bills, must equal the annual bill -- exercising the block-tier day
    Series being correctly filtered by local month rather than summed once
    for the whole period."""
    idx = _idx('2021-12-01 00:05:00', 62)  # Dec + Jan
    traces = _traces(2.0, 0.0, idx)
    prices = _flat_prices(idx)
    net = _network(daily_c=100.0)
    retail = _tou_with_block_import()

    annual = compute_bill(traces, prices, net, retail, state=STATE)
    monthly = compute_monthly_bills(traces, prices, net, retail, state=STATE)

    assert set(monthly.keys()) == {12, 1}

    total_household = sum(b.household_total for b in monthly.values())
    assert total_household == pytest.approx(annual.household_total)

    total_retail_usage = sum(b.retail_usage_cost for b in monthly.values())
    assert total_retail_usage == pytest.approx(annual.retail_usage_cost)

    # Sanity: tier spillover is actually happening, not a degenerate zero.
    naive_retail_usage = 2.0 * N_DAY * 62 * 0.20  # if block tiers had no effect
    assert total_retail_usage < naive_retail_usage


def test_monthly_bills_split_block_import_tier_days_correctly():
    """A period spanning exactly Dec (31 days) + Jan (31 days): December's
    monthly bill's retail_usage_cost should reconcile against a standalone
    compute_bill on December-only data, and likewise for January -- i.e.
    the per-day tier cost Series is being filtered to the right days, not
    just the right count of days."""
    dec_idx = _idx('2021-12-01 00:05:00', 31)
    full_idx = _idx('2021-12-01 00:05:00', 62)  # Dec + Jan

    prices_dec = _flat_prices(dec_idx)
    prices_full = _flat_prices(full_idx)
    net = _network(daily_c=100.0)
    retail = _tou_with_block_import()

    dec_only_bill = compute_bill(
        _traces(2.0, 0.0, dec_idx), prices_dec, net, retail, state=STATE)
    monthly = compute_monthly_bills(
        _traces(2.0, 0.0, full_idx), prices_full, net, retail, state=STATE)

    # December's contribution to the 2-month traces should match a
    # standalone December-only bill, since the import pattern is identical
    # (constant 2 kWh/interval) in both cases and December's calendar
    # days/hours are unaffected by whether January follows.
    assert monthly[12].retail_usage_cost == pytest.approx(
        dec_only_bill.retail_usage_cost)
    assert monthly[12].household_total == pytest.approx(
        dec_only_bill.household_total)


def test_monthly_bills_reconcile_with_zero_import_reward():
    """Zero imports every day, reward window 18-21h, across a Dec+Jan span.
    Every day should qualify; the monthly retail_other_credits, summed,
    must equal the annual credit (62 days x -$1.00/day)."""
    idx = _idx('2021-12-01 00:05:00', 62)
    traces = _traces(0.0, 0.0, idx)  # zero import every interval -> qualifies daily
    prices = _flat_prices(idx)
    net = _network(daily_c=0.0)
    retail = RetailTariff(
        name='ZIR', state=STATE, type='tou', daily_c_per_day=0.0,
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

    annual = compute_bill(traces, prices, net, retail, state=STATE)
    monthly = compute_monthly_bills(traces, prices, net, retail, state=STATE)

    assert set(monthly.keys()) == {12, 1}
    total_credits = sum(b.retail_other_credits for b in monthly.values())
    assert total_credits == pytest.approx(annual.retail_other_credits)
    assert annual.retail_other_credits == pytest.approx(-62.0)

    # December's 31 days and January's 31 days should split evenly, since
    # every day qualifies identically.
    assert monthly[12].retail_other_credits == pytest.approx(-31.0)
    assert monthly[1].retail_other_credits == pytest.approx(-31.0)


def test_monthly_bills_zero_import_reward_partial_qualification():
    """December: zero imports (qualifies every day). January: imports in
    the reward window every day (never qualifies). Monthly credits should
    reflect that split, not an even distribution across the two months."""
    dec_idx = _idx('2021-12-01 00:05:00', 31)
    jan_idx = _idx('2022-01-01 00:05:00', 31)
    full_idx = dec_idx.append(jan_idx)

    n_dec = len(dec_idx)
    n_jan = len(jan_idx)
    imp = np.concatenate([np.zeros(n_dec), np.full(n_jan, 1.0)])
    imp_s = pd.Series(imp, index=full_idx)
    exp_s = pd.Series(np.zeros(len(full_idx)), index=full_idx)
    z = pd.Series(np.zeros(len(full_idx)), index=full_idx)
    traces = SiteTraces(
        site_id='S0001', import_kwh=imp_s, export_kwh=exp_s,
        battery_soc_kwh=z, sol_to_home_kwh=z, sol_to_bat_kwh=z,
        sol_to_grid_kwh=z, bat_to_home_kwh=z, grid_to_home_kwh=imp_s,
        grid_to_bat_kwh=z, bat_to_grid_kwh=exp_s, curtail_kwh=z,
    )
    prices = _flat_prices(full_idx)
    net = _network(daily_c=0.0)
    retail = RetailTariff(
        name='ZIR', state=STATE, type='tou', daily_c_per_day=0.0,
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

    annual = compute_bill(traces, prices, net, retail, state=STATE)
    monthly = compute_monthly_bills(traces, prices, net, retail, state=STATE)

    total_credits = sum(b.retail_other_credits for b in monthly.values())
    assert total_credits == pytest.approx(annual.retail_other_credits)

    # December (zero imports) qualifies every day; January (imports in
    # window) qualifies none.
    assert monthly[12].retail_other_credits == pytest.approx(-31.0)
    assert monthly[1].retail_other_credits == pytest.approx(0.0)


# ---------------- vpp_traces slicing ----------------

def test_monthly_bills_vpp_traces_sliced_per_month():
    """Forced export in December only should show up as a retail_other_credits
    reward in month 12 and be exactly zero in month 1."""
    idx = _idx('2021-12-01 00:05:00', 62)
    n = len(idx)
    is_dec = idx.month == 12

    imp = pd.Series(np.zeros(n), index=idx)
    exp = pd.Series(np.zeros(n), index=idx)
    traces = SiteTraces(
        site_id='S0001', import_kwh=imp, export_kwh=exp,
        battery_soc_kwh=pd.Series(np.zeros(n), index=idx),
        sol_to_home_kwh=pd.Series(np.zeros(n), index=idx),
        sol_to_bat_kwh=pd.Series(np.zeros(n), index=idx),
        sol_to_grid_kwh=pd.Series(np.zeros(n), index=idx),
        bat_to_home_kwh=pd.Series(np.zeros(n), index=idx),
        grid_to_home_kwh=imp,
        grid_to_bat_kwh=pd.Series(np.zeros(n), index=idx),
        bat_to_grid_kwh=exp,
        curtail_kwh=pd.Series(np.zeros(n), index=idx),
    )

    # 1 kWh/interval forced export, December intervals only.
    f_exp = pd.Series(np.where(is_dec, 1.0, 0.0), index=idx)
    z = pd.Series(np.zeros(n), index=idx)
    vpp_t = VppTraces(site_id='S0001', n_imp_kwh=z, f_imp_kwh=z,
                      n_exp_kwh=z, f_exp_kwh=f_exp)

    cfg = VppConfig(
        exp_reward_c_per_kwh=100.0, imp_reward_c_per_kwh=100.0,
        kwh_limit=10_000.0, limit_applies_to='export_only',
        default_mode='self_consumption', event_window_hours=1.0,
        vpp_threshold_dollars_per_mwh=100.0,
    )
    retail = RetailTariff(
        name='VppPlan', state=STATE, type='tou', daily_c_per_day=0.0,
        usage_tariffs=(TariffPeriod(0, 24, 20.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 5.0),),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
        vpp=cfg,
    )
    prices = _flat_prices(idx)
    net = _network(daily_c=0.0)

    monthly = compute_monthly_bills(traces, prices, net, retail, state=STATE,
                                    vpp_traces=vpp_t)

    n_dec_intervals = int(is_dec.sum())
    expected_dec_reward = -(n_dec_intervals * 1.0 * 1.0)  # 100c/kWh = $1/kWh
    assert monthly[12].retail_other_credits == pytest.approx(expected_dec_reward)
    assert monthly[1].retail_other_credits == pytest.approx(0.0)


def test_monthly_bills_site_id_preserved_across_slices():
    idx = _idx('2022-06-01 00:05:00', 30)
    traces = _traces(1.0, 0.0, idx)
    prices = _flat_prices(idx)
    monthly = compute_monthly_bills(traces, prices, _network(), _tou_retail(),
                                    state=STATE)
    for bill in monthly.values():
        assert bill.site_id == 'S0001'


# ---------- DST year-boundary daily-charge day count (NSW/VIC/SA) ----------
#
# NSW/VIC/SA observe DST; the last AEST hour of 31 December converts to
# local 1 January of the FOLLOWING year (Australian DST runs roughly
# October-April, so New Year's Eve/Day always falls inside it for these
# three states). Since month buckets are keyed by bare month number, not
# year, that one sliver lands in the same January bucket as the entire
# preceding January. _span_days handles this by counting distinct
# (month, day) pairs, ignoring year: the stray "1 January of next year"
# collapses onto the SAME (month=1, day=1) key already present from local
# 1 January of the year being billed, rather than counting as a 32nd
# distinct day. This gives EXACT reconciliation between the annual bill
# and the sum of the 12 monthly bills, with no residual at all -- see
# _span_days' docstring for the full reasoning and the two earlier,
# incomplete attempts at this fix (one that left a single-day residual by
# counting full dates including year; an even earlier, over-generalised
# one that counted AEST rather than local dates and left a multi-day
# residual scattered across several month boundaries).
#
# QLD is used everywhere else in this file specifically to AVOID DST
# entirely while testing the slicing logic in isolation; this is
# deliberately the one test in the file that uses a DST-observing state,
# on purpose, to pin this exact mechanism.

def _full_calendar_year_index(year: int) -> pd.DatetimeIndex:
    """Matches ingest_fixture_builder.build_datetime_index exactly: AEST
    00:05 on Jan 1 through 00:00 on Jan 1 the following year, 5-min."""
    start = pd.Timestamp(f'{year}-01-01 00:05:00', tz='Etc/GMT-10')
    end = pd.Timestamp(f'{year + 1}-01-01 00:00:00', tz='Etc/GMT-10')
    return pd.date_range(start=start, end=end, freq='5min')


def test_monthly_bills_nsw_dst_year_boundary_reconciles_exactly():
    """NSW (DST-observing), a full real calendar year (2022, non-leap).

    Pins exact reconciliation: annual daily-charge day count is 365;
    the SUM of the 12 monthly day counts is ALSO exactly 365, not 366
    (an earlier, incomplete fix) or 371 (an even earlier, over-generalised
    one) -- see this file's module-level comment above and _span_days'
    docstring for the full mechanism. December and January are each
    exactly their own 31 days, with no cross-year duplicate absorbed into
    either.
    """
    idx = _full_calendar_year_index(2022)  # NSW, non-leap year, 365 days
    traces = _traces(1.0, 0.0, idx)
    prices = _flat_prices(idx)
    net = _network(daily_c=100.0)
    retail = _tou_retail(daily_c=50.0)

    annual = compute_bill(traces, prices, net, retail, state='NSW')
    monthly = compute_monthly_bills(traces, prices, net, retail, state='NSW')

    # Annual: exactly 365 days for a non-leap year.
    assert annual.network_daily_charges == pytest.approx(100.0 / 100.0 * 365)
    assert annual.retail_daily_charges == pytest.approx(50.0 / 100.0 * 365)

    total_network_daily = sum(b.network_daily_charges for b in monthly.values())
    total_retail_daily = sum(b.retail_daily_charges for b in monthly.values())

    # Exact reconciliation: no residual at all, in either direction.
    assert total_network_daily == pytest.approx(annual.network_daily_charges)
    assert total_retail_daily == pytest.approx(annual.retail_daily_charges)

    # December and January should each be exactly 31 days -- neither one
    # picking up (or losing) the cross-year duplicate.
    assert monthly[12].network_daily_charges == pytest.approx(
        100.0 / 100.0 * 31)
    assert monthly[1].network_daily_charges == pytest.approx(
        100.0 / 100.0 * 31)

    # household_total should reconcile exactly.
    total_household = sum(b.household_total for b in monthly.values())
    assert total_household == pytest.approx(annual.household_total)

def test_monthly_bills_qld_no_dst_residual():
    """Control case: QLD has no DST (AEST = local year-round), so a full
    real calendar year should reconcile EXACTLY -- annual and summed-
    monthly day counts both 365, no residual at all. If this test starts
    failing, the bug is not DST-related and needs different treatment."""
    idx = _full_calendar_year_index(2022)
    traces = _traces(1.0, 0.0, idx)
    prices = _flat_prices(idx)
    net = _network(daily_c=100.0)
    retail = _tou_retail(daily_c=50.0)

    annual = compute_bill(traces, prices, net, retail, state='QLD')
    monthly = compute_monthly_bills(traces, prices, net, retail, state='QLD')

    total_network_daily = sum(b.network_daily_charges for b in monthly.values())
    assert total_network_daily == pytest.approx(annual.network_daily_charges)

    total_household = sum(b.household_total for b in monthly.values())
    assert total_household == pytest.approx(annual.household_total)