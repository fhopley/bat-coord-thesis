"""Tests for billing.py.

Uses small hand-constructable inputs with analytically known answers.
Tariff fixtures are constructed inline (not loaded from YAML) so the
expected arithmetic is fully visible in each test.

Key fixtures use round numbers:
  - network: usage 10 c/kWh flat, feedin 0 c/kWh flat, daily 100 c/day,
    loss factors import 1.0 / export 1.0 (so loss factors don't obscure
    the arithmetic unless a test sets them otherwise).
  - prices: flat 100 $/MWh = 0.1 $/kWh.
"""
import numpy as np
import pandas as pd
import pytest

from model.core.tariff import NetworkTariff, RetailTariff, TariffPeriod
from model.core.site_result import SiteTraces
from model.components.billing import compute_bill, bill_all_sites, _n_days


def _idx(n, freq='5min'):
    return pd.date_range('2022-01-03 00:05:00', periods=n, freq=freq,
                         tz='Etc/GMT-10')  # 2022-01-03 is a Monday


def _traces(import_kwh, export_kwh, index):
    z = pd.Series(np.zeros(len(index)), index=index)
    return SiteTraces(
        site_id='S0001',
        import_kwh=pd.Series(import_kwh, index=index),
        export_kwh=pd.Series(export_kwh, index=index),
        battery_soc_kwh=z, sol_to_home_kwh=z, sol_to_bat_kwh=z,
        sol_to_grid_kwh=z, bat_to_home_kwh=z, grid_to_home_kwh=z,
        grid_to_bat_kwh=z, bat_to_grid_kwh=z, curtail_kwh=z,
    )


def _flat_network(usage_c=10.0, feedin_c=0.0, daily_c=100.0,
                  usage_dlf=1.0, export_dlf=1.0, import_tlf=1.0, export_tlf=1.0):
    return NetworkTariff(
        name='TestNet', state='NSW',
        usage_tariffs=(TariffPeriod(0, 24, usage_c),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, feedin_c),),
        daily_c_per_day=daily_c,
        usage_dlf=usage_dlf, export_dlf=export_dlf,
        import_tlf=import_tlf, export_tlf=export_tlf,
    )


def _tou_retail(usage_c=20.0, feedin_c=5.0, usage_other=0.0,
                feedin_other=0.0, daily_c=50.0):
    return RetailTariff(
        name='TestTOU', state='NSW', type='tou', daily_c_per_day=daily_c,
        usage_tariffs=(TariffPeriod(0, 24, usage_c),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, feedin_c),),
        usage_other_c_per_kwh=usage_other,
        feedin_other_c_per_kwh=feedin_other,
    )


def _wholesale_retail(usage_other=2.0, feedin_other=0.0, daily_c=80.0):
    return RetailTariff(
        name='TestWh', state='NSW', type='wholesale_exposed',
        daily_c_per_day=daily_c,
        usage_tariffs=(), usage_tariffs_weekend=None, feedin_tariffs=(),
        usage_other_c_per_kwh=usage_other,
        feedin_other_c_per_kwh=feedin_other,
    )


def _flat_prices(value_mwh, index):
    return pd.Series(np.full(len(index), value_mwh), index=index)


# ---------------- n_days ----------------

def test_n_days_full_non_leap_year():
    idx = pd.date_range('2022-01-01 00:05:00', '2023-01-01 00:00:00',
                        freq='5min', tz='Etc/GMT-10')
    assert _n_days(idx, 5/60) == 365


def test_n_days_single_day():
    # 288 intervals from 00:05 to 00:00 next day = one calendar day
    idx = pd.date_range('2022-01-03 00:05:00', periods=288, freq='5min',
                        tz='Etc/GMT-10')
    assert _n_days(idx, 5/60) == 1


# ---------------- Wholesale layer ----------------

def test_wholesale_usage_cost_simple():
    """10 kWh import at 100 $/MWh = 0.1 $/kWh -> $1.00, loss factor 1."""
    idx = _idx(2)
    traces = _traces([5.0, 5.0], [0.0, 0.0], idx)
    prices = _flat_prices(100.0, idx)
    bill = compute_bill(traces, prices, _flat_network(), _wholesale_retail(), state='NSW')
    # 10 kWh * 0.1 $/kWh * 1.0 = 1.0
    assert bill.wholesale_usage_cost == pytest.approx(1.0)


def test_wholesale_feedin_credit_negative():
    """Export at positive price -> negative credit (reduces bill)."""
    idx = _idx(2)
    traces = _traces([0.0, 0.0], [4.0, 6.0], idx)
    prices = _flat_prices(100.0, idx)
    bill = compute_bill(traces, prices, _flat_network(), _wholesale_retail(), state='NSW')
    # -(10 kWh * 0.1) = -1.0
    assert bill.wholesale_feedin_credit == pytest.approx(-1.0)


def test_wholesale_negative_price_import_is_credit():
    """Importing during negative price -> negative usage cost (customer paid)."""
    idx = _idx(1)
    traces = _traces([10.0], [0.0], idx)
    prices = _flat_prices(-50.0, idx)  # -0.05 $/kWh
    bill = compute_bill(traces, prices, _flat_network(), _wholesale_retail(), state='NSW')
    assert bill.wholesale_usage_cost == pytest.approx(-0.5)


def test_wholesale_negative_price_export_is_cost():
    """Exporting during negative price -> positive feedin value (customer pays)."""
    idx = _idx(1)
    traces = _traces([0.0], [10.0], idx)
    prices = _flat_prices(-50.0, idx)
    bill = compute_bill(traces, prices, _flat_network(), _wholesale_retail(), state='NSW')
    # -(10 * -0.05) = +0.5
    assert bill.wholesale_feedin_credit == pytest.approx(0.5)


def test_wholesale_loss_factors_applied():
    """Loss factors scale the wholesale layer."""
    idx = _idx(1)
    traces = _traces([10.0], [10.0], idx)
    prices = _flat_prices(100.0, idx)
    net = _flat_network(usage_dlf=1.05, export_dlf=0.95,
                        import_tlf=1.01, export_tlf=1.02)
    bill = compute_bill(traces, prices, net, _wholesale_retail(), state='NSW')
    # import: 10*0.1*(1.05*1.01) = 1.0*1.0605 = 1.0605
    assert bill.wholesale_usage_cost == pytest.approx(10 * 0.1 * 1.05 * 1.01)
    # export: -(10*0.1*(0.95*1.02)) = -(1.0*0.969)
    assert bill.wholesale_feedin_credit == pytest.approx(-(10 * 0.1 * 0.95 * 1.02))


# ---------------- Network layer ----------------

def test_network_usage_cost():
    idx = _idx(2)
    traces = _traces([3.0, 7.0], [0.0, 0.0], idx)
    prices = _flat_prices(0.0, idx)
    bill = compute_bill(traces, prices, _flat_network(usage_c=10.0),
                        _wholesale_retail(), state='NSW')
    # 10 kWh * 0.10 $/kWh * 1.0 = 1.0
    assert bill.network_usage_cost == pytest.approx(1.0)


def test_network_feedin_penalty_is_positive_cost():
    """A negative network feedin rate (penalty) -> positive stored value."""
    idx = _idx(1)
    traces = _traces([0.0], [10.0], idx)
    prices = _flat_prices(0.0, idx)
    net = _flat_network(feedin_c=-2.0)  # -2 c/kWh penalty
    bill = compute_bill(traces, prices, net, _wholesale_retail(), state='NSW')
    # -(10 * -0.02) = +0.2  (customer pays the penalty)
    assert bill.network_feedin_credit == pytest.approx(0.2)


def test_network_feedin_reward_is_negative_credit():
    idx = _idx(1)
    traces = _traces([0.0], [10.0], idx)
    prices = _flat_prices(0.0, idx)
    net = _flat_network(feedin_c=3.0)  # 3 c/kWh reward
    bill = compute_bill(traces, prices, net, _wholesale_retail(), state='NSW')
    # -(10 * 0.03) = -0.3
    assert bill.network_feedin_credit == pytest.approx(-0.3)


def test_network_daily_charges():
    idx = pd.date_range('2022-01-03 00:05:00', periods=288, freq='5min',
                        tz='Etc/GMT-10')  # one day
    traces = _traces(np.zeros(288), np.zeros(288), idx)
    prices = _flat_prices(0.0, idx)
    bill = compute_bill(traces, prices, _flat_network(daily_c=100.0),
                        _wholesale_retail(), state='NSW')
    # 100 c/day / 100 * 1 day = $1.00
    assert bill.network_daily_charges == pytest.approx(1.0)


# ---------------- Retail layer: TOU ----------------

def test_tou_usage_cost_with_adder():
    idx = _idx(2)
    traces = _traces([5.0, 5.0], [0.0, 0.0], idx)
    prices = _flat_prices(100.0, idx)  # ignored for household TOU bill
    retail = _tou_retail(usage_c=20.0, usage_other=2.0)
    bill = compute_bill(traces, prices, _flat_network(), retail, state='NSW')
    # 10 kWh * (20+2)/100 = 10 * 0.22 = 2.2
    assert bill.retail_usage_cost == pytest.approx(2.2)


def test_tou_feedin_credit_with_adder():
    idx = _idx(2)
    traces = _traces([0.0, 0.0], [4.0, 6.0], idx)
    prices = _flat_prices(100.0, idx)
    retail = _tou_retail(feedin_c=5.0, feedin_other=1.0)
    bill = compute_bill(traces, prices, _flat_network(), retail, state='NSW')
    # -(10 * (5+1)/100) = -(10*0.06) = -0.6
    assert bill.retail_feedin_credit == pytest.approx(-0.6)


def test_tou_household_total_is_retail_only():
    """For TOU, household pays only the retail layer."""
    idx = pd.date_range('2022-01-03 00:05:00', periods=288, freq='5min',
                        tz='Etc/GMT-10')
    imp = np.full(288, 0.1)   # 28.8 kWh
    exp = np.zeros(288)
    traces = _traces(imp, exp, idx)
    prices = _flat_prices(100.0, idx)
    retail = _tou_retail(usage_c=20.0, daily_c=50.0)
    bill = compute_bill(traces, prices, _flat_network(), retail, state='NSW')
    # retail usage: 28.8 * 0.20 = 5.76 ; daily: 50/100*1 = 0.50
    expected = 28.8 * 0.20 + 0.5
    assert bill.household_total == pytest.approx(expected)
    # household_total must NOT include wholesale/network for TOU
    assert bill.household_total == pytest.approx(
        bill.retail_usage_cost + bill.retail_feedin_credit
        + bill.retail_daily_charges
    )


# ---------------- Retail layer: wholesale-exposed ----------------

def test_wholesale_retail_usage_is_adders_only():
    idx = _idx(2)
    traces = _traces([5.0, 5.0], [0.0, 0.0], idx)
    prices = _flat_prices(100.0, idx)
    retail = _wholesale_retail(usage_other=2.0)
    bill = compute_bill(traces, prices, _flat_network(), retail, state='NSW')
    # retail layer is just the adder: 10 * 0.02 = 0.2
    assert bill.retail_usage_cost == pytest.approx(0.2)


def test_wholesale_household_total_is_full_stack():
    idx = pd.date_range('2022-01-03 00:05:00', periods=288, freq='5min',
                        tz='Etc/GMT-10')
    imp = np.full(288, 0.1)   # 28.8 kWh
    exp = np.zeros(288)
    traces = _traces(imp, exp, idx)
    prices = _flat_prices(100.0, idx)  # 0.1 $/kWh
    net = _flat_network(usage_c=10.0, daily_c=100.0)
    retail = _wholesale_retail(usage_other=2.0, daily_c=80.0)
    bill = compute_bill(traces, prices, net, retail, state='NSW')
    # wholesale usage: 28.8 * 0.1 = 2.88
    # network usage: 28.8 * 0.10 = 2.88 ; network daily: 1.0
    # retail usage (adder): 28.8 * 0.02 = 0.576 ; retail daily: 0.8
    expected = 2.88 + 2.88 + 1.0 + 0.576 + 0.8
    assert bill.household_total == pytest.approx(expected)


# ---------------- Retailer margin ----------------

def test_retailer_margin_tou():
    """For TOU: margin = retail revenue - wholesale - network costs."""
    idx = pd.date_range('2022-01-03 00:05:00', periods=288, freq='5min',
                        tz='Etc/GMT-10')
    imp = np.full(288, 0.1)
    exp = np.zeros(288)
    traces = _traces(imp, exp, idx)
    prices = _flat_prices(100.0, idx)
    net = _flat_network(usage_c=10.0, daily_c=100.0)
    retail = _tou_retail(usage_c=20.0, daily_c=50.0)
    bill = compute_bill(traces, prices, net, retail, state='NSW')
    wholesale_net = bill.wholesale_usage_cost + bill.wholesale_feedin_credit
    network_net = (bill.network_usage_cost + bill.network_feedin_credit
                   + bill.network_daily_charges)
    assert bill.retailer_margin == pytest.approx(
        bill.household_total - wholesale_net - network_net
    )


def test_retailer_margin_wholesale_is_roughly_adders_plus_subscription():
    """For wholesale-exposed: customer bears spot+network, so retailer margin
    is essentially the retail-layer take (adders + subscription)."""
    idx = pd.date_range('2022-01-03 00:05:00', periods=288, freq='5min',
                        tz='Etc/GMT-10')
    imp = np.full(288, 0.1)
    exp = np.zeros(288)
    traces = _traces(imp, exp, idx)
    prices = _flat_prices(100.0, idx)
    net = _flat_network(usage_c=10.0, daily_c=100.0)
    retail = _wholesale_retail(usage_other=2.0, daily_c=80.0)
    bill = compute_bill(traces, prices, net, retail, state='NSW')
    # margin should equal the retail layer net (since household bears the rest)
    retail_net = (bill.retail_usage_cost + bill.retail_feedin_credit
                  + bill.retail_daily_charges)
    assert bill.retailer_margin == pytest.approx(retail_net)


# ---------------- Weekday/weekend ----------------

def test_tou_weekday_weekend_distinction():
    """Weekend intervals use weekend rates."""
    # Saturday 2022-01-08
    sat = pd.date_range('2022-01-08 00:05:00', periods=288, freq='5min',
                        tz='Etc/GMT-10')
    imp = np.full(288, 0.1)
    traces = _traces(imp, np.zeros(288), sat)
    prices = _flat_prices(0.0, sat)
    retail = RetailTariff(
        name='T', state='NSW', type='tou', daily_c_per_day=0.0,
        usage_tariffs=(TariffPeriod(0, 24, 30.0),),          # weekday 30
        usage_tariffs_weekend=(TariffPeriod(0, 24, 10.0),),  # weekend 10
        feedin_tariffs=(TariffPeriod(0, 24, 0.0),),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
    )
    bill = compute_bill(traces, prices, _flat_network(), retail, state='NSW')
    # All Saturday -> weekend rate 10 c/kWh: 28.8 * 0.10 = 2.88
    assert bill.retail_usage_cost == pytest.approx(2.88)


# ---------------- Errors ----------------

def test_nan_price_raises():
    idx = _idx(3)
    traces = _traces([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], idx)
    prices = pd.Series([100.0, np.nan, 100.0], index=idx)
    with pytest.raises(ValueError, match='NaN'):
        compute_bill(traces, prices, _flat_network(), _wholesale_retail(), state='NSW')


def test_misaligned_index_raises():
    idx = _idx(3)
    traces = _traces([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], idx)
    prices = _flat_prices(100.0, _idx(2))
    with pytest.raises(ValueError, match='index'):
        compute_bill(traces, prices, _flat_network(), _wholesale_retail(), state='NSW')


# ---------------- bill_all_sites ----------------

def test_bill_all_sites():
    idx = _idx(2)
    t1 = _traces([5.0, 5.0], [0.0, 0.0], idx)
    t2 = SiteTraces(
        site_id='S0002',
        import_kwh=pd.Series([2.0, 2.0], index=idx),
        export_kwh=pd.Series([0.0, 0.0], index=idx),
        battery_soc_kwh=pd.Series([0.0, 0.0], index=idx),
        sol_to_home_kwh=pd.Series([0.0, 0.0], index=idx),
        sol_to_bat_kwh=pd.Series([0.0, 0.0], index=idx),
        sol_to_grid_kwh=pd.Series([0.0, 0.0], index=idx),
        bat_to_home_kwh=pd.Series([0.0, 0.0], index=idx),
        grid_to_home_kwh=pd.Series([0.0, 0.0], index=idx),
        grid_to_bat_kwh=pd.Series([0.0, 0.0], index=idx),
        bat_to_grid_kwh=pd.Series([0.0, 0.0], index=idx),
        curtail_kwh=pd.Series([0.0, 0.0], index=idx),
    )
    prices = _flat_prices(100.0, idx)
    bills = bill_all_sites({'S0001': t1, 'S0002': t2}, prices,
                           _flat_network(), _wholesale_retail(), state='NSW')
    assert set(bills.keys()) == {'S0001', 'S0002'}
    assert bills['S0001'].wholesale_usage_cost == pytest.approx(1.0)
    assert bills['S0002'].wholesale_usage_cost == pytest.approx(0.4)


def test_sitebill_is_frozen():
    idx = _idx(1)
    traces = _traces([1.0], [0.0], idx)
    prices = _flat_prices(100.0, idx)
    bill = compute_bill(traces, prices, _flat_network(), _wholesale_retail(), state='NSW')
    with pytest.raises(Exception):
        bill.total = 5.0

# ---------------- mixed plans (cap_account_spec.md Sections 16-18) ----------------

from model.core.prices import build_layer_rates, build_price_vectors


def _mixed_retail(usage_c=20.0, usage_other=0.0, feedin_other=0.0,
                  daily_c=110.0, share=0.9, usage_tariffs=None,
                  basis='wholesale', floor_c=0.0):
    """Mixed plan. Defaults are the Tesla Electric settings: share 0.9 of
    the loss-adjusted spot value, floored at 0 c/kWh."""
    return RetailTariff(
        name='TestMixed', state='NSW', type='mixed', daily_c_per_day=daily_c,
        usage_tariffs=usage_tariffs or (TariffPeriod(0, 24, usage_c),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(),
        usage_other_c_per_kwh=usage_other,
        feedin_other_c_per_kwh=feedin_other,
        wholesale_export_share=share,
        export_share_basis=basis,
        export_floor_c_per_kwh=floor_c,
    )


@pytest.mark.parametrize('price_mwh', [100.0, 0.0, -50.0, -1000.0, 5000.0])
@pytest.mark.parametrize('net_feedin_c', [0.0, 2.0, -3.0])
@pytest.mark.parametrize('basis', ['wholesale', 'wholesale_and_network'])
@pytest.mark.parametrize('floor_c', [0.0, 1.5, None])
def test_mixed_layer_rates_export_is_floored_share(price_mwh, net_feedin_c,
                                                   basis, floor_c):
    idx = _idx(4)
    net = _flat_network(feedin_c=net_feedin_c, export_dlf=0.97, export_tlf=0.99)
    lr = build_layer_rates(
        idx, _flat_prices(price_mwh, idx), net,
        _mixed_retail(feedin_other=0.5, basis=basis, floor_c=floor_c), 'NSW')
    lf = 0.97 * 0.99
    spot_value = lf * price_mwh / 1000.0
    net_value = lf * net_feedin_c / 100.0
    np.testing.assert_allclose(lr.wholesale_export, spot_value)
    np.testing.assert_allclose(lr.network_export, net_value)
    value = spot_value + (net_value if basis == 'wholesale_and_network' else 0.0)
    credit = 0.9 * value
    if floor_c is not None:
        credit = max(floor_c / 100.0, credit)
    np.testing.assert_allclose(
        lr.retail_export, credit + 0.005,
        err_msg=f"price={price_mwh} net={net_feedin_c}c basis={basis} "
                f"floor={floor_c}: V={value:.6f}, expected credit {credit:.6f}")


def test_mixed_export_floor_binds_only_below_it():
    """Per-interval floor on a mixed series: intervals whose share of value
    falls below the floor take the floor, the rest are untouched, and the
    wholesale and network layers are not floored."""
    idx = _idx(6)
    price = pd.Series([-1000.0, -30.0, 0.0, 10.0, 300.0, 15000.0], index=idx)
    lr = build_layer_rates(idx, price, _flat_network(feedin_c=-3.0),
                           _mixed_retail(), 'NSW')
    share = 0.9 * price.to_numpy() / 1000.0
    expected = np.maximum(0.0, share)
    # Non-vacuity: the series must contain intervals on both sides.
    assert (share < 0).sum() == 2 and (share > 0).sum() == 3
    # Floored intervals take the floor exactly, not merely approximately.
    assert (lr.retail_export[share <= 0] == 0.0).all(), lr.retail_export
    np.testing.assert_allclose(lr.retail_export, expected, rtol=1e-12)
    np.testing.assert_allclose(lr.wholesale_export, price.to_numpy() / 1000.0)
    np.testing.assert_allclose(lr.network_export, -0.03)


def test_mixed_without_floor_passes_negative_value_through():
    idx = _idx(2)
    lr = build_layer_rates(idx, _flat_prices(-500.0, idx),
                           _flat_network(feedin_c=-3.0),
                           _mixed_retail(basis='wholesale_and_network',
                                         floor_c=None), 'NSW')
    np.testing.assert_allclose(lr.retail_export, 0.9 * (-0.50 - 0.03))


def test_mixed_export_rate_rejects_other_types():
    from model.core.prices import mixed_export_rate
    with pytest.raises(ValueError, match="needs a 'mixed' retail tariff"):
        mixed_export_rate(np.zeros(2), np.zeros(2), _tou_retail())


def test_mixed_layer_rates_import_is_tou_plus_adder():
    idx = _idx(4)
    lr = build_layer_rates(idx, _flat_prices(5000.0, idx), _flat_network(),
                           _mixed_retail(usage_c=20.0, usage_other=1.0), 'NSW')
    np.testing.assert_allclose(lr.retail_import, 0.21)


def test_mixed_price_vectors_are_dispatch_prices():
    """The optimiser's import price is the billed TOU rate plus positive
    spot; its export price is the billed credit (model_spec.md, Mixed-plan
    dispatch prices). Detailed coverage is in test_prices.py."""
    idx = _idx(4)
    net = _flat_network(feedin_c=2.0)
    p_imp, p_exp = build_price_vectors(idx, _flat_prices(300.0, idx), net,
                                       _mixed_retail(usage_c=20.0), 'NSW')
    np.testing.assert_allclose(p_imp.to_numpy(), 0.20 + 0.30)
    np.testing.assert_allclose(p_exp.to_numpy(), 0.9 * 0.30)


def test_unknown_retail_type_raises():
    idx = _idx(4)
    bogus = RetailTariff(
        name='Bogus', state='NSW', type='bogus', daily_c_per_day=0.0,
        usage_tariffs=(), usage_tariffs_weekend=None, feedin_tariffs=(),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
    )
    with pytest.raises(ValueError, match="unknown retail tariff type 'bogus'"):
        build_layer_rates(idx, _flat_prices(100.0, idx), _flat_network(),
                          bogus, 'NSW')
    with pytest.raises(ValueError, match="unknown retail tariff type 'bogus'"):
        build_price_vectors(idx, _flat_prices(100.0, idx), _flat_network(),
                            bogus, 'NSW')


def test_mixed_bill_hand_built():
    """Two intervals in one day: 1 kWh import, then 2 kWh export.

    Price 1000 $/MWh = $1.00/kWh. Network usage 10 c, feedin -2 c, daily
    100 c. Loss factors 1. Plan: TOU 20 c, adder 1 c, daily 110 c, share 0.9
    of spot, floor 0.
      V                = 2 * (1.00 - 0.02)       = 1.96
      credit           = 2 * 0.9 * 1.00          = 1.80
      household_total  = 1 * 0.21 + 1.10 - 1.80  = -0.49
      wholesale_net    = 1.00 - 2.00             = -1.00
      network_net      = 0.10 + 0.04 + 1.00      =  1.14
      retailer_margin  = -0.49 + 1.00 - 1.14     = -0.63
    which is TOU revenue + plan daily - import cost + (V - credit)
    - network daily = 0.21 + 1.10 - 1.10 + 0.16 - 1.00 = -0.63.
    """
    idx = _idx(2)
    traces = _traces([1.0, 0.0], [0.0, 2.0], idx)
    net = _flat_network(usage_c=10.0, feedin_c=-2.0, daily_c=100.0)
    bill = compute_bill(traces, _flat_prices(1000.0, idx), net,
                        _mixed_retail(usage_c=20.0, usage_other=1.0), 'NSW')
    assert bill.plan_type == 'mixed'
    assert bill.retail_usage_cost == pytest.approx(0.21)
    assert bill.retail_feedin_credit == pytest.approx(-1.80)
    assert bill.retail_daily_charges == pytest.approx(1.10)
    assert bill.household_total == pytest.approx(-0.49)
    assert bill.retailer_margin == pytest.approx(-0.63)


def test_mixed_bill_negative_spot_export_costs_household_nothing():
    """As test_mixed_bill_hand_built at -500 $/MWh. The floor zeroes the
    household's export credit; the retailer pays the negative export value
    and keeps the network export charge.
      household_total  = 0.21 + 1.10                       =  1.31
      wholesale_net    = 1 * -0.50 - 2 * -0.50             =  0.50
      network_net      = 0.10 + 0.04 + 1.00                =  1.14
      retailer_margin  = 1.31 - 0.50 - 1.14                = -0.33
    """
    idx = _idx(2)
    traces = _traces([1.0, 0.0], [0.0, 2.0], idx)
    net = _flat_network(usage_c=10.0, feedin_c=-2.0, daily_c=100.0)
    bill = compute_bill(traces, _flat_prices(-500.0, idx), net,
                        _mixed_retail(usage_c=20.0, usage_other=1.0), 'NSW')
    assert bill.retail_feedin_credit == 0.0
    assert bill.wholesale_feedin_credit == pytest.approx(1.00)
    assert bill.household_total == pytest.approx(1.31)
    assert bill.retailer_margin == pytest.approx(-0.33)


def test_mixed_net_position_identity_full_leap_year():
    """cap_account_spec.md Section 17: over a full calendar year the four
    components reproduce the net position computed from the two bills.

    The bills come from compute_bill. The components are computed here
    directly from the Section 17 definitions, with period lookups done by
    hand rather than through build_layer_rates, so the test is independent
    of the rate construction it checks. QLD is used so that local time is
    AEST and a period lookup is the hour of the interval start.

    2024 is a leap year: the annual bill counts 366 days, and every fixed
    term here is taken at that count.
    """
    state = 'QLD'
    idx = pd.date_range('2024-01-01 00:05:00', '2025-01-01 00:00:00',
                        freq='5min', tz='Etc/GMT-10')
    n = len(idx)
    assert n == 366 * 288
    rng = np.random.default_rng(20260923)

    # Prices: mostly ordinary, some negative, some spikes well above TOU.
    price_mwh = rng.normal(90.0, 60.0, n)
    spikes = rng.random(n) < 0.01
    price_mwh[spikes] = rng.uniform(500.0, 15000.0, spikes.sum())
    negs = rng.random(n) < 0.05
    price_mwh[negs] = rng.uniform(-1000.0, -1.0, negs.sum())
    prices = pd.Series(price_mwh, index=idx)

    # Flows: each interval imports or exports, never both.
    amount = rng.exponential(0.3, n)
    importing = rng.random(n) < 0.55
    imp = np.where(importing, amount, 0.0)
    exp = np.where(importing, 0.0, amount)
    traces = _traces(imp, exp, idx)

    net_usage_periods = ((0, 7, 8.0), (7, 16, 12.0), (16, 21, 30.0), (21, 24, 8.0))
    net_feedin_periods = ((0, 10, 0.0), (10, 16, -1.5), (16, 21, 2.0), (21, 24, 0.0))
    tou_periods = ((0, 15, 24.0), (15, 21, 50.0), (21, 24, 24.0))
    net = NetworkTariff(
        name='TestNet', state=state,
        usage_tariffs=tuple(TariffPeriod(*p) for p in net_usage_periods),
        usage_tariffs_weekend=None,
        feedin_tariffs=tuple(TariffPeriod(*p) for p in net_feedin_periods),
        daily_c_per_day=80.0,
        usage_dlf=1.04, export_dlf=0.98, import_tlf=1.01, export_tlf=0.99,
    )
    amber = _wholesale_retail(usage_other=4.081, daily_c=82.1918)
    plan = _mixed_retail(usage_other=0.821, daily_c=110.0, share=0.9,
                         usage_tariffs=tuple(TariffPeriod(*p) for p in tou_periods))

    bill_c = compute_bill(traces, prices, net, plan, state)
    bill_a = compute_bill(traces, prices, net, amber, state)

    # --- Components from the definitions ---
    hour = (idx - pd.Timedelta(minutes=5)).hour.to_numpy()

    def by_hour(periods):
        out = np.full(n, np.nan)
        for start, end, c in periods:
            out[(hour >= start) & (hour < end)] = c
        assert not np.isnan(out).any()
        return out / 100.0

    lf_i, lf_e = 1.04 * 1.01, 0.98 * 0.99
    price = price_mwh / 1000.0
    r = by_hour(tou_periods) + 0.00821
    c = lf_i * (price + by_hour(net_usage_periods)) + 0.04081
    v = lf_e * (price + by_hour(net_feedin_periods)) * exp
    credit_c = np.maximum(0.0, 0.9 * lf_e * price) * exp

    days = 366
    net_fixed = (110.0 - 80.0) / 100.0 * days
    import_margin = float(np.sum(np.clip(r - c, 0.0, None) * imp))
    cost_absorbed = float(np.sum(np.clip(c - r, 0.0, None) * imp))
    credit_retained = float(np.sum(v) - np.sum(credit_c))

    # The data must exercise every branch the identity covers.
    assert import_margin > 0 and cost_absorbed > 0
    assert (v < 0).any() and (v > 0).any()
    assert ((price < 0) & (exp > 0)).any()          # the floor binds
    assert ((price > 0) & (exp > 0)).any()          # the share applies

    decomposed = net_fixed + import_margin - cost_absorbed + credit_retained
    from_bills = bill_c.household_total - (bill_a.household_total
                                           - bill_a.retail_daily_charges)
    assert bill_a.retail_daily_charges == pytest.approx(0.821918 * days)
    assert decomposed == pytest.approx(from_bills, rel=1e-9, abs=1e-6)


# ---------------- GST on wholesale imports (model_spec.md, The settlement model) ----------------

import dataclasses

from model.components.billing import GST_RATE, compute_bill_and_monthly_bills


def _gst_case(n=6):
    """Spot prices spanning negative, zero, ordinary and extreme values, with
    imports and exports in every interval and non-unit loss factors, so a GST
    term applied to the wrong flow, the wrong layer or only to positive
    prices changes the answer."""
    idx = _idx(n)
    prices = pd.Series([100.0, -50.0, 0.0, 15000.0, -1000.0, 37.5][:n],
                       index=idx)
    imp = np.array([1.0, 2.0, 0.5, 0.3, 1.5, 0.8][:n])
    exp = np.array([0.4, 0.6, 1.0, 2.0, 0.2, 0.1][:n])
    net = _flat_network(usage_c=10.0, feedin_c=-1.0, usage_dlf=1.04,
                        import_tlf=1.01, export_dlf=0.97, export_tlf=0.99)
    return idx, prices, _traces(imp, exp, idx), net


def test_gst_on_wholesale_import_adds_gst_on_spot_imports_only():
    idx, prices, traces, net = _gst_case()
    plain = _wholesale_retail()
    gst = dataclasses.replace(plain, gst_on_wholesale_import=True)
    b0 = compute_bill(traces, prices, net, plain, state='NSW')
    b1 = compute_bill(traces, prices, net, gst, state='NSW')

    import_lf = 1.04 * 1.01
    spot_import = float(np.sum(traces.import_kwh.to_numpy()
                               * prices.to_numpy() / 1000.0 * import_lf))
    expected_gst = GST_RATE * spot_import
    # Non-vacuity: the negative-price intervals move the GST by more than
    # the tolerance, so GST on positive prices only would fail.
    neg = prices.to_numpy() < 0
    neg_part = GST_RATE * float(np.sum(traces.import_kwh.to_numpy()[neg]
                                       * prices.to_numpy()[neg] / 1000.0
                                       * import_lf))
    assert abs(neg_part) > 0.1
    assert b1.wholesale_usage_cost == pytest.approx(spot_import)

    ctx = (f"b0={b0}\nb1={b1}\nexpected GST {expected_gst:.6f} "
           f"(negative-price part {neg_part:.6f})")
    assert b1.retail_usage_cost - b0.retail_usage_cost == pytest.approx(
        expected_gst, abs=1e-9), ctx
    assert b1.household_total - b0.household_total == pytest.approx(
        expected_gst, abs=1e-9), ctx
    assert b1.retailer_margin - b0.retailer_margin == pytest.approx(
        expected_gst, abs=1e-9), ctx
    for field in ('wholesale_usage_cost', 'wholesale_feedin_credit',
                  'network_usage_cost', 'network_feedin_credit',
                  'network_daily_charges', 'retail_feedin_credit',
                  'retail_daily_charges', 'retail_other_credits'):
        assert getattr(b1, field) == getattr(b0, field), (field, ctx)


def test_gst_on_wholesale_import_monthly_bills_carry_their_own_gst():
    # Two local months: the last day of January and the first of February.
    idx = pd.date_range('2022-01-31 00:05:00', periods=576, freq='5min',
                        tz='Etc/GMT-10')
    rng = np.random.default_rng(7)
    prices = pd.Series(rng.uniform(-200.0, 600.0, len(idx)), index=idx)
    traces = _traces(rng.uniform(0.0, 0.3, len(idx)),
                     rng.uniform(0.0, 0.3, len(idx)), idx)
    net = _flat_network(usage_dlf=1.04, export_dlf=0.97)
    plain = _wholesale_retail()
    gst = dataclasses.replace(plain, gst_on_wholesale_import=True)
    a0, m0 = compute_bill_and_monthly_bills(traces, prices, net, plain, 'NSW',
                                            5 / 60)
    a1, m1 = compute_bill_and_monthly_bills(traces, prices, net, gst, 'NSW',
                                            5 / 60)
    assert len(m1) == 2
    for m in m1:
        d = m1[m].household_total - m0[m].household_total
        assert d == pytest.approx(GST_RATE * m1[m].wholesale_usage_cost,
                                  abs=1e-9), m
        assert d != pytest.approx(0.0, abs=1e-3), m
    # Monthly and annual day counts differ under DST (_reduce_settlement),
    # so compare the GST deltas rather than the totals.
    assert sum(m1[m].household_total - m0[m].household_total
               for m in m1) == pytest.approx(
        a1.household_total - a0.household_total, abs=1e-9)
    assert a1.household_total - a0.household_total == pytest.approx(
        GST_RATE * a1.wholesale_usage_cost, abs=1e-9)


def test_gst_on_wholesale_import_leaves_dispatch_prices_unchanged():
    idx, prices, _, net = _gst_case()
    plain = _wholesale_retail()
    gst = dataclasses.replace(plain, gst_on_wholesale_import=True)
    p0 = build_price_vectors(idx, prices, net, plain, 'NSW')
    p1 = build_price_vectors(idx, prices, net, gst, 'NSW')
    for a, b in zip(p0, p1):
        np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())
