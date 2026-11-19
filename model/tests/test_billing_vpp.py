"""Unit tests for model.components.billing's VppTraces / vpp_traces handling.

Ported from test_perfect_foresight_vpp.py's billing integration section as
part of removing perfect_foresight_vpp.py (superseded by vpp_sc_event.py).
These tests build SiteTraces/VppTraces by hand and exercise compute_bill
directly -- none of them call perfect_foresight_vpp or any other dispatch
component -- so they port verbatim with no adaptation. They test the shared
billing contract (VppConfig-driven reward handling in retail_other_credits)
that applies identically regardless of which component produced the traces.
"""
import numpy as np
import pandas as pd
import pytest

from model.core.site_result import SiteTraces, VppTraces
from model.core.tariff import NetworkTariff, RetailTariff, TariffPeriod, VppConfig
from model.components.billing import compute_bill


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def _idx(n: int) -> pd.DatetimeIndex:
    """n-interval index at 1-hour cadence (kW = kWh/interval)."""
    return pd.date_range('2022-01-03 01:00:00', periods=n, freq='60min',
                         tz='Etc/GMT-10')


def _vpp_cfg(
    exp_reward: float = 10.0,
    imp_reward: float = 10.0,
    kwh_limit: float = 250.0,
    limit_applies_to: str = 'export_only',
    default_mode: str = 'self_consumption',
    event_window_hours: float = 1.0,
    threshold: float = 100.0,
) -> VppConfig:
    return VppConfig(
        exp_reward_c_per_kwh=exp_reward,
        imp_reward_c_per_kwh=imp_reward,
        kwh_limit=kwh_limit,
        limit_applies_to=limit_applies_to,
        default_mode=default_mode,
        event_window_hours=event_window_hours,
        vpp_threshold_dollars_per_mwh=threshold,
    )


def _retail_tariff_vpp(vpp_cfg: VppConfig) -> RetailTariff:
    """Minimal TOU RetailTariff with VPP config attached."""
    return RetailTariff(
        name='Test VPP Plan', state='NSW', type='tou',
        daily_c_per_day=100.0,
        usage_tariffs=(TariffPeriod(0, 24, 30.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 5.0),),
        usage_other_c_per_kwh=0.0,
        feedin_other_c_per_kwh=0.0,
        vpp=vpp_cfg,
    )


def _flat_network_tariff() -> NetworkTariff:
    """Minimal NetworkTariff with zero rates, isolating the retail layer."""
    return NetworkTariff(
        name='Test Net', state='NSW',
        usage_tariffs=(TariffPeriod(0, 24, 0.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 0.0),),
        daily_c_per_day=0.0,
        usage_dlf=1.0, export_dlf=1.0,
        import_tlf=1.0, export_tlf=1.0,
    )


# ---------------------------------------------------------------------------
# Billing integration
# ---------------------------------------------------------------------------

def test_billing_vpp_reward_in_retail_other_credits():
    """compute_bill with vpp_traces folds forced-flow rewards into retail_other_credits.

    f_exp=2 kWh, exp_reward=100 c/kWh = 1.00 $/kWh -> reward = -2.00 dollars.
    f_imp=0. retail_other_credits should equal -2.00 (credit reduces the bill).
    """
    n = 2
    idx = _idx(n)
    z = pd.Series(np.zeros(n), index=idx)

    # Build a minimal SiteTraces (billing only uses import_kwh and export_kwh)
    import_kwh = pd.Series([1., 0.], index=idx)
    export_kwh = pd.Series([0., 2.], index=idx)

    traces = SiteTraces(
        site_id='S', import_kwh=import_kwh, export_kwh=export_kwh,
        battery_soc_kwh=z, sol_to_home_kwh=z, sol_to_bat_kwh=z,
        sol_to_grid_kwh=z, bat_to_home_kwh=z, grid_to_home_kwh=import_kwh,
        grid_to_bat_kwh=z, bat_to_grid_kwh=export_kwh, curtail_kwh=z,
    )
    vpp_t = VppTraces(
        site_id='S',
        n_imp_kwh=import_kwh,
        f_imp_kwh=z,
        n_exp_kwh=z,
        f_exp_kwh=export_kwh,   # all export is forced
    )

    # VppConfig: exp_reward=100 c/kWh, imp_reward=100 c/kWh
    cfg = _vpp_cfg(exp_reward=100.0, imp_reward=100.0)
    rt  = _retail_tariff_vpp(cfg)

    # Dummy prices (flat, no wholesale layer for a TOU plan)
    prices = pd.Series(np.full(n, 50.0), index=idx)
    net = _flat_network_tariff()

    bill = compute_bill(traces, prices, net, rt, state='NSW',
                        interval_hours=1.0, vpp_traces=vpp_t)

    # f_exp=2 kWh @ 100 c/kWh = 2.00 dollars reward -> -2.00
    np.testing.assert_allclose(bill.retail_other_credits, -2.0, atol=1e-6)


def test_billing_vpp_reward_additive_with_zero_imp():
    """f_imp reward adds to f_exp reward correctly.

    f_exp=1 kWh, f_imp=1 kWh, both at 100 c/kWh = 1.00 $/kWh.
    Expected retail_other_credits = -(1*1 + 1*1) = -2.00 dollars.
    """
    n = 2
    idx = _idx(n)
    z   = pd.Series(np.zeros(n), index=idx)

    import_kwh = pd.Series([1., 0.], index=idx)
    export_kwh = pd.Series([0., 1.], index=idx)
    f_imp_kwh  = pd.Series([1., 0.], index=idx)
    f_exp_kwh  = pd.Series([0., 1.], index=idx)

    traces = SiteTraces(
        site_id='S', import_kwh=import_kwh, export_kwh=export_kwh,
        battery_soc_kwh=z, sol_to_home_kwh=z, sol_to_bat_kwh=z,
        sol_to_grid_kwh=z, bat_to_home_kwh=z, grid_to_home_kwh=import_kwh,
        grid_to_bat_kwh=z, bat_to_grid_kwh=export_kwh, curtail_kwh=z,
    )
    vpp_t = VppTraces(site_id='S', n_imp_kwh=z, f_imp_kwh=f_imp_kwh,
                      n_exp_kwh=z, f_exp_kwh=f_exp_kwh)

    cfg = _vpp_cfg(exp_reward=100.0, imp_reward=100.0,
                   limit_applies_to='import_and_export')
    rt  = _retail_tariff_vpp(cfg)
    prices = pd.Series(np.full(n, 50.0), index=idx)
    net = _flat_network_tariff()

    bill = compute_bill(traces, prices, net, rt, state='NSW',
                        interval_hours=1.0, vpp_traces=vpp_t)
    np.testing.assert_allclose(bill.retail_other_credits, -2.0, atol=1e-6)


def test_billing_no_vpp_traces_unchanged():
    """compute_bill without vpp_traces produces retail_other_credits=0."""
    n = 2
    idx = _idx(n)
    z   = pd.Series(np.zeros(n), index=idx)
    import_kwh = pd.Series([1., 1.], index=idx)

    traces = SiteTraces(
        site_id='S', import_kwh=import_kwh, export_kwh=z,
        battery_soc_kwh=z, sol_to_home_kwh=z, sol_to_bat_kwh=z,
        sol_to_grid_kwh=z, bat_to_home_kwh=z, grid_to_home_kwh=import_kwh,
        grid_to_bat_kwh=z, bat_to_grid_kwh=z, curtail_kwh=z,
    )
    # Plain TOU plan (no vpp)
    rt = RetailTariff(
        name='Plain', state='NSW', type='tou', daily_c_per_day=0.0,
        usage_tariffs=(TariffPeriod(0, 24, 30.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 5.0),),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
    )
    prices = pd.Series(np.full(n, 50.0), index=idx)
    net = _flat_network_tariff()
    bill = compute_bill(traces, prices, net, rt, state='NSW', interval_hours=1.0)
    np.testing.assert_allclose(bill.retail_other_credits, 0., atol=1e-6)


def test_billing_vpp_traces_without_vpp_config_raises():
    """Supplying vpp_traces when the retail tariff has no VppConfig raises."""
    n = 2
    idx = _idx(n)
    z   = pd.Series(np.zeros(n), index=idx)
    import_kwh = pd.Series([1., 0.], index=idx)

    traces = SiteTraces(
        site_id='S', import_kwh=import_kwh, export_kwh=z,
        battery_soc_kwh=z, sol_to_home_kwh=z, sol_to_bat_kwh=z,
        sol_to_grid_kwh=z, bat_to_home_kwh=z, grid_to_home_kwh=import_kwh,
        grid_to_bat_kwh=z, bat_to_grid_kwh=z, curtail_kwh=z,
    )
    vpp_t = VppTraces(site_id='S', n_imp_kwh=import_kwh, f_imp_kwh=z,
                      n_exp_kwh=z, f_exp_kwh=z)
    rt = RetailTariff(
        name='Plain', state='NSW', type='tou', daily_c_per_day=0.0,
        usage_tariffs=(TariffPeriod(0, 24, 30.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 5.0),),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
    )
    prices = pd.Series(np.full(n, 50.0), index=idx)
    net = _flat_network_tariff()
    with pytest.raises(ValueError, match='VppConfig'):
        compute_bill(traces, prices, net, rt, state='NSW',
                     interval_hours=1.0, vpp_traces=vpp_t)