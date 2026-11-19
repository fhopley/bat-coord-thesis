"""Unit tests for min_export_price_per_kwh -- minimum forced export price.

Covers three areas:

  1. Budget allocation algorithm (_compute_min_export_price):
     - Known-answer test with uniform prices and zero load.
     - Budget larger than total available throughput returns 0.0.
     - Non-zero average load reduces effective export and raises the threshold.

  2. LP bound enforcement in vpp_sc_event:
     - f_exp is zero at sub-threshold intervals and non-zero only at or above
       the threshold.
     - Setting min_export_price_per_kwh=0.0 (default) is unrestricted.

  3. YAML parsing via VppConfig:
     - Field is None when key is absent.
     - Correct value parsed when present.
     - Negative value rejected.
"""
import numpy as np
import pandas as pd
import pytest

from model.core.scenario import BatterySpec
from model.components.vpp_sc_event import vpp_sc_event
from model.orchestrate import _compute_min_export_price


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

INTERVAL_HOURS = 5 / 60


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range('2022-01-01 00:05', periods=n, freq='5min',
                         tz='Etc/GMT-10')


def _bat(capacity_kwh=10.0, power_kw=5.0, backup_reserve_pct=0.0) -> BatterySpec:
    return BatterySpec(
        capacity_kwh=capacity_kwh,
        backup_reserve_pct=backup_reserve_pct,
        power_kw=power_kw,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
        c_deg_per_kwh=0.0,
    )


def _s(arr, idx) -> pd.Series:
    return pd.Series(arr, index=idx)


def _run_vpp(solar, load, op_imp, op_exp, event, battery=None,
             min_export_price_per_kwh=0.0, soc_init=None, kwh_limit=1000.0):
    """Convenience wrapper around vpp_sc_event."""
    n = len(solar)
    idx = _idx(n)
    bat = battery or _bat()
    soc_init_kwh = soc_init if soc_init is not None else (
        bat.backup_reserve_pct / 100.0 * bat.capacity_kwh)
    return vpp_sc_event(
        site_id='TEST',
        solar_kwh=_s(solar, idx),
        load_kwh=_s(load, idx),
        op_price_imp=_s(op_imp, idx),
        op_price_exp=_s(op_exp, idx),
        vpp_event=np.array(event, dtype=bool),
        battery_spec=bat,
        soc_init_kwh=soc_init_kwh,
        vpp_exp_reward=0.0,    # no reward -- isolates price threshold effect
        vpp_imp_reward=0.0,
        vpp_kwh_limit=kwh_limit,
        vpp_limit_applies_to='export_only',
        interval_hours=INTERVAL_HOURS,
        min_export_price_per_kwh=min_export_price_per_kwh,
    )


# ---------------------------------------------------------------------------
# Section 1: Budget allocation algorithm
# ---------------------------------------------------------------------------

def test_compute_min_export_price_known_answer():
    """Known-answer test: zero load, uniform power, linearly decreasing prices.

    Setup: T=12 intervals, power_kw=5, interval_hours=5/60, kwh_limit=1.0.
    Each interval contributes 5 * 5/60 = 0.4167 kWh.
    Budget exhausted after ceil(1.0 / 0.4167) = 3 intervals.
    Prices: [12, 11, 10, 9, ..., 1] $/MWh descending.
    Top 3 intervals have prices [12, 11, 10], cumulative = [0.417, 0.833, 1.25].
    Budget exhausted at interval 3 (price=10 $/MWh).
    """
    T = 12
    idx = _idx(T)
    prices_mwh = pd.Series(np.arange(T, 0, -1, dtype=float), index=idx)
    # Zero load -- effective export = full power
    site_traces = {'SITE': {'gross_load_kwh': pd.Series(np.zeros(T), index=idx)}}
    power_kw = 5.0
    kwh_limit = 1.0

    result = _compute_min_export_price(
        prices_mwh=prices_mwh,
        site_traces=site_traces,
        site_ids=['SITE'],
        power_kw=power_kw,
        interval_hours=INTERVAL_HOURS,
        kwh_limit=kwh_limit,
    )

    # Contribution per interval = 5 * 5/60 = 0.4167 kWh
    # Cumulative after 3 intervals = 3 * 0.4167 = 1.25 >= 1.0
    # Marginal interval is the 3rd highest price = 10 $/MWh
    assert result == pytest.approx(10.0, abs=1e-6), (
        f'Expected marginal price 10.0 $/MWh, got {result:.4f}'
    )


def test_compute_min_export_price_budget_never_exhausted():
    """Returns 0.0 when total available throughput is less than kwh_limit."""
    T = 6
    idx = _idx(T)
    prices_mwh = pd.Series(np.ones(T) * 100.0, index=idx)
    site_traces = {'SITE': {'gross_load_kwh': pd.Series(np.zeros(T), index=idx)}}

    # kwh_limit = 1000 kWh, total available = 6 * 5 * 5/60 = 2.5 kWh
    result = _compute_min_export_price(
        prices_mwh=prices_mwh,
        site_traces=site_traces,
        site_ids=['SITE'],
        power_kw=5.0,
        interval_hours=INTERVAL_HOURS,
        kwh_limit=1000.0,
    )
    assert result == 0.0, f'Expected 0.0 (no restriction), got {result}'


def test_compute_min_export_price_load_reduces_effective_export():
    """Average load reduces effective export, raising the marginal price.

    Same setup as known_answer but with avg_load_kw = 2.5 kW.
    Effective export = 5.0 - 2.5 = 2.5 kW per interval.
    Contribution per interval = 2.5 * 5/60 = 0.2083 kWh.
    Budget (1.0 kWh) exhausted after ceil(1.0 / 0.2083) = 5 intervals.
    Prices [12..1] descending: 5th highest = 8 $/MWh.
    """
    T = 12
    idx = _idx(T)
    prices_mwh = pd.Series(np.arange(T, 0, -1, dtype=float), index=idx)
    # Load of 2.5 kW * interval_hours = 2.5 * 5/60 = 0.2083 kWh per interval
    load_kwh = pd.Series(np.full(T, 2.5 * INTERVAL_HOURS), index=idx)
    site_traces = {'SITE': {'gross_load_kwh': load_kwh}}

    result = _compute_min_export_price(
        prices_mwh=prices_mwh,
        site_traces=site_traces,
        site_ids=['SITE'],
        power_kw=5.0,
        interval_hours=INTERVAL_HOURS,
        kwh_limit=1.0,
    )

    assert result == pytest.approx(8.0, abs=1e-6), (
        f'Expected marginal price 8.0 $/MWh with load, got {result:.4f}'
    )


# ---------------------------------------------------------------------------
# Section 2: LP bound enforcement
# ---------------------------------------------------------------------------

def test_min_export_price_restricts_f_exp_at_low_price_intervals():
    """f_exp is zero at intervals where op_exp < min_export_price_per_kwh.

    Setup: 8-interval event window, all intervals have solar=0 and moderate
    load. Battery starts full. op_exp alternates high/low so some intervals
    would naturally attract forced export and some would not.
    min_export_price_per_kwh = 0.20 $/kWh (= 200 $/MWh).
    Low-price intervals have op_exp = 0.05 $/kWh -- below threshold.
    High-price intervals have op_exp = 0.50 $/kWh -- above threshold.

    Verify: f_exp is zero at all low-price intervals.
    """
    n = 8
    solar  = np.zeros(n)
    load   = np.full(n, 0.05)   # small load so battery can export
    op_imp = np.full(n, 0.70)   # must exceed op_exp at all intervals
    # Alternate: even intervals low price, odd intervals high price.
    # All values below op_imp=0.70 to satisfy price ordering check.
    op_exp = np.array([0.05, 0.50, 0.05, 0.50, 0.05, 0.50, 0.05, 0.50])
    event  = np.ones(n, dtype=bool)

    bat = _bat(capacity_kwh=5.0, power_kw=5.0, backup_reserve_pct=0.0)
    traces, vt = _run_vpp(
        solar, load, op_imp, op_exp, event,
        battery=bat,
        soc_init=bat.capacity_kwh,       # full battery
        min_export_price_per_kwh=0.20,   # 200 $/MWh threshold
    )

    f_exp = vt.f_exp_kwh.to_numpy()
    low_price_mask = op_exp < 0.20       # even indices
    assert (f_exp[low_price_mask] < 1e-9).all(), (
        f'f_exp non-zero at sub-threshold intervals: {f_exp[low_price_mask]}'
    )


def test_min_export_price_zero_is_unrestricted():
    """Default min_export_price_per_kwh=0.0 places no restriction on f_exp."""
    n = 8
    solar  = np.zeros(n)
    load   = np.full(n, 0.05)
    op_imp = np.full(n, 0.30)
    op_exp = np.full(n, 0.10)   # all positive, all should be exportable
    event  = np.ones(n, dtype=bool)

    bat = _bat(capacity_kwh=5.0, power_kw=5.0, backup_reserve_pct=0.0)
    traces, vt = _run_vpp(
        solar, load, op_imp, op_exp, event,
        battery=bat,
        soc_init=bat.capacity_kwh,
        min_export_price_per_kwh=0.0,   # no restriction
    )

    # With a full battery, positive op_exp, and near-zero load, the LP should
    # discharge to the grid (f_exp > 0 somewhere).
    f_exp_total = float(vt.f_exp_kwh.sum())
    assert f_exp_total > 1e-6, (
        f'Expected some forced export with no price restriction, got {f_exp_total}'
    )


def test_min_export_price_above_all_op_exp_prevents_all_f_exp():
    """When threshold exceeds all op_exp values, f_exp is zero everywhere."""
    n = 8
    solar  = np.zeros(n)
    load   = np.full(n, 0.05)
    op_imp = np.full(n, 0.30)
    op_exp = np.full(n, 0.10)   # max op_exp = 0.10 $/kWh = 100 $/MWh
    event  = np.ones(n, dtype=bool)

    bat = _bat(capacity_kwh=5.0, power_kw=5.0, backup_reserve_pct=0.0)
    traces, vt = _run_vpp(
        solar, load, op_imp, op_exp, event,
        battery=bat,
        soc_init=bat.capacity_kwh,
        min_export_price_per_kwh=0.50,  # 500 $/MWh -- above all op_exp values
    )

    f_exp = vt.f_exp_kwh.to_numpy()
    assert (f_exp < 1e-9).all(), (
        f'Expected all f_exp=0 when threshold exceeds all prices, '
        f'got max={f_exp.max():.6f}'
    )


# ---------------------------------------------------------------------------
# Section 3: YAML parsing via VppConfig
# ---------------------------------------------------------------------------

def test_vpp_config_min_export_price_defaults_none():
    """min_export_price_dollars_per_mwh defaults to None when absent from YAML."""
    import yaml
    from model.core.tariff import load_retail_tariff

    base = {
        'tariff_class': 'retail', 'type': 'tou', 'name': 'Test',
        'state': 'NSW',
        'usage_tariffs': [[0, 24, 30.0]],
        'feedin_tariffs': [[0, 24, 5.0]],
        'usage_other_c_per_kwh': {},
        'daily_charges_c_per_day': {'supply': 100.0},
        'vpp': {
            'exp_reward_c_per_kwh': 0.0001, 'imp_reward_c_per_kwh': 0.0001,
            'kwh_limit': 200.0, 'limit_applies_to': 'export_only',
            'default_mode': 'tou', 'event_window_hours': 2.0,
            'vpp_threshold_dollars_per_mwh': 1000.0,
            # min_export_price_dollars_per_mwh absent
        },
    }
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as tmp:
        p = pathlib.Path(tmp) / 'test_vpp.yaml'
        p.write_text(yaml.dump(base))
        t = load_retail_tariff('test_vpp', pathlib.Path(tmp))
    assert t.vpp.min_export_price_dollars_per_mwh is None


def test_vpp_config_min_export_price_parsed_correctly():
    """min_export_price_dollars_per_mwh is parsed as a float when present."""
    import yaml
    from model.core.tariff import load_retail_tariff

    base = {
        'tariff_class': 'retail', 'type': 'tou', 'name': 'Test',
        'state': 'NSW',
        'usage_tariffs': [[0, 24, 30.0]],
        'feedin_tariffs': [[0, 24, 5.0]],
        'usage_other_c_per_kwh': {},
        'daily_charges_c_per_day': {'supply': 100.0},
        'vpp': {
            'exp_reward_c_per_kwh': 0.0001, 'imp_reward_c_per_kwh': 0.0001,
            'kwh_limit': 200.0, 'limit_applies_to': 'export_only',
            'default_mode': 'tou', 'event_window_hours': 2.0,
            'vpp_threshold_dollars_per_mwh': 1000.0,
            'min_export_price_dollars_per_mwh': 750.0,
        },
    }
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as tmp:
        p = pathlib.Path(tmp) / 'test_vpp.yaml'
        p.write_text(yaml.dump(base))
        t = load_retail_tariff('test_vpp', pathlib.Path(tmp))
    assert t.vpp.min_export_price_dollars_per_mwh == pytest.approx(750.0)


def test_vpp_config_min_export_price_negative_rejected():
    """Negative min_export_price_dollars_per_mwh is rejected with ValueError."""
    from model.core.tariff import VppConfig
    with pytest.raises(ValueError, match='min_export_price_dollars_per_mwh'):
        VppConfig(
            exp_reward_c_per_kwh=0.0001, imp_reward_c_per_kwh=0.0001,
            kwh_limit=200.0, limit_applies_to='export_only',
            default_mode='tou', event_window_hours=2.0,
            vpp_threshold_dollars_per_mwh=1000.0,
            min_export_price_dollars_per_mwh=-1.0,
        )