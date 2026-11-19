"""Tests for the TOU export-order nudge outside perfect_foresight_lp_advanced.

Covers the shared helpers in prices.py, the rolling window planner
(AdvancedWindowPlanner), the degradation-cost guard, and the scenario YAML /
ScenarioConfig handling of apply_export_order_nudge (PB-10).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from model.core.prices import (
    TOU_EXPORT_NUDGE_MAX, apply_tou_export_order_nudge,
    build_lp_price_vectors_advanced, build_tou_export_order_nudge,
    check_export_order_nudge_degradation,
)
from model.core.scenario import (
    BatterySpec, EXPORT_ORDER_NUDGE_COORDINATION_TYPES, RollingConfig,
    ScenarioConfig,
)
from model.core.tariff import NetworkTariff, RetailTariff, TariffPeriod
from model.components.perfect_foresight_lp_advanced import (
    perfect_foresight_lp_advanced,
)
from model.components.rolling_advanced import AdvancedWindowPlanner
from model.main import load_scenarios

IH = 1.0
STATE = 'NSW'


def _idx(n, start='2022-07-04 01:00:00'):
    return pd.date_range(start, periods=n, freq='60min', tz='Etc/GMT-10')


def _network():
    """All rates 0 and loss factors 1, so the price vectors are the retail
    layer alone."""
    return NetworkTariff(
        name='N', state='NSW',
        usage_tariffs=(TariffPeriod(0, 24, 0.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 0.0),),
        daily_c_per_day=0.0,
        usage_dlf=1.0, export_dlf=1.0, import_tlf=1.0, export_tlf=1.0,
    )


def _flat_feedin_tariff():
    return RetailTariff(
        name='FlatFeedin', state='NSW', type='tou', daily_c_per_day=0.0,
        usage_tariffs=(TariffPeriod(0, 24, 20.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 10.0),),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
    )


def _free_window_tariff():
    """Zero feed-in all day and free imports 12:00-15:00, so the import and
    export prices are equal (both zero) inside the free window."""
    return RetailTariff(
        name='FreeWindow', state='NSW', type='tou', daily_c_per_day=0.0,
        usage_tariffs=(TariffPeriod(0, 12, 20.0), TariffPeriod(12, 15, 0.0),
                       TariffPeriod(15, 24, 20.0)),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 0.0),),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
    )


def _bspec(c_deg=0.005, capacity=4.0, power=2.0):
    return BatterySpec(capacity_kwh=capacity, backup_reserve_pct=0.0,
                       power_kw=power, charge_efficiency=1.0,
                       discharge_efficiency=1.0, c_deg_per_kwh=c_deg)


def _planner(rt, n, bspec=None, nudge=True, horizon=None):
    return AdvancedWindowPlanner(
        index=_idx(n), retail_tariff=rt, network_tariff=_network(),
        state=STATE, battery_spec=bspec or _bspec(),
        horizon_intervals=horizon or n, interval_hours=IH,
        curtail_cost_per_kwh_dollars=0.0,
        apply_export_order_nudge=nudge)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def test_apply_nudge_caps_at_the_import_price():
    p_imp = np.array([0.20, 0.00, 0.10, 0.05])
    p_exp = np.array([0.10, 0.00, 0.10, 0.06])   # last: already above p_imp
    nudge = np.array([1e-6, 1e-6, 1e-6, 1e-6])
    out = apply_tou_export_order_nudge(p_imp, p_exp, nudge)
    np.testing.assert_allclose(out, [0.10 + 1e-6, 0.0, 0.10, 0.06])


def test_degradation_guard_threshold():
    check_export_order_nudge_degradation(TOU_EXPORT_NUDGE_MAX, 'P')
    with pytest.raises(ValueError, match='c_deg_per_kwh'):
        check_export_order_nudge_degradation(TOU_EXPORT_NUDGE_MAX / 2, 'P')
    with pytest.raises(ValueError, match='c_deg_per_kwh'):
        check_export_order_nudge_degradation(0.0, 'P')


# ---------------------------------------------------------------------------
# AdvancedWindowPlanner
# ---------------------------------------------------------------------------

def test_planner_base_export_carries_the_nudge():
    n = 48
    rt = _flat_feedin_tariff()
    off = _planner(rt, n, nudge=False)
    on = _planner(rt, n, nudge=True)
    nudge = build_tou_export_order_nudge(_idx(n), rt, STATE, IH).to_numpy()
    np.testing.assert_allclose(on.base_exp - off.base_exp, nudge, atol=1e-15)
    np.testing.assert_array_equal(on.base_imp, off.base_imp)


def test_planner_window_prices_carry_the_nudge():
    n = 24
    rt = _flat_feedin_tariff()
    on = _planner(rt, n, nudge=True, horizon=6)
    nudge = build_tou_export_order_nudge(_idx(n), rt, STATE, IH).to_numpy()
    _, p_exp = on.window_prices(None, 5, 6)
    np.testing.assert_allclose(p_exp, 0.10 + nudge[5:11], atol=1e-15)


def test_planner_nudge_is_dropped_where_import_equals_export():
    n = 24
    rt = _free_window_tariff()
    off = _planner(rt, n, nudge=False)
    on = _planner(rt, n, nudge=True)
    equal = np.isclose(off.base_imp, off.base_exp)
    assert equal.any(), "fixture should have intervals with p_imp == p_exp"
    np.testing.assert_array_equal(on.base_exp[equal], off.base_exp[equal])
    nudge = build_tou_export_order_nudge(_idx(n), rt, STATE, IH).to_numpy()
    np.testing.assert_allclose(on.base_exp[~equal],
                               off.base_exp[~equal] + nudge[~equal],
                               atol=1e-15)
    assert (on.base_exp <= on.base_imp).all()


def test_planner_rejects_the_nudge_without_degradation_cost():
    with pytest.raises(ValueError, match='c_deg_per_kwh'):
        _planner(_flat_feedin_tariff(), 24, bspec=_bspec(c_deg=0.0))
    # Off, the same battery is accepted.
    _planner(_flat_feedin_tariff(), 24, bspec=_bspec(c_deg=0.0), nudge=False)


def test_planner_window_charges_from_the_first_surplus():
    """The rolling counterpart of test_export_order_nudge_charges_before_exporting
    in test_perfect_foresight_lp_advanced: a 4-interval window with a flat
    feed-in rate, 2 kWh of surplus per interval, and a 4 kWh battery that must
    be full at the end. With the nudge the window charges at t=0 and t=1 and
    exports at t=2 and t=3."""
    n = 4
    p = _planner(_flat_feedin_tariff(), n)
    p_imp, p_exp = p.window_prices(None, 0, n)
    x = p.solve_window(0, n, np.full(n, 3.0), np.full(n, 1.0),
                       p_imp, p_exp, 0.0, 4.0)
    assert x is not None
    charge = x[0:n]
    export = x[3 * n:4 * n]
    np.testing.assert_allclose(charge, [2.0, 2.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(export, [0.0, 0.0, 2.0, 2.0], atol=1e-6)


# ---------------------------------------------------------------------------
# perfect_foresight_lp_advanced
# ---------------------------------------------------------------------------

def test_perfect_foresight_rejects_the_nudge_without_degradation_cost():
    n = 4
    idx = _idx(n)
    rt = _flat_feedin_tariff()
    prices = pd.Series(100.0, index=idx)
    p_imp, p_exp = build_lp_price_vectors_advanced(
        idx, prices, _network(), rt, STATE, IH)
    with pytest.raises(ValueError, match='c_deg_per_kwh'):
        perfect_foresight_lp_advanced(
            'S', pd.Series(3.0, index=idx), pd.Series(1.0, index=idx),
            p_imp, p_exp, _bspec(c_deg=0.0), soc_init_kwh=0.0,
            soc_final_kwh=4.0, retail_tariff=rt, prices_mwh=prices,
            state=STATE, interval_hours=IH, apply_export_order_nudge=True)


def test_perfect_foresight_nudge_is_capped_in_a_free_window():
    """Inside the free window import and export are both zero. An uncapped
    nudge would make importing and exporting the same kWh profitable."""
    n = 24
    idx = _idx(n, start='2022-07-04 00:00:00')
    rt = _free_window_tariff()
    prices = pd.Series(100.0, index=idx)
    p_imp, p_exp = build_lp_price_vectors_advanced(
        idx, prices, _network(), rt, STATE, IH)
    tr = perfect_foresight_lp_advanced(
        'S', pd.Series(0.0, index=idx), pd.Series(1.0, index=idx),
        p_imp, p_exp, _bspec(capacity=10.0, power=5.0), soc_init_kwh=0.0,
        soc_final_kwh=0.0, retail_tariff=rt, prices_mwh=prices,
        state=STATE, interval_hours=IH, apply_export_order_nudge=True)
    assert tr.export_kwh.max() == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# ScenarioConfig and load_scenarios (PB-10)
# ---------------------------------------------------------------------------

def _rolling():
    return RollingConfig(forecast_source='perfect', horizon_hours=18.0)


def test_scenario_accepts_the_nudge_on_each_type_that_applies_it():
    for ct in EXPORT_ORDER_NUDGE_COORDINATION_TYPES:
        ScenarioConfig(
            state='NSW', year=2022, cohort='solar_battery',
            network_tariff_id='n', retail_tariff_id='r',
            coordination_type=ct, battery_spec=_bspec(),
            rolling=_rolling() if ct.startswith('rolling') else None,
            op_retail_tariff_id='op' if 'vpp' in ct else None,
            apply_export_order_nudge=True)


@pytest.mark.parametrize('ct', ['self_consumption',
                                'rolling_perfect_foresight_lp',
                                'vpp_sc_event'])
def test_scenario_rejects_the_nudge_where_it_has_no_effect(ct):
    with pytest.raises(ValueError, match='apply_export_order_nudge'):
        ScenarioConfig(
            state='NSW', year=2022, cohort='solar_battery',
            network_tariff_id='n', retail_tariff_id='r',
            coordination_type=ct, battery_spec=_bspec(),
            rolling=_rolling() if ct.startswith('rolling') else None,
            op_retail_tariff_id='op' if 'vpp' in ct else None,
            apply_export_order_nudge=True)


def _yaml(tmp_path: Path, **extra) -> Path:
    data = {
        'name': 'nudge_test',
        'states': ['NSW'], 'years': [2022],
        'cohort': 'solar_battery',
        'coordination_type': 'rolling_perfect_foresight_lp_advanced',
        'network_tariff_id': 'n', 'retail_tariff_id': 'r',
        'battery_spec': {'capacity_kwh': 10, 'backup_reserve_pct': 20,
                         'power_kw': 5, 'charge_efficiency': 0.95,
                         'discharge_efficiency': 0.95,
                         'c_deg_per_kwh': 0.039},
        'rolling': {'forecast_source': 'perfect', 'horizon_hours': 18.0},
    }
    data.update(extra)
    path = tmp_path / 'nudge.yaml'
    with open(path, 'w') as f:
        yaml.safe_dump(data, f)
    return path


def test_load_scenarios_reads_the_nudge(tmp_path):
    _, scenarios = load_scenarios(_yaml(tmp_path, apply_export_order_nudge=True))
    assert [s.apply_export_order_nudge for s in scenarios] == [True]
    _, scenarios = load_scenarios(_yaml(tmp_path))
    assert [s.apply_export_order_nudge for s in scenarios] == [False]


def test_load_scenarios_rejects_a_non_boolean_nudge(tmp_path):
    with pytest.raises(ValueError, match='true or false'):
        load_scenarios(_yaml(tmp_path, apply_export_order_nudge='yes please'))


def test_load_scenarios_rejects_unknown_top_level_keys(tmp_path):
    with pytest.raises(ValueError, match='apply_export_nudge'):
        load_scenarios(_yaml(tmp_path, apply_export_nudge=True))


def test_every_repo_config_loads():
    """Unknown top-level keys are now an error, so every config in
    model/configs must use only the known set."""
    configs = sorted(
        (Path(__file__).resolve().parents[1] / 'configs').glob('*.yaml'))
    assert configs
    for path in configs:
        load_scenarios(path)
