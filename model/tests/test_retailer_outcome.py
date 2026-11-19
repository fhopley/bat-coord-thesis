"""Tests for analysis/scripts/retailer_outcome_analysis.py: retailer margin ex
GST from bills, Amber's subscription check, the hour-of-day split and the
weighting."""
import dataclasses
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from model.core.tariff import NetworkTariff, RetailTariff, TariffPeriod
from model.core.site_result import SiteTraces
from model.core.prices import build_layer_rates
from model.components.billing import compute_bill
from model.core.localtime import local_wall_clock

SCRIPTS = Path(__file__).resolve().parents[2] / 'analysis' / 'scripts'
sys.path.insert(0, str(SCRIPTS))
_spec = importlib.util.spec_from_file_location('retailer_outcome_analysis',
                                               SCRIPTS / 'retailer_outcome_analysis.py')
ro = importlib.util.module_from_spec(_spec)
sys.modules['retailer_outcome_analysis'] = ro
_spec.loader.exec_module(ro)

SITES = ['A1', 'B2', 'C3']
GP_C = 4.5            # GreenPower, c/kWh inc GST
ENV_C = 1.9636        # env + market, c/kWh ex GST
H = 5 / 60


def _network():
    return NetworkTariff(
        name='Net', state='NSW',
        usage_tariffs=(TariffPeriod(0, 15, 5.9), TariffPeriod(15, 21, 35.8),
                       TariffPeriod(21, 24, 5.9)),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 10, 0.0), TariffPeriod(10, 15, -1.36),
                        TariffPeriod(15, 24, 3.86)),
        daily_c_per_day=82.1, usage_dlf=1.0429, export_dlf=0.9943,
        import_tlf=1.004, export_tlf=1.004)


def _tou():
    return RetailTariff(
        name='TOU', state='NSW', type='tou', daily_c_per_day=139.0,
        usage_tariffs=(TariffPeriod(0, 17, 29.8), TariffPeriod(17, 21, 54.0),
                       TariffPeriod(21, 24, 29.8)),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 17, 3.0), TariffPeriod(17, 21, 29.0),
                        TariffPeriod(21, 24, 3.0)),
        usage_other_c_per_kwh=GP_C, feedin_other_c_per_kwh=0.0)


ADDERS_C = 4.081


def _amber(gst=True):
    return RetailTariff(
        name='Amber', state='NSW', type='wholesale_exposed', daily_c_per_day=82.1918,
        usage_tariffs=(), usage_tariffs_weekend=None, feedin_tariffs=(),
        usage_other_c_per_kwh=ADDERS_C, feedin_other_c_per_kwh=0.0,
        gst_on_wholesale_import=gst)


def _inputs():
    idx = pd.date_range('2023-03-01 00:05:00', periods=288 * 3, freq='5min',
                        tz='Etc/GMT-10')
    rng = np.random.default_rng(3)
    prices = pd.Series(rng.uniform(-100.0, 400.0, len(idx)), index=idx)
    flows = pd.DataFrame({s: rng.uniform(-0.3, 0.4, len(idx)) for s in SITES}, index=idx)
    return idx, prices, flows


def _bills(flows, prices, retail):
    rows = []
    z = pd.Series(0.0, index=flows.index)
    for s in flows.columns:
        tr = SiteTraces(site_id=s, import_kwh=flows[s].clip(lower=0),
                        export_kwh=(-flows[s]).clip(lower=0), battery_soc_kwh=z,
                        sol_to_home_kwh=z, sol_to_bat_kwh=z, sol_to_grid_kwh=z,
                        bat_to_home_kwh=z, grid_to_home_kwh=z, grid_to_bat_kwh=z,
                        bat_to_grid_kwh=z, curtail_kwh=z)
        b = compute_bill(tr, prices, _network(), retail, state='NSW')
        rows.append(dataclasses.asdict(b))
    return pd.DataFrame(rows).set_index('site_id')


def test_tou_margins_hand_arithmetic():
    bills = pd.DataFrame({
        'plan_type': ['tou'], 'retail_usage_cost': [1100.0 + 45.0],
        'retail_daily_charges': [550.0], 'retail_feedin_credit': [-100.0],
        'retail_other_credits': [-10.0], 'wholesale_usage_cost': [400.0],
        'wholesale_feedin_credit': [-50.0], 'network_usage_cost': [660.0],
        'network_feedin_credit': [-11.0], 'network_daily_charges': [330.0],
    }, index=pd.Index(['S1'], name='site_id'))
    imports = pd.Series({'S1': 1000.0})
    m = ro.tou_margins(bills, imports, greenpower_c=4.5, env_market_c=2.0).iloc[0]
    assert m['import_revenue'] == pytest.approx(1000.0)        # (1145 - 45) / 1.1
    assert m['daily_revenue'] == pytest.approx(500.0)
    assert m['feedin_paid'] == pytest.approx(-100.0)
    assert m['other_credits'] == pytest.approx(-10.0)
    assert m['wholesale'] == pytest.approx(-350.0)
    assert m['network'] == pytest.approx(-890.0)               # (660 - 11 + 330) / 1.1
    assert m['env_market'] == pytest.approx(-20.0)
    assert m['hedging'] == 0.0
    assert m['margin'] == pytest.approx(1000 + 500 - 100 - 10 - 350 - 890 - 20)


def test_tou_margin_matches_interval_rebuild_from_layer_rates():
    """The bill-based margin equals the same margin built interval by interval
    from the layer rates, so no bill column is mis-signed or GST-treated
    twice."""
    idx, prices, flows = _inputs()
    bills = _bills(flows, prices, _tou())
    imports = flows.clip(lower=0).sum()
    m = ro.tou_margins(bills, imports, GP_C, ENV_C)
    lr = build_layer_rates(idx, prices, _network(), _tou(), 'NSW', H)
    n_days = 3
    for s in SITES:
        imp = flows[s].clip(lower=0).to_numpy()
        exp = (-flows[s]).clip(lower=0).to_numpy()
        expected = (
            np.dot(imp, lr.retail_import - GP_C / 100) / 1.1
            + 1.39 * n_days / 1.1
            - np.dot(exp, lr.retail_export)
            - (np.dot(imp, lr.wholesale_import) - np.dot(exp, lr.wholesale_export))
            - (np.dot(imp, lr.network_import) - np.dot(exp, lr.network_export)
               + 0.821 * n_days) / 1.1
            - ENV_C / 100 * imp.sum())
        assert m.loc[s, 'margin'] == pytest.approx(expected, abs=1e-9), s
    assert (m['feedin_paid'] < -1.0).all()                  # exports present


def test_hourly_split_sums_to_bill_import_terms():
    idx, prices, flows = _inputs()
    bills = _bills(flows, prices, _tou())
    imports = flows.clip(lower=0).sum()
    m = ro.tou_margins(bills, imports, GP_C, ENV_C)
    w = pd.Series([0.5, 0.3, 0.2], index=SITES)
    imp_mean = pd.Series(flows.clip(lower=0).to_numpy() @ w.to_numpy(), index=idx)
    lr = build_layer_rates(idx, prices, _network(), _tou(), 'NSW', H)
    hours = local_wall_clock(idx, 'NSW', H).hour
    h = ro.hourly_import_margin(imp_mean, lr, hours, GP_C, ENV_C)
    assert list(h.index) == list(range(24))
    wm = ro.weighted_mean(m, w, list(ro.COMPONENT_COLS))
    assert h['import_revenue'].sum() == pytest.approx(wm['import_revenue'], abs=1e-9)
    assert h['env_market'].sum() == pytest.approx(wm['env_market'], abs=1e-9)
    billed = ro.weighted_mean(bills, w, ('retail_usage_cost', 'network_usage_cost',
                                         'wholesale_usage_cost'))
    assert h['network_import'].sum() == pytest.approx(
        -billed['network_usage_cost'] / 1.1, abs=1e-9)
    assert h['wholesale_import'].sum() == pytest.approx(
        -billed['wholesale_usage_cost'], abs=1e-9)
    assert ro.billed_import_residual(billed, imp_mean, lr) < 1e-9
    # Non-vacuity: a wrong import series, or any one billed column off, is caught.
    assert ro.billed_import_residual(billed, imp_mean * 1.01, lr) > 0.01
    for col in billed.index:
        off = billed.copy()
        off[col] += 0.5
        assert ro.billed_import_residual(off, imp_mean, lr) > 0.4, col
    # Hours past 15:00 carry the network peak rate.
    assert h.loc[16, 'network_import'] / h.loc[16, 'import_kwh'] < \
        h.loc[3, 'network_import'] / h.loc[3, 'import_kwh']


def test_amber_check_passes_on_gst_bills_and_catches_unpatched_ones():
    idx, prices, flows = _inputs()
    imports = flows.clip(lower=0).sum()
    good = _bills(flows, prices, _amber(gst=True))
    comp, resid = ro.amber_margins(good, imports, ADDERS_C)
    assert resid < 1e-9
    assert comp['margin'].to_numpy() == pytest.approx(0.821918 * 3 / 1.1)
    old = _bills(flows, prices, _amber(gst=False))
    _, resid_old = ro.amber_margins(old, imports, ADDERS_C)
    assert resid_old == pytest.approx(
        (0.1 * old['wholesale_usage_cost']).abs().max(), abs=1e-9)
    assert resid_old > 0.05
    _, resid_imp = ro.amber_margins(good, imports * 1.02, ADDERS_C)
    assert resid_imp > 0.05


def test_same_dispatch_diff():
    idx, prices, flows = _inputs()
    a = _bills(flows, prices, _amber())
    t = _bills(flows, prices, _tou())
    assert ro.same_dispatch_diff(t, a) < 1e-9
    t2 = _bills(flows * 1.05, prices, _tou())
    assert ro.same_dispatch_diff(t2, a) > 0.05


def test_weighted_quantiles():
    v = np.array([4.0, 1.0, 3.0, 2.0, 7.0])
    qs = [0.1, 0.25, 0.5, 0.75, 0.9]
    np.testing.assert_allclose(ro.weighted_quantiles(v, np.ones(5), qs),
                               np.quantile(v, qs, method='inverted_cdf'))
    # Weight 3 on one value equals three copies of it.
    a = ro.weighted_quantiles(np.array([1.0, 5.0]), np.array([3.0, 1.0]), [0.25, 0.5])
    b = ro.weighted_quantiles(np.array([1.0, 1.0, 1.0, 5.0]), np.ones(4), [0.25, 0.5])
    np.testing.assert_allclose(a, b)


def test_national_mix_renormalises_over_states_present():
    rows = pd.DataFrame({'cohort': ['c', 'c', 'd'], 'plan': ['p', 'p', 'p'],
                         'state': ['NSW', 'SA', 'NSW'], 'margin': [100.0, 200.0, 50.0]})
    out = ro.national_mix(rows, ['cohort', 'plan'], ['margin'],
                          {'NSW': 0.3, 'SA': 0.1, 'VIC': 0.6})
    got = out.set_index('cohort')['margin']
    assert got['c'] == pytest.approx((0.3 * 100 + 0.1 * 200) / 0.4)
    assert got['d'] == pytest.approx(50.0)


def test_spread_share_below_and_year_weighting():
    sites = pd.DataFrame({
        'cohort': 'c', 'plan': 'p', 'state': 'NSW',
        'year': ['2021', '2021', '2022'],
        'weight': [0.5, 0.5, 1.0],
        'margin': [100.0, 300.0, 100.0]})
    s = ro.spread_stats(sites, subscription=200.0, state_mix=None).iloc[0]
    # 2021 half below, 2022 all below, years weighted equally.
    assert s['share_below_subscription'] == pytest.approx(0.75)


def test_spread_national_gives_each_state_its_mix_whatever_its_years():
    sites = pd.DataFrame({
        'cohort': 'c', 'plan': 'p',
        'state': ['NSW', 'NSW', 'SA'], 'year': ['2021', '2022', '2021'],
        'weight': [1.0, 1.0, 1.0], 'margin': [100.0, 300.0, 100.0]})
    s = ro.spread_stats(sites, subscription=200.0,
                        state_mix={'NSW': 0.5, 'SA': 0.5}).iloc[0]
    # NSW: half its weight below; SA: all below.
    assert s['share_below_subscription'] == pytest.approx(0.75)


def test_repo_tariff_components():
    assert ro.greenpower_c_per_kwh('NSW_agl_battery_rewards_tou_2026') == pytest.approx(1.1)
    assert ro.greenpower_c_per_kwh('NSW_ausgrid_ea025_ea029_2026') == 0.0
    assert ro.env_market_c_per_kwh_ex_gst('NSW_amber_wholesale_2026') == pytest.approx(
        (1.11 + 1.05) / 1.1)


def test_run_tariff_ids_from_rebill_source(tmp_path):
    run = tmp_path / 'full_run_no_cer_test'
    run.mkdir()
    (run / 'rebill_source.yaml').write_text(yaml.safe_dump({
        'source_config': 'model/configs/full_run_no_cer_amber.yaml',
        'retail_tariff_ids': {'NSW': {2022: 'NSW_agl_battery_rewards_tou_2026'}}}))
    retail, network = ro.run_tariff_ids(tmp_path, 'full_run_no_cer_test')
    assert retail['NSW']['2022'] == 'NSW_agl_battery_rewards_tou_2026'
    assert network['NSW']['2022'] == 'NSW_ausgrid_ea025_ea029_2026'
    with pytest.raises(ro.CheckError):
        ro.run_tariff_ids(tmp_path, 'full_run_missing')
