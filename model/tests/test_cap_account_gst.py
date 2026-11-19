"""GST on spot imports in analysis/scripts/cap_account_analysis.py
(cap_account_spec.md sections 2, 5, 6 and 17): household amounts carry the
GST billing charges on the spot energy of imports; exports carry none."""
import dataclasses
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model.core.tariff import (NetworkTariff, RetailTariff, TariffPeriod,
                               load_retail_tariff)
from model.core.site_result import SiteTraces
from model.core.prices import build_layer_rates
from model.components.billing import GST_RATE, compute_bill

SCRIPTS = Path(__file__).resolve().parents[2] / 'analysis' / 'scripts'
sys.path.insert(0, str(SCRIPTS))
_spec = importlib.util.spec_from_file_location('cap_account_analysis',
                                               SCRIPTS / 'cap_account_analysis.py')
ca = importlib.util.module_from_spec(_spec)
sys.modules['cap_account_analysis'] = ca
_spec.loader.exec_module(ca)

TARIFFS = Path(__file__).resolve().parents[1] / 'tariffs'
ILF, ELF = 1.04 * 1.01, 0.97 * 0.99
SITES = ['A', 'B', 'C']


def _network():
    return NetworkTariff(
        name='Net', state='NSW',
        usage_tariffs=(TariffPeriod(0, 15, 5.9), TariffPeriod(15, 21, 35.8),
                       TariffPeriod(21, 24, 5.9)),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 10, 0.0), TariffPeriod(10, 15, -1.36),
                        TariffPeriod(15, 24, 3.86)),
        daily_c_per_day=82.1, usage_dlf=1.04, export_dlf=0.97,
        import_tlf=1.01, export_tlf=0.99)


def _amber(gst):
    return RetailTariff(
        name='Amber', state='NSW', type='wholesale_exposed', daily_c_per_day=82.1918,
        usage_tariffs=(), usage_tariffs_weekend=None, feedin_tariffs=(),
        usage_other_c_per_kwh=4.081, feedin_other_c_per_kwh=0.0,
        gst_on_wholesale_import=gst)


def _inputs():
    """Two days with negative, ordinary and above-$300 prices; every site
    both imports and exports."""
    idx = pd.date_range('2023-03-01 00:05:00', periods=576, freq='5min',
                        tz='Etc/GMT-10')
    rng = np.random.default_rng(5)
    p = rng.uniform(-150.0, 250.0, len(idx))
    spikes = rng.random(len(idx)) < 0.05
    p[spikes] = rng.uniform(400.0, 12000.0, spikes.sum())
    prices = pd.Series(p, index=idx)
    net = pd.DataFrame({s: rng.uniform(-0.4, 0.5, len(idx)) for s in SITES}, index=idx)
    return idx, prices, net.clip(lower=0), (-net).clip(lower=0)


def _bills(imp, exp, prices, retail):
    z = pd.Series(0.0, index=imp.index)
    rows = []
    for s in imp.columns:
        tr = SiteTraces(site_id=s, import_kwh=imp[s], export_kwh=exp[s],
                        battery_soc_kwh=z, sol_to_home_kwh=z, sol_to_bat_kwh=z,
                        sol_to_grid_kwh=z, bat_to_home_kwh=z, grid_to_home_kwh=z,
                        grid_to_bat_kwh=z, bat_to_grid_kwh=z, curtail_kwh=z)
        rows.append(dataclasses.asdict(compute_bill(tr, prices, _network(), retail,
                                                    state='NSW')))
    return pd.DataFrame(rows).set_index('site_id')


def test_wholesale_net_gst_on_imports_only():
    idx = pd.date_range('2023-03-01 00:05:00', periods=3, freq='5min', tz='Etc/GMT-10')
    imp = pd.DataFrame({'A': [1.0, 2.0, 0.0]}, index=idx)
    exp = pd.DataFrame({'A': [0.0, 0.0, 3.0]}, index=idx)
    p = pd.Series([0.10, -0.05, 0.20], index=idx)        # $/kWh
    got = ca._wholesale_net(imp, exp, p, ILF, ELF, import_gst=0.1)['A'].to_numpy()
    np.testing.assert_allclose(got, [1.1 * 0.10 * ILF, 1.1 * 2 * -0.05 * ILF,
                                     -3 * 0.20 * ELF])
    plain = ca._wholesale_net(imp, exp, p, ILF, ELF)['A'].to_numpy()
    np.testing.assert_allclose(plain, [0.10 * ILF, 2 * -0.05 * ILF, -3 * 0.20 * ELF])


def test_cap_delta_gst_on_imports_only():
    idx = pd.date_range('2023-03-01 00:05:00', periods=3, freq='5min', tz='Etc/GMT-10')
    imp = pd.DataFrame({'A': [1.0, 1.0, 0.5]}, index=idx)
    exp = pd.DataFrame({'A': [0.0, 2.0, 0.5]}, index=idx)
    p = pd.Series([1.0, 1.3, 0.2], index=idx)             # $/kWh; x = 0.3
    got = ca._cap_delta(imp, exp, p, 0.3, ILF, ELF, import_gst=0.1)['A'].to_numpy()
    np.testing.assert_allclose(got, [0.7 * 1.1 * ILF,
                                     1.0 * (1.1 * ILF - 2 * ELF),
                                     0.0])


def test_capped_household_bill_matches_billing_at_capped_prices():
    """Baseline household_total - baseline wholesale + capped wholesale
    (spec section 7) equals the core bill at prices capped at x, for a plan
    with GST on spot imports, so the GST is neither dropped nor doubled."""
    idx, prices, imp, exp = _inputs()
    x = 300.0
    amber = _amber(True)
    base = _bills(imp, exp, prices, amber)
    capped = _bills(imp, exp, prices.clip(upper=x), amber)
    g = ca._import_gst(amber)
    p, pc = prices / 1000.0, np.minimum(prices / 1000.0, x / 1000.0)
    got = (base['household_total']
           - ca._wholesale_net(imp, exp, p, ILF, ELF, g).sum()
           + ca._wholesale_net(imp, exp, pc, ILF, ELF, g).sum())
    diff = (got - capped['household_total']).abs().max()
    assert diff < 1e-9, diff
    # Non-vacuity: the cap binds and the GST matters.
    assert (base['household_total'] - capped['household_total']).abs().max() > 5.0
    wrong = (base['household_total'] - ca._wholesale_net(imp, exp, p, ILF, ELF).sum()
             + ca._wholesale_net(imp, exp, pc, ILF, ELF).sum())
    assert (wrong - capped['household_total']).abs().max() > 0.05


def test_amber_import_rate_reproduces_the_amber_import_charges():
    idx, prices, imp, exp = _inputs()
    for gst in (True, False):
        amber = _amber(gst)
        bills = _bills(imp, exp, prices, amber)
        rates = build_layer_rates(idx, prices, _network(), amber, 'NSW', 5 / 60)
        c = ca._amber_import_rate(rates, amber)
        for s in SITES:
            billed = (bills.loc[s, 'wholesale_usage_cost'] + bills.loc[s, 'network_usage_cost']
                      + bills.loc[s, 'retail_usage_cost'])
            assert float(np.dot(imp[s], c)) == pytest.approx(billed, abs=1e-9), (gst, s)


def test_bills_gst_residual_catches_unpatched_bills():
    idx, prices, imp, exp = _inputs()
    amber = _amber(True)
    g = ca._import_gst(amber)
    good = _bills(imp, exp, prices, amber)
    assert ca._bills_gst_residual(good, imp, amber, g) < 1e-9
    old = _bills(imp, exp, prices, _amber(False))
    assert ca._bills_gst_residual(old, imp, amber, g) > 0.05


def test_import_gst_by_tariff():
    assert ca._import_gst(_amber(True)) == GST_RATE
    assert ca._import_gst(_amber(False)) == 0.0
    assert ca._import_gst(load_retail_tariff('NSW_amber_wholesale_2026', TARIFFS)) == GST_RATE
    assert ca._import_gst(load_retail_tariff('NSW_operator_wholesale_2026', TARIFFS)) == 0.0
    assert ca._import_gst(load_retail_tariff('NSW_agl_battery_rewards_tou_2026', TARIFFS)) == 0.0
