"""Tests for the no-VPP self-consumption curve in financial_analysis.py
(docs/financial_analysis_spec.md, section 1a): the full-range view draws each
plan's self-consumption without VPP participation."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'analysis' / 'scripts'))
_spec = importlib.util.spec_from_file_location(
    'financial_analysis', REPO_ROOT / 'analysis' / 'scripts' / 'financial_analysis.py')
fa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fa)

NO_VPP_RUNS = {
    'Amber': 'full_run_self_consumption_amber',
    'GloBird ZeroHero': 'full_run_self_consumption_globird_novpp',
    'AGL Battery Rewards': 'full_run_self_consumption_agl_novpp',
    'EnergyAustralia Battery Ease': 'full_run_self_consumption_ea',
    'Origin Battery Lite': 'full_run_self_consumption_origin_novpp',
}


def _plans(basis):
    return {fa._offer_name(p): p for p in fa.plans_for_basis(basis)}


def test_registry_names_the_no_vpp_runs():
    for basis in fa.RUN_BASES:
        for offer, p in _plans(basis).items():
            assert p['sc_no_vpp_dir'] == NO_VPP_RUNS[offer]


def test_no_vpp_mode_by_basis():
    for basis in fa.RUN_BASES:
        p = _plans(basis)
        # plain self-consumption on every basis: their own mode
        assert fa.sc_no_vpp_mode(p['Amber']) == 'SC'
        assert fa.sc_no_vpp_mode(p['EnergyAustralia Battery Ease']) == 'SC_EA'
        assert fa.sc_no_vpp_mode(p['GloBird ZeroHero']) == 'SC_GB_NO_VPP'
        # Origin's conservative self-consumption is the no-VPP run itself
        expected = 'SC_ORIGIN' if basis == 'conservative' else 'SC_ORIGIN_NO_VPP'
        assert fa.sc_no_vpp_mode(p['Origin Battery Lite']) == expected
        # AGL's conservative run carries the enrolment payment, so it needs
        # its own unadjusted mode on every basis
        assert fa.sc_no_vpp_mode(p['AGL Battery Rewards']) == 'SC_AGL_NO_VPP'
    assert fa.sc_no_vpp_mode(dict(sc_mode='X', sc_dir='a')) == 'X'


def test_full_view_draws_no_vpp_self_consumption():
    fa.install_plans(fa.plans_for_basis('achievable'), 'SC')
    try:
        modes = [m for _, m, ls, _ in fa._view_series('full') if ls == '-']
        assert modes == ['SC', 'SC_GB_NO_VPP', 'SC_AGL_NO_VPP', 'SC_EA',
                         'SC_ORIGIN_NO_VPP']
        # the default view keeps the as-joined self-consumption runs
        default = [m for _, m, _, _ in fa._view_series('default')]
        assert 'SC_AGL' in default and 'SC_AGL_NO_VPP' not in default
        assert 'SC_ORIGIN' in default
    finally:
        fa.install_plans([dict(p) for p in fa._BASE_PLANS])


def _write_bills(root, run, scenario, rows):
    d = root / 'data' / 'output' / 'runs' / run / 'scenarios' / scenario
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=['site_id', 'household_total',
                                'wholesale_usage_cost',
                                'wholesale_feedin_credit']).to_csv(
        d / 'bills.csv', index=False)


def _fixture(tmp_path, basis):
    """Two sites, every run of every plan on one basis, NSW 2023. Each run's
    household bill is a distinct number so the loaded mode can be told
    apart. Returns (root, {run: household_total at every size})."""
    plans = fa.plans_for_basis(basis)
    runs = {fa.DEFAULT_BASELINE_RUN}
    for p in plans:
        runs |= {p['sc_dir'], p['lp_dir'], p['sc_no_vpp_dir']}
        if p['zero_batt_dir']:
            runs.add(p['zero_batt_dir'])
    bill = {run: 1000.0 - 10 * i for i, run in enumerate(sorted(runs))}
    solar = {fa.DEFAULT_BASELINE_RUN} | {p['zero_batt_dir'] for p in plans
                                          if p['zero_batt_dir']}
    for run in runs:
        rows = [('S1', bill[run], 100.0, -20.0), ('S2', bill[run], 100.0, -20.0)]
        if run in solar:
            _write_bills(tmp_path, run, 'NSW_2023', rows)
        else:
            for size in fa.BATTERY_SIZES:
                _write_bills(tmp_path, run, f'NSW_2023_{size}kWh', rows)
    return tmp_path, plans, bill


def test_curves_load_the_no_vpp_run_without_payment(tmp_path):
    root, plans, bill = _fixture(tmp_path, 'conservative')
    fa.install_plans(plans, 'SC')
    try:
        build = fa.make_scenario_builder(root, None)
        built = build('NSW', '2023')
        curves = built['curves']
        fa.apply_basis_adjustments(curves, 'conservative')
        hh = curves['hh_saving']
        base = bill[fa.DEFAULT_BASELINE_RUN]
        agl = next(p for p in plans if p['sc_mode'] == 'SC_AGL')
        assert agl['sc_dir'] == agl['sc_no_vpp_dir']
        # the no-VPP AGL curve is the run itself; SC_AGL adds the payment
        nv = base - bill['full_run_self_consumption_agl_novpp']
        assert np.allclose(hh['SC_AGL_NO_VPP'][1:], nv)
        assert np.allclose(hh['SC_AGL'][1:],
                           nv + fa.CONSERVATIVE_PAYMENTS['SC_AGL'])
        assert np.allclose(hh['SC_GB_NO_VPP'][1:],
                           base - bill['full_run_self_consumption_globird_novpp'])
        assert 'SC_ORIGIN_NO_VPP' not in hh and 'SC_EA_NO_VPP' not in hh
        rows = pd.DataFrame(fa._curve_rows('NSW_2023', curves))
        nv_rows = rows[rows['series'] == 'self-consumption, no VPP']
        assert set(nv_rows['mode']) == {'SC_AGL_NO_VPP', 'SC_GB_NO_VPP'}
        assert set(nv_rows['offer']) == {'AGL Battery Rewards',
                                         'GloBird ZeroHero'}
        # the site savings CSV keeps only the as-joined and optimised modes
        sites = fa._site_rows('NSW', '2023', built, 'conservative')
        assert not sites['mode'].str.endswith(fa.NO_VPP_SUFFIX).any()
    finally:
        fa.install_plans([dict(p) for p in fa._BASE_PLANS])


def test_achievable_no_vpp_curves_differ_from_the_vpp_runs(tmp_path):
    root, plans, bill = _fixture(tmp_path, 'achievable')
    fa.install_plans(plans, 'SC')
    try:
        hh = fa.make_scenario_builder(root, None)('NSW', '2023')['curves'][
            'hh_saving']
        base = bill[fa.DEFAULT_BASELINE_RUN]
        for mode, vpp_run, no_vpp_run in (
                ('SC_AGL', 'full_run_rolling_self_consumption_agl',
                 'full_run_self_consumption_agl_novpp'),
                ('SC_ORIGIN', 'full_run_rolling_self_consumption_origin',
                 'full_run_self_consumption_origin_novpp')):
            assert np.allclose(hh[mode][1:], base - bill[vpp_run])
            assert np.allclose(hh[mode + fa.NO_VPP_SUFFIX][1:],
                               base - bill[no_vpp_run])
        # the zero-capacity anchor is shared with the plan's other curves
        assert hh['SC_AGL_NO_VPP'][0] == hh['SC_AGL'][0]
    finally:
        fa.install_plans([dict(p) for p in fa._BASE_PLANS])


def test_full_range_figure_draws(tmp_path):
    root, plans, _ = _fixture(tmp_path, 'achievable')
    fa.install_plans(plans, 'SC')
    try:
        curves = fa.make_scenario_builder(root, None)('NSW', '2023')['curves']
        fig = fa._figure_single(curves['hh_saving'], 'full', 'Full range',
                                'upper left', None)
        n_sc = sum(1 for line in fig.axes[0].get_lines()
                   if line.get_linestyle() == '-' and len(line.get_xdata()) > 20)
        assert n_sc == 5
    finally:
        fa.install_plans([dict(p) for p in fa._BASE_PLANS])
