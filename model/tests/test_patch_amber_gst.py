"""Tests for analysis/scripts/patch_amber_gst.py: the GST patch arithmetic,
its agreement with the core model's billing, and the marker that stops a
scenario being patched twice."""
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
from model.components.billing import GST_RATE, compute_bill_and_monthly_bills
from model.orchestrate import _bill_row

SCRIPTS = Path(__file__).resolve().parents[2] / 'analysis' / 'scripts'
_spec = importlib.util.spec_from_file_location(
    'patch_amber_gst', SCRIPTS / 'patch_amber_gst.py')
pg = importlib.util.module_from_spec(_spec)
sys.modules['patch_amber_gst'] = pg
_spec.loader.exec_module(pg)

SITES = ['1001', '1002', '0103']   # a leading zero must survive the round trip


def _network():
    return NetworkTariff(
        name='TestNet', state='NSW',
        usage_tariffs=(TariffPeriod(0, 24, 10.0),), usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, -1.0),), daily_c_per_day=100.0,
        usage_dlf=1.04, export_dlf=0.97, import_tlf=1.01, export_tlf=0.99,
    )


def _amber(gst: bool):
    return RetailTariff(
        name='TestAmber', state='NSW', type='wholesale_exposed',
        daily_c_per_day=82.0, usage_tariffs=(), usage_tariffs_weekend=None,
        feedin_tariffs=(), usage_other_c_per_kwh=4.0,
        feedin_other_c_per_kwh=0.0, gst_on_wholesale_import=gst,
    )


def _inputs():
    """Two local months (31 Jan and 1 Feb) of net flows and spot prices with
    many negative-price intervals."""
    idx = pd.date_range('2022-01-31 00:05:00', periods=576, freq='5min',
                        tz='Etc/GMT-10')
    rng = np.random.default_rng(11)
    prices = pd.Series(rng.uniform(-300.0, 500.0, len(idx)), index=idx)
    flows = pd.DataFrame({s: rng.uniform(-0.4, 0.5, len(idx)) for s in SITES},
                         index=idx)
    return flows, prices


def _core_bills(flows, prices, gst: bool):
    return pg.core_reference_bills(
        flows, prices, _network(), _amber(gst), 'NSW',
        ['bills.csv', 'bills_01.csv', 'bills_02.csv'])


def _write_scenario(tmp_path, bills: dict) -> Path:
    scn = tmp_path / 'NSW_2022_20kWh'
    scn.mkdir()
    for f, df in bills.items():
        (scn / f).write_bytes(pg.bills_bytes(df))
    return scn


def _snapshot(scn: Path) -> dict:
    return {p.name: p.read_bytes() for p in sorted(scn.iterdir())}


@pytest.fixture
def old_scenario(tmp_path):
    """A scenario billed without GST on spot imports, as the existing runs
    were, plus the bills the fixed billing gives for the same flows."""
    flows, prices = _inputs()
    old = _core_bills(flows, prices, gst=False)
    new = _core_bills(flows, prices, gst=True)
    scn = _write_scenario(tmp_path, old)
    gst = new['bills.csv']['household_total'] - old['bills.csv']['household_total']
    assert (gst.abs() > 0.05).all()             # the patch changes every site
    return scn, flows, prices, old, new


def _core_reference(flows, prices):
    def reference(files):
        return pg.core_reference_bills(flows, prices, _network(), _amber(True),
                                       'NSW', files)
    return reference


PROV = {'script': 'test', 'git_commit': 'test', 'written': 'test'}


def test_patch_bills_adds_gst_to_three_columns_only():
    df = pd.DataFrame({'site_id': ['1', '2'], 'plan_type': 'wholesale_exposed',
                       **{c: [1.0, -2.0] for c in pg.BILL_COLS}})
    df['wholesale_usage_cost'] = [10.0, -4.0]
    out = pg.patch_bills(df, GST_RATE)
    for c in pg.BILL_COLS:
        expected = df[c] + (GST_RATE * df['wholesale_usage_cost']
                            if c in pg.PATCHED_COLS else 0.0)
        np.testing.assert_allclose(out[c], expected, err_msg=c)
    assert list(out.columns) == list(df.columns)


def test_patched_bills_equal_core_billing_with_gst(old_scenario):
    scn, flows, prices, old, new = old_scenario
    rep = pg.patch_scenario(scn, GST_RATE, _core_reference(flows, prices), PROV)
    assert rep['status'] == 'PASS', rep
    assert rep['max_verify_diff'] < 1e-9, rep
    for f, ref in new.items():
        got = pg.read_bills(scn / f).set_index('site_id')
        ref = ref.set_index('site_id')
        assert list(got.index) == SITES
        diff = (got[list(pg.BILL_COLS)] - ref.loc[got.index, list(pg.BILL_COLS)]).abs()
        assert float(diff.to_numpy().max()) < 1e-9, (f, diff.max())
    marker = yaml.safe_load((scn / pg.MARKER).read_text())
    assert marker['status'] == 'complete'
    assert sorted(marker['files']) == ['bills.csv', 'bills_01.csv', 'bills_02.csv']
    for f, h in marker['files'].items():
        assert pg.md5((scn / f).read_bytes()) == h['post_md5']
        assert pg.md5(pg.bills_bytes(old[f])) == h['pre_md5']


def test_second_run_skips_and_changes_nothing(old_scenario):
    scn, flows, prices, *_ = old_scenario
    pg.patch_scenario(scn, GST_RATE, _core_reference(flows, prices), PROV)
    before = _snapshot(scn)
    rep = pg.patch_scenario(scn, GST_RATE, _core_reference(flows, prices), PROV)
    assert rep['status'] == 'SKIP', rep
    assert _snapshot(scn) == before


def _interrupt_after(monkeypatch, n_writes):
    real = pg._write_atomic
    calls = {'n': 0}

    def flaky(path, data):
        if calls['n'] == n_writes:
            raise OSError('simulated interruption')
        calls['n'] += 1
        real(path, data)
    monkeypatch.setattr(pg, '_write_atomic', flaky)


def test_interrupted_patch_resumes_without_patching_twice(old_scenario, monkeypatch,
                                                         tmp_path):
    scn, flows, prices, old, new = old_scenario
    # Writes: marker, bills.csv, then bills_01.csv fails.
    _interrupt_after(monkeypatch, 2)
    with pytest.raises(OSError, match='simulated'):
        pg.patch_scenario(scn, GST_RATE, _core_reference(flows, prices), PROV)
    monkeypatch.undo()
    marker = yaml.safe_load((scn / pg.MARKER).read_text())
    assert marker['status'] == 'in_progress'
    states = {f: ('post' if pg.md5((scn / f).read_bytes()) == h['post_md5']
                  else 'pre' if pg.md5((scn / f).read_bytes()) == h['pre_md5']
                  else '?') for f, h in marker['files'].items()}
    assert states == {'bills.csv': 'post', 'bills_01.csv': 'pre',
                      'bills_02.csv': 'pre'}

    rep = pg.patch_scenario(scn, GST_RATE, _core_reference(flows, prices), PROV)
    assert rep['status'] == 'PASS' and rep['message'] == 'resumed', rep
    clean = tmp_path / 'clean'
    clean.mkdir()
    ref_scn = _write_scenario(clean, old)
    pg.patch_scenario(ref_scn, GST_RATE, _core_reference(flows, prices), PROV)
    for f in old:
        assert (scn / f).read_bytes() == (ref_scn / f).read_bytes(), f
    got = pg.read_bills(scn / 'bills.csv')
    np.testing.assert_allclose(
        got['household_total'],
        old['bills.csv']['household_total']
        + GST_RATE * old['bills.csv']['wholesale_usage_cost'])
    assert yaml.safe_load((scn / pg.MARKER).read_text())['status'] == 'complete'


def test_file_changed_after_interruption_stops(old_scenario, monkeypatch):
    scn, flows, prices, *_ = old_scenario
    _interrupt_after(monkeypatch, 2)
    with pytest.raises(OSError):
        pg.patch_scenario(scn, GST_RATE, _core_reference(flows, prices), PROV)
    monkeypatch.undo()
    df = pg.read_bills(scn / 'bills_02.csv')
    df.loc[0, 'household_total'] += 1.0
    (scn / 'bills_02.csv').write_bytes(pg.bills_bytes(df))
    before = _snapshot(scn)
    with pytest.raises(pg.PatchError, match='matches neither'):
        pg.patch_scenario(scn, GST_RATE, _core_reference(flows, prices), PROV)
    assert _snapshot(scn) == before


def test_verification_failure_writes_nothing(old_scenario):
    scn, flows, prices, *_ = old_scenario
    before = _snapshot(scn)
    # Reference bills without GST: the patched rows must disagree.
    def reference(files):
        return pg.core_reference_bills(flows, prices, _network(), _amber(False),
                                       'NSW', files)
    rep = pg.patch_scenario(scn, GST_RATE, reference, PROV)
    assert rep['status'] == 'FAIL' and rep['max_verify_diff'] > 0.05, rep
    assert _snapshot(scn) == before


def test_wrong_rate_fails_verification(old_scenario):
    scn, flows, prices, *_ = old_scenario
    before = _snapshot(scn)
    rep = pg.patch_scenario(scn, 0.05, _core_reference(flows, prices), PROV)
    assert rep['status'] == 'FAIL', rep
    assert _snapshot(scn) == before


def test_dry_run_writes_nothing(old_scenario):
    scn, flows, prices, *_ = old_scenario
    before = _snapshot(scn)
    rep = pg.patch_scenario(scn, GST_RATE, _core_reference(flows, prices), PROV,
                            dry_run=True)
    assert rep['status'] == 'DRY', rep
    assert _snapshot(scn) == before


def test_rejects_other_plan_types(old_scenario):
    scn, flows, prices, old, _ = old_scenario
    for f, df in old.items():
        df = df.copy()
        df['plan_type'] = 'tou'
        (scn / f).write_bytes(pg.bills_bytes(df))
    with pytest.raises(pg.PatchError, match='plan_type'):
        pg.patch_scenario(scn, GST_RATE, _core_reference(flows, prices), PROV)
    assert not (scn / pg.MARKER).exists()


def test_sample_sites_spread_and_bounded():
    sites = [f'{i:04d}' for i in range(100)]
    pick = pg.sample_sites(sites, 5)
    assert pick[0] == '0000' and pick[-1] == '0099' and len(pick) == 5
    assert pg.sample_sites(sites[:3], 5) == sites[:3]


def test_monthly_bills_are_verified_too(old_scenario):
    scn, flows, prices, *_ = old_scenario
    df = pg.read_bills(scn / 'bills_02.csv')
    df.loc[1, 'retail_usage_cost'] += 0.5      # annual bill left correct
    (scn / 'bills_02.csv').write_bytes(pg.bills_bytes(df))
    before = _snapshot(scn)
    rep = pg.patch_scenario(scn, GST_RATE, _core_reference(flows, prices), PROV)
    assert rep['status'] == 'FAIL' and rep['max_verify_diff'] > 0.4, rep
    assert _snapshot(scn) == before
