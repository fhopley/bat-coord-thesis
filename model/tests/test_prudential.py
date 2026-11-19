"""Tests for analysis/scripts/prudential_analysis.py: the AEMO MCL by the 2026
calculator (checked against AEMO's own workbook, recalculated by LibreOffice)
and by CLP v10.0, CLP rounding, cohort energy, simulated outstandings and the
state, national and retailer aggregation."""
import importlib.util
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / 'analysis' / 'scripts'
sys.path.insert(0, str(SCRIPTS))
_spec = importlib.util.spec_from_file_location('prudential_analysis',
                                               SCRIPTS / 'prudential_analysis.py')
pa = importlib.util.module_from_spec(_spec)
sys.modules['prudential_analysis'] = pa
_spec.loader.exec_module(pa)

REF = pa.REFERENCE_DIR
_HAVE_REF = all((REF / f).exists() for f in pa.CALCULATOR_FILES.values())
needs_ref = pytest.mark.skipif(not _HAVE_REF, reason='AEMO calculators not in docs/references')


@pytest.fixture(scope='module')
def params():
    return pa.load_regional_params(REF)


def _rp(p=100.0, rlwp=90.0, vf_osl=1.5, vf_pm=3.0, phh=None, erl=None):
    ti = np.arange(pa.TI_PER_DAY)
    if phh is None:
        phh = 80.0 + 60.0 * np.exp(-((ti - 216) / 20.0) ** 2) - 30.0 * np.exp(-((ti - 150) / 30.0) ** 2)
    if erl is None:
        erl = 500.0 + 150.0 * np.exp(-((ti - 220) / 25.0) ** 2)
    return pa.RegionalParams(p=p, rlwp=rlwp, vf_osl=vf_osl, vf_pm=vf_pm,
                             phh=np.asarray(phh, float), erl=np.asarray(erl, float))


def _household_profiles(seed=0, solar=1.0):
    """Debit and credit energy per TI, MWh, of a mean household: evening-peaked
    load, midday solar, net split by sign as the script does (so the two are
    never both positive in one TI), with meter noise."""
    rng = np.random.default_rng(seed)
    ti = np.arange(pa.TI_PER_DAY)
    load = (0.04 + 0.05 * np.exp(-((ti - 222) / 18.0) ** 2) + 0.02 * np.exp(-((ti - 90) / 15.0) ** 2))
    pv = solar * 0.14 * np.clip(np.sin((ti - 78) / 132.0 * np.pi), 0, None)
    net = (load - pv + rng.normal(0, 0.002, ti.size)) / 1000.0     # MWh
    return np.clip(net, 0, None) * 1.047, np.clip(-net, 0, None) * 0.998


# ── Calculator against AEMO's workbook ────────────────────────────────────────

_LO_STATES = {'NSW': 0, 'QLD': 21, 'SA': 42, 'VIC': 84}    # ParticipantData row offsets
_LO_CASES = {'debit': 1.0, 'credit': 6.0}                  # PV scale; 'credit' is net credit


@pytest.fixture(scope='module')
def aemo_recalculated(tmp_path_factory):
    """AEMO's three calculators with household profiles entered for four
    regions, recalculated by LibreOffice in one call. Returns
    {(season, case): (workbook values, {state: (load, gen)})}."""
    soffice = shutil.which('soffice') or shutil.which('libreoffice')
    if not (_HAVE_REF and soffice):
        pytest.skip('needs the AEMO calculators and LibreOffice')
    import openpyxl
    d = tmp_path_factory.mktemp('aemo')
    inputs, files = {}, []
    for season, fname in pa.CALCULATOR_FILES.items():
        for case, solar in _LO_CASES.items():
            wb = openpyxl.load_workbook(REF / fname)
            ws = wb['ParticipantData']
            profiles = {}
            for k, (state, off) in enumerate(_LO_STATES.items()):
                load, gen = _household_profiles(seed=k, solar=solar)
                load, gen = load * 50_000, gen * 50_000
                for i in range(pa.TI_PER_DAY):
                    for r, v in ((8, load[i]), (9, load[i]), (15, gen[i]), (16, gen[i])):
                        ws.cell(r + off, 18 + i).value = float(v)
                ws.cell(8 + off, 9).value = float(load.sum())
                ws.cell(15 + off, 9).value = float(gen.sum())
                profiles[state] = (load, gen)
            path = d / f'{season}_{case}.xlsx'
            wb.save(path)
            files.append(str(path))
            inputs[(season, case)] = profiles
    out = d / 'out'
    subprocess.run([soffice, '--headless', '--convert-to', 'xlsx:Calc MS Excel 2007 XML',
                    '--outdir', str(out)] + files, check=True, capture_output=True, timeout=300)
    res = {}
    for (season, case), profiles in inputs.items():
        wb = openpyxl.load_workbook(out / f'{season}_{case}.xlsx', data_only=True)
        res[(season, case)] = (wb, profiles)
    return res


@pytest.mark.slow
@needs_ref
@pytest.mark.parametrize('season', pa.SEASONS)
@pytest.mark.parametrize('case', list(_LO_CASES))
def test_calculator_matches_aemo_workbook(aemo_recalculated, params, season, case):
    """Per region: PRAFs, OSL (MCL!D41..) and unfloored PM (MCL!N40..) equal
    AEMO's workbook to $0.01; the total MCL (MCL!J11) equals the national
    assembly of the four regions."""
    wb, profiles = aemo_recalculated[(season, case)]
    m, p = wb['MCL'], wb['ParticipantData']
    cols = dict(NSW='D', QLD='E', SA='F', VIC='H')
    pm_cols = dict(NSW='N', QLD='O', SA='P', VIC='R')
    osl_sum = pm_sum = 0.0
    for state, off in _LO_STATES.items():
        load, gen = profiles[state]
        r = pa.mcl_calculator(load, gen, params[(state, season)])
        got = dict(prafl=p.cell(9 + off, 12).value, prafg=p.cell(12 + off, 12).value,
                   osl=m[f'{cols[state]}41'].value, pm_raw=m[f'{pm_cols[state]}40'].value)
        assert r['prafl'] == pytest.approx(got['prafl'], abs=1e-9), (state, r, got)
        assert r['prafg'] == pytest.approx(got['prafg'], abs=1e-9), (state, r, got)
        assert r['osl'] == pytest.approx(got['osl'], abs=0.01), (state, r, got)
        assert r['pm_raw'] == pytest.approx(got['pm_raw'], abs=0.01), (state, r, got)
        osl_sum += r['osl']
        pm_sum += r['pm_raw']
    if case == 'credit':        # non-vacuity: the net-credit branch is exercised
        assert osl_sum < 0, osl_sum
    assert pa.assemble_mcl(osl_sum, pm_sum) == pytest.approx(m['J11'].value, abs=0.05)


def test_acil_allen_table_3_14():
    """ACIL Allen's Ausgrid 2026-27 figures (DMO 2026-27 final, Table 3.14,
    p. 80) from its own inputs, 1 MWh/day of load: seasonal OSL, PML and MCL to
    $1 and the time-weighted average MCL of $10,148."""
    acil = {'summer': (97.09, 1.0014, 1.46, 2.95, 5465, 2209, 7674),
            'winter': (157.80, 1.0520, 1.62, 2.95, 10354, 3771, 14125),
            'shoulder': (98.27, 0.8886, 1.57, 2.19, 5278, 1472, 6750)}
    mcls = {}
    for season, (p, praf, vfo, vfp, osl, pml, mcl) in acil.items():
        v = pa.calculator_values(1.0, 0.0, praf, pa.PRAF_DEFAULT_GEN, p, vfo, vfp)
        assert v['osl'] == pytest.approx(osl, abs=1.0), (season, v)
        assert v['pm_raw'] == pytest.approx(pml, abs=1.0), (season, v)
        mcls[season] = pa.assemble_mcl(v['osl'], v['pm_raw'])
        assert mcls[season] == pytest.approx(mcl, abs=1.0)
    avg = sum(pa.SEASON_DAYS[s] * mcls[s] for s in pa.SEASONS) / sum(pa.SEASON_DAYS.values())
    assert avg == pytest.approx(10148, abs=1.0)


def test_calculator_net_credit_divides_by_vf():
    """Net credit: OSL and PM are the larger, less negative forms, divided by
    their volatility factors (MCL!D40, N40)."""
    v = pa.calculator_values(0.0, 1.0, 1.0, 0.8, 100.0, 1.5, 3.0)
    net = -0.8 * 100.0 * 1.1
    assert v['osl'] == pytest.approx(35 * net)           # VFOSL cancels
    assert v['pm_raw'] == pytest.approx(7 * net)
    v = pa.calculator_values(1.0, 0.0, 1.0, 0.95, 100.0, 1.5, 3.0)
    assert v['osl'] == pytest.approx(35 * 110.0 * 1.5)


# ── PRAF and the CLP formula ──────────────────────────────────────────────────

def test_praf_default_rounding_and_square():
    rp = _rp(rlwp=100.0, phh=np.full(pa.TI_PER_DAY, 100.0))
    assert pa.calculator_praf(np.zeros(pa.TI_PER_DAY), rp, 1.05) == 1.05
    flat = np.ones(pa.TI_PER_DAY)
    assert pa.calculator_praf(flat, rp, 1.05) == 1.0
    hi = _rp(rlwp=100.0, phh=np.full(pa.TI_PER_DAY, 120.0))
    assert pa.calculator_praf(flat, hi, 1.05) == 1.44          # LWPR 1.2 squared
    lo = _rp(rlwp=100.0, phh=np.full(pa.TI_PER_DAY, 80.0))
    assert pa.calculator_praf(flat, lo, 0.95) == 0.8            # LWPR 0.8, not squared
    # Excel rounds half away from zero: LWPR = 100.5/100 = 1.005 -> 1.01 (Python: 1.0)
    half = _rp(rlwp=100.0, phh=np.full(pa.TI_PER_DAY, 100.5))
    assert pa.calculator_praf(flat, half, 1.05) == 1.02         # ROUND(1.01^2, 2)
    assert pa.xl_round(2.675, 2) == 2.68 and pa.xl_round(-0.125, 2) == -0.13


@needs_ref
def test_regional_load_shape_has_praf_one(params):
    """A participant shaped like the regional load has a PRAF of 1 and is
    valued at PR (by construction of RLWPR); every region and season."""
    assert set(params) == {(s, se) for s in pa.RRP_COL for se in pa.SEASONS}
    for key, rp in params.items():
        assert pa.calculator_praf(rp.erl, rp, 1.05) == 1.0, key


@needs_ref
def test_regional_params_parsed(params):
    """Published 2026 values (AEMO NEM Regional Volatility and Price page)."""
    assert (params[('NSW', 'summer')].p, params[('NSW', 'summer')].vf_osl,
            params[('NSW', 'summer')].vf_pm) == (78.32, 1.46, 2.95)
    assert (params[('SA', 'winter')].p, params[('SA', 'winter')].vf_osl,
            params[('SA', 'winter')].vf_pm) == (109.70, 1.79, 3.46)
    assert (params[('VIC', 'shoulder')].p, params[('VIC', 'shoulder')].rlwp) == (46.88, 48.04)


def test_tod_segments_cover_the_day():
    seg = pa.tod_segment_of_ti()
    counts = np.bincount(seg)
    assert counts.tolist() == [72, 48, 72, 48, 48]
    assert seg[71] == 0 and seg[72] == 1 and seg[191] == 2 and seg[192] == 3 and seg[240] == 4


def test_clp_flat_profile_and_hand_case():
    """A flat profile is valued at PR x mean(PTI)/RLWPR; a one-segment case is
    checked by hand, including the net-credit branch (/VF) and TOSL 21."""
    rp = _rp()
    flat = np.full(pa.TI_PER_DAY, 0.02 / pa.TI_PER_DAY)
    r = pa.mcl_clp(flat, np.zeros(pa.TI_PER_DAY), rp)
    v = 0.02 * rp.p * rp.phh.mean() / rp.rlwp * 1.1
    assert r['osl'] == pytest.approx(21 * v * rp.vf_osl)
    assert r['pm_raw'] == pytest.approx(7 * v * rp.vf_pm)

    seg = pa.tod_segment_of_ti()
    gen = np.where(seg == 2, 0.03 / 72, 0.0)                     # middle of day only
    load = np.where(seg == 3, 0.01 / 48, 0.0)                    # afternoon peak only
    p_tod = pa.tod_prices(rp)
    val = (0.01 * p_tod[3] - 0.03 * p_tod[2]) * 1.1
    assert val < 0
    r = pa.mcl_clp(load, gen, rp)
    assert r['osl'] == pytest.approx(21 * val)                   # VF cancels on credit
    assert r['pm_raw'] == pytest.approx(7 * val)
    assert pa.assemble_mcl(r['osl'], r['pm_raw']) == 0.0


@pytest.mark.parametrize('osl, pm_raw, want', [
    (1234.0, 500.0, (2000.0, 1000.0, 10_000.0)),
    (-5500.0, 3000.0, (-3000.0, 3000.0, 0.0)),          # OSL no less than -PM
    (-500.0, -2000.0, (0.0, 0.0, 0.0)),                 # PM floored at 0
    (180_000.0, 70_000.0, (180_000.0, 70_000.0, 250_000.0)),
    (180_000.0, 70_000.5, (180_000.0, 71_000.0, 300_000.0)),
    (12_000.0, 1.0, (12_000.0, 1000.0, 20_000.0)),
])
def test_clp_round(osl, pm_raw, want):
    assert pa.clp_round(osl, pm_raw) == want


def test_assemble_mcl_floors():
    assert pa.assemble_mcl(100.0, -20.0) == 100.0       # PM floored before adding
    assert pa.assemble_mcl(-100.0, 30.0) == 0.0
    assert pa.assemble_mcl(-10.0, 30.0) == 20.0


# ── Cohort energy ─────────────────────────────────────────────────────────────

def _year_index(year=2023):
    return pd.date_range(f'{year}-01-01 00:05', f'{year + 1}-01-01 00:00', freq='5min',
                         tz='Etc/GMT-10', name='datetime_aest')


def test_split_energy_signs_and_loss_factors():
    net = pd.Series([1.0, -2.0, 0.0, -0.0001])
    d, c = pa.split_energy(net, 1.05, 0.99)
    assert d.tolist() == pytest.approx([1.05, 0.0, 0.0, 0.0])
    assert c.tolist() == pytest.approx([0.0, 1.98, 0.0, 0.000099])


def test_season_ti_sums_alignment_and_gaps():
    idx = _year_index()
    start = idx - pd.Timedelta(minutes=5)
    e = pd.Series(((start.hour * 60 + start.minute) // 5).astype(float) * 1000.0, index=idx)
    out = pa.season_ti_sums(e)
    assert {s: n for s, (_, n) in out.items()} == {'summer': 121, 'winter': 153,
                                                   'shoulder': 91}
    for s, (sums, n) in out.items():
        assert sums.tolist() == pytest.approx((np.arange(288) * n).tolist()), s
    # 24:00 on 31 Mar belongs to the last TI of a summer day
    end_mar = idx[idx.get_loc(pd.Timestamp('2023-04-01 00:00', tz='Etc/GMT-10'))]
    assert pa.interval_start(pd.DatetimeIndex([end_mar]))[0].month == 3
    with pytest.raises(pa.CheckError):
        pa.season_ti_sums(e.drop(e.index[5000]))


# ── Simulated outstandings ────────────────────────────────────────────────────

def test_outstandings_calendar():
    """$1 per day from Sunday 1 Jan 2023: weeks end Saturday and are paid 12
    days later (Thursday). The first valid day is the payment day of the week
    before the data; outstanding then cycles 12..18 days."""
    days = pd.date_range('2023-01-01', periods=70, freq='D')
    assert days[0].dayofweek == 6
    o = pa.outstandings(pd.Series(1.0, index=days))
    assert o.index[0] == pd.Timestamp('2023-01-12')       # Sat 31 Dec + 12
    # Thu 12 Jan: nothing paid yet from data weeks -> 12 days outstanding
    assert o.loc['2023-01-12'] == 12
    assert o.loc['2023-01-18'] == 18                     # Wed before 1st payment
    assert o.loc['2023-01-19'] == 12                     # week 1 paid Thu 19 Jan
    assert o.max() == 18 and o.min() == 12
    # mid-week start: the first, partial week must be paid before a day counts
    o2 = pa.outstandings(pd.Series(1.0, index=days[3:]))
    assert o2.index[0] == pd.Timestamp('2023-01-19')
    assert o2.loc['2023-01-19'] == 12
    with pytest.raises(pa.CheckError):
        pa.outstandings(pd.Series(1.0, index=days.delete(10)))


def test_daily_settlement_gst_and_day():
    idx = pd.date_range('2023-01-01 00:05', periods=2 * 288, freq='5min', tz='Etc/GMT-10')
    debit = pd.Series(1.0, index=idx)
    credit = pd.Series(0.0, index=idx)
    credit.iloc[288:] = 3.0
    rrp = pd.Series(100.0, index=idx)
    s = pa.daily_settlement(debit, credit, rrp)
    assert s.index.tolist() == [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-02')]
    assert s.tolist() == pytest.approx([288 * 0.1 * 1.1, -2 * 288 * 0.1 * 1.1])


def test_chained_outstandings_across_years_and_gaps():
    """Two calendar years joined: January of the second year is covered and
    equals the single-series result; a missing year starts a new run."""
    y1 = pd.Series(1.0, index=pd.date_range('2022-01-01', '2022-12-31', freq='D'))
    y2 = pd.Series(2.0, index=pd.date_range('2023-01-01', '2023-12-31', freq='D'))
    o = pa.chained_outstandings(pd.concat([y2, y1]))
    whole = pa.outstandings(pd.concat([y1, y2]))
    pd.testing.assert_series_equal(o, whole, check_freq=False)
    assert pd.Timestamp('2023-01-03') in o.index
    y4 = pd.Series(1.0, index=pd.date_range('2025-01-01', '2025-12-31', freq='D'))
    o2 = pa.chained_outstandings(pd.concat([y1, y4]))
    assert o2.index.min() == pa.outstandings(y1).index.min()
    assert o2[o2.index.year == 2025].index.min() == pa.outstandings(y4).index.min()
    with pytest.raises(pa.CheckError):
        pa.chained_outstandings(pd.concat([y1, y1.iloc[:3]]))


def _year_rows(mcl, pm_raw, year='2023', state='NSW', cohort='c'):
    return pd.DataFrame([dict(cohort=cohort, state=state, year=year, season=se,
                              method=m, mcl=mcl[se], pm_raw=pm_raw[se], osl=mcl[se] - pm_raw[se],
                              el=0.02, eg=0.0)
                         for se in pa.SEASONS for m in pa.METHODS])


def test_credit_support_daily_and_table():
    """Required = max(MCL, PM + outstanding) per day; the table's peak,
    binding constraint, retailer amount, annual weights and mean over years."""
    days = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    season_of = {m: s for s, ms in pa.SEASON_MONTHS.items() for m in ms}
    out = np.where(days.month == 6, 300.0, 10.0)            # a June spike
    out_days = pd.DataFrame({'cohort': 'c', 'state': 'NSW', 'date': days,
                             'season': [season_of[m] for m in days.month],
                             'outstanding': out})
    yr = _year_rows({'summer': 100.0, 'winter': 200.0, 'shoulder': 0.0},
                    {'summer': 40.0, 'winter': 50.0, 'shoulder': -5.0})
    yr.loc[yr['method'] == 'clp_v10', 'mcl'] = 9999.0       # ignored
    cs = pa.credit_support_daily(out_days, yr)
    jun = cs[cs['date'] == pd.Timestamp('2023-06-15')].iloc[0]
    assert (jun['pm'], jun['required']) == (50.0, 350.0)
    jan = cs[cs['date'] == pd.Timestamp('2023-01-15')].iloc[0]
    assert jan['required'] == 100.0                          # MCL binds
    t = pa.credit_support_table(cs, 50_000, 0.025)
    w = t[(t.season == 'winter') & (t.year == '2023')].iloc[0]
    assert (w['peak_outstanding'], w['required'], w['binding']) == (300.0, 350.0,
                                                                  'trading_limit')
    assert w['mcl_retailer'] == 10_000_000
    assert w['required_retailer'] == 17_500_000             # PM 2.5m + 50,000 x 300
    su = t[(t.season == 'summer') & (t.year == '2023')].iloc[0]
    assert (su['binding'], su['required_retailer']) == ('mcl', 5_000_000)
    sh = t[(t.season == 'shoulder') & (t.year == '2023')].iloc[0]
    assert (sh['pm'], sh['required'], sh['required_retailer']) == (0.0, 10.0, 500_000)
    a = t[(t.season == 'annual') & (t.year == '2023')].iloc[0]
    want = (121 * 5e6 + 153 * 17.5e6 + 91 * 5e5) / 365
    assert a['required_retailer'] == pytest.approx(want)
    assert a['guarantee_cost_hh'] == pytest.approx(0.025 * want / 50_000)
    m = t[(t.season == 'annual') & (t.year == 'mean')].iloc[0]
    assert m['required_retailer'] == pytest.approx(want)
    # a second year halves the spike: the mean averages the annual rows
    days2 = days + pd.DateOffset(years=1)
    od2 = out_days.assign(date=days2, outstanding=np.where(days2.month == 6, 150.0, 10.0))
    cs2 = pa.credit_support_daily(pd.concat([out_days, od2]),
                                  pd.concat([yr, yr.assign(year='2024')]))
    t2 = pa.credit_support_table(cs2, 50_000, 0.025)
    a24 = t2[(t2.season == 'annual') & (t2.year == '2024')].iloc[0]['required']
    a23 = t2[(t2.season == 'annual') & (t2.year == '2023')].iloc[0]['required']
    m2 = t2[(t2.season == 'annual') & (t2.year == 'mean')].iloc[0]['required']
    assert a24 < a23 and m2 == pytest.approx((a23 + a24) / 2)
    mx = t2[t2.season == 'max'].iloc[0]
    assert (mx['mcl'], mx['required'], mx['above_mcl'], mx['peak_when']) == (
        200.0, 350.0, 150.0, 'winter 2023')
    assert mx['required_retailer'] == 17_500_000 and mx['mcl_retailer'] == 10_000_000
    assert mx['guarantee_cost_retailer'] == pytest.approx(0.025 * 17_500_000)
    # the largest MCL can come from a different season than the largest requirement
    cs3 = cs2.copy()
    cs3.loc[(cs3['year'] == 2024) & (cs3['season'] == 'summer'), 'mcl'] = 500.0
    cs3['required'] = np.maximum(cs3['mcl'], cs3['pm'] + cs3['outstanding'])
    mx3 = pa.credit_support_table(cs3, 50_000, 0.025).query("season == 'max'").iloc[0]
    assert (mx3['mcl'], mx3['required'], mx3['above_mcl']) == (500.0, 500.0, 0.0)
    with pytest.raises(pa.CheckError):
        pa.credit_support_daily(od2, yr)                    # no 2024 MCL
    clp = pa.credit_support_daily(out_days, yr, 'clp_v10')
    assert (clp['mcl'] == 9999.0).all() and (clp['method'] == 'clp_v10').all()


def test_national_rows_by_year():
    rows = pd.concat([_year_rows({s: 100.0 for s in pa.SEASONS}, {s: 30.0 for s in pa.SEASONS},
                                 year=y, state=st)
                      for y in ('2022', '2023') for st in ('NSW', 'SA')], ignore_index=True)
    rows.loc[(rows.year == '2023') & (rows.state == 'SA'), 'osl'] = 170.0
    n = pa.national_rows(rows, {'NSW': 0.5, 'SA': 0.5})
    assert len(n) == 2 * len(pa.SEASONS) * len(pa.METHODS)
    assert set(n['year']) == {'2022', '2023'}
    assert n[n.year == '2023']['osl'].tolist() == pytest.approx([120.0] * 6)
    assert n[n.year == '2022']['osl'].tolist() == pytest.approx([70.0] * 6)


def test_cohort_scenarios_and_labels():
    sc = pa.cohort_scenarios()
    assert set(sc) == set(pa.COHORTS)
    assert pa._cohort_name(sc['self_consumption'], 20) == 'Self-consumption, 20 kWh'
    assert pa._name(sc['coordinated'], 13) == 'Amber, 13 kWh'
    assert pa._name(sc['no_cer'], 20) == 'No CER (load only)'


def test_round_up_mcl():
    assert [pa.round_up_mcl(v) for v in (0.0, 1.0, 250_000.0, 250_001.0)] == [
        0.0, 10_000.0, 250_000.0, 300_000.0]


# ── Aggregation ───────────────────────────────────────────────────────────────

def test_national_sums_pm_before_floor():
    rows = pd.DataFrame([
        dict(cohort='c', state='NSW', season='summer', method='calculator', el=1.0, eg=0.0,
             osl=100.0, pm_raw=40.0, mcl=140.0),
        dict(cohort='c', state='SA', season='summer', method='calculator', el=0.0, eg=1.0,
             osl=-150.0, pm_raw=-60.0, mcl=0.0),
    ])
    n = pa.national_rows(rows, {'NSW': 0.5, 'SA': 0.5, 'QLD': 0.3})
    r = n.iloc[0]
    assert (r['osl'], r['pm_raw'], r['mcl']) == (-25.0, -10.0, 0.0)
    assert r['states'] == 'NSW,SA'
    n2 = pa.national_rows(rows.assign(pm_raw=[40.0, -20.0]), {'NSW': 0.5, 'SA': 0.5})
    assert n2.iloc[0]['pm_raw'] == 10.0 and n2.iloc[0]['mcl'] == 0.0   # max(0, -25 + 10)


def test_retailer_table_rounding_annual_and_cost():
    per_hh = pd.DataFrame([dict(cohort='c', state='NSW', season=s, method='calculator',
                                el=0.02, eg=0.0, prafl=1.0, prafg=0.95, osl=o, pm_raw=p,
                                mcl=o + p)
                           for s, o, p in (('summer', 100.0, 40.0), ('winter', 200.0, 60.0),
                                           ('shoulder', 50.0, 20.0))])
    t = pa.retailer_table(per_hh, 50_000, 0.025)
    s = t.set_index('season')
    assert s.loc['summer', 'mcl_retailer'] == 7_000_000      # 5.0m + 2.0m
    assert s.loc['winter', 'mcl_retailer'] == 13_000_000
    a = s.loc['annual']
    want = (121 * 7_000_000 + 153 * 13_000_000 + 91 * 3_500_000) / 365
    assert a['mcl_retailer'] == pytest.approx(want)
    assert a['mcl'] == pytest.approx((121 * 140 + 153 * 260 + 91 * 70) / 365)
    assert a['guarantee_cost_retailer'] == pytest.approx(0.025 * want)
    assert a['guarantee_cost_hh'] == pytest.approx(0.025 * want / 50_000)
    assert a['el_kwh_day'] == pytest.approx(20.0)


# ── End to end ────────────────────────────────────────────────────────────────

@needs_ref
def test_main_end_to_end(tmp_path, params):
    """Two sites, unweighted, one NSW year, all four cohorts: the summary's
    per-household No CER calculator MCL equals the formula on the hand-built
    mean profile with the run's loss factors, and the outstandings file
    covers every cohort and season."""
    idx = _year_index()
    rng = np.random.default_rng(3)
    start = idx - pd.Timedelta(minutes=5)
    tod = (start.hour * 60 + start.minute).to_numpy() / 60.0
    load = 0.03 + 0.04 * np.exp(-((tod - 18.5) / 1.5) ** 2)
    pv = 0.12 * np.clip(np.sin((tod - 6.5) / 11.0 * np.pi), 0, None)
    runs = tmp_path / 'data' / 'output' / 'runs'
    flows_by_run = {}
    for ckey, (_, run, has_bat) in pa.COHORTS.items():
        sid = 'NSW_2023' + ('_20kWh' if has_bat else '')
        solar = 0.0 if ckey == 'no_cer' else 1.0
        f = pd.DataFrame({s: load * k - solar * pv + rng.normal(0, 0.003, len(idx))
                          for s, k in (('S1', 1.0), ('S2', 1.6))}, index=idx)
        d = runs / run / 'scenarios' / sid
        d.mkdir(parents=True)
        f.to_parquet(d / 'flows.parquet')
        pd.DataFrame({'site_id': ['S1', 'S2']}).to_csv(d / 'bills.csv', index=False)
        flows_by_run[ckey] = f
    prices = tmp_path / 'data' / 'clean' / 'nem_prices'
    prices.mkdir(parents=True)
    pdf = pd.DataFrame({c: 80.0 + 40.0 * np.sin(tod / 24 * 2 * np.pi) for c in pa.RRP_COL.values()},
                       index=idx)
    pdf.index.name = 'datetime_aest'
    pdf.reset_index().to_parquet(prices / '2023_nem_prices.parquet')
    out = tmp_path / 'out'
    assert pa.main(['--data-root', str(tmp_path), '--unweighted', '--output-dir', str(out)]) == 0

    s = pd.read_csv(out / 'prudential_summary.csv')
    from model.core.tariff import load_network_tariff
    _, net_ids = pa.run_tariff_ids(runs, 'full_run_no_cer_amber')
    nt = load_network_tariff(net_ids['NSW']['2023'], pa.TARIFF_DIR)
    debit, credit = pa.split_energy(flows_by_run['no_cer'].mean(axis=1),
                                    nt.import_loss_factor, nt.export_loss_factor)
    sums = pa.season_ti_sums(debit)
    for season in pa.SEASONS:
        prof, n = sums[season]
        want = pa.mcl_calculator(prof / n, np.zeros(288), params[('NSW', season)])
        row = s[(s.cohort == 'no_cer') & (s.state == 'NSW') & (s.season == season)
                & (s.method == 'calculator')].iloc[0]
        assert row['mcl'] == pytest.approx(pa.assemble_mcl(want['osl'], want['pm_raw']))
        assert row['mcl'] > 0
        assert row['mcl_retailer'] == pa.clp_round(50_000 * want['osl'],
                                                   50_000 * want['pm_raw'])[2]
    nat = s[(s.state == 'National') & (s.season == 'annual')]
    assert len(nat) == 2 * len(pa.COHORTS)
    o = pd.read_csv(out / 'prudential_outstandings.csv')
    assert set(zip(o.cohort, o.season)) >= {(c, se) for c in pa.COHORTS for se in pa.SEASONS}
    cs = pd.read_csv(out / 'prudential_credit_support.csv', dtype={'year': str})
    nsw = cs[(cs.state == 'NSW') & (cs.cohort == 'no_cer') & (cs.year == '2023')
             & (cs.season != 'annual') & (cs.method == 'calculator')]
    for _, r in nsw.iterrows():
        row = s[(s.cohort == 'no_cer') & (s.state == 'NSW') & (s.season == r['season'])
                & (s.method == 'calculator')].iloc[0]
        assert r['mcl'] == pytest.approx(row['mcl'])        # one year: pooled = year
        assert r['required'] >= r['mcl']
    assert set(cs[cs.season == 'annual']['year']) == {'2023', 'mean'}
    mx = cs[cs.season == 'max']
    assert len(mx) == 2 * len(pa.COHORTS) * len(pa.METHODS)
    assert (mx['required'] >= mx['mcl']).all() and set(mx['method']) == set(pa.METHODS)
    assert set(cs['state']) == {'NSW', 'National'}
    for name in ('prudential_credit_support_NSW.png', 'prudential_credit_support_National.png',
                 'prudential_credit_support_max.png',
                 'prudential_credit_support_max_clp_v10.png'):
        assert (out / name).exists(), name
