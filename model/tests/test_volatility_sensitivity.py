"""Tests for analysis/scripts/volatility_sensitivity.py.

See docs/volatility_sensitivity_spec.md. The script lives with the analysis
scripts, not in a package, so it is loaded by file path.
"""
import importlib.util
import math
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    'volatility_sensitivity',
    REPO_ROOT / 'analysis' / 'scripts' / 'volatility_sensitivity.py')
vs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vs)

TZ = 'Etc/GMT-10'
FIVE = pd.Timedelta(minutes=5)


def close(a, b, tol=1e-9):
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def _prices(start, end, levels=None, base=50.0):
    """Interval-end AEST RRP covering the AEST dates start..end. The
    intervals starting in hour 18 take the level for their month (default
    base). QLD local time is AEST, so these are QLD local days."""
    idx = pd.date_range(pd.Timestamp(start, tz=TZ) + FIVE,
                        pd.Timestamp(end, tz=TZ) + pd.Timedelta(days=1),
                        freq='5min')
    s = pd.Series(base, index=idx, dtype=float)
    st = idx - FIVE
    for (year, month), level in (levels or {}).items():
        s[(st.hour == 18) & (st.year == year) & (st.month == month)] = level
    return s


def _row(vol, year, month):
    r = vol[(vol['year'] == year) & (vol['month'] == month)]
    assert len(r) == 1, f"expected one row for {year}-{month}, got {len(r)}"
    return r.iloc[0]


# ── Volatility measures ───────────────────────────────────────────────────────

def test_cap_value_and_daily_spread_known_values():
    # Each day: 12 intervals at 1000, 276 at 50. Cap value is the mean of
    # max(RRP - 300, 0) over the day's 288 intervals; the highest two hours
    # are the 12 at 1000 and 12 at 50, the lowest two hours are all 50.
    p = _prices('2023-02-01', '2023-02-28', {(2023, 2): 1000.0})
    row = _row(vs.state_month_volatility(p, 'QLD'), 2023, 2)
    assert close(row['cap_value'], 700 * 12 / 288)
    assert close(row['daily_spread'], (1000 + 50) / 2 - 50)
    assert row['n_days'] == 28 and row['n_intervals'] == 28 * 288
    assert row['days_in_month'] == 28


def test_months_and_days_are_state_local():
    # One interval ending 2023-04-01 00:00 AEST starts at 23:55 AEST on 31
    # March, which is March in QLD (AEST) and 00:55 on 1 April in NSW (AEDT).
    p = _prices('2023-03-01', '2023-04-30')
    p[pd.Timestamp('2023-04-01 00:00', tz=TZ)] = 1300.0
    qld = vs.state_month_volatility(p, 'QLD')
    nsw = vs.state_month_volatility(p, 'NSW')
    for vol, spike_month, other in ((qld, 3, 4), (nsw, 4, 3)):
        hit = _row(vol, 2023, spike_month)
        assert close(hit['cap_value'] * hit['n_intervals'], 1000.0)
        assert _row(vol, 2023, other)['cap_value'] == 0.0


def test_missing_prices_are_left_out():
    p = _prices('2023-02-01', '2023-02-28', {(2023, 2): 1000.0})
    day = p.index[(p.index > pd.Timestamp('2023-02-10 00:00', tz=TZ))
                  & (p.index <= pd.Timestamp('2023-02-10 03:20', tz=TZ))]
    p[day[:40]] = np.nan          # 40 missing: day 10 has 248 valid intervals
    row = _row(vs.state_month_volatility(p, 'QLD'), 2023, 2)
    assert row['n_days'] == 27 and row['days_in_month'] == 28
    assert row['n_intervals'] == 28 * 288 - 40
    assert close(row['daily_spread'], 475.0)
    # the 40 missing intervals leave the mean, the day's others stay in it
    assert close(row['cap_value'], 28 * 12 * 700 / (28 * 288 - 40))


def test_daily_spread_needs_enough_intervals():
    v = np.arange(vs.MIN_DAY_INTERVALS, dtype=float)
    assert not math.isnan(vs._daily_spread(v))
    assert math.isnan(vs._daily_spread(v[:-1]))
    # 276 intervals (a 23-hour DST day) pass
    assert not math.isnan(vs._daily_spread(np.arange(276.0)))


# ── Quartiles ─────────────────────────────────────────────────────────────────

def test_quartiles_and_boundary_values():
    b = vs.quartile_bounds(np.arange(1.0, 10.0))
    assert len(b) == 3 and close(b[0], 3.0) and close(b[1], 5.0) and close(b[2], 7.0)
    got = vs.assign_quartile([1, 2, 3, 4, 5, 6, 7, 8, 9], (3.5, 5.5, 7.5))
    assert list(got) == ['q1'] * 3 + ['q2'] * 2 + ['q3'] * 2 + ['q4'] * 2
    # a value on a boundary belongs to the lower group
    assert list(vs.assign_quartile([3.0, 5.0, 7.0], (3.0, 5.0, 7.0))) == [
        'q1', 'q2', 'q3']
    # months with no cap value, more than a quarter of the total, are all q1
    cap = np.array([0.0] * 5 + [1.0, 2.0, 3.0, 4.0])
    got = vs.assign_quartile(cap, vs.quartile_bounds(cap))
    assert list(got[:5]) == ['q1'] * 5 and 'q4' in set(got)


def test_tick_labels_give_the_quartile_ranges():
    assert vs.quartile_tick_labels((1.0, 5.0, 10.0)) == [
        '< 1.0', '1.0 - 5.0', '5.0 - 10.0', '> 10.0']
    assert vs.quartile_tick_labels((0.0, 0.4, 12.34))[0] == '0'


def test_volatility_table_places_current_year_against_modelled_bounds():
    levels = {(2023, m): L for m, L in
              zip(range(1, 7), (100.0, 300.0, 700.0, 1300.0, 2500.0, 5300.0))}
    levels.update({(2026, 1): 1100.0, (2026, 2): 100.0})
    p = pd.concat([_prices('2023-01-01', '2023-06-30', levels),
                   _prices('2026-01-01', '2026-02-28', levels)])
    modelled, current, bounds = vs.volatility_table({'QLD': p})
    expected = [max(L - 300, 0) * 12 / 288 for L in
                (100, 300, 700, 1300, 2500, 5300)]
    assert np.allclose(modelled['cap_value'], expected)
    assert close(bounds[0], 0.25 * expected[2])
    assert close(bounds[1], (expected[2] + expected[3]) / 2)
    assert close(bounds[2], expected[3] + 0.75 * (expected[4] - expected[3]))
    assert list(modelled['quartile']) == ['q1', 'q1', 'q2', 'q3', 'q4', 'q4']
    assert len(current) == 2 and set(current['year']) == {2026}
    jan = _row(current, 2026, 1)
    feb = _row(current, 2026, 2)
    assert jan['quartile'] == 'q3' and feb['quartile'] == 'q1'
    assert close(jan['cap_percentile'], 100 * 3 / 6)
    assert close(feb['cap_percentile'], 0.0)
    assert not current['partial'].any()


# ── Runs and monthly savings ──────────────────────────────────────────────────

def _bills(rows):
    return pd.DataFrame(rows, columns=['site_id', 'household_total',
                                       'wholesale_usage_cost',
                                       'wholesale_feedin_credit'])


def _write(root, run, scenario, month, rows):
    d = root / run / 'scenarios' / scenario
    d.mkdir(parents=True, exist_ok=True)
    _bills(rows).to_csv(d / f'bills_{month:02d}.csv', index=False)


PLAN = dict(offer='Amber', mode='SC', color='#2bffa4', excluded_states=set(),
            runs=dict(solar='r_solar', sc='r_sc', ach='r_ach', pf='r_pf',
                      amber_solar='r_amber_solar'))

# site, household_total, wholesale_usage_cost, wholesale_feedin_credit
SOLAR = [('S1', 1000, 400, -100), ('S2', 2000, 800, -200),
         ('S3', 3000, 900, -100), ('S0588', 99999, 99999, 0)]
SC = [('S1', 900, 350, -100), ('S2', 1700, 700, -150),
      ('S3', 2500, 800, -100), ('S4', 1, 1, 0), ('S0588', 1, 1, 0)]
ACH = [('S1', 800, 300, -100), ('S2', 1500, 650, -150), ('S3', 2000, 700, -100),
       ('S0588', 1, 1, 0)]
PF = [('S1', 700, 250, -100), ('S2', 1400, 600, -200), ('S3', 1900, 650, -100),
      ('S0588', 1, 1, 0)]


AMBER_SOLAR = [('S1', 1100, 450, -100), ('S2', 2100, 850, -200),
               ('S3', 3100, 950, -100), ('S0588', 99999, 99999, 0)]


def _runs(tmp_path):
    for month in (1, 2):
        _write(tmp_path, 'r_amber_solar', 'QLD_2023', month, AMBER_SOLAR)
        _write(tmp_path, 'r_solar', 'QLD_2023', month, SOLAR)
        _write(tmp_path, 'r_sc', 'QLD_2023_20kWh', month, SC)
        _write(tmp_path, 'r_ach', 'QLD_2023_20kWh', month, ACH)
        # S2 is missing from the perfect foresight run in month 2
        _write(tmp_path, 'r_pf', 'QLD_2023_20kWh', month,
               PF if month == 1 else [r for r in PF if r[0] != 'S2'])
    return tmp_path


def _modelled(months=(1, 2, 3)):
    return pd.DataFrame(dict(state='QLD', year=2023, month=list(months),
                             cap_value=[10.0, 20.0, 30.0][:len(months)],
                             daily_spread=100.0,
                             quartile=['q1', 'q2', 'q3'][:len(months)]))


def test_state_month_savings_weighting_signs_and_site_sets(tmp_path):
    root = _runs(tmp_path)
    usable, missing = vs.check_runs(root, [PLAN], [('QLD', 2023)], 20)
    assert usable == {('Amber', 'QLD', 2023): True} and not missing
    sw = {('QLD', 2023): pd.Series({'S1': 1.0, 'S2': 3.0, 'S0588': 5.0})}
    sm = vs.state_month_rows(root, [PLAN], usable, _modelled(), [('QLD', 2023)],
                             sw, 20)
    # month 3 has no bills; S3 has no weight; S4 is not in every run; S0588
    # is an excluded site
    assert list(sm['month']) == [1, 2]
    m1 = sm.iloc[0]
    assert m1['n_sites'] == 2 and m1['low_site_count']
    # savings are Solar only less the battery run, weights 1 and 3
    assert close(m1['hh_sc'], (100 + 3 * 300) / 4)
    assert close(m1['hh_ach'], (200 + 3 * 500) / 4)
    assert close(m1['hh_pf'], (300 + 3 * 600) / 4)
    # net wholesale = usage cost + feed-in credit
    assert close(m1['ws_sc'], 50.0)
    assert close(m1['ws_ach'], 100.0)
    assert close(m1['ws_pf'], (150 + 3 * 200) / 4)
    assert m1['quartile'] == 'q1' and m1['cap_value'] == 10.0
    # common baseline: Amber Solar only less the run
    assert close(m1['hh_sc_common'], (200 + 3 * 400) / 4)
    assert close(m1['hh_ach_common'], (300 + 3 * 600) / 4)
    assert close(m1['ws_sc_common'], 100.0)
    assert close(m1['ws_ach_common'], 150.0)
    # month 2: S2 drops out of every saving, leaving S1
    m2 = sm.iloc[1]
    assert m2['n_sites'] == 1 and m2['low_site_count']
    assert close(m2['hh_ach_common'], 300.0)
    assert close(m2['hh_sc'], 100.0) and close(m2['hh_pf'], 300.0)


def test_check_runs_lists_a_run_without_monthly_bills(tmp_path):
    root = _runs(tmp_path)
    for f in (root / 'r_pf' / 'scenarios' / 'QLD_2023_20kWh').glob('*.csv'):
        f.unlink()
    usable, missing = vs.check_runs(root, [PLAN], [('QLD', 2023)], 20)
    assert usable == {('Amber', 'QLD', 2023): False}
    assert len(missing) == 1
    assert 'r_pf' in missing[0] and 'QLD_2023_20kWh' in missing[0]
    sm = vs.state_month_rows(root, [PLAN], usable, _modelled(),
                             [('QLD', 2023)],
                             {('QLD', 2023): pd.Series({'S1': 1.0})}, 20)
    assert sm.empty


def test_plans_take_run_names_from_financial_analysis():
    sc = {p['offer']: p['runs']['sc'] for p in vs.offer_plans()}
    assert sc['AGL Battery Rewards'] == 'full_run_self_consumption_agl_novpp'
    assert sc['Origin Battery Lite'] == 'full_run_self_consumption_origin_novpp'
    assert sc['GloBird ZeroHero'] == 'full_run_self_consumption_globird_novpp'
    assert sc['Amber'] == 'full_run_self_consumption_amber'
    assert sc['EnergyAustralia Battery Ease'] == 'full_run_self_consumption_ea'
    for p in vs.offer_plans():
        assert '_novpp' not in p['runs']['ach'] and '_novpp' not in p['runs']['pf']
    plans = vs.offer_plans()
    assert [p['offer'] for p in plans][0] == 'Amber'
    ach = {p['sc_mode']: p for p in vs.fa.plans_for_basis('achievable')}
    pf = {p['sc_mode']: p for p in vs.fa.plans_for_basis('perfect_foresight')}
    for p in plans:
        assert p['runs']['sc'] == ach[p['mode']]['sc_no_vpp_dir']
        assert p['runs']['ach'] == ach[p['mode']]['lp_dir']
        assert p['runs']['pf'] == pf[p['mode']]['lp_dir']
    amber = plans[0]
    assert amber['runs']['solar'] == vs.fa.DEFAULT_BASELINE_RUN
    for p in plans:
        assert p['runs']['amber_solar'] == vs.fa.DEFAULT_BASELINE_RUN


# ── Group metrics ─────────────────────────────────────────────────────────────

def _sm_row(offer, quartile, sc, ach, pf, ws=(0.0, 0.0, 0.0), low=False):
    return dict(state='QLD', year=2023, month=1, offer=offer,
                cap_value={'q1': 1.0, 'q2': 10.0, 'q3': 30.0, 'q4': 100.0}[quartile],
                daily_spread=50.0, quartile=quartile, n_sites=25,
                low_site_count=low, hh_sc=sc, hh_ach=ach, hh_pf=pf,
                ws_sc=ws[0], ws_ach=ws[1], ws_pf=ws[2],
                hh_sc_common=sc + 10, hh_ach_common=ach + 10,
                ws_sc_common=ws[0] - 1, ws_ach_common=ws[1] - 1)


BOUNDS = (5.0, 20.0, 50.0)


def _group_fixture():
    rows = [
        # q1: one month with self-consumption near zero
        _sm_row('Amber', 'q1', 0.01, 5.0, 10.0, ws=(0.01, 2.0, 4.0)),
        _sm_row('Amber', 'q1', 100.0, 150.0, 200.0, ws=(50.0, 60.0, 90.0),
                low=True),
        _sm_row('Amber', 'q2', 100.0, 200.0, 400.0, ws=(40.0, 80.0, 120.0)),
        _sm_row('Amber', 'q3', 150.0, 300.0, 500.0, ws=(50.0, 100.0, 150.0)),
        _sm_row('Amber', 'q4', 200.0, 800.0, 1000.0, ws=(60.0, 360.0, 460.0)),
        # a ratio against a reference that is not positive
        _sm_row('Neg', 'q1', -5.0, 1.0, 2.0, ws=(-5.0, 1.0, 2.0)),
        _sm_row('Neg', 'q2', 10.0, 20.0, 30.0, ws=(10.0, 20.0, 30.0)),
        _sm_row('Neg', 'q3', 10.0, 20.0, 30.0, ws=(10.0, 20.0, 30.0)),
        _sm_row('Neg', 'q4', 10.0, 20.0, 30.0, ws=(10.0, 20.0, 30.0)),
    ]
    return pd.DataFrame(rows)


def test_group_metrics_are_ratios_of_sums():
    g = vs.group_metrics(_group_fixture(), BOUNDS).set_index(
        ['offer', 'quartile'])
    assert len(g) == 8
    q1 = g.loc[('Amber', 'q1')]
    assert q1['n_state_months'] == 2 and q1['n_low_site_count'] == 1
    assert close(q1['sum_hh_sc'], 100.01) and close(q1['sum_hh_ach'], 155.0)
    assert close(q1['rel_value_hh'], 155.0 / 100.01)
    assert close(q1['capture_hh'], (155.0 - 100.01) / (210.0 - 100.01))
    assert close(q1['rel_value_ws'], 62.0 / 50.01)
    assert close(q1['capture_ws'], (62.0 - 50.01) / (94.0 - 50.01))
    # the mean of the monthly ratios is a different (and unstable) number
    mean_of_ratios = np.mean([5.0 / 0.01, 150.0 / 100.0])
    assert abs(q1['rel_value_hh'] - mean_of_ratios) > 100
    # shares of the offer's total coordination value
    total = (155 - 100.01) + 100 + 150 + 600
    assert close(g.loc[('Amber', 'q4'), 'share_coord_hh'], 600 / total)
    qs = ('q1', 'q2', 'q3', 'q4')
    assert close(sum(g.loc[('Amber', q), 'share_coord_hh'] for q in qs), 1.0)
    assert close(sum(g.loc[('Amber', q), 'share_coord_ws'] for q in qs), 1.0)
    # mean annual saving against the common baseline: 12 x the monthly mean
    assert close(q1['ann_hh_sc_common'], 12 * (10.01 + 110.0) / 2)
    assert close(q1['ann_hh_ach_common'], 12 * (15.0 + 160.0) / 2)
    assert close(q1['ann_ws_ach_common'], 12 * (1.0 + 59.0) / 2)
    # quartile edges
    assert g.loc[('Amber', 'q1'), 'cap_lower'] == -np.inf
    assert g.loc[('Amber', 'q1'), 'cap_upper'] == BOUNDS[0]
    assert g.loc[('Amber', 'q2'), 'cap_lower'] == BOUNDS[0]
    assert g.loc[('Amber', 'q3'), 'cap_upper'] == BOUNDS[2]
    assert g.loc[('Amber', 'q4'), 'cap_lower'] == BOUNDS[2]
    assert g.loc[('Amber', 'q4'), 'cap_upper'] == np.inf


def test_ratio_is_nan_without_a_positive_reference():
    g = vs.group_metrics(_group_fixture(), BOUNDS).set_index(
        ['offer', 'quartile'])
    assert math.isnan(g.loc[('Neg', 'q1'), 'rel_value_hh'])
    assert not math.isnan(g.loc[('Neg', 'q2'), 'rel_value_hh'])
    assert math.isnan(vs.ratio(1.0, 0.0)) and math.isnan(vs.ratio(1.0, -2.0))


# ── Charts ────────────────────────────────────────────────────────────────────

def _png_size(path):
    img = mpimg.imread(path)
    return img.shape[1], img.shape[0]


def test_charts_are_drawn_at_thesis_size(tmp_path):
    sm = _group_fixture()
    group = vs.group_metrics(sm, BOUNDS)
    plans = [dict(offer='Amber', color='#2bffa4'),
             dict(offer='Neg', color='#e5037f')]
    width = round(vs.ts.TEXT_WIDTH_IN * vs.ts.DPI)
    p1 = tmp_path / 'by_volatility'
    vs.draw_by_volatility(group, plans, BOUNDS, 20, p1, vs.notes_text(sm, 'own'))
    w, h = _png_size(p1.with_suffix('.png'))
    assert w == width and h == round(vs.BY_VOLATILITY_HEIGHT_IN * vs.ts.DPI)

    p2 = tmp_path / 'saving'
    vs.draw_saving_common(group, plans, BOUNDS, 20, p2,
                           vs.notes_text(sm, 'amber'))
    w, h = _png_size(p2.with_suffix('.png'))
    assert w == width and h == round(vs.SAVING_HEIGHT_IN * vs.ts.DPI)

    # a state-month with no cap value (log axis) and 2026 state-months
    sm.loc[0, 'cap_value'] = 0.0
    current = pd.DataFrame(dict(state=['QLD', 'QLD', 'NSW'],
                                cap_value=[0.0, 20.0, 400.0]))
    p3 = tmp_path / 'scatter'
    vs.draw_scatter(sm, current, BOUNDS, 'Amber', 20, p3,
                    vs.notes_text(sm, 'own'))
    w, h = _png_size(p3.with_suffix('.png'))
    assert w == width and h == round(vs.SCATTER_HEIGHT_IN * vs.ts.DPI)


def test_title_suffix_and_notes():
    assert vs.title_suffix(20) == ', imperfect foresight (achievable), 20 kWh'
    sm = pd.DataFrame(dict(state=['QLD', 'NSW', 'QLD'], year=[2021, 2025, 2023]))
    own = vs.notes_text(sm, 'own').split('\n')
    assert own[0] == 'National (NSW, QLD), 2021-2025' and len(own) == 3
    assert own[2] == "Savings vs each offer's own Solar only"
    assert vs.notes_text(sm, 'amber').split('\n')[2] == (
        'Savings vs Amber Solar only')
