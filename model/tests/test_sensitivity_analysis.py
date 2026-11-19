"""Tests for analysis/scripts/sensitivity/ (docs/sensitivity_runs_spec.md).

The sensitivity scripts live with the analysis scripts, not in a package, so
they are loaded by file path.

What these guarantee:
  - savings are solar-only less the run's bill, so a lower bill reads positive;
  - comparisons are paired: only sites present in every run are used, and the
    weights are renormalised over that set;
  - the VPP reward is read with its sign flipped, so a credit reads positive;
  - value capture is pooled, not a mean of per-site ratios;
  - no nudge arm carries a VPP enrolment payment, and the arms read the
    same runs the retailer briefing draws;
  - a two-panel chart never shows two views of the same quantity.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SENS_DIR = REPO_ROOT / 'analysis' / 'scripts' / 'sensitivity'


def _load(name: str):
    """Load an analysis script by file path. The sensitivity scripts and the
    scripts they compare against are not packages, so both directories go on
    the path and the file is found in whichever holds it."""
    sys.path.insert(0, str(SENS_DIR))
    sys.path.insert(0, str(SENS_DIR.parent))
    path = SENS_DIR / f'{name}.py'
    if not path.exists():
        path = SENS_DIR.parent / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sc = _load('sensitivity_common')
horizon = _load('horizon')
nudge = _load('nudge')
degradation = _load('degradation')


# ── helpers ───────────────────────────────────────────────────────────────────

def write_bills(root: Path, run: str, scenario: str, sites, household,
                net_wholesale, other_credits=0.0, margin=100.0):
    """A bills.csv with the real column set. net_wholesale is split across the
    two wholesale columns the loader adds together."""
    n = len(sites)
    d = root / 'data' / 'output' / 'runs' / run / 'scenarios' / scenario
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        'site_id': sites, 'plan_type': 'x',
        'wholesale_usage_cost': np.full(n, 0.0) + np.asarray(net_wholesale) + 10.0,
        'wholesale_feedin_credit': np.full(n, -10.0),
        'network_usage_cost': 0.0, 'network_feedin_credit': 0.0,
        'network_daily_charges': 0.0, 'retail_usage_cost': 0.0,
        'retail_feedin_credit': 0.0, 'retail_daily_charges': 0.0,
        'retail_other_credits': np.full(n, 0.0) + other_credits,
        'household_total': np.asarray(household, dtype=float),
        'retailer_margin': np.full(n, float(margin)),
    }).to_csv(d / 'bills.csv', index=False)


@pytest.fixture
def root(tmp_path):
    return tmp_path


# ── savings and bill loading ──────────────────────────────────────────────────

def test_savings_are_positive_when_the_battery_lowers_the_bill(root):
    sites = ['A', 'B']
    write_bills(root, 'solar', 'NSW_2024', sites, [2000, 2000], [1000, 1000])
    write_bills(root, 'batt', 'NSW_2024_20kWh', sites, [1500, 1700], [600, 900])

    runs = sc.runs_root(root)
    solar = sc.load_solar_only(runs, 'solar', 'NSW', '2024')
    bills = sc.load_bills(runs, 'batt', 20, 'NSW', '2024')
    out = sc.savings(bills, solar)

    assert out.loc['A', 'hh_saving'] == pytest.approx(500.0)
    assert out.loc['B', 'hh_saving'] == pytest.approx(300.0)
    assert out.loc['A', 'ws_saving'] == pytest.approx(400.0)


def test_vpp_reward_reads_positive_from_a_negative_credit(root):
    write_bills(root, 'vpp', 'NSW_2024_20kWh', ['A'], [1000], [500],
                other_credits=-55.0)
    bills = sc.load_bills(sc.runs_root(root), 'vpp', 20, 'NSW', '2024')
    assert bills.loc['A', 'vpp_reward'] == pytest.approx(55.0)


def test_excluded_sites_are_dropped(root):
    excluded = sorted(sc.EXCLUDED_SITES)[0]
    sites = ['A', excluded]
    write_bills(root, 'batt', 'NSW_2024_20kWh', sites, [1, 2], [1, 2])
    bills = sc.load_bills(sc.runs_root(root), 'batt', 20, 'NSW', '2024')
    assert excluded not in bills.index
    assert 'A' in bills.index


def test_missing_run_loads_as_none_not_an_error(root):
    assert sc.load_bills(sc.runs_root(root), 'absent', 20, 'NSW', '2024') is None


# ── pairing and weighting ─────────────────────────────────────────────────────

def test_common_index_is_the_intersection():
    a = pd.DataFrame(index=['A', 'B', 'C'])
    b = pd.DataFrame(index=['B', 'C', 'D'])
    c = pd.DataFrame(index=['C', 'D'])
    assert list(sc.common_index(a, b, c)) == ['C']


def test_wmean_renormalises_over_the_intersection():
    # The weight series covers a site the values do not. The mean must use
    # only the shared site, renormalised, not treat the absent one as zero.
    values = pd.Series({'A': 10.0, 'B': 20.0})
    weight = pd.Series({'A': 0.1, 'B': 0.1, 'C': 0.8})
    assert sc.wmean(values, weight) == pytest.approx(15.0)


def test_wmean_weights_actually_bite():
    values = pd.Series({'A': 0.0, 'B': 100.0})
    weight = pd.Series({'A': 0.75, 'B': 0.25})
    assert sc.wmean(values, weight) == pytest.approx(25.0)
    assert sc.wmean(values, None) == pytest.approx(50.0)


def test_a_subset_run_is_compared_only_on_its_own_sites(root):
    """The reference covers three sites and the sensitivity run one. The
    comparison must use the shared site, not the reference's three-site mean."""
    write_bills(root, 'solar', 'NSW_2024', ['A', 'B', 'C'],
                [2000, 2000, 2000], [1000, 1000, 1000])
    write_bills(root, 'ref', 'NSW_2024_20kWh', ['A', 'B', 'C'],
                [1500, 1000, 500], [900, 900, 900])
    write_bills(root, 'sens', 'NSW_2024_20kWh', ['A'], [1400], [850])

    runs = sc.runs_root(root)
    solar = sc.load_solar_only(runs, 'solar', 'NSW', '2024')
    ref = sc.load_bills(runs, 'ref', 20, 'NSW', '2024')
    sens = sc.load_bills(runs, 'sens', 20, 'NSW', '2024')
    idx = sc.common_index(solar, ref, sens)

    assert list(idx) == ['A']
    ref_sav = sc.wmean(sc.savings(ref.loc[idx], solar.loc[idx])['hh_saving'], None)
    sens_sav = sc.wmean(sc.savings(sens.loc[idx], solar.loc[idx])['hh_saving'], None)
    # Site A only: 500 against 600. The unpaired reference mean would be 1000.
    assert ref_sav == pytest.approx(500.0)
    assert sens_sav == pytest.approx(600.0)


# ── value capture ─────────────────────────────────────────────────────────────

def test_value_capture_is_pooled_not_a_mean_of_ratios():
    """One site with near-zero headroom must not dominate. Pooled is
    (sum numerator)/(sum denominator); the mean of per-site ratios here is
    an order of magnitude larger."""
    sc_sav = pd.Series({'A': 100.0, 'B': 100.0})
    pf_sav = pd.Series({'A': 200.0, 'B': 101.0})
    roll = pd.Series({'A': 150.0, 'B': 100.9})

    num = sc.wmean(roll - sc_sav, None)
    den = sc.wmean(pf_sav - sc_sav, None)
    pooled = num / den
    per_site_mean = float(((roll - sc_sav) / (pf_sav - sc_sav)).mean())

    # Numerator mean (50 + 0.9)/2 = 25.45; denominator mean (100 + 1)/2 = 50.5.
    assert pooled == pytest.approx(25.45 / 50.5)
    # Site B's own ratio is 0.9 on $1 of headroom, which drags the plain mean
    # of ratios up to 0.70 against the pooled 0.50.
    assert per_site_mean == pytest.approx(0.70, abs=0.01)
    assert pooled != pytest.approx(per_site_mean, abs=0.05)


def test_horizon_production_constant_matches_the_run_map():
    assert horizon.PRODUCTION_HORIZON in horizon.RUNS
    assert horizon.RUNS[horizon.PRODUCTION_HORIZON] == sc.REF_ROLLING_AMBER


# ── nudge enrolment ───────────────────────────────────────────────────────────

def test_no_arm_carries_an_enrolment_payment(root, monkeypatch):
    """Every nudge arm is on the retailer's no-VPP tariff, so each saving is
    exactly the solar-only bill less that arm's bill, with nothing added."""
    sites = ['A', 'B']
    write_bills(root, 'solar_ea', 'NSW_2024', sites, [2000, 2000], [1000, 1000])
    write_bills(root, 'sc_ea', 'NSW_2024_20kWh', sites, [1600, 1600], [800, 800])
    write_bills(root, 'roll_ea', 'NSW_2024_20kWh', sites, [1500, 1500], [700, 700])
    write_bills(root, 'nudge_ea', 'NSW_2024_20kWh', sites, [1400, 1400], [650, 650])

    monkeypatch.setattr(nudge, 'RETAILERS', [dict(
        key='ea', label='EnergyAustralia Battery Ease', solar='solar_ea',
        self_cons='sc_ea', no_nudge='roll_ea', nudge='nudge_ea',
        dark='#000000', mid='#888888')])
    monkeypatch.setattr(sc, 'BATTERY_SIZES', [20])
    monkeypatch.setattr(nudge.sc, 'BATTERY_SIZES', [20])

    df = nudge.collect(root, 'NSW', '2024', weighted=False)
    by_arm = df.set_index('arm')['hh_saving']

    assert by_arm[nudge.SELF_CONSUMPTION] == pytest.approx(400.0)
    assert by_arm[nudge.NO_NUDGE] == pytest.approx(500.0)
    assert by_arm[nudge.WITH_NUDGE] == pytest.approx(600.0)
    assert df['nudge_gain'].dropna().iloc[0] == pytest.approx(100.0)
    # An enrolment payment would show up as a constant added to the rolling
    # arms, lifting them against self consumption.
    assert 'enrolment_applied' not in df.columns


def test_nudge_reads_the_runs_the_retailer_briefing_draws(root):
    """The no-nudge and self-consumption arms must be the same runs the
    retailer briefing draws, so the two charts are comparable. Read from
    retailer_briefing itself rather than restated, so a change there fails
    here instead of silently diverging."""
    briefing = _load('retailer_briefing')

    origin_modes = {m['key']: m['run'] for m in briefing.ORIGIN_MODES}
    ea_modes = {m['key']: m['run'] for m in briefing.EA_MODES}
    by_key = {r['key']: r for r in nudge.RETAILERS}

    # Origin: the briefing's imperfect-foresight no-VPP line and the
    # self-consumption line it draws on that panel.
    assert by_key['origin']['no_nudge'] == origin_modes['LP_OR_NOVPP_IF']
    assert by_key['origin']['self_cons'] == origin_modes['SC_OR_NOVPP']
    assert (by_key['origin']['solar']
            == briefing.BRIEFINGS['origin']['baseline_run'])

    # EnergyAustralia: same two series on its own panel.
    assert by_key['ea']['no_nudge'] == ea_modes['LP_EA_NOVPP_IF']
    assert by_key['ea']['self_cons'] == ea_modes['SC_EA']
    assert by_key['ea']['solar'] == briefing.BRIEFINGS['ea']['baseline_run']


def test_briefing_draws_its_novpp_line_without_an_enrolment_payment():
    """The payment belongs to the band edge standing for an enrolled but
    undispatched battery, not to the plain no-VPP line the nudge chart
    reuses. If the briefing ever moves it onto that line, the nudge chart
    has to follow."""
    briefing = _load('retailer_briefing')
    ea = briefing.BRIEFINGS['ea']

    novpp_lines = [s for s in ea['series']
                   if s['kind'] == 'line' and s.get('mode') == 'LP_EA_NOVPP']
    assert novpp_lines, 'EA briefing no longer draws a plain no-VPP line'
    # The payment is a chart-level setting consumed by the band edge only.
    assert 'vpp_annual_payment' not in novpp_lines[0]

    bands = [s for s in ea['series'] if s['kind'] == 'band']
    assert any(b.get('lower_mode') == 'LP_EA_NOVPP' for b in bands), \
        'EA briefing no longer uses the no-VPP run as its band floor'


def test_shared_arms_keep_the_briefing_line_style():
    """A series the retailer briefing also draws must look the same on both
    charts, so only the nudge arm may carry a style of its own. The briefing
    draws self consumption and TOU optimised solid."""
    briefing = _load('retailer_briefing')

    assert nudge.ARM_ENCODING[nudge.SELF_CONSUMPTION][1] == briefing.SOLID
    assert nudge.ARM_ENCODING[nudge.NO_NUDGE][1] == briefing.SOLID
    # The nudge is new here, and must not borrow a style the briefing has
    # already given another meaning: its short dash is perfect foresight and
    # its dotted line a VPP band edge.
    nudge_style = nudge.ARM_ENCODING[nudge.WITH_NUDGE][1]
    assert nudge_style not in (briefing.SOLID, briefing.DASHED,
                               briefing.DOTTED)


def test_every_arm_has_an_encoding():
    assert set(nudge.ARM_ENCODING) == set(nudge.ARMS)


def test_arms_are_ordered_highest_saving_first():
    """The legend reads in ARMS order, so the nudge arm heads it and self
    consumption closes it, matching the vertical stacking on the chart and
    the order retailer_briefing.py lists its series in."""
    assert nudge.ARMS[0] == nudge.WITH_NUDGE
    assert nudge.ARMS[-1] == nudge.SELF_CONSUMPTION


def test_arm_names_carry_no_foresight_basis():
    """The basis is one per chart and belongs in the caption, not repeated
    in every series name."""
    for arm in nudge.ARMS:
        assert 'foresight' not in arm.lower()


def test_no_retailer_carries_an_enrolment_field():
    """No arm is enrolled, so the registry carries no payment to apply."""
    assert all('enrolment' not in r for r in nudge.RETAILERS)


# ── panel selection ───────────────────────────────────────────────────────────

def test_panels_never_show_two_views_of_one_quantity():
    """Cycles per day is discharge rescaled. Ranking by movement alone would
    pick both; the family rule must not."""
    df = pd.DataFrame({
        'c_deg_c_per_kwh': [3.4, 3.9, 4.6],
        'hh_saving': [1000.0, 1000.0, 1000.0],
        'ws_saving': [500.0, 499.0, 498.0],
        'coordination_value': [300.0, 300.0, 300.0],
        'discharge_kwh': [4400.0, 4000.0, 3600.0],
        'cycles_per_day': [0.75, 0.68, 0.61],
    })
    panels = degradation.rank_panels(df)

    assert len(panels) == 2
    families = {degradation.PANEL_FAMILY[p] for p in panels}
    assert len(families) == 2
    assert not {'throughput', 'cycles'} <= set(panels)


def test_every_panel_has_a_family():
    assert set(degradation.PANEL_SPECS) == set(degradation.PANEL_FAMILY)
