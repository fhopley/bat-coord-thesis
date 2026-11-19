"""Tests for the hedge arithmetic in analysis/scripts/cap_hedge_analysis.py.

See docs/cap_hedge_spec.md (Formulation). The script
sits with the analysis scripts and imports its neighbours by bare name, so
that directory goes on sys.path and the module is loaded by file path.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / 'analysis' / 'scripts'
sys.path.insert(0, str(SCRIPTS))
_spec = importlib.util.spec_from_file_location('cap_hedge_analysis',
                                               SCRIPTS / 'cap_hedge_analysis.py')
ch = importlib.util.module_from_spec(_spec)
sys.modules['cap_hedge_analysis'] = ch      # dataclasses look the module up
_spec.loader.exec_module(ch)

# Two-quarter example: H = 2,200 h, F = $20/MWh both quarters.
# Spike quarter CSP $60, L $100; quiet quarter CSP $2, L $5.
A2 = np.array([2.2 * (60 - 20), 2.2 * (2 - 20)])      # [88, -39.6]
B2 = np.array([-100.0, -5.0])
X_CROSS = 95.0 / 127.6
W_CROSS = -39.6 * X_CROSS - 5.0


def _random_lines(seed, n=20):
    rng = np.random.default_rng(seed)
    a = rng.normal(-3.0, 40.0, n)
    b = -np.abs(rng.normal(30.0, 60.0, n))
    assert (a > 0).any() and (a < 0).any()       # bounded maximum exists
    return a, b


def _grid_max(a, b, k, half_width=5.0, step=1e-3):
    centre = ch.maximise_lowest_k(a, b, k)[0]
    xs = np.arange(centre - half_width, centre + half_width, step)
    w = ch.lowest_k_mean(a, b, xs, k)
    i = int(np.argmax(w))
    assert 0 < i < len(xs) - 1, 'grid maximum on the window edge'
    return xs[i], w.max(), step


# ---------------------------------------------------------------------------
# W_k and its maximiser
# ---------------------------------------------------------------------------

def test_two_quarter_example_peak():
    lo, hi, w = ch.maximise_lowest_k(A2, B2, 1)
    assert lo == pytest.approx(X_CROSS, rel=1e-12) and hi == lo
    assert w == pytest.approx(W_CROSS, rel=1e-12)


@pytest.mark.parametrize('seed', [1, 2, 3])
@pytest.mark.parametrize('k', [1, 2, 4, 10])
def test_exact_maximum_agrees_with_brute_force_grid(seed, k):
    a, b = _random_lines(seed)
    lo, hi, w = ch.maximise_lowest_k(a, b, k)
    assert np.isfinite(lo) and np.isfinite(hi)
    x_grid, w_grid, step = _grid_max(a, b, k)
    assert w >= w_grid - 1e-9, f"exact {w} below grid {w_grid} (k={k})"
    assert w - w_grid <= np.abs(a).max() * step, f"exact {w} vs grid {w_grid}"
    assert lo - step <= x_grid <= hi + step, f"argmax {lo}..{hi} vs grid {x_grid}"


def test_unbounded_when_every_line_slopes_the_same_way():
    b = np.array([-10.0, -20.0, -5.0])
    assert ch.maximise_lowest_k(np.array([1.0, 2.0, 3.0]), b, 1)[:2] == (np.inf, np.inf)
    assert ch.maximise_lowest_k(-np.array([1.0, 2.0, 3.0]), b, 1)[:2] == (-np.inf, -np.inf)


def test_flat_maximum_reported_as_range():
    # Worst of: rising line, falling line, and a flat line that sits below
    # their crossing -> W_1 is flat between the flat line's two crossings.
    a = np.array([1.0, -1.0, 0.0])
    b = np.array([0.0, 0.0, -2.0])
    lo, hi, w = ch.maximise_lowest_k(a, b, 1)
    assert (lo, hi, w) == pytest.approx((-2.0, 2.0, -2.0))


# ---------------------------------------------------------------------------
# Tier summary: x_0 and outcomes at x_worst
# ---------------------------------------------------------------------------

def _tier():
    a, b_nc = _random_lines(6, n=12)
    rng = np.random.default_rng(6)
    b_k = b_nc * 0.4 + rng.normal(0, 5, 12)          # a scenario with less exposure
    m = {'No CER': rng.normal(1.3, 0.5, 400), 'Amber': rng.normal(-1.5, 1.0, 400)}
    e = rng.gamma(1.5, 400.0, 400)
    return ch.Tier(name='T', years_label='2022', quarters=[(2022, q % 4 + 1) for q in range(12)],
                   a=a, b={'No CER': b_nc, 'Amber': b_k}, m=m,
                   w_count={k: np.ones(400) for k in m},
                   w_excess={k: e * ch.INTERVAL_HOURS for k in m},
                   n_gt={k: 400 for k in m})


def test_exposure_neutral_is_payout_weighted_mean_import():
    t = _tier()
    m, w = t.m['Amber'], t.w_excess['Amber']
    x0 = ch.exposure_neutral(t, 'Amber')
    assert np.sum((x0 - m) * w) == pytest.approx(0.0, abs=1e-9 * np.sum(np.abs(m * w)))


def test_summary_outcomes_at_x_worst():
    t = _tier()
    s = ch.analyse(t, ['No CER', 'Amber']).set_index('scenario')
    for lab in ('No CER', 'Amber'):
        b = t.b[lab]
        lo, hi, w_max = ch.maximise_lowest_k(t.a, b, 1)
        r = s.loc[lab]
        assert np.isfinite(lo)
        assert r['x_worst_lo_kw'] == lo and r['x_worst_k1_kw'] == lo
        assert r['W1_at_xworst'] == pytest.approx(w_max, rel=1e-12)
        assert r['mean_N_at_xworst'] == pytest.approx(np.mean(t.a * lo + b), rel=1e-12)
        assert r['mean_N_at_0'] == pytest.approx(np.mean(b), rel=1e-12)
        y, q = t.quarters[int(np.argmin(t.a * lo + b))]
        assert r['worst_q_at_xworst'] == f'{y} Q{q}'


# ---------------------------------------------------------------------------
# Premium basis and the per-quarter table
# ---------------------------------------------------------------------------

def _state_inputs():
    """NSW 2022-2023: synthetic 5-minute prices with spikes, and two
    scenarios whose above-cap cost differs by quarter."""
    rng = np.random.default_rng(11)
    years = ['2022', '2023']
    rrp = {}
    for y in years:
        idx = pd.date_range(f'{y}-01-01 00:05', f'{int(y) + 1}-01-01 00:00',
                            freq='5min', tz='Etc/GMT-10')
        p = pd.Series(60.0, index=idx)
        p.iloc[rng.choice(len(idx), 300, replace=False)] = rng.uniform(301, 9000, 300)
        rrp[y] = p
    quarters = [(int(y), q) for y in years for q in (1, 2, 3, 4)]
    mi = pd.MultiIndex.from_tuples(quarters)
    runs = {}
    for lab, scale in (('No CER', 1.0), ('Amber', -0.5)):
        run = ch.StateRun()
        run.years = list(years)
        run.m_gt = [rng.normal(scale, 0.3, 50)]
        run.e_gt = [rng.uniform(1, 5000, 50)]
        run.L_q = [pd.Series(scale * rng.uniform(5, 150, len(quarters)), index=mi)]
        run.finish()
        runs[(lab, 'NSW')] = run
    scen = [dict(label='No CER'), dict(label='Amber')]
    return runs, rrp, scen, quarters


def test_sample_mean_premium_charges_the_mean_price_every_quarter():
    runs, rrp, scen, quarters = _state_inputs()
    soq = ch.build_state_tier('NSW', runs, rrp, scen, 'start_of_quarter')
    flat = ch.build_state_tier('NSW', runs, rrp, scen, 'sample_mean')
    expected = [ch.ASX_CAP_BY_REGION['NSW1'][q] for q in quarters]
    assert soq.quarters == flat.quarters == quarters
    assert list(soq.price) == expected
    assert np.allclose(flat.price, np.mean(expected))
    assert len(set(expected)) > 1                         # the swap changes something
    assert np.allclose(flat.a, flat.hours * (flat.csp - np.mean(expected)) / 1000)
    assert np.allclose(soq.a, soq.hours * (soq.csp - np.array(expected)) / 1000)
    for lab in ('No CER', 'Amber'):                       # exposure is untouched
        assert np.array_equal(soq.b[lab], flat.b[lab])


def test_quarter_lines_rank_one_is_the_worst_quarter():
    runs, rrp, scen, _ = _state_inputs()
    tier = ch.build_state_tier('NSW', runs, rrp, scen)
    summary = ch.analyse(tier, ['No CER', 'Amber']).set_index('scenario')
    lines = ch.quarter_lines(tier, summary.reset_index())
    assert len(lines) == 2 * len(tier.quarters)
    checked = 0
    for lab in ('No CER', 'Amber'):
        rows = lines[lines['scenario'] == lab]
        r = summary.loc[lab]
        if not np.isfinite(r['x_worst_lo_kw']):
            continue
        checked += 1
        worst = rows.loc[rows['rank_at_xworst'] == 1]
        assert worst['quarter'].item() == r['worst_q_at_xworst']
        assert worst['N_at_xworst'].item() == pytest.approx(r['W1_at_xworst'], rel=1e-9)
        assert np.allclose(rows['above_cap_cost'], -rows['N_at_0'])
    assert checked > 0, 'no scenario had a finite x_worst'
