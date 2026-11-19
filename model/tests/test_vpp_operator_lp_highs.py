"""Tests for the HiGHS-backed operator LP (Part C, solve-time restructure).

The single property that matters is equivalence: the persistent model must
produce the same dispatch as the from-scratch builder it replaces. A
performance change that silently altered dispatch would be invisible -- both
paths return plausible numbers, and every downstream invariant would still
hold.

So most of this file is differential testing against
vpp_operator_lp.solve_operator_window over randomised inputs, including the
cases the fixed-size model has to get right: a short event span with a
deactivated tail, a binding cap, a binding budget, blocked forced export, and
no-grid-charge windows.
"""
import warnings

import numpy as np
import pytest

from model.core.scenario import BatterySpec
from model.components.vpp_operator_lp import solve_operator_window
from model.components.vpp_operator_lp_highs import HighsOperatorLp

INTERVAL_H = 5 / 60
W = 24


def _bat(capacity=13.5, power=5.0, reserve=10.0, eta=0.95, c_deg=0.039):
    return BatterySpec(
        capacity_kwh=capacity, backup_reserve_pct=reserve, power_kw=power,
        charge_efficiency=eta, discharge_efficiency=eta, c_deg_per_kwh=c_deg,
    )


def _case(rng, n_eff, *, cap=100.0, budget=5.0, limit='import_and_export',
          export_limit=np.inf, min_export=0.0, p_curtail=None, ngc=False,
          spike=False):
    n = W
    # Solar can be NEGATIVE in real EDP traces -- overnight inverter parasitic
    # draw, documented on SiteTraces.sol_to_home_kwh. Generating only
    # non-negative solar is what let an unclamped curtail bound reach a real
    # run: 8,106 of 8,296 operator windows infeasible while every test passed.
    solar = rng.uniform(-0.01, 0.6, n)
    load = rng.uniform(0.0, 0.4, n)
    op_imp = rng.uniform(0.05, 0.40, n)
    op_exp = op_imp - rng.uniform(0.0, 0.05, n)
    if spike:
        op_exp[min(n_eff - 1, n - 1)] = 4.0
    n_exp_hi = rng.uniform(0.0, 0.3, n)
    ngc_win = (rng.random(n) < 0.4) if ngc else None
    return dict(
        solar=solar, load=load, op_imp=op_imp, op_exp=op_exp, n=n,
        n_eff=n_eff, n_imp_budget=budget, n_exp_hi=n_exp_hi,
        soc_entry=6.0, vpp_exp_reward=1.0, vpp_imp_reward=1.0,
        remaining_cap=cap, min_export_price_per_kwh=min_export,
        no_grid_charge_window=ngc_win,
    ), dict(limit=limit, export_limit=export_limit, p_curtail=p_curtail,
            ngc=ngc)


def _both(kw, meta, bat=None):
    """Solve the same case through both implementations."""
    bat = bat or _bat()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ref = solve_operator_window(
            solar_window=kw['solar'], load_window=kw['load'],
            op_imp=kw['op_imp'], op_exp=kw['op_exp'],
            n=kw['n'], n_eff=kw['n_eff'],
            n_imp_budget=kw['n_imp_budget'], n_exp_hi=kw['n_exp_hi'],
            battery_spec=bat, soc_entry=kw['soc_entry'],
            vpp_exp_reward=kw['vpp_exp_reward'],
            vpp_imp_reward=kw['vpp_imp_reward'],
            remaining_cap=kw['remaining_cap'],
            vpp_limit_applies_to=meta['limit'],
            export_limit=meta['export_limit'],
            interval_hours=INTERVAL_H, site_id='S', k=0,
            p_curtail=meta['p_curtail'],
            min_export_price_per_kwh=kw['min_export_price_per_kwh'],
            no_grid_charge_window=kw['no_grid_charge_window'],
        )
        lp = HighsOperatorLp(
            w=W, battery_spec=bat, interval_hours=INTERVAL_H,
            vpp_limit_applies_to=meta['limit'],
            has_no_grid_charge=meta['ngc'],
            p_curtail=meta['p_curtail'], export_limit=meta['export_limit'])
        got = lp.solve(**kw)
    return ref, got


DEGENERATE = []          # cases where the escape below was used


def _assert_same(ref, got, label, atol=1e-7):
    """Equivalence, with an explicit and reported escape for degenerate ties.

    The LP genuinely has alternative optima. Natural and forced import differ
    by exactly the constant reward, so once the budget row binds, WHERE inside
    the horizon the natural import is booked is free -- the objective, the
    totals and the committed interval are all identical and only the interior
    placement moves. Demanding per-variable equality there would be asserting
    a tie-break, not a behaviour.

    So the escape requires the committed interval AND every block total to
    match exactly, which is everything the loop consumes and everything that
    reaches VppTraces in aggregate. It is recorded in DEGENERATE so it cannot
    be used silently: a run where most cases take the escape would mean the
    strict comparison had stopped testing anything.
    """
    assert (ref is None) == (got is None), (
        f"{label}: feasibility disagrees -- reference "
        f"{'infeasible' if ref is None else 'feasible'}, highs "
        f"{'infeasible' if got is None else 'feasible'}"
    )
    if ref is None:
        return
    assert got.n_eff == ref.n_eff

    diff = np.abs(got.x - ref.x)
    if diff.max() <= atol:
        return

    n = ref.x.size // 10
    for key in ref.committed:
        a, b = ref.committed[key], got.committed[key]
        assert abs(a - b) <= atol, (
            f"{label}: committed {key} differs, reference={a:.9g} "
            f"highs={b:.9g}. This is the interval the loop actually commits, "
            f"so it is a real difference, not a degenerate tie."
        )
    for block in range(10):
        a = ref.x[block * n:(block + 1) * n].sum()
        b = got.x[block * n:(block + 1) * n].sum()
        assert abs(a - b) <= atol * n, (
            f"{label}: block {block} total differs, reference={a:.9g} "
            f"highs={b:.9g}"
        )
    DEGENERATE.append(label)


# ---------- equivalence ----------

@pytest.mark.parametrize('n_eff', [1, 2, 7, W - 1, W])
def test_matches_the_reference_across_event_span_lengths(n_eff):
    """The fixed-size model with a deactivated tail must equal a model built
    at n_eff. This is what makes building at W sound rather than convenient."""
    rng = np.random.default_rng(100 + n_eff)
    kw, meta = _case(rng, n_eff, spike=True)
    ref, got = _both(kw, meta)
    assert ref is not None, "reference infeasible; fixture proves nothing"
    _assert_same(ref, got, f"n_eff={n_eff}")


def test_the_deactivated_tail_carries_no_dispatch():
    rng = np.random.default_rng(7)
    kw, meta = _case(rng, 5, spike=True)
    _, got = _both(kw, meta)
    assert got is not None
    n, e = kw['n'], kw['n_eff']
    for block in (0, 1, 2, 3, 5, 6, 7, 8, 9):
        np.testing.assert_allclose(
            got.x[block * n + e:(block + 1) * n], 0.0, atol=1e-9,
            err_msg=f"block {block} tail is live")
    np.testing.assert_allclose(got.x[4 * n + e:5 * n], got.x[4 * n + e - 1],
                               atol=1e-9)


@pytest.mark.parametrize('seed', range(25))
def test_matches_the_reference_on_randomised_windows(seed):
    rng = np.random.default_rng(seed)
    n_eff = int(rng.integers(1, W + 1))
    kw, meta = _case(
        rng, n_eff,
        cap=float(rng.choice([0.05, 1.0, 100.0])),
        budget=float(rng.choice([0.0, 0.5, 10.0])),
        limit=str(rng.choice(['export_only', 'import_and_export'])),
        export_limit=float(rng.choice([0.2, np.inf])),
        min_export=float(rng.choice([0.0, 0.5])),
        p_curtail=float(rng.choice([0.08])) if rng.random() < 0.5 else None,
        ngc=bool(rng.random() < 0.5),
        spike=bool(rng.random() < 0.7),
    )
    ref, got = _both(kw, meta)
    _assert_same(ref, got, f"seed={seed} n_eff={n_eff} {meta}")


def test_a_binding_cap_matches():
    rng = np.random.default_rng(3)
    kw, meta = _case(rng, W, cap=0.1, limit='export_only', spike=True)
    ref, got = _both(kw, meta)
    assert ref is not None
    n = kw['n']
    assert ref.x[9 * n:10 * n].sum() == pytest.approx(0.1, abs=1e-6), (
        "cap did not bind; fixture proves nothing")
    _assert_same(ref, got, 'binding cap')


def test_a_binding_budget_matches():
    rng = np.random.default_rng(4)
    kw, meta = _case(rng, W, budget=0.05)
    kw['load'] = np.full(kw['n'], 0.4)
    kw['solar'] = np.zeros(kw['n'])
    ref, got = _both(kw, meta)
    assert ref is not None
    n = kw['n']
    assert ref.x[6 * n:7 * n].sum() == pytest.approx(0.05, abs=1e-6), (
        "budget did not bind; fixture proves nothing")
    _assert_same(ref, got, 'binding budget')


def test_blocked_forced_export_matches():
    rng = np.random.default_rng(5)
    kw, meta = _case(rng, W, min_export=2.0, spike=True)
    ref, got = _both(kw, meta)
    assert ref is not None
    _assert_same(ref, got, 'blocked f_exp')


def test_no_grid_charge_matches():
    rng = np.random.default_rng(6)
    kw, meta = _case(rng, W, ngc=True, spike=True)
    assert kw['no_grid_charge_window'].any(), "mask never fires"
    ref, got = _both(kw, meta)
    assert ref is not None
    _assert_same(ref, got, 'no-grid-charge')


def test_infeasible_windows_agree():
    """A full battery with surplus solar, curtailment off and export blocked
    has nowhere to put the energy. Both paths must refuse it."""
    rng = np.random.default_rng(8)
    kw, meta = _case(rng, W, cap=0.0, export_limit=0.0)
    kw['solar'] = np.full(kw['n'], 1.0)
    kw['load'] = np.zeros(kw['n'])
    kw['soc_entry'] = 13.5
    kw['n_exp_hi'] = np.zeros(kw['n'])
    ref, got = _both(kw, meta)
    assert ref is None, "fixture is feasible; it proves nothing"
    assert got is None


# ---------- reuse ----------

def test_repeated_solves_do_not_leak_state():
    """The model is reused across windows, so data from one solve must not
    survive into the next. Solving A, then B, then A again must reproduce A."""
    rng = np.random.default_rng(11)
    kw_a, meta = _case(rng, 6, spike=True)
    kw_b, _ = _case(rng, W, cap=0.2, min_export=1.0, spike=True)

    lp = HighsOperatorLp(
        w=W, battery_spec=_bat(), interval_hours=INTERVAL_H,
        vpp_limit_applies_to=meta['limit'], has_no_grid_charge=meta['ngc'],
        p_curtail=meta['p_curtail'], export_limit=meta['export_limit'])
    first = lp.solve(**kw_a)
    lp.solve(**kw_b)
    again = lp.solve(**kw_a)
    assert first is not None and again is not None
    np.testing.assert_allclose(again.x, first.x, atol=1e-9)
    assert lp.solves == 3


def test_alternating_span_lengths_do_not_leak():
    """Going from a long span to a short one must deactivate the tail, and
    back again must reactivate it."""
    rng = np.random.default_rng(12)
    kw_long, meta = _case(rng, W, spike=True)
    kw_short, _ = _case(rng, 3, spike=True)
    lp = HighsOperatorLp(
        w=W, battery_spec=_bat(), interval_hours=INTERVAL_H,
        vpp_limit_applies_to=meta['limit'], has_no_grid_charge=meta['ngc'],
        p_curtail=meta['p_curtail'], export_limit=meta['export_limit'])
    a1 = lp.solve(**kw_long)
    s1 = lp.solve(**kw_short)
    a2 = lp.solve(**kw_long)
    s2 = lp.solve(**kw_short)
    np.testing.assert_allclose(a2.x, a1.x, atol=1e-9)
    np.testing.assert_allclose(s2.x, s1.x, atol=1e-9)


def test_rejects_an_event_span_longer_than_the_window():
    lp = HighsOperatorLp(
        w=8, battery_spec=_bat(), interval_hours=INTERVAL_H,
        vpp_limit_applies_to='export_only', has_no_grid_charge=False)
    rng = np.random.default_rng(13)
    kw, _ = _case(rng, 8)
    kw['n_eff'] = 12
    with pytest.raises(ValueError, match=r'n_eff must be in'):
        lp.solve(**kw)


# ---------- structure ----------

def test_the_model_is_built_once_and_reused():
    """The point of the class: one HiGHS instance for the life of the site,
    carrying its basis between windows."""
    rng = np.random.default_rng(14)
    kw, meta = _case(rng, W, spike=True)
    lp = HighsOperatorLp(
        w=W, battery_spec=_bat(), interval_hours=INTERVAL_H,
        vpp_limit_applies_to=meta['limit'], has_no_grid_charge=meta['ngc'],
        p_curtail=meta['p_curtail'], export_limit=meta['export_limit'])
    model_id = id(lp._h)
    for _ in range(3):
        lp.solve(**kw)
    assert id(lp._h) == model_id, "the HiGHS model was replaced between solves"
    assert lp.solves == 3


def test_degeneracy_escape_is_not_load_bearing():
    """If most comparisons took the degenerate escape, the strict per-variable
    check would have stopped testing anything. Runs last by name ordering
    within the module is not guaranteed, so this asserts a bound rather than a
    count."""
    assert len(DEGENERATE) <= 6, (
        f"{len(DEGENERATE)} comparisons fell back to the degenerate escape: "
        f"{DEGENERATE}"
    )


# ---------- negative solar ----------
# Regression for the curtail bound. Found on a real 2022 site-year, where
# 8,106 of 8,296 operator windows were infeasible under HiGHS and none under
# the from-scratch solver. Cause: the curtail upper bound was set to `solar`
# directly, and a negative upper against a zero lower is an infeasible
# variable. Both reference implementations clamp with np.maximum(0.0, solar).

@pytest.mark.parametrize('n_eff', [1, 9, W])
def test_negative_solar_does_not_make_the_window_infeasible(n_eff):
    """A single negative solar interval must not sink the whole window."""
    rng = np.random.default_rng(500 + n_eff)
    kw, meta = _case(rng, n_eff, p_curtail=0.08, spike=True)
    kw['solar'] = np.abs(kw['solar'])
    kw['solar'][0] = -6.977e-04          # the magnitude seen in the real trace
    ref, got = _both(kw, meta)
    assert ref is not None, "reference infeasible; the fixture proves nothing"
    assert got is not None, (
        "HiGHS refused a window the reference solved -- the curtail upper "
        "bound is probably unclamped against negative solar"
    )
    _assert_same(ref, got, f'negative solar n_eff={n_eff}')


def test_negative_solar_with_curtailment_disabled_still_matches():
    """With p_curtail None the curtail block is pinned to zero, so the clamp
    is irrelevant -- but the flow balance still carries negative solar."""
    rng = np.random.default_rng(521)
    kw, meta = _case(rng, W, p_curtail=None, spike=True)
    kw['solar'] = np.abs(kw['solar'])
    kw['solar'][:4] = -6.977e-04
    ref, got = _both(kw, meta)
    assert ref is not None
    _assert_same(ref, got, 'negative solar, no curtailment')


def test_an_all_negative_solar_window_matches():
    """A full overnight window: every interval has parasitic draw."""
    rng = np.random.default_rng(522)
    kw, meta = _case(rng, W, p_curtail=0.08)
    kw['solar'] = np.full(kw['n'], -6.977e-04)
    kw['n_exp_hi'] = np.zeros(kw['n'])
    ref, got = _both(kw, meta)
    assert ref is not None, "reference infeasible; the fixture proves nothing"
    _assert_same(ref, got, 'all-negative solar')
