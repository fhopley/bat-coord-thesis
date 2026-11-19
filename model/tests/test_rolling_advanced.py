"""Unit tests for model.components.rolling_advanced.

Two gates matter here and they check different things.

The EQUIVALENCE gate solves each window twice: once through
AdvancedWindowSolver, and once by assembling the same window with the real
builders (_build_lp plus _build_block_tier_data) and handing it to a fresh
solver. If the two objectives disagree, the window solver is not solving the
advanced LP.

The STRUCTURAL gate mutates a solver forward many windows, reads the
constraint matrix back out of HiGHS and compares it against a model built
fresh for the same window. Objective agreement alone is not enough: a wrong
coefficient that happens not to bind produces a correct objective and a wrong
model, and that is exactly the failure that showed up when this scheme was
first benchmarked.
"""
import math

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import highspy

from model.core.prices import (
    block_window_mask, build_lp_price_vectors_advanced,
    build_zero_import_penalty,
)
from model.core.scenario import BatterySpec
from model.core.tariff import (
    BlockTariffWindow, BlockTier, NetworkTariff, RetailTariff, CriticalRate,
    TariffPeriod, ZeroImportReward,
)
from model.components.perfect_foresight_lp import _build_lp
from model.components.perfect_foresight_lp_advanced import (
    _build_block_tier_data, _build_day_map,
)
from model.components.rolling_advanced import (
    AdvancedWindowSolver, block_window_specs, tier_upper_bounds,
)

IH = 1.0          # hourly intervals, as in test_perfect_foresight_lp_advanced
STATE = 'NSW'
H = 18            # window length in intervals


def _idx(n, start='2022-07-04 01:00:00'):
    """Winter start (no NSW DST) so block window hours are AEST hours."""
    return pd.date_range(start, periods=n, freq='60min', tz='Etc/GMT-10')


def _bspec():
    return BatterySpec(capacity_kwh=13.5, backup_reserve_pct=10.0,
                       power_kw=5.0, charge_efficiency=0.95,
                       discharge_efficiency=0.95, c_deg_per_kwh=0.02)


def _net():
    return NetworkTariff(
        name='net', state='NSW',
        usage_tariffs=(TariffPeriod(0, 24, 10.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 2.0),),
        daily_c_per_day=60.0,
        usage_dlf=1.05, export_dlf=1.05, import_tlf=1.0, export_tlf=1.0,
    )


def _rt(import_window=(8, 10), export_window=(14, 16), weekends=True):
    """TOU plan with a block import window and a block export window.

    When the block window applies on weekends, the underlying usage rate in
    those hours is 0.0, matching the real YAML convention where the block
    tier is the only import cost for the window. When it does NOT apply on
    weekends the underlying rate must be non-zero, because on a weekend the
    block masking does not fire and that rate is what the customer pays --
    a weekday-only window over a 0.0 rate would price weekend imports at
    nothing, which is a tariff definition error rather than a modelling one.
    """
    in_window_usage_c = 0.0 if weekends else 20.0
    in_window_feedin_c = 0.0 if weekends else 5.0
    return RetailTariff(
        name='Adv', state='NSW', type='tou', daily_c_per_day=0.0,
        usage_tariffs=(TariffPeriod(0, import_window[0], 20.0),
                       TariffPeriod(*import_window, in_window_usage_c),
                       TariffPeriod(import_window[1], 24, 20.0)),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, export_window[0], 5.0),
                        TariffPeriod(*export_window, in_window_feedin_c),
                        TariffPeriod(export_window[1], 24, 5.0)),
        # usage_other must exceed the feed-in rate: inside a block import
        # window it IS the per-interval import price, and a lower one would
        # make importing to export profitable.
        usage_other_c_per_kwh=6.0, feedin_other_c_per_kwh=0.0,
        block_import_tiers=(BlockTariffWindow(
            start_hour=import_window[0], end_hour=import_window[1],
            tiers=(BlockTier(4.0, 0.0), BlockTier(math.inf, 30.0)),
            applies_weekends=weekends),),
        block_export_tiers=(BlockTariffWindow(
            start_hour=export_window[0], end_hour=export_window[1],
            tiers=(BlockTier(3.0, 12.0), BlockTier(math.inf, 3.0)),
            applies_weekends=weekends),),
    )


# ---------------------------------------------------------------------------
# Period data and membership, mirroring what the planner will precompute
# ---------------------------------------------------------------------------

def _period(n_intervals=120, rt=None, seed=0):
    rt = rt or _rt()
    idx = _idx(n_intervals)
    rng = np.random.default_rng(seed)
    hour = np.arange(n_intervals) % 24
    solar = 2.5 * np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None)
    solar *= rng.uniform(0.4, 1.0, n_intervals)
    load = 0.4 + 0.8 * np.exp(-0.5 * ((hour - 19) / 2.0) ** 2)
    # Built by the real builder rather than by hand, so block masking and the
    # weekday-only rule are applied exactly as the perfect-foresight path
    # applies them. A TOU plan's vectors do not depend on the wholesale price,
    # so the zero series here is not a simplification.
    zero_prices = pd.Series(np.zeros(n_intervals), index=idx)
    p_imp_s, p_exp_s = build_lp_price_vectors_advanced(
        idx, zero_prices, _net(), rt, STATE, IH)
    p_imp = p_imp_s.to_numpy()
    p_exp = p_exp_s.to_numpy()
    _, day_map = _build_day_map(idx, STATE, IH)
    masks = tuple(block_window_mask(idx, w, STATE, IH)
                  for w in (rt.block_import_tiers or ()))
    masks += tuple(block_window_mask(idx, w, STATE, IH)
                   for w in (rt.block_export_tiers or ()))
    return dict(idx=idx, rt=rt, solar=solar, load=load, p_imp=p_imp,
                p_exp=p_exp, day_map=day_map, masks=masks)


def _n_slots(day_map, n, H):
    """Maximum local dates any window of length H touches, as the planner
    computes it: from the day map, not from the horizon."""
    return max(int(day_map[min(k + H, n) - 1] - day_map[k]) + 1
               for k in range(n - H + 1))


def _runs_for_window(per, k, n, n_slots):
    """Membership run per (block window, day slot) for the window at k."""
    day_map, masks = per['day_map'], per['masks']
    day0 = day_map[k]
    runs = {}
    for w_idx, mask in enumerate(masks):
        for slot in range(n_slots):
            sel = np.where(mask[k:k + n] & (day_map[k:k + n] == day0 + slot))[0]
            runs[(w_idx, slot)] = ((int(sel[0]), int(sel[-1]) + 1)
                                   if sel.size else None)
            if sel.size:
                # The incremental update relies on contiguity; assert it here
                # so a violation shows up as a test failure rather than a
                # silently wrong linking row.
                assert sel[-1] - sel[0] + 1 == sel.size, (
                    f"window {w_idx} slot {slot} at k={k} is not contiguous: "
                    f"{sel}")
    return runs


# ---------------------------------------------------------------------------
# Reference: assemble the same window with the real builders
# ---------------------------------------------------------------------------

def _reference_objective(per, k, n, soc_init, soc_final, day0_used,
                         p_imp=None, p_exp=None, rrp_window=None):
    """Solve the window by the whole-period route, restricted to the window.

    Uses _build_lp and _build_block_tier_data directly -- the same functions
    perfect_foresight_lp_advanced uses -- so this is a check against the real
    builders rather than against a second reconstruction.

    p_imp / p_exp (window-length) override the period vectors. rrp_window
    (the window's forecast) makes tier membership the effective masks from
    prices.block_tier_masks, as perfect_foresight_lp_advanced builds them.
    """
    idx, rt = per['idx'], per['rt']
    sl = slice(k, k + n)
    widx = idx[sl]
    n_days, day_map_w = _build_day_map(widx, STATE, IH)

    c_std, A_std, b_std, lo_std, hi_std = _build_lp(
        n, per['solar'][sl], per['load'][sl],
        per['p_imp'][sl] if p_imp is None else p_imp,
        per['p_exp'][sl] if p_exp is None else p_exp,
        _bspec(), soc_init, soc_final, np.inf, IH)

    masks = {}
    if rrp_window is not None:
        from model.core.prices import block_tier_masks
        imp_m, exp_m = block_tier_masks(widx, rt, STATE, IH, rrp_window)
        masks = dict(import_window_masks=imp_m or None,
                     export_window_masks=exp_m or None)

    n_imp = len(rt.block_import_tiers or ())
    imp_used = {w: v for w, v in day0_used.items() if w < n_imp}
    exp_used = {w - n_imp: v for w, v in day0_used.items() if w >= n_imp}

    n_tv, tier_lo, tier_hi, tier_costs, linking = _build_block_tier_data(
        n, widx, rt, IH, n_days, day_map_w, STATE,
        first_day_block_import_used=imp_used,
        first_day_block_export_used=exp_used, **masks)

    if n_tv:
        c = np.concatenate([c_std, tier_costs])
        lo = np.concatenate([lo_std, tier_lo])
        hi = np.concatenate([hi_std, tier_hi])
        A = sp.vstack([sp.hstack([A_std, sp.csc_matrix((2 * n, n_tv))],
                                 format='csc'), linking], format='csc')
        b = np.concatenate([b_std, np.zeros(linking.shape[0])])
    else:
        c, lo, hi, A, b = c_std, lo_std, hi_std, A_std, b_std

    A = sp.csc_matrix(A)
    lp = highspy.HighsLp()
    lp.num_col_ = A.shape[1]
    lp.num_row_ = A.shape[0]
    lp.col_cost_ = c
    lp.col_lower_ = lo
    lp.col_upper_ = np.where(np.isinf(hi), highspy.kHighsInf, hi)
    lp.row_lower_ = b
    lp.row_upper_ = b
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr.astype(np.int32)
    lp.a_matrix_.index_ = A.indices.astype(np.int32)
    lp.a_matrix_.value_ = A.data
    h = highspy.Highs()
    h.setOptionValue('output_flag', False)
    h.passModel(lp)
    h.run()
    if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        return None
    return float(h.getObjectiveValue())


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def test_specs_follow_the_canonical_order():
    """Import windows first, then export, matching _build_block_tier_data."""
    specs = block_window_specs(_rt())
    assert [s.is_import for s in specs] == [True, False]
    # Import tier costs are positive (a cost), export negative (a credit).
    assert specs[0].costs == (0.0, 0.30)
    assert specs[1].costs == (-0.12, -0.03)
    assert specs[0].capacities == (4.0, math.inf)


def _rt_three_tier_import():
    """Import window with two FINITE tiers before the unbounded one.

    Carry-in that overflows the first tier has to spill the remainder into
    the second. With only one finite tier the remainder always lands on an
    infinite tier and the spill rule is unobservable, so a two-tier fixture
    cannot tell a correct implementation from one that drops it.
    """
    return RetailTariff(
        name='ThreeTier', state='NSW', type='tou', daily_c_per_day=0.0,
        usage_tariffs=(TariffPeriod(0, 8, 20.0), TariffPeriod(8, 10, 0.0),
                       TariffPeriod(10, 24, 20.0)),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 5.0),),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
        block_import_tiers=(BlockTariffWindow(
            start_hour=8, end_hour=10,
            tiers=(BlockTier(2.0, 0.0), BlockTier(5.0, 15.0),
                   BlockTier(math.inf, 30.0)),
        ),),
    )


@pytest.mark.parametrize('rt_factory, used, label', [
    (_rt, {0: 5.0, 1: 1.0}, 'two tiers, carry-in past the finite tier'),
    (_rt_three_tier_import, {0: 4.0}, 'three tiers, spill between finite tiers'),
    (_rt_three_tier_import, {0: 0.5}, 'three tiers, partial first tier only'),
    (_rt_three_tier_import, {0: 99.0}, 'three tiers, everything exhausted'),
])
def test_tier_bounds_match_the_whole_period_builder(rt_factory, used, label):
    """Carry-in spill from one tier into the next must match
    _build_block_tier_data, which owns the rule for the perfect-foresight
    path."""
    rt = rt_factory()
    idx = _idx(120)
    n_days, day_map = _build_day_map(idx, STATE, IH)
    specs = block_window_specs(rt)

    n_imp = len(rt.block_import_tiers or ())
    imp_used = {w: v for w, v in used.items() if w < n_imp}
    exp_used = {w - n_imp: v for w, v in used.items() if w >= n_imp}
    _, _, ref_hi, _, _ = _build_block_tier_data(
        len(idx), idx, rt, IH, n_days, day_map, STATE,
        first_day_block_import_used=imp_used,
        first_day_block_export_used=exp_used)
    got_hi = tier_upper_bounds(specs, n_days, day0_used=used)
    np.testing.assert_array_equal(
        got_hi, ref_hi, err_msg=f"tier bounds differ ({label})")


def test_surplus_slots_are_pinned_to_zero():
    """A window touching fewer dates than there are slots leaves an empty
    linking row, which must force those tier variables to zero rather than
    letting them earn an export credit for free."""
    per = _period()
    n = H
    solver = AdvancedWindowSolver(n, _bspec(), np.inf, IH, 0.0,
                                  block_window_specs(per['rt']), n_slots=2)
    k = 0
    runs = _runs_for_window(per, k, n, 2)
    # Force every slot-1 pair to empty, whatever the window actually spans.
    runs = {key: (run if key[1] == 0 else None) for key, run in runs.items()}
    solver.set_membership(runs)
    x = solver.solve(per['solar'][k:k + n], per['load'][k:k + n],
                     per['p_imp'][k:k + n], per['p_exp'][k:k + n],
                     6.0, 6.0, tier_upper_bounds(
                         block_window_specs(per['rt']), 2))
    solver.restore_soc_bounds()
    assert x is not None
    tier = x[solver.N_CORE:]
    slot1 = [solver._tier_col[(w, 1)] - solver.N_CORE
             for w in range(len(solver.specs))]
    for base, spec in zip(slot1, solver.specs):
        block = tier[base:base + spec.n_tiers]
        np.testing.assert_allclose(block, 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# Equivalence gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('day0_used', [
    {},
    {0: 2.5, 1: 1.0},
    {0: 9.0},           # import tier 1 fully exhausted, spilling into tier 2
])
def test_window_solver_matches_the_real_builders(day0_used):
    """Every window, solved twice: once through the mutated solver, once by
    assembling the same window with _build_lp and _build_block_tier_data."""
    per = _period(n_intervals=120)
    n = H
    n_int = len(per['idx'])
    n_slots = _n_slots(per['day_map'], n_int, n)
    specs = block_window_specs(per['rt'])
    solver = AdvancedWindowSolver(n, _bspec(), np.inf, IH, 0.0, specs, n_slots)
    tier_hi = tier_upper_bounds(specs, n_slots, day0_used)

    soc_init = soc_final = 6.0
    n_windows = n_int - n + 1
    mismatches = []
    for k in range(n_windows):
        solver.set_membership(_runs_for_window(per, k, n, n_slots))
        x = solver.solve(per['solar'][k:k + n], per['load'][k:k + n],
                         per['p_imp'][k:k + n], per['p_exp'][k:k + n],
                         soc_init, soc_final, tier_hi)
        got = solver.objective_value() if x is not None else None
        solver.restore_soc_bounds()
        want = _reference_objective(per, k, n, soc_init, soc_final, day0_used)
        if got is None or want is None:
            if got is not want:
                mismatches.append((k, got, want))
        elif abs(got - want) > 1e-8 * max(1.0, abs(want)):
            mismatches.append((k, got, want))

    assert not mismatches, (
        f"{len(mismatches)} of {n_windows} windows disagree with the "
        f"reference build. First five (k, solver, reference): "
        f"{mismatches[:5]}"
    )


def test_solver_matches_reference_with_weekday_only_windows():
    """Weekend days have no block intervals at all, so runs go empty and come
    back. A stale coefficient would survive that transition unnoticed."""
    per = _period(n_intervals=216, rt=_rt(weekends=False))
    n = H
    n_int = len(per['idx'])
    n_slots = _n_slots(per['day_map'], n_int, n)
    specs = block_window_specs(per['rt'])
    solver = AdvancedWindowSolver(n, _bspec(), np.inf, IH, 0.0, specs, n_slots)
    tier_hi = tier_upper_bounds(specs, n_slots)

    empties = 0
    for k in range(n_int - n + 1):
        runs = _runs_for_window(per, k, n, n_slots)
        empties += sum(1 for r in runs.values() if r is None)
        solver.set_membership(runs)
        x = solver.solve(per['solar'][k:k + n], per['load'][k:k + n],
                         per['p_imp'][k:k + n], per['p_exp'][k:k + n],
                         6.0, 6.0, tier_hi)
        got = solver.objective_value() if x is not None else None
        solver.restore_soc_bounds()
        want = _reference_objective(per, k, n, 6.0, 6.0, {})
        assert got is not None and want is not None, f"window {k} not optimal"
        assert abs(got - want) <= 1e-8 * max(1.0, abs(want)), (
            f"window {k}: solver {got}, reference {want}")
    assert empties > 0, "fixture never produced an empty run; test is vacuous"


# ---------------------------------------------------------------------------
# Structural gate
# ---------------------------------------------------------------------------

def test_structural_gate_matrix_matches_a_fresh_build():
    """Mutate forward across the whole period, then compare the constraint
    matrix against a model built fresh for the final window."""
    per = _period(n_intervals=240)
    n = H
    n_int = len(per['idx'])
    n_slots = _n_slots(per['day_map'], n_int, n)
    specs = block_window_specs(per['rt'])

    mutated = AdvancedWindowSolver(n, _bspec(), np.inf, IH, 0.0, specs, n_slots)
    last_k = n_int - n
    for k in range(last_k + 1):
        mutated.set_membership(_runs_for_window(per, k, n, n_slots))

    fresh = AdvancedWindowSolver(n, _bspec(), np.inf, IH, 0.0, specs, n_slots)
    fresh.set_membership(_runs_for_window(per, last_k, n, n_slots))

    got, want = mutated.constraint_matrix(), fresh.constraint_matrix()
    assert got.shape == want.shape
    diff = abs(got - want)
    if diff.nnz:
        rows, cols = diff.nonzero()
        detail = []
        for r, c in list(zip(rows, cols))[:8]:
            band = ('tier var' if c >= mutated.N_CORE
                    else 'flow balance' if r < n
                    else 'soc' if r < 2 * n else 'linking')
            detail.append(f"row {r} col {c} ({band}): "
                          f"mutated={got[r, c]} fresh={want[r, c]}")
        pytest.fail(
            f"{diff.nnz} differing coefficients after {last_k} mutations:\n  "
            + "\n  ".join(detail))

    # Per-linking-row nonzero counts are what the incremental update is
    # supposed to preserve, so report them even on success.
    for key in sorted(mutated._link_row):
        a = mutated.linking_row_positions(key)
        b = fresh.linking_row_positions(key)
        np.testing.assert_array_equal(
            a, b, err_msg=f"linking row {key} positions differ")


def test_membership_update_touches_only_the_moving_ends():
    """The efficiency claim, asserted rather than assumed: a one-interval
    slide changes at most two coefficients per linking row."""
    per = _period(n_intervals=120)
    n = H
    n_slots = _n_slots(per['day_map'], len(per['idx']), n)
    specs = block_window_specs(per['rt'])
    solver = AdvancedWindowSolver(n, _bspec(), np.inf, IH, 0.0, specs, n_slots)

    solver.set_membership(_runs_for_window(per, 0, n, n_slots))
    solver.n_coeff_changes = 0
    steps = 60
    for k in range(1, steps + 1):
        solver.set_membership(_runs_for_window(per, k, n, n_slots))

    per_step = solver.n_coeff_changes / steps
    assert per_step <= 2 * solver.n_linking, (
        f"{per_step:.2f} coefficient changes per step against a ceiling of "
        f"{2 * solver.n_linking} ({solver.n_linking} linking rows x 2 ends)"
    )


def test_set_membership_requires_every_pair():
    per = _period()
    n = H
    specs = block_window_specs(per['rt'])
    solver = AdvancedWindowSolver(n, _bspec(), np.inf, IH, 0.0, specs, 2)
    runs = _runs_for_window(per, 0, n, 2)
    runs.pop((0, 1))
    with pytest.raises(ValueError, match="every .window, slot. pair"):
        solver.set_membership(runs)


def test_set_membership_rejects_an_out_of_range_run():
    per = _period()
    n = H
    specs = block_window_specs(per['rt'])
    solver = AdvancedWindowSolver(n, _bspec(), np.inf, IH, 0.0, specs, 2)
    runs = {key: None for key in solver._link_row}
    runs[(0, 0)] = (5, n + 3)
    with pytest.raises(ValueError, match="half-open range"):
        solver.set_membership(runs)


def test_no_block_features_gives_the_plain_core_model():
    """The common TOU plan has no block tiers, so there are no tier variables
    and no linking rows -- the matrix is the wholesale one and never mutates."""
    rt = RetailTariff(
        name='Plain', state='NSW', type='tou', daily_c_per_day=0.0,
        usage_tariffs=(TariffPeriod(0, 24, 20.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 5.0),),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
    )
    specs = block_window_specs(rt)
    assert specs == ()
    solver = AdvancedWindowSolver(H, _bspec(), np.inf, IH, 0.0, specs, 2)
    assert solver.n_tier_vars == 0 and solver.n_linking == 0
    assert solver.constraint_matrix().shape == (2 * H, 6 * H)

    solver.set_membership({})
    per = _period(rt=rt)
    x = solver.solve(per['solar'][:H], per['load'][:H],
                     per['p_imp'][:H], per['p_exp'][:H], 6.0, 6.0)
    solver.restore_soc_bounds()
    assert x is not None and len(x) == 6 * H
    assert solver.n_coeff_changes == 0


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

from model.core.prices import build_lp_price_vectors_advanced  # noqa: E402
from model.components.rolling_advanced import (  # noqa: E402
    AdvancedWindowPlanner, _contiguous_runs,
)
from model.components.billing import _zero_import_day_credits  # noqa: E402


def _rt_full(weekends=True):
    """The GloBird shape: block import, block export, zero-import reward
    sharing the export window, and both critical rates."""
    base = _rt(weekends=weekends)
    return RetailTariff(
        name='Full', state='NSW', type='tou', daily_c_per_day=181.5,
        usage_tariffs=base.usage_tariffs,
        usage_tariffs_weekend=None,
        feedin_tariffs=base.feedin_tariffs,
        usage_other_c_per_kwh=6.0, feedin_other_c_per_kwh=0.0,
        block_import_tiers=base.block_import_tiers,
        block_export_tiers=base.block_export_tiers,
        zero_import_reward=ZeroImportReward(
            start_hour=14, end_hour=16, reward_c_per_day=100.0,
            tolerance_kwh_per_hour=0.03),
        critical_export_rate=CriticalRate(2000.0, 100.0),
        critical_import_rate=CriticalRate(-100.0, -5.0),
    )


def _planner(rt=None, n_intervals=120, H_=None, settled=None):
    """settled defaults to a flat, never-critical settled price, so tests
    that commit flows on the GloBird shape have the day-state input the
    planner requires."""
    rt = rt or _rt_full()
    idx = _idx(n_intervals)
    if settled is None:
        settled = np.full(n_intervals, 100.0)
    return AdvancedWindowPlanner(
        index=idx, retail_tariff=rt, network_tariff=_net(), state=STATE,
        battery_spec=_bspec(), horizon_intervals=H_ or H,
        interval_hours=IH, curtail_cost_per_kwh_dollars=0.0,
        settled_rrp_mwh=settled)


def test_planner_rejects_a_wholesale_plan():
    rt = RetailTariff(
        name='Amber', state='NSW', type='wholesale_exposed',
        daily_c_per_day=82.0, usage_tariffs=(), usage_tariffs_weekend=None,
        feedin_tariffs=(), usage_other_c_per_kwh=4.0,
        feedin_other_c_per_kwh=1.0)
    with pytest.raises(ValueError, match="rolling_perfect_foresight_lp"):
        _planner(rt=rt)


def test_requires_price_forecast_follows_the_critical_rates():
    assert _planner(rt=_rt_full()).requires_price_forecast is True
    assert _planner(rt=_rt()).requires_price_forecast is False


def test_slot_count_comes_from_the_day_map():
    """An 18-interval window on an hourly index touches at most 2 local
    dates; a 30-interval one touches at most 3."""
    assert _planner(H_=18).n_slots == 2
    assert _planner(H_=30).n_slots == 3
    assert _planner(H_=1).n_slots == 1


def test_contiguous_runs():
    m = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], dtype=bool)
    np.testing.assert_array_equal(_contiguous_runs(m),
                                  np.array([[1, 3], [5, 6], [7, 10]]))
    assert _contiguous_runs(np.zeros(5, dtype=bool)).shape == (0, 2)


# -- prices -----------------------------------------------------------------

def test_window_prices_reproduce_the_perfect_foresight_vectors():
    """With the settled price as the 'forecast', the window's price vectors
    must equal what the perfect-foresight path builds for those intervals,
    up to the zero-import availability term."""
    rt = _rt_full()
    p = _planner(rt=rt)
    idx = p.index
    rng = np.random.default_rng(11)
    rrp = rng.normal(200.0, 900.0, len(idx))
    rrp[5:9] = 5000.0        # export-critical, reaching into the import window
    rrp[20:24] = -400.0      # import-critical
    rrp[14:16] = 5000.0      # export-critical inside the export window
    rrp[32:34] = -400.0      # import-critical inside the import window
    rrp[38] = -400.0         # import-critical inside the export window

    ref_imp, ref_exp = build_lp_price_vectors_advanced(
        idx, pd.Series(rrp, index=idx), _net(), rt, STATE, IH)
    penalty = build_zero_import_penalty(idx, rt, STATE, IH).to_numpy()

    for k in (0, 7, 30, 61):
        got_imp, got_exp = p.window_prices(rrp[k:k + H], k, H)
        # Nothing committed yet, so the reward is available everywhere and the
        # penalty applies exactly as the perfect-foresight path applies it.
        np.testing.assert_allclose(
            got_imp, ref_imp.to_numpy()[k:k + H] + penalty[k:k + H],
            rtol=0, atol=1e-15, err_msg=f"p_imp at k={k}")
        np.testing.assert_allclose(
            got_exp, ref_exp.to_numpy()[k:k + H],
            rtol=0, atol=1e-15, err_msg=f"p_exp at k={k}")


def test_critical_rate_follows_the_forecast_not_the_outcome():
    """The same interval priced from two different forecasts: under the spike
    p_exp is the critical rate and p_imp moves by the same delta. This is the
    whole mechanism of forecast error on this path."""
    p = _planner()
    quiet = np.full(H, 100.0)
    spike = quiet.copy()
    spike[3] = 5000.0
    a_imp, a_exp = p.window_prices(quiet, 0, H)
    b_imp, b_exp = p.window_prices(spike, 0, H)
    rho_x = (100.0 + 0.0) / 100.0      # critical export rate + feedin_other
    assert b_exp[3] == pytest.approx(rho_x)
    assert b_imp[3] - a_imp[3] == pytest.approx(rho_x - a_exp[3])
    # The shift keeps the spread untouched.
    np.testing.assert_allclose(b_imp - b_exp, a_imp - a_exp, atol=1e-15)
    # No other interval moves.
    others = [j for j in range(H) if j != 3]
    np.testing.assert_allclose(b_exp[others], a_exp[others], atol=1e-15)


def test_price_forecast_required_only_when_critical_rates_exist():
    plain = _planner(rt=_rt())
    p_imp, _ = plain.window_prices(None, 0, H)   # must not raise
    assert np.isfinite(p_imp).all()
    with pytest.raises(ValueError, match="price forecast is required"):
        _planner().window_prices(None, 0, H)


def test_penalty_stops_once_the_day_reward_is_lost():
    """Linear then gone: once today's committed window imports exceed the
    tolerance, today's remaining penalty goes to zero, but the next local
    day's stays.

    The window has to be long enough to contain BOTH days' reward windows,
    or clearing the whole window is indistinguishable from clearing today's
    share of it.
    """
    H_long = 30
    p = _planner(H_=H_long)
    H = H_long
    rrp = np.full(H, 100.0)
    # Interval 14 starts at 14:00 local on day 0, inside the reward window
    # (the index carries interval-END labels, so interval j starts at hour j).
    k_reward = int(np.flatnonzero(p._reward_mask)[0])
    assert p.day_map[k_reward] == 0
    k0 = 10
    before, _ = p.window_prices(rrp, k0, H)
    assert (before > 0).all()
    penalised_before = int((before > p.base_imp[k0:k0 + H] + 1e-12).sum())
    assert penalised_before > 0, "fixture has no penalised intervals"

    p.observe_commit(k_reward, import_kwh=5.0, export_kwh=0.0)
    assert p.reward_available(k_reward) is False

    after, _ = p.window_prices(rrp, k0, H)
    day0 = p.day_map[k0:k0 + H] == p.day_map[k0]
    np.testing.assert_allclose(after[day0], p.base_imp[k0:k0 + H][day0],
                               atol=1e-15)
    # Next day's reward is still live, so its penalty is untouched.
    later = ~day0
    assert (before[later] > p.base_imp[k0:k0 + H][later] + 1e-12).any(), (
        "the window contains no penalised intervals on a later local day; "
        "the test cannot tell 'clear today' from 'clear everything'")
    np.testing.assert_allclose(after[later], before[later], atol=1e-15)


# -- day state --------------------------------------------------------------

def test_reward_threshold_matches_billing():
    """A day whose window imports sit either side of the tolerance must get
    the same verdict from the planner as from billing."""
    rt = _rt_full()
    p = _planner(rt=rt)
    tol = rt.zero_import_reward.tolerance_kwh
    idx = p.index
    reward_intervals = np.flatnonzero(
        (p.day_map == p.day_map[0]) & p._reward_mask)
    assert reward_intervals.size >= 2

    for amount, expect_available in ((tol * 0.5, True), (tol * 2.0, False)):
        planner = _planner(rt=rt)
        planner.observe_commit(int(reward_intervals[0]), amount, 0.0)
        imports = pd.Series(np.zeros(len(idx)), index=idx)
        imports.iloc[reward_intervals[0]] = amount
        credits = _zero_import_day_credits(
            imports, idx, rt.zero_import_reward, IH, STATE)
        billing_pays = bool(credits.iloc[0] < 0)
        assert planner.reward_available(int(reward_intervals[0])) is expect_available
        assert billing_pays is expect_available, (
            f"billing and planner disagree at {amount} kWh against a "
            f"tolerance of {tol}")


def test_solver_residual_does_not_lose_the_reward():
    """The threshold rule exists for this case: a float residual must not
    cost the household a reward that billing still pays."""
    rt = RetailTariff(
        name='ExactZero', state='NSW', type='tou', daily_c_per_day=0.0,
        usage_tariffs=(TariffPeriod(0, 24, 20.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 5.0),),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
        zero_import_reward=ZeroImportReward(
            start_hour=14, end_hour=16, reward_c_per_day=100.0,
            tolerance_kwh_per_hour=0.0),
    )
    p = _planner(rt=rt)
    k = int(np.flatnonzero(p._reward_mask)[0])
    p.observe_commit(k, import_kwh=1e-14, export_kwh=0.0)
    assert p.reward_available(k) is True, (
        "a 1e-14 residual lost the reward; billing would still have paid it")


def test_day_state_resets_at_local_midnight():
    p = _planner()
    k = int(np.flatnonzero(p.day_map == 0)[-1])     # last interval of day 0
    p.observe_commit(k, import_kwh=3.0, export_kwh=2.0)
    k_next = int(np.flatnonzero(p.day_map == 1)[0])
    assert p.day0_used(k_next) == {}
    assert p.reward_available(k_next) is True


def test_day0_used_matches_a_direct_sum_over_the_block_mask():
    """The accumulator must agree with summing the committed flows over the
    same mask and the same local day -- the grouping billing uses."""
    rt = _rt_full()
    p = _planner(rt=rt)
    rng = np.random.default_rng(5)
    n = 48
    imp = rng.uniform(0.0, 1.5, n)
    exp = rng.uniform(0.0, 1.5, n)
    for k in range(n):
        p.observe_commit(k, imp[k], exp[k])

    last = n - 1
    day = p.day_map[last]
    same_day = (p.day_map[:n] == day)
    for w_idx, spec in enumerate(p.specs):
        # Both windows must have contributed on this day, or an accumulator
        # reading the wrong flow would be indistinguishable from a correct one.
        assert (_mask_of(p, w_idx)[:n] & same_day).any(), (
            f"block window {w_idx} has no committed intervals on the day "
            f"being checked; the assertion below would be vacuous")
        mask = np.zeros(p.T_p, dtype=bool)
        for start, stop in p._runs[w_idx]:
            mask[start:stop] = True
        flows = imp if spec.is_import else exp
        want = float(flows[mask[:n] & same_day].sum())
        got = p.day0_used(last).get(w_idx, 0.0)
        assert got == pytest.approx(want), (
            f"block window {w_idx} ({'import' if spec.is_import else 'export'})"
            f": accumulator {got}, direct sum {want}")


def test_day0_used_feeds_through_to_the_tier_bounds():
    """The accumulator is only useful if it actually tightens the LP."""
    p = _planner()
    k = int(np.flatnonzero(
        (p.day_map == 0) & _mask_of(p, 0))[0])
    p.observe_commit(k, import_kwh=3.0, export_kwh=0.0)
    hi = tier_upper_bounds(p.specs, p.n_slots, p.day0_used(k))
    base = tier_upper_bounds(p.specs, p.n_slots)
    assert hi[0] < base[0], "day-0 import tier capacity was not reduced"
    assert hi[0] == pytest.approx(1.0)     # 4.0 capacity less 3.0 used


def _mask_of(planner, w_idx):
    mask = np.zeros(planner.T_p, dtype=bool)
    for start, stop in planner._runs[w_idx]:
        mask[start:stop] = True
    return mask


# -- membership ------------------------------------------------------------

def test_window_runs_match_a_direct_mask_computation():
    """The run-based membership must equal what a mask-and-day-index scan
    gives, for every window in the period."""
    p = _planner(n_intervals=240)
    n = H
    masks = [_mask_of(p, w) for w in range(len(p.specs))]
    for k in range(p.T_p - n + 1):
        got = p.window_runs(k, n)
        day0 = p.day_map[k]
        for w_idx, mask in enumerate(masks):
            for slot in range(p.n_slots):
                sel = np.flatnonzero(
                    mask[k:k + n] & (p.day_map[k:k + n] == day0 + slot))
                want = ((int(sel[0]), int(sel[-1]) + 1) if sel.size else None)
                assert got[(w_idx, slot)] == want, (
                    f"k={k} window={w_idx} slot={slot}: "
                    f"runs gave {got[(w_idx, slot)]}, mask scan gave {want}")


def test_planner_solves_every_window_and_matches_the_reference():
    """End to end: the planner's own price vectors and membership, solved
    through the mutated model, against the reference build."""
    rt = _rt()          # no critical rates, so prices need no forecast
    p = _planner(rt=rt, n_intervals=120)
    per = _period(n_intervals=120, rt=rt)
    n = H
    for k in range(p.T_p - n + 1):
        p_imp, p_exp = p.window_prices(None, k, n)
        np.testing.assert_allclose(p_imp, per['p_imp'][k:k + n], atol=1e-15)
        x = p.solve_window(k, n, per['solar'][k:k + n], per['load'][k:k + n],
                           p_imp, p_exp, 6.0, 6.0)
        assert x is not None, f"window {k} not optimal"
        want = _reference_objective(per, k, n, 6.0, 6.0, {})
        got = p.last_objective
        assert abs(got - want) <= 1e-8 * max(1.0, abs(want)), (
            f"window {k}: planner {got}, reference {want}")


def test_planner_matches_the_reference_with_forecast_critical_intervals():
    """As above on the GloBird shape, with a forecast whose critical
    intervals move from window to window and land inside both block windows
    on both sides. The reference's tier membership comes from
    block_tier_masks on the window's forecast, independently of the
    planner's membership bookkeeping, so a critical interval left in (or
    wrongly taken out of) a linking row changes the objective."""
    rt = _rt_full()
    p = _planner(rt=rt, n_intervals=96)
    per = _period(n_intervals=96, rt=rt)
    n = H
    checked_in_window = 0
    for k in range(p.T_p - n + 1):
        rrp = _spiky_forecast(p, k, n, seed=9)
        p_imp, p_exp = p.window_prices(rrp, k, n)
        x = p.solve_window(k, n, per['solar'][k:k + n], per['load'][k:k + n],
                           p_imp, p_exp, 6.0, 6.0)
        assert x is not None, f"window {k} not optimal"
        want = _reference_objective(per, k, n, 6.0, 6.0, {}, p_imp=p_imp,
                                    p_exp=p_exp, rrp_window=rrp)
        got = p.last_objective
        assert abs(got - want) <= 1e-8 * max(1.0, abs(want)), (
            f"window {k}: planner {got}, reference {want}")
        members = p.window_members(k, n)
        checked_in_window += any(isinstance(v, np.ndarray) for v in members.values())
    assert checked_in_window > 10, "fixture rarely put a critical interval in a window"


# ---------------------------------------------------------------------------
# Integration with the rolling loop
# ---------------------------------------------------------------------------

from model.core.forecast import ForecastSlice, LoadSlice  # noqa: E402
from model.core.scenario import RollingConfig  # noqa: E402
from model.components.rolling_optimisation import (  # noqa: E402
    _WholesalePlanner, rolling_optimisation,
)
from model.components.perfect_foresight_lp_advanced import (  # noqa: E402
    perfect_foresight_lp_advanced,
)
from model.components.billing import compute_bill  # noqa: E402


class _PerfectPriceProvider:
    """Settled prices returned as the forecast, run_time at the decision
    instant so causality holds exactly."""

    def __init__(self, prices, index, interval_hours):
        self.p = np.asarray(prices, dtype=float)
        self.index = index
        self.ih = interval_hours

    def horizon_available(self):
        return len(self.p)

    def get(self, decision_instant, n_intervals):
        k = int(self.index.get_indexer(
            [decision_instant + pd.Timedelta(hours=self.ih)])[0])
        if k < 0:
            return None
        seg = self.p[k:k + n_intervals]
        if seg.size == 0:
            return None
        return ForecastSlice(run_time=decision_instant, rrp_mwh=seg.copy(),
                             is_infill=np.zeros(seg.size, dtype=bool))


def _rolling(rt, per, horizon_intervals, prices=None, soc=6.0):
    idx = per['idx']
    planner = AdvancedWindowPlanner(
        index=idx, retail_tariff=rt, network_tariff=_net(), state=STATE,
        battery_spec=_bspec(), horizon_intervals=horizon_intervals,
        interval_hours=IH, curtail_cost_per_kwh_dollars=0.0,
        settled_rrp_mwh=prices)
    provider = (None if prices is None
                else _PerfectPriceProvider(prices, idx, IH))
    return rolling_optimisation(
        site_id='S1',
        solar_kwh=pd.Series(per['solar'], index=idx),
        load_kwh=pd.Series(per['load'], index=idx),
        forecast_provider=provider,
        price_offsets=None,
        battery_spec=_bspec(),
        soc_init_kwh=soc, soc_final_kwh=soc,
        config=RollingConfig(forecast_source='perfect',
                             horizon_hours=float(horizon_intervals) * IH),
        interval_hours=IH,
        curtail_cost_per_kwh_dollars=0.0,
        planner=planner,
    )


@pytest.mark.parametrize('in_windows', [False, True],
                         ids=['critical_outside_windows', 'critical_inside_windows'])
def test_correctness_gate_full_horizon_reproduces_perfect_foresight(in_windows):
    """The gate for this component.

    With perfect information and the horizon set to the whole period, every
    window reaches the period end, so every window pins the same terminal SOC
    and each is the tail subproblem of the first. The committed sequence must
    coincide with the full-period optimum.

    The LP is degenerate, so the assertion is layered. The household bill and
    the total energy through the meter must agree to solver tolerance; the
    per-interval flows are reported rather than asserted, because two
    intervals at the same price are genuinely interchangeable and the LP is
    free to pick either. Asserting on them would fail on a difference the
    household never sees.

    The in-windows case puts critical intervals inside both block windows, on
    both sides, so tier exclusion in the window LP (forecast) and in the day
    state (settled) is exercised against the perfect-foresight effective
    masks.
    """
    rt = _rt_full()
    per = _period(n_intervals=72, rt=rt)
    idx = per['idx']
    rng = np.random.default_rng(21)
    prices = rng.normal(150.0, 800.0, len(idx))
    prices[10:13] = 4000.0        # export-critical
    prices[40:42] = -300.0        # import-critical
    if in_windows:
        prices[14] = 4000.0       # export-critical in the export window (day 0)
        prices[8] = -300.0        # import-critical in the import window (day 0)
        prices[33] = 4000.0       # export-critical in the import window (day 1)
        prices[38] = -300.0       # import-critical in the export window (day 1)
        prices[63] = 4000.0       # export-critical in the export window (day 2)
        from model.core.prices import critical_masks
        x, m = critical_masks(prices, rt)
        assert x[14] and m[8] and x[33] and m[38] and x[63]

    p_imp, p_exp = build_lp_price_vectors_advanced(
        idx, pd.Series(prices, index=idx), _net(), rt, STATE, IH)
    reference = perfect_foresight_lp_advanced(
        site_id='S1',
        solar_kwh=pd.Series(per['solar'], index=idx),
        load_kwh=pd.Series(per['load'], index=idx),
        price_imp=p_imp, price_exp=p_exp,
        battery_spec=_bspec(), soc_init_kwh=6.0, soc_final_kwh=6.0,
        retail_tariff=rt, prices_mwh=pd.Series(prices, index=idx),
        state=STATE, interval_hours=IH,
    )

    traces, diags = _rolling(rt, per, horizon_intervals=len(idx),
                             prices=prices)

    assert diags.causality_ok
    assert diags.sc_fallback_no_forecast == 0
    assert diags.sc_fallback_infeasible == 0

    # Diagnostic context, gathered whether or not the gate passes: where the
    # two dispatches placed the same energy differently.
    detail = []
    for name in ('import_kwh', 'export_kwh'):
        a = getattr(reference, name).to_numpy()
        b = getattr(traces, name).to_numpy()
        d = np.abs(a - b)
        n_diff = int((d > 1e-6).sum())
        if n_diff:
            j = int(np.argmax(d))
            lo, hi = max(0, j - 2), min(len(a), j + 3)
            detail.append(
                f"{name}: {n_diff} of {len(a)} intervals differ, worst "
                f"{d.max():.3e} at {j}\n" + "\n".join(
                    f"    {'>>' if i == j else '  '} [{i}] {idx[i]} "
                    f"perfect={a[i]:.9f} rolling={b[i]:.9f}"
                    for i in range(lo, hi)))
    ctx = ("\n" + "\n".join(detail)) if detail else ""

    price_series = pd.Series(prices, index=idx)
    bill_ref = compute_bill(reference, price_series, _net(), rt, STATE, IH)
    bill_got = compute_bill(traces, price_series, _net(), rt, STATE, IH)
    assert bill_got.household_total == pytest.approx(
        bill_ref.household_total, abs=1e-6), (
        f"household bill differs: perfect {bill_ref.household_total}, "
        f"rolling {bill_got.household_total}{ctx}")
    assert bill_got.retail_other_credits == pytest.approx(
        bill_ref.retail_other_credits, abs=1e-9), (
        f"zero-import reward credits differ{ctx}")

    # Same total energy through the meter.
    assert reference.import_kwh.sum() == pytest.approx(
        traces.import_kwh.sum(), abs=1e-6), ctx
    assert reference.export_kwh.sum() == pytest.approx(
        traces.export_kwh.sum(), abs=1e-6), ctx


def test_loop_skips_the_price_provider_when_no_critical_rates():
    """A plain block TOU plan must run with no price forecast at all, rather
    than falling back to self-consumption over a forecast it cannot use."""
    rt = _rt()
    per = _period(n_intervals=72, rt=rt)
    traces, diags = _rolling(rt, per, horizon_intervals=H, prices=None)
    assert diags.sc_fallback_no_forecast == 0
    assert diags.n_windows_solved == len(per['idx'])
    assert traces.import_kwh.notna().all()


def test_loop_requires_a_forecast_when_critical_rates_exist():
    rt = _rt_full()
    per = _period(n_intervals=48, rt=rt)
    with pytest.raises(ValueError, match="forecast_provider is required"):
        _rolling(rt, per, horizon_intervals=H, prices=None)


def test_loop_requires_offsets_or_a_planner():
    per = _period(n_intervals=24)
    with pytest.raises(ValueError, match="either price_offsets"):
        rolling_optimisation(
            site_id='S1',
            solar_kwh=pd.Series(per['solar'], index=per['idx']),
            load_kwh=pd.Series(per['load'], index=per['idx']),
            forecast_provider=None, price_offsets=None,
            battery_spec=_bspec(), soc_init_kwh=6.0, soc_final_kwh=6.0,
            config=RollingConfig(forecast_source='perfect', horizon_hours=6.0),
            interval_hours=IH)


def test_day_state_advances_across_the_run():
    """The accumulator has to be fed by the loop, not just exist. A run over
    several days must leave it holding the last day's committed flows."""
    rt = _rt()
    per = _period(n_intervals=72, rt=rt)
    idx = per['idx']
    planner = AdvancedWindowPlanner(
        index=idx, retail_tariff=rt, network_tariff=_net(), state=STATE,
        battery_spec=_bspec(), horizon_intervals=H, interval_hours=IH,
        curtail_cost_per_kwh_dollars=0.0)
    traces, _ = rolling_optimisation(
        site_id='S1',
        solar_kwh=pd.Series(per['solar'], index=idx),
        load_kwh=pd.Series(per['load'], index=idx),
        forecast_provider=None, price_offsets=None,
        battery_spec=_bspec(), soc_init_kwh=6.0, soc_final_kwh=6.0,
        config=RollingConfig(forecast_source='perfect',
                             horizon_hours=float(H) * IH),
        interval_hours=IH, curtail_cost_per_kwh_dollars=0.0, planner=planner)

    last_day = planner.day_map[-1]
    same_day = planner.day_map == last_day
    for w_idx, spec in enumerate(planner.specs):
        mask = _mask_of(planner, w_idx) & same_day
        flows = (traces.import_kwh if spec.is_import
                 else traces.export_kwh).to_numpy()
        want = float(flows[mask].sum())
        got = planner.day0_used(len(idx) - 1).get(w_idx, 0.0)
        assert got == pytest.approx(want, abs=1e-9), (
            f"block window {w_idx}: accumulator {got}, committed traces {want}")


def test_wholesale_planner_has_no_day_state():
    p = _WholesalePlanner(None, _bspec(), np.inf, IH, 0.0)
    assert p.requires_price_forecast is True
    assert p.observe_commit(0, 1.0, 2.0) is None


class _GappyLoadProvider:
    """Real load, except over a stated range of intervals where it returns
    None, which drives the loop into its self-consumption fallback."""

    def __init__(self, load, gap):
        self.load = np.asarray(load, dtype=float)
        self.gap = gap

    def get(self, k, n_intervals):
        if self.gap[0] <= k < self.gap[1]:
            return None
        seg = self.load[k:k + n_intervals]
        if seg.size == 0:
            return None
        return LoadSlice(load_kwh=seg.copy())


def test_day_state_accumulates_across_fallback_intervals():
    """A fallback commits real flows through the meter, so they count against
    the block tiers exactly like a solved interval's. Skipping them would let
    the accumulator drift below what actually happened, and the LP would then
    plan against tier capacity the household no longer has.
    """
    rt = _rt()
    per = _period(n_intervals=72, rt=rt)
    idx = per['idx']
    # Scaled until solar in the export window EXCEEDS the battery's charge
    # power limit, so surplus has to leave through the meter whatever the
    # state of charge. At the base fixture's 6.5 kWh daily surplus the battery
    # swallows everything and both block windows carry zero; at a smaller
    # scale it depends on the SOC the preceding solved windows happened to
    # leave, which is not something a test should rest on.
    solar = per['solar'] * 8.0
    assert (solar[62:64] > _bspec().power_kw * IH).all(), (
        "solar in the export window no longer exceeds the charge power limit; "
        "the fallback intervals are not guaranteed to export")
    planner = AdvancedWindowPlanner(
        index=idx, retail_tariff=rt, network_tariff=_net(), state=STATE,
        battery_spec=_bspec(), horizon_intervals=H, interval_hours=IH,
        curtail_cost_per_kwh_dollars=0.0)

    # The gap covers the last local day's block EXPORT window. It has to be a
    # window where the fixture actually puts energy through the meter under
    # self-consumption: over the import window this fixture's solar exceeds
    # load, so nothing is imported either way and the assertion below would
    # compare zero against zero.
    last_day = planner.day_map[-1]
    block = np.flatnonzero(_mask_of(planner, 1) & (planner.day_map == last_day))
    gap = (int(block[0]), int(block[-1]) + 1)

    traces, diags = rolling_optimisation(
        site_id='S1',
        solar_kwh=pd.Series(solar, index=idx),
        load_kwh=pd.Series(per['load'], index=idx),
        forecast_provider=None, price_offsets=None,
        battery_spec=_bspec(), soc_init_kwh=6.0, soc_final_kwh=6.0,
        config=RollingConfig(forecast_source='perfect',
                             horizon_hours=float(H) * IH),
        interval_hours=IH, curtail_cost_per_kwh_dollars=0.0,
        load_forecast_provider=_GappyLoadProvider(per['load'], gap),
        planner=planner)

    assert diags.sc_fallback_no_load_forecast >= len(block), (
        f"fixture produced only {diags.sc_fallback_no_load_forecast} "
        f"fallbacks; the test cannot tell an observed fallback from an "
        f"ignored one")

    same_day = planner.day_map == last_day
    for w_idx, spec in enumerate(planner.specs):
        mask = _mask_of(planner, w_idx) & same_day
        assert mask.any()
        flows = (traces.import_kwh if spec.is_import
                 else traces.export_kwh).to_numpy()
        want = float(flows[mask].sum())
        got = planner.day0_used(len(idx) - 1).get(w_idx, 0.0)
        assert got == pytest.approx(want, abs=1e-9), (
            f"block window {w_idx}: accumulator {got}, committed traces "
            f"{want}; {diags.sc_fallback_no_load_forecast} fallback interval(s)"
            f" in the run")

    # Non-vacuity: the fallback intervals must themselves have carried energy
    # through the block window, or an unobserved fallback is indistinguishable
    # from an observed one.
    fallback_flow = float(
        traces.export_kwh.to_numpy()[gap[0]:gap[1]].sum())
    assert fallback_flow > 1e-6, (
        f"the fallback intervals moved {fallback_flow} kWh through the export "
        f"block window; the accumulator assertion above cannot detect a "
        f"skipped fallback")


def test_planner_rejects_a_plan_where_importing_to_export_pays():
    """A block import window masked below the feed-in rate is free money.

    The LP would import and export in the same interval; the committed traces
    net those flows while the LP's own variables do not, so the day-state
    accumulator and the bill would then disagree about how much of the tier
    was used. Caught at construction rather than discovered as a puzzling
    accumulator mismatch, which is how this was actually found.
    """
    rt = RetailTariff(
        name='FreeMoney', state='NSW', type='tou', daily_c_per_day=0.0,
        usage_tariffs=(TariffPeriod(0, 8, 20.0), TariffPeriod(8, 10, 0.0),
                       TariffPeriod(10, 24, 20.0)),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 5.0),),
        usage_other_c_per_kwh=0.0,        # below the 5c feed-in rate
        feedin_other_c_per_kwh=0.0,
        block_import_tiers=(BlockTariffWindow(
            start_hour=8, end_hour=10,
            tiers=(BlockTier(4.0, 0.0), BlockTier(math.inf, 30.0))),),
    )
    with pytest.raises(ValueError, match="p_imp < p_exp"):
        _planner(rt=rt)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def test_diagnostics_are_none_when_the_plan_lacks_the_feature():
    """A plan with no zero-import reward must report None, not zero. Zero
    would read as 'never lost', which is a different fact."""
    rt = _rt()                                   # block tiers, no reward
    per = _period(n_intervals=48, rt=rt)
    _, diags = _rolling(rt, per, horizon_intervals=H, prices=None)
    assert diags.reward_days_lost is None
    assert diags.block_tier_days_exhausted is not None
    assert len(diags.block_tier_days_exhausted) == 2

    plain = RetailTariff(
        name='Plain', state='NSW', type='tou', daily_c_per_day=0.0,
        usage_tariffs=(TariffPeriod(0, 24, 20.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 5.0),),
        usage_other_c_per_kwh=6.0, feedin_other_c_per_kwh=0.0)
    per2 = _period(n_intervals=48, rt=plain)
    _, d2 = _rolling(plain, per2, horizon_intervals=H, prices=None)
    assert d2.block_tier_days_exhausted is None
    assert d2.windows_with_tier_exhausted is None
    assert d2.reward_days_lost is None


def test_wholesale_diagnostics_unchanged():
    """The default planner contributes nothing, so the wholesale path's
    diagnostics keep every advanced field at None."""
    p = _WholesalePlanner(None, _bspec(), np.inf, IH, 0.0)
    assert p.plan_diagnostics() is None


def test_reward_days_lost_counts_each_day_once():
    """Including the day still open when the counters are read -- a run that
    ends mid-day must not silently drop its last day."""
    rt = _rt_full()
    per = _period(n_intervals=72, rt=rt)
    planner = AdvancedWindowPlanner(
        index=per['idx'], retail_tariff=rt, network_tariff=_net(),
        state=STATE, battery_spec=_bspec(), horizon_intervals=H,
        interval_hours=IH, curtail_cost_per_kwh_dollars=0.0,
        settled_rrp_mwh=np.full(len(per['idx']), 100.0))
    reward = np.flatnonzero(planner._reward_mask)
    # Lose the reward on two separate local days.
    days = []
    for k in reward:
        d = int(planner.day_map[k])
        if d not in days:
            days.append(d)
            planner.observe_commit(int(k), import_kwh=5.0, export_kwh=0.0)
        if len(days) == 2:
            break
    assert len(days) == 2
    assert planner.plan_diagnostics()['reward_days_lost'] == 2
    # Idempotent: reading twice must not double-count the open day.
    assert planner.plan_diagnostics()['reward_days_lost'] == 2


def test_tier_exhaustion_counts_days_and_windows():
    rt = _rt()
    per = _period(n_intervals=72, rt=rt)
    planner = AdvancedWindowPlanner(
        index=per['idx'], retail_tariff=rt, network_tariff=_net(),
        state=STATE, battery_spec=_bspec(), horizon_intervals=H,
        interval_hours=IH, curtail_cost_per_kwh_dollars=0.0)
    block = np.flatnonzero(_mask_of(planner, 0) & (planner.day_map == 0))
    # The import window's finite capacity is 4.0 kWh.
    planner.observe_commit(int(block[0]), import_kwh=4.0, export_kwh=0.0)
    d = planner.plan_diagnostics()
    assert d['block_tier_days_exhausted'][0] == 1
    assert d['block_tier_days_exhausted'][1] == 0

    # A window planned after that has no free capacity left on day 0.
    before = planner._windows_with_tier_exhausted
    planner.solve_window(int(block[0]), H, per['solar'][:H], per['load'][:H],
                         *planner.window_prices(None, int(block[0]), H),
                         6.0, 6.0)
    assert planner._windows_with_tier_exhausted == before + 1


def test_summary_mentions_the_advanced_counters():
    rt = _rt_full()
    per = _period(n_intervals=48, rt=rt)
    rng = np.random.default_rng(3)
    prices = rng.normal(100.0, 300.0, len(per['idx']))
    _, diags = _rolling(rt, per, horizon_intervals=H, prices=prices)
    text = diags.summary()
    assert 'zero-import reward' in text
    assert 'block tiers used up' in text


# ---------------------------------------------------------------------------
# Block window runs split at local-day boundaries
# ---------------------------------------------------------------------------
# window_runs keys each run to a single day slot, taken from the day of its
# first interval, and the solver holds one tier variable per (window, day).
# A run spanning midnight would post a whole day's extra flow to the first
# day's tier variable. Half-open hour-of-day windows inside a day already
# break at midnight, so this only bites for a window covering the WHOLE day --
# EA Battery Ease's 0-24 block export tier, whose mask is unbroken across the
# entire period.

def test_splitting_is_a_no_op_for_a_window_inside_a_day():
    """The property that makes this change safe for every existing plan: a
    within-day window is already one run per day, so the splitter returns it
    untouched and behaviour is bit-for-bit what it was."""
    from model.components.rolling_advanced import (
        _contiguous_runs, _split_runs_by_day,
    )
    day_map = np.repeat([0, 1, 2], 288)
    mask = np.zeros(864, dtype=bool)
    for d in range(3):
        mask[d * 288 + 204:d * 288 + 252] = True      # 17:00-21:00
    runs = _contiguous_runs(mask)
    assert len(runs) == 3, "fixture is not one run per day"
    assert np.array_equal(_split_runs_by_day(runs, day_map), runs)


def test_a_whole_day_window_splits_into_one_run_per_day():
    from model.components.rolling_advanced import (
        _contiguous_runs, _split_runs_by_day,
    )
    day_map = np.repeat([0, 1, 2], 288)
    runs = _contiguous_runs(np.ones(864, dtype=bool))
    assert len(runs) == 1, "fixture is not a single spanning run"
    split = _split_runs_by_day(runs, day_map)
    assert split.tolist() == [[0, 288], [288, 576], [576, 864]]


def test_splitting_preserves_the_covered_intervals():
    """Splitting must partition, not resample: the union of the runs is the
    mask it came from, whatever the day lengths."""
    from model.components.rolling_advanced import (
        _contiguous_runs, _split_runs_by_day,
    )
    rng = np.random.default_rng(4)
    day_map = np.repeat(np.arange(5), 100)
    mask = rng.random(500) < 0.6
    runs = _contiguous_runs(mask)
    split = _split_runs_by_day(runs, day_map)
    covered = np.zeros(500, dtype=bool)
    for lo, hi in split:
        covered[lo:hi] = True
    np.testing.assert_array_equal(covered, mask)


def test_split_runs_never_span_a_local_day():
    from model.components.rolling_advanced import (
        _contiguous_runs, _split_runs_by_day,
    )
    day_map = np.repeat(np.arange(4), 77)          # uneven day lengths, as DST gives
    mask = np.ones(308, dtype=bool)
    split = _split_runs_by_day(_contiguous_runs(mask), day_map)
    for lo, hi in split:
        assert day_map[lo] == day_map[hi - 1], (
            f"run [{lo},{hi}) spans local days "
            f"{day_map[lo]}..{day_map[hi - 1]}"
        )


def test_a_whole_day_block_window_is_accepted_by_the_planner():
    """EA Battery Ease's shape: a 0-24 block export tier. Before the split
    this raised 'run spanning local days' at construction, so the plan could
    not be run on the rolling path at all."""
    rt = _rt()
    whole_day = RetailTariff(
        name='EAish', state=STATE, type='tou',
        daily_c_per_day=rt.daily_c_per_day,
        usage_tariffs=rt.usage_tariffs,
        usage_tariffs_weekend=rt.usage_tariffs_weekend,
        feedin_tariffs=rt.feedin_tariffs,
        usage_other_c_per_kwh=rt.usage_other_c_per_kwh,
        feedin_other_c_per_kwh=rt.feedin_other_c_per_kwh,
        block_export_tiers=(BlockTariffWindow(
            start_hour=0, end_hour=24,
            tiers=(BlockTier(10.0, 8.0), BlockTier(math.inf, 3.0)),
            applies_weekends=True),),
    )
    n = 864
    planner = AdvancedWindowPlanner(
        index=_idx(n), retail_tariff=whole_day, network_tariff=_net(),
        state=STATE, battery_spec=_bspec(), horizon_intervals=48,
        interval_hours=IH,
    )
    runs = planner._runs[0]
    days = np.unique(planner.day_map)
    assert len(runs) == len(days), (
        f"{len(runs)} run(s) for {len(days)} local day(s); a whole-day window "
        f"must give exactly one run per day"
    )


# ---------------------------------------------------------------------------
# Critical rates: tier membership in the window LP and in the day state
# ---------------------------------------------------------------------------

def _spiky_forecast(p, k, n, seed):
    """A forecast for the window at k with critical intervals that move from
    window to window, landing inside both block windows on both sides."""
    rng = np.random.default_rng(seed + k)
    rrp = np.full(n, 100.0)
    pos = rng.choice(n, size=4, replace=False)
    rrp[pos[:2]] = 5000.0
    rrp[pos[2:]] = -400.0
    return rrp


def test_window_members_leave_out_forecast_critical_positions_on_their_side():
    p = _planner(n_intervals=120)
    spec_sides = [spec.is_import for spec in p.specs]
    seen_hole = {True: False, False: False}
    for k in range(0, 120 - H):
        rrp = _spiky_forecast(p, k, H, seed=3)
        p.window_prices(rrp, k, H)
        runs = p.window_runs(k, H)
        members = p.window_members(k, H)
        for key, run in runs.items():
            got = (set() if members[key] is None
                   else set(range(*members[key])) if isinstance(members[key], tuple)
                   else set(int(j) for j in members[key]))
            want = set() if run is None else set(range(*run))
            is_import = spec_sides[key[0]]
            crit = rrp < -100.0 if is_import else rrp > 2000.0
            other = rrp > 2000.0 if is_import else rrp < -100.0
            want -= set(np.flatnonzero(crit).tolist())
            assert got == want, f"k={k} key={key}: {sorted(got)} != {sorted(want)}"
            if run is not None and (crit[run[0]:run[1]]).any():
                seen_hole[is_import] = True
            # Opposite-side criticality never removes a position.
            if run is not None:
                for j in np.flatnonzero(other[run[0]:run[1]]) + run[0]:
                    assert j in got, f"k={k} key={key}: opposite side removed {j}"
    assert seen_hole[True] and seen_hole[False], (
        f"fixture never put a critical interval in a block window: {seen_hole}")


def test_window_members_requires_window_prices_for_the_same_window():
    p = _planner()
    with pytest.raises(ValueError, match="window_prices was not called"):
        p.window_members(0, H)
    p.window_prices(np.full(H, 100.0), 0, H)
    p.window_members(0, H)                       # same window: fine
    with pytest.raises(ValueError, match="window_prices was not called"):
        p.window_members(1, H)


def test_structural_gate_holds_with_moving_critical_holes():
    """Mutate a solver forward through windows whose forecast criticality
    moves, then compare with a model built fresh for the final window."""
    n_int = 240
    p = _planner(n_intervals=n_int)
    mutated = AdvancedWindowSolver(H, _bspec(), np.inf, IH, 0.0, p.specs, p.n_slots)
    last_k = n_int - H
    holes = 0
    for k in range(last_k + 1):
        p.window_prices(_spiky_forecast(p, k, H, seed=5), k, H)
        members = p.window_members(k, H)
        holes += sum(1 for v in members.values() if isinstance(v, np.ndarray))
        mutated.set_membership(members)
    assert holes > 0, "fixture never punched a hole in a block run"

    fresh = AdvancedWindowSolver(H, _bspec(), np.inf, IH, 0.0, p.specs, p.n_slots)
    fresh.set_membership(members)
    got, want = mutated.constraint_matrix(), fresh.constraint_matrix()
    diff = abs(got - want)
    if diff.nnz:
        rows, cols = diff.nonzero()
        pytest.fail(f"{diff.nnz} differing coefficients after {last_k} "
                    f"mutations, first at row {rows[0]} col {cols[0]}")
    for key in sorted(mutated._link_row):
        np.testing.assert_array_equal(mutated.linking_row_positions(key),
                                      fresh.linking_row_positions(key))


def test_day_state_counts_block_usage_by_the_settled_price():
    """Committed export in the block export window counts towards the tier
    unless the SETTLED price made that interval export-critical. Forecast
    criticality does not affect the day state."""
    n_int = 120
    settled = np.full(n_int, 100.0)
    settled[14] = 5000.0            # hour 14, day 0, in the export window: critical
    settled[8] = -400.0             # hour 8, day 0, in the import window: critical
    p = _planner(n_intervals=n_int, settled=settled)
    w_exp = next(i for i, sp in enumerate(p.specs) if not sp.is_import)
    w_imp = next(i for i, sp in enumerate(p.specs) if sp.is_import)
    assert _in_window(p, w_exp, 14) and _in_window(p, w_exp, 15)
    assert _in_window(p, w_imp, 8) and _in_window(p, w_imp, 9)

    # A forecast that calls hour 15 critical must not change the day state.
    p.window_prices(np.where(np.arange(H) == 1, 5000.0, 100.0), 14, H)

    p.observe_commit(8, import_kwh=1.5, export_kwh=0.0)   # settled-critical: skipped
    p.observe_commit(9, import_kwh=0.7, export_kwh=0.0)   # counts
    p.observe_commit(14, import_kwh=0.0, export_kwh=2.0)  # settled-critical: skipped
    p.observe_commit(15, import_kwh=0.0, export_kwh=0.4)  # counts (forecast only)
    used = p.day0_used(15)
    assert used.get(w_imp, 0.0) == pytest.approx(0.7)
    assert used.get(w_exp, 0.0) == pytest.approx(0.4)


def _in_window(p, w_idx, k):
    from model.components.rolling_advanced import _in_runs
    return _in_runs(p._runs[w_idx], k)


def test_observe_commit_needs_settled_prices_when_a_critical_rate_meets_a_window():
    idx = _idx(48)
    planner = AdvancedWindowPlanner(
        index=idx, retail_tariff=_rt_full(), network_tariff=_net(), state=STATE,
        battery_spec=_bspec(), horizon_intervals=H, interval_hours=IH,
        curtail_cost_per_kwh_dollars=0.0)          # no settled prices
    with pytest.raises(ValueError, match="needs settled_rrp_mwh"):
        planner.observe_commit(0, 0.1, 0.0)
    # A plan without critical rates never needs them.
    plain = AdvancedWindowPlanner(
        index=idx, retail_tariff=_rt(), network_tariff=_net(), state=STATE,
        battery_spec=_bspec(), horizon_intervals=H, interval_hours=IH,
        curtail_cost_per_kwh_dollars=0.0)
    plain.observe_commit(0, 0.1, 0.0)


# ---------------------------------------------------------------------------
# Mixed plans (model_spec.md, Mixed-plan dispatch prices)
# ---------------------------------------------------------------------------

from model.core.prices import build_price_vectors  # noqa: E402

_MIXED_PERIODS = ((0, 15, 24.02), (15, 21, 49.71), (21, 24, 24.02))


def _rt_mixed(periods=_MIXED_PERIODS, usage_other=0.821, basis='wholesale',
              floor_c=0.0):
    return RetailTariff(
        name='TestMixed', state='NSW', type='mixed', daily_c_per_day=110.0,
        usage_tariffs=tuple(TariffPeriod(*p) for p in periods),
        usage_tariffs_weekend=None, feedin_tariffs=(),
        usage_other_c_per_kwh=usage_other, feedin_other_c_per_kwh=0.0,
        wholesale_export_share=0.9, export_share_basis=basis,
        export_floor_c_per_kwh=floor_c,
    )


def _mixed_prices(n, seed=5):
    rng = np.random.default_rng(seed)
    p = rng.normal(120.0, 300.0, n)
    p[[5, 6, 41]] = [6000.0, 9000.0, 4000.0]
    p[[30, 31, 55]] = [-400.0, -150.0, -800.0]
    return p


def test_mixed_planner_requires_a_price_forecast():
    pl = _planner(rt=_rt_mixed())
    assert pl.requires_price_forecast is True
    assert pl.dispatch_stack is not None
    with pytest.raises(ValueError, match="mixed plan's window prices"):
        pl.window_prices(None, 0, H)


def test_tou_planner_has_no_dispatch_stack():
    assert _planner(rt=_rt_full()).dispatch_stack is None


def test_mixed_window_prices_follow_the_forecast_not_the_settled_price():
    """Given the settled price as its forecast, a window reproduces
    build_price_vectors over the same positions. Given a forecast that misses
    a spike, the spike positions are priced at the forecast, even though the
    planner holds the settled price."""
    n = 72
    idx = _idx(n)
    rt = _rt_mixed()
    prices = _mixed_prices(n)
    pl = _planner(rt=rt, n_intervals=n, settled=prices)
    full_imp, full_exp = build_price_vectors(
        idx, pd.Series(prices, index=idx), _net(), rt, STATE, IH)
    full_imp, full_exp = full_imp.to_numpy(), full_exp.to_numpy()
    for k in (0, 5, 29, n - H):
        p_imp, p_exp = pl.window_prices(prices[k:k + H], k, H)
        np.testing.assert_array_equal(p_imp, full_imp[k:k + H])
        np.testing.assert_array_equal(p_exp, full_exp[k:k + H])

    forecast = prices.copy()
    forecast[5:7] = 100.0                       # the forecast misses the spike
    p_imp, p_exp = pl.window_prices(forecast[:H], 0, H)
    lf = _net().import_loss_factor
    for j in (5, 6):
        assert p_imp[j] == pytest.approx(
            full_imp[j] - prices[j] / 1000.0 * lf + 0.1 * lf, abs=1e-12)
        assert p_exp[j] == pytest.approx(
            0.9 * 0.1 * _net().export_loss_factor, abs=1e-12)
        assert p_exp[j] < full_exp[j] - 1.0     # non-vacuity: spike missed


def test_mixed_planner_refuses_a_plan_whose_ordering_can_fail():
    """Zero TOU rate, network export in V, no floor: at RRP = 0 the export
    credit exceeds the import price."""
    rt = _rt_mixed(periods=((0, 24, 0.0),), usage_other=0.0,
                   basis='wholesale_and_network', floor_c=None)
    with pytest.raises(ValueError, match=r"p_imp >= p_exp at 0 \$/MWh"):
        _planner(rt=rt)


def test_mixed_correctness_gate_full_horizon_reproduces_perfect_foresight():
    """As the TOU gate: perfect prices over a full horizon reproduce the
    full-period optimum on a mixed plan, with spikes large enough for grid
    charging and negative prices where the export floor binds."""
    rt = _rt_mixed()
    per = _period(n_intervals=72, rt=rt)
    idx = per['idx']
    prices = _mixed_prices(len(idx))

    p_imp, p_exp = build_lp_price_vectors_advanced(
        idx, pd.Series(prices, index=idx), _net(), rt, STATE, IH)
    reference = perfect_foresight_lp_advanced(
        site_id='S1',
        solar_kwh=pd.Series(per['solar'], index=idx),
        load_kwh=pd.Series(per['load'], index=idx),
        price_imp=p_imp, price_exp=p_exp,
        battery_spec=_bspec(), soc_init_kwh=6.0, soc_final_kwh=6.0,
        retail_tariff=rt, prices_mwh=pd.Series(prices, index=idx),
        state=STATE, interval_hours=IH,
    )
    # Non-vacuity: the reference uses the battery against the spikes.
    assert reference.bat_to_grid_kwh.iloc[[5, 6, 41]].sum() > 1.0

    traces, diags = _rolling(rt, per, horizon_intervals=len(idx),
                             prices=prices)
    assert diags.causality_ok
    assert diags.sc_fallback_no_forecast == 0
    assert diags.sc_fallback_infeasible == 0

    price_series = pd.Series(prices, index=idx)
    bill_ref = compute_bill(reference, price_series, _net(), rt, STATE, IH)
    bill_got = compute_bill(traces, price_series, _net(), rt, STATE, IH)
    diff = np.abs(reference.export_kwh.to_numpy() - traces.export_kwh.to_numpy())
    ctx = (f"export differs at {np.flatnonzero(diff > 1e-6).tolist()}; "
           f"reference {reference.export_kwh.to_numpy()[diff > 1e-6]}, "
           f"rolling {traces.export_kwh.to_numpy()[diff > 1e-6]}")
    assert bill_got.household_total == pytest.approx(
        bill_ref.household_total, abs=1e-6), ctx
    assert reference.import_kwh.sum() == pytest.approx(
        traces.import_kwh.sum(), abs=1e-6), ctx
    assert reference.export_kwh.sum() == pytest.approx(
        traces.export_kwh.sum(), abs=1e-6), ctx
