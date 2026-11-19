"""Tests for the self-consumption shadow trajectory (Part C, step C2).

The shadow supplies the natural/forced bounds under rolling, where the event
window boundary is not known in advance. It solves no LP, so everything here
is exact arithmetic against run_sc_with_battery.

Several tests assert their own non-vacuity before the real assertion. Part B
found tests that passed while testing nothing, every one because fixture data
never reached the case under test -- and this module has exactly that shape of
state: per-event, accumulating, with a clamp that only fires on some inputs.
"""
import numpy as np
import pytest

from model.core.scenario import BatterySpec
from model.components.self_consumption import run_sc_with_battery
from model.components.vpp_shadow import ShadowTrajectory


INTERVAL_H = 5 / 60


def _bat(capacity=10.0, power=5.0, reserve=10.0, eta=1.0):
    return BatterySpec(
        capacity_kwh=capacity, backup_reserve_pct=reserve, power_kw=power,
        charge_efficiency=eta, discharge_efficiency=eta,
    )


def _shadow(**kw):
    return ShadowTrajectory(_bat(**kw), INTERVAL_H)


# ---------- anchoring and lifecycle ----------

def test_anchor_sets_entry_soc():
    s = _shadow()
    s.anchor(k0=100, soc_kwh=6.0)
    assert s.active
    assert s.k0 == 100
    assert s.soc_prev == pytest.approx(6.0)


def test_anchor_twice_without_settle_raises():
    """A second anchor over a live event would silently discard the first
    event's accrued budget and its committed totals."""
    s = _shadow()
    s.anchor(k0=0, soc_kwh=5.0)
    with pytest.raises(RuntimeError, match='still active'):
        s.anchor(k0=1, soc_kwh=5.0)


def test_anchor_outside_soc_range_raises():
    s = _shadow(capacity=10.0, reserve=10.0)
    with pytest.raises(ValueError, match='outside'):
        s.anchor(k0=0, soc_kwh=11.0)
    with pytest.raises(ValueError, match='outside'):
        s.anchor(k0=0, soc_kwh=0.5)  # below soc_min = 1.0


def test_bounds_commit_settle_require_an_active_event():
    s = _shadow()
    with pytest.raises(RuntimeError, match='no active event'):
        s.bounds(np.zeros(3), np.zeros(3), 3)
    with pytest.raises(RuntimeError, match='no active event'):
        s.commit(0.0, 0.0, 0.0, 0.0)
    with pytest.raises(RuntimeError, match='no active event'):
        s.settle()


def test_settle_releases_the_event():
    s = _shadow()
    s.anchor(k0=0, soc_kwh=5.0)
    s.commit(0.0, 0.1, 0.0, 0.1)
    s.settle()
    assert not s.active
    assert s.soc_prev is None
    s.anchor(k0=10, soc_kwh=5.0)   # a second block may now start


# ---------- bounds match the perfect-foresight formulas ----------

def test_bounds_match_compute_bounds_on_the_same_trajectory():
    """The rolling bounds must be the perfect-foresight formulas evaluated on
    the shadow SOC, not a reimplementation that drifts from them."""
    from model.components.vpp_sc_event import _compute_bounds, _make_soc_prev

    # A deliberate day/night shape rather than random data: the first half
    # has surplus solar (exercising the export ceiling and filling the shadow
    # battery), the second has load with no solar and a battery too drained to
    # cover it (exercising the import budget). Random uniforms happened to
    # produce a fixture that never imported under SC, which the non-vacuity
    # assertions below rejected.
    solar = np.concatenate([np.full(12, 0.9), np.zeros(12)])
    load = np.concatenate([np.full(12, 0.05), np.full(12, 0.9)])
    bat = _bat(capacity=3.0, power=2.0, eta=0.95)
    soc_entry = 1.0

    s = ShadowTrajectory(bat, INTERVAL_H)
    s.anchor(k0=0, soc_kwh=soc_entry)
    got = s.bounds(solar, load, n_eff=24)

    soc_arr, _ = run_sc_with_battery(solar, load, bat, soc_entry, INTERVAL_H)
    soc_prev = _make_soc_prev(soc_arr, soc_entry)
    exp_total, exp_n_exp_hi = _compute_bounds(
        solar, load, soc_prev,
        (bat.backup_reserve_pct / 100.0) * bat.capacity_kwh,
        bat.capacity_kwh, bat.charge_efficiency, bat.discharge_efficiency,
        bat.power_kw * INTERVAL_H,
    )

    # Non-vacuity: a fixture where every bound is zero would compare equal
    # while testing nothing.
    assert exp_n_exp_hi.max() > 0.0, "fixture never exports under SC"
    assert exp_total > 0.0, "fixture never imports under SC"

    np.testing.assert_allclose(got.n_exp_hi, exp_n_exp_hi, atol=1e-12)
    assert got.full_budget == pytest.approx(exp_total)


def test_bounds_does_not_mutate_state():
    """bounds() is called once per interval and its projection is discarded.
    If it advanced the shadow, the trajectory would run ahead of the run."""
    s = _shadow()
    s.anchor(k0=0, soc_kwh=5.0)
    before = s.soc_prev
    solar = np.full(12, 0.5)
    load = np.full(12, 0.1)
    for _ in range(3):
        s.bounds(solar, load, n_eff=12)
    assert s.soc_prev == pytest.approx(before)
    assert s.plan_diagnostics()['event_intervals'] == 0


def test_bounds_rejects_a_window_shorter_than_n_eff():
    s = _shadow()
    s.anchor(k0=0, soc_kwh=5.0)
    with pytest.raises(ValueError, match='shorter than n_eff'):
        s.bounds(np.zeros(4), np.zeros(4), n_eff=6)


# ---------- the committed prefix is never recomputed ----------

def test_committed_prefix_is_extended_not_recomputed():
    """The shadow advances one interval at a time and must land on the same
    trajectory as a single SC run over the whole span. This is what lets an
    event outrun the horizon it was declared in."""
    rng = np.random.default_rng(11)
    solar = rng.uniform(0.0, 0.6, 40)
    load = rng.uniform(0.0, 0.4, 40)
    bat = _bat(eta=0.95)
    soc_entry = 5.0

    s = ShadowTrajectory(bat, INTERVAL_H)
    s.anchor(k0=0, soc_kwh=soc_entry)
    for t in range(40):
        s.commit(solar[t], load[t], n_imp=0.0, import_total=0.0)

    soc_arr, _ = run_sc_with_battery(solar, load, bat, soc_entry, INTERVAL_H)
    assert s.soc_prev == pytest.approx(soc_arr[-1])


def test_event_may_outrun_the_horizon_it_was_declared_in():
    """No bound on event length: committing more intervals than any single
    window covers must work and stay on the SC trajectory."""
    n = 300                      # longer than an 18h horizon at 5-min
    solar = np.full(n, 0.2)
    load = np.full(n, 0.3)
    bat = _bat()
    s = ShadowTrajectory(bat, INTERVAL_H)
    s.anchor(k0=0, soc_kwh=8.0)
    for t in range(n):
        s.commit(solar[t], load[t], n_imp=0.0, import_total=0.0)
    soc_arr, _ = run_sc_with_battery(solar, load, bat, 8.0, INTERVAL_H)
    assert s.soc_prev == pytest.approx(soc_arr[-1])
    assert s.plan_diagnostics()['event_intervals'] == n


# ---------- the running import budget ----------

def test_budget_is_full_span_less_committed():
    """The budget spans the whole expected event, not just the committed part,
    and committed natural import draws it down."""
    solar = np.zeros(12)
    load = np.full(12, 0.4)
    bat = _bat(capacity=10.0, reserve=100.0)   # empty battery: all import
    s = ShadowTrajectory(bat, INTERVAL_H)
    s.anchor(k0=0, soc_kwh=10.0)

    b0 = s.bounds(solar, load, n_eff=12)
    assert b0.full_budget == pytest.approx(12 * 0.4)
    assert b0.n_imp_budget == pytest.approx(12 * 0.4)

    s.commit(solar[0], load[0], n_imp=0.4, import_total=0.4)
    b1 = s.bounds(solar[1:], load[1:], n_eff=11)
    # accrued 0.4 + projected 11*0.4, less 0.4 committed
    assert b1.full_budget == pytest.approx(12 * 0.4)
    assert b1.n_imp_budget == pytest.approx(11 * 0.4)
    assert not b1.clamped


def test_budget_carries_unspent_allowance_forward():
    """Committing zero natural import must leave the allowance intact: the
    aggregate budget exists precisely so import can be shifted in time."""
    solar = np.zeros(12)
    load = np.full(12, 0.4)
    bat = _bat(reserve=100.0)
    s = ShadowTrajectory(bat, INTERVAL_H)
    s.anchor(k0=0, soc_kwh=10.0)
    s.bounds(solar, load, n_eff=12)
    s.commit(solar[0], load[0], n_imp=0.0, import_total=0.4)   # all forced
    b1 = s.bounds(solar[1:], load[1:], n_eff=11)
    assert b1.n_imp_budget == pytest.approx(12 * 0.4)


def test_budget_clamps_at_zero_when_over_committed():
    """The LP may front-load natural import against budget the event never
    delivers. sum(n_imp) <= negative is infeasible against n_imp >= 0, so the
    value is clamped and the condition reported."""
    solar = np.zeros(12)
    load = np.full(12, 0.4)
    bat = _bat(reserve=100.0)
    s = ShadowTrajectory(bat, INTERVAL_H)
    s.anchor(k0=0, soc_kwh=10.0)
    # Book far more natural import than one interval of accrual justifies.
    s.commit(solar[0], load[0], n_imp=4.0, import_total=4.0)
    b = s.bounds(solar[1:2], load[1:2], n_eff=1)

    # Non-vacuity: the raw value must actually be negative, or "clamped at
    # zero" would be indistinguishable from a budget that was zero anyway.
    assert b.full_budget < 4.0, "fixture did not over-commit"
    assert b.clamped
    assert b.n_imp_budget == 0.0


def test_clamp_counter_is_driven_by_the_caller_not_by_bounds():
    """bounds() is pure, so a discarded solve cannot inflate the counter."""
    solar = np.zeros(4)
    load = np.full(4, 0.4)
    s = ShadowTrajectory(_bat(reserve=100.0), INTERVAL_H)
    s.anchor(k0=0, soc_kwh=10.0)
    s.commit(solar[0], load[0], n_imp=9.0, import_total=9.0)
    b = s.bounds(solar[1:], load[1:], n_eff=3)
    assert b.clamped
    assert s.plan_diagnostics()['n_imp_budget_clamps'] == 0
    s.note_clamp()
    assert s.plan_diagnostics()['n_imp_budget_clamps'] == 1


# ---------- retrospective settlement ----------

def test_settle_moves_over_claimed_import_to_forced():
    """When the LP booked more natural import than the event in retrospect
    justified, the excess settles as forced and the household is paid for it."""
    solar = np.zeros(6)
    load = np.full(6, 0.4)
    bat = _bat(reserve=100.0)
    s = ShadowTrajectory(bat, INTERVAL_H)
    s.anchor(k0=0, soc_kwh=10.0)
    # Two intervals only, but 2.0 kWh booked as natural against 0.8 accrued.
    for t in range(2):
        s.commit(solar[t], load[t], n_imp=1.0, import_total=1.0)
    n_imp, f_imp = s.settle()

    assert n_imp == pytest.approx(0.8)      # true budget over the real span
    assert f_imp == pytest.approx(1.2)      # the rest is forced
    assert n_imp + f_imp == pytest.approx(2.0)   # total import preserved
    d = s.plan_diagnostics()
    assert d['n_imp_resettled_kwh'] == pytest.approx(-1.2)
    assert d['blocks_resettled'] == 1


def test_settle_moves_under_claimed_import_to_natural():
    """The correction is symmetric. If the LP booked import as forced against
    a budget the event turned out to deliver, it settles as natural."""
    solar = np.zeros(6)
    load = np.full(6, 0.4)
    s = ShadowTrajectory(_bat(reserve=100.0), INTERVAL_H)
    s.anchor(k0=0, soc_kwh=10.0)
    for t in range(4):
        s.commit(solar[t], load[t], n_imp=0.0, import_total=0.4)
    n_imp, f_imp = s.settle()

    assert n_imp == pytest.approx(1.6)   # all of it was natural after all
    assert f_imp == pytest.approx(0.0)
    assert s.plan_diagnostics()['n_imp_resettled_kwh'] == pytest.approx(1.6)


def test_settle_is_a_noop_when_the_split_was_already_right():
    solar = np.zeros(6)
    load = np.full(6, 0.4)
    s = ShadowTrajectory(_bat(reserve=100.0), INTERVAL_H)
    s.anchor(k0=0, soc_kwh=10.0)
    for t in range(3):
        s.commit(solar[t], load[t], n_imp=0.4, import_total=0.4)
    n_imp, f_imp = s.settle()
    assert n_imp == pytest.approx(1.2)
    assert f_imp == pytest.approx(0.0)
    d = s.plan_diagnostics()
    assert d['n_imp_resettled_kwh'] == pytest.approx(0.0)
    assert d['blocks_resettled'] == 0


def test_settle_never_changes_total_import():
    """Settlement moves volume between n_imp and f_imp only. Total import per
    interval is untouched, so dispatch and the energy component of the bill
    cannot move."""
    rng = np.random.default_rng(3)
    solar = rng.uniform(0.0, 0.5, 20)
    load = rng.uniform(0.0, 0.6, 20)
    s = ShadowTrajectory(_bat(eta=0.95), INTERVAL_H)
    s.anchor(k0=0, soc_kwh=5.0)
    total = 0.0
    for t in range(20):
        imp = float(max(0.0, load[t] - solar[t]))
        total += imp
        s.commit(solar[t], load[t], n_imp=imp * 0.5, import_total=imp)
    n_imp, f_imp = s.settle()
    assert n_imp + f_imp == pytest.approx(total)


# ---------- per-event isolation ----------

def test_a_new_block_does_not_inherit_the_previous_one():
    """Under the un-declaration rule an event can end and another begin
    minutes later. Budget, committed totals and the SOC anchor must all
    reset, or the second block would be settled against the first's accruals."""
    solar = np.zeros(8)
    load = np.full(8, 0.4)
    s = ShadowTrajectory(_bat(reserve=100.0), INTERVAL_H)

    s.anchor(k0=0, soc_kwh=10.0)
    for t in range(3):
        s.commit(solar[t], load[t], n_imp=0.4, import_total=0.4)
    s.settle()

    s.anchor(k0=20, soc_kwh=10.0)
    b = s.bounds(solar, load, n_eff=4)
    assert b.committed_n_imp == 0.0
    assert b.full_budget == pytest.approx(4 * 0.4)   # no carry-in
    n_imp, f_imp = s.settle()
    assert (n_imp, f_imp) == (0.0, 0.0)


def test_block_lengths_are_recorded_for_the_flapping_diagnostic():
    """Whether un-declaration causes flapping is measured, not assumed. A run
    of one- and two-interval blocks is the signal to revisit the rule."""
    s = ShadowTrajectory(_bat(), INTERVAL_H)
    for k0, n in ((0, 1), (10, 5), (30, 1)):
        s.anchor(k0=k0, soc_kwh=5.0)
        for _ in range(n):
            s.commit(0.0, 0.1, 0.0, 0.1)
        s.settle()
    d = s.plan_diagnostics()
    assert d['events_anchored'] == 3
    assert d['event_intervals'] == 7
    assert d['event_block_len_min'] == 1
    assert d['event_block_len_max'] == 5
    assert d['event_blocks_len_1'] == 2


def test_plan_diagnostics_does_not_mutate():
    """Read mid-run without disturbing the counters."""
    s = ShadowTrajectory(_bat(), INTERVAL_H)
    s.anchor(k0=0, soc_kwh=5.0)
    s.commit(0.0, 0.1, 0.0, 0.1)
    first = s.plan_diagnostics()
    second = s.plan_diagnostics()
    assert first == second
    assert s.active                      # the open block is still open
    assert first['event_intervals'] == 1


# ---------- bound properties ----------

def test_export_ceiling_is_zero_when_the_shadow_can_absorb_all_surplus():
    """An empty shadow battery with headroom absorbs the surplus, so SC would
    export nothing and every exported kWh in the LP is forced."""
    solar = np.full(6, 0.3)
    load = np.zeros(6)
    s = ShadowTrajectory(_bat(capacity=100.0, power=50.0), INTERVAL_H)
    s.anchor(k0=0, soc_kwh=10.0)
    b = s.bounds(solar, load, n_eff=6)
    assert np.all(b.n_exp_hi == 0.0)


def test_export_ceiling_opens_once_the_shadow_is_full():
    """Non-vacuity partner to the test above: with no headroom the surplus has
    nowhere to go and SC exports it, so the ceiling must be positive."""
    solar = np.full(6, 0.3)
    load = np.zeros(6)
    s = ShadowTrajectory(_bat(capacity=10.0, power=5.0), INTERVAL_H)
    s.anchor(k0=0, soc_kwh=10.0)          # already at capacity
    b = s.bounds(solar, load, n_eff=6)
    assert np.all(b.n_exp_hi == pytest.approx(0.3))


def test_bounds_are_non_negative():
    rng = np.random.default_rng(19)
    solar = rng.uniform(0.0, 1.0, 50)
    load = rng.uniform(0.0, 1.0, 50)
    s = ShadowTrajectory(_bat(eta=0.9), INTERVAL_H)
    s.anchor(k0=0, soc_kwh=5.0)
    b = s.bounds(solar, load, n_eff=50)
    assert np.all(b.n_exp_hi >= 0.0)
    assert b.n_imp_budget >= 0.0
    assert b.full_budget >= 0.0
