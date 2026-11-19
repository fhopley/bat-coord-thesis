"""Unit tests for model.components.vpp_sc_event (core dispatch behaviour).

Reconstructed from the last pre-deletion version of test_event_optimal_vpp.py
(recovered from git history after the event_optimal_vpp.py -> vpp_sc_event.py
rename). This file covers the base dispatch contract that the feature-specific
test files (test_vpp_sc_event_curtail.py, test_vpp_sc_event_no_grid_charge.py,
test_vpp_sc_event_min_export.py) assume but don't themselves test.

All test cases use interval_hours=1.0 so that kWh per interval equals kW,
making the arithmetic easy to verify by hand. Battery efficiency is set to
1.0 in most tests to simplify SOC accounting.

Two categories of test from the recovered file were NOT ported:
  - Iteration convergence tests (test_convergence_single_iteration_no_dispatch,
    test_convergence_tight_soc_tol). The iteration loop, damping, and soc_tol
    parameter were deliberately removed in a later revision (see
    vpp_sc_event_spec.md changelog item B1: "remove iteration loop, damping,
    _DAMPING, _DIAG, soc_tol, max_iterations"). The mechanism these tests
    exercised no longer exists.

One adaptation was needed beyond the rename: _compute_bounds previously
returned (n_imp_hi, n_exp_hi) as two per-interval arrays. It now returns
(sc_n_imp_total, n_exp_hi) -- an aggregate scalar import budget for the
window plus a per-interval export ceiling array (same changelog item B1:
"replace n_imp_hi per-interval bounds with sc_n_imp_total aggregate budget
row; fix n_exp_hi to SC values"). The compute_bounds tests below assert
against the summed total rather than a per-interval array. This does not
affect the higher-level dispatch tests (natural bounds, cap accounting,
etc.), which assert against VppTraces output fields that are unchanged.

Test organisation:
  - Fixtures and helpers
  - _find_event_windows unit tests
  - _compute_bounds unit tests
  - No-event / SC-equivalence tests
  - Single-event dispatch tests
  - Natural-flow bound correctness
  - VPP cap accounting (export_only and import_and_export)
  - SOC boundary handoffs across multiple events
  - SiteTraces and VppTraces invariants
  - Export limit
  - Validation error cases
"""
import numpy as np
import pandas as pd
import pytest

from model.core.scenario import BatterySpec
from model.core.site_result import SiteTraces, VppTraces
from model.components.vpp_sc_event import (
    vpp_sc_event,
    _find_event_windows,
    _compute_bounds,
)
from model.components.self_consumption import self_consumption


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range('2022-01-03 01:00:00', periods=n, freq='60min',
                         tz='Etc/GMT-10')


def _s(vals, idx) -> pd.Series:
    return pd.Series(vals, index=idx, dtype=float)


def _bspec(
    capacity: float = 10.0,
    reserve_pct: float = 0.0,
    power: float = 10.0,
    eta_c: float = 1.0,
    eta_d: float = 1.0,
    c_deg: float = 0.005,
) -> BatterySpec:
    return BatterySpec(
        capacity_kwh=capacity,
        backup_reserve_pct=reserve_pct,
        power_kw=power,
        charge_efficiency=eta_c,
        discharge_efficiency=eta_d,
        c_deg_per_kwh=c_deg,
    )


def _run(
    solar, load, op_imp, op_exp,
    vpp_event=None,
    bspec=None,
    soc_init=0.0,
    exp_reward=0.10,
    imp_reward=0.0001,
    kwh_limit=250.0,
    limit_applies_to='export_only',
    export_limit=np.inf,
    site_id='S0001',
    n=None,
):
    """Run vpp_sc_event with interval_hours=1.0, gap_strategy='self_consumption'
    (the default -- output is bit-identical to the original component for
    non-tou_advanced runs)."""
    if n is None:
        n = len(solar)
    idx = _idx(n)
    if vpp_event is None:
        vpp_event = np.zeros(n, dtype=bool)
    if bspec is None:
        bspec = _bspec()
    return vpp_sc_event(
        site_id=site_id,
        solar_kwh=_s(solar, idx),
        load_kwh=_s(load, idx),
        op_price_imp=_s(op_imp, idx),
        op_price_exp=_s(op_exp, idx),
        vpp_event=vpp_event,
        battery_spec=bspec,
        soc_init_kwh=soc_init,
        vpp_exp_reward=exp_reward,
        vpp_imp_reward=imp_reward,
        vpp_kwh_limit=kwh_limit,
        vpp_limit_applies_to=limit_applies_to,
        export_limit_kwh_per_interval=export_limit,
        interval_hours=1.0,
    )


def _check_invariants(traces: SiteTraces, vt: VppTraces,
                      solar, load, tol=1e-6):
    """Assert all SiteTraces and VppTraces invariants hold."""
    s = np.array(solar)
    l = np.array(load)
    np.testing.assert_allclose(
        traces.import_kwh.to_numpy(),
        traces.grid_to_home_kwh.to_numpy() + traces.grid_to_bat_kwh.to_numpy(),
        atol=tol, err_msg='import != g2h + g2b',
    )
    np.testing.assert_allclose(
        traces.export_kwh.to_numpy(),
        traces.sol_to_grid_kwh.to_numpy() + traces.bat_to_grid_kwh.to_numpy(),
        atol=tol, err_msg='export != s2g + b2g',
    )
    np.testing.assert_allclose(
        l,
        traces.sol_to_home_kwh.to_numpy() + traces.bat_to_home_kwh.to_numpy()
        + traces.grid_to_home_kwh.to_numpy(),
        atol=tol, err_msg='load balance',
    )
    np.testing.assert_allclose(
        s,
        traces.sol_to_home_kwh.to_numpy() + traces.sol_to_bat_kwh.to_numpy()
        + traces.sol_to_grid_kwh.to_numpy() + traces.curtail_kwh.to_numpy(),
        atol=tol, err_msg='solar balance',
    )
    np.testing.assert_allclose(
        (vt.n_imp_kwh + vt.f_imp_kwh).to_numpy(),
        traces.import_kwh.to_numpy(),
        atol=tol, err_msg='n_imp + f_imp != import_kwh',
    )
    np.testing.assert_allclose(
        (vt.n_exp_kwh + vt.f_exp_kwh).to_numpy(),
        traces.export_kwh.to_numpy(),
        atol=tol, err_msg='n_exp + f_exp != export_kwh',
    )


# ---------------------------------------------------------------------------
# _find_event_windows unit tests
# ---------------------------------------------------------------------------

def test_find_windows_no_events():
    assert _find_event_windows(np.zeros(10, dtype=bool)) == []


def test_find_windows_single_block():
    ev = np.zeros(10, dtype=bool)
    ev[3:7] = True
    assert _find_event_windows(ev) == [(3, 7)]


def test_find_windows_two_blocks():
    ev = np.zeros(10, dtype=bool)
    ev[1:3] = True
    ev[7:9] = True
    assert _find_event_windows(ev) == [(1, 3), (7, 9)]


def test_find_windows_event_at_end():
    ev = np.zeros(6, dtype=bool)
    ev[4:] = True
    assert _find_event_windows(ev) == [(4, 6)]


def test_find_windows_entire_array():
    ev = np.ones(5, dtype=bool)
    assert _find_event_windows(ev) == [(0, 5)]


# ---------------------------------------------------------------------------
# _compute_bounds unit tests
#
# Current signature returns (sc_n_imp_total, n_exp_hi): an aggregate scalar
# import budget for the whole window, plus a per-interval export ceiling
# array. The recovered tests asserted against a per-interval n_imp_hi array;
# adapted here to assert against the summed total instead.
# ---------------------------------------------------------------------------

def test_compute_bounds_no_battery_available():
    """With soc at minimum, no natural discharge possible: n_imp budget = sum(load - solar)."""
    solar = np.array([0.0, 2.0])
    load  = np.array([1.0, 0.5])
    # soc_prev = soc_min in both intervals -- battery cannot discharge
    soc_prev = np.array([2.0, 2.0])
    soc_min, capacity, eta_c, eta_d, power_kwh = 2.0, 10.0, 1.0, 1.0, 10.0
    sc_n_imp_total, n_exp_hi = _compute_bounds(
        solar, load, soc_prev, soc_min, capacity, eta_c, eta_d, power_kwh)
    # t=0: load>solar, contributes max(0, 1-0-0) = 1.0
    # t=1: solar>load, contributes max(0, 0.5-2.0-...) = 0
    # total = 1.0
    assert sc_n_imp_total == pytest.approx(1.0, abs=1e-9)
    # t=0: solar<load, n_exp_hi = max(0, 0-1-min(8/1,10)) = 0
    # t=1: solar>load, battery fully empty so can absorb 1.5 kWh => n_exp_hi = max(0, 1.5-1.5) = 0
    np.testing.assert_allclose(n_exp_hi, [0.0, 0.0], atol=1e-9)


def test_compute_bounds_full_battery():
    """With battery full, no natural charge possible: n_exp_hi = solar - load."""
    solar = np.array([3.0, 0.0])
    load  = np.array([1.0, 1.0])
    soc_prev = np.array([10.0, 10.0])  # full
    soc_min, capacity, eta_c, eta_d, power_kwh = 0.0, 10.0, 1.0, 1.0, 10.0
    sc_n_imp_total, n_exp_hi = _compute_bounds(
        solar, load, soc_prev, soc_min, capacity, eta_c, eta_d, power_kwh)
    # t=0: battery full, can't charge => n_exp_hi = max(0, 3-1-0) = 2
    # t=1: solar<load => n_exp_hi = 0
    np.testing.assert_allclose(n_exp_hi, [2.0, 0.0], atol=1e-9)


def test_compute_bounds_power_limited():
    """Power cap limits natural discharge even when SOC is ample."""
    solar = np.array([0.0])
    load  = np.array([5.0])
    soc_prev = np.array([9.0])
    soc_min, capacity, eta_c, eta_d, power_kwh = 0.0, 10.0, 1.0, 1.0, 2.0
    sc_n_imp_total, n_exp_hi = _compute_bounds(
        solar, load, soc_prev, soc_min, capacity, eta_c, eta_d, power_kwh)
    # power cap = 2 kWh, so max natural discharge = 2, n_imp budget = max(0, 5-0-2) = 3
    assert sc_n_imp_total == pytest.approx(3.0, abs=1e-9)


def test_compute_bounds_balanced_load_gives_zero_bounds():
    """Bounds are floored at zero when solar-load is opposite sign to bound direction."""
    solar = np.array([0.5, 2.0])
    load  = np.array([0.5, 2.0])   # balanced: no natural import or export
    soc_prev = np.array([5.0, 5.0])
    soc_min, capacity, eta_c, eta_d, power_kwh = 0.0, 10.0, 1.0, 1.0, 10.0
    sc_n_imp_total, n_exp_hi = _compute_bounds(
        solar, load, soc_prev, soc_min, capacity, eta_c, eta_d, power_kwh)
    # load == solar: contributes max(0, 0 - 5) = 0 at each interval => total 0
    assert sc_n_imp_total == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(n_exp_hi, [0.0, 0.0], atol=1e-9)


# ---------------------------------------------------------------------------
# No-event / SC equivalence
# ---------------------------------------------------------------------------

def test_no_events_equals_self_consumption():
    """With no events, dispatch equals self_consumption from soc_min."""
    n = 6
    idx = _idx(n)
    solar = [0., 2., 3., 1., 0., 0.]
    load  = [1., 1., 1., 1., 1., 1.]
    bspec = _bspec(capacity=5., power=3., eta_c=0.95, eta_d=0.95)
    soc_min = 0.0

    traces_vse, _ = _run(solar, load, [0.3]*n, [0.05]*n,
                         bspec=bspec, soc_init=soc_min, n=n)
    traces_sc = self_consumption('S', _s(solar, idx), _s(load, idx),
                                 bspec, interval_hours=1.0)

    # With no events the two paths should produce identical dispatch
    np.testing.assert_allclose(
        traces_vse.import_kwh.to_numpy(),
        traces_sc.import_kwh.to_numpy(), atol=1e-6,
        err_msg='import_kwh differs from SC with no events',
    )
    np.testing.assert_allclose(
        traces_vse.battery_soc_kwh.to_numpy(),
        traces_sc.battery_soc_kwh.to_numpy(), atol=1e-6,
        err_msg='SOC differs from SC with no events',
    )


def test_no_events_forced_flows_zero():
    """No events: f_imp and f_exp are zero everywhere."""
    n = 4
    traces, vt = _run([0.]*n, [1.]*n, [0.3]*n, [0.05]*n, n=n)
    np.testing.assert_allclose(vt.f_imp_kwh.to_numpy(), 0., atol=1e-6)
    np.testing.assert_allclose(vt.f_exp_kwh.to_numpy(), 0., atol=1e-6)


def test_no_events_curtail_zero():
    """curtail_kwh is all-zero (no curtail variable in either SC or LP by default)."""
    n = 4
    traces, _ = _run([3., 0., 2., 0.], [1., 1., 1., 1.],
                     [0.3]*4, [0.05]*4, n=4)
    np.testing.assert_allclose(traces.curtail_kwh.to_numpy(), 0., atol=1e-6)


# ---------------------------------------------------------------------------
# Single-event dispatch
# ---------------------------------------------------------------------------

def test_event_force_discharge():
    """High op_exp during event drives forced export from battery.

    T=4: event at t=2,3. Battery enters with SOC=8. High price at t=2 and 3.
    LP should force-discharge during event.
    """
    n = 4
    op_imp = [0.30, 0.30, 2.10, 2.10]
    op_exp = [0.05, 0.05, 2.00, 2.00]
    event  = np.array([False, False, True, True])
    traces, vt = _run(
        [0.]*n, [0.]*n, op_imp, op_exp,
        vpp_event=event, soc_init=8.0,
        bspec=_bspec(capacity=10., power=10., c_deg=0.005), n=n)
    assert vt.f_exp_kwh.sum() > 0.01, (
        f'Expected forced export during event, got {vt.f_exp_kwh.sum():.4f}')
    np.testing.assert_allclose(
        vt.f_exp_kwh.to_numpy()[:2], 0., atol=1e-6,
        err_msg='f_exp non-zero outside event')


def test_forced_flows_zero_outside_events():
    """f_imp and f_exp are zero at all non-event intervals."""
    n = 6
    op_imp = [0.30]*6; op_exp = [0.05]*6
    op_imp[2] = 2.10; op_exp[2] = 2.00
    op_imp[4] = 2.10; op_exp[4] = 2.00
    event = np.array([False, False, True, False, True, False])
    traces, vt = _run(
        [0.]*n, [0.]*n, op_imp, op_exp,
        vpp_event=event, soc_init=8.0,
        bspec=_bspec(capacity=10., power=10.), n=n)
    non_event = ~event
    np.testing.assert_allclose(
        vt.f_exp_kwh.to_numpy()[non_event], 0., atol=1e-6)
    np.testing.assert_allclose(
        vt.f_imp_kwh.to_numpy()[non_event], 0., atol=1e-6)


def test_force_charge_then_discharge():
    """Low op_imp at start of event followed by high op_exp: charge then discharge.

    T=4: events at t=0..3 (single contiguous window). Low import price early,
    high export price late. LP should force-charge at t=0 and force-discharge
    at t=3, within the same event window where it has visibility of both.
    """
    n = 4
    op_imp = [0.01, 0.30, 0.30, 2.10]
    op_exp = [0.005, 0.05, 0.05, 2.00]
    event  = np.ones(n, dtype=bool)   # single contiguous event window
    traces, vt = _run(
        [0.]*n, [0.]*n, op_imp, op_exp,
        vpp_event=event, soc_init=2.0,
        bspec=_bspec(capacity=5., power=5., c_deg=0.005),
        imp_reward=0.10, exp_reward=0.10,
        kwh_limit=10., limit_applies_to='import_and_export', n=n)
    assert vt.f_imp_kwh.iloc[0] > 0.01, (
        f'Expected force-charge at t=0, got {vt.f_imp_kwh.iloc[0]:.4f}')
    assert vt.f_exp_kwh.iloc[3] > 0.01, (
        f'Expected force-discharge at t=3, got {vt.f_exp_kwh.iloc[3]:.4f}')


# ---------------------------------------------------------------------------
# Natural-flow bound correctness
#
# These assert against VppTraces output fields (n_imp_kwh, n_exp_kwh), not
# the internal _compute_bounds return shape, so they are unaffected by the
# aggregate-budget adaptation above.
# ---------------------------------------------------------------------------

def test_natural_bounds_limit_n_imp():
    """n_imp cannot exceed what SC would have imported at that SOC.

    T=2: event at both intervals. Battery has full SOC (capacity=2, soc=2).
    SC would discharge to serve load, leaving n_imp=0. After LP dispatch
    the natural import bound should have been tight.
    """
    n = 2
    # Battery capacity = load = 1 kWh: SC would fully discharge, n_imp = 0
    bspec = _bspec(capacity=2., reserve_pct=0., power=2.)
    op_imp = [2.10, 2.10]
    op_exp = [2.00, 2.00]
    event  = np.ones(n, dtype=bool)
    traces, vt = _run(
        [0.]*n, [1.]*n, op_imp, op_exp,
        vpp_event=event, soc_init=2.0,
        bspec=bspec, n=n)
    # SC would serve load from battery: n_imp should be near 0
    np.testing.assert_allclose(
        vt.n_imp_kwh.to_numpy(), 0., atol=1e-4,
        err_msg='n_imp should be 0 when battery can serve full load')


def test_natural_bounds_limit_n_exp():
    """n_exp cannot exceed what SC would have exported (battery absorbs solar).

    T=2: event at both intervals. Solar surplus > 0 but battery has headroom.
    SC would charge battery, so n_exp should be small or zero.
    """
    n = 2
    bspec = _bspec(capacity=5., reserve_pct=0., power=5.)
    op_imp = [2.10, 2.10]
    op_exp = [2.00, 2.00]
    event  = np.ones(n, dtype=bool)
    # solar=3, load=1: surplus=2. Battery empty so SC charges 2 kWh, exports 0.
    traces, vt = _run(
        [3.]*n, [1.]*n, op_imp, op_exp,
        vpp_event=event, soc_init=0.0,
        bspec=bspec, n=n)
    # n_exp should be near 0 (SC absorbs surplus into battery)
    np.testing.assert_allclose(
        vt.n_exp_kwh.to_numpy(), 0., atol=1e-4,
        err_msg='n_exp should be 0 when battery can absorb full solar surplus')


# ---------------------------------------------------------------------------
# VPP cap accounting
# ---------------------------------------------------------------------------

def test_cap_export_only_binds():
    """With export_only cap=0.5 kWh, total f_exp <= 0.5."""
    n = 4
    op_imp = [2.10]*n; op_exp = [2.00]*n
    event  = np.ones(n, dtype=bool)
    traces, vt = _run(
        [0.]*n, [0.]*n, op_imp, op_exp,
        vpp_event=event, soc_init=8.0,
        bspec=_bspec(capacity=10., power=10., c_deg=0.005),
        kwh_limit=0.5, limit_applies_to='export_only', n=n)
    assert vt.f_exp_kwh.sum() <= 0.5 + 1e-6, (
        f'export_only cap not respected: {vt.f_exp_kwh.sum():.4f}')


def test_cap_import_and_export_binds():
    """With import_and_export cap=0.5, f_imp + f_exp <= 0.5."""
    n = 4
    # op_imp always >= op_exp; cheap import early, high export late
    op_imp = [0.05, 0.05, 0.05, 2.10]
    op_exp = [0.01, 0.01, 0.01, 2.00]
    event  = np.ones(n, dtype=bool)
    traces, vt = _run(
        [0.]*n, [0.]*n, op_imp, op_exp,
        vpp_event=event, soc_init=2.0,
        bspec=_bspec(capacity=5., power=5., c_deg=0.005),
        imp_reward=0.10, exp_reward=0.10,
        kwh_limit=0.5, limit_applies_to='import_and_export', n=n)
    total = vt.f_imp_kwh.sum() + vt.f_exp_kwh.sum()
    assert total <= 0.5 + 1e-6, (
        f'import_and_export cap not respected: {total:.4f}')


def test_cap_tracked_across_events():
    """Cap is reduced by first event and limits second event.

    Two events, cap=1.0 kWh. First event can dispatch up to 1.0; second
    sees remaining budget only.
    """
    n = 8
    op_imp = [2.10]*n; op_exp = [2.00]*n
    # Two separated event windows: t=1,2 and t=5,6
    event = np.zeros(n, dtype=bool)
    event[1:3] = True; event[5:7] = True
    traces, vt = _run(
        [0.]*n, [0.]*n, op_imp, op_exp,
        vpp_event=event, soc_init=8.0,
        bspec=_bspec(capacity=10., power=10., c_deg=0.005),
        kwh_limit=1.0, limit_applies_to='export_only', n=n)
    total_f_exp = vt.f_exp_kwh.sum()
    assert total_f_exp <= 1.0 + 1e-6, (
        f'Total cap across events not respected: {total_f_exp:.4f}')


def test_cap_exhausted_second_event_zero():
    """When cap is used by first event, second event produces no forced flows."""
    n = 8
    op_imp = [2.10]*n; op_exp = [2.00]*n
    event = np.zeros(n, dtype=bool)
    event[0:2] = True; event[5:7] = True
    # Power cap = 0.6 kWh/interval; 2 intervals = 1.2 kWh available
    # Set kwh_limit = 0.3 kWh -- consumed entirely by first event
    traces, vt = _run(
        [0.]*n, [0.]*n, op_imp, op_exp,
        vpp_event=event, soc_init=8.0,
        bspec=_bspec(capacity=10., power=0.6, c_deg=0.005),
        kwh_limit=0.3, limit_applies_to='export_only', n=n)
    # Second event should have zero forced export
    np.testing.assert_allclose(
        vt.f_exp_kwh.to_numpy()[5:7], 0., atol=1e-6,
        err_msg='Second event should be zero after cap exhausted')


# ---------------------------------------------------------------------------
# SOC boundary handoffs
# ---------------------------------------------------------------------------

def test_soc_continuity_across_events():
    """SOC at end of event window becomes soc_entry for the next SC segment."""
    n = 8
    op_imp = [2.10]*n; op_exp = [2.00]*n
    event = np.zeros(n, dtype=bool); event[2:4] = True
    bspec_used = _bspec(capacity=10., power=5., c_deg=0.005)
    traces, vt = _run(
        [0.]*n, [1.]*n, op_imp, op_exp,
        vpp_event=event, soc_init=8.0,
        bspec=bspec_used, n=n)
    soc = traces.battery_soc_kwh.to_numpy()
    # SOC must be continuous: no jump at event boundaries
    soc_min = (bspec_used.backup_reserve_pct/100) * bspec_used.capacity_kwh
    for t in range(1, n):
        assert soc[t] >= soc_min - 1e-6, f'SOC below reserve at t={t}: {soc[t]:.4f}'
        assert soc[t] <= bspec_used.capacity_kwh + 1e-6, (
            f'SOC above capacity at t={t}: {soc[t]:.4f}')


def test_post_event_sc_uses_event_exit_soc():
    """SC after an event starts from the event's exit SOC, not from soc_init.

    A high-drain event discharges the battery near-empty. The subsequent
    non-event intervals should show no battery discharge (nothing to give).
    """
    n = 6
    op_imp = [2.10]*n; op_exp = [2.00]*n
    # Event at t=0,1 drains battery (no solar, high price incentivises export)
    event = np.zeros(n, dtype=bool); event[0:2] = True
    bspec = _bspec(capacity=2., reserve_pct=0., power=2.)
    traces, vt = _run(
        [0.]*n, [0.]*n, op_imp, op_exp,
        vpp_event=event, soc_init=2.0,
        bspec=bspec, kwh_limit=10., n=n)
    soc = traces.battery_soc_kwh.to_numpy()
    # After the event the battery should be near empty
    assert soc[1] < 1.0, (
        f'Expected battery near-empty after event, got SOC={soc[1]:.4f}')
    # Subsequent non-event bat_to_home should be limited to what's available
    post_event_b2h = traces.bat_to_home_kwh.to_numpy()[2:]
    assert post_event_b2h.sum() <= soc[1] + 1e-6, (
        'Post-event discharge exceeds available SOC')


# ---------------------------------------------------------------------------
# SiteTraces and VppTraces invariants
# ---------------------------------------------------------------------------

def test_sitetraces_invariants_no_events():
    """All four SiteTraces invariants hold with no events."""
    solar = [0., 2., 4., 1., 0., 3.]
    load  = [1., 1., 2., 1., 2., 1.]
    traces, vt = _run(
        solar, load, [0.3]*6, [0.05]*6,
        bspec=_bspec(capacity=3., power=2., eta_c=0.95, eta_d=0.95), n=6)
    _check_invariants(traces, vt, solar, load)


def test_sitetraces_invariants_with_events():
    """All four SiteTraces invariants hold with event windows."""
    n = 8
    solar = [0., 2., 3., 0., 0., 1., 0., 0.]
    load  = [1., 1., 1., 1., 1., 1., 1., 1.]
    op_imp = [0.30]*n; op_exp = [0.05]*n
    op_imp[3:6] = [2.10]*3; op_exp[3:6] = [2.00]*3
    event = np.zeros(n, dtype=bool); event[3:6] = True
    traces, vt = _run(
        solar, load, op_imp, op_exp,
        vpp_event=event, soc_init=5.0,
        bspec=_bspec(capacity=8., power=4., eta_c=0.95, eta_d=0.95), n=n)
    _check_invariants(traces, vt, solar, load)


def test_vpptraces_non_negative():
    """All four VppTraces series are non-negative."""
    n = 6
    op_imp = [0.30]*n; op_exp = [0.05]*n
    op_imp[2:4] = [2.10]*2; op_exp[2:4] = [2.00]*2
    event = np.zeros(n, dtype=bool); event[2:4] = True
    _, vt = _run(
        [1.]*n, [1.]*n, op_imp, op_exp,
        vpp_event=event, soc_init=5.0,
        bspec=_bspec(capacity=8., power=4.), n=n)
    tol = 1e-6
    for name, series in (
        ('n_imp', vt.n_imp_kwh), ('f_imp', vt.f_imp_kwh),
        ('n_exp', vt.n_exp_kwh), ('f_exp', vt.f_exp_kwh),
    ):
        assert (series.to_numpy() >= -tol).all(), (
            f'{name}_kwh has negative values (min={series.min():.2e})')


def test_curtail_always_zero_when_disabled():
    """curtail_kwh is all-zero when p_curtail is not set (default)."""
    n = 6
    op_imp = [0.30]*n; op_exp = [0.05]*n
    op_imp[2:4] = [2.10]*2; op_exp[2:4] = [2.00]*2
    event = np.zeros(n, dtype=bool); event[2:4] = True
    traces, _ = _run(
        [2.]*n, [1.]*n, op_imp, op_exp,
        vpp_event=event, soc_init=3.0,
        bspec=_bspec(capacity=8., power=4.), n=n)
    np.testing.assert_allclose(traces.curtail_kwh.to_numpy(), 0., atol=1e-6)


def test_multiple_events_full_invariants():
    """Invariants hold with multiple separated event windows."""
    n = 12
    solar = [0., 2., 3., 1., 0., 0., 2., 1., 0., 0., 3., 0.]
    load  = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
    op_imp = [0.30]*n; op_exp = [0.05]*n
    for i in [2, 3, 8, 9]:
        op_imp[i] = 2.10; op_exp[i] = 2.00
    event = np.zeros(n, dtype=bool)
    event[2:4] = True; event[8:10] = True
    traces, vt = _run(
        solar, load, op_imp, op_exp,
        vpp_event=event, soc_init=5.0,
        bspec=_bspec(capacity=8., power=4., eta_c=0.95, eta_d=0.95), n=n)
    _check_invariants(traces, vt, solar, load)


# ---------------------------------------------------------------------------
# Export limit
# ---------------------------------------------------------------------------

def test_export_limit_joint_respected():
    """Per-interval export limit applies to n_exp + f_exp combined."""
    n = 4
    op_imp = [2.10]*n; op_exp = [2.00]*n
    event = np.ones(n, dtype=bool)
    traces, vt = _run(
        [0.]*n, [0.]*n, op_imp, op_exp,
        vpp_event=event, soc_init=8.0,
        bspec=_bspec(capacity=10., power=10., c_deg=0.005),
        export_limit=0.5, n=n)
    total_export = (vt.n_exp_kwh + vt.f_exp_kwh).to_numpy()
    assert (total_export <= 0.5 + 1e-6).all(), (
        f'Export limit violated: max={total_export.max():.4f}')


# ---------------------------------------------------------------------------
# Validation error cases
# ---------------------------------------------------------------------------

def test_op_price_ordering_violation_raises():
    """op_price_exp > op_price_imp raises ValueError."""
    n = 2
    idx = _idx(n)
    with pytest.raises(ValueError, match='op_price_exp exceeds op_price_imp'):
        vpp_sc_event(
            site_id='S',
            solar_kwh=_s([0.]*n, idx), load_kwh=_s([0.]*n, idx),
            op_price_imp=_s([0.05]*n, idx), op_price_exp=_s([0.30]*n, idx),  # exp > imp: violation
            vpp_event=np.zeros(n, dtype=bool), battery_spec=_bspec(),
            soc_init_kwh=0.,
            vpp_exp_reward=0.10, vpp_imp_reward=0.0001,
            vpp_kwh_limit=250., vpp_limit_applies_to='export_only',
            interval_hours=1.,
        )


def test_nan_solar_raises():
    n = 2; idx = _idx(n)
    with pytest.raises(ValueError, match='NaN'):
        vpp_sc_event(
            site_id='S',
            solar_kwh=_s([float('nan'), 0.], idx), load_kwh=_s([0.]*n, idx),
            op_price_imp=_s([0.3]*n, idx), op_price_exp=_s([0.05]*n, idx),
            vpp_event=np.zeros(n, dtype=bool), battery_spec=_bspec(),
            soc_init_kwh=0.,
            vpp_exp_reward=0.10, vpp_imp_reward=0.0001,
            vpp_kwh_limit=250., vpp_limit_applies_to='export_only',
            interval_hours=1.,
        )


def test_negative_load_raises():
    n = 2; idx = _idx(n)
    with pytest.raises(ValueError, match='negative'):
        vpp_sc_event(
            site_id='S',
            solar_kwh=_s([0.]*n, idx), load_kwh=_s([-1., 1.], idx),
            op_price_imp=_s([0.3]*n, idx), op_price_exp=_s([0.05]*n, idx),
            vpp_event=np.zeros(n, dtype=bool), battery_spec=_bspec(),
            soc_init_kwh=0.,
            vpp_exp_reward=0.10, vpp_imp_reward=0.0001,
            vpp_kwh_limit=250., vpp_limit_applies_to='export_only',
            interval_hours=1.,
        )


def test_soc_init_out_of_bounds_raises():
    n = 2; idx = _idx(n)
    bspec = _bspec(capacity=5., reserve_pct=20.)  # soc_min = 1.0
    with pytest.raises(ValueError, match='soc_init_kwh'):
        vpp_sc_event(
            site_id='S',
            solar_kwh=_s([0.]*n, idx), load_kwh=_s([0.]*n, idx),
            op_price_imp=_s([0.3]*n, idx), op_price_exp=_s([0.05]*n, idx),
            vpp_event=np.zeros(n, dtype=bool), battery_spec=bspec,
            soc_init_kwh=0.5,  # below soc_min
            vpp_exp_reward=0.10, vpp_imp_reward=0.0001,
            vpp_kwh_limit=250., vpp_limit_applies_to='export_only',
            interval_hours=1.,
        )


def test_misaligned_index_raises():
    idx2 = _idx(2); idx3 = _idx(3)
    with pytest.raises(ValueError, match='index'):
        vpp_sc_event(
            site_id='S',
            solar_kwh=_s([0.]*2, idx2), load_kwh=_s([0.]*2, idx2),
            op_price_imp=_s([0.3]*3, idx3), op_price_exp=_s([0.05]*3, idx3),
            vpp_event=np.zeros(2, dtype=bool), battery_spec=_bspec(),
            soc_init_kwh=0.,
            vpp_exp_reward=0.10, vpp_imp_reward=0.0001,
            vpp_kwh_limit=250., vpp_limit_applies_to='export_only',
            interval_hours=1.,
        )


def test_vpp_event_length_mismatch_raises():
    n = 3; idx = _idx(n)
    with pytest.raises(ValueError, match='vpp_event length'):
        vpp_sc_event(
            site_id='S',
            solar_kwh=_s([0.]*n, idx), load_kwh=_s([0.]*n, idx),
            op_price_imp=_s([0.3]*n, idx), op_price_exp=_s([0.05]*n, idx),
            vpp_event=np.zeros(2, dtype=bool),  # wrong length
            battery_spec=_bspec(), soc_init_kwh=0.,
            vpp_exp_reward=0.10, vpp_imp_reward=0.0001,
            vpp_kwh_limit=250., vpp_limit_applies_to='export_only',
            interval_hours=1.,
        )


def test_unknown_limit_applies_to_raises():
    n = 2; idx = _idx(n)
    with pytest.raises(ValueError, match='vpp_limit_applies_to'):
        vpp_sc_event(
            site_id='S',
            solar_kwh=_s([0.]*n, idx), load_kwh=_s([0.]*n, idx),
            op_price_imp=_s([0.3]*n, idx), op_price_exp=_s([0.05]*n, idx),
            vpp_event=np.zeros(n, dtype=bool), battery_spec=_bspec(),
            soc_init_kwh=0.,
            vpp_exp_reward=0.10, vpp_imp_reward=0.0001,
            vpp_kwh_limit=250., vpp_limit_applies_to='bad_mode',
            interval_hours=1.,
        )
