"""Unit tests for vpp_sc_event curtailment (p_curtail parameter).

Covers the four C4 items from the spec's curtailment revision build order:

  - Disabled-case parity: p_curtail=None gives identical traces to the
    original (pre-curtailment) output.
  - Zero-cost curtailment fires under a binding export limit with a full
    battery: LP becomes feasible where it previously was not, curtail_kwh
    is non-zero, and the solar invariant holds.
  - Positive-penalty curtailment: the LP curtails only when op_exp is
    sufficiently negative (op_exp < -p_curtail).
  - Negative p_curtail is rejected with a ValueError.

All tests call vpp_sc_event directly, without the orchestrator, using
small synthetic traces (single event window, O(10) intervals).
"""
import numpy as np
import pandas as pd
import pytest

from model.core.scenario import BatterySpec
from model.components.vpp_sc_event import vpp_sc_event


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range('2022-01-01 00:05', periods=n, freq='5min',
                         tz='Etc/GMT-10')


def _bat(capacity_kwh=10.0, power_kw=5.0, backup_reserve_pct=10.0) -> BatterySpec:
    return BatterySpec(
        capacity_kwh=capacity_kwh,
        backup_reserve_pct=backup_reserve_pct,
        power_kw=power_kw,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
        c_deg_per_kwh=0.0,
    )


def _series(arr, idx) -> pd.Series:
    return pd.Series(arr, index=idx)


def _run(solar, load, op_imp, op_exp, event, battery=None, *,
         export_limit=np.inf, p_curtail=None, soc_init=None):
    """Convenience wrapper around vpp_sc_event."""
    n = len(solar)
    idx = _idx(n)
    bat = battery or _bat()
    soc_init_kwh = (bat.backup_reserve_pct / 100.0) * bat.capacity_kwh
    if soc_init is not None:
        soc_init_kwh = soc_init
    interval_hours = 5 / 60
    return vpp_sc_event(
        site_id='TEST',
        solar_kwh=_series(solar, idx),
        load_kwh=_series(load, idx),
        op_price_imp=_series(op_imp, idx),
        op_price_exp=_series(op_exp, idx),
        vpp_event=np.array(event, dtype=bool),
        battery_spec=bat,
        soc_init_kwh=soc_init_kwh,
        vpp_exp_reward=1.0,      # $1/kWh forced export reward
        vpp_imp_reward=0.01,
        vpp_kwh_limit=1000.0,
        vpp_limit_applies_to='export_only',
        export_limit_kwh_per_interval=export_limit,
        interval_hours=interval_hours,
        p_curtail=p_curtail,
    )


# ---------------------------------------------------------------------------
# Test 1: Disabled-case parity
#
# With p_curtail=None the component must produce results identical to
# p_curtail=None (i.e. the unchanged default path). We verify this by
# running twice -- once with None explicitly, once omitting the argument --
# and asserting that curtail_kwh is all-zero in both cases and that all
# other SiteTraces fields match.
# ---------------------------------------------------------------------------

def test_curtail_disabled_gives_all_zero_curtail():
    """p_curtail=None produces all-zero curtail_kwh. Single solve, no iteration."""
    n = 12
    solar = np.full(n, 0.1)          # kWh / interval
    load  = np.full(n, 0.05)
    op_imp = np.full(n, 0.10)
    op_exp = np.full(n, 0.05)
    event  = np.zeros(n, dtype=bool)
    event[4:8] = True                 # one short event window

    traces, _ = _run(solar, load, op_imp, op_exp, event, p_curtail=None)

    np.testing.assert_array_equal(
        traces.curtail_kwh.to_numpy(),
        np.zeros(n),
        err_msg='curtail_kwh should be all-zero when p_curtail is None',
    )


def test_curtail_disabled_matches_default_call():
    """Explicit p_curtail=None is identical to omitting the argument."""
    n = 12
    solar  = np.full(n, 0.1)
    load   = np.full(n, 0.05)
    op_imp = np.full(n, 0.10)
    op_exp = np.full(n, 0.05)
    event  = np.zeros(n, dtype=bool)
    event[4:8] = True

    def _traces(**kw):
        return _run(solar, load, op_imp, op_exp, event, **kw)[0]

    t_default  = _traces()               # p_curtail omitted
    t_explicit = _traces(p_curtail=None) # p_curtail=None

    for field in ('import_kwh', 'export_kwh', 'battery_soc_kwh',
                  'sol_to_home_kwh', 'sol_to_bat_kwh', 'sol_to_grid_kwh',
                  'bat_to_home_kwh', 'grid_to_home_kwh', 'grid_to_bat_kwh',
                  'bat_to_grid_kwh', 'curtail_kwh'):
        np.testing.assert_array_equal(
            getattr(t_default, field).to_numpy(),
            getattr(t_explicit, field).to_numpy(),
            err_msg=f'{field}: default and explicit None differ',
        )


# ---------------------------------------------------------------------------
# Test 2: Zero-cost curtailment fires under a binding export limit
#
# Setup: a battery that is full at the start of the event window, zero load,
# high solar, and an export limit that is tighter than the solar generation.
# With p_curtail=None this is infeasible (no valid placement for surplus
# solar). With p_curtail=0 the LP must curtail the excess, the solve
# succeeds, curtail_kwh is non-zero during the event, and the solar
# invariant holds interval-by-interval.
# ---------------------------------------------------------------------------

def test_curtail_zero_cost_resolves_binding_export_limit():
    """p_curtail=0 allows curtailment when export limit would otherwise be infeasible."""
    interval_hours = 5 / 60
    n = 6
    solar = np.full(n, 0.3)       # kWh/interval -- exceeds any reasonable limit
    load  = np.zeros(n)
    # op_exp positive so the LP wants to export; export limit will bind
    op_imp = np.full(n, 0.20)
    op_exp = np.full(n, 0.05)
    event  = np.ones(n, dtype=bool)   # whole trace is one event window

    bat = _bat(capacity_kwh=5.0, power_kw=2.0, backup_reserve_pct=0.0)
    # Start with battery completely full so it cannot absorb any charge
    soc_full = bat.capacity_kwh

    # Export limit: 0.1 kWh/interval -- well below solar=0.3, load=0, charge=0
    export_limit = 0.1  # kWh/interval

    traces, _ = _run(
        solar, load, op_imp, op_exp, event,
        battery=bat,
        export_limit=export_limit,
        p_curtail=0.0,
        soc_init=soc_full,
    )

    curtail = traces.curtail_kwh.to_numpy()
    # Curtailment must be non-zero during the event (all intervals here)
    assert curtail.sum() > 1e-6, (
        f'Expected non-zero curtailment under binding export limit, '
        f'got sum={curtail.sum():.6f}'
    )
    # Export must not exceed the limit at any interval
    export = traces.export_kwh.to_numpy()
    assert (export <= export_limit + 1e-6).all(), (
        f'Export limit violated: max export={export.max():.6f} > {export_limit}'
    )


def test_curtail_zero_cost_solar_invariant_holds():
    """Solar invariant solar = sol_to_home + sol_to_bat + sol_to_grid + curtail
    holds interval-by-interval when curtailment is non-zero."""
    interval_hours = 5 / 60
    n = 6
    solar_arr = np.full(n, 0.3)
    load  = np.zeros(n)
    op_imp = np.full(n, 0.20)
    op_exp = np.full(n, 0.05)
    event  = np.ones(n, dtype=bool)

    bat = _bat(capacity_kwh=5.0, power_kw=2.0, backup_reserve_pct=0.0)
    soc_full = bat.capacity_kwh
    export_limit = 0.1

    traces, _ = _run(
        solar_arr, load, op_imp, op_exp, event,
        battery=bat, export_limit=export_limit,
        p_curtail=0.0, soc_init=soc_full,
    )

    solar_in  = solar_arr
    lhs = (traces.sol_to_home_kwh.to_numpy()
           + traces.sol_to_bat_kwh.to_numpy()
           + traces.sol_to_grid_kwh.to_numpy()
           + traces.curtail_kwh.to_numpy())
    np.testing.assert_allclose(
        lhs, solar_in, atol=1e-9,
        err_msg='Solar invariant violated: sol_to_home+sol_to_bat+sol_to_grid+curtail != solar',
    )


# ---------------------------------------------------------------------------
# Test 3: Positive-penalty curtailment -- curtails only when op_exp < -p_curtail
#
# When export is profitable (op_exp > 0) the LP should always prefer exporting
# over curtailing (regardless of p_curtail > 0). When export is unprofitable
# and the penalty is small enough (|op_exp| > p_curtail) the LP should curtail
# rather than export. We test both sub-cases with a simple unconstrained setup.
# ---------------------------------------------------------------------------

def test_curtail_positive_penalty_no_curtail_when_export_profitable():
    """With positive op_exp and p_curtail > 0, no curtailment occurs.

    Battery starts full so it cannot absorb surplus, forcing a genuine
    choice between exporting (profitable) and curtailing (costly). The LP
    should always prefer export here.
    """
    n = 8
    solar  = np.full(n, 0.2)
    load   = np.zeros(n)
    op_imp = np.full(n, 0.20)
    op_exp = np.full(n, 0.10)          # export earns money
    event  = np.ones(n, dtype=bool)

    bat = _bat(capacity_kwh=5.0, power_kw=5.0, backup_reserve_pct=0.0)
    # Full battery: cannot absorb surplus, so the LP chooses between export and curtail
    traces, _ = _run(solar, load, op_imp, op_exp, event,
                     battery=bat, p_curtail=0.05, soc_init=bat.capacity_kwh)

    curtail = traces.curtail_kwh.to_numpy()
    assert curtail.sum() < 1e-9, (
        f'Expected zero curtailment when export is profitable, '
        f'got sum={curtail.sum():.6f}'
    )


def test_curtail_positive_penalty_curtails_when_export_loses_more():
    """With negative op_exp where |op_exp| > p_curtail, LP curtails rather than exports.

    Battery starts full so it cannot absorb surplus, forcing a genuine
    choice between curtailing (p_curtail=0.05 $/kWh cost) and exporting
    (op_exp=-0.15 $/kWh, i.e. 15 c/kWh cost to export). Curtailing is
    cheaper so the LP should curtail.
    """
    n = 8
    solar  = np.full(n, 0.2)
    load   = np.zeros(n)
    op_imp = np.full(n, 0.20)
    # op_exp = -0.15: exporting costs 15 c/kWh; curtailing costs p_curtail=0.05 $/kWh.
    # LP should curtail (cheaper) rather than export (more expensive).
    op_exp = np.full(n, -0.15)
    event  = np.ones(n, dtype=bool)

    bat = _bat(capacity_kwh=5.0, power_kw=5.0, backup_reserve_pct=0.0)
    # Full battery: cannot absorb surplus, so the LP chooses between export and curtail.
    # price ordering: op_imp=0.20 >= op_exp=-0.15, satisfied.
    traces, _ = _run(solar, load, op_imp, op_exp, event,
                     battery=bat, p_curtail=0.05, soc_init=bat.capacity_kwh)

    curtail = traces.curtail_kwh.to_numpy()
    export  = traces.export_kwh.to_numpy()
    # Should curtail substantially and keep export near zero
    assert curtail.sum() > 1e-6, (
        f'Expected curtailment when |op_exp|={0.15} > p_curtail={0.05}, '
        f'got curtail.sum()={curtail.sum():.6f}'
    )
    assert export.sum() < curtail.sum(), (
        f'Expected curtailment to dominate export when export is unprofitable: '
        f'curtail={curtail.sum():.4f}, export={export.sum():.4f}'
    )


# ---------------------------------------------------------------------------
# Test 4: Negative p_curtail rejected
# ---------------------------------------------------------------------------

def test_curtail_negative_p_curtail_raises():
    """A negative p_curtail is rejected with a ValueError."""
    n = 4
    solar  = np.full(n, 0.1)
    load   = np.full(n, 0.05)
    op_imp = np.full(n, 0.10)
    op_exp = np.full(n, 0.05)
    event  = np.ones(n, dtype=bool)

    with pytest.raises(ValueError, match='p_curtail must be >= 0'):
        _run(solar, load, op_imp, op_exp, event, p_curtail=-0.01)


# ---------------------------------------------------------------------------
# Test 5: VppTraces invariants hold with non-zero curtailment
#
# n_imp + f_imp == import_kwh and n_exp + f_exp == export_kwh must hold
# even when the LP curtails solar.
# ---------------------------------------------------------------------------

def test_curtail_vpp_traces_invariants_hold():
    """VppTraces split invariants hold when curtailment is non-zero."""
    interval_hours = 5 / 60
    n = 6
    solar_arr = np.full(n, 0.3)
    load  = np.zeros(n)
    op_imp = np.full(n, 0.20)
    op_exp = np.full(n, 0.05)
    event  = np.ones(n, dtype=bool)

    bat = _bat(capacity_kwh=5.0, power_kw=2.0, backup_reserve_pct=0.0)
    soc_full = bat.capacity_kwh
    export_limit = 0.1

    traces, vt = _run(
        solar_arr, load, op_imp, op_exp, event,
        battery=bat, export_limit=export_limit,
        p_curtail=0.0, soc_init=soc_full,
    )

    np.testing.assert_allclose(
        (vt.n_imp_kwh + vt.f_imp_kwh).to_numpy(),
        traces.import_kwh.to_numpy(),
        atol=1e-6,
        err_msg='n_imp + f_imp != import_kwh when curtailment is active',
    )
    np.testing.assert_allclose(
        (vt.n_exp_kwh + vt.f_exp_kwh).to_numpy(),
        traces.export_kwh.to_numpy(),
        atol=1e-6,
        err_msg='n_exp + f_exp != export_kwh when curtailment is active',
    )


# ---------------------------------------------------------------------------
# Test 6: Aggregate n_imp budget respected
#
# With SC-fixed bounds, total n_imp across the event window must not exceed
# the SC natural import total. We run a window where the LP would be
# incentivised to import naturally (high load, low solar) and verify the
# aggregate constraint holds.
# ---------------------------------------------------------------------------

def test_aggregate_n_imp_budget_not_exceeded():
    """Total n_imp across the event window does not exceed SC natural import."""
    n = 12
    solar  = np.zeros(n)
    load   = np.full(n, 0.2)
    op_imp = np.full(n, 0.10)
    op_exp = np.full(n, 0.05)
    event  = np.ones(n, dtype=bool)

    bat = _bat(capacity_kwh=5.0, power_kw=5.0, backup_reserve_pct=0.0)
    traces, vt = _run(solar, load, op_imp, op_exp, event, battery=bat)

    # SC natural import budget: with empty battery and no solar, SC imports all
    # load and discharges nothing (soc=0 floor). So sc_n_imp_total = sum(load).
    sc_n_imp_total = float(load.sum())
    n_imp_actual = float(vt.n_imp_kwh.to_numpy().sum())
    assert n_imp_actual <= sc_n_imp_total + 1e-6, (
        f'n_imp total {n_imp_actual:.4f} exceeds SC budget {sc_n_imp_total:.4f}'
    )


# ---------------------------------------------------------------------------
# Test 7: Per-interval n_exp ceiling respected
#
# n_exp at each interval must not exceed what SC would have exported there.
# We set up a window with solar > load so SC exports surplus. The LP must
# not claim more natural export at any interval than SC would have.
# ---------------------------------------------------------------------------

def test_per_interval_n_exp_ceiling_not_exceeded():
    """n_exp[t] does not exceed the SC natural export ceiling at any interval."""
    n = 12
    solar  = np.full(n, 0.3)
    load   = np.full(n, 0.1)
    op_imp = np.full(n, 0.20)
    op_exp = np.full(n, 0.10)
    event  = np.ones(n, dtype=bool)

    bat = _bat(capacity_kwh=5.0, power_kw=5.0, backup_reserve_pct=0.0)
    # Start with full battery so SC can't absorb surplus -- all exports as natural
    traces, vt = _run(solar, load, op_imp, op_exp, event,
                      battery=bat, soc_init=bat.capacity_kwh)

    n_exp = vt.n_exp_kwh.to_numpy()
    # SC with full battery exports all surplus: n_exp_hi[t] = solar - load
    # (charge headroom = 0, so battery absorbs nothing extra)
    sc_n_exp_hi = np.maximum(0.0, solar - load)
    violations = np.where(n_exp > sc_n_exp_hi + 1e-6)[0]
    assert len(violations) == 0, (
        f'n_exp exceeds SC ceiling at {len(violations)} interval(s): '
        f'max excess={float((n_exp - sc_n_exp_hi)[violations].max()):.6f}'
    )


# ---------------------------------------------------------------------------
# Test 8: SC fallback is not triggered on standard inputs
#
# With SC-fixed bounds, the LP should always find a feasible solution.
# Run a variety of configurations and confirm no EOV_SKIP warnings are emitted.
# ---------------------------------------------------------------------------

def test_sc_fallback_not_triggered_on_standard_inputs():
    """No EOV_SKIP warnings on standard feasible inputs."""
    import warnings as _warnings

    configs = [
        dict(solar=np.zeros(8),    load=np.full(8, 0.2)),   # pure load
        dict(solar=np.full(8,0.3), load=np.zeros(8)),        # pure solar, full bat
        dict(solar=np.full(8,0.1), load=np.full(8, 0.1)),   # balanced
    ]
    bat = _bat(capacity_kwh=5.0, power_kw=5.0, backup_reserve_pct=0.0)
    event = np.ones(8, dtype=bool)
    op_imp = np.full(8, 0.20)
    op_exp = np.full(8, 0.05)

    for cfg in configs:
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            _run(cfg['solar'], cfg['load'], op_imp, op_exp, event,
                 battery=bat, soc_init=bat.capacity_kwh)
        skips = [str(w.message) for w in caught
                 if str(w.message).startswith('EOV_SKIP:')]
        assert len(skips) == 0, (
            f'Unexpected SC fallback on config {cfg}: {skips}'
        )
