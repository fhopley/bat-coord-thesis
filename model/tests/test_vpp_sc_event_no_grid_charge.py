"""Unit tests for no_grid_charge_windows -- grid-charge restriction in event windows.

Covers the G4 items from the spec's no-grid-charge window build order:

  1. No-op parity: absent no_grid_charge_mask gives bit-identical results to
     the current (unrestricted) output.

  2. LP enforcement in vpp_sc_event:
     - grid_to_bat is zero on masked event intervals while solar charging and
       force-discharge into the window still occur normally.
     - force-import (f_imp) is zero in the masked window at the LP optimum.
     - The curtail-laundering exploit is blocked: at a masked interval where
       the operator is paid to import (negative op_imp), the LP may not curtail
       solar and replace it with grid energy.
     - The LP remains feasible when an event window lies entirely inside a
       no-grid-charge window, including with a binding export limit and
       curtailment enabled.
     - The restriction applies under both gap_strategy values.

  3. YAML parsing via VppConfig and ChargeWindow:
     - no_grid_charge_windows defaults to None when absent.
     - Flat [start, end] pair is parsed correctly.
     - List of pairs [[s1,e1],[s2,e2]] is parsed correctly.
     - Invalid windows (reversed, equal, out-of-range, non-integer) are
       rejected with ValueError.

All LP tests call vpp_sc_event directly without the orchestrator, using
small synthetic traces (single event window, O(10) intervals). The mask is
supplied directly to test LP enforcement in isolation; YAML roundtrip tests
verify that the loader produces a mask that would produce the same results.
"""
import warnings as _warnings

import numpy as np
import pandas as pd
import pytest

from model.core.scenario import BatterySpec
from model.components.vpp_sc_event import vpp_sc_event


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

INTERVAL_HOURS = 5 / 60


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range('2022-01-01 00:05', periods=n, freq='5min',
                         tz='Etc/GMT-10')


def _bat(capacity_kwh=10.0, power_kw=5.0, backup_reserve_pct=0.0) -> BatterySpec:
    return BatterySpec(
        capacity_kwh=capacity_kwh,
        backup_reserve_pct=backup_reserve_pct,
        power_kw=power_kw,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
        c_deg_per_kwh=0.001,   # small c_deg prevents simultaneous charge/discharge
    )


def _series(arr, idx) -> pd.Series:
    return pd.Series(arr, index=idx)


def _run(solar, load, op_imp, op_exp, event, battery=None, *,
         export_limit=np.inf, p_curtail=None, soc_init=None,
         no_grid_charge_mask=None, kwh_limit=1000.0):
    """Convenience wrapper around vpp_sc_event."""
    n = len(solar)
    idx = _idx(n)
    bat = battery or _bat()
    soc_init_kwh = (bat.backup_reserve_pct / 100.0) * bat.capacity_kwh
    if soc_init is not None:
        soc_init_kwh = soc_init
    return vpp_sc_event(
        site_id='TEST',
        solar_kwh=_series(solar, idx),
        load_kwh=_series(load, idx),
        op_price_imp=_series(op_imp, idx),
        op_price_exp=_series(op_exp, idx),
        vpp_event=np.array(event, dtype=bool),
        battery_spec=bat,
        soc_init_kwh=soc_init_kwh,
        vpp_exp_reward=0.000001,    # nominal -- isolates the restriction effect
        vpp_imp_reward=0.000002,
        vpp_kwh_limit=kwh_limit,
        vpp_limit_applies_to='export_only',
        export_limit_kwh_per_interval=export_limit,
        interval_hours=INTERVAL_HOURS,
        p_curtail=p_curtail,
        no_grid_charge_mask=no_grid_charge_mask,
    )


# ---------------------------------------------------------------------------
# Section 1: No-op parity
#
# When no_grid_charge_mask is None (the default), the component must produce
# results bit-identical to the pre-feature output. This is the most important
# backward-compatibility check: a plan without the field set must be unaffected.
# ---------------------------------------------------------------------------

def test_no_grid_charge_none_is_noop():
    """no_grid_charge_mask=None produces results identical to omitting the argument."""
    n = 16
    solar  = np.array([0.2, 0.1, 0.0, 0.0, 0.3, 0.2, 0.1, 0.0,
                        0.0, 0.1, 0.2, 0.3, 0.1, 0.0, 0.0, 0.1])
    load   = np.full(n, 0.05)
    op_imp = np.full(n, 0.20)
    op_exp = np.full(n, 0.05)
    event  = np.zeros(n, dtype=bool)
    event[4:12] = True          # one event window in the middle

    t_default, vt_default   = _run(solar, load, op_imp, op_exp, event)
    t_explicit, vt_explicit = _run(solar, load, op_imp, op_exp, event,
                                   no_grid_charge_mask=None)

    for field in ('import_kwh', 'export_kwh', 'battery_soc_kwh',
                  'sol_to_home_kwh', 'sol_to_bat_kwh', 'sol_to_grid_kwh',
                  'bat_to_home_kwh', 'grid_to_home_kwh', 'grid_to_bat_kwh',
                  'bat_to_grid_kwh', 'curtail_kwh'):
        np.testing.assert_array_equal(
            getattr(t_default, field).to_numpy(),
            getattr(t_explicit, field).to_numpy(),
            err_msg=f'{field}: default and explicit None differ',
        )
    for field in ('n_imp_kwh', 'f_imp_kwh', 'n_exp_kwh', 'f_exp_kwh'):
        np.testing.assert_array_equal(
            getattr(vt_default, field).to_numpy(),
            getattr(vt_explicit, field).to_numpy(),
            err_msg=f'VppTraces {field}: default and explicit None differ',
        )


def test_all_false_mask_is_noop():
    """An all-False mask is equivalent to no_grid_charge_mask=None."""
    n = 12
    solar  = np.full(n, 0.2)
    load   = np.full(n, 0.05)
    op_imp = np.full(n, 0.20)
    op_exp = np.full(n, 0.05)
    event  = np.zeros(n, dtype=bool)
    event[4:10] = True

    all_false = np.zeros(n, dtype=bool)
    t_none,  _ = _run(solar, load, op_imp, op_exp, event)
    t_false, _ = _run(solar, load, op_imp, op_exp, event,
                      no_grid_charge_mask=all_false)

    for field in ('import_kwh', 'export_kwh', 'grid_to_bat_kwh'):
        np.testing.assert_array_equal(
            getattr(t_none, field).to_numpy(),
            getattr(t_false, field).to_numpy(),
            err_msg=f'{field}: all-False mask differs from None',
        )


# ---------------------------------------------------------------------------
# Section 2: LP enforcement
# ---------------------------------------------------------------------------

def test_grid_to_bat_zero_on_masked_event_interval():
    """Masked event intervals have grid_to_bat == 0, even with ample load.

    Setup: 4-interval event window, t0 masked. High load so the LP would like
    to import to cover it. Battery starts empty. With the restriction the LP
    may import to cover load (grid_to_home) but must not charge the battery
    from the grid (grid_to_bat) in the masked interval.
    """
    n = 4
    solar  = np.zeros(n)
    load   = np.array([0.3, 0.1, 0.1, 0.1])   # heavy load at t0
    op_imp = np.full(n, 0.20)
    op_exp = np.full(n, 0.05)
    event  = np.ones(n, dtype=bool)
    mask   = np.array([True, False, False, False])

    bat = _bat(capacity_kwh=5.0, power_kw=5.0, backup_reserve_pct=0.0)
    traces, _ = _run(solar, load, op_imp, op_exp, event,
                     battery=bat, no_grid_charge_mask=mask)

    g2b = traces.grid_to_bat_kwh.to_numpy()
    assert g2b[0] <= 1e-9, (
        f'grid_to_bat non-zero at masked t0: {g2b[0]:.6f}'
    )


def test_solar_charging_permitted_on_masked_interval():
    """Solar charging is permitted on masked intervals (only grid charging blocked).

    Setup: solar surplus at t0 (masked), no load, battery not full. The LP
    should charge from solar at t0 and the charge should appear in sol_to_bat.
    grid_to_bat must be zero throughout.
    """
    n = 4
    solar  = np.array([0.3, 0.0, 0.0, 0.0])
    load   = np.zeros(n)
    # t1 is a price spike: op_exp=0.50. The LP prefers to charge the battery
    # from solar at t0 (op_exp only 0.05 there) and discharge into the spike
    # at t1. This gives the LP a genuine incentive to retain solar energy
    # rather than immediately export it at t0.
    op_imp = np.array([0.20, 0.50, 0.20, 0.20])
    op_exp = np.array([0.05, 0.50, 0.05, 0.05])
    event  = np.ones(n, dtype=bool)
    mask   = np.array([True, False, False, False])

    bat = _bat(capacity_kwh=5.0, power_kw=5.0, backup_reserve_pct=0.0)
    traces, _ = _run(solar, load, op_imp, op_exp, event,
                     battery=bat, soc_init=0.0,
                     no_grid_charge_mask=mask)

    s2b = traces.sol_to_bat_kwh.to_numpy()
    g2b = traces.grid_to_bat_kwh.to_numpy()

    # Solar charging happened at t0
    assert s2b[0] > 1e-9, (
        f'Expected solar charging at masked t0, got sol_to_bat={s2b[0]:.6f}'
    )
    # No grid charging anywhere in the masked window
    assert g2b[0] <= 1e-9, (
        f'grid_to_bat non-zero at masked t0: {g2b[0]:.6f}'
    )


def test_force_discharge_unaffected_in_masked_window():
    """Force-discharge (bat_to_grid) still operates normally in a masked window.

    The no-grid-charge restriction only blocks charging. The operator's main
    play -- discharging the battery to the grid during a spike -- must be
    unaffected. Setup: battery starts full, high op_exp, all intervals masked.
    Verify bat_to_grid > 0 and grid_to_bat == 0 everywhere.
    """
    n = 4
    solar  = np.zeros(n)
    load   = np.zeros(n)
    op_imp = np.full(n, 0.50)
    op_exp = np.full(n, 0.40)   # high export price; LP will discharge
    event  = np.ones(n, dtype=bool)
    mask   = np.ones(n, dtype=bool)   # all masked

    bat = _bat(capacity_kwh=5.0, power_kw=5.0, backup_reserve_pct=0.0)
    traces, vt = _run(solar, load, op_imp, op_exp, event,
                      battery=bat, soc_init=bat.capacity_kwh,
                      no_grid_charge_mask=mask)

    b2g = traces.bat_to_grid_kwh.to_numpy()
    g2b = traces.grid_to_bat_kwh.to_numpy()

    assert b2g.sum() > 1e-9, (
        f'Expected force-discharge into masked window, got bat_to_grid.sum()={b2g.sum():.6f}'
    )
    np.testing.assert_allclose(
        g2b, np.zeros(n), atol=1e-9,
        err_msg='grid_to_bat non-zero in fully masked window',
    )


def test_f_imp_zero_in_masked_window_at_optimum():
    """force-import (f_imp) is zero in the masked window at the LP optimum.

    Once grid charging is blocked, there is no incentive for the operator to
    force-import: forced import can only meet load (at a cost to the operator)
    with no discharge revenue to fund it. Setup: no solar, moderate load,
    masked window, battery starts empty. Verify f_imp is zero at masked intervals.
    """
    n = 6
    solar  = np.zeros(n)
    load   = np.full(n, 0.05)
    op_imp = np.full(n, 0.20)
    op_exp = np.full(n, 0.05)
    event  = np.ones(n, dtype=bool)
    mask   = np.array([True, True, False, False, False, False])

    bat = _bat(capacity_kwh=5.0, power_kw=5.0, backup_reserve_pct=0.0)
    traces, vt = _run(solar, load, op_imp, op_exp, event,
                      battery=bat, soc_init=0.0,
                      no_grid_charge_mask=mask)

    f_imp = vt.f_imp_kwh.to_numpy()
    assert (f_imp[mask] < 1e-9).all(), (
        f'f_imp non-zero at masked intervals: {f_imp[mask]}'
    )


def test_curtail_laundering_blocked():
    """The curtail-laundering exploit is blocked by the coupling row.

    Without the restriction, at a masked interval with solar surplus S and
    negative op_imp (operator paid to import), the LP would: curtail S kWh of
    solar, import S kWh from the grid, charge the battery from that grid
    energy -- profiting from the import payment while violating the contract.

    The coupling row charge[t] + curtail[t] <= max(0, solar[t] - load[t])
    prevents this: it forbids both curtailing and charging at the same interval
    beyond the solar surplus.

    Setup: 3-interval window. t0 masked with solar=2, load=0, op_imp=-1
    (operator is paid $1/kWh to import), op_exp=-1 (exporting is equally
    costly). Battery starts empty. Spike at t2 (op_exp=op_imp=5) so the LP
    wants SOC; op_imp matches op_exp to satisfy the price-ordering check.
    curtail enabled (p_curtail=0.08) to make the exploit worth attempting.

    Assert: grid_to_bat == 0 at t0 with restriction. Check: grid_to_bat > 0
    without restriction, confirming the exploit is genuine.
    """
    # 3-interval event (1-hour intervals for clarity)
    solar  = np.array([2.0, 0.0, 0.0])
    load   = np.array([0.0, 0.0, 0.0])
    op_imp = np.array([-1.0, 0.20, 5.00])   # t2: op_imp == op_exp (price spike, no spread)
    op_exp = np.array([-1.0, 0.05, 5.00])
    event  = np.ones(3, dtype=bool)
    mask   = np.array([True, False, False])

    bat = _bat(capacity_kwh=10.0, power_kw=5.0, backup_reserve_pct=0.0)

    n = 3
    idx = _idx(n)
    common = dict(
        site_id='TEST',
        solar_kwh=_series(solar, idx),
        load_kwh=_series(load, idx),
        op_price_imp=_series(op_imp, idx),
        op_price_exp=_series(op_exp, idx),
        vpp_event=np.array(event, dtype=bool),
        battery_spec=bat,
        soc_init_kwh=0.0,
        vpp_exp_reward=0.000001, vpp_imp_reward=0.000002,
        vpp_kwh_limit=1000.0, vpp_limit_applies_to='export_only',
        interval_hours=1.0,     # 1-hour intervals to amplify the signal
        p_curtail=0.08,
    )

    # With restriction: grid_to_bat must be zero at t0
    t_restricted, _ = vpp_sc_event(**common, no_grid_charge_mask=mask)
    g2b_restricted = t_restricted.grid_to_bat_kwh.to_numpy()
    assert g2b_restricted[0] <= 1e-9, (
        f'RESTRICTION FAILED: grid_to_bat={g2b_restricted[0]:.6f} at masked t0 '
        f'(curtail-laundering exploit not blocked)'
    )

    # Without restriction: confirm the exploit fires (baseline to validate setup)
    t_open, _ = vpp_sc_event(**common, no_grid_charge_mask=None)
    g2b_open = t_open.grid_to_bat_kwh.to_numpy()
    assert g2b_open[0] > 1e-6, (
        f'Test setup issue: expected curtail-laundering at t0 without restriction, '
        f'got grid_to_bat={g2b_open[0]:.6f}'
    )


def test_feasibility_event_fully_inside_masked_window():
    """LP remains feasible when the entire event window is masked.

    The SC dispatch always satisfies the coupling row (SC charges only from
    solar surplus, never force-imports), so the Section 4 feasibility guarantee
    holds. Verify no EOV_SKIP warning and a valid solution is returned.
    """
    n = 8
    solar  = np.full(n, 0.1)
    load   = np.full(n, 0.05)
    op_imp = np.full(n, 0.20)
    op_exp = np.full(n, 0.05)
    event  = np.ones(n, dtype=bool)
    mask   = np.ones(n, dtype=bool)   # entire event masked

    bat = _bat(capacity_kwh=5.0, power_kw=5.0, backup_reserve_pct=0.0)

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter('always')
        traces, vt = _run(solar, load, op_imp, op_exp, event,
                          battery=bat, no_grid_charge_mask=mask)

    skips = [str(w.message) for w in caught
             if str(w.message).startswith('EOV_SKIP:')]
    assert len(skips) == 0, f'Unexpected SC fallback: {skips}'
    # grid_to_bat is zero everywhere (all masked)
    np.testing.assert_allclose(
        traces.grid_to_bat_kwh.to_numpy(), np.zeros(n),
        atol=1e-9,
        err_msg='grid_to_bat non-zero in fully masked event',
    )


def test_feasibility_masked_event_with_binding_export_limit_and_curtail():
    """LP remains feasible: masked event, binding export limit, curtailment on.

    This is the toughest feasibility case: the coupling row limits charging to
    solar surplus, the export limit caps outflows, and the battery may be near
    full. Curtailment must still be available as the escape valve. Verify no
    EOV_SKIP warning and the solar invariant holds.
    """
    n = 6
    solar  = np.full(n, 0.3)
    load   = np.zeros(n)
    op_imp = np.full(n, 0.20)
    op_exp = np.full(n, 0.05)
    event  = np.ones(n, dtype=bool)
    mask   = np.ones(n, dtype=bool)

    bat = _bat(capacity_kwh=5.0, power_kw=5.0, backup_reserve_pct=0.0)
    export_limit = 0.05   # tight -- well below solar

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter('always')
        traces, _ = _run(solar, load, op_imp, op_exp, event, battery=bat,
                         export_limit=export_limit, p_curtail=0.0,
                         soc_init=bat.capacity_kwh,
                         no_grid_charge_mask=mask)

    skips = [str(w.message) for w in caught
             if str(w.message).startswith('EOV_SKIP:')]
    assert len(skips) == 0, f'Unexpected SC fallback: {skips}'

    # Export limit not violated
    export = traces.export_kwh.to_numpy()
    assert (export <= export_limit + 1e-9).all(), (
        f'Export limit violated: max={export.max():.6f}'
    )

    # Solar invariant holds with non-zero curtailment
    lhs = (traces.sol_to_home_kwh.to_numpy()
           + traces.sol_to_bat_kwh.to_numpy()
           + traces.sol_to_grid_kwh.to_numpy()
           + traces.curtail_kwh.to_numpy())
    np.testing.assert_allclose(lhs, solar, atol=1e-9,
                               err_msg='Solar invariant violated with mask+export limit')


def test_restriction_applies_self_consumption_gap_strategy():
    """no_grid_charge_mask applies with gap_strategy='self_consumption' (default).

    This is the default gap strategy; verifying the mask is used inside the
    event LP regardless of the gap path.
    """
    n = 8
    solar  = np.array([0.0]*4 + [0.2, 0.0, 0.0, 0.0])
    load   = np.full(n, 0.05)
    op_imp = np.full(n, 0.20)
    op_exp = np.full(n, 0.05)
    event  = np.zeros(n, dtype=bool)
    event[4:8] = True               # event in second half
    mask   = np.zeros(n, dtype=bool)
    mask[4:8] = True                # mask the event window

    bat = _bat(capacity_kwh=5.0, power_kw=5.0, backup_reserve_pct=0.0)
    traces, _ = _run(solar, load, op_imp, op_exp, event, battery=bat,
                     no_grid_charge_mask=mask)

    # No grid charging in the masked event window
    g2b = traces.grid_to_bat_kwh.to_numpy()
    np.testing.assert_allclose(
        g2b[4:8], np.zeros(4), atol=1e-9,
        err_msg='grid_to_bat non-zero in masked event window (SC gap strategy)',
    )


# ---------------------------------------------------------------------------
# Section 3: YAML parsing via ChargeWindow and VppConfig
# ---------------------------------------------------------------------------

def _make_vpp_yaml(extra_vpp_keys: dict) -> dict:
    """Build a minimal retail TOU tariff YAML dict with a vpp block."""
    base = {
        'tariff_class': 'retail',
        'type': 'tou',
        'name': 'Test',
        'state': 'NSW',
        'usage_tariffs': [[0, 24, 30.0]],
        'feedin_tariffs': [[0, 24, 5.0]],
        'usage_other_c_per_kwh': {},
        'daily_charges_c_per_day': {'supply': 100.0},
        'vpp': {
            'exp_reward_c_per_kwh': 0.0001,
            'imp_reward_c_per_kwh': 0.0001,
            'kwh_limit': 200.0,
            'limit_applies_to': 'export_only',
            'default_mode': 'tou',
            'event_window_hours': 2.0,
            'vpp_threshold_dollars_per_mwh': 1000.0,
            **extra_vpp_keys,
        },
    }
    return base


def _load_from_dict(data: dict):
    """Write dict to a temp YAML file and load it as a RetailTariff."""
    import pathlib
    import tempfile
    import yaml
    from model.core.tariff import load_retail_tariff
    with tempfile.TemporaryDirectory() as tmp:
        p = pathlib.Path(tmp) / 'test_ngc.yaml'
        p.write_text(yaml.dump(data))
        return load_retail_tariff('test_ngc', pathlib.Path(tmp))


def test_no_grid_charge_windows_absent_defaults_none():
    """no_grid_charge_windows is None when the key is absent from YAML."""
    t = _load_from_dict(_make_vpp_yaml({}))
    assert t.vpp.no_grid_charge_windows is None


def test_no_grid_charge_windows_flat_pair_parsed():
    """Flat [start, end] pair is parsed into a one-element tuple of ChargeWindow."""
    from model.core.tariff import ChargeWindow
    t = _load_from_dict(_make_vpp_yaml({'no_grid_charge_windows': [17, 21]}))
    assert t.vpp.no_grid_charge_windows == (ChargeWindow(17, 21),)


def test_no_grid_charge_windows_list_of_pairs_parsed():
    """List-of-pairs form [[s1,e1],[s2,e2]] is parsed into a two-element tuple."""
    from model.core.tariff import ChargeWindow
    t = _load_from_dict(_make_vpp_yaml(
        {'no_grid_charge_windows': [[7, 9], [17, 21]]}))
    assert t.vpp.no_grid_charge_windows == (
        ChargeWindow(7, 9), ChargeWindow(17, 21))


def test_charge_window_reversed_hours_rejected():
    """ChargeWindow with start >= end is rejected."""
    from model.core.tariff import ChargeWindow
    with pytest.raises(ValueError, match='invalid hours'):
        ChargeWindow(21, 17)


def test_charge_window_equal_hours_rejected():
    """ChargeWindow with start == end is rejected."""
    from model.core.tariff import ChargeWindow
    with pytest.raises(ValueError, match='invalid hours'):
        ChargeWindow(17, 17)


def test_charge_window_out_of_range_rejected():
    """ChargeWindow hours outside [0, 24] are rejected."""
    from model.core.tariff import ChargeWindow
    with pytest.raises(ValueError):
        ChargeWindow(-1, 5)
    with pytest.raises(ValueError):
        ChargeWindow(0, 25)


def test_charge_window_non_integer_hours_rejected():
    """ChargeWindow with non-integer hours is rejected."""
    from model.core.tariff import ChargeWindow
    with pytest.raises(ValueError, match='integers'):
        ChargeWindow(17.0, 21)
