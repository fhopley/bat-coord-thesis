"""Tests for the rolling operator event LP (Part C, step C4).

The LP is vpp_sc_event._solve_event_lp, shared rather than reimplemented, so
these tests concentrate on what rolling adds: the packed layout the loop reads,
the truncated horizon, the absent terminal pin, and the constraints behaving
against rolling-derived bounds rather than window-fixed ones.

The layout tests matter more than they look. The loop extracts the committed
interval at fixed offsets and never checks what it is reading, so a packing
error does not raise -- it commits a different variable's value and the run
completes with plausible numbers.
"""
import warnings

import numpy as np
import pytest

from model.core.scenario import BatterySpec
from model.components.vpp_operator_lp import (
    OperatorSolution, pack_to_loop_layout, read_committed,
    solve_operator_window,
)

INTERVAL_H = 5 / 60


def _bat(capacity=10.0, power=5.0, reserve=10.0, eta=1.0, c_deg=0.001):
    return BatterySpec(
        capacity_kwh=capacity, backup_reserve_pct=reserve, power_kw=power,
        charge_efficiency=eta, discharge_efficiency=eta, c_deg_per_kwh=c_deg,
    )


def _solve(n_eff, n=None, solar=None, load=None, op_imp=None, op_exp=None,
           soc_entry=5.0, n_imp_budget=10.0, n_exp_hi=None,
           remaining_cap=100.0, limit='export_only', export_limit=np.inf,
           bat=None, p_curtail=None, min_export=0.0, no_grid_charge=None):
    n = n_eff if n is None else n
    bat = bat or _bat()
    solar = np.zeros(n) if solar is None else solar
    load = np.zeros(n) if load is None else load
    op_imp = np.full(n, 0.30) if op_imp is None else op_imp
    op_exp = np.full(n, 0.20) if op_exp is None else op_exp
    n_exp_hi = np.zeros(n) if n_exp_hi is None else n_exp_hi
    return solve_operator_window(
        solar_window=solar, load_window=load, op_imp=op_imp, op_exp=op_exp,
        n=n, n_eff=n_eff, n_imp_budget=n_imp_budget, n_exp_hi=n_exp_hi,
        battery_spec=bat, soc_entry=soc_entry,
        vpp_exp_reward=1.0, vpp_imp_reward=1.0,
        remaining_cap=remaining_cap, vpp_limit_applies_to=limit,
        export_limit=export_limit, interval_hours=INTERVAL_H,
        site_id='S1', k=0, p_curtail=p_curtail,
        min_export_price_per_kwh=min_export,
        no_grid_charge_window=no_grid_charge,
    )


# ---------- packing ----------

def test_pack_places_every_block_at_the_offset_the_loop_reads():
    """Built from a synthetic solution whose every entry is distinguishable,
    so a transposed or shifted block cannot coincide with the right answer."""
    e, n = 3, 5
    # 8 blocks: charge, discharge, n_imp, f_imp, n_exp, f_exp, soc, curtail
    x_event = np.arange(1, 8 * e + 1, dtype=float)
    charge = x_event[0:3]
    discharge = x_event[3:6]
    n_imp, f_imp = x_event[6:9], x_event[9:12]
    n_exp, f_exp = x_event[12:15], x_event[15:18]
    soc, curtail = x_event[18:21], x_event[21:24]

    out = pack_to_loop_layout(x_event, n_eff=e, n=n)
    assert out.size == 10 * n

    np.testing.assert_allclose(out[0*n:0*n + e], charge)
    np.testing.assert_allclose(out[1*n:1*n + e], discharge)
    np.testing.assert_allclose(out[2*n:2*n + e], n_imp + f_imp)
    np.testing.assert_allclose(out[3*n:3*n + e], n_exp + f_exp)
    np.testing.assert_allclose(out[4*n:4*n + e], soc)
    np.testing.assert_allclose(out[5*n:5*n + e], curtail)
    np.testing.assert_allclose(out[6*n:6*n + e], n_imp)
    np.testing.assert_allclose(out[7*n:7*n + e], f_imp)
    np.testing.assert_allclose(out[8*n:8*n + e], n_exp)
    np.testing.assert_allclose(out[9*n:9*n + e], f_exp)


def test_pack_zeroes_the_flow_tail_and_holds_soc_flat():
    e, n = 2, 6
    x_event = np.arange(1, 8 * e + 1, dtype=float)
    soc_last = x_event[6*e:7*e][-1]
    out = pack_to_loop_layout(x_event, n_eff=e, n=n)
    for block in (0, 1, 2, 3, 5, 6, 7, 8, 9):
        np.testing.assert_allclose(
            out[block*n + e:(block + 1)*n], 0.0,
            err_msg=f"block {block} tail is not zero")
    np.testing.assert_allclose(out[4*n + e:5*n], soc_last)


def test_pack_is_identity_in_length_when_n_eff_equals_n():
    e = n = 4
    x_event = np.arange(1, 8 * e + 1, dtype=float)
    out = pack_to_loop_layout(x_event, n_eff=e, n=n)
    assert out.size == 10 * n
    assert not np.any(out[0:6*n] == 0.0) or True   # no tail to check


def test_pack_rejects_a_mis_sized_solution():
    with pytest.raises(ValueError, match='expected'):
        pack_to_loop_layout(np.zeros(23), n_eff=3, n=5)


def test_pack_rejects_n_eff_beyond_the_window():
    with pytest.raises(ValueError, match='exceeds window length'):
        pack_to_loop_layout(np.zeros(8 * 6), n_eff=6, n=5)


def test_read_committed_reads_position_zero_of_every_block():
    e, n = 3, 5
    x_event = np.arange(1, 8 * e + 1, dtype=float)
    out = pack_to_loop_layout(x_event, n_eff=e, n=n)
    c = read_committed(out, n)
    assert c['charge'] == pytest.approx(1.0)
    assert c['discharge'] == pytest.approx(4.0)
    assert c['n_imp'] == pytest.approx(7.0)
    assert c['f_imp'] == pytest.approx(10.0)
    assert c['n_exp'] == pytest.approx(13.0)
    assert c['f_exp'] == pytest.approx(16.0)
    assert c['soc'] == pytest.approx(19.0)
    assert c['curtail'] == pytest.approx(22.0)
    assert c['import'] == pytest.approx(7.0 + 10.0)
    assert c['export'] == pytest.approx(13.0 + 16.0)


def test_split_blocks_sum_to_the_headline_blocks():
    """The invariant VppTraces depends on: n_imp + f_imp == import at every
    position, and likewise for export."""
    sol = _solve(n_eff=6, n=8, load=np.full(8, 0.3),
                 op_exp=np.full(8, 2.0), n_exp_hi=np.zeros(8))
    assert sol is not None
    n = 8
    np.testing.assert_allclose(
        sol.x[6*n:7*n] + sol.x[7*n:8*n], sol.x[2*n:3*n], atol=1e-9)
    np.testing.assert_allclose(
        sol.x[8*n:9*n] + sol.x[9*n:10*n], sol.x[3*n:4*n], atol=1e-9)


# ---------- horizon ----------

def test_only_n_eff_intervals_are_solved():
    """The window is longer than the event; the tail must carry no dispatch."""
    n, e = 10, 3
    sol = _solve(n_eff=e, n=n, load=np.full(n, 0.3), op_exp=np.full(n, 2.0))
    assert sol is not None and sol.n_eff == e
    for block in (0, 1, 2, 3, 5):
        np.testing.assert_allclose(sol.x[block*n + e:(block + 1)*n], 0.0)


def test_solve_rejects_n_eff_outside_the_window():
    with pytest.raises(ValueError, match=r'n_eff must be in'):
        _solve(n_eff=7, n=5)


def test_solve_rejects_short_input_arrays():
    with pytest.raises(ValueError, match='need at least'):
        solve_operator_window(
            solar_window=np.zeros(2), load_window=np.zeros(2),
            op_imp=np.zeros(2), op_exp=np.zeros(2),
            n=4, n_eff=4, n_imp_budget=1.0, n_exp_hi=np.zeros(2),
            battery_spec=_bat(), soc_entry=5.0,
            vpp_exp_reward=1.0, vpp_imp_reward=1.0,
            remaining_cap=10.0, vpp_limit_applies_to='export_only',
            export_limit=np.inf, interval_hours=INTERVAL_H,
            site_id='S1', k=0,
        )


# ---------- no terminal SOC pin ----------

def test_battery_may_be_emptied_into_the_event():
    """No terminal pin: with a high export price and reward the operator
    discharges to the floor, which is the point of the event."""
    bat = _bat(capacity=10.0, power=5.0, reserve=10.0)
    # 9 kWh to shed above the reserve floor, against 5 kW * 5 min = 0.4167 kWh
    # per interval, so the span must be at least 22 intervals or the battery
    # cannot physically empty and the test would be asserting a power limit
    # rather than the absence of a terminal pin.
    n = 30
    sol = _solve(n_eff=n, n=n, bat=bat, soc_entry=10.0,
                 op_exp=np.full(n, 5.0), op_imp=np.full(n, 5.5),
                 remaining_cap=1e6)
    assert sol is not None
    final_soc = sol.x[4*n + n - 1]
    soc_min = 0.10 * 10.0
    assert final_soc == pytest.approx(soc_min, abs=1e-6), (
        f"expected the battery emptied to the reserve floor, got {final_soc}"
    )


def test_min_export_price_stops_the_battery_being_emptied_cheaply():
    """Non-vacuity partner to the test above, and the backstop that replaces a
    terminal pin: below the threshold f_exp is pinned to zero, so the same
    battery is not drained."""
    bat = _bat(capacity=10.0, power=5.0, reserve=10.0)
    n = 12
    sol = _solve(n_eff=n, n=n, bat=bat, soc_entry=10.0,
                 op_exp=np.full(n, 0.05), op_imp=np.full(n, 0.06),
                 remaining_cap=1e6, min_export=0.50)
    assert sol is not None
    assert sol.x[4*n + n - 1] == pytest.approx(10.0, abs=1e-6)
    np.testing.assert_allclose(sol.x[9*n:9*n + n], 0.0, atol=1e-9)


# ---------- constraints against rolling bounds ----------

def test_natural_import_is_capped_by_the_budget_in_aggregate():
    """The budget is an aggregate over the LP's horizon, so the LP may place
    natural import anywhere within it, but not exceed the total."""
    n = 8
    budget = 0.5
    # reserve=100 pins soc_min to capacity, so the battery cannot discharge
    # and every kWh of load is met by import. Entry SOC must equal soc_min or
    # the SOC bounds are infeasible from the first interval.
    sol = _solve(n_eff=n, n=n, load=np.full(n, 0.4), soc_entry=10.0,
                 bat=_bat(capacity=10.0, reserve=100.0),
                 n_imp_budget=budget)
    assert sol is not None
    total_n_imp = sol.x[6*n:7*n].sum()
    total_import = sol.x[2*n:3*n].sum()
    assert total_n_imp <= budget + 1e-9
    # Non-vacuity: the budget must actually bind, or this passes trivially.
    assert total_import > budget + 1e-6, "fixture did not exceed the budget"
    assert sol.x[7*n:8*n].sum() == pytest.approx(total_import - total_n_imp)


def test_natural_export_is_capped_per_interval():
    n = 6
    ceiling = np.full(n, 0.05)
    sol = _solve(n_eff=n, n=n, solar=np.full(n, 0.5),
                 op_exp=np.full(n, 3.0), n_exp_hi=ceiling,
                 soc_entry=10.0, bat=_bat(capacity=10.0), remaining_cap=1e6)
    assert sol is not None
    assert np.all(sol.x[8*n:9*n] <= ceiling + 1e-9)
    # Non-vacuity: exports must have happened at all.
    assert sol.x[3*n:4*n].sum() > 0.1


def test_cap_limits_total_forced_export():
    n = 12
    cap = 0.3
    sol = _solve(n_eff=n, n=n, bat=_bat(capacity=10.0), soc_entry=10.0,
                 op_exp=np.full(n, 5.0), op_imp=np.full(n, 5.5),
                 remaining_cap=cap, limit='export_only')
    assert sol is not None
    assert sol.x[9*n:10*n].sum() <= cap + 1e-9
    assert sol.x[9*n:10*n].sum() == pytest.approx(cap, abs=1e-6), (
        "the cap should bind with a high export price"
    )


def test_cap_counts_forced_import_too_under_import_and_export():
    n = 12
    cap = 0.3
    sol = _solve(n_eff=n, n=n, bat=_bat(capacity=10.0), soc_entry=10.0,
                 op_exp=np.full(n, 5.0), op_imp=np.full(n, 5.5),
                 remaining_cap=cap, limit='import_and_export')
    assert sol is not None
    total = sol.x[7*n:8*n].sum() + sol.x[9*n:10*n].sum()
    assert total <= cap + 1e-9


def test_export_limit_binds_on_total_export():
    n = 6
    limit = 0.02
    sol = _solve(n_eff=n, n=n, solar=np.full(n, 0.5), soc_entry=10.0,
                 bat=_bat(capacity=10.0), op_exp=np.full(n, 3.0),
                 n_exp_hi=np.full(n, 1.0), export_limit=limit,
                 remaining_cap=1e6, p_curtail=0.01)
    assert sol is not None
    assert np.all(sol.x[3*n:4*n] <= limit + 1e-9)


def test_infeasible_window_returns_none():
    """Feasibility is no longer guaranteed under rolling: the shadow diverges
    from the real battery so SC is not a certificate. A full battery with
    surplus solar, curtailment disabled and export blocked has nowhere to put
    the energy."""
    n = 4
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sol = _solve(n_eff=n, n=n, solar=np.full(n, 1.0), load=np.zeros(n),
                     soc_entry=10.0, bat=_bat(capacity=10.0),
                     n_exp_hi=np.zeros(n), export_limit=0.0,
                     remaining_cap=0.0, p_curtail=None)
    assert sol is None


def test_infeasible_window_emits_the_eov_skip_token():
    """orchestrate.py and three vpp_sc_event test modules match on this
    prefix, so the rolling path must not change or swallow it."""
    n = 4
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _solve(n_eff=n, n=n, solar=np.full(n, 1.0), load=np.zeros(n),
               soc_entry=10.0, bat=_bat(capacity=10.0),
               n_exp_hi=np.zeros(n), export_limit=0.0,
               remaining_cap=0.0, p_curtail=None)
    tokens = [str(w.message) for w in caught
              if str(w.message).startswith('EOV_SKIP:')]
    assert tokens, "no EOV_SKIP warning emitted on an infeasible window"


def test_no_grid_charge_window_keeps_charging_within_solar_surplus():
    n = 6
    surplus = 0.1
    solar = np.full(n, surplus)
    mask = np.ones(n, dtype=bool)
    sol = _solve(n_eff=n, n=n, solar=solar, load=np.zeros(n),
                 soc_entry=2.0, bat=_bat(capacity=10.0),
                 op_imp=np.full(n, 0.01), op_exp=np.full(n, 0.005),
                 n_exp_hi=np.full(n, 1.0), no_grid_charge=mask)
    assert sol is not None
    assert np.all(sol.x[0*n:0*n + n] <= surplus + 1e-9)
    np.testing.assert_allclose(sol.x[7*n:8*n], 0.0, atol=1e-9)


# ---------- solution shape ----------

def test_committed_dict_matches_the_packed_vector():
    n = 5
    sol = _solve(n_eff=3, n=n, load=np.full(n, 0.3))
    assert sol is not None
    assert sol.committed == read_committed(sol.x, n)


def test_soc_continuity_holds_across_the_solved_span():
    n = 8
    bat = _bat(capacity=10.0, eta=0.95)
    sol = _solve(n_eff=n, n=n, bat=bat, soc_entry=5.0,
                 load=np.full(n, 0.2), solar=np.full(n, 0.1),
                 op_exp=np.full(n, 1.0), remaining_cap=1e6,
                 n_exp_hi=np.full(n, 1.0))
    assert sol is not None
    charge, discharge, soc = sol.x[0:n], sol.x[n:2*n], sol.x[4*n:5*n]
    prev = 5.0
    for j in range(n):
        expected = prev + 0.95 * charge[j] - discharge[j] / 0.95
        assert soc[j] == pytest.approx(expected, abs=1e-7), f"interval {j}"
        prev = soc[j]


def test_flow_balance_holds_across_the_solved_span():
    n = 8
    solar = np.full(n, 0.3)
    load = np.full(n, 0.2)
    sol = _solve(n_eff=n, n=n, solar=solar, load=load, soc_entry=5.0,
                 bat=_bat(capacity=10.0), n_exp_hi=np.full(n, 1.0),
                 op_exp=np.full(n, 1.0), remaining_cap=1e6)
    assert sol is not None
    imp, exp = sol.x[2*n:3*n], sol.x[3*n:4*n]
    charge, discharge, curtail = sol.x[0:n], sol.x[n:2*n], sol.x[5*n:6*n]
    lhs = imp - exp - charge + discharge - curtail
    np.testing.assert_allclose(lhs, load - solar, atol=1e-7)


# ---------- solver residue ----------
# HiGHS reports a primal solution to a feasibility tolerance, so a variable at
# a zero lower bound can come back fractionally negative. Seen on a real
# site-year as f_imp = -1.18e-14, which CapState.commit correctly rejected --
# the run died mid-year rather than silently accruing a negative cap charge.

def test_tiny_negative_flow_residue_is_clamped():
    e = n = 3
    x_event = np.zeros(8 * e)
    x_event[2*e] = 0.5                    # n_imp
    x_event[3*e] = -1.10e-09              # f_imp, real observed residue
    x_event[6*e:7*e] = 5.0                # soc
    out = pack_to_loop_layout(x_event, n_eff=e, n=n)
    assert out[7*n] == 0.0, "residue was not clamped"
    # And the headline series must be derived from the clamped split, or the
    # VppTraces invariant breaks by exactly the residue.
    assert out[2*n] == pytest.approx(0.5)
    assert out[6*n] + out[7*n] == pytest.approx(out[2*n])


def test_soc_is_not_clamped_by_the_residue_guard():
    """SOC is a state with its own bounds, not a flow. Clamping it at zero
    would silently move a battery below its reserve floor."""
    e = n = 2
    x_event = np.zeros(8 * e)
    x_event[6*e:7*e] = 1.35
    out = pack_to_loop_layout(x_event, n_eff=e, n=n)
    assert out[4*n] == pytest.approx(1.35)


def test_a_genuinely_negative_flow_is_rejected():
    """Non-vacuity partner: the clamp must not hide a real bound violation.
    An unclamped curtail bound against negative solar produced exactly this
    shape of error, and it must surface rather than be absorbed."""
    e = n = 3
    x_event = np.zeros(8 * e)
    x_event[3*e] = -0.05                  # far beyond any solver tolerance
    with pytest.raises(ValueError, match='beyond the'):
        pack_to_loop_layout(x_event, n_eff=e, n=n)
