"""Tests for the two-mode VPP window planner (Part C, step C5).

Most of these drive the REAL rolling_optimisation loop rather than calling the
planner directly. The planner's correctness is mostly a property of call
ordering -- window_prices decides the mode, solve_window consumes it,
observe_commit acts on it, and a block is settled at the boundary between two
of those -- and a mock loop would encode my assumption about that ordering
rather than test it against the loop that actually runs.
"""
import numpy as np
import pandas as pd
import pytest

from model.core.scenario import BatterySpec, RollingConfig
from model.core.prices import build_price_stack_offsets
from model.core.tariff import (
    NetworkTariff, RetailTariff, TariffPeriod,
)
from model.components.rolling_optimisation import rolling_optimisation
from model.components.vpp_declaration import CapState, window_intervals
from model.components.vpp_window_planner import (
    HOUSEHOLD, OPERATOR, VppWindowPlanner,
)
from model.forecast.perfect import PerfectForecastProvider

INTERVAL_H = 5 / 60
THR = 1000.0


def _bat(capacity=10.0, power=5.0, reserve=10.0, eta=0.95, c_deg=0.001):
    return BatterySpec(
        capacity_kwh=capacity, backup_reserve_pct=reserve, power_kw=power,
        charge_efficiency=eta, discharge_efficiency=eta, c_deg_per_kwh=c_deg,
    )


def _index(T):
    return pd.date_range('2022-06-01', periods=T, freq='5min',
                         tz='Etc/GMT-10')


def _network():
    return NetworkTariff(
        name='N', state='NSW',
        usage_tariffs=(TariffPeriod(0, 24, 8.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 0.0),),
        daily_c_per_day=50.0,
        usage_dlf=1.0, export_dlf=1.0, import_tlf=1.0, export_tlf=1.0,
    )


def _op_retail():
    return RetailTariff(
        name='Op', state='NSW', type='wholesale_exposed',
        daily_c_per_day=0.0, usage_tariffs=(), usage_tariffs_weekend=None,
        feedin_tariffs=(), usage_other_c_per_kwh=0.0,
        feedin_other_c_per_kwh=0.0,
    )


def _offsets(index):
    return build_price_stack_offsets(
        index, _network(), _op_retail(), 'NSW', INTERVAL_H)


def _planner(index, prices, *, w_hours=0.5, cap_kwh=100.0,
             limit='export_only', min_export=0.0, bat=None,
             p_curtail=None, household=None, export_limit=np.inf):
    bat = bat or _bat()
    T = len(index)
    return VppWindowPlanner(
        T=T,
        op_price_offsets=_offsets(index),
        battery_spec=bat,
        w=window_intervals(w_hours, INTERVAL_H),
        vpp_threshold_dollars_per_mwh=THR,
        vpp_exp_reward=1.0, vpp_imp_reward=1.0,
        cap=CapState(limit_kwh=cap_kwh, limit_applies_to=limit),
        min_export_price_dollars_per_mwh=min_export,
        export_limit_kwh_per_interval=export_limit,
        interval_hours=INTERVAL_H, site_id='S1',
        p_curtail=p_curtail, household_planner=household,
    )


def _run(prices, solar, load, *, planner=None, horizon_hours=1.0,
         bat=None, soc_init=None, **pk):
    """Drive the real loop with a perfect price provider."""
    T = len(prices)
    index = _index(T)
    bat = bat or _bat()
    price_s = pd.Series(prices, index=index)
    solar_s = pd.Series(solar, index=index)
    load_s = pd.Series(load, index=index)
    planner = planner or _planner(index, price_s, bat=bat, **pk)

    provider = PerfectForecastProvider(price_s, interval_minutes=5)
    soc_min = (bat.backup_reserve_pct / 100.0) * bat.capacity_kwh
    soc0 = soc_min if soc_init is None else soc_init

    traces, diag = rolling_optimisation(
        site_id='S1',
        solar_kwh=solar_s, load_kwh=load_s,
        price_offsets=_offsets(index),
        forecast_provider=provider,
        battery_spec=bat,
        soc_init_kwh=soc0, soc_final_kwh=soc_min,
        config=RollingConfig(forecast_source='perfect',
                             horizon_hours=horizon_hours),
        interval_hours=INTERVAL_H,
        planner=planner,
    )
    planner.finalize()
    return planner, traces, diag


def _prices(spikes, T=60, base=50.0, spike=5000.0):
    p = np.full(T, base)
    p[list(spikes)] = spike
    return p


# ---------- protocol conformance ----------

def test_requires_price_forecast_is_always_true():
    """The operator reads the forecast to declare events even when the
    household plan's prices never move with RRP, so this must not be
    delegated to the household sub-planner."""
    p = _planner(_index(20), pd.Series(_prices([], T=20), index=_index(20)))
    assert p.requires_price_forecast is True


def test_solution_vector_starts_with_the_loop_six_block():
    """The loop extracts at fixed offsets and never learns what else the
    vector holds."""
    T, n = 40, 12
    planner = _planner(_index(T), None)
    planner._pending_mode = (0, HOUSEHOLD, -1)
    x = planner.solve_window(
        0, n, np.zeros(n), np.full(n, 0.1), np.zeros(n), np.zeros(n),
        soc_init=5.0, soc_final=5.0)
    assert x is not None
    assert x.size >= 6 * n


# ---------- mode selection ----------

def test_no_spike_means_household_throughout():
    T = 60
    planner, traces, _ = _run(_prices([], T=T), np.zeros(T), np.full(T, 0.1))
    assert set(planner.mode) == {HOUSEHOLD}
    assert planner.f_exp.sum() == 0.0
    assert planner.f_imp.sum() == 0.0


def test_a_spike_declares_an_event_over_the_preceding_window():
    """w_hours=0.5 is 6 intervals, so a spike at 30 puts intervals 25..30
    under operator control and nothing else."""
    T = 60
    planner, _, _ = _run(_prices([30], T=T), np.zeros(T), np.full(T, 0.1))
    marked = np.flatnonzero(planner.mode == OPERATOR)
    assert marked.tolist() == [25, 26, 27, 28, 29, 30]


def test_the_event_ends_immediately_when_the_spike_passes():
    """Un-declaration: control returns to the household at the interval after
    the spike, not after a fixed duration."""
    T = 60
    planner, _, _ = _run(_prices([30], T=T), np.zeros(T), np.full(T, 0.1))
    assert planner.mode[30] == OPERATOR
    assert planner.mode[31] == HOUSEHOLD


def test_two_spikes_closer_than_w_form_one_block():
    T = 60
    planner, _, _ = _run(_prices([30, 33], T=T), np.zeros(T), np.full(T, 0.1))
    marked = np.flatnonzero(planner.mode == OPERATOR)
    assert marked.tolist() == list(range(25, 34))
    assert planner.plan_diagnostics()['events_anchored'] == 1


def test_two_spikes_far_apart_form_two_blocks():
    T = 80
    planner, _, _ = _run(_prices([20, 60], T=T), np.zeros(T), np.full(T, 0.1))
    assert planner.plan_diagnostics()['events_anchored'] == 2


# ---------- the cap ----------

def test_cap_exhaustion_ends_participation_permanently():
    """Once the cap is spent no further events are declared, however many
    spikes follow. The state is terminal."""
    T = 120
    prices = _prices([20, 50, 80, 110], T=T)
    bat = _bat(capacity=10.0)
    planner, _, _ = _run(prices, np.zeros(T), np.full(T, 0.05),
                         bat=bat, soc_init=10.0, cap_kwh=0.5,
                         limit='export_only')
    d = planner.plan_diagnostics()
    assert d['cap_exhausted_at'] is not None
    after = d['cap_exhausted_at']
    assert not np.any(planner.mode[after + 1:] == OPERATOR), (
        "events were declared after the cap was exhausted"
    )
    # Non-vacuity: events must have happened before exhaustion.
    assert np.any(planner.mode[:after + 1] == OPERATOR)


def test_cap_is_decremented_only_by_committed_forced_flows():
    """A cap decremented from the plan would exhaust roughly one horizon's
    worth of intervals too fast."""
    T = 60
    bat = _bat(capacity=10.0)
    planner, _, _ = _run(_prices([30], T=T), np.zeros(T), np.full(T, 0.05),
                         bat=bat, soc_init=10.0, cap_kwh=1e6)
    d = planner.plan_diagnostics()
    assert d['cap_used_kwh'] == pytest.approx(planner.f_exp.sum(), abs=1e-9)


# ---------- traces and invariants ----------

def test_split_reconstructs_the_committed_import_and_export():
    """The VppTraces invariant: n + f equals the headline series at every
    interval, including household and fallback intervals."""
    T = 60
    solar = np.tile(np.concatenate([np.zeros(6), np.full(6, 0.4)]), 5)
    load = np.full(T, 0.15)
    planner, traces, _ = _run(_prices([30], T=T), solar, load,
                              soc_init=5.0, cap_kwh=1e6)
    np.testing.assert_allclose(
        planner.n_imp + planner.f_imp, traces.import_kwh.to_numpy(), atol=1e-7)
    np.testing.assert_allclose(
        planner.n_exp + planner.f_exp, traces.export_kwh.to_numpy(), atol=1e-7)


def test_household_intervals_are_entirely_natural():
    T = 60
    solar = np.tile(np.concatenate([np.zeros(6), np.full(6, 0.4)]), 5)
    planner, _, _ = _run(_prices([30], T=T), solar, np.full(T, 0.15),
                         soc_init=5.0)
    hh = planner.mode == HOUSEHOLD
    assert hh.sum() > 0
    assert np.all(planner.f_imp[hh] == 0.0)
    assert np.all(planner.f_exp[hh] == 0.0)


def test_household_sc_mode_matches_plain_self_consumption():
    """Between events the SC variant must reproduce self_consumption exactly.
    If it did not, the rolling machinery would be changing dispatch at
    intervals where it is supposed to do nothing."""
    from model.components.self_consumption import run_sc_with_battery
    T = 48
    solar = np.tile(np.concatenate([np.zeros(6), np.full(6, 0.4)]), 4)
    load = np.full(T, 0.15)
    bat = _bat()
    planner, traces, _ = _run(_prices([], T=T), solar, load,
                              bat=bat, soc_init=5.0)
    assert set(planner.mode) == {HOUSEHOLD}

    soc_arr, flows = run_sc_with_battery(solar, load, bat, 5.0, INTERVAL_H)
    np.testing.assert_allclose(
        traces.import_kwh.to_numpy(),
        flows['grid_to_home'] + flows['grid_to_bat'], atol=1e-9)
    np.testing.assert_allclose(
        traces.battery_soc_kwh.to_numpy(), soc_arr, atol=1e-9)


def test_sc_household_mode_does_not_report_an_infeasible_fallback():
    """The SC variant returns a vector rather than None. None is the loop's
    infeasibility signal and would report a failure where there was a normal
    deterministic dispatch."""
    T = 40
    _, _, diag = _run(_prices([], T=T), np.zeros(T), np.full(T, 0.1))
    assert diag.sc_fallback_infeasible == 0


# ---------- retrospective settlement ----------

def test_a_block_is_settled_when_it_ends():
    T = 60
    planner, _, _ = _run(_prices([30], T=T), np.zeros(T), np.full(T, 0.3),
                         soc_init=5.0)
    d = planner.plan_diagnostics()
    assert d['events_anchored'] == 1
    assert d['event_block_len_max'] == 6      # the block was closed
    assert not planner.shadow.active


def test_settlement_preserves_total_import_per_interval():
    """Settlement moves volume between n_imp and f_imp only."""
    T = 60
    planner, traces, _ = _run(_prices([30], T=T), np.zeros(T),
                              np.full(T, 0.3), soc_init=5.0)
    np.testing.assert_allclose(
        planner.n_imp + planner.f_imp, traces.import_kwh.to_numpy(), atol=1e-7)


def test_finalize_closes_a_block_open_at_the_end_of_the_period():
    """A spike in the final intervals leaves the block open; without
    finalize() its import split would never be settled."""
    T = 40
    planner, _, _ = _run(_prices([T - 1], T=T), np.zeros(T), np.full(T, 0.2),
                         soc_init=5.0)
    assert not planner.shadow.active
    assert planner.plan_diagnostics()['events_anchored'] == 1


# ---------- diagnostics ----------

def test_plan_diagnostics_does_not_mutate():
    T = 60
    planner, _, _ = _run(_prices([30], T=T), np.zeros(T), np.full(T, 0.1))
    assert planner.plan_diagnostics() == planner.plan_diagnostics()


def test_diagnostics_carry_the_cap_and_event_counters():
    T = 60
    planner, _, _ = _run(_prices([30], T=T), np.zeros(T), np.full(T, 0.1))
    d = planner.plan_diagnostics()
    for key in ('events_anchored', 'event_intervals', 'cap_used_kwh',
                'cap_remaining_kwh', 'cap_exhausted_at', 'n_lt_W',
                'f_imp_total_kwh', 'f_exp_total_kwh',
                'n_imp_budget_clamps', 'n_imp_resettled_kwh'):
        assert key in d, f"missing diagnostic {key}"


def test_forced_totals_match_the_recorded_arrays():
    T = 60
    bat = _bat(capacity=10.0)
    planner, _, _ = _run(_prices([30], T=T), np.zeros(T), np.full(T, 0.05),
                         bat=bat, soc_init=10.0, cap_kwh=1e6)
    d = planner.plan_diagnostics()
    assert d['f_exp_total_kwh'] == pytest.approx(planner.f_exp.sum())
    assert d['f_imp_total_kwh'] == pytest.approx(planner.f_imp.sum())
    # Non-vacuity: the event must actually have forced something.
    assert d['f_exp_total_kwh'] > 0.0


# ---------- fallbacks inside an event ----------
# Feasibility is not guaranteed on this path: the shadow diverges from the
# real battery, so self-consumption is no longer a certificate for the
# operator LP. These cover what happens when a window inside a declared event
# cannot be solved.

def _infeasible_event_run(T=60, spike_at=30):
    """A run whose operator windows cannot be solved.

    Battery full, solar surplus, curtailment disabled and export pinned to
    zero, so the LP has nowhere to put the energy. The household intervals
    stay feasible because self-consumption always is.
    """
    solar = np.full(T, 0.5)
    load = np.zeros(T)
    bat = _bat(capacity=10.0, reserve=10.0)
    index = _index(T)
    planner = _planner(index, None, bat=bat, cap_kwh=1e6,
                       export_limit=0.0, p_curtail=None)
    return _run(_prices([spike_at], T=T), solar, load,
                planner=planner, bat=bat, soc_init=10.0)


def test_an_infeasible_operator_window_falls_back_rather_than_raising():
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter('ignore')
        planner, traces, diag = _infeasible_event_run()
    d = planner.plan_diagnostics()
    # Non-vacuity: the fixture must actually have declared an event and
    # actually have failed to solve it.
    assert d['events_anchored'] >= 1, "no event was declared"
    assert d['operator_infeasible'] > 0, "no operator window was infeasible"
    assert d['event_intervals_fallback'] > 0
    assert len(traces.import_kwh) == 60


def test_fallback_intervals_inside_an_event_are_entirely_natural():
    """A fallback is a statement about what the battery did, not an operator
    dispatch: its flows earn no reward and must not touch the cap."""
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter('ignore')
        planner, traces, _ = _infeasible_event_run()

    fb = (planner.mode == OPERATOR) & (planner.f_imp + planner.f_exp == 0.0)
    assert fb.sum() > 0, "no fallback interval inside an event"
    imports = traces.import_kwh.to_numpy()
    exports = traces.export_kwh.to_numpy()
    np.testing.assert_allclose(planner.n_imp[fb], imports[fb], atol=1e-9)
    np.testing.assert_allclose(planner.n_exp[fb], exports[fb], atol=1e-9)
    assert np.all(planner.f_imp[fb] == 0.0)
    assert np.all(planner.f_exp[fb] == 0.0)


def test_a_fallback_inside_an_event_does_not_consume_the_cap():
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter('ignore')
        planner, _, _ = _infeasible_event_run()
    d = planner.plan_diagnostics()
    assert d['event_intervals_fallback'] > 0
    assert d['cap_used_kwh'] == pytest.approx(
        planner.f_imp.sum() + planner.f_exp.sum(), abs=1e-9) or \
        d['cap_used_kwh'] == pytest.approx(planner.f_exp.sum(), abs=1e-9)


def test_the_event_survives_a_fallback_interval():
    """An infeasible window does not end the event: declaration is a property
    of the forecast, not of whether the LP could be solved."""
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter('ignore')
        planner, _, _ = _infeasible_event_run()
    marked = np.flatnonzero(planner.mode == OPERATOR)
    assert marked.size > 1
    assert marked.tolist() == list(range(marked[0], marked[-1] + 1)), (
        "the event block was broken by a fallback"
    )


# ---------- solve-time reductions ----------

def test_sc_household_mode_solves_only_the_committed_interval():
    """Self-consumption is deterministic and only interval 0 is committed, so
    the rest of the window is computed and thrown away. Measured at 75.7x the
    cost of one interval over an 18-hour horizon."""
    T, n = 40, 12
    planner = _planner(_index(T), None)
    planner._pending_mode = (0, HOUSEHOLD, -1)
    solar = np.zeros(n)
    load = np.full(n, 0.2)
    x = planner.solve_window(0, n, solar, load, np.zeros(n), np.zeros(n),
                             soc_init=5.0, soc_final=5.0)
    assert x is not None
    # Flow blocks carry the committed interval and nothing beyond it.
    for block in (0, 1, 2, 3, 5):
        assert np.all(x[block*n + 1:(block + 1)*n] == 0.0), (
            f"block {block} has a non-zero tail; the window is still being "
            f"solved in full"
        )
    # SOC is a state and is held flat, not zeroed.
    assert np.all(x[4*n:5*n] == pytest.approx(x[4*n]))
    assert x[4*n] > 0.0


def test_committed_dispatch_still_matches_self_consumption_exactly():
    """Non-vacuity partner: narrowing the solve must not change what is
    committed. The whole-run comparison against run_sc_with_battery is the
    real guard, and it is asserted here over a full window sequence."""
    from model.components.self_consumption import run_sc_with_battery
    T = 48
    solar = np.tile(np.concatenate([np.zeros(6), np.full(6, 0.4)]), 4)
    load = np.full(T, 0.15)
    bat = _bat()
    _, traces, _ = _run(_prices([], T=T), solar, load, bat=bat, soc_init=5.0)
    soc_arr, flows = run_sc_with_battery(solar, load, bat, 5.0, INTERVAL_H)
    np.testing.assert_allclose(
        traces.battery_soc_kwh.to_numpy(), soc_arr, atol=1e-9)
    np.testing.assert_allclose(
        traces.import_kwh.to_numpy(),
        flows['grid_to_home'] + flows['grid_to_bat'], atol=1e-9)


def test_sc_variant_declares_it_does_not_need_the_terminal_pin():
    """The loop's SC-forward run costs ~30 s per site-year and nothing on this
    path reads it: the operator LP has no terminal constraint and the
    household dispatch is deterministic."""
    p = _planner(_index(20), None)
    assert p.needs_terminal_soc is False


def test_a_planner_with_a_household_sub_planner_still_needs_the_pin():
    """The TOU variant delegates to AdvancedWindowPlanner, which does pin its
    terminal SOC, so the flag must not be unconditionally False."""
    p = _planner(_index(20), None, household=object())
    assert p.needs_terminal_soc is True


def test_the_loop_defaults_to_computing_the_pin():
    """Planners written before this property must be unaffected."""
    import inspect
    from model.components import rolling_optimisation as ro
    src = inspect.getsource(ro.rolling_optimisation)
    assert "getattr(planner, 'needs_terminal_soc', True)" in src, (
        "the loop must default to True for planners lacking the property"
    )
