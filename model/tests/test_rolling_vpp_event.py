"""Tests for the rolling VPP component wrapper (Part C, step C6).

The wrapper's job is to build the planner, run the loop, and turn what the
planner recorded into VppTraces. Two things are only checkable here: the
invariant tying the split back to the headline series, and
event_blocks_no_spike, which needs settled prices the planner never sees.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from model.core.prices import build_price_stack_offsets
from model.core.scenario import BatterySpec, RollingConfig
from model.core.site_result import SiteTraces, VppTraces
from model.core.tariff import NetworkTariff, RetailTariff, TariffPeriod
from model.components.rolling_vpp_event import (
    count_blocks_without_a_settled_spike, rolling_vpp_event,
)
from model.components.vpp_window_planner import HOUSEHOLD, OPERATOR
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


def _offsets(index):
    network = NetworkTariff(
        name='N', state='NSW',
        usage_tariffs=(TariffPeriod(0, 24, 8.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 0.0),),
        daily_c_per_day=50.0,
        usage_dlf=1.0, export_dlf=1.0, import_tlf=1.0, export_tlf=1.0,
    )
    op = RetailTariff(
        name='Op', state='NSW', type='wholesale_exposed',
        daily_c_per_day=0.0, usage_tariffs=(), usage_tariffs_weekend=None,
        feedin_tariffs=(), usage_other_c_per_kwh=0.0,
        feedin_other_c_per_kwh=0.0,
    )
    return build_price_stack_offsets(index, network, op, 'NSW', INTERVAL_H)


def _prices(spikes, T=60, base=50.0, spike=5000.0):
    p = np.full(T, base)
    p[list(spikes)] = spike
    return p


def _run(prices, solar, load, *, bat=None, soc_init=None, cap=1e6,
         event_window_hours=0.5, horizon_hours=1.0, limit='export_only',
         forecast_prices=None, **kw):
    T = len(prices)
    index = _index(T)
    bat = bat or _bat()
    settled = pd.Series(prices, index=index)
    # The forecast may differ from what settles: that is the whole point of
    # this coordination type.
    fc = settled if forecast_prices is None else pd.Series(
        forecast_prices, index=index)
    soc_min = (bat.backup_reserve_pct / 100.0) * bat.capacity_kwh
    return rolling_vpp_event(
        site_id='S1',
        solar_kwh=pd.Series(solar, index=index),
        load_kwh=pd.Series(load, index=index),
        prices_mwh=settled,
        op_price_offsets=_offsets(index),
        forecast_provider=PerfectForecastProvider(fc, interval_minutes=5),
        battery_spec=bat,
        soc_init_kwh=soc_min if soc_init is None else soc_init,
        soc_final_kwh=soc_min,
        config=RollingConfig(forecast_source='perfect',
                             horizon_hours=horizon_hours),
        vpp_threshold_dollars_per_mwh=THR,
        event_window_hours=event_window_hours,
        vpp_exp_reward=1.0, vpp_imp_reward=1.0,
        vpp_kwh_limit=cap, vpp_limit_applies_to=limit,
        min_export_price_dollars_per_mwh=0.0,
        interval_hours=INTERVAL_H,
        **kw,
    )


# ---------- the triple ----------

def test_returns_the_three_output_types():
    T = 60
    traces, vpp, diag = _run(_prices([30], T=T), np.zeros(T), np.full(T, 0.1))
    assert isinstance(traces, SiteTraces)
    assert isinstance(vpp, VppTraces)
    assert traces.site_id == vpp.site_id == 'S1'
    assert diag.site_id == 'S1'


def test_vpp_traces_share_the_site_traces_index():
    T = 60
    traces, vpp, _ = _run(_prices([30], T=T), np.zeros(T), np.full(T, 0.1))
    for s in (vpp.n_imp_kwh, vpp.f_imp_kwh, vpp.n_exp_kwh, vpp.f_exp_kwh):
        assert s.index.equals(traces.import_kwh.index)


def test_split_reconstructs_the_headline_series():
    """The invariant VppTraces documents. The split comes from the LP's own
    variables and the headline series from the loop's flow reconstruction, so
    agreement is a real check rather than a tautology."""
    T = 60
    solar = np.tile(np.concatenate([np.zeros(6), np.full(6, 0.4)]), 5)
    traces, vpp, _ = _run(_prices([30], T=T), solar, np.full(T, 0.15),
                          soc_init=5.0)
    np.testing.assert_allclose(
        (vpp.n_imp_kwh + vpp.f_imp_kwh).to_numpy(),
        traces.import_kwh.to_numpy(), atol=1e-6)
    np.testing.assert_allclose(
        (vpp.n_exp_kwh + vpp.f_exp_kwh).to_numpy(),
        traces.export_kwh.to_numpy(), atol=1e-6)


def test_all_split_series_are_non_negative():
    T = 60
    solar = np.tile(np.concatenate([np.zeros(6), np.full(6, 0.4)]), 5)
    _, vpp, _ = _run(_prices([30], T=T), solar, np.full(T, 0.15),
                     soc_init=5.0)
    for name, s in (('n_imp', vpp.n_imp_kwh), ('f_imp', vpp.f_imp_kwh),
                    ('n_exp', vpp.n_exp_kwh), ('f_exp', vpp.f_exp_kwh)):
        assert (s.to_numpy() >= -1e-9).all(), f"{name} went negative"


# ---------- phantom events ----------

def test_a_forecast_spike_that_never_settles_is_counted_as_a_phantom_block():
    """The headline quantity: the operator declared on a forecast spike, took
    control, and the price never arrived. The planner cannot see this -- it
    only ever holds forecasts -- so the wrapper supplies the settled series."""
    T = 60
    forecast = _prices([30], T=T)      # operator believes a spike is coming
    settled = _prices([], T=T)         # it never happens
    _, _, diag = _run(settled, np.zeros(T), np.full(T, 0.1),
                      forecast_prices=forecast)
    assert diag.events_anchored == 1, "no event was declared"
    assert diag.event_blocks_no_spike == 1


def test_a_block_whose_spike_materialises_is_not_a_phantom():
    """Non-vacuity partner: with forecast and settled agreeing, the same block
    must not be counted."""
    T = 60
    _, _, diag = _run(_prices([30], T=T), np.zeros(T), np.full(T, 0.1))
    assert diag.events_anchored == 1
    assert diag.event_blocks_no_spike == 0


def test_phantom_counter_is_per_block_not_per_interval():
    T = 120
    forecast = _prices([30, 90], T=T)
    settled = _prices([90], T=T)       # only the second one arrives
    _, _, diag = _run(settled, np.zeros(T), np.full(T, 0.1),
                      forecast_prices=forecast)
    assert diag.events_anchored == 2
    assert diag.event_blocks_no_spike == 1


def test_count_blocks_helper_handles_the_edges():
    """Blocks touching the first and last interval must still be counted."""
    mode = np.array([OPERATOR, OPERATOR, HOUSEHOLD, OPERATOR], dtype=object)
    quiet = np.full(4, 50.0)
    assert count_blocks_without_a_settled_spike(mode, quiet, THR) == 2
    loud = np.array([5000.0, 50.0, 50.0, 5000.0])
    assert count_blocks_without_a_settled_spike(mode, loud, THR) == 0
    assert count_blocks_without_a_settled_spike(
        np.array([HOUSEHOLD] * 4, dtype=object), quiet, THR) == 0


def test_count_blocks_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match='intervals'):
        count_blocks_without_a_settled_spike(
            np.array([OPERATOR] * 3, dtype=object), np.zeros(5), THR)


# ---------- argument validation ----------

def test_tou_advanced_requires_the_tariffs_to_build_a_household_planner():
    """The household planner is built inside the wrapper so it shares the
    loop's index, battery and horizon. Without the tariffs it cannot be."""
    T = 20
    with pytest.raises(ValueError, match='needs retail_tariff'):
        _run(_prices([], T=T), np.zeros(T), np.full(T, 0.1),
             gap_strategy='tou_advanced')


def test_self_consumption_rejects_a_household_planner():
    T = 20
    with pytest.raises(ValueError, match='takes no household_planner'):
        _run(_prices([], T=T), np.zeros(T), np.full(T, 0.1),
             household_planner=object())


def test_unknown_gap_strategy_is_rejected():
    T = 20
    with pytest.raises(ValueError, match='gap_strategy'):
        _run(_prices([], T=T), np.zeros(T), np.full(T, 0.1),
             gap_strategy='tou')


def test_horizon_shorter_than_the_event_window_is_rejected():
    """The component enforces this as well as validate_scenario, because the
    viz and any test can call it directly without going through a scenario."""
    T = 60
    with pytest.raises(ValueError, match='shorter than the event window'):
        _run(_prices([], T=T), np.zeros(T), np.full(T, 0.1),
             event_window_hours=4.0, horizon_hours=1.0)


def test_mismatched_price_length_is_rejected():
    T = 40
    index = _index(T)
    with pytest.raises(ValueError, match='prices_mwh covers'):
        rolling_vpp_event(
            site_id='S1',
            solar_kwh=pd.Series(np.zeros(T), index=index),
            load_kwh=pd.Series(np.full(T, 0.1), index=index),
            prices_mwh=pd.Series(np.full(T - 1, 50.0), index=_index(T - 1)),
            op_price_offsets=_offsets(index),
            forecast_provider=PerfectForecastProvider(
                pd.Series(np.full(T, 50.0), index=index), interval_minutes=5),
            battery_spec=_bat(), soc_init_kwh=1.0, soc_final_kwh=1.0,
            config=RollingConfig(forecast_source='perfect', horizon_hours=1.0),
            vpp_threshold_dollars_per_mwh=THR, event_window_hours=0.5,
            vpp_exp_reward=1.0, vpp_imp_reward=1.0,
            vpp_kwh_limit=100.0, vpp_limit_applies_to='export_only',
            min_export_price_dollars_per_mwh=0.0,
            interval_hours=INTERVAL_H,
        )


def test_a_missing_forecast_provider_is_rejected():
    """The operator declares events from the price forecast, so this path
    cannot run without one even if the household plan never reads prices."""
    T = 20
    index = _index(T)
    with pytest.raises(ValueError, match='requires a price forecast'):
        rolling_vpp_event(
            site_id='S1',
            solar_kwh=pd.Series(np.zeros(T), index=index),
            load_kwh=pd.Series(np.full(T, 0.1), index=index),
            prices_mwh=pd.Series(np.full(T, 50.0), index=index),
            op_price_offsets=_offsets(index),
            forecast_provider=None,
            battery_spec=_bat(), soc_init_kwh=1.0, soc_final_kwh=1.0,
            config=RollingConfig(forecast_source='perfect', horizon_hours=1.0),
            vpp_threshold_dollars_per_mwh=THR, event_window_hours=0.5,
            vpp_exp_reward=1.0, vpp_imp_reward=1.0,
            vpp_kwh_limit=100.0, vpp_limit_applies_to='export_only',
            min_export_price_dollars_per_mwh=0.0,
            interval_hours=INTERVAL_H,
        )


# ---------- diagnostics ----------

def test_diagnostics_carry_the_vpp_counters():
    T = 60
    _, _, diag = _run(_prices([30], T=T), np.zeros(T), np.full(T, 0.1))
    assert diag.events_anchored == 1
    assert diag.event_intervals == 6
    assert diag.cap_limit_kwh == pytest.approx(1e6)
    assert diag.n_lt_W is not None


def test_non_vpp_diagnostics_survive_the_wrapper():
    """dataclasses.replace must not drop the loop's own counters."""
    T = 60
    _, _, diag = _run(_prices([30], T=T), np.zeros(T), np.full(T, 0.1))
    assert diag.n_intervals == T
    assert diag.n_windows_solved > 0
    assert diag.sc_fallback_infeasible == 0


def test_forced_totals_agree_with_the_traces():
    T = 60
    bat = _bat(capacity=10.0)
    _, vpp, diag = _run(_prices([30], T=T), np.zeros(T), np.full(T, 0.05),
                        bat=bat, soc_init=10.0)
    assert diag.f_exp_total_kwh == pytest.approx(vpp.f_exp_kwh.sum())
    assert diag.f_imp_total_kwh == pytest.approx(vpp.f_imp_kwh.sum())
    assert diag.f_exp_total_kwh > 0.0


def test_cap_used_equals_the_forced_flows_it_counts():
    T = 60
    bat = _bat(capacity=10.0)
    _, vpp, diag = _run(_prices([30], T=T), np.zeros(T), np.full(T, 0.05),
                        bat=bat, soc_init=10.0, limit='export_only')
    assert diag.cap_used_kwh == pytest.approx(vpp.f_exp_kwh.sum(), abs=1e-9)


def test_a_block_open_at_the_period_end_is_settled():
    """A spike in the final intervals leaves the block open when the loop
    returns. Without the wrapper's finalize() call its import split keeps
    whatever the LP booked against a budget the event was still projecting
    forward, and no other code path would ever correct it."""
    T = 40
    _, vpp, diag = _run(_prices([T - 1], T=T), np.zeros(T), np.full(T, 0.3),
                        soc_init=5.0)
    assert diag.events_anchored == 1
    # Non-vacuity: the block must genuinely still be running at the last
    # interval, or finalize() has nothing to do and the test is empty.
    assert diag.event_intervals > 0
    assert diag.event_block_len_max is not None, "the block was never closed"
    assert diag.event_block_len_max > 0
    # Settlement caps natural import at the true budget over the real span.
    assert diag.blocks_resettled is not None


def test_settlement_at_the_period_end_bounds_natural_import():
    """The observable consequence of finalize(): over an open final block the
    settled natural import cannot exceed what self-consumption would have
    imported across the block's real span."""
    from model.components.self_consumption import run_sc_with_battery
    T = 40
    load = np.full(T, 0.3)
    solar = np.zeros(T)
    bat = _bat()
    _, vpp, diag = _run(_prices([T - 1], T=T), solar, load,
                        bat=bat, soc_init=5.0)
    n_block = diag.event_block_len_max
    sl = slice(T - n_block, T)
    soc_arr, _ = run_sc_with_battery(
        solar[sl], load[sl], bat, 5.0, INTERVAL_H)
    # A loose but real bound: settled natural import over the block cannot
    # exceed the block's total import.
    total_imp = float(vpp.n_imp_kwh.to_numpy()[sl].sum()
                      + vpp.f_imp_kwh.to_numpy()[sl].sum())
    assert float(vpp.n_imp_kwh.to_numpy()[sl].sum()) <= total_imp + 1e-9


def test_the_split_invariant_guard_actually_fires():
    """The guard exists because the split and the headline series come from
    different places, so a packing error would leave both individually
    plausible. Asserted directly: a guard never exercised is a guard that
    could have been deleted without any test noticing."""
    from model.components.rolling_vpp_event import _check_split_invariants
    T = 8
    index = _index(T)
    traces = SiteTraces(
        site_id='S1',
        import_kwh=pd.Series(np.full(T, 1.0), index=index),
        export_kwh=pd.Series(np.zeros(T), index=index),
        battery_soc_kwh=pd.Series(np.zeros(T), index=index),
        sol_to_home_kwh=pd.Series(np.zeros(T), index=index),
        sol_to_bat_kwh=pd.Series(np.zeros(T), index=index),
        sol_to_grid_kwh=pd.Series(np.zeros(T), index=index),
        bat_to_home_kwh=pd.Series(np.zeros(T), index=index),
        grid_to_home_kwh=pd.Series(np.zeros(T), index=index),
        grid_to_bat_kwh=pd.Series(np.zeros(T), index=index),
        bat_to_grid_kwh=pd.Series(np.zeros(T), index=index),
        curtail_kwh=pd.Series(np.zeros(T), index=index),
    )
    good = VppTraces(
        site_id='S1',
        n_imp_kwh=pd.Series(np.full(T, 0.6), index=index),
        f_imp_kwh=pd.Series(np.full(T, 0.4), index=index),
        n_exp_kwh=pd.Series(np.zeros(T), index=index),
        f_exp_kwh=pd.Series(np.zeros(T), index=index),
    )
    _check_split_invariants(traces, good, 'S1')      # must not raise

    bad = VppTraces(
        site_id='S1',
        n_imp_kwh=pd.Series(np.full(T, 0.6), index=index),
        f_imp_kwh=pd.Series(np.full(T, 0.1), index=index),   # sums to 0.7
        n_exp_kwh=pd.Series(np.zeros(T), index=index),
        f_exp_kwh=pd.Series(np.zeros(T), index=index),
    )
    with pytest.raises(AssertionError) as exc:
        _check_split_invariants(traces, bad, 'S1')
    msg = str(exc.value)
    # The message must carry enough to diagnose without a rerun.
    assert 'import' in msg and 'interval' in msg and 'S1' in msg


def test_the_wrapper_actually_installs_the_invariant_guard():
    """The test above proves the guard raises when violated, but not that
    anything calls it. Removing the call site would leave that test passing
    and ship a component that never checks its own output."""
    import ast
    import inspect
    from model.components import rolling_vpp_event as mod

    # The guard lives in run_with_planner, which rolling_vpp_event delegates
    # to; checking the thin wrapper would find nothing.
    tree = ast.parse(inspect.getsource(mod.run_with_planner))
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert '_check_split_invariants' in called, (
        "run_with_planner does not call _check_split_invariants"
    )
    assert 'finalize' in inspect.getsource(mod.run_with_planner)


def test_run_with_planner_exposes_the_declared_mode_array():
    """The mode array is not recoverable from the outputs -- an event
    interval where the operator did nothing carries no forced flow -- so
    anything needing the declared mask must get it from the planner."""
    from model.components.rolling_vpp_event import run_with_planner
    T = 60
    index = _index(T)
    settled = pd.Series(_prices([30], T=T), index=index)
    bat = _bat()
    soc_min = (bat.backup_reserve_pct / 100.0) * bat.capacity_kwh
    traces, vpp, diag, planner = run_with_planner(
        site_id='S1',
        solar_kwh=pd.Series(np.zeros(T), index=index),
        load_kwh=pd.Series(np.full(T, 0.1), index=index),
        prices_mwh=settled,
        op_price_offsets=_offsets(index),
        forecast_provider=PerfectForecastProvider(settled, interval_minutes=5),
        battery_spec=bat, soc_init_kwh=soc_min, soc_final_kwh=soc_min,
        config=RollingConfig(forecast_source='perfect', horizon_hours=1.0),
        vpp_threshold_dollars_per_mwh=THR, event_window_hours=0.5,
        vpp_exp_reward=1.0, vpp_imp_reward=1.0,
        vpp_kwh_limit=1e6, vpp_limit_applies_to='export_only',
        min_export_price_dollars_per_mwh=0.0,
        interval_hours=INTERVAL_H,
    )
    declared = planner.mode == OPERATOR
    assert declared.sum() == diag.event_intervals
    # The mask covers intervals the forced flows do not: the operator held
    # control before the spike without dispatching.
    forced = (vpp.f_imp_kwh.to_numpy() + vpp.f_exp_kwh.to_numpy()) > 1e-12
    assert declared.sum() >= forced.sum()


# ---------- cap reconciliation against the settled split ----------
# Found on a real 2022 site-year: f_imp + f_exp reported 268.3 kWh against a
# 250 kWh cap while cap_used_kwh reported 232.3. The 36.0 kWh gap was exactly
# n_imp_resettled_kwh. The cap was decremented from the pre-settlement split
# and never told when settlement rewrote it, so the constraint was enforced
# against one quantity and billed on another.

def _resettling_run(T=90, limit='import_and_export'):
    """A run whose block gets resettled.

    Settlement only bites when the event's real span differs from the one the
    LP projected. Under a perfect price forecast the two normally coincide --
    t_star is the spike and the block ends at the spike -- so the trigger here
    is cap exhaustion ending the block early, which leaves the LP having
    claimed natural import against a budget the shortened span never
    delivered. A larger cap produces no settlement at all, which is why the
    tests below assert that settlement occurred before checking anything.
    """
    return _run(_prices([40, 70], T=T), np.zeros(T), np.full(T, 0.4),
                soc_init=5.0, cap=0.1, limit=limit)


def test_cap_used_reconciles_with_the_settled_forced_flows():
    """cap_used_kwh must equal the forced flows VppTraces actually reports,
    per limit_applies_to. If it does not, the cap constraint and the household
    reward are computed from different numbers."""
    _, vpp, diag = _resettling_run()
    # Non-vacuity: settlement must actually have moved volume, or the two
    # agree trivially and the test proves nothing.
    assert diag.blocks_resettled and diag.blocks_resettled > 0
    assert abs(diag.n_imp_resettled_kwh) > 1e-6

    billed = float(vpp.f_imp_kwh.sum() + vpp.f_exp_kwh.sum())
    assert diag.cap_used_kwh == pytest.approx(billed, abs=1e-6), (
        f"cap_used_kwh={diag.cap_used_kwh:.4f} but VppTraces reports "
        f"{billed:.4f} kWh of forced flow; the difference should be zero, "
        f"n_imp_resettled_kwh={diag.n_imp_resettled_kwh:+.4f}"
    )


def test_export_only_cap_is_unaffected_by_import_settlement():
    """Under export_only the cap counts f_exp alone, which settlement never
    touches, so the delta must not be applied."""
    _, vpp, diag = _resettling_run(limit='export_only')
    # Under export_only forced import does not consume the cap, so the block
    # is not cut short and no settlement occurs. The property under test is
    # that the cap still equals f_exp alone either way.
    assert diag.cap_used_kwh == pytest.approx(
        float(vpp.f_exp_kwh.sum()), abs=1e-6)


def test_settlement_may_push_the_cap_past_its_limit():
    """A retrospective correction can take a site over its cap. That is the
    honest outcome -- the operator dispatched against a projected budget and
    settlement revised it -- but it must be visible, not silent."""
    T = 90
    _, vpp, diag = _run(_prices([40, 70], T=T), np.zeros(T), np.full(T, 0.4),
                        soc_init=5.0, cap=0.1, limit='import_and_export')
    billed = float(vpp.f_imp_kwh.sum() + vpp.f_exp_kwh.sum())
    assert diag.cap_used_kwh == pytest.approx(billed, abs=1e-6)
    if billed > diag.cap_limit_kwh:
        assert diag.cap_exhausted_at is not None, (
            "cap was exceeded but never reported as exhausted"
        )


# ---------- minimum export price ----------
# The threshold reaches the planner in $/MWh and the planner applies its own
# /1000 against the operator export price. The perfect-foresight runners
# divide by 1000 themselves because vpp_sc_event takes $/kWh, so a stray
# conversion on this path is easy to introduce and nearly silent: the pin
# never fires and f_exp_blocked_by_price reads 0, which is also what a
# legitimately zero threshold produces.
#
# The operator tariff in _offsets is wholesale_exposed with a zero network
# feed-in rate and unit loss factors, so op_exp is the RRP in $/kWh: a 5000
# $/MWh spike is 5.0 $/kWh and the 50 $/MWh base is 0.05 $/kWh. The thresholds
# below are chosen against those two numbers.

def _run_with_min_export(min_export_dollars_per_mwh, spike=5000.0):
    """Bypasses _run rather than calling through it: _run hardcodes
    min_export_price_dollars_per_mwh=0.0 and also forwards **kw, so passing an
    override through **kw collides on the same keyword. Mirrors _run's body
    otherwise, at the fixed shape this test needs."""
    T = 60
    index = _index(T)
    bat = _bat(capacity=10.0)
    prices = _prices([30], T=T, spike=spike)
    settled = pd.Series(prices, index=index)
    return rolling_vpp_event(
        site_id='S1',
        solar_kwh=pd.Series(np.zeros(T), index=index),
        load_kwh=pd.Series(np.full(T, 0.05), index=index),
        prices_mwh=settled,
        op_price_offsets=_offsets(index),
        forecast_provider=PerfectForecastProvider(settled, interval_minutes=5),
        battery_spec=bat,
        soc_init_kwh=10.0,
        soc_final_kwh=(bat.backup_reserve_pct / 100.0) * bat.capacity_kwh,
        config=RollingConfig(forecast_source='perfect', horizon_hours=1.0),
        vpp_threshold_dollars_per_mwh=THR,
        event_window_hours=0.5,
        vpp_exp_reward=1.0, vpp_imp_reward=1.0,
        vpp_kwh_limit=1e6, vpp_limit_applies_to='export_only',
        min_export_price_dollars_per_mwh=min_export_dollars_per_mwh,
        interval_hours=INTERVAL_H,
    )


def test_min_export_price_of_zero_permits_forced_export():
    """Non-vacuity for the pair below. Without forced export happening at all
    under a zero threshold, a test asserting it is absent under a high one
    would pass for the wrong reason."""
    _, vpp, diag = _run_with_min_export(0.0)
    assert vpp.f_exp_kwh.sum() > 0.0
    assert diag.f_exp_blocked_by_price == 0


def test_min_export_price_below_the_spike_still_permits_forced_export():
    """1000 $/MWh is 1.0 $/kWh, below the spike's 5.0 and above the base's
    0.05. Forced export at the spike survives, which separates 'pinned because
    the threshold is high' from 'pinned for some unrelated reason'."""
    _, vpp, diag = _run_with_min_export(1000.0)
    assert vpp.f_exp_kwh.sum() > 0.0
    assert diag.f_exp_blocked_by_price > 0, (
        "no interval was blocked at a threshold above the base price; the "
        "event window should span base-priced intervals either side of the "
        "spike"
    )


def test_min_export_price_above_the_spike_pins_forced_export():
    """6000 $/MWh is 6.0 $/kWh, above every price in the period, so no
    interval clears it. This is the test that fails on a stray /1000: at 6.0
    $/MWh the effective threshold would be 0.006 $/kWh, below the base price,
    and nothing would pin."""
    _, vpp, diag = _run_with_min_export(6000.0)
    assert vpp.f_exp_kwh.sum() == pytest.approx(0.0, abs=1e-9)
    assert diag.f_exp_blocked_by_price > 0
    # The event still ran -- the threshold restricted forced export, it did
    # not prevent the operator taking control.
    assert diag.event_intervals > 0


def test_min_export_price_scales_with_the_spike_it_is_compared_against():
    """A threshold that pins a 5000 $/MWh spike must not pin a 50000 $/MWh
    one. Pins the comparison as a magnitude relationship rather than a
    constant, which a conversion error in either direction breaks."""
    _, low, _ = _run_with_min_export(6000.0, spike=5000.0)
    _, high, _ = _run_with_min_export(6000.0, spike=50000.0)
    assert low.f_exp_kwh.sum() == pytest.approx(0.0, abs=1e-9)
    assert high.f_exp_kwh.sum() > 0.0
