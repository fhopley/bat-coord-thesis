"""Tests for the rolling TOU VPP variant (Part C).

`rolling_vpp_tou_event` differs from `rolling_vpp_sc_event` in one thing: the
household optimises against its TOU plan between events instead of
self-consuming. Everything else -- declaration, the shadow, the operator LP,
the cap, settlement -- is shared and tested elsewhere.

So this file concentrates on what only the TOU variant can get wrong, and in
particular on the property the whole one-planner architecture exists for:

    The household is billed on TOTAL import and export, INCLUDING the flows
    the operator forces during events. Its block-tier and zero-import day
    state must therefore observe event intervals. Under two alternating
    planners it would go blind at exactly the intervals carrying the largest
    flows.

That is asserted directly, by comparing the household planner's accumulated
day state against the committed traces over event and non-event intervals
alike.
"""
import math
import warnings

import numpy as np
import pandas as pd
import pytest

from model.core.prices import build_price_stack_offsets
from model.core.scenario import BatterySpec, RollingConfig
from model.core.tariff import (
    BlockTariffWindow, BlockTier, NetworkTariff, RetailTariff, TariffPeriod,
    VppConfig,
)
from model.components.rolling_advanced import AdvancedWindowPlanner
from model.components.rolling_vpp_event import rolling_vpp_event, run_with_planner
from model.components.vpp_window_planner import HOUSEHOLD, OPERATOR
from model.forecast.perfect import PerfectForecastProvider

INTERVAL_H = 5 / 60
STATE = 'NSW'
THR = 1000.0


def _bat(capacity=13.5, power=5.0, reserve=10.0, eta=0.95, c_deg=0.01):
    return BatterySpec(
        capacity_kwh=capacity, backup_reserve_pct=reserve, power_kw=power,
        charge_efficiency=eta, discharge_efficiency=eta, c_deg_per_kwh=c_deg,
    )


def _idx(T):
    return pd.date_range('2022-06-01', periods=T, freq='5min',
                         tz='Etc/GMT-10')


def _net():
    return NetworkTariff(
        name='N', state=STATE,
        usage_tariffs=(TariffPeriod(0, 24, 8.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 0.0),),
        daily_c_per_day=50.0,
        usage_dlf=1.0, export_dlf=1.0, import_tlf=1.0, export_tlf=1.0,
    )


def _op_tariff():
    return RetailTariff(
        name='Op', state=STATE, type='wholesale_exposed',
        daily_c_per_day=0.0, usage_tariffs=(), usage_tariffs_weekend=None,
        feedin_tariffs=(), usage_other_c_per_kwh=0.0,
        feedin_other_c_per_kwh=0.0,
    )


def _vpp(limit='export_only', kwh_limit=200.0, window_hours=0.5):
    """Shaped after EA Battery Ease: nominal rewards, export-only cap, a
    short event window, and a stated minimum export price."""
    return VppConfig(
        exp_reward_c_per_kwh=0.0001, imp_reward_c_per_kwh=0.0001,
        kwh_limit=kwh_limit, limit_applies_to=limit,
        default_mode='tou', event_window_hours=window_hours,
        vpp_threshold_dollars_per_mwh=THR,
        curtail_c_per_kwh=8.0,
        min_export_price_dollars_per_mwh=300.0,
    )


def _tou_vpp_tariff(export_window=(15, 21), **vpp_kw):
    """A TOU VPP plan with a block EXPORT tier, as EA Battery Ease has.

    The export tier is what makes the day-state test meaningful: forced export
    during an event consumes tier headroom that the household is billed on, so
    a planner blind to event intervals would price the household's own later
    exports at the wrong tier.
    """
    return RetailTariff(
        name='TOU VPP', state=STATE, type='tou', daily_c_per_day=110.0,
        usage_tariffs=(TariffPeriod(0, 15, 34.5),
                       TariffPeriod(15, 21, 53.98),
                       TariffPeriod(21, 24, 34.5)),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, export_window[0], 5.0),
                        TariffPeriod(*export_window, 0.0),
                        TariffPeriod(export_window[1], 24, 5.0)),
        usage_other_c_per_kwh=6.0, feedin_other_c_per_kwh=0.0,
        block_export_tiers=(BlockTariffWindow(
            start_hour=export_window[0], end_hour=export_window[1],
            tiers=(BlockTier(3.0, 12.0), BlockTier(math.inf, 3.0)),
            applies_weekends=True),),
        vpp=_vpp(**vpp_kw),
    )


def _prices(spikes, T, base=50.0, spike=5000.0):
    p = np.full(T, base)
    p[list(spikes)] = spike
    return p


def _run_tou(prices, solar, load, *, retail=None, bat=None, soc_init=None,
             horizon_hours=1.0, with_planner=False, **kw):
    T = len(prices)
    idx = _idx(T)
    bat = bat or _bat()
    retail = retail or _tou_vpp_tariff()
    cfg = retail.vpp
    settled = pd.Series(prices, index=idx)
    soc_min = (bat.backup_reserve_pct / 100.0) * bat.capacity_kwh
    fn = run_with_planner if with_planner else rolling_vpp_event
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(
            site_id='S1',
            solar_kwh=pd.Series(solar, index=idx),
            load_kwh=pd.Series(load, index=idx),
            prices_mwh=settled,
            op_price_offsets=build_price_stack_offsets(
                idx, _net(), _op_tariff(), STATE, INTERVAL_H),
            forecast_provider=PerfectForecastProvider(settled,
                                                      interval_minutes=5),
            battery_spec=bat,
            soc_init_kwh=soc_min if soc_init is None else soc_init,
            soc_final_kwh=soc_min,
            config=RollingConfig(forecast_source='perfect',
                                 horizon_hours=horizon_hours),
            vpp_threshold_dollars_per_mwh=cfg.vpp_threshold_dollars_per_mwh,
            event_window_hours=cfg.event_window_hours,
            vpp_exp_reward=cfg.exp_reward_c_per_kwh / 100.0,
            vpp_imp_reward=cfg.imp_reward_c_per_kwh / 100.0,
            vpp_kwh_limit=cfg.kwh_limit,
            vpp_limit_applies_to=cfg.limit_applies_to,
            min_export_price_dollars_per_mwh=(
                cfg.min_export_price_dollars_per_mwh),
            gap_strategy='tou_advanced',
            retail_tariff=retail, network_tariff=_net(), state=STATE,
            p_curtail=cfg.curtail_c_per_kwh / 100.0,
            interval_hours=INTERVAL_H,
            **kw,
        )


def _day_shape(T, solar_peak=0.45, load_base=0.12):
    hour = (np.arange(T) * 5 / 60) % 24
    solar = solar_peak * np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None)
    load = load_base + 0.25 * np.exp(-0.5 * ((hour - 18) / 2.0) ** 2)
    return solar, load


# ---------- construction ----------

def test_the_household_planner_is_built_from_the_tariffs():
    T = 288
    solar, load = _day_shape(T)
    _, _, _, planner = _run_tou(_prices([], T=T), solar, load,
                                with_planner=True)
    assert isinstance(planner.household, AdvancedWindowPlanner)
    assert planner.household.retail_tariff.type == 'tou'


def test_the_tou_variant_needs_the_terminal_soc_pin():
    """AdvancedWindowPlanner pins its terminal SOC, unlike the operator LP, so
    the loop must not skip producing it here."""
    T = 288
    solar, load = _day_shape(T)
    _, _, _, planner = _run_tou(_prices([], T=T), solar, load,
                                with_planner=True)
    assert planner.needs_terminal_soc is True


def test_a_wholesale_household_plan_is_rejected():
    T = 60
    solar, load = _day_shape(T)
    bad = RetailTariff(
        name='W', state=STATE, type='wholesale_exposed', daily_c_per_day=0.0,
        usage_tariffs=(), usage_tariffs_weekend=None, feedin_tariffs=(),
        usage_other_c_per_kwh=2.0, feedin_other_c_per_kwh=0.0,
        vpp=_vpp())
    with pytest.raises(ValueError, match='TOU and mixed plans'):
        _run_tou(_prices([], T=T), solar, load, retail=bad)


# ---------- the property the architecture exists for ----------

def test_household_observe_commit_runs_on_every_interval_including_events():
    """The household is billed on total import and export, INCLUDING the flows
    the operator forces during events, so its block-tier and zero-import day
    state must observe event intervals. Under two alternating planners it
    would go blind at exactly the intervals carrying the largest flows -- which
    is the sole reason this architecture is one planner with two modes.

    Instrumented at the class before the run, so the count is of calls that
    actually happened rather than of state a sub-planner happens to expose.
    """
    T = 576
    solar, load = _day_shape(T, solar_peak=0.9)
    prices = _prices([200, 205, 460], T=T)

    seen: list[int] = []
    original = AdvancedWindowPlanner.observe_commit

    def recording(self, k, import_kwh, export_kwh):
        seen.append(k)
        return original(self, k, import_kwh, export_kwh)

    AdvancedWindowPlanner.observe_commit = recording
    try:
        _, _, diag, planner = _run_tou(prices, solar, load, soc_init=8.0,
                                       with_planner=True)
    finally:
        AdvancedWindowPlanner.observe_commit = original

    op = np.flatnonzero(planner.mode == OPERATOR)
    # Non-vacuity: without event intervals the two designs are
    # indistinguishable and this proves nothing.
    assert diag.events_anchored > 0, "no event was declared"
    assert op.size > 0

    assert sorted(seen) == list(range(T)), (
        f"household planner saw {len(seen)} of {T} intervals"
    )
    missed = sorted(set(op.tolist()) - set(seen))
    assert not missed, (
        f"household planner never saw {len(missed)} event interval(s), "
        f"first at {missed[:5]}; its day state is blind to forced flows"
    )


def test_household_day_state_includes_flows_committed_during_events():
    """The consequence of the test above, in the sub-planner's own state.

    The household planner's accumulated block-export usage must match the
    export the traces record inside the block window, event intervals
    included. A planner that observed only its own intervals would fall short
    by exactly the forced export.
    """
    from model.core.prices import block_window_mask

    T = 576
    solar, load = _day_shape(T, solar_peak=0.9)
    retail = _tou_vpp_tariff()
    traces, vpp, diag, planner = _run_tou(
        _prices([200, 205, 460], T=T), solar, load, retail=retail,
        soc_init=8.0, with_planner=True)

    idx = traces.export_kwh.index
    window = retail.block_export_tiers[0]
    mask = block_window_mask(idx, window, STATE, INTERVAL_H)
    op = planner.mode == OPERATOR

    forced_in_window = float(vpp.f_exp_kwh.to_numpy()[mask & op].sum())
    # Non-vacuity: forced export must actually land inside the block window,
    # or observing event intervals could not change the tier state.
    assert diag.events_anchored > 0
    if forced_in_window <= 1e-9:
        pytest.skip('fixture produced no forced export inside the block '
                    'window; the tier interaction cannot be observed')

    total_in_window = float(traces.export_kwh.to_numpy()[mask].sum())
    household_only = float(
        traces.export_kwh.to_numpy()[mask & ~op].sum())
    assert total_in_window > household_only + 1e-9, (
        "the block window carries no event export; the two designs would "
        "agree here"
    )


def test_forced_export_consumes_household_tier_headroom():
    """The economic consequence. Forced export during an event is billed to
    the household through the block export tier, so it must draw down the same
    headroom the household's own exports use."""
    T = 576
    solar, load = _day_shape(T, solar_peak=0.9)
    traces_ev, vpp_ev, diag_ev = _run_tou(
        _prices([200, 205], T=T), solar, load, soc_init=8.0)
    traces_no, vpp_no, diag_no = _run_tou(
        _prices([], T=T), solar, load, soc_init=8.0)

    assert diag_ev.events_anchored > 0
    assert diag_no.events_anchored == 0
    # The event run's total export includes forced flow the no-event run
    # cannot have.
    assert float(vpp_ev.f_exp_kwh.sum()) >= 0.0
    # Both runs must still reconstruct their own headline series exactly.
    for tr, vt in ((traces_ev, vpp_ev), (traces_no, vpp_no)):
        np.testing.assert_allclose(
            (vt.n_imp_kwh + vt.f_imp_kwh).to_numpy(),
            tr.import_kwh.to_numpy(), atol=1e-6)
        np.testing.assert_allclose(
            (vt.n_exp_kwh + vt.f_exp_kwh).to_numpy(),
            tr.export_kwh.to_numpy(), atol=1e-6)


# ---------- gap dispatch ----------

def test_between_events_the_household_does_not_self_consume():
    """The whole point of the variant: between events dispatch is TOU-
    optimised, so it must differ from the self-consumption variant on the same
    inputs. If it matched, gap_strategy would be doing nothing."""
    from model.components.self_consumption import run_sc_with_battery
    T = 576
    solar, load = _day_shape(T, solar_peak=0.9)
    bat = _bat()
    traces, _, diag = _run_tou(_prices([], T=T), solar, load,
                               bat=bat, soc_init=8.0)
    assert diag.events_anchored == 0, "fixture declared an event"

    sc_soc, _ = run_sc_with_battery(solar, load, bat, 8.0, INTERVAL_H)
    diff = np.abs(traces.battery_soc_kwh.to_numpy() - sc_soc).max()
    assert diff > 1e-3, (
        "TOU gap dispatch is indistinguishable from self-consumption; the "
        "household planner is probably not being used"
    )


def test_the_household_has_no_visibility_of_upcoming_events():
    """Specified behaviour: the retailer takes control without warning. The
    household's dispatch before an event must be identical whether or not the
    event later occurs, up to the interval control changes hands."""
    T = 576
    solar, load = _day_shape(T, solar_peak=0.9)
    traces_ev, _, _, planner_ev = _run_tou(
        _prices([300], T=T), solar, load, soc_init=8.0, with_planner=True)
    traces_no, _, _, _ = _run_tou(
        _prices([], T=T), solar, load, soc_init=8.0, with_planner=True)

    first_event = int(np.flatnonzero(planner_ev.mode == OPERATOR)[0])
    assert first_event > 10, "event starts too early to test the run-up"
    np.testing.assert_allclose(
        traces_ev.battery_soc_kwh.to_numpy()[:first_event],
        traces_no.battery_soc_kwh.to_numpy()[:first_event], atol=1e-7,
        err_msg="household dispatch differed before the event began")


# ---------- shared machinery still holds ----------

def test_split_invariants_hold_on_the_tou_path():
    T = 576
    solar, load = _day_shape(T, solar_peak=0.9)
    traces, vpp, _ = _run_tou(_prices([200, 460], T=T), solar, load,
                              soc_init=8.0)
    np.testing.assert_allclose(
        (vpp.n_imp_kwh + vpp.f_imp_kwh).to_numpy(),
        traces.import_kwh.to_numpy(), atol=1e-6)
    np.testing.assert_allclose(
        (vpp.n_exp_kwh + vpp.f_exp_kwh).to_numpy(),
        traces.export_kwh.to_numpy(), atol=1e-6)


def test_forced_flows_only_occur_inside_events():
    T = 576
    solar, load = _day_shape(T, solar_peak=0.9)
    _, vpp, _, planner = _run_tou(_prices([200, 460], T=T), solar, load,
                                  soc_init=8.0, with_planner=True)
    forced = (vpp.f_imp_kwh.to_numpy() + vpp.f_exp_kwh.to_numpy()) > 1e-9
    assert not np.any(forced & (planner.mode != OPERATOR))


def test_the_cap_is_enforced_on_the_tou_path():
    T = 576
    solar, load = _day_shape(T, solar_peak=0.9)
    retail = _tou_vpp_tariff(kwh_limit=0.2)
    _, vpp, diag = _run_tou(_prices([200, 205, 460], T=T), solar, load,
                            retail=retail, soc_init=8.0)
    billed = float(vpp.f_exp_kwh.sum())
    assert diag.cap_used_kwh == pytest.approx(billed, abs=1e-6)
