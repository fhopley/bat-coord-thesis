"""Tests for rolling_optimisation.

The correctness gate is the first test and the one that matters: if a perfect
forecast at full-period horizon does not reproduce perfect_foresight_lp, the
wrapper is wrong and nothing downstream is worth looking at.
"""
import numpy as np
import pandas as pd
import pytest

from model.core.prices import build_price_stack_offsets, build_price_vectors
from model.core.scenario import BatterySpec, RollingConfig
from model.core.tariff import NetworkTariff, RetailTariff, TariffPeriod
from model.components.perfect_foresight_lp import perfect_foresight_lp
from model.components.rolling_optimisation import (
    RollingDiagnostics, rolling_optimisation,
)
from model.components.self_consumption import run_sc_with_battery
from model.forecast import (
    PerfectForecastProvider, TruncatedPerfectForecastProvider,
)

TZ = 'Etc/GMT-10'
STATE = 'SA'
INTERVAL = 5 / 60


# ------------------------------------------------------------------ fixtures

def _network(export_dlf=1.0811):
    return NetworkTariff(
        name='net', state=STATE,
        usage_tariffs=(TariffPeriod(0, 10, 10.439), TariffPeriod(10, 16, 3.124),
                       TariffPeriod(16, 17, 10.439), TariffPeriod(17, 21, 35.178),
                       TariffPeriod(21, 24, 10.439)),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 10, 0.0), TariffPeriod(10, 16, -1.1),
                        TariffPeriod(16, 24, 0.0)),
        daily_c_per_day=70.8,
        usage_dlf=1.0811, export_dlf=export_dlf,
        import_tlf=1.0012, export_tlf=1.0012,
    )


def _retail():
    return RetailTariff(
        name='amber', state=STATE, type='wholesale_exposed',
        daily_c_per_day=82.19,
        usage_tariffs=(), usage_tariffs_weekend=None, feedin_tariffs=(),
        usage_other_c_per_kwh=4.081, feedin_other_c_per_kwh=0.0,
    )


def _battery(**kw):
    base = dict(capacity_kwh=13.5, backup_reserve_pct=5.0, power_kw=5.0,
                charge_efficiency=0.95, discharge_efficiency=0.95,
                c_deg_per_kwh=0.039)
    base.update(kw)
    return BatterySpec(**base)


def _site(days=2, seed=11):
    idx = pd.date_range('2026-02-09 00:05', periods=288 * days, freq='5min', tz=TZ)
    rng = np.random.default_rng(seed)
    tod = (np.arange(len(idx)) % 288) / 288
    solar = pd.Series(
        np.clip(np.sin((tod - 0.25) * np.pi / 0.5), 0, None) * 2.2 * INTERVAL * 6
        * (1 + 0.2 * rng.standard_normal(len(idx))).clip(0.3, 1.8), index=idx)
    load = pd.Series(
        (0.35 + 0.25 * np.exp(-((tod - 0.80) ** 2) / 0.004)) * INTERVAL * 6
        * (1 + 0.25 * rng.standard_normal(len(idx))).clip(0.2, 3.0), index=idx)
    rrp = pd.Series(
        (60 + 200 * np.exp(-((tod - 0.79) ** 2) / 0.0012)
         - 70 * np.exp(-((tod - 0.5) ** 2) / 0.006))
        * (1 + 0.5 * rng.standard_normal(len(idx))).clip(-0.5, 6.0), index=idx)
    return idx, solar, load, rrp


def _setup(days=2, battery=None, **net_kw):
    idx, solar, load, rrp = _site(days)
    net, ret = _network(**net_kw), _retail()
    bs = battery or _battery()
    offsets = build_price_stack_offsets(idx, net, ret, STATE, INTERVAL)
    p_imp, p_exp = build_price_vectors(idx, rrp, net, ret, STATE, INTERVAL)
    soc0 = (bs.backup_reserve_pct / 100) * bs.capacity_kwh
    soc_arr, _ = run_sc_with_battery(solar.to_numpy(), load.to_numpy(),
                                     bs, soc0, INTERVAL)
    return dict(idx=idx, solar=solar, load=load, rrp=rrp, offsets=offsets,
                p_imp=p_imp, p_exp=p_exp, bs=bs, soc0=soc0,
                soc_end=float(soc_arr[-1]))


def _invariants(tr, solar, load, tol=1e-9):
    np.testing.assert_allclose(
        tr.import_kwh, tr.grid_to_home_kwh + tr.grid_to_bat_kwh, atol=tol)
    np.testing.assert_allclose(
        tr.export_kwh, tr.sol_to_grid_kwh + tr.bat_to_grid_kwh, atol=tol)
    np.testing.assert_allclose(
        load, tr.sol_to_home_kwh + tr.bat_to_home_kwh + tr.grid_to_home_kwh, atol=tol)
    np.testing.assert_allclose(
        solar, tr.sol_to_home_kwh + tr.sol_to_bat_kwh + tr.sol_to_grid_kwh
        + tr.curtail_kwh, atol=tol)


# -------------------------------------------------------- the correctness gate

@pytest.mark.slow
def test_perfect_forecast_full_horizon_reproduces_perfect_foresight():
    """With a perfect forecast and a horizon covering the whole period, every
    window reaches the period end and pins the SAME global terminal SOC, so
    each is exactly the tail subproblem of the first. By Bellman optimality
    the committed sequence must coincide with the full-period optimum.

    This is the gate. If it fails, the wrapper is wrong.
    """
    s = _setup(days=2)
    pf = perfect_foresight_lp('S1', s['solar'], s['load'], s['p_imp'], s['p_exp'],
                              s['bs'], s['soc0'], s['soc_end'],
                              interval_hours=INTERVAL)
    cfg = RollingConfig(forecast_source='perfect', horizon_hours=len(s['idx']) / 12)
    tr, dg = rolling_optimisation(
        'S1', s['solar'], s['load'], PerfectForecastProvider(s['rrp']),
        s['offsets'], s['bs'], s['soc0'], s['soc_end'], cfg,
        interval_hours=INTERVAL)

    for field in ('import_kwh', 'export_kwh', 'battery_soc_kwh', 'curtail_kwh',
                  'grid_to_bat_kwh', 'bat_to_grid_kwh', 'sol_to_bat_kwh'):
        np.testing.assert_allclose(
            getattr(tr, field).to_numpy(), getattr(pf, field).to_numpy(),
            atol=1e-9, err_msg=f"{field} diverges from perfect_foresight_lp")

    cost = lambda t: float((t.import_kwh * s['p_imp']
                            - t.export_kwh * s['p_exp']).sum())
    assert cost(tr) == pytest.approx(cost(pf), abs=1e-9)
    assert dg.causality_ok


@pytest.mark.slow
def test_gate_run_satisfies_site_traces_invariants():
    s = _setup(days=2)
    cfg = RollingConfig(forecast_source='perfect', horizon_hours=len(s['idx']) / 12)
    tr, _ = rolling_optimisation(
        'S1', s['solar'], s['load'], PerfectForecastProvider(s['rrp']),
        s['offsets'], s['bs'], s['soc0'], s['soc_end'], cfg,
        interval_hours=INTERVAL)
    _invariants(tr, s['solar'].to_numpy(), s['load'].to_numpy())


# ------------------------------------------------------------- terminal SOC

def test_period_ends_at_the_global_self_consumption_target():
    """Windows reaching the period end pin the global target, so the rolling,
    perfect-foresight and self-consumption dispatches all finish at the same
    state and the bill comparison needs no residual adjustment."""
    s = _setup(days=3)
    cfg = RollingConfig(forecast_source='truncated_perfect', horizon_hours=24)
    tr, dg = rolling_optimisation(
        'S1', s['solar'], s['load'],
        TruncatedPerfectForecastProvider(s['rrp']),
        s['offsets'], s['bs'], s['soc0'], s['soc_end'], cfg,
        interval_hours=INTERVAL)
    assert dg.terminal_soc_kwh == pytest.approx(s['soc_end'], abs=1e-7)
    assert tr.battery_soc_kwh.iloc[-1] == pytest.approx(s['soc_end'], abs=1e-7)


def test_soc_stays_within_battery_bounds():
    s = _setup(days=3)
    cfg = RollingConfig(forecast_source='truncated_perfect', horizon_hours=24)
    tr, _ = rolling_optimisation(
        'S1', s['solar'], s['load'], TruncatedPerfectForecastProvider(s['rrp']),
        s['offsets'], s['bs'], s['soc0'], s['soc_end'], cfg,
        interval_hours=INTERVAL)
    soc_min = (s['bs'].backup_reserve_pct / 100) * s['bs'].capacity_kwh
    assert tr.battery_soc_kwh.min() >= soc_min - 1e-7
    assert tr.battery_soc_kwh.max() <= s['bs'].capacity_kwh + 1e-7


def test_terminal_pin_does_not_leak_between_windows():
    """The pin tightens bounds on one column of a REUSED model. Without
    restoring them, a later window would inherit the previous window's target
    as a hard constraint."""
    s = _setup(days=3)
    cfg = RollingConfig(forecast_source='truncated_perfect', horizon_hours=12)
    tr, dg = rolling_optimisation(
        'S1', s['solar'], s['load'], TruncatedPerfectForecastProvider(s['rrp']),
        s['offsets'], s['bs'], s['soc0'], s['soc_end'], cfg,
        interval_hours=INTERVAL)
    # A leaked pin would freeze the SOC trajectory; a healthy run varies widely.
    assert tr.battery_soc_kwh.std() > 0.5
    assert dg.sc_fallback_infeasible == 0


# ------------------------------------------------------------ rung 2 decomposition

@pytest.mark.slow
def test_forty_eight_hour_horizon_costs_almost_nothing_against_full_foresight():
    """Rung 2a: perfect prices at 5-minute resolution, restricted to the
    pre-dispatch sawtooth horizon with infill beyond. Isolates horizon
    truncation and terminal pin drift, with no forecast error and no
    resolution loss. Justifies the 48 h default."""
    s = _setup(days=4)
    pf = perfect_foresight_lp('S1', s['solar'], s['load'], s['p_imp'], s['p_exp'],
                              s['bs'], s['soc0'], s['soc_end'],
                              interval_hours=INTERVAL)
    cfg = RollingConfig(forecast_source='truncated_perfect', horizon_hours=48)
    tr, _ = rolling_optimisation(
        'S1', s['solar'], s['load'],
        TruncatedPerfectForecastProvider(s['rrp'], coarse_resolution='5min'),
        s['offsets'], s['bs'], s['soc0'], s['soc_end'], cfg,
        interval_hours=INTERVAL)
    cost = lambda t: float((t.import_kwh * s['p_imp']
                            - t.export_kwh * s['p_exp']).sum())
    assert cost(tr) >= cost(pf) - 1e-6, "cannot beat full perfect foresight"
    assert cost(tr) - cost(pf) < 0.02 * abs(cost(pf))


@pytest.mark.slow
def test_coarse_resolution_costs_more_than_horizon_alone():
    """Rung 2b minus 2a: the price of only seeing half-hour averages beyond
    the first hour, isolated from every other effect."""
    s = _setup(days=4)
    cfg = RollingConfig(forecast_source='truncated_perfect', horizon_hours=48)
    cost = lambda t: float((t.import_kwh * s['p_imp']
                            - t.export_kwh * s['p_exp']).sum())
    out = {}
    for res in ('5min', '30min'):
        tr, _ = rolling_optimisation(
            'S1', s['solar'], s['load'],
            TruncatedPerfectForecastProvider(s['rrp'], coarse_resolution=res),
            s['offsets'], s['bs'], s['soc0'], s['soc_end'], cfg,
            interval_hours=INTERVAL)
        out[res] = cost(tr)
    assert out['30min'] > out['5min'], "coarse resolution must cost something"


# ------------------------------------------------------------------ fallbacks

class _NoForecast:
    def horizon_available(self): return 0
    def get(self, decision_instant, n_intervals): return None


def test_falls_back_to_self_consumption_when_no_forecast_exists():
    s = _setup(days=1)
    cfg = RollingConfig(forecast_source='none', horizon_hours=24)
    tr, dg = rolling_optimisation(
        'S1', s['solar'], s['load'], _NoForecast(), s['offsets'],
        s['bs'], s['soc0'], s['soc_end'], cfg, interval_hours=INTERVAL)
    assert dg.sc_fallback_no_forecast == len(s['idx'])
    assert dg.n_windows_solved == 0
    # Self-consumption never charges from the grid or exports from the battery.
    assert tr.grid_to_bat_kwh.sum() == pytest.approx(0.0)
    assert tr.bat_to_grid_kwh.sum() == pytest.approx(0.0)
    _invariants(tr, s['solar'].to_numpy(), s['load'].to_numpy())


def test_start_of_period_falls_back_rather_than_raising():
    """The first interval has no prior interval to serve as its decision
    instant, and the earliest windows precede the first coarse run."""
    s = _setup(days=2)
    cfg = RollingConfig(forecast_source='truncated_perfect', horizon_hours=24)
    _, dg = rolling_optimisation(
        'S1', s['solar'], s['load'], TruncatedPerfectForecastProvider(s['rrp']),
        s['offsets'], s['bs'], s['soc0'], s['soc_end'], cfg,
        interval_hours=INTERVAL)
    assert 1 <= dg.sc_fallback_no_forecast <= 12
    assert dg.fallback_timestamps[0] == s['idx'][0]


def test_infeasible_window_falls_back_and_warns():
    """Curtailment pinned plus a binding export limit can make a window
    infeasible. The run must continue, count it, and say so."""
    s = _setup(days=1, battery=_battery(capacity_kwh=2.0, power_kw=0.5))
    cfg = RollingConfig(forecast_source='truncated_perfect', horizon_hours=12)
    with pytest.warns(UserWarning, match="infeasible"):
        tr, dg = rolling_optimisation(
            'S1', s['solar'], s['load'],
            TruncatedPerfectForecastProvider(s['rrp']), s['offsets'],
            s['bs'], s['soc0'], s['soc_end'], cfg,
            export_limit_kwh_per_interval=0.0,
            curtail_cost_per_kwh_dollars=None,
            interval_hours=INTERVAL)
    assert dg.sc_fallback_infeasible > 0
    assert len(dg.fallback_timestamps) > 0


# ---------------------------------------------------------------- diagnostics

@pytest.mark.slow
def test_diagnostics_report_infill_and_staleness():
    s = _setup(days=3)
    cfg = RollingConfig(forecast_source='truncated_perfect', horizon_hours=48)
    _, dg = rolling_optimisation(
        'S1', s['solar'], s['load'], TruncatedPerfectForecastProvider(s['rrp']),
        s['offsets'], s['bs'], s['soc0'], s['soc_end'], cfg,
        interval_hours=INTERVAL)
    assert 0.0 < dg.infill_fraction_mean < 1.0
    assert len(dg.infill_fraction_by_hour) > 1
    assert dg.staleness_minutes_max <= 30.0
    assert dg.solve_ms_median > 0
    assert isinstance(dg, RollingDiagnostics)
    assert 'rolling_optimisation' in dg.summary()


@pytest.mark.slow
def test_infill_fraction_varies_by_hour_of_day():
    """The sawtooth showing through. Reported so a publication-schedule
    artefact is not attributed to battery behaviour."""
    s = _setup(days=4)
    cfg = RollingConfig(forecast_source='truncated_perfect', horizon_hours=48)
    _, dg = rolling_optimisation(
        'S1', s['solar'], s['load'], TruncatedPerfectForecastProvider(s['rrp']),
        s['offsets'], s['bs'], s['soc0'], s['soc_end'], cfg,
        interval_hours=INTERVAL)
    by_hour = dg.infill_fraction_by_hour
    assert max(by_hour.values()) - min(by_hour.values()) > 0.1


def test_plan_retention_captures_the_decision_context():
    s = _setup(days=3)
    day = str(s['idx'][300].date())
    cfg = RollingConfig(forecast_source='truncated_perfect', horizon_hours=24,
                        retain_plans_from=day, retain_plans_to=day)
    _, dg = rolling_optimisation(
        'S1', s['solar'], s['load'], TruncatedPerfectForecastProvider(s['rrp']),
        s['offsets'], s['bs'], s['soc0'], s['soc_end'], cfg,
        interval_hours=INTERVAL)
    assert len(dg.retained) > 0
    w = dg.retained[0]
    assert w.rrp_mwh.size == w.is_infill.size == w.plan_soc.size
    assert w.is_fine.size == w.rrp_mwh.size
    # The planned GRID flows, taken from the solution vector rather than
    # re-derived downstream.
    assert w.plan_import.size == w.plan_export.size == w.plan_soc.size
    assert w.plan_curtail.size == w.plan_soc.size
    assert (w.plan_import >= -1e-9).all() and (w.plan_export >= -1e-9).all()
    assert not ((w.plan_import > 1e-9) & (w.plan_export > 1e-9)).any(), (
        "simultaneous planned import and export means the wrong solution "
        "blocks were sliced")
    assert w.run_time <= w.decision_instant
    assert w.coarse_run_time <= w.run_time
    assert np.isfinite(w.committed_soc)
    assert w.solved


def test_retention_covers_every_instant_in_the_window():
    """Contiguity matters: a hole in the retained sequence reads on a forecast
    surface as missing data rather than as a fallback.

    Naming one date retains that whole day, both ends inclusive.
    """
    s = _setup(days=3)
    day = str(s['idx'][300].date())
    cfg = RollingConfig(forecast_source='truncated_perfect', horizon_hours=24,
                        retain_plans_from=day, retain_plans_to=day)
    _, dg = rolling_optimisation(
        'S1', s['solar'], s['load'], TruncatedPerfectForecastProvider(s['rrp']),
        s['offsets'], s['bs'], s['soc0'], s['soc_end'], cfg,
        interval_hours=INTERVAL)
    instants = [w.decision_instant for w in dg.retained]
    assert instants == sorted(instants)
    gaps = np.diff(pd.DatetimeIndex(instants).asi8)
    assert (gaps == gaps[0]).all(), "retained decision instants must be contiguous"


def test_missing_forecast_fallbacks_are_retained_as_unsolved():
    """The earliest windows precede the first coarse run, so retaining from
    the first day exercises the no-forecast fallback path."""
    s = _setup(days=2)
    day = str(s['idx'][0].date())
    cfg = RollingConfig(forecast_source='truncated_perfect', horizon_hours=24,
                        retain_plans_from=day, retain_plans_to=day)
    _, dg = rolling_optimisation(
        'S1', s['solar'], s['load'], TruncatedPerfectForecastProvider(s['rrp']),
        s['offsets'], s['bs'], s['soc0'], s['soc_end'], cfg,
        interval_hours=INTERVAL)
    unsolved = [w for w in dg.retained if not w.solved]
    assert len(unsolved) == dg.sc_fallback_no_forecast
    w = unsolved[0]
    assert w.plan_soc is None and w.plan_charge is None
    assert w.plan_import is None and w.plan_export is None
    assert w.rrp_mwh.size == 0, "there was no forecast to retain"
    assert np.isfinite(w.committed_soc)


def test_retention_bounds_are_inclusive_whole_dates():
    """The naive reading of 'to' as a timestamp gives midnight at the START of
    that day, silently dropping all but one interval of it. The dropped day is
    the last one, which is where a reader is least likely to look."""
    s = _setup(days=3)
    d0 = str(s['idx'][0].date())
    d1 = str((s['idx'][0] + pd.Timedelta(days=1)).date())
    cfg = RollingConfig(forecast_source='truncated_perfect', horizon_hours=24,
                        retain_plans_from=d0, retain_plans_to=d1)
    _, dg = rolling_optimisation(
        'S1', s['solar'], s['load'], TruncatedPerfectForecastProvider(s['rrp']),
        s['offsets'], s['bs'], s['soc0'], s['soc_end'], cfg,
        interval_hours=INTERVAL)
    dates = {w.decision_instant.date() for w in dg.retained}
    assert dates == {pd.Timestamp(d0).date(), pd.Timestamp(d1).date()}
    # Count against the index rather than a hand-derived number: the expected
    # value depends on where the period starts relative to midnight, which is
    # exactly the sort of arithmetic a test should not be repeating.
    lo = pd.Timestamp(d0, tz=TZ)
    hi = pd.Timestamp(d1, tz=TZ) + pd.Timedelta(days=1)
    di = s['idx'] - pd.Timedelta(hours=INTERVAL)
    assert len(dg.retained) == int(((di >= lo) & (di < hi)).sum())


def test_retention_rejects_a_time_of_day():
    """Reinterpreting it would be guessing at intent; honouring it would
    reintroduce the short window the inclusive rule exists to prevent."""
    s = _setup(days=2)
    day = str(s['idx'][0].date())
    cfg = RollingConfig(forecast_source='truncated_perfect', horizon_hours=24,
                        retain_plans_from=day,
                        retain_plans_to=f"{day} 12:00")
    with pytest.raises(ValueError, match="time of day"):
        rolling_optimisation(
            'S1', s['solar'], s['load'],
            TruncatedPerfectForecastProvider(s['rrp']), s['offsets'],
            s['bs'], s['soc0'], s['soc_end'], cfg, interval_hours=INTERVAL)


def test_retention_rejects_reversed_bounds():
    s = _setup(days=2)
    d0 = str(s['idx'][0].date())
    d1 = str((s['idx'][0] + pd.Timedelta(days=1)).date())
    cfg = RollingConfig(forecast_source='truncated_perfect', horizon_hours=24,
                        retain_plans_from=d1, retain_plans_to=d0)
    with pytest.raises(ValueError, match="precedes"):
        rolling_optimisation(
            'S1', s['solar'], s['load'],
            TruncatedPerfectForecastProvider(s['rrp']), s['offsets'],
            s['bs'], s['soc0'], s['soc_end'], cfg, interval_hours=INTERVAL)


def test_planned_grid_flows_match_the_plans_energy_balance():
    """net grid = load - solar + charge - discharge. The LP solves for all of
    them, so import less export must reproduce it. This is the check that
    catches a mis-sliced offset block, which would otherwise yield arrays of
    the right shape holding the wrong variable."""
    s = _setup(days=3)
    day = str(s['idx'][300].date())
    cfg = RollingConfig(forecast_source='truncated_perfect', horizon_hours=24,
                        retain_plans_from=day, retain_plans_to=day)
    _, dg = rolling_optimisation(
        'S1', s['solar'], s['load'], TruncatedPerfectForecastProvider(s['rrp']),
        s['offsets'], s['bs'], s['soc0'], s['soc_end'], cfg,
        interval_hours=INTERVAL)
    solar = s['solar'].to_numpy()
    load = s['load'].to_numpy()
    idx = s['idx']
    pos = {t: i for i, t in enumerate(idx)}
    checked = 0
    for w in dg.retained:
        if not w.solved:
            continue
        k = pos[w.decision_instant + pd.Timedelta(hours=INTERVAL)]
        n = w.plan_soc.size
        if k + n > len(idx):
            continue
        net = w.plan_import - w.plan_export
        expect = (load[k:k + n] - solar[k:k + n] + w.plan_curtail
                  + w.plan_charge - w.plan_discharge)
        np.testing.assert_allclose(net, expect, atol=1e-6)
        checked += 1
        if checked >= 20:
            break
    assert checked >= 20


# ------------------------------------------------------------- staleness

def test_coarse_staleness_is_reported_separately_from_fine():
    """The fine run is fresh at every decision instant, so its staleness is
    zero by construction on a healthy store. The coarse run is routinely up to
    25 minutes old, and that is the age that shapes the forward window."""
    s = _setup(days=3)
    cfg = RollingConfig(forecast_source='truncated_perfect', horizon_hours=24)
    _, dg = rolling_optimisation(
        'S1', s['solar'], s['load'], TruncatedPerfectForecastProvider(s['rrp']),
        s['offsets'], s['bs'], s['soc0'], s['soc_end'], cfg,
        interval_hours=INTERVAL)
    assert dg.staleness_minutes_max == 0.0
    assert dg.coarse_staleness_minutes_max > 0.0
    assert dg.coarse_staleness_minutes_max <= 30.0
    assert 0.0 < dg.coarse_staleness_minutes_mean < 30.0
    assert 'coarse run staleness' in dg.summary()


def test_no_retention_by_default():
    s = _setup(days=1)
    cfg = RollingConfig(forecast_source='truncated_perfect', horizon_hours=12)
    _, dg = rolling_optimisation(
        'S1', s['solar'], s['load'], TruncatedPerfectForecastProvider(s['rrp']),
        s['offsets'], s['bs'], s['soc0'], s['soc_end'], cfg,
        interval_hours=INTERVAL)
    assert dg.retained == ()


# ----------------------------------------------------------------- validation

def test_rejects_misaligned_traces():
    s = _setup(days=1)
    cfg = RollingConfig(forecast_source='perfect', horizon_hours=12)
    with pytest.raises(ValueError, match="same index"):
        rolling_optimisation('S1', s['solar'], s['load'].iloc[:-1],
                             PerfectForecastProvider(s['rrp']), s['offsets'],
                             s['bs'], s['soc0'], s['soc_end'], cfg,
                             interval_hours=INTERVAL)


def test_rejects_offsets_covering_a_different_period():
    s = _setup(days=1)
    short = build_price_stack_offsets(s['idx'][:100], _network(), _retail(),
                                      STATE, INTERVAL)
    cfg = RollingConfig(forecast_source='perfect', horizon_hours=12)
    with pytest.raises(ValueError, match="price_offsets covers"):
        rolling_optimisation('S1', s['solar'], s['load'],
                             PerfectForecastProvider(s['rrp']), short,
                             s['bs'], s['soc0'], s['soc_end'], cfg,
                             interval_hours=INTERVAL)


def test_rejects_soc_outside_battery_bounds():
    s = _setup(days=1)
    cfg = RollingConfig(forecast_source='perfect', horizon_hours=12)
    with pytest.raises(ValueError, match="soc_init_kwh"):
        rolling_optimisation('S1', s['solar'], s['load'],
                             PerfectForecastProvider(s['rrp']), s['offsets'],
                             s['bs'], 99.0, s['soc_end'], cfg,
                             interval_hours=INTERVAL)
