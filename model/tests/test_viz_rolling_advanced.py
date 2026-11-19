"""End-to-end smoke test for the rolling advanced viz.

run_rolling_advanced_analysis is the function the viz script runs before it
draws anything, and it exercises the whole stack: the orchestrator's per-site
runner, the planner, the window solver, billing, and the display price
construction. It needs no site data -- only a trace-shaped DataFrame and a
price series -- so it can be tested here even though the plotting cannot.

Uses the real GloBird tariff from model/tariffs so this is a check against the
plan the component is built for.
"""
import matplotlib
matplotlib.use('Agg')          # no display in CI; import must precede pyplot

import numpy as np
import pandas as pd
import pytest

from model.core.paths import default_paths
from model.core.scenario import (
    BatterySpec, RollingConfig, SolarForecastConfig, UsageForecastConfig,
)
from model.core.tariff import (
    NetworkTariff, RetailTariff, TariffPeriod, load_retail_tariff,
)
from model.dev.viz_rolling_perfect_foresight_lp_advanced import (
    RollingAdvancedAnalysis, run_rolling_advanced_analysis,
)

INTERVAL = 5 / 60
STATE = 'NSW'
DAYS = 5
N = int(DAYS * 24 / INTERVAL)


def _trace_and_prices(seed=1):
    idx = pd.date_range('2023-06-05 00:05', periods=N, freq='5min',
                        tz='Etc/GMT-10')
    rng = np.random.default_rng(seed)
    per_day = int(round(24 / INTERVAL))
    tod = (np.arange(N) % per_day) * INTERVAL
    solar = 0.45 * np.clip(np.sin((tod - 6) / 12 * np.pi), 0, None) ** 1.5
    solar *= np.repeat(rng.uniform(0.4, 1.0, DAYS + 1), per_day)[:N]
    load = 0.03 + 0.06 * np.exp(-0.5 * ((tod - 19) / 2.0) ** 2)
    load *= rng.uniform(0.9, 1.1, N)
    trace = pd.DataFrame({'pv_site_net_kwh': solar, 'gross_load_kwh': load},
                         index=idx)
    prices = pd.Series(rng.normal(90.0, 250.0, N), index=idx)
    prices.iloc[220:226] = 4000.0      # export-critical
    prices.iloc[900:906] = -400.0      # import-critical
    return trace, prices


def _network():
    return NetworkTariff(
        name='net', state='NSW',
        usage_tariffs=(TariffPeriod(0, 24, 12.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 1.0),),
        daily_c_per_day=90.0,
        usage_dlf=1.05, export_dlf=1.05, import_tlf=1.0, export_tlf=1.0)


def _battery():
    return BatterySpec(capacity_kwh=13.5, backup_reserve_pct=10.0,
                       power_kw=5.0, charge_efficiency=0.95,
                       discharge_efficiency=0.95, c_deg_per_kwh=0.039)


def _globird():
    tariffs_dir = default_paths().tariffs_dir
    tariff_id = 'NSW_globird_zerohero_2026'
    assert (tariffs_dir / f'{tariff_id}.yaml').exists(), (
        f"{tariff_id}.yaml not found in {tariffs_dir}; run from project root")
    return load_retail_tariff(tariff_id, tariffs_dir)


def _config(**kw):
    base = dict(forecast_source='perfect', horizon_hours=6.0,
                usage_forecast=UsageForecastConfig(),
                solar_forecast=SolarForecastConfig())
    base.update(kw)
    return RollingConfig(**base)


def test_analysis_runs_end_to_end_on_the_real_globird_plan():
    trace, prices = _trace_and_prices()
    a = run_rolling_advanced_analysis(
        'S0001', 2023, trace, prices, _network(), _globird(), _battery(),
        _config(), export_limit_kw=10.0, state=STATE,
        interval_hours=INTERVAL)

    assert isinstance(a, RollingAdvancedAnalysis)
    assert a.backend == 'rolling'
    assert a.diagnostics.causality_ok
    assert a.diagnostics.n_intervals == N

    # Every series the plotting code reads must be aligned and finite.
    for name in ('p_imp', 'p_exp', 'p_imp_billing', 'p_exp_billing',
                 'p_imp_display', 'p_exp_display', 'curtail'):
        s = getattr(a, name)
        assert s.index.equals(trace.index), f"{name} index misaligned"
        assert np.isfinite(s.to_numpy()).all(), f"{name} has non-finite values"

    # SOC bounds and the terminal pin -- the checks the verification block
    # prints, asserted here so a regression fails the suite rather than
    # waiting for someone to read the output.
    soc = a.lp_result.battery_soc_kwh
    min_soc = 0.10 * 13.5
    assert (soc >= min_soc - 1e-6).all() and (soc <= 13.5 + 1e-6).all()
    assert abs(float(soc.iloc[-1]) - a.soc_final_kwh) <= 1e-3

    balance = trace['pv_site_net_kwh'] - (
        a.lp_result.sol_to_home_kwh + a.lp_result.sol_to_bat_kwh
        + a.lp_result.sol_to_grid_kwh + a.curtail)
    assert balance.abs().max() < 1e-6


def test_advanced_diagnostics_reach_the_analysis():
    """The counters exist to be read here; a None would mean the planner never
    reached RollingDiagnostics."""
    trace, prices = _trace_and_prices()
    a = run_rolling_advanced_analysis(
        'S0001', 2023, trace, prices, _network(), _globird(), _battery(),
        _config(), export_limit_kw=10.0, state=STATE,
        interval_hours=INTERVAL)
    assert a.diagnostics.reward_days_lost is not None
    assert a.diagnostics.block_tier_days_exhausted is not None
    assert len(a.diagnostics.block_tier_days_exhausted) == 2
    assert 'zero-import reward' in a.diagnostics.summary()


def test_reward_credits_agree_with_the_planner_count():
    """Billing walks the committed traces; the planner accumulates as it goes.
    They are different code on different inputs, so agreement on how many days
    the reward was earned is worth asserting rather than assuming."""
    trace, prices = _trace_and_prices(seed=5)
    rt = _globird()
    a = run_rolling_advanced_analysis(
        'S0001', 2023, trace, prices, _network(), rt, _battery(),
        _config(), export_limit_kw=10.0, state=STATE,
        interval_hours=INTERVAL)

    reward_per_day = rt.zero_import_reward.reward_c_per_day / 100.0
    days_paid = round(-a.lp_bill.retail_other_credits / reward_per_day)
    days_lost = a.diagnostics.reward_days_lost
    assert days_paid + days_lost == DAYS, (
        f"{days_paid} day(s) paid by billing plus {days_lost} recorded lost "
        f"by the planner should account for all {DAYS} days")


def test_plan_without_critical_rates_needs_no_price_forecast():
    """The provider is never built, so a run works with no store and no
    forecast source that could reach one."""
    rt = _globird()
    import dataclasses
    plain = dataclasses.replace(rt, critical_export_rate=None,
                                critical_import_rate=None)
    trace, prices = _trace_and_prices()
    a = run_rolling_advanced_analysis(
        'S0001', 2023, trace, prices, _network(), plain, _battery(),
        _config(forecast_source='predispatch'),   # would need a store if used
        export_limit_kw=10.0, state=STATE, interval_hours=INTERVAL)
    d = a.diagnostics
    # The claim is specifically that no window fell back for want of a PRICE
    # forecast. Site-forecast warm-up on the first local day is expected and
    # accounted for separately.
    assert d.sc_fallback_no_forecast == 0
    assert d.n_windows_solved + d.n_fallbacks == N
    assert d.sc_fallback_no_load_forecast + d.sc_fallback_no_solar_forecast \
        == d.n_fallbacks, "fallbacks not attributable to site forecast warm-up"


def test_wholesale_plan_is_refused_by_the_planner():
    rt = RetailTariff(
        name='amber', state='NSW', type='wholesale_exposed',
        daily_c_per_day=82.0, usage_tariffs=(), usage_tariffs_weekend=None,
        feedin_tariffs=(), usage_other_c_per_kwh=4.0,
        feedin_other_c_per_kwh=1.0)
    trace, prices = _trace_and_prices()
    with pytest.raises(ValueError, match="rolling_perfect_foresight_lp"):
        run_rolling_advanced_analysis(
            'S0001', 2023, trace, prices, _network(), rt, _battery(),
            _config(), export_limit_kw=10.0, state=STATE,
            interval_hours=INTERVAL)


# ---------------------------------------------------------------------------
# Mixed plans: the real Tesla Electric plan through both advanced viz paths
# ---------------------------------------------------------------------------

def _tesla():
    tariffs_dir = default_paths().tariffs_dir
    tariff_id = 'NSW_tesla_electric_mixed_2026'
    assert (tariffs_dir / f'{tariff_id}.yaml').exists(), (
        f"{tariff_id}.yaml not found in {tariffs_dir}; run from project root")
    return load_retail_tariff(tariff_id, tariffs_dir)


def test_analysis_runs_end_to_end_on_the_real_tesla_plan():
    """Billing prices are the billed rates (TOU, no spot term); the panels
    show the dispatch prices; both are aligned and finite."""
    from model.core.prices import build_layer_rates
    trace, prices = _trace_and_prices()
    rt = _tesla()
    a = run_rolling_advanced_analysis(
        'S0001', 2023, trace, prices, _network(), rt, _battery(),
        _config(), export_limit_kw=10.0, state=STATE,
        interval_hours=INTERVAL)
    assert a.diagnostics.causality_ok
    assert a.diagnostics.n_intervals == N
    for name in ('p_imp', 'p_exp', 'p_imp_billing', 'p_exp_billing',
                 'p_imp_display', 'p_exp_display', 'curtail'):
        s = getattr(a, name)
        assert s.index.equals(trace.index), f"{name} index misaligned"
        assert np.isfinite(s.to_numpy()).all(), f"{name} has non-finite values"

    lr = build_layer_rates(trace.index, prices, _network(), rt, STATE, INTERVAL)
    np.testing.assert_array_equal(a.p_imp_billing.to_numpy(), lr.retail_import)
    np.testing.assert_array_equal(a.p_exp_billing.to_numpy(), lr.retail_export)
    np.testing.assert_array_equal(a.p_imp_display.to_numpy(), a.p_imp.to_numpy())
    spot = a.p_imp.to_numpy() - a.p_imp_billing.to_numpy()
    assert (spot > 0).any(), "dispatch import never above the billed rate"


def test_perfect_foresight_viz_reconciles_on_the_real_tesla_plan(capsys):
    """viz_perfect_foresight_lp_advanced's LP price reconciliation must PASS
    on a mixed plan, with the spot term subtracted."""
    from model.dev.viz_perfect_foresight_lp_advanced import (
        _print_verification_advanced, run_lp_advanced_analysis,
    )
    trace, prices = _trace_and_prices()
    a = run_lp_advanced_analysis(
        'S0001', 2023, trace, prices, _network(), _tesla(), _battery(),
        export_limit_kw=10.0, state=STATE, interval_hours=INTERVAL)
    _print_verification_advanced(a)
    out = capsys.readouterr().out
    spot_lines = [ln for ln in out.splitlines() if 'mixed spot term' in ln]
    assert len(spot_lines) == 1, out
    assert float(spot_lines[0].split('$')[1].replace(',', '')) > 0, spot_lines
    diff_lines = [ln for ln in out.splitlines() if '|diff|=' in ln]
    assert len(diff_lines) == 1 and 'PASS' in diff_lines[0], out
