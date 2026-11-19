"""End-to-end tests for orchestrate.py and main.py.

Build a small synthetic fixture tree (3 sites, 2 NSW + 1 VIC, full year),
copy production tariffs into it, then exercise:

  - validate_scenario (success and various failures)
  - run_scenario (returns SiteTraces + SiteBills per site)
  - cross-product expansion in main.load_scenarios
  - full main() run against a YAML, asserting output files appear
  - strict-abort on validation failure

These are slower than the unit tests because each builds a full-year
fixture file, but they exercise the entire wired-together pipeline.
"""
from pathlib import Path
import math
import shutil
import yaml
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from model.core import ScenarioConfig, BatterySpec
from model.core.paths import Paths
from model.core.scenario import RollingConfig
from model.core.tariff import NetworkTariff, RetailTariff, TariffPeriod
from model.orchestrate import (
    validate_scenario, run_scenario,
    ScenarioValidationError, ScenarioResult,
    write_bills_csv, write_scenario_summary,
    write_rolling_diagnostics_csv,
    _resolve_per_site_export_limits,
)
from model.main import load_scenarios, main

from model.tests.fixtures.ingest_fixture_builder import build_full_fixture


PRODUCTION_TARIFFS_DIR = Path(__file__).parent.parent / 'tariffs'


def _paths_for(root: Path) -> Paths:
    """Paths pointing at a tmp fixture tree, but tariffs from the real
    model/tariffs/ folder (we want to exercise the actual tariff loaders
    end-to-end, not test fixtures)."""
    return Paths(
        edp_clean_dir=root / 'data' / 'clean' / 'edp',
        edp_metadata_dir=root / 'data' / 'processed' / 'edp',
        nem_prices_dir=root / 'data' / 'clean' / 'nem_prices',
        tariffs_dir=PRODUCTION_TARIFFS_DIR,
    )


def _scenario_nsw_solar(tmp_path) -> ScenarioConfig:
    return ScenarioConfig(
        state='NSW', year=2022, cohort='solar',
        network_tariff_id='ausgrid_ea025_network_2024',
        retail_tariff_id='powershop_tou_2024',
    )


def _scenario_nsw_solar_battery() -> ScenarioConfig:
    return ScenarioConfig(
        state='NSW', year=2022, cohort='solar_battery',
        coordination_type='self_consumption',
        network_tariff_id='ausgrid_ea025_network_2024',
        retail_tariff_id='amber_wholesale_2024',
        battery_spec=BatterySpec(
            capacity_kwh=13.5, backup_reserve_pct=10.0, power_kw=5.0,
            charge_efficiency=0.95, discharge_efficiency=0.95,
        ),
    )


# ---------- validate_scenario ----------

def test_validate_scenario_success(tmp_path):
    build_full_fixture(tmp_path, year=2022)
    validate_scenario(_scenario_nsw_solar(tmp_path), _paths_for(tmp_path))
    # No exception = pass


def test_validate_scenario_missing_metadata(tmp_path):
    # No fixture built - metadata file absent
    with pytest.raises(ScenarioValidationError, match='metadata file not found'):
        validate_scenario(_scenario_nsw_solar(tmp_path), _paths_for(tmp_path))


def test_validate_scenario_missing_price_file(tmp_path):
    build_full_fixture(tmp_path, year=2022)
    # Delete the price file
    price = tmp_path / 'data' / 'clean' / 'nem_prices' / '2022_nem_prices.parquet'
    price.unlink()
    with pytest.raises(ScenarioValidationError, match='price file not found'):
        validate_scenario(_scenario_nsw_solar(tmp_path), _paths_for(tmp_path))


def test_validate_scenario_nan_prices(tmp_path):
    build_full_fixture(tmp_path, year=2022)
    price = tmp_path / 'data' / 'clean' / 'nem_prices' / '2022_nem_prices.parquet'
    df = pd.read_parquet(price)
    df.loc[100, 'NSW1_rrp'] = float('nan')
    df.to_parquet(price, engine='pyarrow', compression='snappy', index=False)
    with pytest.raises(ScenarioValidationError, match='NaN values'):
        validate_scenario(_scenario_nsw_solar(tmp_path), _paths_for(tmp_path))


def test_validate_scenario_no_sites_for_state(tmp_path):
    build_full_fixture(tmp_path, year=2022)
    # The fixture has NSW and VIC sites; QLD has none
    scenario = ScenarioConfig(
        state='QLD', year=2022, cohort='solar',
        network_tariff_id='ausgrid_ea025_network_2024',
        retail_tariff_id='powershop_tou_2024',
    )
    with pytest.raises(ScenarioValidationError, match='no sites in metadata'):
        validate_scenario(scenario, _paths_for(tmp_path))


# ---------- run_scenario ----------

def test_run_scenario_solar_returns_result(tmp_path):
    build_full_fixture(tmp_path, year=2022)
    result = run_scenario(_scenario_nsw_solar(tmp_path), _paths_for(tmp_path))
    assert isinstance(result, ScenarioResult)
    # Fixture has 2 NSW sites (S0001, S0002)
    assert set(result.site_traces) == {'S0001', 'S0002'}
    assert set(result.site_bills) == {'S0001', 'S0002'}
    # with_monthly_bills defaults to False -> site_monthly_bills stays None,
    # distinguishing "not requested" from "requested, nothing to write".
    assert result.site_monthly_bills is None


def test_run_scenario_solar_battery_returns_result(tmp_path):
    build_full_fixture(tmp_path, year=2022)
    result = run_scenario(_scenario_nsw_solar_battery(), _paths_for(tmp_path))
    # Both NSW sites should have a battery dispatch trace
    for site_id, trace in result.site_traces.items():
        # battery_soc_kwh should not be all zero in a fixture with non-zero solar
        assert (trace.battery_soc_kwh > 0).any(), \
            f"{site_id}: battery never charged"
    assert result.site_monthly_bills is None


def test_run_scenario_solar_with_monthly_bills(tmp_path):
    """Regression guard for the standard (no_cer/solar/self_consumption)
    dispatch path: with_monthly_bills=True must populate
    result.site_monthly_bills for every site, and each site's monthly
    household_total must reconcile EXACTLY with its annual site_bills
    entry -- both fixture sites are NSW (DST-observing), and _span_days
    counts distinct (month, day) pairs ignoring year specifically so this
    reconciles exactly even across the December/January year boundary
    (see billing.py's _span_days docstring and test_billing_monthly.py's
    test_monthly_bills_nsw_dst_year_boundary_reconciles_exactly, which
    pins the mechanism with round numbers). Two earlier, incomplete
    versions of this fix left a small residual here (a single duplicated
    day, then briefly six); if this regresses to either, it's a sign
    _span_days has drifted back to counting year-qualified dates.
    """
    build_full_fixture(tmp_path, year=2022)
    result = run_scenario(_scenario_nsw_solar(tmp_path), _paths_for(tmp_path),
                         with_monthly_bills=True)
    assert result.site_monthly_bills is not None
    assert set(result.site_monthly_bills) == {'S0001', 'S0002'}
    for site_id, monthly in result.site_monthly_bills.items():
        assert len(monthly) > 0, f"{site_id}: no months present"
        total = sum(b.household_total for b in monthly.values())
        annual_total = result.site_bills[site_id].household_total
        assert total == pytest.approx(annual_total), (
            f"{site_id}: monthly household_total sum ({total}) doesn't "
            f"reconcile exactly with the annual bill ({annual_total}) -- "
            f"check _span_days for a regression"
        )


def test_run_scenario_progress_callback(tmp_path):
    build_full_fixture(tmp_path, year=2022)
    calls = []
    def cb(i, n, site_id):
        calls.append((i, n, site_id))
    run_scenario(_scenario_nsw_solar(tmp_path), _paths_for(tmp_path),
                 progress_callback=cb)
    assert len(calls) == 2
    assert calls[0][:2] == (1, 2)
    assert calls[1][:2] == (2, 2)


# ---------- ScenarioResult billing-context population (all LP-based
# coordination types) ----------
#
# Regression guard for the with_monthly_bills wiring: run_scenario's
# with_monthly_bills flag must be threaded through all 5 dispatch paths
# (some of which run per-site worker functions in separate processes via
# ProcessPoolExecutor, so the flag has to survive a pickle round-trip in
# the worker's args tuple), and each path's ScenarioResult(...) construction
# must populate site_monthly_bills. build_full_fixture's full-year data
# makes a real LP solve per coordination type too slow for a unit test
# (~15-25 s per site), so -- following the same approach
# test_orchestrate_vpp.py already uses for vpp_sc_event -- ingest() and
# validate_scenario() are patched and the dispatch runs in-process against a
# tiny synthetic 1-day (288-interval) single-site trace. This checks the
# wiring and month-reconciliation, not dispatch correctness (covered
# elsewhere, e.g. test_perfect_foresight_lp.py) or settlement-arithmetic
# correctness (covered by test_billing_monthly.py).

def _synthetic_ctx_for_lp(n_intervals: int = 288) -> MagicMock:
    """Single NSW site, small solar/load pattern, matches the pattern used
    in test_orchestrate_vpp.py's _build_synthetic_ctx (kept separate here
    since that helper isn't exported and the two files should stay
    independently readable)."""
    idx = pd.date_range('2022-01-01 00:05', periods=n_intervals, freq='5min',
                        tz='Etc/GMT-10')
    t = np.arange(n_intervals) / n_intervals
    solar = np.maximum(0, 2.0 * np.sin(np.pi * t) * (5 / 60))  # kWh/interval
    load = np.full(n_intervals, 0.5 * (5 / 60))                # flat 0.5 kW

    site_trace = pd.DataFrame({
        'pv_site_net_kwh': solar,
        'gross_load_kwh': load,
    }, index=idx)

    prices = pd.Series(np.full(n_intervals, 50.0), index=idx, name='NSW1_rrp')

    metadata = pd.DataFrame({
        'edp_site_id': ['S0001'],
        'state': ['NSW'],
    })

    ctx = MagicMock()
    ctx.site_traces = {'S0001': site_trace}
    ctx.prices = prices
    ctx.site_metadata = metadata
    ctx.network_tariff = NetworkTariff(
        name='Test Network', state='NSW',
        usage_tariffs=(TariffPeriod(0, 24, 8.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 0.0),),
        daily_c_per_day=50.0,
        usage_dlf=1.0, export_dlf=1.0, import_tlf=1.0, export_tlf=1.0,
    )
    ctx.retail_tariff = RetailTariff(
        name='Test TOU', state='NSW', type='tou',
        daily_c_per_day=100.0,
        usage_tariffs=(TariffPeriod(0, 24, 30.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 5.0),),
        usage_other_c_per_kwh=0.0, feedin_other_c_per_kwh=0.0,
    )
    return ctx


def _lp_battery() -> BatterySpec:
    return BatterySpec(
        capacity_kwh=13.5, backup_reserve_pct=10.0, power_kw=5.0,
        charge_efficiency=0.95, discharge_efficiency=0.95, c_deg_per_kwh=0.005,
    )


@pytest.mark.parametrize('coordination_type', [
    'perfect_foresight',
    'perfect_foresight_custom',
    'perfect_foresight_advanced',
])
@patch('model.orchestrate.validate_scenario')
@patch('model.orchestrate.ingest')
def test_run_scenario_populates_billing_context_for_lp_types(
    mock_ingest, mock_validate, coordination_type, tmp_path,
):
    """Every LP-based coordination_type must, when with_monthly_bills=True,
    populate result.site_monthly_bills with a reconciling monthly breakdown
    for every site -- the regression this guards against is a dispatch path
    (or its worker function, some of which run in separate processes via
    ProcessPoolExecutor) forgetting to thread with_monthly_bills through, or
    forgetting to populate site_monthly_bills in its ScenarioResult
    construction. See module-level comment above for why this uses a
    synthetic ctx rather than build_full_fixture (speed)."""
    ctx = _synthetic_ctx_for_lp()
    mock_ingest.return_value = ctx

    paths = Paths(
        edp_clean_dir=tmp_path, edp_metadata_dir=tmp_path,
        nem_prices_dir=tmp_path, tariffs_dir=tmp_path,
    )
    scenario = ScenarioConfig(
        state='NSW', year=2022, cohort='solar_battery',
        coordination_type=coordination_type,
        network_tariff_id='ausgrid_ea025_network_2024',  # unused: ctx is mocked
        retail_tariff_id='powershop_tou_2024',            # unused: ctx is mocked
        battery_spec=_lp_battery(),
    )

    # with_monthly_bills=False (default): explicitly stays None.
    result_off = run_scenario(scenario, paths, n_workers=1)
    assert result_off.site_monthly_bills is None, (
        f"{coordination_type}: site_monthly_bills should be None when "
        f"with_monthly_bills=False"
    )

    # with_monthly_bills=True: populated and reconciling with the annual bill.
    result = run_scenario(scenario, paths, n_workers=1, with_monthly_bills=True)

    assert isinstance(result, ScenarioResult)
    assert 'S0001' in result.site_traces
    assert 'S0001' in result.site_bills
    assert result.site_monthly_bills is not None, (
        f"{coordination_type}: result.site_monthly_bills not set with "
        f"with_monthly_bills=True"
    )
    assert 'S0001' in result.site_monthly_bills, (
        f"{coordination_type}: S0001 missing from site_monthly_bills"
    )
    monthly = result.site_monthly_bills['S0001']
    assert len(monthly) > 0, f"{coordination_type}: no months present"
    total = sum(b.household_total for b in monthly.values())
    annual_total = result.site_bills['S0001'].household_total
    diff = abs(total - annual_total)
    # Bounded, not exact: _synthetic_ctx_for_lp is a 1-day (288-interval)
    # AEST index for NSW (DST-observing). Converted to local time, that
    # single AEST day's last ~hour falls on local Jan 2, not Jan 1 -- so
    # the annual bill's day count (AEST-based: 1 distinct date) and the
    # monthly bucket's day count (local-based, by design -- see
    # billing.py's _span_days docstring: 2 distinct local dates) can
    # legitimately differ by up to a day's daily charges for a
    # short/partial-period fixture like this one, purely as a consequence
    # of AEST and local dates diverging under DST. This test's purpose is
    # the with_monthly_bills wiring, not billing precision under DST --
    # that's covered precisely, with a full real calendar year, by
    # test_billing_monthly.py's dedicated NSW/QLD DST tests.
    assert diff < 5.0, (
        f"{coordination_type}: monthly household_total sum ({total}) "
        f"differs from the annual bill ({annual_total}) by {diff:.2f}, "
        f"more than the expected sub-one-day fixture-DST divergence should "
        f"allow"
    )


# ---------- rolling_perfect_foresight_lp ----------
# Mirrors the LP-types billing test above for speed and structure (a
# mocked ingest/validate_scenario against a tiny synthetic in-process ctx,
# not a full fixture), but needs its own ctx: build_price_stack_offsets
# requires a wholesale_exposed retail tariff, which _synthetic_ctx_for_lp's
# TOU tariff is not.

def _wholesale_retail_tariff() -> RetailTariff:
    return RetailTariff(
        name='Test Wholesale', state='NSW', type='wholesale_exposed',
        daily_c_per_day=80.0,
        usage_tariffs=(), usage_tariffs_weekend=None, feedin_tariffs=(),
        usage_other_c_per_kwh=2.0, feedin_other_c_per_kwh=0.0,
    )


def _synthetic_ctx_for_rolling(n_intervals: int = 288) -> MagicMock:
    """Same shape as _synthetic_ctx_for_lp, wholesale tariff swapped in."""
    ctx = _synthetic_ctx_for_lp(n_intervals)
    ctx.retail_tariff = _wholesale_retail_tariff()
    return ctx


@patch('model.orchestrate.validate_scenario')
@patch('model.orchestrate.ingest')
def test_run_scenario_rolling_populates_diagnostics_and_monthly_bills(
    mock_ingest, mock_validate, tmp_path,
):
    """Regression guard for _run_rolling_perfect_foresight_lp /
    _rolling_perfect_foresight_site_worker: result.rolling_diagnostics must
    be populated (None for every OTHER coordination type -- see
    ScenarioResult's own docstring), keyed by site, with a passing causality
    check, and with_monthly_bills must be threaded through exactly as it is
    for the perfect_foresight family (see
    test_run_scenario_populates_billing_context_for_lp_types above).
    forecast_source='truncated_perfect' throughout so this needs no
    pre-dispatch store fixture -- it is testing the orchestrator wiring, not
    forecast quality (that is rolling_optimisation_spec.md's measurement
    ladder's job)."""
    ctx = _synthetic_ctx_for_rolling()
    mock_ingest.return_value = ctx

    paths = Paths(
        edp_clean_dir=tmp_path, edp_metadata_dir=tmp_path,
        nem_prices_dir=tmp_path, tariffs_dir=tmp_path,
    )
    scenario = ScenarioConfig(
        state='NSW', year=2022, cohort='solar_battery',
        coordination_type='rolling_perfect_foresight_lp',
        network_tariff_id='ausgrid_ea025_network_2024',  # unused: ctx is mocked
        retail_tariff_id='amber_wholesale_2024',          # unused: ctx is mocked
        battery_spec=_lp_battery(),
        rolling=RollingConfig(forecast_source='truncated_perfect',
                              horizon_hours=1.0),
    )

    result_off = run_scenario(scenario, paths, n_workers=1)
    assert result_off.site_monthly_bills is None
    assert result_off.rolling_diagnostics is not None
    assert 'S0001' in result_off.rolling_diagnostics
    assert result_off.rolling_diagnostics['S0001'].causality_ok

    result = run_scenario(scenario, paths, n_workers=1, with_monthly_bills=True)
    assert result.site_monthly_bills is not None
    assert 'S0001' in result.site_monthly_bills
    monthly = result.site_monthly_bills['S0001']
    total = sum(b.household_total for b in monthly.values())
    annual_total = result.site_bills['S0001'].household_total
    assert abs(total - annual_total) < 5.0, (
        f"monthly household_total sum ({total}) doesn't reconcile with the "
        f"annual bill ({annual_total}); see the DST-fixture note on "
        f"test_run_scenario_populates_billing_context_for_lp_types above "
        f"for why this is a bound rather than an exact match"
    )


def test_validate_scenario_rolling_rejects_non_wholesale_tariff(tmp_path):
    build_full_fixture(tmp_path, year=2022)
    scenario = ScenarioConfig(
        state='NSW', year=2022, cohort='solar_battery',
        coordination_type='rolling_perfect_foresight_lp',
        network_tariff_id='ausgrid_ea025_network_2024',
        retail_tariff_id='powershop_tou_2024',   # TOU, not wholesale_exposed
        battery_spec=_lp_battery(),
        rolling=RollingConfig(forecast_source='truncated_perfect'),
    )
    with pytest.raises(ScenarioValidationError, match='wholesale_exposed'):
        validate_scenario(scenario, _paths_for(tmp_path))


def test_validate_scenario_rolling_rejects_multi_site_retain_plans(tmp_path):
    """The fixture has 2 NSW sites; retain_plans_from/to is only permitted
    for a scenario resolving to exactly one."""
    build_full_fixture(tmp_path, year=2022)
    scenario = ScenarioConfig(
        state='NSW', year=2022, cohort='solar_battery',
        coordination_type='rolling_perfect_foresight_lp',
        network_tariff_id='ausgrid_ea025_network_2024',
        retail_tariff_id='amber_wholesale_2024',
        battery_spec=_lp_battery(),
        rolling=RollingConfig(forecast_source='truncated_perfect',
                              retain_plans_from='2022-06-01',
                              retain_plans_to='2022-06-02'),
    )
    with pytest.raises(ScenarioValidationError, match='single-site'):
        validate_scenario(scenario, _paths_for(tmp_path))


def test_validate_scenario_rolling_rejects_non_reportable_horizon(tmp_path):
    """forecast_source='perfect' at a horizon covering the whole scenario
    period is the correctness gate (see registry.is_reportable), not a
    reportable result -- ported from run_cohort_rungs.py's per-rung guard."""
    build_full_fixture(tmp_path, year=2022)
    scenario = ScenarioConfig(
        state='NSW', year=2022, cohort='solar_battery',
        coordination_type='rolling_perfect_foresight_lp',
        network_tariff_id='ausgrid_ea025_network_2024',
        retail_tariff_id='amber_wholesale_2024',
        battery_spec=_lp_battery(),
        rolling=RollingConfig(forecast_source='perfect', horizon_hours=10_000.0),
    )
    with pytest.raises(ScenarioValidationError, match='correctness gate'):
        validate_scenario(scenario, _paths_for(tmp_path))


def test_load_scenarios_rolling_perfect_foresight_lp_accepted(tmp_path):
    """coordination_type='rolling_perfect_foresight_lp' with a full rolling:
    block must be accepted by load_scenarios (regression guard for
    main._parse_rolling, exercising every field it parses -- usage_forecast/
    solar_forecast/on_degraded_period/max_staleness_hours were all added
    after this style of test was first written for perfect_foresight_advanced
    above, and each was independently found missing from the known-key set
    at least once)."""
    yaml_path = tmp_path / 'rolling.yaml'
    _write_yaml(yaml_path, {
        'name': 'rolling_test',
        'states': ['NSW'], 'years': [2022],
        'cohort': 'solar_battery',
        'coordination_type': 'rolling_perfect_foresight_lp',
        'network_tariff_id': 'ausgrid_ea025_network_2024',
        'retail_tariff_id': 'amber_wholesale_2024',
        'battery_spec': {
            'capacity_kwh': 13.5, 'backup_reserve_pct': 10.0,
            'power_kw': 5.0, 'charge_efficiency': 0.95,
            'discharge_efficiency': 0.95,
        },
        'rolling': {
            'forecast_source': 'predispatch',
            'horizon_hours': 18.0,
            'current_interval_price': 'price_and_next',
            'usage_forecast': {},
            'solar_forecast': {},
            'on_degraded_period': 'infill',
            'max_staleness_hours': 800.0,
        },
    })
    run_name, scenarios = load_scenarios(yaml_path)
    assert len(scenarios) == 1
    s = scenarios[0]
    assert s.coordination_type == 'rolling_perfect_foresight_lp'
    assert s.rolling.forecast_source == 'predispatch'
    assert s.rolling.current_interval_price == 'price_and_next'
    assert s.rolling.usage_forecast is not None
    assert s.rolling.solar_forecast is not None
    assert s.rolling.on_degraded_period == 'infill'
    assert s.rolling.max_staleness_hours == 800.0


def test_write_rolling_diagnostics_csv(tmp_path):
    ctx = _synthetic_ctx_for_rolling()
    paths = Paths(
        edp_clean_dir=tmp_path, edp_metadata_dir=tmp_path,
        nem_prices_dir=tmp_path, tariffs_dir=tmp_path,
    )
    scenario = ScenarioConfig(
        state='NSW', year=2022, cohort='solar_battery',
        coordination_type='rolling_perfect_foresight_lp',
        network_tariff_id='ausgrid_ea025_network_2024',
        retail_tariff_id='amber_wholesale_2024',
        battery_spec=_lp_battery(),
        rolling=RollingConfig(forecast_source='truncated_perfect',
                              horizon_hours=1.0),
    )
    with patch('model.orchestrate.validate_scenario'), \
         patch('model.orchestrate.ingest', return_value=ctx):
        result = run_scenario(scenario, paths, n_workers=1)
    out = tmp_path / 'rolling_diagnostics.csv'
    write_rolling_diagnostics_csv(result, out)
    df = pd.read_csv(out)
    assert len(df) == 1
    assert df.iloc[0]['site_id'] == 'S0001'
    assert 'causality_ok' in df.columns
    assert 'n_fallbacks' in df.columns


def test_write_rolling_diagnostics_csv_noop_for_non_rolling(tmp_path):
    """Called unconditionally in main.py for every scenario (same style as
    write_bills_csv); must write nothing when rolling_diagnostics is None,
    rather than raising or writing an empty file."""
    build_full_fixture(tmp_path, year=2022)
    result = run_scenario(_scenario_nsw_solar(tmp_path), _paths_for(tmp_path))
    assert result.rolling_diagnostics is None
    out = tmp_path / 'rolling_diagnostics.csv'
    write_rolling_diagnostics_csv(result, out)
    assert not out.exists()


# ---------- Per-site export-limit resolution ----------

def _export_scenario(epp=None):
    return ScenarioConfig(
        state='NSW', year=2022, cohort='solar',
        network_tariff_id='ausgrid_ea025_network_2024',
        retail_tariff_id='powershop_tou_2024',
        export_limit_kw_per_phase=epp,
    )


def _phase_meta():
    return pd.DataFrame({
        'edp_site_id': ['A', 'B', 'C'],
        'ac_load_net_phases': [1, 3, 1],
        'pv_site_net_phases': [1, 1, 3],
    })


def test_resolve_export_limits_none_gives_inf():
    limits = _resolve_per_site_export_limits(_export_scenario(), _phase_meta())
    assert set(limits) == {'A', 'B', 'C'}
    assert all(math.isinf(v) for v in limits.values())


def test_resolve_export_limits_epp_times_max_phases():
    # A: max(1,1)*5=5 ; B: max(3,1)*5=15 ; C: max(1,3)*5=15
    limits = _resolve_per_site_export_limits(_export_scenario(5.0), _phase_meta())
    assert limits == {'A': 5.0, 'B': 15.0, 'C': 15.0}


def test_resolve_export_limits_missing_column_raises():
    with pytest.raises(ValueError, match='phase columns'):
        _resolve_per_site_export_limits(
            _export_scenario(5.0),
            _phase_meta().drop(columns=['pv_site_net_phases']))


def test_resolve_export_limits_nan_phase_raises():
    meta = _phase_meta()
    meta.loc[1, 'ac_load_net_phases'] = np.nan
    with pytest.raises(ValueError, match='missing phase data'):
        _resolve_per_site_export_limits(_export_scenario(5.0), meta)


# ---------- write helpers ----------

def test_write_bills_csv(tmp_path):
    build_full_fixture(tmp_path, year=2022)
    result = run_scenario(_scenario_nsw_solar(tmp_path), _paths_for(tmp_path))
    out = tmp_path / 'bills.csv'
    write_bills_csv(result, out)
    df = pd.read_csv(out)
    assert len(df) == 2
    expected_cols = {
        'site_id', 'plan_type', 'wholesale_usage_cost', 'wholesale_feedin_credit',
        'network_usage_cost', 'network_feedin_credit', 'network_daily_charges',
        'retail_usage_cost', 'retail_feedin_credit', 'retail_daily_charges',
        'retail_other_credits', 'household_total', 'retailer_margin',
    }
    assert set(df.columns) == expected_cols


def test_write_scenario_summary(tmp_path):
    build_full_fixture(tmp_path, year=2022)
    result = run_scenario(_scenario_nsw_solar(tmp_path), _paths_for(tmp_path))
    out = tmp_path / 'summary.txt'
    text = write_scenario_summary(result, out)
    assert out.exists()
    # spot checks
    assert 'NSW_2022' in text
    assert 'household_total' in text
    assert 'COHORT TOTALS' in text


# ---------- YAML loading and cross-product ----------

def _write_yaml(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.safe_dump(data, f)


def test_load_scenarios_single_scenario(tmp_path):
    yaml_path = tmp_path / 'single.yaml'
    _write_yaml(yaml_path, {
        'name': 'test_single',
        'states': ['NSW'], 'years': [2022],
        'cohort': 'solar',
        'network_tariff_id': 'ausgrid_ea025_network_2024',
        'retail_tariff_id': 'powershop_tou_2024',
    })
    run_name, scenarios = load_scenarios(yaml_path)
    assert run_name == 'test_single'
    assert len(scenarios) == 1
    assert scenarios[0].scenario_id == 'NSW_2022'


def test_load_scenarios_multi_battery_cross_product(tmp_path):
    yaml_path = tmp_path / 'multi.yaml'
    _write_yaml(yaml_path, {
        'states': ['NSW'], 'years': [2022],
        'cohort': 'solar_battery',
        'coordination_type': 'self_consumption',
        'network_tariff_id': 'ausgrid_ea025_network_2024',
        'retail_tariff_id': 'amber_wholesale_2024',
        'battery_specs': [
            {'name': 'small', 'capacity_kwh': 6, 'backup_reserve_pct': 10,
             'power_kw': 3, 'charge_efficiency': 0.95,
             'discharge_efficiency': 0.95},
            {'name': 'large', 'capacity_kwh': 27, 'backup_reserve_pct': 10,
             'power_kw': 7, 'charge_efficiency': 0.95,
             'discharge_efficiency': 0.95},
        ],
    })
    run_name, scenarios = load_scenarios(yaml_path)
    assert len(scenarios) == 2
    assert {s.scenario_id for s in scenarios} == {'NSW_2022_small', 'NSW_2022_large'}


def test_load_scenarios_multi_state_year_cross_product(tmp_path):
    yaml_path = tmp_path / 'multi.yaml'
    _write_yaml(yaml_path, {
        'states': ['NSW', 'VIC'],
        'years': [2022, 2023],
        'cohort': 'solar',
        'network_tariff_ids': {
            'NSW': {2022: 'ausgrid_ea025_network_2024',
                    2023: 'ausgrid_ea025_network_2024'},
            'VIC': {2022: 'ausgrid_ea025_network_2024',  # placeholder
                    2023: 'ausgrid_ea025_network_2024'},
        },
        'retail_tariff_ids': {
            'NSW': {2022: 'powershop_tou_2024', 2023: 'powershop_tou_2024'},
            'VIC': {2022: 'powershop_tou_2024', 2023: 'powershop_tou_2024'},
        },
    })
    run_name, scenarios = load_scenarios(yaml_path)
    assert len(scenarios) == 4  # 2 states * 2 years * 1 (no battery)
    ids = {s.scenario_id for s in scenarios}
    assert ids == {'NSW_2022', 'NSW_2023', 'VIC_2022', 'VIC_2023'}


def test_load_scenarios_singular_tariff_with_multi_state_rejects(tmp_path):
    yaml_path = tmp_path / 'bad.yaml'
    _write_yaml(yaml_path, {
        'states': ['NSW', 'VIC'], 'years': [2022],
        'cohort': 'solar',
        'network_tariff_id': 'ausgrid_ea025_network_2024',  # singular not allowed
        'retail_tariff_id': 'powershop_tou_2024',
    })
    with pytest.raises(ValueError, match='string form'):
        load_scenarios(yaml_path)


def test_load_scenarios_battery_specs_require_name(tmp_path):
    yaml_path = tmp_path / 'bad.yaml'
    _write_yaml(yaml_path, {
        'states': ['NSW'], 'years': [2022],
        'cohort': 'solar_battery',
        'coordination_type': 'self_consumption',
        'network_tariff_id': 'ausgrid_ea025_network_2024',
        'retail_tariff_id': 'amber_wholesale_2024',
        'battery_specs': [
            # missing name
            {'capacity_kwh': 13.5, 'backup_reserve_pct': 10,
             'power_kw': 5, 'charge_efficiency': 0.95,
             'discharge_efficiency': 0.95},
        ],
    })
    with pytest.raises(ValueError, match="'name'"):
        load_scenarios(yaml_path)


def test_load_scenarios_no_cer_rejects_battery(tmp_path):
    yaml_path = tmp_path / 'bad.yaml'
    _write_yaml(yaml_path, {
        'states': ['NSW'], 'years': [2022],
        'cohort': 'no_cer',
        'network_tariff_id': 'ausgrid_ea025_network_2024',
        'retail_tariff_id': 'powershop_tou_2024',
        'battery_spec': {'capacity_kwh': 13.5, 'backup_reserve_pct': 10,
                         'power_kw': 5, 'charge_efficiency': 0.95,
                         'discharge_efficiency': 0.95},
    })
    with pytest.raises(ValueError, match='not allowed'):
        load_scenarios(yaml_path)


def test_load_scenarios_perfect_foresight_advanced_accepted(tmp_path):
    """coordination_type='perfect_foresight_advanced' must be accepted by
    load_scenarios without error (regression guard for the Literal update)."""
    yaml_path = tmp_path / 'adv.yaml'
    _write_yaml(yaml_path, {
        'name': 'adv_test',
        'states': ['NSW'], 'years': [2022],
        'cohort': 'solar_battery',
        'coordination_type': 'perfect_foresight_advanced',
        'network_tariff_id': 'ausgrid_ea025_network_2024',
        'retail_tariff_id': 'globird_advanced_tou_2025',
        'battery_spec': {
            'capacity_kwh': 13.5, 'backup_reserve_pct': 10.0,
            'power_kw': 5.0, 'charge_efficiency': 0.95,
            'discharge_efficiency': 0.95,
        },
    })
    run_name, scenarios = load_scenarios(yaml_path)
    assert len(scenarios) == 1
    assert scenarios[0].coordination_type == 'perfect_foresight_advanced'
    assert scenarios[0].scenario_id == 'NSW_2022'


# ---------- main() end-to-end ----------

def test_main_runs_and_writes_outputs(tmp_path):
    build_full_fixture(tmp_path, year=2022)

    yaml_path = tmp_path / 'config.yaml'
    _write_yaml(yaml_path, {
        'name': 'integration_test',
        'states': ['NSW'], 'years': [2022],
        'cohort': 'solar',
        'network_tariff_id': 'ausgrid_ea025_network_2024',
        'retail_tariff_id': 'powershop_tou_2024',
    })

    rc = main([str(yaml_path), '--project-root', str(tmp_path),
               '--tariffs-dir', str(PRODUCTION_TARIFFS_DIR)])
    assert rc == 0

    scenario_dir = (tmp_path / 'data' / 'output' / 'runs'
                    / 'integration_test' / 'scenarios' / 'NSW_2022')
    assert (scenario_dir / 'bills.csv').exists()
    assert (scenario_dir / 'summary.txt').exists()
    run_summary = (tmp_path / 'data' / 'output' / 'runs'
                   / 'integration_test' / 'run_summary.txt')
    assert run_summary.exists()


def test_main_aborts_on_validation_failure(tmp_path):
    # Fixture not built - metadata is missing, should abort cleanly
    yaml_path = tmp_path / 'config.yaml'
    _write_yaml(yaml_path, {
        'states': ['NSW'], 'years': [2022],
        'cohort': 'solar',
        'network_tariff_id': 'ausgrid_ea025_network_2024',
        'retail_tariff_id': 'powershop_tou_2024',
    })
    rc = main([str(yaml_path), '--project-root', str(tmp_path)])
    assert rc == 3  # validation-failure exit code


def test_main_save_trace(tmp_path):
    build_full_fixture(tmp_path, year=2022)
    yaml_path = tmp_path / 'config.yaml'
    _write_yaml(yaml_path, {
        'name': 'trace_test',
        'states': ['NSW'], 'years': [2022],
        'cohort': 'solar',
        'network_tariff_id': 'ausgrid_ea025_network_2024',
        'retail_tariff_id': 'powershop_tou_2024',
    })
    rc = main([str(yaml_path), '--save-trace', 'S0001',
               '--project-root', str(tmp_path),
               '--tariffs-dir', str(PRODUCTION_TARIFFS_DIR)])
    assert rc == 0
    trace_path = (tmp_path / 'data' / 'output' / 'runs' / 'trace_test'
                  / 'scenarios' / 'NSW_2022' / 'trace_S0001.csv')
    assert trace_path.exists()

# ---------- mixed retail tariffs (model_spec.md, Validation) ----------

def _mixed_scenario(cohort, coordination_type):
    kwargs = dict(
        state='NSW', year=2022, cohort=cohort,
        network_tariff_id='ausgrid_ea025_network_2024',
        retail_tariff_id='NSW_tesla_electric_mixed_2026',
    )
    if coordination_type is not None:
        kwargs.update(coordination_type=coordination_type,
                      battery_spec=_lp_battery())
    return ScenarioConfig(**kwargs)


@pytest.mark.parametrize('cohort, coordination_type', [
    ('solar', None),
    ('no_cer', None),
    ('solar_battery', 'self_consumption'),
    ('solar_battery', 'perfect_foresight_advanced'),
])
def test_validate_scenario_accepts_mixed_retail_tariff(tmp_path, cohort,
                                                       coordination_type):
    build_full_fixture(tmp_path, year=2022)
    validate_scenario(_mixed_scenario(cohort, coordination_type),
                      _paths_for(tmp_path))


@pytest.mark.parametrize('coordination_type', [
    'perfect_foresight', 'perfect_foresight_custom',
])
def test_validate_scenario_rejects_mixed_for_other_types(tmp_path,
                                                         coordination_type):
    build_full_fixture(tmp_path, year=2022)
    with pytest.raises(ScenarioValidationError,
                       match=f"type 'mixed', which coordination_type="
                             f"'{coordination_type}' does not support"):
        validate_scenario(_mixed_scenario('solar_battery', coordination_type),
                          _paths_for(tmp_path))


def test_validate_scenario_runs_the_mixed_ordering_proof(tmp_path, monkeypatch):
    """perfect_foresight_advanced on a mixed plan builds the dispatch stack
    during validation, so a tariff pair whose ordering can fail is refused
    before ingest."""
    import model.orchestrate as orch
    build_full_fixture(tmp_path, year=2022)

    def refuse(*a, **k):
        raise ValueError("ordering can fail (test)")
    monkeypatch.setattr(orch, 'build_mixed_dispatch_stack', refuse)
    with pytest.raises(ScenarioValidationError, match="ordering can fail"):
        validate_scenario(_mixed_scenario('solar_battery',
                                          'perfect_foresight_advanced'),
                          _paths_for(tmp_path))


def test_perfect_foresight_advanced_worker_passes_the_mixed_curtail_cost(
        monkeypatch):
    """The worker derives the curtailment cost from the plan, so the
    tie-break reaches the LP without a new field in the worker tuple."""
    import model.orchestrate as orch
    from model.core.prices import MIXED_CURTAIL_COST_PER_KWH
    from model.core.tariff import load_retail_tariff, load_network_tariff

    captured = {}

    class _Stop(Exception):
        pass

    def fake_lp(*args, **kwargs):
        captured.update(kwargs)
        raise _Stop

    monkeypatch.setattr(orch, 'perfect_foresight_lp_advanced', fake_lp)
    idx = pd.date_range('2022-07-04 00:05', periods=24, freq='5min',
                        tz='Etc/GMT-10')
    zeros = pd.Series(np.zeros(24), index=idx)
    prices = pd.Series(np.full(24, 100.0), index=idx)
    for tid, expected in (('NSW_tesla_electric_mixed_2026',
                           MIXED_CURTAIL_COST_PER_KWH),
                          ('powershop_tou_2024', 0.0)):
        rt = load_retail_tariff(tid, PRODUCTION_TARIFFS_DIR)
        nt = load_network_tariff('ausgrid_ea025_network_2024',
                                 PRODUCTION_TARIFFS_DIR)
        args = ('S1', zeros, zeros, zeros, zeros, _lp_battery(), prices, nt,
                rt, 1.35, np.inf, 5 / 60, False, 'NSW', False)
        captured.clear()
        with pytest.raises(_Stop):
            orch._perfect_foresight_advanced_site_worker(args)
        assert captured['curtail_cost_per_kwh_dollars'] == expected, tid
