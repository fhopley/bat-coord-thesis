"""Tests for VPP-specific orchestrate.py additions.

Ported from the pre-removal version of this file as part of removing
perfect_foresight_vpp.py (superseded by vpp_sc_event.py). op_retail_tariff_id,
ScenarioResult.vpp_traces, and validate_scenario's VPP checks are generic to
any 'vpp'-family coordination type -- confirmed still required by
vpp_sc_event.py's orchestrator wiring (_run_vpp_sc_event loads
scenario.op_retail_tariff_id directly) -- so these tests are adapted by
changing coordination_type='vpp' to coordination_type='vpp_sc_event'
throughout, not deleted. The wiring-test section (previously validating
_run_perfect_foresight_vpp) is rebuilt against _run_vpp_sc_event's actual
worker contract (_vpp_sc_event_site_worker, confirmed by direct code review)
rather than assumed to carry over unchanged.

NOTE: the exact ValueError message text asserted in the ScenarioConfig /
validate_scenario sections below is carried over from the pre-removal file
and has not been re-verified against the current model/core/scenario.py --
confirm these match before treating this file as passing. If scenario.py's
op_retail_tariff_id validation is scoped to coordination_type == 'vpp'
specifically (rather than covering the 'vpp'-family generally), it will need
updating alongside the perfect_foresight_vpp removal for these tests to pass.

Covers:
  - ScenarioConfig validation for op_retail_tariff_id
  - validate_scenario VPP checks (has_vpp guard, op tariff type guard)
  - ScenarioResult.vpp_traces field shape and defaults
  - _run_vpp_sc_event wiring via a synthetic in-process run

The validate_scenario tests use build_full_fixture + PRODUCTION_TARIFFS_DIR,
matching the pattern in test_orchestrate.py. They require agl_battery_rewards_vpp_2025.yaml
and agl_operator_wholesale_2025.yaml to be present in model/tariffs/.

The run_scenario tests patch ingest, load_retail_tariff, and validate_scenario,
running the VPP dispatch in-process against a synthetic 288-interval trace.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import dataclasses

import numpy as np
import pandas as pd
import pytest

from model.core.paths import Paths
from model.core.scenario import BatterySpec, ScenarioConfig
from model.core.site_result import SiteTraces, SiteBill, VppTraces
from model.core.tariff import (
    NetworkTariff, RetailTariff, TariffPeriod, VppConfig,
)
from model.orchestrate import (
    ScenarioResult, ScenarioValidationError, validate_scenario, run_scenario,
    resolve_min_export_price, write_scenario_summary,
)
from model.tests.fixtures.ingest_fixture_builder import build_full_fixture

PRODUCTION_TARIFFS_DIR = Path(__file__).parent.parent / 'tariffs'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bat() -> BatterySpec:
    return BatterySpec(
        capacity_kwh=13.5, backup_reserve_pct=10.0, power_kw=5.0,
        charge_efficiency=0.95, discharge_efficiency=0.95, c_deg_per_kwh=0.005,
    )


def _vpp_cfg() -> VppConfig:
    return VppConfig(
        exp_reward_c_per_kwh=100.0,
        imp_reward_c_per_kwh=100.0,
        kwh_limit=250.0,
        limit_applies_to='import_and_export',
        default_mode='self_consumption',
        event_window_hours=10.0,
        vpp_threshold_dollars_per_mwh=1000.0,
        min_export_price_mode='based_on_this_year',
    )


def _retail_vpp() -> RetailTariff:
    """Minimal TOU retail tariff with VPP config."""
    return RetailTariff(
        name='AGL VPP Test', state='NSW', type='tou',
        daily_c_per_day=130.0,
        usage_tariffs=(TariffPeriod(0, 24, 30.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 5.0),),
        usage_other_c_per_kwh=0.0,
        feedin_other_c_per_kwh=0.0,
        vpp=_vpp_cfg(),
    )


def _retail_plain() -> RetailTariff:
    """Minimal TOU retail tariff without VPP config."""
    return RetailTariff(
        name='Plain TOU', state='NSW', type='tou',
        daily_c_per_day=100.0,
        usage_tariffs=(TariffPeriod(0, 24, 30.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 5.0),),
        usage_other_c_per_kwh=0.0,
        feedin_other_c_per_kwh=0.0,
    )


def _retail_op() -> RetailTariff:
    """Minimal wholesale_exposed operator tariff (zero adders)."""
    return RetailTariff(
        name='Operator Wholesale', state='NSW', type='wholesale_exposed',
        daily_c_per_day=0.0,
        usage_tariffs=(),
        usage_tariffs_weekend=None,
        feedin_tariffs=(),
        usage_other_c_per_kwh=0.0,
        feedin_other_c_per_kwh=0.0,
    )


def _network() -> NetworkTariff:
    return NetworkTariff(
        name='Test Network', state='NSW',
        usage_tariffs=(TariffPeriod(0, 24, 8.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 0.0),),
        daily_c_per_day=50.0,
        usage_dlf=1.02, export_dlf=0.98,
        import_tlf=1.01, export_tlf=0.99,
    )


def _vpp_scenario(coordination_type: str = 'vpp_sc_event') -> ScenarioConfig:
    return ScenarioConfig(
        state='NSW', year=2022, cohort='solar_battery',
        coordination_type=coordination_type,
        network_tariff_id='ausgrid_ea025_network_2024',
        retail_tariff_id='agl_battery_rewards_vpp_2025',
        op_retail_tariff_id='agl_operator_wholesale_2025',
        battery_spec=_bat(),
    )


# ---------------------------------------------------------------------------
# ScenarioConfig validation for op_retail_tariff_id
# ---------------------------------------------------------------------------

def test_scenario_config_vpp_requires_op_tariff_id():
    """op_retail_tariff_id is required when coordination_type='vpp_sc_event'."""
    with pytest.raises(ValueError, match='op_retail_tariff_id is required'):
        ScenarioConfig(
            state='NSW', year=2022, cohort='solar_battery',
            coordination_type='vpp_sc_event',
            network_tariff_id='net', retail_tariff_id='ret',
            battery_spec=_bat(),
            # op_retail_tariff_id omitted
        )


def test_scenario_config_op_tariff_id_must_be_none_for_non_vpp():
    """op_retail_tariff_id must be None for non-vpp coordination types."""
    with pytest.raises(ValueError, match='op_retail_tariff_id must be None'):
        ScenarioConfig(
            state='NSW', year=2022, cohort='solar_battery',
            coordination_type='perfect_foresight',
            network_tariff_id='net', retail_tariff_id='ret',
            battery_spec=_bat(),
            op_retail_tariff_id='some_tariff',
        )


def test_scenario_config_vpp_valid():
    """A fully-specified vpp_sc_event ScenarioConfig constructs without error."""
    s = _vpp_scenario()
    assert s.coordination_type == 'vpp_sc_event'
    assert s.op_retail_tariff_id == 'agl_operator_wholesale_2025'


def test_scenario_config_non_vpp_op_tariff_none():
    """Non-VPP scenarios have op_retail_tariff_id=None by default."""
    s = ScenarioConfig(
        state='NSW', year=2022, cohort='solar_battery',
        coordination_type='perfect_foresight',
        network_tariff_id='net', retail_tariff_id='ret',
        battery_spec=_bat(),
    )
    assert s.op_retail_tariff_id is None


# ---------------------------------------------------------------------------
# validate_scenario VPP checks
# ---------------------------------------------------------------------------


def _paths_for(root: Path) -> Paths:
    """Paths pointing at tmp fixture tree with tariffs from real model/tariffs/."""
    return Paths(
        edp_clean_dir=root / 'data' / 'clean' / 'edp',
        edp_metadata_dir=root / 'data' / 'processed' / 'edp',
        nem_prices_dir=root / 'data' / 'clean' / 'nem_prices',
        tariffs_dir=PRODUCTION_TARIFFS_DIR,
    )


def test_validate_scenario_vpp_no_vpp_config_raises(tmp_path):
    """validate_scenario raises if retail tariff has no VppConfig block."""
    build_full_fixture(tmp_path, year=2022)
    # powershop_tou_2024 is a plain TOU plan with no VppConfig
    scenario = ScenarioConfig(
        state='NSW', year=2022, cohort='solar_battery',
        coordination_type='vpp_sc_event',
        network_tariff_id='ausgrid_ea025_network_2024',
        retail_tariff_id='powershop_tou_2024',
        op_retail_tariff_id='agl_operator_wholesale_2025',
        battery_spec=_bat(),
    )
    with pytest.raises(ScenarioValidationError, match='no VppConfig block'):
        validate_scenario(scenario, _paths_for(tmp_path))


def test_validate_scenario_vpp_op_tariff_wrong_type_raises(tmp_path):
    """validate_scenario raises if op tariff is not wholesale_exposed."""
    build_full_fixture(tmp_path, year=2022)
    # powershop_tou_2024 is TOU, not wholesale_exposed -- wrong for operator tariff
    scenario = ScenarioConfig(
        state='NSW', year=2022, cohort='solar_battery',
        coordination_type='vpp_sc_event',
        network_tariff_id='ausgrid_ea025_network_2024',
        retail_tariff_id='agl_battery_rewards_vpp_2025',
        op_retail_tariff_id='powershop_tou_2024',
        battery_spec=_bat(),
    )
    with pytest.raises(ScenarioValidationError, match='wholesale_exposed'):
        validate_scenario(scenario, _paths_for(tmp_path))


def test_validate_scenario_vpp_passes_with_correct_config(tmp_path):
    """validate_scenario passes when retail has VppConfig and op is wholesale_exposed."""
    build_full_fixture(tmp_path, year=2022)
    scenario = ScenarioConfig(
        state='NSW', year=2022, cohort='solar_battery',
        coordination_type='vpp_sc_event',
        network_tariff_id='ausgrid_ea025_network_2024',
        retail_tariff_id='agl_battery_rewards_vpp_2025',
        op_retail_tariff_id='agl_operator_wholesale_2025',
        battery_spec=_bat(),
    )
    # Should not raise
    validate_scenario(scenario, _paths_for(tmp_path))


# ---------------------------------------------------------------------------
# ScenarioResult.vpp_traces field
# ---------------------------------------------------------------------------

def _make_site_traces(site_id: str, n: int = 4) -> SiteTraces:
    idx = pd.date_range('2022-01-01 00:05', periods=n, freq='5min',
                        tz='Etc/GMT-10')
    z = pd.Series(np.zeros(n), index=idx)
    return SiteTraces(
        site_id=site_id,
        import_kwh=z, export_kwh=z, battery_soc_kwh=z,
        sol_to_home_kwh=z, sol_to_bat_kwh=z, sol_to_grid_kwh=z,
        bat_to_home_kwh=z, grid_to_home_kwh=z, grid_to_bat_kwh=z,
        bat_to_grid_kwh=z, curtail_kwh=z,
    )


def _make_site_bill(site_id: str) -> SiteBill:
    return SiteBill(
        site_id=site_id, plan_type='tou',
        wholesale_usage_cost=10.0, wholesale_feedin_credit=-2.0,
        network_usage_cost=5.0, network_feedin_credit=-1.0,
        network_daily_charges=8.0, retail_usage_cost=30.0,
        retail_feedin_credit=-5.0, retail_daily_charges=12.0,
        household_total=47.0, retailer_margin=25.0,
    )


def _make_vpp_traces(site_id: str, n: int = 4) -> VppTraces:
    idx = pd.date_range('2022-01-01 00:05', periods=n, freq='5min',
                        tz='Etc/GMT-10')
    z = pd.Series(np.zeros(n), index=idx)
    return VppTraces(
        site_id=site_id,
        n_imp_kwh=z, f_imp_kwh=z, n_exp_kwh=z, f_exp_kwh=z,
    )


def test_scenario_result_vpp_traces_defaults_to_none():
    """ScenarioResult.vpp_traces and vpp_event are None by default (non-VPP)."""
    result = ScenarioResult(
        scenario=MagicMock(),
        site_traces={'S0001': _make_site_traces('S0001')},
        site_bills={'S0001': _make_site_bill('S0001')},
    )
    assert result.vpp_traces is None
    assert result.vpp_event is None


def test_scenario_result_vpp_traces_populated():
    """ScenarioResult.vpp_traces holds per-site VppTraces when supplied."""
    vt = _make_vpp_traces('S0001')
    result = ScenarioResult(
        scenario=MagicMock(),
        site_traces={'S0001': _make_site_traces('S0001')},
        site_bills={'S0001': _make_site_bill('S0001')},
        vpp_traces={'S0001': vt},
    )
    assert result.vpp_traces is not None
    assert 'S0001' in result.vpp_traces
    assert isinstance(result.vpp_traces['S0001'], VppTraces)


def test_scenario_result_vpp_traces_site_ids_match_traces():
    """vpp_traces site IDs match site_traces site IDs."""
    sites = ['S0001', 'S0002']
    result = ScenarioResult(
        scenario=MagicMock(),
        site_traces={s: _make_site_traces(s) for s in sites},
        site_bills={s: _make_site_bill(s) for s in sites},
        vpp_traces={s: _make_vpp_traces(s) for s in sites},
    )
    assert set(result.vpp_traces.keys()) == set(result.site_traces.keys())


def test_scenario_result_vpp_traces_four_series():
    """Each VppTraces in vpp_traces has the expected four series."""
    vt = _make_vpp_traces('S0001')
    result = ScenarioResult(
        scenario=MagicMock(),
        site_traces={'S0001': _make_site_traces('S0001')},
        site_bills={'S0001': _make_site_bill('S0001')},
        vpp_traces={'S0001': vt},
    )
    vt_out = result.vpp_traces['S0001']
    for series_name in ('n_imp_kwh', 'f_imp_kwh', 'n_exp_kwh', 'f_exp_kwh'):
        assert hasattr(vt_out, series_name)
        assert isinstance(getattr(vt_out, series_name), pd.Series)


# ---------------------------------------------------------------------------
# _run_vpp_sc_event wiring via a minimal synthetic in-process run
#
# Rather than building the full fixture tree and relying on file I/O, we
# patch ingest() and load_retail_tariff() and run the VPP dispatch in-process
# against synthetic 1-day traces. This validates the orchestrator wiring
# (price-vector construction, event classification, per-site worker dispatch
# via _vpp_sc_event_site_worker, billing with vpp_traces, ScenarioResult
# population) without requiring production data files.
#
# n_workers=1 is used throughout to keep the worker call in-process
# (sequential path in _run_vpp_sc_event) rather than exercising
# ProcessPoolExecutor, which would require the MagicMock-wrapped ctx
# fixtures below to be picklable.
# ---------------------------------------------------------------------------

def _build_synthetic_ctx(n_intervals: int = 288) -> MagicMock:
    """Build a synthetic DataContext with a single NSW site, 1-day traces."""
    idx = pd.date_range('2022-01-01 00:05', periods=n_intervals, freq='5min',
                        tz='Etc/GMT-10')

    # Simple solar/load pattern: solar peaks midday, load peaks morning/evening
    t = np.arange(n_intervals) / n_intervals  # 0..1 over the period
    solar = np.maximum(0, 2.0 * np.sin(np.pi * t) * (5 / 60))  # kWh/interval
    load  = np.full(n_intervals, 0.5 * (5 / 60))                # flat 0.5 kW

    site_trace = pd.DataFrame({
        'pv_site_net_kwh': solar,
        'gross_load_kwh':  load,
    }, index=idx)

    # Prices: mostly flat at 50 $/MWh; two spikes above 1000 $/MWh
    prices_arr = np.full(n_intervals, 50.0)
    prices_arr[200] = 1500.0
    prices_arr[250] = 1200.0
    prices = pd.Series(prices_arr, index=idx, name='NSW1_rrp')

    metadata = pd.DataFrame({
        'edp_site_id': ['S0001'],
        'state': ['NSW'],
    })

    ctx = MagicMock()
    ctx.site_traces = {'S0001': site_trace}
    ctx.prices = prices
    ctx.site_metadata = metadata
    ctx.network_tariff = _network()
    ctx.retail_tariff  = _retail_vpp()
    return ctx


@patch('model.orchestrate.validate_scenario')
@patch('model.orchestrate.load_retail_tariff')
@patch('model.orchestrate.ingest')
def test_run_vpp_populates_vpp_traces(mock_ingest, mock_load_retail, mock_validate, tmp_path):
    """_run_vpp_sc_event returns a ScenarioResult with vpp_traces."""

    ctx = _build_synthetic_ctx()
    mock_ingest.return_value = ctx
    mock_load_retail.return_value = _retail_op()

    paths = Paths(
        edp_clean_dir=tmp_path, edp_metadata_dir=tmp_path,
        nem_prices_dir=tmp_path, tariffs_dir=tmp_path,
    )
    scenario = _vpp_scenario()

    result = run_scenario(scenario, paths, n_workers=1)

    assert isinstance(result, ScenarioResult)
    assert result.vpp_traces is not None, 'vpp_traces should be populated'
    assert 'S0001' in result.vpp_traces
    assert isinstance(result.vpp_traces['S0001'], VppTraces)
    assert result.vpp_event is not None, 'vpp_event should be populated'
    assert isinstance(result.vpp_event, pd.Series)
    assert result.vpp_event.dtype == bool
    assert len(result.vpp_event) == 288


@pytest.mark.parametrize('coordination_type', ['vpp_sc_event', 'vpp_tou_event'])
@patch('model.orchestrate.validate_scenario')
@patch('model.orchestrate.load_retail_tariff')
@patch('model.orchestrate.ingest')
def test_run_vpp_populates_billing_context(
    mock_ingest, mock_load_retail, mock_validate, coordination_type, tmp_path,
):
    """Regression guard: with_monthly_bills=True must populate
    result.site_monthly_bills, reconciling with the annual bill, for both
    vpp_sc_event and vpp_tou_event -- both routed through
    _run_vpp_sc_event's picklable worker (gap_strategy differs), sharing one
    ScenarioResult(...) construction site and one args-tuple shape the
    with_monthly_bills flag has to survive intact across the
    ProcessPoolExecutor pickle boundary. This is the counterpart to
    test_orchestrate.py's test_run_scenario_populates_billing_context_for_lp_types,
    which covers the non-VPP coordination types; together the two sweep every
    coordination_type in the model."""
    ctx = _build_synthetic_ctx()
    mock_ingest.return_value = ctx
    mock_load_retail.return_value = _retail_op()

    paths = Paths(
        edp_clean_dir=tmp_path, edp_metadata_dir=tmp_path,
        nem_prices_dir=tmp_path, tariffs_dir=tmp_path,
    )
    scenario = _vpp_scenario(coordination_type)

    # with_monthly_bills=False (default): explicitly stays None.
    result_off = run_scenario(scenario, paths, n_workers=1)
    assert result_off.site_monthly_bills is None, (
        f"{coordination_type}: site_monthly_bills should be None when "
        f"with_monthly_bills=False"
    )

    result = run_scenario(scenario, paths, n_workers=1, with_monthly_bills=True)

    assert isinstance(result, ScenarioResult)
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
    # Bounded, not exact -- see the identical comment in
    # test_orchestrate.py's test_run_scenario_populates_billing_context_for_lp_types:
    # _build_synthetic_ctx is a 1-day AEST index for NSW (DST-observing),
    # so the annual bill's AEST-based day count and the monthly bucket's
    # local-based day count can legitimately differ by up to a day's daily
    # charges for a fixture this short, purely from AEST/local divergence
    # under DST. This test's purpose is the with_monthly_bills wiring, not
    # billing precision under DST -- that's covered precisely, with a full
    # real calendar year, by test_billing_monthly.py's dedicated tests.
    assert diff < 5.0, (
        f"{coordination_type}: monthly household_total sum ({total}) "
        f"differs from the annual bill ({annual_total}) by {diff:.2f}, "
        f"more than the expected sub-one-day fixture-DST divergence should "
        f"allow"
    )


@patch('model.orchestrate.validate_scenario')
@patch('model.orchestrate.load_retail_tariff')
@patch('model.orchestrate.ingest')
def test_run_vpp_has_baseline_traces(mock_ingest, mock_load_retail, mock_validate, tmp_path):
    """_run_vpp_sc_event populates baseline_traces for SC vs VPP delta."""

    ctx = _build_synthetic_ctx()
    mock_ingest.return_value = ctx
    mock_load_retail.return_value = _retail_op()

    paths = Paths(
        edp_clean_dir=tmp_path, edp_metadata_dir=tmp_path,
        nem_prices_dir=tmp_path, tariffs_dir=tmp_path,
    )

    result = run_scenario(_vpp_scenario(), paths, n_workers=1)

    assert result.baseline_traces is not None, 'baseline_traces should be populated'
    assert result.baseline_bills  is not None, 'baseline_bills should be populated'
    assert 'S0001' in result.baseline_traces
    assert 'S0001' in result.baseline_bills
    assert result.vpp_event is not None, 'vpp_event should be populated for VPP runs'


@patch('model.orchestrate.validate_scenario')
@patch('model.orchestrate.load_retail_tariff')
@patch('model.orchestrate.ingest')
def test_run_vpp_vpp_traces_invariants(mock_ingest, mock_load_retail, mock_validate, tmp_path):
    """VppTraces invariants hold in the orchestrated result.

    n_imp + f_imp == import_kwh and n_exp + f_exp == export_kwh for the VPP
    dispatch returned by the orchestrator (validates the wiring end-to-end,
    not just the vpp_sc_event component in isolation).
    """

    ctx = _build_synthetic_ctx()
    mock_ingest.return_value = ctx
    mock_load_retail.return_value = _retail_op()

    paths = Paths(
        edp_clean_dir=tmp_path, edp_metadata_dir=tmp_path,
        nem_prices_dir=tmp_path, tariffs_dir=tmp_path,
    )

    result = run_scenario(_vpp_scenario(), paths, n_workers=1)

    for site_id in result.site_traces:
        traces = result.site_traces[site_id]
        vt     = result.vpp_traces[site_id]
        np.testing.assert_allclose(
            (vt.n_imp_kwh + vt.f_imp_kwh).to_numpy(),
            traces.import_kwh.to_numpy(), atol=1e-6,
            err_msg=f'{site_id}: n_imp + f_imp != import_kwh',
        )
        np.testing.assert_allclose(
            (vt.n_exp_kwh + vt.f_exp_kwh).to_numpy(),
            traces.export_kwh.to_numpy(), atol=1e-6,
            err_msg=f'{site_id}: n_exp + f_exp != export_kwh',
        )

    # vpp_event index must align with site traces index
    assert result.vpp_event is not None
    assert result.vpp_event.index.equals(
        next(iter(result.site_traces.values())).import_kwh.index
    ), 'vpp_event index must match site traces index'


@patch('model.orchestrate.validate_scenario')
@patch('model.orchestrate.load_retail_tariff')
@patch('model.orchestrate.ingest')
def test_run_vpp_bill_has_retail_other_credits(mock_ingest, mock_load_retail, mock_validate, tmp_path):
    """VPP run bills include retail_other_credits from the VPP reward.

    The synthetic price trace has two spikes above the 1000 $/MWh threshold;
    the event classifier marks windows around them. If the LP discharges during
    those events, f_exp > 0 and retail_other_credits < 0 (credit).
    """

    ctx = _build_synthetic_ctx()
    mock_ingest.return_value = ctx
    mock_load_retail.return_value = _retail_op()

    paths = Paths(
        edp_clean_dir=tmp_path, edp_metadata_dir=tmp_path,
        nem_prices_dir=tmp_path, tariffs_dir=tmp_path,
    )

    result = run_scenario(_vpp_scenario(), paths, n_workers=1)

    for site_id, bill in result.site_bills.items():
        # The bill's retail_other_credits is zero or negative (a credit).
        # It may be zero if no forced export was profitable (battery SOC
        # too low or event too short), but it must never be positive.
        assert bill.retail_other_credits <= 1e-9, (
            f'{site_id}: retail_other_credits should be 0 or negative (credit), '
            f'got {bill.retail_other_credits:.4f}'
        )

    # vpp_event should mark the two spike intervals (and their windows)
    assert result.vpp_event is not None
    assert result.vpp_event.any(), 'at least some intervals should be event intervals'


@patch('model.orchestrate.validate_scenario')
@patch('model.orchestrate.load_retail_tariff')
@patch('model.orchestrate.ingest')
def test_run_vpp_event_series_shape_and_content(
    mock_ingest, mock_load_retail, mock_validate, tmp_path
):
    """vpp_event is a boolean Series with the correct index and at least one True.

    The synthetic ctx has two price spikes above the 1000 $/MWh threshold at
    intervals 200 and 250. With a 10-hour event window (120 intervals), the
    classifier marks windows around both spikes so vpp_event.any() must be True.
    """
    ctx = _build_synthetic_ctx()
    mock_ingest.return_value = ctx
    mock_load_retail.return_value = _retail_op()

    paths = Paths(
        edp_clean_dir=tmp_path, edp_metadata_dir=tmp_path,
        nem_prices_dir=tmp_path, tariffs_dir=tmp_path,
    )

    result = run_scenario(_vpp_scenario(), paths, n_workers=1)

    assert result.vpp_event is not None
    assert isinstance(result.vpp_event, pd.Series)
    assert result.vpp_event.dtype == bool
    # Index must match the price series index exactly
    assert result.vpp_event.index.equals(ctx.prices.index)
    # Both spikes at 200 and 250 are above the 1000 $/MWh threshold so at
    # least those intervals must be marked True
    assert result.vpp_event.iloc[200], 'interval 200 (spike at 1500) must be event'
    assert result.vpp_event.iloc[250], 'interval 250 (spike at 1200) must be event'


@patch('model.orchestrate.validate_scenario')
@patch('model.orchestrate.load_retail_tariff')
@patch('model.orchestrate.ingest')
def test_run_vpp_non_vpp_coordination_does_not_set_vpp_traces(
    mock_ingest, mock_load_retail, mock_validate, tmp_path
):
    """A self_consumption run does not populate vpp_traces."""

    # Use plain retail tariff (no VPP config) for a self_consumption scenario
    ctx = _build_synthetic_ctx()
    ctx.retail_tariff = _retail_plain()
    mock_ingest.return_value = ctx
    mock_load_retail.return_value = _retail_plain()

    paths = Paths(
        edp_clean_dir=tmp_path, edp_metadata_dir=tmp_path,
        nem_prices_dir=tmp_path, tariffs_dir=tmp_path,
    )

    sc_scenario = ScenarioConfig(
        state='NSW', year=2022, cohort='solar_battery',
        coordination_type='self_consumption',
        network_tariff_id='net', retail_tariff_id='ret',
        battery_spec=_bat(),
    )

    result = run_scenario(sc_scenario, paths, n_workers=1)

    assert result.vpp_traces is None, (
        'self_consumption result should not have vpp_traces'
    )
    assert result.vpp_event is None, (
        'self_consumption result should not have vpp_event'
    )

# ---------------------------------------------------------------------------
# validate_scenario checks for the rolling VPP coordination types
# ---------------------------------------------------------------------------
# These use FIXTURE tariffs rather than production ones, unlike the
# perfect-foresight VPP checks above. The rules under test turn on whether a
# tariff sets min_export_price_dollars_per_mwh, and pinning that to a
# production plan would make the test fail whenever the plan is edited for
# unrelated reasons. Two fixtures differing only in that key isolate it:
#   test_vpp_no_min_export  -- omits both the value and the mode
#   test_vpp_rolling        -- sets the value, mode defaults to 'fixed'
#   test_vpp_this_year      -- omits the value, mode='based_on_this_year'
#   test_vpp_last_year      -- sets the value, mode='based_on_last_year'
# See rolling_vpp_event_spec.md, "Minimum export price".

FIXTURE_TARIFFS_DIR = Path(__file__).parent / 'fixtures' / 'tariffs'


def _fixture_paths(root: Path) -> Paths:
    """Paths with tariffs served from the fixture directory."""
    return Paths(
        edp_clean_dir=root / 'data' / 'clean' / 'edp',
        edp_metadata_dir=root / 'data' / 'processed' / 'edp',
        nem_prices_dir=root / 'data' / 'clean' / 'nem_prices',
        tariffs_dir=FIXTURE_TARIFFS_DIR,
    )


def _rolling_vpp_scenario(
    coordination_type: str = 'rolling_vpp_sc_event',
    retail_tariff_id: str = 'test_vpp_rolling',
    horizon_hours: float = 18.0,
) -> ScenarioConfig:
    from model.core.scenario import RollingConfig
    return ScenarioConfig(
        state='NSW', year=2022, cohort='solar_battery',
        coordination_type=coordination_type,
        network_tariff_id='test_network',
        retail_tariff_id=retail_tariff_id,
        op_retail_tariff_id='test_operator_wholesale',
        battery_spec=_bat(),
        rolling=RollingConfig(forecast_source='perfect',
                              horizon_hours=horizon_hours),
    )


@pytest.mark.parametrize('ct', ['rolling_vpp_sc_event',
                                'rolling_vpp_tou_event'])
def test_validate_rolling_vpp_passes_with_correct_config(tmp_path, ct):
    build_full_fixture(tmp_path, year=2022)
    validate_scenario(_rolling_vpp_scenario(coordination_type=ct),
                      _fixture_paths(tmp_path))


def _pf_vpp_fixture_scenario(coordination_type, retail_tariff_id):
    """A perfect-foresight VPP scenario. No RollingConfig, so it exercises the
    shared min export price rules without the rolling-only checks."""
    return ScenarioConfig(
        state='NSW', year=2022, cohort='solar_battery',
        coordination_type=coordination_type,
        network_tariff_id='test_network',
        retail_tariff_id=retail_tariff_id,
        op_retail_tariff_id='test_operator_wholesale',
        battery_spec=_bat(),
    )


@pytest.mark.parametrize('ct', ['vpp_sc_event', 'vpp_tou_event',
                                'rolling_vpp_sc_event',
                                'rolling_vpp_tou_event'])
def test_validate_vpp_requires_min_export_price_under_fixed(tmp_path, ct):
    """All four VPP types, because they share resolve_min_export_price and so
    must reject the same configurations. The fixture states neither the mode
    nor the value, so the mode defaults to 'fixed' with nothing to use.

    This combination once meant "derive from this year's prices" by
    implication and was valid on the perfect-foresight types. It now fails,
    which is the point: the implicit convention is gone rather than silently
    resolving differently than it used to."""
    build_full_fixture(tmp_path, year=2022)
    scenario = (_rolling_vpp_scenario(retail_tariff_id='test_vpp_no_min_export')
                if ct.startswith('rolling_')
                else _pf_vpp_fixture_scenario(ct, 'test_vpp_no_min_export'))
    scenario = dataclasses.replace(scenario, coordination_type=ct)
    with pytest.raises(ScenarioValidationError,
                       match='min_export_price_dollars_per_mwh'):
        validate_scenario(scenario, _fixture_paths(tmp_path))


@pytest.mark.parametrize('ct', ['vpp_sc_event', 'vpp_tou_event'])
def test_validate_perfect_foresight_vpp_accepts_a_stated_value(tmp_path, ct):
    """Non-vacuity for the test above: the same fixture set must still pass on
    these types when a value IS stated, or that test would be passing because
    the perfect-foresight path rejects everything."""
    build_full_fixture(tmp_path, year=2022)
    validate_scenario(_pf_vpp_fixture_scenario(ct, 'test_vpp_rolling'),
                      _fixture_paths(tmp_path))


@pytest.mark.parametrize('ct', ['vpp_sc_event', 'vpp_tou_event'])
def test_validate_perfect_foresight_vpp_accepts_this_year_without_a_value(
        tmp_path, ct):
    """The migration path for a tariff that used to omit the value and rely on
    the implicit derivation: state the mode instead."""
    build_full_fixture(tmp_path, year=2022)
    validate_scenario(_pf_vpp_fixture_scenario(ct, 'test_vpp_this_year'),
                      _fixture_paths(tmp_path))


@pytest.mark.parametrize('ct', ['vpp_sc_event', 'vpp_tou_event'])
def test_validate_perfect_foresight_vpp_rejects_based_on_last_year(
        tmp_path, ct):
    """Previously rejected only on the rolling types, so a perfect-foresight
    scenario naming it reached the resolver's backstop mid-run instead of
    failing before ingest."""
    build_full_fixture(tmp_path, year=2022)
    with pytest.raises(ScenarioValidationError, match='not implemented'):
        validate_scenario(_pf_vpp_fixture_scenario(ct, 'test_vpp_last_year'),
                          _fixture_paths(tmp_path))


def test_validate_rolling_vpp_requires_vpp_block(tmp_path):
    build_full_fixture(tmp_path, year=2022)
    scenario = _rolling_vpp_scenario(retail_tariff_id='test_tou')
    with pytest.raises(ScenarioValidationError, match='vpp block'):
        validate_scenario(scenario, _fixture_paths(tmp_path))


def test_validate_rolling_vpp_rejects_horizon_shorter_than_event_window(tmp_path):
    """The declaration lookahead is min(W, n). A horizon below one event
    window truncates every lookahead, so the run would declare fewer events
    for a configuration reason and the result would read as a forecast
    effect. The fixture's event_window_hours is 10.0."""
    build_full_fixture(tmp_path, year=2022)
    scenario = _rolling_vpp_scenario(horizon_hours=6.0)
    with pytest.raises(ScenarioValidationError, match='shorter than the'):
        validate_scenario(scenario, _fixture_paths(tmp_path))


@pytest.mark.parametrize('horizon_hours', [10.0, 9.99])
def test_validate_rolling_vpp_accepts_horizon_at_the_boundary(
        tmp_path, horizon_hours):
    """Equal is sufficient, and the comparison is in INTERVALS: 9.99h and
    10.0h are both 120 five-minute intervals after the rounding
    rolling_optimisation applies, so 9.99h must be accepted. An hours-only
    comparison would reject it and disagree with the loop."""
    build_full_fixture(tmp_path, year=2022)
    validate_scenario(_rolling_vpp_scenario(horizon_hours=horizon_hours),
                      _fixture_paths(tmp_path))


def test_validate_rolling_vpp_rejects_horizon_just_below_the_boundary(tmp_path):
    """9.9h rounds to 119 intervals against the event window's 120, so it is
    genuinely short. Paired with the test above, this pins the rounding
    behaviour rather than just the direction of the comparison."""
    build_full_fixture(tmp_path, year=2022)
    with pytest.raises(ScenarioValidationError, match='shorter than the'):
        validate_scenario(_rolling_vpp_scenario(horizon_hours=9.9),
                          _fixture_paths(tmp_path))


def test_validate_rolling_vpp_tou_rejects_a_wholesale_household_plan(tmp_path):
    """rolling_vpp_tou_event optimises the household against a TOU plan
    between events; a wholesale-exposed household plan belongs on
    rolling_vpp_sc_event."""
    build_full_fixture(tmp_path, year=2022)
    scenario = _rolling_vpp_scenario(
        coordination_type='rolling_vpp_tou_event',
        retail_tariff_id='test_vpp_wholesale')
    with pytest.raises(ScenarioValidationError, match='is type'):
        validate_scenario(scenario, _fixture_paths(tmp_path))


def test_validate_rolling_vpp_sc_accepts_a_wholesale_household_plan(tmp_path):
    """Non-vacuity guard for the test above. The same tariff must be valid on
    rolling_vpp_sc_event, or that test would be passing because the fixture is
    rejected for some unrelated reason rather than because the TOU rule fires.
    A household that self-consumes between events places no constraint on the
    household plan type."""
    build_full_fixture(tmp_path, year=2022)
    scenario = _rolling_vpp_scenario(
        coordination_type='rolling_vpp_sc_event',
        retail_tariff_id='test_vpp_wholesale')
    validate_scenario(scenario, _fixture_paths(tmp_path))


# ---------------------------------------------------------------------------
# Rolling VPP: end-to-end through run_scenario
# ---------------------------------------------------------------------------
# The wiring tests compare the args tuple across its definitions; these check
# that a scenario actually executes and returns the shapes downstream code
# reads. Both run serially (workers=1), which still exercises
# _init_rolling_worker via the same path the pool uses.

@pytest.mark.slow
@pytest.mark.parametrize('ct', ['rolling_vpp_sc_event',
                                'rolling_vpp_tou_event'])
def test_rolling_vpp_scenario_runs_end_to_end(tmp_path, ct):
    build_full_fixture(tmp_path, year=2022)
    from model.orchestrate import run_scenario
    scenario = _rolling_vpp_scenario(coordination_type=ct)
    result = run_scenario(scenario, _fixture_paths(tmp_path), n_workers=1)

    assert result.site_traces, "no sites returned"
    for sid, traces in result.site_traces.items():
        assert sid in result.site_bills
        assert sid in result.baseline_bills, "self-consumption baseline missing"
        assert sid in result.rolling_diagnostics
        # VppTraces must reach the result: billing needs it for the reward,
        # and nothing downstream can reconstruct the split from the headline
        # series alone.
        assert result.vpp_traces is not None
        vt = result.vpp_traces[sid]
        np.testing.assert_allclose(
            (vt.n_imp_kwh + vt.f_imp_kwh).to_numpy(),
            traces.import_kwh.to_numpy(), atol=1e-6)
        np.testing.assert_allclose(
            (vt.n_exp_kwh + vt.f_exp_kwh).to_numpy(),
            traces.export_kwh.to_numpy(), atol=1e-6)


@pytest.mark.slow
def test_rolling_vpp_diagnostics_reach_the_scenario_result(tmp_path):
    """The VPP counters are what a bulk run is read through, so they must
    survive the worker and the result assembly, not just exist on the
    planner."""
    build_full_fixture(tmp_path, year=2022)
    from model.orchestrate import run_scenario
    result = run_scenario(_rolling_vpp_scenario(), _fixture_paths(tmp_path),
                          n_workers=1)
    diag = next(iter(result.rolling_diagnostics.values()))
    for field in ('events_anchored', 'event_intervals', 'cap_used_kwh',
                  'cap_remaining_kwh', 'n_imp_budget_clamps',
                  'f_exp_total_kwh', 'n_lt_W'):
        assert getattr(diag, field) is not None, (
            f"{field} is None on a rolling VPP run; it should be populated"
        )
    # And the loop's own counters are not lost by the VPP path.
    assert diag.n_intervals > 0
    assert diag.causality_ok is True


@pytest.mark.slow
def test_the_two_rolling_vpp_types_produce_different_dispatch(tmp_path):
    """SLOW -- runs two full-year scenarios back to back, several minutes.

    Non-vacuity for the pair above: if gap_strategy were ignored, both
    coordination types would return identical traces and every test that runs
    only one of them would still pass."""
    build_full_fixture(tmp_path, year=2022)
    from model.orchestrate import run_scenario
    p = _fixture_paths(tmp_path)
    sc = run_scenario(_rolling_vpp_scenario(
        coordination_type='rolling_vpp_sc_event'), p, n_workers=1)
    tou = run_scenario(_rolling_vpp_scenario(
        coordination_type='rolling_vpp_tou_event'), p, n_workers=1)
    sid = next(iter(sc.site_traces))
    diff = float(np.abs(
        sc.site_traces[sid].battery_soc_kwh.to_numpy()
        - tou.site_traces[sid].battery_soc_kwh.to_numpy()).max())
    assert diff > 1e-6, (
        "the SC and TOU variants produced identical dispatch; gap_strategy "
        "is probably not reaching the component"
    )


# ---------------------------------------------------------------------------
# resolve_min_export_price
# ---------------------------------------------------------------------------
# Unit tests on plain Series and a traces dict, no scenario fixture: the
# resolver is separated from the runner precisely so this is possible.
#
# The floor rule is max(derived, configured or 0). The configured value is the
# operator's own price floor and the derivation can only tighten it, which
# matters most where the derivation returns zero because the available
# throughput could not exhaust kwh_limit.
# See rolling_vpp_event_spec.md, "Minimum export price".

_MEP_INTERVAL_H = 5 / 60


def _mep_cfg(mode='based_on_this_year', configured=None, kwh_limit=10.0,
             event_window_hours=0.5, threshold=1000.0):
    return VppConfig(
        exp_reward_c_per_kwh=100.0,
        imp_reward_c_per_kwh=100.0,
        kwh_limit=kwh_limit,
        limit_applies_to='export_only',
        default_mode='self_consumption',
        event_window_hours=event_window_hours,
        vpp_threshold_dollars_per_mwh=threshold,
        min_export_price_dollars_per_mwh=configured,
        min_export_price_mode=mode,
    )


def _mep_inputs(T=288, spike_from=100, n_spikes=12, base=50.0,
                spike_high=5000.0, spike_low=1200.0, load_kwh=0.05):
    """A period with one spike cluster, and a one-site cohort.

    The spike prices descend so the marginal price lands strictly inside the
    cluster rather than at an endpoint, which is what makes the derived value
    sensitive to the accumulation being right rather than to the mask alone.
    """
    idx = pd.date_range('2022-06-01', periods=T, freq='5min', tz='Etc/GMT-10')
    prices = pd.Series(np.full(T, base), index=idx)
    if n_spikes:
        prices.iloc[spike_from:spike_from + n_spikes] = np.linspace(
            spike_high, spike_low, n_spikes)
    traces = {'S1': {'gross_load_kwh': pd.Series(np.full(T, load_kwh),
                                                 index=idx)}}
    return prices, traces


def _resolve(cfg, prices=None, traces=None, power_kw=5.0):
    if prices is None:
        prices, traces = _mep_inputs()
    return resolve_min_export_price(
        vpp_cfg=cfg, prices_mwh=prices, site_traces=traces,
        site_ids=['S1'], power_kw=power_kw,
        interval_hours=_MEP_INTERVAL_H,
    )


def test_resolve_fixed_uses_the_configured_value():
    r = _resolve(_mep_cfg(mode='fixed', configured=500.0))
    assert r.mode == 'fixed'
    assert r.value_dollars_per_mwh == pytest.approx(500.0)
    assert r.source == 'configured'
    assert r.derived_dollars_per_mwh is None, (
        "'fixed' must not run the estimator; a derived value here means the "
        "mode branch is not being taken"
    )


def test_resolve_fixed_without_a_value_raises():
    """validate_scenario catches this before ingest; the resolver is the
    backstop for a caller that bypassed it."""
    with pytest.raises(ValueError, match='fixed'):
        _resolve(_mep_cfg(mode='fixed', configured=None))


def test_resolve_rejects_based_on_last_year():
    with pytest.raises(ValueError, match='not implemented'):
        _resolve(_mep_cfg(mode='based_on_last_year', configured=500.0))


def test_resolve_this_year_derives_a_threshold():
    """Non-vacuity first: the fixture must actually declare events and the
    accumulation must actually reach kwh_limit, or every assertion below is
    about the empty case rather than the derivation."""
    r = _resolve(_mep_cfg(configured=None, kwh_limit=2.0))
    assert r.n_mask_intervals > 0, "fixture declared no events"
    assert r.budget_exhausted is True, "fixture never reached kwh_limit"
    assert r.derived_dollars_per_mwh > 0.0
    assert r.value_dollars_per_mwh == pytest.approx(
        r.derived_dollars_per_mwh)
    assert r.source == 'derived'


def test_resolve_this_year_uses_the_configured_value_when_it_is_higher():
    """The case this mode is most often run with: an operator floor that the
    derivation does not clear. Paired with the test above so the pair pins the
    max rather than either branch alone."""
    low = _resolve(_mep_cfg(configured=None, kwh_limit=2.0))
    floor = low.derived_dollars_per_mwh + 1000.0

    r = _resolve(_mep_cfg(configured=floor, kwh_limit=2.0))
    assert r.derived_dollars_per_mwh == pytest.approx(
        low.derived_dollars_per_mwh), (
        "the estimate must not depend on the configured value"
    )
    assert r.value_dollars_per_mwh == pytest.approx(floor)
    assert r.source == 'configured'


def test_resolve_this_year_ignores_a_lower_configured_value():
    low = _resolve(_mep_cfg(configured=None, kwh_limit=2.0))
    r = _resolve(_mep_cfg(configured=1.0, kwh_limit=2.0))
    assert r.value_dollars_per_mwh == pytest.approx(
        low.derived_dollars_per_mwh)
    assert r.source == 'derived'
    assert r.configured_dollars_per_mwh == pytest.approx(1.0), (
        "the configured value must still be recorded even when not selected"
    )


def test_resolve_this_year_floor_governs_when_the_cap_cannot_be_spent():
    """A kwh_limit far beyond the available throughput. The estimator returns
    zero, and without the floor the restriction would vanish at exactly the
    configurations where event intervals are scarcest."""
    r = _resolve(_mep_cfg(configured=300.0, kwh_limit=1e6))
    assert r.budget_exhausted is False
    assert r.derived_dollars_per_mwh == pytest.approx(0.0)
    assert r.value_dollars_per_mwh == pytest.approx(300.0)
    assert r.source == 'configured'
    assert r.available_kwh > 0.0, (
        "throughput was zero, so this is the empty-mask case rather than the "
        "cap-unreachable case the test is about"
    )


def test_resolve_this_year_with_no_events_reports_an_empty_mask():
    """A period with no spike declares nothing, so there is no forced-export
    opportunity to conserve. The mask size must read 0 rather than the period
    length: _compute_min_export_price treats an all-False mask as no mask at
    all and falls through to all-year prices, which is the overstatement the
    mask exists to prevent."""
    prices, traces = _mep_inputs(n_spikes=0)
    r = _resolve(_mep_cfg(configured=300.0), prices, traces)
    assert r.n_mask_intervals == 0
    assert r.available_kwh == pytest.approx(0.0)
    assert r.budget_exhausted is False
    assert r.value_dollars_per_mwh == pytest.approx(300.0)


def test_resolve_returns_dollars_per_mwh_not_per_kwh():
    """The unit boundary. VppWindowPlanner takes $/MWh and applies its own
    /1000, while the perfect-foresight runners divide by 1000 because
    vpp_sc_event takes $/kWh. A stray conversion here would put the threshold
    three orders of magnitude too low, the f_exp pin would never fire, and the
    only symptom would be f_exp_blocked_by_price = 0 -- which is also what a
    legitimately zero threshold produces."""
    prices, traces = _mep_inputs()
    r = _resolve(_mep_cfg(configured=None, kwh_limit=2.0), prices, traces)
    within_the_spike = float(prices.max()) >= r.value_dollars_per_mwh >= float(
        prices.iloc[100:112].min())
    assert within_the_spike, (
        f"derived {r.value_dollars_per_mwh} $/MWh is not among the spike "
        f"prices it was selected from ({prices.iloc[100:112].min()} to "
        f"{prices.max()}); the value is probably in the wrong unit"
    )


def test_resolve_this_year_is_insensitive_to_battery_power_ordering():
    """A larger battery clears more throughput per interval, so it reaches
    kwh_limit sooner in the price-sorted list and the threshold can only rise
    or hold. Pins the direction of the power dependence, which is the part of
    the algorithm most easily inverted.

    kwh_limit=1.0 rather than 2.0: at power_kw=2.0 the 12-interval fixture's
    total available throughput is 1.95 kWh, so kwh_limit=2.0 never exhausts at
    the smaller power and the test would be asserting nothing about it."""
    small = _resolve(_mep_cfg(configured=None, kwh_limit=1.0), power_kw=2.0)
    large = _resolve(_mep_cfg(configured=None, kwh_limit=1.0), power_kw=8.0)
    assert small.budget_exhausted and large.budget_exhausted
    assert large.derived_dollars_per_mwh >= small.derived_dollars_per_mwh


# ---------------------------------------------------------------------------
# validate_scenario: min_export_price_mode
# ---------------------------------------------------------------------------

def test_validate_rolling_vpp_this_year_allows_absent_min_export_price(tmp_path):
    """Under a derived mode the configured value is a floor and may be
    omitted. Companion to test_validate_rolling_vpp_requires_min_export_price,
    which covers the same absent key under mode='fixed'."""
    build_full_fixture(tmp_path, year=2022)
    scenario = _rolling_vpp_scenario(retail_tariff_id='test_vpp_this_year')
    validate_scenario(scenario, _fixture_paths(tmp_path))


def test_validate_rolling_vpp_rejects_based_on_last_year(tmp_path):
    """The fixture states a min_export_price as well, so this cannot be
    passing for the absent-value reason instead."""
    build_full_fixture(tmp_path, year=2022)
    scenario = _rolling_vpp_scenario(retail_tariff_id='test_vpp_last_year')
    with pytest.raises(ScenarioValidationError, match='not implemented'):
        validate_scenario(scenario, _fixture_paths(tmp_path))


def test_validate_rolling_vpp_last_year_message_names_a_usable_mode(tmp_path):
    """A rejection that does not say what to write instead sends the reader to
    the source."""
    build_full_fixture(tmp_path, year=2022)
    scenario = _rolling_vpp_scenario(retail_tariff_id='test_vpp_last_year')
    with pytest.raises(ScenarioValidationError) as exc:
        validate_scenario(scenario, _fixture_paths(tmp_path))
    assert 'based_on_this_year' in str(exc.value)


# ---------------------------------------------------------------------------
# The resolution reaches the result and the summary
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_rolling_vpp_resolution_reaches_the_result_and_the_summary(tmp_path):
    """A run whose derived estimate collapsed to the configured floor is
    otherwise indistinguishable from one configured 'fixed', and a run whose
    threshold was derived from the modelled year's own prices is otherwise
    indistinguishable from a fully causal one."""
    build_full_fixture(tmp_path, year=2022)
    scenario = _rolling_vpp_scenario(retail_tariff_id='test_vpp_this_year')
    result = run_scenario(scenario, _fixture_paths(tmp_path), n_workers=1)

    r = result.min_export_price_resolution
    assert r is not None, "resolution did not survive to the ScenarioResult"
    assert r.mode == 'based_on_this_year'
    floor = 0.0 if r.configured_dollars_per_mwh is None \
        else r.configured_dollars_per_mwh
    assert r.value_dollars_per_mwh == pytest.approx(
        max(r.derived_dollars_per_mwh, floor)), (
        "the operative value is not max(derived, floor)"
    )

    text = write_scenario_summary(result, tmp_path / 'summary.txt')
    assert 'MINIMUM EXPORT PRICE' in text
    assert 'based_on_this_year' in text
    assert ('reached' in text), (
        "the summary must say whether the cap was reachable; a zero threshold "
        "means something different in each case"
    )


def test_scenario_summary_omits_the_block_when_there_is_no_resolution(tmp_path):
    """Non-vacuity for the test above: the block must be absent on a run that
    resolved nothing, or its presence there proves nothing."""
    build_full_fixture(tmp_path, year=2022)
    scenario = ScenarioConfig(
        state='NSW', year=2022, cohort='solar_battery',
        coordination_type='self_consumption',
        network_tariff_id='test_network',
        retail_tariff_id='test_tou',
        battery_spec=_bat(),
    )
    result = run_scenario(scenario, _fixture_paths(tmp_path), n_workers=1)
    assert result.min_export_price_resolution is None
    text = write_scenario_summary(result, tmp_path / 'summary.txt')
    assert 'MINIMUM EXPORT PRICE' not in text


# ---------------------------------------------------------------------------
# Site forecast providers reach the rolling VPP path
# ---------------------------------------------------------------------------
# run_rolling_vpp_event_site must build UsageForecastProvider and
# SolarForecastProvider from rolling_config exactly as
# run_rolling_perfect_foresight_lp_site does, and pass commit_actual_load
# through. It once did none of that, and the failure was silent:
# rolling_vpp_event defaults both providers to None and commit_actual_load to
# True, so a run configured for imperfect site knowledge completed with
# perfect knowledge, every invariant held, and only load_forecast_used /
# solar_forecast_used showed it. A full site-year was run and reported before
# anyone read those two flags.
#
# The assertions are on the diagnostics rather than on the call, because a
# test that patched the component would pass against a site function that
# built the providers and then dropped them.

def _rolling_vpp_scenario_with_site_forecasts(**overrides):
    from model.core.scenario import (
        RollingConfig, SolarForecastConfig, UsageForecastConfig,
    )
    scenario = _rolling_vpp_scenario(**overrides)
    return dataclasses.replace(
        scenario,
        rolling=RollingConfig(
            forecast_source='perfect',
            horizon_hours=scenario.rolling.horizon_hours,
            usage_forecast=UsageForecastConfig(),
            solar_forecast=SolarForecastConfig(),
        ),
    )


@pytest.mark.slow
@pytest.mark.parametrize('ct', ['rolling_vpp_sc_event',
                                'rolling_vpp_tou_event'])
def test_rolling_vpp_uses_the_configured_site_forecasts(tmp_path, ct):
    """Both coordination types, because the SC variant is the one where state
    is otherwise unused and the providers are easiest to omit."""
    build_full_fixture(tmp_path, year=2022)
    scenario = _rolling_vpp_scenario_with_site_forecasts(coordination_type=ct)
    result = run_scenario(scenario, _fixture_paths(tmp_path), n_workers=1)

    diag = next(iter(result.rolling_diagnostics.values()))
    assert diag.load_forecast_used is True, (
        "usage_forecast is configured but the run reported perfect load "
        "knowledge; the provider is not reaching rolling_vpp_event"
    )
    assert diag.solar_forecast_used is True, (
        "solar_forecast is configured but the run reported perfect solar "
        "knowledge; the provider is not reaching rolling_vpp_event"
    )


@pytest.mark.slow
def test_rolling_vpp_without_site_forecasts_reports_perfect_knowledge(tmp_path):
    """Non-vacuity for the test above. Absent config must report False, or
    those assertions would pass against a site function that hardcoded a
    provider regardless of what was asked for."""
    build_full_fixture(tmp_path, year=2022)
    result = run_scenario(_rolling_vpp_scenario(), _fixture_paths(tmp_path),
                          n_workers=1)
    diag = next(iter(result.rolling_diagnostics.values()))
    assert diag.load_forecast_used is False
    assert diag.solar_forecast_used is False


def test_rolling_vpp_site_forecast_without_a_state_fails_loudly(tmp_path):
    """Both providers resolve local days from state, which the SC variant
    otherwise never reads. Without this the failure surfaces inside the
    provider as something unrelated to the missing field."""
    from model.core.scenario import RollingConfig, UsageForecastConfig
    from model.orchestrate import run_rolling_vpp_event_site

    T = 288
    idx = pd.date_range('2022-06-01', periods=T, freq='5min', tz='Etc/GMT-10')
    with pytest.raises(ValueError, match='state is None'):
        run_rolling_vpp_event_site(
            'S1',
            pd.Series(np.zeros(T), index=idx),
            pd.Series(np.full(T, 0.05), index=idx),
            pd.Series(np.full(T, 50.0), index=idx),
            _bat(), None, None,
            RollingConfig(forecast_source='perfect', horizon_hours=1.0,
                          usage_forecast=UsageForecastConfig()),
            vpp_threshold_dollars_per_mwh=1000.0,
            event_window_hours=0.5,
            vpp_exp_reward=1.0, vpp_imp_reward=1.0,
            vpp_kwh_limit=250.0, vpp_limit_applies_to='export_only',
            min_export_price_dollars_per_mwh=0.0,
            gap_strategy='self_consumption',
            state=None,
        )


@pytest.mark.slow
def test_both_paths_resolve_the_same_yaml_to_the_same_threshold(tmp_path):
    """The property the shared resolver exists to guarantee.

    The value capture ratio between a rolling run and its perfect-foresight
    benchmark is meant to isolate the cost of foresight. Before the paths
    shared resolve_min_export_price, the perfect-foresight runner branched on
    whether the value was absent and never read min_export_price_mode, so a
    tariff setting BOTH a value and mode='based_on_this_year' -- which the AGL
    and EA 2026 plans do -- resolved to the stated value here and to the
    derived one there. That difference sat inside every ratio quoted between
    the two with nothing to attribute it to.

    test_vpp_this_year states the mode and no value, so both paths must derive
    the same number rather than both trivially reading the same constant.
    """
    build_full_fixture(tmp_path, year=2022)
    paths = _fixture_paths(tmp_path)

    pf = run_scenario(_pf_vpp_fixture_scenario('vpp_sc_event', 'test_vpp_this_year'),
                      paths, n_workers=1)
    rolling = run_scenario(
        _rolling_vpp_scenario(retail_tariff_id='test_vpp_this_year'),
        paths, n_workers=1)

    a, b = pf.min_export_price_resolution, rolling.min_export_price_resolution
    assert a is not None and b is not None
    assert a.mode == b.mode == 'based_on_this_year'
    assert a.derived_dollars_per_mwh == pytest.approx(
        b.derived_dollars_per_mwh), (
        "the two paths derived different thresholds from the same YAML; the "
        "value capture ratio between them would carry that difference "
        "unattributed"
    )
    assert a.value_dollars_per_mwh == pytest.approx(b.value_dollars_per_mwh)
    assert a.n_mask_intervals == b.n_mask_intervals, (
        "same mask rule, same prices, same event window -- a difference here "
        "means one path is classifying events differently"
    )
