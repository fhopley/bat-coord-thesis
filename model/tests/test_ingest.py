"""Tests for ingest.py.

Uses ingest_fixture_builder to create a synthetic on-disk fixture tree
at a tmp_path, then exercises ingest() against it. The fixture uses
constant trace values so assertions can use exact equality.
"""
from pathlib import Path
import warnings

import pytest
import pandas as pd

from model.core import ScenarioConfig, BatterySpec, Paths
from model.components.ingest import ingest

from model.tests.fixtures.ingest_fixture_builder import (
    build_full_fixture,
    write_site_metadata,
    write_site_trace,
    write_nem_prices,
)


FIXTURE_TARIFFS_DIR = Path(__file__).parent / 'fixtures' / 'tariffs'


def _paths_for(root: Path) -> Paths:
    """Construct Paths pointing at a tmp fixture tree, but with tariffs
    coming from the test fixtures dir (not the production model/tariffs/)."""
    return Paths(
        edp_clean_dir=root / 'data' / 'clean' / 'edp',
        edp_metadata_dir=root / 'data' / 'processed' / 'edp',
        nem_prices_dir=root / 'data' / 'clean' / 'nem_prices',
        tariffs_dir=FIXTURE_TARIFFS_DIR,
    )


def _basic_scenario(state: str = 'NSW', year: int = 2022) -> ScenarioConfig:
    return ScenarioConfig(
        state=state,
        year=year,
        cohort='solar',
        network_tariff_id='test_network',
        retail_tariff_id='test_tou',
    )


# ---------- Happy path ----------

def test_ingest_basic_solar_scenario(tmp_path):
    build_full_fixture(tmp_path, year=2022)
    ctx = ingest(_basic_scenario(), _paths_for(tmp_path))

    # Two NSW sites in metadata
    assert set(ctx.site_traces.keys()) == {'S0001', 'S0002'}
    assert len(ctx.site_metadata) == 2
    assert (ctx.site_metadata['state'] == 'NSW').all()


def test_ingest_filters_to_state(tmp_path):
    """VIC site in metadata must not appear when scenario is NSW."""
    build_full_fixture(tmp_path, year=2022)
    ctx = ingest(_basic_scenario(state='NSW'), _paths_for(tmp_path))
    assert 'S0003' not in ctx.site_traces

    ctx_vic = ingest(_basic_scenario(state='VIC'), _paths_for(tmp_path))
    assert set(ctx_vic.site_traces.keys()) == {'S0003'}


def test_ingest_wh_to_kwh_conversion(tmp_path):
    """Fixture has 100 Wh per interval for PV; ingest should expose 0.1 kWh."""
    build_full_fixture(tmp_path, year=2022)
    ctx = ingest(_basic_scenario(), _paths_for(tmp_path))
    trace = ctx.site_traces['S0001']

    # Columns renamed _wh -> _kwh
    assert 'pv_site_net_kwh' in trace.columns
    assert 'gross_load_kwh' in trace.columns
    assert 'pv_site_net_wh' not in trace.columns

    # 100 Wh -> 0.1 kWh
    assert trace['pv_site_net_kwh'].iloc[0] == pytest.approx(0.1)
    assert trace['gross_load_kwh'].iloc[0] == pytest.approx(0.2)
    assert trace['ac_load_net_kwh'].iloc[0] == pytest.approx(0.1)


def test_ingest_non_energy_columns_pass_through(tmp_path):
    build_full_fixture(tmp_path, year=2022)
    ctx = ingest(_basic_scenario(), _paths_for(tmp_path))
    trace = ctx.site_traces['S0001']
    assert 'pv_flag' in trace.columns
    assert 'load_flag' in trace.columns
    assert 'pv_infill' in trace.columns
    assert 'load_infill' in trace.columns
    assert (trace['pv_flag'] == 0).all()
    assert (trace['load_flag'] == 0).all()


def test_ingest_prices_in_dollars_per_mwh(tmp_path):
    """Prices stay in $/MWh — ingest does not convert."""
    build_full_fixture(tmp_path, year=2022)
    ctx = ingest(_basic_scenario(), _paths_for(tmp_path))
    assert ctx.prices.iloc[0] == 50.0  # NSW fixture value
    assert ctx.prices.name == 'NSW_rrp_dollars_per_mwh'


def test_ingest_prices_match_state(tmp_path):
    build_full_fixture(tmp_path, year=2022)
    ctx_vic = ingest(_basic_scenario(state='VIC'), _paths_for(tmp_path))
    assert ctx_vic.prices.iloc[0] == 55.0  # VIC1 fixture value


def test_ingest_index_alignment(tmp_path):
    """Site trace index and price index should be identical."""
    build_full_fixture(tmp_path, year=2022)
    ctx = ingest(_basic_scenario(), _paths_for(tmp_path))
    trace = ctx.site_traces['S0001']
    assert trace.index.equals(ctx.prices.index)


def test_ingest_full_year_interval_count(tmp_path):
    """2022 is a non-leap year -> 105,120 intervals."""
    build_full_fixture(tmp_path, year=2022)
    ctx = ingest(_basic_scenario(), _paths_for(tmp_path))
    assert len(ctx.prices) == 105_120
    assert len(ctx.site_traces['S0001']) == 105_120


def test_ingest_leap_year_interval_count(tmp_path):
    """2020 is a leap year -> 105,408 intervals."""
    build_full_fixture(tmp_path, year=2020)
    ctx = ingest(_basic_scenario(year=2020), _paths_for(tmp_path))
    assert len(ctx.prices) == 105_408


def test_ingest_metadata_columns(tmp_path):
    build_full_fixture(tmp_path, year=2022)
    ctx = ingest(_basic_scenario(), _paths_for(tmp_path))
    expected = {
        'edp_site_id', 'state', 'inverter_manufacturer',
        'inverter_model', 'inverter_ac_rating_kw', 'subarray_dc_rating_kw',
        'ac_load_net_phases', 'pv_site_net_phases',
        'annual_generation_kwh', 'avg_daily_gross_load_kwh',
    }
    assert set(ctx.site_metadata.columns) == expected


def test_ingest_metadata_load_wh_to_kwh(tmp_path):
    """Fixture has avg_daily_gross_load_wh=20000; ingest exposes 20.0 kWh."""
    build_full_fixture(tmp_path, year=2022)
    ctx = ingest(_basic_scenario(), _paths_for(tmp_path))
    assert ctx.site_metadata['avg_daily_gross_load_kwh'].iloc[0] == pytest.approx(20.0)
    # Original Wh column dropped to avoid confusion
    assert 'avg_daily_gross_load_wh' not in ctx.site_metadata.columns
    # annual_generation_kwh passed through unchanged
    assert ctx.site_metadata['annual_generation_kwh'].iloc[0] == pytest.approx(8000.0)


def test_ingest_returns_loaded_tariffs(tmp_path):
    build_full_fixture(tmp_path, year=2022)
    ctx = ingest(_basic_scenario(), _paths_for(tmp_path))
    assert ctx.retail_tariff.type == 'tou'
    assert ctx.retail_tariff.daily_c_per_day == 100.0
    assert ctx.network_tariff.name == 'Test Network'
    assert ctx.network_tariff.usage_dlf == 1.05
    assert ctx.network_tariff.export_dlf == 0.95


def test_ingest_includes_original_scenario(tmp_path):
    build_full_fixture(tmp_path, year=2022)
    scenario = _basic_scenario()
    ctx = ingest(scenario, _paths_for(tmp_path))
    assert ctx.scenario is scenario


def test_ingest_solar_battery_scenario(tmp_path):
    """solar_battery scenario should ingest exactly the same data — the
    battery is added later, not at ingest."""
    build_full_fixture(tmp_path, year=2022)
    scenario = ScenarioConfig(
        state='NSW', year=2022, cohort='solar_battery',
        network_tariff_id='test_network',
        retail_tariff_id='test_tou',
        coordination_type='self_consumption',
        battery_spec=BatterySpec(
            capacity_kwh=13.5, backup_reserve_pct=20.0, power_kw=5.0,
            charge_efficiency=0.95, discharge_efficiency=0.95,
        ),
    )
    ctx = ingest(scenario, _paths_for(tmp_path))
    assert set(ctx.site_traces.keys()) == {'S0001', 'S0002'}


# ---------- Failure modes ----------

def test_ingest_missing_metadata_file(tmp_path):
    paths = _paths_for(tmp_path)
    with pytest.raises(FileNotFoundError, match='Site metadata file not found'):
        ingest(_basic_scenario(), paths)


def test_ingest_no_sites_for_state(tmp_path):
    """If metadata has no sites for the scenario state, raise."""
    build_full_fixture(tmp_path, year=2022)
    with pytest.raises(ValueError, match='No sites found'):
        ingest(_basic_scenario(state='SA'), _paths_for(tmp_path))


def test_ingest_missing_site_trace(tmp_path):
    """Metadata lists a site but the per-site trace file is missing."""
    write_site_metadata(
        tmp_path / 'data' / 'processed' / 'edp' / '2022_selected_sites_clean.csv',
        sites=[('S0001', 'NSW'), ('S0099', 'NSW')],  # S0099 has no trace file
    )
    write_site_trace(
        tmp_path / 'data' / 'clean' / 'edp' / '2022' / 'edp_clean_S0001_2022.parquet',
        year=2022,
    )
    write_nem_prices(
        tmp_path / 'data' / 'clean' / 'nem_prices' / '2022_nem_prices.parquet',
        year=2022,
    )
    with pytest.raises(FileNotFoundError, match='Clean trace file not found'):
        ingest(_basic_scenario(), _paths_for(tmp_path))


def test_ingest_missing_price_file(tmp_path):
    write_site_metadata(
        tmp_path / 'data' / 'processed' / 'edp' / '2022_selected_sites_clean.csv',
        sites=[('S0001', 'NSW')],
    )
    write_site_trace(
        tmp_path / 'data' / 'clean' / 'edp' / '2022' / 'edp_clean_S0001_2022.parquet',
        year=2022,
    )
    with pytest.raises(FileNotFoundError, match='NEM price file not found'):
        ingest(_basic_scenario(), _paths_for(tmp_path))


def test_ingest_metadata_missing_expected_column(tmp_path):
    """Metadata file exists but is missing a required column."""
    # Write a minimal metadata file lacking inverter_ac_rating_kw
    path = tmp_path / 'data' / 'processed' / 'edp' / '2022_selected_sites_clean.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'edp_site_id': ['S0001'], 'state': ['NSW']}).to_csv(path, index=False)
    with pytest.raises(ValueError, match='missing expected columns'):
        ingest(_basic_scenario(), _paths_for(tmp_path))


def test_ingest_warns_when_tariff_state_mismatches(tmp_path):
    """Tariff fixture is NSW; scenario VIC should warn but not error."""
    # Build fixture with a VIC site so the state filter doesn't fail first.
    write_site_metadata(
        tmp_path / 'data' / 'processed' / 'edp' / '2022_selected_sites_clean.csv',
        sites=[('S0003', 'VIC')],
    )
    write_site_trace(
        tmp_path / 'data' / 'clean' / 'edp' / '2022' / 'edp_clean_S0003_2022.parquet',
        year=2022,
    )
    write_nem_prices(
        tmp_path / 'data' / 'clean' / 'nem_prices' / '2022_nem_prices.parquet',
        year=2022,
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        ingest(_basic_scenario(state='VIC'), _paths_for(tmp_path))
        assert any('state' in str(warning.message).lower() for warning in w)


def test_ingest_warns_when_prices_have_missing_flag(tmp_path):
    """price_flag=1 anywhere should emit a warning."""
    build_full_fixture(tmp_path, year=2022)
    # Overwrite the price file with one flagged interval
    price_path = tmp_path / 'data' / 'clean' / 'nem_prices' / '2022_nem_prices.parquet'
    df = pd.read_parquet(price_path)
    df.loc[100, 'price_flag'] = 1
    df.loc[100, 'NSW1_rrp'] = float('nan')
    df.to_parquet(price_path, engine='pyarrow', compression='snappy', index=False)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        ingest(_basic_scenario(), _paths_for(tmp_path))
        assert any('price_flag=1' in str(warning.message) for warning in w)


def test_ingest_data_context_is_frozen(tmp_path):
    """Top-level DataContext fields cannot be reassigned."""
    build_full_fixture(tmp_path, year=2022)
    ctx = ingest(_basic_scenario(), _paths_for(tmp_path))
    with pytest.raises(Exception):  # FrozenInstanceError
        ctx.prices = pd.Series([1, 2, 3])