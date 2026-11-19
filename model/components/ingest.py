"""Ingest component: loads and filters data for a single scenario.

This is the only component that reads from disk. All downstream components
receive their data via arguments.

For a given ScenarioConfig, ingest:
  1. Loads site metadata for the scenario state and year.
  2. Loads the clean EDP trace file for each site in the metadata.
  3. Loads the NEM price series for the scenario state and year.
  4. Loads the network and retail tariffs identified by the scenario.
  5. Verifies that site traces and prices share the same 5-min datetime index.
  6. Converts trace energy values from Wh to kWh.
  7. Returns a DataContext bundling everything together.

Failure modes (all raise informative errors):
  - missing metadata file
  - no sites in metadata for the scenario state
  - missing clean trace file for a listed site
  - missing NEM price file
  - missing tariff file
  - misaligned datetime indices between traces and prices
"""
from pathlib import Path
from typing import Callable, Optional
import warnings

import pandas as pd

from model.core.scenario import ScenarioConfig
from model.core.tariff import load_network_tariff, load_retail_tariff
from model.core.data_context import DataContext
from model.core.paths import Paths


# Trace columns retained in DataContext (after Wh -> kWh conversion).
# Pipeline produces values in Wh per 5-min interval; we convert to kWh.
_TRACE_ENERGY_COLUMNS = [
    'pv_site_net_wh',                  # net PV (post any battery on the PV circuit)
    'gross_load_wh',                   # total household load
    'ac_load_net_wh',                  # load net of self-consumption from solar
    'pv_original_wh',                  # PV value before any infill replacement
    'ac_load_net_original_wh',         # ac_load_net value before any infill replacement
]

# Non-energy trace columns retained as-is (no unit conversion).
# pv_flag / load_flag are separate quality codes produced by the EDP cleaner
# (pipeline/edp/), one per stream. pv_infill / load_infill are the matching
# infill-method markers (0=none, 1=Method 1 short-gap, 2=Method 2 synthetic /
# same-DOW).
_TRACE_OTHER_COLUMNS = [
    'pv_flag',
    'load_flag',
    'pv_infill',
    'load_infill',
]

# Metadata columns retained from YYYY_selected_sites_clean.csv.
# Per Fred 2026-05-27: only the inverter/PV system fields and the annual
# generation / avg daily load summaries are likely to be relevant to
# downstream modelling. Everything else relates to the cleaning and infill
# process and is dropped here.
#
# ac_load_net_phases / pv_site_net_phases are the connection phase counts,
# retained for per-site export-limit resolution (see
# orchestrate._resolve_per_site_export_limits). Dimensionless; not converted.
#
# avg_daily_gross_load_wh is converted to kWh and renamed in ingest output
# (see _load_site_metadata) to keep all energy units in DataContext in kWh.
_METADATA_COLUMNS = [
    'edp_site_id',
    'state',
    'inverter_manufacturer',
    'inverter_model',
    'inverter_ac_rating_kw',
    'subarray_dc_rating_kw',
    'ac_load_net_phases',
    'pv_site_net_phases',
    'annual_generation_kwh',
    'avg_daily_gross_load_wh',
]

# Map ScenarioConfig.state to the column name in the NEM price file.
# Per nem_price_spec.md, clean price files have NSW1_rrp, QLD1_rrp etc.
_STATE_TO_RRP_COLUMN = {
    'NSW': 'NSW1_rrp',
    'QLD': 'QLD1_rrp',
    'SA': 'SA1_rrp',
    'VIC': 'VIC1_rrp',
}

# Expected interval count per year (matches EDP and NEM specs).
_LEAP_YEAR_INTERVALS = 105_408
_NON_LEAP_YEAR_INTERVALS = 105_120


def _expected_interval_count(year: int) -> int:
    is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    return _LEAP_YEAR_INTERVALS if is_leap else _NON_LEAP_YEAR_INTERVALS


def _load_site_metadata(scenario: ScenarioConfig, paths: Paths) -> pd.DataFrame:
    """Load site metadata filtered to the scenario state.

    Reads YYYY_selected_sites_clean.csv from edp_metadata_dir and returns
    a DataFrame containing only the metadata columns relevant downstream,
    filtered to sites in the scenario state.
    """
    metadata_path = paths.edp_metadata_dir / f'{scenario.year}_selected_sites_clean.csv'
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Site metadata file not found: {metadata_path}\n"
            f"Expected output of pipeline script C_Check_Consistency.py "
            f"for year {scenario.year}."
        )

    df = pd.read_csv(metadata_path)

    missing_cols = [c for c in _METADATA_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Site metadata file {metadata_path} is missing expected "
            f"columns: {missing_cols}. Available columns: {list(df.columns)}"
        )

    df = df[_METADATA_COLUMNS].copy()
    df_state = df[df['state'] == scenario.state].reset_index(drop=True)

    if df_state.empty:
        raise ValueError(
            f"No sites found in {metadata_path} for state={scenario.state!r}. "
            f"Available states: {sorted(df['state'].unique().tolist())}"
        )

    # Convert avg_daily_gross_load_wh -> avg_daily_gross_load_kwh for
    # consistency with the kWh convention used elsewhere in DataContext.
    df_state['avg_daily_gross_load_kwh'] = df_state['avg_daily_gross_load_wh'] / 1000.0
    df_state = df_state.drop(columns=['avg_daily_gross_load_wh'])

    return df_state


def _load_site_trace(site_id: str, year: int, edp_clean_dir: Path) -> pd.DataFrame:
    """Load a single site's clean trace file and convert energy columns to kWh.

    Returns a DataFrame indexed by datetime_aest (tz-aware UTC+10), with
    energy columns in kWh per 5-min interval.

    Clean trace files are stored as Parquet (snappy-compressed) per the EDP
    data spec, written by the last pipeline step, L_Convert_To_Parquet.py.
    """
    trace_path = edp_clean_dir / str(year) / f'edp_clean_{site_id}_{year}.parquet'
    if not trace_path.exists():
        raise FileNotFoundError(
            f"Clean trace file not found for site {site_id} year {year}: "
            f"{trace_path}. Build the clean traces with "
            f"`python3 -u pipeline/run_pipeline.py {year}` "
            f"(see docs/edp_data_spec.md)."
        )

    df = pd.read_parquet(trace_path)

    missing_energy = [c for c in _TRACE_ENERGY_COLUMNS if c not in df.columns]
    if missing_energy:
        raise ValueError(
            f"Site trace {trace_path} missing expected energy columns: "
            f"{missing_energy}"
        )

    missing_other = [c for c in _TRACE_OTHER_COLUMNS if c not in df.columns]
    if missing_other:
        raise ValueError(
            f"Site trace {trace_path} missing expected columns: {missing_other}"
        )

    # Convert Wh -> kWh and rename columns to make the unit explicit
    df_out = pd.DataFrame(
        index=pd.to_datetime(df['datetime_aest']).dt.tz_convert('Etc/GMT-10')
    )
    df_out.index.name = 'datetime_aest'

    for col in _TRACE_ENERGY_COLUMNS:
        new_name = col.replace('_wh', '_kwh')
        df_out[new_name] = df[col].values / 1000.0

    for col in _TRACE_OTHER_COLUMNS:
        df_out[col] = df[col].values

    return df_out


def _load_prices(scenario: ScenarioConfig, paths: Paths) -> pd.Series:
    """Load the NEM RRP series for the scenario state and year.

    Returns a Series indexed by datetime_aest in $/MWh (no unit conversion).
    Warns (does not raise) if any RRP values are missing (price_flag == 1);
    the NaN prices are passed through for downstream components to handle.

    Clean price files are stored as Parquet (snappy-compressed) per the NEM
    price spec, written by B_Clean_NEM_Prices.py.
    """
    price_path = paths.nem_prices_dir / f'{scenario.year}_nem_prices.parquet'
    if not price_path.exists():
        raise FileNotFoundError(
            f"NEM price file not found: {price_path}. Build the price files "
            f"with pipeline/nem_prices/A_Download_NEM_Prices.py and "
            f"B_Clean_NEM_Prices.py (see docs/nem_price_spec.md)."
        )

    rrp_col = _STATE_TO_RRP_COLUMN[scenario.state]
    df = pd.read_parquet(price_path)

    if rrp_col not in df.columns:
        raise ValueError(
            f"NEM price file {price_path} missing expected column {rrp_col!r}. "
            f"Available columns: {list(df.columns)}"
        )
    if 'price_flag' not in df.columns:
        raise ValueError(
            f"NEM price file {price_path} missing expected column 'price_flag'."
        )

    flagged_count = int((df['price_flag'] == 1).sum())
    if flagged_count > 0:
        # Flagged intervals indicate genuine missing prices (e.g. NEM suspension).
        # Surface this rather than silently NaN-handling downstream.
        warnings.warn(
            f"NEM price file {price_path} has {flagged_count} intervals "
            f"with price_flag=1 (missing prices). Downstream components "
            f"must handle NaN prices explicitly."
        )

    series = pd.Series(
        df[rrp_col].values,
        index=pd.to_datetime(df['datetime_aest']).dt.tz_convert('Etc/GMT-10'),
        name=f'{scenario.state}_rrp_dollars_per_mwh',
    )
    series.index.name = 'datetime_aest'
    return series


def _verify_index_alignment(
    site_traces: dict[str, pd.DataFrame],
    prices: pd.Series,
    year: int,
) -> None:
    """Verify all site traces and the price series share an identical
    5-min datetime index covering the full year.

    Raises if any site index does not exactly match the price index.

    Index comparison is on absolute instants (both indexes converted to UTC),
    not on the timezone label. This means an index labelled 'UTC+10:00'
    (fixed offset) and one labelled 'Etc/GMT-10' (named zone) compare equal,
    as they represent the same instants. Using .equals() would reject them
    because it also compares the tz label / dtype.
    """
    expected_count = _expected_interval_count(year)
    if len(prices) != expected_count:
        raise ValueError(
            f"Price series has {len(prices)} intervals, expected "
            f"{expected_count} for year {year}."
        )

    price_index = prices.index
    price_index_utc = price_index.tz_convert("UTC")
    for site_id, trace in site_traces.items():
        if len(trace) != expected_count:
            raise ValueError(
                f"Site {site_id} trace has {len(trace)} intervals, expected "
                f"{expected_count} for year {year}."
            )
        same_instants = (
            len(trace.index) == len(price_index)
            and (trace.index.tz_convert("UTC") == price_index_utc).all()
        )
        if not same_instants:
            # Help diagnose the mismatch - common cause is timezone awareness.
            raise ValueError(
                f"Site {site_id} trace index does not match price index. "
                f"Trace tz={trace.index.tz}, price tz={price_index.tz}. "
                f"First trace ts={trace.index[0]}, first price ts={price_index[0]}. "
                f"Last trace ts={trace.index[-1]}, last price ts={price_index[-1]}."
            )


def ingest(
    scenario: ScenarioConfig,
    paths: Paths,
    *,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> DataContext:
    """Load and filter all input data for a single scenario.

    This is the only component that reads from disk.

    Args:
        scenario: Fully-validated ScenarioConfig.
        paths: Filesystem locations of input data and tariff files.
        progress_callback: Optional callable(i, n, site_id) invoked after each
            site trace file finishes loading. Lets long-running ingests show
            activity to the user. Called 1..n inclusive; defaults to no-op.

    Returns:
        A DataContext bundling site traces (in kWh), the regional price
        series (in $/MWh), site metadata, the loaded Tariff, and the
        original ScenarioConfig.

    Raises:
        FileNotFoundError: if any required input file is missing.
        ValueError: if loaded data is missing expected columns, has the
            wrong shape, or does not align across sources.
    """
    site_metadata = _load_site_metadata(scenario, paths)
    site_ids = site_metadata['edp_site_id'].tolist()
    n_sites = len(site_ids)

    site_traces: dict[str, pd.DataFrame] = {}
    for i, site_id in enumerate(site_ids, start=1):
        site_traces[site_id] = _load_site_trace(
            site_id=site_id,
            year=scenario.year,
            edp_clean_dir=paths.edp_clean_dir,
        )
        if progress_callback is not None:
            progress_callback(i, n_sites, site_id)

    prices = _load_prices(scenario, paths)

    _verify_index_alignment(site_traces, prices, scenario.year)

    network_tariff = load_network_tariff(scenario.network_tariff_id,
                                         paths.tariffs_dir)
    retail_tariff = load_retail_tariff(scenario.retail_tariff_id,
                                       paths.tariffs_dir)

    for label, tariff_id, tstate in (
        ('Network', scenario.network_tariff_id, network_tariff.state),
        ('Retail', scenario.retail_tariff_id, retail_tariff.state),
    ):
        if tstate != scenario.state:
            warnings.warn(
                f"{label} tariff {tariff_id!r} is configured for state "
                f"{tstate!r} but scenario state is {scenario.state!r}. "
                f"Proceeding, but check this is intentional."
            )

    return DataContext(
        site_traces=site_traces,
        prices=prices,
        site_metadata=site_metadata,
        network_tariff=network_tariff,
        retail_tariff=retail_tariff,
        scenario=scenario,
    )