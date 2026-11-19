"""Fixture builder for ingest tests.

Constructs synthetic EDP clean files, NEM price files, and site metadata
matching the schemas described in docs/edp_data_spec.md and docs/nem_price_spec.md.

Used by test_ingest.py to set up a complete on-disk fixture tree without
depending on real (large, private) production data.

The fixtures are deliberately small — typically 3 sites for a single short
year window — and use predictable patterns rather than realistic values
so test assertions can use exact equality.
"""
import datetime as dt
from pathlib import Path
import pandas as pd
import numpy as np

# The timezone the files carry: a fixed UTC+10:00 offset, which is what
# parsing '+10:00' timestamp strings produces. Converting the index directly
# gives the same column as formatting it to strings and parsing them back,
# without a full-year strftime (about 5 s per file).
_AEST_FIXED_OFFSET = dt.timezone(dt.timedelta(hours=10))


def build_datetime_index(year: int) -> pd.DatetimeIndex:
    """Build the complete 5-min AEST interval-end index for a year.

    Matches the convention in the EDP and NEM clean files: AEST (UTC+10)
    fixed offset, interval-end timestamps, starting at YYYY-01-01 00:05:00+10:00
    and ending at YYYY+1-01-01 00:00:00+10:00.
    """
    start = pd.Timestamp(f'{year}-01-01 00:05:00', tz='Etc/GMT-10')
    end = pd.Timestamp(f'{year + 1}-01-01 00:00:00', tz='Etc/GMT-10')
    return pd.date_range(start=start, end=end, freq='5min')


def write_site_trace(
    out_path: Path,
    year: int,
    pv_wh_per_interval: float = 100.0,
    load_wh_per_interval: float = 200.0,
) -> None:
    """Write a synthetic site clean file with constant trace values.

    Constant values per interval make downstream test assertions trivial
    (e.g. total annual PV = constant * interval_count) without obscuring
    bugs in unit conversion.

    Schema matches what pipeline scripts E/G/I produce.
    """
    idx = build_datetime_index(year)
    n = len(idx)
    df = pd.DataFrame({
        'datetime_aest': idx.tz_convert(_AEST_FIXED_OFFSET),
        'pv_site_net_wh': np.full(n, pv_wh_per_interval, dtype=float),
        'gross_load_wh': np.full(n, load_wh_per_interval, dtype=float),
        'ac_load_net_wh': np.full(
            n, load_wh_per_interval - pv_wh_per_interval, dtype=float
        ),
        'pv_original_wh': np.full(n, pv_wh_per_interval, dtype=float),
        'ac_load_net_original_wh': np.full(
            n, load_wh_per_interval - pv_wh_per_interval, dtype=float
        ),
        'pv_flag': np.zeros(n, dtype=int),
        'load_flag': np.zeros(n, dtype=int),
        'pv_infill': np.zeros(n, dtype=int),
        'load_infill': np.zeros(n, dtype=int),
    })
    # Ensure tz-aware datetime so the Parquet output matches the cleaner's
    # native timezone storage.
    df['datetime_aest'] = pd.to_datetime(df['datetime_aest'])
    if df['datetime_aest'].dt.tz is None:
        df['datetime_aest'] = df['datetime_aest'].dt.tz_localize('Etc/GMT-10')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, engine='pyarrow', compression='snappy',
                  index=False)


def write_nem_prices(
    out_path: Path,
    year: int,
    rrp_values: dict[str, float] | None = None,
) -> None:
    """Write a synthetic NEM prices file with constant per-region RRPs.

    Matches the schema from docs/nem_price_spec.md.
    """
    if rrp_values is None:
        rrp_values = {'NSW1': 50.0, 'QLD1': 60.0, 'SA1': 70.0, 'VIC1': 55.0}

    idx = build_datetime_index(year)
    n = len(idx)
    df = pd.DataFrame({
        'datetime_aest': idx.tz_convert(_AEST_FIXED_OFFSET),
        'NSW1_rrp': np.full(n, rrp_values['NSW1']),
        'QLD1_rrp': np.full(n, rrp_values['QLD1']),
        'SA1_rrp': np.full(n, rrp_values['SA1']),
        'VIC1_rrp': np.full(n, rrp_values['VIC1']),
        'price_flag': np.zeros(n, dtype=int),
    })
    df['datetime_aest'] = pd.to_datetime(df['datetime_aest'])
    if df['datetime_aest'].dt.tz is None:
        df['datetime_aest'] = df['datetime_aest'].dt.tz_localize('Etc/GMT-10')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, engine='pyarrow', compression='snappy',
                  index=False)


def write_site_metadata(
    out_path: Path,
    sites: list[tuple[str, str]],
    inverter_kw: float = 5.0,
    subarray_kw: float = 6.6,
    annual_generation_kwh: float = 8000.0,
    avg_daily_gross_load_wh: float = 20000.0,
    ac_load_net_phases: int = 1,
    pv_site_net_phases: int = 1,
) -> None:
    """Write a synthetic YYYY_selected_sites_clean.csv with the given sites.

    Args:
        out_path: Output CSV path.
        sites: List of (site_id, state) tuples.
        inverter_kw: AC rating applied to all sites.
        subarray_kw: DC rating applied to all sites.
        annual_generation_kwh: PV annual generation applied to all sites.
        avg_daily_gross_load_wh: Average daily gross load (Wh) applied to
            all sites. In Wh in the source file to match the pipeline
            naming; ingest converts to kWh.
        ac_load_net_phases: Number of load connection phases (uniform across
            sites). Used by orchestrate._resolve_per_site_export_limits.
        pv_site_net_phases: Number of PV connection phases (uniform across
            sites). Used by orchestrate._resolve_per_site_export_limits.
    """
    n = len(sites)
    df = pd.DataFrame({
        'edp_site_id': [s[0] for s in sites],
        'state': [s[1] for s in sites],
        'inverter_manufacturer': ['TestMfr'] * n,
        'inverter_model': ['TestModel'] * n,
        'inverter_ac_rating_kw': [inverter_kw] * n,
        'subarray_dc_rating_kw': [subarray_kw] * n,
        'ac_load_net_phases': [ac_load_net_phases] * n,
        'pv_site_net_phases': [pv_site_net_phases] * n,
        'annual_generation_kwh': [annual_generation_kwh] * n,
        'avg_daily_gross_load_wh': [avg_daily_gross_load_wh] * n,
        # Extra columns ingest should ignore - prove they don't cause issues
        'cleaning_field_1': ['x'] * n,
        'cleaning_field_2': [1.234] * n,
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def build_full_fixture(root: Path, year: int = 2022) -> None:
    """Build a complete fixture tree at `root`.

    Layout (mirrors production):
      root/
        data/
          clean/
            edp/
              YYYY/
                edp_clean_S0001_YYYY.parquet  (NSW)
                edp_clean_S0002_YYYY.parquet  (NSW)
                edp_clean_S0003_YYYY.parquet  (VIC)
            nem_prices/
              YYYY_nem_prices.parquet
          processed/
            edp/
              YYYY_selected_sites_clean.csv
        model/
          tariffs/
            (no production tariffs needed - tests use fixtures/tariffs)
    """
    # Site metadata (small file - stays CSV)
    write_site_metadata(
        root / 'data' / 'processed' / 'edp' / f'{year}_selected_sites_clean.csv',
        sites=[('S0001', 'NSW'), ('S0002', 'NSW'), ('S0003', 'VIC')],
    )

    # Per-site clean files (Parquet)
    clean_dir = root / 'data' / 'clean' / 'edp' / str(year)
    write_site_trace(clean_dir / f'edp_clean_S0001_{year}.parquet', year)
    write_site_trace(clean_dir / f'edp_clean_S0002_{year}.parquet', year)
    write_site_trace(clean_dir / f'edp_clean_S0003_{year}.parquet', year)

    # NEM prices (Parquet)
    write_nem_prices(
        root / 'data' / 'clean' / 'nem_prices' / f'{year}_nem_prices.parquet',
        year,
    )