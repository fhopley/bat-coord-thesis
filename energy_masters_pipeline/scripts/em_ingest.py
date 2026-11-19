"""Ingestion for Energy Masters Amber telemetry sites.

Implements the two shapes designed in amber_site_ingest_design.md:

    load_em_site_trace(site_id, valid_periods, ...)
        -> DataFrame[pv_site_net_kwh, gross_load_kwh], indexed by
        datetime_aest. Feeds self_consumption / perfect_foresight_lp
        exactly like _dev_common.load_site_trace does for EDP sites.

    build_em_actual_traces(site_id, valid_periods, ...)
        -> SiteTraces built from the site's actual Amber-coordinated
        behaviour. Only import_kwh/export_kwh are real; the other eight
        fields (the flow decomposition, battery_soc_kwh) are all-zero,
        since compute_bill was confirmed to never read them.

    reconcile_prices(site_id, valid_periods, ...)
        Compares our derived (p_imp, p_exp) -- built the same way the
        perfect-foresight LP and billing build them -- against Amber's own
        published import_rate/export_rate for the same intervals.

Fixed for the whole Energy Masters cohort: state='SA'. Default tariffs:
network SA_SAPN_RTOU_2026, retail SA_amber_wholesale_2026 (network tariff
may need a 2025 vintage added later to match the data's timeframe --
2025-09 through 2026-03 for EM-427 -- see the module docstring note below).

USAGE
-----
Run as a module, from the project root, exactly like
viz_perfect_foresight_lp.py documents for itself -- NOT as a direct file
path. Running `python3 energy_masters_pipeline/scripts/em_ingest.py EM-427` puts energy_masters_pipeline/scripts/ on
sys.path instead of the project root, so `import model.core.paths` fails
with ModuleNotFoundError. The -m form puts the current working directory
(the project root, assuming you're standing in it) on sys.path instead:

    python3 -m energy_masters_pipeline.scripts.em_ingest EM-427
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from model.core.paths import default_paths, Paths
from model.core.tariff import (
    NetworkTariff, RetailTariff, load_network_tariff, load_retail_tariff,
)
from model.core.prices import build_price_vectors
from model.core.site_result import SiteTraces

STATE = 'SA'  # fixed for the whole Energy Masters cohort, confirmed
DEFAULT_NETWORK_TARIFF_ID = 'SA_SAPN_RESELE_2025'
DEFAULT_RETAIL_TARIFF_ID = 'SA_amber_wholesale_2026'
DEFAULT_TELEMETRY_PATH = Path('data/energy_masters_amber_data/amber_telemetry.parquet')
DEFAULT_SITE_SUMMARY_PATH = Path('data/processed/energy_masters/site_summary_results.csv')

# Confirmed 30-minute block-average (AEMO TWAP) reconstruction, validated
# against EM-427's full 201-day valid period and unanimous across the
# clean-14 shortlist (98.0-99.7% of blocks exactly 6 ticks wide). Content
# grouping bundles timestamps t+5..t+30 into the block labelled t
# (phase_offset=5, matching TariffPeriod's own interval-START assignment
# convention one level up); the broadcast shift then lands the result on
# the ticks 5 minutes earlier than its own constituent members, where
# Amber actually displays it. See amber_site_ingest_design.md's
# "Settlement resolution" section for the full derivation.
SETTLEMENT_PHASE_OFFSET_MINUTES = 5
SETTLEMENT_BROADCAST_SHIFT_TICKS = -1

# Confirmed necessary by a real case (EM-414, 2025-12-25 10:00, pv_dc__1
# jumped to 15,297 Wh in one 5-min interval from single-digit Wh readings
# either side -- implying ~184kW, physically impossible for this cohort).
# Present-but-implausible values aren't caught by interpolation (which
# only fills genuinely missing/NaN values) -- treating anything beyond
# this threshold as if it were missing closes that gap with the same
# "basic, not the full EDP flagging system" policy already agreed for
# missing data, rather than leaving a confirmed-real glitch unaddressed.
# Applies to raw Wh energy-flow columns only (ac_load_net, pv_dc,
# battery), never to price/rate columns, where a genuine value can
# legitimately be this large (see the price-spike investigation earlier
# in this thread -- clipping there would remove exactly the events we
# care about most).
MAX_PLAUSIBLE_WH_PER_INTERVAL = 5000.0

INTERVAL_HOURS = 5 / 60


# ---------------------------------------------------------------------------
# Valid-period lookup
# ---------------------------------------------------------------------------

def load_valid_periods_from_summary(
    site_id: str, summary_csv_path: Path = DEFAULT_SITE_SUMMARY_PATH,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Parse a site's valid REAL_TIME period(s) out of the site summary CSV
    produced by build_site_summary_table.py.

    The valid_periods column is a printed repr of a list of
    (Timestamp, Timestamp) tuples, not valid CSV or JSON, so this parses
    it by pulling out timestamp-shaped substrings in order and pairing
    them up -- the same approach already validated against the real
    output earlier in this thread. Bounds are in UTC (the same
    utc_reading_datetime values build_site_summary_table.py used to find
    them), not yet converted to the fixed-offset AEST index.
    """
    import re
    df = pd.read_csv(summary_csv_path)
    rows = df[df['site_id'] == site_id]
    if len(rows) == 0:
        raise ValueError(f"{site_id!r} not found in {summary_csv_path}")
    row = rows.iloc[0]
    if not row['include']:
        raise ValueError(
            f"{site_id!r} is marked include=False in {summary_csv_path} "
            f"(exclude_reason: {row.get('exclude_reason')!r}). "
            "Not meant to be ingested."
        )
    ts_strs = re.findall(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", str(row['valid_periods']))
    if not ts_strs or len(ts_strs) % 2 != 0:
        raise ValueError(
            f"Could not parse valid_periods for {site_id!r}: {row['valid_periods']!r}"
        )
    timestamps = [pd.Timestamp(t) for t in ts_strs]
    return list(zip(timestamps[0::2], timestamps[1::2]))


# ---------------------------------------------------------------------------
# Raw telemetry loading and shaping
# ---------------------------------------------------------------------------

def _load_full_telemetry(telemetry_path: Path = DEFAULT_TELEMETRY_PATH) -> pd.DataFrame:
    df = pd.read_parquet(telemetry_path)
    df['utc_reading_datetime'] = pd.to_datetime(df['utc_reading_datetime'])
    return df


def _site_rows(df: pd.DataFrame, site_id: str) -> pd.DataFrame:
    sub = df[df['em_id'] == site_id]
    if sub.empty:
        raise ValueError(f"No rows found for site_id={site_id!r}")
    return sub


def _matches_family(appliance_label: str, family_prefix: str) -> bool:
    """True if appliance_label is exactly family_prefix or family_prefix__N.
    Deliberately not a loose startswith: 'load_hot_water' is a literal
    string prefix of 'load_hot_water_heat_pump', so a loose match would
    silently merge two different appliance families. Not currently a risk
    for the three families actually used in this file (pv_dc,
    battery_storage, battery_dc don't collide with anything), but this
    keeps the helper correct if it's ever reused for a family with a
    colliding name."""
    return appliance_label == family_prefix or appliance_label.startswith(family_prefix + '__')


def battery_family_for(sub: pd.DataFrame) -> Optional[str]:
    """Which battery channel type this site uses, if any. Sites with both
    (only EM-339 in the cohort as checked) prefer battery_storage, matching
    the convention used throughout the site-selection scripts."""
    labels = set(sub['appliance_label'].unique())
    if any(_matches_family(l, 'battery_storage') for l in labels):
        return 'battery_storage'
    if any(_matches_family(l, 'battery_dc') for l in labels):
        return 'battery_dc'
    return None


def _family_sum(sub: pd.DataFrame, value_col: str, family_prefix: str) -> pd.Series:
    """Sum a value column across every appliance_label instance of one
    family (e.g. all of pv_dc__1, pv_dc__15 for a dual-system site),
    indexed by utc_reading_datetime. Empty Series (not an error) if the
    family isn't present at this site -- callers decide whether that's
    fatal."""
    mask = sub['appliance_label'].apply(lambda l: _matches_family(l, family_prefix))
    family_rows = sub[mask]
    if family_rows.empty:
        return pd.Series(dtype=float)
    wide = family_rows.pivot_table(
        index='utc_reading_datetime', columns='appliance_label',
        values=value_col, aggfunc='first',
    )
    return wide.sum(axis=1, skipna=True)


def _single_channel(sub: pd.DataFrame, appliance_label: str, value_col: str) -> pd.Series:
    """Value column for a channel that never has multiple instances at one
    site (ac_load_net)."""
    rows = sub[sub['appliance_label'] == appliance_label]
    if rows.empty:
        raise ValueError(f"Expected exactly one {appliance_label!r} channel, found none")
    return rows.set_index('utc_reading_datetime')[value_col]


def _to_datetime_aest(utc_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """UTC -> fixed-offset 'AEST' (Etc/GMT-10, no DST), matching the EDP
    pipeline's datetime_aest convention exactly (see
    L_Convert_To_Parquet.py and amber_site_ingest_design.md's timezone
    section). Deliberately NOT built from site_reading_datetime, which
    carries real local DST and would conflict with how the rest of the
    model resolves local time via core/localtime.py + an explicit state
    parameter."""
    return (utc_index + pd.Timedelta(hours=10)).tz_localize('Etc/GMT-10')


def _full_grid(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq='5min')


def _reindex_interpolate(
    series: pd.Series,
    full_index: pd.DatetimeIndex,
    plausible_max_abs: Optional[float] = None,
) -> pd.Series:
    """Reindex onto a complete 5-min grid (introducing NaN for any
    timestamp with no row at all, not just an existing-but-null value),
    then plain linear interpolation, filling any leading/trailing gap by
    extending from the nearest available value.

    This is the "basic interpolation, no flagging, no gap-length cap"
    policy from amber_site_ingest_design.md's missing-data section --
    deliberately simpler than the EDP pipeline's flag-and-two-method
    infill, with that as the documented upgrade path if this data proves
    out.

    If plausible_max_abs is given, any value whose absolute value exceeds
    it is treated as missing (masked to NaN) before interpolation -- a
    present-but-implausible raw value doesn't otherwise get touched by
    interpolation at all, since that only fills genuine gaps. Only pass
    this for energy-flow columns (Wh); never for price/rate columns,
    where a large value can be a genuine, important price spike.
    """
    reindexed = series.reindex(full_index)
    if plausible_max_abs is not None:
        implausible = reindexed.abs() > plausible_max_abs
        if implausible.any():
            reindexed = reindexed.mask(implausible)
    return reindexed.interpolate(method='linear', limit_direction='both')


def _check_no_tariff_misconfigured(sub: pd.DataFrame, valid_periods, site_id: str) -> None:
    """Defensive check per amber_site_ingest_design.md: a
    TARIFF_MISCONFIGURED row inside what's supposed to be a REAL_TIME-only
    valid period means something is wrong with the valid-period
    definition, not something to silently proceed past."""
    ac = sub[sub['appliance_label'] == 'ac_load_net']
    mask = pd.Series(False, index=ac.index)
    for s, e in valid_periods:
        mask |= (ac['utc_reading_datetime'] >= s) & (ac['utc_reading_datetime'] <= e)
    bad = ac[mask & (ac['rate_type'] == 'TARIFF_MISCONFIGURED')]
    if len(bad) > 0:
        raise ValueError(
            f"{site_id}: {len(bad)} TARIFF_MISCONFIGURED rows found inside "
            "the supposedly REAL_TIME-only valid period(s). The valid-period "
            "definition needs to be rechecked before ingesting this site."
        )


# ---------------------------------------------------------------------------
# LP / self-consumption feed
# ---------------------------------------------------------------------------

def load_em_site_trace(
    site_id: str,
    valid_periods: list[tuple[pd.Timestamp, pd.Timestamp]],
    telemetry_path: Path = DEFAULT_TELEMETRY_PATH,
    raw_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build the (pv_site_net_kwh, gross_load_kwh) DataFrame for one site,
    restricted to its valid REAL_TIME period(s), to feed self_consumption
    and perfect_foresight_lp exactly like _dev_common.load_site_trace does
    for EDP sites.

    gross_load_kwh = ac_load_net_kwh + pv_dc_kwh - battery_kwh, with
    battery_kwh from whichever channel (battery_storage or battery_dc)
    this site uses. Sign convention (ac_load_net: +import/-export;
    battery: +charge/-discharge) validated earlier across all 23 battery
    sites via a correlation-based check, not just asserted here.
    """
    df = raw_df if raw_df is not None else _load_full_telemetry(telemetry_path)
    sub = _site_rows(df, site_id)
    _check_no_tariff_misconfigured(sub, valid_periods, site_id)

    battery_family = battery_family_for(sub)
    if battery_family is None:
        raise ValueError(f"{site_id}: no battery channel present, cannot ingest")

    ac_load_net_wh = _single_channel(sub, 'ac_load_net', 'energy')
    pv_dc_wh = _family_sum(sub, 'energy', 'pv_dc')
    battery_wh = _family_sum(sub, 'energy', battery_family)

    frames = []
    for start, end in valid_periods:
        full_index = _full_grid(start, end)
        ac = _reindex_interpolate(ac_load_net_wh, full_index, MAX_PLAUSIBLE_WH_PER_INTERVAL)
        pv = _reindex_interpolate(pv_dc_wh, full_index, MAX_PLAUSIBLE_WH_PER_INTERVAL)
        bat = _reindex_interpolate(battery_wh, full_index, MAX_PLAUSIBLE_WH_PER_INTERVAL)

        gross_load_kwh = (ac + pv - bat) / 1000.0
        pv_site_net_kwh = pv / 1000.0

        period_df = pd.DataFrame({
            'pv_site_net_kwh': pv_site_net_kwh,
            'gross_load_kwh': gross_load_kwh,
        }, index=full_index)
        frames.append(period_df)

    out = pd.concat(frames)
    out.index = _to_datetime_aest(out.index)
    out.index.name = 'datetime_aest'
    return out


def load_em_site_component_flows(
    site_id: str,
    valid_periods: list[tuple[pd.Timestamp, pd.Timestamp]],
    telemetry_path: Path = DEFAULT_TELEMETRY_PATH,
    raw_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """The three separately-measured channels (solar, net grid, battery)
    for one site, restricted to its valid REAL_TIME period(s), for
    feeding reconstruct_flows.py.

    This is deliberately a sibling of load_em_site_trace, not a
    refactoring of it -- it applies the exact same per-channel masking
    and interpolation (same _single_channel/_family_sum +
    _reindex_interpolate calls, same MAX_PLAUSIBLE_WH_PER_INTERVAL), but
    returns the three components separately instead of combining them
    into gross_load_kwh. This matters for reconstruct_flows.py's
    bat_to_grid_a/bat_to_grid_b reconciliation check: that check is only meaningful if it uses the
    IDENTICAL already-cleaned series that gross_load was itself built
    from, not a re-derived copy -- summing these three columns here
    reproduces load_em_site_trace's gross_load_kwh exactly.

    Returns a DataFrame indexed by datetime_aest with columns:
        pv_dc_kwh:       solar, sign as measured (can be negative --
                         parasitic inverter draw overnight).
        ac_load_net_kwh: net grid flow, +import / -export.
        battery_kwh:     battery flow, +charge / -discharge.
    """
    df = raw_df if raw_df is not None else _load_full_telemetry(telemetry_path)
    sub = _site_rows(df, site_id)
    _check_no_tariff_misconfigured(sub, valid_periods, site_id)

    battery_family = battery_family_for(sub)
    if battery_family is None:
        raise ValueError(f"{site_id}: no battery channel present, cannot ingest")

    ac_load_net_wh = _single_channel(sub, 'ac_load_net', 'energy')
    pv_dc_wh = _family_sum(sub, 'energy', 'pv_dc')
    battery_wh = _family_sum(sub, 'energy', battery_family)

    frames = []
    for start, end in valid_periods:
        full_index = _full_grid(start, end)
        ac = _reindex_interpolate(ac_load_net_wh, full_index, MAX_PLAUSIBLE_WH_PER_INTERVAL)
        pv = _reindex_interpolate(pv_dc_wh, full_index, MAX_PLAUSIBLE_WH_PER_INTERVAL)
        bat = _reindex_interpolate(battery_wh, full_index, MAX_PLAUSIBLE_WH_PER_INTERVAL)

        period_df = pd.DataFrame({
            'pv_dc_kwh': pv / 1000.0,
            'ac_load_net_kwh': ac / 1000.0,
            'battery_kwh': bat / 1000.0,
        }, index=full_index)
        frames.append(period_df)

    out = pd.concat(frames)
    out.index = _to_datetime_aest(out.index)
    out.index.name = 'datetime_aest'
    return out


# ---------------------------------------------------------------------------
# Actual-measured SiteTraces, for billing the real outcome
# ---------------------------------------------------------------------------

def build_em_actual_traces(
    site_id: str,
    valid_periods: list[tuple[pd.Timestamp, pd.Timestamp]],
    telemetry_path: Path = DEFAULT_TELEMETRY_PATH,
    raw_df: Optional[pd.DataFrame] = None,
) -> SiteTraces:
    """Build a SiteTraces from EM-427-style actual measured behaviour.

    Only import_kwh and export_kwh are real; compute_bill was confirmed
    (by reading billing.py's _build_settlement_base) to never read the
    other eight fields, so they are genuine all-zero series here, not an
    approximation -- matching the existing convention in site_result.py
    for solar-only cohorts.

    import_kwh / export_kwh come directly from ac_load_net's own
    energy_pos / energy_neg decomposition (verified earlier: energy ==
    energy_pos - energy_neg holds for every row in the file, zero
    violations), not recomputed via max(ac_load_net, 0) -- one less place
    for the sign convention to be assumed twice.
    """
    df = raw_df if raw_df is not None else _load_full_telemetry(telemetry_path)
    sub = _site_rows(df, site_id)
    _check_no_tariff_misconfigured(sub, valid_periods, site_id)

    ac_rows = sub[sub['appliance_label'] == 'ac_load_net'].set_index('utc_reading_datetime')
    energy_pos_wh = ac_rows['energy_pos']
    energy_neg_wh = ac_rows['energy_neg']

    frames = []
    for start, end in valid_periods:
        full_index = _full_grid(start, end)
        pos = _reindex_interpolate(energy_pos_wh, full_index, MAX_PLAUSIBLE_WH_PER_INTERVAL)
        neg = _reindex_interpolate(energy_neg_wh, full_index, MAX_PLAUSIBLE_WH_PER_INTERVAL)
        frames.append(pd.DataFrame({'import_wh': pos, 'export_wh': neg}, index=full_index))

    flows = pd.concat(frames)
    flows.index = _to_datetime_aest(flows.index)
    flows.index.name = 'datetime_aest'

    import_kwh = flows['import_wh'] / 1000.0
    export_kwh = flows['export_wh'] / 1000.0
    zeros = pd.Series(0.0, index=flows.index)

    return SiteTraces(
        site_id=site_id,
        import_kwh=import_kwh,
        export_kwh=export_kwh,
        battery_soc_kwh=zeros,
        sol_to_home_kwh=zeros,
        sol_to_bat_kwh=zeros,
        sol_to_grid_kwh=zeros,
        bat_to_home_kwh=zeros,
        grid_to_home_kwh=zeros,
        grid_to_bat_kwh=zeros,
        bat_to_grid_kwh=zeros,
        curtail_kwh=zeros,
    )


# ---------------------------------------------------------------------------
# Price reconciliation
# ---------------------------------------------------------------------------

def _load_nem_prices_multi_year(
    state: str, years: list[int], paths: Paths,
) -> pd.Series:
    """Load and concatenate NEM RRP across however many calendar years the
    valid period spans (EM-427's does: 2025-09 through 2026-03). Mirrors
    viz_perfect_foresight_lp.py's _load_prices_mwh, generalised to
    multiple years since that function assumes a single-year scenario."""
    pieces = []
    for year in years:
        price_path = paths.nem_prices_dir / f'{year}_nem_prices.parquet'
        if not price_path.exists():
            raise FileNotFoundError(f"NEM price file not found: {price_path}")
        df = pd.read_parquet(price_path)
        rrp_col = f'{state}1_rrp'
        if rrp_col not in df.columns:
            raise ValueError(
                f"NEM price file {price_path} missing column {rrp_col!r}. "
                f"Available: {list(df.columns)}"
            )
        idx = pd.to_datetime(df['datetime_aest']).dt.tz_convert('Etc/GMT-10')
        pieces.append(pd.Series(df[rrp_col].to_numpy(), index=idx))
    series = pd.concat(pieces).sort_index()
    series = series[~series.index.duplicated(keep='first')]
    series.name = f'{state}_rrp_dollars_per_mwh'
    series.index.name = 'datetime_aest'
    return series


def block_average(series: pd.Series, phase_offset_minutes: int = SETTLEMENT_PHASE_OFFSET_MINUTES) -> pd.Series:
    """30-min block average of `series`, broadcast back onto the original
    5-min index. Content grouping only -- see amber_comparable_price for
    the additional broadcast shift needed to match where Amber actually
    displays the result."""
    shifted_index = series.index - pd.Timedelta(minutes=phase_offset_minutes)
    block_key = shifted_index.floor('30min')
    return series.groupby(block_key).transform('mean')


def amber_comparable_price(
    derived_5min: pd.Series,
    phase_offset_minutes: int = SETTLEMENT_PHASE_OFFSET_MINUTES,
    broadcast_shift_ticks: int = SETTLEMENT_BROADCAST_SHIFT_TICKS,
) -> pd.Series:
    """Transform a 5-min derived price into the form directly comparable
    to Amber's published 30-min-settled rate. Confirmed against EM-427's
    full valid period and unanimous across the clean-14 settlement
    check -- see amber_site_ingest_design.md's "Settlement resolution"
    section. Only meaningful for a site confirmed 30-min settled; check
    with check_settlement_resolution first for any site not yet checked.
    """
    return block_average(derived_5min, phase_offset_minutes).shift(broadcast_shift_ticks)


def check_settlement_resolution(
    site_id: str,
    valid_periods: list[tuple[pd.Timestamp, pd.Timestamp]],
    telemetry_path: Path = DEFAULT_TELEMETRY_PATH,
    raw_df: Optional[pd.DataFrame] = None,
) -> dict:
    """Dominant block width and transition tick-position for a site's
    import_rate/export_rate, restricted to its valid period(s). >80% of
    blocks at exactly 6 ticks (30 min) wide indicates 30-min settlement
    (the block-average correction in amber_comparable_price applies);
    <20% indicates genuine 5-min settlement (use the raw derived price
    directly for reconciliation instead). Confirmed 30-min-settled for
    all 14 sites in the clean shortlist as of the last check -- still
    worth running for any new site added later, since this is a per-site
    meter/config property that happened to be uniform across this cohort,
    not a guaranteed universal constant.
    """
    df = raw_df if raw_df is not None else _load_full_telemetry(telemetry_path)
    sub = _site_rows(df, site_id)
    ac = sub[sub['appliance_label'] == 'ac_load_net'].sort_values('utc_reading_datetime')

    mask = pd.Series(False, index=ac.index)
    for start, end in valid_periods:
        mask |= (ac['utc_reading_datetime'] >= start) & (ac['utc_reading_datetime'] <= end)
    ac = ac[mask]

    def _summarize(series: pd.Series) -> dict:
        s = series.dropna()
        if len(s) < 2:
            return {'pct_6tick_blocks': None, 'dominant_tick_pos': None}
        changed = s.diff().fillna(0) != 0
        change_times = s.index[changed]
        if len(change_times) < 2:
            return {'pct_6tick_blocks': None, 'dominant_tick_pos': None}
        gaps_minutes = change_times.to_series().diff().dropna().dt.total_seconds() / 60
        gaps_ticks = (gaps_minutes / 5).round().astype(int)
        width_counts = gaps_ticks.value_counts()
        pct_6tick = 100 * width_counts.get(6, 0) / width_counts.sum()
        tick_position = change_times.minute % 30
        pos_counts = tick_position.value_counts()
        return {
            'pct_6tick_blocks': round(pct_6tick, 1),
            'dominant_tick_pos': int(pos_counts.idxmax()),
        }

    import_summary = _summarize(ac.set_index('utc_reading_datetime')['import_rate'])
    export_summary = _summarize(ac.set_index('utc_reading_datetime')['export_rate'])
    pct = import_summary['pct_6tick_blocks']
    if pct is None:
        regime = 'insufficient data'
    elif pct >= 80:
        regime = '30-min settled'
    elif pct <= 20:
        regime = '5-min settled'
    else:
        regime = 'ambiguous'
    return {'import': import_summary, 'export': export_summary, 'regime': regime}


def reconcile_prices(
    site_id: str,
    valid_periods: list[tuple[pd.Timestamp, pd.Timestamp]],
    network_tariff_id: str = DEFAULT_NETWORK_TARIFF_ID,
    retail_tariff_id: str = DEFAULT_RETAIL_TARIFF_ID,
    paths: Optional[Paths] = None,
    telemetry_path: Path = DEFAULT_TELEMETRY_PATH,
    raw_df: Optional[pd.DataFrame] = None,
    apply_30min_correction: bool = True,
) -> pd.DataFrame:
    """Compare our derived (p_imp, p_exp) against Amber's own published
    import_rate / export_rate for the same intervals. Returns a DataFrame
    with the raw 5-min comparison AND (when apply_30min_correction is
    True, the default) the 30-min-settlement-corrected comparison, so a
    caller can see both. run_checks below prints the summary of both.

    apply_30min_correction defaults to True since this is confirmed for
    all 14 clean-shortlist sites (see check_settlement_resolution) -- set
    False for a site confirmed 5-min settled instead, or if you just want
    the uncorrected raw comparison for some other reason.

    Some divergence is expected even after correction (our network_tariff/
    retail_tariff are our best approximation of SA_SAPN_RTOU_2026 /
    SA_amber_wholesale_2026, not necessarily an exact match to what Amber
    actually charged this specific site) -- the point is confirming the
    two are in the right ballpark, not forcing exact agreement.
    """
    paths = paths or default_paths()
    df = raw_df if raw_df is not None else _load_full_telemetry(telemetry_path)
    sub = _site_rows(df, site_id)

    ac_rows = sub[sub['appliance_label'] == 'ac_load_net'].set_index('utc_reading_datetime')
    import_rate = ac_rows['import_rate']
    export_rate = ac_rows['export_rate']

    network_tariff = load_network_tariff(network_tariff_id, paths.tariffs_dir)
    retail_tariff = load_retail_tariff(retail_tariff_id, paths.tariffs_dir)

    years = sorted({s.year for s, _ in valid_periods} | {e.year for _, e in valid_periods})
    prices_mwh_full = _load_nem_prices_multi_year(STATE, years, paths)

    frames = []
    for start, end in valid_periods:
        full_index = _full_grid(start, end)
        rate_imp = _reindex_interpolate(import_rate, full_index)
        rate_exp = _reindex_interpolate(export_rate, full_index)

        aest_index = _to_datetime_aest(full_index)
        prices_mwh = prices_mwh_full.reindex(aest_index)
        if prices_mwh.isna().any():
            n_missing = int(prices_mwh.isna().sum())
            raise ValueError(
                f"{site_id}: {n_missing} NEM price intervals missing over "
                f"{start} to {end}. build_price_vectors will reject NaN "
                "prices; check the NEM price file(s) cover this span."
            )

        p_imp, p_exp = build_price_vectors(
            aest_index, prices_mwh, network_tariff, retail_tariff, STATE, INTERVAL_HOURS,
        )
        p_imp = pd.Series(p_imp.to_numpy(), index=aest_index)
        p_exp = pd.Series(p_exp.to_numpy(), index=aest_index)

        period_df = pd.DataFrame({
            'p_imp_derived': p_imp,
            'import_rate_amber': rate_imp.to_numpy() if len(rate_imp) else np.nan,
            'p_exp_derived': p_exp,
            'export_rate_amber': rate_exp.to_numpy() if len(rate_exp) else np.nan,
        }, index=aest_index)

        if apply_30min_correction:
            period_df['p_imp_block_avg'] = amber_comparable_price(p_imp)
            period_df['p_exp_block_avg'] = amber_comparable_price(p_exp)

        frames.append(period_df)

    out = pd.concat(frames)
    out.index.name = 'datetime_aest'
    out['import_residual'] = out['p_imp_derived'] - out['import_rate_amber']
    out['export_residual'] = out['p_exp_derived'] - out['export_rate_amber']
    if apply_30min_correction:
        out['import_residual_corrected'] = out['p_imp_block_avg'] - out['import_rate_amber']
        out['export_residual_corrected'] = out['p_exp_block_avg'] - out['export_rate_amber']
    return out


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def run_checks(
    site_id: str = 'EM-427',
    summary_csv_path: Path = DEFAULT_SITE_SUMMARY_PATH,
    telemetry_path: Path = DEFAULT_TELEMETRY_PATH,
    network_tariff_id: str = DEFAULT_NETWORK_TARIFF_ID,
    retail_tariff_id: str = DEFAULT_RETAIL_TARIFF_ID,
) -> None:
    """End-to-end sense-check: load, build both outputs, reconcile prices,
    print a verification report. Emits 'FAIL' tokens on any check failure,
    matching the convention viz_perfect_foresight_lp.py's own
    _print_verification uses.
    """
    print(f"=== {site_id} ingestion check ===")

    valid_periods = load_valid_periods_from_summary(site_id, summary_csv_path)
    print(f"Valid period(s): {valid_periods}")

    print("\nLoading raw telemetry (single pass, shared across all steps below)...")
    raw_df = _load_full_telemetry(telemetry_path)

    print("\n--- LP feed (load_em_site_trace) ---")
    trace = load_em_site_trace(site_id, valid_periods, raw_df=raw_df)
    print(f"Rows: {len(trace)}")
    print(trace.describe().to_string())

    # Plausibility guard: 5 kWh in a single 5-min interval implies 60kW,
    # an extremely generous upper bound for any site in this cohort. This
    # would have caught EM-414's family-summing outlier automatically
    # rather than relying on eyeballing the describe() output above.
    MAX_PLAUSIBLE_KWH_PER_INTERVAL = 5.0
    for col in ['pv_site_net_kwh', 'gross_load_kwh']:
        col_max = trace[col].abs().max()
        if col_max > MAX_PLAUSIBLE_KWH_PER_INTERVAL:
            worst_ts = trace[col].abs().idxmax()
            print(f"FAIL: {col} has an implausible single-interval value: "
                  f"{trace.loc[worst_ts, col]:.3f} kWh at {worst_ts} "
                  f"(> {MAX_PLAUSIBLE_KWH_PER_INTERVAL} kWh threshold, implying "
                  f">{MAX_PLAUSIBLE_KWH_PER_INTERVAL * 12:.0f} kW -- needs investigation "
                  "before this site's ingestion is trusted)")

    pct_negative = (trace['gross_load_kwh'] < 0).mean() * 100
    summary_row = pd.read_csv(summary_csv_path)
    summary_row = summary_row[summary_row['site_id'] == site_id].iloc[0]
    expected_pct_negative = summary_row['pct_negative_in_valid_period']
    diff = abs(pct_negative - expected_pct_negative)
    # Small differences are expected, not a bug: build_site_summary_table.py
    # computed this from raw (un-interpolated) data, while load_em_site_trace
    # interpolates missing values first -- a wider gap than that would
    # suggest something else changed, not just the interpolation.
    print(f"\npct_negative gross_load: {pct_negative:.2f}%  "
          f"(site summary table's own figure: {expected_pct_negative:.2f}%; "
          f"{'FAIL -- diverges more than interpolation alone should explain' if diff > 0.5 else 'OK'})")

    print("\n--- Actual traces (build_em_actual_traces) ---")
    traces = build_em_actual_traces(site_id, valid_periods, raw_df=raw_df)
    print(f"Rows: {len(traces.import_kwh)}")
    print(f"Total import: {traces.import_kwh.sum():.1f} kWh")
    print(f"Total export: {traces.export_kwh.sum():.1f} kWh")

    # Consistency contract: import_kwh - export_kwh == ac_load_net_kwh,
    # recomputed independently from the raw signed energy column as a
    # cross-check (not from energy_pos/energy_neg, which is what
    # import_kwh/export_kwh were built from -- this is deliberately an
    # independent recomputation, not a tautology).
    sub = _site_rows(raw_df, site_id)
    ac_signed_wh = _single_channel(sub, 'ac_load_net', 'energy')
    check_frames = []
    for start, end in valid_periods:
        full_index = _full_grid(start, end)
        check_frames.append(_reindex_interpolate(ac_signed_wh, full_index, MAX_PLAUSIBLE_WH_PER_INTERVAL))
    ac_signed_kwh = pd.concat(check_frames) / 1000.0
    ac_signed_kwh.index = _to_datetime_aest(ac_signed_kwh.index)

    net_from_flows = (traces.import_kwh - traces.export_kwh)
    residual = (net_from_flows - ac_signed_kwh).abs()
    max_residual = residual.max()
    print(f"\nInvariant check: (import_kwh - export_kwh) == ac_load_net_kwh")
    print(f"  max absolute residual: {max_residual:.6f} kWh  "
          f"({'FAIL' if max_residual > 1e-6 else 'OK'})")

    print("\n--- Settlement resolution check ---")
    settlement = check_settlement_resolution(site_id, valid_periods, raw_df=raw_df)
    print(f"Regime: {settlement['regime']}  "
          f"(import: {settlement['import']['pct_6tick_blocks']}% 6-tick blocks, "
          f"export: {settlement['export']['pct_6tick_blocks']}% 6-tick blocks)")
    apply_correction = settlement['regime'] == '30-min settled'
    if settlement['regime'] == 'ambiguous':
        print("AMBIGUOUS regime -- applying the 30-min correction anyway since that's been "
              "the pattern for every site checked so far, but this site's result should be "
              "looked at more closely rather than trusted blindly.")
        apply_correction = True
    elif settlement['regime'] == 'insufficient data':
        print("Insufficient data to determine regime -- skipping the 30-min correction.")

    print("\n--- Price reconciliation ---")
    price_check = reconcile_prices(
        site_id, valid_periods, network_tariff_id, retail_tariff_id, raw_df=raw_df,
        apply_30min_correction=apply_correction,
    )
    print("Raw 5-min comparison (expected to look noisy if this site is 30-min settled):")
    print(price_check[['import_residual', 'export_residual']].describe().to_string())
    if apply_correction:
        print("\n30-min-corrected comparison (the one that should actually be trusted "
              "for a 30-min-settled site):")
        print(price_check[['import_residual_corrected', 'export_residual_corrected']]
              .describe().to_string())

    print("\n=== Done ===")


if __name__ == '__main__':
    site = sys.argv[1] if len(sys.argv) > 1 else 'EM-427'
    run_checks(site)