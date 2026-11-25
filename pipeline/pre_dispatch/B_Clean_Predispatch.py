"""Stage B: clean raw AEMO pre-dispatch frames into the canonical store.

Input is what stage A cached: raw MMSDM frames with all 119 (P5MIN) or 36
(PREDISPATCH) columns. Output is the canonical store from
predispatch_data_spec.md:

    data/clean/predispatch/{table}/{REGION}/{YYYYMM}.parquet
    columns: run_time (datetime64[us, Etc/GMT-10]), lead_minutes (int16),
             rrp_mwh (float64)

Three of the four traps in the spec are handled here. The fourth, the shifted
monthly partition clock, belongs to stage A because it is about which SOURCE
file a run lives in.

Every failure mode below produces a store that LOADS CLEANLY AND IS WRONG,
which is why each is a hard error rather than a warning.

Why target_time is not stored
-----------------------------
target_time = run_time + lead_minutes exactly, for both tables: P5MIN targets
sit on the 5-minute grid from the run label and PREDISPATCH targets at
run_time + 30 * (PERIODID - 1). Storing the lead is more compact and is the
natural key for error-by-lead analysis, which is the main downstream use
besides the model itself.

Why float64
-----------
These prices flow into money calculations and the affine reconstruction test
in prices.py asserts exactness. The compressed size difference does not
justify the doubt.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MARKET_TZ = 'Etc/GMT-10'
CANONICAL_COLUMNS = ['run_time', 'lead_minutes', 'rrp_mwh']

# Filename stem nemweb_direct uses for each table's raw monthly cache, so this
# stage can rebuild the store from cache without touching the network. Mirrors
# nemweb_direct._token: PREDISPATCH tables are prefixed rather than joined with
# an underscore.
RAW_TOKEN = {
    'p5min': 'P5MIN_REGIONSOLUTION',
    'predispatch': 'PREDISPATCHPRICE',
}

# Per-table source column names and target spacing. PREDISPATCH has no run
# time column in the source at all; stage A derives PREDISPATCH_RUN_DATETIME
# from PREDISPATCHSEQNO arithmetically.
TABLE_SPEC = {
    'p5min': dict(run_col='RUN_DATETIME', target_col='INTERVAL_DATETIME',
                  resolution_minutes=5, run_cadence_minutes=5),
    'predispatch': dict(run_col='PREDISPATCH_RUN_DATETIME', target_col='DATETIME',
                        resolution_minutes=30, run_cadence_minutes=30),
}


def clean_forecast_frame(raw: pd.DataFrame, table: str) -> pd.DataFrame:
    """Clean one raw frame into canonical form, keeping the region column.

    Returns [run_time, region, lead_minutes, rrp_mwh]; region is dropped on
    write, where it becomes the partition.

    Raises:
        ValueError: on a missing required column, a surviving duplicate, a
            negative or misaligned lead, or a null price.
    """
    if table not in TABLE_SPEC:
        raise ValueError(f"unknown table {table!r}, expected one of {list(TABLE_SPEC)}")
    spec = TABLE_SPEC[table]
    run_col, target_col = spec['run_col'], spec['target_col']

    # Explicit column selection. Schema drift across 2021 to 2026 is ADDITIVE
    # only (18 columns added to P5MIN, 2 to PREDISPATCH, none removed or
    # renamed), so naming the columns is exactly what makes a uniform loader
    # safe across the whole period.
    required = [run_col, target_col, 'REGIONID', 'RRP', 'INTERVENTION']
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(
            f"{table}: missing required column(s) {missing}. Present: "
            f"{sorted(raw.columns)[:12]}{'...' if len(raw.columns) > 12 else ''}"
        )
    df = raw.loc[:, required].copy()

    # TRAP 1: INTERVENTION. Filtering to case 0 is both deduplicating and
    # lossless: every intervention interval carries BOTH cases, verified on
    # June 2022 across 124,455 duplicate keys with zero missing counterparts.
    # Case 0 is the counterfactual pricing run and matches settled prices
    # exactly (MAE $0.00); case 1 is the physical dispatch run's shadow prices
    # and does not (MAE $89.71). The two differ on 40% of intervention
    # intervals by a mean of $1,143/MWh, which is precisely where battery
    # arbitrage value concentrates, so the choice is not cosmetic.
    n_before = len(df)
    df = df[df['INTERVENTION'] == 0].drop(columns='INTERVENTION')
    n_intervention = n_before - len(df)

    df = df.rename(columns={run_col: 'run_time', target_col: 'target_time',
                            'REGIONID': 'region', 'RRP': 'rrp_mwh'})
    df['run_time'] = pd.to_datetime(df['run_time'])
    df['target_time'] = pd.to_datetime(df['target_time'])
    df['region'] = df['region'].astype(str)

    dup = df.duplicated(subset=['run_time', 'region', 'target_time'], keep=False)
    if dup.any():
        raise ValueError(
            f"{table}: {int(dup.sum())} row(s) duplicated on "
            "(run_time, region, target_time) AFTER the INTERVENTION == 0 "
            "filter. Exactly duplicated rows are dropped upstream, so these "
            "differ in value and need a real tie-break rule rather than a "
            f"silent drop_duplicates.\n{df[dup].head(4)}"
        )

    delta = (df['target_time'] - df['run_time']).dt.total_seconds()
    if (delta < 0).any():
        raise ValueError(
            f"{table}: {int((delta < 0).sum())} row(s) have a target before "
            "their run time, which means the run time derivation is wrong. "
            "For PREDISPATCH that points at the PREDISPATCHSEQNO arithmetic."
        )
    res_s = spec['resolution_minutes'] * 60
    if (delta % res_s != 0).any():
        raise ValueError(
            f"{table}: {int((delta % res_s != 0).sum())} row(s) have a lead "
            f"that is not a whole multiple of {spec['resolution_minutes']} "
            "minutes, so target_time could not be reconstructed from "
            "run_time + lead_minutes."
        )
    lead = delta / 60
    if lead.max() > np.iinfo(np.int16).max:
        raise ValueError(f"{table}: lead of {lead.max():.0f} min overflows int16")

    # TRAP 2: dtypes are data-dependent per month. nemseer downcasts float64
    # first to integer where integral, then to float32 where it round-trips,
    # so the SAME column changes dtype between adjacent months of the same
    # table. Everything is cast explicitly; nothing inherits.
    out = pd.DataFrame({
        'run_time': df['run_time'].dt.tz_localize(MARKET_TZ),
        'region': df['region'],
        'lead_minutes': lead.to_numpy().astype(np.int16),
        'rrp_mwh': df['rrp_mwh'].to_numpy().astype(np.float64),
    })

    if out['rrp_mwh'].isna().any():
        raise ValueError(
            f"{table}: {int(out['rrp_mwh'].isna().sum())} null price(s). RRP "
            "had zero nulls across every month checked in the investigation, "
            "so this is a defect rather than something to fill."
        )

    out.attrs['n_intervention_rows_dropped'] = n_intervention
    return out.sort_values(['region', 'run_time', 'lead_minutes'],
                           ignore_index=True)


def partition_path(store_root: Path, table: str, region: str,
                   period: pd.Period) -> Path:
    return Path(store_root) / table / region / f"{period.strftime('%Y%m')}.parquet"


def write_partitions(clean: pd.DataFrame, table: str, store_root: Path,
                     merge: bool = False) -> list[Path]:
    """Write a clean frame out as region/month parquet files.

    Partitioned on the CALENDAR month of run_time, deliberately NOT on AEMO's
    shifted trading-day clock. The shifted clock is a property of the source
    files and is stage A's problem; a reader of this store should be able to
    assume 2026-02.parquet holds February run times and nothing else. Since one
    source file spills across a month boundary, a single source month can write
    into two partitions, which is why merge exists.

    Args:
        merge: Combine with anything already at the path, keeping the newer row
            per (run_time, lead_minutes). Needed when consecutive source months
            both contribute to one partition; leave False for a clean rebuild.
    """
    written = []
    frame = clean.copy()
    # tz_localize(None) first: to_period would otherwise drop the timezone with
    # a warning. The instants are already market time, so dropping the offset
    # after converting is exact, not a reinterpretation.
    frame['_period'] = (frame['run_time'].dt.tz_convert(MARKET_TZ)
                        .dt.tz_localize(None).dt.to_period('M'))
    for (region, period), grp in frame.groupby(['region', '_period'], observed=True):
        path = partition_path(store_root, table, region, period)
        path.parent.mkdir(parents=True, exist_ok=True)
        part = grp[CANONICAL_COLUMNS].reset_index(drop=True)
        if merge and path.exists():
            part = (pd.concat([pd.read_parquet(path), part], ignore_index=True)
                    .drop_duplicates(subset=['run_time', 'lead_minutes'],
                                     keep='last')
                    .sort_values(['run_time', 'lead_minutes'], ignore_index=True))
        part.to_parquet(path, index=False)
        written.append(path)
    return sorted(written)


def read_store(store_root: Path, table: str, region: str,
               start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Read the canonical store for one region over a run-time range.

    The provider's entry point. Months are selected on the calendar month of
    run_time, which is safe precisely because the store is partitioned that way
    regardless of how the source files were.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    frames = []
    for period in pd.period_range(start, end, freq='M'):
        path = partition_path(store_root, table, region, period)
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in
                             (('run_time', f'datetime64[us, {MARKET_TZ}]'),
                              ('lead_minutes', 'int16'),
                              ('rrp_mwh', 'float64'))})
    df = pd.concat(frames, ignore_index=True)
    keep = (df['run_time'] >= start) & (df['run_time'] <= end)
    return df[keep].sort_values(['run_time', 'lead_minutes'], ignore_index=True)


def clean_and_write(raw: pd.DataFrame, table: str, store_root: Path,
                    merge: bool = True) -> tuple[pd.DataFrame, list[Path]]:
    clean = clean_forecast_frame(raw, table)
    return clean, write_partitions(clean, table, store_root, merge=merge)


# ---------------------------------------------------------------------------
# CLI: rebuild the store from the raw cache
# ---------------------------------------------------------------------------

def _known_gaps() -> dict:
    """Stage C's known-archive-gap table, imported lazily.

    C imports partition_path, read_store etc. from this module at its own top
    level, so importing C from here at module scope would be circular. By the
    time this is actually called both modules are fully loaded, regardless of
    which was imported first.
    """
    from pipeline.pre_dispatch.C_Validate_Predispatch import KNOWN_ARCHIVE_GAPS
    return KNOWN_ARCHIVE_GAPS


def raw_cache_path(raw_cache: Path, table: str, period: pd.Period) -> Path:
    return Path(raw_cache) / f"{RAW_TOKEN[table]}_{period.strftime('%Y%m')}.parquet"


def clean_range(start: str, end: str,
                tables=('p5min', 'predispatch'),
                regions=('NSW1', 'QLD1', 'SA1', 'TAS1', 'VIC1'),
                raw_cache: Path = Path('data/raw/predispatch'),
                store_root: Path = Path('data/clean/predispatch'),
                verbose: bool = True) -> list[dict]:
    """Rebuild the canonical store from the raw cache, no network required.

    Separate from stage A on purpose. Downloading is slow and rarely needs
    repeating; cleaning rules change. Keeping this its own entry point means a
    change to the INTERVENTION handling, the dtype policy or the partitioning
    can be applied to five years of data in minutes from cache.

    Fails soft per table-month so one absent cache file does not lose the rest,
    but the cleaning itself fails hard: a cleaning error is a defect.
    """
    out = []
    for period in pd.period_range(start, end, freq='M'):
        for table in tables:
            rec = {'table': table, 'period': str(period), 'rows': 0,
                   'partitions': 0, 'dropped': 0, 'error': '', 'known_gap': False}
            frames, missing = [], []
            # Read the PREVIOUS month's source file as well. MMSDM partitions
            # on a shifted trading-day clock, so a calendar month's FIRST runs
            # live in the previous month's file: P5MIN file M starts at
            # M-01 00:05 and PREDISPATCH file M at M-01 04:30. Reading only
            # file M silently loses the 00:00 run for P5MIN and the nine runs
            # from 00:00 to 04:00 for PREDISPATCH, with no error raised. That
            # is exactly the trap nemweb_direct's month-shift exists to solve,
            # and reading source files directly here reintroduced it.
            for src in ((period - 1), period):
                path = raw_cache_path(raw_cache, table, src)
                if path.exists():
                    frames.append(pd.read_parquet(path))
                else:
                    missing.append(path.name)
            own_missing = raw_cache_path(raw_cache, table, period).name in missing
            is_known_gap = (table, str(period)) in _known_gaps()
            if is_known_gap and own_missing:
                rec['known_gap'] = True
                if not frames:
                    if verbose:
                        print(f"  [GAP]  {table:<12} {period}  known archive "
                              "gap, not acquired by stage A, and the previous "
                              "month's source file is absent too: "
                              f"{_known_gaps()[(table, str(period))][:90]}")
                    out.append(rec)
                    continue
                # The month's own source file does not exist, but its leading
                # runs (00:00 to 04:00 for PREDISPATCH) live in the previous
                # month's file and do. They are written as a short partition
                # so the month is present in the store, and therefore in the
                # manifest with known_archive_gap set, which is what the
                # forecast provider's on_degraded_period policy keys on.
                if verbose:
                    print(f"  [GAP]  {table:<12} {period}  known archive "
                          "gap; writing only the leading runs held in the "
                          f"previous month's source file. "
                          f"{_known_gaps()[(table, str(period))][:90]}")
            elif not frames or own_missing:
                rec['error'] = f"raw cache missing: {', '.join(missing)}"
                if verbose:
                    print(f"  [SKIP] {table:<12} {period}  {rec['error']}")
                out.append(rec)
                continue
            raw = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
            keep = raw['REGIONID'].astype(str).isin(regions)
            cleaned = clean_forecast_frame(raw[keep], table)
            # Keep only runs belonging to THIS calendar month, so the previous
            # month's partition is never rewritten from a partial read.
            in_month = (cleaned['run_time'].dt.tz_convert(MARKET_TZ)
                        .dt.tz_localize(None).dt.to_period('M') == period)
            cleaned = cleaned[in_month]
            if len(missing) and verbose and not rec['known_gap']:
                prev_is_known_gap = (
                    missing[0] == raw_cache_path(raw_cache, table, period - 1).name
                    and (table, str(period - 1)) in _known_gaps())
                if prev_is_known_gap:
                    print(f"  [NOTE] {table:<12} {period}  previous month "
                          f"({period - 1}) is a known archive gap, not an "
                          "unacquired one; this month's leading runs are "
                          "permanently short and re-running stage A will not "
                          "fill them.")
                else:
                    print(f"  [NOTE] {table:<12} {period}  previous month's "
                          f"source file absent ({missing[0]}); the first runs "
                          "of this month may be missing. Acquire one month "
                          "earlier to fill.")
            paths = write_partitions(cleaned, table, store_root, merge=True)
            rec['rows'] = len(cleaned)
            rec['partitions'] = len(paths)
            rec['dropped'] = int(cleaned.attrs.get('n_intervention_rows_dropped', 0))
            if verbose:
                # Intervention drops are reported because they are not
                # uniform: interventions are rare but heavily clustered, so a
                # month with a large count is a month whose prices deserve a
                # second look rather than a defect.
                print(f"  [OK]   {table:<12} {period}  {len(cleaned):>9,} rows "
                      f"-> {len(paths)} partition(s)"
                      + (f"   {rec['dropped']:,} intervention row(s) dropped"
                         if rec['dropped'] else ''), flush=True)
            out.append(rec)
    ok = [r for r in out if not r['error']]
    gaps = [r for r in ok if r['known_gap']]
    print(f"\nCleaned {len(ok) - len(gaps)}/{len(out) - len(gaps)} table-month(s), "
          f"{sum(r['rows'] for r in ok):,} rows, "
          f"{sum(r['dropped'] for r in ok):,} intervention row(s) dropped")
    if gaps:
        print(f"{len(gaps)} known archive gap(s), excluded from the month count "
              "above and not fixable by re-running stage A: "
              + ', '.join(f"{r['table']} {r['period']} ({r['rows']:,} rows "
                          "from the previous month's source file)"
                          for r in gaps))
    missing = [r for r in out if r['error']]
    if missing:
        print(f"{len(missing)} raw cache file(s) absent; run "
              "pipeline.pre_dispatch.A_Download_Predispatch first")
    print("\nNext: python3 -m pipeline.pre_dispatch.C_Validate_Predispatch")
    return out


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', required=True, help='YYYY-MM inclusive')
    parser.add_argument('--end', required=True, help='YYYY-MM inclusive')
    parser.add_argument('--tables', default='p5min,predispatch')
    parser.add_argument('--regions', default='NSW1,QLD1,SA1,TAS1,VIC1')
    parser.add_argument('--raw-cache', type=Path,
                        default=Path('data/raw/predispatch'))
    parser.add_argument('--store', type=Path,
                        default=Path('data/clean/predispatch'))
    args = parser.parse_args()
    clean_range(args.start, args.end,
                tables=tuple(t.strip() for t in args.tables.split(',')),
                regions=tuple(r.strip() for r in args.regions.split(',')),
                raw_cache=args.raw_cache, store_root=args.store)
