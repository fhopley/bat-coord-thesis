"""Stage C: validate the canonical pre-dispatch store and write the manifest.

Re-runnable independently of acquisition. Every check here exists because the
corresponding failure produces a store that loads cleanly and is wrong.

The closed form for PREDISPATCH target counts is load-bearing and is the
reason this module exists rather than a few asserts inside stage B. AEMO's
horizon sawtooths between 15.5 and 39 hours on a daily cycle, so any check that
compares a run's target count against a global median or mean will flag roughly
half of all runs. That artefact appeared in the raw investigation output and is
superseded here.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from model.core.market_prices import market_price_cap, market_price_floor
from pipeline.pre_dispatch.B_Clean_Predispatch import (
    MARKET_TZ, TABLE_SPEC, partition_path, read_store,
)

P5MIN_TARGETS = 12          # leads 0..55 min, constant, no exceptions observed
TRADING_DAY_START_HOUR = 4  # NEM trading day runs 04:00 to 04:00
BID_CLOSURE_HOUR = 13       # runs labelled 13:00 or later see one day further

# Runs at the start of a calendar month that live in the PREVIOUS month's
# source file, because MMSDM partitions on a trading-day clock. Absent whenever
# the previous month was not acquired, which is normal at the edge of a range.
# Mirrors _LEADING_RUNS_IN_PREVIOUS_FILE in stage A.
BOUNDARY_RUNS = {'p5min': 1, 'predispatch': 9}

# Months the archive does not hold and never will, so a re-run cannot fix them.
# Recorded here rather than left to look like an acquisition that has not
# happened yet: those two states are otherwise indistinguishable, and only one
# of them is worth retrying.
KNOWN_ARCHIVE_GAPS = {
    ('predispatch', '2022-10'):
        'PREDISPATCHPRICE is absent from MMSDM for this month in every form '
        '(all seven SQLLoader subfolders searched). P5MIN is complete for the '
        'month, so the fine segment is available and only the coarse forward '
        'window is missing. See predispatch_data_spec.md, trap 4.',
}


def predispatch_horizon_end(run_time: pd.Timestamp) -> pd.Timestamp:
    """End of a PREDISPATCH run's horizon, in market time.

    Validated against all 1,345 February 2026 runs at 100% match. A run sees to
    the end of the last trading day for which bid submission has closed, and
    closure happens at 12:30, so runs labelled 13:00 or later reach one trading
    day further than earlier ones.
    """
    base = run_time.normalize()
    days = 2 if run_time.hour >= BID_CLOSURE_HOUR else 1
    return base + pd.Timedelta(days=days, hours=TRADING_DAY_START_HOUR)


def expected_target_count(run_time: pd.Timestamp, table: str) -> int:
    if table == 'p5min':
        return P5MIN_TARGETS
    end = predispatch_horizon_end(run_time)
    return int((end - run_time).total_seconds() // (30 * 60)) + 1


@dataclass
class TableReport:
    """Validation outcome for one (table, region, month)."""
    table: str
    region: str
    period: str
    n_runs: int
    n_rows: int
    expected_runs: int
    missing_runs: list = field(default_factory=list)
    n_missing_internal: int = 0
    n_missing_leading: int = 0
    n_missing_trailing: int = 0
    wrong_target_count: list = field(default_factory=list)
    max_lead_minutes: int = 0
    min_rrp: float = float('nan')
    max_rrp: float = float('nan')
    out_of_range: int = 0
    price_cap: float = float('nan')
    duplicates: int = 0
    errors: list = field(default_factory=list)

    @property
    def known_gap(self) -> bool:
        """This month is absent from the archive and always will be."""
        return (self.table, self.period) in KNOWN_ARCHIVE_GAPS

    @property
    def ok(self) -> bool:
        return (not self.errors and self.out_of_range == 0
                and self.duplicates == 0 and self.n_missing_internal == 0)

    @property
    def partial(self) -> bool:
        """Month is truncated at one or both ends rather than gappy inside.

        Expected at the edges of an acquisition range: the first month of a
        range is missing the runs that live in the previous month's SOURCE
        file, which is 1 run for P5MIN and 9 for PREDISPATCH. Also the normal
        state of a month whose acquisition is only partially complete, or of
        a deliberately small test fixture -- neither is a defect, which is
        why this is reported rather than folded into `ok`. Must still be
        VISIBLE: a partition holding one day of a month would otherwise look
        complete.
        """
        return (self.n_missing_leading + self.n_missing_trailing) > 0

    @property
    def degraded(self) -> bool:
        """True when the month has no usable lead dimension at all.

        October 2022's `_D` variant is the motivating case: the DATA/ copy
        loads at 100% run coverage of the month but at lead 0 only, so a
        store built from it would look complete while being useless for
        rolling optimisation. This is deliberately narrow -- a short or
        partial partition (an in-progress acquisition, a test fixture, the
        current month still being written) is NOT degraded merely for being
        short, provided the runs it does have carry a real forecast lead.
        `partial` already reports shortness on its own terms; conflating the
        two here previously flagged every partial partition as if it had no
        usable forecast, which is a different and much narrower failure.
        See known_gap for the archive-confirmed-absent case, which is
        independent of this and does not require max_lead_minutes to be 0
        (October 2022's single surviving run has a perfectly normal lead).
        """
        return self.n_runs > 0 and self.max_lead_minutes == 0

    @property
    def degraded_reason(self) -> str:
        if (self.table, self.period) in KNOWN_ARCHIVE_GAPS:
            return KNOWN_ARCHIVE_GAPS[(self.table, self.period)]
        if not self.degraded:
            return ''
        return 'no multi-lead forecast in the archive for this period'


def validate_partition(store_root: Path, table: str, region: str,
                       period: pd.Period) -> TableReport:
    """Validate one month of one region."""
    spec = TABLE_SPEC[table]
    cadence = pd.Timedelta(minutes=spec['run_cadence_minutes'])
    path = partition_path(store_root, table, region, period)
    rep = TableReport(table=table, region=region, period=str(period),
                      n_runs=0, n_rows=0, expected_runs=0)
    if not path.exists():
        rep.errors.append(f"partition missing: {path}")
        return rep

    df = pd.read_parquet(path)
    rep.n_rows = len(df)
    if df.empty:
        rep.errors.append("partition is empty")
        return rep

    runs = np.sort(df['run_time'].unique())
    rep.n_runs = len(runs)
    rep.max_lead_minutes = int(df['lead_minutes'].max())
    rep.min_rrp = float(df['rrp_mwh'].min())
    rep.max_rrp = float(df['rrp_mwh'].max())
    rep.duplicates = int(df.duplicated(subset=['run_time', 'lead_minutes']).sum())

    # Run grid, against the FULL CALENDAR MONTH rather than the observed
    # range. Deriving the grid from the first and last run present makes
    # truncation invisible: a partition holding one day of a month would report
    # complete coverage of that day and nothing would flag it. Missing runs are
    # then split into leading, trailing and internal, because only internal
    # gaps are defects. Leading and trailing gaps are the normal consequence of
    # acquiring a bounded range, since a month's first runs live in the
    # PREVIOUS month's source file.
    #
    # For PREDISPATCH the grid is also the only cheap guard on the
    # PREDISPATCHSEQNO arithmetic: run times are derived from a sequence number
    # rather than read, so if AEMO's convention ever changed, every run time
    # would shift silently with no error raised anywhere else.
    month_start = period.to_timestamp().tz_localize(runs[0].tz)
    month_end = (period + 1).to_timestamp().tz_localize(runs[0].tz) - cadence
    grid = pd.date_range(month_start, month_end, freq=cadence)
    rep.expected_runs = len(grid)
    observed = pd.DatetimeIndex(runs)
    missing = grid.difference(observed)
    if len(missing):
        first_obs, last_obs = observed[0], observed[-1]
        rep.n_missing_leading = int((missing < first_obs).sum())
        rep.n_missing_trailing = int((missing > last_obs).sum())
        rep.n_missing_internal = (len(missing) - rep.n_missing_leading
                                  - rep.n_missing_trailing)
    internal = missing[(missing >= observed[0]) & (missing <= observed[-1])] \
        if len(missing) else missing
    rep.missing_runs = [str(t) for t in internal[:20]]
    if len(internal) > 20:
        rep.missing_runs.append(f"... and {len(internal) - 20} more")

    off_grid = pd.DatetimeIndex(runs).difference(grid)
    if len(off_grid):
        rep.errors.append(
            f"{len(off_grid)} run time(s) off the "
            f"{spec['run_cadence_minutes']}-minute grid, first {off_grid[0]}")

    # Target counts against the closed form, never against a median.
    counts = df.groupby('run_time', observed=True).size()
    for run_time, n in counts.items():
        want = expected_target_count(pd.Timestamp(run_time), table)
        if n != want:
            rep.wrong_target_count.append(
                {'run_time': str(run_time), 'got': int(n), 'expected': want})
        if len(rep.wrong_target_count) >= 20:
            rep.wrong_target_count.append({'note': 'truncated at 20'})
            break

    # Range validation, per financial year. This lives here rather than in the
    # provider read path because the cap is reset every 1 July, so the correct
    # value is a per-year fact and the check should run once at ingest rather
    # than 105,000 times per site-year.
    mid = pd.Timestamp(runs[len(runs) // 2])
    try:
        cap = market_price_cap(mid)
    except KeyError as e:
        rep.errors.append(str(e))
        return rep
    floor = market_price_floor(mid)
    rep.price_cap = cap
    bad = (df['rrp_mwh'] < floor - 1e-6) | (df['rrp_mwh'] > cap + 1e-6)
    rep.out_of_range = int(bad.sum())
    if rep.out_of_range:
        rep.errors.append(
            f"{rep.out_of_range} price(s) outside [{floor:,.0f}, {cap:,.0f}] "
            f"$/MWh for FY containing {mid.date()}: observed "
            f"[{rep.min_rrp:,.2f}, {rep.max_rrp:,.2f}]")
    return rep


def validate_store(store_root: Path, tables=('p5min', 'predispatch'),
                   regions=('NSW1', 'QLD1', 'SA1', 'TAS1', 'VIC1'),
                   start: str | None = None, end: str | None = None,
                   verbose: bool = True) -> list[TableReport]:
    """Validate every partition present, or a bounded range."""
    reports = []
    for table in tables:
        for region in regions:
            region_dir = Path(store_root) / table / region
            if not region_dir.exists():
                continue
            periods = sorted(pd.Period(p.stem[:4] + '-' + p.stem[4:], freq='M')
                             for p in region_dir.glob('*.parquet'))
            if start:
                periods = [p for p in periods if p >= pd.Period(start, freq='M')]
            if end:
                periods = [p for p in periods if p <= pd.Period(end, freq='M')]
            for period in periods:
                rep = validate_partition(store_root, table, region, period)
                reports.append(rep)
                if verbose:
                    flag = ('GAP ' if rep.known_gap
                            else 'OK  ' if rep.ok else 'FAIL')
                    extra = (' DEGRADED' if rep.degraded
                             else ' PARTIAL' if rep.partial else '')
                    print(f"  [{flag}]{extra} {table:<12} {region:<5} {rep.period} "
                          f"runs {rep.n_runs:>5}/{rep.expected_runs:<5} "
                          f"rows {rep.n_rows:>7,} "
                          f"maxlead {rep.max_lead_minutes:>5} min")
    return reports


def build_manifest(reports: list[TableReport], store_root: Path) -> dict:
    """Write manifest.json recording coverage and degraded periods.

    The manifest is what lets a provider distinguish "this month is complete"
    from "this month exists but has no lead dimension". Without it, October 2022
    from the DATA/ `_D` variant is indistinguishable from a healthy month, and
    a rolling run over it would silently optimise against a one-step forecast.
    """
    entries = []
    for r in reports:
        entries.append({
            'table': r.table, 'region': r.region, 'period': r.period,
            'n_runs': r.n_runs, 'n_rows': r.n_rows,
            'max_lead_minutes': r.max_lead_minutes,
            'missing_runs_internal': r.n_missing_internal,
            'missing_runs_leading': r.n_missing_leading,
            'missing_runs_trailing': r.n_missing_trailing,
            'partial': r.partial,
            'wrong_target_count': len(r.wrong_target_count),
            'out_of_range': r.out_of_range,
            'degraded': r.degraded,
            'known_archive_gap': r.known_gap,
            'ok': r.ok,
            'reason': r.degraded_reason,
        })
    manifest = {
        'schema_version': 1,
        'generated_utc': pd.Timestamp.now('UTC').isoformat(),
        'market_timezone': MARKET_TZ,
        'entries': entries,
    }
    path = Path(store_root) / 'manifest.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    return manifest


def print_summary(reports: list[TableReport]) -> bool:
    gaps = [r for r in reports if r.known_gap]
    n_ok = sum(r.ok and not r.known_gap for r in reports)
    degraded = [r for r in reports if r.degraded and not r.known_gap]
    partial = [r for r in reports if r.partial and not r.known_gap]
    missing = sum(r.n_missing_internal for r in reports)
    wrong = sum(len([w for w in r.wrong_target_count if 'note' not in w])
                for r in reports)
    print(f"\n=== Store validation: {n_ok}/{len(reports) - len(gaps)} "
          "partitions OK ===")
    if gaps:
        print(f"  known archive gaps      {len(gaps)} "
              "(excluded from the count above; not retryable)")
        for r in gaps[:6]:
            print(f"    {r.table} {r.region} {r.period}: "
                  f"{r.n_runs}/{r.expected_runs} runs")
        if len(gaps) > 6:
            print(f"    ... and {len(gaps) - 6} more")
    print(f"  missing runs (internal) {missing:,}")
    print(f"  partial months          {len(partial)}")
    if partial:
        lead = sum(r.n_missing_leading for r in partial)
        trail = sum(r.n_missing_trailing for r in partial)
        print(f"    {lead:,} leading and {trail:,} trailing run(s) absent.")
        if lead:
            print("    Leading: a month's FIRST runs live in the previous "
                  "month's source file,")
            print("    so acquire one month earlier to fill them.")
        if trail:
            print("    Trailing: the month was never acquired, or only its "
                  "spillover runs were")
            print("    (the previous month's file reaches into it). Acquire "
                  "the month to fill.")
        for r in partial[:6]:
            print(f"      {r.table} {r.region} {r.period}: "
                  f"{r.n_missing_leading} leading, {r.n_missing_trailing} trailing")
    print(f"  wrong target counts     {wrong:,}")
    print(f"  degraded periods        {len(degraded)}")
    for r in degraded:
        print(f"    {r.table} {r.region} {r.period}: max lead "
              f"{r.max_lead_minutes} min")
    failed = [r for r in reports if not r.ok]
    if failed:
        print(f"\n  {len(failed)} partition(s) FAILED:")
        for r in failed[:10]:
            print(f"    {r.table} {r.region} {r.period}: {'; '.join(r.errors)[:140]}")
    return not failed


if __name__ == '__main__':
    import argparse
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store', type=Path,
                        default=Path('data/clean/predispatch'))
    parser.add_argument('--tables', type=str, default='p5min,predispatch')
    parser.add_argument('--regions', type=str,
                        default='NSW1,QLD1,SA1,TAS1,VIC1')
    parser.add_argument('--start', type=str, default=None, help='YYYY-MM')
    parser.add_argument('--end', type=str, default=None, help='YYYY-MM')
    args = parser.parse_args()

    reports = validate_store(
        args.store,
        tables=tuple(t.strip() for t in args.tables.split(',')),
        regions=tuple(r.strip() for r in args.regions.split(',')),
        start=args.start, end=args.end)
    if not reports:
        print(f"No partitions found under {args.store.resolve()}")
        sys.exit(1)
    ok = print_summary(reports)
    build_manifest(reports, args.store)
    print(f"\nManifest written to {(args.store / 'manifest.json').resolve()}")
    sys.exit(0 if ok else 1)
