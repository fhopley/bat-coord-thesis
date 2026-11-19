"""Stage A: acquire raw AEMO pre-dispatch and build the canonical store.

Orchestration around nemweb_direct, which does the actual fetching and which
was validated against the real archive during the investigation. Nothing here
constructs a filename or parses a zip; this module decides WHICH months to ask
for, caches the answers, hands them to stage B, and fails soft.

Two tables, from different archive folders:

    P5MIN / REGIONSOLUTION      DATA/              5-minute, 12 targets per run
    PREDISPATCH / PRICE         PREDISP_ALL_DATA/  30-minute, sawtooth horizon

The PREDISP_ALL_DATA routing is not optional and not cosmetic. The DATA/ copy
of PREDISPATCHPRICE carries a `_D` suffix and holds exactly one forecast per
target interval, at lead 0. It loads cleanly, reports full coverage, and has no
lead-time dimension at all, which makes it useless for rolling optimisation in
a way no row count reveals. nemweb_direct routes correctly; this module simply
must not override it.

Month selection
---------------
Run windows are given as CALENDAR months here, and nemweb_direct shifts the
SOURCE file selection backwards to compensate for AEMO partitioning on a
trading-day clock (4h30m for PREDISPATCH, 5 min for P5MIN). Without that shift
roughly 600 PREDISPATCH runs and 67 P5MIN runs go missing across 2021 to 2026,
silently, because they live in the previous month's file. Passing calendar
months and letting nemweb_direct do the shifting is what keeps the two
conventions from being confused.

Usage (from project root):
    python3 -m pipeline.pre_dispatch.A_Download_Predispatch --start 2025-09 --end 2026-03
    python3 -m pipeline.pre_dispatch.A_Download_Predispatch --start 2021-01 --end 2026-06
    python3 -m pipeline.pre_dispatch.A_Download_Predispatch --start 2025-09 --end 2025-09 \\
        --tables predispatch

Downloads and caches only. Stage B builds the canonical store from that cache,
and is separate so a change to the cleaning rules can be applied to years of
data in minutes without re-downloading. Pass --clean to do both in one go.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from pipeline.pre_dispatch.B_Clean_Predispatch import (
    clean_forecast_frame, partition_path, write_partitions,
)

DEFAULT_RAW_CACHE = Path('data/raw/predispatch')
DEFAULT_STORE = Path('data/clean/predispatch')
ALL_REGIONS = ('NSW1', 'QLD1', 'SA1', 'TAS1', 'VIC1')

# nemweb_direct's forecast_type and table name for each of our table keys.
SOURCE = {
    'p5min': ('P5MIN', 'REGIONSOLUTION'),
    'predispatch': ('PREDISPATCH', 'PRICE'),
}

# How far past the end of a run window a forecast can reach. PREDISPATCH tops
# out at 39 hours; three days is margin, and over-wide costs nothing because
# nemweb_direct filters on the run window as well.
_TARGET_MARGIN = pd.Timedelta(days=3)
_DTFMT = '%Y/%m/%d %H:%M'

# Runs at the start of a calendar month that live in the PREVIOUS month's
# source file, because MMSDM partitions on a trading-day clock: 00:00 to 04:00
# for PREDISPATCH at 30-minute cadence, and the single 00:00 run for P5MIN.
# Absent whenever an acquisition range starts at this month, so the idempotence
# check must tolerate them or the first month is never skippable.
_LEADING_RUNS_IN_PREVIOUS_FILE = {'p5min': 1, 'predispatch': 9}


@dataclass
class MonthResult:
    table: str
    period: str
    n_raw_rows: int = 0
    n_clean_rows: int = 0
    n_intervention_dropped: int = 0
    partitions: tuple = ()
    missing_source_months: tuple = ()
    seconds: float = 0.0
    skipped: bool = False
    error: str = ''

    @property
    def ok(self) -> bool:
        return not self.error


def _run_window(period: pd.Period) -> tuple[str, str, str, str]:
    """Calendar-month run window, plus a generous target window."""
    start = period.to_timestamp()
    end = (period + 1).to_timestamp()
    return (start.strftime(_DTFMT), end.strftime(_DTFMT),
            start.strftime(_DTFMT), (end + _TARGET_MARGIN).strftime(_DTFMT))


def _already_built(store_root: Path, table: str, period: pd.Period,
                   regions) -> bool:
    """True when every region's partition holds more than a bare stub.

    Existence is NOT enough, and testing it was a real defect rather than a
    theoretical one. A month's partition is created as a one-run stub by the
    PREVIOUS month's download: the run window is inclusive at both ends, so the
    run labelled exactly at midnight on the 1st belongs to the new calendar
    month and gets filed there. An existence test then declares the month
    already present and skips it.

    In a sequential multi-month run that is not a single lost month. Every
    month stubs the next, so every month after the first is skipped and the run
    reports success in seconds with essentially nothing acquired.

    The threshold is deliberately NOT "close to a full calendar month" -- an
    earlier version compared against the expected grid size, which correctly
    caught the stub case but also flagged any deliberately partial or
    in-progress acquisition (a smaller real range, a month whose archive
    coverage is genuinely thin like October 2022, a test fixture) as
    "not built", defeating idempotence for anything other than a complete
    month. The distinguishing fact that actually matters is narrower: a bare
    spillover stub can hold AT MOST _LEADING_RUNS_IN_PREVIOUS_FILE[table] runs,
    because that is the entire mechanism that creates one. Anything beyond
    that bound could only have arrived via an acquire() call that actually
    targeted this month, spillover cannot produce it. So "more than the bound"
    is both necessary and sufficient to distinguish a real (if partial)
    acquisition from an untouched stub, without needing to know how large a
    complete month should be.
    """
    bound = _LEADING_RUNS_IN_PREVIOUS_FILE[table]
    for region in regions:
        path = partition_path(store_root, table, region, period)
        if not path.exists():
            return False
        try:
            n_runs = pd.read_parquet(
                path, columns=['run_time'])['run_time'].nunique()
        except Exception:                                   # noqa: BLE001
            return False                                    # unreadable: rebuild
        if n_runs <= bound:
            return False
    return True


def download_month(
    period: pd.Period,
    table: str,
    raw_cache: Path = DEFAULT_RAW_CACHE,
    compile_data: Optional[Callable] = None,
) -> pd.DataFrame:
    """Fetch one table-month as a raw frame.

    nemweb_direct caches the raw monthly parquet itself, so a repeat call is a
    cache read rather than a download. compile_data is injectable so the
    orchestration can be tested without touching the network.
    """
    if table not in SOURCE:
        raise ValueError(f"unknown table {table!r}, expected one of {list(SOURCE)}")
    if compile_data is None:
        from pipeline.pre_dispatch.nemweb_direct import compile_data as _cd
        compile_data = _cd

    forecast_type, source_table = SOURCE[table]
    run_start, run_end, fc_start, fc_end = _run_window(period)
    got = compile_data(
        run_start=run_start, run_end=run_end,
        forecasted_start=fc_start, forecasted_end=fc_end,
        forecast_type=forecast_type, tables=[source_table],
        raw_cache=str(raw_cache), data_format='df',
    )
    return got[source_table]


def _is_unavailable_month(exc: BaseException) -> bool:
    """True when a failure is 'the archive does not have this month'.

    Imported lazily to keep nemweb_direct off this module's import path, which
    is what lets the orchestration be tested with an injected compile_data and
    no network stack present.
    """
    try:
        from pipeline.pre_dispatch.nemweb_direct import SourceMonthUnavailable
    except Exception:                                       # noqa: BLE001
        return False
    return isinstance(exc, SourceMonthUnavailable)


def acquire(
    start: str,
    end: str,
    tables=('p5min', 'predispatch'),
    regions=ALL_REGIONS,
    raw_cache: Path = DEFAULT_RAW_CACHE,
    store_root: Path = DEFAULT_STORE,
    clean: bool = False,
    force: bool = False,
    compile_data: Optional[Callable] = None,
) -> list[MonthResult]:
    """Acquire a range of months, cleaning each into the canonical store.

    Fail-soft per table-month. Acquiring five and a half years is one long run
    and a single unavailable month must not lose the other sixty-five; failures
    are collected and reported at the end, matching run_cohort_attribution.
    Stages B and C fail hard by contrast, because a cleaning or validation
    failure is a defect rather than a data condition.

    Idempotent: a month whose partitions all exist is skipped unless force is
    set, so extending the store by one month costs one month.
    """
    periods = pd.period_range(start, end, freq='M')
    results: list[MonthResult] = []
    print(f"Pre-dispatch acquisition: {len(periods)} month(s) "
          f"{periods[0]} to {periods[-1]}, tables {list(tables)}")
    print(f"  raw cache {Path(raw_cache).resolve()}")
    print(f"  store     {Path(store_root).resolve()}\n")

    t_start = time.perf_counter()
    for i, period in enumerate(periods, 1):
        for table in tables:
            res = MonthResult(table=table, period=str(period))
            if clean and not force and _already_built(store_root, table, period, regions):
                res.skipped = True
                results.append(res)
                print(f"[{i}/{len(periods)}] {table:<12} {period}  skipped "
                      "(partitions already present)")
                continue
            t0 = time.perf_counter()
            try:
                raw = download_month(period, table, raw_cache, compile_data)
                res.n_raw_rows = len(raw)
                res.missing_source_months = tuple(
                    raw.attrs.get('missing_source_months', ()))
                if res.missing_source_months:
                    print(f"    NOTE {table} {period}: source month(s) "
                          f"{', '.join(res.missing_source_months)} absent from "
                          "the archive, so this month is short at its leading "
                          "edge. Stage C will report it as PARTIAL.")
                if clean:
                    keep = raw['REGIONID'].astype(str).isin(regions)
                    cleaned = clean_forecast_frame(raw[keep], table)
                    res.n_clean_rows = len(cleaned)
                    res.n_intervention_dropped = int(
                        cleaned.attrs.get('n_intervention_rows_dropped', 0))
                    # merge=True: one source file spills across a month
                    # boundary, so consecutive source months can both write
                    # into a single calendar-month partition.
                    res.partitions = tuple(
                        str(p) for p in write_partitions(cleaned, table,
                                                         store_root, merge=True))
            except Exception as e:                          # noqa: BLE001
                res.error = f"{type(e).__name__}: {e}"
                print(f"[{i}/{len(periods)}] {table:<12} {period}  FAILED: "
                      f"{res.error}")
                # A month absent from the archive is a data condition and its
                # traceback is noise -- the summary already names it and points
                # at the spec. Anything else is a defect and the traceback is
                # the most useful thing on the screen.
                if not _is_unavailable_month(e):
                    traceback.print_exc()
            res.seconds = time.perf_counter() - t0
            results.append(res)
            if res.ok and not res.skipped:
                tail = (f"  clean {res.n_clean_rows:>9,}  "
                        f"dropped {res.n_intervention_dropped:>7,}" if clean else "")
                print(f"[{i}/{len(periods)}] {table:<12} {period}  "
                      f"raw {res.n_raw_rows:>9,}{tail}  "
                      f"{res.seconds:>5.0f} s", flush=True)

    print_summary(results, time.perf_counter() - t_start)
    return results


def print_summary(results: list[MonthResult], wall_seconds: float) -> bool:
    done = [r for r in results if r.ok and not r.skipped]
    skipped = [r for r in results if r.skipped]
    failed = [r for r in results if not r.ok]
    print(f"\n=== Acquisition summary ===")
    print(f"  built    {len(done)} table-month(s)")
    print(f"  skipped  {len(skipped)} (already present)")
    print(f"  failed   {len(failed)}")
    if done:
        raw_rows = sum(r.n_raw_rows for r in done)
        if any(r.n_clean_rows for r in done):
            print(f"  rows     {raw_rows:,} raw, "
                  f"{sum(r.n_clean_rows for r in done):,} clean, "
                  f"{sum(r.n_intervention_dropped for r in done):,} "
                  "intervention rows dropped")
        else:
            print(f"  rows     {raw_rows:,} raw cached (stage B builds the store)")
    print(f"  wall     {wall_seconds / 60:.1f} min")
    if failed:
        print("\n  Failures:")
        for r in failed:
            print(f"    {r.table} {r.period}: {r.error[:160]}")
        print("\n  October 2022 PREDISPATCH is a KNOWN archive gap: the "
              "multi-lead price table\n  does not exist for that month in any "
              "form. See predispatch_data_spec.md.")
    print("\nNOTE: the FIRST month of a range is short by the runs that live in "
          "the previous\n      month's source file (1 for P5MIN, 9 for "
          "PREDISPATCH, being 00:00 to 04:00).\n      Start one month earlier "
          "if you need the boundary filled. Stage C reports this\n      as "
          "PARTIAL rather than a failure.")
    print("\nNext: python3 -m pipeline.pre_dispatch.B_Clean_Predispatch"
          "\n      python3 -m pipeline.pre_dispatch.C_Validate_Predispatch")
    return not failed


if __name__ == '__main__':
    import argparse
    import sys
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--start', required=True, help='YYYY-MM inclusive')
    parser.add_argument('--end', required=True, help='YYYY-MM inclusive')
    parser.add_argument('--tables', default='p5min,predispatch')
    parser.add_argument('--regions', default=','.join(ALL_REGIONS))
    parser.add_argument('--raw-cache', type=Path, default=DEFAULT_RAW_CACHE)
    parser.add_argument('--store', type=Path, default=DEFAULT_STORE)
    parser.add_argument('--clean', action='store_true',
                        help='Also run stage B inline. Off by default: stages '
                             'are separate so the store can be rebuilt from '
                             'cache without re-downloading.')
    parser.add_argument('--force', action='store_true',
                        help='Rebuild months whose partitions already exist')
    args = parser.parse_args()

    results = acquire(
        args.start, args.end,
        tables=tuple(t.strip() for t in args.tables.split(',')),
        regions=tuple(r.strip() for r in args.regions.split(',')),
        raw_cache=args.raw_cache, store_root=args.store,
        clean=args.clean, force=args.force,
    )
    sys.exit(0 if all(r.ok for r in results) else 1)
