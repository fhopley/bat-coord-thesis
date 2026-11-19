"""
L_Convert_To_Parquet.py

Converts all per-site clean CSV files to Parquet format after the full
pipeline (A–K) has completed. Original CSV files are removed after each
successful conversion to avoid doubling storage.

Parquet advantages over CSV for model ingest:
- datetime_aest stored as timezone-aware timestamp (Etc/GMT-10), eliminating
  per-row string parsing on load
- Typed columns — integer flags and float energy values stored natively
- Faster reads (columnar format, snappy compression)
- Smaller on disk (~40–60% of CSV size with snappy compression)

This script is the final pipeline step and is called automatically by
run_pipeline.py. It can also be run standalone to convert a specific year.

Conversion details
------------------
- datetime_aest: parsed as timezone-aware (Etc/GMT-10) on read, stored
  as native timestamp in Parquet. Downstream model code should read with
  pd.read_parquet() — datetime_aest will be tz-aware without further parsing.
- Compression: snappy (pyarrow default). Fast reads, modest file-size
  reduction.
- Integer columns (pv_flag, load_flag, pv_infill, load_infill) stay
  integer; energy columns stay float. Types inferred from CSV on first read.
- CSV files are removed after successful Parquet write. The pipeline must
  be complete before running this script — do not run mid-pipeline.

Usage:
    python3 pipeline/edp/L_Convert_To_Parquet.py 2022
    python3 pipeline/edp/L_Convert_To_Parquet.py 2022 2023 2024
"""

import sys
import time
from pathlib import Path

import pandas as pd


def _format_size(bytes_: int) -> str:
    if bytes_ < 1024:
        return f"{bytes_}B"
    if bytes_ < 1024 ** 2:
        return f"{bytes_/1024:.1f}KB"
    if bytes_ < 1024 ** 3:
        return f"{bytes_/(1024**2):.1f}MB"
    return f"{bytes_/(1024**3):.2f}GB"


def convert_year(year_dir: Path) -> tuple[int, float, float, int, int]:
    """Convert all CSVs in a year directory, removing originals after success."""
    csvs = sorted(year_dir.glob('edp_clean_*.csv'))
    if not csvs:
        return 0, 0.0, 0.0, 0, 0

    print(f"\n=== {year_dir.name}: {len(csvs)} files ===", flush=True)
    n = 0
    total_read = 0.0
    total_write = 0.0
    total_csv_size = 0
    total_parquet_size = 0

    for i, csv_path in enumerate(csvs, start=1):
        parquet_path = csv_path.with_suffix('.parquet')
        csv_size = csv_path.stat().st_size

        t0 = time.perf_counter()
        df = pd.read_csv(csv_path, parse_dates=['datetime_aest'])
        ts = df['datetime_aest']
        if ts.dt.tz is None:
            df['datetime_aest'] = ts.dt.tz_localize('Etc/GMT-10')
        else:
            df['datetime_aest'] = ts.dt.tz_convert('Etc/GMT-10')
        t_read = time.perf_counter() - t0

        t0 = time.perf_counter()
        df.to_parquet(parquet_path, engine='pyarrow', compression='snappy',
                      index=False)
        t_write = time.perf_counter() - t0

        parquet_size = parquet_path.stat().st_size
        total_csv_size += csv_size
        total_parquet_size += parquet_size
        total_read += t_read
        total_write += t_write

        # Remove CSV only after Parquet is confirmed written
        csv_path.unlink()
        n += 1

        if i <= 3 or i == len(csvs) or i % 50 == 0:
            print(f"  [{i}/{len(csvs)}] {csv_path.name:40} "
                  f"read={t_read:.2f}s write={t_write:.2f}s "
                  f"{_format_size(csv_size)} -> {_format_size(parquet_size)}",
                  flush=True)

    print(f"  Year totals: read={total_read:.1f}s  write={total_write:.1f}s  "
          f"size {_format_size(total_csv_size)} -> "
          f"{_format_size(total_parquet_size)} "
          f"({100 * total_parquet_size / total_csv_size:.0f}% of CSV)",
          flush=True)
    return n, total_read, total_write, total_csv_size, total_parquet_size


def main(argv=None) -> int:
    # Accept years as positional command-line arguments (matches pipeline convention)
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("Usage: python3 L_Convert_To_Parquet.py <YEAR> [YEAR ...]")
        print("Example: python3 L_Convert_To_Parquet.py 2022")
        return 1

    years = []
    for a in args:
        try:
            years.append(int(a))
        except ValueError:
            print(f"ERROR: expected integer year, got '{a}'", file=sys.stderr)
            return 1

    root      = Path(__file__).parent.parent.parent  # pipeline/edp/ → pipeline/ → project root
    clean_root = root / 'data' / 'clean' / 'edp'
    if not clean_root.is_dir():
        print(f"ERROR: not found: {clean_root}", file=sys.stderr)
        return 1

    year_dirs = [clean_root / str(y) for y in years]

    grand_n = 0
    grand_read = 0.0
    grand_write = 0.0
    grand_csv = 0
    grand_parquet = 0

    for yd in year_dirs:
        if not yd.is_dir():
            print(f"WARNING: not a directory, skipping: {yd}", file=sys.stderr)
            continue
        n, r, w, c, p = convert_year(yd)
        grand_n += n
        grand_read += r
        grand_write += w
        grand_csv += c
        grand_parquet += p

    if grand_n == 0:
        print("\nNo files converted.", flush=True)
        return 1

    print(f"\n=== GRAND TOTAL ===")
    print(f"  Files converted: {grand_n}")
    print(f"  CSV read time:   {grand_read:.1f}s")
    print(f"  Parquet write:   {grand_write:.1f}s")
    print(f"  CSV total size:  {_format_size(grand_csv)}")
    print(f"  Parquet total:   {_format_size(grand_parquet)} "
          f"({100 * grand_parquet / grand_csv:.0f}% of CSV)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
