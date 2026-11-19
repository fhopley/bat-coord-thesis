"""
edp_annual_totals.py

Annual gross load and PV generation per validated site-year, summed from the
infilled clean files (data/clean/edp/YYYY/edp_clean_{site_id}_{YYYY}.parquet).
Input to Figure C.4 (appendix_c_edp_summary.py).

Writes data/processed/edp/YYYY_site_annual_totals.csv, one row per clean file:

    edp_site_id, year, n_intervals, annual_gross_load_kwh, annual_pv_kwh

Run where the clean parquet files are, from the project root:

    python3 analysis/scripts/edp_annual_totals.py --data-root "/Volumes/Manual/Thesis Project"

The output goes under --data-root as well; copy the five CSVs into the repo's
data/processed/edp/.
"""
import argparse
from pathlib import Path

import pandas as pd

COLUMNS = ['gross_load_wh', 'pv_site_net_wh']


def year_totals(data_root: Path, year: int) -> pd.DataFrame:
    clean_dir = data_root / 'data' / 'clean' / 'edp' / str(year)
    files = sorted(clean_dir.glob(f'edp_clean_*_{year}.parquet'))
    if not files:
        raise SystemExit(f"No clean parquet files in {clean_dir}")
    rows = []
    for i, f in enumerate(files, 1):
        site_id = f.stem.split('_')[2]
        df = pd.read_parquet(f, columns=COLUMNS)
        rows.append({
            'edp_site_id': site_id,
            'year': year,
            'n_intervals': len(df),
            'annual_gross_load_kwh': df['gross_load_wh'].sum() / 1000.0,
            'annual_pv_kwh': df['pv_site_net_wh'].sum() / 1000.0,
        })
        if i % 50 == 0 or i == len(files):
            print(f"  {year}: {i}/{len(files)}")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path, default=Path('.'))
    ap.add_argument('--years', nargs='*', type=int,
                    default=[2021, 2022, 2023, 2024, 2025])
    args = ap.parse_args()
    out_dir = args.data_root / 'data' / 'processed' / 'edp'
    for year in args.years:
        totals = year_totals(args.data_root, year)
        out = out_dir / f'{year}_site_annual_totals.csv'
        totals.to_csv(out, index=False, float_format='%.3f')
        print(f"Saved {len(totals)} site-years to {out}")


if __name__ == '__main__':
    main()
