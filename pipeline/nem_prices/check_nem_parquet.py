"""
check_nem_parquet.py — diagnostic for NaN values in cleaned NEM price parquets.
Run from repo root: python3 check_nem_parquet.py
"""
import pandas as pd
from pathlib import Path

CLEAN_DIR = Path("data/clean/nem_prices")
YEARS = [2022, 2023, 2024, 2025, 2026]
REGION_COLS = ['NSW1_rrp', 'QLD1_rrp', 'SA1_rrp', 'VIC1_rrp']

for year in YEARS:
    p = CLEAN_DIR / f'{year}_nem_prices.parquet'
    if not p.exists():
        print(f"{year}: FILE MISSING")
        continue

    df = pd.read_parquet(p)
    if 'datetime_aest' in df.columns:
        df = df.set_index('datetime_aest')

    print(f"\n{'='*60}")
    print(f"{year}: {len(df):,} rows")

    # NaN counts per RRP column
    nan_counts = {col: df[col].isnull().sum() for col in REGION_COLS if col in df.columns}
    print(f"  NaN counts: {nan_counts}")

    # price_flag distribution
    if 'price_flag' in df.columns:
        flag_counts = df['price_flag'].value_counts().to_dict()
        print(f"  price_flag distribution: {flag_counts}")

        # Are all NaNs accounted for by price_flag==1?
        for col in REGION_COLS:
            if col not in df.columns:
                continue
            nan_mask = df[col].isnull()
            n_nan = nan_mask.sum()
            if n_nan > 0:
                n_flagged   = (df.loc[nan_mask, 'price_flag'] == 1).sum()
                n_unflagged = (df.loc[nan_mask, 'price_flag'] == 0).sum()
                print(f"  {col}: {n_nan} NaNs — {n_flagged} have price_flag=1, "
                      f"{n_unflagged} have price_flag=0 (unexpected)")
                # Show a sample of when the NaNs occur
                sample_times = df.index[nan_mask][:5].tolist()
                print(f"    First few NaN timestamps: {sample_times}")
    else:
        print("  WARNING: no price_flag column found")

print("\nDone.")
