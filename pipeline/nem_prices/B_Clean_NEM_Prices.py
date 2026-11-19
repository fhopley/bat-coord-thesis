"""
B_Clean_NEM_Prices.py

Reads raw CSV output from A_Download_NEM_Prices.py for each year in YEARS.
Localises timestamps to AEST (UTC+10, fixed), pivots to wide format, reindexes
to a complete 5-minute grid, and back-fills 30-minute pre-5MS intervals.
Prints quarterly TWAP per region for validation against AEMO QED reports.

Run from repo root: python3 pipeline/nem_prices/B_Clean_NEM_Prices.py

Output: data/clean/nem_prices/YYYY_nem_prices.parquet
"""

import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed" / "nem_prices"
CLEAN_DIR = ROOT_DIR / "data" / "clean" / "nem_prices"

# ── Configuration — narrow YEARS list for development/testing ──────────────────
#YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
REGIONS = ['NSW1', 'QLD1', 'SA1', 'VIC1']
REGION_COLS = [f'{r}_rrp' for r in REGIONS]
LEAP_YEARS = {2020, 2024}

# ── Process ────────────────────────────────────────────────────────────────────
for year in YEARS:
    print(f"\nYear {year}: cleaning NEM price data...")

    raw_file = PROCESSED_DIR / f'{year}_nem_prices_raw.csv'
    output_file = CLEAN_DIR / f'{year}_nem_prices.parquet'

    # 1. Load and localise timestamps
    # NEMOSIS returns SETTLEMENTDATE in AEST without explicit tzinfo; attach +10:00.
    df = pd.read_csv(raw_file)
    df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'])
    df['SETTLEMENTDATE'] = df['SETTLEMENTDATE'].dt.tz_localize('Etc/GMT-10')
    df = df.rename(columns={'SETTLEMENTDATE': 'datetime_aest'})
    print(f"  Loaded {len(df):,} rows from raw file")

    # 2. Duplicate check before pivot
    dupes = df.duplicated(subset=['datetime_aest', 'REGIONID'])
    if dupes.any():
        print(f"  WARNING: {dupes.sum()} duplicate (datetime_aest, REGIONID) pairs — keeping first")
        df = df[~df.duplicated(subset=['datetime_aest', 'REGIONID'], keep='first')]

    # 3. Pivot long → wide: one row per interval, one column per region
    df_wide = df.pivot(index='datetime_aest', columns='REGIONID', values='RRP')
    df_wide.columns.name = None
    df_wide.columns = [f'{col}_rrp' for col in df_wide.columns]

    # 4. Reindex to complete 5-minute AEST grid, then back-fill pre-5MS slots only.
    #
    # SETTLEMENTDATE is end-of-period: the 30-minute price at (e.g.) 00:30 applies
    # to all five preceding 5-minute slots (00:05–00:25). bfill fills each NaN slot
    # from the next settlement value. limit=5 is intentional: between two consecutive
    # present 30-minute settlements there are exactly 5 NaN slots, so limit=5 fills
    # those and stops. A genuine missing 30-minute settlement produces 11 consecutive
    # NaN slots; the 6 beyond the limit remain NaN with price_flag=1.
    #
    # bfill is applied ONLY to the pre-5MS portion of the index (before 1 Oct 2021).
    # Post-5MS, every 5-minute slot should already be present in TRADINGPRICE — any
    # NaN after reindex is a genuine missing interval and must not be filled silently.
    # Splitting the reindex ensures a missing post-5MS interval (e.g. during the
    # June 2022 NEM suspension) is correctly left as NaN with price_flag=1.
    expected_count = 105_408 if year in LEAP_YEARS else 105_120
    full_index = pd.date_range(
        start=f'{year}-01-01 00:05:00',
        end=f'{year + 1}-01-01 00:00:00',
        freq='5min',
        tz='Etc/GMT-10',
    )
    full_index.name = 'datetime_aest'

    TRANSITION = pd.Timestamp('2021-10-01 00:05:00', tz='Etc/GMT-10')
    pre_index = full_index[full_index < TRANSITION]
    post_index = full_index[full_index >= TRANSITION]

    df_pre = df_wide.reindex(pre_index).bfill(limit=5)
    df_post = df_wide.reindex(post_index)  # no fill — missing stays NaN with price_flag=1

    df_wide = pd.concat([df_pre, df_post])

    # 5. Completeness check
    actual_count = len(df_wide)
    if actual_count != expected_count:
        print(f"  WARNING: Expected {expected_count:,} rows, got {actual_count:,}")
    else:
        print(f"  Row count: {actual_count:,} ✓")

    # 6. Flag missing intervals (NaN after all filling = genuine source gap)
    nan_mask = df_wide[REGION_COLS].isna().any(axis=1)
    df_wide['price_flag'] = nan_mask.astype(int)
    nan_count = nan_mask.sum()
    if nan_count > 0:
        print(f"  WARNING: {nan_count} intervals with missing price data (price_flag=1)")
    else:
        print(f"  No missing intervals after back-fill ✓")

    # 7. Quarterly TWAP — cross-check against AEMO QED reports
    print(f"  Quarterly TWAP ($/MWh):")
    for q in range(1, 5):
        q_mask = df_wide.index.quarter == q
        twap = df_wide.loc[q_mask, REGION_COLS].mean()
        twap_str = "  ".join(f"{col}: {v:.2f}" for col, v in twap.items())
        print(f"    Q{q}: {twap_str}")

    # 8. Save as Parquet
    # Writing directly from the in-memory DataFrame avoids a CSV round-trip.
    # pyarrow stores datetime_aest as a timezone-aware timestamp (UTC-normalised
    # with Etc/GMT-10 metadata), so readers get a tz-aware series automatically
    # without needing to re-localise on load.
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    df_out = df_wide.reset_index()
    df_out.to_parquet(output_file, engine='pyarrow', compression='snappy', index=False)
    parquet_size_mb = output_file.stat().st_size / (1024 ** 2)
    print(f"  Saved to: {output_file.name} ({parquet_size_mb:.1f} MB)")

print("\nClean complete.")