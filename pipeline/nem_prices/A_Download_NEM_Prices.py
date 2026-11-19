"""
A_Download_NEM_Prices.py

Downloads AEMO TRADINGPRICE data and saves a raw compiled CSV per year.

For 2022–July 2024: uses NEMOSIS (PUBLIC_DVD_* archive files, no issues).
For Aug 2024 onwards: downloads directly from the AEMO aggregated price/demand
endpoint, which has plain URLs with no # characters.

Run from repo root: python3 pipeline/nem_prices/A_Download_NEM_Prices.py

Output: data/processed/nem_prices/YYYY_nem_prices_raw.csv

── Why two paths? ─────────────────────────────────────────────────────────────
From Aug 2024, AEMO renamed MMSDM files to PUBLIC_ARCHIVE#TABLE#FILE01#*.zip.
NEMWeb's IIS server returns 400 Bad Request for %23-encoded paths and 404 for
literal # paths — the files cannot be downloaded programmatically via HTTP.
This is an unresolved bug in NEMOSIS v3.8.1 (GitHub issue #40).

The AEMO aggregated price/demand endpoint provides identical 5-minute trading
price data via clean monthly per-region CSV files:
  https://aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_{YYYYMM}_{REGION}.csv
Columns: REGION, SETTLEMENTDATE, TOTALDEMAND, RRP, PERIODTYPE
"""

from datetime import date
from pathlib import Path

import pandas as pd
import requests
import nemosis

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR   = SCRIPT_DIR.parent.parent
RAW_CACHE_DIR = ROOT_DIR / "data" / "raw" / "nem_prices"
PROCESSED_DIR = ROOT_DIR / "data" / "processed" / "nem_prices"

# ── Configuration ──────────────────────────────────────────────────────────────
#YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
REGIONS = ['NSW1', 'QLD1', 'SA1', 'VIC1']

# First month that requires the direct-download path
NEMOSIS_CUTOFF = date(2024, 8, 1)

AEMO_URL = (
    "https://aemo.com.au/aemo/data/nem/priceanddemand/"
    "PRICE_AND_DEMAND_{yyyymm}_{region}.csv"
)

CHROME_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# ── Setup ──────────────────────────────────────────────────────────────────────
RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
today = date.today()


def fetch_direct(year: int, month: int) -> pd.DataFrame | None:
    """
    Fetch one month of 5-min price data from AEMO's aggregated endpoint.
    Returns DataFrame with columns [SETTLEMENTDATE, REGIONID, RRP], or None.
    """
    dfs = []
    for region in REGIONS:
        url = AEMO_URL.format(yyyymm=f"{year}{month:02d}", region=region)
        try:
            r = requests.get(url, headers={"User-Agent": CHROME_UA}, timeout=60)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"    {region} {year}-{month:02d}: FAILED — {e}")
            continue

        df = pd.read_csv(pd.io.common.StringIO(r.text))
        df.columns = [c.strip().upper() for c in df.columns]

        # Rename REGION → REGIONID if needed
        if 'REGION' in df.columns and 'REGIONID' not in df.columns:
            df = df.rename(columns={'REGION': 'REGIONID'})

        df = df[['SETTLEMENTDATE', 'REGIONID', 'RRP']].copy()
        df['RRP'] = pd.to_numeric(df['RRP'], errors='coerce')
        # Normalise date format: AEMO aggregated uses YYYY/MM/DD HH:MM:SS
        # but NEMOSIS produces YYYY-MM-DD HH:MM:SS — standardise to the latter
        df['SETTLEMENTDATE'] = (
            pd.to_datetime(df['SETTLEMENTDATE'], dayfirst=False)
            .dt.strftime('%Y-%m-%d %H:%M:%S')
        )
        dfs.append(df)

    if not dfs:
        return None
    result = pd.concat(dfs, ignore_index=True)
    print(f"    Direct fetch {year}-{month:02d}: {len(result):,} rows across {len(dfs)} regions")
    return result


# ── Download ───────────────────────────────────────────────────────────────────
for year in YEARS:
    if year > today.year:
        continue

    print(f"\nYear {year}: downloading TRADINGPRICE...")

    dfs = []

    # NEMOSIS for months before the cutoff
    if date(year, 1, 1) < NEMOSIS_CUTOFF:
        if date(year, 12, 31) < NEMOSIS_CUTOFF:
            # Entire year is in the archive — run to start of next year
            nemosis_end = f"{year + 1}/01/01 00:00:00"
        else:
            # Partial year — run up to the cutoff month boundary
            nemosis_end = f"{NEMOSIS_CUTOFF.year}/{NEMOSIS_CUTOFF.month:02d}/01 00:00:00"
        print(f"  NEMOSIS: {year}/01/01 → {nemosis_end}")
        try:
            df_nem = nemosis.dynamic_data_compiler(
                start_time=f"{year}/01/01 00:00:00",
                end_time=nemosis_end,
                table_name="TRADINGPRICE",
                raw_data_location=str(RAW_CACHE_DIR),
                select_columns=["SETTLEMENTDATE", "REGIONID", "RRP"],
                filter_cols=["REGIONID"],
                filter_values=(["NSW1", "QLD1", "SA1", "VIC1"],),
            )
            print(f"  NEMOSIS: {len(df_nem):,} rows")
            dfs.append(df_nem)
        except Exception as e:
            print(f"  NEMOSIS ERROR: {e}")

    # Direct fetch for months from cutoff onwards
    first_direct_month = max(date(year, 1, 1), NEMOSIS_CUTOFF)
    if first_direct_month.year == year:
        last_month = today.month if year == today.year else 12
        months = range(first_direct_month.month, last_month + 1)
        if months:
            print(f"  Direct fetch: {year}-{first_direct_month.month:02d} → {year}-{last_month:02d}")
            direct_dfs = [fetch_direct(year, m) for m in months]
            direct_dfs = [d for d in direct_dfs if d is not None]
            if direct_dfs:
                dfs.extend(direct_dfs)

    if not dfs:
        print(f"  ERROR: No data retrieved for {year} — skipping")
        continue

    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["SETTLEMENTDATE", "REGIONID"])

    for region in REGIONS:
        print(f"  {region}: {(df['REGIONID'] == region).sum():,} rows")

    output_file = PROCESSED_DIR / f"{year}_nem_prices_raw.csv"
    df.to_csv(output_file, index=False)
    print(f"  Total: {len(df):,} rows → {output_file.name}")

print("\nDownload complete.")