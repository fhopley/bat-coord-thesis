"""
D_Write_H0001.py

Writes the H0001 clean trace as Parquet and builds the H0001 site metadata row.

1. Parquet: data/clean/edp/{YYYY}/edp_clean_H0001_{YYYY}.parquet, with
   datetime_aest stored tz-aware (Etc/GMT-10) as in
   pipeline/edp/L_Convert_To_Parquet.py. The CSV is kept alongside it.

2. Metadata: data/processed/h0001/{YYYY}_selected_sites_clean.csv, one row
   with the columns of the EDP metadata file written by
   pipeline/edp/C_Check_Consistency.py. Static fields are hard-coded below.
   The quality-check fields use the C_Check_Consistency.py definitions and
   thresholds, computed from the original (pre-infill) columns of the clean
   trace. Local hour for the solar window is Australia/Sydney.

   The file is kept apart from data/processed/edp/, which holds the EDP cohort
   metadata. The public snapshot places it at
   data/processed/edp/{YYYY}_selected_sites_clean.csv, where model ingest reads it.

Usage from project root:
    python3 pipeline/h0001/D_Write_H0001.py 2022
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2022

SITE = {
    'edp_site_id': 'H0001',
    'state': 'NSW',
    'site_timezone': 'Australia/Sydney',
    'inverter_manufacturer': 'Enphase',
    'inverter_model': 'IQ7+',
    'inverter_ac_rating_kw': 11.6,
    'subarray_dc_rating_kw': 14.8,
    'ac_load_net_phases': 3,
    'pv_site_net_phases': 3,
    'has_battery_flag': False,
    'islandable': 'No',
    'has_battery': np.nan,
    'battery_size_make_model': np.nan,
    'battery_channel_label': np.nan,
    'battery_sign_convention': np.nan,
    'manual_exclusion': False,
}

# Thresholds from C_Check_Consistency.py
SOLAR_HOUR_START = 9
SOLAR_HOUR_END = 16
EXPORT_NOISE_WH = 50
FLATLINE_NOISE_WH = 2
MIN_PV_THRESHOLD_WH = 100
PV_NEGATIVE_WH = -20.0
CORR_THRESHOLD = -0.2
EXPORT_COUNT_THRESHOLD = 10
GROSS_NEG_THRESHOLD = 0.10
FLATLINE_THRESHOLD = 0.05
MIN_ANNUAL_GENERATION_WH = 500_000
MIN_AVG_DAILY_GROSS_LOAD_WH = 5_000
PV_NEGATIVE_THRESHOLD = 0.01
GROSS_PV_CORR_THRESHOLD = 0.475

COLUMNS = [
    'edp_site_id', 'state', 'site_timezone', 'inverter_manufacturer',
    'inverter_model', 'inverter_ac_rating_kw', 'subarray_dc_rating_kw',
    'ac_load_net_coverage_pct', 'ac_load_net_max_gap_min', 'ac_load_net_phases',
    'pv_site_net_coverage_pct', 'pv_site_net_max_gap_min', 'pv_site_net_phases',
    'has_battery_flag', 'islandable', 'has_battery', 'battery_size_make_model',
    'battery_channel_label', 'battery_sign_convention', 'corr_coefficient',
    'corr_worst_month', 'corr_worst_month_value', 'export_excess_intervals',
    'export_excess_max_w', 'export_excess_worst_month', 'gross_neg_fraction',
    'ac_flatline_fraction', 'pv_flatline_fraction', 'annual_generation_kwh',
    'avg_daily_gross_load_wh', 'pv_daytime_negative_fraction', 'gross_pv_corr',
    'corr_pass', 'export_pass', 'ac_flatline_pass', 'pv_flatline_pass',
    'generation_pass', 'pv_negative_pass', 'gross_pv_pass', 'overall_pass',
    'manual_exclusion',
]

ROOT_DIR = Path(__file__).parent.parent.parent
SITE_ID = SITE['edp_site_id']
csv_path = ROOT_DIR / 'data' / 'clean' / 'edp' / str(YEAR) / f'edp_clean_{SITE_ID}_{YEAR}.csv'
parquet_path = csv_path.with_suffix('.parquet')
meta_dir = ROOT_DIR / 'data' / 'processed' / 'h0001'
meta_path = meta_dir / f'{YEAR}_selected_sites_clean.csv'


def coverage(values):
    """Percent of intervals present, and longest gap in minutes."""
    missing = np.isnan(values)
    edges = np.diff(np.r_[0, missing.astype(int), 0])
    runs = np.where(edges == -1)[0] - np.where(edges == 1)[0]
    max_gap = float(runs.max() * 5) if len(runs) else 0.0
    return round(100 * (1 - missing.mean()), 1), max_gap


def corr(a, b, min_n):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) >= min_n else np.nan


if not csv_path.exists():
    print(f"ERROR: clean file not found: {csv_path}")
    sys.exit(1)

df = pd.read_csv(csv_path)
df['datetime_aest'] = pd.to_datetime(df['datetime_aest'], utc=True).dt.tz_convert('Etc/GMT-10')
if df[['pv_site_net_wh', 'ac_load_net_wh', 'gross_load_wh']].isna().any().any():
    print("WARNING: NaN values remain in the energy columns after infill")

# 1. Parquet
df.to_parquet(parquet_path, engine='pyarrow', compression='snappy', index=False)

# 2. Metadata
local = df['datetime_aest'].dt.tz_convert(SITE['site_timezone'])
hour = local.dt.hour.values
month = local.dt.month.values
pv = df['pv_original_wh'].values
ac = df['ac_load_net_original_wh'].values
solar = (hour >= SOLAR_HOUR_START) & (hour < SOLAR_HOUR_END)
both = ~np.isnan(pv) & ~np.isnan(ac)

row = dict(SITE)
row['ac_load_net_coverage_pct'], row['ac_load_net_max_gap_min'] = coverage(ac)
row['pv_site_net_coverage_pct'], row['pv_site_net_max_gap_min'] = coverage(pv)

# Check 1: PV vs ac_load_net correlation in solar hours, overall and worst month
sel = both & solar & (pv >= MIN_PV_THRESHOLD_WH)
month_corr = {m: corr(pv[sel & (month == m)], ac[sel & (month == m)], 10)
              for m in range(1, 13)}
month_corr = {m: c for m, c in month_corr.items() if not np.isnan(c)}
overall = corr(pv[sel], ac[sel], 10)
row['corr_coefficient'] = round(overall, 3) if not np.isnan(overall) else np.nan
worst = max(month_corr, key=month_corr.get) if month_corr else np.nan
row['corr_worst_month'] = worst
row['corr_worst_month_value'] = round(month_corr[worst], 3) if month_corr else np.nan

# Check 2: export exceeding PV
excess = np.where(both & (ac < 0), -ac - pv, -np.inf)
is_excess = excess > EXPORT_NOISE_WH
row['export_excess_intervals'] = int(is_excess.sum())
row['export_excess_max_w'] = round(float(excess[is_excess].max()) * 12, 1) if is_excess.any() else 0.0
row['export_excess_worst_month'] = (int(pd.Series(month[is_excess]).value_counts().idxmax())
                                    if is_excess.any() else np.nan)
row['gross_neg_fraction'] = round(float(((ac + pv)[both] < 0).mean()), 4)

# Checks 3 and 4: near-zero channels in solar hours
ac_solar = ~np.isnan(ac) & solar
pv_solar = ~np.isnan(pv) & solar
row['ac_flatline_fraction'] = round(float((np.abs(ac[ac_solar]) <= FLATLINE_NOISE_WH).mean()), 4)
row['pv_flatline_fraction'] = round(float((np.abs(pv[pv_solar]) <= FLATLINE_NOISE_WH).mean()), 4)

# Check 5: generation and load
row['annual_generation_kwh'] = round(float(np.nansum(pv)) / 1000, 1)
row['avg_daily_gross_load_wh'] = round(float((ac + pv)[both].mean()) * 288, 1)

# Check 6: negative PV in solar hours
row['pv_daytime_negative_fraction'] = round(float((pv[pv_solar] < PV_NEGATIVE_WH).mean()), 4)

# Check 7: gross load vs PV correlation in solar hours
sel7 = both & solar & (pv > 0)
g = corr(pv[sel7], (ac + pv)[sel7], 100)
row['gross_pv_corr'] = round(g, 3) if not np.isnan(g) else np.nan

row['corr_pass'] = bool(row['corr_coefficient'] <= CORR_THRESHOLD)
row['export_pass'] = bool(row['export_excess_intervals'] < EXPORT_COUNT_THRESHOLD
                          and row['gross_neg_fraction'] < GROSS_NEG_THRESHOLD)
row['ac_flatline_pass'] = bool(row['ac_flatline_fraction'] <= FLATLINE_THRESHOLD)
row['pv_flatline_pass'] = bool(row['pv_flatline_fraction'] <= FLATLINE_THRESHOLD)
row['generation_pass'] = bool(row['annual_generation_kwh'] >= MIN_ANNUAL_GENERATION_WH / 1000
                              and row['avg_daily_gross_load_wh'] >= MIN_AVG_DAILY_GROSS_LOAD_WH)
row['pv_negative_pass'] = bool(row['pv_daytime_negative_fraction'] <= PV_NEGATIVE_THRESHOLD)
row['gross_pv_pass'] = bool(np.isnan(row['gross_pv_corr'])
                            or row['gross_pv_corr'] < GROSS_PV_CORR_THRESHOLD)
checks = ['corr_pass', 'export_pass', 'ac_flatline_pass', 'pv_flatline_pass',
          'generation_pass', 'pv_negative_pass', 'gross_pv_pass']
row['overall_pass'] = all(row[c] for c in checks) and not row['manual_exclusion']

meta_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame([row])[COLUMNS].to_csv(meta_path, index=False)

print(f"D_Write_H0001 — year {YEAR}")
print(f"  Parquet:  {parquet_path}")
print(f"  Metadata: {meta_path}")
print(f"  annual_generation_kwh={row['annual_generation_kwh']}  "
      f"avg_daily_gross_load_wh={row['avg_daily_gross_load_wh']}")
failed = [c for c in checks if not row[c]]
print(f"  Checks failed: {failed if failed else 'none'}")
