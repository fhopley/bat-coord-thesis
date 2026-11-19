"""
run_h0001.py

Builds the H0001 clean traces and metadata by running scripts A to D for each
year. Stops on the first failure.

Usage from project root:
    python3 pipeline/h0001/run_h0001.py             (2022 to 2025)
    python3 pipeline/h0001/run_h0001.py 2023        (one year)

Inputs:  data/raw/H0001/edp_raw_H0001_{YYYY}.csv
         data/processed/edp/{YYYY}_pv_profile_library.csv
Outputs: data/clean/edp/{YYYY}/edp_clean_H0001_{YYYY}.csv and .parquet
         data/processed/h0001/{YYYY}_selected_sites_clean.csv
"""

import subprocess
import sys
from pathlib import Path

YEARS = [int(sys.argv[1])] if len(sys.argv) > 1 else [2022, 2023, 2024, 2025]
SCRIPTS = [
    'A_Flag_H0001.py',
    'B_Infill_Short_H0001.py',
    'C_Infill_Long_H0001.py',
    'D_Write_H0001.py',
]
SCRIPTS_DIR = Path(__file__).parent

for year in YEARS:
    print(f"\n{'=' * 60}\n  H0001 {year}\n{'=' * 60}")
    for script in SCRIPTS:
        result = subprocess.run([sys.executable, '-u', str(SCRIPTS_DIR / script), str(year)])
        if result.returncode != 0:
            print(f"\nFAILED: {script} {year} (exit code {result.returncode})")
            sys.exit(result.returncode)
print("\nDone.")
