"""
run_pipeline.py

Runs the full data preparation pipeline for a given year by calling
scripts A through L in sequence. Each script is called with the year
as a command-line argument.

If any script fails the pipeline stops immediately — subsequent scripts
should not run on data produced by a failed script.

Script K (K_Highlight_Marginal_Sites.py) runs after J and prints a
review of sites near each consistency check threshold. Review its output
before the pipeline proceeds to L.

Script L (L_Convert_To_Parquet.py) runs last, converting all per-site
clean CSV files to Parquet and removing the originals. After L completes,
only Parquet files remain in data/clean/YYYY/.

Preparation:
    rm data/clean/edp/2022/edp_clean_*.csv
    cd data/processed/edp
    cp 2022_selected_sites_clean.csv 2022_selected_sites_clean_prev.csv
    cp 2022_selected_sites_excluded.csv 2022_selected_sites_excluded_prev.csv
    cp 2022_consistency_checks.csv 2022_consistency_checks_prev.csv

Usage from project root:
    python3 -u pipeline/run_pipeline.py 2021 | tee logs/pipeline_2021.log
    python3 -u pipeline/run_pipeline.py 2022 | tee logs/pipeline_2022.log
    python3 -u pipeline/run_pipeline.py 2023 | tee logs/pipeline_2023.log
    python3 -u pipeline/run_pipeline.py 2024 | tee logs/pipeline_2024.log
    python3 -u pipeline/run_pipeline.py 2025 | tee logs/pipeline_2025.log

Logs:
    tail -f logs/pipeline_YYYY.log

"""

import os
import subprocess
import sys
import time
from pathlib import Path
 
# ── Year argument ──────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python3 -u pipeline/run_pipeline.py <YEAR>")
    print("Example: python3 -u pipeline/run_pipeline.py 2021")
    sys.exit(1)
 
YEAR = sys.argv[1]
try:
    int(YEAR)
except ValueError:
    print(f"Error: YEAR must be an integer, got '{YEAR}'")
    sys.exit(1)
 
# ── Pipeline scripts in order ──────────────────────────────────────────────────
SCRIPTS = [
    'A_Site_Summary.py',
    'B0_Battery_Site_Screen.py',
    'B_Site_Selection.py',
    'C_Check_Consistency.py',
    'D_Summarise_Site_Funnel.py',
    'E_Flag_All_Sites.py',
    'F_Site_Stats.py',
    'G_Infill_Short_All_Sites.py',
    'H_Build_PV_Profile.py',
    'I_Infill_Long_All_Sites.py',
    'J_Site_Stats.py',
    'K_Highlight_Marginal_Sites.py',
    'L_Convert_To_Parquet.py',
]
 
SCRIPTS_DIR = Path(__file__).parent / 'edp'
 
# ── Run pipeline ───────────────────────────────────────────────────────────────
t_pipeline = time.time()
print(f"\n{'='*60}")
print(f"  Pipeline starting for YEAR={YEAR}")
print(f"{'='*60}\n")
 
for script in SCRIPTS:
    script_path = SCRIPTS_DIR / script
    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        sys.exit(1)
 
    print(f"\n{'='*60}")
    print(f"  Running {script} (YEAR={YEAR})")
    print(f"{'='*60}\n")
 
    t_script = time.time()
    result = subprocess.run(
        [sys.executable, '-u', str(script_path), YEAR],
        check=False,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env={**os.environ, 'PYTHONUTF8': '1'},
    )
 
    elapsed = time.time() - t_script
    if result.returncode != 0:
        print(f"\n{'='*60}")
        print(f"  ERROR: {script} failed with return code {result.returncode}")
        print(f"  Pipeline stopped after {time.time()-t_pipeline:.0f}s")
        print(f"{'='*60}\n")
        sys.exit(result.returncode)
 
    print(f"\n  {script} completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")
 
# ── Done ───────────────────────────────────────────────────────────────────────
total = time.time() - t_pipeline
print(f"\n{'='*60}")
print(f"  Pipeline complete for YEAR={YEAR}")
print(f"  Total time: {total:.0f}s ({total/60:.1f} min)")
print(f"{'='*60}\n")