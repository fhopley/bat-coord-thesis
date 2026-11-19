"""
reference_household.py

Annual gross load, PV system size and annual PV generation of the weighted
reference household quoted in Methodology, Representativeness and Weighting.

Uses the same weighting as financial_analysis.py: per-site PV size band weights
within state and year, a straight mean across years, then the state_mix weights
from analysis/config/pv_size_bands.yaml.  Site load and generation come from
data/processed/edp/YYYY_selected_sites_clean.csv, so no run data is needed.

    python3 analysis/scripts/reference_household.py

Writes analysis/output/reference_household.csv (per state and year) and prints
the national figures.
"""
import sys, warnings
from pathlib import Path
import pandas as pd, numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'analysis'/'scripts'))
warnings.filterwarnings('ignore')
import financial_analysis as fa

cfg = fa.load_weights_config(ROOT/'analysis'/'config'/'pv_size_bands.yaml')
bands, by_state, state_mix = cfg['bands'], cfg['by_state'], cfg['state_mix']
years = ['2021','2022','2023','2024','2025']
states = ['NSW','QLD','SA','VIC']

rows=[]
for year in years:
    full = pd.read_csv(ROOT/'data'/'processed'/'edp'/f'{year}_selected_sites_clean.csv')
    full = full.rename(columns={'edp_site_id':'site_id'})
    extra = full.set_index('site_id')[['avg_daily_gross_load_wh','annual_generation_kwh']]
    allmeta = pd.concat([fa.load_site_meta(ROOT, year, s).assign(state=s) for s in states])
    ratio = fa.compute_dc_ac_ratio(allmeta)
    pooled = fa.assign_direct_size(allmeta, ratio)
    for s in states:
        m = fa.load_site_meta(ROOT, year, s)
        m = fa.assign_direct_size(m, ratio)
        m = fa.estimate_missing_sizes(m, s, pooled)
        w, _ = fa.assign_bands_and_weights(m, s, bands, by_state)
        m = m.set_index('site_id')
        w = w.reindex(m.index).fillna(0.0)
        if w.sum() == 0: continue
        load = extra['avg_daily_gross_load_wh'].reindex(m.index)*365/1000.0
        gen  = extra['annual_generation_kwh'].reindex(m.index)
        ok = load.notna() & m['size_kw'].notna() & (w>0)
        rows.append(dict(year=year, state=s, n=int(ok.sum()),
            load=float(np.average(load[ok], weights=w[ok])),
            pv=float(np.average(m['size_kw'][ok], weights=w[ok])),
            gen=float(np.average(gen[ok].fillna(0), weights=w[ok]))))
df = pd.DataFrame(rows)
per_state = df.groupby('state')[['load','pv','gen']].mean()
mix = pd.Series(state_mix)
nat = (per_state.T*mix).T.sum()/mix.sum()
pd.set_option('display.width',200)
out = ROOT/'analysis'/'output'/'reference_household.csv'
df.to_csv(out, index=False)
print(df.to_string(index=False))
print(f'\nwrote {out}')
print('\nper-state mean across years:\n', per_state.round(0).to_string())
print('\nNATIONAL (state_mix weighted):')
print(f"  annual gross load  {nat['load']:,.0f} kWh")
print(f"  PV system size     {nat['pv']:.1f} kW")
print(f"  annual PV generation {nat['gen']:,.0f} kWh")
