"""
nem_price_analysis.py

Estimates AGL's net cost per MW of 5–9 pm feedin capacity under the
29 c/kWh Battery Rewards Plan on a quarterly basis, expressed in $/MWh
to match ASX $300 cap contract quoting convention.

How ASX $300 cap contracts work
────────────────────────────────
Defined in asx_caps.py and described in docs/cap_hedge_spec.md. Each
contract is 1 MW base load for a calendar quarter, quoted in $/MWh. The
buyer receives Σ max(P − 300, 0) × interval hours over every interval in
the quarter, not max(avg_quarterly_spot − 300, 0).

AGL net cost in the same units
───────────────────────────────
For 1 MW of constant 5–9 pm export over a quarter:
  MWh exported = 4 hrs/day × days_in_quarter
  AGL pays     = MWh_exported × $290/MWh
  WS value     = MWh_exported × avg_rrp_5to9pm
  Net cost ($) = AGL pays − WS value

To express as $/MWh in the same way ASX quotes caps, we divide by the
quarter's BASE load contract size (1 MW × all hours), not by MWh
exported. This puts AGL's cost on the same denominator as the cap price:

  AGL net cost ($/MWh, cap-equivalent)
      = net_cost ($) / (days_in_quarter × 24 hrs)

This is directly comparable to the ASX cap quoted price.

Run from repo root: python3 analysis/scripts/nem_price_analysis.py

Outputs:
  analysis/output/agl_cap_quarterly_chart.png  — time series vs forward curve
  analysis/output/agl_cap_quarterly_table.png  — detailed table
  Console: quarterly table by state
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR   = SCRIPT_DIR.parent.parent
CLEAN_DIR  = ROOT_DIR / "data" / "clean" / "nem_prices"
OUTPUT_DIR = ROOT_DIR / "analysis" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────────────
YEARS   = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
REGIONS = ['NSW1', 'QLD1', 'SA1', 'VIC1']
REGION_LABELS = {'NSW1': 'NSW', 'QLD1': 'QLD', 'SA1': 'SA', 'VIC1': 'VIC'}

AGL_FEEDIN_RATE = 290.0  # $/MWh (29 c/kWh)
PEAK_HOURS      = set(range(17, 21))

from asx_caps import (
    CAP_STRIKE, ASX_NSW_CAP, ASX_QLD_CAP, ASX_SA_CAP, ASX_VIC_CAP,
    FIRST_FORWARD_QUARTER,
)

# ── Style ──────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    'font.family':       'serif',
    'font.size':         10,
    'axes.titlesize':    11,
    'axes.labelsize':    10,
    'legend.fontsize':   9,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'figure.dpi':        150,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.35,
    'grid.linewidth':    0.5,
    'grid.linestyle':    '--',
})

STATE_COLOURS = {
    'NSW1': '#1f77b4',
    'QLD1': '#d62728',
    'SA1':  '#ff7f0e',
    'VIC1': '#2ca02c',
}

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading parquet files...")
dfs = []
for year in YEARS:
    p = CLEAN_DIR / f'{year}_nem_prices.parquet'
    if not p.exists():
        print(f"  MISSING: {p} — skipping")
        continue
    df_yr = pd.read_parquet(p)
    if 'datetime_aest' in df_yr.columns:
        df_yr = df_yr.set_index('datetime_aest')
    if df_yr.index.tz is None:
        df_yr.index = df_yr.index.tz_localize('Etc/GMT-10')
    dfs.append(df_yr)

df = pd.concat(dfs).sort_index()
df = df[df['price_flag'] == 0]
df['hour']    = df.index.hour
df['in_peak'] = df['hour'].isin(PEAK_HOURS)
df['year']    = df.index.year
df['quarter'] = df.index.quarter

# ── Compute per-quarter, per-region metrics ────────────────────────────────────
print("Computing quarterly metrics...")
records = []

for (year, quarter), grp in df.groupby(['year', 'quarter']):
    grp_peak = grp[grp['in_peak']]
    n_days    = grp.index.normalize().nunique()

    # Base load contract size for this quarter: 1 MW × all hours
    contract_mwh = float(n_days * 24)   # MWh per MW of capacity (= contract size denominator)

    # MWh exported by 1 MW at 4 hrs/day during peak window
    mwh_exported = 4.0 * n_days

    for region in REGIONS:
        col = f'{region}_rrp'
        if col not in grp.columns:
            continue

        avg_peak_rrp  = grp_peak[col].mean()
        agl_pays      = mwh_exported * AGL_FEEDIN_RATE
        wholesale_val = mwh_exported * avg_peak_rrp
        net_cost      = agl_pays - wholesale_val   # $ per MW per quarter

        # Express as $/MWh using same denominator as ASX cap price
        # = net_cost / (days × 24 hrs) — i.e. per MWh of 24hr capacity
        net_cost_per_mwh = net_cost / contract_mwh

        # Cap value fraction for this quarter
        excess_total = (grp[col] - CAP_STRIKE).clip(lower=0).sum()
        excess_peak  = (grp.loc[grp['in_peak'], col] - CAP_STRIKE).clip(lower=0).sum()
        cap_frac     = (excess_peak / excess_total) if excess_total > 0 else np.nan

        # ASX NSW cap price — historical at-start-of-quarter or forward contract
        asx_fwd = ASX_NSW_CAP.get((year, quarter)) if region == 'NSW1' else None

        # Quarter label for display
        q_labels = {1: 'Q1 (J–M)', 2: 'Q2 (A–J)', 3: 'Q3 (J–S)', 4: 'Q4 (O–D)'}
        quarter_label = f"{year} {q_labels[quarter]}"

        records.append({
            'year':              year,
            'quarter':           quarter,
            'quarter_label':     quarter_label,
            'period_sort':       year * 10 + quarter,
            'region':            region,
            'n_days':            n_days,
            'contract_mwh':      contract_mwh,
            'mwh_exported':      mwh_exported,
            'avg_peak_rrp':      avg_peak_rrp,
            'agl_pays':          agl_pays,
            'wholesale_val':     wholesale_val,
            'net_cost':          net_cost,
            'net_cost_per_mwh':  net_cost_per_mwh,
            'cap_frac':          cap_frac,
            'asx_fwd_per_mwh':   asx_fwd,
        })

results = pd.DataFrame(records).sort_values('period_sort')

# ── Console output ─────────────────────────────────────────────────────────────
print()
for region in REGIONS:
    sub = results[results['region'] == region]
    print(f"\n{'─'*85}")
    print(f"  {REGION_LABELS[region]}  —  1 MW constant 5–9 pm export, quarterly")
    print(f"{'─'*85}")
    print(f"  {'Quarter':<18} {'Days':>4} {'Avg RRP':>8} {'AGL pays':>10} "
          f"{'WS value':>10} {'Net cost':>10} {'Net $/MWh':>10} "
          f"{'Cap frac':>9} {'ASX fwd':>8}")
    print(f"  {'':18} {'':4} {'$/MWh':>8} {'$/qtr':>10} {'$/qtr':>10} "
          f"{'$/qtr':>10} {'(cap eqv)':>10} {'5–9pm':>9} {'$/MWh':>8}")
    print(f"  {'-'*88}")
    for _, row in sub.iterrows():
        asx_str = f"{row['asx_fwd_per_mwh']:>7.2f}" if row['asx_fwd_per_mwh'] is not None else "     —"
        cf_str  = f"{row['cap_frac']*100:>7.1f}%" if not np.isnan(row['cap_frac']) else "      —"
        print(
            f"  {row['quarter_label']:<18} {int(row['n_days']):>4} "
            f"{row['avg_peak_rrp']:>8.1f} "
            f"${row['agl_pays']:>9,.0f} "
            f"${row['wholesale_val']:>9,.0f} "
            f"${row['net_cost']:>9,.0f} "
            f"{row['net_cost_per_mwh']:>10.2f} "
            f"{cf_str} "
            f"{asx_str}"
        )
print(f"\n  Net $/MWh = quarterly net cost ÷ (days × 24 hrs), same denominator as ASX cap price.")
print(f"  ASX fwd = current NSW quarterly $300 cap forward prices ($/MWh) from ASX24, June 2026.")
print(f"  Positive net cost = AGL paid more than wholesale value of the export.")

# ── Figure 1: time-series by state ───────────────────────────────────────────
# ASX cap dicts for each region — add ASX_QLD_CAP, ASX_SA_CAP, ASX_VIC_CAP to
# the configuration section above and they will automatically plot here.
ASX_CAP_BY_REGION = {
    'NSW1': ASX_NSW_CAP,
}
for _name, _key in [('ASX_QLD_CAP', 'QLD1'), ('ASX_SA_CAP', 'SA1'), ('ASX_VIC_CAP', 'VIC1')]:
    if _name in dir():
        ASX_CAP_BY_REGION[_key] = eval(_name)

print("\nGenerating charts...")

fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
axes = axes.flatten()

# Historical quarters only — x positions and year-only tick labels
all_quarters = (results[['period_sort', 'quarter_label']]
                .drop_duplicates()
                .sort_values('period_sort'))
q_x = {row['period_sort']: i for i, (_, row) in enumerate(all_quarters.iterrows())}

# Year tick positions: place at first quarter of each year, label = year only
year_tick_positions, year_tick_labels = [], []
prev_yr = None
for ps, xi_ in q_x.items():
    yr = ps // 10
    if yr != prev_yr:
        year_tick_positions.append(xi_)
        year_tick_labels.append(str(yr))
        prev_yr = yr

for ax, region in zip(axes, ['NSW1', 'QLD1', 'SA1', 'VIC1']):
    sub   = results[results['region'] == region].sort_values('period_sort')
    color = STATE_COLOURS[region]

    xi = [q_x[ps] for ps in sub['period_sort']]
    yi = sub['net_cost_per_mwh'].values

    # AGL net cost line
    ax.plot(xi, yi, color=color, linewidth=1.8, marker='o', markersize=4,
            label='AGL net cost/MWh', zorder=3)
    ax.axhline(0, color='grey', linewidth=0.6, zorder=0)

    # ASX cap price series for this region (historical quarters only)
    asx_dict = ASX_CAP_BY_REGION.get(region)
    if asx_dict:
        asx_hist = {k: v for k, v in asx_dict.items() if k < FIRST_FORWARD_QUARTER}
        asx_xi = [q_x[yr * 10 + q] for yr, q in sorted(asx_hist)
                  if yr * 10 + q in q_x]
        asx_yi = [asx_hist[(yr, q)] for yr, q in sorted(asx_hist)
                  if yr * 10 + q in q_x]
        if asx_xi:
            ax.plot(asx_xi, asx_yi, color='black', linewidth=1.6, marker='s',
                    markersize=4, linestyle='-',
                    label=f'ASX {REGION_LABELS[region]} $300 cap ($/MWh)', zorder=4)

    # Year-only x-axis
    ax.set_xticks(year_tick_positions)
    ax.set_xticklabels(year_tick_labels, fontsize=9)

    ax.set_title(REGION_LABELS[region], fontweight='bold')
    ax.set_ylabel('$/MWh (cap-equivalent)')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'${v:.0f}'))
    ax.legend(fontsize=7.5, framealpha=0.9, loc='upper right')

    # Year separator lines
    prev_yr = None
    for ps, xi_ in q_x.items():
        yr = ps // 10
        if yr != prev_yr:
            if prev_yr is not None:
                ax.axvline(xi_ - 0.5, color='lightgrey', linewidth=0.8, zorder=0)
            prev_yr = yr

fig.suptitle(
    'AGL Battery Rewards Plan — Quarterly Net Cost vs ASX $300 Cap Price\n'
    '(1 MW constant export, 5–9 pm; $/MWh matches ASX cap quoting convention)',
    fontweight='bold', fontsize=11
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out1 = OUTPUT_DIR / 'agl_cap_quarterly_chart.png'
fig.savefig(out1, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {out1}")

# ── Figure 2: styled quarterly table ─────────────────────────────────────────
print("Generating table figure...")

# NSW only — single unified row list: historical data + forward-only future quarters
sub_nsw = results[results['region'] == 'NSW1'].sort_values('period_sort')

q_labels_map = {1: 'Q1 (J–M)', 2: 'Q2 (A–J)', 3: 'Q3 (J–S)', 4: 'Q4 (O–D)'}

# Historical quarters (from price data)
display_rows = []
for _, row in sub_nsw.iterrows():
    display_rows.append({
        'quarter_label':    row['quarter_label'],
        'avg_peak_rrp':     row['avg_peak_rrp'],
        'net_cost_per_mwh': row['net_cost_per_mwh'],
        'cap_frac':         row['cap_frac'],
        'asx_fwd_per_mwh':  row['asx_fwd_per_mwh'],
        'is_forward':       False,
    })

# Forward-only quarters (no price data, just ASX contract price)
for (yr, q) in sorted(k for k in ASX_NSW_CAP if k >= FIRST_FORWARD_QUARTER):
    # Skip if already covered by historical data rows
    if any(r['quarter_label'] == f"{yr} {q_labels_map[q]}" for r in display_rows):
        continue
    display_rows.append({
        'quarter_label':    f"{yr} {q_labels_map[q]}",
        'avg_peak_rrp':     np.nan,
        'net_cost_per_mwh': np.nan,
        'cap_frac':         np.nan,
        'asx_fwd_per_mwh':  ASX_NSW_CAP[(yr, q)],
        'is_forward':       True,
    })

col_labels = ['Quarter', 'Avg RRP 5–9pm\n($/MWh)', 'AGL net cost\n($/MWh cap-equiv)',
              '% cap value\nin 5–9 pm', 'ASX NSW $300\ncap ($/MWh)']

table_data = []
row_is_fwd = []
for r in display_rows:
    def f(v, fmt): return fmt.format(v) if (v is not None and not (isinstance(v, float) and np.isnan(v))) else '—'
    table_data.append([
        r['quarter_label'],
        f(r['avg_peak_rrp'],     '{:.0f}'),
        f(r['net_cost_per_mwh'], '{:.2f}'),
        f(r['cap_frac'],         '{:.0%}') if not (isinstance(r['cap_frac'], float) and np.isnan(r['cap_frac'])) else '—',
        f(r['asx_fwd_per_mwh'],  '{:.2f}'),
    ])
    row_is_fwd.append(r['is_forward'])

# ── Averages row ──────────────────────────────────────────────────────────────
# Average each numeric column, ignoring '—' cells.
def _col_avg(col_idx):
    vals = []
    for row in table_data:
        v = row[col_idx]
        if v != '—':
            try:
                vals.append(float(v.strip('%').replace(',', '')))
            except ValueError:
                pass
    return sum(vals) / len(vals) if vals else None

avg_rrp     = _col_avg(1)
avg_net     = _col_avg(2)
avg_capfrac = _col_avg(3)
avg_asx     = _col_avg(4)

def _fmt_avg(v, fmt): return fmt.format(v) if v is not None else '—'

table_data.append([
    'Average',
    _fmt_avg(avg_rrp,     '{:.0f}'),
    _fmt_avg(avg_net,     '{:.2f}'),
    _fmt_avg(avg_capfrac, '{:.0f}%'),   # already in %, was stripped of %
    _fmt_avg(avg_asx,     '{:.2f}'),
])
row_is_fwd.append(False)   # style it like a normal row; we'll bold it separately

n_rows = len(table_data)
fig_h  = max(8, n_rows * 0.30 + 1.5)
fig2, ax = plt.subplots(figsize=(10, fig_h))
ax.axis('off')

tbl = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    loc='center',
    cellLoc='center',
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.auto_set_column_width(list(range(len(col_labels))))

# Header row needs extra height for two-line labels
for j in range(len(col_labels)):
    tbl[0, j].set_height(0.08)

# Header
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor('#2c3e50')
    tbl[0, j].set_text_props(color='white', fontweight='bold')

# Row shading
for i, is_fwd in enumerate(row_is_fwd):
    yr_str = table_data[i][0][:4]
    if is_fwd:
        bg = '#f0f0f0'
        for j in range(len(col_labels)):
            tbl[i + 1, j].set_facecolor(bg)
            tbl[i + 1, j].set_text_props(color='dimgrey', fontstyle='italic')
    else:
        # Alternate by year (skip for the Average row)
        try:
            yr_idx = YEARS.index(int(yr_str)) if int(yr_str) in YEARS else 0
        except ValueError:
            yr_idx = 0
        bg = '#dce9f5' if yr_idx % 2 == 0 else 'white'
        for j in range(len(col_labels)):
            tbl[i + 1, j].set_facecolor(bg)

# Separator line before forward rows
first_fwd = next((i for i, f in enumerate(row_is_fwd) if f), None)
if first_fwd is not None:
    for j in range(len(col_labels)):
        tbl[first_fwd, j].set_edgecolor('dimgrey')

# Average row — dark header-style shading, bold text
avg_row_i = len(table_data)   # last row (1-indexed in tbl = avg_row_i)
for j in range(len(col_labels)):
    tbl[avg_row_i, j].set_facecolor('#2c3e50')
    tbl[avg_row_i, j].set_text_props(color='white', fontweight='bold')

fig2.suptitle(
    'AGL Battery Rewards Plan — NSW Quarterly Cost Analysis\n'
    '(Historical net cost vs ASX $300 cap forward curve; $/MWh cap-equivalent)',
    fontweight='bold', fontsize=10, y=0.99
)
fig2.tight_layout(rect=[0, 0, 1, 0.97])
out2 = OUTPUT_DIR / 'agl_cap_quarterly_table.png'
fig2.savefig(out2, bbox_inches='tight')
plt.close(fig2)
print(f"  Saved: {out2}")
print("\nDone.")