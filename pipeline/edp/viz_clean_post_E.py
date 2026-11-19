"""
viz_clean_post_E.py

Interactive daily plot tool for flagged and infilled clean site data.
Reads from data/clean/edp/YYYY/edp_clean_{SITE_ID}_{YYYY}.csv and plots
pv_site_net, ac_load_net, and gross_load in watts.

Flagged intervals are marked with coloured circle (PV) or square (load)
markers. Infilled intervals are additionally marked with a triangle overlay
to distinguish synthetic/interpolated values from original measurements:
  - Upward triangle (^): Method 1 infill (linear interpolation)
  - Downward triangle (v): Method 2 infill (synthetic day profile / load)

On startup prints a flag and infill summary by month, and a list of
flagged/infilled date ranges to help navigate to dates of interest.

Usage:
    python3 pipeline/edp/viz_clean_post_E.py
    Enter a date (YYYY-MM-DD) to plot, or Q to quit.
    Press Enter to repeat the last date.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from pathlib import Path

# ── Settings ───────────────────────────────────────────────────────────────────
SITE_ID = 'S0008'
YEAR    = 2022

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR   = SCRIPT_DIR.parent.parent
clean_file = ROOT_DIR / "data" / "clean" / "edp" / str(YEAR) / \
             f"edp_clean_{SITE_ID}_{YEAR}.csv"

# ── Flag and infill metadata ───────────────────────────────────────────────────
FLAG_NAMES  = {0:'clean', 1:'missing', 2:'flatline', 3:'DST gap',
               4:'partial phase', 5:'invalid', 6:'too large'}
FLAG_COLORS = {0:'none',  1:'lightgrey', 2:'orange', 3:'purple',
               4:'cyan',  5:'red',       6:'darkred'}

INFILL_NAMES   = {1:'Method 1 (interpolated)', 2:'Method 2 (synthetic)'}
INFILL_MARKERS = {1:'^', 2:'v'}  # up triangle = M1, down triangle = M2
INFILL_COLOR   = 'black'

# ── Load data ──────────────────────────────────────────────────────────────────
print(f"Loading {clean_file.name}...")
df = pd.read_csv(clean_file, parse_dates=['datetime_aest'])
df['datetime_aest'] = (pd.to_datetime(df['datetime_aest'], utc=True)
                         .dt.tz_convert('Australia/Brisbane'))

if 'pv_infill' not in df.columns:
    df['pv_infill'] = 0
if 'load_infill' not in df.columns:
    df['load_infill'] = 0
if 'pv_original_wh' not in df.columns:
    df['pv_original_wh'] = np.nan
if 'ac_load_net_original_wh' not in df.columns:
    df['ac_load_net_original_wh'] = np.nan

# Convert Wh to W
df['pv_w']           = df['pv_site_net_wh'] * 12
df['load_w']         = df['ac_load_net_wh'] * 12
df['gross_w']        = df['gross_load_wh']  * 12
df['pv_orig_w']      = df['pv_original_wh'] * 12
df['load_orig_w']    = df['ac_load_net_original_wh'] * 12
df['plot_time'] = df['datetime_aest'].dt.tz_localize(None)

df['date']  = df['datetime_aest'].dt.date
df['month'] = df['datetime_aest'].dt.month
available_dates = sorted(df['date'].unique())

# ── Startup summary ────────────────────────────────────────────────────────────
print(f"\n{'─'*75}")
print(f"  {SITE_ID} {YEAR} — flag and infill summary by month")
print(f"{'─'*75}")
print(f"  {'Month':<8}  {'PV flags':<28}  {'Load flags':<28}  {'Infill'}")
print(f"  {'─'*7}  {'─'*28}  {'─'*28}  {'─'*20}")

for month in range(1, 13):
    mdf = df[df['month'] == month]
    if mdf.empty:
        continue
    month_name = pd.Timestamp(f'{YEAR}-{month:02d}-01').strftime('%b')

    def flag_str(flags):
        parts = [f"F{c}:{(flags==c).sum()}"
                 for c in sorted(flags.unique()) if c != 0]
        return '  '.join(parts) if parts else '—'

    def infill_str(pv_inf, load_inf):
        parts = []
        for code in [1, 2]:
            pv_n   = (pv_inf == code).sum()
            load_n = (load_inf == code).sum()
            if pv_n > 0 or load_n > 0:
                parts.append(f"M{code}:PV={pv_n}/Ld={load_n}")
        return '  '.join(parts) if parts else '—'

    print(f"  {month_name:<8}  {flag_str(mdf['pv_flag']):<28}  "
          f"{flag_str(mdf['load_flag']):<28}  "
          f"{infill_str(mdf['pv_infill'], mdf['load_infill'])}")

# ── Flagged/infilled date ranges ───────────────────────────────────────────────
print(f"\n{'─'*75}")
print(f"  Flagged or infilled date ranges")
print(f"{'─'*75}")

interest_dates = sorted(df[
    (df['pv_flag'] > 0) | (df['load_flag'] > 0) |
    (df['pv_infill'] > 0) | (df['load_infill'] > 0)
]['date'].unique())

if interest_dates:
    ranges = []
    rs, re = interest_dates[0], interest_dates[0]
    for d in interest_dates[1:]:
        if (pd.Timestamp(d) - pd.Timestamp(re)).days == 1:
            re = d
        else:
            ranges.append((rs, re))
            rs = re = d
    ranges.append((rs, re))

    for start, end in ranges:
        rdf = df[(df['date'] >= start) & (df['date'] <= end)]
        pv_flags   = [f"F{c}" for c in sorted(rdf['pv_flag'].unique())   if c != 0]
        load_flags = [f"F{c}" for c in sorted(rdf['load_flag'].unique()) if c != 0]
        pv_inf     = [f"M{c}" for c in sorted(rdf['pv_infill'].unique()) if c != 0]
        load_inf   = [f"M{c}" for c in sorted(rdf['load_infill'].unique()) if c != 0]
        notes = []
        if pv_flags or load_flags:
            notes.append(
                f"Flags PV:[{','.join(pv_flags)}] Ld:[{','.join(load_flags)}]"
            )
        if pv_inf or load_inf:
            notes.append(
                f"Infill PV:[{','.join(pv_inf)}] Ld:[{','.join(load_inf)}]"
            )
        note = '  '.join(notes)
        if start == end:
            print(f"  {start}              {note}")
        else:
            n_days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
            print(f"  {start} → {end}  ({n_days}d)  {note}")
else:
    print("  No flagged or infilled intervals found.")

print(f"{'─'*75}\n")

# ── Plot function ──────────────────────────────────────────────────────────────
def plot_day(date_str):
    try:
        date = pd.to_datetime(date_str).date()
    except Exception:
        print(f"  Could not parse '{date_str}'. Use YYYY-MM-DD format.")
        return

    day_df = df[df['date'] == date].sort_values('datetime_aest')
    if day_df.empty:
        print(f"  No data for {date_str}.")
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    # Main traces
    ax.plot(day_df['plot_time'], day_df['pv_w'],
            color='gold', linewidth=1.5, label='pv_site_net', zorder=2)
    ax.plot(day_df['plot_time'], day_df['load_w'],
            color='steelblue', linewidth=1.5, label='ac_load_net', zorder=2)
    ax.plot(day_df['plot_time'], day_df['gross_w'],
            color='coral', linewidth=1.5, linestyle='--',
            label='gross_load', zorder=2)

    # Flag markers — circle=PV, square=load
    for flag_code in range(1, 7):
        color = FLAG_COLORS[flag_code]
        pv_f  = day_df[day_df['pv_flag']   == flag_code]
        ld_f  = day_df[day_df['load_flag'] == flag_code]
        if not pv_f.empty:
            ax.scatter(pv_f['plot_time'], pv_f['pv_w'].fillna(0),
                       color=color, s=40, zorder=5,
                       marker='o', edgecolors='black', linewidths=0.5)
        if not ld_f.empty:
            ax.scatter(ld_f['plot_time'], ld_f['load_w'].fillna(0),
                       color=color, s=40, zorder=5,
                       marker='s', edgecolors='black', linewidths=0.5)

    # Infill markers — triangle overlay showing method used
    for method_code, marker in INFILL_MARKERS.items():
        pv_i  = day_df[day_df['pv_infill']   == method_code]
        ld_i  = day_df[day_df['load_infill'] == method_code]
        if not pv_i.empty:
            ax.scatter(pv_i['plot_time'], pv_i['pv_w'].fillna(0),
                       color=INFILL_COLOR, s=25, zorder=6,
                       marker=marker, alpha=0.6)
        if not ld_i.empty:
            ax.scatter(ld_i['plot_time'], ld_i['load_w'].fillna(0),
                       color=INFILL_COLOR, s=25, zorder=6,
                       marker=marker, alpha=0.6)

    # Original values — show faded X markers where data was replaced by infill
    # This shows what the bad value was before replacement
    pv_replaced   = day_df[(day_df['pv_infill'] > 0) &
                            day_df['pv_orig_w'].notna()]
    load_replaced = day_df[(day_df['load_infill'] > 0) &
                            day_df['load_orig_w'].notna()]
    if not pv_replaced.empty:
        ax.scatter(pv_replaced['plot_time'], pv_replaced['pv_orig_w'],
                   color='gold', s=30, zorder=4,
                   marker='x', linewidths=1.5, alpha=0.5,
                   label='_nolegend_')
    if not load_replaced.empty:
        ax.scatter(load_replaced['plot_time'], load_replaced['load_orig_w'],
                   color='steelblue', s=30, zorder=4,
                   marker='x', linewidths=1.5, alpha=0.5,
                   label='_nolegend_')

    ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.set_xlabel('Time (AEST)')
    ax.set_ylabel('Power (W)')
    ax.grid(True, alpha=0.3)

    # Legend
    trace_handles = [
        mpatches.Patch(color='gold',      label='pv_site_net'),
        mpatches.Patch(color='steelblue', label='ac_load_net'),
        mpatches.Patch(color='coral',     label='gross_load'),
    ]
    flag_handles = [
        mpatches.Patch(color=FLAG_COLORS[c],
                       label=f"F{c}: {FLAG_NAMES[c]}")
        for c in range(1, 7)
        if (day_df['pv_flag'] == c).any() or (day_df['load_flag'] == c).any()
    ]
    infill_handles = [
        mlines.Line2D([], [], color=INFILL_COLOR, marker=INFILL_MARKERS[c],
                      linestyle='None', markersize=7, alpha=0.6,
                      label=INFILL_NAMES[c])
        for c in [1, 2]
        if (day_df['pv_infill'] == c).any() or
           (day_df['load_infill'] == c).any()
    ]
    # Add original value marker to legend if any replaced values exist
    has_orig = ((day_df['pv_infill'] > 0) & day_df['pv_orig_w'].notna()).any() or \
               ((day_df['load_infill'] > 0) & day_df['load_orig_w'].notna()).any()
    if has_orig:
        infill_handles.append(
            mlines.Line2D([], [], color='grey', marker='x',
                          linestyle='None', markersize=7, alpha=0.6,
                          label='original (before infill)')
        )
    ax.legend(handles=trace_handles + flag_handles + infill_handles,
              loc='upper left', fontsize=8)

    n_pv_flag  = (day_df['pv_flag']   > 0).sum()
    n_ld_flag  = (day_df['load_flag'] > 0).sum()
    n_pv_inf   = (day_df['pv_infill'] > 0).sum()
    n_ld_inf   = (day_df['load_infill'] > 0).sum()
    ax.set_title(
        f"{SITE_ID}  —  {date_str} (AEST)\n"
        f"Flagged: PV={n_pv_flag} Load={n_ld_flag} | "
        f"Infilled: PV={n_pv_inf} Load={n_ld_inf}",
        fontsize=11
    )

    plt.tight_layout()
    plt.show()

# ── Input loop ─────────────────────────────────────────────────────────────────
print("Enter a date to plot (YYYY-MM-DD), or Q to quit.")
print("Press Enter to advance to the next day.")
current_date = str(available_dates[0])

while True:
    next_date = str((pd.Timestamp(current_date) +
                     pd.Timedelta(days=1)).date())
    # Cap at last available date
    last_date = str(available_dates[-1])
    if next_date > last_date:
        next_date = last_date
    date_input = input(f"Date [{next_date}]: ").strip()
    if date_input.upper() == 'Q':
        print("Exiting.")
        break
    if date_input == '':
        date_input = next_date
    current_date = date_input
    plot_day(current_date)
    print()