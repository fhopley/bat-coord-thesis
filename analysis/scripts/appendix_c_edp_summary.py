"""
appendix_c_edp_summary.py

Appendix C (EDP data pipeline) tables and figures, for NSW, QLD, SA and VIC,
2021-2025, from the per-year pipeline outputs in data/processed/edp/
(docs/edp_data_spec.md). Manual exclusions are taken from
data/raw/edp/manual_exclusions.csv for every year listed there, whether or
not the pipeline run applied them.

Tables (printed as markdown, and written as CSV to analysis/output/appendix_c/):
  site_selection.csv     coverage, inverter and battery filters, each alone
  consistency_checks.csv consistency checks, each alone, and manual exclusions
  interval_flags.csv     interval flags and infill, validated sites
  validated_by_state.csv validated sites by state and year

Figures (thesis/figures/appendix_c/):
  01_validated_sites_by_state.png  Figure C.1
  04_annual_load_vs_generation.png Figure C.4, from YYYY_site_annual_totals.csv
                                   (edp_annual_totals.py, run on the clean
                                   parquet files), read from the processed
                                   or raw EDP directory

  python3 analysis/scripts/appendix_c_edp_summary.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import thesis_style as ts  # noqa: E402
from site_exclusions import EXCLUDED_SITES  # noqa: E402

YEARS = [2021, 2022, 2023, 2024, 2025]
STATES = ['NSW', 'QLD', 'SA', 'VIC']

# B_Site_Selection.py thresholds
MIN_COVERAGE_PCT = 85.0
MAX_COVERAGE_PCT = 100.5
MIN_INVERTER_KW = 1.0

# C_Check_Consistency.py pass columns, by check number in Table C.2
CHECK_COLUMNS = {
    1: 'corr_pass',
    2: 'export_pass',
    3: 'ac_flatline_pass',
    4: 'pv_flatline_pass',
    5: 'generation_pass',
    6: 'pv_negative_pass',
    7: 'gross_pv_pass',
}

# Figure C.4 axis bands (kWh/yr): the lower band holds the population, its
# limits set from the data in steps of AXIS_STEP; the upper band holds the
# non-residential sites. X_HEADROOM leaves the right of the plot clear for
# the legend.
AXIS_STEP = 5_000
X_HEADROOM = 1.2
UPPER_LIM = (85_000, 120_000)
UPPER_TICKS = [90_000, 110_000]
NONRES_COLOUR = ts.CATEGORICAL_COLOURS[1]
FIG4_HEIGHT_IN = 3.0


# ---- Data ----------------------------------------------------------------

def load_metadata(raw_dir: Path) -> pd.DataFrame:
    """Site metadata deduplicated as in B_Site_Selection.py."""
    m = pd.read_csv(raw_dir / 'edp_sites_metadata_2020_2025.csv')
    m['islandable'] = m['islandable'].str.strip()
    m = (m.sort_values('inverter_ac_rating_kw', ascending=False)
          .drop_duplicates(subset='edp_site_id', keep='first'))
    m['has_battery_flag'] = ((m['islandable'] == 'Yes')
                             | (m['has_battery'] == 't')
                             | m['battery_size_make_model'].notna())
    return m[['edp_site_id', 'state', 'inverter_ac_rating_kw',
              'has_battery_flag']]


def manual_exclusions(raw_dir: Path) -> dict[int, set[str]]:
    m = pd.read_csv(raw_dir / 'manual_exclusions.csv')
    return {y: set(g['edp_site_id']) for y, g in m.groupby('year')}


def site_selection(proc: Path, meta: pd.DataFrame, year: int) -> pd.DataFrame:
    """One row per raw site in STATES, with each filter's outcome."""
    cov = pd.read_csv(proc / f'{year}_site_coverage.csv')
    screen = pd.read_csv(proc / f'{year}_battery_site_screen.csv',
                         usecols=['edp_site_id', 'category', 'usable'])
    d = cov.merge(meta, on='edp_site_id', how='inner')
    d = d.merge(screen, on='edp_site_id', how='left')
    d = d[d['state'].isin(STATES)].copy()
    d['usable'] = d['usable'].eq(True)
    d['battery_evidence'] = d['has_battery_flag'] | d['category'].notna()
    d['fail_load_cov'] = d['ac_load_net_coverage_pct'] < MIN_COVERAGE_PCT
    d['fail_pv_cov'] = d['pv_site_net_coverage_pct'] < MIN_COVERAGE_PCT
    d['fail_inverter'] = (d['inverter_ac_rating_kw'] < MIN_INVERTER_KW) \
        & d['inverter_ac_rating_kw'].notna()
    d['fail_battery'] = d['battery_evidence'] & ~d['usable']
    d['fail_upper'] = ((d['ac_load_net_coverage_pct'] > MAX_COVERAGE_PCT)
                       | (d['pv_site_net_coverage_pct'] > MAX_COVERAGE_PCT))
    d['selected'] = ~d[['fail_load_cov', 'fail_pv_cov', 'fail_inverter',
                        'fail_battery', 'fail_upper']].any(axis=1)

    # Cross-check against the pipeline's own selection.
    sel = pd.read_csv(proc / f'{year}_selected_sites.csv')
    sel = set(sel.loc[sel['state'].isin(STATES), 'edp_site_id'])
    mine = set(d.loc[d['selected'], 'edp_site_id'])
    if mine != sel:
        raise ValueError(f"{year}: reproduced selection differs from "
                         f"{year}_selected_sites.csv by "
                         f"{sorted(mine ^ sel)}")
    return d


def consistency(proc: Path, selected: pd.DataFrame, year: int,
                manual: set[str]) -> pd.DataFrame:
    """One row per site passing selection, with each check's outcome."""
    cc = pd.read_csv(proc / f'{year}_consistency_checks.csv')
    d = selected[['edp_site_id', 'state']].merge(cc, on='edp_site_id',
                                                 how='left')
    if d['overall_pass'].isna().any():
        raise ValueError(f"{year}: selected sites missing from "
                         f"{year}_consistency_checks.csv")
    for n, col in CHECK_COLUMNS.items():
        d[f'fail_{n}'] = ~d[col].astype(bool)
    fail_cols = [f'fail_{n}' for n in CHECK_COLUMNS]
    d['fail_any'] = d[fail_cols].any(axis=1)
    d['manual'] = d['edp_site_id'].isin(manual)
    d['validated'] = ~d['fail_any'] & ~d['manual']

    clean = pd.read_csv(proc / f'{year}_selected_sites_clean.csv')
    clean = set(clean.loc[clean['state'].isin(STATES), 'edp_site_id'])
    passing = set(d.loc[~d['fail_any'], 'edp_site_id'])
    if passing != clean:
        raise ValueError(f"{year}: sites passing all checks differ from "
                         f"{year}_selected_sites_clean.csv by "
                         f"{sorted(passing ^ clean)}")
    return d


# ---- Tables --------------------------------------------------------------

def with_total(rows: dict[str, list]) -> pd.DataFrame:
    t = pd.DataFrame(rows, index=YEARS).T
    t['Total'] = t.sum(axis=1)
    return t


def selection_table(sel: dict[int, pd.DataFrame],
                    chk: dict[int, pd.DataFrame]) -> pd.DataFrame:
    s = {y: sel[y] for y in YEARS}
    return with_total({
        'Sites with raw data': [len(s[y]) for y in YEARS],
        'Net load coverage below 85%': [s[y]['fail_load_cov'].sum() for y in YEARS],
        'PV coverage below 85%': [s[y]['fail_pv_cov'].sum() for y in YEARS],
        'Inverter rating below 1.0 kW': [s[y]['fail_inverter'].sum() for y in YEARS],
        'Coverage above 100.5%': [s[y]['fail_upper'].sum() for y in YEARS],
        'Battery evidence': [s[y]['battery_evidence'].sum() for y in YEARS],
        'Battery, not recoverable': [s[y]['fail_battery'].sum() for y in YEARS],
        'Removed by all filters combined': [(~s[y]['selected']).sum() for y in YEARS],
        'Sites passing coverage checks': [s[y]['selected'].sum() for y in YEARS],
        'of which battery sites recovered': [
            (s[y]['selected'] & s[y]['battery_evidence']).sum() for y in YEARS],
    })


def checks_table(chk: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = {'Sites passing coverage checks': [len(chk[y]) for y in YEARS]}
    for n in CHECK_COLUMNS:
        rows[f'Check {n}'] = [chk[y][f'fail_{n}'].sum() for y in YEARS]
    rows['Failing one or more checks'] = [chk[y]['fail_any'].sum() for y in YEARS]
    rows['Manual exclusions (passing all checks)'] = [
        (chk[y]['manual'] & ~chk[y]['fail_any']).sum() for y in YEARS]
    rows['Validated sites'] = [chk[y]['validated'].sum() for y in YEARS]
    return with_total(rows)


def flags_table(proc: Path, chk: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Interval flags and infill as a share of validated-site intervals."""
    out = {}
    for y in YEARS:
        fs = pd.read_csv(proc / f'{y}_flag_summary.csv')
        fs = fs[fs['edp_site_id'].isin(
            chk[y].loc[chk[y]['validated'], 'edp_site_id'])]
        out[y] = fs.sum(numeric_only=True)
    tot = sum(out.values())
    out['Total'] = tot
    rows = {}
    for ch, name in (('pv', 'PV'), ('load', 'Load')):
        def pct(col, ch=ch):
            return [100 * out[k][f'{ch}_{col}'] / out[k]['n_intervals']
                    for k in out]
        def pct_sum(cols, ch=ch):
            return [100 * sum(out[k][f'{ch}_{c}'] for c in cols)
                    / out[k]['n_intervals'] for k in out]
        rows[f'{name}: clean'] = pct('flag_0')
        rows[f'{name}: missing'] = pct('flag_1')
        rows[f'{name}: other flags'] = pct_sum(
            ['flag_2', 'flag_3', 'flag_4', 'flag_5', 'flag_6'])
        rows[f'{name}: largest other flag'] = [
            100 * max(out[k].get(f'{ch}_flag_{f}', 0) for f in range(2, 7))
            / out[k]['n_intervals'] for k in out]
        rows[f'{name}: infilled, interpolation'] = pct('method1')
        rows[f'{name}: infilled, synthetic day'] = pct('method2')
        rows[f'{name}: infilled, total'] = pct_sum(['method1', 'method2'])
    t = pd.DataFrame(rows, index=list(out)).T
    return t


def by_state_table(chk: dict[int, pd.DataFrame]) -> pd.DataFrame:
    t = pd.DataFrame({y: chk[y][chk[y]['validated']]
                      .groupby('state').size().reindex(STATES, fill_value=0)
                      for y in YEARS})
    t.loc['Total'] = t.sum()
    t['Total'] = t.sum(axis=1)
    return t


def print_markdown(title: str, t: pd.DataFrame, fmt: str = '{:,.0f}') -> None:
    print(f'\n{title}\n')
    cols = [str(c) for c in t.columns]
    print('| | ' + ' | '.join(cols) + ' |')
    print('|---|' + '---|' * len(cols))
    for idx, r in t.iterrows():
        print(f'| {idx} | ' + ' | '.join(fmt.format(v) for v in r) + ' |')


# ---- Figures -------------------------------------------------------------

def figure_by_state(t: pd.DataFrame, out: Path) -> None:
    """Figure C.1: stacked bars of validated sites by state and year."""
    counts = t.loc[STATES, YEARS]
    fig = ts.new_figure(2.6)
    ax = fig.add_subplot()
    x = np.arange(len(YEARS))
    bottom = np.zeros(len(YEARS))
    for i, state in enumerate(STATES):
        v = counts.loc[state].to_numpy(dtype=float)
        ax.bar(x, v, width=0.72, bottom=bottom, label=state,
               color=ts.CATEGORICAL_COLOURS[i], edgecolor='none')
        bottom += v
    totals = bottom
    for xi, tot in zip(x, totals):
        ax.annotate(f'{tot:,.0f}', (xi, tot), xytext=(0, 2),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=ts.ANNOTATION_SIZE)
    ax.set_xticks(x, [str(y) for y in YEARS])
    ax.tick_params(axis='x', length=0)
    ax.grid(axis='x', visible=False)
    ax.set_ylim(0, totals.max() * 1.12)
    ax.set_ylabel('Validated sites')
    ax.set_title('Validated EDP sites by state and year')
    # Legend order matches the stack, top first.
    h, lab = ax.get_legend_handles_labels()
    ax.legend(h[::-1], lab[::-1], loc='upper right')
    ts.save_figure(fig, out)


def break_marks(ax_top, ax_bottom) -> None:
    """Diagonal marks on both side spines where the axis is broken."""
    kw = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=6, linestyle='none',
              color='k', mec='k', mew=0.6, clip_on=False)
    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kw)
    ax_bottom.plot([0, 1], [1, 1], transform=ax_bottom.transAxes, **kw)


def figure_load_vs_generation(v: pd.DataFrame, out: Path) -> None:
    """Figure C.4: annual gross load against annual PV generation per
    site-year, broken y axis, non-residential site-years marked."""
    res = v[~v['nonresidential']]
    nonres = v[v['nonresidential']]
    def step_up(x):
        return np.ceil(x / AXIS_STEP) * AXIS_STEP
    y_lim = (0, step_up(res['annual_gross_load_kwh'].max() * 1.03))
    x_lim = (0, step_up(v['annual_pv_kwh'].max() * X_HEADROOM))
    if len(nonres) and not nonres['annual_gross_load_kwh'].between(*UPPER_LIM).all():
        raise ValueError('non-residential site-years outside UPPER_LIM')

    fig = ts.new_figure(FIG4_HEIGHT_IN, layout=None)
    ax_top, ax_bot = fig.subplots(
        2, 1, sharex=True,
        gridspec_kw=dict(height_ratios=(1, 4), hspace=0.07, left=0.11,
                         right=0.965, bottom=0.13, top=0.92))

    point = dict(s=9, alpha=0.8, edgecolors='white', linewidths=0.5)
    for ax in (ax_top, ax_bot):
        ax.scatter(res['annual_pv_kwh'], res['annual_gross_load_kwh'],
                   color=ts.LINE_COLOUR, zorder=3, **point)
        ax.scatter(nonres['annual_pv_kwh'], nonres['annual_gross_load_kwh'],
                   color=NONRES_COLOUR, marker='D', zorder=4,
                   **{**point, 's': 12})
        ax.plot([0, y_lim[1]], [0, y_lim[1]], color=ts.REFERENCE_COLOUR,
                ls='--', lw=ts.REFERENCE_LW, zorder=2)

    ax_bot.set_xlim(*x_lim)
    ax_bot.set_ylim(*y_lim)
    # No tick at the top of the lower band, where it would crowd the break.
    ax_bot.set_yticks(np.arange(0, y_lim[1], AXIS_STEP))
    ax_bot.spines['top'].set_visible(False)
    ax_top.set_ylim(*UPPER_LIM)
    ax_top.set_yticks(UPPER_TICKS)
    ax_top.spines['bottom'].set_visible(False)
    ax_top.tick_params(axis='x', bottom=False, labelbottom=False)
    for ax in (ax_top, ax_bot):
        ax.yaxis.set_major_formatter('{x:,.0f}')
    ax_bot.xaxis.set_major_formatter('{x:,.0f}')
    break_marks(ax_top, ax_bot)

    ax_bot.set_xlabel('Annual PV generation (kWh)')
    fig.supylabel('Annual gross load (kWh)', x=0.01,
                  fontsize=ts.RC['axes.labelsize'])
    ax_top.set_title('Annual gross load against annual PV generation, '
                     'per site-year')

    handles = [
        Line2D([], [], ls='none', marker='o', ms=3.5, mfc=ts.LINE_COLOUR,
               mec='white', mew=0.5),
        Line2D([], [], ls='none', marker='D', ms=3.2, mfc=NONRES_COLOUR,
               mec='white', mew=0.5),
        Line2D([], [], color=ts.REFERENCE_COLOUR, ls='--',
               lw=ts.REFERENCE_LW)]
    labels = [f'Validated site-years (n = {len(res):,})',
              f'Non-residential, excluded (n = {len(nonres)})',
              'Load = generation']
    # At the top right of the whole plot, overlapping the break into the
    # lower band: drawn on the upper band, raised above the lower one.
    ax_top.set_zorder(ax_bot.get_zorder() + 1)
    leg = ax_top.legend(handles, labels, loc='upper right')
    leg.set_zorder(20)
    ts.save_figure(fig, out)


def load_volumes(dirs: list[Path],
                 chk: dict[int, pd.DataFrame]) -> pd.DataFrame | None:
    frames = []
    for y in YEARS:
        name = f'{y}_site_annual_totals.csv'
        f = next((d / name for d in dirs if (d / name).exists()), None)
        if f is None:
            print(f"  {name} not found in {', '.join(map(str, dirs))}: run "
                  f"edp_annual_totals.py on the clean parquet files; "
                  f"Figure C.4 skipped")
            return None
        t = pd.read_csv(f)
        c = chk[y]
        keep = c['validated'] | c['edp_site_id'].isin(EXCLUDED_SITES)
        t = t.merge(c.loc[keep, ['edp_site_id', 'state', 'validated']],
                    on='edp_site_id', how='inner')
        missing = set(c.loc[c['validated'], 'edp_site_id']) - set(t['edp_site_id'])
        if missing:
            raise ValueError(f"{y}: validated sites with no annual total: "
                             f"{sorted(missing)}")
        frames.append(t)
    v = pd.concat(frames, ignore_index=True)
    v['nonresidential'] = v['edp_site_id'].isin(EXCLUDED_SITES)
    return v


# ---- Main ----------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--processed-dir', type=Path,
                    default=REPO_ROOT / 'data/processed/edp')
    ap.add_argument('--raw-dir', type=Path, default=REPO_ROOT / 'data/raw/edp')
    ap.add_argument('--out-dir', type=Path,
                    default=REPO_ROOT / 'analysis/output/appendix_c')
    ap.add_argument('--fig-dir', type=Path,
                    default=REPO_ROOT / 'thesis/figures/appendix_c')
    args = ap.parse_args(argv)
    proc = args.processed_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_metadata(args.raw_dir)
    manual = manual_exclusions(args.raw_dir)
    sel = {y: site_selection(proc, meta, y) for y in YEARS}
    chk = {y: consistency(proc, sel[y][sel[y]['selected']], y,
                          manual.get(y, set())) for y in YEARS}

    t_sel = selection_table(sel, chk)
    t_chk = checks_table(chk)
    t_flag = flags_table(proc, chk)
    t_state = by_state_table(chk)
    for name, t in (('site_selection', t_sel), ('consistency_checks', t_chk),
                    ('interval_flags', t_flag), ('validated_by_state', t_state)):
        t.to_csv(args.out_dir / f'{name}.csv')
    print_markdown('Site selection (sites removed by each filter alone)', t_sel)
    print_markdown('Consistency checks (sites failing each check alone)', t_chk)
    print_markdown('Interval flags and infill (% of validated-site intervals)',
                   t_flag, '{:.2f}')
    print_markdown('Validated sites by state', t_state)

    print('\nManual exclusions by year (site: stage reached):')
    for y in YEARS:
        c = chk[y]
        m = sorted(manual.get(y, set()))
        stage = []
        for s in m:
            if s in set(c['edp_site_id']):
                f = c.loc[c['edp_site_id'] == s, 'fail_any'].iloc[0]
                stage.append(f"{s} {'failed checks' if f else 'passed checks'}")
            elif s in set(sel[y]['edp_site_id']):
                stage.append(f'{s} failed coverage')
            else:
                stage.append(f'{s} no raw data')
        print(f'  {y}: ' + '; '.join(stage))

    ts.apply_style()
    figure_by_state(t_state, args.fig_dir / '01_validated_sites_by_state.png')

    v = load_volumes([proc, args.raw_dir], chk)
    if v is not None:
        v.to_csv(args.out_dir / 'site_year_volumes.csv', index=False)
        res = v[~v['nonresidential']]
        nr = v[v['nonresidential']]
        med = res['annual_gross_load_kwh'].median()
        print(f"\nValidated site-years: {len(res):,}; median annual load "
              f"{med:,.0f} kWh, max {res['annual_gross_load_kwh'].max():,.0f}; "
              f"median PV {res['annual_pv_kwh'].median():,.0f} kWh")
        for s, g in nr.groupby('edp_site_id'):
            lo, hi = g['annual_gross_load_kwh'].min(), g['annual_gross_load_kwh'].max()
            print(f"  {s} ({', '.join(map(str, sorted(g['year'])))}): "
                  f"{lo:,.0f}-{hi:,.0f} kWh/yr, {lo / med:.0f}-{hi / med:.0f}x median")
        figure_load_vs_generation(v, args.fig_dir / '04_annual_load_vs_generation.png')
    return 0


if __name__ == '__main__':
    sys.exit(main())
