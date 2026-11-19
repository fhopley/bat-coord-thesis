"""Sensitivity to the VPP event window (docs/sensitivity_runs_spec.md, 3).

Origin Battery Lite, perfect foresight, NSW 2024, 20 kWh, at event windows of
15 minutes, 1 hour and 10 hours. 1 hour is the production setting, drawn in the
emphasis colour rather than labelled, so the chart reads as three settings
rather than one result and two variants.

The 15-minute run covers NSW 2024 only, so its result is taken as
representative on the strength of a check: the 1 h to 10 h difference in
NSW 2024 is compared against the same difference nationally, where both runs
exist for every state, year and size. --check-only runs just that check.

Savings are against the Origin Solar only run, so each is the value of the
battery on that offer rather than the value of switching to it.

Outputs, in --output-dir:
  control_window.csv              every metric at each window, and the check
  control_window_sensitivity.png  two bar panels, chosen by --panels

Usage (from project root):
    python3 analysis/scripts/sensitivity/control_window.py --data-root "/Volumes/Manual/Thesis Project"
    python3 analysis/scripts/sensitivity/control_window.py --data-root ... --print-values
    python3 analysis/scripts/sensitivity/control_window.py --data-root ... --check-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sensitivity_common as sc          # noqa: E402
import financial_analysis as fa          # noqa: E402
import flow_exposure_analysis as fea     # noqa: E402
import thesis_style as ts                # noqa: E402

SIZE = 20
HEIGHT_IN = 3.0
PRODUCTION_WINDOW = '1 h'

# Window label -> run directory. 1 h is the production run.
RUNS = {
    '15 min': 'sens_w15min_optimised_origin_lite',
    '1 h': sc.REF_ORIGIN_1H,
    '10 h': sc.REF_ORIGIN_10H,
}
WINDOW_ORDER = ['15 min', '1 h', '10 h']

PANEL_SPECS = {
    'household': dict(column='hh_saving', title='Household bill saving',
                      ylabel='Saving against Solar only (\\$/yr)'),
    'wholesale': dict(column='ws_saving',
                      title='Net wholesale cost saving',
                      ylabel='Saving against Solar only (\\$/yr)'),
    'vpp_reward': dict(column='vpp_reward', title='VPP reward',
                       ylabel='Reward (\\$/yr)'),
    'forced_export': dict(column='forced_export_kwh',
                          title='Forced discharge',
                          ylabel='Forced discharge (kWh/yr)'),
    'forced_import': dict(column='forced_import_kwh', title='Forced charge',
                          ylabel='Forced charge (kWh/yr)'),
    'margin': dict(column='retailer_margin', title='Retailer margin',
                   ylabel='Margin (\\$/yr)'),
}
# One panel per family, so a chart never shows the same quantity twice.
# The VPP reward is a per-kWh payment on forced flow, so on a plan that pays
# a flat rate it is the forced flow rescaled: Origin Battery Lite pays
# $1/kWh on forced export, which makes its reward in dollars equal to its
# forced export in kWh.
PANEL_FAMILY = {
    'household': 'household',
    'wholesale': 'wholesale',
    'vpp_reward': 'forced', 'forced_export': 'forced',
    'forced_import': 'forced',
    'margin': 'margin',
}


def _scenario_metrics(runs: Path, run: str, solar: pd.DataFrame,
                      idx: pd.Index, weight: pd.Series | None,
                      state: str, year: str, size: int) -> dict:
    bills = sc.require_bills(runs, run, size, state, year).loc[idx]
    sav = sc.savings(bills, solar.loc[idx])
    forced_imp, forced_exp = fea.load_forced_totals(
        runs, run, state, year, size, idx)
    return {
        'hh_saving': sc.wmean(sav['hh_saving'], weight),
        'ws_saving': sc.wmean(sav['ws_saving'], weight),
        'vpp_reward': sc.wmean(sav['vpp_reward'], weight),
        'retailer_margin': sc.wmean(bills['retailer_margin'], weight),
        'forced_import_kwh': sc.wmean(forced_imp, weight),
        'forced_export_kwh': sc.wmean(forced_exp, weight),
    }


def collect(data_root: Path, state: str, year: str,
            weighted: bool) -> pd.DataFrame:
    runs = sc.runs_root(data_root)
    weight = sc.site_weights(data_root, state, year) if weighted else None
    solar = sc.load_solar_only(runs, sc.REF_SOLAR_ORIGIN, state, year)

    bills = {w: sc.require_bills(runs, r, SIZE, state, year)
             for w, r in RUNS.items()}
    idx = sc.common_index(solar, *bills.values())
    if len(idx) == 0:
        raise SystemExit('No sites common to every control-window run.')

    rows = []
    for window in WINDOW_ORDER:
        row = {'window': window, 'run': RUNS[window], 'battery_kwh': SIZE,
               'scope': f'{state} {year}', 'n_sites': len(idx)}
        row.update(_scenario_metrics(runs, RUNS[window], solar, idx, weight,
                                     state, year, SIZE))
        rows.append(row)

    df = pd.DataFrame(rows)
    base = df.loc[df['window'] == PRODUCTION_WINDOW].iloc[0]
    for key in ('hh_saving', 'ws_saving', 'vpp_reward', 'retailer_margin',
                'forced_import_kwh', 'forced_export_kwh'):
        df[f'{key}_change'] = df[key] - base[key]
    return df


def representativeness(data_root: Path, state: str, year: str,
                       weighted: bool) -> pd.DataFrame:
    """The 1 h to 10 h difference in the subset against the national one.

    Both full runs exist for every state, year and size, so the national
    difference is the same quantity measured on the whole study population.
    If the two agree, the 15-minute result measured on the subset carries.
    """
    runs = sc.runs_root(data_root)
    cfg = fa.load_weights_config(sc.WEIGHTS_PATH)
    rows = []

    scopes = [(state, [year])]
    national = [(s, [str(y) for y in (2021, 2022, 2023, 2024, 2025)])
                for s in cfg['state_mix']]
    scopes.extend(national)

    for scope_state, scope_years in scopes:
        per_year = []
        for scope_year in scope_years:
            try:
                solar = sc.load_solar_only(runs, sc.REF_SOLAR_ORIGIN,
                                           scope_state, scope_year)
            except SystemExit:
                continue
            b1 = sc.load_bills(runs, sc.REF_ORIGIN_1H, SIZE, scope_state,
                               scope_year)
            b10 = sc.load_bills(runs, sc.REF_ORIGIN_10H, SIZE, scope_state,
                                scope_year)
            if b1 is None or b10 is None:
                continue
            idx = sc.common_index(solar, b1, b10)
            if len(idx) == 0:
                continue
            weight = (sc.site_weights(data_root, scope_state, scope_year)
                      if weighted else None)
            s1 = sc.savings(b1.loc[idx], solar.loc[idx])
            s10 = sc.savings(b10.loc[idx], solar.loc[idx])
            per_year.append({
                'hh_diff': (sc.wmean(s10['hh_saving'], weight)
                            - sc.wmean(s1['hh_saving'], weight)),
                'ws_diff': (sc.wmean(s10['ws_saving'], weight)
                            - sc.wmean(s1['ws_saving'], weight)),
                'n_sites': len(idx),
            })
        if not per_year:
            continue
        frame = pd.DataFrame(per_year)
        rows.append({
            'scope': f'{scope_state} {scope_years[0]}' if len(scope_years) == 1
                     else scope_state,
            'state': scope_state,
            'years': len(frame),
            'hh_diff_10h_less_1h': frame['hh_diff'].mean(),
            'ws_diff_10h_less_1h': frame['ws_diff'].mean(),
            'n_sites': int(frame['n_sites'].sum()),
        })

    df = pd.DataFrame(rows)
    # National is the state mix over each state's across-year mean, as in
    # financial_analysis's tier-3 combination.
    states = df[df['scope'] == df['state']]
    if not states.empty:
        mix = cfg['state_mix']
        w = states['state'].map(mix).to_numpy(float)
        w = w / w.sum()
        df = pd.concat([df, pd.DataFrame([{
            'scope': 'National', 'state': '-', 'years': int(states['years'].max()),
            'hh_diff_10h_less_1h': float(np.average(
                states['hh_diff_10h_less_1h'], weights=w)),
            'ws_diff_10h_less_1h': float(np.average(
                states['ws_diff_10h_less_1h'], weights=w)),
            'n_sites': int(states['n_sites'].sum()),
        }])], ignore_index=True)
    return df


def rank_panels(df: pd.DataFrame) -> list[str]:
    spread = {}
    for key, spec in PANEL_SPECS.items():
        col = spec['column']
        if col not in df or df[col].isna().any():
            continue
        base = df.loc[df['window'] == PRODUCTION_WINDOW, col].iloc[0]
        if base == 0:
            continue
        spread[key] = 100.0 * (df[col].max() - df[col].min()) / abs(base)
    chosen, seen = [], set()
    for key, _ in sorted(spread.items(), key=lambda kv: -kv[1]):
        if PANEL_FAMILY[key] in seen:
            continue
        chosen.append(key)
        seen.add(PANEL_FAMILY[key])
        if len(chosen) == 2:
            break
    return chosen


def draw(df: pd.DataFrame, panels: list[str], out_path: Path,
         thesis_dir: Path | None) -> None:
    ts.apply_style()
    fig = ts.new_figure(height_in=HEIGHT_IN)
    axes = fig.subplots(1, 2)

    ordered = df.set_index('window').loc[WINDOW_ORDER]
    x = np.arange(len(WINDOW_ORDER))
    colours = [ts.CATEGORICAL_COLOURS[0], ts.EMPHASIS_COLOUR,
               ts.CATEGORICAL_COLOURS[1]]

    for ax, key in zip(axes, panels):
        spec = PANEL_SPECS[key]
        values = ordered[spec['column']].to_numpy()
        bars = ax.bar(x, values, width=0.6, color=colours,
                      edgecolor='white', linewidth=0.5)
        ax.set_title(spec['title'], fontsize=ts.SUBTITLE_SIZE)
        ax.set_ylabel(spec['ylabel'])
        ax.set_xticks(x)
        ax.set_xticklabels(WINDOW_ORDER)
        ax.set_xlabel('VPP event window')
        ax.grid(axis='x', visible=False)
        for bar, value in zip(bars, values):
            ax.annotate(f'{value:,.0f}',
                        (bar.get_x() + bar.get_width() / 2, value),
                        ha='center', va='bottom',
                        fontsize=ts.ANNOTATION_SIZE, xytext=(0, 2),
                        textcoords='offset points')
        ax.margins(y=0.18)

    ts.panel_suptitle(
        fig, f'Sensitivity to the VPP event window at {SIZE} kWh')
    sc.save_figure(fig, out_path, thesis_dir)


def print_values(df: pd.DataFrame, check: pd.DataFrame) -> None:
    print(f'\nValues for the text (NSW 2024, Origin Battery Lite, {SIZE} kWh, '
          f'n = {int(df["n_sites"].iloc[0])} sites)')
    print('-' * 78)
    print(f'  {"window":>8} {"hh saving":>11} {"vpp reward":>11} '
          f'{"ws saving":>11} {"forced dis":>11} {"forced chg":>11} '
          f'{"margin":>9}')
    for _, r in df.set_index('window').loc[WINDOW_ORDER].reset_index().iterrows():
        mark = ' *' if r['window'] == PRODUCTION_WINDOW else '  '
        print(f'  {r["window"]:>6}{mark}{r["hh_saving"]:>11,.0f} '
              f'{r["vpp_reward"]:>11,.0f} {r["ws_saving"]:>11,.0f} '
              f'{r["forced_export_kwh"]:>11,.0f} '
              f'{r["forced_import_kwh"]:>11,.0f} '
              f'{r["retailer_margin"]:>9,.0f}')

    print_check(check)


def print_check(check: pd.DataFrame) -> None:
    print('\nIs the subset representative? 10 h less 1 h, by scope')
    print('-' * 78)
    for _, r in check.iterrows():
        print(f'  {r["scope"]:<12} household {r["hh_diff_10h_less_1h"]:>+9,.0f}   '
              f'wholesale {r["ws_diff_10h_less_1h"]:>+9,.0f}   '
              f'(n = {int(r["n_sites"])})')
    subset = check[check['scope'].str.contains(' ')]
    nat = check[check['scope'] == 'National']
    if not subset.empty and not nat.empty:
        for label, col in (('household', 'hh_diff_10h_less_1h'),
                           ('wholesale', 'ws_diff_10h_less_1h')):
            s, n = float(subset[col].iloc[0]), float(nat[col].iloc[0])
            gap = 'n/a' if n == 0 else f'{100 * (s - n) / abs(n):+.0f}%'
            print(f'  {label}: subset {s:+,.0f} against national {n:+,.0f} '
                  f'({gap} of the national difference)')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sc.add_common_args(parser)
    parser.add_argument('--panels', default=None,
                        help='Two of ' + ', '.join(PANEL_SPECS)
                             + '. Default: the two that move most.')
    parser.add_argument('--check-only', action='store_true',
                        help='Only run the NSW 2024 against national check.')
    args = parser.parse_args(argv)

    out_dir = args.output_dir or (sc.OUTPUT_ROOT / 'control_window')
    weighted = not args.unweighted

    check = representativeness(args.data_root, args.state, args.year, weighted)
    sc.write_csv(check, out_dir / 'control_window_representativeness.csv')
    if args.check_only:
        print_check(check)
        return 0

    df = collect(args.data_root, args.state, args.year, weighted)
    sc.write_csv(df, out_dir / 'control_window.csv')

    panels = args.panels.split(',') if args.panels else rank_panels(df)
    unknown = [p for p in panels if p not in PANEL_SPECS]
    if unknown:
        parser.error(f'Unknown panel(s) {unknown}; choose from {list(PANEL_SPECS)}')
    if len(panels) != 2:
        parser.error('Exactly two panels are drawn.')
    print(f'  Panels: {panels[0]} and {panels[1]}')

    draw(df, panels, out_dir / 'control_window_sensitivity.png', args.thesis_dir)
    if args.print_values:
        print_values(df, check)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
