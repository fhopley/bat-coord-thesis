"""Sensitivity to the battery degradation cost (docs/sensitivity_runs_spec.md, 1).

Amber, achievable imperfect foresight, NSW 2024, 20 kWh, at c_deg 3.4, 3.9 and
4.6 c/kWh. 3.9 is the production value and 3.4 and 4.6 are the ends of the
calibration range.

Self consumption and Solar only do not depend on the degradation cost, so the
production runs serve as the reference at every c_deg.

Outputs, in --output-dir:
  degradation_sensitivity.csv   every metric at each c_deg
  degradation_sensitivity.png   two bar panels, chosen by --panels

Usage (from project root):
    python3 analysis/scripts/sensitivity/degradation.py --data-root "/Volumes/Manual/Thesis Project"
    python3 analysis/scripts/sensitivity/degradation.py --data-root ... --print-values
    python3 analysis/scripts/sensitivity/degradation.py --data-root ... --panels household,wholesale
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
import thesis_style as ts                # noqa: E402
import matplotlib.pyplot as plt          # noqa: E402

SIZE = 20
HEIGHT_IN = 3.0

# c_deg (c/kWh) -> run directory. 3.9 is the production run.
RUNS = {
    3.4: 'sens_cdeg034_rolling_amber',
    3.9: sc.REF_ROLLING_AMBER,
    4.6: 'sens_cdeg046_rolling_amber',
}
PRODUCTION_CDEG = 3.9

# The self-consumption column each panel's reference line reads, where one
# exists. Self consumption does not depend on the degradation cost, so every
# one of these is a constant across the three runs.
SC_REFERENCE_COLUMN = {
    'household': 'sc_hh_saving',
    'wholesale': 'sc_ws_saving',
    'throughput': 'sc_discharge_kwh',
    'cycles': 'sc_cycles_per_day',
}

PANEL_SPECS = {
    'household': dict(column='hh_saving',
                      title='Household bill saving',
                      ylabel='Saving against Solar only (\\$/yr)'),
    'wholesale': dict(column='ws_saving',
                      title='Net wholesale cost saving',
                      ylabel='Saving against Solar only (\\$/yr)'),
    'coordination': dict(column='coordination_value',
                         title='Coordination value',
                         ylabel='Coordinated less self-consumption (\\$/yr)'),
    'throughput': dict(column='discharge_kwh',
                       title='Annual battery discharge',
                       ylabel='Discharge (kWh/yr)'),
    'cycles': dict(column='cycles_per_day',
                   title='Equivalent full cycles per day',
                   ylabel='Cycles per day'),
}


def collect(data_root: Path, state: str, year: str, weighted: bool) -> pd.DataFrame:
    runs = sc.runs_root(data_root)
    weight = sc.site_weights(data_root, state, year) if weighted else None

    solar = sc.load_solar_only(runs, sc.REF_SOLAR_AMBER, state, year)
    self_cons = sc.require_bills(runs, sc.REF_SC_AMBER, SIZE, state, year)
    sc_throughput = sc.battery_throughput(runs, sc.REF_SC_AMBER, SIZE,
                                          state, year)

    bills = {c: sc.require_bills(runs, r, SIZE, state, year)
             for c, r in RUNS.items()}
    throughput = {c: sc.battery_throughput(runs, r, SIZE, state, year)
                  for c, r in RUNS.items()}

    # Paired over the sites every run shares, including the references.
    idx = sc.common_index(solar, self_cons, *bills.values())
    for t in throughput.values():
        if t is not None:
            idx = idx.intersection(t[0].index)
    if len(idx) == 0:
        raise SystemExit('No sites common to every run; nothing to compare.')

    sc_sav = sc.savings(self_cons.loc[idx], solar.loc[idx])

    # Self consumption is rule-based, so its cycling does not depend on the
    # degradation cost and is one constant across the three runs.
    sc_cycles = np.nan
    sc_discharge = np.nan
    if sc_throughput is not None:
        sc_flows, sc_days = sc_throughput
        shared = idx.intersection(sc_flows.index)
        if len(shared):
            sc_discharge = sc.wmean(sc_flows.loc[shared, 'discharge_kwh'],
                                    weight)
            sc_cycles = sc.cycles_per_day(sc_discharge, SIZE, sc_days)

    rows = []
    for cdeg, run in RUNS.items():
        sav = sc.savings(bills[cdeg].loc[idx], solar.loc[idx])
        row = {
            'c_deg_c_per_kwh': cdeg,
            'run': run,
            'n_sites': len(idx),
            'hh_saving': sc.wmean(sav['hh_saving'], weight),
            'ws_saving': sc.wmean(sav['ws_saving'], weight),
            'sc_hh_saving': sc.wmean(sc_sav['hh_saving'], weight),
            'sc_ws_saving': sc.wmean(sc_sav['ws_saving'], weight),
            'sc_discharge_kwh': sc_discharge,
            'sc_cycles_per_day': sc_cycles,
        }
        row['coordination_value'] = row['hh_saving'] - row['sc_hh_saving']
        row['ws_coordination_value'] = row['ws_saving'] - row['sc_ws_saving']
        if throughput[cdeg] is not None:
            flows, days = throughput[cdeg]
            discharge = sc.wmean(flows.loc[idx, 'discharge_kwh'], weight)
            row['discharge_kwh'] = discharge
            row['charge_kwh'] = sc.wmean(flows.loc[idx, 'charge_kwh'], weight)
            row['cycles_per_day'] = sc.cycles_per_day(discharge, SIZE, days)
        else:
            row['discharge_kwh'] = np.nan
            row['charge_kwh'] = np.nan
            row['cycles_per_day'] = np.nan
        rows.append(row)

    df = pd.DataFrame(rows).sort_values('c_deg_c_per_kwh').reset_index(drop=True)
    base = df.loc[df['c_deg_c_per_kwh'] == PRODUCTION_CDEG].iloc[0]
    for col in ('hh_saving', 'ws_saving', 'coordination_value',
                'discharge_kwh', 'cycles_per_day'):
        df[f'{col}_change'] = df[col] - base[col]
        df[f'{col}_pct_change'] = np.where(
            base[col] != 0, 100.0 * (df[col] - base[col]) / abs(base[col]), np.nan)
    return df


# Panels drawn from the same underlying quantity would show the same shape
# twice: cycles per day is annual discharge rescaled, and the coordination
# value is the household saving less a reference that does not move with
# c_deg. One panel per family.
PANEL_FAMILY = {
    'household': 'household', 'coordination': 'household',
    'wholesale': 'wholesale',
    'throughput': 'cycling', 'cycles': 'cycling',
}


def rank_panels(df: pd.DataFrame) -> list[str]:
    """The two metrics that move most across the calibration range, taken
    from different families so the panels are not two views of one number."""
    spread = {}
    for key, spec in PANEL_SPECS.items():
        col = spec['column']
        if col not in df or df[col].isna().any():
            continue
        base = df.loc[df['c_deg_c_per_kwh'] == PRODUCTION_CDEG, col].iloc[0]
        if base == 0:
            continue
        spread[key] = 100.0 * (df[col].max() - df[col].min()) / abs(base)

    chosen, seen = [], set()
    for key, _ in sorted(spread.items(), key=lambda kv: -kv[1]):
        family = PANEL_FAMILY[key]
        if family in seen:
            continue
        chosen.append(key)
        seen.add(family)
        if len(chosen) == 2:
            break
    return chosen


def draw(df: pd.DataFrame, panels: list[str], out_path: Path,
         sc_reference: bool, thesis_dir: Path | None,
         mode: str = 'change') -> None:
    """Two bar panels, one per metric.

    mode 'level' draws the metric itself, with bars from zero. Over the
    calibration range the savings move by about a percent, so on a zero
    baseline the three bars are indistinguishable, which states the finding
    but does not quantify it.

    mode 'change' draws each metric as a percentage change from the
    production 3.9 c/kWh value. Zero then means no change rather than no
    saving, so the bars carry the size of the effect, and a shared axis puts
    the two metrics on the same scale.
    """
    ts.apply_style()
    fig = ts.new_figure(height_in=HEIGHT_IN)
    axes = fig.subplots(1, 2, sharey=(mode == 'change'))

    labels = [f'{c:.1f}' for c in df['c_deg_c_per_kwh']]
    x = np.arange(len(df))
    colours = [ts.CATEGORICAL_COLOURS[0], ts.EMPHASIS_COLOUR,
               ts.CATEGORICAL_COLOURS[1]]

    handles: list = []
    for i, (ax, key) in enumerate(zip(axes, panels)):
        spec = PANEL_SPECS[key]
        column = (spec['column'] if mode == 'level'
                  else f"{spec['column']}_pct_change")
        values = df[column].to_numpy()
        bars = ax.bar(x, values, width=0.6, color=colours,
                      edgecolor='white', linewidth=0.5)
        ax.set_title(spec['title'], fontsize=ts.SUBTITLE_SIZE)
        if mode == 'level':
            ax.set_ylabel(spec['ylabel'])
        elif i == 0:
            ax.set_ylabel('Change from $c_{\\mathrm{deg}}$ = '
                          f'{PRODUCTION_CDEG:.1f} c/kWh (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel('$c_{\\mathrm{deg}}$ (c/kWh)')
        ax.grid(axis='x', visible=False)

        for bar, value in zip(bars, values):
            if mode == 'change':
                text = '0' if value == 0 else f'{value:+.1f}%'
            else:
                text = f'{value:,.0f}' if abs(value) >= 10 else f'{value:,.2f}'
            # Label sits outside the bar, so below a negative one.
            ax.annotate(text,
                        (bar.get_x() + bar.get_width() / 2, value),
                        ha='center', va='top' if value < 0 else 'bottom',
                        fontsize=ts.ANNOTATION_SIZE,
                        xytext=(0, -3 if value < 0 else 2),
                        textcoords='offset points')

        if mode == 'change':
            ax.axhline(0, color=ts.REFERENCE_COLOUR, lw=ts.REFERENCE_LW)
        # Self consumption does not depend on c_deg, so it is a flat
        # reference on a level chart and identically zero on a change one.
        # It goes on every panel or none: on one panel alone it reads as a
        # missing line on the other.
        elif sc_reference:
            ref_col = SC_REFERENCE_COLUMN.get(key)
            if ref_col and not pd.isna(df[ref_col].iloc[0]):
                line = ax.axhline(df[ref_col].iloc[0],
                                  color=ts.REFERENCE_COLOUR,
                                  lw=ts.REFERENCE_LW, ls='--')
                handles.append(line)
        ax.margins(y=0.22)

    if handles:
        fig.legend([handles[0]], ['Self consumption'],
                   loc='outside lower center')
    ts.panel_suptitle(
        fig, f'Sensitivity to the degradation cost at {SIZE} kWh')
    sc.save_figure(fig, out_path, thesis_dir)


def print_values(df: pd.DataFrame) -> None:
    base = df.loc[df['c_deg_c_per_kwh'] == PRODUCTION_CDEG].iloc[0]
    print(f'\nValues for the text (NSW 2024, {SIZE} kWh, '
          f'n = {int(base["n_sites"])} sites)')
    print('-' * 72)
    print(f'Production c_deg {PRODUCTION_CDEG:.1f} c/kWh: '
          f'household saving ${base["hh_saving"]:,.0f}, '
          f'coordination value ${base["coordination_value"]:,.0f}, '
          f'wholesale saving ${base["ws_saving"]:,.0f}, '
          f'{base["cycles_per_day"]:.3f} cycles/day')
    for _, r in df[df['c_deg_c_per_kwh'] != PRODUCTION_CDEG].iterrows():
        print(f'\nc_deg {r["c_deg_c_per_kwh"]:.1f} c/kWh:')
        for label, col in (('household saving', 'hh_saving'),
                           ('coordination value', 'coordination_value'),
                           ('wholesale saving', 'ws_saving')):
            print(f'  {label:<20} ${r[col]:>8,.0f}  '
                  f'({r[col + "_change"]:+,.0f}, '
                  f'{r[col + "_pct_change"]:+.1f}%)')
        print(f'  {"cycles per day":<20} {r["cycles_per_day"]:>9.3f}  '
              f'({r["cycles_per_day_change"]:+.3f}, '
              f'{r["cycles_per_day_pct_change"]:+.1f}%)')
        print(f'  {"discharge kWh/yr":<20} {r["discharge_kwh"]:>9,.0f}  '
              f'({r["discharge_kwh_change"]:+,.0f}, '
              f'{r["discharge_kwh_pct_change"]:+.1f}%)')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sc.add_common_args(parser)
    parser.add_argument('--panels', default=None,
                        help='Two of ' + ', '.join(PANEL_SPECS)
                             + '. Default: the two that move most.')
    parser.add_argument('--sc-reference', action='store_true',
                        help='Draw the self-consumption value as a flat '
                             'reference line on every panel. Off by default: '
                             'self consumption does not depend on the '
                             'degradation cost, so the line carries nothing '
                             'about the sensitivity the chart is about.')
    parser.add_argument('--mode', choices=('level', 'change'), default='level',
                        help="'level' (default) draws the metric itself; "
                             "'change' draws it as a percentage change from "
                             "the production value.")
    args = parser.parse_args(argv)

    out_dir = args.output_dir or (sc.OUTPUT_ROOT / 'degradation')
    df = collect(args.data_root, args.state, args.year, not args.unweighted)
    sc.write_csv(df, out_dir / 'degradation_sensitivity.csv')

    panels = args.panels.split(',') if args.panels else rank_panels(df)
    unknown = [p for p in panels if p not in PANEL_SPECS]
    if unknown:
        parser.error(f'Unknown panel(s) {unknown}; choose from '
                     f'{list(PANEL_SPECS)}')
    if len(panels) != 2:
        parser.error('Exactly two panels are drawn.')
    print(f'  Panels: {panels[0]} and {panels[1]}')

    draw(df, panels, out_dir / 'degradation_sensitivity.png',
         args.sc_reference, args.thesis_dir, args.mode)

    if args.print_values:
        print_values(df)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
