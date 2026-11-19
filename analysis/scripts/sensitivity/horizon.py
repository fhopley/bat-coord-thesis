"""Sensitivity to the rolling optimisation horizon (docs/sensitivity_runs_spec.md, 2).

Amber, achievable imperfect foresight, NSW 2024, every battery size, at
horizons of 6, 12, 18, 24 and 48 hours. 18 h is the production horizon.

The value capture ratio is the share of the perfect-foresight headroom the
rolling run captures,

    (rolling saving - self-consumption saving)
    / (perfect-foresight saving - self-consumption saving),

for the household bill and for the net wholesale cost. It is pooled: the
weighted mean of the numerator over the weighted mean of the denominator,
rather than the mean of per-site ratios, so sites whose own headroom is near
zero cannot blow the ratio up.

Window solve times are the median over sites of each site's own median and
p95 window solve time.

Outputs, in --output-dir:
  horizon_value_capture.csv   every metric at each horizon and battery size
  horizon_value_capture.png   household and wholesale value capture by size

Usage (from project root):
    python3 analysis/scripts/sensitivity/horizon.py --data-root "/Volumes/Manual/Thesis Project"
    python3 analysis/scripts/sensitivity/horizon.py --data-root ... --print-values
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

HEIGHT_IN = 3.0
PRODUCTION_HORIZON = 18

# Horizon (hours) -> run directory. 18 h is the production run.
RUNS = {
    6: 'sens_h06_rolling_amber',
    12: 'sens_h12_rolling_amber',
    18: sc.REF_ROLLING_AMBER,
    24: 'sens_h24_rolling_amber',
    48: 'sens_h48_rolling_amber',
}

# Threshold for "a longer horizon adds more than about this much".
MATERIAL_GAIN = 0.01


def collect(data_root: Path, state: str, year: str,
            weighted: bool) -> pd.DataFrame:
    runs = sc.runs_root(data_root)
    weight = sc.site_weights(data_root, state, year) if weighted else None
    solar = sc.load_solar_only(runs, sc.REF_SOLAR_AMBER, state, year)

    rows = []
    for size in sc.BATTERY_SIZES:
        self_cons = sc.load_bills(runs, sc.REF_SC_AMBER, size, state, year)
        perfect = sc.load_bills(runs, sc.REF_OPTIMISED_AMBER, size, state, year)
        if self_cons is None or perfect is None:
            print(f'  Skipping {size} kWh: missing self-consumption or '
                  f'perfect-foresight reference.', file=sys.stderr)
            continue

        bills = {h: b for h, r in RUNS.items()
                 if (b := sc.load_bills(runs, r, size, state, year)) is not None}
        missing = [h for h in RUNS if h not in bills]
        if missing:
            print(f'  {size} kWh: no run at horizon(s) {missing}; the other '
                  f'horizons are still charted at this size.', file=sys.stderr)
        if not bills:
            continue

        # Paired over the sites every horizon at this size shares, so the
        # horizons are compared on one population.
        idx = sc.common_index(solar, self_cons, perfect, *bills.values())
        if len(idx) == 0:
            print(f'  Skipping {size} kWh: no common sites.', file=sys.stderr)
            continue

        sc_sav = sc.savings(self_cons.loc[idx], solar.loc[idx])
        pf_sav = sc.savings(perfect.loc[idx], solar.loc[idx])
        sc_hh = sc.wmean(sc_sav['hh_saving'], weight)
        sc_ws = sc.wmean(sc_sav['ws_saving'], weight)
        head_hh = sc.wmean(pf_sav['hh_saving'], weight) - sc_hh
        head_ws = sc.wmean(pf_sav['ws_saving'], weight) - sc_ws

        for hours in sorted(bills):
            sav = sc.savings(bills[hours].loc[idx], solar.loc[idx])
            hh = sc.wmean(sav['hh_saving'], weight)
            ws = sc.wmean(sav['ws_saving'], weight)
            diag = sc.load_diagnostics(runs, RUNS[hours], size, state, year)
            if diag is not None:
                diag = diag.reindex(idx)

            rows.append({
                'horizon_hours': hours,
                'battery_kwh': size,
                'run': RUNS[hours],
                'n_sites': len(idx),
                'hh_saving': hh,
                'ws_saving': ws,
                'sc_hh_saving': sc_hh,
                'sc_ws_saving': sc_ws,
                'pf_hh_saving': sc_hh + head_hh,
                'pf_ws_saving': sc_ws + head_ws,
                'coordination_value': hh - sc_hh,
                'ws_coordination_value': ws - sc_ws,
                'hh_value_capture': (hh - sc_hh) / head_hh if head_hh else np.nan,
                'ws_value_capture': (ws - sc_ws) / head_ws if head_ws else np.nan,
                'infill_fraction_mean': (
                    sc.wmean(diag['infill_fraction_mean'], weight)
                    if diag is not None
                    and 'infill_fraction_mean' in diag else np.nan),
                'solve_ms_median': (
                    float(diag['solve_ms_median'].median())
                    if diag is not None
                    and 'solve_ms_median' in diag else np.nan),
                # rolling_diagnostics carries one median and one p95 per
                # site. Both are summarised as the median over sites, so
                # each reads as the typical site's figure; a p95 of the p95
                # column would be a quantile of a quantile.
                'solve_ms_p95': (
                    float(diag['solve_ms_p95'].median())
                    if diag is not None
                    and 'solve_ms_p95' in diag else np.nan),
            })

    if not rows:
        raise SystemExit('No horizon/size combination had a complete run set.')
    return pd.DataFrame(rows).sort_values(['horizon_hours', 'battery_kwh'])


def draw(df: pd.DataFrame, out_path: Path, thesis_dir: Path | None,
         notes: str | None = None) -> None:
    ts.apply_style()
    fig = ts.new_figure(height_in=HEIGHT_IN)
    axes = fig.subplots(1, 2, sharey=True)

    horizons = sorted(df['horizon_hours'].unique())
    others = [h for h in horizons if h != PRODUCTION_HORIZON]
    colours = {h: ts.CATEGORICAL_COLOURS[i % len(ts.CATEGORICAL_COLOURS)]
               for i, h in enumerate(others)}

    panels = (('hh_value_capture', 'Household bill'),
              ('ws_value_capture', 'Net wholesale cost'))
    for ax, (column, title) in zip(axes, panels):
        for hours in horizons:
            sub = df[df['horizon_hours'] == hours].sort_values('battery_kwh')
            emphasised = hours == PRODUCTION_HORIZON
            ax.plot(sub['battery_kwh'], sub[column],
                    marker='o',
                    color=(ts.EMPHASIS_COLOUR if emphasised
                           else colours[hours]),
                    lw=ts.EMPHASIS_LW if emphasised else ts.MEMBER_LW,
                    ms=ts.MARKER_SIZE if emphasised else ts.GRID_MARKS.ms,
                    zorder=3 if emphasised else 2,
                    label=f'{hours} h')
        ax.set_title(title, fontsize=ts.SUBTITLE_SIZE)
        ax.set_xlabel('Battery capacity (kWh)')
        ax.set_xticks(sc.BATTERY_SIZES)
        if notes:
            ax.text(0.02, 0.97, notes, transform=ax.transAxes,
                    ha='left', va='top', fontsize=ts.NOTE_SIZE,
                    color=ts.REFERENCE_COLOUR, bbox=ts.LABEL_BOX, zorder=6)

    axes[0].set_ylabel('Value capture ratio')
    # Keys in horizon order, so the legend reads as a sequence rather than
    # ranking the production setting first.
    handles, labels = axes[0].get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: horizons[i])
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc='outside lower center', ncol=len(labels), frameon=False)
    ts.panel_suptitle(fig, 'Value capture ratio by rolling horizon and battery '
                           'capacity, Amber, imperfect foresight (achievable)')
    sc.save_figure(fig, out_path, thesis_dir)


def print_values(df: pd.DataFrame) -> None:
    print('\nValues for the text (NSW 2024, Amber, achievable)')
    print('-' * 72)
    for size in (20, 50):
        sub = df[df['battery_kwh'] == size]
        if sub.empty:
            continue
        print(f'\n{size} kWh (n = {int(sub["n_sites"].iloc[0])} sites):')
        print(f'  {"horizon":>8} {"household VC":>13} {"wholesale VC":>13} '
              f'{"infill":>8} {"solve ms med":>13} {"solve ms p95":>13}')
        for _, r in sub.sort_values('horizon_hours').iterrows():
            mark = ' *' if r['horizon_hours'] == PRODUCTION_HORIZON else '  '
            print(f'  {int(r["horizon_hours"]):>6} h{mark}'
                  f'{r["hh_value_capture"]:>12.3f} '
                  f'{r["ws_value_capture"]:>13.3f} '
                  f'{r["infill_fraction_mean"]:>8.1%} '
                  f'{r["solve_ms_median"]:>13.1f} '
                  f'{r["solve_ms_p95"]:>13.1f}')

    # Where a horizon longer than production adds more than MATERIAL_GAIN.
    base = df[df['horizon_hours'] == PRODUCTION_HORIZON].set_index('battery_kwh')
    longer = df[df['horizon_hours'] > PRODUCTION_HORIZON]
    print(f'\nBattery sizes where a horizon beyond {PRODUCTION_HORIZON} h adds '
          f'more than {MATERIAL_GAIN:.0%} of household value capture:')
    hits = []
    for size, rows in longer.groupby('battery_kwh'):
        gain = rows['hh_value_capture'].max() - base.loc[size, 'hh_value_capture']
        if gain > MATERIAL_GAIN:
            best = rows.loc[rows['hh_value_capture'].idxmax(), 'horizon_hours']
            hits.append((size, gain, int(best)))
    if hits:
        for size, gain, best in sorted(hits):
            print(f'  {size:>3} kWh: +{gain:.3f} at {best} h')
        print(f'  Smallest such size: {min(h[0] for h in hits)} kWh')
    else:
        print(f'  None. No battery size gains more than {MATERIAL_GAIN:.0%} '
              f'from a longer horizon.')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sc.add_common_args(parser)
    parser.add_argument('--no-notes', action='store_true',
                        help='Leave the data notes off, for the thesis, '
                             'where they belong in the caption.')
    args = parser.parse_args(argv)

    out_dir = args.output_dir or (sc.OUTPUT_ROOT / 'horizon')
    df = collect(args.data_root, args.state, args.year, not args.unweighted)
    sc.write_csv(df, out_dir / 'horizon_value_capture.csv')

    weight_label = 'Unweighted' if args.unweighted else 'Weighted by PV size'
    notes = (f'{args.state} {args.year} (n = {int(df["n_sites"].iloc[0])})\n'
             f'{weight_label}\nCapture against perfect foresight')
    print(f'  Note: {notes.replace(chr(10), ", ")}')
    draw(df, out_dir / 'horizon_value_capture.png', args.thesis_dir,
         None if args.no_notes else notes)
    if args.print_values:
        print_values(df)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
