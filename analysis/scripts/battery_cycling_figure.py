"""Battery cycling against warranty allowances and VPP forced flows against
their caps, under imperfect foresight (achievable), as specified in
docs/battery_cycling_chart_spec.md.

Reads the CSVs flow_exposure_analysis.py writes on the achievable basis
(<exposure-dir>/imperfect_foresight_achievable/national/) and draws:

  battery_cycling_and_forced_flows   the thesis figure: equivalent full cycles
                                     per day by offer against the warranty
                                     allowances, beside each VPP offer's
                                     forced discharge, charge and total against
                                     its annual cap. Written beside the CSVs
                                     and to --thesis-dir.
  cycles_by_battery_size             exploratory: cycles per day against
                                     battery size by offer, from
                                     battery_cycling_by_size.csv (written by
                                     flow_exposure_analysis.py --cycling-sweep).

It also prints the values the text cites.

Usage (from project root, after flow_exposure_analysis.py):
    python3 analysis/scripts/battery_cycling_figure.py
    python3 analysis/scripts/battery_cycling_figure.py --kwh 20
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import financial_analysis as fa            # noqa: E402
import flow_exposure_analysis as fea       # noqa: E402
import savings_bar_figure as sbf           # noqa: E402
import thesis_style as ts                  # noqa: E402

BASIS = 'achievable'
HEIGHT_IN = 3.0
TITLE = 'Battery cycling and VPP forced flows at {kwh} kWh'
LEFT_SUBTITLE = 'Battery cycling against warranty allowances'
RIGHT_SUBTITLE = 'VPP forced flows against annual caps'
CYCLES_LABEL = 'Equivalent full cycles per day'
CYCLES_TOP = 1.3
# Tick labels on two lines where one would crowd the narrower panels.
TICK_LABELS = {'Self-consumption': 'Self-\nconsumption',
               'EnergyAustralia': 'Energy\nAustralia'}
FORCED_TOP = 300

# Warranty throughput allowances (Methodology, Battery Utilisation and
# Warranty Implications), in MWh/yr of discharge for a 20 kWh battery. In
# cycles per day they are the same at every battery size.
WARRANTY_KWH = 20
WARRANTY_ALLOWANCES = (
    ('Tesla Powerwall', 5.6),
    ('SigEnergy, FoxESS (low)', 7.4),
    ('FoxESS (high)', 8.4),
)


# Usable capacities (kWh) a battery line can be installed at, by warranty
# line. Each entry is (unit usable kWh, minimum stack, maximum stack).
# Combinations above MAX_MARKER_KWH are not marked.
MAX_MARKER_KWH = 50
SIZE_CHART_LINE_ALPHA = 0.35   # lines fainter than the size markers
SIZE_MARKER = 'D'              # differs from the data lines' round markers
SIZE_MARKER_SCALE = 0.65
PRODUCT_UNITS = {
    'Tesla Powerwall': [(13.5, 1, 3)],
    'SigEnergy, FoxESS (low)': [(5.2, 1, 6), (6.0, 1, 6), (7.8, 1, 6),
                                (9.0, 1, 6), (10.34, 1, 4), (11.58, 1, 4)],
    'FoxESS (high)': [(4.66, 1, 9), (5.46, 2, 9)],
}


def product_sizes(name: str) -> list[float]:
    """Installable usable capacities (kWh) for a warranty line, up to
    MAX_MARKER_KWH."""
    sizes = {round(unit * n, 2)
             for unit, lo, hi in PRODUCT_UNITS[name]
             for n in range(lo, hi + 1)
             if unit * n <= MAX_MARKER_KWH}
    return sorted(sizes)


def warranty_cycles() -> list[tuple[str, float]]:
    return [(name, mwh * 1000.0 / WARRANTY_KWH / 365.0)
            for name, mwh in WARRANTY_ALLOWANCES]


def offer_colours() -> dict[str, str]:
    """Display name -> colour, from flow_exposure_analysis.py's registry."""
    colours = {fea.SELF_CONSUMPTION_NAME: sbf.KEY_COLOUR}
    for p in fea._BASE_PLANS:
        colours[fea.OFFER_NAMES.get(p['label'], p['label'])] = p['color']
    return colours


def offer_inks() -> dict[str, str]:
    """Display name -> marker edge colour, darker where the colour itself is
    too light to outline a marker on white."""
    inks = {}
    for p in fea._BASE_PLANS:
        inks[fea.OFFER_NAMES.get(p['label'], p['label'])] = fa.PLAN_INK.get(
            p['sc_mode'], p['color'])
    return inks


def draw_warranty_lines(ax, label_x: float, ha: str,
                        mark_sizes: bool = False) -> None:
    """Dashed reference lines, labelled at label_x (axes fraction) above the
    line. With mark_sizes, each line also carries a marker at every battery
    capacity its product line comes in (x axis in kWh)."""
    for name, cycles in warranty_cycles():
        ax.axhline(cycles, color=ts.REFERENCE_COLOUR, lw=ts.REFERENCE_LW,
                   ls='--', zorder=2,
                   alpha=SIZE_CHART_LINE_ALPHA if mark_sizes else 1.0)
        if mark_sizes:
            sizes = product_sizes(name)
            ax.plot(sizes, [cycles] * len(sizes), ls='none', marker=SIZE_MARKER,
                    ms=ts.MARKER_SIZE * SIZE_MARKER_SCALE, mfc=ts.REFERENCE_COLOUR,
                    mec=ts.REFERENCE_COLOUR, zorder=2, clip_on=True)
        ax.annotate(f'{name} {cycles:.2f}', (label_x, cycles),
                    xycoords=('axes fraction', 'data'), xytext=(0, 1.5),
                    textcoords='offset points', ha=ha, va='bottom',
                    fontsize=ts.ANNOTATION_SIZE, color=ts.REFERENCE_COLOUR,
                    bbox=ts.LABEL_BOX, zorder=4)


def _two_line_ticks(ax) -> None:
    ax.set_xticklabels([TICK_LABELS.get(t.get_text(), t.get_text())
                        for t in ax.get_xticklabels()])


def make_thesis_figure(table: pd.DataFrame, kwh: int, out_paths: list[Path]) -> None:
    colours = offer_colours()
    fig = ts.new_figure(HEIGHT_IN)
    left, right = fig.subplots(1, 2)

    bars = [(r.offer, colours.get(r.offer, sbf.KEY_COLOUR), r.cycles_per_day)
            for r in table.itertuples()]
    top = fea.draw_cycles_bars(left, bars)
    for i, (_, _, cycles) in enumerate(bars):
        # Boxed: a label can sit on a warranty line.
        sbf._label(left, i, cycles, f'{cycles:.2f}', bbox=ts.LABEL_BOX)
    _two_line_ticks(left)
    draw_warranty_lines(left, 0.99, 'right')
    left.set_ylim(0, max(CYCLES_TOP, top * 1.12))
    left.set_ylabel(CYCLES_LABEL)
    left.set_title(LEFT_SUBTITLE, fontsize=ts.SUBTITLE_SIZE)

    vpp = table[table['cap_kwh'].notna()]
    groups = [(r.offer, colours.get(r.offer, sbf.KEY_COLOUR),
               r.forced_discharge_kwh, r.forced_charge_kwh)
              for r in vpp.itertuples()]
    xpos, top = fea.draw_waterfall(right, groups, lambda v: f'{v:,.0f}')
    fea.draw_caps(right, xpos, {r.offer: (r.cap_kwh, r.cap_column)
                                for r in vpp.itertuples()}, outer=True)
    right.set_ylim(0, max(FORCED_TOP, top * 1.12, vpp['cap_kwh'].max() * 1.12))
    right.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    _two_line_ticks(right)
    right.set_ylabel('Forced flow (kWh/yr)')
    right.set_title(RIGHT_SUBTITLE, fontsize=ts.SUBTITLE_SIZE)
    handles = [mpatches.Patch(**fea._column_style(c, sbf.KEY_COLOUR))
               for c in fea.WATERFALL_COLUMNS]
    # Below the panel, where no cap or value label can sit.
    right.legend(handles=handles,
                 labels=[fea.WATERFALL_KEY[c] for c in fea.WATERFALL_COLUMNS],
                 loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3,
                 frameon=False)

    ts.panel_suptitle(fig, TITLE.format(kwh=kwh))
    for path in out_paths:
        fig.savefig(path.with_suffix('.png'), dpi=ts.DPI)
        print(f'  Saved: {path.with_suffix(".png")}')
    plt.close(fig)


def make_size_chart(sweep: pd.DataFrame, order: list[str], out_path: Path) -> None:
    colours, inks = offer_colours(), offer_inks()
    fig = ts.new_figure(HEIGHT_IN)
    ax = fig.subplots()
    for offer in [o for o in order if o in set(sweep['offer'])]:
        d = sweep[sweep['offer'] == offer].sort_values('battery_kwh')
        colour = colours.get(offer, sbf.KEY_COLOUR)
        ax.plot(d['battery_kwh'], d['cycles_per_day'], color=colour,
                marker='o', ms=ts.MARKER_SIZE, mec=inks.get(offer, colour),
                mew=ts.MARKER_EDGE_WIDTH, label=offer, zorder=3)
    draw_warranty_lines(ax, 0.99, 'right', mark_sizes=True)
    sizes = fea.BATTERY_SIZES
    pad = 0.03 * (sizes[-1] - sizes[0])
    ax.set_xlim(sizes[0] - pad, sizes[-1] + pad)
    ax.set_xticks(sizes)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.set_ylim(0, max(CYCLES_TOP, sweep['cycles_per_day'].max() * 1.1))
    ax.set_xlabel('Battery capacity (kWh)')
    ax.set_ylabel(CYCLES_LABEL)
    ax.set_title(f'Battery cycling by battery capacity, '
                 f'{fa.RUN_BASES[BASIS]["title"]}')
    handles, labels = ax.get_legend_handles_labels()
    # Fourth column: the warranty line and the stack-size marker.
    handles += [
        mlines.Line2D([], [], color=ts.REFERENCE_COLOUR, lw=ts.REFERENCE_LW,
                      ls='--', alpha=SIZE_CHART_LINE_ALPHA),
        mlines.Line2D([], [], ls='none', marker=SIZE_MARKER,
                      ms=ts.MARKER_SIZE * SIZE_MARKER_SCALE,
                      mfc=ts.REFERENCE_COLOUR, mec=ts.REFERENCE_COLOUR)]
    labels += ['Battery warranty', 'Battery stack size']
    ax.legend(handles, labels, loc='lower left', ncol=4)
    ts.save_figure(fig, out_path)


def print_text_values(table: pd.DataFrame, sweep: pd.DataFrame | None, kwh: int) -> None:
    print(f'\nAt {kwh} kWh (national, achievable):')
    print(f"  {'Offer':18s} {'Discharge kWh/yr':>17s} {'Cycles/day':>11s}")
    for r in table.itertuples():
        print(f'  {r.offer:18s} {r.annual_discharge_kwh:17,.0f} {r.cycles_per_day:11.2f}')
    print('\nForced flow on the capped quantity, as a share of annual discharge:')
    for r in table[table['cap_kwh'].notna()].itertuples():
        capped = (r.forced_discharge_kwh if r.cap_column == 'discharge'
                  else r.forced_total_kwh)
        print(f'  {r.offer:18s} {r.cap_column:9s} {capped:7,.0f} kWh of '
              f'{r.annual_discharge_kwh:,.0f} = {capped / r.annual_discharge_kwh:.1%}'
              f'  (cap {r.cap_kwh:,.0f})')
    if sweep is None:
        return
    print('\nCycles per day by battery capacity:')
    for offer in (fea.SELF_CONSUMPTION_NAME, 'Amber'):
        d = sweep[sweep['offer'] == offer].set_index('battery_kwh')['cycles_per_day']
        vals = ', '.join(f'{k} kWh {d[k]:.2f}' for k in (10, 20, 50) if k in d.index)
        print(f'  {offer:18s} {vals}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--exposure-dir', type=Path,
                        default=Path('analysis/output/exposure'),
                        help="flow_exposure_analysis.py's output directory")
    parser.add_argument('--thesis-dir', type=Path,
                        default=Path('thesis/figures/results'),
                        help='Where the thesis copy of the figure is written')
    parser.add_argument('--kwh', type=int, default=20,
                        help='Battery capacity of the thesis figure')
    args = parser.parse_args(argv)

    # The imported analysis modules set their own rcParams on import.
    matplotlib.rcdefaults()
    ts.apply_style()

    national = args.exposure_dir / fa.RUN_BASES[BASIS]['folder'] / 'national'
    table_path = national / f'battery_cycling_{args.kwh}kwh.csv'
    if not table_path.exists():
        sys.exit(f'No {table_path}; run flow_exposure_analysis.py on the '
                 f'achievable basis first.')
    table = pd.read_csv(table_path)
    args.thesis_dir.mkdir(parents=True, exist_ok=True)
    make_thesis_figure(table, args.kwh,
                       [national / 'battery_cycling_and_forced_flows',
                        args.thesis_dir / 'battery_cycling_and_forced_flows'])

    sweep_path = national / 'battery_cycling_by_size.csv'
    sweep = None
    if sweep_path.exists():
        sweep = pd.read_csv(sweep_path)
        make_size_chart(sweep, list(table['offer']),
                        national / 'cycles_by_battery_size')
    else:
        print(f'  No {sweep_path}; run flow_exposure_analysis.py with '
              f'--cycling-sweep for the battery size chart.')
    print_text_values(table, sweep, args.kwh)


if __name__ == '__main__':
    main()
