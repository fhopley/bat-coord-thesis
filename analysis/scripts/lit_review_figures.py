"""
lit_review_figures.py

Thesis figures for the literature review (Figures 3 to 6), in the format of
docs/thesis_figure_spec.md.

Data are the CSV tables in data/processed/literature_review, extracted from
the working spreadsheet:

  figure3_lhs.csv  cumulative installations, count        (plotted in 000s)
  figure3_rhs.csv  cumulative installed capacity, GWh     (plotted as is)
  figure4.csv      monthly average installed size, kWh
  figure5.csv      AEMO residential battery capacity, GW
  figure6.csv      AEMO VPP participation, fraction of the fleet (plotted in %)

Figures 1 and 2 (coordination dimensions and taxonomy tree) are text and
box diagrams and are kept as slides.

  python3 analysis/scripts/lit_review_figures.py
  python3 analysis/scripts/lit_review_figures.py --only 3 5
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_style import (ANNOTATION_SIZE, DATA_LW, LINE_COLOUR,  # noqa: E402
                          MARKER_SIZE, SUBTITLE_SIZE,
                          apply_style, new_figure, panel_suptitle,
                          save_figure)

DATA_DIR = REPO_ROOT / 'data/processed/literature_review'
LIT_DIR = REPO_ROOT / 'thesis/figures/literature_review'

# ---- Colours. Bars: installs before the Cheaper Home Battery Scheme (CHBS)
# and under it. Lines: AEMO scenario colours, as in the AEMO publications.
PRE_CHBS_COLOUR = LINE_COLOUR
CHBS_COLOUR = '#eb6834'
SCENARIO_COLOURS = {
    'Slower Growth': '#9b2240',
    'Step Change': '#752082',
    'Accelerated Transition': '#40c1ad',
    '2024 Step Change': '#752082',
}
SCENARIO_LABELS = {
    'Slower Growth': 'Slower growth',
    'Step Change': 'Step change',
    'Accelerated Transition': 'Accelerated transition',
    '2024 Step Change': 'Step change (2024 ISP)',
}
SCENARIO_LINESTYLES = {'2024 Step Change': (0, (4, 2))}

LEGEND_LABELS = ['Before Cheaper Home Battery Scheme',
                 'Under Cheaper Home Battery Scheme']
REBATE_NOTE = ('STC Rebate Changes:\n'
               '  First 14 kWh  100% of STCs / kWh\n'
               '  Next 14 kWh  60% of STCs / kWh\n'
               '  Final 22 kWh  15% of STCs / kWh')
BAR_WIDTH = 0.72
FIG3_HEIGHT_IN = 2.8
FIG4_HEIGHT_IN = 3.0
LINE_HEIGHT_IN = 3.0

def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name, encoding='utf-8-sig')


def style_bar_axes(ax) -> None:
    ax.grid(axis='x', visible=False)
    ax.tick_params(axis='x', length=0)


def label_totals(ax, x, totals, fmt) -> None:
    for xi, t in zip(x, totals):
        ax.annotate(fmt(t), (xi, t), xytext=(0, 2), textcoords='offset points',
                    ha='center', va='bottom', fontsize=ANNOTATION_SIZE)


# ---------------------------------------------------------------- Figure 3
def figure_3(out_dir: Path) -> None:
    lhs = read_csv('figure3_lhs.csv').fillna(0.0)
    rhs = read_csv('figure3_rhs.csv').fillna(0.0)
    labels = [str(p) for p in lhs['Year']]
    x = np.arange(len(labels))

    panels = (
        (lhs, 1e3, 'Installations (thousands)', lambda v: f'{v:.0f}'),
        (rhs, 1.0, 'Capacity (GWh)', lambda v: f'{v:.1f}'),
    )
    fig = new_figure(FIG3_HEIGHT_IN)
    axes = fig.subplots(1, 2)
    for ax, (df, scale, subtitle, fmt) in zip(axes, panels):
        pre = df['Pre CHBS'].to_numpy() / scale
        chbs = df['CHBS'].to_numpy() / scale
        ax.bar(x, pre, BAR_WIDTH, color=PRE_CHBS_COLOUR, linewidth=0)
        ax.bar(x, chbs, BAR_WIDTH, bottom=pre, color=CHBS_COLOUR, linewidth=0)
        label_totals(ax, x, pre + chbs, fmt)
        ax.set_title(subtitle, fontsize=SUBTITLE_SIZE)
        ax.set_xticks(x, labels)
        ax.set_ylim(0, (pre + chbs).max() * 1.1)
        style_bar_axes(ax)
    fig.legend([Patch(color=PRE_CHBS_COLOUR), Patch(color=CHBS_COLOUR)],
               LEGEND_LABELS, loc='outside lower center',
               ncol=2, frameon=False)
    panel_suptitle(fig, 'Cumulative home battery installations')
    save_figure(fig, out_dir / '03_cumulative_home_battery_installs')


# ---------------------------------------------------------------- Figure 4
def figure_4(out_dir: Path) -> None:
    df = read_csv('figure4.csv')
    labels = [t.replace('pre CHBS', 'Pre\nCHBS') for t in df['Timeframe']]
    size = df['Install size (kWh)'].to_numpy()
    x = np.arange(len(df))
    colours = [PRE_CHBS_COLOUR if t == 'pre CHBS' else CHBS_COLOUR
               for t in df['Timeframe']]

    fig = new_figure(FIG4_HEIGHT_IN)
    ax = fig.add_subplot()
    ax.bar(x, size, BAR_WIDTH, color=colours, linewidth=0)
    label_totals(ax, x, size, lambda v: f'{v:.1f}')
    ax.set_xticks(x, labels)
    ax.set_ylim(0, size.max() * 1.1)
    ax.set_ylabel('Average installed size (kWh)')
    ax.set_title('Monthly average installed home battery size')
    style_bar_axes(ax)
    # Callout at the rebate change, between Apr-26 and May-26.
    gap = list(df['Timeframe']).index('May-26') - 0.5
    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.annotate(REBATE_NOTE, xy=(gap, 29), xytext=(gap + 0.2, 39),
                ha='left', va='top', fontsize=ANNOTATION_SIZE,
                bbox=dict(facecolor='white', edgecolor='0.7', pad=3.0),
                arrowprops=dict(arrowstyle='-|>', lw=0.6, color='black',
                                shrinkA=0, shrinkB=0, mutation_scale=6))
    ax.legend([Patch(color=PRE_CHBS_COLOUR), Patch(color=CHBS_COLOUR)],
              LEGEND_LABELS, loc='upper left')
    save_figure(fig, out_dir / '04_monthly_average_battery_size')


# ------------------------------------------------------------ Figures 5, 6
def scenario_lines(df: pd.DataFrame, scale: float, ylabel: str, title: str,
                   end_fmt, ylim, out_path: Path, start_label=None,
                   yticks=None) -> None:
    periods = list(df.columns[1:])
    x = np.arange(len(periods))
    series = {r['Scenario']: r[periods].to_numpy(float) * scale
              for _, r in df.iterrows()}
    # Highest line first, so the vertical legend reads like the chart.
    order = sorted(series, key=lambda s: series[s][-1], reverse=True)

    fig = new_figure(LINE_HEIGHT_IN)
    ax = fig.add_subplot()
    handles = []
    for name in order:
        y = series[name]
        ls = SCENARIO_LINESTYLES.get(name, '-')
        colour = SCENARIO_COLOURS[name]
        ax.plot(x, y, color=colour, ls=ls, lw=DATA_LW, marker='o',
                ms=MARKER_SIZE, mew=0, zorder=3)
        ax.annotate(end_fmt(y[-1]), (x[-1], y[-1]), xytext=(5, 0),
                    textcoords='offset points', ha='left', va='center',
                    color=colour, fontsize=ANNOTATION_SIZE)
        handles.append(Line2D([], [], color=colour, ls=ls, lw=DATA_LW,
                              marker='o', ms=MARKER_SIZE, mew=0,
                              label=SCENARIO_LABELS[name]))
    if start_label is not None:
        name, text = start_label
        ax.annotate(text, (x[0], series[name][0]), xytext=(-5, 0),
                    textcoords='offset points', ha='right', va='center',
                    color=SCENARIO_COLOURS[name], fontsize=ANNOTATION_SIZE)
    ax.set_xticks(x, periods)
    ax.set_xlim(-0.45, len(x) - 1 + 0.45)
    ax.set_ylim(*ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(handles=handles, loc='upper left')
    save_figure(fig, out_path)


def figure_5(out_dir: Path) -> None:
    scenario_lines(read_csv('figure5.csv'), 1.0, 'Capacity (GW)',
                   'AEMO residential battery capacity by scenario',
                   lambda v: f'{v:.1f}', (0, 20),
                   out_dir / '05_aemo_battery_capacity_forecast',
                   yticks=range(0, 21, 4))


def figure_6(out_dir: Path) -> None:
    scenario_lines(read_csv('figure6.csv'), 100.0,
                   'VPP participation (% of installed batteries)',
                   'AEMO residential battery VPP participation by scenario',
                   lambda v: f'{v:.0f}%', (0, 70),
                   out_dir / '06_aemo_vpp_participation_forecast',
                   start_label=('Step Change', '17%'),
                   yticks=range(0, 71, 10))


FIGURES = {
    '3': lambda: figure_3(LIT_DIR),
    '4': lambda: figure_4(LIT_DIR),
    '5': lambda: figure_5(LIT_DIR),
    '6': lambda: figure_6(LIT_DIR),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--only', nargs='+', choices=list(FIGURES),
                    help="figures to draw (default all)")
    args = ap.parse_args(argv)
    apply_style()
    for key in args.only or FIGURES:
        FIGURES[key]()
    return 0


if __name__ == '__main__':
    sys.exit(main())
