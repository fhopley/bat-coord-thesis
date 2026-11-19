"""
alt_retailer_composition_figure.py

Thesis figure for the composition of the retailer's mean cohort-month gross
margin under the final offers at the $300/MWh wholesale cap, 20 kWh battery:
cohort hedging and retailer pool retention under daily and monthly
settlement, and the Tesla Electric plan on trial tariff rates. One waterfall
per offer, all on the same six columns so one legend serves every panel:

    subscription / net fixed charges + import margin - cost absorbed
    + credit retained + surcharge retained = gross margin

A column an offer does not have is left empty: import margin exists only for
the Tesla plan and surcharge retained only for cohort hedging.

Reads retailer_composition_means.csv, written by cap_account_retailer.py
(national means per site per month, weighted by state):

    python3 analysis/scripts/cap_account_retailer.py --battery-name 20kWh \\
        --threshold-per-mwh 300 \\
        --subscription-change retailer_x300_daily_a0.9=-5.16 \\
                              retailer_x300_monthly_a0.9=-5.79 \\
        --dynamic dynamic_tesla_electric_mixed_2026_modelled

Format from thesis_style.py (docs/thesis_figure_spec.md).

  python3 analysis/scripts/alt_retailer_composition_figure.py
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from thesis_style import (ANNOTATION_SIZE, CATEGORICAL_COLOURS,  # noqa: E402
                          SEPARATOR_COLOUR, SUBTITLE_SIZE, apply_style,
                          new_figure, panel_suptitle, save_figure)

MEANS_CSV = (REPO_ROOT / 'analysis/output/cap_account_retailer/20kWh/x300/'
             'retailer_composition_means.csv')

TITLE = (r"Composition of the retailer's mean cohort-month gross margin, "
         r"\$300/MWh wholesale cap, 20 kWh")
YLABEL = r'Mean margin (\$ per site per month)'

# Waterfall columns in order: (CSV column, legend entry, sign, colour).
COLUMNS = (
    ('subscription_or_net_fixed', 'Subscription / net fixed charges', 1.0,
     CATEGORICAL_COLOURS[0]),
    ('import_margin', 'Import margin', 1.0, CATEGORICAL_COLOURS[2]),
    ('cost_absorbed', 'Cost absorbed', -1.0, CATEGORICAL_COLOURS[7]),
    ('credit_retained', 'Credit retained', 1.0, CATEGORICAL_COLOURS[3]),
    ('surcharge_retained', 'Surcharge retained', 1.0, CATEGORICAL_COLOURS[4]),
)
GROSS_MARGIN = ('gross_margin', 'Gross margin', 'black')

# Panel (row, column, column span), title and variant, top to bottom.
PANELS = (
    ((0, 0, 1), 'Cohort hedging, daily settlement (1 c/kWh)',
     'cohort_x300_daily_s0.01'),
    ((0, 1, 1), 'Cohort hedging, monthly settlement (1 c/kWh)',
     'cohort_x300_monthly_s0.01'),
    ((1, 0, 1), r'Retailer pool retention, daily settlement ($\alpha$ 0.9)',
     'retailer_x300_daily_a0.9'),
    ((1, 1, 1), r'Retailer pool retention, monthly settlement ($\alpha$ 0.9)',
     'retailer_x300_monthly_a0.9'),
    ((2, 0, 2), 'Tesla Electric (trial tariff rates)',
     'dynamic_tesla_electric_mixed_2026_modelled'),
)

FIG_HEIGHT_IN = 6.2
BAR_WIDTH = 0.72
LABEL_OFFSET_PT = 2
Y_PAD = 0.10            # headroom for the value labels, fraction of range


def fmt(value: float, signed: bool) -> str:
    text = f'{value:+,.2f}' if signed else f'{value:,.2f}'
    return text.replace('-', '−')


def waterfall_extent(rec: pd.Series) -> tuple[float, float]:
    run, lo, hi = 0.0, 0.0, 0.0
    for col, _, sign, _ in COLUMNS:
        if pd.notna(rec[col]):
            run += sign * rec[col]
            lo, hi = min(lo, run), max(hi, run)
    return lo, hi


def shown_columns(means: pd.DataFrame) -> dict[int, list[str]]:
    """Columns drawn in each panel width (the span), in waterfall order: a
    column appears where any panel of that width has it, so panels of equal
    width keep the same slots and line up, and a column no panel of that
    width has takes no room."""
    out = {}
    for span in {p[0][2] for p in PANELS}:
        variants = [v for (_, _, sp), _, v in PANELS if sp == span]
        out[span] = [c for c, _, _, _ in COLUMNS
                     if means.loc[variants, c].notna().any()]
    return out


def draw_waterfall(ax, rec: pd.Series, columns: list[str]) -> None:
    run = 0.0
    prev_end = None
    for i, (col, _, sign, colour) in enumerate(
            [c for c in COLUMNS if c[0] in columns]):
        if pd.isna(rec[col]):
            continue
        delta = sign * rec[col]
        ax.bar(i, delta, bottom=run, width=BAR_WIDTH, color=colour,
               linewidth=0, zorder=3)
        if prev_end is not None:
            ax.plot([prev_end[0] + BAR_WIDTH / 2, i - BAR_WIDTH / 2],
                    [run, run], color=SEPARATOR_COLOUR, lw=0.8, zorder=2)
        run += delta
        ax.annotate(fmt(delta, True), (i, max(run, run - delta)),
                    xytext=(0, LABEL_OFFSET_PT), textcoords='offset points',
                    ha='center', va='bottom', fontsize=ANNOTATION_SIZE,
                    color='black')
        prev_end = (i, run)
    n = len(columns)
    gm_col, _, gm_colour = GROSS_MARGIN
    ax.plot([prev_end[0] + BAR_WIDTH / 2, n - BAR_WIDTH / 2], [run, run],
            color=SEPARATOR_COLOUR, lw=0.8, zorder=2)
    ax.bar(n, rec[gm_col], width=BAR_WIDTH, color=gm_colour, linewidth=0,
           zorder=3)
    ax.annotate(fmt(rec[gm_col], False), (n, rec[gm_col]),
                xytext=(0, LABEL_OFFSET_PT), textcoords='offset points',
                ha='center', va='bottom', fontsize=ANNOTATION_SIZE,
                color='black')
    if abs(run - rec[gm_col]) > 0.005:
        raise ValueError(f"{rec['variant']}: components sum to {run:.3f}, "
                         f"gross margin is {rec[gm_col]:.3f}")


def draw_figure(means: pd.DataFrame):
    means = means.set_index('variant', drop=False)
    extents = [waterfall_extent(means.loc[v]) for _, _, v in PANELS]
    lo = min(e[0] for e in extents)
    hi = max(max(e[1], means.loc[v, 'gross_margin'])
             for e, (_, _, v) in zip(extents, PANELS))
    shown = shown_columns(means)
    pad = Y_PAD * (hi - lo)
    ylim = (lo - (pad if lo < 0 else 0.0), hi + pad)

    fig = new_figure(FIG_HEIGHT_IN)
    gs = fig.add_gridspec(3, 2)
    first = None
    axes = []
    for (row, col, span), title, variant in PANELS:
        n_cols = len(shown[span])
        ax = fig.add_subplot(gs[row, col:col + span], sharey=first)
        first = first or ax
        draw_waterfall(ax, means.loc[variant], shown[span])
        ax.axhline(0, color='black', lw=0.6, zorder=4)
        ax.grid(axis='x', visible=False)
        ax.set_xticks([])
        ax.set_xlim(-0.6, n_cols + 0.6)
        ax.set_title(title, fontsize=SUBTITLE_SIZE)
        if col == 1:
            ax.tick_params(axis='y', labelleft=False)
        axes.append(ax)
    first.set_ylim(*ylim)
    fig.supylabel(YLABEL, fontsize=plt.rcParams['axes.labelsize'])

    # Two rows of three. Matplotlib fills columns first, so the entries are
    # listed down each legend column: subscription over credit retained,
    # import margin over surcharge retained, cost absorbed over gross margin.
    entries = [(COLUMNS[0][1], COLUMNS[0][3]), (COLUMNS[3][1], COLUMNS[3][3]),
               (COLUMNS[1][1], COLUMNS[1][3]), (COLUMNS[4][1], COLUMNS[4][3]),
               (COLUMNS[2][1], COLUMNS[2][3]),
               (GROSS_MARGIN[1], GROSS_MARGIN[2])]
    fig.legend([Patch(facecolor=c, edgecolor='none') for _, c in entries],
               [n for n, _ in entries], loc='outside lower center', ncol=3,
               frameon=False, handlelength=1.2, handleheight=0.9,
               columnspacing=2.0)
    panel_suptitle(fig, TITLE)
    return fig


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--means', type=Path, default=MEANS_CSV)
    ap.add_argument('--out', type=Path, default=REPO_ROOT
                    / 'analysis/output/alt_offers/alt_retailer_composition.png')
    args = ap.parse_args(argv)

    means = pd.read_csv(args.means)
    missing = [v for _, _, v in PANELS if v not in set(means['variant'])]
    if missing:
        raise ValueError(f"{args.means}: no rows for {missing}; re-run "
                         f"cap_account_retailer.py with the final offers")
    apply_style()
    save_figure(draw_figure(means), args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
