"""
alt_retailer_spread_figure.py

Thesis figure for the spread of the retailer's gross margin across
cohort-months under Amber, the final capped import price offers and the
Tesla Electric plan, 20 kWh battery, drawn once per wholesale cap
threshold. One box per offer: interquartile range, median, mean, 5th-95th
percentile whiskers and the cohort-months beyond them, all weighted by
state as in cap_account_retailer.py.

The y axis is linear between -$50 and +$50 and logarithmic beyond, so the
Tesla Electric outliers fit without compressing the alternative offers. The
transition is smooth: the log segment continues the linear one at the same
slope, y' = T (1 + ln(|y| / T)). Both thresholds share the axis, so the two
figures read against each other.

Reads retailer_cohort_months.csv, written by cap_account_retailer.py (the
weight column is each cohort-month's national weight), one per threshold:

    python3 analysis/scripts/cap_account_retailer.py --battery-name 20kWh \\
        --threshold-per-mwh 300 \\
        --subscription-change retailer_x300_daily_a0.9=-5.16 \\
                              retailer_x300_monthly_a0.9=-5.79 \\
        --dynamic dynamic_tesla_electric_mixed_2026_modelled

    python3 analysis/scripts/cap_account_retailer.py --battery-name 20kWh \\
        --threshold-per-mwh 150 \\
        --subscription-change retailer_x150_daily_a0.9=-5.18 \\
                              retailer_x150_monthly_a0.66=-25 \\
                              retailer_x150_monthly_a0.9=-6.7 \\
        --dynamic dynamic_tesla_electric_mixed_2026_modelled

The variants shown at each threshold are the four final offers and the two
comparators; the alpha calibrated to a zero subscription fee
(retailer_x300_monthly_a0.6, retailer_x150_monthly_a0.66) is a calibration
point rather than an offer and is not drawn.

Writes a colour version (offer colours: Amber, alternatives, Tesla) and a
black and white version to analysis/output/alt_offers/.

Format from thesis_style.py (docs/thesis_figure_spec.md).

  python3 analysis/scripts/alt_retailer_spread_figure.py
  python3 analysis/scripts/alt_retailer_spread_figure.py --threshold 150
"""
import argparse
import colorsys
import sys
from pathlib import Path
from typing import NamedTuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from thesis_style import (MARKS, REFERENCE_COLOUR, apply_style,  # noqa: E402
                          new_figure, save_figure)

RETAILER_DIR = REPO_ROOT / 'analysis/output/cap_account_retailer/20kWh'
VALUE_COL = 'net_position_per_site'

YLABEL = r'Gross margin (\$ per site per month)'

# Offer colours: Amber's and Tesla's from the retailer registry
# (flow_exposure_analysis.py, retailer_briefing.TESLA), the alternatives'
# from retailer_comparison.ALT_COLOUR. Inks are the step darker used for
# edges and lines on a tint of the colour.
INK_LIGHTNESS = 0.7
AMBER_COLOUR, AMBER_INK = '#2bffa4', '#0e8f5c'
ALT_COLOUR = '#8e3fb5'
TESLA_COLOUR, TESLA_INK = '#c8202f', '#8c1521'
BOX_ALPHA = 0.35


def darker(colour: str) -> str:
    r, g, b = mcolors.to_rgb(colour)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return mcolors.to_hex(colorsys.hls_to_rgb(h, l * INK_LIGHTNESS, s))


ALT_INK = darker(ALT_COLOUR)


class Threshold(NamedTuple):
    """One cap threshold's figure.

    offers:       (variant, colour, ink), left to right.
    group_labels: (first box, last box, text) for the offer name centred
                  over its boxes, the first of the two x label lines.
    tick_labels:  what distinguishes each box within its offer, the
                  second line.
    stem:         output file stem, before the colour/mono suffix.
    """
    cap: str
    offers: tuple
    group_labels: tuple
    tick_labels: tuple
    stem: str


# The surcharge rate the cohort pool needs differs by settlement period at
# $150/MWh (3 c/kWh daily, 1 c/kWh monthly) but not at $300/MWh, so at
# $150 the rate sits on the per-box label rather than the offer name.
THRESHOLDS = {
    '300': Threshold(
        cap='300',
        offers=(
            ('Amber', AMBER_COLOUR, AMBER_INK),
            ('cohort_x300_daily_s0.01', ALT_COLOUR, ALT_INK),
            ('cohort_x300_monthly_s0.01', ALT_COLOUR, ALT_INK),
            ('retailer_x300_daily_a0.9', ALT_COLOUR, ALT_INK),
            ('retailer_x300_monthly_a0.9', ALT_COLOUR, ALT_INK),
            ('dynamic_tesla_electric_mixed_2026_modelled', TESLA_COLOUR, TESLA_INK),
        ),
        group_labels=(
            (0, 0, 'Amber'),
            (1, 2, 'Cohort hedging (1 c/kWh)'),
            (3, 4, 'Retailer pool retention ($\\alpha$ 0.9)'),
            (5, 5, 'Tesla Electric'),
        ),
        tick_labels=('', 'daily', 'monthly',
                     'daily', 'monthly', '(trial tariff)'),
        stem='alt_retailer_spread',
    ),
    '150': Threshold(
        cap='150',
        offers=(
            ('Amber', AMBER_COLOUR, AMBER_INK),
            ('cohort_x150_daily_s0.03', ALT_COLOUR, ALT_INK),
            ('cohort_x150_monthly_s0.01', ALT_COLOUR, ALT_INK),
            ('retailer_x150_daily_a0.9', ALT_COLOUR, ALT_INK),
            ('retailer_x150_monthly_a0.9', ALT_COLOUR, ALT_INK),
            ('dynamic_tesla_electric_mixed_2026_modelled', TESLA_COLOUR, TESLA_INK),
        ),
        group_labels=(
            (0, 0, 'Amber'),
            (1, 2, 'Cohort hedging'),
            (3, 4, 'Retailer pool retention ($\\alpha$ 0.9)'),
            (5, 5, 'Tesla Electric'),
        ),
        tick_labels=('', 'daily (3 c/kWh)', 'monthly (1 c/kWh)',
                     'daily', 'monthly', '(trial tariff)'),
        stem='alt_retailer_spread_x150',
    ),
}

# Axis: linear within +-LINEAR_LIMIT, logarithmic beyond. Shared by both
# thresholds so the figures are comparable.
LINEAR_LIMIT = 50.0
Y_LIM = (-285.0, 115.0)
# 25 is Amber's subscription; -25 balances it about zero.
Y_TICKS = [-200, -100, -50, -25, 0, 25, 50, 100]

FIG_HEIGHT_IN = 3.4
BOX_WIDTH = 0.6
BOX_LW = 0.8
FLIER_SIZE = 2.2
MEAN_SIZE = 3.6
NEUTRAL = '0.2'


def forward(y):
    y = np.asarray(y, dtype=float)
    mag = np.abs(y)
    out = np.where(mag <= LINEAR_LIMIT, y,
                   np.sign(y) * LINEAR_LIMIT * (
                       1 + np.log(np.maximum(mag, LINEAR_LIMIT)
                                  / LINEAR_LIMIT)))
    return out


def inverse(z):
    z = np.asarray(z, dtype=float)
    mag = np.abs(z)
    return np.where(mag <= LINEAR_LIMIT, z,
                    np.sign(z) * LINEAR_LIMIT * np.exp(
                        (np.maximum(mag, LINEAR_LIMIT) - LINEAR_LIMIT)
                        / LINEAR_LIMIT))


def weighted_percentile(values, weights, q):
    """Cumulative-weight interpolation on the midpoint convention, as in
    cap_account_retailer.py."""
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(q / 100.0, cum, v))


def box_stats(values: np.ndarray, weights: np.ndarray) -> dict:
    lo = weighted_percentile(values, weights, 5.0)
    hi = weighted_percentile(values, weights, 95.0)
    return {
        'med': weighted_percentile(values, weights, 50.0),
        'q1': weighted_percentile(values, weights, 25.0),
        'q3': weighted_percentile(values, weights, 75.0),
        'whislo': lo, 'whishi': hi,
        'mean': float(np.average(values, weights=weights)),
        'fliers': values[(values < lo) | (values > hi)],
    }


def draw_figure(cohort_months: pd.DataFrame, spec: Threshold, colour: bool):
    stats = []
    for variant, _, _ in spec.offers:
        m = cohort_months['variant'] == variant
        if not m.any():
            raise ValueError(f"no rows for {variant}; re-run "
                             f"cap_account_retailer.py")
        stats.append(box_stats(cohort_months.loc[m, VALUE_COL].to_numpy(),
                               cohort_months.loc[m, 'weight'].to_numpy()))
    lo = min(min(s['whislo'], s['fliers'].min() if len(s['fliers']) else 1e9)
             for s in stats)
    hi = max(max(s['whishi'], s['fliers'].max() if len(s['fliers']) else -1e9)
             for s in stats)
    if lo < Y_LIM[0] or hi > Y_LIM[1]:
        raise ValueError(f"data span {lo:.1f} to {hi:.1f}, outside the axis "
                         f"limits {Y_LIM}")

    fig = new_figure(FIG_HEIGHT_IN)
    ax = fig.subplots()
    ax.set_yscale('function', functions=(forward, inverse))
    for pos, (s, (_, fill, ink)) in enumerate(zip(stats, spec.offers)):
        edge = ink if colour else 'black'
        ax.bxp([s], positions=[pos], widths=BOX_WIDTH, patch_artist=True,
               showmeans=True, showfliers=True, manage_ticks=False,
               boxprops=dict(facecolor=fill if colour else 'white',
                             edgecolor=edge, linewidth=BOX_LW),
               whiskerprops=dict(color=edge, linewidth=BOX_LW),
               capprops=dict(color=edge, linewidth=BOX_LW),
               medianprops=dict(color=edge,
                                linewidth=MARKS.lw,
                                linestyle='--' if colour else '-'),
               meanprops=dict(marker='D', markerfacecolor='white',
                              markeredgecolor=edge, markeredgewidth=0.7,
                              markersize=MEAN_SIZE),
               flierprops=dict(marker='o', markersize=FLIER_SIZE,
                               markerfacecolor=fill if colour else '0.45',
                               markeredgecolor=edge if colour else 'none',
                               markeredgewidth=0.4, alpha=0.7))
    if colour:
        # A tint of the offer colour for the box, without tinting its edge
        # or its median line.
        for patch, (_, fill, _) in zip(ax.patches, spec.offers):
            patch.set_facecolor(mcolors.to_rgba(fill, BOX_ALPHA))

    ax.set_ylim(*Y_LIM)
    ax.set_yticks(Y_TICKS)
    ax.set_yticklabels([f'{t:d}'.replace('-', '−') for t in Y_TICKS])
    ax.minorticks_off()
    ax.grid(axis='x', visible=False)
    ax.set_xlim(-0.6, len(spec.offers) - 0.4)
    ax.set_xticks(range(len(spec.offers)))
    # The empty first line leaves room for the offer names, which are
    # drawn at the top of the tick label area.
    ax.set_xticklabels(['\n' + t if t else '' for t in spec.tick_labels])
    ax.tick_params(axis='x', length=0)
    for first, last, text in spec.group_labels:
        ax.annotate(text, xy=((first + last) / 2, 0),
                    xycoords=('data', 'axes fraction'), xytext=(0, -9.0),
                    textcoords='offset points', ha='center', va='baseline',
                    annotation_clip=False)
    ax.set_ylabel(YLABEL)
    ax.set_title(r"Spread of the retailer's gross margin across "
                 r"cohort-months, \$" + spec.cap +
                 r"/MWh wholesale cap, 20 kWh")

    line = NEUTRAL if not colour else NEUTRAL
    handles = [
        Line2D([], [], marker='D', linestyle='none', markersize=MEAN_SIZE,
               markerfacecolor='white', markeredgecolor=line,
               markeredgewidth=0.7, label='Mean'),
        Line2D([], [], color=line, lw=MARKS.lw,
               linestyle='--' if colour else '-', label='Median'),
        Patch(facecolor='0.85' if colour else 'white', edgecolor=line,
              linewidth=BOX_LW, label='Interquartile range (25–75)'),
        Line2D([], [], color=line, lw=BOX_LW,
               label='Whiskers (5–95)'),
        Line2D([], [], marker='o', linestyle='none', markersize=FLIER_SIZE,
               markerfacecolor='0.45', markeredgecolor='none', alpha=0.7,
               label='Cohort-month outliers'),
    ]
    ax.legend(handles=handles, loc='lower left', handlelength=1.6,
              borderpad=0.4, borderaxespad=0.5)
    return fig, stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--threshold', nargs='*', choices=sorted(THRESHOLDS),
                    default=sorted(THRESHOLDS, reverse=True),
                    help="Cap thresholds, $/MWh. Default: all.")
    ap.add_argument('--retailer-dir', type=Path, default=RETAILER_DIR,
                    help="cap_account_retailer.py output for the battery "
                         "size, holding one x{threshold} directory per cap.")
    ap.add_argument('--out-dir', type=Path,
                    default=REPO_ROOT / 'analysis/output/alt_offers')
    args = ap.parse_args(argv)

    apply_style()
    for cap in args.threshold:
        spec = THRESHOLDS[cap]
        csv = args.retailer_dir / f'x{cap}' / 'retailer_cohort_months.csv'
        cohort_months = pd.read_csv(csv,
                                    usecols=['variant', VALUE_COL, 'weight'])
        for suffix, colour in (('colour', True), ('mono', False)):
            fig, stats = draw_figure(cohort_months, spec, colour)
            if colour:
                print(f"\n${cap}/MWh cap")
                print(f"{'variant':36s} {'mean':>7s} {'median':>7s} "
                      f"{'q1':>7s} {'q3':>7s} {'p5':>7s} {'p95':>7s} "
                      f"outliers")
                for (v, *_), s in zip(spec.offers, stats):
                    print(f"{v:36s} {s['mean']:7.2f} {s['med']:7.2f} "
                          f"{s['q1']:7.2f} {s['q3']:7.2f} {s['whislo']:7.2f} "
                          f"{s['whishi']:7.2f} {len(s['fliers'])}")
            save_figure(fig, args.out_dir / f'{spec.stem}_{suffix}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
