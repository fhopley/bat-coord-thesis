"""
alt_offers_by_size_figures.py

Thesis figures for how the alternative offers' retailer and household
outcomes change with battery size (docs/cap_account_battery_size_spec.md),
at the $300/MWh cap, daily settlement on the left and monthly on the right.

  alt_retailer_margin_by_size          mean gross margin per household per
                                       month, with Retailer Pool Retention's
                                       subscription
  alt_retailer_spread_by_size          25-75 and 5-95 percentile bands of the
                                       monthly gross margin, the median and
                                       the worst month
  alt_household_bill_impact_by_size    5-95 percentile band and median of the
                                       annual bill change against Amber, as a
                                       share of coordination value

Reads offer_outcomes_by_size.csv, written by offer_outcomes_by_size.py:

    python3 analysis/scripts/offer_outcomes_by_size.py
    python3 analysis/scripts/alt_offers_by_size_figures.py

Writes to analysis/output/alt_offers/ and copies each figure to
thesis/figures/results/.

Format from thesis_style.py (docs/thesis_figure_spec.md).
"""
import argparse
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from alt_retailer_spread_figure import (LINEAR_LIMIT, forward,  # noqa: E402
                                        inverse)
from thesis_style import (CATEGORICAL_COLOURS, MARKS,  # noqa: E402
                          SUBTITLE_SIZE,
                          apply_style, new_figure, panel_suptitle,
                          save_figure)

OUTCOMES_CSV = (REPO_ROOT / 'analysis/output/cap_account_retailer/by_size/'
                'offer_outcomes_by_size.csv')
OUT_DIR = REPO_ROOT / 'analysis/output/alt_offers'
THESIS_DIR = REPO_ROOT / 'thesis/figures/results'
FIG_HEIGHT_IN = 3.0

# Offer colours as in alt_household_bill_impact_figure.py.
COHORT = ('Cohort hedging (1 c/kWh)', CATEGORICAL_COLOURS[0])
RETAILER = (r'Retailer pool retention ($\alpha$ 0.9)', CATEGORICAL_COLOURS[2])
TESLA = ('Tesla Electric', CATEGORICAL_COLOURS[7])

# Panel -> series (offer, settlement period, label, colour).
PANELS = (
    ('Daily settlement', 'daily', (
        ('Cohort Hedging', *COHORT),
        ('Retailer Pool Retention', *RETAILER),
    )),
    ('Monthly settlement', 'monthly', (
        ('Cohort Hedging', *COHORT),
        ('Retailer Pool Retention', *RETAILER),
        ('Tesla Electric', *TESLA),
    )),
)

XLABEL = 'Battery size (kWh)'
BAND_OUTER_ALPHA = 0.12
BAND_INNER_ALPHA = 0.28
WORST_MARKER_SIZE = 2.6


def series_rows(df: pd.DataFrame, offer: str, period: str) -> pd.DataFrame:
    rows = df[(df['offer'] == offer) & (df['settlement_period'] == period)]
    if rows.empty:
        raise ValueError(f"no rows for {offer}, {period}; re-run "
                         f"offer_outcomes_by_size.py")
    return rows.sort_values('battery_kwh')


def new_panels(df: pd.DataFrame, ylabel: str):
    fig = new_figure(FIG_HEIGHT_IN)
    axes = fig.subplots(1, 2, sharey=True)
    sizes = np.sort(df['battery_kwh'].unique())
    pad = 0.03 * (sizes.max() - sizes.min())
    for ax, (title, _, _) in zip(axes, PANELS):
        ax.set_title(title, fontsize=SUBTITLE_SIZE)
        ax.set_xlabel(XLABEL)
        ax.set_xticks(sizes)
        ax.set_xticklabels([f'{s:g}' for s in sizes])
        ax.set_xlim(sizes.min() - pad, sizes.max() + pad)
        ax.minorticks_off()
    axes[0].set_ylabel(ylabel)
    return fig, axes


def finish(fig, title: str, handles: list) -> None:
    # One row where it fits the text width, two otherwise.
    ncols = len(handles) if len(handles) <= 5 else -(-len(handles) // 2)
    fig.legend(handles=handles, loc='outside lower center', ncols=ncols,
               frameon=False, handlelength=1.6, columnspacing=1.2)
    panel_suptitle(fig, title)


def offer_handles(df: pd.DataFrame) -> list:
    """One entry per offer present, in panel order."""
    seen, handles = set(), []
    for _, period, series in PANELS:
        for offer, label, colour in series:
            if label not in seen:
                seen.add(label)
                handles.append(Line2D([], [], color=colour, lw=MARKS.lw,
                                      label=label))
    return handles


def margin_figure(df: pd.DataFrame):
    fig, axes = new_panels(df, r'Gross margin (\$ per household per month)')
    for ax, (_, period, series) in zip(axes, PANELS):
        for offer, label, colour in series:
            r = series_rows(df, offer, period)
            x = r['battery_kwh']
            ax.plot(x, r['mean_gross_margin'], color=colour, lw=MARKS.lw,
                    marker='o', ms=MARKS.ms, zorder=4)
            if offer == 'Retailer Pool Retention':
                ax.plot(x, r['subscription_or_net_fixed'], color=colour,
                        lw=MARKS.lw, ls=':', zorder=3)
    handles = offer_handles(df) + [
        Line2D([], [], color='0.2', lw=MARKS.lw, ls=':',
               label='Retailer pool retention subscription'),
    ]
    finish(fig, r'Retailer gross margin by battery size, \$300/MWh wholesale '
                'cap', handles)
    return fig


def _log_axis(ax, lo: float, hi: float) -> None:
    """Linear within +-LINEAR_LIMIT and logarithmic beyond, as in
    alt_retailer_spread_figure.py; limits padded in the transformed scale."""
    ax.set_yscale('function', functions=(forward, inverse))
    pad = 0.06 * (forward(hi) - forward(lo))
    ax.set_ylim(float(inverse(forward(lo) - pad)),
                float(inverse(forward(hi) + pad)))
    ticks = [t for t in (-200, -100, -50, -25, 0, 25, 50, 100, 200)
             if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{t:d}'.replace('-', '−') for t in ticks])
    ax.minorticks_off()


def spread_figure(df: pd.DataFrame):
    fig, axes = new_panels(df, r'Gross margin (\$ per household per month)')
    for ax, (_, period, series) in zip(axes, PANELS):
        for offer, label, colour in series:
            r = series_rows(df, offer, period)
            x = r['battery_kwh'].to_numpy()
            ax.fill_between(x, r['p5_gross_margin'], r['p95_gross_margin'],
                            color=colour, alpha=BAND_OUTER_ALPHA, lw=0,
                            zorder=2)
            ax.fill_between(x, r['p25_gross_margin'], r['p75_gross_margin'],
                            color=colour, alpha=BAND_INNER_ALPHA, lw=0,
                            zorder=3)
            ax.plot(x, r['p50_gross_margin'], color=colour, lw=MARKS.lw,
                    zorder=4)
            ax.plot(x, r['worst_month_gross_margin'], color=colour, lw=0,
                    marker='v', ms=WORST_MARKER_SIZE, mew=0, zorder=5)
    lo = min(df['worst_month_gross_margin'].min(), df['p5_gross_margin'].min())
    hi = df['p95_gross_margin'].max()
    _log_axis(axes[0], lo, hi)
    handles = offer_handles(df) + [
        Line2D([], [], color='0.2', lw=MARKS.lw, label='Median'),
        Patch(facecolor='0.2', alpha=BAND_INNER_ALPHA, lw=0,
              label='25–75 percentile'),
        Patch(facecolor='0.2', alpha=BAND_OUTER_ALPHA, lw=0,
              label='5–95 percentile'),
        Line2D([], [], color='0.2', lw=0, marker='v', ms=WORST_MARKER_SIZE,
               mew=0, label='Worst month'),
    ]
    finish(fig, r'Spread of the retailer gross margin by battery size, '
                r'\$300/MWh wholesale cap', handles)
    return fig


def household_figure(df: pd.DataFrame):
    fig, axes = new_panels(
        df, 'Annual bill change against Amber (% of coordination value)')
    for ax, (_, period, series) in zip(axes, PANELS):
        ax.axhline(0, color='0.65', lw=0.5, zorder=2)
        for offer, label, colour in series:
            r = series_rows(df, offer, period)
            x = r['battery_kwh'].to_numpy()
            ax.fill_between(x, r['p5_pct_of_coordination_savings'],
                            r['p95_pct_of_coordination_savings'],
                            color=colour, alpha=BAND_INNER_ALPHA, lw=0,
                            zorder=3)
            ax.plot(x, r['p50_pct_of_coordination_savings'], color=colour,
                    lw=MARKS.lw, zorder=4)
    handles = offer_handles(df) + [
        Line2D([], [], color='0.2', lw=MARKS.lw, label='Median'),
        Patch(facecolor='0.2', alpha=BAND_INNER_ALPHA, lw=0,
              label='5–95 percentile'),
    ]
    finish(fig, r'Annual household bill change by battery size, \$300/MWh '
                'wholesale cap', handles)
    return fig


FIGURES = (
    ('alt_retailer_margin_by_size', margin_figure),
    ('alt_retailer_spread_by_size', spread_figure),
    ('alt_household_bill_impact_by_size', household_figure),
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--outcomes', type=Path, default=OUTCOMES_CSV)
    ap.add_argument('--out-dir', type=Path, default=OUT_DIR)
    ap.add_argument('--thesis-dir', type=Path, default=THESIS_DIR,
                    help="Each figure is also copied here. Pass '' to skip.")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.outcomes)
    apply_style()
    for stem, draw in FIGURES:
        fig = draw(df)
        save_figure(fig, args.out_dir / stem)
        if str(args.thesis_dir):
            args.thesis_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(args.out_dir / f'{stem}.png',
                        args.thesis_dir / f'{stem}.png')
    return 0


if __name__ == '__main__':
    sys.exit(main())
