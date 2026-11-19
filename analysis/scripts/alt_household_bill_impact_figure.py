"""
alt_household_bill_impact_figure.py

Thesis figure comparing the annual household bill change against Amber,
as a share of the coordination savings, across the alternative capped
import price offers, drawn once per wholesale cap threshold: daily
settlement on the left, monthly settlement on the right. Weighted step
histograms of site-years, pooled nationally (NSW, QLD, SA, VIC,
2021-2025), 20 kWh battery.

Reads the site-year data written by cap_account_aggregate.py:

    {aggregate-root}/{battery}/{variant}/national/
        pct_of_coordination_savings_histogram_weighted[_sub<fee>]_site_years.csv

(columns state, year, site_id, value, weight; the weight is the national
PV band x year x state weight the aggregate histogram uses). Shares are
the weight in each bin over the total weight, in percent. Site-years
beyond the axis range are counted in the end bins.

Each retailer variant names the subscription change it is drawn at, in
SUB below, and that file is read by name. A variant aggregated at several
fees holds one site-year file per fee, so reading whichever file happens
to be there would silently draw a different offer than the one the
retailer figures are calibrated to.

Writes two versions per threshold, with and without the Tesla Electric
dynamic plan (monthly panel only), to analysis/output/alt_offers/.

Format from thesis_style.py (docs/thesis_figure_spec.md).

  python3 analysis/scripts/alt_household_bill_impact_figure.py
  python3 analysis/scripts/alt_household_bill_impact_figure.py --threshold 150
"""
import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from thesis_style import (CATEGORICAL_COLOURS, LABEL_BOX, MARKS,  # noqa: E402
                          NOTE_SIZE, REFERENCE_COLOUR, SUBTITLE_SIZE,
                          apply_style, new_figure, panel_suptitle,
                          save_figure)

METRIC_STEM = 'pct_of_coordination_savings_histogram_weighted'
BATTERY = '20kWh'

COHORT_COLOUR = CATEGORICAL_COLOURS[0]
RETAILER_COLOUR = CATEGORICAL_COLOURS[2]
TESLA_COLOUR = CATEGORICAL_COLOURS[7]

RETAILER_LABEL = r'Retailer pool retention ($\alpha$ 0.9)'
TESLA_LABEL = 'Tesla Electric'
TESLA_VARIANT = 'dynamic_tesla_electric_mixed_2026_modelled'


class Series(NamedTuple):
    """One histogram.

    sub: the subscription change the variant was aggregated at, $/month,
         naming the site-year file; None where the variant carries no fee
         change and the file has no _sub tag.
    """
    variant: str
    sub: float | None
    label: str
    colour: str


class Threshold(NamedTuple):
    """One cap threshold's figure. Panels are (title, series), and series
    are drawn in order, so the last is on top. stems are the output file
    names, keyed as VERSIONS is."""
    cap: str
    panels: tuple
    stems: dict


# The surcharge rate the cohort pool needs differs by settlement period at
# $150/MWh (3 c/kWh daily, 1 c/kWh monthly) but not at $300/MWh, so the
# rate sits on each series' own legend label.
THRESHOLDS = {
    '300': Threshold(
        cap='300',
        panels=(
            ('Daily settlement', (
                Series('cohort_x300_daily_s0.01', None,
                       'Cohort hedging (1c/kWh)', COHORT_COLOUR),
                Series('retailer_x300_daily_a0.9', -5.15851,
                       RETAILER_LABEL, RETAILER_COLOUR),
            )),
            ('Monthly settlement', (
                Series('cohort_x300_monthly_s0.01', None,
                       'Cohort hedging (1c/kWh)', COHORT_COLOUR),
                Series('retailer_x300_monthly_a0.9', -5.79053,
                       RETAILER_LABEL, RETAILER_COLOUR),
                Series(TESLA_VARIANT, None, TESLA_LABEL, TESLA_COLOUR),
            )),
        ),
        stems={'plain': 'alt_household_bill_impact',
               'tesla': 'alt_household_bill_impact_tesla'},
    ),
    '150': Threshold(
        cap='150',
        panels=(
            ('Daily settlement', (
                Series('cohort_x150_daily_s0.03', None,
                       'Cohort hedging (3c/kWh)', COHORT_COLOUR),
                Series('retailer_x150_daily_a0.9', -4.54262,
                       RETAILER_LABEL, RETAILER_COLOUR),
            )),
            ('Monthly settlement', (
                Series('cohort_x150_monthly_s0.01', None,
                       'Cohort hedging (1c/kWh)', COHORT_COLOUR),
                Series('retailer_x150_monthly_a0.9', -6.31042,
                       RETAILER_LABEL, RETAILER_COLOUR),
                Series(TESLA_VARIANT, None, TESLA_LABEL, TESLA_COLOUR),
            )),
        ),
        stems={'plain': 'alt_household_bill_impact_x150',
               'tesla': 'alt_household_bill_impact_x150_tesla'},
    ),
}

XLABEL = 'Annual bill change against Amber (% of coordination value)'
YLABEL = 'Share of site-years (%)'

# Axis range (percent, symmetric), bin width and x tick spacing, by version.
VERSIONS = {
    'plain': dict(tesla=False, limit=15, bin_w=1.0, tick=5),
    'tesla': dict(tesla=True, limit=50, bin_w=2.0, tick=10),
}
FIG_HEIGHT_IN = 3.0
YMAX_HEADROOM = 1.5     # clears the legend above the tallest bin


def site_year_path(root: Path, s: Series) -> Path:
    # cap_account_aggregate.py builds the tag with the sign flag,
    # so a fee that rose would read _sub+1.5 rather than _sub1.5.
    tag = '' if s.sub is None else f'_sub{s.sub:+g}'
    return root / BATTERY / s.variant / 'national' / \
        f'{METRIC_STEM}{tag}_site_years.csv'


def load_site_years(root: Path, s: Series) -> pd.DataFrame:
    """Site-year values and weights for one series."""
    path = site_year_path(root, s)
    if not path.exists():
        folder = path.parent
        present = sorted(p.name for p in folder.glob(
            f'{METRIC_STEM}*_site_years.csv')) if folder.is_dir() else []
        sub = ('' if s.sub is None else
               f' --subscription-change-per-month {s.sub:g}')
        raise FileNotFoundError(
            f"{path} not found (site-year files present: "
            f"{present or 'none'}). Write it with:\n"
            f"    python3 analysis/scripts/cap_account_aggregate.py "
            f"--battery-name {BATTERY} --variant {s.variant}{sub}")
    df = pd.read_csv(path)
    if df['value'].isna().any() or (df['weight'] <= 0).any():
        raise ValueError(f"{path}: missing values or non-positive weights")
    return df


def share_by_bin(df: pd.DataFrame, edges: np.ndarray) -> np.ndarray:
    """Percent of the total weight in each bin, with values beyond the
    range counted in the end bins."""
    v = np.clip(df['value'].to_numpy(), edges[0],
                np.nextafter(edges[-1], -np.inf))
    counts, _ = np.histogram(v, bins=edges, weights=df['weight'].to_numpy())
    return 100.0 * counts / df['weight'].sum()


def draw_figure(data: dict, threshold: Threshold, spec: dict, notes: bool):
    limit, bin_w = spec['limit'], spec['bin_w']
    edges = np.arange(-limit, limit + bin_w / 2, bin_w)
    panels = [(title, [s for s in series
                       if spec['tesla'] or s.variant != TESLA_VARIANT])
              for title, series in threshold.panels]
    shares = {s.variant: share_by_bin(data[s.variant], edges)
              for _, series in panels for s in series}
    ymax = YMAX_HEADROOM * max(s.max() for s in shares.values())

    fig = new_figure(FIG_HEIGHT_IN)
    axes = fig.subplots(1, 2, sharex=True, sharey=True)
    for ax, (title, series) in zip(axes, panels):
        for s in series:
            ax.stairs(shares[s.variant], edges, color=s.colour, lw=MARKS.lw,
                      label=s.label, zorder=3)
        ax.axvline(0, color='0.65', lw=0.5, zorder=2)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(0, ymax)
        ax.xaxis.set_major_locator(MultipleLocator(spec['tick']))
        ax.set_title(title, fontsize=SUBTITLE_SIZE)
        ax.set_xlabel(XLABEL)
        ax.legend(loc='upper right', handlelength=1.2, handletextpad=0.5,
                  borderpad=0.3, borderaxespad=0.3).set_zorder(20)
        if notes:
            n = len(next(iter(data.values())))
            ax.text(0.98, 0.03,
                    f'n = {n:,} site-years, {BATTERY[:-3]} kWh battery\n'
                    f'weighted by PV size band and state\n'
                    f'beyond ±{limit}% in the end bins',
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=NOTE_SIZE, color=REFERENCE_COLOUR,
                    bbox=LABEL_BOX, zorder=10)
    axes[1].tick_params(axis='y', labelleft=False)
    fig.supylabel(YLABEL, fontsize=plt.rcParams['axes.labelsize'])
    panel_suptitle(fig, r'Alternative offers annual bill change against '
                        r'Amber, \$' + threshold.cap +
                        r'/MWh wholesale cap, 20 kWh')
    return fig, shares, edges


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--threshold', nargs='*', choices=sorted(THRESHOLDS),
                    default=sorted(THRESHOLDS, reverse=True),
                    help="Cap thresholds, $/MWh. Default: all.")
    ap.add_argument('--aggregate-root', type=Path,
                    default=REPO_ROOT / 'analysis/output/'
                                        'cap_account_aggregate')
    ap.add_argument('--out-dir', type=Path,
                    default=REPO_ROOT / 'analysis/output/alt_offers')
    ap.add_argument('--notes', action='store_true',
                    help="draw the data notes (presentation version)")
    args = ap.parse_args(argv)

    apply_style()
    for cap in args.threshold:
        threshold = THRESHOLDS[cap]
        data = {s.variant: load_site_years(args.aggregate_root, s)
                for _, series in threshold.panels for s in series}
        for version, spec in VERSIONS.items():
            fig, shares, edges = draw_figure(data, threshold, spec,
                                             args.notes)
            name = threshold.stems[version]
            print(f"\n{name}  (axis ±{spec['limit']}%, "
                  f"bin {spec['bin_w']:g}%)")
            for v in shares:
                df = data[v]
                w = df['weight'] / df['weight'].sum()
                beyond = 100 * w[df['value'].abs() > spec['limit']].sum()
                print(f"  {v:36s} n={len(df)}  beyond range {beyond:5.1f}%  "
                      f"bins sum {shares[v].sum():.1f}%")
            save_figure(fig, args.out_dir / name)
    return 0


if __name__ == '__main__':
    sys.exit(main())
