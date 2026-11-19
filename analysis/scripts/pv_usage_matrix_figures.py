"""
pv_usage_matrix_figures.py

Two thesis figures drawn from the PV size band by annual usage matrix in
pv_usage_matrix.py.  Each is a 5 x 5 small-multiples grid with the usage bins
down the left (lowest at the top) and the PV array size bands along the
bottom, and a small chart in each cell.  The margins (the All row and column) are not
drawn.

  1. Average annual usage and PV generation (MWh/yr): two bars per cell, on one
     scale across every cell.
  2. Battery capacity for a $0 bill, and the share of site-years at $0: a box
     per cell for the interquartile range of the capacity at which each
     site-year's own bill reaches $0, with the median as a line, beside a bar
     for the share.  The capacity axis is on the left and the share axis on
     the right, each the same in every cell.

Cells with fewer than --min-sites (20) site-years show no data.  Usage-bin labels
carry two significant digits in MWh/yr, the top bin running up to the
--max-usage cutoff given to pv_usage_matrix.py.

The capacity box and median are over the site-years that reach $0 within the
modelled capacities, so they describe the share reaching $0 at any capacity
rather than the whole cell.  --share chooses what the bar shows: the share at
$0 with the reported battery (at-kwh, the default) or the share reaching $0 at
any modelled capacity (any), which is the population the box describes.

Reads the figure data CSV that pv_usage_matrix.py writes beside its tables, so
the figures can be restyled without re-running the model.  Figures follow
docs/thesis_figure_spec.md through thesis_style.  --notes draws each cell's
site count, for presentation versions; the thesis versions carry none.

Usage (from project root, after pv_usage_matrix.py):
    python3 analysis/scripts/pv_usage_matrix_figures.py
    python3 analysis/scripts/pv_usage_matrix_figures.py --share any
    python3 analysis/scripts/pv_usage_matrix_figures.py --only usage --notes
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import thesis_style as ts

DEFAULT_INPUT = Path('analysis/output/pv_usage_matrix/'
                     'pv_usage_matrix_achievable_amber_figure_data.csv')
DEFAULT_OUT = Path('analysis/output/pv_usage_matrix')

USAGE_COLOUR = ts.CATEGORICAL_COLOURS[0]
PV_COLOUR = ts.CATEGORICAL_COLOURS[1]
CAPACITY_COLOUR = ts.CATEGORICAL_COLOURS[0]
SHARE_COLOUR = ts.CATEGORICAL_COLOURS[2]
BOX_FILL_ALPHA = 0.30

BAR_WIDTH = 0.72                 # of the category spacing
X_LIM = (-0.5, 1.5)              # two categories at x = 0 and x = 1
LABEL_GAP = 0.02                 # of the axis range, between a bar and its label
LABEL_ROOM = 0.22                # of the axis range, above the tallest value

# Figure 2 scales.  One capacity axis and one share axis, the same in every
# cell, drawn as a single axes: the share bar is plotted at share / PCT_PER_KWH
# so that 100 % stands level with the largest modelled capacity.
CAPACITY_MAX_KWH = 50
PCT_PER_KWH = 100 / CAPACITY_MAX_KWH
CAPACITY_YLIM = CAPACITY_MAX_KWH * (1 + LABEL_ROOM + 0.02)


@dataclass
class Layout:
    """Margins and cell size, in inches."""
    left: float
    right: float
    top: float
    bottom: float
    cell_h: float
    wgap: float = 0.07
    hgap: float = 0.08


# ── Labels ────────────────────────────────────────────────────────────────────

def two_sig(v: float) -> str:
    """Two significant digits: 3.7, 5.0, 10, 14."""
    x = float(f'{v:.2g}')
    return f'{x:.1f}' if x < 10 else f'{x:.0f}'


def round_half_up(v: float) -> int:
    return int(np.floor(v + 0.5))


def range_label(lo: float, hi: float) -> str:
    a, b = round_half_up(lo), round_half_up(hi)
    return f'{a}' if a == b else f'{a}–{b}'


def usage_labels(bins: pd.DataFrame) -> list[str]:
    """One label per usage bin in MWh/yr: '<4.8' for the first, 'lo–hi' after,
    the last running to the cutoff."""
    out = []
    for i, r in enumerate(bins.itertuples()):
        if i == 0:
            out.append(f'<{two_sig(r.bin_hi_kwh / 1000)}')
        else:
            out.append(f'{two_sig(r.bin_lo_kwh / 1000)}–'
                       f'{two_sig(r.bin_hi_kwh / 1000)}')
    return out


# ── Grid ──────────────────────────────────────────────────────────────────────

def make_grid(layout: Layout, n_rows: int, n_cols: int):
    """Figure and axes array, laid out with explicit margins.  Panels share
    both axes, carry left and bottom spines only, and have no x ticks, since a
    cell's two marks are named by the legend."""
    w = ts.TEXT_WIDTH_IN
    h = (layout.top + layout.bottom + n_rows * layout.cell_h
         + (n_rows - 1) * layout.hgap)
    fig = ts.new_figure(h, layout=None)
    cell_w = (w - layout.left - layout.right
              - (n_cols - 1) * layout.wgap) / n_cols
    gs = fig.add_gridspec(
        n_rows, n_cols, left=layout.left / w, right=1 - layout.right / w,
        top=1 - layout.top / h, bottom=layout.bottom / h,
        wspace=layout.wgap / cell_w, hspace=layout.hgap / layout.cell_h)
    axes = gs.subplots(sharex=True, sharey=True, squeeze=False)
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_linewidth(ts.GRID_SPINE_LW)
        ax.tick_params(width=ts.GRID_SPINE_LW, length=ts.GRID_TICK_LEN)
        ax.grid(True, axis='y')
        ax.grid(False, axis='x')
        ax.set_xlim(*X_LIM)
        ax.set_xticks([])
    for ax in axes[:, 1:].flat:
        ax.tick_params(labelleft=False)
    return fig, axes, (w, h)


def add_headings(fig, axes, size, col_labels, row_labels, col_title,
                 row_title, tick_clear_in=0.30):
    """Column strips and title below the grid, row strips and title to its
    left.  tick_clear_in is the clearance left of the grid for the y tick
    labels."""
    w, h = size
    x0 = axes[0, 0].get_position().x0
    x1 = axes[0, -1].get_position().x1
    y_top = axes[0, 0].get_position().y1
    y_bot = axes[-1, 0].get_position().y0
    for j, lab in enumerate(col_labels):
        pos = axes[-1, j].get_position()
        fig.text((pos.x0 + pos.x1) / 2, y_bot - 0.05 / h, lab,
                 ha='center', va='top', fontsize=ts.SUBTITLE_SIZE)
    fig.text((x0 + x1) / 2, y_bot - 0.24 / h, col_title,
             ha='center', va='top', fontsize=ts.SUBTITLE_SIZE)
    strip_x = x0 - tick_clear_in / w
    for i, lab in enumerate(row_labels):
        pos = axes[i, 0].get_position()
        fig.text(strip_x, (pos.y0 + pos.y1) / 2, lab, rotation=90,
                 ha='right', va='center', fontsize=ts.SUBTITLE_SIZE)
    fig.text(strip_x - 0.19 / w, (y_top + y_bot) / 2, row_title,
             rotation=90, ha='right', va='center',
             fontsize=ts.SUBTITLE_SIZE)


def label_above(ax, x, y, text, ylim_top):
    ax.text(x, y + LABEL_GAP * ylim_top, text, ha='center', va='bottom',
            fontsize=ts.ANNOTATION_SIZE, color='black')


def site_note(ax, n):
    ax.text(0.03, 0.97, f'n = {n}', transform=ax.transAxes, ha='left',
            va='top', fontsize=ts.NOTE_SIZE, color=ts.REFERENCE_COLOUR,
            bbox=ts.LABEL_BOX)


def prepare(d: pd.DataFrame, min_sites: int):
    """Row and column order, headings, and the cell lookup."""
    bins = (d.drop_duplicates('usage_bin')
            [['usage_bin', 'bin_lo_kwh', 'bin_hi_kwh']].reset_index(drop=True))
    bands = d.drop_duplicates('pv_band')[['pv_band', 'pv_band_label']]
    cell = d.set_index(['usage_bin', 'pv_band'])
    return bins, list(bands['pv_band']), list(bands['pv_band_label']), cell


# ── Figure 1: usage and PV generation ─────────────────────────────────────────

def figure_usage_pv(d: pd.DataFrame, min_sites: int, notes: bool):
    bins, band_ids, band_labels, cell = prepare(d, min_sites)
    shown = d[d['n_site_years'] >= min_sites]
    if shown.empty:
        print('  No cell reaches the site-year threshold; usage figure not '
              'drawn.')
        return None
    vmax = float(shown[['avg_usage_kwh', 'avg_pv_kwh']].max().max()) / 1000
    ytop = vmax * (1 + LABEL_ROOM)

    layout = Layout(left=0.85, right=0.10, top=0.38, bottom=0.80, cell_h=0.74)
    fig, axes, size = make_grid(layout, len(bins), len(band_ids))
    axes[0, 0].set_ylim(0, ytop)
    axes[0, 0].yaxis.set_major_locator(
        mticker.MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))

    for i, b in enumerate(bins['usage_bin']):
        for j, band in enumerate(band_ids):
            ax = axes[i, j]
            r = cell.loc[(b, band)]
            if r['n_site_years'] < min_sites:
                continue
            vals = [r['avg_usage_kwh'] / 1000, r['avg_pv_kwh'] / 1000]
            ax.bar([0, 1], vals, width=BAR_WIDTH, linewidth=0,
                   color=[USAGE_COLOUR, PV_COLOUR])
            for x, v in zip((0, 1), vals):
                label_above(ax, x, v, two_sig(v), ytop)
            if notes:
                site_note(ax, int(r['n_site_years']))

    add_headings(fig, axes, size, band_labels, usage_labels(bins),
                 'PV array size (kW)', 'Annual usage (MWh/yr)')
    handles = [Patch(facecolor=USAGE_COLOUR,
                     label='Average annual usage (MWh/yr)'),
               Patch(facecolor=PV_COLOUR,
                     label='Average annual PV generation (MWh/yr)')]
    centre = (axes[0, 0].get_position().x0
              + axes[0, -1].get_position().x1) / 2
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False,
               bbox_to_anchor=(centre, 0.03 / size[1]))
    ts.panel_suptitle(fig, 'Average annual usage and PV generation',
                      y=1 - 0.04 / size[1])
    return fig


# ── Figure 2: capacity for a $0 bill and share at $0 ──────────────────────────

def figure_zero_bill(d: pd.DataFrame, min_sites: int, notes: bool,
                     share: str):
    bins, band_ids, band_labels, cell = prepare(d, min_sites)
    shown = d[(d['n_site_years'] >= min_sites)]
    if shown.empty:
        print('  No cell reaches the site-year threshold; capacity figure '
              'not drawn.')
        return None
    kwh = int(d['kwh'].iloc[0])
    share_col = 'share_zero_at_kwh' if share == 'at-kwh' else 'share_zero_any'
    if shown[['own_zero_q25', share_col]].isna().all().all():
        print('  The figure data has no modelled results (capacity and share '
              'are blank); capacity figure not drawn.')
        return None

    layout = Layout(left=0.85, right=0.45, top=0.38, bottom=0.80,
                    cell_h=0.74)
    fig, axes, size = make_grid(layout, len(bins), len(band_ids))
    axes[0, 0].set_ylim(0, CAPACITY_YLIM)
    axes[0, 0].set_yticks([0, 25, 50])
    for ax in axes[:, -1]:
        sec = ax.secondary_yaxis(
            'right', functions=(lambda y: y * PCT_PER_KWH,
                                lambda y: y / PCT_PER_KWH))
        sec.set_yticks([0, 50, 100])
        sec.tick_params(width=ts.GRID_SPINE_LW, length=ts.GRID_TICK_LEN)
        sec.spines['right'].set_linewidth(ts.GRID_SPINE_LW)

    half = BAR_WIDTH / 2
    for i, b in enumerate(bins['usage_bin']):
        for j, band in enumerate(band_ids):
            ax = axes[i, j]
            r = cell.loc[(b, band)]
            if r['n_site_years'] < min_sites:
                continue
            q25, q50, q75 = r['own_zero_q25'], r['own_zero_q50'], r['own_zero_q75']
            if np.isfinite(q25):
                ax.add_patch(Rectangle(
                    (0 - half, q25), BAR_WIDTH, q75 - q25,
                    facecolor=mcolors.to_rgba(CAPACITY_COLOUR, BOX_FILL_ALPHA),
                    edgecolor=CAPACITY_COLOUR, linewidth=0.6))
                ax.plot([0 - half, 0 + half], [q50, q50], color='black',
                        linewidth=0.9, solid_capstyle='butt')
                label_above(ax, 0, q75, range_label(q25, q75), CAPACITY_YLIM)
            pct = r[share_col]
            if np.isfinite(pct):
                height = pct / PCT_PER_KWH
                ax.bar(1, height, width=BAR_WIDTH, linewidth=0,
                       color=SHARE_COLOUR)
                label_above(ax, 1, height, f'{pct:.0f}%', CAPACITY_YLIM)
            if notes:
                site_note(ax, int(r['n_site_years']))

    add_headings(fig, axes, size, band_labels, usage_labels(bins),
                 'PV array size (kW)', 'Annual usage (MWh/yr)')
    x1 = axes[0, -1].get_position().x1

    if share == 'at-kwh':
        share_text = f'Share of site-years at \\$0 with {kwh} kWh (%)'
        title = (f'Battery capacity for a \\$0 bill, and share of site-years '
                 f'at \\$0 with {kwh} kWh')
    else:
        share_text = 'Share of site-years reaching \\$0 (%)'
        title = ('Battery capacity for a \\$0 bill, and share of site-years '
                 'reaching \\$0')
    handles = [
        Patch(facecolor=mcolors.to_rgba(CAPACITY_COLOUR, BOX_FILL_ALPHA),
              edgecolor=CAPACITY_COLOUR, linewidth=0.6,
              label='Battery size for \\$0 bill, interquartile range (kWh)'),
        Patch(facecolor=SHARE_COLOUR, label=share_text)]
    centre = (axes[0, 0].get_position().x0 + x1) / 2
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False,
               columnspacing=1.4, bbox_to_anchor=(centre, 0.03 / size[1]))
    ts.panel_suptitle(fig, title, y=1 - 0.04 / size[1])
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description='Thesis figures from the PV size band by usage matrix.')
    p.add_argument('--input', type=Path, default=DEFAULT_INPUT,
                   help='The figure data CSV from pv_usage_matrix.py '
                        f'(default: {DEFAULT_INPUT}).')
    p.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    p.add_argument('--min-sites', type=int, default=20,
                   help='Cells with fewer site-years show no data '
                        '(default: 20).')
    p.add_argument('--share', choices=['at-kwh', 'any'], default='at-kwh',
                   help="Share bar in the capacity figure: site-years at $0 "
                        "with the reported battery (default), or reaching $0 "
                        "at any modelled capacity.")
    p.add_argument('--only', choices=['usage', 'capacity'], default=None,
                   help='Draw one figure only.')
    p.add_argument('--notes', action='store_true',
                   help="Draw each cell's site count (presentation versions).")
    args = p.parse_args(argv)

    if not args.input.exists():
        print(f"ERROR: no {args.input}.  Run pv_usage_matrix.py first; it "
              f"writes this file beside its tables.", file=sys.stderr)
        sys.exit(1)
    d = pd.read_csv(args.input)
    ts.apply_style()

    if args.only in (None, 'usage'):
        fig = figure_usage_pv(d, args.min_sites, args.notes)
        if fig is not None:
            ts.save_figure(fig, args.output_dir / 'pv_usage_matrix_usage_pv')
    if args.only in (None, 'capacity'):
        fig = figure_zero_bill(d, args.min_sites, args.notes, args.share)
        if fig is not None:
            suffix = '' if args.share == 'at-kwh' else '_any'
            ts.save_figure(fig, args.output_dir
                           / f'pv_usage_matrix_zero_bill{suffix}')


if __name__ == '__main__':
    main()
