"""Chart the measurement ladder from run_cohort_rungs.py.

Two figures, in order of importance:

  A. Per site, small multiples. The primary result. Every column is computed on
     the same site, days, battery and c_deg, so capacity, headroom per kWh and
     spike-day luck set the LEVEL of each panel and do not reorder it within a
     panel. That is what makes the spacing readable even though the level is
     specific to that site and window.
  B. Pooled, same format. A summary, not a substitute: if the per-site panels
     disagree on ordering or spacing, the pooled bar is hiding something.

Design decisions worth knowing before reading either figure.

Levels, not deltas
------------------
Bars are VCR levels side by side. No decomposition into shares summing to a
total, because the gap to `actual` cannot be attributed: Amber has a better
price forecast than pre-dispatch, a worse site forecast than our perfect solar
and load, and its own risk appetite, and those push in opposite directions.

The break before `actual`
-------------------------
Columns 1 to 5 form a controlled sequence in which each step changes exactly
one thing. `actual` changes several at once and is an observation rather than a
step in that sequence. Putting it in an unbroken row of bars would invite
exactly the causal reading the analysis cannot support, so it is separated by a
gap and a rule and drawn in a different colour.

Truncation at 100%
------------------
`perfect_18h` can land microscopically above 1.0 (observed up to 1.002).
Rolling over an 18-hour window cannot genuinely beat full-period perfect
foresight; it has strictly less information. The cause is LP tolerance: perfect
foresight over ~78,000 intervals is a ~470,000-variable solve, while rolling
does ~78,000 small ones each to tight tolerance, and a sequence of small exact
solves edges out one large approximate one. Bars are truncated at 1.0 for
display and the CSV keeps the true value.

Usage (from project root):
    python3 -m energy_masters_pipeline.scripts.chart_cohort_rungs
    python3 -m energy_masters_pipeline.scripts.chart_cohort_rungs \\
        --csv data/processed/energy_masters/cohort_rungs_site.csv \\
        --out-dir data/processed/energy_masters/charts
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from energy_masters_pipeline.scripts.run_cohort_rungs import (
    SITE_HEADROOM_FLOOR_DOLLARS, cohort_vcr,
)
# Imported rather than copied so the site to colour mapping cannot drift
# between this figure and the VCR results figures.
from energy_masters_pipeline.scripts.chart_vcr_results import _site_colours
from analysis.scripts import thesis_style as ts

# Ladder order, left to right. Each entry: (key, short label, long label).
# One monotone sequence: each rung removes exactly one thing the controller
# would not really know, ending at the column built entirely from information
# available before the decision instant. Then two breaks -- the first to
# predispatch_before, which answers a different question (when does the price
# become known), and the second to observed Amber dispatch, which changes
# several things at once.
#
# shaped_18h_lf and shaped_18h_lf_sf are computed and sit in the CSV, feeding
# the attribution figure, but are not drawn: they duplicate the load and solar
# steps at a price level the ladder already covers.
LADDER = (
    ('perfect_foresight', 'perfect\nforesight', 'single-shot LP over the whole period'),
    ('shaped_18h', 'actual\nprices', 'rolling 18 h, actual prices in pre-dispatch shape'),
    ('predispatch_full', 'predispatch &\ninterval price', 'pre-dispatch, price known and forward window refreshed (~39 s)'),
    ('predispatch_full_lf', '+ load\nforecast', 'pre-dispatch, price known and window refreshed, load forecast from site history'),
    ('predispatch_full_lf_sf', '+ solar\nforecast', 'pre-dispatch, price known and window refreshed, load and solar forecast from site history'),
    ('predispatch_before', 'predispatch\nonly', 'pre-dispatch, forecast only, at interval start (0 s)'),
)
ACTUAL = ('actual', 'actual\n(Amber)', 'actual (Amber)')

# The end of the controlled sequence: everything the controller could know
# before the decision instant, and nothing it could not. Figure F compares
# this one rung against observed Amber dispatch, so it is named once here and
# read from the ladder everywhere else.
REFINED = 'predispatch_full_lf_sf'

LADDER_LOOKUP = tuple((k, lbl) for k, lbl, _ in LADDER)

# The name registry. Every rung the run produces has an entry here, including
# ones no longer drawn in the ladder -- dropping a rung from a figure should
# not silently remove its name from every other figure. Used by anything that
# shows a rung outside the ladder row: scatter panels, legends, tables.
RUNG_LABELS = {
    'perfect_foresight': 'perfect foresight',
    'perfect_18h': 'rolling,\nactual prices',
    'shaped_18h': 'actual prices',
    'shaped_18h_lf': 'actual prices,\nload forecast',
    'predispatch_full': 'price known\n+ new forecast',
    'predispatch_full_lf': 'price known + new forecast,\nload forecast',
    'predispatch_price': 'price known',
    'predispatch_before': 'forecast only',
    'actual': 'actual (Amber)',
    'self_consumption': 'self-consumption',
    'shaped_18h_lf_sf': 'actual prices,\nload and solar forecast',
    'predispatch_full_lf_sf': 'price known + new forecast,\nload and solar forecast',
}


def rung_label(key: str, oneline: bool = False) -> str:
    """Standalone name for a rung, independent of whether it is on the ladder."""
    lbl = RUNG_LABELS.get(key) or dict(LADDER_LOOKUP).get(key) or key
    return lbl.replace('\n', ' ') if oneline else lbl

# Sequential blue: darkest where information is best, lightening as it degrades.
# The load rung continues the sequence, because it IS worse information, but
# carries a hatch so the change of dimension is visible without a legend.
# The three pre-dispatch rungs share a colour: they differ in what the SITE
# forecast knows, not in the price information, and the hatch density carries
# that -- none, load, load and solar.
LADDER_COLOURS = ('#08306b', '#2171b5', '#4292c6', '#4292c6', '#4292c6',
                  '#c6dbef')
LADDER_HATCH = (None, None, None, '///', 'xxx', None)

# Looked up rather than restated, so figure F cannot end up drawing the
# refined rung in a colour the ladder no longer uses for it.
_LADDER_KEYS = tuple(k for k, _, _ in LADDER)
REFINED_COLOUR = dict(zip(_LADDER_KEYS, LADDER_COLOURS))[REFINED]
REFINED_HATCH = dict(zip(_LADDER_KEYS, LADDER_HATCH))[REFINED]

# The load rung is a VARIATION on the rung before it, not the next step of the
# price sequence. Two earlier attempts to signal that failed: trailing it at
# the end of the row read as "forecast only, plus load forecast", and a hatch
# plus a gap did not fix it, because position is read before texture.
#
# So it is PAIRED instead -- butted against its parent with no gap, sharing
# its colour, separated by a gap from whatever follows. A reader sees one bar
# and its variant, which is what it is. The price sequence resumes after the
# gap.
PAIR_GAP = 0.06       # retained for the position helper; no pairs are drawn
LADDER_PAIRED = ()

# A section gap is inserted AFTER each of these keys. The actual bar always
# follows one.
SECTION_AFTER = ('predispatch_full_lf_sf', 'predispatch_before')

# One gap size for every section boundary, including the one before the actual
# bar. The break used to be wider than the others to mark it as a different
# kind of boundary, but the dotted dividers already do that, and the extra
# width bought nothing except crowded tick labels in the per-site panels.
SECTION_GAP = 0.375

# The 2x2 behind the load step, for the attribution figure. Keys are
# (perfect price + perfect load, perfect price + forecast load,
#  forecast price + perfect load, forecast price + forecast load).
ATTRIB_2X2 = ('shaped_18h', 'shaped_18h_lf',
              'predispatch_full', 'predispatch_full_lf')
ACTUAL_COLOUR = '#2bffa4'      # Amber's brand green, as used in the other charts

DEFAULT_CSV = Path('data/processed/energy_masters/cohort_rungs_site.csv')
DEFAULT_DAILY_CSV = Path('data/processed/energy_masters/cohort_rungs_daily.csv')

# Figure C: which rung does Amber behave like, day by day.
#
# Rungs compared, left to right, shaped_18h first because it is the anchor
# rather than a candidate: it is what a controller with perfect prices does.
# shaped_18h is not among them. Its daily VCR is 1.007 plus or minus 0.019,
# so it has no spread to correlate against and its panel said only that a
# controller with perfect prices beats Amber on every day that carries value,
# which figures A and B already say. Dropping it gives the three real
# candidates the horizontal room instead.
SCATTER_RUNGS = ('predispatch_full', 'predispatch_price', 'predispatch_before')

# Minimum daily headroom for a site-day to enter the scatter.
#
# This floor is not tidying, it is what makes the statistic exist. Daily VCR
# divides by the day's headroom, and across the Energy Masters cohort the
# MEDIAN daily headroom is $0.76, 198 site-days have NEGATIVE headroom (the
# daily split does not carry the global terminal SOC pin, so perfect foresight
# can cost more than self-consumption on a single day), and one daily VCR is
# infinite. Unfiltered, daily VCR ranges from -369 to +115 and the scatter
# measures nothing but division by small numbers.
#
# The floor is deliberately exposed as an argument and its effect is charted
# separately (figure D), because between $5 and $25 the sample falls from 69
# site-days to 18 and one rung's correlation moves from 0.51 to 0.09. A single
# number at a single floor would be a choice presented as a fact.
# Charted as one row per floor rather than a single choice, because the floor
# is a real decision with a real effect and the reader should see it made.
DEFAULT_HEADROOM_FLOORS = (5.0, 10.0)

# Axis window for the scatter. Points outside are drawn on the boundary as
# open triangles rather than dropped, so a clipped outlier is visible as a
# clipped outlier.
VCR_CLIP = (-0.6, 1.3)

# Figure F draws one daily panel rather than one per floor, so it makes the
# floor choice explicitly instead of inheriting the row: the higher of the two
# defaults, where the denominator is least likely to be doing the work. The
# cost of that choice is charted in figure D, which is where it belongs.
REFINED_FLOOR = 10.0

# A tighter window than VCR_CLIP, because a single panel does not have to
# accommodate the low floors that produce the extreme daily ratios. More
# points land on the boundary as a result, and they are drawn there.
REFINED_VCR_CLIP = (-0.25, 1.0)


def _finish(fig, name: str, out_dir: Optional[Path]):
    if out_dir is None:
        plt.show()
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'{name}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {path}")


def _bar_positions() -> tuple[np.ndarray, float]:
    """x positions for the ladder bars plus the separated actual bar.

    Every section boundary gets the same SECTION_GAP; the dotted dividers,
    not the width, are what tell the reader which kind of boundary it is.
    Within a pair the variant is butted against its parent by PAIR_GAP.
    """
    xs, x = [], 0.0
    for k, _, _ in LADDER:
        if k in LADDER_PAIRED and xs:
            x = xs[-1] + 1.0 + PAIR_GAP        # butt against the parent
        xs.append(x)
        x += 1.0
        if k in SECTION_AFTER:
            x += SECTION_GAP
    x_actual = x if LADDER[-1][0] in SECTION_AFTER else x + SECTION_GAP
    return np.asarray(xs), x_actual


def _section_dividers(xs: np.ndarray, x_actual: float) -> list[float]:
    """Midpoints of every section boundary, ladder breaks and the actual break.

    Derived from the bar positions rather than recomputed from the gap
    constants, so a divider cannot drift away from the space it marks.
    """
    stops = list(xs) + [x_actual]
    return [(stops[i] + stops[i + 1]) / 2
            for i in range(len(stops) - 1)
            if stops[i + 1] - stops[i] > 1.0 + PAIR_GAP + 1e-9]


def _draw_panel(ax, values: dict, title: str, subtitle: str = '',
                show_labels: bool = True, label_fontsize: float = 6.5):
    """One ladder panel. `values` maps dispatch key to VCR."""
    xs, x_actual = _bar_positions()
    heights = [min(values.get(k, np.nan), 1.0) for k, _, _ in LADDER]
    bars = ax.bar(xs, heights, width=0.72, color=LADDER_COLOURS, zorder=3)
    for bar, h in zip(bars, LADDER_HATCH):
        if h:
            # White, not a darker blue: a variant bar shares its parent's
            # fill, so a blue hatch on the darker pair is drawn in the fill
            # colour and vanishes. White reads on every tone in the ramp.
            bar.set_hatch(h)
            bar.set_edgecolor('white')
            bar.set_linewidth(0)

    # Values above 1.0 are truncated for display; mark them so the reader can
    # see it happened rather than wondering why a bar is exactly flat.
    for x, (k, _, _) in zip(xs, LADDER):
        v = values.get(k, np.nan)
        if np.isfinite(v) and v > 1.0005:
            ax.plot([x], [1.0], marker='^', ms=5, color='white',
                    markeredgecolor='#08306b', zorder=5)

    a = values.get(ACTUAL[0], np.nan)
    ax.bar([x_actual], [min(a, 1.0)], width=0.72, color=ACTUAL_COLOUR, zorder=3)

    # Section dividers. The last one is the break between the controlled
    # sequence and observed Amber dispatch, drawn heavier; the rest separate
    # a pair from the sequence resuming.
    divs = _section_dividers(xs, x_actual)
    for i, xd in enumerate(divs):
        last = (i == len(divs) - 1)
        ax.axvline(xd, ls=(0, (2, 3)) if last else (0, (1, 4)),
                   lw=1.1 if last else 0.9,
                   color='0.45' if last else '0.7', zorder=2)

    for x, k in list(zip(xs, [k for k, _, _ in LADDER])) + [(x_actual, ACTUAL[0])]:
        v = values.get(k, np.nan)
        if np.isfinite(v):
            ax.text(x, min(v, 1.0) + 0.02, f'{v:.2f}', ha='center', va='bottom',
                    fontsize=label_fontsize, zorder=4)

    ax.set_ylim(0, 1.16)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks(list(xs) + [x_actual])
    if show_labels:
        ax.set_xticklabels([lbl for _, lbl, _ in LADDER] + [ACTUAL[1]],
                           fontsize=label_fontsize)
    else:
        ax.set_xticklabels([])
    ax.set_xlim(-0.7, x_actual + 0.7)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.set_title(title + (f"\n{subtitle}" if subtitle else ''),
                 fontsize=8.5, pad=4, linespacing=1.35)


def chart_by_site(site_df: pd.DataFrame, out_dir: Optional[Path] = None):
    """Small multiples, one panel per site. The primary figure."""
    piv = site_df.pivot_table(index='site_id', columns='dispatch', values='vcr')
    meta = (site_df[site_df['dispatch'] == 'actual']
            .set_index('site_id')[['battery_kwh', 'headroom',
                                   'period_start', 'period_end']])
    order = meta.sort_values('headroom', ascending=False).index

    n = len(order)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.5 * nrows),
                             squeeze=False)
    for i, site_id in enumerate(order):
        ax = axes[i // ncols][i % ncols]
        m = meta.loc[site_id]
        _draw_panel(ax, piv.loc[site_id].to_dict(),
                    title=f"{site_id}   {m['battery_kwh']:.1f} kWh",
                    subtitle=f"${m['headroom']:,.0f} headroom  |  "
                             f"{m['period_start']} to {m['period_end']}",
                    show_labels=(i // ncols == nrows - 1))
        if i % ncols == 0:
            ax.set_ylabel('VCR', fontsize=9)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')
    # Restore x labels on the last populated panel of each column.
    for col in range(ncols):
        last = max((i for i in range(n) if i % ncols == col), default=None)
        if last is not None:
            ax = axes[last // ncols][col]
            ax.set_xticklabels([lbl for _, lbl, _ in LADDER] + [ACTUAL[1]],
                               fontsize=6.5)

    fig.suptitle(
        'Value captured against perfect foresight, by information available\n'
        'Each column left of the dashed line changes exactly one thing from '
        'the one before it; "actual" changes several at once',
        fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    _finish(fig, 'cohort_rungs_by_site', out_dir)


def chart_pooled(site_df: pd.DataFrame, out_dir: Optional[Path] = None):
    """Pooled ladder, same format. A summary, not a substitute."""
    coh = cohort_vcr(site_df).set_index('dispatch')['cohort_vcr']
    n_sites = int(cohort_vcr(site_df)['n_sites'].max())

    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    _draw_panel(ax, coh.to_dict(),
                title='', show_labels=True, label_fontsize=9)
    ax.set_ylabel('Coordination Value Capture Ratio\n(0 = self-consumption, 1 = perfect foresight)',
                  fontsize=9)
    ax.set_title(
        f'Value captured against perfect foresight, pooled over {n_sites} '
        'Energy Masters sites\n'
        'Summed numerator over summed denominator; each column computed on the '
        'same sites, days and batteries',
        fontsize=10)

    # No step-by-step notes here: the explainer slide that precedes these
    # figures carries them, and repeating them crowds the chart.
    fig.tight_layout()
    _finish(fig, 'cohort_rungs_pooled', out_dir)



# ---------------------------------------------------------------------------
# Thesis figures: pooled ladders in the thesis format
# (docs/thesis_figure_spec.md). Each entry is one figure, written under the
# file name it has in thesis/figures/.
# ---------------------------------------------------------------------------

THESIS_CHART_DIR = Path('energy_masters_pipeline/charts')
THESIS_HEIGHT_IN = 3.0

# Bars: (dispatch key, tick label, colour, hatch). Sequential blue from best to
# worst information; a hatched bar shares its parent's colour. A key has the
# same label in every thesis figure; the per-site grid keeps the shorter
# LADDER labels. A section
# break follows each key in breaks_after; the actual bar always follows one.
THESIS_LADDERS = {
    'em_vcr_price_information': dict(
        title='Value capture ratio by price information',
        bars=(
            ('perfect_foresight', 'perfect\nforesight', '#08306b', None),
            ('shaped_18h', 'actual\nprices', '#2171b5', None),
            ('predispatch_full', 'dispatch price &\nnew predispatch', '#4292c6', None),
            ('predispatch_price', 'dispatch price &\npredispatch', '#6baed6', None),
            ('predispatch_before', 'predispatch\nonly', '#c6dbef', None),
        ),
        breaks_after=(),
    ),
    'em_vcr_site_forecasts': dict(
        title='Value capture ratio with load and solar forecasts',
        bars=(
            ('perfect_foresight', 'perfect\nforesight', '#08306b', None),
            ('shaped_18h', 'actual\nprices', '#2171b5', None),
            ('predispatch_full', 'dispatch price &\nnew predispatch', '#4292c6', None),
            ('predispatch_full_lf', '+ load\nforecast', '#4292c6', '///'),
            ('predispatch_full_lf_sf', '+ solar\nforecast', '#4292c6', 'xxx'),
            ('predispatch_before', 'predispatch\nonly', '#c6dbef', None),
        ),
        breaks_after=('predispatch_full_lf_sf',),
    ),
}


def _thesis_ladder(ax, values: dict, bars, breaks_after) -> None:
    """One pooled ladder: bars, value labels, section separators."""
    xs, x = [], 0.0
    for key, _, _, _ in bars:
        xs.append(x)
        x += 1.0 + (SECTION_GAP if key in breaks_after else 0.0)
    x_actual = x + SECTION_GAP
    keys = [b[0] for b in bars] + [ACTUAL[0]]
    pos = xs + [x_actual]
    colours = [b[2] for b in bars] + [ACTUAL_COLOUR]
    hatches = [b[3] for b in bars] + [None]

    for xb, key, colour, hatch in zip(pos, keys, colours, hatches):
        v = values.get(key, np.nan)
        if not np.isfinite(v):
            raise KeyError(f"no pooled VCR for {key!r}")
        bar = ax.bar([xb], [min(v, 1.0)], width=0.72, color=colour,
                     zorder=3)[0]
        if hatch:
            bar.set_hatch(hatch)
            bar.set_edgecolor('white')
            bar.set_linewidth(0)
        if v > 1.0005:
            ax.plot([xb], [1.0], marker='^', color='white',
                    markeredgecolor='#08306b', zorder=5)
        ax.text(xb, min(v, 1.0) + 0.02, f'{v:.2f}', ha='center',
                va='bottom', fontsize=ts.ANNOTATION_SIZE, zorder=4)

    # Separators: between sections of the modelled sequence, and a darker one
    # before observed Amber dispatch.
    stops = pos
    for i in range(len(stops) - 1):
        if stops[i + 1] - stops[i] > 1.0 + 1e-9:
            last = (i == len(stops) - 2)
            ax.axvline((stops[i] + stops[i + 1]) / 2, ls=':',
                       lw=ts.SEPARATOR_LW,
                       color=ts.REFERENCE_COLOUR if last else ts.SEPARATOR_COLOUR,
                       zorder=2)

    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks(pos)
    ax.set_xticklabels([b[1] for b in bars] + [ACTUAL[1]])
    ax.tick_params(axis='x', length=0)
    ax.set_xlim(-0.6, x_actual + 0.6)
    ax.grid(axis='x', visible=False)


def chart_thesis_ladders(site_df: pd.DataFrame,
                         out_dir: Path = THESIS_CHART_DIR) -> None:
    """Write each THESIS_LADDERS figure from the pooled VCRs."""
    coh = cohort_vcr(site_df).set_index('dispatch')['cohort_vcr'].to_dict()
    ts.apply_style()
    for name, spec in THESIS_LADDERS.items():
        fig = ts.new_figure(THESIS_HEIGHT_IN)
        ax = fig.subplots()
        _thesis_ladder(ax, coh, spec['bars'], spec['breaks_after'])
        ax.set_ylabel('Value capture ratio')
        ax.set_title(spec['title'])
        ts.save_figure(fig, Path(out_dir) / f'{name}.png')


# ---------------------------------------------------------------------------
# Figure C and D: which rung does Amber behave like
# ---------------------------------------------------------------------------

def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation, via ranks and Pearson. No scipy dependency.

    Reported alongside Pearson because the two disagreeing IS the finding
    whenever it happens: on unnormalised dollars they disagree completely,
    because a handful of extreme days carry the Pearson value.
    """
    if len(x) < 3:
        return float('nan')
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float('nan')
    return float(np.corrcoef(rx, ry)[0, 1])


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def load_daily(csv_path: Path, floor: float,
               rungs=SCATTER_RUNGS) -> pd.DataFrame:
    """Daily rows with a usable denominator, plus a VCR per dispatch.

    `rungs` is which dispatches need a vcr_ column; it defaults to the scatter
    row so existing callers are unchanged. Figure F passes its own rung, which
    is not in SCATTER_RUNGS.
    """
    d = pd.read_csv(csv_path, parse_dates=['date'])
    keep = d[d['headroom'] >= floor].copy()
    for c in tuple(rungs) + ('actual',):
        col = f'vcr_{c}'
        if col not in keep.columns:
            if c not in keep.columns:
                raise KeyError(
                    f"the daily table has neither '{col}' nor '{c}'; re-run "
                    f"run_cohort_rungs with the '{c}' rung enabled")
            keep[col] = (keep['sc'] - keep[c]) / keep['headroom']
    return keep


def _scatter_panel(ax, day: pd.DataFrame, rung: str, colours: dict,
                   show_ylabel: bool, clip=VCR_CLIP):
    lo, hi = clip
    x = day[f'vcr_{rung}'].to_numpy()
    y = day['vcr_actual'].to_numpy()
    h = day['headroom'].to_numpy()

    # The least squares fit, NOT the identity line.
    #
    # An identity line frames the panel as an equality test, which is not the
    # question. Correlation is invariant to slope and intercept: a rung that
    # was reliably half of actual every single day would correlate perfectly
    # while no point sat on the identity line at all. What the correlation
    # measures is tightness about the fitted line, so that is the line drawn.
    # Level comparisons belong in figures A and B.
    if len(x) >= 3 and np.std(x) > 0:
        b, a0 = np.polyfit(x, y, 1)
        xs = np.array([lo, hi])
        ax.plot(xs, a0 + b * xs, ls=(0, (4, 3)), lw=1.2, color='0.45',
                zorder=1, label='least squares fit')
    ax.axhline(0, color='0.8', lw=0.7, zorder=0)
    ax.axvline(0, color='0.8', lw=0.7, zorder=0)

    # Area with headroom, so the eye weights a $200 day above a $10 one
    # without the statistic doing so.
    size = 18 + 210 * np.sqrt(np.clip(h, 0, None) / max(h.max(), 1e-9))
    inside = (x >= lo) & (x <= hi) & (y >= lo) & (y <= hi)
    for site_id, g in day.groupby('site_id'):
        m = (day['site_id'] == site_id).to_numpy() & inside
        if m.any():
            ax.scatter(x[m], y[m], s=size[m], color=colours[site_id],
                       alpha=0.75, edgecolor='white', linewidth=0.6,
                       zorder=3, label=site_id)
    out = ~inside
    if out.any():
        ax.scatter(np.clip(x[out], lo, hi), np.clip(y[out], lo, hi),
                   s=32, marker='^', facecolor='none', edgecolor='0.35',
                   linewidth=1.0, zorder=4,
                   label=f'clipped ({int(out.sum())})')

    # Co-movement only. Bias and mean absolute deviation from the identity
    # line were reported here and removed: they answer whether the two are the
    # same, which is a different question and one figures A and B already
    # cover.
    p, sp = _pearson(x, y), _spearman(x, y)
    ax.text(0.035, 0.965, f"r {p:.2f}\n\u03c1 {sp:.2f}",
            transform=ax.transAxes, va='top', ha='left', fontsize=8.5,
            bbox=dict(fc='white', ec='0.7', alpha=0.95, pad=3), zorder=6)
    # One line: the scatter panels are wide and the registry's line breaks are
    # there for cramped contexts, not this one.
    ax.set_title(rung_label(rung, oneline=True), fontsize=9, pad=4)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.25, zorder=0)
    ax.set_xlabel('rung daily VCR', fontsize=8.5)
    if show_ylabel:
        ax.set_ylabel('actual (Amber) daily VCR', fontsize=8.5)


def chart_actual_resemblance(daily_csv: Path,
                             floors=DEFAULT_HEADROOM_FLOORS,
                             out_dir: Optional[Path] = None) -> pd.DataFrame:
    """Daily VCR of actual against each rung, one row per headroom floor.

    Why daily VCR rather than dollars. Value captured and gap to perfect
    foresight sum to headroom, which is common to every dispatch on a given
    site-day, so correlating either in dollars mostly correlates a shared
    headroom term: every rung agrees that big days are big. On this cohort
    that produced a spurious result driven almost entirely by 26 January.
    Normalising by headroom removes the common scale and asks whether the rung
    and actual agree about how WELL a day went.

    The cost is that the denominator has to be meaningful, which is what the
    floor is for, and why three of them are drawn rather than one.

    This figure asks ONE question: do the rung and actual rise and fall
    together across days. It does not ask whether they reach the same level,
    which is why there is no identity line and no bias statistic; correlation
    is invariant to slope and intercept, so an equality reference would
    mislead. Level comparisons are figures A and B.

    What this figure does NOT do is reconcile to the pooled VCR in figure B.
    That figure is summed numerator over summed denominator, which is a
    headroom-weighted mean of daily VCRs; this scatter is unweighted by
    construction, because weighting would restore exactly the dominance the
    normalisation was there to remove.
    """
    floors = tuple(float(f) for f in floors)
    colours = _site_colours(sorted(pd.read_csv(daily_csv)['site_id'].unique()))
    nr, nc = len(floors), len(SCATTER_RUNGS)
    fig, axes = plt.subplots(nr, nc, figsize=(4.6 * nc, 4.5 * nr),
                             squeeze=False)
    kept = []
    for r, floor in enumerate(floors):
        day = load_daily(daily_csv, floor)
        kept.append(day.assign(floor=floor))
        for c, rung in enumerate(SCATTER_RUNGS):
            ax = axes[r][c]
            _scatter_panel(ax, day, rung, colours, show_ylabel=(c == 0))
            if r == 0:
                ax.set_title(
                    rung_label(rung, oneline=True), fontsize=9.5, pad=6)
            else:
                ax.set_title('')
            if r < nr - 1:
                ax.set_xlabel('')
            if c == 0:
                ax.set_ylabel(
                    f"headroom \u2265 ${floor:,.0f}   ({len(day)} site-days)"
                    "\nactual daily VCR", fontsize=9)

    handles, labels = axes[0][0].get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    fig.legend(h2, l2, loc='lower center', ncol=min(10, len(l2)),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(
        'Daily Value Capture Ratio: actual against each modelled rung\n'
        'Rows are minimum daily headroom for a site-day to be included. '
        'Area is the day\u2019s headroom; the dashed line is the least '
        'squares fit.\n'
        'r is the Pearson correlation and \u03c1 the Spearman rank '
        'correlation; both measure whether the two move together.',
        fontsize=11.5, y=0.998)
    fig.tight_layout(rect=(0, 0.035, 1, 0.955))
    _finish(fig, 'cohort_rungs_actual_resemblance', out_dir)
    return pd.concat(kept, ignore_index=True)


def chart_floor_sensitivity(daily_csv: Path,
                            floors=DEFAULT_HEADROOM_FLOORS,
                            out_dir: Optional[Path] = None) -> pd.DataFrame:
    """Figure D. How the figure C correlations depend on the floor.

    Published beside figure C because the floor is a real choice with a real
    effect, and a correlation quoted at one floor would hide it. It also shows
    the sample collapsing as the floor rises, which is the other half of the
    trade.
    """
    grid = np.arange(2.0, 30.5, 1.0)
    rows = []
    for f in grid:
        day = load_daily(daily_csv, f)
        if len(day) < 5:
            continue
        a = day['vcr_actual'].to_numpy()
        rec = {'floor': f, 'n': len(day)}
        for rung in SCATTER_RUNGS:
            rec[rung] = _pearson(day[f'vcr_{rung}'].to_numpy(), a)
        rows.append(rec)
    sens = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for rung, colour in zip(SCATTER_RUNGS, LADDER_COLOURS[2:]):
        ax.plot(sens['floor'], sens[rung], '-o', ms=3.5, lw=1.8, color=colour,
                label=rung_label(rung, oneline=True))
    for i, f in enumerate(floors):
        ax.axvline(f, color='#c0392b', lw=1.1, ls='--',
                   label='floors drawn in the scatter' if i == 0 else None)
    ax.axhline(0, color='grey', lw=0.8)
    ax.set_xlabel('minimum daily headroom for a site-day to be included ($)')
    ax.set_ylabel('correlation of daily VCR with actual')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='center left', framealpha=0.95)

    ax2 = ax.twinx()
    ax2.plot(sens['floor'], sens['n'], color='0.55', lw=1.2, ls=':')
    ax2.set_ylabel('site-days included (dotted)', fontsize=8.5, color='0.4')
    ax2.tick_params(axis='y', labelcolor='0.4')

    ax.set_title(
        'Sensitivity of the daily VCR correlation to the headroom floor\n'
        'Below about $5 the denominator is too small for a daily VCR to mean '
        'anything; above it the ordering is stable and the sample shrinks',
        fontsize=10)
    fig.tight_layout()
    _finish(fig, 'cohort_rungs_floor_sensitivity', out_dir)
    return sens


def load_attribution(site_df: pd.DataFrame) -> pd.DataFrame:
    """Split the drop from perfect information into price, load and interaction.

    With A, B, C, D the four corners of ATTRIB_2X2:

        price effect  = A - C     (imperfect prices, load still perfect)
        load effect   = A - B     (imperfect load, prices still perfect)

    Anchored on shaped_18h, so the price leg is exactly the CLEAN_CONTRASTS
    pair 'forward prices are forecasts' and nothing else. Anchoring on
    perfect_18h would fold coarse resolution and infill into the price leg as
    well, making it a bundle of three causes against the load leg's one.
        combined      = A - D
        interaction   = combined - price - load  =  B + C - A - D

    The interaction is the term worth reporting. If it is near zero the two
    errors are separable and each can be discussed on its own. If it is
    positive they compound, which is what should be expected here: both bite
    hardest on the same hot, high-price days, and a controller that
    mis-forecasts the evening peak AND mis-sizes the household load walks into
    the spike twice.
    """
    A, B, C, D = ATTRIB_2X2
    piv = site_df.pivot_table(index='site_id', columns='dispatch', values='vcr')
    missing = [k for k in ATTRIB_2X2 if k not in piv.columns]
    if missing:
        raise KeyError(
            f"attribution needs all four corners of the 2x2; missing "
            f"{missing}. Re-run run_cohort_rungs with those rungs enabled.")
    out = pd.DataFrame({
        'perfect': piv[A],
        'price_effect': piv[A] - piv[C],
        'load_effect': piv[A] - piv[B],
        'combined': piv[A] - piv[D],
        'both': piv[D],
    })
    out['interaction'] = (out['combined'] - out['price_effect']
                          - out['load_effect'])
    return out.dropna()


def chart_load_attribution(site_df: pd.DataFrame,
                           out_dir: Optional[Path] = None):
    """Figure E: how much of the VCR drop is price, how much is load.

    A waterfall per site rather than four bars per site. Four bars would make
    the reader do the subtraction, and the subtraction is the finding.
    """
    att = load_attribution(site_df).sort_values('perfect', ascending=False)
    if att.empty:
        print("  skipping the attribution figure: no site has all four corners")
        return

    sites = list(att.index)
    n = len(sites)
    fig, ax = plt.subplots(figsize=(max(7.0, 1.15 * n + 3.0), 4.6))

    xs = np.arange(n, dtype=float)
    w = 0.62
    base = att['perfect'].to_numpy()
    price = att['price_effect'].to_numpy()
    load = att['load_effect'].to_numpy()
    inter = att['interaction'].to_numpy()
    end = att['both'].to_numpy()

    # Each segment starts where the previous one finished, so the column reads
    # top-down as "this is what each loss of information costs".
    top = base
    for vals, colour, hatch, lab in (
            (price, '#4292c6', None, 'prices not known'),
            (load, '#deebf7', '///', 'load not known'),
            (inter, '#f4a582', 'xx', 'interaction')):
        ax.bar(xs, -vals, bottom=top, width=w, color=colour, hatch=hatch,
               edgecolor='#2171b5' if hatch else 'none', linewidth=0.6,
               zorder=3, label=lab)
        top = top - vals

    ax.plot(xs, base, ls='none', marker='_', ms=22, mew=2.0, color='#08306b',
            zorder=5, label='both known (shaped_18h)')
    ax.plot(xs, end, ls='none', marker='_', ms=22, mew=2.0, color='0.25',
            zorder=5, label='neither known (predispatch_full_lf)')

    # Focus the axis on the band the decomposition occupies. Unlike the VCR
    # ladder, this figure is about DIFFERENCES between two levels, so a
    # zero-anchored axis spends most of its height on empty space and squeezes
    # the thing being read. Zero is drawn if it falls inside the band.
    lo = float(np.nanmin(np.minimum(end, top))) - 0.06
    hi = float(np.nanmax(base)) + 0.06
    ax.set_ylim(lo, hi)
    if lo <= 0 <= hi:
        ax.axhline(0, lw=0.8, color='0.4', zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(sites, fontsize=7.5, rotation=45, ha='right')
    ax.set_ylabel('VCR')
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.legend(fontsize=7.5, ncol=5, loc='lower center',
              bbox_to_anchor=(0.5, -0.42), frameon=False)

    med_i = float(np.median(inter))
    share = (100 * np.median(load / np.where(att['combined'] == 0, np.nan,
                                             att['combined'])))
    ax.set_title(
        "Where the value goes when the controller stops knowing things\n"
        f"median interaction {med_i:+.3f} VCR; load forecast is a median "
        f"{share:.0f}% of the combined loss",
        fontsize=9.5, pad=8, linespacing=1.35)
    _finish(fig, 'E_load_attribution', out_dir)


# ---------------------------------------------------------------------------
# Figure F: the most refined model against observed Amber dispatch
# ---------------------------------------------------------------------------

def _refined_period_panel(ax, site_df: pd.DataFrame, colours: dict) -> dict:
    """Whole-period VCR, refined rung against actual, one point per site.

    Same shape as the daily panels beside it, one level up: a point is a site
    over the whole window rather than a site-day. Same reference line and same
    statistics too, so reading across the figure costs the reader nothing;
    only the unit of observation changes.

    Read the correlation here as description, not inference. A cohort of ten
    sites puts a confidence interval on r wide enough to cover most of the
    range, so n is printed beside r and rho rather than left for the reader to
    recall. What the panel is for is the spread of the points and where they
    sit relative to each other; the statistics are a summary of that, not a
    test of anything.

    Axes float to the data rather than spanning 0 to 1, but x and y share one
    window, as they do in the daily panels. Sharing it is what keeps the level
    comparison legible without an equality line: points above the diagonal of
    the panel are sites where Amber captured more.
    """
    piv = site_df.pivot_table(index='site_id', columns='dispatch', values='vcr')
    missing = [k for k in (REFINED, ACTUAL[0]) if k not in piv.columns]
    if missing:
        raise KeyError(f"figure F needs {missing} in the site table")

    meta = (site_df[site_df['dispatch'] == 'actual']
            .set_index('site_id')[['headroom']])
    sites = list(meta.sort_values('headroom', ascending=False).index)
    x = np.array([piv.loc[s, REFINED] for s in sites], dtype=float)
    y = np.array([piv.loc[s, ACTUAL[0]] for s in sites], dtype=float)
    h = meta.loc[sites, 'headroom'].to_numpy(dtype=float)

    # Pooled is still computed and returned for the caller, but not drawn: it
    # is a summed-numerator-over-summed-denominator summary of the same points
    # and adds a marker without adding an observation.
    coh = cohort_vcr(site_df).set_index('dispatch')['cohort_vcr']
    px, py = float(coh.get(REFINED, np.nan)), float(coh.get(ACTUAL[0], np.nan))

    # One window for both axes, floated to the data with a margin
    # proportional to the spread so the points never sit on the frame.
    pad = 0.08 * max(float(np.nanmax([x.max(), y.max()])
                           - np.nanmin([x.min(), y.min()])), 1e-6)
    lo = float(np.nanmin([x.min(), y.min()])) - pad
    hi = float(np.nanmax([x.max(), y.max()])) + pad

    # The same least squares fit the daily panels draw, for the same reason:
    # the line summarises how the two move together across the points, and
    # correlation is invariant to slope and intercept, so an equality line
    # would answer a question the statistic is not asking.
    if len(x) >= 3 and np.std(x) > 0:
        b, a0 = np.polyfit(x, y, 1)
        ends = np.array([lo, hi])
        ax.plot(ends, a0 + b * ends, ls=(0, (4, 3)), lw=1.2, color='0.45',
                zorder=1, label='least squares fit')

    # Area with headroom, as in the daily panels, so a site carrying most of
    # the cohort's value does not read the same as one carrying almost none.
    size = 30 + 230 * np.sqrt(np.clip(h, 0, None) / max(h.max(), 1e-9))
    for s, xi, yi, si in zip(sites, x, y, size):
        ax.scatter([xi], [yi], s=si, color=colours.get(s, '0.5'), alpha=0.8,
                   edgecolor='white', linewidth=0.8, zorder=3, label=s)

    # n alongside r and rho. On the daily panels the sample size is in the
    # title; here it is ten points, which is small enough that omitting it
    # from the box would be the misleading choice.
    pr, sp = _pearson(x, y), _spearman(x, y)
    fmt = lambda v: f'{v:.2f}' if np.isfinite(v) else '\u2014'
    ax.text(0.035, 0.965, f"r {fmt(pr)}\n\u03c1 {fmt(sp)}\nn {len(x)}",
            transform=ax.transAxes, va='top', ha='left', fontsize=8.5,
            bbox=dict(fc='white', ec='0.7', alpha=0.95, pad=3), zorder=6)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.25, zorder=0)
    ax.set_xlabel(f'model VCR ({rung_label(REFINED, oneline=True)})',
                  fontsize=8.5)
    ax.set_ylabel('actual (Amber) VCR', fontsize=8.5)

    gap = x - y
    return {'pooled_model': px, 'pooled_actual': py, 'pooled_gap': px - py,
            'n_sites': len(sites),
            'n_model_ahead': int(np.sum(gap > 0)),
            'median_site_gap': float(np.nanmedian(gap))}


def chart_refined_vs_actual(site_df: pd.DataFrame,
                            daily_csv: Optional[Path] = None,
                            floor: float = REFINED_FLOOR,
                            out_dir: Optional[Path] = None) -> dict:
    """Figure F. One rung, two questions, two panels of the same kind.

    Figures A to E place every rung against every other. This one drops the
    ladder and asks only about `predispatch_full_lf_sf`, because that is the
    rung the thesis argues from: it is the last column built entirely from
    information a controller has before the decision instant, so it is the
    only one that can be read as "what a well-built controller could have
    done" rather than as a diagnostic.

    Left, the whole period, one point per site. Right, the same two columns
    day by day, at a single headroom floor.

    The two panels answer different questions and are placed together because
    the answers can disagree. The left asks how much of the available value
    each captured over the period. The right asks whether they succeed on the
    same days: a model that clears actual by a wide margin on the left while
    correlating weakly on the right is beating Amber for reasons that have
    nothing to do with reading the same days correctly, and that distinction
    matters more than the margin does.

    Same reference line and same statistics in both, so only the unit of
    observation changes as the reader moves across. The windows differ,
    because a site over a whole period and a site-day are not on the same
    scale, and forcing one window on both would flatten whichever panel lost.

    Neither panel supports a causal reading of the gap. Amber has a better
    price forecast than pre-dispatch, a worse site forecast than the fitted
    load and solar, and its own risk appetite, and those push in opposite
    directions.
    """
    floor = float(floor)
    have_daily = False
    if daily_csv is not None and Path(daily_csv).exists():
        cols = set(pd.read_csv(daily_csv, nrows=1).columns)
        have_daily = bool({REFINED, f'vcr_{REFINED}'} & cols)
        if not have_daily:
            print(f"  figure F: '{REFINED}' is not in {daily_csv}; drawing the "
                  f"period panel only")

    # One colour map for the whole figure, built from the daily table when
    # there is one so the site colours match figure C exactly.
    ids = (sorted(pd.read_csv(daily_csv)['site_id'].unique()) if have_daily
           else sorted(site_df['site_id'].unique()))
    colours = _site_colours(ids)

    ncols = 2 if have_daily else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5.4 * ncols, 5.4),
                             squeeze=False)
    ax_left = axes[0][0]

    stats = _refined_period_panel(ax_left, site_df, colours)
    ax_left.set_title('Whole period, by site', fontsize=10, pad=6)

    if have_daily:
        day = load_daily(Path(daily_csv), floor, rungs=(REFINED,))
        ax_right = axes[0][1]
        _scatter_panel(ax_right, day, REFINED, colours, show_ylabel=True,
                       clip=REFINED_VCR_CLIP)
        ax_right.set_title(f"Daily, headroom \u2265 ${floor:,.0f}"
                           f"   ({len(day)} site-days)", fontsize=10, pad=6)
        ax_right.set_xlabel('model daily VCR', fontsize=8.5)
        ax_right.set_ylabel('actual (Amber) daily VCR', fontsize=8.5)
        stats['n_site_days'] = len(day)

    # One legend under both panels. Site colours and the fitted line appear in
    # both, so a key inside each would spend two lots of panel space on the
    # same thing.
    handles, labels = [], []
    for ax in axes[0]:
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        # The clipped entry keeps its count: there is one daily panel, so the
        # count is unambiguous and worth carrying.
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    fig.legend(h2, l2, loc='upper center', ncol=min(7, len(l2)), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.075))

    fig.tight_layout(rect=(0, 0.085, 1, 1))
    _finish(fig, 'F_refined_vs_actual', out_dir)
    return stats


# ---------------------------------------------------------------------------
# Thesis figure: model against Amber VCR (docs/thesis_figure_spec.md)
# ---------------------------------------------------------------------------

THESIS_SCATTER_HEIGHT_IN = 3.6
# Marker area is proportional to headroom: SIZE_MAX_PT2 at the largest key
# value of each panel. Each panel has its own key, because a site over the
# whole period and a site-day differ in headroom by an order of magnitude.
SIZE_MAX_PT2 = 150.0
PERIOD_SIZE_KEY = (100, 300, 700)       # $ headroom, whole period per site
DAILY_SIZE_KEY = (10, 50, 200)          # $ headroom, per site-day


def _area(h, key_max) -> np.ndarray:
    return SIZE_MAX_PT2 * np.clip(np.asarray(h, dtype=float), 0, None) / key_max


def _thesis_scatter(ax, x, y, h, site_ids, colours, key, key_title,
                    lims) -> None:
    """Points coloured by site with area proportional to headroom, the least
    squares fit, r and rho, and a headroom size key."""
    lo, hi = lims
    x, y, h = map(np.asarray, (x, y, h))
    site_ids = np.asarray(site_ids)
    b, a0 = np.polyfit(x, y, 1)
    ends = np.array([lo, hi])
    ax.plot(ends, a0 + b * ends, ls='--', lw=ts.REFERENCE_LW,
            color=ts.REFERENCE_COLOUR, zorder=2, label='Least squares fit')
    if lo < 0:
        ax.axhline(0, color=ts.SEPARATOR_COLOUR, lw=0.5, zorder=1)
        ax.axvline(0, color=ts.SEPARATOR_COLOUR, lw=0.5, zorder=1)

    inside = (x >= lo) & (x <= hi) & (y >= lo) & (y <= hi)
    area = _area(h, max(key))
    for site in sorted(set(site_ids)):
        m = (site_ids == site) & inside
        if m.any():
            ax.scatter(x[m], y[m], s=area[m], color=colours[site], alpha=0.8,
                       edgecolor='white', linewidth=0.5, zorder=3, label=site)
    if (~inside).any():
        ax.scatter(np.clip(x[~inside], lo, hi), np.clip(y[~inside], lo, hi),
                   s=20, marker='^', facecolor='none', edgecolor='0.35',
                   linewidth=0.8, zorder=4,
                   label=f'Clipped ({int((~inside).sum())})')

    ax.text(0.04, 0.96, f"r = {_pearson(x, y):.2f}\n\u03c1 = {_spearman(x, y):.2f}",
            transform=ax.transAxes, ha='left', va='top',
            fontsize=ts.ANNOTATION_SIZE, zorder=6, bbox=ts.LABEL_BOX)

    handles = [ax.scatter([], [], s=_area(v, max(key)), color='0.6',
                          edgecolor='white', linewidth=0.5) for v in key]
    # handleheight in font units, sized so the largest key marker clears
    # the box edge; scatteryoffsets places the marker centre level with the
    # label (0.5 sits it above the label in a tall handle box).
    size_key = ax.legend(handles, [f'\\${v:,}' for v in key], title=key_title,
                         loc='lower right', ncol=len(key), columnspacing=0.8,
                         handletextpad=0.3, borderpad=0.5, handleheight=2.4,
                         scatteryoffsets=[0.3],
                         title_fontsize=ts.ANNOTATION_SIZE)
    size_key.set_zorder(20)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    # Square box with equal x and y ranges, so one unit is the same length
    # on both axes. set_box_aspect rather than set_aspect: constrained layout
    # allows for it when placing titles.
    ax.set_box_aspect(1)
    ax.set_xlabel('Model value capture ratio')


def chart_thesis_model_vs_amber(site_df: pd.DataFrame, daily_csv: Path,
                                floor: float = REFINED_FLOOR,
                                out_dir: Path = THESIS_CHART_DIR) -> None:
    """Model (REFINED rung) against Amber VCR: by site over the whole period
    (left) and by site-day at the headroom floor (right)."""
    piv = site_df.pivot_table(index='site_id', columns='dispatch', values='vcr')
    meta = site_df[site_df['dispatch'] == 'actual'].set_index('site_id')
    sites = sorted(piv.index)
    px = piv.loc[sites, REFINED].to_numpy(dtype=float)
    py = piv.loc[sites, ACTUAL[0]].to_numpy(dtype=float)
    ph = meta.loc[sites, 'headroom'].to_numpy(dtype=float)
    # Extra room at the low end leaves the size key clear of the points.
    spread = max(px.max(), py.max()) - min(px.min(), py.min())
    p_lims = (min(px.min(), py.min()) - 0.17 * spread,
              max(px.max(), py.max()) + 0.08 * spread)

    day = load_daily(Path(daily_csv), float(floor), rungs=(REFINED,))
    colours = dict(zip(sorted(set(sites) | set(day['site_id'])),
                       ts.CATEGORICAL_COLOURS))

    ts.apply_style()
    fig = ts.new_figure(THESIS_SCATTER_HEIGHT_IN)
    ax_p, ax_d = fig.subplots(1, 2)
    _thesis_scatter(ax_p, px, py, ph, sites, colours, PERIOD_SIZE_KEY,
                    'Site headroom', p_lims)
    ax_p.set_title(f'Whole period, by site (n = {len(sites)})',
                   fontsize=ts.SUBTITLE_SIZE)
    _thesis_scatter(ax_d, day[f'vcr_{REFINED}'], day['vcr_actual'],
                    day['headroom'], day['site_id'], colours, DAILY_SIZE_KEY,
                    'Site-day headroom', REFINED_VCR_CLIP)
    ax_d.set_title(f'Site-days with headroom \u2265 \\${floor:,.0f} '
                   f'(n = {len(day)})', fontsize=ts.SUBTITLE_SIZE)
    fig.supylabel('Amber value capture ratio', fontsize=ts.RC['axes.labelsize'])

    # Site colours and the fit are common to both panels: one legend below.
    seen = {}
    for ax in (ax_p, ax_d):
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    labels = sorted([l for l in seen if l.startswith('EM-')]) + \
        [l for l in seen if not l.startswith('EM-')]
    handles = []
    for l in labels:
        if l.startswith('EM-'):
            handles.append(plt.Line2D([], [], ls='none', marker='o', ms=5,
                                      mfc=colours[l], mec='white', mew=0.5,
                                      alpha=0.8))
        else:
            handles.append(seen[l])
    fig.legend(handles, labels, loc='outside lower center', ncol=len(labels),
               frameon=False, columnspacing=1.0, handletextpad=0.3)
    ts.panel_suptitle(fig, 'Model vs Amber value capture ratio')
    ts.save_figure(fig, Path(out_dir) / 'em_vcr_model_vs_amber.png')


def chart_all(csv_path: Path, out_dir: Optional[Path] = None,
              daily_csv: Optional[Path] = None,
              headroom_floors=DEFAULT_HEADROOM_FLOORS) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    thick = df[df['headroom'] >= SITE_HEADROOM_FLOOR_DOLLARS]
    dropped = set(df['site_id']) - set(thick['site_id'])
    if dropped:
        print(f"  excluding {len(dropped)} site(s) below the "
              f"${SITE_HEADROOM_FLOOR_DOLLARS:.0f} headroom floor: "
              f"{', '.join(sorted(dropped))}")
    chart_by_site(thick, out_dir)
    chart_pooled(thick, out_dir)
    chart_thesis_ladders(thick)
    try:
        chart_load_attribution(thick, out_dir)
    except KeyError as e:
        print(f"  skipping the attribution figure: {e}")
    try:
        chart_refined_vs_actual(thick, daily_csv, out_dir=out_dir)
    except KeyError as e:
        print(f"  skipping the refined-vs-actual figure: {e}")
    if daily_csv is not None and Path(daily_csv).exists():
        chart_thesis_model_vs_amber(thick, Path(daily_csv))
        chart_actual_resemblance(Path(daily_csv), headroom_floors, out_dir)
        chart_floor_sensitivity(Path(daily_csv), headroom_floors, out_dir)
    elif daily_csv is not None:
        print(f"  skipping the resemblance figures: {daily_csv} not found")
    return thick


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--csv', type=Path, default=DEFAULT_CSV)
    parser.add_argument('--daily-csv', type=Path, default=DEFAULT_DAILY_CSV,
                        help='Per-day table, for the resemblance figures')
    parser.add_argument('--headroom-floors',
                        default=','.join(f'{f:g}'
                                         for f in DEFAULT_HEADROOM_FLOORS),
                        help='Comma separated minimum daily headroom values, '
                             'one row of the scatter each. Below roughly $5 a '
                             'daily VCR is a ratio of two small numbers; see '
                             'the sensitivity figure.')
    parser.add_argument('--out-dir', type=Path, default=None,
                        help='Omit to show interactively')
    args = parser.parse_args()
    chart_all(args.csv, args.out_dir, daily_csv=args.daily_csv,
              headroom_floors=tuple(float(f) for f in
                                    args.headroom_floors.split(',') if f))