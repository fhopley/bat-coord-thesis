#!/usr/bin/env python3
"""Poster figures.

Figures drawn for the A1 poster (thesis/poster/poster_spec.md). Each is built from
saved results so it can be restyled without re-running the model.

value_ratio_iqr
    Relative value of coordination on Amber, household bill saving and net
    wholesale saving on one axes. For each battery size the household and
    wholesale interquartile ranges share a centre line, one above the other,
    with a median rule in each. No whiskers.

    The site ratios come from retailer_briefing.py: the imperfect-foresight
    (achievable) run over the self-consumption run, per site and capacity,
    weighted by PV size band and pooled nationally by state mix exactly as
    the briefing's national ratio charts (coordination_household_value_ratio
    and coordination_wholesale_value_ratio). Nothing about loading, floors,
    weighting or quantiles is reimplemented; this script calls the
    briefing's functions.

saving_and_ratio
    Poster only. Two panes under one title. Left: mean annual household bill
    saving on Amber by battery size (thesis Figure 19 in the household
    colour): perfect foresight, imperfect foresight as the conservative to
    achievable range, and self-consumption, every modelled capacity marked,
    ticks at 10 kWh. Right: the value_ratio_iqr boxes at the 10 kWh
    capacities. The plot boxes span the same edges as the panes of the
    capped-offers figure (poster_offers_figure.py), so the two line up on
    the poster.

    The curves are the briefing's national curve set (each year in a state
    equally, states by state mix), written to
    poster_household_saving_curves.csv by the same pass that writes the
    ratio quantiles.

event_exposure
    Mean net import per site in intervals above $300/MWh for four cohorts,
    20 kWh battery (thesis Figure 34 reduced to No CER, Solar only,
    Self-consumption and Amber). The four values are the printed values of
    Figure 34, held in poster_event_exposure.csv.

retailer_margin_spread
    Distribution of the retailer's gross margin per household per month
    across the 240 state-months for Amber, Cohort Hedging and Retailer Pool
    Retention, monthly settlement, 20 kWh, $300/MWh cap (thesis Figure 45
    without Tesla Electric and daily settlement). Read from
    analysis/output/cap_account_retailer/20kWh/x300/retailer_cohort_months.csv
    and weighted as alt_retailer_spread_figure.py does.

Each figure is drawn in the thesis format by default and in the draft poster
format (poster_style.py, printed size, 18 pt text) with --style poster.

Files
-----
    analysis/output/poster/cache/amber_<state>_<year>.pkl
    analysis/output/poster/cache/amber_curves_<state>_<year>.pkl
        Per-scenario site-ratio samples and mean saving curves. Written once,
        so an interrupted run resumes at the first scenario without a cache
        file.
    analysis/output/poster/poster_household_saving_curves.csv
        National mean saving by metric, mode and capacity (0 to 50 kWh).
    analysis/output/poster/poster_value_ratio_stats.csv
        Weighted quantiles for each metric and capacity.
    analysis/output/poster/poster_value_ratio_iqr.png
        The figure, in the thesis format (docs/thesis_figure_spec.md). The
        poster format is applied once the figure set is agreed.

Usage
-----
    python3 analysis/scripts/poster_figures.py
    python3 analysis/scripts/poster_figures.py --replot
    python3 analysis/scripts/poster_figures.py --replot --no-notes
    python3 analysis/scripts/poster_figures.py --replot --style poster \\
        --figure value_ratio_iqr event_exposure retailer_margin_spread
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

import financial_analysis as fa
import alt_retailer_spread_figure as asl
import retailer_briefing as rb
import poster_style as ps
import thesis_style as ts

BRIEFING = 'amber'
# The mode the briefing's ratio charts draw: imperfect foresight, achievable.
NUMERATOR_MODE = 'LP_ROLL'

# Household and wholesale are told apart by colour alone, so they take two
# members of the categorical set that separate for colour-vision deficiency.
# INK is the darker step that carries the box edge and median rule.
HOUSEHOLD = dict(metric='hh_saving', label='Household bill saving',
                 color=ts.CATEGORICAL_COLOURS[0], ink='#1a4f8f')
WHOLESALE = dict(metric='ws_saving', label='Net wholesale saving',
                 color=ts.CATEGORICAL_COLOURS[1], ink='#a8431a')
SERIES = (WHOLESALE, HOUSEHOLD)   # highest first, as the legend reads

# Both ranges share a centre, so the box takes the single-series width.
BOX_WIDTH = rb.SINGLE_BOX_WIDTH
BOX_ALPHA = rb.BOX_ALPHA
BOX_EDGE_WIDTH = rb.BOX_EDGE_WIDTH
MEDIAN_WIDTH = BOX_EDGE_WIDTH * 1.6

# Draft poster sizes (cm). The hero spans two columns, the others one.
HERO_WIDTH_CM = ps.WIDE_W_CM
HERO_HEIGHT_CM = 18.2
EXPOSURE_HEIGHT_CM = 8.0
MARGIN_HEIGHT_CM = 9.0

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
STAT_COLUMNS = ('p05', 'q1', 'median', 'q3', 'p95')

DEFAULT_WEIGHTS = Path('analysis/config/pv_size_bands.yaml')
DEFAULT_OUTPUT = Path('analysis/output/poster')
STATS_NAME = 'poster_value_ratio_stats.csv'
FIGURES = ('value_ratio_iqr', 'saving_and_ratio', 'event_exposure',
           'retailer_margin_spread')
CURVES_NAME = 'poster_household_saving_curves.csv'
FIGURE_NAME = 'poster_value_ratio_iqr'


# ── Site ratio samples ────────────────────────────────────────────────────────

def _scenario_data(args, briefing, build_scenario, state: str, year: str,
                  cache_dir: Path) -> tuple[dict, dict] | None:
    """Site ratios and mean saving curves for one scenario, from the cache
    where both exist."""
    cache = cache_dir / f'{BRIEFING}_{state}_{year}.pkl'
    curve_cache = cache_dir / f'{BRIEFING}_curves_{state}_{year}.pkl'
    if cache.exists() and curve_cache.exists():
        with open(cache, 'rb') as f, open(curve_cache, 'rb') as g:
            return pickle.load(f), pickle.load(g)
    print(f'Loading {state} {year}...')
    built = build_scenario(state, year)
    if built is None:
        return None
    samples, reports = rb._collect_ratio_samples(
        built['savings'], built['site_weight'], briefing['charts'])
    for name, floor, dropped, total in reports:
        if dropped:
            print(f'  {name}: {dropped} of {total} site-capacity ratio(s) '
                  f'dropped for {state} {year} (self-consumption saving at '
                  f'or below ${floor:.0f}).')
    curves = built['curves']
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache, 'wb') as f:
        pickle.dump(samples, f)
    with open(curve_cache, 'wb') as f:
        pickle.dump(curves, f)
    return samples, curves


def compute_stats(args) -> pd.DataFrame:
    """Weighted quantiles of the national site-ratio distribution."""
    matplotlib.rcdefaults()
    ts.apply_style()
    briefing = rb.BRIEFINGS[BRIEFING]
    plans = briefing['plans']
    fa.install_plans(plans, plans[0]['sc_mode'])
    weights_cfg = fa.load_weights_config(args.weights_file)
    build_scenario = fa.make_scenario_builder(args.data_root, weights_cfg,
                                              'own')

    scenarios = fa.discover_scenarios(args.data_root)
    if args.states:
        scenarios = [(s, y) for s, y in scenarios if s in args.states]
    if args.years:
        scenarios = [(s, y) for s, y in scenarios if y in args.years]
    if not scenarios:
        raise SystemExit('No scenarios match --states/--years.')

    by_state: dict[str, list[dict]] = {}
    curves_by_state: dict[str, list[dict]] = {}
    for state, year in scenarios:
        data = _scenario_data(args, briefing, build_scenario, state, year,
                              args.output_dir / 'cache')
        if data is not None:
            by_state.setdefault(state, []).append(data[0])
            curves_by_state.setdefault(state, []).append(data[1])

    # Each year in a state counts equally; states are combined by state mix.
    # The same structure as the briefing's national ratio charts.
    mix = weights_cfg['state_mix']
    national = rb._merge_samples([
        rb._scale_samples(rb._merge_samples(
            [rb._scale_samples(s, 1.0 / len(sets)) for s in sets]),
            mix.get(state, 0.0))
        for state, sets in by_state.items()])

    rows = []
    q = np.array(QUANTILES)
    for spec in SERIES:
        for size in fa.BATTERY_SIZES:
            drawn = national.get((spec['metric'], NUMERATOR_MODE, size))
            if drawn is None:
                continue
            vals = rb._weighted_quantile(drawn[0], drawn[1], q)
            rows.append(dict(metric=spec['metric'], battery_kwh=size,
                             n_site_years=len(drawn[0]),
                             **dict(zip(STAT_COLUMNS, vals))))
    stats = pd.DataFrame(rows)
    write_curves(curves_by_state, mix, args.output_dir)
    # The scenario set travels with the numbers so a replot can state it.
    stats['states'] = ','.join(sorted(by_state))
    stats['years'] = ','.join(sorted({y for _, y in scenarios}))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats.to_csv(args.output_dir / STATS_NAME, index=False)
    print(f'  Saved: {args.output_dir / STATS_NAME}')
    return stats


def write_curves(curves_by_state: dict[str, list[dict]], mix: dict,
                 output_dir: Path) -> pd.DataFrame:
    """National mean saving curves, pooled as the briefing's national
    savings charts: a straight mean across years in a state, then state mix."""
    per_state = {
        state: (fa.average_curves_across_years(sets) if len(sets) > 1
                else sets[0])
        for state, sets in curves_by_state.items()}
    national = fa.combine_curves_state_weighted(per_state, mix)
    xs = [0, *fa.BATTERY_SIZES]
    rows = [dict(metric=metric, mode=mode, battery_kwh=x, saving=float(v))
            for metric, by_mode in national.items()
            for mode, ys in by_mode.items() for x, v in zip(xs, ys)]
    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / CURVES_NAME, index=False)
    print(f'  Saved: {output_dir / CURVES_NAME}')
    return df


# ── Figure ────────────────────────────────────────────────────────────────────

def _draw_iqr_box(ax, x: float, q1: float, med: float, q3: float,
                  spec: dict, edge_lw: float = BOX_EDGE_WIDTH,
                  median_lw: float = MEDIAN_WIDTH,
                  width: float = BOX_WIDTH) -> None:
    """IQR rectangle and median rule. No whiskers."""
    half = width / 2
    ax.add_patch(mpatches.Rectangle(
        (x - half, q1), width, q3 - q1, facecolor=spec['color'],
        alpha=BOX_ALPHA, edgecolor=spec['ink'], lw=edge_lw, zorder=2))
    ax.plot([x - half, x + half], [med, med], color=spec['ink'],
            lw=median_lw, zorder=3)


def overlaps(stats: pd.DataFrame) -> list[int]:
    """Capacities at which the household and wholesale IQRs overlap."""
    hh = stats[stats.metric == HOUSEHOLD['metric']].set_index('battery_kwh')
    ws = stats[stats.metric == WHOLESALE['metric']].set_index('battery_kwh')
    common = hh.index.intersection(ws.index)
    return [int(s) for s in common if ws.loc[s, 'q1'] <= hh.loc[s, 'q3']]


def make_figure(stats: pd.DataFrame, output_dir: Path, note: str | None,
                poster: bool = False):
    if poster:
        ps.apply_style()
        fig = ps.new_figure(HERO_WIDTH_CM, HERO_HEIGHT_CM, layout=None)
        edge_lw, median_lw, ref_lw = ps.BOX_LW, ps.MEDIAN_LW, ps.REFERENCE_LW
    else:
        fig = ts.new_figure(height_in=3.0)
        edge_lw, median_lw, ref_lw = (BOX_EDGE_WIDTH, MEDIAN_WIDTH,
                                      ts.REFERENCE_LW)
    if poster:
        # Plot box on the shared wide-figure edges (poster_style).
        w, h = HERO_WIDTH_CM, HERO_HEIGHT_CM
        bottom, top = 2.5, ps.TITLE_BAND_CM + 0.3
        ax = fig.add_axes([ps.PLOT_LEFT_CM / w, bottom / h,
                           (ps.PLOT_RIGHT_CM - ps.PLOT_LEFT_CM) / w,
                           1 - (bottom + top) / h])
    else:
        ax = fig.add_subplot()
    sizes = np.array(fa.BATTERY_SIZES, dtype=float)

    drawn_y = [1.0]
    for spec in SERIES:
        sub = stats[stats.metric == spec['metric']].set_index('battery_kwh')
        for size in fa.BATTERY_SIZES:
            if float(size) not in sub.index:
                continue
            r = sub.loc[float(size)]
            _draw_iqr_box(ax, float(size), r.q1, r['median'], r.q3, spec,
                          edge_lw, median_lw)
            drawn_y.extend((r.q1, r.q3))

    ax.axhline(1.0, color=ts.REFERENCE_COLOUR, lw=ref_lw, ls='--', zorder=1)
    ax.set_xlabel('Battery capacity (kWh)')
    if poster:
        ax.set_ylabel('Ratio to self-consumption')
        ps.add_title(fig, ax, 'Value of coordination relative to '
                     'self-consumption (Amber)')
    else:
        ax.set_ylabel('Ratio to self-consumption saving')
        ax.set_title('Value of coordination relative to self-consumption, '
                     'Amber')
    pad = BOX_WIDTH / 2 + 0.6
    ax.set_xlim(sizes[0] - pad, sizes[-1] + pad)
    lo, hi = min(drawn_y), max(drawn_y)
    span = max(hi - lo, 0.1)
    # Headroom above the boxes holds the data note, upper right, and is left
    # out when the note is.
    ax.set_ylim(lo - 0.12 * span, hi + (0.26 if note else 0.10) * span)
    ax.set_xticks(sizes)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f'{v:.1f}\u00d7'))

    handles = [mpatches.Patch(facecolor=s['color'], alpha=BOX_ALPHA,
                              edgecolor=s['ink'], lw=edge_lw)
               for s in SERIES]
    ax.legend(handles=handles, labels=[s['label'] for s in SERIES],
              loc='upper left', ncol=1)
    if note:
        # _draw_notes takes the legend corner and draws the note opposite it,
        # so 'lower left' puts the note upper right, in the headroom, clear of
        # the legend and of the parity line.
        rb._draw_notes(ax, note, 'lower left')
    _save(fig, output_dir, FIGURE_NAME, poster)


# ── Saving and ratio (poster) ─────────────────────────────────────────────────

SAVING_RATIO_NAME = 'poster_saving_and_ratio'
SAVING_RATIO_TITLE = ('Household bill saving and the relative value of '
                      'coordination (Amber)')
# Pane geometry (cm) as in poster_offers_figure.py: two equal plot boxes
# between the shared wide-figure edges with a 2 cm gap, plot tops 2.6 cm
# below the figure top (title band and a pane title).
PANE_GAP_CM = 2.0
PANE_TOP_CM = 2.6
PANE_BOTTOM_CM = 2.4
RATIO_SIZES = (10, 20, 30, 40, 50)
RATIO_BOX_WIDTH = BOX_WIDTH * 2          # sizes 10 kWh apart, not 5


def _pane(fig, left_cm: float, width_cm: float, h_cm: float):
    w = HERO_WIDTH_CM
    return fig.add_axes([left_cm / w, PANE_BOTTOM_CM / h_cm, width_cm / w,
                         1 - (PANE_BOTTOM_CM + PANE_TOP_CM) / h_cm])


def make_saving_and_ratio_figure(curves: pd.DataFrame, stats: pd.DataFrame,
                                 output_dir: Path) -> None:
    ps.apply_style()
    h = HERO_HEIGHT_CM
    fig = ps.new_figure(HERO_WIDTH_CM, h, layout=None)
    pw = (ps.PLOT_RIGHT_CM - ps.PLOT_LEFT_CM - PANE_GAP_CM) / 2
    ax_s = _pane(fig, ps.PLOT_LEFT_CM, pw, h)
    ax_r = _pane(fig, ps.PLOT_LEFT_CM + pw + PANE_GAP_CM, pw, h)

    # Left: the briefing's household savings series, in the household colour.
    briefing = rb.BRIEFINGS[BRIEFING]
    chart = next(c for c in briefing['charts']
                 if c.get('kind', 'series') == 'series'
                 and c['metric'] == HOUSEHOLD['metric'])
    series = [dict(spec, color=HOUSEHOLD['color'],
                   marker_edge=HOUSEHOLD['ink'])
              for spec in rb._series_for(briefing, chart)]
    hh = curves[curves.metric == HOUSEHOLD['metric']]
    by_mode = {m: g.sort_values('battery_kwh')['saving'].to_numpy()
               for m, g in hh.groupby('mode')}
    marks = ts.Marks(ps.DATA_LW, ps.MARKER, ps.FRAME_LW * 0.6)
    handles, labels = rb._draw_series(ax_s, by_mode, series, 0.0,
                                      marks=marks)
    ax_s.set_xticks(range(0, 51, 10))
    ax_s.set_xlabel('Battery capacity (kWh)')
    ax_s.set_ylim(bottom=0)
    ax_s.set_title('Mean annual bill saving ($/site)', fontsize=ps.TEXT)
    ax_s.legend(handles=handles, labels=labels, loc='lower right', ncol=1,
                labelspacing=0.25, borderpad=0.3, borderaxespad=0.3,
                handleheight=1.3, handlelength=2.2,
                handler_map={rb._RangeKey: rb._RangeKeyHandler()})

    # Right: ratio boxes at the 10 kWh capacities.
    drawn_y = [1.0]
    for spec in SERIES:
        sub = stats[stats.metric == spec['metric']].set_index('battery_kwh')
        for size in RATIO_SIZES:
            if float(size) not in sub.index:
                continue
            r = sub.loc[float(size)]
            _draw_iqr_box(ax_r, float(size), r.q1, r['median'], r.q3, spec,
                          ps.BOX_LW, ps.MEDIAN_LW, RATIO_BOX_WIDTH)
            drawn_y.extend((r.q1, r.q3))
    ax_r.axhline(1.0, color=ts.REFERENCE_COLOUR, lw=ps.REFERENCE_LW, ls='--',
                 zorder=1)
    pad = RATIO_BOX_WIDTH / 2 + 1.5
    ax_r.set_xlim(RATIO_SIZES[0] - pad, RATIO_SIZES[-1] + pad)
    ax_r.set_ylim(0, max(drawn_y) * 1.08)
    ax_r.set_xticks(RATIO_SIZES)
    ax_r.set_xlabel('Battery capacity (kWh)')
    ax_r.yaxis.set_major_locator(mticker.MultipleLocator(2))
    ax_r.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f'{v:g}\u00d7'))
    ax_r.set_title('Coordinated saving relative to self-consumption',
                   fontsize=ps.TEXT)
    box_handles = [mpatches.Patch(facecolor=sp['color'], alpha=BOX_ALPHA,
                                  edgecolor=sp['ink'], lw=ps.BOX_LW)
                   for sp in SERIES]
    ax_r.legend(handles=box_handles, labels=[sp['label'] for sp in SERIES],
                loc='upper left', ncol=1)

    ps.add_title(fig, None, SAVING_RATIO_TITLE)
    ps.save_figure(fig, output_dir / (SAVING_RATIO_NAME + '_poster'))


def _save(fig, output_dir: Path, name: str, poster: bool) -> None:
    (ps if poster else ts).save_figure(
        fig, output_dir / (name + ('_poster' if poster else '')))


# ── Event exposure ────────────────────────────────────────────────────────────

EXPOSURE_NAME = 'poster_event_exposure'
EXPOSURE_COLUMNS = ('label', 'value_kw')
# Thesis Figure 34 values at 20 kWh, achievable imperfect foresight, national,
# 2021 to 2025, net import in intervals above $300/MWh (kW per site).
EXPOSURE_VALUES = (('No CER (load only)', 1.31), ('Solar only', 0.92),
                   ('Self-consumption', 0.56), ('Amber (coordinated)', -1.18))
EXPOSURE_COLOURS = ('black', '#aaaaaa', '#c8c8c8', rb.AMBER)


def write_exposure_csv(output_dir: Path) -> pd.DataFrame:
    df = pd.DataFrame(EXPOSURE_VALUES, columns=EXPOSURE_COLUMNS)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / (EXPOSURE_NAME + '.csv'), index=False)
    return df


def make_event_exposure_figure(df: pd.DataFrame, output_dir: Path,
                               poster: bool = False):
    """Horizontal bars, one per cohort, the labels in the left margin."""
    if poster:
        ps.apply_style()
        fig = ps.new_figure(ps.ONE_COL_CM, EXPOSURE_HEIGHT_CM, title=True)
        text = ps.TEXT
    else:
        fig = ts.new_figure(height_in=2.4)
        text = ts.ANNOTATION_SIZE
    ax = fig.add_subplot()
    y = np.arange(len(df))[::-1]
    values = df['value_kw'].to_numpy()
    for yi, v, c, label in zip(y, values, EXPOSURE_COLOURS, df['label']):
        hatch = '///' if label.startswith('Self') else None
        ax.barh(yi, v, height=0.72, color=c, linewidth=0, zorder=3,
                **(dict(edgecolor='white', hatch=hatch) if hatch else {}))
        ax.annotate(f'{v:.2f}'.replace('-', '\u2212'), (v, yi),
                    xytext=(5 if v >= 0 else -5, 0), textcoords='offset points',
                    ha='left' if v >= 0 else 'right', va='center',
                    fontsize=text, zorder=4)
    ax.axvline(0, color=ts.REFERENCE_COLOUR, lw=0.8, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(df['label'])
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='y', visible=False)
    lo, hi = min(values.min(), 0), max(values.max(), 0)
    span = hi - lo
    ax.set_xlim(lo - 0.22 * span, hi + 0.22 * span)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f'{v:.1f}'.replace('-', '\u2212')))
    ax.set_xlabel('Mean net import per site (kW)' if poster else
                  r'Mean net import above \$300/MWh (kW per site)')
    if poster:
        ps.add_title(fig, ax, 'Net import during price spikes (>$300/MWh)')
    else:
        ax.set_title('Net import in intervals above \\$300/MWh, 20 kWh')
    _save(fig, output_dir, EXPOSURE_NAME, poster)


# ── Retailer margin spread ────────────────────────────────────────────────────

MARGIN_NAME = 'poster_retailer_margin_spread'
MARGIN_CSV = Path('analysis/output/cap_account_retailer/20kWh/x300/'
                  'retailer_cohort_months.csv')
MARGIN_OFFERS = (
    ('Amber', 'Amber', asl.AMBER_COLOUR, asl.AMBER_INK),
    ('cohort_x300_monthly_s0.01', 'Cohort\nHedging', asl.ALT_COLOUR,
     asl.ALT_INK),
    ('retailer_x300_monthly_a0.9', 'Retailer Pool\nRetention',
     asl.ALT_COLOUR, asl.ALT_INK),
)


def margin_stats(csv: Path) -> pd.DataFrame:
    """Weighted box statistics per offer, the figure's saved results."""
    d = pd.read_csv(csv, usecols=['variant', asl.VALUE_COL, 'weight'])
    rows = []
    for variant, label, _, _ in MARGIN_OFFERS:
        m = d['variant'] == variant
        if not m.any():
            raise ValueError(f'no rows for {variant} in {csv}')
        s = asl.box_stats(d.loc[m, asl.VALUE_COL].to_numpy(),
                          d.loc[m, 'weight'].to_numpy())
        rows.append(dict(variant=variant, label=label.replace('\n', ' '),
                         n_cohort_months=int(m.sum()), mean=s['mean'],
                         p05=s['whislo'], q1=s['q1'], median=s['med'],
                         q3=s['q3'], p95=s['whishi']))
    return pd.DataFrame(rows)


def make_margin_spread_figure(stats: pd.DataFrame, output_dir: Path,
                              poster: bool = False):
    """IQR box, median rule, mean diamond and 5th to 95th percentile whiskers
    on a linear axis."""
    if poster:
        ps.apply_style()
        fig = ps.new_figure(ps.ONE_COL_CM, MARGIN_HEIGHT_CM, title=True)
        box_lw, med_lw, ms = ps.BOX_LW, ps.MEDIAN_LW, ps.MARKER
    else:
        fig = ts.new_figure(height_in=2.6)
        box_lw, med_lw, ms = (BOX_EDGE_WIDTH, MEDIAN_WIDTH, ts.MARKER_SIZE)
    ax = fig.add_subplot()
    for i, (r, (_, label, fill, ink)) in enumerate(
            zip(stats.itertuples(), MARGIN_OFFERS)):
        half = 0.3
        ax.plot([i, i], [r.p05, r.p95], color=ink, lw=box_lw, zorder=2)
        for yv in (r.p05, r.p95):
            ax.plot([i - 0.12, i + 0.12], [yv, yv], color=ink, lw=box_lw,
                    zorder=2)
        ax.add_patch(mpatches.Rectangle(
            (i - half, r.q1), 2 * half, max(r.q3 - r.q1, 0.0),
            facecolor=fill, alpha=0.35, edgecolor=ink, lw=box_lw, zorder=3))
        ax.plot([i - half, i + half], [r.median, r.median], color=ink,
                lw=med_lw, zorder=4)
        ax.plot([i], [r.mean], marker='D', ms=ms, mfc='white', mec=ink,
                mew=box_lw * 0.8, ls='none', zorder=5)
    ax.set_xlim(-0.6, len(stats) - 0.4)
    ax.set_ylim(0, 55)
    ax.set_yticks([0, 25, 50])
    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels([o[1] for o in MARGIN_OFFERS])
    ax.tick_params(axis='x', length=0)
    ax.grid(axis='x', visible=False)
    ax.set_ylabel('$ per site per month' if poster else
                  r'Gross margin (\$ per site per month)')
    if poster:
        ps.add_title(fig, ax, 'Retailer margin per household per month')
    else:
        ax.set_title("Retailer's gross margin across state-months, "
                     r"\$300/MWh cap, 20 kWh")
    _save(fig, output_dir, MARGIN_NAME, poster)


def _note(stats: pd.DataFrame) -> str:
    states = str(stats['states'].iloc[0]).split(',')
    years = str(stats['years'].iloc[0]).split(',')
    where = f"National ({', '.join(states)})" if states else 'National'
    when = (f'{years[0]}–{years[-1]}' if years and len(years) > 1
            else (years[0] if years else ''))
    return (f'{where}, {when}\nWeighted by PV size & state mix vs solar only\n'
            f'Boxes: individual site ratios — weighted median and IQR')


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root', type=Path, default=Path('.'))
    p.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--weights-file', type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument('--states', nargs='+', default=None)
    p.add_argument('--years', nargs='+', default=None)
    p.add_argument('--replot', action='store_true',
                   help=f'Draw from {STATS_NAME} without touching the runs.')
    p.add_argument('--no-notes', dest='notes', action='store_false',
                   help='Leave the data notes off the figure.')
    p.add_argument('--figure', nargs='+', choices=FIGURES, default=None,
                   help='Figures to draw (default: all).')
    p.add_argument('--style', choices=('thesis', 'poster'), default='thesis',
                   help='thesis: 15.92 cm wide, 6.5 pt text. poster: printed '
                        'size, 18 pt text.')
    args = p.parse_args(argv)
    figures = args.figure or list(FIGURES)
    poster = args.style == 'poster'

    stats = None
    if {'value_ratio_iqr', 'saving_and_ratio'} & set(figures):
        if args.replot:
            stats = pd.read_csv(args.output_dir / STATS_NAME, dtype=str)
            stats[['battery_kwh', *STAT_COLUMNS]] = stats[
                ['battery_kwh', *STAT_COLUMNS]].apply(pd.to_numeric)
        else:
            stats = compute_stats(args)
        clash = overlaps(stats)
        if clash:
            print(f'  NOTE: household and wholesale IQRs overlap at {clash} '
                  f'kWh.')
        else:
            print('  Household and wholesale IQRs do not overlap at any '
                  'capacity.')
        print(stats.round(2).to_string(index=False))

    if 'value_ratio_iqr' in figures:
        matplotlib.rcdefaults()
        ts.apply_style()
        # The poster figure carries no data note: the caption does.
        make_figure(stats, args.output_dir,
                    _note(stats) if args.notes and not poster else None,
                    poster)

    if 'saving_and_ratio' in figures:
        curves = pd.read_csv(args.output_dir / CURVES_NAME)
        hh = curves[(curves.metric == HOUSEHOLD['metric'])
                    & curves.battery_kwh.isin([20])]
        print(hh.round(0).to_string(index=False))
        matplotlib.rcdefaults()
        make_saving_and_ratio_figure(curves, stats, args.output_dir)

    if 'event_exposure' in figures:
        df = write_exposure_csv(args.output_dir)
        matplotlib.rcdefaults()
        ts.apply_style()
        make_event_exposure_figure(df, args.output_dir, poster)

    if 'retailer_margin_spread' in figures:
        mstats = margin_stats(args.data_root / MARGIN_CSV)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        mstats.to_csv(args.output_dir / (MARGIN_NAME + '_stats.csv'),
                      index=False)
        print(mstats.round(2).to_string(index=False))
        matplotlib.rcdefaults()
        ts.apply_style()
        make_margin_spread_figure(mstats, args.output_dir, poster)
    return 0


if __name__ == '__main__':
    sys.exit(main())
