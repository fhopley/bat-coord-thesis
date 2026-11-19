"""Charts for the cohort VCR results.

Consumes the two things run_cohort_attribution.py writes:
  - cohort_value_attribution_summary.csv  (one row per site)
  - {site_id}_daily_value_attribution.csv (one row per site-day)

and produces five charts, in the order they are most useful for the
argument rather than the order they were designed:

  B  band contribution     -- WHERE each site's VCR comes from, by day size
  C  per-site VCR           -- each site with its leave-one-out fragility
  A  capture by band        -- the transferable capture-rate relationship
  D  concentration          -- what the opportunity is concentrated ON
  E  VCR vs battery size    -- capacity, within comparable exposure

TWO RULES THAT GOVERN ALL OF THESE
----------------------------------
1. Every line here is POOLED, never regressed, and never a mean of ratios.
   Within any group the figure is sum(captured)/sum(headroom). A mean of
   per-day or per-site ratios silently overweights whatever had the smallest
   denominator -- the same trap run_cohort_attribution documents for cohort
   VCR and sweep_c_deg documents for pooling c_deg.

2. Nothing is compared across sites on a RANK basis. Site windows run from
   31 to 274 days, so "top 10 days" means 3.6% of one site's period and 32%
   of another's. All grouping is by ABSOLUTE daily headroom per kWh of
   usable battery, which means the same thing everywhere.

WHAT THE DATA WILL NOT SUPPORT
------------------------------
Three calendar dates carry 45% of all headroom in this cohort, and they are
market-wide events that hit every site recording at the time. Three sites
were not recording then and have no exposure at all. So:

  - per-day VCR is not plotted. It ranges over +/-2 for reasons unrelated to
    coordination (starting SOC, where midnight fell relative to an event),
    and an earlier scatter of 792 site-days was unreadable.
  - the interval on the top band comes from a bootstrap over DATES, not
    days, because the site-days in it are not independent observations.
  - no line is fitted to nine sites, or to the six with event exposure.
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from energy_masters_pipeline.scripts.analyse_daily_value_gaps import vcr_by_opportunity

DEFAULT_SUMMARY_CSV = Path('data/processed/energy_masters/cohort_value_attribution_summary.csv')

# A site is flagged as having too little to say when EITHER its window is too
# short or its total opportunity too small. VCR is captured/headroom, so a
# site with nothing to capture produces a ratio that is noise. Flagged sites
# are drawn hollow and excluded from pooled figures, never silently dropped.
MIN_SITE_DAYS = 45
MIN_SITE_HEADROOM_DOLLARS = 50.0

# Bands of DAILY headroom per kWh of usable battery, doubling. Absolute bands
# rather than per-site rank tiers ("top 10 days"): with windows from 31 to 274
# days, a rank tier means something different at every site, while $/kWh/day
# means the same thing everywhere. Per kWh rather than raw dollars so a 7 kWh
# and a 21 kWh battery are compared on the opportunity each actually faced.
HEADROOM_BANDS_PER_KWH = (0.05, 0.10, 0.20, 0.40, 0.80, np.inf)

# How many of the cohort's largest calendar dates count as "the events".
N_MARKET_EVENTS = 3

# Range the pooled headline VCR moves over across the plausible
# c_deg_per_kwh range, from the sensitivity sweep in sweep_c_deg.py. Shown as
# a band around the pooled line on chart E.
#
# NOT derived from the data loaded here -- it is the result of re-running the
# whole cohort at several c_deg values, which this script does not do. So it
# is a hand-maintained constant and WILL go stale if the cohort run changes.
# _draw_pooled_vcr_band warns when the computed pooled value falls outside
# it, which is the cheapest available check that the two are still in step.
# Set to None to omit the band.
POOLED_VCR_SENSITIVITY = (0.420, 0.433)

# Sites removed by manual review, as opposed to sites failing the automatic
# MIN_SITE_* rules above. Kept separate and named so the two reasons for a
# site being absent stay distinguishable: this list is a judgement, the rules
# are a filter.
EXCLUDE_SITES = ()

# Capacity groups for chart F. Bands rather than a continuous fit because
# there are far too few sites for a slope, and because the interesting
# question is ordinal: does a bigger battery capture more of the same
# relative opportunity.
CAPACITY_GROUPS = ((0.0, 10.0, '<10 kWh'),
                   (10.0, 15.0, '10-15 kWh'),
                   (15.0, np.inf, '15+ kWh'))
DEFAULT_DAILY_GLOB = '*_daily_value_attribution.csv'

# Pooled-line bins DOUBLE in width. Equal-count quantile bins were tried
# first and rejected: they lump everything above ~$4 into one point, and on
# this cohort that single bin holds 61% of all headroom, hiding the whole
# top end. Doubling bands give resolution where the money is and read
# naturally ("days worth $4 to $8"). Bins with fewer than MIN_BIN_DAYS days
# are dropped rather than plotted as a one-day "trend".
MIN_BIN_DAYS = 8

TIER_COLOURS = {'top 10 headroom days': '#1B6B8C',
                'next 40 days': '#5BA3C4',
                'remaining days': '#C4D8E2'}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_summary(path: Path = DEFAULT_SUMMARY_CSV,
                 exclude=EXCLUDE_SITES) -> pd.DataFrame:
    df = pd.read_csv(path).set_index('site_id')
    return df.drop(index=[s for s in exclude if s in df.index])


def load_daily(daily_dir: Path = Path('data/processed/energy_masters'),
               pattern: str = DEFAULT_DAILY_GLOB,
               exclude=EXCLUDE_SITES) -> pd.DataFrame:
    """All sites' per-day tables concatenated, with a site_id column."""
    files = sorted(glob.glob(str(Path(daily_dir) / pattern)))
    if not files:
        raise FileNotFoundError(
            f"no files matching {pattern!r} in {Path(daily_dir).resolve()}; "
            "run run_cohort_attribution.py first"
        )
    frames = []
    for f in files:
        site_id = Path(f).name.split('_')[0]
        if site_id in exclude:
            continue
        df = pd.read_csv(f)
        df['site_id'] = site_id
        frames.append(df)
    if not frames:
        raise ValueError('every site was excluded')
    return pd.concat(frames, ignore_index=True)


def market_event_dates(daily: pd.DataFrame, n: int = N_MARKET_EVENTS) -> list:
    """The calendar dates carrying the most headroom across the whole cohort.

    These are market-wide events, not per-site good days: on this cohort the
    top three dates carry 45% of all headroom and hit every site that was
    recording at the time. That makes the effective sample size behind any
    spike-day figure the number of EVENTS, not the number of site-days.
    """
    by_date = (daily.assign(_d=pd.to_datetime(daily['day'].astype(str).str[:10]))
                    .groupby('_d')['gap_lp_vs_sc'].sum())
    return list(by_date.nlargest(n).index)


def site_flags(daily: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Per-site exposure to the market events, and whether the site is usable.

    Exposure matters because a site whose window missed the events is not
    measuring the same thing as one that caught them, even though both
    produce a number called VCR.
    """
    events = market_event_dates(daily)
    d = daily.assign(_d=pd.to_datetime(daily['day'].astype(str).str[:10]))
    rows = {}
    for site_id, g in d.groupby('site_id'):
        pos = g.loc[g['gap_lp_vs_sc'] > 0, 'gap_lp_vs_sc'].sum()
        on_events = g.loc[g['_d'].isin(events), 'gap_lp_vs_sc'].sum()
        n_days = len(g)
        headroom = float(summary.loc[site_id, 'gap_lp_vs_sc']) if site_id in summary.index else pos
        rows[site_id] = {
            'n_days': n_days,
            'total_headroom': headroom,
            'event_headroom_share': (on_events / pos) if pos > 0 else np.nan,
            'spike_exposed': bool(on_events > 0),
            'usable': bool(n_days >= MIN_SITE_DAYS
                           and headroom >= MIN_SITE_HEADROOM_DOLLARS),
        }
    return pd.DataFrame(rows).T


def _flag_note(flags: pd.DataFrame) -> str:
    bad = list(flags.index[~flags['usable'].astype(bool)])
    if not bad:
        return ''
    return (f"hollow markers = insufficient data ({', '.join(bad)}): "
            f"under {MIN_SITE_DAYS} days or ${MIN_SITE_HEADROOM_DOLLARS:.0f} headroom, "
            "excluded from pooled figures")


def _site_colours(sites) -> dict:
    cmap = plt.get_cmap('tab10' if len(sites) <= 10 else 'tab20')
    return {s: cmap(i % cmap.N) for i, s in enumerate(sorted(sites))}


def _site_label(summary: pd.DataFrame, site_id: str) -> str:
    """Site id with its usable capacity, so capacity is readable off any legend."""
    if site_id in summary.index and 'capacity_kwh' in summary.columns:
        return f"{site_id} ({summary.loc[site_id, 'capacity_kwh']:.1f} kWh)"
    return site_id


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


# ---------------------------------------------------------------------------
# Pooled binning -- the shared engine for every line in this module
# ---------------------------------------------------------------------------

def doubling_bands(lo: float, hi: float) -> np.ndarray:
    """Bin edges doubling from lo until they cover hi."""
    edges = [lo]
    while edges[-1] < hi:
        edges.append(edges[-1] * 2)
    return np.array(edges)


def pooled_by_bin(daily: pd.DataFrame, x_col: str,
                  min_days: int = MIN_BIN_DAYS) -> pd.DataFrame:
    """Pooled VCR within doubling-width bins of x_col.

    Returns one row per bin: the bin's median x, its pooled VCR
    (sum captured / sum headroom), the day count, and the headroom behind it.

    Pooled, not averaged: a bin holding one $200 day and fifty $1 days should
    be dominated by the $200 day, because that is where the money is. The
    same reasoning as the cohort VCR in run_cohort_attribution and the c_deg
    pooling in sweep_c_deg.
    """
    d = daily.dropna(subset=[x_col])
    d = d[d[x_col] > 0]
    if len(d) < min_days:
        return pd.DataFrame(columns=['x', 'vcr', 'n_days', 'headroom', 'lo', 'hi'])
    edges = doubling_bands(float(d[x_col].min()), float(d[x_col].max()))
    # Trailing bands are merged upward into a single open-ended top band.
    # Without this the largest days -- 41% of all headroom on this cohort --
    # fall into a run of bands holding two or three days each and are dropped
    # by the min_days rule, silently removing the part of the curve that
    # matters most.
    while len(edges) > 2 and (d[x_col] >= edges[-2]).sum() < min_days:
        edges = np.delete(edges, -2)

    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        last = (i == len(edges) - 2)
        g = d[(d[x_col] >= lo) & ((d[x_col] <= hi) if last else (d[x_col] < hi))]
        headroom = g['gap_lp_vs_sc'].sum()
        if len(g) < min_days or headroom <= 0:
            continue
        rows.append({'x': g[x_col].median(),
                     'vcr': g['gap_actual_vs_sc'].sum() / headroom,
                     'n_days': len(g), 'headroom': headroom,
                     'lo': lo, 'hi': (np.inf if last else hi)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Chart B -- tier contribution
# ---------------------------------------------------------------------------

def chart_band_contribution(daily: pd.DataFrame, summary: pd.DataFrame,
                            out_dir: Optional[Path] = None):
    """Headline VCR decomposed by ABSOLUTE day size, not by rank.

    An earlier version split each site into its top 10 days, next 40, and the
    rest. That was not comparable across sites: with windows from 31 to 274
    days, "top 10 days" is 3.6% of one site's period and 32% of another's, so
    the same segment name described completely different things.

    Absolute bands of $/kWh/day fix that, and make the real structure
    visible: three sites have NOTHING in the upper bands, because their
    windows fell entirely after the January price events. Those sites are not
    capturing spike value badly, they never had the opportunity, and the rank
    version silently relabelled their ordinary days as "top 10".

    Segments are band captured / site total headroom, so they stack exactly to
    the headline VCR regardless of any band's own ratio being undefined.
    """
    flags = site_flags(daily, summary)
    d = daily.copy()
    d['cap'] = d['site_id'].map(summary['capacity_kwh'])
    d['hpk'] = d['gap_lp_vs_sc'] / d['cap']
    edges = list(HEADROOM_BANDS_PER_KWH)
    band_names = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        band_names.append(f"${lo:g}+" if not np.isfinite(hi) else f"${lo:g}-{hi:g}")

    rows = {}
    for site_id, g in d.groupby('site_id'):
        total = g['gap_lp_vs_sc'].sum()
        if abs(total) < 1e-9:
            continue
        vals = {}
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            sel = g[(g['hpk'] >= lo) & (g['hpk'] < hi)]
            vals[band_names[i]] = sel['gap_actual_vs_sc'].sum() / total
        # Days below the smallest band still carry value; keep them so the
        # segments continue to stack exactly to the headline.
        sel = g[g['hpk'] < edges[0]]
        vals[f'<${edges[0]:g}'] = sel['gap_actual_vs_sc'].sum() / total
        rows[site_id] = vals

    order = [f'<${edges[0]:g}'] + band_names
    contrib = pd.DataFrame(rows).T.reindex(columns=order)
    contrib = contrib.reindex(contrib.sum(axis=1).sort_values(ascending=False).index)

    cmap = plt.get_cmap('YlGnBu')
    colours = {b: cmap(0.15 + 0.75 * i / max(1, len(order) - 1))
               for i, b in enumerate(order)}

    fig, ax = plt.subplots(figsize=(12, 6.5))
    x = np.arange(len(contrib))
    bottom_pos = np.zeros(len(contrib)); bottom_neg = np.zeros(len(contrib))
    for band in order:
        vals = contrib[band].fillna(0.0).to_numpy()
        base = np.where(vals >= 0, bottom_pos, bottom_neg)
        ax.bar(x, vals, bottom=base, color=colours[band], edgecolor='white',
               linewidth=0.6, label=f'{band} /kWh/day')
        bottom_pos = bottom_pos + np.where(vals >= 0, vals, 0)
        bottom_neg = bottom_neg + np.where(vals < 0, vals, 0)

    ax.plot(x, contrib.sum(axis=1), 'o', color='black', markersize=7, zorder=5,
            label='headline VCR')
    ax.axhline(0, color='black', linewidth=0.8)
    labels = []
    for site_id in contrib.index:
        lab = _site_label(summary, site_id)
        if not bool(flags.loc[site_id, 'usable']):
            lab += '\n(insufficient data)'
        elif not bool(flags.loc[site_id, 'spike_exposed']):
            lab += '\n(no event exposure)'
        labels.append(lab)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('contribution to headline VCR')
    ax.set_title('Value Capture Ratio contribution by day headroom per kWh of usable battery\n'
                 '(Segments are band captured / total headroom and stack to the headline)',
                 fontsize=11)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(top=0.6)
    fig.tight_layout()
    _finish(fig, 'B_band_contribution', out_dir)
    return contrib


# ---------------------------------------------------------------------------
# Chart C -- per-site VCR with fragility
# ---------------------------------------------------------------------------

def chart_site_vcr(summary: pd.DataFrame, out_dir: Optional[Path] = None,
                   flags: Optional[pd.DataFrame] = None):
    """Per-site headline VCR against total headroom, with leave-one-out bars.

    The error bars are the point of the chart. They are not sampling error:
    they are the range the headline moves over when any ONE day is removed.
    A site whose bar spans 0.3 has a headline that depends on whether a
    single extreme price event fell inside its window.

    Total headroom on the x axis puts low-opportunity sites at the left,
    where a reader can see that a poor VCR there means "nothing to capture"
    rather than "captured badly".
    """
    d = summary.dropna(subset=['vcr_volumetric']).copy()
    colours = _site_colours(d.index)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    usable_mask = None if flags is None else flags['usable'].astype(bool)

    lo = (d['vcr_volumetric'] - d['vcr_loo_min']).clip(lower=0)
    hi = (d['vcr_loo_max'] - d['vcr_volumetric']).clip(lower=0)
    for site_id in d.index:
        ok = True if usable_mask is None else bool(usable_mask.get(site_id, True))
        ax.errorbar(d.loc[site_id, 'gap_lp_vs_sc'], d.loc[site_id, 'vcr_volumetric'],
                    yerr=[[lo[site_id]], [hi[site_id]]],
                    fmt='o', markersize=10, capsize=4, elinewidth=1.4,
                    color=colours[site_id],
                    markerfacecolor=colours[site_id] if ok else 'none',
                    markeredgewidth=1.6,
                    label=_site_label(summary, site_id) + ('' if ok else ' *'))

    keep = d if usable_mask is None else d[usable_mask.reindex(d.index).fillna(True)]
    pooled = keep['gap_actual_vs_sc'].sum() / keep['gap_lp_vs_sc'].sum()
    ax.axhline(pooled, color='black', linestyle='--', linewidth=1.6,
               label=f'pooled VCR, usable sites = {pooled:.3f}')
    ax.axhline(0, color='grey', linewidth=0.8)
    ax.set_xscale('log')
    # Default log ticks label only the decades, which on a 30-to-800 range
    # means a single labelled value. Explicit ticks make the axis readable.
    ticks = [30, 50, 100, 200, 300, 500, 800]
    lo_x, hi_x = d['gap_lp_vs_sc'].min(), d['gap_lp_vs_sc'].max()
    ticks = [t for t in ticks if lo_x / 1.6 <= t <= hi_x * 1.6]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'${t:,}' for t in ticks])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel('total headroom over the site\'s period ($, log scale)')
    ax.set_ylabel('headline VCR')
    ax.set_title('Per-site VCR, with the range it moves over when any single day is removed',
                 fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _finish(fig, 'C_site_vcr_with_fragility', out_dir)


# ---------------------------------------------------------------------------
# Band geometry, shared by charts A and F
# ---------------------------------------------------------------------------

def _band_edges(daily: pd.DataFrame, summary: pd.DataFrame) -> np.ndarray:
    """Band edges with the open top band drawn as one further doubling.

    The top band is defined as open-ended and its largest day is around
    $17/kWh, but closing it there would make that single step wider on a log
    axis than every other band put together, giving the most sparsely
    populated bucket the most visual weight. Every band is a doubling, so
    closing the last at one more doubling keeps the steps equal width and
    lets the axis label carry the "and above" instead.
    """
    edges = np.array(HEADROOM_BANDS_PER_KWH, dtype=float)
    edges[-1] = edges[-2] * 2
    return edges


def _draw_band_guides(ax, edges: np.ndarray) -> None:
    """Edge rules and ticks so the buckets are readable.

    On a log axis the default ticks fall on decades, which says nothing about
    where one bucket stops and the next starts. Putting the ticks ON the
    edges, with a rule at each, means the reader can see exactly what range
    each plotted value covers.

    Deliberately NO band shading: shading already means the 95% interval on
    these charts, and using one visual channel for two unrelated things makes
    both harder to read.
    """
    # Scale FIRST. set_xscale installs a LogLocator and LogFormatter, so any
    # fixed ticks applied before it are silently thrown away -- which is what
    # left these charts showing only 10^-1 and 10^0.
    ax.set_xscale('log')
    for e in edges[1:-1]:
        ax.axvline(e, color='#999999', linewidth=0.7, linestyle=':', zorder=1)
    ax.set_xticks(edges[:-1])
    ax.set_xticklabels([f'${e:g}' for e in edges[:-1]])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlim(edges[0] * 0.92, edges[-1] * 1.02)
    # The final bucket has no upper edge to label, so it is named in place.
    ax.annotate(f'${edges[-2]:g}+  (no upper limit)',
                xy=(np.sqrt(edges[-2] * edges[-1]), 0.015),
                xycoords=('data', 'axes fraction'), ha='center',
                fontsize=7.5, color='#666666')


def _staircase(ax, edges, band_lo, values, lo95, hi95, colour, style, label, zorder):
    """Draw band values as a staircase spanning each band's real width.

    Plotting a band's value as a point at the band's geometric centre and
    joining the points implies the rate varies continuously and is known at
    those particular x positions. It is not: it is one number that applies
    across the whole bucket. A staircase says that, and makes the bucket
    boundaries readable without a separate legend.
    """
    idx = {round(float(b), 6): i for i, b in enumerate(edges[:-1])}
    xs, ys, los, his = [], [], [], []
    for b, v, lo, hi in zip(band_lo, values, lo95, hi95):
        i = idx.get(round(float(b), 6))
        if i is None:
            continue
        xs += [edges[i], edges[i + 1]]
        ys += [v, v]
        los += [lo, lo]
        his += [hi, hi]
    if not xs:
        return
    ax.fill_between(xs, los, his, color=colour, alpha=0.13, zorder=zorder,
                    linewidth=0)
    ax.plot(xs, ys, style, color=colour, linewidth=2.8, zorder=zorder + 1,
            label=label, solid_capstyle='butt')


def _band_mid(edges: np.ndarray, band_lo) -> np.ndarray:
    """Geometric centre of the band a value belongs to, using DRAWN edges.

    Markers for individual observations sit at band centres while the pooled
    value is drawn as a step across the whole band, so both have to agree
    about where the band is.
    """
    idx = {round(float(b), 6): i for i, b in enumerate(edges[:-1])}
    out = []
    for b in band_lo:
        i = idx.get(round(float(b), 6))
        out.append(np.nan if i is None else np.sqrt(edges[i] * edges[i + 1]))
    return np.array(out)


# ---------------------------------------------------------------------------
# Chart A -- day-level scatter
# ---------------------------------------------------------------------------

def band_cells(daily: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Pooled VCR for every (site, headroom band) cell.

    Aggregating days into cells is the point. A per-DAY VCR is captured over
    headroom for a single day, and it swings from -2 to +2 because a day's
    capture depends on the SOC it started with, on where an arbitrary midnight
    boundary fell relative to a price event, and on whether the battery
    happened to be full. None of that is coordination quality; it is noise
    that averages out over days but never over a single one. Pooling within a
    cell removes it while keeping the two comparisons that matter: across
    bands, and across sites within a band.
    """
    d = daily.copy()
    d['cap'] = d['site_id'].map(summary['capacity_kwh'])
    d['hpk'] = d['gap_lp_vs_sc'] / d['cap']
    d = d[d['gap_lp_vs_sc'] > 0]
    edges = list(HEADROOM_BANDS_PER_KWH)
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        g_all = d[(d['hpk'] >= lo) & (d['hpk'] < hi)]
        for site_id, g in g_all.groupby('site_id'):
            headroom = g['gap_lp_vs_sc'].sum()
            if headroom <= 0:
                continue
            rows.append({'site_id': site_id, 'band_lo': lo, 'band_hi': hi,
                         'band_mid': np.sqrt(lo * (hi if np.isfinite(hi) else lo * 4)),
                         'vcr': g['gap_actual_vs_sc'].sum() / headroom,
                         'n_days': len(g), 'headroom': headroom})
    return pd.DataFrame(rows)


def bootstrap_band_vcr(daily: pd.DataFrame, summary: pd.DataFrame,
                       sites: list, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Pooled VCR per band with a CLUSTER bootstrap over calendar dates.

    Resampling days independently would badly understate the uncertainty at
    the top: the largest days are a handful of market-wide events, and every
    exposed site experiences the same one simultaneously. Resampling DATES
    keeps those clusters intact, so the interval reflects "we saw three
    events" rather than "we saw twenty-one site-days".
    """
    rng = np.random.default_rng(seed)
    d = daily[daily['site_id'].isin(sites)].copy()
    d['cap'] = d['site_id'].map(summary['capacity_kwh'])
    d['hpk'] = d['gap_lp_vs_sc'] / d['cap']
    d['_d'] = pd.to_datetime(d['day'].astype(str).str[:10])
    d = d[d['gap_lp_vs_sc'] > 0]

    edges = list(HEADROOM_BANDS_PER_KWH)
    dates = np.sort(d['_d'].unique())
    n_dates = len(dates)
    out = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        band = d[(d['hpk'] >= lo) & (d['hpk'] < hi)]
        if not len(band):
            continue
        # Per-date totals, then resample DATES via multinomial weights. This
        # is the same cluster bootstrap as drawing dates one at a time, but
        # expressed as a matrix product so 2000 replicates are instant.
        g = band.groupby('_d')[['gap_actual_vs_sc', 'gap_lp_vs_sc']].sum()
        g = g.reindex(dates, fill_value=0.0)
        num, den = g['gap_actual_vs_sc'].to_numpy(), g['gap_lp_vs_sc'].to_numpy()
        w = rng.multinomial(n_dates, np.full(n_dates, 1.0 / n_dates), size=n_boot)
        nums, dens = w @ num, w @ den
        ok = dens > 0
        draws = nums[ok] / dens[ok]
        out.append({'band_lo': lo, 'band_hi': hi,
                    'band_mid': np.sqrt(lo * (hi if np.isfinite(hi) else lo * 4)),
                    'vcr': num.sum() / den.sum(), 'n_days': len(band),
                    'headroom': band['gap_lp_vs_sc'].sum(),
                    'n_dates': band['_d'].nunique(),
                    'lo95': np.percentile(draws, 2.5) if len(draws) else np.nan,
                    'hi95': np.percentile(draws, 97.5) if len(draws) else np.nan})
    return pd.DataFrame(out)


def chart_capture_by_band(daily: pd.DataFrame, summary: pd.DataFrame,
                          out_dir: Optional[Path] = None):
    """Capture rate against opportunity size -- the transferable relationship.

    Replaces an earlier per-day scatter. That chart plotted 792 individual
    site-days and was unreadable: per-day VCR ranges over +/-2 for reasons
    that have nothing to do with coordination, and the noise swamped the
    signal at every opportunity size except the very largest.

    Here each marker is one site in one band, pooled over that site's days in
    it, sized by the money behind it. The band gives the cohort-pooled rate
    with a date-cluster 95% interval.

    This is the object to carry to other LP modelling: apply the band rates
    to the target dataset's own distribution of daily headroom per kWh. The
    interval on the top band is wide for an unavoidable reason, and the
    number of independent dates behind each band is printed so the reader can
    see it.
    """
    flags = site_flags(daily, summary)
    usable = list(flags.index[flags['usable'].astype(bool)])
    cells = band_cells(daily, summary)
    boot = bootstrap_band_vcr(daily, summary, usable)
    colours = _site_colours(daily['site_id'].unique())
    edges = _band_edges(daily, summary)

    fig, ax = plt.subplots(figsize=(12, 7))
    _draw_band_guides(ax, edges)

    for site_id, g in cells.groupby('site_id'):
        ok = bool(flags.loc[site_id, 'usable'])
        mid = _band_mid(edges, g['band_lo'])
        ax.plot(mid, g['vcr'], '-', color=colours[site_id],
                alpha=0.35, linewidth=1.0, zorder=2)
        ax.scatter(mid, g['vcr'],
                   s=np.clip(g['headroom'] * 1.2, 18, 320),
                   facecolor=colours[site_id] if ok else 'none',
                   edgecolor=colours[site_id], linewidths=1.4,
                   alpha=0.85 if ok else 1.0, zorder=3,
                   label=_site_label(summary, site_id))

    if len(boot):
        _staircase(ax, edges, boot['band_lo'], boot['vcr'],
                   boot['lo95'], boot['hi95'], 'black', '-',
                   'Pooled, usable sites (95% interval bootstrapped over dates)', 4)
        for _, r in boot.iterrows():
            mid = float(_band_mid(edges, [r['band_lo']])[0])
            ax.annotate(f"{r['n_days']:.0f} days\n{r['n_dates']:.0f} dates",
                        xy=(mid, r['vcr']), xytext=(0, -34),
                        textcoords='offset points', ha='center', fontsize=7,
                        color='#444444')

    ax.axhline(0, color='grey', linewidth=0.8)
    # NB: no set_xscale here -- _draw_band_guides owns the scale, and setting
    # it again would wipe the edge ticks.
    ax.set_xlabel('Daily Headroom per kWh of Usable Battery ($/kWh/day, log scale)')
    ax.set_ylabel('Pooled VCR within the band')
    note = _flag_note(flags)
    ax.set_title('Value Capture Ratio against Daily Headroom per kWh\n'
                 '(each step spans its bucket; marker area proportional to '
                 'value captured)' + (f'; {note}' if note else ''), fontsize=11)
    ax.legend(fontsize=8, ncol=2, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    _finish(fig, 'A_capture_by_band', out_dir)
    return boot, cells


def capacity_group(capacity_kwh: float) -> str:
    for lo, hi, label in CAPACITY_GROUPS:
        if lo <= capacity_kwh < hi:
            return label
    return CAPACITY_GROUPS[-1][2]


def chart_capture_by_band_and_capacity(daily: pd.DataFrame, summary: pd.DataFrame,
                                       out_dir: Optional[Path] = None):
    """Chart A, split by battery capacity: the object to actually apply.

    Chart A gives one capture rate per opportunity band, which can be applied
    to another dataset's distribution of daily headroom per kWh. What it
    cannot do is distinguish batteries, and the per-site lines behind it sit
    visibly higher for larger ones. This splits the same pooled-by-band
    calculation three ways so a capacity adjustment can be carried across
    too.

    Note what the x axis has already removed. Headroom is expressed PER kWh
    of usable battery, so the mechanical advantage of a larger battery seeing
    more absolute opportunity is already divided out. Any remaining
    separation between the curves is a difference in how much of the SAME
    relative opportunity each group converts, which is the question worth
    asking.

    WHAT THIS CANNOT SEPARATE. Capacity is confounded with period and with
    event exposure in this cohort, and with only a handful of sites there is
    no way to break that. In particular the 15+ group is a single site whose
    window contains no market event at all, so its curve stops before the top
    bands and its level cannot be distinguished from a period effect. The
    per-point annotations give sites and dates behind every marker precisely
    so this is visible rather than buried.
    """
    flags = site_flags(daily, summary)
    usable = set(flags.index[flags['usable'].astype(bool)])
    groups = [g for _, _, g in CAPACITY_GROUPS]
    cmap = plt.get_cmap('viridis')
    colours = {g: cmap(0.12 + 0.72 * i / max(1, len(groups) - 1))
               for i, g in enumerate(groups)}
    grp_of = {site_id: capacity_group(float(cap))
              for site_id, cap in summary['capacity_kwh'].items()}
    edges = _band_edges(daily, summary)

    fig, ax = plt.subplots(figsize=(12.5, 7.5))
    _draw_band_guides(ax, edges)

    overall = bootstrap_band_vcr(daily, summary, sorted(usable))
    if len(overall):
        _staircase(ax, edges, overall['band_lo'], overall['vcr'],
                   overall['vcr'], overall['vcr'], '#999999', ':',
                   'All usable sites pooled (chart A)', 2)

    summary_rows = []
    for group in groups:
        sites = sorted(s for s in usable if grp_of.get(s) == group)
        if not sites:
            continue
        exposed = [s for s in sites if bool(flags.loc[s, 'spike_exposed'])]
        boot = bootstrap_band_vcr(daily, summary, sites)
        if not len(boot):
            continue
        label = f'{group}  (n={len(sites)} site{"s" if len(sites) > 1 else ""}'
        label += ')' if exposed else ', no event exposure)'
        _staircase(ax, edges, boot['band_lo'], boot['vcr'],
                   boot['lo95'], boot['hi95'], colours[group],
                   '-' if len(sites) > 1 else '--', label, 4)
        for _, r in boot.iterrows():
            i = int(np.argmin(np.abs(edges[:-1] - r['band_lo'])))
            mid = np.sqrt(edges[i] * edges[i + 1])
            ax.annotate(f"{r['n_days']:.0f}d / {r['n_dates']:.0f}dt",
                        xy=(mid, r['vcr']), xytext=(0, 7),
                        textcoords='offset points', ha='center', fontsize=6.5,
                        color=colours[group])
            summary_rows.append({'group': group, 'band_lo': r['band_lo'],
                                 'band_hi': r['band_hi'], 'vcr': r['vcr'],
                                 'lo95': r['lo95'], 'hi95': r['hi95'],
                                 'n_days': r['n_days'], 'n_dates': r['n_dates'],
                                 'n_sites': len(sites), 'headroom': r['headroom']})

    ax.axhline(0, color='grey', linewidth=0.8)
    # NB: no set_xscale here -- _draw_band_guides owns the scale, and setting
    # it again would wipe the edge ticks.
    ax.set_xlabel('Daily Headroom per kWh of Usable Battery ($/kWh/day, log scale)')
    ax.set_ylabel('Pooled VCR within the band')
    ax.set_title('Value Capture Ratio against Daily Headroom per kWh, '
                 'by Battery Capacity\n'
                 '(each step spans its bucket; shading is the 95% interval, '
                 'bootstrapped over dates)', fontsize=11)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    _finish(fig, 'F_capture_by_band_and_capacity', out_dir)
    return pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# Chart D -- concentration
# ---------------------------------------------------------------------------

def chart_concentration(daily: pd.DataFrame, summary: pd.DataFrame,
                        out_dir: Optional[Path] = None):
    """How concentrated the opportunity is, and what it is concentrated ON.

    Left: days sorted by headroom descending and accumulated. The gap
    between the two pooled curves is the argument for reporting VCR
    conditioned on opportunity rather than as one number -- if capture
    tracked opportunity evenly the curves would coincide.

    Right: total headroom by calendar date across all sites. This panel
    exists because the left one is easy to misread. The concentration is not
    "each site has a few good days"; it is that a couple of MARKET-WIDE
    events dominate every site at once. The effective sample size behind the
    figure that drives cohort VCR is the number of those events, not the
    number of site-days, and that distinction governs how far any of this
    generalises.
    """
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    colours = _site_colours(daily['site_id'].unique())

    for site_id, g in daily.groupby('site_id'):
        g = g.sort_values('gap_lp_vs_sc', ascending=False)
        pos = g[g['gap_lp_vs_sc'] > 0]
        if not len(pos):
            continue
        share_days = np.arange(1, len(pos) + 1) / len(pos) * 100
        ax.plot(share_days, pos['gap_lp_vs_sc'].cumsum() / pos['gap_lp_vs_sc'].sum() * 100,
                color=colours[site_id], linewidth=1.0, alpha=0.65,
                label=_site_label(summary, site_id))

    pooled = daily.sort_values('gap_lp_vs_sc', ascending=False)
    pooled = pooled[pooled['gap_lp_vs_sc'] > 0]
    share_days = np.arange(1, len(pooled) + 1) / len(pooled) * 100
    ax.plot(share_days,
            pooled['gap_lp_vs_sc'].cumsum() / pooled['gap_lp_vs_sc'].sum() * 100,
            color='black', linewidth=2.8, zorder=6, label='pooled: headroom')
    ax.plot(share_days,
            pooled['gap_actual_vs_sc'].cumsum() / pooled['gap_actual_vs_sc'].sum() * 100,
            color='#C0392B', linewidth=2.8, zorder=6, linestyle='--',
            label='pooled: value actually captured')
    ax.plot([0, 100], [0, 100], color='grey', linewidth=0.9, linestyle=':',
            label='even distribution')
    ax.set_xlabel('cumulative share of days, largest opportunity first (%)')
    ax.set_ylabel('cumulative share of total (%)')
    ax.set_title('Opportunity is concentrated in a handful of days', fontsize=11)
    ax.legend(fontsize=8, ncol=2, loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 100); ax.set_ylim(0, 105)

    # Take the date directly off the label: the day index is already a
    # normalised local date, and a UTC conversion would shift every
    # +10:00 timestamp back one calendar day.
    by_date = (daily.assign(date=pd.to_datetime(daily['day'].astype(str).str[:10]))
                    .groupby('date')['gap_lp_vs_sc'].sum().sort_index())
    by_date = by_date[by_date > 0]
    ax2.bar(by_date.index, by_date.values, width=1.6, color='#1B6B8C')
    total = by_date.sum()
    for date, value in by_date.nlargest(3).items():
        ax2.annotate(f"{date}\n{100*value/total:.0f}% of all headroom",
                     xy=(date, value), xytext=(-10, -4), textcoords='offset points',
                     fontsize=8, ha='right', color='#333333')
    ax2.set_ylabel('total headroom across all sites ($)')
    ax2.set_title('...and those days are a few market-wide events,\n'
                  'not independent per-site opportunities', fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(axis='x', rotation=30)

    fig.tight_layout()
    _finish(fig, 'D_concentration', out_dir)
    return by_date


# ---------------------------------------------------------------------------
# Chart E -- VCR against battery size
# ---------------------------------------------------------------------------

def _draw_pooled_vcr_band(ax, pooled: float, sensitivity, colour='#1B6B8C'):
    """Pooled headline VCR line, with the c_deg sensitivity range around it.

    The line is computed from the loaded data; the band is not. Keeping them
    visually together is the point of the chart, but it means a stale
    constant would quietly misrepresent the result, so the mismatch is
    checked rather than assumed.
    """
    if sensitivity is not None:
        lo, hi = float(min(sensitivity)), float(max(sensitivity))
        if not (lo <= pooled <= hi):
            print(f"  WARNING: pooled headline VCR {pooled:.4f} falls outside the "
                  f"stated c_deg sensitivity range {lo:.3f} to {hi:.3f}. "
                  "POOLED_VCR_SENSITIVITY is probably stale -- re-run the "
                  "c_deg sweep or update the constant.")
        ax.axhspan(lo, hi, color='#000000', alpha=0.10, zorder=0)
        label_y = max(hi, pooled)
        note = (f'Pooled Headline VCR {pooled:.3f}\n'
                f'c_deg sensitivity {lo:.3f} to {hi:.3f}')
    else:
        label_y = pooled
        note = f'Pooled Headline VCR {pooled:.3f}'

    ax.axhline(pooled, color=colour, linestyle='--', linewidth=1.4, zorder=1)
    ax.annotate(note, xy=(0.995, label_y), xycoords=('axes fraction', 'data'),
                xytext=(0, 5), textcoords='offset points',
                ha='right', va='bottom', fontsize=8.5, color=colour,
                linespacing=1.4)


def chart_vcr_vs_capacity(summary: pd.DataFrame, daily: pd.DataFrame,
                          out_dir: Optional[Path] = None,
                          vcr_sensitivity=POOLED_VCR_SENSITIVITY):
    """VCR against battery capacity, split by exposure to the market events.

    The split is not cosmetic. Comparing a site whose window caught the
    January events against one whose window did not is comparing two
    different quantities, and mixing them was hiding the relationship: across
    all nine sites the top-10 VCR looked flat, because three of those sites'
    "top 10 days" were ordinary days. Within the exposed group alone the same
    figure correlates strongly with capacity.

    Still no fitted line. Six exposed sites, one capacity cluster near
    12 kWh and one near 7-10 kWh, is a suggestive contrast rather than an
    estimated slope.
    """
    flags = site_flags(daily, summary)
    d = summary.dropna(subset=['capacity_kwh', 'vcr_volumetric']).copy()
    d = d.join(flags[['spike_exposed', 'usable']])
    fig, ax = plt.subplots(figsize=(11, 6.5))

    for exposed, marker, tag in ((True, 'o', 'caught a market event'),
                                 (False, 's', 'no event in window')):
        g = d[d['spike_exposed'].astype(bool) == exposed]
        if not len(g):
            continue
        for series, colour, name in (('vcr_volumetric', '#1B6B8C', 'headline VCR'),
                                     ('vcr_top_10', '#C0392B', 'VCR, top 10 days')):
            if series not in g.columns:
                continue
            ok = g['usable'].astype(bool)
            ax.scatter(g.loc[ok, 'capacity_kwh'], g.loc[ok, series], marker=marker,
                       s=110, facecolor=colour, edgecolor=colour, linewidths=1.5,
                       label=f'{name} -- {tag}')
            ax.scatter(g.loc[~ok, 'capacity_kwh'], g.loc[~ok, series], marker=marker,
                       s=110, facecolor='none', edgecolor=colour, linewidths=1.5)

    for site_id, row in d.iterrows():
        ax.annotate(site_id, (row['capacity_kwh'], row['vcr_volumetric']),
                    textcoords='offset points', xytext=(7, -11), fontsize=8,
                    color='#555555')

    # Pooled headline VCR across the usable sites: summed numerator over
    # summed denominator, NOT the mean of the plotted points. The two differ
    # here because the sites carry very different amounts of headroom, and
    # the mean would give a 31-day site the same weight as a 274-day one.
    # Same figure as the reference line on chart C, by construction.
    keep = d[d['usable'].astype(bool)]
    pooled_headline = keep['gap_actual_vs_sc'].sum() / keep['gap_lp_vs_sc'].sum()
    _draw_pooled_vcr_band(ax, pooled_headline, vcr_sensitivity)

    exposed_ok = d[d['spike_exposed'].astype(bool) & d['usable'].astype(bool)]
    if len(exposed_ok) > 2:
        r_head = exposed_ok['capacity_kwh'].corr(exposed_ok['vcr_volumetric'])
        r_top = exposed_ok['capacity_kwh'].corr(exposed_ok['vcr_top_10'])
        sub = (f'event-exposed sites only (n={len(exposed_ok)}): '
               f'headline r={r_head:+.2f}, top-10 r={r_top:+.2f}')
    else:
        sub = ''

    ax.axhline(0, color='grey', linewidth=0.8)
    # The capacity buckets chart F splits on, drawn here so the two charts can
    # be read against each other and so it is obvious how few sites land in
    # each one.
    lo_x, hi_x = ax.get_xlim()
    for i, (b_lo, b_hi, b_label) in enumerate(CAPACITY_GROUPS):
        if i % 2 == 1:
            ax.axvspan(max(b_lo, lo_x), min(b_hi, hi_x), color='#000000',
                       alpha=0.035, zorder=0)
        if lo_x < b_lo < hi_x:
            ax.axvline(b_lo, color='#999999', linewidth=0.7, linestyle=':', zorder=1)
        centre = np.sqrt(max(b_lo, lo_x) * min(b_hi, hi_x))
        n_in = int(((d['capacity_kwh'] >= b_lo) & (d['capacity_kwh'] < b_hi)).sum())
        ax.annotate(f'{b_label}  (n={n_in})', xy=(centre, 1.0),
                    xycoords=('data', 'axes fraction'), xytext=(0, -12),
                    textcoords='offset points', ha='center', fontsize=7.5,
                    color='#666666')
    ax.set_xlim(lo_x, hi_x)
    ax.set_xlabel('Usable Battery Capacity (kWh)')
    ax.set_ylabel('VCR')
    ax.set_ylim(0.2, 0.7)
    ax.set_title('Value Capture Ratio (VCR) against Battery Capacity', fontsize=11)
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _finish(fig, 'E_vcr_vs_capacity', out_dir)


# ---------------------------------------------------------------------------

def make_all(summary_csv: Path = DEFAULT_SUMMARY_CSV,
             daily_dir: Path = Path('data/processed/energy_masters'),
             out_dir: Optional[Path] = Path('energy_masters_pipeline/charts'),
             exclude=EXCLUDE_SITES):
    summary = load_summary(summary_csv, exclude=exclude)
    daily = load_daily(daily_dir, exclude=exclude)
    if exclude:
        print(f"excluded by review: {', '.join(exclude)}")
    flags = site_flags(daily, summary)
    events = market_event_dates(daily)

    print(f"{len(summary)} site(s), {len(daily):,} site-days")
    print(f"dominant market dates: {', '.join(d.strftime('%Y-%m-%d') for d in events)}")
    print(flags.to_string())
    unusable = list(flags.index[~flags['usable'].astype(bool)])
    if unusable:
        print(f"excluded from pooled figures: {', '.join(unusable)}")
    print()

    chart_band_contribution(daily, summary, out_dir=out_dir)
    chart_site_vcr(summary, out_dir=out_dir, flags=flags)
    chart_capture_by_band(daily, summary, out_dir=out_dir)
    table = chart_capture_by_band_and_capacity(daily, summary, out_dir=out_dir)
    if out_dir is not None and len(table):
        path = Path(out_dir) / 'F_capture_rates.csv'
        table.to_csv(path, index=False)
        print(f"  wrote {path}   <- the rates to apply to other LP modelling")
    chart_concentration(daily, summary, out_dir=out_dir)
    chart_vcr_vs_capacity(summary, daily, out_dir=out_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--summary', type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument('--daily-dir', type=Path, default=Path('data/processed/energy_masters'),
                        help='Directory holding the {site}_daily_value_attribution.csv files.')
    parser.add_argument('--out-dir', type=Path, default=Path('energy_masters_pipeline/charts'),
                        help='Where to write PNGs. Pass --show to open them instead.')
    parser.add_argument('--exclude', type=str, default=','.join(EXCLUDE_SITES),
                        help='Comma-separated sites removed by manual review. '
                             f'Default: {",".join(EXCLUDE_SITES) or "none"}. '
                             'Pass an empty string to keep every site.')
    parser.add_argument('--show', action='store_true',
                        help='Display interactively instead of writing files.')
    args = parser.parse_args()
    exclude = tuple(x.strip() for x in args.exclude.split(',') if x.strip())
    make_all(summary_csv=args.summary, daily_dir=args.daily_dir,
             out_dir=None if args.show else args.out_dir, exclude=exclude)