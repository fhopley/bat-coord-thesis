"""
pv_usage_matrix.py

Results for subsets of the modelled site-year population, arranged as a
matrix of PV array size band against annual usage quantile.  One cell is the
set of site-years whose estimated PV array falls in that band and whose
annual gross load falls in that quantile; every reported variable is a
summary over that cell.

The run basis is Amber imperfect foresight achievable: the optimised series
is full_run_rolling_amber, the self-consumption series is
full_run_self_consumption_amber, and savings are measured against
full_run_solar_only_amber.  The registry, the run names and the savings
baseline are taken from financial_analysis rather than restated here, so a
change there carries through.  --basis selects a different run basis and
--plan a different offer where that is wanted.

Dimensions
----------
PV size band comes from financial_analysis: the tiered size_kw estimate
(trusted subarray rating, else inverter rating times the pooled DC:AC ratio,
else a per-state regression on annual generation), assigned to the bands in
--bands-file.  Those are the matrix's own bands, not the weighting bands in
pv_size_bands.yaml, whose four entries are tied one-for-one to the per-state
weights beside them; the default bands file adds an open band below the
weighting bands' 4.5 kW floor and opens the top band, so the matrix covers
the whole modelled population and its margins are the full base.

Usage quantile is the site-year's annual gross household load, cut at the
quantiles of the matrix population itself, so the bins hold equal numbers of
site-years by construction and the cut points are reported with the tables.
--usage-quantiles sets how many bins and --usage-cuts gives explicit cut
points instead.  Load comes from
data/raw/edp/{year}_site_annual_totals.csv, the per-interval sum written by
edp_annual_totals.py, which is also the source of annual PV generation so
that the two are measured over the same intervals.  --usage-source metadata
instead derives load from avg_daily_gross_load_wh in the EDP site metadata;
the two agree to within a few kWh for most
site-years and by more than 5% for 22 of 1,436.

Equal-count bins are unequal in width, because the load distribution has a
long upper tail: over 2021-2025 the quartiles span 1.4x in load for Q2 and Q3
but 2.8x for Q4.  Table 1 reports each cell's median load and load range
beside the mean so the width is visible, and --max-usage trims the tail where
its influence needs testing.  The sites whose load is not residential at all
are already out: site_exclusions.EXCLUDED_SITES removes them from every
analysis, financial_analysis and this script alike.

Weighting
---------
Cell means are unweighted means over site-years.  The PV size-band weights in
pv_size_bands.yaml cannot change a cell mean, because every site-year in a
cell sits in the same band and so carries the same weight.  State weighting
is not applied: the cells are too small to carry it, and table 6 reports the
state composition that this leaves confounded.

Capacity for a $0 bill
----------------------
Two readings, because they answer different questions and diverge in a wide
bin.  The bracket and its interpolation are read off the cell's mean bill
curve, so they give the capacity at which the cell's average bill reaches
zero, and a mean bill follows the largest bills in the cell.  The median and
interquartile range beside them are taken over each site-year's own crossing,
interpolated on its own bill curve, so they give what a typical site-year in
the cell needs.  Both are taken over the site-years that reach $0 within the
modelled capacities, so they are read with the share beside them: the share
reaching $0 at any modelled capacity says how much of the cell the median and
range describe, and the share at the reported capacity says how much of the
cell is at $0 with that battery.  The range needs at least four site-years
reaching $0.  A site-year already at or below $0 at the smallest modelled
capacity is counted at that capacity, which is a floor on its true crossing.

Both interpolations read between modelled capacities 5 kWh apart and cannot
see a kink between them, so where a site-year's curve flattens inside the
bracketing pair the crossing is overstated by up to about half a kWh.  The
bracket itself is exact; the interpolation is a convenience within it.

Variables
---------
METRICS below is the registry of reported variables and TABLES groups them
into the emitted tables.  A new variable is a Metric entry naming its label,
format and a function of the cell, plus its key in a TABLES group; nothing
else needs changing.  A metric is computed only where the cell holds at least
--min-sites site-years, unless it is marked always, which is for the
variables describing the cell's population rather than a modelled result.

Output
------
Written to --output-dir:
    pv_usage_matrix_<basis>_<plan>.md           tables as markdown
    pv_usage_matrix_<basis>_<plan>.html         the same tables, for a browser
    pv_usage_matrix_<basis>_<plan>.csv          one row per cell and metric
    pv_usage_matrix_<basis>_<plan>_cells.csv    the site-years behind the cells
    pv_usage_matrix_<basis>_<plan>_figure_data.csv   one row per cell, numeric,
                                                for pv_usage_matrix_figures.py

Usage (from project root):
    python3 analysis/scripts/pv_usage_matrix.py --data-root "/Volumes/Manual/Thesis Project"
    python3 analysis/scripts/pv_usage_matrix.py --data-root "/Volumes/Manual/Thesis Project" --kwh 15
    python3 analysis/scripts/pv_usage_matrix.py --data-root "/Volumes/Manual/Thesis Project" --usage-quantiles 5
    python3 analysis/scripts/pv_usage_matrix.py --data-root "/Volumes/Manual/Thesis Project" --max-usage 20000
"""
from __future__ import annotations

import argparse
import html
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import financial_analysis as fa

# A band bound at or above this marks the band open at that end, and its
# header is rendered as an inequality rather than a range.
OPEN_BOUND = 900.0

# Where edp_annual_totals.py's output is looked for, in order.  Its docstring
# names data/processed/edp under the data root; the files as written sit in
# the repository's data/raw/edp, so both are tried under both roots.
TOTALS_DIRS = [
    Path('data') / 'raw' / 'edp',
    Path('data') / 'processed' / 'edp',
]


# ── Cell data ─────────────────────────────────────────────────────────────────

@dataclass
class Cell:
    """One matrix cell: its site-years, and the modelled series over them.

    site_years  one row per site-year in the cell, with state, year, size_kw,
                annual_pv_kwh and annual_gross_load_kwh.
    rows        the per-site-year bills and savings by mode and capacity.
    bills       mean household bill by mode and capacity.
    savings     savings by mode and capacity, as sums and means over the cell.
    solar_bill  mean household bill of the solar-only baseline.
    """
    label: str
    pv_band: str
    usage_q: str
    site_years: pd.DataFrame
    rows: pd.DataFrame
    bills: pd.DataFrame
    savings: pd.DataFrame
    solar_bill: float
    opt_mode: str
    sc_mode: str

    @property
    def n(self) -> int:
        return len(self.site_years)

    def bill(self, mode: str, kwh: int) -> float:
        """Mean household bill at one capacity, $/yr."""
        s = self.bills
        hit = s[(s['mode'] == mode) & (s['battery_kwh'] == kwh)]['household_total']
        return float(hit.iloc[0]) if len(hit) else np.nan

    def bill_curve(self, mode: str) -> pd.Series:
        """Mean household bill by capacity, ascending, $/yr."""
        s = self.bills[self.bills['mode'] == mode]
        if s.empty:
            return pd.Series(dtype=float)
        return s.set_index('battery_kwh')['household_total'].sort_index()

    def saving_sum(self, mode: str, kwh: int, col: str) -> float:
        s = self.savings
        hit = s[(s['mode'] == mode) & (s['battery_kwh'] == kwh)][f'{col}_sum']
        return float(hit.iloc[0]) if len(hit) else np.nan

    def saving_mean(self, mode: str, kwh: int, col: str) -> float:
        s = self.savings
        hit = s[(s['mode'] == mode) & (s['battery_kwh'] == kwh)][f'{col}_mean']
        return float(hit.iloc[0]) if len(hit) else np.nan

    def own_zero_capacity(self, mode: str) -> pd.Series:
        """Each site-year's own interpolated capacity for a $0 bill.

        NaN where the site-year's bill never reaches $0 within the modelled
        capacities, and the smallest modelled capacity where it is already
        there, which is a floor on its true crossing.
        """
        d = self.rows[self.rows['mode'] == mode]
        if d.empty:
            return pd.Series(dtype=float)
        wide = d.pivot_table(index=['site_id', 'year'], columns='battery_kwh',
                             values='household_total')
        sizes = wide.columns.to_numpy(float)
        v = wide.to_numpy(float)
        out = np.full(len(wide), np.nan)
        out[v[:, 0] <= 0] = sizes[0]
        for i in range(1, len(sizes)):
            hit = np.isnan(out) & (v[:, i] <= 0) & (v[:, i - 1] > 0)
            if not hit.any():
                continue
            lo, hi = sizes[i - 1], sizes[i]
            v_lo, v_hi = v[hit, i - 1], v[hit, i]
            out[hit] = lo + (hi - lo) * v_lo / (v_lo - v_hi)
        return pd.Series(out, index=wide.index)


def ratio(numerator: float, denominator: float) -> float:
    """numerator / denominator, NaN where the denominator is not positive.

    A ratio of sums is a relative value only against a positive reference.
    Same rule as volatility_sensitivity.ratio, so the relative values here are
    comparable with the ones reported there.
    """
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return np.nan
    return float(numerator / denominator) if denominator > 0 else np.nan


def rel_value(cell: Cell, kwh: int, col: str) -> float:
    """Relative value of coordination: the cell's summed optimised saving over
    its summed self-consumption saving at one capacity."""
    return ratio(cell.saving_sum(cell.opt_mode, kwh, col),
                 cell.saving_sum(cell.sc_mode, kwh, col))


def zero_bill_bracket(cell: Cell, mode: str) -> tuple[float, float] | None:
    """The two modelled capacities bracketing a mean household bill of $0.

    Returns (lower, upper) at the first crossing from a positive mean bill to
    a non-positive one, (nan, smallest) where the mean bill is already
    non-positive at the smallest modelled capacity, (largest, nan) where it is
    still positive at the largest, and None where there is no curve to read.
    """
    curve = cell.bill_curve(mode).dropna()
    if curve.empty:
        return None
    sizes = curve.index.to_numpy(float)
    vals = curve.to_numpy(float)
    if vals[0] <= 0:
        return (np.nan, sizes[0])
    for i in range(1, len(vals)):
        if vals[i] <= 0:
            return (sizes[i - 1], sizes[i])
    return (sizes[-1], np.nan)


def zero_bill_interp(cell: Cell, mode: str) -> float:
    """Linear interpolation of the capacity at which the mean household bill
    reaches $0, between the bracketing capacities.  NaN where the crossing is
    outside the modelled range."""
    br = zero_bill_bracket(cell, mode)
    if br is None or not np.isfinite(br[0]) or not np.isfinite(br[1]):
        return np.nan
    curve = cell.bill_curve(mode)
    lo, hi = br
    v_lo, v_hi = float(curve[lo]), float(curve[hi])
    if v_lo == v_hi:
        return np.nan
    return lo + (hi - lo) * v_lo / (v_lo - v_hi)


def fmt_bracket(cell: Cell, mode: str) -> str:
    br = zero_bill_bracket(cell, mode)
    if br is None:
        return '—'
    lo, hi = br
    if not np.isfinite(lo):
        return f'≤{hi:g}'
    if not np.isfinite(hi):
        return f'>{lo:g}'
    return f'{lo:g}–{hi:g}'


def share_reaching_zero(cell: Cell) -> float:
    """Share of the cell's site-years whose own bill reaches $0 within the
    modelled capacities, as a percentage."""
    own = cell.own_zero_capacity(cell.opt_mode)
    if own.empty:
        return np.nan
    return 100.0 * float(own.notna().mean())


def share_zero_at(cell: Cell, kwh: int) -> float:
    """Share of the cell's site-years whose own bill is $0 or below at one
    capacity, as a percentage."""
    d = cell.rows[(cell.rows['mode'] == cell.opt_mode)
                  & (cell.rows['battery_kwh'] == kwh)]
    if d.empty:
        return np.nan
    return 100.0 * float((d['household_total'] <= 0).mean())


MIN_REACHING = 4


def own_zero_quartiles(cell: Cell) -> tuple[float, float, float]:
    """(25th, 50th, 75th percentile) of the site-years' own $0 crossings,
    over those that reach $0 within the modelled capacities.  NaN where fewer
    than MIN_REACHING do."""
    own = cell.own_zero_capacity(cell.opt_mode).dropna()
    if len(own) < MIN_REACHING:
        return (np.nan, np.nan, np.nan)
    return tuple(float(own.quantile(q)) for q in (.25, .5, .75))


def own_zero_median(cell: Cell) -> float:
    """Median of the site-years' own $0 crossings, over those that reach $0
    within the modelled capacities, so that it sits inside the interquartile
    range beside it."""
    return own_zero_quartiles(cell)[1]


def own_zero_iqr(cell: Cell) -> str:
    """Interquartile range of the site-years' own $0 crossings, over those
    that reach it."""
    lo, _, hi = own_zero_quartiles(cell)
    if not np.isfinite(lo):
        return '—'
    return f'{lo:,.0f}–{hi:,.0f}'


def usage_span(cell: Cell) -> str:
    s = cell.site_years['annual_gross_load_kwh']
    if s.empty:
        return '—'
    return f'{s.min():,.0f}–{s.max():,.0f}'


# ── Metric registry ───────────────────────────────────────────────────────────

@dataclass
class Metric:
    key: str
    label: str
    fn: Callable[[Cell, int], object]
    fmt: str = '{:,.0f}'
    # A metric describing the cell's population rather than a modelled result
    # is reported even where the cell is below --min-sites.
    always: bool = False

    def value(self, cell: Cell, kwh: int):
        return self.fn(cell, kwh)

    def label_for(self, kwh: int) -> str:
        """The label, with {kwh} replaced by the reported capacity."""
        return self.label.replace('{kwh}', f'{kwh}')

    def render(self, value) -> str:
        if value is None:
            return '—'
        if isinstance(value, str):
            return value
        if not np.isfinite(value):
            return '—'
        return self.fmt.format(value)


METRICS: list[Metric] = [
    Metric('n_site_years', 'Site-years',
           lambda c, k: float(c.n), '{:,.0f}', always=True),
    Metric('avg_pv_kwh', 'Avg PV generation (kWh/yr)',
           lambda c, k: c.site_years['annual_pv_kwh'].mean(), '{:,.0f}',
           always=True),
    Metric('avg_usage_kwh', 'Avg usage (kWh/yr)',
           lambda c, k: c.site_years['annual_gross_load_kwh'].mean(),
           '{:,.0f}', always=True),
    Metric('med_usage_kwh', 'Median usage (kWh/yr)',
           lambda c, k: c.site_years['annual_gross_load_kwh'].median(),
           '{:,.0f}', always=True),
    Metric('usage_span', 'Usage range (kWh/yr)',
           lambda c, k: usage_span(c), always=True),
    Metric('avg_pv_kw', 'Avg PV array (kW)',
           lambda c, k: c.site_years['size_kw'].mean(), '{:,.2f}',
           always=True),
    Metric('bill_solar_only', 'Bill, solar only ($/yr)',
           lambda c, k: c.solar_bill, '{:,.0f}'),
    Metric('bill_sc', 'Bill, self-consumption ($/yr)',
           lambda c, k: c.bill(c.sc_mode, k), '{:,.0f}'),
    Metric('bill_opt', 'Bill, optimised ($/yr)',
           lambda c, k: c.bill(c.opt_mode, k), '{:,.0f}'),
    Metric('hh_sav_sc', 'Household saving, self-consumption ($/yr)',
           lambda c, k: c.saving_mean(c.sc_mode, k, 'hh_saving'), '{:,.0f}'),
    Metric('hh_sav_opt', 'Household saving, optimised ($/yr)',
           lambda c, k: c.saving_mean(c.opt_mode, k, 'hh_saving'), '{:,.0f}'),
    Metric('rel_value_hh', 'Relative value, household bill',
           lambda c, k: rel_value(c, k, 'hh_saving'), '{:,.2f}'),
    Metric('ws_sav_sc', 'Wholesale saving, self-consumption ($/yr)',
           lambda c, k: c.saving_mean(c.sc_mode, k, 'ws_saving'), '{:,.0f}'),
    Metric('ws_sav_opt', 'Wholesale saving, optimised ($/yr)',
           lambda c, k: c.saving_mean(c.opt_mode, k, 'ws_saving'), '{:,.0f}'),
    Metric('rel_value_ws', 'Relative value, net wholesale',
           lambda c, k: rel_value(c, k, 'ws_saving'), '{:,.2f}'),
    Metric('zero_bill_range', 'Mean bill: capacity range (kWh)',
           lambda c, k: fmt_bracket(c, c.opt_mode)),
    Metric('zero_bill_interp', 'Mean bill: interpolated (kWh)',
           lambda c, k: zero_bill_interp(c, c.opt_mode), '{:,.1f}'),
    Metric('own_zero_median', 'Per site-year: median (kWh)',
           lambda c, k: own_zero_median(c), '{:,.1f}'),
    Metric('own_zero_iqr', 'Per site-year: interquartile range (kWh)',
           lambda c, k: own_zero_iqr(c)),
    Metric('share_zero_bill', 'Share reaching $0 at any capacity (%)',
           lambda c, k: share_reaching_zero(c), '{:,.0f}'),
    Metric('share_zero_at_kwh', 'Share at $0 with {kwh} kWh (%)',
           lambda c, k: share_zero_at(c, k), '{:,.0f}'),
]

METRIC_BY_KEY = {m.key: m for m in METRICS}


@dataclass
class Table:
    title: str
    keys: list[str]
    note: str = ''
    rows: list[Metric] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rows = [METRIC_BY_KEY[k] for k in self.keys]


def tables(kwh: int, max_kwh: int) -> list[Table]:
    """The emitted tables.  Add a variable by naming its metric key here."""
    return [
        Table('1. Population',
              ['n_site_years', 'avg_pv_kwh', 'avg_usage_kwh',
               'med_usage_kwh', 'usage_span'],
              'Site-years in each cell, and the annual PV generation and '
              'gross household load over them.  The median and the range sit '
              'beside the mean because the load bins hold equal counts and so '
              'are unequal in width.'),
        Table('2. Average household bill',
              ['bill_solar_only', 'bill_sc', 'bill_opt'],
              f'Mean annual household bill with a {kwh} kWh battery, and with '
              f'no battery for reference.  Negative is a net credit.'),
        Table(f'3. Relative value of coordination at {kwh} kWh',
              ['hh_sav_sc', 'hh_sav_opt', 'rel_value_hh',
               'ws_sav_sc', 'ws_sav_opt', 'rel_value_ws'],
              'Savings are means per site-year against the solar-only bill.  '
              'The relative value is the summed optimised saving over the '
              'summed self-consumption saving, left blank where that '
              'self-consumption saving is not positive.'),
        Table('4. PV array size',
              ['avg_pv_kw'],
              'The tiered size_kw estimate from financial_analysis, averaged '
              'over the cell.'),
        Table('5. Capacity for a $0 bill',
              ['zero_bill_range', 'zero_bill_interp', 'own_zero_median',
               'own_zero_iqr', 'share_zero_bill', 'share_zero_at_kwh'],
              f'On the optimised run.  The first two read the capacity at '
              f'which the cell\'s mean bill reaches $0 off its mean bill '
              f'curve.  The next two take each site-year\'s own crossing on '
              f'its own bill curve, so they describe a typical site-year '
              f'rather than the average bill, over the site-years that reach '
              f'$0 by {max_kwh} kWh; the share at any capacity says how much '
              f'of the cell that is, and the last row how much of it is at '
              f'$0 with a {kwh} kWh battery.'),
    ]


# ── Bands ─────────────────────────────────────────────────────────────────────

def load_bands(path: Path) -> list[tuple[str, float, float]]:
    """(label, lo, hi) per band from a YAML holding a size_bands list."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    bands = []
    for label in raw['size_bands']:
        lo_str, hi_str = label.split('-')
        bands.append((label, float(lo_str), float(hi_str)))
    for (_, _, hi), (_, lo_next, _) in zip(bands, bands[1:]):
        if hi != lo_next:
            raise ValueError(
                f"Band config error in {path}: bands must be contiguous, but "
                f"one ends at {hi:g} and the next starts at {lo_next:g}.")
    return bands


def assign_band(size_kw: float,
                bands: list[tuple[str, float, float]]) -> str | None:
    """The band a size falls in, or None outside the outermost bounds.  Bounds
    are inclusive at the lower band, as in financial_analysis."""
    if pd.isna(size_kw) or size_kw < bands[0][1] or size_kw > bands[-1][2]:
        return None
    for label, lo, hi in bands:
        if lo <= size_kw <= hi:
            return label
    return None


def band_header(band: str, bands: list[tuple[str, float, float]]) -> str:
    """A band's column header, as an inequality where the band is open."""
    if band == 'All':
        return 'All'
    label, lo, hi = next(b for b in bands if b[0] == band)
    if label == bands[0][0] and lo <= 0:
        return f'≤{hi:g} kW'
    if label == bands[-1][0] and hi >= OPEN_BOUND:
        return f'>{lo:g} kW'
    return f'{lo:g}–{hi:g} kW'


# ── Site-year population ──────────────────────────────────────────────────────

def load_sizes(data_root: Path, scenarios: list[tuple[str, str]],
               states_for_pooling: list[str]) -> pd.DataFrame:
    """size_kw per site-year, by the tiered estimate in financial_analysis.

    The pooled DC:AC ratio and the pooled regression fallback are computed per
    year over every state in states_for_pooling, as financial_analysis does
    over every state in its weighting config, so a state's sizes do not depend
    on which states were asked for.
    """
    frames = []
    for year in sorted({y for _, y in scenarios}):
        pooled = []
        for state in states_for_pooling:
            try:
                pooled.append(fa.load_site_meta(data_root, year, state)
                              .assign(state=state))
            except FileNotFoundError:
                continue
        if not pooled:
            continue
        pooled = pd.concat(pooled, ignore_index=True)
        dc_ac = fa.compute_dc_ac_ratio(pooled)
        print(f"  NOTE: {year} pooled DC:AC ratio = {dc_ac:.3f}")
        pooled_sized = fa.assign_direct_size(pooled, dc_ac)
        for state in sorted({s for s, y in scenarios if y == year}):
            meta = pooled_sized[pooled_sized['state'] == state].copy()
            if meta.empty:
                continue
            meta = fa.estimate_missing_sizes(meta, state,
                                             fallback_meta=pooled_sized)
            frames.append(meta.assign(state=state, year=year))
    if not frames:
        return pd.DataFrame(columns=['site_id', 'state', 'year', 'size_kw',
                                     'size_source', 'size_estimated'])
    return pd.concat(frames, ignore_index=True)


def load_usage(data_root: Path, years: list[str], source: str,
               repo_root: Path) -> pd.DataFrame:
    """annual_gross_load_kwh and annual_pv_kwh per site-year.

    source 'totals' reads edp_annual_totals.py's per-interval sums, looked for
    under the data root and then the repository.  source 'metadata' derives
    both from the EDP site metadata instead.
    """
    cols = ['site_id', 'year', 'n_intervals',
            'annual_gross_load_kwh', 'annual_pv_kwh']
    frames = []
    if source == 'totals':
        for year in years:
            found = None
            for root in (data_root, repo_root):
                for rel in TOTALS_DIRS:
                    cand = root / rel / f'{year}_site_annual_totals.csv'
                    if cand.exists():
                        found = cand
                        break
                if found:
                    break
            if found is None:
                print(f"  WARNING: no {year}_site_annual_totals.csv under "
                      f"{data_root} or {repo_root}; {year} site-years have no "
                      f"usage and are left out of the matrix.  Run "
                      f"analysis/scripts/edp_annual_totals.py, or pass "
                      f"--usage-source metadata.", file=sys.stderr)
                continue
            t = pd.read_csv(found).rename(columns={'edp_site_id': 'site_id'})
            frames.append(t.assign(year=str(year))[cols])
    else:
        for year in years:
            f = (data_root / 'data' / 'processed' / 'edp'
                 / f'{year}_selected_sites_clean.csv')
            if not f.exists():
                continue
            t = pd.read_csv(f, usecols=['edp_site_id',
                                        'avg_daily_gross_load_wh',
                                        'annual_generation_kwh'])
            t = t.rename(columns={'edp_site_id': 'site_id',
                                  'annual_generation_kwh': 'annual_pv_kwh'})
            t['annual_gross_load_kwh'] = (t['avg_daily_gross_load_wh']
                                          * 365 / 1000)
            t['n_intervals'] = np.nan
            frames.append(t.assign(year=str(year))[cols])
    if not frames:
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True)


def bin_labels(n: int) -> list[str]:
    """Labels for n usage bins, Q1 upward whatever the count."""
    return [f'Q{i + 1}' for i in range(n)]


def build_population(sizes: pd.DataFrame, usage: pd.DataFrame,
                     modelled: pd.DataFrame,
                     bands: list[tuple[str, float, float]],
                     n_bins: int, explicit_cuts: list[float] | None,
                     max_usage: float | None
                     ) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """The matrix population: modelled site-years with a band and a usage
    figure, cut into usage bins.  Returns it with the cut points and the bin
    labels, and prints the funnel from modelled site-years to matrix
    site-years."""
    pop = modelled.merge(sizes[['site_id', 'state', 'year', 'size_kw',
                                'size_source', 'size_estimated']],
                         on=['site_id', 'state', 'year'], how='left')
    pop = pop.merge(usage, on=['site_id', 'year'], how='left')
    n_modelled = len(pop)

    no_meta = pop['size_source'].isna()
    pop['pv_band'] = pop['size_kw'].apply(lambda kw: assign_band(kw, bands))
    no_band = pop['pv_band'].isna() & ~no_meta
    no_usage = pop['annual_gross_load_kwh'].isna() & ~no_meta & ~no_band
    over = pd.Series(False, index=pop.index)
    if max_usage is not None:
        over = ((pop['annual_gross_load_kwh'] > max_usage)
                & ~no_meta & ~no_band & ~no_usage)

    print('\nMatrix population')
    print(f"  modelled site-years               {n_modelled:>5}")
    print(f"  no EDP metadata                  -{no_meta.sum():>5}")
    print(f"  no size, or outside the bands    -{no_band.sum():>5}")
    print(f"  no annual usage                  -{no_usage.sum():>5}")
    if max_usage is not None:
        print(f"  usage over {max_usage:>9,.0f} kWh       -{over.sum():>5}")
    for mask, what in ((no_meta, 'no metadata'), (no_usage, 'no usage'),
                       (no_band, 'no band')):
        if mask.any() and mask.sum() <= 25:
            print(f"    {what}: "
                  f"{', '.join(sorted(set(pop.loc[mask, 'site_id'])))}")

    pop = pop[~(no_meta | no_band | no_usage | over)].copy()
    print(f"  matrix site-years                 {len(pop):>5}")
    if pop.empty:
        return pop, np.array([]), []

    load = pop['annual_gross_load_kwh']
    if explicit_cuts:
        cuts = np.array(sorted(float(c) for c in explicit_cuts))
        labels = bin_labels(len(cuts) + 1)
    else:
        cuts = load.quantile([i / n_bins
                              for i in range(1, n_bins)]).to_numpy()
        labels = bin_labels(n_bins)
    if len(np.unique(cuts)) != len(cuts):
        raise ValueError(f"Usage cut points are not distinct: {cuts}.  Ask "
                         f"for fewer than {n_bins} bins.")
    pop['usage_q'] = pd.cut(load, [-np.inf, *cuts, np.inf], labels=labels)
    return pop, cuts, labels


# ── Cell assembly ─────────────────────────────────────────────────────────────

def modelled_site_years(data_root: Path, scenarios: list[tuple[str, str]],
                        basis: str, baseline_mode: str
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every scenario's per-site savings and its solar-only baseline bill.

    Returns (savings, solar) where savings has one row per site-year, capacity
    and mode, and solar one row per site-year.
    """
    sav_frames, solar_frames = [], []
    for state, year in scenarios:
        data = fa.load_scenario(data_root, state, year, baseline_mode)
        if data is None:
            continue
        df = fa.build_savings(data)
        df = fa.apply_site_basis_adjustments(
            df[['site_id', 'battery_kwh', 'mode', 'household_total',
                'net_wholesale', 'hh_saving', 'ws_saving']], basis)
        sav_frames.append(df.assign(state=state, year=year))
        solar_frames.append(data['solar_only'].assign(state=state, year=year))
    if not sav_frames:
        return pd.DataFrame(), pd.DataFrame()
    return (pd.concat(sav_frames, ignore_index=True),
            pd.concat(solar_frames, ignore_index=True))


def make_cells(pop: pd.DataFrame, savings: pd.DataFrame, solar: pd.DataFrame,
               bands: list[str], labels: list[str], opt_mode: str,
               sc_mode: str) -> dict[tuple[str, str], Cell]:
    """A Cell for every (usage bin, PV band) pair, and for the margins
    ('All', band), (bin, 'All') and ('All', 'All')."""
    key = ['site_id', 'year']
    membership = pop[[*key, 'pv_band', 'usage_q']]
    sav = savings.merge(membership, on=key, how='inner')
    sol = solar.merge(membership, on=key, how='inner')

    cells: dict[tuple[str, str], Cell] = {}
    for q in [*labels, 'All']:
        for b in [*bands, 'All']:
            sy, s, sb = pop, sav, sol
            if q != 'All':
                sy = sy[sy['usage_q'] == q]
                s = s[s['usage_q'] == q]
                sb = sb[sb['usage_q'] == q]
            if b != 'All':
                sy = sy[sy['pv_band'] == b]
                s = s[s['pv_band'] == b]
                sb = sb[sb['pv_band'] == b]

            if s.empty:
                bills = pd.DataFrame(columns=['mode', 'battery_kwh',
                                              'household_total'])
                agg = pd.DataFrame(columns=[
                    'mode', 'battery_kwh', 'hh_saving_sum', 'hh_saving_mean',
                    'ws_saving_sum', 'ws_saving_mean', 'n'])
            else:
                bills = (s.groupby(['mode', 'battery_kwh'], as_index=False)
                         ['household_total'].mean())
                agg = (s.groupby(['mode', 'battery_kwh'])
                       .agg(hh_saving_sum=('hh_saving', 'sum'),
                            hh_saving_mean=('hh_saving', 'mean'),
                            ws_saving_sum=('ws_saving', 'sum'),
                            ws_saving_mean=('ws_saving', 'mean'),
                            n=('hh_saving', 'size'))
                       .reset_index())
            cells[(q, b)] = Cell(
                label=f'{q} / {b}', pv_band=b, usage_q=q,
                site_years=sy, rows=s, bills=bills, savings=agg,
                solar_bill=(float(sb['hh_solar'].mean()) if len(sb)
                            else np.nan),
                opt_mode=opt_mode, sc_mode=sc_mode)
    return cells


# ── Rendering ─────────────────────────────────────────────────────────────────

def bin_header(q: str, cuts: np.ndarray, labels: list[str]) -> str:
    if q == 'All':
        return 'All'
    i = labels.index(q)
    if i == 0:
        return f'{q} (≤{cuts[0]:,.0f})'
    if i == len(labels) - 1:
        return f'{q} (>{cuts[-1]:,.0f})'
    return f'{q} ({cuts[i - 1]:,.0f}–{cuts[i]:,.0f})'


def cell_value(metric: Metric, cell: Cell, kwh: int, min_sites: int):
    if cell.n == 0:
        return None
    if cell.n < min_sites and not metric.always:
        return None
    return metric.value(cell, kwh)


def table_grid(table: Table, cells: dict,
               bands: list[tuple[str, float, float]], cuts: np.ndarray,
               labels: list[str], kwh: int,
               min_sites: int) -> list[list[str]]:
    """Header row, then one row per usage bin and metric."""
    cols = [*[b[0] for b in bands], 'All']
    grid = [['Usage bin (kWh/yr)', 'Variable',
             *[band_header(b, bands) for b in cols]]]
    for q in [*labels, 'All']:
        for i, metric in enumerate(table.rows):
            row = [bin_header(q, cuts, labels) if i == 0 else '',
                   metric.label_for(kwh)]
            for b in cols:
                row.append(metric.render(
                    cell_value(metric, cells[(q, b)], kwh, min_sites)))
            grid.append(row)
    return grid


def state_grid(cells: dict, bands: list[tuple[str, float, float]],
               cuts: np.ndarray, labels: list[str],
               states: list[str]) -> list[list[str]]:
    cols = [*[b[0] for b in bands], 'All']
    grid = [['Usage bin (kWh/yr)', 'State',
             *[band_header(b, bands) for b in cols]]]
    for q in [*labels, 'All']:
        for i, st in enumerate(states):
            row = [bin_header(q, cuts, labels) if i == 0 else '', st]
            for b in cols:
                sy = cells[(q, b)].site_years
                n = int((sy['state'] == st).sum()) if len(sy) else 0
                row.append(f'{n:,}' if n else '—')
            grid.append(row)
    return grid


def md_table(grid: list[list[str]]) -> str:
    widths = [max(len(r[i]) for r in grid) for i in range(len(grid[0]))]
    def line(cells: list[str]) -> str:
        return ('| ' + ' | '.join(c.ljust(w) for c, w in zip(cells, widths))
                + ' |')
    out = [line(grid[0]), '|' + '|'.join('-' * (w + 2) for w in widths) + '|']
    out += [line(r) for r in grid[1:]]
    return '\n'.join(out)


def html_table(grid: list[list[str]]) -> str:
    rows = ['<table>',
            '<thead><tr>'
            + ''.join(f'<th>{html.escape(c)}</th>' for c in grid[0])
            + '</tr></thead>',
            '<tbody>']
    for r in grid[1:]:
        cls = ' class="group"' if r[0] else ''
        cells = ''.join(
            f'<td class="{"lab" if i < 2 else "num"}">{html.escape(c)}</td>'
            for i, c in enumerate(r))
        rows.append(f'<tr{cls}>{cells}</tr>')
    rows += ['</tbody>', '</table>']
    return '\n'.join(rows)


HTML_CSS = """
body { font-family: -apple-system, Helvetica, Arial, sans-serif;
       margin: 2rem auto; max-width: 1250px; color: #1a1a1a; line-height: 1.5; }
h1 { font-size: 1.45rem; } h2 { font-size: 1.05rem; margin-top: 2rem; }
p.note { color: #555; font-size: 0.88rem; max-width: 70ch; }
table { border-collapse: collapse; font-size: 0.85rem;
        margin: 0.75rem 0 1.5rem; }
th, td { padding: 0.3rem 0.65rem; }
thead th { background: #f2f2f2; text-align: left; border-bottom: 1px solid #bbb; }
thead th:nth-child(n+3) { text-align: right; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
tr.group td { border-top: 1px solid #ddd; }
td:nth-child(1) { font-weight: 600; white-space: nowrap; }
pre { background: #f7f7f7; padding: 0.75rem; font-size: 0.8rem;
      overflow-x: auto; }
"""


def render(title: str, preamble: list[str],
           blocks: list[tuple[str, str, list]], as_html: bool) -> str:
    if as_html:
        parts = [f'<!DOCTYPE html>\n<meta charset="utf-8">\n'
                 f'<title>{html.escape(title)}</title>\n'
                 f'<style>{HTML_CSS}</style>',
                 f'<h1>{html.escape(title)}</h1>',
                 '<pre>' + html.escape('\n'.join(preamble)) + '</pre>']
        for heading, note, grid in blocks:
            parts.append(f'<h2>{html.escape(heading)}</h2>')
            if note:
                parts.append(f'<p class="note">{html.escape(note)}</p>')
            parts.append(html_table(grid))
        return '\n'.join(parts) + '\n'
    parts = [f'# {title}', '', '```', *preamble, '```']
    for heading, note, grid in blocks:
        parts += ['', f'## {heading}', '']
        if note:
            parts += [note, '']
        parts.append(md_table(grid))
    return '\n'.join(parts) + '\n'


def long_rows(cells: dict, bands: list[str], labels: list[str], kwh: int,
              min_sites: int) -> pd.DataFrame:
    """One row per cell and metric, for reuse by other scripts."""
    rows = []
    for q in [*labels, 'All']:
        for b in [*bands, 'All']:
            cell = cells[(q, b)]
            for metric in METRICS:
                v = cell_value(metric, cell, kwh, min_sites)
                text, number = None, None
                if isinstance(v, str):
                    text = v
                elif v is not None and np.isfinite(v):
                    number = float(v)
                rows.append(dict(usage_bin=q, pv_band=b,
                                 n_site_years=cell.n, kwh=kwh,
                                 metric=metric.key, label=metric.label_for(kwh),
                                 value=number, value_text=text))
    return pd.DataFrame(rows)


def figure_rows(cells: dict, bands: list[tuple[str, float, float]],
                labels: list[str], cuts: np.ndarray, kwh: int,
                top_usage: float) -> pd.DataFrame:
    """One row per matrix cell (margins excluded), numeric and ungated, for
    pv_usage_matrix_figures.py, which applies its own site-year threshold.

    bin_lo_kwh is blank for the first usage bin, and bin_hi_kwh for the last
    is top_usage: the --max-usage cutoff where one was set, otherwise the
    largest load in the population.  own_zero_q* are over the site-years that
    reach $0 within the modelled capacities.
    """
    rows = []
    for i, q in enumerate(labels):
        lo = float(cuts[i - 1]) if i > 0 else np.nan
        hi = float(cuts[i]) if i < len(cuts) else float(top_usage)
        for label, _, _ in bands:
            c = cells[(q, label)]
            q25, q50, q75 = own_zero_quartiles(c)
            rows.append(dict(
                usage_bin=q, pv_band=label,
                pv_band_label=band_header(label, bands).replace(' kW', ''),
                n_site_years=c.n, bin_lo_kwh=lo, bin_hi_kwh=hi,
                avg_usage_kwh=(c.site_years['annual_gross_load_kwh'].mean()
                               if c.n else np.nan),
                avg_pv_kwh=(c.site_years['annual_pv_kwh'].mean()
                            if c.n else np.nan),
                own_zero_q25=q25, own_zero_q50=q50, own_zero_q75=q75,
                share_zero_any=share_reaching_zero(c) if c.n else np.nan,
                share_zero_at_kwh=share_zero_at(c, kwh) if c.n else np.nan,
                kwh=kwh))
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description='PV size band by usage quantile matrix of modelled '
                    'household and wholesale results.')
    p.add_argument('--data-root', type=Path, default=Path('.'),
                   help='Root holding data/output/runs and data/processed/edp.')
    p.add_argument('--output-dir', type=Path,
                   default=Path('analysis/output/pv_usage_matrix'))
    p.add_argument('--basis', default='achievable', choices=list(fa.RUN_BASES),
                   help='Run basis (default: achievable, which is '
                        'full_run_rolling_amber for Amber).')
    p.add_argument('--plan', default='amber', choices=list(fa.PLAN_KEYS),
                   help='Offer to report (default: amber).')
    p.add_argument('--kwh', type=int, default=20,
                   help='Capacity for the bill and the relative value '
                        '(default: 20).')
    p.add_argument('--states', nargs='+', default=None)
    p.add_argument('--years', nargs='+', default=None)
    p.add_argument('--min-sites', type=int, default=5,
                   help='Cells with fewer site-years report only the '
                        'population variables (default: 5).')
    p.add_argument('--usage-quantiles', type=int, default=4,
                   help='Number of equal-count usage bins (default: 4).')
    p.add_argument('--usage-cuts', nargs='+', type=float, default=None,
                   help='Explicit usage cut points in kWh/yr, instead of '
                        'equal-count bins.')
    p.add_argument('--max-usage', type=float, default=None,
                   help='Leave out site-years with annual load above this, '
                        'for testing the influence of the upper tail.')
    p.add_argument('--usage-source', choices=['totals', 'metadata'],
                   default='totals',
                   help="'totals' uses edp_annual_totals.py's per-interval "
                        "sums (default); 'metadata' derives load from "
                        "avg_daily_gross_load_wh.")
    p.add_argument('--bands-file', type=Path,
                   default=Path('analysis/config/pv_usage_matrix_bands.yaml'),
                   help='PV size bands for the matrix (default: '
                        'analysis/config/pv_usage_matrix_bands.yaml).')
    p.add_argument('--weights-file', type=Path,
                   default=Path('analysis/config/pv_size_bands.yaml'),
                   help='Read only for its state list, which sets the states '
                        'the size estimates pool over (default: '
                        'analysis/config/pv_size_bands.yaml).')
    p.add_argument('--baseline', choices=['amber', 'own'], default='amber')
    args = p.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    for f in (args.bands_file, args.weights_file):
        if not f.exists():
            print(f"ERROR: no config at {f}", file=sys.stderr)
            sys.exit(1)
    bands = load_bands(args.bands_file)
    band_names = [b[0] for b in bands]
    states_for_pooling = list(
        fa.load_weights_config(args.weights_file)['by_state'])

    plan = next(pl for pl in fa.plans_for_basis(args.basis)
                if pl['sc_mode'] == fa.PLAN_KEYS[args.plan])
    fa.install_plans([plan], plan['sc_mode'])
    opt_mode, sc_mode = plan['lp_mode'], fa.sc_no_vpp_mode(plan)
    offer = fa._offer_name(plan)
    print(f"Offer {offer} on the {fa.RUN_BASES[args.basis]['title']} basis")
    print(f"  optimised          {plan['lp_dir']}")
    print(f"  self-consumption   {plan['sc_dir']}")
    print(f"  savings baseline   {fa.DEFAULT_BASELINE_RUN}")

    scenarios = fa.discover_scenarios(args.data_root)
    if not scenarios:
        print(f"ERROR: no scenarios under {args.data_root}.  The runs live on "
              f"the external drive, so pass --data-root.", file=sys.stderr)
        sys.exit(1)
    if args.states:
        scenarios = [s for s in scenarios if s[0] in args.states]
    if args.years:
        scenarios = [s for s in scenarios if s[1] in args.years]
    if not scenarios:
        print('ERROR: no scenarios left after --states / --years.',
              file=sys.stderr)
        sys.exit(1)
    print(f"  scenarios          "
          f"{', '.join(f'{s}_{y}' for s, y in scenarios)}")

    savings, solar = modelled_site_years(args.data_root, scenarios,
                                         args.basis, args.baseline)
    if savings.empty:
        print('ERROR: no modelled results loaded.', file=sys.stderr)
        sys.exit(1)
    for mode in (opt_mode, sc_mode):
        if mode not in set(savings['mode']):
            print(f"ERROR: mode '{mode}' is not in the loaded runs "
                  f"({sorted(set(savings['mode']))}).", file=sys.stderr)
            sys.exit(1)
    sizes_present = sorted(int(k) for k in savings['battery_kwh'].unique())
    if args.kwh not in sizes_present:
        print(f"ERROR: --kwh {args.kwh} is not a modelled capacity "
              f"({', '.join(str(k) for k in sizes_present)}).",
              file=sys.stderr)
        sys.exit(1)

    sizes = load_sizes(args.data_root, scenarios, states_for_pooling)
    usage = load_usage(args.data_root, sorted({y for _, y in scenarios}),
                       args.usage_source, repo_root)
    modelled = (savings[['site_id', 'state', 'year']]
                .drop_duplicates().reset_index(drop=True))
    pop, cuts, labels = build_population(
        sizes, usage, modelled, bands, args.usage_quantiles,
        args.usage_cuts, args.max_usage)
    if pop.empty:
        print('ERROR: no site-years left for the matrix.', file=sys.stderr)
        sys.exit(1)

    cells = make_cells(pop, savings, solar, band_names, labels,
                       opt_mode, sc_mode)
    states = sorted(pop['state'].unique())

    preamble = [
        f'Offer              {offer}',
        f'Run basis          {fa.RUN_BASES[args.basis]["title"]}',
        f'Optimised run      {plan["lp_dir"]}',
        f'Self-consumption   {plan["sc_dir"]}',
        f'Savings baseline   {fa.DEFAULT_BASELINE_RUN}',
        f'Capacity reported  {args.kwh} kWh',
        f'Scenarios          '
        f'{", ".join(f"{s}_{y}" for s, y in scenarios)}',
        f'Matrix site-years  {len(pop)} over '
        f'{pop["site_id"].nunique()} sites',
        f'Usage bins         {len(labels)}, cut at '
        f'{" / ".join(f"{c:,.0f}" for c in cuts)} kWh'
        + ('  (explicit)' if args.usage_cuts else '  (equal count)'),
        f'Usage source       {args.usage_source}',
        f'PV size bands      {args.bands_file}',
    ]
    if args.max_usage is not None:
        preamble.append(f'Usage trimmed      site-years above '
                        f'{args.max_usage:,.0f} kWh/yr left out')
    preamble.append(
        f'Cell means are unweighted over site-years.  Cells below '
        f'{args.min_sites} site-years report population variables only.')

    blocks = [(t.title, t.note,
               table_grid(t, cells, bands, cuts, labels, args.kwh,
                          args.min_sites))
              for t in tables(args.kwh, max(sizes_present))]
    blocks.append((
        '6. State composition',
        'Site-years by state in each cell.  State weighting is not applied to '
        'the cell means; this is what that leaves confounded.',
        state_grid(cells, bands, cuts, labels, states)))

    title = (f'PV size band by usage quantile — {offer}, '
             f'{fa.RUN_BASES[args.basis]["title"]}')
    stem = f'pv_usage_matrix_{args.basis}_{args.plan}'
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{stem}.md').write_text(render(title, preamble, blocks, False))
    (out / f'{stem}.html').write_text(render(title, preamble, blocks, True))
    long_rows(cells, band_names, labels, args.kwh, args.min_sites).to_csv(
        out / f'{stem}.csv', index=False)
    pop.drop(columns=['subarray_dc_rating_kw', 'inverter_ac_rating_kw'],
             errors='ignore').to_csv(out / f'{stem}_cells.csv', index=False)
    top_usage = (args.max_usage if args.max_usage is not None
                 else float(pop['annual_gross_load_kwh'].max()))
    figure_rows(cells, bands, labels, cuts, args.kwh, top_usage).to_csv(
        out / f'{stem}_figure_data.csv', index=False)

    print(f"\nWrote {out / f'{stem}.md'}")
    print(f"      {out / f'{stem}.html'}")
    print(f"      {out / f'{stem}.csv'}")
    print(f"      {out / f'{stem}_cells.csv'}")
    print(f"      {out / f'{stem}_figure_data.csv'}")
    print('\n' + md_table(blocks[0][2]))


if __name__ == '__main__':
    main()
