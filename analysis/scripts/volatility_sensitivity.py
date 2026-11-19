"""Coordination value and price volatility (docs/volatility_sensitivity_spec.md).

Tests how the relative value of coordination and the value captured with
imperfect foresight change with price volatility, for the household bill and
the net wholesale saving, from the existing runs' monthly bills. No model
runs.

The unit of analysis is the state-month (4 states x 60 months, 2021 to 2025)
at one battery capacity (20 kWh). Each state-month has two volatility
measures from the regional RRP in state local time:

  cap value     mean over the month's intervals of max(RRP - 300, 0), $/MWh
  daily spread  mean over the month's days of (mean of the highest two hours
                of RRP less the mean of the lowest two hours)

State-months are split into quartiles of cap value (q1 lowest). 2026-to-date
state-months (prices only) are placed against the 2021-2025 boundaries.

For each offer the monthly saving is the PV-size-weighted mean per site of
the Solar only bill less the battery run's, for household_total and for the
net wholesale cost (wholesale_usage_cost + wholesale_feedin_credit, as
ws_saving in financial_analysis.py), against the offer's own Solar only run,
for three runs:

  self-consumption      the offer's self-consumption run without VPP events
                        or critical rates (sc_no_vpp_dir in the registry)
  achievable optimised  imperfect foresight, maximum VPP dispatch
  perfect foresight     the registry's optimised run, maximum VPP dispatch

All run names come from financial_analysis.plans_for_basis. The self-consumption reference has no VPP
events, so the coordination value includes the offer's event dispatch,
critical rates and VPP payments; the achievable and perfect foresight values
are measured from the same reference.

Group metrics are ratios of sums over a group's state-months. The common
baseline savings (against the Amber Solar only run) for self-consumption and
achievable optimised give the mean annual saving charts: 12 x the group's mean
monthly saving.

Usage (from project root):
    python3 analysis/scripts/volatility_sensitivity.py --data-root "/Volumes/Manual/Thesis Project"
    python3 analysis/scripts/volatility_sensitivity.py --from-csv

--from-csv redraws the charts and reprints the text values from the saved
CSVs, with no data root.

Writes to analysis/output/sensitivity/volatility/ (see OUTPUT_FILES).
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import financial_analysis as fa  # noqa: E402
import flow_exposure_analysis as fe  # noqa: E402
import thesis_style as ts  # noqa: E402
from site_exclusions import EXCLUDED_SITES  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / 'analysis' / 'output' / 'sensitivity' / 'volatility'
WEIGHTS_FILE = REPO_ROOT / 'analysis' / 'config' / 'pv_size_bands.yaml'

# State -> local timezone, from the model's single definition. Loaded from the
# file for the reason given in flow_exposure_analysis.py.
_spec = importlib.util.spec_from_file_location(
    'localtime', REPO_ROOT / 'model' / 'core' / 'localtime.py')
_localtime = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_localtime)
STATE_TIMEZONE = _localtime.STATE_TIMEZONE

BATTERY_KWH = 20
MODEL_YEARS = tuple(range(2021, 2026))
CURRENT_YEAR = 2026
PRICE_YEARS = (2020,) + MODEL_YEARS + (CURRENT_YEAR,)   # 2020: first local hour of 2021
CAP_THRESHOLD = 300.0           # $/MWh
SPREAD_HOURS = 2
INTERVAL_MIN = 5
INTERVALS_PER_HOUR = 60 // INTERVAL_MIN
# A local day with fewer valid intervals than this is left out of the daily
# spread: 23-hour DST days (276) pass, partial first and last days of a
# price file and days lost to missing prices do not.
MIN_DAY_INTERVALS = 270
LOW_SITE_COUNT = 20
QUARTILES = ('q1', 'q2', 'q3', 'q4')
RUN_ROLES = ('solar', 'sc', 'ach', 'pf', 'amber_solar')

OWN_ROLES = ('solar', 'sc', 'ach', 'pf')
SAVINGS = ('sc', 'ach', 'pf')
COMMON_COLS = [f'{k}_{r}_common' for k in ('hh', 'ws') for r in ('sc', 'ach')]

OUTPUT_FILES = dict(
    state_month='state_month_metrics.csv',
    group='volatility_group_metrics.csv',
    current='volatility_2026_to_date.csv',
    by_volatility='coordination_by_volatility',
    saving='saving_by_volatility_common',
    scatter='coordination_vs_cap_value',
)

BY_VOLATILITY_HEIGHT_IN = 3.6
SAVING_HEIGHT_IN = 3.6
SAVING_YMAX = 5000.0
SCATTER_HEIGHT_IN = 3.0
DOLLAR = r'\$'


# ── Volatility measures ───────────────────────────────────────────────────────

def load_nem_prices(data_root: Path, years=PRICE_YEARS) -> dict[str, pd.Series]:
    """State -> RRP series ($/MWh, interval-end, AEST) over the years found.
    Missing years are reported and skipped; NaN prices are kept."""
    frames = []
    for year in years:
        path = (data_root / 'data' / 'clean' / 'nem_prices'
                / f'{year}_nem_prices.parquet')
        if not path.exists():
            print(f"  NOTE: no NEM price file for {year}, skipping.")
            continue
        df = pd.read_parquet(path)
        if 'datetime_aest' in df.columns:
            df = df.set_index('datetime_aest')
        if df.index.tz is None:
            df.index = df.index.tz_localize('Etc/GMT-10')
        frames.append(df[list(fe.RRP_COL.values())])
    if not frames:
        raise FileNotFoundError(
            f"No NEM price files under {data_root / 'data' / 'clean' / 'nem_prices'}")
    prices = pd.concat(frames)
    prices = prices[~prices.index.duplicated(keep='first')].sort_index()
    return {state: prices[col] for state, col in fe.RRP_COL.items()}


def _daily_spread(values) -> float:
    """Mean of the highest SPREAD_HOURS hours less the mean of the lowest, or
    NaN for a day with too few valid intervals."""
    v = np.asarray(values, float)
    if len(v) < MIN_DAY_INTERVALS:
        return np.nan
    v = np.sort(v)
    k = SPREAD_HOURS * INTERVALS_PER_HOUR
    return float(v[-k:].mean() - v[:k].mean())


def state_month_volatility(rrp: pd.Series, state: str) -> pd.DataFrame:
    """Cap value and daily spread for each local month of one state's RRP.

    Months and days are the state's local calendar months and dates at the
    interval-start instant, as in the monthly bills. NaN prices are left out.
    Columns: state, year, month, cap_value, daily_spread, n_intervals, n_days,
    days_in_month.
    """
    s = rrp.dropna()
    start = (s.index - pd.Timedelta(minutes=INTERVAL_MIN)).tz_convert(
        STATE_TIMEZONE[state])
    df = pd.DataFrame({'rrp': s.to_numpy(), 'date': start.normalize(),
                       'year': start.year, 'month': start.month})
    df['cap'] = np.maximum(df['rrp'] - CAP_THRESHOLD, 0.0)
    monthly = df.groupby(['year', 'month']).agg(
        cap_value=('cap', 'mean'), n_intervals=('rrp', 'size'))

    daily = df.groupby('date')['rrp'].agg(_daily_spread).dropna()
    by_month = daily.groupby([daily.index.year, daily.index.month])
    monthly['daily_spread'] = by_month.mean().rename_axis(['year', 'month'])
    monthly['n_days'] = by_month.size().rename_axis(['year', 'month'])
    out = monthly.dropna(subset=['daily_spread']).reset_index()
    out['n_days'] = out['n_days'].astype(int)
    out['days_in_month'] = [
        pd.Period(year=int(y), month=int(m), freq='M').days_in_month
        for y, m in zip(out['year'], out['month'])]
    out.insert(0, 'state', state)
    return out[['state', 'year', 'month', 'cap_value', 'daily_spread',
                'n_intervals', 'n_days', 'days_in_month']]


def quartile_bounds(cap_values) -> tuple[float, float, float]:
    """The cap values at the first, second and third quartile of the modelled
    state-months."""
    return tuple(float(b) for b in
                 np.quantile(np.asarray(cap_values, float), [.25, .5, .75]))


def assign_quartile(cap_values, bounds) -> np.ndarray:
    """q1 up to and including the first boundary, q2 above it up to and
    including the second, q3 up to and including the third, q4 above. A value
    on a boundary belongs to the lower group, so that state-months with no cap
    value, which can be more than a quarter of them, all fall in q1."""
    idx = np.digitize(np.asarray(cap_values, float), bounds, right=True)
    return np.array(QUARTILES, dtype=object)[idx]


def volatility_table(prices: dict[str, pd.Series]
                     ) -> tuple[pd.DataFrame, pd.DataFrame, tuple[float, ...]]:
    """(modelled, current-year, bounds): state-month volatility for 2021-2025
    with quartiles, and for 2026 to date with the quartile each falls in under
    the 2021-2025 boundaries and its percentile among the modelled months."""
    vol = pd.concat([state_month_volatility(prices[s], s) for s in prices],
                    ignore_index=True)
    modelled = vol[vol['year'].isin(MODEL_YEARS)].copy()
    bounds = quartile_bounds(modelled['cap_value'])
    modelled['quartile'] = assign_quartile(modelled['cap_value'], bounds)
    if bounds[0] <= 0:
        print(f"  WARNING: {int((modelled['cap_value'] <= 0).sum())} of "
              f"{len(modelled)} modelled state-months have no cap value, so "
              f"q1 is the months with none and the groups are "
              f"unequal: {modelled['quartile'].value_counts().to_dict()}")
    short = modelled[modelled['n_days'] < modelled['days_in_month']]
    if len(short):
        print(f"  WARNING: {len(short)} modelled state-month(s) have days "
              f"missing from the daily spread (missing prices):")
        print(short[['state', 'year', 'month', 'n_days', 'days_in_month']]
              .to_string(index=False))

    current = vol[vol['year'] == CURRENT_YEAR].copy()
    current['quartile'] = assign_quartile(current['cap_value'], bounds)
    ref = np.sort(modelled['cap_value'].to_numpy())
    current['cap_percentile'] = 100.0 * np.searchsorted(
        ref, current['cap_value'].to_numpy(), side='left') / len(ref)
    current['partial'] = current['n_days'] < current['days_in_month']
    return (modelled.reset_index(drop=True),
            current.reset_index(drop=True), bounds)


# ── Runs and monthly savings ──────────────────────────────────────────────────

def offer_plans() -> list[dict]:
    """One entry per offer: name, colour, and the run directories.

    'amber_solar' is the Amber Solar only run (the common baseline),
    'solar' is the offer's own Solar only run, 'sc' its no-VPP
    self-consumption run (sc_no_vpp_dir), 'ach' the achievable basis's
    optimised run and 'pf' the perfect foresight optimised run, all from
    financial_analysis.plans_for_basis so that run names have one home."""
    pf = {p['sc_mode']: p for p in fa.plans_for_basis('perfect_foresight')}
    plans = []
    for ach in fa.plans_for_basis('achievable'):
        mode = ach['sc_mode']
        plans.append(dict(
            offer=fa._offer_name(ach), mode=mode, color=ach['color'],
            excluded_states=ach['excluded_states'],
            runs=dict(solar=ach['zero_batt_dir'] or fa.DEFAULT_BASELINE_RUN,
                      sc=ach['sc_no_vpp_dir'], ach=ach['lp_dir'],
                      pf=pf[mode]['lp_dir'],
                      amber_solar=fa.DEFAULT_BASELINE_RUN)))
    return plans


def scenario_name(role: str, state: str, year: int, battery_kwh: int) -> str:
    base = f'{state}_{year}'
    return base if role in ('solar', 'amber_solar') else f'{base}_{battery_kwh}kWh'


def bills_path(runs_root: Path, run: str, scenario: str, month: int) -> Path:
    return runs_root / run / 'scenarios' / scenario / f'bills_{month:02d}.csv'


def check_runs(runs_root: Path, plans: list[dict],
               scenarios: list[tuple[str, int]], battery_kwh: int
               ) -> tuple[dict, list[str]]:
    """Which (offer, state, year) have monthly bills in all its runs.

    A run and scenario with no bills_MM.csv at all is listed and the offer is
    left out for that state-year. Returns ({(offer, state, year): bool},
    list of lines for the report)."""
    usable, missing = {}, []
    for plan in plans:
        for state, year in scenarios:
            if state in plan['excluded_states']:
                continue
            ok = True
            for role in RUN_ROLES:
                scen = scenario_name(role, state, year, battery_kwh)
                run = plan['runs'][role]
                if role == 'amber_solar' and run == plan['runs']['solar']:
                    continue
                if not any(bills_path(runs_root, run, scen, m).exists()
                           for m in range(1, 13)):
                    ok = False
                    missing.append(f"{plan['offer']}: run '{run}' has no "
                                   f"monthly bills for {scen}")
            usable[(plan['offer'], state, year)] = ok
    return usable, missing


def load_month(runs_root: Path, run: str, scenario: str, month: int
               ) -> pd.DataFrame | None:
    """household_total and net_wholesale by site_id for one month, or None."""
    f = bills_path(runs_root, run, scenario, month)
    if not f.exists():
        return None
    df = pd.read_csv(f, usecols=['site_id', 'household_total',
                                 'wholesale_usage_cost',
                                 'wholesale_feedin_credit'])
    df['net_wholesale'] = (df['wholesale_usage_cost']
                           + df['wholesale_feedin_credit'])
    df = df[~df['site_id'].isin(EXCLUDED_SITES)]
    return df.set_index('site_id')[['household_total', 'net_wholesale']]


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    w = weights.reindex(values.index)
    keep = w.notna()
    if not keep.any() or w[keep].sum() == 0:
        return np.nan
    return float(np.average(values[keep].to_numpy(), weights=w[keep].to_numpy()))


def month_savings(frames: dict[str, pd.DataFrame], site_weight: pd.Series
                  ) -> dict | None:
    """PV-size-weighted mean per site of the household and wholesale saving
    for self-consumption, achievable and perfect foresight against the offer's
    own Solar only run, and for self-consumption and achievable against the
    Amber Solar only run (_common), over the sites present in all runs with a
    weight. frames is keyed by RUN_ROLES."""
    sites = frames['solar'].index
    for role in RUN_ROLES[1:]:
        sites = sites.intersection(frames[role].index)
    sites = sites.intersection(site_weight.index)
    if len(sites) == 0:
        return None
    out = {'n_sites': int(len(sites))}
    solar = frames['solar'].loc[sites]
    for role in SAVINGS:
        run = frames[role].loc[sites]
        out[f'hh_{role}'] = weighted_mean(
            solar['household_total'] - run['household_total'], site_weight)
        out[f'ws_{role}'] = weighted_mean(
            solar['net_wholesale'] - run['net_wholesale'], site_weight)
    amber = frames['amber_solar'].loc[sites]
    for role in ('sc', 'ach'):
        run = frames[role].loc[sites]
        out[f'hh_{role}_common'] = weighted_mean(
            amber['household_total'] - run['household_total'], site_weight)
        out[f'ws_{role}_common'] = weighted_mean(
            amber['net_wholesale'] - run['net_wholesale'], site_weight)
    return out


def state_month_rows(runs_root: Path, plans: list[dict], usable: dict,
                     modelled: pd.DataFrame, scenarios, site_weights: dict,
                     battery_kwh: int) -> pd.DataFrame:
    """One row per offer and modelled state-month with its volatility
    measures, quartile and monthly savings."""
    vol = modelled.set_index(['state', 'year', 'month'])
    rows = []
    for state, year in scenarios:
        sw = site_weights[(state, year)]
        for plan in plans:
            if not usable.get((plan['offer'], state, year)):
                continue
            for month in range(1, 13):
                if (state, year, month) not in vol.index:
                    continue
                frames = {}
                for role in RUN_ROLES:
                    df = load_month(runs_root, plan['runs'][role],
                                    scenario_name(role, state, year,
                                                  battery_kwh), month)
                    if df is None:
                        break
                    frames[role] = df
                else:
                    sav = month_savings(frames, sw)
                    if sav is None:
                        continue
                    v = vol.loc[(state, year, month)]
                    rows.append(dict(
                        state=state, year=year, month=month,
                        offer=plan['offer'], cap_value=v['cap_value'],
                        daily_spread=v['daily_spread'], quartile=v['quartile'],
                        **sav,
                        low_site_count=sav['n_sites'] < LOW_SITE_COUNT))
    cols = ['state', 'year', 'month', 'offer', 'cap_value', 'daily_spread',
            'quartile', 'n_sites', 'low_site_count',
            *[f'{k}_{r}' for k in ('hh', 'ws') for r in SAVINGS],
            *COMMON_COLS]
    return pd.DataFrame(rows, columns=cols)


# ── Group metrics ─────────────────────────────────────────────────────────────

def ratio(numerator: float, denominator: float) -> float:
    """numerator / denominator, NaN where the denominator is not positive
    (a ratio of sums is only a relative value or a capture share for a
    positive reference)."""
    return float(numerator / denominator) if denominator > 0 else np.nan


def group_metrics(state_months: pd.DataFrame, bounds) -> pd.DataFrame:
    """One row per offer and quartile, from sums over the group's state-months.

    rel_value_*      sum achievable / sum self-consumption saving
    capture_*        sum (achievable - self-consumption) /
                     sum (perfect foresight - self-consumption)
    share_coord_*    the group's share of the offer's total 2021-2025
                     coordination value (achievable - self-consumption)
    ann_*_common     12 x the group's mean monthly saving against the Amber
                     Solar only run
    """
    sm = state_months.dropna(subset=[f'{k}_{r}' for k in ('hh', 'ws')
                                     for r in SAVINGS])
    edges = dict(zip(QUARTILES, zip((-np.inf, *bounds), (*bounds, np.inf))))
    rows = []
    for offer, off in sm.groupby('offer', sort=False):
        total = {k: (off[f'{k}_ach'] - off[f'{k}_sc']).sum()
                 for k in ('hh', 'ws')}
        for quartile in QUARTILES:
            g = off[off['quartile'] == quartile]
            row = dict(offer=offer, quartile=quartile,
                       cap_lower=edges[quartile][0], cap_upper=edges[quartile][1],
                       n_state_months=len(g),
                       n_low_site_count=int(g['low_site_count'].sum()),
                       mean_cap_value=g['cap_value'].mean(),
                       mean_daily_spread=g['daily_spread'].mean())
            for k in ('hh', 'ws'):
                s = {r: g[f'{k}_{r}'].sum() for r in SAVINGS}
                coord = s['ach'] - s['sc']
                row.update({
                    f'sum_{k}_sc': s['sc'], f'sum_{k}_ach': s['ach'],
                    f'sum_{k}_pf': s['pf'],
                    f'rel_value_{k}': ratio(s['ach'], s['sc']),
                    f'capture_{k}': ratio(coord, s['pf'] - s['sc']),
                    f'share_coord_{k}': ratio(coord, total[k]),
                })
            for c in COMMON_COLS:
                row[f'ann_{c}'] = 12.0 * g[c].mean() if len(g) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


# ── Charts ────────────────────────────────────────────────────────────────────

def _style() -> None:
    """financial_analysis sets global rcParams on import; start from the
    defaults so only the thesis style applies."""
    matplotlib.rcdefaults()
    ts.apply_style()


def _offer_colours(plans: list[dict]) -> dict[str, str]:
    return {p['offer']: p['color'] for p in plans}


def _fmt_bound(v: float) -> str:
    return f'{v:.1f}' if v >= 1 else f'{v:.2g}'


def quartile_tick_labels(bounds) -> list[str]:
    """'< b1', 'b1 - b2', 'b2 - b3', '> b3' from the 2021-2025 boundaries."""
    b = [_fmt_bound(v) for v in bounds]
    first = '0' if bounds[0] <= 0 else f'< {b[0]}'
    return [first, f'{b[0]} - {b[1]}', f'{b[1]} - {b[2]}', f'> {b[2]}']


def title_suffix(battery_kwh: int) -> str:
    return f', imperfect foresight (achievable), {battery_kwh} kWh'


def notes_text(state_months: pd.DataFrame, baseline: str,
               plans: list[dict] | None = None) -> str:
    """Data notes drawn top left of each panel (presentation versions;
    --no-notes leaves them off). baseline: 'own' or 'amber'. With plans, a
    line for each offer left out of some states (EnergyAustralia: NSW only)."""
    state_list = sorted(state_months['state'].unique())
    years = sorted(state_months['year'].unique())
    base = ('Savings vs Amber Solar only' if baseline == 'amber'
            else "Savings vs each offer's own Solar only")
    lines = [f"National ({', '.join(state_list)}), {years[0]}-{years[-1]}",
             'Weighted by PV size, state-months pooled equally', base]
    if plans:
        modes = {p['mode'] for p in plans}
        lines += fa.restricted_offer_notes(
            state_list, [p for p in fa._BASE_PLANS if p['sc_mode'] in modes])
    return '\n'.join(lines)


def _draw_notes(ax, notes: str | None) -> None:
    if notes:
        ax.text(0.008, 0.97, notes, transform=ax.transAxes, va='top',
                ha='left', fontsize=ts.NOTE_SIZE, color='0.3',
                bbox=ts.LABEL_BOX, zorder=6)


def _fit_title(fig, ax, suffix: str) -> None:
    """Break the title before its suffix when it is wider than the panel."""
    fig.canvas.draw()
    t = ax.title
    if t.get_window_extent().width > ax.get_window_extent().width:
        text = t.get_text()
        if suffix in text:
            t.set_text(text.replace(suffix, ',\n' + suffix[2:]))


def _bar_labels(ax, xs, vals, text) -> None:
    """Value labels above the bar (below a negative one)."""
    for x, v in zip(xs, vals):
        if np.isnan(v):
            continue
        up = v >= 0
        ax.annotate(text(v), (x, v), xytext=(0, 1.5 if up else -1.5),
                    textcoords='offset points', ha='center',
                    va='bottom' if up else 'top',
                    fontsize=ts.ANNOTATION_SIZE, color='black')


def _stacked_bar_figure(panels, series, bounds, height_in, label_text,
                        ylabel, legend_cols, out_path, ymax=None, notes=None,
                        suffix=''):
    """Two full-width panels, one above the other, of bars grouped by cap
    value quartile. panels: (title, key, reference line or None); series:
    (legend label, colour, hatch or None, {panel key: four values}).
    label_text formats the value labels (None for no labels); ymax fixes the
    top of the y axis, with a tick every 1,000."""
    _style()
    fig = ts.new_figure(height_in)
    axes = fig.subplots(len(panels), 1)
    n = len(series)
    width = 0.72 / n
    for ax, (title, key, ref) in zip(axes, panels):
        allv = []
        for i, (_, colour, hatch, vals_by_key) in enumerate(series):
            vals = np.asarray(vals_by_key[key], float)
            xs = np.arange(len(QUARTILES)) + (i - (n - 1) / 2) * width
            ax.bar(xs, np.nan_to_num(vals), width=width, color=colour,
                   linewidth=0, zorder=3, hatch=hatch,
                   edgecolor='white' if hatch else None)
            if label_text is not None:
                _bar_labels(ax, xs, vals, label_text)
            allv.extend(vals[~np.isnan(vals)])
        if ref is not None:
            ax.axhline(ref, color=ts.REFERENCE_COLOUR, ls='--',
                       lw=ts.REFERENCE_LW, zorder=2)
        lo = min(0.0, min(allv, default=0.0))
        hi = max(ref or 0.0, max(allv, default=1.0))
        if ymax is not None:
            if hi > ymax:
                print(f"  NOTE: {title} reaches {hi:,.0f}, above the "
                      f"{ymax:,.0f} axis maximum; axis extended.")
                ymax = float(np.ceil(hi / 1000) * 1000)
            ax.set_ylim(lo, ymax)
            ax.set_yticks(np.arange(np.floor(lo / 1000) * 1000, ymax + 1, 1000))
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
        else:
            pad = 0.14 * (hi - lo)
            ax.set_ylim(lo - (pad if lo < 0 else 0), hi + pad)
        ax.set_title(title)
        _draw_notes(ax, notes)
        ax.set_xticks(range(len(QUARTILES)), quartile_tick_labels(bounds))
        ax.tick_params(axis='x', length=0)
        ax.grid(axis='x', visible=False)
        ax.set_xlabel(f'{DOLLAR}300 cap value quartiles ({DOLLAR}/MWh)')
        if ylabel:
            ax.set_ylabel(ylabel)
    handles = [mpatches.Patch(facecolor=c, label=lab, hatch=h,
                              edgecolor='white' if h else c, linewidth=0)
               for lab, c, h, _ in series]
    fig.legend(handles=handles, loc='outside lower center', ncol=legend_cols,
               frameon=False)
    if suffix:
        for ax in axes:
            _fit_title(fig, ax, suffix)
    ts.save_figure(fig, out_path)


def draw_by_volatility(group: pd.DataFrame, plans: list[dict], bounds,
                       battery_kwh: int, out_path: Path,
                       notes: str | None = None) -> None:
    """Relative value of coordination by cap value quartile, one bar per
    offer; household bill above net wholesale saving."""
    colours = _offer_colours(plans)
    present = set(group['offer'])
    offers = [p['offer'] for p in plans if p['offer'] in present]
    series = []
    for offer in offers:
        g = group[group['offer'] == offer].set_index('quartile')
        series.append((offer, colours[offer], None, {
            k: [g[f'rel_value_{k}'].get(q, np.nan) for q in QUARTILES]
            for k in ('hh', 'ws')}))
    suffix = title_suffix(battery_kwh)
    panels = [('Relative value of coordination by volatility, household bill'
               + suffix, 'hh', 1.0),
              ('Relative value of coordination by volatility, net wholesale '
               'saving' + suffix, 'ws', 1.0)]
    _stacked_bar_figure(panels, series, bounds, BY_VOLATILITY_HEIGHT_IN,
                        lambda v: f'{v:.1f}', None, len(series), out_path,
                        notes=notes, suffix=suffix)


def draw_saving_common(group: pd.DataFrame, plans: list[dict], bounds,
                       battery_kwh: int, out_path: Path,
                       notes: str | None = None) -> None:
    """Mean annual saving against the Amber Solar only run by cap value
    quartile: Amber self-consumption (hatched) and each offer's achievable
    optimised run; household bill above net wholesale saving."""
    colours = _offer_colours(plans)
    present = set(group['offer'])
    offers = [p['offer'] for p in plans if p['offer'] in present]
    series = []
    amber = group[group['offer'] == 'Amber'].set_index('quartile')
    series.append(('Amber self-consumption', colours['Amber'], '///', {
        k: [amber[f'ann_{k}_sc_common'].get(q, np.nan) for q in QUARTILES]
        for k in ('hh', 'ws')}))
    for offer in offers:
        g = group[group['offer'] == offer].set_index('quartile')
        series.append((offer, colours[offer], None, {
            k: [g[f'ann_{k}_ach_common'].get(q, np.nan) for q in QUARTILES]
            for k in ('hh', 'ws')}))
    suffix = title_suffix(battery_kwh)
    panels = [('Mean annual household bill saving by volatility' + suffix,
               'hh', None),
              ('Mean annual wholesale saving by volatility' + suffix,
               'ws', None)]
    _stacked_bar_figure(panels, series, bounds, SAVING_HEIGHT_IN,
                        None, f'Mean annual saving ({DOLLAR}/site)', 3,
                        out_path, ymax=SAVING_YMAX, notes=notes, suffix=suffix)


def draw_scatter(state_months: pd.DataFrame, current: pd.DataFrame,
                 bounds, offer: str, battery_kwh: int,
                 out_path: Path, notes: str | None = None) -> None:
    """The offer's monthly household coordination value (achievable optimised
    less self-consumption, $ per site) against cap value on a log axis, one
    marker per state-month, coloured by state. State-months with no cap value
    sit on the left boundary as open triangles. 2026-to-date state-months are
    squares of their state's colour at their cap value, halfway between zero
    and the lower y limit."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter
    _style()
    sm = state_months[state_months['offer'] == offer].copy()
    states = sorted(sm['state'].unique())
    colour = dict(zip(states, ts.CATEGORICAL_COLOURS))
    positive = pd.concat([sm['cap_value'], current['cap_value']])
    positive = positive[positive > 0]
    xmin = 10 ** np.floor(np.log10(positive.min())) if len(positive) else 0.1
    xmax = 10 ** np.ceil(np.log10(max(positive.max(), bounds[-1])))
    y = sm['hh_ach'] - sm['hh_sc']
    ytop = max(float(y.max()), 1.0)
    ybottom = min(float(y.min()), 0.0)
    ybottom = ybottom - 0.12 * (ytop - ybottom)
    ytop = ytop + 0.05 * (ytop - ybottom)

    fig = ts.new_figure(SCATTER_HEIGHT_IN)
    ax = fig.add_subplot()
    n_zero = int((sm['cap_value'] <= 0).sum())
    for state in states:
        m = sm['state'] == state
        pos = m & (sm['cap_value'] > 0)
        zero = m & (sm['cap_value'] <= 0)
        ax.scatter(sm.loc[pos, 'cap_value'], y[pos], s=14, color=colour[state],
                   alpha=0.8, edgecolor='white', linewidth=0.5, zorder=3)
        ax.scatter(np.full(int(zero.sum()), xmin), y[zero], s=14, marker='<',
                   facecolor='none', edgecolor=colour[state], linewidth=0.7,
                   zorder=3, clip_on=False)
        cur = current[current['state'] == state]
        ax.scatter(np.maximum(cur['cap_value'], xmin),
                   np.full(len(cur), ybottom / 2), s=14, marker='s',
                   color=colour[state], alpha=0.8, edgecolor='white',
                   linewidth=0.5, zorder=3)
    ax.set_xscale('log')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ybottom, ytop)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.axhline(0, color='0.65', lw=0.5, zorder=1)
    for b in bounds:
        ax.axvline(b, color=ts.REFERENCE_COLOUR, ls='--', lw=ts.REFERENCE_LW,
                   zorder=2)
    ax.set_title(f'{offer} household bill coordination value against cap '
                 f'value, imperfect foresight (achievable), {battery_kwh} kWh')
    _draw_notes(ax, notes)
    ax.set_xlabel(f'Cap value ({DOLLAR}/MWh)')
    ax.set_ylabel(f'Coordination value ({DOLLAR} per site per month)')
    handles = [Line2D([], [], ls='none', marker='o', ms=4, color=colour[s],
                      markeredgecolor='white', label=s) for s in states]
    handles.append(Line2D([], [], color=ts.REFERENCE_COLOUR, ls='--',
                          lw=ts.REFERENCE_LW, label='Quartile boundary'))
    handles.append(Line2D([], [], ls='none', marker='s', ms=4, color='0.45',
                          markeredgecolor='white', label='2026 to date'))
    if n_zero:
        handles.append(Line2D([], [], ls='none', marker='<', ms=4,
                              markerfacecolor='none', color='0.3',
                              label=f'No cap value (n = {n_zero})'))
    fig.legend(handles=handles, loc='outside lower center',
               ncol=len(handles), frameon=False)
    ts.save_figure(fig, out_path)


# ── Values for the text ───────────────────────────────────────────────────────

def print_text_values(group: pd.DataFrame, current: pd.DataFrame,
                      bounds) -> None:
    pd.set_option('display.width', 220, 'display.max_columns', 40,
                  'display.float_format', '{:.2f}'.format)
    print("\nCap value quartile boundaries: "
          + ', '.join(f'{b:.2f}' for b in bounds) + " $/MWh")
    cols = ['offer', 'quartile', 'n_state_months', 'rel_value_hh',
            'capture_hh', 'rel_value_ws', 'capture_ws']
    g = group.copy()
    g['quartile'] = pd.Categorical(g['quartile'], QUARTILES, ordered=True)

    amber = g[g['offer'] == 'Amber'].sort_values('quartile')
    print("\nAmber: relative value and value capture by quartile")
    print(amber[cols].to_string(index=False))
    print("\nAmber: share of coordination value earned in each quartile")
    print(amber[['quartile', 'n_state_months', 'share_coord_hh',
                 'share_coord_ws']].to_string(index=False))
    print("\nTime-of-use and VPP offers: Q1 and Q4")
    others = g[(g['offer'] != 'Amber') & g['quartile'].isin(['q1', 'q4'])]
    print(others.sort_values(['offer', 'quartile'])[cols].to_string(index=False))

    print("\n2026 to date against the 2021-2025 quartiles (state-months)")
    if current.empty:
        print("  no 2026 prices")
        return
    cnt = (current.groupby(['state', 'quartile']).size().unstack(fill_value=0)
           .reindex(columns=list(QUARTILES), fill_value=0))
    print(cnt.to_string())
    print(current[['state', 'month', 'cap_value', 'daily_spread', 'quartile',
                   'cap_percentile', 'partial']].to_string(index=False))


# ── Main ──────────────────────────────────────────────────────────────────────

def _read_outputs(out_dir: Path):
    sm = pd.read_csv(out_dir / OUTPUT_FILES['state_month'])
    group = pd.read_csv(out_dir / OUTPUT_FILES['group'])
    current = pd.read_csv(out_dir / OUTPUT_FILES['current'])
    first = group[group['offer'] == group['offer'].iloc[0]].set_index('quartile')
    bounds = tuple(float(first.loc[q, 'cap_upper']) for q in QUARTILES[:3])
    return sm, group, current, bounds


def _draw_all(sm, group, current, bounds, plans, battery_kwh,
              out_dir: Path, notes: bool = True) -> None:
    own = notes_text(sm, 'own', plans) if notes else None
    common = notes_text(sm, 'amber', plans) if notes else None
    # The scatter shows Amber alone, so it carries no offer availability line.
    amber_only = notes_text(sm, 'own') if notes else None
    draw_by_volatility(group, plans, bounds, battery_kwh,
                       out_dir / OUTPUT_FILES['by_volatility'], own)
    draw_saving_common(group, plans, bounds, battery_kwh,
                       out_dir / OUTPUT_FILES['saving'], common)
    draw_scatter(sm, current, bounds, 'Amber', battery_kwh,
                 out_dir / OUTPUT_FILES['scatter'], amber_only)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--data-root', type=Path, default=Path('.'),
                    help='Project root holding data/output/runs, data/clean '
                         'and data/processed (default: cwd)')
    ap.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    ap.add_argument('--weights-file', type=Path, default=WEIGHTS_FILE)
    ap.add_argument('--battery-kwh', type=int, default=BATTERY_KWH)
    ap.add_argument('--states', nargs='+', default=None)
    ap.add_argument('--no-notes', dest='notes', action='store_false',
                    help='Leave the data notes off the charts, for the thesis, '
                         'where they go in the caption.')
    ap.add_argument('--from-csv', action='store_true',
                    help='Redraw the charts and reprint the text values from '
                         'the saved CSVs; needs no data root.')
    args = ap.parse_args(argv)
    plans = offer_plans()
    out_dir = args.output_dir

    if args.from_csv:
        sm, group, current, bounds = _read_outputs(out_dir)
        _draw_all(sm, group, current, bounds, plans, args.battery_kwh,
                  out_dir, args.notes)
        print_text_values(group, current, bounds)
        return 0

    runs_root = args.data_root / 'data' / 'output' / 'runs'
    prices = load_nem_prices(args.data_root)
    modelled, current, bounds = volatility_table(prices)
    print(f"Modelled state-months: {len(modelled)}; 2026 to date: "
          f"{len(current)}; cap value quartile boundaries "
          + ", ".join(f"{b:.2f}" for b in bounds) + " $/MWh")

    scenarios = [(s, int(y)) for s, y in fa.discover_scenarios(args.data_root)
                 if int(y) in MODEL_YEARS
                 and (args.states is None or s in args.states)]
    if not scenarios:
        print(f"ERROR: no 2021-2025 scenarios under {runs_root}",
              file=sys.stderr)
        return 1
    usable, missing = check_runs(runs_root, plans, scenarios,
                                 args.battery_kwh)
    if missing:
        print("\nRuns without monthly bills, left out:")
        for line in missing:
            print(f"  {line}")
    if not any(usable.values()):
        print("ERROR: no offer has monthly bills in all their runs.",
              file=sys.stderr)
        return 1

    weights_cfg = fa.load_weights_config(args.weights_file)
    site_weights, dc_ac_cache, meta_cache = {}, {}, {}
    for state, year in scenarios:
        site_weights[(state, year)] = fe.compute_site_weight_for_scenario(
            args.data_root, weights_cfg, state, str(year),
            dc_ac_cache, meta_cache)

    sm = state_month_rows(runs_root, plans, usable, modelled, scenarios,
                          site_weights, args.battery_kwh)
    if sm.empty:
        print("ERROR: no state-month savings could be computed.",
              file=sys.stderr)
        return 1
    print("\nState-months with savings, by offer:")
    print(sm.groupby('offer', sort=False).size().to_string())
    low = sm[sm['low_site_count']].groupby(['state', 'year']).size()
    if len(low):
        print(f"\nState-years with fewer than {LOW_SITE_COUNT} sites "
              f"(flagged, included):\n{low.to_string()}")

    group = group_metrics(sm, bounds)
    out_dir.mkdir(parents=True, exist_ok=True)
    sm.to_csv(out_dir / OUTPUT_FILES['state_month'], index=False)
    group.to_csv(out_dir / OUTPUT_FILES['group'], index=False)
    current.to_csv(out_dir / OUTPUT_FILES['current'], index=False)
    print(f"\nWrote {len(sm)} state-month rows, {len(group)} group rows and "
          f"{len(current)} 2026 state-months to {out_dir}")

    _draw_all(sm, group, current, bounds, plans, args.battery_kwh, out_dir,
              args.notes)
    print_text_values(group, current, bounds)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
