"""
prudential_analysis.py

AEMO prudential requirement (maximum credit limit, MCL) per household for the
No CER, Solar Only, Self Consumption and Amber coordinated cohorts, per state
and nationally, and for a retailer of --households such households, with the
annual cost of a bank guarantee for it. Design, formulas and sources:
docs/prudential_spec.md.

Two methods, both on AEMO's 2026 seasonal parameters (read from the MCL
calculators in docs/references/aemo_prudential/):
  calculator  AEMO's 2026 MCL calculator: TOSL 35 days, participant risk
              adjustment factor (PRAF) from the 288-interval daily profile.
  clp_v10     Credit Limit Procedures v10.0: TOSL 21 days, five time-of-day
              segments, regional volatility factor applied to every segment.

Simulated outstandings: each cohort's settlement amounts at historical prices
under the shortened settlement calendar (billing week Sunday-Saturday, paid 12
days after the week ends), chained across years. Credit support required on a
day is max(MCL, PM + outstandings): the MCL, and enough above the PM to keep
outstandings below the trading limit (CLP cl 12). Calculator method, MCL and
PM per season of each year.

Outputs (analysis/output/prudential/):
  prudential_state_year.csv   per cohort, state, year, season, method; per household
  prudential_summary.csv      years pooled; states and national; seasons and annual;
                              per household and per retailer (AEMO rounding)
  prudential_outstandings.csv simulated outstandings per household against the MCL
  prudential_credit_support.csv  credit support required per season and year, annual
                              per year and the mean over years; per household and retailer
  prudential_credit_support_daily.csv  daily MCL, PM and outstanding, calculator method
  prudential_credit_support_{REGION}.png  daily credit support required by cohort
  prudential_credit_support_max.png       maximum credit support required over the
                                          period: largest MCL + trading limit above it

Charts follow docs/thesis_figure_spec.md and are drawn from the CSVs, so
--charts-only redraws them without the flows. --no-notes leaves the data notes
off, for the thesis.

Run from repo root:
  python3 analysis/scripts/prudential_analysis.py --data-root "/Volumes/Manual/Thesis Project"
  python3 analysis/scripts/prudential_analysis.py --charts-only --no-notes
"""
import argparse
import math
import sys
import textwrap
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import financial_analysis as fa
import savings_bar_figure as sbf
import thesis_style as ts
from cap_hedge_analysis import _name, _years_label, build_scenarios
from financial_analysis import load_weights_config
from flow_exposure_analysis import (
    RRP_COL, compute_site_weight_for_scenario, discover_scenarios, get_prices, load_flows,
)
from retailer_outcome_analysis import TARIFF_DIR, read_bills, run_tariff_ids

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from model.components.billing import GST_RATE  # noqa: E402
from model.core.tariff import load_network_tariff  # noqa: E402

REFERENCE_DIR = REPO_ROOT / 'docs' / 'references' / 'aemo_prudential'

SEASONS = ('summer', 'winter', 'shoulder')
SEASON_MONTHS = {'summer': (12, 1, 2, 3), 'winter': (4, 5, 6, 7, 8),
                 'shoulder': (9, 10, 11)}
SEASON_DAYS = {'summer': 121, 'winter': 153, 'shoulder': 91}      # non-leap year
CALCULATOR_FILES = {s: f'MCL Calculator {s.capitalize()}.xlsx' for s in SEASONS}
CALCULATOR_STATES = ('NSW', 'QLD', 'SA', 'TAS', 'VIC')            # RegionalData row order

TI_PER_DAY = 288
CALC_TOSL = 35            # days, AEMO 2026 MCL calculator (MCL!E7)
CLP_TOSL = 21             # days, CLP v10.0 cl 4.3.1(b)
TRP = 7                   # days, CLP v10.0 cl 4.3.1(b); calculator MCL!F7
PRAF_DEFAULT_LOAD = 1.05  # calculator ParticipantData!L9 with no profile
PRAF_DEFAULT_GEN = 0.95   # calculator ParticipantData!L12 with no profile
TOD_SEGMENTS = (('EM', 0, 6), ('MP', 6, 10), ('MD', 10, 16), ('AP', 16, 20),
                ('LE', 20, 24))            # market time hours, CLP v10.0 cl 8
PAYMENT_LAG_DAYS = 12     # billing week end (Saturday) to payment (Thursday)
DEFAULT_HOUSEHOLDS = 50_000
DEFAULT_GUARANTEE_RATE = 0.025   # per year; ACIL Allen DMO 2026-27 final, p. 79
METHODS = ('calculator', 'clp_v10')
# The coordinated cohort is full_run_rolling_amber, the achievable basis of
# financial_analysis.RUN_BASES.
BASIS_SUFFIX = ', imperfect foresight (achievable)'
METHOD_TITLES = {'calculator': "AEMO's 2026 MCL calculator",
                 'clp_v10': 'CLP v10.0'}

# Cohort key -> (label, run providing the flows, has_battery).
METHOD_NOTES = {'calculator': 'AEMO 2026 MCL calculator method (TOSL 35 days, PRAF)',
                'clp_v10': ('CLP v10.0 method (TOSL 21 days, five ToD segments; PR,ToD from '
                            "the calculator's price profile, regional VF)")}

COHORTS = {
    'no_cer': ('No CER', 'full_run_no_cer_amber', False),
    'solar_only': ('Solar Only', 'full_run_solar_only_amber', False),
    'self_consumption': ('Self Consumption', 'full_run_self_consumption_amber', True),
    'coordinated': ('Amber (coordinated)', 'full_run_rolling_amber', True),
}

REGIONS = ('NSW', 'QLD', 'SA', 'VIC', 'National')


class CheckError(RuntimeError):
    """An input fails a consistency check; the analysis stops."""


# ── AEMO regional parameters ──────────────────────────────────────────────────

@dataclass(frozen=True)
class RegionalParams:
    """One region and season of AEMO's 2026 parameters. Prices in $/MWh."""
    p: float               # average regional price, PR
    rlwp: float            # regional load-weighted price, RLWPR
    vf_osl: float          # outstandings limit volatility factor, VFOSLR
    vf_pm: float           # prudential margin volatility factor, VFPMR
    phh: np.ndarray        # price profile by trading interval, PTI,R (288)
    erl: np.ndarray        # regional load profile by trading interval, ERLTI,R (288)


def load_regional_params(ref_dir: Path = REFERENCE_DIR) -> dict[tuple[str, str], RegionalParams]:
    """(state, season) -> RegionalParams from AEMO's three 2026 MCL calculators.
    Checks the calculator's TOSL, TRP and GST against the constants here and
    each region's RLWPR against its own load and price profiles."""
    import openpyxl
    out = {}
    for season, fname in CALCULATOR_FILES.items():
        path = ref_dir / fname
        if not path.exists():
            raise CheckError(f'AEMO calculator not found: {path}')
        wb = openpyxl.load_workbook(path, data_only=True)

        def cells(name):
            sheet, ref = wb.defined_names[name].attr_text.split('!')
            return wb[sheet.strip("'")][ref.replace('$', '')]

        title = str(wb['RegionalData']['D2'].value)
        if season.capitalize() not in title or '2026' not in title:
            raise CheckError(f'{fname}: RegionalData title is {title!r}')
        for name, want in (('TOSL', CALC_TOSL), ('TRP', TRP), ('GST', GST_RATE)):
            if cells(name).value != want:
                raise CheckError(f'{fname}: {name} = {cells(name).value}, expected {want}')
        rd = wb['RegionalData']
        erl_row = next(r for r in range(1, rd.max_row + 1)
                       if rd.cell(r, 2).value == 'ERLTI,R')
        for i, state in enumerate(CALCULATOR_STATES):
            if state not in RRP_COL:
                continue
            if rd.cell(erl_row + i, 3).value != state:
                raise CheckError(f'{fname}: ERLTI,R row {erl_row + i} is not {state}')
            erl = np.array([rd.cell(erl_row + i, c).value for c in range(4, 4 + TI_PER_DAY)],
                           dtype=float)
            phh = np.array([c.value for c in cells(f'{state}_PHH')[0]], dtype=float)
            rp = RegionalParams(p=float(cells(f'{state}_P').value),
                                rlwp=float(cells(f'{state}_RLWP').value),
                                vf_osl=float(cells(f'{state}_VFOSL').value),
                                vf_pm=float(cells(f'{state}_VFPM').value), phh=phh, erl=erl)
            if len(phh) != TI_PER_DAY or np.isnan(phh).any() or np.isnan(erl).any():
                raise CheckError(f'{fname} {state}: incomplete profiles')
            if abs(np.dot(erl, phh) / erl.sum() - rp.rlwp) > 0.01:
                raise CheckError(f'{fname} {state}: RLWPR {rp.rlwp} does not match its '
                                 f'profiles ({np.dot(erl, phh) / erl.sum():.3f})')
            out[(state, season)] = rp
    return out


# ── MCL formulas ──────────────────────────────────────────────────────────────

def xl_round(x: float, digits: int) -> float:
    """Excel ROUND: half away from zero on the decimal representation."""
    q = Decimal(1).scaleb(-digits)
    d = Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP)
    return float(d)


def calculator_praf(profile: np.ndarray, rp: RegionalParams, default: float) -> float:
    """PRAF as the calculator computes it (ParticipantData!L9:N9 for load,
    L12:N12 for generation): ROUND(MAX(LWPR, LWPR^2), 2) with LWPR =
    ROUND(ROUND(profile-weighted PTI, 2) / RLWPR, 2); `default` with no energy."""
    s = float(np.sum(profile))
    if s == 0:
        return default
    lwp = xl_round(float(np.dot(profile, rp.phh)) / s, 2)
    lwpr = xl_round(lwp / rp.rlwp, 2)
    return xl_round(max(lwpr, lwpr * lwpr), 2)


def mcl_calculator(load: np.ndarray, gen: np.ndarray, rp: RegionalParams) -> dict:
    """OSL and unfloored PM, $, by the 2026 calculator (MCL!D16:D41, N16:N40,
    no reallocations, no PM offset). `load` and `gen`: debit and credit energy
    per trading interval of an average day, MWh (288 each)."""
    el, eg = float(np.sum(load)), float(np.sum(gen))
    prafl = calculator_praf(load, rp, PRAF_DEFAULT_LOAD)
    prafg = calculator_praf(gen, rp, PRAF_DEFAULT_GEN)
    return dict(el=el, eg=eg, prafl=prafl, prafg=prafg,
                **calculator_values(el, eg, prafl, prafg, rp.p, rp.vf_osl, rp.vf_pm))


def calculator_values(el: float, eg: float, prafl: float, prafg: float, p: float,
                      vf_osl: float, vf_pm: float, tosl: float = CALC_TOSL) -> dict:
    """OSL and unfloored PM, $, from daily debit and credit energy (MWh/day),
    their PRAFs and the regional price and volatility factors."""
    g = 1.0 + GST_RATE
    net_osl = (el * prafl - eg * prafg) * p * vf_osl * g
    net_pm = (el * prafl - eg * prafg) * p * vf_pm * g
    return dict(osl=max(net_osl * tosl, net_osl * tosl / vf_osl),
                pm_raw=max(net_pm * TRP, net_pm * TRP / vf_pm))


def tod_segment_of_ti() -> np.ndarray:
    """ToD segment index (0..4) of each trading interval 0..287 by its start time."""
    start_hour = np.arange(TI_PER_DAY) * 5 // 60
    seg = np.empty(TI_PER_DAY, dtype=int)
    for k, (_, h0, h1) in enumerate(TOD_SEGMENTS):
        seg[(start_hour >= h0) & (start_hour < h1)] = k
    return seg


def tod_prices(rp: RegionalParams) -> np.ndarray:
    """PR,ToD per segment, $/MWh: PR scaled by each segment's mean PTI relative
    to RLWPR, so a regional-load-shaped participant is valued at PR."""
    seg = tod_segment_of_ti()
    return np.array([rp.p * rp.phh[seg == k].mean() / rp.rlwp
                     for k in range(len(TOD_SEGMENTS))])


def mcl_clp(load: np.ndarray, gen: np.ndarray, rp: RegionalParams) -> dict:
    """OSL and unfloored PM (PM Limited Offset), $, by the CLP v10.0 cl 5 and
    cl 6 formulas with ED and EC per ToD segment, PR,ToD from tod_prices and
    the regional VF for every segment."""
    seg = tod_segment_of_ti()
    ed = np.bincount(seg, weights=load, minlength=len(TOD_SEGMENTS))
    ec = np.bincount(seg, weights=gen, minlength=len(TOD_SEGMENTS))
    value = float(np.dot(ed - ec, tod_prices(rp))) * (1.0 + GST_RATE)
    net_osl, net_pm = value * rp.vf_osl, value * rp.vf_pm
    return dict(el=float(np.sum(load)), eg=float(np.sum(gen)), prafl=np.nan, prafg=np.nan,
                osl=CLP_TOSL * max(net_osl, net_osl / rp.vf_osl),
                pm_raw=TRP * max(net_pm, net_pm / rp.vf_pm))


def assemble_mcl(osl: float, pm_raw: float) -> float:
    """MCL, $, unrounded: PM floored at 0, MCL = MAX(0, OSL + PM) (CLP v10.0
    cl 4.3.4(d), cl 10.1(a))."""
    return max(0.0, osl + max(pm_raw, 0.0))


def clp_round(osl: float, pm_raw: float) -> tuple[float, float, float]:
    """(OSL, PM, MCL), $, rounded per CLP v10.0 cl 10.1: OSL and PM up to the
    next $1,000, PM >= 0, OSL >= -PM, MCL = OSL + PM >= 0 up to the next
    $10,000, or the next $100,000 above $250,000."""
    pm = math.ceil(max(pm_raw, 0.0) / 1000.0) * 1000.0
    osl_r = max(math.ceil(osl / 1000.0) * 1000.0, -pm)
    return osl_r, pm, round_up_mcl(max(0.0, osl_r + pm))


def round_up_mcl(total: float) -> float:
    """CLP cl 10.1(b): up to the next $10,000, or $100,000 above $250,000."""
    step = 10_000.0 if total <= 250_000.0 else 100_000.0
    return math.ceil(total / step) * step


# ── Cohort energy ─────────────────────────────────────────────────────────────

def interval_start(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Start of each 5-minute interval in market time (AEST) from interval-end
    timestamps."""
    if idx.tz is None:
        idx = idx.tz_localize('Etc/GMT-10')
    return idx.tz_convert('Etc/GMT-10') - pd.Timedelta(minutes=5)


def split_energy(net_kwh: pd.Series, import_lf: float, export_lf: float
                 ) -> tuple[pd.Series, pd.Series]:
    """(debit, credit) energy per interval, kWh per household, loss-adjusted,
    from the cohort's mean net grid flow (import positive)."""
    return (net_kwh.clip(lower=0.0) * import_lf,
            (-net_kwh).clip(lower=0.0) * export_lf)


def season_ti_sums(energy: pd.Series) -> dict[str, tuple[np.ndarray, int]]:
    """season -> (energy summed per trading interval of the day, MWh; days).
    Raises CheckError unless every day in a season has all 288 intervals."""
    start = interval_start(energy.index)
    ti = (start.hour * 60 + start.minute) // 5
    month = start.month
    date = start.normalize()
    out = {}
    for season, months in SEASON_MONTHS.items():
        m = np.isin(month, months)
        if not m.any():
            continue
        counts = pd.Series(1, index=date[m]).groupby(level=0).sum()
        if (counts != TI_PER_DAY).any():
            bad = counts[counts != TI_PER_DAY]
            raise CheckError(f'{season}: days without {TI_PER_DAY} intervals: '
                             f'{bad.index[:3].tolist()}')
        sums = np.bincount(ti[m], weights=energy.to_numpy()[m] / 1000.0,
                           minlength=TI_PER_DAY)
        out[season] = (sums, len(counts))
    return out


# ── Simulated outstandings ────────────────────────────────────────────────────

def daily_settlement(debit_kwh: pd.Series, credit_kwh: pd.Series,
                     rrp: pd.Series) -> pd.Series:
    """Settlement amount owed to AEMO per trading day (market-time calendar
    day of the interval start), $ per household, GST on both legs."""
    amt = (rrp.to_numpy() / 1000.0 * (debit_kwh.to_numpy() - credit_kwh.to_numpy())
           * (1.0 + GST_RATE))
    day = interval_start(debit_kwh.index).normalize().tz_localize(None)
    return pd.Series(amt, index=day).groupby(level=0).sum()


def outstandings(daily: pd.Series, lag_days: int = PAYMENT_LAG_DAYS) -> pd.Series:
    """Amount outstanding at the end of each trading day, payments due that day
    made: sum of daily amounts whose billing week (Sunday-Saturday) is not yet
    paid, a week being paid lag_days after its Saturday. Only days on which
    every unpaid day lies inside the data are returned."""
    days = pd.DatetimeIndex(daily.index)
    if len(days) and not days.equals(pd.date_range(days[0], days[-1], freq='D')):
        raise CheckError('daily settlement is not on consecutive days')
    sat = days + pd.to_timedelta((5 - days.dayofweek) % 7, unit='D')
    pay = sat + pd.Timedelta(days=lag_days)
    paid = daily.groupby(pay).sum().reindex(days, fill_value=0.0).cumsum()
    out = daily.cumsum() - paid.to_numpy()
    first = days[0] - pd.Timedelta(days=1)
    first_valid = (first + pd.Timedelta(days=(5 - first.dayofweek) % 7)
                   + pd.Timedelta(days=lag_days))
    return out[days >= first_valid]


def chained_outstandings(daily: pd.Series, lag_days: int = PAYMENT_LAG_DAYS) -> pd.Series:
    """outstandings over daily amounts joined across years: each run of
    consecutive days is treated separately, so only the start of a run loses
    its first days."""
    daily = daily.sort_index()
    if daily.index.has_duplicates:
        raise CheckError('daily settlement has duplicate days')
    run = (daily.index.to_series().diff() != pd.Timedelta(days=1)).cumsum()
    parts = [outstandings(g, lag_days) for _, g in daily.groupby(run.to_numpy())]
    return pd.concat(parts) if parts else daily.iloc[:0]


def credit_support_daily(out_days: pd.DataFrame, year_rows: pd.DataFrame,
                         method: str = 'calculator') -> pd.DataFrame:
    """out_days (cohort, state, date, season, outstanding) joined to the
    `method` MCL and PM of that cohort, state, calendar year and season;
    required = max(MCL, PM + outstanding), PM floored at 0."""
    yr = year_rows[year_rows['method'] == method][
        ['cohort', 'state', 'year', 'season', 'mcl', 'pm_raw']].copy()
    yr['year'] = yr['year'].astype(int)
    d = out_days.drop(columns=['year'], errors='ignore').copy()
    d['year'] = pd.DatetimeIndex(d['date']).year
    d = d.merge(yr, on=['cohort', 'state', 'year', 'season'], how='left', validate='m:1')
    if d['mcl'].isna().any():
        bad = d[d['mcl'].isna()][['cohort', 'state', 'year', 'season']].drop_duplicates()
        raise CheckError(f'no MCL for {bad.values.tolist()[:3]}')
    d['pm'] = d['pm_raw'].clip(lower=0.0)
    d['required'] = np.maximum(d['mcl'], d['pm'] + d['outstanding'])
    d['method'] = method
    return d.drop(columns=['pm_raw'])


def credit_support_table(cs: pd.DataFrame, households: int,
                         guarantee_rate: float) -> pd.DataFrame:
    """Per (cohort, state, year, season): MCL, PM, peak outstanding and the
    peak credit support required, per household and for `households`; then
    an annual row per year (seasons weighted by SEASON_DAYS), a 'mean' row
    averaging the annual rows over years, and a 'max' row: the largest
    seasonal MCL, the largest seasonal requirement (at least that MCL) and
    the season it fell in. The retailer requires max(rounded MCL, rounded
    PM + households x peak outstanding). Guarantee cost on annual and max
    rows: rate x required_retailer (for 'max', the peak held all year)."""
    rows = []
    for (cohort, state, year, season), g in cs.groupby(['cohort', 'state', 'year', 'season'],
                                                       sort=False):
        mcl, pm = float(g['mcl'].iloc[0]), float(g['pm'].iloc[0])
        peak = float(g['outstanding'].max())
        _, pm_r, mcl_r = clp_round(households * (mcl - pm), households * pm)
        req_r = max(mcl_r, pm_r + households * peak)
        rows.append(dict(cohort=cohort, state=state, year=str(year), season=season,
                         days=len(g), mcl=mcl, pm=pm, peak_outstanding=peak,
                         required=max(mcl, pm + peak),
                         binding='trading_limit' if pm + peak > mcl else 'mcl',
                         mcl_retailer=mcl_r, required_retailer=req_r))
    seasonal = pd.DataFrame(rows)
    vals = ['mcl', 'pm', 'peak_outstanding', 'required', 'mcl_retailer', 'required_retailer']
    annual = []
    for (cohort, state, year), g in seasonal.groupby(['cohort', 'state', 'year'], sort=False):
        if set(g['season']) != set(SEASONS):
            continue
        w = g['season'].map(SEASON_DAYS).to_numpy() / sum(SEASON_DAYS.values())
        annual.append(dict(cohort=cohort, state=state, year=year, season='annual',
                           days=int(g['days'].sum()),
                           **{c: float(np.dot(w, g[c])) for c in vals}))
    annual = pd.DataFrame(annual)
    means = []
    if not annual.empty:
        for (cohort, state), g in annual.groupby(['cohort', 'state'], sort=False):
            means.append(dict(cohort=cohort, state=state, year='mean', season='annual',
                              days=int(g['days'].sum()),
                              **{c: float(g[c].mean()) for c in vals}))
    maxima = []
    for (cohort, state), g in seasonal.groupby(['cohort', 'state'], sort=False):
        top = g.loc[g['required'].idxmax()]
        maxima.append(dict(cohort=cohort, state=state, year='max', season='max',
                           days=int(g['days'].sum()), mcl=float(g['mcl'].max()),
                           required=float(top['required']),
                           mcl_retailer=float(g['mcl_retailer'].max()),
                           required_retailer=float(g['required_retailer'].max()),
                           peak_when=f"{top['season']} {top['year']}"))
    maxima = pd.DataFrame(maxima)
    if not maxima.empty:        # required >= MCL in every season, so max required >= max MCL
        maxima['above_mcl'] = maxima['required'] - maxima['mcl']
    out = pd.concat([seasonal, annual, pd.DataFrame(means), maxima], ignore_index=True)
    out['households'] = households
    a = out['season'].isin(['annual', 'max'])
    out.loc[a, 'guarantee_cost_retailer'] = guarantee_rate * out.loc[a, 'required_retailer']
    out.loc[a, 'guarantee_cost_hh'] = out.loc[a, 'guarantee_cost_retailer'] / households
    return out


# ── Aggregation ───────────────────────────────────────────────────────────────

def mcl_rows(ti_sums: dict, params: dict, keys: dict) -> list[dict]:
    """One row per season and method from pooled (debit, credit) per-TI sums:
    ti_sums[season] = (debit_sums, credit_sums, days)."""
    rows = []
    for season, (d_sum, c_sum, days) in ti_sums.items():
        rp = params[(keys['state'], season)]
        load, gen = d_sum / days, c_sum / days
        for method, fn in (('calculator', mcl_calculator), ('clp_v10', mcl_clp)):
            r = fn(load, gen, rp)
            r['mcl'] = assemble_mcl(r['osl'], r['pm_raw'])
            rows.append({**keys, 'season': season, 'method': method, 'days': days, **r})
    return rows


def national_rows(state_rows: pd.DataFrame, state_mix: dict[str, float]) -> pd.DataFrame:
    """National per household: state_mix-weighted OSL and unfloored PM summed
    across regions (calculator MCL!I43, S43), renormalised over the states
    present; PM floored and MCL assembled on the totals. Grouped by year too
    when `state_rows` has a year column."""
    out = []
    keys = ['cohort', 'season', 'method'] + (['year'] if 'year' in state_rows else [])
    for key, g in state_rows.groupby(keys, sort=False):
        g = g[g['state'].map(lambda s: state_mix.get(s, 0.0)) > 0]
        if g.empty:
            continue
        w = g['state'].map(state_mix).to_numpy()
        w = w / w.sum()
        osl, pm_raw = float(np.dot(w, g['osl'])), float(np.dot(w, g['pm_raw']))
        out.append(dict(**dict(zip(keys, key)), state='National',
                        states=','.join(g['state']), el=float(np.dot(w, g['el'])),
                        eg=float(np.dot(w, g['eg'])), prafl=np.nan, prafg=np.nan,
                        osl=osl, pm_raw=pm_raw, mcl=assemble_mcl(osl, pm_raw)))
    return pd.DataFrame(out)


def retailer_table(per_hh: pd.DataFrame, households: int,
                   guarantee_rate: float) -> pd.DataFrame:
    """Per-household rows scaled to `households`, rounded per CLP cl 10.1 per
    season, plus an annual row per (cohort, state, method): seasonal values
    weighted by SEASON_DAYS, and the guarantee cost on the annual MCL."""
    rows = []
    for _, r in per_hh.iterrows():
        osl_r, pm_r, mcl_r = clp_round(households * r['osl'], households * r['pm_raw'])
        rows.append({**r.to_dict(), 'households': households, 'osl_retailer': osl_r,
                     'pm_retailer': pm_r, 'mcl_retailer': mcl_r})
    seasonal = pd.DataFrame(rows)
    seasonal['pm'] = seasonal['pm_raw'].clip(lower=0.0)
    annual = []
    for (cohort, state, method), g in seasonal.groupby(['cohort', 'state', 'method'],
                                                       sort=False):
        if set(g['season']) != set(SEASONS):
            continue
        w = g['season'].map(SEASON_DAYS).to_numpy() / sum(SEASON_DAYS.values())
        a = dict(cohort=cohort, state=state, season='annual', method=method,
                 households=households)
        for c in ('el', 'eg', 'osl', 'pm', 'mcl', 'osl_retailer', 'pm_retailer',
                  'mcl_retailer'):
            a[c] = float(np.dot(w, g[c]))
        a['guarantee_cost_retailer'] = guarantee_rate * a['mcl_retailer']
        a['guarantee_cost_hh'] = a['guarantee_cost_retailer'] / households
        annual.append(a)
    out = pd.concat([seasonal, pd.DataFrame(annual)], ignore_index=True)
    out['el_kwh_day'] = out['el'] * 1000.0
    out['eg_kwh_day'] = out['eg'] * 1000.0
    cols = ['cohort', 'state', 'season', 'method', 'el_kwh_day', 'eg_kwh_day', 'prafl',
            'prafg', 'osl', 'pm', 'mcl', 'households', 'osl_retailer', 'pm_retailer',
            'mcl_retailer', 'guarantee_cost_retailer', 'guarantee_cost_hh']
    return out.reindex(columns=cols)


def outstanding_stats(out_days: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Per (cohort, state, season): mean, p98 and peak simulated outstanding per
    household, against the pooled per-household MCL of each method."""
    rows = []
    for (cohort, state, season), g in out_days.groupby(['cohort', 'state', 'season'],
                                                       sort=False):
        v = g['outstanding'].to_numpy()
        r = dict(cohort=cohort, state=state, season=season, days=len(v),
                 outstanding_mean=float(v.mean()),
                 outstanding_p98=float(np.quantile(v, 0.98)),
                 outstanding_max=float(v.max()))
        for method in METHODS:
            m = summary[(summary['cohort'] == cohort) & (summary['state'] == state)
                        & (summary['season'] == season) & (summary['method'] == method)]
            mcl = float(m['mcl'].iloc[0]) if len(m) else np.nan
            r[f'mcl_{method}'] = mcl
            r[f'share_days_above_{method}'] = float((v > mcl).mean()) if len(m) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


# ── Loading ───────────────────────────────────────────────────────────────────

def _scenario_id(state: str, year: str, battery_kwh: int | None) -> str:
    return f'{state}_{year}' + (f'_{battery_kwh}kWh' if battery_kwh else '')


def load_all(data_root: Path, scenarios: list[tuple[str, str]], params: dict,
             weighted: bool, weights_cfg: dict | None, battery_kwh: int
             ) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """One pass over every (cohort, state, year). Returns (state_year rows,
    pooled per-TI sums keyed (cohort, state) -> season -> [debit, credit, days],
    simulated outstanding per day, chained across years)."""
    runs_root = data_root / 'data' / 'output' / 'runs'
    price_cache, dc_ac_cache, meta_cache = {}, {}, {}
    year_rows, pooled, daily = [], {}, {}
    for state, year in scenarios:
        site_weight = (compute_site_weight_for_scenario(
            data_root, weights_cfg, state, year, dc_ac_cache, meta_cache)
            if weighted else None)
        rrp_all = get_prices(data_root, year, price_cache)[RRP_COL[state]]
        for ckey, (_, run, has_bat) in COHORTS.items():
            sid = _scenario_id(state, year, battery_kwh if has_bat else None)
            flows = load_flows(runs_root, run, state, year,
                               battery_kwh=battery_kwh if has_bat else None)
            bills = read_bills(runs_root, run, sid)
            if flows is None or bills is None:
                print(f'  WARNING: no flows or bills for {run} {sid}, skipping.',
                      file=sys.stderr)
                continue
            flows.columns = [str(c) for c in flows.columns]
            sites = [s for s in bills.index if s in set(flows.columns)
                     and (site_weight is None or s in site_weight.index)]
            if not sites:
                raise CheckError(f'{run} {sid}: no sites in common')
            w = (pd.Series(1.0, index=sites) if site_weight is None
                 else site_weight.loc[sites].astype(float))
            w = w / w.sum()
            net = pd.Series(flows[sites].to_numpy(dtype=float) @ w.to_numpy(),
                            index=flows.index)
            _, network_ids = run_tariff_ids(runs_root, run)
            nt = load_network_tariff(network_ids[state][year], TARIFF_DIR)
            debit, credit = split_energy(net, nt.import_loss_factor, nt.export_loss_factor)

            d_sums, c_sums = season_ti_sums(debit), season_ti_sums(credit)
            ti = {s: (d_sums[s][0], c_sums[s][0], d_sums[s][1]) for s in d_sums}
            year_rows += mcl_rows(ti, params, dict(cohort=ckey, state=state, year=year))
            acc = pooled.setdefault((ckey, state), {})
            for s, (d, c, n) in ti.items():
                a = acc.setdefault(s, [np.zeros(TI_PER_DAY), np.zeros(TI_PER_DAY), 0])
                a[0] += d
                a[1] += c
                a[2] += n

            rrp = rrp_all.reindex(flows.index)
            if rrp.isna().any():
                raise CheckError(f'{sid}: {int(rrp.isna().sum())} intervals without a price')
            daily.setdefault((ckey, state), []).append(daily_settlement(debit, credit, rrp))
            print(f'  {COHORTS[ckey][0]:<20} {sid:<16} {len(sites):>4} sites')
    season_of = {m: s for s, ms in SEASON_MONTHS.items() for m in ms}
    out_frames = []
    for (ckey, state), parts in daily.items():
        o = chained_outstandings(pd.concat(parts))
        out_frames.append(pd.DataFrame({
            'cohort': ckey, 'state': state, 'year': o.index.year.astype(str), 'date': o.index,
            'season': [season_of[m] for m in o.index.month], 'outstanding': o.to_numpy()}))
    return (pd.DataFrame(year_rows), pooled,
            pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame())


# ── Charts ────────────────────────────────────────────────────────────────────
#
# docs/thesis_figure_spec.md through thesis_style. draw_charts reads the CSVs
# the run writes, so --charts-only redraws every chart without the flows.
# Cohorts take the scenario names and colours of the other thesis charts
# (cap_hedge_analysis.build_scenarios). Data notes (period, weighting, method)
# are drawn unless --no-notes is given.

HEIGHT_IN = 3.0
PANEL_HEIGHT_IN = 1.15           # per cohort, in the daily credit support chart
BAR_GROUP_WIDTH = 0.72           # of the region spacing
BAR_GAP = 0.02                   # white gap between the bars of a group
BAND_ALPHA = ts.BAND_ALPHA
# Cohort key -> the cap_hedge_analysis scenario it is drawn as.
COHORT_SCENARIO = {'no_cer': 'No CER', 'solar_only': 'Solar Only',
                   'self_consumption': 'Self Consumption', 'coordinated': 'Amber'}
DAILY_COLUMNS = ['cohort', 'state', 'date', 'season', 'mcl', 'pm', 'outstanding']


def cohort_scenarios() -> dict[str, dict]:
    """cohort key -> scenario dict (label, colour, has_battery) as the other
    thesis charts draw it."""
    by_label = {sc['label']: sc for sc in build_scenarios(core_only=True)}
    return {c: by_label[label] for c, label in COHORT_SCENARIO.items()}


LIMIT_HATCH = '....'          # the trading limit above the MCL, in every cohort


def _sc_limit_patches(ink) -> tuple[dict, dict]:
    """The self-consumption box (outlined, hatched) with dots in place of the
    diagonal hatch: the trading limit above the MCL."""
    fill, edge = sbf._sc_patches(ink)
    return {**fill, 'hatch': LIMIT_HATCH}, edge


def _cohort_name(sc: dict, battery_kwh: int) -> str:
    """Legend / panel name. Self-consumption needs no retailer: its prudential
    requirement is the same on any plan, so the '(Amber)' is dropped."""
    return _name(sc, battery_kwh).replace(' (Amber)', '')


def _retailer_axis(ax, households: int, label: str | None):
    """Secondary y axis in $m for a retailer of `households` households."""
    sec = ax.secondary_yaxis('right', functions=(lambda y: y * households / 1e6,
                                                 lambda y: y * 1e6 / households))
    if label:
        sec.set_ylabel(label)
    return sec


def _scenario_key(sc: dict):
    """Legend handle for a scenario, as the other thesis bar charts."""
    if sc['label'] == 'Self Consumption':
        return tuple(mpatches.Patch(**kw) for kw in sbf._sc_patches(sc['color']))
    return mpatches.Patch(color=sc['color'], linewidth=0)


def fig_credit_support(daily: pd.DataFrame, region: str, households: int,
                       note: str | None, out: Path, battery_kwh: int = 20) -> None:
    """One panel per cohort: daily credit support required (PM + outstanding,
    at least the MCL) and the seasonal MCL, per household; shading where the
    trading limit needs more than the MCL."""
    d = daily[daily['state'] == region]
    scen = cohort_scenarios()
    cohorts = [c for c in COHORTS if c in set(d['cohort'])]
    if not cohorts:
        return
    fig = ts.new_figure(PANEL_HEIGHT_IN * len(cohorts) + 0.9)
    axes = fig.subplots(len(cohorts), 1, sharex=True, sharey=True, squeeze=False)[:, 0]
    days = pd.date_range(pd.to_datetime(d['date']).min(), pd.to_datetime(d['date']).max(),
                         freq='D')
    for ax, c in zip(axes, cohorts):
        sc = scen[c]
        g = d[d['cohort'] == c].assign(date=lambda x: pd.to_datetime(x['date'])
                                       ).set_index('date').reindex(days)
        need = g['pm'] + g['outstanding']
        ax.plot(days, need, color=sc['color'], lw=ts.MEMBER_LW, zorder=3)
        ax.plot(days, g['mcl'], color=ts.REFERENCE_COLOUR, lw=ts.REFERENCE_LW, ls='--',
                drawstyle='steps-post', zorder=2)
        ax.fill_between(days, g['mcl'], need, where=(need > g['mcl']).to_numpy(),
                        color=sc['color'], alpha=BAND_ALPHA, lw=0, zorder=1)
        ax.axhline(0, color=ts.SEPARATOR_COLOUR, lw=0.5, zorder=1)
        ax.set_title(_cohort_name(sc, battery_kwh), fontsize=ts.SUBTITLE_SIZE)
        _retailer_axis(ax, households, 'Retailer (\\$m)')
    fig.supylabel('Credit support required (\\$/household)',
                  fontsize=mpl.rcParams['axes.labelsize'])
    key = sbf.KEY_COLOUR
    fig.legend(handles=[Line2D([], [], color=key, lw=ts.MEMBER_LW),
                        Line2D([], [], color=ts.REFERENCE_COLOUR, lw=ts.REFERENCE_LW, ls='--'),
                        mpatches.Patch(color=key, alpha=BAND_ALPHA, linewidth=0)],
               labels=['Performance margin + outstanding', 'AEMO maximum credit limit',
                       'Above the MCL (trading limit binds)'],
               loc='outside lower center', ncol=3, frameon=False)
    if note:
        fa._draw_notes(axes[-1], note, 'lower right')
    ts.panel_suptitle(fig, f'Credit support required, {region}' + BASIS_SUFFIX)
    ts.save_figure(fig, out / f'prudential_credit_support_{region}')


def fig_credit_support_max(table: pd.DataFrame, households: int, years: str,
                           out: Path, battery_kwh: int = 20,
                           method: str = 'calculator') -> None:
    """Maximum credit support required per household over the period by
    region and cohort: the largest formula MCL, plus the trading-limit
    requirement above it (the bar's colour with a white hatch), stacked."""
    a = table[(table['season'] == 'max') & (table['method'] == method)]
    regions = [r for r in REGIONS if r in set(a['state'])]
    cohorts = [c for c in COHORTS if c in set(a['cohort'])]
    if not regions:
        return
    scen = cohort_scenarios()
    fig = ts.new_figure(HEIGHT_IN)
    ax = fig.subplots()
    n = len(cohorts)
    width = (BAR_GROUP_WIDTH - (n - 1) * BAR_GAP) / n
    x = np.arange(len(regions))
    top = 0.0
    for i, c in enumerate(cohorts):
        sc = scen[c]
        sub = a[a['cohort'] == c].set_index('state').reindex(regions)
        xs = x + (i - (n - 1) / 2) * (width + BAR_GAP)
        for xv, mcl, above, req in zip(xs, sub['mcl'], sub['above_mcl'], sub['required']):
            if not np.isfinite(req):
                continue
            if c == 'self_consumption':
                # Outlined box for the MCL, and a second box in dots for the
                # trading limit above it.
                for kw in sbf._sc_patches(sc['color']):
                    ax.bar([xv], [mcl], width=width, zorder=3, **kw)
                if above > 0:
                    for kw in _sc_limit_patches(sc['color']):
                        ax.bar([xv], [above], bottom=[mcl], width=width, zorder=3, **kw)
            else:
                ax.bar([xv], [mcl], width=width, zorder=3, color=sc['color'], linewidth=0)
                if above > 0:
                    ax.bar([xv], [above], bottom=[mcl], width=width, color=sc['color'],
                           edgecolor='white', hatch=LIMIT_HATCH, linewidth=0, zorder=3)
            # Left-aligned from the bar's left edge: each bar in a group is
            # taller than the one to its right, so the label clears it.
            ax.annotate(f'\\${req:,.0f}', (xv - width / 2, req), xytext=(0, 1.5),
                        textcoords='offset points', ha='left', va='bottom',
                        fontsize=ts.ANNOTATION_SIZE, zorder=4)
            top = max(top, req)
    ax.set_xticks(x, regions)
    ax.tick_params(axis='x', length=0)
    ax.grid(axis='x', visible=False)
    ax.set_xlim(-0.55, len(regions) - 0.45)
    ax.set_ylim(0, top * 1.1)    # room for the labels
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'\\${v:,.0f}'))
    ax.set_ylabel('Maximum credit support required (\\$/household)')
    _retailer_axis(ax, households, f'{households:,} households (\\$m)')
    ax.set_title(f'Maximum credit support required, {years}'
                 + ('' if method == 'calculator' else f', {METHOD_TITLES[method]}')
                 + BASIS_SUFFIX)
    key = sbf.KEY_COLOUR
    limit_key = mpatches.Patch(facecolor=key, edgecolor='white', hatch=LIMIT_HATCH, linewidth=0)
    handles = [_scenario_key(scen[c]) for c in cohorts]
    handles += [mpatches.Patch(color=key, linewidth=0), limit_key]
    labels = [_cohort_name(scen[c], battery_kwh) for c in cohorts]
    labels += ['Maximum credit limit (MCL)', 'Trading limit above the MCL']
    # Columns: the cohorts (two columns), then the MCL and the limit above it.
    fig.legend(handles=handles, labels=labels, loc='outside lower center', ncol=3,
               frameon=False)
    ts.save_figure(fig, out / ('prudential_credit_support_max'
                               + ('' if method == 'calculator' else f'_{method}')))


def draw_charts(out: Path, battery_kwh: int, weighted: bool, notes: bool) -> None:
    """Every chart, from the CSVs under out."""
    mpl.rcdefaults()             # financial_analysis sets its own on import
    ts.apply_style()
    table = pd.read_csv(out / 'prudential_credit_support.csv')
    households = int(table['households'].iloc[0])
    years = _years_label(sorted(str(y) for y in
                                pd.read_csv(out / 'prudential_state_year.csv')['year'].unique()))
    weight = ('Weighted by PV size; national by state mix' if weighted
              else 'Unweighted; national by equal state weights')

    def note(method: str, daily: bool = False) -> str | None:
        if not notes:
            return None
        text = (f'{years}, {battery_kwh} kWh battery. {weight}. {METHOD_NOTES[method]}, '
                "applied to each year's flows; outstandings at that year's prices, "
                'shortened settlement calendar. Required = max(MCL, PM + outstanding).')
        return textwrap.fill(text, width=120 if daily else 46)

    daily_path = out / 'prudential_credit_support_daily.csv'
    if daily_path.exists():
        daily = pd.read_csv(daily_path)
        for region in REGIONS:
            fig_credit_support(daily, region, households, note('calculator', True), out,
                               battery_kwh)
    else:
        print(f'  NOTE: no {daily_path}; run without --charts-only to write it. '
              'Daily credit support charts skipped.')
    for m in METHODS:
        fig_credit_support_max(table, households, years, out, battery_kwh, m)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', type=Path, default=Path('.'),
                        help='Project root holding data/output/runs, data/clean/nem_prices '
                             'and data/processed/edp (default: cwd)')
    parser.add_argument('--output-dir', type=Path, default=Path('analysis/output/prudential'))
    parser.add_argument('--reference-dir', type=Path, default=REFERENCE_DIR,
                        help='Folder holding the AEMO 2026 MCL calculators')
    parser.add_argument('--states', nargs='+', default=None)
    parser.add_argument('--years', nargs='+', default=None)
    parser.add_argument('--battery-kwh', type=int, default=20)
    parser.add_argument('--households', type=int, default=DEFAULT_HOUSEHOLDS,
                        help=f'Retailer size (default: {DEFAULT_HOUSEHOLDS:,})')
    parser.add_argument('--guarantee-rate', type=float, default=DEFAULT_GUARANTEE_RATE,
                        help='Bank guarantee cost, fraction of the MCL per year '
                             f'(default: {DEFAULT_GUARANTEE_RATE})')
    parser.add_argument('--unweighted', action='store_true',
                        help='Straight per-site means instead of size-band weighting')
    parser.add_argument('--weights-file', type=Path,
                        default=Path('analysis/config/pv_size_bands.yaml'))
    parser.add_argument('--charts-only', action='store_true',
                        help='Redraw the charts from the CSVs in --output-dir without '
                             'reading the flows')
    parser.add_argument('--no-notes', dest='notes', action='store_false',
                        help='Leave the data notes off, for the thesis')
    args = parser.parse_args(argv)
    if args.households <= 0:
        parser.error('--households must be positive')

    weighted = not args.unweighted
    if args.charts_only:
        draw_charts(args.output_dir, args.battery_kwh, weighted, args.notes)
        return 0
    weights_cfg = load_weights_config(args.weights_file) if weighted else None
    state_mix = (weights_cfg['state_mix'] if weighted
                 else {s: 1.0 for s in RRP_COL})
    scenarios = discover_scenarios(args.data_root)
    if args.states:
        scenarios = [(s, y) for s, y in scenarios if s in args.states]
    if args.years:
        scenarios = [(s, y) for s, y in scenarios if y in args.years]
    if not scenarios:
        print('ERROR: no scenarios found or none match the filter.', file=sys.stderr)
        return 1
    print(f'Scenarios found: {scenarios}')

    try:
        params = load_regional_params(args.reference_dir)
        year_rows, pooled, out_days = load_all(args.data_root, scenarios, params, weighted,
                                               weights_cfg, args.battery_kwh)
    except CheckError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 1
    if year_rows.empty:
        print('ERROR: no flows found.', file=sys.stderr)
        return 1

    state_rows = []
    for (ckey, state), ti in pooled.items():
        state_rows += mcl_rows({s: tuple(v) for s, v in ti.items()}, params,
                               dict(cohort=ckey, state=state))
    state_rows = pd.DataFrame(state_rows)
    per_hh = pd.concat([state_rows, national_rows(state_rows, state_mix)], ignore_index=True)
    summary = retailer_table(per_hh, args.households, args.guarantee_rate)
    summary['cohort_label'] = summary['cohort'].map(lambda c: COHORTS[c][0])

    nat_days = []
    for (ckey, season), g in out_days.groupby(['cohort', 'season']):
        piv = g.pivot_table(index='date', columns='state', values='outstanding')
        piv = piv[[s for s in piv.columns if state_mix.get(s, 0.0) > 0]].dropna()
        if piv.empty:
            continue
        w = np.array([state_mix[s] for s in piv.columns])
        nat_days.append(pd.DataFrame({'cohort': ckey, 'state': 'National', 'season': season,
                                      'date': piv.index,
                                      'outstanding': piv.to_numpy() @ (w / w.sum())}))
    out_all = pd.concat([out_days] + nat_days, ignore_index=True)
    ostats = outstanding_stats(out_all, summary)
    year_all = pd.concat([year_rows, national_rows(year_rows, state_mix)], ignore_index=True)
    try:
        cs_by = {m: credit_support_daily(out_all, year_all, m) for m in METHODS}
    except CheckError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 1
    cs = cs_by['calculator']
    cs_table = pd.concat([credit_support_table(d, args.households, args.guarantee_rate)
                          .assign(method=m) for m, d in cs_by.items()], ignore_index=True)
    cs_table.insert(2, 'method', cs_table.pop('method'))

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    year_rows.to_csv(out / 'prudential_state_year.csv', index=False)
    summary.to_csv(out / 'prudential_summary.csv', index=False)
    ostats.to_csv(out / 'prudential_outstandings.csv', index=False)
    cs_table.to_csv(out / 'prudential_credit_support.csv', index=False)
    cs[DAILY_COLUMNS].to_csv(out / 'prudential_credit_support_daily.csv', index=False)
    draw_charts(out, args.battery_kwh, weighted, args.notes)

    a = summary[summary['season'] == 'annual']
    show = a.pivot_table(index=['state', 'cohort'], columns='method',
                         values=['mcl', 'mcl_retailer', 'guarantee_cost_hh'], sort=False)
    with pd.option_context('display.width', 200, 'display.float_format', '{:,.2f}'.format):
        print('\nAnnual MCL per household, retailer MCL (rounded) and guarantee cost per '
              f'household per year ({args.households:,} households, rate '
              f'{args.guarantee_rate:.2%}):')
        print(show)
        m = cs_table[cs_table['season'] == 'max']
        print('\nMaximum credit support required over the period:')
        print(m.set_index(['method', 'state', 'cohort'])[[
            'mcl', 'required', 'required_retailer', 'peak_when']].to_string())
        nsw = summary[(summary['cohort'] == 'no_cer') & (summary['state'] == 'NSW')
                      & (summary['method'] == 'calculator') & (summary['season'] != 'annual')]
        if len(nsw):
            print('\nNo CER NSW PRAF (calculator) against ACIL Allen Ausgrid 2026-27 '
                  '(1.0014 / 1.0520 / 0.8886):')
            print(nsw[['season', 'prafl', 'el_kwh_day']].to_string(index=False))
    print(f'\nOutputs written to {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
