"""
cap_hedge_analysis.py

$300 cap hedging need of a solar/battery cohort, per state (all years
pooled) and nationally (state_mix weighted). See docs/cap_hedge_spec.md.

  Chart 1  flow_duration_300_count.png
           flow_duration_300_weighted.png
                                      cohort net import in every >$300/MWh
                                      interval, sorted, each interval equal
                                      width (count) or width proportional to
                                      its price above $300/MWh (weighted)
  Chart 2  net_position_vs_cover.png  mean quarter (a) and worst quarter (b)
                                      net position against cap cover x
  Chart 3  outcomes_at_x_worst.png    each scenario at its own worst-quarter
                                      cover: the cover, the worst quarter and
                                      the mean outcome
  Chart 4  state_months/state_month_outcomes.png
                                      histogram over state-months of the
                                      above-cap net position with no cover
  hedge_summary.csv                   hedge levels and outcomes per scenario
  quarter_lines.csv                   per quarter and scenario: cap terms,
                                      above-cap cost, net position at 0, x_0
                                      and x_worst
  state_months/state_months.csv       per state, month and scenario: >$300
                                      interval count, above-cap cost, net
                                      position with no cover

--premium sample_mean charges every quarter the state's mean start-of-quarter
cap price instead of that quarter's own (output: analysis/output/cap_hedge_sample_mean).

Sign conventions: m (cohort net import) positive = import, kW/site. x is
cap cover in kW/site; negative x is cap sold. N_q(x) = a_q·x + b_q is the
per-site above-cap net position in quarter q ($/site), with
a_q = H_q·(CSP_q − F_q)/1000 and b_q = −L_q.

Charts follow docs/thesis_figure_spec.md and are drawn from the CSVs (including
flow_duration_300.csv, the thinned chart 1 curves), so --charts-only redraws
them without the flows. --no-notes leaves the data notes off, for the thesis.

Run from repo root:
  python3 analysis/scripts/cap_hedge_analysis.py --data-root "/Volumes/Manual/Thesis Project" --core-only
  python3 analysis/scripts/cap_hedge_analysis.py --charts-only --no-notes
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from asx_caps import ASX_CAP_BY_REGION, CAP_STRIKE, cap_settlement, contract_quarter
import financial_analysis as fa
import flow_exposure_analysis as fe
import savings_bar_figure as sbf
import thesis_style as ts
from financial_analysis import load_weights_config
from flow_exposure_analysis import (
    COORDINATION_PLANS, INTERVAL_HOURS,
    REFERENCE_SCENARIOS, RRP_COL,
    compute_site_weight_for_scenario, discover_scenarios, get_prices,
    load_flows, weighted_row_mean,
)

K_VALUES = (1, 2, 4, 10)
GRID_MIN, GRID_MAX, GRID_STEP, GRID_MARGIN = -6.0, 2.0, 0.05, 0.5
NO_CER = 'No CER'
CORE_PLANS = ('Amber',)
OPTIONAL_PLANS = ('GloBird', 'AGL', 'Origin Battery Lite', 'Energy Australia')
# Battery plans use their rolling (imperfect foresight) run, named from the
# perfect-foresight lp_dir; plans without one are left out at run time.
PF_PREFIX, ROLLING_PREFIX = 'full_run_optimised_', 'full_run_rolling_'
# Reference colours as on the flow_exposure_analysis.py bar charts: No CER
# black, Solar Only #aaaaaa, Self-consumption the key grey (drawn outlined and
# hatched on bar charts).
REFERENCE_COLOURS = {'Self Consumption': sbf.KEY_COLOUR}
QUARTERS_PER_YEAR = 4


def build_scenarios(core_only: bool) -> list[dict]:
    """Scenario list in display order. Labels, colours and run directories
    come from flow_exposure_analysis.py, with REFERENCE_COLOURS applied and
    each battery plan's lp_dir swapped for its rolling run.
    """
    ref = {r['label']: r for r in REFERENCE_SCENARIOS}
    plans = {p['label']: p for p in COORDINATION_PLANS}
    out = []
    for label, ref_label in [(NO_CER, 'No CER (load only)'),
                             ('Solar Only', 'Solar Only'),
                             ('Self Consumption', 'Self Consumption (Amber)')]:
        r = ref[ref_label]
        out.append(dict(label=label, color=REFERENCE_COLOURS.get(label, r['color']),
                        run_dir=r['run_dir'], has_battery=r['has_battery'], ls='-'))
    for label in CORE_PLANS + (() if core_only else OPTIONAL_PLANS):
        p = plans[label]
        out.append(dict(label=label, color=p['color'],
                        run_dir=p['lp_dir'].replace(PF_PREFIX, ROLLING_PREFIX),
                        has_battery=True, ls='--'))
    return out


# ── Loading ───────────────────────────────────────────────────────────────────

@dataclass
class StateRun:
    """One scenario in one state, all loaded years."""
    years: list[str] = field(default_factory=list)
    m_gt: list[np.ndarray] = field(default_factory=list)   # kW/site, P > 300
    e_gt: list[np.ndarray] = field(default_factory=list)   # $/MWh, P − 300
    L_q: list[pd.Series] = field(default_factory=list)     # $/site per quarter
    L_mo: list[pd.Series] = field(default_factory=list)    # $/site per month
    n_mo: list[pd.Series] = field(default_factory=list)    # >$300 intervals per month

    def finish(self) -> None:
        self.m = np.concatenate(self.m_gt) if self.m_gt else np.array([])
        self.e = np.concatenate(self.e_gt) if self.e_gt else np.array([])
        self.L = pd.concat(self.L_q).groupby(level=[0, 1]).sum()
        empty = pd.Series(dtype=float, index=pd.MultiIndex.from_arrays([[], []]))
        self.L_month = (pd.concat(self.L_mo).groupby(level=[0, 1]).sum()
                        if self.L_mo else empty)
        self.n_month = (pd.concat(self.n_mo).groupby(level=[0, 1]).sum()
                        if self.n_mo else empty)


def load_all(data_root: Path, scenarios: list[tuple[str, str]],
             scen_list: list[dict], weighted: bool, weights_cfg: dict | None,
             battery_kwh: int):
    """One pass over every (scenario, state, year) flows file. Returns
    runs[(label, state)] -> StateRun and rrp[state][year] -> price Series.
    """
    runs_root = data_root / 'data' / 'output' / 'runs'
    price_cache: dict[str, pd.DataFrame] = {}
    sw_cache: dict[tuple[str, str], pd.Series] = {}
    dc_ac_cache: dict[str, float] = {}
    meta_cache: dict[str, pd.DataFrame] = {}
    runs: dict[tuple[str, str], StateRun] = {}
    rrp: dict[str, dict[str, pd.Series]] = {}
    rows = []

    for sc in scen_list:
        for state, year in scenarios:
            df = load_flows(runs_root, sc['run_dir'], state, year,
                            battery_kwh=battery_kwh if sc['has_battery'] else None)
            if df is None:
                print(f"  WARNING: no flows for '{sc['run_dir']}' {state}_{year}"
                      f"{f'_{battery_kwh}kWh' if sc['has_battery'] else ''}, skipping.",
                      file=sys.stderr)
                continue
            site_weight = None
            if weighted:
                if (state, year) not in sw_cache:
                    sw_cache[(state, year)] = compute_site_weight_for_scenario(
                        data_root, weights_cfg, state, year, dc_ac_cache, meta_cache)
                site_weight = sw_cache[(state, year)]
                n_used = len(set(df.columns) & set(site_weight.index))
            else:
                n_used = len(df.columns)

            price = get_prices(data_root, year, price_cache)[RRP_COL[state]]
            rrp.setdefault(state, {})[year] = price

            m = weighted_row_mean(df, site_weight) / INTERVAL_HOURS
            p = price.reindex(m.index)
            valid = m.notna() & p.notna()
            e = (p - CAP_STRIKE).clip(lower=0)
            gt = valid & (p > CAP_STRIKE)
            quarters = contract_quarter(m.index)
            cost = (m * e * INTERVAL_HOURS / 1000.0).where(valid, 0.0)
            L_q = cost.groupby([quarters.get_level_values(0),
                                quarters.get_level_values(1)]).sum()
            # Calendar month of the interval start, as for contract_quarter.
            start = m.index - pd.Timedelta(hours=INTERVAL_HOURS)
            L_mo = cost.groupby([start.year, start.month]).sum()
            n_mo = gt.groupby([start.year, start.month]).sum()

            run = runs.setdefault((sc['label'], state), StateRun())
            run.years.append(year)
            run.m_gt.append(m[gt].to_numpy())
            run.e_gt.append(e[gt].to_numpy())
            run.L_q.append(L_q)
            run.L_mo.append(L_mo)
            run.n_mo.append(n_mo)
            rows.append((sc['label'], f'{state}_{year}', len(df.columns), n_used,
                         int((~valid).sum()), int(gt.sum()), float(m.mean())))

    for run in runs.values():
        run.finish()

    print(f"\n{'Scenario':<20} {'State_year':<10} {'Sites':>5} {'Used':>5} "
          f"{'NaN rows':>8} {'>$300':>6} {'Mean net kW':>11}")
    for r in rows:
        print(f"{r[0]:<20} {r[1]:<10} {r[2]:>5} {r[3]:>5} {r[4]:>8} {r[5]:>6} {r[6]:>11.3f}")
    return runs, rrp


# ── Tiers ─────────────────────────────────────────────────────────────────────

@dataclass
class Tier:
    """Everything the charts and summary need for one tier (state or national).

    a[i]          $/site per kW of cover in quarter i (scenario-independent)
    b[label][i]   −L_q, $/site
    m, w_count, w_excess[label]  chart 1 points and their widths
    """
    name: str
    years_label: str
    quarters: list[tuple[int, int]]
    a: np.ndarray
    b: dict[str, np.ndarray]
    m: dict[str, np.ndarray]
    w_count: dict[str, np.ndarray]
    w_excess: dict[str, np.ndarray]
    n_gt: dict[str, int]
    csp: np.ndarray | None = None      # $/MWh per quarter (state tiers only)
    price: np.ndarray | None = None    # F_q used, $/MWh (state tiers only)
    hours: np.ndarray | None = None    # H_q (state tiers only)
    premium: str = 'start_of_quarter'


def _years_label(years: list[str]) -> str:
    ys = sorted(set(years))
    return ys[0] if len(ys) == 1 else f'{ys[0]}–{ys[-1]}'


def build_state_tier(state: str, runs: dict, rrp_by_year: dict[str, pd.Series],
                     scen_list: list[dict], premium: str = 'start_of_quarter') -> Tier | None:
    region = RRP_COL[state].replace('_rrp', '')
    labels = [sc['label'] for sc in scen_list if (sc['label'], state) in runs]
    if NO_CER not in labels:
        print(f"  {state}: no '{NO_CER}' run, state skipped.", file=sys.stderr)
        return None

    settle = cap_settlement(pd.concat([rrp_by_year[y] for y in sorted(rrp_by_year)]))
    for q, row in settle[settle['n_nan'] > 0].iterrows():
        print(f"  NOTE: {state} {q[0]} Q{q[1]}: {int(row['n_nan'])} NaN prices "
              f"excluded from CSP.")
    fwd = ASX_CAP_BY_REGION[region]

    all_q = set()
    common = None
    for lab in labels:
        qs = set(runs[(lab, state)].L.index)
        all_q |= qs
        common = qs if common is None else common & qs
    common = {q for q in common if q in fwd and q in settle.index
              and pd.notna(settle.loc[q, 'csp'])}
    dropped = sorted(all_q - common)
    if dropped:
        print(f"  {state}: quarters dropped from charts 2-3 (not common to every "
              f"scenario, or no F_q/CSP): {', '.join(f'{y} Q{q}' for y, q in dropped)}")
    quarters = sorted(common)
    if not quarters:
        print(f"  {state}: no usable quarters, state skipped.", file=sys.stderr)
        return None

    csp = np.array([settle.loc[q, 'csp'] for q in quarters])
    hours = np.array([settle.loc[q, 'hours'] for q in quarters])
    price = np.array([fwd[q] for q in quarters])
    if premium == 'sample_mean':
        # Every quarter charged the state's mean start-of-quarter price over
        # the tier's quarters: cover bought well ahead at a stable price.
        price = np.full(len(quarters), price.mean())
    a = hours * (csp - price) / 1000.0
    tier = Tier(name=state,
                years_label=_years_label(runs[(NO_CER, state)].years),
                quarters=quarters, a=a, b={}, m={}, w_count={}, w_excess={}, n_gt={},
                csp=csp, price=price, hours=hours, premium=premium)
    for lab in labels:
        run = runs[(lab, state)]
        tier.b[lab] = -run.L.reindex(quarters).to_numpy()
        tier.m[lab] = run.m
        tier.w_count[lab] = np.ones_like(run.m)
        tier.w_excess[lab] = run.e * INTERVAL_HOURS
        tier.n_gt[lab] = len(run.m)

    # Identity: summed payout at x_0 equals summed L, when every flows
    # quarter is in the chart 2 sample.
    for lab in labels:
        run = runs[(lab, state)]
        if set(run.L.index) != set(quarters):
            print(f"  CHECK {state} {lab}: payout(x_0) = ΣL not checked, "
                  f"{len(set(run.L.index) - set(quarters))} flows quarter(s) outside "
                  f"the chart 2 sample.")
            continue
        x0 = exposure_neutral(tier, lab)
        csp_h = np.array([settle.loc[q, 'csp'] * settle.loc[q, 'hours'] for q in quarters])
        payout = x0 / 1000.0 * csp_h.sum()
        total_L = run.L.sum()
        ok = np.isclose(payout, total_L, rtol=1e-9, atol=1e-9)
        print(f"  CHECK {state} {lab}: payout(x_0) {payout:,.4f} vs ΣL "
              f"{total_L:,.4f} $/site  {'OK' if ok else 'MISMATCH'}")
    return tier


def build_national_tier(tiers: dict[str, Tier], runs: dict, scen_list: list[dict],
                        state_mix: dict[str, float]) -> Tier | None:
    states = [s for s in tiers if state_mix.get(s, 0) > 0]
    if len(states) < 2:
        return None
    labels = [sc['label'] for sc in scen_list
              if all(sc['label'] in tiers[s].b for s in states)]
    missing = [sc['label'] for sc in scen_list if sc['label'] not in labels]
    if missing:
        print(f"  National: scenarios not present in every state, left out: {missing}")

    quarters = sorted(set().union(*(tiers[s].quarters for s in states)))
    a = np.zeros(len(quarters))
    b = {lab: np.zeros(len(quarters)) for lab in labels}
    for i, q in enumerate(quarters):
        present = [s for s in states if q in tiers[s].quarters]
        w = np.array([state_mix[s] for s in present])
        w = w / w.sum()
        for s, ws in zip(present, w):
            j = tiers[s].quarters.index(q)
            a[i] += ws * tiers[s].a[j]
            for lab in labels:
                b[lab][i] += ws * tiers[s].b[lab][j]

    tier = Tier(name='National',
                years_label=_years_label([y for s in states
                                          for y in runs[(NO_CER, s)].years]),
                quarters=quarters, a=a, b=b, m={}, w_count={}, w_excess={}, n_gt={},
                premium=tiers[states[0]].premium)
    for lab in labels:
        # Each (state, interval) point weighted by state_mix[s] / Y_s.
        ws = {s: state_mix[s] / len(runs[(lab, s)].years) for s in states}
        tier.m[lab] = np.concatenate([tiers[s].m[lab] for s in states])
        tier.w_count[lab] = np.concatenate(
            [np.full(len(tiers[s].m[lab]), ws[s]) for s in states])
        tier.w_excess[lab] = np.concatenate(
            [ws[s] * tiers[s].w_excess[lab] for s in states])
        tier.n_gt[lab] = sum(tiers[s].n_gt[lab] for s in states)
    return tier


# ── Hedge arithmetic ──────────────────────────────────────────────────────────

def exposure_neutral(tier: Tier, lab: str) -> float:
    """x_0: payout-weighted mean net import, kW/site."""
    w = tier.w_excess[lab]
    return float(np.sum(tier.m[lab] * w) / np.sum(w)) if w.sum() > 0 else np.nan


def lowest_k_mean(a: np.ndarray, b: np.ndarray, x, k: int) -> np.ndarray:
    """W_k(x): mean of the k lowest N_q(x) = a_q·x + b_q, for each x."""
    vals = np.outer(np.atleast_1d(x), a) + b
    vals.sort(axis=1)
    return vals[:, :min(k, len(a))].mean(axis=1)


def maximise_lowest_k(a: np.ndarray, b: np.ndarray, k: int):
    """(lo, hi, W_max) of the x maximising W_k. W_k is concave and piecewise
    linear with breakpoints only where two quarter lines cross, so it is
    evaluated exactly at every crossing. lo = hi = ±inf when W_k rises
    without limit; lo/hi = ±inf at the end of a flat tail.
    """
    k = min(k, len(a))
    s = np.sort(a)
    slope_plus, slope_minus = s[:k].mean(), s[-k:].mean()
    scale = max(1.0, float(np.abs(a).max()))
    tol = 1e-12 * scale
    if slope_plus > tol:
        return np.inf, np.inf, np.inf
    if slope_minus < -tol:
        return -np.inf, -np.inf, np.inf
    i, j = np.triu_indices(len(a), 1)
    da = a[i] - a[j]
    keep = np.abs(da) > tol
    cand = np.unique((b[j][keep] - b[i][keep]) / da[keep])
    if len(cand) == 0:                     # every slope equal and zero
        return -np.inf, np.inf, float(lowest_k_mean(a, b, 0.0, k)[0])
    w = lowest_k_mean(a, b, cand, k)
    w_max = float(w.max())
    at_max = cand[w >= w_max - 1e-9 * max(1.0, abs(w_max))]
    lo = -np.inf if abs(slope_minus) <= tol else float(at_max.min())
    hi = np.inf if abs(slope_plus) <= tol else float(at_max.max())
    return lo, hi, w_max


def worst_quarter(tier: Tier, b: np.ndarray, x: float) -> str:
    if not np.isfinite(x):
        return ''
    y, q = tier.quarters[int(np.argmin(tier.a * x + b))]
    return f'{y} Q{q}'


def analyse(tier: Tier, labels: list[str]) -> pd.DataFrame:
    """Summary rows for one tier, one per scenario, in display order."""
    a, mean_a = tier.a, float(tier.a.mean())
    rows = []
    for lab in labels:
        b = tier.b[lab]
        x0 = exposure_neutral(tier, lab)
        lo, hi, w_max = maximise_lowest_k(a, b, 1)
        wc, we, m = tier.w_count[lab], tier.w_excess[lab], tier.m[lab]

        def mean_n(x):
            return mean_a * x + b.mean() if np.isfinite(x) else np.nan

        def w1(x):
            return float(lowest_k_mean(a, b, x, 1)[0]) if np.isfinite(x) else np.nan

        row = dict(
            tier=tier.name, scenario=lab, years=tier.years_label,
            n_quarters=len(tier.quarters),
            n_intervals_gt300=tier.n_gt[lab],
            mean_import_gt300_kw=float(np.sum(m * wc) / np.sum(wc)) if len(m) else np.nan,
            share_importing_gt300=float(np.sum(wc[m > 0]) / np.sum(wc)) if len(m) else np.nan,
            x0_kw=x0, x_worst_lo_kw=lo, x_worst_hi_kw=hi,
        )
        for k in K_VALUES:
            row[f'x_worst_k{k}_kw'] = maximise_lowest_k(a, b, k)[0]
        ks = np.array([row[f'x_worst_k{k}_kw'] for k in K_VALUES], dtype=float)
        ks = ks[np.isfinite(ks)]
        row['x_worst_k_min_kw'] = ks.min() if len(ks) else np.nan
        row['x_worst_k_max_kw'] = ks.max() if len(ks) else np.nan
        row.update(
            mean_N_at_0=mean_n(0.0), mean_N_at_x0=mean_n(x0), mean_N_at_xworst=mean_n(lo),
            W1_at_0=w1(0.0), W1_at_xworst=w1(lo),
            worst_q_at_0=worst_quarter(tier, b, 0.0),
            worst_q_at_x0=worst_quarter(tier, b, x0),
            worst_q_at_xworst=worst_quarter(tier, b, lo),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def x_grid(summary: pd.DataFrame) -> np.ndarray:
    marks = summary[['x0_kw', 'x_worst_lo_kw', 'x_worst_hi_kw']].to_numpy(float).ravel()
    marks = marks[np.isfinite(marks)]
    lo = min(GRID_MIN, np.floor((marks.min() - GRID_MARGIN) / GRID_STEP) * GRID_STEP) if len(marks) else GRID_MIN
    hi = max(GRID_MAX, np.ceil((marks.max() + GRID_MARGIN) / GRID_STEP) * GRID_STEP) if len(marks) else GRID_MAX
    return np.arange(lo, hi + GRID_STEP / 2, GRID_STEP)


# ── Charts ────────────────────────────────────────────────────────────────────
#
# docs/thesis_figure_spec.md through thesis_style. draw_charts reads the CSVs
# the tiers write, so --charts-only redraws every chart without the flows.
# Scenario names, tick labels, colours and the outlined, hatched
# Self-consumption bar follow the flow_exposure_analysis.py bar charts. Data
# notes (scope, years, battery, weighting) are drawn unless --no-notes is
# given.

HEIGHT_IN = 3.0
NET_POSITION_HEIGHT_IN = 4.4
STATE_MONTH_HEIGHT_IN = 3.2
BAR_WIDTH = 0.72
# Resolution of the saved flow duration curves: the first and last point in
# each cell of this width (% of the x axis) are kept, finer than a pixel.
FLOW_CELL_PCT = 0.02
# This script's labels as flow_exposure_analysis.py's REFERENCE_SCENARIOS name
# them, for its display names and tick labels.
FE_LABELS = {NO_CER: 'No CER (load only)',
             'Self Consumption': 'Self Consumption (Amber)'}
DOLLAR = '\\$'
# Battery scenarios use the full_run_rolling_* runs (build_scenarios), the
# achievable basis of financial_analysis.RUN_BASES.
BASIS_SUFFIX = ', imperfect foresight (achievable)'


def _name(sc: dict, battery_kwh: int) -> str:
    name = fe._display_name(FE_LABELS.get(sc['label'], sc['label']))
    return f'{name}, {battery_kwh} kWh' if sc['has_battery'] else name


def _tick(label: str) -> str:
    return fe._tick_label(FE_LABELS.get(label, label))


def _minus(text: str) -> str:
    return text.replace('-', '−')


def _scenario_bar(ax, x: float, y: float, sc: dict, width: float) -> None:
    if sc['label'] == 'Self Consumption':
        for kw in sbf._sc_patches(sc['color']):
            ax.bar([x], [y], width=width, zorder=3, **kw)
    else:
        ax.bar([x], [y], width=width, color=sc['color'], linewidth=0, zorder=3)


def _note(tier: Tier, weighted: bool, battery_kwh: int, quarters: bool = True) -> str:
    note = (f'{tier.name}, {tier.years_label}, {battery_kwh} kWh battery\n'
            f'{fe._weight_note(weighted, national=(tier.name == "National"))}')
    if quarters:
        basis = ('start-of-quarter ASX price' if tier.premium == 'start_of_quarter' else
                 "each state's mean start-of-quarter ASX price in every quarter")
        note += f'\n{len(tier.quarters)} quarters; cap premium: {basis}'
    return note


def tier_from_csv(summary: pd.DataFrame, qlines: pd.DataFrame, premium: str) -> Tier:
    """The quarter terms of a tier from its saved CSVs (what the net position
    and outcome charts use; the flow duration points are not rebuilt)."""
    quarters = [(int(q[:4]), int(q[-1])) for q in dict.fromkeys(qlines['quarter'])]
    first = qlines[qlines['scenario'] == qlines['scenario'].iloc[0]]
    b = {lab: g['N_at_0'].to_numpy() for lab, g in qlines.groupby('scenario', sort=False)}
    return Tier(name=summary['tier'].iloc[0], years_label=str(summary['years'].iloc[0]),
                quarters=quarters, a=first['slope_per_kw'].to_numpy(), b=b,
                m={}, w_count={}, w_excess={}, n_gt={}, premium=premium)


def _thin(x: np.ndarray) -> np.ndarray:
    """Mask keeping the first and last point of each FLOW_CELL_PCT cell."""
    cell = np.floor(x / FLOW_CELL_PCT).astype(np.int64)
    change = cell[1:] != cell[:-1]
    return np.concatenate([[True], change]) | np.concatenate([change, [True]])


def flow_duration_curves(tier: Tier, labels: list[str]) -> pd.DataFrame:
    """The chart 1 step curves, per scenario and weighting: x as the % of the
    weighted intervals, net import in kW/site, thinned to FLOW_CELL_PCT."""
    frames = []
    for lab in labels:
        if lab not in tier.m or len(tier.m[lab]) == 0:
            continue
        order = np.argsort(-tier.m[lab], kind='stable')
        ms = tier.m[lab][order]
        y = np.concatenate([ms, ms[-1:]])
        for kind, w in (('count', tier.w_count[lab]), ('weighted', tier.w_excess[lab])):
            ws = w[order]
            x = np.concatenate([[0.0], np.cumsum(ws) / ws.sum() * 100.0])
            keep = _thin(x)
            frames.append(pd.DataFrame({'scenario': lab, 'weighting': kind,
                                        'x_pct': x[keep], 'net_import_kw': y[keep]}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fig_flow_duration(tier: Tier, curves: pd.DataFrame, scen_list: list[dict],
                      summary: pd.DataFrame, note: str | None, battery_kwh: int,
                      out_dir: Path) -> None:
    """Chart 1 as two figures: flow_duration_300_count (each interval equal
    width) and flow_duration_300_weighted (width proportional to P − 300,
    with each scenario's x_0 dotted)."""
    x0s = summary.set_index('scenario')['x0_kw']
    figures = (
        ('count', f'Cohort net import in >{DOLLAR}300/MWh intervals' + BASIS_SUFFIX,
         f'Share of >{DOLLAR}300/MWh intervals (%)'),
        ('weighted', f'Cohort net import in >{DOLLAR}300/MWh weighted intervals'
                     + BASIS_SUFFIX,
         f'Share of >{DOLLAR}300/MWh intervals, weighted by price above '
         f'{DOLLAR}300/MWh (%)'),
    )
    for kind, title, xlabel in figures:
        fig = ts.new_figure(HEIGHT_IN)
        ax = fig.subplots()
        for sc in scen_list:
            lab = sc['label']
            g = curves[(curves['scenario'] == lab) & (curves['weighting'] == kind)]
            if g.empty:
                continue
            name = _name(sc, battery_kwh)
            if kind == 'weighted':
                name += _minus(f' ($x_0$ = {x0s[lab]:.2f} kW/site)')
                ax.axhline(x0s[lab], color=sc['color'], ls=':', lw=ts.REFERENCE_LW,
                           zorder=2)
            ax.step(g['x_pct'], g['net_import_kw'], where='post', color=sc['color'],
                    ls=sc['ls'], lw=ts.DATA_LW, label=name, zorder=3)
        ax.axhline(0, color=ts.SEPARATOR_COLOUR, lw=0.5, zorder=1)
        ax.set_xlim(0, 100)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: _minus(f'{v:g}')))
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Cohort net import (kW/site)')
        ax.legend(loc='upper right')
        if note:
            fa._draw_notes(ax, note, 'lower left')
        ts.save_figure(fig, out_dir / f'flow_duration_300_{kind}')


def fig_net_position(tier: Tier, scen_list: list[dict], summary: pd.DataFrame,
                     xs: np.ndarray, note: str | None, battery_kwh: int,
                     out_dir: Path) -> None:
    fig = ts.new_figure(NET_POSITION_HEIGHT_IN)
    ax_a, ax_b = fig.subplots(2, 1, sharex=True)
    s = summary.set_index('scenario')
    mean_a = tier.a.mean()
    handles = []
    for sc in scen_list:
        lab = sc['label']
        if lab not in tier.b:
            continue
        b = tier.b[lab]
        line = dict(color=sc['color'], ls=sc['ls'], lw=ts.DATA_LW, zorder=3)
        ax_a.plot(xs, mean_a * xs + b.mean(), **line)
        ax_b.plot(xs, lowest_k_mean(tier.a, b, xs, 1), **line)
        handles.append(Line2D([], [], label=_name(sc, battery_kwh), **line))
        x0 = s.loc[lab, 'x0_kw']
        if np.isfinite(x0):
            for ax, fn in ((ax_a, lambda x: mean_a * x + b.mean()),
                           (ax_b, lambda x: lowest_k_mean(tier.a, b, x, 1)[0])):
                ax.plot([x0], [fn(x0)], 'o', ms=ts.MARKER_SIZE + 1, mfc=sc['color'],
                        mec='white', mew=0.5, zorder=4)
        xw = s.loc[lab, 'x_worst_lo_kw']
        if np.isfinite(xw):
            ax_b.plot([xw], [s.loc[lab, 'W1_at_xworst']], '^', ms=ts.MARKER_SIZE + 1.5,
                      mfc=sc['color'], mec='black', mew=ts.MARKER_EDGE_WIDTH, zorder=5)

    for ax, title in ((ax_a, f'Mean quarter net position ({DOLLAR}/site/quarter)'),
                      (ax_b, f'Worst-quarter net position ({DOLLAR}/site/quarter)')):
        ax.axhline(0, color=ts.SEPARATOR_COLOUR, lw=0.5, zorder=1)
        ax.axvline(0, color=ts.SEPARATOR_COLOUR, lw=0.5, zorder=1)
        ax.set_title(title, fontsize=ts.SUBTITLE_SIZE)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fe._price_tick))
    ax_b.set_xlabel('Cap cover $x$ (kW/site; negative = cap sold)')
    ax_a.set_xlim(xs[0], xs[-1])
    cost = -mean_a
    ax_a.text(0.02, 0.05,
              f"Each kW/site of cover {'costs' if cost >= 0 else 'earns'} "
              f"{DOLLAR}{abs(cost):,.2f}/site/quarter on average",
              transform=ax_a.transAxes, fontsize=ts.ANNOTATION_SIZE, va='bottom',
              bbox=ts.LABEL_BOX, zorder=6)
    handles += [Line2D([], [], marker='o', ls='', mfc='grey', mec='white', mew=0.5,
                       ms=ts.MARKER_SIZE + 1, label='$x_0$'),
                Line2D([], [], marker='^', ls='', mfc='grey', mec='black',
                       mew=ts.MARKER_EDGE_WIDTH, ms=ts.MARKER_SIZE + 1.5,
                       label='$x_{\\mathrm{worst}}$')]
    if note:
        fa._draw_notes(ax_b, note, 'lower right')
    fig.legend(handles=handles, loc='outside lower center', ncol=len(handles),
               frameon=False)
    ts.panel_suptitle(fig, f'Above-{DOLLAR}300 net position against {DOLLAR}300 cap cover'
                      + BASIS_SUFFIX)
    ts.save_figure(fig, out_dir / 'net_position_vs_cover')


def fig_outcomes_at_xworst(tier: Tier, scen_list: list[dict], summary: pd.DataFrame,
                           note: str | None, out_dir: Path) -> None:
    """Each scenario at its own x_worst: the cover held, the worst quarter and
    the mean outcome there, each labelled with its change from No CER."""
    s = summary.set_index('scenario')
    rows = [sc for sc in scen_list if sc['label'] in s.index]
    panels = (
        ('x_worst_lo_kw', 'Cover at $x_{\\mathrm{worst}}$ (kW/site)', 1.0,
         '{:+.2f}', '{:+.2f}'),
        ('W1_at_xworst', f'Worst quarter at $x_{{\\mathrm{{worst}}}}$\n({DOLLAR}/site/quarter)',
         1.0, DOLLAR + '{:,.0f}', '{:+,.0f}'),
        ('mean_N_at_xworst', f'Mean outcome at $x_{{\\mathrm{{worst}}}}$\n({DOLLAR}/site/year)',
         QUARTERS_PER_YEAR, DOLLAR + '{:,.0f}', '{:+,.0f}'),
    )
    fig = ts.new_figure(HEIGHT_IN)
    axes = fig.subplots(1, 3)
    nc = s.loc[NO_CER] if NO_CER in s.index else None
    for ax, (col, title, scale, vfmt, dfmt) in zip(axes, panels):
        vals = [s.loc[sc['label'], col] * scale for sc in rows]
        ref = nc[col] * scale if nc is not None else np.nan
        shown = []
        for i, (sc, v) in enumerate(zip(rows, vals)):
            if not np.isfinite(v):
                ax.text(i, 0, 'no maximum', rotation=90, ha='center', va='bottom',
                        fontsize=ts.ANNOTATION_SIZE, color=ts.REFERENCE_COLOUR)
                continue
            _scenario_bar(ax, i, v, sc, BAR_WIDTH)
            shown.append(v)
            if col == 'x_worst_lo_kw':
                lo_k = s.loc[sc['label'], 'x_worst_k_min_kw']
                hi_k = s.loc[sc['label'], 'x_worst_k_max_kw']
                if np.isfinite(lo_k) and np.isfinite(hi_k):
                    ax.errorbar(i + 0.28, v, yerr=[[max(v - lo_k, 0)], [max(hi_k - v, 0)]],
                                fmt='none', ecolor=ts.REFERENCE_COLOUR, elinewidth=0.6,
                                capsize=2, capthick=0.6, zorder=4)
            txt = _minus(vfmt.format(v).replace(DOLLAR + '-', '-' + DOLLAR))
            if sc['label'] != NO_CER and np.isfinite(ref):
                txt += _minus(f"\n({dfmt.format(v - ref)})")
            fe._signed_label(ax, i, v, txt)
        ax.axhline(0, color=ts.SEPARATOR_COLOUR, lw=0.5, zorder=2)
        # Three lines for Self-consumption: panels this narrow cannot take
        # "consumption" beside its neighbours' labels.
        fe._frame_categories(ax, [_tick(sc['label']).replace('\nconsumption',
                                                              '\nconsump-\ntion')
                                  for sc in rows])
        fe._signed_ylim(ax, shown, pad=0.9 if note else 0.6)   # room for the note
        ax.set_title(title, fontsize=ts.SUBTITLE_SIZE)
        if col != 'x_worst_lo_kw':
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(fe._price_tick))
    if note:
        # Wrapped to the width of a panel.
        note = '\n'.join(textwrap.fill(line, 38) for line in note.split('\n'))
        fa._draw_notes(axes[-1], note, 'upper right')
    ts.panel_suptitle(fig, 'Each scenario at its own worst-quarter cover' + BASIS_SUFFIX)
    ts.save_figure(fig, out_dir / 'outcomes_at_x_worst')


# ── Chart 4: state-month outcomes with no cover ──────────────────────────────

def state_month_table(runs: dict, scen_list: list[dict]) -> pd.DataFrame:
    """One row per (scenario, state, year, month): >$300 interval count,
    above-cap cost L ($/site) and net position with no cover, N = −L.
    Also checks that each quarter's months sum to its L_q."""
    rows = []
    for sc in scen_list:
        for (lab, state), run in sorted(runs.items(), key=lambda kv: kv[0][1]):
            if lab != sc['label']:
                continue
            q_from_months = run.L_month.groupby(
                [run.L_month.index.get_level_values(0),
                 (run.L_month.index.get_level_values(1) - 1) // 3 + 1]).sum()
            diff = (q_from_months - run.L.reindex(q_from_months.index)).abs().max()
            print(f"  CHECK {state} {lab}: months summed to quarters vs L_q, "
                  f"max |diff| {diff:.2e} $/site")
            for (year, month), L in run.L_month.items():
                rows.append(dict(scenario=lab, state=state, year=int(year),
                                 month=int(month),
                                 n_intervals_gt300=int(run.n_month[(year, month)]),
                                 above_cap_cost=float(L), N_at_0=float(-L)))
    return pd.DataFrame(rows)


def _nice_width(span: float, target_bins: int = 50) -> float:
    raw = span / target_bins
    for w in (1, 2, 5, 10, 20, 25, 50, 100, 200, 500):
        if w >= raw:
            return float(w)
    return float(np.ceil(raw / 1000) * 1000)


def fig_state_months(table: pd.DataFrame, scen_list: list[dict], battery_kwh: int,
                     note: str | None, out_dir: Path, reference: str = NO_CER) -> None:
    """Histogram of N_at_0 over state-months: the reference scenario (No CER
    by default, or Self Consumption) as a black outline, battery plans filled.
    Bins are multiples of one width with an edge at zero; months with no >$300
    intervals (N = 0) fall in the bin starting at zero and are hatched. A table
    gives each scenario's counts and worst month."""
    show = [sc for sc in scen_list
            if sc['label'] == reference or sc['ls'] == '--']
    show = [sc for sc in show if sc['label'] in set(table['scenario'])]
    if not show:
        return
    vals = table[table['scenario'].isin([sc['label'] for sc in show])]['N_at_0']
    w = _nice_width(vals.max() - vals.min())
    edges = np.arange(np.floor(vals.min() / w) - 1, np.ceil(vals.max() / w) + 2) * w
    left = edges[:-1]

    fig = ts.new_figure(STATE_MONTH_HEIGHT_IN)
    ax = fig.subplots()
    handles, rows, names = [], [], []
    for sc in show:
        d = table[table['scenario'] == sc['label']]
        nil_mask = d['n_intervals_gt300'] == 0
        n = d.loc[~nil_mask, 'N_at_0'].to_numpy()
        counts, _ = np.histogram(n, bins=edges)
        nil_counts, _ = np.histogram(np.zeros(int(nil_mask.sum())), bins=edges)
        name = _name(sc, battery_kwh)
        if sc['label'] == reference:
            total = counts + nil_counts
            ax.step(edges, np.concatenate([[0], total]), where='pre',
                    color='black', lw=ts.DATA_LW, zorder=4)
            ax.bar(left, nil_counts, width=w, align='edge', fill=False,
                   hatch='///', edgecolor='black', lw=0, zorder=4)
            handles.append(mpatches.Patch(facecolor='none', edgecolor='black',
                                          lw=ts.DATA_LW, label=name))
        else:
            ax.bar(left, nil_counts, width=w, align='edge', color=sc['color'],
                   alpha=0.6, hatch='///', edgecolor='white', lw=0, zorder=2)
            ax.bar(left, counts, bottom=nil_counts, width=w, align='edge',
                   color=sc['color'], alpha=0.6, lw=0, zorder=2)
            handles.append(mpatches.Patch(facecolor=sc['color'], alpha=0.6,
                                          linewidth=0, label=name))
        cost = d[~nil_mask & (d['N_at_0'] < 0)]
        if len(cost):
            worst = cost.loc[cost['N_at_0'].idxmin()]
            worst_txt = (f"−{DOLLAR}{-worst['N_at_0']:,.2f} ({worst['state']} "
                         f"{int(worst['year'])}-{int(worst['month']):02d})")
        else:
            worst_txt = '—'
        rows.append([str(int((n > 0).sum())), str(int((n < 0).sum())),
                     str(int(nil_mask.sum())), worst_txt])
        names.append(_tick(sc['label']).replace('\n', ''))
    handles.append(mpatches.Patch(facecolor=sbf.KEY_COLOUR, edgecolor='white',
                                  hatch='///', linewidth=0,
                                  label=f'No >{DOLLAR}300/MWh intervals'))

    ax.axvline(0, color=ts.SEPARATOR_COLOUR, lw=0.5, zorder=1)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fe._price_tick))
    ax.set_title(f'Above {DOLLAR}300/MWh net earnings per site, by state-month'
                 + BASIS_SUFFIX)
    ax.set_xlabel(f'Net earnings in >{DOLLAR}300/MWh intervals ({DOLLAR}/site per '
                  'month; negative = cost)')
    ax.set_ylabel('State-months')
    top = right = 0.98
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(right, top),
              borderaxespad=0)

    # The table's sizes are set in points from the final axes size.
    fig.draw_without_rendering()
    ax_w, ax_h = ax.get_position().size * fig.get_size_inches() * 72
    fs = mpl.rcParams['font.size']
    char = 0.58 * fs
    span_txt = 'State-month net earnings'
    # The three count columns share the spanning header above them, so each
    # is at least a third of that header's width.
    count_pt = max(3 * char + 14, (len(span_txt) * char + 12) / 3)
    col_pt = [max(len(nm) for nm in names) * char + 12, count_pt,
              count_pt, count_pt,
              max(len(r[3]) for r in rows) * char + 12]
    row_h = [1.0, 0.6] + [1.0] * len(rows)
    row_pt = 2.0 * fs
    tbl_h = sum(row_h) * row_pt / ax_h
    tbl_w = sum(col_pt) / ax_w
    widths = [c / sum(col_pt) for c in col_pt]
    header = [['', '', '', '', ''],
              ['', f'> {DOLLAR}0', f'< {DOLLAR}0', 'Nil', '']]
    # Centred on the midpoint of the y gridline interval nearest the middle of
    # the axes, so its outline sits between gridlines.
    ylo, yhi = ax.get_ylim()
    ticks = [t for t in ax.get_yticks() if ylo <= t <= yhi]
    mids = [(t0 + t1) / 2 for t0, t1 in zip(ticks[:-1], ticks[1:])]
    y_mid = min(mids, key=lambda m: abs(m - (ylo + yhi) / 2)) if mids else (ylo + yhi) / 2
    centre = (y_mid - ylo) / (yhi - ylo)
    bbox = [right - tbl_w, centre - tbl_h / 2, tbl_w, tbl_h]
    tbl = ax.table(cellText=header + [[nm] + r for nm, r in zip(names, rows)],
                   colWidths=widths, cellLoc='center', bbox=bbox, zorder=5)
    # One white backing patch, so cell faces do not paint over neighbours' edges.
    ax.add_patch(mpatches.Rectangle(bbox[:2], bbox[2], bbox[3], transform=ax.transAxes,
                                    facecolor='white', edgecolor='none', zorder=4.9))
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fs)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_height(row_h[r] * row_pt / ax_h)
        cell.set_edgecolor('0.85')
        cell.set_linewidth(0.5)
        cell.set_facecolor('none')
        if c == 0:
            cell.set_text_props(ha='left')
            cell.PAD = 0.04
        if r == 0:
            cell.visible_edges = {0: 'LT', 1: 'LT', 2: 'T', 3: 'TR', 4: 'LTR'}[c]
        if r == 1:
            cell.visible_edges = 'LRB'
    # Spanning header text: "State-month net earnings" over the three count
    # columns (top header row), "Worst cost month" over both rows.
    frac = np.cumsum([0.0] + row_h) / sum(row_h)   # row boundaries from the top
    def _x(col_from, col_to):
        mid = sum(widths[:col_from]) + sum(widths[col_from:col_to]) / 2
        return bbox[0] + bbox[2] * mid
    for x, y, txt in ((_x(1, 4), 1 - (frac[0] + frac[1]) / 2, span_txt),
                      (_x(4, 5), 1 - (frac[0] + frac[2]) / 2,
                       f'Worst cost month\n({DOLLAR}/site)')):
        ax.text(x, bbox[1] + bbox[3] * y, txt, transform=ax.transAxes, ha='center',
                va='center', zorder=6)
    if note:
        fa._draw_notes(ax, note + f'\nBin width {DOLLAR}{w:,.0f}/site; months with no '
                       f'>{DOLLAR}300/MWh intervals (hatched) sit in the bin starting '
                       f'at {DOLLAR}0', 'lower right')
    ts.save_figure(fig, out_dir / ('state_month_outcomes'
                                   + ('' if reference == NO_CER else '_vs_sc')))


def draw_charts(out: Path, scen_list: list[dict], battery_kwh: int, weighted: bool,
                premium: str, notes: bool, reference: str) -> None:
    """Every chart, from the CSVs under out."""
    mpl.rcdefaults()             # financial_analysis sets its own on import
    ts.apply_style()
    tiers = sorted(d for d in out.glob('*_all_years') if d.is_dir()) + [out / 'national']
    for d in tiers:
        if not (d / 'hedge_summary.csv').exists():
            continue
        summary = pd.read_csv(d / 'hedge_summary.csv')
        tier = tier_from_csv(summary, pd.read_csv(d / 'quarter_lines.csv'), premium)
        shown = [sc for sc in scen_list if sc['label'] in set(summary['scenario'])]
        note = _note(tier, weighted, battery_kwh) if notes else None
        flow_note = _note(tier, weighted, battery_kwh, quarters=False) if notes else None
        if (d / 'flow_duration_300.csv').exists():
            fig_flow_duration(tier, pd.read_csv(d / 'flow_duration_300.csv'), shown,
                              summary, flow_note, battery_kwh, d)
        else:
            print(f"  NOTE: no {d / 'flow_duration_300.csv'}; run without --charts-only "
                  "to write it. Flow duration charts skipped.")
        fig_net_position(tier, shown, summary, x_grid(summary), note, battery_kwh, d)
        fig_outcomes_at_xworst(tier, shown, summary, note, d)

    d = out / 'state_months'
    if (d / 'state_months.csv').exists():
        table = pd.read_csv(d / 'state_months.csv')
        if reference not in set(table['scenario']):
            print(f"NOTE: no '{reference}' run; state-month histogram skipped.")
            return
        note = None
        if notes:
            years = _years_label([str(y) for y in sorted(table['year'].unique())])
            note = (f"{', '.join(sorted(table['state'].unique()))}, {years}, "
                    f'{battery_kwh} kWh battery\n'
                    + ('Weighted by PV size within each state' if weighted
                       else 'Unweighted') + '; each state-month counts once')
        fig_state_months(table, scen_list, battery_kwh, note, d, reference)


# ── Output ────────────────────────────────────────────────────────────────────

def print_summary(summary: pd.DataFrame) -> None:
    def f(v, fmt):
        return fmt.format(v) if isinstance(v, (int, float, np.floating)) and np.isfinite(v) \
            else ('inf' if v == np.inf else '-inf' if v == -np.inf else '—')
    print(f"\n  {summary['tier'].iloc[0]} ({summary['years'].iloc[0]}, "
          f"{summary['n_quarters'].iloc[0]} quarters)")
    print(f"  {'Scenario':<20} {'>$300':>6} {'mean m':>7} {'%imp':>5} {'x_0':>6} "
          f"{'x_worst':>8} {'k=2':>6} {'k=4':>6} {'k=10':>6} {'W1(0)':>8} {'W1(xw)':>8} "
          f"{'mean(0)':>8} {'mean(xw)':>8}")
    for _, r in summary.iterrows():
        print(f"  {r['scenario']:<20} {r['n_intervals_gt300']:>6} "
              f"{f(r['mean_import_gt300_kw'], '{:.2f}'):>7} "
              f"{f(100 * r['share_importing_gt300'], '{:.0f}'):>5} "
              f"{f(r['x0_kw'], '{:.2f}'):>6} {f(r['x_worst_lo_kw'], '{:.2f}'):>8} "
              f"{f(r['x_worst_k2_kw'], '{:.2f}'):>6} {f(r['x_worst_k4_kw'], '{:.2f}'):>6} "
              f"{f(r['x_worst_k10_kw'], '{:.2f}'):>6} "
              f"{f(r['W1_at_0'], '{:,.0f}'):>8} {f(r['W1_at_xworst'], '{:,.0f}'):>8} "
              f"{f(r['mean_N_at_0'], '{:,.0f}'):>8} {f(r['mean_N_at_xworst'], '{:,.0f}'):>8}")


def quarter_lines(tier: Tier, summary: pd.DataFrame) -> pd.DataFrame:
    """One row per (quarter, scenario): the quarter's cap terms and the
    scenario's above-cap cost and net position at 0, x_0 and x_worst, with the
    quarter's rank at x_worst (1 = worst).
    """
    s = summary.set_index('scenario')
    nq = len(tier.quarters)
    nan = np.full(nq, np.nan)
    rows = []
    for lab, b in tier.b.items():
        x0, xw = s.loc[lab, 'x0_kw'], s.loc[lab, 'x_worst_lo_kw']
        n_xw = tier.a * xw + b if np.isfinite(xw) else nan
        rank = (pd.Series(n_xw).rank(method='first').to_numpy()
                if np.isfinite(xw) else nan)
        for i, (y, q) in enumerate(tier.quarters):
            rows.append(dict(
                tier=tier.name, quarter=f'{y} Q{q}', scenario=lab,
                csp=tier.csp[i] if tier.csp is not None else np.nan,
                price=tier.price[i] if tier.price is not None else np.nan,
                hours=tier.hours[i] if tier.hours is not None else np.nan,
                slope_per_kw=tier.a[i], above_cap_cost=-b[i],
                N_at_0=b[i], N_at_x0=tier.a[i] * x0 + b[i], N_at_xworst=n_xw[i],
                rank_at_xworst=rank[i]))
    return pd.DataFrame(rows)


def run_tier(tier: Tier, scen_list: list[dict], out_dir: Path) -> pd.DataFrame:
    labels = [sc['label'] for sc in scen_list if sc['label'] in tier.b]
    summary = analyse(tier, labels)
    print_summary(summary)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / 'hedge_summary.csv', index=False)
    quarter_lines(tier, summary).to_csv(out_dir / 'quarter_lines.csv', index=False)
    flow_duration_curves(tier, labels).to_csv(out_dir / 'flow_duration_300.csv',
                                              index=False)
    print(f"  Saved: {out_dir / 'hedge_summary.csv'}, quarter_lines.csv, "
          "flow_duration_300.csv")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', type=Path, default=Path('.'),
                        help='Project root holding data/output/runs, data/clean/nem_prices '
                             'and data/processed/edp (default: cwd)')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Output directory (default: analysis/output/cap_hedge, or '
                             'analysis/output/cap_hedge_sample_mean with '
                             '--premium sample_mean)')
    parser.add_argument('--premium', choices=('start_of_quarter', 'sample_mean'),
                        default='start_of_quarter',
                        help="Cap price charged each quarter: that quarter's "
                             "start-of-quarter ASX price (default), or the state's "
                             "mean of those prices over the sample in every quarter")
    parser.add_argument('--states', nargs='+', default=None,
                        help='Restrict to specific states (default: all found)')
    parser.add_argument('--years', nargs='+', default=None,
                        help='Restrict to specific years (default: all found)')
    parser.add_argument('--battery-kwh', type=int, default=20,
                        help='Battery size for battery scenarios (default: 20)')
    parser.add_argument('--state-month-reference', choices=('no_cer', 'self_consumption'),
                        default='no_cer',
                        help='Reference outline on the state-month histogram: No CER '
                             '(default, state_month_outcomes.png) or Self Consumption '
                             '(state_month_outcomes_vs_sc.png)')
    parser.add_argument('--charts-only', action='store_true',
                        help='Redraw the charts from the CSVs in --output-dir without '
                             'reading the flows')
    parser.add_argument('--no-notes', dest='notes', action='store_false',
                        help='Leave the data notes off, for the thesis')
    parser.add_argument('--core-only', action='store_true',
                        help='Only No CER, Solar Only, Self Consumption and Amber')
    parser.add_argument('--unweighted', action='store_true',
                        help='Straight per-site means instead of size-band weighting')
    parser.add_argument('--weights-file', type=Path,
                        default=Path('analysis/config/pv_size_bands.yaml'),
                        help='Weighting YAML (default: analysis/config/pv_size_bands.yaml)')
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = Path('analysis/output/cap_hedge' + (
            '' if args.premium == 'start_of_quarter' else '_sample_mean'))

    weighted = not args.unweighted
    sm_ref = NO_CER if args.state_month_reference == 'no_cer' else 'Self Consumption'
    if args.charts_only:
        draw_charts(args.output_dir, build_scenarios(args.core_only), args.battery_kwh,
                    weighted, args.premium, args.notes, sm_ref)
        return 0
    weights_cfg = None
    if weighted:
        if not args.weights_file.exists():
            print(f"ERROR: weighted mode requires a weights file, none found at "
                  f"{args.weights_file}", file=sys.stderr)
            return 1
        weights_cfg = load_weights_config(args.weights_file)

    scenarios = discover_scenarios(args.data_root)
    if args.states:
        scenarios = [(s, y) for s, y in scenarios if s in args.states]
    if args.years:
        scenarios = [(s, y) for s, y in scenarios if y in args.years]
    if not scenarios:
        print("ERROR: no scenarios found or none match the filter.", file=sys.stderr)
        return 1
    print(f"Scenarios found: {scenarios}")

    runs_root = args.data_root / 'data' / 'output' / 'runs'
    scen_list = build_scenarios(args.core_only)
    absent = [sc for sc in scen_list if not (runs_root / sc['run_dir']).exists()]
    for sc in absent:
        print(f"NOTE: '{sc['label']}' left out: no run directory {sc['run_dir']}.")
    scen_list = [sc for sc in scen_list if sc not in absent]
    if not any(sc['label'] == NO_CER for sc in scen_list):
        print("ERROR: no No CER run found.", file=sys.stderr)
        return 1
    print("\nLoading flows...")
    runs, rrp = load_all(args.data_root, scenarios, scen_list, weighted, weights_cfg,
                         args.battery_kwh)
    if not runs:
        print("ERROR: no flows.parquet files found.", file=sys.stderr)
        return 1

    print("\nPer-state tiers:")
    tiers: dict[str, Tier] = {}
    for state in sorted({s for s, _ in scenarios}):
        if state not in rrp:
            continue
        tier = build_state_tier(state, runs, rrp[state], scen_list, args.premium)
        if tier is None:
            continue
        tiers[state] = tier
        run_tier(tier, scen_list, args.output_dir / f'{state}_all_years')

    if weighted and len(tiers) > 1:
        print("\nNational tier:")
        national = build_national_tier(tiers, runs, scen_list, weights_cfg['state_mix'])
        if national is not None:
            run_tier(national, scen_list, args.output_dir / 'national')
    else:
        print("\nNOTE: national tier needs weighting and more than one state; skipped.")

    print("\nState-months:")
    table = state_month_table(runs, scen_list)
    sm_dir = args.output_dir / 'state_months'
    sm_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(sm_dir / 'state_months.csv', index=False)
    print(f"  Saved: {sm_dir / 'state_months.csv'}")

    print("\nCharts:")
    draw_charts(args.output_dir, scen_list, args.battery_kwh, weighted, args.premium,
                args.notes, sm_ref)
    print("\nDone.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
