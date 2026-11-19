"""Flow exposure analysis: grid/retailer exposure from physical import/export flows.

Loads flows.parquet files produced by model runs with --save-flows and joins
them with NEM wholesale prices to characterise the grid/retailer exposure of
each coordination strategy, across every state and year the data supports.

See flow_exposure_spec.md for the full design rationale, including the
pooling model (conditional-import and per-plan decomposition charts pool
every discovered state/year into one series; event/peak-window/midday
charts stay per-scenario) and the per-site PV-size weighting (on by
default, same pipeline as financial_analysis.py).

Data loading is a single pass: each flows.parquet is read exactly once per
invocation, feeding both the pooled chart family and the per-scenario chart
family, since flows files are by far the dominant I/O cost.

Run bases (financial_analysis.py's RUN_BASES): perfect_foresight,
conservative and achievable imperfect foresight. Each is written to its own
subdirectory of --output-dir. On the achievable basis the script also writes
the battery cycling CSVs that battery_cycling_figure.py draws the thesis
figure from (docs/battery_cycling_chart_spec.md).

Every chart follows docs/thesis_figure_spec.md through thesis_style, with
data notes that --no-notes leaves off.

Usage (from project root):
    python3 analysis/scripts/flow_exposure_analysis.py --data-root "/Volumes/Manual/Thesis Project"
    python3 analysis/scripts/flow_exposure_analysis.py --basis achievable --cycling-sweep
    python3 analysis/scripts/flow_exposure_analysis.py --unweighted
    python3 analysis/scripts/flow_exposure_analysis.py --states NSW --years 2022

Figures are saved as PNG at 300 dpi.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yaml

# Sizing / weighting pipeline is reused, not duplicated, from
# financial_analysis.py (see flow_exposure_spec.md, Weighting).
sys.path.insert(0, str(Path(__file__).parent))
import financial_analysis as fa
import savings_bar_figure as sbf
import thesis_style as ts
from financial_analysis import (
    load_site_meta, compute_dc_ac_ratio, assign_direct_size,
    estimate_missing_sizes, assign_bands_and_weights, load_weights_config,
)

# local_wall_clock is loaded directly from its file, not via a normal
# `from model.core.localtime import ...`, because model/core/__init__.py
# also imports scenario/tariff/data_context/site_result/paths -- none of
# which localtime.py itself needs (it only uses numpy/pandas/dataclasses).
# A normal package import would execute all of that just to reach one
# small utility function, and would fail entirely if any of those other
# modules have import-time dependencies this analysis script doesn't carry
# (e.g. a solver library). Loading the file directly bypasses the __init__
# chain and only ever depends on localtime.py's own imports.
_REPO_ROOT = Path(__file__).parent.parent.parent
_localtime_path = _REPO_ROOT / 'model' / 'core' / 'localtime.py'
_spec = importlib.util.spec_from_file_location('localtime', _localtime_path)
_localtime_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_localtime_module)
local_wall_clock = _localtime_module.local_wall_clock

BATTERY_SIZES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
INTERVALS_PER_HOUR = 12
INTERVAL_HOURS     = 1.0 / INTERVALS_PER_HOUR
# Imported, not redefined -- see site_exclusions.py for the criterion.
from site_exclusions import EXCLUDED_SITES

RRP_COL = {
    'NSW': 'NSW1_rrp',
    'QLD': 'QLD1_rrp',
    'SA':  'SA1_rrp',
    'VIC': 'VIC1_rrp',
}

# ── Plan registry (independent of financial_analysis.py's RETAIL_PLANS) ──────
#
# No CER and Solar Only have no dispatch decision and are shown only as
# reference lines/bars, never as their own per-plan decomposition chart.
# Self Consumption (Amber) is also shown as a fixed reference: on the
# conditional-import charts as a reference line, and on the per-scenario
# event/window exposure bar charts as a reference column immediately to
# the right of Solar Only (see _bar_order), styled a lighter grey than
# Solar Only since it's one step further from the no-battery baseline.
#
REFERENCE_SCENARIOS = [
    dict(label='No CER (load only)', color='#000000',
        run_dir='full_run_no_cer_amber', has_battery=False),
    dict(label='Solar Only', color='#aaaaaa',
        run_dir='full_run_solar_only_amber', has_battery=False),
    dict(label='Self Consumption (Amber)', color='#888888',
        run_dir='full_run_self_consumption_amber', has_battery=True),
]

COORDINATION_PLANS = [
    dict(label='Amber', sc_mode='SC', lp_mode='LP', color='#2bffa4',
        sc_dir='full_run_self_consumption_amber',
        lp_dir='full_run_optimised_amber',
        default_mode='lp_dir', opt_type='vpp'),
    dict(label='GloBird', sc_mode='SC_GB', lp_mode='LP_GB', color='#e5037f',
        sc_dir='full_run_self_consumption_globird',
        lp_dir='full_run_optimised_globird',
        default_mode='sc_dir', opt_type='vpp'),
    dict(label='AGL', sc_mode='SC_AGL', lp_mode='LP_AGL', color='#1018a1',
        sc_dir='full_run_self_consumption_agl',
        lp_dir='full_run_optimised_agl',
        default_mode='sc_dir', opt_type='user'),
    dict(label='Energy Australia', sc_mode='SC_EA', lp_mode='LP_EA', color='#197e2b',
        sc_dir='full_run_self_consumption_ea',
        lp_dir='full_run_optimised_ea',
        default_mode='lp_dir', opt_type='vpp'),
    dict(label='Origin Battery Lite', sc_mode='SC_ORIGIN', lp_mode='LP_ORIGIN', color='#ffba46',
        sc_dir='full_run_self_consumption_origin_lite',
        lp_dir='full_run_optimised_origin_lite',
        default_mode='sc_dir', opt_type='user'),
]

# Strategies shown in the battery-size chart -- kept to just these two,
# (see flow_exposure_spec.md, Evening peak by battery size).
BATTERY_SIZE_STRATEGIES = [
    dict(run_dir='full_run_self_consumption_amber', label='Self Consumption',
        color='#2ca02c'),
    dict(run_dir='full_run_optimised_amber', label='Amber Optimised',
        color='#2bffa4'),
]

# ── Run bases ─────────────────────────────────────────────────────────────────
#
# The registry above is the perfect_foresight basis. The other bases take each
# plan's (sc_dir, lp_dir) from financial_analysis.PLAN_RUNS by sc_mode, so run
# names have one home. A plan whose optimised run is a capture-ratio
# placeholder in financial_analysis.PLACEHOLDERS (Amber on the conservative
# basis) is left out: a blended saving has no equivalent in flows.
_BASE_PLANS = [dict(p) for p in COORDINATION_PLANS]
_BASE_SIZE_STRATEGIES = [dict(s) for s in BATTERY_SIZE_STRATEGIES]


def plans_for_basis(basis: str) -> list[dict]:
    """The registry with each plan's runs set for one run basis."""
    placeholders = fa.PLACEHOLDERS.get(basis, {})
    plans = []
    for base in _BASE_PLANS:
        if base['lp_mode'] in placeholders:
            print(f"  NOTE: {base['label']} left out of the {basis} basis: "
                  f"its optimised run is a placeholder in financial_analysis.py.")
            continue
        plan = dict(base)
        runs = fa.PLAN_RUNS.get(plan['sc_mode'], {}).get(basis)
        if runs is not None:
            plan['sc_dir'], plan['lp_dir'] = runs
        plans.append(plan)
    return plans


def set_basis(basis: str) -> None:
    """Point the module-level registries, which every chart reads, at one run
    basis. The battery-size chart's optimised strategy follows Amber's
    optimised run, and is dropped where Amber is."""
    global COORDINATION_PLANS, BATTERY_SIZE_STRATEGIES
    COORDINATION_PLANS = plans_for_basis(basis)
    amber = next((p for p in COORDINATION_PLANS if p['sc_mode'] == 'SC'), None)
    strategies = [dict(_BASE_SIZE_STRATEGIES[0])]
    if amber is not None:
        strategies.append(dict(_BASE_SIZE_STRATEGIES[1], run_dir=amber['lp_dir']))
    BATTERY_SIZE_STRATEGIES = strategies


def _excluded_states(plan: dict) -> set[str]:
    """States left out for a plan, from financial_analysis.RETAIL_PLANS by
    sc_mode, so that availability has one home (EnergyAustralia: NSW only)."""
    return next((p['excluded_states'] for p in fa._BASE_PLANS
                 if p['sc_mode'] == plan['sc_mode']), set())


def plans_in_state(state: str) -> list[dict]:
    """The current registry less the plans left out in state."""
    return [p for p in COORDINATION_PLANS if state not in _excluded_states(p)]


def run_excluded(run_dir: str, state: str) -> bool:
    """True where every plan using run_dir is left out in state. A run that
    is also a reference (Amber self-consumption) is never left out."""
    owners = [p for p in COORDINATION_PLANS
              if run_dir in (p['sc_dir'], p['lp_dir'])]
    if any(r['run_dir'] == run_dir for r in REFERENCE_SCENARIOS):
        return False
    return bool(owners) and all(state in _excluded_states(p) for p in owners)


# Extra data note lines on the national charts, e.g. 'EnergyAustralia
# results for NSW only'. Set in run_basis before the national charts.
NATIONAL_NOTE_LINES: list[str] = []


def _national_lines(state: str) -> str:
    return ''.join(f'\n{line}' for line in NATIONAL_NOTE_LINES
                   ) if state == 'National' else ''


# Offer names on the thesis-format charts (docs/battery_cycling_chart_spec.md).
OFFER_NAMES = {
    'Amber': 'Amber',
    'GloBird': 'GloBird',
    'AGL': 'AGL',
    'Energy Australia': 'EnergyAustralia',
    'Origin Battery Lite': 'Origin',
}
SELF_CONSUMPTION_NAME = 'Self-consumption'

# Per-scenario metrics computed in the single loading pass. Each is
# (metric_key, kind, parameter): kind 'price' masks intervals by
# rrp > threshold, kind 'window' masks by local hour in [start, end).
SCENARIO_METRICS = [
    ('ev300',   'price',  300),
    ('ev600',   'price',  600),
    ('ev1000',  'price',  1000),
    ('morning', 'window', (6, 9)),
    ('evening', 'window', (16, 21)),
    ('midday',  'window', (11, 14)),
]


def _compute_metric_means(comb: pd.DataFrame, col: str) -> dict[str, float]:
    """Mean of comb[col] under each SCENARIO_METRICS mask (price threshold
    on 'rrp', or local-hour window on 'local_hour'). Shared by both the
    per-scenario stats and the per-state multi-year pooled stats, since
    both are ultimately "take a comb-shaped frame, apply the same five
    masks" -- the only difference between them is what rows are in comb
    (one scenario's intervals, or several years' pooled together).
    """
    result = {}
    for mkey, kind, param in SCENARIO_METRICS:
        if kind == 'price':
            mask = comb['rrp'] > param
        else:
            h0, h1 = param
            mask = (comb['local_hour'] >= h0) & (comb['local_hour'] < h1)
        result[mkey] = (float(comb.loc[mask, col].mean())
                        if mask.any() else (0.0 if col == 'val' else float('nan')))
    return result


# ── Scenario discovery ────────────────────────────────────────────────────────

_SCENARIO_RE = re.compile(r'^([A-Z]{2,3})_(\d{4})$')


def discover_scenarios(data_root: Path) -> list[tuple[str, str]]:
    """Find (state, year) pairs using the No CER baseline, the one scenario
    guaranteed to exist with no battery-size suffix.
    """
    baseline_scenarios = (data_root / 'data' / 'output' / 'runs'
                          / 'full_run_no_cer_amber' / 'scenarios')
    if not baseline_scenarios.exists():
        return []
    found = []
    for d in sorted(baseline_scenarios.iterdir()):
        if not d.is_dir():
            continue
        m = _SCENARIO_RE.match(d.name)
        if m:
            found.append((m.group(1), m.group(2)))
    return found


# ── Data loading ───────────────────────────────────────────────────────────────

def load_flows(runs_root: Path, run_dir: str, state: str, year: str,
               battery_kwh: int | None = None) -> pd.DataFrame | None:
    """Load one flows.parquet (site columns x 5-min interval rows), dropping
    excluded sites. Returns None if the file doesn't exist. Handles both
    index-stored and column-stored datetime_aest.
    """
    scenario_id = f'{state}_{year}' + (f'_{battery_kwh}kWh' if battery_kwh else '')
    path = runs_root / run_dir / 'scenarios' / scenario_id / 'flows.parquet'
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if 'datetime_aest' in df.columns:
        df = df.set_index('datetime_aest')
    drop_cols = [c for c in df.columns if c in EXCLUDED_SITES]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def get_prices(data_root: Path, year: str,
               price_cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Load (and cache per year) the NEM price file: all four states' RRP
    columns, indexed by datetime_aest.
    """
    if year not in price_cache:
        price_path = (data_root / 'data' / 'clean' / 'nem_prices'
                      / f'{year}_nem_prices.parquet')
        if not price_path.exists():
            raise FileNotFoundError(f"NEM price file not found: {price_path}")
        cols = ['datetime_aest'] + list(RRP_COL.values())
        prices = pd.read_parquet(price_path, columns=cols)
        price_cache[year] = prices.set_index('datetime_aest')
    return price_cache[year]


def weighted_row_mean(df: pd.DataFrame, site_weight: pd.Series | None) -> pd.Series:
    """Mean per interval across site columns.

    If site_weight is None, a plain unweighted mean. Otherwise a weighted
    mean using site_weight, renormalised over whichever sites are actually
    present as columns in this particular flows.parquet (a plan's own run
    may have a slightly different site set than the full metadata
    population).
    """
    if site_weight is None:
        return df.mean(axis=1)
    cols = [c for c in df.columns if c in site_weight.index]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    w = site_weight.loc[cols].to_numpy()
    w_sum = w.sum()
    if w_sum == 0:
        return pd.Series(np.nan, index=df.index)
    w = w / w_sum
    return pd.Series(df[cols].to_numpy() @ w, index=df.index)


def compute_site_weight_for_scenario(
    data_root: Path, weights_cfg: dict, state: str, year: str,
    dc_ac_ratio_cache: dict[str, float],
    all_meta_cache: dict[str, pd.DataFrame],
) -> pd.Series:
    """Compute the (state, year) site_weight Series, reusing the same
    trusted/ratio/regression sizing pipeline as financial_analysis.py, and
    caching the pooled cross-state metadata/ratio per year.
    """
    if year not in all_meta_cache:
        pooled_frames = []
        for other_state in weights_cfg['by_state'].keys():
            try:
                m = load_site_meta(data_root, year, other_state)
                pooled_frames.append(m)
            except FileNotFoundError:
                continue
        pooled_meta = (pd.concat(pooled_frames, ignore_index=True)
                      if pooled_frames else pd.DataFrame(
                          columns=['site_id', 'subarray_dc_rating_kw',
                                  'inverter_ac_rating_kw', 'annual_generation_kwh']))
        ratio = compute_dc_ac_ratio(pooled_meta)
        print(f"  NOTE: {year} pooled DC:AC ratio = {ratio:.3f}.")
        dc_ac_ratio_cache[year] = ratio
        all_meta_cache[year] = assign_direct_size(pooled_meta, ratio)

    meta = load_site_meta(data_root, year, state)
    meta = assign_direct_size(meta, dc_ac_ratio_cache[year])
    meta = estimate_missing_sizes(meta, state, fallback_meta=all_meta_cache[year])
    site_weight, excluded = assign_bands_and_weights(
        meta, state, weights_cfg['bands'], weights_cfg['by_state'])
    if excluded:
        print(f"  {len(excluded)} site(s) excluded from weighted "
              f"calculation for {state} {year}.")
    return site_weight


# ── Single-pass gather: pooled sweeps + per-scenario stats ────────────────────

PRICE_STEP = 100
DISPLAY_MAX = 5000


def _threshold_sweep(pooled: pd.DataFrame, val_col: str) -> tuple[np.ndarray, np.ndarray]:
    """Conditional-mean sweep: for each threshold p, mean of val_col over
    intervals with rrp >= p.
    """
    rrp = pooled['rrp'].dropna()
    rrp_max = float(rrp.max())
    rrp_min = float(rrp.min())
    price_floor = min(-1000, int(rrp_min // PRICE_STEP) * PRICE_STEP)
    thresholds = np.arange(price_floor, rrp_max + PRICE_STEP, PRICE_STEP)
    y_vals = np.full(len(thresholds), np.nan)
    for i, p in enumerate(thresholds):
        mask = pooled['rrp'] >= p
        if mask.sum() >= 1:
            y_vals[i] = float(pooled.loc[mask, val_col].mean())
    return thresholds, y_vals


def _collect_run_specs() -> dict[str, dict]:
    """Deduplicated set of run directories needed, with whether each also
    needs import/export decomposition components (only plan sc/lp dirs do)
    and whether it's a battery-strategy run (needs the _{size}kWh scenario
    suffix) or a no-battery baseline (No CER, Solar Only -- no suffix at
    all). Deduplication matters: e.g. full_run_self_consumption_amber is
    both a reference and Amber's sc_dir, and should be read once, not
    twice.
    """
    specs: dict[str, dict] = {}
    for ref in REFERENCE_SCENARIOS:
        specs.setdefault(ref['run_dir'], {'decomp': False,
                                          'has_battery': ref['has_battery']})
    for plan in COORDINATION_PLANS:
        for d in (plan['sc_dir'], plan['lp_dir']):
            specs.setdefault(d, {'decomp': False, 'has_battery': True})
            specs[d]['decomp'] = True
            specs[d]['has_battery'] = True
    return specs


def gather_all_data(
    data_root: Path, runs_root: Path, scenarios: list[tuple[str, str]],
    weighted: bool, weights_cfg: dict | None, battery_kwh: int,
) -> tuple[dict, dict, dict, dict]:
    """One pass over every needed (run_dir, scenario): each flows.parquet is
    read exactly once, feeding all three chart families.

    Returns (pooled_sweeps, pooled_info, scen_stats, state_year_stats,
             site_weight_cache):
      pooled_sweeps[run_dir] = {'net': (thresholds, y), and for plan dirs
                                also 'import': ..., 'export': ...}
      pooled_info[run_dir]   = {'n_sites': distinct sites across scenarios,
                                'n_scenarios': scenarios with data}
      scen_stats[(state, year)] = {
          'stats':    {run_dir: {metric_key: mean net import kW}},
          'mean_rrp': {metric_key: mean RRP over that metric's intervals},
          'n_by_run': {run_dir: sites used}}
      state_year_stats[state] = same shape as one scen_stats entry, but
          pooling every year available for that state (per run_dir) into
          one set of intervals before applying each metric's mask -- see
          the spec's per-state multi-year tier for why this is pooled
          intervals, not an average of each year's own number. Only populated for states where at
          least one run_dir actually has more than one year of data;
          'years' records which years contributed, per run_dir, since not
          every run necessarily has the same set of years available (e.g.
          a plan missing in one state for one year).
    """
    price_cache: dict[str, pd.DataFrame] = {}
    site_weight_cache: dict[tuple[str, str], pd.Series] = {}
    dc_ac_ratio_cache: dict[str, float] = {}
    all_meta_cache: dict[str, pd.DataFrame] = {}
    # Local (state) wall-clock hour per (state, year), not AEST hour, since
    # the morning/evening/midday windows are specified in real local time
    # (NSW/VIC shift with DST, SA is offset AEST+0.5 plus its own DST; only
    # QLD's local time ever equals AEST). Computed once per scenario from
    # whichever run's index is loaded first for that (state, year) --
    # every run shares the same interval grid for a given scenario, so this
    # is safe to reuse across all of them rather than recomputing per run.
    local_hour_cache: dict[tuple[str, str], np.ndarray] = {}

    pooled_sweeps: dict[str, dict] = {}
    pooled_info: dict[str, dict] = {}
    scen_stats: dict[tuple[str, str], dict] = {
        key: {'stats': {}, 'mean_rrp': None, 'n_by_run': {}} for key in scenarios}
    all_states = sorted({s for s, _ in scenarios})
    state_year_stats: dict[str, dict] = {
        s: {'stats': {}, 'mean_rrp': None, 'n_by_run': {}, 'years': {}}
        for s in all_states}

    summary_rows = []
    specs = _collect_run_specs()

    for run_dir, spec in specs.items():
        parts = {'net': []}
        if spec['decomp']:
            parts['import'] = []
            parts['export'] = []
        sites_used: set[str] = set()
        n_scen = 0
        run_battery_kwh = battery_kwh if spec['has_battery'] else None

        # Per-state accumulators for this run_dir only, reset each run_dir
        # since which years are available can differ run to run.
        state_year_parts: dict[str, list[pd.DataFrame]] = {}
        state_year_n: dict[str, list[int]] = {}
        state_year_labels: dict[str, list[str]] = {}

        for state, year in scenarios:
            if run_excluded(run_dir, state):
                continue
            df = load_flows(runs_root, run_dir, state, year,
                            battery_kwh=run_battery_kwh)
            if df is None:
                print(f"  WARNING: flows.parquet not found for '{run_dir}' "
                      f"{state}_{year}"
                      f"{f'_{run_battery_kwh}kWh' if run_battery_kwh else ''}, "
                      f"skipping.", file=sys.stderr)
                continue
            n_scen += 1

            site_weight = None
            if weighted:
                key = (state, year)
                if key not in site_weight_cache:
                    site_weight_cache[key] = compute_site_weight_for_scenario(
                        data_root, weights_cfg, state, year,
                        dc_ac_ratio_cache, all_meta_cache)
                site_weight = site_weight_cache[key]
                used_cols = set(df.columns) & set(site_weight.index)
            else:
                used_cols = set(df.columns)
            sites_used |= used_cols

            rrp = get_prices(data_root, year, price_cache)[RRP_COL[state]].rename('rrp')
            net_kw = weighted_row_mean(df, site_weight) / INTERVAL_HOURS
            comb = pd.DataFrame({'val': net_kw}).join(rrp, how='left')
            parts['net'].append(comb)

            if spec['decomp']:
                site_kw = df / INTERVAL_HOURS
                imp_kw = weighted_row_mean(site_kw.clip(lower=0), site_weight)
                exp_kw = weighted_row_mean(site_kw.clip(upper=0), site_weight)
                parts['import'].append(pd.DataFrame({'val': imp_kw}).join(rrp, how='left'))
                parts['export'].append(pd.DataFrame({'val': exp_kw}).join(rrp, how='left'))

            # Per-scenario stats (event / window means) from the same load.
            # local_hour is attached as a column on comb (rather than kept
            # as a separate array) so it travels along unchanged when comb
            # rows get pooled across years below.
            if (state, year) not in local_hour_cache:
                local_hour_cache[(state, year)] = local_wall_clock(
                    comb.index, state, INTERVAL_HOURS).hour
            comb = comb.assign(local_hour=local_hour_cache[(state, year)])

            st = scen_stats[(state, year)]
            if st['mean_rrp'] is None:
                st['mean_rrp'] = _compute_metric_means(comb, 'rrp')

            st['stats'][run_dir] = _compute_metric_means(comb, 'val')
            st['n_by_run'][run_dir] = len(used_cols)

            summary_rows.append((run_dir, f'{state}_{year}', len(df.columns),
                                 len(used_cols), float(net_kw.mean())))

            # Accumulate this year's (val, rrp, local_hour) rows for the
            # per-state multi-year pool, keyed by state for this run_dir.
            state_year_parts.setdefault(state, []).append(
                comb[['val', 'rrp', 'local_hour']])
            state_year_n.setdefault(state, []).append(len(used_cols))
            state_year_labels.setdefault(state, []).append(year)

        if parts['net']:
            sweeps = {}
            for component, plist in parts.items():
                pooled = pd.concat(plist, ignore_index=True)
                sweeps[component] = _threshold_sweep(pooled, 'val')
            pooled_sweeps[run_dir] = sweeps
            pooled_info[run_dir] = {'n_sites': len(sites_used), 'n_scenarios': n_scen}

        # Per-state multi-year pooling for this run_dir: only for states
        # where more than one year actually loaded for this specific run
        # (a plan missing in one year for a state just means that state
        # falls back to no multi-year figure for this run this round --
        # handled by fig_scenario_bar simply not finding this run_dir in
        # a given state's stats, same as any other missing-run case).
        for state, year_frames in state_year_parts.items():
            if len(year_frames) < 2:
                continue
            pooled_state = pd.concat(year_frames, ignore_index=True)
            sy = state_year_stats[state]
            if sy['mean_rrp'] is None:
                sy['mean_rrp'] = _compute_metric_means(pooled_state, 'rrp')
            sy['stats'][run_dir] = _compute_metric_means(pooled_state, 'val')
            # Conservative choice: the minimum sites-used across the pooled
            # years, not a range, since fig_scenario_bar's existing n-note
            # logic compares this value across different bars on the same
            # chart (min/max across runs) and expects a plain int -- a
            # within-run year-to-year range is a different axis and is
            # conveyed instead via the year label passed in from main().
            sy['n_by_run'][run_dir] = min(state_year_n[state])
            sy['years'][run_dir] = sorted(state_year_labels[state])

    # Diagnostics: what was actually loaded, so a bad join or missing run is
    # visible at a glance rather than only as a strange-looking chart.
    print()
    print(f"{'Run':<42} {'Scenario':<10} {'Sites':>5} {'Used':>5} {'Mean net kW':>12}")
    print('-' * 80)
    for run_dir, scen, n_all, n_used, mean_kw in summary_rows:
        print(f"{run_dir:<42} {scen:<10} {n_all:>5} {n_used:>5} {mean_kw:>12.3f}")
    print()

    return pooled_sweeps, pooled_info, scen_stats, state_year_stats, site_weight_cache


# ── Chart helpers ──────────────────────────────────────────────────────────────

def _weight_note(weighted: bool, national: bool = False) -> str:
    if not weighted:
        return 'Unweighted'
    return 'Weighted by PV size & state mix' if national else 'Weighted by PV size'


# ── Thesis-format line and bar charts ─────────────────────────────────────────
#
# docs/thesis_figure_spec.md through thesis_style. Offers are named by
# OFFER_NAMES and keep their registry colours; the reference scenarios keep
# theirs. Data notes (state, years, sites, weighting, mean price) are drawn
# unless --no-notes is given.
#
# Bar hatch says which run a bar shows, in the bar's own colour with a white
# hatch: none for the VPP-optimised run, '///' for self-consumption and 'xxx'
# for user-optimised. A bar is exactly one of these. The default-mode chart
# shows each offer's own default run (self-consumption or VPP-optimised) and
# the optimised-mode chart each offer's optimised run (VPP- or
# user-optimised). The Self-consumption reference bar is outlined, as
# elsewhere.

HATCH_SC = '///'
HATCH_USER = 'xxx'
HATCH_VPP = None

REFERENCE_NAMES = {'Solar Only': 'Solar only',
                   'Self Consumption (Amber)': 'Self-consumption (Amber)'}
BAR_TICK_LABELS = {'No CER (load only)': 'No CER\n(load only)',
                   'Solar only': 'Solar\nonly',
                   'Self-consumption (Amber)': 'Self-\nconsumption',
                   'EnergyAustralia': 'Energy\nAustralia'}
SC_REFERENCE_NAME = REFERENCE_NAMES['Self Consumption (Amber)']


def _num2(v: float) -> str:
    """Two decimals with a true minus sign, as the axis tick labels."""
    return f'{v:.2f}'.replace('-', '\u2212')


def _price_tick(v: float, _=None) -> str:
    return f'\u2212\\${-int(v):,}' if v < 0 else f'\\${int(v):,}'


def _display_name(label: str) -> str:
    return REFERENCE_NAMES.get(label, OFFER_NAMES.get(label, label))


def _tick_label(label: str) -> str:
    name = _display_name(label)
    return BAR_TICK_LABELS.get(name, name)


def _threshold_note(scenarios: list[tuple[str, str]], pooled_info: dict,
                    run_dirs: list[str], battery_kwh: int, weighted: bool) -> str:
    states = ', '.join(sorted({s for s, _ in scenarios}))
    years = ', '.join(sorted({y for _, y in scenarios}))
    n_sites = max((pooled_info[r]['n_sites'] for r in run_dirs if r in pooled_info),
                  default=0)
    # Short lines: the note sits low left, under curves that fall to the
    # right, and a wide box would hide them.
    return (f'{states} ({years})\nn = {n_sites} sites\n'
            f'{len(scenarios)} scenario(s) pooled\n{battery_kwh} kWh battery\n'
            f'{_weight_note(weighted)}')


def _draw_threshold_axes(ax, title: str, price_floor: float, ylabel: str) -> None:
    """Axes shared by the conditional-import and decomposition charts, with
    the $300 and $1,000 event thresholds as reference lines."""
    ax.axhline(0, color=ts.SEPARATOR_COLOUR, lw=0.5, zorder=2)
    ax.set_xlim(price_floor, DISPLAY_MAX)
    ax.set_xticks(np.arange(-1000, DISPLAY_MAX + 1, 1000))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_price_tick))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: _num2(v)))
    ax.set_xlabel('Wholesale price threshold (\\$/MWh)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # Headroom above the data for the threshold labels.
    y_bot, y_top = ax.get_ylim()
    y_top += 0.1 * (y_top - y_bot)
    ax.set_ylim(y_bot, y_top)
    for threshold, text in [(300, '\\$300/MWh'), (1000, '\\$1,000/MWh')]:
        ax.axvline(threshold, color=ts.REFERENCE_COLOUR, lw=ts.REFERENCE_LW,
                   ls='--', zorder=2)
        ax.annotate(text, (threshold, y_top), xytext=(2, -2),
                    textcoords='offset points', ha='left', va='top',
                    color=ts.REFERENCE_COLOUR, fontsize=ts.ANNOTATION_SIZE,
                    bbox=ts.LABEL_BOX, zorder=4)


# ── Conditional import charts (default mode / optimised mode) ────────────────

def fig_conditional_import(
    mode: str,  # 'default' or 'optimised'
    pooled_sweeps: dict, pooled_info: dict, scenarios: list[tuple[str, str]],
    weighted: bool, battery_kwh: int, output_dir: Path,
    notes: bool = True, basis_title: str = '',
) -> None:
    price_floor = -1000.0
    plotted_runs = []
    with _thesis_rc():
        fig = ts.new_figure(SINGLE_HEIGHT_IN)
        ax = fig.subplots()

        for ref in REFERENCE_SCENARIOS:
            sweeps = pooled_sweeps.get(ref['run_dir'])
            if sweeps is None:
                print(f"  WARNING: no data for reference '{ref['label']}', skipping.",
                      file=sys.stderr)
                continue
            thresholds, y_vals = sweeps['net']
            price_floor = min(price_floor, float(thresholds[0]))
            ax.plot(thresholds, y_vals, color=ref['color'], ls='-', zorder=3,
                    label=_display_name(ref['label']))
            plotted_runs.append(ref['run_dir'])

        for plan in COORDINATION_PLANS:
            if mode == 'default':
                run_dir = plan[plan['default_mode']]
                ls = '-' if plan['default_mode'] == 'sc_dir' else '--'
            else:
                run_dir = plan['lp_dir']
                ls = '--'
            sweeps = pooled_sweeps.get(run_dir)
            if sweeps is None:
                print(f"  WARNING: no data for plan '{plan['label']}' ({mode}), "
                      f"skipping.", file=sys.stderr)
                continue
            thresholds, y_vals = sweeps['net']
            price_floor = min(price_floor, float(thresholds[0]))
            ax.plot(thresholds, y_vals, color=plan['color'], ls=ls, zorder=3,
                    label=_display_name(plan['label']))
            plotted_runs.append(run_dir)

        if not plotted_runs:
            print(f"  Skipping conditional import ({mode}): nothing to plot.",
                  file=sys.stderr)
            plt.close(fig)
            return

        _draw_threshold_axes(
            ax, f'Conditional mean net import by price threshold, '
                f'{mode} mode{basis_title}',
            price_floor, 'Mean net import at or above threshold (kW)')
        # No part of the axes is clear of the lines, so the key goes below.
        fig.legend(loc='outside lower center', ncol=4, frameon=False)
        if notes:
            fa._draw_notes(ax, _threshold_note(scenarios, pooled_info,
                                               plotted_runs, battery_kwh,
                                               weighted), 'lower left')
        ts.save_figure(fig, output_dir / f'conditional_import_{mode}_mode')


# ── Per-plan decomposition chart ──────────────────────────────────────────────

def fig_decomposition_for_plan(
    plan: dict, pooled_sweeps: dict, pooled_info: dict,
    scenarios: list[tuple[str, str]], weighted: bool, battery_kwh: int,
    output_dir: Path, notes: bool = True, basis_title: str = '',
) -> None:
    """One chart per plan: self-consumption vs optimised, each decomposed
    into net/import/export. Component -> line style; mode -> colour, not
    opacity -- self-consumption always a fixed neutral dark grey, optimised
    always the plan's own colour, since two opacity levels of the same hue
    were too close together to read apart.
    """
    mode_color = {'sc': '#404040', 'lp': plan['color']}
    component_ls = {'net': '-', 'import': '--', 'export': ':'}
    price_floor = -1000.0
    plotted_runs = []
    name = _display_name(plan['label'])

    with _thesis_rc():
        fig = ts.new_figure(SINGLE_HEIGHT_IN)
        ax = fig.subplots()
        for mode_key, run_dir_field, mode_label in [
                ('sc', 'sc_dir', 'Self-consumption'), ('lp', 'lp_dir', 'Optimised')]:
            run_dir = plan[run_dir_field]
            sweeps = pooled_sweeps.get(run_dir)
            if sweeps is None or 'import' not in sweeps:
                print(f"  WARNING: no data for '{plan['label']}' ({mode_label}), "
                      f"skipping that mode.", file=sys.stderr)
                continue
            plotted_runs.append(run_dir)
            for component in ('net', 'import', 'export'):
                thresholds, y_vals = sweeps[component]
                price_floor = min(price_floor, float(thresholds[0]))
                ax.plot(thresholds, y_vals, color=mode_color[mode_key],
                        lw=ts.DATA_LW if component == 'net' else ts.MEMBER_LW,
                        ls=component_ls[component], zorder=3,
                        label=f'{mode_label}, {component}')

        if not plotted_runs:
            print(f"  Skipping decomposition for '{plan['label']}': no data at all.",
                  file=sys.stderr)
            plt.close(fig)
            return

        _draw_threshold_axes(
            ax, f'{name}: self-consumption and optimised net, import and '
                f'export{basis_title}',
            price_floor, 'Mean power at or above threshold (kW)')
        fig.legend(loc='outside lower center', ncol=3, frameon=False)
        if notes:
            fa._draw_notes(ax, _threshold_note(scenarios, pooled_info,
                                               plotted_runs, battery_kwh,
                                               weighted), 'lower left')
        stem = 'decomposition_' + plan['label'].lower().replace(' ', '_')
        ts.save_figure(fig, output_dir / stem)


# ── Per-scenario bar charts: event exposure, peak window, midday export ──────

def _bar_order(mode: str) -> list[tuple[str, str, str, str | None]]:
    """(label, colour, run_dir, hatch) for per-scenario bar charts.

    mode: 'default' -- each plan's own default_mode run.
          'optimised' -- each plan's lp_dir run regardless of its default.

    Bar order is No CER, Solar only, Self-consumption (Amber), then the
    coordination plans.
    """
    out = [(ref['label'], ref['color'], ref['run_dir'], None)
           for ref in REFERENCE_SCENARIOS[:2]]
    sc_ref = REFERENCE_SCENARIOS[2]
    out.append((sc_ref['label'], sbf.KEY_COLOUR, sc_ref['run_dir'], None))
    for plan in COORDINATION_PLANS:
        if mode == 'default' and plan['default_mode'] == 'sc_dir':
            run_dir, hatch = plan['sc_dir'], HATCH_SC
        else:
            run_dir = plan['lp_dir']
            hatch = HATCH_VPP if plan['opt_type'] == 'vpp' else HATCH_USER
        out.append((plan['label'], plan['color'], run_dir, hatch))
    return out


def _hatch_legend_handles(mode: str) -> tuple[list, list[str]]:
    """Key swatches (grey, white hatch) for the run types on this mode's
    chart, with their labels."""
    def swatch(hatch):
        return mpatches.Patch(facecolor=sbf.KEY_COLOUR, edgecolor='white',
                              hatch=hatch, linewidth=0)
    if mode == 'default':
        return ([swatch(HATCH_SC), swatch(HATCH_VPP)],
                ['Self-consumption (default)', 'VPP optimised (default)'])
    return ([swatch(HATCH_VPP), swatch(HATCH_USER)],
            ['VPP optimised', 'User optimised'])


def _bar_style(hatch, colour) -> dict:
    kw = dict(facecolor=colour, linewidth=0)
    if hatch:
        kw.update(edgecolor='white', hatch=hatch)
    return kw


def _signed_label(ax, x: float, value: float, text: str, **kw) -> None:
    """Value label past the end of a bar, below it when the value is
    negative."""
    up = value >= 0
    ax.annotate(text, (x, value), xytext=(0, 1.5 if up else -1.5),
                textcoords='offset points', ha='center',
                va='bottom' if up else 'top', fontsize=ts.ANNOTATION_SIZE,
                linespacing=1.0, zorder=4, **kw)


def _signed_ylim(ax, values, pad: float = 0.12) -> None:
    """Zero-based limits, padded by a fraction of the span on the side of
    each bar's end (room for its value label), and not below zero when
    nothing is negative."""
    lo, hi = min(min(values, default=0), 0), max(max(values, default=0), 0)
    span = (hi - lo) or 1
    ax.set_ylim(lo - pad * span if lo < 0 else 0,
                hi + pad * span if hi > 0 else 1)


def fig_scenario_bar(
    metric_key: str, mode: str, title: str, stem: str,
    state: str, year: str, scen_stat: dict, weighted: bool, battery_kwh: int,
    output_dir: Path, rrp_note_label: str = 'Mean RRP', show_n: bool = True,
    notes: bool = True,
) -> None:
    values = []
    n_counts = []
    for label, color, run_dir, hatch in _bar_order(mode):
        run_stats = scen_stat['stats'].get(run_dir)
        if run_stats is None:
            continue
        values.append((label, color, run_stats[metric_key], hatch))
        n_counts.append(scen_stat['n_by_run'][run_dir])

    if not values:
        print(f"  Skipping {stem}: no data.", file=sys.stderr)
        return

    mean_rrp = scen_stat['mean_rrp'].get(metric_key) if scen_stat['mean_rrp'] else None
    # n is only meaningful for a single (state, year) scenario, where every
    # bar shares one real site count. A multi-year pooled value is a mean
    # over qualifying intervals, not sites, so no single number describes
    # how many sites it represents; n is omitted there and for national
    # calls (show_n=False).
    n_note = ''
    if show_n:
        n_note = (f', n = {n_counts[0]}' if len(set(n_counts)) == 1
                  else f', n = {min(n_counts)}\u2013{max(n_counts)}')
    rrp_str = (f'\n{rrp_note_label}: \\${mean_rrp:,.0f}/MWh'
               if mean_rrp is not None and not np.isnan(mean_rrp) else '')
    note = (f'{state}, {year}{n_note}, {battery_kwh} kWh battery\n'
            f'{_weight_note(weighted, national=(state == "National"))}{rrp_str}'
            f'{_national_lines(state)}')

    vals = [v[2] for v in values]
    with _thesis_rc():
        fig = ts.new_figure(WATERFALL_HEIGHT_IN)
        ax = fig.subplots()
        for i, (label, colour, val, hatch) in enumerate(values):
            if _display_name(label) == SC_REFERENCE_NAME:
                for kw in sbf._sc_patches(colour):
                    ax.bar([i], [val], width=BAR_WIDTH, zorder=3, **kw)
            else:
                ax.bar([i], [val], width=BAR_WIDTH, zorder=3,
                       **_bar_style(hatch, colour))
            if abs(val) > 1e-4:
                _signed_label(ax, i, val, _num2(val))
        _frame_categories(ax, [_tick_label(v[0]) for v in values])
        ax.axhline(0, color=ts.SEPARATOR_COLOUR, lw=0.5, zorder=2)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: _num2(v)))
        _signed_ylim(ax, vals)
        ax.set_ylabel('Mean per-site net import (kW)')
        ax.set_title(title)
        if notes:
            fa._draw_notes(ax, note, 'upper right')
        handles, labels = _hatch_legend_handles(mode)
        fig.legend(handles=handles, labels=labels, loc='outside lower center',
                   ncol=2, frameon=False)
        ts.save_figure(fig, output_dir / stem)


# ── Battery-size chart (Self-consumption vs Amber optimised) ─────────────────

def fig_evening_peak_by_battery_size(
    data_root: Path, runs_root: Path, state: str, year: str,
    weighted: bool, site_weight: pd.Series | None, output_dir: Path,
    notes: bool = True, basis_title: str = '',
) -> None:
    HOUR_START, HOUR_END = 16, 21
    price_cache: dict[str, pd.DataFrame] = {}
    rrp_col = get_prices(data_root, year, price_cache)[RRP_COL[state]].rename('rrp')
    local_hours: np.ndarray | None = None

    def _evening_stats(df_flows: pd.DataFrame) -> tuple[float, float]:
        """(mean net import kW in window, mean RRP in window)."""
        nonlocal local_hours
        mean_kw = weighted_row_mean(df_flows, site_weight if weighted else None) / INTERVAL_HOURS
        combined = pd.DataFrame({'val': mean_kw}).join(rrp_col, how='left')
        if local_hours is None:
            local_hours = local_wall_clock(combined.index, state, INTERVAL_HOURS).hour
        mask = (local_hours >= HOUR_START) & (local_hours < HOUR_END)
        if not mask.any():
            return 0.0, float('nan')
        return (float(combined.loc[mask, 'val'].mean()),
                float(combined.loc[mask, 'rrp'].mean()))

    mean_rrp = None

    ref_values = []
    for ref in REFERENCE_SCENARIOS[:2]:
        df = load_flows(runs_root, ref['run_dir'], state, year)
        if df is None:
            continue
        val, rrp_mean = _evening_stats(df)
        mean_rrp = mean_rrp if mean_rrp is not None else rrp_mean
        ref_values.append((ref['label'], ref['color'], val))

    group_values = {st['label']: {} for st in BATTERY_SIZE_STRATEGIES}
    for st in BATTERY_SIZE_STRATEGIES:
        for size in BATTERY_SIZES:
            df = load_flows(runs_root, st['run_dir'], state, year, battery_kwh=size)
            if df is None:
                continue
            val, rrp_mean = _evening_stats(df)
            mean_rrp = mean_rrp if mean_rrp is not None else rrp_mean
            group_values[st['label']][size] = val

    # The pair of strategies fills 0.68 of a group's spacing (as
    # savings_bar_figure.py); reference bars are one and a half pair-bars wide.
    pair_w, pair_gap, group_gap, ref_gap, ref_space = 0.34, 0.04, 0.32, 0.45, 1.2
    strategies = [st['label'] for st in BATTERY_SIZE_STRATEGIES]
    n_st = len(strategies)
    group_width = n_st * pair_w + (n_st - 1) * pair_gap
    bat_sizes = sorted(BATTERY_SIZES)
    x_ref = np.arange(len(ref_values), dtype=float) * ref_space
    first_centre = (x_ref[-1] + ref_gap + group_width / 2 + 0.5
                    if len(x_ref) else group_width / 2)
    x_bat = first_centre + np.arange(len(bat_sizes)) * (group_width + group_gap)
    offsets = (np.arange(n_st) - (n_st - 1) / 2) * (pair_w + pair_gap)
    all_values = ([v for _, _, v in ref_values]
                  + [v for d in group_values.values() for v in d.values()])

    with _thesis_rc():
        fig = ts.new_figure(WATERFALL_HEIGHT_IN)
        ax = fig.subplots()
        for xpos, (label, color, val) in zip(x_ref, ref_values):
            ax.bar(xpos, val, color=color, width=pair_w * 1.5, linewidth=0,
                   zorder=3)
            if abs(val) > 1e-4:
                _signed_label(ax, xpos, val, _num2(val))

        for si, st in enumerate(BATTERY_SIZE_STRATEGIES):
            is_sc = st['label'] == 'Self Consumption'
            for gi, size in enumerate(bat_sizes):
                if size not in group_values[st['label']]:
                    continue
                val = group_values[st['label']][size]
                xpos = x_bat[gi] + offsets[si]
                if is_sc:
                    for kw in sbf._sc_patches(sbf.KEY_COLOUR):
                        ax.bar([xpos], [val], width=pair_w, zorder=3, **kw)
                else:
                    ax.bar([xpos], [val], width=pair_w, color=st['color'],
                           linewidth=0, zorder=3)
                if abs(val) > 1e-4:
                    # Rotated: twenty bars at one printed width are too narrow
                    # for horizontal labels.
                    up = val >= 0
                    ax.annotate(_num2(val), (xpos, val),
                                xytext=(0, 1.5 if up else -1.5),
                                textcoords='offset points', ha='center',
                                va='bottom' if up else 'top', rotation=90,
                                fontsize=ts.ANNOTATION_SIZE, zorder=4)

        # Separator between the reference bars and the battery groups.
        if len(x_ref) and len(x_bat):
            ax.axvline((x_ref[-1] + pair_w * 0.75 + x_bat[0] - group_width / 2) / 2,
                       color=ts.SEPARATOR_COLOUR, lw=ts.SEPARATOR_LW, ls=':',
                       zorder=1)

        ax.axhline(0, color=ts.SEPARATOR_COLOUR, lw=0.5, zorder=2)
        ax.set_xticks(list(x_ref) + list(x_bat))
        ax.set_xticklabels([_tick_label(l) for l, _, _ in ref_values]
                           + [f'{s:g}' for s in bat_sizes])
        ax.tick_params(axis='x', length=0)
        ax.grid(axis='x', visible=False)
        _signed_ylim(ax, all_values, pad=0.2)
        ax.set_ylabel('Mean per-site net import in window (kW)')
        ax.set_xlabel('Battery capacity (kWh)')
        if len(x_bat):
            # The label belongs to the battery groups, not the reference bars.
            lo, hi = ax.get_xlim()
            ax.xaxis.label.set_x(((x_bat[0] + x_bat[-1]) / 2 - lo) / (hi - lo))
        ax.set_title(f'Net import during the evening peak (4pm\u20139pm) by '
                     f'battery capacity{basis_title}')
        if notes:
            rrp_str = (f'\nMean RRP in window: \\${mean_rrp:,.0f}/MWh'
                       if mean_rrp is not None and not np.isnan(mean_rrp) else '')
            fa._draw_notes(ax, f'{state}, {year}\n'
                               f'{_weight_note(weighted)}{rrp_str}', 'upper right')
        # One entry per strategy drawn: the optimised one is absent on a basis
        # that has no Amber optimised run.
        handles, labels = [], []
        for st in BATTERY_SIZE_STRATEGIES:
            if st['label'] == 'Self Consumption':
                handles.append(mpatches.Patch(
                    facecolor='white', edgecolor=sbf.KEY_COLOUR,
                    hatch=sbf.SC_HATCH, linewidth=sbf.SC_EDGE_LW))
                labels.append(SELF_CONSUMPTION_NAME)
            else:
                handles.append(mpatches.Patch(facecolor=st['color'], linewidth=0))
                labels.append(st['label'].capitalize())
        fig.legend(handles=handles, labels=labels, loc='outside lower center',
                   ncol=len(handles), frameon=False)
        ts.save_figure(fig, output_dir / 'evening_peak_by_battery_size')


# ── Forced-flow decomposition chart ───────────────────────────────────────────
#
# A different kind of chart from everything above: annual kWh totals per
# site, not conditional/window means in kW. See the spec's Forced-flow
# summary for the full methodology (natural/forced/voluntary decomposition, the
# plan-independence assumption behind "natural", and why the multi-year
# tier needs no special pooling method -- these are unconditional sums,
# so mean-of-years and pooling agree exactly).

def _annual_totals(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Per-site annual total import and export (kWh), from a flows-shaped
    DataFrame (site columns, kWh per interval, positive=import,
    negative=export). No interval_hours conversion here -- unlike the kW
    charts elsewhere in this script, these are sums of the raw per-interval
    kWh values, not means converted to a power rate.
    """
    imp_total = df.clip(lower=0).sum()
    exp_total = df.clip(upper=0).abs().sum()
    return imp_total, exp_total


def load_forced_flows(runs_root: Path, run_dir: str, state: str, year: str,
                      battery_kwh: int, reference_columns) -> pd.DataFrame | None:
    """Load vpp_forced_flows.parquet for one run/scenario, if it exists.

    Structurally identical to flows.parquet. Not
    every run has this file -- only plans that actually run VPP events do,
    and presence/absence is the only test used (no inference from
    opt_type or anything else). Reindexed against reference_columns
    (flows.parquet's own site set for the same run/scenario) with missing
    sites filled at zero, since it isn't confirmed the two files always
    share an identical column set.
    """
    scenario_id = f'{state}_{year}_{battery_kwh}kWh'
    path = runs_root / run_dir / 'scenarios' / scenario_id / 'vpp_forced_flows.parquet'
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if 'datetime_aest' in df.columns:
        df = df.set_index('datetime_aest')
    return df.reindex(columns=reference_columns, fill_value=0.0)


def _scenario_dir(runs_root: Path, run_dir: str, state: str, year: str,
                  battery_kwh: int) -> Path:
    return runs_root / run_dir / 'scenarios' / f'{state}_{year}_{battery_kwh}kWh'


def read_run_summary_fields(runs_root: Path, run_dir: str, state: str,
                            year: str, battery_kwh: int) -> dict[str, str]:
    """key=value fields from a scenario's summary.txt (coordination_type,
    retail_tariff, ...). Raises FileNotFoundError if there is no summary."""
    path = _scenario_dir(runs_root, run_dir, state, year, battery_kwh) / 'summary.txt'
    fields = {}
    for token in path.read_text().split():
        if '=' in token:
            key, _, value = token.partition('=')
            fields.setdefault(key, value)
    return fields


def load_forced_totals(runs_root: Path, run_dir: str, state: str, year: str,
                       battery_kwh: int, reference_columns
                       ) -> tuple[pd.Series, pd.Series]:
    """Per-site annual forced import and forced export (kWh, both >= 0),
    indexed by reference_columns (the run's flows.parquet site set).

    Sources, in order:
      1. vpp_forced_flows.parquet (the perfect foresight VPP types);
      2. zero, when summary.txt names a coordination type without VPP events
         (the rolling non-VPP types write the forced-flow columns of
         rolling_diagnostics.csv empty);
      3. f_imp_total_kwh / f_exp_total_kwh in rolling_diagnostics.csv (the
         rolling VPP types, which write no forced-flow parquet: PB-13 in
         docs/review/possible_bugs.md). These are the sums of the billed
         forced split, read after the planner settles the last event.
    A VPP run without either source, or whose diagnostics lack a value for a
    site, raises rather than reading as zero forced flow.
    """
    forced = load_forced_flows(runs_root, run_dir, state, year, battery_kwh,
                               reference_columns)
    if forced is not None:
        return _annual_totals(forced)

    scenario = f'{run_dir} {state}_{year}_{battery_kwh}kWh'
    coordination_type = read_run_summary_fields(
        runs_root, run_dir, state, year, battery_kwh).get('coordination_type', '')
    if 'vpp' not in coordination_type:
        zero = pd.Series(0.0, index=reference_columns)
        return zero, zero.copy()

    cols = ['f_imp_total_kwh', 'f_exp_total_kwh']
    diag_path = (_scenario_dir(runs_root, run_dir, state, year, battery_kwh)
                 / 'rolling_diagnostics.csv')
    diag = pd.read_csv(diag_path) if diag_path.exists() else None
    if diag is None or not set(cols) <= set(diag.columns):
        raise ValueError(
            f"{scenario} is {coordination_type} but has neither "
            f"vpp_forced_flows.parquet nor forced-flow columns in "
            f"rolling_diagnostics.csv.")
    totals = diag.set_index('site_id')[cols].reindex(list(reference_columns))
    bad = totals.index[totals.isna().any(axis=1)].tolist()
    if bad:
        raise ValueError(
            f"{diag_path}: no forced-flow value for {len(bad)} site(s) in "
            f"flows.parquet, e.g. {bad[:3]}.")
    return (totals['f_imp_total_kwh'].astype(float),
            totals['f_exp_total_kwh'].astype(float))


# Column of the forced-flow waterfall each VPP cap is drawn on, by the
# tariff's limit_applies_to.
CAP_COLUMN = {'export_only': 'discharge', 'import_and_export': 'total'}


def read_vpp_cap(runs_root: Path, run_dir: str, state: str, year: str,
                 battery_kwh: int) -> tuple[float, str] | None:
    """(kwh_limit, limit_applies_to) from the vpp block of the retail tariff
    named in the scenario's summary.txt, or None for a run whose coordination
    type has no VPP events.
    Checked against cap_limit_kwh in rolling_diagnostics.csv where present.
    """
    try:
        fields = read_run_summary_fields(runs_root, run_dir, state, year, battery_kwh)
    except FileNotFoundError:
        return None
    tariff = fields.get('retail_tariff')
    if not tariff or 'vpp' not in fields.get('coordination_type', ''):
        return None
    path = _REPO_ROOT / 'model' / 'tariffs' / f'{tariff}.yaml'
    if not path.exists():
        print(f"  WARNING: tariff {path} not found; no cap for {run_dir}.",
              file=sys.stderr)
        return None
    vpp = (yaml.safe_load(path.read_text()) or {}).get('vpp')
    if not vpp or 'kwh_limit' not in vpp:
        return None
    cap = (float(vpp['kwh_limit']), vpp['limit_applies_to'])
    diag_path = (_scenario_dir(runs_root, run_dir, state, year, battery_kwh)
                 / 'rolling_diagnostics.csv')
    if diag_path.exists():
        diag = pd.read_csv(diag_path, usecols=lambda c: c == 'cap_limit_kwh')
        if 'cap_limit_kwh' in diag and not np.allclose(diag['cap_limit_kwh'], cap[0]):
            print(f"  WARNING: {run_dir} {state}_{year}: cap_limit_kwh in "
                  f"rolling_diagnostics.csv differs from {tariff} "
                  f"kwh_limit {cap[0]:g}.", file=sys.stderr)
    return cap


def weighted_series_mean(s: pd.Series, site_weight: pd.Series | None) -> float:
    """Weighted mean of a per-site Series (e.g. annual kWh totals), same
    renormalise-over-the-intersection convention as weighted_row_mean and
    financial_analysis.py's weighted_scalar_mean.
    """
    if site_weight is None:
        return float(s.mean())
    idx = s.index.intersection(site_weight.index)
    if len(idx) == 0:
        return float('nan')
    w = site_weight.loc[idx].to_numpy()
    w_sum = w.sum()
    if w_sum == 0:
        return float('nan')
    return float(np.average(s.loc[idx].to_numpy(), weights=w / w_sum))


def compute_plan_flow_decomposition(
    runs_root: Path, plan: dict, state: str, year: str, battery_kwh: int,
    weighted: bool, site_weight: pd.Series | None,
) -> tuple[dict[str, float], int] | None:
    """The six weighted annual kWh/site quantities for one plan, one
    scenario: natural/forced/voluntary x import/export. Returns
    (values_dict, n_sites_used) or None if this plan's SC or LP flows
    aren't available for this scenario at all.
    """
    sc_df = load_flows(runs_root, plan['sc_dir'], state, year, battery_kwh=battery_kwh)
    lp_df = load_flows(runs_root, plan['lp_dir'], state, year, battery_kwh=battery_kwh)
    if sc_df is None or lp_df is None:
        return None

    sc_imp, sc_exp = _annual_totals(sc_df)
    lp_imp, lp_exp = _annual_totals(lp_df)

    sc_f_imp, sc_f_exp = load_forced_totals(runs_root, plan['sc_dir'], state,
                                            year, battery_kwh, sc_df.columns)

    natural_imp = (sc_imp - sc_f_imp).reindex(lp_df.columns, fill_value=0.0)
    natural_exp = (sc_exp - sc_f_exp).reindex(lp_df.columns, fill_value=0.0)

    forced_imp, forced_exp = load_forced_totals(runs_root, plan['lp_dir'], state,
                                                year, battery_kwh, lp_df.columns)

    voluntary_imp = lp_imp - natural_imp - forced_imp
    voluntary_exp = lp_exp - natural_exp - forced_exp

    sw = site_weight if weighted else None
    n_sites = (len(set(lp_df.columns) & set(site_weight.index))
              if weighted and site_weight is not None else len(lp_df.columns))

    values = {
        'natural_import':   weighted_series_mean(natural_imp, sw),
        'natural_export':   weighted_series_mean(natural_exp, sw),
        'forced_import':    weighted_series_mean(forced_imp, sw),
        'forced_export':    weighted_series_mean(forced_exp, sw),
        'voluntary_import': weighted_series_mean(voluntary_imp, sw),
        'voluntary_export': weighted_series_mean(voluntary_exp, sw),
    }
    return values, n_sites


def load_battery_flows(runs_root: Path, run_dir: str, state: str, year: str,
                       battery_kwh: int) -> pd.DataFrame | None:
    """Load battery_flows.parquet for one run/scenario, if it exists.

    Same wide format as flows.parquet: net battery AC flow per site
    (sol_to_bat_kwh + grid_to_bat_kwh - bat_to_home_kwh - bat_to_grid_kwh),
    positive = charging. Expected to exist for every plan's optimised run,
    but existence is still checked rather than assumed.
    """
    scenario_id = f'{state}_{year}_{battery_kwh}kWh'
    path = runs_root / run_dir / 'scenarios' / scenario_id / 'battery_flows.parquet'
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if 'datetime_aest' in df.columns:
        df = df.set_index('datetime_aest')
    return df


def compute_battery_throughput_for_rundir(
    runs_root: Path, run_dir: str, state: str, year: str, battery_kwh: int,
    weighted: bool, site_weight: pd.Series | None,
) -> tuple[dict[str, float], int] | None:
    """Weighted annual charge/discharge kWh/site for one run directory,
    from battery_flows.parquet. Returns (values_dict, n_sites_used) or
    None if that file doesn't exist for this run/scenario.
    """
    df = load_battery_flows(runs_root, run_dir, state, year, battery_kwh)
    if df is None:
        return None

    charge, discharge = _annual_totals(df)
    sw = site_weight if weighted else None
    n_sites = (len(set(df.columns) & set(site_weight.index))
              if weighted and site_weight is not None else len(df.columns))

    values = {
        'charge_import':      weighted_series_mean(charge, sw),
        'discharge_export':   weighted_series_mean(discharge, sw),
    }
    return values, n_sites


def compute_plan_battery_throughput(
    runs_root: Path, plan: dict, state: str, year: str, battery_kwh: int,
    weighted: bool, site_weight: pd.Series | None,
) -> tuple[dict[str, float], int] | None:
    """Weighted annual charge/discharge kWh/site for one plan's optimised
    run. Thin wrapper over compute_battery_throughput_for_rundir.
    """
    return compute_battery_throughput_for_rundir(
        runs_root, plan['lp_dir'], state, year, battery_kwh, weighted, site_weight)


# Reference run used for the Self Consumption comparison bar on the
# battery throughput chart -- Amber's self-consumption run, since it's a
# plain, VPP-independent baseline (same rationale as the "Self Consumption
# (Amber)" reference line on the pooled conditional-import charts).
SELF_CONSUMPTION_REFERENCE_RUN = 'full_run_self_consumption_amber'
SELF_CONSUMPTION_REFERENCE_LABEL = 'Self Consumption'


# ── Thesis-format summary charts ──────────────────────────────────────────────
#
# docs/thesis_figure_spec.md through thesis_style. Self-consumption comes
# first, then the offers in registry order, named by OFFER_NAMES. On a single
# bar Self-consumption takes the self-consumption style of
# household_saving_by_offer_own_plan (outlined, light hatch, grey key colour);
# in a waterfall its columns are in the grey key colour.
#
# Waterfall columns per offer: discharge from zero (the offer colour with a
# white hatch), charge floating on discharge (the offer colour taken towards
# white, as savings_bar_figure.py's range segment) and their total from zero
# (solid). Discharge starts from zero so a discharge-only cap can be marked on
# it directly.

SINGLE_HEIGHT_IN = 3.0
# Waterfalls carry their column key as a legend row below the axes.
WATERFALL_HEIGHT_IN = 3.2
BAR_WIDTH = 0.72
COLUMN_WIDTH = 0.24
COLUMN_GAP = 0.03
DISCHARGE_HATCH = '///'
WATERFALL_COLUMNS = ('discharge', 'charge', 'total')
WATERFALL_KEY = {'discharge': 'Discharge', 'charge': 'Charge', 'total': 'Total'}
LABEL_LINE_PT = ts.ANNOTATION_SIZE * 1.2     # second line of a value label


@contextlib.contextmanager
def _thesis_rc():
    """The thesis format for one figure, leaving this script's working
    rcParams in place for the charts that have not moved to it."""
    with mpl.rc_context():
        mpl.rcdefaults()
        ts.apply_style()
        yield


def _summary_rows(plan_data: dict, include_sc: bool = True
                  ) -> list[tuple[str, str, str]]:
    """(plan_data label, display name, colour) in display order."""
    rows = []
    if include_sc and SELF_CONSUMPTION_REFERENCE_LABEL in plan_data:
        rows.append((SELF_CONSUMPTION_REFERENCE_LABEL, SELF_CONSUMPTION_NAME,
                     sbf.KEY_COLOUR))
    for p in COORDINATION_PLANS:
        if p['label'] in plan_data:
            rows.append((p['label'], OFFER_NAMES.get(p['label'], p['label']),
                         p['color']))
    return rows


def _summary_note(state: str, year_label: str, n_counts: list[int],
                  show_n: bool, battery_kwh: int, weighted: bool) -> str:
    n_note = ''
    if show_n and n_counts:
        n_note = (f', n = {n_counts[0]}' if len(set(n_counts)) == 1
                  else f', n = {min(n_counts)}–{max(n_counts)}')
    return (f'{state}, {year_label}{n_note}, {battery_kwh} kWh battery\n'
            f'{_weight_note(weighted, national=(state == "National"))}'
            f'{_national_lines(state)}')


def _frame_categories(ax, names: list[str]) -> None:
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.tick_params(axis='x', length=0)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.grid(axis='x', visible=False)


def _column_style(column: str, colour) -> dict:
    if column == 'discharge':
        # The hatch takes the edge colour; linewidth 0 leaves no outline.
        return dict(facecolor=colour, edgecolor='white', hatch=DISCHARGE_HATCH,
                    linewidth=0)
    if column == 'charge':
        return dict(facecolor=sbf._tint(colour, sbf.RANGE_TINT), linewidth=0)
    return dict(facecolor=colour, linewidth=0)


def _column_legend(fig) -> None:
    handles = [mpatches.Patch(**_column_style(c, sbf.KEY_COLOUR))
               for c in WATERFALL_COLUMNS]
    fig.legend(handles=handles, labels=[WATERFALL_KEY[c] for c in WATERFALL_COLUMNS],
               loc='outside lower center', ncol=3, frameon=False)


def draw_waterfall(ax, groups: list[tuple[str, object, float, float]],
                   fmt) -> tuple[dict[str, dict[str, float]], float]:
    """Three columns per group (name, colour, discharge, charge), labelled
    with fmt(value). Returns each group's column x positions, for marks drawn
    on a column, and the tallest column."""
    offsets = np.array([-1, 0, 1]) * (COLUMN_WIDTH + COLUMN_GAP)
    xpos: dict[str, dict[str, float]] = {}
    top = 0.0
    for i, (name, colour, discharge, charge) in enumerate(groups):
        columns = {'discharge': (discharge, 0.0),
                   'charge': (charge, discharge),
                   'total': (discharge + charge, 0.0)}
        xpos[name] = {}
        for (column, (height, bottom)), offset in zip(columns.items(), offsets):
            x = i + offset
            ax.bar([x], [height], bottom=[bottom], width=COLUMN_WIDTH, zorder=3,
                   **_column_style(column, colour))
            sbf._label(ax, x, bottom + height, fmt(height))
            xpos[name][column] = x
        top = max(top, discharge + charge)
    _frame_categories(ax, [g[0] for g in groups])
    return xpos, top


def draw_caps(ax, xpos: dict[str, dict[str, float]],
              caps: dict[str, tuple[float, str]], outer: bool = False) -> None:
    """A short dashed reference line across the column each cap applies to,
    labelled with its value. caps: display name -> (kWh, column). outer: for
    a narrow panel, label on the column's outer edge rather than centred."""
    for name, (limit, column) in caps.items():
        if name not in xpos:
            continue
        x = xpos[name][column]
        half = COLUMN_WIDTH / 2
        ax.plot([x - half, x + half], [limit, limit], color=ts.REFERENCE_COLOUR,
                lw=ts.REFERENCE_LW, ls='--', zorder=4)
        # Centred on the cap line. In a narrow panel (outer=True) it is
        # anchored on the column's outer edge instead, so its box extends
        # away from the neighbouring column rather than over it.
        if outer:
            lx, ha = ((x + half, 'right') if column == 'discharge'
                      else (x - half, 'left'))
        else:
            lx, ha = x, 'center'
        ax.annotate(f'Cap {limit:,.0f}', (lx, limit), xytext=(0, 1.5),
                    textcoords='offset points', ha=ha, va='bottom',
                    color=ts.REFERENCE_COLOUR, fontsize=ts.ANNOTATION_SIZE,
                    bbox=ts.LABEL_BOX, zorder=4)


def draw_cycles_bars(ax, rows: list[tuple[str, object, float]]) -> float:
    """One bar per (name, colour, cycles per day), Self-consumption in its
    outlined style, each labelled to two decimals. Returns the tallest bar."""
    for i, (name, colour, cycles) in enumerate(rows):
        if name == SELF_CONSUMPTION_NAME:
            for kw in sbf._sc_patches(colour):
                ax.bar([i], [cycles], width=BAR_WIDTH, zorder=3, **kw)
        else:
            ax.bar([i], [cycles], width=BAR_WIDTH, color=colour, linewidth=0,
                   zorder=3)
    _frame_categories(ax, [r[0] for r in rows])
    return max((r[2] for r in rows), default=0.0)


def fig_battery_throughput_summary(
    plan_data: dict[str, tuple[dict[str, float], int]],  # label -> (values, n)
    state: str, year_label: str, weighted: bool, battery_kwh: int,
    output_dir: Path, show_n: bool = True, notes: bool = True,
    basis_title: str = '',
) -> None:
    """Battery discharge, charge and total throughput (MWh/yr per site), one
    waterfall per offer from its optimised run, after Self-consumption (Amber's
    self-consumption run). See the spec's battery throughput summary."""
    rows = _summary_rows(plan_data)
    if not rows:
        print("  Skipping battery_throughput_summary: no data.", file=sys.stderr)
        return
    groups = [(name, colour,
               max(plan_data[label][0]['discharge_export'], 0.0) / 1000.0,
               max(plan_data[label][0]['charge_import'], 0.0) / 1000.0)
              for label, name, colour in rows]
    with _thesis_rc():
        fig = ts.new_figure(WATERFALL_HEIGHT_IN)
        ax = fig.subplots()
        _, top = draw_waterfall(ax, groups, lambda v: f'{v:.1f}')
        ax.set_ylim(0, top * 1.12 if top > 0 else 1)
        ax.set_ylabel('Battery throughput (MWh/yr)')
        ax.set_title(f'Battery throughput at {battery_kwh} kWh by offer{basis_title}')
        if notes:
            fa._draw_notes(ax, _summary_note(
                state, year_label, [plan_data[r[0]][1] for r in rows], show_n,
                battery_kwh, weighted), 'upper right')
        _column_legend(fig)
        ts.save_figure(fig, output_dir / 'battery_throughput_summary')


def fig_battery_discharge_summary(
    plan_data: dict[str, tuple[dict[str, float], int]],  # label -> (values, n)
    state: str, year_label: str, weighted: bool, battery_kwh: int,
    output_dir: Path, show_n: bool = True, notes: bool = True,
    basis_title: str = '',
) -> None:
    """Battery discharge as equivalent full cycles per day, one bar per offer
    after Self-consumption, from the same plan_data as the throughput chart.
    Warranty terms are set on discharge alone, so charge is left out. Each
    bar is labelled with its cycles per day and, below that, its annual
    discharge in MWh. See the spec's battery discharge summary."""
    rows = _summary_rows(plan_data)
    if not rows:
        print("  Skipping battery_discharge_summary: no data.", file=sys.stderr)
        return
    discharge_kwh = [max(plan_data[label][0]['discharge_export'], 0.0)
                     for label, _, _ in rows]
    bars = [(name, colour, kwh / battery_kwh / 365.0)
            for (_, name, colour), kwh in zip(rows, discharge_kwh)]
    with _thesis_rc():
        fig = ts.new_figure(SINGLE_HEIGHT_IN)
        ax = fig.subplots()
        top = draw_cycles_bars(ax, bars)
        for i, ((_, _, cycles), kwh) in enumerate(zip(bars, discharge_kwh)):
            ax.annotate(f'({kwh / 1000.0:.1f} MWh/yr)', (i, cycles),
                        xytext=(0, 1.5), textcoords='offset points',
                        ha='center', va='bottom', fontsize=ts.ANNOTATION_SIZE,
                        color=ts.REFERENCE_COLOUR, zorder=4)
            ax.annotate(f'{cycles:.2f}', (i, cycles),
                        xytext=(0, 1.5 + LABEL_LINE_PT), textcoords='offset points',
                        ha='center', va='bottom', fontsize=ts.ANNOTATION_SIZE,
                        zorder=4)
        ax.set_ylim(0, top * 1.2 if top > 0 else 1)
        ax.set_ylabel('Equivalent full cycles per day')
        ax.set_title(f'Battery discharge at {battery_kwh} kWh by offer{basis_title}')
        if notes:
            fa._draw_notes(ax, _summary_note(
                state, year_label, [plan_data[r[0]][1] for r in rows], show_n,
                battery_kwh, weighted), 'upper right')
        ts.save_figure(fig, output_dir / 'battery_discharge_summary')


def fig_forced_flow_summary(
    plan_data: dict[str, tuple[dict[str, float], int]],  # label -> (values, n)
    state: str, year_label: str, weighted: bool, battery_kwh: int,
    output_dir: Path, show_n: bool = True, notes: bool = True,
    basis_title: str = '', caps: dict[str, tuple[float, str]] | None = None,
) -> None:
    """VPP forced discharge, forced charge and their total (kWh/yr per site),
    one waterfall per offer with any forced flow, with each offer's annual cap
    on the column it applies to. caps: plan label -> (kWh, column). Offers
    with no forced flow are left out. See the spec's Forced-flow summary."""
    rows = [r for r in _summary_rows(plan_data, include_sc=False)
            if plan_data[r[0]][0]['forced_export']
            + plan_data[r[0]][0]['forced_import'] > 0]
    if not rows:
        print("  Skipping forced_flow_summary: no data.", file=sys.stderr)
        return
    groups = [(name, colour,
               max(plan_data[label][0]['forced_export'], 0.0),
               max(plan_data[label][0]['forced_import'], 0.0))
              for label, name, colour in rows]
    named_caps = {name: (caps or {})[label] for label, name, _ in rows
                  if label in (caps or {})}
    with _thesis_rc():
        fig = ts.new_figure(WATERFALL_HEIGHT_IN)
        ax = fig.subplots()
        xpos, top = draw_waterfall(ax, groups, lambda v: f'{v:,.0f}')
        draw_caps(ax, xpos, named_caps)
        top = max([top] + [c[0] for c in named_caps.values()])
        # Headroom above the tallest column for the data notes.
        ax.set_ylim(0, top * 1.3 if top > 0 else 1)
        ax.set_ylabel('Forced flow (kWh/yr)')
        ax.set_title(f'VPP forced flows at {battery_kwh} kWh by offer{basis_title}')
        if notes:
            fa._draw_notes(ax, _summary_note(
                state, year_label, [plan_data[r[0]][1] for r in rows], show_n,
                battery_kwh, weighted), 'upper left')
        _column_legend(fig)
        ts.save_figure(fig, output_dir / 'forced_flow_summary')


def resolve_caps(cap_sets: dict[str, set]) -> dict[str, tuple[float, str]]:
    """Plan label -> (kWh, waterfall column) from the caps read across the
    scenarios drawn. A plan whose tariffs disagree gets no cap mark."""
    caps = {}
    for label, found in cap_sets.items():
        if len(found) == 1:
            limit, applies_to = next(iter(found))
            caps[label] = (limit, CAP_COLUMN[applies_to])
        elif found:
            print(f"  WARNING: {label} has differing VPP caps across scenarios "
                  f"{sorted(found)}; no cap mark drawn.", file=sys.stderr)
    return caps


# ── Battery cycling CSVs (docs/battery_cycling_chart_spec.md) ─────────────────

def write_cycling_csv(throughput: dict, forced: dict,
                      caps: dict[str, tuple[float, str]], battery_kwh: int,
                      out_path: Path) -> None:
    """Per offer, at battery_kwh: annual discharge, cycles per day, forced
    discharge, charge and total, and the VPP cap with the column it applies
    to. Self-consumption first, then the offers in registry order."""
    rows = []
    for label, name, _ in _summary_rows(throughput):
        discharge = throughput[label][0]['discharge_export']
        f = forced.get(label, ({},))[0]
        f_dis = f.get('forced_export', np.nan)
        f_chg = f.get('forced_import', np.nan)
        cap = caps.get(label)
        rows.append(dict(
            offer=name, battery_kwh=battery_kwh,
            annual_discharge_kwh=discharge,
            cycles_per_day=discharge / battery_kwh / 365.0,
            forced_discharge_kwh=f_dis, forced_charge_kwh=f_chg,
            forced_total_kwh=f_dis + f_chg,
            cap_kwh=cap[0] if cap else np.nan,
            cap_column=cap[1] if cap else ''))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")


def compute_cycling_sweep(runs_root: Path, scenarios: list[tuple[str, str]],
                          site_weight_cache: dict, state_mix: dict[str, float]
                          ) -> pd.DataFrame:
    """National annual discharge and cycles per day at every battery size, for
    Self-consumption and each offer's optimised run: PV size weighting within
    each scenario, years averaged within a state, states by state_mix."""
    runs = [(SELF_CONSUMPTION_NAME, SELF_CONSUMPTION_REFERENCE_RUN)] + [
        (OFFER_NAMES.get(p['label'], p['label']), p['lp_dir'])
        for p in COORDINATION_PLANS]
    rows = []
    for name, run_dir in runs:
        for size in BATTERY_SIZES:
            per_state: dict[str, list] = {}
            for state, year in scenarios:
                if run_excluded(run_dir, state):
                    continue
                result = compute_battery_throughput_for_rundir(
                    runs_root, run_dir, state, year, size, True,
                    site_weight_cache.get((state, year)))
                if result is None:
                    print(f"  WARNING: no battery_flows.parquet for {run_dir} "
                          f"{state}_{year}_{size}kWh.", file=sys.stderr)
                    continue
                per_state.setdefault(state, []).append(result)
            if not per_state:
                continue
            by_state = {s: {name: _average_plan_data_across_years(r)}
                        for s, r in per_state.items()}
            national = combine_states_plan_data(by_state, state_mix).get(name)
            if national is None:
                continue
            discharge = national[0]['discharge_export']
            rows.append(dict(offer=name, battery_kwh=size,
                             annual_discharge_kwh=discharge,
                             cycles_per_day=discharge / size / 365.0,
                             states=' '.join(sorted(per_state))))
        print(f"  Sweep: {name} done.")
    return pd.DataFrame(rows)


# ── National (state-weighted) aggregation ─────────────────────────────────────
#
# One combination method used uniformly across every chart family (see the
# spec's Weighting, national tier): each state's own representative result is computed first
# (its per-state multi-year pooled/averaged number if it has more than one
# year, or its single year's own result otherwise -- mirroring
# financial_analysis.py's per_state_curves fallback), then combined across
# states via state_mix, renormalised per run/plan/metric over whichever
# states actually have data for it, since not every plan or run necessarily
# has data in every state.

def _state_mix_weighted_average(values: list[float], states: list[str],
                                state_mix: dict[str, float]) -> float | None:
    """Weighted average of one value per state, renormalised over the
    states actually present. Returns None if none of them have any
    state_mix weight at all.
    """
    weights = np.array([state_mix.get(s, 0.0) for s in states])
    w_sum = weights.sum()
    if w_sum == 0:
        return None
    weights = weights / w_sum
    return float(np.average(values, weights=weights))


def combine_states_scenario_stats(per_state_stats: dict[str, dict],
                                  state_mix: dict[str, float]) -> dict:
    """Combine several states' scen_stat-shaped dicts (stats/mean_rrp/
    n_by_run, as produced per (state, year) or per-state-multi-year) into
    one national dict, via state_mix, for the event/peak-window/midday bar
    chart family.
    """
    combined = {'stats': {}, 'mean_rrp': {}, 'n_by_run': {}}
    all_run_dirs = set()
    for s in per_state_stats.values():
        all_run_dirs |= set(s['stats'].keys())

    for run_dir in all_run_dirs:
        states_with_run = [s for s in per_state_stats if run_dir in per_state_stats[s]['stats']]
        if not states_with_run:
            continue
        metrics = {}
        for mkey, _, _ in SCENARIO_METRICS:
            vals = [per_state_stats[s]['stats'][run_dir][mkey] for s in states_with_run]
            avg = _state_mix_weighted_average(vals, states_with_run, state_mix)
            if avg is not None:
                metrics[mkey] = avg
        if metrics:
            combined['stats'][run_dir] = metrics
            combined['n_by_run'][run_dir] = int(np.mean(
                [per_state_stats[s]['n_by_run'][run_dir] for s in states_with_run]))

    for mkey, _, _ in SCENARIO_METRICS:
        vals, states_with_val = [], []
        for s, stat in per_state_stats.items():
            v = stat['mean_rrp'].get(mkey) if stat['mean_rrp'] else None
            if v is not None and not np.isnan(v):
                vals.append(v)
                states_with_val.append(s)
        avg = (_state_mix_weighted_average(vals, states_with_val, state_mix)
              if vals else None)
        combined['mean_rrp'][mkey] = avg if avg is not None else float('nan')

    return combined


def combine_states_plan_data(
    per_state_plan_data: dict[str, dict[str, tuple[dict[str, float], int]]],
    state_mix: dict[str, float],
) -> dict[str, tuple[dict[str, float], int]]:
    """Combine several states' plan_data-shaped dicts (label -> (values,
    n)) into one national dict, via state_mix. Shared by the forced-flow
    and battery-throughput summary charts, since both use this same shape.
    """
    combined = {}
    all_labels = set()
    for pd_ in per_state_plan_data.values():
        all_labels |= set(pd_.keys())

    for label in all_labels:
        states_with_label = [s for s in per_state_plan_data if label in per_state_plan_data[s]]
        if not states_with_label:
            continue
        value_keys = per_state_plan_data[states_with_label[0]][label][0].keys()
        combined_values = {}
        for k in value_keys:
            vals = [per_state_plan_data[s][label][0][k] for s in states_with_label]
            avg = _state_mix_weighted_average(vals, states_with_label, state_mix)
            if avg is not None:
                combined_values[k] = avg
        if len(combined_values) == len(value_keys):
            combined_n = int(np.mean(
                [per_state_plan_data[s][label][1] for s in states_with_label]))
            combined[label] = (combined_values, combined_n)

    return combined


def _average_plan_data_across_years(
    year_results: list[tuple[dict[str, float], int]],
) -> tuple[dict[str, float], int]:
    """Mean of however many years' own (already site-weighted) plan_data
    results are available (1 or more) -- averaging a single year's result
    is just that result, so this works uniformly whether a state has one
    year or several, since these are annual sums (the mean of years equals
    pooling).
    """
    keys = year_results[0][0].keys()
    avg_values = {k: float(np.mean([r[0][k] for r in year_results])) for k in keys}
    avg_n = int(np.mean([r[1] for r in year_results]))
    return avg_values, avg_n


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', type=Path, default=Path('.'),
                        help='Project root; flows, prices, and metadata are '
                             'found at their usual relative paths underneath '
                             'this (default: cwd)')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('analysis/output/exposure'),
                        help='Output directory for figures '
                             '(default: analysis/output/exposure)')
    parser.add_argument('--states', nargs='+', default=None,
                        help='Restrict to specific states (default: all found)')
    parser.add_argument('--years', nargs='+', default=None,
                        help='Restrict to specific years (default: all found)')
    parser.add_argument('--battery-kwh', type=int, default=20,
                        help='Battery size used for every battery-strategy '
                             'chart except the battery-size sweep itself, '
                             'which always shows all sizes (default: 20)')
    parser.add_argument('--state', default=None,
                        help='State for the battery-size chart only '
                             '(default: first state found)')
    parser.add_argument('--year', default=None,
                        help='Year for the battery-size chart only '
                             '(default: first year found)')
    parser.add_argument('--unweighted', action='store_true',
                        help='Use straight per-site means instead of size-band '
                             'weighting (default: weighting is on)')
    parser.add_argument('--weights-file', type=Path,
                        default=Path('analysis/config/pv_size_bands.yaml'),
                        help='Path to the weighting YAML '
                             '(default: analysis/config/pv_size_bands.yaml)')
    parser.add_argument('--basis', nargs='+', default=['all'],
                        choices=list(fa.RUN_BASES) + ['all'],
                        help='Run bases to draw, each into its own '
                             'subdirectory of --output-dir (default: all)')
    parser.add_argument('--no-notes', dest='notes', action='store_false',
                        help='Leave the data notes off the thesis-format '
                             'charts, for the thesis, where they go in the '
                             'caption')
    parser.add_argument('--cycling-sweep', action='store_true',
                        help='On the achievable basis, also write the national '
                             'cycles per day at every battery size '
                             '(battery_cycling_by_size.csv). Reads every '
                             "size's battery_flows.parquet, so it is slow.")
    args = parser.parse_args(argv)

    weighted = not args.unweighted
    weights_cfg = None
    if weighted:
        if not args.weights_file.exists():
            print(f"ERROR: weighted mode requires a weights file, none found at "
                  f"{args.weights_file}", file=sys.stderr)
            return 1
        weights_cfg = load_weights_config(args.weights_file)

    scenarios = discover_scenarios(args.data_root)
    if not scenarios:
        print("ERROR: no scenarios discovered.", file=sys.stderr)
        return 1
    if args.states:
        scenarios = [(s, y) for s, y in scenarios if s in args.states]
    if args.years:
        scenarios = [(s, y) for s, y in scenarios if y in args.years]
    if not scenarios:
        print("ERROR: no scenarios match the requested filter.", file=sys.stderr)
        return 1
    print(f"Scenarios found: {scenarios}")

    runs_root = args.data_root / 'data' / 'output' / 'runs'

    bases = list(fa.RUN_BASES) if 'all' in args.basis else list(
        dict.fromkeys(args.basis))
    for basis in bases:
        print(f"\n=== Run basis: {basis} ===")
        set_basis(basis)
        rc = run_basis(basis, args, scenarios, runs_root, weighted, weights_cfg)
        if rc:
            return rc

    print("\nDone.")
    return 0


def run_basis(basis: str, args, scenarios: list[tuple[str, str]],
              runs_root: Path, weighted: bool, weights_cfg: dict | None) -> int:
    """Every chart for one run basis, into its own subdirectory of
    --output-dir. The registries must already be set for it (set_basis)."""
    output_dir = args.output_dir / fa.RUN_BASES[basis]['folder']
    basis_title = f", {fa.RUN_BASES[basis]['title']}"
    summary_kw = dict(notes=args.notes, basis_title=basis_title)

    print("\nLoading all flows (single pass)...")
    pooled_sweeps, pooled_info, scen_stats, state_year_stats, site_weight_cache = gather_all_data(
        args.data_root, runs_root, scenarios, weighted, weights_cfg,
        args.battery_kwh)

    if not pooled_sweeps:
        print("ERROR: no flows.parquet files found for any run. "
              "Run model scenarios with --save-flows first.", file=sys.stderr)
        return 1

    print("Generating pooled charts...")
    fig_conditional_import('default', pooled_sweeps, pooled_info, scenarios,
                           weighted, args.battery_kwh, output_dir, **summary_kw)
    fig_conditional_import('optimised', pooled_sweeps, pooled_info, scenarios,
                           weighted, args.battery_kwh, output_dir, **summary_kw)
    for plan in COORDINATION_PLANS:
        fig_decomposition_for_plan(plan, pooled_sweeps, pooled_info, scenarios,
                                   weighted, args.battery_kwh, output_dir,
                                   **summary_kw)

    print("\nGenerating per-scenario charts...")
    BAR_METRICS = [
        ('ev300', 'Net import during >\\$300/MWh events', 'event_exposure_300',
         'Mean RRP'),
        ('ev600', 'Net import during >\\$600/MWh events', 'event_exposure_600',
         'Mean RRP'),
        ('ev1000', 'Net import during >\\$1,000/MWh events', 'event_exposure_1000',
         'Mean RRP'),
        ('morning', 'Net import during the morning peak (6am–9am)', 'peak_window_morning',
         'Mean RRP in window'),
        ('evening', 'Net import during the evening peak (4pm–9pm)', 'peak_window_evening',
         'Mean RRP in window'),
        ('midday', 'Net import during midday (11am–2pm)', 'midday_export_exposure',
         'Mean RRP in window'),
    ]
    MODE_TITLE_SUFFIX = {'default': 'default mode', 'optimised': 'optimised mode'}

    state_plan_decomp_years: dict[str, dict[str, list[tuple[dict, int]]]] = {}
    state_cap_sets: dict[str, dict[str, set]] = {}
    all_cap_sets: dict[str, set] = {}
    state_plan_throughput_years: dict[str, dict[str, list[tuple[dict, int]]]] = {}

    for state, year in scenarios:
        scen_stat = scen_stats[(state, year)]
        if not scen_stat['stats']:
            print(f"  Skipping {state} {year}: no runs loaded.", file=sys.stderr)
            continue
        out_dir = output_dir / f'{state}_{year}'
        print(f"\n{state}, {year}:")
        for metric_key, title_base, stem_base, rrp_label in BAR_METRICS:
            for mode in ('default', 'optimised'):
                fig_scenario_bar(
                    metric_key, mode,
                    f'{title_base}, {MODE_TITLE_SUFFIX[mode]}{basis_title}',
                    f'{stem_base}_{mode}_mode',
                    state, year, scen_stat, weighted, args.battery_kwh, out_dir,
                    rrp_note_label=rrp_label, notes=args.notes)

        site_weight = site_weight_cache.get((state, year)) if weighted else None
        scen_cap_sets: dict[str, set] = {}
        for plan in plans_in_state(state):
            cap = read_vpp_cap(runs_root, plan['lp_dir'], state, year,
                               args.battery_kwh)
            if cap is not None:
                for sets in (scen_cap_sets, state_cap_sets.setdefault(state, {}),
                             all_cap_sets):
                    sets.setdefault(plan['label'], set()).add(cap)
        plan_data = {}
        for plan in plans_in_state(state):
            result = compute_plan_flow_decomposition(
                runs_root, plan, state, year, args.battery_kwh, weighted, site_weight)
            if result is not None:
                plan_data[plan['label']] = result
                state_plan_decomp_years.setdefault(state, {}).setdefault(
                    plan['label'], []).append(result)
            else:
                print(f"  WARNING: no SC/LP flows for '{plan['label']}' in "
                      f"{state}_{year}, skipping it on the forced-flow chart.",
                      file=sys.stderr)
        fig_forced_flow_summary(plan_data, state, year, weighted,
                                args.battery_kwh, out_dir,
                                caps=resolve_caps(scen_cap_sets), **summary_kw)

        throughput_data = {}
        for plan in plans_in_state(state):
            result = compute_plan_battery_throughput(
                runs_root, plan, state, year, args.battery_kwh, weighted, site_weight)
            if result is not None:
                throughput_data[plan['label']] = result
                state_plan_throughput_years.setdefault(state, {}).setdefault(
                    plan['label'], []).append(result)
            else:
                print(f"  WARNING: no battery_flows.parquet for '{plan['label']}' "
                      f"optimised run in {state}_{year}, skipping it on the "
                      f"battery throughput chart.", file=sys.stderr)

        sc_ref = compute_battery_throughput_for_rundir(
            runs_root, SELF_CONSUMPTION_REFERENCE_RUN, state, year,
            args.battery_kwh, weighted, site_weight)
        if sc_ref is not None:
            throughput_data[SELF_CONSUMPTION_REFERENCE_LABEL] = sc_ref
            state_plan_throughput_years.setdefault(state, {}).setdefault(
                SELF_CONSUMPTION_REFERENCE_LABEL, []).append(sc_ref)
        else:
            print(f"  WARNING: no battery_flows.parquet for the Self Consumption "
                  f"reference in {state}_{year}, skipping that bar.", file=sys.stderr)

        fig_battery_throughput_summary(throughput_data, state, year, weighted,
                                       args.battery_kwh, out_dir, **summary_kw)
        fig_battery_discharge_summary(throughput_data, state, year, weighted,
                                      args.battery_kwh, out_dir, **summary_kw)

    print("\nGenerating per-state multi-year charts...")
    state_repr_scen_stat: dict[str, dict] = {}
    state_repr_plan_data: dict[str, dict] = {}
    state_repr_throughput_data: dict[str, dict] = {}

    for state in sorted({s for s, _ in scenarios}):
        years_for_state = sorted({y for s, y in scenarios if s == state})
        year_label = (f'{years_for_state[0]}\u2013{years_for_state[-1]}'
                     if len(years_for_state) > 1 else years_for_state[0])

        # Per-state representative result for national aggregation: the
        # multi-year pooled/averaged result if this state has more than
        # one year, otherwise its single year's own result -- same
        # fallback financial_analysis.py uses for per_state_curves.
        sy_stat = state_year_stats[state]
        if len(years_for_state) > 1 and sy_stat['stats']:
            state_repr_scen_stat[state] = sy_stat
        elif len(years_for_state) == 1:
            state_repr_scen_stat[state] = scen_stats[(state, years_for_state[0])]

        plan_years = state_plan_decomp_years.get(state, {})
        if plan_years:
            state_repr_plan_data[state] = {
                label: _average_plan_data_across_years(results)
                for label, results in plan_years.items()}

        throughput_years = state_plan_throughput_years.get(state, {})
        if throughput_years:
            state_repr_throughput_data[state] = {
                label: _average_plan_data_across_years(results)
                for label, results in throughput_years.items()}

        if not sy_stat['stats']:
            continue  # this state never had >1 year for any run_dir -- no
                      # multi-year chart to produce, but its representative
                      # result above is still available for national use
        out_dir = output_dir / f'{state}_all_years'
        print(f"\n{state}, all years ({year_label}):")
        for metric_key, title_base, stem_base, rrp_label in BAR_METRICS:
            for mode in ('default', 'optimised'):
                fig_scenario_bar(
                    metric_key, mode,
                    f'{title_base}, {MODE_TITLE_SUFFIX[mode]}{basis_title}',
                    f'{stem_base}_{mode}_mode',
                    state, year_label, sy_stat, weighted, args.battery_kwh, out_dir,
                    rrp_note_label=rrp_label, show_n=False,
                    notes=args.notes)

        # Mean of each year's own (already site-weighted) six quantities --
        # exactly equivalent to interval-pooling here since these are
        # unconditional annual sums, not conditional means.
        plan_data_avg = {label: vals for label, vals in
                         state_repr_plan_data.get(state, {}).items()
                         if len(plan_years.get(label, [])) > 1}
        if plan_data_avg:
            fig_forced_flow_summary(plan_data_avg, state, year_label,
                                    weighted, args.battery_kwh, out_dir,
                                    show_n=False,
                                    caps=resolve_caps(state_cap_sets.get(state, {})),
                                    **summary_kw)

        throughput_data_avg = {label: vals for label, vals in
                               state_repr_throughput_data.get(state, {}).items()
                               if len(throughput_years.get(label, [])) > 1}
        if throughput_data_avg:
            fig_battery_throughput_summary(throughput_data_avg, state, year_label,
                                           weighted, args.battery_kwh, out_dir,
                                           show_n=False, **summary_kw)
            fig_battery_discharge_summary(throughput_data_avg, state, year_label,
                                          weighted, args.battery_kwh, out_dir,
                                          show_n=False, **summary_kw)

    if weighted and len(state_repr_scen_stat) > 1:
        print("\nGenerating national (state-weighted) charts...")
        all_years_seen = sorted({y for _, y in scenarios})
        national_year_label = (f'{all_years_seen[0]}\u2013{all_years_seen[-1]}'
                               if len(all_years_seen) > 1 else all_years_seen[0])
        national_states = sorted(state_repr_scen_stat.keys())
        modes = {p['sc_mode'] for p in COORDINATION_PLANS}
        NATIONAL_NOTE_LINES[:] = fa.restricted_offer_notes(
            national_states, [p for p in fa._BASE_PLANS if p['sc_mode'] in modes])
        out_dir = output_dir / 'national'
        print(f"States included: {national_states}")

        national_scen_stat = combine_states_scenario_stats(
            state_repr_scen_stat, weights_cfg['state_mix'])
        for metric_key, title_base, stem_base, rrp_label in BAR_METRICS:
            for mode in ('default', 'optimised'):
                fig_scenario_bar(
                    metric_key, mode,
                    f'{title_base}, {MODE_TITLE_SUFFIX[mode]}{basis_title}',
                    f'{stem_base}_{mode}_mode',
                    'National', national_year_label, national_scen_stat,
                    weighted, args.battery_kwh, out_dir, rrp_note_label=rrp_label,
                    show_n=False, notes=args.notes)

        national_caps = resolve_caps(all_cap_sets)
        national_plan_data = {}
        if len(state_repr_plan_data) > 1:
            national_plan_data = combine_states_plan_data(
                state_repr_plan_data, weights_cfg['state_mix'])
            fig_forced_flow_summary(national_plan_data, 'National',
                                    national_year_label, weighted,
                                    args.battery_kwh, out_dir, show_n=False,
                                    caps=national_caps, **summary_kw)

        if len(state_repr_throughput_data) > 1:
            national_throughput_data = combine_states_plan_data(
                state_repr_throughput_data, weights_cfg['state_mix'])
            fig_battery_throughput_summary(national_throughput_data, 'National',
                                           national_year_label, weighted,
                                           args.battery_kwh, out_dir,
                                           show_n=False, **summary_kw)
            fig_battery_discharge_summary(national_throughput_data, 'National',
                                          national_year_label, weighted,
                                          args.battery_kwh, out_dir,
                                          show_n=False, **summary_kw)
            # The thesis battery cycling figure is drawn from these by
            # battery_cycling_figure.py (docs/battery_cycling_chart_spec.md).
            if basis == 'achievable':
                write_cycling_csv(
                    national_throughput_data, national_plan_data, national_caps,
                    args.battery_kwh,
                    out_dir / f'battery_cycling_{args.battery_kwh}kwh.csv')

        if basis == 'achievable' and args.cycling_sweep:
            print("\nBattery cycling by size (all battery_flows.parquet sizes)...")
            sweep = compute_cycling_sweep(runs_root, scenarios, site_weight_cache,
                                          weights_cfg['state_mix'])
            sweep_path = out_dir / 'battery_cycling_by_size.csv'
            sweep_path.parent.mkdir(parents=True, exist_ok=True)
            sweep.to_csv(sweep_path, index=False)
            print(f"  Saved: {sweep_path}")
    elif weighted:
        print("\nNOTE: only one state found, skipping national aggregation.")
    else:
        print("\nNOTE: --unweighted given, skipping national aggregation "
              "(state_mix is a weighting concept).")

    print("\nGenerating battery-size chart...")
    bs_state = args.state or scenarios[0][0]
    bs_year = args.year or scenarios[0][1]
    site_weight = None
    if weighted:
        site_weight = compute_site_weight_for_scenario(
            args.data_root, weights_cfg, bs_state, bs_year, {}, {})
    fig_evening_peak_by_battery_size(args.data_root, runs_root, bs_state, bs_year,
                                     weighted, site_weight, output_dir,
                                     **summary_kw)
    return 0


if __name__ == '__main__':
    sys.exit(main())