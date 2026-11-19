"""
retailer_outcome_analysis.py

Retailer margin per site per year, excluding GST, of three traditional TOU
retailers (AGL, Energy Australia, Origin) for the No CER, Solar Only, Self
Consumption and Optimised cohorts, against Amber's subscription for a
coordinated battery (rolling optimisation).  Optimised is each retailer's VPP
plan with the battery under that retailer's rolling optimisation
(OPTIMISED_RUNS).

Margin of a TOU retailer, per site per year, ex GST:

    import revenue   (retail_usage_cost - greenpower x imports) / 1.1
  + daily revenue    retail_daily_charges / 1.1
  + feed-in paid     retail_feedin_credit          (GST-free, <= 0)
  + other credits    retail_other_credits          (GST-free, <= 0)
  - wholesale        wholesale_usage_cost + wholesale_feedin_credit
  - network          (network usage + feed-in + daily) / 1.1
  - env + market     Amber's environmental and market_operator adders / 1.1
                     x imports
  - hedging          0 (placeholder)

retail_daily_charges is net of any standing VPP credit in the tariff's daily
charges (AGL, EA); retail_other_credits holds the VPP event rewards.

GreenPower is taken out of revenue and cost alike. Imports per site come from
the Amber reference run's flows: the cohorts' dispatch does not depend on the
tariff, which is checked per site (wholesale_usage_cost equal in both bills).
Optimised dispatch differs by retailer, so each Optimised plan takes its
imports from its own run's flows, over the sites of the Amber coordinated run.

Amber's margin is its subscription, retail_daily_charges / 1.1: wholesale,
network and per-kWh adders are passed through. Each Amber bill is checked:
retail_usage_cost = 0.1 x wholesale_usage_cost (GST on spot imports) + adders x
imports, per site.

  {STATE}_all_years/, national/   margin_summary.csv, margin_by_hour.csv,
                                  margin_bars.png, margin_components.png,
                                  margin_spread.png, margin_by_hour.png
  national/margin_by_year.png     national margin per year
  state_year/margin_state_years.csv

Retail and network tariffs are the 2026-27 tariffs, applied to 2021-2025
prices and flows.

Plan sets (--plan-set):
  battery   each retailer's battery plan, TOU only, for every cohort; output in
            analysis/output/retailer_outcomes/.
  standard  each retailer's default no-CER plan for No CER and default solar
            plan for Solar Only (STANDARD_RUNS, re-billed runs, NSW only);
            Self Consumption stays on the battery plan, which a site with a
            battery qualifies for. Output in
            analysis/output/retailer_outcomes_standard/.

Run from repo root:
  python3 analysis/scripts/retailer_outcome_analysis.py --data-root "/Volumes/Manual/Thesis Project"
  python3 analysis/scripts/retailer_outcome_analysis.py --data-root "/Volumes/Manual/Thesis Project" \
      --plan-set standard --states NSW
"""
import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D

import financial_analysis as fa
import flow_exposure_analysis as fe
import thesis_style as ts
from cap_hedge_analysis import _years_label
from financial_analysis import load_weights_config
from flow_exposure_analysis import (
    INTERVAL_HOURS, RRP_COL, compute_site_weight_for_scenario,
    discover_scenarios, get_prices, load_flows,
)
from site_exclusions import EXCLUDED_SITES

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from model.components.billing import GST_RATE  # noqa: E402
from model.core.localtime import local_wall_clock  # noqa: E402
from model.core.prices import build_layer_rates  # noqa: E402
from model.core.tariff import load_network_tariff, load_retail_tariff  # noqa: E402

GST_DIV = 1.0 + GST_RATE
_ABS_TOL = 0.01      # $ per site per year
TARIFF_DIR = REPO_ROOT / 'model' / 'tariffs'
CONFIG_DIR = REPO_ROOT / 'model' / 'configs'

# Cohort key -> (label, Amber reference run, has_battery).
COHORTS = {
    'no_cer': ('No CER', 'full_run_no_cer_amber', False),
    'solar_only': ('Solar Only', 'full_run_solar_only_amber', False),
    'self_consumption': ('Self Consumption', 'full_run_self_consumption_amber', True),
}
AMBER_COORDINATED = ('Amber', 'full_run_rolling_amber')
PLANS = {
    'AGL': dict(runs={
        'no_cer': 'full_run_no_cer_agl',
        'solar_only': 'full_run_solar_only_agl',
        'self_consumption': 'full_run_self_consumption_agl_novpp'}),
    'Energy Australia': dict(runs={
        'no_cer': 'full_run_no_cer_ea',
        'solar_only': 'full_run_solar_only_ea',
        'self_consumption': 'full_run_self_consumption_ea'}),
    'Origin': dict(runs={
        'no_cer': 'full_run_no_cer_origin_starter',
        'solar_only': 'full_run_solar_only_origin_starter',
        'self_consumption': 'full_run_self_consumption_origin_novpp'}),
}
# Optimised cohort: retailer -> rolling run on its VPP plan. Each run has its
# own dispatch; every plan set uses these runs.
OPTIMISED_LABEL = 'Optimised'
# The Optimised runs are the _novpp rolling runs, the conservative basis of
# financial_analysis.RUN_BASES. The Amber coordinated run supplies only
# Amber's subscription, the site set and weights, and the adders, none of
# which depend on dispatch.
BASIS_SUFFIX = ', imperfect foresight (conservative)'
OPTIMISED_RUNS = {
    'AGL': 'full_run_rolling_agl_novpp',
    'Energy Australia': 'full_run_rolling_ea_novpp',
    'Origin': 'full_run_rolling_origin_novpp',
}
# --plan-set standard: each retailer's default no-CER and solar plan in place of
# its battery plan for the No CER and Solar Only cohorts (NSW only so far);
# Self Consumption keeps the battery plan's TOU-only run.
STANDARD_RUNS = {
    'AGL': {'no_cer': 'full_run_no_cer_agl_smart_saver',
            'solar_only': 'full_run_solar_only_agl_solar_savers'},
    'Energy Australia': {'no_cer': 'full_run_no_cer_ea_flexi_plan',
                         'solar_only': 'full_run_solar_only_ea_solar_max'},
    'Origin': {'no_cer': 'full_run_no_cer_origin_everyday',
               'solar_only': 'full_run_solar_only_origin_solar_boost'},
}

REVENUE_COLS = ('import_revenue', 'daily_revenue')
COST_COLS = ('feedin_paid', 'other_credits', 'wholesale', 'network', 'env_market',
             'hedging')
COMPONENT_COLS = REVENUE_COLS + COST_COLS
COMPONENT_LABELS = {
    'import_revenue': 'Import revenue', 'daily_revenue': 'Daily charge revenue',
    'feedin_paid': 'Feed-in paid', 'other_credits': 'Other credits',
    'wholesale': 'Wholesale', 'network': 'Network', 'env_market': 'Environmental + market',
    'hedging': 'Hedging',
}
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


class CheckError(RuntimeError):
    """An input fails a consistency check; the analysis stops."""


# ── Tariff lookups ────────────────────────────────────────────────────────────

def _tariff_components(tariff_id: str, field: str) -> dict[str, float]:
    raw = yaml.safe_load((TARIFF_DIR / f'{tariff_id}.yaml').read_text())
    comp = raw.get(field) or {}
    return {k: float(v) for k, v in comp.items()}


def greenpower_c_per_kwh(tariff_id: str) -> float:
    """The 'greenpower' component of a retail tariff's usage adders, c/kWh inc
    GST; 0 when the tariff has none."""
    return _tariff_components(tariff_id, 'usage_other_c_per_kwh').get('greenpower', 0.0)


def env_market_c_per_kwh_ex_gst(amber_tariff_id: str) -> float:
    """Amber's environmental and market_operator adders, c/kWh ex GST."""
    comp = _tariff_components(amber_tariff_id, 'usage_other_c_per_kwh')
    missing = {'environmental', 'market_operator'} - set(comp)
    if missing:
        raise CheckError(f'{amber_tariff_id}: no {sorted(missing)} adder')
    return (comp['environmental'] + comp['market_operator']) / GST_DIV


def run_tariff_ids(runs_root: Path, run_dir: str) -> tuple[dict, dict]:
    """(retail_tariff_ids, network_tariff_ids) as state -> year -> id, from the
    run's model config, or from rebill_source.yaml for a re-billed run (whose
    network tariffs are its source run's)."""
    cfg_path = CONFIG_DIR / f'{run_dir}.yaml'
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text())
        retail, network = cfg['retail_tariff_ids'], cfg['network_tariff_ids']
    else:
        src_path = runs_root / run_dir / 'rebill_source.yaml'
        if not src_path.exists():
            raise CheckError(f'{run_dir}: no model config and no rebill_source.yaml')
        src = yaml.safe_load(src_path.read_text())
        retail = src['retail_tariff_ids']
        network = yaml.safe_load((REPO_ROOT / src['source_config']).read_text())[
            'network_tariff_ids']

    def norm(d):
        return {st: {str(y): t for y, t in by_year.items()} for st, by_year in d.items()}
    return norm(retail), norm(network)


# ── Per-site margins ──────────────────────────────────────────────────────────

def read_bills(runs_root: Path, run_dir: str, scenario_id: str) -> pd.DataFrame | None:
    path = runs_root / run_dir / 'scenarios' / scenario_id / 'bills.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={'site_id': str}).set_index('site_id')
    return df.drop(index=[s for s in df.index if s in EXCLUDED_SITES])


def tou_margins(bills: pd.DataFrame, imports_kwh: pd.Series, greenpower_c: float,
                env_market_c: float) -> pd.DataFrame:
    """Margin components per site, $/yr ex GST, for a 'tou' plan. Revenue
    positive, costs negative; margin is their sum."""
    if not (bills['plan_type'] == 'tou').all():
        raise CheckError('tou_margins needs tou bills')
    imp = imports_kwh.reindex(bills.index)
    if imp.isna().any():
        raise CheckError(f'no imports for sites {list(imp.index[imp.isna()])}')
    out = pd.DataFrame(index=bills.index)
    out['import_revenue'] = (bills['retail_usage_cost'] - greenpower_c / 100.0 * imp) / GST_DIV
    out['daily_revenue'] = bills['retail_daily_charges'] / GST_DIV
    out['feedin_paid'] = bills['retail_feedin_credit']
    out['other_credits'] = bills['retail_other_credits']
    out['wholesale'] = -(bills['wholesale_usage_cost'] + bills['wholesale_feedin_credit'])
    out['network'] = -(bills['network_usage_cost'] + bills['network_feedin_credit']
                       + bills['network_daily_charges']) / GST_DIV
    out['env_market'] = -env_market_c / 100.0 * imp
    out['hedging'] = 0.0
    out['margin'] = out[list(COMPONENT_COLS)].sum(axis=1)
    out['imports_kwh'] = imp
    return out


def amber_margins(bills: pd.DataFrame, imports_kwh: pd.Series,
                  adders_c: float) -> tuple[pd.DataFrame, float]:
    """Amber's margin per site, $/yr ex GST: the subscription. Returns the
    components frame and the largest per-site residual of
    retail_usage_cost - GST_RATE x wholesale_usage_cost - adders x imports,
    which is zero when the bills carry GST on spot imports and the imports
    match the bills."""
    if not (bills['plan_type'] == 'wholesale_exposed').all():
        raise CheckError('amber_margins needs wholesale_exposed bills')
    imp = imports_kwh.reindex(bills.index)
    if imp.isna().any():
        raise CheckError(f'no imports for sites {list(imp.index[imp.isna()])}')
    resid = (bills['retail_usage_cost'] - GST_RATE * bills['wholesale_usage_cost']
             - adders_c / 100.0 * imp)
    out = pd.DataFrame(0.0, index=bills.index, columns=list(COMPONENT_COLS))
    out['daily_revenue'] = bills['retail_daily_charges'] / GST_DIV
    out['margin'] = out['daily_revenue']
    out['imports_kwh'] = imp
    return out, float(resid.abs().max())


def same_dispatch_diff(bills: pd.DataFrame, reference: pd.DataFrame) -> float:
    """Largest per-site |wholesale_usage_cost| difference between a plan's bills
    and the Amber reference bills: zero when both bill the same flows."""
    missing = set(bills.index) - set(reference.index)
    if missing:
        raise CheckError(f'sites {sorted(missing)[:5]} not in the reference bills')
    return float((bills['wholesale_usage_cost']
                  - reference.loc[bills.index, 'wholesale_usage_cost']).abs().max())


# ── Margin by hour of day ─────────────────────────────────────────────────────

def hourly_import_margin(imp: pd.Series, rates, hours: np.ndarray, greenpower_c: float,
                         env_market_c: float) -> pd.DataFrame:
    """Import margin by local hour, $/yr ex GST, for one cohort-mean import
    series `imp` (kWh per interval per site) and the plan's LayerRates. The
    columns sum over hours to the import terms of tou_margins."""
    x = imp.to_numpy(dtype=float)
    df = pd.DataFrame({
        'hour': hours,
        'import_kwh': x,
        'import_revenue': x * (rates.retail_import - greenpower_c / 100.0) / GST_DIV,
        'network_import': -x * rates.network_import / GST_DIV,
        'wholesale_import': -x * rates.wholesale_import,
        'env_market': -x * env_market_c / 100.0,
    })
    g = df.groupby('hour').sum()
    g['import_margin'] = g[['import_revenue', 'network_import', 'wholesale_import',
                            'env_market']].sum(axis=1)
    return g.reindex(range(24), fill_value=0.0)


def billed_import_residual(bills_mean: pd.Series, imp: pd.Series, rates) -> float:
    """Largest difference between the cohort-mean billed import costs and the
    same costs rebuilt from imp and the layer rates."""
    x = imp.to_numpy(dtype=float)
    pairs = (('retail_usage_cost', rates.retail_import),
             ('network_usage_cost', rates.network_import),
             ('wholesale_usage_cost', rates.wholesale_import))
    return max(abs(float(bills_mean[c]) - float(np.dot(x, r))) for c, r in pairs)


# ── Weighting and aggregation ─────────────────────────────────────────────────

def weighted_mean(df: pd.DataFrame, w: pd.Series, cols) -> pd.Series:
    w = w.reindex(df.index)
    if w.isna().any() or w.sum() <= 0:
        raise CheckError('weights missing for some sites')
    return (df[list(cols)].mul(w, axis=0).sum() / w.sum())


def weighted_quantiles(values: np.ndarray, weights: np.ndarray, qs) -> np.ndarray:
    """Quantiles of a weighted sample: the smallest value whose cumulative
    weight share reaches q (inverted CDF), so a weight of k equals k copies."""
    order = np.argsort(values, kind='stable')
    v, w = np.asarray(values)[order], np.asarray(weights, dtype=float)[order]
    cw = np.cumsum(w) / w.sum()
    i = np.searchsorted(cw, np.asarray(qs, dtype=float) - 1e-12, side='left')
    return v[np.minimum(i, len(v) - 1)]


def national_mix(rows: pd.DataFrame, by: list[str], cols: list[str],
                 state_mix: dict[str, float]) -> pd.DataFrame:
    """state_mix-weighted mean over the 'state' column of `rows`, within each
    `by` group, renormalised over the states present."""
    out = []
    for key, g in rows.groupby(by, sort=False):
        g = g[g['state'].map(lambda s: state_mix.get(s, 0.0)) > 0]
        if g.empty:
            continue
        w = g['state'].map(state_mix).to_numpy()
        w = w / w.sum()
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        row['states'] = ','.join(g['state'])
        for c in cols:
            row[c] = float(np.dot(w, g[c].to_numpy()))
        out.append(row)
    return pd.DataFrame(out)


# ── Loading ───────────────────────────────────────────────────────────────────

def _scenario_id(state: str, year: str, battery_kwh: int | None) -> str:
    return f'{state}_{year}' + (f'_{battery_kwh}kWh' if battery_kwh else '')


def _time_axis(flows: pd.DataFrame, price_all: pd.Series, sid: str, state: str):
    """(tz-aware index, prices on it, local hour of each interval) for a flows
    frame."""
    idx = flows.index
    if idx.tz is None:
        idx = idx.tz_localize('Etc/GMT-10')
    prices = pd.Series(price_all.reindex(flows.index).to_numpy(), index=idx)
    if prices.isna().any():
        raise CheckError(f'{sid}: {int(prices.isna().sum())} intervals without a price')
    return idx, prices, local_wall_clock(idx, state, INTERVAL_HOURS).hour


def _plan_margins(runs_root: Path, run: str, sid: str, state: str, year: str,
                  sites: list, imports: pd.Series, w: pd.Series, imp_mean: pd.Series,
                  idx, prices: pd.Series, hours, network, env_c: float,
                  reference: tuple[str, pd.DataFrame] | None):
    """Margin components per site and import margin by hour for one TOU plan
    run, or None when it has no bills. With reference (run name, bills), the
    plan's bills must bill the same flows as the reference run. Returns
    (components, hours frame, dispatch difference, hourly residual)."""
    bills = read_bills(runs_root, run, sid)
    if bills is None:
        print(f"  WARNING: no bills for {run} {sid}, skipping.", file=sys.stderr)
        return None
    missing = set(sites) - set(bills.index)
    if missing:
        raise CheckError(f'{run} {sid}: sites {sorted(missing)[:5]} missing')
    bills = bills.loc[sites]
    d_disp = np.nan
    if reference is not None:
        ref_run, ref_bills = reference
        d_disp = same_dispatch_diff(bills, ref_bills)
        if d_disp >= _ABS_TOL:
            raise CheckError(f'{run} {sid}: wholesale_usage_cost differs from '
                             f'{ref_run} by ${d_disp:.4f}; not the same dispatch')
    rid, _ = run_tariff_ids(runs_root, run)
    tid = rid[state][year]
    gp = greenpower_c_per_kwh(tid)
    comp = tou_margins(bills, imports, gp, env_c)
    rates = build_layer_rates(idx, prices, network, load_retail_tariff(tid, TARIFF_DIR),
                              state, INTERVAL_HOURS)
    billed = weighted_mean(bills, w, ('retail_usage_cost', 'network_usage_cost',
                                      'wholesale_usage_cost'))
    r_hour = billed_import_residual(billed, imp_mean, rates)
    if r_hour >= _ABS_TOL:
        raise CheckError(f'{run} {sid}: import costs rebuilt by hour differ '
                         f'from the bills by ${r_hour:.4f}')
    h = hourly_import_margin(imp_mean, rates, hours, gp, env_c).reset_index()
    return comp, h, d_disp, r_hour


def _hour_rows(h: pd.DataFrame, cohort: str, plan: str, state: str,
               year: str) -> pd.DataFrame:
    h = h.copy()
    h.insert(0, 'year', year)
    h.insert(0, 'state', state)
    h.insert(0, 'plan', plan)
    h.insert(0, 'cohort', cohort)
    return h


def load_all(data_root: Path, scenarios: list[tuple[str, str]], weighted: bool,
             weights_cfg: dict | None, battery_kwh: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One pass over every (cohort, state, year). Returns (sites, hours):
    sites has one row per (cohort, plan, state, year, site) with its weight,
    normalised to sum to 1 within the group, and margin components; hours has
    one row per (cohort, plan, state, year, hour) with the cohort-mean import
    margin terms.  The Optimised cohort is taken over the sites of the Amber
    coordinated run, with the same weights."""
    runs_root = data_root / 'data' / 'output' / 'runs'
    price_cache: dict[str, pd.DataFrame] = {}
    dc_ac_cache: dict[str, float] = {}
    meta_cache: dict[str, pd.DataFrame] = {}
    site_frames, hour_frames, log = [], [], []
    cohorts = dict(COHORTS)
    cohorts['coordinated'] = (AMBER_COORDINATED[0] + ' (coordinated)',
                              AMBER_COORDINATED[1], True)

    for state, year in scenarios:
        site_weight = (compute_site_weight_for_scenario(
            data_root, weights_cfg, state, year, dc_ac_cache, meta_cache)
            if weighted else None)
        price_all = get_prices(data_root, year, price_cache)[RRP_COL[state]]
        coordinated = None           # (sites, weights, env_c) of the Amber run

        for ckey, (clabel, amber_run, has_bat) in cohorts.items():
            kwh = battery_kwh if has_bat else None
            sid = _scenario_id(state, year, kwh)
            flows = load_flows(runs_root, amber_run, state, year, battery_kwh=kwh)
            amber_bills = read_bills(runs_root, amber_run, sid)
            if flows is None or amber_bills is None:
                print(f"  WARNING: no flows or bills for {amber_run} {sid}, skipping.",
                      file=sys.stderr)
                continue
            flows.columns = [str(c) for c in flows.columns]
            sites = [s for s in amber_bills.index if s in set(flows.columns)
                     and (site_weight is None or s in site_weight.index)]
            if not sites:
                raise CheckError(f'{amber_run} {sid}: no sites in common')
            w = (pd.Series(1.0, index=sites) if site_weight is None
                 else site_weight.loc[sites].astype(float))
            w = w / w.sum()
            imports = flows[sites].clip(lower=0).sum()

            a_retail, a_network = run_tariff_ids(runs_root, amber_run)
            amber_tid = a_retail[state][year]
            adders = sum(_tariff_components(amber_tid, 'usage_other_c_per_kwh').values())
            env_c = env_market_c_per_kwh_ex_gst(amber_tid)
            comp, resid = amber_margins(amber_bills.loc[sites], imports, adders)
            if resid >= _ABS_TOL:
                raise CheckError(
                    f'{amber_run} {sid}: retail_usage_cost differs from GST on spot '
                    f'imports + adders x imports by ${resid:.4f} (bills not GST-'
                    f'patched, or imports do not match the bills)')
            site_frames.append(_tag(comp, w, clabel, 'Amber', state, year))
            log.append((clabel, 'Amber', sid, len(sites), resid, np.nan, np.nan))
            if ckey == 'coordinated':
                coordinated = (sites, w, env_c)
                continue

            idx, prices, hours = _time_axis(flows, price_all, sid, state)
            imp_mean = pd.Series(flows[sites].clip(lower=0).to_numpy(dtype=float)
                                 @ w.to_numpy(), index=idx)
            network = load_network_tariff(a_network[state][year], TARIFF_DIR)

            for plan, spec in PLANS.items():
                run = spec['runs'][ckey]
                res = _plan_margins(runs_root, run, sid, state, year, sites, imports, w,
                                    imp_mean, idx, prices, hours, network, env_c,
                                    (amber_run, amber_bills))
                if res is None:
                    continue
                comp, h, d_disp, r_hour = res
                site_frames.append(_tag(comp, w, clabel, plan, state, year))
                hour_frames.append(_hour_rows(h, clabel, plan, state, year))
                log.append((clabel, plan, sid, len(sites), np.nan, d_disp, r_hour))

        # Optimised: each retailer's own rolling dispatch on its VPP plan.
        if coordinated is None:
            continue
        sites, w, env_c = coordinated
        sid = _scenario_id(state, year, battery_kwh)
        for plan, run in OPTIMISED_RUNS.items():
            flows = load_flows(runs_root, run, state, year, battery_kwh=battery_kwh)
            if flows is None:
                print(f"  WARNING: no flows for {run} {sid}, skipping.", file=sys.stderr)
                continue
            flows.columns = [str(c) for c in flows.columns]
            missing = set(sites) - set(flows.columns)
            if missing:
                raise CheckError(f'{run} {sid}: sites {sorted(missing)[:5]} not in flows')
            idx, prices, hours = _time_axis(flows, price_all, sid, state)
            imports = flows[sites].clip(lower=0).sum()
            imp_mean = pd.Series(flows[sites].clip(lower=0).to_numpy(dtype=float)
                                 @ w.to_numpy(), index=idx)
            _, o_network = run_tariff_ids(runs_root, run)
            network = load_network_tariff(o_network[state][year], TARIFF_DIR)
            res = _plan_margins(runs_root, run, sid, state, year, sites, imports, w,
                                imp_mean, idx, prices, hours, network, env_c, None)
            if res is None:
                continue
            comp, h, d_disp, r_hour = res
            site_frames.append(_tag(comp, w, OPTIMISED_LABEL, plan, state, year))
            hour_frames.append(_hour_rows(h, OPTIMISED_LABEL, plan, state, year))
            log.append((OPTIMISED_LABEL, plan, sid, len(sites), np.nan, d_disp, r_hour))

    print(f"\n{'Cohort':<26} {'Plan':<17} {'Scenario':<16} {'Sites':>5} "
          f"{'Amber resid':>11} {'Dispatch':>9} {'Hourly':>9}")
    for r in log:
        print(f"{r[0]:<26} {r[1]:<17} {r[2]:<16} {r[3]:>5} {r[4]:>11.2e} "
              f"{r[5]:>9.2e} {r[6]:>9.2e}")
    sites_df = pd.concat(site_frames, ignore_index=True) if site_frames else pd.DataFrame()
    hours_df = pd.concat(hour_frames, ignore_index=True) if hour_frames else pd.DataFrame()
    return sites_df, hours_df


def _tag(comp: pd.DataFrame, w: pd.Series, cohort: str, plan: str, state: str,
         year: str) -> pd.DataFrame:
    out = comp.copy()
    out.index.name = 'site_id'
    out = out.reset_index()
    out.insert(0, 'weight', w.reindex(comp.index).to_numpy())
    out.insert(0, 'year', year)
    out.insert(0, 'state', state)
    out.insert(0, 'plan', plan)
    out.insert(0, 'cohort', cohort)
    return out


# ── Tiers ─────────────────────────────────────────────────────────────────────

VALUE_COLS = list(COMPONENT_COLS) + ['margin', 'imports_kwh']
HOUR_COLS = ['import_kwh', 'import_revenue', 'network_import', 'wholesale_import',
             'env_market', 'import_margin']


def state_year_table(sites: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in sites.groupby(['cohort', 'plan', 'state', 'year'], sort=False):
        row = dict(zip(['cohort', 'plan', 'state', 'year'], key))
        row['n_sites'] = len(g)
        for c in VALUE_COLS:
            row[c] = float(np.dot(g['weight'], g[c]))
        rows.append(row)
    return pd.DataFrame(rows)


def state_tier(state_years: pd.DataFrame) -> pd.DataFrame:
    """Per state, the mean over its years of the state-year means."""
    g = state_years.groupby(['cohort', 'plan', 'state'], sort=False)
    out = g[VALUE_COLS].mean().reset_index()
    out['years'] = g['year'].agg(lambda y: _years_label(list(y))).to_numpy()
    return out


def spread_stats(sites: pd.DataFrame, subscription: float,
                 state_mix: dict[str, float] | None) -> pd.DataFrame:
    """Weighted per-site margin distribution per (cohort, plan). Each year of
    a state carries equal weight; with state_mix, states carry their mix."""
    d = sites.copy()
    n_years = d.groupby('state')['year'].transform('nunique')
    d['w'] = d['weight'] / n_years
    if state_mix is not None:
        present = d['state'].unique()
        tot = sum(state_mix.get(s, 0.0) for s in present)
        d['w'] = d['w'] * d['state'].map(lambda s: state_mix.get(s, 0.0) / tot)
    rows = []
    for (cohort, plan), g in d.groupby(['cohort', 'plan'], sort=False):
        q = weighted_quantiles(g['margin'].to_numpy(), g['w'].to_numpy(), QUANTILES)
        row = {'cohort': cohort, 'plan': plan,
               'share_below_subscription': (
                   np.nan if plan == 'Amber'
                   else float(g.loc[g['margin'] < subscription, 'w'].sum() / g['w'].sum()))}
        row.update({f'p{int(round(100 * x))}': v for x, v in zip(QUANTILES, q)})
        rows.append(row)
    return pd.DataFrame(rows)


def hour_state_tier(hours: pd.DataFrame) -> pd.DataFrame:
    return (hours.groupby(['cohort', 'plan', 'state', 'hour'], sort=False)[HOUR_COLS]
            .mean().reset_index())


def amber_subscription(summary: pd.DataFrame) -> float:
    s = summary.loc[(summary['plan'] == 'Amber')
                    & summary['cohort'].str.endswith('(coordinated)'), 'margin']
    if s.empty:
        raise CheckError('no Amber coordinated rows')
    return float(s.iloc[0])


# ── Charts ────────────────────────────────────────────────────────────────────
# Drawn from the saved CSVs in thesis format (docs/thesis_figure_spec.md), so
# they can be restyled without re-running the model. Retailers take the colours
# and names of flow_exposure_analysis; data notes are drawn only with --notes.

COHORT_ORDER = [v[0] for v in COHORTS.values()] + [OPTIMISED_LABEL]
BATTERY_COHORTS = {'Self Consumption', OPTIMISED_LABEL}
PLAN_ORDER = list(PLANS)
# Retailer -> its label in the flow_exposure_analysis registry.
REGISTRY_LABEL = {'AGL': 'AGL', 'Energy Australia': 'Energy Australia',
                  'Origin': 'Origin Battery Lite', 'Amber': 'Amber'}
SUBSCRIPTION_LABEL = 'Amber subscription'
GROUP_GAP = 0.375               # between groups of bars, in bar spacings
BAR_WIDTH = 0.72
HEIGHT_IN = 3.0
GRID_HEIGHT_IN = 2.5            # panels of a grid
COMPONENT_COLOURS = dict(zip(COMPONENT_COLS, ts.CATEGORICAL_COLOURS))


def _colour(plan: str) -> str:
    return {p['label']: p['color'] for p in fe.COORDINATION_PLANS}[REGISTRY_LABEL[plan]]


def _plan_name(plan: str) -> str:
    return fe.OFFER_NAMES.get(REGISTRY_LABEL[plan], plan)


def _cohort_title(cohort: str, battery_kwh: int) -> str:
    # Self-consumption has no retailer: the plan sets only the revenue side.
    return _name_of(cohort).replace(' (Amber)', '') + (
        f', {battery_kwh} kWh' if cohort in BATTERY_COHORTS else '')


def _name_of(cohort: str) -> str:
    return fe._display_name({'No CER': 'No CER (load only)',
                             'Self Consumption': 'Self Consumption (Amber)',
                             OPTIMISED_LABEL: 'TOU Optimised - no VPP'}.get(cohort, cohort))


def _row(summary: pd.DataFrame, cohort: str, plan: str) -> pd.Series | None:
    r = summary[(summary['cohort'] == cohort) & (summary['plan'] == plan)]
    return None if r.empty else r.iloc[0]


def _dollars(v: float) -> str:
    return ('−' if v < 0 else '') + f'\\${abs(v):,.0f}'


def _dollar_axis(ax) -> None:
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: _dollars(v)))


def _positions() -> tuple[list, float, list]:
    """(cohort, plan, x) for each retailer bar, the x of the Amber bar and the
    x of the separator after each group."""
    pos, seps, x = [], [], 0.0
    for c in COHORT_ORDER:
        for p in PLAN_ORDER:
            pos.append((c, p, x))
            x += 1.0
        seps.append(x - 0.5 + GROUP_GAP / 2)
        x += GROUP_GAP
    return pos, x, seps


def _group_axis(ax, pos, amber_x, battery_kwh) -> None:
    centres = [np.mean([x for c2, _, x in pos if c2 == c]) for c in COHORT_ORDER]
    # One line each, so a label is not read as belonging to the middle bar.
    labels = [_name_of(c).replace(' (Amber)', '')
              + (f'\n({battery_kwh} kWh)' if c in BATTERY_COHORTS else '')
              for c in COHORT_ORDER]
    ax.set_xticks(centres + [amber_x], labels + [f'Amber\n({battery_kwh} kWh)'])
    ax.tick_params(axis='x', length=0)
    ax.grid(axis='x', visible=False)


def _separators(ax, seps) -> None:
    for x in seps:
        ax.axvline(x, color=ts.SEPARATOR_COLOUR, ls=':', lw=ts.SEPARATOR_LW, zorder=1)


def _subscription_line(ax, sub: float) -> None:
    ax.axhline(sub, color=ts.REFERENCE_COLOUR, ls='--', lw=ts.REFERENCE_LW, zorder=1)


def _retailer_handles(sub: float, extra=()) -> tuple[list, list]:
    handles = [mpatches.Patch(color=_colour(p), linewidth=0) for p in PLAN_ORDER]
    handles.append(mpatches.Patch(color=_colour('Amber'), linewidth=0))
    handles.append(Line2D([], [], color=ts.REFERENCE_COLOUR, ls='--', lw=ts.REFERENCE_LW))
    labels = [_plan_name(p) for p in PLAN_ORDER] + ['Amber', f'{SUBSCRIPTION_LABEL} '
                                                    f'{_dollars(sub)}']
    return handles + [h for h, _ in extra], labels + [t for _, t in extra]


def _bottom(summary: pd.DataFrame, columns) -> float:
    return min(0.0, float(summary[list(columns)].min().min()))


def fig_bars(summary, sub, title, note, battery_kwh, out: Path) -> None:
    fig = ts.new_figure(HEIGHT_IN)
    ax = fig.subplots()
    pos, amber_x, seps = _positions()
    values = [sub]
    for c, p, x in pos:
        r = _row(summary, c, p)
        if r is None:
            continue
        ax.bar(x, r['margin'], width=BAR_WIDTH, color=_colour(p), linewidth=0, zorder=2)
        fe._signed_label(ax, x, float(r['margin']), _dollars(r['margin']))
        values.append(float(r['margin']))
    ax.bar(amber_x, sub, width=BAR_WIDTH, color=_colour('Amber'), linewidth=0, zorder=2)
    fe._signed_label(ax, amber_x, sub, _dollars(sub))
    ax.axhline(0, color='black', lw=0.6, zorder=3)
    _separators(ax, seps)
    _subscription_line(ax, sub)
    _group_axis(ax, pos, amber_x, battery_kwh)
    _dollar_axis(ax)
    ax.set_xlim(-0.6, amber_x + 0.6)
    lo, hi = min(0.0, min(values)), max(values)
    ax.set_ylim(lo * 1.25, hi * (1.55 if note else 1.12))
    ax.set_ylabel('Retailer gross margin ex GST (\\$/site/yr)')
    ax.set_title(title)
    handles, labels = _retailer_handles(sub)
    fig.legend(handles, labels, loc='outside lower center', ncol=5, frameon=False)
    if note:
        fa._draw_notes(ax, note, 'upper left')
    ts.save_figure(fig, out / 'margin_bars')


def fig_components(summary, sub, title, battery_kwh, out: Path) -> None:
    fig = ts.new_figure(GRID_HEIGHT_IN + 0.5)
    axes = ts.grid_axes(fig, 1, len(COHORT_ORDER), '')
    used = set()
    for ax, c in zip(axes[0], COHORT_ORDER):
        for i, p in enumerate(PLAN_ORDER):
            r = _row(summary, c, p)
            if r is None:
                continue
            up = down = 0.0
            for k in COMPONENT_COLS:
                v = float(r[k])
                if abs(v) < 1e-9:
                    continue
                ax.bar(i, v, bottom=up if v > 0 else down, width=BAR_WIDTH,
                       color=COMPONENT_COLOURS[k], linewidth=0, zorder=2)
                used.add(k)
                if v > 0:
                    up += v
                else:
                    down += v
            ax.plot(i, r['margin'], 'D', color='black', ms=ts.GRID_MARKS.ms,
                    zorder=3)
            fe._signed_label(ax, i, up, _dollars(r['margin']))
        _subscription_line(ax, sub)
        ax.axhline(0, color='black', lw=ts.GRID_SPINE_LW, zorder=3)
        ax.set_xticks(range(len(PLAN_ORDER)), [fe._tick_label(_plan_name(p))
                                                for p in PLAN_ORDER])
        ax.tick_params(axis='x', length=0)
        ax.grid(axis='x', visible=False)
        ax.set_title(_cohort_title(c, battery_kwh), fontsize=ts.SUBTITLE_SIZE)
        _dollar_axis(ax)
    fig.supylabel('Retailer margin ex GST (\\$/site/yr)',
                  fontsize=mpl.rcParams['axes.labelsize'])
    handles = [mpatches.Patch(color=COMPONENT_COLOURS[k], linewidth=0)
               for k in COMPONENT_COLS if k in used]
    labels = [COMPONENT_LABELS[k] for k in COMPONENT_COLS if k in used]
    handles += [Line2D([], [], marker='D', color='black', ls='none',
                       ms=ts.GRID_MARKS.ms),
                Line2D([], [], color=ts.REFERENCE_COLOUR, ls='--', lw=ts.REFERENCE_LW)]
    labels += ['Margin', f'{SUBSCRIPTION_LABEL} {_dollars(sub)}']
    fig.legend(handles, labels, loc='outside lower center', ncol=5, frameon=False)
    ts.panel_suptitle(fig, title)
    ts.save_figure(fig, out / 'margin_components')


def fig_spread(summary, sub, title, note, battery_kwh, out: Path) -> None:
    fig = ts.new_figure(HEIGHT_IN)
    ax = fig.subplots()
    pos, amber_x, seps = _positions()
    half = BAR_WIDTH / 2
    for c, p, x in pos:
        r = _row(summary, c, p)
        if r is None:
            continue
        col = _colour(p)
        ax.bar(x, r['p75'] - r['p25'], bottom=r['p25'], width=BAR_WIDTH, color=col,
               alpha=ts.BAND_ALPHA, linewidth=0, zorder=2)
        for a, b in (('p10', 'p25'), ('p75', 'p90')):
            ax.plot([x, x], [r[a], r[b]], color=col, lw=ts.MEMBER_LW, zorder=2)
        ax.plot([x - half, x + half], [r['p50']] * 2, color=col, lw=ts.EMPHASIS_LW,
                zorder=3)
        ax.plot(x, r['margin'], 'o', color='black', ms=ts.MARKER_SIZE, zorder=4)
        ax.text(x, 0.98, f"{100 * r['share_below_subscription']:.0f}%",
                transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontsize=ts.ANNOTATION_SIZE)
    ax.plot(amber_x, sub, 'o', color=_colour('Amber'), ms=ts.MARKER_SIZE + 1,
            mec=ts.REFERENCE_COLOUR, mew=ts.MARKER_EDGE_WIDTH, zorder=4)
    ax.axhline(0, color='black', lw=0.6, zorder=3)
    _separators(ax, seps)
    _subscription_line(ax, sub)
    _group_axis(ax, pos, amber_x, battery_kwh)
    ax.set_xlim(-0.6, amber_x + 0.6)
    _dollar_axis(ax)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (0.45 if note else 0.17) * (hi - lo))
    ax.set_ylabel('Retailer gross margin ex GST (\\$/site/yr)')
    ax.set_title(title)
    # Key to the percentages along the top: the share of each group's sites
    # with a gross margin below the Amber subscription line. Centred on the
    # separator between Solar only and Self-consumption, one line below them.
    ax.annotate('% of sites below the Amber subscription',
                xy=(seps[1], 0.98), xycoords=ax.get_xaxis_transform(),
                xytext=(0, -1.6 * ts.ANNOTATION_SIZE), textcoords='offset points',
                ha='center', va='top', fontsize=ts.ANNOTATION_SIZE,
                bbox=ts.LABEL_BOX, zorder=6)
    handles, labels = _retailer_handles(sub)
    fig.legend(handles, labels, loc='outside lower center', ncol=5, frameon=False)
    if note:
        fa._draw_notes(ax, note, 'upper left')
    ts.save_figure(fig, out / 'margin_spread')


def fig_by_year(by_year, title, battery_kwh, out: Path) -> None:
    fig = ts.new_figure(GRID_HEIGHT_IN + 0.5)
    axes = ts.grid_axes(fig, 1, len(COHORT_ORDER), 'Year')
    amber = by_year[(by_year['plan'] == 'Amber')
                    & by_year['cohort'].str.endswith('(coordinated)')].sort_values('year')
    marks = ts.GRID_MARKS
    years = sorted(by_year['year'].unique())
    for ax, c in zip(axes[0], COHORT_ORDER):
        for p in PLAN_ORDER:
            d = by_year[(by_year['cohort'] == c) & (by_year['plan'] == p)].sort_values('year')
            if d.empty:
                continue
            ax.plot(d['year'], d['margin'], '-o', color=_colour(p), lw=marks.lw,
                    ms=marks.ms, mew=marks.mew, mec='white',
                    zorder=3)
        if not amber.empty:
            ax.plot(amber['year'], amber['margin'], '--', color=ts.REFERENCE_COLOUR,
                    lw=ts.REFERENCE_LW, zorder=2)
        ax.axhline(0, color=ts.SEPARATOR_COLOUR, lw=0.5, zorder=1)
        ax.set_title(_cohort_title(c, battery_kwh), fontsize=ts.SUBTITLE_SIZE)
        _dollar_axis(ax)
    axes[0][0].set_xticks(years, [str(y) for y in years])
    fig.supylabel('Retailer margin ex GST (\\$/site/yr)',
                  fontsize=mpl.rcParams['axes.labelsize'])
    handles = [Line2D([], [], color=_colour(p), lw=marks.lw, marker='o',
                      ms=marks.ms, mec='white', mew=marks.mew)
               for p in PLAN_ORDER]
    handles.append(Line2D([], [], color=ts.REFERENCE_COLOUR, ls='--', lw=ts.REFERENCE_LW))
    fig.legend(handles, [_plan_name(p) for p in PLAN_ORDER] + [SUBSCRIPTION_LABEL],
               loc='outside lower center', ncol=4, frameon=False)
    ts.panel_suptitle(fig, title)
    ts.save_figure(fig, out / 'margin_by_year')


def fig_by_hour(hours, title, battery_kwh, out: Path) -> None:
    fig = ts.new_figure(GRID_HEIGHT_IN + 0.5)
    axes = ts.grid_axes(fig, 1, len(COHORT_ORDER), 'Time of day')
    for ax, c in zip(axes[0], COHORT_ORDER):
        for p in PLAN_ORDER:
            d = hours[(hours['cohort'] == c) & (hours['plan'] == p)].sort_values('hour')
            if d.empty:
                continue
            y = d['import_margin'].to_numpy()
            ax.step(np.append(d['hour'].to_numpy(), 24), np.append(y, y[-1]), where='post',
                    color=_colour(p), lw=ts.GRID_MARKS.lw, zorder=3)
        ax.axhline(0, color=ts.SEPARATOR_COLOUR, lw=0.5, zorder=1)
        ax.set_title(_cohort_title(c, battery_kwh), fontsize=ts.SUBTITLE_SIZE)
        _dollar_axis(ax)
    axes[0][0].set_xlim(0, 24)
    axes[0][0].set_xticks(range(0, 25, 6), [f'{h:02d}:00' for h in range(0, 25, 6)])
    fig.supylabel('Import margin ex GST (\\$/site/yr in the hour)',
                  fontsize=mpl.rcParams['axes.labelsize'])
    fig.legend([Line2D([], [], color=_colour(p), lw=ts.GRID_MARKS.lw)
                for p in PLAN_ORDER], [_plan_name(p) for p in PLAN_ORDER],
               loc='outside lower center', ncol=3, frameon=False)
    ts.panel_suptitle(fig, title)
    ts.save_figure(fig, out / 'margin_by_hour')


def draw_charts(out: Path, battery_kwh: int, weighted: bool, plan_set: str,
                notes: bool) -> None:
    """Every chart, from the CSVs under out."""
    mpl.rcdefaults()             # financial_analysis sets its own on import
    ts.apply_style()
    state_years = out / 'state_year' / 'margin_state_years.csv'
    all_years = (_years_label([str(y) for y in pd.read_csv(state_years)['year'].unique()])
                 if state_years.exists() else '')
    tiers = sorted(d for d in out.glob('*_all_years') if d.is_dir()) + [out / 'national']
    for d in tiers:
        if not (d / 'margin_summary.csv').exists():
            continue
        national = d.name == 'national'
        name = 'National' if national else d.name.split('_')[0]
        summary = pd.read_csv(d / 'margin_summary.csv')
        years = str(summary['years'].iloc[0]) if 'years' in summary else all_years
        sub = amber_subscription(summary)
        where = f'{name}, {years}'
        note = None
        if notes:
            note = textwrap.fill(
                ('Plans: default no-CER and solar plans for No CER and Solar Only, battery '
                 'plans (TOU only) for Self-consumption. ' if plan_set == 'standard'
                 else 'Plans: battery plans (TOU only) for No CER, Solar only and '
                      'Self-consumption. ')
                + 'Optimised: VPP plans under each retailer\'s rolling optimisation. '
                + ('Weighted by PV size' + (' and state mix' if national else '')
                   if weighted else 'Unweighted')
                + ". 2026-27 retail and network tariffs applied to each year's prices. "
                  "GreenPower excluded; environmental + market costs at Amber's adder "
                  'rates; hedging cost 0.', width=58)
        fig_bars(summary, sub, f'Retailer gross margin after supply costs, {where}{BASIS_SUFFIX}', note,
                 battery_kwh, d)
        fig_components(summary, sub, f'Retailer margin components, {where}{BASIS_SUFFIX}', battery_kwh, d)
        spread_note = note and (note + '\nBox p25\u2013p75, whiskers p10\u2013p90, line '
                                'median, dot mean;\n% = share of sites below the Amber '
                                'subscription.')
        fig_spread(summary, sub, f'Retailer gross margin across sites, {where}{BASIS_SUFFIX}', spread_note,
                   battery_kwh, d)
        if (d / 'margin_by_hour.csv').exists():
            fig_by_hour(pd.read_csv(d / 'margin_by_hour.csv'),
                        f'Import margin by hour of day, {where}{BASIS_SUFFIX}', battery_kwh, d)
    if (out / 'national' / 'margin_by_year.csv').exists():
        fig_by_year(pd.read_csv(out / 'national' / 'margin_by_year.csv'),
                    f'Retailer margin after supply costs by year, National{BASIS_SUFFIX}', battery_kwh,
                    out / 'national')


# ── Main ──────────────────────────────────────────────────────────────────────

def tier_outputs(name, yl, summary, spread, hours, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    sub = amber_subscription(summary)
    s = summary.merge(spread, on=['cohort', 'plan'], how='left')
    s['margin_minus_subscription'] = s['margin'] - sub
    s.to_csv(out / 'margin_summary.csv', index=False)
    hours.to_csv(out / 'margin_by_hour.csv', index=False)
    cols = ['cohort', 'plan'] + list(COMPONENT_COLS) + ['margin', 'share_below_subscription']
    print(f"\n{name} ({yl}), $/site/yr ex GST:")
    print(s[cols].round(2).to_string(index=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', type=Path, default=Path('.'),
                        help='Project root holding data/output/runs, data/clean/nem_prices '
                             'and data/processed/edp (default: cwd)')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Output directory (default: analysis/output/retailer_outcomes, '
                             'or retailer_outcomes_standard with --plan-set standard)')
    parser.add_argument('--plan-set', choices=('battery', 'standard'), default='battery',
                        help="'battery': each retailer's battery plan for every cohort "
                             "(default). 'standard': its default no-CER and solar plans "
                             "for No CER and Solar Only (STANDARD_RUNS).")
    parser.add_argument('--states', nargs='+', default=None,
                        help='Restrict to specific states (default: all found)')
    parser.add_argument('--years', nargs='+', default=None,
                        help='Restrict to specific years (default: all found)')
    parser.add_argument('--battery-kwh', type=int, default=20,
                        help='Battery size for battery scenarios (default: 20)')
    parser.add_argument('--unweighted', action='store_true',
                        help='Straight per-site means instead of size-band weighting')
    parser.add_argument('--weights-file', type=Path,
                        default=Path('analysis/config/pv_size_bands.yaml'),
                        help='Weighting YAML (default: analysis/config/pv_size_bands.yaml)')
    parser.add_argument('--charts-only', action='store_true',
                        help='Redraw the charts from the saved CSVs in the output directory')
    parser.add_argument('--notes', action='store_true',
                        help='Draw the data notes on the bar and spread charts '
                             '(presentation versions; the thesis versions have none)')
    args = parser.parse_args(argv)

    if args.plan_set == 'standard':
        for plan, runs in STANDARD_RUNS.items():
            PLANS[plan]['runs'].update(runs)
    if args.output_dir is None:
        args.output_dir = Path('analysis/output/retailer_outcomes'
                               + ('_standard' if args.plan_set == 'standard' else ''))
    weighted = not args.unweighted
    if args.charts_only:
        draw_charts(args.output_dir, args.battery_kwh, weighted, args.plan_set, args.notes)
        return 0
    weights_cfg = load_weights_config(args.weights_file) if weighted else None
    state_mix = weights_cfg['state_mix'] if weighted else None

    scenarios = discover_scenarios(args.data_root)
    if args.states:
        scenarios = [(s, y) for s, y in scenarios if s in args.states]
    if args.years:
        scenarios = [(s, y) for s, y in scenarios if y in args.years]
    if not scenarios:
        print("ERROR: no scenarios found or none match the filter.", file=sys.stderr)
        return 1
    print(f"Scenarios found: {scenarios}")

    try:
        sites, hours = load_all(args.data_root, scenarios, weighted, weights_cfg,
                                args.battery_kwh)
    except CheckError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    if sites.empty:
        print("ERROR: no bills found.", file=sys.stderr)
        return 1

    out = args.output_dir
    sy = state_year_table(sites)
    (out / 'state_year').mkdir(parents=True, exist_ok=True)
    sy.to_csv(out / 'state_year' / 'margin_state_years.csv', index=False)

    st = state_tier(sy)
    hst = hour_state_tier(hours)
    for state in st['state'].unique():
        s = st[st['state'] == state]
        sub = amber_subscription(s)
        spread = spread_stats(sites[sites['state'] == state], sub, None)
        tier_outputs(state, s['years'].iloc[0], s.drop(columns=['state']), spread,
                     hst[hst['state'] == state].drop(columns=['state']),
                     out / f'{state}_all_years')

    if st['state'].nunique() > 1 and state_mix is not None:
        nat = national_mix(st, ['cohort', 'plan'], VALUE_COLS, state_mix)
        yl = _years_label(list(sy['year']))
        sub = amber_subscription(nat)
        spread = spread_stats(sites, sub, state_mix)
        hnat = national_mix(hst, ['cohort', 'plan', 'hour'], HOUR_COLS, state_mix)
        tier_outputs('National', yl, nat.drop(columns=['states']), spread,
                     hnat.drop(columns=['states']), out / 'national')
        by_year = national_mix(sy, ['cohort', 'plan', 'year'], VALUE_COLS, state_mix)
        by_year.to_csv(out / 'national' / 'margin_by_year.csv', index=False)
    draw_charts(out, args.battery_kwh, weighted, args.plan_set, args.notes)
    return 0


if __name__ == '__main__':
    sys.exit(main())
