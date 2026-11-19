"""
import_price_analysis.py

Import-weighted average wholesale price (IWAP) of a solar/battery cohort's
grid imports, per scenario. See docs/import_price_spec.md.

  Chart 1  {STATE}_all_years/, national/  iwap_columns.png
                                      IWAP per scenario, labelled with the
                                      saving against No CER and import volume
  Chart 2  state_months/iwap_box.png  state-month IWAP distribution
  Chart 3  state_months/iwap_saving_box.png
                                      state-month saving against No CER
  Chart 4  state_year/iwap_by_state_and_year.png
                                      IWAP per scenario, one panel per
                                      state and year
  Chart 5  {STATE}_all_years/, national/  iwap_by_period.png
                                      IWAP per scenario within each network
                                      TOU period, with import share
  Chart 6  {STATE}_all_years/, national/  iwap_saving_split.png
                                      saving per pair, split into between-
                                      and within-period terms
  state_months/state_months.csv       per scenario, state and month
  {STATE}_all_years/, national/ iwap_summary.csv, iwap_by_period.csv,
                                saving_split.csv
  state_year/state_years.csv          per scenario, state and year

Prices carry the network tariff's import loss factor (usage_dlf · import_tlf).
TOU periods come from each state's network tariff: highest rate = Peak,
lowest rate = Off-peak (restricted to the export-penalty window where the
lowest rate covers it), everything else = Shoulder.

Sign convention: saving_vs_no_cer = IWAP(No CER) − IWAP(scenario), $/MWh;
positive = cheaper import.

Charts follow docs/thesis_figure_spec.md and are drawn from the CSVs, so
--charts-only redraws them without the flows. --no-notes leaves the data
notes off, for the thesis.

Run from repo root:
  python3 analysis/scripts/import_price_analysis.py --data-root "/Volumes/Manual/Thesis Project" --core-only
  python3 analysis/scripts/import_price_analysis.py --charts-only --no-notes
"""
import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import yaml
from matplotlib.legend_handler import HandlerTuple

import financial_analysis as fa
import flow_exposure_analysis as fe
import savings_bar_figure as sbf
import thesis_style as ts
from cap_hedge_analysis import NO_CER, _years_label, build_scenarios
from financial_analysis import load_weights_config
from flow_exposure_analysis import (
    INTERVAL_HOURS, RRP_COL, compute_site_weight_for_scenario,
    discover_scenarios, get_prices, load_flows,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from model.core.localtime import local_wall_clock  # noqa: E402
from model.core.tariff import load_network_tariff  # noqa: E402

SELF_CONSUMPTION = 'Self Consumption'
PERIODS = ('Peak', 'Shoulder', 'Off-peak')
PERIOD_KEY = {'Peak': 'peak', 'Shoulder': 'shoulder', 'Off-peak': 'offpeak'}
# Battery scenarios use the full_run_rolling_* runs (cap_hedge_analysis.build_scenarios),
# the achievable basis of financial_analysis.RUN_BASES.
BASIS_SUFFIX = ', imperfect foresight (achievable)'


# ── TOU periods ───────────────────────────────────────────────────────────────

def network_tariff_ids(scen_list: list[dict]) -> dict[str, dict[int, str]]:
    """network_tariff_ids from each scenario's model config; all scenarios
    must agree."""
    ids = None
    for sc in scen_list:
        path = REPO_ROOT / 'model' / 'configs' / f"{sc['run_dir']}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"No model config for '{sc['label']}': {path}")
        cfg = yaml.safe_load(path.read_text())['network_tariff_ids']
        cfg = {st: {int(y): t for y, t in by_year.items()} for st, by_year in cfg.items()}
        if ids is None:
            ids = cfg
        elif cfg != ids:
            raise ValueError(f"network_tariff_ids in {path.name} differ from the other "
                             "scenarios' configs; one network tariff per state-year is needed.")
    return ids


def hour_periods(tariff) -> np.ndarray:
    """Period label for each local hour 0-23 from the network import rates:
    highest rate = Peak, lowest = Off-peak, else Shoulder. Where the lowest
    rate covers the export-penalty window (negative feed-in rate), only that
    window is Off-peak and the rest of the lowest-rate hours are Shoulder."""
    if tariff.usage_tariffs_weekend is not None:
        raise ValueError(f"{tariff.name}: weekend usage rates are not supported "
                         "by the period split.")
    rate = np.full(24, np.nan)
    for tp in tariff.usage_tariffs:
        rate[tp.start_hour:tp.end_hour] = tp.c_per_kwh
    penalty = np.zeros(24, dtype=bool)
    for tp in tariff.feedin_tariffs:
        if tp.c_per_kwh < 0:
            penalty[tp.start_hour:tp.end_hour] = True
    hi, lo = np.nanmax(rate), np.nanmin(rate)
    low = np.isclose(rate, lo)
    out = np.where(np.isclose(rate, hi), 'Peak', 'Shoulder').astype(object)
    if penalty.any() and np.all(low[penalty]):
        out[penalty] = 'Off-peak'
    else:
        out[low] = 'Off-peak'
    return out


def _spans(labels: np.ndarray, period: str) -> str:
    hrs = [h for h in range(24) if labels[h] == period]
    if not hrs:
        return '—'
    spans, start = [], hrs[0]
    for a, b in zip(hrs, hrs[1:] + [None]):
        if b != a + 1:
            spans.append((start, a + 1))
            start = b
    if len(spans) > 1 and spans[0][0] == 0 and spans[-1][1] == 24:
        spans = [(spans[-1][0], spans[0][1])] + spans[1:-1]
    return ', '.join(f'{s}–{e}' for s, e in spans)


# ── Loading ───────────────────────────────────────────────────────────────────

def cohort_import(df: pd.DataFrame, site_weight: pd.Series | None) -> pd.Series:
    """Per-interval weighted mean gross import per site (kWh/site). Gross
    import is each site's net flow clipped at zero, so one site's export is
    not netted against another's import. Weights are renormalised over the
    sites present in this flows file."""
    imp = df.clip(lower=0)
    if site_weight is None:
        return imp.mean(axis=1)
    cols = [c for c in imp.columns if c in site_weight.index]
    if not cols:
        return pd.Series(np.nan, index=imp.index)
    w = site_weight.loc[cols].to_numpy()
    return pd.Series(imp[cols].to_numpy() @ w / w.sum(), index=imp.index)


def load_all(data_root: Path, scenarios: list[tuple[str, str]], scen_list: list[dict],
             weighted: bool, weights_cfg: dict | None, battery_kwh: int) -> pd.DataFrame:
    """One pass over every (scenario, state, year) flows file. Returns one
    row per (scenario, state, year, month, period): import_kwh (per site),
    cost ($/site, loss factor applied), n_dropped."""
    runs_root = data_root / 'data' / 'output' / 'runs'
    price_cache: dict[str, pd.DataFrame] = {}
    sw_cache: dict[tuple[str, str], pd.Series] = {}
    dc_ac_cache: dict[str, float] = {}
    meta_cache: dict[str, pd.DataFrame] = {}
    frames, rows = [], []

    ids = network_tariff_ids(scen_list)
    tariff_dir = REPO_ROOT / 'model' / 'tariffs'
    tariffs: dict[str, object] = {}
    print("\nTOU periods (local time) and import loss factors:")
    for state, year in scenarios:
        tid = ids[state][int(year)]
        if tid not in tariffs:
            t = load_network_tariff(tid, tariff_dir)
            tariffs[tid] = (t, hour_periods(t))
            hp = tariffs[tid][1]
            print(f"  {state} {tid}: " + '; '.join(f'{p} {_spans(hp, p)}' for p in PERIODS)
                  + f"; LF {t.import_loss_factor:.4f}")

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

            tariff, hp = tariffs[ids[state][int(year)]]
            lf = tariff.import_loss_factor
            price = get_prices(data_root, year, price_cache)[RRP_COL[state]]
            imp = cohort_import(df, site_weight)
            p = price.reindex(imp.index)
            valid = imp.notna() & p.notna()
            start = imp.index - pd.Timedelta(hours=INTERVAL_HOURS)
            idx = imp.index if imp.index.tz is not None else imp.index.tz_localize('Etc/GMT-10')
            period = hp[local_wall_clock(idx, state, INTERVAL_HOURS).hour]
            key = [start.year, start.month, period]
            g = pd.DataFrame({
                'import_kwh': imp.where(valid, 0.0),
                'cost': (imp * lf * p / 1000.0).where(valid, 0.0),
                'n_dropped': (~valid).astype(int),
            }).groupby(key).sum()
            g.index.names = ['year', 'month', 'period']
            g = g.reset_index()
            g.insert(0, 'state', state)
            g.insert(0, 'scenario', sc['label'])
            frames.append(g)
            rows.append((sc['label'], f'{state}_{year}', len(df.columns), n_used,
                         int((~valid).sum()), g['import_kwh'].sum() / 1000.0,
                         _iwap(g['cost'].sum(), g['import_kwh'].sum())))

    print(f"\n{'Scenario':<20} {'State_year':<10} {'Sites':>5} {'Used':>5} "
          f"{'Dropped':>7} {'MWh/site':>8} {'IWAP':>8}")
    for r in rows:
        print(f"{r[0]:<20} {r[1]:<10} {r[2]:>5} {r[3]:>5} {r[4]:>7} {r[5]:>8.2f} {r[6]:>8.2f}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _iwap(cost: float, import_kwh: float) -> float:
    """$/MWh from $/site and kWh/site."""
    return 1000.0 * cost / import_kwh if import_kwh > 0 else np.nan


# ── Aggregation ───────────────────────────────────────────────────────────────

def state_month_table(months: pd.DataFrame) -> pd.DataFrame:
    t = months.copy()
    t['import_mwh_per_site'] = t['import_kwh'] / 1000.0
    t['import_cost_per_site'] = t['cost']
    t['iwap'] = [_iwap(c, i) for c, i in zip(t['cost'], t['import_kwh'])]
    ref = t[t['scenario'] == NO_CER].set_index(['state', 'year', 'month'])['iwap']
    key = pd.MultiIndex.from_frame(t[['state', 'year', 'month']])
    t['saving_vs_no_cer'] = ref.reindex(key).to_numpy() - t['iwap'].to_numpy()
    return t[['scenario', 'state', 'year', 'month', 'import_mwh_per_site',
              'import_cost_per_site', 'iwap', 'saving_vs_no_cer', 'n_dropped']]


def summarise(months: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Pool state-month rows by `by` (must include 'scenario'): IWAP as a
    ratio of sums, import and cost per site per year, saving against No CER."""
    g = months.groupby(by, sort=False).agg(
        import_kwh=('import_kwh', 'sum'), cost=('cost', 'sum'),
        n_years=('year', 'nunique'),
        years=('year', lambda y: _years_label([str(v) for v in y]))).reset_index()
    g['iwap'] = [_iwap(c, i) for c, i in zip(g['cost'], g['import_kwh'])]
    g['import_mwh_per_site_yr'] = g['import_kwh'] / 1000.0 / g['n_years']
    g['import_cost_per_site_yr'] = g['cost'] / g['n_years']
    rest = [b for b in by if b != 'scenario']
    if rest:
        ref = g[g['scenario'] == NO_CER].set_index(rest)['iwap']
        g['saving_vs_no_cer'] = (ref.reindex(pd.MultiIndex.from_frame(g[rest]) if len(rest) > 1
                                             else g[rest[0]]).to_numpy() - g['iwap'].to_numpy())
    else:
        ref = g.loc[g['scenario'] == NO_CER, 'iwap']
        g['saving_vs_no_cer'] = (ref.iloc[0] if len(ref) else np.nan) - g['iwap']
    return g[by + ['years', 'import_mwh_per_site_yr', 'import_cost_per_site_yr',
                   'iwap', 'saving_vs_no_cer']]


def national_summary(state_sums: dict[str, pd.DataFrame], state_mix: dict[str, float],
                     order: list[str]) -> pd.DataFrame:
    """state_mix-weighted mean of the per-state summaries, renormalised per
    scenario over the states present."""
    cols = ['import_mwh_per_site_yr', 'import_cost_per_site_yr', 'iwap', 'saving_vs_no_cer']
    rows = []
    for lab in order:
        present = [(s, d.set_index('scenario').loc[lab]) for s, d in state_sums.items()
                   if lab in set(d['scenario']) and state_mix.get(s, 0) > 0]
        if not present:
            continue
        w = np.array([state_mix[s] for s, _ in present])
        w = w / w.sum()
        row = {'scenario': lab,
               'years': _years_label([y for _, r in present for y in str(r['years']).split('–')])}
        for c in cols:
            row[c] = float(np.sum(w * np.array([r[c] for _, r in present])))
        rows.append(row)
    return pd.DataFrame(rows)


def period_summary(detail: pd.DataFrame) -> pd.DataFrame:
    """One row per (scenario, period) present in `detail`: import_share of the
    scenario's import and IWAP within the period."""
    g = detail.groupby(['scenario', 'period'], sort=False)[['import_kwh', 'cost']].sum()
    tot = g['import_kwh'].groupby(level=0).transform('sum')
    out = pd.DataFrame({'import_share': g['import_kwh'] / tot,
                        'iwap': [_iwap(c, i) for c, i in zip(g['cost'], g['import_kwh'])]},
                       index=g.index).reset_index()
    return _full_periods(out)


def _full_periods(ps: pd.DataFrame) -> pd.DataFrame:
    """Reindex to every (scenario, period) so a period with no import shows
    share 0 and IWAP NaN."""
    idx = pd.MultiIndex.from_product([list(dict.fromkeys(ps['scenario'])), PERIODS],
                                     names=['scenario', 'period'])
    out = ps.set_index(['scenario', 'period']).reindex(idx)
    out['import_share'] = out['import_share'].fillna(0.0)
    return out.reset_index()


def national_periods(state_ps: dict[str, pd.DataFrame], state_mix: dict[str, float]
                     ) -> pd.DataFrame:
    """state_mix-weighted mean of the per-state period shares and IWAPs,
    renormalised over the states with a value."""
    frames = []
    for st, d in state_ps.items():
        if state_mix.get(st, 0) > 0:
            frames.append(d.assign(w=state_mix[st]))
    allp = pd.concat(frames, ignore_index=True)
    rows = []
    for (lab, per), d in allp.groupby(['scenario', 'period'], sort=False):
        row = {'scenario': lab, 'period': per}
        for c in ('import_share', 'iwap'):
            m = d[c].notna()
            row[c] = float(np.average(d.loc[m, c], weights=d.loc[m, 'w'])) if m.any() else np.nan
        rows.append(row)
    return _full_periods(pd.DataFrame(rows))


def saving_pairs(scen_list: list[dict], present: set[str]) -> list[tuple[str, str]]:
    """(scenario, comparator): every other scenario against No CER, and each
    battery plan against Self Consumption."""
    pairs = [(sc['label'], NO_CER) for sc in scen_list
             if sc['label'] != NO_CER and sc['label'] in present]
    if SELF_CONSUMPTION in present:
        pairs += [(sc['label'], SELF_CONSUMPTION) for sc in scen_list
                  if sc['ls'] == '--' and sc['label'] in present]
    return pairs


def saving_split(ps: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Midpoint split of IWAP_c − IWAP_k into between- and within-period terms."""
    t = ps.set_index(['scenario', 'period'])
    rows = []
    for k, c in pairs:
        row = {'scenario': k, 'comparator': c, 'between': 0.0, 'within': 0.0}
        for per in PERIODS:
            vk, vc = t.loc[(k, per), 'import_share'], t.loc[(c, per), 'import_share']
            pk, pc = t.loc[(k, per), 'iwap'], t.loc[(c, per), 'iwap']
            pbar = np.nanmean([pk, pc]) if np.isfinite(pk) or np.isfinite(pc) else 0.0
            diff = pc - pk if np.isfinite(pk) and np.isfinite(pc) else np.nan
            within = 0.5 * (vk + vc) * diff if np.isfinite(diff) else 0.0
            row['between'] += (vc - vk) * pbar
            row['within'] += within
            row[f'within_{PERIOD_KEY[per]}'] = within
            row[f'diff_{PERIOD_KEY[per]}'] = diff
        row['total'] = row['between'] + row['within']
        rows.append(row)
    cols = (['scenario', 'comparator', 'between', 'within', 'total']
            + [f'within_{PERIOD_KEY[p]}' for p in PERIODS]
            + [f'diff_{PERIOD_KEY[p]}' for p in PERIODS])
    return pd.DataFrame(rows, columns=cols)


def national_split(state_sp: dict[str, pd.DataFrame], state_mix: dict[str, float]
                   ) -> pd.DataFrame:
    """state_mix-weighted mean of the per-state splits, renormalised per pair
    over the states present (NaN-aware per column)."""
    frames = [d.assign(w=state_mix[st]) for st, d in state_sp.items()
              if state_mix.get(st, 0) > 0]
    allp = pd.concat(frames, ignore_index=True)
    num = [c for c in allp.columns if c not in ('scenario', 'comparator', 'w')]
    rows = []
    for (k, c), d in allp.groupby(['scenario', 'comparator'], sort=False):
        row = {'scenario': k, 'comparator': c}
        for col in num:
            m = d[col].notna()
            row[col] = float(np.average(d.loc[m, col], weights=d.loc[m, 'w'])) if m.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=['scenario', 'comparator'] + num)


# ── Charts ────────────────────────────────────────────────────────────────────
#
# docs/thesis_figure_spec.md through thesis_style. draw_charts reads the CSVs
# the tiers write, so --charts-only redraws every chart without the flows.
# Scenario names, tick labels, colours and the outlined, hatched
# Self-consumption bar follow the flow_exposure_analysis.py bar charts. Data
# notes (scope, years, battery, weighting) are drawn unless --no-notes is
# given.

HEIGHT_IN = 3.0
BAR_WIDTH = 0.72
GROUP_GAP = 0.375                # extra spacing at a separator between groups
# The three bars of a pair fill 0.82 of the spacing, wider than single bars
# so that the labels of adjacent bars, at different heights, do not overlap.
SPLIT_BAR, SPLIT_GAP = 0.26, 0.02
SPLIT_COLOURS = {'between': '#2171b5', 'within': '#9ecae1'}
SPLIT_NAMES = {'between': 'Between periods', 'within': 'Within periods',
               'total': 'Total saving'}
IWAP_LABEL = 'Import-weighted wholesale price (\\$/MWh)'
SAVING_LABEL = 'Saving in import-weighted price (\\$/MWh)'
GRID_PANEL_IN = 0.82             # panel height in the state-year grid
# This script's labels as flow_exposure_analysis.py's REFERENCE_SCENARIOS
# name them, for its display names and tick labels.
FE_LABELS = {NO_CER: 'No CER (load only)',
             SELF_CONSUMPTION: 'Self Consumption (Amber)'}
# Shorter names inside the two-line pair labels of the saving split.
PAIR_NAMES = {NO_CER: 'No CER', SELF_CONSUMPTION: 'Self-consumption'}


def _name(label: str) -> str:
    return fe._display_name(FE_LABELS.get(label, label))


def _tick(label: str) -> str:
    return fe._tick_label(FE_LABELS.get(label, label))


def _num2(v: float) -> str:
    return f'{v:,.2f}'.replace('-', '\u2212')


def _scenario_bar(ax, x: float, y: float, sc: dict, width: float) -> None:
    if sc['label'] == SELF_CONSUMPTION:
        for kw in sbf._sc_patches(sc['color']):
            ax.bar([x], [y], width=width, zorder=3, **kw)
    else:
        ax.bar([x], [y], width=width, color=sc['color'], linewidth=0, zorder=3)


def _scenario_patch(sc: dict):
    if sc['label'] == SELF_CONSUMPTION:
        return tuple(mpatches.Patch(**kw) for kw in sbf._sc_patches(sc['color']))
    return mpatches.Patch(color=sc['color'], linewidth=0)


def _scenario_legend(fig, shown: list[dict]) -> None:
    fig.legend(handles=[_scenario_patch(sc) for sc in shown],
               labels=[_name(sc['label']) for sc in shown],
               loc='outside lower center', ncol=len(shown), frameon=False)


def _zero_line(ax, values) -> None:
    if min(values, default=0) < 0:
        ax.axhline(0, color=ts.SEPARATOR_COLOUR, lw=0.5, zorder=2)


def fig_columns(summary: pd.DataFrame, scen_list: list[dict], note: str | None,
                out_dir: Path) -> None:
    """IWAP per scenario, labelled with the import per site-year below it."""
    s = summary.set_index('scenario')
    shown = [sc for sc in scen_list if sc['label'] in s.index]
    fig = ts.new_figure(HEIGHT_IN)
    ax = fig.subplots()
    vals = []
    for i, sc in enumerate(shown):
        v = s.loc[sc['label'], 'iwap']
        if not np.isfinite(v):
            continue
        _scenario_bar(ax, i, v, sc, BAR_WIDTH)
        mwh = s.loc[sc['label'], 'import_mwh_per_site_yr']
        fe._signed_label(ax, i, v, f'{_num2(v)}\n({mwh:,.2f} MWh/site/yr)')
        vals.append(v)
    fe._frame_categories(ax, [_tick(sc['label']) for sc in shown])
    _zero_line(ax, vals)
    fe._signed_ylim(ax, vals, pad=0.16)
    ax.set_ylabel(IWAP_LABEL)
    ax.set_title('Import-weighted wholesale price by scenario' + BASIS_SUFFIX)
    if note:
        fa._draw_notes(ax, note, 'upper right')
    ts.save_figure(fig, out_dir / 'iwap_columns')


def fig_periods(ps: pd.DataFrame, scen_list: list[dict], note: str | None,
                out_dir: Path) -> None:
    """IWAP per scenario within each TOU period, labelled with the period's
    share of the scenario's import."""
    t = ps.set_index(['scenario', 'period'])
    shown = [sc for sc in scen_list if sc['label'] in set(ps['scenario'])]
    w = BAR_WIDTH / len(shown)
    fig = ts.new_figure(HEIGHT_IN)
    ax = fig.subplots()
    vals = []
    for i, sc in enumerate(shown):
        for j, per in enumerate(PERIODS):
            x = j - BAR_WIDTH / 2 + w * (i + 0.5)
            v = t.loc[(sc['label'], per), 'iwap']
            share = t.loc[(sc['label'], per), 'import_share']
            if np.isfinite(v):
                _scenario_bar(ax, x, v, sc, w * 0.9)
                fe._signed_label(ax, x, v, f'{_num2(v)}\n({share:.0%})')
                vals.append(v)
            else:
                fe._signed_label(ax, x, 0, 'no\nimport')
    fe._frame_categories(ax, list(PERIODS))
    _zero_line(ax, vals)
    fe._signed_ylim(ax, vals, pad=0.16)
    ax.set_ylabel(IWAP_LABEL)
    ax.set_title('Import-weighted wholesale price by TOU period' + BASIS_SUFFIX)
    if note:
        fa._draw_notes(ax, note + "\nBrackets: share of the scenario's import "
                       'in the period', 'upper right')
    _scenario_legend(fig, shown)
    ts.save_figure(fig, out_dir / 'iwap_by_period')


def fig_split(sp: pd.DataFrame, scen_list: list[dict], note: str | None,
              out_dir: Path) -> None:
    """Saving per pair as its between- and within-period terms and their
    total, the pairs against each comparator grouped and separated."""
    by_label = {sc['label']: sc for sc in scen_list}
    xs, x = [], 0.0
    for i, c in enumerate(sp['comparator']):
        if i and c != sp['comparator'].iloc[i - 1]:
            x += GROUP_GAP
        xs.append(x)
        x += 1.0
    offsets = (np.arange(3) - 1) * (SPLIT_BAR + SPLIT_GAP)
    fig = ts.new_figure(HEIGHT_IN)
    ax = fig.subplots()
    vals = []
    for xi, r in zip(xs, sp.itertuples()):
        for off, part in zip(offsets, ('between', 'within', 'total')):
            v = getattr(r, part)
            if part == 'total':
                _scenario_bar(ax, xi + off, v, by_label[r.scenario], SPLIT_BAR)
            else:
                ax.bar([xi + off], [v], width=SPLIT_BAR, color=SPLIT_COLOURS[part],
                       linewidth=0, zorder=3)
            fe._signed_label(ax, xi + off, v, _num2(v))
            vals.append(v)
    for a, b in zip(xs, xs[1:]):
        if b - a > 1.0 + 1e-9:
            ax.axvline((a + b) / 2, ls=':', lw=ts.SEPARATOR_LW,
                       color=ts.SEPARATOR_COLOUR, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{PAIR_NAMES.get(k, _name(k))}\nvs {PAIR_NAMES.get(c, _name(c))}"
                        for k, c in zip(sp['scenario'], sp['comparator'])])
    ax.tick_params(axis='x', length=0)
    ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
    ax.grid(axis='x', visible=False)
    ax.axhline(0, color=ts.SEPARATOR_COLOUR, lw=0.5, zorder=2)
    fe._signed_ylim(ax, vals, pad=0.12)
    ax.set_ylabel(SAVING_LABEL)
    ax.set_title('Import price saving, between and within TOU periods' + BASIS_SUFFIX)
    # The total takes each scenario's colour, so its key shows them all.
    total_key = tuple(_scenario_patch(by_label[k]) for k in dict.fromkeys(sp['scenario']))
    ax.legend(handles=[mpatches.Patch(color=SPLIT_COLOURS['between'], linewidth=0),
                       mpatches.Patch(color=SPLIT_COLOURS['within'], linewidth=0),
                       total_key],
              labels=[SPLIT_NAMES[p] for p in ('between', 'within', 'total')],
              loc='upper left', handler_map={total_key: HandlerTuple(ndivide=None, pad=0)})
    if note:
        fa._draw_notes(ax, note + '\nSaving = comparator IWAP \u2212 scenario IWAP',
                       'lower right')
    ts.save_figure(fig, out_dir / 'iwap_saving_split')


def fig_box(table: pd.DataFrame, col: str, scen_list: list[dict], title: str,
            ylabel: str, note: str | None, out_dir: Path, stem: str,
            skip_reference: bool) -> None:
    shown = [sc for sc in scen_list if sc['label'] in set(table['scenario'])
             and not (skip_reference and sc['label'] == NO_CER)]
    data = [table.loc[table['scenario'] == sc['label'], col].dropna().to_numpy()
            for sc in shown]
    fig = ts.new_figure(HEIGHT_IN)
    ax = fig.subplots()
    line = dict(linewidth=0.6, color='0.2')
    bp = ax.boxplot(data, positions=range(len(shown)), widths=0.5, patch_artist=True,
                    showmeans=True, whiskerprops=line, capprops=line,
                    medianprops=dict(color='black', linewidth=1.0),
                    meanprops=dict(marker='D', markerfacecolor='white',
                                   markeredgecolor='black', markersize=3,
                                   markeredgewidth=0.6),
                    flierprops=dict(marker='o', markersize=2, markerfacecolor='none',
                                    markeredgecolor=ts.REFERENCE_COLOUR,
                                    markeredgewidth=0.5))
    for patch, med, sc in zip(bp['boxes'], bp['medians'], shown):
        if sc['label'] == SELF_CONSUMPTION:
            patch.set(facecolor='white', edgecolor=sc['color'], hatch=sbf.SC_HATCH,
                      linewidth=sbf.SC_EDGE_LW)
        else:
            patch.set(facecolor=sc['color'], edgecolor='0.2', linewidth=0.6)
        if sc['color'].lower() in ('#000000', 'black', 'k'):
            med.set_color('white')
    fe._frame_categories(ax, [f"{_tick(sc['label'])}\nn = {len(d)}"
                              for sc, d in zip(shown, data)])
    if skip_reference:
        ax.axhline(0, color=ts.SEPARATOR_COLOUR, lw=0.5, zorder=2)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if note:
        fa._draw_notes(ax, note + '\nBox: interquartile range and median; diamond: mean'
                       '\nWhiskers: furthest point within 1.5 \u00d7 IQR', 'upper left')
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.3 * (hi - lo))
    ts.save_figure(fig, out_dir / stem)


def fig_state_year(sy: pd.DataFrame, scen_list: list[dict], note: str | None,
                   out_dir: Path) -> None:
    """Small-multiples grid, states by years, of IWAP per scenario."""
    states = sorted(sy['state'].unique())
    years = sorted(sy['year'].unique())
    shown = [sc for sc in scen_list if sc['label'] in set(sy['scenario'])]
    fig = ts.new_figure(GRID_PANEL_IN * len(states) + 0.9)
    axes = ts.grid_axes(fig, len(states), len(years), xlabel='')
    top = np.nanmax(sy['iwap'].to_numpy())
    low = min(0.0, np.nanmin(sy['iwap'].to_numpy()))
    for i, st in enumerate(states):
        for j, yr in enumerate(years):
            ax = axes[i, j]
            ax.set_title(f'{st} {yr}', fontsize=ts.SUBTITLE_SIZE)
            ax.tick_params(axis='x', length=0, labelbottom=False)
            ax.grid(axis='x', visible=False)
            d = sy[(sy['state'] == st) & (sy['year'] == yr)].set_index('scenario')
            for k, sc in enumerate(shown):
                if sc['label'] not in d.index or not np.isfinite(d.loc[sc['label'], 'iwap']):
                    continue
                v = d.loc[sc['label'], 'iwap']
                _scenario_bar(ax, k, v, sc, BAR_WIDTH)
                fe._signed_label(ax, k, v, f'{v:,.0f}'.replace('-', '\u2212'))
    axes[0, 0].set_xlim(-0.6, len(shown) - 0.4)
    span = top - low
    axes[0, 0].set_ylim(low - 0.15 * span if low < 0 else 0, top + 0.25 * span)
    fig.supylabel(IWAP_LABEL, fontsize=mpl.rcParams['axes.labelsize'])
    _scenario_legend(fig, shown)
    title = 'Import-weighted wholesale price by state and year' + BASIS_SUFFIX
    ts.panel_suptitle(fig, f"{title} ({note.replace(chr(10), ', ')})" if note else title)
    ts.save_figure(fig, out_dir / 'iwap_by_state_and_year')


def draw_charts(out: Path, scen_list: list[dict], battery_kwh: int, weighted: bool,
                notes: bool) -> None:
    """Every chart, from the CSVs under out."""
    mpl.rcdefaults()             # financial_analysis sets its own on import
    ts.apply_style()

    def note(scope: str, years: str, national: bool = False, extra: str = '') -> str | None:
        if not notes:
            return None
        return (f'{scope}, {years}, {battery_kwh} kWh battery\n'
                f'{fe._weight_note(weighted, national)}{extra}')

    tiers = sorted(d for d in out.glob('*_all_years') if d.is_dir()) + [out / 'national']
    for d in tiers:
        if not (d / 'iwap_summary.csv').exists():
            continue
        national = d.name == 'national'
        scope = 'National' if national else d.name.split('_')[0]
        s = pd.read_csv(d / 'iwap_summary.csv')
        n = note(scope, str(s['years'].iloc[0]), national)
        fig_columns(s, scen_list, n, d)
        fig_periods(pd.read_csv(d / 'iwap_by_period.csv'), scen_list, n, d)
        fig_split(pd.read_csv(d / 'saving_split.csv'), scen_list, n, d)

    d = out / 'state_months'
    if (d / 'state_months.csv').exists():
        table = pd.read_csv(d / 'state_months.csv')
        n = note(', '.join(sorted(table['state'].unique())),
                 _years_label([str(y) for y in table['year'].unique()]),
                 extra=' within each state; each state-month counts once')
        fig_box(table, 'iwap', scen_list,
                'Import-weighted wholesale price per state-month' + BASIS_SUFFIX, IWAP_LABEL, n, d,
                'iwap_box', skip_reference=False)
        fig_box(table, 'saving_vs_no_cer', scen_list,
                'Import price saving against No CER per state-month' + BASIS_SUFFIX,
                'Saving against No CER (\\$/MWh)',
                n and n + '\nSaving = No CER IWAP \u2212 scenario IWAP', d,
                'iwap_saving_box', skip_reference=True)

    d = out / 'state_year'
    if (d / 'state_years.csv').exists():
        fig_state_year(pd.read_csv(d / 'state_years.csv'), scen_list,
                       f'{battery_kwh} kWh battery, {fe._weight_note(weighted).lower()}'
                       if notes else None, d)


# ── Main ──────────────────────────────────────────────────────────────────────

def _check(months: pd.DataFrame, state_sums: dict[str, pd.DataFrame],
           table: pd.DataFrame) -> None:
    for st, s in state_sums.items():
        for _, r in s.iterrows():
            d = months[(months['state'] == st) & (months['scenario'] == r['scenario'])]
            diff = abs(_iwap(d['cost'].sum(), d['import_kwh'].sum()) - r['iwap'])
            print(f"  CHECK {st} {r['scenario']}: IWAP from state-months vs tier, "
                  f"|diff| {diff:.2e} $/MWh")
    ref = table.loc[table['scenario'] == NO_CER, 'saving_vs_no_cer'].dropna()
    print(f"  CHECK No CER saving: max |value| {ref.abs().max() if len(ref) else 0:.2e} $/MWh")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', type=Path, default=Path('.'),
                        help='Project root holding data/output/runs, data/clean/nem_prices '
                             'and data/processed/edp (default: cwd)')
    parser.add_argument('--output-dir', type=Path, default=Path('analysis/output/import_price'),
                        help='Output directory (default: analysis/output/import_price)')
    parser.add_argument('--states', nargs='+', default=None,
                        help='Restrict to specific states (default: all found)')
    parser.add_argument('--years', nargs='+', default=None,
                        help='Restrict to specific years (default: all found)')
    parser.add_argument('--battery-kwh', type=int, default=20,
                        help='Battery size for battery scenarios (default: 20)')
    parser.add_argument('--core-only', action='store_true',
                        help='Only No CER, Solar Only, Self Consumption and Amber')
    parser.add_argument('--unweighted', action='store_true',
                        help='Straight per-site means instead of size-band weighting')
    parser.add_argument('--weights-file', type=Path,
                        default=Path('analysis/config/pv_size_bands.yaml'),
                        help='Weighting YAML (default: analysis/config/pv_size_bands.yaml)')
    parser.add_argument('--charts-only', action='store_true',
                        help='Redraw the charts from the CSVs in --output-dir without '
                             'reading the flows')
    parser.add_argument('--no-notes', dest='notes', action='store_false',
                        help='Leave the data notes off, for the thesis')
    args = parser.parse_args(argv)

    weighted = not args.unweighted
    if args.charts_only:
        draw_charts(args.output_dir, build_scenarios(core_only=False), args.battery_kwh,
                    weighted, args.notes)
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
    detail = load_all(args.data_root, scenarios, scen_list, weighted, weights_cfg,
                      args.battery_kwh)
    if detail.empty:
        print("ERROR: no flows.parquet files found.", file=sys.stderr)
        return 1
    months = (detail.groupby(['scenario', 'state', 'year', 'month'], sort=False)
              [['import_kwh', 'cost', 'n_dropped']].sum().reset_index())
    order = [sc['label'] for sc in scen_list]
    pairs = saving_pairs(scen_list, set(months['scenario']))
    out = args.output_dir

    def tier_outputs(s: pd.DataFrame, ps: pd.DataFrame, sp: pd.DataFrame, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        s.to_csv(d / 'iwap_summary.csv', index=False)
        ps.to_csv(d / 'iwap_by_period.csv', index=False)
        sp.to_csv(d / 'saving_split.csv', index=False)
        print(s.round(2).to_string(index=False))
        print(sp[['scenario', 'comparator', 'between', 'within', 'total']]
              .round(2).to_string(index=False))

    # Per-state tiers
    print("\nPer-state tiers:")
    state_sums: dict[str, pd.DataFrame] = {}
    state_ps: dict[str, pd.DataFrame] = {}
    state_sp: dict[str, pd.DataFrame] = {}
    for st in sorted(months['state'].unique()):
        s = summarise(months[months['state'] == st], ['scenario'])
        ps = period_summary(detail[detail['state'] == st])
        sp = saving_split(ps, pairs)
        state_sums[st], state_ps[st], state_sp[st] = s, ps, sp
        tier_outputs(s, ps, sp, out / f'{st}_all_years')

    # National tier
    nat_sp = None
    if weighted and len(state_sums) > 1:
        print("\nNational tier:")
        mix = weights_cfg['state_mix']
        nat = national_summary(state_sums, mix, order)
        nat_ps = national_periods(state_ps, mix)
        nat_sp = national_split(state_sp, mix)
        tier_outputs(nat, nat_ps, nat_sp, out / 'national')
    else:
        print("\nNOTE: national tier needs weighting and more than one state; skipped.")

    # State-months
    print("\nState-months:")
    table = state_month_table(months)
    d = out / 'state_months'
    d.mkdir(parents=True, exist_ok=True)
    table.to_csv(d / 'state_months.csv', index=False)
    print(f"  Saved: {d / 'state_months.csv'}")

    # State-years
    sy = summarise(months, ['scenario', 'state', 'year'])
    d = out / 'state_year'
    d.mkdir(parents=True, exist_ok=True)
    sy.to_csv(d / 'state_years.csv', index=False)

    print("\nCharts:")
    draw_charts(out, scen_list, args.battery_kwh, weighted, args.notes)

    print("\nChecks:")
    _check(months, state_sums, table)
    tiers = dict(state_sp)
    if nat_sp is not None:
        tiers['National'] = nat_sp
    for name, sp in tiers.items():
        gap = (sp['between'] + sp['within'] - sp['total']).abs().max()
        print(f"  CHECK {name}: between + within vs total, max |diff| {gap:.2e} $/MWh")
    for st, sp in state_sp.items():
        s = state_sums[st].set_index('scenario')['iwap']
        gap = max(abs((s[c] - s[k]) - t) for k, c, t in
                  zip(sp['scenario'], sp['comparator'], sp['total']))
        print(f"  CHECK {st}: split total vs IWAP difference, max |diff| {gap:.2e} $/MWh")
    for st, ps in state_ps.items():
        gap = (ps.groupby('scenario')['import_share'].sum() - 1).abs().max()
        print(f"  CHECK {st}: period shares sum to 1, max |diff| {gap:.2e}")
    print("\nDone.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
