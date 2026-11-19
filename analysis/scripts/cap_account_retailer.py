#!/usr/bin/env python3
"""Retailer position across capped-volatility variants (spec section 14).

Reads cohort_period_summary.csv for every variant found under one run and
compares the retailer's position across them, including an uncapped Amber
baseline synthesised rather than run (under passthrough every cap account
quantity is zero by definition).

The unit of observation is the cohort-month and every figure is in dollars
per site per month, so composition, spread and break-even subscription all
read on one set of units. Settlement periods shorter than a month are
summed into calendar months first, so a daily variant contributes 12
observations per cohort-year like any other.

Variants are grouped by cap threshold and each group is aggregated and
written on its own, into {output-dir}/{battery}/x{threshold}/. Variants at
different thresholds are different offers, not points on one comparison:
the cover ratio, the break-even subscription and the cohort's self-hedge
all move with x, so placing them on shared axes would compare quantities
that are not alternatives to each other. The battery segment is there for
the same reason one run must not overwrite another: --battery-name selects
which scenario directories are read, so without it in the path a 10 kWh
run lands on top of a 20 kWh one.

With --dynamic, Offer C (spec Part V) is added to every threshold group as
a fixed comparator in the spread and grid charts, and gets its own
composition chart written once to {output-dir}/{battery}/dynamic/.

Reads already-written output and computes nothing new: no flows, prices or
settlement are re-run.

Usage (from project root):
    python3 analysis/scripts/cap_account_retailer.py \\
        --data-root "/Volumes/Manual/Thesis Project" \\
        --battery-name 20kWh
"""
from __future__ import annotations

import argparse
import re
import textwrap
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Components of the section 14 identity:
#   retailer_net_cost = cost_absorbed - credit_retained - surcharge_retained
_COMPONENTS = ('retailer_cost_absorbed', 'retailer_credit_retained',
               'retailer_surcharge_retained')
_SUMMARY_NAME = 'cohort_period_summary.csv'
# Above this cover ratio the retailer is nowhere near exposure under any
# alpha, so the exact value carries no information and would otherwise
# stretch the axis by many orders of magnitude.
_COVER_OFF_SCALE = 1e4
# Symmetric floor: below this the cohort has essentially no credit against a
# real cost, and the exact value carries no more information than the bin.
_COVER_UNDER_SCALE = 1e-3
_AMBER = 'Amber'
_DYNAMIC_SUMMARY_NAME = 'dynamic_month_summary.csv'
# Offer C's components (spec section 17), in waterfall order, with the sign
# each carries into the net position.
_DYNAMIC_STEPS = (('net_fixed', 1.0), ('import_margin', 1.0),
                  ('cost_absorbed', -1.0), ('credit_retained', 1.0))
_DYNAMIC_STYLE = {
    'net_fixed':       ('#4c72b0', 'Net fixed (plan daily - network daily)'),
    'import_margin':   ('#55a868', 'Import margin'),
    'cost_absorbed':   ('#c44e52', 'Cost absorbed'),
    'credit_retained': ('#c9a227', 'Credit retained'),
}

# Composition pane colours: revenue/retained positive, absorbed negative.
# Order matches the waterfall's draw order, so the legend reads left to
# right in the same sequence as the steps.
_COMPONENT_STYLE = {
    'subscription':                 ('#4c72b0', 'Subscription'),
    'retailer_cost_absorbed':       ('#c44e52', 'Above-cap cost absorbed'),
    'retailer_credit_retained':     ('#55a868', 'Above-cap credit retained'),
    'retailer_surcharge_retained':  ('#c9a227', 'Surcharge retained'),
}

# A loss smaller than half a cent per site per month is floating-point
# residue from the settlement arithmetic, not exposure. Without this the
# "months exposed" count reports every month in which the mechanism washed
# to zero but left 1e-15 behind.
_MATERIAL_LOSS = 0.005

# The same residue at the cohort level. A period whose uncollected cost or
# unpaid credit is a fraction of a cent did not have a cap account in any
# meaningful sense; left unfloored it produces cover ratios of 1e-15 and
# stretches the axis fifteen orders of magnitude to the left over nothing.
_MATERIAL_DOLLARS = 0.01


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the retailer's position across capped-"
                    "volatility variants, per cap_account_spec.md "
                    "section 14.")
    parser.add_argument('--results-root', type=Path,
                        default=Path('analysis/output/cap_account/'
                                     'full_run_rolling_amber'),
                        help="One run's output directory, i.e. "
                             "{cap_account_analysis --output-dir}/"
                             "{run_name}.")
    parser.add_argument('--battery-name', required=True,
                        help="e.g. '20kWh'. Must match exactly.")
    parser.add_argument('--variants', nargs='*', default=None,
                        help="Variant folder names to include. Default: "
                             "every variant found. The Amber baseline is "
                             "always included; pass --no-amber to drop it.")
    parser.add_argument('--no-amber', action='store_true',
                        help="Omit the synthesised uncapped baseline.")
    parser.add_argument('--weights-file', type=Path,
                        default=Path('analysis/config/pv_size_bands.yaml'),
                        help="Source of state_mix. PV size bands are not "
                             "used here: the unit is the cohort-month, so "
                             "there is no site dimension to band.")
    parser.add_argument('--subscription-change', nargs='*', default=None,
                        metavar='VARIANT=CHANGE',
                        help="Per-variant change to the subscription fee, "
                             "$/month, negative where the fee falls: e.g. "
                             "retailer_x300_monthly_a0.6=-25. Without it "
                             "every variant is shown keeping the full "
                             "baseline fee, which for Offer B credits the "
                             "retailer with both the fee and the wholesale "
                             "gain the fee reduction was meant to hand "
                             "back. Variants not named default to no "
                             "change. Pass 'auto' alone to set each Offer "
                             "B variant's change to its own "
                             "subscription_change_vs_amber (the fee leaving "
                             "the retailer as well off as under uncapped "
                             "passthrough); Offer A and Offer C variants "
                             "are left unchanged.")
    parser.add_argument('--dynamic', default=None, metavar='VARIANT',
                        help="Include Offer C, e.g. "
                             "dynamic_tesla_electric_mixed_2026_modelled, in the "
                             "spread and grid charts of every threshold "
                             "group, and write its composition chart to "
                             "{output-dir}/{battery}/dynamic/. Default: off.")
    parser.add_argument('--unweighted', action='store_true',
                        help="Combine states by plain concatenation "
                             "instead of state_mix.")
    parser.add_argument('--output-dir', type=Path,
                        default=Path('analysis/output/cap_account_retailer'),
                        help="Parent directory. Output is written to "
                             "{output-dir}/{battery}/x{threshold}/, so "
                             "neither a different battery size nor a "
                             "different cap threshold overwrites results "
                             "already there.")
    parser.add_argument('--threshold-per-mwh', nargs='*', type=float,
                        default=None, metavar='X',
                        help="Restrict to these cap thresholds, $/MWh. "
                             "Default: aggregate every threshold found, "
                             "each into its own subdirectory.")
    return parser.parse_args(argv)


def _discover(results_root: Path, battery_name: str
              ) -> list[tuple[str, int, str, Path]]:
    """Every {state}_{year}_{battery}/{variant}/cohort_period_summary.csv
    under results_root, as (state, year, variant, path)."""
    pattern = re.compile(r'^([A-Z]{2,3})_(\d{4})_'
                         + re.escape(battery_name) + r'$')
    found: list[tuple[str, int, str, Path]] = []
    if not results_root.exists():
        return found
    for scenario in sorted(results_root.iterdir()):
        if not scenario.is_dir():
            continue
        m = pattern.match(scenario.name)
        if not m:
            continue
        for variant in sorted(scenario.iterdir()):
            if variant.is_dir() and (variant / _SUMMARY_NAME).exists():
                found.append((m.group(1), int(m.group(2)), variant.name,
                              variant / _SUMMARY_NAME))
    return found


def _variant_label(name: str, show_threshold: bool) -> str:
    """'cohort_x300_monthly_s0.01' -> 'A: monthly, 1c'.
    'retailer_x300_daily_a0.5'    -> 'B: daily, a=0.5'.
    Falls back to the raw folder name if it doesn't parse, rather than
    guessing: a mislabelled variant in a comparison chart is worse than an
    ugly one."""
    if name.startswith('dynamic_'):
        return f'C: {_dynamic_plan_label(name)}'
    m = re.match(r'^(cohort|retailer)_x([\d.]+)_(\w+?)_([sa])([\d.]+)$', name)
    if not m:
        return name
    offer, threshold, period, kind, value = m.groups()
    parts = ['A' if offer == 'cohort' else 'B']
    if show_threshold:
        parts.append(f'${threshold}')
    parts.append(period)
    if kind == 's':
        parts.append(f'{float(value) * 100:g}c')
    else:
        parts.append(f'a={float(value):g}')
    return f'{parts[0]}: ' + ', '.join(parts[1:])


_OFFER_WORDS = {'A': 'Cohort', 'B': 'Retailer', 'C': 'Dynamic'}


def _dynamic_plan_label(name: str) -> str:
    """'dynamic_tesla_electric_mixed_2026_modelled' -> 'Tesla Electric':
    the plan's retailer, from the tariff id in the variant name."""
    plan = re.sub(r'_mixed(_\d{4})?(_modelled)?$', '',
                  name[len('dynamic_'):])
    return plan.replace('_', ' ').title()


def _worded(label: str) -> str:
    """'A: monthly, 1c' -> 'Cohort: monthly, 1c'. Applied on the
    composition and spread charts, which are read without the thesis
    around them and so cannot rely on the A/B naming."""
    head, sep, tail = label.partition(': ')
    if sep and head in _OFFER_WORDS:
        return f'{_OFFER_WORDS[head]}{sep}{tail}'
    return label


def _variant_threshold(name: str) -> float | None:
    m = re.match(r'^(?:cohort|retailer)_x([\d.]+)_', name)
    return float(m.group(1)) if m else None


def _cover_threshold(name: str) -> float | None:
    """Cover ratio at which the retailer's position turns from gain to
    loss. Under Offer A the cohort ceases to cover itself below 1 -- a
    solvency point. Under Offer B the retailer holds both sides regardless
    and loses below 1/(1-alpha) -- a profitability point. The two are not
    the same kind of threshold and the chart must say so."""
    if name.startswith('cohort_'):
        return 1.0
    m = re.match(r'^retailer_.*_a([\d.]+)$', name)
    if not m:
        return None
    alpha = float(m.group(1))
    return np.inf if alpha >= 1.0 else 1.0 / (1.0 - alpha)


def _parse_subscription_changes(items) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in items or []:
        if '=' not in item:
            raise SystemExit(f"--subscription-change expects VARIANT=CHANGE, "
                             f"got {item!r}")
        variant, value = item.rsplit('=', 1)
        try:
            out[variant] = float(value)
        except ValueError:
            raise SystemExit(f"--subscription-change: {value!r} in {item!r} "
                             f"is not a number")
    return out


def _load_variant(path: Path) -> pd.DataFrame:
    """One variant-scenario's summary, aggregated to calendar months.

    Sub-monthly settlement periods are summed into months so that a daily
    variant's worst month is its worst month rather than its worst day:
    comparing a worst-day figure against a worst-month one would make daily
    settlement look worse for a reason that is purely granularity.
    """
    df = pd.read_csv(path, index_col='period', parse_dates=True)
    missing = [c for c in _COMPONENTS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path}: missing {missing}. This file predates the section 14 "
            f"retailer columns; re-run cap_account_analysis.py for it.")

    # Prefer the billing month the analysis script wrote. Re-deriving it
    # from the period key is only correct while every period key sits in
    # the target year, which is a property of the fold in _period_keys
    # rather than something this script can assume.
    month = (pd.PeriodIndex(df['month'], freq='M') if 'month' in df.columns
             else df.index.to_period('M'))
    out = df.groupby(month)[list(_COMPONENTS) + ['total_cost_uncollected',
                                                 'total_credit_unpaid',
                                                 'n_intervals_above_cap']].sum(
        min_count=0)
    # surcharge_retained is NaN on non-anchor rows by design; summing with
    # min_count=0 treats those as zero, which is what they are.
    out['retailer_net_cost'] = (out['retailer_cost_absorbed']
                               - out['retailer_credit_retained']
                               - out['retailer_surcharge_retained'])
    out['n_sites'] = int(df['n_sites'].iloc[0])
    out['subscription_per_month'] = float(df['subscription_per_month'].iloc[0])

    # Cover ratio stays at the variant's own settlement period, unlike
    # every dollar figure above. Aggregating it to months hides exactly
    # what sub-monthly settlement does: a month can over-cover many times
    # over while individual days within it did not, so a monthly ratio
    # reports a comfortable cohort in months that actually cost the
    # retailer money. The ECDF normalises differing period counts, so
    # nothing is lost by leaving each variant on its own cadence.
    cost = df['total_cost_uncollected'].to_numpy()
    credit = df['total_credit_unpaid'].to_numpy()
    live_cost = cost >= _MATERIAL_DOLLARS
    live_credit = credit >= _MATERIAL_DOLLARS
    ratios = pd.DataFrame({
        'cover_ratio': np.where(
            # No cap event, or nothing material on either side: no ratio in
            # either direction, excluded from the curve.
            (df['n_intervals_above_cap'].to_numpy() == 0)
            | (~live_cost & ~live_credit), np.nan,
            # No material cost: the retailer cannot lose, off-scale right.
            np.where(~live_cost, np.inf,
                     np.where(live_credit, credit / np.where(cost, cost, 1.0),
                              0.0))),
    }, index=df.index)
    return out, ratios


def _discover_dynamic(results_root: Path, battery_name: str, variant: str
                      ) -> list[tuple[str, int, Path]]:
    """Every {state}_{year}_{battery}/{variant}/dynamic_month_summary.csv
    under results_root, as (state, year, path)."""
    pattern = re.compile(r'^([A-Z]{2,3})_(\d{4})_'
                         + re.escape(battery_name) + r'$')
    found = []
    if not results_root.exists():
        return found
    for scenario in sorted(results_root.iterdir()):
        m = pattern.match(scenario.name) if scenario.is_dir() else None
        path = scenario / variant / _DYNAMIC_SUMMARY_NAME
        if m and path.exists():
            found.append((m.group(1), int(m.group(2)), path))
    return found


def _load_dynamic(path: Path) -> pd.DataFrame:
    """One Offer C scenario, shaped like _load_variant's monthly frame.

    retailer_net_cost is set to subscription * n_sites - net_position, so
    _build_frame's net_position = subscription - retailer_net_cost / n_sites
    returns Offer C's own position with no special case (spec section 20).
    The cap-account columns have nothing to hold and are NaN.
    """
    df = pd.read_csv(path)
    out = pd.DataFrame(index=pd.PeriodIndex(df['month'], freq='M',
                                            name='month'))
    for col in list(_COMPONENTS) + ['total_cost_uncollected',
                                    'total_credit_unpaid',
                                    'n_intervals_above_cap']:
        out[col] = np.nan
    n_sites = int(df['n_sites'].iloc[0])
    subscription = float(df['subscription_per_month'].iloc[0])
    out['retailer_net_cost'] = (subscription * n_sites
                                - df['net_position'].to_numpy())
    out['n_sites'] = n_sites
    out['subscription_per_month'] = subscription
    for comp, _ in _DYNAMIC_STEPS:
        out[comp + '_per_site'] = df[comp].to_numpy() / n_sites
    return out


def _dynamic_rows(found, variant: str) -> list[pd.DataFrame]:
    return [_load_dynamic(path).assign(state=state, year=year,
                                       variant=variant)
            for state, year, path in found]


def _amber_frame(template: pd.DataFrame) -> pd.DataFrame:
    """The uncapped baseline, synthesised rather than run: every cap
    account quantity is zero under passthrough, so the retailer's position
    is the subscription and nothing else."""
    out = template.copy()
    for col in list(_COMPONENTS) + ['retailer_net_cost',
                                    'total_cost_uncollected',
                                    'total_credit_unpaid',
                                    'n_intervals_above_cap']:
        out[col] = 0.0
    return out


def _build_frame(discovered, variants, include_amber, fee_changes=None,
                 dynamic_rows=None):
    """Returns (monthly frame, per-settlement-period cover ratio frame).
    dynamic_rows, from _dynamic_rows, are appended as they are: Offer C has
    no cover ratio and is not a template for the Amber rows."""
    rows = []
    ratio_rows = []
    amber_source: dict[tuple[str, int], pd.DataFrame] = {}
    for state, year, variant, path in discovered:
        if variants is not None and variant not in variants:
            continue
        df, ratios = _load_variant(path)
        df = df.assign(state=state, year=year, variant=variant)
        rows.append(df)
        ratio_rows.append(ratios.assign(state=state, year=year,
                                        variant=variant))
        amber_source.setdefault((state, year), df)

    if include_amber:
        for (state, year), template in amber_source.items():
            rows.append(_amber_frame(template).assign(variant=_AMBER))

    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    rows.extend(dynamic_rows or [])

    frame = pd.concat(rows).reset_index().rename(columns={'index': 'month',
                                                          'period': 'month'})
    # The fee actually offered under each variant. Kept separate from
    # baseline_subscription, which stays the uncapped passthrough fee and is
    # what the Section 14 amber-equivalent figure is measured against -- so
    # applying a change here cannot feed back into the calculation of what
    # change the retailer could afford.
    frame['baseline_subscription'] = frame['subscription_per_month']
    if fee_changes:
        frame['subscription_per_month'] = (
            frame['subscription_per_month']
            + frame['variant'].map(fee_changes).fillna(0.0))
    frame['net_position_per_site'] = (frame['subscription_per_month']
                                     - frame['retailer_net_cost']
                                     / frame['n_sites'])
    for col in _COMPONENTS:
        frame[col + '_per_site'] = frame[col] / frame['n_sites']
    frame['subscription'] = frame['subscription_per_month']

    # A period with no above-cap cost has an infinite ratio -- the retailer
    # cannot lose -- and is counted above every threshold. A period with no
    # above-cap intervals at all has no ratio in either direction and is
    # excluded from the curve, since a cohort that never faced a cap event
    # is not the same as one that faced them and covered them.
    ratio_frame = pd.concat(ratio_rows).reset_index(drop=True)
    return frame, ratio_frame


def _national_weights(frame: pd.DataFrame, state_mix: dict | None
                      ) -> pd.Series:
    """One weight per cohort-month. Years need no adjustment: every
    state-year contributes exactly 12 cohort-months, so years carry equal
    weight by construction. States combine by state_mix."""
    if state_mix is None:
        return pd.Series(1.0, index=frame.index)

    # Renormalise over the states actually present, matching
    # financial_analysis.combine_curves_state_weighted: a run covering a
    # subset of states should still pool sensibly using the states it has,
    # rather than being diluted by ones it was never run for.
    present = [s for s in frame['state'].unique() if s in state_mix]
    total = sum(state_mix[s] for s in present)
    if total <= 0:
        return pd.Series(0.0, index=frame.index)

    weights = pd.Series(0.0, index=frame.index)
    for (variant, state), idx in frame.groupby(['variant', 'state']).groups.items():
        if state not in state_mix:
            continue
        weights.loc[idx] = (state_mix[state] / total) / len(idx)
    return weights


def _weighted_percentile(values: np.ndarray, weights: np.ndarray | None,
                         q: float) -> float:
    """Cumulative-weight interpolation on the midpoint convention, matching
    cap_account_aggregate.py so the two scripts' percentiles agree."""
    if values.size == 0:
        return float('nan')
    if weights is None:
        return float(np.percentile(values, q))
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(q / 100.0, cum, v))


def _box_stats(values: np.ndarray, weights: np.ndarray | None,
               label: str) -> dict:
    """Precomputed box for ax.bxp: matplotlib's boxplot cannot take
    weights, so percentiles are computed here and the box drawn from them.
    Whiskers at the 5th/95th with everything beyond plotted individually --
    the outliers are the point, since the worst month is the figure a
    retailer hedging this book cares about."""
    lo = _weighted_percentile(values, weights, 5.0)
    hi = _weighted_percentile(values, weights, 95.0)
    return {
        'label': label,
        'med': _weighted_percentile(values, weights, 50.0),
        'q1': _weighted_percentile(values, weights, 25.0),
        'q3': _weighted_percentile(values, weights, 75.0),
        'whislo': lo,
        'whishi': hi,
        'fliers': values[(values < lo) | (values > hi)],
    }


def _wrap(label: str) -> str:
    """'A: monthly, 1c' -> 'A: monthly\\n1c', so x-axis labels sit
    horizontally without crowding."""
    return label.replace(', ', '\n', 1)


# Offer A variants read as a progression from most mutualised to least:
# monthly nets within the month and claws back at a month-averaged rate,
# daily at a day-averaged rate, interval removes netting altogether. Offer
# B then removes the cohort entirely. Alphabetical order on the folder name
# would give daily, interval, monthly, which is none of those things.
_PERIOD_ORDER = {'monthly': 0, 'weekly': 1, 'daily': 2, 'interval': 3}
_SPREAD_Y_MAX = 100.0


def _ordered_variants(frame: pd.DataFrame) -> list[str]:
    """Amber, then Offer A, then Offer C, then Offer B. Within A and B,
    coarsest settlement first and then ascending alpha, so the settlement
    progression reads the same way in both offers."""
    def key(v):
        if v == _AMBER:
            return (0, 0.0, 0.0, '')
        if v.startswith('cohort_'):
            m = re.match(r'^cohort_x[\d.]+_(\w+?)_s', v)
            period = m.group(1) if m else ''
            return (1, _PERIOD_ORDER.get(period, 99), 0.0, v)
        if v.startswith('dynamic_'):
            return (2, 0.0, 0.0, v)
        mp = re.match(r'^retailer_x[\d.]+_(\w+?)_a', v)
        ma = re.match(r'^retailer_.*_a([\d.]+)$', v)
        period = mp.group(1) if mp else ''
        return (3, _PERIOD_ORDER.get(period, 99),
                float(ma.group(1)) if ma else 0.0, v)
    return sorted(frame['variant'].unique(), key=key)


def _variant_stats(frame, weights, variants):
    """Weighted mean of each component and of the net position, per site
    per month, for every variant."""
    out = {}
    for v in variants:
        m = frame['variant'] == v
        w = weights[m].to_numpy()
        if not w.sum():
            continue
        rec = {'net': float(np.average(frame.loc[m, 'net_position_per_site'],
                                      weights=w)),
               'subscription': float(np.average(
                   frame.loc[m, 'subscription_per_month'], weights=w))}
        for comp in _COMPONENTS:
            rec[comp] = float(np.average(frame.loc[m, comp + '_per_site'],
                                         weights=w))
        out[v] = rec
    return out


def _plot_composition(frame, weights, labels, out_path, battery_name,
                      cap_label,
                      weight_label):
    """Waterfall per variant: subscription, then each component of the
    section 14 identity as a step, ending at the net position. A stacked
    bar with a negative segment reads badly -- the eye cannot add a bar
    below the axis to bars above it -- whereas a waterfall shows the
    arithmetic directly."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    variants = _ordered_variants(frame)
    stats = _variant_stats(frame, weights, variants)
    steps = [('subscription', 1.0), ('retailer_cost_absorbed', -1.0),
             ('retailer_credit_retained', 1.0),
             ('retailer_surcharge_retained', 1.0)]
    step_labels = ['Subscription', 'Cost\nabsorbed', 'Credit\nretained',
                   'Surcharge\nretained', 'Net']

    lo, hi = 0.0, 0.0
    for rec in stats.values():
        run = 0.0
        for comp, sign in steps:
            run += sign * rec[comp]
            lo, hi = min(lo, run), max(hi, run)
    pad = 0.08 * (hi - lo) if hi > lo else 1.0

    ncols = min(3, len(variants))
    nrows = int(np.ceil(len(variants) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.4 * ncols + 0.8,
                                      0.8 * (3.6 * nrows + 1.9)),
                             sharey=True, squeeze=False)
    flat = [ax for row in axes for ax in row]
    for ax in flat[len(variants):]:
        ax.axis('off')
    for ax, v, label in zip(flat, variants, labels):
        rec = stats[v]
        run = 0.0
        for i, (comp, sign) in enumerate(steps):
            delta = sign * rec[comp]
            colour = _COMPONENT_STYLE[comp][0]
            ax.bar(i, delta, bottom=run, color=colour, width=0.62)
            if i:
                ax.plot([i - 1 + 0.31, i - 0.31], [run, run],
                        color='#888888', linewidth=0.8, linestyle='-')
            run += delta
            # Under Offer A the components are cents against a $25
            # subscription, so the bar is invisible at a shared scale that
            # also has to hold Offer B. The label carries the value.
            ax.annotate(f'{delta:+,.2f}', (i, max(run, run - delta)),
                        textcoords='offset points', xytext=(0, 3),
                        ha='center', fontsize=7, color='#444444')
        ax.bar(len(steps), run, color='#3b3b3b', width=0.62)
        ax.plot([len(steps) - 1 + 0.31, len(steps) - 0.31], [run, run],
                color='#888888', linewidth=0.8)
        ax.annotate(f'${run:,.2f}', (len(steps), run),
                    textcoords='offset points', xytext=(0, 4),
                    ha='center', fontsize=8)
        ax.axhline(0, color='black', linewidth=1)
        # Steps are identified by the legend and by their left-to-right
        # order, so tick labels would only repeat it six times over.
        ax.set_xticks([])
        ax.set_title(label, fontsize=10)
        ax.set_ylim(lo - pad, hi + pad)
        ax.grid(axis='y', alpha=0.3)
    for row in axes:
        row[0].set_ylabel('$ per site per month')

    handles = [plt.Rectangle((0, 0), 1, 1, color=c)
               for c, _ in _COMPONENT_STYLE.values()]
    handles.append(plt.Rectangle((0, 0), 1, 1, color='#3b3b3b'))
    names = [n for _, n in _COMPONENT_STYLE.values()] + ['Net position']
    fig.legend(handles, names, loc='lower center', ncol=5, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, 0.055))
    fig.suptitle(f"Composition of the retailer's mean cohort-month  |  "
                 f"{battery_name}  |  {cap_label}  |  {weight_label}",
                 fontsize=12)
    fig.text(0.5, 0.005,
             "Under Offer A, credit retained is the cohort's own above-cap "
             "export credit redirected to cover other members' forgiven "
             "costs: it is min(cost, credit) and so never exceeds cost "
             "absorbed.\nIt is mutualisation passing through the retailer's "
             "book, not margin. Under Offer B it is margin -- (1-alpha) of "
             "all above-cap export value, unrelated to what any cost was.",
             ha='center', va='bottom', fontsize=8, color='#555555')
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_spread(frame, weights, labels, out_path, battery_name,
                 cap_label,
                 weight_label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    variants = _ordered_variants(frame)
    stats = [_box_stats(frame.loc[frame['variant'] == v,
                                  'net_position_per_site'].to_numpy(),
                        weights[frame['variant'] == v].to_numpy(), label)
             for v, label in zip(variants, labels)]

    # Per-variant width only; the constant is axis furniture and does not
    # scale with the number of variants. Wider boxes in the same slot
    # take the reduction out of the gaps rather than the boxes.
    fig, ax = plt.subplots(figsize=(1.4 * len(variants) + 3.4, 6))
    ax.bxp(stats, positions=list(range(len(stats))), widths=0.7,
           showfliers=True,
           boxprops={'color': '#333333'},
           whiskerprops={'color': '#333333'},
           capprops={'color': '#333333'},
           medianprops={'color': '#c44e52', 'linewidth': 2},
           flierprops={'marker': 'o', 'markersize': 3.5, 'alpha': 0.45,
                       'markerfacecolor': '#777777',
                       'markeredgecolor': 'none'})

    amber = frame.loc[frame['variant'] == _AMBER, 'net_position_per_site']
    if not amber.empty:
        ax.axhline(float(amber.iloc[0]), color='#4c72b0', linestyle='--',
                   linewidth=1.2)

    handles = [
        plt.Line2D([], [], color='#c44e52', linewidth=2, label='Median'),
        plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='#333333',
                      label='Interquartile range (25-75)'),
        plt.Line2D([], [], color='#333333', label='Whiskers (5-95)'),
        plt.Line2D([], [], marker='o', linestyle='none', markersize=3.5,
                   markerfacecolor='#777777', markeredgecolor='none',
                   alpha=0.45, label='Cohort-months beyond the whiskers'),
        plt.Line2D([], [], color='#4c72b0', linestyle='--',
                   label='Uncapped passthrough (subscription only)'),
    ]
    # Offer B's credit-retained months run far above the rest and would
    # otherwise set the scale for every variant. Points above the limit are
    # counted on the axis rather than silently dropped.
    ax.set_ylim(top=_SPREAD_Y_MAX)
    clipped = 0
    for pos, v in enumerate(variants):
        vals = frame.loc[frame['variant'] == v,
                         'net_position_per_site'].to_numpy()
        n_above = int((vals > _SPREAD_Y_MAX).sum())
        if n_above:
            clipped += n_above
            ax.annotate(f'{n_above} above', xy=(pos, _SPREAD_Y_MAX),
                        xytext=(0, -12), textcoords='offset points',
                        ha='center', fontsize=7.5, color='#777777')

    ax.legend(handles=handles, fontsize=8, loc='lower left')
    ax.set_xlim(-0.6, len(stats) - 0.4)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([_wrap(l) for l in labels], fontsize=9)
    ax.set_ylabel('Retailer net position, $ per site per month')
    ax.set_title(f"Spread of the retailer's position across cohort-months  "
                 f"|  {battery_name}  |  {cap_label}  |  {weight_label}\n"
                 f"Below the dashed line, the cap mechanism cost the "
                 f"retailer money that month", fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    # Wrapped to the figure width: the note overflows the canvas once the
    # variant count is low enough to make the figure narrow.
    note = textwrap.wrap(
        "Where no box is visible the interquartile range has collapsed "
        "onto the median: most cohort-months have no cap event, so the "
        "position sits at exactly the subscription and the 25th, 50th "
        "and 75th percentiles coincide.",
        width=max(60, int(fig.get_figwidth() * 15)))
    fig.tight_layout(rect=(0, 0.035 + 0.022 * len(note), 1, 1))
    fig.text(0.01, 0.01, '\n'.join(note),
             ha='left', va='bottom', fontsize=8, color='#555555')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_subscription(frame, weights, labels, out_path, battery_name,
                       cap_label,
                       weight_label):
    """The fee at which the retailer is as well off as under uncapped
    passthrough -- NOT the fee at which the cap mechanism alone breaks even.

    The model represents only the wholesale layer, so a fee that zeroes the
    mechanism's own position would imply the subscription could fall to
    nothing, when in reality it covers acquisition, billing and everything
    else the model does not see. The meaningful quantity is the change
    against the current fee, holding the retailer as well off as it is
    today.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    variants = _ordered_variants(frame)
    stats = _variant_stats(frame, weights, variants)
    # variants[0] is the uncapped baseline, whose fee is never adjusted, so
    # `current` stays the baseline fee. The offered fee cancels out of
    # `equiv` algebraically -- sub_v - (sub_v - cost_v - current) -- so
    # applying a --subscription-change cannot feed back into the figure for
    # what change the retailer could afford.
    current = stats[variants[0]]['subscription']
    equiv = [stats[v]['subscription'] - (stats[v]['net'] - current)
             for v in variants]
    deltas = [e - current for e in equiv]
    offered = [stats[v]['subscription'] for v in variants]

    fig, ax = plt.subplots(figsize=(1.7 * len(variants) + 3.6, 5.6))
    def colour(d):
        if abs(d) < 0.005:
            return '#4c72b0'          # unchanged: the baseline itself
        return '#55a868' if d < 0 else '#c44e52'

    ax.bar(range(len(variants)), equiv, color=[colour(d) for d in deltas],
           width=0.6)
    ax.axhline(current, color='black', linestyle='--', linewidth=1.2,
               label=f'Current fee ${current:.2f}')
    for i, (e, d) in enumerate(zip(equiv, deltas)):
        # Label outside the bar, so it never lands on the reference line.
        offset = 6 if e >= 0 else -18
        ax.annotate(f'${e:,.2f}\n({d:+.2f})', (i, e),
                    textcoords='offset points', xytext=(0, offset),
                    ha='center', fontsize=8)

    if any(abs(o - current) > 1e-9 for o in offered):
        ax.scatter(range(len(variants)), offered, marker='_', s=420,
                   color='black', linewidths=2.2, zorder=4,
                   label='Fee actually offered')
    pts = equiv + offered + [current, 0.0]
    span = max(pts) - min(pts)
    ax.set_ylim(min(pts) - 0.25 * max(span, 0.5),
                max(pts) + 0.25 * max(span, 0.5))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([_wrap(l) for l in labels], fontsize=9)
    ax.set_ylabel('Subscription, $ per site per month')
    ax.set_title(f"Subscription leaving the retailer as well off as "
                 f"uncapped passthrough  |  {battery_name}  |  {cap_label}"
                 f"  |  {weight_label}\nGreen: the fee could fall.  Red: the fee "
                 f"would have to rise.  Blue: the baseline itself.\n"
                 f"Where a black dash sits on its bar, the fee offered "
                 f"matches what the retailer can afford.",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_cover_ratio(ratio_frame, ratio_weights, frame, weights, labels,
                      cap_label,
                      out_path, battery_name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ordered = _ordered_variants(frame)
    variants = [v for v in ordered if v != _AMBER]
    labels = [l for v, l in zip(ordered, labels) if v != _AMBER]
    fig, ax = plt.subplots(figsize=(11, 7.5))
    colours = plt.cm.tab10(np.linspace(0, 1, 10))

    excluded_note = []
    table_rows = []
    # The unadjusted baseline fee: row 0 is whichever row sorts first, and
    # may carry a --subscription-change.
    subscription = float(frame['baseline_subscription'].iloc[0])
    for i, (v, label) in enumerate(zip(variants, labels)):
        # A cover ratio says which side of break-even a period fell, not
        # how much was at stake: a ratio of 0.01 can be cents or thousands.
        # The table carries the magnitude the curve cannot, in dollars per
        # site per month.
        dm = frame['variant'] == v
        wv = weights[dm].to_numpy()
        loss = np.maximum(0.0, (frame.loc[dm, 'retailer_net_cost']
                                / frame.loc[dm, 'n_sites']).to_numpy())
        exposed = float(np.average((loss > _MATERIAL_LOSS).astype(float),
                                   weights=wv))

        m = ratio_frame['variant'] == v
        ratios = ratio_frame.loc[m, 'cover_ratio'].to_numpy()
        w = ratio_weights[m].to_numpy()
        n_no_event = int(np.isnan(ratios).sum())  # no above-cap intervals
        keep = ~np.isnan(ratios)
        ratios, w = ratios[keep], w[keep]
        if ratios.size == 0:
            continue
        # A month with no above-cap cost has an infinite ratio, and a month
        # with a trivially small one runs to 1e17 -- either way the retailer
        # is nowhere near exposure. Both fold into a single off-scale bin at
        # the right, so the axis stays on the range where the threshold
        # actually sits instead of spanning twenty orders of magnitude.
        plotted = np.where(ratios > _COVER_OFF_SCALE, _COVER_OFF_SCALE * 2,
                           ratios)
        plotted = np.where(plotted < _COVER_UNDER_SCALE,
                           _COVER_UNDER_SCALE / 2, plotted)
        n_off = int((ratios > _COVER_OFF_SCALE).sum())
        n_under = int((ratios < _COVER_UNDER_SCALE).sum())
        order = np.argsort(plotted)
        ecdf = np.cumsum(w[order]) / w.sum()
        ax.step(plotted[order], ecdf, where='post', color=colours[i % 10],
                label=label, linewidth=1.8)

        # Read straight off the curve at its own threshold, on the same
        # denominator, so the table's first column and the chart agree by
        # construction rather than by coincidence.
        threshold = _cover_threshold(v)
        below = (float(np.average((ratios < threshold).astype(float),
                                  weights=w))
                 if threshold is not None and np.isfinite(threshold) else
                 float('nan'))
        table_rows.append((label, below * 100, exposed * 100,
                           float(np.average(loss, weights=wv)) / subscription
                           * 100, float(loss.max())))

        if threshold is not None and np.isfinite(threshold):
            ax.axvline(threshold, color=colours[i % 10], linestyle=':',
                       linewidth=1.2, alpha=0.8)
        excluded_note.append(f'{label}: {n_no_event} with no cap event, '
                            f'{n_under} below scale, {n_off} above scale, '
                            f'of {len(keep)} periods')

    ax.set_xscale('log')
    ax.set_xlim(_COVER_UNDER_SCALE / 3, _COVER_OFF_SCALE * 3)
    ax.axvspan(_COVER_OFF_SCALE, _COVER_OFF_SCALE * 3, color='#eeeeee',
               zorder=0)
    ax.axvspan(_COVER_UNDER_SCALE / 3, _COVER_UNDER_SCALE, color='#eeeeee',
               zorder=0)
    # Anchored to the axis figure fraction, not a data coordinate: the
    # bands are narrow on a log axis (a factor of 3), so rotated text
    # sized to fit one clips the axis edge on the other. Figure-fraction
    # placement with axes-relative x guarantees the text sits inside its
    # band's screen width regardless of how that maps to data units.
    # Rotated text extends from its anchor in both directions, so even
    # axes-fraction placement clips a two-line label in a band this narrow.
    # One short line, anchored so it grows away from the plotted curves
    # rather than across the band's centre.
    # Below both the legend (top-left, roughly y>0.78) and the table
    # (roughly y=0.35-0.62), in the gap between them -- not overlapping
    # either.
    ax.annotate(f'never exposed (>{_COVER_OFF_SCALE:g})',
                (0.99, 0.70), xycoords='axes fraction',
                ha='right', va='center', fontsize=6.5, color='#777777',
                rotation=90)
    ax.annotate(f'no cover (<{_COVER_UNDER_SCALE:g})',
                (0.01, 0.70), xycoords='axes fraction',
                ha='left', va='center', fontsize=6.5, color='#777777',
                rotation=90)
    ax.set_xlabel('Cover ratio, total_credit_unpaid / total_cost_uncollected '
                  '(log scale)')
    ax.set_ylabel("Fraction of that variant's own settlement periods "
                  "at or below")
    ax.set_title(f'Cohort cover ratio by variant  |  {cap_label}\n'
                 'Dotted line = that variant\'s own break-even: solvency '
                 '(ratio 1) under Offer A, profitability (1/(1-alpha)) '
                 'under Offer B', fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='upper left')

    # Field width fits the longest label the naming scheme produces
    # ('B: monthly, a=0.75'), so nothing truncates.
    # 'months' field widened to 14: '(at a loss)' is 11 chars, so at
    # width 11 it filled the column edge to edge with no padding on either
    # side and read as spanning both neighbouring columns rather than
    # right-aligned under its own header.
    lines = [f"{'':20s}{'periods':>10s}{'months':>14s}{'shortfall':>11s}"
             f"{'worst mth':>11s}",
             f"{'':20s}{'uncovered':>10s}{'(at a loss)':>14s}"
             f"{'/revenue':>11s}{'($/site)':>11s}"]
    for label, below, exposed, share, worst in table_rows:
        lines.append(f"{label[:19]:20s}{below:9.1f}%{exposed:13.1f}%"
                     f"{share:10.2f}%{worst:11.2f}")
    ax.text(0.008, 0.45, '\n'.join(lines), transform=ax.transAxes,
            ha='left', va='center', fontsize=7.5, family='monospace',
            bbox={'facecolor': 'white', 'edgecolor': '#cccccc',
                  'boxstyle': 'round,pad=0.5'})

    fig.tight_layout(rect=(0, 0.30, 1, 1))
    fig.text(0.01, 0.28,
             'CURVES are per settlement period: height where a curve '
             'crosses its own dotted line = fraction of that variant\'s '
             'periods in which the cohort failed to cover itself.\n'
             'Each variant stays on its own cadence rather than a common '
             'monthly one. A month can over-cover many times over while '
             'individual days within it did not, so a monthly ratio\n'
             'reports a comfortable cohort in months that cost the retailer '
             'money. The ECDF normalises the differing period counts.\n'
             'TABLE, first column: fraction of periods below the '
             'threshold, read off the curve above on the same denominator. '
             'Remaining columns are per cohort-month, in dollars.\n'
             '"Shortfall / revenue" is measured across every cohort-month, '
             'not only the ones at a loss: it is total shortfall over the '
             'full period as a share of total subscription revenue.\n'
             '"Periods uncovered" is the wholesale position alone -- did '
             'the cohort cover itself. "Months at a loss" is after the '
             'hedging surcharge, which is pooled monthly and\noften covers '
             'the gap, so the two differ: the gap between them is how much '
             'work the surcharge is doing. A daily variant can still show '
             'more months at a loss than\nuncovered days, since one bad day '
             'carries its whole month.\n'
             'Per variant, months with no cap event (no ratio in either '
             'direction, excluded) and months in the off-scale bin:\n  '
             + '\n  '.join(excluded_note),
             ha='left', va='top', fontsize=7.5, color='#555555')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_grid(frame, labels_by_variant, out_path, battery_name,
               cap_label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    states = sorted(frame['state'].unique())
    years = sorted(frame['year'].unique())
    variants = _ordered_variants(frame)
    labels = [labels_by_variant[v] for v in variants]

    fig, axes = plt.subplots(len(states), len(years),
                             figsize=(3.4 * len(years) + 0.8,
                                      2.8 * len(states) + 0.9),
                             sharex=True, sharey=True, squeeze=False)
    # Percentile rather than min/max limits here: the grid is for spotting
    # patterns across state and year, and a single extreme cohort-month
    # would otherwise flatten every other box to a line. The national
    # spread pane keeps full limits, since reading the tails is its job.
    vals = frame['net_position_per_site'].to_numpy()
    lo, hi = float(vals.min()), float(vals.max())
    pad = 0.08 * (hi - lo) if hi > lo else 1.0
    # Full range, not a percentile clip. Zero is not forced in either: with
    # the subscription at ~$25 the meaningful reference is uncapped
    # passthrough, not the origin, and forcing zero squeezed every box into
    # the top tenth of the panel. Once zero is gone the true min and max fit
    # comfortably, so nothing needs clipping off the bottom.
    ylim = (lo - pad, hi + pad)
    # The unadjusted baseline fee: row 0 is whichever row sorts first, and
    # may carry a --subscription-change.
    subscription = float(frame['baseline_subscription'].iloc[0])

    for r, state in enumerate(states):
        for c, year in enumerate(years):
            ax = axes[r][c]
            cell = frame[(frame['state'] == state) & (frame['year'] == year)]
            if cell.empty:
                ax.axis('off')
                continue
            stats = []
            for v, label in zip(variants, labels):
                m = cell['variant'] == v
                if not m.any():
                    continue
                stats.append(_box_stats(
                    cell.loc[m, 'net_position_per_site'].to_numpy(),
                    None, label))
            if stats:
                ax.bxp(stats, positions=list(range(len(stats))),
                       showfliers=True,
                       boxprops={'color': '#333333'},
                       whiskerprops={'color': '#333333'},
                       capprops={'color': '#333333'},
                       medianprops={'color': '#c44e52', 'linewidth': 1.6},
                       flierprops={'marker': 'o', 'markersize': 3,
                                   'alpha': 0.45,
                                   'markerfacecolor': '#777777',
                                   'markeredgecolor': 'none'})
            ax.axhline(subscription, color='#4c72b0', linestyle='--',
                       linewidth=0.9)
            ax.set_ylim(ylim)
            ax.set_title(f"{state} {year}  (n={int(cell['n_sites'].iloc[0])} "
                         f"sites)", fontsize=9)
            ax.tick_params(axis='x', labelrotation=45, labelsize=7)
            ax.grid(axis='y', alpha=0.3)
            if c == 0:
                ax.set_ylabel('$ / site / month', fontsize=8)

    fig.suptitle(f"Retailer net position per site per month  |  "
                 f"{battery_name}  |  {cap_label}  |  "
                 f"12 cohort-months per panel, "
                 f"unweighted, full range\nDashed line: uncapped "
                 f"passthrough (subscription only). Red line: median. Where "
                 f"no box is visible the interquartile range has collapsed "
                 f"onto it.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_dynamic_composition(frame: pd.DataFrame, state_mix, out_dir: Path,
                              battery_name: str, variant: str) -> None:
    """Waterfall of Offer C's section 17 components, per site per month:
    a national panel weighted by state_mix and one panel per state, against
    a dashed line at the Amber subscription. Also writes the component
    means to dynamic_summary.csv. Written once, not per threshold group,
    because nothing in it depends on the threshold (spec section 20)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    weights = _national_weights(frame, state_mix)
    national = 'National' + (' (weighted by state)' if state_mix is not None
                             else ' (unweighted)')
    panels = [(national, frame.index, weights)]
    for state in sorted(frame['state'].unique()):
        idx = frame.index[frame['state'] == state]
        panels.append((state, idx, pd.Series(1.0, index=idx)))

    rows = []
    for name, idx, w in panels:
        rec = {'panel': name, 'n_cohort_months': len(idx)}
        for comp, _ in _DYNAMIC_STEPS:
            rec[comp] = float(np.average(frame.loc[idx, comp + '_per_site'],
                                         weights=w.loc[idx]))
        rec['net_position'] = float(np.average(
            frame.loc[idx, 'net_position_per_site'], weights=w.loc[idx]))
        rows.append(rec)
    summary = pd.DataFrame(rows)
    subscription = float(frame['baseline_subscription'].iloc[0])

    lo, hi = min(0.0, subscription), subscription
    for rec in rows:
        run = 0.0
        for comp, sign in _DYNAMIC_STEPS:
            run += sign * rec[comp]
            lo, hi = min(lo, run), max(hi, run)
    pad = 0.08 * (hi - lo) if hi > lo else 1.0

    ncols = min(3, len(rows))
    nrows = int(np.ceil(len(rows) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.4 * ncols + 0.8, 3.2 * nrows + 1.6),
                             sharey=True, squeeze=False)
    flat = [ax for row in axes for ax in row]
    for ax in flat[len(rows):]:
        ax.axis('off')
    for ax, rec in zip(flat, rows):
        run = 0.0
        for i, (comp, sign) in enumerate(_DYNAMIC_STEPS):
            delta = sign * rec[comp]
            ax.bar(i, delta, bottom=run, color=_DYNAMIC_STYLE[comp][0],
                   width=0.62)
            if i:
                ax.plot([i - 1 + 0.31, i - 0.31], [run, run],
                        color='#888888', linewidth=0.8)
            run += delta
            ax.annotate(f'{delta:+,.2f}', (i, max(run, run - delta)),
                        textcoords='offset points', xytext=(0, 3),
                        ha='center', fontsize=7, color='#444444')
        n = len(_DYNAMIC_STEPS)
        ax.bar(n, run, color='#3b3b3b', width=0.62)
        ax.plot([n - 1 + 0.31, n - 0.31], [run, run], color='#888888',
                linewidth=0.8)
        ax.annotate(f'${run:,.2f}', (n, run), textcoords='offset points',
                    xytext=(0, 4), ha='center', fontsize=8)
        ax.axhline(0, color='black', linewidth=1)
        ax.axhline(subscription, color='#4c72b0', linestyle='--',
                   linewidth=1.2)
        ax.set_xticks([])
        ax.set_title(f"{rec['panel']}  (n={rec['n_cohort_months']} "
                     f"cohort-months)", fontsize=9.5)
        ax.set_ylim(lo - pad, hi + pad)
        ax.grid(axis='y', alpha=0.3)
    for row in axes:
        row[0].set_ylabel('$ per site per month')

    handles = [plt.Rectangle((0, 0), 1, 1, color=c)
               for c, _ in _DYNAMIC_STYLE.values()]
    handles += [plt.Rectangle((0, 0), 1, 1, color='#3b3b3b'),
                plt.Line2D([], [], color='#4c72b0', linestyle='--')]
    names = ([n for _, n in _DYNAMIC_STYLE.values()]
             + ['Net position', f'Amber subscription ${subscription:.2f}'])
    fig.legend(handles, names, loc='lower center', ncol=3, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Composition of the retailer's mean cohort-month, "
                 f"{_worded(_variant_label(variant, False))}  |  "
                 f"{battery_name}", fontsize=12)
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / 'retailer_composition_dynamic.png', dpi=150)
    plt.close(fig)
    summary.to_csv(out_dir / 'dynamic_summary.csv', index=False)


def _threshold_group(variant: str) -> str:
    """Subdirectory name for a variant's cap threshold. A variant whose
    folder name does not parse goes to 'x_unparsed' rather than being
    pooled with a threshold it may not share."""
    x = _variant_threshold(variant)
    return f'x{x:g}' if x is not None else 'x_unparsed'


def _group_sort_key(group: str) -> tuple[int, float]:
    """Numeric order, so x150 precedes x300 and x1000 follows both."""
    if group == 'x_unparsed':
        return (1, 0.0)
    return (0, float(group[1:]))


def _composition_means(frame: pd.DataFrame, weights: pd.Series
                       ) -> pd.DataFrame:
    """National weighted mean of each gross margin component, per site per
    month, on one basis across the cap account offers and the dynamic plan,
    for the thesis composition figure (alt_retailer_composition_figure.py).

    subscription_or_net_fixed is the subscription for the cap account offers
    and the plan's daily charges net of the network's for the dynamic plan.
    A component an offer does not have is left empty rather than zero:
    import_margin exists only for the dynamic plan and surcharge_retained
    only for the cohort offers. gross_margin is the sum of the components
    with cost absorbed subtracted."""
    rows = []
    for v in _ordered_variants(frame):
        if v == _AMBER:
            continue
        m = frame['variant'] == v
        w = weights[m].to_numpy()
        is_dynamic = v.startswith('dynamic_')

        def mean(col):
            return float(np.average(frame.loc[m, col].to_numpy(), weights=w))

        rows.append({
            'variant': v,
            'n_cohort_months': int(m.sum()),
            'subscription_or_net_fixed': mean('net_fixed_per_site'
                                              if is_dynamic
                                              else 'subscription_per_month'),
            'import_margin': (mean('import_margin_per_site')
                              if is_dynamic else np.nan),
            'cost_absorbed': mean('cost_absorbed_per_site' if is_dynamic
                                  else 'retailer_cost_absorbed_per_site'),
            'credit_retained': mean('credit_retained_per_site' if is_dynamic
                                    else 'retailer_credit_retained_per_site'),
            'surcharge_retained': (mean('retailer_surcharge_retained_per_site')
                                   if v.startswith('cohort_') else np.nan),
            'gross_margin': mean('net_position_per_site'),
            'amber_subscription': mean('baseline_subscription'),
        })
    return pd.DataFrame(rows)


def _auto_fee_changes(discovered, args, state_mix) -> dict[str, float]:
    """Each Offer B variant's own subscription_change_vs_amber: the national
    weighted mean of the per-site retailer net cost, which does not depend
    on the subscription. Offer A and Offer C variants are not included."""
    frame, _ = _build_frame(discovered, args.variants, False, None, None)
    if frame.empty:
        return {}
    weights = _national_weights(frame, state_mix)
    changes = {}
    for v in sorted(frame['variant'].unique()):
        if not v.startswith('retailer_'):
            continue
        m = frame['variant'] == v
        cost = (frame.loc[m, 'retailer_net_cost']
                / frame.loc[m, 'n_sites']).to_numpy()
        changes[v] = float(np.average(cost, weights=weights[m].to_numpy()))
    return changes


def _aggregate_group(discovered, args, fee_changes, state_mix,
                     out_dir: Path, dynamic_rows=None) -> pd.DataFrame | None:
    """Aggregate one threshold's variants and write that threshold's
    outputs. Returns its summary frame, or None if the group yielded
    nothing to aggregate. dynamic_rows (Offer C) enter the spread and grid
    charts and the summary only (spec section 20)."""
    if fee_changes == 'auto':
        fee_changes = _auto_fee_changes(discovered, args, state_mix)
        print("Subscription change (auto): "
              + ', '.join(f"{v} {c:+.2f}" for v, c in fee_changes.items()))
    frame, ratio_frame = _build_frame(discovered, args.variants,
                                      not args.no_amber, fee_changes,
                                      dynamic_rows)
    if frame.empty:
        print(f"{out_dir.name}: nothing to aggregate after filtering.",
              file=sys.stderr)
        return None

    # Every chart states the threshold: with more than one cap level on
    # disk, a figure that names only the battery is ambiguous.
    x = _variant_threshold(next(v for v in frame['variant'].unique()
                                if v != _AMBER and v != args.dynamic))
    cap_label = f'cap ${x:g}/MWh' if x is not None else 'cap unknown'

    # Every variant in the group shares one threshold, so the label need
    # not carry it: the directory name does.
    labels_by_variant = {v: (_AMBER if v == _AMBER
                             else _variant_label(v, False))
                         for v in frame['variant'].unique()}

    if state_mix is not None:
        present = sorted(frame['state'].unique())
        missing = [st for st in present if st not in state_mix]
        for st in missing:
            print(f"WARNING: no state_mix entry for {st}; excluded from the "
                 f"national pool.", file=sys.stderr)
        if all(st in missing for st in present):
            print(f"ERROR: {out_dir.name}: no state present has a state_mix "
                 f"entry, so the national pool would be empty.",
                 file=sys.stderr)
            return None

    weights = _national_weights(frame, state_mix)
    weight_label = ('National, weighted by state' if state_mix is not None
                    else 'National, unweighted')
    ordered = _ordered_variants(frame)
    labels = [labels_by_variant[v] for v in ordered]

    # The composition, subscription and cover-ratio charts are built on the
    # section 14 identity and the cap account, which Offer C does not have.
    is_cap = frame['variant'] != args.dynamic
    frame_cap, weights_cap = frame[is_cap], weights[is_cap]
    labels_cap = [labels_by_variant[v] for v in _ordered_variants(frame_cap)]

    out_dir.mkdir(parents=True, exist_ok=True)
    # The national weight is saved with each cohort-month so a figure drawn
    # from this file weights as the charts here do.
    frame.assign(weight=weights).to_csv(
        out_dir / 'retailer_cohort_months.csv', index=False)

    summary_rows = []
    subscription = float(frame['subscription_per_month'].iloc[0])
    for v in ordered:
        m = frame['variant'] == v
        w = weights[m].to_numpy()
        net = frame.loc[m, 'net_position_per_site'].to_numpy()
        cost = (frame.loc[m, 'retailer_net_cost']
                / frame.loc[m, 'n_sites']).to_numpy()
        row = {
            'variant': v,
            'label': labels_by_variant[v],
            'n_cohort_months': int(m.sum()),
            'mean_net_position_per_site_month': np.average(net, weights=w),
            'p5_net_position': _weighted_percentile(net, w, 5.0),
            'p95_net_position': _weighted_percentile(net, w, 95.0),
            'worst_month_net_position': net.min(),
            # The fee leaving the retailer as well off as under uncapped
            # passthrough. NOT the fee at which the mechanism alone breaks
            # even: the model sees only the wholesale layer, so that figure
            # would imply the subscription could fall to nothing.
            'amber_equivalent_subscription_per_month': (
                subscription + np.average(cost, weights=w)),
            'subscription_change_vs_amber': np.average(cost, weights=w),
            'months_retailer_lost_money': int((cost > _MATERIAL_LOSS).sum()),
            'subscription_change_used': fee_changes.get(v, 0.0),
        }
        if v == args.dynamic:
            # No subscription is being solved for under Offer C, and
            # "lost money" is measured against the Amber position rather
            # than zero (spec section 20).
            row['amber_equivalent_subscription_per_month'] = np.nan
            row['subscription_change_vs_amber'] = np.nan
            row['months_retailer_lost_money'] = int(
                (net < subscription - _MATERIAL_LOSS).sum())
        for comp in _COMPONENTS:
            row[comp + '_per_site_month'] = np.average(
                frame.loc[m, comp + '_per_site'].to_numpy(), weights=w)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / 'retailer_summary.csv', index=False)
    _composition_means(frame, weights).to_csv(
        out_dir / 'retailer_composition_means.csv', index=False)

    worded = [_worded(l) for l in labels]
    _plot_composition(frame_cap, weights_cap, [_worded(l) for l in labels_cap],
                      out_dir / 'retailer_composition.png',
                      args.battery_name, cap_label, weight_label)
    _plot_spread(frame, weights, worded,
                 out_dir / 'retailer_spread.png',
                 args.battery_name, cap_label, weight_label)
    _plot_subscription(frame_cap, weights_cap, labels_cap,
                       out_dir / 'retailer_subscription.png',
                       args.battery_name, cap_label, weight_label)
    _plot_cover_ratio(ratio_frame, _national_weights(ratio_frame, state_mix),
                      frame_cap, weights_cap, labels_cap, cap_label,
                      out_dir / 'retailer_cover_ratio.png',
                      args.battery_name)
    _plot_grid(frame, labels_by_variant,
               out_dir / 'retailer_position_grid.png',
               args.battery_name, cap_label)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    discovered = _discover(args.results_root, args.battery_name)
    if not discovered:
        print(f"No {{state}}_{{year}}_{args.battery_name}/{{variant}}/"
             f"{_SUMMARY_NAME} found under {args.results_root}. "
             f"--results-root should point at one run's output directory.",
             file=sys.stderr)
        return 1

    found_variants = sorted({v for _, _, v, _ in discovered})
    print(f"Found {len(discovered)} variant-scenario(s) across "
         f"{len(found_variants)} variant(s): {found_variants}")

    if args.threshold_per_mwh is not None:
        wanted = {f'x{x:g}' for x in args.threshold_per_mwh}
        discovered = [d for d in discovered
                      if _threshold_group(d[2]) in wanted]
        if not discovered:
            print(f"No variants at threshold(s) {sorted(wanted)} under "
                 f"{args.results_root}.", file=sys.stderr)
            return 1
        found_variants = sorted({v for _, _, v, _ in discovered})

    if args.subscription_change and 'auto' in args.subscription_change:
        if args.subscription_change != ['auto']:
            print("ERROR: --subscription-change auto cannot be combined "
                  "with VARIANT=CHANGE values.", file=sys.stderr)
            return 1
        fee_changes = 'auto'
    else:
        fee_changes = _parse_subscription_changes(args.subscription_change)
    if fee_changes != 'auto' and args.dynamic in fee_changes:
        print(f"ERROR: --subscription-change does not apply to "
              f"{args.dynamic}: Offer C has no subscription being solved for "
              f"(cap_account_spec.md Section 20).", file=sys.stderr)
        return 1
    unknown = (set() if fee_changes == 'auto'
               else set(fee_changes) - set(found_variants))
    if unknown:
        print(f"WARNING: --subscription-change names variant(s) not found "
             f"on disk: {sorted(unknown)}", file=sys.stderr)

    state_mix = None
    if not args.unweighted:
        # Resolved relative to cwd, not to any data root: the weights config
        # lives in the code repo, matching financial_analysis.py and
        # cap_account_aggregate.py.
        if not args.weights_file.exists():
            print(f"ERROR: weighted mode needs a weights file, none at "
                 f"{args.weights_file}. Run from the repo root, or pass "
                 f"--unweighted to combine states by plain concatenation "
                 f"instead.", file=sys.stderr)
            return 1
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import financial_analysis as fa
        state_mix = fa.load_weights_config(args.weights_file)['state_mix']

    dynamic_rows = None
    if args.dynamic is not None:
        dyn_found = _discover_dynamic(args.results_root, args.battery_name,
                                      args.dynamic)
        if not dyn_found:
            print(f"ERROR: --dynamic {args.dynamic}: no "
                  f"{{state}}_{{year}}_{args.battery_name}/{args.dynamic}/"
                  f"{_DYNAMIC_SUMMARY_NAME} under {args.results_root}.",
                  file=sys.stderr)
            return 1
        print(f"Offer C: {len(dyn_found)} scenario(s) of {args.dynamic}")
        dynamic_rows = _dynamic_rows(dyn_found, args.dynamic)

    groups: dict[str, list] = {}
    for entry in discovered:
        groups.setdefault(_threshold_group(entry[2]), []).append(entry)
    print(f"Cap threshold group(s): "
          f"{sorted(groups, key=_group_sort_key)}")

    written = 0
    for group in sorted(groups, key=_group_sort_key):
        out_dir = args.output_dir / args.battery_name / group
        print(f"\n=== {args.battery_name}, {group} ===")
        summary = _aggregate_group(groups[group], args, fee_changes,
                                   state_mix, out_dir, dynamic_rows)
        if summary is None:
            continue
        written += 1
        print(summary[['label', 'mean_net_position_per_site_month',
                       'worst_month_net_position',
                       'amber_equivalent_subscription_per_month',
                       'subscription_change_vs_amber',
                       'months_retailer_lost_money']]
              .round(3).to_string(index=False))
        print(f"Output written to {out_dir}/")

    if not written:
        print("Nothing aggregated.", file=sys.stderr)
        return 1

    if dynamic_rows:
        dyn_frame = pd.concat(dynamic_rows).reset_index()
        dyn_frame['baseline_subscription'] = dyn_frame['subscription_per_month']
        dyn_frame['net_position_per_site'] = (
            dyn_frame['subscription_per_month']
            - dyn_frame['retailer_net_cost'] / dyn_frame['n_sites'])
        out_dir = args.output_dir / args.battery_name / 'dynamic'
        _plot_dynamic_composition(dyn_frame, state_mix, out_dir,
                                  args.battery_name, args.dynamic)
        print(f"Offer C composition written to {out_dir}/")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
