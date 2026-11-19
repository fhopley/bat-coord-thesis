"""Daily value attribution: where does the actual-vs-LP value gap come from?

Given an EMAnalysis built by
viz_energy_masters.load_and_analyze(simulate_lp=True), decomposes the bill
difference between actual / self-consumption / perfect-foresight by day
and by economic mechanism.

The decomposition is an identity, not a heuristic: daily fixed charges are
identical across scenarios and cancel, so the whole difference is
volumetric and splits exactly per interval. The code asserts that
additivity rather than assuming it.

Usage (from project root):
    python3 -m energy_masters_pipeline.scripts.analyse_daily_value_gaps EM-427
    python3 -m energy_masters_pipeline.scripts.analyse_daily_value_gaps EM-427 --top-days 20
    python3 -m energy_masters_pipeline.scripts.analyse_daily_value_gaps --synthetic   # self-test

TESTING STATUS
--------------
The arithmetic core (per-interval gap decomposition, day grouping, bucket
assignment, additivity, curtailment valuation) IS verified, by the
synthetic self-test in __main__ which builds hand-computable inputs and
checks the results against independently calculated expected values.

Whether p_imp/p_exp cover every billed layer is checked at runtime by the
reconciliation check rather than assumed.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ADDITIVITY_TOLERANCE = 1e-6
"""Per-day and per-bucket sums must reproduce the total gap to this
tolerance. Tighter than a 'roughly agrees' check on purpose: these are
algebraic identities, so anything above floating-point noise is a code
bug, not a data property."""

RECONCILIATION_TOLERANCE_DOLLARS = 0.01
"""Volumetric gap vs household_total gap. Looser than the additivity
tolerance because this one is NOT an identity we control -- it depends on
whether p_imp/p_exp cover every billed layer, which is the thing being
tested."""

DEFAULT_HIGH_PRICE_PERCENTILE = 90.0

VCR_DENOMINATOR_FLOOR_DOLLARS = 0.10
"""Per-day VCR is only computed where (cost_sc - cost_lp) exceeds this.
Below it the ratio is numerically unstable and economically meaningless
(a day with essentially no arbitrage opportunity), so the column is left
empty rather than filled with an impressive-looking artifact."""

BUCKET_NAMES = [
    'both_prices_negative',
    'import_price_negative',
    'export_price_negative',
    'high_export_price',
    'high_import_price',
    'normal',
]


@dataclass
class DailyAttribution:
    site_id: str
    daily: pd.DataFrame
    buckets: pd.DataFrame
    totals: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)


def _volumetric_cost(import_kwh: pd.Series, export_kwh: pd.Series,
                     p_imp: pd.Series, p_exp: pd.Series) -> pd.Series:
    """Per-interval volumetric cost. Positive = money out.

    Sign convention follows the price vectors as built by
    build_price_vectors and plotted by viz_energy_masters: p_imp is what
    you pay per kWh imported, p_exp is what you RECEIVE per kWh exported
    (so p_exp < 0 means you pay to export, during negative wholesale).
    """
    return import_kwh * p_imp - export_kwh * p_exp


def _assign_buckets(p_imp: pd.Series, p_exp: pd.Series,
                    high_imp_threshold: float, high_exp_threshold: float) -> pd.Series:
    """Assign every interval to exactly one mechanism bucket.

    Order matters and is deliberate: negative-price regimes are tested
    first because they are the economically distinctive case, and the
    high-price buckets are defined to exclude anything already claimed by
    a negative bucket. Exhaustiveness and mutual exclusivity are asserted
    by the caller, not trusted to this ordering being right by eye.
    """
    imp_neg = p_imp < 0
    exp_neg = p_exp < 0

    bucket = pd.Series('normal', index=p_imp.index, dtype=object)
    bucket[imp_neg & exp_neg] = 'both_prices_negative'
    bucket[imp_neg & ~exp_neg] = 'import_price_negative'
    bucket[~imp_neg & exp_neg] = 'export_price_negative'

    neither_neg = ~imp_neg & ~exp_neg
    high_exp = neither_neg & (p_exp >= high_exp_threshold)
    bucket[high_exp] = 'high_export_price'
    high_imp = neither_neg & ~high_exp & (p_imp >= high_imp_threshold)
    bucket[high_imp] = 'high_import_price'
    return bucket


def analyse_daily_value_gaps(
    site_id: str,
    p_imp: pd.Series,
    p_exp: pd.Series,
    actual_import: pd.Series, actual_export: pd.Series,
    sc_import: pd.Series, sc_export: pd.Series,
    lp_import: pd.Series, lp_export: pd.Series,
    lp_curtail: pd.Series,
    bill_gap_lp_vs_actual: Optional[float] = None,
    bill_gap_actual_vs_sc: Optional[float] = None,
    high_price_percentile: float = DEFAULT_HIGH_PRICE_PERCENTILE,
) -> DailyAttribution:
    """Decompose the value gaps by day and by mechanism.

    Takes bare series rather than an EMAnalysis so it can be unit-tested
    without constructing the full analysis object (which needs the whole
    project's data layer). The CLI wrapper does the unpacking.

    bill_gap_* are the corresponding differences of household_total from
    compute_bill, used only for the reconciliation report. Pass None to
    skip that check.
    """
    for name, s in [('p_exp', p_exp), ('actual_import', actual_import),
                    ('sc_import', sc_import), ('lp_import', lp_import)]:
        if not s.index.equals(p_imp.index):
            raise ValueError(f"{site_id}: {name} index does not match p_imp index")

    cost_actual = _volumetric_cost(actual_import, actual_export, p_imp, p_exp)
    cost_sc = _volumetric_cost(sc_import, sc_export, p_imp, p_exp)
    cost_lp = _volumetric_cost(lp_import, lp_export, p_imp, p_exp)

    # Per-interval gaps. These are exact: summing any subset reproduces
    # that subset's contribution to the total difference.
    gap_lp_vs_actual_t = cost_actual - cost_lp
    gap_actual_vs_sc_t = cost_sc - cost_actual
    gap_lp_vs_sc_t = cost_sc - cost_lp

    # Curtailment value: what the curtailed energy would have been worth
    # (or cost) had it been exported instead, holding all else fixed.
    # Positive when curtailing during negative export prices.
    curtail_value_t = -lp_curtail * p_exp

    day = p_imp.index.normalize()

    frame = pd.DataFrame({
        'day': day,
        'cost_actual': cost_actual.to_numpy(),
        'cost_sc': cost_sc.to_numpy(),
        'cost_lp': cost_lp.to_numpy(),
        'gap_lp_vs_actual': gap_lp_vs_actual_t.to_numpy(),
        'gap_actual_vs_sc': gap_actual_vs_sc_t.to_numpy(),
        'gap_lp_vs_sc': gap_lp_vs_sc_t.to_numpy(),
        'curtail_kwh_lp': lp_curtail.to_numpy(),
        'curtail_value_lp': curtail_value_t.to_numpy(),
        'import_actual': actual_import.to_numpy(),
        'export_actual': actual_export.to_numpy(),
        'import_sc': sc_import.to_numpy(),
        'export_sc': sc_export.to_numpy(),
        'import_lp': lp_import.to_numpy(),
        'export_lp': lp_export.to_numpy(),
    })

    daily = frame.groupby('day').sum(numeric_only=True)
    daily['n_intervals'] = frame.groupby('day').size()

    # Per-day VCR only where the denominator is materially positive: a raw
    # per-day ratio of two small numbers is meaningless.
    denom = daily['gap_lp_vs_sc']
    numer = daily['gap_actual_vs_sc']
    daily['vcr_day'] = np.where(denom > VCR_DENOMINATOR_FLOOR_DOLLARS, numer / denom, np.nan)

    # --- Mechanism buckets ---
    high_imp_threshold = float(np.percentile(p_imp.to_numpy(), high_price_percentile))
    high_exp_threshold = float(np.percentile(p_exp.to_numpy(), high_price_percentile))
    bucket = _assign_buckets(p_imp, p_exp, high_imp_threshold, high_exp_threshold)

    total_gap_lp_vs_actual = float(gap_lp_vs_actual_t.sum())

    bucket_frame = pd.DataFrame({
        'bucket': bucket.to_numpy(),
        'gap_lp_vs_actual': gap_lp_vs_actual_t.to_numpy(),
        'gap_actual_vs_sc': gap_actual_vs_sc_t.to_numpy(),
        'curtail_kwh_lp': lp_curtail.to_numpy(),
        'curtail_value_lp': curtail_value_t.to_numpy(),
    })
    buckets = bucket_frame.groupby('bucket').sum(numeric_only=True)
    buckets['n_intervals'] = bucket_frame.groupby('bucket').size()
    buckets = buckets.reindex(BUCKET_NAMES).fillna(0.0)
    buckets['n_intervals'] = buckets['n_intervals'].astype(int)
    buckets['share_of_gap_pct'] = np.where(
        abs(total_gap_lp_vs_actual) > 1e-12,
        100 * buckets['gap_lp_vs_actual'] / total_gap_lp_vs_actual,
        np.nan,
    )

    # --- Verification: these are identities, so assert rather than hope ---
    T = len(p_imp)
    n_bucket_intervals = int(buckets['n_intervals'].sum())
    if n_bucket_intervals != T:
        raise AssertionError(
            f"{site_id}: bucket assignment is not exhaustive/exclusive -- "
            f"{n_bucket_intervals} bucketed intervals vs {T} total."
        )
    daily_sum = float(daily['gap_lp_vs_actual'].sum())
    if abs(daily_sum - total_gap_lp_vs_actual) > ADDITIVITY_TOLERANCE:
        raise AssertionError(
            f"{site_id}: per-day gaps sum to {daily_sum:.9f} but total is "
            f"{total_gap_lp_vs_actual:.9f} -- day grouping lost or double-counted intervals."
        )
    bucket_sum = float(buckets['gap_lp_vs_actual'].sum())
    if abs(bucket_sum - total_gap_lp_vs_actual) > ADDITIVITY_TOLERANCE:
        raise AssertionError(
            f"{site_id}: per-bucket gaps sum to {bucket_sum:.9f} but total is "
            f"{total_gap_lp_vs_actual:.9f} -- bucket assignment lost or double-counted intervals."
        )

    total_curtail_value = float(curtail_value_t.sum())
    total_curtail_kwh = float(lp_curtail.sum())
    curtail_at_negative_price_kwh = float(lp_curtail[p_exp < 0].sum())

    totals = {
        'total_gap_lp_vs_actual': total_gap_lp_vs_actual,
        'total_gap_actual_vs_sc': float(gap_actual_vs_sc_t.sum()),
        'total_gap_lp_vs_sc': float(gap_lp_vs_sc_t.sum()),
        'total_curtail_kwh': total_curtail_kwh,
        'curtail_at_negative_price_kwh': curtail_at_negative_price_kwh,
        'total_curtail_value': total_curtail_value,
        'curtail_value_pct_of_gap': (
            100 * total_curtail_value / total_gap_lp_vs_actual
            if abs(total_gap_lp_vs_actual) > 1e-12 else float('nan')
        ),
        'n_intervals': T,
        'n_days': int(daily.shape[0]),
    }

    # Aggregate VCR from summed numerator/denominator -- never a mean of
    # per-day ratios (see spec).
    if abs(totals['total_gap_lp_vs_sc']) > 1e-12:
        totals['vcr_volumetric'] = totals['total_gap_actual_vs_sc'] / totals['total_gap_lp_vs_sc']
    else:
        totals['vcr_volumetric'] = float('nan')

    # --- Reconciliation against the real bills ---
    if bill_gap_lp_vs_actual is not None:
        diff = total_gap_lp_vs_actual - bill_gap_lp_vs_actual
        totals['bill_gap_lp_vs_actual'] = bill_gap_lp_vs_actual
        totals['reconciliation_diff_lp_vs_actual'] = diff
        totals['reconciled_lp_vs_actual'] = bool(abs(diff) <= RECONCILIATION_TOLERANCE_DOLLARS)
    if bill_gap_actual_vs_sc is not None:
        diff = totals['total_gap_actual_vs_sc'] - bill_gap_actual_vs_sc
        totals['bill_gap_actual_vs_sc'] = bill_gap_actual_vs_sc
        totals['reconciliation_diff_actual_vs_sc'] = diff
        totals['reconciled_actual_vs_sc'] = bool(abs(diff) <= RECONCILIATION_TOLERANCE_DOLLARS)

    thresholds = {
        'high_price_percentile': high_price_percentile,
        'high_import_price_threshold_per_kwh': high_imp_threshold,
        'high_export_price_threshold_per_kwh': high_exp_threshold,
    }

    return DailyAttribution(site_id=site_id, daily=daily, buckets=buckets,
                            totals=totals, thresholds=thresholds)


def _print_concentration_and_robustness(attr: DailyAttribution) -> None:
    """How concentrated is the value, and how fragile is the headline VCR?

    Added after the first real EM-427 run showed extreme concentration: a
    single day carried 33% of all headroom and 55% of everything actual
    captured. When value is that concentrated, a period-level VCR is
    really a statement about a handful of days, and reporting it without
    this context invites over-reading it as a stable property of the
    coordination scheme.
    """
    d = attr.daily
    n_days = len(d)
    if n_days < 3:
        return

    num_total = d['gap_actual_vs_sc'].sum()
    den_total = d['gap_lp_vs_sc'].sum()
    if abs(den_total) < 1e-9:
        return

    print("--- Concentration ---")
    print(f"  {'':22s} {'top 1':>9s} {'top 5':>9s} {'top 10':>9s} {'top 20':>9s}")
    for col, label, total in [
        ('gap_lp_vs_sc', 'total headroom', den_total),
        ('gap_actual_vs_sc', 'value actual captured', num_total),
        ('gap_lp_vs_actual', 'LP-vs-actual gap', d['gap_lp_vs_actual'].sum()),
    ]:
        if abs(total) < 1e-9:
            continue
        s = d[col].sort_values(ascending=False)
        shares = [100 * s.head(n).sum() / total for n in (1, 5, 10, 20)]
        print(f"  {label:22s} " + "".join(f"{v:>8.1f}%" for v in shares))
    print(f"  (share of the {n_days} days: top 1 = {100/n_days:.1f}%, "
          f"top 20 = {100*20/n_days:.1f}%)")
    print()

    print("--- VCR by opportunity size ---")
    print("  VCR is not a uniform property: it is worth knowing whether the coordination")
    print("  captures value evenly, or only when the opportunity is large.")
    for label, n_sub, headroom, captured, vcr_tier in vcr_by_opportunity(d):
        print(f"  {label:22s} n={n_sub:4d}  headroom ${headroom:9,.2f}  "
              f"captured ${captured:9,.2f}  VCR {vcr_tier:7.3f}")
    print()

    n_destroying, destroyed, created = days_worse_than_sc(d)
    print("--- Days where actual did WORSE than self-consumption ---")
    print(f"  {n_destroying} of {n_days} days ({100*n_destroying/n_days:.1f}%), "
          f"costing ${-destroyed:,.2f}")
    print(f"  offset by ${created:,.2f} created on the other {n_days - n_destroying} days "
          f"(net ${num_total:,.2f})")
    print()

    # Leave-one-out: how much does the headline VCR depend on any single day?
    loo_stats = leave_one_out_vcr(d)
    if loo_stats is not None:
        loo_min, loo_max, worst_day = loo_stats
        vcr = num_total / den_total
        print("--- Robustness of the headline VCR ---")
        print(f"  headline VCR: {vcr:.4f}")
        print(f"  leave-one-day-out range: {loo_min:.4f} to {loo_max:.4f}")
        print(f"    most influential day: {pd.Timestamp(worst_day).strftime('%Y-%m-%d')} "
              f"-- dropping it moves VCR to {loo_min:.4f} ({loo_min - vcr:+.4f})")
        s2 = d.sort_values('gap_lp_vs_sc', ascending=False)
        drops = []
        for k in (1, 3, 5):
            sub = s2.iloc[k:]
            dd = sub['gap_lp_vs_sc'].sum()
            if abs(dd) > 1e-9:
                drops.append(f"drop top {k}: {sub['gap_actual_vs_sc'].sum() / dd:.4f}")
        if drops:
            print("  excluding the largest-opportunity days: " + ";  ".join(drops))
        if abs(loo.min() - vcr) > 0.05:
            print("  NOTE: a single day moves VCR by more than 0.05. Treat the headline")
            print("  figure as heavily dependent on whether particular extreme price events")
            print("  fall inside the analysis window, not as a stable property of the")
            print("  coordination scheme. Worth reporting alongside the range, not alone.")
        print()

    # Monthly view: the same concentration usually shows up seasonally.
    monthly = d.copy()
    monthly['month'] = pd.DatetimeIndex(monthly.index).strftime('%Y-%m')
    mg = monthly.groupby('month').agg(
        headroom=('gap_lp_vs_sc', 'sum'),
        captured=('gap_actual_vs_sc', 'sum'),
        gap=('gap_lp_vs_actual', 'sum'),
        days=('n_intervals', 'size'),
    )
    mg['vcr'] = np.where(mg['headroom'].abs() > 1e-9, mg['captured'] / mg['headroom'], np.nan)
    print("--- Monthly ---")
    print(f"  {'month':9s} {'days':>5s} {'headroom $':>11s} {'captured $':>11s} "
          f"{'gap $':>9s} {'VCR':>7s}")
    for month, row in mg.iterrows():
        print(f"  {month:9s} {int(row['days']):>5d} {row['headroom']:>11,.2f} "
              f"{row['captured']:>11,.2f} {row['gap']:>9,.2f} {row['vcr']:>7.3f}")
    print()


OPPORTUNITY_TIERS = ((10, 'top 10 headroom days'), (40, 'next 40 days'), (None, 'remaining days'))


def vcr_by_opportunity(daily: pd.DataFrame) -> list:
    """VCR split by how large the day's opportunity was.

    VCR is not a uniform property: a coordinator can capture well when the
    prize is large and almost nothing otherwise, and the split is the whole
    story about what drives a site to the top or bottom of the range.

    Returns [(label, n_days, headroom, captured, vcr), ...]. Shared by
    print_attribution and the cohort summary so the two cannot diverge.
    """
    s = daily.sort_values('gap_lp_vs_sc', ascending=False)
    out, start = [], 0
    for size, label in OPPORTUNITY_TIERS:
        sub = s.iloc[start:] if size is None else s.iloc[start:start + size]
        if size is not None:
            start += size
        if len(sub) == 0:
            continue
        captured = float(sub['gap_actual_vs_sc'].sum())
        headroom = float(sub['gap_lp_vs_sc'].sum())
        # Signed test, not abs(): a tier whose headroom is NEGATIVE (the LP did
        # worse than self-consumption across those days in aggregate, which the
        # terminal-SOC constraint makes possible on a subset of days) has no
        # meaningful VCR. Dividing two negatives yields a positive ratio that
        # looks entirely plausible and means nothing. Matches the signed test
        # already used for the per-day vcr_day above.
        vcr = captured / headroom if headroom > VCR_DENOMINATOR_FLOOR_DOLLARS else float('nan')
        out.append((label, len(sub), headroom, captured, vcr))
    return out


def leave_one_out_vcr(daily: pd.DataFrame):
    """(min, max, most_influential_day) for VCR with each single day dropped.

    Answers how much the headline depends on one extreme price event falling
    inside the analysis window, which for this cohort it very often does.
    Returns None when the range cannot be computed.
    """
    num_total = float(daily['gap_actual_vs_sc'].sum())
    den_total = float(daily['gap_lp_vs_sc'].sum())
    loo = (num_total - daily['gap_actual_vs_sc']) / (den_total - daily['gap_lp_vs_sc'])
    loo = loo.replace([np.inf, -np.inf], np.nan).dropna()
    if not len(loo):
        return None
    return float(loo.min()), float(loo.max()), loo.idxmin()


def days_worse_than_sc(daily: pd.DataFrame) -> tuple:
    """(n_days, value_destroyed, value_created) against self-consumption.

    Destroyed is returned NEGATIVE, as stored, so the two sum to the net.
    """
    worse = daily['gap_actual_vs_sc'] < 0
    return (
        int(worse.sum()),
        float(daily.loc[worse, 'gap_actual_vs_sc'].sum()),
        float(daily.loc[~worse, 'gap_actual_vs_sc'].sum()),
    )


def print_attribution(attr: DailyAttribution, top_days: int = 15) -> None:
    t = attr.totals
    print(f"=== {attr.site_id} daily value attribution ===")
    print(f"{t['n_intervals']:,} intervals across {t['n_days']} days")
    print()

    print("--- Reconciliation: volumetric decomposition vs actual bills ---")
    print("The decomposition below is volumetric only (daily fixed charges are identical")
    print("across scenarios and cancel). If these reconcile, the attribution accounts for")
    print("the WHOLE bill difference; if not, it covers only the reconciled portion.")
    if 'bill_gap_lp_vs_actual' in t:
        status = 'OK' if t['reconciled_lp_vs_actual'] else 'MISMATCH'
        print(f"  LP vs actual:  volumetric ${t['total_gap_lp_vs_actual']:,.2f}  "
              f"bill ${t['bill_gap_lp_vs_actual']:,.2f}  "
              f"diff ${t['reconciliation_diff_lp_vs_actual']:,.4f}  {status}")
    if 'bill_gap_actual_vs_sc' in t:
        status = 'OK' if t['reconciled_actual_vs_sc'] else 'MISMATCH'
        print(f"  actual vs SC:  volumetric ${t['total_gap_actual_vs_sc']:,.2f}  "
              f"bill ${t['bill_gap_actual_vs_sc']:,.2f}  "
              f"diff ${t['reconciliation_diff_actual_vs_sc']:,.4f}  {status}")
    if 'bill_gap_lp_vs_actual' in t and not t['reconciled_lp_vs_actual']:
        print("  NOTE: a mismatch means p_imp/p_exp do not cover every billed volumetric")
        print("  layer. Treat the per-day and per-bucket numbers below as covering only")
        print("  the part of the bill those price vectors represent.")
    print()

    print("--- Period totals ---")
    print(f"  gap LP vs actual:   ${t['total_gap_lp_vs_actual']:,.2f}  "
          "(what better dispatch was worth)")
    print(f"  gap actual vs SC:   ${t['total_gap_actual_vs_sc']:,.2f}  "
          "(what actual captured over self-consumption)")
    print(f"  gap LP vs SC:       ${t['total_gap_lp_vs_sc']:,.2f}  (total headroom)")
    print(f"  VCR (volumetric):   {t['vcr_volumetric']:.4f}")
    print()

    print("--- Curtailment ---")
    print(f"  curtailed:                 {t['total_curtail_kwh']:,.1f} kWh")
    print(f"  of which at negative p_exp: {t['curtail_at_negative_price_kwh']:,.1f} kWh "
          f"({100 * t['curtail_at_negative_price_kwh'] / t['total_curtail_kwh']:.1f}%)"
          if t['total_curtail_kwh'] > 0 else "  (no curtailment)")
    print(f"  curtailment value:         ${t['total_curtail_value']:,.2f}  "
          f"({t['curtail_value_pct_of_gap']:.1f}% of the LP-vs-actual gap)")
    print("  Meaning: what the curtailed energy would have cost/earned had it been exported")
    print("  instead, holding all other flows fixed. This is a decomposition term, NOT the")
    print("  answer to 'how much worse would the LP do if curtailment were banned' -- that")
    print("  needs a re-solve with the constraint removed, since the LP would then change")
    print("  its charging and storage decisions everywhere else too.")
    print()

    print("--- Mechanism buckets (exhaustive, mutually exclusive) ---")
    th = attr.thresholds
    print(f"  high-price threshold: {th['high_price_percentile']:.0f}th percentile  "
          f"(import >= ${th['high_import_price_threshold_per_kwh']:.4f}/kWh, "
          f"export >= ${th['high_export_price_threshold_per_kwh']:.4f}/kWh)")
    print()
    b = attr.buckets
    print(f"  {'bucket':24s} {'intervals':>10s} {'gap $':>12s} {'share %':>9s} "
          f"{'curtail kWh':>12s}")
    for name in BUCKET_NAMES:
        row = b.loc[name]
        print(f"  {name:24s} {int(row['n_intervals']):>10,d} "
              f"{row['gap_lp_vs_actual']:>12,.2f} {row['share_of_gap_pct']:>9.1f} "
              f"{row['curtail_kwh_lp']:>12,.1f}")
    print()

    _print_concentration_and_robustness(attr)

    print(f"--- Top {top_days} days by LP-vs-actual gap ---")
    print("(check n_intervals: a partial day at the period boundary is not a real outlier)")
    top = attr.daily.sort_values('gap_lp_vs_actual', ascending=False).head(top_days)
    print(f"  {'day':12s} {'gap $':>10s} {'act-sc $':>10s} {'headroom $':>11s} "
          f"{'curtail kWh':>12s} {'curt $':>9s} {'n':>6s}")
    for day, row in top.iterrows():
        day_str = pd.Timestamp(day).strftime('%Y-%m-%d')
        print(f"  {day_str:12s} {row['gap_lp_vs_actual']:>10,.2f} "
              f"{row['gap_actual_vs_sc']:>10,.2f} {row['gap_lp_vs_sc']:>11,.2f} "
              f"{row['curtail_kwh_lp']:>12,.1f} {row['curtail_value_lp']:>9,.2f} "
              f"{int(row['n_intervals']):>6d}")
    print()
    print("  Investigate these days with:")
    for day, _ in top.head(5).iterrows():
        day_str = pd.Timestamp(day).strftime('%Y-%m-%d')
        print(f"    python3 -m energy_masters_pipeline.scripts.viz_energy_masters {attr.site_id} --compare-day {day_str}")
    print()


def attribution_from_analysis(analysis, high_price_percentile: float = DEFAULT_HIGH_PRICE_PERCENTILE):
    """Unpack an EMAnalysis (with simulate_lp=True) into the bare series
    analyse_daily_value_gaps wants."""
    if analysis.lp_result is None:
        raise ValueError(
            f"{analysis.site_id}: EMAnalysis has no lp_result -- build it with "
            "load_and_analyze(..., simulate_lp=True)."
        )
    if analysis.sc_result is None:
        raise ValueError(f"{analysis.site_id}: EMAnalysis has no sc_result.")

    bill_gap_lp_vs_actual = (analysis.actual_bill.household_total
                             - analysis.lp_bill.household_total)
    bill_gap_actual_vs_sc = (analysis.sc_bill.household_total
                             - analysis.actual_bill.household_total)

    return analyse_daily_value_gaps(
        site_id=analysis.site_id,
        p_imp=analysis.p_imp, p_exp=analysis.p_exp,
        actual_import=analysis.actual_result.import_kwh,
        actual_export=analysis.actual_result.export_kwh,
        sc_import=analysis.sc_result.import_kwh,
        sc_export=analysis.sc_result.export_kwh,
        lp_import=analysis.lp_result.import_kwh,
        lp_export=analysis.lp_result.export_kwh,
        lp_curtail=analysis.lp_result.curtail_kwh,
        bill_gap_lp_vs_actual=bill_gap_lp_vs_actual,
        bill_gap_actual_vs_sc=bill_gap_actual_vs_sc,
        high_price_percentile=high_price_percentile,
    )


# Sentinel: None already means "pinned", so it cannot double as "unspecified".
_INHERIT_DEFAULT = '__inherit_default__'


def run_checks(site_id: str, top_days: int = 15,
               export_limit_kw: Optional[float] = None,
               curtail_cost_per_kwh_dollars=_INHERIT_DEFAULT,
               c_deg_per_kwh_override: Optional[float] = None) -> DailyAttribution:
    from energy_masters_pipeline.scripts.viz_energy_masters import (
        load_and_analyze, describe_curtailment,
        DEFAULT_CURTAIL_COST_PER_KWH_DOLLARS,
    )
    if curtail_cost_per_kwh_dollars is _INHERIT_DEFAULT:
        curtail_cost_per_kwh_dollars = DEFAULT_CURTAIL_COST_PER_KWH_DOLLARS

    analysis = load_and_analyze(
        site_id, simulate_lp=True, export_limit_kw=export_limit_kw,
        curtail_cost_per_kwh_dollars=curtail_cost_per_kwh_dollars,
        c_deg_per_kwh_override=c_deg_per_kwh_override,
    )
    print(describe_curtailment(curtail_cost_per_kwh_dollars))
    print(f"c_deg_per_kwh used for LP: {analysis.battery_spec.c_deg_per_kwh:.4f} $/kWh")
    attr = attribution_from_analysis(analysis)
    print_attribution(attr, top_days=top_days)

    out_dir = Path('data/processed/energy_masters')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{site_id}_daily_value_attribution.csv'
    attr.daily.to_csv(out_path)
    print(f"Per-day detail written to {out_path.resolve()}")
    bucket_path = out_dir / f'{site_id}_value_buckets.csv'
    attr.buckets.to_csv(bucket_path)
    print(f"Bucket detail written to {bucket_path.resolve()}")
    return attr


# ---------------------------------------------------------------------------
# Synthetic self-test
# ---------------------------------------------------------------------------

def _synthetic_self_test() -> None:
    """Hand-computable case: verify the decomposition is exact, additive,
    and that the bucket/curtailment numbers match independent calculation.
    """
    print("=== Synthetic self-test ===")
    # Two days, 4 intervals each, chosen so every bucket is exercised and
    # every number can be checked by hand.
    idx = pd.to_datetime([
        '2025-01-01 00:00', '2025-01-01 06:00', '2025-01-01 12:00', '2025-01-01 18:00',
        '2025-01-02 00:00', '2025-01-02 06:00', '2025-01-02 12:00', '2025-01-02 18:00',
    ])
    #                        both-  imp-   exp-   normal | high  high   normal normal
    #                        neg    neg    neg            exp   imp
    p_imp = pd.Series([-0.10, -0.20, 0.30, 0.25, 0.40, 0.90, 0.30, 0.20], index=idx)
    p_exp = pd.Series([-0.05, 0.10, -0.15, 0.08, 0.85, 0.20, 0.09, 0.07], index=idx)

    actual_import = pd.Series([1.0, 0.0, 2.0, 1.0, 0.0, 3.0, 1.0, 1.0], index=idx)
    actual_export = pd.Series([0.0, 2.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0], index=idx)
    sc_import = pd.Series([1.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.0, 1.0], index=idx)
    sc_export = pd.Series([0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0], index=idx)
    lp_import = pd.Series([3.0, 2.0, 2.0, 1.0, 0.0, 0.0, 1.0, 1.0], index=idx)
    lp_export = pd.Series([0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0], index=idx)
    lp_curtail = pd.Series([0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0], index=idx)

    attr = analyse_daily_value_gaps(
        'SYNTH-001', p_imp, p_exp,
        actual_import, actual_export, sc_import, sc_export,
        lp_import, lp_export, lp_curtail,
    )

    # --- Independent hand calculation ---
    cost_actual = (actual_import * p_imp - actual_export * p_exp).sum()
    cost_lp = (lp_import * p_imp - lp_export * p_exp).sum()
    cost_sc = (sc_import * p_imp - sc_export * p_exp).sum()
    expected_gap = cost_actual - cost_lp
    got_gap = attr.totals['total_gap_lp_vs_actual']
    ok = abs(expected_gap - got_gap) < 1e-12
    print(f"total gap LP vs actual: expected {expected_gap:.6f}, got {got_gap:.6f}  "
          f"{'PASS' if ok else 'FAIL'}")

    # Curtailment: 3.0 kWh at p_exp = -0.15 -> value = -3.0 * -0.15 = +0.45
    expected_curtail_value = 0.45
    got_curtail_value = attr.totals['total_curtail_value']
    ok_c = abs(expected_curtail_value - got_curtail_value) < 1e-12
    print(f"curtail value: expected {expected_curtail_value:.4f}, got "
          f"{got_curtail_value:.4f}  {'PASS' if ok_c else 'FAIL'}")

    # Bucket assignment, by construction of the price series above.
    expected_buckets = {
        'both_prices_negative': 1,    # interval 0
        'import_price_negative': 1,   # interval 1
        'export_price_negative': 1,   # interval 2
        'high_export_price': 1,       # interval 4
        'high_import_price': 1,       # interval 5
        'normal': 3,                  # intervals 3, 6, 7
    }
    all_ok = True
    for name, expected_n in expected_buckets.items():
        got_n = int(attr.buckets.loc[name, 'n_intervals'])
        match = got_n == expected_n
        all_ok = all_ok and match
        print(f"  bucket {name:24s} expected {expected_n}, got {got_n}  "
              f"{'PASS' if match else 'FAIL'}")

    # Additivity (also asserted inside, but check explicitly here too).
    day_sum = attr.daily['gap_lp_vs_actual'].sum()
    bucket_sum = attr.buckets['gap_lp_vs_actual'].sum()
    print(f"additivity: days {day_sum:.9f}, buckets {bucket_sum:.9f}, total {got_gap:.9f}  "
          f"{'PASS' if abs(day_sum - got_gap) < 1e-9 and abs(bucket_sum - got_gap) < 1e-9 else 'FAIL'}")

    # Day grouping
    print(f"days found: {attr.totals['n_days']} (expected 2)  "
          f"{'PASS' if attr.totals['n_days'] == 2 else 'FAIL'}")

    # VCR floor behaviour: build a case where the denominator is tiny and
    # confirm vcr_day is NaN rather than an exploded ratio.
    tiny_lp_import = sc_import.copy()
    tiny_lp_export = sc_export.copy()
    attr2 = analyse_daily_value_gaps(
        'SYNTH-002', p_imp, p_exp,
        actual_import, actual_export, sc_import, sc_export,
        tiny_lp_import, tiny_lp_export, pd.Series(0.0, index=idx),
    )
    n_nan = int(attr2.daily['vcr_day'].isna().sum())
    print(f"VCR floor: {n_nan} of {attr2.totals['n_days']} days left NaN when LP==SC  "
          f"{'PASS' if n_nan == attr2.totals['n_days'] else 'FAIL'}")

    print()
    print("Bucket table rendering check:")
    print_attribution(attr, top_days=2)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--synthetic':
        _synthetic_self_test()
        sys.exit(0)
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('site_id', help='Site identifier, e.g. EM-427')
    parser.add_argument('--top-days', type=int, default=15)
    parser.add_argument('--export-limit-kw', type=float, default=None)
    from energy_masters_pipeline.scripts.viz_energy_masters import (
        resolve_curtailment_arg, CURTAILMENT_ARG_HELP, DEFAULT_C_DEG_PER_KWH,
    )
    parser.add_argument('--curtailment', type=str, default=None, metavar='MODE',
                        help=CURTAILMENT_ARG_HELP)
    parser.add_argument('--c-deg-per-kwh', type=float, default=None,
                        help='Override the LP battery degradation cost, in dollars '
                             '(not cents) per kWh of throughput. Default: '
                             f'{DEFAULT_C_DEG_PER_KWH}. See sweep_c_deg.py for '
                             'calibrating this value against actual battery-use metrics.')
    args = parser.parse_args()
    try:
        curtail_cost_per_kwh_dollars = resolve_curtailment_arg(args.curtailment)
    except ValueError as e:
        parser.error(str(e))
    run_checks(args.site_id, top_days=args.top_days, export_limit_kw=args.export_limit_kw,
               curtail_cost_per_kwh_dollars=curtail_cost_per_kwh_dollars,
               c_deg_per_kwh_override=args.c_deg_per_kwh)