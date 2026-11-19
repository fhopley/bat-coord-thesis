"""Run the value attribution across every accepted site in the cohort.

Loops the accepted/override sites from battery_capacity_review.csv,
runs load_and_analyze(simulate_lp=True) + analyse_daily_value_gaps on
each, and collects a one-row-per-site summary (VCR, bills, curtailment,
mechanism shares) plus the per-site CSVs.

Deliberately fail-soft per site: one site erroring out (bad data, LP
infeasible, unreviewed) should not lose the results for the other nine.
Failures are collected and reported at the end rather than aborting.

Usage (from project root):
    python3 -m energy_masters_pipeline.scripts.run_cohort_attribution
    python3 -m energy_masters_pipeline.scripts.run_cohort_attribution --sites EM-427,EM-414
    python3 -m energy_masters_pipeline.scripts.run_cohort_attribution --out cohort_summary.csv

Runtime note: each site does a full LP solve (~7s for EM-427 over ~7
months) plus ingestion, so a 10-site run is minutes, not seconds.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd

from energy_masters_pipeline.scripts.analyse_daily_value_gaps import (
    vcr_by_opportunity, leave_one_out_vcr, days_worse_than_sc,
)

# None already means "curtailment pinned", so it cannot double as "unspecified".
_INHERIT_DEFAULT = '__inherit_default__'

DEFAULT_REVIEW_CSV = Path('energy_masters_pipeline/battery_capacity_review.csv')

# Sites whose total headroom is below this are reported but excluded from the
# sensitivity figure: VCR is captured/headroom, so a site with nothing to
# capture produces a ratio that is noise, not a measure of coordination. Same
# small-denominator trap the per-day VCR floor already guards against.
SITE_HEADROOM_FLOOR_DOLLARS = 5.0


def _pct(numerator, denominator):
    return (100.0 * numerator / denominator) if denominator else float('nan')
DEFAULT_DATA_DIR = Path('data/processed/energy_masters')
DEFAULT_OUT_CSV = DEFAULT_DATA_DIR / 'cohort_value_attribution_summary.csv'


def accepted_sites(review_csv_path: Path = DEFAULT_REVIEW_CSV) -> list[str]:
    """Sites with decision 'accept' or 'override'.

    Excluded sites are skipped, and (importantly) so are sites absent
    from the file entirely -- capacity_review.py treats "not yet
    reviewed" as distinct from "accepted", and silently including an
    unreviewed site in a cohort-wide result would defeat the point of
    having the review step at all.
    """
    from energy_masters_pipeline.scripts import capacity_review as cr

    df = cr._load_review_df(review_csv_path)
    out = []
    for site_id in df['site_id'].tolist():
        review = cr.get_review(site_id, review_csv_path)
        if review is not None and review.decision in ('accept', 'override'):
            out.append(site_id)
    return sorted(out)


def _summary_row(site_id: str, analysis, attr) -> dict:
    t = attr.totals
    b = attr.buckets
    row = {
        'site_id': site_id,
        'capacity_kwh': analysis.battery_spec.capacity_kwh,
        'power_kw': analysis.battery_spec.power_kw,
        'charge_eff': analysis.battery_spec.charge_efficiency,
        'export_limit_kw': analysis.export_limit_kw,
        'n_days': t['n_days'],
        'n_intervals': t['n_intervals'],
        'bill_actual': analysis.actual_bill.household_total,
        'bill_sc': analysis.sc_bill.household_total,
        'bill_lp': analysis.lp_bill.household_total,
        'vcr_from_bills': (
            (analysis.sc_bill.household_total - analysis.actual_bill.household_total)
            / (analysis.sc_bill.household_total - analysis.lp_bill.household_total)
            if abs(analysis.sc_bill.household_total - analysis.lp_bill.household_total) > 1e-9
            else float('nan')
        ),
        'vcr_volumetric': t['vcr_volumetric'],
        'gap_lp_vs_actual': t['total_gap_lp_vs_actual'],
        'gap_actual_vs_sc': t['total_gap_actual_vs_sc'],
        'gap_lp_vs_sc': t['total_gap_lp_vs_sc'],
        'curtail_kwh': t['total_curtail_kwh'],
        'curtail_value': t['total_curtail_value'],
        'curtail_value_pct_of_gap': t['curtail_value_pct_of_gap'],
        'reconciled_lp_vs_actual': t.get('reconciled_lp_vs_actual'),
        'reconciliation_diff_lp_vs_actual': t.get('reconciliation_diff_lp_vs_actual'),
        # Settings actually in force, recorded per row: a summary CSV that
        # does not say which c_deg and curtailment mode produced it cannot be
        # compared against another run.
        'c_deg_per_kwh': analysis.battery_spec.c_deg_per_kwh,
        'curtail_mode': ('pinned' if analysis.curtail_cost_per_kwh_dollars is None
                         else 'free' if analysis.curtail_cost_per_kwh_dollars == 0.0
                         else f'{analysis.curtail_cost_per_kwh_dollars:g}/kWh'),
        'period_start': str(analysis.trace.index[0].date()),
        'period_end': str(analysis.trace.index[-1].date()),
        # Reconstruction-quality carry-through, so a site with a shaky
        # reconstruction is visible in the cohort table rather than only
        # in its own individual run.
        'unexplained_neg_load_intervals':
            analysis.diagnostics.sol_to_home_unexplained_negative_intervals,
        'soc_overflow_kwh': analysis.diagnostics.soc_overflow_kwh,
        'soc_underflow_kwh': analysis.diagnostics.soc_underflow_kwh,
        # Clipping RELATIVE to what was attempted. The absolute kWh above says
        # little without the denominator, and it is the relative figure that
        # flags a site whose actual bill rests on flows that were partly
        # clipped away -- a candidate explanation for a negative VCR that has
        # nothing to do with coordination quality.
        'soc_overflow_pct': _pct(analysis.diagnostics.soc_overflow_kwh,
                                 analysis.diagnostics.total_charge_attempted_dc_kwh),
        'soc_underflow_pct': _pct(analysis.diagnostics.soc_underflow_kwh,
                                  analysis.diagnostics.total_discharge_attempted_dc_kwh),
    }

    # VCR is not one number per site. These columns carry its SHAPE, which is
    # what distinguishes a site that captures value evenly from one that only
    # does so on spike days, and a robust headline from one resting on a
    # single day falling inside the window.
    for label, n_sub, headroom, captured, vcr_tier in vcr_by_opportunity(attr.daily):
        key = label.split()[0] + '_' + label.split()[1]   # 'top_10', 'next_40', 'remaining_days'
        row[f'vcr_{key}'] = vcr_tier
        row[f'headroom_{key}'] = headroom
    loo = leave_one_out_vcr(attr.daily)
    if loo is not None:
        row['vcr_loo_min'], row['vcr_loo_max'], worst = loo
        row['most_influential_day'] = str(pd.Timestamp(worst).date())
    n_worse, destroyed, created = days_worse_than_sc(attr.daily)
    row['n_days_worse_than_sc'] = n_worse
    row['value_destroyed'] = destroyed
    row['value_created'] = created
    for name in b.index:
        row[f'gap_share_{name}_pct'] = b.loc[name, 'share_of_gap_pct']
    return row


def run_cohort(
    sites: Optional[list[str]] = None,
    review_csv_path: Path = DEFAULT_REVIEW_CSV,
    out_csv: Path = DEFAULT_OUT_CSV,
    write_per_site_csvs: bool = True,
    curtail_cost_per_kwh_dollars=_INHERIT_DEFAULT,
    c_deg_per_kwh_override: Optional[float] = None,
) -> pd.DataFrame:
    """Run the value attribution across the cohort under ONE set of settings.

    No cohort-wide export_limit_kw: a single value could only ever be right
    for one site, and per-site limits now come from export_kw_override in the
    review CSV.
    """
    from energy_masters_pipeline.scripts.viz_energy_masters import (
        load_and_analyze, describe_curtailment,
        DEFAULT_CURTAIL_COST_PER_KWH_DOLLARS,
    )
    from energy_masters_pipeline.scripts.analyse_daily_value_gaps import attribution_from_analysis
    if curtail_cost_per_kwh_dollars is _INHERIT_DEFAULT:
        curtail_cost_per_kwh_dollars = DEFAULT_CURTAIL_COST_PER_KWH_DOLLARS

    if sites is None:
        sites = accepted_sites(review_csv_path)
    print(f"Running value attribution for {len(sites)} site(s): {', '.join(sites)}")
    print(describe_curtailment(curtail_cost_per_kwh_dollars))
    if c_deg_per_kwh_override is not None:
        print(f"c_deg_per_kwh override: {c_deg_per_kwh_override}")
    print()

    rows = []
    failures = []
    for i, site_id in enumerate(sites, 1):
        print(f"[{i}/{len(sites)}] {site_id} ...", flush=True)
        try:
            analysis = load_and_analyze(
                site_id, simulate_lp=True,
                curtail_cost_per_kwh_dollars=curtail_cost_per_kwh_dollars,
                c_deg_per_kwh_override=c_deg_per_kwh_override,
            )
            attr = attribution_from_analysis(analysis)
            rows.append(_summary_row(site_id, analysis, attr))
            if write_per_site_csvs:
                DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
                attr.daily.to_csv(
                    DEFAULT_DATA_DIR / f'{site_id}_daily_value_attribution.csv')
            print(f"    VCR {attr.totals['vcr_volumetric']:.3f}   "
                  f"gap ${attr.totals['total_gap_lp_vs_actual']:,.2f}   "
                  f"curtail {attr.totals['total_curtail_kwh']:,.0f} kWh "
                  f"(${attr.totals['total_curtail_value']:,.2f})")
        except Exception as e:  # fail-soft: one bad site must not lose the rest
            failures.append((site_id, e))
            print(f"    FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()

    if not rows:
        print("\nNo sites completed successfully.")
        return pd.DataFrame()

    summary = pd.DataFrame(rows).set_index('site_id')
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv)

    print()
    print("=== Cohort summary ===")
    cols = ['vcr_volumetric', 'gap_actual_vs_sc', 'gap_lp_vs_sc',
            'curtail_kwh', 'curtail_value_pct_of_gap']
    print(summary[cols].to_string(float_format=lambda v: f"{v:,.3f}"))
    print()

    # Cohort VCR from SUMMED numerator/denominator, never a mean of
    # per-site ratios -- same reasoning as the per-day case in
    # analyse_daily_value_gaps.py: a mean of ratios silently overweights
    # sites with small denominators.
    num = summary['gap_actual_vs_sc'].sum()
    den = summary['gap_lp_vs_sc'].sum()
    if abs(den) > 1e-9:
        print(f"Cohort VCR (summed numerator / summed denominator): {num / den:.4f}")
    else:
        print("Cohort VCR: undefined (total headroom across the cohort is ~zero)")
    print(f"  (for contrast, the MEAN of per-site VCRs is "
          f"{summary['vcr_volumetric'].mean():.4f} -- report the summed version, "
          "the mean overweights sites with little headroom)")
    print()

    # Sensitivity to low-headroom sites. A site whose window contained no
    # extreme price event has almost nothing to capture, so its VCR is a ratio
    # of two small numbers. Including such sites in a cohort figure imports
    # their noise; excluding them silently would overstate the cohort. Report
    # both, and name the sites.
    thin = summary[summary['gap_lp_vs_sc'] < SITE_HEADROOM_FLOOR_DOLLARS]
    if len(thin):
        thick = summary[summary['gap_lp_vs_sc'] >= SITE_HEADROOM_FLOOR_DOLLARS]
        den_thick = thick['gap_lp_vs_sc'].sum()
        print(f"{len(thin)} site(s) have total headroom below "
              f"${SITE_HEADROOM_FLOOR_DOLLARS:.2f}: {', '.join(thin.index)}")
        print("  Their VCR is a ratio of two small numbers and should not be read as "
              "a measure of coordination quality.")
        if abs(den_thick) > 1e-9:
            print(f"  Cohort VCR excluding them: "
                  f"{thick['gap_actual_vs_sc'].sum() / den_thick:.4f} "
                  f"({len(thick)} site(s))")
        print()

    shape_cols = [c for c in ('vcr_top_10', 'vcr_next_40', 'vcr_remaining_days',
                              'vcr_loo_min', 'vcr_volumetric', 'vcr_loo_max',
                              'n_days', 'n_days_worse_than_sc') if c in summary.columns]
    if shape_cols:
        print("=== VCR shape per site ===")
        print("Headline VCR sits between its leave-one-out bounds; the tier columns show")
        print("whether value is captured evenly or only when the opportunity is large.")
        print(summary[shape_cols].to_string(float_format=lambda v: f"{v:,.3f}"))
        print()

    unreconciled = summary[summary['reconciled_lp_vs_actual'] == False]  # noqa: E712
    if len(unreconciled):
        print(f"WARNING: {len(unreconciled)} site(s) failed the volumetric/bill "
              f"reconciliation: {', '.join(unreconciled.index)}")
        print("  Their attribution covers only part of the bill difference.")
        print()

    shaky = summary[summary['unexplained_neg_load_intervals'] > 0]
    if len(shaky):
        print(f"NOTE: {len(shaky)} site(s) have unexplained negative-load intervals "
              f"in their reconstruction: {', '.join(shaky.index)}")
        print("  Small volumes are unlikely to matter, but check the per-site output.")
        print()

    if failures:
        print(f"{len(failures)} site(s) FAILED and are absent from the summary:")
        for site_id, e in failures:
            print(f"  {site_id}: {type(e).__name__}: {e}")
        print()

    print(f"Summary written to {out_csv.resolve()}")
    return summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sites', type=str, default=None,
                        help='Comma-separated site list. Default: all accept/override '
                             'sites in battery_capacity_review.csv.')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument('--no-per-site-csv', action='store_true')
    from energy_masters_pipeline.scripts.viz_energy_masters import (
        resolve_curtailment_arg, CURTAILMENT_ARG_HELP, DEFAULT_C_DEG_PER_KWH,
    )
    parser.add_argument('--curtailment', type=str, default=None, metavar='MODE',
                        help=CURTAILMENT_ARG_HELP)
    parser.add_argument('--c-deg-per-kwh', type=float, default=None,
                        help='Override the LP battery degradation cost, dollars (not '
                             f'cents) per kWh of throughput. Default: {DEFAULT_C_DEG_PER_KWH}. '
                             'See sweep_c_deg.py for fitting this against actual '
                             'battery use across the cohort.')
    args = parser.parse_args()
    try:
        curtail_cost = resolve_curtailment_arg(args.curtailment)
    except ValueError as e:
        parser.error(str(e))

    sites = [s.strip() for s in args.sites.split(',')] if args.sites else None
    run_cohort(sites=sites, out_csv=args.out,
               write_per_site_csvs=not args.no_per_site_csv,
               curtail_cost_per_kwh_dollars=curtail_cost,
               c_deg_per_kwh_override=args.c_deg_per_kwh)