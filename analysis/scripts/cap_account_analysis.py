"""Capped-volatility retail offers: re-bill verified dispatch flows under a
wholesale price cap and settle the resulting cap account.

See cap_account_spec.md for the full design. Three offers (spec sections
7, 10 and Part V):

  --offer cohort     Offer A. The cap account is mutualised within the
                     cohort: uncollected costs are covered from unpaid
                     export credits, the retailer absorbs the remainder,
                     and a hedging surcharge on all imports transfers value
                     from importers to the exporters providing the hedge.
  --offer retailer   Offer B. The retailer absorbs the whole imbalance and
                     retains the surplus, less a fraction (--share-fraction)
                     shared back to each site on its own above-cap export
                     value. No cohort pooling, no surcharge.
  --offer dynamic    Offer C. Imports at the mixed plan's TOU rates, exports
                     credited its floored share of real-time export value;
                     compared with Amber on the full bill. No cap account.

Under Offers A and B dispatch is not changed: the retailer holds dispatch
rights and optimises against the real, uncapped price, so the run's
verified flows are the correct dispatch. Offer C takes its dispatch from
the plan's own modelled run (--offer-run-config, default
full_run_rolling_tesla), which optimises against the mixed-plan dispatch
prices, and compares it with the Amber run named by --run-config.
--dynamic-dispatch rebill instead bills the Amber run's flows under the
plan, measuring the plan's billing on Amber's dispatch (spec sections 2
and 16).

Usage (from project root):
    python3 analysis/scripts/cap_account_analysis.py \\
        --data-root "/Volumes/Manual/Thesis Project" \\
        --state NSW --year 2022 --battery-name 20kWh \\
        --threshold-per-mwh 300 --settlement-period monthly

--run-config both selects the coordination run (default
full_run_rolling_amber) and supplies the network tariff id needed for
import_lf/export_lf, which is only known via the run's own config YAML,
exactly as bill_reconciliation.py does it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Sites excluded as too large to be residential. Defined once in
# site_exclusions.py -- see there for the criterion, the measured figures,
# and why S0241 and the high-load cap beneficiaries are retained.
# Also spec section 3.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from site_exclusions import EXCLUDED_SITES

# Sanity-check tolerance for the recomputed baseline wholesale total against
# the recorded bills.csv wholesale columns (section-6-style tolerance,
# matching bill_reconciliation.py's).
_ABS_TOL_DOLLARS = 0.01
_REL_TOL = 1e-6

# Coordination types that optimise against the price the retailer holding
# dispatch rights faces, which is what spec section 2 assumes. The advanced
# types carry the mixed plan, so Offer C's own run is one of them.
_RETAILER_DISPATCH_TYPES = ('rolling_perfect_foresight_lp',
                            'rolling_perfect_foresight_lp_advanced',
                            'perfect_foresight', 'perfect_foresight_advanced')

# Settlement periods, coarsest last. Nothing longer than a month: bills are
# monthly and no balance is carried across a bill (spec section 4).
_PERIODS = ('interval', 'daily', 'weekly', 'monthly')


def _load_self_consumption_household_total(data_root: Path, state: str,
                                           year: int, battery_name: str
                                           ) -> pd.Series | None:
    """Household_total from the self_consumption (uncoordinated dispatch,
    same battery size) Amber run, indexed by site_id. Used to isolate the
    value of price-aware coordination specifically -- comparing against
    self_consumption (same hardware, dumber dispatch) rather than
    solar-only (no battery at all), since the latter conflates hardware
    value with coordination value. Returns None if the run isn't present
    for this state/year/battery -- callers should degrade gracefully (skip
    the dependent metric), not fail the whole script over an auxiliary
    comparison dataset.
    """
    path = (data_root / 'data' / 'output' / 'runs' / 'full_run_self_consumption_amber'
           / 'scenarios' / f'{state}_{year}_{battery_name}' / 'bills.csv')
    if not path.exists():
        return None
    df = pd.read_csv(path, usecols=['site_id', 'household_total'])
    df['site_id'] = df['site_id'].astype(str)
    return df.set_index('site_id')['household_total']


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capped-volatility retail offers: re-bill verified "
                    "flows under a wholesale price cap and settle the cap "
                    "account, per cap_account_spec.md.",
    )
    parser.add_argument('--project-root', type=Path, default=Path('.'),
                        help="Code repo root; model/tariffs/ and the run "
                             "config YAML are resolved here. Default: cwd.")
    parser.add_argument('--data-root', type=Path, default=Path('.'),
                        help="Data root; run outputs and NEM prices are "
                             "found here. Default: cwd.")
    parser.add_argument('--run-config', type=Path,
                        default=Path('model/configs/full_run_rolling_amber.yaml'),
                        help="Run config YAML, resolved relative to "
                             "--project-root if not absolute. Selects the "
                             "coordination run and supplies the network "
                             "tariff id. Point at "
                             "full_run_optimised_amber.yaml for the "
                             "perfect-foresight comparison.")
    parser.add_argument('--state', default='NSW',
                        help="State to analyse (default: NSW).")
    parser.add_argument('--year', type=int, default=2022)
    parser.add_argument('--battery-name', default='20kWh')
    parser.add_argument('--threshold-per-mwh', type=float, default=300.0,
                        help="Cap threshold x, $/MWh, on the raw wholesale "
                             "spot price (default: 300).")
    parser.add_argument('--offer', choices=('cohort', 'retailer', 'dynamic'),
                        default='cohort',
                        help="'cohort' = Offer A, cohort mutualisation "
                             "(spec section 7). 'retailer' = Offer B, "
                             "retailer as counterparty with a fractional "
                             "value share (spec section 10). 'dynamic' = "
                             "Offer C, TOU imports and a share of real-time "
                             "export value (spec Part V). Default: cohort.")
    parser.add_argument('--settlement-period', choices=_PERIODS,
                        default='monthly',
                        help="Window over which a site's own above-cap "
                             "imports and exports net, and over which the "
                             "cohort pool is drawn (spec section 4). "
                             "Default: monthly.")
    parser.add_argument('--surcharge-rate-per-kwh', type=float, default=0.01,
                        help="Offer A only. Hedging surcharge (spec "
                             "section 8), $/kWh on all imports, every "
                             "interval (default: 0.01, i.e. 1 c/kWh; 0 "
                             "disables it, reducing exactly to section 7).")
    parser.add_argument('--share-fraction', type=float, default=0.5,
                        help="Offer B only. Fraction of each site's own "
                             "above-cap export value shared back to it "
                             "(spec section 10). Default: 0.5.")
    parser.add_argument('--subscription-per-month', type=float, default=None,
                        help="Override the subscription fee, $/month. By "
                             "default it is read from the run's retail "
                             "tariff (spec section 14); pass this only to "
                             "model a variant at a deliberately different "
                             "fee.")
    parser.add_argument('--offer-run-config', type=Path,
                        default=Path('model/configs/'
                                     'full_run_rolling_tesla.yaml'),
                        help="Offer C only. Run config of the plan's own "
                             "modelled run, resolved as --run-config. Its "
                             "flows are the plan's dispatch and its retail "
                             "tariff is the plan. Not read under "
                             "--dynamic-dispatch rebill.")
    parser.add_argument('--dynamic-dispatch', choices=('own', 'rebill'),
                        default='own',
                        help="Offer C only. 'own' bills the plan's own "
                             "modelled run. 'rebill' bills the Amber run's "
                             "flows under --dynamic-tariff-id, which gives "
                             "the plan's billing on Amber's dispatch. "
                             "Default: own.")
    parser.add_argument('--dynamic-tariff-id',
                        default='{state}_tesla_electric_mixed_2026',
                        help="Offer C with --dynamic-dispatch rebill only. "
                             "The mixed retail tariff, formatted with "
                             "--state (default: "
                             "'{state}_tesla_electric_mixed_2026'). The "
                             "export share is read from the tariff. Under "
                             "'own' the plan is the offer run's own retail "
                             "tariff.")
    parser.add_argument('--output-dir', type=Path,
                        default=Path('analysis/output/cap_account'))
    args = parser.parse_args(argv)
    if args.offer == 'dynamic':
        for dest, flag in _CAP_ONLY_ARGS:
            if getattr(args, dest) != parser.get_default(dest):
                parser.error(f"{flag} does not apply to --offer dynamic, "
                             f"which has no cap account (spec section 19).")
    return args


def _variant_name(args) -> str:
    """Folder name for Offers A and B, encoding every parameter that
    changes the numbers, so sweeps don't overwrite each other (spec section
    12). Offer C names its own folder in _dynamic_variant."""
    parts = [args.offer, f'x{args.threshold_per_mwh:g}', args.settlement_period]
    if args.offer == 'cohort':
        parts.append(f's{args.surcharge_rate_per_kwh:g}')
    else:
        parts.append(f'a{args.share_fraction:g}')
    return '_'.join(parts)


def _split_net_flow(net: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a net (import - export) flows DataFrame into non-negative
    import_kwh / export_kwh DataFrames. A site cannot both import and
    export in one interval, which spec section 10 relies on."""
    import_kwh = net.clip(lower=0)
    export_kwh = (-net).clip(lower=0)
    return import_kwh, export_kwh


def _wholesale_net(import_kwh: pd.DataFrame, export_kwh: pd.DataFrame,
                   effective_price_kwh: pd.Series,
                   import_lf: float, export_lf: float,
                   import_gst: float = 0.0) -> pd.DataFrame:
    """Wholesale-layer net cost (SiteBill sign convention: cost positive,
    credit negative), per site per interval, at whatever per-interval price
    is supplied. Used for both the real baseline (effective_price =
    real_price always) and the capped billing (effective_price = min(real
    price, x)) -- same formula, different price series in.

    import_gst is the GST rate on the spot energy charge of imports (spec
    section 2): 0 for the wholesale layer as bills.csv records it, the
    plan's rate for what the household pays. Exports carry no GST."""
    usage_cost = (import_kwh.mul(effective_price_kwh, axis=0) * import_lf
                  * (1.0 + import_gst))
    feedin_credit = -(export_kwh.mul(effective_price_kwh, axis=0) * export_lf)
    return usage_cost + feedin_credit


def _cap_delta(import_kwh: pd.DataFrame, export_kwh: pd.DataFrame,
               price_kwh: pd.Series, x_kwh: float,
               import_lf: float, export_lf: float,
               import_gst: float = 0.0) -> pd.DataFrame:
    """Per-site, per-interval cap account delta (spec section 6), in
    household bill dollars: GST at import_gst on the import side, none on
    the export side. Positive = uncollected cost (net high-price importer),
    negative = unpaid credit (net high-price exporter). Zero on intervals
    where price <= x."""
    price_excess = (price_kwh - x_kwh).clip(lower=0)   # 0 when price <= x
    net_kwh_value = (import_kwh * import_lf * (1.0 + import_gst)
                     - export_kwh * export_lf)
    return net_kwh_value.mul(price_excess, axis=0)


def _import_gst(retail_tariff) -> float:
    """GST rate on the spot energy charge of imports for this plan: the
    billing rate where the tariff sets gst_on_wholesale_import, else 0."""
    from model.components.billing import GST_RATE
    return GST_RATE if getattr(retail_tariff, 'gst_on_wholesale_import',
                               False) else 0.0


def _amber_import_rate(rates_amber, amber_tariff) -> np.ndarray:
    """c[t] of spec section 17: what the Amber household pays per imported
    kWh apart from the subscription -- spot with its GST, network and the
    adder stack."""
    return (rates_amber.wholesale_import * (1.0 + _import_gst(amber_tariff))
            + rates_amber.network_import + rates_amber.retail_import)


def _bills_gst_residual(bills_df: pd.DataFrame, import_kwh: pd.DataFrame,
                        retail_tariff, import_gst: float) -> float:
    """Largest per-site |retail_usage_cost - import_gst x
    wholesale_usage_cost - usage adders x annual imports| over the sites in
    import_kwh. Zero when a wholesale_exposed run's bills carry the GST on
    spot imports that billing charges (patch_amber_gst.py) and the flows
    match the bills."""
    imports = import_kwh.sum()
    sites = [s for s in imports.index if s in bills_df.index]
    b = bills_df.loc[sites]
    resid = (b['retail_usage_cost'] - import_gst * b['wholesale_usage_cost']
             - retail_tariff.usage_other_c_per_kwh / 100.0 * imports[sites])
    return float(resid.abs().max())


def _period_keys(index: pd.Index, lwc, settlement_period: str,
                 target_year: int) -> tuple[pd.Series, pd.Series]:
    """(period_key, month_key) per interval, both derived from local wall
    clock rather than the AEST-indexed timestamps so they follow billing
    semantics through DST (spec section 4).

    period_key is the start of the settlement period as a naive Timestamp;
    month_key is the calendar month as a Period, used for billing and for
    every monthly output regardless of settlement period.

    Because DST pushes local time forward relative to AEST, the last ~12
    intervals of the AEST-indexed year fall on local 1 January of the
    following year. Those are folded back into the final period of the
    target year rather than forming a near-empty extra period. At
    'interval' settlement there is nothing to fold: each interval settles
    alone, so only month_key is clamped.
    """
    local = pd.Series(pd.DatetimeIndex(lwc.date), index=index)
    if local.dt.tz is not None:
        local = local.dt.tz_localize(None)

    month_key = local.dt.to_period('M')
    target_december = pd.Period(f'{target_year}-12', freq='M')
    month_key = month_key.where(month_key.dt.year == target_year,
                                target_december)

    in_year = local.dt.year == target_year
    if settlement_period == 'monthly':
        period_key = month_key.dt.to_timestamp()
    elif settlement_period == 'daily':
        period_key = local.dt.normalize()
        last = pd.Timestamp(f'{target_year}-12-31')
        period_key = period_key.where(in_year, last)
    elif settlement_period == 'weekly':
        # 7-day blocks anchored at the first of each calendar month, with a
        # ragged final block of 1-7 days. No settlement period may span two
        # calendar months, so ISO weeks are not usable (spec section 4).
        block = (local.dt.day - 1) // 7
        period_key = (month_key.dt.to_timestamp()
                      + pd.to_timedelta(block * 7, unit='D'))
        last = pd.Timestamp(f'{target_year}-12-29')
        period_key = period_key.where(in_year, last)
    elif settlement_period == 'interval':
        # The AEST index, NOT the local wall clock. lwc.date is a tz-aware
        # LOCAL DATE at midnight resolution, so using it here grouped every
        # interval in a day together and made 'interval' settlement an exact
        # duplicate of 'daily' -- byte-identical across all 240 cohort-months
        # when first run. Each interval settles alone, so its own timestamp
        # is the key, and the AEST index is both unique and monotonic. Local
        # time is not: when DST ends the 02:00-03:00 hour repeats, so a
        # naive local key would silently net twelve pairs of intervals
        # against each other.
        period_key = pd.Series(index, index=index)
    else:
        raise ValueError(f"unknown settlement period {settlement_period!r}")

    return period_key, month_key


def _settle_cohort(S: pd.DataFrame, period_month: pd.Series,
                   monthly_R: pd.Series
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Offer A settlement (spec sections 7-8), in two stages.

    Stage 1 runs the cap account settlement per settlement period with no
    surcharge. Stage 2 applies each month's surcharge revenue R -- which
    always pools over the calendar month, whatever the settlement period,
    because the surcharge gathers value across every import interval to
    fund shortfalls arising in rare events -- in priority order: first to
    reduce the retailer's position, then to refund the skim taken from
    credit-side sites, then as a bonus above their full original credit.

    At settlement_period='monthly' this reduces algebraically to the
    single-stage form p = clip(0, 1, (cost - R) / credit),
    bonus_rate = max(0, R - cost) / credit; with R = 0 it reduces further
    to spec section 7 unmodified. Both are regression checks.

    S: per settlement period (index) per site (columns) cap account balance.
    period_month: settlement period -> calendar month.
    monthly_R: surcharge revenue per calendar month.

    Returns (adjustment, summary). adjustment has S's shape and sums
    correctly at both period and month level; monthly-scope stage 2
    corrections land on the last period row of their month. summary is one
    row per settlement period, with monthly-scope columns likewise written
    only on the last period row of each month, so every column sums
    correctly down the file.
    """
    credit_side = S.clip(upper=0)                    # <= 0
    cost = S.clip(lower=0).sum(axis=1)               # per period
    credit = (-credit_side).sum(axis=1)              # per period, >= 0

    p = pd.Series(1.0, index=S.index)
    has_credit = credit > 0
    p[has_credit] = np.minimum(1.0, cost[has_credit] / credit[has_credit])

    adjustment = credit_side.mul(1.0 - p, axis=0)    # <= 0, a credit
    skim = (-credit_side).mul(p, axis=0)             # >= 0
    shortfall = (cost - credit).clip(lower=0)

    summary = pd.DataFrame({
        'total_cost_uncollected': cost,
        'total_credit_unpaid': credit,
        'p': p,
        'retailer_cost_absorbed': cost,
        'retailer_credit_retained': skim.sum(axis=1),
    })
    for col in ('R', 'bonus_rate', 'retailer_surcharge_retained'):
        summary[col] = np.nan

    last_period = period_month.groupby(period_month).apply(
        lambda g: g.index.max())

    for month, R in monthly_R.items():
        rows = period_month.index[period_month == month]
        if len(rows) == 0:
            continue
        anchor = last_period.loc[month]
        total_shortfall = shortfall.loc[rows].sum()
        total_skim = skim.loc[rows].to_numpy().sum()
        total_credit = credit.loc[rows].sum()

        retailer_net_cost = max(0.0, total_shortfall - R)
        R_rem = max(0.0, R - total_shortfall)
        refund_frac = min(1.0, R_rem / total_skim) if total_skim > 0 else 0.0
        bonus_rate = (max(0.0, R_rem - total_skim) / total_credit
                      if total_credit > 0 else 0.0)

        adjustment.loc[anchor] = (
            adjustment.loc[anchor]
            - refund_frac * skim.loc[rows].sum(axis=0)
            + bonus_rate * credit_side.loc[rows].sum(axis=0)   # <= 0
        )
        summary.loc[anchor, 'R'] = R
        summary.loc[anchor, 'bonus_rate'] = bonus_rate
        # Spec section 14: the retailer's position is
        # cost_absorbed - credit_retained - surcharge_retained. This is NOT
        # max(0, total_shortfall - R): where a month has no credit pool the
        # whole of R is retained (section 8) and the floored form reports
        # $0.00 against a true gain of R.
        summary.loc[anchor, 'retailer_surcharge_retained'] = (
            R - refund_frac * total_skim - bonus_rate * total_credit)

    summary['retailer_net_cost'] = (
        summary['retailer_cost_absorbed']
        - summary['retailer_credit_retained']
        - summary['retailer_surcharge_retained'].fillna(0.0))

    return adjustment, summary


def _settle_retailer(S: pd.DataFrame, alpha: float
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Offer B settlement (spec section 10). No cohort pool: the retailer
    absorbs every uncollected cost, retains every unpaid credit, and shares
    back alpha of each site's own above-cap export value.

    Because sum over sites of max(0, -S) is total_credit_unpaid by
    definition, the period totals are closed-form:
        total_value_returned = alpha * credit
        retailer_net_cost    = cost - (1 - alpha) * credit
    which, unlike Offer A's, is not floored at zero.
    """
    credit_side = S.clip(upper=0)                    # <= 0
    cost = S.clip(lower=0).sum(axis=1)
    credit = (-credit_side).sum(axis=1)

    adjustment = alpha * credit_side                 # <= 0, a credit
    summary = pd.DataFrame({
        'total_cost_uncollected': cost,
        'total_credit_unpaid': credit,
        'p': np.nan,
        'R': np.nan,
        'bonus_rate': np.nan,
        'retailer_cost_absorbed': cost,
        'retailer_credit_retained': (1.0 - alpha) * credit,
        'retailer_surcharge_retained': np.nan,
        'retailer_net_cost': cost - (1.0 - alpha) * credit,
    })
    return adjustment, summary


# ---------------------------------------------------------------------------
# Offer C -- dynamic export share (cap_account_spec.md Part V)
# ---------------------------------------------------------------------------

# Arguments that apply only to Offers A and B, as (dest, flag). Offer C has
# no cap threshold, no settlement period, no surcharge and no alpha
# (spec section 19).
_CAP_ONLY_ARGS = (
    ('threshold_per_mwh', '--threshold-per-mwh'),
    ('settlement_period', '--settlement-period'),
    ('surcharge_rate_per_kwh', '--surcharge-rate-per-kwh'),
    ('share_fraction', '--share-fraction'),
)

# $0.50/month: the same materiality threshold Offers A and B use for
# n_months_better_off / n_months_worse_off.
_MONTH_MATERIALITY = 0.50


def _dynamic_tariff_id(args) -> str:
    """--dynamic-tariff-id with {state} filled in. The plan under
    --dynamic-dispatch rebill only."""
    return args.dynamic_tariff_id.format(state=args.state)


def _dynamic_variant(plan_id: str, state: str, dispatch: str) -> str:
    """'dynamic_{tariff}', the tariff id without its state prefix, tagged
    '_modelled' where the plan's own run supplies the dispatch, so the two
    sit side by side. No threshold or period, since Offer C has neither
    (spec section 19)."""
    prefix = f'{state}_'
    if plan_id.startswith(prefix):
        plan_id = plan_id[len(prefix):]
    tag = '_modelled' if dispatch == 'own' else ''
    return f'dynamic_{plan_id}{tag}'


def _run_dynamic(args, *, run_name: str, scenario, offer_run_name: str,
                 offer_scenario, import_kwh: pd.DataFrame,
                 export_kwh: pd.DataFrame, offer_import_kwh: pd.DataFrame,
                 offer_export_kwh: pd.DataFrame, bills_df: pd.DataFrame,
                 offer_bills_df: pd.DataFrame, plan_id: str,
                 prices_mwh: pd.Series, network_tariff, amber_tariff,
                 subscription_per_month: float, tariffs_dir: Path,
                 data_root: Path) -> int:
    """Offer C: the mixed plan billed on its own run's dispatch, against
    Amber billed on the Amber run's (spec sections 16, 17 and 19).

    The offer_ arguments are the plan's run and the unprefixed ones the
    Amber run; under --dynamic-dispatch rebill they are the same run.

    Every bill comes from the core model's billing, per site. The retailer
    components come from the layer rate vectors applied across all sites
    at once, so the annual identity check in section 17 tests the two
    against each other rather than one calculation against itself. That
    identity holds between the plan and Amber on a single dispatch, so it
    is taken on the plan's own flows on both sides.
    """
    from model.core.tariff import load_retail_tariff
    from model.core.prices import build_layer_rates
    from model.core.localtime import local_wall_clock
    from model.core.site_result import SiteTraces
    from model.components.billing import compute_bill_and_monthly_bills

    state = args.state
    interval_hours = 5 / 60
    rebill = args.dynamic_dispatch == 'rebill'
    plan = load_retail_tariff(plan_id, tariffs_dir)
    if plan.type != 'mixed':
        print(f"--offer dynamic needs a 'mixed' retail tariff; {plan_id!r} "
              f"is type {plan.type!r}.", file=sys.stderr)
        return 1
    if amber_tariff.type != 'wholesale_exposed':
        print(f"--offer dynamic compares against the run's own Amber tariff, "
              f"which must be 'wholesale_exposed'; "
              f"{scenario.retail_tariff_id!r} is type "
              f"{amber_tariff.type!r}.", file=sys.stderr)
        return 1

    index = offer_import_kwh.index
    sites = [str(s) for s in offer_import_kwh.columns]
    base_sites = [str(s) for s in import_kwh.columns]
    months_of = {m: pd.Period(f'{args.year}-{m:02d}', freq='M')
                 for m in range(1, 13)}

    # --- Bills, from the core model's own billing (spec section 19) ---
    # Three billings. 'offer' and 'amber_on_offer' are the plan and Amber on
    # the plan's dispatch, which the section 17 identity and the retailer
    # position rest on; 'amber_base' is Amber on the Amber run's dispatch,
    # which is the household's actual alternative. Under rebill the plan's
    # dispatch is the Amber run's, so the last two coincide.
    # Billing reads only import_kwh and export_kwh from SiteTraces, so the
    # decomposition fields are zero-filled.
    zeros = pd.Series(0.0, index=index)
    keys = ('offer', 'amber_on_offer', 'amber_base')
    annual_bills: dict[str, dict] = {k: {} for k in keys}
    monthly_bills: dict[str, dict] = {k: {} for k in keys}

    def _bill(site: str, imp: pd.Series, exp: pd.Series, key: str,
              tariff) -> None:
        traces = SiteTraces(
            site_id=site, import_kwh=imp, export_kwh=exp,
            battery_soc_kwh=zeros,
            sol_to_home_kwh=zeros, sol_to_bat_kwh=zeros,
            sol_to_grid_kwh=zeros, bat_to_home_kwh=zeros,
            grid_to_home_kwh=zeros, grid_to_bat_kwh=zeros,
            bat_to_grid_kwh=zeros, curtail_kwh=zeros,
        )
        annual, by_month = compute_bill_and_monthly_bills(
            traces, prices_mwh, network_tariff, tariff, state,
            interval_hours)
        annual_bills[key][site] = annual
        monthly_bills[key][site] = by_month

    for site in sites:
        _bill(site, offer_import_kwh[site], offer_export_kwh[site],
              'offer', plan)
        _bill(site, offer_import_kwh[site], offer_export_kwh[site],
              'amber_on_offer', amber_tariff)
    if rebill:
        annual_bills['amber_base'] = annual_bills['amber_on_offer']
        monthly_bills['amber_base'] = monthly_bills['amber_on_offer']
    else:
        for site in base_sites:
            _bill(site, import_kwh[site], export_kwh[site],
                  'amber_base', amber_tariff)

    def _annual(key: str, field: str) -> pd.Series:
        return pd.Series({s: getattr(b, field)
                          for s, b in annual_bills[key].items()})

    def _monthly(key: str, field: str) -> pd.DataFrame:
        """Calendar months as rows, sites as columns."""
        return pd.DataFrame(
            {s: {months_of[m]: getattr(b, field)
                 for m, b in by_month.items()}
             for s, by_month in monthly_bills[key].items()}).sort_index()

    # --- Reconciliation: each recomputed bill against the bills.csv of the
    # run it was billed on (section 19) ---
    recorded = bills_df['household_total']
    amber_annual = _annual('amber_base', 'household_total')
    offer_annual = _annual('offer', 'household_total')
    common_sites = (offer_annual.index.intersection(recorded.index)
                    .intersection(amber_annual.index))
    offer_imp = offer_import_kwh[common_sites]
    offer_exp = offer_export_kwh[common_sites]

    def _recon(label: str, computed: pd.Series,
               recorded_total: pd.Series) -> str:
        shared = computed.index.intersection(recorded_total.index)
        abs_diff = (computed[shared] - recorded_total[shared]).abs()
        rel_diff = abs_diff / recorded_total[shared].abs().clip(lower=1e-9)
        n_fail = int((~((abs_diff < _ABS_TOL_DOLLARS)
                        | (rel_diff < _REL_TOL))).sum())
        if n_fail > 0:
            line = (f"WARNING: recomputed {label} household_total differs "
                    f"from bills.csv for {n_fail}/{len(shared)} sites "
                    f"(max abs diff ${abs_diff.max():.4f}).")
            print(line, file=sys.stderr)
        else:
            line = (f"{label} household_total matches bills.csv for all "
                    f"{len(shared)} sites.")
            print(line)
        return line

    recon_line = _recon('Amber', amber_annual, recorded)
    offer_recon_line = None if rebill else _recon(
        plan.name, offer_annual, offer_bills_df['household_total'])

    # --- Retailer components (spec section 17) ---
    lwc = local_wall_clock(index, state, interval_hours)
    rates_amber = build_layer_rates(index, prices_mwh, network_tariff,
                                    amber_tariff, state, interval_hours,
                                    lwc=lwc)
    rates_offer = build_layer_rates(index, prices_mwh, network_tariff, plan,
                                    state, interval_hours, lwc=lwc)
    # r[t]: what the plan charges per imported kWh. c[t]: what that kWh
    # costs the retailer, with Amber's adder stack as pass-through cost.
    r = rates_offer.retail_import
    c = _amber_import_rate(rates_amber, amber_tariff)
    amber_export_rate = (rates_amber.wholesale_export
                         + rates_amber.network_export
                         + rates_amber.retail_export)
    import_margin = offer_imp.mul(np.clip(r - c, 0.0, None), axis=0)
    cost_absorbed = offer_imp.mul(np.clip(c - r, 0.0, None), axis=0)
    credit_retained = offer_exp.mul(
        amber_export_rate - rates_offer.retail_export, axis=0)

    # Local calendar month, as compute_monthly_bills assigns it, so the
    # monthly components sit in the same months as the monthly bills.
    month_key = pd.Series(lwc.date.month.to_numpy(), index=index).map(months_of)
    margin_m = import_margin.groupby(month_key).sum()
    absorbed_m = cost_absorbed.groupby(month_key).sum()
    retained_m = credit_retained.groupby(month_key).sum()
    # Fixed charges on section 14's subscription_per_month convention, the
    # same every month, so one Amber line serves all three offers.
    fixed_per_month = ((plan.daily_c_per_day - network_tariff.daily_c_per_day)
                       * 365 / 100.0 / 12.0)
    net_fixed_m = pd.DataFrame(fixed_per_month, index=margin_m.index,
                               columns=margin_m.columns)
    net_position_m = net_fixed_m + margin_m - absorbed_m + retained_m

    # --- Identity check: annual, every term as billed (section 17) ---
    net_fixed_billed = (_annual('offer', 'retail_daily_charges')
                        - _annual('offer', 'network_daily_charges')
                        )[common_sites]
    decomposed = (net_fixed_billed + import_margin.sum() - cost_absorbed.sum()
                  + credit_retained.sum())
    from_bills = (_annual('offer', 'household_total')
                  - (_annual('amber_on_offer', 'household_total')
                     - _annual('amber_on_offer', 'retail_daily_charges'))
                  )[common_sites]
    id_abs = (decomposed - from_bills).abs()
    id_rel = id_abs / from_bills.abs().clip(lower=1e-9)
    n_id_fail = int((~((id_abs < _ABS_TOL_DOLLARS)
                       | (id_rel < _REL_TOL))).sum())
    n_common = len(common_sites)
    if n_id_fail > 0:
        identity_line = (f"WARNING: section 17 identity fails for "
                         f"{n_id_fail}/{n_common} sites (max abs diff "
                         f"${id_abs.max():.4f}).")
        print(identity_line, file=sys.stderr)
    else:
        identity_line = (f"Section 17 identity holds for all {n_common} "
                         f"sites (max abs diff ${id_abs.max():.2e}).")
        print(identity_line)

    # --- Household outcome, full bill (section 16) ---
    # The household's alternative is Amber as Amber dispatches, so each side
    # carries its own run's bill. diff splits into what the plan's billing
    # does on one dispatch and what the change of dispatch does at Amber's
    # billing; under rebill the second term is zero.
    amber_m = _monthly('amber_base', 'household_total')
    offer_m = _monthly('offer', 'household_total')
    diff_m = offer_m - amber_m

    baseline_total = recorded[common_sites]
    offer_total = offer_annual[common_sites]
    amber_on_offer_total = _annual('amber_on_offer',
                                   'household_total')[common_sites]
    diff = offer_total - baseline_total
    diff_billing = offer_total - amber_on_offer_total
    diff_dispatch = amber_on_offer_total - baseline_total

    price_kwh = prices_mwh / 1000.0
    annual_export_earnings = (offer_exp.mul(price_kwh, axis=0)
                              * network_tariff.export_loss_factor).sum()
    pct_of_export_earnings = (diff / annual_export_earnings) * 100
    pct_of_export_earnings[annual_export_earnings.abs() < 1.0] = np.nan

    hh_self_consumption = _load_self_consumption_household_total(
        data_root, state, args.year, args.battery_name)
    if hh_self_consumption is not None:
        hh_coordination_saving = (hh_self_consumption.reindex(common_sites)
                                  - baseline_total)
        pct_of_coordination_savings = (diff / hh_coordination_saving) * 100
        pct_of_coordination_savings[hh_coordination_saving < 1.0] = np.nan
    else:
        print(f"WARNING: self_consumption baseline not found for {state} "
              f"{args.year} {args.battery_name} -- "
              f"pct_of_coordination_savings will be all-NaN.", file=sys.stderr)
        pct_of_coordination_savings = pd.Series(np.nan, index=common_sites)
        hh_coordination_saving = pd.Series(np.nan, index=common_sites)

    diff_m_common = diff_m[common_sites]
    per_site_annual = pd.DataFrame({
        'baseline_household_total': baseline_total,
        'offer_household_total': offer_total,
        'diff': diff,
        'diff_billing': diff_billing,
        'diff_dispatch': diff_dispatch,
        'pct_of_wholesale_export_earnings': pct_of_export_earnings,
        'annual_coordination_savings': hh_coordination_saving,
        'pct_of_coordination_savings': pct_of_coordination_savings,
        'n_months_better_off': (diff_m_common < -_MONTH_MATERIALITY).sum(axis=0),
        'n_months_worse_off': (diff_m_common > _MONTH_MATERIALITY).sum(axis=0),
    })
    per_site_annual.index.name = 'site_id'

    n_sites = len(common_sites)
    month_summary = pd.DataFrame({
        'month': net_position_m.index.astype(str),
        'n_sites': n_sites,
        'subscription_per_month': subscription_per_month,
        'net_fixed': net_fixed_m.sum(axis=1).to_numpy(),
        'import_margin': margin_m.sum(axis=1).to_numpy(),
        'cost_absorbed': absorbed_m.sum(axis=1).to_numpy(),
        'credit_retained': retained_m.sum(axis=1).to_numpy(),
        'net_position': net_position_m.sum(axis=1).to_numpy(),
    })

    # --- Outputs (section 19) ---
    output_dir = (args.output_dir / run_name
                  / f'{state}_{args.year}_{args.battery_name}'
                  / _dynamic_variant(plan_id, state, args.dynamic_dispatch))
    output_dir.mkdir(parents=True, exist_ok=True)

    monthly_metrics = {
        'amber_household_total': amber_m,
        'offer_household_total': offer_m,
        'diff': diff_m,
        'net_fixed': net_fixed_m,
        'import_margin': margin_m,
        'cost_absorbed': absorbed_m,
        'credit_retained': retained_m,
    }
    for metric_name, monthly_df in monthly_metrics.items():
        by_site_month = monthly_df.T   # rows: site_id, columns: month
        by_site_month.index.name = 'site_id'
        by_site_month.to_csv(output_dir / f'by_site_month_{metric_name}.csv')
    month_summary.to_csv(output_dir / 'dynamic_month_summary.csv', index=False)
    per_site_annual.to_csv(output_dir / 'per_site_annual.csv')

    label = f'{plan.name} vs Amber'
    _dynamic_charts(per_site_annual, diff_m_common, output_dir, label,
                    f'{state} {args.year} {args.battery_name}')

    per_site_month = month_summary[['net_fixed', 'import_margin',
                                    'cost_absorbed', 'credit_retained',
                                    'net_position']] / n_sites
    months_below = int((per_site_month['net_position']
                        < subscription_per_month).sum())
    lines = [
        f"Offer C, dynamic export share: {plan.name} "
        f"(share={plan.wholesale_export_share:g} of "
        f"{plan.export_share_basis}, floor="
        f"{plan.export_floor_c_per_kwh} c/kWh), {state} {args.year} "
        f"{args.battery_name}",
        f"Baseline run: {run_name}, scenario: {scenario.scenario_id}",
        (f"Dispatch: {run_name}, re-billed under the plan" if rebill else
         f"Dispatch: {offer_run_name} "
         f"({offer_scenario.coordination_type}), scenario: "
         f"{offer_scenario.scenario_id}"),
        "",
        f"Sites: {len(per_site_annual)}",
        recon_line,
        *([] if offer_recon_line is None else [offer_recon_line]),
        identity_line,
        "",
        "Retailer position, $ per site per month, mean over the year "
        "(fixed charges at c/day x 365 / 12; spec section 17):",
    ]
    for col in ('net_fixed', 'import_margin', 'cost_absorbed',
                'credit_retained', 'net_position'):
        lines.append(f"  {col:16s} {per_site_month[col].mean():10.2f}")
    lines += [
        f"  Amber comparator: subscription ${subscription_per_month:.2f}",
        f"  months with net_position below the Amber subscription: "
        f"{months_below} of {len(per_site_month)}",
        f"  worst month net_position: "
        f"${per_site_month['net_position'].min():.2f}",
        "",
        "Retailer position by month, $ per site:",
    ]
    for month, row in zip(month_summary['month'], per_site_month.itertuples()):
        lines.append(f"  {month}: net_position ${row.net_position:8.2f}  "
                     f"(margin {row.import_margin:.2f}, absorbed "
                     f"{row.cost_absorbed:.2f}, retained "
                     f"{row.credit_retained:.2f})")
    lines += [
        "",
        "Household bill impact (offer - Amber, full bill), $:",
        f"  mean={diff.mean():,.2f}  std={diff.std():,.2f}  "
        f"min={diff.min():,.2f}  max={diff.max():,.2f}",
        f"  sites better off (diff < 0): {(diff < 0).sum()}",
        f"  sites worse off (diff > 0):  {(diff > 0).sum()}",
        f"  of which billing, on the plan's dispatch: "
        f"mean={diff_billing.mean():,.2f}",
        f"  of which dispatch, at Amber's billing:    "
        f"mean={diff_dispatch.mean():,.2f}",
        "",
        f"Consistency (materiality threshold ${_MONTH_MATERIALITY:.2f}/month):",
        f"  sites never better off in any month:  "
        f"{(per_site_annual['n_months_better_off'] == 0).sum()}",
        f"  sites never worse off in any month:    "
        f"{(per_site_annual['n_months_worse_off'] == 0).sum()}",
    ]
    (output_dir / 'summary.txt').write_text('\n'.join(lines) + '\n')
    print(f"Report written to {output_dir}/")
    return 0


def _dynamic_charts(per_site_annual: pd.DataFrame, diff_month: pd.DataFrame,
                    output_dir: Path, label: str, scope: str) -> None:
    """Offer C's per-scenario charts: the same four as Offers A and B
    (spec section 11), on the full-bill difference."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    diff = per_site_annual['diff']
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(diff, bins=40, color='#4c72b0', edgecolor='white')
    ax.axvline(0, color='black', linewidth=1, linestyle='--')
    ax.set_xlabel('Household bill impact, offer - Amber, full bill ($)')
    ax.set_ylabel('Number of sites')
    ax.set_title(f'Annual household bill impact, {label}\n'
                 f'{scope} (n={len(diff)} sites)')
    fig.tight_layout()
    fig.savefig(output_dir / 'annual_diff_histogram.png', dpi=150)
    plt.close(fig)

    for column, stem, xlabel, title, colour in (
            ('pct_of_wholesale_export_earnings',
             'pct_of_export_earnings_histogram',
             'Bill impact as % of annual wholesale export earnings',
             'Annual bill impact relative to export earnings', '#55a868'),
            ('pct_of_coordination_savings',
             'pct_of_coordination_savings_histogram',
             'Bill impact as % of savings from coordination '
             '(vs self_consumption, same battery)',
             'Annual bill impact relative to coordination savings',
             '#c44e52')):
        values = per_site_annual[column].dropna()
        if len(values) == 0:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(values, bins=40, color=colour, edgecolor='white')
        ax.axvline(0, color='black', linewidth=1, linestyle='--')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Number of sites')
        n_excluded = len(per_site_annual) - len(values)
        excl_note = f", {n_excluded} site(s) excluded" if n_excluded else ""
        ax.set_title(f'{title}, {label}\n'
                     f'{scope} (n={len(values)} sites{excl_note})')
        fig.tight_layout()
        fig.savefig(output_dir / f'{stem}.png', dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot([diff_month.loc[m].values for m in diff_month.index],
               tick_labels=[str(m) for m in diff_month.index],
               showfliers=True)
    ax.axhline(0, color='black', linewidth=1, linestyle='--')
    ax.set_ylabel('Household bill impact, offer - Amber ($)')
    ax.set_title(f'Monthly distribution of household bill impact, {label}\n'
                 f'{scope}')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    fig.tight_layout()
    fig.savefig(output_dir / 'monthly_diff_boxplot.png', dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    project_root = args.project_root.resolve()
    data_root = args.data_root.resolve()
    state = args.state
    sys.path.insert(0, str(project_root))

    from model.main import load_scenarios
    from model.core import Paths
    from model.core.tariff import load_network_tariff, load_retail_tariff
    from model.components.ingest import _load_prices
    from model.core.localtime import local_wall_clock

    run_config_path = args.run_config
    if not run_config_path.is_absolute():
        run_config_path = project_root / run_config_path

    run_name, scenarios = load_scenarios(run_config_path)
    matching = [s for s in scenarios if s.state == state and s.year == args.year
               and s.battery_name == args.battery_name]
    if not matching:
        print(f"No scenario found for {state} {args.year} "
             f"{args.battery_name!r} in {run_config_path}", file=sys.stderr)
        return 1
    scenario = matching[0]

    if scenario.coordination_type not in _RETAILER_DISPATCH_TYPES:
        print(f"WARNING: scenario coordination_type="
             f"{scenario.coordination_type!r}, but this analysis assumes "
             f"retailer-optimised dispatch against the real price (spec "
             f"section 2). Proceeding anyway, but results may not be "
             f"meaningful.", file=sys.stderr)

    scenario_dir = (data_root / 'data' / 'output' / 'runs' / run_name
                    / 'scenarios' / scenario.scenario_id)
    flows_path = scenario_dir / 'flows.parquet'
    bills_path = scenario_dir / 'bills.csv'
    if not flows_path.exists() or not bills_path.exists():
        print(f"Missing flows.parquet or bills.csv in {scenario_dir}",
             file=sys.stderr)
        return 1

    flows_df = pd.read_parquet(flows_path)
    flows_df.columns = [str(c) for c in flows_df.columns]
    flows_df = flows_df.drop(columns=[c for c in EXCLUDED_SITES
                                      if c in flows_df.columns])
    import_kwh, export_kwh = _split_net_flow(flows_df)

    bills_df = pd.read_csv(bills_path)
    bills_df['site_id'] = bills_df['site_id'].astype(str)
    bills_df = bills_df.set_index('site_id')

    paths = Paths(
        edp_clean_dir=data_root / 'data' / 'clean' / 'edp',
        edp_metadata_dir=data_root / 'data' / 'processed' / 'edp',
        nem_prices_dir=data_root / 'data' / 'clean' / 'nem_prices',
        tariffs_dir=project_root / 'model' / 'tariffs',
    )
    prices_mwh = _load_prices(scenario, paths)   # raw $/MWh, per ingest.py
    if not prices_mwh.index.equals(flows_df.index):
        raise ValueError(
            f"Price index does not match flows.parquet index. "
            f"prices len={len(prices_mwh)}, flows len={len(flows_df)}. "
            f"First price ts={prices_mwh.index[0]}, "
            f"first flow ts={flows_df.index[0]}."
        )

    network_tariff = load_network_tariff(scenario.network_tariff_id,
                                         paths.tariffs_dir)
    import_lf = network_tariff.import_loss_factor
    export_lf = network_tariff.export_loss_factor

    # Subscription fee: the whole of the retailer's revenue under the
    # uncapped baseline (spec section 14). load_retail_tariff sums
    # daily_charges_c_per_day into daily_c_per_day, which for a
    # wholesale_exposed tariff is the subscription. On a 'tou' tariff the
    # same field is a supply charge, not retailer margin, so it is not
    # usable as a subscription without saying so. Recorded here rather than
    # left to cap_account_retailer.py so that the summary CSV is
    # self-contained.
    retail_tariff = load_retail_tariff(scenario.retail_tariff_id,
                                       paths.tariffs_dir)
    if args.subscription_per_month is not None:
        subscription_per_month = args.subscription_per_month
    elif retail_tariff.type == 'wholesale_exposed':
        # 365 always, including leap years: the c/day figure was itself
        # derived by dividing an annual subscription by 365, so using 366
        # in a leap year would overstate the fee rather than track it.
        # Monthly is the annual figure over 12, not the month's own day
        # count, so every month carries the same subscription.
        subscription_per_month = (retail_tariff.daily_c_per_day
                                  * 365 / 100.0 / 12.0)
    else:
        subscription_per_month = float('nan')
        print(f"WARNING: retail tariff {retail_tariff.name!r} has type "
             f"{retail_tariff.type!r}; its daily charge is a supply charge, "
             f"not a subscription, so subscription_per_month is left "
             f"unset. Pass --subscription-per-month to override.",
             file=sys.stderr)

    # Billing charges GST on spot imports where the tariff sets
    # gst_on_wholesale_import; household amounts below follow it (spec
    # section 2). Bills written before that rule must be patched first,
    # or the baseline household_total and the recomputed wholesale terms
    # would disagree by the GST.
    import_gst = _import_gst(retail_tariff)
    if retail_tariff.type == 'wholesale_exposed':
        gst_resid = _bills_gst_residual(bills_df, import_kwh, retail_tariff,
                                        import_gst)
        if gst_resid >= _ABS_TOL_DOLLARS:
            print(f"bills.csv retail_usage_cost differs from GST on spot "
                  f"imports + adders x imports by up to ${gst_resid:.4f} per "
                  f"site in {scenario_dir}; the run's bills do not match "
                  f"current billing (run patch_amber_gst.py).",
                  file=sys.stderr)
            return 1

    if args.offer == 'dynamic':
        # The plan's own run supplies its dispatch and its tariff; the run
        # loaded above stays the Amber baseline and comparator. Under
        # --dynamic-dispatch rebill the two are the same run, which is the
        # earlier treatment (spec section 16).
        if args.dynamic_dispatch == 'own':
            offer_config_path = args.offer_run_config
            if not offer_config_path.is_absolute():
                offer_config_path = project_root / offer_config_path
            offer_run_name, offer_scenarios = load_scenarios(offer_config_path)
            offer_matching = [s for s in offer_scenarios
                              if s.state == state and s.year == args.year
                              and s.battery_name == args.battery_name]
            if not offer_matching:
                print(f"No scenario found for {state} {args.year} "
                      f"{args.battery_name!r} in {offer_config_path}",
                      file=sys.stderr)
                return 1
            offer_scenario = offer_matching[0]
            if offer_scenario.network_tariff_id != scenario.network_tariff_id:
                print(f"Offer run network tariff "
                      f"{offer_scenario.network_tariff_id!r} differs from "
                      f"the baseline run's {scenario.network_tariff_id!r}; "
                      f"the network layer would not be comparable.",
                      file=sys.stderr)
                return 1
            if offer_scenario.coordination_type not in \
                    _RETAILER_DISPATCH_TYPES:
                print(f"WARNING: offer run coordination_type="
                      f"{offer_scenario.coordination_type!r}, but Offer C "
                      f"assumes the plan's retailer optimises against its "
                      f"own dispatch prices (spec section 16). Proceeding "
                      f"anyway.", file=sys.stderr)
            offer_dir = (data_root / 'data' / 'output' / 'runs'
                         / offer_run_name / 'scenarios'
                         / offer_scenario.scenario_id)
            if not (offer_dir / 'flows.parquet').exists() or \
                    not (offer_dir / 'bills.csv').exists():
                print(f"Missing flows.parquet or bills.csv in {offer_dir}",
                      file=sys.stderr)
                return 1
            offer_flows_df = pd.read_parquet(offer_dir / 'flows.parquet')
            offer_flows_df.columns = [str(c) for c in offer_flows_df.columns]
            offer_flows_df = offer_flows_df.drop(
                columns=[c for c in EXCLUDED_SITES
                         if c in offer_flows_df.columns])
            if not offer_flows_df.index.equals(flows_df.index):
                print(f"Offer run flows index does not match the baseline "
                      f"run's: {len(offer_flows_df)} intervals against "
                      f"{len(flows_df)}.", file=sys.stderr)
                return 1
            offer_import_kwh, offer_export_kwh = _split_net_flow(
                offer_flows_df)
            offer_bills_df = pd.read_csv(offer_dir / 'bills.csv')
            offer_bills_df['site_id'] = offer_bills_df['site_id'].astype(str)
            offer_bills_df = offer_bills_df.set_index('site_id')
            plan_id = offer_scenario.retail_tariff_id
        else:
            offer_run_name, offer_scenario = run_name, scenario
            offer_import_kwh, offer_export_kwh = import_kwh, export_kwh
            offer_bills_df = bills_df
            plan_id = _dynamic_tariff_id(args)
        return _run_dynamic(
            args, run_name=run_name, scenario=scenario,
            offer_run_name=offer_run_name, offer_scenario=offer_scenario,
            import_kwh=import_kwh, export_kwh=export_kwh,
            offer_import_kwh=offer_import_kwh,
            offer_export_kwh=offer_export_kwh,
            bills_df=bills_df, offer_bills_df=offer_bills_df,
            plan_id=plan_id,
            prices_mwh=prices_mwh, network_tariff=network_tariff,
            amber_tariff=retail_tariff,
            subscription_per_month=subscription_per_month,
            tariffs_dir=paths.tariffs_dir, data_root=data_root)

    price_kwh = prices_mwh / 1000.0
    x_kwh = args.threshold_per_mwh / 1000.0
    effective_price_capped = np.minimum(price_kwh, x_kwh)

    # Household amounts: GST on the import side (spec sections 5 and 6).
    baseline_wholesale = _wholesale_net(import_kwh, export_kwh, price_kwh,
                                       import_lf, export_lf, import_gst)
    capped_wholesale_realtime = _wholesale_net(import_kwh, export_kwh,
                                              effective_price_capped,
                                              import_lf, export_lf, import_gst)
    cap_delta = _cap_delta(import_kwh, export_kwh, price_kwh, x_kwh,
                          import_lf, export_lf, import_gst)

    # Hedging surcharge (spec section 8): a flat per-kWh charge on every
    # import, every interval, regardless of price -- kept as its own
    # DataFrame, separate from capped_wholesale_realtime and cap_delta
    # (neither of which the surcharge touches; S(site) itself is
    # unchanged by design, per spec section 8). Added into the household's
    # actual real-time total below, and its cohort-level monthly sum (R)
    # feeds into settlement.
    surcharge_rate = (args.surcharge_rate_per_kwh if args.offer == 'cohort'
                      else 0.0)
    surcharge_cost = import_kwh * surcharge_rate
    capped_realtime_total = capped_wholesale_realtime + surcharge_cost

    # Sanity check: recomputed annual baseline wholesale should match the
    # recorded bills.csv wholesale columns closely. A mismatch here would
    # indicate a price/loss-factor/unit-conversion bug in this script, not
    # in the verified baseline (spec section 5) -- checked before trusting
    # anything else.
    lwc = local_wall_clock(flows_df.index, state, interval_hours=5 / 60)
    period_key, month_key = _period_keys(flows_df.index, lwc,
                                         args.settlement_period, args.year)
    # Against the wholesale layer as bills.csv records it: no GST.
    recomputed_annual = _wholesale_net(import_kwh, export_kwh, price_kwh,
                                       import_lf, export_lf).sum()
    recorded_annual = (bills_df['wholesale_usage_cost']
                      + bills_df['wholesale_feedin_credit'])
    common_sites = recomputed_annual.index.intersection(recorded_annual.index)
    abs_diff = (recomputed_annual[common_sites]
               - recorded_annual[common_sites]).abs()
    rel_diff = abs_diff / recorded_annual[common_sites].abs().clip(lower=1e-9)
    n_fail = int((~((abs_diff < _ABS_TOL_DOLLARS)
                    | (rel_diff < _REL_TOL))).sum())
    if n_fail > 0:
        print(f"WARNING: recomputed baseline wholesale differs from "
             f"bills.csv for {n_fail}/{len(common_sites)} sites (max abs "
             f"diff ${abs_diff.max():.4f}). Treat results with caution -- "
             f"see cap_account_spec.md section 3/4.", file=sys.stderr)
    else:
        print(f"Sanity check passed: recomputed baseline wholesale matches "
             f"bills.csv for all {len(common_sites)} sites.")

    # Cap account balances net within the settlement period (spec section
    # 4): the same window sets both a site's own self-hedging and the
    # cohort pool. Everything else is grouped by calendar month, which is
    # the billing period regardless of settlement period.
    period_S = cap_delta.groupby(period_key).sum()
    monthly_capped_realtime = capped_realtime_total.groupby(month_key).sum()
    monthly_baseline = baseline_wholesale.groupby(month_key).sum()
    monthly_surcharge_by_site = surcharge_cost.groupby(month_key).sum()
    monthly_R = monthly_surcharge_by_site.sum(axis=1)   # cohort-level pool

    # settlement period -> calendar month, one row per settlement period
    period_month = (pd.DataFrame({'p': period_key, 'm': month_key})
                    .drop_duplicates().set_index('p')['m']
                    .reindex(period_S.index))

    if args.offer == 'cohort':
        period_adjustment, cohort_summary = _settle_cohort(
            period_S, period_month, monthly_R)
    else:
        period_adjustment, cohort_summary = _settle_retailer(
            period_S, args.share_fraction)

    cohort_summary.insert(0, 'settlement_period', args.settlement_period)
    cohort_summary.insert(0, 'offer', args.offer)
    cohort_summary['alpha'] = (args.share_fraction
                              if args.offer == 'retailer' else np.nan)
    # Constant per run, carried so cap_account_retailer.py needs nothing
    # but this file (spec section 14).
    cohort_summary['month'] = period_month.astype(str)
    cohort_summary['n_sites'] = period_S.shape[1]
    cohort_summary['subscription_per_month'] = subscription_per_month
    cohort_summary['total_value_returned'] = -period_adjustment.sum(axis=1)

    # Count of 5-min intervals above the cap, per settlement period --
    # distinguishes "no price spikes this period" (0 intervals, p defaults
    # to 1.0 as a non-event) from any other edge case that might also show
    # p=1.0 or p=0 for a less obvious reason.
    above_cap_mask = price_kwh > x_kwh
    cohort_summary['n_intervals_above_cap'] = (
        above_cap_mask.groupby(period_key).sum())
    cohort_summary = cohort_summary[[
        'offer', 'settlement_period', 'month', 'n_sites',
        'subscription_per_month',
        'total_cost_uncollected', 'total_credit_unpaid',
        'n_intervals_above_cap', 'total_value_returned',
        'retailer_cost_absorbed', 'retailer_credit_retained',
        'retailer_surcharge_retained', 'retailer_net_cost',
        'p', 'bonus_rate', 'R', 'alpha']]
    cohort_summary.index.name = 'period'

    # Settlement lands on bills monthly whatever the settlement period.
    adjustment = period_adjustment.groupby(period_month).sum()
    monthly_S = period_S.groupby(period_month).sum()
    monthly_final_wholesale = monthly_capped_realtime + adjustment

    # --- Per-site annual rollup ---
    annual_final_wholesale = monthly_final_wholesale.sum()
    annual_baseline_wholesale = monthly_baseline.sum()
    annual_S = monthly_S.sum()

    baseline_total = bills_df.loc[common_sites, 'household_total']
    capped_total = (baseline_total
                   - annual_baseline_wholesale[common_sites]
                   + annual_final_wholesale[common_sites])
    diff = capped_total - baseline_total

    # Percentage impact relative to the site's own real (uncapped) annual
    # export earnings -- not relative to baseline_household_total. With
    # full wholesale exposure and a sizeable battery, many sites' annual
    # bill is small or negative (net sellers to the market over the year),
    # which makes percentage-of-bill meaningless or sign-flipped for a
    # large fraction of the cohort (spec section 11). Export earnings is
    # well-defined for any site that exports at all.
    annual_export_earnings = (export_kwh.mul(price_kwh, axis=0)
                              * export_lf).sum()[common_sites]
    pct_of_export_earnings = (diff / annual_export_earnings) * 100
    pct_of_export_earnings[annual_export_earnings.abs() < 1.0] = np.nan

    # Percentage impact relative to the site's own savings from price-aware
    # coordination specifically -- household_total under self_consumption
    # (same battery size, uncoordinated dispatch) minus this scenario's
    # baseline_household_total (perfect_foresight, coordinated). Comparing
    # against self_consumption rather than solar-only isolates the value
    # of coordination itself, not the combined value of hardware+
    # coordination -- solar-only would conflate the two, since it also
    # credits the cap/surcharge comparison with the fact that the site has
    # a battery at all, which has nothing to do with the cap account.
    hh_self_consumption = _load_self_consumption_household_total(
        data_root, state, args.year, args.battery_name)
    if hh_self_consumption is not None:
        hh_coordination_saving = (hh_self_consumption.reindex(common_sites)
                                  - baseline_total)
        pct_of_coordination_savings = (diff / hh_coordination_saving) * 100
        pct_of_coordination_savings[hh_coordination_saving < 1.0] = np.nan
    else:
        print(f"WARNING: self_consumption baseline not found for {state} "
             f"{args.year} {args.battery_name} -- "
             f"pct_of_coordination_savings will be all-NaN.", file=sys.stderr)
        pct_of_coordination_savings = pd.Series(np.nan, index=common_sites)
        hh_coordination_saving = pd.Series(np.nan, index=common_sites)

    # Per-site count of months materially better/worse off (spec section 11;
    # $0.50 materiality threshold matches the household-diff boxplot's
    # implicit granularity -- small enough to catch real months, large
    # enough to ignore float noise).
    household_diff_month = (monthly_final_wholesale - monthly_baseline)[common_sites]
    _MATERIALITY = 0.50
    n_months_better_off = (household_diff_month < -_MATERIALITY).sum(axis=0)
    n_months_worse_off = (household_diff_month > _MATERIALITY).sum(axis=0)

    per_site_annual = pd.DataFrame({
        'baseline_household_total': baseline_total,
        'capped_household_total': capped_total,
        'diff': diff,
        'annual_cap_delta': annual_S[common_sites],
        'annual_value_returned': -adjustment.sum()[common_sites],
        'pct_of_wholesale_export_earnings': pct_of_export_earnings,
        'annual_coordination_savings': hh_coordination_saving,
        'pct_of_coordination_savings': pct_of_coordination_savings,
        'n_months_better_off': n_months_better_off,
        'n_months_worse_off': n_months_worse_off,
    })

    # --- Outputs ---
    # Nested under a scenario-named subfolder (state_year_battery), not
    # written flat into --output-dir, so results from different scenarios
    # don't overwrite each other and multiple runs can later be pulled
    # together for a cross-scenario summary.
    output_dir = (args.output_dir / run_name
                  / f'{state}_{args.year}_{args.battery_name}'
                  / _variant_name(args))
    output_dir.mkdir(parents=True, exist_ok=True)

    # One CSV per metric, site_id as rows and month as columns -- sites
    # (207) outnumber months (12) by more than an order of magnitude, so
    # this is the natural table shape (207 rows x 12 columns per file)
    # rather than one file with month rows and 5 x 207 metric-site columns.
    # capped_realtime_total includes the surcharge (it's part of what a
    # household actually pays in real time); surcharge_cost is broken out
    # separately so the surcharge's own contribution is visible on its own,
    # not just folded invisibly into a bigger number.
    monthly_metrics = {
        'baseline_wholesale_net': monthly_baseline,
        'capped_realtime_total': monthly_capped_realtime,
        'surcharge_cost': monthly_surcharge_by_site,
        'cap_delta': monthly_S,
        'adjustment': adjustment,
        'final_wholesale_net': monthly_final_wholesale,
    }
    for metric_name, monthly_df in monthly_metrics.items():
        by_site_month = monthly_df.T   # rows: site_id, columns: month
        by_site_month.index.name = 'site_id'
        by_site_month.to_csv(output_dir / f'by_site_month_{metric_name}.csv')

    cohort_summary.to_csv(output_dir / 'cohort_period_summary.csv')
    per_site_annual.to_csv(output_dir / 'per_site_annual.csv')

    total_retailer_net_cost = cohort_summary['retailer_net_cost'].sum()
    total_baseline_wholesale_cost = annual_baseline_wholesale[common_sites].sum()
    retailer_fraction = (total_retailer_net_cost / total_baseline_wholesale_cost
                        if total_baseline_wholesale_cost else float('nan'))

    # --- Charts: distribution of outcomes, annual and monthly ---
    # Required to interpret the results (a mean/std summary alone hides
    # skew and the number of sites at either tail). Household-level
    # wholesale-layer impact per month (final - baseline) is the monthly
    # equivalent of the annual `diff` column.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(diff, bins=40, color='#4c72b0', edgecolor='white')
    ax.axvline(0, color='black', linewidth=1, linestyle='--')
    ax.set_xlabel('Household bill impact, capped - baseline ($)')
    ax.set_ylabel('Number of sites')
    ax.set_title(f'Annual household bill impact under cap\n'
                f'{state} {args.year} {args.battery_name}, '
                f'x=${args.threshold_per_mwh:.0f}/MWh (n={len(diff)} sites)')
    fig.tight_layout()
    fig.savefig(output_dir / 'annual_diff_histogram.png', dpi=150)
    plt.close(fig)

    pct_export = per_site_annual['pct_of_wholesale_export_earnings'].dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pct_export, bins=40, color='#55a868', edgecolor='white')
    ax.axvline(0, color='black', linewidth=1, linestyle='--')
    ax.set_xlabel('Bill impact as % of annual wholesale export earnings')
    ax.set_ylabel('Number of sites')
    n_excluded = len(per_site_annual) - len(pct_export)
    excl_note = (f", {n_excluded} site(s) excluded: near-zero export earnings"
                if n_excluded else "")
    ax.set_title(f'Annual bill impact relative to export earnings\n'
                f'{state} {args.year} {args.battery_name}, '
                f'x=${args.threshold_per_mwh:.0f}/MWh (n={len(pct_export)} sites'
                f'{excl_note})')
    fig.tight_layout()
    fig.savefig(output_dir / 'pct_of_export_earnings_histogram.png', dpi=150)
    plt.close(fig)

    pct_coord = per_site_annual['pct_of_coordination_savings'].dropna()
    if len(pct_coord) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(pct_coord, bins=40, color='#c44e52', edgecolor='white')
        ax.axvline(0, color='black', linewidth=1, linestyle='--')
        ax.set_xlabel('Bill impact as % of savings from coordination '
                      '(vs self_consumption, same battery)')
        ax.set_ylabel('Number of sites')
        n_excluded = len(per_site_annual) - len(pct_coord)
        excl_note = (f", {n_excluded} site(s) excluded: no/negative "
                    f"coordination savings" if n_excluded else "")
        ax.set_title(f'Annual bill impact relative to coordination savings\n'
                    f'{state} {args.year} {args.battery_name}, '
                    f'x=${args.threshold_per_mwh:.0f}/MWh '
                    f'(n={len(pct_coord)} sites{excl_note})')
        fig.tight_layout()
        fig.savefig(output_dir / 'pct_of_coordination_savings_histogram.png',
                   dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    month_labels = [str(m) for m in household_diff_month.index]
    ax.boxplot(
        [household_diff_month.loc[m].values for m in household_diff_month.index],
        tick_labels=month_labels, showfliers=True,
    )
    ax.axhline(0, color='black', linewidth=1, linestyle='--')
    ax.set_ylabel('Household wholesale-layer impact, capped - baseline ($)')
    ax.set_title(f'Monthly distribution of household wholesale-layer impact\n'
                f'{state} {args.year} {args.battery_name}, '
                f'x=${args.threshold_per_mwh:.0f}/MWh')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    fig.tight_layout()
    fig.savefig(output_dir / 'monthly_diff_boxplot.png', dpi=150)
    plt.close(fig)

    offer_label = ('Offer A, cohort mutualisation' if args.offer == 'cohort'
                   else 'Offer B, retailer as counterparty')
    param_label = (f"surcharge=${args.surcharge_rate_per_kwh:.4f}/kWh"
                   if args.offer == 'cohort'
                   else f"alpha={args.share_fraction:g}")
    lines = [
        f"{offer_label}: {state} {args.year} {args.battery_name}, "
        f"x=${args.threshold_per_mwh:.0f}/MWh, "
        f"settlement={args.settlement_period}, {param_label}",
        f"Run: {run_name}, scenario: {scenario.scenario_id}",
        "",
        f"Sites: {len(per_site_annual)}",
        "",
        f"Retailer total net cost: ${total_retailer_net_cost:,.2f}"
        f"  (negative = retailer gain; spec section 14)",
        f"  as fraction of cohort baseline wholesale cost: "
        f"{retailer_fraction:.4%}",
        "",
        "Retailer net cost by settlement period (monthly-scope values on "
        "the last period of each month):",
    ]

    def _f(v, fmt):
        return 'n/a' if pd.isna(v) else format(v, fmt)

    for period, row in cohort_summary.iterrows():
        lines.append(f"  {period}: ${_f(row['retailer_net_cost'], ',.2f')}  "
                     f"(p={_f(row['p'], '.4f')}, "
                     f"R=${_f(row['R'], ',.2f')}, "
                     f"bonus_rate={_f(row['bonus_rate'], '.4f')}, "
                     f"{int(row['n_intervals_above_cap'])} intervals above cap)")
    charts_line = ("Charts: annual_diff_histogram.png, "
                  "pct_of_export_earnings_histogram.png, "
                  "monthly_diff_boxplot.png")
    if len(pct_coord) > 0:
        charts_line += ", pct_of_coordination_savings_histogram.png"
    lines += [
        "",
        charts_line,
        "",
        "Household bill impact (capped - baseline), $:",
        f"  mean={diff.mean():,.2f}  std={diff.std():,.2f}  "
        f"min={diff.min():,.2f}  max={diff.max():,.2f}",
        f"  sites better off (diff < 0): {(diff < 0).sum()}",
        f"  sites worse off (diff > 0):  {(diff > 0).sum()}",
        f"  sites unchanged:             {(diff == 0).sum()}",
        "",
        "Consistency (materiality threshold $0.50/month):",
        f"  sites never better off in any month:  "
        f"{(per_site_annual['n_months_better_off'] == 0).sum()}",
        f"  sites never worse off in any month:    "
        f"{(per_site_annual['n_months_worse_off'] == 0).sum()}",
        "",
        "Concentration of total household benefit (sites ranked by most "
        "negative diff):",
    ]
    total_benefit = -diff[diff < 0].sum()   # positive number: total $ benefit pool
    sorted_benefit = (-diff[diff < 0]).sort_values(ascending=False)
    for n in (1, 5, 10):
        if total_benefit > 0:
            share = sorted_benefit.head(n).sum() / total_benefit
        else:
            share = float('nan')
        lines.append(f"  top {n:2d} site(s): {share:.1%} of total benefit")
    (output_dir / 'summary.txt').write_text('\n'.join(lines) + '\n')

    print(f"\nDone. Retailer net cost ${total_retailer_net_cost:,.2f} "
         f"({retailer_fraction:.4%} of baseline wholesale cost).")
    print(f"Report written to {output_dir}/")

    return 0


if __name__ == '__main__':
    sys.exit(main())