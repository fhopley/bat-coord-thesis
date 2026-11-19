"""Billing component: compute settlement results for a site.

Takes the physical energy flows (SiteTraces) and computes three settlement
layers - wholesale, network, retail - then the two views (household_total,
retailer_margin). See SiteBill for the meaning of each field.

Per-interval rate construction (unit conversions and loss factors) lives in
`model.core.prices`. Billing applies those loss-adjusted $/kWh rate vectors to
the flows and sums them; it does not build or loss-adjust rates itself. This
keeps billing and the perfect-foresight optimiser on a single, shared set of
prices. On a mixed plan the optimiser's import price deliberately differs
from the billed rate (prices.build_price_vectors); the export rate is shared
through prices.mixed_export_rate.

SIGN CONVENTIONS (canonical version in docs/model_spec.md)
----------------------------------------------------------
- import_kwh and export_kwh are stored non-negative; direction is encoded
  by which field, not by sign.
- Costs are stored positive (add to the bill); credits are stored negative
  (reduce the bill). Every "total" is therefore a straight sum of its
  component fields.
- All feedin fields use the uniform formula:  -sum(export_kwh * rate).
  The export rate vectors from prices.py are the value EARNED per exported
  kWh (positive = reward/credit, negative = penalty/cost), so this single
  formula yields a credit (negative) for a reward and a cost (positive) for a
  penalty, and keeps household_total a straight sum.
- usage_other_c_per_kwh / feedin_other_c_per_kwh adders are folded into the
  retail rate vectors by prices.py.
- Loss factors apply to the wholesale and network layers only (applied in
  prices.py), never the retail customer-facing layer.

DAILY CHARGES
-------------
Daily supply/network charges are c/day -> $/day * n_days. They are not
per-interval rates and are not shared with the optimiser, so they are computed
here (using the shared C_TO_DOLLARS constant), not in prices.py.

SINGLE-PASS SETTLEMENT (annual + monthly share one build)
-----------------------------------------------------------
compute_bill (one SiteBill for the whole period) and compute_monthly_bills
(one SiteBill per calendar month present) both funnel through
_build_settlement_base, which does all of the genuinely expensive work --
build_layer_rates, local_wall_clock, and the block-tier / zero-import-reward
day-grouping loops -- exactly once per site, regardless of how many bills are
ultimately derived from it. _reduce_settlement then aggregates the resulting
_SettlementBase into a SiteBill, either for the whole period (month=None) or
for one local-calendar-month's intervals/days (month=1..12).

This works because every settlement component is one of two shapes:
  - a per-interval array (wholesale/network/retail usage & feedin, VPP
    reward, the block-tier "standard rate undo"): trivially summable over
    any subset of intervals via a boolean mask.
  - a per-local-calendar-day value (block tier cost, zero-import reward
    credit; also the plain day-count behind daily charges): the block-tier
    and zero-import-reward loops already group by local day internally
    before collapsing to a total -- _build_settlement_base keeps that
    per-day breakdown as a pd.Series instead of summing it immediately, so
    _reduce_settlement can filter it by local month and sum, or sum it in
    full for the annual bill.
Neither shape needs re-deriving rates or re-running a day-loop to be
aggregated a second (or thirteenth) way.

ERRORS
------
Raises ValueError if the price series misaligns with the traces index or
contains any NaN over the period (missing prices must be cleaned in the price
pipeline, not silently handled here). For large runs this should also be
checked at a pre-flight stage in the orchestrator/main before any computation
begins, so a fatal data error fails fast rather than after hours of processing.
"""
from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from model.core.prices import (
    build_layer_rates, block_window_mask, block_tier_masks,
    apply_critical_rates, C_TO_DOLLARS,
)

# GST on the spot energy charge of a wholesale_exposed plan whose tariff
# sets gst_on_wholesale_import. Every other rate in the tariff files already
# includes GST, and export payments to households are GST-free.
GST_RATE = 0.10
from model.core.localtime import local_wall_clock, LocalWallClock
from model.core.tariff import NetworkTariff, RetailTariff, ZeroImportReward
from model.core.site_result import SiteTraces, SiteBill, VppTraces


def _span_days(dates: pd.DatetimeIndex) -> int:
    """Number of distinct calendar days present in `dates`.

    Counts distinct (month, day) pairs, deliberately ignoring year, rather
    than distinct full dates or the span between the first and last date
    (an earlier version tried both of those in turn; see below for why
    neither was right). This function assumes -- as the rest of the
    monthly-billing design already does, by keying buckets on bare month
    number 1-12 rather than (year, month) -- that `dates` never genuinely
    spans more than one calendar year. It is not a multi-year day counter
    and was never intended to be; if that's ever needed, monthly buckets
    would need to become (year, month)-keyed throughout, a larger change.

    Used two ways, on two different kinds of date array:
      - _n_days (whole-period): AEST interval-start dates. For a single
        contiguous year (always true here) every (month, day) pair is
        already unique, so stripping year changes nothing versus counting
        full dates.
      - _reduce_settlement (per-month, month=1..12 case): LOCAL calendar
        dates, not AEST ones -- deliberately, to avoid a separate DST
        artifact (see below). In a DST-observing state (NSW/VIC/SA), the
        last AEST hour of 31 December has local date 1 January of the
        FOLLOWING year -- and since month buckets are keyed by bare month
        number, that sliver lands in the SAME January bucket as the
        correct 31 dates already there. Counting distinct full dates would
        treat it as a 32nd distinct day (since it's technically a
        different year's 1 January); stripping year means it collapses
        onto the SAME (month=1, day=1) key already present from local
        1 January of the year being billed, so it's silently absorbed
        rather than counted as an extra day. December is untouched (the
        sliver never lands there), so the total across all 12 months comes
        back to exactly the annual figure -- full reconciliation, not a
        reduced residual.

    Why local dates (not AEST) for the per-month case: AEST and local
    dates diverge under DST (during DST, local = AEST+1 constantly); using
    AEST dates for a per-month bucket would make every local day that
    starts while DST is active appear to "borrow" a date from the
    previous month, at every month boundary within the DST season
    (roughly October-April in Australia), not just the December/January
    one -- an artifact of counting the wrong kind of date entirely, which
    ignoring year alone would not fix (a borrowed date from, say,
    31 October would collide with an existing 31 October already in the
    November bucket only if date-stripping happened to line up that way,
    which it doesn't in general). Local dates avoid that class of problem
    from the start; ignoring year on top of that resolves the one
    remaining case (the year-boundary sliver) that local dates alone don't.

    Shared by _n_days and _reduce_settlement so both funnel through one
    counting formula and can't drift apart on that part, even though they
    intentionally pass different kinds of dates in.
    """
    month_day = dates.month * 100 + dates.day
    return int(len(np.unique(month_day)))


def _n_days(index: pd.DatetimeIndex, interval_hours: float) -> int:
    """Number of calendar days spanned by the period.

    Uses interval-START dates (timestamp - interval) so the 00:00 Jan-1-next
    interval-end (which covers 23:55-00:00 of Dec 31) counts as Dec 31.
    A full non-leap year then gives 365.
    """
    interval_delta = pd.Timedelta(hours=interval_hours)
    starts = index - interval_delta
    return _span_days(starts.normalize())


def _block_tier_components(
    flows_kwh: pd.Series,
    index: pd.DatetimeIndex,
    windows: tuple,
    rates_retail: np.ndarray,
    is_import: bool,
    interval_hours: float,
    state: str,
    other_rate: float = 0.0,
    lwc: Optional[LocalWallClock] = None,
    masks: Optional[tuple[np.ndarray, ...]] = None,
) -> tuple[np.ndarray, pd.Series]:
    """Per-interval standard-rate-undo array + per-local-date tier-cost Series.

    Returns (undo_arr, day_cost_series):

    undo_arr: length-T ndarray, the TOU ENERGY component already charged in
        retail_usage_cost / retail_feedin_credit for window intervals
        (`rates_retail - other_rate`, applied at window positions, zero
        elsewhere). rates_retail is the layer retail rate before critical
        rates are applied; with `masks` from block_tier_masks no critical
        interval is in a window mask anyway. The caller subtracts
        (import) or adds (export, since feedin values are credit-signed) the
        SUM of this array over
        whatever interval subset it is aggregating, to avoid double-
        counting the standard rate alongside the block tier cost.
        usage_other / feedin_other (`other_rate`) is deliberately excluded,
        so those flat levies remain charged at the standard rate for window
        intervals -- see the architecture note below.
    day_cost_series: pd.Series indexed by local calendar date (one entry per
        day that had any window intervals), value = that day's tiered cost
        (positive for import cost, negative for export credit -- the
        is_import sign flip is applied once to the whole Series here, not
        per-window, since it must apply uniformly to every day regardless
        of which window contributed to it). Summing the whole Series gives
        the annual block cost; filtering to one local month and summing
        gives that month's contribution -- see _reduce_settlement.

    This function replaces an earlier version that accumulated both pieces
    directly into scalars: it is otherwise identical (same per-window loop,
    same tier-filling logic, same architecture note below), just returning
    the pre-reduction components instead of the reduction itself, so
    _build_settlement_base can compute them once per site and
    _reduce_settlement can aggregate them for the annual bill or any single
    month without re-running this loop.

    Args:
        state: the site's actual location (scenario.state). Block window
            hours and the daily cumulative-threshold reset are both in local
            time, not AEST -- see core/localtime.py.
        other_rate: usage_other_rate ($/kWh) for import windows, or
            feedin_other_rate for export windows. Subtracted from
            rates_retail before computing undo_arr, so the flat levy is
            preserved in the caller's retail_usage_cost / retail_feedin_credit.
        lwc: Optional precomputed LocalWallClock for (index, state,
            interval_hours). Computed internally if not given, so existing
            direct callers (e.g. tests) are unaffected.
        masks: Optional tier membership mask per window, in `windows` order.
            compute_bill passes prices.block_tier_masks, which leaves out
            intervals critical on the window's side. Plain block_window_mask
            is used when not given.

    Architecture note on the undo step
    -----------------------------------
    `build_layer_rates` has no knowledge of block windows and computes
    `retail_import_rates` as `TOU_rate + usage_other` for all intervals,
    including block window ones. `retail_usage_cost` therefore includes
    the TOU component for window intervals before this function is
    accounted for. The undo step removes that TOU component (via
    `rates_retail - other_rate`) so it is not double-counted alongside the
    block tier cost.

    The undo step handles non-zero TOU window rates defensively: if a tariff
    YAML sets a non-zero period rate inside the window (rather than zeroing it),
    the undo will still correctly replace it with the tier rate. The YAML loader
    emits a UserWarning in that case (block/TOU overlap), but billing remains
    correct regardless.
    """
    if lwc is None:
        lwc = local_wall_clock(index, state, interval_hours)

    n = len(index)
    undo_arr = np.zeros(n, dtype=float)
    day_costs: dict = {}  # local date -> unsigned (cost-perspective) dollars

    C = C_TO_DOLLARS

    if masks is not None and len(masks) != len(windows):
        raise ValueError(
            f"_block_tier_components: {len(masks)} masks for "
            f"{len(windows)} windows; one mask per window is required"
        )

    for w_idx, window in enumerate(windows):
        mask = (block_window_mask(index, window, state, interval_hours, lwc=lwc)
                if masks is None else np.asarray(masks[w_idx], dtype=bool))
        if not mask.any():
            continue

        # Standard-rate undo: a plain per-interval elementwise product,
        # restricted to window positions. Safe to slice+sum for any subset.
        energy_only_rates = rates_retail[mask] - other_rate
        undo_arr[mask] += flows_kwh.to_numpy()[mask] * energy_only_rates

        # Group window flows by local calendar day, then apply the tier
        # structure to each day's total independently.
        window_flows = flows_kwh.iloc[mask]
        window_dates = lwc.date[mask]
        daily = window_flows.groupby(window_dates).sum()

        tiers = window.tiers
        for day, day_total in daily.items():
            remaining = float(day_total)
            prev_threshold = 0.0
            day_cost = 0.0
            for tier in tiers:
                tier_capacity = (
                    float('inf') if np.isinf(tier.threshold_kwh)
                    else tier.threshold_kwh - prev_threshold
                )
                volume = min(remaining, tier_capacity)
                day_cost += volume * (tier.rate_c_per_kwh / C)
                remaining -= volume
                if remaining <= 0:
                    break
                prev_threshold = tier.threshold_kwh
            day_costs[day] = day_costs.get(day, 0.0) + day_cost

    if day_costs:
        day_cost_series = pd.Series(day_costs, dtype=float)
        day_cost_series.index = pd.DatetimeIndex(day_cost_series.index)
    else:
        day_cost_series = pd.Series(dtype=float)

    # For export: block cost is a credit (negative bill contribution).
    # Applied once to the whole Series (equivalent to flipping each day's
    # contribution, since every day accumulated here goes through this same
    # sign convention) rather than per-window, matching the previous
    # scalar-accumulating version's single end-of-loop flip.
    if not is_import:
        day_cost_series = -day_cost_series

    return undo_arr, day_cost_series


def _zero_import_day_credits(
    import_kwh: pd.Series,
    index: pd.DatetimeIndex,
    reward: ZeroImportReward,
    interval_hours: float,
    state: str,
    lwc: Optional[LocalWallClock] = None,
) -> pd.Series:
    """Per-local-date credit Series for the zero-import reward.

    One entry per calendar day that has any interval in the reward window,
    value = -reward_per_day if that day's window imports are within
    tolerance, else 0.0 (both cases get an entry, so summing the whole
    Series or a month-filtered slice of it gives exactly
    "qualifying_days_in_scope * -reward_per_day", matching the previous
    scalar version's semantics for whatever period it's aggregated over).

    When tolerance_kwh_per_hour is 0, requires exactly zero imports in the
    window (threshold < 1e-9 kWh to allow for floating-point residuals from
    the LP solver).

    Args:
        state: the site's actual location (scenario.state). The reward
            window's hours and the daily qualification reset are both in
            local time, not AEST -- see core/localtime.py.
        lwc: Optional precomputed LocalWallClock for (index, state,
            interval_hours). Computed internally if not given, so existing
            direct callers (e.g. tests) are unaffected.
    """
    if lwc is None:
        lwc = local_wall_clock(index, state, interval_hours)
    start_hour = lwc.hour

    in_window = (
        (start_hour >= reward.start_hour) & (start_hour < reward.end_hour)
    )
    if not in_window.any():
        return pd.Series(dtype=float)

    window_imports = import_kwh.iloc[in_window]
    window_dates = lwc.date[in_window]
    daily_window_imports = window_imports.groupby(window_dates).sum()

    # 1e-9 kWh margin absorbs LP residuals (zero tolerance) and floating-point
    # summation error (imports exactly at a non-zero tolerance)
    threshold = reward.tolerance_kwh + 1e-9
    reward_per_day = reward.reward_c_per_day / C_TO_DOLLARS

    credits = np.where(
        daily_window_imports.to_numpy() <= threshold, -reward_per_day, 0.0,
    )
    return pd.Series(
        credits, index=pd.DatetimeIndex(daily_window_imports.index),
    )


@dataclass(frozen=True)
class _SettlementBase:
    """Precomputed per-interval/per-day settlement data for one site.

    Everything expensive (build_layer_rates, local_wall_clock, the block-
    tier and zero-import-reward day-loops) has already been done by the time
    this exists. _reduce_settlement turns this into a SiteBill, either for
    the whole period or for one local-calendar-month, without touching any
    of the above again. See module docstring for why every field here is
    shaped to make that possible (a per-interval array or a per-local-date
    Series).
    """
    site_id: str
    aest_dates: pd.DatetimeIndex     # per-interval AEST interval-start date (annual day count)
    local_dates: pd.DatetimeIndex    # per-interval local calendar date (per-month day count)
    local_months: np.ndarray         # per-interval local month (1-12), for month filtering
    wholesale_usage_arr: np.ndarray
    wholesale_feedin_arr: np.ndarray
    network_usage_arr: np.ndarray
    network_feedin_arr: np.ndarray
    retail_usage_arr: np.ndarray
    retail_feedin_arr: np.ndarray
    block_import_undo_arr: Optional[np.ndarray]
    block_import_day_costs: Optional[pd.Series]
    block_export_undo_arr: Optional[np.ndarray]
    block_export_day_costs: Optional[pd.Series]
    zero_import_day_credits: Optional[pd.Series]
    vpp_reward_arr: Optional[np.ndarray]


def _build_settlement_base(
    traces: SiteTraces,
    prices: pd.Series,
    network_tariff: NetworkTariff,
    retail_tariff: RetailTariff,
    state: str,
    interval_hours: float,
    vpp_traces: Optional[VppTraces] = None,
) -> _SettlementBase:
    """Do all the per-site settlement work exactly once.

    See module docstring and _SettlementBase. Raises the same ValueErrors
    compute_bill documents (misaligned/NaN prices via build_layer_rates;
    vpp_traces without a VppConfig) -- validation happens here, once, not
    repeated per reduction.
    """
    import_kwh = traces.import_kwh
    export_kwh = traces.export_kwh
    index = import_kwh.index

    lwc = local_wall_clock(index, state, interval_hours)

    rates = build_layer_rates(
        index, prices, network_tariff, retail_tariff, state, interval_hours,
        lwc=lwc,
    )

    imp = import_kwh.to_numpy(dtype=float, copy=False)
    exp = export_kwh.to_numpy(dtype=float, copy=False)

    wholesale_usage_arr = imp * rates.wholesale_import
    wholesale_feedin_arr = -(exp * rates.wholesale_export)
    network_usage_arr = imp * rates.network_import
    network_feedin_arr = -(exp * rates.network_export)

    # Critical rates replace the retail rate on the critical side of each
    # critical interval (shift_opposite=False: billing charges the real
    # rates). A no-op when the plan has none, so the standard path is
    # unaffected.
    retail_import_rates = rates.retail_import
    retail_export_rates = rates.retail_export
    if retail_tariff.has_critical_rates:
        retail_imp_s = pd.Series(retail_import_rates, index=index)
        retail_exp_s = pd.Series(retail_export_rates, index=index)
        retail_imp_s, retail_exp_s = apply_critical_rates(
            retail_imp_s, retail_exp_s, prices, retail_tariff,
            shift_opposite=False,
        )
        retail_import_rates = retail_imp_s.to_numpy()
        retail_export_rates = retail_exp_s.to_numpy()

    retail_usage_arr = imp * retail_import_rates
    retail_feedin_arr = -(exp * retail_export_rates)
    if retail_tariff.gst_on_wholesale_import:
        # GST on the spot energy charge, negative prices included. Dispatch
        # prices (prices.build_price_vectors) leave it out, so the optimiser
        # keeps p_imp >= p_exp at the price floor.
        retail_usage_arr = retail_usage_arr + GST_RATE * wholesale_usage_arr

    interval_delta = pd.Timedelta(hours=interval_hours)
    aest_dates = (index - interval_delta).normalize()

    block_import_undo_arr = None
    block_import_day_costs = None
    block_export_undo_arr = None
    block_export_day_costs = None
    zero_import_day_credits = None

    if retail_tariff.has_advanced_features:
        # Tier membership leaves out the critical side of critical intervals
        # (settled price), so the undo and the tier cost never touch a
        # critical rate: it stays in retail_usage_cost / retail_feedin_credit
        # and the flow does not count towards the tier. The undo uses the
        # pre-critical-rate layer rates, the TOU component the tier replaces.
        imp_masks, exp_masks = block_tier_masks(
            index, retail_tariff, state, interval_hours, prices, lwc=lwc)
        if retail_tariff.block_import_tiers:
            block_import_undo_arr, block_import_day_costs = _block_tier_components(
                traces.import_kwh, index, retail_tariff.block_import_tiers,
                rates.retail_import, is_import=True,
                interval_hours=interval_hours, state=state,
                other_rate=retail_tariff.usage_other_c_per_kwh / C_TO_DOLLARS,
                lwc=lwc, masks=imp_masks,
            )

        if retail_tariff.block_export_tiers:
            block_export_undo_arr, block_export_day_costs = _block_tier_components(
                traces.export_kwh, index, retail_tariff.block_export_tiers,
                rates.retail_export, is_import=False,
                interval_hours=interval_hours, state=state,
                other_rate=retail_tariff.feedin_other_c_per_kwh / C_TO_DOLLARS,
                lwc=lwc, masks=exp_masks,
            )

        if retail_tariff.zero_import_reward:
            zero_import_day_credits = _zero_import_day_credits(
                traces.import_kwh, index, retail_tariff.zero_import_reward,
                interval_hours, state, lwc=lwc,
            )

    vpp_reward_arr = None
    if vpp_traces is not None:
        if not retail_tariff.has_vpp:
            raise ValueError(
                f"vpp_traces supplied for site '{traces.site_id}' but "
                f"retail_tariff '{retail_tariff.name}' has no VppConfig. "
                f"Ensure the retail tariff YAML includes a 'vpp' block."
            )
        vpp = retail_tariff.vpp
        exp_reward = vpp.exp_reward_c_per_kwh / C_TO_DOLLARS  # $/kWh
        imp_reward = vpp.imp_reward_c_per_kwh / C_TO_DOLLARS  # $/kWh
        vpp_reward_arr = -(
            vpp_traces.f_exp_kwh.to_numpy(dtype=float) * exp_reward
            + vpp_traces.f_imp_kwh.to_numpy(dtype=float) * imp_reward
        )

    return _SettlementBase(
        site_id=traces.site_id,
        aest_dates=aest_dates,
        local_dates=lwc.date,
        local_months=lwc.date.month.to_numpy(),
        wholesale_usage_arr=wholesale_usage_arr,
        wholesale_feedin_arr=wholesale_feedin_arr,
        network_usage_arr=network_usage_arr,
        network_feedin_arr=network_feedin_arr,
        retail_usage_arr=retail_usage_arr,
        retail_feedin_arr=retail_feedin_arr,
        block_import_undo_arr=block_import_undo_arr,
        block_import_day_costs=block_import_day_costs,
        block_export_undo_arr=block_export_undo_arr,
        block_export_day_costs=block_export_day_costs,
        zero_import_day_credits=zero_import_day_credits,
        vpp_reward_arr=vpp_reward_arr,
    )


def _reduce_settlement(
    base: _SettlementBase,
    network_tariff: NetworkTariff,
    retail_tariff: RetailTariff,
    month: Optional[int] = None,
) -> SiteBill:
    """Aggregate a _SettlementBase into one SiteBill.

    month=None aggregates the whole period (the annual bill, byte-for-byte
    equivalent to summing every array/Series in full, and day-counted the
    same way _n_days always has: distinct AEST interval-start dates). An
    int 1-12 aggregates just that local-calendar-month's intervals/days --
    the caller (compute_monthly_bills) is responsible for only calling this
    with a month it has already confirmed has at least one interval.

    Day count for the monthly case uses distinct LOCAL dates, not AEST
    dates. AEST and local dates diverge under DST (during DST, local =
    AEST+1 constantly), which is invisible within a month but would
    corrupt a per-month day count if AEST dates were used: every local day
    that starts while DST is active would appear to "borrow" a date from
    the previous month at every month boundary within the DST season, not
    just the December/January one -- an artifact of counting the wrong
    thing, not a genuine daily-charge ambiguity. Local dates don't have
    that artifact: a local calendar day's date is its date, regardless of
    which AEST date its first interval happens to carry. This leaves
    exactly one genuine residual, at the edge of the dataset itself: the
    last AEST hour of 31 December (during DST) has local date 1 January of
    the FOLLOWING year, and since month buckets are keyed by bare month
    number (not year), that one sliver still merges into January's bucket
    -- a single, narrow +1, not a scattered one at every DST-season month
    boundary.
    """
    mask = None if month is None else (base.local_months == month)

    def _sum(arr: np.ndarray) -> float:
        return float(np.sum(arr)) if mask is None else float(np.sum(arr[mask]))

    def _day_sum(day_series: Optional[pd.Series]) -> float:
        if day_series is None or len(day_series) == 0:
            return 0.0
        if month is None:
            return float(day_series.sum())
        return float(day_series[day_series.index.month == month].sum())

    if mask is None:
        dates = base.aest_dates
    else:
        dates = base.local_dates[mask]
    n_days = _span_days(dates)
    C = C_TO_DOLLARS

    wholesale_usage_cost = _sum(base.wholesale_usage_arr)
    wholesale_feedin_credit = _sum(base.wholesale_feedin_arr)

    network_usage_cost = _sum(base.network_usage_arr)
    network_feedin_credit = _sum(base.network_feedin_arr)
    network_daily_charges = float(network_tariff.daily_c_per_day / C * n_days)

    retail_usage_cost = _sum(base.retail_usage_arr)
    retail_feedin_credit = _sum(base.retail_feedin_arr)
    retail_daily_charges = float(retail_tariff.daily_c_per_day / C * n_days)

    retail_other_credits = 0.0

    if base.block_import_day_costs is not None:
        standard_credit = _sum(base.block_import_undo_arr)
        block_cost = _day_sum(base.block_import_day_costs)
        retail_usage_cost -= standard_credit   # undo double-count
        retail_usage_cost += block_cost

    if base.block_export_day_costs is not None:
        standard_credit = _sum(base.block_export_undo_arr)
        block_cost = _day_sum(base.block_export_day_costs)
        retail_feedin_credit += standard_credit  # undo: feedin_credit is negative, add back to reduce magnitude
        retail_feedin_credit += block_cost

    if base.zero_import_day_credits is not None:
        retail_other_credits += _day_sum(base.zero_import_day_credits)

    if base.vpp_reward_arr is not None:
        retail_other_credits += _sum(base.vpp_reward_arr)

    # ---------------- Views ----------------
    network_net = (network_usage_cost + network_feedin_credit
                   + network_daily_charges)
    wholesale_net = wholesale_usage_cost + wholesale_feedin_credit
    retail_net = (retail_usage_cost + retail_feedin_credit
                  + retail_daily_charges)

    if retail_tariff.type == 'wholesale_exposed':
        household_total = wholesale_net + network_net + retail_net + retail_other_credits
    elif retail_tariff.type in ('tou', 'mixed'):
        # mixed: the retail layer already carries the household's floored
        # share of export value (prices.mixed_export_rate); wholesale and
        # network are retailer-side, as for tou.
        household_total = retail_net + retail_other_credits
    else:
        raise ValueError(
            f"unknown retail tariff type {retail_tariff.type!r} "
            f"({retail_tariff.name})"
        )

    retailer_margin = household_total - wholesale_net - network_net

    return SiteBill(
        site_id=base.site_id,
        plan_type=retail_tariff.type,
        wholesale_usage_cost=wholesale_usage_cost,
        wholesale_feedin_credit=wholesale_feedin_credit,
        network_usage_cost=network_usage_cost,
        network_feedin_credit=network_feedin_credit,
        network_daily_charges=network_daily_charges,
        retail_usage_cost=retail_usage_cost,
        retail_feedin_credit=retail_feedin_credit,
        retail_daily_charges=retail_daily_charges,
        household_total=household_total,
        retailer_margin=retailer_margin,
        retail_other_credits=retail_other_credits,
    )


def compute_bill(
    traces: SiteTraces,
    prices: pd.Series,
    network_tariff: NetworkTariff,
    retail_tariff: RetailTariff,
    state: str,
    interval_hours: float = 5 / 60,
    vpp_traces: Optional[VppTraces] = None,
) -> SiteBill:
    """Compute the SiteBill for a single site.

    Args:
        traces: Physical energy flows for the site (import_kwh, export_kwh,
            plus the decomposition). Only import_kwh and export_kwh are used
            by billing.
        prices: NEM RRP in $/MWh, indexed by datetime_aest. Must align with
            the traces index. Passed explicitly (not via DataContext) so the
            orchestrator can bill against true vs adjusted prices for the VCR
            comparison.
        network_tariff: Network charges and loss factors.
        retail_tariff: Retail plan (tou, wholesale_exposed or mixed).
        state: the site's actual location (scenario.state), used to resolve
            local time for all clock-driven tariff features (TOU periods,
            block tiers, zero-import reward). Deliberately independent of
            network_tariff.state / retail_tariff.state -- see
            build_layer_rates in prices.py.
        interval_hours: Interval length in hours (default 5/60).
        vpp_traces: Optional natural/forced flow split from perfect_foresight_vpp.
            When supplied, the VPP per-kWh dispatch rewards are computed from
            the forced flows and folded into retail_other_credits. The rewards
            are read from retail_tariff.vpp; if vpp_traces is supplied but
            retail_tariff.has_vpp is False a ValueError is raised. When None
            (the default), billing is unchanged from the non-VPP path.

    Returns:
        SiteBill with all three layers and both views populated.

    Raises:
        ValueError: if indices misalign, prices contain NaN, or vpp_traces is
            supplied without a VppConfig on the retail tariff.
    """
    base = _build_settlement_base(
        traces, prices, network_tariff, retail_tariff, state, interval_hours,
        vpp_traces=vpp_traces,
    )
    return _reduce_settlement(base, network_tariff, retail_tariff, month=None)


def bill_all_sites(
    traces_by_site: Mapping[str, SiteTraces],
    prices: pd.Series,
    network_tariff: NetworkTariff,
    retail_tariff: RetailTariff,
    state: str,
    interval_hours: float = 5 / 60,
) -> dict[str, SiteBill]:
    """Compute SiteBills for a collection of sites under one tariff set."""
    return {
        site_id: compute_bill(traces, prices, network_tariff,
                              retail_tariff, state, interval_hours)
        for site_id, traces in traces_by_site.items()
    }


def _reduce_all_months(
    base: _SettlementBase,
    network_tariff: NetworkTariff,
    retail_tariff: RetailTariff,
) -> dict[int, SiteBill]:
    """Reduce a _SettlementBase to one SiteBill per calendar month present.

    Shared by compute_monthly_bills and compute_bill_and_monthly_bills so
    both funnel through one "which months, reduce each" implementation.
    """
    months_present = np.unique(base.local_months)
    return {
        int(m): _reduce_settlement(base, network_tariff, retail_tariff, month=int(m))
        for m in months_present
    }


def compute_monthly_bills(
    traces: SiteTraces,
    prices: pd.Series,
    network_tariff: NetworkTariff,
    retail_tariff: RetailTariff,
    state: str,
    interval_hours: float = 5 / 60,
    vpp_traces: Optional[VppTraces] = None,
) -> dict[int, SiteBill]:
    """Compute one SiteBill per calendar month (1-12) for a single site.

    Builds ONE _SettlementBase for the whole period (build_layer_rates,
    local_wall_clock, and the block-tier/zero-import-reward day-loops each
    run exactly once, regardless of how many months are present), then
    reduces it once per month via _reduce_settlement. This means monthly
    bills always use exactly the same settlement logic as the annual bill
    (three layers, block tiers, zero-import reward, VPP rewards, critical
    rates): there is no separate monthly billing implementation to
    drift out of sync with compute_bill, and no re-derivation of rates or
    local time per month -- see the module docstring and _SettlementBase.

    NOTE: if you also need the annual SiteBill for the same traces, use
    compute_bill_and_monthly_bills instead of calling this function
    alongside compute_bill separately -- calling both separately means two
    independent _build_settlement_base builds (each re-deriving rates and
    local time from scratch) where one would do.

    Monthly daily charges (network_daily_charges / retail_daily_charges)
    are a day-count of distinct LOCAL (month, day) pairs in that month's
    bucket, deliberately ignoring year (via _span_days) -- this reconciles
    exactly against the annual bill's day count, including across DST
    transitions and the December/January year boundary, for the
    single-calendar-year traces this function assumes (see _span_days'
    docstring for the full reasoning and an earlier, incomplete version of
    this fix that left a small residual). This does assume `traces` never
    genuinely spans more than one calendar year -- monthly buckets are
    keyed by bare month number 1-12, not (year, month), so a period longer
    than 12 months would silently merge different years' same-numbered
    months. Not a concern for this model's actual usage (every
    ScenarioConfig has exactly one year), but worth knowing if this
    function is ever reused somewhere that assumption doesn't hold.

    Args:
        traces: Full-period SiteTraces for the site (the same traces billed
            by compute_bill for the annual bills.csv).
        prices, network_tariff, retail_tariff, state, interval_hours,
            vpp_traces: passed through unchanged to _build_settlement_base.

    Returns:
        Dict keyed by month number (1-12). A month with zero intervals in
        traces (e.g. a partial-year run) is omitted from the dict.
    """
    base = _build_settlement_base(
        traces, prices, network_tariff, retail_tariff, state, interval_hours,
        vpp_traces=vpp_traces,
    )
    return _reduce_all_months(base, network_tariff, retail_tariff)


def compute_bill_and_monthly_bills(
    traces: SiteTraces,
    prices: pd.Series,
    network_tariff: NetworkTariff,
    retail_tariff: RetailTariff,
    state: str,
    interval_hours: float = 5 / 60,
    vpp_traces: Optional[VppTraces] = None,
) -> tuple[SiteBill, dict[int, SiteBill]]:
    """Compute the annual SiteBill and the per-month SiteBill dict together,
    sharing ONE _build_settlement_base build.

    Use this instead of calling compute_bill(...) and
    compute_monthly_bills(...) separately when you need both for the same
    traces: calling them separately means build_layer_rates, local_wall_clock,
    and the block-tier/zero-import-reward day-loops each run TWICE (once per
    call) instead of once total. This is the function run_scenario uses when
    monthly billing has been requested for a scenario, so that requesting
    monthly bills alongside the annual one costs a handful of extra
    _reduce_settlement calls (cheap: masking + summing already-built arrays),
    not a second full settlement build.

    Args:
        traces, prices, network_tariff, retail_tariff, state, interval_hours,
            vpp_traces: identical to compute_bill / compute_monthly_bills.

    Returns:
        (annual_bill, monthly_bills) -- annual_bill is exactly what
        compute_bill(...) would return; monthly_bills is exactly what
        compute_monthly_bills(...) would return, keyed 1-12, omitting any
        month not present.
    """
    base = _build_settlement_base(
        traces, prices, network_tariff, retail_tariff, state, interval_hours,
        vpp_traces=vpp_traces,
    )
    annual_bill = _reduce_settlement(base, network_tariff, retail_tariff, month=None)
    monthly_bills = _reduce_all_months(base, network_tariff, retail_tariff)
    return annual_bill, monthly_bills