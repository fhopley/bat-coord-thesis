"""Tariff representation and YAML loaders.

The settlement model has two tariff roles, represented by two dataclasses:

  NetworkTariff  - physical network charges and loss factors. Always present
                   in a scenario. Properties of the connection point, not of
                   any retail plan. Used to compute BOTH the network layer and
                   (via its loss factors) the wholesale layer, regardless of
                   what retail plan the site is on.

  RetailTariff   - what the customer is billed by their retailer. A
                   time-of-use plan ('tou'), wholesale-exposed
                   ('wholesale_exposed'), or TOU imports with a floored
                   share of real-time export value ('mixed').

Both share the TariffPeriod building block and the period_rate_series()
function, which maps a list of time-of-day periods onto a 5-min interval
index. Period lookup uses interval-START (timestamp - interval) to respect
the interval-end timestamp convention of the EDP and NEM data: the rate for
an interval is the rate applying at the instant the interval began. This
instant is converted to the site's real local time (see core/localtime.py)
before being compared against a period's start_hour/end_hour, since tariff
periods are specified by DNSPs and retailers in local time, not AEST.

Loss factors (NetworkTariff)
----------------------------
Distribution Loss Factor (DLF) and Transmission Loss Factor (TLF) scale
metered energy to the regional reference node where wholesale price is set.
They are directional:
    import effective factor = usage_dlf * import_tlf
    export effective factor = export_dlf * export_tlf
export_dlf is typically < 1 (exporting reduces losses) while usage_dlf is
typically > 1, so a site is paid slightly less per exported kWh than the
nodal price and pays slightly more per imported kWh. These factors apply to
the network layer always, and to the wholesale layer always (the wholesale
layer reads them from the NetworkTariff) - even for a TOU customer who never
sees them in their own bill but whose retailer still incurs loss-adjusted
wholesale and network costs.
"""
import math
import warnings
from dataclasses import dataclass
from typing import Literal, Optional
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

from model.core.localtime import local_wall_clock, LocalWallClock


RetailTariffType = Literal['tou', 'wholesale_exposed', 'mixed']
ExportShareBasis = Literal['wholesale', 'wholesale_and_network']


# ----------------------------------------------------------------------
# Advanced tariff feature dataclasses (Steps 2-3)
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class BlockTier:
    """One tier within a block tariff window.

    threshold_kwh: cumulative daily energy upper bound for this tier (kWh).
        Use math.inf for the final (unbounded) tier.
    rate_c_per_kwh: rate charged/credited for energy within this tier.
    """
    threshold_kwh: float   # math.inf for final tier
    rate_c_per_kwh: float

    def __post_init__(self):
        if self.threshold_kwh <= 0:
            raise ValueError(
                f"BlockTier.threshold_kwh must be > 0, got {self.threshold_kwh}"
            )


@dataclass(frozen=True)
class BlockTariffWindow:
    """A daily time window with stepped (block) pricing.

    start_hour, end_hour: [start_hour, end_hour) window in hour-of-day.
        Values 0-24; end_hour may equal 24.
    tiers: ordered tuple of BlockTier (ascending threshold). Final tier
        must have threshold_kwh = math.inf.
    applies_weekends: if False, block rates only apply on weekdays; weekend
        intervals in this window fall back to the regular period rates.
    """
    start_hour: int
    end_hour: int
    tiers: tuple[BlockTier, ...]
    applies_weekends: bool = True

    def __post_init__(self):
        if not (0 <= self.start_hour < self.end_hour <= 24):
            raise ValueError(
                f"BlockTariffWindow: invalid hours [{self.start_hour}, "
                f"{self.end_hour})"
            )
        if len(self.tiers) < 2:
            raise ValueError("BlockTariffWindow: at least two tiers required")
        if not math.isinf(self.tiers[-1].threshold_kwh):
            raise ValueError(
                "BlockTariffWindow: final tier must have threshold_kwh=inf"
            )
        for i in range(len(self.tiers) - 1):
            if self.tiers[i].threshold_kwh >= self.tiers[i + 1].threshold_kwh:
                raise ValueError(
                    "BlockTariffWindow: tiers must be in ascending threshold order"
                )


@dataclass(frozen=True)
class ZeroImportReward:
    """A daily credit for avoiding imports above a tolerance during a window.

    reward_c_per_day: credit earned (in cents) when window imports are at or
        below the tolerance on a given day.
    tolerance_kwh_per_hour: retailer-specified import tolerance in kWh per
        hour of the window. Total tolerance = tolerance_kwh_per_hour x
        window_hours. Defaults to 0.0 (exact zero required). Used to calibrate
        the LP penalty: pi = reward_c_per_day / max(total_tolerance, 1.01)
        / C_TO_DOLLARS ($/kWh); see prices.build_zero_import_penalty.
    """
    start_hour: int
    end_hour: int
    reward_c_per_day: float
    tolerance_kwh_per_hour: float = 0.0

    def __post_init__(self):
        if not (0 <= self.start_hour < self.end_hour <= 24):
            raise ValueError(
                f"ZeroImportReward: invalid hours [{self.start_hour}, "
                f"{self.end_hour})"
            )
        if self.reward_c_per_day <= 0:
            raise ValueError("ZeroImportReward.reward_c_per_day must be > 0")
        if self.tolerance_kwh_per_hour < 0:
            raise ValueError(
                "ZeroImportReward.tolerance_kwh_per_hour must be >= 0"
            )

    @property
    def window_hours(self) -> int:
        return self.end_hour - self.start_hour

    @property
    def tolerance_kwh(self) -> float:
        """Total daily tolerance kWh across the full window."""
        return self.tolerance_kwh_per_hour * self.window_hours


@dataclass(frozen=True)
class CriticalRate:
    """Fixed per-interval rate that replaces one side's retail energy rate
    while the wholesale price is past a threshold.

    Critical export rate: while prices_mwh[t] > threshold, exports earn
        rate_c_per_kwh instead of the TOU or block tier export rate, and do
        not count towards any export block tier.
    Critical import rate: while prices_mwh[t] < threshold, imports cost
        rate_c_per_kwh instead of the TOU or block tier import rate, and do
        not count towards any import block tier. A negative rate pays the
        household to import.
    usage_other / feedin_other still apply on top. The opposite side of a
    critical interval is unchanged. See perfect_foresight_advanced_spec.md.

    threshold_dollars_per_mwh: the crossing threshold in $/MWh.
    rate_c_per_kwh: the replacement rate in c/kWh (finite; may be negative).
    """
    threshold_dollars_per_mwh: float
    rate_c_per_kwh: float

    def __post_init__(self):
        if not np.isfinite(self.threshold_dollars_per_mwh):
            raise ValueError("CriticalRate.threshold_dollars_per_mwh must be finite")
        if not np.isfinite(self.rate_c_per_kwh):
            raise ValueError("CriticalRate.rate_c_per_kwh must be finite")


@dataclass(frozen=True)
class ChargeWindow:
    """A daily time window during which a constraint on charging applies.

    start_hour, end_hour: [start_hour, end_hour) window in hour-of-day.
        Values 0-24; end_hour may equal 24. Same half-open convention as
        BlockTariffWindow. Carries no rate: it marks a time band only.

    Used by VppConfig.no_grid_charge_windows to mark the intervals in which
    the operator may not charge the battery from the grid during a VPP event.
    """
    start_hour: int
    end_hour: int

    def __post_init__(self):
        if not (isinstance(self.start_hour, int) and isinstance(self.end_hour, int)):
            raise ValueError(
                f"ChargeWindow hours must be integers, got "
                f"[{self.start_hour!r}, {self.end_hour!r})"
            )
        if not (0 <= self.start_hour < self.end_hour <= 24):
            raise ValueError(
                f"ChargeWindow: invalid hours [{self.start_hour}, "
                f"{self.end_hour})"
            )


MIN_EXPORT_PRICE_MODES = ('fixed', 'based_on_this_year', 'based_on_last_year')
"""Accepted values of VppConfig.min_export_price_mode.

Named once so the dataclass validation, validate_scenario and the resolver
cannot drift from each other. 'based_on_last_year' is accepted here and
rejected by validate_scenario: parsing it must produce a message about
implementation status rather than one about an unrecognised value.
"""


@dataclass(frozen=True)
class VppConfig:
    """VPP contract terms for a retail plan that includes VPP participation.

    Holds the household-facing contract parameters and the event-classification
    parameters used by vpp_event_classifier. All are properties of the retail
    plan, not of the scenario, because the operator is typically the retailer
    and these terms are defined in the customer's plan agreement.

    Attributes:
        exp_reward_c_per_kwh: Reward paid to the household per kWh of forced
            export (battery to grid during a VPP event). Must be >= 0.
        imp_reward_c_per_kwh: Reward paid to the household per kWh of forced
            import (grid to battery during a VPP event, i.e. operator
            force-charge). Must be >= 0. If the contract pays no import reward,
            set a small positive value (e.g. 0.01 c/kWh) rather than zero;
            the YAML loader (_parse_vpp_config) replaces an absent or zero
            value with 0.01 c/kWh to preserve the natural/forced split
            tie-break in the LP.
        kwh_limit: Annual cap on total operator-dispatched energy (kWh).
            Must be > 0. (AGL example: 250 kWh/yr.)
        limit_applies_to: Which forced flows count toward the annual cap.
            'export_only': only f_exp counts.
            'import_and_export': f_imp + f_exp count.
        default_mode: 'tou' or 'self_consumption'. Parsed and validated but
            not read by any current component: between-event behaviour is set
            by the coordination type (vpp_sc_event / vpp_tou_event and their
            rolling counterparts).
        event_window_hours: Length (hours) of the pre-spike window that the
            operator takes control of the battery before each price spike.
            (AGL example: 10.0 hours.) Must be > 0.
        vpp_threshold_dollars_per_mwh: Wholesale price level ($/MWh) above
            which an interval is classified as a price spike that triggers a
            VPP event window. Set equal to exp_reward_c_per_kwh * 10 to mark
            all spikes where gross forced-export revenue covers the reward;
            set higher to model a more conservative operator strategy. Must
            be > 0.
    """
    exp_reward_c_per_kwh: float
    imp_reward_c_per_kwh: float
    kwh_limit: float
    limit_applies_to: Literal['export_only', 'import_and_export']
    default_mode: Literal['tou', 'self_consumption']
    event_window_hours: float
    vpp_threshold_dollars_per_mwh: float
    curtail_c_per_kwh: Optional[float] = None
    """Curtailment penalty applied to event-window solar curtailment (c/kWh).
    None (default) disables curtailment entirely. 0 allows curtailment at zero
    cost. Positive values price each curtailed kWh in the operator LP objective;
    the LP then curtails only when exporting would cost more than curtailing.
    Must be >= 0 when provided. This is a dispatch-shaping parameter only and
    does not appear as a line item in the household bill.
    """
    min_export_price_dollars_per_mwh: Optional[float] = None
    """Minimum wholesale price ($/MWh) at which forced export (f_exp) is
    permitted within an event window. At intervals where op_exp < this
    threshold, f_exp is pinned to zero. Must be >= 0 when provided explicitly.

    What None means depends on the coordination type and the mode:

    The same rules apply to the perfect-foresight and rolling VPP types (both
    resolve through orchestrate.resolve_min_export_price):
    - min_export_price_mode='fixed': required. There is nothing to fall back
      to; validate_scenario rejects an absent value.
    - a derived mode: optional. The configured value acts as a FLOOR under the
      derived estimate rather than as the value itself, and absent means a
      floor of zero.
    """
    min_export_price_mode: Literal[
        'fixed', 'based_on_this_year', 'based_on_last_year'] = 'fixed'
    """Where the operative minimum export price comes from on any VPP run.

    'fixed' (default) uses min_export_price_dollars_per_mwh unchanged for the
    whole period.

    'based_on_this_year' derives the threshold from the modelled year's own
    prices and cohort load, by the same budget allocation algorithm the
    perfect-foresight path uses, and takes the greater of that estimate and the
    configured value. The configured value is the operator's own price floor,
    which the derivation can only tighten. This matters most where the
    derivation returns zero -- which it does whenever the available throughput
    across event intervals cannot exhaust kwh_limit -- since without a floor
    that zero would remove the restriction entirely.

    It is a look-ahead over part of what a rolling run measures, accepted
    because the threshold is a fleet planning parameter chosen once before the
    period rather than a dispatch decision taken during it. Operator-side value
    capture from such a run reads as an upper bound and is not comparable with
    a 'fixed' run of the same scenario. The mode and the resolved value are
    recorded on the scenario summary so the two are never confused after the
    fact.

    'based_on_last_year' would derive from year Y-1 instead. NOT IMPLEMENTED:
    accepted by this dataclass so a YAML naming it parses, then rejected by
    validate_scenario. See rolling_vpp_event_spec.md, "Minimum export price".

    Either way the result is a single scalar for the whole period. There is no
    within-run update, because the only information worth updating on --
    realised cap depletion -- is per-site state inside independent workers with
    no channel between them.

    Applies to all four VPP coordination types: the perfect-foresight and
    rolling runners both resolve through orchestrate.resolve_min_export_price.
    """
    no_grid_charge_windows: Optional[tuple[ChargeWindow, ...]] = None
    """Daily windows in which the operator may not charge the battery from the
    grid during a VPP event. None (default) means no restriction. Within a
    listed window, event-window charging is limited to the available solar
    surplus (via a coupling constraint on charge and curtail), so no grid energy
    reaches the battery; charging from the household's own solar and
    force-discharge into the window are unaffected. Validation of each window is
    handled by ChargeWindow.__post_init__. See vpp_sc_event_spec.md Section 5.y.
    """

    def __post_init__(self):
        if self.exp_reward_c_per_kwh < 0:
            raise ValueError(
                f"VppConfig.exp_reward_c_per_kwh must be >= 0, "
                f"got {self.exp_reward_c_per_kwh}"
            )
        if self.imp_reward_c_per_kwh < 0:
            raise ValueError(
                f"VppConfig.imp_reward_c_per_kwh must be >= 0, "
                f"got {self.imp_reward_c_per_kwh}"
            )
        if self.kwh_limit <= 0:
            raise ValueError(
                f"VppConfig.kwh_limit must be > 0, got {self.kwh_limit}"
            )
        if self.limit_applies_to not in ('export_only', 'import_and_export'):
            raise ValueError(
                f"VppConfig.limit_applies_to must be 'export_only' or "
                f"'import_and_export', got {self.limit_applies_to!r}"
            )
        if self.default_mode not in ('tou', 'self_consumption'):
            raise ValueError(
                f"VppConfig.default_mode must be 'tou' or 'self_consumption', "
                f"got {self.default_mode!r}"
            )
        if self.event_window_hours <= 0:
            raise ValueError(
                f"VppConfig.event_window_hours must be > 0, "
                f"got {self.event_window_hours}"
            )
        if self.vpp_threshold_dollars_per_mwh <= 0:
            raise ValueError(
                f"VppConfig.vpp_threshold_dollars_per_mwh must be > 0, "
                f"got {self.vpp_threshold_dollars_per_mwh}"
            )
        if self.curtail_c_per_kwh is not None and self.curtail_c_per_kwh < 0:
            raise ValueError(
                f"VppConfig.curtail_c_per_kwh must be >= 0 when provided, "
                f"got {self.curtail_c_per_kwh}"
            )
        if self.min_export_price_mode not in MIN_EXPORT_PRICE_MODES:
            raise ValueError(
                f"VppConfig.min_export_price_mode must be one of "
                f"{', '.join(repr(m) for m in MIN_EXPORT_PRICE_MODES)}, "
                f"got {self.min_export_price_mode!r}"
            )

        if (self.min_export_price_dollars_per_mwh is not None
                and self.min_export_price_dollars_per_mwh < 0):
            raise ValueError(
                f"VppConfig.min_export_price_dollars_per_mwh must be >= 0 "
                f"when provided, got {self.min_export_price_dollars_per_mwh}"
            )


# ----------------------------------------------------------------------
# Core tariff building blocks (unchanged)
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class TariffPeriod:
    """A time-of-day window with a single c/kWh rate.

    start_hour is inclusive, end_hour is exclusive. Together the periods of
    a tariff should tile [0, 24) without overlap or gap. An interval is
    assigned to the period containing its START instant (interval-end
    timestamp minus the interval length), converted to the site's real
    local time. start_hour/end_hour are therefore local hours as published
    by the network/retailer, not AEST hours.
    """
    start_hour: int       # 0-23
    end_hour: int         # 1-24
    c_per_kwh: float

    def __post_init__(self):
        if not 0 <= self.start_hour < 24:
            raise ValueError(f"start_hour must be 0-23, got {self.start_hour}")
        if not 1 <= self.end_hour <= 24:
            raise ValueError(f"end_hour must be 1-24, got {self.end_hour}")
        if self.start_hour >= self.end_hour:
            raise ValueError(
                f"start_hour ({self.start_hour}) must be < "
                f"end_hour ({self.end_hour})"
            )


@dataclass(frozen=True)
class NetworkTariff:
    """Network charges and loss factors for a connection point.

    Attributes:
        name: Human-readable identifier.
        state: NEM region.
        usage_tariffs: Time-varying network import rates (weekday or all-days).
        usage_tariffs_weekend: Optional weekend variant of usage rates.
        feedin_tariffs: Network export rates (may include penalties/rewards;
            e.g. Ausgrid EA029).
        daily_c_per_day: Combined network daily fixed charge (sum of access,
            metering, export access etc - summed by the loader from named
            sub-components in the YAML).
        usage_dlf: Distribution loss factor applied to imports (typically >1).
        export_dlf: Distribution loss factor applied to exports (typically <1).
        import_tlf: Transmission loss factor applied to imports.
        export_tlf: Transmission (generation) loss factor applied to exports.

    Loss factors are stored raw; billing combines them as needed:
        import effective = usage_dlf * import_tlf
        export effective = export_dlf * export_tlf
    """
    name: str
    state: str
    usage_tariffs: tuple[TariffPeriod, ...]
    usage_tariffs_weekend: Optional[tuple[TariffPeriod, ...]]
    feedin_tariffs: tuple[TariffPeriod, ...]
    daily_c_per_day: float
    usage_dlf: float
    export_dlf: float
    import_tlf: float
    export_tlf: float

    @property
    def import_loss_factor(self) -> float:
        return self.usage_dlf * self.import_tlf

    @property
    def export_loss_factor(self) -> float:
        return self.export_dlf * self.export_tlf


@dataclass(frozen=True)
class RetailTariff:
    """What the customer is billed by their retailer.

    Three types:
      'tou': customer pays time-of-use period rates on imports and receives
             feedin period rates on exports. usage_other / feedin_other are
             flat per-kWh adders (e.g. greenpower). Wholesale price not used
             in the household bill.
      'wholesale_exposed': customer pays the wholesale spot price directly
             (captured in the wholesale layer, not here) plus flat per-kWh
             adders (usage_other = market/environmental cost recovery) and a
             daily charge (subscription). No period rates.
      'mixed': customer pays time-of-use period rates on imports, as for
             'tou', with the TOU rate bundling energy and network. Exports
             are credited max(floor, wholesale_export_share * V) per
             interval, where V is the loss-adjusted wholesale export rate
             ('wholesale' basis) or that plus the loss-adjusted network
             export rate ('wholesale_and_network'). No feedin period rates
             and no advanced features. Optimisers dispatch it against
             prices that differ from these billed rates (see
             prices.build_price_vectors).

    All types may carry flat per-kWh adders (usage_other_c_per_kwh,
    feedin_other_c_per_kwh). The type only governs whether energy is charged
    at TOU period rates (tou) or via wholesale exposure (wholesale_exposed).

    Attributes:
        name: Human-readable identifier.
        state: NEM region.
        type: 'tou', 'wholesale_exposed' or 'mixed'.
        daily_c_per_day: Retail daily charge (supply charge for TOU,
            subscription for wholesale_exposed, plan daily charge for
            mixed).
        usage_tariffs: TOU import periods. Empty for wholesale_exposed.
        usage_tariffs_weekend: Optional TOU weekend variant. None for
            wholesale_exposed.
        feedin_tariffs: TOU export periods. Empty for wholesale_exposed
            and mixed.
        usage_other_c_per_kwh: Flat per-kWh adder on imports.
        feedin_other_c_per_kwh: Flat per-kWh adder/adjustment on exports.

    Advanced feature fields (all optional; None = feature absent):
        block_import_tiers: Stepped import rates for daily time windows.
        block_export_tiers: Stepped export rates for daily time windows.
        zero_import_reward: Daily credit for avoiding window imports.
        critical_export_rate: Replacement export rate at high wholesale prices.
        critical_import_rate: Replacement import rate at low wholesale prices.
        vpp: VPP contract terms (rewards, cap, event classification parameters).
            Present only on plans that include VPP participation.

    Mixed-plan fields (None on every other type):
        wholesale_export_share: Fraction of the export value V credited to
            the household, in [0, 1].
        export_share_basis: 'wholesale' (V = loss-adjusted spot) or
            'wholesale_and_network' (V also includes the loss-adjusted
            network export rate).
        export_floor_c_per_kwh: Floor on wholesale_export_share * V, c/kWh,
            applied every interval. None means no floor.

    Wholesale-exposed plans only:
        gst_on_wholesale_import: True when the household pays GST on the
            spot energy charge for imports (billing.GST_RATE x the
            wholesale usage cost, at negative prices too). Billing adds it
            to the retail usage cost; dispatch prices leave it out. False on
            operator tariffs, whose wholesale exposure is GST-exclusive.
            Always False on 'tou' and 'mixed'.
    """
    name: str
    state: str
    type: RetailTariffType
    daily_c_per_day: float
    usage_tariffs: tuple[TariffPeriod, ...]
    usage_tariffs_weekend: Optional[tuple[TariffPeriod, ...]]
    feedin_tariffs: tuple[TariffPeriod, ...]
    usage_other_c_per_kwh: float
    feedin_other_c_per_kwh: float

    # Advanced features (all optional; None = feature absent).
    # Appended after all existing fields so existing construction sites are
    # unaffected -- these default to None and are ignored by standard paths.
    block_import_tiers: Optional[tuple[BlockTariffWindow, ...]] = None
    block_export_tiers: Optional[tuple[BlockTariffWindow, ...]] = None
    zero_import_reward: Optional[ZeroImportReward] = None
    critical_export_rate: Optional[CriticalRate] = None
    critical_import_rate: Optional[CriticalRate] = None
    vpp: Optional[VppConfig] = None

    # Mixed plans only.
    wholesale_export_share: Optional[float] = None
    export_share_basis: Optional[ExportShareBasis] = None
    export_floor_c_per_kwh: Optional[float] = None

    # Wholesale-exposed plans only.
    gst_on_wholesale_import: bool = False

    @property
    def has_advanced_features(self) -> bool:
        """True if any advanced (block/critical/reward/vpp) feature is present."""
        return any([
            self.block_import_tiers,
            self.block_export_tiers,
            self.zero_import_reward,
            self.critical_export_rate,
            self.critical_import_rate,
            self.vpp,
        ])

    @property
    def has_critical_rates(self) -> bool:
        """True if either critical rate is present."""
        return (self.critical_export_rate is not None
                or self.critical_import_rate is not None)

    @property
    def has_vpp(self) -> bool:
        """True if this plan includes VPP participation terms."""
        return self.vpp is not None


# ----------------------------------------------------------------------
# Period parsing and validation
# ----------------------------------------------------------------------

def _parse_periods(period_list: list) -> tuple[TariffPeriod, ...]:
    """Parse a YAML period list into TariffPeriods.

    Accepts either mapping form {start_hour, end_hour, c_per_kwh} or
    sequence form [start_hour, end_hour, c_per_kwh] (tuple-style, matching
    the source convention).
    """
    out = []
    for p in period_list:
        if isinstance(p, dict):
            out.append(TariffPeriod(**p))
        else:
            start, end, rate = p
            out.append(TariffPeriod(start_hour=int(start), end_hour=int(end),
                                    c_per_kwh=float(rate)))
    return tuple(out)


def _validate_period_coverage(periods: tuple[TariffPeriod, ...], label: str) -> None:
    """Check periods tile [0, 24) exactly: no overlap, no gap."""
    if not periods:
        raise ValueError(f"{label}: at least one period required")
    sorted_periods = sorted(periods, key=lambda p: p.start_hour)
    if sorted_periods[0].start_hour != 0:
        raise ValueError(
            f"{label}: first period must start at hour 0, "
            f"got {sorted_periods[0].start_hour}"
        )
    if sorted_periods[-1].end_hour != 24:
        raise ValueError(
            f"{label}: last period must end at hour 24, "
            f"got {sorted_periods[-1].end_hour}"
        )
    for i in range(len(sorted_periods) - 1):
        if sorted_periods[i].end_hour != sorted_periods[i + 1].start_hour:
            raise ValueError(
                f"{label}: gap or overlap between period ending at "
                f"{sorted_periods[i].end_hour} and period starting at "
                f"{sorted_periods[i + 1].start_hour}"
            )


def period_rate_series(
    periods: tuple[TariffPeriod, ...],
    index: pd.DatetimeIndex,
    interval_hours: float,
    state: str,
    weekend_periods: Optional[tuple[TariffPeriod, ...]] = None,
    lwc: Optional[LocalWallClock] = None,
) -> pd.Series:
    """Map a list of time-of-day periods onto a 5-min interval index.

    Returns a Series of c/kWh rates aligned to `index`. The rate for each
    interval is the rate of the period containing the interval's START
    instant (index timestamp minus interval_hours), converted to the
    site's local time for `state` (see core/localtime.py) -- tariff periods
    are specified by DNSPs and retailers in real local time, not AEST.

    If weekend_periods is given, weekend intervals (Saturday/Sunday by the
    local interval-start date) use weekend_periods; weekday intervals use
    periods.

    This single function serves network usage, network feedin, retail TOU
    usage, and retail TOU feedin lookups - any time-of-day period mapping.

    Args:
        lwc: Optional precomputed LocalWallClock for (index, state,
            interval_hours). build_layer_rates calls this function up to 4
            times per invocation (network usage/feedin, retail usage/feedin);
            passing a precomputed lwc avoids 4 redundant local_wall_clock
            calls (each of which does a real tz_convert) doing the exact
            same conversion. Computed internally if not given, so existing
            callers are unaffected.
    """
    if lwc is None:
        lwc = local_wall_clock(index, state, interval_hours)
    start_hour = lwc.hour
    is_weekend = lwc.is_weekend

    rates = np.full(len(index), np.nan, dtype=float)

    def _apply(period_set, mask):
        for p in period_set:
            sel = mask & (start_hour >= p.start_hour) & (start_hour < p.end_hour)
            rates[sel] = p.c_per_kwh

    if weekend_periods is not None:
        _apply(periods, ~is_weekend)
        _apply(weekend_periods, is_weekend)
    else:
        _apply(periods, np.ones(len(index), dtype=bool))

    if np.isnan(rates).any():
        n_missing = int(np.isnan(rates).sum())
        raise ValueError(
            f"period_rate_series: {n_missing} intervals did not match any "
            f"period - periods do not cover all hours present in the index."
        )

    return pd.Series(rates, index=index, name='c_per_kwh')


# ----------------------------------------------------------------------
# Advanced block window helpers
# ----------------------------------------------------------------------

def _validate_block_windows(
    windows: tuple[BlockTariffWindow, ...],
    label: str,
) -> None:
    """Validate a list of BlockTariffWindows: no two windows may overlap.

    Tier ordering and inf-final-tier are already enforced by
    BlockTariffWindow.__post_init__. This function only checks inter-window
    overlap, which requires the full list.
    """
    if not windows:
        return
    sorted_wins = sorted(windows, key=lambda w: w.start_hour)
    for i in range(len(sorted_wins) - 1):
        if sorted_wins[i].end_hour > sorted_wins[i + 1].start_hour:
            raise ValueError(
                f"{label}: block windows overlap: "
                f"[{sorted_wins[i].start_hour}, {sorted_wins[i].end_hour}) "
                f"and [{sorted_wins[i + 1].start_hour}, "
                f"{sorted_wins[i + 1].end_hour})"
            )


def _validate_block_tier_convexity(
    windows: tuple[BlockTariffWindow, ...],
    is_import: bool,
    label: str,
) -> None:
    """Validate that block tier rates are convex (LP-compatible).

    Import windows must have non-decreasing rates (cheap tier first).
    Export windows must have non-increasing rates (high-reward tier first).
    Raises ValueError so the LP cannot silently misallocate energy across tiers.
    """
    for w in windows:
        rates = [t.rate_c_per_kwh for t in w.tiers]
        if is_import:
            for i in range(len(rates) - 1):
                if rates[i] > rates[i + 1]:
                    raise ValueError(
                        f"{label}: block import window [{w.start_hour}, "
                        f"{w.end_hour}) has non-convex rates (must be "
                        f"non-decreasing for LP correctness): {rates}"
                    )
        else:
            for i in range(len(rates) - 1):
                if rates[i] < rates[i + 1]:
                    raise ValueError(
                        f"{label}: block export window [{w.start_hour}, "
                        f"{w.end_hour}) has non-convex rates (must be "
                        f"non-increasing for LP correctness): {rates}"
                    )


# ----------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------

def _read_yaml(tariff_id: str, tariff_dir: Path) -> tuple[dict, Path]:
    path = tariff_dir / f"{tariff_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Tariff file not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    return data, path


def load_network_tariff(tariff_id: str, tariff_dir: Path) -> NetworkTariff:
    """Load a NetworkTariff from `{tariff_dir}/{tariff_id}.yaml`.

    The YAML must have tariff_class: network. Daily charge sub-components
    listed under daily_charges_c_per_day (a mapping of named components)
    are summed into daily_c_per_day, preserving the breakdown in the file
    for auditability.
    """
    data, path = _read_yaml(tariff_id, tariff_dir)

    if data.get('tariff_class') != 'network':
        raise ValueError(
            f"{path}: expected tariff_class: network, got "
            f"{data.get('tariff_class')!r}"
        )

    for key in ('name', 'state', 'usage_tariffs', 'feedin_tariffs',
                'usage_dlf', 'export_dlf', 'import_tlf', 'export_tlf'):
        if key not in data:
            raise ValueError(f"{path}: missing required field '{key}'")

    usage_tariffs = _parse_periods(data['usage_tariffs'])
    usage_weekend = (_parse_periods(data['usage_tariffs_weekend'])
                     if 'usage_tariffs_weekend' in data else None)
    feedin_tariffs = _parse_periods(data['feedin_tariffs'])

    _validate_period_coverage(usage_tariffs, f'{path}:usage_tariffs')
    if usage_weekend is not None:
        _validate_period_coverage(usage_weekend, f'{path}:usage_tariffs_weekend')
    _validate_period_coverage(feedin_tariffs, f'{path}:feedin_tariffs')

    # Sum named daily charge sub-components
    daily_components = data.get('daily_charges_c_per_day', {})
    if not isinstance(daily_components, dict) or not daily_components:
        raise ValueError(
            f"{path}: network tariff requires daily_charges_c_per_day as a "
            f"mapping of named components (e.g. access, metering, export_access)"
        )
    daily_total = float(sum(daily_components.values()))

    return NetworkTariff(
        name=data['name'],
        state=data['state'],
        usage_tariffs=usage_tariffs,
        usage_tariffs_weekend=usage_weekend,
        feedin_tariffs=feedin_tariffs,
        daily_c_per_day=daily_total,
        usage_dlf=float(data['usage_dlf']),
        export_dlf=float(data['export_dlf']),
        import_tlf=float(data['import_tlf']),
        export_tlf=float(data['export_tlf']),
    )


def _parse_block_windows(
    data: dict,
    key: str,
    path: Path,
) -> Optional[tuple[BlockTariffWindow, ...]]:
    """Parse a list of block tariff windows from YAML data."""
    raw = data.get(key)
    if not raw:
        return None
    windows = []
    for entry in raw:
        tiers = tuple(
            BlockTier(
                threshold_kwh=float(entry_tier['threshold_kwh']),
                rate_c_per_kwh=float(entry_tier['rate_c_per_kwh']),
            )
            for entry_tier in entry['tiers']
        )
        windows.append(BlockTariffWindow(
            start_hour=int(entry['start_hour']),
            end_hour=int(entry['end_hour']),
            tiers=tiers,
            applies_weekends=bool(entry.get('applies_weekends', True)),
        ))
    return tuple(windows)


def _parse_zero_import_reward(
    data: dict,
    path: Path,
) -> Optional[ZeroImportReward]:
    """Parse a zero_import_reward section from YAML data."""
    raw = data.get('zero_import_reward')
    if not raw:
        return None
    return ZeroImportReward(
        start_hour=int(raw['start_hour']),
        end_hour=int(raw['end_hour']),
        reward_c_per_day=float(raw['reward_c_per_day']),
        tolerance_kwh_per_hour=float(raw.get('tolerance_kwh_per_hour', 0.0)),
    )


def _parse_critical_rate(
    data: dict,
    key: str,
    path: Path,
) -> Optional[CriticalRate]:
    """Parse a critical rate section from YAML data."""
    raw = data.get(key)
    if not raw:
        return None
    if 'bonus_c_per_kwh' in raw or 'rate_c_per_kwh' not in raw:
        raise ValueError(
            f"{path}:{key} needs rate_c_per_kwh (the replacement rate); "
            f"bonus_c_per_kwh is not accepted."
        )
    return CriticalRate(
        threshold_dollars_per_mwh=float(raw['threshold_dollars_per_mwh']),
        rate_c_per_kwh=float(raw['rate_c_per_kwh']),
    )


def _parse_charge_windows(raw_value) -> tuple[ChargeWindow, ...]:
    """Parse the no_grid_charge_windows YAML value into ChargeWindows.

    Accepts the same dual form as the tariff period lists: a single flat pair
    [start_hour, end_hour] for one window, or a list of pairs
    [[s1, e1], [s2, e2], ...] for several. Disambiguates on the first element
    (an int means a single flat pair). Per-window hour validation is performed
    by ChargeWindow.__post_init__.
    """
    if not raw_value:
        raise ValueError(
            "no_grid_charge_windows must be a [start, end] pair or a list of "
            f"such pairs, got {raw_value!r}"
        )
    # Single flat pair: first element is a scalar (int hour).
    if isinstance(raw_value[0], (int, float)):
        pairs = [raw_value]
    else:
        pairs = list(raw_value)
    out = []
    for p in pairs:
        start, end = p
        out.append(ChargeWindow(start_hour=int(start), end_hour=int(end)))
    return tuple(out)


def _parse_vpp_config(
    data: dict,
    path: Path,
) -> Optional[VppConfig]:
    """Parse a vpp section from YAML data."""
    raw = data.get('vpp')
    if not raw:
        return None
    # Floor a zero import reward to a small positive value to preserve the
    # natural/forced split tie-break in the LP (see VppConfig docstring).
    _IMP_REWARD_FLOOR = 0.01  # c/kWh
    imp_reward = float(raw.get('imp_reward_c_per_kwh', 0.0))
    if imp_reward == 0.0:
        imp_reward = _IMP_REWARD_FLOOR
    return VppConfig(
        exp_reward_c_per_kwh=float(raw['exp_reward_c_per_kwh']),
        imp_reward_c_per_kwh=imp_reward,
        kwh_limit=float(raw['kwh_limit']),
        limit_applies_to=str(raw['limit_applies_to']),
        default_mode=str(raw.get('default_mode', 'self_consumption')),
        event_window_hours=float(raw['event_window_hours']),
        vpp_threshold_dollars_per_mwh=float(raw['vpp_threshold_dollars_per_mwh']),
        curtail_c_per_kwh=(float(raw['curtail_c_per_kwh'])
                           if 'curtail_c_per_kwh' in raw else None),
        min_export_price_dollars_per_mwh=(
            float(raw['min_export_price_dollars_per_mwh'])
            if 'min_export_price_dollars_per_mwh' in raw else None),
        min_export_price_mode=str(
            raw.get('min_export_price_mode', 'fixed')),
        no_grid_charge_windows=(
            _parse_charge_windows(raw['no_grid_charge_windows'])
            if 'no_grid_charge_windows' in raw else None),
    )


def _parse_wholesale_export_share(data: dict, path: Path,
                                  rtype: str) -> Optional[float]:
    """wholesale_export_share: required scalar in [0, 1] on a mixed tariff,
    rejected on any other type."""
    raw = data.get('wholesale_export_share')
    if rtype != 'mixed':
        if raw is not None:
            raise ValueError(
                f"{path}: wholesale_export_share is only valid on a mixed "
                f"tariff, not {rtype!r}"
            )
        return None
    if raw is None:
        raise ValueError(f"{path}: mixed tariff requires wholesale_export_share")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(
            f"{path}: wholesale_export_share must be a number, got {raw!r}"
        )
    share = float(raw)
    if not 0.0 <= share <= 1.0:
        raise ValueError(
            f"{path}: wholesale_export_share must be in [0, 1], got {share}"
        )
    return share


def _parse_export_share_basis(data: dict, path: Path,
                              rtype: str) -> Optional[str]:
    """export_share_basis: required on a mixed tariff, one of
    'wholesale' or 'wholesale_and_network'; rejected on any other type."""
    present = 'export_share_basis' in data
    if rtype != 'mixed':
        if present:
            raise ValueError(
                f"{path}: export_share_basis is only valid on a mixed "
                f"tariff, not {rtype!r}"
            )
        return None
    if not present:
        raise ValueError(f"{path}: mixed tariff requires export_share_basis")
    raw = data['export_share_basis']
    if raw not in ('wholesale', 'wholesale_and_network'):
        raise ValueError(
            f"{path}: export_share_basis must be 'wholesale' or "
            f"'wholesale_and_network', got {raw!r}"
        )
    return raw


def _parse_export_floor(data: dict, path: Path,
                        rtype: str) -> Optional[float]:
    """export_floor_c_per_kwh: the key is required on a mixed tariff and
    rejected on any other type. Its value is a finite number (c/kWh) or
    null for no floor."""
    present = 'export_floor_c_per_kwh' in data
    if rtype != 'mixed':
        if present:
            raise ValueError(
                f"{path}: export_floor_c_per_kwh is only valid on a mixed "
                f"tariff, not {rtype!r}"
            )
        return None
    if not present:
        raise ValueError(
            f"{path}: mixed tariff requires export_floor_c_per_kwh "
            f"(null for no floor)"
        )
    raw = data['export_floor_c_per_kwh']
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(
            f"{path}: export_floor_c_per_kwh must be a number or null, "
            f"got {raw!r}"
        )
    floor = float(raw)
    if not math.isfinite(floor):
        raise ValueError(
            f"{path}: export_floor_c_per_kwh must be finite, got {floor}"
        )
    return floor


def _parse_gst_on_wholesale_import(data: dict, path: Path,
                                   rtype: str) -> bool:
    """gst_on_wholesale_import: required boolean on a wholesale_exposed
    tariff, rejected on any other type."""
    present = 'gst_on_wholesale_import' in data
    if rtype != 'wholesale_exposed':
        if present:
            raise ValueError(
                f"{path}: gst_on_wholesale_import is only valid on a "
                f"wholesale_exposed tariff, not {rtype!r}"
            )
        return False
    if not present:
        raise ValueError(
            f"{path}: wholesale_exposed tariff requires "
            f"gst_on_wholesale_import (true for a household plan, false for "
            f"an operator tariff)"
        )
    raw = data['gst_on_wholesale_import']
    if not isinstance(raw, bool):
        raise ValueError(
            f"{path}: gst_on_wholesale_import must be true or false, "
            f"got {raw!r}"
        )
    return raw


def load_retail_tariff(tariff_id: str, tariff_dir: Path) -> RetailTariff:
    """Load a RetailTariff from `{tariff_dir}/{tariff_id}.yaml`.

    The YAML must have tariff_class: retail and type tou, wholesale_exposed
    or mixed.
    Named per-kWh adder sub-components under usage_other_c_per_kwh /
    feedin_other_c_per_kwh (mappings) are summed, preserving the breakdown.
    Daily charge sub-components under daily_charges_c_per_day are summed.
    """
    data, path = _read_yaml(tariff_id, tariff_dir)

    if data.get('tariff_class') != 'retail':
        raise ValueError(
            f"{path}: expected tariff_class: retail, got "
            f"{data.get('tariff_class')!r}"
        )

    rtype = data.get('type')
    if rtype not in ('tou', 'wholesale_exposed', 'mixed'):
        raise ValueError(
            f"{path}: retail type must be 'tou', 'wholesale_exposed' or "
            f"'mixed', got {rtype!r}"
        )

    for key in ('name', 'state'):
        if key not in data:
            raise ValueError(f"{path}: missing required field '{key}'")

    # Daily charges (summed from named components)
    daily_components = data.get('daily_charges_c_per_day', {})
    if not isinstance(daily_components, dict) or not daily_components:
        raise ValueError(
            f"{path}: retail tariff requires daily_charges_c_per_day as a "
            f"mapping of named components"
        )
    daily_total = float(sum(daily_components.values()))

    # Flat per-kWh adders (summed from named components; optional)
    def _sum_adder(field):
        comp = data.get(field, {})
        if comp == {} or comp is None:
            return 0.0
        if not isinstance(comp, dict):
            raise ValueError(f"{path}: {field} must be a mapping of named components")
        return float(sum(comp.values()))

    usage_other = _sum_adder('usage_other_c_per_kwh')
    feedin_other = _sum_adder('feedin_other_c_per_kwh')

    if rtype in ('tou', 'mixed'):
        if 'usage_tariffs' not in data:
            raise ValueError(f"{path}: {rtype} tariff requires usage_tariffs")
        usage_tariffs = _parse_periods(data['usage_tariffs'])
        usage_weekend = (_parse_periods(data['usage_tariffs_weekend'])
                         if 'usage_tariffs_weekend' in data else None)
        _validate_period_coverage(usage_tariffs, f'{path}:usage_tariffs')
        if usage_weekend is not None:
            _validate_period_coverage(usage_weekend, f'{path}:usage_tariffs_weekend')
        if rtype == 'tou':
            feedin_tariffs = _parse_periods(data.get('feedin_tariffs', []))
            if feedin_tariffs:
                _validate_period_coverage(feedin_tariffs, f'{path}:feedin_tariffs')
        else:  # mixed
            if data.get('feedin_tariffs'):
                raise ValueError(
                    f"{path}: mixed tariff must not define feedin_tariffs - "
                    f"exports are credited a floored wholesale_export_share "
                    f"of their real-time value."
                )
            feedin_tariffs = ()
    else:  # wholesale_exposed
        # No period rates; energy charged via wholesale layer.
        if data.get('usage_tariffs') or data.get('feedin_tariffs'):
            raise ValueError(
                f"{path}: wholesale_exposed tariff must not define period "
                f"rates (usage_tariffs/feedin_tariffs) - energy is charged "
                f"via the wholesale layer."
            )
        usage_tariffs = ()
        usage_weekend = None
        feedin_tariffs = ()

    wholesale_export_share = _parse_wholesale_export_share(data, path, rtype)
    export_share_basis = _parse_export_share_basis(data, path, rtype)
    export_floor_c_per_kwh = _parse_export_floor(data, path, rtype)

    # --- Advanced features (all optional) ---

    def _load_block_import_tiers(data, path, rtype):
        windows = _parse_block_windows(data, 'block_import_tiers', path)
        if windows is None:
            return None
        _validate_block_windows(windows, f'{path}:block_import_tiers')
        _validate_block_tier_convexity(
            windows, is_import=True, label=f'{path}:block_import_tiers')
        if rtype == 'wholesale_exposed':
            warnings.warn(
                f"{path}: block_import_tiers on a wholesale_exposed tariff "
                f"is unusual; the block rate is treated as a retail overlay "
                f"on top of wholesale+network components.",
                UserWarning, stacklevel=4,
            )
        # Warn if any block window overlaps a non-zero usage_tariffs period
        if rtype == 'tou' and usage_tariffs:
            for w in windows:
                for p in usage_tariffs:
                    if (p.c_per_kwh != 0.0
                            and w.start_hour < p.end_hour
                            and w.end_hour > p.start_hour):
                        warnings.warn(
                            f"{path}: block_import_tiers window "
                            f"[{w.start_hour}, {w.end_hour}) overlaps "
                            f"usage_tariffs period [{p.start_hour}, "
                            f"{p.end_hour}) at {p.c_per_kwh} c/kWh; "
                            f"the block rate overrides the usage_tariffs "
                            f"rate for those intervals.",
                            UserWarning, stacklevel=4,
                        )
        return windows

    def _load_block_export_tiers(data, path, rtype):
        windows = _parse_block_windows(data, 'block_export_tiers', path)
        if windows is None:
            return None
        _validate_block_windows(windows, f'{path}:block_export_tiers')
        _validate_block_tier_convexity(
            windows, is_import=False, label=f'{path}:block_export_tiers')
        if rtype == 'wholesale_exposed':
            warnings.warn(
                f"{path}: block_export_tiers on a wholesale_exposed tariff "
                f"is unusual; the block rate is treated as a retail overlay.",
                UserWarning, stacklevel=4,
            )
        return windows

    retired = [k for k in ('stress_export_amplifier', 'stress_import_amplifier')
               if k in data]
    if retired:
        raise ValueError(
            f"{path}: {', '.join(retired)} is no longer read. Use "
            f"critical_export_rate / critical_import_rate with rate_c_per_kwh, "
            f"which replaces the export or import rate rather than adding to it "
            f"(perfect_foresight_advanced_spec.md)."
        )

    tariff = RetailTariff(
        name=data['name'],
        state=data['state'],
        type=rtype,
        daily_c_per_day=daily_total,
        usage_tariffs=usage_tariffs,
        usage_tariffs_weekend=usage_weekend,
        feedin_tariffs=feedin_tariffs,
        usage_other_c_per_kwh=usage_other,
        feedin_other_c_per_kwh=feedin_other,
        block_import_tiers=_load_block_import_tiers(data, path, rtype),
        block_export_tiers=_load_block_export_tiers(data, path, rtype),
        zero_import_reward=_parse_zero_import_reward(data, path),
        critical_export_rate=_parse_critical_rate(
            data, 'critical_export_rate', path),
        critical_import_rate=_parse_critical_rate(
            data, 'critical_import_rate', path),
        vpp=_parse_vpp_config(data, path),
        wholesale_export_share=wholesale_export_share,
        export_share_basis=export_share_basis,
        export_floor_c_per_kwh=export_floor_c_per_kwh,
        gst_on_wholesale_import=_parse_gst_on_wholesale_import(
            data, path, rtype),
    )
    # build_base_price_vectors_advanced and billing's advanced branch
    # distinguish only tou and wholesale_exposed, so an advanced feature on a
    # mixed plan would run with no defined meaning.
    if rtype == 'mixed' and tariff.has_advanced_features:
        raise ValueError(
            f"{path}: mixed tariff must not define advanced features "
            f"(block tiers, zero-import reward, critical rates, vpp)."
        )
    if tariff.has_critical_rates and rtype != 'tou':
        raise ValueError(
            f"{path}: critical_export_rate / critical_import_rate are "
            f"defined for tou plans only; got type {rtype!r}."
        )
    cx, cm = tariff.critical_export_rate, tariff.critical_import_rate
    if (cx is not None and cm is not None
            and not cx.threshold_dollars_per_mwh > cm.threshold_dollars_per_mwh):
        raise ValueError(
            f"{path}: critical_export_rate threshold "
            f"({cx.threshold_dollars_per_mwh}) must be above the "
            f"critical_import_rate threshold "
            f"({cm.threshold_dollars_per_mwh}), so no interval is critical "
            f"on both sides."
        )
    return tariff
