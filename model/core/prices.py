"""Shared per-interval price construction for billing and dispatch.

Single source of truth for turning tariffs + the wholesale price series into
per-interval $/kWh rates. This module owns *every* pricing unit conversion
($/MWh -> $/kWh, c/kWh -> $/kWh) and applies the network loss factors, so that
no other module applies them independently. Both `billing.py` and the
perfect-foresight optimiser consume these vectors, which guarantees the
optimiser optimises against exactly the prices billing charges.

Layers (all $/kWh, per interval, with loss factors already applied where they
apply):
  - wholesale : spot price / 1000, x loss factor (import or export)
  - network   : period rate / 100,  x loss factor (import or export)
  - retail    : period rate / 100 (TOU) + flat adder; NO loss factor

Sign of a rate is preserved. A negative network feedin rate is an export
penalty; a negative spot price is a negative wholesale rate. The credit sign
convention (a credit is stored as a negative dollar amount) is NOT applied
here - it lives in billing, which applies these rates to the flows. Here the
export ("feedin") vectors are stored as the value EARNED per exported kWh:
positive means the customer is paid, negative means the customer pays. So the
optimiser's objective is

    minimise  sum_t [ imp[t] * p_imp[t] - exp[t] * p_exp[t] ]

and billing computes feedin credits as `-sum(exp * <export vector>)`.

SIGN / UNIT CONVENTIONS (canonical version in docs/model_spec.md)
-----------------------------------------------------------------
- wholesale price: $/MWh / 1000 -> $/kWh
- tariff rates:    c/kWh  / 100  -> $/kWh
- daily charges:   c/day  / 100  -> $/day   (done by the caller; not a
  per-interval rate and not shared with the optimiser)
- import effective loss factor = usage_dlf * import_tlf  (NetworkTariff property)
- export effective loss factor = export_dlf * export_tlf (NetworkTariff property)
- loss factors apply to the wholesale and network layers only, never retail.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from model.core.tariff import (
    BlockTariffWindow, ChargeWindow, NetworkTariff, RetailTariff,
    period_rate_series,
)
from model.core.localtime import local_wall_clock, LocalWallClock
from model.core.market_prices import (
    MARKET_PRICE_FLOOR_MWH as _MARKET_PRICE_FLOOR_MWH,
    widest_market_price_cap,
)

# Pricing unit conversions live here so no other module hard-codes them.
MWH_TO_KWH = 1000.0    # divide $/MWh by this to get $/kWh
C_TO_DOLLARS = 100.0   # divide c/kWh or c/day by this to get $/kWh or $/day


@dataclass(frozen=True)
class LayerRates:
    """Per-interval $/kWh rates by settlement layer, loss factors applied.

    Every array is length-T, aligned to the index passed to
    `build_layer_rates`. Import vectors are the cost per imported kWh; export
    ("feedin") vectors are the value earned per exported kWh (positive = the
    customer is paid; negative = the customer pays, e.g. an export penalty or a
    negative spot price). The retail vectors carry the flat per-kWh adders and
    are NOT loss-adjusted, except a mixed plan's retail_export, which is a
    floored share of the loss-adjusted wholesale (and optionally network)
    export rate (see mixed_export_rate).
    """
    wholesale_import: np.ndarray
    wholesale_export: np.ndarray
    network_import: np.ndarray
    network_export: np.ndarray
    retail_import: np.ndarray
    retail_export: np.ndarray


def _validate_prices(index: pd.DatetimeIndex, prices_mwh: pd.Series) -> None:
    if not prices_mwh.index.equals(index):
        raise ValueError(
            "prices index does not match the target index "
            f"(prices len={len(prices_mwh)}, index len={len(index)})."
        )
    if prices_mwh.isna().any():
        n = int(prices_mwh.isna().sum())
        raise ValueError(
            f"prices contain {n} NaN values over the period. Missing prices "
            f"must be cleaned in the price pipeline before pricing."
        )


def mixed_export_rate(
    wholesale_export: np.ndarray,
    network_export: np.ndarray,
    retail_tariff: RetailTariff,
) -> np.ndarray:
    """A mixed plan's household export credit rate, $/kWh per interval.

        max(floor, share * V) + feedin_other

    V is the loss-adjusted wholesale export rate under export_share_basis
    'wholesale', and that plus the loss-adjusted network export rate under
    'wholesale_and_network'. The floor (export_floor_c_per_kwh) applies to
    each interval separately; None means no floor.

    The single definition of the rate: build_layer_rates (billing) and the
    mixed dispatch price construction both call it, so the export price an
    optimiser sees and the export credit billing pays cannot drift apart.

    Args:
        wholesale_export: RRP/1000 * export_lf, $/kWh.
        network_export: network feed-in rate * export_lf, $/kWh.
        retail_tariff: a 'mixed' retail tariff.

    Raises:
        ValueError: if retail_tariff is not 'mixed'.
    """
    if retail_tariff.type != 'mixed':
        raise ValueError(
            f"mixed_export_rate needs a 'mixed' retail tariff, got "
            f"{retail_tariff.type!r} ({retail_tariff.name})"
        )
    if retail_tariff.export_share_basis == 'wholesale':
        value = np.asarray(wholesale_export, dtype=float)
    elif retail_tariff.export_share_basis == 'wholesale_and_network':
        value = (np.asarray(wholesale_export, dtype=float)
                 + np.asarray(network_export, dtype=float))
    else:
        raise ValueError(
            f"unknown export_share_basis "
            f"{retail_tariff.export_share_basis!r} ({retail_tariff.name})"
        )
    credit = retail_tariff.wholesale_export_share * value
    if retail_tariff.export_floor_c_per_kwh is not None:
        credit = np.maximum(
            retail_tariff.export_floor_c_per_kwh / C_TO_DOLLARS, credit)
    return credit + retail_tariff.feedin_other_c_per_kwh / C_TO_DOLLARS


def build_layer_rates(
    index: pd.DatetimeIndex,
    prices_mwh: pd.Series,
    network_tariff: NetworkTariff,
    retail_tariff: RetailTariff,
    state: str,
    interval_hours: float = 5 / 60,
    lwc: Optional[LocalWallClock] = None,
) -> LayerRates:
    """Build the per-layer per-interval $/kWh rate vectors.

    Args:
        index: target interval index (datetime_aest, interval-end timestamps).
        prices_mwh: NEM RRP in $/MWh, indexed identically to `index`.
        network_tariff: network rates + loss factors.
        retail_tariff: retail plan (tou, wholesale_exposed or mixed).
        state: the site's actual location (scenario.state), used to resolve
            local time for tariff period lookups. Deliberately independent
            of network_tariff.state / retail_tariff.state so a cross-state
            tariff sensitivity run still applies the site's real local clock.
        interval_hours: interval length in hours (default 5/60).
        lwc: Optional precomputed LocalWallClock for (index, state,
            interval_hours). This function calls period_rate_series up to 4
            times (network usage/feedin, retail usage/feedin); passing a
            precomputed lwc means all 4 share one local_wall_clock call
            instead of each deriving it independently. Computed internally
            if not given, so existing callers are unaffected. A caller (e.g.
            compute_bill) that also needs lwc for other purposes in the same
            call should compute it once and pass it in here too.

    Returns:
        LayerRates with all six vectors as float ndarrays.

    Raises:
        ValueError: if `prices_mwh` misaligns with `index` or contains NaN.
    """
    _validate_prices(index, prices_mwh)
    n = len(index)
    if lwc is None:
        lwc = local_wall_clock(index, state, interval_hours)

    price_kwh = prices_mwh.to_numpy(dtype=float, copy=False) / MWH_TO_KWH
    import_lf = network_tariff.import_loss_factor
    export_lf = network_tariff.export_loss_factor

    # ---------------- Wholesale layer ----------------
    wholesale_import = price_kwh * import_lf
    wholesale_export = price_kwh * export_lf

    # ---------------- Network layer ----------------
    net_usage = period_rate_series(
        network_tariff.usage_tariffs, index, interval_hours, state,
        network_tariff.usage_tariffs_weekend, lwc=lwc,
    ).to_numpy(dtype=float) / C_TO_DOLLARS
    net_feedin = period_rate_series(
        network_tariff.feedin_tariffs, index, interval_hours, state,
        lwc=lwc,
    ).to_numpy(dtype=float) / C_TO_DOLLARS
    network_import = net_usage * import_lf
    network_export = net_feedin * export_lf

    # ---------------- Retail layer ----------------
    # No loss factors, except a mixed plan's export rate, which is a floored
    # share of the loss-adjusted export rates built above.
    usage_other = retail_tariff.usage_other_c_per_kwh / C_TO_DOLLARS
    feedin_other = retail_tariff.feedin_other_c_per_kwh / C_TO_DOLLARS

    if retail_tariff.type in ('tou', 'mixed'):
        retail_usage = period_rate_series(
            retail_tariff.usage_tariffs, index, interval_hours, state,
            retail_tariff.usage_tariffs_weekend, lwc=lwc,
        ).to_numpy(dtype=float) / C_TO_DOLLARS
        retail_import = retail_usage + usage_other
        if retail_tariff.type == 'tou':
            if retail_tariff.feedin_tariffs:
                retail_feedin = period_rate_series(
                    retail_tariff.feedin_tariffs, index, interval_hours, state,
                    lwc=lwc,
                ).to_numpy(dtype=float) / C_TO_DOLLARS
            else:
                retail_feedin = np.zeros(n)
            retail_export = retail_feedin + feedin_other
        else:  # mixed: a floored share of real-time export value
            retail_export = mixed_export_rate(
                wholesale_export, network_export, retail_tariff)
    elif retail_tariff.type == 'wholesale_exposed':
        # Flat adders only; spot energy is in the wholesale layer.
        retail_import = np.full(n, usage_other)
        retail_export = np.full(n, feedin_other)
    else:
        raise ValueError(
            f"unknown retail tariff type {retail_tariff.type!r} "
            f"({retail_tariff.name})"
        )

    return LayerRates(
        wholesale_import=wholesale_import,
        wholesale_export=wholesale_export,
        network_import=network_import,
        network_export=network_export,
        retail_import=retail_import,
        retail_export=retail_export,
    )


def build_price_vectors(
    index: pd.DatetimeIndex,
    prices_mwh: pd.Series,
    network_tariff: NetworkTariff,
    retail_tariff: RetailTariff,
    state: str,
    interval_hours: float = 5 / 60,
) -> tuple[pd.Series, pd.Series]:
    """Build the (p_imp, p_exp) vectors an optimiser dispatches against.

    On tou and wholesale_exposed plans these are the prices the HOUSEHOLD
    faces, matching the `household_total` view in billing:
      - wholesale_exposed: the full stack (wholesale + network + retail adders).
      - tou: the retail layer only (wholesale and network are retailer-side).
    On a mixed plan they are the mixed-plan dispatch prices
    (MixedDispatchStack): the billed export credit, and an import price that
    adds max(0, RRP) * import_lf to the billed TOU rate. The import side
    therefore differs from what billing charges, by design.

    `p_imp` is the cost per imported kWh (positive = cost). `p_exp` is the
    value per exported kWh (positive = the customer is paid). The optimiser
    objective is `sum_t [ imp[t]*p_imp[t] - exp[t]*p_exp[t] ]`.

    Args:
        state: the site's actual location (scenario.state); see
            build_layer_rates for why this is separate from the tariffs'
            own declared state.

    Returns:
        (p_imp, p_exp) as pd.Series in $/kWh, indexed by `index`.
    """
    if retail_tariff.type == 'mixed':
        _validate_prices(index, prices_mwh)
        stack = build_mixed_dispatch_stack(
            index, network_tariff, retail_tariff, state, interval_hours)
        p_imp, p_exp = stack.window_price_vectors(
            prices_mwh.to_numpy(dtype=float), 0)
        return (
            pd.Series(p_imp, index=index, name='p_imp'),
            pd.Series(p_exp, index=index, name='p_exp'),
        )

    lr = build_layer_rates(
        index, prices_mwh, network_tariff, retail_tariff, state, interval_hours,
    )

    if retail_tariff.type == 'wholesale_exposed':
        p_imp = lr.wholesale_import + lr.network_import + lr.retail_import
        p_exp = lr.wholesale_export + lr.network_export + lr.retail_export
    elif retail_tariff.type == 'tou':
        p_imp = lr.retail_import
        p_exp = lr.retail_export
    else:
        raise ValueError(
            f"unknown retail tariff type {retail_tariff.type!r} "
            f"({retail_tariff.name})"
        )

    return (
        pd.Series(p_imp, index=index, name='p_imp'),
        pd.Series(p_exp, index=index, name='p_exp'),
    )


# ---------------------------------------------------------------------------
# Rolling optimisation price stack
# ---------------------------------------------------------------------------

# NEM market price bounds in $/MWh, used only to bound the price-ordering check
# below. Nothing here clips settlement prices.
#
# The cap is reset every 1 July, so there is no single correct value. The
# price-ordering check is a property of a TARIFF PAIR rather than of any
# particular interval, and must hold whenever that pair is in force, so it uses
# the widest cap on record. Per-year limits live in market_prices.py and are
# what forecast-store ingest validates against.
MARKET_PRICE_FLOOR_MWH = _MARKET_PRICE_FLOOR_MWH
MARKET_PRICE_CAP_MWH = widest_market_price_cap()


@dataclass(frozen=True)
class PriceStackOffsets:
    """The price-independent part of the customer price stack.

    For a wholesale_exposed plan, build_price_vectors is affine in RRP:

        p_imp[t] = (rrp[t]/1000) * import_loss_factor + network_import[t] + retail_import[t]
        p_exp[t] = (rrp[t]/1000) * export_loss_factor + network_export[t] + retail_export[t]

    Only the first term depends on price; the rest depends on the target
    interval's clock time alone. rolling_optimisation solves one LP per 5-minute
    interval, so calling build_price_vectors per window would repeat the tariff
    period lookups ~105,000 times per site-year. Splitting the stack once lets
    each window cost a multiply and an add.

    The loss factors are held explicitly rather than folded into a coefficient
    because they come from the NETWORK tariff, not from build_layer_rates, and
    that is the seam where a mistake would be hardest to see.

    Attributes:
        import_loss_factor: network_tariff.import_loss_factor (usage_dlf * import_tlf).
        export_loss_factor: network_tariff.export_loss_factor (export_dlf * export_tlf).
        offset_import: network_import + retail_import, $/kWh, one per interval.
        offset_export: network_export + retail_export, $/kWh, one per interval.
    """
    import_loss_factor: float
    export_loss_factor: float
    offset_import: np.ndarray
    offset_export: np.ndarray

    def __len__(self) -> int:
        return len(self.offset_import)

    def window_price_vectors(
        self, rrp_mwh: np.ndarray, start: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruct (p_imp, p_exp) in $/kWh for one window.

        The single definition of the affine formula. Both rolling_optimisation
        and the equivalence test call this, so they cannot drift apart.

        Args:
            rrp_mwh: forecast or actual RRP in $/MWh for the window, in target
                order.
            start: index of the window's first interval within the full period.

        Returns:
            (p_imp, p_exp) as float ndarrays, same length as rrp_mwh.
        """
        rrp = np.asarray(rrp_mwh, dtype=float)
        stop = start + len(rrp)
        if start < 0 or stop > len(self.offset_import):
            raise ValueError(
                f"window [{start}, {stop}) falls outside the offsets period "
                f"of length {len(self.offset_import)}."
            )
        price_kwh = rrp / MWH_TO_KWH
        return (
            price_kwh * self.import_loss_factor + self.offset_import[start:stop],
            price_kwh * self.export_loss_factor + self.offset_export[start:stop],
        )


def build_price_stack_offsets(
    index: pd.DatetimeIndex,
    network_tariff: NetworkTariff,
    retail_tariff: RetailTariff,
    state: str,
    interval_hours: float = 5 / 60,
    lwc: Optional[LocalWallClock] = None,
    price_floor_mwh: float = MARKET_PRICE_FLOOR_MWH,
    price_cap_mwh: float = MARKET_PRICE_CAP_MWH,
) -> PriceStackOffsets:
    """Split the customer price stack into loss factors and clock-time offsets.

    This is the chokepoint every rolling_optimisation run passes through,
    whichever entry point drives it, so the two configuration guards that
    rolling optimisation depends on live here rather than in
    orchestrate.validate_scenario. (The Energy Masters pipeline calls model
    components directly and never invokes the orchestrator, so a guard placed
    only there would silently never fire on the first and most important use of
    the component.)

    Guard 1: the plan must be wholesale_exposed. For a TOU plan
    build_price_vectors returns the retail layer only, so the customer price
    does not depend on RRP at all and a price forecast cannot change dispatch.
    Rolling optimisation on a TOU plan is perfect foresight with an
    artificially truncated horizon.

    Guard 2: price ordering must hold across the whole market price range. See
    validate_price_ordering.

    The offsets are obtained by running build_layer_rates against a zero price
    series, so the wholesale terms vanish and what remains is exactly the
    network plus retail layers. Deriving them this way rather than
    reimplementing the layer arithmetic is what guarantees the rolling
    optimiser and the bill are computed from the same stack.

    Raises:
        ValueError: if the retail plan is not wholesale_exposed, or if price
            ordering can be violated within the market price range.
    """
    if retail_tariff.type != 'wholesale_exposed':
        raise ValueError(
            f"rolling optimisation requires a wholesale_exposed retail tariff, "
            f"got type='{retail_tariff.type}' ({retail_tariff.name}). A TOU "
            "customer price does not depend on RRP, so a price forecast cannot "
            "change dispatch."
        )

    zero_prices = pd.Series(0.0, index=index)
    lr = build_layer_rates(
        index, zero_prices, network_tariff, retail_tariff, state,
        interval_hours, lwc=lwc,
    )

    offsets = PriceStackOffsets(
        import_loss_factor=network_tariff.import_loss_factor,
        export_loss_factor=network_tariff.export_loss_factor,
        offset_import=lr.network_import + lr.retail_import,
        offset_export=lr.network_export + lr.retail_export,
    )
    validate_price_ordering(offsets, price_floor_mwh, price_cap_mwh)
    return offsets


def validate_price_ordering(
    offsets: PriceStackOffsets,
    price_floor_mwh: float = MARKET_PRICE_FLOOR_MWH,
    price_cap_mwh: float = MARKET_PRICE_CAP_MWH,
) -> None:
    """Prove p_imp >= p_exp cannot be violated anywhere in the market range.

    perfect_foresight_lp._validate_inputs raises on p_exp > p_imp. Under
    perfect foresight that fires once, up front, on a known price series. Under
    rolling optimisation the prices are forecasts and the check would sit
    inside a loop of ~105,000 solves per site-year, so a single bad value could
    kill a run most of the way through. Rather than clamping at run time and
    silently changing the model, the condition is proved impossible before the
    run starts.

    The spread is affine and therefore monotone in RRP:

        p_imp - p_exp = rrp*(import_lf - export_lf)/1000 + (offset_import - offset_export)

    so it suffices to evaluate at both ends of the market price range. This
    also handles the equal-loss-factor case, where the RRP term cancels and the
    spread does not depend on price at all, without a special branch.

    Note the asymmetry that makes this worth checking: import_lf is normally
    above 1 and export_lf below it, so the spread SHRINKS as prices fall and
    the binding case is the price floor, not the cap.

    Raises:
        ValueError: naming the bound that fails and the offending interval.
    """
    spread_offset = offsets.offset_import - offsets.offset_export
    d_lf = offsets.import_loss_factor - offsets.export_loss_factor

    for label, rrp in (('floor', price_floor_mwh), ('cap', price_cap_mwh)):
        spread = rrp * d_lf / MWH_TO_KWH + spread_offset
        worst = int(np.argmin(spread))
        if spread[worst] < 0:
            raise ValueError(
                f"price ordering can be violated at the market price {label} "
                f"({rrp:,.0f} $/MWh): p_exp exceeds p_imp by "
                f"{-spread[worst]:.5f} $/kWh at interval {worst}. "
                f"import_loss_factor={offsets.import_loss_factor:.6f}, "
                f"export_loss_factor={offsets.export_loss_factor:.6f}, "
                f"offset spread at that interval={spread_offset[worst]:.5f} $/kWh. "
                "Rolling optimisation cannot run against this tariff pair."
            )


# ---------------------------------------------------------------------------
# Mixed-plan dispatch prices (model_spec.md, Mixed-plan dispatch prices)
# ---------------------------------------------------------------------------

# Curtailment cost an optimiser carries on a mixed plan, $/kWh. Where the
# export floor binds, exporting and curtailing earn the household the same;
# the plan's retailer cannot curtail most sites, so this breaks the tie
# towards export. Curtailment still happens where a full battery and the
# export limit force it. See perfect_foresight_advanced_spec.md.
MIXED_CURTAIL_COST_PER_KWH = 1e-5


def dispatch_curtail_cost(retail_tariff: RetailTariff) -> float:
    """curtail_cost_per_kwh_dollars for an optimiser on this plan:
    MIXED_CURTAIL_COST_PER_KWH on a mixed plan, 0.0 (free curtailment)
    otherwise."""
    return MIXED_CURTAIL_COST_PER_KWH if retail_tariff.type == 'mixed' else 0.0


@dataclass(frozen=True)
class MixedDispatchStack:
    """A mixed plan's dispatch price stack, split into the parts that depend
    only on the clock and the parts that depend on RRP.

        p_imp[t] = import_base[t] + max(0, RRP[t])/1000 * import_loss_factor
        p_exp[t] = mixed_export_rate(RRP[t]/1000 * export_loss_factor,
                                     network_export[t], retail_tariff)

    Attributes:
        import_base: TOU usage rate + usage_other, $/kWh per interval.
        import_loss_factor: network_tariff.import_loss_factor.
        export_loss_factor: network_tariff.export_loss_factor.
        network_export: loss-adjusted network export rate, $/kWh per
            interval. Whether it enters V is the tariff's export_share_basis,
            applied by mixed_export_rate.
        retail_tariff: the mixed plan; supplies the share, basis, floor and
            feedin_other through mixed_export_rate.
    """
    import_base: np.ndarray
    import_loss_factor: float
    export_loss_factor: float
    network_export: np.ndarray
    retail_tariff: RetailTariff

    def __len__(self) -> int:
        return len(self.import_base)

    def window_price_vectors(
        self, rrp_mwh: np.ndarray, start: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """(p_imp, p_exp) in $/kWh for positions start .. start+len(rrp_mwh).

        The single implementation of the mixed-plan dispatch prices.
        build_price_vectors calls it over the whole period with the settled
        RRP; the rolling advanced planner calls it per window with the
        forecast RRP.
        """
        rrp = np.asarray(rrp_mwh, dtype=float)
        stop = start + len(rrp)
        if start < 0 or stop > len(self.import_base):
            raise ValueError(
                f"window [{start}, {stop}) falls outside the dispatch stack "
                f"period of length {len(self.import_base)}."
            )
        price_kwh = rrp / MWH_TO_KWH
        p_imp = (self.import_base[start:stop]
                 + np.maximum(0.0, price_kwh) * self.import_loss_factor)
        p_exp = mixed_export_rate(price_kwh * self.export_loss_factor,
                                  self.network_export[start:stop],
                                  self.retail_tariff)
        return p_imp, p_exp


def build_mixed_dispatch_stack(
    index: pd.DatetimeIndex,
    network_tariff: NetworkTariff,
    retail_tariff: RetailTariff,
    state: str,
    interval_hours: float = 5 / 60,
    lwc: Optional[LocalWallClock] = None,
    price_floor_mwh: float = MARKET_PRICE_FLOOR_MWH,
    price_cap_mwh: float = MARKET_PRICE_CAP_MWH,
) -> MixedDispatchStack:
    """Build a mixed plan's dispatch stack and prove its price ordering.

    The clock-time parts come from build_layer_rates run against a zero price
    series, so the TOU rates, adders and network export rate are exactly the
    ones billing uses. validate_mixed_price_ordering runs before returning.

    Raises:
        ValueError: if the retail plan is not mixed, or if p_imp < p_exp is
            possible for some RRP in [price_floor_mwh, price_cap_mwh].
    """
    if retail_tariff.type != 'mixed':
        raise ValueError(
            f"build_mixed_dispatch_stack needs a 'mixed' retail tariff, got "
            f"{retail_tariff.type!r} ({retail_tariff.name})"
        )
    zero_prices = pd.Series(0.0, index=index)
    lr = build_layer_rates(
        index, zero_prices, network_tariff, retail_tariff, state,
        interval_hours, lwc=lwc,
    )
    stack = MixedDispatchStack(
        import_base=lr.retail_import,
        import_loss_factor=network_tariff.import_loss_factor,
        export_loss_factor=network_tariff.export_loss_factor,
        network_export=lr.network_export,
        retail_tariff=retail_tariff,
    )
    validate_mixed_price_ordering(stack, price_floor_mwh, price_cap_mwh)
    return stack


def validate_mixed_price_ordering(
    stack: MixedDispatchStack,
    price_floor_mwh: float = MARKET_PRICE_FLOOR_MWH,
    price_cap_mwh: float = MARKET_PRICE_CAP_MWH,
) -> None:
    """Prove p_imp >= p_exp in every interval for any RRP in the market range.

    The spread p_imp - p_exp is piecewise linear in RRP. p_imp is convex,
    with its one bend at RRP = 0. p_exp = max(floor, share * V) +
    feedin_other is also convex, so its bend, where share * V meets the
    floor, makes the spread concave, and a concave bend is never a minimum.
    The minimum over the range is therefore at the market floor, at the cap,
    or at 0. Each candidate is evaluated per
    interval through window_price_vectors, the same code the optimisers use,
    so the proof cannot drift from the prices.

    Raises:
        ValueError: naming the price, the first offending interval and the
            two prices there.
    """
    tariff = stack.retail_tariff
    n = len(stack)
    candidates = [('the market floor', np.full(n, price_floor_mwh)),
                  ('the market cap', np.full(n, price_cap_mwh))]
    if price_floor_mwh < 0.0 < price_cap_mwh:
        candidates.append(('0 $/MWh', np.zeros(n)))
    share = tariff.wholesale_export_share

    for label, rrp in candidates:
        p_imp, p_exp = stack.window_price_vectors(rrp, 0)
        spread = p_imp - p_exp
        worst = int(np.argmin(spread))
        if spread[worst] < -1e-12:
            raise ValueError(
                f"{tariff.name}: mixed-plan dispatch prices can violate "
                f"p_imp >= p_exp at {label} ({rrp[worst]:,.2f} $/MWh): at "
                f"interval {worst} p_imp={p_imp[worst]:.6f} and "
                f"p_exp={p_exp[worst]:.6f} $/kWh. "
                f"import_loss_factor={stack.import_loss_factor:.6f}, "
                f"export_loss_factor={stack.export_loss_factor:.6f}, "
                f"share={share:g}, basis={tariff.export_share_basis}, "
                f"floor={tariff.export_floor_c_per_kwh} c/kWh. An optimiser "
                f"would import and export in the same interval."
            )


# ---------------------------------------------------------------------------
# Advanced tariff helpers (Steps 9+)
# ---------------------------------------------------------------------------

# Slack constant for zero-import penalty calibration. The floor denominator is
# (1.0 + slack) rather than the raw tolerance, ensuring the penalty per kWh is
# always marginally below the reward value per kWh. This guarantees that a
# critical export rate (per-kWh) dominates the zero-import penalty on
# high-price days, so the LP exports freely during price spikes even if doing so
# means forfeiting the daily reward. See perfect_foresight_advanced_spec.md.
# Known limitation: the priority ordering only holds when
#   critical_export_rate.rate_c_per_kwh >= reward_c_per_day.
_ZERO_IMPORT_PENALTY_SLACK_KWH = 0.01



def critical_masks(
    prices_mwh,
    retail_tariff: RetailTariff,
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean (export, import) masks of critical intervals.

    Export is critical where prices_mwh > critical_export_rate's threshold,
    import where prices_mwh < critical_import_rate's threshold. An absent
    rate gives an all-False mask. prices_mwh is any 1-D price array or Series
    in $/MWh: the settled price for perfect foresight and billing, the
    forecast for a rolling window. See perfect_foresight_advanced_spec.md.
    """
    prices = np.asarray(prices_mwh, dtype=float)
    cx, cm = retail_tariff.critical_export_rate, retail_tariff.critical_import_rate
    exp_mask = (prices > cx.threshold_dollars_per_mwh if cx is not None
                else np.zeros(prices.shape, dtype=bool))
    imp_mask = (prices < cm.threshold_dollars_per_mwh if cm is not None
                else np.zeros(prices.shape, dtype=bool))
    return exp_mask, imp_mask


def critical_rates_dollars(retail_tariff: RetailTariff) -> tuple[
        Optional[float], Optional[float]]:
    """(rho_x, rho_m) in $/kWh: each critical rate plus the flat adder on its
    side, or None where the rate is absent."""
    C = C_TO_DOLLARS
    cx, cm = retail_tariff.critical_export_rate, retail_tariff.critical_import_rate
    rho_x = (None if cx is None
             else (cx.rate_c_per_kwh + retail_tariff.feedin_other_c_per_kwh) / C)
    rho_m = (None if cm is None
             else (cm.rate_c_per_kwh + retail_tariff.usage_other_c_per_kwh) / C)
    return rho_x, rho_m


def apply_critical_rates(
    p_imp: pd.Series,
    p_exp: pd.Series,
    prices_mwh,
    retail_tariff: RetailTariff,
    shift_opposite: bool,
) -> tuple[pd.Series, pd.Series]:
    """Replace the critical side's rate in each critical interval.

    In a critical export interval p_exp becomes rho_x, and in a critical
    import interval p_imp becomes rho_m (critical_rates_dollars).

    shift_opposite=False (billing): nothing else changes. This is what the
    retailer charges and pays.

    shift_opposite=True (the advanced LP and the rolling planner): the other
    vector is moved by the same amount, delta = new - old, so the spread
    p_imp - p_exp keeps its value from the input vectors. Without it a
    critical rate can put p_exp above p_imp in the same interval and the LP
    would import and export at once. delta is measured from the vectors
    passed in, which for the LP are the block-masked base vectors; the
    rolling planner holds the same base, so both build the same vectors
    from the same prices.

    Block tier membership is not handled here: the caller uses
    block_tier_masks so the critical side of a critical interval is outside
    its tier. Returns the inputs unchanged (no copy) when the plan has no
    critical rates.
    """
    if not retail_tariff.has_critical_rates:
        return p_imp, p_exp
    if retail_tariff.type != 'tou':
        raise ValueError(
            f"{retail_tariff.name}: critical rates are defined for tou plans "
            f"only; got type {retail_tariff.type!r}."
        )
    exp_mask, imp_mask = critical_masks(prices_mwh, retail_tariff)
    rho_x, rho_m = critical_rates_dollars(retail_tariff)

    imp = p_imp.to_numpy(dtype=float).copy()
    exp = p_exp.to_numpy(dtype=float).copy()
    if rho_x is not None and exp_mask.any():
        delta = rho_x - exp[exp_mask]
        exp[exp_mask] = rho_x
        if shift_opposite:
            imp[exp_mask] += delta
    if rho_m is not None and imp_mask.any():
        delta = rho_m - imp[imp_mask]
        imp[imp_mask] = rho_m
        if shift_opposite:
            exp[imp_mask] += delta
    return (pd.Series(imp, index=p_imp.index, name=p_imp.name),
            pd.Series(exp, index=p_exp.index, name=p_exp.name))


def build_zero_import_penalty(
    index: pd.DatetimeIndex,
    retail_tariff: RetailTariff,
    state: str,
    interval_hours: float = 5 / 60,
) -> Optional[pd.Series]:
    """Return a per-interval penalty ($/kWh) to add to p_imp for the zero-
    import reward window, or None if no ZeroImportReward is defined.

    The penalty is calibrated so that importing 1 kWh during the window costs
    marginally less than the reward value per kWh:

        denominator = max(tolerance_kwh, 1.0 + _ZERO_IMPORT_PENALTY_SLACK_KWH)
        pi = reward_c_per_day / denominator / C_TO_DOLLARS   ($/kWh)

    The floor of (1.0 + slack) ensures the penalty is always slightly below
    the reward value per kWh, so a critical export rate of at least the
    daily reward per kWh dominates on high-price days. See _ZERO_IMPORT_PENALTY_SLACK_KWH for the
    full rationale and known limitation.

    Example (GloBird Advanced TOU 2025): $1/day reward, 3-hour window,
        0.03 kWh/hr tolerance -> tolerance_kwh = 0.09 kWh ->
        denominator = max(0.09, 1.01) = 1.01 ->
        pi = $1 / 1.01 = $0.990/kWh (below the $1/kWh critical export rate).

    This penalty is ARTIFICIAL (not a real tariff component) and must not
    be used in billing. Billing counts qualifying days post-solve and credits
    the real reward amount. See spec section 8.

    Args:
        state: the site's actual location (scenario.state); the reward
            window's start_hour/end_hour are local hours, see
            build_layer_rates for why this is separate from the tariff's
            own declared state.
    """
    reward = retail_tariff.zero_import_reward
    if reward is None:
        return None

    lwc = local_wall_clock(index, state, interval_hours)
    start_hour = lwc.hour
    in_window = (start_hour >= reward.start_hour) & (start_hour < reward.end_hour)

    denominator = max(reward.tolerance_kwh, 1.0 + _ZERO_IMPORT_PENALTY_SLACK_KWH)
    penalty_per_kwh = (reward.reward_c_per_day / C_TO_DOLLARS) / denominator
    values = in_window.astype(float) * penalty_per_kwh
    return pd.Series(values, index=index, name='zero_import_penalty')


# Coefficient for the TOU export ordering nudge ($/kWh per minute elapsed in day).
# Maximum nudge at end of day: 1e-8 * (24*60 - 1) ≈ 0.0000144 $/kWh -- well below
# any real tariff signal. Applies to TOU plans only; wholesale-exposed plans already
# have interval-varying export signals from the spot price.
_TOU_EXPORT_NUDGE_PER_MIN = 1e-8

# Largest value the nudge takes in any interval (the last minute of a local day).
TOU_EXPORT_NUDGE_MAX = _TOU_EXPORT_NUDGE_PER_MIN * (24 * 60 - 1)


def build_tou_export_order_nudge(
    index: pd.DatetimeIndex,
    retail_tariff: RetailTariff,
    state: str,
    interval_hours: float = 5 / 60,
) -> Optional[pd.Series]:
    """Return a per-interval nudge ($/kWh) to add to p_exp for TOU plans, or None.

    On TOU plans the export price is flat within each day (or at most varies by
    period). When a day has excess solar the LP objective is degenerate between
    two physically distinct solutions:
      (a) export early, then charge the battery from remaining solar, or
      (b) charge the battery first, then export the surplus.
    Both have identical cost under a flat export price. The LP solver happens to
    prefer (a) via its internal pivot ordering, which does not reflect real
    controller behaviour (a real controller charges with available solar first).

    This nudge breaks the degeneracy by making later exports marginally more
    attractive: the nudge increases linearly through each local calendar day,
    so the LP prefers exporting at minute 500 over minute 0 by 0.000005 $/kWh.
    The charge-first, export-surplus solution is therefore selected when the
    two are otherwise equal.

    The nudge coefficient (1e-8 $/kWh per minute) gives a maximum end-of-day
    value of approximately 0.0000144 $/kWh -- negligible against any real tariff
    signal (smallest real feedin rates are several c/kWh). It has no material
    effect on billing-equivalent dispatch decisions.

    This nudge is ARTIFICIAL (not a real tariff component) and must not be used
    in billing. It is applied to the LP export price only, through
    apply_tou_export_order_nudge, by perfect_foresight_lp_advanced and by
    rolling_advanced.AdvancedWindowPlanner.

    Returns None for wholesale_exposed plans: the spot price already provides
    genuine interval-varying export signals, and the degeneracy does not arise.

    Args:
        state: the site's actual location (scenario.state). The nudge
            increases through each local calendar day, not each AEST day --
            see build_layer_rates for why this is separate from the
            tariff's own declared state.
    """
    if retail_tariff.type != 'tou':
        return None

    lwc = local_wall_clock(index, state, interval_hours)
    mins_elapsed = (lwc.hour * 60 + lwc.minute).astype(float)
    values = _TOU_EXPORT_NUDGE_PER_MIN * mins_elapsed
    return pd.Series(values, index=index, name='tou_export_order_nudge')


def apply_tou_export_order_nudge(
    p_imp: np.ndarray,
    p_exp: np.ndarray,
    nudge: np.ndarray,
) -> np.ndarray:
    """Return p_exp + nudge, capped at p_imp in each interval.

    Where p_imp and p_exp are equal (a free import window with zero feed-in,
    for example) the uncapped nudge would make importing and exporting the
    same kWh in the same interval profitable with no battery involved, and
    the LP would do so up to the export limit. The cap drops the nudge in
    those intervals, so p_exp <= p_imp holds wherever it held before. An
    interval where p_exp already exceeds p_imp is left at p_exp.

    p_imp is the import price before the zero-import penalty. The penalty
    only raises p_imp, so the ordering also holds after it is added.
    """
    p_imp = np.asarray(p_imp, dtype=float)
    p_exp = np.asarray(p_exp, dtype=float)
    return np.minimum(p_exp + np.asarray(nudge, dtype=float),
                      np.maximum(p_imp, p_exp))


def check_export_order_nudge_degradation(
    c_deg_per_kwh: float,
    plan_name: str,
) -> None:
    """Raise ValueError when the degradation cost does not cover the nudge.

    Where the real import price at one interval equals the real export price
    at a later one (zero-rate import and zero feed-in, say), charging then
    discharging to export gains up to TOU_EXPORT_NUDGE_MAX per kWh from the
    nudge and costs 2 x c_deg_per_kwh. Below the threshold the LP cycles the
    battery to collect the nudge.
    """
    if 2.0 * c_deg_per_kwh <= TOU_EXPORT_NUDGE_MAX:
        raise ValueError(
            f"{plan_name}: apply_export_order_nudge needs c_deg_per_kwh > "
            f"{TOU_EXPORT_NUDGE_MAX / 2:.2e} $/kWh on a TOU plan, got "
            f"{c_deg_per_kwh}. Below that the LP cycles the battery to "
            f"collect the nudge."
        )


def block_window_mask(
    index: pd.DatetimeIndex,
    window: BlockTariffWindow,
    state: str,
    interval_hours: float = 5 / 60,
    lwc: Optional[LocalWallClock] = None,
) -> np.ndarray:
    """Boolean mask (length-T ndarray) of intervals within a BlockTariffWindow.

    Uses interval-START instant (same convention as period_rate_series):
    the rate for an interval is the rate applying at the instant it began,
    converted to local time for `state`. Respects applies_weekends: when
    False, local weekend intervals are excluded from the mask (they fall
    back to the standard period rates).

    Args:
        lwc: Optional precomputed LocalWallClock for (index, state,
            interval_hours). billing.py's _compute_block_tier_cost calls
            this once per window in a tariff's block_import_tiers /
            block_export_tiers; passing a precomputed lwc avoids one
            redundant local_wall_clock call per window. Computed internally
            if not given, so existing callers are unaffected.
    """
    if lwc is None:
        lwc = local_wall_clock(index, state, interval_hours)
    in_hours = (lwc.hour >= window.start_hour) & (lwc.hour < window.end_hour)
    if not window.applies_weekends:
        in_hours = in_hours & ~lwc.is_weekend
    return in_hours


def block_tier_masks(
    index: pd.DatetimeIndex,
    retail_tariff: RetailTariff,
    state: str,
    interval_hours: float = 5 / 60,
    prices_mwh=None,
    lwc: Optional[LocalWallClock] = None,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """Effective block window masks for tier accounting, one per window in
    tariff order: (import window masks, export window masks).

    Each is block_window_mask with the intervals that are critical on the
    window's side removed (critical imports from import windows, critical
    exports from export windows), because critical flow does not count
    towards a block tier. prices_mwh is the price that decides criticality:
    the settled price for perfect foresight and billing. It is required when
    the plan has critical rates, and ignored otherwise.
    """
    if lwc is None:
        lwc = local_wall_clock(index, state, interval_hours)
    if retail_tariff.has_critical_rates:
        if prices_mwh is None:
            raise ValueError(
                f"block_tier_masks: {retail_tariff.name} has critical rates, "
                f"so prices_mwh is required to decide tier membership"
            )
        prices = np.asarray(prices_mwh, dtype=float)
        if prices.shape != (len(index),):
            raise ValueError(
                f"block_tier_masks: prices_mwh has shape {prices.shape}, "
                f"expected ({len(index)},) to match index"
            )
        exp_crit, imp_crit = critical_masks(prices, retail_tariff)
    else:
        exp_crit = imp_crit = np.zeros(len(index), dtype=bool)
    imp_masks = tuple(
        np.asarray(block_window_mask(index, w, state, interval_hours, lwc=lwc))
        & ~imp_crit
        for w in (retail_tariff.block_import_tiers or ()))
    exp_masks = tuple(
        np.asarray(block_window_mask(index, w, state, interval_hours, lwc=lwc))
        & ~exp_crit
        for w in (retail_tariff.block_export_tiers or ()))
    return imp_masks, exp_masks


def charge_window_mask(
    index: pd.DatetimeIndex,
    window: ChargeWindow,
    state: str,
    interval_hours: float = 5 / 60,
) -> np.ndarray:
    """Boolean mask (length-T ndarray) of intervals within a ChargeWindow.

    Uses the same interval-START convention as block_window_mask and
    period_rate_series: an interval belongs to the window if the local
    instant it began falls in [start_hour, end_hour). A ChargeWindow has no
    weekend flag and no tiers, so the comparison is the bare hour test; the
    window applies every day.
    """
    lwc = local_wall_clock(index, state, interval_hours)
    return (lwc.hour >= window.start_hour) & (lwc.hour < window.end_hour)


def build_base_price_vectors_advanced(
    index: pd.DatetimeIndex,
    prices_mwh: pd.Series,
    network_tariff: NetworkTariff,
    retail_tariff: RetailTariff,
    state: str,
    interval_hours: float = 5 / 60,
) -> tuple[pd.Series, pd.Series]:
    """Advanced price vectors WITHOUT critical rates.

    Everything build_lp_price_vectors_advanced does except the final
    critical-rate step: build_price_vectors, then block-window masking. See
    that function for the masking rules, which are documented there and not
    repeated here.

    Split out for the rolling advanced path. Under a rolling horizon the
    critical rates are evaluated against the FORECAST wholesale price, which changes
    every window, while everything below is fixed for the whole period and is
    built once per site. Both callers therefore share one definition of the base
    vectors rather than the rolling path reimplementing the masking.

    Callers that want the complete vectors should call
    build_lp_price_vectors_advanced, not this function.

    On a mixed plan build_price_vectors already returns the mixed-plan
    dispatch prices, and a mixed plan carries no block tiers or critical
    rates (the loader rejects them), so both functions return those prices
    unchanged. The rolling planner holds a MixedDispatchStack instead of
    calling this function, because a mixed plan's prices depend on RRP.
    """
    p_imp, p_exp = build_price_vectors(
        index, prices_mwh, network_tariff, retail_tariff, state, interval_hours,
    )

    C = C_TO_DOLLARS

    if retail_tariff.block_import_tiers:
        usage_other_rate = retail_tariff.usage_other_c_per_kwh / C
        p_imp = p_imp.copy()
        for window in retail_tariff.block_import_tiers:
            mask = block_window_mask(index, window, state, interval_hours)
            if retail_tariff.type == 'tou':
                # Block rate replaces the TOU energy rate for these intervals.
                # Only the flat usage_other component remains per-interval;
                # the tier variable handles the rest of the import cost.
                p_imp[mask] = usage_other_rate
            # wholesale_exposed: retail_import == usage_other already; no-op.

    if retail_tariff.block_export_tiers:
        feedin_other_rate = retail_tariff.feedin_other_c_per_kwh / C
        p_exp = p_exp.copy()
        for window in retail_tariff.block_export_tiers:
            mask = block_window_mask(index, window, state, interval_hours)
            if retail_tariff.type == 'tou':
                # Block rate replaces the TOU feedin rate for these intervals.
                p_exp[mask] = feedin_other_rate
            # wholesale_exposed: retail_export == feedin_other already; no-op.

    return p_imp, p_exp


def build_lp_price_vectors_advanced(
    index: pd.DatetimeIndex,
    prices_mwh: pd.Series,
    network_tariff: NetworkTariff,
    retail_tariff: RetailTariff,
    state: str,
    interval_hours: float = 5 / 60,
) -> tuple[pd.Series, pd.Series]:
    """Build LP-specific per-interval price vectors for advanced retail tariffs.

    For intervals NOT in any block window: identical to build_price_vectors.
    For intervals IN a block import/export window: the TOU energy rate is
    replaced by usage_other_rate / feedin_other_rate only. The blocked retail
    energy component is excluded from the per-interval vector and is instead
    captured by the daily tier variables in the LP objective.

    Critical rates are applied last, after block masking, with the
    opposite-side shift (apply_critical_rates, shift_opposite=True). Tier
    membership for critical intervals is the tier builder's concern
    (block_tier_masks), not this function's.

    For TOU plans, the per-interval components for a block window interval are:
        p_imp[t] = usage_other_rate   (flat levies, e.g. greenpower)
        p_exp[t] = feedin_other_rate  (flat feedin adjustment, if any)

    For wholesale_exposed plans, block tiers are an unusual combination (see
    tariff loader warning). The wholesale + network components still apply
    per-interval; only the retail energy overlay is affected. Since
    wholesale_exposed plans have no TOU retail rate (retail_import = usage_other
    already), the per-interval vector is unchanged for those intervals -- the
    block tier passes through the loader warning and this function is a no-op
    for that combination.

    The zero-import penalty is NOT applied here; it is added by the LP builder
    (perfect_foresight_lp_advanced) directly to p_imp before constructing the
    objective, so it does not contaminate the billing price vectors.

    Args:
        state: the site's actual location (scenario.state); see
            build_layer_rates for why this is separate from the tariffs'
            own declared state.
    """
    p_imp, p_exp = build_base_price_vectors_advanced(
        index, prices_mwh, network_tariff, retail_tariff, state, interval_hours,
    )

    # Critical rates last, measured from the block-masked base: the rolling
    # planner holds the same base and applies the same function, so the two
    # paths build identical vectors from identical prices.
    p_imp, p_exp = apply_critical_rates(
        p_imp, p_exp, prices_mwh, retail_tariff, shift_opposite=True,
    )

    return p_imp, p_exp