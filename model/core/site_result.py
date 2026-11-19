"""Output types produced by model components.

SiteTraces is produced by any component that determines site import/export
behaviour (self_consumption, the perfect-foresight, VPP and rolling
components).
Downstream components (notably billing) consume SiteTraces without needing
to know which upstream component produced it.

SiteTraces holds both the headline import/export traces and the underlying
flow decomposition (sol_to_home, sol_to_bat, etc). The decomposition is
needed by downstream components that build on top of self_consumption
(e.g. VPP, perfect_foresight comparisons against self_consumption) and
can't be reconstructed from import/export alone.

For solar-only or no-CER scenarios the battery flow columns are populated
as all-zero series rather than left as None - this keeps downstream code
uniform and avoids None-checks at every consumer.

SiteBill is produced by billing.py for each site under a scenario.
"""
from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class SiteTraces:
    """5-minute energy flow traces for a single site after applying a
    coordination strategy.

    All energy values are in kWh per 5-min interval. All series share the
    same datetime_aest index.

    Sign conventions:
        Most flows are non-negative (representing energy moving in a
        specific direction). One exception: sol_to_home_kwh can be
        negative when the solar inverter draws power from the household
        circuit overnight (parasitic draw, e.g. inverter standby load).
        In that case the inverter effectively acts as additional load:
        residual_load increases by |negative_solar| and grid or battery
        covers it. gross_load already reflects the inverter draw under
        the EDP convention gross_load = solar + ac_load_net.

    Headline traces:
        site_id: Site identifier.
        import_kwh: Grid -> site (total imports). Non-negative.
        export_kwh: Site -> grid (total exports). Non-negative.
        battery_soc_kwh: Battery state of charge at the end of each
            interval. Non-negative. All-zero series when no battery is
            present.

    Flow decomposition (populated as all-zero series when not applicable
    for a given cohort):
        sol_to_home_kwh: Solar generation consumed by load. Mirrors the
            sign of solar - can be negative for intervals with parasitic
            inverter draw.
        sol_to_bat_kwh: Solar generation charging the battery (AC side).
            Non-negative.
        sol_to_grid_kwh: Solar generation exported to grid. Non-negative.
        bat_to_home_kwh: Battery discharging to load (AC side).
            Non-negative.
        grid_to_home_kwh: Grid power consumed directly by load.
            Non-negative.
        grid_to_bat_kwh: Grid power charging the battery (AC side).
            Non-negative. Zero for self_consumption; non-zero for
            VPP/perfect_foresight.
        bat_to_grid_kwh: Battery discharging to grid (AC side).
            Non-negative. Zero for self_consumption; non-zero for
            VPP/perfect_foresight.
        curtail_kwh: Solar generation curtailed (e.g. by the inverter
            when the battery is full and the export limit is binding, or
            when the export rate is negative). Non-negative. All-zero
            series for self_consumption (no curtailment logic); non-zero
            when the LP component curtails.

    Invariants the producer must enforce:
        import_kwh == grid_to_home_kwh + grid_to_bat_kwh
        export_kwh == sol_to_grid_kwh + bat_to_grid_kwh
        load   ~=  sol_to_home + bat_to_home + grid_to_home
        solar  ~=  sol_to_home + sol_to_bat + sol_to_grid + curtail_kwh
    """
    site_id: str
    import_kwh: pd.Series
    export_kwh: pd.Series
    battery_soc_kwh: pd.Series

    sol_to_home_kwh: pd.Series
    sol_to_bat_kwh: pd.Series
    sol_to_grid_kwh: pd.Series
    bat_to_home_kwh: pd.Series
    grid_to_home_kwh: pd.Series
    grid_to_bat_kwh: pd.Series
    bat_to_grid_kwh: pd.Series
    curtail_kwh: pd.Series


@dataclass(frozen=True)
class SiteBill:
    """Settlement results for a single site over the scenario period.

    All amounts are in dollars. Three settlement layers are ALWAYS computed
    for every site regardless of the retail plan, because the physical energy
    flows incur wholesale and network costs whether or not the customer sees
    them. The customer's actual bill (household_total) is a plan-dependent
    view onto these layers; the retailer's economics (retailer_margin) is a
    uniform formula across plans.

    Sign convention: credits are stored as NEGATIVE numbers, so every
    "total" is a straight sum of its components. Under normal export a
    feedin credit is negative (reduces the bill); during negative-price
    export it becomes positive (the customer pays to export).

    Layers (all always populated):

    Wholesale layer (spot price x loss-adjusted energy):
        wholesale_usage_cost: cost of imports at spot. >=0 normally, <0 if
            importing during negative prices.
        wholesale_feedin_credit: credit for exports at spot. <=0 normally,
            >0 if exporting during negative prices.

    Network layer (network tariff x loss-adjusted energy + daily):
        network_usage_cost: network charge on imports. >=0.
        network_feedin_credit: network credit/penalty on exports.
        network_daily_charges: network daily fixed charge. >=0.

    Retail layer (retail tariff):
        retail_usage_cost: TOU import rates, or wholesale-exposed flat
            market adders, applied to imports.
        retail_feedin_credit: TOU feedin rates, wholesale-exposed feedin
            adjustment, or a mixed plan's floored export share, applied to
            exports.
        retail_daily_charges: retail daily supply charge / subscription. >=0.

    Views:
        plan_type: which retail tariff applied ('tou', 'wholesale_exposed' or
            'mixed').
        household_total: what the customer actually pays (all customer
            payments; the household never pays the network directly).
            - wholesale_exposed: sum of all wholesale, network, and retail
              components (the customer bears the full stack).
            - tou, mixed: sum of retail components only (a mixed plan's
              retail export rate already carries its floored share of export
              value).
        retailer_margin: household_total
            - (wholesale_usage_cost + wholesale_feedin_credit)
            - (network_usage_cost + network_feedin_credit + network_daily_charges)
          i.e. all revenue in minus all net wholesale and network costs.
          Same formula for every plan type.
    """
    site_id: str
    plan_type: str

    wholesale_usage_cost: float
    wholesale_feedin_credit: float

    network_usage_cost: float
    network_feedin_credit: float
    network_daily_charges: float

    retail_usage_cost: float
    retail_feedin_credit: float
    retail_daily_charges: float

    household_total: float
    retailer_margin: float

    retail_other_credits: float = 0.0
    """Non-standard retail credits not captured by usage/feedin period rates.
    Includes: zero-import avoidance rewards, VPP per-kWh dispatch rewards, and
    any other advanced plan bonuses paid directly to the customer.
    Stored as a NEGATIVE number (credit reduces the bill) following the sign
    convention. Folded into household_total for both plan types:
        wholesale_exposed: wholesale_net + network_net + retail_net + retail_other_credits
        tou:               retail_net + retail_other_credits
    Zero for all standard plans; only non-zero when the retail tariff has
    advanced features (RetailTariff.has_advanced_features is True).
    """


@dataclass(frozen=True)
class VppTraces:
    """Natural / forced flow split from a VPP dispatch.

    Produced by the VPP coordination components (vpp_sc_event and
    rolling_vpp_event) alongside the standard SiteTraces.
    Carries the operator-side metadata that billing and operator P&L need and
    cannot reconstruct from import_kwh / export_kwh alone, since those headline
    series contain the sum of natural and forced flows.

    All series share the same datetime_aest index as the companion SiteTraces.
    All values are in kWh per 5-min interval and are non-negative.

    Attributes:
        site_id: Site identifier. Matches SiteTraces.site_id.
        n_imp_kwh: Natural import -- grid power drawn to serve load in excess
            of what solar and the battery can provide. This is the import that
            would occur under self-consumption dispatch; it does not count
            against the VPP annual energy cap.
        f_imp_kwh: Forced import -- additional grid power drawn by the operator
            to charge the battery beyond what self-consumption would do (operator
            force-charge). Counts against the annual cap when limit_applies_to
            is 'import_and_export'. Earns the household vpp_imp_reward.
        n_exp_kwh: Natural export -- solar surplus sent to the grid after
            serving load and battery charging. Does not count against the cap
            and does not earn the VPP export reward.
        f_exp_kwh: Forced export -- battery energy dispatched to the grid by
            the operator during a VPP event. Counts against the annual cap.
            Earns the household vpp_exp_reward.

    Invariants (algebraic from the LP flow balance, holding to solver tolerance):
        n_imp_kwh + f_imp_kwh == SiteTraces.import_kwh
        n_exp_kwh + f_exp_kwh == SiteTraces.export_kwh
    """
    site_id: str
    n_imp_kwh: pd.Series
    f_imp_kwh: pd.Series
    n_exp_kwh: pd.Series
    f_exp_kwh: pd.Series