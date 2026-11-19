"""Scenario configuration types for the coordination model.

A ScenarioConfig fully specifies a single model run for one state and one year.
Multi-state or multi-year analysis is handled by main.py iterating over a list
of ScenarioConfig instances; the components themselves only ever see one
state-year at a time.
"""
from dataclasses import dataclass
from typing import Optional, Literal, get_args


CohortType = Literal['no_cer', 'solar', 'solar_battery']
CoordinationType = Literal['self_consumption', 'perfect_foresight', 'perfect_foresight_advanced', 'perfect_foresight_custom', 'vpp_sc_event', 'vpp_tou_event', 'self_hedging', 'rolling_perfect_foresight_lp',
                           'rolling_perfect_foresight_lp_advanced',
                           'rolling_vpp_sc_event', 'rolling_vpp_tou_event']
StateType = Literal['NSW', 'QLD', 'SA', 'VIC']

# Coordination types that carry a RollingConfig. Named once so the config
# validation, the orchestrator's routing and its shared validation branch all
# test the same set rather than three lists that can drift.
ROLLING_COORDINATION_TYPES: tuple[str, ...] = (
    'rolling_perfect_foresight_lp',
    'rolling_perfect_foresight_lp_advanced',
    'rolling_vpp_sc_event',
    'rolling_vpp_tou_event',
)

# Coordination types that dispatch under operator control during VPP events.
# Grouped for the same anti-drift reason as ROLLING_COORDINATION_TYPES: the
# op_retail_tariff_id requirement and the orchestrator's VPP routing must test
# one set. The rolling members are also in ROLLING_COORDINATION_TYPES; the two
# sets overlap deliberately and neither is a subset of the other.
VPP_COORDINATION_TYPES: tuple[str, ...] = (
    'vpp_sc_event',
    'vpp_tou_event',
    'rolling_vpp_sc_event',
    'rolling_vpp_tou_event',
)

# Coordination types a 'mixed' retail plan may run under. None is the no_cer
# and solar cohorts, which have no coordination type. Self-consumption reads
# no prices; the two advanced optimisers build the mixed-plan dispatch prices
# (model_spec.md). validate_scenario rejects a mixed plan under any other.
# Coordination types that apply ScenarioConfig.apply_export_order_nudge: the
# advanced TOU LP for the whole period, the TOU gaps of vpp_tou_event, and the
# rolling window planner (AdvancedWindowPlanner) on the two rolling types that
# build one.
EXPORT_ORDER_NUDGE_COORDINATION_TYPES: tuple[str, ...] = (
    'perfect_foresight_advanced',
    'vpp_tou_event',
    'rolling_perfect_foresight_lp_advanced',
    'rolling_vpp_tou_event',
)

MIXED_COORDINATION_TYPES: tuple = (
    None,
    'self_consumption',
    'perfect_foresight_advanced',
    'rolling_perfect_foresight_lp_advanced',
)


@dataclass(frozen=True)
class BatterySpec:
    """Specification of a single home battery.

    Initially a single BatterySpec is applied uniformly to all sites in a
    scenario. Future extension: a dict[site_id, BatterySpec] could be held
    in DataContext to allow site-specific batteries.
    """
    capacity_kwh: float           # max usable capacity (battery_kWh_limit)
    backup_reserve_pct: float     # minimum SOC for backup, 0-100 (battery_backup_reserve)
    power_kw: float               # max charge/discharge power (battery_kW_limit)
    charge_efficiency: float      # 0-1 (battery_charge_eff)
    discharge_efficiency: float   # 0-1 (battery_discharge_eff)
    c_deg_per_kwh: float = 0.005 # degradation cost per kWh throughput ($/kWh);
                                  # added to the LP objective for each kWh of charge
                                  # AND discharge. 0.0 = no penalty.
                                  # Practical note: with negative import prices, the LP
                                  # may find it profitable to simultaneously charge from
                                  # the grid and discharge to load, cycling energy purely
                                  # to maximise imports. A value of ~0.005 $/kWh makes
                                  # this unprofitable (penalty exceeds import gain) and
                                  # produces physically sensible dispatch. The default
                                  # 0.005 also provides the margin required by the TOU
                                  # export order nudge in perfect_foresight_lp_advanced
                                  # -- do not set to 0.0 on TOU plans. This cost does
                                  # NOT appear in the household bill -- it is a modelling
                                  # regulariser only.

    def __post_init__(self):
        if not 0 <= self.backup_reserve_pct <= 100:
            raise ValueError(
                f"backup_reserve_pct must be 0-100, got {self.backup_reserve_pct}"
            )
        if not 0 < self.charge_efficiency <= 1:
            raise ValueError(
                f"charge_efficiency must be in (0, 1], got {self.charge_efficiency}"
            )
        if not 0 < self.discharge_efficiency <= 1:
            raise ValueError(
                f"discharge_efficiency must be in (0, 1], got {self.discharge_efficiency}"
            )
        if self.capacity_kwh <= 0:
            raise ValueError(f"capacity_kwh must be > 0, got {self.capacity_kwh}")
        if self.power_kw <= 0:
            raise ValueError(f"power_kw must be > 0, got {self.power_kw}")
        if self.c_deg_per_kwh < 0:
            raise ValueError(
                f"c_deg_per_kwh must be >= 0, got {self.c_deg_per_kwh}"
            )


@dataclass(frozen=True)
class UsageForecastConfig:
    """Settings for the household load forecast used by rolling_optimisation.

    Held here rather than beside its implementation in model/forecast/usage.py
    because RollingConfig must hold it and core must not import from the
    forecast package. Same arrangement as BatterySpec.

    Defaults are fitted over 433 EDP sites, 2022, all four states, on
    cumulative energy error with infilled intervals excluded. See
    usage_forecast_spec.md.

    Attributes:
        history_days: Trailing local days behind the profile.
        min_history_days: Below this many valid history days the provider
            returns None and the caller falls back to self-consumption.
        smoothing_intervals: Width of the circular centred moving average
            applied to the shape along the time-of-day axis. Must be odd.
            1 disables. Its effect is on the recent adjustment rather than on
            the profile: a smoother base gives a cleaner base residual, which
            is what the adjustment averages.
        dow_level: Split the daily-energy level by day-of-week rather than by
            day type.
        lookback_hours: x. Trailing window of observed load behind the
            additive adjustment. Floor is one interval.
        decay_hours: y. Decay constant of the adjustment across the horizon.
            0 disables the adjustment.
        min_day_coverage: Fraction of a local day's intervals that must carry
            a real measurement for that day to enter the profile history.
    """
    history_days: int = 28
    min_history_days: int = 1
    smoothing_intervals: int = 7
    dow_level: bool = False
    lookback_hours: float = 0.5
    decay_hours: float = 1.5
    min_day_coverage: float = 0.85

    def __post_init__(self):
        if self.history_days < 1:
            raise ValueError(
                f"history_days must be >= 1, got {self.history_days}")
        if not 1 <= self.min_history_days <= self.history_days:
            raise ValueError(
                "min_history_days must be in [1, history_days], got "
                f"{self.min_history_days}")
        if self.smoothing_intervals < 1 or self.smoothing_intervals % 2 == 0:
            raise ValueError(
                "smoothing_intervals must be a positive odd number, got "
                f"{self.smoothing_intervals}")
        if self.lookback_hours <= 0:
            raise ValueError(
                f"lookback_hours must be > 0, got {self.lookback_hours}")
        if self.decay_hours < 0:
            raise ValueError(
                f"decay_hours must be >= 0, got {self.decay_hours}")
        if not 0.0 < self.min_day_coverage <= 1.0:
            raise ValueError(
                f"min_day_coverage must be in (0, 1], got "
                f"{self.min_day_coverage}")


SolarForecastMode = Literal['persistence', 'index']


@dataclass(frozen=True)
class SolarForecastConfig:
    """Settings for the solar forecast used by rolling_optimisation.

    See solar_forecast_spec.md. Every method it configures is naive relative to
    an operator forecasting from satellite imagery, so a run using it bounds
    the cost of solar forecast error from above rather than modelling any
    retailer's behaviour.

    Attributes:
        mode: 'persistence' repeats yesterday's trace, a loose bound.
            'index' scales a clear-sky envelope by a forecast index, a tighter
            one. Both are valid bounds.
        history_days: Trailing local days behind the envelope and R_mean.
        envelope_quantile: Per-slot quantile of generation forming the
            clear-sky proxy.
        w_prev_share: Share of the history estimate taken from the previous
            day; the remainder comes from the trailing mean.
        w_today_max: Ceiling on the weight given to today's observation.
        f_on: Fraction of the day's envelope energy over which that weight
            ramps in.
        lookback_hours: x. Trailing window behind the additive index
            adjustment.
        decay_hours: y. Decay constant of that adjustment. 0 disables it.
        index_max: Upper clip on the FORECAST index. The observed index is
            never clipped: the envelope is a high quantile, so a few per cent
            of real observations exceed it by construction.
        min_day_coverage: Intervals a day needs to enter the history.

    Defaults are the equal-weight optimum over 433 EDP sites, 2022, all four
    states. Equal weight rather than pooled because the parameters serve VCR,
    which is scale-free, so a large site should not outvote a small one; the
    two criteria agreed at cohort scale in any case.
    """
    mode: SolarForecastMode = 'index'
    history_days: int = 7
    envelope_quantile: float = 0.95
    w_prev_share: float = 0.6
    w_today_max: float = 0.8
    f_on: float = 0.15
    lookback_hours: float = 5 / 60
    decay_hours: float = 1.0
    index_max: float = 1.0
    min_day_coverage: float = 0.85

    def __post_init__(self):
        if self.mode not in ('persistence', 'index'):
            raise ValueError(
                f"mode must be 'persistence' or 'index', got {self.mode!r}")
        if self.history_days < 1:
            raise ValueError(
                f"history_days must be >= 1, got {self.history_days}")
        if not 0.5 <= self.envelope_quantile <= 1.0:
            raise ValueError(
                "envelope_quantile must be in [0.5, 1.0], got "
                f"{self.envelope_quantile}")
        for name in ('w_prev_share', 'w_today_max'):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {v}")
        if not 0.0 < self.f_on <= 1.0:
            raise ValueError(f"f_on must be in (0, 1], got {self.f_on}")
        if self.lookback_hours <= 0:
            raise ValueError(
                f"lookback_hours must be > 0, got {self.lookback_hours}")
        if self.decay_hours < 0:
            raise ValueError(
                f"decay_hours must be >= 0, got {self.decay_hours}")
        if self.index_max <= 0:
            raise ValueError(f"index_max must be > 0, got {self.index_max}")
        if not 0.0 < self.min_day_coverage <= 1.0:
            raise ValueError(
                f"min_day_coverage must be in (0, 1], got "
                f"{self.min_day_coverage}")


CurrentIntervalPrice = Literal['forecast', 'price_known', 'price_and_next',
                               'next_forecast']


@dataclass(frozen=True)
class RollingConfig:
    """Settings for the rolling coordination types.

    See rolling_optimisation_spec.md (the spec covers the shared rolling
    mechanism, named after the component rather than any one coordination
    type built on it). Required when the coordination type is in
    ROLLING_COORDINATION_TYPES; must be None otherwise.

    Attributes:
        forecast_source: identifier of the ForecastProvider to use; one of
            model.forecast.registry.FORECAST_SOURCES ('perfect',
            'truncated_perfect', 'truncated_perfect_5min', 'predispatch').
        horizon_hours: planning horizon per window. Default 48; the
            production configs use 18, set by a real-data sweep on the
            Energy Masters cohort. Pre-dispatch alone only guarantees ~15.1 h
            measured from the decision instant, so anything beyond that is
            supplied by the provider's infill rule. The horizon must be long
            enough to contain a complete charge-discharge cycle from any
            decision instant, because the terminal SOC pin constrains the
            endpoint rather than the path and only bites when a cycle
            straddles the window boundary.
        infill_lag_hours: lag used to extrapolate beyond the real forecast,
            on the assumption that tomorrow looks like today. Applied
            recursively until the shifted target lands inside the forecast
            window or in the observed past. Default 24.
        current_interval_price: which price the committed interval is optimised
            against.
              'forecast'            - the 5-minute-ahead forecast (conservative).
              'price_known'         - the settled dispatch price for k, which
                                      AEMO publishes ~9 s after k begins, with
                                      the forward window still from the run
                                      available at the interval start.
              'price_and_next'      - as above, plus the run arriving ~39 s in
                                      refreshing the forward window. The most
                                      information a real controller has.
              'next_forecast'       - decomposition only. Uses the ~39 s run
                                      but prices k at that run's own estimate,
                                      ignoring the settled price it already
                                      has. Not a controller anybody builds.
        retain_plans_from / retain_plans_to: optional ISO dates bounding the
            window over which full forecast vectors and plans are retained for
            the forecast evolution diagnostics. None disables retention.
        usage_forecast: settings for the household load forecast. None means
            load is known perfectly over the horizon, which is the original
            scope of this component and remains a legitimate rung of the
            measurement ladder rather than a degenerate case.
        commit_actual_load: when True (the default) the committed interval is
            solved against ACTUAL load and only positions 1 onward use the
            forecast, so the load forecast affects the plan rather than the
            execution. Setting it False brackets the other end of the range
            and is a diagnostic, not a production setting: the export limit is
            enforced inside the LP against whatever load it is given, and
            _reconstruct_flows applies no export clip, so a forecast-load
            position 0 can produce reconstructed export above the limit.
            Ignored when usage_forecast is None.
        solar_forecast: settings for the solar forecast. None means solar is
            known perfectly over the horizon, which is a rung of the ladder
            rather than a degenerate case. commit_actual_load governs solar's
            committed interval too: the flag names what the committed interval
            is solved against, and the two site inputs are treated alike.
        on_degraded_period: what a forecast_source='predispatch' provider does
            about a period the store marks as having no usable forecast lead
            (October 2022 being the known case -- see
            model/forecast/predispatch.py). One of 'fail', 'self_consumption',
            'actuals', 'infill'; 'infill' is the recommended choice for a
            store gap like October 2022's, since it uses no special handling
            at all -- it falls through to the same 24-hour-lag infill applied
            to any ordinary horizon shortfall, using whatever real
            fine-segment coverage still exists (see predispatch.py's own
            docstring for the mechanism). Default 'fail' so the choice is
            conscious: a result that quietly substituted actuals or fell back
            to self_consumption for a month would be indistinguishable from
            one that did not. Ignored for every other forecast_source.
        max_staleness_hours: largest age a forecast_source='predispatch'
            provider may see between consecutive runs anywhere in the modelled
            period before refusing to start (see PredispatchForecastProvider's
            own docstring for why this check exists -- a store that stops
            early produces a plausible-looking number built on carried-forward
            persistence, not a forecast). Default 6.0, generous against real
            data. Raise this deliberately for a scenario spanning a known
            archive gap wider than 6 hours (e.g. a full year containing
            October 2022); it is not meant to be raised routinely. Ignored for
            every other forecast_source.
    """
    forecast_source: str
    horizon_hours: float = 48.0
    infill_lag_hours: float = 24.0
    current_interval_price: CurrentIntervalPrice = 'forecast'
    retain_plans_from: Optional[str] = None
    retain_plans_to: Optional[str] = None
    usage_forecast: Optional[UsageForecastConfig] = None
    solar_forecast: Optional[SolarForecastConfig] = None
    commit_actual_load: bool = True
    on_degraded_period: Literal['fail', 'self_consumption', 'actuals', 'infill'] = 'fail'
    max_staleness_hours: float = 6.0

    def __post_init__(self):
        if not self.forecast_source:
            raise ValueError("forecast_source is required")
        if self.horizon_hours <= 0:
            raise ValueError(
                f"horizon_hours must be > 0, got {self.horizon_hours}")
        if self.infill_lag_hours <= 0:
            raise ValueError(
                f"infill_lag_hours must be > 0, got {self.infill_lag_hours}")
        if self.current_interval_price not in (
                'forecast', 'price_known', 'price_and_next', 'next_forecast'):
            raise ValueError(
                "current_interval_price must be one of 'forecast', "
                "'price_known', 'price_and_next', 'next_forecast'; got "
                f"'{self.current_interval_price}'"
            )
        if (self.retain_plans_from is None) != (self.retain_plans_to is None):
            raise ValueError(
                "retain_plans_from and retain_plans_to must both be set or "
                "both be None"
            )
        if self.on_degraded_period not in ('fail', 'self_consumption',
                                           'actuals', 'infill'):
            raise ValueError(
                "on_degraded_period must be 'fail', 'self_consumption', "
                f"'actuals' or 'infill', got '{self.on_degraded_period}'"
            )
        if self.max_staleness_hours <= 0:
            raise ValueError(
                f"max_staleness_hours must be > 0, got {self.max_staleness_hours}")


@dataclass(frozen=True)
class ScenarioConfig:
    """Fully specifies a single model run for one (state, year, cohort) combination.

    Coordination type and battery spec are required only for the solar_battery
    cohort; for no_cer and solar cohorts there is no battery to coordinate and
    these fields must be None.
    """
    state: StateType
    year: int
    cohort: CohortType
    network_tariff_id: str                         # filename (no ext) in tariffs/ dir
    retail_tariff_id: str                          # filename (no ext) in tariffs/ dir

    coordination_type: Optional[CoordinationType] = None
    battery_spec: Optional[BatterySpec] = None
    battery_name: Optional[str] = None             # for scenario_id when multi-battery

    # Present only for the rolling coordination types
    # (ROLLING_COORDINATION_TYPES).
    rolling: Optional[RollingConfig] = None

    # Per-phase export limit (kW). Resolved per-site against the metadata phase
    # counts by orchestrate._resolve_per_site_export_limits. None => unlimited
    # (non-binding) for all sites. Consumed by perfect_foresight_lp dispatch.
    export_limit_kw_per_phase: Optional[float] = None

    # Fraction (0, 1] of the sorted opportunity list to process. Only used by
    # perfect_foresight_custom — ignored by all other coordination types.
    # 1.0 (default) = full optimisation. Lower values trade solution quality
    # for speed, allowing quick approximate runs across many battery sizes.
    opt_factor: float = 1.0

    # When True, apply the TOU export order nudge (prices.build_tou_export_order_nudge)
    # to the LP export price, biasing the LP toward charging with solar first and
    # exporting the surplus. Roughly triples HiGHS solve time on the
    # perfect-foresight LP. Accepted only for EXPORT_ORDER_NUDGE_COORDINATION_TYPES.
    # Read from the scenario YAML key apply_export_order_nudge.
    apply_export_order_nudge: bool = False

    # Operator-side retail tariff for VPP runs. Must be a wholesale_exposed
    # tariff with zero retail adders and zero daily charges, used to build the
    # operator price vectors (wholesale + network, loss-adjusted). Required
    # when coordination_type is in VPP_COORDINATION_TYPES; must be None for
    # all other types.
    op_retail_tariff_id: Optional[str] = None

    def __post_init__(self):
        if self.cohort == 'solar_battery':
            if self.coordination_type is None:
                raise ValueError(
                    "coordination_type is required when cohort='solar_battery'"
                )
            if self.coordination_type not in get_args(CoordinationType):
                raise ValueError(
                    f"unknown coordination_type '{self.coordination_type}'; "
                    f"expected one of {get_args(CoordinationType)}. A stale "
                    "or mistyped value would otherwise only surface later, "
                    "in orchestrate._dispatch_for_site -- after ingest has "
                    "already run for every site in the scenario."
                )
            if self.battery_spec is None:
                raise ValueError(
                    "battery_spec is required when cohort='solar_battery'"
                )
        else:
            if self.coordination_type is not None:
                raise ValueError(
                    f"coordination_type must be None when cohort='{self.cohort}'"
                )
            if self.battery_spec is not None:
                raise ValueError(
                    f"battery_spec must be None when cohort='{self.cohort}'"
                )
            if self.battery_name is not None:
                raise ValueError(
                    f"battery_name must be None when cohort='{self.cohort}'"
                )

        if self.coordination_type in ROLLING_COORDINATION_TYPES:
            if self.rolling is None:
                raise ValueError(
                    "rolling is required when coordination_type="
                    f"'{self.coordination_type}'"
                )
        elif self.rolling is not None:
            raise ValueError(
                "rolling must be None when "
                f"coordination_type='{self.coordination_type}'"
            )

        if (self.export_limit_kw_per_phase is not None
                and self.export_limit_kw_per_phase < 0):
            raise ValueError(
                "export_limit_kw_per_phase must be >= 0 (or None for "
                f"unlimited), got {self.export_limit_kw_per_phase}"
            )

        if not (0.0 < self.opt_factor <= 1.0):
            raise ValueError(
                f"opt_factor must be in (0, 1], got {self.opt_factor}. "
                "Only meaningful for coordination_type=perfect_foresight_custom."
            )

        if not isinstance(self.apply_export_order_nudge, bool):
            raise ValueError(
                "apply_export_order_nudge must be true or false, got "
                f"{self.apply_export_order_nudge!r}"
            )
        if (self.apply_export_order_nudge
                and self.coordination_type
                not in EXPORT_ORDER_NUDGE_COORDINATION_TYPES):
            raise ValueError(
                "apply_export_order_nudge has no effect under "
                f"coordination_type='{self.coordination_type}'; it applies to "
                f"{list(EXPORT_ORDER_NUDGE_COORDINATION_TYPES)}"
            )

        if self.coordination_type in VPP_COORDINATION_TYPES:
            if self.op_retail_tariff_id is None:
                raise ValueError(
                    "op_retail_tariff_id is required when "
                    f"coordination_type='{self.coordination_type}'"
                )
        else:
            if self.op_retail_tariff_id is not None:
                raise ValueError(
                    "op_retail_tariff_id must be None when "
                    f"coordination_type='{self.coordination_type}'"
                )

    @property
    def scenario_id(self) -> str:
        """Identifier used for output directory naming.

        Forms:
            {state}_{year}                       — single battery (or no battery)
            {state}_{year}_{battery_name}        — multi-battery cross-product
        """
        base = f"{self.state}_{self.year}"
        if self.battery_name is not None:
            base = f"{base}_{self.battery_name}"
        return base