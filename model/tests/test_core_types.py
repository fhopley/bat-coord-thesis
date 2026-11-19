"""Tests for core types: ScenarioConfig, BatterySpec, tariff loading.

These verify the type contracts behave as designed and catch invalid
configurations early.
"""
from pathlib import Path
import pytest

from model.core import (
    ScenarioConfig,
    BatterySpec,
    RollingConfig,
    load_network_tariff,
    load_retail_tariff,
)
from model.core.scenario import UsageForecastConfig, SolarForecastConfig


FIXTURES_DIR = Path(__file__).parent / 'fixtures' / 'tariffs'


# ---------- BatterySpec ----------

def test_battery_spec_valid():
    spec = BatterySpec(
        capacity_kwh=13.5,
        backup_reserve_pct=20.0,
        power_kw=5.0,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
    )
    assert spec.capacity_kwh == 13.5


def test_battery_spec_rejects_bad_efficiency():
    with pytest.raises(ValueError, match='charge_efficiency'):
        BatterySpec(capacity_kwh=10, backup_reserve_pct=0, power_kw=5,
                    charge_efficiency=1.5, discharge_efficiency=0.9)
    with pytest.raises(ValueError, match='discharge_efficiency'):
        BatterySpec(capacity_kwh=10, backup_reserve_pct=0, power_kw=5,
                    charge_efficiency=0.9, discharge_efficiency=0)


def test_battery_spec_rejects_bad_reserve():
    with pytest.raises(ValueError, match='backup_reserve_pct'):
        BatterySpec(capacity_kwh=10, backup_reserve_pct=150, power_kw=5,
                    charge_efficiency=0.9, discharge_efficiency=0.9)


def test_battery_spec_c_deg_defaults_to_regulariser():
    spec = BatterySpec(capacity_kwh=13.5, backup_reserve_pct=20.0, power_kw=5.0,
                       charge_efficiency=0.95, discharge_efficiency=0.95)
    assert spec.c_deg_per_kwh == 0.005


def test_battery_spec_c_deg_positive_accepted():
    spec = BatterySpec(capacity_kwh=13.5, backup_reserve_pct=20.0, power_kw=5.0,
                       charge_efficiency=0.95, discharge_efficiency=0.95,
                       c_deg_per_kwh=0.185)
    assert spec.c_deg_per_kwh == 0.185


def test_battery_spec_rejects_negative_c_deg():
    with pytest.raises(ValueError, match='c_deg_per_kwh'):
        BatterySpec(capacity_kwh=13.5, backup_reserve_pct=20.0, power_kw=5.0,
                    charge_efficiency=0.95, discharge_efficiency=0.95,
                    c_deg_per_kwh=-0.01)


# ---------- ScenarioConfig ----------

def _example_battery():
    return BatterySpec(
        capacity_kwh=13.5, backup_reserve_pct=20.0, power_kw=5.0,
        charge_efficiency=0.95, discharge_efficiency=0.95,
    )


def _scenario(**overrides):
    base = dict(
        state='NSW', year=2022, cohort='no_cer',
        network_tariff_id='test_network', retail_tariff_id='test_tou',
    )
    base.update(overrides)
    return ScenarioConfig(**base)


def test_scenario_no_cer():
    s = _scenario(cohort='no_cer')
    assert s.coordination_type is None
    assert s.battery_spec is None


def test_scenario_solar():
    s = _scenario(cohort='solar')
    assert s.coordination_type is None


def test_scenario_solar_battery_requires_coordination_and_battery():
    with pytest.raises(ValueError, match='coordination_type'):
        _scenario(cohort='solar_battery')

    with pytest.raises(ValueError, match='battery_spec'):
        _scenario(cohort='solar_battery', coordination_type='self_consumption')


def test_scenario_solar_battery_valid():
    s = _scenario(cohort='solar_battery',
                  coordination_type='perfect_foresight',
                  battery_spec=_example_battery())
    assert s.coordination_type == 'perfect_foresight'


def test_scenario_rejects_battery_for_non_battery_cohort():
    with pytest.raises(ValueError, match='battery_spec must be None'):
        _scenario(cohort='solar', battery_spec=_example_battery())


# ---------- rolling_perfect_foresight_lp ----------
# Replaces the former use_rolling_window / window_days tests. Those fields were
# parsed and validated but never read by any component; they are removed and
# superseded by coordination_type='rolling_perfect_foresight_lp' plus a
# RollingConfig. See rolling_optimisation_spec.md (named after the shared
# rolling mechanism, not this specific coordination type -- see that spec's
# Configuration section for why the two names differ).

def _rolling_scenario(**overrides):
    base = dict(
        cohort='solar_battery',
        coordination_type='rolling_perfect_foresight_lp',
        battery_spec=_example_battery(),
        rolling=RollingConfig(forecast_source='predispatch'),
    )
    base.update(overrides)
    return _scenario(**base)


def test_rolling_config_defaults():
    rc = RollingConfig(forecast_source='predispatch')
    assert rc.horizon_hours == 48.0
    assert rc.infill_lag_hours == 24.0
    assert rc.current_interval_price == 'forecast'
    assert rc.retain_plans_from is None and rc.retain_plans_to is None
    assert rc.usage_forecast is None
    assert rc.solar_forecast is None
    assert rc.commit_actual_load is True
    assert rc.on_degraded_period == 'fail'
    assert rc.max_staleness_hours == 6.0


def test_rolling_config_requires_forecast_source():
    with pytest.raises(ValueError, match='forecast_source'):
        RollingConfig(forecast_source='')


@pytest.mark.parametrize('field,value', [
    ('horizon_hours', 0.0),
    ('horizon_hours', -1.0),
    ('infill_lag_hours', 0.0),
    ('infill_lag_hours', -24.0),
    ('max_staleness_hours', 0.0),
    ('max_staleness_hours', -6.0),
])
def test_rolling_config_rejects_nonpositive_durations(field, value):
    with pytest.raises(ValueError, match=field):
        RollingConfig(forecast_source='predispatch', **{field: value})


def test_rolling_config_rejects_unknown_price_mode():
    with pytest.raises(ValueError, match='current_interval_price'):
        RollingConfig(forecast_source='predispatch',
                      current_interval_price='actual')


@pytest.mark.parametrize('mode', ['forecast', 'price_known',
                                  'price_and_next', 'next_forecast'])
def test_rolling_config_accepts_all_four_price_modes(mode):
    rc = RollingConfig(forecast_source='predispatch',
                       current_interval_price=mode)
    assert rc.current_interval_price == mode


def test_rolling_config_retain_plans_must_be_paired():
    with pytest.raises(ValueError, match='both be set or'):
        RollingConfig(forecast_source='predispatch',
                      retain_plans_from='2026-02-09')
    with pytest.raises(ValueError, match='both be set or'):
        RollingConfig(forecast_source='predispatch',
                      retain_plans_to='2026-02-12')
    rc = RollingConfig(forecast_source='predispatch',
                       retain_plans_from='2026-02-09',
                       retain_plans_to='2026-02-12')
    assert rc.retain_plans_from == '2026-02-09'


def test_rolling_config_usage_and_solar_forecast_enabled_with_defaults():
    """None means perfect knowledge (a legitimate rung, not a degenerate
    case -- see RollingConfig's own docstring); {} enables the forecast at
    its fitted defaults, which is the distinction usage_forecast/
    solar_forecast: {} vs the key being absent expresses in YAML."""
    rc = RollingConfig(forecast_source='predispatch',
                       usage_forecast=UsageForecastConfig(),
                       solar_forecast=SolarForecastConfig())
    assert rc.usage_forecast is not None
    assert rc.solar_forecast is not None


def test_rolling_config_commit_actual_load_overridable():
    rc = RollingConfig(forecast_source='predispatch', commit_actual_load=False)
    assert rc.commit_actual_load is False


@pytest.mark.parametrize('policy', ['fail', 'self_consumption', 'actuals',
                                    'infill'])
def test_rolling_config_accepts_all_degraded_policies(policy):
    rc = RollingConfig(forecast_source='predispatch',
                       on_degraded_period=policy)
    assert rc.on_degraded_period == policy


def test_rolling_config_rejects_unknown_degraded_policy():
    with pytest.raises(ValueError, match='on_degraded_period'):
        RollingConfig(forecast_source='predispatch',
                      on_degraded_period='bogus')


def test_scenario_rolling_required_for_rolling_perfect_foresight_lp():
    with pytest.raises(ValueError, match='rolling is required'):
        _rolling_scenario(rolling=None)


def test_scenario_rolling_rejected_for_other_coordination_types():
    with pytest.raises(ValueError, match='rolling must be None'):
        _rolling_scenario(coordination_type='perfect_foresight')


def test_scenario_rolling_perfect_foresight_lp_valid():
    s = _rolling_scenario()
    assert s.coordination_type == 'rolling_perfect_foresight_lp'
    assert s.rolling.forecast_source == 'predispatch'


def test_scenario_rejects_unknown_coordination_type():
    """Previously this passed silently whenever `rolling` was also None,
    since __post_init__ only actively checked the rolling-specific branch --
    a mistyped or stale coordination_type (this exact string, in fact, from
    before the rolling_optimisation -> rolling_perfect_foresight_lp rename)
    would only surface much later, in orchestrate._dispatch_for_site, after
    a full ingest. ScenarioConfig now validates membership directly."""
    with pytest.raises(ValueError, match='unknown coordination_type'):
        _scenario(cohort='solar_battery',
                 coordination_type='rolling_optimisation',  # the old, dead name
                 battery_spec=_example_battery())


def test_scenario_rolling_defaults_none_for_other_types():
    """Regression guard: removing the dead fields must leave the existing
    coordination types untouched."""
    s = _scenario(cohort='solar_battery',
                  coordination_type='perfect_foresight',
                  battery_spec=_example_battery())
    assert s.rolling is None


def test_scenario_export_limit_defaults_none():
    assert _scenario().export_limit_kw_per_phase is None


def test_scenario_export_limit_positive_accepted():
    assert _scenario(export_limit_kw_per_phase=5.0).export_limit_kw_per_phase == 5.0


def test_scenario_export_limit_zero_accepted():
    # 0 is a valid physical constraint (no export permitted)
    assert _scenario(export_limit_kw_per_phase=0.0).export_limit_kw_per_phase == 0.0


def test_scenario_rejects_negative_export_limit():
    with pytest.raises(ValueError, match='export_limit_kw_per_phase'):
        _scenario(export_limit_kw_per_phase=-1.0)


def test_scenario_is_frozen():
    s = _scenario()
    with pytest.raises(Exception):  # FrozenInstanceError
        s.year = 2023


def test_scenario_id_no_battery_name():
    s = _scenario(state='NSW', year=2022)
    assert s.scenario_id == 'NSW_2022'


def test_scenario_id_with_battery_name():
    s = _scenario(cohort='solar_battery',
                  coordination_type='self_consumption',
                  battery_spec=_example_battery(),
                  battery_name='powerwall',
                  state='VIC', year=2023)
    assert s.scenario_id == 'VIC_2023_powerwall'


def test_battery_name_rejected_for_non_battery_cohort():
    with pytest.raises(ValueError, match='battery_name must be None'):
        _scenario(cohort='solar', battery_name='whatever')


# ---------- Network tariff loading ----------

def test_load_network_tariff():
    t = load_network_tariff('test_network', FIXTURES_DIR)
    assert t.state == 'NSW'
    assert len(t.usage_tariffs) == 3
    assert len(t.feedin_tariffs) == 1
    # daily charges summed: 40 + 10 = 50
    assert t.daily_c_per_day == 50.0
    assert t.usage_dlf == 1.05
    assert t.export_dlf == 0.95
    assert t.import_tlf == 1.01
    assert t.export_tlf == 1.01


def test_network_loss_factor_properties():
    t = load_network_tariff('test_network', FIXTURES_DIR)
    assert t.import_loss_factor == pytest.approx(1.05 * 1.01)
    assert t.export_loss_factor == pytest.approx(0.95 * 1.01)


def test_load_network_rejects_retail_file():
    with pytest.raises(ValueError, match='expected tariff_class: network'):
        load_network_tariff('test_tou', FIXTURES_DIR)


# ---------- Retail tariff loading ----------

def test_load_tou_retail_tariff():
    t = load_retail_tariff('test_tou', FIXTURES_DIR)
    assert t.type == 'tou'
    assert t.state == 'NSW'
    assert t.daily_c_per_day == 100.0
    assert len(t.usage_tariffs) == 3
    assert t.usage_tariffs_weekend is not None
    assert len(t.usage_tariffs_weekend) == 1
    assert len(t.feedin_tariffs) == 1
    # greenpower adder summed
    assert t.usage_other_c_per_kwh == 1.0
    assert t.feedin_other_c_per_kwh == 0.0


def test_load_wholesale_retail_tariff():
    t = load_retail_tariff('test_wholesale', FIXTURES_DIR)
    assert t.type == 'wholesale_exposed'
    assert t.usage_tariffs == ()
    assert t.usage_tariffs_weekend is None
    assert t.feedin_tariffs == ()
    assert t.usage_other_c_per_kwh == 2.0
    assert t.daily_c_per_day == 80.0


def test_load_retail_rejects_network_file():
    with pytest.raises(ValueError, match='expected tariff_class: retail'):
        load_retail_tariff('test_network', FIXTURES_DIR)


def test_load_tariff_missing_file():
    with pytest.raises(FileNotFoundError):
        load_network_tariff('does_not_exist', FIXTURES_DIR)
    with pytest.raises(FileNotFoundError):
        load_retail_tariff('does_not_exist', FIXTURES_DIR)


# ---------- Production tariffs ----------

def test_load_real_network_tariff():
    real_dir = Path(__file__).parent.parent / 'tariffs'
    if not (real_dir / 'ausgrid_ea025_network_2024.yaml').exists():
        pytest.skip("Production network tariff not present")
    t = load_network_tariff('ausgrid_ea025_network_2024', real_dir)
    assert t.state == 'NSW'
    # daily = 50.0453 + 7.3671 + 3.3973 = 60.8097
    assert t.daily_c_per_day == pytest.approx(60.8097)


def test_load_real_retail_tariffs():
    real_dir = Path(__file__).parent.parent / 'tariffs'
    for tid in ('amber_wholesale_2024', 'powershop_tou_2024'):
        if not (real_dir / f'{tid}.yaml').exists():
            pytest.skip(f"Production retail tariff {tid} not present")
        t = load_retail_tariff(tid, real_dir)
        assert t.type in ('wholesale_exposed', 'tou')


# ---------- VppConfig.curtail_c_per_kwh ----------

def _base_vpp_cfg(**overrides):
    """Minimal valid VppConfig; overrides applied on top."""
    from model.core.tariff import VppConfig
    kwargs = dict(
        exp_reward_c_per_kwh=100.0,
        imp_reward_c_per_kwh=100.0,
        kwh_limit=250.0,
        limit_applies_to='import_and_export',
        default_mode='self_consumption',
        event_window_hours=10.0,
        vpp_threshold_dollars_per_mwh=1000.0,
    )
    kwargs.update(overrides)
    return VppConfig(**kwargs)


def test_vpp_config_curtail_defaults_none():
    """curtail_c_per_kwh defaults to None when not supplied."""
    cfg = _base_vpp_cfg()
    assert cfg.curtail_c_per_kwh is None


def test_vpp_config_curtail_zero_accepted():
    """curtail_c_per_kwh=0 is a valid configuration (zero-cost curtailment)."""
    cfg = _base_vpp_cfg(curtail_c_per_kwh=0.0)
    assert cfg.curtail_c_per_kwh == 0.0


def test_vpp_config_curtail_positive_accepted():
    """A positive curtail_c_per_kwh is accepted."""
    cfg = _base_vpp_cfg(curtail_c_per_kwh=5.0)
    assert cfg.curtail_c_per_kwh == 5.0


def test_vpp_config_curtail_negative_rejected():
    """A negative curtail_c_per_kwh is rejected with a ValueError."""
    with pytest.raises(ValueError, match='curtail_c_per_kwh'):
        _base_vpp_cfg(curtail_c_per_kwh=-1.0)


def test_vpp_config_curtail_parsed_from_yaml(tmp_path):
    """curtail_c_per_kwh is parsed from YAML when present; absent key yields None."""
    import yaml
    from model.core.tariff import load_retail_tariff

    base_yaml = {
        'tariff_class': 'retail',
        'type': 'tou',
        'name': 'Test VPP',
        'state': 'NSW',
        'usage_tariffs': [[0, 24, 30.0]],
        'feedin_tariffs': [[0, 24, 5.0]],
        'usage_other_c_per_kwh': {},
        'daily_charges_c_per_day': {'supply': 100.0},
        'vpp': {
            'exp_reward_c_per_kwh': 100.0,
            'imp_reward_c_per_kwh': 100.0,
            'kwh_limit': 250.0,
            'limit_applies_to': 'export_only',
            'default_mode': 'self_consumption',
            'event_window_hours': 10.0,
            'vpp_threshold_dollars_per_mwh': 1000.0,
            # curtail_c_per_kwh absent -- should yield None
        },
    }

    # Without curtail_c_per_kwh key
    p = tmp_path / 'test_vpp_no_curtail.yaml'
    p.write_text(yaml.dump(base_yaml))
    t = load_retail_tariff('test_vpp_no_curtail', tmp_path)
    assert t.vpp is not None
    assert t.vpp.curtail_c_per_kwh is None

    # With curtail_c_per_kwh key
    base_yaml['vpp']['curtail_c_per_kwh'] = 3.5
    p2 = tmp_path / 'test_vpp_with_curtail.yaml'
    p2.write_text(yaml.dump(base_yaml))
    t2 = load_retail_tariff('test_vpp_with_curtail', tmp_path)
    assert t2.vpp is not None
    assert t2.vpp.curtail_c_per_kwh == pytest.approx(3.5)


# ---------- VppConfig.min_export_price_mode ----------
# 'based_on_last_year' is accepted by the dataclass and rejected by
# validate_scenario, so a YAML naming it fails with a message about
# implementation status rather than one about an unrecognised value. That
# split is the reason for the pair of tests below.
# See rolling_vpp_event_spec.md, "Minimum export price".

def test_vpp_config_min_export_price_mode_defaults_fixed():
    assert _base_vpp_cfg().min_export_price_mode == 'fixed'


@pytest.mark.parametrize('mode', ['fixed', 'based_on_this_year',
                                  'based_on_last_year'])
def test_vpp_config_min_export_price_mode_accepted(mode):
    """All three parse. Rejecting the unimplemented one is validate_scenario's
    job, not the dataclass's."""
    assert _base_vpp_cfg(min_export_price_mode=mode).min_export_price_mode \
        == mode


def test_vpp_config_min_export_price_mode_unknown_rejected():
    with pytest.raises(ValueError, match='min_export_price_mode'):
        _base_vpp_cfg(min_export_price_mode='based_on_next_year')


def test_vpp_config_min_export_price_mode_error_lists_every_mode():
    """The message must name the modes that ARE available, or a typo sends the
    reader to the source to find out what to write instead."""
    with pytest.raises(ValueError) as exc:
        _base_vpp_cfg(min_export_price_mode='nonsense')
    for mode in ('fixed', 'based_on_this_year', 'based_on_last_year'):
        assert mode in str(exc.value)


def test_vpp_config_min_export_price_absent_under_derived_mode():
    """Absent means a floor of zero under a derived mode, so the dataclass must
    not require it. The requirement is mode-dependent and lives in
    validate_scenario."""
    cfg = _base_vpp_cfg(min_export_price_mode='based_on_this_year')
    assert cfg.min_export_price_dollars_per_mwh is None


def test_vpp_config_min_export_price_mode_parsed_from_yaml(tmp_path):
    """Absent key yields 'fixed'; a stated mode round-trips."""
    import yaml
    from model.core.tariff import load_retail_tariff

    base_yaml = {
        'tariff_class': 'retail',
        'type': 'tou',
        'name': 'Test VPP Mode',
        'state': 'NSW',
        'usage_tariffs': [[0, 24, 30.0]],
        'feedin_tariffs': [[0, 24, 5.0]],
        'usage_other_c_per_kwh': {},
        'daily_charges_c_per_day': {'supply': 100.0},
        'vpp': {
            'exp_reward_c_per_kwh': 100.0,
            'imp_reward_c_per_kwh': 100.0,
            'kwh_limit': 250.0,
            'limit_applies_to': 'export_only',
            'default_mode': 'self_consumption',
            'event_window_hours': 10.0,
            'vpp_threshold_dollars_per_mwh': 1000.0,
            # min_export_price_mode absent -- should yield 'fixed'
        },
    }
    p = tmp_path / 'test_vpp_default_mode.yaml'
    p.write_text(yaml.dump(base_yaml))
    t = load_retail_tariff('test_vpp_default_mode', tmp_path)
    assert t.vpp is not None
    assert t.vpp.min_export_price_mode == 'fixed'
    assert t.vpp.min_export_price_dollars_per_mwh is None

    base_yaml['vpp']['min_export_price_mode'] = 'based_on_this_year'
    base_yaml['vpp']['min_export_price_dollars_per_mwh'] = 400.0
    p2 = tmp_path / 'test_vpp_this_year_mode.yaml'
    p2.write_text(yaml.dump(base_yaml))
    t2 = load_retail_tariff('test_vpp_this_year_mode', tmp_path)
    assert t2.vpp is not None
    assert t2.vpp.min_export_price_mode == 'based_on_this_year'
    assert t2.vpp.min_export_price_dollars_per_mwh == pytest.approx(400.0)


# ---------- rolling VPP coordination types ----------
# Part C adds 'rolling_vpp_sc_event' and 'rolling_vpp_tou_event'. Both are
# rolling (they need a RollingConfig and the shared causality checks) AND VPP
# (they need an op_retail_tariff_id), so they belong to two overlapping sets.
# See rolling_vpp_event_spec.md.

def _rolling_vpp_scenario(**overrides):
    base = dict(
        cohort='solar_battery',
        coordination_type='rolling_vpp_sc_event',
        battery_spec=_example_battery(),
        rolling=RollingConfig(forecast_source='predispatch'),
        op_retail_tariff_id='test_operator',
    )
    base.update(overrides)
    return _scenario(**base)


@pytest.mark.parametrize('ct', ['rolling_vpp_sc_event',
                                'rolling_vpp_tou_event'])
def test_rolling_vpp_scenario_valid(ct):
    s = _rolling_vpp_scenario(coordination_type=ct)
    assert s.coordination_type == ct
    assert s.rolling is not None
    assert s.op_retail_tariff_id == 'test_operator'


@pytest.mark.parametrize('ct', ['rolling_vpp_sc_event',
                                'rolling_vpp_tou_event'])
def test_rolling_vpp_requires_rolling_config(ct):
    with pytest.raises(ValueError, match='rolling is required'):
        _rolling_vpp_scenario(coordination_type=ct, rolling=None)


@pytest.mark.parametrize('ct', ['rolling_vpp_sc_event',
                                'rolling_vpp_tou_event'])
def test_rolling_vpp_requires_op_retail_tariff(ct):
    with pytest.raises(ValueError, match='op_retail_tariff_id is required'):
        _rolling_vpp_scenario(coordination_type=ct, op_retail_tariff_id=None)


def test_rolling_vpp_types_are_in_both_coordination_sets():
    """Both sets are tested independently elsewhere, but a type that made it
    into only one would fail in a confusing place: missing from the rolling
    set it would skip the shared causality checks and be rejected for
    supplying a RollingConfig; missing from the VPP set it would be rejected
    for supplying op_retail_tariff_id."""
    from model.core.scenario import (
        ROLLING_COORDINATION_TYPES, VPP_COORDINATION_TYPES,
    )
    for ct in ('rolling_vpp_sc_event', 'rolling_vpp_tou_event'):
        assert ct in ROLLING_COORDINATION_TYPES
        assert ct in VPP_COORDINATION_TYPES


def test_coordination_type_sets_are_subsets_of_the_literal():
    """A set member absent from CoordinationType would be rejected by
    __post_init__'s membership check before any set-based branch could run,
    so the branch would be unreachable rather than wrong."""
    from typing import get_args
    from model.core.scenario import (
        CoordinationType, ROLLING_COORDINATION_TYPES, VPP_COORDINATION_TYPES,
    )
    literal = set(get_args(CoordinationType))
    assert set(ROLLING_COORDINATION_TYPES) <= literal
    assert set(VPP_COORDINATION_TYPES) <= literal


def test_perfect_foresight_vpp_types_still_reject_rolling_config():
    """Regression: the perfect-foresight VPP types joined VPP_COORDINATION_TYPES
    but must not have joined the rolling set."""
    with pytest.raises(ValueError, match='rolling must be None'):
        _rolling_vpp_scenario(coordination_type='vpp_sc_event')


def test_non_vpp_types_still_reject_op_retail_tariff():
    """Regression: switching the op_retail_tariff_id check from a hardcoded
    pair to VPP_COORDINATION_TYPES must not have widened it."""
    with pytest.raises(ValueError, match='op_retail_tariff_id must be None'):
        _scenario(cohort='solar_battery',
                  coordination_type='perfect_foresight',
                  battery_spec=_example_battery(),
                  op_retail_tariff_id='test_operator')


# ---------- VppConfig.min_export_price_mode ----------

def test_vpp_config_min_export_price_mode_defaults_to_fixed():
    """Must default to 'fixed' so existing tariff YAML, which has no such key,
    keeps its current behaviour."""
    cfg = _base_vpp_cfg()
    assert cfg.min_export_price_mode == 'fixed'


@pytest.mark.parametrize('mode', ['fixed', 'based_on_last_year'])
def test_vpp_config_min_export_price_mode_accepted(mode):
    cfg = _base_vpp_cfg(min_export_price_mode=mode)
    assert cfg.min_export_price_mode == mode


def test_vpp_config_rejects_unknown_min_export_price_mode():
    with pytest.raises(ValueError, match='min_export_price_mode'):
        _base_vpp_cfg(min_export_price_mode='adaptive')


def test_vpp_config_min_export_price_mode_parsed_from_yaml(tmp_path):
    """Present in YAML it is read; absent it yields 'fixed'."""
    import yaml
    from model.core.tariff import load_retail_tariff

    base_yaml = {
        'tariff_class': 'retail',
        'type': 'tou',
        'name': 'Test VPP',
        'state': 'NSW',
        'usage_tariffs': [[0, 24, 30.0]],
        'feedin_tariffs': [[0, 24, 5.0]],
        'usage_other_c_per_kwh': {},
        'daily_charges_c_per_day': {'supply': 100.0},
        'vpp': {
            'exp_reward_c_per_kwh': 100.0,
            'imp_reward_c_per_kwh': 100.0,
            'kwh_limit': 250.0,
            'limit_applies_to': 'export_only',
            'default_mode': 'self_consumption',
            'event_window_hours': 10.0,
            'vpp_threshold_dollars_per_mwh': 1000.0,
            # min_export_price_mode absent -- should yield 'fixed'
        },
    }

    p = tmp_path / 'test_vpp_no_mode.yaml'
    p.write_text(yaml.dump(base_yaml))
    t = load_retail_tariff('test_vpp_no_mode', tmp_path)
    assert t.vpp is not None
    assert t.vpp.min_export_price_mode == 'fixed'

    base_yaml['vpp']['min_export_price_mode'] = 'based_on_last_year'
    p2 = tmp_path / 'test_vpp_with_mode.yaml'
    p2.write_text(yaml.dump(base_yaml))
    t2 = load_retail_tariff('test_vpp_with_mode', tmp_path)
    assert t2.vpp is not None
    assert t2.vpp.min_export_price_mode == 'based_on_last_year'


# ---------- Mixed retail tariff loading (cap_account_spec.md Section 18) ----------

_MIXED_BASE = """\
tariff_class: retail
type: mixed
name: TestMixed
state: NSW
usage_tariffs:
  - [0, 15, 24.0]
  - [15, 21, 50.0]
  - [21, 24, 24.0]
daily_charges_c_per_day:
  subscription: 110.0
"""

# The two export terms every mixed file must carry besides the share.
_MIXED_TERMS = "export_share_basis: wholesale\nexport_floor_c_per_kwh: 0.0\n"


def _write_retail(tmp_path, body, tid='t'):
    (tmp_path / f'{tid}.yaml').write_text(body)
    return tid


def test_load_mixed_retail_tariff():
    t = load_retail_tariff('test_mixed', FIXTURES_DIR)
    assert t.type == 'mixed'
    assert t.wholesale_export_share == 0.9
    assert t.export_share_basis == 'wholesale'
    assert t.export_floor_c_per_kwh == 0.0
    assert len(t.usage_tariffs) == 3
    assert t.usage_tariffs_weekend is None
    assert t.feedin_tariffs == ()
    assert t.usage_other_c_per_kwh == 1.0
    assert t.feedin_other_c_per_kwh == 0.0
    assert t.daily_c_per_day == 110.0
    assert not t.has_advanced_features


def test_share_is_none_on_other_types():
    for tid in ('test_tou', 'test_wholesale'):
        t = load_retail_tariff(tid, FIXTURES_DIR)
        assert t.wholesale_export_share is None
        assert t.export_share_basis is None
        assert t.export_floor_c_per_kwh is None


@pytest.mark.parametrize('share', [0, 1, 0.0, 1.0, 0.5])
def test_mixed_accepts_share_in_unit_interval(tmp_path, share):
    tid = _write_retail(tmp_path, _MIXED_BASE + _MIXED_TERMS
                        + f"wholesale_export_share: {share}\n")
    assert load_retail_tariff(tid, tmp_path).wholesale_export_share == float(share)


def test_mixed_requires_share(tmp_path):
    tid = _write_retail(tmp_path, _MIXED_BASE + _MIXED_TERMS)
    with pytest.raises(ValueError, match='requires wholesale_export_share'):
        load_retail_tariff(tid, tmp_path)


@pytest.mark.parametrize('share', ['-0.1', '1.5', '.nan'])
def test_mixed_rejects_share_outside_unit_interval(tmp_path, share):
    tid = _write_retail(tmp_path, _MIXED_BASE + f"wholesale_export_share: {share}\n")
    with pytest.raises(ValueError, match=r'must be in \[0, 1\]'):
        load_retail_tariff(tid, tmp_path)


@pytest.mark.parametrize('share', ['"0.9"', 'true', '[0.9]'])
def test_mixed_rejects_non_numeric_share(tmp_path, share):
    tid = _write_retail(tmp_path, _MIXED_BASE + f"wholesale_export_share: {share}\n")
    with pytest.raises(ValueError, match='must be a number'):
        load_retail_tariff(tid, tmp_path)


def test_mixed_rejects_feedin_tariffs(tmp_path):
    tid = _write_retail(tmp_path, _MIXED_BASE + _MIXED_TERMS
                        + "wholesale_export_share: 0.9\n"
                        + "feedin_tariffs:\n  - [0, 24, 0.0]\n")
    with pytest.raises(ValueError, match='must not define feedin_tariffs'):
        load_retail_tariff(tid, tmp_path)


def test_mixed_requires_usage_tariffs(tmp_path):
    body = _MIXED_BASE.split('usage_tariffs:')[0] + (
        "daily_charges_c_per_day:\n  subscription: 110.0\n"
        "wholesale_export_share: 0.9\n" + _MIXED_TERMS)
    tid = _write_retail(tmp_path, body)
    with pytest.raises(ValueError, match='mixed tariff requires usage_tariffs'):
        load_retail_tariff(tid, tmp_path)


def test_mixed_rejects_advanced_features(tmp_path):
    tid = _write_retail(tmp_path, _MIXED_BASE + _MIXED_TERMS
                        + "wholesale_export_share: 0.9\n"
                        + "zero_import_reward:\n  start_hour: 7\n  end_hour: 9\n"
                        + "  reward_c_per_day: 100.0\n")
    with pytest.raises(ValueError, match='must not define advanced features'):
        load_retail_tariff(tid, tmp_path)


def test_share_rejected_on_tou(tmp_path):
    body = (FIXTURES_DIR / 'test_tou.yaml').read_text() + "\nwholesale_export_share: 0.9\n"
    tid = _write_retail(tmp_path, body)
    with pytest.raises(ValueError, match="only valid on a mixed tariff, not 'tou'"):
        load_retail_tariff(tid, tmp_path)


def test_share_rejected_on_wholesale_exposed(tmp_path):
    body = (FIXTURES_DIR / 'test_wholesale.yaml').read_text() + "\nwholesale_export_share: 0.9\n"
    tid = _write_retail(tmp_path, body)
    with pytest.raises(ValueError, match="only valid on a mixed tariff, not 'wholesale_exposed'"):
        load_retail_tariff(tid, tmp_path)


@pytest.mark.parametrize('basis', ['wholesale', 'wholesale_and_network'])
@pytest.mark.parametrize('floor_yaml, floor', [('0.0', 0.0), ('-2.5', -2.5),
                                               ('3', 3.0), ('null', None)])
def test_mixed_accepts_basis_and_floor(tmp_path, basis, floor_yaml, floor):
    tid = _write_retail(tmp_path, _MIXED_BASE + "wholesale_export_share: 0.9\n"
                        + f"export_share_basis: {basis}\n"
                        + f"export_floor_c_per_kwh: {floor_yaml}\n")
    t = load_retail_tariff(tid, tmp_path)
    assert t.export_share_basis == basis
    assert t.export_floor_c_per_kwh == floor


def test_mixed_requires_basis(tmp_path):
    tid = _write_retail(tmp_path, _MIXED_BASE + "wholesale_export_share: 0.9\n"
                        + "export_floor_c_per_kwh: 0.0\n")
    with pytest.raises(ValueError, match='requires export_share_basis'):
        load_retail_tariff(tid, tmp_path)


def test_mixed_requires_floor_key(tmp_path):
    # An absent key is an error even though null is accepted: a mixed file
    # must say explicitly whether its export credit is floored.
    tid = _write_retail(tmp_path, _MIXED_BASE + "wholesale_export_share: 0.9\n"
                        + "export_share_basis: wholesale\n")
    with pytest.raises(ValueError, match='requires export_floor_c_per_kwh'):
        load_retail_tariff(tid, tmp_path)


@pytest.mark.parametrize('basis', ['network', 'Wholesale', '1', 'null'])
def test_mixed_rejects_unknown_basis(tmp_path, basis):
    tid = _write_retail(tmp_path, _MIXED_BASE + "wholesale_export_share: 0.9\n"
                        + f"export_share_basis: {basis}\n"
                        + "export_floor_c_per_kwh: 0.0\n")
    with pytest.raises(ValueError, match='export_share_basis must be'):
        load_retail_tariff(tid, tmp_path)


@pytest.mark.parametrize('floor, match', [('"0"', 'must be a number or null'),
                                          ('true', 'must be a number or null'),
                                          ('[0]', 'must be a number or null'),
                                          ('.inf', 'must be finite'),
                                          ('.nan', 'must be finite')])
def test_mixed_rejects_bad_floor(tmp_path, floor, match):
    tid = _write_retail(tmp_path, _MIXED_BASE + "wholesale_export_share: 0.9\n"
                        + "export_share_basis: wholesale\n"
                        + f"export_floor_c_per_kwh: {floor}\n")
    with pytest.raises(ValueError, match=match):
        load_retail_tariff(tid, tmp_path)


@pytest.mark.parametrize('fixture, rtype', [('test_tou', 'tou'),
                                            ('test_wholesale', 'wholesale_exposed')])
@pytest.mark.parametrize('line', ['export_share_basis: wholesale',
                                  'export_floor_c_per_kwh: 0.0',
                                  'export_floor_c_per_kwh: null'])
def test_export_terms_rejected_on_other_types(tmp_path, fixture, rtype, line):
    body = (FIXTURES_DIR / f'{fixture}.yaml').read_text() + f"\n{line}\n"
    tid = _write_retail(tmp_path, body)
    key = line.split(':')[0]
    with pytest.raises(ValueError,
                       match=f"{key} is only valid on a mixed tariff, not '{rtype}'"):
        load_retail_tariff(tid, tmp_path)


def test_load_real_mixed_tariffs():
    real_dir = Path(__file__).parent.parent / 'tariffs'
    for state in ('NSW', 'QLD', 'SA', 'VIC'):
        tid = f'{state}_tesla_electric_mixed_2026'
        if not (real_dir / f'{tid}.yaml').exists():
            pytest.skip(f"Production retail tariff {tid} not present")
        t = load_retail_tariff(tid, real_dir)
        assert t.type == 'mixed'
        assert t.state == state
        assert t.wholesale_export_share == 0.9
        assert t.export_share_basis == 'wholesale'
        assert t.export_floor_c_per_kwh == 0.0
        assert t.feedin_tariffs == ()


# ---------------- gst_on_wholesale_import ----------------

def _wholesale_body_without_gst_key() -> str:
    body = (FIXTURES_DIR / 'test_wholesale.yaml').read_text()
    lines = [l for l in body.splitlines()
             if not l.startswith('gst_on_wholesale_import')]
    assert len(lines) == len(body.splitlines()) - 1   # the fixture had the key
    return '\n'.join(lines) + '\n'


@pytest.mark.parametrize('value, expected', [('true', True), ('false', False)])
def test_gst_key_read_on_wholesale_exposed(tmp_path, value, expected):
    tid = _write_retail(tmp_path, _wholesale_body_without_gst_key()
                        + f"gst_on_wholesale_import: {value}\n")
    assert load_retail_tariff(tid, tmp_path).gst_on_wholesale_import is expected


def test_gst_key_required_on_wholesale_exposed(tmp_path):
    tid = _write_retail(tmp_path, _wholesale_body_without_gst_key())
    with pytest.raises(ValueError, match='requires gst_on_wholesale_import'):
        load_retail_tariff(tid, tmp_path)


@pytest.mark.parametrize('value', ['1', '"true"', 'yes_please', 'null'])
def test_gst_key_must_be_boolean(tmp_path, value):
    tid = _write_retail(tmp_path, _wholesale_body_without_gst_key()
                        + f"gst_on_wholesale_import: {value}\n")
    with pytest.raises(ValueError, match='must be true or false'):
        load_retail_tariff(tid, tmp_path)


@pytest.mark.parametrize('value', ['true', 'false'])
def test_gst_key_rejected_on_tou(tmp_path, value):
    body = ((FIXTURES_DIR / 'test_tou.yaml').read_text()
            + f"\ngst_on_wholesale_import: {value}\n")
    tid = _write_retail(tmp_path, body)
    with pytest.raises(ValueError, match="only valid on a wholesale_exposed "
                                         "tariff, not 'tou'"):
        load_retail_tariff(tid, tmp_path)


def test_gst_key_rejected_on_mixed(tmp_path):
    tid = _write_retail(tmp_path, _MIXED_BASE + _MIXED_TERMS
                        + "wholesale_export_share: 0.9\n"
                        + "gst_on_wholesale_import: false\n")
    with pytest.raises(ValueError, match="only valid on a wholesale_exposed "
                                         "tariff, not 'mixed'"):
        load_retail_tariff(tid, tmp_path)


def test_gst_key_on_repo_wholesale_tariffs():
    """Household Amber plans pay GST on spot imports; operator tariffs do
    not."""
    tariffs_dir = Path(__file__).resolve().parents[1] / 'tariffs'
    seen = {'amber': 0, 'operator': 0}
    for p in sorted(tariffs_dir.glob('*.yaml')):
        if 'type: wholesale_exposed' not in p.read_text():
            continue
        t = load_retail_tariff(p.stem, tariffs_dir)
        kind = 'operator' if 'operator' in p.stem else 'amber'
        assert 'amber' in p.stem or kind == 'operator', p.stem
        seen[kind] += 1
        assert t.gst_on_wholesale_import is (kind == 'amber'), p.stem
    assert seen['amber'] >= 5 and seen['operator'] >= 7, seen
