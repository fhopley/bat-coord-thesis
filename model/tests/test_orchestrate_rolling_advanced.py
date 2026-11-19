"""Validation tests for coordination_type='rolling_perfect_foresight_lp_advanced'.

validate_scenario is the fail-fast guard for a run that otherwise costs hours,
so its branches are worth executing rather than reasoning about. These build a
throwaway data directory (metadata CSV, one site trace, one year of prices,
tariff YAMLs) and run the real function against it.

No model is executed here: validate_scenario checks inputs only.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model.core.paths import Paths
from model.core.scenario import (
    BatterySpec, RollingConfig, ScenarioConfig, UsageForecastConfig,
)
from model.orchestrate import (
    ScenarioValidationError, advanced_plan_needs_price_forecast,
    validate_scenario,
)

YEAR = 2023
STATE = 'NSW'
N_INTERVALS = 105_120          # 2023 is not a leap year
SITE = 'S0001'


ADVANCED_TOU = """\
tariff_class: retail
type: tou
name: Test Advanced TOU
state: NSW
usage_tariffs:
  - [0, 12, 51.7]
  - [12, 15, 0.0]
  - [15, 24, 59.4]
# Mirrors GloBird ZeroHero: the feed-in rate outside the paid evening window
# is a token 0.00001 c/kWh, which is what keeps it below the 0.821 c/kWh
# usage_other that prices imports inside the 12-15 block window. A real
# feed-in rate there would make importing to export profitable.
feedin_tariffs:
  - [0, 16, 0.00001]
  - [16, 18, 2.0]
  - [18, 21, 0.0]
  - [21, 23, 2.0]
  - [23, 24, 0.00001]
usage_other_c_per_kwh:
  greenpower: 0.821
daily_charges_c_per_day:
  supply: 181.5
block_import_tiers:
  - start_hour: 12
    end_hour: 15
    applies_weekends: true
    tiers:
      - threshold_kwh: 50.0
        rate_c_per_kwh: 0.0001
      - threshold_kwh: .inf
        rate_c_per_kwh: 30.8
block_export_tiers:
  - start_hour: 18
    end_hour: 21
    applies_weekends: true
    tiers:
      - threshold_kwh: 15.0
        rate_c_per_kwh: 10.0
      - threshold_kwh: .inf
        rate_c_per_kwh: 2.0
zero_import_reward:
  start_hour: 18
  end_hour: 21
  reward_c_per_day: 100.0
  tolerance_kwh_per_hour: 0.03
"""

CRITICAL_RATES = """\
critical_export_rate:
  threshold_dollars_per_mwh: 2000.0
  rate_c_per_kwh: 100.0
critical_import_rate:
  threshold_dollars_per_mwh: -100.0
  rate_c_per_kwh: -5.0
"""

# A block import window masked to a rate BELOW the feed-in rate for the same
# hours: importing to export pays, so the LP has no physical solution.
FREE_MONEY_TOU = """\
tariff_class: retail
type: tou
name: Free Money TOU
state: NSW
usage_tariffs:
  - [0, 12, 51.7]
  - [12, 15, 0.0]
  - [15, 24, 51.7]
feedin_tariffs:
  - [0, 24, 20.0]
usage_other_c_per_kwh:
  greenpower: 0.0
daily_charges_c_per_day:
  supply: 181.5
block_import_tiers:
  - start_hour: 12
    end_hour: 15
    applies_weekends: true
    tiers:
      - threshold_kwh: 50.0
        rate_c_per_kwh: 0.0001
      - threshold_kwh: .inf
        rate_c_per_kwh: 30.8
"""

WHOLESALE = """\
tariff_class: retail
type: wholesale_exposed
name: Test Wholesale
state: NSW
gst_on_wholesale_import: false
usage_other_c_per_kwh:
  market: 4.0
feedin_other_c_per_kwh:
  adjustment: 1.0
daily_charges_c_per_day:
  subscription: 82.0
"""

MIXED = """\
tariff_class: retail
type: mixed
name: Test Mixed
state: NSW
usage_tariffs:
  - [0, 15, 24.02]
  - [15, 21, 49.71]
  - [21, 24, 24.02]
usage_other_c_per_kwh:
  greenpower: 0.821
daily_charges_c_per_day:
  subscription: 110.0
wholesale_export_share: 0.9
export_share_basis: wholesale
export_floor_c_per_kwh: 0.0
"""

NETWORK = """\
tariff_class: network
name: Test Network
state: NSW
usage_tariffs:
  - [0, 24, 12.0]
feedin_tariffs:
  - [0, 24, 1.0]
daily_charges_c_per_day:
  supply: 90.0
usage_dlf: 1.05
export_dlf: 1.05
import_tlf: 1.0
export_tlf: 1.0
"""


@pytest.fixture(scope='module')
def paths(tmp_path_factory):
    root = tmp_path_factory.mktemp('data')
    tariffs = root / 'tariffs'
    tariffs.mkdir()
    (tariffs / 'test_network.yaml').write_text(NETWORK)
    (tariffs / 'test_advanced_tou.yaml').write_text(ADVANCED_TOU)
    (tariffs / 'test_advanced_tou_amp.yaml').write_text(
        ADVANCED_TOU + CRITICAL_RATES)
    (tariffs / 'test_free_money.yaml').write_text(FREE_MONEY_TOU)
    (tariffs / 'test_wholesale.yaml').write_text(WHOLESALE)
    (tariffs / 'test_mixed.yaml').write_text(MIXED)

    meta_dir = root / 'meta'
    meta_dir.mkdir()
    pd.DataFrame({'edp_site_id': [SITE], 'state': [STATE]}).to_csv(
        meta_dir / f'{YEAR}_selected_sites_clean.csv', index=False)

    idx = pd.date_range(f'{YEAR}-01-01 00:05', periods=N_INTERVALS,
                        freq='5min', tz='Etc/GMT-10')
    clean_dir = root / 'clean' / str(YEAR)
    clean_dir.mkdir(parents=True)
    (clean_dir / f'edp_clean_{SITE}_{YEAR}.parquet').write_bytes(b'')

    price_dir = root / 'prices'
    price_dir.mkdir()
    rng = np.random.default_rng(0)
    pd.DataFrame({
        'datetime_aest': idx,
        'NSW1_rrp': rng.normal(80.0, 200.0, N_INTERVALS),
        'price_flag': np.zeros(N_INTERVALS, dtype=int),
    }).to_parquet(price_dir / f'{YEAR}_nem_prices.parquet')

    return Paths(edp_clean_dir=root / 'clean', edp_metadata_dir=meta_dir,
                 nem_prices_dir=price_dir, tariffs_dir=tariffs)


def _scenario(retail='test_advanced_tou', usage_forecast=True,
              horizon_hours=18.0, coordination='rolling_perfect_foresight_lp_advanced'):
    return ScenarioConfig(
        state=STATE, year=YEAR,
        network_tariff_id='test_network', retail_tariff_id=retail,
        cohort='solar_battery',
        coordination_type=coordination,
        battery_spec=BatterySpec(
            capacity_kwh=13.5, backup_reserve_pct=10.0, power_kw=5.0,
            charge_efficiency=0.95, discharge_efficiency=0.95,
            c_deg_per_kwh=0.039),
        rolling=RollingConfig(
            forecast_source='perfect', horizon_hours=horizon_hours,
            usage_forecast=(UsageForecastConfig() if usage_forecast else None),
        ),
    )


# ---------------------------------------------------------------------------

def test_advanced_tou_scenario_validates():
    """The reference shape: block tiers, a reward, and a usage forecast."""
    validate_scenario(_scenario(), paths_fixture())


def test_wholesale_plan_rejected_on_the_advanced_path():
    with pytest.raises(ScenarioValidationError,
                       match="is for TOU and mixed plans"):
        validate_scenario(_scenario(retail='test_wholesale'), paths_fixture())


def test_mixed_plan_validates_on_price_uncertainty_alone():
    """A mixed plan's window prices come from the price forecast, so it is a
    rolling problem with perfect site knowledge."""
    validate_scenario(_scenario(retail='test_mixed', usage_forecast=False),
                      paths_fixture())


def test_nothing_uncertain_is_rejected():
    """No critical rates and no site forecast is perfect foresight over a
    truncated horizon, not a rolling result."""
    with pytest.raises(ScenarioValidationError,
                       match="nothing in this run is uncertain"):
        validate_scenario(_scenario(usage_forecast=False), paths_fixture())


def test_critical_rates_alone_are_enough():
    """A plan whose critical rates depend on the price forecast is a rolling
    problem even with perfect site knowledge."""
    validate_scenario(
        _scenario(retail='test_advanced_tou_amp', usage_forecast=False),
        paths_fixture())


def test_price_ordering_violation_is_caught_before_any_run():
    """A block window masked below the feed-in rate must fail validation, not
    surface later as a puzzling accumulator mismatch."""
    with pytest.raises(ScenarioValidationError, match="p_imp < p_exp"):
        validate_scenario(_scenario(retail='test_free_money'), paths_fixture())


def test_missing_rolling_config_is_caught_at_config_time():
    """ScenarioConfig itself refuses the combination, so validate_scenario's
    own guard is a backstop for a config built some other way rather than the
    first line of defence. Asserted here so a future relaxation of one is not
    mistaken for coverage by the other."""
    import dataclasses
    with pytest.raises(ValueError, match="rolling is required"):
        dataclasses.replace(_scenario(), rolling=None)


def test_needs_price_forecast_predicate():
    from model.core.tariff import load_retail_tariff
    p = paths_fixture()
    plain = load_retail_tariff('test_advanced_tou', p.tariffs_dir)
    amped = load_retail_tariff('test_advanced_tou_amp', p.tariffs_dir)
    mixed = load_retail_tariff('test_mixed', p.tariffs_dir)
    assert advanced_plan_needs_price_forecast(plain) is False
    assert advanced_plan_needs_price_forecast(amped) is True
    assert advanced_plan_needs_price_forecast(mixed) is True


@pytest.mark.parametrize('retail, expected', [('test_mixed', 1e-5),
                                              ('test_advanced_tou', 0.0)])
def test_rolling_advanced_site_passes_the_plan_curtail_cost(monkeypatch,
                                                            retail, expected):
    """run_rolling_perfect_foresight_lp_advanced_site hands the planner
    prices.dispatch_curtail_cost(retail_tariff): the mixed tie-break, or free
    curtailment on a TOU plan."""
    import model.orchestrate as orch
    from model.core.prices import MIXED_CURTAIL_COST_PER_KWH
    from model.core.tariff import load_network_tariff, load_retail_tariff
    assert MIXED_CURTAIL_COST_PER_KWH == 1e-5

    captured = {}

    class _Stop(Exception):
        pass

    def fake_planner(**kwargs):
        captured.update(kwargs)
        raise _Stop

    monkeypatch.setattr(orch, 'AdvancedWindowPlanner', fake_planner)
    p = paths_fixture()
    idx = pd.date_range(f'{YEAR}-01-01 00:05', periods=48, freq='5min',
                        tz='Etc/GMT-10')
    zeros = pd.Series(np.zeros(48), index=idx)
    sc = _scenario(retail=retail)
    with pytest.raises(_Stop):
        orch.run_rolling_perfect_foresight_lp_advanced_site(
            'S1', zeros, zeros, sc.battery_spec,
            load_network_tariff('test_network', p.tariffs_dir),
            load_retail_tariff(retail, p.tariffs_dir), None, sc.rolling,
            state=STATE)
    assert captured['curtail_cost_per_kwh_dollars'] == expected


# The module-scoped fixture is reached through a helper so the tests above read
# as plain calls; pytest injects it once per module.
_PATHS = {}


@pytest.fixture(autouse=True, scope='module')
def _capture_paths(paths):
    _PATHS['p'] = paths
    yield


def paths_fixture() -> Paths:
    return _PATHS['p']


def test_the_real_globird_tariff_passes_the_ordering_guard():
    """The reference plan this component is built for, loaded from the tariff
    the model actually runs on rather than a fixture that resembles it.

    Deliberately NOT a copy under tests/fixtures: the point is to fail if
    somebody edits the production YAML into a state the planner rejects. The
    guard is tight enough to have rejected a hand-written tariff whose
    feed-in rate sat above the block window's usage_other, so it is worth
    knowing the real one clears it.

    Asserts the file exists rather than skipping. A silent skip is how this
    test would quietly stop covering anything after a rename.
    """
    from model.core.paths import default_paths
    from model.core.tariff import load_network_tariff, load_retail_tariff
    from model.components.rolling_advanced import AdvancedWindowPlanner

    tariffs_dir = default_paths().tariffs_dir
    tariff_id = 'NSW_globird_zerohero_2026'
    assert (tariffs_dir / f'{tariff_id}.yaml').exists(), (
        f"{tariff_id}.yaml not found in {tariffs_dir}. Run from the project "
        f"root; if the tariff was renamed, update this test rather than "
        f"letting it skip."
    )

    rt = load_retail_tariff(tariff_id, tariffs_dir)
    assert rt.has_advanced_features
    assert advanced_plan_needs_price_forecast(rt) is True

    net = load_network_tariff('test_network', paths_fixture().tariffs_dir)
    idx = pd.date_range(f'{YEAR}-01-01 00:05', periods=288 * 8, freq='5min',
                        tz='Etc/GMT-10')
    planner = AdvancedWindowPlanner(
        index=idx, retail_tariff=rt, network_tariff=net, state=STATE,
        battery_spec=BatterySpec(
            capacity_kwh=13.5, backup_reserve_pct=10.0, power_kw=5.0,
            charge_efficiency=0.95, discharge_efficiency=0.95,
            c_deg_per_kwh=0.039),
        horizon_intervals=216)
    assert planner.requires_price_forecast is True
    assert len(planner.specs) == 2          # one import, one export window
    assert planner.n_slots == 2             # 18 h reaches at most two dates
