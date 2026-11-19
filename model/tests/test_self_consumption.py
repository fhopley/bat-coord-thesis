"""Tests for self_consumption.py.

Coverage:
  - no-battery case (vectorised path)
  - with-battery happy paths (charge, discharge, idle)
  - power limit enforcement (AC side, no efficiency factor)
  - DC capacity ceiling (full battery)
  - backup reserve floor (won't discharge below reserve)
  - efficiency bookkeeping (energy in vs energy out)
  - flow invariants (load and solar balance)
  - import/export = decomposition aggregation
  - the no_cer case (solar=0) reduces to grid_to_home == load
  - cross-check vs ac_load_net for solar-only case
  - input validation
  - 30-min cadence via interval_hours parameter

All tests use small hand-constructable inputs with analytically known
expected outputs, not random data.
"""
import numpy as np
import pandas as pd
import pytest

from model.core.scenario import BatterySpec
from model.components.self_consumption import self_consumption


def _make_index(n: int, freq: str = '5min') -> pd.DatetimeIndex:
    return pd.date_range('2022-01-01 00:05:00', periods=n, freq=freq, tz='Etc/GMT-10')


def _basic_battery(
    capacity_kwh: float = 10.0,
    backup_reserve_pct: float = 0.0,
    power_kw: float = 5.0,
    charge_efficiency: float = 1.0,
    discharge_efficiency: float = 1.0,
) -> BatterySpec:
    return BatterySpec(
        capacity_kwh=capacity_kwh,
        backup_reserve_pct=backup_reserve_pct,
        power_kw=power_kw,
        charge_efficiency=charge_efficiency,
        discharge_efficiency=discharge_efficiency,
    )


# ---------- No-battery cases ----------

def test_no_battery_solar_exceeds_load():
    """Solar 0.4, load 0.1 -> sol_to_home 0.1, sol_to_grid 0.3, no import."""
    idx = _make_index(3)
    solar = pd.Series([0.4, 0.4, 0.4], index=idx)
    load = pd.Series([0.1, 0.1, 0.1], index=idx)

    t = self_consumption('S0001', solar, load, battery_spec=None)

    np.testing.assert_array_almost_equal(t.sol_to_home_kwh, [0.1, 0.1, 0.1])
    np.testing.assert_array_almost_equal(t.sol_to_grid_kwh, [0.3, 0.3, 0.3])
    np.testing.assert_array_almost_equal(t.grid_to_home_kwh, [0, 0, 0])
    np.testing.assert_array_almost_equal(t.export_kwh, [0.3, 0.3, 0.3])
    np.testing.assert_array_almost_equal(t.import_kwh, [0, 0, 0])


def test_no_battery_load_exceeds_solar():
    """Solar 0.1, load 0.4 -> sol_to_home 0.1, grid_to_home 0.3, no export."""
    idx = _make_index(3)
    solar = pd.Series([0.1, 0.1, 0.1], index=idx)
    load = pd.Series([0.4, 0.4, 0.4], index=idx)

    t = self_consumption('S0001', solar, load, battery_spec=None)

    np.testing.assert_array_almost_equal(t.sol_to_home_kwh, [0.1, 0.1, 0.1])
    np.testing.assert_array_almost_equal(t.grid_to_home_kwh, [0.3, 0.3, 0.3])
    np.testing.assert_array_almost_equal(t.sol_to_grid_kwh, [0, 0, 0])
    np.testing.assert_array_almost_equal(t.import_kwh, [0.3, 0.3, 0.3])
    np.testing.assert_array_almost_equal(t.export_kwh, [0, 0, 0])


def test_no_battery_all_battery_columns_zero():
    """With no battery, all battery flow columns must be zero."""
    idx = _make_index(3)
    solar = pd.Series([0.2, 0.5, 0.0], index=idx)
    load = pd.Series([0.1, 0.1, 0.4], index=idx)

    t = self_consumption('S0001', solar, load, battery_spec=None)

    assert (t.sol_to_bat_kwh == 0).all()
    assert (t.bat_to_home_kwh == 0).all()
    assert (t.grid_to_bat_kwh == 0).all()
    assert (t.bat_to_grid_kwh == 0).all()
    assert (t.battery_soc_kwh == 0).all()


def test_no_cer_solar_zero_means_import_equals_load():
    """no_cer cohort: solar is zero, so grid covers all load."""
    idx = _make_index(5)
    solar = pd.Series([0, 0, 0, 0, 0], index=idx, dtype=float)
    load = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5], index=idx)

    t = self_consumption('S0001', solar, load, battery_spec=None)

    np.testing.assert_array_almost_equal(t.import_kwh, load)
    np.testing.assert_array_almost_equal(t.export_kwh, np.zeros(5))


def test_no_battery_energy_conservation():
    """For solar-only (no battery): import - export = load - solar.
    This is energy conservation: net grid flow equals net household flow.
    Equivalent to ac_load_net for intervals where load >= solar."""
    idx = _make_index(4)
    solar = pd.Series([0.5, 0.2, 0.0, 0.8], index=idx)
    load = pd.Series([0.3, 0.3, 0.3, 0.3], index=idx)

    t = self_consumption('S0001', solar, load, battery_spec=None)

    # Energy conservation across the meter boundary
    net_grid = t.import_kwh - t.export_kwh
    net_household = load - solar
    np.testing.assert_array_almost_equal(net_grid.values, net_household.values)


# ---------- With-battery happy paths ----------

def test_with_battery_charges_from_surplus_solar():
    """Solar > load: surplus goes to battery up to its capacity, then to grid."""
    idx = _make_index(3)
    # 5 min intervals, power_kw=12 -> 1 kWh/interval limit
    # capacity 2 kWh, start at 0 (reserve=0), 100% efficiency
    solar = pd.Series([1.5, 1.5, 1.5], index=idx)
    load = pd.Series([0.0, 0.0, 0.0], index=idx)

    battery = _basic_battery(capacity_kwh=2.0, power_kw=12.0)
    t = self_consumption('S0001', solar, load, battery)

    # Interval 0: surplus 1.5, AC cap = 12*5/60 = 1.0, DC headroom = 2/1 = 2
    #   sol_to_bat = 1.0, soc -> 1.0, sol_to_grid = 0.5
    # Interval 1: surplus 1.5, AC cap 1.0, DC headroom (2-1)/1 = 1.0
    #   sol_to_bat = 1.0, soc -> 2.0, sol_to_grid = 0.5
    # Interval 2: surplus 1.5, AC cap 1.0, DC headroom 0
    #   sol_to_bat = 0, soc stays 2.0, sol_to_grid = 1.5
    np.testing.assert_array_almost_equal(t.sol_to_bat_kwh, [1.0, 1.0, 0.0])
    np.testing.assert_array_almost_equal(t.sol_to_grid_kwh, [0.5, 0.5, 1.5])
    np.testing.assert_array_almost_equal(t.battery_soc_kwh, [1.0, 2.0, 2.0])


def test_with_battery_discharges_to_load():
    """Load > solar: battery discharges to cover residual."""
    idx = _make_index(3)
    solar = pd.Series([0.0, 0.0, 0.0], index=idx)
    load = pd.Series([0.5, 0.5, 0.5], index=idx)

    # Start battery at half full (1 kWh) by giving it a charging interval first?
    # No - just use a small enough battery that we can predict the discharge curve.
    # capacity 1 kWh, but we need it to start with energy. Use reserve as initial.
    # Reserve 50% of 1 kWh = 0.5 kWh - but then it can never discharge.
    # Easier: use small capacity, no reserve, but seed initial state via solar burst.
    idx = _make_index(5)
    solar = pd.Series([2.0, 0.0, 0.0, 0.0, 0.0], index=idx)  # burst-charge first
    load = pd.Series([0.0, 0.4, 0.4, 0.4, 0.4], index=idx)

    # AC cap = 12*5/60 = 1, capacity 1, no reserve, 100% efficient.
    battery = _basic_battery(capacity_kwh=1.0, power_kw=12.0)
    t = self_consumption('S0001', solar, load, battery)

    # Interval 0: solar 2, load 0. sol_to_home=0, surplus=2.
    #   sol_to_bat = min(2, 1, 1) = 1. soc=1. sol_to_grid = 1.
    # Interval 1: solar 0, load 0.4. b2h = min(0.4, 1, 1*1) = 0.4. soc=0.6.
    # Interval 2: b2h = 0.4. soc=0.2.
    # Interval 3: b2h = min(0.4, 1, 0.2) = 0.2. soc=0.
    # Interval 4: b2h = 0. grid_to_home = 0.4.
    np.testing.assert_array_almost_equal(t.sol_to_bat_kwh, [1.0, 0, 0, 0, 0])
    np.testing.assert_array_almost_equal(t.bat_to_home_kwh, [0, 0.4, 0.4, 0.2, 0])
    np.testing.assert_array_almost_equal(t.grid_to_home_kwh, [0, 0, 0, 0.2, 0.4])
    np.testing.assert_array_almost_equal(t.battery_soc_kwh, [1.0, 0.6, 0.2, 0.0, 0.0])


def test_battery_idle_when_solar_equals_load():
    idx = _make_index(3)
    solar = pd.Series([0.2, 0.2, 0.2], index=idx)
    load = pd.Series([0.2, 0.2, 0.2], index=idx)

    battery = _basic_battery(capacity_kwh=2.0, power_kw=5.0)
    t = self_consumption('S0001', solar, load, battery)

    np.testing.assert_array_almost_equal(t.sol_to_bat_kwh, [0, 0, 0])
    np.testing.assert_array_almost_equal(t.bat_to_home_kwh, [0, 0, 0])
    np.testing.assert_array_almost_equal(t.battery_soc_kwh, [0, 0, 0])
    np.testing.assert_array_almost_equal(t.sol_to_home_kwh, [0.2, 0.2, 0.2])


# ---------- Constraint enforcement ----------

def test_power_limit_caps_charge_rate():
    """power_kw is the AC-side limit - no efficiency factor applied."""
    idx = _make_index(3)
    # Want surplus 1 kWh/interval but power cap of 0.5 kWh/interval
    solar = pd.Series([1.0, 1.0, 1.0], index=idx)
    load = pd.Series([0.0, 0.0, 0.0], index=idx)

    # power_kw=6 -> 6*5/60 = 0.5 kWh per interval AC cap
    battery = _basic_battery(capacity_kwh=10.0, power_kw=6.0)
    t = self_consumption('S0001', solar, load, battery)

    # AC cap binds, not DC capacity. With 100% efficiency, soc increase = sol_to_bat.
    np.testing.assert_array_almost_equal(t.sol_to_bat_kwh, [0.5, 0.5, 0.5])
    np.testing.assert_array_almost_equal(t.sol_to_grid_kwh, [0.5, 0.5, 0.5])
    np.testing.assert_array_almost_equal(t.battery_soc_kwh, [0.5, 1.0, 1.5])


def test_power_limit_caps_discharge_rate():
    """power_kw also limits discharge AC outflow."""
    idx = _make_index(4)
    solar = pd.Series([5.0, 0.0, 0.0, 0.0], index=idx)
    load = pd.Series([0.0, 1.0, 1.0, 1.0], index=idx)

    # power_kw=6 -> 0.5 kWh per interval; capacity 5 kWh
    battery = _basic_battery(capacity_kwh=5.0, power_kw=6.0)
    t = self_consumption('S0001', solar, load, battery)

    # Interval 0: charges 0.5 (AC capped). soc=0.5. sol_to_grid=4.5.
    # Interval 1: load 1, b2h = min(1, 0.5, 0.5) = 0.5. soc=0. grid_to_home=0.5.
    # Interval 2: b2h = 0. grid_to_home = 1.
    np.testing.assert_array_almost_equal(t.sol_to_bat_kwh, [0.5, 0, 0, 0])
    np.testing.assert_array_almost_equal(t.bat_to_home_kwh, [0, 0.5, 0, 0])
    np.testing.assert_array_almost_equal(t.grid_to_home_kwh, [0, 0.5, 1.0, 1.0])


def test_backup_reserve_prevents_discharge_below_floor():
    """With 20% reserve on a 5 kWh battery, only 4 kWh is discharge-able."""
    idx = _make_index(10)
    # Burst-fill, then drain
    solar = pd.Series([5.0] + [0.0] * 9, index=idx)
    load = pd.Series([0.0] + [0.5] * 9, index=idx)

    # power_kw=12 -> 1 kWh AC cap; capacity 5; reserve 20% -> min_soc=1
    battery = _basic_battery(
        capacity_kwh=5.0, backup_reserve_pct=20.0, power_kw=12.0
    )
    t = self_consumption('S0001', solar, load, battery)

    # Interval 0: starts at reserve=1.0. surplus 5, AC cap 1, DC headroom = (5-1)/1 = 4
    #   sol_to_bat = 1. soc=2.0. sol_to_grid=4.
    # Intervals 1..: discharge 0.5 each until soc reaches min_soc=1.
    #   Discharges from 2.0 to 1.0 = 2 intervals of 0.5.
    # Then bat_to_home = 0; grid_to_home = 0.5.
    np.testing.assert_array_almost_equal(t.bat_to_home_kwh[1:3], [0.5, 0.5])
    np.testing.assert_array_almost_equal(t.bat_to_home_kwh[3:], [0] * 7)
    np.testing.assert_array_almost_equal(t.grid_to_home_kwh[3:], [0.5] * 7)
    # SOC never drops below reserve
    assert (t.battery_soc_kwh >= 1.0 - 1e-9).all()


def test_charge_efficiency():
    """With 90% charge efficiency: 1 kWh AC in -> 0.9 kWh DC stored."""
    idx = _make_index(2)
    solar = pd.Series([0.5, 0.0], index=idx)
    load = pd.Series([0.0, 0.0], index=idx)

    battery = _basic_battery(capacity_kwh=10.0, power_kw=12.0,
                             charge_efficiency=0.9)
    t = self_consumption('S0001', solar, load, battery)

    # sol_to_bat = min(0.5, 1.0 AC cap, (10-0)/0.9 = 11.1) = 0.5
    # SOC increase = 0.5 * 0.9 = 0.45
    np.testing.assert_almost_equal(t.sol_to_bat_kwh.iloc[0], 0.5)
    np.testing.assert_almost_equal(t.battery_soc_kwh.iloc[0], 0.45)


def test_discharge_efficiency():
    """With 90% discharge efficiency: 1 kWh DC removed -> 0.9 kWh AC delivered."""
    idx = _make_index(3)
    solar = pd.Series([2.0, 0.0, 0.0], index=idx)
    load = pd.Series([0.0, 0.5, 0.5], index=idx)

    battery = _basic_battery(capacity_kwh=2.0, power_kw=12.0,
                             discharge_efficiency=0.9)
    t = self_consumption('S0001', solar, load, battery)

    # Int 0: charges 1.0 (AC cap), soc=1.0.
    # Int 1: residual load 0.5. AC cap 1.0. DC available in AC = (1.0-0)*0.9 = 0.9.
    #   b2h = min(0.5, 1.0, 0.9) = 0.5. soc = 1.0 - 0.5/0.9 = 0.4444...
    np.testing.assert_almost_equal(t.bat_to_home_kwh.iloc[1], 0.5)
    np.testing.assert_almost_equal(t.battery_soc_kwh.iloc[1], 1.0 - 0.5 / 0.9)


def test_round_trip_efficiency_loss():
    """1 kWh AC in then back out at 90%/90% = 0.81 kWh AC delivered."""
    idx = _make_index(20)
    # Charge first interval, then idle, then discharge until empty
    solar = np.zeros(20)
    solar[0] = 5.0
    load = np.zeros(20)
    load[1:] = 0.2
    solar_s = pd.Series(solar, index=idx)
    load_s = pd.Series(load, index=idx)

    battery = _basic_battery(capacity_kwh=10.0, power_kw=12.0,
                             charge_efficiency=0.9, discharge_efficiency=0.9)
    t = self_consumption('S0001', solar_s, load_s, battery)

    # AC into battery: 1 kWh (AC cap). DC stored: 0.9. AC delivered: 0.9*0.9 = 0.81.
    total_charge_ac = t.sol_to_bat_kwh.sum()
    total_discharge_ac = t.bat_to_home_kwh.sum()
    assert total_charge_ac == pytest.approx(1.0)
    assert total_discharge_ac == pytest.approx(0.81)


# ---------- Invariants ----------

def test_load_balance_invariant():
    """For every interval: load == sol_to_home + bat_to_home + grid_to_home."""
    idx = _make_index(20)
    rng = np.random.default_rng(seed=42)
    solar = pd.Series(rng.uniform(0, 2, 20), index=idx)
    load = pd.Series(rng.uniform(0, 1, 20), index=idx)

    battery = _basic_battery(capacity_kwh=5.0, power_kw=12.0,
                             charge_efficiency=0.95, discharge_efficiency=0.95)
    t = self_consumption('S0001', solar, load, battery)

    reconstructed = t.sol_to_home_kwh + t.bat_to_home_kwh + t.grid_to_home_kwh
    np.testing.assert_array_almost_equal(reconstructed.values, load.values)


def test_solar_balance_invariant():
    """For every interval: solar == sol_to_home + sol_to_bat + sol_to_grid."""
    idx = _make_index(20)
    rng = np.random.default_rng(seed=42)
    solar = pd.Series(rng.uniform(0, 2, 20), index=idx)
    load = pd.Series(rng.uniform(0, 1, 20), index=idx)

    battery = _basic_battery(capacity_kwh=5.0, power_kw=12.0)
    t = self_consumption('S0001', solar, load, battery)

    reconstructed = t.sol_to_home_kwh + t.sol_to_bat_kwh + t.sol_to_grid_kwh
    np.testing.assert_array_almost_equal(reconstructed.values, solar.values)


def test_import_export_decomposition():
    """import == grid_to_home + grid_to_bat; export == sol_to_grid + bat_to_grid."""
    idx = _make_index(20)
    rng = np.random.default_rng(seed=42)
    solar = pd.Series(rng.uniform(0, 2, 20), index=idx)
    load = pd.Series(rng.uniform(0, 1, 20), index=idx)

    battery = _basic_battery(capacity_kwh=5.0, power_kw=12.0)
    t = self_consumption('S0001', solar, load, battery)

    np.testing.assert_array_almost_equal(
        t.import_kwh.values,
        (t.grid_to_home_kwh + t.grid_to_bat_kwh).values
    )
    np.testing.assert_array_almost_equal(
        t.export_kwh.values,
        (t.sol_to_grid_kwh + t.bat_to_grid_kwh).values
    )


def test_self_consumption_grid_to_bat_and_bat_to_grid_always_zero():
    """Self-consumption strategy never charges from grid or exports battery."""
    idx = _make_index(50)
    rng = np.random.default_rng(seed=42)
    solar = pd.Series(rng.uniform(0, 2, 50), index=idx)
    load = pd.Series(rng.uniform(0, 1, 50), index=idx)

    battery = _basic_battery()
    t = self_consumption('S0001', solar, load, battery)

    assert (t.grid_to_bat_kwh == 0).all()
    assert (t.bat_to_grid_kwh == 0).all()


def test_soc_never_exceeds_capacity():
    idx = _make_index(50)
    solar = pd.Series([3.0] * 50, index=idx)
    load = pd.Series([0.0] * 50, index=idx)

    battery = _basic_battery(capacity_kwh=2.0, power_kw=12.0,
                             charge_efficiency=0.95)
    t = self_consumption('S0001', solar, load, battery)
    assert (t.battery_soc_kwh <= 2.0 + 1e-9).all()


def test_soc_never_below_reserve():
    idx = _make_index(50)
    solar = pd.Series([2.0] + [0.0] * 49, index=idx)
    load = pd.Series([0.0] + [0.5] * 49, index=idx)

    battery = _basic_battery(capacity_kwh=5.0, backup_reserve_pct=30.0,
                             power_kw=12.0)
    t = self_consumption('S0001', solar, load, battery)
    assert (t.battery_soc_kwh >= 1.5 - 1e-9).all()


# ---------- Interval cadence ----------

def test_30min_cadence_doubles_per_interval_energy_cap():
    """At 30-min cadence, 1 kW * 0.5 hr = 0.5 kWh/interval cap."""
    idx = _make_index(3, freq='30min')
    solar = pd.Series([0.6, 0.6, 0.6], index=idx)
    load = pd.Series([0.0, 0.0, 0.0], index=idx)

    battery = _basic_battery(capacity_kwh=10.0, power_kw=1.0)
    t = self_consumption('S0001', solar, load, battery, interval_hours=0.5)

    # AC cap = 1.0 * 0.5 = 0.5 kWh per interval
    np.testing.assert_array_almost_equal(t.sol_to_bat_kwh, [0.5, 0.5, 0.5])
    np.testing.assert_array_almost_equal(t.sol_to_grid_kwh, [0.1, 0.1, 0.1])


# ---------- Input validation ----------

def test_mismatched_indices_raises():
    s = pd.Series([1.0], index=_make_index(1))
    l = pd.Series([1.0, 2.0], index=_make_index(2))
    with pytest.raises(ValueError, match='same index'):
        self_consumption('S0001', s, l, battery_spec=None)


def test_negative_solar_treated_as_parasitic_draw():
    """Negative solar (inverter overnight draw) must be handled correctly.

    Per the EDP data convention, gross_load already incorporates the
    inverter draw: gross_load = solar + ac_load_net. So when solar < 0,
    gross_load is the smaller "household-only" load and the inverter
    is an additional consumer. sol_to_home goes negative (the inverter
    is the consumer), residual_load increases, and grid covers it.

    Expected: net grid flow (import - export) == ac_load_net.
    """
    idx = _make_index(3)
    # Three intervals representing overnight: small negative solar,
    # household load present.
    solar = pd.Series([-0.05, -0.04, -0.06], index=idx)
    load = pd.Series([0.25, 0.20, 0.30], index=idx)
    # ac_load_net per EDP convention = load - solar = 0.30, 0.24, 0.36
    expected_ac_load_net = load - solar

    t = self_consumption('S0001', solar, load, battery_spec=None)

    # sol_to_home mirrors solar sign (negative for parasitic draw)
    np.testing.assert_array_almost_equal(t.sol_to_home_kwh.values, solar.values)
    # No surplus to export
    np.testing.assert_array_almost_equal(t.sol_to_grid_kwh.values, [0, 0, 0])
    # Grid covers actual household load PLUS the inverter draw
    np.testing.assert_array_almost_equal(
        t.grid_to_home_kwh.values, expected_ac_load_net.values
    )
    # Net grid flow equals ac_load_net
    net_grid = (t.import_kwh - t.export_kwh).values
    np.testing.assert_array_almost_equal(net_grid, expected_ac_load_net.values)


def test_negative_solar_with_battery():
    """Negative solar + battery: battery can absorb the inverter draw."""
    idx = _make_index(3)
    solar = pd.Series([-0.05, -0.05, -0.05], index=idx)
    load = pd.Series([0.0, 0.0, 0.0], index=idx)

    # Battery starts at reserve=0, plenty of capacity, 100% efficient,
    # AC cap = 5*5/60 = 0.4166... kWh/interval - well above 0.05.
    battery = _basic_battery(capacity_kwh=2.0, power_kw=5.0)
    t = self_consumption('S0001', solar, load, battery)

    # With solar=-0.05 and load=0, residual_load = 0 - (-0.05) = 0.05.
    # Battery starts at 0 (reserve=0). DC available = 0 - so b2h = 0.
    # Grid covers the inverter draw.
    np.testing.assert_array_almost_equal(t.sol_to_home_kwh.values, solar.values)
    np.testing.assert_array_almost_equal(t.grid_to_home_kwh.values, [0.05, 0.05, 0.05])
    np.testing.assert_array_almost_equal(t.bat_to_home_kwh.values, [0, 0, 0])
    # No charging (no surplus)
    np.testing.assert_array_almost_equal(t.sol_to_bat_kwh.values, [0, 0, 0])
    # Battery stays at 0
    np.testing.assert_array_almost_equal(t.battery_soc_kwh.values, [0, 0, 0])


def test_negative_load_raises():
    idx = _make_index(2)
    s = pd.Series([0.0, 0.0], index=idx)
    l = pd.Series([1.0, -0.1], index=idx)
    with pytest.raises(ValueError, match='load_kwh contains negative'):
        self_consumption('S0001', s, l, battery_spec=None)


def test_nan_in_solar_raises():
    idx = _make_index(2)
    s = pd.Series([1.0, np.nan], index=idx)
    l = pd.Series([1.0, 1.0], index=idx)
    with pytest.raises(ValueError, match='solar_kwh contains NaN'):
        self_consumption('S0001', s, l, battery_spec=None)


def test_nan_in_load_raises():
    idx = _make_index(2)
    s = pd.Series([1.0, 1.0], index=idx)
    l = pd.Series([1.0, np.nan], index=idx)
    with pytest.raises(ValueError, match='load_kwh contains NaN'):
        self_consumption('S0001', s, l, battery_spec=None)


# ---------- Output structure ----------

def test_output_index_matches_input():
    idx = _make_index(5)
    solar = pd.Series([0.1] * 5, index=idx)
    load = pd.Series([0.1] * 5, index=idx)
    t = self_consumption('S0001', solar, load, battery_spec=None)
    assert t.import_kwh.index.equals(idx)
    assert t.battery_soc_kwh.index.equals(idx)


def test_output_site_id_preserved():
    idx = _make_index(2)
    s = pd.Series([0.0, 0.0], index=idx)
    l = pd.Series([0.0, 0.0], index=idx)
    t = self_consumption('S0042', s, l, battery_spec=None)
    assert t.site_id == 'S0042'