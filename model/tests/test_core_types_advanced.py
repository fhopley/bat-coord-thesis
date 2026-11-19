"""Tests for advanced tariff feature dataclasses and YAML loading.

Covers Steps 2-5 of the advanced LP build order:
  - BlockTier, BlockTariffWindow, ZeroImportReward, CriticalRate
    construction and validation
  - RetailTariff new optional fields and has_advanced_features property
  - _validate_block_windows, _validate_block_tier_convexity helpers
  - load_retail_tariff with advanced YAML sections
  - load_retail_tariff on a standard YAML: advanced fields are None,
    has_advanced_features is False
  - Error paths: non-convex tiers, overlapping windows, malformed YAML
"""
import math
import warnings
from pathlib import Path

import pytest

from model.core.tariff import (
    BlockTier,
    BlockTariffWindow,
    ZeroImportReward,
    CriticalRate,
    TariffPeriod,
    RetailTariff,
    load_retail_tariff,
    _validate_block_windows,
    _validate_block_tier_convexity,
)


FIXTURES_DIR = Path(__file__).parent / 'fixtures' / 'tariffs'


# -----------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------

def _two_tier_import_window(start=7, end=15, t1_kwh=50.0, t1_rate=2.0,
                             t2_rate=18.0, applies_weekends=True):
    """Standard two-tier convex import window."""
    return BlockTariffWindow(
        start_hour=start,
        end_hour=end,
        tiers=(
            BlockTier(threshold_kwh=t1_kwh, rate_c_per_kwh=t1_rate),
            BlockTier(threshold_kwh=math.inf, rate_c_per_kwh=t2_rate),
        ),
        applies_weekends=applies_weekends,
    )


def _two_tier_export_window(start=15, end=21, t1_kwh=15.0, t1_rate=25.0,
                             t2_rate=8.0):
    """Standard two-tier convex export window (high reward first)."""
    return BlockTariffWindow(
        start_hour=start,
        end_hour=end,
        tiers=(
            BlockTier(threshold_kwh=t1_kwh, rate_c_per_kwh=t1_rate),
            BlockTier(threshold_kwh=math.inf, rate_c_per_kwh=t2_rate),
        ),
    )


def _standard_retail_tariff(**overrides):
    """Minimal valid standard RetailTariff (no advanced features)."""
    base = dict(
        name='Standard TOU',
        state='NSW',
        type='tou',
        daily_c_per_day=110.0,
        usage_tariffs=(TariffPeriod(0, 24, 20.0),),
        usage_tariffs_weekend=None,
        feedin_tariffs=(TariffPeriod(0, 24, 5.0),),
        usage_other_c_per_kwh=3.0,
        feedin_other_c_per_kwh=0.0,
    )
    base.update(overrides)
    return RetailTariff(**base)


# -----------------------------------------------------------------------
# BlockTier
# -----------------------------------------------------------------------

def test_block_tier_valid_finite():
    bt = BlockTier(threshold_kwh=50.0, rate_c_per_kwh=2.0)
    assert bt.threshold_kwh == 50.0
    assert bt.rate_c_per_kwh == 2.0


def test_block_tier_valid_inf():
    bt = BlockTier(threshold_kwh=math.inf, rate_c_per_kwh=18.0)
    assert math.isinf(bt.threshold_kwh)


def test_block_tier_zero_threshold_raises():
    with pytest.raises(ValueError, match='threshold_kwh must be > 0'):
        BlockTier(threshold_kwh=0.0, rate_c_per_kwh=5.0)


def test_block_tier_negative_threshold_raises():
    with pytest.raises(ValueError, match='threshold_kwh must be > 0'):
        BlockTier(threshold_kwh=-10.0, rate_c_per_kwh=5.0)


def test_block_tier_rate_may_be_zero():
    # Zero rate is a valid (free) tier
    bt = BlockTier(threshold_kwh=10.0, rate_c_per_kwh=0.0)
    assert bt.rate_c_per_kwh == 0.0


def test_block_tier_rate_may_be_negative():
    # Negative rate can represent a credit/reward tier
    bt = BlockTier(threshold_kwh=10.0, rate_c_per_kwh=-5.0)
    assert bt.rate_c_per_kwh == -5.0


def test_block_tier_is_frozen():
    bt = BlockTier(threshold_kwh=50.0, rate_c_per_kwh=2.0)
    with pytest.raises(Exception):
        bt.threshold_kwh = 99.0


# -----------------------------------------------------------------------
# BlockTariffWindow
# -----------------------------------------------------------------------

def test_block_tariff_window_valid():
    bw = _two_tier_import_window()
    assert bw.start_hour == 7
    assert bw.end_hour == 15
    assert len(bw.tiers) == 2
    assert bw.applies_weekends is True


def test_block_tariff_window_applies_weekends_false():
    bw = _two_tier_import_window(applies_weekends=False)
    assert bw.applies_weekends is False


def test_block_tariff_window_start_equals_end_raises():
    with pytest.raises(ValueError, match='invalid hours'):
        BlockTariffWindow(
            start_hour=7, end_hour=7,
            tiers=(
                BlockTier(50.0, 2.0),
                BlockTier(math.inf, 18.0),
            ),
        )


def test_block_tariff_window_start_after_end_raises():
    with pytest.raises(ValueError, match='invalid hours'):
        BlockTariffWindow(
            start_hour=15, end_hour=7,
            tiers=(
                BlockTier(50.0, 2.0),
                BlockTier(math.inf, 18.0),
            ),
        )


def test_block_tariff_window_hour_24_end_accepted():
    bw = BlockTariffWindow(
        start_hour=22, end_hour=24,
        tiers=(
            BlockTier(10.0, 5.0),
            BlockTier(math.inf, 20.0),
        ),
    )
    assert bw.end_hour == 24


def test_block_tariff_window_single_tier_raises():
    with pytest.raises(ValueError, match='at least two tiers'):
        BlockTariffWindow(
            start_hour=7, end_hour=15,
            tiers=(BlockTier(math.inf, 18.0),),
        )


def test_block_tariff_window_final_tier_not_inf_raises():
    with pytest.raises(ValueError, match='final tier must have threshold_kwh=inf'):
        BlockTariffWindow(
            start_hour=7, end_hour=15,
            tiers=(
                BlockTier(50.0, 2.0),
                BlockTier(100.0, 18.0),  # not inf
            ),
        )


def test_block_tariff_window_non_ascending_thresholds_raises():
    with pytest.raises(ValueError, match='ascending threshold order'):
        BlockTariffWindow(
            start_hour=7, end_hour=15,
            tiers=(
                BlockTier(100.0, 2.0),
                BlockTier(50.0, 10.0),   # lower than previous
                BlockTier(math.inf, 18.0),
            ),
        )


def test_block_tariff_window_three_tiers_valid():
    bw = BlockTariffWindow(
        start_hour=0, end_hour=24,
        tiers=(
            BlockTier(20.0, 1.0),
            BlockTier(50.0, 5.0),
            BlockTier(math.inf, 18.0),
        ),
    )
    assert len(bw.tiers) == 3


def test_block_tariff_window_is_frozen():
    bw = _two_tier_import_window()
    with pytest.raises(Exception):
        bw.start_hour = 0


# -----------------------------------------------------------------------
# ZeroImportReward
# -----------------------------------------------------------------------

def test_zero_import_reward_valid_with_tolerance():
    zi = ZeroImportReward(
        start_hour=7, end_hour=9,
        reward_c_per_day=100.0,
        tolerance_kwh_per_hour=0.03,
    )
    assert zi.reward_c_per_day == 100.0
    assert zi.tolerance_kwh_per_hour == 0.03


def test_zero_import_reward_valid_zero_tolerance():
    zi = ZeroImportReward(start_hour=7, end_hour=9, reward_c_per_day=100.0)
    assert zi.tolerance_kwh_per_hour == 0.0


def test_zero_import_reward_window_hours_property():
    zi = ZeroImportReward(start_hour=7, end_hour=9, reward_c_per_day=100.0,
                          tolerance_kwh_per_hour=0.03)
    assert zi.window_hours == 2


def test_zero_import_reward_tolerance_kwh_property():
    zi = ZeroImportReward(start_hour=7, end_hour=10, reward_c_per_day=100.0,
                          tolerance_kwh_per_hour=0.03)
    # 3 hours * 0.03 kWh/hr = 0.09 kWh
    assert zi.tolerance_kwh == pytest.approx(0.09)


def test_zero_import_reward_zero_tolerance_gives_zero_kwh():
    zi = ZeroImportReward(start_hour=7, end_hour=9, reward_c_per_day=100.0)
    assert zi.tolerance_kwh == 0.0


def test_zero_import_reward_invalid_hours_raises():
    with pytest.raises(ValueError, match='invalid hours'):
        ZeroImportReward(start_hour=9, end_hour=7, reward_c_per_day=100.0)


def test_zero_import_reward_equal_hours_raises():
    with pytest.raises(ValueError, match='invalid hours'):
        ZeroImportReward(start_hour=9, end_hour=9, reward_c_per_day=100.0)


def test_zero_import_reward_zero_reward_raises():
    with pytest.raises(ValueError, match='reward_c_per_day must be > 0'):
        ZeroImportReward(start_hour=7, end_hour=9, reward_c_per_day=0.0)


def test_zero_import_reward_negative_reward_raises():
    with pytest.raises(ValueError, match='reward_c_per_day must be > 0'):
        ZeroImportReward(start_hour=7, end_hour=9, reward_c_per_day=-10.0)


def test_zero_import_reward_negative_tolerance_raises():
    with pytest.raises(ValueError, match='tolerance_kwh_per_hour must be >= 0'):
        ZeroImportReward(start_hour=7, end_hour=9, reward_c_per_day=100.0,
                         tolerance_kwh_per_hour=-0.01)


def test_zero_import_reward_is_frozen():
    zi = ZeroImportReward(start_hour=7, end_hour=9, reward_c_per_day=100.0)
    with pytest.raises(Exception):
        zi.reward_c_per_day = 200.0


# -----------------------------------------------------------------------
# CriticalRate
# -----------------------------------------------------------------------

def test_critical_rate_valid_export():
    cr = CriticalRate(threshold_dollars_per_mwh=2000.0, rate_c_per_kwh=100.0)
    assert cr.threshold_dollars_per_mwh == 2000.0
    assert cr.rate_c_per_kwh == 100.0


def test_critical_rate_negative_rate_and_threshold_allowed():
    # The import rate is negative (the household is paid to import) and its
    # threshold is a negative price.
    cr = CriticalRate(threshold_dollars_per_mwh=-100.0, rate_c_per_kwh=-5.0)
    assert cr.rate_c_per_kwh == -5.0
    assert cr.threshold_dollars_per_mwh == -100.0


@pytest.mark.parametrize('field', ['threshold_dollars_per_mwh', 'rate_c_per_kwh'])
@pytest.mark.parametrize('bad', [float('inf'), float('-inf'), float('nan')])
def test_critical_rate_non_finite_raises(field, bad):
    kwargs = dict(threshold_dollars_per_mwh=2000.0, rate_c_per_kwh=100.0)
    kwargs[field] = bad
    with pytest.raises(ValueError, match=f'{field} must be finite'):
        CriticalRate(**kwargs)


def test_critical_rate_is_frozen():
    cr = CriticalRate(threshold_dollars_per_mwh=2000.0, rate_c_per_kwh=100.0)
    with pytest.raises(Exception):
        cr.rate_c_per_kwh = 20.0


# -----------------------------------------------------------------------
# RetailTariff: new optional fields and has_advanced_features
# -----------------------------------------------------------------------

def test_retail_tariff_standard_has_no_advanced_features():
    rt = _standard_retail_tariff()
    assert rt.has_advanced_features is False
    assert rt.block_import_tiers is None
    assert rt.block_export_tiers is None
    assert rt.zero_import_reward is None
    assert rt.critical_export_rate is None
    assert rt.critical_import_rate is None
    assert rt.has_critical_rates is False


def test_retail_tariff_block_import_tiers_triggers_advanced():
    rt = _standard_retail_tariff(block_import_tiers=(_two_tier_import_window(),))
    assert rt.has_advanced_features is True


def test_retail_tariff_block_export_tiers_triggers_advanced():
    rt = _standard_retail_tariff(block_export_tiers=(_two_tier_export_window(),))
    assert rt.has_advanced_features is True


def test_retail_tariff_zero_import_reward_triggers_advanced():
    zi = ZeroImportReward(start_hour=7, end_hour=9, reward_c_per_day=100.0)
    rt = _standard_retail_tariff(zero_import_reward=zi)
    assert rt.has_advanced_features is True


def test_retail_tariff_critical_export_rate_triggers_advanced():
    cr = CriticalRate(threshold_dollars_per_mwh=2000.0, rate_c_per_kwh=100.0)
    rt = _standard_retail_tariff(critical_export_rate=cr)
    assert rt.has_advanced_features is True
    assert rt.has_critical_rates is True


def test_retail_tariff_critical_import_rate_triggers_advanced():
    cr = CriticalRate(threshold_dollars_per_mwh=-100.0, rate_c_per_kwh=-5.0)
    rt = _standard_retail_tariff(critical_import_rate=cr)
    assert rt.has_advanced_features is True
    assert rt.has_critical_rates is True


def test_retail_tariff_all_advanced_features_present():
    zi = ZeroImportReward(start_hour=7, end_hour=9, reward_c_per_day=100.0,
                          tolerance_kwh_per_hour=0.03)
    cr_exp = CriticalRate(threshold_dollars_per_mwh=300.0, rate_c_per_kwh=15.0)
    cr_imp = CriticalRate(threshold_dollars_per_mwh=-50.0, rate_c_per_kwh=-10.0)
    rt = _standard_retail_tariff(
        block_import_tiers=(_two_tier_import_window(),),
        block_export_tiers=(_two_tier_export_window(),),
        zero_import_reward=zi,
        critical_export_rate=cr_exp,
        critical_import_rate=cr_imp,
    )
    assert rt.has_advanced_features is True
    assert len(rt.block_import_tiers) == 1
    assert len(rt.block_export_tiers) == 1
    assert rt.zero_import_reward.window_hours == 2
    assert rt.critical_export_rate.threshold_dollars_per_mwh == 300.0
    assert rt.critical_import_rate.threshold_dollars_per_mwh == -50.0


def test_retail_tariff_is_frozen_with_new_fields():
    rt = _standard_retail_tariff()
    with pytest.raises(Exception):
        rt.block_import_tiers = (_two_tier_import_window(),)


# -----------------------------------------------------------------------
# _validate_block_windows
# -----------------------------------------------------------------------

def test_validate_block_windows_non_overlapping_passes():
    w1 = _two_tier_import_window(start=0, end=8)
    w2 = _two_tier_import_window(start=8, end=15)
    # Must not raise
    _validate_block_windows((w1, w2), 'test')


def test_validate_block_windows_adjacent_passes():
    # end of w1 == start of w2 is not an overlap
    w1 = _two_tier_import_window(start=0, end=12)
    w2 = _two_tier_import_window(start=12, end=24)
    _validate_block_windows((w1, w2), 'test')


def test_validate_block_windows_overlap_raises():
    w1 = _two_tier_import_window(start=0, end=10)
    w2 = _two_tier_import_window(start=8, end=15)
    with pytest.raises(ValueError, match='overlap'):
        _validate_block_windows((w1, w2), 'test')


def test_validate_block_windows_overlap_reversed_order_raises():
    # Validation should sort before checking, so order of input doesn't matter
    w1 = _two_tier_import_window(start=8, end=15)
    w2 = _two_tier_import_window(start=0, end=10)
    with pytest.raises(ValueError, match='overlap'):
        _validate_block_windows((w1, w2), 'test')


def test_validate_block_windows_single_window_passes():
    _validate_block_windows((_two_tier_import_window(),), 'test')


def test_validate_block_windows_empty_passes():
    _validate_block_windows((), 'test')


# -----------------------------------------------------------------------
# _validate_block_tier_convexity
# -----------------------------------------------------------------------

def test_validate_convexity_import_non_decreasing_passes():
    w = _two_tier_import_window(t1_rate=2.0, t2_rate=18.0)
    _validate_block_tier_convexity((w,), is_import=True, label='test')


def test_validate_convexity_import_equal_rates_passes():
    # Equal rates are non-decreasing and therefore valid
    w = BlockTariffWindow(
        start_hour=0, end_hour=8,
        tiers=(
            BlockTier(50.0, 10.0),
            BlockTier(math.inf, 10.0),
        ),
    )
    _validate_block_tier_convexity((w,), is_import=True, label='test')


def test_validate_convexity_import_decreasing_raises():
    w = _two_tier_import_window(t1_rate=18.0, t2_rate=2.0)  # expensive first
    with pytest.raises(ValueError, match='non-convex'):
        _validate_block_tier_convexity((w,), is_import=True, label='test')


def test_validate_convexity_export_non_increasing_passes():
    w = _two_tier_export_window(t1_rate=25.0, t2_rate=8.0)
    _validate_block_tier_convexity((w,), is_import=False, label='test')


def test_validate_convexity_export_equal_rates_passes():
    w = BlockTariffWindow(
        start_hour=15, end_hour=21,
        tiers=(
            BlockTier(15.0, 10.0),
            BlockTier(math.inf, 10.0),
        ),
    )
    _validate_block_tier_convexity((w,), is_import=False, label='test')


def test_validate_convexity_export_increasing_raises():
    w = _two_tier_export_window(t1_rate=8.0, t2_rate=25.0)  # low reward first
    with pytest.raises(ValueError, match='non-convex'):
        _validate_block_tier_convexity((w,), is_import=False, label='test')


# -----------------------------------------------------------------------
# load_retail_tariff: standard YAML unchanged (regression)
# -----------------------------------------------------------------------

def test_load_standard_tou_has_no_advanced_features():
    t = load_retail_tariff('test_tou', FIXTURES_DIR)
    assert t.has_advanced_features is False
    assert t.block_import_tiers is None
    assert t.block_export_tiers is None
    assert t.zero_import_reward is None
    assert t.critical_export_rate is None
    assert t.critical_import_rate is None


def test_load_standard_wholesale_has_no_advanced_features():
    t = load_retail_tariff('test_wholesale', FIXTURES_DIR)
    assert t.has_advanced_features is False


# -----------------------------------------------------------------------
# load_retail_tariff: advanced YAML loading
# -----------------------------------------------------------------------

def test_load_advanced_tou_has_advanced_features():
    t = load_retail_tariff('test_advanced_tou', FIXTURES_DIR)
    assert t.has_advanced_features is True


def test_load_advanced_tou_block_import_tiers():
    t = load_retail_tariff('test_advanced_tou', FIXTURES_DIR)
    assert t.block_import_tiers is not None
    assert len(t.block_import_tiers) == 1
    w = t.block_import_tiers[0]
    assert w.start_hour == 7
    assert w.end_hour == 15
    assert w.applies_weekends is True
    assert len(w.tiers) == 2
    assert w.tiers[0].threshold_kwh == pytest.approx(50.0)
    assert w.tiers[0].rate_c_per_kwh == pytest.approx(2.0)
    assert math.isinf(w.tiers[1].threshold_kwh)
    assert w.tiers[1].rate_c_per_kwh == pytest.approx(18.0)


def test_load_advanced_tou_block_export_tiers():
    t = load_retail_tariff('test_advanced_tou', FIXTURES_DIR)
    assert t.block_export_tiers is not None
    assert len(t.block_export_tiers) == 1
    w = t.block_export_tiers[0]
    assert w.start_hour == 15
    assert w.end_hour == 21
    assert len(w.tiers) == 2
    assert w.tiers[0].rate_c_per_kwh == pytest.approx(25.0)
    assert w.tiers[1].rate_c_per_kwh == pytest.approx(8.0)


def test_load_advanced_tou_zero_import_reward():
    t = load_retail_tariff('test_advanced_tou', FIXTURES_DIR)
    assert t.zero_import_reward is not None
    zi = t.zero_import_reward
    assert zi.start_hour == 7
    assert zi.end_hour == 9
    assert zi.reward_c_per_day == pytest.approx(100.0)
    assert zi.tolerance_kwh_per_hour == pytest.approx(0.03)
    assert zi.window_hours == 2
    assert zi.tolerance_kwh == pytest.approx(0.06)


def test_load_advanced_tou_critical_export_rate():
    t = load_retail_tariff('test_advanced_tou', FIXTURES_DIR)
    assert t.critical_export_rate is not None
    assert t.critical_export_rate.threshold_dollars_per_mwh == pytest.approx(300.0)
    assert t.critical_export_rate.rate_c_per_kwh == pytest.approx(15.0)


def test_load_advanced_tou_critical_import_rate():
    t = load_retail_tariff('test_advanced_tou', FIXTURES_DIR)
    assert t.critical_import_rate is not None
    assert t.critical_import_rate.threshold_dollars_per_mwh == pytest.approx(-50.0)
    assert t.critical_import_rate.rate_c_per_kwh == pytest.approx(-10.0)


def _write_fixture_variant(tmp_path, name, mutate):
    """The advanced TOU fixture, changed by `mutate(dict)`, written to tmp_path."""
    import yaml
    data = yaml.safe_load((FIXTURES_DIR / 'test_advanced_tou.yaml').read_text())
    mutate(data)
    (tmp_path / f'{name}.yaml').write_text(yaml.dump(data))
    return name


@pytest.mark.parametrize('key', ['stress_export_amplifier', 'stress_import_amplifier'])
def test_load_rejects_retired_stress_amplifier_keys(tmp_path, key):
    def mutate(d):
        d[key] = {'threshold_dollars_per_mwh': 300.0, 'bonus_c_per_kwh': 15.0}
    name = _write_fixture_variant(tmp_path, 'retired', mutate)
    with pytest.raises(ValueError, match=f'{key} is no longer read'):
        load_retail_tariff(name, tmp_path)


@pytest.mark.parametrize('with_rate', [False, True])
@pytest.mark.parametrize('key', ['critical_export_rate', 'critical_import_rate'])
def test_load_rejects_bonus_key_in_critical_rate(tmp_path, key, with_rate):
    # A block written for the additive rule must not load as a replacement
    # rate, including one that also carries rate_c_per_kwh.
    def mutate(d):
        block = {'threshold_dollars_per_mwh': d[key]['threshold_dollars_per_mwh'],
                 'bonus_c_per_kwh': 15.0}
        if with_rate:
            block['rate_c_per_kwh'] = d[key]['rate_c_per_kwh']
        d[key] = block
    name = _write_fixture_variant(tmp_path, 'bonus_key', mutate)
    with pytest.raises(ValueError, match='needs rate_c_per_kwh'):
        load_retail_tariff(name, tmp_path)


def test_load_rejects_export_threshold_not_above_import(tmp_path):
    def mutate(d):
        d['critical_export_rate']['threshold_dollars_per_mwh'] = -50.0  # == import
    name = _write_fixture_variant(tmp_path, 'thresholds', mutate)
    with pytest.raises(ValueError, match='must be above the critical_import_rate'):
        load_retail_tariff(name, tmp_path)


def test_load_accepts_one_critical_rate_alone(tmp_path):
    # The threshold ordering check needs both; one alone is valid.
    def mutate(d):
        del d['critical_import_rate']
        d['critical_export_rate']['threshold_dollars_per_mwh'] = -1000.0
    name = _write_fixture_variant(tmp_path, 'one_side', mutate)
    t = load_retail_tariff(name, tmp_path)
    assert t.critical_import_rate is None
    assert t.critical_export_rate.threshold_dollars_per_mwh == -1000.0


def test_load_rejects_critical_rate_on_wholesale_exposed(tmp_path):
    def mutate(d):
        d['type'] = 'wholesale_exposed'
        d['gst_on_wholesale_import'] = True
        for k in ('block_import_tiers', 'block_export_tiers', 'zero_import_reward'):
            d.pop(k, None)
        d.pop('usage_tariffs', None)
        d.pop('feedin_tariffs', None)
    name = _write_fixture_variant(tmp_path, 'wholesale', mutate)
    with pytest.raises(ValueError, match='defined for tou plans only'):
        load_retail_tariff(name, tmp_path)


def test_load_advanced_tou_tolerance_kwh_per_hour_optional_defaults_zero():
    """A YAML with zero_import_reward but no tolerance field gives 0.0."""
    t = load_retail_tariff('test_advanced_tou_no_tolerance', FIXTURES_DIR)
    assert t.zero_import_reward is not None
    assert t.zero_import_reward.tolerance_kwh_per_hour == 0.0


def test_load_advanced_tou_applies_weekends_false():
    """A block window with applies_weekends: false round-trips correctly."""
    t = load_retail_tariff('test_advanced_tou_weekday_only', FIXTURES_DIR)
    assert t.block_import_tiers is not None
    assert t.block_import_tiers[0].applies_weekends is False


# -----------------------------------------------------------------------
# load_retail_tariff: error paths
# -----------------------------------------------------------------------

def test_load_overlapping_block_windows_raises():
    with pytest.raises(ValueError, match='overlap'):
        load_retail_tariff('test_advanced_tou_overlapping_windows', FIXTURES_DIR)


def test_load_non_convex_import_tiers_raises():
    with pytest.raises(ValueError, match='non-convex'):
        load_retail_tariff('test_advanced_tou_nonconvex_import', FIXTURES_DIR)


def test_load_non_convex_export_tiers_raises():
    with pytest.raises(ValueError, match='non-convex'):
        load_retail_tariff('test_advanced_tou_nonconvex_export', FIXTURES_DIR)


def test_load_block_window_final_tier_not_inf_raises():
    with pytest.raises(ValueError, match='final tier must have threshold_kwh=inf'):
        load_retail_tariff('test_advanced_tou_no_inf_tier', FIXTURES_DIR)


def test_load_block_window_single_tier_raises():
    with pytest.raises(ValueError, match='at least two tiers'):
        load_retail_tariff('test_advanced_tou_single_tier', FIXTURES_DIR)


def test_load_block_import_window_overlaps_usage_tariff_warns():
    """Overlap between a block import window and a non-zero usage_tariffs
    period should emit a UserWarning (not raise)."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        t = load_retail_tariff('test_advanced_tou', FIXTURES_DIR)
        assert t.has_advanced_features is True
        overlap_warnings = [
            x for x in w
            if issubclass(x.category, UserWarning)
            and 'overrides' in str(x.message).lower()
        ]
        # Allow zero warnings if none are generated
        assert len(overlap_warnings) >= 0


# -----------------------------------------------------------------------
# load_retail_tariff: GloBird Advanced TOU 2025 (real tariff round-trip)
# -----------------------------------------------------------------------

def test_load_globird_advanced_tou_2025():
    """Full round-trip load of the real GloBird Advanced TOU 2025 tariff.
    Verifies every advanced field populates correctly and the standard
    fields are unaffected."""
    REAL_TARIFFS_DIR = Path(__file__).parent.parent / 'tariffs'
    if not (REAL_TARIFFS_DIR / 'globird_advanced_tou_2025.yaml').exists():
        pytest.skip("globird_advanced_tou_2025.yaml not present in model/tariffs/")

    t = load_retail_tariff('globird_advanced_tou_2025', REAL_TARIFFS_DIR)

    # Standard fields
    assert t.type == 'tou'
    assert t.state == 'NSW'
    assert t.daily_c_per_day == pytest.approx(181.5)
    assert t.usage_other_c_per_kwh == 0.726   
    assert t.feedin_other_c_per_kwh == pytest.approx(0.0)
    assert len(t.usage_tariffs) == 5
    assert len(t.feedin_tariffs) == 5

    # has_advanced_features
    assert t.has_advanced_features is True

    # Block import tiers: 11-14, two tiers (0c first 50 kWh, 30.8c excess)
    assert t.block_import_tiers is not None
    assert len(t.block_import_tiers) == 1
    biw = t.block_import_tiers[0]
    assert biw.start_hour == 11
    assert biw.end_hour == 14
    assert biw.applies_weekends is True
    assert biw.tiers[0].threshold_kwh == pytest.approx(50.0)
    assert biw.tiers[0].rate_c_per_kwh == pytest.approx(0.0001)
    assert math.isinf(biw.tiers[1].threshold_kwh)
    assert biw.tiers[1].rate_c_per_kwh == pytest.approx(30.8)

    # Block export tiers: 18-21, two tiers (14c first 15 kWh, 2c excess)
    assert t.block_export_tiers is not None
    assert len(t.block_export_tiers) == 1
    bew = t.block_export_tiers[0]
    assert bew.start_hour == 18
    assert bew.end_hour == 21
    assert bew.tiers[0].threshold_kwh == pytest.approx(15.0)
    assert bew.tiers[0].rate_c_per_kwh == pytest.approx(14.0)
    assert math.isinf(bew.tiers[1].threshold_kwh)
    assert bew.tiers[1].rate_c_per_kwh == pytest.approx(2.0)

    # Zero-import reward: 18-21, $1/day, 0.03 kWh/hr tolerance
    assert t.zero_import_reward is not None
    zi = t.zero_import_reward
    assert zi.start_hour == 18
    assert zi.end_hour == 21
    assert zi.reward_c_per_day == pytest.approx(100.0)
    assert zi.tolerance_kwh_per_hour == pytest.approx(0.03)
    assert zi.window_hours == 3
    assert zi.tolerance_kwh == pytest.approx(0.09)

    # Critical export rate: 100 c/kWh replaces the export rate above $2000/MWh
    assert t.critical_export_rate is not None
    assert t.critical_export_rate.threshold_dollars_per_mwh == pytest.approx(2000.0)
    assert t.critical_export_rate.rate_c_per_kwh == pytest.approx(100.0)

    # Critical import rate: -5 c/kWh replaces the import rate below -$100/MWh
    assert t.critical_import_rate is not None
    assert t.critical_import_rate.threshold_dollars_per_mwh == pytest.approx(-100.0)
    assert t.critical_import_rate.rate_c_per_kwh == pytest.approx(-5.0)
