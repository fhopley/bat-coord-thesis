"""Tests for causal event declaration and the annual cap (Part C, step C3).

The declaration rule is the causal form of what classify_vpp_events does
non-causally. The relationship between the two is asserted precisely here --
where they agree, where they must differ, and why -- because that difference
lands inside every VCR reported for the rolling VPP coordination types and
must not be mistaken for forecast error.
"""
import numpy as np
import pytest

from model.components.vpp_declaration import (
    CapState, declare, declared_mask, window_intervals,
)
from model.components.vpp_event_classifier import classify_vpp_events
import pandas as pd


THR = 1000.0


def _prices(spikes, T=40, base=50.0, spike=5000.0):
    p = np.full(T, base)
    p[list(spikes)] = spike
    return p


# ---------- window_intervals ----------

def test_window_intervals_rounds_up():
    """A fractional window must cover at least the requested duration, which
    is what classify_vpp_events does."""
    assert window_intervals(10.0, 5 / 60) == 120
    assert window_intervals(0.5, 5 / 60) == 6
    assert window_intervals(0.51, 5 / 60) == 7      # 6.12 -> 7


def test_window_intervals_rejects_nonpositive():
    with pytest.raises(ValueError, match='must be > 0'):
        window_intervals(0.0, 5 / 60)


# ---------- the rule ----------

def test_no_spike_in_reach_means_no_event():
    p = _prices([30], T=40)
    active, j = declare(p, n=40, w=5, threshold_dollars_per_mwh=THR)
    assert not active
    assert j == -1


def test_a_spike_beyond_the_event_window_does_not_declare():
    """The horizon is far longer than the event window, so the operator
    routinely sees spikes it cannot yet act on. With w=5 a spike at position
    16 is visible but out of reach."""
    p = _prices([16], T=40)
    active, _ = declare(p, n=40, w=5, threshold_dollars_per_mwh=THR)
    assert not active
    # Non-vacuity: the same spike DOES declare once the reach covers it.
    active_wide, j = declare(p, n=40, w=20, threshold_dollars_per_mwh=THR)
    assert active_wide and j == 16


def test_a_spike_at_the_edge_of_the_lookahead_declares():
    """The lookahead is inclusive of position w-1."""
    p = _prices([4], T=40)
    active, j = declare(p, n=40, w=5, threshold_dollars_per_mwh=THR)
    assert active and j == 4
    active_short, _ = declare(p, n=40, w=4, threshold_dollars_per_mwh=THR)
    assert not active_short


def test_j_star_is_the_last_spike_in_reach():
    """The operator LP runs [0, j_star]; beyond the last spike there is
    nothing to dispatch into."""
    p = _prices([1, 3, 7], T=40)
    _, j = declare(p, n=40, w=6, threshold_dollars_per_mwh=THR)
    assert j == 3          # 7 is outside the reach of 6


def test_threshold_is_strict():
    """classify_vpp_events uses > , not >=. A price exactly at the threshold
    is not a spike."""
    p = np.full(10, THR)
    active, _ = declare(p, n=10, w=5, threshold_dollars_per_mwh=THR)
    assert not active
    p[2] = THR + 0.01
    active2, j = declare(p, n=10, w=5, threshold_dollars_per_mwh=THR)
    assert active2 and j == 2


def test_lookahead_is_truncated_by_a_short_window():
    """Near the end of the period n < w, and the reach shrinks with it."""
    p = _prices([4], T=40)
    active_full, _ = declare(p, n=10, w=5, threshold_dollars_per_mwh=THR)
    assert active_full
    active_trunc, _ = declare(p, n=3, w=5, threshold_dollars_per_mwh=THR)
    assert not active_trunc


def test_only_the_first_n_positions_are_read():
    """The price slice can be longer than n when the loop shrank the window
    after the provider was queried. Reading past n would let declaration see
    prices for intervals outside the window."""
    p = _prices([8], T=40)
    active, _ = declare(p, n=4, w=20, threshold_dollars_per_mwh=THR)
    assert not active, "declaration saw a price beyond the window"


def test_non_finite_prices_are_rejected():
    p = _prices([2], T=10)
    p[1] = np.nan
    with pytest.raises(ValueError, match='non-finite'):
        declare(p, n=10, w=5, threshold_dollars_per_mwh=THR)


def test_a_short_forecast_is_rejected_rather_than_silently_shortened():
    with pytest.raises(ValueError, match='lookahead needs|needs'):
        declare(np.full(3, 50.0), n=10, w=5, threshold_dollars_per_mwh=THR)


# ---------- relationship to classify_vpp_events ----------

def _classify(prices, w, interval_hours=5 / 60):
    return classify_vpp_events(
        pd.Series(prices),
        threshold_dollars_per_mwh=THR,
        event_window_hours=w * interval_hours,
        interval_hours=interval_hours,
    )


def test_matches_the_classifier_when_spikes_are_far_apart():
    """With no two spikes closer than w, the classifier's skip never fires and
    the two rules agree exactly. This is as close to a correctness gate as
    this path allows."""
    w = 5
    prices = _prices([6, 20, 34], T=45)
    mine = declared_mask(prices, w, THR)
    theirs = _classify(prices, w)

    # Non-vacuity: an all-False comparison would pass while testing nothing.
    assert mine.sum() > 0
    np.testing.assert_array_equal(mine, theirs)


def test_diverges_from_the_classifier_on_clustered_spikes():
    """The spec's worked example. w=3, spikes at 5 and 6: the backwards scan
    marks {4,5,6} because the spike at 6 opens the window and the spike at 5
    is skipped; the forward rule marks {3,4,5,6}, because from 3 the spike at
    5 is within reach. This is the mask difference carried by every VCR on
    this path, so it is pinned rather than described."""
    w = 3
    prices = _prices([5, 6], T=12)
    mine = declared_mask(prices, w, THR)
    theirs = _classify(prices, w)

    assert set(np.flatnonzero(theirs)) == {4, 5, 6}
    assert set(np.flatnonzero(mine)) == {3, 4, 5, 6}


def test_the_forward_rule_is_always_at_least_as_permissive():
    """The forward rule is the skip-free union, so it can never mark fewer
    intervals than the classifier. If it ever did, the two would be
    incomparable and the VCR difference would have no sign."""
    rng = np.random.default_rng(5)
    for w in (2, 3, 7):
        for _ in range(20):
            prices = np.where(rng.random(60) < 0.15, 5000.0, 50.0)
            mine = declared_mask(prices, w, THR)
            theirs = _classify(prices, w)
            assert np.all(mine | theirs == mine), (
                f"classifier marked an interval the forward rule did not "
                f"(w={w}, spikes={np.flatnonzero(prices > THR)})"
            )


def test_the_two_rules_agree_on_which_intervals_contain_spikes():
    """Both mark every spike interval itself; they differ only on how far back
    the window reaches."""
    rng = np.random.default_rng(9)
    prices = np.where(rng.random(80) < 0.1, 5000.0, 50.0)
    spikes = prices > THR
    assert spikes.sum() > 0
    for w in (2, 4, 9):
        assert np.all(declared_mask(prices, w, THR)[spikes])
        assert np.all(_classify(prices, w)[spikes])


def test_spikes_closer_than_w_form_one_contiguous_block():
    w = 6
    prices = _prices([10, 13], T=30)
    mask = declared_mask(prices, w, THR)
    marked = np.flatnonzero(mask)
    assert marked.tolist() == list(range(marked[0], marked[-1] + 1)), (
        "the block is not contiguous"
    )
    assert marked[0] == 5 and marked[-1] == 13


# ---------- the annual cap ----------

def test_cap_export_only_ignores_forced_import():
    c = CapState(limit_kwh=10.0, limit_applies_to='export_only')
    c.commit(0, f_imp=3.0, f_exp=2.0)
    assert c.used_kwh == pytest.approx(2.0)
    assert c.remaining_kwh == pytest.approx(8.0)


def test_cap_import_and_export_counts_both():
    c = CapState(limit_kwh=10.0, limit_applies_to='import_and_export')
    c.commit(0, f_imp=3.0, f_exp=2.0)
    assert c.used_kwh == pytest.approx(5.0)
    assert c.remaining_kwh == pytest.approx(5.0)


def test_cap_records_the_interval_of_exhaustion_once():
    c = CapState(limit_kwh=5.0, limit_applies_to='export_only')
    c.commit(10, 0.0, 2.0)
    assert not c.exhausted and c.exhausted_at is None
    c.commit(11, 0.0, 3.0)
    assert c.exhausted and c.exhausted_at == 11
    c.commit(12, 0.0, 1.0)
    assert c.exhausted_at == 11, "exhaustion instant must not be overwritten"


def test_cap_is_terminal_and_goes_negative_rather_than_resetting():
    """remaining only shrinks. A reset would let a site re-enter events after
    its contract cap was spent."""
    c = CapState(limit_kwh=1.0, limit_applies_to='export_only')
    c.commit(0, 0.0, 3.0)
    assert c.exhausted
    assert c.remaining_kwh < 0.0
    assert c.plan_diagnostics()['cap_remaining_kwh'] == 0.0


def test_cap_rejects_negative_forced_flows():
    c = CapState(limit_kwh=10.0, limit_applies_to='import_and_export')
    with pytest.raises(ValueError, match='non-negative'):
        c.commit(0, f_imp=-1.0, f_exp=0.0)


def test_cap_rejects_an_unknown_limit_applies_to():
    with pytest.raises(ValueError, match='limit_applies_to'):
        CapState(limit_kwh=10.0, limit_applies_to='export')


def test_cap_zero_limit_is_exhausted_from_the_start():
    """A site that may not participate at all. Not a normal configuration, but
    it must not read as 'unlimited'."""
    c = CapState(limit_kwh=0.0, limit_applies_to='export_only')
    assert c.exhausted


def test_cap_diagnostics_do_not_mutate():
    c = CapState(limit_kwh=10.0, limit_applies_to='export_only')
    c.commit(0, 0.0, 2.0)
    assert c.plan_diagnostics() == c.plan_diagnostics()
    assert c.remaining_kwh == pytest.approx(8.0)


def test_cap_exhaustion_tolerates_solver_residue():
    """remaining_kwh is a difference of accumulated LP outputs, so a cap that
    is exactly spent can land just above zero. Observed at +8.3e-17 kWh after
    a solver change, which left the site declaring events on a cap it had
    used up. An exact `<= 0.0` test is not safe here."""
    # 1e-17 below 1.0 is not representable -- it rounds away and leaves an
    # exact zero, which the old test accepted for the wrong reason. 1e-13 is
    # above the ULP at 1.0 and below the tolerance.
    c = CapState(limit_kwh=1.0, limit_applies_to='export_only')
    c.commit(0, 0.0, 1.0 - 1e-13)
    assert c.remaining_kwh > 0.0, "fixture did not leave a positive residue"
    assert c.exhausted, "a numerically spent cap was not treated as exhausted"
    assert c.exhausted_at == 0


def test_cap_with_real_headroom_is_not_exhausted():
    """Non-vacuity partner: the tolerance must not swallow a real remainder."""
    c = CapState(limit_kwh=1.0, limit_applies_to='export_only')
    c.commit(0, 0.0, 0.9)
    assert not c.exhausted
    assert c.exhausted_at is None
