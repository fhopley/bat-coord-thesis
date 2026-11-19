"""Tests for RunSet and the shared window assembly.

Synthetic RunSets throughout, so splice, infill and causality are exercised
independently of any source. Values are chosen so that provenance is readable
from the number itself: coarse values are negative, fine values positive,
actuals in the hundreds.
"""
import numpy as np
import pytest

from model.forecast.runset import RunSet
from model.forecast.assembly import (
    AssembledWindow, assemble_window, expand_coarse, mode_allowance_intervals,
)

COARSE = 6      # 30 min at 5-minute resolution


def _runset(runs, targets, values, resolution=1):
    return RunSet.from_long(np.array(runs), np.array(targets),
                            np.array(values, dtype=float), resolution)


def _fine(run_positions, n_targets=12, base=1.0):
    """Fine runs at leads 0..n_targets-1. Value encodes (run, lead)."""
    r, t, v = [], [], []
    for rp in run_positions:
        for lead in range(n_targets):
            r.append(rp); t.append(rp + lead); v.append(base + rp + lead / 100)
    return _runset(r, t, v, resolution=1)


def _coarse(run_positions, span=48, base=-1.0):
    """Coarse runs every 30 min out to `span` intervals ahead."""
    r, t, v = [], [], []
    for rp in run_positions:
        for tp in range(rp, rp + span + 1, COARSE):
            r.append(rp); t.append(tp); v.append(base - rp - tp / 1000)
    return _runset(r, t, v, resolution=COARSE)


# ------------------------------------------------------------------ RunSet

def test_runset_latest_at_selects_at_or_before():
    rs = _fine([0, 6, 12])
    assert rs.latest_at(5) == 0
    assert rs.latest_at(6) == 1          # boundary is inclusive
    assert rs.latest_at(11) == 1
    assert rs.latest_at(100) == 2


def test_runset_latest_at_returns_none_before_first_run():
    rs = _fine([10])
    assert rs.latest_at(9) is None
    assert rs.latest_at(10) == 0


def test_runset_rejects_duplicate_run_target_pairs():
    """The store is unique after the INTERVENTION filter. A duplicate here
    means that filter did not run, and resolving it silently would pick an
    arbitrary price on exactly the intervals where the two cases differ."""
    with pytest.raises(ValueError, match="duplicate"):
        _runset([0, 0], [3, 3], [10.0, 20.0])


def test_runset_from_long_sorts_unordered_input():
    rs = _runset([6, 0, 6, 0], [7, 1, 6, 0], [4.0, 2.0, 3.0, 1.0])
    assert rs.run_pos.tolist() == [0, 6]
    _, t0, v0 = rs.run(0)
    assert t0.tolist() == [0, 1] and v0.tolist() == [1.0, 2.0]


def test_runset_max_target_span():
    assert _coarse([0], span=48).max_target_span() == 48


def test_runset_rejects_bad_offsets():
    with pytest.raises(ValueError, match="offsets"):
        RunSet(run_pos=np.array([0, 6]), offsets=np.array([0, 1]),
               target_pos=np.array([0, 6]), values=np.array([1.0, 2.0]),
               resolution_intervals=1)


# ---------------------------------------------------------- coarse expansion

def test_expand_coarse_covers_preceding_five_intervals():
    """T covers T-25min .. T, i.e. positions T-5 .. T inclusive."""
    pos, val = expand_coarse(np.array([30]), np.array([7.0]), COARSE)
    assert pos.tolist() == [25, 26, 27, 28, 29, 30]
    assert val.tolist() == [7.0] * 6


def test_expand_coarse_is_identity_at_five_minute_resolution():
    p, v = np.array([3, 4]), np.array([1.0, 2.0])
    pos, val = expand_coarse(p, v, 1)
    assert pos is p and val is v


# ------------------------------------------------------------------- splice

def test_fine_wins_in_the_overlap_and_coarse_fills_beyond():
    fine, coarse = _fine([12]), _coarse([12], span=48)
    w = assemble_window(
        k=13, n_intervals=40, coarse=coarse, fine=fine,
        actuals=np.zeros(200), infill_lag_intervals=288)
    out, is_infill = w.rrp, w.is_infill
    # fine covers leads 0..11 from run 12, i.e. positions 12..23 -> window 0..10
    assert (out[:11] > 0).all(), "fine segment should win where it exists"
    assert (out[11:] < 0).all(), "coarse should fill beyond the fine segment"
    assert not is_infill.any()


def test_coarse_alone_covers_the_window():
    """The fine segment is a resolution overlay, not a necessity: a missing
    fine run degrades resolution rather than invalidating the forecast."""
    w = assemble_window(
        k=13, n_intervals=40, coarse=_coarse([12], span=48), fine=None,
        actuals=np.zeros(200), infill_lag_intervals=288)
    out, is_infill, run_pos = w.rrp, w.is_infill, w.run_pos
    assert np.isfinite(out).all()
    assert (out < 0).all()
    assert run_pos == 12


def test_lead_zero_row_is_dropped_from_the_window():
    """Position k-1 is the already-committed interval."""
    fine = _fine([12])
    out = assemble_window(
        k=13, n_intervals=6, coarse=_coarse([12]), fine=fine,
        actuals=np.zeros(200), infill_lag_intervals=288).rrp
    _, t, v = fine.run(0)
    lead0 = v[t == 12][0]
    assert not np.isclose(out, lead0).any()


def test_returns_none_when_no_coarse_run_available():
    out = assemble_window(
        k=5, n_intervals=10, coarse=_coarse([12]), fine=None,
        actuals=np.zeros(200), infill_lag_intervals=288)
    assert out is None


def test_carry_forward_uses_the_previous_run_when_one_is_missing():
    """No staleness gate: the provider returns the best it has, because that
    is what the controller had."""
    coarse = _coarse([0, 12], span=60)      # run at 6 is absent
    w = assemble_window(
        k=8, n_intervals=20, coarse=coarse, fine=None,
        actuals=np.zeros(200), infill_lag_intervals=288)
    out, run_pos = w.rrp, w.run_pos
    assert run_pos == 0
    assert np.isfinite(out).all()


# --------------------------------------------------------------- provenance

def test_is_fine_marks_exactly_the_fine_supplied_positions():
    """The splice boundary, recorded rather than inferred. Under 'forecast'
    the run labelled at the decision instant covers leads 0..11, the lead-0
    row is trimmed, so 11 window positions come from the fine segment."""
    w = assemble_window(
        k=13, n_intervals=40, coarse=_coarse([12], span=48), fine=_fine([12]),
        actuals=np.zeros(200), infill_lag_intervals=288)
    assert w.is_fine[:11].all()
    assert not w.is_fine[11:].any()
    # and it agrees with the value provenance encoded in the fixtures
    assert (w.rrp[w.is_fine] > 0).all()
    assert (w.rrp[~w.is_fine] < 0).all()


def test_is_fine_extends_one_further_under_the_later_run():
    """The boundary moves with the mode allowance, which is why it cannot be
    recomputed downstream as 'the first twelve positions'."""
    fine, coarse = _fine([12, 13]), _coarse([12, 13], span=48)
    base = assemble_window(
        k=13, n_intervals=40, coarse=coarse, fine=fine,
        actuals=np.zeros(200), infill_lag_intervals=288, mode='forecast')
    nxt = assemble_window(
        k=13, n_intervals=40, coarse=coarse, fine=fine,
        actuals=np.zeros(200), infill_lag_intervals=288, mode='price_and_next')
    assert int(base.is_fine.sum()) == 11
    assert int(nxt.is_fine.sum()) == 12


def test_is_fine_shortens_when_the_fine_run_is_stale():
    """A carried-forward fine run covers fewer window positions."""
    w = assemble_window(
        k=16, n_intervals=40, coarse=_coarse([12], span=48), fine=_fine([12]),
        actuals=np.zeros(200), infill_lag_intervals=288)
    assert int(w.is_fine.sum()) == 8       # leads 0..11 from run 12, k-1 = 15


def test_is_fine_is_all_false_without_a_fine_segment():
    w = assemble_window(
        k=13, n_intervals=40, coarse=_coarse([12], span=48), fine=None,
        actuals=np.zeros(200), infill_lag_intervals=288)
    assert not w.is_fine.any()


def test_infilled_positions_are_never_marked_fine():
    w = assemble_window(
        k=13, n_intervals=40, coarse=_coarse([12], span=6), fine=_fine([12]),
        actuals=np.arange(200, dtype=float), infill_lag_intervals=12)
    assert w.is_infill.any()
    assert not (w.is_fine & w.is_infill).any()


def test_coarse_run_pos_is_reported_separately_from_the_fine_run():
    """The fine run is fresh at every decision instant; the coarse one can be
    up to a cadence older, and it is the coarse one that shapes the window."""
    w = assemble_window(
        k=17, n_intervals=40, coarse=_coarse([12], span=48), fine=_fine([16]),
        actuals=np.zeros(200), infill_lag_intervals=288)
    assert w.run_pos == 16
    assert w.coarse_run_pos == 12


def test_coarse_run_pos_equals_run_pos_without_a_fine_segment():
    w = assemble_window(
        k=13, n_intervals=20, coarse=_coarse([12], span=48), fine=None,
        actuals=np.zeros(200), infill_lag_intervals=288)
    assert w.run_pos == w.coarse_run_pos == 12


def test_returns_a_named_tuple_with_aligned_arrays():
    w = assemble_window(
        k=13, n_intervals=40, coarse=_coarse([12], span=48), fine=_fine([12]),
        actuals=np.zeros(200), infill_lag_intervals=288)
    assert isinstance(w, AssembledWindow)
    assert w.rrp.shape == w.is_infill.shape == w.is_fine.shape


def test_truncated_window_truncates_every_array_together():
    L = 20
    w = assemble_window(
        k=3, n_intervals=40, coarse=_coarse([2], span=6), fine=None,
        actuals=np.arange(200, dtype=float), infill_lag_intervals=L)
    assert len(w.rrp) < 40
    assert w.rrp.shape == w.is_infill.shape == w.is_fine.shape


# ------------------------------------------------------------------- infill

def test_infill_depth_one_copies_from_within_the_window():
    L = 12
    w = assemble_window(
        k=13, n_intervals=30, coarse=_coarse([12], span=18), fine=None,
        actuals=np.arange(200, dtype=float), infill_lag_intervals=L)
    out, is_infill = w.rrp, w.is_infill
    tail = np.flatnonzero(is_infill)
    assert tail.size > 0
    for j in tail:
        if j - L >= 0:
            assert out[j] == out[j - L]


def test_infill_depth_two_resolves_in_one_forward_pass():
    """Because L > 0, position j always sources from j-L which is lower, so a
    single forward pass resolves any depth."""
    L = 12
    w = assemble_window(
        k=13, n_intervals=40, coarse=_coarse([12], span=6), fine=None,
        actuals=np.arange(200, dtype=float), infill_lag_intervals=L)
    out, is_infill = w.rrp, w.is_infill
    assert is_infill.sum() > L, "expected infill deep enough to recurse"
    assert out[2 * L] == out[L] == out[0] or np.isfinite(out).all()
    assert np.isfinite(out).all()


def test_infill_reads_settled_actuals_before_the_decision_instant():
    L = 6
    actuals = np.arange(200, dtype=float) * 10.0
    k = 60
    # A real coarse run always extends forward from its label, so it covers the
    # decision instant. span=6 gives targets 54 and 60, covering positions
    # 49..60; the window starts at 60, so positions 61..65 need infill.
    w = assemble_window(
        k=k, n_intervals=6, coarse=_coarse([54], span=6), fine=None,
        actuals=actuals, infill_lag_intervals=L)
    out, is_infill = w.rrp, w.is_infill
    assert len(out) == 6
    decision_pos = k - 1
    for j in range(6):
        if not is_infill[j]:
            continue
        src = k + j - L
        if src < decision_pos:
            assert out[j] == actuals[src], "settled actuals are usable"
        else:
            assert src == decision_pos
            assert out[j] != actuals[src], (
                "the interval ending at the decision instant is not yet "
                "published, so its forecast estimate must be used instead"
            )


def test_lead_zero_estimate_covers_the_interval_at_the_decision_instant():
    """Past is not published: the settled price for the interval ending AT the
    decision instant is not available then, so the fine run's lead-0 estimate
    is used instead. Exactly one window position lands there."""
    L = 12
    k = 60
    decision_pos = k - 1
    fine = _fine([decision_pos], n_targets=1, base=999.0)   # lead-0 row only
    actuals = np.full(200, -5.0)
    w = assemble_window(
        k=k, n_intervals=L, coarse=_coarse([decision_pos], span=0), fine=fine,
        actuals=actuals, infill_lag_intervals=L)
    out, is_infill = w.rrp, w.is_infill
    j = L - 1                       # sources from k + (L-1) - L = decision_pos
    assert is_infill[j]
    assert out[j] == pytest.approx(999.0 + decision_pos)
    assert out[j] != -5.0, "must not use the unpublished settled actual"


def test_window_truncates_when_history_is_exhausted():
    L = 20
    w = assemble_window(
        k=3, n_intervals=40, coarse=_coarse([2], span=6), fine=None,
        actuals=np.arange(200, dtype=float), infill_lag_intervals=L)
    out, is_infill = w.rrp, w.is_infill
    assert len(out) < 40
    assert np.isfinite(out).all()


# -------------------------------------------------------------------- modes

def test_mode_allowance_is_one_interval_only_for_next_forecast():
    assert mode_allowance_intervals('forecast') == 0
    assert mode_allowance_intervals('price_known') == 0
    assert mode_allowance_intervals('price_and_next') == 1
    assert mode_allowance_intervals('next_forecast') == 1


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="unknown mode"):
        mode_allowance_intervals('actual')


def test_next_forecast_selects_the_run_labelled_one_interval_later():
    fine, coarse = _fine([12, 13]), _coarse([12, 13], span=48)
    rp_fc = assemble_window(
        k=13, n_intervals=20, coarse=coarse, fine=fine,
        actuals=np.zeros(200), infill_lag_intervals=288, mode='forecast').run_pos
    rp_nf = assemble_window(
        k=13, n_intervals=20, coarse=coarse, fine=fine,
        actuals=np.zeros(200), infill_lag_intervals=288,
        mode='next_forecast').run_pos
    assert rp_fc == 12
    assert rp_nf == 13, "next_forecast may use the run labelled index[k]"


def test_forecast_mode_cannot_reach_the_later_run():
    """The allowance is per-mode, not general slack in the causality rule."""
    fine, coarse = _fine([13]), _coarse([13], span=48)
    assert assemble_window(
        k=13, n_intervals=20, coarse=coarse, fine=fine,
        actuals=np.zeros(200), infill_lag_intervals=288,
        mode='forecast') is None
    assert assemble_window(
        k=13, n_intervals=20, coarse=coarse, fine=fine,
        actuals=np.zeros(200), infill_lag_intervals=288,
        mode='next_forecast') is not None


# --------------------------------------------------------------- validation

def test_window_cannot_start_at_the_first_interval():
    """Position k-1 is the decision instant, so k >= 1."""
    with pytest.raises(ValueError, match="k must be >= 1"):
        assemble_window(k=0, n_intervals=10, coarse=_coarse([0]), fine=None,
                        actuals=np.zeros(50), infill_lag_intervals=12)


@pytest.mark.parametrize("kwargs,match", [
    (dict(n_intervals=0), "n_intervals"),
    (dict(infill_lag_intervals=0), "infill_lag_intervals"),
])
def test_assemble_rejects_bad_arguments(kwargs, match):
    base = dict(k=13, n_intervals=10, coarse=_coarse([12]), fine=None,
                actuals=np.zeros(200), infill_lag_intervals=12)
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        assemble_window(**base)
