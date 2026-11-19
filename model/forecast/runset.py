"""RunSet: forecast runs indexed for lookup at a decision instant.

Every source presents its data as one or two RunSets, a fine one (5-minute
targets) and a coarse one (30-minute targets). Everything downstream of this
module is source-agnostic.

Positions, not timestamps
-------------------------
The whole forecast subsystem works in INTERVAL POSITIONS on the model's
5-minute grid rather than in timestamps. Position p means the interval whose
interval-END label is grid_start + p * 5 min, which is the same indexing the
site traces and the price series use.

This is not premature optimisation. Window assembly runs ~105,000 times per
site-year, and integer position arithmetic keeps the whole path in numpy with
no timestamp handling, no timezone questions, and no pandas allocation. The
conversion happens once, here, at construction.

Storage
-------
Runs are held in CSR form: a sorted array of run positions, an offsets array,
and flat lead/value arrays. Selecting the latest run at a decision instant is
a binary search; reading it is a slice.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RunSet:
    """Forecast runs from one source segment, indexed by run position.

    Attributes:
        run_pos: Sorted, unique interval positions of each run's nominal
            label. int64.
        offsets: CSR offsets into target_pos / values, length len(run_pos)+1.
        target_pos: Interval position of each target. int64. Within a run
            these are strictly increasing.
        values: RRP in $/MWh for each target. float64.
        resolution_intervals: Target spacing in 5-minute intervals. 1 for a
            5-minute segment, 6 for a 30-minute segment.

    A 30-minute target at position p covers the six 5-minute intervals
    p-5 .. p inclusive, following the interval-end convention. Expansion is
    done at assembly time, not here, so the RunSet stays a faithful
    representation of what the source published.
    """
    run_pos: np.ndarray
    offsets: np.ndarray
    target_pos: np.ndarray
    values: np.ndarray
    resolution_intervals: int

    def __post_init__(self):
        if self.run_pos.ndim != 1 or self.offsets.ndim != 1:
            raise ValueError("run_pos and offsets must be 1-D")
        if self.offsets.size != self.run_pos.size + 1:
            raise ValueError(
                f"offsets must have len(run_pos)+1 = {self.run_pos.size + 1} "
                f"entries, got {self.offsets.size}"
            )
        if self.target_pos.size != self.values.size:
            raise ValueError("target_pos and values must be the same length")
        if self.run_pos.size and np.any(np.diff(self.run_pos) <= 0):
            raise ValueError("run_pos must be sorted and unique")
        if self.resolution_intervals < 1:
            raise ValueError(
                f"resolution_intervals must be >= 1, got "
                f"{self.resolution_intervals}"
            )

    @property
    def n_runs(self) -> int:
        return int(self.run_pos.size)

    def latest_at(self, pos: int) -> int | None:
        """Index of the latest run whose label is at or before `pos`.

        This is where causality is enforced structurally: a run labelled after
        `pos` can never be selected, so a provider cannot leak future
        information by accident. The one legitimate exception, the
        `next_forecast` mode, is expressed by passing a later `pos`, which
        keeps it visible at the call site rather than buried here.

        Returns None when no run exists at or before `pos`.
        """
        i = int(np.searchsorted(self.run_pos, pos, side='right')) - 1
        return i if i >= 0 else None

    def run(self, i: int) -> tuple[int, np.ndarray, np.ndarray]:
        """(run position, target positions, values) for run index i."""
        lo, hi = int(self.offsets[i]), int(self.offsets[i + 1])
        return int(self.run_pos[i]), self.target_pos[lo:hi], self.values[lo:hi]

    def max_target_span(self) -> int:
        """Largest (last target position - run position) across all runs.

        The structural horizon of this segment, in intervals.
        """
        if self.n_runs == 0:
            return 0
        last = self.target_pos[self.offsets[1:] - 1]
        return int((last - self.run_pos).max())

    @classmethod
    def from_long(
        cls,
        run_pos: np.ndarray,
        target_pos: np.ndarray,
        values: np.ndarray,
        resolution_intervals: int,
    ) -> "RunSet":
        """Build from long-form (run, target, value) triples.

        Sorts by (run, target) and builds the CSR index. Input need not be
        sorted and may hold runs in any order, which is what a concatenation
        of monthly store files looks like.

        Raises:
            ValueError: on duplicate (run, target) pairs. The store guarantees
                uniqueness after the INTERVENTION filter, so a duplicate here
                means either that filter did not run or the source changed.
                Resolving it silently would pick an arbitrary price on exactly
                the intervals where the two cases differ most.
        """
        run_pos = np.asarray(run_pos, dtype=np.int64)
        target_pos = np.asarray(target_pos, dtype=np.int64)
        values = np.asarray(values, dtype=np.float64)
        if not (run_pos.size == target_pos.size == values.size):
            raise ValueError("run_pos, target_pos and values must be the same length")

        order = np.lexsort((target_pos, run_pos))
        run_pos, target_pos, values = run_pos[order], target_pos[order], values[order]

        if run_pos.size > 1:
            same = (np.diff(run_pos) == 0) & (np.diff(target_pos) == 0)
            if same.any():
                j = int(np.flatnonzero(same)[0])
                raise ValueError(
                    f"duplicate (run, target) pair at run position "
                    f"{run_pos[j]}, target position {target_pos[j]}: "
                    f"values {values[j]} and {values[j + 1]}. The store must "
                    "be unique on (run_time, region, target_time) after the "
                    "INTERVENTION == 0 filter."
                )

        uniq, starts = np.unique(run_pos, return_index=True)
        offsets = np.concatenate([starts, [run_pos.size]]).astype(np.int64)
        return cls(
            run_pos=uniq.astype(np.int64),
            offsets=offsets,
            target_pos=target_pos,
            values=values,
            resolution_intervals=int(resolution_intervals),
        )
