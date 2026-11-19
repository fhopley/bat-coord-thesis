"""Manual review registry for battery capacity estimates.

The automated estimate in estimate_battery_capacity.py assumes a single
static capacity for a site's whole valid period. That assumption needs a
human check per site, since two failure modes are only really visible by
eye on the plot, not from the estimate's own diagnostics:

  - a genuine step change in capacity (e.g. a second battery added
    mid-window) -- the single-static-capacity assumption is simply wrong
    for that site, not just imprecise.
  - a data problem (a gap, a corrupted stretch) that distorts the
    estimate in a way the automated diagnostics don't clearly flag.

This module is a thin CSV-backed registry recording that human review, one
row per site: whether it's accepted for modelling as-is, excluded (with a
reason), or accepted with manually-overridden values. It does not compute
anything -- it only records a decision someone made by looking at the plot,
and provides the accessors other scripts need to apply it.

CSV schema (one row per site):
    site_id             -- e.g. 'EM-427'
    decision            -- one of 'accept', 'exclude', 'override'
    capacity_kwh_override -- usable battery capacity, kWh
    power_kw_override   -- battery max charge/discharge power, kW. Applies
                            to BOTH the flow reconstruction and the LP's
                            BatterySpec, which must agree about the same
                            physical inverter.
    export_kw_override  -- site export limit, kW. Overrides the automated
                            estimate in estimate_export_limit.py, but is
                            itself overridden by an explicit
                            --export-limit-kw on the command line.
    start_date_override -- first date to include, YYYY-MM-DD
    end_date_override   -- last date to INCLUDE, YYYY-MM-DD
    reason              -- free text; the important field. State WHY,
                            not just what (e.g. "second battery visibly
                            added ~2026-01-15, single static capacity
                            assumption invalid for this site" rather than
                            just "capacity changes").
    reviewed_date        -- ISO date the row was last written/updated.

At least one override column must be set when decision == 'override', and
conversely no override column may be set unless decision == 'override'.
The second rule matters more than it looks: without it, filling in an
override while leaving decision at 'accept' would be silently ignored,
which is the worst possible outcome for a field whose entire purpose is to
correct a value someone already knows is wrong.

The date overrides can only CLIP the valid periods that ingest already
validated, never extend them. An override is a statement that some of the
validated data is untrustworthy, which is a narrowing claim; it is not a
mechanism for asserting data ingest rejected.

A site absent from the file is UNREVIEWED, which is a distinct status
from 'accept' -- calling code should be able to tell "no one has looked at
this yet" apart from "someone looked and accepted it", rather than
silently treating an unreviewed site as accepted.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

DEFAULT_REVIEW_CSV_PATH = Path('energy_masters_pipeline/battery_capacity_review.csv')
VALID_DECISIONS = ('accept', 'exclude', 'override')
REVIEW_CSV_COLUMNS = [
    'site_id', 'decision', 'capacity_kwh_override', 'power_kw_override',
    'export_kw_override', 'start_date_override', 'end_date_override',
    'reason', 'reviewed_date',
]
# The override columns, for the "at least one / none unless" validation and
# for reporting which overrides a site actually carries.
OVERRIDE_FIELDS = (
    'capacity_kwh_override', 'power_kw_override', 'export_kw_override',
    'start_date_override', 'end_date_override',
)
# Override fields that must be strictly positive if set.
_POSITIVE_OVERRIDE_FIELDS = (
    'capacity_kwh_override', 'power_kw_override', 'export_kw_override',
)


@dataclass
class CapacityReview:
    site_id: str
    decision: str  # 'accept', 'exclude', 'override'
    capacity_kwh_override: Optional[float]
    reason: str
    reviewed_date: str
    power_kw_override: Optional[float] = None
    export_kw_override: Optional[float] = None
    start_date_override: Optional[pd.Timestamp] = None   # naive, normalised
    end_date_override: Optional[pd.Timestamp] = None     # naive, normalised, INCLUSIVE

    def active_overrides(self) -> dict:
        """The override fields actually set, for validation and reporting."""
        return {f: getattr(self, f) for f in OVERRIDE_FIELDS
                if getattr(self, f) is not None}


def _opt_float(value) -> Optional[float]:
    if value is None or pd.isna(value) or (isinstance(value, str) and not value.strip()):
        return None
    return float(value)


def _opt_date(value, site_id: str, column: str) -> Optional[pd.Timestamp]:
    """Parse a date override cell to a naive normalised Timestamp, or None."""
    if value is None or pd.isna(value) or (isinstance(value, str) and not value.strip()):
        return None
    try:
        ts = pd.Timestamp(value)
    except Exception as e:
        raise ValueError(f"{site_id!r}: could not parse {column}={value!r} as a date: {e}")
    if ts.tz is not None:
        raise ValueError(
            f"{site_id!r}: {column}={value!r} must be a plain date with no timezone; "
            "it is localised to the site's timezone when the periods are clipped"
        )
    return ts.normalize()


def _load_review_df(path: Path = DEFAULT_REVIEW_CSV_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=REVIEW_CSV_COLUMNS)
    df = pd.read_csv(path, dtype={'site_id': str})
    missing_cols = set(REVIEW_CSV_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"{path}: missing expected column(s) {missing_cols}. "
            f"Expected columns: {REVIEW_CSV_COLUMNS}"
        )
    return df


def get_review(site_id: str, path: Path = DEFAULT_REVIEW_CSV_PATH) -> Optional[CapacityReview]:
    """The review row for one site, or None if the site hasn't been
    reviewed yet (absent from the file) -- distinct from an explicit
    'accept' decision.
    """
    df = _load_review_df(path)
    rows = df[df['site_id'] == site_id]
    if len(rows) == 0:
        return None
    if len(rows) > 1:
        raise ValueError(f"{site_id!r} has {len(rows)} rows in {path}; expected at most one")
    row = rows.iloc[0]
    decision = row['decision']
    if decision not in VALID_DECISIONS:
        raise ValueError(
            f"{site_id!r}: decision {decision!r} in {path} is not one of {VALID_DECISIONS}"
        )

    review = CapacityReview(
        site_id=site_id,
        decision=decision,
        capacity_kwh_override=_opt_float(row.get('capacity_kwh_override')),
        power_kw_override=_opt_float(row.get('power_kw_override')),
        export_kw_override=_opt_float(row.get('export_kw_override')),
        start_date_override=_opt_date(
            row.get('start_date_override'), site_id, 'start_date_override'),
        end_date_override=_opt_date(
            row.get('end_date_override'), site_id, 'end_date_override'),
        reason=str(row.get('reason', '')),
        reviewed_date=str(row.get('reviewed_date', '')),
    )

    active = review.active_overrides()
    if decision == 'override' and not active:
        raise ValueError(
            f"{site_id!r}: decision is 'override' but no override column is set in "
            f"{path}. Set one of {list(OVERRIDE_FIELDS)}, or change the decision."
        )
    if decision != 'override' and active:
        raise ValueError(
            f"{site_id!r}: decision is {decision!r} but override column(s) "
            f"{sorted(active)} are set in {path}. An override on a non-'override' "
            "row would be silently ignored; set decision='override'."
        )
    for name in _POSITIVE_OVERRIDE_FIELDS:
        value = getattr(review, name)
        if value is not None and value <= 0:
            raise ValueError(f"{site_id!r}: {name} must be > 0, got {value}")
    if (review.start_date_override is not None and review.end_date_override is not None
            and review.start_date_override > review.end_date_override):
        raise ValueError(
            f"{site_id!r}: start_date_override "
            f"({review.start_date_override.date()}) is after end_date_override "
            f"({review.end_date_override.date()})"
        )
    return review


def upsert_review(
    site_id: str,
    decision: str,
    reason: str,
    capacity_kwh_override: Optional[float] = None,
    power_kw_override: Optional[float] = None,
    export_kw_override: Optional[float] = None,
    start_date_override: Optional[str] = None,
    end_date_override: Optional[str] = None,
    path: Path = DEFAULT_REVIEW_CSV_PATH,
) -> None:
    """Add or update one site's review row. Overwrites any existing row
    for the same site_id rather than appending a duplicate.
    """
    if decision not in VALID_DECISIONS:
        raise ValueError(f"decision must be one of {VALID_DECISIONS}, got {decision!r}")
    overrides = {
        'capacity_kwh_override': capacity_kwh_override,
        'power_kw_override': power_kw_override,
        'export_kw_override': export_kw_override,
        'start_date_override': start_date_override,
        'end_date_override': end_date_override,
    }
    active = {k: v for k, v in overrides.items() if v is not None}
    if decision == 'override' and not active:
        raise ValueError(
            f"at least one of {list(OVERRIDE_FIELDS)} is required when decision='override'"
        )
    if decision != 'override' and active:
        raise ValueError(
            f"override(s) {sorted(active)} given with decision={decision!r}; "
            "they would be silently ignored -- use decision='override'"
        )
    if not reason.strip():
        raise ValueError("reason must not be empty -- state why, not just what")

    df = _load_review_df(path)
    df = df[df['site_id'] != site_id]
    new_row = pd.DataFrame([{
        'site_id': site_id,
        'decision': decision,
        **{k: ('' if v is None else v) for k, v in overrides.items()},
        'reason': reason,
        'reviewed_date': date.today().isoformat(),
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    df = df.sort_values('site_id').reset_index(drop=True)
    df = df[REVIEW_CSV_COLUMNS]
    df.to_csv(path, index=False)


def _guard_not_excluded(review: Optional[CapacityReview], caller: str) -> None:
    if review is not None and review.decision == 'exclude':
        raise ValueError(
            f"{review.site_id!r} is excluded ({review.reason!r}); "
            f"should not be reaching {caller}"
        )


def resolved_capacity_kwh(estimate_capacity_kwh: float, review: Optional[CapacityReview]) -> float:
    """The capacity to actually use for modelling: the manual override if
    one is set, otherwise the automated estimate. Raises if called for an
    excluded site -- callers should check review.decision != 'exclude'
    (or review is None, i.e. unreviewed) before getting this far, since
    there's no sensible capacity value to hand back for a site that's
    been excluded from modelling entirely.

    Keys off the override VALUE rather than the decision: a row can now be
    'override' for its power or dates while leaving capacity to the
    estimate, so decision alone no longer implies a capacity override.
    """
    _guard_not_excluded(review, 'resolved_capacity_kwh')
    if review is not None and review.capacity_kwh_override is not None:
        return review.capacity_kwh_override
    return estimate_capacity_kwh


def resolved_power_kw(estimate_power_kw: float, review: Optional[CapacityReview]) -> float:
    """The battery max charge/discharge power to use for modelling.

    Callers must derive any per-interval power limit FROM this value rather
    than from the estimate, so that the flow reconstruction and the LP's
    BatterySpec cannot end up describing different inverters.
    """
    _guard_not_excluded(review, 'resolved_power_kw')
    if review is not None and review.power_kw_override is not None:
        return review.power_kw_override
    return estimate_power_kw


def apply_period_overrides(
    valid_periods: list, review: Optional[CapacityReview], date_tz,
) -> list:
    """Clip ingest-validated periods to the reviewed date window.

    Clipping only, never extension: each period is intersected with the
    override window and dropped if nothing survives. An override says part
    of the validated data is untrustworthy, which can only narrow.

    Must be called BEFORE anything that consumes valid_periods. In
    particular estimate_battery_capacity takes valid_periods directly, so
    clipping afterwards would leave the capacity estimate computed over
    exactly the stretch the override exists to exclude.

    TIMEZONES. This is the subtle part, and getting it wrong is silent
    rather than loud. Three zones are in play:

      - the override dates in the CSV, which are plain local dates
      - the valid period bounds, which for EM are UTC (see em_ingest,
        load_valid_periods_from_summary) and may be naive or tz-aware
      - the ANALYSIS index, which em_ingest converts to fixed-offset
        Etc/GMT-10

    `date_tz` is the zone the override dates are interpreted IN, and it
    should be the analysis index zone, not the zone the periods happen to
    carry. Every "day" downstream is a midnight in the analysis zone: the
    day browser, the daily attribution grouping, the top-days list. An
    override interpreted in any other zone puts its boundary partway
    through a day and leaves a ragged part-day at the edge of the period.

    The resolved bound is then expressed in whatever representation the
    periods use, tz-aware in their zone or naive, so the comparison is
    like for like. Naive periods are assumed to be UTC, matching
    em_ingest's utc_reading_datetime.

    end_date_override is INCLUSIVE of that calendar day, implemented as an
    exclusive bound at the following midnight, matching how the day
    browser slices a day (>= day_start, < day_end).
    """
    _guard_not_excluded(review, 'apply_period_overrides')
    if review is None:
        return list(valid_periods)
    start_ovr, end_ovr = review.start_date_override, review.end_date_override
    if start_ovr is None and end_ovr is None:
        return list(valid_periods)
    if not valid_periods:
        return list(valid_periods)

    periods_tz = pd.Timestamp(valid_periods[0][0]).tz

    def _bound(naive_date: pd.Timestamp) -> pd.Timestamp:
        """Naive local date -> the same instant, in the periods' own frame."""
        instant = naive_date.tz_localize(date_tz)
        if periods_tz is not None:
            return instant.tz_convert(periods_tz)
        return instant.tz_convert('UTC').tz_localize(None)

    lo = None if start_ovr is None else _bound(start_ovr)
    hi_exclusive = None if end_ovr is None else _bound(end_ovr + pd.Timedelta(days=1))

    clipped = []
    for start, end in valid_periods:
        new_start = start if lo is None else max(start, lo)
        new_end = end if hi_exclusive is None else min(end, hi_exclusive)
        if new_start < new_end:
            clipped.append((new_start, new_end))

    if not clipped:
        lo_s = 'period start' if start_ovr is None else str(start_ovr.date())
        hi_s = 'period end' if end_ovr is None else str(end_ovr.date())
        original = ', '.join(f"{s} to {e}" for s, e in valid_periods)
        raise ValueError(
            f"{review.site_id!r}: date override ({lo_s} to {hi_s}) removes the entire "
            f"valid period. Validated period(s): {original}"
        )
    return clipped


def describe_overrides(review: Optional[CapacityReview]) -> str:
    """One-line summary of the overrides in force, for run output."""
    if review is None:
        return 'overrides: none (site unreviewed)'
    active = review.active_overrides()
    if not active:
        return 'overrides: none'
    parts = []
    for name, value in active.items():
        short = name.replace('_override', '')
        if isinstance(value, pd.Timestamp):
            parts.append(f"{short}={value.date()}")
        else:
            parts.append(f"{short}={value:g}")
    return 'overrides: ' + ', '.join(parts)