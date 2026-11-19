# usage_forecast — Specification

## Purpose and scope

Household load forecasting from a site's own metered history, for the rolling
coordination types. Enabled by `rolling.usage_forecast`; when it is `None` load
is known perfectly over the horizon. Solar has its own forecast
(`solar_forecast_spec.md`), and the two are switched on independently.

Implementation: `model/forecast/usage.py`. Config: `UsageForecastConfig` in
`model/core/scenario.py`.

---

## Parameters

| Parameter | Value | Meaning |
|---|---|---|
| `history_days` | 28 | trailing local days behind the profile |
| `min_history_days` | 1 | below this, `get` returns `None` |
| `smoothing_intervals` | 7 | width of the shape's moving average, odd, 1 disables |
| `dow_level` | `false` | split the daily-energy level by day-of-week |
| `lookback_hours` (x) | 0.5 | trailing window behind the adjustment |
| `decay_hours` (y) | 1.5 | decay constant of the adjustment, 0 disables |
| `min_day_coverage` | 0.85 | real intervals a day needs to enter the history |

These are the `UsageForecastConfig` defaults, fitted on EDP data (see Fitting
and Validation).

---

## The forecast

Three separable pieces: a scalar daily energy, a normalised time-of-day shape,
and a correction from recently observed load.

### Level

For each day `d` in the trailing history window, take the daily total energy
`E_d`. The level for a target day is the mean of `E_d` over history days
matching that target's day type (weekday or weekend), or its day-of-week when
`dow_level` is set.

### Shape

For each history day, normalise `s_d[j] = load_d[j] / E_d`, so each day sums to
one. Average `s_d` across history days of the matching day type, apply a
circular centred moving average of width `smoothing_intervals` across the
time-of-day axis, and renormalise to sum one.

Circular means the 23:45 slot's neighbours include 00:10.

### Recent adjustment

At decision instant `D`, with `P` the base profile prediction:

```
a     = mean over the last x hours of (actual_load − P)
L̂(ℓ) = max(0, level(w) · shape(g)[slot(ℓ)] + a · exp(−ℓ / y))
```

where `w` is the target's day-of-week, `g` its day type, and lead `ℓ` is
measured to interval end, so window position `j` has `ℓ = (j+1) · 5 min`.

The adjustment is additive rather than multiplicative because a scalar
correction diverges wherever the profile approaches zero, which at 5-minute
household resolution is common. Clipping at zero biases the forecast upward
wherever it fires, so clips are counted.

Too long a decay is the dangerous direction: `a` is a mean over a short,
strongly autocorrelated window, so its sampling noise exceeds what the sample
count suggests, and spreading it far across the horizon injects more noise than
it removes bias.

---

## The profile is rebuilt once per local day

Tables built at the start of local day `d` use history strictly before day `d`
and are applied to every target in the window, including targets on day `d+1`.
Rebuilding per target day would leak: at a decision instant inside day `d`, a
table for day `d+1` would have been built from history including day `d`.

Cost is 366 table builds per site-year, maintained incrementally at O(288) per
day.

---

## Causality

> The forecast for interval `k` may use only load observed at or before
> `decision_instant(k)`.

Load is one interval more permissive than price. The settled price for the
interval ending at `D` is not published at `D`; metered load for that interval
is observed at `D` by a local controller. So labels `≤ decision_instant` are
readable and labels `> decision_instant` are not.

A leak would be invisible in every downstream artefact, so it is verified
behaviourally in both directions: re-running against a trace whose intervals
from `k` onward are overwritten must leave every forecast bit-identical, and
poisoning interval `k−1` must change the adjustment.

---

## What the component commits

The rolling loop supplies **actual load at window position 0** and forecast
load from position 1 onward. The load forecast affects the plan, not the
execution.

At 5-minute scheduling against 5-minute data there is no intra-interval, and
real inverters track house load in seconds, so holding the controller to its
forecast of the current interval would model a limitation created by the
discretisation rather than a real one.

This also closes an export-limit hazard. `_reconstruct_flows` derives import
and export from actual solar, load and the committed battery flows with no clip
or assertion on the export limit, which is enforced inside the LP. Solving
against forecast load at position 0 could therefore produce reconstructed
export above the limit; actual load at position 0 cannot.

The consequence is that the measured cost of load forecast error reaches the
result only through the SOC trajectory and timing. `commit_actual_load: false`
brackets the other end of the range and is a diagnostic, not a production
setting: on a 45-day synthetic site it cost five times as much as committing
actual load, and it breached the export limit in 6 intervals where committing
actual load breached none.

---

## Interface

```python
@dataclass(frozen=True)
class LoadSlice:
    load_kwh: np.ndarray      # kWh per window interval, target order
    adjustment_kwh: float     # raw additive term, before decay
    n_lookback: int           # intervals of history behind the adjustment
    n_clipped: int            # window positions clipped at zero
    warm_up: bool             # profile built on less than history_days

class UsageForecastProvider:
    def get(self, k: int, n_intervals: int) -> LoadSlice | None: ...
```

Indexed by interval position, not timestamp: the forecast is built from the
site's own trace on the component's own index, so there is no external run
store to align to.

`None` means no tables exist for the decision instant's local day.

`UsageForecastConfig` lives in `core/scenario.py` beside `BatterySpec`, since
`RollingConfig` holds it and core must not import from the forecast package.
`rolling.usage_forecast = None` means load is known perfectly, which is a rung
of the measurement ladder rather than a degenerate case.

```yaml
rolling:
  commit_actual_load: true
  usage_forecast:
    history_days: 28
    min_history_days: 1
    smoothing_intervals: 7
    dow_level: false
    lookback_hours: 0.5
    decay_hours: 1.5
    min_day_coverage: 0.85
```

Construct with `UsageForecastProvider.from_series(load_kwh, state, config,
max_horizon_intervals, interval_hours, valid)`, which resolves the local clock
so callers need not handle `ClockArrays`.

`RollingDiagnostics` carries `load_forecast_used`,
`sc_fallback_no_load_forecast`, `load_windows_warm_up` and
`load_positions_clipped`, plus `infeasible_from_site_forecast`, shared with the
solar forecast: the count of infeasible windows that became feasible when
re-solved with actual site data, which separates infeasibilities caused by
the site forecasts from those that would have occurred anyway. The re-solve
fires only on failure and its solution is discarded. `RetainedWindow` carries
`load_forecast_kwh`, so a per-interval post-mortem can tell a price-driven
mistake from a load-driven one.

---

## Degradation

Deterministic and ordered, so a short history produces a worse forecast rather
than a hole:

| Missing | Falls back to |
|---|---|
| day type with no history days | all-days shape |
| day-of-week with no history days | day-type mean level |
| day type with no level history | all-days mean level |
| no complete prior local day | `get` returns `None` |

Every slice carries `warm_up`, so warm-up intervals can be excluded from a
reported number. This matters because Energy Masters periods run 61 to 273
days, and requiring a full 28 days of history would cost a third of the
shortest site.

Partial local days at the start and end of a period are excluded by the
coverage gate, since a partial day's depressed total would bias the level
downward.

Local days on a DST transition hold 276 or 300 intervals rather than 288.
Missing slots are filled by circular interpolation before smoothing; a zero
there would assert no consumption rather than no observation.

---

## Infilled intervals

Intervals with non-zero `load_infill` are excluded from the profile history,
gated at day level through `min_day_coverage`, and from any evaluation of
forecast error.

The exclusion is required rather than tidy: EDP's Method 2 load infill copies
the same day of week, which is close to what this forecaster does. Infilled
history restates the profile rather than evidencing it, and an infilled
evaluation target scores the forecaster against a near-copy of itself.

The adjustment still reads whatever the trace holds at the decision instant,
because a controller reading its own meter sees a real number there.

Infill is not spread evenly across a period. A file can open with a contiguous
block of infilled days, in which case the site has no usable history until the
block ends and the forecaster correctly produces no tables until then.

---

## Known limitations

**One global parameter pair harms a tail of households.** At the fitted
`(0.5, 1.5)` across 433 sites, 387 are better off than with no adjustment, 46
are made worse than doing nothing — the worst by 33 % — and 75 capture less
than half the benefit their own parameters would give. Median loss against a
site's own optimum is 1.67 %, with a 90th percentile of 8.4 %.

A real retailer runs one model across its fleet, so this is a property of the
thing being modelled rather than an artefact, but it means Energy Masters sites
should be inspected individually: one of ten falling in the harmed group would
be unremarkable.

**The level lags a trend by half the history window.** A trailing 28-day mean
sits about fourteen days behind a rising load. Over a full year this cancels;
over a two-month window inside a seasonal ramp it does not.

*Fix path:* fit a linear trend to the trailing daily totals and extrapolate one
day, or run a shorter history window for the level than for the shape, since
the level is one scalar per day. Neither is built.

**Error is correlated with value.** A trailing mean under-forecasts load on hot
days, and hot days are where price spikes and battery headroom concentrate.
Under-forecasting load into a high-price evening causes the plan to export on
the assumption that less energy is needed at home, so the battery empties early
and the site imports at spike prices. The recent adjustment is the mitigation.
Per-day error on the highest-headroom dates should be reported separately
rather than pooled.

**Infeasibility risk rises.** With curtailment pinned to zero, an under-forecast
of load on a high-solar interval leaves the LP with surplus solar it cannot
place. The existing self-consumption fallback absorbs each occurrence, but
diagnostics must separate infeasibilities occurring under forecast load from
those that would have occurred under perfect load.

*Fix path:* curtailment beyond position 0 is never committed, so pinning it to
zero across the whole planning window is stricter than the physical constraint
requires. Allowing curtailment at a high cost from position 1 onward, with
position 0 still pinned, lets the LP produce a plan instead of failing. Not
built.

---

## Fitting

`model/dev/fit_usage_forecast.py`. Forecast error only — no LP, no prices.

```
python3 -m model.dev.fit_usage_forecast --year 2022 \
    --data-root "/Volumes/Manual/Thesis Project"
```

**Metric.** Cumulative energy error over 1, 3, 6 and 12 hours from each
decision instant. The battery plans an energy trajectory, so what the LP
consumes is how much the house will take between now and the evening peak, not
what it draws in any particular five minutes. Per-interval MAE is reported
alongside; where they disagree the cumulative metric governs.

Window position 0 is excluded, being committed with actual load.

**The null is `y = 0`.** The adjustment has to beat it.

**Comparing profile configurations** — `history_days`, `smoothing_intervals`,
`dow_level` — means reading the absolute errors on the null row, not the score,
which is normalised to each run's own null.

**Two selection criteria.** The pooled score stacks every decision instant from
every site into one residual matrix and is therefore volume weighted; the
equal-weight criterion takes the median of per-site scores. The harness reports
both, flags disagreement, and reports the count of sites the adjustment harms
at each grid point.

**Grid edges.** A pooled optimum on a grid boundary has run out of grid rather
than found an optimum, and triggers a warning. `x` at one interval is exempt,
being the floor of the parameter space.

**Held-out discipline.** Fit on EDP and apply the result fixed to Energy
Masters. Fitting on EM, even on forecast error rather than on VCR, would put a
free parameter into the reported number. A per-site pair is rejected for the
same reason.

**Site discovery.** With `--year`, sites and their states come from
`<data-root>/data/processed/edp/{year}_selected_sites_clean.csv` and traces
from `<data-root>/data/clean/edp/{year}/edp_clean_{site}_{year}.parquet`. State
is resolved per site, since the local day boundary that the day-type split and
day-of-week level depend on is state-specific, and a wrong state silently
mislabels every weekend. `--edp` therefore refuses to run without an explicit
`--state`.

Results are written to `<data-root>/data/processed/usage_fit/` as
`usage_fit_pooled_{year}.csv`, `usage_fit_per_site_{year}.csv`,
`usage_fit_equal_weight_{year}.csv` and `usage_fit_grid_by_site_{year}.csv`.

---

## Verification

`model/dev/check_usage_forecast.py`, on a synthetic trace by default or a real
site with `--year` and `--site`. Asserts:

- shapes sum to one, are non-negative, and differ between day types
- the smoothing machinery reduces roughness when enabled
- tables exist contiguously from the first usable day
- the warm-up flag clears once history is deep enough
- causality in both directions
- level bias matches a plain trailing mean of daily totals, which separates
  implementation error from the estimator's inherent lag
- the adjustment is positive on a constructed anomaly and reduces error there
- three days of history still produces a forecast, and the first interval of a
  period returns `None`

Whether the adjustment beats the null on average is not asserted. That is an
empirical question about real households, answered by the fit; the check prints
the comparison and asserts only the mechanism.

There is no pytest module for this component; the check script is the only
verification. Fix path: move its assertions into `model/tests/` against a
synthetic fixture.

---

## Validation

Fit on 433 EDP sites, 2022, all four states, cumulative energy error, infilled
intervals excluded. `x` and `y` are interior to the fitted grid. The profile
settings were each tested by re-running the fit with one changed and comparing
the no-adjustment row: 56 history days was 4.4–7.2 % worse than 28,
`dow_level` on was 2.8–5.7 % worse than off, and smoothing left the profile
unchanged but improved the fitted result at 36 of 37 grid points.

### Accuracy at the fitted parameters

433 EDP sites, 2022, mean absolute cumulative energy error in kWh:

| Horizon | No adjustment | Fitted | Change |
|---|---|---|---|
| 1 h | 0.3940 | 0.3073 | −22.0 % |
| 3 h | 1.0417 | 0.9156 | −12.1 % |
| 6 h | 1.8354 | 1.6977 | −7.5 % |
| 12 h | 3.1800 | 3.0332 | −4.6 % |

The profile alone is 22–27 % better than repeating yesterday.

---

## Conventions

- Day-of-week and weekday/weekend are local calendar concepts and route through
  `core/localtime.py` with the site's state. `local_wall_clock` evaluates at
  the interval-start instant, which is the decision instant.
- `datetime_aest` is `Etc/GMT-10`, interval-end, no DST.
- Scripts live in `model/dev/` and run from the project root as modules with
  package-qualified imports.
