# solar_forecast — Specification

## Purpose and scope

Solar forecasting from a site's own generation history, for the rolling
coordination types. The counterpart to `usage_forecast_spec.md`. Enabled by
`rolling.solar_forecast`; when it is `None` solar is known perfectly over the
horizon.

Implementation: `model/forecast/solar.py`. Config: `SolarForecastConfig` in
`model/core/scenario.py`.

**Everything here is naive.** A real operator forecasts solar from satellite
imagery, tracking cloud fields and advecting them forward. That is a different
class of information, and nothing built from a site's own generation history
approaches it. No method in this document is a candidate for what Amber does;
they differ in how tight a bound they give, not in kind.

That is the point, and it yields two bounds from one artefact:

1. **A lower bound on what a wholesale-exposed plan can achieve.** With price
   and load forecasts already in place, a rolling run using only information
   available before each decision instant is achievable by construction, so it
   holds regardless of what any retailer does. A better solar forecast is
   purchasable, so real achievable performance is at least this.
2. **An upper bound on the cost of solar forecast error.** A real forecast is
   better than a history-only one, so real error costs less than the gap
   between perfect solar and this.

Both bounds improve as the forecast improves — the lower bound rises, the
upper bound tightens — so there is no tension in making it as good as history
allows.

The one assumption underneath (2) is that an operator's forecast is at least
as good as a history-only one. Satellite products are generic, and a site's own
history carries shading, soiling and orientation that a generic product does
not, so this holds because operators combine the two rather than because
satellite data dominates on its own.

No irradiance data, no numerical weather prediction, no external input of any
kind.

---

## Modes

| Mode | Method | Bound |
|---|---|---|
| `persistence` | the trace 288 intervals (24 h of market time) earlier, repeated | loose |
| `index` (default) | envelope × forecast clear-sky index | tighter |

Both are valid bounds; `index` is simply less wasteful of what the history
contains. `persistence` returns `None` for the first 288 intervals, clips
negatives to zero, and sets no diagnostics or `warm_up`. Its 24-hour lag is in
market time, so across a DST change it repeats a trace offset by an hour in
local time.

---

## The `index` method

Solar splits cleanly into a deterministic part and a stochastic one. Sun
position fixes the shape; cloud fixes the level. So the forecast estimates the
shape from history and predicts only the level.

### Envelope

Per time-of-day slot, the `envelope_quantile` (default 0.95) of generation over
the trailing `history_days`. A clear-sky proxy built from the site's own data,
absorbing season, soiling, shading and array orientation without being told
about any of them.

### Index

The index over any set of slots is a **ratio of sums**:

```
R = Σ actual / Σ envelope
```

not a mean of per-slot ratios. Being energy-weighted, it needs no daylight
window: dawn and dusk carry almost no envelope energy, so they contribute
almost nothing without having to be excluded. It also makes `R` the same kind
of object as `f` below — both are fractions of envelope energy — so
`R_today` is energy achieved so far divided by envelope energy elapsed so far.

Undefined only where the denominator is zero, i.e. overnight.

### Predicting the index

Two stages: form an estimate from history alone, then mix in what today has
shown so far.

```
R_hist = w_prev_share·R_prev + (1 − w_prev_share)·R_mean
f      = envelope energy elapsed so far today / envelope energy for the whole day
a      = min(1, f / f_on) · w_today_max
R̂      = a·R_today + (1 − a)·R_hist
```

| term | meaning |
|---|---|
| `R_today` | today's index so far |
| `R_prev` | the previous day's index |
| `R_mean` | the index over the whole trailing `history_days` |

`R_hist` does not depend on the time of day. It is formed once per day and is
the whole forecast overnight, when nothing of today has been observed.

**`R_mean` is what makes `R_hist` a shrinkage estimator rather than a copy of
yesterday.** A single day's index is noisy, so a forecast that just repeats it
is over-dispersed: too many days predicted very clear or very cloudy. The
weight on the trailing mean pulls predictions toward climatology by the amount
that noise warrants.

**`f` is measured in envelope energy, not clock time.** The envelope is zero
before dawn and after dusk, so `f` is zero all night, rises through the day and
reaches one at dusk — no separate daylight calculation is needed. It also
weights each period by how much generation is at stake in it, so the first hour
after dawn counts for the few per cent it is worth.

**`a` is the weight on today's observation.** It is zero overnight, ramps in
over the opening fraction `f_on` of the day — below which too little of today
has been seen for `R_today` to mean anything — and then holds at
`w_today_max`. It does not keep rising through the day: as the day proceeds,
the remaining hours being forecast shrink and shift later, so the morning is
predicting an ever smaller and more distant window, and today's observation
does not become a proportionally better guide to it.

For target intervals on a **later** day, `a = 0`: today's partial index says
nothing about tomorrow beyond what today's completed index will say, and that
is not known until today ends.

### Recent adjustment

```
a     = (Σ actual − Σ forecast) / Σ envelope     over the last x hours
Î(ℓ)  = clip(R̂ + a · exp(−ℓ / y), 0, index_max)
Ŝ(ℓ) = Î(ℓ) · envelope[slot(ℓ)]
```

The residual is energy-weighted for the same reason the index is, so a
lookback window straddling dawn or dusk is not dominated by intervals where
almost nothing was generated either way.

Applied to the **index**, not to kWh. This is the one place solar is easier
than load: the index is already scale-free, so the near-zero blow-up that
forced an additive adjustment on load does not arise, and the envelope
restores the scale afterwards.

Clipping is asymmetric by design. The **forecast** index is clipped to
`[0, 1]`, so the most it can predict is a day as good as the envelope. The
**observed** index, used for `R_today`, `R_prev`, `R_mean` and the adjustment
residual, is left unclipped: the envelope is a high quantile, so a few per cent
of real observations exceed it by construction, and clipping those would bias
every input downward.

### Forecast

`envelope[slot] × Î` per interval, in target order.

---

## Parameters

`SolarForecastConfig` defaults, fitted on EDP data (see Fitting and Validation):

| Parameter | Default | Meaning |
|---|---|---|
| `mode` | `index` | `persistence` or `index` |
| `history_days` | 7 | trailing local days behind the envelope and `R_mean` |
| `envelope_quantile` | 0.95 | clear-sky proxy per slot |
| `w_prev_share` | 0.6 | share of `R_hist` taken from the previous day, rest from the trailing mean |
| `w_today_max` | 0.8 | ceiling on the weight given to today's observation |
| `f_on` | 0.15 | elapsed fraction over which that weight ramps in |
| `lookback_hours` (x) | 5/60 | trailing window behind the adjustment |
| `decay_hours` (y) | 1.0 | decay constant of the adjustment |
| `index_max` | 1.0 | upper clip on the FORECAST index |
| `min_day_coverage` | 0.85 | intervals a day needs to enter the history |

---

## Rebuild cadence

The envelope is rebuilt once per local day from days strictly before it, and
applied to every target in the window. Rebuilding per target day would leak:
at a decision instant inside day `d`, a table for day `d+1` would draw on day
`d`, most of which has not happened.

The index prediction updates every interval. It is two scalars and a weight, so
there is no cost argument for updating it less often.

---

## Causality

> The forecast for interval `k` may use only generation observed at or before
> `decision_instant(k)`.

Same rule and same one-interval-more-permissive boundary as load: metered
generation for the interval ending at `D` is observed at `D`. Verified the same
way — poisoning the trace from `k` onward must leave every forecast identical,
and poisoning `k−1` must change the adjustment.

---

## What the component commits

Actual solar at window position 0, forecast from position 1 onward, matching
the load treatment. At 5-minute scheduling against 5-minute data there is no
intra-interval, and inverters track generation in real time.

---

## Interface

```python
@dataclass(frozen=True)                  # model/core/forecast.py
class SolarSlice:
    solar_kwh: np.ndarray                # kWh per window interval, >= 0
    index_hist: float = 0.0              # R_hist
    index_today: float = 0.0             # blended R̂ for same-day targets
    intraday_weight: float = 0.0         # a
    adjustment: float = 0.0              # index adjustment before decay
    n_clipped: int = 0                   # positions clipped at 0 or index_max
    warm_up: bool = False                # envelope built on < history_days

class SolarForecastProvider:             # model/forecast/solar.py
    @classmethod
    def from_series(cls, solar_kwh, state, config, max_horizon_intervals,
                    interval_hours=5/60, valid=None) -> SolarForecastProvider: ...
    def get(self, k: int, n_intervals: int) -> SolarSlice | None: ...
```

Indexed by interval position, as the load provider is. `None` when no tables
exist for the decision instant's local day (no complete prior local day); the
caller falls back to self-consumption. `valid` gates the envelope history only;
the rolling loop passes `None` (see Infilled intervals).

`RollingDiagnostics` carries `solar_forecast_used`,
`sc_fallback_no_solar_forecast`, `solar_windows_warm_up` and
`solar_positions_clipped`, and shares `infeasible_from_site_forecast` with the
load forecast. `RetainedWindow` carries `solar_forecast_kwh`.

---

## Degradation

| Missing | Falls back to |
|---|---|
| fewer than `history_days` of history | envelope from what exists, `warm_up` set |
| no previous-day index | `R_mean` in place of `R_prev` |
| envelope zero at a slot | forecast zero at that slot |
| no complete prior local day | `get` returns `None` |

---

## Infilled intervals

`pv_infill` is **not** excluded from dispatch. It cannot be — the LP needs a
continuous trace — and unlike `load_infill` it is not the site's own history
restated, so there is no circularity.

It is excluded from **fitting**, because infilled days are an easier problem
than real ones. Measured on S0008: 36 days at ≥90% infilled have mean index
0.75 against 0.51 for real days, and within-day variability of 0.134 against
0.271 — **half as variable**, because a state-average profile has other sites'
cloud averaged out of it.

The brightness largely cancels in VCR, which is a ratio against perfect
foresight on the same trace. The smoothness does not: an easier day brings the
forecast rungs closer to perfect foresight than a real day would.

At 10% of days it is a footnote; at 30% the site's solar forecast is being
scored on a materially easier problem than reality. The fit harness records
`pv_infill_frac` per site; run results do not (see Known limitations).

---

## Known limitations

**No weather information.** The method exploits persistence and intraday state
only. It cannot anticipate a front arriving, which is precisely what a real
operator's forecast does.

**Overnight blind spot.** At 2–6 am the intraday term is zero, so the
charge-versus-leave-room-for-solar decision — the single most valuable use of a
solar forecast — is made on persistence alone.

**The envelope lags a step change.** A trailing quantile takes up to
`history_days` to follow a panel fault, new shading or a cleaning.

**The forecast has no within-day shape information.** It is the envelope scaled
by a single number, so it predicts how much a day will generate but never when
cloud will arrive. That is the honest position — a site's own history carries
no evidence about today's cloud timing, and repeating yesterday's shape would
assert timing there is no basis for. The cost is that plans underestimate how
often the battery fills or the export limit binds, which biases the forecast
rungs slightly optimistic. Committing actual solar at position 0 limits how far
that reaches.

**Two parameters are weakly identified.** `f_on` is close to uniform across
the cohort's per-site optima (34/105/136/87/71 sites across the five grid
values), and the fitted `lookback_hours` of one interval is a minority choice:
278 of 433 sites prefer 0.25 h or longer, and 0.25 h is the modal choice in
every one of the four states. Both sit on a very flat part of the surface --
the top ten pooled configurations span 0.7609 to 0.7628 -- so the cost of the
disagreement is small, but neither value should be reported as though the data
pinned it down.

**`history_days` and `envelope_quantile` trade off against each other**
(correlation −0.41 across per-site optima): a short window wants a high
quantile, because the best of seven days is already close to a clear-sky
ceiling. They should be moved together, not independently.

**Infill share is not reported with run results.** `pv_infill_frac` is
computed by the fit harness only; a rolling or cohort run does not carry it
alongside the result, as `run_cohort_rungs.py` does `load_warm_up_frac`. Fix
path: add it to the per-site result rows beside `load_warm_up_frac`.

---

## Fitting

Same harness pattern as `fit_usage_forecast`: cumulative energy error over 1,
3, 6 and 12 hours; window position 0 excluded; the null is no adjustment; fit
on EDP and apply fixed to Energy Masters; grid-edge warnings; pooled and
equal-weight criteria reported side by side.

The harness is `model/dev/fit_solar_forecast.py` (a copy of the usage harness
rather than a shared abstraction, since the two diverge at the envelope/index
step). It takes `--year`, `--data-root`, `--state`/`--edp`, a sampling
`--stride` and a grid over every parameter, and writes
`solar_fit_grid_{tag}.csv`, `solar_fit_per_site_{tag}.csv` and
`solar_fit_equal_weight_{tag}.csv` to `<data-root>/data/processed/solar_fit/`.

---

## Verification

`model/dev/check_solar_forecast.py --year 2022 --site S0008 --data-root ...`
asserts, on a real site:

- tables exist contiguously from the first usable day; the envelope is
  non-negative and daylight-shaped, and some observations exceed it, as a
  quantile should
- the trailing index is a sane fraction
- the intraday weight is zero before the envelope lights up, never exceeds its
  ceiling, reaches it by mid-afternoon, and where it is zero the forecast is
  the history estimate
- causality in both directions: no forecast depends on generation at or after
  the decision instant, and the interval ending at the decision instant is used
- both modes produce finite non-negative forecasts

Whether `index` beats `persistence`, or either beats no adjustment, is answered
by the fit, not asserted. There is no pytest module for this component. Fix
path: move the check's assertions into `model/tests/` against a synthetic
fixture.

---

## Validation

Fit on 433 EDP sites, 2022, all four states: the equal-weight optimum on a grid
of 241,920 configurations, with no boundary hit. Equal weight rather than pooled
because the parameters serve VCR, which is scale-free; the two criteria agreed
at cohort scale.

Against the history estimate alone, cumulative energy error: **−41.7 % at 1 h, −28.2 % at 3 h,
−17.2 % at 6 h, −8.5 % at 12 h.** Nearly all the value is at short lead, which
is the overnight blind spot quantified.

At the time the component was built, the VCR gap on the Energy Masters cohort
left unexplained by price and load forecast error was 0.14; this component
exists to bound how much of that solar forecast error can account for.

---

## Conventions

- Local calendar concepts route through `core/localtime.py` with the site's
  state.
- `datetime_aest` is `Etc/GMT-10`, interval-end, no DST.
- Scripts live in `model/dev/`, run from the project root as modules, with
  package-qualified imports.
