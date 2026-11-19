# forecast_providers — Specification

## Purpose and scope

The price-forecast side of the rolling coordination types. A provider
answers one question:

> What did the price forecast look like at this instant, over this many
> intervals?

Sources differ in horizon, resolution, update cadence and publication lag;
all of that is absorbed inside the provider. The rolling loop receives a
finished vector of wholesale RRP in $/MWh, one per window interval, or `None`.

Covers the contract in `model/core/forecast.py`, the shared assembly in
`model/forecast/` (`runset.py`, `assembly.py`), the providers (`perfect.py`,
`predispatch.py`) and the registry (`registry.py`). The pre-dispatch store
the providers read is specified in `predispatch_data_spec.md`. How the rolling
loop uses the providers is specified in `rolling_optimisation_spec.md`. The
household load and solar forecasts, whose `LoadSlice`/`SolarSlice` contracts
share `model/core/forecast.py`, are specified in `usage_forecast_spec.md` and
`solar_forecast_spec.md`.

---

## Contract

```python
@dataclass(frozen=True)
class ForecastSlice:
    run_time: pd.Timestamp                    # label of the fine run (freshest in the window)
    rrp_mwh: np.ndarray                       # $/MWh, window order
    is_infill: np.ndarray                     # bool: extrapolated, not forecast
    coarse_run_time: pd.Timestamp | None = None   # label of the coarse run; defaults to run_time
    is_fine: np.ndarray | None = None             # bool: from the 5-minute segment; defaults to all True

class ForecastProvider(Protocol):
    def horizon_available(self) -> int: ...
    def get(self, decision_instant: pd.Timestamp,
            n_intervals: int) -> ForecastSlice | None: ...
```

- Window position `j` is the interval whose interval-end label is
  `decision_instant + (j+1) × 5 min`. Position 0 is the interval about to be
  committed.
- `get` returns `None` when no run exists at or before the decision instant,
  at the first interval of the period, or in a degraded period configured to
  fall back. The caller then commits the self-consumption action for that
  interval.
- The slice may be **shorter** than `n_intervals` (near the start of history
  or the end of the period). The caller shortens its window to match.
- `ForecastSlice` rejects an empty window, non-finite values and mismatched
  array shapes. `infill_fraction` and `fine_fraction` are reported per window
  as diagnostics.
- `run_time` and `coarse_run_time` are reported separately because they
  differ systematically. A fine run exists at every decision instant, so
  `run_time` is essentially never stale. The coarse run shapes everything
  beyond the first hour and can be up to 25 minutes older.
- Arrays, not Series: `get` runs about 105,000 times per site-year, so the
  assembly path never constructs a pandas object. It works in integer
  interval positions on the model's 5-minute grid; timestamps are converted
  once, at provider construction.
- The current-interval price mode is fixed when the provider is constructed,
  not passed per call.

---

## Window assembly (`assembly.assemble_window`)

Every windowed provider runs the same four stages. Only the supply of runs
differs by source.

### 1. Select runs

Each source presents one or two `RunSet`s: a **fine** segment (5-minute
targets; P5MIN) and a **coarse** segment (30-minute targets; PREDISPATCH).
A `RunSet` holds runs in CSR form: sorted run positions, offsets, and flat
target and value arrays. `latest_at(pos)` binary-searches for the latest run
labelled at or before `pos`. A run labelled later can never be selected, which
enforces causality structurally. A missing run simply yields the previous one
(carry-forward), and no staleness gate is applied. If the freshest run is old,
that is still the forecast the controller had, and substituting
self-consumption would model a different controller. `RunSet.from_long`
rejects duplicate (run, target) pairs, which would mean the store's
`INTERVENTION == 0` filter did not run.

The selection position is the decision instant plus the mode allowance (see
Current-interval price modes).

### 2. Splice

The coarse run alone spans the whole window from the decision instant: a run
labelled `R ≤ decision instant` covers from `R` to its horizon end. The fine
segment is a higher-resolution overlay on the first hour. A missing fine run
therefore lowers resolution without invalidating the forecast.

A 30-minute target labelled `T` is expanded flat over the six intervals
labelled `T−25 min … T` (interval-end convention, confirmed against the data).
Where the segments overlap, the fine segment wins, and `is_fine` records exactly the
positions it supplied. That boundary moves with the mode allowance and with
fine-run staleness, which is why it is recorded rather than assumed to be the
first twelve positions. Flat expansion leaves the LP indifferent to placement
within a half hour, but only in the tail, because the committed interval is
always in the fine segment.

Targets before the decision instant are dropped. The target labelled exactly
at the decision instant (the fine run's lead-0 row, an interval already
committed) is not part of the window. It is kept for infill.

### 3. Infill

Beyond forecast coverage, prices are extrapolated on the assumption that
tomorrow looks like today. With lag `L = infill_lag_hours` in intervals, in one
forward pass:

```
for each unfilled position j:
    s = j − L
    if s ≥ 0:  value[j] = value[s]                 (already filled, since L > 0)
    else:      value[j] = known_history(k + s)
    is_infill[j] = True
```

Past is not the same as published: the settled price for the interval ending
at the decision instant is not available at that instant. Known history is
therefore:

| Source interval | Value |
|---|---|
| ending at or before `decision_instant − 5 min` | settled actual |
| ending exactly at `decision_instant` | the fine run's lead-0 estimate, kept from stage 2 |

With `L = 288` and a window longer than 288, exactly one position (`j = 287`)
uses the lead-0 value. When history does not reach back far enough, the window
is truncated at the first position that cannot be filled.

### 4. Assert

The selected run must lie at or before the permitted selection position, and
every returned value must be finite. Both are assertions; a failure is a
provider bug.

**No price clipping.** Bounding forecasts to the market range was rejected.
The cap changes each financial year, which would put year-indexed knowledge
into a path run 105,000 times per site-year. It is also unnecessary: the
sources respect the caps, and infill only copies values already present. Range
checking happens once, at store ingest (`predispatch_data_spec.md`, using
`model/core/market_prices.py`). The price-ordering proof in
`prices.build_price_stack_offsets` covers the whole market range, so it is
unaffected.

---

## Current-interval price modes

`RollingConfig.current_interval_price` sets what the committed interval
(position 0) is priced at. Three coherent information sets are available within
interval k, ordered by when they become actionable:

| Mode | Available | Position 0 | Positions 1+ | Allowance |
|---|---|---|---|---|
| `forecast` (default) | at the interval start | 5-minute-ahead forecast from the run labelled at the decision instant | same run | 0 |
| `price_known` | about 9 s in | settled dispatch price for k | run labelled at the decision instant | 0 |
| `price_and_next` | about 39 s in | settled dispatch price for k | run labelled one interval later | 1 |
| `next_forecast` | about 39 s in | that later run's lead-0 estimate | run labelled one interval later | 1 |

The timings come from measurement:

- AEMO writes the dispatch price for an interval about 4.85 minutes before
  the interval ends, so a controller has it with almost the whole interval
  left.
- P5MIN publishes about 4.35 minutes before its label (3.12 minutes at worst
  observed), so the run labelled one interval later arrives inside interval
  k. That is the one-interval allowance.
- The allowance is a named per-mode exception in `mode_allowance_intervals`,
  not slack in the general rule.

`price_known` and `price_and_next` are what a real controller can do.
`next_forecast` is kept for decomposition only. It prices k at the P5MIN lead-0
estimate even though the settled price has been available for 30 seconds.
P5MIN approximates dispatch, and its lead-0 row is measurably worse than the
published price (MAE $70/MWh, $1,707/MWh on spikes). With the forward window
held equal, `next_forecast` against `price_and_next` isolates that
approximation cost.

Providers implement the modes the same way: stage 1 selects at the decision
instant plus `mode_allowance_intervals(mode)`. For
`uses_actual_for_committed(mode)`, position 0 is overwritten with the actual
price for k. The mode belongs in the provider because only the provider knows
its source's publication lag. If the rolling loop chose the later run, the
causality assertion could not tell the legitimate case from a leak.

---

## Causality

The provider owns causality; the rolling loop checks `run_time` but never
computes what was available.

Pre-dispatch run labels are used as published. Both tables publish **before**
their label (P5MIN by 4.35 minutes on average, PREDISPATCH by 28.39), so using
a run from its label onward is conservative. Subtracting the observed lead to
"correct" a label to its publication time would hand the battery information
minutes before it existed, and providers must not do it.

The permitted run label is at or before `decision_instant` for `forecast` and
`price_known`, and at or before `decision_instant + 5 min` for
`price_and_next` and `next_forecast`.

The committed interval's settled price under `price_known`/`price_and_next`
is not a causality exception: it is published about 9 seconds into the
interval.

---

## Providers

Built by `registry.build_provider(name, prices_mwh, ...)` from
`RollingConfig.forecast_source`. These names map onto the measurement ladder in
`rolling_optimisation_spec.md`:

| `forecast_source` | Provider | Ladder rung |
|---|---|---|
| `perfect` | `PerfectForecastProvider` | correctness gate; perfect price foresight over the window |
| `truncated_perfect_5min` | `TruncatedPerfectForecastProvider(coarse_resolution='5min')` | 2a |
| `truncated_perfect` | `TruncatedPerfectForecastProvider(coarse_resolution='30min')` | 2b |
| `predispatch` | `PredispatchForecastProvider` | 3, real forecast error |

`registry.region_for_state` maps a state to its NEM region (`'SA'` →
`'SA1'`).

### `PerfectForecastProvider`

Returns actual RRP from position 0 onward, with no infill and a horizon of the
whole remaining period. It does not go through `assemble_window` and so
bypasses the causality assertion by construction.

At a horizon covering the whole period it is the correctness gate: every
window pins the same global terminal SOC, so the committed sequence must
reproduce `perfect_foresight_lp`. At a finite horizon it is perfect price
foresight within the window only, a legitimate reference that isolates the
cost of the rolling window itself. `registry.is_reportable(name,
horizon_hours, period_hours)` is `False` only for `perfect` at a horizon
covering the whole period, and errs toward `False` when the horizons are not
known.

### `TruncatedPerfectForecastProvider`

Actual prices restricted to pre-dispatch's availability structure: a
P5MIN-like fine run every interval with 12 targets (leads 0–55 min), and a
PREDISPATCH-like coarse run every 30 minutes whose horizon follows the
sawtooth:

```
R = the 30-minute boundary run
horizon_end(R) = 04:00 on R.date + 2 days   if R.hour ≥ 13
                 04:00 on R.date + 1 day    otherwise
```

The closed form matched all 1,345 February 2026 PREDISPATCH runs. The 13:00
and 04:00 boundaries are market time, and the model index is already market
time (`Etc/GMT-10`), so `core/localtime.py` is not involved.

Forecast error is zero, so the gap to perfect foresight measures horizon
truncation, terminal-pin drift and, with `'30min'`, coarse-resolution loss.
With `'5min'`, the coarse segment carries actual 5-minute prices, isolating
horizon and pin effects. With `'30min'` it carries the mean of the six covered
actuals, adding the cost of only seeing half-hour averages beyond the first
hour. Running both separates the two. Neither needs forecast data.

### `PredispatchForecastProvider`

Fine segment P5MIN, coarse segment PREDISPATCH, read from the canonical store
with `pipeline.pre_dispatch.B_Clean_Predispatch.read_store`. The read starts
from one day before the period so that early decision instants find a run. The
store has already filtered `INTERVENTION == 0`, cast dtypes, checked uniqueness
and corrected the month boundaries, so the provider does no cleaning. It
converts `run_time` and `lead_minutes` to grid positions, rejecting anything
off the 5-minute grid, and runs the shared assembly.

**Coverage check (`max_staleness_hours`, default 6).** At construction the
provider measures the largest gap any decision instant would see:

- between the period start and the first run;
- between consecutive runs;
- between the last run and the period end.

It raises if the gap exceeds the limit. This is not a staleness gate, since the
provider still serves the best run it has. It catches a store that does not
span the period. That failure is silent: carried-forward runs plus
24-hour-lag infill produce a plausible result that is really yesterday's
prices. It was observed on a real cohort run where two sites sat after the
store ended and returned 98.7% infill with no error. The whole sweep of
2021-01 to 2026-06 found four isolated missing PREDISPATCH runs in about 96,000,
so 6 hours is generous. Raise it deliberately for a period containing a known
archive gap (October 2022).

**Degraded periods.** The store's `manifest.json` marks periods with no usable
forecast lead: `degraded` (a lead-0-only month, which looks complete from the
data alone) or `known_archive_gap` (October 2022, curated). The provider reads
the manifest at construction. Behaviour is set by `on_degraded_period`:

| Value | Behaviour |
|---|---|
| `fail` (default) | Raise at construction, naming the periods |
| `self_consumption` | Return `None` throughout the period |
| `actuals` | Serve actual prices, marked `is_infill` everywhere and `is_fine` nowhere |
| `infill` | No special handling. For October 2022 the complete P5MIN segment still supplies the first hour, and the rest falls to the ordinary infill. This is the recommended choice for a gap of that shape. |

The default is `fail` so the choice is conscious: a month quietly served
from actuals would be indistinguishable from a real result. An absent manifest
is not an error.

---

## Verification

- **`test_forecast_assembly.py`**
  - `RunSet` construction, ordering, duplicates and `latest_at`.
  - Coarse expansion.
  - The fine segment wins and the coarse fills beyond; coarse-only windows.
  - The lead-0 row is dropped from the window but used for infill.
  - Carry-forward over a missing run.
  - `is_fine` bookkeeping, including under the later run and a stale fine run.
  - Coarse and fine run positions are reported separately.
  - Infill at depth 1 and 2, known history, and truncation when history
    runs out.
  - Mode allowances, and `forecast` cannot reach the later run.
  - Argument validation.
- **`test_forecast_perfect.py`**
  - `ForecastSlice` validation and fractions.
  - The perfect provider's values, truncation and provenance.
  - The truncated provider's sawtooth: maximum and minimum horizon, and the
    step at 13:00.
  - Infill behaviour and its daily rhythm; 5-minute versus 30-minute coarse
    segments (block means).
  - The splice boundary.
  - Every mode's position-0 price and run selection; the three coherent modes
    form an information ladder.
  - `run_time` is never later than permitted.
- **`test_forecast_predispatch.py`**, against a synthetic store:
  - Loading, a missing store, and the horizon.
  - Causality of run labels; fine and coarse provenance.
  - Modes; infill only beyond coverage.
  - The infill share at an 18-hour horizon is small.
  - All four degraded policies, including a known archive gap without lead-0
    degradation.
  - An absent or healthy manifest.
  - The coverage guard: a store ending early or starting late, the error
    message, an override, and an isolated missing run.
  - `price_known` beats `forecast` on the committed interval.
- **`test_forecast_registry.py`**
  - Construction by name; unknown names.
  - `predispatch` requires a store and region.
  - `is_reportable`; region mapping.

---

## Known limitations

- **24-hour infill lag.** Diurnal persistence carries the daily shape, but it
  is weakest across weekday/weekend boundaries and weather changes. The lag is
  configurable (`infill_lag_hours`); a 7-day lag or a blend is the natural
  alternative and has not been measured.
- **Amber forecasts are not implemented.** See Amber providers (not built).

---

## Amber providers (not built)

Two further providers are designed against the Amber price capture store (see
the Amber capture system's own spec) but not implemented, and are not in
`FORECAST_SOURCES`. Each would supply stage 1 only and reuse the shared
assembly.

### `AmberSpotForecastProvider`

Reads `spotPerKwh` from the capture store, converted from c/kWh to $/MWh by
multiplying by 10.

It is not known whether this field is AEMO pre-dispatch passed straight
through. If it is, this provider is a second, independent copy of the
pre-dispatch signal and a cross-check on the store. If it is not, it is a
distinct forecast worth comparing.

Capture rows are timestamped by `capture_nemTime`, which is market time. The
capture fires a few seconds after each 5-minute boundary, so it contains the
update published during the preceding cycle, which makes the capture time a
conservative availability marker. Two checks are needed before the provider
can be trusted:

- Confirm that marker against the capture store.
- Do a separate timing analysis for the modes with a one-interval allowance,
  because P5MIN's 4.35-minute publication lead does not carry over to Amber.

### `AmberAdvancedForecastProvider`

Reads `advancedPrice_predicted`, Amber's own model output and the field the
capture spec identifies as driving control decisions.

It is retail-inclusive c/kWh, split across the `general` and `feedIn`
channels, and stacked onto one specific customer's tariff. It must be
inverted back to a wholesale equivalent: feeding one customer's stacked price
into a model of a different tariff is incoherent, not merely approximate.

The inversion is not designed. It is more than differencing two captured
fields, because loss factors apply to the wholesale price and the import and
export loss factors usually differ, so the two channels need
direction-specific treatment.

### Open questions

1. Whether `spotPerKwh` is pass-through pre-dispatch. This decides whether
   `AmberSpotForecastProvider` is an independent source or a cross-check.
2. The correct availability marker for the capture, and the timing analysis
   for the one-interval allowance modes.
3. How to invert `advancedPrice_predicted` to wholesale with
   direction-specific loss factors.
