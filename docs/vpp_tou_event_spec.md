# vpp_tou_event — Specification

## Purpose and scope

`vpp_tou_event` models a battery enrolled in a VPP where the household
optimises against its TOU plan between events and the retailer takes over
during events to dispatch for its own benefit. Energy Australia Battery Ease
is a contract of this kind: the household is paid a flat monthly
participation credit rather than a per-kWh reward, and the retailer
dispatches without warning.

It is not a separate module. The coordination type runs `vpp_sc_event` with
`gap_strategy='tou_advanced'`. The event LP, natural-flow bounds, minimum
export price, no-grid-charge windows, output assembly and `VppTraces` are all
as described in `vpp_sc_event_spec.md`. This document covers only what differs:
how each between-event gap is filled.

---

## Between-event gaps (`vpp_sc_event._run_tou_gap`)

Each gap `[gap_start, gap_end)` — including the tail after the last event —
is filled by `perfect_foresight_lp_advanced` against the household's
advanced price vectors (`build_lp_price_vectors_advanced` on the retail
tariff), so block tiers, zero-import rewards and critical rates all apply.

**Horizon.** The LP runs over `[gap_start, min(win_end + 24 h, T))`: the gap,
the following event window and 24 hours beyond it. For the tail the horizon is
`[gap_start, T)`. Without the look-ahead the LP would drain the battery at the
end of the gap, because stored energy has no value at the horizon end. Across
the event window and look-ahead the LP sees only household TOU prices. It has
no knowledge of the event, just as the household has no warning in practice.

**Terminal SOC.** `run_sc_with_battery` runs over the same horizon from the
gap entry SOC, and its final SOC becomes `soc_final_kwh`. It is always
reachable, so the LP stays feasible. It stands in for the value of stored energy
at the horizon end without forcing a particular trajectory.

**Output.** Only the gap part `[gap_start, gap_end)` of the solution is kept;
the event and look-ahead parts are discarded. The gap's final SOC is the event
LP's entry SOC. Gap segments are assembled by `_append_tou_segment`, which
keeps `bat_to_grid` (the TOU LP exports from the battery, for example into
an evening feed-in peak) and records all flow as natural.

**Block tiers across a mid-day start.** Tiers reset at local midnight. When
a horizon starts partway through a local day, the kWh already imported and
exported in each block window earlier that day are summed from the
assembled segments (gap and event alike), over the effective masks, so flow
that was critical on the window's side by the settled price is left out. The sums are passed as
`first_day_block_import_used`/`first_day_block_export_used`, which reduce
day 0's tier capacity (see `perfect_foresight_advanced_spec.md`). Tiers with
weekly or monthly resets would need a longer accumulation; no modelled plan
has them.

**Export-order nudge.** `ScenarioConfig.apply_export_order_nudge` is passed to
every gap LP (see `perfect_foresight_advanced_spec.md`). It is off by default
for bulk runs; the viz script turns it on.

**Natural/forced split.** The event bounds still come from a
self-consumption run through the gap, not from the TOU trajectory. The
counterfactual for "what the battery would have done without the operator"
is self-consumption, whatever the household does between events. The event LP
nevertheless enters at the TOU gap's exit SOC. See `vpp_sc_event_spec.md`,
Feasibility and fallback.

---

## Billing

`compute_bill` on the full-year traces with the household retail tariff and
`vpp_traces`. For plans such as EA Battery Ease:

- the participation credit is a negative daily charge in the tariff (e.g.
  `daily_charges_c_per_day: vpp: -49.315`);
- `exp_reward_c_per_kwh`/`imp_reward_c_per_kwh` are nominal (0.0001 c/kWh),
  so the natural/forced split is used for accounting, not payment.

Where a plan does pay per kWh of forced flow, `compute_bill` credits it to
`retail_other_credits` as for `vpp_sc_event`.

---

## Configuration and orchestration

`coordination_type: vpp_tou_event` routes to `_run_vpp_sc_event` with
`gap_strategy='tou_advanced'`. The scenario and tariff requirements are those of
`vpp_sc_event` (see `vpp_sc_event_spec.md`). In addition, the runner builds
the household vectors with `build_lp_price_vectors_advanced` and passes them to
the worker along with `retail_tariff`, `prices_mwh`, `state` and
`apply_export_order_nudge`. Example: `model/configs/example_vpp_tou_event.yaml`.

---

## Verification

There are no component-level tests of the TOU gap path. `test_orchestrate_vpp.py`
runs `vpp_tou_event` through the orchestrator for monthly-bill wiring and
minimum-export-price validation. `test_vpp_sc_event.py` covers the shared event
machinery with self-consumption gaps.

Dev tools:

- `model/dev/check_vpp_tou_event.py SITE_ID YEAR` runs a real site and
  reports the same checks as `check_vpp_sc_event`, plus a pre-event SOC
  analysis.
- `model/dev/viz_vpp_tou_event.py` is the interactive view (nudge on).

---

## Known limitations

- **The gap LP ignores the site export limit.** `_run_tou_gap` does not pass
  `export_limit_kwh_per_interval` to `perfect_foresight_lp_advanced`, so gap
  dispatch is not export-limited. Event windows are. Fix path: thread the
  limit through `_run_tou_gap`.
- **No event anticipation.** The household's LP may discharge for TOU value
  just before a high-value event, leaving the retailer less to dispatch. This
  matches the contract (no warning), so it is behaviour, not error.
- **SC-derived terminal SOC.** It is a reasonable but imperfect proxy for
  the value of energy at the horizon end. The 24-hour look-ahead limits its
  influence on the kept gap dispatch.
- **Event bounds and entry SOC come from different trajectories**, so the
  event LP is not guaranteed feasible, and its fallback is not SOC-continuous
  (see `vpp_sc_event_spec.md`).
- **No component-level tests** of gap dispatch, block-tier carry-over across a
  mid-day start, or agreement with a whole-period TOU LP.
