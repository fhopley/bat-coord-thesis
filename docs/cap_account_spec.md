# Capped-Volatility Retail Offers -- Model Spec

Three retail offers are specified here. Cohort Hedging and Retailer Pool
Retention cap the wholesale price a household sees and differ only in who
absorbs the resulting imbalance. Tesla Electric removes wholesale exposure
from imports altogether. In the code, options and output names the three
are `cohort`, `retailer` and `dynamic`, and some chart labels use A, B and C.

- **Cohort Hedging.** The imbalance is settled within the
  participating cohort: uncollected costs are covered from unpaid export
  credits, and the retailer absorbs only what the cohort cannot cover.
  The retailer never profits from the mechanism (Part II, Section 7).
- **Retailer Pool Retention.** The retailer absorbs the whole
  imbalance and retains the surplus, less a fraction shared back to each
  household on its own above-cap export value. No cohort pooling; a
  household's outcome depends only on its own flows.
- **Tesla Electric.** Imports are billed at a fixed
  time-of-use rate that bundles energy and network charges; exports are
  credited a fixed share of their real-time wholesale value, floored at
  zero. The retailer carries all import-side wholesale and network cost,
  the network export tariff, and the rest of export value, including all
  of it when the spot price is negative. No cap threshold, no settlement
  period, no cap account.
  Modelled on Tesla Electric's Dynamic plan (Part V).

All three bill already-computed, already-verified dispatch flows, and none
changes dispatch (Section 2). Cohort Hedging and Retailer Pool Retention re-bill the Amber run's flows.
Tesla Electric bills the plan's own modelled run and compares it with the Amber
run (Section 16).

---

# Part I -- Common machinery

## 1. Purpose

Amber's wholesale-passthrough arrangement exposes households to the full
range of NEM spot prices (-$1,000/MWh to $20,300/MWh). This is attractive
for arbitrage but can be off-putting due to perceived risk. Both offers cap
that exposure and differ in how the resulting shortfall or surplus is
allocated. The model quantifies, for a single cohort (one state, year,
battery size) and a single price threshold, how each offer changes
individual household bills relative to the verified uncapped baseline, and
how much residual risk the retailer bears.

Both mechanisms settle realised outcomes only. Neither requires a forward
view on volatility, which distinguishes them from a fixed-price offer, a
bill-outcome guarantee, or a subscription priced to expected cost -- all of
which need a volatility forecast that is currently hard to form (battery
penetration has suppressed volatility through 2026, and the coal-exit
trajectory is unresolved). The corresponding limitation is that neither
offer gives the customer bill certainty at the point of signing: the cap
bounds the price, not the bill.

## 2. Scope and standing assumptions

- **Data source**: Amber wholesale tariff only, any state. The default is
  the completed `full_run_rolling_amber` run, whose
  `rolling_perfect_foresight_lp` coordination implements imperfect foresight
  of price, load and solar generation. `full_run_optimised_amber`
  (`perfect_foresight`) stays selectable through `--run-config`
  (Section 12) as an upper-bound comparison. `--state` is a normal CLI
  flag; all states and years reconcile cleanly under
  `full_run_optimised_amber` (`bill_reconciliation.py`).
  **TO CONFIRM**: whether the same holds for `full_run_rolling_amber`.
  Section 5's per-site reconciliation check runs on whichever run is
  selected and warns on any mismatch.
- **The retailer holds dispatch rights.** Cohort Hedging and Retailer Pool Retention make the cap
  conditional on the retailer controlling the battery; customers who
  prefer full exposure stay on unmodified wholesale passthrough. The
  retailer optimises against the real, uncapped price, so the run's
  verified flows are the correct dispatch under both offers. This is why
  re-billing without re-optimising is valid here, rather than a deferral.
  Rolling imperfect foresight is the more defensible representation of
  what a retailer could actually achieve, hence the default. Tesla Electric's
  plan dispatches against household-side prices rather than the real
  price, so it takes its flows from the plan's own modelled run and sets
  them against the Amber run's (Section 16).
- **Wholesale layer only (Cohort Hedging and Retailer Pool Retention).** Network charges and Amber's flat retail adders
  (`usage_other_c_per_kwh`, `feedin_other_c_per_kwh`) are untouched, in
  real time and at settlement. Only `wholesale_usage_cost` and
  `wholesale_feedin_credit` are affected. A fixed subscription fee sits
  outside this layer and enters no household output: at a given fee it
  shifts every household's bill by the same amount and changes nothing
  between offers. It is not outside the retailer's position, where it is
  the whole of the revenue in the uncapped baseline (Section 14). Tesla Electric
  replaces the network and retail layers as well, so it is compared on the
  full bill (Section 16).
- **One cohort at a time.** A cohort is every site in a single
  `(state, year, battery_size)` scenario. No pooling across battery sizes.
- **`x` is a threshold on the raw wholesale spot price** ($/MWh), compared
  before any loss-factor scaling. Single value per run.
- **No floor.** Negative prices pass through unchanged on both import and
  export; the cap only binds when price exceeds `x`.
- **Household dollars, GST on spot imports.** Every amount in this model --
  bills, cap account, settlement, retailer position -- is in household bill
  dollars. Where the run's retail tariff sets `gst_on_wholesale_import`
  (the Amber tariffs), the household pays GST at `GST_RATE` (0.10,
  `model/components/billing.py`) on the spot energy charge of every
  import, negative prices included, and none on exports (`model_spec.md`,
  The settlement model). `g` below is that rate, or 0 where the tariff
  does not set it. The `$25/month` subscription the retailer position is
  set against also includes GST. The run's `bills.csv` must carry the GST
  (`patch_amber_gst.py` for runs billed before the rule): the script
  checks per site that `retail_usage_cost - g * wholesale_usage_cost`
  equals the adder stack times the site's imports, to `$0.01`, and stops
  otherwise.

## 3. Inputs

- `flows.parquet` for the target scenario (`data/output/runs/<run>/
  scenarios/<state>_<year>_<battery_name>/flows.parquet`), split into
  `import_kwh`/`export_kwh` per site per interval by clipping the net
  series at zero, per `bill_reconciliation.py`'s method. A site cannot
  both import and export in the same interval, which Part III, Section 10
  relies on.
- The scenario's `bills.csv`: the baseline for comparison, and the source
  of the network/retail layer amounts that pass through untouched.
- The NEM price series for the target state/year
  (`{year}_nem_prices.parquet`, column `{state}1_rrp`), loaded via
  `_load_prices`. Its index must equal `flows.parquet`'s index exactly;
  a mismatch raises rather than silently reindexing.
- `import_lf`, `export_lf`: static scalars from the scenario's network
  tariff, read from the `NetworkTariff` object via `load_network_tariff`.
  Loss factors vary by state/tariff, not by time.
- `x`: the price threshold, $/MWh. Cohort Hedging and Retailer Pool Retention only.
- Tesla Electric only: the plan's retail tariff, a `type: mixed` YAML
  (Section 18), and the run's own Amber retail tariff, whose adders and
  subscription define the comparator (Section 17).
- `full_run_self_consumption_amber`'s `bills.csv` for the target
  state/year/battery size (`household_total` only), used to size the
  offer's impact against each site's savings from price-aware coordination
  specifically. Compared against `self_consumption` (same battery
  hardware, uncoordinated dispatch) rather than a no-battery baseline,
  which would conflate the value of having a battery with the value of
  coordinating it. Absence degrades one output column to all-`NaN` with a
  warning rather than failing the run.
- `EXCLUDED_SITES = {'S0588', 'W0164'}`, dropped from `flows.parquet`'s
  columns before any computation so they are absent from every downstream
  output.

  **The criterion is annual gross household load too large to be
  residential**, taken from `avg_daily_gross_load_kwh` in the EDP site
  metadata. Against a population median of 7,200 kWh a year and a maximum
  of 26,300 among all retained sites, `S0588` runs 89,800-107,700 and
  `W0164` 107,400-114,300 -- twelve to sixteen times the median, and around
  310 kWh a day at the top, which is not a dwelling. `S0588` also matches
  `financial_analysis.py`'s own exclusion and is the ~$27k baseline site
  that dominated early "biggest winner" findings.

  `S0241` was previously excluded with no recorded reason and does not meet
  this test: at 27,300 kWh it sits inside the range of retained sites and
  below three of them. It is retained.

  A stated criterion applied uniformly is used rather than a case-by-case
  list, so that the question of what else was removed has an answer.

  **The high-load sites that remain are retained deliberately.** Six sites
  (`S0176`, `S0172`, `S0155`, `S0579`, `S0308`, `W0179`) span 10,300-30,300
  kWh a year, sit on a continuous tail of the population rather than
  detached from it, and have unremarkable PV -- 4,000-15,300 kWh. They are
  the only sites in 1,428 site-years that are net above-cap importers, and
  so the only ones the import cap protects. Removing them would leave the
  import side of both offers with no beneficiaries and make the mechanism's
  purpose unevidenced. They are also the closest thing in a uniform-battery
  cohort to what a mixed battery-size cohort would produce (Section 14).

  **Defined once**, in `analysis/scripts/site_exclusions.py`, and imported
  by this script, `financial_analysis.py` and `flow_exposure_analysis.py`.
  It previously lived separately in all three and drifted: the two analysis
  scripts carried `{'S0588'}` while this one carried three sites, so
  `W0164` was present in those results and absent from these. In VIC that
  mattered -- `W0164` is 17-18x the cohort median load and made up 25% to
  52% of total cohort load depending on the year. The shared module is kept
  free of dependencies rather than living in `financial_analysis.py`, which
  sets global matplotlib styling at import time.

## 4. Settlement periods

Every quantity in Sections 5-6 is computed per interval and then grouped
into a **settlement period**. The settlement period is a run parameter,
not a fixed month:

- `interval` -- each 5-minute interval settles on its own.
- `daily` -- local calendar day.
- `weekly` -- 7-day blocks anchored at the first day of each calendar
  month, with a ragged final block of 1-7 days.
- `monthly` -- local calendar month (the default).

Periods never span a calendar month, because bills are issued monthly and
no balance is carried across a bill. This rules out ISO weeks, hence the
month-anchored blocks above. Periods longer than a month are out of scope
for the same reason.

Period keys are derived from `local_wall_clock(index, state,
interval_hours=5/60)`, not the AEST-indexed timestamps, so periods follow
billing semantics through DST. Because DST pushes local time forward
relative to AEST, the last ~12 intervals of the AEST-indexed year fall on
local 1 January of the following year. Any interval whose local date falls
outside the target year is folded into the final period of the target
year, rather than forming a near-empty extra period.

**The settlement period sets both the self-hedging window and the cohort
pooling window.** A site's own above-cap imports and exports net against
each other within the period before any cohort-level step, and the cohort
pool is drawn from the same period: whatever schedule a site settles its
own hedging on is the schedule on which the cohort offsets it.

Household billing remains monthly regardless. Settlement amounts are
computed per period and summed into calendar months for the bill, so every
monthly output is well-defined at any settlement period.

## 5. Real-time capped billing

For every interval `t`, with `p_eff[t] = min(real_price[t], x)`:

```
wholesale_usage_cost'[t]    =   import_kwh[t] * p_eff[t] * import_lf * (1 + g)
wholesale_feedin_credit'[t] = -(export_kwh[t] * p_eff[t] * export_lf)
```

Sign convention matches `SiteBill`: cost positive, credit negative. Both
import and export are capped in real time under both offers; the offers
differ only in what is returned at settlement. Below-threshold intervals
are billed exactly as in the verified baseline.

The recomputed annual baseline (the same formula with
`p_eff[t] = real_price[t]` and `g = 0`, since `bills.csv` records the
wholesale layer without GST) is checked against `bills.csv`'s
`wholesale_usage_cost + wholesale_feedin_credit` per site, to `$0.01`
absolute or `1e-6` relative tolerance, matching `bill_reconciliation.py`.
A mismatch warns rather than aborts, and indicates a price, loss-factor or
unit bug in this script rather than in the verified baseline.

## 6. Cap account

The cap account balance is the wholesale-layer amount that Section 5's
capped billing left uncollected or unpaid, relative to the real price:

```
cap_delta[site, t] = (real_price[t] - x)+ * ( import_kwh[site,t] * import_lf * (1 + g)
                                            - export_kwh[site,t] * export_lf )
```

An above-cap import kWh therefore counts `(1 + g)` times an above-cap
export kWh at the same price when they net within a period: it is what the
household would have paid, GST included, against what it would have been
credited.

zero where `real_price[t] <= x`. A net above-cap **importer** accrues a
positive `cap_delta`: real billing would have cost them more than the cap
charged, so an uncollected cost sits against their name. A net above-cap
**exporter** accrues a negative `cap_delta`: an unpaid credit.

`cap_delta` is summed over the settlement period to give `S(site, period)`.
Because this is a single sum, a site that is sometimes an above-cap
importer and sometimes an above-cap exporter within the period has these
netted before any offer-specific step -- self-hedging needs no separate
mechanism.

Nothing is drawn from or added to a site's bill at this point. Section 5's
capped billing already reflects what the site pays in real time; the cap
account is a parallel ledger settled per Part II or Part III.

**Retailer's position from the mechanism**, before any offer-specific
allocation, is `-sum over sites of S(site, period)`: the retailer loses on
uncollected costs and gains on unpaid credits.

---

# Part II -- Cohort Hedging

## 7. Settlement

Per cohort, per settlement period:

```
total_cost_uncollected = sum( S(site) for site where S(site) > 0 )
total_credit_unpaid    = sum( -S(site) for site where S(site) < 0 )

p = min(1, total_cost_uncollected / total_credit_unpaid)

adjustment[site] = min(S(site), 0) * (1 - p)

retailer_net_cost = max(0, total_cost_uncollected - total_credit_unpaid)
```

A site with `S(site) > 0` gets `adjustment = 0`: the uncollected cost is
forgiven outright, never billed. A site with `S(site) < 0` gets an
additional credit equal to a `(1 - p)` fraction of what it was underpaid in
real time. The remaining `p` fraction of every unpaid credit covers the
cohort's uncollected costs. `retailer_net_cost` is what remains once every
unpaid credit has been drawn on to the fullest extent. It is the retailer's
whole position under Cohort Hedging, not just its shortfall: the retailer's gross
loss on uncollected costs, less what it retains from unpaid credits
(`p * total_credit_unpaid`), is `total_cost_uncollected - p *
total_credit_unpaid`, which equals `max(0, cost - credit)` for either value
of `p`. Retailer Pool Retention uses the same column name (Section 10), where it is not
floored at zero.

Where `total_credit_unpaid == 0`, `p` is held at its basic-model default of
`1.0`: there is no credit-side site for it to apply to, so `adjustment` is
all-zero regardless, and the whole `total_cost_uncollected` falls to the
retailer.

**Household's final wholesale-layer amount** = Section 5's capped real-time
total over the period, plus `adjustment[site]`, summed into calendar months.

**Household's final bill** = baseline `household_total`, minus the
baseline's own `wholesale_usage_cost + wholesale_feedin_credit`, plus the
capped/settled wholesale-layer amount. Every other bill component is
identical to the verified baseline.

**The retailer is value-neutral whenever `p <= 1`.** Its net wholesale
outlay under the cap equals its outlay under real billing; exposure appears
only when the cohort is in aggregate an above-cap importer for the period
(`total_cost_uncollected > total_credit_unpaid`). The retailer never books
a gain from the mechanism -- any cohort surplus flows back to credit-side
sites through `p < 1`.

**Shortening the settlement period weakly increases `retailer_net_cost`.**
`sum over sub-periods of max(0, cost - credit)` is greater than or equal to
`max(0, cost - credit)` over their union, strictly so whenever some
sub-periods are cohort-net-importing and others cohort-net-exporting. The
same applies to a site's own self-hedging: a shorter period gives fewer
opportunities for a site's above-cap exports to offset its own above-cap
imports. Sweeping the settlement period therefore prices the cost of faster
settlement.

## 8. Hedging surcharge

The surcharge does two things: it protects the retailer, and it transfers
value from importers making use of the cap back to the exporters providing
the hedge. Without it, a site that is always an above-cap importer
contributes nothing toward the pool that protects it -- its uncollected
cost is forgiven, funded by whichever sites happen to be exporting at the
time. The surcharge spreads a contribution across every importing site,
every interval, all year.

Because the transfer is the point and not only a by-product, the rate is a
design parameter for how much value moves from importers to exporters, not
merely a premium sized to expected shortfall. Revenue exceeding
`total_cost_uncollected` is returned to credit-side sites as `bonus_rate`
rather than retained, so over-collection widens the transfer rather than
accruing to the retailer.

### Real-time billing

```
surcharge[site, t] = surcharge_rate_per_kwh * import_kwh[site, t]
```

default $0.01/kWh, GST-inclusive like every customer-facing rate, applied
to every import regardless of price, on top of whatever Section 5 charges. A genuine new cost, visible in real time, not a
reallocation.

### Settlement

**`R` pools over the calendar month whatever the settlement period.** The
surcharge gathers value across every import interval in order to fund
shortfalls arising in rare events; pooling it over a shorter period would
set a fraction of that revenue against a single period's uncollected cost,
and at `interval` settlement it would be meaningless.

```
R = surcharge_rate_per_kwh * sum(import_kwh[site,t] for all sites, all
                                 intervals in the calendar month)
```

Settlement therefore runs in two stages whenever the settlement period is
shorter than a month. Stage 1 is Section 7's cap account settlement, per
period `d`, with no surcharge:

```
p[d]          = min(1, cost[d] / credit[d])           # 1 if credit[d] == 0
shortfall[d]  = max(0, cost[d] - credit[d])
skim[d, site] = -min(S(site, d), 0) * p[d]            # >= 0
```

`skim` is what each credit-side site contributed to covering the cohort's
uncollected costs that period.

Stage 2 applies `R` at month end in priority order -- first to reduce the
retailer's position, then to refund the skim, then as a bonus to
credit-side sites above their full original credit:

```
total_shortfall = sum over d of shortfall[d]
total_skim      = sum over d, sites of skim[d, site]
total_credit    = sum over d of credit[d]

retailer_net_cost = max(0, total_shortfall - R)
R_rem             = max(0, R - total_shortfall)
refund_frac       = min(1, R_rem / total_skim)        # 0 if total_skim == 0
bonus_rate        = max(0, R_rem - total_skim) / total_credit
                                                      # 0 if total_credit == 0

adjustment[site] = sum over d of min(S(site,d), 0) * (1 - p[d])
                   - refund_frac * sum over d of skim[d, site]
                   - bonus_rate  * sum over d of -min(S(site,d), 0)
```

`adjustment[site]` is negative, a credit; a site with `S(site, d) >= 0` in
every period receives nothing beyond the forgiveness it already has.

At `--settlement-period monthly` this reduces to the single-stage form
`p = clip(0, 1, (cost - R) / credit)`,
`bonus_rate = max(0, R - cost) / credit`,
`adjustment[site] = S(site) * (1 - p + bonus_rate)`, and with
`surcharge_rate_per_kwh = 0` it reduces further to Section 7 unmodified.
Both reductions are regression checks on the implementation.

Where `credit[d] == 0` there is no credit-side site to draw on and the
whole of `cost[d]` falls to the retailer. Where `total_credit == 0` for the
month, surcharge revenue beyond what reduces `retailer_net_cost` has no
pool to distribute into and is retained by the retailer for that month.

Refunding that stranded revenue to the sites that paid it was considered
and rejected: the amounts are small (in NSW 2022 at `x = $300/MWh` and
`surcharge_rate_per_kwh = 0.01`, January was the only such month, at
`$467` across 206 sites), and a premium that is sometimes refunded is
harder to explain to a customer than one that is not.

`S(site)` is unchanged by the surcharge: it is a separate cohort-level pool
entering only at settlement, not a per-site ledger entry.

## 9. Worked example

`x = $300/MWh = $0.30/kWh`. One interval at `$1,000/MWh = $1.00/kWh`, loss
factors omitted. Site A imports 2 kWh, Site B exports 3 kWh; no other sites
or intervals in the period.

- Real-time: A pays `0.30 * 2 = $0.60`; real billing would have cost
  `$2.00`, so `cap_delta[A] = +1.40`. B earns `0.30 * 3 = $0.90`; real
  billing would have credited `$3.00`, so `cap_delta[B] = -2.10`.
- Settlement: `p = min(1, 1.40/2.10) = 0.667`. A's adjustment is 0, final
  wholesale cost `$0.60`. B's adjustment is `-2.10 * (1-0.667) = -$0.70`,
  final wholesale amount `-$1.60`.
- Retailer: pays B `$1.60`, collects `$0.60` from A, net outlay `$1.00` --
  identical to the uncapped case (`$3.00 - $2.00`).

With `surcharge_rate_per_kwh = $0.01` and 500 kWh of cohort import in the
period (chosen to make `R` material against a one-interval example, not a
realistic volume): `R = $5.00`. Since `R > total_cost_uncollected = 1.40`,
`p = clip(0,1, (1.40-5.00)/2.10) = 0`, `surplus = 3.60`,
`bonus_rate = 3.60/2.10 = 1.714`. B's adjustment is
`-2.10 * (1 - 0 + 1.714) = -$5.70`: full original credit of `$2.10` plus a
`$3.60` bonus. `retailer_net_cost = 0`.

---

# Part III -- Retailer Pool Retention

Shares Parts I and IV with Cohort Hedging; differs only in Section 10's
settlement rule.

## 10. Settlement

No cohort pool. The retailer absorbs every uncollected cost and retains
every unpaid credit, and shares back a fraction `alpha` of each site's own
above-cap export value.

Per site, per settlement period:

```
share_base[site] = max(0, -S(site, period))
payment[site]    = -alpha * share_base[site]        # negative: a credit

adjustment[site] = payment[site]
```

A site that nets to an above-cap import position over the period receives
no payment but keeps its forgiveness, exactly as under Cohort Hedging. A site that
nets to an above-cap export position receives `alpha` of that net value
back on top of its capped real-time credit.

**Household's final bill** is formed exactly as in Section 7, with this
`adjustment`.

**Retailer's net cost** for the period:

```
retailer_net_cost = total_cost_uncollected - total_credit_unpaid
                    + sum( -payment[site] )
```

Reported in the same `retailer_net_cost` column as Cohort Hedging (Section 7),
but not floored at zero: under Retailer Pool Retention the retailer books a gain in periods
where the cohort is a net above-cap exporter, which Cohort Hedging's pool
returns to households instead. The two are not directly comparable and must
not be plotted on one axis without saying so.

### Period dependence and the two share bases

Two share bases were proposed: a fraction of gross above-cap export value
(period-independent), and a fraction of net above-cap value floored at zero
(period-dependent). These are the same formula at different settlement
periods, not two mechanisms.

Because a site cannot both import and export in one interval (Section 3),
at `interval` settlement `max(0, -S(site, t))` equals
`(real_price[t] - x)+ * export_kwh[site,t] * export_lf`, which is the gross
export base exactly. So:

- **`--settlement-period interval` gives the gross-export share.** No
  netting against the site's own above-cap imports; identical at any
  reporting frequency, because there is no floor left to bind.
- **Any longer period gives the net share**, with the floor binding on
  periods where a site's above-cap imports exceed its above-cap exports.

`sum over periods of max(0, -S)` is weakly decreasing in period length, so
shortening the settlement period weakly increases the household's payment
and weakly increases the retailer's net cost. This runs in the same
direction as Cohort Hedging's shortfall result (Section 7), which lets one
settlement-period sweep serve both offers.

### Boundary cases

- `alpha = 0`: pure import insurance. The retailer retains all above-cap
  export value; the household is protected and shares nothing.
- `alpha = 1` at `interval` settlement: full export passthrough with
  import protection funded entirely by the retailer.
  `retailer_net_cost = total_cost_uncollected`, with no offset. This is the
  upper bound of the retailer's exposure across the parameter space.

`alpha` defaults to `0.5`, and is a swept parameter.

### No surcharge

Section 8's surcharge does not apply to Retailer Pool Retention: there is no cohort pool
for it to fund, so it would be a straight price increase. A fixed
subscription fee, if the offer needs one, sits outside the wholesale layer
(Section 2).

### Why a fraction rather than a fixed rate

A fixed $/kWh export rate (as in existing VPP offers, e.g. $1/kWh) makes
the retailer's marginal gain from discharging `(real_price - x - rate) *
export_lf`, creating a dead band of width `rate` above the cap in which the
retailer will not dispatch. At $1/kWh that band is $300-$1,000/MWh, which
is where most above-cap volume sits. A fraction preserves dispatch ordering
for any `alpha < 1`, since the retailer retains `(1 - alpha)` of every
marginal dollar.

The share is also computable by the customer from their own meter data,
published spot prices and published loss factors, with no reference to
cohort aggregates or the retailer's book.

---

# Part IV -- Outputs, interface, and deferred items

## 11. Outputs

Per cohort (state/year/battery size), written to a variant subfolder
(Section 12):

Cohort Hedging and Retailer Pool Retention write the same files with the same column names, in the
same order, so output from different variants concatenates without
reshaping. Columns that do not apply to an offer are written as `NaN`.
Tesla Electric writes its own file set (Section 19): it has no cap account, so
`cohort_period_summary.csv` and the cap-account columns have nothing to
hold.

**`per_site_annual.csv`** -- one row per site: `baseline_household_total`,
`capped_household_total`, `diff`, `annual_cap_delta` (sum of
`S(site, period)` before settlement), `annual_value_returned` (sum of
`-adjustment[site]`: what settlement gave back on top of capped real-time
billing, whether through Cohort Hedging's `1 - p` and bonus or Retailer Pool Retention's share),
`annual_coordination_savings` (the denominator of
`pct_of_coordination_savings`, written out so downstream can recompute the
percentage after a subscription adjustment rather than recovering it by
division), and:

- `pct_of_wholesale_export_earnings`: `diff` as a percentage of the site's
  total real (uncapped) annual export credit. Not a percentage of
  `baseline_household_total` -- with full wholesale exposure and a sizeable
  battery, many sites' annual bill is small or negative, making
  percentage-of-bill meaningless or sign-flipped for much of the cohort.
  `NaN` where annual export earnings are under `$1` in absolute value, a
  site with genuinely tiny export earnings otherwise showing an extreme
  percentage from a small absolute `diff`.
- `pct_of_coordination_savings`: `diff` as a percentage of the site's own
  annual saving from price-aware coordination -- `household_total` under
  `self_consumption` (same battery size, uncoordinated dispatch) minus this
  scenario's `baseline_household_total`. `NaN` where that saving is under
  `$1`, including sites where coordination makes things worse. Falls back
  to all-`NaN` with a warning if the `self_consumption` run is absent.
- `n_months_better_off` / `n_months_worse_off`: count of calendar months
  where the site's monthly wholesale-layer impact (`final - baseline`)
  exceeds `$0.50` in the beneficial/adverse direction. Distinguishes a site
  consistently on one side of the mechanism from one only occasionally so.

**`cohort_period_summary.csv`** -- one row per settlement period, with a
single column set across both offers, in this order:

| column | A | B | |
|---|---|---|---|
| `offer` | yes | yes | `cohort` or `retailer` |
| `settlement_period` | yes | yes | constant within a file; carried so variants concatenate |
| `month` | yes | yes | billing month of the period, so `cap_account_retailer.py` need not re-derive it from the period key |
| `n_sites` | yes | yes | constant within a file (Section 14) |
| `subscription_per_month` | yes | yes | constant within a file (Section 14) |
| `total_cost_uncollected` | yes | yes | |
| `total_credit_unpaid` | yes | yes | |
| `n_intervals_above_cap` | yes | yes | separates "no price spikes this period" from other reasons a period might show `p = 1` or `p = 0` |
| `total_value_returned` | yes | yes | `sum over sites of -adjustment[site]` |
| `retailer_cost_absorbed` | yes | yes | `total_cost_uncollected`, restated for the Section 14 identity |
| `retailer_credit_retained` | yes | yes | `p * credit` under A, `(1 - alpha) * credit` under B |
| `retailer_surcharge_retained` | yes | `NaN` | `R` less what stage 2 refunded and bonused away; written on the last period of each month only |
| `retailer_net_cost` | yes | yes | `cost_absorbed - credit_retained - surcharge_retained`, negative where the retailer gains (Section 14) |
| `p` | yes | `NaN` | |
| `bonus_rate` | yes | `NaN` | written on the last period of each month only |
| `R` | yes | `NaN` | pooled monthly, so written on the last period of each month only (Section 8) |
| `alpha` | `NaN` | yes | |

Monthly-scope columns sit on the last period row of their month and are
`NaN` elsewhere, so every column sums correctly down the file.

**`by_site_month_{metric}.csv`**, one file per metric, sites as rows and
calendar months as columns (sites outnumber months by more than an order of
magnitude, so this is the natural table shape): `baseline_wholesale_net`,
`capped_realtime_total`, `surcharge_cost`, `cap_delta`, `adjustment`,
`final_wholesale_net`. `capped_realtime_total` includes the surcharge, as
part of what the household actually pays in real time; `surcharge_cost` is
broken out separately so its own contribution is visible.

**`summary.txt`** -- total `retailer_net_cost` in dollars and as a
fraction of the cohort's total baseline wholesale cost;
per-period breakdown; `diff` mean/std/min/max and counts better/worse off;
counts of sites never better and never worse off in any month; and the
concentration summary: cumulative share of total household benefit (sum of
negative `diff`) captured by the top 1, 5 and 10 sites, ranked by most
negative `diff`.

**Charts** -- `annual_diff_histogram.png`,
`pct_of_export_earnings_histogram.png`,
`pct_of_coordination_savings_histogram.png` (skipped when the
`self_consumption` baseline is absent), and `monthly_diff_boxplot.png`, one
box per calendar month of household wholesale-layer impact.

## 12. Interface

```
python3 analysis/scripts/cap_account_analysis.py
    --project-root PATH      model/tariffs/ and run config location
                             (default: cwd)
    --data-root PATH         run outputs, site metadata, NEM prices
                             (default: cwd)
    --run-config PATH        run config YAML; resolved against
                             --project-root if relative. Default:
                             model/configs/full_run_rolling_amber.yaml.
                             Point at full_run_optimised_amber.yaml for the
                             perfect-foresight comparison. This flag both
                             selects the coordination run and supplies the
                             network tariff id (and so
                             import_lf/export_lf), which is only known via
                             the run's own config, per
                             bill_reconciliation.py's convention
    --state STATE            (default 'NSW')
    --year YEAR              (default 2022)
    --battery-name NAME      (default '20kWh')
    --threshold-per-mwh X    the cap, $/MWh (default 300)
    --offer {cohort,retailer,dynamic}
                             Cohort Hedging, Retailer Pool Retention or
                             Tesla Electric (default 'cohort')
    --settlement-period {interval,daily,weekly,monthly}
                             (default 'monthly')
    --surcharge-rate-per-kwh RATE   Cohort Hedging only, $/kWh on all imports
                             (default 0.01; 0 disables, reducing to
                             Section 7)
    --share-fraction ALPHA   Retailer Pool Retention only, fraction of above-cap export
                             value shared back (default 0.5)
    --subscription-per-month FEE
                             override the subscription read from the
                             run's retail tariff, $/month (Section 14)
    --offer-run-config PATH  Tesla Electric only, the run config of the plan's
                             own modelled run (default
                             model/configs/full_run_rolling_tesla.yaml).
                             Its flows are the plan's dispatch and its
                             retail tariff is the plan
    --dynamic-dispatch {own,rebill}
                             Tesla Electric only, whether the plan is billed on
                             its own run or on the Amber run's flows
                             (default 'own')
    --dynamic-tariff-id ID   Tesla Electric with --dynamic-dispatch rebill only,
                             the mixed retail tariff, formatted with
                             --state (default
                             '{state}_tesla_electric_mixed_2026')
    --output-dir PATH        (default: analysis/output/cap_account)
```

The run warns, but proceeds, if the resolved scenario's
`coordination_type` is not one of `rolling_perfect_foresight_lp`,
`rolling_perfect_foresight_lp_advanced`, `perfect_foresight` or
`perfect_foresight_advanced`. Tesla Electric's own run is warned on the same way.

Output goes to
`{output-dir}/{run_name}/{STATE}_{year}_{battery_name}/{variant}/`, where
`{variant}` encodes every parameter that changes the numbers:
`{offer}_x{threshold}_{period}` plus `_s{surcharge}` for Cohort Hedging or
`_a{alpha}` for Retailer Pool Retention. Tesla Electric's variant is `dynamic_{tariff}`, with
`_modelled` appended where the plan's own run supplies the dispatch
(Section 19). The run name level is the Amber run for all three offers, so
Tesla Electric's folder sits beside the cap variants it is drawn with. Without the variant level, runs at different thresholds or
surcharges would overwrite each other. The run name is the outermost level
so that rolling and perfect-foresight results never collide and
`cap_account_aggregate.py` can be pointed at one run by `--results-root`
alone, with its `{state}_{year}_{battery}` directory discovery unchanged.

## 13. Aggregating household charts across years and states

`cap_account_aggregate.py` combines the two headline household charts
(`annual_diff_histogram` and `pct_of_coordination_savings_histogram`)
across every year and state already run for one battery size. Retailer
shortfall, concentration and the monthly boxplot are not extended to this
aggregation.

**Reads already-written output, computes nothing new.** Input is every
`{state}_{year}_{battery_name}/.../per_site_annual.csv` found under
`cap_account_analysis.py`'s `--output-dir`. No flows, prices or settlement
are re-run.

**Unit of observation: the site-year, not the site.** Most sites appear in
only one or two years (participation reflects data-sharing-agreement
enrolment, not attrition tied to outcomes), so tracking households across
years is not viable. Each `(site_id, year)` pair is one independent
observation.

### Unweighted mode

Every site-year counts as 1, concatenated across every state and year found
for the requested battery size. A plain baseline, not the primary output.

### Weighted mode (default)

One flag controls PV-size, year and state weighting together, matching
`financial_analysis.py`'s principle that weighting is applied consistently
rather than as independent toggles:

```
site_weight(i, s, y) = by_state[s][band(i, s, y)] / count_in_band(s, y)
```

reusing `financial_analysis.py`'s banding pipeline directly
(`load_weights_config`, `load_site_meta`, `compute_dc_ac_ratio`,
`assign_direct_size`, `estimate_missing_sizes`,
`assign_bands_and_weights`), imported rather than duplicated. The pooled
national DC:AC ratio and pooled metadata are computed once per year across
every state and reused for each state that year, mirroring that script's
own main-loop caching.

**Year-equalisation falls out of PV weighting.** Because `by_state[s]`'s
band shares sum to 1 and `site_weight` splits each band's share evenly
across the sites in it, `site_weight` summed over a `(state, year)` is
always exactly 1, whether that year has 207 sites or 10. Concatenating each
year's PV-weighted site-years within a state therefore gives every year
equal total influence with no further adjustment. This is the opposite
correction from `flow_exposure_analysis.py`'s multi-year interval pooling
(`flow_exposure_spec.md`, Event and window exposure), which lets a year with more
qualifying price events count for more because that imbalance reflects real
market conditions. Here the site-count imbalance reflects enrolment timing.

**National weight**:

```
national_weight(i, s, y) = site_weight(i, s, y) / n_years(s) * state_mix[s]
```

`n_years(s)` rescales that state's pooled total from `n_years(s)` back to 1
before applying `state_mix`, so states combine by population share
regardless of how many years each has. A state with no `state_mix` entry is
excluded from the national pool with a warning.

Histograms are drawn with these weights passed to
`matplotlib.axes.Axes.hist(..., weights=...)`, not by pre-aggregating bins.

### Output

- `{state}_all_years/{stem}_{weighted|unweighted}.png`, per state with more
  than one year of data, using `site_weight` unscaled by `n_years` since no
  cross-state combination happens at this tier.
- `national/{stem}_{weighted|unweighted}.png`, using `national_weight`.
- `state_year_grid/{stem}_{weighted|unweighted}_grid.png`, a state x year
  small-multiples grid mirroring `financial_analysis.py`'s
  `make_grid_figure`: rows are states, columns are years, missing
  combinations blank (`ax.axis('off')`). Each panel is labelled with its
  own `n` (in weighted mode, sites with a usable PV weight, matching that
  script's convention). Every panel shares one set of bin edges and one
  y-limit, computed across every included cell before any panel is drawn --
  the same principle as that script's `_shared_ylim`. Each panel is that
  `(state, year)`'s own within-year PV-weighted histogram, not renormalised
  further.
- The `pct_of_coordination_savings` pool inherits the `NaN` exclusions
  already in each `per_site_annual.csv`.

Filenames are suffixed by weighting mode, matching
`financial_analysis.py`'s `weight_tag` convention; without it a weighted
and an unweighted run of the same battery size overwrite each other.

Single-figure charts (not grid panels) shade and annotate a central
interval, default the 5th-95th weighted percentile, via `--interval`
(0 disables). Weighted percentiles use cumulative-weight interpolation on
the midpoint convention, which reduces exactly to `np.percentile` at equal
weights.

### Subscription-inclusive household outcomes

`--subscription-change-per-month` adds `12 x` its value to every site's
`diff`, negative where the fee falls, and recomputes
`pct_of_coordination_savings` on the adjusted figure so the two charts
describe the same households.

Without it the charts show the wholesale layer alone, which for Retailer Pool Retention
strips out the thing `alpha` was chosen against. On NSW/QLD/SA/VIC
2021-2025 at `alpha = 0.6` with the fee removed entirely, the median
site-year moves from `+$246` to `-$54` and the share better off from 1.1%
to 58.6%: the sign of the household outcome flips. Comparing variants on
the unadjusted figure compares them on an incomplete bill.

The fee change is a design input, not a model output. The model says what
the retailer could afford -- `subscription_change_vs_amber` in Section 14 --
and what fee is actually offered is a choice, so it is passed in rather
than inferred. It is carried into the chart label and the output filename,
since two runs of one variant at different fees are different offers.

The denominator comes from `annual_coordination_savings` (Section 11) where
present, and for files written before that column existed is recovered as
`100 * diff / pct`. That is exact, both scaling together, except where
`diff` is exactly zero, and those sites drop out of that one metric with a
warning.

### Interface

```
python3 analysis/scripts/cap_account_aggregate.py
    --results-root PATH      one run's output directory, i.e.
                             {cap_account_analysis --output-dir}/{run_name}
                             (default:
                             analysis/output/cap_account/full_run_rolling_amber)
    --data-root PATH         project root for site metadata
    --battery-name NAME      e.g. '20kWh'; required
    --variant NAME           which variant subfolder to aggregate, per
                             Section 12's naming
    --subscription-change-per-month CHANGE
                             fee change this offer allows, $/month,
                             negative where the fee falls (default 0)
    --unweighted             plain concatenation (default: off)
    --weights-file PATH      (default: analysis/config/pv_size_bands.yaml)
    --interval PCT           central interval to shade (default 90; 0 off)
    --pct-axis-limit PCT     fix the pct_of_coordination_savings x-axis to
                             [-PCT, +PCT] so runs share one scale; values
                             outside are not drawn (default: data range)
    --output-dir PATH        (default: analysis/output/cap_account_aggregate)
```

Charts are written under `{output-dir}/{battery}/{variant}/`, so neither a
different battery size nor a different variant overwrites another. With a
non-zero `--subscription-change-per-month` the filename suffix gains
`_sub{change}` and the chart label states the change.

Tesla Electric's `per_site_annual.csv` carries the columns this script reads, so
`--variant dynamic_...` needs no new code path; its differences are in
Section 20.

## 14. Retailer position across variants

A separate script, `cap_account_retailer.py`, reading
`cohort_period_summary.csv` across variants. It is not an extension of
`cap_account_aggregate.py`: the unit of observation is the cohort-month
rather than the site-year, so PV size banding does not apply;
`cap_account_aggregate.py` takes a single `--variant` by construction,
whereas this reads across variants, which is its whole purpose.

Every figure in this section is in dollars per site per month, so the
composition, the spread and the break-even subscription all read on one
set of units.

Tesla Electric enters this section only through `--dynamic`, and only in the
spread and grid charts; its own composition chart is separate
(Section 20).

**Variants are grouped by cap threshold** and each group is aggregated and
written on its own, to `{output-dir}/{battery}/x{threshold}/`. Variants at
different thresholds are different offers rather than points on one
comparison: the cover ratio, the break-even subscription and the cohort's
self-hedge all move with `x`. A variant folder name that does not parse is
written to `x_unparsed` rather than pooled with a threshold it may not
share. The threshold is stated on every chart.

### Interface

```
python3 analysis/scripts/cap_account_retailer.py
    --results-root PATH      one run's output directory (default:
                             analysis/output/cap_account/full_run_rolling_amber)
    --battery-name NAME      required
    --variants NAME ...      variant folders to include (default: all found)
    --no-amber               omit the synthesised uncapped baseline
    --threshold-per-mwh X ...
                             restrict to these thresholds (default: every
                             threshold found, each in its own group)
    --subscription-change VARIANT=CHANGE ...
                             fee actually offered under a variant, as a
                             change from the baseline fee, $/month
                             (default: no change for any variant)
    --unweighted             combine states by concatenation, not state_mix
    --weights-file PATH      source of state_mix (default:
                             analysis/config/pv_size_bands.yaml)
    --output-dir PATH        (default: analysis/output/cap_account_retailer)
```

`--subscription-change` sets the fee used in the net position for the
named variant only. `amber_equivalent_subscription` is computed from the
baseline fee, not the offered one, so applying a change cannot feed back
into the figure for what change the retailer could afford.

### The variant set

One uncapped baseline and one row per offer configuration, e.g.

| variant | offer | settlement | parameter |
|---|---|---|---|
| Amber | uncapped passthrough | n/a | n/a |
| A: monthly | cohort | monthly | surcharge 1c/kWh |
| A: daily | cohort | daily | surcharge 1c/kWh |
| B: alpha 0 | retailer | monthly | alpha 0 |
| B: alpha 0.5 | retailer | monthly | alpha 0.5 |
| B: alpha 0.75 | retailer | monthly | alpha 0.75 |

The Amber baseline is synthesised rather than run: under uncapped
passthrough every cap account quantity is zero by definition, so its row is
constructed directly and no scenario directory is read for it.

### Decomposition of the retailer's position

Under both offers the retailer's position is the same three-term identity,
which is why one chart can carry every variant:

```
retailer_net_cost = cost_absorbed - credit_retained - surcharge_retained
```

| | `cost_absorbed` | `credit_retained` | `surcharge_retained` |
|---|---|---|---|
| Amber | 0 | 0 | 0 |
| Cohort Hedging | `total_cost_uncollected` | `p * credit` | `R` less refund and bonus paid |
| Retailer Pool Retention | `total_cost_uncollected` | `(1 - alpha) * credit` | 0 |

`credit_retained` means different things under the two offers and the
charts must say so. Under Cohort Hedging, `p = min(1, cost/credit)`, so
`p * credit = min(cost, credit)`: it is the cohort's own above-cap export
credit redirected to cover other members' forgiven costs, it never exceeds
`cost_absorbed`, and it is the cohort's pool passing through the retailer's
book rather than margin. Under Retailer Pool Retention it is `(1 - alpha)` of all above-cap
export value, unrelated to what any cost was, and it is margin.

**`retailer_net_cost` must be computed from this identity rather than as
Section 7's `max(0, total_shortfall - R)`.** The floored form is correct
whenever the month has a credit pool, but where `total_credit_unpaid == 0`
the whole of `R` is retained (Section 8) and the floor reports `$0.00`
against a true gain of `R`. In NSW 2022 that is the entire difference: the
floored figure hides January's `$467` and nothing else. Verified on
synthetic cohorts -- the two forms agree to the cent in every month with a
credit pool, and differ by exactly the stranded surcharge otherwise. This
makes the three components new columns on `cohort_period_summary.csv`
(Section 11) and requires re-running any variant whose retailer position is
to be reported.

**Retailer net position per site per month** is the reporting unit:

```
net_position[cohort, month] = subscription_per_month
                              - retailer_net_cost[month] / n_sites
```

where `n_sites` is the cohort's own modelled site count. This is a
per-modelled-site figure, not a population-weighted one, and must be
labelled as such: scaling settlement to a population is the unresolved
question in Section 15 and this section does not resolve it.

Settlement periods shorter than a month are summed into calendar months
first, so a daily variant contributes 12 observations per cohort-year like
any other.

### Subscription fee

Read from the run's own retail tariff via `load_retail_tariff`, whose
`daily_c_per_day` sums `daily_charges_c_per_day` and so is the
subscription for a `wholesale_exposed` tariff. Taking it from the tariff
rather than a flag keeps it consistent with the baseline automatically when
the tariff changes, in the same way the network tariff supplies
`import_lf`/`export_lf`. `--subscription-per-month` overrides it where a
variant is modelled at a deliberately different fee.

```
subscription_per_month = daily_c_per_day * 365 / 100 / 12
```

365 always, including leap years, because the c/day figure was itself
derived by dividing an annual subscription by 365; using 366 in a leap year
would overstate the fee rather than track it. Monthly is the annual figure
over 12 rather than the month's own day count, so every month carries the
same subscription (`82.1918` c/day is `$300.00` a year, `$25.00` a month).

On a `tou` tariff the same field is a supply charge rather than retailer
margin, so the value is left unset with a warning instead of being reported
as a subscription.

It is the whole of the retailer's revenue under the uncapped baseline and
identical across variants unless deliberately varied.

The model does not determine the fee, so the derived quantity is more
useful than the input:

```
amber_equivalent_subscription = subscription_per_month
                                + mean over cohort-months of
                                  retailer_net_cost[month] / n_sites
```

the fee at which each variant leaves the retailer **as well off as under
uncapped passthrough**. This is not the fee at which the cap mechanism
alone breaks even: the model represents only the wholesale layer, so a fee
zeroing the mechanism's own position would imply the subscription could
fall to nothing, when it in fact covers acquisition, billing and everything
else the model does not see. Holding the retailer at its current position
and asking how the fee moves is the question the model can answer.

The delta is reported alongside as `subscription_change_vs_amber`. Negative
means the fee could fall, which is where Retailer Pool Retention at low `alpha` is expected
to have something to show, since the retailer retains cohort value it does
not retain under Cohort Hedging.

### Charts

Four figures at the national level, examined before the grid, plus the
grid.

Separate figures rather than panes of one, so each carries its own legend
and can be used on its own. Variant labels are wrapped to two lines and set
horizontally.

Each threshold group also writes `retailer_cohort_months.csv`, the frame
behind every chart, and `retailer_summary.csv`, one row per variant: mean,
5th and 95th percentile and worst-month net position,
`amber_equivalent_subscription_per_month`, `subscription_change_vs_amber`,
the count of months in which the retailer lost money, and the mean of each
identity component.

**`retailer_composition.png`.** Waterfall per variant, small multiples on a
shared y-axis: subscription, then each component of the identity above as a
step, ending at the net position. The legend runs in the same order as the
steps are drawn. A waterfall rather than a stacked bar
because `cost_absorbed` is negative and the eye cannot add a segment below
the axis to segments above it. Shows at a glance that Cohort Hedging runs a nearly
flat book while Retailer Pool Retention runs a large two-sided one -- a distinction the net
figure alone hides. Each step is labelled with its value, since under
Cohort Hedging the components are small against a `$25` subscription and at a
scale that also has to hold Retailer Pool Retention the bar is nearly invisible. Panels
wrap at three per row rather than extending horizontally, and carry no x
tick labels: the legend and the left-to-right step order identify them
without repeating the same five words under every panel.

**`retailer_spread.png`.** Box per variant over every cohort-month in the
pool, whiskers at the 5th and 95th percentiles with everything beyond
plotted individually. The outliers are the point: the worst month is the
figure a retailer hedging this book cares about, and it is what the mean
composition cannot show. A dashed reference line at the uncapped
passthrough position marks where the cap mechanism cost the retailer money.
The Amber baseline appears as a flat line at exactly the subscription; any
width to it would indicate an aggregation error.

Most cohort-months have no cap event, so the position sits at exactly the
subscription and the 25th, 50th and 75th percentiles coincide: the box
collapses onto the median line. That is the correct rendering of the data,
and the figure says so rather than leaving it to be puzzled over.

The y-axis is capped at `$100` per site per month. Retailer Pool Retention's credit-retained
months run far above the rest and would otherwise set the scale for every
variant; points above the cap are counted on the axis rather than dropped.

**`retailer_subscription.png`.** Bar per variant of
`amber_equivalent_subscription`, against a reference line at the tariff's
actual fee, annotated with the delta. Where `--subscription-change` is
given, the fee actually offered is marked on each bar.

**`retailer_cover_ratio.png`.** Cover ratio is
`total_credit_unpaid / total_cost_uncollected` **per settlement period**,
each variant on its own cadence. Aggregating it to a common calendar month
hides exactly what sub-monthly settlement does: a month can over-cover many
times over while individual days within it did not. Checked on NSW 2024 and
SA 2021, every month in which daily settlement cost the retailer money had
a monthly cover ratio of 1.19, 13.9 and 26.0 -- a monthly ratio reports a
comfortable cohort in the months that actually hurt. The ECDF normalises
the differing period counts, so nothing is lost by leaving each variant on
its own period. The
same number carries a different meaning per offer, so the chart marks each
variant's own break-even threshold rather than a single line: the retailer
loses under Cohort Hedging when the ratio falls below 1, and under Retailer Pool Retention when it
falls below `1 / (1 - alpha)` -- 1 at `alpha = 0`, 2 at `0.5`, 4 at `0.75`.
Under A the threshold is a solvency point, the cohort ceasing to cover
itself; under B the retailer holds both sides regardless and the threshold
is a profitability point. The axis label must say which.

One ECDF per variant, with that variant's threshold as a vertical line.
The reading that matters is where each curve crosses its own threshold:
that height is the fraction of months in which the retailer lost money on
the cap mechanism. Drawn on a log x-axis, since the ratio spans several
orders of magnitude.

A cover ratio says which side of break-even a period fell on, not how much
was at stake: a ratio of `0.01` can be cents or thousands. An inset table
carries the magnitude the curve cannot. Its first column, periods below
that variant's own threshold, is read off the curve on the same
denominator, so the two agree by construction rather than by coincidence.
The remaining columns are per cohort-month: the share in which the
mechanism cost the retailer money, the shortfall as a share of subscription
revenue, and the worst single month in dollars per site.

`shortfall / revenue` is measured across every cohort-month, not only the
ones at a loss. Since the subscription is constant, the mean loss per
cohort-month divided by the subscription is exactly total shortfall over
the full period as a share of total subscription revenue, which is the
figure worth quoting; the column heading says so, because "mean loss" alone
reads as if it were conditional on a loss having occurred.

Those first two columns measure deliberately different things and the
figure says so. `periods uncovered` is the wholesale position alone --
whether the cohort covered itself. `months at a loss` is after the hedging
surcharge, which is pooled monthly (Section 8) and frequently covers the
gap. The distance between them is how much work the surcharge is doing,
which makes the pair more informative than either alone. A daily variant
can still show more months at a loss than uncovered days, since one bad day
carries its whole month.

Two materiality floors apply, both against the same floating-point residue
left by settlement arithmetic that washes to zero.

A cohort-level floor of one cent on `total_cost_uncollected` and
`total_credit_unpaid` before the ratio is formed. A period with neither
side material had no cap account in any meaningful sense and is excluded;
one with no material cost is off-scale right, since the retailer cannot
lose. Unfloored, residue on the credit side produces ratios of `1e-15` and
stretches the axis fifteen orders of magnitude to the left over nothing.
Ratios are also bounded below at `1e-3` by an off-scale bin symmetric with
the one at `1e4`: below it the cohort has essentially no cover against a
real cost, and the exact value says no more than the bin does.

A per-site floor of half a cent per month on the exposure count. Without it the count reports every month in which the mechanism
washed to zero but left floating-point residue behind -- on NSW/QLD/SA/VIC
2021-2025, that was all 23 of the monthly variant's apparently exposed
months, every one of them a loss of `1e-15` or smaller, against a true
count of zero.

Cover ratios above `1e4` are folded with the infinite ones into a single
off-scale bin at the right. A month whose ratio runs to `1e17` -- a trivial
above-cap cost against a large credit -- tells you no more than one at
`1e4`, and including it stretches the axis across twenty orders of
magnitude, crushing the range where the thresholds actually sit.

This does not duplicate the spread pane above. That pane shows which side
of break-even the retailer landed on and by how much in dollars; the cover
ratio shows how much margin there was against the cohort's own capacity to
cover, which is what tells you whether a month was comfortable or a near
miss at the same dollar outcome.

Two degenerate cases need handling rather than silently dropping. A month
with `total_cost_uncollected == 0` has an infinite ratio: the retailer
cannot lose, so those months are counted above every threshold and drawn
at the right edge. A month with no above-cap intervals at all has no ratio
in either direction; those are excluded from the curve and reported as a
count in the caption, since a variant whose cohort simply never faced a
cap event is not the same as one that faced them and covered them.

**Frequency comparability.** For any figure comparing variants on
period-level outcomes, periods are first aggregated to calendar months, so
a daily variant's worst month is its worst month rather than its worst day.
Reporting a worst-day figure against a worst-month figure would make daily
settlement look worse for a reason that is purely an artefact of
granularity.

**Amber in every chart.** Its cover ratio is undefined (`0/0`) and its
position has no spread: it is a flat line at the subscription fee, and the
absence of dispersion is the point being shown rather than a gap in the
data.

**`retailer_position_grid.png`.** State x year small multiples of per-site
monthly net position by variant, mirroring `financial_analysis.py`'s
`make_grid_figure` conventions: shared y-limits across every panel, missing
combinations blank, each panel labelled with its cohort site count. Each
panel holds 12 cohort-months, so the panel form is the spread pane's box
per variant rather than a single bar.

Shared limits are the full pooled range. Zero is not forced into it and the
reference line sits at the subscription rather than the origin: with the
fee at `$25` and the mechanism worth a few dollars at most, including zero
pushes every box into the top tenth of the panel. Once zero is gone the
true minimum and maximum fit comfortably, so nothing is clipped -- a
percentile clip would cut off exactly the worst months the grid exists to
show.

### Aggregation

No PV size banding: there is no site dimension. Years are equalised by
construction: every state-year contributes exactly 12 cohort-months, so
years carry equal weight with no further adjustment, exactly as in
Section 13. States combine by `state_mix`. A state with no `state_mix`
entry is excluded from the national pool with a warning.

### Cohort composition

Every result in this section is for a cohort in which every site has the
same battery. That is a strong assumption and it does more work than it
appears to.

The cover ratio is what the two offers turn on, and it is a function of
battery size. A large battery has stored energy to export through a price
spike and rarely needs to import during one, so it contributes credit and
almost no cost. A small battery runs out during a sustained high-price
stretch and imports above the cap, contributing cost and less credit. A
20 kWh cohort therefore produces the extreme cover ratios seen here
(2,077 on NSW 2021), and smaller batteries will produce lower ones, more
retailer exposure, and a materially different picture. The size sweep is
cheap: the scenarios already exist per battery size and this section's
scripts take `--battery-name`, so nothing needs re-running.

A realistic mixed-size cohort matters more, and the two offers behave
differently under it:

- **Cohort Hedging changes character.** In a homogeneous cohort, which sites are
  short in a given period varies with the event, so the pool is
  insurance. In a mixed cohort the small batteries are systematically the
  ones covered and the large ones systematically the cover, so the same
  mechanism becomes a persistent transfer from large systems to small ones.
  That is the free-riding problem the surcharge in Section 8 exists to
  address, but structural rather than incidental, and whether a flat
  per-kWh import surcharge is the right instrument against a persistent
  rather than an intermittent imbalance is an open question.
- **Retailer Pool Retention is unaffected in kind.** With no cohort pool, a household's
  outcome depends only on its own flows, so cohort composition changes the
  aggregate figures but introduces no cross-subsidy. Robustness to cohort
  composition is an argument for Retailer Pool Retention that the homogeneous results do
  not surface.

Constructing a mixed cohort needs no re-optimisation. Each site's dispatch
is optimised independently against price, with no interaction between sites
in the LP, so splicing each site's flows from the scenario matching its
assigned battery size gives exactly the flows that site would have had.
The assignment would come from an installed-capacity distribution, in the
same spirit as the PV size bands already in `pv_size_bands.yaml`. This is a
modelling addition rather than an approximation, and it is the main thing
standing between these results and a claim about a real retail book.

## 15. Deferred and open

The implementation notes here are about how current code should be
structured so that adding these later does not force a rewrite.

- **Offsetting cap account shortfall against all exports, not just
  above-cap exports.** Would run as an adjustment to positive `S(site)`
  balances before settlement.
- **Eligibility requirements for cap protection**, as an alternative or
  complement to the universal surcharge: require a demonstrated minimum
  hedging contribution to qualify for capped-import protection, targeting a
  site that is a serial beneficiary with little to offer the cohort. Harder
  to specify than it looks. A static test ("has exported at some point",
  "has a system above some nameplate size") is insufficient: a large PV
  system on a large load may rarely have real export surplus, while a small
  system on a small load might export reliably. PV and battery size only
  mean something relative to that site's own load. A workable version
  likely needs a trailing-period realised metric -- does this site show up
  with above-cap export capacity when it matters, relative to its own
  consumption -- not a hardware specification. Separate from the surcharge,
  which is universal rather than a membership gate.
- **Targeting the hedging surcharge more precisely.** The flat, always-on
  surcharge is charged on import volume, not on whether an import reflects
  the free-riding it is meant to counter. Checked against NSW 2022 20kWh
  data, it correlates with the free-rider measure `annual_cap_delta`
  (r=0.80) but reshuffles outcomes for some genuine above-cap exporters --
  a handful flipped from net winner to net loser despite a negative
  `annual_cap_delta` in the no-surcharge run. The redistribution is small
  relative to each site's own wholesale export earnings even where it
  changes rank. Two alternatives: (a) surcharge imports only when the price
  exceeds a second, lower threshold `y < x`, tying the surcharge to the
  same price-risk driver as the cap, so imports at cheap or negative prices
  are never surcharged; or (b) surcharge only imports outside a
  solar-adequacy window (e.g. 11am-3pm), a time-of-day proxy for
  grid-dependence, easier to explain but weakly linked to price risk. (a)
  is the smaller change, reusing the existing price series and
  threshold-comparison; (b) needs a time-of-day mask via
  `local_wall_clock`, already imported.
- **Water-filling (non-uniform) allocation of `total_cost_uncollected`**
  against unpaid credit balances, instead of the flat `p`. Settlement
  (Section 7) is already its own function taking per-site `S(site)` and
  returning per-site adjustments, so the rule can be swapped without
  touching `cap_delta`.
- **Population weighting by PV system size** inside settlement. Not a
  drop-in reuse of `financial_analysis.py`'s pattern: there, `site_weight`
  turns a mean across sites into a population-weighted mean, a
  reporting-only device. Cohort Hedging's settlement is a closed pool across the
  real modelled sites, so weighting is a modelling question: should each
  site's `cap_delta` be scaled by `site_weight` before summing into cohort
  totals (changing `p` and `retailer_net_cost`), or should settlement run
  unweighted with `site_weight` applied only to the household-outcome
  distribution in Section 11 -- in which case the retailer P&L and the
  household distribution answer differently-weighted questions, and must
  say so. Needs an explicit decision, not a flag bolted on by analogy.
  Deferred; the model runs unweighted, matching the real modelled cohort.
  **Implementation note**: keep every per-site quantity (`cap_delta`,
  `S(site)`, `adjustment[site]`, `payment[site]`) as an explicit per-site
  Series or column throughout, collapsed to a cohort figure only at the
  final summation, so whichever design is chosen is a join of `site_weight`
  onto already-present per-site figures at that one point. Once
  implemented, unweighted mode should reproduce current numbers exactly as
  a regression check.
- **Mixed battery-size cohorts** (Section 14). Splice each site's flows
  from the scenario matching its assigned size; no re-optimisation, since
  sites do not interact in the LP.
- **Threshold sweeps and multi-battery-size / multi-state cohorts.**
  Section 12 takes a single `x`, state and battery size per run. The
  per-interval and settlement functions take `x` as an argument at the call
  site, so wrapping the script in a sweep is a looping exercise, and
  Section 12's variant folders keep the runs apart.
- **Dispatch response to the capped signal.** Both offers assume retailer
  dispatch against the real price (Section 2), so the verified flows are
  correct. The size of what the household gives up by ceding control can be
  measured by re-running `perfect_foresight_lp.py` against the clipped
  price series `min(price, x)` with the degradation term unchanged, and
  comparing the household objective to the re-billed real-dispatch result.

---

# Part V -- Tesla Electric

Tesla Electric shares Section 3's inputs and Section 13's aggregation with
Cohort Hedging and Retailer Pool Retention. Nothing in Sections 4-10 applies: there is
no cap threshold, no settlement period and no cap account.

## 16. Settlement

Every interval and every site settles on its own. The plan's rates are read
from the `type: mixed` tariff (Section 18); the network tariff is the run's
own, exactly as for Cohort Hedging and Retailer Pool Retention.

**Imports.** The household pays the plan's time-of-use rate plus its flat
adder:

```
household_import[site, t] = import_kwh[site, t] * r[t]
r[t] = tou_rate[t] + usage_other            # usage_other = greenpower 0.821 c
```

The TOU rate bundles energy and network. The household has no wholesale
exposure on imports at any price, so its maximum import price is the peak
TOU rate plus the adder, independent of spot: NSW 50.53, QLD 40.95,
SA 66.15, VIC 45.32 c/kWh.

**Exports.** The real-time value of the export to the retailer is
loss-adjusted spot plus the network export tariff. The household is
credited a share of the spot part only, floored at zero in every
interval:

```
V[site, t] = export_kwh[site, t] * export_lf
             * (real_price[t] + network_feedin[t])
household_export_credit[site, t] = export_kwh[site, t]
             * max(0, share * export_lf * real_price[t])
```

`share = 0.9`, the `wholesale` basis and the zero floor come from the
tariff (Section 18). `V` is the same quantity the Amber bill credits as
`wholesale_feedin_credit + network_feedin_credit` (with sign reversed).
The network export tariff, whether a credit or a charge such as Ausgrid
EA029's, stays with the retailer. At a negative spot price the household
is credited nothing and pays nothing for the export. Flows are not
curtailed at negative prices: the plan's retailer can curtail some sites
but not all, and this is not modelled; the household's outcome is the
same either way.

**Daily.** The household pays the plan's daily charge (NSW 110, QLD 112,
SA 113, VIC 76 c/day). The network daily charge falls to the retailer,
where under Amber it passes through to the household.

Nothing nets across intervals, so there is nothing for a settlement period
to set. There is no self-hedging component either. Under Cohort Hedging and Retailer Pool Retention
self-hedging lets a household's above-cap exports offset its own above-cap
imports; under Tesla Electric the household cannot trade a spike against the
retailer at all, because the retailer holds dispatch. No cohort pool and no
surcharge.

**Dispatch.** The plan's retailer dispatches against household-side
prices rather than the real price: the billed TOU rate plus the positive
part of the loss-adjusted spot price on imports, and the household's export
credit on exports (`model_spec.md`, Mixed-plan dispatch prices). Those
flows are modelled by the `full_run_*_tesla` runs, and Tesla Electric is taken
from one of them, the achievable `full_run_rolling_tesla` by default. The
Amber side keeps the Amber run's own flows, so each side of the comparison
is the dispatch its own plan produces. The two runs must share a network
tariff, a year and a battery size.

`--dynamic-dispatch rebill` bills the Amber run's flows under the plan
instead. That measures the plan's billing on Amber's dispatch, which is
the plan's billing terms isolated from its dispatch, and it is how Tesla Electric
was computed before the `full_run_*_tesla` runs existed.

**Household comparison: the full bill.**

```
diff[site] = household_total_C[site] - household_total_Amber[site]
```

each on its own run's flows. Cohort Hedging and Retailer Pool Retention change only the wholesale
layer, so their `diff` is a wholesale-layer difference (Section 7). Under Tesla Electric the
network layer and the retail adders move from the household to the
retailer and the fixed charge changes, so a wholesale-layer difference
would omit most of the change. The difference in fixed charges is inside
`diff`, which is why Section 13's `--subscription-change-per-month` must be
zero for this variant (Section 20).

`diff` carries both the plan's billing terms and its dispatch, and splits
into the two on the plan's own flows:

```
diff_billing[site]  = household_total_C - household_total_Amber_on_C_flows
diff_dispatch[site] = household_total_Amber_on_C_flows - household_total_Amber
diff                = diff_billing + diff_dispatch
```

where `household_total_Amber_on_C_flows` is the Amber tariff billed on the
plan's flows. Under `--dynamic-dispatch rebill` the second term is zero by
construction. Both terms are written per site (Section 19); the figures
use `diff`.

## 17. Retailer position

**The Amber comparator is the `$25/month` subscription.** Everything else
on the Amber bill is treated as pass-through cost, including the `1.100` c
hedging component of Amber's adder stack: any retailer carrying import
risk incurs a hedging cost, and Tesla Electric carries more of it than
Amber. Tesla Electric's position is then what the plan collects less
what the Amber household would have paid for everything except the
subscription, both on the plan's own dispatch, which is the dispatch this
retailer actually serves:

```
net_position[site, month] = household_total_C - (household_total_Amber
                                                 - subscription_Amber)
```

set against the Amber retailer's `$25`. It decomposes into four parts:

| component | definition |
|---|---|
| `net_fixed` | plan daily charge - network daily charge |
| `import_margin` | sum over import intervals with `r[t] > c[t]` of `(r[t] - c[t]) * import_kwh` |
| `cost_absorbed` | sum over import intervals with `c[t] > r[t]` of `(c[t] - r[t]) * import_kwh` |
| `credit_retained` | Amber export credit - Tesla Electric export credit |
| `net_position` | `net_fixed + import_margin - cost_absorbed + credit_retained` |

where `c[t]` is the cost of an imported kWh to the retailer:

```
c[t] = import_lf * (real_price[t] * (1 + g) + network_usage[t]) + amber_usage_other
```

(`_amber_import_rate`), the Amber household's charge per imported kWh
apart from the subscription.

`amber_usage_other` is the Amber adder stack, 4.081 c, not loss-adjusted,
as for every retail adder in `prices.py`. Because
`feedin_other_c_per_kwh` is empty in all four Amber tariffs and in the
plan,

```
credit_retained = V - export_kwh * max(0, share * export_lf * real_price)
```

which is the network export value, the unshared part of positive spot
value, and the whole of negative spot value. It is negative in intervals
where the spot price is negative, or where a negative network export rate
outweighs the retained spot value.

`import_margin` and `cost_absorbed` are kept apart rather than netted,
because they are different kinds of result: the first is ordinary retail
margin earned in most intervals, the second is the price-spike exposure the
retailer has taken on from the household. Netted, a year of small margins
hides the months in which spikes cost more than the margin earned.

**Identity check.** The decomposition must reproduce the net position
computed from the two bills. The bills come from the core billing path
(`compute_bill`, Section 19); the components come from the layer rate
vectors in `cap_account_analysis.py`. Agreement therefore tests the `mixed`
billing implementation and the decomposition against each other rather than
one calculation against itself. Both bills in the identity are on the
plan's own flows, so it is exact; the household comparison's Amber bill,
on the Amber run's flows, does not enter it.

The check is per site over the full calendar year, using the annual bills:

```
net_fixed + import_margin - cost_absorbed + credit_retained
    = household_total_C - (household_total_Amber - daily_Amber)
```

to Section 5's tolerances. Here every fixed term (the plan daily charge
and the network daily charge in `net_fixed`, and `daily_Amber`, the Amber
subscription) is taken at the annual bill's own day count. That count is
distinct AEST interval-start dates, as in `_reduce_settlement`, so the
check covers `net_fixed` and holds exactly in leap years. Summing the
monthly bills would not reproduce it, because monthly bills count local
dates and the last hour of 31 December falls on a local 1 January
(Section 4).

**Day counts in reporting.** The retailer figures use Section 14's
`subscription_per_month` convention for all fixed charges: c/day x 365 /
12, the same in every month. One `$25` Amber line can then serve all three
offers in the spread and grid charts. The household `diff` uses billing's
own day count. A reported monthly `net_position` therefore differs from the
billed identity by the day-count difference in fixed charges and nothing
else.

## 18. Tariff model support: `type: mixed`

The plan is a `type: mixed` retail tariff,
`model/tariffs/{NSW,QLD,SA,VIC}_tesla_electric_mixed_2026.yaml`. Its fields,
loader rules, layer rates, billing view and dispatch prices are specified in
`model_spec.md` (RetailTariff, Price construction). The plan's settings are
`wholesale_export_share: 0.9`, `export_share_basis: wholesale` and
`export_floor_c_per_kwh: 0.0`. The header comment of each file states that
the import rates bundle energy and network; that the network tariff supplies
the loss factors, the network import cost and daily charge, and the network
export tariff, all on the retailer's side; and that all prices are
provisional.

**Tests** (`model/tests/test_core_types.py`, `model/tests/test_billing.py`):

- Loader: the four YAMLs parse; a `mixed` file missing any of
  `wholesale_export_share`, `export_share_basis` or `export_floor_c_per_kwh`
  raises, as does any of the three on another type, an unknown basis, a
  share outside `[0, 1]`, `feedin_tariffs` on `mixed` and any advanced
  feature on `mixed`.
- Layer rates: `retail_export == max(floor, share * V) + feedin_other` at
  positive, zero and negative spot, under both bases, with and without a
  floor, and with a negative network feed-in rate.
- Billing, on a hand-built trace: `household_total` = TOU import cost +
  daily - the floored export credit; `retailer_margin` = `r[t] * import_kwh`
  + plan daily - `import_lf * (price + network_usage) * import_kwh` +
  `credit_retained` - network daily.
- Section 17's identity, on a synthetic full-year trace spanning a leap
  year, with positive, negative and above-TOU spot prices and a negative
  network feed-in rate. Both bills come from `compute_bill`. The components
  are computed directly from the rate definitions in Section 17, not
  through `build_layer_rates`, so that the test is independent of the code
  it checks.
- The existing `tou` and `wholesale_exposed` tests pass unchanged.

## 19. Outputs and interface

`cap_account_analysis.py --offer dynamic`. `--threshold-per-mwh`,
`--settlement-period`, `--surcharge-rate-per-kwh` and `--share-fraction`
do not apply and raise if set away from their defaults. The share, its
basis and the floor come from the tariff, not the command line, because
they are published terms of the plan rather than design parameters to
sweep.

**Variant folder:** `dynamic_{tariff}`, the tariff id without its state
prefix, with `_modelled` appended under `--dynamic-dispatch own`, e.g.
`dynamic_tesla_electric_mixed_2026_modelled`. No threshold or period, since
neither exists. The run name level above it is the Amber run named by
`--run-config`, which is the baseline the variant is measured against.

**Computation.**

- After the Amber run's flows, prices, network tariff and retail tariff are
  loaded, the offer branches before any cap-account code. Under
  `--dynamic-dispatch own` it then loads the offer run's scenario, flows and
  `bills.csv`, and takes the plan from that scenario's `retail_tariff_id`;
  the two runs are required to share a network tariff id and a flow index.
  Sites are the intersection of the two runs' billed sites.
- **Bills.** Every bill comes from the core model's own billing,
  `compute_bill_and_monthly_bills`, per site. Three are computed: the plan
  on the plan's flows, Amber on the plan's flows, and Amber on the Amber
  run's flows. Billing reads only `import_kwh` and `export_kwh` from
  `SiteTraces`, so the remaining trace fields are zero-filled. No billing
  logic is duplicated in the script. Under `--dynamic-dispatch rebill` the
  second and third are the same billing.
- **Components.** `build_layer_rates` runs once per tariff, since rates do
  not depend on flows. The Section 17 components are built from those
  vectors applied to the plan's `import_kwh` / `export_kwh` DataFrames
  across all sites at once.
- **Reconciliation.** Each recomputed annual `household_total` is checked
  against the `bills.csv` of the run it was billed on, for every site, at
  Section 5's tolerances. This extends Section 5's check from the wholesale
  layer to the full bill, which is what Tesla Electric's `diff` depends on. It
  also confirms that the plan's modelled run reproduces under this script's
  billing. Section 17's annual identity is checked after.

**`per_site_annual.csv`** -- `baseline_household_total` (Amber, from
`bills.csv`), `offer_household_total`, `diff`, `diff_billing`,
`diff_dispatch` (Section 16), `pct_of_wholesale_export_earnings`, `annual_coordination_savings`,
`pct_of_coordination_savings`, `n_months_better_off`,
`n_months_worse_off`, each as defined in Section 11 but on the full bill.
`annual_cap_delta` and `annual_value_returned` are not written.

**`by_site_month_{metric}.csv`** -- `amber_household_total`,
`offer_household_total`, `diff`, `net_fixed`, `import_margin`,
`cost_absorbed`, `credit_retained`.

**`dynamic_month_summary.csv`** -- one row per calendar month: `month`,
`n_sites`, `subscription_per_month` (Amber, Section 14's convention), and
cohort totals of `net_fixed`, `import_margin`, `cost_absorbed`,
`credit_retained` and `net_position`. Deliberately not named
`cohort_period_summary.csv`, so `cap_account_retailer.py`'s discovery does
not mistake it for a cap variant.

**`summary.txt` and charts** -- as Section 11, less the retailer
per-period breakdown, with the retailer figures replaced by the Section 17
components and titles naming the plan against Amber rather than a cap.

`summary.txt` names both runs and the dispatch the variant was computed on,
and reports the means of `diff_billing` and `diff_dispatch` beneath the
`diff` summary.

`runners/run_all_cap_analysis.sh` gains one dynamic run per state-year.

## 20. Aggregation and retailer charts

**`cap_account_aggregate.py`.** Runs unchanged on
`--variant dynamic_tesla_electric_mixed_2026_modelled`: it reads only `diff`,
`pct_of_coordination_savings` and `annual_coordination_savings`. For
`dynamic_*` variants the titles and x-labels read "Offer C - Amber" (the script's label) rather
than "capped - baseline", and a non-zero `--subscription-change-per-month`
raises, since the fixed charges are already in `diff`.

**`cap_account_retailer.py`.**

```
    --dynamic VARIANT        include Tesla Electric, e.g.
                             dynamic_tesla_electric_mixed_2026_modelled
                             (default: off)
```

- Discovery: `{state}_{year}_{battery}/{VARIANT}/dynamic_month_summary.csv`.
- The dynamic rows are appended to every threshold group. Tesla Electric has no
  threshold, so it is the same comparator in each.
- Each row carries Amber's `subscription_per_month` and
  `retailer_net_cost = subscription_per_month * n_sites - net_position`,
  so Section 14's `net_position = subscription_per_month -
  retailer_net_cost / n_sites` returns Tesla Electric's position with no special
  case. The cap-account component columns are `NaN`.
  `--subscription-change` does not apply and raises if it names the
  dynamic variant.
- Ordering: Amber, Cohort Hedging, Tesla Electric, Retailer Pool Retention.
  Label `C: Tesla Electric`,
  worded `Dynamic: Tesla Electric`.
- **Included:** `retailer_spread.png`, `retailer_position_grid.png`, and a
  row in `retailer_summary.csv` with the net-position statistics and
  `months_retailer_lost_money` (component and subscription-change columns
  `NaN`).
- **Excluded:** `retailer_composition.png`, `retailer_subscription.png`,
  `retailer_cover_ratio.png`. The first two are built on Section 14's
  three-term identity, which Tesla Electric does not have; the cover ratio is
  undefined without a cap account.
- `_plot_grid` and `_plot_cover_ratio` take the reference subscription from
  the Amber rows rather than `frame['subscription_per_month'].iloc[0]`,
  which is whichever row sorts first.

**`retailer_composition_dynamic.png`.** Waterfall of `net_fixed`,
`import_margin`, `cost_absorbed`, `credit_retained` and `net_position`, per
site per month, with a dashed line at the Amber `$25`. A national panel
weighted by `state_mix` (Section 14's aggregation), and one panel per state.
It does not depend on the threshold, so it is written once, to
`{output-dir}/{battery}/dynamic/`, with `dynamic_summary.csv` of the
component means beside it.

### Deferred

- **Curtailment at negative prices** where the retailer controls the
  inverter.
- **Windowed or tiered export share**, if a plan appears that uses one.
