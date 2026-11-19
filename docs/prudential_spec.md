# Prudential requirement — Specification

## Purpose and scope

`analysis/scripts/prudential_analysis.py` estimates the credit support that AEMO requires of a retailer for its customers' spot purchases: the maximum credit limit (MCL). It reports the MCL per household for four cohorts, per state and nationally, and for a retailer of `--households` such households.

The MCL is the minimum credit support a participant must lodge (CLP cl 10). Outstandings must also stay below the trading limit, which is credit support less the PM (cl 12). The script therefore also reports the credit support required to meet both conditions over 2021–2025, and the annual cost of holding a bank guarantee for it.

The default retailer size, 50,000 households, is about the size of Amber's battery-equipped customer base. Per-household MCLs are a few hundred dollars at most. AEMO's rounding only matters at retailer scale.

| Cohort key | Label | Flows from run | Scenario |
|---|---|---|---|
| `no_cer` | No CER | `full_run_no_cer_amber` | `{STATE}_{YEAR}` |
| `solar_only` | Solar Only | `full_run_solar_only_amber` | `{STATE}_{YEAR}` |
| `self_consumption` | Self Consumption | `full_run_self_consumption_amber` | `{STATE}_{YEAR}_{N}kWh` |
| `coordinated` | Amber (coordinated) | `full_run_rolling_amber` | `{STATE}_{YEAR}_{N}kWh` |

Each cohort's flows come from the named run, and its site set comes from that run's `bills.csv`. The MCL depends only on a cohort's grid flows, not on its retail plan.

AEMO's 2026 parameters (seasons from Summer 2026) are applied to the 2021–2025 flows. Part 3 does the same with the 2026-27 tariffs.

The following are not modelled:
- reallocations;
- hedge contracts and owned generation, because neither changes the MCL without a reallocation;
- ancillary services;
- regulated stand-alone power systems;
- PM Full Offset.

PM Limited Offset applies, as it does by default (CLP cl 4.3.3(b)(ii)).

## Formulation

### Sources

The MCL calculators are in `docs/references/aemo_prudential/`. The other sources are publicly available and are not included in the repository.

| File | Used for |
|---|---|
| `NEM Credit Limit Procedures.pdf` | CLP v10.0, effective 11 December 2026. Provides the season definitions, TOSL, TRP, GST, the OSL (cl 5) and PM (cl 6) formulas, the ToD segments (cl 8), rounding (cl 10.1), and the debit and credit energy definitions (glossary). |
| `MCL Calculator {Summer,Winter,Shoulder}.xlsx` | AEMO's 2026 calculators. Provide the regional parameters on `RegionalData` and the formulas on `MCL` and `ParticipantData`. The script reads the parameters at run time. |
| `AEMO _ NEM Regional Volatility and Price.pdf` | The published 2026 PR, VFOSL and VFPM. The calculators carry the same values. |
| `NEM Market Calendars.zip` | `NEM_CSV_Calendar_2027.csv`: billing weeks and payment dates under the shortened settlement calendar. |
| `ACIL Allen - Final determination - Default market offer 2026–27 - …pdf` | The bank-guarantee rate (p. 79) and the Ausgrid benchmark (Table 3.14, p. 80). |

### Cohort energy

Let n_t be the cohort's mean net grid flow in interval t, in kWh per household, with import positive. It is weighted by PV size band, as in parts 1–3, or unweighted with `--unweighted`.

Debit and credit energy per household are:

- d_t = max(n_t, 0) × λ_imp
- c_t = max(−n_t, 0) × λ_exp

λ_imp and λ_exp are the run's network tariff `import_loss_factor` and `export_loss_factor`, the factors billing applies to the wholesale layer. CLP debit and credit energy are loss-adjusted (glossary).

Imports and exports are netted per interval across the cohort. A retailer is settled on its customers' metered energy in aggregate, so one household's export offsets another's import in the same interval. Household exports therefore count as credit energy.

Seasons follow CLP cl 4.3.1(a):
- Summer: 1 December to 31 March.
- Winter: 1 April to 31 August.
- Shoulder: 1 September to 30 November.

Trading interval k = 1..288 is indexed by interval start in market time (AEST). Flows are stamped at interval end. The calculators' TI 1 is taken as 00:00–00:05: the regional load profile reaches its minimum at TI 36–66 (03:00–05:30), and its maximum at TI 204–232.

For a season, the debit profile is L_k = (Σ d_t over intervals in TI k) / days, in MWh, and EL = Σ_k L_k (MWh/day). The credit profile G_k and EG are defined the same way. Pooling across years sums the per-TI energy and the days, so the pooled profile is the mean day over every year's season.

### Method 1: the 2026 calculator (`calculator`)

This method follows the calculator's formulas with no reallocations, no dollar offsets and PM offset "No". For region R and season:

- PRAFL = ROUND(MAX(LWPR, LWPR²), 2)
  - LWPR = ROUND(PLWP / RLWPR, 2)
  - PLWP = ROUND(Σ_k L_k × PTI_k / Σ_k L_k, 2)
  - These are `ParticipantData!L9:N9`. PRAFL is 1.05 when EL = 0.
- PRAFG is the same formula on G_k (`L12:N12`). It is 0.95 when EG = 0.
- VEL = EL × PR × PRAFL × VFOSL × (1 + GST), and VEG = EG × PR × PRAFG × VFOSL × (1 + GST) (`MCL!D16`, `D17`).
- OSL = MAX((VEL − VEG) × TOSL, (VEL − VEG) × TOSL / VFOSL), with TOSL = 35 (`MCL!D39:D41`, `E7`).
- PM_raw = MAX(V × TRP, V × TRP / VFPM), with V = (EL × PRAFL − EG × PRAFG) × PR × VFPM × (1 + GST) and TRP = 7 (`MCL!N16`, `N17`, `N40`).

PTI_k, RLWPR, PR, VFOSL and VFPM are AEMO's regional values from `RegionalData`. The calculator's load PRAF divides by the sum of row `ELTI,R` and weights by row `ELTI,M,R`. Both rows are filled with L_k.

The squared LWPR raises the PRAF of a load that buys at above-average prices. On credit energy, PRAFG is usually LWPR itself, because solar exports earn below-average prices. When VEL < VEG, the /VF form is the larger. A net credit is then counted without its volatility factor.

`calculator_praf` uses Excel's ROUND (`xl_round`: half away from zero on the decimal value), not Python's `round`. The two differ on values such as 1.005.

### Method 2: CLP v10.0 (`clp_v10`)

CLP v10.0 removes the PRAF and values energy per time-of-day segment. The five segments are 00:00–05:59, 06:00–09:59, 10:00–15:59, 16:00–19:59 and 20:00–23:59 market time (cl 8). For segment s:

- ED_s = Σ_{k∈s} L_k and EC_s = Σ_{k∈s} G_k.
- PR,s = PR × mean_{k∈s}(PTI_k) / RLWPR.
- VED − VEC = Σ_s (ED_s − EC_s) × PR,s × VF × (1 + GST), with VF = VFOSL for the OSL and VFPM for the PM.
- OSL = TOSL × MAX(VED − VEC, (VED − VEC) / VFOSL), with TOSL = 21 days (cl 4.3.1(b), cl 5).
- PM_raw = TRP × MAX(VED − VEC, (VED − VEC) / VFPM) (cl 6, PM_R,E).

AEMO has not published PR,ToD or VF,ToD for 2026; its published 2026 parameters are regional only. PR,s is therefore built from the calculator's price profile and scaled by RLWPR, so that a participant shaped like the regional load is valued at PR, as it is in method 1. The regional VF applies to every segment.

The two methods differ through three changes:
- TOSL falls from 35 to 21 days.
- Five segments replace the 288-interval profile.
- The squared PRAF no longer applies.

The first change alone scales the OSL by 0.6.

### Floors and rounding

The unrounded MCL is PM = MAX(PM_raw, 0) and MCL = MAX(0, OSL + PM) (cl 4.3.4(d), cl 10.1(a)). The OSL may be negative but no lower than −PM. Under the MAX this is the same as the floor on the sum.

At retailer scale, `clp_round` applies cl 10.1 to N × OSL and N × PM_raw:
- OSL and PM are rounded up to the next $1,000, with PM ≥ 0 and OSL ≥ −PM.
- MCL = OSL + PM ≥ 0, rounded up to the next $10,000, or to the next $100,000 above $250,000.

Per-household figures are unrounded.

### Aggregation

- **State:** one MCL per season from the pooled profiles.
- **National:** state_mix (`analysis/config/pv_size_bands.yaml`) weights the states' OSL and PM_raw, renormalised over the states present. Equal weights apply with `--unweighted`. PM is floored after the regions are summed, as the calculator does (`MCL!I43`, `S43`), and MCL is assembled on the totals. A national retailer of N households is N times the weighted household.

  A region that is net credit contributes a negative OSL. Its credit is valued without its VF, because the /VF form is the larger for a net credit (cl 4.3.3(a)(v)). That negative OSL offsets the other regions' OSLs, and the floor MCL ≥ 0 limits the offset to their total. A national retailer can therefore have an MCL of 0 while some of its regions, taken alone, have a positive MCL.
- **Annual:** seasonal values are weighted by days per season. The weights are 121, 153 and 91 days (`SEASON_DAYS`), which reproduce ACIL Allen's "Average MCL". The annual retailer MCL weights the rounded seasonal MCLs.
- **Guarantee cost:** `--guarantee-rate` × annual retailer MCL, per year. The default is 2.5%, from ACIL Allen DMO 2026-27 final, p. 79: "The cost of funding a bank guarantee … is assumed to be a 2.5 percent annual charge". ACIL Allen spreads the MCL over 42 days of purchases and charges 2.5% × 42/365 on it. That is the same as 2.5% a year on an MCL held continuously.

### Simulated outstandings

The daily settlement amount owed to AEMO is:

S_D = Σ_{t∈D} RRP_t / 1000 × (d_t − c_t) × (1 + GST)

- D is a trading day, taken as the market-time calendar day of interval start.
- RRP_t is the historical regional price.
- GST applies to energy in both directions (cl 4.3.1(c)(ii)).

Billing weeks run Sunday to Saturday. A week is paid `PAYMENT_LAG_DAYS` = 12 days after its Saturday, on the Thursday. This is the lag of 46 of the 52 weeks in `NEM_CSV_Calendar_2027.csv`; the rest are 13–17 days around public holidays.

The amount outstanding at the end of day D, after that day's payments, is the sum of S over days whose week is not yet paid. It cycles between 12 and 18 days of trading.

Daily amounts are joined across consecutive years before outstandings are computed (`chained_outstandings`). A day is reported only if every unpaid day falls inside the data. The first reported day of each run of consecutive years is therefore the payment day of the week containing the day before the run starts.

For each cohort, state and season, the script reports the mean, 98th percentile and peak of the outstanding amount, pooled over years. It also reports the share of days on which the outstanding amount exceeds each method's MCL. National outstanding amounts are the state_mix-weighted daily amounts on dates common to all states.

These are historical prices set against an MCL built from 2026 parameters. The comparison shows how the 2026 MCL would have covered a cohort's actual exposure in 2021–2025.

### Credit support required

The credit support required on day D, per household, is:

CS_D = max(MCL, PM + O_D)

- O_D is the simulated amount outstanding at the end of day D.
- MCL and PM are the values of one method for D's season, computed from the flows of D's calendar year (`prudential_state_year.csv`), with PM floored at 0. Both methods are computed; the daily charts use the calculator method.
- The first term is the minimum lodgement (cl 10). The second keeps O_D below the trading limit, CS − PM (cl 12).

For each cohort, state, year and season, the credit support required is the peak of CS_D, which is max(MCL, PM + peak O_D). The `binding` column records which term sets it. Annual figures weight the seasons by `SEASON_DAYS`, and the 2021–2025 annual figure is the mean of the annual figures.

The maximum credit support required over the period (`season` = `max`) is the largest seasonal requirement. It is at least the largest seasonal MCL, because the requirement is at least the MCL in every season. `above_mcl` is the difference between the two, and `peak_when` names the season and year of the largest requirement. Its guarantee cost is the rate applied to that maximum held for a whole year, which is an upper bound.

At retailer scale, the required amount is max(MCL_r, PM_r + N × peak O_D), where MCL_r and PM_r are rounded per cl 10.1 and the outstanding amount is not rounded. The guarantee cost is `--guarantee-rate` × the annual retailer requirement.

National figures use state_mix weights:
- the per-year national MCL and PM come from `national_rows` on the year rows;
- the national O_D is the weighted daily amount on dates common to all states.

The outstandings use each year's actual prices, while the MCL uses 2026 parameters. Where prices ran well above the 2026 PR, as in winter 2022, the trading-limit term reflects that price gap as well as the cohort's own risk.

## Interface

Command line:

```
python3 analysis/scripts/prudential_analysis.py --data-root "/Volumes/Manual/Thesis Project"
    [--output-dir analysis/output/prudential] [--reference-dir docs/references/aemo_prudential]
    [--states ...] [--years ...] [--battery-kwh 20] [--households 50000]
    [--guarantee-rate 0.025] [--unweighted] [--weights-file analysis/config/pv_size_bands.yaml]
```

Outputs in `--output-dir` (dollars are $ nominal, energy is per household):

- **`prudential_state_year.csv`** has one row per (`cohort`, `state`, `year`, `season`, `method`). Columns:
  - `days`, `el`, `eg` (MWh/day);
  - `prafl`, `prafg` (NaN for `clp_v10`);
  - `osl`, `pm_raw`, `mcl` (unrounded).
- **`prudential_summary.csv`** has years pooled and one row per (`cohort`, `state` incl. `National`, `season` incl. `annual`, `method`). Columns:
  - `el_kwh_day`, `eg_kwh_day`, `prafl`, `prafg`;
  - `osl`, `pm`, `mcl` per household, unrounded;
  - `households`;
  - `osl_retailer`, `pm_retailer`, `mcl_retailer` (CLP-rounded per season; day-weighted on `annual` rows);
  - `guarantee_cost_retailer`, `guarantee_cost_hh` ($/yr, `annual` rows only);
  - `cohort_label`.
- **`prudential_outstandings.csv`** has one row per (`cohort`, `state` incl. `National`, `season`). Columns:
  - `days`;
  - `outstanding_mean`, `outstanding_p98`, `outstanding_max` ($ per household);
  - `mcl_calculator`, `mcl_clp_v10`;
  - `share_days_above_calculator`, `share_days_above_clp_v10`.
- **`prudential_credit_support.csv`** has one row per (`cohort`, `state` incl. `National`, `method`, `year`, `season`), plus `annual` rows per year and `year` = `mean` rows. Columns:
  - `days`;
  - `mcl`, `pm`, `peak_outstanding`, `required` ($ per household);
  - `binding` (`mcl` or `trading_limit`, seasonal rows);
  - `mcl_retailer`, `required_retailer`, `households`;
  - `guarantee_cost_retailer`, `guarantee_cost_hh` ($/yr, `annual` rows only).
- **`prudential_credit_support_{REGION}.png`** uses the calculator method. It has one figure per state and `National`, with one panel per cohort. Each panel shows daily PM + O_D, the seasonal MCL as a step line, and shading where PM + O_D exceeds the MCL. Values are per household, with a secondary axis for the retailer in $m.
- **`prudential_credit_support_max.png`** is the summary chart for the calculator method, and `prudential_credit_support_max_clp_v10.png` is the same chart for CLP v10.0. For each region and cohort, the largest formula MCL over the period is drawn solid, and the requirement above it (`above_mcl`) is stacked on top, hatched. Values are per household, with a secondary axis for the retailer in $m.
- In `prudential_credit_support.csv`, the `max` rows add `above_mcl` and `peak_when`.

Functions (all `$` values in dollars, energy in MWh unless stated):

| Function | Contract |
|---|---|
| `load_regional_params(ref_dir)` | Returns `(state, season) -> RegionalParams(p, rlwp, vf_osl, vf_pm, phh[288], erl[288])` for NSW, QLD, SA and VIC. |
| `calculator_praf(profile, rp, default)` | Calculator PRAF, as above. |
| `calculator_values(el, eg, prafl, prafg, p, vf_osl, vf_pm, tosl=35)` | `{osl, pm_raw}`. |
| `mcl_calculator(load, gen, rp)` | `{el, eg, prafl, prafg, osl, pm_raw}`. `load` and `gen` are the 288-interval mean-day profiles. |
| `mcl_clp(load, gen, rp)` | The same keys, with PRAFs NaN. |
| `tod_segment_of_ti()`, `tod_prices(rp)` | Segment of each TI (0..4) and PR,s. |
| `assemble_mcl(osl, pm_raw)` | The unrounded MCL. |
| `clp_round(osl, pm_raw)` | `(osl, pm, mcl)` rounded per cl 10.1. |
| `split_energy(net_kwh, import_lf, export_lf)` | `(debit, credit)` in kWh, both ≥ 0. |
| `season_ti_sums(energy_kwh)` | `season -> (per-TI sums in MWh, days)`. |
| `daily_settlement(debit, credit, rrp)` | $ per trading day. |
| `outstandings(daily, lag_days=12)` | $ outstanding per valid day. |
| `chained_outstandings(daily, lag_days=12)` | `outstandings` over each run of consecutive days. |
| `credit_support_daily(out_days, year_rows, method='calculator')` | Daily rows with `mcl`, `pm`, `required` and `method`. Raises `CheckError` when a day has no MCL. |
| `credit_support_table(cs, households, guarantee_rate)` | `prudential_credit_support.csv`. |
| `round_up_mcl(total)` | cl 10.1(b) step rounding. |
| `national_rows`, `retailer_table`, `outstanding_stats` | The aggregations above. `national_rows` groups by year when the rows have one. |

## Configuration

| Constant | Value | Source |
|---|---|---|
| `CALC_TOSL` | 35 days | Calculator `MCL!E7`; checked on load |
| `CLP_TOSL` | 21 days | CLP v10.0 cl 4.3.1(b) |
| `TRP` | 7 days | CLP v10.0 cl 4.3.1(b); calculator `MCL!F7`, checked |
| GST | `model.components.billing.GST_RATE` (0.10) | Calculator `MCL!G7`, checked |
| `PRAF_DEFAULT_LOAD`, `PRAF_DEFAULT_GEN` | 1.05, 0.95 | Calculator `ParticipantData!L9`, `L12` |
| `TOD_SEGMENTS` | 0–6, 6–10, 10–16, 16–20, 20–24 h | CLP v10.0 cl 8 |
| `SEASON_MONTHS`, `SEASON_DAYS` | Dec–Mar / Apr–Aug / Sep–Nov; 121 / 153 / 91 | CLP v10.0 cl 4.3.1(a) |
| `PAYMENT_LAG_DAYS` | 12 | `NEM_CSV_Calendar_2027.csv` modal lag |
| `DEFAULT_HOUSEHOLDS` | 50,000 | `--households` |
| `DEFAULT_GUARANTEE_RATE` | 0.025 per year | ACIL Allen DMO 2026-27 final, p. 79; `--guarantee-rate` |

Reading the calculators requires `openpyxl`.

## Failure handling

`CheckError` stops the run (exit 1) in these cases:
- a calculator is missing;
- a calculator's `RegionalData` title is not that season of 2026;
- TOSL, TRP or GST differ from the constants;
- an `ERLTI,R` row is out of the expected state order;
- a profile is incomplete;
- RLWPR differs by more than $0.01/MWh from the load-weighted PTI profile it summarises;
- a season has a day without 288 intervals;
- an interval has no price;
- a cohort's flows and bills share no sites;
- the daily settlement series has a gap or a duplicate day;
- a day of outstandings has no calculator MCL for its cohort, state, year and season.

A missing `flows.parquet` or `bills.csv` for a cohort and scenario is skipped with a warning.

## Verification

`model/tests/test_prudential.py` checks the following properties:

- **Calculator equivalence** (`@pytest.mark.slow`; skipped without LibreOffice or the calculators):
  - Household profiles are entered into AEMO's three calculators for NSW, QLD, SA and VIC, and LibreOffice recalculates them.
  - Per region, `mcl_calculator` reproduces the workbook's PRAFL and PRAFG exactly and OSL and PM within $0.01.
  - The four-region total reproduces `MCL!J11`.
  - Both a net-debit and a net-credit case run; the net-credit case asserts that the OSL is negative.
- **ACIL Allen Table 3.14:** from its inputs, `calculator_values` gives each season's OSL, PML and MCL to $1. The `SEASON_DAYS` weighting gives the average MCL of $10,148.
- **PRAF:** the defaults with no energy; the square above 1 and the plain ratio below 1; Excel half-away-from-zero rounding.
- **PRAF on the regional load shape:** a profile shaped like the regional load has PRAF 1 in every region and season. This also checks that load and price rows are paired by region.
- **Regional parameters:** the parsed values equal the published 2026 values.
- **Net credit:** OSL and PM are divided by their VF.
- **CLP method:**
  - the segments cover 72/48/72/48/48 TIs with the right boundaries;
  - a flat profile is valued at PR × mean(PTI)/RLWPR;
  - a hand-built case with load in one segment and credit in another follows cl 5 and cl 6 with TOSL 21.
- **Rounding** follows cl 10.1, including OSL ≥ −PM, the PM floor and the $250,000 step change. `assemble_mcl` floors PM before adding.
- **Energy:**
  - debit and credit take their own loss factors and signs;
  - per-TI sums align TI 1 with 00:00–00:05 and assign 24:00 on 31 March to summer;
  - a missing interval raises.
- **Outstandings:**
  - on $1 a day, the first valid day, the 12–18 day cycle and the payment on Thursday 12 days after Saturday are exact;
  - a mid-week start delays the first valid day;
  - a gap raises;
  - daily settlement applies GST to both legs and dates by interval start;
  - joining two years gives the same series as one continuous series, January included;
  - a missing year starts a new run.
- **Credit support:**
  - required = max(MCL, PM + outstanding), using the calculator rows and the PM floored at 0;
  - the table's peak, `binding` and retailer requirement (rounded PM plus N × peak) are correct;
  - annual rows use `SEASON_DAYS` weights, the `mean` row averages the years, and the guarantee is costed on the requirement;
  - the `max` row takes the largest MCL and the largest requirement, which can fall in different seasons, along with `above_mcl`, `peak_when` and the guarantee cost on the maximum;
  - a day without an MCL raises;
  - national rows are formed per year.
- **Aggregation:**
  - national PM is summed before it is floored, renormalised over the states present;
  - the retailer table rounds per season, weights annual figures by `SEASON_DAYS` and costs the guarantee on the annual retailer MCL.
- **End to end:** synthetic runs for all four cohorts go through `main`. The No CER per-household and retailer MCLs equal the formulas on the hand-built mean profile with the run's loss factors, every cohort and season has outstandings, and the credit support table and charts are written for NSW and National.

## Known limitations

- **Method 2 parameters.** Method 2 uses PR,ToD and VF,ToD built from regional parameters. Fix path: when AEMO publishes per-ToD parameters under v10.0, read them in `load_regional_params` and use them in place of `tod_prices` and the uniform VF in `mcl_clp`.
- **Trading days.** Trading days are market-time calendar days, and payment-date shifts for public holidays are ignored. This moves at most a day's amount between weeks. Fix path: date each settlement interval with the NEM trading-day definition, and take each week's payment date from the calendar CSVs.
- **Homogeneous retailer.** The retailer supplies one cohort only. A mixed retailer's MCL is the household-weighted sum of the cohort OSLs and PMs, with floors on the totals. This can be built from `prudential_summary.csv`.
- **Energy estimate.** AEMO estimates debit energy from history "and evident trends" (cl 9.2.1). Here it is the pooled seasonal mean of the simulated years.
- **Net-credit cohorts.** A cohort that is net credit in a season has an MCL of 0 under both methods, but its outstandings can be positive on some days. The credit support required covers this.
- **No AEMO review.** A sustained excess of accrued liabilities over credit support lets AEMO reset the OSL to the 21-day average liability (cl 9.3). The simulation keeps the formula MCL and reports the support that would have avoided a trading-limit breach.

## Validation

**Formula checks** (`test_prudential.py`, container, 16 October 2026):
- The workbook equivalence test passes for all three 2026 calculators, four regions, and the net-debit and net-credit cases.
- The ACIL Allen Table 3.14 reproduction passes.

**Full run, 2021–2025, weighted** (Fred's run, 16 October 2026). Maximum credit support required per household, given as the largest MCL plus the requirement above it:

| Cohort | NSW | QLD | SA | VIC | National |
|---|---|---|---|---|---|
| No CER | $306 + $117 | $300 + $106 | $391 + $0 | $281 + $0 | $276 + $82 |
| Solar Only | $222 + $38 | $188 + $128 | $303 + $0 | $237 + $0 | $218 + $33 |
| Self Consumption | $123 + $78 | $100 + $119 | $167 + $46 | $187 + $0 | $118 + $68 |
| Amber coordinated | $0 + $78 | $43 + $0 | $74 + $0 | $31 + $0 | $4 + $11 |

Every requirement above the MCL falls in winter 2022. The largest MCLs in SA and VIC are in winter 2021.

The coordinated cohort has a positive MCL in only a few region-seasons: QLD summer 2021 and 2025, and SA and VIC winter 2021. In each of these the other regions are net credit, so the national MCL is $4 in summer 2021 and $0 otherwise.

The two methods are not close. In the full run's pooled annual MCL, CLP v10.0 gives 29–63% less than the calculator for No CER, Solar Only and Self Consumption, mostly from the shorter TOSL. For Amber coordinated it gives $6–11 per household where the calculator gives $0, because its five time-of-day segments value the batteries' exports at segment-average prices. The calculator's PRAF instead credits the high prices they export into. Results are presented with the calculator method; CLP v10.0 is kept for comparison.
