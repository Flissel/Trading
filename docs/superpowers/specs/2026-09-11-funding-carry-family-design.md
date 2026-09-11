# Funding Carry Family Design

Status: Draft for review on 2026-09-11. `research_only`. Not approved for
implementation. Pre-registers experiment family `funding_carry_panel_v1` under
`PHASE_0_EVALUATION_PROTOCOL.md` sections 1 to 16 as they stand on 2026-09-11.
No live, paper, or shadow authority follows from it.

## 1. Purpose

P1.27 (`P1_27_DECISION_2026-09-10.md`) rejected every pre-registered momentum
member on a weekly panel of 860 USDT perpetuals, and named mechanical funding
carry as the next question in order of prior. This family asks it: does a
delta-hedged position that is long spot and short the perpetual, selected on
recently paid funding, earn the funding it collects after both legs' costs,
basis drift, and multiple-testing correction?

The question is different in kind from P1.15 and P1.27. Momentum needed a
directional forecast to be right often enough to pay for its own turnover;
its best member won 51.6% of weeks and could not be told from zero. Funding is
a payment the perpetual's own mechanism makes to whoever holds the short when
the contract trades rich. There is no forecast; the positive expectation is
structural, and the open question is whether execution and the hedge leg leave
any of it. That is the question this family is built to answer with the same
gates, the same fold geometry, and the same refusal to weaken a threshold after
seeing a number.

Decision rule on completion (section 12): a pass registers the passing member
as the first candidate with a positive net expectation in this repository and
opens the question of execution realism; a fail records that the only
structural premium available to a public-data solo operator does not clear
realistic costs at this scale, and the honest next step is to measure costs
rather than to search for signals.

## 2. Evidence

### 2.1 Published

Funding-rate carry on perpetual futures is documented as a persistent,
regime-dependent premium: perpetuals trade rich in bull regimes and pay the
short leg, and the premium concentrates in the highest-funding names. The
premium is not free: it is compensation for basis risk during liquidation
cascades and for the tail in which funding flips sign faster than a weekly
book can turn.

### 2.2 This repository's own data, and what it already saw

Before this design, a spike read the 2,629,011 funding settlements in the
repaired P1.27 capture (`data/captures/2026-09-10-binance-um-usdt-perps-1d-repaired`)
and measured the funding a short-perpetual leg would have collected per unit
of perpetual notional, selecting the top decile of a 40-plus-name universe by
the prior week's realised funding, then holding for one to twenty-six weeks.
Gross of the hedge leg and gross of all costs, over 316 weeks from 2020-08 to
2026-08:

| Hold | Gross bps per week held | Median | Positive weeks |
| ---: | ---: | ---: | ---: |
| 1 | 42.1 | 23.7 | 90.2% |
| 4 | 36.5 | 20.4 | 88.3% |
| 13 | 32.6 | 20.4 | 89.2% |
| 26 | 30.0 | 24.2 | 90.5% |

Two things in that table drove this design and must be stated as such.
First, the premium decays slowly with holding period while a weekly round
trip does not, so the holding period is the lever; a family that rebalanced
weekly would be testing execution cost, not carry. Second, the weekly
hit rate near 90% is the signature of a premium, not of a forecast, and it is
the reason this family is worth registering at all.

A second spike measured basis noise on the hedge leg, the difference between
the perpetual's daily close and the spot close, over 2023-01 to 2026-08:

| Perpetual | Mean basis bps | SD of 4-week basis change bps |
| --- | ---: | ---: |
| BTCUSDT | -1.6 | 5.5 |
| ETHUSDT | -1.3 | 5.9 |
| SOLUSDT | -1.3 | 7.4 |
| DOGEUSDT | -1.4 | 7.0 |
| ARBUSDT | -1.0 | 7.5 |
| GALAUSDT | -0.8 | 9.6 |
| APEUSDT | -3.5 | 11.0 |
| 1000PEPEUSDT | -0.7 | 51.0 |

On liquid names the hedge leg adds single-digit basis points of noise over a
four-week hold against roughly 145 basis points of gross carry in the same
window. On the one scaled-multiplier meme contract it adds fifty. That is why
section 7 excludes scaled-multiplier contracts and floors liquidity on both
legs.

**Disclosure of contamination.** Both spikes read the full funding and price
history through 2026-08-31, which includes the 182-day window this family
locks as its final holdout from 2026-03-08. What was seen is the aggregate
funding collected per week by hold, not any pair-level net P&L, not any cost,
and not any basis-adjusted result. The family's gates are applied to the nine
walk-forward test folds only, which end 2025-10-27, and the holdout is not
evaluated. The aggregate glance is nevertheless a contamination of the
holdout's funding aggregate and is recorded here so that a later reader does
not mistake the holdout for untouched.

### 2.3 What is new relative to the spike

Everything the spike did not measure: the spot leg and its own fees and
slippage, basis drift over the holding period, a liquidity floor on both legs,
tiered execution cost, forced closes on delisting of either leg, the adverse
scenario, and the full gate set. The spike answered "is there anything to
test"; this family answers "does it survive".

## 3. Problem Definition

### 3.1 Position and cadence

- Venue: Binance, spot market and USD-M USDT-margined perpetuals, both from
  the public dumps at `data.binance.vision`.
- A position is a **pair**: long spot notional `w` and short perpetual notional
  `w` on the same underlying, opened at the Sunday close and held `H` weeks
  through overlapping cohorts (section 8.1).
- Decision cadence: weekly at the Sunday daily close, as in P1.27. One episode
  is one weekly rebalance of the whole book; the pairs inside it are not
  independent samples.
- Return: the pair's weekly net P&L per unit of **capital deployed across both
  legs**, fully collateralised, no leverage. Funding accrues on the perpetual
  notional only, so per unit of capital it is half the funding rate. This is
  the conservative basis and the one the gates read; the per-perpetual-notional
  figure is reported alongside for comparison with the spike.

### 3.2 Data budget

- Perpetual daily bars and funding: already captured and repaired, 860
  contracts, 637,308 bars, 2,629,011 settlements at 1, 2, 4 and 8-hour
  intervals. Funding is summed per settlement over the holding window; no
  interval assumption is made.
- Spot daily bars: not yet captured. Of the 860 perpetuals, 467 have a
  same-symbol spot pair in the bucket and 7 more a spot pair under a 1000x or
  1,000,000x multiplier; 386 have no spot leg and are out of scope. Spot daily
  klines carry the same twelve-column schema as the perpetual ones. Estimated
  capture: 474 symbols across their listed months, roughly 24,000 sources,
  about five hours at the measured rate, well under 200 MB.
- Both captures go through the same quality gate and the same daily-dump
  repair as P1.27.

### 3.3 Costs, declared

Binance's standard tier without BNB discount, verified 2026-09-11 against the
published schedule: spot taker 10 bps per side, USD-M futures taker 5 bps per
side. Slippage per side per leg by liquidity tier as in P1.27: 5 bps for tier
one and 10 for tier two in base, doubled in adverse. A pair's tier is the
worse of its two legs' tiers.

Funding in base is the actual settlement stream. Adverse applies a 25% haircut
to receipts, reflecting partial fills and settlement-timing slippage, and
doubles payments. This differs from P1.27's adverse rule, which zeroed
receipts, because zeroing the payment a carry strategy exists to collect does
not stress the strategy, it deletes it. The protocol's section 7.3 asks for
funding stress that is unfavourable, and a one-quarter haircut on receipts
plus doubled payments is that.

## 4. Design Options

### Option A — synthetic pair as a single instrument through the P1.27 pipeline

Rejected. The panel accounting charges cost on a single close series and
cannot represent two legs with different fees, different liquidity, and a
funding stream on one side only. Forcing a pair into it would misprice every
episode.

### Option B — new carry accounting emitting the P1.27 report schema (recommended)

Build a small carry-specific accounting module and fold runner, and have them
emit fold reports in exactly the schema `panel_decision.py` already pools.
Then the decision module, its gates, its declared-deviations block, and its
tests are reused unchanged, as are the manifest builder, the calendar, the
quality gate, the capture, the repair, and the statistics. New code is
confined to: a spot-market path in capture, the pair universe, the carry
selection, and two-leg accounting.

### Option C — full generalisation of the panel to n-leg instruments

Deferred. Correct in principle, but it would reopen every module P1.27
verified for a benefit no current question needs.

## 5. Family Declaration (pre-registration)

Frozen when this spec is approved and committed, and mirrored byte-for-byte in
`configs/funding-carry-panel-v1.json`, whose hash travels in every report.

### 5.1 Hypothesis

Among pairs of a USDT perpetual and its spot underlying that clear a
liquidity floor on both legs, the pairs with the highest realised funding over
a trailing lookback continue to pay funding over the following holding period
in excess of both legs' execution costs and basis drift, with positive net
expectancy under base costs and non-negative under adverse costs, not
concentrated in one fold, one pair, or one week.

### 5.2 Members (three trials, each fixed, no validation selection)

| Member | Lookback L | Hold H | Construction |
| --- | ---: | ---: | --- |
| `carry_l1w_h4w` | 1 week | 4 weeks | top decile of eligible pairs by trailing L-week realised funding, positive funding required; equal weight; overlapping cohorts of 1/H |
| `carry_l4w_h4w` | 4 weeks | 4 weeks | same |
| `carry_l4w_h13w` | 4 weeks | 13 weeks | same |

The first member is the spike's construction and is declared as such. The
second replaces the one-week lookback with a smoother one at the same hold.
The third is the long hold. No member has a fitted parameter.

### 5.3 Controls (not trials, not promotable)

- `no_trade`: zero weights.
- `random_pairs`: a random decile of eligible pairs each week under the same
  cohort mechanics as `carry_l1w_h4w`, `random_seed = 17`. Dominance control.
- `all_pairs_ew`: equal weight over every eligible pair with positive
  trailing one-week funding, same cohorts. Context only: it shows what the
  premium pays without selection.

### 5.4 Primary metric and direction

Mean net return per weekly episode under base costs, pooled over all test
folds, per unit of capital across both legs. Direction positive. Secondary
requirements identical to P1.27: adverse pooled mean non-negative; base net
positive in at least two thirds of folds; 95% moving-block bootstrap lower
bound above zero, block length 4, 2,000 repetitions, seed 17;
Benjamini-Hochberg q at most 0.10 within this family of three; no fold, pair,
or episode above half of pooled base net P&L; dominance over the strongest of
`no_trade` and `random_pairs` under both scenarios. Deflated Sharpe reported
over three trials.

### 5.5 Budget

Two attempts, as in P1.27. Attempt 1 needs a revision reason independent of
attempt 0's test results.

## 6. Data and Capture

The perpetual capture is the repaired P1.27 capture, bound by hash. The spot
capture reuses `panel_capture.py` with a market parameter: the kline URL
prefix becomes `data/spot/monthly/klines/` and `data/spot/daily/klines/` for
repair, the venue string becomes `BINANCE_SPOT`, and no funding is fetched.
Discovery, retry, resume, parameter binding, absent-source recording,
quality report and repair all apply unchanged. Capture name:
`data/captures/<date>-binance-spot-usdt-1d`.

Symbol mapping: a perpetual `XUSDT` maps to spot `XUSDT`; a perpetual
`1000XUSDT` or `1000000XUSDT` maps to spot `XUSDT` with the stated multiplier.
The mapping is recorded in the family declaration as a fixed list generated
from the bucket listing on 2026-09-11, so the universe cannot drift between
runs.

## 7. Universe and Identity

### 7.1 Eligibility of a pair at decision time `t`

All of the following, using only bars whose close time is at or before `t`:

1. both legs have at least 91 daily bars closing at or before `t`;
2. both legs have a bar closing exactly at `t`;
3. both legs have a complete 30-calendar-day window ending at `t` with no
   missing day, and the median quote volume over that window is at least
   5,000,000 USDT on each leg;
4. the perpetual has at least one funding settlement inside the trailing
   lookback window;
5. the pair is not a scaled-multiplier pair. The seven such pairs are excluded
   by declaration because their measured basis noise is an order of magnitude
   above the rest (section 2.2); they are listed in the declaration so the
   exclusion is auditable.

Rank survivors by the perpetual's 30-day median quote volume, keep the top
100, and assign tier one to ranks 1 to 20 and tier two beyond, exactly as
P1.27. A pair's cost tier is the worse of its perpetual tier and the tier its
spot leg would hold by the same rule applied to spot volume.

Fewer than 40 eligible pairs: `UNIVERSE_TOO_SMALL`, no position, excluded
from the pooled series as in P1.27.

### 7.2 Identity

Each leg keeps the contract identity rule of P1.27, `symbol:first_open_time`.
A pair's identity is the ordered pair of its legs' identities. A recycled
ticker on either leg produces a new pair.

## 8. Portfolio Construction and Accounting

### 8.1 Selection and cohorts

At decision `t`, compute each eligible pair's trailing realised funding as the
sum of the perpetual's settlements in `(t − L weeks, t]`. Keep pairs whose
trailing funding is strictly positive. If fewer than 8 remain, the cohort
formed at `t` is empty and is recorded as `NO_CARRY_COHORT`; the book still
carries its older cohorts. Otherwise select the top decile, at least 8 pairs,
and give each pair an equal share of the cohort's capital.

The book at any week is the union of the last `H` cohorts, each holding `1/H`
of capital. A pair present in several cohorts holds the sum of its shares.
This is the standard overlapping-portfolio construction: every week is a
rebalance of `1/H` of the book, every week is an episode, the weekly episode
grain and the 200-episode floor are preserved, and turnover is charged on the
book's actual weekly change through the existing turnover mechanism rather
than assumed.

### 8.2 Pair return

For a pair with capital share `c` held over week `k` to `k + 1`, with
perpetual close `P` and spot close `S` adjusted by the multiplier:

```text
spot_leg     =  c/2 × (S[k+1] / S[k] − 1)
perp_leg     = −c/2 × (P[k+1] / P[k] − 1)
funding      =  c/2 × Σ settlements in (close[k], close[k+1]]   (received when positive)
gross        =  spot_leg + perp_leg + funding
```

`spot_leg + perp_leg` is the basis change, the hedge leg's noise. Prices are
Decimal from canonical strings; the Decimal precision contexts and the
Inexact traps of P1.27 apply.

### 8.3 Costs

Charged on turnover per leg with each leg's own fee and its own tier
slippage, using the existing drifted-weight turnover: each leg's weight
drifts with its own return, and the cost is `Σ |w_new − w_drifted| × (fee +
slippage(tier)) / 10,000` per leg. Funding under adverse applies the section
3.3 rule. A leg without a bar at the exit close is force-closed at its last
available close inside the week, one side of that leg's cost, and the pair's
other leg is closed with it at its own exit price, one side of its cost,
because an unhedged leg is not the position being tested.

```text
net = gross − spot_turnover_cost − perp_turnover_cost − funding_adverse_adjustment − forced_close_costs
```

Per-pair net attribution sums exactly to `net`, as P1.27 requires, so the
concentration gate reads net contributions.

### 8.4 Reported

Everything P1.27 reports, per member and scenario, plus: mean funding
collected per unit capital, mean basis P&L per unit capital, the split of
turnover cost between legs, count of `NO_CARRY_COHORT` weeks, and count of
pairs force-closed by leg.

## 9. Evaluation Geometry and Gates

Identical to P1.27: `WalkForwardConfig` over weekly rebalance samples with
365-day train, 91-day validation, 182-day test, 182-day step, 14-day embargo,
182-day final holdout locked from the end of the capture. Nothing is fitted;
train and validation exist for geometry and history. Same purge rule. Pooled
episode floor 200. Same bootstrap, same BH, same concentration, same
dominance, same reason-code vocabulary. `panel_decision.py` is reused
unchanged; the carry fold runner emits its report schema.

The walk-forward manifest binds the perpetual capture, whose calendar defines
the decisions. Each fold report binds both captures' root hashes and the spot
capture is verified before any fold runs; the decision module's linkage check
extends to the spot hash through the fold reports.

## 10. Modules and Interfaces

| Module | Job | Depends on |
| --- | --- | --- |
| `panel_capture.py` (modify) | a `market` parameter, `um` or `spot`, selecting URL prefixes and venue; no funding in spot mode | existing |
| `carry_config.py` | frozen declaration, mirror of `panel_config.py` with the pair mapping and exclusions | `canonical` |
| `carry_universe.py` | pair eligibility, two-leg liquidity, tiering | `panel_universe.build_contract_histories`, `panel_reader` |
| `carry_signals.py` | trailing funding, cohort selection, book assembly, controls | `carry_universe` |
| `carry_accounting.py` | two-leg episode P&L with per-leg costs, forced closes, exact net attribution; emits `EpisodeResult` | `panel_config.PanelCostTable`, which already carries the funding multipliers, extended with one field, `spot_fee_bps_per_side` |
| `carry_fold_run.py` | one fold, both captures, P1.27 report schema | all above, `panel_samples.verify_panel_manifest`, `registry` |
| `cli.py` (modify) | `panel-capture --market spot`, `carry-fold` | existing |

`panel_decision.py`, `panel_samples.py`, `panel_statistics.py`,
`panel_dataset.py`, `panel_reader.py` are reused without change.

## 11. Artifacts and Naming

- Spot capture: `data/captures/<date>-binance-spot-usdt-1d`, repaired sibling `-repaired`.
- Manifest: `artifacts/carry/carry-walk-forward-<date>-usdt-pairs-1d-w1-v1.json`.
- Fold reports: `artifacts/carry/funding-carry-<date>-fold<K>-v1.json`.
- Decision report: `artifacts/carry/funding-carry-<date>-decision-v1.json`.
- Registry: `artifacts/carry/metadata-funding-carry-v1.sqlite3`.
- Decision record: `P1_28_DECISION_<date>.md` in the P1.15 template.
- Backlog: `### P1.28 Funding carry family`.

## 12. Sequencing, Stop Conditions, and Decision Rule

1. Approve and commit this spec with the frozen declaration before the spot
   capture is started.
2. Capture spot, repair, verify. Stop on `CAPTURE_QUALITY_FAILED` after
   repair; fix the capture, do not touch the family.
3. Build the manifest. Stop with `INSUFFICIENT_EVIDENCE` below 200 pooled
   episodes.
4. Run all three members and three controls on every fold in one invocation
   per fold.
5. Decision report, then `P1_28_DECISION_<date>.md`.

Decision rule:

- **A member is eligible.** It becomes the repository's first registered
  candidate with positive net expectation. The next question is execution
  realism: measured spot and perpetual spreads on the selected names, fill
  behaviour, and the capital efficiency of the hedge under real margin. No
  live authority follows; the protocol's shadow and paper sections apply.
- **No member is eligible.** Record it. The structural premium does not clear
  realistic costs at this scale on public daily data, and the honest next step
  is to measure costs on the names carry would have selected, not to search
  for another signal.
- **`INSUFFICIENT_EVIDENCE`.** Extend the capture by a registered revision.

Prohibited: changing the lookback set, the hold set, the decile, the liquidity
floor, the exclusion list, or the cost table after seeing test results;
opening the holdout; reporting the best member without the other two.

## 13. Risks and Honest Priors

- **Capital basis halves the spike.** Per unit of fully collateralised
  capital across both legs, the spike's 36.5 gross basis points per week at a
  four-week hold become about 18, against a base cost of roughly 6 per week at
  that hold once spot fees are included. Positive, and thin; the adverse
  scenario will decide it.
- **Funding regime.** The all-contract mean funding annualised was negative in
  2022, 2025 and 2026 even though most settlements were positive. The top
  decile stayed positive in the spike, but the left tail is heavy: the worst
  single week of the spike's decile book lost 126 basis points of perpetual
  notional. The concentration gate and the fold-positivity gate exist for this.
- **Basis during stress.** Section 2.2 measured basis noise in ordinary weeks.
  Liquidation cascades widen basis precisely when funding is highest. Forced
  closes and the adverse slippage capture part of this; the rest is the
  premium's price.
- **Spot liquidity may thin the universe.** The 5,000,000 USDT floor applies to
  the spot leg too, and spot volume on smaller alts is often below the
  perpetual's. If fewer than 40 pairs clear both floors in a week, that week is
  `UNIVERSE_TOO_SMALL` and drops from the pooled series; enough such weeks
  would leave the family at `INSUFFICIENT_EVIDENCE`, which is the intended
  outcome rather than a reason to lower the floor.
- **Spot delisting.** A spot pair can be delisted while the perpetual
  continues, or the reverse. Both are forced closes of the pair.
- **Contamination.** Section 2.2's disclosure stands: the aggregate funding
  by hold was seen through 2026-08.

## 14. Non-Goals

- No leverage modelling, no margin engine, no liquidation simulation; the
  capital basis is fully collateralised by declaration.
- No cross-exchange basis, no OKX leg; OKX's cost journal informs execution
  realism later, not this family.
- No dynamic hold, no funding-rate forecasting, no signal beyond realised
  trailing funding.
- No change to P1.27's modules beyond the capture's market parameter.
- No shadow, paper or live authority; no holdout opening.

## 15. Sources

- Binance fee schedule, standard tier, spot taker 0.10%, USD-M taker 0.05%, verified 2026-09-11:
  [tradersunion.com](https://tradersunion.com/brokers/crypto/view/binance/fees/),
  [bitget.com](https://www.bitget.com/academy/binance-fees-2026),
  [feeflux.com](https://feeflux.com/en/articles/binance-fees-guide/).
- Binance public data, spot and USD-M daily klines and funding, verified 2026-09-11.
- `P1_27_DECISION_2026-09-10.md`; `docs/superpowers/specs/2026-09-08-xs-momentum-family-design.md`;
  `PHASE_0_EVALUATION_PROTOCOL.md` sections 7, 9 and 16.
