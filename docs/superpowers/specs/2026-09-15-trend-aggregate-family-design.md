# Trend Aggregate Family Design (directional)

Status: Decided autonomously on 2026-09-15 under the standing goal "make it better, decide
autonomously until a reliable solution exists", after the user asked for directional
("rising or falling prices") bets. `research_only`. Pre-registers experiment family
`trend_aggregate_panel_v1` under `PHASE_0_EVALUATION_PROTOCOL.md` sections 1 to 16 as they
stand on 2026-09-15, before any fold is run. No live, paper, or shadow authority follows.

## 1. Purpose

Two directional families were rejected: P1.15 (15-minute features, 1h/4h horizons, BTC)
and P1.27 (weekly cross-sectional and time-series momentum over 860 perpetuals; best
member `ts_mom_4w` won 51.6% of weeks, p 0.18, 60% of PnL in one fold). Both records name
the same next question: a **CTREND-style aggregate** — many simple technical signals
combined into one trend score — which the 2025 comparative literature reports as
materially stronger than plain momentum and as rendering plain momentum insignificant in
its presence. This family asks that question on the P1.27 panel with the P1.27 gates,
holding everything except the signal fixed.

The honest prior is low: costs bound P1.15, P1.27, P1.28 and P1.29. The family is worth
running because it is the last *price-only* directional question the protocol's ordering
leaves open, and because its answer decides whether directional work continues on price
data or moves to a different information source.

## 2. What was seen before this design (disclosure)

P1.27's per-member results (all six rejected; time-series beat cross-sectional; a 4-week
lookback beat 1 and 12 weeks; turnover was the binding cost). The literature's design
(a vote over trend indicators at several horizons) was chosen because of that literature,
not tuned to P1.27's folds; but the choice to include a 4-week-hold member and to weight
the time-series book like P1.27's `ts_mom` members is informed by P1.27's record. The
holdout from 2026-03-08 stays locked and was never read by any fold.

## 3. Signal

For contract *c* at decision *t* (Sunday close), on its daily close series `P`:

**Indicators** (each a vote in {−1, 0, +1}, computed from data at or before *t*, each
requiring its own full window or abstaining with 0):

1. `ma_20`: sign(P_t − SMA_20)
2. `ma_50`: sign(P_t − SMA_50)
3. `ma_100`: sign(P_t − SMA_100)
4. `ma_cross_20_50`: sign(SMA_20 − SMA_50)
5. `ma_cross_50_100`: sign(SMA_50 − SMA_100)
6. `breakout_20`: +1 if P_t equals the 20-day high, −1 if the 20-day low, else 0
7. `breakout_50`: same over 50 days
8. `breakout_100`: same over 100 days
9. `roc_20`: sign(P_t / P_{t−20} − 1)
10. `roc_60`: sign(P_t / P_{t−60} − 1)
11. `macd_12_26`: sign(EMA_12 − EMA_26)
12. `roc_120`: sign(P_t / P_{t−120} − 1)

**Score** `S_c(t)` = mean of the twelve votes (in [−1, 1]); a contract with fewer than
twelve computable votes has no score and is not rankable.

**Windows are calendar spans** (P1.27 protocol section 16.1, inherited): an indicator over
*n* days needs a close on every one of its calendar days ending at *t* (SMA and breakout:
the *n* days; ROC: the *n + 1* days from *t − n* to *t*; MACD: the score's whole 121-day
span, after which both EMAs run over the contract's whole observed series). An indicator
whose span has a missing day is absent, so the contract has no score at that decision —
equivalently, a contract is scored only when its last 121 calendar days are complete.
Clarified 2026-09-15 before any fold ran.

All arithmetic is `Decimal`; EMAs use the standard `α = 2/(n+1)` recursion seeded with
the first close of the contract's history; signs of an exact zero difference are 0.

## 4. Family declaration

Everything not stated here is identical to `xs_momentum_panel_v1`
(`docs/superpowers/specs/2026-09-08-xs-momentum-family-design.md` sections 6–9): same
capture (`data/captures/2026-09-10-binance-um-usdt-perps-1d-repaired`), same universe rule
(91 bars, 30-day complete window, 5,000,000 USDT median, top 100, minimum 40, tier one =
ranks 1–20), same weekly calendar and folds, same cost tables (fee 5, slippage 5/10 →
10/20, funding receipts ×0 and payments ×2 under adverse, forced close ×1 → ×2), same
statistics and gates, same `panel_decision.py`.

### 4.1 Hypothesis

A trend score aggregated from twelve fixed technical indicators over 20 to 120 days
predicts the sign of the next week's return on a liquid USDT-perpetual universe well
enough that a weekly-rebalanced book built on it earns positive net expectancy under
base costs, non-negative under adverse costs, in enough folds, without single-fold,
single-contract or single-week concentration, after correction across four members.

### 4.2 Members (four trials, each fixed)

| Member | Kind | Rule | Hold |
| --- | --- | --- | --- |
| `ta_ts_t02` | time-series | long every contract with `S ≥ 0.2`, short every contract with `S ≤ −0.2`, flat otherwise | 1 week |
| `ta_ts_t05` | time-series | as above with threshold 0.5 | 1 week |
| `ta_ts_t02_h4w` | time-series | threshold 0.2, positions held four weeks in overlapping cohorts of 1/4 capital each (P1.28's cohort mechanics with `formed_size`) | 4 weeks |
| `ta_xs_q5` | cross-sectional | long the top quintile by `S`, short the bottom quintile, equal weight within each leg, dollar-neutral | 1 week |

Time-series weights follow P1.27's `ts_mom` construction: inverse-volatility weights
(30-day realised, calendar-complete window), water-filled at a cap of
`time_series_cap_numerator / n` (2/n, P1.27's declared value), scaled so gross exposure is
1 when at least one contract has a signal; a member with no signalled contract holds
nothing (`MEMBER_HELD_NOTHING`). The cross-sectional member uses the P1.27 quintile
construction (minimum quintile size 8, at most 100 contracts). The numbers are the frozen
P1.27 declaration's, copied byte-for-byte into `configs/trend-aggregate-panel-v1.json`.

### 4.3 Controls (not trials)

`no_trade` and `random_ranks` (dominance), `passive_long_ew` (context only) — P1.27's,
unchanged.

### 4.4 Primary metric, budget, gates

P1.27's section 5.4/5.5 and section 9 unchanged: mean net return per retained weekly
episode under base costs; block bootstrap 2000 × block 4 × seed 17; BH q ≤ 0.10 over four
members; pooled floor 200; adverse ≥ 0; 2/3 positive folds; concentration ≤ 0.5;
dominance over both controls; deflated Sharpe reported only.

## 5. Modules

- `trend_signals.py`: indicators, votes, `trend_score(history, decision_close_ns) ->
  Decimal | None`, member weight builders for the three kinds (reusing `panel_signals`'
  water-filling cap and quintile helpers where they exist, otherwise copies with the same
  semantics and tests pinning equality).
- `trend_config.py`: `TrendFamilySpec` (frozen members/controls/indicators list), loaded
  through `panel_config.load_family_spec` (dispatch on `family_name`).
- `trend_fold_run.py`: one fold; the P1.27 report schema (so `panel_decision.py` pools it)
  plus per-episode extras `signalled_contracts`, `abstained_contracts`, `mean_score`.
  The four-week member's book at *t* is the sum of the last four weekly weight vectors
  (its own included) divided by four, aged by calendar; an empty vector contributes nothing
  (capital undeployed); a contract without a bar at *t* is dropped; an in-window
  `UNIVERSE_TOO_SMALL` week resets the retained vectors and the carried position (P1.27).
  **Warm-up (declared 2026-09-15, before any fold):** the three Sundays before each fold's
  first test decision form weight vectors only — no episode, nothing carried — so the
  four-week member opens on a full book instead of a one-quarter stub (the P1.29 rule);
  a warm-up Sunday whose universe is too small contributes nothing and resets nothing.
  Those Sundays fall inside the embargo/validation span and read only data at or before
  themselves. The fold report carries `warm_up_weeks: 3` and the reason code
  `FOLD_OPENING_BOOK_WARMED_FROM_PRIOR_WEEKS`; the decision record must repeat both.
  Contracts a cohort still holds after leaving the eligible universe are charged tier-two
  slippage on exit (`panel_accounting`'s default), which is disclosed in the record.
- `cli.py`: `trend-fold`. Manifest: `panel-manifest` with the trend family spec.

## 6. Artifacts

`configs/trend-aggregate-panel-v1.json`; `artifacts/trend/trend-walk-forward-<date>-usdt-perps-1d-w1-v1.json`,
`artifacts/trend/trend-aggregate-<date>-fold<i>-v1.json`, `artifacts/trend/trend-aggregate-<date>-decision-v1.json`,
registry `artifacts/trend/metadata-trend-aggregate-v1.sqlite3`; decision record
`P1_31_DECISION_<date>.md`; backlog P1.31.

## 7. Decision rule

A passing member becomes the first directional candidate with an edge and the base over
which a learned gate may later be declared. A fail closes price-only directional work in
this repository: the next directional declaration must bring a new information source
(order-book, funding/basis regime, on-chain), not a new transformation of the same closes.
No threshold moves either way. Sequencing: this family is independent of P1.30 (different
signal, different data leg) and may run before it; the two decision records are written on
different days.
