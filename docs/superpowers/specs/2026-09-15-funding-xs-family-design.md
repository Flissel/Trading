# Funding Cross-Section Family Design (perpetual-only)

Status: Decided autonomously on 2026-09-15 under the standing goal "make it better, decide
autonomously until a reliable solution exists". `research_only`. Pre-registers experiment
family `funding_xs_panel_v1` under `PHASE_0_EVALUATION_PROTOCOL.md` sections 1 to 16 as
they stand on 2026-09-15, before any fold is run. No live, paper, or shadow authority
follows.

## 1. Purpose

Three findings bound this design. P1.28/P1.29: funding carry has a real, statistically
distinguishable gross premium, and what defeats it is execution — the spot hedge leg
costs 10 bps per side plus slippage, and the adverse funding rule hurts. P1.31: the
directional trend vote has no edge, but the *cross-sectional* ranking construction does
something plain time-series bets do not — its long-minus-short book earned +25.8 bps per
week gross and 7.9 bps of that was funding received on the short leg.

This family combines those facts into a different position: no spot leg at all. It ranks
the eligible perpetuals by their trailing funding, goes long the lowest-funding quintile
and short the highest-funding quintile, dollar-neutral, equal weight within each leg. The
short leg collects the high funding; the long leg holds names whose funding is lowest
(often negative, so the long is paid too); the price exposure is the spread between the
two quintiles, which the literature on funding-rate cross-sections reports as favourable
to the same direction (rich-funding names underperform). Both legs are USD-M perpetuals at
5 bps fee, so a round trip costs half of the carry pair's.

## 2. What was seen before this design (disclosure)

P1.28, P1.29 and P1.31's fold results on the same nine test windows, and the 2026-09-11
spike's funding aggregates (which include the locked holdout's funding). The design is a
recombination of what those records showed; it is not tuned to any fold, and every rule
below is fixed here. The holdout from 2026-03-08 stays locked and was never read by a fold.

## 3. Signal and book

For contract *c* at decision *t* (Sunday close), `F_L(c, t)` = sum of its funding
settlements in `(t − L weeks, t]` (per unit of notional, the P1.28 definition,
`carry_signals.trailing_funding`). A contract with no settlement in the window has no score.

Book: P1.27's cross-sectional quintile construction (`panel_signals._cross_sectional_weights`
with `reverse=True`): rankable contracts sorted by `F_L`, long the bottom quintile, short the
top quintile, equal weight within each leg, each leg one half of gross, quintile size at
least the declared minimum (8), computed from the rankable set as P1.27 declared. Fewer
than two full quintiles → the member holds nothing (`MEMBER_HELD_NOTHING`).

Hold: one week, or four weeks as overlapping cohorts of one quarter of capital each (the
P1.31 cohort book: sum of the last four weekly vectors divided by four, calendar-aged,
empty vector contributes nothing, contract without a bar at *t* dropped, three-Sunday
warm-up forming vectors only, in-window `UNIVERSE_TOO_SMALL` resets).

Exit rule (member 4 only): at every decision, a contract held in any retained cohort whose
trailing one-week funding has the *wrong sign for its leg* — non-positive for a short, non-
negative for a long — is removed from every cohort that holds it; its share stays
undeployed until the cohort ages out (the P1.29 rule, applied per leg).

## 4. Family declaration

Everything not stated here is identical to `xs_momentum_panel_v1` (capture, universe rule,
weights rules, folds, statistics, gates, `panel_decision.py`), with one declared change to
the cost tables: the adverse funding rule is P1.28's (receipts ×0.75, payments ×2), not
P1.27's (receipts ×0), because this family's return *is* funding and a zero-receipt
scenario would test whether funding exists rather than whether the book survives a worse
funding regime. Base multipliers 1/1. Fees 5 bps per side both scenarios; slippage 5/10 →
10/20; forced close ×1 → ×2.

### 4.1 Hypothesis

A dollar-neutral book that is long the lowest-funding quintile and short the highest-
funding quintile of a liquid USDT-perpetual universe, ranked on trailing one- or four-week
funding and held one or four weeks, earns funding on both legs net of both legs' execution
costs and the quintile return spread, with positive expectancy under base costs and
non-negative under adverse costs, without single-fold, single-contract or single-week
concentration, after correction across four members.

### 4.2 Members (four trials, each fixed)

| Member | Lookback | Hold | Exit rule |
| --- | ---: | ---: | --- |
| `fx_q5_l4w_h1w` | 4 | 1 | off |
| `fx_q5_l4w_h4w` | 4 | 4 | off |
| `fx_q5_l1w_h4w` | 1 | 4 | off |
| `fx_q5_l4w_h4w_exit` | 4 | 4 | on |

### 4.3 Controls (not trials)

`no_trade`, `random_ranks` (dominance), `passive_long_ew` (context only) — P1.27's; the
one-week-hold controls of P1.27 (no cohorts).

### 4.4 Primary metric, budget, gates

P1.27's sections 5.4/5.5 and 9 unchanged; BH q ≤ 0.10 over four members.

### 4.5 Reported extras

Per episode: `signalled_contracts`, `funding_collected` (minus `funding_cost`),
`exit_rule_removals`, `mean_score` (mean `F_L` over rankable contracts).

## 5. Modules

- `funding_xs_config.py`: `FundingXsFamilySpec` (members with `lookback_weeks`,
  `hold_weeks`, `exit_on_sign_flip`; P1.27's controls; universe/weights/costs/folds/
  statistics models reused), dispatched by `panel_config.load_family_spec`.
- `funding_xs_signals.py`: `funding_scores(funding_by_contract, eligible, decision, L)`,
  `build_funding_xs_weight_vectors(histories, snapshot, funding_by_contract, *, spec)`
  (members via `_cross_sectional_weights(scores, rules, reverse=True)`; controls exactly
  P1.27's), `sign_flip_exits(cohort_vectors, trailing_one_week)`.
- `vector_fold_run.py`: the P1.31 fold loop (`trend_fold_run.py`) generalised over a
  weight-vector builder and an optional per-decision exit rule; `trend_fold_run.py` and the
  new `funding_xs_fold_run.py` are thin wrappers; each family keeps its own module list
  for the code hash. Report schema: P1.27's plus `warm_up_weeks` and the extras above.
- `cli.py`: `funding-xs-fold`.

## 6. Artifacts

`configs/funding-xs-panel-v1.json`; `artifacts/fxs/fxs-walk-forward-<date>-usdt-perps-1d-w1-v1.json`,
`artifacts/fxs/funding-xs-<date>-fold<i>-v1.json`, `artifacts/fxs/funding-xs-<date>-decision-v1.json`,
registry `artifacts/fxs/metadata-funding-xs-v1.sqlite3`; decision record
`P1_32_DECISION_<date>.md`; backlog P1.32.

## 7. Decision rule

A passing member is the repository's first eligible candidate: a market-neutral funding
book with no spot leg, on data through 2025-10 with the holdout still locked; it becomes
the base for measured-cost re-declaration (with the Binance journal's perpetual tiers) and
for a learned gate later. A fail says the funding cross-section does not pay its own
turnover under these rules; the remaining path for funding is the measured-cost carry
(P1.30). No threshold moves either way. Independent of P1.30 and P1.31; records on
different days from P1.31's.
