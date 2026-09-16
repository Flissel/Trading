# Phase 1 Implementation Backlog

- Status: Proposed v0.1
- Date: 2026-08-24
- Scope: local C: workspace, bounded data samples, no live capital

## Delivery rule

Each item is implemented test-first where practical, produces an auditable
artifact, and is complete only after its acceptance checks pass. No task may
silently add E: storage, private credentials, real order routing, or an
unauthorized data source.

## P0 — Safety and reproducibility foundation

### P1.1 Repository scaffold

Create a strict Python project with a locked environment, typed configuration,
project-local structured logger, test runner, linting, and deterministic seed
utility.

Acceptance:

- clean install on the current machine;
- unit test, typecheck, and lint commands pass;
- no secrets or machine-specific absolute paths in tracked configuration;
- default mode is `backtest` and live mode cannot be selected.

### P1.2 Path and storage guard

Implement canonical path resolution and storage preflight.

Acceptance:

- any E: path, drive root, missing reserve, or cross-volume temp path is rejected;
- insufficient-space tests fail before a download starts;
- partial output is recoverable and never published as a valid manifest.

### P1.3 Canonical schemas

Implement the records in `PHASE_0_CONTRACTS.md` with strict validation and
canonical serialization.

Acceptance:

- round-trip and hash stability tests pass;
- unknown fields and incompatible major versions fail closed;
- monetary values never pass through binary floating-point canonical fields;
- property tests cover invalid timestamps, quantities, and causation chains.

### P1.4 Metadata registry

Implement SQLite migrations and repositories for manifests, experiments,
artifacts, promotions, and risk state.

Acceptance:

- transactions preserve referential integrity;
- artifacts are verified by hash before registration and retrieval;
- restart test recovers persisted state;
- raw payloads and model tensors are not stored in SQLite.

## P1 — Bounded data pipeline

### P1.5 OKX sample importer

Import one small official BTC perpetual historical sample into L0 and L1.

Acceptance:

- rights/source metadata recorded;
- raw payload and normalized output hashes verify;
- duplicates, malformed rows, and time anomalies have deterministic outcomes;
- the run respects the C: reserve.

### P1.6 Binance reference sample importer

Import a matching public Binance USD-M interval for comparison only.

Acceptance:

- instrument mapping is effective-dated;
- venue timestamp and symbol semantics remain separate;
- no private API credentials are required.

### P1.7 Order-book reconstructor

Reconstruct a short L2 segment using venue-specific sequence rules.

Acceptance:

- snapshot plus valid deltas yields deterministic state hashes;
- missing, duplicate, reordered, and corrupt deltas are tested;
- a gap invalidates downstream book features until resnapshot.

### P1.8 Dataset manifest and quality report

Publish a bounded L1/L2 Parquet dataset plus report.

Acceptance:

- schema, files, row count, time coverage, exclusions, and root hash verify;
- rerun with identical inputs produces identical semantic content;
- DuckDB can query the dataset directly without copying it into a database.

## P2 — Leakage-safe research loop

### P1.9 Point-in-time feature engine

Implement a minimal market-only feature set: returns, volatility, spread,
imbalance, trade flow, funding context, and feed-health indicators.

Acceptance:

- every output records availability time and input lineage;
- automated leakage tests fail when future rows or globally fitted scalers are
  introduced;
- gaps and staleness follow the frozen feature definitions.

### P1.10 Label engine

Implement multi-horizon return, volatility, cost-adjusted direction, and
excursion labels in a process that cannot read feature outputs.

Acceptance:

- label availability is explicit;
- overlap intervals are recorded;
- entry/exit and cost conventions are reproducible.

### P1.11 Walk-forward view builder

Materialize the rolling train/validation/test manifests with purge and embargo.

Acceptance:

- exact membership is immutable;
- no label interval crosses a protected boundary;
- final holdout is inaccessible to training and selection code.

### P1.12 Experiment registry and evaluator

Record every trial, including failures, and compute forecast/economic metrics,
block-bootstrap intervals, and multiple-testing corrections.

Acceptance:

- repeated run with the same inputs and seeds matches within declared numeric
  tolerance;
- promotion is a deterministic decision with reason codes;
- unregistered ad-hoc results cannot be promoted.

## P3 — Baselines before CTM

### P1.13 Non-learned baselines

Implement no-trade, random-with-cost, persistence, simple momentum, and simple
mean-reversion references.

Acceptance:

- all use the same event, cost, split, and execution assumptions;
- random baseline distribution uses frozen seeds and repeated trials.

### P1.14 Statistical and small ML baselines

Implement calibrated logistic/linear models and a small tree-based challenger.

Acceptance:

- preprocessing fits training data only;
- calibration is out of sample;
- feature importance is diagnostic, not promotion evidence;
- adverse-cost and dominance checks are reported.

### P1.15 Compact sequence baseline

Implement one bounded sequence model that fits the 12 GB VRAM constraint.

Acceptance:

- memory preflight and batch bounds prevent system exhaustion;
- checkpoint includes dataset, schema, code, config, seed, and calibration IDs;
- it must beat simpler baselines under the evaluation protocol to continue.

## P4 — CTM and hybrid intelligence challengers

### P1.16 CTM feasibility spike

Implement the smallest faithful CTM-style temporal challenger supported by the
available reference implementation and current hardware.

Acceptance:

- parameter count, memory, latency, and training budget are recorded;
- comparison uses identical data and costs;
- failure to beat baselines stops expansion.

### P1.17 Residual/uncertainty challenger

Model forecast residuals and uncertainty without letting predicted prices become
direct rewards.

Acceptance:

- residual targets are generated strictly out of sample;
- calibration and interval coverage improve without economic degradation;
- no target leakage from realized outcomes.

### P1.18 Bounded evidence-agent pipeline

Implement data-quality, news/on-chain fixture, contradiction, and validation
agents emitting `AgentAssessment` only.

Acceptance:

- agents cannot import or call execution adapters;
- every claim references admitted evidence;
- missing/conflicting evidence exercises the abstention path;
- synthetic fixtures are technically blocked from experiment promotion.

### P1.19 Alternative-data ablation

Add permitted official news and Coin Metrics Community data only after the
market-only baseline is frozen.

Acceptance:

- rights and retention metadata pass admission;
- market-only versus augmented folds are paired;
- added data must pass the incremental-value gate or remain excluded.

## P5 — Decision, risk, and execution simulation

### P1.20 Deterministic EV policy

Convert calibrated forecasts to `DecisionIntent` with an explicit hold path.

Acceptance:

- non-positive adverse-cost EV always holds or flattens;
- stale or ineligible forecasts cannot produce new exposure;
- decisions reproduce from registered inputs.

### P1.21 Deterministic risk engine

Implement `PHASE_0_RISK_POLICY.md` as a pure policy core plus persisted state.

Acceptance:

- boundary and property tests cover every limit;
- risk can reduce/reject but never increase exposure;
- restart, cooldown, high-water mark, and kill-switch tests pass.

### P1.22 Event-driven execution simulator

Model order state, latency, fees, spread, slippage, partial fills, funding, and
unknown outcomes.

Acceptance:

- transport timeout does not imply rejection;
- duplicate commands are idempotent;
- base and adverse cost scenarios are replayable;
- fill assumptions are reported, not hidden.

### P1.23 End-to-end backtest

Run canonical events through features, model, decision, risk, execution, and
audit without shortcut paths.

Acceptance:

- every fill traces to admitted source events and an approved risk decision;
- audit/hash chain verifies;
- injected corruption fails closed;
- results include all baselines and rejected trials.

## P6 — Shadow and paper readiness

### P1.24 Public-feed shadow runner

Consume bounded live public data, generate decisions, and record what would have
happened without an order route.

Acceptance:

- shadow mode has no execution credentials or live client factory;
- reconnect, staleness, clock, and disk-reserve failures are exercised;
- restart preserves state and audit continuity.

### P1.25 Paper adapter and reconciliation

Connect only to a verified sandbox/paper endpoint or local paper adapter.

Acceptance:

- environment and account identity are checked at startup;
- unknown orders block conflicting risk until reconciled;
- no configuration value can redirect paper mode to live.

### P1.26 Paper observation window

Run for at least eight weeks and 200 independent-equivalent position episodes.

Acceptance:

- evaluation and risk gates pass without unauthorized overrides;
- operational incidents and model drift are included;
- the output is a promotion recommendation, never automatic live activation.

### P1.27 Cross-sectional daily momentum family

Evaluate the pre-registered family `xs_momentum_panel_v1` on a Binance USD-M
USDT-perpetual panel with weekly holding, under
`docs/superpowers/specs/2026-09-08-xs-momentum-family-design.md` and section 16
of the evaluation protocol.

Acceptance:

- the capture verifies, and no contract has more than 3 missing days inside its
  listed span;
- the manifest publishes at least the declared fold geometry and the pooled
  out-of-sample episode count reaches 200, otherwise the family stops at
  `INSUFFICIENT_EVIDENCE`;
- all six members and three controls are evaluated on every fold in one
  invocation per fold;
- a member is `eligible_for_further_review` only with a positive pooled base
  mean, a positive 95% block-bootstrap lower bound, a non-negative pooled
  adverse mean, positive base PnL in at least two thirds of folds, a
  Benjamini-Hochberg q of 0.10 or less within the six, no fold, contract or
  episode above half of pooled base net PnL, and dominance over the strongest
  control under both cost scenarios;
- the decision report and a root-level `P1_27_DECISION_<date>.md` record every
  member, including the rejected ones, with turnover and deflated Sharpe;
- the final holdout stays closed.

### P1.28 Funding carry family

Evaluate the pre-registered family `funding_carry_panel_v1`, long-spot short-perpetual
pairs on Binance selected on trailing realised funding and held through overlapping
weekly cohorts, under
`docs/superpowers/specs/2026-09-11-funding-carry-family-design.md` and the evaluation
protocol as of 2026-09-11.

Acceptance:

- both captures verify and pass the capture quality gate after repair;
- the manifest binds both captures and reaches the 200-episode floor, otherwise the
  family stops at `INSUFFICIENT_EVIDENCE`;
- all three members and three controls are evaluated on every fold in one invocation
  per fold, and the decision module's gates are the P1.27 gates unchanged;
- the decision report carries per-member funding collected, basis P&L and the split of
  turnover cost between legs;
- the decision record `P1_28_DECISION_<date>.md` records every member, the two spikes
  that informed the design, and the contamination of the holdout's funding aggregate;
- the final holdout stays closed.

### P1.29 Funding carry v2 family

Evaluate the pre-registered family `funding_carry_panel_v2` — the four members
`carry_l4w_h26w`, `carry_l4w_h13w_exit`, `carry_l4w_h26w_exit` and
`carry_l4w_h26w_exit_hurdle2` — a 26-week hold, an exit rule that drops a held pair once
its trailing one-week funding turns non-positive, and a cost hurdle at entry with
multiple 2, under
`docs/superpowers/specs/2026-09-12-funding-carry-v2-family-design.md` and the evaluation
protocol as of 2026-09-12.

Acceptance:

- both captures verify and pass the capture quality gate after repair;
- the manifest binds both captures and reaches the 200-episode floor, otherwise the
  family stops at `INSUFFICIENT_EVIDENCE`;
- all four members and three controls are evaluated on every fold in one invocation per
  fold, and the decision module's gates are the P1.27 gates unchanged;
- the decision report carries the eight-key `extras_mean` per member —
  `funding_collected`, `basis_pnl`, `spot_trading_cost`, `perpetual_trading_cost`,
  `forced_spot_legs`, `forced_perpetual_legs`, `exit_rule_removals` and
  `hurdle_rejections`;
- the decision record `P1_29_DECISION_<date>.md` records every member, including the
  rejected ones, with turnover and deflated Sharpe;
- the final holdout stays closed.

### P1.30 Funding carry v3 on measured execution costs

Declare `funding_carry_panel_v3` only from the Binance cost journal's finalisation receipt
(`docs/superpowers/specs/2026-09-12-binance-cost-journal-design.md`, section 5 fixes the
only admissible reading for a declaration without a `capital` block: base slippage tiers =
`tier_p50_of_p50` of the worse leg at 5,000 USDT, adverse = `tier_p50_of_p90` at
50,000 USDT, rounded up to whole basis points, receipt hash cited; section 5.1, added
2026-09-16, fixes a second, capital-declared reading for a declaration that carries one —
see P1.33). The journal `data/cost-journals/binance-carry-v1` was created and launched on
2026-09-12 (15 pairs, 30 instruments, 11,000 rounds at 61 s).

Acceptance:

- the receipt verifies, every tier median the declaration uses has at least four
  contributing instruments, and the eligibility floors it records are the declared ones;
- the v3 declaration differs from v2 only in the two cost tables and cites the receipt;
- the chain (manifest, nine folds, decision) runs unchanged and `P1_30_DECISION_<date>.md`
  records the outcome; if the measured tiers still fail the adverse floor, the carry is
  closed at this venue and scale;
- the final holdout stays closed.

### P1.31 Trend aggregate family

Evaluate the pre-registered family `trend_aggregate_panel_v1` — the four members
`ta_ts_t02`, `ta_ts_t05`, `ta_ts_t02_h4w` and `ta_xs_q5` — a twelve-indicator trend vote
over 20 to 120 days of daily closes, thresholded at 0.2 and at 0.5, held one week or in
four overlapping weekly cohorts of a quarter of capital each, plus a cross-sectional
quintile member, on P1.27's perpetual panel under
`docs/superpowers/specs/2026-09-15-trend-aggregate-family-design.md` and the evaluation
protocol as of 2026-09-15. May run before or after P1.30 (independent families); the two decision records
are written on different days.

Acceptance:

- the repaired perpetual capture verifies and the manifest binds it, reaching the
  200-episode floor, otherwise the family stops at `INSUFFICIENT_EVIDENCE`;
- the declaration's universe, weights, costs, folds and statistics are P1.27's values
  unchanged, and `panel_signals.py`, `panel_fold_run.py` and
  `configs/xs-momentum-panel-v1.json` are untouched, so P1.27 stays reproducible;
- all four members and the three P1.27 controls are evaluated on every fold in one
  invocation per fold, and the decision module's gates are the P1.27 gates unchanged;
- each fold warms the three Sundays before its first decision so the four-week member's
  book opens full, recorded as `warm_up_weeks` and
  `FOLD_OPENING_BOOK_WARMED_FROM_PRIOR_WEEKS`;
- the decision report carries the three-key `extras_mean` per member —
  `signalled_contracts`, `abstained_contracts` and `mean_score`;
- the decision record `P1_31_DECISION_<date>.md` records every member, including the
  rejected ones, with turnover and deflated Sharpe, and states which way the directional
  question was closed;
- the final holdout stays closed.

### P1.32 Funding cross-section family

Evaluate the pre-registered family `funding_xs_panel_v1` — the four members
`fx_q5_l4w_h1w`, `fx_q5_l4w_h4w`, `fx_q5_l1w_h4w` and `fx_q5_l4w_h4w_exit` — a
perpetual-only, dollar-neutral book long the lowest-funding quintile and short the
highest-funding quintile of P1.27's universe, ranked on trailing one- or four-week funding
and held one week or in four overlapping weekly cohorts of a quarter of capital each, on
P1.27's perpetual panel under
`docs/superpowers/specs/2026-09-15-funding-xs-family-design.md` and the evaluation
protocol as of 2026-09-15. No spot leg, so no hedge capture and half the carry pair's
round trip. Independent of P1.30 and P1.31; the decision records are written on different
days.

Acceptance:

- the repaired perpetual capture verifies and the manifest binds it, reaching the
  200-episode floor, otherwise the family stops at `INSUFFICIENT_EVIDENCE`;
- the declaration's universe, weights, folds and statistics are P1.27's values unchanged
  and its costs are P1.27's but for the one declared change — the adverse table's funding
  receipts at 0.75 and payments at 2, P1.28's rule, because this family's return *is*
  funding — and `panel_signals.py`, `panel_fold_run.py` and
  `configs/xs-momentum-panel-v1.json` are untouched, so P1.27 stays reproducible;
- P1.31 stays reproducible across the fold loop's generalisation into
  `vector_fold_run.py`: fold 1 of the trend fixture reproduces
  `tests/fixtures/trend_fold1_expected.json` digit for digit, only `code_hash` and
  `report_hash` having moved;
- all four members and the three P1.27 controls are evaluated on every fold in one
  invocation per fold (`funding-xs-fold`), and the decision module's gates are the P1.27
  gates unchanged;
- each fold warms the three Sundays before its first decision so the four-week members'
  books open full, recorded as `warm_up_weeks` and
  `FOLD_OPENING_BOOK_WARMED_FROM_PRIOR_WEEKS`;
- the exit rule runs for `fx_q5_l4w_h4w_exit` alone and for no control: a held leg whose
  trailing one week of funding has the wrong sign for that leg (`None` counted as wrong)
  is zeroed in every retained cohort vector before the book is assembled, never during the
  warm-up, its share left undeployed until the cohort ages out, and counted per episode as
  `exit_rule_removals`;
- the decision report carries the four-key `extras_mean` per member --
  `signalled_contracts`, `funding_collected`, `exit_rule_removals` and `mean_score`;
- the decision record `P1_32_DECISION_<date>.md` records every member, including the
  rejected ones, with turnover and deflated Sharpe, and states whether the funding
  cross-section pays its own turnover under these rules;
- the final holdout stays closed.

### P1.33 Funding carry v4 (declared capital, slot book)

Evaluate the pre-registered family `funding_carry_panel_v4` and its measured form
`funding_carry_panel_v4_measured` (the evaluated family) — the four members
`carry_s10_l4w_h13w`, `carry_s10_l4w_h13w_exit`, `carry_s10_l4w_h26w` and
`carry_s10_l4w_h26w_exit` — a spot-hedged funding carry declared against the book the user
would actually run: a book of at most 10 000 USDT in ten equal pair slots, at most 500 USDT per leg
per order, slots filled from the top decile of eligible pairs by trailing four-week
funding and held up to 13 or 26 weeks, two members additionally vacating a slot when its
pair's trailing one-week funding turns non-positive, under
`docs/superpowers/specs/2026-09-16-funding-carry-v4-small-book-design.md`, the journal
spec's section 5.1, and the evaluation protocol as of 2026-09-16. Independent of P1.30;
the decision records are written on different days.

Acceptance:

- the Binance cost journal's finalisation receipt must exist and verify; the chain
  refuses to declare or fold without it;
- the base declaration carries a `capital` block with its four frozen values —
  `book_usdt` 10,000, `pair_slots` 10, `per_leg_notional_usdt` 500, `fee_tier`
  `standard_taker_no_bnb` — and the per-leg notional equals `book_usdt / (2 × pair_slots)`;
- the measured declaration's four slippage values are exactly section 5.1's reading of the
  receipt at the declared capital — base `tier_p50_of_p50` of the worse leg at 500 USDT,
  adverse `tier_p50_of_p90` at 5,000 USDT, each rounded up to the next whole basis point —
  and `carry-verify-measured` re-derives all four from the receipt and the base
  declaration;
- the slot mechanics run as declared: a slot releases when its pair has been held `H`
  weeks or has lost a leg's bar; the two exit members additionally release a slot whose
  pair's trailing one-week funding is non-positive or absent; empty slots fill from the
  top decile of paying pairs by trailing four-week funding, skipping pairs already held
  and, for the exit members, pairs whose trailing one-week funding is non-positive or
  absent; a pair released and refilled in the same decision keeps its weights and is
  charged no turnover;
- no warm-up: every slot opens empty at a fold's first decision (`warm_up_weeks` 0);
- each episode carries the twelve extras — the eight carry extras plus `filled_slots`,
  `slot_fills`, `slot_releases` and `no_fill` — and each fold, candidate and scenario
  carries `uncharged_final_exit_cost`;
- P1.28's and P1.29's cohort path stays reproducible across the fold runner's slot-mode
  addition: fold 0 of the v1 fixture reproduces
  `tests/fixtures/carry_v1_fold0_expected.json` digit for digit, only `code_hash` and
  `report_hash` having moved;
- the decision record `P1_33_DECISION_<date>.md` records every member, including the
  rejected ones, with turnover and deflated Sharpe, subtracts `uncharged_final_exit_cost`
  from any member that passes narrowly, and states whether the small-book carry pays its
  own execution at the declared fee tier;
- the final holdout stays closed.

## Explicitly deferred

- real-capital execution;
- leverage above 1x;
- RL or contextual-bandit sizing before deterministic baselines pass;
- X/Twitter model features;
- paid market-data subscriptions;
- full historical L2 download on current storage;
- distributed agents, Kafka/Redpanda, Ray, and Kubernetes;
- use of the anticipated 480 GB VRAM environment before topology verification.
