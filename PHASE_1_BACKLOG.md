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

## Explicitly deferred

- real-capital execution;
- leverage above 1x;
- RL or contextual-bandit sizing before deterministic baselines pass;
- X/Twitter model features;
- paid market-data subscriptions;
- full historical L2 download on current storage;
- distributed agents, Kafka/Redpanda, Ray, and Kubernetes;
- use of the anticipated 480 GB VRAM environment before topology verification.
