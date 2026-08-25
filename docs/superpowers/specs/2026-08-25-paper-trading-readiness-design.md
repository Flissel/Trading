# Gate-First Paper-Trading Readiness Design

Status: Approved in chat on 2026-08-25

## 1. Purpose

This design takes the existing leakage-safe BTC research foundation to a
verifiable paper-trading system in three sequential phases:

1. recover or falsify a market-only edge under frozen economic gates;
2. build operationally safe simulation, shadow, local-paper, and OKX-demo
   infrastructure;
3. evaluate compact sequence, faithful CTM, and non-executing evidence-agent
   challengers under the same gates.

The target is paper-trading readiness, not real-capital execution. A technical
component may be completed after an economic gate fails, but it must remain
`research_only` or `hold_only` and cannot acquire promotion authority.

## 2. Current Baseline

The repository already contains:

- immutable public OKX and Binance candle captures and canonical datasets;
- point-in-time market features and horizon-bound labels;
- leakage-safe h4 (one-hour) and h16 (four-hour) walk-forward manifests;
- no-trade, momentum, mean-reversion, frozen-seed random, Ridge, calibrated
  Logistic, and boosted-stump candidates;
- moving-block bootstrap, Benjamini-Hochberg correction, aggregate gates, and
  dominance reports;
- deterministic risk and execution primitives plus audit-chain support;
- a final chronological holdout that has not been evaluated.

Neither existing horizon has an eligible candidate. The h16 boosted-stump
candidate is a research lead because it is base-positive in all three folds,
but it is adverse-cost negative and fails the multiple-testing gate.

## 3. Global Invariants

The following rules apply to every phase and adapter:

- No real-capital mode, live endpoint, or automatic live promotion exists.
- The final holdout remains closed until a candidate and all configuration are
  frozen and the user grants an explicit one-time release.
- A failed holdout is not used for further tuning.
- Training-only transformations never observe validation, test, shadow, paper,
  or final-holdout outcomes.
- Validation may select calibration and decision thresholds but may not fit
  model parameters already assigned to training.
- Every attempt, including failures, is registered append-only.
- Base, adverse, fold-stability, episode-count, and multiple-testing gates are
  jointly required for an economic promotion claim.
- Models and agents cannot increase exposure, modify risk limits, or bypass
  reconciliation, cooldown, or kill-switch state.
- Missing, stale, contradictory, unverifiable, or lineage-invalid inputs fail
  closed to abstention or `no_new_risk`.
- Historical backtests cannot substitute for the real paper observation
  window.

## 4. Phase A: Gate-First Edge Recovery

### 4.1 Data and cost audit

Each research dataset receives a hash-bound audit containing:

- missing intervals, duplicates, chronology errors, and venue gaps;
- receive-time and source-time staleness;
- cross-venue price and clock anomalies;
- observed or admitted spread evidence;
- fee schedule identity and effective interval;
- slippage and liquidity proxies with their limitations;
- funding assumptions over each expected holding interval;
- base and adverse cost scenarios derived from the same evidence.

Missing cost evidence cannot be replaced by a favorable zero. If evidence is
insufficient, the affected experiment is ineligible.

### 4.2 Pre-registered market hypotheses

Phase A admits three hypothesis families only:

1. multi-scale trend and mean reversion;
2. volatility and liquidity regime conditioning;
3. cross-venue basis and market activity.

Each feature specification records its semantic name, sources, receive-time
rule, lookback, units, null behavior, training-only transformations, and exact
input IDs. Features may use candle and admitted public market microstructure
data but not news, social media, or agent-produced text.

Each family receives one pre-registered primary configuration and at most one
reasoned revision. A defect correction is linked to the failed attempt and is
not erased from the multiplicity ledger. A new economic idea requires a new
hypothesis-family registration.

### 4.3 Forecast uncertainty and abstention

Candidates emit a forecast with calibrated uncertainty rather than an
unqualified direction. A deterministic abstention policy may produce exposure
only when a conservative expected return remains positive after the adverse
cost estimate. Calibration and abstention thresholds are selected on their
declared validation membership only.

The policy is compared with the same model without abstention. Reduced trading
frequency must still meet the frozen episode and effective-sample floors.

### 4.4 Paired evaluation and Phase-A decision

All candidates run on the same h4 and h16 folds, costs, seeds, bootstrap rules,
and baseline reports. An `eligible_candidate` requires:

- positive aggregate base return;
- positive aggregate adverse return;
- positive base return in at least two of three folds;
- the frozen minimum trade and independent-episode evidence;
- the frozen Benjamini-Hochberg q-value gate;
- dominance over no-trade and the existing simpler baselines.

Phase A publishes either an immutable eligible-candidate decision or a
verifiable `no_edge_found` decision. With `no_edge_found`, Phase B can continue
only with `hold_only` authority.

## 5. Phase B: Simulation, Shadow, and Paper Infrastructure

### 5.1 Shared event chain

Every mode uses the same semantic chain:

`MarketEvent -> FeatureSnapshot -> Forecast -> DecisionIntent -> RiskDecision
-> OrderCommand -> ExecutionEvent -> Reconciliation -> AuditRecord`

Each record has a version, stable identity, event time, receive time, lineage,
and content hash. Mode adapters can change transport behavior but cannot change
the meaning of upstream decisions or risk approvals.

### 5.2 Modes

- `backtest` replays immutable historical events deterministically.
- `shadow` consumes bounded public live data and has no credentials, private
  client factory, or order route.
- `paper_local` simulates latency, fees, spread, slippage, funding, partial
  fills, rejection, cancellation, and unknown outcomes.
- `paper_okx_demo` communicates only with pinned OKX demo or sandbox endpoints
  after environment, account, instrument, and demo-mode verification.

No configuration string can redirect a paper adapter to a live endpoint. There
is no live fallback after a connection or authentication failure.

### 5.3 State, idempotency, and reconciliation

Order commands use deterministic idempotency keys. Commands, venue responses,
fills, positions, balances, risk transitions, and audit records are journaled
append-only. Snapshots accelerate recovery but the journal remains
authoritative.

Startup and periodic reconciliation compare local order, position, and balance
state with the active adapter. A timeout is `unknown`, never presumed rejected.
Unknown orders, divergent positions, failed account checks, or incomplete audit
state activate `no_new_risk` until a deterministic reconciliation succeeds.

Mode-specific configuration, credentials, state directories, journals, and
audit chains remain isolated. Secrets never enter Git or immutable research
artifacts.

### 5.4 Observation window

Operational paper readiness requires both:

- at least eight consecutive weeks of observation;
- at least 200 independent-equivalent position episodes.

It also requires no unauthorized mode transition, no hard risk-limit breach,
continuous audit and reconciliation evidence, recorded incidents, and measured
drift and realized costs. A `hold_only` run can prove operational stability but
cannot fabricate episodes or support an economic promotion recommendation.

## 6. Phase C: Sequence, CTM, and Evidence MAS

### 6.1 Compact sequence baseline

The first neural challenger is a bounded causal TCN or GRU, selected in its
phase-specific design before experiments begin. It consumes only admitted
point-in-time feature windows and produces h4 and h16 forecast, uncertainty,
and abstention inputs.

Parameter count, sequence length, batch size, precision, seed, calibration ID,
and checkpoint lineage are frozen. A memory preflight must fit the current
12 GB VRAM limit. Out-of-memory or timeout outcomes are registered failures;
they do not trigger hidden post-test architecture search.

### 6.2 Faithful CTM feasibility spike

The CTM spike pins a reviewed commit from Sakana AI's official implementation:

- https://github.com/SakanaAI/continuous-thought-machines
- https://arxiv.org/abs/2505.05522

It must retain the defining internal temporal axis, neuron-level temporal
processing, and synchronization-based latent representation. An ordinary
recurrent model cannot be relabeled as CTM. The spike uses the same admitted
windows, labels, costs, folds, and compute budget as the compact sequence
baseline. Failure to add value stops CTM expansion.

The anticipated larger GPU environment is not used until its actual topology,
backend support, memory behavior, and I/O path are verified.

### 6.3 Non-executing evidence-agent system

The bounded MAS contains four roles:

- data-quality assessment;
- admitted news or on-chain evidence assessment;
- contradiction detection;
- validation and abstention assessment.

Agents emit typed `AgentAssessment` artifacts containing evidence IDs,
receive-time availability, confidence, contradiction state, lineage, and a
reasoned abstention outcome. Agent packages cannot import or call risk, order,
credential, or execution adapters.

A deterministic aggregator can confirm, weaken, or block a forecast. It cannot
increase requested exposure. Synthetic fixtures are useful for contract and
failure tests but are technically ineligible for experiment promotion.

### 6.4 Alternative-data ablation

Only sources with recorded rights, retention, provenance, and receive-time
semantics are admitted. Market-only and augmented candidates run as paired
ablations. Missing evidence, source conflict, later corrections, or stale
events exercise abstention. Added data remains excluded unless it provides
incremental economic OOS value under the same adverse-cost and multiplicity
gates.

## 7. Error Handling and Safety States

The system uses explicit, persisted states rather than free-text recovery:

- `research_only`: may produce experiment artifacts but no runtime exposure;
- `hold_only`: runtime pipeline operates but every new-risk intent resolves to
  hold or reject;
- `no_new_risk`: existing state is managed conservatively while entries are
  blocked;
- `paper_eligible`: a frozen candidate may request risk review in paper modes;
- `reconciliation_required`: order, position, or balance truth is unresolved;
- `kill_switch`: the active policy forbids new exposure until its declared
  recovery authority acts.

Schema, hash, chronology, registry, storage-reserve, clock, source, model,
credential, audit, and reconciliation failures cannot be downgraded silently.

## 8. Verification Strategy

Verification is layered:

1. unit and property tests for chronology, availability, sizing, limits, and
   state transitions;
2. adapter contract tests across backtest, shadow, local paper, and OKX demo;
3. deterministic golden replays and repeat-run hash checks;
4. fault injection for disconnects, rate limits, duplicate events, partial
   fills, unknown outcomes, process termination, corrupt snapshots, clock
   drift, disk reserve, and audit-sink failure;
5. restart recovery from journal plus snapshot;
6. offline end-to-end tests from source event through audit record;
7. bounded shadow soak tests;
8. local and OKX-demo paper observation.

Tests prove behavior and authority boundaries. Source-text checks alone do not
prove that agents cannot execute or that paper cannot redirect to live.

## 9. Work-Package Decomposition

Each package receives a separate specification, TDD implementation plan,
Conventional Commit, and verification gate.

### Phase A packages

1. data-quality and cost audit;
2. feature-hypothesis and experiment-budget registry;
3. uncertainty and adverse-cost abstention;
4. frozen edge or no-edge decision report.

### Phase B packages

5. shared event and adapter contracts;
6. end-to-end simulator and fault injection;
7. public-feed shadow runner;
8. local paper adapter and restart replay;
9. OKX-demo endpoint guard, adapter, and reconciliation;
10. observation-window recorder and promotion report.

### Phase C packages

11. compact sequence baseline;
12. faithful CTM feasibility spike;
13. evidence-agent contracts and synthetic failure fixtures;
14. admitted alternative-data ingestion and paired ablation.

## 10. Completion Evidence

The overall objective is complete only when current evidence proves all of the
following:

- verified dataset, feature, cost, model, calibration, and split manifests;
- an immutable `eligible_candidate` or an explicit `hold_only` economic state;
- complete audit and reconciliation chains for the evaluated paper modes;
- passing restart, idempotency, authority-boundary, and fault-injection reports;
- at least eight weeks and 200 independent-equivalent paper episodes for any
  economic paper-readiness claim;
- drift, incident, realized-cost, and risk-limit reports;
- a final promotion recommendation with deterministic reason codes;
- no real-capital execution path or automatic live authorization.

The final report may recommend continued research, continued `hold_only`
operation, or paper eligibility. It cannot authorize live trading.

## 11. Non-Goals

The following remain outside this objective:

- real-capital execution or exchange-live credentials;
- leverage above 1x;
- RL or contextual-bandit sizing;
- X/Twitter features;
- autonomous model, risk-policy, or code self-modification;
- distributed training or orchestration before larger hardware is verified;
- profitability claims derived only from backtests, prediction accuracy, or a
  favorable subset of folds.
