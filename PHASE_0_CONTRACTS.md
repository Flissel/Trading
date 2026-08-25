# Phase 0 Canonical Contracts

- Status: Draft v0.1
- Date: 2026-08-24
- Applies to: research, backtest, shadow, and paper modes

## 1. Contract principles

All component boundaries use versioned, machine-validatable records. Required
fields may not be inferred silently. Unknown values are represented explicitly,
not as zero, empty text, or a fabricated default.

Every record carries:

- a globally unique `event_id`;
- `schema_name` and semantic `schema_version`;
- `created_at` in UTC nanoseconds;
- `mode` (`backtest`, `shadow`, `paper`, or `tiny_live`);
- producer identity and version;
- causation and correlation identifiers;
- a content hash after canonical serialization.

`tiny_live` is a representable value for compatibility but is rejected by the
Phase 1 runtime policy.

## 2. Time contract

Each observation distinguishes at least:

- `event_time`: when the source says the event occurred;
- `available_time`: earliest instant the system could legally and technically
  use the observation;
- `received_time`: local collector receipt time;
- `processed_time`: completion time of the current transformation.

For source data without a trustworthy event timestamp, `event_time` is null and
the limitation is recorded in `quality_flags`. Backtests gate features on
`available_time`, never on publication date, file date, or corrected future
knowledge.

## 3. EventEnvelope

The common envelope contains:

```text
event_id: UUID
schema_name: string
schema_version: semver
mode: enum
producer_id: string
producer_version: string
created_at_ns: int64
correlation_id: UUID
causation_id: UUID | null
run_id: UUID
payload_hash: sha256
payload: typed object
```

The envelope is immutable after emission. Corrections are new events referencing
the superseded event.

## 4. MarketEvent

Required common fields:

```text
venue: string
instrument_id: string
instrument_type: enum
event_type: enum[trade, quote, book_delta, book_snapshot, mark, index, funding]
event_time_ns: int64 | null
available_time_ns: int64
received_time_ns: int64
sequence: int64 | null
source_record_id: string
source_payload_hash: sha256
quality_flags: list[string]
```

Numeric price and quantity fields use fixed-precision decimal semantics. Binary
floating point is not canonical for monetary values.

Order-book deltas are accepted only when their snapshot base and sequence chain
are known. A detected gap invalidates the book until a new verified snapshot.

## 5. EvidenceItem

An evidence item is a bounded, source-grounded observation usable by an agent or
feature job:

```text
evidence_id: UUID
source_type: enum[market, news, onchain, social, derived]
source_name: string
source_uri_or_record_id: string
observed_at_ns: int64 | null
available_time_ns: int64
retrieved_at_ns: int64
license_class: string
retention_class: string
content_hash: sha256
extractor_id: string
extractor_version: string
text_or_value: typed union
quality_score: decimal[0,1]
quality_flags: list[string]
```

Derived evidence includes all parent evidence IDs and transformation hashes.
Unsupported social data is rejected before feature or training admission.

## 6. AgentAssessment

Agents emit analysis, never orders:

```text
assessment_id: UUID
agent_role: enum[data_quality, market_structure, news, onchain, contradiction, validator]
as_of_ns: int64
instrument_ids: list[string]
horizon_ns: int64
evidence_ids: list[UUID]
claim: string
direction: enum[negative, neutral, positive, unknown]
confidence: decimal[0,1]
novelty: decimal[0,1]
contradictions: list[typed object]
missing_evidence: list[string]
abstain: bool
```

If evidence is absent, expired, contradictory beyond policy, or outside license,
the agent must abstain. Confidence is a reported belief, not a position size.

## 7. Forecast

Forecasts predict a distribution, not an exact future price:

```text
forecast_id: UUID
model_id: string
model_version: string
dataset_hash: sha256
feature_set_hash: sha256
calibration_version: string
instrument_id: string
as_of_ns: int64
horizon_ns: int64
target: string
mean: decimal
quantiles: map[decimal, decimal]
prob_up: decimal[0,1]
prob_down: decimal[0,1]
expected_volatility: decimal >= 0
epistemic_uncertainty: decimal >= 0
aleatoric_uncertainty: decimal >= 0
regime_probabilities: map[string, decimal]
staleness_ns: int64
input_event_ids: list[UUID]
```

Probabilities must be calibrated out of sample. Missing quantiles or stale input
may make a forecast ineligible for decision use without making the record
invalid for diagnostics.

## 8. DecisionIntent

The deterministic policy converts eligible forecasts and assessments into a
proposal:

```text
decision_id: UUID
policy_id: string
policy_version: string
instrument_id: string
as_of_ns: int64
forecast_ids: list[UUID]
assessment_ids: list[UUID]
action: enum[long, short, flatten, hold]
target_exposure: decimal
expected_gross_return: decimal
expected_cost: decimal
expected_net_return: decimal
decision_confidence: decimal[0,1]
expiry_ns: int64
reason_codes: list[string]
```

An exact abstention/hold path is mandatory. No order exists at this stage.

## 9. RiskDecision

Only the deterministic risk engine can approve a proposal:

```text
risk_decision_id: UUID
decision_id: UUID
risk_policy_id: string
risk_policy_version: string
evaluated_at_ns: int64
approved: bool
approved_exposure: decimal
approved_notional: decimal
limit_snapshots: map[string, decimal]
reason_codes: list[string]
kill_switch_state: enum[armed, triggered]
```

Risk may reduce or reject exposure but may not increase it. Any missing state,
stale account view, schema error, or triggered kill switch fails closed.

## 10. OrderIntent

An order intent is created only from an approved risk decision:

```text
order_intent_id: UUID
risk_decision_id: UUID
execution_policy_id: string
instrument_id: string
side: enum[buy, sell]
order_type: enum[market, limit, stop_market, stop_limit]
quantity: decimal > 0
limit_price: decimal | null
stop_price: decimal | null
time_in_force: enum
reduce_only: bool
created_at_ns: int64
expires_at_ns: int64
client_order_id: string
```

The execution adapter rejects duplicate client IDs, expired intents, unsupported
order semantics, non-approved modes, and any intent whose risk decision cannot
be verified.

## 11. ExecutionEvent

Execution state is an append-only event stream:

```text
execution_event_id: UUID
order_intent_id: UUID
client_order_id: string
venue_order_id: string | null
state: enum[created, submitted, accepted, partially_filled, filled, canceled, rejected, expired, unknown]
event_time_ns: int64 | null
received_time_ns: int64
last_quantity: decimal
cumulative_quantity: decimal
last_price: decimal | null
fees: list[typed object]
raw_status: string
reason_code: string | null
```

Transport timeout never implies rejection. Ambiguous outcomes enter `unknown`
and require reconciliation before new conflicting actions are allowed.

## 12. DatasetManifest

Every derived dataset has a manifest:

```text
dataset_id: UUID
dataset_name: string
dataset_version: semver
schema_hash: sha256
content_hash: sha256
normalizer_version: string
parent_manifest_ids: list[UUID]
source_ranges: list[typed object]
row_count: int64
min_available_time_ns: int64
max_available_time_ns: int64
quality_report_hash: sha256
license_classes: list[string]
created_at_ns: int64
```

A dataset is inadmissible if lineage, time bounds, schema, licensing, or quality
report is missing.

## 13. Validation order

Every boundary validates in this order:

1. canonical serialization and content hash;
2. schema and semantic version compatibility;
3. provenance and licensing;
4. time availability and staleness;
5. referential integrity and causation chain;
6. component-specific policy;
7. append-only audit emission.

Failure at any step produces a typed rejection event. It never falls back to an
unvalidated free-text path.

## 14. Versioning rules

- patch: clarification or validation tightening without changing valid payloads;
- minor: backward-compatible optional fields or enum extensions;
- major: required-field, meaning, unit, or invariant changes.

Historical records are never rewritten to a new schema in place. Migrations
produce a derived dataset with an explicit parent manifest.
