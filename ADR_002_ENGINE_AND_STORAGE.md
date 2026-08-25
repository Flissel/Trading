# ADR 002: Initial Engine and Storage Architecture

- Status: Accepted for Phase 1 implementation
- Date: 2026-08-24
- Scope: local research, replay, backtest, shadow, and paper trading

## Context

The first implementation must run on a Windows 11 workstation with an RTX 3060
(12 GB VRAM), 32 GB RAM, and little free space on the system SSD. The E: drive
is explicitly excluded because it is currently unreliable and has caused stalled
filesystem operations. No live trading is authorized.

The architecture must support:

- event-driven replay without look-ahead;
- identical domain contracts in backtest, shadow, and paper modes;
- raw-data provenance and deterministic dataset reconstruction;
- multi-model forecasts and bounded agent interpretation;
- deterministic risk control outside all learned components;
- later migration to larger compute without coupling Phase 1 to a cluster.

## Decision

### 1. Trading and replay engine

Use **NautilusTrader** as the initial event-driven trading and replay engine.

Reasons:

- it provides event-driven backtesting and trading nodes;
- its OKX and Binance adapters provide a migration path from public data to
  sandbox or paper execution;
- its Parquet catalog supports reusable normalized market data;
- strategy and actor contracts can carry from research into later runtime modes.

NautilusTrader is an infrastructure choice, not evidence that a strategy is
correct, realistic, profitable, or safe. All venue behavior used in simulation
must still be validated against recorded messages and paper observations.

### 2. Raw capture and normalization

Raw capture is separated from the trading engine.

The collector writes append-only, compressed source records before any
normalization. Each record retains source, venue, channel, receive timestamp,
event timestamp when supplied, sequence identifiers, payload hash, and schema
version. Secrets and private account payloads are not written into research
datasets.

A deterministic normalization job converts raw records into canonical events
and, where appropriate, NautilusTrader domain objects. Re-running the same
normalizer version over the same raw manifest must produce the same dataset
hash.

### 3. Storage format

Use immutable **Parquet** datasets with Zstandard compression for normalized
market data, features, labels, forecasts, and evaluation outputs.

Use Hive-style partitions only on stable, frequently filtered dimensions:

```text
dataset=<name>/venue=<venue>/instrument=<instrument>/date=<YYYY-MM-DD>/part-*.parquet
```

Do not partition on high-cardinality fields such as model ID, event ID, or
timestamp. Compact small files as a controlled derived-data operation; raw
files remain immutable.

### 4. Query and transformation layer

Use **DuckDB** for local SQL validation, joins, time-bounded extraction,
manifest checks, and experiment summaries. Query Parquet directly instead of
duplicating full datasets into a DuckDB database while storage is constrained.

Use **Polars lazy frames** for typed feature pipelines and streaming-compatible
transformations when Python orchestration is clearer than SQL. Pandas may be
used only at library boundaries or for small diagnostic tables.

### 5. Metadata and experiment registry

Use a project-local **SQLite** registry for dataset manifests, experiment
metadata, model lineage, promotion decisions, and paper-run state. Large arrays,
raw payloads, checkpoints, and result tables stay outside SQLite and are
referenced by content hash and relative artifact path.

The registry is metadata, not the source of market truth. Every registered
artifact must be independently hash-verifiable.

### 6. Runtime topology

Phase 1 is a modular monolith with explicit interfaces:

```text
source collectors
    -> raw append-only records
    -> deterministic normalization
    -> canonical event log / Parquet catalog
    -> feature and model workers
    -> forecast ensemble
    -> deterministic decision policy
    -> deterministic risk engine
    -> simulated execution adapter
    -> immutable audit events
```

Components communicate in process or through durable files and the metadata
registry. Kafka, Redpanda, Ray, Kubernetes, and distributed feature stores are
deferred. They add failure modes without solving a demonstrated Phase 1 limit.

### 7. Agent boundary

Multi-agent components may interpret bounded evidence, detect contradictions,
propose hypotheses, and emit structured assessments. They may not:

- submit, modify, or cancel orders;
- override risk limits;
- fabricate missing observations;
- silently convert low-confidence evidence into a trading signal;
- write directly to canonical datasets.

Every agent output is untrusted input. It must pass schema validation,
provenance checks, time-availability checks, and deterministic policy gates.

### 8. Model-serving boundary

Models run behind a narrow forecast interface. The initial implementation uses
local batch or in-process inference. A separate serving layer is introduced
only after profiling proves it is required.

Training and inference artifacts include exact code revision, dataset hash,
feature schema, model configuration, random seeds, environment lock, and
calibration version.

### 9. Execution modes

The same `DecisionIntent`, `RiskDecision`, and `OrderIntent` contracts are used
in all modes:

- `backtest`: historical event replay with simulated fills;
- `shadow`: live public data, decisions recorded, no order route;
- `paper`: venue sandbox or local execution simulation, no real capital;
- `tiny_live`: forbidden until separately authorized and promoted.

Mode is explicit in every audit event. There is no implicit fallback from paper
or shadow to live.

### 10. C: storage safety

Until additional reliable storage is installed:

- keep code, specifications, registry, and only small sampled datasets on C:;
- require a preflight free-space threshold before downloads or compaction;
- estimate compressed and temporary working space before every ingestion job;
- stop ingestion before the configured reserve is crossed;
- never spill to E: automatically;
- do not begin full historical L2 ingestion on the present C: capacity.

The concrete reserve and download size limits become Phase 1 configuration after
a fresh C: capacity measurement. No storage path may default to a drive root.

## Rejected alternatives

### Build a custom trading engine first

Rejected because exchange semantics, event ordering, order state, and replay
correctness would dominate Phase 1 before any hypothesis is tested.

### Freqtrade as the central research engine

Rejected as the central engine because the planned work requires richer event
and order-book replay plus stricter separation of evidence, forecasts, risk,
and execution. It may still be used later as a benchmark.

### Vector database as canonical storage

Rejected. Vector search is useful for selected text evidence, not as the source
of truth for time-series market events or experiment lineage.

### Distributed multi-agent services from day one

Rejected until local profiling demonstrates a throughput, isolation, or
availability requirement that a modular monolith cannot meet.

## Consequences

Positive:

- the first implementation is feasible on current compute;
- simulation and later paper paths share domain contracts;
- raw evidence remains reconstructable and auditable;
- deterministic controls stay outside models and agents;
- larger hardware can be used later without redesigning the core contracts.

Costs and limitations:

- two representations may exist for some market data: canonical analytics
  Parquet and Nautilus catalog objects;
- adapter behavior still requires venue-specific verification;
- the current C: capacity only permits schemas, samples, and bounded pilots;
- additional reliable SSD/NVMe storage is required before serious L2 history.

## Revisit conditions

Revisit this ADR when:

- reliable additional storage is installed;
- measured replay throughput misses the experiment requirement;
- paper observations expose material engine/venue semantic mismatch;
- a workload needs process isolation or distributed compute;
- the later GPU environment topology is verified.

## Evidence checked

- NautilusTrader adapters: https://nautilustrader.io/docs/latest/developer_guide/adapters/
- NautilusTrader external-data and Parquet workflow: https://nautilustrader.io/docs/latest/how_to/loading_external_data/
- DuckDB Parquet query behavior: https://duckdb.org/docs/current/guides/file_formats/query_parquet
- DuckDB file-format and storage guidance: https://duckdb.org/docs/current/guides/performance/file_formats
