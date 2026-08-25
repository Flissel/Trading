# Phase 0 Dataset and Timestamp Specification

- Status: Draft v0.1
- Date: 2026-08-24
- Initial instruments: BTC and ETH perpetuals
- Initial venues: OKX EEA primary, Binance USD-M public reference

## 1. Purpose

This specification defines what a dataset means, when information becomes
usable, how raw data becomes a research table, and which validation failures
make a dataset inadmissible. Its primary goal is to prevent look-ahead,
survivorship, silent correction, order-book corruption, and unverifiable
results.

## 2. Dataset layers

### L0: source records

Append-only records preserving the received source payload plus capture
metadata. L0 is never used directly by a model.

### L1: canonical events

Typed, normalized events conforming to `PHASE_0_CONTRACTS.md`. Units,
instrument identity, timestamps, numeric precision, sequence semantics, and
quality flags are explicit.

### L2: reconstructed state

Verified order books, bars, funding intervals, venue health, and other stateful
series derived from L1. Every row references its input range and reconstruction
version.

### L3: features and evidence

Point-in-time-safe features and bounded alternative-data evidence. Each value
retains its computation version, parents, availability time, and staleness.

### L4: labels

Future outcomes generated independently from features. Labels include the
complete observation interval and `label_available_time`.

### L5: experiment views

Immutable train, validation, test, and final-holdout manifests referencing L3
and L4 rows. Membership is stored, not recalculated implicitly during a run.

## 3. Time fields and invariants

Canonical time is UTC Unix nanoseconds. Source precision is retained in a
separate field when coarser than nanoseconds.

Required semantics:

```text
event_time       source-claimed occurrence time, nullable
published_time   source publication time, nullable
received_time    first local receipt time
available_time   earliest legal and technical use time
processed_time   transformation completion time
decision_time    forecast information cutoff
label_end_time   end of the outcome interval
label_available_time earliest time the complete label is knowable
```

Invariants:

- `available_time >= received_time` for live-captured data;
- `available_time >= published_time` when publication is required;
- `processed_time >= received_time`;
- a feature is eligible only if `available_time <= decision_time`;
- a training label is eligible only after `label_available_time`;
- null or implausible source times never fall back to file modification time;
- exchange timestamps are not presumed synchronized across venues;
- all clock corrections and measured offsets are recorded as data, not hidden.

For historical downloads, local download time is not the simulated availability
time. A source-specific availability model must be documented. If historical
publication latency cannot be established, conservative latency is applied and
the uncertainty is flagged.

## 4. Instrument identity

Raw venue symbols are mapped through a versioned instrument master containing:

- canonical instrument ID;
- venue and venue symbol;
- product type and settlement asset;
- base and quote assets;
- contract multiplier;
- tick and lot sizes;
- minimum quantity and notional;
- listing and delisting times;
- effective interval for every rule change.

No symbol string is joined across venues without this mapping. Instrument rule
changes create new effective-dated records rather than rewriting history.

## 5. Market datasets

### Trades

Retain trade ID when supplied, price, quantity, aggressor indicator when
defined, event and receipt times, and source sequence. Deduplication keys are
venue-specific and versioned. Equal price, quantity, and timestamp alone do not
prove duplication.

### Quotes and L2 order books

Each reconstructed book segment begins with a verified snapshot. Deltas are
applied only when venue-specific sequence rules pass. The segment becomes
invalid on:

- missing or out-of-order sequence;
- checksum mismatch when supported;
- crossed or otherwise impossible state not explained by venue semantics;
- reconnect without a verified continuation rule;
- negative size or invalid precision.

After invalidation, no book-derived feature is emitted until a new verified
snapshot. Gaps are data, not interpolated order-book updates.

### Funding, mark, index, and open interest

Store the value, measurement time, publication/availability time, interval to
which it applies, and whether it is announced, estimated, or settled. Never
join a settled funding value into decisions made before settlement.

### Cross-venue data

Cross-venue features use the maximum availability time of all inputs. They also
include observed age and clock-offset uncertainty for each leg.

## 6. Alternative-data datasets

News and on-chain evidence must follow `EvidenceItem`. Corrections and deleted
or retracted items are new lifecycle events.

X/Twitter data is not admitted to training or automated decision datasets in
Phase 1. Synthetic or expressly reusable fixtures may test interfaces, and are
marked `fixture_only=true` so they cannot enter an experiment view.

Any later alternative source must pass a rights review covering permitted use,
model training, retention, derived features, deletion duties, and redistribution.

## 7. Features

Every feature definition declares:

- name, version, type, unit, and null semantics;
- exact input datasets and versions;
- lookback interval and minimum observations;
- availability rule and maximum permitted age;
- warm-up behavior;
- transformation code revision and parameters;
- behavior during gaps, venue outages, and invalid books.

Rolling windows are left-closed only at information available by the decision
cutoff. Global normalization over future rows is forbidden. Scalers, encoders,
vocabularies, feature selection, and imputation are fit on the training partition
only and then frozen for validation and test.

Missingness indicators are retained where missing data could convey venue or
regime state. Forward filling requires an explicit maximum age.

## 8. Labels

Initial labels include:

- multi-horizon log return;
- realized volatility;
- return quantiles or distribution parameters;
- direction after base and adverse costs;
- maximum favorable and adverse excursion within the horizon.

Each label records entry convention, horizon, price source, fee/slippage
scenario, and whether the outcome interval overlaps another label. Exact future
price is diagnostic only, not the primary success target.

Labels are generated by a separate job that cannot write feature columns.
Feature generation cannot read L4 or later layers.

## 9. Partitioning and file rules

Normalized Parquet uses stable partitions:

```text
dataset=<name>/venue=<venue>/instrument=<instrument>/date=<YYYY-MM-DD>/part-*.parquet
```

Rules:

- Zstandard compression;
- deterministic column order and schema hash;
- immutable files after manifest publication;
- content hash for every file and a root manifest hash;
- temporary output written beside the target on C: and atomically renamed;
- no paths, caches, temp directories, or spill files on E:;
- no full L2 historical download on current storage.

Small-file compaction creates a new dataset version and preserves the parent
manifest. It never mutates a published dataset.

## 10. Storage preflight

Every ingestion or transformation estimates:

- download/input bytes;
- expected final compressed bytes;
- maximum temporary working bytes;
- configured C: reserve;
- current free bytes on the exact target volume.

The job starts only if:

```text
free_bytes - worst_case_required_bytes >= reserve_bytes
```

The reserve is configuration, has no permissive default, and must be set before
Phase 1 ingestion. The job rechecks capacity while running and stops cleanly
before violating the reserve.

## 11. Quality report

Each manifest includes a machine-readable report containing at least:

- row count and time coverage;
- duplicate count by rule;
- missing sequence ranges;
- invalid book intervals;
- nulls and out-of-range values per field;
- timestamp ordering and latency distributions;
- clock-offset diagnostics;
- source outages and reconnects;
- schema and precision violations;
- excluded intervals and reason codes;
- comparison against venue aggregates when available.

Quality thresholds are dataset-specific and must be frozen before evaluation.
A failed threshold makes the dataset inadmissible; it cannot be waived silently
inside an experiment.

## 12. Point-in-time joins

All joins are as-of joins constrained by `available_time`. The selected right
row must be the latest eligible row at or before the left decision time and must
pass its staleness limit. Join code emits the chosen evidence/event IDs for
audit.

Outer joins and resampling never manufacture market observations. Empty buckets
remain explicit unless a feature definition specifies a bounded transformation.

## 13. Split and leakage rules

Experiment views follow `PHASE_0_EVALUATION_PROTOCOL.md`:

- chronological rolling windows;
- purge overlapping label intervals at boundaries;
- apply an embargo based on maximum dependency horizon;
- freeze preprocessing on training data;
- isolate the final holdout from model, feature, and threshold selection;
- store exact row or interval membership in the manifest.

Multiple instruments and venues do not create independent samples when driven
by the same market episode. Effective sample-size calculations must account for
cross-series dependence.

## 14. Admission gate

A dataset is usable in an experiment only when all are true:

1. source use and retention are permitted;
2. raw-to-derived lineage is complete;
3. schema and manifest hashes verify;
4. time semantics and availability model are documented;
5. quality thresholds pass;
6. instrument rules cover the full interval;
7. gap exclusions are explicit;
8. experiment membership is frozen;
9. the exact code and environment are reproducible;
10. no excluded drive or unauthorized source is referenced.

## 15. Phase 1 bootstrap dataset

Before bulk acquisition, Phase 1 builds only a small end-to-end sample:

- one bounded BTC perpetual interval from OKX;
- the matching Binance USD-M reference interval where publicly available;
- trades, best bid/ask, a short verified L2 segment, mark/index, and funding;
- synthetic news/on-chain fixtures solely to validate evidence contracts;
- one deterministic feature and label pipeline;
- one manifest, quality report, replay, and leakage test.

This bootstrap proves the pipeline shape, not strategy quality or profitability.
