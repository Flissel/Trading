# Phase A Edge Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pre-registered, leakage-safe Phase-A pipeline that audits data and costs, evaluates three bounded market-feature hypotheses with calibrated adverse-cost abstention, and publishes an immutable `eligible_candidate` or `no_edge_found` decision without reading the final holdout.

**Architecture:** Keep immutable evidence, hypothesis admission, feature production, model fitting, abstention, fold evaluation, and final decision in separate modules. Existing Ridge, Logistic, and boosted-stump implementations remain the control algorithms, but consume a shared train-fitted feature-vector contract. Every published artifact is canonical-JSON hashed and linked to the capture, dataset, split, hypothesis ledger, cost audit, feature schema, and model configuration.

**Tech Stack:** Python 3.12, standard-library `dataclasses`/`decimal`/`sqlite3`, Pydantic 2, DuckDB, PyArrow, pytest, Ruff, Mypy; no new runtime dependency.

**Spec:** `docs/superpowers/specs/2026-08-25-paper-trading-readiness-design.md`

## Global Constraints

- No real-capital mode, live endpoint, or automatic live promotion exists.
- Do not read or evaluate `final_holdout_ids`; Phase-A data access ends at each fold's `test_end_ns`.
- Use the existing h4 and h16 manifests, three folds, block length 96, 2,000 bootstrap repetitions, and q-value gate `<= 0.10`.
- A candidate needs positive aggregate base and adverse return, at least two positive base folds, at least 200 validation trades, and dominance over no-trade.
- Each hypothesis family permits one primary attempt and at most one reasoned revision; failed attempts remain append-only.
- Missing cost, chronology, lineage, registry, or feature evidence fails closed to ineligible.
- Thresholds and calibration fit validation membership only; model parameters and preprocessing fit training membership only.
- Use `Decimal` for economic values and canonical JSON plus SHA-256 for immutable artifacts.
- Preserve backward verification of existing h1/h4/h16 reports and manifests.
- All implementation follows RED-GREEN-REFACTOR, strict Mypy, Ruff, and Conventional Commits.

## File Map

- `src/trading_bot/audit_contracts.py`: immutable data-quality and cost-evidence records.
- `src/trading_bot/cost_capture.py`: credentials-free public BBO/depth and funding evidence recorder.
- `src/trading_bot/phase_a_audit.py`: build and verify capture-bound Phase-A audit reports.
- `src/trading_bot/hypothesis_registry.py`: immutable hypothesis specifications and SQLite attempt budget.
- `src/trading_bot/market_features.py`: predeclared h4/h16 enriched market features and lineage.
- `src/trading_bot/feature_matrix.py`: train-fitted numeric matrix contract shared by learned baselines.
- `src/trading_bot/uncertainty.py`: train/calibration-safe residual intervals.
- `src/trading_bot/abstention.py`: deterministic adverse-cost expected-value policy.
- `src/trading_bot/phase_a_fold_run.py`: unified fold fit, validation selection, test evaluation, and registration.
- `src/trading_bot/phase_a_decision.py`: complete-fold aggregation and immutable Phase-A decision.
- `src/trading_bot/cli.py`: bounded CLI entry points for audit, fold run, and decision publication.
- Tests mirror each module under `tests/`.

---

### Task 1: Data-quality and cost-evidence contracts

**Files:**
- Create: `src/trading_bot/audit_contracts.py`
- Create: `tests/test_audit_contracts.py`

**Interfaces:**
- Consumes: `Sha256Hex` from `trading_bot.registry`.
- Produces: `DataQualityEvidence`, `CostObservation`, `FundingObservation`, `CostEvidence`, `AuditEligibility`, and `PhaseAAuditMaterial`.

- [ ] **Step 1: Write failing validation tests**

```python
def test_cost_evidence_rejects_favorable_missing_components() -> None:
    with pytest.raises(ValueError, match="cost evidence is incomplete"):
        CostEvidence(
            fee_evidence_id="OKX:fee-tier:fixture",
            fee_bps_per_side=None,
            spread_bps_p50=Decimal("1"),
            spread_bps_p95=Decimal("2"),
            slippage_bps_per_side_p50=None,
            slippage_bps_per_side_p95=None,
            funding_bps_p50=Decimal("0"),
            funding_bps_p95=Decimal("0"),
            observations=(),
            funding_observations=(),
        )


def test_quality_evidence_rejects_unordered_capture_times() -> None:
    with pytest.raises(ValueError, match="chronology"):
        DataQualityEvidence(
            row_count=2,
            first_open_time_ns=20,
            last_open_time_ns=10,
            missing_interval_count=0,
            duplicate_count=0,
            stale_row_count=0,
            cross_venue_anomaly_count=0,
        )
```

- [ ] **Step 2: Run the tests and confirm RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_audit_contracts.py -q`

Expected: collection fails because `trading_bot.audit_contracts` does not exist.

- [ ] **Step 3: Implement strict frozen records**

```python
class AuditEligibility(StrEnum):
    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"


@dataclass(frozen=True, slots=True)
class CostObservation:
    source_id: str
    received_time_ns: int
    spread_bps: Decimal
    slippage_bps_per_side: Decimal

    def __post_init__(self) -> None:
        if not self.source_id or self.received_time_ns < 0:
            raise ValueError("cost observation identity is invalid")
        if self.spread_bps < 0 or self.slippage_bps_per_side < 0:
            raise ValueError("cost observation values must be non-negative")


@dataclass(frozen=True, slots=True)
class FundingObservation:
    source_id: str
    received_time_ns: int
    funding_bps: Decimal

    def __post_init__(self) -> None:
        if not self.source_id or self.received_time_ns < 0:
            raise ValueError("funding observation identity is invalid")
        if not self.funding_bps.is_finite():
            raise ValueError("funding observation must be finite")


@dataclass(frozen=True, slots=True)
class CostEvidence:
    fee_evidence_id: str
    fee_bps_per_side: Decimal | None
    spread_bps_p50: Decimal | None
    spread_bps_p95: Decimal | None
    slippage_bps_per_side_p50: Decimal | None
    slippage_bps_per_side_p95: Decimal | None
    funding_bps_p50: Decimal | None
    funding_bps_p95: Decimal | None
    observations: tuple[CostObservation, ...]
    funding_observations: tuple[FundingObservation, ...]

    def __post_init__(self) -> None:
        required = (
            self.fee_bps_per_side,
            self.spread_bps_p50,
            self.spread_bps_p95,
            self.slippage_bps_per_side_p50,
            self.slippage_bps_per_side_p95,
            self.funding_bps_p50,
            self.funding_bps_p95,
        )
        if any(value is None for value in required):
            raise ValueError("cost evidence is incomplete")
        if not self.fee_evidence_id:
            raise ValueError("fee evidence identity is invalid")
        if not self.observations or not self.funding_observations:
            raise ValueError("cost evidence observations are incomplete")
        if _required(self.spread_bps_p95) < _required(self.spread_bps_p50):
            raise ValueError("spread quantiles are unordered")
        if (
            _required(self.slippage_bps_per_side_p95)
            < _required(self.slippage_bps_per_side_p50)
        ):
            raise ValueError("slippage quantiles are unordered")

    def base_scenario(self) -> CostScenario:
        return CostScenario(
            "base",
            _required(self.fee_bps_per_side),
            Decimal(1),
            _required(self.slippage_bps_per_side_p50),
            _required(self.funding_bps_p50),
        )

    def adverse_scenario(self) -> CostScenario:
        return CostScenario(
            "adverse",
            _required(self.fee_bps_per_side),
            Decimal(1),
            _required(self.slippage_bps_per_side_p95),
            _required(self.funding_bps_p95),
        )


def _required(value: Decimal | None) -> Decimal:
    if value is None:
        raise ValueError("cost evidence is incomplete")
    return value
```

Use these exact remaining contracts:

```python
@dataclass(frozen=True, slots=True)
class DataQualityEvidence:
    row_count: int
    first_open_time_ns: int
    last_open_time_ns: int
    missing_interval_count: int
    duplicate_count: int
    stale_row_count: int
    cross_venue_anomaly_count: int

    def __post_init__(self) -> None:
        counts = (
            self.row_count,
            self.missing_interval_count,
            self.duplicate_count,
            self.stale_row_count,
            self.cross_venue_anomaly_count,
        )
        if any(value < 0 for value in counts):
            raise ValueError("quality counts must be non-negative")
        if self.row_count and self.last_open_time_ns < self.first_open_time_ns:
            raise ValueError("quality chronology is invalid")


@dataclass(frozen=True, slots=True)
class PhaseAAuditMaterial:
    capture_root_hash: str
    dataset_root_hash: str
    interval_ns: int
    data_quality: DataQualityEvidence
    costs: CostEvidence
    eligibility: AuditEligibility
    reason_codes: tuple[str, ...]
```

- [ ] **Step 4: Run focused tests and static checks**

Run: `.venv\Scripts\python.exe -m pytest tests/test_audit_contracts.py -q`

Expected: PASS.

Run: `.venv\Scripts\mypy.exe src/trading_bot/audit_contracts.py tests/test_audit_contracts.py`

Expected: `Success: no issues found`.

- [ ] **Step 5: Commit**

```powershell
git add src/trading_bot/audit_contracts.py tests/test_audit_contracts.py
git commit -m "feat: define phase-a audit contracts"
```

### Task 2: Public market-cost evidence recorder

**Files:**
- Create: `src/trading_bot/cost_capture.py`
- Create: `tests/test_cost_capture.py`
- Modify: `src/trading_bot/cli.py`

**Interfaces:**
- Consumes: injected public OKX BBO/depth/funding payload fetchers, a fixed safe notional, and storage policy.
- Produces: `record_cost_observation(...) -> CostObservation`, `publish_cost_capture(...) -> CostCaptureArtifact`, and `verify_cost_capture(path: Path) -> bool`.

- [ ] **Step 1: Write failing parsing, chronology, and eligibility-floor tests**

```python
def test_cost_observation_uses_receive_time_and_depth_for_slippage() -> None:
    observation = record_cost_observation(
        source_id="OKX:BTC-USDT-SWAP:1",
        received_time_ns=10_000,
        bids=((Decimal("99"), Decimal("2")),),
        asks=((Decimal("101"), Decimal("1")), (Decimal("102"), Decimal("2"))),
        safe_notional=Decimal("150"),
    )
    assert observation.received_time_ns == 10_000
    assert observation.spread_bps == (Decimal("2") / Decimal("100")) * Decimal(10_000)
    assert observation.slippage_bps_per_side > 0


def test_cost_capture_requires_seven_days_and_ten_thousand_observations(tmp_path: Path) -> None:
    artifact = publish_cost_capture(
        observations=cost_observations(count=9_999, span_days=7),
        fee_evidence_id="OKX:fee-tier:fixture",
        fee_bps_per_side=Decimal("2"),
        funding_observations=(
            FundingObservation("OKX:FUNDING:1", 1, Decimal("0.1")),
        ),
        output_directory=tmp_path / "costs",
    )
    document = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    assert document["eligibility"] == "ineligible"
    assert "COST_OBSERVATION_FLOOR_NOT_MET" in document["reason_codes"]
```

Implement exact tests for unordered receive times, negative depth, insufficient
displayed liquidity, immutable output, content-hash mutation, and
storage-reserve denial.

Define `cost_observations(count, span_days)` in the test file by creating
`count` valid `CostObservation` records at evenly spaced receive times from zero
through `span_days * 86_400_000_000_000`, with spread `2` and slippage `1` bps.

- [ ] **Step 2: Run and confirm RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_cost_capture.py -q`

Expected: import failure for `trading_bot.cost_capture`.

- [ ] **Step 3: Implement deterministic cost evidence publication**

Walk the ask side for a buy and bid side for a sell at the same fixed safe
notional. Compute volume-weighted price, half-spread reference, and the worse
per-side slippage. Refuse an observation if either side lacks enough displayed
liquidity. Store canonical raw observations locally and publish a manifest with
row hashes, first/last receive times, source identities, safe notional, fee
evidence identity, funding samples, p50/p95 spread/slippage/funding, eligibility,
and reason codes.

Return this immutable publication pointer:

```python
@dataclass(frozen=True, slots=True)
class CostCaptureArtifact:
    manifest_path: Path
    root_hash: str
    observation_count: int
    eligibility: AuditEligibility
```

Eligibility requires at least 10,000 valid observations spanning at least seven
consecutive 24-hour periods. The recorder is public-data-only and has no private
client or order method. Implement CLI `capture-public-cost-evidence` with
`--workspace-root`, `--output`, `--safe-notional`, `--reserve-bytes`, and bounded
sampling arguments. Unit tests inject payloads and never call the network.

- [ ] **Step 4: Verify focused tests and public-adapter regressions**

Run: `.venv\Scripts\python.exe -m pytest tests/test_cost_capture.py tests/test_depth_adapters.py tests/test_storage_policy.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/trading_bot/cost_capture.py src/trading_bot/cli.py tests/test_cost_capture.py
git commit -m "feat: capture public market cost evidence"
```

### Task 3: Immutable capture-bound Phase-A audit

**Files:**
- Create: `src/trading_bot/phase_a_audit.py`
- Create: `tests/test_phase_a_audit.py`
- Modify: `src/trading_bot/cli.py`

**Interfaces:**
- Consumes: verified candle capture, verified cost-capture manifest, and contracts from Task 1.
- Produces: `build_phase_a_audit(capture_root: Path, cost_capture_path: Path, output_path: Path) -> PhaseAAuditArtifact` and `verify_phase_a_audit(path: Path) -> bool`.

- [ ] **Step 1: Write a failing end-to-end audit test**

```python
def test_phase_a_audit_is_hash_bound_and_reports_missing_intervals(tmp_path: Path) -> None:
    # Define capture_with_gap in this test file with publish_candle_capture:
    # OKX/Binance opens are 1_000, 901_000, and 2_701_000 ms, so one 15m
    # interval is absent while capture and dataset manifests remain valid.
    capture_root = capture_with_gap(tmp_path)
    output = tmp_path / "phase-a-audit.json"
    artifact = build_phase_a_audit(
        capture_root,
        cost_capture_path=complete_cost_capture(tmp_path),
        output_path=output,
    )
    document = json.loads(output.read_text(encoding="utf-8"))
    assert artifact.report_hash == document["report_hash"]
    assert document["data_quality"]["missing_interval_count"] == 1
    assert document["eligibility"] == "ineligible"
    assert "MISSING_INTERVALS" in document["reason_codes"]
    assert verify_phase_a_audit(output) is True
```

The test file defines complete evidence through the Task-2 publisher:

```python
def complete_cost_capture(tmp_path: Path) -> Path:
    artifact = publish_cost_capture(
        observations=cost_observations(count=10_000, span_days=7),
        fee_evidence_id="OKX:fee-tier:fixture",
        fee_bps_per_side=Decimal("2"),
        funding_observations=(
            FundingObservation("OKX:FUNDING:1", 1_000, Decimal("1")),
        ),
        output_directory=tmp_path / "costs",
    )
    return artifact.manifest_path
```

Implement immutable-output, changed-hash, empty-observation, duplicate-row,
and capture/dataset-hash-mismatch tests with those exact failure dimensions.

Name and implement those tests as:

```python
def test_phase_a_audit_refuses_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "audit.json"
    capture = valid_capture(tmp_path)
    cost_capture = complete_cost_capture(tmp_path)
    build_phase_a_audit(
        capture,
        cost_capture_path=cost_capture,
        output_path=output,
    )
    with pytest.raises(PhaseAAuditError, match="immutable"):
        build_phase_a_audit(
            capture,
            cost_capture_path=cost_capture,
            output_path=output,
        )


def test_phase_a_audit_verifier_rejects_changed_hash(tmp_path: Path) -> None:
    output = published_audit(tmp_path)
    document = json.loads(output.read_text(encoding="utf-8"))
    document["reason_codes"] = []
    output.write_text(json.dumps(document), encoding="utf-8")
    assert verify_phase_a_audit(output) is False
```

Use a parametrized publisher fixture for `DUPLICATE_ROWS` and
`CAPTURE_DATASET_HASH_MISMATCH`, and assert the exact reason code for each.

- [ ] **Step 2: Run and confirm RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_phase_a_audit.py -q`

Expected: import failure for `phase_a_audit`.

- [ ] **Step 3: Implement deterministic audit generation**

Load only the verified canonical dataset and a cost-capture manifest that passes
`verify_cost_capture`. Compute expected interval from consecutive primary-venue
opens, count gaps and duplicates, measure reference-venue coverage, and convert
evidence to canonical records. Bind the audit to the cost-capture root hash.
Determine eligibility using explicit reason codes:

```python
reason_codes: list[str] = []
if missing_interval_count > 0:
    reason_codes.append("MISSING_INTERVALS")
if duplicate_count > 0:
    reason_codes.append("DUPLICATE_ROWS")
if cost_capture_eligibility != "eligible":
    reason_codes.append("COST_CAPTURE_INELIGIBLE")
eligibility = "eligible" if not reason_codes else "ineligible"
```

Publish with `canonical_json` and `content_sha256`; refuse overwrite. Implement
CLI command `phase-a-audit` with `--workspace-root`, `--capture`,
`--cost-evidence`, and `--output`; resolve every path and reject any path that
is not relative to the workspace.

- [ ] **Step 4: Verify GREEN and CLI behavior**

Run: `.venv\Scripts\python.exe -m pytest tests/test_phase_a_audit.py -q`

Expected: PASS.

Run: `.venv\Scripts\python.exe -m pytest tests/test_phase_a_audit.py tests/test_market_capture.py tests/test_candle_dataset.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/trading_bot/phase_a_audit.py src/trading_bot/cli.py tests/test_phase_a_audit.py
git commit -m "feat: publish immutable phase-a audits"
```

### Task 4: Hypothesis specification and two-attempt budget

**Files:**
- Create: `src/trading_bot/hypothesis_registry.py`
- Create: `tests/test_hypothesis_registry.py`
- Modify: `src/trading_bot/registry.py`
- Modify: `tests/test_registry.py`

**Interfaces:**
- Consumes: SQLite path and immutable h4/h16 split and audit hashes.
- Produces: `HypothesisFamily`, `HypothesisAttempt`, `HypothesisFoldRun`, `HypothesisRegistry.reserve_attempt(...) -> HypothesisAttempt`, and `HypothesisRegistry.record_fold_run(...) -> HypothesisFoldRun`.

- [ ] **Step 1: Write failing budget and immutability tests**

```python
def test_hypothesis_family_allows_primary_and_one_revision_only(tmp_path: Path) -> None:
    family = HypothesisFamily(
        family_id=UUID("00000000-0000-0000-0000-000000000001"),
        name="multi_scale_trend",
        statement="multi-scale returns add adverse-cost OOS value",
        feature_schema_hash="a" * 64,
        audit_report_hash="b" * 64,
        h4_split_manifest_hash="c" * 64,
        h16_split_manifest_hash="d" * 64,
    )
    with HypothesisRegistry(tmp_path / "metadata.sqlite3") as registry:
        primary = registry.reserve_attempt(family, config_hash=HASH_A, revision_reason=None)
        revision = registry.reserve_attempt(
            family,
            config_hash=HASH_B,
            revision_reason="validation threshold produced too few trades",
        )
        assert primary.attempt_index == 0
        assert revision.attempt_index == 1
        with pytest.raises(HypothesisBudgetError, match="attempt budget exhausted"):
            registry.reserve_attempt(family, config_hash=HASH_C, revision_reason="third search")
```

Also assert that reusing the same family ID with changed semantic content raises `RegistryConflictError` and that a revision without a reason is rejected.

- [ ] **Step 2: Run and confirm RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_hypothesis_registry.py -q`

Expected: import failure.

- [ ] **Step 3: Implement models and append-only tables**

```python
class HypothesisFamily(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    family_id: UUID
    name: Literal[
        "multi_scale_trend",
        "volatility_liquidity_regime",
        "cross_venue_activity",
    ]
    statement: str
    feature_schema_hash: Sha256Hex
    audit_report_hash: Sha256Hex
    h4_split_manifest_hash: Sha256Hex
    h16_split_manifest_hash: Sha256Hex


class HypothesisAttempt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    attempt_id: UUID
    family_id: UUID
    attempt_index: Literal[0, 1]
    config_hash: Sha256Hex
    revision_reason: str | None
    created_at_ns: int


class HypothesisFoldRun(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    attempt_id: UUID
    horizon_bars: Literal[4, 16]
    model_kind: Literal["ridge", "logistic", "boosted_stumps"]
    fold_index: Literal[0, 1, 2]
    status: Literal["completed", "failed"]
    artifact_hash: Sha256Hex | None
    error_code: str | None
    recorded_at_ns: int
```

Create `hypothesis_families`, `hypothesis_attempts`, and
`hypothesis_fold_runs` tables transactionally. One attempt is one frozen
family-level campaign across the three model kinds, both horizons, and all
three folds; its `config_hash` binds that complete 18-run grid. Count existing
attempts while reserving; index 0 requires no revision reason, index 1 requires
a non-empty reason, and index 2 is impossible. Give fold runs a unique key on
`(attempt_id, horizon_bars, model_kind, fold_index)`. Replaying an identical
outcome is idempotent; changing an already recorded outcome raises
`RegistryConflictError`. A completed outcome requires `artifact_hash` and no
error code; a failed outcome requires an error code and no artifact hash.

Add tests for all four fold-outcome invariants, including a conflicting replay.

- [ ] **Step 4: Verify focused and registry regression tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_hypothesis_registry.py tests/test_registry.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/trading_bot/hypothesis_registry.py src/trading_bot/registry.py tests/test_hypothesis_registry.py tests/test_registry.py
git commit -m "feat: enforce phase-a experiment budgets"
```

### Task 5: Predeclared enriched market features

**Files:**
- Create: `src/trading_bot/market_features.py`
- Create: `tests/test_market_features.py`
- Modify: `src/trading_bot/bar_research.py`
- Modify: `tests/test_bar_research.py`

**Interfaces:**
- Consumes: chronological `ResearchBar` tuples and horizon-aware `BarSample` records.
- Produces: `MarketFeatureSchema`, `EnrichedBarSample`, and `build_enriched_samples(primary, reference, *, horizon_bars, schema) -> tuple[EnrichedBarSample, ...]`.

- [ ] **Step 1: Write failing point-in-time and numeric tests**

```python
def test_enriched_features_use_only_ids_available_at_decision() -> None:
    schema = phase_a_schema()
    samples = build_enriched_samples(primary_bars(), reference_bars(), horizon_bars=4, schema=schema)
    sample = samples[20]
    admitted = {
        bar.source_id
        for bar in primary_bars() + reference_bars()
        if bar.available_time_ns <= sample.decision_time_ns
    }
    assert set(sample.feature_input_source_ids).issubset(admitted)
    assert sample.label_available_time_ns > sample.decision_time_ns


def test_multi_scale_returns_and_regime_values_are_exact() -> None:
    sample = build_enriched_samples(monotonic_bars(), (), horizon_bars=4, schema=phase_a_schema())[30]
    assert sample.return_4 == Decimal(130) / Decimal(126) - Decimal(1)
    assert sample.return_16 == Decimal(130) / Decimal(114) - Decimal(1)
    assert sample.realized_volatility_16 >= 0
    assert sample.volume_zscore_32 is not None
```

Implement a perturbation test that changes every bar strictly after a decision
and asserts the sample's features and lineage remain byte-identical while its
future label may change.

Define the test-local fixtures with this deterministic constructor:

```python
def bars(prices: tuple[int, ...], venue: str) -> tuple[ResearchBar, ...]:
    return tuple(
        ResearchBar(
            source_id=f"{venue}:{index}",
            venue=venue,
            open_time_ns=index * 900_000_000_000,
            available_time_ns=(index + 1) * 900_000_000_000,
            close=Decimal(price),
            high=Decimal(price + 1),
            low=Decimal(price - 1),
            base_volume=Decimal(100 + index),
        )
        for index, price in enumerate(prices)
    )
```

`monotonic_bars()` returns `bars(tuple(range(100, 180)), "OKX")`.
`primary_bars()` uses the same OKX sequence and `reference_bars()` uses prices
one unit lower with venue `BINANCE`. `phase_a_schema()` returns the exact eight
ordered fields shown in `EnrichedBarSample`.

- [ ] **Step 2: Run and confirm RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_market_features.py -q`

Expected: import failure.

- [ ] **Step 3: Implement explicit schema and feature calculations**

```python
@dataclass(frozen=True, slots=True)
class EnrichedBarSample:
    base: BarSample
    feature_input_source_ids: tuple[str, ...]
    return_4: Decimal | None
    return_16: Decimal | None
    realized_volatility_16: Decimal | None
    realized_volatility_64: Decimal | None
    range_ratio_16: Decimal | None
    volume_zscore_32: Decimal | None
    basis_mean_16: Decimal | None
    basis_change_4: Decimal | None

    @property
    def decision_time_ns(self) -> int:
        return self.base.decision_time_ns
```

Calculate windows ending at the current bar only. Return `None` until a complete lookback exists. Cross-venue features require matched reference bars whose `available_time_ns <= decision_time_ns`. Hash the ordered field definitions as `MarketFeatureSchema.schema_hash`.

- [ ] **Step 4: Verify GREEN and existing horizon behavior**

Run: `.venv\Scripts\python.exe -m pytest tests/test_market_features.py tests/test_bar_research.py -q`

Expected: PASS.

Run: `.venv\Scripts\mypy.exe src/trading_bot/market_features.py tests/test_market_features.py`

Expected: success.

- [ ] **Step 5: Commit**

```powershell
git add src/trading_bot/market_features.py src/trading_bot/bar_research.py tests/test_market_features.py tests/test_bar_research.py
git commit -m "feat: add predeclared market feature families"
```

### Task 6: Shared train-fitted feature matrix

**Files:**
- Create: `src/trading_bot/feature_matrix.py`
- Create: `tests/test_feature_matrix.py`
- Modify: `src/trading_bot/linear_baseline.py`
- Modify: `src/trading_bot/logistic_baseline.py`
- Modify: `src/trading_bot/tree_baseline.py`
- Test: `tests/test_linear_baseline.py`
- Test: `tests/test_logistic_baseline.py`
- Test: `tests/test_tree_baseline.py`

**Interfaces:**
- Consumes: `EnrichedBarSample` and an ordered hypothesis-specific field list.
- Produces: `FeatureMatrixSpec`, `FittedFeatureMatrix`, `fit_feature_matrix(...)`, `transform_feature_matrix(...)`, `RidgeVectorModel`, `LogisticVectorModel`, and `BoostedStumpVectorModel` row APIs.

- [ ] **Step 1: Write failing training-boundary tests**

```python
def test_feature_matrix_statistics_fit_training_only() -> None:
    fitted = fit_feature_matrix(training_samples(), fields=("return_4", "basis_change_4"))
    before = fitted
    transformed = transform_feature_matrix(fitted, extreme_test_samples())
    assert fitted == before
    assert transformed.rows[0][0] > Decimal("100")


def test_missing_values_use_training_median_and_missing_indicator() -> None:
    fitted = fit_feature_matrix(training_with_missing_basis(), fields=("basis_change_4",))
    transformed = transform_feature_matrix(fitted, test_with_missing_basis())
    assert transformed.rows[0] == (Decimal("0"), Decimal("1"))
```

Define the test sample builder explicitly:

```python
def enriched(sample_id: str, return_4: Decimal, basis: Decimal | None) -> EnrichedBarSample:
    base = BarSample(
        sample_id=sample_id,
        decision_time_ns=int(sample_id.removeprefix("s")),
        label_available_time_ns=int(sample_id.removeprefix("s")) + 1,
        input_source_ids=(sample_id,),
        observed_return=Decimal(0),
        range_bps=Decimal(1),
        volume_change=Decimal(0),
        cross_venue_basis=basis,
        forward_return=Decimal(0),
    )
    return EnrichedBarSample(
        base=base,
        feature_input_source_ids=(sample_id,),
        return_4=return_4,
        return_16=None,
        realized_volatility_16=Decimal("0.01"),
        realized_volatility_64=None,
        range_ratio_16=Decimal(1),
        volume_zscore_32=Decimal(0),
        basis_mean_16=basis,
        basis_change_4=basis,
    )
```

`training_samples()` returns `s1/s2/s3` with return values `1/2/3` and basis
`1/None/3`; `extreme_test_samples()` returns `s4` with both values `1000`.
The missing-basis helpers select these same records so the expected median,
standardized zero, and missing indicator are deterministic.

- [ ] **Step 2: Run and confirm RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_feature_matrix.py -q`

Expected: import failure.

- [ ] **Step 3: Implement immutable preprocessing**

```python
@dataclass(frozen=True, slots=True)
class FittedFeatureMatrix:
    fields: tuple[str, ...]
    medians: tuple[Decimal, ...]
    means: tuple[Decimal, ...]
    scales: tuple[Decimal, ...]
    schema_hash: str


@dataclass(frozen=True, slots=True)
class TransformedFeatureMatrix:
    sample_ids: tuple[str, ...]
    rows: tuple[tuple[Decimal, ...], ...]
```

For each declared field, fit median, mean, and non-zero scale on training only. Emit a standardized value plus a missing indicator. Reject unknown fields and non-finite values. Generalize the three model fit functions to consume numeric rows and target tuples while keeping their existing `BarSample` wrappers backward compatible.

Create these exact numeric model records:

```python
@dataclass(frozen=True, slots=True)
class RidgeVectorModel:
    intercept: Decimal
    coefficients: tuple[Decimal, ...]


@dataclass(frozen=True, slots=True)
class LogisticVectorModel:
    intercept: Decimal
    coefficients: tuple[Decimal, ...]


@dataclass(frozen=True, slots=True)
class BoostedStumpVectorModel:
    base_value: Decimal
    learning_rate: Decimal
    stumps: tuple[RegressionStump, ...]
```

Expose exact functions `fit_ridge_rows(rows, targets, *, alpha) ->
RidgeVectorModel`, `predict_ridge_row(model, row) -> Decimal`,
`fit_logistic_rows(rows, labels, *, l2, iterations, learning_rate) ->
LogisticVectorModel`, `predict_probability_row(model, row) -> Decimal`,
`fit_boosted_stumps_rows(rows, targets, *, estimator_count, learning_rate,
minimum_leaf_samples, maximum_split_candidates) -> BoostedStumpVectorModel`,
and `predict_tree_row(model, row) -> Decimal`. Move the existing deterministic
solver loops to operate on numeric rows, validate equal row widths/counts, and
make the old `BarSample` functions construct their legacy four-field rows before
delegating.

- [ ] **Step 4: Verify new and old model tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_feature_matrix.py tests/test_linear_baseline.py tests/test_logistic_baseline.py tests/test_tree_baseline.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/trading_bot/feature_matrix.py src/trading_bot/linear_baseline.py src/trading_bot/logistic_baseline.py src/trading_bot/tree_baseline.py tests/test_feature_matrix.py tests/test_linear_baseline.py tests/test_logistic_baseline.py tests/test_tree_baseline.py
git commit -m "refactor: share train-fitted feature matrices"
```

### Task 7: Residual uncertainty and adverse-cost abstention

**Files:**
- Create: `src/trading_bot/uncertainty.py`
- Create: `src/trading_bot/abstention.py`
- Create: `tests/test_uncertainty.py`
- Create: `tests/test_abstention.py`

**Interfaces:**
- Consumes: chronological calibration predictions/outcomes, observed spread, and `CostScenario`.
- Produces: `ResidualIntervalCalibrator`, `ForecastEnvelope`, `AbstentionDecision`, `fit_residual_interval(...)`, and `decide_with_abstention(...)`.

- [ ] **Step 1: Write failing calibration and conservative-EV tests**

```python
def test_residual_interval_is_fit_from_calibration_membership_only() -> None:
    calibrator = fit_residual_interval(
        predictions=(Decimal("0.01"), Decimal("0.02"), Decimal("0.03")),
        outcomes=(Decimal("0.00"), Decimal("0.03"), Decimal("0.01")),
        coverage=Decimal("0.80"),
    )
    assert calibrator.lower_residual == Decimal("-0.02")
    assert calibrator.upper_residual == Decimal("0.01")


def test_adverse_cost_non_positive_forecast_abstains() -> None:
    decision = decide_with_abstention(
        ForecastEnvelope(Decimal("0.001"), Decimal("-0.002"), Decimal("0.004")),
        CostScenario("adverse", Decimal("2"), Decimal("2"), Decimal("2"), Decimal("1")),
        observed_spread_bps=Decimal("2"),
    )
    assert decision.signal == 0
    assert decision.reason_code == "ADVERSE_EV_NON_POSITIVE"
```

Implement symmetry tests for short forecasts, exact-zero conservative EV,
missing spread, and invalid intervals.

Use these exact assertions: a forecast with `upper < -round_trip_cost` produces
signal `-1`; equality with either cost boundary produces signal `0`; negative
spread raises `ValueError`; and `lower > point` or `point > upper` raises
`ValueError` in `ForecastEnvelope.__post_init__`.

- [ ] **Step 2: Run and confirm RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_uncertainty.py tests/test_abstention.py -q`

Expected: import failure.

- [ ] **Step 3: Implement deterministic calibration and policy**

```python
@dataclass(frozen=True, slots=True)
class ForecastEnvelope:
    point: Decimal
    lower: Decimal
    upper: Decimal


def decide_with_abstention(
    forecast: ForecastEnvelope,
    costs: CostScenario,
    *,
    observed_spread_bps: Decimal,
) -> AbstentionDecision:
    round_trip = costs.round_trip_cost_return(observed_spread_bps)
    if forecast.lower > round_trip:
        return AbstentionDecision(1, forecast.lower - round_trip, "POSITIVE_ADVERSE_EV")
    if forecast.upper < -round_trip:
        return AbstentionDecision(-1, -forecast.upper - round_trip, "NEGATIVE_ADVERSE_EV")
    return AbstentionDecision(0, Decimal(0), "ADVERSE_EV_NON_POSITIVE")
```

Use deterministic nearest-rank residual quantiles and record calibration membership hash and coverage. Reject empty or unordered calibration inputs.

- [ ] **Step 4: Verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/test_uncertainty.py tests/test_abstention.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/trading_bot/uncertainty.py src/trading_bot/abstention.py tests/test_uncertainty.py tests/test_abstention.py
git commit -m "feat: add adverse-cost forecast abstention"
```

### Task 8: Unified Phase-A fold runner

**Files:**
- Create: `src/trading_bot/phase_a_fold_run.py`
- Create: `tests/test_phase_a_fold_run.py`
- Modify: `src/trading_bot/cli.py`

**Interfaces:**
- Consumes: verified capture, h4/h16 split, eligible audit, reserved hypothesis attempt, feature schema, model kind, and frozen evaluation settings.
- Produces: `run_phase_a_fold(config: PhaseAFoldConfig) -> PhaseAFoldArtifact` and `verify_phase_a_fold(path: Path) -> bool`.

- [ ] **Step 1: Write a failing membership-isolation integration test**

```python
def test_phase_a_fold_fits_trains_calibrates_and_tests_on_disjoint_membership(tmp_path: Path) -> None:
    artifact = run_phase_a_fold(phase_a_fold_fixture(tmp_path, model_kind="ridge"))
    document = json.loads(artifact.output_path.read_text(encoding="utf-8"))
    assert document["horizon_bars"] == 4
    assert document["train_membership_hash"] != document["calibration_membership_hash"]
    assert document["calibration_membership_hash"] != document["selection_membership_hash"]
    assert document["selection_membership_hash"] != document["test_membership_hash"]
    assert document["final_holdout_read_count"] == 0
    assert verify_phase_a_fold(artifact.output_path) is True


def test_phase_a_fold_never_requests_data_after_test_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = phase_a_fold_fixture(tmp_path, model_kind="ridge")
    test_end_ns = split_test_end(config.split_manifest_path, config.fold_index)
    requested_boundaries: list[int | None] = []
    real_loader = phase_a_fold_run.load_research_bars

    def guarded_loader(
        dataset_root: Path,
        *,
        available_before_ns: int | None = None,
    ) -> tuple[ResearchBar, ...]:
        requested_boundaries.append(available_before_ns)
        if available_before_ns is None or available_before_ns > test_end_ns:
            raise AssertionError("runner attempted to read beyond test boundary")
        return real_loader(dataset_root, available_before_ns=available_before_ns)

    monkeypatch.setattr(phase_a_fold_run, "load_research_bars", guarded_loader)
    run_phase_a_fold(config)
    assert requested_boundaries == [test_end_ns]
```

Implement rejection tests for an ineligible audit, unreserved attempt,
mismatched split hash, fewer than 200 selected validation trades, unavailable
horizon sample, and immutable output overwrite.

Implement `phase_a_fold_fixture` in the test file by publishing a 2,048-bar
directional capture, a horizon-specific manifest, an eligible audit, and a
reserved primary attempt into `tmp_path`. The helper returns a complete
`PhaseAFoldConfig`; keyword overrides may change only `model_kind`, input hash,
trade floor, and output path. Each rejection test changes one field and asserts
the exact corresponding `PhaseAFoldError` message.

- [ ] **Step 2: Run and confirm RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_phase_a_fold_run.py -q`

Expected: import failure.

- [ ] **Step 3: Implement the frozen fold flow**

```python
@dataclass(frozen=True, slots=True)
class PhaseAFoldConfig:
    capture_root: Path
    split_manifest_path: Path
    audit_path: Path
    registry_path: Path
    output_path: Path
    hypothesis_family_id: UUID
    attempt_id: UUID
    model_kind: Literal["ridge", "logistic", "boosted_stumps"]
    fold_index: int
    random_seed: int
    block_length: int = 96
    bootstrap_repetitions: int = 2000
    minimum_selection_trades: int = 200
```

Load bars with `available_before_ns=test_end_ns`. Build enriched samples at the manifest horizon. Split validation chronologically into calibration and selection halves. Fit matrix and model on training, residual interval on calibration, and no-trade/adverse-EV threshold on selection. Evaluate abstained signals on test under the same base/adverse costs as controls. Include all input hashes, counts, membership hashes, model parameters, interval, threshold, trade counts, bootstrap result, q-value, and reason codes. Register completed and failed attempts append-only.

Map hypotheses to ordered fields exactly:

| Hypothesis | Fields |
| --- | --- |
| `multi_scale_trend` | `return_4`, `return_16` |
| `volatility_liquidity_regime` | `realized_volatility_16`, `realized_volatility_64`, `range_ratio_16` |
| `cross_venue_activity` | `volume_zscore_32`, `basis_mean_16`, `basis_change_4` |

Ridge and boosted stumps predict forward return directly. Logistic fits the
direction label and converts calibrated probability to expected return using
training-only conditional magnitudes:

```python
expected_return = (
    calibrated_probability * mean_positive_training_return
    + (Decimal(1) - calibrated_probability) * mean_nonpositive_training_return
)
```

Fit the Platt calibrator and the residual interval on the calibration half;
select only the abstention/trade threshold on the later selection half. Use
`spread_bps_p50` for base evaluation and `spread_bps_p95` for adverse evaluation,
with the corresponding `CostEvidence` scenarios.

- [ ] **Step 4: Verify focused and legacy fold tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_phase_a_fold_run.py tests/test_linear_fold_run.py tests/test_logistic_fold_run.py tests/test_tree_fold_run.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/trading_bot/phase_a_fold_run.py src/trading_bot/cli.py tests/test_phase_a_fold_run.py
git commit -m "feat: run registered phase-a folds"
```

### Task 9: Aggregate, paired dominance, and immutable Phase-A decision

**Files:**
- Create: `src/trading_bot/phase_a_decision.py`
- Create: `tests/test_phase_a_decision.py`
- Modify: `src/trading_bot/cli.py`

**Interfaces:**
- Consumes: exactly three verified Phase-A fold reports per attempt plus verified existing baseline aggregate and hypothesis/audit records.
- Produces: `build_phase_a_decision(inputs: PhaseADecisionInputs, output_path: Path) -> PhaseADecisionArtifact` and `verify_phase_a_decision(path: Path) -> bool`.

Use these exact boundary records:

```python
@dataclass(frozen=True, slots=True)
class PhaseADecisionInputs:
    fold_report_paths: tuple[Path, ...]
    baseline_aggregate_paths: tuple[Path, Path]
    audit_path: Path
    registry_path: Path
    expected_feature_schema_hash: str


@dataclass(frozen=True, slots=True)
class PhaseADecisionArtifact:
    output_path: Path
    decision_hash: str
    decision_status: Literal["eligible_candidate", "no_edge_found"]
    strongest_eligible_candidate: str | None
    candidate_count: int
```

A candidate identity is the exact tuple `(hypothesis_family_id, attempt_id,
model_kind, horizon_bars)`. Require exactly fold indexes 0, 1, and 2 for every
candidate identity. Compare h4 candidates only with the verified h4 baseline
aggregate and h16 candidates only with the h16 aggregate. The decision may
admit a candidate at one horizon without admitting the same family/model at the
other horizon, but it must publish both outcomes.

- [ ] **Step 1: Write failing eligible and no-edge tests**

```python
def test_phase_a_decision_requires_every_economic_gate(tmp_path: Path) -> None:
    output = tmp_path / "decision.json"
    build_phase_a_decision(adverse_negative_inputs(tmp_path), output_path=output)
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["decision_status"] == "no_edge_found"
    assert document["strongest_eligible_candidate"] is None
    assert "AGGREGATE_ADVERSE_NET_NON_POSITIVE" in document["candidates"][0]["reason_codes"]


def test_phase_a_decision_excludes_incomplete_fold_family(tmp_path: Path) -> None:
    with pytest.raises(PhaseADecisionError, match="exactly three folds"):
        build_phase_a_decision(two_fold_inputs(tmp_path), output_path=tmp_path / "decision.json")
```

Also test q-value `0.10` passes, `0.1000001` fails, 1/3 positive folds fails, fewer than 200 selected trades fails, and audit/split/feature hash disagreement fails closed.

Create test-local verified fold documents with a `write_phase_a_fold_report`
helper that canonicalizes the supplied candidate metrics and adds its report
hash. `adverse_negative_inputs` writes folds with positive base return and
negative adverse return. `two_fold_inputs` deliberately omits fold index 2.
Every boundary test changes only the named metric.

- [ ] **Step 2: Run and confirm RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_phase_a_decision.py -q`

Expected: import failure.

- [ ] **Step 3: Implement complete-family gating**

Aggregate base/adverse returns, positive folds, minimum selected trades, and maximum q-value. Compare every Phase-A candidate with no-trade and the strongest existing simple baseline. Publish:

```python
decision_status = (
    "eligible_candidate"
    if eligible_candidate_names
    else "no_edge_found"
)
```

Record every rejected candidate and reason; do not publish only the winner.
Include `final_holdout_status: "locked"` and no holdout membership or outcome
data. Implement CLI command `phase-a-decision` with repeated `--fold-report`,
exactly two repeated `--baseline-aggregate` values (h4 and h16), one `--output`,
and workspace-contained paths.

- [ ] **Step 4: Verify GREEN and dominance regression**

Run: `.venv\Scripts\python.exe -m pytest tests/test_phase_a_decision.py tests/test_fold_aggregate.py tests/test_model_dominance.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/trading_bot/phase_a_decision.py src/trading_bot/cli.py tests/test_phase_a_decision.py
git commit -m "feat: publish frozen phase-a edge decisions"
```

### Task 10: Deterministic orchestration and artifact verifier

**Files:**
- Create: `src/trading_bot/phase_a_pipeline.py`
- Create: `tests/test_phase_a_pipeline.py`
- Create in Task 11: `configs/phase-a-run-v1.json`
- Modify: `src/trading_bot/cli.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: one immutable run specification listing audit, hypotheses, attempts, h4/h16 manifests, seeds, and output names.
- Produces: `run_phase_a_pipeline(workspace_root: Path, spec: PhaseAPipelineSpec, *, dry_run: bool = False) -> PhaseAPipelineArtifact` and `verify_phase_a_pipeline(path: Path) -> bool`.

Use these exact public contracts:

```python
PhaseAModelKind = Literal["ridge", "logistic", "boosted_stumps"]


class PhaseAPipelineSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal["1.0.0"]
    capture_root: str
    audit_path: str
    registry_path: str
    h4_split_manifest_path: str
    h16_split_manifest_path: str
    baseline_aggregate_paths: tuple[str, str]
    output_directory: str
    hypothesis_attempt_ids: tuple[str, str, str]
    model_kinds: tuple[PhaseAModelKind, ...]
    seeds: tuple[int, ...]
    feature_schema_hash: str
    block_length: int = 96
    bootstrap_repetitions: int = 2000
    minimum_selection_trades: int = 200

    @model_validator(mode="after")
    def validate_frozen_shape(self) -> Self:
        if self.model_kinds != (
            "ridge",
            "logistic",
            "boosted_stumps",
        ):
            raise ValueError("model kinds must use the frozen phase-a order")
        if len(set(self.hypothesis_attempt_ids)) != 3:
            raise ValueError("phase-a requires three distinct primary attempts")
        if len(self.seeds) != 54 or len(set(self.seeds)) != 54:
            raise ValueError("phase-a requires 54 distinct fixed seeds")
        if (
            self.block_length != 96
            or self.bootstrap_repetitions != 2000
            or self.minimum_selection_trades != 200
        ):
            raise ValueError("phase-a numeric gates are frozen")
        return self


@dataclass(frozen=True, slots=True)
class PhaseAPipelineArtifact:
    manifest_path: Path | None
    pipeline_hash: str
    planned_fold_count: int
    created_paths: tuple[Path, ...]
    reused_paths: tuple[Path, ...]
```

Every string path must be workspace-relative, normalized, and unable to escape
the workspace root. `manifest_path` is `None` on a dry run. The canonical spec
hash excludes no fields.

- [ ] **Step 1: Write a failing dry-run and resume test**

```python
def test_phase_a_pipeline_dry_run_lists_outputs_without_writing(tmp_path: Path) -> None:
    result = run_phase_a_pipeline(tmp_path, pipeline_spec(), dry_run=True)
    assert result.planned_fold_count == 54
    assert result.created_paths == ()
    assert result.manifest_path is None


def test_phase_a_pipeline_resumes_only_verified_immutable_outputs(tmp_path: Path) -> None:
    spec = pipeline_spec()
    first = run_phase_a_pipeline(tmp_path, spec)
    second = run_phase_a_pipeline(tmp_path, spec)
    assert second.pipeline_hash == first.pipeline_hash
    assert second.reused_paths == first.created_paths
```

The 54 folds are three hypothesis families times three model kinds times three
folds times two horizons. `pipeline_spec` constructs all three primary
hypothesis attempts with fixed seeds and workspace-relative output paths. Add
`test_phase_a_pipeline_refuses_invalid_existing_output`, place invalid JSON at
the first planned output, and assert `PhaseAPipelineError` contains
`existing output failed verification`.

- [ ] **Step 2: Run and confirm RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_phase_a_pipeline.py -q`

Expected: import failure.

- [ ] **Step 3: Implement hash-bound orchestration**

Implement the frozen Pydantic model above and compute its canonical hash before
execution. For each planned output: reuse only if the correct verifier passes
and all linkage hashes equal the spec; otherwise fail. Never delete or
overwrite. Publish a final pipeline manifest containing every artifact hash and
decision hash.

Implement CLI `phase-a-pipeline --spec <path> [--dry-run]`. Document commands,
artifacts, and the fact that `no_edge_found` is a valid result.

- [ ] **Step 4: Verify pipeline and full suite**

Run: `.venv\Scripts\python.exe -m pytest tests/test_phase_a_pipeline.py -q`

Expected: PASS.

Run: `.venv\Scripts\python.exe -m pytest -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/trading_bot/phase_a_pipeline.py src/trading_bot/cli.py tests/test_phase_a_pipeline.py README.md
git commit -m "feat: orchestrate reproducible phase-a research"
```

### Task 11: Real h4/h16 Phase-A run and decision checkpoint

**Files:**
- Create locally and keep ignored: `artifacts/phase-a-*.json`, `artifacts/phase-a-metadata.sqlite3`
- Create and commit before execution: `configs/phase-a-run-v1.json`
- Modify: `P1_15_DECISION_2026-08-25.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: verified 580-day capture, existing h4/h16 manifests, admitted cost audit, and the frozen Phase-A pipeline spec.
- Produces: verified local audit/fold/aggregate/decision artifacts and a committed evidence summary.

- [ ] **Step 1: Publish the frozen run specification and perform dry run**

Create `configs/phase-a-run-v1.json` with the verified audit, h4/h16 manifest,
feature-schema, hypothesis-family, model-config, seed, block-length, bootstrap,
and minimum-trade hashes/values produced by Tasks 1-10. Use only workspace-
relative paths. Commit it before running:

```powershell
git add configs/phase-a-run-v1.json
git commit -m "chore: freeze phase-a run specification"
```

Run: `.venv\Scripts\python.exe -m trading_bot phase-a-pipeline --workspace-root . --spec configs/phase-a-run-v1.json --dry-run`

Expected: lists 54 primary fold outputs, family/model aggregates, and one final decision; reports zero writes and no final-holdout path.

- [ ] **Step 2: Execute the immutable pipeline**

Run: `.venv\Scripts\python.exe -m trading_bot phase-a-pipeline --workspace-root . --spec configs/phase-a-run-v1.json`

Expected: all planned artifacts publish once; a second identical run reuses verified outputs and produces the same pipeline hash.

- [ ] **Step 3: Verify linkage and holdout lock**

Run: `.venv\Scripts\python.exe -m trading_bot verify-phase-a --workspace-root . --pipeline-manifest artifacts/phase-a-pipeline-v1.json`

Expected: capture, dataset, audit, hypothesis, feature, fold, aggregate, and decision hashes verify; output states `final_holdout_status=locked` and `final_holdout_read_count=0`.

- [ ] **Step 4: Run all quality gates**

Run: `.venv\Scripts\python.exe -m pytest -q`

Expected: all tests pass.

Run: `.venv\Scripts\ruff.exe format --check .`

Expected: all files formatted.

Run: `.venv\Scripts\ruff.exe check .`

Expected: all checks pass.

Run: `.venv\Scripts\mypy.exe src tests`

Expected: no issues.

Run: `uv lock --check`

Expected: lock is current.

Run: `uv build`

Expected: source distribution and wheel build successfully.

- [ ] **Step 5: Record the evidence-backed decision**

Update `P1_15_DECISION_2026-08-25.md` and `README.md` with exact artifact hashes, economic metrics, decision status, and explicit non-claims. If status is `no_edge_found`, state that Phase B proceeds `hold_only`; do not soften the result because a candidate is numerically best.

- [ ] **Step 6: Commit documentation only**

```powershell
git add P1_15_DECISION_2026-08-25.md README.md
git commit -m "docs: record phase-a edge decision"
```

## Final Phase-A Review Gate

Before moving to the Phase-B design and plan, verify:

- every task commit is present in order;
- the worktree contains no uncommitted product changes;
- all local artifacts are ignored and independently verified;
- the final decision lists every admitted and failed attempt;
- the final holdout was never read;
- `eligible_candidate` is claimed only if every frozen gate passed;
- otherwise runtime authority is exactly `hold_only`.
