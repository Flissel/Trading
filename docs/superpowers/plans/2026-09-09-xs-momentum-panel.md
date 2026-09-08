# Cross-Sectional Daily Momentum Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the panel research line that evaluates the pre-registered experiment family `xs_momentum_panel_v1` — six fixed momentum and reversal members on a Binance USD-M USDT-perpetual panel with weekly holding — through the repository's existing bootstrap, Benjamini-Hochberg, walk-forward and registry primitives.

**Architecture:** One sample is one weekly rebalance date, never a coin-day. That keeps `SplitSample`, `WalkForwardConfig`, `build_walk_forward_views` and the manifest format unchanged, because the sample series stays strictly chronological. The panel lives *inside* a sample: a member's "signal" is a weight vector over eligible contracts and its "outcome" is the realised portfolio net return. New `panel_*` modules are isolated from the 15-minute BTC line; no existing module is modified except `cli.py`, which gains four subcommands.

**Tech Stack:** Python 3.12 (`>=3.12,<3.13`), stdlib `decimal`/`urllib`/`zipfile`/`csv`, `pyarrow` for Parquet, `duckdb` for reads, `pydantic` v2 for the frozen config, `pytest` + `mypy --strict` + `ruff`. No torch, no pandas, no numpy, no new dependency.

**Spec:** `docs/superpowers/specs/2026-09-08-xs-momentum-family-design.md` (approved 2026-09-09, commit `4f91042`). The protocol addendum is `PHASE_0_EVALUATION_PROTOCOL.md` section 16.

## Global Constraints

- **Money is `Decimal`, never `float`.** `canonical.py` raises `CanonicalizationError` on binary floats. The only permitted float conversion is inside the deflated-Sharpe normal CDF (Task 10), which is a reported statistic and never a gate input.
- **Time is integer nanoseconds.** `1 day = 86_400_000_000_000`; `7 days = 604_800_000_000_000`; `14 days = 1_209_600_000_000_000`; `91 days = 7_862_400_000_000_000`; `182 days = 15_724_800_000_000_000`; `365 days = 31_536_000_000_000_000`.
- **Artifacts are immutable.** Every publisher raises if its output path exists, writes to a `.tmp` sibling, then `replace()`s. Follow `walk_forward_run.py:77-78` and `candle_dataset.py:67-68`.
- **Hashing is canonical.** Use `trading_bot.canonical.canonical_json` and `content_sha256`. Every report stores its own hash under a key removed from the hashed material, exactly as `walk_forward_run.py:104-107`.
- **Dataclasses are `@dataclass(frozen=True, slots=True)`.** Pydantic models use `model_config = ConfigDict(frozen=True, extra="forbid")`.
- **Storage.** Every write path calls `StoragePolicy(workspace_root, reserve_bytes).authorize(...)` with `reserve_bytes` defaulting to `20_000_000_000`. `E:` stays excluded. Panel capture declares `worst_case_required_bytes=500_000_000`.
- **Network hosts** are restricted to `data.binance.vision`, `s3-ap-northeast-1.amazonaws.com` and `fapi.binance.com` by `panel_capture._ALLOWED_PANEL_HOSTS`. `market_capture._ALLOWED_HOSTS` is **not** modified; the panel line gets its own client so the 15-minute line is untouched.
- **No test may touch the network.** All capture tests inject a fake fetch callable.
- **Frozen numbers** (from spec sections 5, 8, 9), identical in code defaults and in `configs/xs-momentum-panel-v1.json`: quintile long-short with `0.5` gross per leg; minimum quintile size `8`; minimum eligible universe `40`; universe cap `100`; liquidity floor `5_000_000` USDT median over `30` days; minimum history `91` days; volatility window `30` days with floor `0.20`; time-series weight cap `2 / n_eligible`; fee `5` bps per side; slippage `5`/`10` bps per side by tier (adverse `10`/`20`); block length `4`; bootstrap repetitions `2000`; `random_seed = 17`; confidence `0.95`; BH gate `q <= 0.10`; pooled episode floor `200`; concentration limit `0.5`.
- **Member and control names** are exactly: members `xs_mom_1w`, `xs_mom_4w`, `xs_mom_12w`, `ts_mom_4w`, `ts_mom_12w`, `xs_rev_1w`; controls `no_trade`, `random_ranks`, `passive_long_ew`. Only the six members enter the Benjamini-Hochberg family.
- **Commit trailer** on every commit:
  ```
  Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
  ```
- **Verification command** for the whole suite, with TEMP forced onto C: (E: breaks one storage test):
  ```powershell
  $env:TEMP='C:\Users\User\AppData\Local\Temp'; $env:TMP=$env:TEMP; uv run pytest -q
  ```

### Refinements to the spec's module table

The spec's section 10 lists seven modules. Implementation splits two of them for testability, which changes no behaviour and no interface named in the spec:

- `panel_capture.py` is split into `panel_dataset.py` (Parquet publication and verification) and `panel_capture.py` (listing, download, parsing, orchestration). The spec's capture job is unchanged.
- Reading is `panel_reader.py` rather than being implicit in `panel_universe.py`, so DuckDB access has its own tests.
- `panel_statistics.py` holds the deflated Sharpe ratio, which the spec assigns to `panel_decision.py`; it is a self-contained statistic with its own test file.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `configs/xs-momentum-panel-v1.json` | Frozen family declaration; its SHA-256 is `family_spec_hash` in every report |
| `src/trading_bot/panel_config.py` | Pydantic model of the declaration; load and hash |
| `src/trading_bot/panel_dataset.py` | Parquet publication and verification for daily candles and funding, with per-instrument quality |
| `src/trading_bot/panel_capture.py` | Bucket listing, zip download, CSV parsing, storage preflight, capture manifest |
| `src/trading_bot/panel_reader.py` | DuckDB reads: `PanelBar`, `FundingEvent`, `contract_id` assignment |
| `src/trading_bot/panel_universe.py` | Eligibility at a decision time, liquidity tiering, `UNIVERSE_TOO_SMALL` |
| `src/trading_bot/panel_signals.py` | The six members and three controls as weight vectors |
| `src/trading_bot/panel_accounting.py` | Holding returns, drift, turnover, funding, forced closes, episode net returns |
| `src/trading_bot/panel_samples.py` | Rebalance calendar, `SplitSample` build, walk-forward manifest publication |
| `src/trading_bot/panel_fold_run.py` | Per-fold report carrying every member's episode series |
| `src/trading_bot/panel_statistics.py` | Annualised Sharpe and deflated Sharpe ratio |
| `src/trading_bot/panel_decision.py` | Pooling, bootstrap, BH, concentration, dominance, decision report |
| `src/trading_bot/cli.py` | Modify: add `panel-capture`, `panel-manifest`, `panel-fold`, `panel-decision` |
| `tests/test_panel_*.py` | One test file per module |

Data flow: capture → dataset → reader → universe → signals → accounting → (samples → manifest) → fold run → decision.

---

## Task 1: Frozen family declaration

**Files:**
- Create: `configs/xs-momentum-panel-v1.json`
- Create: `src/trading_bot/panel_config.py`
- Test: `tests/test_panel_config.py`

**Interfaces:**
- Consumes: `trading_bot.canonical.content_sha256`.
- Produces: `PanelFamilySpec` (pydantic, frozen), `load_panel_family_spec(path: Path) -> tuple[PanelFamilySpec, str]` returning the spec and its `family_spec_hash`; `PanelMember`, `PanelCostTable`, `PanelFoldGeometry`; constants `MEMBER_NAMES: tuple[str, ...]`, `CONTROL_NAMES: tuple[str, ...]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_panel_config.py
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from trading_bot.panel_config import (
    CONTROL_NAMES,
    MEMBER_NAMES,
    load_panel_family_spec,
)

REPOSITORY_CONFIG = Path("configs/xs-momentum-panel-v1.json")


def test_repository_config_declares_the_frozen_family() -> None:
    spec, spec_hash = load_panel_family_spec(REPOSITORY_CONFIG)
    assert spec.family_name == "xs_momentum_panel_v1"
    assert tuple(member.name for member in spec.members) == MEMBER_NAMES
    assert tuple(control.name for control in spec.controls) == CONTROL_NAMES
    assert spec.statistics.block_length == 4
    assert spec.statistics.bootstrap_repetitions == 2000
    assert spec.statistics.pooled_episode_floor == 200
    assert len(spec_hash) == 64


def test_member_set_is_closed(tmp_path: Path) -> None:
    document = json.loads(REPOSITORY_CONFIG.read_text(encoding="utf-8"))
    document["members"].append(
        {"name": "xs_mom_26w", "kind": "cross_sectional", "lookback_days": 182}
    )
    path = tmp_path / "extra-member.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_panel_family_spec(path)


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    document = json.loads(REPOSITORY_CONFIG.read_text(encoding="utf-8"))
    document["tuning_knob"] = 3
    path = tmp_path / "extra-field.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_panel_family_spec(path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_panel_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading_bot.panel_config'`

- [ ] **Step 3: Write the configuration file**

```json
{
  "spec_version": "1.0.0",
  "family_name": "xs_momentum_panel_v1",
  "hypothesis": "Trailing-return momentum on a liquid panel of USDT perpetuals, held one week, has positive net expectancy under the base cost scenario and non-negative net expectancy under the adverse scenario, with the effect not concentrated in a single fold, contract, or episode.",
  "venue": "BINANCE_UM",
  "holding_days": 7,
  "members": [
    {"name": "xs_mom_1w", "kind": "cross_sectional", "lookback_days": 7, "reversed": false},
    {"name": "xs_mom_4w", "kind": "cross_sectional", "lookback_days": 28, "reversed": false},
    {"name": "xs_mom_12w", "kind": "cross_sectional", "lookback_days": 84, "reversed": false},
    {"name": "ts_mom_4w", "kind": "time_series", "lookback_days": 28, "reversed": false},
    {"name": "ts_mom_12w", "kind": "time_series", "lookback_days": 84, "reversed": false},
    {"name": "xs_rev_1w", "kind": "cross_sectional", "lookback_days": 7, "reversed": true}
  ],
  "controls": [
    {"name": "no_trade", "kind": "no_trade"},
    {"name": "random_ranks", "kind": "random_ranks"},
    {"name": "passive_long_ew", "kind": "passive_long"}
  ],
  "universe": {
    "minimum_history_days": 91,
    "liquidity_window_days": 30,
    "minimum_median_quote_volume": "5000000",
    "maximum_contracts": 100,
    "minimum_contracts": 40,
    "tier_one_rank_limit": 20
  },
  "weights": {
    "leg_gross": "0.5",
    "minimum_quintile_size": 8,
    "volatility_window_days": 30,
    "volatility_floor": "0.20",
    "time_series_cap_numerator": "2"
  },
  "costs": {
    "base": {
      "name": "base",
      "fee_bps_per_side": "5",
      "slippage_bps_per_side_tier_one": "5",
      "slippage_bps_per_side_tier_two": "10",
      "funding_receipt_multiplier": "1",
      "funding_payment_multiplier": "1",
      "forced_close_multiplier": "1"
    },
    "adverse": {
      "name": "adverse",
      "fee_bps_per_side": "5",
      "slippage_bps_per_side_tier_one": "10",
      "slippage_bps_per_side_tier_two": "20",
      "funding_receipt_multiplier": "0",
      "funding_payment_multiplier": "2",
      "forced_close_multiplier": "2"
    }
  },
  "folds": {
    "train_duration_ns": 31536000000000000,
    "validation_duration_ns": 7862400000000000,
    "test_duration_ns": 15724800000000000,
    "step_ns": 15724800000000000,
    "embargo_ns": 1209600000000000,
    "holdout_duration_ns": 15724800000000000
  },
  "statistics": {
    "block_length": 4,
    "bootstrap_repetitions": 2000,
    "random_seed": 17,
    "confidence": "0.95",
    "false_discovery_gate": "0.10",
    "pooled_episode_floor": 200,
    "concentration_limit": "0.5",
    "positive_fold_numerator": 2,
    "positive_fold_denominator": 3
  }
}
```

- [ ] **Step 4: Write the module**

```python
# src/trading_bot/panel_config.py
"""Frozen declaration of the cross-sectional momentum experiment family."""

import json
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from trading_bot.canonical import content_sha256

MEMBER_NAMES: tuple[str, ...] = (
    "xs_mom_1w",
    "xs_mom_4w",
    "xs_mom_12w",
    "ts_mom_4w",
    "ts_mom_12w",
    "xs_rev_1w",
)
CONTROL_NAMES: tuple[str, ...] = ("no_trade", "random_ranks", "passive_long_ew")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PanelMember(_Frozen):
    name: Literal["xs_mom_1w", "xs_mom_4w", "xs_mom_12w", "ts_mom_4w", "ts_mom_12w", "xs_rev_1w"]
    kind: Literal["cross_sectional", "time_series"]
    lookback_days: int
    reversed: bool

    @field_validator("lookback_days")
    @classmethod
    def validate_lookback(cls, value: int) -> int:
        if value < 1:
            raise ValueError("lookback_days must be positive")
        return value


class PanelControl(_Frozen):
    name: Literal["no_trade", "random_ranks", "passive_long_ew"]
    kind: Literal["no_trade", "random_ranks", "passive_long"]


class PanelUniverseRules(_Frozen):
    minimum_history_days: int
    liquidity_window_days: int
    minimum_median_quote_volume: Decimal
    maximum_contracts: int
    minimum_contracts: int
    tier_one_rank_limit: int


class PanelWeightRules(_Frozen):
    leg_gross: Decimal
    minimum_quintile_size: int
    volatility_window_days: int
    volatility_floor: Decimal
    time_series_cap_numerator: Decimal


class PanelCostTable(_Frozen):
    name: Literal["base", "adverse"]
    fee_bps_per_side: Decimal
    slippage_bps_per_side_tier_one: Decimal
    slippage_bps_per_side_tier_two: Decimal
    funding_receipt_multiplier: Decimal
    funding_payment_multiplier: Decimal
    forced_close_multiplier: Decimal


class PanelCosts(_Frozen):
    base: PanelCostTable
    adverse: PanelCostTable


class PanelFoldGeometry(_Frozen):
    train_duration_ns: int
    validation_duration_ns: int
    test_duration_ns: int
    step_ns: int
    embargo_ns: int
    holdout_duration_ns: int


class PanelStatistics(_Frozen):
    block_length: int
    bootstrap_repetitions: int
    random_seed: int
    confidence: Decimal
    false_discovery_gate: Decimal
    pooled_episode_floor: int
    concentration_limit: Decimal
    positive_fold_numerator: int
    positive_fold_denominator: int


class PanelFamilySpec(_Frozen):
    spec_version: Literal["1.0.0"]
    family_name: Literal["xs_momentum_panel_v1"]
    hypothesis: str
    venue: Literal["BINANCE_UM"]
    holding_days: int
    members: tuple[PanelMember, ...]
    controls: tuple[PanelControl, ...]
    universe: PanelUniverseRules
    weights: PanelWeightRules
    costs: PanelCosts
    folds: PanelFoldGeometry
    statistics: PanelStatistics

    @field_validator("members")
    @classmethod
    def validate_members(cls, value: tuple[PanelMember, ...]) -> tuple[PanelMember, ...]:
        if tuple(item.name for item in value) != MEMBER_NAMES:
            raise ValueError("the member set is frozen and ordered")
        return value

    @field_validator("controls")
    @classmethod
    def validate_controls(cls, value: tuple[PanelControl, ...]) -> tuple[PanelControl, ...]:
        if tuple(item.name for item in value) != CONTROL_NAMES:
            raise ValueError("the control set is frozen and ordered")
        return value


def load_panel_family_spec(path: Path) -> tuple[PanelFamilySpec, str]:
    """Load the frozen declaration and return it with its canonical hash."""
    document = json.loads(path.read_text(encoding="utf-8"))
    spec = PanelFamilySpec.model_validate(document)
    return spec, content_sha256(document)
```

- [ ] **Step 5: Run tests, lint and types**

Run:
```powershell
$env:TEMP='C:\Users\User\AppData\Local\Temp'; $env:TMP=$env:TEMP
uv run pytest tests/test_panel_config.py -v
uv run ruff check src/trading_bot/panel_config.py tests/test_panel_config.py
uv run mypy
```
Expected: 3 passed, no lint findings, no type errors.

- [ ] **Step 6: Commit**

```bash
git add configs/xs-momentum-panel-v1.json src/trading_bot/panel_config.py tests/test_panel_config.py
git commit -m "feat: freeze the xs momentum family declaration

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Panel dataset publication

**Files:**
- Create: `src/trading_bot/panel_dataset.py`
- Test: `tests/test_panel_dataset.py`

**Interfaces:**
- Consumes: `trading_bot.canonical.canonical_json`, `content_sha256`.
- Produces: `PanelCandleRow`, `PanelFundingRow`, `InstrumentQuality`, `PanelDatasetArtifact`, `publish_panel_dataset(candles, funding, *, output_directory, raw_source_hashes) -> PanelDatasetArtifact`, `verify_panel_dataset(dataset_root) -> tuple[bool, tuple[str, ...]]`, `PanelDatasetError`.

Layout produced under `output_directory`:
```
dataset-manifest.json
quality-report.json
dataset=daily_candles/venue=BINANCE_UM/instrument=<SYMBOL>/part-00000.parquet
dataset=funding/venue=BINANCE_UM/instrument=<SYMBOL>/part-00000.parquet
```

- [ ] **Step 1: Write the failing test**

```python
# tests/test_panel_dataset.py
import json
from decimal import Decimal
from pathlib import Path

import pytest

from trading_bot.panel_dataset import (
    PanelCandleRow,
    PanelDatasetError,
    PanelFundingRow,
    publish_panel_dataset,
    verify_panel_dataset,
)

DAY_NS = 86_400_000_000_000
SOURCE_HASHES = ("a" * 64,)


def candle(symbol: str, index: int, *, close: str = "100") -> PanelCandleRow:
    open_time_ns = index * DAY_NS
    return PanelCandleRow(
        venue="BINANCE_UM",
        instrument_id=symbol,
        open_time_ns=open_time_ns,
        close_time_ns=open_time_ns + DAY_NS - 1_000_000,
        available_time_ns=open_time_ns + DAY_NS,
        interval_ns=DAY_NS,
        open=Decimal("100"),
        high=Decimal("101"),
        low=Decimal("99"),
        close=Decimal(close),
        base_volume=Decimal("10"),
        quote_volume=Decimal("1000"),
        trade_count=5,
        source_payload_hash="b" * 64,
    )


def funding(symbol: str, index: int) -> PanelFundingRow:
    return PanelFundingRow(
        venue="BINANCE_UM",
        instrument_id=symbol,
        calc_time_ns=index * DAY_NS,
        funding_interval_hours=8,
        rate=Decimal("0.0001"),
    )


def test_publish_writes_partitions_and_per_instrument_quality(tmp_path: Path) -> None:
    rows = tuple(candle("BTCUSDT", index) for index in range(3))
    rows += tuple(candle("ETHUSDT", index) for index in range(2))
    artifact = publish_panel_dataset(
        rows,
        (funding("BTCUSDT", 0),),
        output_directory=tmp_path / "dataset",
        raw_source_hashes=SOURCE_HASHES,
    )
    assert (
        artifact.dataset_root
        / "dataset=daily_candles"
        / "venue=BINANCE_UM"
        / "instrument=BTCUSDT"
        / "part-00000.parquet"
    ).is_file()
    quality = json.loads(artifact.quality_report_path.read_text(encoding="utf-8"))
    assert quality["candle_row_count"] == 5
    btc = next(item for item in quality["instruments"] if item["instrument_id"] == "BTCUSDT")
    assert btc["row_count"] == 3
    assert btc["missing_days"] == 0
    assert verify_panel_dataset(artifact.dataset_root) == (True, ())


def test_missing_day_is_counted(tmp_path: Path) -> None:
    rows = (candle("BTCUSDT", 0), candle("BTCUSDT", 1), candle("BTCUSDT", 3))
    artifact = publish_panel_dataset(
        rows, (), output_directory=tmp_path / "dataset", raw_source_hashes=SOURCE_HASHES
    )
    quality = json.loads(artifact.quality_report_path.read_text(encoding="utf-8"))
    btc = quality["instruments"][0]
    assert btc["missing_days"] == 1


def test_duplicate_rows_are_dropped(tmp_path: Path) -> None:
    rows = (candle("BTCUSDT", 0), candle("BTCUSDT", 0), candle("BTCUSDT", 1))
    artifact = publish_panel_dataset(
        rows, (), output_directory=tmp_path / "dataset", raw_source_hashes=SOURCE_HASHES
    )
    quality = json.loads(artifact.quality_report_path.read_text(encoding="utf-8"))
    assert quality["duplicate_rows"] == 1
    assert quality["candle_row_count"] == 2


def test_output_is_immutable(tmp_path: Path) -> None:
    target = tmp_path / "dataset"
    publish_panel_dataset(
        (candle("BTCUSDT", 0),), (), output_directory=target, raw_source_hashes=SOURCE_HASHES
    )
    with pytest.raises(PanelDatasetError):
        publish_panel_dataset(
            (candle("BTCUSDT", 0),), (), output_directory=target, raw_source_hashes=SOURCE_HASHES
        )


def test_tampering_is_detected(tmp_path: Path) -> None:
    artifact = publish_panel_dataset(
        (candle("BTCUSDT", 0),), (), output_directory=tmp_path / "d", raw_source_hashes=SOURCE_HASHES
    )
    path = (
        artifact.dataset_root
        / "dataset=daily_candles"
        / "venue=BINANCE_UM"
        / "instrument=BTCUSDT"
        / "part-00000.parquet"
    )
    path.write_bytes(path.read_bytes() + b"x")
    valid, errors = verify_panel_dataset(artifact.dataset_root)
    assert not valid
    assert any(error.startswith("PARQUET_HASH_MISMATCH") for error in errors)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_panel_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading_bot.panel_dataset'`

- [ ] **Step 3: Write the module**

```python
# src/trading_bot/panel_dataset.py
"""Immutable Parquet publication for the daily perpetual panel."""

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from trading_bot.canonical import canonical_json, content_sha256

DAY_NS = 86_400_000_000_000


class PanelDatasetError(RuntimeError):
    """Raised when a panel dataset cannot be published or verified."""


@dataclass(frozen=True, slots=True)
class PanelCandleRow:
    venue: str
    instrument_id: str
    open_time_ns: int
    close_time_ns: int
    available_time_ns: int
    interval_ns: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    base_volume: Decimal
    quote_volume: Decimal
    trade_count: int | None
    source_payload_hash: str


@dataclass(frozen=True, slots=True)
class PanelFundingRow:
    venue: str
    instrument_id: str
    calc_time_ns: int
    funding_interval_hours: int
    rate: Decimal


@dataclass(frozen=True, slots=True)
class InstrumentQuality:
    instrument_id: str
    row_count: int
    first_open_time_ns: int
    last_open_time_ns: int
    missing_days: int


@dataclass(frozen=True, slots=True)
class PanelDatasetArtifact:
    dataset_root: Path
    manifest_path: Path
    quality_report_path: Path
    root_hash: str
    instruments: tuple[InstrumentQuality, ...]


_CANDLE_SCHEMA = pa.schema(
    [
        pa.field("venue", pa.string(), nullable=False),
        pa.field("instrument_id", pa.string(), nullable=False),
        pa.field("open_time_ns", pa.int64(), nullable=False),
        pa.field("close_time_ns", pa.int64(), nullable=False),
        pa.field("available_time_ns", pa.int64(), nullable=False),
        pa.field("interval_ns", pa.int64(), nullable=False),
        pa.field("open", pa.string(), nullable=False),
        pa.field("high", pa.string(), nullable=False),
        pa.field("low", pa.string(), nullable=False),
        pa.field("close", pa.string(), nullable=False),
        pa.field("base_volume", pa.string(), nullable=False),
        pa.field("quote_volume", pa.string(), nullable=False),
        pa.field("trade_count", pa.int64(), nullable=True),
        pa.field("source_payload_hash", pa.string(), nullable=False),
    ]
)

_FUNDING_SCHEMA = pa.schema(
    [
        pa.field("venue", pa.string(), nullable=False),
        pa.field("instrument_id", pa.string(), nullable=False),
        pa.field("calc_time_ns", pa.int64(), nullable=False),
        pa.field("funding_interval_hours", pa.int64(), nullable=False),
        pa.field("rate", pa.string(), nullable=False),
    ]
)


def publish_panel_dataset(
    candles: tuple[PanelCandleRow, ...],
    funding: tuple[PanelFundingRow, ...],
    *,
    output_directory: Path,
    raw_source_hashes: tuple[str, ...],
) -> PanelDatasetArtifact:
    if output_directory.exists():
        raise PanelDatasetError("panel dataset already exists and is immutable")
    if not raw_source_hashes or any(len(value) != 64 for value in raw_source_hashes):
        raise PanelDatasetError("raw source hashes are required")
    admitted, duplicate_rows = _admit_candles(candles)
    if not admitted:
        raise PanelDatasetError("panel dataset has no candle rows")
    admitted_funding = _admit_funding(funding)
    instruments = _instrument_quality(admitted)

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_directory.name}.", dir=output_directory.parent)
    )
    try:
        files = _write_candle_partitions(temporary, admitted)
        files += _write_funding_partitions(temporary, admitted_funding)
        quality_record: dict[str, object] = {
            "candle_row_count": len(admitted),
            "funding_row_count": len(admitted_funding),
            "duplicate_rows": duplicate_rows,
            "instrument_count": len(instruments),
            "instruments": [
                {
                    "instrument_id": item.instrument_id,
                    "row_count": item.row_count,
                    "first_open_time_ns": item.first_open_time_ns,
                    "last_open_time_ns": item.last_open_time_ns,
                    "missing_days": item.missing_days,
                }
                for item in instruments
            ],
        }
        quality_bytes = canonical_json(quality_record)
        (temporary / "quality-report.json").write_bytes(quality_bytes)
        material: dict[str, object] = {
            "dataset_name": "usdt_perp_panel",
            "schema_version": "1.0.0",
            "candle_schema_hash": content_sha256(str(_CANDLE_SCHEMA)),
            "funding_schema_hash": content_sha256(str(_FUNDING_SCHEMA)),
            "decimal_encoding": "canonical_string",
            "compression": "zstd",
            "candle_row_count": len(admitted),
            "funding_row_count": len(admitted_funding),
            "files": files,
            "quality_report_hash": hashlib.sha256(quality_bytes).hexdigest(),
            "raw_source_hashes": list(raw_source_hashes),
        }
        root_hash = content_sha256(material)
        manifest = dict(material)
        manifest["root_hash"] = root_hash
        (temporary / "dataset-manifest.json").write_bytes(canonical_json(manifest))
        temporary.replace(output_directory)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return PanelDatasetArtifact(
        dataset_root=output_directory,
        manifest_path=output_directory / "dataset-manifest.json",
        quality_report_path=output_directory / "quality-report.json",
        root_hash=root_hash,
        instruments=instruments,
    )


def verify_panel_dataset(dataset_root: Path) -> tuple[bool, tuple[str, ...]]:
    errors: list[str] = []
    try:
        manifest = json.loads((dataset_root / "dataset-manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, ("MANIFEST_UNREADABLE",)
    if not isinstance(manifest, dict):
        return False, ("MANIFEST_STRUCTURE_INVALID",)
    recorded_root = manifest.get("root_hash")
    material = {key: value for key, value in manifest.items() if key != "root_hash"}
    if content_sha256(material) != recorded_root:
        errors.append("DATASET_ROOT_HASH_MISMATCH")
    try:
        quality_bytes = (dataset_root / "quality-report.json").read_bytes()
    except OSError:
        return False, tuple([*errors, "QUALITY_REPORT_UNREADABLE"])
    if hashlib.sha256(quality_bytes).hexdigest() != manifest.get("quality_report_hash"):
        errors.append("QUALITY_REPORT_HASH_MISMATCH")
    files = manifest.get("files")
    if not isinstance(files, list):
        return False, tuple([*errors, "MANIFEST_STRUCTURE_INVALID"])
    for entry in files:
        if not isinstance(entry, dict):
            errors.append("MANIFEST_STRUCTURE_INVALID")
            continue
        relative = str(entry.get("relative_path"))
        path = dataset_root / relative
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            errors.append(f"PARQUET_MISSING:{relative}")
            continue
        if digest != entry.get("sha256"):
            errors.append(f"PARQUET_HASH_MISMATCH:{relative}")
    return not errors, tuple(errors)


def _admit_candles(
    candles: tuple[PanelCandleRow, ...],
) -> tuple[tuple[PanelCandleRow, ...], int]:
    seen: dict[tuple[str, str, int], PanelCandleRow] = {}
    duplicates = 0
    for row in candles:
        if row.interval_ns != DAY_NS:
            raise PanelDatasetError("panel candles must be daily")
        key = (row.venue, row.instrument_id, row.open_time_ns)
        if key in seen:
            duplicates += 1
            continue
        seen[key] = row
    ordered = sorted(seen.values(), key=lambda item: (item.instrument_id, item.open_time_ns))
    return tuple(ordered), duplicates


def _admit_funding(funding: tuple[PanelFundingRow, ...]) -> tuple[PanelFundingRow, ...]:
    seen: dict[tuple[str, str, int], PanelFundingRow] = {}
    for row in funding:
        seen[(row.venue, row.instrument_id, row.calc_time_ns)] = row
    ordered = sorted(seen.values(), key=lambda item: (item.instrument_id, item.calc_time_ns))
    return tuple(ordered)


def _instrument_quality(rows: tuple[PanelCandleRow, ...]) -> tuple[InstrumentQuality, ...]:
    grouped: dict[str, list[PanelCandleRow]] = {}
    for row in rows:
        grouped.setdefault(row.instrument_id, []).append(row)
    quality: list[InstrumentQuality] = []
    for instrument_id in sorted(grouped):
        series = sorted(grouped[instrument_id], key=lambda item: item.open_time_ns)
        first = series[0].open_time_ns
        last = series[-1].open_time_ns
        expected = (last - first) // DAY_NS + 1
        quality.append(
            InstrumentQuality(
                instrument_id=instrument_id,
                row_count=len(series),
                first_open_time_ns=first,
                last_open_time_ns=last,
                missing_days=expected - len(series),
            )
        )
    return tuple(quality)


def _write_candle_partitions(
    root: Path, rows: tuple[PanelCandleRow, ...]
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[PanelCandleRow]] = {}
    for row in rows:
        grouped.setdefault((row.venue, row.instrument_id), []).append(row)
    files: list[dict[str, object]] = []
    for venue, instrument_id in sorted(grouped):
        series = grouped[(venue, instrument_id)]
        table = pa.Table.from_pydict(
            {
                "venue": [item.venue for item in series],
                "instrument_id": [item.instrument_id for item in series],
                "open_time_ns": [item.open_time_ns for item in series],
                "close_time_ns": [item.close_time_ns for item in series],
                "available_time_ns": [item.available_time_ns for item in series],
                "interval_ns": [item.interval_ns for item in series],
                "open": [str(item.open) for item in series],
                "high": [str(item.high) for item in series],
                "low": [str(item.low) for item in series],
                "close": [str(item.close) for item in series],
                "base_volume": [str(item.base_volume) for item in series],
                "quote_volume": [str(item.quote_volume) for item in series],
                "trade_count": [item.trade_count for item in series],
                "source_payload_hash": [item.source_payload_hash for item in series],
            },
            schema=_CANDLE_SCHEMA,
        )
        relative = (
            f"dataset=daily_candles/venue={venue}/instrument={instrument_id}/part-00000.parquet"
        )
        files.append(_write_table(root, relative, table, len(series)))
    return files


def _write_funding_partitions(
    root: Path, rows: tuple[PanelFundingRow, ...]
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[PanelFundingRow]] = {}
    for row in rows:
        grouped.setdefault((row.venue, row.instrument_id), []).append(row)
    files: list[dict[str, object]] = []
    for venue, instrument_id in sorted(grouped):
        series = grouped[(venue, instrument_id)]
        table = pa.Table.from_pydict(
            {
                "venue": [item.venue for item in series],
                "instrument_id": [item.instrument_id for item in series],
                "calc_time_ns": [item.calc_time_ns for item in series],
                "funding_interval_hours": [item.funding_interval_hours for item in series],
                "rate": [str(item.rate) for item in series],
            },
            schema=_FUNDING_SCHEMA,
        )
        relative = f"dataset=funding/venue={venue}/instrument={instrument_id}/part-00000.parquet"
        files.append(_write_table(root, relative, table, len(series)))
    return files


def _write_table(
    root: Path, relative: str, table: pa.Table, row_count: int
) -> dict[str, object]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"relative_path": relative, "sha256": digest, "row_count": row_count}
```

- [ ] **Step 4: Run tests, lint and types**

Run:
```powershell
$env:TEMP='C:\Users\User\AppData\Local\Temp'; $env:TMP=$env:TEMP
uv run pytest tests/test_panel_dataset.py -v
uv run ruff check src/trading_bot/panel_dataset.py tests/test_panel_dataset.py
uv run mypy
```
Expected: 5 passed, clean.

- [ ] **Step 5: Commit**

```bash
git add src/trading_bot/panel_dataset.py tests/test_panel_dataset.py
git commit -m "feat: publish immutable panel datasets with per-instrument quality

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Panel capture from the public dumps

**Files:**
- Create: `src/trading_bot/panel_capture.py`
- Test: `tests/test_panel_capture.py`
- Modify: `src/trading_bot/cli.py` (add the `panel-capture` subcommand)

**Interfaces:**
- Consumes: `panel_dataset.publish_panel_dataset`, `PanelCandleRow`, `PanelFundingRow`; `storage.StoragePolicy`.
- Produces: `PanelCaptureError`; `PanelFetch = Callable[[str], PanelPayload]`; `PanelPayload(url, raw_bytes, received_time_ns)`; `build_kline_zip_url(symbol, month) -> str`; `build_funding_zip_url(symbol, month) -> str`; `parse_kline_zip(payload, *, symbol) -> tuple[PanelCandleRow, ...]`; `parse_funding_zip(payload, *, symbol) -> tuple[PanelFundingRow, ...]`; `capture_panel(*, workspace_root, output_directory, reserve_bytes, symbols, months, fetch) -> PanelCaptureArtifact`; `verify_panel_capture(capture_root) -> tuple[bool, tuple[str, ...]]`.

Capture layout mirrors the v2 candle capture: `capture-manifest.json`, `raw/<symbol>/<file>.zip`, `dataset/`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_panel_capture.py
import io
import zipfile
from pathlib import Path

import pytest

from trading_bot.panel_capture import (
    PanelCaptureError,
    PanelPayload,
    build_funding_zip_url,
    build_kline_zip_url,
    capture_panel,
    parse_funding_zip,
    parse_kline_zip,
    verify_panel_capture,
)

KLINE_HEADER = (
    "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
    "taker_buy_volume,taker_buy_quote_volume,ignore"
)
DAY_MS = 86_400_000


def zip_bytes(name: str, text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, text)
    return buffer.getvalue()


def kline_csv(days: int, *, with_header: bool = True) -> str:
    lines = [KLINE_HEADER] if with_header else []
    for index in range(days):
        open_ms = index * DAY_MS
        lines.append(
            f"{open_ms},100,101,99,{100 + index},10,{open_ms + DAY_MS - 1},1000,5,6,600,0"
        )
    return "\n".join(lines) + "\n"


def test_urls_are_pinned_to_the_public_bucket() -> None:
    assert build_kline_zip_url("BTCUSDT", "2024-01") == (
        "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1d/"
        "BTCUSDT-1d-2024-01.zip"
    )
    assert build_funding_zip_url("BTCUSDT", "2024-01") == (
        "https://data.binance.vision/data/futures/um/monthly/fundingRate/BTCUSDT/"
        "BTCUSDT-fundingRate-2024-01.zip"
    )


def test_parse_kline_zip_reads_rows_with_and_without_header() -> None:
    payload = PanelPayload(url="u", raw_bytes=zip_bytes("a.csv", kline_csv(2)), received_time_ns=1)
    rows = parse_kline_zip(payload, symbol="BTCUSDT")
    assert len(rows) == 2
    assert rows[0].instrument_id == "BTCUSDT"
    assert str(rows[1].close) == "101"
    assert rows[0].available_time_ns == rows[0].close_time_ns + 1
    bare = PanelPayload(
        url="u", raw_bytes=zip_bytes("a.csv", kline_csv(1, with_header=False)), received_time_ns=1
    )
    assert len(parse_kline_zip(bare, symbol="BTCUSDT")) == 1


def test_parse_funding_zip() -> None:
    text = "calc_time,funding_interval_hours,last_funding_rate\n0,8,-0.00006120\n"
    payload = PanelPayload(url="u", raw_bytes=zip_bytes("f.csv", text), received_time_ns=1)
    rows = parse_funding_zip(payload, symbol="BTCUSDT")
    assert len(rows) == 1
    assert str(rows[0].rate) == "-0.00006120"
    assert rows[0].funding_interval_hours == 8


def test_capture_publishes_and_verifies(tmp_path: Path) -> None:
    def fetch(url: str) -> PanelPayload:
        if "fundingRate" in url:
            text = "calc_time,funding_interval_hours,last_funding_rate\n0,8,0.0001\n"
            return PanelPayload(url=url, raw_bytes=zip_bytes("f.csv", text), received_time_ns=1)
        return PanelPayload(url=url, raw_bytes=zip_bytes("k.csv", kline_csv(3)), received_time_ns=1)

    artifact = capture_panel(
        workspace_root=tmp_path,
        output_directory=tmp_path / "capture",
        reserve_bytes=0,
        symbols=("BTCUSDT", "ETHUSDT"),
        months=("2024-01",),
        fetch=fetch,
    )
    assert artifact.capture_root_hash
    assert (artifact.capture_root / "raw" / "BTCUSDT").is_dir()
    assert verify_panel_capture(artifact.capture_root) == (True, ())


def test_capture_rejects_a_foreign_host(tmp_path: Path) -> None:
    def fetch(url: str) -> PanelPayload:
        return PanelPayload(url="https://evil.example/x.zip", raw_bytes=b"", received_time_ns=1)

    with pytest.raises(PanelCaptureError):
        capture_panel(
            workspace_root=tmp_path,
            output_directory=tmp_path / "capture",
            reserve_bytes=0,
            symbols=("BTCUSDT",),
            months=("2024-01",),
            fetch=fetch,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_panel_capture.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading_bot.panel_capture'`

- [ ] **Step 3: Write the module**

```python
# src/trading_bot/panel_capture.py
"""Bounded capture of Binance USD-M daily klines and funding from public dumps."""

import csv
import hashlib
import io
import json
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.panel_dataset import (
    DAY_NS,
    PanelCandleRow,
    PanelFundingRow,
    publish_panel_dataset,
    verify_panel_dataset,
)
from trading_bot.storage import StoragePolicy

_ALLOWED_PANEL_HOSTS = frozenset(
    {"data.binance.vision", "s3-ap-northeast-1.amazonaws.com", "fapi.binance.com"}
)
_VENUE = "BINANCE_UM"
_MAX_ZIP_BYTES = 32_000_000
_MONTH_PATTERN = re.compile(r"^\d{4}-\d{2}$")
_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9_]{2,32}$")


class PanelCaptureError(RuntimeError):
    """Raised when panel capture fails closed."""


@dataclass(frozen=True, slots=True)
class PanelPayload:
    url: str
    raw_bytes: bytes
    received_time_ns: int


@dataclass(frozen=True, slots=True)
class PanelCaptureArtifact:
    capture_root: Path
    capture_manifest_path: Path
    dataset_root: Path
    capture_root_hash: str
    dataset_root_hash: str


PanelFetch = Callable[[str], PanelPayload]


@dataclass(frozen=True, slots=True)
class PanelZipClient:
    timeout_seconds: int = 60

    def fetch(self, url: str) -> PanelPayload:
        _validate_panel_url(url)
        request = urllib.request.Request(
            url, headers={"User-Agent": "hybrid-trading-research/0.1"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                _validate_panel_url(response.geturl())
                raw = response.read(_MAX_ZIP_BYTES + 1)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            raise PanelCaptureError(f"panel dump request failed: {error}") from error
        if len(raw) > _MAX_ZIP_BYTES:
            raise PanelCaptureError("panel dump exceeds the byte limit")
        return PanelPayload(url=url, raw_bytes=raw, received_time_ns=time.time_ns())


def build_kline_zip_url(symbol: str, month: str) -> str:
    _validate_symbol(symbol)
    _validate_month(month)
    return (
        "https://data.binance.vision/data/futures/um/monthly/klines/"
        f"{symbol}/1d/{symbol}-1d-{month}.zip"
    )


def build_funding_zip_url(symbol: str, month: str) -> str:
    _validate_symbol(symbol)
    _validate_month(month)
    return (
        "https://data.binance.vision/data/futures/um/monthly/fundingRate/"
        f"{symbol}/{symbol}-fundingRate-{month}.zip"
    )


def parse_kline_zip(payload: PanelPayload, *, symbol: str) -> tuple[PanelCandleRow, ...]:
    digest = hashlib.sha256(payload.raw_bytes).hexdigest()
    rows: list[PanelCandleRow] = []
    for record in _zip_rows(payload.raw_bytes):
        if record[0].startswith("open_time"):
            continue
        if len(record) < 9:
            raise PanelCaptureError("kline row has too few columns")
        open_time_ns = int(record[0]) * 1_000_000
        close_time_ns = int(record[6]) * 1_000_000
        rows.append(
            PanelCandleRow(
                venue=_VENUE,
                instrument_id=symbol,
                open_time_ns=open_time_ns,
                close_time_ns=close_time_ns,
                available_time_ns=close_time_ns + 1,
                interval_ns=DAY_NS,
                open=Decimal(record[1]),
                high=Decimal(record[2]),
                low=Decimal(record[3]),
                close=Decimal(record[4]),
                base_volume=Decimal(record[5]),
                quote_volume=Decimal(record[7]),
                trade_count=int(record[8]),
                source_payload_hash=digest,
            )
        )
    return tuple(rows)


def parse_funding_zip(payload: PanelPayload, *, symbol: str) -> tuple[PanelFundingRow, ...]:
    rows: list[PanelFundingRow] = []
    for record in _zip_rows(payload.raw_bytes):
        if record[0].startswith("calc_time"):
            continue
        if len(record) < 3:
            raise PanelCaptureError("funding row has too few columns")
        rows.append(
            PanelFundingRow(
                venue=_VENUE,
                instrument_id=symbol,
                calc_time_ns=int(record[0]) * 1_000_000,
                funding_interval_hours=int(record[1]),
                rate=Decimal(record[2]),
            )
        )
    return tuple(rows)


def capture_panel(
    *,
    workspace_root: Path,
    output_directory: Path,
    reserve_bytes: int,
    symbols: tuple[str, ...],
    months: tuple[str, ...],
    fetch: PanelFetch | None = None,
) -> PanelCaptureArtifact:
    if not symbols or not months:
        raise PanelCaptureError("panel capture needs at least one symbol and one month")
    workspace = workspace_root.resolve()
    target = output_directory.resolve()
    StoragePolicy(workspace, reserve_bytes).authorize(
        target=target,
        temporary_directory=target.parent,
        free_bytes=shutil.disk_usage(workspace).free,
        worst_case_required_bytes=500_000_000,
    )
    if target.exists():
        raise PanelCaptureError("panel capture already exists and is immutable")
    download = fetch if fetch is not None else PanelZipClient().fetch

    raw_root = target / "raw"
    sources: list[dict[str, object]] = []
    candles: list[PanelCandleRow] = []
    funding: list[PanelFundingRow] = []
    for symbol in symbols:
        for month in months:
            for kind, url in (
                ("klines", build_kline_zip_url(symbol, month)),
                ("fundingRate", build_funding_zip_url(symbol, month)),
            ):
                payload = download(url)
                _validate_panel_url(payload.url)
                relative = f"raw/{symbol}/{kind}-{month}.zip"
                path = target / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload.raw_bytes)
                sources.append(
                    {
                        "symbol": symbol,
                        "month": month,
                        "kind": kind,
                        "url": payload.url,
                        "received_time_ns": payload.received_time_ns,
                        "raw_relative_path": relative,
                        "raw_sha256": hashlib.sha256(payload.raw_bytes).hexdigest(),
                    }
                )
                if kind == "klines":
                    candles.extend(parse_kline_zip(payload, symbol=symbol))
                else:
                    funding.extend(parse_funding_zip(payload, symbol=symbol))
    if not raw_root.is_dir():
        raise PanelCaptureError("panel capture produced no raw payloads")

    dataset = publish_panel_dataset(
        tuple(candles),
        tuple(funding),
        output_directory=target / "dataset",
        raw_source_hashes=tuple(str(item["raw_sha256"]) for item in sources),
    )
    material: dict[str, object] = {
        "capture_version": "1.0.0",
        "venue": _VENUE,
        "interval": "1d",
        "symbols": list(symbols),
        "months": list(months),
        "sources": sources,
        "dataset_root_hash": dataset.root_hash,
    }
    capture_root_hash = content_sha256(material)
    document = dict(material)
    document["capture_root_hash"] = capture_root_hash
    (target / "capture-manifest.json").write_bytes(canonical_json(document))
    return PanelCaptureArtifact(
        capture_root=target,
        capture_manifest_path=target / "capture-manifest.json",
        dataset_root=dataset.dataset_root,
        capture_root_hash=capture_root_hash,
        dataset_root_hash=dataset.root_hash,
    )


def verify_panel_capture(capture_root: Path) -> tuple[bool, tuple[str, ...]]:
    errors: list[str] = []
    try:
        manifest = json.loads((capture_root / "capture-manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, ("CAPTURE_MANIFEST_UNREADABLE",)
    if not isinstance(manifest, dict):
        return False, ("MANIFEST_STRUCTURE_INVALID",)
    recorded = manifest.get("capture_root_hash")
    material = {key: value for key, value in manifest.items() if key != "capture_root_hash"}
    if content_sha256(material) != recorded:
        errors.append("CAPTURE_ROOT_HASH_MISMATCH")
    sources = manifest.get("sources")
    if not isinstance(sources, list):
        return False, tuple([*errors, "MANIFEST_STRUCTURE_INVALID"])
    for entry in sources:
        if not isinstance(entry, dict):
            errors.append("MANIFEST_STRUCTURE_INVALID")
            continue
        relative = str(entry.get("raw_relative_path"))
        try:
            digest = hashlib.sha256((capture_root / relative).read_bytes()).hexdigest()
        except OSError:
            errors.append(f"RAW_MISSING:{relative}")
            continue
        if digest != entry.get("raw_sha256"):
            errors.append(f"RAW_HASH_MISMATCH:{relative}")
    dataset_valid, dataset_errors = verify_panel_dataset(capture_root / "dataset")
    if not dataset_valid:
        errors.extend(dataset_errors)
    try:
        dataset_manifest = json.loads(
            (capture_root / "dataset" / "dataset-manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False, tuple([*errors, "DATASET_MANIFEST_UNREADABLE"])
    if dataset_manifest.get("root_hash") != manifest.get("dataset_root_hash"):
        errors.append("CAPTURE_DATASET_LINK_MISMATCH")
    return not errors, tuple(errors)


def _zip_rows(raw: bytes) -> list[list[str]]:
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            names = archive.namelist()
            if len(names) != 1:
                raise PanelCaptureError("panel dump must contain exactly one member")
            text = archive.read(names[0]).decode("utf-8")
    except (zipfile.BadZipFile, UnicodeDecodeError) as error:
        raise PanelCaptureError(f"panel dump is unreadable: {error}") from error
    return [row for row in csv.reader(io.StringIO(text)) if row]


def _validate_panel_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in _ALLOWED_PANEL_HOSTS:
        raise PanelCaptureError(f"panel URL is not allowed: {url}")


def _validate_symbol(symbol: str) -> None:
    if not _SYMBOL_PATTERN.match(symbol):
        raise PanelCaptureError(f"invalid symbol: {symbol}")


def _validate_month(month: str) -> None:
    if not _MONTH_PATTERN.match(month):
        raise PanelCaptureError(f"invalid month: {month}")
```

- [ ] **Step 4: Add the CLI subcommand**

In `src/trading_bot/cli.py`, add the import and, after the `evaluate-fold` parser block (`cli.py:54-63`), insert:

```python
    panel_capture = commands.add_parser("panel-capture")
    panel_capture.add_argument("--workspace-root", type=Path, default=Path.cwd())
    panel_capture.add_argument("--output", type=Path, required=True)
    panel_capture.add_argument("--reserve-bytes", type=int, default=20_000_000_000)
    panel_capture.add_argument("--symbols", required=True, help="comma-separated symbol list")
    panel_capture.add_argument("--months", required=True, help="comma-separated YYYY-MM list")
```

and before `raise AssertionError("unreachable command")`:

```python
    if parsed.command == "panel-capture":
        workspace = parsed.workspace_root.resolve()
        output = parsed.output.resolve()
        if not output.is_relative_to(workspace):
            raise ValueError("panel capture paths must stay inside workspace")
        capture_panel(
            workspace_root=workspace,
            output_directory=output,
            reserve_bytes=parsed.reserve_bytes,
            symbols=tuple(item for item in parsed.symbols.split(",") if item),
            months=tuple(item for item in parsed.months.split(",") if item),
        )
        return 0
```

with `from trading_bot.panel_capture import capture_panel` added to the imports.

- [ ] **Step 5: Run tests, lint and types**

Run:
```powershell
$env:TEMP='C:\Users\User\AppData\Local\Temp'; $env:TMP=$env:TEMP
uv run pytest tests/test_panel_capture.py -v
uv run ruff check src/trading_bot/panel_capture.py src/trading_bot/cli.py tests/test_panel_capture.py
uv run mypy
```
Expected: 5 passed, clean.

- [ ] **Step 6: Commit**

```bash
git add src/trading_bot/panel_capture.py src/trading_bot/cli.py tests/test_panel_capture.py
git commit -m "feat: capture the usdt perpetual panel from public dumps

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Panel reader and contract identity

**Files:**
- Create: `src/trading_bot/panel_reader.py`
- Test: `tests/test_panel_reader.py`

**Interfaces:**
- Consumes: `panel_dataset` layout; `duckdb`.
- Produces: `PanelBar(contract_id, instrument_id, open_time_ns, close_time_ns, available_time_ns, close, quote_volume)`; `FundingEvent(contract_id, instrument_id, calc_time_ns, rate)`; `load_panel_bars(dataset_root, *, available_before_ns=None) -> tuple[PanelBar, ...]`; `load_funding_events(dataset_root) -> tuple[FundingEvent, ...]`; `PanelReaderError`. `contract_id` is `f"{instrument_id}:{first_open_time_ns}"` using the earliest bar of that instrument in the dataset.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_panel_reader.py
from decimal import Decimal
from pathlib import Path

import pytest

from trading_bot.panel_dataset import PanelCandleRow, PanelFundingRow, publish_panel_dataset
from trading_bot.panel_reader import PanelReaderError, load_funding_events, load_panel_bars

DAY_NS = 86_400_000_000_000


def row(symbol: str, index: int, *, volume: str = "1000") -> PanelCandleRow:
    open_time_ns = index * DAY_NS
    return PanelCandleRow(
        venue="BINANCE_UM",
        instrument_id=symbol,
        open_time_ns=open_time_ns,
        close_time_ns=open_time_ns + DAY_NS - 1_000_000,
        available_time_ns=open_time_ns + DAY_NS,
        interval_ns=DAY_NS,
        open=Decimal("100"),
        high=Decimal("101"),
        low=Decimal("99"),
        close=Decimal(100 + index),
        base_volume=Decimal("10"),
        quote_volume=Decimal(volume),
        trade_count=5,
        source_payload_hash="b" * 64,
    )


def dataset(tmp_path: Path) -> Path:
    artifact = publish_panel_dataset(
        tuple(row("BTCUSDT", index) for index in range(3))
        + tuple(row("ETHUSDT", index) for index in range(1, 3)),
        (
            PanelFundingRow(
                venue="BINANCE_UM",
                instrument_id="BTCUSDT",
                calc_time_ns=DAY_NS,
                funding_interval_hours=8,
                rate=Decimal("0.0001"),
            ),
        ),
        output_directory=tmp_path / "dataset",
        raw_source_hashes=("a" * 64,),
    )
    return artifact.dataset_root


def test_contract_id_is_bound_to_the_first_bar(tmp_path: Path) -> None:
    bars = load_panel_bars(dataset(tmp_path))
    btc = [bar for bar in bars if bar.instrument_id == "BTCUSDT"]
    eth = [bar for bar in bars if bar.instrument_id == "ETHUSDT"]
    assert {bar.contract_id for bar in btc} == {"BTCUSDT:0"}
    assert {bar.contract_id for bar in eth} == {f"ETHUSDT:{DAY_NS}"}
    assert [bar.open_time_ns for bar in btc] == [0, DAY_NS, 2 * DAY_NS]


def test_availability_boundary_is_applied(tmp_path: Path) -> None:
    bars = load_panel_bars(dataset(tmp_path), available_before_ns=2 * DAY_NS)
    assert max(bar.open_time_ns for bar in bars) == DAY_NS


def test_funding_events_carry_contract_ids(tmp_path: Path) -> None:
    events = load_funding_events(dataset(tmp_path))
    assert len(events) == 1
    assert events[0].contract_id == "BTCUSDT:0"
    assert events[0].rate == Decimal("0.0001")


def test_invalid_boundary_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(PanelReaderError):
        load_panel_bars(dataset(tmp_path), available_before_ns=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_panel_reader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading_bot.panel_reader'`

- [ ] **Step 3: Write the module**

```python
# src/trading_bot/panel_reader.py
"""DuckDB reads over the published perpetual panel."""

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import duckdb


class PanelReaderError(RuntimeError):
    """Raised when the panel dataset cannot be read safely."""


@dataclass(frozen=True, slots=True)
class PanelBar:
    contract_id: str
    instrument_id: str
    open_time_ns: int
    close_time_ns: int
    available_time_ns: int
    close: Decimal
    quote_volume: Decimal


@dataclass(frozen=True, slots=True)
class FundingEvent:
    contract_id: str
    instrument_id: str
    calc_time_ns: int
    rate: Decimal


def load_panel_bars(
    dataset_root: Path, *, available_before_ns: int | None = None
) -> tuple[PanelBar, ...]:
    if available_before_ns is not None and available_before_ns <= 0:
        raise PanelReaderError("panel availability boundary must be positive")
    glob = (dataset_root / "dataset=daily_candles" / "**" / "*.parquet").as_posix()
    query = """
        SELECT instrument_id, open_time_ns, close_time_ns, available_time_ns,
               close, quote_volume
        FROM read_parquet(?)
    """
    parameters: list[object] = [glob]
    if available_before_ns is not None:
        query += " WHERE available_time_ns < ?"
        parameters.append(available_before_ns)
    query += " ORDER BY instrument_id, open_time_ns"
    rows = duckdb.sql(query, params=parameters).fetchall()
    first_open = _first_open_times(dataset_root)
    bars: list[PanelBar] = []
    for record in rows:
        if len(record) != 6:
            raise PanelReaderError("unexpected panel candle schema")
        instrument_id = str(record[0])
        bars.append(
            PanelBar(
                contract_id=f"{instrument_id}:{first_open[instrument_id]}",
                instrument_id=instrument_id,
                open_time_ns=int(record[1]),
                close_time_ns=int(record[2]),
                available_time_ns=int(record[3]),
                close=Decimal(str(record[4])),
                quote_volume=Decimal(str(record[5])),
            )
        )
    return tuple(bars)


def load_funding_events(dataset_root: Path) -> tuple[FundingEvent, ...]:
    glob = (dataset_root / "dataset=funding" / "**" / "*.parquet").as_posix()
    try:
        rows = duckdb.sql(
            "SELECT instrument_id, calc_time_ns, rate FROM read_parquet(?) "
            "ORDER BY instrument_id, calc_time_ns",
            params=[glob],
        ).fetchall()
    except duckdb.IOException:
        return ()
    first_open = _first_open_times(dataset_root)
    events: list[FundingEvent] = []
    for record in rows:
        instrument_id = str(record[0])
        if instrument_id not in first_open:
            continue
        events.append(
            FundingEvent(
                contract_id=f"{instrument_id}:{first_open[instrument_id]}",
                instrument_id=instrument_id,
                calc_time_ns=int(record[1]),
                rate=Decimal(str(record[2])),
            )
        )
    return tuple(events)


def _first_open_times(dataset_root: Path) -> dict[str, int]:
    glob = (dataset_root / "dataset=daily_candles" / "**" / "*.parquet").as_posix()
    rows = duckdb.sql(
        "SELECT instrument_id, MIN(open_time_ns) FROM read_parquet(?) GROUP BY instrument_id",
        params=[glob],
    ).fetchall()
    return {str(record[0]): int(record[1]) for record in rows}
```

- [ ] **Step 4: Run tests, lint and types**

Run:
```powershell
$env:TEMP='C:\Users\User\AppData\Local\Temp'; $env:TMP=$env:TEMP
uv run pytest tests/test_panel_reader.py -v
uv run ruff check src/trading_bot/panel_reader.py tests/test_panel_reader.py
uv run mypy
```
Expected: 4 passed, clean.

- [ ] **Step 5: Commit**

```bash
git add src/trading_bot/panel_reader.py tests/test_panel_reader.py
git commit -m "feat: read the panel dataset with contract identity

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Universe eligibility and liquidity tiers

**Files:**
- Create: `src/trading_bot/panel_universe.py`
- Test: `tests/test_panel_universe.py`

**Interfaces:**
- Consumes: `panel_reader.PanelBar`; `panel_config.PanelUniverseRules`.
- Produces: `ContractHistory(contract_id, closes: dict[int, Decimal], quote_volumes: dict[int, Decimal])`; `build_contract_histories(bars) -> dict[str, ContractHistory]`; `EligibleContract(contract_id, median_quote_volume, liquidity_rank, tier)`; `UniverseSnapshot(decision_close_ns, contracts: tuple[EligibleContract, ...], reason_codes: tuple[str, ...])`; `select_universe(histories, *, decision_close_ns, rules) -> UniverseSnapshot`.

Rules, all evaluated with bars whose `close_time_ns <= decision_close_ns`: at least `minimum_history_days` bars; a bar exactly at `decision_close_ns`; trailing `liquidity_window_days` median `quote_volume` at least `minimum_median_quote_volume`; keep the top `maximum_contracts` by that median, ties broken by `contract_id`; `tier` is `1` for ranks `1..tier_one_rank_limit` and `2` beyond. Fewer than `minimum_contracts` survivors yields an empty snapshot carrying `("UNIVERSE_TOO_SMALL",)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_panel_universe.py
from decimal import Decimal

from trading_bot.panel_config import PanelUniverseRules
from trading_bot.panel_reader import PanelBar
from trading_bot.panel_universe import build_contract_histories, select_universe

DAY_NS = 86_400_000_000_000
RULES = PanelUniverseRules(
    minimum_history_days=5,
    liquidity_window_days=3,
    minimum_median_quote_volume=Decimal("100"),
    maximum_contracts=3,
    minimum_contracts=2,
    tier_one_rank_limit=1,
)


def bars(symbol: str, *, days: int, volume: str, first_day: int = 0) -> list[PanelBar]:
    output: list[PanelBar] = []
    for index in range(first_day, first_day + days):
        open_time_ns = index * DAY_NS
        close_time_ns = open_time_ns + DAY_NS - 1_000_000
        output.append(
            PanelBar(
                contract_id=f"{symbol}:{first_day * DAY_NS}",
                instrument_id=symbol,
                open_time_ns=open_time_ns,
                close_time_ns=close_time_ns,
                available_time_ns=close_time_ns + 1,
                close=Decimal(100 + index),
                quote_volume=Decimal(volume),
            )
        )
    return output


def test_ranking_capping_and_tiers() -> None:
    rows = (
        bars("AAAUSDT", days=8, volume="900")
        + bars("BBBUSDT", days=8, volume="800")
        + bars("CCCUSDT", days=8, volume="700")
        + bars("DDDUSDT", days=8, volume="600")
    )
    histories = build_contract_histories(tuple(rows))
    snapshot = select_universe(histories, decision_close_ns=7 * DAY_NS - 1_000_000, rules=RULES)
    assert [item.contract_id for item in snapshot.contracts] == [
        "AAAUSDT:0",
        "BBBUSDT:0",
        "CCCUSDT:0",
    ]
    assert snapshot.contracts[0].tier == 1
    assert snapshot.contracts[1].tier == 2
    assert snapshot.reason_codes == ()


def test_short_history_and_low_liquidity_are_excluded() -> None:
    rows = (
        bars("AAAUSDT", days=8, volume="900")
        + bars("BBBUSDT", days=8, volume="800")
        + bars("SHORTUSDT", days=3, volume="900", first_day=5)
        + bars("THINUSDT", days=8, volume="10")
    )
    histories = build_contract_histories(tuple(rows))
    snapshot = select_universe(histories, decision_close_ns=7 * DAY_NS - 1_000_000, rules=RULES)
    assert [item.contract_id for item in snapshot.contracts] == ["AAAUSDT:0", "BBBUSDT:0"]


def test_missing_decision_day_excludes_the_contract() -> None:
    rows = bars("AAAUSDT", days=8, volume="900") + bars("BBBUSDT", days=7, volume="800")
    histories = build_contract_histories(tuple(rows))
    snapshot = select_universe(histories, decision_close_ns=7 * DAY_NS - 1_000_000, rules=RULES)
    assert snapshot.contracts == ()
    assert snapshot.reason_codes == ("UNIVERSE_TOO_SMALL",)


def test_too_small_universe_is_reported() -> None:
    histories = build_contract_histories(tuple(bars("AAAUSDT", days=8, volume="900")))
    snapshot = select_universe(histories, decision_close_ns=7 * DAY_NS - 1_000_000, rules=RULES)
    assert snapshot.contracts == ()
    assert snapshot.reason_codes == ("UNIVERSE_TOO_SMALL",)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_panel_universe.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading_bot.panel_universe'`

- [ ] **Step 3: Write the module**

```python
# src/trading_bot/panel_universe.py
"""Point-in-time eligibility and liquidity tiering for the perpetual panel."""

from dataclasses import dataclass
from decimal import Decimal

from trading_bot.panel_config import PanelUniverseRules
from trading_bot.panel_reader import PanelBar

DAY_NS = 86_400_000_000_000


@dataclass(frozen=True, slots=True)
class ContractHistory:
    contract_id: str
    instrument_id: str
    closes: dict[int, Decimal]
    quote_volumes: dict[int, Decimal]
    close_times: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class EligibleContract:
    contract_id: str
    median_quote_volume: Decimal
    liquidity_rank: int
    tier: int


@dataclass(frozen=True, slots=True)
class UniverseSnapshot:
    decision_close_ns: int
    contracts: tuple[EligibleContract, ...]
    reason_codes: tuple[str, ...]


def build_contract_histories(bars: tuple[PanelBar, ...]) -> dict[str, ContractHistory]:
    """Index bars by contract and close time for point-in-time lookups."""
    closes: dict[str, dict[int, Decimal]] = {}
    volumes: dict[str, dict[int, Decimal]] = {}
    instruments: dict[str, str] = {}
    for bar in bars:
        closes.setdefault(bar.contract_id, {})[bar.close_time_ns] = bar.close
        volumes.setdefault(bar.contract_id, {})[bar.close_time_ns] = bar.quote_volume
        instruments[bar.contract_id] = bar.instrument_id
    return {
        contract_id: ContractHistory(
            contract_id=contract_id,
            instrument_id=instruments[contract_id],
            closes=closes[contract_id],
            quote_volumes=volumes[contract_id],
            close_times=tuple(sorted(closes[contract_id])),
        )
        for contract_id in sorted(closes)
    }


def select_universe(
    histories: dict[str, ContractHistory],
    *,
    decision_close_ns: int,
    rules: PanelUniverseRules,
) -> UniverseSnapshot:
    scored: list[tuple[Decimal, str]] = []
    for contract_id in sorted(histories):
        history = histories[contract_id]
        observed = [value for value in history.close_times if value <= decision_close_ns]
        if len(observed) < rules.minimum_history_days:
            continue
        if not observed or observed[-1] != decision_close_ns:
            continue
        window = observed[-rules.liquidity_window_days :]
        median = _median(tuple(history.quote_volumes[value] for value in window))
        if median < rules.minimum_median_quote_volume:
            continue
        scored.append((median, contract_id))

    scored.sort(key=lambda item: (-item[0], item[1]))
    if len(scored) < rules.minimum_contracts:
        return UniverseSnapshot(decision_close_ns, (), ("UNIVERSE_TOO_SMALL",))
    selected = scored[: rules.maximum_contracts]
    contracts = tuple(
        EligibleContract(
            contract_id=contract_id,
            median_quote_volume=median,
            liquidity_rank=index + 1,
            tier=1 if index + 1 <= rules.tier_one_rank_limit else 2,
        )
        for index, (median, contract_id) in enumerate(selected)
    )
    return UniverseSnapshot(decision_close_ns, contracts, ())


def _median(values: tuple[Decimal, ...]) -> Decimal:
    if not values:
        return Decimal(0)
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / Decimal(2)
```

- [ ] **Step 4: Run tests, lint and types**

Run:
```powershell
$env:TEMP='C:\Users\User\AppData\Local\Temp'; $env:TMP=$env:TEMP
uv run pytest tests/test_panel_universe.py -v
uv run ruff check src/trading_bot/panel_universe.py tests/test_panel_universe.py
uv run mypy
```
Expected: 4 passed, clean.

- [ ] **Step 5: Commit**

```bash
git add src/trading_bot/panel_universe.py tests/test_panel_universe.py
git commit -m "feat: select the point-in-time panel universe

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Member and control weight vectors

**Files:**
- Create: `src/trading_bot/panel_signals.py`
- Test: `tests/test_panel_signals.py`

**Interfaces:**
- Consumes: `panel_universe.ContractHistory`, `UniverseSnapshot`; `panel_config.PanelFamilySpec`, `PanelMember`, `PanelWeightRules`.
- Produces: `WeightVector(decision_close_ns, member, weights: tuple[tuple[str, Decimal], ...], reason_codes)`; `build_weight_vectors(histories, snapshot, *, spec) -> dict[str, WeightVector]` returning one entry per member and control name.

Construction, all from closes at or before `decision_close_ns`:
- Trailing return `close[t] / close[t - lookback_days] - 1`; a contract without the lagged close is dropped from that member's ranking.
- Cross-sectional: sort by `(trailing_return, contract_id)`; `quintile = max(minimum_quintile_size, len(ranked) // 5)`; bottom `quintile` get `-leg_gross / quintile`, top `quintile` get `+leg_gross / quintile`; `reversed` swaps the signs.
- Time-series: `raw = sign(trailing_return) / max(sigma, volatility_floor)` where `sigma` is the annualised sample standard deviation of the last `volatility_window_days` daily log returns; normalise to unit gross, then apply `cap = time_series_cap_numerator / len(ranked)` and renormalise, at most 10 rounds.
- `no_trade`: empty weights. `passive_long_ew`: `1 / n` on every eligible contract. `random_ranks`: replace the trailing return by `Random(random_seed ^ decision_close_ns).random()` per contract, then the `xs_mom` construction.
- An empty snapshot yields empty weights and the snapshot's reason codes for every name.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_panel_signals.py
from decimal import Decimal
from pathlib import Path

from trading_bot.panel_config import load_panel_family_spec
from trading_bot.panel_reader import PanelBar
from trading_bot.panel_signals import build_weight_vectors
from trading_bot.panel_universe import build_contract_histories, select_universe

DAY_NS = 86_400_000_000_000
SPEC, _ = load_panel_family_spec(Path("configs/xs-momentum-panel-v1.json"))


def series(symbol: str, closes: list[str]) -> list[PanelBar]:
    output: list[PanelBar] = []
    for index, close in enumerate(closes):
        open_time_ns = index * DAY_NS
        close_time_ns = open_time_ns + DAY_NS - 1_000_000
        output.append(
            PanelBar(
                contract_id=f"{symbol}:0",
                instrument_id=symbol,
                open_time_ns=open_time_ns,
                close_time_ns=close_time_ns,
                available_time_ns=close_time_ns + 1,
                close=Decimal(close),
                quote_volume=Decimal("10000000"),
            )
        )
    return output


def panel(count: int, days: int = 100) -> tuple[PanelBar, ...]:
    rows: list[PanelBar] = []
    for index in range(count):
        symbol = f"C{index:03d}USDT"
        drift = Decimal(index) / Decimal(1000)
        closes = [str(Decimal(100) * (Decimal(1) + drift) ** day) for day in range(days)]
        rows.extend(series(symbol, closes))
    return tuple(rows)


def test_cross_sectional_legs_are_balanced_and_sorted() -> None:
    bars = panel(50)
    histories = build_contract_histories(bars)
    decision = 99 * DAY_NS - 1_000_000
    snapshot = select_universe(histories, decision_close_ns=decision, rules=SPEC.universe)
    vectors = build_weight_vectors(histories, snapshot, spec=SPEC)
    momentum = vectors["xs_mom_4w"]
    positive = [item for item in momentum.weights if item[1] > 0]
    negative = [item for item in momentum.weights if item[1] < 0]
    assert len(positive) == len(negative) == 10
    assert sum((item[1] for item in positive), Decimal(0)) == Decimal("0.5")
    assert sum((item[1] for item in negative), Decimal(0)) == Decimal("-0.5")
    winners = {item[0] for item in positive}
    assert "C049USDT:0" in winners
    assert "C000USDT:0" not in winners


def test_reversal_is_the_mirror_of_momentum() -> None:
    bars = panel(50)
    histories = build_contract_histories(bars)
    decision = 99 * DAY_NS - 1_000_000
    snapshot = select_universe(histories, decision_close_ns=decision, rules=SPEC.universe)
    vectors = build_weight_vectors(histories, snapshot, spec=SPEC)
    momentum = dict(vectors["xs_mom_1w"].weights)
    reversal = dict(vectors["xs_rev_1w"].weights)
    assert set(momentum) == set(reversal)
    assert all(reversal[key] == -value for key, value in momentum.items())


def test_time_series_weights_are_unit_gross_and_capped() -> None:
    bars = panel(50)
    histories = build_contract_histories(bars)
    decision = 99 * DAY_NS - 1_000_000
    snapshot = select_universe(histories, decision_close_ns=decision, rules=SPEC.universe)
    vectors = build_weight_vectors(histories, snapshot, spec=SPEC)
    weights = vectors["ts_mom_12w"].weights
    gross = sum((abs(value) for _, value in weights), Decimal(0))
    assert abs(gross - Decimal(1)) < Decimal("0.0000000001")
    cap = Decimal(2) / Decimal(len(weights))
    assert all(abs(value) <= cap + Decimal("0.0000000001") for _, value in weights)


def test_controls_are_present_and_shaped() -> None:
    bars = panel(50)
    histories = build_contract_histories(bars)
    decision = 99 * DAY_NS - 1_000_000
    snapshot = select_universe(histories, decision_close_ns=decision, rules=SPEC.universe)
    vectors = build_weight_vectors(histories, snapshot, spec=SPEC)
    assert vectors["no_trade"].weights == ()
    passive = vectors["passive_long_ew"].weights
    assert len(passive) == len(snapshot.contracts)
    assert all(value > 0 for _, value in passive)
    random_ranks = vectors["random_ranks"].weights
    assert sum((abs(value) for _, value in random_ranks), Decimal(0)) == Decimal(1)


def test_empty_universe_propagates_reason_codes() -> None:
    bars = panel(3)
    histories = build_contract_histories(bars)
    decision = 99 * DAY_NS - 1_000_000
    snapshot = select_universe(histories, decision_close_ns=decision, rules=SPEC.universe)
    vectors = build_weight_vectors(histories, snapshot, spec=SPEC)
    assert vectors["xs_mom_1w"].weights == ()
    assert vectors["xs_mom_1w"].reason_codes == ("UNIVERSE_TOO_SMALL",)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_panel_signals.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading_bot.panel_signals'`

- [ ] **Step 3: Write the module**

```python
# src/trading_bot/panel_signals.py
"""Weight vectors for the frozen panel members and controls."""

from dataclasses import dataclass
from decimal import Decimal
from random import Random

from trading_bot.panel_config import PanelFamilySpec, PanelMember, PanelWeightRules
from trading_bot.panel_universe import ContractHistory, UniverseSnapshot

DAY_NS = 86_400_000_000_000
_TOLERANCE = Decimal("0.0000000001")
_MAX_CAP_ROUNDS = 10


@dataclass(frozen=True, slots=True)
class WeightVector:
    decision_close_ns: int
    member: str
    weights: tuple[tuple[str, Decimal], ...]
    reason_codes: tuple[str, ...]


def build_weight_vectors(
    histories: dict[str, ContractHistory],
    snapshot: UniverseSnapshot,
    *,
    spec: PanelFamilySpec,
) -> dict[str, WeightVector]:
    """Build every member and control weight vector for one rebalance."""
    decision = snapshot.decision_close_ns
    names = [member.name for member in spec.members] + [item.name for item in spec.controls]
    if not snapshot.contracts:
        return {
            name: WeightVector(decision, name, (), snapshot.reason_codes) for name in names
        }

    eligible = tuple(item.contract_id for item in snapshot.contracts)
    vectors: dict[str, WeightVector] = {}
    for member in spec.members:
        returns = _trailing_returns(histories, eligible, decision, member.lookback_days)
        if member.kind == "cross_sectional":
            weights = _cross_sectional_weights(returns, spec.weights, reverse=member.reversed)
        else:
            weights = _time_series_weights(
                returns, histories, eligible, decision, spec.weights
            )
        vectors[member.name] = WeightVector(decision, member.name, weights, ())

    vectors["no_trade"] = WeightVector(decision, "no_trade", (), ())
    share = Decimal(1) / Decimal(len(eligible))
    vectors["passive_long_ew"] = WeightVector(
        decision,
        "passive_long_ew",
        tuple((contract_id, share) for contract_id in eligible),
        (),
    )
    random = Random(spec.statistics.random_seed ^ decision)
    scores = {contract_id: Decimal(str(random.random())) for contract_id in eligible}
    vectors["random_ranks"] = WeightVector(
        decision,
        "random_ranks",
        _cross_sectional_weights(scores, spec.weights, reverse=False),
        (),
    )
    return vectors


def _trailing_returns(
    histories: dict[str, ContractHistory],
    eligible: tuple[str, ...],
    decision_close_ns: int,
    lookback_days: int,
) -> dict[str, Decimal]:
    lagged_close_ns = decision_close_ns - lookback_days * DAY_NS
    returns: dict[str, Decimal] = {}
    for contract_id in eligible:
        closes = histories[contract_id].closes
        current = closes.get(decision_close_ns)
        previous = closes.get(lagged_close_ns)
        if current is None or previous is None or previous <= 0:
            continue
        returns[contract_id] = current / previous - Decimal(1)
    return returns


def _cross_sectional_weights(
    returns: dict[str, Decimal], rules: PanelWeightRules, *, reverse: bool
) -> tuple[tuple[str, Decimal], ...]:
    ranked = sorted(returns.items(), key=lambda item: (item[1], item[0]))
    quintile = max(rules.minimum_quintile_size, len(ranked) // 5)
    if len(ranked) < 2 * quintile:
        return ()
    leg = rules.leg_gross / Decimal(quintile)
    losers = ranked[:quintile]
    winners = ranked[-quintile:]
    sign = Decimal(-1) if reverse else Decimal(1)
    weights = [(contract_id, -leg * sign) for contract_id, _ in losers]
    weights += [(contract_id, leg * sign) for contract_id, _ in winners]
    return tuple(sorted(weights, key=lambda item: item[0]))


def _time_series_weights(
    returns: dict[str, Decimal],
    histories: dict[str, ContractHistory],
    eligible: tuple[str, ...],
    decision_close_ns: int,
    rules: PanelWeightRules,
) -> tuple[tuple[str, Decimal], ...]:
    raw: dict[str, Decimal] = {}
    for contract_id in eligible:
        trailing = returns.get(contract_id)
        if trailing is None or trailing == 0:
            continue
        sigma = _annualised_volatility(
            histories[contract_id], decision_close_ns, rules.volatility_window_days
        )
        if sigma is None:
            continue
        raw[contract_id] = (Decimal(1) if trailing > 0 else Decimal(-1)) / max(
            sigma, rules.volatility_floor
        )
    if not raw:
        return ()
    cap = rules.time_series_cap_numerator / Decimal(len(raw))
    weights = _normalise(raw)
    for _ in range(_MAX_CAP_ROUNDS):
        clipped = {
            contract_id: max(-cap, min(cap, value)) for contract_id, value in weights.items()
        }
        weights = _normalise(clipped)
        if all(abs(value) <= cap + _TOLERANCE for value in weights.values()):
            break
    return tuple(sorted(weights.items(), key=lambda item: item[0]))


def _normalise(weights: dict[str, Decimal]) -> dict[str, Decimal]:
    gross = sum((abs(value) for value in weights.values()), Decimal(0))
    if gross == 0:
        return weights
    return {contract_id: value / gross for contract_id, value in weights.items()}


def _annualised_volatility(
    history: ContractHistory, decision_close_ns: int, window_days: int
) -> Decimal | None:
    observed = [value for value in history.close_times if value <= decision_close_ns]
    window = observed[-(window_days + 1) :]
    if len(window) < window_days + 1:
        return None
    log_returns: list[Decimal] = []
    for previous, current in zip(window, window[1:], strict=True):
        earlier = history.closes[previous]
        later = history.closes[current]
        if earlier <= 0 or later <= 0:
            return None
        log_returns.append((later / earlier).ln())
    count = Decimal(len(log_returns))
    mean = sum(log_returns, Decimal(0)) / count
    variance = sum(((value - mean) ** 2 for value in log_returns), Decimal(0)) / (
        count - Decimal(1)
    )
    return variance.sqrt() * Decimal(365).sqrt()
```

- [ ] **Step 4: Run tests, lint and types**

Run:
```powershell
$env:TEMP='C:\Users\User\AppData\Local\Temp'; $env:TMP=$env:TEMP
uv run pytest tests/test_panel_signals.py -v
uv run ruff check src/trading_bot/panel_signals.py tests/test_panel_signals.py
uv run mypy
```
Expected: 5 passed, clean.

- [ ] **Step 5: Commit**

```bash
git add src/trading_bot/panel_signals.py tests/test_panel_signals.py
git commit -m "feat: build panel member and control weight vectors

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Portfolio accounting with turnover, funding and forced closes

**Files:**
- Create: `src/trading_bot/panel_accounting.py`
- Test: `tests/test_panel_accounting.py`

**Interfaces:**
- Consumes: `panel_universe.ContractHistory`; `panel_reader.FundingEvent`; `panel_config.PanelCostTable`.
- Produces: `EpisodeResult(sample_id, member, scenario, gross_return, turnover, trading_cost, funding_cost, forced_close_cost, net_return, gross_exposure, net_exposure, forced_close_count, contract_contributions, contract_net_contributions, drifted_weights)`; `evaluate_episode(...) -> EpisodeResult`; `PanelAccountingError`.

`contract_contributions` is the gross `w_i × r_i`. `contract_net_contributions` charges each contract its own costs, so the values sum exactly to `net_return`; Task 11's concentration gate reads that field.

Cost algebra, per unit of turnover and per side, in basis points:
- `trading_cost = Σ_i |w_i − drifted_prev_i| × (fee_bps_per_side + slippage_bps(tier_i)) / 10_000`
- `funding_cost = Σ_i Σ_events term(i, e)` where `term = w_i × rate`, multiplied by `funding_payment_multiplier` when positive and by `funding_receipt_multiplier` when negative. Weights are the decision weights, not drifted: a documented, conservative simplification.
- `forced_close_cost = Σ_{forced i} |w_i| × (fee + slippage(tier_i)) / 10_000 × forced_close_multiplier`
- `net_return = gross_return − trading_cost − funding_cost − forced_close_cost`
- `drifted_i = w_i × (1 + r_i) / (1 + gross_return)`, and `0` for a force-closed contract. If `1 + gross_return == 0`, every drifted weight is `0`.
- A contract present only in `previous_weights` is charged at tier 2.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_panel_accounting.py
from decimal import Decimal

from trading_bot.panel_accounting import evaluate_episode
from trading_bot.panel_config import PanelCostTable
from trading_bot.panel_reader import FundingEvent
from trading_bot.panel_universe import ContractHistory

DAY_NS = 86_400_000_000_000
DECISION = 6 * DAY_NS - 1_000_000
NEXT = DECISION + 7 * DAY_NS

BASE = PanelCostTable(
    name="base",
    fee_bps_per_side=Decimal("5"),
    slippage_bps_per_side_tier_one=Decimal("5"),
    slippage_bps_per_side_tier_two=Decimal("10"),
    funding_receipt_multiplier=Decimal("1"),
    funding_payment_multiplier=Decimal("1"),
    forced_close_multiplier=Decimal("1"),
)
ADVERSE = PanelCostTable(
    name="adverse",
    fee_bps_per_side=Decimal("5"),
    slippage_bps_per_side_tier_one=Decimal("10"),
    slippage_bps_per_side_tier_two=Decimal("20"),
    funding_receipt_multiplier=Decimal("0"),
    funding_payment_multiplier=Decimal("2"),
    forced_close_multiplier=Decimal("2"),
)


def history(contract_id: str, closes: dict[int, str]) -> ContractHistory:
    parsed = {key: Decimal(value) for key, value in closes.items()}
    return ContractHistory(
        contract_id=contract_id,
        instrument_id=contract_id.split(":")[0],
        closes=parsed,
        quote_volumes={key: Decimal("1000") for key in parsed},
        close_times=tuple(sorted(parsed)),
    )


def histories(exit_b: str = "95", *, b_exit_time: int = NEXT) -> dict[str, ContractHistory]:
    return {
        "A:0": history("A:0", {DECISION: "100", NEXT: "110"}),
        "B:0": history("B:0", {DECISION: "100", b_exit_time: exit_b}),
    }


WEIGHTS = (("A:0", Decimal("0.5")), ("B:0", Decimal("-0.5")))
TIERS = {"A:0": 1, "B:0": 1}


def test_base_episode_arithmetic() -> None:
    result = evaluate_episode(
        sample_id="BINANCE_UM:1:w1",
        member="xs_mom_1w",
        decision_close_ns=DECISION,
        holding_days=7,
        weights=WEIGHTS,
        previous_weights=(),
        histories=histories(),
        tiers=TIERS,
        funding_by_contract={
            "A:0": (
                FundingEvent(
                    contract_id="A:0",
                    instrument_id="A",
                    calc_time_ns=DECISION + DAY_NS,
                    rate=Decimal("0.001"),
                ),
            )
        },
        cost_table=BASE,
    )
    assert result.gross_return == Decimal("0.075")
    assert result.turnover == Decimal("1")
    assert result.trading_cost == Decimal("0.001")
    assert result.funding_cost == Decimal("0.0005")
    assert result.forced_close_cost == Decimal("0")
    assert result.net_return == Decimal("0.0735")
    assert result.gross_exposure == Decimal("1")
    assert result.net_exposure == Decimal("0")
    assert dict(result.contract_contributions)["A:0"] == Decimal("0.05")
    net_attribution = dict(result.contract_net_contributions)
    assert net_attribution["A:0"] == Decimal("0.049")
    assert net_attribution["B:0"] == Decimal("0.0245")
    assert sum(net_attribution.values(), Decimal(0)) == result.net_return


def test_adverse_doubles_slippage_and_funding_payments() -> None:
    result = evaluate_episode(
        sample_id="BINANCE_UM:1:w1",
        member="xs_mom_1w",
        decision_close_ns=DECISION,
        holding_days=7,
        weights=WEIGHTS,
        previous_weights=(),
        histories=histories(),
        tiers=TIERS,
        funding_by_contract={
            "A:0": (
                FundingEvent(
                    contract_id="A:0",
                    instrument_id="A",
                    calc_time_ns=DECISION + DAY_NS,
                    rate=Decimal("0.001"),
                ),
            )
        },
        cost_table=ADVERSE,
    )
    assert result.trading_cost == Decimal("0.0015")
    assert result.funding_cost == Decimal("0.001")
    assert result.net_return == Decimal("0.0725")


def test_adverse_drops_funding_receipts() -> None:
    result = evaluate_episode(
        sample_id="BINANCE_UM:1:w1",
        member="xs_mom_1w",
        decision_close_ns=DECISION,
        holding_days=7,
        weights=WEIGHTS,
        previous_weights=(),
        histories=histories(),
        tiers=TIERS,
        funding_by_contract={
            "B:0": (
                FundingEvent(
                    contract_id="B:0",
                    instrument_id="B",
                    calc_time_ns=DECISION + DAY_NS,
                    rate=Decimal("0.001"),
                ),
            )
        },
        cost_table=ADVERSE,
    )
    assert result.funding_cost == Decimal("0")


def test_forced_close_uses_the_last_available_close_and_costs_one_side() -> None:
    result = evaluate_episode(
        sample_id="BINANCE_UM:1:w1",
        member="xs_mom_1w",
        decision_close_ns=DECISION,
        holding_days=7,
        weights=WEIGHTS,
        previous_weights=(),
        histories=histories("90", b_exit_time=NEXT - DAY_NS),
        tiers=TIERS,
        funding_by_contract={},
        cost_table=BASE,
    )
    assert result.forced_close_count == 1
    assert result.forced_close_cost == Decimal("0.0005")
    assert dict(result.contract_contributions)["B:0"] == Decimal("0.05")
    assert dict(result.drifted_weights)["B:0"] == Decimal("0")


def test_turnover_uses_drifted_previous_weights() -> None:
    first = evaluate_episode(
        sample_id="BINANCE_UM:1:w1",
        member="xs_mom_1w",
        decision_close_ns=DECISION,
        holding_days=7,
        weights=WEIGHTS,
        previous_weights=(),
        histories=histories(),
        tiers=TIERS,
        funding_by_contract={},
        cost_table=BASE,
    )
    second = evaluate_episode(
        sample_id="BINANCE_UM:2:w1",
        member="xs_mom_1w",
        decision_close_ns=NEXT,
        holding_days=7,
        weights=WEIGHTS,
        previous_weights=first.drifted_weights,
        histories={
            "A:0": ContractHistory(
                contract_id="A:0",
                instrument_id="A",
                closes={NEXT: Decimal("110"), NEXT + 7 * DAY_NS: Decimal("110")},
                quote_volumes={},
                close_times=(NEXT, NEXT + 7 * DAY_NS),
            ),
            "B:0": ContractHistory(
                contract_id="B:0",
                instrument_id="B",
                closes={NEXT: Decimal("95"), NEXT + 7 * DAY_NS: Decimal("95")},
                quote_volumes={},
                close_times=(NEXT, NEXT + 7 * DAY_NS),
            ),
        },
        tiers=TIERS,
        funding_by_contract={},
        cost_table=BASE,
    )
    assert second.turnover < Decimal("0.15")
    assert second.turnover > Decimal("0")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_panel_accounting.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading_bot.panel_accounting'`

- [ ] **Step 3: Write the module**

```python
# src/trading_bot/panel_accounting.py
"""Weekly portfolio accounting for the perpetual panel."""

from dataclasses import dataclass
from decimal import Decimal

from trading_bot.panel_config import PanelCostTable
from trading_bot.panel_reader import FundingEvent
from trading_bot.panel_universe import ContractHistory

DAY_NS = 86_400_000_000_000
_BPS = Decimal(10_000)


class PanelAccountingError(RuntimeError):
    """Raised when an episode cannot be evaluated."""


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    sample_id: str
    member: str
    scenario: str
    gross_return: Decimal
    turnover: Decimal
    trading_cost: Decimal
    funding_cost: Decimal
    forced_close_cost: Decimal
    net_return: Decimal
    gross_exposure: Decimal
    net_exposure: Decimal
    forced_close_count: int
    contract_contributions: tuple[tuple[str, Decimal], ...]
    contract_net_contributions: tuple[tuple[str, Decimal], ...]
    drifted_weights: tuple[tuple[str, Decimal], ...]


def evaluate_episode(
    *,
    sample_id: str,
    member: str,
    decision_close_ns: int,
    holding_days: int,
    weights: tuple[tuple[str, Decimal], ...],
    previous_weights: tuple[tuple[str, Decimal], ...],
    histories: dict[str, ContractHistory],
    tiers: dict[str, int],
    funding_by_contract: dict[str, tuple[FundingEvent, ...]],
    cost_table: PanelCostTable,
) -> EpisodeResult:
    """Evaluate one weekly rebalance under a single cost table."""
    if holding_days < 1:
        raise PanelAccountingError("holding period must be positive")
    exit_close_ns = decision_close_ns + holding_days * DAY_NS
    weight_map = dict(weights)
    previous_map = dict(previous_weights)

    returns: dict[str, Decimal] = {}
    forced: set[str] = set()
    for contract_id, _ in weights:
        history = histories.get(contract_id)
        if history is None:
            raise PanelAccountingError(f"missing history for {contract_id}")
        entry = history.closes.get(decision_close_ns)
        if entry is None or entry <= 0:
            raise PanelAccountingError(f"missing entry close for {contract_id}")
        exit_price = history.closes.get(exit_close_ns)
        if exit_price is None:
            candidates = [
                value
                for value in history.close_times
                if decision_close_ns < value < exit_close_ns
            ]
            forced.add(contract_id)
            exit_price = history.closes[candidates[-1]] if candidates else entry
        returns[contract_id] = exit_price / entry - Decimal(1)

    contributions = {
        contract_id: weight_map[contract_id] * returns[contract_id] for contract_id in weight_map
    }
    gross_return = sum(contributions.values(), Decimal(0))

    universe = sorted(set(weight_map) | set(previous_map))
    turnover = Decimal(0)
    trading_by_contract: dict[str, Decimal] = {}
    for contract_id in universe:
        change = abs(
            weight_map.get(contract_id, Decimal(0)) - previous_map.get(contract_id, Decimal(0))
        )
        if change == 0:
            continue
        turnover += change
        trading_by_contract[contract_id] = (
            change * _per_side_bps(cost_table, tiers.get(contract_id, 2)) / _BPS
        )
    trading_cost = sum(trading_by_contract.values(), Decimal(0))

    funding_by_id: dict[str, Decimal] = {}
    for contract_id, weight in weight_map.items():
        charged = Decimal(0)
        for event in funding_by_contract.get(contract_id, ()):
            if not decision_close_ns < event.calc_time_ns <= exit_close_ns:
                continue
            term = weight * event.rate
            if term > 0:
                charged += term * cost_table.funding_payment_multiplier
            elif term < 0:
                charged += term * cost_table.funding_receipt_multiplier
        if charged != 0:
            funding_by_id[contract_id] = charged
    funding_cost = sum(funding_by_id.values(), Decimal(0))

    forced_by_id: dict[str, Decimal] = {}
    for contract_id in sorted(forced):
        forced_by_id[contract_id] = (
            abs(weight_map[contract_id])
            * _per_side_bps(cost_table, tiers.get(contract_id, 2))
            / _BPS
            * cost_table.forced_close_multiplier
        )
    forced_close_cost = sum(forced_by_id.values(), Decimal(0))

    net_contributions = {
        contract_id: contributions.get(contract_id, Decimal(0))
        - trading_by_contract.get(contract_id, Decimal(0))
        - funding_by_id.get(contract_id, Decimal(0))
        - forced_by_id.get(contract_id, Decimal(0))
        for contract_id in universe
    }
    net_return = gross_return - trading_cost - funding_cost - forced_close_cost
    denominator = Decimal(1) + gross_return
    drifted: dict[str, Decimal] = {}
    for contract_id, weight in weight_map.items():
        if contract_id in forced or denominator == 0:
            drifted[contract_id] = Decimal(0)
            continue
        drifted[contract_id] = weight * (Decimal(1) + returns[contract_id]) / denominator

    return EpisodeResult(
        sample_id=sample_id,
        member=member,
        scenario=cost_table.name,
        gross_return=gross_return,
        turnover=turnover,
        trading_cost=trading_cost,
        funding_cost=funding_cost,
        forced_close_cost=forced_close_cost,
        net_return=net_return,
        gross_exposure=sum((abs(value) for value in weight_map.values()), Decimal(0)),
        net_exposure=sum(weight_map.values(), Decimal(0)),
        forced_close_count=len(forced),
        contract_contributions=tuple(sorted(contributions.items(), key=lambda item: item[0])),
        contract_net_contributions=tuple(
            sorted(net_contributions.items(), key=lambda item: item[0])
        ),
        drifted_weights=tuple(sorted(drifted.items(), key=lambda item: item[0])),
    )


def _per_side_bps(cost_table: PanelCostTable, tier: int) -> Decimal:
    slippage = (
        cost_table.slippage_bps_per_side_tier_one
        if tier == 1
        else cost_table.slippage_bps_per_side_tier_two
    )
    return cost_table.fee_bps_per_side + slippage
```

- [ ] **Step 4: Run tests, lint and types**

Run:
```powershell
$env:TEMP='C:\Users\User\AppData\Local\Temp'; $env:TMP=$env:TEMP
uv run pytest tests/test_panel_accounting.py -v
uv run ruff check src/trading_bot/panel_accounting.py tests/test_panel_accounting.py
uv run mypy
```
Expected: 5 passed, clean.

- [ ] **Step 5: Commit**

```bash
git add src/trading_bot/panel_accounting.py tests/test_panel_accounting.py
git commit -m "feat: account weekly panel episodes with turnover and funding

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Rebalance calendar and walk-forward manifest

**Files:**
- Create: `src/trading_bot/panel_samples.py`
- Test: `tests/test_panel_samples.py`
- Modify: `src/trading_bot/cli.py` (add the `panel-manifest` subcommand)

**Interfaces:**
- Consumes: `panel_reader.load_panel_bars`; `panel_capture.verify_panel_capture`; `panel_config.PanelFamilySpec`; `splits.SplitSample`, `WalkForwardConfig`, `build_walk_forward_views`; `canonical`.
- Produces: `PanelSamplesError`; `rebalance_close_times(bars) -> tuple[int, ...]`; `build_rebalance_samples(bars, *, holding_days) -> tuple[SplitSample, ...]`; `derive_panel_config(samples, *, folds) -> WalkForwardConfig`; `publish_panel_walk_forward(capture_root, *, output_path, spec, family_spec_hash) -> PanelManifestArtifact(output_path, capture_root_hash, dataset_root_hash, manifest_hash, fold_count, pooled_test_sample_count)`.

A rebalance date is the close time of a daily bar whose UTC weekday is Sunday. The Unix epoch began on a Thursday, so the test is `(open_time_ns // DAY_NS) % 7 == 3`, exact integer arithmetic with no calendar library. `sample_id` is `f"BINANCE_UM:{decision_close_ns}:w1"`. Publication fails closed with `INSUFFICIENT_EVIDENCE` when the pooled test-sample count across folds is below `spec.statistics.pooled_episode_floor`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_panel_samples.py
from decimal import Decimal
from pathlib import Path

import pytest

from trading_bot.panel_config import PanelFoldGeometry, load_panel_family_spec
from trading_bot.panel_reader import PanelBar
from trading_bot.panel_samples import (
    PanelSamplesError,
    build_rebalance_samples,
    derive_panel_config,
    rebalance_close_times,
)

DAY_NS = 86_400_000_000_000
SPEC, _ = load_panel_family_spec(Path("configs/xs-momentum-panel-v1.json"))


def bar(day_index: int) -> PanelBar:
    open_time_ns = day_index * DAY_NS
    close_time_ns = open_time_ns + DAY_NS - 1_000_000
    return PanelBar(
        contract_id="BTCUSDT:0",
        instrument_id="BTCUSDT",
        open_time_ns=open_time_ns,
        close_time_ns=close_time_ns,
        available_time_ns=close_time_ns + 1,
        close=Decimal("100"),
        quote_volume=Decimal("1000"),
    )


def test_rebalance_dates_are_sundays() -> None:
    bars = tuple(bar(index) for index in range(21))
    times = rebalance_close_times(bars)
    assert [value // DAY_NS for value in times] == [3, 10, 17]


def test_samples_are_chronological_and_carry_the_holding_label() -> None:
    bars = tuple(bar(index) for index in range(21))
    samples = build_rebalance_samples(bars, holding_days=7)
    assert samples[0].sample_id == f"BINANCE_UM:{4 * DAY_NS - 1_000_000}:w1"
    assert samples[0].label_end_time_ns - samples[0].decision_time_ns == 7 * DAY_NS
    assert all(
        later.decision_time_ns > earlier.decision_time_ns
        for earlier, later in zip(samples, samples[1:], strict=True)
    )


def test_config_places_the_holdout_at_the_end() -> None:
    bars = tuple(bar(index) for index in range(400))
    samples = build_rebalance_samples(bars, holding_days=7)
    geometry = PanelFoldGeometry(
        train_duration_ns=100 * DAY_NS,
        validation_duration_ns=20 * DAY_NS,
        test_duration_ns=40 * DAY_NS,
        step_ns=40 * DAY_NS,
        embargo_ns=14 * DAY_NS,
        holdout_duration_ns=40 * DAY_NS,
    )
    config = derive_panel_config(samples, folds=geometry)
    assert config.final_holdout_start_ns < samples[-1].decision_time_ns
    assert config.final_holdout_start_ns > samples[0].decision_time_ns


def test_short_history_is_rejected() -> None:
    samples = build_rebalance_samples(tuple(bar(index) for index in range(10)), holding_days=7)
    with pytest.raises(PanelSamplesError):
        derive_panel_config(samples, folds=SPEC.folds)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_panel_samples.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading_bot.panel_samples'`

- [ ] **Step 3: Write the module**

```python
# src/trading_bot/panel_samples.py
"""Weekly rebalance calendar and walk-forward manifest for the panel."""

import json
from dataclasses import dataclass
from pathlib import Path

from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.panel_capture import verify_panel_capture
from trading_bot.panel_config import PanelFamilySpec, PanelFoldGeometry
from trading_bot.panel_reader import PanelBar, load_panel_bars
from trading_bot.splits import (
    SplitSample,
    WalkForwardConfig,
    WalkForwardFold,
    build_walk_forward_views,
)

DAY_NS = 86_400_000_000_000
_SUNDAY_REMAINDER = 3  # the Unix epoch began on a Thursday


class PanelSamplesError(RuntimeError):
    """Raised when the panel rebalance calendar cannot be published."""


@dataclass(frozen=True, slots=True)
class PanelManifestArtifact:
    output_path: Path
    capture_root_hash: str
    dataset_root_hash: str
    manifest_hash: str
    fold_count: int
    pooled_test_sample_count: int


def rebalance_close_times(bars: tuple[PanelBar, ...]) -> tuple[int, ...]:
    """Return the sorted close times of Sunday bars observed anywhere in the panel."""
    times = {
        bar.close_time_ns
        for bar in bars
        if (bar.open_time_ns // DAY_NS) % 7 == _SUNDAY_REMAINDER
    }
    return tuple(sorted(times))


def build_rebalance_samples(
    bars: tuple[PanelBar, ...], *, holding_days: int
) -> tuple[SplitSample, ...]:
    if holding_days < 1:
        raise PanelSamplesError("holding period must be positive")
    return tuple(
        SplitSample(
            sample_id=f"BINANCE_UM:{close_time_ns}:w1",
            decision_time_ns=close_time_ns,
            label_end_time_ns=close_time_ns + holding_days * DAY_NS,
        )
        for close_time_ns in rebalance_close_times(bars)
    )


def derive_panel_config(
    samples: tuple[SplitSample, ...], *, folds: PanelFoldGeometry
) -> WalkForwardConfig:
    if len(samples) < 2:
        raise PanelSamplesError("panel capture cannot define a rebalance calendar")
    spacing = samples[-1].decision_time_ns - samples[-2].decision_time_ns
    if folds.holdout_duration_ns < spacing:
        raise PanelSamplesError("holdout duration is shorter than one rebalance interval")
    holdout_start = samples[-1].decision_time_ns - folds.holdout_duration_ns + spacing
    if holdout_start <= samples[0].decision_time_ns:
        raise PanelSamplesError("panel capture is too short for the declared geometry")
    return WalkForwardConfig(
        train_duration_ns=folds.train_duration_ns,
        validation_duration_ns=folds.validation_duration_ns,
        test_duration_ns=folds.test_duration_ns,
        step_ns=folds.step_ns,
        embargo_ns=folds.embargo_ns,
        final_holdout_start_ns=holdout_start,
    )


def publish_panel_walk_forward(
    capture_root: Path,
    *,
    output_path: Path,
    spec: PanelFamilySpec,
    family_spec_hash: str,
) -> PanelManifestArtifact:
    valid, errors = verify_panel_capture(capture_root)
    if not valid:
        raise PanelSamplesError("panel capture verification failed: " + ",".join(errors))
    if output_path.exists():
        raise PanelSamplesError("panel manifest already exists and is immutable")
    capture_manifest = _load_object(capture_root / "capture-manifest.json")
    dataset_manifest = _load_object(capture_root / "dataset" / "dataset-manifest.json")
    capture_hash = str(capture_manifest["capture_root_hash"])
    dataset_hash = str(dataset_manifest["root_hash"])

    bars = load_panel_bars(capture_root / "dataset")
    samples = build_rebalance_samples(bars, holding_days=spec.holding_days)
    config = derive_panel_config(samples, folds=spec.folds)
    views = build_walk_forward_views(list(samples), config)
    if not views.folds:
        raise PanelSamplesError("panel capture does not contain a complete fold")
    pooled = sum(len(fold.test_ids) for fold in views.folds)
    if pooled < spec.statistics.pooled_episode_floor:
        raise PanelSamplesError(
            f"INSUFFICIENT_EVIDENCE: pooled test samples {pooled} below "
            f"{spec.statistics.pooled_episode_floor}"
        )

    material: dict[str, object] = {
        "manifest_version": "1.0.0",
        "family_name": spec.family_name,
        "family_spec_hash": family_spec_hash,
        "capture_root_hash": capture_hash,
        "dataset_root_hash": dataset_hash,
        "holding_days": spec.holding_days,
        "split_manifest_hash": views.manifest_hash,
        "config": {
            "train_duration_ns": config.train_duration_ns,
            "validation_duration_ns": config.validation_duration_ns,
            "test_duration_ns": config.test_duration_ns,
            "step_ns": config.step_ns,
            "embargo_ns": config.embargo_ns,
            "final_holdout_start_ns": config.final_holdout_start_ns,
        },
        "folds": [_fold_record(fold) for fold in views.folds],
        "final_holdout_ids": list(views.final_holdout_ids),
        "pooled_test_sample_count": pooled,
    }
    manifest_hash = content_sha256(material)
    document = dict(material)
    document["manifest_hash"] = manifest_hash
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_bytes(canonical_json(document))
    temporary.replace(output_path)
    return PanelManifestArtifact(
        output_path=output_path,
        capture_root_hash=capture_hash,
        dataset_root_hash=dataset_hash,
        manifest_hash=manifest_hash,
        fold_count=len(views.folds),
        pooled_test_sample_count=pooled,
    )


def verify_panel_manifest(path: Path) -> bool:
    try:
        document = _load_object(path)
        recorded = str(document["manifest_hash"])
        material = {key: value for key, value in document.items() if key != "manifest_hash"}
        return content_sha256(material) == recorded
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False


def _fold_record(fold: WalkForwardFold) -> dict[str, object]:
    return {
        "fold_index": fold.fold_index,
        "train_start_ns": fold.train_start_ns,
        "train_end_ns": fold.train_end_ns,
        "validation_start_ns": fold.validation_start_ns,
        "validation_end_ns": fold.validation_end_ns,
        "test_start_ns": fold.test_start_ns,
        "test_end_ns": fold.test_end_ns,
        "train_ids": list(fold.train_ids),
        "validation_ids": list(fold.validation_ids),
        "test_ids": list(fold.test_ids),
    }


def _load_object(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise PanelSamplesError(f"{path.name} must contain a JSON object")
    return document
```

- [ ] **Step 4: Add the CLI subcommand**

In `src/trading_bot/cli.py`, add after the `panel-capture` parser block:

```python
    panel_manifest = commands.add_parser("panel-manifest")
    panel_manifest.add_argument("--workspace-root", type=Path, default=Path.cwd())
    panel_manifest.add_argument("--capture", type=Path, required=True)
    panel_manifest.add_argument("--output", type=Path, required=True)
    panel_manifest.add_argument("--family-spec", type=Path, required=True)
```

and in the dispatch chain:

```python
    if parsed.command == "panel-manifest":
        workspace = parsed.workspace_root.resolve()
        paths = (parsed.capture.resolve(), parsed.output.resolve(), parsed.family_spec.resolve())
        if any(not path.is_relative_to(workspace) for path in paths):
            raise ValueError("panel manifest paths must stay inside workspace")
        spec, spec_hash = load_panel_family_spec(paths[2])
        publish_panel_walk_forward(
            paths[0], output_path=paths[1], spec=spec, family_spec_hash=spec_hash
        )
        return 0
```

with imports `from trading_bot.panel_config import load_panel_family_spec` and `from trading_bot.panel_samples import publish_panel_walk_forward`.

- [ ] **Step 5: Run tests, lint and types**

Run:
```powershell
$env:TEMP='C:\Users\User\AppData\Local\Temp'; $env:TMP=$env:TEMP
uv run pytest tests/test_panel_samples.py -v
uv run ruff check src/trading_bot/panel_samples.py src/trading_bot/cli.py tests/test_panel_samples.py
uv run mypy
```
Expected: 4 passed, clean.

- [ ] **Step 6: Commit**

```bash
git add src/trading_bot/panel_samples.py src/trading_bot/cli.py tests/test_panel_samples.py
git commit -m "feat: publish the weekly panel walk-forward manifest

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Fold runner

**Files:**
- Create: `src/trading_bot/panel_fold_run.py`
- Test: `tests/test_panel_fold_run.py`
- Modify: `src/trading_bot/cli.py` (add the `panel-fold` subcommand)

**Interfaces:**
- Consumes: everything from Tasks 1 to 8; `registry.MetadataRegistry`, `ExperimentRecord`.
- Produces: `PanelFoldError`; `PanelFoldArtifact(output_path, report_hash, fold_index, episode_count, skipped_sample_count)`; `run_panel_fold(capture_root, *, manifest_path, family_spec_path, output_path, registry_path, fold_index) -> PanelFoldArtifact`; `verify_panel_fold_report(path) -> bool`.

Behaviour:
- Verify the capture and the manifest, and refuse when `capture_root_hash`, `dataset_root_hash` or `family_spec_hash` disagree with the manifest.
- Load bars with `available_before_ns = test_end_ns + 1`. That admits exactly the purged test window: an exit bar closing before `test_end_ns` has `available_time_ns <= test_end_ns`, while a bar closing at or after the boundary does not.
- Walk the fold's `test_ids` in chronological order. For each, build the universe, build every member and control weight vector, and evaluate one episode per scenario, chaining `drifted_weights` per member and scenario. The first episode of a fold starts flat.
- A rebalance whose universe is empty produces no episode; its `sample_id` is recorded under `skipped_sample_ids`, the running position resets to flat, and `SKIPPED_WEEK_EXIT_COST_UNCHARGED` is added to `reason_codes`.
- Emit every episode's fields so Task 11 can pool them. Register one completed `ExperimentRecord` per member.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_panel_fold_run.py
import io
import json
import zipfile
from pathlib import Path

import pytest

from trading_bot.panel_capture import PanelPayload, capture_panel
from trading_bot.panel_config import load_panel_family_spec
from trading_bot.panel_fold_run import PanelFoldError, run_panel_fold, verify_panel_fold_report
from trading_bot.panel_samples import publish_panel_walk_forward
from trading_bot.registry import MetadataRegistry

DAY_MS = 86_400_000
MONTHS = ("2020-01", "2020-02", "2020-03", "2020-04", "2020-05", "2020-06", "2020-07")
SYMBOLS = tuple(f"C{index:02d}USDT" for index in range(12))
KLINE_HEADER = (
    "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
    "taker_buy_volume,taker_buy_quote_volume,ignore"
)
MONTH_START_DAY = {
    "2020-01": 0,
    "2020-02": 31,
    "2020-03": 60,
    "2020-04": 91,
    "2020-05": 121,
    "2020-06": 152,
    "2020-07": 182,
}
MONTH_DAYS = {
    "2020-01": 31,
    "2020-02": 29,
    "2020-03": 31,
    "2020-04": 30,
    "2020-05": 31,
    "2020-06": 30,
    "2020-07": 31,
}
EPOCH_DAY_2020 = 18_262  # 2020-01-01 in days since the Unix epoch


def zip_bytes(name: str, text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, text)
    return buffer.getvalue()


def kline_csv(symbol: str, month: str) -> str:
    seed = int(symbol[1:3])
    lines = [KLINE_HEADER]
    for offset in range(MONTH_DAYS[month]):
        day = EPOCH_DAY_2020 + MONTH_START_DAY[month] + offset
        open_ms = day * DAY_MS
        close = 100 + seed * 10 + (day % 11) + (seed * (day % 5)) / 4
        lines.append(
            f"{open_ms},{close},{close},{close},{close},10,{open_ms + DAY_MS - 1},"
            f"50000000,100,50,25000000,0"
        )
    return "\n".join(lines) + "\n"


def fetch(url: str) -> PanelPayload:
    symbol = next(item for item in SYMBOLS if f"/{item}/" in url or f"/{item}-" in url)
    month = next(item for item in MONTHS if item in url)
    if "fundingRate" in url:
        day = EPOCH_DAY_2020 + MONTH_START_DAY[month]
        text = "calc_time,funding_interval_hours,last_funding_rate\n"
        text += f"{day * DAY_MS},8,0.0001\n"
        return PanelPayload(url=url, raw_bytes=zip_bytes("f.csv", text), received_time_ns=1)
    return PanelPayload(
        url=url, raw_bytes=zip_bytes("k.csv", kline_csv(symbol, month)), received_time_ns=1
    )


def small_config(tmp_path: Path) -> Path:
    document = json.loads(Path("configs/xs-momentum-panel-v1.json").read_text(encoding="utf-8"))
    document["universe"].update(
        {
            "minimum_history_days": 20,
            "liquidity_window_days": 5,
            "maximum_contracts": 10,
            "minimum_contracts": 10,
            "tier_one_rank_limit": 4,
        }
    )
    document["weights"]["minimum_quintile_size"] = 2
    document["weights"]["volatility_window_days"] = 10
    day_ns = 86_400_000_000_000
    document["folds"] = {
        "train_duration_ns": 60 * day_ns,
        "validation_duration_ns": 14 * day_ns,
        "test_duration_ns": 28 * day_ns,
        "step_ns": 28 * day_ns,
        "embargo_ns": 14 * day_ns,
        "holdout_duration_ns": 28 * day_ns,
    }
    document["statistics"].update({"block_length": 2, "pooled_episode_floor": 4})
    path = tmp_path / "small-panel.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    capture_panel(
        workspace_root=tmp_path,
        output_directory=tmp_path / "capture",
        reserve_bytes=0,
        symbols=SYMBOLS,
        months=MONTHS,
        fetch=fetch,
    )
    config_path = small_config(tmp_path)
    spec, spec_hash = load_panel_family_spec(config_path)
    publish_panel_walk_forward(
        tmp_path / "capture",
        output_path=tmp_path / "manifest.json",
        spec=spec,
        family_spec_hash=spec_hash,
    )
    return tmp_path, tmp_path / "capture", config_path


def test_fold_report_carries_every_member_and_its_episodes(
    workspace: tuple[Path, Path, Path],
) -> None:
    root, capture_root, config_path = workspace
    artifact = run_panel_fold(
        capture_root,
        manifest_path=root / "manifest.json",
        family_spec_path=config_path,
        output_path=root / "fold0.json",
        registry_path=root / "registry.sqlite3",
        fold_index=0,
    )
    document = json.loads(artifact.output_path.read_text(encoding="utf-8"))
    assert document["status"] == "development_only"
    assert document["fold_index"] == 0
    names = [item["candidate_name"] for item in document["candidates"]]
    assert names == [
        "xs_mom_1w",
        "xs_mom_4w",
        "xs_mom_12w",
        "ts_mom_4w",
        "ts_mom_12w",
        "xs_rev_1w",
        "no_trade",
        "random_ranks",
        "passive_long_ew",
    ]
    momentum = next(item for item in document["candidates"] if item["candidate_name"] == "xs_mom_1w")
    assert momentum["role"] == "member"
    assert momentum["episode_count"] == artifact.episode_count > 0
    assert len(momentum["base"]["episodes"]) == artifact.episode_count
    assert len(momentum["adverse"]["episodes"]) == artifact.episode_count
    episode = momentum["base"]["episodes"][0]
    assert set(episode) >= {
        "sample_id",
        "net_return",
        "gross_return",
        "turnover",
        "trading_cost",
        "funding_cost",
        "forced_close_cost",
        "gross_exposure",
        "net_exposure",
        "forced_close_count",
        "contract_net_contributions",
    }
    assert verify_panel_fold_report(artifact.output_path)


def test_no_trade_control_has_zero_returns(workspace: tuple[Path, Path, Path]) -> None:
    root, capture_root, config_path = workspace
    run_panel_fold(
        capture_root,
        manifest_path=root / "manifest.json",
        family_spec_path=config_path,
        output_path=root / "fold0.json",
        registry_path=root / "registry.sqlite3",
        fold_index=0,
    )
    document = json.loads((root / "fold0.json").read_text(encoding="utf-8"))
    no_trade = next(
        item for item in document["candidates"] if item["candidate_name"] == "no_trade"
    )
    assert all(episode["net_return"] == "0" for episode in no_trade["base"]["episodes"])


def test_members_are_registered(workspace: tuple[Path, Path, Path]) -> None:
    root, capture_root, config_path = workspace
    run_panel_fold(
        capture_root,
        manifest_path=root / "manifest.json",
        family_spec_path=config_path,
        output_path=root / "fold0.json",
        registry_path=root / "registry.sqlite3",
        fold_index=0,
    )
    with MetadataRegistry(root / "registry.sqlite3") as registry:
        rows = registry.list_experiments()
    assert {row.candidate_name for row in rows} == {
        "xs_mom_1w",
        "xs_mom_4w",
        "xs_mom_12w",
        "ts_mom_4w",
        "ts_mom_12w",
        "xs_rev_1w",
    }
    assert all(row.outcome == "completed" for row in rows)


def test_report_is_immutable(workspace: tuple[Path, Path, Path]) -> None:
    root, capture_root, config_path = workspace
    for _ in range(1):
        run_panel_fold(
            capture_root,
            manifest_path=root / "manifest.json",
            family_spec_path=config_path,
            output_path=root / "fold0.json",
            registry_path=root / "registry.sqlite3",
            fold_index=0,
        )
    with pytest.raises(PanelFoldError):
        run_panel_fold(
            capture_root,
            manifest_path=root / "manifest.json",
            family_spec_path=config_path,
            output_path=root / "fold0.json",
            registry_path=root / "registry.sqlite3",
            fold_index=0,
        )


def test_unknown_fold_index_is_rejected(workspace: tuple[Path, Path, Path]) -> None:
    root, capture_root, config_path = workspace
    with pytest.raises(PanelFoldError):
        run_panel_fold(
            capture_root,
            manifest_path=root / "manifest.json",
            family_spec_path=config_path,
            output_path=root / "fold9.json",
            registry_path=root / "registry.sqlite3",
            fold_index=9,
        )
```

- [ ] **Step 2: Add `list_experiments` to the registry if it is absent**

Run: `grep -n "def list_experiments" src/trading_bot/registry.py`

If the method does not exist, add it next to `get_experiment`:

```python
    def list_experiments(self) -> tuple[ExperimentRecord, ...]:
        connection = self._require_connection()
        rows = connection.execute(
            """
            SELECT experiment_id, family_id, candidate_name, hypothesis, split_manifest_hash,
                   code_hash, random_seed, outcome, result_hash, failure_reason, created_at_ns
            FROM experiments ORDER BY candidate_name, experiment_id
            """
        ).fetchall()
        return tuple(
            ExperimentRecord(
                experiment_id=UUID(row[0]),
                family_id=UUID(row[1]),
                candidate_name=row[2],
                hypothesis=row[3],
                split_manifest_hash=row[4],
                code_hash=row[5],
                random_seed=row[6],
                outcome=row[7],
                result_hash=row[8],
                failure_reason=row[9],
                created_at_ns=row[10],
            )
            for row in rows
        )
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_panel_fold_run.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading_bot.panel_fold_run'`

- [ ] **Step 4: Write the module**

```python
# src/trading_bot/panel_fold_run.py
"""Evaluate one walk-forward fold of the panel family."""

import json
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.panel_accounting import EpisodeResult, evaluate_episode
from trading_bot.panel_capture import verify_panel_capture
from trading_bot.panel_config import PanelCostTable, PanelFamilySpec, load_panel_family_spec
from trading_bot.panel_reader import FundingEvent, load_funding_events, load_panel_bars
from trading_bot.panel_samples import verify_panel_manifest
from trading_bot.panel_signals import build_weight_vectors
from trading_bot.panel_universe import build_contract_histories, select_universe
from trading_bot.registry import ExperimentRecord, MetadataRegistry

_PANEL_MODULES = (
    "panel_config.py",
    "panel_dataset.py",
    "panel_capture.py",
    "panel_reader.py",
    "panel_universe.py",
    "panel_signals.py",
    "panel_accounting.py",
    "panel_samples.py",
    "panel_fold_run.py",
    "evaluation.py",
    "strategy.py",
)


class PanelFoldError(RuntimeError):
    """Raised when a panel fold cannot be evaluated or published."""


@dataclass(frozen=True, slots=True)
class PanelFoldArtifact:
    output_path: Path
    report_hash: str
    fold_index: int
    episode_count: int
    skipped_sample_count: int


def run_panel_fold(
    capture_root: Path,
    *,
    manifest_path: Path,
    family_spec_path: Path,
    output_path: Path,
    registry_path: Path,
    fold_index: int,
) -> PanelFoldArtifact:
    valid, errors = verify_panel_capture(capture_root)
    if not valid:
        raise PanelFoldError("panel capture verification failed: " + ",".join(errors))
    if not verify_panel_manifest(manifest_path):
        raise PanelFoldError("panel manifest verification failed")
    if output_path.exists():
        raise PanelFoldError("panel fold report already exists and is immutable")

    spec, family_spec_hash = load_panel_family_spec(family_spec_path)
    manifest = _load_object(manifest_path)
    if manifest.get("family_spec_hash") != family_spec_hash:
        raise PanelFoldError("family declaration does not match the manifest")
    capture_manifest = _load_object(capture_root / "capture-manifest.json")
    dataset_manifest = _load_object(capture_root / "dataset" / "dataset-manifest.json")
    if manifest.get("capture_root_hash") != capture_manifest.get("capture_root_hash"):
        raise PanelFoldError("manifest is not linked to this capture")
    if manifest.get("dataset_root_hash") != dataset_manifest.get("root_hash"):
        raise PanelFoldError("manifest is not linked to this dataset")

    folds = manifest.get("folds")
    if not isinstance(folds, list):
        raise PanelFoldError("manifest folds are malformed")
    fold = next((item for item in folds if item.get("fold_index") == fold_index), None)
    if fold is None:
        raise PanelFoldError(f"fold {fold_index} is not in the manifest")

    test_end_ns = int(fold["test_end_ns"])
    bars = load_panel_bars(capture_root / "dataset", available_before_ns=test_end_ns + 1)
    histories = build_contract_histories(bars)
    funding_by_contract: dict[str, tuple[FundingEvent, ...]] = {}
    for event in load_funding_events(capture_root / "dataset"):
        funding_by_contract.setdefault(event.contract_id, ())
        funding_by_contract[event.contract_id] += (event,)

    test_ids = [str(value) for value in fold["test_ids"]]
    decisions = sorted(int(value.split(":")[1]) for value in test_ids)
    names = [member.name for member in spec.members] + [item.name for item in spec.controls]
    scenarios: tuple[PanelCostTable, ...] = (spec.costs.base, spec.costs.adverse)
    episodes: dict[tuple[str, str], list[EpisodeResult]] = {
        (name, scenario.name): [] for name in names for scenario in scenarios
    }
    carried: dict[tuple[str, str], tuple[tuple[str, Decimal], ...]] = {
        key: () for key in episodes
    }
    skipped: list[str] = []

    for decision_close_ns in decisions:
        sample_id = f"BINANCE_UM:{decision_close_ns}:w1"
        snapshot = select_universe(
            histories, decision_close_ns=decision_close_ns, rules=spec.universe
        )
        if not snapshot.contracts:
            skipped.append(sample_id)
            for key in carried:
                carried[key] = ()
            continue
        tiers = {item.contract_id: item.tier for item in snapshot.contracts}
        vectors = build_weight_vectors(histories, snapshot, spec=spec)
        for name in names:
            for scenario in scenarios:
                key = (name, scenario.name)
                result = evaluate_episode(
                    sample_id=sample_id,
                    member=name,
                    decision_close_ns=decision_close_ns,
                    holding_days=spec.holding_days,
                    weights=vectors[name].weights,
                    previous_weights=carried[key],
                    histories=histories,
                    tiers=tiers,
                    funding_by_contract=funding_by_contract,
                    cost_table=scenario,
                )
                episodes[key].append(result)
                carried[key] = result.drifted_weights

    episode_count = len(episodes[(names[0], "base")])
    reason_codes: list[str] = []
    if skipped:
        reason_codes.append("SKIPPED_WEEK_EXIT_COST_UNCHARGED")
    if episode_count == 0:
        reason_codes.append("NO_EPISODES_IN_FOLD")

    member_names = {member.name for member in spec.members}
    candidates = [
        {
            "candidate_name": name,
            "role": "member" if name in member_names else "control",
            "episode_count": len(episodes[(name, "base")]),
            "base": _scenario_record(episodes[(name, "base")]),
            "adverse": _scenario_record(episodes[(name, "adverse")]),
        }
        for name in names
    ]

    material: dict[str, object] = {
        "report_version": "1.0.0",
        "status": "development_only",
        "reason_codes": reason_codes,
        "family_name": spec.family_name,
        "family_spec_hash": family_spec_hash,
        "capture_root_hash": str(manifest["capture_root_hash"]),
        "dataset_root_hash": str(manifest["dataset_root_hash"]),
        "split_manifest_hash": str(manifest["split_manifest_hash"]),
        "manifest_hash": str(manifest["manifest_hash"]),
        "fold_index": fold_index,
        "fold_count": len(folds),
        "train_sample_count": len(fold["train_ids"]),
        "validation_sample_count": len(fold["validation_ids"]),
        "test_sample_count": len(test_ids),
        "train_membership_hash": content_sha256(list(fold["train_ids"])),
        "validation_membership_hash": content_sha256(list(fold["validation_ids"])),
        "test_membership_hash": content_sha256(test_ids),
        "random_seed": spec.statistics.random_seed,
        "block_length": spec.statistics.block_length,
        "bootstrap_repetitions": spec.statistics.bootstrap_repetitions,
        "skipped_sample_ids": skipped,
        "code_hash": _code_hash(),
        "candidates": candidates,
    }
    report_hash = content_sha256(material)
    document = dict(material)
    document["report_hash"] = report_hash
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_bytes(canonical_json(document))
    temporary.replace(output_path)

    _register(spec, family_spec_hash, manifest, report_hash, registry_path)
    return PanelFoldArtifact(
        output_path=output_path,
        report_hash=report_hash,
        fold_index=fold_index,
        episode_count=episode_count,
        skipped_sample_count=len(skipped),
    )


def verify_panel_fold_report(path: Path) -> bool:
    try:
        document = _load_object(path)
        recorded = str(document["report_hash"])
        material = {key: value for key, value in document.items() if key != "report_hash"}
        return content_sha256(material) == recorded
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False


def _scenario_record(results: list[EpisodeResult]) -> dict[str, object]:
    return {
        "total_net_return": sum((item.net_return for item in results), Decimal(0)),
        "episodes": [
            {
                "sample_id": item.sample_id,
                "net_return": item.net_return,
                "gross_return": item.gross_return,
                "turnover": item.turnover,
                "trading_cost": item.trading_cost,
                "funding_cost": item.funding_cost,
                "forced_close_cost": item.forced_close_cost,
                "gross_exposure": item.gross_exposure,
                "net_exposure": item.net_exposure,
                "forced_close_count": item.forced_close_count,
                "contract_net_contributions": [
                    [contract_id, value] for contract_id, value in item.contract_net_contributions
                ],
            }
            for item in results
        ],
    }


def _register(
    spec: PanelFamilySpec,
    family_spec_hash: str,
    manifest: dict[str, object],
    report_hash: str,
    registry_path: Path,
) -> None:
    split_hash = str(manifest["split_manifest_hash"])
    family_id = uuid5(NAMESPACE_URL, f"{split_hash}:{spec.family_name}")
    created_at_ns = time.time_ns()
    with MetadataRegistry(registry_path) as registry:
        for member in spec.members:
            registry.register_experiment(
                ExperimentRecord(
                    experiment_id=uuid5(
                        family_id,
                        f"{member.name}:{spec.statistics.random_seed}:{report_hash}",
                    ),
                    family_id=family_id,
                    candidate_name=member.name,
                    hypothesis=spec.hypothesis,
                    split_manifest_hash=split_hash,
                    code_hash=_code_hash(),
                    random_seed=spec.statistics.random_seed,
                    outcome="completed",
                    result_hash=report_hash,
                    failure_reason=None,
                    created_at_ns=created_at_ns,
                )
            )
        _ = family_spec_hash


def _code_hash() -> str:
    root = Path(__file__).parent
    material = "".join(
        (root / name).read_text(encoding="utf-8") for name in _PANEL_MODULES
    )
    return content_sha256(material)


def _load_object(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise PanelFoldError(f"{path.name} must contain a JSON object")
    return document
```

- [ ] **Step 5: Add the CLI subcommand**

In `src/trading_bot/cli.py`, add the parser:

```python
    panel_fold = commands.add_parser("panel-fold")
    panel_fold.add_argument("--workspace-root", type=Path, default=Path.cwd())
    panel_fold.add_argument("--capture", type=Path, required=True)
    panel_fold.add_argument("--manifest", type=Path, required=True)
    panel_fold.add_argument("--family-spec", type=Path, required=True)
    panel_fold.add_argument("--output", type=Path, required=True)
    panel_fold.add_argument("--registry", type=Path, required=True)
    panel_fold.add_argument("--fold-index", type=int, required=True)
```

and the dispatch:

```python
    if parsed.command == "panel-fold":
        workspace = parsed.workspace_root.resolve()
        paths = (
            parsed.capture.resolve(),
            parsed.manifest.resolve(),
            parsed.family_spec.resolve(),
            parsed.output.resolve(),
            parsed.registry.resolve(),
        )
        if any(not path.is_relative_to(workspace) for path in paths):
            raise ValueError("panel fold paths must stay inside workspace")
        run_panel_fold(
            paths[0],
            manifest_path=paths[1],
            family_spec_path=paths[2],
            output_path=paths[3],
            registry_path=paths[4],
            fold_index=parsed.fold_index,
        )
        return 0
```

with `from trading_bot.panel_fold_run import run_panel_fold`.

- [ ] **Step 6: Run tests, lint and types**

Run:
```powershell
$env:TEMP='C:\Users\User\AppData\Local\Temp'; $env:TMP=$env:TEMP
uv run pytest tests/test_panel_fold_run.py -v
uv run ruff check src/trading_bot/panel_fold_run.py src/trading_bot/cli.py src/trading_bot/registry.py tests/test_panel_fold_run.py
uv run mypy
```
Expected: 5 passed, clean.

- [ ] **Step 7: Commit**

```bash
git add src/trading_bot/panel_fold_run.py src/trading_bot/cli.py src/trading_bot/registry.py tests/test_panel_fold_run.py
git commit -m "feat: evaluate one panel walk-forward fold

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Annualised and deflated Sharpe

**Files:**
- Create: `src/trading_bot/panel_statistics.py`
- Test: `tests/test_panel_statistics.py`

**Interfaces:**
- Consumes: nothing from the panel modules; stdlib only.
- Produces: `PanelStatisticsError`; `sharpe_ratio(net_returns) -> Decimal` (per period); `annualised_sharpe(net_returns, *, periods_per_year=52) -> Decimal`; `deflated_sharpe_ratio(net_returns, *, trial_sharpes) -> Decimal`.

The deflated Sharpe ratio is Bailey and López de Prado (2014):

```text
SR*  = sd(trial Sharpes) × [ (1 − γ)·Z⁻¹(1 − 1/N) + γ·Z⁻¹(1 − 1/(N·e)) ],  γ = 0.5772156649
DSR  = Z[ (SR − SR*)·sqrt(T − 1) / sqrt(1 − skew·SR + (kurtosis − 1)/4 · SR²) ]
```

`N` is the number of trials, `T` the number of observations, `skew` the sample skewness and `kurtosis` the non-excess sample kurtosis of the return series. `statistics.NormalDist` supplies `cdf` and `inv_cdf`. This is the **only** place floats are permitted: inputs and outputs are `Decimal`, the conversion happens inside the function, and the result is reported, never used as a gate.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_panel_statistics.py
from decimal import Decimal

import pytest

from trading_bot.panel_statistics import (
    PanelStatisticsError,
    annualised_sharpe,
    deflated_sharpe_ratio,
    sharpe_ratio,
)


def alternating(mean: str, deviation: str, count: int) -> tuple[Decimal, ...]:
    high = Decimal(mean) + Decimal(deviation)
    low = Decimal(mean) - Decimal(deviation)
    return tuple(high if index % 2 == 0 else low for index in range(count))


def test_sharpe_of_a_two_point_series() -> None:
    series = alternating("0.01", "0.02", 40)
    ratio = sharpe_ratio(series)
    assert abs(ratio - Decimal("0.5")) < Decimal("0.02")
    annual = annualised_sharpe(series)
    assert abs(annual - ratio * Decimal(52).sqrt()) < Decimal("0.000001")


def test_zero_variance_is_rejected() -> None:
    with pytest.raises(PanelStatisticsError):
        sharpe_ratio((Decimal("0.01"),) * 10)


def test_short_series_is_rejected() -> None:
    with pytest.raises(PanelStatisticsError):
        sharpe_ratio((Decimal("0.01"),))


def test_deflated_sharpe_is_a_probability_and_falls_with_trial_dispersion() -> None:
    series = alternating("0.01", "0.02", 200)
    observed = sharpe_ratio(series)
    tight = deflated_sharpe_ratio(
        series, trial_sharpes=(observed, observed - Decimal("0.01"), observed + Decimal("0.01"))
    )
    wide = deflated_sharpe_ratio(
        series, trial_sharpes=(observed, observed - Decimal("2"), observed + Decimal("2"))
    )
    assert Decimal(0) <= wide < tight <= Decimal(1)


def test_deflated_sharpe_needs_at_least_two_trials() -> None:
    series = alternating("0.01", "0.02", 50)
    with pytest.raises(PanelStatisticsError):
        deflated_sharpe_ratio(series, trial_sharpes=(Decimal("0.5"),))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_panel_statistics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading_bot.panel_statistics'`

- [ ] **Step 3: Write the module**

```python
# src/trading_bot/panel_statistics.py
"""Sharpe ratios and the multiple-testing-aware deflated Sharpe ratio."""

import math
from decimal import Decimal
from statistics import NormalDist

_EULER_MASCHERONI = 0.5772156649015329


class PanelStatisticsError(ValueError):
    """Raised when a statistic is not defined for the given series."""


def sharpe_ratio(net_returns: tuple[Decimal, ...]) -> Decimal:
    """Return the per-period Sharpe ratio with a zero risk-free rate."""
    mean, deviation = _mean_and_deviation(net_returns)
    return mean / deviation


def annualised_sharpe(
    net_returns: tuple[Decimal, ...], *, periods_per_year: int = 52
) -> Decimal:
    if periods_per_year < 1:
        raise PanelStatisticsError("periods per year must be positive")
    return sharpe_ratio(net_returns) * Decimal(periods_per_year).sqrt()


def deflated_sharpe_ratio(
    net_returns: tuple[Decimal, ...], *, trial_sharpes: tuple[Decimal, ...]
) -> Decimal:
    """Return the probability that the observed Sharpe exceeds the trial maximum.

    Bailey and Lopez de Prado (2014). Floats are used only inside this function
    because the normal distribution has no exact Decimal form; the result is
    reported evidence and never a gate input.
    """
    if len(trial_sharpes) < 2:
        raise PanelStatisticsError("the deflated Sharpe ratio needs at least two trials")
    observations = len(net_returns)
    if observations < 3:
        raise PanelStatisticsError("the deflated Sharpe ratio needs at least three observations")
    _, _ = _mean_and_deviation(net_returns)

    values = [float(value) for value in net_returns]
    mean = sum(values) / observations
    variance = sum((value - mean) ** 2 for value in values) / (observations - 1)
    deviation = math.sqrt(variance)
    skewness = sum(((value - mean) / deviation) ** 3 for value in values) / observations
    kurtosis = sum(((value - mean) / deviation) ** 4 for value in values) / observations
    observed = mean / deviation

    trials = [float(value) for value in trial_sharpes]
    trial_mean = sum(trials) / len(trials)
    trial_variance = sum((value - trial_mean) ** 2 for value in trials) / (len(trials) - 1)
    trial_deviation = math.sqrt(trial_variance)

    normal = NormalDist()
    count = len(trials)
    expected_maximum = trial_deviation * (
        (1.0 - _EULER_MASCHERONI) * normal.inv_cdf(1.0 - 1.0 / count)
        + _EULER_MASCHERONI * normal.inv_cdf(1.0 - 1.0 / (count * math.e))
    )
    denominator = 1.0 - skewness * observed + (kurtosis - 1.0) / 4.0 * observed**2
    if denominator <= 0.0:
        raise PanelStatisticsError("the deflated Sharpe denominator is not positive")
    statistic = (observed - expected_maximum) * math.sqrt(observations - 1) / math.sqrt(
        denominator
    )
    return Decimal(str(normal.cdf(statistic)))


def _mean_and_deviation(net_returns: tuple[Decimal, ...]) -> tuple[Decimal, Decimal]:
    count = len(net_returns)
    if count < 2:
        raise PanelStatisticsError("at least two observations are required")
    mean = sum(net_returns, Decimal(0)) / Decimal(count)
    variance = sum(((value - mean) ** 2 for value in net_returns), Decimal(0)) / Decimal(
        count - 1
    )
    if variance <= 0:
        raise PanelStatisticsError("a zero-variance series has no Sharpe ratio")
    return mean, variance.sqrt()
```

- [ ] **Step 4: Run tests, lint and types**

Run:
```powershell
$env:TEMP='C:\Users\User\AppData\Local\Temp'; $env:TMP=$env:TEMP
uv run pytest tests/test_panel_statistics.py -v
uv run ruff check src/trading_bot/panel_statistics.py tests/test_panel_statistics.py
uv run mypy
```
Expected: 5 passed, clean.

- [ ] **Step 5: Commit**

```bash
git add src/trading_bot/panel_statistics.py tests/test_panel_statistics.py
git commit -m "feat: add annualised and deflated sharpe ratios

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: Decision report and gates

**Files:**
- Create: `src/trading_bot/panel_decision.py`
- Test: `tests/test_panel_decision.py`
- Modify: `src/trading_bot/cli.py` (add the `panel-decision` subcommand)

**Interfaces:**
- Consumes: `evaluation.block_bootstrap_mean_test`, `benjamini_hochberg`; `panel_statistics`; `panel_config`; `panel_fold_run.verify_panel_fold_report`.
- Produces: `PanelDecisionError`; `PanelDecisionArtifact(output_path, report_hash, decision_status, eligible_member_names)`; `build_panel_decision(fold_report_paths, *, family_spec_path, output_path, registry_path) -> PanelDecisionArtifact`.

Gate order per member, evaluated on the pooled out-of-sample episode series:

| Check | Reason code on failure | Status |
| --- | --- | --- |
| pooled episodes ≥ `pooled_episode_floor` | `EPISODE_FLOOR_NOT_MET` | `insufficient_evidence` |
| pooled base mean > 0 | `AGGREGATE_BASE_NET_NON_POSITIVE` | `rejected` |
| base bootstrap lower bound > 0 | `BASE_LOWER_BOUND_NON_POSITIVE` | `rejected` |
| pooled adverse mean ≥ 0 | `AGGREGATE_ADVERSE_NET_NON_POSITIVE` | `rejected` |
| positive base folds ≥ `(2·folds + 2) // 3` | `POSITIVE_FOLD_FRACTION_NOT_MET` | `rejected` |
| BH q ≤ 0.10 across the six members | `MULTIPLE_TESTING_GATE_NOT_MET` | `rejected` |
| no fold, contract or episode above 50% of pooled base net PnL | `CONCENTRATION_LIMIT_EXCEEDED` | `rejected` |
| base and adverse totals above the strongest control | `BASE_CONTROL_DOMINANCE_NOT_MET`, `ADVERSE_CONTROL_DOMINANCE_NOT_MET` | `rejected` |

An evidence failure outranks an economic one, matching `fold_evaluation.py:208-236`. Controls (`no_trade`, `random_ranks`) supply the dominance benchmark and never enter the BH family; `passive_long_ew` is reported as context only. Concentration shares are computed on pooled **base** net contributions and are only meaningful when the pooled base total is positive; when it is not, the check is skipped because the member is already rejected.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_panel_decision.py
import json
from decimal import Decimal
from pathlib import Path

import pytest

from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.panel_config import load_panel_family_spec
from trading_bot.panel_decision import PanelDecisionError, build_panel_decision

SPEC_PATH = Path("configs/xs-momentum-panel-v1.json")
SPEC, SPEC_HASH = load_panel_family_spec(SPEC_PATH)
MEMBERS = tuple(item.name for item in SPEC.members)
CONTROLS = tuple(item.name for item in SPEC.controls)
EPISODES_PER_FOLD = 40


def episodes(sample_prefix: str, value: str, count: int) -> list[dict[str, object]]:
    return [
        {
            "sample_id": f"{sample_prefix}:{index}",
            "net_return": value,
            "gross_return": value,
            "turnover": "1",
            "trading_cost": "0",
            "funding_cost": "0",
            "forced_close_cost": "0",
            "gross_exposure": "1",
            "net_exposure": "0",
            "forced_close_count": 0,
            "contract_net_contributions": [
                ["A:0", str(Decimal(value) / 2)],
                ["B:0", str(Decimal(value) / 2)],
            ],
        }
        for index in range(count)
    ]


def write_fold(path: Path, fold_index: int, returns: dict[str, tuple[str, str]]) -> None:
    candidates = []
    for name in MEMBERS + CONTROLS:
        base_value, adverse_value = returns.get(name, ("0", "0"))
        candidates.append(
            {
                "candidate_name": name,
                "role": "member" if name in MEMBERS else "control",
                "episode_count": EPISODES_PER_FOLD,
                "base": {
                    "total_net_return": str(
                        Decimal(base_value) * Decimal(EPISODES_PER_FOLD)
                    ),
                    "episodes": episodes(f"f{fold_index}", base_value, EPISODES_PER_FOLD),
                },
                "adverse": {
                    "total_net_return": str(
                        Decimal(adverse_value) * Decimal(EPISODES_PER_FOLD)
                    ),
                    "episodes": episodes(f"f{fold_index}", adverse_value, EPISODES_PER_FOLD),
                },
            }
        )
    material = {
        "report_version": "1.0.0",
        "status": "development_only",
        "reason_codes": [],
        "family_name": SPEC.family_name,
        "family_spec_hash": SPEC_HASH,
        "capture_root_hash": "c" * 64,
        "dataset_root_hash": "d" * 64,
        "split_manifest_hash": "e" * 64,
        "manifest_hash": "f" * 64,
        "fold_index": fold_index,
        "fold_count": 6,
        "train_sample_count": 10,
        "validation_sample_count": 10,
        "test_sample_count": EPISODES_PER_FOLD,
        "train_membership_hash": "0" * 64,
        "validation_membership_hash": "1" * 64,
        "test_membership_hash": "2" * 64,
        "random_seed": 17,
        "block_length": 4,
        "bootstrap_repetitions": 2000,
        "skipped_sample_ids": [],
        "code_hash": "3" * 64,
        "candidates": candidates,
    }
    document = dict(material)
    document["report_hash"] = content_sha256(material)
    path.write_bytes(canonical_json(document))


def build(tmp_path: Path, returns: dict[str, tuple[str, str]]) -> dict[str, object]:
    paths = []
    for fold_index in range(6):
        path = tmp_path / f"fold{fold_index}.json"
        write_fold(path, fold_index, returns)
        paths.append(path)
    artifact = build_panel_decision(
        tuple(paths),
        family_spec_path=SPEC_PATH,
        output_path=tmp_path / "decision.json",
        registry_path=tmp_path / "registry.sqlite3",
    )
    return json.loads(artifact.output_path.read_text(encoding="utf-8"))


def test_a_flat_family_is_rejected_without_eligible_members(tmp_path: Path) -> None:
    document = build(tmp_path, {})
    assert document["decision_status"] == "no_eligible_member"
    momentum = next(
        item for item in document["members"] if item["candidate_name"] == "xs_mom_4w"
    )
    assert "AGGREGATE_BASE_NET_NON_POSITIVE" in momentum["reason_codes"]


def test_negative_adverse_is_rejected(tmp_path: Path) -> None:
    document = build(tmp_path, {"xs_mom_4w": ("0.01", "-0.001")})
    momentum = next(
        item for item in document["members"] if item["candidate_name"] == "xs_mom_4w"
    )
    assert momentum["decision_status"] == "rejected"
    assert "AGGREGATE_ADVERSE_NET_NON_POSITIVE" in momentum["reason_codes"]


def test_pooled_counts_and_hashes_are_recorded(tmp_path: Path) -> None:
    document = build(tmp_path, {})
    assert document["pooled_episode_count"] == 6 * EPISODES_PER_FOLD
    assert document["fold_count"] == 6
    assert len(document["source_report_hashes"]) == 6
    assert document["family_spec_hash"] == SPEC_HASH


def test_duplicate_fold_indices_are_rejected(tmp_path: Path) -> None:
    write_fold(tmp_path / "a.json", 0, {})
    write_fold(tmp_path / "b.json", 0, {})
    with pytest.raises(PanelDecisionError):
        build_panel_decision(
            (tmp_path / "a.json", tmp_path / "b.json"),
            family_spec_path=SPEC_PATH,
            output_path=tmp_path / "decision.json",
            registry_path=tmp_path / "registry.sqlite3",
        )


def test_tampered_report_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "fold0.json"
    write_fold(path, 0, {})
    document = json.loads(path.read_text(encoding="utf-8"))
    document["fold_index"] = 1
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(PanelDecisionError):
        build_panel_decision(
            (path,),
            family_spec_path=SPEC_PATH,
            output_path=tmp_path / "decision.json",
            registry_path=tmp_path / "registry.sqlite3",
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_panel_decision.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading_bot.panel_decision'`

- [ ] **Step 3: Write the module**

```python
# src/trading_bot/panel_decision.py
"""Pooled gates and the decision report for the panel family."""

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.evaluation import benjamini_hochberg, block_bootstrap_mean_test
from trading_bot.panel_config import PanelFamilySpec, load_panel_family_spec
from trading_bot.panel_fold_run import verify_panel_fold_report
from trading_bot.panel_statistics import (
    PanelStatisticsError,
    annualised_sharpe,
    deflated_sharpe_ratio,
    sharpe_ratio,
)


class PanelDecisionError(RuntimeError):
    """Raised when the panel decision cannot be derived or published."""


@dataclass(frozen=True, slots=True)
class PanelDecisionArtifact:
    output_path: Path
    report_hash: str
    decision_status: str
    eligible_member_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Pooled:
    base_returns: tuple[Decimal, ...]
    adverse_returns: tuple[Decimal, ...]
    base_total: Decimal
    adverse_total: Decimal
    fold_base_totals: tuple[Decimal, ...]
    contract_totals: dict[str, Decimal]
    episode_count: int


def build_panel_decision(
    fold_report_paths: tuple[Path, ...],
    *,
    family_spec_path: Path,
    output_path: Path,
    registry_path: Path,
) -> PanelDecisionArtifact:
    if not fold_report_paths:
        raise PanelDecisionError("at least one fold report is required")
    if output_path.exists():
        raise PanelDecisionError("panel decision already exists and is immutable")
    spec, family_spec_hash = load_panel_family_spec(family_spec_path)

    documents = []
    for path in fold_report_paths:
        if not verify_panel_fold_report(path):
            raise PanelDecisionError(f"fold report failed verification: {path.name}")
        documents.append(_load_object(path))
    indices = [int(document["fold_index"]) for document in documents]
    if len(set(indices)) != len(indices):
        raise PanelDecisionError("fold reports must have distinct fold indices")
    linkage = {
        (
            str(document["family_spec_hash"]),
            str(document["split_manifest_hash"]),
            str(document["dataset_root_hash"]),
        )
        for document in documents
    }
    if len(linkage) != 1:
        raise PanelDecisionError("fold reports do not share one family and manifest")
    if next(iter(linkage))[0] != family_spec_hash:
        raise PanelDecisionError("fold reports were produced under a different declaration")
    documents.sort(key=lambda item: int(item["fold_index"]))

    member_names = tuple(item.name for item in spec.members)
    control_names = tuple(item.name for item in spec.controls)
    pooled = {
        name: _pool(documents, name) for name in member_names + control_names
    }
    fold_count = len(documents)

    tests = {}
    for name in member_names:
        series = pooled[name].base_returns
        if len(series) < spec.statistics.block_length:
            continue
        try:
            tests[name] = block_bootstrap_mean_test(
                series,
                block_length=spec.statistics.block_length,
                repetitions=spec.statistics.bootstrap_repetitions,
                seed=spec.statistics.random_seed,
                confidence=spec.statistics.confidence,
            )
        except ValueError as error:
            raise PanelDecisionError(f"bootstrap failed for {name}: {error}") from error
    q_values = (
        benjamini_hochberg({name: test.one_sided_p_value for name, test in tests.items()})
        if tests
        else {}
    )

    trial_sharpes: list[Decimal] = []
    for name in member_names:
        try:
            trial_sharpes.append(sharpe_ratio(pooled[name].base_returns))
        except PanelStatisticsError:
            trial_sharpes.append(Decimal(0))

    strongest_base = max(
        (pooled[name].base_total for name in ("no_trade", "random_ranks")), default=Decimal(0)
    )
    strongest_adverse = max(
        (pooled[name].adverse_total for name in ("no_trade", "random_ranks")),
        default=Decimal(0),
    )

    members: list[dict[str, object]] = []
    eligible: list[str] = []
    for name in member_names:
        record, status = _member_record(
            name,
            pooled[name],
            spec=spec,
            fold_count=fold_count,
            test=tests.get(name),
            q_value=q_values.get(name),
            trial_sharpes=tuple(trial_sharpes),
            strongest_base=strongest_base,
            strongest_adverse=strongest_adverse,
        )
        members.append(record)
        if status == "eligible_for_further_review":
            eligible.append(name)

    controls = [
        {
            "candidate_name": name,
            "episode_count": pooled[name].episode_count,
            "base_total_net_return": pooled[name].base_total,
            "adverse_total_net_return": pooled[name].adverse_total,
        }
        for name in control_names
    ]

    decision_status = "eligible_member_available" if eligible else "no_eligible_member"
    material: dict[str, object] = {
        "decision_version": "1.0.0",
        "status": "development_only",
        "family_name": spec.family_name,
        "family_spec_hash": family_spec_hash,
        "capture_root_hash": str(documents[0]["capture_root_hash"]),
        "dataset_root_hash": str(documents[0]["dataset_root_hash"]),
        "split_manifest_hash": str(documents[0]["split_manifest_hash"]),
        "fold_count": fold_count,
        "pooled_episode_count": pooled[member_names[0]].episode_count,
        "source_report_hashes": [str(document["report_hash"]) for document in documents],
        "skipped_sample_ids": sorted(
            {value for document in documents for value in document["skipped_sample_ids"]}
        ),
        "block_length": spec.statistics.block_length,
        "bootstrap_repetitions": spec.statistics.bootstrap_repetitions,
        "random_seed": spec.statistics.random_seed,
        "members": members,
        "controls": controls,
        "eligible_member_names": eligible,
        "decision_status": decision_status,
    }
    report_hash = content_sha256(material)
    document = dict(material)
    document["report_hash"] = report_hash
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_bytes(canonical_json(document))
    temporary.replace(output_path)
    _ = registry_path
    return PanelDecisionArtifact(
        output_path=output_path,
        report_hash=report_hash,
        decision_status=decision_status,
        eligible_member_names=tuple(eligible),
    )


def _pool(documents: list[dict[str, object]], name: str) -> _Pooled:
    base: list[Decimal] = []
    adverse: list[Decimal] = []
    fold_totals: list[Decimal] = []
    contracts: dict[str, Decimal] = {}
    for document in documents:
        candidates = document["candidates"]
        if not isinstance(candidates, list):
            raise PanelDecisionError("fold report candidates are malformed")
        record = next(
            (item for item in candidates if item.get("candidate_name") == name), None
        )
        if record is None:
            raise PanelDecisionError(f"fold report is missing candidate {name}")
        fold_total = Decimal(0)
        for episode in record["base"]["episodes"]:
            value = Decimal(str(episode["net_return"]))
            base.append(value)
            fold_total += value
            for contract_id, contribution in episode["contract_net_contributions"]:
                contracts[str(contract_id)] = contracts.get(
                    str(contract_id), Decimal(0)
                ) + Decimal(str(contribution))
        fold_totals.append(fold_total)
        for episode in record["adverse"]["episodes"]:
            adverse.append(Decimal(str(episode["net_return"])))
    return _Pooled(
        base_returns=tuple(base),
        adverse_returns=tuple(adverse),
        base_total=sum(base, Decimal(0)),
        adverse_total=sum(adverse, Decimal(0)),
        fold_base_totals=tuple(fold_totals),
        contract_totals=contracts,
        episode_count=len(base),
    )


def _member_record(
    name: str,
    pooled: _Pooled,
    *,
    spec: PanelFamilySpec,
    fold_count: int,
    test: object,
    q_value: Decimal | None,
    trial_sharpes: tuple[Decimal, ...],
    strongest_base: Decimal,
    strongest_adverse: Decimal,
) -> tuple[dict[str, object], str]:
    evidence: list[str] = []
    economic: list[str] = []
    if pooled.episode_count < spec.statistics.pooled_episode_floor:
        evidence.append("EPISODE_FLOOR_NOT_MET")
    if test is None:
        evidence.append("BOOTSTRAP_NOT_AVAILABLE")

    base_mean = (
        pooled.base_total / Decimal(pooled.episode_count)
        if pooled.episode_count
        else Decimal(0)
    )
    adverse_mean = (
        pooled.adverse_total / Decimal(len(pooled.adverse_returns))
        if pooled.adverse_returns
        else Decimal(0)
    )
    lower = getattr(getattr(test, "interval", None), "lower", None)
    if base_mean <= 0:
        economic.append("AGGREGATE_BASE_NET_NON_POSITIVE")
    if lower is not None and lower <= 0:
        economic.append("BASE_LOWER_BOUND_NON_POSITIVE")
    if adverse_mean < 0:
        economic.append("AGGREGATE_ADVERSE_NET_NON_POSITIVE")
    required = (
        spec.statistics.positive_fold_numerator * fold_count
        + spec.statistics.positive_fold_denominator
        - 1
    ) // spec.statistics.positive_fold_denominator
    positive_folds = sum(1 for value in pooled.fold_base_totals if value > 0)
    if positive_folds < required:
        economic.append("POSITIVE_FOLD_FRACTION_NOT_MET")
    if q_value is not None and q_value > spec.statistics.false_discovery_gate:
        economic.append("MULTIPLE_TESTING_GATE_NOT_MET")

    shares: dict[str, Decimal] = {}
    if pooled.base_total > 0:
        limit = spec.statistics.concentration_limit
        shares = {
            "largest_fold_share": max(pooled.fold_base_totals, default=Decimal(0))
            / pooled.base_total,
            "largest_contract_share": max(
                pooled.contract_totals.values(), default=Decimal(0)
            )
            / pooled.base_total,
            "largest_episode_share": max(pooled.base_returns, default=Decimal(0))
            / pooled.base_total,
        }
        if any(value > limit for value in shares.values()):
            economic.append("CONCENTRATION_LIMIT_EXCEEDED")
    if pooled.base_total <= strongest_base:
        economic.append("BASE_CONTROL_DOMINANCE_NOT_MET")
    if pooled.adverse_total <= strongest_adverse:
        economic.append("ADVERSE_CONTROL_DOMINANCE_NOT_MET")

    if evidence:
        status = "insufficient_evidence"
    elif economic:
        status = "rejected"
    else:
        status = "eligible_for_further_review"

    try:
        annual = annualised_sharpe(pooled.base_returns)
        deflated = deflated_sharpe_ratio(pooled.base_returns, trial_sharpes=trial_sharpes)
    except PanelStatisticsError:
        annual = Decimal(0)
        deflated = Decimal(0)

    record: dict[str, object] = {
        "candidate_name": name,
        "episode_count": pooled.episode_count,
        "base_total_net_return": pooled.base_total,
        "base_mean_net_return": base_mean,
        "adverse_total_net_return": pooled.adverse_total,
        "adverse_mean_net_return": adverse_mean,
        "positive_base_fold_count": positive_folds,
        "required_positive_fold_count": required,
        "base_bootstrap_lower": lower,
        "base_bootstrap_p_value": getattr(test, "one_sided_p_value", None),
        "bh_q_value": q_value,
        "annualised_sharpe": annual,
        "deflated_sharpe_ratio": deflated,
        "concentration": shares,
        "decision_status": status,
        "reason_codes": evidence + economic,
    }
    return record, status


def _load_object(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise PanelDecisionError(f"{path.name} must contain a JSON object")
    return document
```

- [ ] **Step 4: Add the CLI subcommand**

Parser:

```python
    panel_decision = commands.add_parser("panel-decision")
    panel_decision.add_argument("--workspace-root", type=Path, default=Path.cwd())
    panel_decision.add_argument("--fold-report", type=Path, action="append", required=True)
    panel_decision.add_argument("--family-spec", type=Path, required=True)
    panel_decision.add_argument("--output", type=Path, required=True)
    panel_decision.add_argument("--registry", type=Path, required=True)
```

Dispatch:

```python
    if parsed.command == "panel-decision":
        workspace = parsed.workspace_root.resolve()
        reports = tuple(path.resolve() for path in parsed.fold_report)
        paths = (*reports, parsed.family_spec.resolve(), parsed.output.resolve(),
                 parsed.registry.resolve())
        if any(not path.is_relative_to(workspace) for path in paths):
            raise ValueError("panel decision paths must stay inside workspace")
        build_panel_decision(
            reports,
            family_spec_path=parsed.family_spec.resolve(),
            output_path=parsed.output.resolve(),
            registry_path=parsed.registry.resolve(),
        )
        return 0
```

with `from trading_bot.panel_decision import build_panel_decision`.

- [ ] **Step 5: Run tests, lint and types**

Run:
```powershell
$env:TEMP='C:\Users\User\AppData\Local\Temp'; $env:TMP=$env:TEMP
uv run pytest tests/test_panel_decision.py -v
uv run ruff check src/trading_bot/panel_decision.py src/trading_bot/cli.py tests/test_panel_decision.py
uv run mypy
```
Expected: 5 passed, clean.

- [ ] **Step 6: Commit**

```bash
git add src/trading_bot/panel_decision.py src/trading_bot/cli.py tests/test_panel_decision.py
git commit -m "feat: derive the panel family decision under the frozen gates

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: Backlog entry, README section and the end-to-end check

**Files:**
- Modify: `PHASE_1_BACKLOG.md` (add `### P1.27` after the `### P1.26 Paper observation window` block)
- Modify: `README.md` (add a panel research section next to the P1.15 paragraph around line 206)
- Create: `tests/test_panel_end_to_end.py`

**Interfaces:**
- Consumes: the CLI subcommands from Tasks 3, 8, 9 and 11.
- Produces: no new production interface. This task proves the four commands compose.

- [ ] **Step 1: Write the failing end-to-end test**

```python
# tests/test_panel_end_to_end.py
import json
from pathlib import Path
from unittest.mock import patch

from tests.test_panel_fold_run import MONTHS, SYMBOLS, fetch, small_config

from trading_bot.cli import main


def test_capture_manifest_folds_and_decision_compose(tmp_path: Path) -> None:
    config_path = small_config(tmp_path)
    with patch("trading_bot.panel_capture.PanelZipClient.fetch", side_effect=fetch):
        assert (
            main(
                [
                    "panel-capture",
                    "--workspace-root",
                    str(tmp_path),
                    "--output",
                    str(tmp_path / "capture"),
                    "--reserve-bytes",
                    "0",
                    "--symbols",
                    ",".join(SYMBOLS),
                    "--months",
                    ",".join(MONTHS),
                ]
            )
            == 0
        )
    assert (
        main(
            [
                "panel-manifest",
                "--workspace-root",
                str(tmp_path),
                "--capture",
                str(tmp_path / "capture"),
                "--output",
                str(tmp_path / "manifest.json"),
                "--family-spec",
                str(config_path),
            ]
        )
        == 0
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    reports = []
    for fold in manifest["folds"]:
        output = tmp_path / f"fold{fold['fold_index']}.json"
        assert (
            main(
                [
                    "panel-fold",
                    "--workspace-root",
                    str(tmp_path),
                    "--capture",
                    str(tmp_path / "capture"),
                    "--manifest",
                    str(tmp_path / "manifest.json"),
                    "--family-spec",
                    str(config_path),
                    "--output",
                    str(output),
                    "--registry",
                    str(tmp_path / "registry.sqlite3"),
                    "--fold-index",
                    str(fold["fold_index"]),
                ]
            )
            == 0
        )
        reports.append(output)
    arguments = [
        "panel-decision",
        "--workspace-root",
        str(tmp_path),
        "--family-spec",
        str(config_path),
        "--output",
        str(tmp_path / "decision.json"),
        "--registry",
        str(tmp_path / "registry.sqlite3"),
    ]
    for report in reports:
        arguments.extend(["--fold-report", str(report)])
    assert main(arguments) == 0
    decision = json.loads((tmp_path / "decision.json").read_text(encoding="utf-8"))
    assert decision["decision_status"] in {"eligible_member_available", "no_eligible_member"}
    assert decision["fold_count"] == len(reports)
    assert len(decision["members"]) == 6
    assert decision["status"] == "development_only"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_panel_end_to_end.py -v`
Expected: FAIL until the CLI wiring of Tasks 3, 8, 9 and 11 is complete; if those tasks are done, this passes on the first run and the step is a confirmation rather than a red test.

- [ ] **Step 3: Add the backlog entry**

Append to `PHASE_1_BACKLOG.md` after the `### P1.26 Paper observation window` block:

```markdown
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
```

- [ ] **Step 4: Add the README section**

Insert after the existing P1.15 paragraph in `README.md`:

````markdown
## Panel-Forschung (P1.27)

Neben der 15-Minuten-BTC-Linie gibt es eine zweite, davon unabhängige Linie:
ein Wochen-Rebalancing auf einem Panel von Binance-USD-M-USDT-Perpetuals aus
den öffentlichen Tagesdumps. Ein Sample ist ein Rebalance-Termin, nicht ein
Coin-Tag; die Kontrakte innerhalb einer Woche sind keine unabhängigen Samples
(Protokoll Abschnitt 16). Vier Befehle bilden die Kette:

```powershell
uv run trading-research panel-capture --output data/captures/<datum>-binance-um-usdt-perps-1d --symbols <liste> --months <liste>
uv run trading-research panel-manifest --capture <capture> --output artifacts/panel-walk-forward-v1.json --family-spec configs/xs-momentum-panel-v1.json
uv run trading-research panel-fold --capture <capture> --manifest <manifest> --family-spec configs/xs-momentum-panel-v1.json --output artifacts/panel/fold0.json --registry artifacts/panel/metadata-xs-momentum-v1.sqlite3 --fold-index 0
uv run trading-research panel-decision --fold-report artifacts/panel/fold0.json --family-spec configs/xs-momentum-panel-v1.json --output artifacts/panel/decision-v1.json --registry artifacts/panel/metadata-xs-momentum-v1.sqlite3
```

Die Familie ist vor dem ersten Datenabruf eingefroren: sechs Mitglieder, drei
Kontrollen, keine Parametersuche, alles in `configs/xs-momentum-panel-v1.json`,
dessen SHA-256 in jedem Report als `family_spec_hash` steht.
````

- [ ] **Step 5: Run the whole suite, lint and types**

Run:
```powershell
$env:TEMP='C:\Users\User\AppData\Local\Temp'; $env:TMP=$env:TEMP
uv run pytest -q
uv run ruff check .
uv run mypy
```
Expected: every pre-existing test still passes (580 before this plan) plus the new panel tests; no lint findings; no type errors. If the count of pre-existing passes changed, stop and investigate before committing.

- [ ] **Step 6: Commit**

```bash
git add PHASE_1_BACKLOG.md README.md tests/test_panel_end_to_end.py
git commit -m "docs: register p1.27 and document the panel research line

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## After the plan

The code is then ready to run against real data. That run is **not** part of this plan and follows spec section 12 in order: capture, quality stop condition, manifest with the episode-floor stop, all folds, decision report, and finally a root-level `P1_27_DECISION_<date>.md` in the template of `P1_15_DECISION_2026-08-25.md` with the gate table extended by turnover and deflated-Sharpe columns.

Two things must not happen after seeing results: changing the universe rules, lookbacks or cost table, and opening the final holdout. Either one voids the family and requires a new declaration under a new name.

## Self-review

**Spec coverage.** Section 5 (declaration) is Task 1; section 6 (data and capture) Tasks 2 and 3; section 7 (universe and identity) Tasks 4 and 5; section 8.1 (weights) Task 6; sections 8.2 to 8.4 (returns, costs, reporting) Task 7; section 9.1 (folds) Task 8; section 9.2 (episodes and dependence) Tasks 7 and 8; section 9.3 (statistics) Tasks 10 and 11; section 9.4 (deflated Sharpe) Task 10; section 9.5 (eligibility) Task 11; section 10 (modules and interfaces) the file structure, with the three documented refinements; section 11 (artifacts and naming) Tasks 8, 9, 11 and 12; section 12 (sequencing) the closing note; section 13 (protocol addendum) already committed as protocol section 16.

**Deliberate deviations, each stated where it occurs.** The spec's `EpisodeResult` gains `contract_net_contributions`, without which the concentration gate cannot be computed on net PnL. `panel_capture` carries its own host allowlist rather than widening `market_capture._ALLOWED_HOSTS`, which keeps the 15-minute line untouched. Weeks flagged `UNIVERSE_TOO_SMALL` reset the position to flat without charging the exit, recorded as `SKIPPED_WEEK_EXIT_COST_UNCHARGED`.

**Type consistency.** `PanelCandleRow` and `PanelFundingRow` are defined in Task 2 and consumed in Tasks 3 and 4. `PanelBar` and `FundingEvent` come from Task 4 and are consumed in Tasks 5, 6, 7 and 9. `ContractHistory` is defined in Task 5 and used in Tasks 6, 7 and 9. `WeightVector.weights` is `tuple[tuple[str, Decimal], ...]` everywhere. `EpisodeResult` field names are identical in Tasks 7, 9 and 11. `PanelCostTable` comes from Task 1 and is consumed only by Task 7. `verify_panel_fold_report` is defined in Task 9 and called in Task 11.
