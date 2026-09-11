# Funding Carry Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate the pre-registered family `funding_carry_panel_v1` — long-spot, short-perpetual pairs selected on trailing realised funding and held through overlapping weekly cohorts — through the repository's existing walk-forward manifest, statistics and decision gates.

**Architecture:** The P1.27 panel line already has capture, repair, quality gate, calendar, manifest, accounting, statistics and decision modules. This plan adds a spot market to the capture, four small `carry_*` modules (declaration, pair universe, cohort selection, two-leg accounting), and a carry fold runner that emits the exact fold-report schema `panel_decision.py` already pools. The decision module's logic is untouched; only its family loader learns to dispatch, and its linkage check learns to bind the second capture.

**Tech Stack:** Python 3.12, `Decimal`, `duckdb`, `pyarrow`, `pydantic` v2, `pytest`, `mypy --strict`, `ruff`. No new dependency.

**Spec:** `docs/superpowers/specs/2026-09-11-funding-carry-family-design.md` (approved 2026-09-11, commit `735d52a`).

## Global Constraints

- **Money is `Decimal`, never `float`.** `canonical.py` raises on binary floats. The only float use in the panel line is inside `panel_statistics.deflated_sharpe_ratio`, and it stays there.
- **Time is integer nanoseconds.** `DAY_NS = 86_400_000_000_000`; one week is `604_800_000_000_000`.
- **Artifacts are immutable:** refuse an existing output, write to a `.tmp` sibling, `replace()`. Every report stores its own hash over material excluding that key, via `trading_bot.canonical.content_sha256`.
- **Dataclasses are `@dataclass(frozen=True, slots=True)`; pydantic models use `ConfigDict(frozen=True, extra="forbid")`.**
- **Precision:** two-leg accounting reuses `panel_accounting.evaluate_episode`, whose `_NET_RETURN_CONTEXT` traps `Inexact`. Per-pair net attribution must sum exactly to the episode's net return.
- **No test touches the network.** Every fetch is injected.
- **Frozen numbers** from spec sections 3.3, 5, 7 and 9, identical in code defaults and in `configs/funding-carry-panel-v1.json`: members `carry_l1w_h4w` (lookback 1, hold 4), `carry_l4w_h4w` (4, 4), `carry_l4w_h13w` (4, 13); controls `no_trade`, `random_pairs`, `all_pairs_ew`; decile denominator `10`; minimum selected `8`; universe `minimum_history_days 91`, `liquidity_window_days 30`, `minimum_median_quote_volume 5000000`, `maximum_pairs 100`, `minimum_pairs 40`, `tier_one_rank_limit 20`; perpetual fee `5` bps per side, spot fee `10`; slippage base `5`/`10` by tier, adverse `10`/`20`; base funding multipliers receipt `1` payment `1`; adverse receipt `0.75` payment `2`; forced-close multiplier base `1` adverse `2`; folds identical to P1.27; statistics identical to P1.27 (block length `4`, `2000` repetitions, seed `17`, confidence `0.95`, gate `0.10`, floor `200`, concentration `0.5`, positive folds `2`/`3`).
- **Deviation from the spec's section 10, declared here:** the spec says `panel_decision.py` and `panel_samples.py` are reused unchanged. Their *logic* is unchanged, but both load the family through `PanelFamilySpec`, whose `family_name` is the closed literal `xs_momentum_panel_v1`. Task 1 adds a dispatching loader, `load_family_spec`, and Tasks 6 and 8 switch those two modules to it. The cost table is a new `CarryCostTable` rather than an extension of `PanelCostTable`, so P1.27's declaration model is untouched.
- **Verification command**, with TEMP forced onto C: because the storage policy excludes E: by design:
  ```
  uv run pytest -q --basetemp=C:/Users/User/AppData/Local/Temp/pytest-carry
  ```
  Baseline on `codex/phase1-foundation` at `735d52a`: **309 passed**. `uv run ruff check .` and `uv run mypy` must be clean.
- **Environment warning.** A bare `python -c` can resolve `trading_bot` to the main checkout instead of the worktree. Always run through `uv run` from inside the worktree.
- **Commit trailer**, last line of every commit:
  ```
  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  ```

---

## File Structure

| File | Responsibility |
| --- | --- |
| `configs/funding-carry-panel-v1.json` | frozen declaration incl. the pair mapping and the excluded scaled pairs |
| `src/trading_bot/carry_config.py` | `CarryFamilySpec` and its parts; `CarryCostTable` |
| `src/trading_bot/panel_config.py` (modify) | `load_family_spec` dispatching on `family_name` |
| `src/trading_bot/panel_capture.py` (modify) | `market` parameter: `um` (default) or `spot`; venue, URL prefixes, no funding for spot |
| `src/trading_bot/carry_universe.py` | pair eligibility on both legs, tiering |
| `src/trading_bot/carry_signals.py` | trailing funding, cohort selection, book assembly, controls |
| `src/trading_bot/panel_accounting.py` (modify) | `fee_overrides` per contract id |
| `src/trading_bot/carry_accounting.py` | two-leg episode via `evaluate_episode`, pair-level attribution, extras |
| `src/trading_bot/panel_samples.py` (modify) | optional hedge capture bound in the manifest; `load_family_spec` |
| `src/trading_bot/carry_fold_run.py` | one fold over both captures, P1.27 report schema plus extras |
| `src/trading_bot/panel_decision.py` (modify) | `load_family_spec`; hedge-hash linkage; extras aggregation |
| `src/trading_bot/cli.py` (modify) | `panel-capture --market`, `panel-manifest --hedge-capture`, `carry-fold` |
| `tests/carry_fixtures.py` | shared fake fetches for both markets and the reduced carry declaration |
| `tests/test_carry_*.py`, additions to existing panel tests | one test file per new module |

Data flow: perp capture (exists) + spot capture → manifest binding both → carry fold ×9 → decision → `P1_28_DECISION_<date>.md`.

---

## Task 1: Frozen family declaration and the dispatching loader

**Files:**
- Create: `configs/funding-carry-panel-v1.json`
- Create: `src/trading_bot/carry_config.py`
- Modify: `src/trading_bot/panel_config.py` (add `load_family_spec`)
- Test: `tests/test_carry_config.py`

**Interfaces:**
- Consumes: `trading_bot.panel_config.PanelFoldGeometry`, `PanelStatistics`, `_Frozen`, `content_sha256`.
- Produces: `CarryMember(name, lookback_weeks, hold_weeks)`, `CarryControl(name, kind)`, `CarryPair(perpetual, spot, multiplier)`, `CarryUniverseRules`, `CarrySelectionRules`, `CarryCostTable`, `CarryCosts`, `CarryFamilySpec`; constants `MEMBER_NAMES`, `CONTROL_NAMES`; `load_carry_family_spec(path) -> tuple[CarryFamilySpec, str]`; and in `panel_config`, `load_family_spec(path) -> tuple[PanelFamilySpec | CarryFamilySpec, str]`.

- [ ] **Step 1: Generate the pair mapping from the bucket, once, outside the tests**

This is a one-off, network-touching build step, not a test. Save the script in the session scratchpad (outside the repository), run it from the worktree root, and commit only its output. It writes the declaration with the pair list; the scalar values are the frozen numbers from the Global Constraints. A perpetual whose own symbol exists on spot is a direct pair even if its name starts with `1000`; only a perpetual without a same-named spot symbol is tried as a scaled pair.

```python
# <scratchpad>/build_carry_declaration.py  (not committed; its output is)
import json, re, sys, urllib.parse, urllib.request
from pathlib import Path

BASE = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"

def prefixes(prefix: str) -> set[str]:
    out, token = set(), None
    while True:
        q = {"delimiter": "/", "prefix": prefix, "max-keys": "1000", "list-type": "2"}
        if token:
            q["continuation-token"] = token
        with urllib.request.urlopen(BASE + "?" + urllib.parse.urlencode(q), timeout=60) as r:
            xml = r.read().decode()
        out |= set(re.findall(r"<Prefix>" + re.escape(prefix) + r"([^/]+)/</Prefix>", xml))
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if not m or "<IsTruncated>false" in xml:
            return out
        token = m.group(1)

perps = sorted(s for s in prefixes("data/futures/um/monthly/klines/") if re.fullmatch(r"[A-Z0-9]+USDT", s))
spot = prefixes("data/spot/monthly/klines/")
pairs, excluded = [], []
for p in perps:
    if p in spot:
        pairs.append({"perpetual": p, "spot": p, "multiplier": 1})
        continue
    m = re.match(r"^(1000000|1000)([A-Z0-9]+USDT)$", p)
    if m and m.group(2) in spot:
        excluded.append({"perpetual": p, "spot": m.group(2), "multiplier": int(m.group(1))})

document = {
  "spec_version": "1.0.0",
  "family_name": "funding_carry_panel_v1",
  "hypothesis": "Among pairs of a USDT perpetual and its spot underlying that clear a liquidity floor on both legs, the pairs with the highest realised funding over a trailing lookback continue to pay funding over the following holding period in excess of both legs' execution costs and basis drift, with positive net expectancy under base costs and non-negative under adverse costs, not concentrated in one fold, one pair, or one week.",
  "perpetual_venue": "BINANCE_UM",
  "spot_venue": "BINANCE_SPOT",
  "holding_days": 7,
  "members": [
    {"name": "carry_l1w_h4w", "lookback_weeks": 1, "hold_weeks": 4},
    {"name": "carry_l4w_h4w", "lookback_weeks": 4, "hold_weeks": 4},
    {"name": "carry_l4w_h13w", "lookback_weeks": 4, "hold_weeks": 13}
  ],
  "controls": [
    {"name": "no_trade", "kind": "no_trade"},
    {"name": "random_pairs", "kind": "random_pairs"},
    {"name": "all_pairs_ew", "kind": "all_pairs"}
  ],
  "pairs": pairs,
  "excluded_pairs": excluded,
  "universe": {
    "minimum_history_days": 91, "liquidity_window_days": 30,
    "minimum_median_quote_volume": "5000000", "maximum_pairs": 100,
    "minimum_pairs": 40, "tier_one_rank_limit": 20
  },
  "selection": {"decile_denominator": 10, "minimum_selected": 8},
  "costs": {
    "base": {"name": "base", "perpetual_fee_bps_per_side": "5", "spot_fee_bps_per_side": "10",
             "slippage_bps_per_side_tier_one": "5", "slippage_bps_per_side_tier_two": "10",
             "funding_receipt_multiplier": "1", "funding_payment_multiplier": "1",
             "forced_close_multiplier": "1"},
    "adverse": {"name": "adverse", "perpetual_fee_bps_per_side": "5", "spot_fee_bps_per_side": "10",
                "slippage_bps_per_side_tier_one": "10", "slippage_bps_per_side_tier_two": "20",
                "funding_receipt_multiplier": "0.75", "funding_payment_multiplier": "2",
                "forced_close_multiplier": "2"}
  },
  "folds": {"train_duration_ns": 31536000000000000, "validation_duration_ns": 7862400000000000,
            "test_duration_ns": 15724800000000000, "step_ns": 15724800000000000,
            "embargo_ns": 1209600000000000, "holdout_duration_ns": 15724800000000000},
  "statistics": {"block_length": 4, "bootstrap_repetitions": 2000, "random_seed": 17,
                 "confidence": "0.95", "false_discovery_gate": "0.10", "pooled_episode_floor": 200,
                 "concentration_limit": "0.5", "positive_fold_numerator": 2,
                 "positive_fold_denominator": 3}
}
Path("configs/funding-carry-panel-v1.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
sys.stdout.write(f"pairs {len(pairs)} excluded {len(excluded)}\n")
```

Run: `uv run python <scratchpad>/build_carry_declaration.py`
Expected: `pairs 470 excluded 7` (spec section 3.2 as corrected on 2026-09-11). If the counts differ, stop and report; the bucket has changed and the spec's section 3.2 needs updating before the family is frozen.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_carry_config.py
import json
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from trading_bot.carry_config import (
    CONTROL_NAMES,
    MEMBER_NAMES,
    CarryFamilySpec,
    load_carry_family_spec,
)
from trading_bot.panel_config import PanelFamilySpec, load_family_spec

CONFIG = Path("configs/funding-carry-panel-v1.json")


def test_repository_declaration_is_the_frozen_family() -> None:
    spec, spec_hash = load_carry_family_spec(CONFIG)
    assert spec.family_name == "funding_carry_panel_v1"
    assert tuple(m.name for m in spec.members) == MEMBER_NAMES
    assert tuple(c.name for c in spec.controls) == CONTROL_NAMES
    assert [(m.lookback_weeks, m.hold_weeks) for m in spec.members] == [(1, 4), (4, 4), (4, 13)]
    assert spec.costs.base.spot_fee_bps_per_side == Decimal("10")
    assert spec.costs.adverse.funding_receipt_multiplier == Decimal("0.75")
    assert spec.selection.minimum_selected == 8
    assert len(spec.pairs) == 470
    assert len(spec.excluded_pairs) == 7
    assert all(p.multiplier == 1 for p in spec.pairs)
    assert all(p.multiplier > 1 for p in spec.excluded_pairs)
    assert len({p.perpetual for p in spec.pairs}) == len(spec.pairs)
    assert len(spec_hash) == 64


def test_member_set_is_closed_and_ordered(tmp_path: Path) -> None:
    document = json.loads(CONFIG.read_text(encoding="utf-8"))
    document["members"] = [document["members"][1], document["members"][0], document["members"][2]]
    path = tmp_path / "reordered.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_carry_family_spec(path)


def test_scaled_pair_in_the_included_list_is_rejected(tmp_path: Path) -> None:
    document = json.loads(CONFIG.read_text(encoding="utf-8"))
    document["pairs"].append({"perpetual": "1000XYZUSDT", "spot": "XYZUSDT", "multiplier": 1000})
    path = tmp_path / "scaled.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_carry_family_spec(path)


def test_dispatching_loader_returns_the_right_model() -> None:
    carry, carry_hash = load_family_spec(CONFIG)
    panel, panel_hash = load_family_spec(Path("configs/xs-momentum-panel-v1.json"))
    assert isinstance(carry, CarryFamilySpec)
    assert isinstance(panel, PanelFamilySpec)
    assert carry_hash != panel_hash


def test_dispatching_loader_rejects_an_unknown_family(tmp_path: Path) -> None:
    document = json.loads(CONFIG.read_text(encoding="utf-8"))
    document["family_name"] = "something_else_v1"
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_family_spec(path)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_carry_config.py -v --basetemp=C:/Users/User/AppData/Local/Temp/pytest-carry`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading_bot.carry_config'`

- [ ] **Step 4: Write the carry declaration module**

```python
# src/trading_bot/carry_config.py
"""Frozen declaration of the funding carry experiment family."""

import json
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from trading_bot.canonical import content_sha256
from trading_bot.panel_config import PanelFoldGeometry, PanelStatistics

MEMBER_NAMES: tuple[str, ...] = ("carry_l1w_h4w", "carry_l4w_h4w", "carry_l4w_h13w")
CONTROL_NAMES: tuple[str, ...] = ("no_trade", "random_pairs", "all_pairs_ew")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CarryMember(_Frozen):
    name: Literal["carry_l1w_h4w", "carry_l4w_h4w", "carry_l4w_h13w"]
    lookback_weeks: int
    hold_weeks: int

    @field_validator("lookback_weeks", "hold_weeks")
    @classmethod
    def validate_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("lookback_weeks and hold_weeks must be positive")
        return value


class CarryControl(_Frozen):
    name: Literal["no_trade", "random_pairs", "all_pairs_ew"]
    kind: Literal["no_trade", "random_pairs", "all_pairs"]


class CarryPair(_Frozen):
    perpetual: str
    spot: str
    multiplier: int

    @field_validator("multiplier")
    @classmethod
    def validate_multiplier(cls, value: int) -> int:
        if value < 1:
            raise ValueError("multiplier must be at least one")
        return value


class CarryUniverseRules(_Frozen):
    minimum_history_days: int
    liquidity_window_days: int
    minimum_median_quote_volume: Decimal
    maximum_pairs: int
    minimum_pairs: int
    tier_one_rank_limit: int


class CarrySelectionRules(_Frozen):
    decile_denominator: int
    minimum_selected: int


class CarryCostTable(_Frozen):
    name: Literal["base", "adverse"]
    perpetual_fee_bps_per_side: Decimal
    spot_fee_bps_per_side: Decimal
    slippage_bps_per_side_tier_one: Decimal
    slippage_bps_per_side_tier_two: Decimal
    funding_receipt_multiplier: Decimal
    funding_payment_multiplier: Decimal
    forced_close_multiplier: Decimal


class CarryCosts(_Frozen):
    base: CarryCostTable
    adverse: CarryCostTable


class CarryFamilySpec(_Frozen):
    spec_version: Literal["1.0.0"]
    family_name: Literal["funding_carry_panel_v1"]
    hypothesis: str
    perpetual_venue: Literal["BINANCE_UM"]
    spot_venue: Literal["BINANCE_SPOT"]
    holding_days: int
    members: tuple[CarryMember, ...]
    controls: tuple[CarryControl, ...]
    pairs: tuple[CarryPair, ...]
    excluded_pairs: tuple[CarryPair, ...]
    universe: CarryUniverseRules
    selection: CarrySelectionRules
    costs: CarryCosts
    folds: PanelFoldGeometry
    statistics: PanelStatistics

    @field_validator("members")
    @classmethod
    def validate_members(cls, value: tuple[CarryMember, ...]) -> tuple[CarryMember, ...]:
        if tuple(item.name for item in value) != MEMBER_NAMES:
            raise ValueError("the member set is frozen and ordered")
        return value

    @field_validator("controls")
    @classmethod
    def validate_controls(cls, value: tuple[CarryControl, ...]) -> tuple[CarryControl, ...]:
        if tuple(item.name for item in value) != CONTROL_NAMES:
            raise ValueError("the control set is frozen and ordered")
        return value

    @field_validator("pairs")
    @classmethod
    def validate_pairs(cls, value: tuple[CarryPair, ...]) -> tuple[CarryPair, ...]:
        if any(item.multiplier != 1 for item in value):
            raise ValueError("included pairs must map one-to-one; scaled pairs are excluded")
        perpetuals = [item.perpetual for item in value]
        if len(set(perpetuals)) != len(perpetuals):
            raise ValueError("a perpetual may appear in at most one pair")
        return value

    @field_validator("excluded_pairs")
    @classmethod
    def validate_excluded(cls, value: tuple[CarryPair, ...]) -> tuple[CarryPair, ...]:
        if any(item.multiplier == 1 for item in value):
            raise ValueError("only scaled-multiplier pairs are excluded by declaration")
        return value


def load_carry_family_spec(path: Path) -> tuple[CarryFamilySpec, str]:
    """Load the frozen declaration and return it with its canonical hash."""
    document = json.loads(path.read_text(encoding="utf-8"))
    return CarryFamilySpec.model_validate(document), content_sha256(document)
```

- [ ] **Step 5: Add the dispatching loader to `panel_config.py`**

Append at the end of `src/trading_bot/panel_config.py`, after `load_panel_family_spec`:

```python
def load_family_spec(path: Path) -> "tuple[PanelFamilySpec | CarryFamilySpec, str]":
    """Load whichever frozen family declaration the file holds.

    Dispatches on ``family_name`` so the walk-forward manifest and the
    decision module can serve every family without each family's model
    knowing about the others. Any unknown name falls through to the panel
    model, whose closed ``family_name`` literal rejects it.
    """
    from trading_bot.carry_config import CarryFamilySpec  # local: avoids an import cycle

    document = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(document, dict) and document.get("family_name") == "funding_carry_panel_v1":
        return CarryFamilySpec.model_validate(document), content_sha256(document)
    return PanelFamilySpec.model_validate(document), content_sha256(document)
```

Add `from typing import TYPE_CHECKING` at the top and, under `if TYPE_CHECKING:`, `from trading_bot.carry_config import CarryFamilySpec`, so the string annotation resolves for mypy without a runtime cycle.

- [ ] **Step 6: Run tests, lint and types**

```
uv run pytest tests/test_carry_config.py tests/test_panel_config.py -v --basetemp=C:/Users/User/AppData/Local/Temp/pytest-carry
uv run ruff check .
uv run mypy
```
Expected: all passing, clean.

- [ ] **Step 7: Commit**

```bash
git add configs/funding-carry-panel-v1.json src/trading_bot/carry_config.py src/trading_bot/panel_config.py tests/test_carry_config.py
git commit -m "feat: freeze the funding carry family declaration

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Task 2: Spot market in the capture

**Files:**
- Modify: `src/trading_bot/panel_capture.py`
- Modify: `src/trading_bot/cli.py` (`panel-capture --market`)
- Test: additions to `tests/test_panel_capture.py`

**Interfaces:**
- Produces: `Market = Literal["um", "spot"]`; `build_kline_zip_url(symbol, month, *, market="um")`; `build_daily_kline_zip_url(symbol, date, *, market="um")`; `discover_panel_months(fetch, *, symbol, kind, market="um")`; `capture_panel(..., market="um")`; `repair_panel_capture` reads `market` from the source manifest; capture manifests carry `"market"` and the market's venue string; `venue_for_market(market) -> str`.
- Spot captures fetch no funding and record `discovered_months[symbol]["fundingRate"] = []`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_panel_capture.py`, reusing its existing `zip_bytes`, `kline_csv` helpers and `PanelPayload` import:

```python
def test_spot_urls_use_the_spot_prefix() -> None:
    from trading_bot.panel_capture import build_daily_kline_zip_url, build_kline_zip_url

    assert build_kline_zip_url("BTCUSDT", "2024-01", market="spot") == (
        "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2024-01.zip"
    )
    assert build_daily_kline_zip_url("BTCUSDT", "2024-01-15", market="spot") == (
        "https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1d/BTCUSDT-1d-2024-01-15.zip"
    )
    assert build_kline_zip_url("BTCUSDT", "2024-01") == (
        "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2024-01.zip"
    )


def test_spot_capture_fetches_no_funding_and_records_its_market(tmp_path: Path) -> None:
    urls: list[str] = []

    def fetch(url: str) -> PanelPayload:
        urls.append(url)
        assert "fundingRate" not in url
        assert "/data/spot/" in url
        return PanelPayload(url=url, raw_bytes=zip_bytes("k.csv", kline_csv(3)), received_time_ns=1)

    artifact = capture_panel(
        workspace_root=tmp_path,
        output_directory=tmp_path / "spot",
        reserve_bytes=0,
        symbols=("BTCUSDT",),
        months=("2024-01",),
        fetch=fetch,
        market="spot",
    )
    manifest = json.loads(artifact.capture_manifest_path.read_text(encoding="utf-8"))
    assert manifest["market"] == "spot"
    assert manifest["venue"] == "BINANCE_SPOT"
    assert {s["kind"] for s in manifest["sources"]} == {"klines"}
    dataset_manifest = json.loads((artifact.dataset_root / "dataset-manifest.json").read_text(encoding="utf-8"))
    assert dataset_manifest["funding_row_count"] == 0
    assert verify_panel_capture(artifact.capture_root) == (True, ())
    assert len(urls) == 1


def test_default_market_manifest_names_um(tmp_path: Path) -> None:
    def fetch(url: str) -> PanelPayload:
        if "fundingRate" in url:
            text = "calc_time,funding_interval_hours,last_funding_rate\n0,8,0.0001\n"
            return PanelPayload(url=url, raw_bytes=zip_bytes("f.csv", text), received_time_ns=1)
        return PanelPayload(url=url, raw_bytes=zip_bytes("k.csv", kline_csv(3)), received_time_ns=1)

    artifact = capture_panel(
        workspace_root=tmp_path, output_directory=tmp_path / "um", reserve_bytes=0,
        symbols=("BTCUSDT",), months=("2024-01",), fetch=fetch,
    )
    manifest = json.loads(artifact.capture_manifest_path.read_text(encoding="utf-8"))
    assert manifest["market"] == "um"
    assert manifest["venue"] == "BINANCE_UM"


def test_repair_of_a_spot_capture_stays_on_the_spot_market(tmp_path: Path) -> None:
    def fetch(url: str) -> PanelPayload:
        assert "/data/spot/" in url
        return PanelPayload(url=url, raw_bytes=zip_bytes("k.csv", kline_csv(3)), received_time_ns=1)

    source = capture_panel(
        workspace_root=tmp_path, output_directory=tmp_path / "spot", reserve_bytes=0,
        symbols=("BTCUSDT",), months=("2024-01",), fetch=fetch, market="spot",
    )
    repaired = repair_panel_capture(
        workspace_root=tmp_path, source_capture_root=source.capture_root,
        output_directory=tmp_path / "spot-repaired", reserve_bytes=0, fetch=fetch,
    )
    manifest = json.loads(repaired.capture_manifest_path.read_text(encoding="utf-8"))
    assert manifest["market"] == "spot"
    assert manifest["venue"] == "BINANCE_SPOT"
    assert verify_panel_capture(repaired.capture_root) == (True, ())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_panel_capture.py -k "spot or default_market" -v --basetemp=C:/Users/User/AppData/Local/Temp/pytest-carry`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'market'`

- [ ] **Step 3: Thread `market` through `panel_capture.py`**

Near the top, replace the single venue constant:

```python
# before
_VENUE = "BINANCE_UM"

# after
Market = Literal["um", "spot"]
_MARKET_PATH: dict[str, str] = {"um": "data/futures/um", "spot": "data/spot"}
_MARKET_VENUE: dict[str, str] = {"um": "BINANCE_UM", "spot": "BINANCE_SPOT"}
_VENUE = _MARKET_VENUE["um"]


def venue_for_market(market: str) -> str:
    if market not in _MARKET_VENUE:
        raise PanelCaptureError(f"invalid panel market: {market}")
    return _MARKET_VENUE[market]


def _market_path(market: str) -> str:
    if market not in _MARKET_PATH:
        raise PanelCaptureError(f"invalid panel market: {market}")
    return _MARKET_PATH[market]
```

Add `Literal` to the `typing` import. The three URL builders and the listing helpers gain a keyword-only `market: str = "um"` and build from `_market_path(market)`:

```python
def build_kline_zip_url(symbol: str, month: str, *, market: str = "um") -> str:
    _validate_symbol(symbol)
    _validate_month(month)
    return (
        f"https://data.binance.vision/{_market_path(market)}/monthly/klines/"
        f"{symbol}/1d/{symbol}-1d-{month}.zip"
    )


def build_daily_kline_zip_url(symbol: str, date: str, *, market: str = "um") -> str:
    _validate_symbol(symbol)
    _validate_date(date)
    return (
        f"https://data.binance.vision/{_market_path(market)}/daily/klines/"
        f"{symbol}/1d/{symbol}-1d-{date}.zip"
    )


def build_funding_zip_url(symbol: str, month: str) -> str:
    _validate_symbol(symbol)
    _validate_month(month)
    return (
        "https://data.binance.vision/data/futures/um/monthly/fundingRate/"
        f"{symbol}/{symbol}-fundingRate-{month}.zip"
    )


def build_month_listing_url(symbol: str, kind: str, *, market: str = "um") -> str:
    prefix = _listing_prefix(symbol, kind, market)
    ...  # unchanged body


def discover_panel_months(
    fetch: PanelFetch, *, symbol: str, kind: str, market: str = "um"
) -> tuple[str, ...]:
    ...  # pass market to build_month_listing_url and _listing_key_pattern


def _listing_prefix(symbol: str, kind: str, market: str = "um") -> str:
    _validate_symbol(symbol)
    if kind == "klines":
        return f"{_market_path(market)}/monthly/klines/{symbol}/1d/"
    if kind == "fundingRate":
        if market != "um":
            raise PanelCaptureError("funding exists only on the um market")
        return "data/futures/um/monthly/fundingRate/" + f"{symbol}/"
    raise PanelCaptureError(f"invalid panel source kind: {kind}")


def _listing_key_pattern(symbol: str, kind: str, market: str = "um") -> re.Pattern[str]:
    escaped = re.escape(symbol)
    base = re.escape(_market_path(market))
    if kind == "klines":
        return re.compile(
            rf"^{base}/monthly/klines/{escaped}/1d/{escaped}-1d-(\d{{4}}-\d{{2}})\.zip$"
        )
    return re.compile(
        rf"^data/futures/um/monthly/fundingRate/{escaped}/"
        rf"{escaped}-fundingRate-(\d{{4}}-\d{{2}})\.zip$"
    )
```

`parse_kline_zip` and `parse_funding_zip` gain `venue: str = _VENUE` and use it in place of the constant. `capture_panel` gains `market: str = "um"` after `fetch`, validates it through `venue_for_market`, records it:

```python
    venue = venue_for_market(market)
    capture_parameters: dict[str, object] = {
        "capture_version": _CAPTURE_VERSION,
        "market": market,
        "symbols": list(symbols),
        "months": list(months) if months is not None else None,
        "month_from": month_from,
        "month_to": month_to,
    }
```

In the per-symbol block, funding is discovered and fetched only on `um`:

```python
            if months is not None:
                kline_months: tuple[str, ...] = months
                funding_months: tuple[str, ...] = months if market == "um" else ()
                ordered_months: tuple[str, ...] = months
            else:
                discovered_kline_months = discover_panel_months(
                    download, symbol=symbol, kind="klines", market=market
                )
                discovered_funding_months = (
                    discover_panel_months(download, symbol=symbol, kind="fundingRate")
                    if market == "um"
                    else ()
                )
                ...  # unchanged, the rest of the block reads kline_months / funding_months
            for month in ordered_months:
                for kind, url_builder, kind_months in (
                    ("klines", lambda s, m: build_kline_zip_url(s, m, market=market), kline_months),
                    ("fundingRate", build_funding_zip_url, funding_months),
                ):
```

Both `parse_*` calls in the loop pass `venue=venue`. `_fill_gap_days` gains `market: str` and `venue: str` keyword parameters and passes them to `build_daily_kline_zip_url(..., market=market)` and `parse_kline_zip(..., venue=venue)`. Both manifest `material` dicts write `"market": market` and `"venue": venue`. `repair_panel_capture` reads `market = str(source_manifest.get("market", "um"))` and `venue = venue_for_market(market)`, passes them to `_fill_gap_days` and to its `parse_*` calls, and writes them into its material. The `_validate_panel_url` host allowlist is unchanged; `data.binance.vision` already covers both paths.

- [ ] **Step 4: Add `--market` to the CLI**

In the `panel-capture` parser block:

```python
    panel_capture.add_argument("--market", choices=("um", "spot"), default="um")
```

and in its dispatch, pass `market=parsed.market` to `capture_panel`. `panel-capture-repair` needs no flag; it reads the market from the source manifest.

- [ ] **Step 5: Run the capture tests, then the full suite**

```
uv run pytest tests/test_panel_capture.py -v --basetemp=C:/Users/User/AppData/Local/Temp/pytest-carry
uv run pytest -q --basetemp=C:/Users/User/AppData/Local/Temp/pytest-carry
uv run ruff check .
uv run mypy
```
Expected: the existing capture tests still pass with the default market; the four new ones pass; full suite green; clean.

- [ ] **Step 6: Commit**

```bash
git add src/trading_bot/panel_capture.py src/trading_bot/cli.py tests/test_panel_capture.py
git commit -m "feat: capture the binance spot market alongside the perpetuals

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Task 3: Pair universe

**Files:**
- Create: `src/trading_bot/carry_universe.py`
- Test: `tests/test_carry_universe.py`

**Interfaces:**
- Consumes: `panel_universe.ContractHistory`, `select_universe`, `PanelUniverseRules`; `carry_config.CarryPair`, `CarryUniverseRules`.
- Produces: `EligiblePair(pair_id, perpetual_contract_id, spot_contract_id, perpetual_tier, spot_tier, tier, liquidity_rank, median_quote_volume)`; `PairUniverseSnapshot(decision_close_ns, pairs, reason_codes)`; `select_pair_universe(perp_histories, spot_histories, *, pairs, decision_close_ns, rules) -> PairUniverseSnapshot`.

`pair_id` is the perpetual's contract id, since a perpetual maps to exactly one spot leg. Leg eligibility reuses `select_universe` per leg with the rank cap lifted and the minimum lowered to one, so the pair-level minimum of `minimum_pairs` is the only floor that can produce `UNIVERSE_TOO_SMALL`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_carry_universe.py
from decimal import Decimal

from trading_bot.carry_config import CarryPair, CarryUniverseRules
from trading_bot.carry_universe import select_pair_universe
from trading_bot.panel_reader import PanelBar
from trading_bot.panel_universe import build_contract_histories

DAY_NS = 86_400_000_000_000
RULES = CarryUniverseRules(
    minimum_history_days=5, liquidity_window_days=3, minimum_median_quote_volume=Decimal("100"),
    maximum_pairs=3, minimum_pairs=2, tier_one_rank_limit=1,
)
DECISION = 7 * DAY_NS - 1_000_000


def bars(symbol: str, *, days: int, volume: str, first_day: int = 0) -> list[PanelBar]:
    out = []
    for index in range(first_day, first_day + days):
        open_time_ns = index * DAY_NS
        close_time_ns = open_time_ns + DAY_NS - 1_000_000
        out.append(PanelBar(
            contract_id=f"{symbol}:{first_day * DAY_NS}", instrument_id=symbol,
            open_time_ns=open_time_ns, close_time_ns=close_time_ns,
            available_time_ns=close_time_ns + 1, close=Decimal(100 + index),
            quote_volume=Decimal(volume),
        ))
    return out


PAIRS = tuple(CarryPair(perpetual=s, spot=s, multiplier=1) for s in ("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT"))


def test_pairs_need_both_legs_and_are_ranked_by_the_perpetual() -> None:
    perp = build_contract_histories(tuple(
        bars("AAAUSDT", days=8, volume="900") + bars("BBBUSDT", days=8, volume="800")
        + bars("CCCUSDT", days=8, volume="700") + bars("DDDUSDT", days=8, volume="600")))
    spot = build_contract_histories(tuple(
        bars("AAAUSDT", days=8, volume="500") + bars("BBBUSDT", days=8, volume="900")
        + bars("CCCUSDT", days=8, volume="10")))  # CCC spot is illiquid, DDD has no spot bars
    snapshot = select_pair_universe(perp, spot, pairs=PAIRS, decision_close_ns=DECISION, rules=RULES)
    assert [p.pair_id for p in snapshot.pairs] == ["AAAUSDT:0", "BBBUSDT:0"]
    assert snapshot.pairs[0].perpetual_tier == 1 and snapshot.pairs[1].perpetual_tier == 2
    # spot ranks among spot legs: BBB (900) rank 1 -> tier 1, AAA (500) rank 2 -> tier 2
    assert snapshot.pairs[0].spot_tier == 2 and snapshot.pairs[1].spot_tier == 1
    # the pair tier is the worse of the two legs
    assert [p.tier for p in snapshot.pairs] == [2, 2]
    assert snapshot.reason_codes == ()


def test_too_few_pairs_is_universe_too_small() -> None:
    perp = build_contract_histories(tuple(bars("AAAUSDT", days=8, volume="900")))
    spot = build_contract_histories(tuple(bars("AAAUSDT", days=8, volume="900")))
    snapshot = select_pair_universe(perp, spot, pairs=PAIRS, decision_close_ns=DECISION, rules=RULES)
    assert snapshot.pairs == () and snapshot.reason_codes == ("UNIVERSE_TOO_SMALL",)


def test_a_hole_in_either_leg_excludes_the_pair() -> None:
    perp = build_contract_histories(tuple(bars("AAAUSDT", days=8, volume="900") + bars("BBBUSDT", days=8, volume="800")))
    spot_rows = bars("AAAUSDT", days=8, volume="900") + bars("BBBUSDT", days=8, volume="800")
    spot_rows = [b for b in spot_rows if not (b.instrument_id == "BBBUSDT" and b.open_time_ns == 5 * DAY_NS)]
    spot = build_contract_histories(tuple(spot_rows))
    snapshot = select_pair_universe(perp, spot, pairs=PAIRS, decision_close_ns=DECISION,
                                    rules=CarryUniverseRules(**{**RULES.model_dump(), "minimum_pairs": 1}))
    assert [p.pair_id for p in snapshot.pairs] == ["AAAUSDT:0"]


def test_maximum_pairs_truncates_after_ranking() -> None:
    symbols = [f"S{i}USDT" for i in range(6)]
    pairs = tuple(CarryPair(perpetual=s, spot=s, multiplier=1) for s in symbols)
    perp = build_contract_histories(tuple(b for i, s in enumerate(symbols) for b in bars(s, days=8, volume=str(900 - i))))
    spot = build_contract_histories(tuple(b for s in symbols for b in bars(s, days=8, volume="900")))
    snapshot = select_pair_universe(perp, spot, pairs=pairs, decision_close_ns=DECISION, rules=RULES)
    assert len(snapshot.pairs) == 3
    assert [p.liquidity_rank for p in snapshot.pairs] == [1, 2, 3]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_carry_universe.py -v --basetemp=C:/Users/User/AppData/Local/Temp/pytest-carry`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading_bot.carry_universe'`

- [ ] **Step 3: Write the module**

```python
# src/trading_bot/carry_universe.py
"""Point-in-time eligibility of long-spot short-perpetual pairs."""

from dataclasses import dataclass
from decimal import Decimal

from trading_bot.carry_config import CarryPair, CarryUniverseRules
from trading_bot.panel_config import PanelUniverseRules
from trading_bot.panel_universe import ContractHistory, EligibleContract, select_universe

_UNCAPPED = 1_000_000


@dataclass(frozen=True, slots=True)
class EligiblePair:
    pair_id: str
    perpetual_contract_id: str
    spot_contract_id: str
    perpetual_tier: int
    spot_tier: int
    tier: int
    liquidity_rank: int
    median_quote_volume: Decimal


@dataclass(frozen=True, slots=True)
class PairUniverseSnapshot:
    decision_close_ns: int
    pairs: tuple[EligiblePair, ...]
    reason_codes: tuple[str, ...]


def select_pair_universe(
    perp_histories: dict[str, ContractHistory],
    spot_histories: dict[str, ContractHistory],
    *,
    pairs: tuple[CarryPair, ...],
    decision_close_ns: int,
    rules: CarryUniverseRules,
) -> PairUniverseSnapshot:
    """Apply the P1.27 leg rules to both legs, then rank and cap the pairs.

    Each leg is screened by ``select_universe`` with the rank cap lifted and
    the minimum lowered to one, so the only floor that can declare the
    universe too small is the pair-level ``minimum_pairs``. The spot tier is
    the tier the spot leg holds among all eligible spot legs, the perpetual
    tier follows the pair ranking, and a pair's cost tier is the worse of
    the two, as spec section 7.1 states.
    """
    leg_rules = PanelUniverseRules(
        minimum_history_days=rules.minimum_history_days,
        liquidity_window_days=rules.liquidity_window_days,
        minimum_median_quote_volume=rules.minimum_median_quote_volume,
        maximum_contracts=_UNCAPPED,
        minimum_contracts=1,
        tier_one_rank_limit=rules.tier_one_rank_limit,
    )
    perp_snapshot = select_universe(perp_histories, decision_close_ns=decision_close_ns, rules=leg_rules)
    spot_snapshot = select_universe(spot_histories, decision_close_ns=decision_close_ns, rules=leg_rules)
    perp_by_symbol = _by_symbol(perp_snapshot.contracts, perp_histories)
    spot_by_symbol = _by_symbol(spot_snapshot.contracts, spot_histories)

    candidates: list[tuple[Decimal, str, EligibleContract, EligibleContract]] = []
    for pair in pairs:
        perp = perp_by_symbol.get(pair.perpetual)
        spot = spot_by_symbol.get(pair.spot)
        if perp is None or spot is None:
            continue
        candidates.append((perp.median_quote_volume, perp.contract_id, perp, spot))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    if len(candidates) < rules.minimum_pairs:
        return PairUniverseSnapshot(decision_close_ns, (), ("UNIVERSE_TOO_SMALL",))

    selected = candidates[: rules.maximum_pairs]
    eligible = []
    for index, (median, _, perp, spot) in enumerate(selected):
        rank = index + 1
        perpetual_tier = 1 if rank <= rules.tier_one_rank_limit else 2
        eligible.append(
            EligiblePair(
                pair_id=perp.contract_id,
                perpetual_contract_id=perp.contract_id,
                spot_contract_id=spot.contract_id,
                perpetual_tier=perpetual_tier,
                spot_tier=spot.tier,
                tier=max(perpetual_tier, spot.tier),
                liquidity_rank=rank,
                median_quote_volume=median,
            )
        )
    return PairUniverseSnapshot(decision_close_ns, tuple(eligible), ())


def _by_symbol(
    contracts: tuple[EligibleContract, ...], histories: dict[str, ContractHistory]
) -> dict[str, EligibleContract]:
    return {histories[item.contract_id].instrument_id: item for item in contracts}
```

- [ ] **Step 4: Run tests, lint and types**

```
uv run pytest tests/test_carry_universe.py -v --basetemp=C:/Users/User/AppData/Local/Temp/pytest-carry
uv run ruff check .
uv run mypy
```
Expected: 4 passed, clean.

- [ ] **Step 5: Commit**

```bash
git add src/trading_bot/carry_universe.py tests/test_carry_universe.py
git commit -m "feat: select the eligible spot-perpetual pair universe

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Task 4: Trailing funding, cohorts and the book

**Files:**
- Create: `src/trading_bot/carry_signals.py`
- Test: `tests/test_carry_signals.py`

**Interfaces:**
- Consumes: `carry_universe.EligiblePair`, `PairUniverseSnapshot`; `panel_reader.FundingEvent`; `carry_config.CarryMember`, `CarryControl`, `CarrySelectionRules`.
- Produces: `WEEK_NS`; `perp_leg(contract_id) -> str`, `spot_leg(contract_id) -> str` producing `perp:<id>` and `spot:<id>`; `trailing_funding(events, *, decision_close_ns, lookback_weeks) -> Decimal | None`; `CohortEntry(pair_id, perpetual_leg, spot_leg, tier)`; `Cohort(decision_close_ns, entries, reason_codes)`; `select_member_cohort(snapshot, *, trailing, selection) -> Cohort`; `select_control_cohort(snapshot, *, kind, trailing, selection, random_seed) -> Cohort`; `assemble_book(cohorts, *, hold_weeks, decision_close_ns) -> tuple[tuple[str, Decimal], ...]` returning leg weights sorted by leg id.

Capital: each retained cohort holds `1/H` of capital, each pair in it an equal share `c = 1/(H·n)`, expressed as spot leg `+c/2` and perpetual leg `−c/2`. A cohort is retained while `decision − cohort.decision < H weeks`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_carry_signals.py
from decimal import Decimal

from trading_bot.carry_config import CarrySelectionRules
from trading_bot.carry_signals import (
    Cohort,
    CohortEntry,
    assemble_book,
    perp_leg,
    select_control_cohort,
    select_member_cohort,
    spot_leg,
    trailing_funding,
)
from trading_bot.carry_universe import EligiblePair, PairUniverseSnapshot
from trading_bot.panel_reader import FundingEvent

WEEK_NS = 604_800_000_000_000
DECISION = 10 * WEEK_NS
SELECTION = CarrySelectionRules(decile_denominator=10, minimum_selected=2)


def event(cid: str, t: int, rate: str) -> FundingEvent:
    return FundingEvent(contract_id=cid, instrument_id=cid.split(":")[0], calc_time_ns=t, rate=Decimal(rate))


def pair(i: int) -> EligiblePair:
    return EligiblePair(pair_id=f"P{i}:0", perpetual_contract_id=f"P{i}:0", spot_contract_id=f"P{i}:7",
                        perpetual_tier=1, spot_tier=1, tier=1, liquidity_rank=i + 1, median_quote_volume=Decimal(10))


def test_trailing_funding_sums_only_inside_the_window() -> None:
    events = (event("P0:0", DECISION - 2 * WEEK_NS, "0.5"), event("P0:0", DECISION - WEEK_NS + 1, "0.1"),
              event("P0:0", DECISION, "0.2"), event("P0:0", DECISION + 1, "9"))
    assert trailing_funding(events, decision_close_ns=DECISION, lookback_weeks=1) == Decimal("0.3")
    assert trailing_funding((), decision_close_ns=DECISION, lookback_weeks=1) is None


def test_member_cohort_takes_the_top_decile_with_positive_funding() -> None:
    snapshot = PairUniverseSnapshot(DECISION, tuple(pair(i) for i in range(20)), ())
    trailing = {f"P{i}:0": Decimal(i) - Decimal(5) for i in range(20)}  # P0..P5 <= 0
    trailing["P19:0"] = None
    cohort = select_member_cohort(snapshot, trailing=trailing, selection=SELECTION)
    assert [e.pair_id for e in cohort.entries] == ["P18:0", "P17:0"]
    assert cohort.reason_codes == ()


def test_member_cohort_is_empty_when_too_few_pay_funding() -> None:
    snapshot = PairUniverseSnapshot(DECISION, tuple(pair(i) for i in range(20)), ())
    trailing = {f"P{i}:0": Decimal(-1) for i in range(20)}
    trailing["P3:0"] = Decimal("0.5")
    cohort = select_member_cohort(snapshot, trailing=trailing, selection=SELECTION)
    assert cohort.entries == () and cohort.reason_codes == ("NO_CARRY_COHORT",)


def test_controls() -> None:
    snapshot = PairUniverseSnapshot(DECISION, tuple(pair(i) for i in range(20)), ())
    trailing = {f"P{i}:0": Decimal(1) if i % 2 else Decimal(-1) for i in range(20)}
    everything = select_control_cohort(snapshot, kind="all_pairs", trailing=trailing, selection=SELECTION, random_seed=17)
    assert len(everything.entries) == 10 and all(int(e.pair_id[1:].split(":")[0]) % 2 for e in everything.entries)
    random_a = select_control_cohort(snapshot, kind="random_pairs", trailing=trailing, selection=SELECTION, random_seed=17)
    random_b = select_control_cohort(snapshot, kind="random_pairs", trailing=trailing, selection=SELECTION, random_seed=17)
    assert random_a.entries == random_b.entries and len(random_a.entries) == 2
    assert select_control_cohort(snapshot, kind="no_trade", trailing=trailing, selection=SELECTION, random_seed=17).entries == ()


def test_book_holds_the_last_h_cohorts_at_equal_capital() -> None:
    def cohort(week: int, ids: list[str]) -> Cohort:
        return Cohort(week * WEEK_NS, tuple(CohortEntry(i, perp_leg(i), spot_leg(i.replace(":0", ":7")), 1) for i in ids), ())

    cohorts = (
        cohort(6, ["Z:0"]),
        cohort(7, ["A:0", "B:0"]),
        cohort(8, ["C:0"]),
        cohort(9, ["A:0"]),
        cohort(10, ["D:0", "E:0", "F:0", "G:0"]),
    )
    book = dict(assemble_book(cohorts, hold_weeks=4, decision_close_ns=10 * WEEK_NS))
    # week 6 is out of the window: 10 - 6 = 4 >= H; weeks 7..10 are retained
    assert "perp:Z:0" not in book
    # each cohort holds 1/4; a pair in a two-pair cohort holds 1/8, split +1/16 spot / -1/16 perp
    assert book["perp:B:0"] == Decimal("-0.0625")
    assert book["spot:C:7"] == Decimal("0.125") and book["perp:C:0"] == Decimal("-0.125")
    # A:0 is in the week-7 and week-9 cohorts: 1/16 + 1/8 = 3/16
    assert book["perp:A:0"] == Decimal("-0.1875")
    assert book["perp:D:0"] == Decimal("-0.03125")
    assert sum(abs(v) for v in book.values()) == Decimal(1)
    assert sum(book.values()) == Decimal(0)
    assert list(book) == sorted(book)


def test_book_with_an_empty_cohort_deploys_less_than_unit_gross() -> None:
    cohorts = (Cohort(9 * WEEK_NS, (), ("NO_CARRY_COHORT",)),
               Cohort(10 * WEEK_NS, (CohortEntry("A:0", "perp:A:0", "spot:A:7", 1),), ()))
    book = dict(assemble_book(cohorts, hold_weeks=2, decision_close_ns=10 * WEEK_NS))
    assert sum(abs(v) for v in book.values()) == Decimal("0.5")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_carry_signals.py -v --basetemp=C:/Users/User/AppData/Local/Temp/pytest-carry`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading_bot.carry_signals'`

- [ ] **Step 3: Write the module**

```python
# src/trading_bot/carry_signals.py
"""Trailing funding, cohort selection and book assembly for the carry family."""

from dataclasses import dataclass
from decimal import Decimal
from random import Random

from trading_bot.carry_config import CarrySelectionRules
from trading_bot.carry_universe import PairUniverseSnapshot
from trading_bot.panel_reader import FundingEvent

WEEK_NS = 604_800_000_000_000
NO_CARRY_COHORT = "NO_CARRY_COHORT"


def perp_leg(contract_id: str) -> str:
    return f"perp:{contract_id}"


def spot_leg(contract_id: str) -> str:
    return f"spot:{contract_id}"


@dataclass(frozen=True, slots=True)
class CohortEntry:
    pair_id: str
    perpetual_leg: str
    spot_leg: str
    tier: int


@dataclass(frozen=True, slots=True)
class Cohort:
    decision_close_ns: int
    entries: tuple[CohortEntry, ...]
    reason_codes: tuple[str, ...]


def trailing_funding(
    events: tuple[FundingEvent, ...], *, decision_close_ns: int, lookback_weeks: int
) -> Decimal | None:
    """Sum the settlements in ``(decision - L weeks, decision]``; None if there are none."""
    start = decision_close_ns - lookback_weeks * WEEK_NS
    inside = [e.rate for e in events if start < e.calc_time_ns <= decision_close_ns]
    if not inside:
        return None
    return sum(inside, Decimal(0))


def _entries(snapshot: PairUniverseSnapshot, pair_ids: list[str]) -> tuple[CohortEntry, ...]:
    by_id = {pair.pair_id: pair for pair in snapshot.pairs}
    return tuple(
        CohortEntry(
            pair_id=pair_id,
            perpetual_leg=perp_leg(by_id[pair_id].perpetual_contract_id),
            spot_leg=spot_leg(by_id[pair_id].spot_contract_id),
            tier=by_id[pair_id].tier,
        )
        for pair_id in pair_ids
    )


def _decile_size(count: int, selection: CarrySelectionRules) -> int:
    return max(selection.minimum_selected, count // selection.decile_denominator)


def select_member_cohort(
    snapshot: PairUniverseSnapshot,
    *,
    trailing: dict[str, Decimal | None],
    selection: CarrySelectionRules,
) -> Cohort:
    """Top decile of the eligible pairs by trailing funding, positive funding required."""
    paying = [
        (value, pair.pair_id)
        for pair in snapshot.pairs
        for value in (trailing.get(pair.pair_id),)
        if value is not None and value > 0
    ]
    if len(paying) < selection.minimum_selected:
        return Cohort(snapshot.decision_close_ns, (), (NO_CARRY_COHORT,))
    paying.sort(key=lambda item: (-item[0], item[1]))
    size = min(len(paying), _decile_size(len(snapshot.pairs), selection))
    return Cohort(snapshot.decision_close_ns, _entries(snapshot, [pid for _, pid in paying[:size]]), ())


def select_control_cohort(
    snapshot: PairUniverseSnapshot,
    *,
    kind: str,
    trailing: dict[str, Decimal | None],
    selection: CarrySelectionRules,
    random_seed: int,
) -> Cohort:
    if kind == "no_trade":
        return Cohort(snapshot.decision_close_ns, (), ())
    if kind == "all_pairs":
        ids = sorted(
            pair.pair_id
            for pair in snapshot.pairs
            for value in (trailing.get(pair.pair_id),)
            if value is not None and value > 0
        )
        return Cohort(snapshot.decision_close_ns, _entries(snapshot, ids), ())
    if kind == "random_pairs":
        ids = sorted(pair.pair_id for pair in snapshot.pairs)
        Random(random_seed ^ snapshot.decision_close_ns).shuffle(ids)
        size = min(len(ids), _decile_size(len(snapshot.pairs), selection))
        return Cohort(snapshot.decision_close_ns, _entries(snapshot, sorted(ids[:size])), ())
    raise ValueError(f"unknown control kind: {kind}")


def assemble_book(
    cohorts: tuple[Cohort, ...], *, hold_weeks: int, decision_close_ns: int
) -> tuple[tuple[str, Decimal], ...]:
    """Union of the last H cohorts, each on 1/H of capital, as leg weights.

    A pair share ``c`` is expressed as spot ``+c/2`` and perpetual ``-c/2``,
    so a full book has unit gross across both legs and zero net exposure,
    which is the capital basis spec section 3.1 declares. An empty cohort
    leaves its share of capital undeployed rather than redistributing it.
    """
    if hold_weeks < 1:
        raise ValueError("hold_weeks must be positive")
    share_per_cohort = Decimal(1) / Decimal(hold_weeks)
    weights: dict[str, Decimal] = {}
    for cohort in cohorts:
        age = decision_close_ns - cohort.decision_close_ns
        if age < 0 or age >= hold_weeks * WEEK_NS or not cohort.entries:
            continue
        pair_share = share_per_cohort / Decimal(len(cohort.entries))
        for entry in cohort.entries:
            weights[entry.spot_leg] = weights.get(entry.spot_leg, Decimal(0)) + pair_share / 2
            weights[entry.perpetual_leg] = weights.get(entry.perpetual_leg, Decimal(0)) - pair_share / 2
    return tuple(sorted(weights.items()))
```

- [ ] **Step 4: Run tests, lint and types**

```
uv run pytest tests/test_carry_signals.py -v --basetemp=C:/Users/User/AppData/Local/Temp/pytest-carry
uv run ruff check .
uv run mypy
```
Expected: 6 passed, clean.

- [ ] **Step 5: Commit**

```bash
git add src/trading_bot/carry_signals.py tests/test_carry_signals.py
git commit -m "feat: select carry cohorts and assemble the two-leg book

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Task 5: Per-leg fees in the accounting, and the carry wrapper

**Files:**
- Modify: `src/trading_bot/panel_accounting.py` (`fee_overrides`)
- Create: `src/trading_bot/carry_accounting.py`
- Test: additions to `tests/test_panel_accounting.py`; `tests/test_carry_accounting.py`

**Interfaces:**
- `evaluate_episode(..., fee_overrides: dict[str, Decimal] | None = None)`: a contract id present in `fee_overrides` is charged that fee per side instead of `cost_table.fee_bps_per_side`. Default behaviour unchanged.
- Produces: `CarryEpisode(result: EpisodeResult, funding_collected: Decimal, basis_pnl: Decimal, spot_trading_cost: Decimal, perpetual_trading_cost: Decimal)`; `evaluate_carry_episode(*, sample_id, member, decision_close_ns, holding_days, leg_weights, previous_leg_weights, histories, tiers, funding_by_leg, cost_table: CarryCostTable, pair_of_leg: dict[str, str]) -> CarryEpisode`.

The wrapper maps the carry cost table onto a `PanelCostTable` whose `fee_bps_per_side` is the perpetual fee, and passes the spot fee for every `spot:` leg through `fee_overrides`. The returned `EpisodeResult` has its per-contract contributions re-aggregated to pair level so the concentration gate reads pairs, and its `drifted_weights` stay at leg level for the fold runner to carry.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_panel_accounting.py` (it already defines `BASE`, `histories`, `WEIGHTS`, `TIERS`, `DECISION`):

```python
def test_fee_overrides_apply_per_contract() -> None:
    plain = evaluate_episode(
        sample_id="s", member="m", decision_close_ns=DECISION, holding_days=7, weights=WEIGHTS,
        previous_weights=(), histories=histories(), tiers=TIERS, funding_by_contract={}, cost_table=BASE,
    )
    overridden = evaluate_episode(
        sample_id="s", member="m", decision_close_ns=DECISION, holding_days=7, weights=WEIGHTS,
        previous_weights=(), histories=histories(), tiers=TIERS, funding_by_contract={}, cost_table=BASE,
        fee_overrides={"A:0": Decimal("10")},
    )
    # A:0 turnover 0.5 at (10 + 5) bps instead of (5 + 5): +0.5 * 5 / 10000
    assert overridden.trading_cost - plain.trading_cost == Decimal("0.00025")
    assert dict(overridden.contract_net_contributions)["B:0"] == dict(plain.contract_net_contributions)["B:0"]
```

```python
# tests/test_carry_accounting.py
from decimal import Decimal

from trading_bot.carry_accounting import evaluate_carry_episode
from trading_bot.carry_config import CarryCostTable
from trading_bot.panel_reader import FundingEvent
from trading_bot.panel_universe import ContractHistory

DAY_NS = 86_400_000_000_000
DECISION = 6 * DAY_NS - 1_000_000
EXIT = DECISION + 7 * DAY_NS
BASE = CarryCostTable(
    name="base", perpetual_fee_bps_per_side=Decimal("5"), spot_fee_bps_per_side=Decimal("10"),
    slippage_bps_per_side_tier_one=Decimal("5"), slippage_bps_per_side_tier_two=Decimal("10"),
    funding_receipt_multiplier=Decimal("1"), funding_payment_multiplier=Decimal("1"),
    forced_close_multiplier=Decimal("1"),
)
ADVERSE = BASE.model_copy(update={
    "name": "adverse", "slippage_bps_per_side_tier_one": Decimal("10"),
    "slippage_bps_per_side_tier_two": Decimal("20"), "funding_receipt_multiplier": Decimal("0.75"),
    "funding_payment_multiplier": Decimal("2"), "forced_close_multiplier": Decimal("2"),
})


def history(leg: str, entry: str, exit_: str) -> ContractHistory:
    closes = {DECISION: Decimal(entry), EXIT: Decimal(exit_)}
    return ContractHistory(contract_id=leg, instrument_id=leg, closes=closes,
                           quote_volumes={k: Decimal(1) for k in closes}, close_times=tuple(sorted(closes)))


HIST = {"spot:A:7": history("spot:A:7", "100", "110"), "perp:A:0": history("perp:A:0", "100", "112")}
LEGS = (("perp:A:0", Decimal("-0.5")), ("spot:A:7", Decimal("0.5")))
TIERS = {"perp:A:0": 1, "spot:A:7": 1}
PAIR_OF = {"perp:A:0": "A:0", "spot:A:7": "A:0"}
FUNDING = {"perp:A:0": (FundingEvent(contract_id="A:0", instrument_id="A", calc_time_ns=DECISION + DAY_NS, rate=Decimal("0.001")),)}


def run(table: CarryCostTable):
    return evaluate_carry_episode(
        sample_id="s", member="carry_l1w_h4w", decision_close_ns=DECISION, holding_days=7,
        leg_weights=LEGS, previous_leg_weights=(), histories=HIST, tiers=TIERS,
        funding_by_leg=FUNDING, cost_table=table, pair_of_leg=PAIR_OF,
    )


def test_base_arithmetic_by_hand() -> None:
    episode = run(BASE)
    r = episode.result
    # basis: spot +10% on 0.5, perp +12% on -0.5 -> 0.05 - 0.06 = -0.01
    assert episode.basis_pnl == Decimal("-0.01")
    assert r.gross_return == Decimal("-0.01")
    # funding: short perp collects 0.5 * 0.001
    assert episode.funding_collected == Decimal("0.0005")
    # costs: spot 0.5 turnover at (10+5) bps = 0.00075, perp 0.5 at (5+5) = 0.0005
    assert episode.spot_trading_cost == Decimal("0.00075")
    assert episode.perpetual_trading_cost == Decimal("0.0005")
    assert r.net_return == Decimal("-0.01") + Decimal("0.0005") - Decimal("0.00125")
    # attribution is per pair and sums to net
    assert dict(r.contract_net_contributions) == {"A:0": r.net_return}
    assert r.gross_exposure == Decimal(1) and r.net_exposure == Decimal(0)


def test_adverse_haircuts_receipts_and_doubles_slippage() -> None:
    episode = run(ADVERSE)
    assert episode.funding_collected == Decimal("0.000375")
    assert episode.spot_trading_cost == Decimal("0.001")
    assert episode.perpetual_trading_cost == Decimal("0.00075")


def test_forced_leg_is_closed_and_its_partner_keeps_drifting() -> None:
    """The panel accounting force-closes the leg without an exit bar; the
    partner leg is closed by the fold runner at the next rebalance (Task 7),
    so here it still carries a drifted weight."""
    hist = dict(HIST)
    hist["perp:A:0"] = ContractHistory(contract_id="perp:A:0", instrument_id="perp:A:0",
                                       closes={DECISION: Decimal("100"), EXIT - DAY_NS: Decimal("105")},
                                       quote_volumes={}, close_times=(DECISION, EXIT - DAY_NS))
    episode = evaluate_carry_episode(
        sample_id="s", member="carry_l1w_h4w", decision_close_ns=DECISION, holding_days=7,
        leg_weights=LEGS, previous_leg_weights=(), histories=hist, tiers=TIERS,
        funding_by_leg={}, cost_table=BASE, pair_of_leg=PAIR_OF,
    )
    r = episode.result
    assert r.forced_close_count == 1
    drifted = dict(r.drifted_weights)
    assert drifted["perp:A:0"] == Decimal(0)
    assert drifted["spot:A:7"] > 0
    # forced close of the perp leg at the last close 105: 0.5 * 10 bps * multiplier 1
    assert r.forced_close_cost == Decimal("0.0005")
    # pair attribution still sums to net
    assert dict(r.contract_net_contributions) == {"A:0": r.net_return}


def test_previous_spot_leg_exit_is_charged_the_spot_fee() -> None:
    """A spot leg present only in the previous weights is still a spot leg."""
    episode = evaluate_carry_episode(
        sample_id="s", member="carry_l1w_h4w", decision_close_ns=DECISION, holding_days=7,
        leg_weights=(), previous_leg_weights=LEGS, histories=HIST, tiers=TIERS,
        funding_by_leg={}, cost_table=BASE, pair_of_leg=PAIR_OF,
    )
    assert episode.spot_trading_cost == Decimal("0.00075")
    assert episode.perpetual_trading_cost == Decimal("0.0005")
    assert episode.result.trading_cost == Decimal("0.00125")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_carry_accounting.py tests/test_panel_accounting.py::test_fee_overrides_apply_per_contract -v --basetemp=C:/Users/User/AppData/Local/Temp/pytest-carry`
Expected: FAIL (`TypeError` on `fee_overrides`, `ModuleNotFoundError` for `carry_accounting`)

- [ ] **Step 3: Add `fee_overrides` to `evaluate_episode`**

In `src/trading_bot/panel_accounting.py`, add the parameter after `cost_table`:

```python
    cost_table: PanelCostTable,
    fee_overrides: dict[str, Decimal] | None = None,
) -> EpisodeResult:
```

Change `_per_side_bps` and its two call sites:

```python
def _per_side_bps(cost_table: PanelCostTable, tier: int, fee: Decimal | None = None) -> Decimal:
    slippage = (
        cost_table.slippage_bps_per_side_tier_one
        if tier == 1
        else cost_table.slippage_bps_per_side_tier_two
    )
    return (cost_table.fee_bps_per_side if fee is None else fee) + slippage
```

In the turnover loop and the forced-close loop, replace `_per_side_bps(cost_table, tiers.get(contract_id, 2))` with `_per_side_bps(cost_table, tiers.get(contract_id, 2), overrides.get(contract_id))` where `overrides = fee_overrides or {}` is bound once at the top of the function.

- [ ] **Step 4: Write the carry wrapper**

```python
# src/trading_bot/carry_accounting.py
"""Two-leg carry episodes on top of the panel accounting."""

from dataclasses import dataclass
from decimal import Decimal

from trading_bot.carry_config import CarryCostTable
from trading_bot.panel_accounting import EpisodeResult, evaluate_episode
from trading_bot.panel_config import PanelCostTable
from trading_bot.panel_reader import FundingEvent
from trading_bot.panel_universe import ContractHistory

_BPS = Decimal(10_000)


@dataclass(frozen=True, slots=True)
class CarryEpisode:
    result: EpisodeResult
    funding_collected: Decimal
    basis_pnl: Decimal
    spot_trading_cost: Decimal
    perpetual_trading_cost: Decimal


def evaluate_carry_episode(
    *,
    sample_id: str,
    member: str,
    decision_close_ns: int,
    holding_days: int,
    leg_weights: tuple[tuple[str, Decimal], ...],
    previous_leg_weights: tuple[tuple[str, Decimal], ...],
    histories: dict[str, ContractHistory],
    tiers: dict[str, int],
    funding_by_leg: dict[str, tuple[FundingEvent, ...]],
    cost_table: CarryCostTable,
    pair_of_leg: dict[str, str],
) -> CarryEpisode:
    """One weekly episode of a long-spot short-perpetual book.

    Legs are ordinary contracts to the panel accounting: the perpetual leg is
    a negative weight, so a positive settlement is a receipt through the
    existing sign rule, and the spot leg carries its own fee through
    ``fee_overrides``. A leg that loses its exit bar is force-closed by the
    panel accounting; the fold runner then drops the pair from every retained
    cohort so the partner leaves at the next rebalance, because an unhedged
    leg is not the position under test. Per-leg attributions are
    re-aggregated to the pair so the concentration gate reads pairs.
    """
    panel_table = PanelCostTable(
        name=cost_table.name,
        fee_bps_per_side=cost_table.perpetual_fee_bps_per_side,
        slippage_bps_per_side_tier_one=cost_table.slippage_bps_per_side_tier_one,
        slippage_bps_per_side_tier_two=cost_table.slippage_bps_per_side_tier_two,
        funding_receipt_multiplier=cost_table.funding_receipt_multiplier,
        funding_payment_multiplier=cost_table.funding_payment_multiplier,
        forced_close_multiplier=cost_table.forced_close_multiplier,
    )
    overrides = {
        leg: cost_table.spot_fee_bps_per_side
        for leg, _ in (*leg_weights, *previous_leg_weights)
        if leg.startswith("spot:")
    }
    result = evaluate_episode(
        sample_id=sample_id,
        member=member,
        decision_close_ns=decision_close_ns,
        holding_days=holding_days,
        weights=leg_weights,
        previous_weights=previous_leg_weights,
        histories=histories,
        tiers=tiers,
        funding_by_contract=funding_by_leg,
        cost_table=panel_table,
        fee_overrides=overrides,
    )
    pair_net: dict[str, Decimal] = {}
    pair_gross: dict[str, Decimal] = {}
    for leg, value in result.contract_net_contributions:
        pair = pair_of_leg[leg]
        pair_net[pair] = pair_net.get(pair, Decimal(0)) + value
    for leg, value in result.contract_contributions:
        pair = pair_of_leg[leg]
        pair_gross[pair] = pair_gross.get(pair, Decimal(0)) + value
    aggregated = EpisodeResult(
        sample_id=result.sample_id,
        member=result.member,
        scenario=result.scenario,
        gross_return=result.gross_return,
        turnover=result.turnover,
        trading_cost=result.trading_cost,
        funding_cost=result.funding_cost,
        forced_close_cost=result.forced_close_cost,
        net_return=result.net_return,
        gross_exposure=result.gross_exposure,
        net_exposure=result.net_exposure,
        forced_close_count=result.forced_close_count,
        contract_contributions=tuple(sorted(pair_gross.items())),
        contract_net_contributions=tuple(sorted(pair_net.items())),
        drifted_weights=result.drifted_weights,
    )
    spot_cost, perp_cost = _leg_turnover_costs(leg_weights, previous_leg_weights, tiers, cost_table)
    return CarryEpisode(
        result=aggregated,
        funding_collected=-result.funding_cost,
        basis_pnl=result.gross_return,
        spot_trading_cost=spot_cost,
        perpetual_trading_cost=perp_cost,
    )


def _leg_turnover_costs(
    weights: tuple[tuple[str, Decimal], ...],
    previous: tuple[tuple[str, Decimal], ...],
    tiers: dict[str, int],
    table: CarryCostTable,
) -> tuple[Decimal, Decimal]:
    new = dict(weights)
    old = dict(previous)
    spot = Decimal(0)
    perp = Decimal(0)
    for leg in sorted(set(new) | set(old)):
        change = abs(new.get(leg, Decimal(0)) - old.get(leg, Decimal(0)))
        if change == 0:
            continue
        slippage = (
            table.slippage_bps_per_side_tier_one
            if tiers.get(leg, 2) == 1
            else table.slippage_bps_per_side_tier_two
        )
        if leg.startswith("spot:"):
            spot += change * (table.spot_fee_bps_per_side + slippage) / _BPS
        else:
            perp += change * (table.perpetual_fee_bps_per_side + slippage) / _BPS
    return spot, perp
```

The docstring's partner-close sentence is a promise Task 7 keeps: the fold runner removes a pair whose leg was force-closed from every retained cohort, so the surviving leg leaves the book at the next rebalance through ordinary turnover, which charges exactly one side as spec 8.3 requires. Do not try to close the partner inside this module; the panel accounting only zeroes a leg's drifted weight when that leg itself lacks a bar. `ContractHistory` is imported for the type of `histories` only.

- [ ] **Step 5: Run tests, lint and types**

```
uv run pytest tests/test_carry_accounting.py tests/test_panel_accounting.py -v --basetemp=C:/Users/User/AppData/Local/Temp/pytest-carry
uv run ruff check .
uv run mypy
```
Expected: all passing, clean.

- [ ] **Step 6: Commit**

```bash
git add src/trading_bot/panel_accounting.py src/trading_bot/carry_accounting.py tests/test_panel_accounting.py tests/test_carry_accounting.py
git commit -m "feat: account two-leg carry episodes with per-leg fees

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Task 6: Bind the hedge capture in the manifest

**Files:**
- Modify: `src/trading_bot/panel_samples.py`
- Modify: `src/trading_bot/cli.py` (`panel-manifest --hedge-capture`)
- Test: additions to `tests/test_panel_samples.py`

**Interfaces:**
- `publish_panel_walk_forward(capture_root, *, output_path, spec, family_spec_hash, hedge_capture_root: Path | None = None)` where `spec` is `PanelFamilySpec | CarryFamilySpec`. When a hedge capture is given it is verified, quality-gated with the same rule, and the manifest gains `hedge_capture_root_hash`, `hedge_dataset_root_hash`, `hedge_absent_at_source_days`.
- The module imports `load_family_spec` for the CLI and types `spec` as the union.

- [ ] **Step 1: Write the shared fixtures module**

`tests/` is a package (`tests/test_panel_end_to_end.py` already imports `tests.test_panel_fold_run`). Create `tests/carry_fixtures.py`; Tasks 7 and 9 import it too.

```python
# tests/carry_fixtures.py
"""Shared fixtures for the funding carry tests.

Two small captures over the twelve `SYMBOLS` and seven `MONTHS` of the
P1.27 fold-run fixture: a perpetual capture with weekly funding rows whose
rate rises with the symbol's seed (C11USDT pays negative funding), and a spot
capture whose closes sit one unit below the perpetual so the basis is
non-zero. `capture_panel` fills interior gaps from the daily dumps, so every
fake fetch answers a `/daily/klines/` request with `PanelSourceAbsent`.
"""

import json
from decimal import Decimal
from pathlib import Path

from tests.test_panel_fold_run import (
    DAY_MS,
    EPOCH_DAY_2020,
    MONTH_DAYS,
    MONTH_START_DAY,
    MONTHS,
    SYMBOLS,
    kline_csv,
    zip_bytes,
)
from trading_bot.panel_capture import PanelPayload, PanelSourceAbsent, capture_panel

NEGATIVE_FUNDING_SYMBOL = "C11USDT"
# Under `small_carry_config`, fold 0's test window holds the Sundays at day
# offsets 109, 116 and 123 (the P1.27 fixture confirmed this empirically).
# C10USDT pays the highest funding, so every member holds it; its perpetual
# lacks the three days 121..123, so the position entered on 116 has no exit
# bar on 123 and is force-closed, and the pair is ineligible on 123.
HOLE_SYMBOL = "C10USDT"
HOLE_DAY_OFFSETS = frozenset({121, 122, 123})


def funding_rate(symbol: str) -> str:
    seed = int(symbol[1:3])
    if symbol == NEGATIVE_FUNDING_SYMBOL:
        return "-0.0001"
    return str(Decimal("0.0001") * (seed + 1))


def funding_csv(symbol: str, month: str) -> str:
    text = "calc_time,funding_interval_hours,last_funding_rate\n"
    for offset in range(0, MONTH_DAYS[month], 7):
        day = EPOCH_DAY_2020 + MONTH_START_DAY[month] + offset
        text += f"{day * DAY_MS},8,{funding_rate(symbol)}\n"
    return text


def spot_kline_csv(symbol: str, month: str) -> str:
    """The perpetual's bars with every price one unit lower."""
    lines = kline_csv(symbol, month).splitlines()
    out = [lines[0]]
    for line in lines[1:]:
        fields = line.split(",")
        close = str(Decimal(fields[4]) - 1)
        fields[1] = fields[2] = fields[3] = fields[4] = close
        out.append(",".join(fields))
    return "\n".join(out) + "\n"


def _symbol_and_month(url: str) -> tuple[str, str]:
    symbol = next(item for item in SYMBOLS if f"/{item}/" in url or f"/{item}-" in url)
    month = next(item for item in MONTHS if item in url)
    return symbol, month


def _payload(url: str, name: str, text: str) -> PanelPayload:
    return PanelPayload(url=url, raw_bytes=zip_bytes(name, text), received_time_ns=1)


def perp_fetch(url: str) -> PanelPayload:
    if "/daily/klines/" in url:
        raise PanelSourceAbsent("404: no daily dump")
    symbol, month = _symbol_and_month(url)
    if "fundingRate" in url:
        return _payload(url, "f.csv", funding_csv(symbol, month))
    return _payload(url, "k.csv", kline_csv(symbol, month))


def perp_fetch_with_a_hole(url: str) -> PanelPayload:
    if "/daily/klines/" in url or "fundingRate" in url:
        return perp_fetch(url)
    symbol, month = _symbol_and_month(url)
    if symbol != HOLE_SYMBOL:
        return perp_fetch(url)
    lines = kline_csv(symbol, month).splitlines()
    kept = [lines[0]] + [
        line
        for line in lines[1:]
        if int(line.split(",")[0]) // DAY_MS - EPOCH_DAY_2020 not in HOLE_DAY_OFFSETS
    ]
    return _payload(url, "k.csv", "\n".join(kept) + "\n")


def spot_fetch(url: str) -> PanelPayload:
    assert "fundingRate" not in url, "the spot market has no funding"
    assert "/data/spot/" in url
    if "/daily/klines/" in url:
        raise PanelSourceAbsent("404: no daily dump")
    symbol, month = _symbol_and_month(url)
    return _payload(url, "k.csv", spot_kline_csv(symbol, month))


def spot_fetch_with_a_hole(url: str) -> PanelPayload:
    """C00USDT's spot bars lack day offsets 70..72, proven absent at source."""
    if "/daily/klines/" in url:
        raise PanelSourceAbsent("404: no daily dump")
    symbol, month = _symbol_and_month(url)
    if symbol != "C00USDT" or month != "2020-03":
        return spot_fetch(url)
    lines = spot_kline_csv(symbol, month).splitlines()
    kept = [lines[0]] + [
        line
        for line in lines[1:]
        if int(line.split(",")[0]) // DAY_MS - EPOCH_DAY_2020 not in {70, 71, 72}
    ]
    return _payload(url, "k.csv", "\n".join(kept) + "\n")


def build_captures(
    root: Path,
    *,
    perp_fetch_function=perp_fetch,
    spot_fetch_function=spot_fetch,
) -> tuple[Path, Path]:
    """Capture both markets under `root`; returns (perp_root, spot_root)."""
    perp = capture_panel(
        workspace_root=root, output_directory=root / "perp", reserve_bytes=0,
        symbols=SYMBOLS, months=MONTHS, fetch=perp_fetch_function,
    )
    spot = capture_panel(
        workspace_root=root, output_directory=root / "spot", reserve_bytes=0,
        symbols=SYMBOLS, months=MONTHS, fetch=spot_fetch_function, market="spot",
    )
    return perp.capture_root, spot.capture_root


def small_carry_config(tmp_path: Path) -> Path:
    """The frozen carry declaration with the P1.27 test fold geometry and a
    twelve-symbol pair list, so a seven-month fixture yields real folds."""
    document = json.loads(Path("configs/funding-carry-panel-v1.json").read_text(encoding="utf-8"))
    document["pairs"] = [{"perpetual": s, "spot": s, "multiplier": 1} for s in SYMBOLS]
    document["excluded_pairs"] = []
    document["universe"].update(
        {
            "minimum_history_days": 20,
            "liquidity_window_days": 5,
            "maximum_pairs": 10,
            "minimum_pairs": 6,
            "tier_one_rank_limit": 4,
        }
    )
    document["selection"]["minimum_selected"] = 2
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
    path = tmp_path / "small-carry.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path
```

Type the two `*_fetch_function` parameters as `Callable[[str], PanelPayload]` (import `Callable` from `collections.abc`) so mypy is clean.

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_panel_samples.py`:

```python
from tests.carry_fixtures import build_captures, small_carry_config, spot_fetch_with_a_hole
import trading_bot.panel_samples as panel_samples_module
from trading_bot.panel_config import load_family_spec


def test_manifest_binds_a_hedge_capture(tmp_path: Path) -> None:
    perp, spot = build_captures(tmp_path)
    spec, spec_hash = load_family_spec(small_carry_config(tmp_path))
    artifact = publish_panel_walk_forward(
        perp, output_path=tmp_path / "m.json", spec=spec,
        family_spec_hash=spec_hash, hedge_capture_root=spot,
    )
    manifest = json.loads(artifact.output_path.read_text(encoding="utf-8"))
    spot_manifest = json.loads((spot / "capture-manifest.json").read_text(encoding="utf-8"))
    spot_dataset = json.loads((spot / "dataset" / "dataset-manifest.json").read_text(encoding="utf-8"))
    assert manifest["hedge_capture_root_hash"] == spot_manifest["capture_root_hash"]
    assert manifest["hedge_dataset_root_hash"] == spot_dataset["root_hash"]
    assert manifest["hedge_absent_at_source_days"] == {}
    assert manifest["family_name"] == "funding_carry_panel_v1"
    assert manifest["hedge_capture_root_hash"] != manifest["capture_root_hash"]
    assert verify_panel_manifest(artifact.output_path)


def test_manifest_without_a_hedge_has_no_hedge_keys(tmp_path: Path) -> None:
    perp, _ = build_captures(tmp_path)
    spec, spec_hash = load_family_spec(small_carry_config(tmp_path))
    artifact = publish_panel_walk_forward(
        perp, output_path=tmp_path / "m.json", spec=spec, family_spec_hash=spec_hash,
    )
    manifest = json.loads(artifact.output_path.read_text(encoding="utf-8"))
    assert not any(key.startswith("hedge_") for key in manifest)


def test_hedge_capture_names_its_absent_at_source_days(tmp_path: Path) -> None:
    perp, spot = build_captures(tmp_path, spot_fetch_function=spot_fetch_with_a_hole)
    spec, spec_hash = load_family_spec(small_carry_config(tmp_path))
    artifact = publish_panel_walk_forward(
        perp, output_path=tmp_path / "m.json", spec=spec,
        family_spec_hash=spec_hash, hedge_capture_root=spot,
    )
    manifest = json.loads(artifact.output_path.read_text(encoding="utf-8"))
    assert set(manifest["hedge_absent_at_source_days"]) == {"C00USDT"}
    assert len(manifest["hedge_absent_at_source_days"]["C00USDT"]) == 3
    assert manifest["absent_at_source_days"] == {}


def test_hedge_capture_goes_through_the_same_quality_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    perp, spot = build_captures(tmp_path)
    spec, spec_hash = load_family_spec(small_carry_config(tmp_path))
    real_gate = panel_samples_module._enforce_capture_quality
    seen: list[Path] = []

    def gate(path: Path, manifest: dict[str, object], bars: object) -> dict[str, tuple[str, ...]]:
        seen.append(path)
        if path.is_relative_to(spot):
            raise PanelSamplesError("CAPTURE_QUALITY_FAILED: hedge capture")
        return real_gate(path, manifest, bars)  # type: ignore[arg-type]

    monkeypatch.setattr(panel_samples_module, "_enforce_capture_quality", gate)
    with pytest.raises(PanelSamplesError, match="CAPTURE_QUALITY_FAILED"):
        publish_panel_walk_forward(
            perp, output_path=tmp_path / "m.json", spec=spec,
            family_spec_hash=spec_hash, hedge_capture_root=spot,
        )
    assert seen == [
        perp / "dataset" / "quality-report.json",
        spot / "dataset" / "quality-report.json",
    ]
    assert not (tmp_path / "m.json").exists()
```

Match the `gate` stub's return annotation to the real `_enforce_capture_quality` signature when writing it (read the function; it returns the absent-at-source mapping).

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_panel_samples.py -k hedge -v --basetemp=C:/Users/User/AppData/Local/Temp/pytest-carry`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'hedge_capture_root'`

- [ ] **Step 4: Extend the publisher**

In `src/trading_bot/panel_samples.py`: import `load_family_spec` and `CarryFamilySpec`; change the signature and add the hedge block before `material`:

```python
def publish_panel_walk_forward(
    capture_root: Path,
    *,
    output_path: Path,
    spec: "PanelFamilySpec | CarryFamilySpec",
    family_spec_hash: str,
    hedge_capture_root: Path | None = None,
) -> PanelManifestArtifact:
    ...
    hedge: dict[str, object] = {}
    if hedge_capture_root is not None:
        hedge_valid, hedge_errors = verify_panel_capture(hedge_capture_root)
        if not hedge_valid:
            raise PanelSamplesError("hedge capture verification failed: " + ",".join(hedge_errors))
        hedge_manifest = _load_object(hedge_capture_root / "capture-manifest.json")
        hedge_dataset = _load_object(hedge_capture_root / "dataset" / "dataset-manifest.json")
        hedge_bars = load_panel_bars(hedge_capture_root / "dataset")
        hedge_absent = _enforce_capture_quality(
            hedge_capture_root / "dataset" / "quality-report.json", hedge_manifest, hedge_bars
        )
        hedge = {
            "hedge_capture_root_hash": str(hedge_manifest["capture_root_hash"]),
            "hedge_dataset_root_hash": str(hedge_dataset["root_hash"]),
            "hedge_absent_at_source_days": {
                instrument_id: list(days) for instrument_id, days in sorted(hedge_absent.items())
            },
        }
    material: dict[str, object] = {
        ...  # unchanged keys
        **hedge,
    }
```

Keep the P1.27 manifest byte-identical when no hedge is given: the three keys are absent, not null.

- [ ] **Step 5: CLI**

Add `panel_manifest.add_argument("--hedge-capture", type=Path, default=None)`; in the dispatch, include it in the workspace check when given, use `load_family_spec` instead of `load_panel_family_spec`, and pass `hedge_capture_root=`.

- [ ] **Step 6: Run tests, lint and types**

```
uv run pytest tests/test_panel_samples.py -v --basetemp=C:/Users/User/AppData/Local/Temp/pytest-carry
uv run ruff check .
uv run mypy
```
Expected: all passing, clean.

- [ ] **Step 7: Commit**

```bash
git add src/trading_bot/panel_samples.py src/trading_bot/cli.py tests/carry_fixtures.py tests/test_panel_samples.py
git commit -m "feat: bind a hedge capture in the walk-forward manifest

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Task 7: Carry fold runner

**Files:**
- Create: `src/trading_bot/carry_fold_run.py`
- Modify: `src/trading_bot/cli.py` (`carry-fold`)
- Test: `tests/test_carry_fold_run.py`

**Interfaces:**
- Consumes: everything above; `panel_fold_run.MEMBER_HELD_NOTHING_REASON_CODE`, `verify_panel_fold_report`; `registry.MetadataRegistry`, `ExperimentRecord`.
- Produces: `CarryFoldError`; `CarryFoldArtifact(output_path, report_hash, fold_index, episode_count, skipped_sample_count)`; `run_carry_fold(perp_capture_root, spot_capture_root, *, manifest_path, family_spec_path, output_path, registry_path, fold_index) -> CarryFoldArtifact`.

The report is the P1.27 schema (`panel_fold_run.py` lines 182–206 and `_scenario_record`) plus: top-level `hedge_capture_root_hash`, `hedge_dataset_root_hash`, `no_carry_cohort_sample_ids` per member; per-episode `extras` = `{"funding_collected", "basis_pnl", "spot_trading_cost", "perpetual_trading_cost"}` as canonical Decimal strings. `code_hash` covers `_CARRY_MODULES`.

Behaviour:
- Verify both captures and the manifest; require `family_spec_hash`, `capture_root_hash`, `dataset_root_hash`, `hedge_capture_root_hash`, `hedge_dataset_root_hash` to match.
- Load both bar sets with `available_before_ns = test_end_ns + 1`; build histories keyed by leg id (`perp:<contract_id>`, `spot:<contract_id>`); funding keyed by `perp:` leg id.
- Cohort state per member persists across decisions within the fold, starting empty (a fold starts flat, as P1.27). Drifted leg weights are carried per member and scenario.
- A `UNIVERSE_TOO_SMALL` week is skipped: sample id recorded, all cohorts and positions reset, `SKIPPED_WEEK_EXIT_COST_UNCHARGED` in the report. A `NO_CARRY_COHORT` week is not a skip: the book keeps its older cohorts and the sample id is listed under that member's `no_carry_cohort_sample_ids`.
- A member whose assembled book is empty is marked `MEMBER_HELD_NOTHING` for that episode, exactly as P1.27.
- **Partner close (spec 8.3):** after evaluating an episode, any pair with a force-closed leg (`drifted_weights` zero for that leg while its weight was non-zero) is removed from every retained cohort of that member, so its partner leg is closed at the next rebalance through ordinary turnover.
- The last-episode exit remains uncharged and is named `FOLD_FINAL_EXIT_COST_UNCHARGED`, as P1.27.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_carry_fold_run.py
import json
from decimal import Decimal
from pathlib import Path

import pytest

from tests.carry_fixtures import (
    HOLE_SYMBOL,
    build_captures,
    perp_fetch_with_a_hole,
    small_carry_config,
)
from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.carry_config import CONTROL_NAMES, MEMBER_NAMES
from trading_bot.carry_fold_run import CarryFoldError, run_carry_fold
from trading_bot.panel_config import load_family_spec
from trading_bot.panel_fold_run import verify_panel_fold_report
from trading_bot.panel_samples import publish_panel_walk_forward
from trading_bot.registry import MetadataRegistry

Workspace = tuple[Path, Path, Path, Path]  # root, perp capture, spot capture, config


def _publish(root: Path, perp: Path, spot: Path) -> Path:
    config_path = small_carry_config(root)
    spec, spec_hash = load_family_spec(config_path)
    publish_panel_walk_forward(
        perp, output_path=root / "manifest.json", spec=spec,
        family_spec_hash=spec_hash, hedge_capture_root=spot,
    )
    return config_path


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    perp, spot = build_captures(tmp_path)
    return tmp_path, perp, spot, _publish(tmp_path, perp, spot)


@pytest.fixture
def workspace_with_a_hole(tmp_path: Path) -> Workspace:
    perp, spot = build_captures(tmp_path, perp_fetch_function=perp_fetch_with_a_hole)
    return tmp_path, perp, spot, _publish(tmp_path, perp, spot)


def _run(space: Workspace, fold_index: int = 0) -> dict[str, object]:
    root, perp, spot, config_path = space
    artifact = run_carry_fold(
        perp, spot, manifest_path=root / "manifest.json", family_spec_path=config_path,
        output_path=root / f"fold{fold_index}.json", registry_path=root / "registry.sqlite3",
        fold_index=fold_index,
    )
    assert artifact.episode_count > 0
    assert verify_panel_fold_report(artifact.output_path)
    document: dict[str, object] = json.loads(artifact.output_path.read_text(encoding="utf-8"))
    return document


def _candidate(document: dict[str, object], name: str) -> dict[str, object]:
    candidates = document["candidates"]
    assert isinstance(candidates, list)
    return next(item for item in candidates if item["candidate_name"] == name)


def _episodes(document: dict[str, object], name: str, scenario: str = "base") -> list[dict[str, object]]:
    scenario_record = _candidate(document, name)[scenario]
    assert isinstance(scenario_record, dict)
    episodes = scenario_record["episodes"]
    assert isinstance(episodes, list)
    return episodes


def test_carry_fold_report_has_the_panel_schema_plus_extras(workspace: Workspace) -> None:
    document = _run(workspace)
    manifest = json.loads((workspace[0] / "manifest.json").read_text(encoding="utf-8"))
    assert document["status"] == "development_only"
    assert document["family_name"] == "funding_carry_panel_v1"
    assert document["hedge_capture_root_hash"] == manifest["hedge_capture_root_hash"]
    assert document["hedge_dataset_root_hash"] == manifest["hedge_dataset_root_hash"]
    assert document["capture_root_hash"] == manifest["capture_root_hash"]
    assert document["fold_index"] == 0 and document["fold_count"] == len(manifest["folds"])
    assert document["reason_codes"] == ["FOLD_FINAL_EXIT_COST_UNCHARGED"]
    names = [item["candidate_name"] for item in document["candidates"]]
    assert names == list(MEMBER_NAMES + CONTROL_NAMES)
    member = _candidate(document, "carry_l1w_h4w")
    assert member["role"] == "member" and member["no_carry_cohort_sample_ids"] == []
    assert _candidate(document, "no_trade")["role"] == "control"
    episode = _episodes(document, "carry_l1w_h4w")[0]
    assert set(episode) == {
        "sample_id", "net_return", "gross_return", "turnover", "trading_cost", "funding_cost",
        "forced_close_cost", "gross_exposure", "net_exposure", "forced_close_count",
        "reason_codes", "contract_net_contributions", "extras",
    }
    assert set(episode["extras"]) == {
        "funding_collected", "basis_pnl", "spot_trading_cost", "perpetual_trading_cost",
    }
    # per-pair attribution sums to the episode's net return exactly
    total = sum((Decimal(value) for _, value in episode["contract_net_contributions"]), Decimal(0))
    assert total == Decimal(episode["net_return"])
    # pair ids are perpetual contract ids, not leg ids
    assert all(not cid.startswith(("perp:", "spot:")) for cid, _ in episode["contract_net_contributions"])
    # the two legs of every pair are hedged: zero net exposure
    assert Decimal(episode["net_exposure"]) == 0


def test_no_trade_control_is_flat(workspace: Workspace) -> None:
    document = _run(workspace)
    for scenario in ("base", "adverse"):
        for episode in _episodes(document, "no_trade", scenario):
            assert Decimal(episode["net_return"]) == 0
            assert Decimal(episode["gross_exposure"]) == 0
            assert episode["reason_codes"] == []


def test_book_ramps_over_the_first_hold_weeks(workspace: Workspace) -> None:
    """A fold starts flat; with H = 4 the first three Sundays deploy 1/4, 2/4, 3/4."""
    document = _run(workspace)
    exposures = [Decimal(e["gross_exposure"]) for e in _episodes(document, "carry_l1w_h4w")]
    assert exposures == [Decimal("0.25"), Decimal("0.5"), Decimal("0.75")]
    # the funding leg collects: every member episode has positive funding_collected
    assert all(Decimal(e["extras"]["funding_collected"]) > 0 for e in _episodes(document, "carry_l1w_h4w"))
    # the adverse scenario haircuts receipts by a quarter
    base = _episodes(document, "carry_l1w_h4w", "base")
    adverse = _episodes(document, "carry_l1w_h4w", "adverse")
    for b, a in zip(base, adverse, strict=True):
        assert Decimal(a["extras"]["funding_collected"]) == Decimal(b["extras"]["funding_collected"]) * Decimal("0.75")


def test_all_pairs_control_excludes_negative_funding(workspace: Workspace) -> None:
    document = _run(workspace)
    episode = _episodes(document, "all_pairs_ew")[0]
    assert not any(cid.startswith("C11USDT:") for cid, _ in episode["contract_net_contributions"])


def test_forced_leg_closes_its_partner_at_the_next_rebalance(workspace_with_a_hole: Workspace) -> None:
    document = _run(workspace_with_a_hole)
    episodes = _episodes(document, "carry_l1w_h4w")
    assert len(episodes) == 3
    # week 2 (decision 116): the perpetual leg of the hole symbol has no exit bar
    assert episodes[1]["forced_close_count"] == 1
    assert Decimal(episodes[1]["forced_close_cost"]) > 0
    # week 3 (decision 123): the pair is out of the universe and out of the book, and its
    # spot leg is charged its exit through ordinary turnover under the pair's id
    assert any(cid.startswith(f"{HOLE_SYMBOL}:") for cid, _ in episodes[2]["contract_net_contributions"])
    assert episodes[2]["forced_close_count"] == 0
    assert Decimal(episodes[2]["gross_exposure"]) == Decimal("0.75")
    assert Decimal(episodes[2]["turnover"]) > Decimal("0.25")


def test_members_are_registered(workspace: Workspace) -> None:
    document = _run(workspace)
    manifest = json.loads((workspace[0] / "manifest.json").read_text(encoding="utf-8"))
    family_id = uuid5(NAMESPACE_URL, f"{manifest['split_manifest_hash']}:funding_carry_panel_v1")
    with MetadataRegistry(workspace[0] / "registry.sqlite3") as registry:
        records = registry.list_experiments(family_id)
    assert sorted(record.candidate_name for record in records) == sorted(MEMBER_NAMES)
    assert {record.result_hash for record in records} == {document["report_hash"]}
    assert {record.code_hash for record in records} == {document["code_hash"]}


def test_report_is_immutable(workspace: Workspace) -> None:
    _run(workspace)
    with pytest.raises(CarryFoldError, match="immutable"):
        _run(workspace)


def _rewrite_manifest_field(path: Path, key: str, value: object) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    document[key] = value
    material = {k: v for k, v in document.items() if k != "manifest_hash"}
    document["manifest_hash"] = content_sha256(material)
    path.write_bytes(canonical_json(document))


@pytest.mark.parametrize("key", ["hedge_capture_root_hash", "hedge_dataset_root_hash", "capture_root_hash"])
def test_linkage_to_both_captures_is_enforced(workspace: Workspace, key: str) -> None:
    _rewrite_manifest_field(workspace[0] / "manifest.json", key, "0" * 64)
    with pytest.raises(CarryFoldError, match=key):
        _run(workspace)


def test_family_spec_mismatch_is_rejected(workspace: Workspace) -> None:
    root, perp, spot, config_path = workspace
    document = json.loads(config_path.read_text(encoding="utf-8"))
    document["hypothesis"] += " (edited)"
    other = root / "other.json"
    other.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(CarryFoldError, match="declaration"):
        _run((root, perp, spot, other))
```

Add `from uuid import NAMESPACE_URL, uuid5` to the imports.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_carry_fold_run.py -v --basetemp=C:/Users/User/AppData/Local/Temp/pytest-carry`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading_bot.carry_fold_run'`

- [ ] **Step 3: Write the module**

```python
# src/trading_bot/carry_fold_run.py
"""Evaluate one walk-forward fold of the funding carry family."""

import json
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.carry_accounting import CarryEpisode, evaluate_carry_episode
from trading_bot.carry_config import CarryCostTable, CarryFamilySpec, load_carry_family_spec
from trading_bot.carry_signals import (
    Cohort,
    assemble_book,
    select_control_cohort,
    select_member_cohort,
    trailing_funding,
)
from trading_bot.carry_universe import select_pair_universe
from trading_bot.panel_capture import verify_panel_capture
from trading_bot.panel_fold_run import MEMBER_HELD_NOTHING_REASON_CODE
from trading_bot.panel_reader import FundingEvent, load_funding_events, load_panel_bars
from trading_bot.panel_samples import verify_panel_manifest
from trading_bot.panel_universe import ContractHistory, build_contract_histories
from trading_bot.registry import ExperimentRecord, MetadataRegistry

_CARRY_MODULES = (
    "panel_config.py", "panel_dataset.py", "panel_capture.py", "panel_reader.py",
    "panel_universe.py", "panel_accounting.py", "panel_samples.py",
    "carry_config.py", "carry_universe.py", "carry_signals.py", "carry_accounting.py",
    "carry_fold_run.py", "evaluation.py", "strategy.py",
)


class CarryFoldError(RuntimeError):
    """Raised when a carry fold cannot be evaluated or published."""


@dataclass(frozen=True, slots=True)
class CarryFoldArtifact:
    output_path: Path
    report_hash: str
    fold_index: int
    episode_count: int
    skipped_sample_count: int


def run_carry_fold(
    perp_capture_root: Path,
    spot_capture_root: Path,
    *,
    manifest_path: Path,
    family_spec_path: Path,
    output_path: Path,
    registry_path: Path,
    fold_index: int,
) -> CarryFoldArtifact:
    for root, label in ((perp_capture_root, "perpetual"), (spot_capture_root, "spot")):
        valid, errors = verify_panel_capture(root)
        if not valid:
            raise CarryFoldError(f"{label} capture verification failed: " + ",".join(errors))
    if not verify_panel_manifest(manifest_path):
        raise CarryFoldError("panel manifest verification failed")
    if output_path.exists():
        raise CarryFoldError("carry fold report already exists and is immutable")

    spec, family_spec_hash = load_carry_family_spec(family_spec_path)
    manifest = _load_object(manifest_path)
    if manifest.get("family_spec_hash") != family_spec_hash:
        raise CarryFoldError("family declaration does not match the manifest")
    _require_link(manifest, "capture_root_hash", _load_object(perp_capture_root / "capture-manifest.json")["capture_root_hash"])
    _require_link(manifest, "dataset_root_hash", _load_object(perp_capture_root / "dataset" / "dataset-manifest.json")["root_hash"])
    _require_link(manifest, "hedge_capture_root_hash", _load_object(spot_capture_root / "capture-manifest.json")["capture_root_hash"])
    _require_link(manifest, "hedge_dataset_root_hash", _load_object(spot_capture_root / "dataset" / "dataset-manifest.json")["root_hash"])

    folds = manifest.get("folds")
    if not isinstance(folds, list):
        raise CarryFoldError("manifest folds are malformed")
    fold = next((item for item in folds if item.get("fold_index") == fold_index), None)
    if fold is None:
        raise CarryFoldError(f"fold {fold_index} is not in the manifest")

    test_end_ns = int(fold["test_end_ns"])
    perp_bars = load_panel_bars(perp_capture_root / "dataset", available_before_ns=test_end_ns + 1)
    spot_bars = load_panel_bars(spot_capture_root / "dataset", available_before_ns=test_end_ns + 1)
    perp_histories = build_contract_histories(perp_bars)
    spot_histories = build_contract_histories(spot_bars)
    leg_histories: dict[str, ContractHistory] = {}
    for cid, history in perp_histories.items():
        leg_histories[f"perp:{cid}"] = history
    for cid, history in spot_histories.items():
        leg_histories[f"spot:{cid}"] = history
    funding_by_leg: dict[str, tuple[FundingEvent, ...]] = {}
    for event in load_funding_events(perp_capture_root / "dataset"):
        key = f"perp:{event.contract_id}"
        funding_by_leg[key] = funding_by_leg.get(key, ()) + (event,)

    test_ids = [str(value) for value in fold["test_ids"]]
    decisions = sorted(int(value.split(":")[1]) for value in test_ids)
    members = {m.name: m for m in spec.members}
    controls = {c.name: c for c in spec.controls}
    names = [m.name for m in spec.members] + [c.name for c in spec.controls]
    scenarios: tuple[CarryCostTable, ...] = (spec.costs.base, spec.costs.adverse)
    episodes: dict[tuple[str, str], list[CarryEpisode]] = {(n, s.name): [] for n in names for s in scenarios}
    held_nothing: dict[str, list[bool]] = {n: [] for n in names}
    cohorts: dict[str, list[Cohort]] = {n: [] for n in names}
    carried: dict[tuple[str, str], tuple[tuple[str, Decimal], ...]] = {k: () for k in episodes}
    no_carry: dict[str, list[str]] = {n: [] for n in names}
    skipped: list[str] = []
    # Fold-persistent: a leg keeps its pair after it leaves the universe, so an
    # exit-only leg can still be attributed to its pair.
    pair_of_leg: dict[str, str] = {}
    hold_of = {**{m.name: m.hold_weeks for m in spec.members}, **{c.name: members["carry_l1w_h4w"].hold_weeks for c in spec.controls}}
    lookback_of = {**{m.name: m.lookback_weeks for m in spec.members}, **{c.name: members["carry_l1w_h4w"].lookback_weeks for c in spec.controls}}

    for decision_close_ns in decisions:
        sample_id = f"BINANCE_UM:{decision_close_ns}:w1"
        snapshot = select_pair_universe(
            perp_histories, spot_histories, pairs=spec.pairs,
            decision_close_ns=decision_close_ns, rules=spec.universe,
        )
        if not snapshot.pairs:
            skipped.append(sample_id)
            for key in carried:
                carried[key] = ()
            for name in names:
                cohorts[name] = []
            continue
        # Tiers follow P1.27: this week's universe tier, tier two for a leg that
        # is only being exited (see panel_accounting's `tiers.get(id, 2)`).
        tiers: dict[str, int] = {}
        for pair in snapshot.pairs:
            perp_key = f"perp:{pair.perpetual_contract_id}"
            spot_key = f"spot:{pair.spot_contract_id}"
            tiers[perp_key] = pair.tier
            tiers[spot_key] = pair.tier
            pair_of_leg.setdefault(perp_key, pair.pair_id)
            pair_of_leg.setdefault(spot_key, pair.pair_id)
        for name in names:
            trailing = {
                pair.pair_id: trailing_funding(
                    funding_by_leg.get(f"perp:{pair.perpetual_contract_id}", ()),
                    decision_close_ns=decision_close_ns, lookback_weeks=lookback_of[name],
                )
                for pair in snapshot.pairs
            }
            if name in members:
                cohort = select_member_cohort(snapshot, trailing=trailing, selection=spec.selection)
            else:
                cohort = select_control_cohort(
                    snapshot, kind=controls[name].kind, trailing=trailing,
                    selection=spec.selection, random_seed=spec.statistics.random_seed,
                )
            if "NO_CARRY_COHORT" in cohort.reason_codes:
                no_carry[name].append(sample_id)
            cohorts[name].append(cohort)
            weights = assemble_book(tuple(cohorts[name]), hold_weeks=hold_of[name], decision_close_ns=decision_close_ns)
            held_nothing[name].append(name in members and not weights)
            forced_pairs: set[str] = set()
            for scenario in scenarios:
                key = (name, scenario.name)
                episode = evaluate_carry_episode(
                    sample_id=sample_id, member=name, decision_close_ns=decision_close_ns,
                    holding_days=spec.holding_days, leg_weights=weights, previous_leg_weights=carried[key],
                    histories=leg_histories, tiers=tiers, funding_by_leg=funding_by_leg,
                    cost_table=scenario, pair_of_leg=pair_of_leg,
                )
                episodes[key].append(episode)
                carried[key] = episode.result.drifted_weights
                drifted = dict(episode.result.drifted_weights)
                for leg, weight in weights:
                    if weight != 0 and drifted.get(leg, Decimal(0)) == 0:
                        forced_pairs.add(pair_of_leg[leg])
            if forced_pairs:
                cohorts[name] = [
                    Cohort(c.decision_close_ns, tuple(e for e in c.entries if e.pair_id not in forced_pairs), c.reason_codes)
                    for c in cohorts[name]
                ]
        # prune cohorts older than the longest hold so state stays bounded
        longest = max(hold_of.values()) * 604_800_000_000_000
        for name in names:
            cohorts[name] = [c for c in cohorts[name] if decision_close_ns - c.decision_close_ns < longest]

    episode_count = len(episodes[(names[0], "base")])
    reason_codes: list[str] = []
    if skipped:
        reason_codes.append("SKIPPED_WEEK_EXIT_COST_UNCHARGED")
    if episode_count == 0:
        reason_codes.append("NO_EPISODES_IN_FOLD")
    else:
        reason_codes.append("FOLD_FINAL_EXIT_COST_UNCHARGED")

    candidates = [
        {
            "candidate_name": name,
            "role": "member" if name in members else "control",
            "episode_count": len(episodes[(name, "base")]),
            "no_carry_cohort_sample_ids": no_carry[name],
            "base": _scenario_record(episodes[(name, "base")], held_nothing[name]),
            "adverse": _scenario_record(episodes[(name, "adverse")], held_nothing[name]),
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
        "hedge_capture_root_hash": str(manifest["hedge_capture_root_hash"]),
        "hedge_dataset_root_hash": str(manifest["hedge_dataset_root_hash"]),
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
    _register(spec, manifest, report_hash, registry_path)
    return CarryFoldArtifact(output_path, report_hash, fold_index, episode_count, len(skipped))


def _scenario_record(results: list[CarryEpisode], held_nothing: list[bool]) -> dict[str, object]:
    return {
        "total_net_return": sum((item.result.net_return for item in results), Decimal(0)),
        "episodes": [
            {
                "sample_id": item.result.sample_id,
                "net_return": item.result.net_return,
                "gross_return": item.result.gross_return,
                "turnover": item.result.turnover,
                "trading_cost": item.result.trading_cost,
                "funding_cost": item.result.funding_cost,
                "forced_close_cost": item.result.forced_close_cost,
                "gross_exposure": item.result.gross_exposure,
                "net_exposure": item.result.net_exposure,
                "forced_close_count": item.result.forced_close_count,
                "reason_codes": [MEMBER_HELD_NOTHING_REASON_CODE] if flag else [],
                "contract_net_contributions": [[cid, v] for cid, v in item.result.contract_net_contributions],
                "extras": {
                    "funding_collected": item.funding_collected,
                    "basis_pnl": item.basis_pnl,
                    "spot_trading_cost": item.spot_trading_cost,
                    "perpetual_trading_cost": item.perpetual_trading_cost,
                },
            }
            for item, flag in zip(results, held_nothing, strict=True)
        ],
    }


def _require_link(manifest: dict[str, object], key: str, expected: object) -> None:
    if manifest.get(key) != expected:
        raise CarryFoldError(f"manifest is not linked to this capture ({key})")


def _register(spec: CarryFamilySpec, manifest: dict[str, object], report_hash: str, registry_path: Path) -> None:
    split_hash = str(manifest["split_manifest_hash"])
    family_id = uuid5(NAMESPACE_URL, f"{split_hash}:{spec.family_name}")
    created_at_ns = time.time_ns()
    with MetadataRegistry(registry_path) as registry:
        for member in spec.members:
            registry.register_experiment(ExperimentRecord(
                experiment_id=uuid5(family_id, f"{member.name}:{spec.statistics.random_seed}:{report_hash}"),
                family_id=family_id, candidate_name=member.name, hypothesis=spec.hypothesis,
                split_manifest_hash=split_hash, code_hash=_code_hash(),
                random_seed=spec.statistics.random_seed, outcome="completed",
                result_hash=report_hash, failure_reason=None, created_at_ns=created_at_ns,
            ))


def _code_hash() -> str:
    root = Path(__file__).parent
    return content_sha256("".join((root / name).read_text(encoding="utf-8") for name in _CARRY_MODULES))


def _load_object(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise CarryFoldError(f"{path.name} must contain a JSON object")
    return document
```

Three details to get right when implementing: `MEMBER_HELD_NOTHING` applies to members only, as in P1.27; a retained cohort can hold a pair that has since left the universe, and its legs are then costed at tier two like P1.27's exit-only contracts, while `pair_of_leg` keeps their attribution; and the controls use `carry_l1w_h4w`'s lookback and hold, which is the plan's reading of spec section 4's "same size and hold as the members" for a control that has no lookback of its own.

- [ ] **Step 4: CLI**

```python
    carry_fold = commands.add_parser("carry-fold")
    carry_fold.add_argument("--workspace-root", type=Path, default=Path.cwd())
    carry_fold.add_argument("--capture", type=Path, required=True)
    carry_fold.add_argument("--hedge-capture", type=Path, required=True)
    carry_fold.add_argument("--manifest", type=Path, required=True)
    carry_fold.add_argument("--family-spec", type=Path, required=True)
    carry_fold.add_argument("--output", type=Path, required=True)
    carry_fold.add_argument("--registry", type=Path, required=True)
    carry_fold.add_argument("--fold-index", type=int, required=True)
```

Dispatch with its own local name `carry_paths`, the same workspace check, and a call to `run_carry_fold`.

- [ ] **Step 5: Run tests, lint and types**

```
uv run pytest tests/test_carry_fold_run.py -v --basetemp=C:/Users/User/AppData/Local/Temp/pytest-carry
uv run ruff check .
uv run mypy
```
Expected: 5 passed, clean.

- [ ] **Step 6: Commit**

```bash
git add src/trading_bot/carry_fold_run.py src/trading_bot/cli.py tests/test_carry_fold_run.py
git commit -m "feat: evaluate one funding carry walk-forward fold

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Task 8: Decision module: family loader, hedge linkage, extras

**Files:**
- Modify: `src/trading_bot/panel_decision.py`
- Test: additions to `tests/test_panel_decision.py`

**Interfaces:**
- `build_panel_decision` unchanged in signature; loads the family through `load_family_spec`.
- Linkage: if any fold report carries `hedge_capture_root_hash`, every report must, and all must agree on `(hedge_capture_root_hash, hedge_dataset_root_hash)`; the decision report copies both.
- Extras: if a candidate's episodes carry an `extras` mapping, the decision report's member and control records gain `extras_mean`: per key, the mean over the candidate's **retained** episodes, computed under `_POOLING_CONTEXT`. P1.27 reports carry no extras and the field is absent for them.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_panel_decision.py`. It already imports `json`, `Decimal`, `Path`, `pytest`, `canonical_json`, `content_sha256`, `PanelDecisionError`, `build_panel_decision`, `MEMBER_HELD_NOTHING_REASON_CODE`, and defines `SPEC_PATH`, `SPEC`, `SPEC_HASH`, `MEMBERS`, `CONTROLS`, `EPISODES_PER_FOLD`, `episodes(...)` and `write_fold(...)`:

```python
from collections.abc import Callable

from trading_bot.carry_config import CONTROL_NAMES as CARRY_CONTROLS
from trading_bot.carry_config import MEMBER_NAMES as CARRY_MEMBERS
from trading_bot.carry_config import load_carry_family_spec

CARRY_SPEC_PATH = Path("configs/funding-carry-panel-v1.json")
CARRY_SPEC, CARRY_SPEC_HASH = load_carry_family_spec(CARRY_SPEC_PATH)
EXTRAS = {
    "funding_collected": "0.001",
    "basis_pnl": "-0.0005",
    "spot_trading_cost": "0.0002",
    "perpetual_trading_cost": "0.0001",
}


def _rewrite_report(path: Path, mutate: Callable[[dict[str, object]], None]) -> None:
    """Mutate a fold report in place and re-derive its report_hash."""
    document = json.loads(path.read_text(encoding="utf-8"))
    mutate(document)
    material = {key: value for key, value in document.items() if key != "report_hash"}
    document["report_hash"] = content_sha256(material)
    path.write_bytes(canonical_json(document))


def _six_folds(tmp_path: Path) -> list[Path]:
    paths = []
    for fold_index in range(6):
        path = tmp_path / f"fold{fold_index}.json"
        write_fold(path, fold_index, {})
        paths.append(path)
    return paths


def _decide(tmp_path: Path, paths: list[Path], spec_path: Path = SPEC_PATH) -> dict[str, object]:
    artifact = build_panel_decision(
        tuple(paths),
        family_spec_path=spec_path,
        output_path=tmp_path / "decision.json",
        registry_path=tmp_path / "registry.sqlite3",
    )
    document: dict[str, object] = json.loads(artifact.output_path.read_text(encoding="utf-8"))
    return document


def _add_hedge(capture_hash: str) -> Callable[[dict[str, object]], None]:
    def mutate(document: dict[str, object]) -> None:
        document["hedge_capture_root_hash"] = capture_hash
        document["hedge_dataset_root_hash"] = "c" * 64

    return mutate


def _member_record(decision: dict[str, object], name: str) -> dict[str, object]:
    members = decision["members"]
    assert isinstance(members, list)
    return next(item for item in members if item["candidate_name"] == name)


def test_hedge_linkage_must_agree_across_folds(tmp_path: Path) -> None:
    paths = _six_folds(tmp_path)
    for index, path in enumerate(paths):
        _rewrite_report(path, _add_hedge(("a" if index < 5 else "b") * 64))
    with pytest.raises(PanelDecisionError, match="hedge"):
        _decide(tmp_path, paths)


def test_hedge_linkage_must_be_present_on_every_fold_or_none(tmp_path: Path) -> None:
    paths = _six_folds(tmp_path)
    _rewrite_report(paths[2], _add_hedge("a" * 64))
    with pytest.raises(PanelDecisionError, match="hedge"):
        _decide(tmp_path, paths)


def test_hedge_linkage_is_copied_when_it_agrees(tmp_path: Path) -> None:
    paths = _six_folds(tmp_path)
    for path in paths:
        _rewrite_report(path, _add_hedge("a" * 64))
    decision = _decide(tmp_path, paths)
    assert decision["hedge_capture_root_hash"] == "a" * 64
    assert decision["hedge_dataset_root_hash"] == "c" * 64


def test_panel_reports_without_hedge_or_extras_are_unchanged(tmp_path: Path) -> None:
    decision = _decide(tmp_path, _six_folds(tmp_path))
    assert not any(key.startswith("hedge_") for key in decision)
    assert "extras_mean" not in _member_record(decision, MEMBERS[0])


def test_extras_are_averaged_over_retained_base_episodes(tmp_path: Path) -> None:
    paths = _six_folds(tmp_path)
    member = MEMBERS[0]

    def add_extras(document: dict[str, object]) -> None:
        candidates = document["candidates"]
        assert isinstance(candidates, list)
        for candidate in candidates:
            if candidate["candidate_name"] != member:
                continue
            for scenario in ("base", "adverse"):
                for index, episode in enumerate(candidate[scenario]["episodes"]):
                    episode["extras"] = dict(EXTRAS)
                    if index == 0:
                        # a held-nothing episode is excluded from the mean, so its
                        # absurd extras value must leave no trace
                        episode["reason_codes"] = [MEMBER_HELD_NOTHING_REASON_CODE]
                        episode["extras"]["funding_collected"] = "9"

    for path in paths:
        _rewrite_report(path, add_extras)
    decision = _decide(tmp_path, paths)
    record = _member_record(decision, member)
    assert record["extras_mean"] == EXTRAS
    assert "extras_mean" not in _member_record(decision, MEMBERS[1])


def write_carry_fold(path: Path, fold_index: int) -> None:
    candidates = []
    for name in CARRY_MEMBERS + CARRY_CONTROLS:
        value = "0.001" if name in CARRY_MEMBERS else "0"
        rows = episodes(f"f{fold_index}", value, EPISODES_PER_FOLD)
        for row in rows:
            row["extras"] = dict(EXTRAS)
        candidates.append(
            {
                "candidate_name": name,
                "role": "member" if name in CARRY_MEMBERS else "control",
                "episode_count": EPISODES_PER_FOLD,
                "no_carry_cohort_sample_ids": [],
                "base": {
                    "total_net_return": str(Decimal(value) * EPISODES_PER_FOLD),
                    "episodes": rows,
                },
                "adverse": {
                    "total_net_return": str(Decimal(value) * EPISODES_PER_FOLD),
                    "episodes": [dict(row) for row in rows],
                },
            }
        )
    material = {
        "report_version": "1.0.0",
        "status": "development_only",
        "reason_codes": [],
        "family_name": CARRY_SPEC.family_name,
        "family_spec_hash": CARRY_SPEC_HASH,
        "capture_root_hash": "c" * 64,
        "dataset_root_hash": "d" * 64,
        "hedge_capture_root_hash": "a" * 64,
        "hedge_dataset_root_hash": "b" * 64,
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


def test_carry_family_is_pooled_through_the_same_gates(tmp_path: Path) -> None:
    paths = []
    for fold_index in range(6):
        path = tmp_path / f"carry{fold_index}.json"
        write_carry_fold(path, fold_index)
        paths.append(path)
    decision = _decide(tmp_path, paths, CARRY_SPEC_PATH)
    assert decision["family_name"] == "funding_carry_panel_v1"
    assert decision["family_spec_hash"] == CARRY_SPEC_HASH
    assert decision["decision_status"] in {"eligible_member_available", "no_eligible_member"}
    members = decision["members"]
    assert isinstance(members, list)
    assert [item["candidate_name"] for item in members] == list(CARRY_MEMBERS)
    assert all(item["extras_mean"] == EXTRAS for item in members)
    assert decision["hedge_capture_root_hash"] == "a" * 64


def test_carry_reports_are_rejected_against_the_panel_declaration(tmp_path: Path) -> None:
    paths = []
    for fold_index in range(6):
        path = tmp_path / f"carry{fold_index}.json"
        write_carry_fold(path, fold_index)
        paths.append(path)
    with pytest.raises(PanelDecisionError):
        _decide(tmp_path, paths, SPEC_PATH)
```

Constant episodes of `0.001` give a zero-variance bootstrap; if the decision module raises on a degenerate series rather than deciding, use `gaussian_episode_values` (already in the file) for the members' base and adverse rows instead, keeping the extras constant. The plan's assertion is only that the carry family reaches a terminal decision status through the unchanged gates.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_panel_decision.py -k "hedge or extras or carry_family" -v --basetemp=C:/Users/User/AppData/Local/Temp/pytest-carry`
Expected: FAIL (hedge test does not raise; extras field absent; carry config rejected by the panel loader)

- [ ] **Step 3: Implement**

In `panel_decision.py`:

```python
# import
from trading_bot.panel_config import PanelFamilySpec, load_family_spec
from trading_bot.carry_config import CarryFamilySpec   # for typing only if needed

# in build_panel_decision, replace load_panel_family_spec with load_family_spec

# after the code_hash agreement block:
    hedge_links = {
        (document.get("hedge_capture_root_hash"), document.get("hedge_dataset_root_hash"))
        for document in documents
    }
    if len(hedge_links) != 1:
        raise PanelDecisionError("fold reports do not agree on the hedge capture linkage")
    hedge_capture_root_hash, hedge_dataset_root_hash = next(iter(hedge_links))
```

and copy both into `material` only when they are not `None`. Presence must be all-or-none: a report set where some folds carry the keys and others do not is a linkage error too (the set above then holds two tuples, one with `None`s, and fails the same check).

In `_Pooled` add `base_extras_totals: dict[str, Decimal]` and `base_extras_count: int`. In `_pool`, where the base scenario's **retained** episodes are accumulated (skip every episode `_held_nothing` marks, exactly as `base_returns` does), read `episode.get("extras")`; when it is a dict, add each `Decimal(str(value))` into the totals under `_POOLING_CONTEXT` and increment the count. Reports without extras leave the count at zero. In `_member_record` and the control record, when `base_extras_count > 0`, add `"extras_mean": {key: total / Decimal(count) for key, total in sorted(totals.items())}` computed with ordinary Decimal division outside the trapped context. The spec's section 8.4 fields — funding collected, basis P&L, cost split by leg — are exactly those four keys' means.

`build_panel_decision` reads `spec.members`, `spec.controls`, `spec.statistics` and `spec.family_name` only (verified at plan time). If, while implementing, it turns out to touch any field absent on `CarryFamilySpec`, stop and report: the fix is a plan change, not a new field on the carry model.

- [ ] **Step 4: Run tests, lint and types**

```
uv run pytest tests/test_panel_decision.py -v --basetemp=C:/Users/User/AppData/Local/Temp/pytest-carry
uv run pytest -q --basetemp=C:/Users/User/AppData/Local/Temp/pytest-carry
uv run ruff check .
uv run mypy
```
Expected: all passing, full suite green, clean.

- [ ] **Step 5: Commit**

```bash
git add src/trading_bot/panel_decision.py tests/test_panel_decision.py
git commit -m "feat: pool carry fold reports through the panel decision gates

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Task 9: Backlog, README and the end-to-end chain

**Files:**
- Modify: `PHASE_1_BACKLOG.md` (add `### P1.28 Funding carry family` after P1.27)
- Modify: `README.md` (a short subsection after the P1.27 panel section)
- Create: `tests/test_carry_end_to_end.py`

- [ ] **Step 1: End-to-end test**

```python
# tests/test_carry_end_to_end.py
import json
from pathlib import Path
from unittest.mock import patch

from tests.carry_fixtures import perp_fetch, small_carry_config, spot_fetch
from tests.test_panel_fold_run import MONTHS, SYMBOLS
from trading_bot.cli import main

EXTRAS = {"funding_collected", "basis_pnl", "spot_trading_cost", "perpetual_trading_cost"}


def test_spot_capture_manifest_carry_folds_and_decision_compose(tmp_path: Path) -> None:
    config_path = small_carry_config(tmp_path)
    capture_arguments = [
        "--workspace-root", str(tmp_path), "--reserve-bytes", "0",
        "--symbols", ",".join(SYMBOLS), "--months", ",".join(MONTHS),
    ]
    with patch("trading_bot.panel_capture.PanelZipClient.fetch", side_effect=perp_fetch):
        assert main(["panel-capture", "--output", str(tmp_path / "perp"), *capture_arguments]) == 0
    with patch("trading_bot.panel_capture.PanelZipClient.fetch", side_effect=spot_fetch):
        assert main([
            "panel-capture", "--market", "spot", "--output", str(tmp_path / "spot"), *capture_arguments,
        ]) == 0
    spot_manifest = json.loads((tmp_path / "spot" / "capture-manifest.json").read_text(encoding="utf-8"))
    assert spot_manifest["market"] == "spot"

    assert main([
        "panel-manifest", "--workspace-root", str(tmp_path),
        "--capture", str(tmp_path / "perp"), "--hedge-capture", str(tmp_path / "spot"),
        "--output", str(tmp_path / "manifest.json"), "--family-spec", str(config_path),
    ]) == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["hedge_capture_root_hash"] == spot_manifest["capture_root_hash"]

    reports = []
    for fold in manifest["folds"]:
        output = tmp_path / f"fold{fold['fold_index']}.json"
        assert main([
            "carry-fold", "--workspace-root", str(tmp_path),
            "--capture", str(tmp_path / "perp"), "--hedge-capture", str(tmp_path / "spot"),
            "--manifest", str(tmp_path / "manifest.json"), "--family-spec", str(config_path),
            "--output", str(output), "--registry", str(tmp_path / "registry.sqlite3"),
            "--fold-index", str(fold["fold_index"]),
        ]) == 0
        reports.append(output)

    arguments = [
        "panel-decision", "--workspace-root", str(tmp_path), "--family-spec", str(config_path),
        "--output", str(tmp_path / "decision.json"), "--registry", str(tmp_path / "registry.sqlite3"),
    ]
    for report in reports:
        arguments.extend(["--fold-report", str(report)])
    assert main(arguments) == 0
    decision = json.loads((tmp_path / "decision.json").read_text(encoding="utf-8"))
    assert decision["status"] == "development_only"
    assert decision["decision_status"] in {"eligible_member_available", "no_eligible_member"}
    assert decision["fold_count"] == len(reports)
    assert len(decision["members"]) == 3
    assert decision["hedge_capture_root_hash"] == manifest["hedge_capture_root_hash"]
    assert decision["hedge_dataset_root_hash"] == manifest["hedge_dataset_root_hash"]
    assert all(set(member["extras_mean"]) == EXTRAS for member in decision["members"])
```

- [ ] **Step 2: Backlog entry**

```markdown
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
```

- [ ] **Step 3: README subsection**

````markdown
### Funding-Carry (P1.28)

Zweite Panel-Familie: long Spot, short Perpetual, ausgewählt nach dem zuletzt gezahlten
Funding, gehalten über überlappende Wochen-Kohorten. Das Spot-Bein kommt über
`panel-capture --market spot`, das Manifest bindet beide Captures, der Fold-Runner
schreibt das P1.27-Reportschema, und `panel-decision` bleibt dieselbe Instanz.

```powershell
uv run trading-research panel-capture --market spot --output data/captures/<datum>-binance-spot-usdt-1d --symbols <liste>
uv run trading-research panel-manifest --capture <perp> --hedge-capture <spot> --output artifacts/carry/carry-walk-forward-v1.json --family-spec configs/funding-carry-panel-v1.json
uv run trading-research carry-fold --capture <perp> --hedge-capture <spot> --manifest <manifest> --family-spec configs/funding-carry-panel-v1.json --output artifacts/carry/fold0.json --registry artifacts/carry/metadata-funding-carry-v1.sqlite3 --fold-index 0
uv run trading-research panel-decision --fold-report artifacts/carry/fold0.json --family-spec configs/funding-carry-panel-v1.json --output artifacts/carry/decision-v1.json --registry artifacts/carry/metadata-funding-carry-v1.sqlite3
```
````

- [ ] **Step 4: Full verification and commit**

```
uv run pytest -q --basetemp=C:/Users/User/AppData/Local/Temp/pytest-carry
uv run ruff check .
uv run mypy
```
Expected: every pre-existing test still passes plus the new ones; clean.

```bash
git add PHASE_1_BACKLOG.md README.md tests/test_carry_end_to_end.py
git commit -m "docs: register p1.28 and document the funding carry chain

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## After the plan

Not part of this plan, and to be run in this order per spec section 12: spot capture of the 470 declared spot symbols (discovery mode, no bounds, about five hours), repair it, build the manifest with the repaired P1.27 perpetual capture as `--capture` and the repaired spot capture as `--hedge-capture`, run all folds, build the decision, write `P1_28_DECISION_<date>.md`. The decision record must repeat the spec's section 2.2 contamination disclosure.

## Self-review

**Spec coverage.** 3.1 capital basis and cadence: Task 4 (`assemble_book`) and Task 7. 3.2 spot capture: Task 2. 3.3 costs incl. spot fee and the adverse funding rule: Tasks 1 and 5. 5 declaration: Task 1. 6 symbol mapping: Task 1's generated pair list. 7.1 pair eligibility incl. both-leg liquidity, exclusion of scaled pairs, funding-in-lookback (rule 4, applied at selection): Tasks 1, 3, 4. 7.2 identity: inherited from `panel_reader`. 8.1 cohorts: Task 4. 8.2 pair return: Task 5, where the panel accounting computes each leg's return and funding on the perpetual leg. 8.3 costs and partner close: Tasks 5 and 7. 8.4 extras: Tasks 7 and 8. 9 geometry and gates: Task 6 and the unchanged decision gates. 10 modules: as listed, with the declared loader deviation. 11 naming: Task 9 and the closing note. 12 sequencing: the closing note.

**Deliberate deviations, each stated where it occurs.** `load_family_spec` dispatch instead of an unchanged loader. A new `CarryCostTable` instead of extending `PanelCostTable`. `fee_overrides` added to `evaluate_episode`, which enters P1.27's `_PANEL_MODULES` code hash for any future P1.27 rerun; P1.27's artifacts bind their own hash and are unaffected. The partner of a force-closed leg is closed at the next rebalance rather than in the same episode, which charges exactly one side either way.

**Type consistency.** `EligiblePair` fields are used identically in Tasks 3, 4 and 7. `CohortEntry`/`Cohort` in Tasks 4 and 7. `CarryEpisode` in Tasks 5 and 7. `CarryCostTable` in Tasks 1, 5 and 7. `evaluate_episode(..., fee_overrides=)` in Tasks 5 and 7 via the wrapper. Leg ids `perp:<contract_id>` and `spot:<contract_id>` in Tasks 4, 5 and 7.
