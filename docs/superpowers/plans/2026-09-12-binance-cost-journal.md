# Binance Cost Journal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A resumable, hash-chained public order-book journal on Binance spot and USD-M perpetuals for a fixed sample of carry pairs, with a finalisation receipt that a later carry declaration can cite for measured slippage tiers.

**Architecture:** One new module, `src/trading_bot/binance_cost_journal.py`, with pydantic V1 models (spec, per-instrument observation, segment, chain head, receipt), pure functions for the book walk and the quantiles, a `create_journal` that derives the sample from the two carry captures, a `run_journal` loop with an injected fetcher, `verify_journal` and `finalize_journal`. Three CLI subcommands. A supervisor script mirrors the OKX journal's. Nothing else changes.

**Tech Stack:** Python 3.12, `Decimal`, pydantic v2, `urllib` (stdlib) for the fetcher, pytest with fake fetchers; `mypy --strict`, `ruff`.

**Spec:** `docs/superpowers/specs/2026-09-12-binance-cost-journal-design.md`.

## Global Constraints

- The v1 carry plan's Global Constraints bind (Decimal, integer ns, immutable artifacts via `.tmp` + `replace()`, `content_sha256` self-hashes, frozen models, no network in tests, `uv run` inside the worktree, `--basetemp=C:/Users/User/AppData/Local/Temp/pytest-carry`, ruff/mypy clean, commit trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`).
- Baseline: `codex/phase1-foundation` at `bbb15a1` (370 tests at e2939b8; the v2 branch is separate).
- Frozen numbers (spec 2–5): notionals `("500", "5000", "50000")` USDT; depth limit 500 on both markets; sample interval 61 s; target 10 000 rounds; eligibility ≥ 10 000 non-null observations at 5 000 USDT and ≥ 7 days span; sample = 16 pairs (ranks 1–8 and ranks 9, 12, 15, 19, 23, 27, 31, 35 of the pair universe at the latest supported decision); fees spot `10` / USD-M `5` bps per side with evidence id `BINANCE:fee-schedule:2026-09-11:standard-taker`; hosts `api.binance.com` and `fapi.binance.com` only; quantiles p50/p90/p99 with the OKX journal's `_quantile` rule (ceil(n·q)−1 index on the sorted values); receipt rule for a v3 declaration copied verbatim from spec 5.
- Storage: journal root under the workspace; storage policy authorised with `worst_case_required_bytes = 500_000_000` and the CLI's `--reserve-bytes` (default 10 GB for this journal — C: is at ~20 GB free).

## File Structure

| File | Responsibility |
| --- | --- |
| `src/trading_bot/binance_cost_journal.py` | models, book walk, sample derivation, run loop, verify, finalize |
| `src/trading_bot/cli.py` (modify) | `binance-cost-journal-create` / `-run` / `-finalize` |
| `tests/test_binance_cost_journal.py` | fake-fetcher tests for every function |
| `C:\Users\User\.trading-jobs\binance-cost-journal.ps1` + Startup `.cmd` | supervisor (not in the repo; written by the controller after merge) |

---

## Task 1: Models, book walk, observation

**Files:** create `src/trading_bot/binance_cost_journal.py`, `tests/test_binance_cost_journal.py`.

**Interfaces (all in the new module):**
```python
NOTIONALS: tuple[Decimal, ...] = (Decimal("500"), Decimal("5000"), Decimal("50000"))
DEPTH_LIMIT = 500
SAMPLE_INTERVAL_SECONDS = 61
TARGET_ROUNDS = 10_000
MINIMUM_OBSERVATIONS = 10_000
MINIMUM_SPAN_NS = 7 * 86_400_000_000_000
ALLOWED_HOSTS = frozenset({"api.binance.com", "fapi.binance.com"})

class JournalInstrument(_Frozen):   # pydantic frozen, extra=forbid
    instrument_id: str              # "spot:BTCUSDT" | "perp:BTCUSDT"
    market: Literal["spot", "um"]
    symbol: str
    pair_symbol: str                # the perpetual symbol naming the pair
    tier: Literal[1, 2]
    depth_url: str
    premium_index_url: str | None   # perpetuals only

class BinanceCostJournalSpec(_Frozen):
    version: Literal["binance-cost-journal/1.0.0"]
    run_id: str
    created_time_ns: int
    sample_decision_close_ns: int
    perpetual_capture_root_hash: str
    spot_capture_root_hash: str
    instruments: tuple[JournalInstrument, ...]
    notionals: tuple[Decimal, ...]
    depth_limit: int
    sample_interval_seconds: int
    target_rounds: int
    spot_fee_bps_per_side: Decimal
    perpetual_fee_bps_per_side: Decimal
    fee_evidence_id: str

class InstrumentObservation(_Frozen):
    instrument_id: str
    received_time_ns: int
    ok: bool
    reason: str | None                       # set when ok is False
    spread_bps: Decimal | None
    slippage_bps_per_side: dict[str, Decimal | None]   # keyed by the notional string, None = insufficient depth
    displayed_notional_thinner_side: Decimal | None
    funding_rate: Decimal | None            # perpetuals
    basis_bps: Decimal | None               # perpetuals: (mark - index) / index * 1e4

class JournalSegment(_Frozen):
    version: Literal["binance-cost-journal/1.0.0"]
    sequence: int
    spec_hash: str
    previous_segment_hash: str              # 64 zeros for sequence 0
    received_time_ns: int
    observations: tuple[InstrumentObservation, ...]
    content_hash: str

class ChainHead(_Frozen):
    version: Literal["binance-cost-journal/1.0.0"]
    spec_hash: str
    segment_count: int
    last_sequence: int
    final_segment_hash: str
    content_hash: str

def walk_notional(levels: Sequence[tuple[Decimal, Decimal]], notional: Decimal) -> Decimal | None:
    """VWAP paid to fill `notional` (quote units) from `levels` ordered best-first; None if the displayed book cannot fill it."""

def depth_observation(payload: Mapping[str, object], *, instrument: JournalInstrument, received_time_ns: int,
                      notionals: Sequence[Decimal], premium_index: Mapping[str, object] | None) -> InstrumentObservation:
    """Parse one depth payload (Binance REST shape: lastUpdateId, bids [[price, qty], ...], asks) and optional premiumIndex payload
    (markPrice, indexPrice, lastFundingRate) into an observation; validation failures become ok=False with a reason, never an exception."""
```
Book walk semantics (spec 2): mid = (best_bid + best_ask)/2; buy VWAP from asks ascending, sell VWAP from bids descending; per-notional slippage = max(buy_vwap/mid − 1, 1 − sell_vwap/mid) × 10 000; if either side cannot fill, that notional is `None`. `displayed_notional_thinner_side` = min over sides of Σ price × qty over the fetched levels. Spread = (ask − bid)/mid × 10 000. Levels must be positive Decimals parsed from strings; crossed or empty books → `ok=False, reason="crossed book"/"empty book"`.

- [ ] Tests (write first): hand-computed walk on a three-level book for all three notionals including one `None`; observation fields from a synthetic payload; crossed and empty books; premium index parsed into `basis_bps` and `funding_rate`; malformed payload → `ok=False` with reason, no exception; models reject extra fields.
- [ ] Implement; run focused tests, ruff, mypy; commit `feat: binance cost journal models and book walk`.

## Task 2: Journal creation, run loop, verification

**Files:** modify `src/trading_bot/binance_cost_journal.py`, `src/trading_bot/cli.py`; tests.

**Interfaces:**
```python
def derive_sample(perp_capture_root: Path, spot_capture_root: Path, *, family_spec_path: Path) -> tuple[int, tuple[JournalInstrument, ...]]:
    """Latest decision both captures support (min of the two last closes, stepped back one week as the universe needs a complete window),
    pair universe via carry_universe.select_pair_universe with the family's rules (minimum_pairs lowered to 1 for this derivation only,
    because the sample must exist even in a thin regime); ranks 1-8 and 9,12,15,19,23,27,31,35 by liquidity_rank; tier 1 for the first eight,
    tier 2 for the rest (the journal's tier, not the family's); returns (decision_close_ns, instruments) with both legs per pair, spot first."""

def create_journal(*, workspace_root: Path, journal_root: Path, reserve_bytes: int, run_id: str,
                   perp_capture_root: Path, spot_capture_root: Path, family_spec_path: Path) -> Path:
    """Writes <journal_root>/journal-spec.json (immutable; refuses to overwrite) and an empty segments/ directory."""

Fetcher = Callable[[str], Mapping[str, object]]     # url -> parsed JSON object; raises on transport failure

def run_journal(*, workspace_root: Path, journal_root: Path, reserve_bytes: int, rounds: int,
                fetcher: Fetcher, clock: Callable[[], int] = time.time_ns, sleep: Callable[[float], None] = time.sleep) -> ChainHead:
    """Resume from the chain head (verify it first), then append up to `rounds` segments; stops early when target_rounds is reached.
    Per instrument: fetch depth (and premium index for perps); a fetcher exception or invalid payload becomes ok=False with the reason;
    if every instrument in a round is ok=False raise BinanceCostJournalError. Segment written to segments/<sequence:010d>.json via .tmp,
    chain-head.json rewritten after each segment (via .tmp)."""

def verify_journal(journal_root: Path) -> tuple[bool, tuple[str, ...]]:
    """Recomputes every segment's content hash and the previous-hash links and the chain head; returns (ok, reasons)."""

def public_binance_json_fetcher(url: str) -> Mapping[str, object]:
    """HTTPS GET with a 20 s timeout; host must be in ALLOWED_HOSTS; raises BinanceCostJournalError on non-200, non-object JSON, or transport failure."""
```
CLI: `binance-cost-journal-create --workspace-root --journal --run-id --perp-capture --spot-capture --family-spec --reserve-bytes`; `binance-cost-journal-run --workspace-root --journal --rounds (default target) --reserve-bytes`; storage authorised like `panel-capture`. Exit code 2 on usage/spec mismatch (the supervisor stops on 2), 1 on other failures.

- [ ] Tests: `derive_sample` on the reduced carry fixtures from `tests/carry_fixtures.py` (build both captures; expect 12 pairs → with only 12 pairs the ranks beyond 12 are absent, so the sample is ranks 1–8 plus ranks 9 and 12 — assert exactly that and the tier split); `create_journal` writes the spec and refuses a second time; `run_journal` with a fake fetcher and a fake clock writes N segments with correct links, resumes after a restart (second call continues the sequence), records a failing instrument as `ok=False`, aborts when all fail; `verify_journal` detects a tampered segment; the fetcher rejects a foreign host without network (assert before any I/O).
- [ ] Implement; full suite, ruff, mypy; commit `feat: binance cost journal creation, run loop and verification`.

## Task 3: Finalisation receipt

**Files:** modify `src/trading_bot/binance_cost_journal.py`, `cli.py`; tests.

**Interfaces:**
```python
class InstrumentStatistics(_Frozen):
    instrument_id: str; tier: int; market: str; eligible: bool; reason_codes: tuple[str, ...]
    observation_count: int; first_time_ns: int | None; last_time_ns: int | None
    spread_bps_p50: Decimal | None
    slippage: dict[str, dict[str, Decimal | int | None]]   # notional -> {"count", "insufficient_depth", "p50", "p90", "p99"}
    basis_bps_p50: Decimal | None

class TierStatistics(_Frozen):
    tier: int; market: str; instrument_count: int
    slippage: dict[str, dict[str, Decimal | None]]         # notional -> {"p50_of_p50", "p50_of_p90"}

class FinalizationReceipt(_Frozen):
    version: Literal["binance-cost-journal-receipt/1.0.0"]
    spec_hash: str; chain_head_hash: str; segment_count: int
    spot_fee_bps_per_side: Decimal; perpetual_fee_bps_per_side: Decimal; fee_evidence_id: str
    instruments: tuple[InstrumentStatistics, ...]; tiers: tuple[TierStatistics, ...]
    declaration_rule: str      # spec section 5's sentence, verbatim
    content_hash: str

def finalize_journal(*, workspace_root: Path, journal_root: Path, output_path: Path, reserve_bytes: int) -> Path
```
Eligibility per instrument: `COST_OBSERVATION_FLOOR_NOT_MET` (< 10 000 non-null at "5000"), `COST_CAPTURE_SPAN_FLOOR_NOT_MET` (< 7 days). Tier statistics over eligible instruments only; a tier with none is emitted with `instrument_count 0` and `None`s. Quantile rule as the OKX journal. CLI `binance-cost-journal-finalize --workspace-root --journal --output --reserve-bytes`.

- [ ] Tests: quantiles by hand on a small journal built with the fake fetcher (use `MINIMUM_OBSERVATIONS`/`MINIMUM_SPAN_NS` monkeypatched down to make an eligible journal in a test, and one ineligible); receipt immutability; `declaration_rule` present verbatim.
- [ ] Implement; full suite, ruff, mypy; commit `feat: binance cost journal finalisation receipt`.

## After the plan

Controller: merge, `binance-cost-journal-create` against the two repaired captures, write the supervisor and Startup `.cmd`, launch, verify the first segments, record in memory. Finalisation after ≥ 7 days; `funding_carry_panel_v3` declared then.
