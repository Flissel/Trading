# Shadow Book and Measurement Stream Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the weekly shadow captures (base capture + daily kline dumps + REST funding, reconciled against monthly dumps), the shadow-book runner over the existing fold loop, and the permanent Binance measurement journal, so the P1.33 release candidate's shadow phase can start the day after a confirmed holdout — without any order, credential or execution path.

**Architecture:** Three new modules beside the existing ones. `shadow_capture.py` builds an ordinary, self-contained panel capture from a base capture plus a fetched tail and seals it in the existing manifest format, so `verify_panel_capture`, the dataset publisher and `capture_lineage.verify_capture_superset` apply unchanged. `shadow_book.py` runs `carry_fold_run.evaluate_carry_decisions` from a fixed anchor Sunday and reads the final slot state through one small additive field on `DecisionRun`. `binance_measurement_journal.py` is a second, unbounded journal sharing the cost journal's fetcher, depth walk, throttle and atomic publish. One config, four CLI commands, fixture-only tests.

**Tech Stack:** Python 3.12, `Decimal`, pydantic v2 (where the neighbours use it) or frozen dataclasses (where they do), DuckDB/parquet via the existing dataset publisher, `urllib` only, pytest, `mypy --strict`, `ruff`.

**Spec:** `docs/superpowers/specs/2026-09-22-shadow-book-and-measurement-stream-design.md`; protocol section 13; backlog P1.24.

## Global Constraints

- Decimal everywhere; integer nanoseconds; immutable artifacts (write to `.tmp`, fsync, replace; refuse an existing target); frozen models; `content_sha256` sealing with the hash outside the material; no network in tests (fetchers injected, `Callable[[str], PanelPayload]` / `Callable[[str], Mapping | list]`); `uv run` inside the worktree; pytest `--basetemp=C:/Users/User/AppData/Local/Temp/pytest-shadow` with `TEMP`/`TMP` on C:; `ruff check` and `mypy --strict src tests` clean; commit trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Baseline `codex/phase1-foundation` at `21e9fc3` (728 tests). Worktree `.worktrees/shadow`, branch `codex/shadow-book`.
- **Reproducibility:** `run_carry_fold` output stays byte-identical (`tests/fixtures/carry_v1_fold0_expected.json`, all v2/v4 fold tests); the cost journal v1's models, spec hash and receipt are untouched; `verify_panel_capture` is not modified — a shadow capture must pass it as written.
- **Frozen by the spec:** anchor Sunday `2026-09-13` (the first Sunday after the holdout window's last exit; decisions before it are never evaluated); Phase A artifacts carry no P&L fields at all; Phase B requires the holdout artifact's `report_hash` in the config and a `holdout_confirmed` verdict in it; the weekly capture requires the Sunday bar for every symbol that has a Saturday bar, retrying hourly up to 24 h then refusing; a reconciliation mismatch refuses; measurement stream cadence 61 s with the all-symbol snapshots every fifth round; the cost tables of the family are never derived from the measurement stream.
- **Data kinds (new `sources` kinds):** `klines_daily_tail` (`month` field holds the date `YYYY-MM-DD`, as `klines_daily_fill` does), `fundingRate_rest` (`month` field holds `<from>_<to>` dates). `m` label: `+1` if the pair's trailing one-week funding at S exceeds the base round-trip cost of its tier spread over the member's hold (`round_trip_cost_bps(base_table, tier) / hold_weeks / 10_000`), `−1` if it is below zero, `0` otherwise.
- Never evaluate a member other than the release candidate and the two dominance controls on real data; the fixture families are the only ones the tests run.

## Review Focus

1. A symbol whose Sunday daily dump is still absent at 06:00 UTC Monday while its Saturday dump exists: the capture must wait and then refuse, never publish a capture with a silent hole at the decision — Task 1 tests the refusal after the deadline.
2. A REST funding response with a settlement outside the requested window or with a duplicate `fundingTime`: rows must be filtered to the window and duplicates refused — Task 1.
3. A base capture that is itself a weekly capture (tail rows already present) followed by a monthly capture that covers those months: reconciliation must compare every previously-tailed row and refuse on the first mismatch, and must pass when they agree — Task 2.
4. Phase A must not leak: a Phase A artifact must contain no `pnl`, no `running_totals`, and no per-episode net returns even in nested fields — Task 4 asserts the key set.
5. A measurement round in which one of the four requests fails (429, malformed JSON, non-array): the segment must still be published with the failure recorded, the chain must stay valid, and the next round must back off as the venue asked — Task 6.

## File Structure

| File | Responsibility |
| --- | --- |
| `src/trading_bot/shadow_capture.py` | weekly capture builder: base rows + tail (daily klines, REST funding), reconciliation, manifest sealing in the panel format |
| `src/trading_bot/carry_fold_run.py` (modify) | `DecisionRun.final_books` (additive), `FinalBook`/`SlotEntry` |
| `src/trading_bot/shadow_book.py` | anchor run, measurements per pair, `m` labels, Phase A/B artifact, registry |
| `src/trading_bot/shadow_config.py` | `ShadowDeclaration` model + loader (`configs/shadow-carry-v4.json`) |
| `src/trading_bot/binance_measurement_journal.py` | spec, round, chained segments (day-partitioned), run loop, verify, status, snapshot |
| `src/trading_bot/binance_cost_journal.py` (modify) | expose `public_binance_json_array_fetcher` beside the object fetcher; no other change |
| `src/trading_bot/cli.py` (modify) | `shadow-capture`, `shadow-week`, `binance-measurement-journal-create/run/status/snapshot` |
| `configs/shadow-carry-v4.json` | the shadow declaration |
| tests | `tests/test_shadow_capture.py`, `tests/test_carry_fold_run.py` (extend), `tests/test_shadow_book.py`, `tests/test_binance_measurement_journal.py`, `tests/shadow_fixtures.py` |
| `PHASE_1_BACKLOG.md`, `README.md` (modify) | P1.24 status, operating instructions |
| Operations (outside the repo, by the orchestrator) | `C:\Users\User\.trading-jobs\shadow-week.ps1` (Monday 06:00 UTC), `binance-measurement-journal.ps1` + Startup launcher |

---

## Task 1: Shadow capture — tail fetch and sealing

**Files:** create `src/trading_bot/shadow_capture.py`, `tests/shadow_fixtures.py`, `tests/test_shadow_capture.py`.

**Interfaces (consumes):** `panel_capture.PanelPayload`, `PanelSourceAbsent`, `PanelFetch`, `build_daily_kline_zip_url(symbol, date, *, market)`, `parse_kline_zip(payload, *, symbol, venue)`, `parse_funding_zip`, `panel_dataset.publish_panel_dataset(candles, funding, *, output_directory, raw_source_hashes)`, `panel_capture.verify_panel_capture`, `storage.StoragePolicy`, `canonical.content_sha256`.

**Interfaces (produces):**
```python
DAILY_TAIL_KIND = "klines_daily_tail"
FUNDING_REST_KIND = "fundingRate_rest"

def build_funding_rest_url(symbol: str, *, start_time_ms: int, end_time_ms: int) -> str:
    """https://fapi.binance.com/fapi/v1/fundingRate?symbol=…&startTime=…&endTime=…&limit=1000"""

def parse_funding_rest(payload: PanelPayload, *, symbol: str, start_time_ms: int, end_time_ms: int,
                       venue: str = "BINANCE_UM") -> tuple[PanelFundingRow, ...]:
    """Rows inside [start, end] only; funding_interval_hours from consecutive fundingTime deltas
    (the last row takes the previous delta; a single row takes 8); duplicate fundingTime → ShadowCaptureError."""

@dataclass(frozen=True, slots=True)
class ShadowCaptureArtifact:
    capture_root: Path; capture_root_hash: str; dataset_root_hash: str; tail_through: str
    reconciliation: dict[str, object]   # {"compared_rows": int, "mismatches": []} — Task 2 fills mismatches

def build_shadow_capture(*, workspace_root: Path, base_capture_root: Path, output_directory: Path,
                         reserve_bytes: int, tail_through: str, market: str, fetch: PanelFetch,
                         previous_capture_root: Path | None = None, clock=time.time_ns,
                         sleep=time.sleep, sunday_deadline_hours: int = 24) -> ShadowCaptureArtifact
```
Behaviour: verify the base (and the previous capture if given) with `verify_panel_capture`, else `ShadowCaptureError`. Copy every base raw payload into `raw/` (hardlink via `os.link`, falling back to a byte copy; `raw_sha256` re-verified after the copy) and carry its `sources` rows verbatim. From the previous capture carry only rows of kinds `klines_daily_tail`/`fundingRate_rest` whose date lies after the base's last covered month (rows now covered by a base month go to Task 2's reconciliation). Then fetch, per symbol, every date from the day after the last carried date (or the day after the base's last month) through `tail_through`: daily kline zips (absent → an `absent` row exactly as `capture_panel` writes them); for `market == "um"` one REST funding request per symbol over the same span. **Sunday rule:** if `tail_through` is a Sunday and a symbol has a bar for the Saturday but its Sunday dump is absent, retry the Sunday fetch every hour (`sleep(3600)`) until `sunday_deadline_hours` have passed by `clock`, then refuse with `SUNDAY_DUMP_MISSING:<symbol>`. Parse all rows (base rows re-parsed from their payloads with the existing parsers), `publish_panel_dataset` over the union, write the manifest with the panel material (`capture_version "1.0.0"`, `market`, `venue`, `interval "1d"`, `symbols` = the base's, `months` null, `discovered_months` = the base's extended per symbol with the tail months, `sources`, `dataset_root_hash`) plus `base_capture_root_hash`, `previous_capture_root_hash` (or null), `tail_through`, `reconciliation`, sealed as `capture_root_hash`; refuse an existing `capture-manifest.json`; storage policy authorised before any write.

`tests/shadow_fixtures.py`: `daily_kline_csv(symbol, date)` and `daily_fetch(url)` producing the same bars the monthly fixture would for that date (derive from `tests/test_panel_fold_run.py`'s generator so a daily row equals the monthly row); `funding_rest_json(symbol, start_ms, end_ms)` producing settlements consistent with `funding_csv`; a combined fetch that serves monthly, daily and REST URLs, with hooks to make a date absent, delay a Sunday, or return a duplicate settlement.

- [ ] Tests: a shadow capture over the fixture base (`build_captures`' perp capture) with `tail_through` = the Sunday after the base's last month verifies with `verify_panel_capture`, is a superset of the base per `verify_capture_superset` (True, no reasons), its dataset has the tail bars with `available_time_ns = close + 1`, its funding partition has the REST settlements, and the manifest carries the new keys; the daily rows equal the monthly fixture's rows for the same dates (parse both, compare fields); an absent date yields an `absent` row and the capture still publishes; a Sunday dump absent while Saturday exists waits (fake clock advances per sleep) and refuses after 24 h with the reason naming the symbol; a REST response with a settlement outside the window drops it; a duplicate `fundingTime` refuses; an unverifiable base refuses; an existing output refuses; a spot-market capture makes no funding requests (assert the fetch never saw a `fundingRate` URL).
- [ ] Implement; full suite, ruff, mypy; commit `feat: weekly shadow capture from a base capture, daily dumps and REST funding`.

## Task 2: Shadow capture — reconciliation against monthly dumps

**Files:** modify `src/trading_bot/shadow_capture.py`, `tests/test_shadow_capture.py`.

**Interfaces (produces):**
```python
def reconcile_tail_rows(*, previous_rows: tuple[PanelCandleRow | PanelFundingRow, ...],
                        base_rows: tuple[PanelCandleRow | PanelFundingRow, ...]) -> tuple[int, tuple[dict[str, object], ...]]:
    """(compared_count, mismatches). A candle matches on (instrument_id, open_time_ns) with equal
    close_time_ns, open, high, low, close, base_volume, quote_volume, trade_count; a funding row on
    (instrument_id, calc_time_ns) with equal rate. A previous row with no base counterpart is a
    mismatch of kind 'missing_in_base'."""
```
`build_shadow_capture` calls it for every previous-capture tail row whose month the base's monthly rows now cover; `reconciliation = {"compared_rows": n, "mismatches": [...]}` is written into the manifest; any mismatch refuses with `RECONCILIATION_MISMATCH` **after** writing a `reconciliation-refused.json` beside the (unpublished) output so a person can read it.

- [ ] Tests: base = a monthly fixture capture over `MONTHS`, previous = a shadow capture built (Task 1) on a base over `MONTHS[:-1]` with a tail through the end of `MONTHS[-1]`; reconciliation compares every tailed bar and settlement, passes with zero mismatches, and the new capture verifies; alter one tailed bar's close in the previous capture's payload (rebuild that capture from a fixture hook) → refuses with `RECONCILIATION_MISMATCH`, `reconciliation-refused.json` names the instrument, date and both values, and no `capture-manifest.json` exists; a previous funding settlement absent from the base's month → `missing_in_base`.
- [ ] Implement; full suite, ruff, mypy; commit `feat: reconcile shadow tail rows against the monthly dumps`.

## Task 3: Final slot state on `DecisionRun`

**Files:** modify `src/trading_bot/carry_fold_run.py`, `tests/test_carry_fold_run.py`.

**Interfaces (produces):**
```python
@dataclass(frozen=True, slots=True)
class SlotEntry:
    pair_id: str; perpetual_leg: str; spot_leg: str; tier: int
    entry_decision_close_ns: int; weeks_held: int      # (last decision − entry) // WEEK_NS

@dataclass(frozen=True, slots=True)
class FinalBook:
    decision_close_ns: int
    slots: tuple[SlotEntry, ...]                        # slot families only; () for cohort candidates
    leg_weights: tuple[tuple[str, Decimal], ...]        # the weights the last evaluated decision traded on

class DecisionRun:  # additive field, default {}
    final_books: dict[str, FinalBook]
```
`evaluate_carry_decisions` fills `final_books[name]` for every evaluated candidate from the `cohorts[name]` list and the last `weights` after the last decision; an empty `decisions` list yields `{}`. `run_carry_fold`'s `material` does not read the new field (the fold report is unchanged).

- [ ] Tests: the v1 fold-0 pin and every existing carry test pass unchanged; on the v4 fixture the exit member's `final_books` slots equal the pairs held in the last episode (derive from the last episode's non-zero `contract_net_contributions` legs), `weeks_held` counts from the entry Sunday, `leg_weights` sum to spot +½·filled/4 and perp −½·filled/4; a cohort family (v2 fixture) has `slots == ()` and its `leg_weights` equal `assemble_book` of its retained cohorts; an empty decision list gives `{}`.
- [ ] Implement; full suite, ruff, mypy; commit `feat: expose the final slot book of a decision run`.

## Task 4: Shadow declaration and the weekly book artifact

**Files:** create `src/trading_bot/shadow_config.py`, `src/trading_bot/shadow_book.py`, `configs/shadow-carry-v4.json`, `tests/test_shadow_book.py`; modify `tests/shadow_fixtures.py`.

**Interfaces (consumes):** Task 1's captures, Task 3's `FinalBook`, `carry_config.load_carry_family_spec`, `carry_universe.select_pair_universe`, `carry_signals.trailing_funding`, `carry_signals.round_trip_cost_bps`, `panel_samples.rebalance_close_times`, `panel_reader.load_panel_bars/load_funding_events`, `panel_universe.build_contract_histories`, `registry.MetadataRegistry/ArtifactRecord`.

**Interfaces (produces):**
```python
class ShadowDeclaration(_Frozen):            # shadow_config.py
    version: Literal["1.0.0"]; family_spec_path: str; family_spec_hash: str
    candidate: str; controls: tuple[str, str] = ("no_trade", "random_pairs")
    anchor_decision_close_date: str          # "2026-09-13"
    phase: Literal["A", "B"]
    holdout_report_hash: str | None          # required non-null when phase == "B"
    artifact_root: str; registry_path: str

def load_shadow_declaration(path: Path) -> tuple[ShadowDeclaration, str]

def run_shadow_week(*, workspace_root: Path, declaration_path: Path, perp_capture_root: Path,
                    spot_capture_root: Path, decision_sunday: str, holdout_report_path: Path | None,
                    measurement_snapshot_path: Path | None) -> ShadowWeekArtifact   # (output_path, report_hash, status)
```
Behaviour: verify both captures; both must be shadow captures (`tail_through` ≥ `decision_sunday`) or monthly captures covering it; the family spec at `family_spec_path` must hash to `family_spec_hash` and its `family_name` must be `funding_carry_panel_v4_measured`; `candidate` must be a member; Phase B requires `holdout_report_path`, whose document must recompute its `report_hash` equal to `holdout_report_hash`, carry `holdout: true` and `confirmation.verdict == "holdout_confirmed"` for the same `family_spec_hash` and `candidate_name`. Load bars with `available_before_ns = S_close + 2`, histories and funding as `run_carry_fold` does; decisions = `rebalance_close_times(perp bars)` filtered to `[anchor_close, S_close]`; refuse if S is not in them; run `evaluate_carry_decisions(spec, ..., decisions, candidate_names=(candidate, *controls))`. Assemble the artifact: `report_version`, `status` (`development_only` in A, `shadow` in B), `family_spec_hash`, `declaration_hash`, capture root and dataset hashes of both captures, `base_capture_root_hash` of each, `anchor_decision_close_ns`, `decision_close_ns`, `data_available_time_ns` (max `received_time_ns` over the tail rows), `code_hash` (carry modules + `shadow_book.py` + `shadow_capture.py`), `measurement_snapshot_hash` (or null), the **book** (Task 3's `FinalBook` of the candidate as plain data), the **universe measurements** (for every pair of the S snapshot: tier, trailing 1-week and 4-week funding, exit-rule verdict, and, when a snapshot is given, its mean premium, basis and spread), the **m labels** for every slot held at S−1 (definition in Global Constraints), and — Phase B only — `pnl` (the candidate's episode for S−1 under both scenarios: net, funding collected, trading cost, forced closes) and `running_totals` since the anchor. Sealed with `report_hash`; path `artifact_root/<family>/<S>.json`; refuse an existing path; registry record kind `shadow_week`, id `uuid5(NAMESPACE_URL, f"shadow:{family_spec_hash}:{S}")`, refusing a duplicate. A refusal after verification writes `artifact_root/<family>/<S>-refused.json` with the reason (unsealed) and raises.

`configs/shadow-carry-v4.json`: `family_spec_path configs/funding-carry-panel-v4-measured.json`, its hash `f75a298485f5e5865e568279418a6280f5880999f777d566efb05904463959bc`, candidate `carry_s10_l4w_h26w_exit`, anchor `2026-09-13`, phase `A`, `holdout_report_hash null`, `artifact_root artifacts/shadow`, `registry_path artifacts/shadow/metadata-shadow.sqlite3`.

- [ ] Tests (fixture: a shadow declaration over the v4 fixture family with anchor = the fixture's first in-window Sunday; two shadow captures from Task 1 as inputs): a Phase A week writes an artifact whose `book` equals the fold runner's `final_books` for the same decisions (run `evaluate_carry_decisions` in the test on the same inputs and compare), whose key set contains no `pnl` and no `running_totals` (assert recursively that no key named `net_return` appears), and registers one `shadow_week` record; a second run for the same Sunday refuses; a Sunday not in the capture refuses; a candidate not in the family refuses; Phase B without a holdout report refuses; Phase B with a fixture holdout report (built by `carry_holdout_run` on the fixture chain, or a hand-sealed document with `verdict holdout_failed`) refuses on the verdict and passes with `holdout_confirmed`, then carries `pnl` for S−1 equal to the runner's episode and `running_totals`; the `m` labels equal a hand computation from the fixture funding; the universe measurements list every ranked pair with its tier; the fold runner's pin still holds.
- [ ] Implement; full suite, ruff, mypy; commit `feat: weekly shadow book artifact over the carry decision loop`.

## Task 5: Measurement journal — spec, round, chained segments

**Files:** create `src/trading_bot/binance_measurement_journal.py`, `tests/test_binance_measurement_journal.py`; modify `src/trading_bot/binance_cost_journal.py` (add `public_binance_json_array_fetcher(url) -> list[object]`, same host and scheme checks as the object fetcher; nothing else).

**Interfaces (produces):**
```python
class BinanceMeasurementJournalSpec(BaseModel):   # frozen
    version: Literal["binance-measurement-journal/1.0.0"]; run_id: str; created_time_ns: int
    cost_journal_spec_hash: str                    # the v1 spec this continues
    instruments: tuple[JournalInstrument, ...]     # the 15 pairs' 30 instruments, copied from v1
    notionals: tuple[Decimal, ...]; depth_limit: int
    sample_interval_seconds: int                   # 61
    snapshot_every_rounds: int                     # 5
    spot_fee_bps_per_side: Decimal; perpetual_fee_bps_per_side: Decimal; fee_evidence_id: str
    spec_hash: str

class PremiumRow: symbol, mark_price, index_price, last_funding_rate, next_funding_time_ms, basis_bps
class BookRow: symbol, bid_price, bid_qty, ask_price, ask_qty, spread_bps
class MeasurementSegment: version, sequence, spec_hash, previous_segment_hash, received_time_ns,
    depth: tuple[InstrumentObservation, ...], premium_index: tuple[PremiumRow, ...] | None,
    perp_book: tuple[BookRow, ...] | None, spot_book: tuple[BookRow, ...] | None,
    failures: tuple[str, ...], content_hash

def create_measurement_journal(*, workspace_root, journal_root, reserve_bytes, run_id,
                               cost_journal_root: Path) -> Path      # copies instruments/notionals/fees from v1's spec
def run_measurement_journal(*, workspace_root, journal_root, reserve_bytes, rounds: int | None,
                            fetcher, array_fetcher, clock=time.time_ns, sleep=time.sleep) -> ChainHead
```
A round every `sample_interval_seconds`: the depth walk of the 30 instruments (reuse `depth_observation` and the v1 URL builders); on every `snapshot_every_rounds`-th round also `GET /fapi/v1/premiumIndex` (all), `GET /fapi/v1/ticker/bookTicker` (all), `GET /api/v3/ticker/bookTicker` (all), keeping only symbols ending in `USDT`. A failed request (throttle, malformed, non-array) is recorded in `failures` as `<endpoint>:<status or reason>` and the segment is still published; the throttle's `Retry-After` extends the next interval exactly as v1 does. Segments are published atomically (reuse `_publish`) under `segments/<YYYY-MM-DD>/<sequence:010d>.json`, chained by `previous_segment_hash`, with `chain-head.json` as in v1. `rounds=None` runs until the process is stopped.

- [ ] Tests (fake fetchers returning fixture depth, premium index, book tickers; `FakeClock`/`FakeSleep` as in the v1 tests): create copies v1's instruments, notionals and fees and writes a sealed spec; five rounds produce five chained segments, only the fifth carries the three all-symbol snapshots, non-USDT symbols are dropped, `basis_bps` and `spread_bps` equal hand computations; a 429 on the premium request yields a segment with `failures == ("premiumIndex:429",)`, depth still recorded, and the next round starts after the venue's `Retry-After`; a malformed book payload is a recorded failure, not an exception; segments land in day directories by `received_time_ns`; a spec change after creation is refused on run (spec hash bound).
- [ ] Implement; full suite, ruff, mypy; commit `feat: permanent binance measurement journal with chained segments`.

## Task 6: Measurement journal — verify, status, snapshot

**Files:** modify `src/trading_bot/binance_measurement_journal.py`, `tests/test_binance_measurement_journal.py`.

**Interfaces (produces):**
```python
def verify_measurement_journal(journal_root: Path) -> tuple[bool, tuple[str, ...]]
def measurement_status(journal_root: Path, *, last: int) -> MeasurementStatus   # segment count, newest age, failure rate per endpoint over the last N
def snapshot_measurement_journal(*, workspace_root, journal_root, output_path, reserve_bytes,
                                 window_start_ns: int, window_end_ns: int) -> Path
```
A snapshot seals, for the window: per USDT perpetual the mean and last `last_funding_rate`, mean `basis_bps`, mean `spread_bps` and the count of rounds observed; per spot pair the mean spread; per cost pair the tier medians (p50-of-p50 / p50-of-p90 at each notional, the v1 arithmetic) over the window; plus `chain_head_hash`, `first_sequence`, `last_sequence`, `spec_hash`, `content_hash`. It is immutable and never edits a declaration.

- [ ] Tests: verify passes on a fixture chain and fails on an edited segment (`SEGMENT_HASH_MISMATCH`), a broken link and a mismatched head; status reports the newest segment's age from the fake clock and the failure rate; a snapshot over rounds 2–7 equals hand computations (means over exactly the rounds inside the window), refuses an existing output, and refuses a window with no rounds.
- [ ] Implement; full suite, ruff, mypy; commit `feat: verify, status and windowed snapshots for the measurement journal`.

## Task 7: CLI, docs and operations text

**Files:** modify `src/trading_bot/cli.py`, `tests/test_cli_shadow.py` (new), `PHASE_1_BACKLOG.md`, `README.md`.

Commands (all paths bounded to `--workspace-root` as the existing commands do, refusals as non-zero exits):
- `shadow-capture --base-capture --output --tail-through --market {um,spot} [--previous-capture] [--reserve-bytes]`
- `shadow-week --declaration --capture --hedge-capture --decision-sunday [--holdout-report] [--measurement-snapshot]`
- `binance-measurement-journal-create --journal --run-id --cost-journal`
- `binance-measurement-journal-run --journal [--rounds] [--reserve-bytes]`
- `binance-measurement-journal-status --journal [--last]`
- `binance-measurement-journal-snapshot --journal --output --window-start --window-end`

Docs: `PHASE_1_BACKLOG.md` P1.24 gains a status paragraph (built 2026-09, Phase A, what Phase B requires); `README.md` gains a German section "Shadow-Buch und Messstrom (P1.24)" with the weekly Monday chain (`shadow-capture` ×2 → `shadow-week`), the journal commands, the Phase A/B rule and the sentence that no order and no key exists anywhere in this path.

- [ ] Tests: each command round-trips on the fixtures through `main([...])` (exit 0), refuses a path outside the workspace, and `shadow-week` in Phase A exits 0 with an artifact lacking `pnl`.
- [ ] Implement; full suite, ruff, mypy; commit `feat: shadow and measurement-journal commands; document P1.24`.

## After the plan

Orchestrator: merge into `codex/phase1-foundation`; create the measurement journal (`binance-measurement-journal-create` with the v1 journal as source) and start its supervisor with a Startup launcher; write `C:\Users\User\.trading-jobs\shadow-week.ps1` (Monday 06:00 UTC: two `shadow-capture`s from the latest base, then `shadow-week` in Phase A); add both to the daily liveness check. Switching `configs/shadow-carry-v4.json` to Phase B is a separate commit after `P1_33_HOLDOUT_<date>.md` records `holdout_confirmed`.
