# Funding Carry v4 (declared capital, slot book) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pre-register `funding_carry_panel_v4` — a ten-slot spot/perpetual funding-carry book sized to a 10 000 USDT capital declaration — and make the Binance cost journal's receipt readable at that book's order notional, so the family can be evaluated as `funding_carry_panel_v4_measured` the moment the receipt exists.

**Architecture:** The carry line gains a second book mode beside weekly cohorts: slots, each holding one pair on `1/pair_slots` of capital, released on age-out, exit rule, forced close or lost bar, and refilled every decision from the same ranking that forms a cohort. The receipt gains a second, capital-keyed reading rule fixed in the journal spec's section 5.1; `carry_measured_costs` dispatches on the base declaration's `capital` block. P1.27/P1.28 accounting, universe, manifest and decision modules are untouched; the cohort path stays byte-identical (v1 fold-0 pin, v2 tests).

**Tech Stack:** Python 3.12, `Decimal`, pydantic v2, pytest, `mypy --strict`, `ruff`.

**Spec:** `docs/superpowers/specs/2026-09-16-funding-carry-v4-small-book-design.md` and the journal spec's section 5.1 (`docs/superpowers/specs/2026-09-12-binance-cost-journal-design.md`).

## Global Constraints

- The v1 carry plan's Global Constraints bind (Decimal, integer ns, immutable artifacts, frozen models, no network in tests, `uv run` inside the worktree, pytest `--basetemp=C:/Users/User/AppData/Local/Temp/pytest-carry-v4`, ruff/mypy clean, trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`).
- Baseline `codex/phase1-foundation` at the commit that carries this plan (640 tests). Worktree `.worktrees/carry-v4`, branch `codex/carry-v4`.
- **Reproducibility:** `tests/fixtures/carry_v1_fold0_expected.json` must still pass; v2 fold reports must keep exactly their eight extras keys; `funding_carry_panel_v3` declare/verify behaviour must be unchanged (existing tests untouched except for the receipt gaining the one new field).
- **Frozen numbers (spec 3–4):** `capital` = `book_usdt` "10000", `pair_slots` 10, `per_leg_notional_usdt` "500", `fee_tier` "standard_taker_no_bnb"; members `carry_s10_l4w_h13w` (L4, H13, exit off), `carry_s10_l4w_h13w_exit` (L4, H13, on), `carry_s10_l4w_h26w` (L4, H26, off), `carry_s10_l4w_h26w_exit` (L4, H26, on), all `book` "slots", no hurdle; controls P1.28's three (`random_pairs` as a slot book with the shortest member's H, `all_pairs_ew` unchanged in cohort mode with the shortest member's H); the base declaration is v2's JSON with `family_name`, `hypothesis` (spec 4.1 verbatim), `members`, `capital` changed and nothing else.
- **Rule (spec 5.1):** base slippage tier `t` = `tier_p50_of_p50` of the worse leg at `ladder(N)`, the smallest receipt notional ≥ `per_leg_notional_usdt`; adverse = `tier_p50_of_p90` at `10 × ladder(N)`; both rounded up to whole bps; refuse if either notional is absent from the receipt or a needed median is withheld. Section 5's rule (5 000 / 50 000) stays the reading for a base declaration without `capital`.
- The PowerShell chain script lives outside the repo and is written by the orchestrator, not by a task.

## File Structure

| File | Responsibility |
| --- | --- |
| `src/trading_bot/carry_config.py` (modify) | `CarryCapital`, `CarryMember.book`, v4 names, base-to-measured map, validators |
| `configs/funding-carry-panel-v4.json` | pre-registered base declaration |
| `src/trading_bot/cost_evidence_rule.py` (modify) | `CAPITAL_DECLARATION_RULE` |
| `src/trading_bot/binance_cost_journal.py` (modify) | receipt carries `capital_declaration_rule` |
| `src/trading_bot/carry_measured_costs.py` (modify) | capital-keyed reading, declare/verify dispatch |
| `src/trading_bot/carry_signals.py` (modify) | `rank_paying_pairs`, `fill_slots`, `assemble_slot_book` |
| `src/trading_bot/carry_fold_run.py` (modify) | slot mode, extras, `uncharged_final_exit_cost`, warm-up 0 |
| `tests/carry_fixtures.py` (modify) | `small_carry_v4_config` |
| tests | `tests/test_carry_config.py`, `tests/test_carry_measured_costs.py`, `tests/test_binance_cost_journal.py` (one field), `tests/test_carry_signals.py`, `tests/test_carry_fold_run.py`, `tests/test_carry_end_to_end.py` (extend the existing files where they exist) |
| `PHASE_1_BACKLOG.md`, `README.md` (modify) | P1.33 entry, chain documentation |

---

## Task 1: Declaration models and the v4 base declaration

**Files:** modify `src/trading_bot/carry_config.py`, `tests/test_carry_config.py` (or the carry config tests where they live), `tests/carry_fixtures.py`; create `configs/funding-carry-panel-v4.json` by script from `configs/funding-carry-panel-v2.json`.

**Interfaces (produces):**
```python
class CarryCapital(_Frozen):
    book_usdt: Decimal                 # > 0
    pair_slots: int                    # >= 1
    per_leg_notional_usdt: Decimal     # must equal book_usdt / (2 * pair_slots) exactly
    fee_tier: Literal["standard_taker_no_bnb"]

class CarryMember(_Frozen):
    name: str; lookback_weeks: int; hold_weeks: int
    exit_on_negative_funding: bool = False
    hurdle_multiple: Decimal | None = None
    book: Literal["cohorts", "slots"] = "cohorts"

MEMBER_NAMES_V4 = ("carry_s10_l4w_h13w", "carry_s10_l4w_h13w_exit", "carry_s10_l4w_h26w", "carry_s10_l4w_h26w_exit")
MEASURED_FAMILY_BY_BASE: dict[str, str] = {
    "funding_carry_panel_v2": "funding_carry_panel_v3",
    "funding_carry_panel_v4": "funding_carry_panel_v4_measured",
}
MEASURED_COST_FAMILY = "funding_carry_panel_v3"          # kept for existing callers
MEASURED_FAMILIES = frozenset(MEASURED_FAMILY_BY_BASE.values())
MEMBER_NAMES_BY_FAMILY[...]: v1, v2, v3 as now; "funding_carry_panel_v4" and "funding_carry_panel_v4_measured" -> MEMBER_NAMES_V4

class CarryFamilySpec(_Frozen):
    family_name: Literal["funding_carry_panel_v1", "funding_carry_panel_v2", "funding_carry_panel_v3",
                         "funding_carry_panel_v4", "funding_carry_panel_v4_measured"]
    ...
    capital: CarryCapital | None = None
    # validators: a family in MEASURED_FAMILIES requires cost_evidence, any other forbids it;
    # every member's book is "slots" iff capital is present (mixed books refused);
    # member names frozen per family as now.
```
`panel_config.load_family_spec` already dispatches on `family_name.startswith("funding_carry_panel_v") and family_name in MEMBER_NAMES_BY_FAMILY`; confirm it needs no change. `carry_fold_run.warm_up_weeks_of` and `control_reference` keep working (Task 4 changes warm-up for slot families).

`tests/carry_fixtures.py`: add `small_carry_v4_config(tmp_path) -> Path`: `_reduced` of the v4 base declaration with the fixture universe/selection reductions the v2 fixture applies, `pair_slots` 4, `book_usdt` "10000", `per_leg_notional_usdt` "1250", and member `hold_weeks` **1, 1, 2, 2** in declaration order (names unchanged; the fixture's three fold-0 Sundays must exercise age-out), `exit_on_negative_funding` false, true, false, true.

- [ ] Tests: `CarryCapital` rejects a notional that is not `book/(2·slots)`, a non-positive book, zero slots, another fee tier; `book` defaults to "cohorts"; a spec with `capital` and a cohort member is refused, a spec with a slot member and no `capital` is refused; `funding_carry_panel_v4` carries no `cost_evidence` and `funding_carry_panel_v4_measured` requires it; `MEMBER_NAMES_BY_FAMILY` binds the v4 names to both v4 families; `configs/funding-carry-panel-v4.json` loads, equals v2's document except `family_name`, `hypothesis`, `members`, `capital` (compare key by key), its members are the four frozen entries with `book` "slots", and `load_family_spec` dispatches it to `CarryFamilySpec`; the v4 fixture loads with 4 slots and holds 1, 1, 2, 2.
- [ ] Implement; full suite, ruff, mypy; commit `feat: declare funding carry v4 with its capital block and slot members`.

## Task 2: Capital-declared receipt reading

**Files:** modify `src/trading_bot/cost_evidence_rule.py`, `src/trading_bot/binance_cost_journal.py`, `src/trading_bot/carry_config.py` (`CostEvidenceReference.rule`), `src/trading_bot/carry_measured_costs.py`, `tests/test_carry_measured_costs.py`, `tests/test_binance_cost_journal.py`.

**Interfaces:**
```python
# cost_evidence_rule.py
CAPITAL_DECLARATION_RULE = (
    "a capital-declared family sets `slippage_bps_per_side_tier_<t>` in its base table to "
    "`tier_p50_of_p50` of the *worse leg* at the smallest ladder notional that is at least its "
    "per-leg order notional and in its adverse table to `tier_p50_of_p90` at ten times that "
    "notional, rounded **up** to the next whole basis point, and cites the receipt hash together "
    "with its declared capital."
)
# binance_cost_journal.FinalizationReceipt: new field `capital_declaration_rule: str`, validated
# verbatim against CAPITAL_DECLARATION_RULE; `finalize_journal` writes it beside `declaration_rule`.
# carry_config.CostEvidenceReference.rule: accepts DECLARATION_RULE or CAPITAL_DECLARATION_RULE.
# carry_measured_costs.py
def declaration_notionals(receipt: FinalizationReceipt, *, capital: CarryCapital | None) -> tuple[str, str]:
    """('5000','50000') without capital; with capital ('<ladder(N)>', '<10*ladder(N)>') as the receipt's
    own notional keys; refuses (MeasuredCostError) if no receipt notional >= N or 10*ladder(N) is not one."""
def measured_slippage_tiers(receipt: dict[str, object], *, capital: CarryCapital | None = None) -> tuple[dict[int, Decimal], dict[int, Decimal]]
def declare_measured_cost_family(*, receipt_path, base_declaration_path, output_path) -> tuple[Path, str]:
    # base family must be a key of MEASURED_FAMILY_BY_BASE; output family = the mapped name;
    # capital read from the base document (CarryFamilySpec.model_validate(base).capital);
    # cost_evidence.base_notional / adverse_notional = declaration_notionals(...); rule = the sentence used.
def verify_measured_declaration(*, receipt_path, spec_path, base_declaration_path=DEFAULT_BASE_DECLARATION) -> str:
    # same dispatch; the measured family name must be the base's mapped name; cited notionals and rule
    # must be the ones the base's capital selects; every other check as today.
```
The receipt's `notionals` are Decimals; compare `ladder` selection on Decimal values and emit the key as the receipt's slippage dict key (`_notional_keys` formatting) — the existing `_worse_leg` reads `rows[0].slippage.get(notional)` by string key, so pass the exact key string.

- [ ] Tests: the receipt carries the capital rule verbatim and refuses an edited one (mirror `test_the_declaration_rule_is_the_spec_s_sentence_verbatim`); the test receipt builder gains the field and every existing v3 test still passes unchanged; `declaration_notionals` without capital is ("5000", "50000"); with a capital of 10 000 / 10 slots on a 500/5000/50000 ladder it is ("500", "5000"); with per-leg 1 250 it is ("5000", "50000"); with per-leg 60 000 it refuses; with a ladder lacking the 10× notional it refuses; a v4 base declaration declares `funding_carry_panel_v4_measured` with tiers read at 500 (p50 of p50, worse leg) and 5 000 (p50 of p90), rounded up, `capital` carried unchanged, evidence citing 500/5000 and the capital rule; verification of that declaration passes and refuses an edited slippage, an edited notional, the v3 rule text in a v4 citation, and a v4 measured declaration whose base was v2; the CLI `carry-declare-measured`/`carry-verify-measured` round-trip on the v4 fixture base.
- [ ] Implement; full suite, ruff, mypy; commit `feat: capital-declared reading of the cost journal receipt`.

## Task 3: Slot book signals

**Files:** modify `src/trading_bot/carry_signals.py`, `tests/test_carry_signals.py`.

**Interfaces:**
```python
def rank_paying_pairs(snapshot: PairUniverseSnapshot, *, trailing: dict[str, Decimal | None],
                      selection: CarrySelectionRules, hurdle: dict[str, Decimal] | None = None) -> tuple[list[str], int]:
    """Top-decile pair ids by trailing funding (positive only; hurdle as select_member_cohort applies it),
    sorted by (-funding, pair_id), cut to _decile_size(len(snapshot.pairs), selection); ([] , rejections)
    when fewer than minimum_selected pay. select_member_cohort is re-expressed through it and must
    return exactly what it returns today (v1 fold-0 pin, v2 tests)."""
def random_pair_order(snapshot: PairUniverseSnapshot, *, selection: CarrySelectionRules, random_seed: int) -> list[str]:
    """The seeded shuffle select_control_cohort uses for random_pairs, cut to the decile, in shuffle order
    (select_control_cohort keeps sorting its cohort's entries, unchanged)."""
def fill_slots(ranked: list[str], *, held: set[str], skip: set[str], free: int) -> list[str]:
    """The first `free` ids of `ranked` that are neither held nor skipped, in rank order."""
def assemble_slot_book(slots: tuple[Cohort, ...], *, pair_slots: int, hold_weeks: int,
                       decision_close_ns: int) -> tuple[tuple[str, Decimal], ...]:
    """Each slot (a Cohort with exactly one entry) with 0 <= age < hold_weeks weeks contributes
    spot +1/(2*pair_slots), perpetual -1/(2*pair_slots); sorted leg weights; ValueError on a
    cohort with more than one entry or pair_slots < 1."""
```
- [ ] Tests: `rank_paying_pairs` equals the ids of `select_member_cohort`'s entries in order on the existing fixtures (with and without hurdle) and returns `[]` under the minimum; `fill_slots` skips held and skipped ids, respects `free`, returns fewer when the list is short; `assemble_slot_book` with 4 slots and two filled gives ±1/8 per leg and gross 0.5, drops an aged-out slot (age = hold), refuses a two-entry cohort; `random_pair_order` reproduces `select_control_cohort(kind="random_pairs")`'s entry set for the same seed.
- [ ] Implement; full suite, ruff, mypy; commit `feat: slot book signals for the small-capital carry`.

## Task 4: Slot mode in the carry fold runner, end to end

**Files:** modify `src/trading_bot/carry_fold_run.py`, `tests/test_carry_fold_run.py`, `tests/test_carry_end_to_end.py`.

**Behaviour (spec 3.2–3.3, 4.6):** for a family with `capital`, every member and `random_pairs` run in slot mode; `no_trade` holds nothing; `all_pairs_ew` runs in cohort mode with the reference hold as today. `warm_up_weeks_of(spec)` returns 0 for a family with `capital`; `FOLD_OPENING_BOOK_WARMED_FROM_PRIOR_WEEKS` is appended only when `warm_up_weeks > 0`. Per decision, per slot candidate, in this order: (1) release slots with age ≥ H weeks (`slot_releases`), slots whose pair has a leg without a bar at this decision, and slots force-closed in the previous episode (already removed after evaluation, as cohorts are); (2) exit rule for members with `exit_on_negative_funding` (`exit_rule_removals`); (3) fill: `ranked` = `rank_paying_pairs` (members) or `random_pair_order` (random_pairs); `skip` = held ids ∪ (exit members only) pairs whose trailing one-week funding is None or ≤ 0; `new` = `fill_slots(ranked, held=held, skip=skip, free=pair_slots − len(slots))`; each new id becomes `Cohort(decision, (entry,), (), formed_size=1)`; `slot_fills` = len(new); `no_fill` = 1 when `ranked` is empty (fewer than `minimum_selected` paying), and that sample id joins `no_carry_cohort_sample_ids`; (4) weights = `assemble_slot_book(...)`; evaluate both scenarios exactly as the cohort path does; forced pairs drop their slots. A `UNIVERSE_TOO_SMALL` week empties every slot. Extras for every candidate of a slot family: the eight existing keys plus `filled_slots` (after fill), `slot_fills`, `slot_releases`, `no_fill`; cohort families keep exactly eight. Per candidate and scenario in the fold report: `uncharged_final_exit_cost` = Σ over the last episode's drifted legs of |w| × (spot or perpetual fee per side + slippage of the leg's tier at that decision, tier two when unknown) / 10 000, `0` when nothing is held or the fold has no episodes; emitted for slot families only.

- [ ] Tests (v4 fixture, fold 0 with its three Sundays at day offsets 109/116/123): `warm_up_weeks` is 0 and the warmed reason code is absent; at the first decision every member fills up to 4 slots from the top of the ranking, each filled pair weighs spot +1/8 / perp −1/8, gross exposure = filled/4, and the first episode's turnover equals the whole book's entry; the hold-1 members release all four slots at the second decision and refill them in the same decision (`slot_releases` 4, `slot_fills` counts only pairs not re-filled, turnover only for pairs that changed); the hold-2 members release at the third decision; the exit members drop `C11USDT` at the decision after its negative-funding week while the non-exit siblings keep it, and its slot is refilled from the next-ranked paying pair not held (`exit_rule_removals` 1, `slot_fills` 1); `C10USDT`'s lost bar releases its slot at day 123 (`forced_close_count`/release as the cohort path records it); `no_trade` is flat; `random_pairs` fills from the seeded order and holds ≤ 4; `all_pairs_ew` equals the v2 fixture's `all_pairs_ew` episode for episode (same reference hold); `uncharged_final_exit_cost` is `0` for `no_trade`, equals a hand computation for one member (fees 10/5 plus tier slippage on each drifted leg), and is absent from a v2 fold report; a v2 fold report has exactly eight extras keys and the v1 fold-0 pin still passes; the end-to-end CLI chain (`panel-capture` ×2 → `panel-manifest` → `carry-fold` × folds → `panel-decision`) on the v4 fixture yields four members, the four new extras keys in `extras_mean`, and a decision status.
- [ ] Implement; full suite, ruff, mypy; commit `feat: slot book mode in the carry fold runner`.

## Task 5: Register P1.33 and document the chain

**Files:** modify `PHASE_1_BACKLOG.md` (P1.33 after P1.32, acceptance list in the style of P1.32: receipt required, capital block, section 5.1 rule, slot mechanics, extras, `uncharged_final_exit_cost`, record `P1_33_DECISION_<date>.md`, holdout closed), `README.md` (a "Funding-Carry v4 (P1.33)" section in the style of the P1.32 section: what the family is, the CLI chain `carry-declare-measured --base-config configs/funding-carry-panel-v4.json` → `carry-verify-measured` → `panel-manifest --hedge-capture` → `carry-fold` × 9 → `panel-decision`).

- [ ] Write both; commit `docs: register p1.33 and document the small-book carry chain`.

## After the plan

Orchestrator: merge `codex/carry-v4` into `codex/phase1-foundation`; write `C:/Users/User/.trading-jobs/carry-v4-chain.ps1` (refuses without `artifacts/cost/binance-carry-v1-receipt.json`; declare → verify → manifest → folds → decision under `artifacts/carry/carry-v4-*` / `funding-carry-v4-*`); append its launch to `carry-v3-chain.ps1`'s completion line; `P1_33_DECISION_<date>.md` when the decision exists.
