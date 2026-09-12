# Funding Carry v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate `funding_carry_panel_v2` — the P1.28 carry with a 26-week hold, an exit on non-positive trailing funding, and a cost hurdle at entry — through the unchanged P1.27/P1.28 pipeline.

**Architecture:** Three additive changes to the carry line: member flags in the declaration (`exit_on_negative_funding`, `hurdle_multiple`) with per-family frozen member sets; two pure functions in `carry_signals` (hurdle threshold, exit-rule pairs) used by the fold runner before cohort formation and before book assembly; the runner's controls and warm-up length derived from the declaration instead of hard-coded. Accounting, universe, manifest, decision and CLI are untouched.

**Tech Stack:** Python 3.12, `Decimal`, pydantic v2, pytest, `mypy --strict`, `ruff`.

**Spec:** `docs/superpowers/specs/2026-09-12-funding-carry-v2-family-design.md`.

## Global Constraints

- Everything in `docs/superpowers/plans/2026-09-11-funding-carry-panel.md` Global Constraints still binds (Decimal, integer ns, immutable artifacts, frozen dataclasses/pydantic, no network in tests, `uv run` inside the worktree, `--basetemp=C:/Users/User/AppData/Local/Temp/pytest-carry`, ruff/mypy clean, commit trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`).
- Baseline: `codex/phase1-foundation` at `e2939b8`, **370 passed**.
- **P1.28 must be reproducible:** `configs/funding-carry-panel-v1.json` is frozen; every v1 test keeps passing unchanged; the v1 runner semantics (controls = `carry_l1w_h4w`'s L/H, warm-up 12) must fall out of the new generic rules, not be special-cased.
- Frozen v2 numbers: members `carry_l4w_h26w` (L4, H26, exit off, no hurdle), `carry_l4w_h13w_exit` (L4, H13, exit on), `carry_l4w_h26w_exit` (L4, H26, exit on), `carry_l4w_h26w_exit_hurdle2` (L4, H26, exit on, hurdle 2); controls `no_trade`, `random_pairs`, `all_pairs_ew`; everything else copied from v1's JSON.
- Rules (spec 3.3): exit rule — a held pair with trailing one-week funding `≤ 0` or no settlement in `(t − 1w, t]` is removed from every cohort before assembly (formed_size kept); hurdle — pair qualifies iff `F_L × H / (2 L) ≥ k × c_rt(tier)`, `c_rt = (spot_fee + perpetual_fee + 2 × slippage_tier) / 10 000` from the **base** cost table; controls use the lookback/hold of the member with the shortest hold (ties → declaration order), never the exit rule or hurdle; warm-up = `max(hold_weeks) − 1` Sundays.

---

## Task 1: Declaration and loader

**Files:** modify `src/trading_bot/carry_config.py`, `src/trading_bot/panel_config.py`; create `configs/funding-carry-panel-v2.json`; test `tests/test_carry_config.py`.

**Interfaces:** `CarryMember` gains `exit_on_negative_funding: bool = False`, `hurdle_multiple: Decimal | None = None` (validator: `> 0` when set); `CarryMember.name: str`; `CarryFamilySpec.family_name: Literal["funding_carry_panel_v1", "funding_carry_panel_v2"]`; module constant `MEMBER_NAMES_BY_FAMILY: dict[str, tuple[str, ...]]` with the v1 triple and the v2 quadruple; keep `MEMBER_NAMES` as the v1 alias for existing imports; a `model_validator(mode="after")` checks `tuple(m.name for m in members) == MEMBER_NAMES_BY_FAMILY[family_name]`; `panel_config.load_family_spec` dispatches when `family_name` starts with `"funding_carry_panel_v"` and is in the known set.

- [ ] Write failing tests: v2 JSON loads with the four members and their flags (`exit_on_negative_funding` True for three, `hurdle_multiple == Decimal("2")` for the last); v1 JSON still loads unchanged with the flags defaulting to False/None; a v2 document with v1's member names is rejected; a member with `hurdle_multiple: "0"` is rejected; `load_family_spec` returns `CarryFamilySpec` for both files with distinct hashes.
- [ ] Generate `configs/funding-carry-panel-v2.json` from v1's JSON by script: copy everything, set `family_name`, replace `hypothesis` with spec 3.1's text, replace `members` with the four entries (JSON: `{"name": ..., "lookback_weeks": 4, "hold_weeks": 26, "exit_on_negative_funding": false, "hurdle_multiple": null}` etc.; write `hurdle_multiple` as the string `"2"` where set). Keep pairs/excluded_pairs/universe/selection/costs/folds/statistics byte-identical to v1.
- [ ] Implement, run `tests/test_carry_config.py tests/test_panel_config.py tests/test_carry_fold_run.py`, full suite, ruff, mypy; commit `feat: declare the funding carry v2 family`.

## Task 2: Rules in signals and runner

**Files:** modify `src/trading_bot/carry_signals.py`, `src/trading_bot/carry_fold_run.py`; tests `tests/test_carry_signals.py`, `tests/test_carry_fold_run.py`; fixtures `tests/carry_fixtures.py`.

**Interfaces (carry_signals):**
```python
def round_trip_cost_bps(cost_table: CarryCostTable, tier: int) -> Decimal:
    """(spot_fee + perpetual_fee + 2 * slippage_tier), in bps per unit of pair capital."""

def hurdle_minimum_trailing(*, cost_table: CarryCostTable, tier: int, multiple: Decimal,
                            lookback_weeks: int, hold_weeks: int) -> Decimal:
    """Minimum F_L (per unit perpetual notional, as a fraction) for a pair to qualify:
    2 * multiple * round_trip_cost_bps / 10_000 * lookback_weeks / hold_weeks."""

def select_member_cohort(snapshot, *, trailing, selection, hurdle: dict[str, Decimal] | None = None) -> Cohort:
    # a pair with hurdle[pair_id] set qualifies only if trailing[pair_id] >= hurdle[pair_id];
    # decile size unchanged (from len(snapshot.pairs)); Cohort gains reason_codes entry
    # "HURDLE_APPLIED" is NOT added — instead return the count via a new field:
    # Cohort.hurdle_rejections: int = 0

def exit_rule_pairs(cohorts: list[Cohort], *, trailing_one_week: dict[str, Decimal | None]) -> set[str]:
    """Pair ids held in any cohort whose trailing one-week funding is None or <= 0."""
```
`Cohort` gains `hurdle_rejections: int = 0` as a trailing default field (existing positional constructions stay valid).

**Runner:**
- `hold_of`/`lookback_of` for controls come from `min(spec.members, key=lambda m: m.hold_weeks)` (Python's `min` keeps the first on ties, which is declaration order).
- `WARM_UP_WEEKS` becomes `warm_up_weeks = max(m.hold_weeks for m in spec.members) - 1`, reported as `"warm_up_weeks"`.
- `_cohort_for` receives the member (or None for a control) and, when `member.hurdle_multiple` is set, builds `hurdle = {pair_id: hurdle_minimum_trailing(cost_table=spec.costs.base, tier=pair.tier, multiple=..., lookback_weeks=..., hold_weeks=...)}` and passes it.
- Exit rule: in the decision loop, for a member with `exit_on_negative_funding`, compute `trailing_one_week` for every pair held in `cohorts[name]` (via `trailing_funding(..., lookback_weeks=1)` on the perp leg's events), take `exit_rule_pairs`, strip with `_without_pairs`, and count them; this happens **before** the untradeable strip and assembly, and not during the warm-up (the warm-up forms cohorts only; the first in-window decision applies the rule to the warmed cohorts, which is correct because the rule reads only data at or before that decision).
- Per-episode extras gain `"exit_rule_removals"` and `"hurdle_rejections"` (Decimal integers; controls always 0). Update every test that pins the extras key set to eight keys (`tests/test_carry_fold_run.py`, `tests/test_panel_decision.py` EXTRAS + `write_carry_fold`, `tests/test_carry_end_to_end.py`).

**Tests to write (all deterministic, no network):**
- `round_trip_cost_bps` = 25 (tier 1) and 35 (tier 2) on the v1 base table; `hurdle_minimum_trailing(multiple=2, L=4, H=26)` tier 1 = `Decimal("0.00153846…")` — assert against the exact formula `2 * 2 * 25 / 10000 * 4 / 26` computed in Decimal, not a literal.
- `select_member_cohort` with a hurdle: pairs below their threshold are excluded from ranking, `hurdle_rejections` counts them, fewer than `minimum_selected` survivors → `NO_CARRY_COHORT`.
- `exit_rule_pairs`: None and ≤ 0 removed, positive kept.
- Runner on the v1 fixture with a v2-style reduced config (`tests/carry_fixtures.py::small_carry_v2_config`: v2 JSON with the same reductions as `small_carry_config`, members as declared): (a) warm-up weeks reported = 25 and the first episode's gross exposure for `carry_l4w_h26w` is within `1e-25` of 1 (26 cohorts × 1/26 — check the fixture's history is long enough: the first decision at day 109 minus 25 weeks = day −66 is BEFORE the fixture's first bar (day 0), so warm-up Sundays before day 0 contribute no cohort; assert instead that the exposure equals `(number of warm-up Sundays ≥ day 20 with a full universe + 1) / 26` — derive the exact expected count from the fixture: Sundays at 109 − 7k for k = 1..25 that are ≥ 20 + 4 (history 20 bars + liquidity window) — compute it in the test from the fixture constants rather than hard-coding); (b) with `perp_fetch_with_negative_funding_weeks`, `carry_l4w_h13w_exit` removes pairs at the decisions whose trailing week is negative — assert `exit_rule_removals > 0` at decisions 109 and 116 and `== 0` at 123, and that `carry_l4w_h26w` (exit off) has `exit_rule_removals == 0` everywhere; (c) `carry_l4w_h26w_exit_hurdle2` with the standard fixture reports `hurdle_rejections >= 0` and the v1 fixture family (`small_carry_config`) still produces byte-identical fold reports for the P1.28 members — pin by asserting the v1 fold-0 `report_hash` before and after this task is unchanged?? The code hash changes with this task (carry_fold_run.py is in `_CARRY_MODULES`), so the report hash cannot be identical; instead assert that every v1 candidate's base and adverse `total_net_return` and every episode's `gross_exposure`/`turnover` equal the values produced at `e2939b8` — capture those values ONCE at the start of the task by running the v1 fixture on the unmodified code and writing them to `tests/fixtures/carry_v1_fold0_expected.json` (commit that file first, in its own commit, before touching the runner).
- Controls under v2: `hold_of["random_pairs"] == 13`, `lookback_of == 4` (assert via the report: `random_pairs` first-episode gross exposure pattern is that of a 13-cohort warm-up).

- [ ] Commit `test: pin the v1 fold-0 economics before the v2 rules` (expected-values file) then `feat: exit rule, cost hurdle and declaration-driven controls for carry v2`.

## Task 3: Registration and end-to-end

**Files:** `PHASE_1_BACKLOG.md` (P1.29 entry after P1.28, acceptance mirroring P1.28's with the v2 members), `README.md` (one paragraph in the carry subsection: v2 config and rules), `tests/test_carry_end_to_end.py` (parametrize the existing test over both configs).

- [ ] Commit `docs: register p1.29 and cover the v2 chain end to end`.

## After the plan

Run: `panel-manifest --capture <perp repaired> --hedge-capture <spot repaired> --family-spec configs/funding-carry-panel-v2.json --output artifacts/carry/carry-v2-walk-forward-2026-09-12-usdt-pairs-1d-w1-v1.json`, nine `carry-fold`s, `panel-decision`, then `P1_29_DECISION_2026-09-12.md`.
