# Final Holdout Read Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make protocol section 16.2's single-use holdout read executable for carry families: a capture-lineage check, a `carry-holdout` command that derives the candidate, evaluates it with the family's own fold mechanics on the original manifest's holdout, applies the fixed confirmation criteria and seals a single-use artifact.

**Architecture:** One new leaf module for capture lineage; the carry runner's in-window loop extracted into a function that takes explicit decisions, a candidate filter and a warm-up count (the fold path keeps its output byte-identical); a holdout runner beside it; one CLI command. Accounting, universe, manifest and decision modules untouched.

**Tech Stack:** Python 3.12, `Decimal`, pydantic v2, pytest, `mypy --strict`, `ruff`.

**Spec:** `docs/superpowers/specs/2026-09-17-holdout-read-design.md` and `PHASE_0_EVALUATION_PROTOCOL.md` section 16.2.

## Global Constraints

- The v1 carry plan's Global Constraints bind (Decimal, integer ns, immutable artifacts, frozen models, no network in tests, `uv run` inside the worktree, pytest `--basetemp=C:/Users/User/AppData/Local/Temp/pytest-holdout`, ruff/mypy clean, trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`).
- Baseline `codex/phase1-foundation` at `4d434e7` (693 tests). Worktree `.worktrees/holdout-read`, branch `codex/holdout-read`.
- **Reproducibility:** `run_carry_fold` output is byte-identical after the extraction: `tests/fixtures/carry_v1_fold0_expected.json` and every v2/v4 fold-run test pass unchanged.
- **Frozen rules (spec 2–6, protocol 16.2):** holdout ids and calendar from the original manifest only; captures must be verified supersets; candidate = eligible member with the highest `adverse_total_net_return − pooled adverse uncharged_final_exit_cost`, ties by declaration order; only the candidate, `no_trade` and `random_pairs` are evaluated; criteria: base total > 0; adverse total − adverse uncharged exit ≥ 0; base total > 0 and > `random_pairs` base; adverse total after subtraction ≥ 0 and ≥ `random_pairs` adverse; largest episode share ≤ 0.5 and largest pair share ≤ 0.5; skipped decisions ≤ 4; no statistical test; single use per family.
- Never evaluate a member other than the candidate on the holdout, not even in a test on real data; tests use the fixture captures only.

## File Structure

| File | Responsibility |
| --- | --- |
| `src/trading_bot/capture_lineage.py` | `verify_capture_superset` |
| `src/trading_bot/carry_fold_run.py` (modify) | `evaluate_carry_decisions` extracted; `run_carry_fold` unchanged in output |
| `src/trading_bot/carry_holdout_run.py` | `run_carry_holdout`, candidate derivation, criteria, sealed report |
| `src/trading_bot/cli.py` (modify) | `carry-holdout` |
| tests | `tests/test_capture_lineage.py`, `tests/test_carry_fold_run.py` (extend), `tests/test_carry_holdout_run.py` |
| `PHASE_1_BACKLOG.md`, `README.md` (modify) | P1.34 entry (holdout read machinery), chain documentation |

---

## Task 1: Capture lineage

**Files:** create `src/trading_bot/capture_lineage.py`, `tests/test_capture_lineage.py`.

**Interfaces:**
```python
def verify_capture_superset(original_root: Path, extended_root: Path) -> tuple[bool, tuple[str, ...]]:
    """True when the extended capture manifest contains every `sources` row of the original —
    same (kind, symbol, month) with identical `raw_sha256` and `status` — and both manifests
    agree on `venue`, `interval` and `market` (absent market means "um"). Reasons name the
    first 20 mismatches as "<kind>:<symbol>:<month>: missing|hash|status". Reads
    `<root>/capture-manifest.json` only; a missing or malformed manifest is a single reason."""
```
- [ ] Tests (fixture captures from `tests/carry_fixtures.py::build_captures` with `capture_panel` called directly where a different month set or fetch is needed; `MONTHS` and `SYMBOLS` come from `tests/test_panel_fold_run.py`): a capture over `MONTHS + ("2020-08",)` with the same fetch is a superset of one over `MONTHS` (True, no reasons); the reverse direction is False with `missing` reasons; a capture over the same months with `perp_fetch_with_a_liquidity_dip` differs in the dipped month's rows (False, `hash` reasons name the month); a spot capture is not a superset of a perp capture (market mismatch); a missing manifest is one reason.
- [ ] Implement; full suite, ruff, mypy; commit `feat: verify that an extended capture is a superset of its original`.

## Task 2: Extract the runner's decision loop

**Files:** modify `src/trading_bot/carry_fold_run.py`, `tests/test_carry_fold_run.py`.

**Interfaces:**
```python
@dataclass(frozen=True, slots=True)
class DecisionRun:
    candidates: list[dict[str, object]]     # exactly what run_carry_fold's `candidates` list is today
    skipped_sample_ids: list[str]
    episode_count: int
    reason_codes: list[str]                  # FOLD_WARMED..., SKIPPED_WEEK..., NO_EPISODES_IN_FOLD, FOLD_FINAL_EXIT_COST_UNCHARGED
    warm_up_weeks: int

def evaluate_carry_decisions(
    spec: CarryFamilySpec, *, perp_histories, spot_histories, leg_histories, funding_by_leg,
    decisions: list[int], candidate_names: tuple[str, ...] | None = None,
) -> DecisionRun:
    """The in-window loop of run_carry_fold from `names` onward, with `warm_up_weeks_of(spec)`
    warm-up before `decisions[0]`. `candidate_names` restricts `names` to those members and
    controls, in declaration order (None = all). Unknown names raise CarryFoldError."""
```
`run_carry_fold` becomes: verification and linkage as today → load bars and funding as today → `evaluate_carry_decisions(spec, ..., decisions=<fold test decisions>)` → the same `material` dict as today (unchanged keys and values). Nothing about the fold report may change.

- [ ] Tests: the v1 fold-0 pin and every existing carry fold-run test pass unchanged; `evaluate_carry_decisions` on the v4 fixture with `candidate_names=("carry_s10_l4w_h26w_exit", "no_trade", "random_pairs")` returns exactly three candidate records identical (field for field) to the corresponding records of a full run; an unknown name raises `CarryFoldError`; the warm-up count in the result equals `warm_up_weeks_of(spec)`.
- [ ] Implement; full suite, ruff, mypy; commit `refactor: extract the carry decision loop from the fold runner`.

## Task 3: Holdout runner and CLI

**Files:** create `src/trading_bot/carry_holdout_run.py`, `tests/test_carry_holdout_run.py`; modify `src/trading_bot/cli.py`.

**Interfaces:**
```python
HOLDOUT_ARTIFACT_KIND = "holdout"
MAX_SKIPPED_HOLDOUT_DECISIONS = 4
CONCENTRATION_LIMIT = Decimal("0.5")

class CarryHoldoutError(RuntimeError): ...

def derive_candidate(decision: dict[str, object], fold_reports: list[dict[str, object]]) -> tuple[str, Decimal]:
    """(name, adverse total − pooled adverse uncharged_final_exit_cost) of the eligible member with
    the highest such value, ties by the decision's member order; CarryHoldoutError when
    `eligible_member_names` is empty. Cohort families contribute 0 for the uncharged cost."""

def run_carry_holdout(
    perp_capture_root: Path, spot_capture_root: Path, *, original_perp_capture_root: Path,
    original_spot_capture_root: Path, manifest_path: Path, family_spec_path: Path,
    decision_path: Path, fold_report_paths: tuple[Path, ...], output_path: Path, registry_path: Path,
) -> CarryHoldoutArtifact:   # (output_path, report_hash, candidate_name, verdict)
```
Order of checks, each a `CarryHoldoutError`: output exists → refuse; both extended captures verify (`verify_panel_capture`); `verify_capture_superset` for perp and for spot; manifest verifies and its `family_spec_hash` equals the spec's; the decision document's `report_hash` recomputes (content hash of the document without `report_hash`, as `panel_decision` seals it), its `family_spec_hash` equals the spec's and its `source_report_hashes` equal the given fold reports' `report_hash` values in order; every fold report verifies (`verify_panel_fold_report`); the registry has no artifact `uuid5(family_id, "holdout")` where `family_id = uuid5(NAMESPACE_URL, f"{split_manifest_hash}:{family_name}")` (as `_register` derives it); the last holdout decision + `holding_days` days must have bars in both extended captures for `BTCUSDT` (the calendar reference) — else "extended capture does not cover the holdout's last exit". Then: holdout decisions = `manifest["final_holdout_ids"]`; bars loaded with `available_before_ns = last decision + holding_days*DAY_NS + 1` from the extended captures; `evaluate_carry_decisions(spec, ..., decisions=holdout decisions, candidate_names=(candidate, "no_trade", "random_pairs"))`; criteria per the Global Constraints (largest pair share from the candidate's base `contract_net_contributions` pooled over episodes: max |pair total| / |base total| — reuse `panel_decision`'s concentration helper by importing it; if it is private, give it a public alias there rather than duplicating it); reported values: base and adverse mean weekly net, positive-week fraction, the decision's `base_mean_net_return` and `base_bootstrap_lower` for the candidate and whether the holdout base mean ≥ that lower bound. Report document: the fold-report keys (`report_version`, `status`, `reason_codes`, `family_name`, `family_spec_hash`, capture/dataset hashes of the extended captures, `split_manifest_hash`, `manifest_hash`, `random_seed`, `block_length`, `bootstrap_repetitions`, `skipped_sample_ids`, `warm_up_weeks`, `code_hash`, `candidates`) plus `holdout: true`, `candidate_name`, `decision_report_hash`, `original_capture_root_hash`, `original_hedge_capture_root_hash`, `holdout_sample_count`, `holdout_membership_hash`, `confirmation` (`verdict` "holdout_confirmed" | "holdout_failed", `base_total_net_return`, `adverse_total_net_return`, `adverse_uncharged_final_exit_cost`, `adverse_total_after_uncharged_exit`, `criteria` dict of the eight booleans and the two shares and the skipped count, `reported` dict), sealed with `report_hash` = content hash, written atomically, then the registry artifact (kind `holdout`, id as above, `relative_path` relative to the registry's parent's parent or the workspace — mirror `_register`'s conventions) — a `RegistryConflictError` or an existing record refuses before anything is written. `_CARRY_MODULES` plus `carry_holdout_run.py` and `capture_lineage.py` form the code hash.
CLI `carry-holdout`: `--workspace-root`, `--capture`, `--hedge-capture`, `--original-capture`, `--original-hedge-capture`, `--manifest`, `--family-spec`, `--decision`, `--fold-report` (append, required), `--output`, `--registry`; all paths inside the workspace as `carry-fold` checks; exit 0 on either verdict (the verdict is data), non-zero on any refusal.

- [ ] Tests (fixture chain on `small_carry_v4_config`; the extended captures are `build_captures`-style captures over `MONTHS + ("2020-08",)`, the originals over `MONTHS`; fold reports from the real `carry-fold` runs; the decision document hand-sealed the way `panel_decision` seals it, with `eligible_member_names` and `source_report_hashes` set as needed): `derive_candidate` picks the highest adverse-after-exit among two eligible members and breaks a tie by order, refuses an empty list; a run on the fixture writes a report with `holdout: true`, exactly three candidate records, the holdout ids from the manifest, `warm_up_weeks` 0, a verdict and all criteria fields; the verdict equals a hand computation from the report's own totals; the registry carries one `holdout` artifact and a second run (new output path) refuses; a non-superset extended capture refuses with the lineage reasons; a decision whose `source_report_hashes` do not match refuses; a decision without eligible members refuses; a missing last-exit bar refuses; the CLI round-trip exits 0 and refuses a path outside the workspace; `run_carry_fold` still passes the pin.
- [ ] Implement; full suite, ruff, mypy; commit `feat: single-use final holdout read for carry families`.

## Task 4: Register and document

**Files:** modify `PHASE_1_BACKLOG.md` (`### P1.34 Final holdout read machinery` after P1.33: paragraph + Acceptance bullets mirroring spec 2–6 and protocol 16.2; holdout stays closed until the user opens it), `README.md` (a German section "Holdout-Read (P1.34)" after the P1.33 section: what it is, the lineage rule, the derived candidate, the criteria, the `carry-holdout` command line with the extended and original captures, the single-use rule).

- [ ] Write both; commit `docs: register the holdout read machinery`.

## After the plan

Orchestrator: merge into `codex/phase1-foundation`; in October (after the September dumps): recapture both markets through 2026-09, repair, then, only on the user's explicit go, run `carry-holdout` for the official P1.33 decision and write `P1_33_HOLDOUT_<date>.md`.
