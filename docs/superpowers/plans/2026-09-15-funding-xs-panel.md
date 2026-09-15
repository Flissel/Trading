# Funding Cross-Section Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate `funding_xs_panel_v1` — a perpetual-only, dollar-neutral book long the lowest-funding quintile and short the highest-funding quintile of the P1.27 universe — through the unchanged P1.27 pipeline and gates.

**Architecture:** A declaration module and a signal module beside the panel line, and one generalisation: the P1.31 fold loop (`trend_fold_run.py`) becomes `vector_fold_run.py`, parameterised by a weight-vector builder, an optional per-decision exit rule and the family's module list; `trend_fold_run.py` and the new `funding_xs_fold_run.py` are thin wrappers. Accounting, universe, manifest, decision, P1.27/P1.28 modules untouched.

**Tech Stack:** Python 3.12, `Decimal`, pydantic v2, pytest, `mypy --strict`, `ruff`.

**Spec:** `docs/superpowers/specs/2026-09-15-funding-xs-family-design.md`.

## Global Constraints

- The v1 carry plan's Global Constraints bind (Decimal, integer ns, immutable artifacts, frozen models, no network in tests, `uv run` inside the worktree, pytest `--basetemp=C:/Users/User/AppData/Local/Temp/pytest-fxs`, ruff/mypy clean, trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`).
- Baseline `codex/phase1-foundation` at `26b77c9` (596 tests).
- **P1.31 reproducibility:** after the refactor, `run_trend_fold` on the trend fixture must produce fold reports whose every field except `code_hash`/`report_hash` is unchanged — pin fold-1 economics of the trend fixture in `tests/fixtures/trend_fold1_expected.json` captured BEFORE the refactor (its own commit), compared after. P1.27/P1.28/P1.29 modules untouched except `panel_config.load_family_spec` dispatch and type-only widenings.
- Frozen numbers (spec 3–4): members `fx_q5_l4w_h1w` (L4, H1, exit off), `fx_q5_l4w_h4w` (L4, H4, off), `fx_q5_l1w_h4w` (L1, H4, off), `fx_q5_l4w_h4w_exit` (L4, H4, on); controls P1.27's three (one-week hold); score = trailing L-week funding sum (`carry_signals.trailing_funding`), no settlement → no score; weights `panel_signals._cross_sectional_weights(scores, rules, reverse=True)`; costs = P1.27's tables except adverse funding receipts ×0.75 / payments ×2 (base 1/1); everything else byte-identical to `configs/xs-momentum-panel-v1.json`.
- Exit rule: at every in-window decision, before the untradeable strip and assembly, a contract held in any retained cohort vector with a positive weight (long) whose trailing one-week funding is ≥ 0, or negative weight (short) whose trailing one-week funding is ≤ 0 (None counts as flipped), is removed from every retained vector (weight set to 0, formed size unchanged — the vector's other weights stay as they are, so its capital stays undeployed); never in the warm-up; never for controls; count per episode as `exit_rule_removals`.

## File Structure

| File | Responsibility |
| --- | --- |
| `configs/funding-xs-panel-v1.json` | frozen declaration |
| `src/trading_bot/funding_xs_config.py` | `FundingXsMember`, `FundingXsFamilySpec`, `load_funding_xs_family_spec` |
| `src/trading_bot/panel_config.py` (modify) | dispatch `funding_xs_panel_v1` |
| `src/trading_bot/funding_xs_signals.py` | scores, weight vectors, sign-flip exits |
| `src/trading_bot/vector_fold_run.py` | generic weekly weight-vector fold loop (from `trend_fold_run.py`) |
| `src/trading_bot/trend_fold_run.py` (modify) | wrapper over `vector_fold_run` |
| `src/trading_bot/funding_xs_fold_run.py` | wrapper with the funding-xs builder and exit rule |
| `src/trading_bot/cli.py` (modify) | `funding-xs-fold` |
| tests | `tests/test_funding_xs_config.py`, `tests/test_funding_xs_signals.py`, `tests/test_funding_xs_fold_run.py`, `tests/test_funding_xs_end_to_end.py`, `tests/fixtures/trend_fold1_expected.json` |

---

## Task 1: Declaration and signals

**Files:** create `funding_xs_config.py`, `configs/funding-xs-panel-v1.json`, `funding_xs_signals.py`, `tests/test_funding_xs_config.py`, `tests/test_funding_xs_signals.py`; modify `panel_config.py`.

**Interfaces:**
```python
FUNDING_XS_MEMBER_NAMES = ("fx_q5_l4w_h1w", "fx_q5_l4w_h4w", "fx_q5_l1w_h4w", "fx_q5_l4w_h4w_exit")
FUNDING_XS_CONTROL_NAMES = ("no_trade", "random_ranks", "passive_long_ew")

class FundingXsMember(_Frozen):
    name: str; lookback_weeks: int; hold_weeks: int; exit_on_sign_flip: bool = False

class FundingXsControl(_Frozen):
    name: str; kind: Literal["no_trade", "random_ranks", "passive_long"]

class FundingXsFamilySpec(_Frozen):
    spec_version: Literal["1.0.0"]; family_name: Literal["funding_xs_panel_v1"]; hypothesis: str
    venue: Literal["BINANCE_UM"]; holding_days: int
    members: tuple[FundingXsMember, ...]   # names == FUNDING_XS_MEMBER_NAMES in order
    controls: tuple[FundingXsControl, ...] # names == FUNDING_XS_CONTROL_NAMES in order
    universe: PanelUniverseRules; weights: PanelWeightRules; costs: PanelCosts
    folds: PanelFoldGeometry; statistics: PanelStatistics

def load_funding_xs_family_spec(path: Path) -> tuple[FundingXsFamilySpec, str]

def funding_scores(funding_by_contract: dict[str, tuple[FundingEvent, ...]], eligible: tuple[str, ...],
                   decision_close_ns: int, lookback_weeks: int) -> dict[str, Decimal]:
    """trailing_funding per eligible contract; contracts with None omitted."""

def build_funding_xs_weight_vectors(histories, snapshot, funding_by_contract, *, spec) -> dict[str, WeightVector]:
    """Members: _cross_sectional_weights(funding_scores(...), spec.weights, reverse=True); controls exactly P1.27's."""

def sign_flip_exits(vector: tuple[tuple[str, Decimal], ...], trailing_one_week: dict[str, Decimal | None]) -> set[str]:
    """Contracts whose trailing one-week funding has the wrong sign for their leg (None counts as flipped)."""
```
JSON generated by script from `configs/xs-momentum-panel-v1.json`: copy `universe`, `weights`, `folds`, `statistics`, `holding_days`, `venue`, `spec_version`; `costs` copied with `adverse.funding_receipt_multiplier` set to `"0.75"` (payments already `"2"`); `family_name`, `hypothesis` (spec 4.1 verbatim), `members`, `controls`.

- [ ] Tests: JSON loads with the four members; copied blocks equal P1.27's value-for-value except the one adverse multiplier; reordered members rejected; `load_family_spec` dispatch; `funding_scores` omits contracts without settlements and sums only `(t − L, t]`; weight vectors: with 10 scored contracts (minimum quintile size 2 under a reduced rules object) the two lowest are long +0.25 and the two highest short −0.25 (leg gross 0.5), controls equal `panel_signals.build_weight_vectors`'s on the same snapshot; `sign_flip_exits` on a long with funding ≥ 0, a short with ≤ 0, None, and correctly signed cases.
- [ ] Implement; full suite, ruff, mypy; commit `feat: declare the funding cross-section family and its signals`.

## Task 2: Generic vector fold loop, funding-xs runner, CLI, end to end

**Files:** create `tests/fixtures/trend_fold1_expected.json` (own commit first, captured with the unmodified trend runner: for the trend fixture's fold 1, every candidate's base/adverse `total_net_return` and every episode's `sample_id`, `gross_exposure`, `turnover`, `net_return`, `contract_net_contributions`), `vector_fold_run.py`, `funding_xs_fold_run.py`, tests; modify `trend_fold_run.py`, `cli.py`, `PHASE_1_BACKLOG.md`, `README.md`.

**Interfaces:**
```python
# vector_fold_run.py
WeightBuilder = Callable[[dict[str, ContractHistory], UniverseSnapshot, dict[str, tuple[FundingEvent, ...]]], dict[str, WeightVector]]
ExitRule = Callable[[str, tuple[tuple[str, Decimal], ...], int], set[str]]   # (candidate name, retained vector, decision) -> contracts to zero
ExtrasBuilder = Callable[[str, tuple[tuple[str, Decimal], ...], UniverseSnapshot, int], dict[str, Decimal]]  # per candidate/decision

class VectorFamily(Protocol): members (with .name/.hold_weeks), controls (.name), costs, statistics, universe, holding_days, family_name, hypothesis

def run_vector_fold(capture_root, *, manifest_path, family_spec_path, output_path, registry_path, fold_index,
                    load_spec: Callable[[Path], tuple[VectorFamily, str]], build_vectors: WeightBuilder,
                    exit_rule: ExitRule | None, exit_rule_members: frozenset[str], extras: ExtrasBuilder,
                    modules: tuple[str, ...], error: type[RuntimeError]) -> VectorFoldArtifact
```
`trend_fold_run.run_trend_fold` becomes a call to `run_vector_fold` with the trend builder, no exit rule, the trend extras and `_TREND_MODULES` (+ `vector_fold_run.py`); its fold-1 economics must equal the pinned fixture. `funding_xs_fold_run.run_funding_xs_fold` passes `build_funding_xs_weight_vectors`, the sign-flip exit rule (members with `exit_on_sign_flip`; uses `trailing_funding(..., lookback_weeks=1)` on the perp's events), extras `signalled_contracts`, `funding_collected` (filled by the runner from the episode: `-funding_cost`), `exit_rule_removals`, `mean_score`, and `_FUNDING_XS_MODULES = (*_PANEL_MODULES, "funding_xs_config.py", "funding_xs_signals.py", "vector_fold_run.py", "funding_xs_fold_run.py")`. Note `funding_collected` needs the episode result — let the runner accept extras from both the builder stage (per decision) and the result stage (`-result.funding_cost`), merging them.

- [ ] Tests: trend fold-1 economics unchanged (fixture); funding-xs report schema = panel's + `warm_up_weeks` + extras; `no_trade` flat; a fixture where one contract's funding is high (short) and another's low (long) → the one-week member holds exactly those legs at ±0.25 under the reduced quintile rules; the four-week book averages the last four vectors; the exit member zeroes a short whose trailing week turned non-positive (fixture with a funding-sign flip) while the non-exit member keeps it; warm-up empty Sunday contributes nothing; linkage and immutability; the end-to-end CLI chain (`panel-capture` → `panel-manifest` → `funding-xs-fold` ×folds → `panel-decision`) with four members and the four extras keys.
- [ ] Commits: `test: pin the trend fold-1 economics before generalising the runner`; `feat: generic weekly vector fold loop and the funding cross-section runner`; `docs: register p1.32 and document the funding cross-section chain`.

## After the plan

`panel-manifest` with the funding-xs config on the repaired perpetual capture, nine `funding-xs-fold`s, `panel-decision`, `P1_32_DECISION_<date>.md`.
