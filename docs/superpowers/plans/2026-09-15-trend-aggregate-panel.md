# Trend Aggregate Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate `trend_aggregate_panel_v1` — a twelve-indicator trend vote on the P1.27 perpetual panel, three time-series members (two thresholds, one four-week cohort hold) and one cross-sectional member — through the unchanged P1.27 pipeline and gates.

**Architecture:** Three new modules beside the panel line: `trend_config` (frozen declaration reusing the panel's universe/weight/cost/fold/statistics models), `trend_signals` (indicators → votes → score → weight vectors, reusing `panel_signals`' volatility, water-filling and quintile helpers), `trend_fold_run` (the P1.27 fold loop with an added four-week cohort book, emitting the P1.27 report schema plus three extras). `panel_config.load_family_spec` dispatches the new family name. Accounting, universe, manifest, decision and the P1.27 modules are untouched.

**Tech Stack:** Python 3.12, `Decimal`, pydantic v2, pytest, `mypy --strict`, `ruff`.

**Spec:** `docs/superpowers/specs/2026-09-15-trend-aggregate-family-design.md`.

## Global Constraints

- The v1 carry plan's Global Constraints bind (Decimal, integer ns, immutable artifacts, frozen models, no network in tests, `uv run` inside the worktree, pytest `--basetemp=C:/Users/User/AppData/Local/Temp/pytest-trend`, ruff/mypy clean, trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`).
- Baseline `codex/phase1-foundation` at `a106739` (506 tests).
- **P1.27 must be reproducible:** `panel_signals.py`, `panel_fold_run.py`, `panel_config.py`'s models and `configs/xs-momentum-panel-v1.json` are not modified except for the loader dispatch; every panel test keeps passing unchanged.
- Frozen numbers (spec 3–4): twelve indicators exactly as listed with windows 20/50/100 (SMA), 20/50/100 (breakout), 20/60/120 (ROC), EMA 12/26; score = mean of twelve votes, no score with fewer than twelve computable votes; members `ta_ts_t02` (threshold 0.2, hold 1), `ta_ts_t05` (0.5, hold 1), `ta_ts_t02_h4w` (0.2, hold 4), `ta_xs_q5` (cross-sectional quintiles); controls `no_trade`, `random_ranks`, `passive_long_ew`; universe/weights/costs/folds/statistics byte-identical to `configs/xs-momentum-panel-v1.json`.
- Time-series weights: for signalled contracts, `sign / max(sigma, floor)` with the panel's 30-day calendar-complete annualised volatility, normalised to unit gross, water-filled at `time_series_cap_numerator / n` (P1.27's `_time_series_weights` with the trailing-return sign replaced by the score's thresholded sign). Cross-sectional: P1.27's `_cross_sectional_weights` on the scores.
- Four-week member: the book at *t* is the sum over the last four weekly weight vectors (including *t*'s) of `vector / 4`; a vector that is empty contributes nothing (capital undeployed); a contract with no bar at *t* is dropped from the assembly; a skipped week resets the vectors (P1.27 semantics); warm-up of the three Sundays before each fold's first decision (weights only, no episodes, carried weights empty) so the first episode opens on a full book, reason code `FOLD_OPENING_BOOK_WARMED_FROM_PRIOR_WEEKS`.

## File Structure

| File | Responsibility |
| --- | --- |
| `configs/trend-aggregate-panel-v1.json` | frozen declaration |
| `src/trading_bot/trend_config.py` | `TrendMember`, `TrendControl`, `TrendFamilySpec`, `load_trend_family_spec` |
| `src/trading_bot/panel_config.py` (modify) | `load_family_spec` dispatches `trend_aggregate_panel_v1` |
| `src/trading_bot/trend_signals.py` | indicators, votes, score, weight vectors |
| `src/trading_bot/trend_fold_run.py` | one fold, P1.27 schema + extras, four-week cohort book |
| `src/trading_bot/cli.py` (modify) | `trend-fold` |
| `tests/test_trend_config.py`, `tests/test_trend_signals.py`, `tests/test_trend_fold_run.py`, `tests/test_trend_end_to_end.py` | tests |

---

## Task 1: Declaration and loader

**Files:** create `trend_config.py`, `configs/trend-aggregate-panel-v1.json`, `tests/test_trend_config.py`; modify `panel_config.py`.

**Interfaces:**
```python
TREND_MEMBER_NAMES = ("ta_ts_t02", "ta_ts_t05", "ta_ts_t02_h4w", "ta_xs_q5")
TREND_CONTROL_NAMES = ("no_trade", "random_ranks", "passive_long_ew")
INDICATOR_NAMES = ("ma_20", "ma_50", "ma_100", "ma_cross_20_50", "ma_cross_50_100", "breakout_20", "breakout_50", "breakout_100", "roc_20", "roc_60", "macd_12_26", "roc_120")

class TrendMember(_Frozen):
    name: str
    kind: Literal["time_series", "cross_sectional"]
    threshold: Decimal | None      # required for time_series (0 < threshold <= 1), must be None for cross_sectional
    hold_weeks: int                # 1 or 4

class TrendControl(_Frozen):
    name: str
    kind: Literal["no_trade", "random_ranks", "passive_long"]

class TrendFamilySpec(_Frozen):
    spec_version: Literal["1.0.0"]
    family_name: Literal["trend_aggregate_panel_v1"]
    hypothesis: str
    venue: Literal["BINANCE_UM"]
    holding_days: int              # 7
    indicators: tuple[str, ...]    # must equal INDICATOR_NAMES in order
    members: tuple[TrendMember, ...]   # names must equal TREND_MEMBER_NAMES in order
    controls: tuple[TrendControl, ...] # names must equal TREND_CONTROL_NAMES in order
    universe: PanelUniverseRules
    weights: PanelWeightRules
    costs: PanelCosts
    folds: PanelFoldGeometry
    statistics: PanelStatistics

def load_trend_family_spec(path: Path) -> tuple[TrendFamilySpec, str]
```
`panel_config.load_family_spec` returns `TrendFamilySpec` when `family_name == "trend_aggregate_panel_v1"` (local import). The JSON is generated by script from `configs/xs-momentum-panel-v1.json`: copy `universe`, `weights`, `costs`, `folds`, `statistics`, `holding_days`, `venue`, `spec_version`; set `family_name`, `hypothesis` (spec 4.1 verbatim), `indicators`, `members` (`{"name","kind","threshold","hold_weeks"}` with thresholds as strings `"0.2"`/`"0.5"`, `null` for the cross-sectional member), `controls` (P1.27's three with kinds).

- [ ] Tests: JSON loads with the four members and thresholds; the five copied blocks equal P1.27's byte-for-byte (compare parsed values); a reordered member set is rejected; a time-series member without threshold is rejected; a cross-sectional member with a threshold is rejected; `load_family_spec` returns the trend model and still returns the panel/carry models for their files.
- [ ] Implement; full suite, ruff, mypy; commit `feat: declare the trend aggregate family`.

## Task 2: Indicators, score, weight vectors

**Files:** create `trend_signals.py`, `tests/test_trend_signals.py`.

**Interfaces:**
```python
def closes_before(history: ContractHistory, decision_close_ns: int) -> tuple[Decimal, ...]:
    """Closes at or before the decision, ascending by close time (uses history.close_times)."""

def indicator_votes(closes: tuple[Decimal, ...]) -> dict[str, int]:
    """Every indicator in INDICATOR_NAMES → vote in {-1, 0, 1}; an indicator whose window exceeds len(closes) is absent from the dict."""

def trend_score(closes: tuple[Decimal, ...]) -> Decimal | None:
    """Mean of the twelve votes as Decimal, or None if any indicator is absent."""

def sma(closes, n) -> Decimal; def ema(closes, n) -> Decimal  # alpha = 2/(n+1), seeded with closes[0], over the whole series
def build_trend_weight_vectors(histories, snapshot, *, spec: TrendFamilySpec) -> dict[str, WeightVector]:
    """One WeightVector per member and control for one decision, as panel_signals.build_weight_vectors does:
    time-series members: sign = +1 if score >= threshold, -1 if score <= -threshold, else no position; weight = sign / max(sigma, floor)
    with panel_signals._annualised_volatility(history, decision, spec.weights.volatility_window_days); normalise to unit gross;
    water-fill at spec.weights.time_series_cap_numerator / n (reuse panel_signals._water_fill/_normalise);
    cross-sectional member: panel_signals._cross_sectional_weights(scores, spec.weights, reverse=False) over contracts with a score;
    controls exactly as P1.27 (no_trade empty; passive_long_ew equal weight over the eligible set; random_ranks seeded with
    spec.statistics.random_seed ^ decision)."""
```
Definitions (spec 3): `ma_n = sign(P_t − SMA_n)`; `ma_cross_a_b = sign(SMA_a − SMA_b)`; `breakout_n`: +1 if `P_t == max(last n closes)`, −1 if `== min`, else 0 (ties: if both, 0); `roc_n = sign(P_t / P_{t−n} − 1)` (needs n+1 closes); `macd_12_26 = sign(EMA_12 − EMA_26)` (needs ≥ 26 closes); sign of exact zero is 0. SMA over the last n closes.

- [ ] Tests: hand-computed votes on a 130-close synthetic series (a monotone rise → all +1 except breakouts by construction; a flat series → all 0; a series where `P_t` equals both the 20-day high and low → 0); each indicator's minimum window (absent below it, present at it); `trend_score` None with 11 votes; EMA against a hand-rolled recursion on five values; time-series weights: two signalled contracts with different sigmas produce inverse-vol weights summing to unit gross and respecting the cap; threshold boundaries (score exactly 0.2 is long); cross-sectional quintiles via the panel helper; controls equal to `panel_signals.build_weight_vectors`'s controls on the same snapshot (pin equality).
- [ ] Implement; full suite, ruff, mypy; commit `feat: trend aggregate score and weight vectors`.

## Task 3: Fold runner, CLI, end to end

**Files:** create `trend_fold_run.py`, `tests/test_trend_fold_run.py`, `tests/test_trend_end_to_end.py`; modify `cli.py`.

**Interfaces:** `run_trend_fold(capture_root, *, manifest_path, family_spec_path, output_path, registry_path, fold_index) -> TrendFoldArtifact(output_path, report_hash, fold_index, episode_count, skipped_sample_count)`; `TrendFoldError`; `_TREND_MODULES = panel modules + ("trend_config.py", "trend_signals.py", "trend_fold_run.py")`.

Behaviour: mirror `run_panel_fold` (verify capture and manifest; linkage on `family_spec_hash`, `capture_root_hash`, `dataset_root_hash`; bars with `available_before_ns = test_end_ns + 1`; `select_universe` per decision; skipped weeks reset everything and are `SKIPPED_WEEK_EXIT_COST_UNCHARGED`; `MEMBER_HELD_NOTHING` for members with empty weights; `FOLD_FINAL_EXIT_COST_UNCHARGED`; registry as P1.27). Per decision: `vectors = build_trend_weight_vectors(...)`; for `hold_weeks == 1` members and controls, weights are the vector; for the four-week member, keep a per-fold list of the last four vectors (reset on skip), the book = Σ (vector_k / 4) over retained vectors, dropping any contract without a bar at *t*, then evaluate with the panel accounting exactly as the others. Warm-up: the three Sundays before the first decision form vectors only (universe from the same histories; no episode), reason code `FOLD_OPENING_BOOK_WARMED_FROM_PRIOR_WEEKS` when at least one decision exists; report field `warm_up_weeks: 3`. Extras per episode: `signalled_contracts` (contracts with a non-zero weight for that candidate), `abstained_contracts` (eligible contracts with a score inside the dead band, time-series members only, else 0), `mean_score` (mean score over eligible contracts with a score, or 0 when none), all Decimal. Report material otherwise identical to `panel_fold_run`'s (same keys and order) so `panel_decision.py` pools it; `panel_decision` reads `spec.members/controls/statistics/family_name` and dominance controls by kind (`no_trade`, `random_ranks`) — `TrendControl.kind` uses the panel kinds so no decision change is needed.

CLI `trend-fold` with the `panel-fold` arguments. End-to-end test: `panel-capture` (fake fetch from `tests/test_panel_fold_run.py`), `panel-manifest` with a reduced trend config (same reductions as `small_config` in that test file: universe 20/5/10/10/4, weights minimum_quintile_size 2 / volatility_window_days 10, the reduced folds, block 2, floor 4), `trend-fold` per fold, `panel-decision`; assert four members, `decision_status` terminal, extras_mean with the three keys.

- [ ] Tests: report schema + extras; `no_trade` flat; the four-week member's gross exposure ramps to its warmed level and a contract missing at *t* is dropped; skipped week resets the cohort list; `MEMBER_HELD_NOTHING` when the threshold silences every contract (a flat fixture); linkage and immutability; the end-to-end chain.
- [ ] Implement; full suite, ruff, mypy; commit `feat: evaluate one trend aggregate fold`. Then `PHASE_1_BACKLOG.md` P1.31 entry after P1.30, README paragraph; commit `docs: register p1.31 and document the trend chain`.

## After the plan

Run after P1.30's decision (spec 7): `panel-manifest` with the trend config on the repaired perpetual capture, nine `trend-fold`s, `panel-decision`, `P1_31_DECISION_<date>.md`.
