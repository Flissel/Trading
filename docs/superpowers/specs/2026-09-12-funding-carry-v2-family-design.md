# Funding Carry v2 Family Design

Status: Decided autonomously on 2026-09-12 under the standing goal "make it better,
decide autonomously until a reliable solution exists". `research_only`. Pre-registers
experiment family `funding_carry_panel_v2` under `PHASE_0_EVALUATION_PROTOCOL.md`
sections 1 to 16 as they stand on 2026-09-12, before any v2 fold is run. No live, paper,
or shadow authority follows from it.

## 1. Purpose

P1.28 (`P1_28_DECISION_2026-09-11.md`) rejected `funding_carry_panel_v1`. Its long-hold
member `carry_l4w_h13w` was the first candidate in this repository to clear every
statistical gate (bootstrap lower bound above zero, BH q 0.015, 8/9 positive folds) and
failed on exactly two things: the adverse-cost floor and single-fold concentration. The
per-week decomposition was +8.2 bps funding, −3.9 bps execution, basis ≈ 0, and about
−1 bps of doubled funding *payments* under adverse in weeks where a held pair's funding
turned negative. Holding longer improved every metric from 4 to 13 weeks.

This family asks the narrow follow-up: do three mechanical, forecast-free rules that
reduce execution and avoid paying funding — a longer hold, an exit when a held pair's
funding turns negative, and a cost hurdle at entry — lift the carry over the declared
adverse floor without weakening any gate, threshold or cost assumption?

## 2. What this design saw before it was written (contamination disclosure)

Everything in section 1 comes from P1.28's out-of-sample fold results over the same nine
test windows this family will be evaluated on. The rules below were chosen because those
results pointed at cost and at negative-funding weeks. This is a post-hoc-informed
declaration and is recorded as such: the nine folds are no longer an untouched test of
*the idea*; they remain a valid test of *these fixed rules* against the frozen gates,
with the usual multiple-testing correction inside the family. The final holdout from
2026-03-08 stays locked and was never read by any fold. Nothing in this family may be
revisited after its decision except through a new declaration.

Also seen: spot liquidity in 2026 (49 eligible spot legs on 2026-03-01, 35 on
2026-08-24, against the 40-pair floor). The universe rule is **not** changed; a thin
regime is skipped, as the protocol authorises, and the count of skipped weeks is reported.

## 3. Family declaration

Everything not stated here is identical to `funding_carry_panel_v1`
(`docs/superpowers/specs/2026-09-11-funding-carry-family-design.md`, sections 3, 6, 7, 8,
9): same 470 pairs and 7 exclusions, same universe rule, same cost tables (base and
adverse), same capital basis, same cohort mechanics, same fold geometry, same statistics,
same gates, same decision module.

### 3.1 Hypothesis

Among eligible spot/perpetual pairs selected on trailing four-week funding, a book that
holds cohorts for 26 weeks, exits a pair whose trailing one-week funding is non-positive,
or enters only pairs whose trailing funding covers a multiple of the round-trip cost,
earns funding net of both legs' costs and basis drift with positive expectancy under
base costs and non-negative expectancy under adverse costs, without single-fold, single-pair
or single-week concentration.

### 3.2 Members (four trials, each fixed)

| Member | Lookback | Hold | Exit rule | Hurdle multiple |
| --- | ---: | ---: | --- | ---: |
| `carry_l4w_h26w` | 4 | 26 | off | none |
| `carry_l4w_h13w_exit` | 4 | 13 | on | none |
| `carry_l4w_h26w_exit` | 4 | 26 | on | none |
| `carry_l4w_h26w_exit_hurdle2` | 4 | 26 | on | 2 |

`carry_l4w_h13w_exit` isolates the exit rule against P1.28's best member;
`carry_l4w_h26w` isolates the hold; the other two stack the levers. Four trials, one
Benjamini-Hochberg family at q ≤ 0.10.

### 3.3 Rules

**Exit rule.** At every weekly decision `t`, after the universe is selected and before the
book is assembled, every pair in a retained cohort whose trailing one-week funding
`F_1 = Σ rates in (t − 1 week, t]` is `≤ 0` (or has no settlement in that week) is
removed from every cohort that holds it. Its cohort share stays undeployed until the
cohort ages out, exactly as after a forced close (v1 `formed_size`). The legs leave the
book at `t` through ordinary turnover at one side each. A pair removed this way may be
re-selected by a later cohort if it qualifies again.

**Hurdle rule.** At cohort formation at `t`, a pair qualifies only if
`F_L × (H / L) / 2 ≥ k × c_rt(tier)`, where `F_L` is the trailing `L`-week funding sum
(per unit of perpetual notional), `H / L` scales it to the hold, `/ 2` converts to the
pair's unit-capital basis (half the capital is on the perpetual), and `c_rt(tier)` is the
base-scenario round-trip cost per unit of pair capital:
`c_rt = (spot_fee + perpetual_fee + 2 × slippage_tier) / 10 000` — 25 bps for tier one,
35 bps for tier two. With `k = 2`, `H = 26`, `L = 4` a pair needs `F_4 ≥ 15.4 bps` (tier
one) or `21.5 bps` (tier two) over four weeks. Pairs failing the hurdle are not in the
decile ranking; the decile size is still computed from the full eligible universe; fewer
than `minimum_selected` qualifying pairs is `NO_CARRY_COHORT`, not a skip.

**Hold 26.** `assemble_book` and the warm-up are unchanged; the warm-up covers `H_max − 1
= 25` Sundays before each fold's first decision. Cohort pruning uses the longest hold.

**Controls** use lookback 4 and hold 13 (the shortest member's), no exit rule, no hurdle:
`no_trade`, `random_pairs` (dominance), `all_pairs_ew` (context only).

### 3.4 Reported in addition to v1's extras

Per episode: `exit_rule_removals` (pairs removed by the exit rule at that decision) and
`hurdle_rejections` (pairs failing the hurdle at cohort formation). Both aggregate through
`extras_mean`.

### 3.5 Primary metric, budget, gates

Identical to v1 section 5.4 and 5.5 and section 9: mean net return per retained episode
under base costs; block bootstrap 2000 × block 4 × seed 17; BH q ≤ 0.10 over four
members; pooled floor 200; adverse aggregate ≥ 0; 2/3 positive folds; concentration ≤ 0.5
per fold, pair and episode; dominance over `no_trade` and `random_pairs`.

## 4. Data

The two repaired captures bound in P1.28 (perpetual `28c419e2…`, spot `ef631edd…`), the
same walk-forward calendar (split manifest `f33b7b08…`). A new manifest is published for
the v2 family hash. Costs are the v1 tables; a family on *measured* costs from the
Binance cost journal (separate spec, `2026-09-12-binance-cost-journal-design.md`) will be
declared as `funding_carry_panel_v3` once the journal is eligible, not by editing this
one.

## 5. Modules

- `carry_config.py`: `CarryMember` gains `exit_on_negative_funding: bool = False` and
  `hurdle_multiple: Decimal | None = None`; member names are validated against a frozen
  set per `family_name` (`funding_carry_panel_v1` keeps its three, `funding_carry_panel_v2`
  the four above); `CarryFamilySpec.family_name` becomes the literal of both names.
- `panel_config.load_family_spec` dispatches both carry family names.
- `carry_signals.py`: `round_trip_cost_bps(cost_table, tier)`, `hurdle_threshold(...)`,
  `select_member_cohort(..., hurdle=None)` where `hurdle` maps pair id → minimum `F_L`;
  `exit_rule_pairs(cohorts, trailing_one_week) -> set[str]`.
- `carry_fold_run.py`: computes trailing one-week funding for every held pair, applies the
  exit rule before assembly (same `_without_pairs` path as forced closes), passes the
  hurdle to member cohort formation, emits the two new extras, warm-up of `H_max − 1`
  Sundays. `_CARRY_MODULES` unchanged (same files).
- Everything else unchanged: accounting, universe, manifest, decision, CLI.

## 6. Artifacts

`artifacts/carry/carry-v2-walk-forward-<date>-usdt-pairs-1d-w1-v1.json`,
`artifacts/carry/funding-carry-v2-<date>-fold<i>-v1.json`,
`artifacts/carry/funding-carry-v2-<date>-decision-v1.json`, registry
`artifacts/carry/metadata-funding-carry-v2.sqlite3`, decision record
`P1_29_DECISION_<date>.md`, backlog item P1.29.

## 7. Decision rule

A member that passes every gate becomes the repository's first eligible candidate and
the base over which a learned abstention gate (logistic → GRU twin → CTM, in that order,
per protocol section 10) may be declared. A fail with a base-positive, adverse-negative
long-hold member confirms that the remaining lever is measured execution cost, and the
next declaration waits for the journal. No threshold moves either way.
