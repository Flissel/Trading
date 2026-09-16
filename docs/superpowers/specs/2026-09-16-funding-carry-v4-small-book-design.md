# Funding Carry v4 Family Design (declared capital, slot book, measured costs)

Status: Decided autonomously on 2026-09-16 under the standing goal "make it better, decide
autonomously until a reliable solution exists", on two facts the user supplied on
2026-09-15 when asked: the book that would really be run is **under 10 000 USDT**, and the
Binance fee tier is **standard taker without BNB** (spot 10 bps, perpetual 5 bps per side).
The user also chose to pre-register this family now, to be evaluated as soon as the Binance
cost journal's receipt exists and independently of how P1.30 comes out. `research_only`.
Pre-registers experiment family `funding_carry_panel_v4` (base declaration) and its measured
form `funding_carry_panel_v4_measured` (the evaluated family, backlog P1.33) under
`PHASE_0_EVALUATION_PROTOCOL.md` sections 1 to 16 as they stand on 2026-09-16, before any
fold is run and before the receipt exists. No live, paper, or shadow authority follows.

## 1. Purpose

P1.28 and P1.29 established that the spot-hedged funding carry has a real, statistically
distinguishable premium (Sharpe 2.0 to 2.8, Benjamini-Hochberg q below 0.01) and that what
defeats it under the declared adverse costs is execution: assumed slippage of 10 and 20 bps
per side on tier-two legs, doubled funding payments, and one fold carrying more than half
the profit. P1.30 re-declares the same v2 book on the journal's measured slippage at 5 000
and 50 000 USDT per order — the cost model of a book of several million USDT.

That is not the book the user would run. At a book of at most 10 000 USDT the v2
construction — up to ten pairs per weekly cohort held for 26 weeks, about two hundred pair
positions at once — means orders of about 20 USDT per leg, which lot-size rounding and
minimum notionals make unexecutable as a hedge. A small book holds a handful of pairs,
sized so that every order is large enough to fill cleanly and small enough to sit at the
top of the order book. This family declares that book explicitly: ten equal pair slots on a
10 000 USDT book, so at most 500 USDT per leg per order, and reads the journal's receipt at
the notional those orders actually have. The cost model then matches the capital, in both
directions: base costs at 500 USDT and adverse costs at ten times that, exactly as the v3
rule takes its adverse case at ten times its base notional.

## 2. What was seen before this design (disclosure)

- P1.28's and P1.29's fold results on the nine test windows, and P1.32's.
- An interim, read-only preview of the journal's tier medians at 3 580 rounds, taken on
  2026-09-15 with a scratch script and never published: at 500 USDT the tier-two spot leg's
  p50 of p50s was 1.47 bps and at 5 000 USDT its p50 of p90s 4.82 bps; tier one 0.11 and
  0.65 bps. The preview is cost data, not return data. The reading rule below copies the
  structure of the journal spec's section 5 (base p50 at the order notional, adverse p90 at
  ten times it) and its notional follows from the declared capital, so nothing in the rule
  was chosen from the preview. The preview is recorded so that the final receipt can be
  compared with it in the decision record.
- No fold of this family has been run; the holdout from 2026-03-08 stays locked.

## 3. The book

### 3.1 Capital and slots

The declaration carries a `capital` block: `book_usdt` 10 000 (an upper bound; the user's
book is below it, and a smaller book means smaller orders, for which the 500 USDT tier is
conservative), `pair_slots` 10, `per_leg_notional_usdt` 500 = `book_usdt / (2 × pair_slots)`,
`fee_tier` "standard_taker_no_bnb". Each slot holds one spot/perpetual pair on one tenth
of capital: spot +1/20, perpetual −1/20, unit gross and zero net exposure for a full book.
An empty slot is cash. Capital is never redistributed between slots.

### 3.2 Filling, holding, releasing

At every weekly decision (Sunday close), in this order, for each member:

1. **Release.** A slot whose pair has been held for `H` weeks (`hold_weeks`) is emptied.
   A slot whose pair lost a leg's bar at this decision, or whose leg was force-closed in the
   previous episode, is emptied (P1.28's untradeable and forced-close rules).
2. **Exit rule** (members with `exit_on_negative_funding`): a slot whose pair's trailing
   one-week funding is non-positive or absent is emptied (P1.29's rule, per slot).
3. **Fill.** Empty slots are filled from the ranked list of paying pairs — the eligible pairs
   with positive trailing `L`-week funding (`L` = `lookback_weeks` = 4), sorted by trailing
   funding descending, ties by pair id, cut to the top decile exactly as
   `select_member_cohort` cuts it (`max(minimum_selected, eligible // 10)`; fewer than
   `minimum_selected` paying pairs means no fill this week, counted as `no_fill_weeks`) —
   skipping pairs already held and, for members with the exit rule, pairs whose trailing
   one-week funding is non-positive or absent (v2's treatment of its freshly formed
   cohort), in rank order, until no slot is empty or the list is exhausted. A pair
   released in step 1 that still ranks is filled again in the same decision; its weights
   do not change, so no turnover is charged, and its hold clock restarts.

The book is the union of the filled slots. Turnover, fees, slippage, funding and forced
closes are P1.28's accounting unchanged (`carry_accounting.evaluate_carry_episode`).

### 3.3 Fold boundaries

There is no warm-up: every slot is empty at a fold's first decision and is filled there
(`warm_up_weeks` 0, no `FOLD_OPENING_BOOK_WARMED_FROM_PRIOR_WEEKS`). A week with
`UNIVERSE_TOO_SMALL` empties every slot, as it resets every cohort under v2, and the slots
refill at the next tradeable decision. The final exit at a fold's end is not charged, as in
every panel family (`FOLD_FINAL_EXIT_COST_UNCHARGED`); because this book has no warmed
cohorts whose exits fall in-window, the uncharged share is larger than under v2, so every
fold report states `uncharged_final_exit_cost` per member and scenario — the cost of
liquidating the last episode's drifted book at that scenario's fees and tier slippage — and
the decision record subtracts it from any member that passes narrowly.

## 4. Family declaration

Everything not stated here is `funding_carry_panel_v2`'s: the 470 pairs and 7 excluded
pairs, the universe rule (top 100 by 30-day median quote volume, floor 40, tier one the top
20), the selection rule (decile, minimum 8), the four-week lookback, the folds, the
statistics and the gates, the decision module.

### 4.1 Hypothesis

At a declared book of at most 10 000 USDT held in ten equal pair slots — long spot, short
perpetual, orders of at most 500 USDT per leg at Binance's standard taker fees — a slot
that is filled from the top decile of eligible pairs by trailing four-week funding, held
for up to 13 or 26 weeks, refilled when it empties, and (two members) vacated when its
pair's trailing one-week funding is non-positive, earns funding net of both legs' measured
execution costs and basis drift, with positive expectancy under base costs and non-negative
expectancy under adverse costs, without single-fold, single-pair or single-week
concentration, after correction across four members.

### 4.2 Members (four trials, each fixed)

| Member | Lookback | Hold (max) | Exit rule | Book |
| --- | ---: | ---: | --- | --- |
| `carry_s10_l4w_h13w` | 4 | 13 | off | slots |
| `carry_s10_l4w_h13w_exit` | 4 | 13 | on | slots |
| `carry_s10_l4w_h26w` | 4 | 26 | off | slots |
| `carry_s10_l4w_h26w_exit` | 4 | 26 | on | slots |

No hurdle member: P1.29 showed the top decile already clears twice the round trip.

### 4.3 Controls (not trials)

- `no_trade`;
- `random_pairs` (dominance): the same slot book, `H` 13 (the shortest member's), no exit
  rule, slots filled in the seeded random order of P1.28's control instead of by funding;
- `all_pairs_ew` (context only): P1.28's construction — every paying pair, equal weight,
  weekly cohorts held the shortest member's 13 weeks — run **without warm-up** under this
  family (section 3.3 sets `warm_up_weeks` 0 for the whole family), so its book ramps 1/13
  per week at every fold start; it is **not comparable** to P1.28's or P1.30's warmed
  `all_pairs_ew`. It is not executable at the declared capital and is reported as context,
  never as a gate. (corrected 2026-09-16, before any fold ran)

### 4.4 Costs

Fees: spot 10 bps and perpetual 5 bps per side, both scenarios — the user's declared tier.
Slippage: measured, under the journal spec's section 5.1 (added 2026-09-16): base
`tier_p50_of_p50` of the worse leg at the smallest ladder notional at least the per-leg
order notional (500 USDT), adverse `tier_p50_of_p90` at ten times that (5 000 USDT), each
rounded up to the next whole basis point, the receipt hash cited. Funding multipliers 1/1
base and 0.75/2 adverse, forced close ×1/×2, P1.28's. The base declaration
`configs/funding-carry-panel-v4.json` carries v2's assumed tiers (5/10 and 10/20) as
placeholders that the measured declaration replaces; the base family is never evaluated.

### 4.5 Primary metric, budget, gates

P1.27's sections 5.4/5.5 and 9 unchanged; BH q ≤ 0.10 over four members; pooled floor 200;
two thirds positive folds; concentration ≤ 0.5 per fold, contract and episode; dominance
over `no_trade` and `random_pairs` in both scenarios.

### 4.6 Reported extras

Per episode: `filled_slots`, `slot_fills`, `slot_releases` (age-outs and lost-bar releases),
`exit_rule_removals`, `no_fill` (0/1); per fold, member and scenario: `uncharged_final_exit_cost`.

## 5. Modules

- `carry_config.py`: `CarryCapital` (`book_usdt`, `pair_slots`, `per_leg_notional_usdt`,
  `fee_tier`; the notional must equal `book_usdt / (2 × pair_slots)`); `CarryMember.book`
  (`"cohorts"` default, `"slots"`); family names `funding_carry_panel_v4` and
  `funding_carry_panel_v4_measured`, `MEMBER_NAMES_V4`, a base-to-measured family map
  replacing the single `MEASURED_COST_FAMILY`; a slot family requires `capital`, a cohort
  family forbids it; `CostEvidenceReference.rule` accepts section 5's or section 5.1's
  sentence and the citation carries the declared notionals.
- `cost_evidence_rule.py`: `CAPITAL_DECLARATION_RULE`, section 5.1's sentence verbatim.
- `binance_cost_journal.py`: the receipt carries `capital_declaration_rule` verbatim beside
  `declaration_rule`; the model validates both.
- `carry_measured_costs.py`: `measured_slippage_tiers(receipt, *, capital)`; declare and
  verify dispatch on the base declaration's `capital` block; the ladder notional is the
  smallest of the receipt's notionals at least the per-leg notional, refused if none.
- `carry_signals.py`: `fill_slots(...)`, `assemble_slot_book(...)`.
- `carry_fold_run.py`: the slot mode beside the cohort mode (one loop, two book
  assemblers), `warm_up_weeks` 0 for slot families, the extras above,
  `uncharged_final_exit_cost`.
- `cli.py`: `carry-declare-measured` and `carry-verify-measured` unchanged in interface.
- Chain: `C:/Users/User/.trading-jobs/carry-v4-chain.ps1` — refuses without the receipt;
  declare v4 measured → verify → manifest → nine `carry-fold`s → `panel-decision`; launched
  by the v3 chain on completion and runnable by hand.

## 6. Artifacts

`configs/funding-carry-panel-v4.json` (pre-registered base),
`configs/funding-carry-panel-v4-measured.json` (chain output, cites the receipt),
`artifacts/carry/carry-v4-walk-forward-usdt-pairs-1d-w1-v1.json`,
`artifacts/carry/funding-carry-v4-fold<i>-v1.json`,
`artifacts/carry/funding-carry-v4-decision-v1.json`, registry
`artifacts/carry/metadata-funding-carry-v4.sqlite3`; decision record
`P1_33_DECISION_<date>.md`; backlog P1.33.

## 7. Decision rule

A passing member is the first eligible candidate: a small-capital funding carry whose cost
model is the user's actual book and fee tier. The next step is then the protocol's, not a
new family: a single holdout read of that member, after the capture is extended to cover the
holdout. A fail says the spot-hedged funding carry does not pay its own execution at
standard taker fees even at the order sizes a small book has; the one remaining lever is
the fee tier itself (maker execution, or the BNB discount), which is the user's choice and
would be a new declaration. No threshold moves either way. The family is independent of
P1.30's outcome and is evaluated whether P1.30 passes or fails; the two records are written
on different days.
