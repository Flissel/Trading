# Phase 0 Initial Risk Policy

- Status: Draft v0.1 for simulation and paper validation
- Date: 2026-08-24
- Capital basis: EUR 100 equivalent
- Live authorization: none

## 1. Objective

The first objective is survival and measurement quality, not maximizing return.
The risk engine is deterministic, independent of models and agents, and fails
closed whenever required state is missing, stale, contradictory, or invalid.

This policy does not assert that EUR 100 is enough to trade economically after
fees, minimum order sizes, and slippage. Those constraints are part of the paper
evaluation.

## 2. Authority

Only the risk engine may approve a `DecisionIntent`. It may reduce or reject a
requested exposure but may never increase it.

Models and agents cannot:

- change risk limits;
- select leverage;
- bypass cooldowns or kill switches;
- transform an unknown execution state into a presumed rejection;
- authorize real-capital mode.

Every decision, including rejection, emits a `RiskDecision` and an audit event.

## 3. Initial paper limits

All values are EUR-equivalent at the current mark and are deliberately
conservative starting hypotheses:

| Control | Initial paper value |
|---|---:|
| Reference equity | EUR 100 |
| Maximum leverage | 1.0x |
| Maximum gross exposure | EUR 50 |
| Maximum exposure per instrument | EUR 25 |
| Maximum simultaneous directional positions | 2 |
| Maximum planned loss per new position | EUR 0.25 |
| Maximum realized plus unrealized daily loss | EUR 1.00 |
| Maximum rolling 7-day loss | EUR 2.50 |
| Drawdown kill switch from high-water mark | EUR 5.00 |
| Hard cumulative loss boundary | EUR 10.00 |
| Maximum open order intents per instrument | 1 |

BTC and ETH exposures are not treated as independent during market stress. The
EUR 50 portfolio cap applies before any per-instrument allowance.

These values can only be changed by a versioned policy revision decided before
the next evaluation window. They are never tuned on the final holdout.

## 4. Position sizing

The requested notional is the minimum of:

1. policy exposure derived from positive expected net value;
2. uncertainty-adjusted model allowance;
3. loss-at-stop allowance;
4. per-instrument cap;
5. remaining portfolio cap;
6. venue minimum/step-compatible notional.

If the venue-compatible minimum exceeds the safe notional, the order is
rejected. Rounding always reduces risk. Confidence alone never determines size.

No averaging down, martingale sizing, loss chasing, or automatic leverage
increase is permitted.

## 5. Entry gate

A new exposure requires all of the following:

- an unexpired `DecisionIntent` with positive expected return after the adverse
  cost estimate;
- an eligible calibrated forecast and complete lineage;
- no triggered kill switch or active cooldown;
- valid current instrument rules;
- healthy, non-stale market data and a valid book if book data is required;
- synchronized position, balance, and open-order state;
- a defined exit and maximum planned loss;
- projected post-order exposure inside every limit;
- supported execution semantics in the active non-live mode.

Failure of one condition rejects the entry.

## 6. Exit and protective behavior

Every position has at least one deterministic exit path based on invalidation,
time, risk, or protective price. Model disagreement may flatten or reduce a
position but cannot postpone a hard risk exit.

Protective behavior must account for gaps and slippage: a stop price is not a
guaranteed fill price. Backtests report planned and realized loss separately.

When data required for new decisions becomes stale, new exposure is blocked.
Existing exposure follows a predeclared safe-state policy; it is never left to a
free-text agent decision.

## 7. Kill switches

The system enters a no-new-risk state on any of the following:

- daily, weekly, drawdown, or cumulative loss limit reached;
- position or gross exposure limit breach;
- unknown or unreconciled order outcome;
- balance, position, or open-order reconciliation failure;
- invalid order-book sequence or required feed staleness;
- persistent cross-venue clock or price anomaly;
- repeated order rejection or rate-limit response;
- schema/hash/lineage validation failure;
- model artifact mismatch or forecast beyond its expiry;
- audit sink failure;
- process restart without successful state recovery.

The hard cumulative EUR 10 boundary is absorbing for the policy version: no new
risk is allowed until explicit human review and a new authorization. Restarting
a process does not reset risk state.

## 8. Cooldowns

- after a daily-loss trigger: no new risk until the next UTC day and review;
- after three consecutive losing position episodes: minimum 24-hour paper
  cooldown;
- after data-integrity or reconciliation failure: until the root condition is
  verified resolved and state is reconciled;
- after a drawdown or cumulative-loss trigger: manual review required.

Cooldowns use persisted state and survive restarts.

## 9. Cost and liquidity controls

Before approval, expected cost includes applicable fees, spread, modeled
slippage, funding over the expected holding interval, and an adverse uncertainty
buffer.

An order is rejected when:

- expected net return is not positive under the adverse scenario;
- requested size consumes more displayed liquidity than the configured
  participation cap;
- spread or short-term impact exceeds its frozen threshold;
- price or quantity cannot be represented exactly under venue rules.

Phase 1 must measure the minimum economically meaningful order against actual
OKX EEA paper/sandbox rules before any claim that EUR 100 is viable.

## 10. Mode isolation

Backtest, shadow, paper, and any later live mode use different configuration,
state directories, credentials, and explicit startup flags.

Phase 1 permits only:

- `backtest`;
- `shadow` without an execution route;
- `paper` through simulation or an explicitly configured sandbox.

`tiny_live` startup is rejected. Enabling it requires a separate reviewed
policy, account eligibility verification, operational runbook, explicit user
authorization, and a new secrets configuration. There is no automatic fallback
to a live endpoint.

## 11. Paper promotion evidence

This risk policy is considered operationally validated only after:

- all limit boundaries have unit and property tests;
- fault injection proves fail-closed behavior;
- restart tests preserve high-water mark, cooldowns, and kill switches;
- duplicate and unknown order scenarios reconcile safely;
- at least eight weeks and 200 independent-equivalent paper episodes meet the
  evaluation protocol;
- no hard-limit breach or unauthorized mode transition occurs.

Passing these checks still does not authorize or prove profitable live trading.
