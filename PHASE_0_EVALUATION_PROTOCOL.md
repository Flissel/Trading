# Phase 0 Evaluation and Promotion Protocol

Status: Draft v0.2 (v0.1 of 2026-08-24 plus section 16 of 2026-09-09)
Date: 2026-08-24
Applies to: BTC and ETH perpetual research at 15-minute, 1-hour, and 4-hour forecast horizons

## 1. Purpose

This protocol prevents model novelty, repeated backtesting, and favorable cost assumptions from being mistaken for a tradable edge.

Forecast quality, economic utility, risk behavior, and operational correctness are separate gates. Failure of one gate cannot be offset by strength in another.

## 2. Dataset partitions

### 2.1 Chronological partitions

The first implementation must create:

- development history for feature and pipeline debugging;
- rolling training windows;
- rolling validation windows for hyperparameter and policy selection;
- rolling out-of-sample windows for candidate evaluation;
- one final holdout period that remains unopened until thresholds and candidate configuration are locked.

Initial rolling schedule:

```text
training:   trailing 12 months minimum
validation: following 3 months
OOS test:   following 1 month
step:       1 month
```

Longer training contexts may be tested, but they are separate registered experiments. The same OOS month must never move back into training within the same reported fold.

### 2.2 Purging and embargo

- Samples whose labels overlap a validation or test boundary are removed from the preceding partition.
- Embargo length is at least the maximum 4-hour label horizon plus the maximum feature lookback whose state could cross the boundary.
- Alternative-data events use `received_time`, not publication text or later corrections, for availability.
- Hyperparameters, thresholds, ensemble weights, and abstention rules are frozen before each OOS fold.

### 2.3 Final holdout

- The final holdout is opened once for a named release candidate.
- A failed final holdout does not become a new validation set.
- Any post-holdout change creates a new research generation and requires a later untouched period before a new live-readiness claim.

## 3. Experiment registry

Every experiment receives an immutable ID and records:

- hypothesis and expected mechanism;
- dataset and feature-schema versions;
- exact train, validation, OOS, purge, and embargo boundaries;
- source-rights manifest;
- code commit or content hash;
- model and environment configuration;
- random seeds;
- cost scenario;
- all attempted hyperparameters, not only the winner;
- forecast, economic, risk, and compute results;
- acceptance or rejection reason.

Abandoned and negative experiments remain in the registry. Deleting failed trials is prohibited.

## 4. Baselines

All candidates compete on identical folds against:

- unconditional event probability;
- no-change/random-walk forecast;
- simple momentum rule;
- simple mean-reversion rule;
- ridge regression;
- gradient-boosted trees;
- standard volatility model;
- best previously accepted sequence model;
- no-trade policy;
- 1x passive long exposure as a contextual benchmark, not as a perpetual-policy target.

The strongest eligible baseline, not the weakest baseline, is used for promotion comparisons.

## 5. Dependence and uncertainty

Fifteen-minute decisions and multi-hour labels overlap. Ordinary IID confidence intervals are therefore not accepted.

- Confidence intervals use stationary or moving-block bootstrap.
- Block length is selected from measured return, residual, and position autocorrelation and may not be shorter than the maximum label horizon.
- Economic resampling keeps complete position episodes together.
- Results are reported across time folds, assets, regimes, and seeds.
- Effective sample size is reported separately from raw decision count.

## 6. Multiple-testing control

- Every evaluated candidate and material variant counts as a trial family member.
- Primary metrics and direction of improvement are registered before evaluation.
- False-discovery control uses Benjamini-Hochberg with `q <= 0.10` within a declared experiment family.
- Strategy-level Sharpe is accompanied by a deflated or multiple-testing-aware Sharpe assessment.
- Parameter selection surfaces must be inspected for broad stability; isolated optimum spikes are rejection evidence.
- Re-running with different seeds or windows until success without registering failures is prohibited.

## 7. Cost scenarios

### 7.1 Diagnostic

Uses optimistic execution assumptions only to debug model and policy behavior. It cannot support promotion.

### 7.2 Base

Uses documented venue fees, observed spread, size-aware slippage, funding, modeled decision-to-order latency, partial-fill rules, minimum order sizes, and rejected-order behavior.

### 7.3 Adverse

At minimum:

- fees no lower than the base case;
- spread and slippage multiplied by two;
- latency increased to the measured high-percentile condition;
- unfavorable funding stress;
- lower fill probability;
- reconnect or stale-feed periods excluded from tradable time.

Exact numeric inputs are estimated from the data and locked before policy promotion.

## 8. Forecast promotion gate

A forecast model may enter the ensemble challenger registry only if all conditions hold:

1. no leakage, timestamp, schema, or reproducibility failure;
2. positive Brier Skill Score against the appropriate unconditional baseline in aggregate;
3. Brier Skill Score is positive in at least two thirds of eligible OOS folds;
4. median pinball-loss improvement over the strongest eligible baseline is positive;
5. nominal prediction-interval coverage error is no greater than 5 percentage points after calibration;
6. no single asset, fold, or regime supplies more than half of total measured forecast improvement;
7. calibration does not collapse in the high-volatility regime;
8. inference latency fits inside the decision deadline;
9. uncertainty and abstention behavior are defined;
10. all attempted variants are included in the multiple-testing record.

Directional accuracy alone can never promote a model.

## 9. Trading-policy promotion gate

A deterministic or learned policy may enter shadow trading only if:

1. its input forecast has passed the forecast gate or it is an explicitly registered direct-policy baseline;
2. aggregate base-cost net expectancy is positive;
3. the 95% block-bootstrap lower confidence bound for base-cost net expectancy is above zero;
4. at least two thirds of eligible OOS folds have positive base-cost net PnL;
5. adverse-cost aggregate net PnL is non-negative;
6. no single asset, fold, regime, or trade episode contributes more than half of total net PnL;
7. maximum simulated drawdown stays within the 10% hard account-loss boundary;
8. daily loss, exposure, position, leverage, and order constraints are never violated;
9. minimum effective sample size is at least 200 independent-equivalent position episodes or the candidate remains `INSUFFICIENT_EVIDENCE`;
10. small parameter, latency, and cost perturbations do not reverse the complete result;
11. performance exceeds the no-trade policy and provides incremental utility over the strongest simple trading baseline.

`INSUFFICIENT_EVIDENCE` is not a pass or a fail and cannot authorize live capital.

## 10. CTM-specific gate

The CTM is accepted only as a challenger until it demonstrates at least one of:

- forecast-skill improvement over the best non-CTM model on locked OOS folds;
- statistically supported incremental ensemble value;
- materially better calibration or abstention without reducing net economic utility;
- useful regime-state representation that improves a downstream policy under the same data and cost assumptions.

It must also report training time, peak VRAM, inference time, energy/runtime proxy, and parameter count. A gain that disappears when the baseline receives comparable preprocessing and tuning is rejected.

## 11. Alternative-data gate

News, on-chain, or any later policy-approved social evidence is admitted only through an ablation:

```text
market-only baseline
  versus
market + one alternative source family
```

Admission requires:

- identical market features and folds;
- source availability based on receipt time;
- rights manifest allowing the tested use;
- positive incremental OOS forecast skill or net utility after additional costs;
- robustness to removing the highest-impact events;
- no dependence on a single source, author, or retrospective event label.

## 12. Contextual-bandit and RL gate

The learned allocator must beat the locked deterministic expected-value policy under identical observations, actions, costs, and constraints.

- Training is offline and versioned.
- Reward includes net PnL, drawdown, tail risk, turnover, and constraint penalties.
- Evaluation includes unseen regimes and action perturbations.
- Reward hacking and simulator exploitation tests are mandatory.
- A learned policy that improves PnL while worsening the hard drawdown boundary is rejected.
- RL receives no exception from the 95% lower-bound, adverse-cost, fold-stability, or effective-sample gates.

## 13. Shadow and paper promotion

Historical promotion authorizes shadow forecasts only.

Paper promotion additionally requires:

- forecast timing meets deadlines in real time;
- simulated order state reconciles after restart;
- stale or missing data produces no trade;
- fault injection cannot bypass risk;
- predicted costs and observed paper costs are compared continuously;
- model and feature versions are immutable for the evaluation run.

The minimum paper observation period is 8 weeks and must include at least 200 independent-equivalent position episodes. If the episode threshold is not reached, the observation period extends.

## 14. Tiny-live boundary

This protocol does not authorize tiny-live trading. A later explicit approval requires:

- every preceding gate passed;
- account-specific OKX eligibility refreshed;
- trade-only credentials with no withdrawal permission;
- EUR 100 maximum funded capital;
- EUR 10 hard cumulative loss kill switch;
- 1x leverage;
- action-time confirmation before funding or enabling live execution.

## 15. Revisit conditions

Revisit this protocol when:

- observed trade frequency makes the effective-sample threshold impractical;
- the decision interval or maximum horizon changes;
- a new venue or instrument is added;
- transaction-cost evidence contradicts the base scenario;
- the system becomes a commercial product;
- an independent statistical review recommends a stricter gate.

Any relaxation must be prospective. Thresholds may not be weakened to rescue an already evaluated candidate.

## 16. Panel research addendum (v0.2, 2026-09-08)

This addendum extends section 1 to weekly-decision, one-week-holding research
on a panel of Binance USD-M USDT perpetuals. It changes no numeric gate.

- A position episode is one weekly portfolio rebalance; contracts inside an
  episode are not independent samples (PHASE_0_DATASET_SPEC section on
  cross-series dependence).
- The 200-episode floor of section 9.9 applies to the pooled out-of-sample
  episode count of a family.
- Block length for the bootstrap is at least the holding period and is fixed
  before evaluation; four weekly episodes for a one-week holding period.
- Cost scenarios follow section 7 with turnover-based charging; execution
  assumptions that are not measured are declared per family and are at least
  as adverse as the measured BTC evidence.
- Families without fitted parameters register every member as a trial and
  perform no validation selection.
- Non-learned panel rules are direct-policy baselines under section 9.1;
  section 8 does not apply to them.

### 16.1 Clarification (2026-09-10)

Prompted by ICPUSDT: a dead contract, not a live series with a gap (flat
price and zero `quote_volume` for 104 days, then five days absent from both
Binance dumps). Clarifies two implementation corrections without changing
any numeric gate or parameter:

- The dataset-quality stop (panel spec section 12 step 3) discounts, per
  instrument, days a capture attempted from Binance's daily dump and
  recorded absent there too, naming the discounted days in the walk-forward
  manifest; an unattempted or otherwise-unexplained missing day still stops
  the family at the unchanged threshold of three.
- "Trailing N-day" windows for realised volatility and for the liquidity
  median (panel spec sections 8.1 and 7.1) are calendar spans ending at the
  decision, not counts of available observations; a history-length
  requirement stated as a bar count remains a bar count.
