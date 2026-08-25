# Phase 0 Research Charter

Status: Draft v0.3
Date: 2026-08-24
Scope: Research and paper-trading only

## 1. Mission

Build a multimodal, probabilistic crypto market intelligence and trading research system that can discover, measure, and validate an economic edge after realistic costs.

The system must separate:

1. observation and data interpretation;
2. probabilistic forecasting;
3. decision and portfolio allocation;
4. independent validation and risk enforcement;
5. order execution.

No model, LLM, agent, or reinforcement-learning policy receives unrestricted capital authority.

## 2. Success definition

Phase 0 does not define success as an exact price prediction or a profitable historical chart.

A research candidate is successful only when it demonstrates all of the following:

- reproducible, leakage-free results on locked out-of-sample windows;
- calibrated probabilistic forecasts;
- positive expected value after fees, spread, slippage, funding, and modeled latency;
- stability across multiple time windows and market regimes;
- incremental value over simple and linear baselines;
- compliance with the deterministic risk contract;
- complete provenance from raw event to forecast, decision, and simulated order.

No profitability claim is authorized by a backtest alone.

## 3. Initial research scope

### 3.1 Instruments

- BTC perpetual market data;
- ETH perpetual market data;
- primary venue: OKX EEA;
- reference venue: Binance USD-M public market data;
- initial OKX instruments: linear BTC and ETH perpetual swaps;
- Binance is a reference-data source, not an authorized live execution venue.

Later live eligibility remains account-specific and must be verified again at action time. Research-feed selection does not constitute a legal, regulatory, or account-eligibility claim.

### 3.2 Decision and forecast horizons

- decision interval: 15 minutes
- forecast horizons: 15 minutes, 1 hour, and 4 hours
- position states during initial policy research: long, flat, or short
- no high-frequency market making in the initial scope

The 15-minute decision interval is a starting hypothesis, not a profitability claim. The evaluation harness must also test whether lower turnover at 1-hour decisions produces better net results.

### 3.3 Operating mode

- historical replay
- shadow forecasts
- paper trading
- no real-capital trading during the initial program
- no autonomous live model promotion

Any later tiny-live stage requires a new explicit approval. Its initial capital ceiling is EUR 100 equivalent and its hard total-loss budget is EUR 10.

### 3.4 Current compute and storage constraints

Verified local research hardware on 2026-08-24:

- NVIDIA GeForce RTX 3060 with 12 GB VRAM;
- 32 GB system RAM, with availability shared with the rest of the system;
- `C:` Samsung 870 EVO 1 TB SATA SSD with 11.6 GB free;
- `E:` Toshiba 2 TB SATA HDD with 1.27 TB free;
- no NVMe drive currently available.

`E:` is inventory only and is excluded from all project reads, writes, caches,
temporary files, and automatic fallback paths because it has caused stalled
filesystem operations. The local MVP must therefore use bounded samples,
out-of-core processing, sharded Parquet datasets, mixed-precision training,
small batches, and bounded model sizes on `C:`. Large historical downloads are
deferred until reliable additional storage is installed and approved. A later
environment with approximately 480 GB aggregate VRAM is anticipated, but its
GPU topology, interconnect, and per-device memory remain unverified and are not
part of the MVP capacity claim.

## 4. Explicit non-goals

The initial program will not attempt to:

- guarantee exact future prices;
- build an LLM-to-exchange order path;
- support many assets or exchanges;
- perform autonomous leverage selection;
- train model weights directly inside the live execution process;
- implement HFT, latency arbitrage, or production market making;
- use reinforcement learning before a deterministic policy baseline exists;
- treat social sentiment as useful without incremental out-of-sample evidence;
- optimize infrastructure scale before an economic edge is demonstrated.

## 5. Time and information semantics

Every observation must distinguish at least:

- `event_time`: when the event occurred at the source;
- `published_time`: when the source made it public, when applicable;
- `received_time`: when the research system received it;
- `available_time`: the earliest instant the system could legally and
  technically use it;
- `processed_time`: when the current transformation completed;
- `decision_time`: the cutoff after which it is unavailable to the forecast;
- `label_time`: when the complete outcome becomes observable.

Features may only use information with `available_time <= decision_time`.
`available_time` may never precede a required publication or receipt time.

Late-arriving corrections must create a new dataset version. They must not silently rewrite the dataset used by an existing experiment.

## 6. Canonical data foundation

### 6.1 Market and microstructure data

Initial candidates:

- trades;
- best bid and ask;
- L2 order-book deltas and snapshots;
- spread and depth imbalance;
- mark, index, and last price;
- funding rates and funding timestamps;
- open interest;
- liquidations;
- volume and trade-sign imbalance;
- cross-venue basis and dislocation;
- venue status and feed health.

### 6.2 Alternative data

Alternative data enters only after the market-data baseline is operational:

- news;
- X/Twitter;
- on-chain transfers and exchange flows;
- stablecoin flows;
- optional Reddit or Telegram sources.

Every derived event must include source identity, content hash, novelty, reliability, confirmation count, uncertainty, and expected relevance half-life.

Phase 1 source decisions:

- news starts with official public announcements and feeds whose use and retention terms permit the research workflow;
- Coin Metrics Community API is the initial no-key on-chain network-metrics source;
- paid labeled exchange-flow data is deferred until basic on-chain metrics show incremental value;
- self-hosted Bitcoin or Ethereum nodes are outside the local MVP storage budget;
- X data is not an authorized model-training source under the currently published X Developer rules;
- X scraping or browser automation is prohibited;
- the X evidence interface will be developed against synthetic or expressly reusable fixtures;
- any later X API use requires a documented permitted use, official API access, prepaid credits, disabled auto-recharge, a hard spending cap, and explicit approval.

### 6.3 Dataset requirements

- immutable raw-event storage;
- versioned canonical events;
- deterministic feature generation;
- explicit gap and duplicate detection;
- schema validation;
- dataset manifest with code version, parameters, time coverage, and hashes;
- reproducible train, validation, and test membership.

### 6.4 Acquisition and storage strategy

The data program uses both sources below:

1. continuous first-party capture from OKX EEA and Binance USD-M public WebSocket feeds;
2. purchased historical BTC/ETH perpetual data for backfilling and walk-forward evaluation.

Official OKX historical downloads are the initial backfill source. As verified on 2026-08-24, OKX publishes tick-level trades from September 2021, perpetual funding from March 2022, and high-resolution L2 order-book data from March 2023. These files must still pass local completeness, schema, timestamp, reconstruction, and incident checks before use.

The initial Phase 1 paid-data budget is EUR 0. Tardis.dev remains the paid historical-data fallback for longer history or cross-venue Binance L2 coverage. Its currently visible Perpetuals subscriptions are not proportionate to the initial EUR 100 capital basis, and the one-off purchase control was not available in the visible order workflow when checked on 2026-08-24. A paid-data purchase requires a documented missing-data need, sample validation, a current quote, and explicit user approval. This preserves the planned self-collected plus historical-data hybrid without buying data before its incremental value can be measured.

Storage roles:

- current `E:` HDD: excluded from this project;
- future NVMe: active Parquet datasets, derived features, experiment caches, and training shards;
- system SSD: code, specifications, registry, and strictly bounded samples only
  until sufficient reliable free space exists.

## 7. Prediction targets

Raw future price is not the primary training target. The initial target family is:

### 7.1 Multi-horizon log return

For decision time `t` and horizon `h`:

```text
r(t,h) = log(mid_price(t+h) / mid_price(t))
```

Mid-price is the research reference. Executable entry and exit prices are modeled separately by the execution simulator.

### 7.2 Distributional targets

For every horizon:

- return quantiles: q05, q25, q50, q75, q95;
- probability that net return is positive after estimated costs;
- expected volatility;
- maximum favorable excursion;
- maximum adverse excursion;
- regime probabilities;
- abstention or uncertainty score.

### 7.3 Economic labels

An economic label must incorporate a documented execution assumption:

```text
net_return = gross_return - fees - spread - slippage - funding - latency_cost
```

Forecast quality and economic utility are evaluated separately. Neither may substitute for the other.

## 8. Model program

### 8.1 Required baselines

Every advanced model must compete against:

- no-change and random-walk forecasts;
- unconditional base rates;
- simple trend and mean-reversion rules;
- ridge regression;
- gradient-boosted trees;
- a standard volatility model;
- at least one established sequence model.

### 8.2 CTM World Model

The Continuous Thought Machine is an experimental challenger. Its proposed role is to encode evolving market state and produce probabilistic multi-horizon outputs.

It is not assumed to outperform simpler models. Acceptance depends on locked out-of-sample evidence, calibration, incremental ensemble value, and compute cost.

### 8.3 Residual and uncertainty model

A separate model may learn:

- conditional forecast error;
- regime-dependent failure modes;
- prediction-interval width;
- overconfidence and underconfidence;
- conditions requiring abstention.

This model must not have access to outcome information unavailable at the original decision time.

### 8.4 Forecast ensemble

The ensemble may combine models by horizon and regime. It must preserve each component forecast and weight for auditability.

No model may be included solely because it is architecturally novel.

## 9. Multi-agent interpretation contract

Initial agent roles:

- data-quality agent;
- market-microstructure agent;
- on-chain event agent;
- news event agent;
- social/X event agent;
- source-credibility agent;
- regime agent;
- contradiction and validation agent.

Agents emit structured evidence, not orders. A minimum evidence record is:

```json
{
  "evidence_id": "immutable-id",
  "source": "source-id",
  "asset": "BTC",
  "event_type": "exchange_outflow",
  "directional_effect": 0.18,
  "novelty": 0.72,
  "source_reliability": 0.84,
  "confirmation_count": 3,
  "expected_half_life_minutes": 180,
  "uncertainty": 0.41,
  "event_time": "timestamp",
  "received_time": "timestamp",
  "model_version": "content-addressed-version"
}
```

Free-form agent text is retained for research provenance but is not accepted by the policy, risk, or execution interfaces.

## 10. Forecast output contract

The initial canonical forecast contract is:

```json
{
  "forecast_id": "immutable-id",
  "instrument": "BTC-PERP",
  "decision_time": "timestamp",
  "horizon_minutes": 60,
  "return_quantiles": {
    "q05": -0.018,
    "q25": -0.004,
    "q50": 0.006,
    "q75": 0.015,
    "q95": 0.033
  },
  "probability_net_positive": 0.67,
  "expected_volatility": 0.021,
  "regime_probabilities": {
    "trend": 0.58,
    "mean_reversion": 0.17,
    "high_volatility": 0.20,
    "unknown": 0.05
  },
  "uncertainty": 0.39,
  "model_versions": ["version-id"],
  "dataset_version": "dataset-id"
}
```

Production contracts will use decimal-safe numeric encoding and explicit units. The JSON above is conceptual.

## 11. Decision policy

The deterministic baseline policy acts on expected utility after costs and may select:

- `LONG`;
- `SHORT`;
- `FLAT`;
- `NO_TRADE_UNCERTAIN`;
- `NO_TRADE_DATA_QUALITY`;
- `NO_TRADE_RISK`.

The policy must calculate and record:

```text
expected_value =
    P(win) * expected_win
  - P(loss) * expected_loss
  - fees
  - spread
  - slippage
  - funding
  - risk_penalty
```

A contextual bandit or offline RL policy may be evaluated only after this baseline is locked.

## 12. Evaluation protocol

### 12.1 Dataset splitting

- expanding or rolling walk-forward evaluation;
- train, validation, and out-of-sample windows separated chronologically;
- purging around overlapping labels;
- embargo between model selection and evaluation windows;
- one untouched final holdout period;
- hyperparameters and acceptance thresholds locked before opening the final holdout.

### 12.2 Forecast metrics

- pinball loss for quantiles;
- CRPS or an equivalent proper distributional score;
- Brier score for event probabilities;
- calibration error and reliability diagrams;
- interval coverage;
- directional accuracy as a secondary metric;
- error by horizon, asset, venue, and regime.

### 12.3 Economic metrics

- net expectancy;
- net PnL;
- maximum drawdown;
- Sharpe and Sortino with uncertainty estimates;
- profit factor;
- turnover;
- tail loss;
- exposure and concentration;
- average and worst modeled execution cost;
- result by confidence bin and regime.

### 12.4 Cost scenarios

At minimum:

- optimistic diagnostic scenario;
- base scenario using defensible venue assumptions;
- adverse scenario with wider spread, higher slippage, increased latency, and funding stress.

Only base and adverse scenarios may support promotion decisions.

### 12.5 Candidate promotion gate

A candidate may enter shadow evaluation only if:

- all leakage and reproducibility checks pass;
- forecast probabilities show positive skill over the appropriate base rate;
- net expectancy is positive in the aggregate base-cost scenario;
- performance is positive in at least two thirds of eligible walk-forward windows;
- no single asset, regime, or window supplies more than half of total net profit;
- performance remains operationally viable under the adverse-cost scenario;
- parameter and latency perturbations do not reverse the complete result;
- uncertainty or abstention behavior is defined;
- all experiment artifacts and multiple-testing history are preserved.

Statistical confidence thresholds will be finalized after dataset frequency and effective independent sample size are measured. They must be locked before the final holdout is opened.

## 13. Reinforcement-learning boundary

RL is a decision-policy experiment, not a substitute for forecast evaluation.

Initial action space:

```text
target_position in {-0.25, -0.10, 0.00, 0.10, 0.25}
```

Initial reward family:

```text
reward =
    realized_net_pnl
  - drawdown_penalty
  - tail_risk_penalty
  - turnover_penalty
  - constraint_violation_penalty
```

The environment must model costs, order latency, partial fills, funding, and unavailable actions. RL cannot promote itself, modify risk policy, or deploy model weights.

## 14. Deterministic risk contract

The risk layer has final veto authority.

### 14.1 Order and position controls

- allowed instruments and venues;
- maximum order notional;
- maximum position notional;
- maximum position change per decision;
- price-deviation and slippage limits;
- liquidity participation limit;
- reduce-only enforcement when required;
- no withdrawal credentials in any automated runtime.

### 14.2 Portfolio controls

- initial paper and tiny-live capital basis: EUR 100 equivalent;
- maximum gross exposure: 1.0 times account equity;
- maximum position notional: EUR 50 equivalent;
- maximum simultaneous positions: two;
- planned maximum loss per trade: EUR 0.50 equivalent;
- daily loss stop: EUR 2 equivalent;
- hard total drawdown kill switch: EUR 10 equivalent;
- no martingale or loss-driven position increases;
- initial leverage: 1x only;
- gross and net exposure limits;
- correlated exposure limit;
- volatility budget;
- daily loss budget;
- rolling drawdown limit;
- concentration limit;
- funding exposure limit.

### 14.3 System controls

- market-data freshness limit;
- reference-venue consistency check;
- order-rate limit;
- reconciliation health;
- model-version allowlist;
- dataset and feature-schema compatibility;
- global kill switch;
- fail-flat behavior on ambiguous state.

Venue minimum-order constraints, fees, and contract multipliers may force smaller positions or `NO_TRADE_RISK`; they may never justify increasing these limits. Higher leverage may only be evaluated as a separately labeled research challenger and is not approved for paper or live promotion.

## 15. Model lifecycle

```text
New data
  -> offline candidate training
  -> reproducible evaluation
  -> challenger registry
  -> shadow forecasts
  -> paper-trading gate
  -> explicit promotion
```

Rules:

- the live or paper champion is frozen and versioned;
- new labels do not update champion weights in process;
- drift can stop trading or trigger retraining, but cannot auto-promote;
- rollback must restore the complete model, feature, policy, and configuration bundle;
- every forecast records the exact bundle version.

## 16. Component authority matrix

| Component | May observe | May propose | May approve risk | May place orders |
|---|---|---|---|---|
| Interpretation agents | data and approved context | structured evidence | no | no |
| Forecast models | canonical features | distributions and uncertainty | no | no |
| Decision policy | approved forecasts | target position | no | no |
| Validator | artifacts and results | accept or reject candidate | no | no |
| Portfolio service | approved targets and positions | desired exposure | no | no |
| Risk service | portfolio, venue, and system state | approved or reduced intent | yes | no |
| Execution engine | approved intent and venue state | concrete order | no | yes |

## 17. Planned experiment sequence

1. `E00` data integrity and timestamp audit
2. `E01` random-walk, no-change, and base-rate forecasts
3. `E02` simple rule and ridge baselines
4. `E03` gradient boosting and standard sequence models
5. `E04` market-data-only forecast ensemble
6. `E05` structured news, social, and on-chain evidence
7. `E06` CTM World Model challenger
8. `E07` residual and uncertainty calibration
9. `E08` regime-aware ensemble
10. `E09` deterministic expected-value policy
11. `E10` contextual bandit challenger
12. `E11` offline RL challenger
13. `E12` event-driven execution simulation
14. `E13` shadow and paper trading
15. `E14` fault injection and recovery

Experiments may stop early when a gate fails. Complexity is not a reason to continue a rejected line.

## 18. Phase 0 open decisions

These decisions must be resolved before implementation planning is complete:

1. additional NVMe capacity and installation compatibility;
2. topology and availability of the later 480 GB VRAM environment;
3. account-specific OKX EEA eligibility before any later live stage;
4. source-specific retention periods after each applicable data license is reviewed;
5. final statistical promotion thresholds after effective sample-size analysis;
6. paid-data trigger and budget only if official OKX plus self-collected feeds leave a demonstrated research gap;
7. written policy basis or permission before any X-derived feature is admitted to model training or automated trading decisions.

Resolved in Draft v0.2:

- BTC and ETH perpetuals as the initial market;
- OKX EEA as primary venue and Binance USD-M as reference feed;
- self-collected and purchased historical data in parallel;
- official OKX history as the initial free backfill and EUR 0 paid-data budget for Phase 1;
- official/public news plus Coin Metrics Community as the initial alternative-data baseline;
- no X scraping and no X-derived model training under the currently published policy;
- local RTX 3060 / 12 GB VRAM MVP constraint;
- EUR 100 virtual and later tiny-live capital ceiling;
- EUR 10 hard tiny-live total-loss budget;
- 1x initial leverage ceiling.

## 19. Phase 0 deliverables and definition of done

Phase 0 is complete when the following are reviewed and accepted:

- this Research Charter;
- architecture decision record for the initial engine and storage choices;
- canonical event, evidence, forecast, decision, risk, and order schemas;
- dataset and timestamp specification;
- evaluation and multiple-testing protocol;
- experiment registry convention;
- initial risk-policy specification;
- resolved initial venue, data-access, and compute decisions;
- implementation backlog for Phase 1.

No repository-current, connector-ready, data-available, compliant, profitable, or live-safe claim is made by this draft.

Current supporting decisions and protocols:

- `ADR_001_DATA_SOURCES.md`
- `ADR_002_ENGINE_AND_STORAGE.md`
- `PHASE_0_CONTRACTS.md`
- `PHASE_0_DATASET_SPEC.md`
- `PHASE_0_EVALUATION_PROTOCOL.md`
- `PHASE_0_RISK_POLICY.md`
- `PHASE_1_BACKLOG.md`
