# Cross-Sectional Daily Momentum Family Design

Status: Approved in chat on 2026-09-09 (draft 2026-09-08). `research_only`.
Approved for implementation planning only. Pre-registers experiment family `xs_momentum_panel_v1` under
`PHASE_0_EVALUATION_PROTOCOL.md` with the prospective addendum in section 13
of this document. No live, paper, or shadow authority follows from it.

## 1. Purpose

Answer one question before any further model work: does a simple,
few-parameter momentum signal survive this repository's cost and
multiple-testing gates on **some** timescale, given that P1.15 rejected every
candidate at the 15-minute cadence with 1-hour and 4-hour labels?

P1.15 (`P1_15_DECISION_2026-08-25.md`) found that the binding constraint was
the adverse-cost scenario, not forecast skill. The CTM mixture design
(`2026-09-08-ctm-moe-challenger-design.md`) is honest that raw skill
improvement is its least likely outcome and that its value would be
abstention and regime state over a frozen base. A residual over a base with
no edge is meta-labeling over nothing. This family therefore changes the
question, not the model: daily decision cadence, weekly holding, a panel of
USDT perpetuals instead of one instrument, and the momentum definitions that
carry the strongest published evidence across asset classes.

The family is deliberately small, fully predeclared, and free of parameter
search. Every member counts as a trial. If the family fails, the result is as
useful as a pass: it closes the "wrong timescale" objection to P1.15 and
redirects effort to mechanical edges (funding/basis carry) rather than to a
fourth learner.

Decision rule on completion (section 12): a pass makes the passing member the
primary signal and re-targets the CTM work as abstention and regime state over
**that** base; a fail keeps CTM `research_only` and moves the next question to
funding/basis carry with the OKX cost journal as evidence.

## 2. Evidence Review

### 2.1 Time-series momentum

Moskowitz, Ooi, Pedersen, *Time Series Momentum*, Journal of Financial
Economics 104(2), 2012, 228–250. Across 58 futures and forwards over more than
25 years, the past 12-month excess return of an instrument positively predicts
its next-month return; the effect persists for about a year and partially
reverses afterwards. Positions are sign-of-trailing-return, scaled by ex-ante
volatility. Two parameters: lookback and holding period.

### 2.2 Cross-sectional momentum in cryptocurrency

Liu, Tsyvinski, Wu, *Common Risk Factors in Cryptocurrency*, Journal of
Finance 77(2), 2022, 1133–1177. On a broad weekly coin panel, three factors,
market, size, and momentum, price the cross-section; ten characteristics form
long-short strategies with sizeable excess returns, and momentum measured over
one to four weeks is among them. Weekly rebalancing, value-weighted portfolios.

Fieberg, Liedtke, Poddig, Walker, Zaremba, *A Trend Factor for the Cross
Section of Cryptocurrency Returns*, Journal of Financial and Quantitative
Analysis 60(7), 2025, 3116–3153 (open access, doi 10.1017/S0022109024000747).
More than 3,000 coins, April 2015 to May 2022, weekly. An ML aggregate of 28
technical signals (CTREND) earns 3.87% per week long-short on value-weighted
quintiles, survives transaction costs, persists in large and liquid coins, and
renders plain momentum insignificant in its presence. Their Figure 1 reports
lower average return and Sharpe for plain cross-sectional momentum (CMOM)
than for CTREND. This is the honest ceiling for the present family: plain momentum is
the weaker cousin of a signal that needs machine learning to aggregate. The
present family tests the cousin first because it has two parameters and no
fitting; CTREND-style aggregation is a possible later family, not this one.

A recent comparative study (Vilnius University, *Momentum Trading in
Cryptocurrencies: A Comparative Study of Time-Series and Cross-Sectional
Strategies*, eight major coins, 2020-01 to 2025-10, daily rebalancing, gross of
costs) found time-series momentum stronger than cross-sectional momentum,
attributing the cross-sectional weakness to high correlation among coins, and
found both converging to modest positive returns in 2024–2025. Two lessons are
taken: include time-series members alongside cross-sectional ones, and expect
the recent sub-period to be the hardest.

### 2.3 This repository's own result

P1.15: 580 days of BTC 15-minute bars, horizons 4 and 16 bars, three folds.
Every candidate rejected. Momentum and mean-reversion rules rejected on both
horizons; the 4-hour boosted stumps were base-positive in 3/3 folds but
adverse-negative with worst BH q = 0.274. The rules tested there are
sign-of-last-bar rules at 15-minute cadence (`strategy.py:42-53`), not the
weekly-to-quarterly lookbacks the literature supports.

`PHASE_0_DATASET_SPEC.md:267-269` already states that multiple instruments do
not create independent samples when driven by the same market episode and
that effective sample size must account for cross-series dependence. Nothing
in code implements that today; section 9 does.

## 3. Problem Definition

### 3.1 Target and cadence

- Venue and instruments: Binance USD-M USDT-margined perpetuals. Binance is
  the only venue whose complete daily history for several hundred perpetuals,
  including delisted ones, is public and free (`data.binance.vision`). OKX
  remains the primary venue for the 15-minute research line; this family does
  not change that.
- Decision cadence: once per week at the close of the last daily bar of the
  ISO week (Sunday 23:59:59.999 UTC, the `close_time` of the Binance daily
  kline). Positions are held one week and re-formed at the next decision.
- Target: the net return of a unit-gross long-short portfolio over the holding
  week, after fees, slippage, funding, and forced closes. There is no
  per-instrument forecast to score; the forecast gate of protocol section 8
  does not apply, and the family is registered as a **direct-policy baseline**
  under protocol section 9.1.
- Real time: irrelevant at weekly cadence; the decision deadline is the next
  daily bar open, one day after the decision close.

### 3.2 Data budget

Verified on 2026-09-08 against the public bucket:

| Fact | Value |
| --- | --- |
| USDT perpetuals in `fapi/v1/exchangeInfo` | 658 (528 trading, 129 settling, 1 pending) |
| Onboarded before 2024-01-01 / 2025-01-01 | 231 / 360 |
| Delisted symbols still in the dumps | yes (`LUNAUSDT` 1d files 2021-01 to 2022-05, `FTTUSDT`, `SRMUSDT`, `ANCUSDT`, `BTSUSDT` present) |
| Daily kline schema | `open_time, open, high, low, close, volume, close_time, quote_volume, count, taker_buy_volume, taker_buy_quote_volume, ignore` |
| Funding history | `data/futures/um/monthly/fundingRate/<SYMBOL>/`, since 2020-01, schema `calc_time, funding_interval_hours, last_funding_rate` |
| Volume on disk | below 100 MB for all symbols, daily bars plus funding, compressed |

Survivorship bias is avoidable because delisted contracts remain in the
dumps. Ticker recycling is real (spot `LUNAUSDT` continues after the 2022
collapse as a different coin while the perpetual ends 2022-05); identity is
therefore bound to the contract, not the symbol (section 7.2).

### 3.3 Hardware

CPU only. Decimal arithmetic in pure Python over roughly 1.5 million daily
rows and a few hundred weekly rebalances finishes in minutes. No GPU, no
torch. The storage reserve of 20 GB on C: stays in force; E: remains excluded
by `StoragePolicy` (`storage.py:11-52`).

## 4. Design Options

### Option A — push the panel through the existing bar pipeline

Rejected. The bar pipeline is single-instrument by construction: sample ids
carry venue and timestamp only (`bar_research.py:98-102`), split samples must
be strictly chronological (`splits.py:130-132`), every consumer collapses the
dataset into one OKX stream and one Binance stream
(`research_run.py:53-54`, `walk_forward_run.py:84-85`), and PnL has no
portfolio dimension (`evaluation.py:47-83`). Forcing N coins per timestamp
through it means rewriting all of that under the guise of reuse.

### Option B — new panel modules over the instrument-agnostic primitives (recommended)

One sample is one **rebalance date**, not one coin-day. That single choice
keeps `SplitSample`, `WalkForwardConfig`, the fold geometry, the purge rule,
the final-holdout logic, and the manifest format unchanged
(`splits.py`, `walk_forward_run.py`), because the sample series is strictly
chronological again. The panel is consumed inside the sample: a rebalance
sample's "signal" is a weight vector over eligible contracts and its "outcome"
is the realised portfolio return. Reused verbatim: `CostScenario` and
`round_trip_cost_return` (`strategy.py:8-39`),
`block_bootstrap_mean_interval` and `block_bootstrap_mean_test`
(`evaluation.py:86-131`), `benjamini_hochberg` (`evaluation.py:134-146`),
`MetadataRegistry` (`registry.py`), `StoragePolicy`, atomic write and
refuse-to-overwrite conventions, report field names, and the decision-record
template of `P1_15_DECISION_2026-08-25.md`. New and isolated: capture from the
public dumps, universe and identity rules, weight construction, portfolio
accounting with turnover costs and funding, and a decision module that applies
the protocol's episode floor at the campaign level. Section 10 lists them.

### Option C — NautilusTrader per ADR 002

Deferred. ADR 002 chooses NautilusTrader as the eventual replay engine, but no
NautilusTrader code exists in the repository today and the family needs no
event-driven simulation: weekly decisions on daily closes are a portfolio
accounting problem. Introducing the engine here would add a dependency and a
second code path before the first is proven. The panel modules are written so
their weight vectors could later be replayed through an engine.

## 5. Family Declaration (pre-registration)

This section is the registration record. It is frozen when this spec is
approved and committed, and mirrored byte-for-byte in
`configs/xs-momentum-panel-v1.json`; the plan may not add members, lookbacks,
universes, or cost variants without a registered revision.

### 5.1 Hypothesis

Trailing-return momentum on a liquid panel of USDT perpetuals, held one week,
has positive net expectancy under the base cost scenario and non-negative net
expectancy under the adverse scenario, with the effect not concentrated in a
single fold, contract, or episode.

### 5.2 Members (six trials, each fixed, no validation selection)

| Member | Construction | Parameters |
| --- | --- | --- |
| `xs_mom_1w` | rank by trailing 1-week return; long top quintile, short bottom quintile; equal weight within legs; 0.5 gross per leg | lookback 7 days |
| `xs_mom_4w` | same | lookback 28 days |
| `xs_mom_12w` | same | lookback 84 days |
| `ts_mom_4w` | per contract, sign of trailing return; weight proportional to `sign / ex-ante vol`, capped; gross normalised to 1 | lookback 28 days, vol window 30 days |
| `ts_mom_12w` | same | lookback 84 days, vol window 30 days |
| `xs_rev_1w` | the protocol's simple mean-reversion rule: `xs_mom_1w` with legs swapped | lookback 7 days |

Weights are formed from information available at the decision close only.
No member has a fitted parameter; the walk-forward training window is kept
for geometry and for the eligibility history requirement, not for fitting.

### 5.3 Controls (not trials, not promotable)

- `no_trade`: zero weights. Reference for dominance.
- `random_ranks`: quintile long-short on a uniformly random ranking with
  `random_seed = 17`, re-drawn per rebalance from a deterministic stream.
  Reference for dominance.
- `passive_long_ew`: equal-weight long of the eligible universe, unit gross.
  Contextual benchmark only, per protocol section 4; never a dominance
  control for a long-short policy.

The protocol's simple mean-reversion rule (`xs_rev_1w`) is a trial, not a
control: as the mirror of `xs_mom_1w` it would be a vacuous dominance control,
whereas counting it in the family makes the BH gate stricter.

### 5.4 Primary metric and direction

Primary metric: mean net return per rebalance episode under the **base** cost
scenario, pooled over all out-of-sample test folds. Direction: positive.
Secondary, all required for eligibility: adverse-cost pooled mean non-negative;
positive base-cost net PnL in at least two thirds of folds; 95% moving-block
bootstrap lower bound of the pooled base mean above zero; Benjamini-Hochberg
q ≤ 0.10 within this family of six; no single fold, contract, or episode
contributing more than half of pooled net PnL; dominance over the strongest
control under both scenarios.

### 5.5 Budget

Two attempts, following the Phase-A convention: attempt 0 is this
declaration; attempt 1 requires a written revision reason that does not
depend on attempt 0's test results (for example a data defect). A third
attempt is a new family with a new name.

## 6. Data and Capture

### 6.1 Sources

- Daily klines: `https://data.binance.vision/data/futures/um/monthly/klines/<SYMBOL>/1d/<SYMBOL>-1d-<YYYY-MM>.zip`, plus the daily files of the current month when the monthly file is not yet published. Span: 2020-01-01 to the last complete month before capture (2026-08 at the time of writing).
- Funding: `https://data.binance.vision/data/futures/um/monthly/fundingRate/<SYMBOL>/<SYMBOL>-fundingRate-<YYYY-MM>.zip`.
- Contract list: one snapshot of `https://fapi.binance.com/fapi/v1/exchangeInfo` at capture time, used only to enumerate symbols and to record `onboardDate`. Its `status` field is current, not historical, and is **not** used for eligibility (section 7.1).

**Clarification (2026-09-10, after approval):** the daily kline dumps are also used to patch interior holes in a completed capture's historical monthly aggregates -- a documented gap in Binance's monthly klines, not a defect in the capture or a symptom of a bad fetch -- not only to cover the current month before its own monthly file is published. This widens the scope of the daily-dump clause above; it changes no pre-registered parameter of the family. The source host, CSV schema, venue, and interval are unchanged, a 404 on the daily dump is recorded as a genuine absence exactly like an absent monthly source, and every gap-filled row is recorded under its own source kind (`klines_daily_fill`), distinct from the monthly `klines` kind, so the manifest's provenance stays auditable.

`_ALLOWED_HOSTS` (`market_capture.py:27`) gains `data.binance.vision` and
`s3-ap-northeast-1.amazonaws.com` for the bucket listing.

### 6.2 Layout and hashing

The capture follows the existing v2 layout and hash chain
(`market_capture.py:199-294`, `candle_dataset.py`): every downloaded zip is
kept as an immutable raw source with `raw_sha256`; the dataset is Parquet with
Zstandard and canonical Decimal strings; `dataset-manifest.json`,
`quality-report.json`, and `capture-manifest.json` carry the same fields and
root hashes so `verify_candle_capture` extends rather than forks.

Two deviations, both required by ADR 002:

- Partitioning is `dataset=daily_candles/venue=BINANCE_UM/instrument=<SYMBOL>/part-00000.parquet`, one file per contract, and
  `dataset=funding/venue=BINANCE_UM/instrument=<SYMBOL>/part-00000.parquet`. A `date=` partition would create a few hundred thousand tiny files, which ADR 002 section 3 forbids.
- The quality report is per instrument (rows, first and last `open_time_ns`, missing days, duplicates) plus the global totals, because the existing global scalars cannot tell a delisting from a gap.

`interval_ns` is `86_400_000_000_000`. `available_time_ns` is `close_time_ns + 1`.
Funding rows carry `calc_time_ns`, `funding_interval_hours`, and `rate` as a
canonical Decimal.

Capture name: `data/captures/<YYYY-MM-DD>-binance-um-usdt-perps-1d-<span>`.
Storage preflight uses `StoragePolicy.authorize` with a worst-case estimate of
500 MB; the reserve stays 20 GB.

## 7. Universe and Identity

### 7.1 Eligibility at decision time `t`

A contract is eligible for the rebalance at `t` if all hold, using only data
with `available_time_ns <= t`:

1. at least 91 daily bars exist with `close_time_ns <= t` (history for the
   longest lookback plus the volatility window);
2. a bar exists whose `close_time_ns == t` (the contract traded through the
   decision day);
3. the trailing 30-day median `quote_volume` is at least 5,000,000 USDT;
4. it ranks within the top 100 eligible contracts by that median.

If fewer than 40 contracts are eligible, the rebalance is recorded as
`UNIVERSE_TOO_SMALL` and every member holds zero weights for that week. Such
weeks are reported but excluded from the pooled episode series and from the
200-episode count, because they hold no position.

Delisting after `t` is not known at `t` and must not be used. If a held
contract has no daily bar on some day of the holding week, the position is
closed at its last available close (a forced close, section 8.3). This is
realised outcome, not look-ahead.

### 7.2 Identity

`contract_id = f"{symbol}:{first_open_time_ns}"`, where `first_open_time_ns`
is the first daily bar in the dumps. A recycled ticker produces a new
`contract_id`. `onboardDate` from `exchangeInfo` is recorded for
reconciliation but does not define identity.

## 8. Portfolio Construction and Cost Model

### 8.1 Weights

Cross-sectional members: eligible contracts are ranked by the trailing
return `close[t] / close[t - lookback] - 1`; the top quintile receives
`+0.5 / n_long`, the bottom quintile `-0.5 / n_short`; quintile size is
`floor(n_eligible / 5)`, at least 8.

Time-series members: raw weight `sign(trailing return) / sigma_30d`, where
`sigma_30d` is the trailing 30-day standard deviation of daily log returns,
annualised, with a floor of 0.20; raw weights are capped so no contract exceeds
`2 / n_eligible` in absolute value, then scaled so that the sum of absolute
weights is exactly 1. Members are therefore comparable at unit gross; net
exposure may differ from zero for time-series members, and that exposure is
reported.

`xs_rev_1w` is `xs_mom_1w` with legs swapped. `random_ranks` replaces the
trailing return with the random stream. `passive_long_ew` is `1 / n_eligible`
on every eligible contract.

### 8.2 Gross return of an episode

For each held contract, the holding-period return is
`close[t + 7d] / close[t] - 1` (or the forced-close return). The gross
portfolio return is the weight-sum of holding returns. Returns are computed
in Decimal from canonical strings; nothing is floated except inside the
bootstrap, which already accepts Decimal.

### 8.3 Costs

Trading costs are charged on **turnover**, not on a full round trip per week:
`turnover = sum |w_new - w_old_drifted|`, where `w_old_drifted` is the previous
weight vector after the holding-week price drift. This differs from the bar
pipeline, which charges a full round trip per active bar
(`evaluation.py:66`), because weekly panels rebalance partially.

| Component | Base | Adverse |
| --- | --- | --- |
| Fee (Binance USD-M standard taker, documented) | 5 bps per side | 5 bps per side |
| Half-spread plus slippage, contracts ranked 1–20 by median volume | 5 bps per side | 10 bps per side |
| Half-spread plus slippage, contracts ranked 21–100 | 10 bps per side | 20 bps per side |
| Funding | actual historical rate per funding event, paid by longs when positive, received by shorts | favourable receipts set to zero, unfavourable payments doubled |
| Forced close (delisting or missing bar during the week) | closed at last available close, one side of cost | same, cost doubled |
| Latency, partial fills, minimum order size | not modelled at weekly cadence; recorded as an assumption | same |

The 5 and 10 bps execution figures are predeclared assumptions, not
measurements; the OKX cost journal measures BTC only. They are conservative
relative to observed spreads on the top contracts (the BTC perpetual journal
shows 0.01 to 0.02 bps spread) and deliberately punitive on the long tail.
`CostScenario` carries them: `fee_bps_per_side = 5`,
`slippage_bps_per_side = 5 or 10`, `spread_multiplier = 1`, `funding_bps = 0`
(funding is applied from data, not from the scalar), and the adverse scenario
doubles slippage. The per-episode net return is
`gross - turnover * per_side_cost - funding_cost - forced_close_cost`.

### 8.4 Reported per member and scenario

Pooled OOS: episode count, mean and median net return, total net return, win
rate, maximum drawdown, annualised Sharpe, deflated Sharpe (section 9.4),
mean turnover, mean gross and net exposure, largest single-fold share of net
PnL, largest single-contract share, largest single-episode share, number of
forced closes, number of `UNIVERSE_TOO_SMALL` weeks.

## 9. Evaluation Geometry and Gates

### 9.1 Folds

`WalkForwardConfig` over rebalance samples, with `sample_id =
f"BINANCE_UM:{decision_close_ns}:w1"`, `decision_time_ns = decision_close_ns`,
`label_end_time_ns = decision_close_ns + 7 days`:

```text
training:    365 days   (geometry and history only; nothing is fitted)
validation:   91 days   (unused for selection; kept so the manifest matches the protocol)
OOS test:    182 days
step:        182 days
embargo:      14 days   (holding week plus one week of slack)
final holdout: last 182 days, locked
```

With data from 2020-01-05 to 2026-08-31 this yields about nine test folds
whose test windows tile 2021-05 to 2025-10 (a tenth window would cross the
locked holdout and is dropped by the geometry loop), about 225 pooled weekly
episodes after purging; the exact counts are bound by the manifest, and the plan must fail closed if the
pooled episode count is below 200 before any member is evaluated
(`INSUFFICIENT_EVIDENCE`, not a rejection).

The purge rule of `splits.py:118-123` applies unchanged: an episode whose
holding week crosses a boundary is dropped from the earlier partition.

Protocol section 2.2 asks for an embargo of the label horizon plus any
feature lookback whose *state* could cross the boundary. The members' only
inputs are fixed transforms of public past closes (trailing returns and a
trailing standard deviation) with no fitted or estimated state, so the
lookback carries no partition-specific information and the embargo covers the
holding week plus one week of slack.

### 9.2 Episodes and dependence

One weekly portfolio episode is one independent-equivalent position episode.
The 40 or so contract positions inside it are **not** counted separately;
this is how the family satisfies `PHASE_0_DATASET_SPEC.md:267-269`. The
protocol's floor of 200 episodes (section 9.9) is applied to the pooled OOS
count, not per fold, because a 26-week fold cannot hold 200 weeks. This is an
instantiation of the existing rule for a new cadence, not a relaxation; the
per-fold floor in `fold_evaluation.py:40-47` is a bar-cadence implementation
detail and is not reused.

### 9.3 Statistics

- Bootstrap: `block_bootstrap_mean_interval` and `block_bootstrap_mean_test`
  on the pooled OOS episode series, block length 4 (one month, above the
  one-week label horizon), 2,000 repetitions, `random_seed = 17`, confidence
  0.95, one-sided p-value for a positive mean.
- Multiple testing: `benjamini_hochberg` over the six members' pooled
  one-sided p-values; gate `q <= 0.10`. Controls are excluded from the family
  count because they are not candidates for promotion.
- Fold positivity: `required_positive = (2 * fold_count + 2) // 3`, the same
  ceiling rule as `fold_aggregate.py:104`.

### 9.4 Deflated Sharpe

The protocol requires a deflated or multiple-testing-aware Sharpe alongside
the strategy Sharpe (section 6). None exists in code. The panel decision
module implements the Deflated Sharpe Ratio of Bailey and López de Prado
(2014) with the number of trials fixed at six, the observed skewness and
kurtosis of the pooled weekly series, and `T` equal to the pooled episode
count. It is **reported**, with the BH gate remaining the binding
multiple-testing control; a DSR below 0.95 is recorded as a caution in the
decision record.

### 9.5 Eligibility of a member

`eligible_for_further_review` requires all of:

1. pooled episodes ≥ 200 (else `INSUFFICIENT_EVIDENCE`);
2. pooled base mean > 0 and bootstrap lower bound > 0;
3. pooled adverse mean ≥ 0;
4. positive base net PnL in at least two thirds of folds;
5. BH q ≤ 0.10;
6. no fold, contract, or episode contributes more than half of pooled base
   net PnL;
7. base and adverse pooled net PnL exceed the strongest control's
   (`no_trade`, `random_ranks`).

Anything else is `rejected` with the same reason-code vocabulary as
`fold_aggregate.py` and `phase_a_decision.py`
(`AGGREGATE_BASE_NET_NON_POSITIVE`, `AGGREGATE_ADVERSE_NET_NON_POSITIVE`,
`POSITIVE_FOLD_FRACTION_NOT_MET`, `MULTIPLE_TESTING_GATE_NOT_MET`,
`EPISODE_FLOOR_NOT_MET`, `BASE_CONTROL_DOMINANCE_NOT_MET`,
`ADVERSE_CONTROL_DOMINANCE_NOT_MET`, and new
`CONCENTRATION_LIMIT_EXCEEDED`).

The final holdout is not opened by this family. An eligible member earns a
registered release-candidate name and nothing else.

## 10. Modules and Interfaces

All new modules live in `src/trading_bot/`, torch-free, Decimal for money,
integers for nanoseconds, frozen dataclasses, atomic writes, refuse to
overwrite. Each has one job and a test file.

| Module | Job | Depends on |
| --- | --- | --- |
| `panel_capture.py` | list symbols, download and verify zips, publish daily-candle and funding datasets with manifests and per-instrument quality | `candle_dataset.py` helpers, `storage.py`, `market_capture.py` hashing |
| `panel_universe.py` | `contract_id`, eligibility per decision time, liquidity tiering, `UNIVERSE_TOO_SMALL` | dataset readers |
| `panel_signals.py` | trailing returns, ex-ante volatility, the six members and three controls as `dict[str, WeightVector]` per rebalance | `panel_universe.py` |
| `panel_accounting.py` | holding returns, drift, turnover, funding events, forced closes, base and adverse episode net returns | `strategy.CostScenario` |
| `panel_samples.py` | rebalance calendar, `SplitSample` construction, manifest build via `walk_forward_run` | `splits.py`, `walk_forward_run.py` |
| `panel_fold_run.py` | per-fold report in the `candidates[]` field schema of the bar runners, with panel-specific extras (turnover, exposure, forced closes) | all above, `evaluation.py` bootstrap |
| `panel_decision.py` | pooled statistics, BH, DSR, concentration limits, control dominance, decision report | `evaluation.py` |
| `cli.py` additions | `panel-capture`, `panel-manifest`, `panel-fold`, `panel-decision`, each with `--workspace-root`, `--reserve-bytes`, and bounded-path checks | existing parser |

Interfaces:

```python
@dataclass(frozen=True, slots=True)
class WeightVector:
    decision_close_ns: int
    weights: tuple[tuple[str, Decimal], ...]   # (contract_id, weight), sum |w| == 1 or 0
    reason_codes: tuple[str, ...]              # e.g. ("UNIVERSE_TOO_SMALL",)

@dataclass(frozen=True, slots=True)
class EpisodeResult:
    sample_id: str
    member: str
    scenario: str                              # "base" | "adverse"
    gross_return: Decimal
    turnover: Decimal
    trading_cost: Decimal
    funding_cost: Decimal
    forced_close_cost: Decimal
    net_return: Decimal
    gross_exposure: Decimal
    net_exposure: Decimal
    contract_contributions: tuple[tuple[str, Decimal], ...]
```

Registration uses the existing `MetadataRegistry.experiments` table
(`registry.py:90-102`): one `ExperimentRecord` per member and fold with
`family_id = uuid5(NAMESPACE_URL, f"{split_manifest_hash}:xs_momentum_panel_v1")`,
`candidate_name` the member name, `hypothesis` the section 5.1 text,
`code_hash` over the panel modules plus `evaluation.py` and `strategy.py`.
The Phase-A `hypothesis_registry.py` is not used: its family names are a
closed `Literal` and its schema hard-codes horizons 4 and 16.

## 11. Artifacts and Naming

- Capture: `data/captures/<capture-date>-binance-um-usdt-perps-1d-<span>`.
- Frozen family declaration: `configs/xs-momentum-panel-v1.json` (members, controls, universe rules, cost table, fold geometry, statistics), validated with `extra="forbid"`, its SHA-256 recorded in every report as `family_spec_hash`.
- Manifest: `artifacts/panel-walk-forward-<capture-date>-usdt-perps-1d-w1-v1.json`.
- Fold reports: `artifacts/panel/xs-momentum-<capture-date>-fold<K>-v1.json`, one per fold, all members inside.
- Decision report: `artifacts/panel/xs-momentum-<capture-date>-decision-v1.json`.
- Registry: `artifacts/panel/metadata-xs-momentum-v1.sqlite3`.
- Decision record: `P1_27_DECISION_<date>.md` at the repository root, in the
  `P1_15_DECISION_2026-08-25.md` template, with the gate table extended by
  turnover and DSR columns.
- Backlog: `PHASE_1_BACKLOG.md` gains `### P1.27 Cross-sectional daily momentum family` with acceptance bullets equal to section 9.5, added by the implementation plan.

## 12. Sequencing, Stop Conditions, and Decision Rule

1. Approve this spec; commit it together with the protocol addendum of
   section 13 **before** any panel data is downloaded. The addendum is
   prospective by construction.
2. Write the implementation plan (`writing-plans`), TDD per module.
3. Capture; verify hashes; publish per-instrument quality. Stop if any
   contract's daily series has more than 3 missing days inside its listed
   span (record `CAPTURE_QUALITY_FAILED`, fix the capture, do not touch the
   family).
4. Build the manifest. Stop with `INSUFFICIENT_EVIDENCE` if pooled OOS
   episodes are below 200.
5. Run all six members and three controls on every fold in one invocation
   per fold; no member may be run alone or ahead of the others.
6. Produce the decision report and the decision record. Commit
   `configs/`, `artifacts/panel/`, and the record.

Decision rule:

- **At least one member eligible.** Register it as the primary signal
  candidate. The CTM design's base becomes that member's weight vector at
  weekly cadence; the CTM work is re-scoped as abstention and regime state
  over it, which is what its section 3.4 already says it is best at. Plan 2
  of the CTM design is not written before this re-scope.
- **No member eligible.** CTM remains `research_only` under its existing
  sequencing conditions. The next registered question is mechanical
  funding/basis carry, using the OKX cost journal as evidence, not a fourth
  learner on BTC 15-minute bars.
- **`INSUFFICIENT_EVIDENCE`.** Extend the data span or the universe by a
  registered revision (attempt 1), never by loosening the episode floor.

Prohibited: re-running with a different universe threshold, lookback set, or
cost table after seeing test results; opening the final holdout; reporting
the best member without the other five.

## 12.1 Addendum (2026-09-10): calendar-aware windows and the capture-quality gate

ICPUSDT prompted this addendum. Its close is exactly 6.44 USDT for 104
consecutive days, 2022-06-10 through 2022-09-21: it trades 14,142,060 in
`quote_volume` on the first of those days, then `quote_volume` is exactly
zero for the following 103 days, 2022-06-11 through 2022-09-21 — a dormant
contract, not a live series with an ordinary publishing gap. Five days
immediately follow, 2022-09-22 through 2022-09-26, absent from both
Binance's monthly and daily dumps. ICPUSDT is not delisted: it resumes
trading on 2022-09-27 at `quote_volume` 31,000,000 and continues through
2026-08-31 (1,934 rows), eligible in the panel's universe at 236 of the 348
decisions in this capture. Repairing a production capture from the daily
dumps surfaced implementation gaps against this document while
investigating that instrument; all are fixed for every future capture, not
only this one. **No pre-registered numeric gate or parameter changes.**

1. **The capture-quality gate (section 12 step 3) now discounts days proven
   absent at source.** A missing day the capture actually attempted from
   Binance's daily dump, and which the dump itself answered with a 404, is
   not a defect the capture could have prevented; it is a hole in Binance's
   own published data. The capture manifest already records the attempt (a
   `sources` entry of kind `klines_daily_fill`, `status: "absent"`, naming
   the symbol and the exact day); the gate now subtracts, per instrument,
   the count of such proven-absent days before comparing against the
   pre-registered threshold of three missing days, which is unchanged. A day
   never attempted this way, or attempted and missing for any other reason,
   still counts in full, and every subtraction is cross-checked against the
   instrument's own recomputed missing days so a manifest cannot name a day
   as absent at source that the dataset does not actually show as missing.
   The walk-forward manifest records exactly which days were discounted, per
   instrument, in an `absent_at_source_days` block, so the limitation
   travels in the immutable artifact rather than living only in a log line.
2. **`sigma_30d` (section 8.1) is a calendar window, as this document always
   specified.** "Trailing 30-day standard deviation" means the 30 calendar
   days ending at the decision, not the last 30 available observations
   regardless of the span they cover. On a series with an interior hole the
   two disagree: an observation-count window reaches further back and folds
   a multi-day price move into what it treats as a single daily return,
   understating volatility. The window must be complete — every day
   present, no hole — or the contract simply drops out of the time-series
   members at that decision. The implementation is corrected to match the
   specification text; the specification itself does not change.
3. **The trailing 30-day liquidity median (section 7.1 item 3) is a calendar
   window and must likewise be complete.** A median computed over whichever
   days happen to survive a hole is not a safe substitute for the days that
   are missing: missing days are not missing at random, they concentrate on
   halted, dormant, and delisting-adjacent contracts, which is exactly where
   the removed days are the low-volume ones — and the liquidity floor is a
   one-sided gate, so a partial median is biased toward *admitting*
   contracts a complete window would correctly exclude. On this capture, of
   27,940 contract-decision pairs that fail the floor with a complete
   window, removing the three lowest in-window days flips 4.1% of them to
   passing; five days, 6.8%; ten days, 14.3%. Requiring completeness is free
   here: it changes zero of the 348 decisions in this capture, and no
   admitted contract-decision pair anywhere in it has an incomplete window.
   Section 7.1 item 1's at-least-91-daily-bars history requirement is
   explicitly a bar count, not a calendar span, and is unaffected.

None of this changed a result. The corrected and uncorrected weight vectors
are identical at every one of the 348 decisions in this capture, verified
directly rather than only argued: neither the calendar-aware volatility
window nor the completeness requirement on the liquidity median changes a
single universe snapshot or weight vector anywhere in it. At the five
weekly rebalances whose 30-day volatility window straddles ICPUSDT's absent
days, ICPUSDT is outside the eligible universe at all five: its liquidity
window is itself incomplete at four of them (2022-10-02, 2022-10-09, and
2022-10-16 with five missing days each, 2022-10-23 with three), and at the
fifth, 2022-09-25, it has no bar at the decision close at all, because that
Sunday falls inside the gap. The pre-registered gate was, in this capture,
blocking the entire family over a contract already outside the eligible
universe at every decision where the fix could matter. The fix is a
correctness guard for every future capture, not a change to any result
reported from this one.

## 13. Protocol Addendum (prospective, to be appended to `PHASE_0_EVALUATION_PROTOCOL.md`)

```text
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

Prompted by ICPUSDT: a dormant contract, not a live series with an ordinary
publishing gap. Its close is flat at 6.44 USDT for 104 days, 2022-06-10
through 2022-09-21; `quote_volume` is exactly zero for 103 of those days
(2022-06-11 onward, after trading 14,142,060 on the first day). Five days
then follow, 2022-09-22 through 2022-09-26, absent from both Binance dumps.
It is not delisted: it resumes trading on 2022-09-27 and continues through
2026-08-31. Clarifies three implementation corrections without changing any
numeric gate or parameter:

- The dataset-quality stop (panel spec section 12 step 3) discounts, per
  instrument, days a capture attempted from Binance's daily dump and
  recorded absent there too, naming the discounted days in the walk-forward
  manifest; an unattempted or otherwise-unexplained missing day still stops
  the family at the unchanged threshold of three.
- "Trailing N-day" windows for realised volatility and for the liquidity
  median (panel spec sections 8.1 and 7.1) are calendar spans ending at the
  decision, not counts of available observations; a history-length
  requirement stated as a bar count remains a bar count.
- Both of those windows must be complete -- every day present, no hole --
  not merely calendar-bounded: a partial statistic over an incomplete
  window is not a safe substitute, and for the liquidity median in
  particular it is biased toward admitting contracts a complete window
  would correctly exclude, since missing days concentrate on halted,
  dormant, and delisting-adjacent contracts.
```

## 14. Risks and Honest Priors

- **Correlation.** Coins move together; cross-sectional legs may net to
  market-neutral noise. The time-series members exist for this reason.
  Prior: time-series members more likely to pass than cross-sectional ones.
- **Recent sub-period.** Published evidence weakens in 2024–2025. About a
  third of the OOS folds fall there. A pass driven only by 2021–2022 would
  fail the fold-positivity and concentration gates, which is intended.
- **Funding on the short leg.** Momentum shorts are recent losers whose
  funding is often negative, so shorts pay. This is charged from actual data
  in base and doubled in adverse. It is the most likely reason for an
  adverse-cost rejection.
- **Execution assumptions.** 5 and 10 bps per side are declared, not
  measured. A member that passes only with 5 bps and fails with 10 bps is
  reported as such; the decision record must show both tiers' contributions.
- **Universe drift.** The eligible count grows from tens in 2021 to a hundred
  later. Early folds carry fewer names and more concentration; the
  concentration gate catches the failure mode.
- **Delisting mechanics.** Forced closes at the last available close
  approximate settlement; Binance settles delisted perpetuals at a settlement
  price that may differ. Recorded as an assumption; forced-close count is
  reported.
- **Weak ceiling.** Plain momentum is the weaker cousin of CTREND-style
  aggregates. A fail here is not evidence that no panel signal exists; it is
  evidence that the two-parameter version does not survive these costs.

## 15. Non-Goals

- No intraday cadence, no 15-minute panel, no L2 data (storage policy).
- No spot legs, no spot–perpetual basis trade (a separate mechanical family).
- No machine-learned ranking, no CTREND replication, no feature aggregation.
- No CTM, GRU, or residual mixture on the panel in this family.
- No shadow, paper, or live authority; no holdout opening.
- No change to the 15-minute research line, its captures, or its registry.

## 16. Sources

- Moskowitz, Ooi, Pedersen (2012), *Time Series Momentum*, JFE 104(2), 228–250. https://www.sciencedirect.com/science/article/pii/S0304405X11002613
- Liu, Tsyvinski, Wu (2022), *Common Risk Factors in Cryptocurrency*, JF 77(2), 1133–1177. https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.13119
- Fieberg, Liedtke, Poddig, Walker, Zaremba (2025), *A Trend Factor for the Cross Section of Cryptocurrency Returns*, JFQA 60(7), 3116–3153. doi 10.1017/S0022109024000747
- Vilnius University, *Momentum Trading in Cryptocurrencies: A Comparative Study of Time-Series and Cross-Sectional Strategies* (2025). https://www.journals.vu.lt/BATP/en/article/download/44540/42590/138419
- Bailey, López de Prado (2014), *The Deflated Sharpe Ratio*, Journal of Portfolio Management 40(5), 94–107.
- Binance public data: https://data.binance.vision (bucket `s3-ap-northeast-1.amazonaws.com/data.binance.vision`), verified 2026-09-08.
- Repository: `PHASE_0_EVALUATION_PROTOCOL.md`, `PHASE_0_DATASET_SPEC.md`, `ADR_002_ENGINE_AND_STORAGE.md`, `P1_15_DECISION_2026-08-25.md`, `docs/superpowers/specs/2026-09-08-ctm-moe-challenger-design.md`.
