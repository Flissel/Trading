# CTM Mixture-of-Experts Challenger Design

Status: Draft for review on 2026-09-08. `research_only`. Not approved for
implementation. Implements charter `E06` and backlog `P1.16` under spec
`2026-08-25-paper-trading-readiness-design.md` section 6.

## 1. Purpose

Design the smallest faithful Continuous Thought Machine (CTM) challenger that
can be evaluated under the frozen Phase-A gates, and the regime-gated
mixture-of-experts (MoE) that the CTM is meant to serve, so that a real-time
15-minute directional forecast for BTC perpetuals can be produced in shadow
mode without an order route.

The design answers three questions:

1. What a CTM actually computes, and what is and is not known about it.
2. What the 2026 evidence says about neural forecasting of crypto returns and
   about mixtures of experts on financial series.
3. How a CTM-based mixture fits this repository's contracts, gates, data
   budget, and hardware, and in what order it must be built and evaluated.

A technical component may be completed after an economic gate fails, but it
remains `research_only` and cannot acquire promotion authority. The charter
orders CTM after simple baselines pass on frozen OOS data. P1.15 rejected all
simple baselines on 2026-08-25. This design therefore ships as research-only
tooling whose economic attempt is registered only when section 10's
sequencing conditions hold.

## 2. What the CTM Is

Source: Darlow, Regan, Risi, Seely, Jones, *Continuous Thought Machines*,
arXiv 2505.05522 (v1 2025-05-08, v4 2025-10-03), NeurIPS 2025 spotlight.
Reference code: `github.com/SakanaAI/continuous-thought-machines`, Apache 2.0.

### 2.1 Mechanism

The CTM is a recurrent model over an **internal time axis** of `T` ticks that
is decoupled from the data. Per forward pass:

1. **Attention over input tokens.** A query is built from the synchronization
   of `n_synch_action` neurons and attends over the key/value projection of the
   input features (image patches in the paper; here, lagged bar features).
2. **Synapse update.** The attention output is concatenated with the current
   post-activation state and passed through a synapse model (MLP or small
   U-Net) to produce a new pre-activation, which is appended to a rolling
   history of length `memory_length` (M).
3. **Neuron-level models (NLM).** Every neuron has private weights (linear or
   two-layer MLP) that map its own M-step pre-activation history to a scalar
   post-activation. This is the first defining feature: neurons have
   individual temporal dynamics.
4. **Synchronization as representation.** The latent read-out is not the
   activation vector but the decayed inner product of activation histories
   between sampled neuron pairs, `S_ij = Σ_τ exp(-r_ij (t-τ)) z_i(τ) z_j(τ) /
   sqrt(Σ_τ exp(-r_ij (t-τ)))`, with learnable decay `r_ij`, maintained as a
   recurrence so only the current state is stored. This is the second defining
   feature. `n_synch_out` pairs feed the output head; `n_synch_action` pairs
   feed the attention query.
5. **Per-tick output and certainty.** Every tick produces a prediction
   `(B, out_dims, T)` and a certainty `(B, 2, T)` where certainty is
   `1 - normalized entropy` of the softmax.

**Loss.** For each sample the loss is averaged over two ticks only: the tick
with minimum loss and the tick with maximum certainty,
`L = (L[argmin L] + L[argmax C]) / 2`. This couples certainty to correctness
and yields **adaptive compute**: at inference the loop may stop at the first
tick whose certainty exceeds a threshold. The paper reports that with a 0.8
threshold most ImageNet instances halt before 10 of 50 ticks.

### 2.2 Verified constructor surface

`iterations`, `d_model`, `d_input`, `heads`, `n_synch_out`, `n_synch_action`,
`synapse_depth`, `memory_length`, `deep_nlms`, `memory_hidden_dims`,
`do_layernorm_nlm`, `backbone_type`, `positional_embedding_type`, `out_dims`,
`prediction_reshaper`, `dropout`, `neuron_select_type`
(`first-last | random | random-pairing`), `n_random_pairing_self`.

With `random-pairing` the synchronization vector has `n_synch` entries; with
the other modes it has `n_synch (n_synch + 1) / 2`.

### 2.3 What is known and not known

Known:

- Demonstrated on ImageNet-1k (72.47% top-1 with a ResNet-152 backbone,
  T = 50), CIFAR, 2D mazes, sorting, parity, QAMNIST, and PPO on CartPole,
  Acrobot, MiniGrid. The value shown is interpretability, adaptive compute,
  and calibration behaviour, not state-of-the-art accuracy.
- Authors' stated limitations: the internal sequence extends training time;
  NLMs increase parameter count; no SOTA hyperparameter search was run.
- The reference repository is frozen as "an accurate reflection" of the paper
  (maintainer note 2025-12-29). There is no CTM v2 and no Sakana follow-up.

Not known:

- No published application of the CTM to time series, sequences of market
  data, or any financial target, by Sakana or by third parties, as of
  2026-09-08.
- No published regression variant. The certainty definition is
  classification-only. A regression head would need its own uncertainty
  mechanism; this design keeps a binary classification target so the native
  certainty is used unchanged.
- Faithful CTM on a temporally ordered token set is untested. The input
  tokens in every published task are spatial (patches, maze cells) or
  positional (list elements). Treating lag positions as tokens is the same
  mechanism, but the claim that the CTM's attention "walks" the input in a
  useful order is only established for spatial tasks.

### 2.4 What "Sakana mixture of experts" is not

Sakana's multi-model work is AB-MCTS / TreeQuest: inference-time tree search
that lets several LLMs propose and refine answers to a verifiable problem,
selected by a scorer. It is Apache 2.0 and reported gains on ARC-AGI-2. It is
not a mixture of experts in the routing sense and does not apply here: a
streaming forecast has no scorer until the future is observed, so there is
nothing to search over at decision time. The mixture in this design is the
classical gated mixture, informed by the 2026 financial-MoE literature in
section 3.3.

## 3. Evidence Review

### 3.1 CTM outside its paper

Only one third-party integration was found (a humanoid-robot control
architecture pairing CTM with MCP, arXiv 2505.19339). No forecasting use. The
CTM must therefore be treated as an unproven architecture for this task, and
charter section 8.4 applies: no model may be included solely because it is
architecturally novel.

### 3.2 Neural forecasting of crypto returns, 2026

- CryptoGAT (arXiv 2606.27670, 2026-06) finds that LSTM, GRU, and Transformer
  forecasters that work on equities "struggle" on cryptocurrency series and
  reframes the problem as cross-asset graph prediction. No transaction costs.
- *Algorithmic Stability in Turbulent Markets* (MDPI Mathematics 14(6):989,
  2026-03) reports that shallow SVM forecasters are more stable than LSTM and
  GRU on BTC and XRP under regime shifts, while deep models do better in calm
  regimes; economic value depends on regime, not on model depth.
- Cross-sectional predictability with boosted trees (Liu et al. 2023; Cakici
  et al. 2024) is concentrated in small-cap coins and in past-alpha,
  illiquidity, and momentum features. Single-asset BTC at 15 minutes is the
  hard case, not the easy one.
- Foundation-model forecasters (FinCast, arXiv 2508.19609) report point
  accuracy, not net-of-cost trading utility.

Implication: the literature does not license an expectation that a sequence
model beats shallow models on this repository's target. It does support two
design choices: keep shallow experts, and make regime awareness explicit.

### 3.3 Mixture of experts on financial series, 2025–2026

- **Regime-Gated Residual MoE** (arXiv 2608.12251, 2026-08; 1,027 US equities,
  1,552 JP replication, 30 seeds): regime variables enter **only the gate**,
  never the expert inputs; a **frozen base model** predicts first and
  **zero-initialized experts learn residual corrections**; soft gating with a
  load-balancing regularizer. Result: zero training collapses versus 24 of 30
  for a standard MoE; appending regime variables to expert inputs *degraded*
  accuracy; soft routing beat every hard-routing variant; gains concentrated
  in high-volatility periods.
- **RAVEN** (arXiv 2606.24062, 2026-06): scale-specialized experts over nested
  context windows chosen by importance scoring, plus a global branch. Adaptive
  context beats fixed look-back on HS300 and S&P 500 correlation targets.
- **Hybrid Recurrent Expert Gating** (Procedia CS, 2026): softmax gate over
  GRU/LSTM/RNN experts; gate weights are interpretable as regime attributions.

Implication: the recipe is settled. Frozen base, residual zero-init experts,
regime information via the gate only, soft routing, load-balance loss.

### 3.4 This repository's own result

P1.15 (2026-08-25, 580 days BTC 15m, h4 and h16, three folds): every
candidate rejected. The h16 boosted-stump challenger was base-positive in
3/3 folds (+1.507) but adverse-cost negative (-0.612) with worst BH q = 0.274.
Momentum and mean-reversion rules were rejected on both horizons. The
adverse-cost scenario, not forecast skill, was the binding constraint.

Implication: the most valuable thing a new component can add is **better
abstention under adverse costs**, which is evaluation-protocol criterion 10(c),
and **regime state**, criterion 10(d). Raw forecast-skill improvement,
criterion 10(a), is the least likely outcome.

## 4. Problem Definition

### 4.1 Target and "real time"

- Instrument: BTC-USDT perpetual, OKX primary, Binance USD-M reference.
- Decision cadence: every confirmed 15-minute bar close. "Real time" means the
  forecast is available before the next bar opens, using only bars whose close
  time was observable at receive time. Sub-bar or tick-level forecasting is out
  of scope: no L2 stream is captured and storage policy excludes it.
- Target: sign of the forward return over `h` bars, `h ∈ {4, 16}`, exactly the
  existing horizon-bound labels. Output: probability of positive return,
  certainty, and a long/short/hold decision after abstention.
- Latency budget: 5 seconds wall-clock from candle confirmation to a written
  forecast record. If the record is not written before the next bar open the
  runtime state is `no_new_risk`.

### 4.2 Data budget

Per fold with the frozen 365/90/30-day calendar and 96 bars per day:

| Slice | Bars | Independent samples at h4 | at h16 |
| --- | ---: | ---: | ---: |
| Train | 35,040 | ≈ 8,760 | ≈ 2,190 |
| Validation | 8,640 | ≈ 2,160 | ≈ 540 |
| Test | 2,880 | ≈ 720 | ≈ 180 |

Boundary samples whose labels cross a partition are removed; the embargo is
17 bars. The validation slice is split chronologically as in the existing
logistic challenger: an earlier calibration half and a later selection half.
The h16 numbers bound model capacity: anything with more than a few hundred
thousand parameters is unjustified, and early stopping is mandatory.

### 4.3 Hardware

Verified on this machine: NVIDIA GeForce RTX 3060, 12,288 MiB, driver 591.86;
Python 3.12.0; no PyTorch installed. The spec's 12 GB VRAM preflight is the
binding limit. The anticipated 480 GB environment is not used.

## 5. Design Options

### Option A — K faithful CTM experts under a regime gate

One CTM per input representation (slope view, spectral view, rate-of-change
view, summary view), a gate over them, residual over the frozen Phase-A base.

- Pro: literal reading of the goal; each expert has native certainty.
- Con: K full CTM trainings per fold, seed, and horizon; four temporally
  extended recurrent models on ≈ 2,000 independent h16 samples; the multiple
  testing burden of four learned experts plus a gate. The 2026 shallow-beats-
  deep evidence argues directly against deep experts.

### Option B — one CTM, many views as attention tokens

A single CTM whose input token set is the union of all views across lags. The
CTM's own synchronization-driven attention is the router.

- Pro: cheapest faithful CTM; adaptive compute; attention traces show which
  view and lag the model reads, matching the paper's interpretability claim.
- Con: no explicit expert decomposition, so charter 8.4's requirement to
  preserve component forecasts and weights is only met by the base-plus-CTM
  pair; incremental-ensemble-value (10b) is harder to attribute.

### Option C — shallow experts, CTM as regime encoder and gate (recommended)

Experts are cheap, mostly existing models, one per representation. A faithful
CTM consumes the causal bar window, encodes evolving market state, and emits
(i) its own residual forecast and (ii) the gate distribution over experts. The
combination is residual over a frozen Phase-A base.

- Pro: this is the charter's stated CTM role ("encode evolving market state",
  section 8.2); it uses the settled MoE recipe of section 3.3; experts are
  shallow, which the 2026 evidence favours in turbulent regimes; CTM certainty
  becomes the abstention signal, targeting criterion 10(c); gate weights are a
  regime-state representation, targeting criterion 10(d); one CTM training per
  cell; all component forecasts and weights are logged for charter 8.4.
- Con: the CTM does not "predict price" alone; its value must be shown against
  a GRU encoder doing the same job, which is the comparison the protocol
  demands anyway.

**Recommendation: Option C, with Option B as the mandatory CTM-alone ablation
and Option A deferred.** Option A is only revisited if C passes criterion 10(b)
and the user releases a new hypothesis family.

## 6. Recommended Architecture

### 6.1 Components

```text
confirmed bars ──► enriched samples ──► causal window (W bars × F features)
                        │                          │
                        ├──► view features ──► experts E1..E4 (shallow, residual)
                        │                          │
                        └──► frozen base f0 (Phase-A logistic)
                                                   │
                        CTM encoder (faithful) ◄── window tokens
                          ├── residual head r_ctm
                          ├── gate head π over {E1..E4, r_ctm}
                          └── certainty c per tick, early stop at τ
                                                   │
                logit = f0 + Σ_k π_k · r_k  ──► p = σ(logit)
                                                   │
                abstention: c ≥ τ  AND  adverse-EV(envelope) > round-trip cost
                                                   │
                ForecastRecord + audit chain (shadow only, no order route)
```

### 6.2 Experts, mapped from the handwritten design

The two handwritten sheets define four representations and a meta model. They
map as follows. Every feature is trailing-window only and carries an
`available_time`; no centered window, no unconfirmed extremum, no future bar.

| Sheet | Representation | Expert | Features (all causal) | Learner |
| --- | --- | --- | --- | --- |
| 1 | `m × t`, tangents, `∫ m` | E1 slope | OLS and Theil-Sen slope over 8/32/96 bars; slope-of-slope; scale agreement (sign concordance across windows) | ridge, zero-init residual |
| 2 | `FFT²`, "wann, wie oft, wie sicher" | E2 spectral | band power and autocorrelation at predeclared periods 32 (8h funding), 96 (24h), 672 (7d) bars; phase within the 8h funding cycle; Ljung-Box statistic over 96 bars | logistic, zero-init residual |
| 2 | `Σ ṁ`, Veränderungsrate | E3 rate | existing returns at multiple lags and realized volatility | existing boosted stumps, residualized |
| 2 | Zusammenfassung | E4 summary | existing Phase-A matrix: range ratio, volume z-score, basis mean and change | existing ridge, residualized |
| 2 | "prediction regression rewards (richtig/falsch)" | gate | not a fifth predictor; the CTM gate head, trained through the combined loss | CTM |

Local extrema and envelopes from sheet 1 are admitted only as **confirmed**
extrema with a fixed confirmation delay `k = 8` bars and an `available_time`
of `t + k`. Envelope width, not envelope direction, is the admitted feature.

The 8h period is mechanical (funding settles 00:00, 08:00, 16:00 UTC on OKX
and Binance) and is the only spectral hypothesis with a stated mechanism. The
24h and 7d periods are session and weekend liquidity hypotheses. No other
frequency is searched.

### 6.3 CTM encoder configuration (primary attempt)

Faithful to the pinned reference: internal ticks, per-neuron NLMs, and
synchronization read-out are all retained; only the input embedding and the
output heads are task-specific.

| Argument | Value | Reason |
| --- | ---: | --- |
| `backbone_type` | `none` | tokens are lag positions, not pixels |
| token embedding | `Linear(F → d_input)` | one token per lag |
| `positional_embedding_type` | learned, 96 positions | lag index is the position |
| window `W` | 96 bars (24h) | matches block length and the 24h hypothesis |
| `d_input` | 64 | |
| `d_model` | 256 | capacity bound from 4.2 |
| `heads` | 4 | |
| `iterations` T | 20 | adaptive stop below this |
| `memory_length` M | 16 | |
| `deep_nlms` | true, `memory_hidden_dims` 4 | paper default shape |
| `synapse_depth` | 1 | MLP synapse; U-Net unjustified at this size |
| `n_synch_out` | 64, `random-pairing` | |
| `n_synch_action` | 32, `random-pairing` | |
| `dropout` | 0.1 | |
| output heads | residual logit (1) and gate logits (5) from the output synchronization | |

Estimated parameter count is below one million (NLMs ≈ 19k, synapse and
projections ≈ 300–400k). The exact count is recorded per attempt, as protocol
section 10 requires.

### 6.4 Gate and combination

- Base `f0`: the fold's Phase-A calibrated logistic, **frozen** after its own
  training. The gate has five entries, E1–E4 and `r_ctm`, and the weighted
  residual sum is added to `f0`. "Trust base only" is the state in which all
  residuals are near zero; it needs no separate gate entry.
- Experts learn residual logits with zero-initialized output layers, so the
  mixture starts identical to the base.
- Gate `π = softmax(g)` where `g` is the CTM gate head. Soft routing only.
  Regime information reaches the experts **only** through `π`.
- Load-balance regularizer: squared coefficient of variation of the batch-mean
  gate weights, weight `λ_lb = 0.01`, following the RG-ResMoE recipe.
- Loss: binary cross-entropy of `p` against the h-return sign, applied with the
  CTM dual-tick rule (minimum-loss tick and maximum-certainty tick), plus
  `λ_lb` times the load-balance term.

### 6.5 Abstention

Two independent conditions; hold if either fails:

1. **CTM certainty** at the stopping tick, `c = 1 - H(p)/log 2`, must reach
   the threshold `τ` selected on the validation selection half under the
   existing minimum of 200 selection trades. Validation may select `τ`; it may
   not fit parameters.
2. **Adverse expected value** via the existing `decide_with_abstention` with a
   `ForecastEnvelope` from `fit_residual_interval` fitted on the validation
   calibration half. The envelope is on the return scale; `p` is mapped to an
   expected return through the calibration-half regression of realized return
   on `logit(p)`.

Every hold carries a reason code. The certainty trace over ticks is stored in
the forecast record for audit.

### 6.6 Training protocol and leakage controls

Per horizon, fold, and seed:

1. Fit the Phase-A base on train; freeze.
2. Fit view features (medians, scales) on train only via the existing
   `fit_feature_matrix` contract, extended with the E1 and E2 fields.
3. Jointly train CTM encoder, gate, and expert residual heads on train with
   AdamW, batch 256, learning rate 1e-3, at most 30 epochs, early stopping on
   the validation **calibration** half BCE, patience 5.
4. Fit the residual interval calibrator on the calibration half.
5. Select `τ` on the validation **selection** half.
6. Evaluate once on test with base and adverse costs, block bootstrap
   (block 96, 2,000 repetitions), Benjamini-Hochberg over the predeclared
   comparison set.
7. The final holdout is never read. Read count remains zero.

Stacking leakage is addressed structurally rather than by out-of-fold
stacking: experts are shallow and residual, the gate cannot rescale expert
outputs beyond the softmax simplex, the base is frozen, and early stopping
uses a slice the experts never trained on. The stricter alternative, inner
blocked cross-validation on the train window to produce out-of-fold expert
residuals for gate training, is admitted as the single allowed revision if the
primary attempt shows gate weights concentrating on one expert in-sample
(maximum batch-mean weight above 0.6).

Determinism: fixed seeds per cell, `torch.use_deterministic_algorithms(True)`,
CPU inference for shadow so that a forecast is reproducible bit-for-bit from
the checkpoint and window; any non-deterministic kernel is a registered
failure, not a silent fallback.

### 6.7 Real-time inference path (shadow)

1. Trigger on confirmed candle close on both venues. Binance bars count only
   when their close time was observable at receive time, as today.
2. Build the enriched sample and view features through the existing Decimal
   path; any quality flag or gap in the trailing 96 bars fails closed to
   `no_new_risk`.
3. Assemble the window tensor, run E1–E4 (microseconds), run the CTM with
   early stop at `c ≥ τ` and hard cap T = 20.
4. Combine, abstain, and write a `ForecastRecord` with `model_id`,
   `model_version`, `dataset_hash`, `feature_set_hash`, `calibration_version`,
   `as_of_ns`, `horizon_ns`, gate weights, component residuals, certainty
   trace, ticks used, and wall-clock latency.
5. Append to the audit chain. There is no order route, credential, or live
   client factory in this package; it plugs into the P1.24 shadow runner.

Expected latency on CPU for a sub-million-parameter model over 96 tokens and
at most 20 ticks is tens of milliseconds; p50 and p95 are measured and
recorded per protocol section 10.

### 6.8 Interfaces

New package `trading_bot.neural`, importable only when the optional dependency
group `neural` (PyTorch, CUDA 12.x) is installed. The core package stays
torch-free. View features live in the core because they are Decimal and
torch-free.

```python
class CtmMixtureSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal["1.0.0"]
    reference_commit: str          # 40-hex git SHA of the pinned CTM source
    reference_source_hash: str     # sha256 of the vendored ctm.py + modules.py
    feature_schema_hash: str
    window_bars: Literal[96]
    d_model: Literal[256]
    d_input: Literal[64]
    heads: Literal[4]
    iterations: Literal[20]
    memory_length: Literal[16]
    n_synch_out: Literal[64]
    n_synch_action: Literal[32]
    experts: tuple[Literal["slope", "spectral", "rate", "summary"], ...]
    load_balance_weight: Decimal
    certainty_threshold_grid: tuple[Decimal, ...]
    seeds: tuple[int, ...]
    max_epochs: Literal[30]
    vram_limit_bytes: Literal[12884901888]
```

```python
@dataclass(frozen=True, slots=True)
class MixtureForecast:
    sample_id: str
    as_of_ns: int
    horizon_bars: int
    base_logit: Decimal
    expert_residuals: tuple[Decimal, ...]
    gate_weights: tuple[Decimal, ...]
    ctm_residual: Decimal
    probability: Decimal
    certainty: Decimal
    ticks_used: int
    certainty_trace: tuple[Decimal, ...]
    decision: AbstentionDecision
```

```python
@dataclass(frozen=True, slots=True)
class ComputeEvidence:
    parameter_count: int
    peak_vram_bytes: int
    training_seconds: Decimal
    inference_p50_ms: Decimal
    inference_p95_ms: Decimal
    gpu_seconds: Decimal           # energy/runtime proxy required by protocol §10
```

Functions, mirroring the Phase-A fold runner:

```python
def run_ctm_mixture_fold(config: CtmFoldConfig) -> CtmFoldArtifact: ...
def verify_ctm_mixture_fold(path: Path) -> bool: ...
def run_sequence_baseline_fold(config: SequenceFoldConfig) -> CtmFoldArtifact: ...
```

`CtmFoldConfig` extends `PhaseAFoldConfig` with `mixture_spec`, `encoder_kind
∈ {ctm, gru}`, `gate_kind ∈ {learned, uniform, none}`, and
`checkpoint_hash`. The fold artifact binds capture, dataset, split, feature
schema, hypothesis attempt, spec hash, checkpoint hash, reference source hash,
and `ComputeEvidence`.

Module map:

- `src/trading_bot/view_features.py` — E1 and E2 features, Decimal, causal.
- `src/trading_bot/neural/__init__.py` — import guard with actionable error.
- `src/trading_bot/neural/ctm_reference.py` — vendored `ctm.py` and
  `modules.py` from the pinned commit, Apache 2.0 notice, source-hash check.
- `src/trading_bot/neural/windows.py` — causal window tensors from enriched
  samples.
- `src/trading_bot/neural/mixture.py` — base, experts, encoder, gate, loss.
- `src/trading_bot/neural/sequence_baseline.py` — GRU encoder with the same
  heads (spec section 6.1 comparator).
- `src/trading_bot/neural/fold_run.py` — training, selection, evaluation,
  artifact, verifier.
- `src/trading_bot/neural/compute_evidence.py` — parameter count, VRAM,
  timing.
- `tests/` — one test module per source module; synthetic fixtures; a preflight
  test that asserts the spec fits the VRAM limit on a dummy batch.

## 7. Ablations and Decision Rule

The comparison set is predeclared and fixed before the first training run:

| ID | Encoder | Gate | Experts | Purpose |
| --- | --- | --- | --- | --- |
| `base` | — | — | — | Phase-A logistic (frozen `f0`) |
| `best_expert` | — | — | best single of E1–E4 on validation | are experts worth anything alone |
| `uniform_moe` | — | uniform 1/4 | E1–E4 | does routing matter |
| `gru_moe` | GRU | learned | E1–E4 + r_gru | spec 6.1 comparator with identical preprocessing |
| `ctm_alone` | CTM | — | r_ctm only | Option B; P1.16 spike |
| `ctm_moe` | CTM | learned | E1–E4 + r_ctm | the candidate |

`ctm_moe` is **eligible** only if all of the following hold on locked test
folds:

1. Existing Phase-A gates: positive aggregate base and adverse net return,
   at least two of three base folds positive, at least 200 selection trades
   per fold, worst BH q ≤ 0.10 over the comparison set, dominance over
   no-trade.
2. At least one protocol 10 criterion, each tested as a paired block-bootstrap
   difference against `gru_moe`:
   - (a) net return higher;
   - (b) `ctm_moe − uniform_moe` and `ctm_moe − best_expert` both positive;
   - (c) Brier score and interval coverage better with net utility not lower;
   - (d) gate weights from CTM yield higher net utility than `gru_moe` and
     `uniform_moe` gate weights applied to the same experts.
3. The gain does not vanish against `gru_moe`. If `gru_moe` matches `ctm_moe`
   within the bootstrap interval on every criterion, the CTM is rejected per
   protocol 10's final sentence, whatever its absolute performance.

Any other outcome is `no_edge_found` for this family and is a successful,
registered research result. `INSUFFICIENT_EVIDENCE` is reported when fewer
than three folds complete or any fold fails preflight.

## 8. Compute Budget and Preflight

- Preflight: a dummy batch of 256 windows through the full mixture must fit
  under 12 GB with headroom of 25%; failure is a registered attempt outcome.
- Training: 2 horizons × 3 folds × 3 seeds × 3 encoder configurations
  (`gru_moe`, `ctm_alone`, `ctm_moe`) = 54 encoder runs, plus the shallow
  `base`, `best_expert`, and `uniform_moe` fits, which take seconds. Estimated
  20–60 minutes per CTM run on the RTX 3060 at the section 6.3 size; under
  48 GPU-hours total. If a run exceeds 2 hours it is a timeout failure.
- Inference: CPU, measured p50 and p95 recorded.
- Storage: checkpoints under `artifacts/neural/`, gitignored, hashed into the
  fold report.
- Dependency: `torch` pinned to a CUDA 12.x wheel in a `neural` dependency
  group; roughly 2.5 GB in the virtual environment; `uv sync --group neural`.

## 9. Hypothesis Registration and Budget

One family, registered in the hypothesis registry before any training:

- `E06 CTM regime-gated residual mixture v1`, horizons h4 and h16.
- Primary attempt: section 6.3 configuration, seeds frozen in the spec.
- Single permitted revision, chosen before seeing test results, from:
  (i) inner blocked CV for out-of-fold gate training (section 6.6 trigger), or
  (ii) window 96 → 192 bars with `d_model` unchanged.
- No other hyperparameter changes. Repeated search against the same three
  test folds is not a revisit path. A revision is a new `CtmMixtureSpec`
  version (`1.1.0`) with its own hash; the `Literal` bounds in section 6.8
  are widened only by that version bump, never by editing the primary spec.

## 10. Sequencing and Stop Conditions

1. **C.0 Compact sequence baseline.** Build and register `gru_moe` and the
   view features. This is required by spec section 6.1 and is the comparator
   without which every CTM result is automatically rejected.
2. **C.1 CTM spike (P1.16).** Build the vendored reference, the encoder, and
   `ctm_alone`. Record `ComputeEvidence`. Stop condition: preflight or timeout
   failure, or no protocol 10 criterion met against `base` and the GRU
   encoder alone. Failure stops CTM expansion; `ctm_moe` code may exist but
   its economic attempt is not registered.
3. **C.2 CTM mixture.** Register and evaluate `ctm_moe` against the full
   section 7 set. Publish an immutable decision artifact bound to all hashes.
4. **Shadow.** Only an `eligible` `ctm_moe` may be wired into the P1.24 shadow
   runner, and only as `research_only` until the eight-week, 200-episode paper
   window exists. Nothing in this design authorizes paper or live trading.

The charter's order — simple baselines before CTM — is respected by the
attempt registration, not by the build: C.0 through C.2 can be implemented
while Phase A's revision runs, but no CTM economic claim is registered while
Phase A has no `eligible_candidate` unless the user records an explicit
research-only release in the decision file.

## 11. Risks and Honest Priors

- **Prior on criterion 10(a)**, forecast skill beating the GRU mixture after
  adverse costs on single-asset BTC 15m: low, on the order of one in ten,
  given P1.15 and the 2026 literature.
- **Prior on 10(c) or 10(d)**, better abstention or usable regime state: more
  plausible, on the order of one in three. The design is optimized for these.
- **Capacity versus samples.** ≈ 2,000 independent h16 training samples per
  fold against a sub-million-parameter recurrent model. Early stopping,
  dropout, residual structure, and frozen base are the mitigations; expect
  fold-to-fold instability and treat two-of-three positive folds as the floor
  it already is.
- **Gate collapse.** Mitigated by the RG-ResMoE recipe; monitored via the
  batch-mean maximum gate weight; section 6.6 revision trigger.
- **Non-causal features.** The handwritten design's extrema and envelopes are
  the largest leakage risk; only confirmed extrema with `available_time =
  t + k` are admitted, and the existing point-in-time feature tests extend to
  `view_features.py`.
- **Non-determinism.** GPU kernels may be non-deterministic; the design
  requires deterministic algorithms and CPU inference, and registers
  violations.
- **Frozen upstream.** The reference repository accepts no changes; vendoring
  with a source hash is the only stable pinning.
- **Reviewer temptation.** The h16 boosted stump is "almost there" and a
  mixture will look better in base costs. The adverse-cost gate and the BH
  correction are the protections; they are not relaxed for this family.

## 12. Non-Goals

- Real-capital execution, live endpoints, or automatic promotion.
- Sub-bar, tick-level, or order-book-driven prediction.
- Reinforcement-learning or bandit sizing, deferred by the backlog until
  deterministic baselines pass.
- Multiple CTM experts (Option A), multi-asset graph models, or the 480 GB
  environment before topology verification.
- AB-MCTS / TreeQuest.
- Any use of the final holdout.

## 13. Sources

- Darlow et al., Continuous Thought Machines, arXiv 2505.05522 v4, 2025-10-03.
- SakanaAI/continuous-thought-machines, GitHub, Apache 2.0, maintainer note
  2025-12-29.
- Sakana AI, AB-MCTS / TreeQuest, 2025.
- CryptoGAT, arXiv 2606.27670, 2026-06.
- Algorithmic Stability in Turbulent Markets, MDPI Mathematics 14(6):989,
  2026-03.
- Regime-Gated Residual Mixture-of-Experts for Cross-Sectional Volatility
  Forecasting, arXiv 2608.12251, 2026-08.
- RAVEN, arXiv 2606.24062, 2026-06.
- Hybrid Recurrent Expert Gating, Procedia Computer Science, 2026.
- Designing funding rates for perpetual futures, arXiv 2506.08573, 2025.
- OKX and Binance funding documentation: 8-hour settlement at 00:00, 08:00,
  16:00 UTC.
- This repository: `PHASE_0_RESEARCH_CHARTER.md` sections 8.1–8.4,
  `PHASE_0_EVALUATION_PROTOCOL.md` section 10, `PHASE_1_BACKLOG.md` P1.16,
  `P1_15_DECISION_2026-08-25.md`,
  `docs/superpowers/specs/2026-08-25-paper-trading-readiness-design.md`
  sections 3 and 6.
