"""Trend indicators, the twelve-vote score, and the trend family's weight vectors.

Spec 3 of `docs/superpowers/specs/2026-09-15-trend-aggregate-family-design.md`
freezes twelve indicators over a contract's daily closes, each a vote in
{-1, 0, +1}, and defines the trend score as their mean. A contract with fewer
than twelve computable votes has no score and is not rankable.

The weight construction is P1.27's, with the trailing-return sign replaced by
the score's thresholded sign: `panel_signals`' volatility, normalisation,
water-filling and quintile helpers are imported rather than copied, so the two
families provably share one construction (`tests/test_trend_signals.py` pins
the controls against `panel_signals.build_weight_vectors`' own). Importing a
sibling module's private helpers is the same arrangement `carry_accounting`
has with `panel_accounting`; `panel_signals.py` itself stays untouched, so
P1.27 remains reproducible.
"""

from decimal import Decimal
from random import Random

from trading_bot.panel_config import PanelWeightRules
from trading_bot.panel_signals import (
    WeightVector,
    _annualised_volatility,
    _cross_sectional_weights,
    _normalise,
    _water_fill,
)
from trading_bot.panel_universe import ContractHistory, UniverseSnapshot
from trading_bot.trend_config import INDICATOR_NAMES, TrendFamilySpec

_SMA_WINDOWS = (20, 50, 100)
_SMA_CROSS_WINDOWS = ((20, 50), (50, 100))
_BREAKOUT_WINDOWS = (20, 50, 100)
_ROC_WINDOWS = (20, 60, 120)
_MACD_FAST_WINDOW = 12
_MACD_SLOW_WINDOW = 26
_DECLARED_INDICATORS = frozenset(INDICATOR_NAMES)


def closes_before(history: ContractHistory, decision_close_ns: int) -> tuple[Decimal, ...]:
    """Closes at or before the decision, ascending by close time.

    `ContractHistory.close_times` is built sorted (`panel_universe.
    build_contract_histories`), which every point-in-time reader in the panel
    line already relies on, so this filters rather than re-sorts.
    """
    return tuple(
        history.closes[close_time]
        for close_time in history.close_times
        if close_time <= decision_close_ns
    )


def sma(closes: tuple[Decimal, ...], window: int) -> Decimal:
    """The mean of the last `window` closes."""
    if window <= 0:
        raise ValueError("a moving-average window must be positive")
    if len(closes) < window:
        raise ValueError(f"an SMA over {window} closes needs {window} closes, got {len(closes)}")
    return sum(closes[-window:], Decimal(0)) / Decimal(window)


def ema(closes: tuple[Decimal, ...], window: int) -> Decimal:
    """The `alpha = 2 / (window + 1)` recursion over the whole series.

    Seeded with the first close of the series, as spec 3 requires, and run
    forward over every close: unlike an SMA the EMA has no minimum window of
    its own, so the caller decides when the indicator is computable (26 closes
    for `macd_12_26`).
    """
    if window <= 0:
        raise ValueError("an EMA window must be positive")
    if not closes:
        raise ValueError("an EMA needs at least one close")
    alpha = Decimal(2) / Decimal(window + 1)
    value = closes[0]
    for close in closes[1:]:
        value = alpha * close + (Decimal(1) - alpha) * value
    return value


def indicator_votes(closes: tuple[Decimal, ...]) -> dict[str, int]:
    """Every computable indicator's vote in {-1, 0, +1}, in declaration order.

    An indicator whose window exceeds the series is absent from the result
    rather than zero: absence means "not computable" and drives the
    twelve-vote requirement, while zero is a genuine abstention (an exact tie).
    """
    if not closes:
        return {}
    votes: dict[str, int] = {}
    latest = closes[-1]
    for window in _SMA_WINDOWS:
        if len(closes) >= window:
            votes[f"ma_{window}"] = _sign(latest - sma(closes, window))
    for fast, slow in _SMA_CROSS_WINDOWS:
        if len(closes) >= slow:
            votes[f"ma_cross_{fast}_{slow}"] = _sign(sma(closes, fast) - sma(closes, slow))
    for window in _BREAKOUT_WINDOWS:
        if len(closes) >= window:
            recent = closes[-window:]
            at_high = latest == max(recent)
            at_low = latest == min(recent)
            # A window that is constant puts the close at both its high and its
            # low; spec 3's breakout is directional, so the tie is a 0.
            votes[f"breakout_{window}"] = 0 if at_high == at_low else (1 if at_high else -1)
    for window in _ROC_WINDOWS:
        if len(closes) > window:
            # `sign(P_t / P_{t-n} - 1)` equals `sign(P_t - P_{t-n})` for the
            # positive closes a price series carries, and the difference stays
            # exact where the quotient would be rounded to the context's 28
            # digits -- which matters because spec 3 resolves an exact zero to
            # an abstention, and a quotient rounding to exactly 1 would forge
            # one.
            votes[f"roc_{window}"] = _sign(latest - closes[-(window + 1)])
    if len(closes) >= _MACD_SLOW_WINDOW:
        votes["macd_12_26"] = _sign(ema(closes, _MACD_FAST_WINDOW) - ema(closes, _MACD_SLOW_WINDOW))
    # The window constants above and `INDICATOR_NAMES` are two spellings of the
    # same frozen list; if they ever drift apart, say so rather than quietly
    # returning a vote the score would then never count.
    unknown = votes.keys() - _DECLARED_INDICATORS
    if unknown:
        raise ValueError(f"undeclared indicators: {sorted(unknown)}")
    return {name: votes[name] for name in INDICATOR_NAMES if name in votes}


def trend_score(closes: tuple[Decimal, ...]) -> Decimal | None:
    """The mean of the twelve votes, or None when any indicator is absent."""
    votes = indicator_votes(closes)
    if len(votes) < len(INDICATOR_NAMES):
        return None
    total = sum((Decimal(vote) for vote in votes.values()), Decimal(0))
    return total / Decimal(len(INDICATOR_NAMES))


def trend_scores(
    histories: dict[str, ContractHistory],
    contract_ids: tuple[str, ...],
    decision_close_ns: int,
) -> dict[str, Decimal]:
    """Scores at one decision, keyed by contract id, skipping contracts without one."""
    scores: dict[str, Decimal] = {}
    for contract_id in contract_ids:
        score = trend_score(closes_before(histories[contract_id], decision_close_ns))
        if score is not None:
            scores[contract_id] = score
    return scores


def threshold_sign(score: Decimal, threshold: Decimal) -> int:
    """+1 at or above `threshold`, -1 at or below `-threshold`, 0 in the dead band."""
    if score >= threshold:
        return 1
    if score <= -threshold:
        return -1
    return 0


def build_trend_weight_vectors(
    histories: dict[str, ContractHistory],
    snapshot: UniverseSnapshot,
    *,
    spec: TrendFamilySpec,
) -> dict[str, WeightVector]:
    """Build every member and control weight vector for one rebalance.

    The four-week member's vector is the same as its one-week twin's here: the
    cohort assembly that distinguishes them lives in the fold runner, which
    holds the last four of these vectors at a quarter of capital each.
    """
    decision = snapshot.decision_close_ns
    names = [member.name for member in spec.members] + [item.name for item in spec.controls]
    if not snapshot.contracts:
        return {name: WeightVector(decision, name, (), snapshot.reason_codes) for name in names}

    eligible = tuple(item.contract_id for item in snapshot.contracts)
    scores = trend_scores(histories, eligible, decision)
    vectors: dict[str, WeightVector] = {}
    for member in spec.members:
        weights: tuple[tuple[str, Decimal], ...]
        if member.kind == "cross_sectional":
            weights = _cross_sectional_weights(scores, spec.weights, reverse=False)
        else:
            if member.threshold is None:
                raise ValueError(f"time-series member {member.name} has no threshold")
            weights = _time_series_weights(
                scores, histories, eligible, decision, spec.weights, member.threshold
            )
        vectors[member.name] = WeightVector(decision, member.name, weights, ())

    vectors["no_trade"] = WeightVector(decision, "no_trade", (), ())
    share = Decimal(1) / Decimal(len(eligible))
    vectors["passive_long_ew"] = WeightVector(
        decision,
        "passive_long_ew",
        tuple((contract_id, share) for contract_id in sorted(eligible)),
        (),
    )
    rng = Random(spec.statistics.random_seed ^ decision)
    draws = {contract_id: Decimal(str(rng.random())) for contract_id in eligible}
    vectors["random_ranks"] = WeightVector(
        decision,
        "random_ranks",
        _cross_sectional_weights(draws, spec.weights, reverse=False),
        (),
    )
    return vectors


def _time_series_weights(
    scores: dict[str, Decimal],
    histories: dict[str, ContractHistory],
    eligible: tuple[str, ...],
    decision_close_ns: int,
    rules: PanelWeightRules,
    threshold: Decimal,
) -> tuple[tuple[str, Decimal], ...]:
    """P1.27's `_time_series_weights` with the score's thresholded sign.

    A contract without a score, inside the dead band, or without a complete
    volatility window carries no position; the rest take `sign / max(sigma,
    floor)`, normalised to unit gross and water-filled at
    `time_series_cap_numerator / n` over the signalled contracts.
    """
    raw: dict[str, Decimal] = {}
    for contract_id in eligible:
        score = scores.get(contract_id)
        if score is None:
            continue
        sign = threshold_sign(score, threshold)
        if sign == 0:
            continue
        sigma = _annualised_volatility(
            histories[contract_id], decision_close_ns, rules.volatility_window_days
        )
        if sigma is None:
            continue
        raw[contract_id] = Decimal(sign) / max(sigma, rules.volatility_floor)
    if not raw:
        return ()
    cap = rules.time_series_cap_numerator / Decimal(len(raw))
    weights = _water_fill(_normalise(raw), cap)
    return tuple(sorted(weights.items(), key=lambda item: item[0]))


def _sign(value: Decimal) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0
