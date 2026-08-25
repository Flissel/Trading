"""Economic evaluation with dependence-aware deterministic uncertainty."""

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from random import Random

from trading_bot.strategy import CostScenario


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    scenario_name: str
    net_returns: tuple[Decimal, ...]
    trade_count: int
    total_net_return: Decimal
    mean_net_return: Decimal
    win_rate: Decimal
    maximum_drawdown: Decimal


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    lower: Decimal
    upper: Decimal


@dataclass(frozen=True, slots=True)
class BootstrapMeanTest:
    interval: BootstrapInterval
    observed_mean: Decimal
    one_sided_p_value: Decimal


class PromotionStatus(StrEnum):
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    REJECTED = "rejected"
    ELIGIBLE_FOR_SHADOW = "eligible_for_shadow"


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    status: PromotionStatus
    reason_codes: tuple[str, ...]


def evaluate_signals(
    signals: tuple[int, ...],
    forward_returns: tuple[Decimal, ...],
    observed_spread_bps: tuple[Decimal, ...],
    scenario: CostScenario,
) -> EvaluationResult:
    if not (len(signals) == len(forward_returns) == len(observed_spread_bps)):
        raise ValueError("signals, returns, and spreads must have equal length")
    if any(signal not in (-1, 0, 1) for signal in signals):
        raise ValueError("signals must be -1, 0, or 1")

    net_returns: list[Decimal] = []
    active_returns: list[Decimal] = []
    for signal, forward_return, spread_bps in zip(
        signals, forward_returns, observed_spread_bps, strict=True
    ):
        if signal == 0:
            net = Decimal(0)
        else:
            net = Decimal(signal) * forward_return - scenario.round_trip_cost_return(spread_bps)
            active_returns.append(net)
        net_returns.append(net)

    total = sum(net_returns, Decimal(0))
    trade_count = len(active_returns)
    mean = total / Decimal(trade_count) if trade_count else Decimal(0)
    wins = sum(1 for value in active_returns if value > 0)
    win_rate = Decimal(wins) / Decimal(trade_count) if trade_count else Decimal(0)
    return EvaluationResult(
        scenario_name=scenario.name,
        net_returns=tuple(net_returns),
        trade_count=trade_count,
        total_net_return=total,
        mean_net_return=mean,
        win_rate=win_rate,
        maximum_drawdown=_maximum_drawdown(net_returns),
    )


def block_bootstrap_mean_interval(
    values: tuple[Decimal, ...],
    *,
    block_length: int,
    repetitions: int,
    seed: int,
    confidence: Decimal,
) -> BootstrapInterval:
    if not values:
        raise ValueError("bootstrap values must not be empty")
    if block_length < 1 or block_length > len(values):
        raise ValueError("block_length is outside the sample")
    if repetitions < 2:
        raise ValueError("repetitions must be at least two")
    if not Decimal(0) < confidence < Decimal(1):
        raise ValueError("confidence must be between zero and one")

    means = _block_bootstrap_means(values, block_length, repetitions, seed)
    means.sort()
    tail = (Decimal(1) - confidence) / Decimal(2)
    lower_index = int(tail * Decimal(repetitions - 1))
    upper_index = int((Decimal(1) - tail) * Decimal(repetitions - 1))
    return BootstrapInterval(means[lower_index], means[upper_index])


def block_bootstrap_mean_test(
    values: tuple[Decimal, ...],
    *,
    block_length: int,
    repetitions: int,
    seed: int,
    confidence: Decimal,
) -> BootstrapMeanTest:
    interval = block_bootstrap_mean_interval(
        values,
        block_length=block_length,
        repetitions=repetitions,
        seed=seed,
        confidence=confidence,
    )
    observed = sum(values, Decimal(0)) / Decimal(len(values))
    centered = tuple(value - observed for value in values)
    null_means = _block_bootstrap_means(centered, block_length, repetitions, seed)
    exceedances = sum(1 for value in null_means if value >= observed)
    p_value = Decimal(exceedances + 1) / Decimal(repetitions + 1)
    return BootstrapMeanTest(interval, observed, p_value)


def benjamini_hochberg(p_values: dict[str, Decimal]) -> dict[str, Decimal]:
    if not p_values or any(not Decimal(0) <= value <= Decimal(1) for value in p_values.values()):
        raise ValueError("p-values must be a non-empty mapping within zero and one")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    adjusted: dict[str, Decimal] = {}
    running = Decimal(1)
    for reverse_index in range(count - 1, -1, -1):
        name, p_value = ordered[reverse_index]
        rank = reverse_index + 1
        running = min(running, p_value * Decimal(count) / Decimal(rank), Decimal(1))
        adjusted[name] = running
    return {name: adjusted[name] for name in p_values}


def assess_shadow_promotion(
    *,
    base: EvaluationResult,
    adverse: EvaluationResult,
    base_lower_confidence_bound: Decimal,
    minimum_episodes: int,
) -> PromotionDecision:
    if base.trade_count < minimum_episodes:
        return PromotionDecision(PromotionStatus.INSUFFICIENT_EVIDENCE, ("EPISODE_FLOOR_NOT_MET",))
    reasons: list[str] = []
    if base.total_net_return <= 0:
        reasons.append("BASE_NET_NON_POSITIVE")
    if base_lower_confidence_bound <= 0:
        reasons.append("BASE_LOWER_BOUND_NON_POSITIVE")
    if adverse.total_net_return <= 0:
        reasons.append("ADVERSE_NET_NON_POSITIVE")
    if reasons:
        return PromotionDecision(PromotionStatus.REJECTED, tuple(reasons))
    return PromotionDecision(PromotionStatus.ELIGIBLE_FOR_SHADOW, ())


def _maximum_drawdown(net_returns: list[Decimal]) -> Decimal:
    equity = Decimal(1)
    peak = equity
    maximum = Decimal(0)
    for value in net_returns:
        equity *= Decimal(1) + value
        peak = max(peak, equity)
        if peak > 0:
            maximum = max(maximum, (peak - equity) / peak)
    return maximum


def _block_bootstrap_means(
    values: tuple[Decimal, ...], block_length: int, repetitions: int, seed: int
) -> list[Decimal]:
    random = Random(seed)
    means: list[Decimal] = []
    for _ in range(repetitions):
        sample: list[Decimal] = []
        while len(sample) < len(values):
            start = random.randrange(len(values))
            for offset in range(block_length):
                sample.append(values[(start + offset) % len(values)])
                if len(sample) == len(values):
                    break
        means.append(sum(sample, Decimal(0)) / Decimal(len(sample)))
    means.sort()
    return means
