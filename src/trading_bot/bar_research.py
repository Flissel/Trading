"""Leakage-safe candle features and development-only baseline evaluation."""

from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal
from enum import StrEnum
from itertools import pairwise

from trading_bot.evaluation import EvaluationResult, evaluate_signals
from trading_bot.strategy import CostScenario, generate_baseline_signals


@dataclass(frozen=True, slots=True)
class ResearchBar:
    source_id: str
    venue: str
    open_time_ns: int
    available_time_ns: int
    close: Decimal
    high: Decimal
    low: Decimal
    base_volume: Decimal

    def __post_init__(self) -> None:
        if not self.source_id or not self.venue:
            raise ValueError("bar identity must not be empty")
        if self.open_time_ns < 0 or self.available_time_ns < self.open_time_ns:
            raise ValueError("bar timestamps are invalid")
        if self.close <= 0 or self.low <= 0 or self.high < self.low:
            raise ValueError("bar prices are invalid")
        if self.base_volume < 0:
            raise ValueError("bar volume must be non-negative")


@dataclass(frozen=True, slots=True)
class BarSample:
    sample_id: str
    decision_time_ns: int
    label_available_time_ns: int
    input_source_ids: tuple[str, ...]
    observed_return: Decimal
    range_bps: Decimal
    volume_change: Decimal | None
    cross_venue_basis: Decimal | None
    forward_return: Decimal


@dataclass(frozen=True, slots=True)
class BaselineScenarioEvaluation:
    baseline_name: str
    base: EvaluationResult
    adverse: EvaluationResult


class ResearchStatus(StrEnum):
    DEVELOPMENT_ONLY = "development_only"


@dataclass(frozen=True, slots=True)
class DevelopmentResearchReport:
    status: ResearchStatus
    reason_codes: tuple[str, ...]
    total_sample_count: int
    train_sample_count: int
    oos_sample_count: int
    oos_start_time_ns: int
    cross_venue_coverage: Decimal
    baselines: tuple[BaselineScenarioEvaluation, ...]


def build_bar_samples(
    primary: tuple[ResearchBar, ...],
    reference: tuple[ResearchBar, ...],
    *,
    horizon_bars: int = 1,
) -> tuple[BarSample, ...]:
    if horizon_bars < 1:
        raise ValueError("prediction horizon must contain at least one bar")
    _validate_chronology(primary)
    reference_by_time = {item.open_time_ns: item for item in reference}
    samples: list[BarSample] = []
    for index in range(1, len(primary) - horizon_bars):
        previous = primary[index - 1]
        current = primary[index]
        future = primary[index + horizon_bars]
        observed_return = current.close / previous.close - Decimal(1)
        forward_return = future.close / current.close - Decimal(1)
        volume_change = None
        if previous.base_volume > 0:
            volume_change = current.base_volume / previous.base_volume - Decimal(1)
        matched = reference_by_time.get(current.open_time_ns)
        input_ids = [previous.source_id, current.source_id]
        cross_basis = None
        if matched is not None and matched.available_time_ns <= current.available_time_ns:
            cross_basis = current.close / matched.close - Decimal(1)
            input_ids.append(matched.source_id)
        samples.append(
            BarSample(
                sample_id=(
                    f"{current.venue}:{current.open_time_ns}"
                    if horizon_bars == 1
                    else f"{current.venue}:{current.open_time_ns}:h{horizon_bars}"
                ),
                decision_time_ns=current.available_time_ns,
                label_available_time_ns=future.available_time_ns,
                input_source_ids=tuple(input_ids),
                observed_return=observed_return,
                range_bps=(current.high - current.low) / current.close * Decimal(10_000),
                volume_change=volume_change,
                cross_venue_basis=cross_basis,
                forward_return=forward_return,
            )
        )
    return tuple(samples)


def evaluate_development_samples(
    samples: tuple[BarSample, ...],
    *,
    oos_fraction: Decimal,
    minimum_train_samples: int,
    random_seed: int,
    base_costs: CostScenario,
    adverse_costs: CostScenario,
    assumed_spread_bps: Decimal,
) -> DevelopmentResearchReport:
    if not Decimal(0) < oos_fraction < Decimal(1):
        raise ValueError("oos_fraction must be between zero and one")
    oos_count = int(
        (Decimal(len(samples)) * oos_fraction).to_integral_value(rounding=ROUND_CEILING)
    )
    train_count = len(samples) - oos_count
    if train_count < minimum_train_samples or oos_count < 1:
        raise ValueError("insufficient samples for development split")
    observed = tuple(item.observed_return for item in samples)
    baselines = generate_baseline_signals(observed, random_seed=random_seed)
    outcomes = tuple(item.forward_return for item in samples[train_count:])
    spreads = tuple(assumed_spread_bps for _ in range(oos_count))
    evaluations: list[BaselineScenarioEvaluation] = []
    for name, all_signals in baselines.items():
        signals = all_signals[train_count:]
        evaluations.append(
            BaselineScenarioEvaluation(
                baseline_name=name,
                base=evaluate_signals(signals, outcomes, spreads, base_costs),
                adverse=evaluate_signals(signals, outcomes, spreads, adverse_costs),
            )
        )
    matched = sum(1 for sample in samples if sample.cross_venue_basis is not None)
    return DevelopmentResearchReport(
        status=ResearchStatus.DEVELOPMENT_ONLY,
        reason_codes=(
            "HISTORY_TOO_SHORT_FOR_PROTOCOL",
            "NO_REGISTERED_WALK_FORWARD_FOLDS",
            "EPISODE_FLOOR_NOT_MET",
        ),
        total_sample_count=len(samples),
        train_sample_count=train_count,
        oos_sample_count=oos_count,
        oos_start_time_ns=samples[train_count].decision_time_ns,
        cross_venue_coverage=Decimal(matched) / Decimal(len(samples)),
        baselines=tuple(evaluations),
    )


def _validate_chronology(bars: tuple[ResearchBar, ...]) -> None:
    for previous, current in pairwise(bars):
        if current.open_time_ns <= previous.open_time_ns:
            raise ValueError("primary bars must be strictly chronological")
