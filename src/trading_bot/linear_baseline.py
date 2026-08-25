"""Small deterministic ridge baseline with train-only preprocessing."""

from dataclasses import dataclass
from decimal import Decimal

from trading_bot.bar_research import BarSample
from trading_bot.evaluation import evaluate_signals
from trading_bot.strategy import CostScenario


@dataclass(frozen=True, slots=True)
class RidgeModel:
    feature_means: tuple[Decimal, ...]
    feature_scales: tuple[Decimal, ...]
    coefficients: tuple[Decimal, ...]
    target_mean: Decimal
    alpha: Decimal


def fit_ridge(samples: tuple[BarSample, ...], *, alpha: Decimal) -> RidgeModel:
    if len(samples) < 2:
        raise ValueError("ridge training requires at least two samples")
    if not alpha.is_finite() or alpha <= 0:
        raise ValueError("ridge alpha must be positive and finite")
    rows = tuple(_features(sample) for sample in samples)
    width = len(rows[0])
    count = Decimal(len(rows))
    means = tuple(sum((row[index] for row in rows), Decimal(0)) / count for index in range(width))
    scales: list[Decimal] = []
    for index, mean in enumerate(means):
        variance = sum(((row[index] - mean) ** 2 for row in rows), Decimal(0)) / count
        scales.append(variance.sqrt() if variance > 0 else Decimal(1))
    normalized = tuple(
        tuple((value - means[index]) / scales[index] for index, value in enumerate(row))
        for row in rows
    )
    target_mean = sum((sample.forward_return for sample in samples), Decimal(0)) / count
    centered_targets = tuple(sample.forward_return - target_mean for sample in samples)
    gram = [
        [
            sum((row[left] * row[right] for row in normalized), Decimal(0))
            + (alpha if left == right else Decimal(0))
            for right in range(width)
        ]
        for left in range(width)
    ]
    target = [
        sum(
            (row[index] * value for row, value in zip(normalized, centered_targets, strict=True)),
            Decimal(0),
        )
        for index in range(width)
    ]
    coefficients = tuple(_solve_linear_system(gram, target))
    return RidgeModel(means, tuple(scales), coefficients, target_mean, alpha)


def predict_ridge(model: RidgeModel, sample: BarSample) -> Decimal:
    features = _features(sample)
    normalized = tuple(
        (value - model.feature_means[index]) / model.feature_scales[index]
        for index, value in enumerate(features)
    )
    return model.target_mean + sum(
        (
            coefficient * value
            for coefficient, value in zip(model.coefficients, normalized, strict=True)
        ),
        Decimal(0),
    )


def predict_signals(
    model: RidgeModel, samples: tuple[BarSample, ...], *, threshold: Decimal
) -> tuple[int, ...]:
    if threshold < 0:
        raise ValueError("prediction threshold must be non-negative")
    predictions = tuple(predict_ridge(model, sample) for sample in samples)
    return tuple(
        1 if value > threshold else -1 if value < -threshold else 0 for value in predictions
    )


def select_validation_threshold(
    model: RidgeModel,
    samples: tuple[BarSample, ...],
    *,
    scenario: CostScenario,
    spread_bps: Decimal,
    minimum_trades: int = 0,
) -> Decimal:
    if not samples:
        raise ValueError("validation samples must not be empty")
    if minimum_trades < 0:
        raise ValueError("minimum validation trades must be non-negative")
    predictions = tuple(predict_ridge(model, sample) for sample in samples)
    candidates = validation_threshold_candidates(predictions)
    outcomes = tuple(sample.forward_return for sample in samples)
    spreads = tuple(spread_bps for _ in samples)
    scored = []
    for threshold in candidates:
        signals = tuple(
            1 if value > threshold else -1 if value < -threshold else 0 for value in predictions
        )
        result = evaluate_signals(signals, outcomes, spreads, scenario)
        if result.trade_count >= minimum_trades:
            scored.append((result.total_net_return, threshold))
    if not scored:
        raise ValueError("no validation threshold meets the minimum trade count")
    return max(scored)[1]


def validation_threshold_candidates(
    predictions: tuple[Decimal, ...], *, maximum_candidates: int = 9
) -> tuple[Decimal, ...]:
    if not predictions or maximum_candidates < 3:
        raise ValueError("threshold search requires predictions and at least three candidates")
    ordered = tuple(sorted(abs(value) for value in predictions))
    slots = maximum_candidates - 1
    selected = {ordered[index * (len(ordered) - 1) // (slots - 1)] for index in range(slots)}
    return tuple(sorted({Decimal(0), *selected}))


def _features(sample: BarSample) -> tuple[Decimal, ...]:
    return (
        sample.observed_return,
        sample.range_bps,
        sample.volume_change if sample.volume_change is not None else Decimal(0),
        sample.cross_venue_basis if sample.cross_venue_basis is not None else Decimal(0),
    )


def _solve_linear_system(matrix: list[list[Decimal]], target: list[Decimal]) -> list[Decimal]:
    size = len(target)
    augmented = [[*row, target[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if augmented[pivot][column] == 0:
            raise ValueError("ridge system is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column], strict=True)
            ]
    return [augmented[index][-1] for index in range(size)]
