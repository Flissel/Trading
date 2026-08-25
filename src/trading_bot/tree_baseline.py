"""Small deterministic gradient-boosted regression-stump baseline."""

from dataclasses import dataclass
from decimal import Decimal

from trading_bot.bar_research import BarSample
from trading_bot.evaluation import evaluate_signals
from trading_bot.linear_baseline import validation_threshold_candidates
from trading_bot.strategy import CostScenario

FEATURE_NAMES = ("observed_return", "range_bps", "volume_change", "cross_venue_basis")


@dataclass(frozen=True, slots=True)
class RegressionStump:
    feature_index: int
    threshold: Decimal
    left_value: Decimal
    right_value: Decimal
    left_sample_count: int
    right_sample_count: int
    gain: Decimal


@dataclass(frozen=True, slots=True)
class BoostedStumpModel:
    base_value: Decimal
    learning_rate: Decimal
    requested_estimator_count: int
    minimum_leaf_samples: int
    maximum_split_candidates: int
    stumps: tuple[RegressionStump, ...]
    feature_importance: tuple[Decimal, ...]


def fit_boosted_stumps(
    samples: tuple[BarSample, ...],
    *,
    estimator_count: int,
    learning_rate: Decimal,
    minimum_leaf_samples: int,
    maximum_split_candidates: int,
) -> BoostedStumpModel:
    if len(samples) < 2 * minimum_leaf_samples:
        raise ValueError("tree training requires two complete minimum-size leaves")
    if (
        estimator_count < 1
        or not Decimal(0) < learning_rate <= Decimal(1)
        or minimum_leaf_samples < 1
        or maximum_split_candidates < 1
    ):
        raise ValueError("tree training parameters are invalid")
    rows = tuple(tuple(float(value) for value in _features(sample)) for sample in samples)
    targets = tuple(float(sample.forward_return) for sample in samples)
    base_value = sum(targets) / len(targets)
    predictions = [base_value] * len(samples)
    fitted: list[RegressionStump] = []
    gains = [0.0] * len(FEATURE_NAMES)
    step = float(learning_rate)
    for _ in range(estimator_count):
        residuals = tuple(
            target - prediction for target, prediction in zip(targets, predictions, strict=True)
        )
        split = _best_split(
            rows,
            residuals,
            minimum_leaf_samples=minimum_leaf_samples,
            maximum_split_candidates=maximum_split_candidates,
        )
        if split is None:
            break
        feature_index, threshold, left_mean, right_mean, left_count, right_count, gain = split
        left_update = step * left_mean
        right_update = step * right_mean
        for index, row in enumerate(rows):
            predictions[index] += left_update if row[feature_index] <= threshold else right_update
        gains[feature_index] += gain
        fitted.append(
            RegressionStump(
                feature_index=feature_index,
                threshold=Decimal(str(threshold)),
                left_value=Decimal(str(left_update)),
                right_value=Decimal(str(right_update)),
                left_sample_count=left_count,
                right_sample_count=right_count,
                gain=Decimal(str(gain)),
            )
        )
    total_gain = sum(gains)
    importance = tuple(
        Decimal(str(gain / total_gain)) if total_gain > 0 else Decimal(0) for gain in gains
    )
    return BoostedStumpModel(
        base_value=Decimal(str(base_value)),
        learning_rate=learning_rate,
        requested_estimator_count=estimator_count,
        minimum_leaf_samples=minimum_leaf_samples,
        maximum_split_candidates=maximum_split_candidates,
        stumps=tuple(fitted),
        feature_importance=importance,
    )


def predict_tree(model: BoostedStumpModel, sample: BarSample) -> Decimal:
    features = _features(sample)
    return model.base_value + sum(
        (
            stump.left_value
            if features[stump.feature_index] <= stump.threshold
            else stump.right_value
            for stump in model.stumps
        ),
        Decimal(0),
    )


def predict_signals(
    model: BoostedStumpModel, samples: tuple[BarSample, ...], *, threshold: Decimal
) -> tuple[int, ...]:
    if threshold < 0:
        raise ValueError("prediction threshold must be non-negative")
    predictions = tuple(predict_tree(model, sample) for sample in samples)
    return tuple(
        1 if value > threshold else -1 if value < -threshold else 0 for value in predictions
    )


def select_validation_threshold(
    model: BoostedStumpModel,
    samples: tuple[BarSample, ...],
    *,
    scenario: CostScenario,
    spread_bps: Decimal,
    minimum_trades: int,
) -> Decimal:
    if not samples or minimum_trades < 0:
        raise ValueError("validation samples and minimum trades are invalid")
    predictions = tuple(predict_tree(model, sample) for sample in samples)
    candidates = validation_threshold_candidates(predictions)
    outcomes = tuple(sample.forward_return for sample in samples)
    spreads = tuple(spread_bps for _ in samples)
    scored: list[tuple[Decimal, Decimal]] = []
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


def _best_split(
    rows: tuple[tuple[float, ...], ...],
    residuals: tuple[float, ...],
    *,
    minimum_leaf_samples: int,
    maximum_split_candidates: int,
) -> tuple[int, float, float, float, int, int, float] | None:
    parent_sum = sum(residuals)
    parent_squares = sum(value * value for value in residuals)
    parent_loss = parent_squares - parent_sum * parent_sum / len(residuals)
    best: tuple[int, float, float, float, int, int, float] | None = None
    best_gain = 0.0
    for feature_index in range(len(FEATURE_NAMES)):
        values = tuple(row[feature_index] for row in rows)
        for threshold in _split_candidates(
            values,
            minimum_leaf_samples=minimum_leaf_samples,
            maximum_candidates=maximum_split_candidates,
        ):
            left = tuple(
                residual
                for value, residual in zip(values, residuals, strict=True)
                if value <= threshold
            )
            right = tuple(
                residual
                for value, residual in zip(values, residuals, strict=True)
                if value > threshold
            )
            if len(left) < minimum_leaf_samples or len(right) < minimum_leaf_samples:
                continue
            left_sum = sum(left)
            right_sum = sum(right)
            loss = (
                sum(value * value for value in left)
                - left_sum * left_sum / len(left)
                + sum(value * value for value in right)
                - right_sum * right_sum / len(right)
            )
            gain = parent_loss - loss
            if gain > best_gain:
                best_gain = gain
                best = (
                    feature_index,
                    threshold,
                    left_sum / len(left),
                    right_sum / len(right),
                    len(left),
                    len(right),
                    gain,
                )
    return best


def _split_candidates(
    values: tuple[float, ...], *, minimum_leaf_samples: int, maximum_candidates: int
) -> tuple[float, ...]:
    ordered = tuple(sorted(values))
    candidates = tuple(
        (ordered[index] + ordered[index + 1]) / 2
        for index in range(minimum_leaf_samples - 1, len(ordered) - minimum_leaf_samples)
        if ordered[index] < ordered[index + 1]
    )
    if len(candidates) <= maximum_candidates:
        return candidates
    if maximum_candidates == 1:
        return (candidates[len(candidates) // 2],)
    return tuple(
        candidates[index * (len(candidates) - 1) // (maximum_candidates - 1)]
        for index in range(maximum_candidates)
    )


def _features(sample: BarSample) -> tuple[Decimal, ...]:
    return (
        sample.observed_return,
        sample.range_bps,
        sample.volume_change if sample.volume_change is not None else Decimal(0),
        sample.cross_venue_basis if sample.cross_venue_basis is not None else Decimal(0),
    )
