"""Small dependency-free logistic baseline for directional probabilities."""

from dataclasses import dataclass
from decimal import Decimal
from math import exp

from trading_bot.bar_research import BarSample
from trading_bot.evaluation import evaluate_signals
from trading_bot.strategy import CostScenario


@dataclass(frozen=True, slots=True)
class LogisticModel:
    feature_means: tuple[Decimal, ...]
    feature_scales: tuple[Decimal, ...]
    coefficients: tuple[Decimal, ...]
    intercept: Decimal
    l2: Decimal
    iterations: int
    learning_rate: Decimal


@dataclass(frozen=True, slots=True)
class PlattCalibrator:
    slope: Decimal
    intercept: Decimal
    l2: Decimal
    iterations: int
    learning_rate: Decimal


@dataclass(frozen=True, slots=True)
class ProbabilityMetrics:
    brier_score: Decimal
    log_loss: Decimal


def fit_logistic(
    samples: tuple[BarSample, ...],
    *,
    l2: Decimal,
    iterations: int,
    learning_rate: Decimal,
) -> LogisticModel:
    if not samples:
        raise ValueError("logistic training samples must not be empty")
    if l2 < 0 or iterations < 1 or learning_rate <= 0:
        raise ValueError("logistic optimization parameters are invalid")
    rows = tuple(_features(sample) for sample in samples)
    width = len(rows[0])
    count = Decimal(len(rows))
    means = tuple(sum((row[index] for row in rows), Decimal(0)) / count for index in range(width))
    scales: list[Decimal] = []
    for index, mean in enumerate(means):
        variance = sum(((row[index] - mean) ** 2 for row in rows), Decimal(0)) / count
        scales.append(variance.sqrt() if variance > 0 else Decimal(1))
    normalized = tuple(
        tuple(float((value - means[index]) / scales[index]) for index, value in enumerate(row))
        for row in rows
    )
    labels = tuple(1.0 if sample.forward_return > 0 else 0.0 for sample in samples)
    if len(set(labels)) != 2:
        raise ValueError("logistic training requires both direction classes")
    coefficients = [0.0] * width
    intercept = 0.0
    step = float(learning_rate)
    penalty = float(l2)
    sample_count = float(len(samples))
    for _ in range(iterations):
        errors = []
        for row, label in zip(normalized, labels, strict=True):
            logit = intercept + sum(
                value * coefficient for value, coefficient in zip(row, coefficients, strict=True)
            )
            errors.append(_sigmoid_float(logit) - label)
        intercept -= step * sum(errors) / sample_count
        for index in range(width):
            gradient = (
                sum(error * row[index] for error, row in zip(errors, normalized, strict=True))
                / sample_count
                + penalty * coefficients[index]
            )
            coefficients[index] -= step * gradient
    return LogisticModel(
        means,
        tuple(scales),
        tuple(Decimal(str(value)) for value in coefficients),
        Decimal(str(intercept)),
        l2,
        iterations,
        learning_rate,
    )


def predict_probability(model: LogisticModel, sample: BarSample) -> Decimal:
    return _sigmoid_decimal(_logit(model, sample))


def fit_platt_calibrator(
    model: LogisticModel,
    samples: tuple[BarSample, ...],
    *,
    l2: Decimal,
    iterations: int,
    learning_rate: Decimal,
) -> PlattCalibrator:
    if not samples:
        raise ValueError("calibration samples must not be empty")
    if l2 < 0 or iterations < 1 or learning_rate <= 0:
        raise ValueError("calibration optimization parameters are invalid")
    logits = tuple(float(_logit(model, sample)) for sample in samples)
    labels = tuple(1.0 if sample.forward_return > 0 else 0.0 for sample in samples)
    slope = 1.0
    intercept = 0.0
    step = float(learning_rate)
    penalty = float(l2)
    count = float(len(samples))
    for _ in range(iterations):
        errors = tuple(
            _sigmoid_float(slope * logit + intercept) - label
            for logit, label in zip(logits, labels, strict=True)
        )
        intercept -= step * sum(errors) / count
        slope_gradient = (
            sum(error * logit for error, logit in zip(errors, logits, strict=True)) / count
            + penalty * slope
        )
        slope -= step * slope_gradient
    return PlattCalibrator(
        Decimal(str(slope)),
        Decimal(str(intercept)),
        l2,
        iterations,
        learning_rate,
    )


def predict_calibrated_probability(
    model: LogisticModel, calibrator: PlattCalibrator, sample: BarSample
) -> Decimal:
    return _sigmoid_decimal(calibrator.slope * _logit(model, sample) + calibrator.intercept)


def probability_metrics(
    probabilities: tuple[Decimal, ...], samples: tuple[BarSample, ...]
) -> ProbabilityMetrics:
    if not probabilities or len(probabilities) != len(samples):
        raise ValueError("probabilities and samples must have equal non-zero length")
    epsilon = Decimal("1e-15")
    count = Decimal(len(samples))
    brier = Decimal(0)
    log_loss = Decimal(0)
    for probability, sample in zip(probabilities, samples, strict=True):
        bounded = min(max(probability, epsilon), Decimal(1) - epsilon)
        label = Decimal(1) if sample.forward_return > 0 else Decimal(0)
        brier += (bounded - label) ** 2
        log_loss -= label * bounded.ln() + (Decimal(1) - label) * (Decimal(1) - bounded).ln()
    return ProbabilityMetrics(brier / count, log_loss / count)


def predict_signals(
    model: LogisticModel,
    calibrator: PlattCalibrator,
    samples: tuple[BarSample, ...],
    *,
    margin: Decimal,
) -> tuple[int, ...]:
    if not Decimal(0) <= margin <= Decimal("0.5"):
        raise ValueError("probability margin must be between zero and one half")
    probabilities = tuple(
        predict_calibrated_probability(model, calibrator, sample) for sample in samples
    )
    return tuple(
        1
        if probability > Decimal("0.5") + margin
        else -1
        if probability < Decimal("0.5") - margin
        else 0
        for probability in probabilities
    )


def select_validation_margin(
    model: LogisticModel,
    calibrator: PlattCalibrator,
    samples: tuple[BarSample, ...],
    *,
    scenario: CostScenario,
    spread_bps: Decimal,
    minimum_trades: int,
) -> Decimal:
    if not samples or minimum_trades < 0:
        raise ValueError("validation samples and minimum trades are invalid")
    probabilities = tuple(
        predict_calibrated_probability(model, calibrator, sample) for sample in samples
    )
    candidates = probability_margin_candidates(probabilities)
    outcomes = tuple(sample.forward_return for sample in samples)
    spreads = tuple(spread_bps for _ in samples)
    scored: list[tuple[Decimal, Decimal]] = []
    for margin in candidates:
        signals = tuple(
            1
            if probability > Decimal("0.5") + margin
            else -1
            if probability < Decimal("0.5") - margin
            else 0
            for probability in probabilities
        )
        result = evaluate_signals(signals, outcomes, spreads, scenario)
        if result.trade_count >= minimum_trades:
            scored.append((result.total_net_return, margin))
    if not scored:
        raise ValueError("no validation margin meets the minimum trade count")
    return max(scored)[1]


def probability_margin_candidates(
    probabilities: tuple[Decimal, ...], *, maximum_candidates: int = 9
) -> tuple[Decimal, ...]:
    if not probabilities or maximum_candidates < 3:
        raise ValueError("margin search requires probabilities and at least three candidates")
    ordered = tuple(sorted(abs(probability - Decimal("0.5")) for probability in probabilities))
    slots = maximum_candidates - 1
    selected = {ordered[index * (len(ordered) - 1) // (slots - 1)] for index in range(slots)}
    return tuple(sorted({Decimal(0), *selected}))


def _logit(model: LogisticModel, sample: BarSample) -> Decimal:
    features = _features(sample)
    return model.intercept + sum(
        (
            coefficient * (value - model.feature_means[index]) / model.feature_scales[index]
            for index, (coefficient, value) in enumerate(
                zip(model.coefficients, features, strict=True)
            )
        ),
        Decimal(0),
    )


def _features(sample: BarSample) -> tuple[Decimal, ...]:
    return (
        sample.observed_return,
        sample.range_bps,
        sample.volume_change if sample.volume_change is not None else Decimal(0),
        sample.cross_venue_basis if sample.cross_venue_basis is not None else Decimal(0),
    )


def _sigmoid_float(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + exp(-value))
    exponent = exp(value)
    return exponent / (1.0 + exponent)


def _sigmoid_decimal(value: Decimal) -> Decimal:
    if value >= 0:
        return Decimal(1) / (Decimal(1) + (-value).exp())
    exponent = value.exp()
    return exponent / (Decimal(1) + exponent)
