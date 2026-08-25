from decimal import Decimal

from trading_bot.bar_research import BarSample
from trading_bot.logistic_baseline import (
    LogisticModel,
    PlattCalibrator,
    fit_logistic,
    fit_platt_calibrator,
    predict_calibrated_probability,
    predict_probability,
    predict_signals,
    probability_metrics,
    select_validation_margin,
)
from trading_bot.strategy import CostScenario


def sample(index: int, feature: str, target: str) -> BarSample:
    value = Decimal(feature)
    return BarSample(
        sample_id=f"s{index}",
        decision_time_ns=index + 1,
        label_available_time_ns=index + 2,
        input_source_ids=(f"e{index}",),
        observed_return=value,
        range_bps=abs(value) + Decimal(1),
        volume_change=value / Decimal(2),
        cross_venue_basis=value / Decimal(10),
        forward_return=Decimal(target),
    )


def test_logistic_learns_direction_from_training_rows_only() -> None:
    training = (
        sample(0, "-2", "-0.04"),
        sample(1, "-1", "-0.02"),
        sample(2, "1", "0.02"),
        sample(3, "2", "0.04"),
    )

    model = fit_logistic(
        training,
        l2=Decimal("0.1"),
        iterations=200,
        learning_rate=Decimal("0.2"),
    )

    assert model.feature_means[0] == 0
    assert predict_probability(model, sample(10, "3", "-999")) > Decimal("0.5")
    assert predict_probability(model, sample(11, "-3", "999")) < Decimal("0.5")


def test_platt_calibration_reduces_overconfident_brier_score() -> None:
    model = LogisticModel(
        feature_means=(Decimal(0), Decimal(0), Decimal(0), Decimal(0)),
        feature_scales=(Decimal(1), Decimal(1), Decimal(1), Decimal(1)),
        coefficients=(Decimal(4), Decimal(0), Decimal(0), Decimal(0)),
        intercept=Decimal(0),
        l2=Decimal(0),
        iterations=1,
        learning_rate=Decimal("0.1"),
    )
    calibration = (
        sample(20, "1", "1"),
        sample(21, "1", "1"),
        sample(22, "1", "1"),
        sample(23, "1", "-1"),
        sample(24, "-1", "1"),
        sample(25, "-1", "-1"),
        sample(26, "-1", "-1"),
        sample(27, "-1", "-1"),
    )
    raw = tuple(predict_probability(model, item) for item in calibration)

    calibrator = fit_platt_calibrator(
        model,
        calibration,
        l2=Decimal("0.001"),
        iterations=1_000,
        learning_rate=Decimal("0.1"),
    )
    calibrated = tuple(
        predict_calibrated_probability(model, calibrator, item) for item in calibration
    )

    assert (
        probability_metrics(calibrated, calibration).brier_score
        < probability_metrics(raw, calibration).brier_score
    )


def test_validation_margin_excludes_profitable_single_trade_candidate() -> None:
    model = LogisticModel(
        feature_means=(Decimal(0), Decimal(0), Decimal(0), Decimal(0)),
        feature_scales=(Decimal(1), Decimal(1), Decimal(1), Decimal(1)),
        coefficients=(Decimal(1), Decimal(0), Decimal(0), Decimal(0)),
        intercept=Decimal(0),
        l2=Decimal(0),
        iterations=1,
        learning_rate=Decimal("0.1"),
    )
    calibrator = PlattCalibrator(
        slope=Decimal(1),
        intercept=Decimal(0),
        l2=Decimal(0),
        iterations=1,
        learning_rate=Decimal("0.1"),
    )
    selection = (
        sample(30, "3", "0.10"),
        sample(31, "2", "-0.05"),
        sample(32, "0.1", "-0.20"),
    )

    margin = select_validation_margin(
        model,
        calibrator,
        selection,
        scenario=CostScenario("free", Decimal(0), Decimal(0), Decimal(0), Decimal(0)),
        spread_bps=Decimal(0),
        minimum_trades=2,
    )

    assert predict_signals(model, calibrator, selection, margin=margin) == (1, 1, 0)
