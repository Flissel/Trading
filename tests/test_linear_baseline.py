from decimal import Decimal

from trading_bot.bar_research import BarSample
from trading_bot.linear_baseline import (
    RidgeModel,
    fit_ridge,
    predict_ridge,
    predict_signals,
    select_validation_threshold,
    validation_threshold_candidates,
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


def test_ridge_fits_scaler_and_coefficients_from_training_rows_only() -> None:
    training = (
        sample(0, "-2", "-0.04"),
        sample(1, "-1", "-0.02"),
        sample(2, "1", "0.02"),
        sample(3, "2", "0.04"),
    )

    model = fit_ridge(training, alpha=Decimal("0.1"))

    assert model.feature_means[0] == 0
    assert model.target_mean == 0
    assert predict_ridge(model, sample(10, "3", "999")) > 0
    assert predict_ridge(model, sample(11, "-3", "-999")) < 0


def test_validation_threshold_prefers_hold_when_all_trades_lose_after_costs() -> None:
    training = (
        sample(0, "-2", "-0.001"),
        sample(1, "-1", "-0.0005"),
        sample(2, "1", "0.0005"),
        sample(3, "2", "0.001"),
    )
    validation = (
        sample(4, "1", "-0.01"),
        sample(5, "2", "-0.01"),
    )
    model = fit_ridge(training, alpha=Decimal("0.1"))

    threshold = select_validation_threshold(
        model,
        validation,
        scenario=CostScenario("base", Decimal("1"), Decimal("1"), Decimal("1"), Decimal(0)),
        spread_bps=Decimal("2"),
    )

    assert threshold == max(abs(predict_ridge(model, item)) for item in validation)
    assert predict_signals(model, validation, threshold=threshold) == (0, 0)


def test_validation_threshold_search_is_bounded_for_large_folds() -> None:
    predictions = tuple(Decimal(index) / Decimal(10_000) for index in range(10_000))

    candidates = validation_threshold_candidates(predictions)

    assert len(candidates) <= 9
    assert candidates[0] == 0
    assert candidates[-1] == Decimal("0.9999")


def test_validation_threshold_excludes_profitable_single_trade_candidate() -> None:
    model = RidgeModel(
        feature_means=(Decimal(0), Decimal(0), Decimal(0), Decimal(0)),
        feature_scales=(Decimal(1), Decimal(1), Decimal(1), Decimal(1)),
        coefficients=(Decimal(1), Decimal(0), Decimal(0), Decimal(0)),
        target_mean=Decimal(0),
        alpha=Decimal("0.1"),
    )
    validation = (
        sample(20, "0.9", "0.10"),
        sample(21, "0.8", "-0.05"),
        sample(22, "0.1", "-0.20"),
    )

    threshold = select_validation_threshold(
        model,
        validation,
        scenario=CostScenario("free", Decimal(0), Decimal(0), Decimal(0), Decimal(0)),
        spread_bps=Decimal(0),
        minimum_trades=2,
    )

    assert threshold == Decimal("0.1")
    assert predict_signals(model, validation, threshold=threshold) == (1, 1, 0)
