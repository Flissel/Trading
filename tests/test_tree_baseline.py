from decimal import Decimal

from trading_bot.bar_research import BarSample
from trading_bot.strategy import CostScenario
from trading_bot.tree_baseline import (
    BoostedStumpModel,
    RegressionStump,
    fit_boosted_stumps,
    predict_signals,
    predict_tree,
    select_validation_threshold,
)


def sample(index: int, feature: str, target: str) -> BarSample:
    value = Decimal(feature)
    return BarSample(
        sample_id=f"s{index}",
        decision_time_ns=index + 1,
        label_available_time_ns=index + 2,
        input_source_ids=(f"e{index}",),
        observed_return=value,
        range_bps=Decimal(1),
        volume_change=Decimal(0),
        cross_venue_basis=Decimal(0),
        forward_return=Decimal(target),
    )


def test_boosted_stumps_learn_bounded_nonlinearity_without_small_leaves() -> None:
    training = (
        sample(0, "-3", "0.10"),
        sample(1, "-2", "0.08"),
        sample(2, "-1", "-0.05"),
        sample(3, "0", "-0.08"),
        sample(4, "1", "-0.05"),
        sample(5, "2", "0.08"),
        sample(6, "3", "0.10"),
    )

    model = fit_boosted_stumps(
        training,
        estimator_count=8,
        learning_rate=Decimal("0.3"),
        minimum_leaf_samples=2,
        maximum_split_candidates=5,
    )

    assert predict_tree(model, sample(20, "-3", "0")) > 0
    assert predict_tree(model, sample(21, "0", "0")) < 0
    assert predict_tree(model, sample(22, "3", "0")) > 0
    assert all(
        stump.left_sample_count >= 2 and stump.right_sample_count >= 2 for stump in model.stumps
    )
    assert sum(model.feature_importance) == Decimal(1)


def test_tree_validation_threshold_excludes_profitable_single_trade_candidate() -> None:
    model = BoostedStumpModel(
        base_value=Decimal("0.1"),
        learning_rate=Decimal(1),
        requested_estimator_count=2,
        minimum_leaf_samples=1,
        maximum_split_candidates=3,
        stumps=(
            RegressionStump(0, Decimal("0.15"), Decimal(0), Decimal("0.7"), 1, 2, Decimal(1)),
            RegressionStump(0, Decimal("0.25"), Decimal(0), Decimal("0.1"), 2, 1, Decimal(1)),
        ),
        feature_importance=(Decimal(1), Decimal(0), Decimal(0), Decimal(0)),
    )
    validation = (
        sample(30, "0.3", "0.10"),
        sample(31, "0.2", "-0.05"),
        sample(32, "0.1", "-0.20"),
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
