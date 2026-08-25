from decimal import Decimal

from trading_bot.evaluation import (
    PromotionStatus,
    assess_shadow_promotion,
    benjamini_hochberg,
    block_bootstrap_mean_interval,
    block_bootstrap_mean_test,
    evaluate_signals,
)
from trading_bot.strategy import CostScenario, generate_baseline_signals


def test_cost_model_is_shared_by_long_short_and_no_trade() -> None:
    scenario = CostScenario(
        name="base",
        fee_bps_per_side=Decimal("2"),
        spread_multiplier=Decimal("1"),
        slippage_bps_per_side=Decimal("1"),
        funding_bps=Decimal("1"),
    )

    result = evaluate_signals(
        signals=(1, -1, 0),
        forward_returns=(Decimal("0.01"), Decimal("-0.02"), Decimal("0.50")),
        observed_spread_bps=(Decimal("4"), Decimal("4"), Decimal("4")),
        scenario=scenario,
    )

    assert result.net_returns == (Decimal("0.0089"), Decimal("0.0189"), Decimal("0"))
    assert result.trade_count == 2
    assert result.total_net_return == Decimal("0.0278")
    assert result.mean_net_return == Decimal("0.0139")


def test_simple_baselines_are_deterministic_and_do_not_see_future_returns() -> None:
    observed = (Decimal("0.01"), Decimal("-0.02"), Decimal("0"))

    first = generate_baseline_signals(observed, random_seed=7)
    second = generate_baseline_signals(observed, random_seed=7)

    assert first == second
    assert first["no_trade"] == (0, 0, 0)
    assert first["momentum"] == (1, -1, 0)
    assert first["mean_reversion"] == (-1, 1, 0)
    assert set(first["random"]) <= {-1, 0, 1}


def test_block_bootstrap_interval_is_seeded_and_contains_observed_mean() -> None:
    values = tuple(Decimal(value) for value in ("-0.02", "0.01", "0.03", "-0.01", "0.04"))

    first = block_bootstrap_mean_interval(
        values, block_length=2, repetitions=200, seed=11, confidence=Decimal("0.90")
    )
    second = block_bootstrap_mean_interval(
        values, block_length=2, repetitions=200, seed=11, confidence=Decimal("0.90")
    )

    assert first == second
    assert first.lower <= Decimal("0.01") <= first.upper


def test_centered_block_bootstrap_and_bh_correction_are_deterministic() -> None:
    test = block_bootstrap_mean_test(
        (Decimal("0.01"),) * 8,
        block_length=2,
        repetitions=200,
        seed=11,
        confidence=Decimal("0.95"),
    )

    assert test.one_sided_p_value == Decimal(1) / Decimal(201)
    assert benjamini_hochberg(
        {
            "a": Decimal("0.01"),
            "b": Decimal("0.04"),
            "c": Decimal("0.20"),
        }
    ) == {
        "a": Decimal("0.03"),
        "b": Decimal("0.06"),
        "c": Decimal("0.20"),
    }


def test_promotion_stays_insufficient_until_episode_floor_is_met() -> None:
    base = evaluate_signals(
        signals=(1, 1),
        forward_returns=(Decimal("0.02"), Decimal("0.03")),
        observed_spread_bps=(Decimal("1"), Decimal("1")),
        scenario=CostScenario("base", Decimal("0"), Decimal("1"), Decimal("0"), Decimal("0")),
    )
    adverse = evaluate_signals(
        signals=(1, 1),
        forward_returns=(Decimal("0.02"), Decimal("0.03")),
        observed_spread_bps=(Decimal("2"), Decimal("2")),
        scenario=CostScenario("adverse", Decimal("0"), Decimal("2"), Decimal("0"), Decimal("0")),
    )

    decision = assess_shadow_promotion(
        base=base,
        adverse=adverse,
        base_lower_confidence_bound=Decimal("0.001"),
        minimum_episodes=200,
    )

    assert decision.status is PromotionStatus.INSUFFICIENT_EVIDENCE
    assert decision.reason_codes == ("EPISODE_FLOOR_NOT_MET",)


def test_promotion_rejects_non_positive_adverse_result() -> None:
    scenario = CostScenario("base", Decimal("0"), Decimal("1"), Decimal("0"), Decimal("0"))
    base = evaluate_signals((1,), (Decimal("0.01"),), (Decimal("1"),), scenario)
    adverse = evaluate_signals((1,), (Decimal("-0.01"),), (Decimal("1"),), scenario)

    decision = assess_shadow_promotion(
        base=base,
        adverse=adverse,
        base_lower_confidence_bound=Decimal("0.001"),
        minimum_episodes=1,
    )

    assert decision.status is PromotionStatus.REJECTED
    assert "ADVERSE_NET_NON_POSITIVE" in decision.reason_codes
