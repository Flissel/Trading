from dataclasses import replace
from decimal import Decimal

from trading_bot.bar_research import (
    ResearchBar,
    ResearchStatus,
    build_bar_samples,
    evaluate_development_samples,
)
from trading_bot.strategy import CostScenario


def bar(
    source_id: str,
    venue: str,
    open_time_ns: int,
    close: str,
    *,
    available_time_ns: int,
) -> ResearchBar:
    price = Decimal(close)
    return ResearchBar(
        source_id=source_id,
        venue=venue,
        open_time_ns=open_time_ns,
        available_time_ns=available_time_ns,
        close=price,
        high=price + Decimal("1"),
        low=price - Decimal("1"),
        base_volume=Decimal("10"),
    )


def test_bar_features_do_not_change_when_future_label_changes() -> None:
    primary = (
        bar("o0", "OKX", 0, "100", available_time_ns=10),
        bar("o1", "OKX", 10, "110", available_time_ns=20),
        bar("o2", "OKX", 20, "121", available_time_ns=30),
    )
    reference = (bar("b1", "BINANCE", 10, "100", available_time_ns=20),)

    original = build_bar_samples(primary, reference)[0]
    changed = build_bar_samples(
        (primary[0], primary[1], replace(primary[2], close=Decimal("130"))), reference
    )[0]

    assert original.observed_return == Decimal("0.1")
    assert original.cross_venue_basis == Decimal("0.1")
    assert original.input_source_ids == ("o0", "o1", "b1")
    assert original.label_available_time_ns == 30
    assert original.observed_return == changed.observed_return
    assert original.cross_venue_basis == changed.cross_venue_basis
    assert original.forward_return == Decimal("0.1")
    assert changed.forward_return == Decimal("0.181818181818181818181818182")


def test_reference_bar_available_after_decision_is_not_used() -> None:
    primary = (
        bar("o0", "OKX", 0, "100", available_time_ns=10),
        bar("o1", "OKX", 10, "110", available_time_ns=20),
        bar("o2", "OKX", 20, "111", available_time_ns=30),
    )
    late_reference = (bar("b1", "BINANCE", 10, "100", available_time_ns=21),)

    sample = build_bar_samples(primary, late_reference)[0]

    assert sample.cross_venue_basis is None
    assert sample.input_source_ids == ("o0", "o1")


def test_bar_samples_bind_identity_and_availability_to_prediction_horizon() -> None:
    primary = tuple(
        bar(f"o{i}", "OKX", i * 10, str(100 + 10 * i), available_time_ns=i * 10 + 9)
        for i in range(6)
    )

    one_bar = build_bar_samples(primary, ())[0]
    two_bar = build_bar_samples(primary, (), horizon_bars=2)[0]

    assert one_bar.sample_id == "OKX:10"
    assert two_bar.sample_id == "OKX:10:h2"
    assert two_bar.decision_time_ns == 19
    assert two_bar.label_available_time_ns == 39
    assert two_bar.forward_return == Decimal("0.181818181818181818181818182")
    assert two_bar.input_source_ids == ("o0", "o1")


def test_development_evaluation_uses_only_chronological_oos_tail() -> None:
    primary = tuple(
        bar(f"o{i}", "OKX", i * 10, str(100 + i), available_time_ns=i * 10 + 9) for i in range(11)
    )
    samples = build_bar_samples(primary, ())
    report = evaluate_development_samples(
        samples,
        oos_fraction=Decimal("0.25"),
        minimum_train_samples=4,
        random_seed=7,
        base_costs=CostScenario("base", Decimal("1"), Decimal("1"), Decimal("1"), Decimal("0")),
        adverse_costs=CostScenario(
            "adverse", Decimal("2"), Decimal("2"), Decimal("2"), Decimal("1")
        ),
        assumed_spread_bps=Decimal("2"),
    )

    assert report.train_sample_count == 6
    assert report.oos_sample_count == 3
    assert report.oos_start_time_ns == samples[6].decision_time_ns
    assert {result.baseline_name for result in report.baselines} == {
        "no_trade",
        "momentum",
        "mean_reversion",
        "random",
    }
    no_trade = next(item for item in report.baselines if item.baseline_name == "no_trade")
    assert no_trade.base.total_net_return == 0
    assert report.status is ResearchStatus.DEVELOPMENT_ONLY
    assert report.reason_codes == (
        "HISTORY_TOO_SHORT_FOR_PROTOCOL",
        "NO_REGISTERED_WALK_FORWARD_FOLDS",
        "EPISODE_FLOOR_NOT_MET",
    )
