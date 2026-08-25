from decimal import Decimal

import pytest

from trading_bot.features import MarketState, PointInTimeFeatureEngine


def state(
    event_id: str,
    *,
    decision_time_ns: int,
    available_time_ns: int | None = None,
    mid: str,
    bid: str,
    ask: str,
    bid_size: str = "1",
    ask_size: str = "1",
    buy_qty: str = "1",
    sell_qty: str = "1",
    funding: str | None = "0.0001",
    book_valid: bool = True,
    feed_age_ns: int = 10,
) -> MarketState:
    return MarketState(
        event_id=event_id,
        available_time_ns=decision_time_ns if available_time_ns is None else available_time_ns,
        decision_time_ns=decision_time_ns,
        mid_price=Decimal(mid),
        best_bid=Decimal(bid),
        best_ask=Decimal(ask),
        bid_size=Decimal(bid_size),
        ask_size=Decimal(ask_size),
        trade_buy_quantity=Decimal(buy_qty),
        trade_sell_quantity=Decimal(sell_qty),
        funding_rate=None if funding is None else Decimal(funding),
        book_valid=book_valid,
        feed_age_ns=feed_age_ns,
    )


def test_feature_engine_computes_hand_checked_market_features() -> None:
    engine = PointInTimeFeatureEngine(lookback=2, max_feed_age_ns=100)

    rows = engine.transform(
        [
            state("a", decision_time_ns=100, mid="100", bid="99", ask="101"),
            state(
                "b",
                decision_time_ns=200,
                mid="102",
                bid="101",
                ask="103",
                bid_size="3",
                ask_size="1",
                buy_qty="4",
                sell_qty="1",
            ),
        ]
    )

    assert rows[1].simple_return == Decimal("0.02")
    assert rows[1].spread_bps == Decimal("196.0784313725490196078431373")
    assert rows[1].book_imbalance == Decimal("0.5")
    assert rows[1].trade_flow_imbalance == Decimal("0.6")
    assert rows[1].funding_rate == Decimal("0.0001")
    assert rows[1].input_event_ids == ("a", "b")
    assert rows[1].available_time_ns == 200


def test_feature_engine_never_uses_an_observation_available_after_decision() -> None:
    engine = PointInTimeFeatureEngine(lookback=2, max_feed_age_ns=100)

    with pytest.raises(ValueError, match="available after decision"):
        engine.transform(
            [
                state(
                    "future",
                    decision_time_ns=100,
                    available_time_ns=101,
                    mid="100",
                    bid="99",
                    ask="101",
                )
            ]
        )


def test_invalid_or_stale_book_suppresses_book_features() -> None:
    engine = PointInTimeFeatureEngine(lookback=2, max_feed_age_ns=100)

    invalid, stale = engine.transform(
        [
            state(
                "invalid",
                decision_time_ns=100,
                mid="100",
                bid="99",
                ask="101",
                book_valid=False,
            ),
            state(
                "stale",
                decision_time_ns=200,
                mid="101",
                bid="100",
                ask="102",
                feed_age_ns=101,
            ),
        ]
    )

    assert invalid.spread_bps is None
    assert invalid.book_imbalance is None
    assert invalid.feed_healthy is False
    assert stale.spread_bps is None
    assert stale.book_imbalance is None
    assert stale.feed_healthy is False


def test_feature_engine_rejects_non_chronological_decisions() -> None:
    engine = PointInTimeFeatureEngine(lookback=2, max_feed_age_ns=100)

    with pytest.raises(ValueError, match="strictly increasing"):
        engine.transform(
            [
                state("later", decision_time_ns=200, mid="100", bid="99", ask="101"),
                state("earlier", decision_time_ns=100, mid="101", bid="100", ask="102"),
            ]
        )
