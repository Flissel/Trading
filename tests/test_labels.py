from decimal import Decimal

from trading_bot.labels import PricePoint, build_return_labels


def test_label_engine_uses_first_price_at_or_after_horizon() -> None:
    points = [
        PricePoint("a", event_time_ns=0, available_time_ns=5, price=Decimal("100")),
        PricePoint("b", event_time_ns=9, available_time_ns=11, price=Decimal("105")),
        PricePoint("c", event_time_ns=12, available_time_ns=14, price=Decimal("110")),
    ]

    labels = build_return_labels(points, horizon_ns=10, round_trip_cost_bps=Decimal("20"))

    assert len(labels) == 1
    assert labels[0].source_event_id == "a"
    assert labels[0].outcome_event_id == "c"
    assert labels[0].gross_return == Decimal("0.1")
    assert labels[0].net_return == Decimal("0.098")
    assert labels[0].direction == 1
    assert labels[0].label_start_time_ns == 0
    assert labels[0].label_end_time_ns == 12
    assert labels[0].label_available_time_ns == 14


def test_label_engine_records_overlapping_outcome_intervals() -> None:
    points = [
        PricePoint("a", 0, 0, Decimal("100")),
        PricePoint("b", 5, 5, Decimal("101")),
        PricePoint("c", 10, 10, Decimal("102")),
        PricePoint("d", 15, 15, Decimal("103")),
    ]

    labels = build_return_labels(points, horizon_ns=10, round_trip_cost_bps=Decimal("0"))

    assert [label.overlaps_next for label in labels] == [True, False]


def test_cost_adjusted_direction_holds_inside_cost_band() -> None:
    points = [
        PricePoint("a", 0, 0, Decimal("100")),
        PricePoint("b", 10, 10, Decimal("100.1")),
    ]

    label = build_return_labels(points, horizon_ns=10, round_trip_cost_bps=Decimal("20"))[0]

    assert label.gross_return == Decimal("0.001")
    assert label.net_return == Decimal("-0.001")
    assert label.direction == 0
