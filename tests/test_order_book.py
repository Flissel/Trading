from decimal import Decimal

import pytest

from trading_bot.order_book import (
    BookLevel,
    BookStatus,
    BookUnavailableError,
    BookUpdate,
    OrderBookDelta,
    OrderBookReconstructor,
    OrderBookSnapshot,
    SequenceGapError,
)


def level(price: str, quantity: str) -> BookLevel:
    return BookLevel(price=Decimal(price), quantity=Decimal(quantity))


def update(price: str, quantity: str) -> BookUpdate:
    return BookUpdate(price=Decimal(price), quantity=Decimal(quantity))


def snapshot(sequence: int = 100) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        sequence=sequence,
        bids=(level("100.0", "2.0"), level("99.5", "3.0")),
        asks=(level("100.5", "1.5"), level("101.0", "4.0")),
    )


def test_snapshot_exposes_best_bid_and_ask() -> None:
    book = OrderBookReconstructor()

    book.apply_snapshot(snapshot())

    assert book.status is BookStatus.VALID
    assert book.sequence == 100
    assert book.best_bid == level("100.0", "2.0")
    assert book.best_ask == level("100.5", "1.5")


def test_delta_updates_level_and_zero_quantity_removes_level() -> None:
    book = OrderBookReconstructor()
    book.apply_snapshot(snapshot())
    delta = OrderBookDelta(
        previous_sequence=100,
        sequence=101,
        bids=(update("100.0", "0"), update("100.2", "1.25")),
        asks=(update("100.5", "2.0"),),
    )

    book.apply_delta(delta)

    assert book.sequence == 101
    assert book.best_bid == level("100.2", "1.25")
    assert book.best_ask == level("100.5", "2.0")


def test_sequence_gap_invalidates_book_until_new_snapshot() -> None:
    book = OrderBookReconstructor()
    book.apply_snapshot(snapshot())
    gap = OrderBookDelta(
        previous_sequence=102,
        sequence=103,
        bids=(update("100.1", "1.0"),),
        asks=(),
    )

    with pytest.raises(SequenceGapError, match="expected 100"):
        book.apply_delta(gap)

    assert book.status is BookStatus.INVALID
    assert book.best_bid is None
    assert book.best_ask is None

    with pytest.raises(BookUnavailableError, match="snapshot"):
        book.apply_delta(
            OrderBookDelta(
                previous_sequence=103,
                sequence=104,
                bids=(),
                asks=(),
            )
        )

    book.apply_snapshot(snapshot(sequence=200))
    assert book.sequence == 200
    assert book.best_bid == level("100.0", "2.0")


def test_delta_before_snapshot_is_rejected() -> None:
    book = OrderBookReconstructor()

    with pytest.raises(BookUnavailableError, match="snapshot"):
        book.apply_delta(
            OrderBookDelta(
                previous_sequence=0,
                sequence=1,
                bids=(),
                asks=(),
            )
        )

    assert book.status is BookStatus.EMPTY


def test_crossed_snapshot_is_rejected_and_invalidates_book() -> None:
    book = OrderBookReconstructor()
    crossed = OrderBookSnapshot(
        sequence=100,
        bids=(level("101.0", "1.0"),),
        asks=(level("100.5", "1.0"),),
    )

    with pytest.raises(BookUnavailableError, match="crossed"):
        book.apply_snapshot(crossed)

    assert book.status is BookStatus.INVALID


def test_crossed_delta_is_rejected_and_invalidates_book() -> None:
    book = OrderBookReconstructor()
    book.apply_snapshot(snapshot())

    with pytest.raises(BookUnavailableError, match="crossed"):
        book.apply_delta(
            OrderBookDelta(
                previous_sequence=100,
                sequence=101,
                bids=(update("100.6", "1.0"),),
                asks=(),
            )
        )

    assert book.status is BookStatus.INVALID
    assert book.best_bid is None


def test_book_level_rejects_binary_float_price() -> None:
    with pytest.raises(TypeError, match="Decimal"):
        BookLevel(price=100.0, quantity=Decimal("1.0"))  # type: ignore[arg-type]


def test_book_update_rejects_negative_quantity() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        update("100.0", "-1.0")
