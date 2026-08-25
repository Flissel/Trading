"""Fail-closed level-2 order book reconstruction."""

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum


class BookUnavailableError(RuntimeError):
    """Raised when a book cannot safely provide reconstructed state."""


class SequenceGapError(BookUnavailableError):
    """Raised when a delta does not continue the active sequence."""


class BookStatus(StrEnum):
    """Current reconstruction state."""

    EMPTY = "empty"
    VALID = "valid"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class BookLevel:
    price: Decimal
    quantity: Decimal

    def __post_init__(self) -> None:
        _validate_decimal(self.price, field_name="price")
        _validate_decimal(self.quantity, field_name="quantity")
        if self.price <= 0:
            raise ValueError("price must be positive")
        if self.quantity <= 0:
            raise ValueError("book level quantity must be positive")


@dataclass(frozen=True, slots=True)
class BookUpdate:
    price: Decimal
    quantity: Decimal

    def __post_init__(self) -> None:
        _validate_decimal(self.price, field_name="price")
        _validate_decimal(self.quantity, field_name="quantity")
        if self.price <= 0:
            raise ValueError("price must be positive")
        if self.quantity < 0:
            raise ValueError("book update quantity must be non-negative")


@dataclass(frozen=True, slots=True)
class OrderBookSnapshot:
    sequence: int
    bids: tuple[BookLevel, ...]
    asks: tuple[BookLevel, ...]


@dataclass(frozen=True, slots=True)
class OrderBookDelta:
    previous_sequence: int
    sequence: int
    bids: tuple[BookUpdate, ...]
    asks: tuple[BookUpdate, ...]


class OrderBookReconstructor:
    """Reconstruct a book from one snapshot followed by contiguous deltas."""

    def __init__(self) -> None:
        self._status = BookStatus.EMPTY
        self.sequence: int | None = None
        self._bids: dict[Decimal, Decimal] = {}
        self._asks: dict[Decimal, Decimal] = {}

    @property
    def status(self) -> BookStatus:
        return self._status

    @property
    def best_bid(self) -> BookLevel | None:
        if self.status is not BookStatus.VALID or not self._bids:
            return None
        price = max(self._bids)
        return BookLevel(price=price, quantity=self._bids[price])

    @property
    def best_ask(self) -> BookLevel | None:
        if self.status is not BookStatus.VALID or not self._asks:
            return None
        price = min(self._asks)
        return BookLevel(price=price, quantity=self._asks[price])

    def apply_snapshot(self, snapshot: OrderBookSnapshot) -> None:
        bids = {item.price: item.quantity for item in snapshot.bids}
        asks = {item.price: item.quantity for item in snapshot.asks}
        if _is_crossed(bids, asks):
            self._invalidate()
            raise BookUnavailableError("snapshot contains a crossed order book")

        self._bids = bids
        self._asks = asks
        self.sequence = snapshot.sequence
        self._status = BookStatus.VALID

    def apply_delta(self, delta: OrderBookDelta) -> None:
        if self.status is not BookStatus.VALID or self.sequence is None:
            raise BookUnavailableError("a valid snapshot is required before deltas")

        if delta.previous_sequence != self.sequence:
            expected = self.sequence
            self._invalidate()
            raise SequenceGapError(
                f"sequence gap: expected {expected}, got {delta.previous_sequence}"
            )

        bids = self._bids.copy()
        asks = self._asks.copy()
        _apply_updates(bids, delta.bids)
        _apply_updates(asks, delta.asks)
        if _is_crossed(bids, asks):
            self._invalidate()
            raise BookUnavailableError("delta produced a crossed order book")

        self._bids = bids
        self._asks = asks
        self.sequence = delta.sequence

    def _invalidate(self) -> None:
        self._bids.clear()
        self._asks.clear()
        self.sequence = None
        self._status = BookStatus.INVALID


def _validate_decimal(value: object, *, field_name: str) -> None:
    if not isinstance(value, Decimal):
        raise TypeError(f"{field_name} must be a Decimal")


def _apply_updates(
    levels: dict[Decimal, Decimal],
    updates: tuple[BookUpdate, ...],
) -> None:
    for update in updates:
        if update.quantity == 0:
            levels.pop(update.price, None)
        else:
            levels[update.price] = update.quantity


def _is_crossed(bids: dict[Decimal, Decimal], asks: dict[Decimal, Decimal]) -> bool:
    return bool(bids and asks and max(bids) >= min(asks))
