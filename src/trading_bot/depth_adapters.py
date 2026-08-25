"""Strict venue adapters for level-2 order-book payloads."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from trading_bot.order_book import (
    BookLevel,
    BookUpdate,
    OrderBookDelta,
    OrderBookSnapshot,
)

type DepthUpdate = OrderBookSnapshot | OrderBookDelta


class DepthPayloadError(ValueError):
    """Raised when a venue payload cannot be safely normalized."""


@dataclass(frozen=True, slots=True)
class AdaptedDepthMessage:
    """A venue message with canonical identity and integer event time."""

    instrument_id: str
    event_time_ns: int
    update: DepthUpdate


class OkxDepthAdapter:
    """Normalize OKX ``books`` channel snapshots and updates."""

    def __init__(self, venue_symbol: str, instrument_id: str) -> None:
        self._venue_symbol = venue_symbol
        self._instrument_id = instrument_id

    def parse(self, payload: Mapping[str, object]) -> AdaptedDepthMessage:
        argument = _mapping(payload.get("arg"), field_name="arg")
        instrument = _string(argument.get("instId"), field_name="arg.instId")
        if instrument != self._venue_symbol:
            raise DepthPayloadError(
                f"unexpected OKX instrument: expected {self._venue_symbol}, got {instrument}"
            )
        channel = _string(argument.get("channel"), field_name="arg.channel")
        if channel != "books":
            raise DepthPayloadError(f"unsupported OKX channel: {channel}")

        action = _string(payload.get("action"), field_name="action")
        data = _sequence(payload.get("data"), field_name="data")
        if len(data) != 1:
            raise DepthPayloadError("OKX depth payload must contain exactly one data item")
        item = _mapping(data[0], field_name="data[0]")
        sequence = _integer(item.get("seqId"), field_name="data[0].seqId")
        event_time_ns = _integer_string(item.get("ts"), field_name="data[0].ts") * 1_000_000

        if action == "snapshot":
            if _integer(item.get("prevSeqId"), field_name="data[0].prevSeqId") != -1:
                raise DepthPayloadError("OKX snapshot prevSeqId must be -1")
            update: DepthUpdate = OrderBookSnapshot(
                sequence=sequence,
                bids=_snapshot_levels(item.get("bids"), field_name="data[0].bids"),
                asks=_snapshot_levels(item.get("asks"), field_name="data[0].asks"),
            )
        elif action == "update":
            update = OrderBookDelta(
                previous_sequence=_integer(item.get("prevSeqId"), field_name="data[0].prevSeqId"),
                sequence=sequence,
                bids=_delta_levels(item.get("bids"), field_name="data[0].bids"),
                asks=_delta_levels(item.get("asks"), field_name="data[0].asks"),
            )
        else:
            raise DepthPayloadError(f"unsupported OKX depth action: {action}")

        return AdaptedDepthMessage(self._instrument_id, event_time_ns, update)


class BinanceDepthAdapter:
    """Normalize Binance USD-M REST snapshots and WebSocket diffs."""

    def __init__(self, venue_symbol: str, instrument_id: str) -> None:
        self._venue_symbol = venue_symbol
        self._instrument_id = instrument_id

    def parse_snapshot(self, payload: Mapping[str, object]) -> AdaptedDepthMessage:
        update = OrderBookSnapshot(
            sequence=_integer(payload.get("lastUpdateId"), field_name="lastUpdateId"),
            bids=_snapshot_levels(payload.get("bids"), field_name="bids"),
            asks=_snapshot_levels(payload.get("asks"), field_name="asks"),
        )
        return AdaptedDepthMessage(
            self._instrument_id,
            _integer(payload.get("T"), field_name="T") * 1_000_000,
            update,
        )

    def parse_update(self, payload: Mapping[str, object]) -> AdaptedDepthMessage:
        event_type = _string(payload.get("e"), field_name="e")
        if event_type != "depthUpdate":
            raise DepthPayloadError(f"unsupported Binance event type: {event_type}")
        symbol = _string(payload.get("s"), field_name="s")
        if symbol != self._venue_symbol:
            raise DepthPayloadError(
                f"unexpected Binance symbol: expected {self._venue_symbol}, got {symbol}"
            )

        first_sequence = _integer(payload.get("U"), field_name="U")
        sequence = _integer(payload.get("u"), field_name="u")
        if first_sequence > sequence:
            raise DepthPayloadError("Binance first update ID cannot exceed final update ID")
        update = OrderBookDelta(
            previous_sequence=_integer(payload.get("pu"), field_name="pu"),
            sequence=sequence,
            bids=_delta_levels(payload.get("b"), field_name="b"),
            asks=_delta_levels(payload.get("a"), field_name="a"),
        )
        return AdaptedDepthMessage(
            self._instrument_id,
            _integer(payload.get("T"), field_name="T") * 1_000_000,
            update,
        )


def _snapshot_levels(value: object, *, field_name: str) -> tuple[BookLevel, ...]:
    return tuple(
        BookLevel(price=price, quantity=quantity)
        for price, quantity in _level_values(value, field_name=field_name)
    )


def _delta_levels(value: object, *, field_name: str) -> tuple[BookUpdate, ...]:
    return tuple(
        BookUpdate(price=price, quantity=quantity)
        for price, quantity in _level_values(value, field_name=field_name)
    )


def _level_values(value: object, *, field_name: str) -> tuple[tuple[Decimal, Decimal], ...]:
    rows = _sequence(value, field_name=field_name)
    result: list[tuple[Decimal, Decimal]] = []
    for index, raw_row in enumerate(rows):
        row = _sequence(raw_row, field_name=f"{field_name}[{index}]")
        if len(row) < 2:
            raise DepthPayloadError(f"{field_name}[{index}] must contain price and quantity")
        price = _decimal_string(row[0], field_name="price")
        quantity = _decimal_string(row[1], field_name="quantity")
        result.append((price, quantity))
    return tuple(result)


def _mapping(value: object, *, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise DepthPayloadError(f"{field_name} must be an object with string keys")
    return value


def _sequence(value: object, *, field_name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise DepthPayloadError(f"{field_name} must be an array")
    return value


def _string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise DepthPayloadError(f"{field_name} must be a string")
    return value


def _integer(value: object, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise DepthPayloadError(f"{field_name} must be an integer")
    return value


def _integer_string(value: object, *, field_name: str) -> int:
    text = _string(value, field_name=field_name)
    try:
        return int(text)
    except ValueError as error:
        raise DepthPayloadError(f"{field_name} must contain an integer") from error


def _decimal_string(value: object, *, field_name: str) -> Decimal:
    text = _string(value, field_name=field_name)
    try:
        number = Decimal(text)
    except InvalidOperation as error:
        raise DepthPayloadError(f"{field_name} must contain a decimal") from error
    if not number.is_finite():
        raise DepthPayloadError(f"{field_name} must contain a finite decimal")
    return number
