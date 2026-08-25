"""Strict normalization for public OKX and Binance candle payloads."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from trading_bot.canonical import content_sha256


class CandlePayloadError(ValueError):
    """Raised when a venue candle payload violates its frozen contract."""


@dataclass(frozen=True, slots=True)
class Candle:
    venue: str
    instrument_id: str
    open_time_ns: int
    close_time_ns: int
    available_time_ns: int
    interval_ns: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    base_volume: Decimal
    quote_volume: Decimal
    trade_count: int | None
    confirmed: bool
    source_payload_hash: str

    def __post_init__(self) -> None:
        if not self.venue or not self.instrument_id:
            raise CandlePayloadError("candle identity must not be empty")
        if self.open_time_ns < 0 or self.close_time_ns < self.open_time_ns:
            raise CandlePayloadError("candle time range is invalid")
        if self.available_time_ns < self.close_time_ns or self.interval_ns <= 0:
            raise CandlePayloadError("candle availability is invalid")
        prices = (self.open, self.high, self.low, self.close)
        if any(not value.is_finite() or value <= 0 for value in prices):
            raise CandlePayloadError("OHLC prices must be positive finite decimals")
        if self.low > min(self.open, self.close) or self.high < max(self.open, self.close):
            raise CandlePayloadError("OHLC range does not contain open and close")
        if self.low > self.high:
            raise CandlePayloadError("OHLC range is inverted")
        if self.base_volume < 0 or self.quote_volume < 0:
            raise CandlePayloadError("volumes must be non-negative")
        if self.trade_count is not None and self.trade_count < 0:
            raise CandlePayloadError("trade_count must be non-negative")


def parse_okx_candles(
    payload: Mapping[str, object],
    *,
    instrument_id: str,
    interval_ns: int,
    availability_lag_ns: int,
) -> tuple[Candle, ...]:
    code = _string(payload.get("code"), field="code")
    if code != "0":
        raise CandlePayloadError(f"OKX API error {code}")
    rows = _sequence(payload.get("data"), field="data")
    candles: list[Candle] = []
    for index, raw in enumerate(rows):
        row = _sequence(raw, field=f"data[{index}]")
        if len(row) != 9:
            raise CandlePayloadError(f"data[{index}] must contain nine fields")
        open_time_ns = _integer_string(row[0], field="timestamp") * 1_000_000
        close_time_ns = open_time_ns + interval_ns - 1
        candles.append(
            Candle(
                venue="OKX",
                instrument_id=instrument_id,
                open_time_ns=open_time_ns,
                close_time_ns=close_time_ns,
                available_time_ns=close_time_ns + availability_lag_ns,
                interval_ns=interval_ns,
                open=_decimal_string(row[1], field="open"),
                high=_decimal_string(row[2], field="high"),
                low=_decimal_string(row[3], field="low"),
                close=_decimal_string(row[4], field="close"),
                base_volume=_decimal_string(row[6], field="base_volume"),
                quote_volume=_decimal_string(row[7], field="quote_volume"),
                trade_count=None,
                confirmed=_string(row[8], field="confirm") == "1",
                source_payload_hash=content_sha256(list(row)),
            )
        )
    return tuple(sorted(candles, key=lambda candle: candle.open_time_ns))


def parse_binance_klines(
    payload: Sequence[object],
    *,
    instrument_id: str,
    interval_ns: int,
    availability_lag_ns: int,
    confirmed_through_ns: int | None = None,
) -> tuple[Candle, ...]:
    rows = _sequence(payload, field="payload")
    candles: list[Candle] = []
    for index, raw in enumerate(rows):
        row = _sequence(raw, field=f"payload[{index}]")
        if len(row) != 12:
            raise CandlePayloadError(f"payload[{index}] must contain twelve fields")
        close_time_ns = _integer(row[6], field="close_time") * 1_000_000
        candles.append(
            Candle(
                venue="BINANCE",
                instrument_id=instrument_id,
                open_time_ns=_integer(row[0], field="open_time") * 1_000_000,
                close_time_ns=close_time_ns,
                available_time_ns=close_time_ns + availability_lag_ns,
                interval_ns=interval_ns,
                open=_decimal_string(row[1], field="open"),
                high=_decimal_string(row[2], field="high"),
                low=_decimal_string(row[3], field="low"),
                close=_decimal_string(row[4], field="close"),
                base_volume=_decimal_string(row[5], field="base_volume"),
                quote_volume=_decimal_string(row[7], field="quote_volume"),
                trade_count=_integer(row[8], field="trade_count"),
                confirmed=(confirmed_through_ns is None or close_time_ns <= confirmed_through_ns),
                source_payload_hash=content_sha256(list(row)),
            )
        )
    return tuple(sorted(candles, key=lambda candle: candle.open_time_ns))


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise CandlePayloadError(f"{field} must be an array")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise CandlePayloadError(f"{field} must be a string")
    return value


def _integer(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise CandlePayloadError(f"{field} must be an integer")
    return value


def _integer_string(value: object, *, field: str) -> int:
    text = _string(value, field=field)
    try:
        return int(text)
    except ValueError as error:
        raise CandlePayloadError(f"{field} must contain an integer") from error


def _decimal_string(value: object, *, field: str) -> Decimal:
    text = _string(value, field=field)
    try:
        number = Decimal(text)
    except InvalidOperation as error:
        raise CandlePayloadError(f"{field} must contain a decimal") from error
    if not number.is_finite():
        raise CandlePayloadError(f"{field} must contain a finite decimal")
    return number
