"""Bounded importer for Binance USD-M aggregate trades."""

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from trading_bot.canonical import content_sha256
from trading_bot.okx_importer import ImportBatch, ImportQualityReport
from trading_bot.schemas import CaptureMode, MarketEvent, MarketEventType, TradeSide


class ImportValidationError(ValueError):
    """Raised when a Binance source row violates the import contract."""


class InputLimitError(ValueError):
    """Raised when Binance source input exceeds the configured bound."""


class _BinanceAggregateTrade(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    aggregate_trade_id: int = Field(alias="a")
    price: str = Field(alias="p")
    quantity: str = Field(alias="q")
    normal_quantity: str | None = Field(default=None, alias="nq")
    first_trade_id: int = Field(alias="f")
    last_trade_id: int = Field(alias="l")
    trade_time_ms: int = Field(alias="T")
    buyer_is_maker: bool = Field(alias="m")


@dataclass(frozen=True, slots=True)
class BinanceAggTradeImporter:
    """Convert bounded Binance aggregate trades into canonical events."""

    instrument_symbol: str
    max_input_bytes: int
    availability_lag_ns: int

    def import_jsonl(self, source: Path, *, downloaded_at_ns: int) -> ImportBatch:
        input_size = source.stat().st_size
        if input_size > self.max_input_bytes:
            raise InputLimitError(
                f"source size {input_size} exceeds input limit {self.max_input_bytes}"
            )

        raw_bytes = source.read_bytes()
        if len(raw_bytes) > self.max_input_bytes:
            raise InputLimitError(
                f"source size {len(raw_bytes)} exceeds input limit {self.max_input_bytes}"
            )

        events: list[MarketEvent] = []
        seen: dict[int, str] = {}
        duplicate_rows = 0
        input_rows = 0

        for line_number, line in enumerate(raw_bytes.decode("utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            input_rows += 1
            aggregate, raw_payload = self._parse_line(line, line_number)
            record_hash = content_sha256(raw_payload)
            existing_hash = seen.get(aggregate.aggregate_trade_id)
            if existing_hash is not None:
                if existing_hash != record_hash:
                    raise ImportValidationError(
                        f"line {line_number}: conflicting aggregate trade identity"
                    )
                duplicate_rows += 1
                continue
            seen[aggregate.aggregate_trade_id] = record_hash

            event_time_ns = aggregate.trade_time_ms * 1_000_000
            events.append(
                MarketEvent(
                    venue="BINANCE",
                    instrument_id=f"{self.instrument_symbol}-PERP.BINANCE",
                    event_type=MarketEventType.TRADE,
                    capture_mode=CaptureMode.HISTORICAL,
                    event_time_ns=event_time_ns,
                    available_time_ns=event_time_ns + self.availability_lag_ns,
                    received_time_ns=downloaded_at_ns,
                    sequence=aggregate.aggregate_trade_id,
                    source_record_id=(
                        f"binance:aggTrade:{self.instrument_symbol}:{aggregate.aggregate_trade_id}"
                    ),
                    source_payload_hash=record_hash,
                    quality_flags=(),
                    price=Decimal(aggregate.price),
                    quantity=Decimal(aggregate.quantity),
                    side=TradeSide.SELL if aggregate.buyer_is_maker else TradeSide.BUY,
                )
            )

        return ImportBatch(
            events=tuple(events),
            report=ImportQualityReport(
                raw_sha256=hashlib.sha256(raw_bytes).hexdigest(),
                input_rows=input_rows,
                duplicate_rows=duplicate_rows,
            ),
        )

    @staticmethod
    def _parse_line(
        line: str,
        line_number: int,
    ) -> tuple[_BinanceAggregateTrade, dict[str, object]]:
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError("aggregate trade row must be an object")
            aggregate = _BinanceAggregateTrade.model_validate(payload)
            return aggregate, payload
        except (json.JSONDecodeError, TypeError, ValidationError) as error:
            raise ImportValidationError(
                f"line {line_number}: invalid Binance aggregate trade row"
            ) from error
