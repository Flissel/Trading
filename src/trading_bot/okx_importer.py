"""Bounded importer for OKX public trade records."""

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from trading_bot.canonical import content_sha256
from trading_bot.schemas import CaptureMode, MarketEvent, MarketEventType, TradeSide


class InputLimitError(ValueError):
    """Raised when source input exceeds the configured bound."""


class ImportValidationError(ValueError):
    """Raised when any source row violates the import contract."""


@dataclass(frozen=True, slots=True)
class ImportQualityReport:
    """Quality summary for one bounded import batch."""

    raw_sha256: str
    input_rows: int
    duplicate_rows: int


@dataclass(frozen=True, slots=True)
class ImportBatch:
    """Validated events and their quality report."""

    events: tuple[MarketEvent, ...]
    report: ImportQualityReport


class _OkxRawTrade(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    instId: str
    tradeId: str
    px: str
    sz: str
    side: Literal["buy", "sell"]
    source: str
    ts: str


@dataclass(frozen=True, slots=True)
class OkxTradeImporter:
    """Convert bounded OKX JSONL records into canonical market events."""

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
        seen: dict[tuple[str, str], str] = {}
        duplicate_rows = 0
        input_rows = 0

        for line_number, line in enumerate(raw_bytes.decode("utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            input_rows += 1
            raw_trade, raw_payload = self._parse_line(line, line_number)
            record_hash = content_sha256(raw_payload)
            identity = (raw_trade.instId, raw_trade.tradeId)
            existing_hash = seen.get(identity)
            if existing_hash is not None:
                if existing_hash != record_hash:
                    raise ImportValidationError(
                        f"line {line_number}: conflicting duplicate trade identity"
                    )
                duplicate_rows += 1
                continue
            seen[identity] = record_hash

            event_time_ns = self._milliseconds_to_nanoseconds(raw_trade.ts, line_number)
            events.append(
                MarketEvent(
                    venue="OKX",
                    instrument_id=f"{raw_trade.instId}.OKX",
                    event_type=MarketEventType.TRADE,
                    capture_mode=CaptureMode.HISTORICAL,
                    event_time_ns=event_time_ns,
                    available_time_ns=event_time_ns + self.availability_lag_ns,
                    received_time_ns=downloaded_at_ns,
                    sequence=None,
                    source_record_id=(f"okx:trade:{raw_trade.instId}:{raw_trade.tradeId}"),
                    source_payload_hash=record_hash,
                    quality_flags=(),
                    price=Decimal(raw_trade.px),
                    quantity=Decimal(raw_trade.sz),
                    side=TradeSide(raw_trade.side),
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
    def _parse_line(line: str, line_number: int) -> tuple[_OkxRawTrade, dict[str, object]]:
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError("trade row must be an object")
            raw_trade = _OkxRawTrade.model_validate(payload)
            return raw_trade, payload
        except (json.JSONDecodeError, TypeError, ValidationError) as error:
            raise ImportValidationError(f"line {line_number}: invalid OKX trade row") from error

    @staticmethod
    def _milliseconds_to_nanoseconds(timestamp: str, line_number: int) -> int:
        try:
            return int(timestamp) * 1_000_000
        except ValueError as error:
            raise ImportValidationError(
                f"line {line_number}: invalid millisecond timestamp"
            ) from error
