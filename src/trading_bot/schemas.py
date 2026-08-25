"""Canonical domain schemas."""

from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator

from trading_bot.canonical import content_sha256
from trading_bot.runtime import RuntimeMode

Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class MarketEventType(StrEnum):
    """Supported normalized market event families."""

    TRADE = "trade"
    QUOTE = "quote"
    BOOK_DELTA = "book_delta"
    BOOK_SNAPSHOT = "book_snapshot"
    MARK = "mark"
    INDEX = "index"
    FUNDING = "funding"


class CaptureMode(StrEnum):
    """How the source record entered the local dataset."""

    LIVE = "live"
    HISTORICAL = "historical"


class TradeSide(StrEnum):
    """Taker side reported for a public trade."""

    BUY = "buy"
    SELL = "sell"


class MarketEvent(BaseModel):
    """Normalized point-in-time market observation."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    venue: str
    instrument_id: str
    event_type: MarketEventType
    capture_mode: CaptureMode = CaptureMode.LIVE
    event_time_ns: int | None
    available_time_ns: int
    received_time_ns: int
    sequence: int | None
    source_record_id: str
    source_payload_hash: Sha256Hex
    quality_flags: tuple[str, ...]
    price: Decimal | None = None
    quantity: Decimal | None = None
    side: TradeSide | None = None

    @model_validator(mode="after")
    def validate_time_availability(self) -> Self:
        if self.capture_mode is CaptureMode.LIVE and self.available_time_ns < self.received_time_ns:
            raise ValueError("available_time_ns must not precede received_time_ns")
        return self


class EventEnvelope(BaseModel):
    """Versioned immutable envelope around a canonical market event."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    schema_name: str
    schema_version: str
    mode: RuntimeMode
    producer_id: str
    producer_version: str
    created_at_ns: int
    correlation_id: UUID
    causation_id: UUID | None
    run_id: UUID
    payload_hash: Sha256Hex
    payload: MarketEvent

    @classmethod
    def create(
        cls,
        *,
        schema_name: str,
        schema_version: str,
        mode: RuntimeMode,
        producer_id: str,
        producer_version: str,
        created_at_ns: int,
        correlation_id: UUID,
        causation_id: UUID | None,
        run_id: UUID,
        payload: MarketEvent,
    ) -> Self:
        payload_hash = content_sha256(payload.model_dump(mode="json"))
        return cls(
            schema_name=schema_name,
            schema_version=schema_version,
            mode=mode,
            producer_id=producer_id,
            producer_version=producer_version,
            created_at_ns=created_at_ns,
            correlation_id=correlation_id,
            causation_id=causation_id,
            run_id=run_id,
            payload_hash=payload_hash,
            payload=payload,
        )

    def verify_payload_hash(self) -> bool:
        """Return whether the stored hash matches the current payload."""
        return self.payload_hash == content_sha256(self.payload.model_dump(mode="json"))
