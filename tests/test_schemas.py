from decimal import Decimal
from uuid import UUID

import pytest
from pydantic import ValidationError

from trading_bot.runtime import RuntimeMode
from trading_bot.schemas import CaptureMode, EventEnvelope, MarketEvent, MarketEventType

SOURCE_HASH = "a" * 64


def make_trade(**overrides: object) -> MarketEvent:
    values: dict[str, object] = {
        "venue": "OKX",
        "instrument_id": "BTC-USDT-SWAP.OKX",
        "event_type": MarketEventType.TRADE,
        "event_time_ns": 1_000,
        "available_time_ns": 1_010,
        "received_time_ns": 1_010,
        "sequence": 42,
        "source_record_id": "okx:trade:42",
        "source_payload_hash": SOURCE_HASH,
        "quality_flags": (),
        "price": Decimal("101.25"),
        "quantity": Decimal("0.500"),
    }
    values.update(overrides)
    return MarketEvent.model_validate(values)


def test_market_event_preserves_decimal_values() -> None:
    event = make_trade()

    assert event.price == Decimal("101.25")
    assert event.quantity == Decimal("0.500")


def test_market_event_rejects_binary_float_price() -> None:
    with pytest.raises(ValidationError, match="price"):
        make_trade(price=101.25)


def test_market_event_rejects_availability_before_receipt() -> None:
    with pytest.raises(ValidationError, match="available_time_ns"):
        make_trade(available_time_ns=1_009)


def test_historical_event_allows_availability_before_download_receipt() -> None:
    event = make_trade(
        capture_mode=CaptureMode.HISTORICAL,
        available_time_ns=1_001,
        received_time_ns=2_000,
    )

    assert event.available_time_ns == 1_001
    assert event.received_time_ns == 2_000


def test_market_event_rejects_invalid_source_hash() -> None:
    with pytest.raises(ValidationError, match="source_payload_hash"):
        make_trade(source_payload_hash="not-a-sha256")


def test_envelope_hash_verifies_real_payload() -> None:
    event = make_trade()
    envelope = EventEnvelope.create(
        schema_name="market_event",
        schema_version="1.0.0",
        mode=RuntimeMode.BACKTEST,
        producer_id="okx_normalizer",
        producer_version="0.1.0",
        created_at_ns=2_000,
        correlation_id=UUID("00000000-0000-0000-0000-000000000001"),
        causation_id=None,
        run_id=UUID("00000000-0000-0000-0000-000000000002"),
        payload=event,
    )

    assert envelope.verify_payload_hash() is True
    assert len(envelope.payload_hash) == 64


def test_envelope_detects_payload_tampering() -> None:
    event = make_trade()
    envelope = EventEnvelope.create(
        schema_name="market_event",
        schema_version="1.0.0",
        mode=RuntimeMode.BACKTEST,
        producer_id="okx_normalizer",
        producer_version="0.1.0",
        created_at_ns=2_000,
        correlation_id=UUID("00000000-0000-0000-0000-000000000001"),
        causation_id=None,
        run_id=UUID("00000000-0000-0000-0000-000000000002"),
        payload=event,
    )

    tampered = envelope.model_copy(
        update={"payload": event.model_copy(update={"price": Decimal("999.00")})}
    )

    assert tampered.verify_payload_hash() is False
