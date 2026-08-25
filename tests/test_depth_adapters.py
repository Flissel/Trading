from decimal import Decimal

import pytest

from trading_bot.depth_adapters import (
    BinanceDepthAdapter,
    DepthPayloadError,
    OkxDepthAdapter,
)
from trading_bot.order_book import (
    BookLevel,
    BookStatus,
    BookUpdate,
    OrderBookDelta,
    OrderBookReconstructor,
    OrderBookSnapshot,
    SequenceGapError,
)


def test_okx_snapshot_is_mapped_to_canonical_depth_message() -> None:
    adapter = OkxDepthAdapter(
        venue_symbol="BTC-USDT-SWAP",
        instrument_id="BTC-USDT-SWAP.OKX",
    )

    message = adapter.parse(
        {
            "arg": {"channel": "books", "instId": "BTC-USDT-SWAP"},
            "action": "snapshot",
            "data": [
                {
                    "asks": [["60001.1", "1.25", "0", "2"]],
                    "bids": [["60000.9", "0.75", "0", "3"]],
                    "ts": "1720000000123",
                    "checksum": 0,
                    "prevSeqId": -1,
                    "seqId": 900,
                }
            ],
        }
    )

    assert message.instrument_id == "BTC-USDT-SWAP.OKX"
    assert message.event_time_ns == 1_720_000_000_123_000_000
    assert message.update == OrderBookSnapshot(
        sequence=900,
        bids=(BookLevel(Decimal("60000.9"), Decimal("0.75")),),
        asks=(BookLevel(Decimal("60001.1"), Decimal("1.25")),),
    )


def test_okx_delta_preserves_zero_quantity_and_sequence_link() -> None:
    adapter = OkxDepthAdapter("BTC-USDT-SWAP", "BTC-USDT-SWAP.OKX")

    message = adapter.parse(
        {
            "arg": {"channel": "books", "instId": "BTC-USDT-SWAP"},
            "action": "update",
            "data": [
                {
                    "asks": [["60001.1", "0", "0", "0"]],
                    "bids": [["60000.8", "2.5", "0", "1"]],
                    "ts": "1720000000223",
                    "checksum": 0,
                    "prevSeqId": 900,
                    "seqId": 901,
                }
            ],
        }
    )

    assert message.update == OrderBookDelta(
        previous_sequence=900,
        sequence=901,
        bids=(BookUpdate(Decimal("60000.8"), Decimal("2.5")),),
        asks=(BookUpdate(Decimal("60001.1"), Decimal("0")),),
    )


def test_okx_empty_sequence_heartbeat_is_a_safe_noop_delta() -> None:
    adapter = OkxDepthAdapter("BTC-USDT-SWAP", "BTC-USDT-SWAP.OKX")
    book = OrderBookReconstructor()
    book.apply_snapshot(
        OrderBookSnapshot(
            sequence=42,
            bids=(BookLevel(Decimal("99"), Decimal("1")),),
            asks=(BookLevel(Decimal("101"), Decimal("1")),),
        )
    )

    message = adapter.parse(
        {
            "arg": {"channel": "books", "instId": "BTC-USDT-SWAP"},
            "action": "update",
            "data": [
                {
                    "asks": [],
                    "bids": [],
                    "ts": "1720000000323",
                    "checksum": 0,
                    "prevSeqId": 42,
                    "seqId": 42,
                }
            ],
        }
    )
    assert isinstance(message.update, OrderBookDelta)
    book.apply_delta(message.update)

    assert book.status is BookStatus.VALID
    assert book.sequence == 42


def test_okx_rejects_binary_float_prices_and_wrong_instrument() -> None:
    adapter = OkxDepthAdapter("BTC-USDT-SWAP", "BTC-USDT-SWAP.OKX")
    base_payload: dict[str, object] = {
        "arg": {"channel": "books", "instId": "ETH-USDT-SWAP"},
        "action": "snapshot",
        "data": [
            {
                "asks": [[60001.1, "1", "0", "1"]],
                "bids": [["60000", "1", "0", "1"]],
                "ts": "1720000000123",
                "prevSeqId": -1,
                "seqId": 1,
            }
        ],
    }

    with pytest.raises(DepthPayloadError, match="unexpected OKX instrument"):
        adapter.parse(base_payload)

    arg = base_payload["arg"]
    assert isinstance(arg, dict)
    arg["instId"] = "BTC-USDT-SWAP"
    with pytest.raises(DepthPayloadError, match="price must be a string"):
        adapter.parse(base_payload)


def test_binance_snapshot_is_mapped_from_rest_depth() -> None:
    adapter = BinanceDepthAdapter("BTCUSDT", "BTC-USDT-SWAP.BINANCE")

    message = adapter.parse_snapshot(
        {
            "lastUpdateId": 1027024,
            "E": 1589436922972,
            "T": 1589436922959,
            "bids": [["60000.0", "4.5"]],
            "asks": [["60001.0", "2.0"]],
        }
    )

    assert message.instrument_id == "BTC-USDT-SWAP.BINANCE"
    assert message.event_time_ns == 1_589_436_922_959_000_000
    assert message.update == OrderBookSnapshot(
        sequence=1027024,
        bids=(BookLevel(Decimal("60000.0"), Decimal("4.5")),),
        asks=(BookLevel(Decimal("60001.0"), Decimal("2.0")),),
    )


def test_binance_diff_maps_previous_and_final_update_ids() -> None:
    adapter = BinanceDepthAdapter("BTCUSDT", "BTC-USDT-SWAP.BINANCE")

    message = adapter.parse_update(
        {
            "e": "depthUpdate",
            "E": 1720000000123,
            "T": 1720000000120,
            "s": "BTCUSDT",
            "U": 1027025,
            "u": 1027030,
            "pu": 1027024,
            "b": [["60000.0", "0"]],
            "a": [["60001.5", "3.25"]],
        }
    )

    assert message.event_time_ns == 1_720_000_000_120_000_000
    assert message.update == OrderBookDelta(
        previous_sequence=1027024,
        sequence=1027030,
        bids=(BookUpdate(Decimal("60000.0"), Decimal("0")),),
        asks=(BookUpdate(Decimal("60001.5"), Decimal("3.25")),),
    )


def test_binance_adapter_exposes_sequence_gap_to_reconstructor() -> None:
    adapter = BinanceDepthAdapter("BTCUSDT", "BTC-USDT-SWAP.BINANCE")
    book = OrderBookReconstructor()
    snapshot = adapter.parse_snapshot(
        {
            "lastUpdateId": 10,
            "E": 100,
            "T": 99,
            "bids": [["99", "1"]],
            "asks": [["101", "1"]],
        }
    )
    assert isinstance(snapshot.update, OrderBookSnapshot)
    book.apply_snapshot(snapshot.update)
    update = adapter.parse_update(
        {
            "e": "depthUpdate",
            "E": 102,
            "T": 101,
            "s": "BTCUSDT",
            "U": 12,
            "u": 12,
            "pu": 11,
            "b": [],
            "a": [],
        }
    )
    assert isinstance(update.update, OrderBookDelta)

    with pytest.raises(SequenceGapError):
        book.apply_delta(update.update)
    assert book.status is BookStatus.INVALID


def test_binance_rejects_wrong_symbol_and_binary_float_quantity() -> None:
    adapter = BinanceDepthAdapter("BTCUSDT", "BTC-USDT-SWAP.BINANCE")
    payload: dict[str, object] = {
        "e": "depthUpdate",
        "E": 102,
        "T": 101,
        "s": "ETHUSDT",
        "U": 11,
        "u": 11,
        "pu": 10,
        "b": [["99", 1.0]],
        "a": [],
    }

    with pytest.raises(DepthPayloadError, match="unexpected Binance symbol"):
        adapter.parse_update(payload)

    payload["s"] = "BTCUSDT"
    with pytest.raises(DepthPayloadError, match="quantity must be a string"):
        adapter.parse_update(payload)
