from decimal import Decimal
from pathlib import Path

import pytest

from trading_bot.binance_importer import (
    BinanceAggTradeImporter,
    ImportValidationError,
    InputLimitError,
)

SELL_TAKER_LINE = (
    '{"a":26129,"p":"101.250","q":"4.500","nq":"4.500",'
    '"f":27781,"l":27783,"T":1700000000000,"m":true}'
)


def test_imports_binance_aggregate_trade_with_taker_side(tmp_path: Path) -> None:
    source = tmp_path / "agg-trades.jsonl"
    source.write_text(f"{SELL_TAKER_LINE}\n", encoding="utf-8", newline="\n")
    importer = BinanceAggTradeImporter(
        instrument_symbol="BTCUSDT",
        max_input_bytes=10_000,
        availability_lag_ns=5_000_000,
    )

    batch = importer.import_jsonl(source, downloaded_at_ns=1_800_000_000_000_000_000)

    assert len(batch.events) == 1
    assert batch.events[0].instrument_id == "BTCUSDT-PERP.BINANCE"
    assert batch.events[0].event_time_ns == 1_700_000_000_000_000_000
    assert batch.events[0].available_time_ns == 1_700_000_000_005_000_000
    assert batch.events[0].price == Decimal("101.250")
    assert batch.events[0].quantity == Decimal("4.500")
    assert batch.events[0].side == "sell"
    assert batch.events[0].sequence == 26129


def test_maps_buyer_taker_to_buy_side(tmp_path: Path) -> None:
    source = tmp_path / "agg-trades.jsonl"
    source.write_text(
        f"{SELL_TAKER_LINE.replace('true', 'false')}\n",
        encoding="utf-8",
        newline="\n",
    )
    importer = BinanceAggTradeImporter(
        instrument_symbol="BTCUSDT",
        max_input_bytes=10_000,
        availability_lag_ns=0,
    )

    batch = importer.import_jsonl(source, downloaded_at_ns=1_800_000_000_000_000_000)

    assert batch.events[0].side == "buy"


def test_rejects_numeric_price_instead_of_documented_string(tmp_path: Path) -> None:
    source = tmp_path / "agg-trades.jsonl"
    source.write_text(
        f"{SELL_TAKER_LINE.replace(chr(34) + '101.250' + chr(34), '101.250')}\n",
        encoding="utf-8",
        newline="\n",
    )
    importer = BinanceAggTradeImporter(
        instrument_symbol="BTCUSDT",
        max_input_bytes=10_000,
        availability_lag_ns=0,
    )

    with pytest.raises(ImportValidationError, match="line 1"):
        importer.import_jsonl(source, downloaded_at_ns=1_800_000_000_000_000_000)


def test_rejects_binance_input_larger_than_limit(tmp_path: Path) -> None:
    source = tmp_path / "agg-trades.jsonl"
    source.write_bytes(b"x" * 11)
    importer = BinanceAggTradeImporter(
        instrument_symbol="BTCUSDT",
        max_input_bytes=10,
        availability_lag_ns=0,
    )

    with pytest.raises(InputLimitError, match="input limit"):
        importer.import_jsonl(source, downloaded_at_ns=1_800_000_000_000_000_000)


def test_removes_identical_binance_aggregate_trade_duplicates(tmp_path: Path) -> None:
    source = tmp_path / "agg-trades.jsonl"
    source.write_text(
        f"{SELL_TAKER_LINE}\n{SELL_TAKER_LINE}\n",
        encoding="utf-8",
        newline="\n",
    )
    importer = BinanceAggTradeImporter(
        instrument_symbol="BTCUSDT",
        max_input_bytes=10_000,
        availability_lag_ns=0,
    )

    batch = importer.import_jsonl(source, downloaded_at_ns=1_800_000_000_000_000_000)

    assert len(batch.events) == 1
    assert batch.report.input_rows == 2
    assert batch.report.duplicate_rows == 1
