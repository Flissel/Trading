import hashlib
from decimal import Decimal
from pathlib import Path

import pytest

from trading_bot.okx_importer import (
    ImportValidationError,
    InputLimitError,
    OkxTradeImporter,
)

TRADE_LINE = (
    '{"instId":"BTC-USDT-SWAP","tradeId":"42","px":"101.25",'
    '"sz":"0.500","side":"buy","source":"0","ts":"1700000000000"}'
)


def test_imports_okx_trade_without_losing_precision(tmp_path: Path) -> None:
    source = tmp_path / "trades.jsonl"
    source.write_text(f"{TRADE_LINE}\n", encoding="utf-8", newline="\n")
    importer = OkxTradeImporter(max_input_bytes=10_000, availability_lag_ns=5_000_000)

    batch = importer.import_jsonl(source, downloaded_at_ns=1_800_000_000_000_000_000)

    assert len(batch.events) == 1
    assert batch.events[0].instrument_id == "BTC-USDT-SWAP.OKX"
    assert batch.events[0].event_time_ns == 1_700_000_000_000_000_000
    assert batch.events[0].available_time_ns == 1_700_000_000_005_000_000
    assert batch.events[0].received_time_ns == 1_800_000_000_000_000_000
    assert batch.events[0].price == Decimal("101.25")
    assert batch.events[0].quantity == Decimal("0.500")
    assert batch.events[0].side == "buy"
    assert batch.report.raw_sha256 == hashlib.sha256(f"{TRADE_LINE}\n".encode()).hexdigest()


def test_counts_and_removes_identical_duplicate_trades(tmp_path: Path) -> None:
    source = tmp_path / "trades.jsonl"
    source.write_text(f"{TRADE_LINE}\n{TRADE_LINE}\n", encoding="utf-8", newline="\n")
    importer = OkxTradeImporter(max_input_bytes=10_000, availability_lag_ns=0)

    batch = importer.import_jsonl(source, downloaded_at_ns=1_800_000_000_000_000_000)

    assert len(batch.events) == 1
    assert batch.report.input_rows == 2
    assert batch.report.duplicate_rows == 1


def test_rejects_input_larger_than_configured_limit(tmp_path: Path) -> None:
    source = tmp_path / "trades.jsonl"
    source.write_bytes(b"x" * 11)
    importer = OkxTradeImporter(max_input_bytes=10, availability_lag_ns=0)

    with pytest.raises(InputLimitError, match="input limit"):
        importer.import_jsonl(source, downloaded_at_ns=1_800_000_000_000_000_000)


def test_rejects_entire_batch_when_one_row_is_invalid(tmp_path: Path) -> None:
    invalid_line = (
        '{"instId":"BTC-USDT-SWAP","tradeId":"43","px":101.25,'
        '"sz":"0.500","side":"sell","source":"0","ts":"1700000000001"}'
    )
    source = tmp_path / "trades.jsonl"
    source.write_text(
        f"{TRADE_LINE}\n{invalid_line}\n",
        encoding="utf-8",
        newline="\n",
    )
    importer = OkxTradeImporter(max_input_bytes=10_000, availability_lag_ns=0)

    with pytest.raises(ImportValidationError, match="line 2"):
        importer.import_jsonl(source, downloaded_at_ns=1_800_000_000_000_000_000)
