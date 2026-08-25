import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from trading_bot.candle_dataset import DatasetPublicationError, publish_candle_dataset
from trading_bot.candles import Candle


def candle(
    venue: str,
    instrument: str,
    open_time_ns: int,
    *,
    close: str = "101",
    confirmed: bool = True,
    source_hash: str | None = None,
) -> Candle:
    return Candle(
        venue=venue,
        instrument_id=instrument,
        open_time_ns=open_time_ns,
        close_time_ns=open_time_ns + 9,
        available_time_ns=open_time_ns + 10,
        interval_ns=10,
        open=Decimal("100"),
        high=Decimal("102"),
        low=Decimal("99"),
        close=Decimal(close),
        base_volume=Decimal("5"),
        quote_volume=Decimal("500"),
        trade_count=3,
        confirmed=confirmed,
        source_payload_hash=source_hash or f"{open_time_ns:064x}"[-64:],
    )


def test_dataset_reports_duplicates_gaps_and_unconfirmed_exclusions(tmp_path: Path) -> None:
    first = candle("OKX", "BTC-USDT-SWAP.OKX", 0)
    rows = (
        first,
        first,
        candle("OKX", "BTC-USDT-SWAP.OKX", 10),
        candle("OKX", "BTC-USDT-SWAP.OKX", 30),
        candle("OKX", "BTC-USDT-SWAP.OKX", 40, confirmed=False),
    )

    artifact = publish_candle_dataset(
        rows,
        output_directory=tmp_path / "dataset",
        raw_source_hashes=("a" * 64,),
    )

    assert artifact.quality.row_count == 3
    assert artifact.quality.duplicate_rows == 1
    assert artifact.quality.missing_intervals == 1
    assert artifact.quality.unconfirmed_rows_excluded == 1
    assert artifact.manifest_path.exists()
    assert artifact.quality_report_path.exists()


def test_parquet_partitions_are_directly_queryable_with_duckdb(tmp_path: Path) -> None:
    import duckdb

    rows = (
        candle("OKX", "BTC-USDT-SWAP.OKX", 0),
        candle("BINANCE", "BTC-USDT-SWAP.BINANCE", 0),
    )
    artifact = publish_candle_dataset(
        rows,
        output_directory=tmp_path / "dataset",
        raw_source_hashes=("a" * 64, "b" * 64),
    )

    parquet_glob = str(artifact.dataset_root / "**" / "*.parquet")
    result = duckdb.sql(
        "SELECT count(*), count(DISTINCT venue) FROM read_parquet(?)",
        params=[parquet_glob],
    ).fetchone()

    assert result == (2, 2)


def test_semantic_manifest_and_parquet_hashes_are_reproducible(tmp_path: Path) -> None:
    rows = (
        candle("OKX", "BTC-USDT-SWAP.OKX", 0),
        candle("OKX", "BTC-USDT-SWAP.OKX", 10),
    )

    first = publish_candle_dataset(
        rows, output_directory=tmp_path / "one", raw_source_hashes=("a" * 64,)
    )
    second = publish_candle_dataset(
        rows, output_directory=tmp_path / "two", raw_source_hashes=("a" * 64,)
    )
    first_manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    second_manifest = json.loads(second.manifest_path.read_text(encoding="utf-8"))

    assert first.root_hash == second.root_hash
    assert first_manifest["files"] == second_manifest["files"]


def test_conflicting_duplicate_identity_fails_closed(tmp_path: Path) -> None:
    first = candle("OKX", "BTC-USDT-SWAP.OKX", 0, source_hash="a" * 64)
    conflicting = replace(first, close=Decimal("100.5"), source_payload_hash="b" * 64)

    with pytest.raises(DatasetPublicationError, match="conflicting duplicate"):
        publish_candle_dataset(
            (first, conflicting),
            output_directory=tmp_path / "dataset",
            raw_source_hashes=("c" * 64,),
        )
