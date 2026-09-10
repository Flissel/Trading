import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from trading_bot.panel_dataset import (
    PanelCandleRow,
    PanelDatasetError,
    PanelFundingRow,
    admit_candles,
    find_missing_days,
    publish_panel_dataset,
    verify_panel_dataset,
)

DAY_NS = 86_400_000_000_000
SOURCE_HASHES = ("a" * 64,)


def candle(symbol: str, index: int, *, close: str = "100") -> PanelCandleRow:
    open_time_ns = index * DAY_NS
    return PanelCandleRow(
        venue="BINANCE_UM",
        instrument_id=symbol,
        open_time_ns=open_time_ns,
        close_time_ns=open_time_ns + DAY_NS - 1_000_000,
        available_time_ns=open_time_ns + DAY_NS,
        interval_ns=DAY_NS,
        open=Decimal("100"),
        high=Decimal("101"),
        low=Decimal("99"),
        close=Decimal(close),
        base_volume=Decimal("10"),
        quote_volume=Decimal("1000"),
        trade_count=5,
        source_payload_hash="b" * 64,
    )


def funding(symbol: str, index: int) -> PanelFundingRow:
    return PanelFundingRow(
        venue="BINANCE_UM",
        instrument_id=symbol,
        calc_time_ns=index * DAY_NS,
        funding_interval_hours=8,
        rate=Decimal("0.0001"),
    )


def test_publish_writes_partitions_and_per_instrument_quality(tmp_path: Path) -> None:
    rows = tuple(candle("BTCUSDT", index) for index in range(3))
    rows += tuple(candle("ETHUSDT", index) for index in range(2))
    artifact = publish_panel_dataset(
        rows,
        (funding("BTCUSDT", 0),),
        output_directory=tmp_path / "dataset",
        raw_source_hashes=SOURCE_HASHES,
    )
    assert (
        artifact.dataset_root
        / "dataset=daily_candles"
        / "venue=BINANCE_UM"
        / "instrument=BTCUSDT"
        / "part-00000.parquet"
    ).is_file()
    quality = json.loads(artifact.quality_report_path.read_text(encoding="utf-8"))
    assert quality["candle_row_count"] == 5
    btc = next(item for item in quality["instruments"] if item["instrument_id"] == "BTCUSDT")
    assert btc["row_count"] == 3
    assert btc["missing_days"] == 0
    assert verify_panel_dataset(artifact.dataset_root) == (True, ())


def test_missing_day_is_counted(tmp_path: Path) -> None:
    rows = (candle("BTCUSDT", 0), candle("BTCUSDT", 1), candle("BTCUSDT", 3))
    artifact = publish_panel_dataset(
        rows, (), output_directory=tmp_path / "dataset", raw_source_hashes=SOURCE_HASHES
    )
    quality = json.loads(artifact.quality_report_path.read_text(encoding="utf-8"))
    btc = quality["instruments"][0]
    assert btc["missing_days"] == 1


def test_find_missing_days_agrees_with_the_quality_reports_missing_days_count(
    tmp_path: Path,
) -> None:
    rows = (
        candle("BTCUSDT", 0),
        candle("BTCUSDT", 1),
        candle("BTCUSDT", 4),  # days 2 and 3 missing
        candle("ETHUSDT", 0),
        candle("ETHUSDT", 2),  # day 1 missing
    )
    admitted, _ = admit_candles(rows)
    missing = find_missing_days(admitted)
    artifact = publish_panel_dataset(
        rows, (), output_directory=tmp_path / "dataset", raw_source_hashes=SOURCE_HASHES
    )
    quality = json.loads(artifact.quality_report_path.read_text(encoding="utf-8"))
    for item in quality["instruments"]:
        assert len(missing[item["instrument_id"]]) == item["missing_days"]
    assert missing["BTCUSDT"] == (2 * DAY_NS, 3 * DAY_NS)
    assert missing["ETHUSDT"] == (1 * DAY_NS,)


def test_find_missing_days_is_span_relative_not_calendar_relative() -> None:
    # BTCUSDT's own span (index 5..7) has no interior gap, and neither does
    # ETHUSDT's (index 0..2) -- even though BTCUSDT does not exist at index
    # 0-4 and ETHUSDT does not exist at index 3+. Neither instrument's
    # absence outside the other's span may count as a missing day for it.
    rows = (
        candle("BTCUSDT", 5),
        candle("BTCUSDT", 6),
        candle("BTCUSDT", 7),
        candle("ETHUSDT", 0),
        candle("ETHUSDT", 1),
        candle("ETHUSDT", 2),
    )
    admitted, _ = admit_candles(rows)
    assert find_missing_days(admitted) == {"BTCUSDT": (), "ETHUSDT": ()}


def test_admit_candles_rejects_an_off_lattice_open_time() -> None:
    # `_instrument_quality` and `find_missing_days` both walk an instrument's
    # span in fixed DAY_NS steps from its first to its last admitted row; an
    # open_time_ns even one nanosecond off that lattice would make the two
    # silently disagree about how many days are missing. Real Binance klines
    # are always day-aligned -- this only guards a future source that isn't.
    off_lattice = replace(candle("BTCUSDT", 0), open_time_ns=1)
    with pytest.raises(PanelDatasetError):
        admit_candles((off_lattice,))


def test_duplicate_rows_are_dropped(tmp_path: Path) -> None:
    rows = (candle("BTCUSDT", 0), candle("BTCUSDT", 0), candle("BTCUSDT", 1))
    artifact = publish_panel_dataset(
        rows, (), output_directory=tmp_path / "dataset", raw_source_hashes=SOURCE_HASHES
    )
    quality = json.loads(artifact.quality_report_path.read_text(encoding="utf-8"))
    assert quality["duplicate_rows"] == 1
    assert quality["candle_row_count"] == 2


def test_conflicting_candle_rows_raise(tmp_path: Path) -> None:
    rows = (candle("BTCUSDT", 0), candle("BTCUSDT", 0, close="101"))
    with pytest.raises(PanelDatasetError):
        publish_panel_dataset(
            rows, (), output_directory=tmp_path / "dataset", raw_source_hashes=SOURCE_HASHES
        )


def test_conflicting_funding_rows_raise(tmp_path: Path) -> None:
    conflicting = PanelFundingRow(
        venue="BINANCE_UM",
        instrument_id="BTCUSDT",
        calc_time_ns=0,
        funding_interval_hours=8,
        rate=Decimal("0.0002"),
    )
    with pytest.raises(PanelDatasetError):
        publish_panel_dataset(
            (candle("BTCUSDT", 0),),
            (funding("BTCUSDT", 0), conflicting),
            output_directory=tmp_path / "dataset",
            raw_source_hashes=SOURCE_HASHES,
        )


def test_identical_funding_repeat_is_counted_as_duplicate(tmp_path: Path) -> None:
    artifact = publish_panel_dataset(
        (candle("BTCUSDT", 0),),
        (funding("BTCUSDT", 0), funding("BTCUSDT", 0)),
        output_directory=tmp_path / "dataset",
        raw_source_hashes=SOURCE_HASHES,
    )
    quality = json.loads(artifact.quality_report_path.read_text(encoding="utf-8"))
    assert quality["funding_duplicate_rows"] == 1


def test_output_is_immutable(tmp_path: Path) -> None:
    target = tmp_path / "dataset"
    publish_panel_dataset(
        (candle("BTCUSDT", 0),), (), output_directory=target, raw_source_hashes=SOURCE_HASHES
    )
    with pytest.raises(PanelDatasetError):
        publish_panel_dataset(
            (candle("BTCUSDT", 0),), (), output_directory=target, raw_source_hashes=SOURCE_HASHES
        )


def test_tampering_is_detected(tmp_path: Path) -> None:
    artifact = publish_panel_dataset(
        (candle("BTCUSDT", 0),),
        (),
        output_directory=tmp_path / "d",
        raw_source_hashes=SOURCE_HASHES,
    )
    path = (
        artifact.dataset_root
        / "dataset=daily_candles"
        / "venue=BINANCE_UM"
        / "instrument=BTCUSDT"
        / "part-00000.parquet"
    )
    path.write_bytes(path.read_bytes() + b"x")
    valid, errors = verify_panel_dataset(artifact.dataset_root)
    assert not valid
    assert any(error.startswith("PARQUET_HASH_MISMATCH") for error in errors)


def test_unlisted_parquet_file_is_detected(tmp_path: Path) -> None:
    # Simulates copying a partition to a new instrument directory: the manifest
    # still hashes and verifies every file it lists, but `load_panel_bars` globs the
    # whole tree, so an unlisted file would silently be picked up by the reader while
    # verification kept reporting valid. `verify_panel_dataset` must walk the dataset
    # root and flag any parquet file the manifest does not know about.
    artifact = publish_panel_dataset(
        (candle("BTCUSDT", 0),),
        (),
        output_directory=tmp_path / "d",
        raw_source_hashes=SOURCE_HASHES,
    )
    original = (
        artifact.dataset_root
        / "dataset=daily_candles"
        / "venue=BINANCE_UM"
        / "instrument=BTCUSDT"
        / "part-00000.parquet"
    )
    smuggled_dir = (
        artifact.dataset_root
        / "dataset=daily_candles"
        / "venue=BINANCE_UM"
        / "instrument=ETHUSDT"
    )
    smuggled_dir.mkdir(parents=True)
    (smuggled_dir / "part-00000.parquet").write_bytes(original.read_bytes())

    valid, errors = verify_panel_dataset(artifact.dataset_root)
    assert not valid
    assert any(error.startswith("UNLISTED_PARQUET_FILE") for error in errors)
