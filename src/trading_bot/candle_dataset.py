"""Deterministic partitioned Parquet publication for normalized candles."""

import hashlib
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from trading_bot.candles import Candle
from trading_bot.canonical import canonical_json, content_sha256


class DatasetPublicationError(RuntimeError):
    """Raised when a candle dataset cannot be published immutably."""


@dataclass(frozen=True, slots=True)
class CandleDatasetQuality:
    row_count: int
    duplicate_rows: int
    missing_intervals: int
    unconfirmed_rows_excluded: int
    min_open_time_ns: int
    max_open_time_ns: int


@dataclass(frozen=True, slots=True)
class CandleDatasetArtifact:
    dataset_root: Path
    manifest_path: Path
    quality_report_path: Path
    root_hash: str
    quality: CandleDatasetQuality


_SCHEMA = pa.schema(
    [
        pa.field("venue", pa.string(), nullable=False),
        pa.field("instrument_id", pa.string(), nullable=False),
        pa.field("open_time_ns", pa.int64(), nullable=False),
        pa.field("close_time_ns", pa.int64(), nullable=False),
        pa.field("available_time_ns", pa.int64(), nullable=False),
        pa.field("interval_ns", pa.int64(), nullable=False),
        pa.field("open", pa.string(), nullable=False),
        pa.field("high", pa.string(), nullable=False),
        pa.field("low", pa.string(), nullable=False),
        pa.field("close", pa.string(), nullable=False),
        pa.field("base_volume", pa.string(), nullable=False),
        pa.field("quote_volume", pa.string(), nullable=False),
        pa.field("trade_count", pa.int64(), nullable=True),
        pa.field("source_payload_hash", pa.string(), nullable=False),
    ]
)


def publish_candle_dataset(
    candles: tuple[Candle, ...],
    *,
    output_directory: Path,
    raw_source_hashes: tuple[str, ...],
) -> CandleDatasetArtifact:
    if output_directory.exists():
        raise DatasetPublicationError("dataset output already exists and is immutable")
    if not raw_source_hashes or any(len(value) != 64 for value in raw_source_hashes):
        raise DatasetPublicationError("raw source hashes are required")
    admitted, duplicates, unconfirmed = _admit_rows(candles)
    if not admitted:
        raise DatasetPublicationError("dataset has no confirmed rows")
    quality = CandleDatasetQuality(
        row_count=len(admitted),
        duplicate_rows=duplicates,
        missing_intervals=_count_missing_intervals(admitted),
        unconfirmed_rows_excluded=unconfirmed,
        min_open_time_ns=min(item.open_time_ns for item in admitted),
        max_open_time_ns=max(item.open_time_ns for item in admitted),
    )

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_directory.name}.", dir=output_directory.parent)
    )
    try:
        files = _write_partitions(temporary, admitted)
        quality_record = _quality_record(quality)
        quality_bytes = canonical_json(quality_record)
        (temporary / "quality-report.json").write_bytes(quality_bytes)
        material: dict[str, object] = {
            "dataset_name": "candles",
            "schema_version": "1.0.0",
            "schema_hash": content_sha256(str(_SCHEMA)),
            "decimal_encoding": "canonical_string",
            "compression": "zstd",
            "row_count": quality.row_count,
            "files": files,
            "quality_report_hash": hashlib.sha256(quality_bytes).hexdigest(),
            "raw_source_hashes": list(raw_source_hashes),
        }
        root_hash = content_sha256(material)
        manifest = dict(material)
        manifest["root_hash"] = root_hash
        (temporary / "dataset-manifest.json").write_bytes(canonical_json(manifest))
        temporary.replace(output_directory)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return CandleDatasetArtifact(
        dataset_root=output_directory,
        manifest_path=output_directory / "dataset-manifest.json",
        quality_report_path=output_directory / "quality-report.json",
        root_hash=root_hash,
        quality=quality,
    )


def _admit_rows(candles: tuple[Candle, ...]) -> tuple[list[Candle], int, int]:
    seen: dict[tuple[str, str, int], Candle] = {}
    duplicates = 0
    unconfirmed = 0
    for candle in candles:
        identity = (candle.venue, candle.instrument_id, candle.open_time_ns)
        existing = seen.get(identity)
        if existing is not None:
            if existing != candle:
                raise DatasetPublicationError("conflicting duplicate candle identity")
            duplicates += 1
            continue
        seen[identity] = candle
        if not candle.confirmed:
            unconfirmed += 1
    admitted = sorted(
        (item for item in seen.values() if item.confirmed),
        key=lambda item: (item.venue, item.instrument_id, item.open_time_ns),
    )
    return admitted, duplicates, unconfirmed


def _count_missing_intervals(candles: list[Candle]) -> int:
    groups: dict[tuple[str, str], list[Candle]] = {}
    for candle in candles:
        groups.setdefault((candle.venue, candle.instrument_id), []).append(candle)
    missing = 0
    for rows in groups.values():
        rows.sort(key=lambda item: item.open_time_ns)
        for previous, current in pairwise(rows):
            difference = current.open_time_ns - previous.open_time_ns
            if difference > previous.interval_ns:
                missing += difference // previous.interval_ns - 1
            elif difference != previous.interval_ns:
                raise DatasetPublicationError("candle intervals are inconsistent")
    return missing


def _write_partitions(root: Path, candles: list[Candle]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[Candle]] = {}
    for candle in candles:
        date = (
            datetime.fromtimestamp(candle.open_time_ns / 1_000_000_000, tz=UTC).date().isoformat()
        )
        groups.setdefault((candle.venue, candle.instrument_id, date), []).append(candle)

    files: list[dict[str, object]] = []
    for (venue, instrument, date), rows in sorted(groups.items()):
        relative = Path(
            f"dataset=candles/venue={venue}/instrument={instrument}/date={date}/part-00000.parquet"
        )
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist([_candle_record(item) for item in rows], schema=_SCHEMA)
        pq.write_table(
            table,
            target,
            compression="zstd",
            use_dictionary=False,
            write_statistics=True,
            data_page_version="1.0",
        )
        encoded = target.read_bytes()
        files.append(
            {
                "relative_path": relative.as_posix(),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "row_count": len(rows),
            }
        )
    return files


def _candle_record(candle: Candle) -> dict[str, object]:
    return {
        "venue": candle.venue,
        "instrument_id": candle.instrument_id,
        "open_time_ns": candle.open_time_ns,
        "close_time_ns": candle.close_time_ns,
        "available_time_ns": candle.available_time_ns,
        "interval_ns": candle.interval_ns,
        "open": str(candle.open),
        "high": str(candle.high),
        "low": str(candle.low),
        "close": str(candle.close),
        "base_volume": str(candle.base_volume),
        "quote_volume": str(candle.quote_volume),
        "trade_count": candle.trade_count,
        "source_payload_hash": candle.source_payload_hash,
    }


def _quality_record(quality: CandleDatasetQuality) -> dict[str, object]:
    return {
        "row_count": quality.row_count,
        "duplicate_rows": quality.duplicate_rows,
        "missing_intervals": quality.missing_intervals,
        "unconfirmed_rows_excluded": quality.unconfirmed_rows_excluded,
        "min_open_time_ns": quality.min_open_time_ns,
        "max_open_time_ns": quality.max_open_time_ns,
    }
