"""Immutable Parquet publication for the daily perpetual panel."""

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from trading_bot.canonical import canonical_json, content_sha256

DAY_NS = 86_400_000_000_000


class PanelDatasetError(RuntimeError):
    """Raised when a panel dataset cannot be published or verified."""


@dataclass(frozen=True, slots=True)
class PanelCandleRow:
    venue: str
    instrument_id: str
    open_time_ns: int
    close_time_ns: int
    available_time_ns: int
    interval_ns: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    base_volume: Decimal
    quote_volume: Decimal
    trade_count: int | None
    source_payload_hash: str


@dataclass(frozen=True, slots=True)
class PanelFundingRow:
    venue: str
    instrument_id: str
    calc_time_ns: int
    funding_interval_hours: int
    rate: Decimal


@dataclass(frozen=True, slots=True)
class InstrumentQuality:
    instrument_id: str
    row_count: int
    first_open_time_ns: int
    last_open_time_ns: int
    missing_days: int


@dataclass(frozen=True, slots=True)
class PanelDatasetArtifact:
    dataset_root: Path
    manifest_path: Path
    quality_report_path: Path
    root_hash: str
    instruments: tuple[InstrumentQuality, ...]


_CANDLE_SCHEMA = pa.schema(
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

_FUNDING_SCHEMA = pa.schema(
    [
        pa.field("venue", pa.string(), nullable=False),
        pa.field("instrument_id", pa.string(), nullable=False),
        pa.field("calc_time_ns", pa.int64(), nullable=False),
        pa.field("funding_interval_hours", pa.int64(), nullable=False),
        pa.field("rate", pa.string(), nullable=False),
    ]
)


def publish_panel_dataset(
    candles: tuple[PanelCandleRow, ...],
    funding: tuple[PanelFundingRow, ...],
    *,
    output_directory: Path,
    raw_source_hashes: tuple[str, ...],
) -> PanelDatasetArtifact:
    if output_directory.exists():
        raise PanelDatasetError("panel dataset already exists and is immutable")
    if not raw_source_hashes or any(len(value) != 64 for value in raw_source_hashes):
        raise PanelDatasetError("raw source hashes are required")
    admitted, duplicate_rows = _admit_candles(candles)
    if not admitted:
        raise PanelDatasetError("panel dataset has no candle rows")
    admitted_funding, funding_duplicate_rows = _admit_funding(funding)
    instruments = _instrument_quality(admitted)

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_directory.name}.", dir=output_directory.parent)
    )
    try:
        files = _write_candle_partitions(temporary, admitted)
        files += _write_funding_partitions(temporary, admitted_funding)
        quality_record: dict[str, object] = {
            "candle_row_count": len(admitted),
            "funding_row_count": len(admitted_funding),
            "duplicate_rows": duplicate_rows,
            "funding_duplicate_rows": funding_duplicate_rows,
            "instrument_count": len(instruments),
            "instruments": [
                {
                    "instrument_id": item.instrument_id,
                    "row_count": item.row_count,
                    "first_open_time_ns": item.first_open_time_ns,
                    "last_open_time_ns": item.last_open_time_ns,
                    "missing_days": item.missing_days,
                }
                for item in instruments
            ],
        }
        quality_bytes = canonical_json(quality_record)
        (temporary / "quality-report.json").write_bytes(quality_bytes)
        material: dict[str, object] = {
            "dataset_name": "usdt_perp_panel",
            "schema_version": "1.0.0",
            "candle_schema_hash": content_sha256(str(_CANDLE_SCHEMA)),
            "funding_schema_hash": content_sha256(str(_FUNDING_SCHEMA)),
            "decimal_encoding": "canonical_string",
            "compression": "zstd",
            "candle_row_count": len(admitted),
            "funding_row_count": len(admitted_funding),
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

    return PanelDatasetArtifact(
        dataset_root=output_directory,
        manifest_path=output_directory / "dataset-manifest.json",
        quality_report_path=output_directory / "quality-report.json",
        root_hash=root_hash,
        instruments=instruments,
    )


def verify_panel_dataset(dataset_root: Path) -> tuple[bool, tuple[str, ...]]:
    errors: list[str] = []
    try:
        manifest = json.loads(
            (dataset_root / "dataset-manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False, ("MANIFEST_UNREADABLE",)
    if not isinstance(manifest, dict):
        return False, ("MANIFEST_STRUCTURE_INVALID",)
    recorded_root = manifest.get("root_hash")
    material = {key: value for key, value in manifest.items() if key != "root_hash"}
    if content_sha256(material) != recorded_root:
        errors.append("DATASET_ROOT_HASH_MISMATCH")
    try:
        quality_bytes = (dataset_root / "quality-report.json").read_bytes()
    except OSError:
        return False, tuple([*errors, "QUALITY_REPORT_UNREADABLE"])
    if hashlib.sha256(quality_bytes).hexdigest() != manifest.get("quality_report_hash"):
        errors.append("QUALITY_REPORT_HASH_MISMATCH")
    files = manifest.get("files")
    if not isinstance(files, list):
        return False, tuple([*errors, "MANIFEST_STRUCTURE_INVALID"])
    for entry in files:
        if not isinstance(entry, dict):
            errors.append("MANIFEST_STRUCTURE_INVALID")
            continue
        relative = str(entry.get("relative_path"))
        path = dataset_root / relative
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            errors.append(f"PARQUET_MISSING:{relative}")
            continue
        if digest != entry.get("sha256"):
            errors.append(f"PARQUET_HASH_MISMATCH:{relative}")
    return not errors, tuple(errors)


def _admit_candles(
    candles: tuple[PanelCandleRow, ...],
) -> tuple[tuple[PanelCandleRow, ...], int]:
    seen: dict[tuple[str, str, int], PanelCandleRow] = {}
    duplicates = 0
    for row in candles:
        if row.interval_ns != DAY_NS:
            raise PanelDatasetError("panel candles must be daily")
        key = (row.venue, row.instrument_id, row.open_time_ns)
        existing = seen.get(key)
        if existing is not None:
            if existing != row:
                raise PanelDatasetError(
                    "conflicting duplicate candle identity for "
                    f"{row.instrument_id} at open_time_ns={row.open_time_ns}"
                )
            duplicates += 1
            continue
        seen[key] = row
    ordered = sorted(seen.values(), key=lambda item: (item.instrument_id, item.open_time_ns))
    return tuple(ordered), duplicates


def _admit_funding(
    funding: tuple[PanelFundingRow, ...],
) -> tuple[tuple[PanelFundingRow, ...], int]:
    seen: dict[tuple[str, str, int], PanelFundingRow] = {}
    duplicates = 0
    for row in funding:
        key = (row.venue, row.instrument_id, row.calc_time_ns)
        existing = seen.get(key)
        if existing is not None:
            if existing != row:
                raise PanelDatasetError(
                    "conflicting duplicate funding identity for "
                    f"{row.instrument_id} at calc_time_ns={row.calc_time_ns}"
                )
            duplicates += 1
            continue
        seen[key] = row
    ordered = sorted(seen.values(), key=lambda item: (item.instrument_id, item.calc_time_ns))
    return tuple(ordered), duplicates


def _instrument_quality(rows: tuple[PanelCandleRow, ...]) -> tuple[InstrumentQuality, ...]:
    grouped: dict[str, list[PanelCandleRow]] = {}
    for row in rows:
        grouped.setdefault(row.instrument_id, []).append(row)
    quality: list[InstrumentQuality] = []
    for instrument_id in sorted(grouped):
        series = sorted(grouped[instrument_id], key=lambda item: item.open_time_ns)
        first = series[0].open_time_ns
        last = series[-1].open_time_ns
        expected = (last - first) // DAY_NS + 1
        quality.append(
            InstrumentQuality(
                instrument_id=instrument_id,
                row_count=len(series),
                first_open_time_ns=first,
                last_open_time_ns=last,
                missing_days=expected - len(series),
            )
        )
    return tuple(quality)


def _write_candle_partitions(
    root: Path, rows: tuple[PanelCandleRow, ...]
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[PanelCandleRow]] = {}
    for row in rows:
        grouped.setdefault((row.venue, row.instrument_id), []).append(row)
    files: list[dict[str, object]] = []
    for venue, instrument_id in sorted(grouped):
        series = grouped[(venue, instrument_id)]
        table = pa.Table.from_pydict(
            {
                "venue": [item.venue for item in series],
                "instrument_id": [item.instrument_id for item in series],
                "open_time_ns": [item.open_time_ns for item in series],
                "close_time_ns": [item.close_time_ns for item in series],
                "available_time_ns": [item.available_time_ns for item in series],
                "interval_ns": [item.interval_ns for item in series],
                "open": [str(item.open) for item in series],
                "high": [str(item.high) for item in series],
                "low": [str(item.low) for item in series],
                "close": [str(item.close) for item in series],
                "base_volume": [str(item.base_volume) for item in series],
                "quote_volume": [str(item.quote_volume) for item in series],
                "trade_count": [item.trade_count for item in series],
                "source_payload_hash": [item.source_payload_hash for item in series],
            },
            schema=_CANDLE_SCHEMA,
        )
        relative = (
            f"dataset=daily_candles/venue={venue}/instrument={instrument_id}/part-00000.parquet"
        )
        files.append(_write_table(root, relative, table, len(series)))
    return files


def _write_funding_partitions(
    root: Path, rows: tuple[PanelFundingRow, ...]
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[PanelFundingRow]] = {}
    for row in rows:
        grouped.setdefault((row.venue, row.instrument_id), []).append(row)
    files: list[dict[str, object]] = []
    for venue, instrument_id in sorted(grouped):
        series = grouped[(venue, instrument_id)]
        table = pa.Table.from_pydict(
            {
                "venue": [item.venue for item in series],
                "instrument_id": [item.instrument_id for item in series],
                "calc_time_ns": [item.calc_time_ns for item in series],
                "funding_interval_hours": [item.funding_interval_hours for item in series],
                "rate": [str(item.rate) for item in series],
            },
            schema=_FUNDING_SCHEMA,
        )
        relative = f"dataset=funding/venue={venue}/instrument={instrument_id}/part-00000.parquet"
        files.append(_write_table(root, relative, table, len(series)))
    return files


def _write_table(root: Path, relative: str, table: pa.Table, row_count: int) -> dict[str, object]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"relative_path": relative, "sha256": digest, "row_count": row_count}
