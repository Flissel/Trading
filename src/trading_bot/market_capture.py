"""Bounded public HTTPS capture for OKX and Binance candle fixtures."""

import hashlib
import json
import shutil
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from trading_bot.candle_dataset import (
    CandleDatasetArtifact,
    DatasetPublicationError,
    publish_candle_dataset,
)
from trading_bot.candles import Candle, CandlePayloadError, parse_binance_klines, parse_okx_candles
from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.storage import StoragePolicy

_INTERVAL_NS = 900_000_000_000
_HISTORICAL_AVAILABILITY_LAG_NS = 1_000_000_000
_MAX_HISTORY_BARS = 60_000
_ALLOWED_HOSTS = frozenset({"www.okx.com", "fapi.binance.com"})


class CaptureError(RuntimeError):
    """Raised when bounded public capture fails closed."""


@dataclass(frozen=True, slots=True)
class CapturedPayload:
    url: str
    raw_bytes: bytes
    received_time_ns: int


@dataclass(frozen=True, slots=True)
class CandleCaptureArtifact:
    capture_root: Path
    capture_manifest_path: Path
    dataset: CandleDatasetArtifact
    capture_root_hash: str


@dataclass(frozen=True, slots=True)
class CaptureVerification:
    valid: bool
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PublicJsonClient:
    max_response_bytes: int
    timeout_seconds: int

    def __post_init__(self) -> None:
        if self.max_response_bytes <= 0 or self.timeout_seconds <= 0:
            raise CaptureError("HTTP limits must be positive")

    def fetch(self, url: str) -> CapturedPayload:
        _validate_public_url(url)
        request = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "User-Agent": "hybrid-trading-research/0.1"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                final_url = response.geturl()
                _validate_public_url(final_url)
                raw = response.read(self.max_response_bytes + 1)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            raise CaptureError(f"public market-data request failed: {error}") from error
        if len(raw) > self.max_response_bytes:
            raise CaptureError("public market-data response exceeds byte limit")
        return CapturedPayload(url=final_url, raw_bytes=raw, received_time_ns=time.time_ns())


def build_okx_candles_url(*, limit: int, bar: str, after_ms: int | None = None) -> str:
    if not 1 <= limit <= 300:
        raise CaptureError("OKX candle limit must be between 1 and 300")
    parameters = {"instId": "BTC-USDT-SWAP", "bar": bar, "limit": str(limit)}
    if after_ms is not None:
        if after_ms < 0:
            raise CaptureError("OKX history cursor must be non-negative")
        parameters["after"] = str(after_ms)
    query = urllib.parse.urlencode(parameters)
    return f"https://www.okx.com/api/v5/market/history-candles?{query}"


def build_binance_klines_url(*, limit: int, interval: str, end_time_ms: int | None = None) -> str:
    if not 1 <= limit <= 1500:
        raise CaptureError("Binance kline limit must be between 1 and 1500")
    parameters = {"symbol": "BTCUSDT", "interval": interval, "limit": str(limit)}
    if end_time_ms is not None:
        if end_time_ms < 0:
            raise CaptureError("Binance history cursor must be non-negative")
        parameters["endTime"] = str(end_time_ms)
    query = urllib.parse.urlencode(parameters)
    return f"https://fapi.binance.com/fapi/v1/klines?{query}"


def fetch_history_payloads(
    fetch: Callable[[str], CapturedPayload],
    *,
    bars: int,
    end_time_ms: int,
) -> tuple[tuple[CapturedPayload, ...], tuple[CapturedPayload, ...]]:
    """Fetch a bounded number of backward pages from both public venues."""
    if not 1 <= bars <= _MAX_HISTORY_BARS or end_time_ms < 0:
        raise CaptureError("history bounds are invalid")
    okx_pages: list[CapturedPayload] = []
    cursor = end_time_ms
    remaining = bars
    while remaining > 0:
        page = fetch(build_okx_candles_url(limit=min(300, remaining), bar="15m", after_ms=cursor))
        rows = _okx_rows(page.raw_bytes)
        if not rows:
            raise CaptureError("OKX history ended before requested bar count")
        okx_pages.append(page)
        oldest = min(_okx_timestamp(row) for row in rows)
        if oldest >= cursor:
            raise CaptureError("OKX history cursor did not move backward")
        cursor = oldest
        remaining -= len(rows)

    binance_pages: list[CapturedPayload] = []
    cursor = end_time_ms
    remaining = bars
    while remaining > 0:
        page = fetch(
            build_binance_klines_url(limit=min(1500, remaining), interval="15m", end_time_ms=cursor)
        )
        rows = _binance_rows(page.raw_bytes)
        if not rows:
            raise CaptureError("Binance history ended before requested bar count")
        binance_pages.append(page)
        oldest = min(_binance_timestamp(row) for row in rows)
        if oldest > cursor or (oldest == 0 and remaining > len(rows)):
            raise CaptureError("Binance history cursor did not move backward")
        cursor = oldest - 1
        remaining -= len(rows)
    return tuple(okx_pages), tuple(binance_pages)


def capture_public_candles(
    *,
    workspace_root: Path,
    output_directory: Path,
    reserve_bytes: int,
    limit: int,
) -> CandleCaptureArtifact:
    import shutil as disk_shutil

    workspace = workspace_root.resolve()
    target = output_directory.resolve()
    StoragePolicy(workspace, reserve_bytes).authorize(
        target=target,
        temporary_directory=target.parent,
        free_bytes=disk_shutil.disk_usage(workspace).free,
        worst_case_required_bytes=10_000_000,
    )
    client = PublicJsonClient(max_response_bytes=2_000_000, timeout_seconds=20)
    okx = client.fetch(build_okx_candles_url(limit=limit, bar="15m"))
    binance = client.fetch(build_binance_klines_url(limit=limit, interval="15m"))
    return publish_candle_capture(okx, binance, output_directory=target)


def capture_public_candle_history(
    *,
    workspace_root: Path,
    output_directory: Path,
    reserve_bytes: int,
    bars: int,
    end_time_ms: int | None = None,
) -> CandleCaptureArtifact:
    """Capture completed 15-minute candles through bounded backward pagination."""
    workspace = workspace_root.resolve()
    target = output_directory.resolve()
    StoragePolicy(workspace, reserve_bytes).authorize(
        target=target,
        temporary_directory=target.parent,
        free_bytes=shutil.disk_usage(workspace).free,
        worst_case_required_bytes=100_000_000,
    )
    if end_time_ms is None:
        interval_ms = _INTERVAL_NS // 1_000_000
        end_time_ms = time.time_ns() // 1_000_000 // interval_ms * interval_ms - 1
    client = PublicJsonClient(max_response_bytes=8_000_000, timeout_seconds=30)
    okx_pages, binance_pages = fetch_history_payloads(
        client.fetch, bars=bars, end_time_ms=end_time_ms
    )
    return publish_candle_history(okx_pages, binance_pages, output_directory=target, bars=bars)


def publish_candle_history(
    okx_pages: Sequence[CapturedPayload],
    binance_pages: Sequence[CapturedPayload],
    *,
    output_directory: Path,
    bars: int,
) -> CandleCaptureArtifact:
    """Publish immutable raw pages plus exactly the latest requested rows per venue."""
    if output_directory.exists():
        raise CaptureError("capture output already exists and is immutable")
    if not okx_pages or not binance_pages or not 1 <= bars <= _MAX_HISTORY_BARS:
        raise CaptureError("history publication bounds are invalid")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_directory.name}.", dir=output_directory.parent)
    )
    try:
        okx_candles = tuple(
            candle
            for page in okx_pages
            for candle in parse_okx_candles(
                _okx_payload(page.raw_bytes),
                instrument_id="BTC-USDT-SWAP.OKX",
                interval_ns=_INTERVAL_NS,
                availability_lag_ns=_HISTORICAL_AVAILABILITY_LAG_NS,
            )
        )
        binance_candles = tuple(
            candle
            for page in binance_pages
            for candle in parse_binance_klines(
                _binance_rows(page.raw_bytes),
                instrument_id="BTC-USDT-SWAP.BINANCE",
                interval_ns=_INTERVAL_NS,
                availability_lag_ns=_HISTORICAL_AVAILABILITY_LAG_NS,
                confirmed_through_ns=page.received_time_ns,
            )
        )
        candles = _latest_unique(okx_candles, bars=bars) + _latest_unique(
            binance_candles, bars=bars
        )
        sources: list[dict[str, object]] = []
        raw_hashes: list[str] = []
        for venue, pages in (("OKX", okx_pages), ("BINANCE", binance_pages)):
            raw_directory = temporary / "raw" / venue.lower()
            raw_directory.mkdir(parents=True)
            for index, page in enumerate(pages):
                relative = Path("raw") / venue.lower() / f"page-{index:04d}.json"
                (temporary / relative).write_bytes(page.raw_bytes)
                raw_hash = hashlib.sha256(page.raw_bytes).hexdigest()
                raw_hashes.append(raw_hash)
                sources.append(
                    {
                        "venue": venue,
                        "page_index": index,
                        "url": page.url,
                        "received_time_ns": page.received_time_ns,
                        "raw_relative_path": relative.as_posix(),
                        "raw_sha256": raw_hash,
                    }
                )
        dataset = publish_candle_dataset(
            candles,
            output_directory=temporary / "dataset",
            raw_source_hashes=tuple(raw_hashes),
        )
        material: dict[str, object] = {
            "capture_version": "2.0.0",
            "requested_bars_per_venue": bars,
            "sources": sources,
            "dataset_root_hash": dataset.root_hash,
        }
        capture_root_hash = content_sha256(material)
        manifest = dict(material)
        manifest["capture_root_hash"] = capture_root_hash
        (temporary / "capture-manifest.json").write_bytes(canonical_json(manifest))
        temporary.replace(output_directory)
    except (CandlePayloadError, DatasetPublicationError, json.JSONDecodeError) as error:
        shutil.rmtree(temporary, ignore_errors=True)
        raise CaptureError("captured history is invalid or incomplete") from error
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    final_dataset_root = output_directory / "dataset"
    return CandleCaptureArtifact(
        capture_root=output_directory,
        capture_manifest_path=output_directory / "capture-manifest.json",
        dataset=CandleDatasetArtifact(
            dataset_root=final_dataset_root,
            manifest_path=final_dataset_root / "dataset-manifest.json",
            quality_report_path=final_dataset_root / "quality-report.json",
            root_hash=dataset.root_hash,
            quality=dataset.quality,
        ),
        capture_root_hash=capture_root_hash,
    )


def publish_candle_capture(
    okx: CapturedPayload,
    binance: CapturedPayload,
    *,
    output_directory: Path,
) -> CandleCaptureArtifact:
    if output_directory.exists():
        raise CaptureError("capture output already exists and is immutable")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_directory.name}.", dir=output_directory.parent)
    )
    try:
        okx_payload = _decode_json(okx.raw_bytes, expected="object")
        binance_payload = _decode_json(binance.raw_bytes, expected="array")
        if not isinstance(okx_payload, dict) or not isinstance(binance_payload, list):
            raise CaptureError("venue JSON shape is invalid")
        candles = parse_okx_candles(
            okx_payload,
            instrument_id="BTC-USDT-SWAP.OKX",
            interval_ns=_INTERVAL_NS,
            availability_lag_ns=_HISTORICAL_AVAILABILITY_LAG_NS,
        ) + parse_binance_klines(
            binance_payload,
            instrument_id="BTC-USDT-SWAP.BINANCE",
            interval_ns=_INTERVAL_NS,
            availability_lag_ns=_HISTORICAL_AVAILABILITY_LAG_NS,
            confirmed_through_ns=binance.received_time_ns,
        )
        raw_directory = temporary / "raw"
        raw_directory.mkdir()
        (raw_directory / "okx.json").write_bytes(okx.raw_bytes)
        (raw_directory / "binance.json").write_bytes(binance.raw_bytes)
        raw_hashes = (
            hashlib.sha256(okx.raw_bytes).hexdigest(),
            hashlib.sha256(binance.raw_bytes).hexdigest(),
        )
        dataset = publish_candle_dataset(
            candles,
            output_directory=temporary / "dataset",
            raw_source_hashes=raw_hashes,
        )
        material: dict[str, object] = {
            "capture_version": "1.0.0",
            "sources": [
                {
                    "venue": "OKX",
                    "url": okx.url,
                    "received_time_ns": okx.received_time_ns,
                    "raw_sha256": raw_hashes[0],
                },
                {
                    "venue": "BINANCE",
                    "url": binance.url,
                    "received_time_ns": binance.received_time_ns,
                    "raw_sha256": raw_hashes[1],
                },
            ],
            "dataset_root_hash": dataset.root_hash,
        }
        capture_root_hash = content_sha256(material)
        manifest = dict(material)
        manifest["capture_root_hash"] = capture_root_hash
        (temporary / "capture-manifest.json").write_bytes(canonical_json(manifest))
        temporary.replace(output_directory)
    except (CandlePayloadError, json.JSONDecodeError) as error:
        shutil.rmtree(temporary, ignore_errors=True)
        raise CaptureError("captured response is invalid JSON or candle data") from error
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    final_dataset_root = output_directory / "dataset"
    return CandleCaptureArtifact(
        capture_root=output_directory,
        capture_manifest_path=output_directory / "capture-manifest.json",
        dataset=CandleDatasetArtifact(
            dataset_root=final_dataset_root,
            manifest_path=final_dataset_root / "dataset-manifest.json",
            quality_report_path=final_dataset_root / "quality-report.json",
            root_hash=dataset.root_hash,
            quality=dataset.quality,
        ),
        capture_root_hash=capture_root_hash,
    )


def verify_candle_capture(capture_root: Path) -> CaptureVerification:
    """Rehash every published layer without trusting recorded success flags."""
    errors: list[str] = []
    try:
        capture_manifest = _load_object(capture_root / "capture-manifest.json")
        recorded_capture_hash = _required_string(capture_manifest, "capture_root_hash")
        capture_material = dict(capture_manifest)
        del capture_material["capture_root_hash"]
        if content_sha256(capture_material) != recorded_capture_hash:
            errors.append("CAPTURE_ROOT_HASH_MISMATCH")

        sources = capture_manifest.get("sources")
        if not isinstance(sources, list):
            raise CaptureError("capture sources must be an array")
        for source in sources:
            if not isinstance(source, dict):
                raise CaptureError("capture source must be an object")
            venue = _required_string(source, "venue")
            expected = _required_string(source, "raw_sha256")
            relative = source.get("raw_relative_path")
            if relative is None:
                raw_path = capture_root / "raw" / f"{venue.lower()}.json"
            elif isinstance(relative, str):
                raw_path = (capture_root / relative).resolve()
                if not raw_path.is_relative_to(capture_root.resolve()):
                    raise CaptureError("raw payload path escapes capture root")
            else:
                raise CaptureError("raw payload path must be a string")
            if (
                not raw_path.exists()
                or hashlib.sha256(raw_path.read_bytes()).hexdigest() != expected
            ):
                errors.append(f"RAW_HASH_MISMATCH:{venue}")

        dataset_root = capture_root / "dataset"
        dataset_manifest = _load_object(dataset_root / "dataset-manifest.json")
        recorded_dataset_hash = _required_string(dataset_manifest, "root_hash")
        dataset_material = dict(dataset_manifest)
        del dataset_material["root_hash"]
        if content_sha256(dataset_material) != recorded_dataset_hash:
            errors.append("DATASET_ROOT_HASH_MISMATCH")
        if capture_manifest.get("dataset_root_hash") != recorded_dataset_hash:
            errors.append("CAPTURE_DATASET_LINK_MISMATCH")

        quality_hash = _required_string(dataset_manifest, "quality_report_hash")
        quality_path = dataset_root / "quality-report.json"
        if (
            not quality_path.exists()
            or hashlib.sha256(quality_path.read_bytes()).hexdigest() != quality_hash
        ):
            errors.append("QUALITY_REPORT_HASH_MISMATCH")
        _verify_parquet_files(dataset_root, dataset_manifest, errors)
    except (CaptureError, OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        errors.append("MANIFEST_STRUCTURE_INVALID")
    return CaptureVerification(valid=not errors, errors=tuple(errors))


def _validate_public_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https":
        raise CaptureError("public market-data URL must use HTTPS")
    if parsed.hostname not in _ALLOWED_HOSTS:
        raise CaptureError("public market-data host is not approved")
    if parsed.username is not None or parsed.password is not None:
        raise CaptureError("public market-data URL must not contain credentials")


def _decode_json(raw: bytes, *, expected: str) -> object:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CaptureError(f"captured {expected} is not valid JSON") from error
    return value


def _okx_payload(raw: bytes) -> dict[str, object]:
    value = _decode_json(raw, expected="object")
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise CaptureError("OKX history payload must be an object")
    return value


def _okx_rows(raw: bytes) -> Sequence[object]:
    rows = _okx_payload(raw).get("data")
    if not isinstance(rows, list):
        raise CaptureError("OKX history data must be an array")
    return rows


def _binance_rows(raw: bytes) -> Sequence[object]:
    value = _decode_json(raw, expected="array")
    if not isinstance(value, list):
        raise CaptureError("Binance history payload must be an array")
    return value


def _okx_timestamp(row: object) -> int:
    if not isinstance(row, list) or not row or not isinstance(row[0], str):
        raise CaptureError("OKX history row has no timestamp")
    try:
        return int(row[0])
    except ValueError as error:
        raise CaptureError("OKX history timestamp is invalid") from error


def _binance_timestamp(row: object) -> int:
    if not isinstance(row, list) or not row or not isinstance(row[0], int):
        raise CaptureError("Binance history row has no timestamp")
    return row[0]


def _latest_unique(candles: Sequence[Candle], *, bars: int) -> tuple[Candle, ...]:
    unique: dict[int, Candle] = {}
    for candle in candles:
        existing = unique.get(candle.open_time_ns)
        if existing is not None and existing != candle:
            raise CaptureError("history contains a conflicting duplicate candle")
        unique[candle.open_time_ns] = candle
    confirmed = sorted(
        (candle for candle in unique.values() if candle.confirmed),
        key=lambda candle: candle.open_time_ns,
    )
    if len(confirmed) < bars:
        raise CaptureError("history contains fewer confirmed candles than requested")
    return tuple(confirmed[-bars:])


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise CaptureError(f"{path.name} must be a JSON object")
    return value


def _required_string(record: Mapping[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str):
        raise CaptureError(f"manifest field {key} must be a string")
    return value


def _verify_parquet_files(
    dataset_root: Path, manifest: dict[str, object], errors: list[str]
) -> None:
    files = manifest.get("files")
    if not isinstance(files, list):
        raise CaptureError("dataset files must be an array")
    for item in files:
        if not isinstance(item, dict):
            raise CaptureError("dataset file entry must be an object")
        relative = _required_string(item, "relative_path")
        expected = _required_string(item, "sha256")
        path = (dataset_root / relative).resolve()
        if not path.is_relative_to(dataset_root.resolve()):
            raise CaptureError("dataset file path escapes capture root")
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(f"PARQUET_HASH_MISMATCH:{relative}")
