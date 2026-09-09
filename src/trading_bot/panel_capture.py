"""Bounded capture of Binance USD-M daily klines and funding from public dumps."""

import csv
import hashlib
import io
import json
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.panel_dataset import (
    DAY_NS,
    PanelCandleRow,
    PanelFundingRow,
    publish_panel_dataset,
    verify_panel_dataset,
)
from trading_bot.storage import StoragePolicy

_ALLOWED_PANEL_HOSTS = frozenset(
    {"data.binance.vision", "s3-ap-northeast-1.amazonaws.com", "fapi.binance.com"}
)
_VENUE = "BINANCE_UM"
_MAX_ZIP_BYTES = 32_000_000
_MONTH_PATTERN = re.compile(r"^\d{4}-\d{2}$")
_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9_]{2,32}$")


class PanelCaptureError(RuntimeError):
    """Raised when panel capture fails closed."""


@dataclass(frozen=True, slots=True)
class PanelPayload:
    url: str
    raw_bytes: bytes
    received_time_ns: int


@dataclass(frozen=True, slots=True)
class PanelCaptureArtifact:
    capture_root: Path
    capture_manifest_path: Path
    dataset_root: Path
    capture_root_hash: str
    dataset_root_hash: str


PanelFetch = Callable[[str], PanelPayload]


@dataclass(frozen=True, slots=True)
class PanelZipClient:
    timeout_seconds: int = 60

    def fetch(self, url: str) -> PanelPayload:
        _validate_panel_url(url)
        request = urllib.request.Request(
            url, headers={"User-Agent": "hybrid-trading-research/0.1"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                _validate_panel_url(response.geturl())
                raw = response.read(_MAX_ZIP_BYTES + 1)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            raise PanelCaptureError(f"panel dump request failed: {error}") from error
        if len(raw) > _MAX_ZIP_BYTES:
            raise PanelCaptureError("panel dump exceeds the byte limit")
        return PanelPayload(url=url, raw_bytes=raw, received_time_ns=time.time_ns())


def build_kline_zip_url(symbol: str, month: str) -> str:
    _validate_symbol(symbol)
    _validate_month(month)
    return (
        "https://data.binance.vision/data/futures/um/monthly/klines/"
        f"{symbol}/1d/{symbol}-1d-{month}.zip"
    )


def build_funding_zip_url(symbol: str, month: str) -> str:
    _validate_symbol(symbol)
    _validate_month(month)
    return (
        "https://data.binance.vision/data/futures/um/monthly/fundingRate/"
        f"{symbol}/{symbol}-fundingRate-{month}.zip"
    )


def parse_kline_zip(payload: PanelPayload, *, symbol: str) -> tuple[PanelCandleRow, ...]:
    digest = hashlib.sha256(payload.raw_bytes).hexdigest()
    rows: list[PanelCandleRow] = []
    for record in _zip_rows(payload.raw_bytes):
        if record[0].startswith("open_time"):
            continue
        if len(record) < 9:
            raise PanelCaptureError("kline row has too few columns")
        open_time_ns = int(record[0]) * 1_000_000
        close_time_ns = int(record[6]) * 1_000_000
        rows.append(
            PanelCandleRow(
                venue=_VENUE,
                instrument_id=symbol,
                open_time_ns=open_time_ns,
                close_time_ns=close_time_ns,
                available_time_ns=close_time_ns + 1,
                interval_ns=DAY_NS,
                open=Decimal(record[1]),
                high=Decimal(record[2]),
                low=Decimal(record[3]),
                close=Decimal(record[4]),
                base_volume=Decimal(record[5]),
                quote_volume=Decimal(record[7]),
                trade_count=int(record[8]),
                source_payload_hash=digest,
            )
        )
    return tuple(rows)


def parse_funding_zip(payload: PanelPayload, *, symbol: str) -> tuple[PanelFundingRow, ...]:
    rows: list[PanelFundingRow] = []
    for record in _zip_rows(payload.raw_bytes):
        if record[0].startswith("calc_time"):
            continue
        if len(record) < 3:
            raise PanelCaptureError("funding row has too few columns")
        rows.append(
            PanelFundingRow(
                venue=_VENUE,
                instrument_id=symbol,
                calc_time_ns=int(record[0]) * 1_000_000,
                funding_interval_hours=int(record[1]),
                rate=Decimal(record[2]),
            )
        )
    return tuple(rows)


def capture_panel(
    *,
    workspace_root: Path,
    output_directory: Path,
    reserve_bytes: int,
    symbols: tuple[str, ...],
    months: tuple[str, ...],
    fetch: PanelFetch | None = None,
) -> PanelCaptureArtifact:
    if not symbols or not months:
        raise PanelCaptureError("panel capture needs at least one symbol and one month")
    workspace = workspace_root.resolve()
    target = output_directory.resolve()
    StoragePolicy(workspace, reserve_bytes).authorize(
        target=target,
        temporary_directory=target.parent,
        free_bytes=shutil.disk_usage(workspace).free,
        worst_case_required_bytes=500_000_000,
    )
    if target.exists():
        raise PanelCaptureError("panel capture already exists and is immutable")
    download = fetch if fetch is not None else PanelZipClient().fetch

    raw_root = target / "raw"
    sources: list[dict[str, object]] = []
    candles: list[PanelCandleRow] = []
    funding: list[PanelFundingRow] = []
    for symbol in symbols:
        for month in months:
            for kind, url in (
                ("klines", build_kline_zip_url(symbol, month)),
                ("fundingRate", build_funding_zip_url(symbol, month)),
            ):
                payload = download(url)
                _validate_panel_url(payload.url)
                relative = f"raw/{symbol}/{kind}-{month}.zip"
                path = target / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload.raw_bytes)
                sources.append(
                    {
                        "symbol": symbol,
                        "month": month,
                        "kind": kind,
                        "url": payload.url,
                        "received_time_ns": payload.received_time_ns,
                        "raw_relative_path": relative,
                        "raw_sha256": hashlib.sha256(payload.raw_bytes).hexdigest(),
                    }
                )
                if kind == "klines":
                    candles.extend(parse_kline_zip(payload, symbol=symbol))
                else:
                    funding.extend(parse_funding_zip(payload, symbol=symbol))
    if not raw_root.is_dir():
        raise PanelCaptureError("panel capture produced no raw payloads")

    dataset = publish_panel_dataset(
        tuple(candles),
        tuple(funding),
        output_directory=target / "dataset",
        raw_source_hashes=tuple(str(item["raw_sha256"]) for item in sources),
    )
    material: dict[str, object] = {
        "capture_version": "1.0.0",
        "venue": _VENUE,
        "interval": "1d",
        "symbols": list(symbols),
        "months": list(months),
        "sources": sources,
        "dataset_root_hash": dataset.root_hash,
    }
    capture_root_hash = content_sha256(material)
    document = dict(material)
    document["capture_root_hash"] = capture_root_hash
    (target / "capture-manifest.json").write_bytes(canonical_json(document))
    return PanelCaptureArtifact(
        capture_root=target,
        capture_manifest_path=target / "capture-manifest.json",
        dataset_root=dataset.dataset_root,
        capture_root_hash=capture_root_hash,
        dataset_root_hash=dataset.root_hash,
    )


def verify_panel_capture(capture_root: Path) -> tuple[bool, tuple[str, ...]]:
    errors: list[str] = []
    try:
        manifest = json.loads(
            (capture_root / "capture-manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False, ("CAPTURE_MANIFEST_UNREADABLE",)
    if not isinstance(manifest, dict):
        return False, ("MANIFEST_STRUCTURE_INVALID",)
    recorded = manifest.get("capture_root_hash")
    material = {key: value for key, value in manifest.items() if key != "capture_root_hash"}
    if content_sha256(material) != recorded:
        errors.append("CAPTURE_ROOT_HASH_MISMATCH")
    sources = manifest.get("sources")
    if not isinstance(sources, list):
        return False, tuple([*errors, "MANIFEST_STRUCTURE_INVALID"])
    for entry in sources:
        if not isinstance(entry, dict):
            errors.append("MANIFEST_STRUCTURE_INVALID")
            continue
        relative = str(entry.get("raw_relative_path"))
        try:
            digest = hashlib.sha256((capture_root / relative).read_bytes()).hexdigest()
        except OSError:
            errors.append(f"RAW_MISSING:{relative}")
            continue
        if digest != entry.get("raw_sha256"):
            errors.append(f"RAW_HASH_MISMATCH:{relative}")
    dataset_valid, dataset_errors = verify_panel_dataset(capture_root / "dataset")
    if not dataset_valid:
        errors.extend(dataset_errors)
    try:
        dataset_manifest = json.loads(
            (capture_root / "dataset" / "dataset-manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False, tuple([*errors, "DATASET_MANIFEST_UNREADABLE"])
    if dataset_manifest.get("root_hash") != manifest.get("dataset_root_hash"):
        errors.append("CAPTURE_DATASET_LINK_MISMATCH")
    return not errors, tuple(errors)


def _zip_rows(raw: bytes) -> list[list[str]]:
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            names = archive.namelist()
            if len(names) != 1:
                raise PanelCaptureError("panel dump must contain exactly one member")
            text = archive.read(names[0]).decode("utf-8")
    except (zipfile.BadZipFile, UnicodeDecodeError) as error:
        raise PanelCaptureError(f"panel dump is unreadable: {error}") from error
    return [row for row in csv.reader(io.StringIO(text)) if row]


def _validate_panel_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in _ALLOWED_PANEL_HOSTS:
        raise PanelCaptureError(f"panel URL is not allowed: {url}")


def _validate_symbol(symbol: str) -> None:
    if not _SYMBOL_PATTERN.match(symbol):
        raise PanelCaptureError(f"invalid symbol: {symbol}")


def _validate_month(month: str) -> None:
    if not _MONTH_PATTERN.match(month):
        raise PanelCaptureError(f"invalid month: {month}")
