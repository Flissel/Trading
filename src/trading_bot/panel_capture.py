"""Bounded capture of Binance USD-M daily klines and funding from public dumps."""

import csv
import hashlib
import http.client
import io
import json
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

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
_LISTING_MAX_KEYS = 1000


class PanelCaptureError(RuntimeError):
    """Raised when panel capture fails closed."""


class PanelSourceAbsent(PanelCaptureError):
    """Raised when one (symbol, month, kind) source dump does not exist upstream.

    A 404 from the public dumps means the contract was not yet listed or was
    already delisted for that month -- a normal, expected condition for a
    survivorship-bias-free panel, not an outage. Callers should record the
    absence and continue rather than aborting the capture.
    """


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
    max_attempts: int = 4
    sleep: Callable[[float], None] = time.sleep

    def fetch(self, url: str) -> PanelPayload:
        _validate_panel_url(url)
        last_error: BaseException | None = None
        for attempt in range(1, self.max_attempts + 1):
            request = urllib.request.Request(
                url, headers={"User-Agent": "hybrid-trading-research/0.1"}
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    _validate_panel_url(response.geturl())
                    raw = response.read(_MAX_ZIP_BYTES + 1)
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    raise PanelSourceAbsent(f"panel dump is absent: {error}") from error
                if error.code == 429 or error.code >= 500:
                    # A throttle (429) or a server-side failure (5xx) is
                    # transient -- back off and retry. Any other 4xx (403,
                    # 400, ...) is a real problem, not a transient one --
                    # surface it immediately, unretried.
                    last_error = error
                else:
                    raise PanelCaptureError(f"panel dump request failed: {error}") from error
            except (
                urllib.error.URLError,
                TimeoutError,
                ConnectionResetError,
                http.client.IncompleteRead,
            ) as error:
                # Covers both the request itself and draining the response
                # body -- a reset or a short read mid-transfer is exactly as
                # transient as a failure to connect in the first place.
                last_error = error
            except PanelCaptureError:
                raise
            except Exception as error:
                # Anything else unexpected must not escape raw and unwrapped;
                # it is not a failure mode this client recognizes as
                # transient, so it is not retried.
                raise PanelCaptureError(f"panel dump request failed: {error}") from error
            else:
                if len(raw) > _MAX_ZIP_BYTES:
                    raise PanelCaptureError("panel dump exceeds the byte limit")
                return PanelPayload(url=url, raw_bytes=raw, received_time_ns=time.time_ns())
            if attempt < self.max_attempts:
                self.sleep(2 ** (attempt - 1))
        raise PanelCaptureError(
            f"panel dump request failed after {self.max_attempts} attempts: {last_error}"
        ) from last_error


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


def build_month_listing_url(symbol: str, kind: str) -> str:
    prefix = _listing_prefix(symbol, kind)
    encoded_prefix = urllib.parse.quote(prefix, safe="")
    url = (
        "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
        f"?list-type=2&max-keys={_LISTING_MAX_KEYS}&prefix={encoded_prefix}"
    )
    _validate_panel_url(url)
    return url


def discover_panel_months(fetch: PanelFetch, *, symbol: str, kind: str) -> tuple[str, ...]:
    """Ask the bucket which months a symbol actually has for one source kind.

    A symbol has at most about eighty months of history, so a single page of
    up to `_LISTING_MAX_KEYS` keys always suffices. The response is required
    to look exactly like a complete, unpaginated listing -- anything else
    (missing or true `IsTruncated`, a pagination token, a non-listing body,
    or a key count at the page limit) is a hard failure rather than a silent
    partial list.
    """
    url = build_month_listing_url(symbol, kind)
    payload = fetch(url)
    return _parse_listing_months(payload.raw_bytes, symbol=symbol, kind=kind)


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
    months: tuple[str, ...] | None = None,
    month_from: str | None = None,
    month_to: str | None = None,
    fetch: PanelFetch | None = None,
) -> PanelCaptureArtifact:
    if not symbols:
        raise PanelCaptureError("panel capture needs at least one symbol and one month")
    if months is not None:
        if not months:
            raise PanelCaptureError("panel capture needs at least one symbol and one month")
        for month in months:
            _validate_month(month)
    if month_from is not None:
        _validate_month(month_from)
    if month_to is not None:
        _validate_month(month_to)
    if month_from is not None and month_to is not None and month_from > month_to:
        raise PanelCaptureError("month_from must not be after month_to")
    workspace = workspace_root.resolve()
    target = output_directory.resolve()
    StoragePolicy(workspace, reserve_bytes).authorize(
        target=target,
        temporary_directory=target.parent,
        free_bytes=shutil.disk_usage(workspace).free,
        worst_case_required_bytes=500_000_000,
    )
    manifest_path = target / "capture-manifest.json"
    if manifest_path.exists():
        raise PanelCaptureError("panel capture already exists and is immutable")
    target.mkdir(parents=True, exist_ok=True)

    # A previous run may have died between publishing the dataset and writing
    # the manifest above, leaving a `dataset` directory behind with no
    # manifest over it. That directory is entirely derived from the raw
    # payloads under `raw/`, every one of which is still on disk (and, if the
    # fetch phase itself finished, recorded in the progress file read below),
    # so it is safe -- and, since publish_panel_dataset refuses to write over
    # an existing directory, necessary -- to remove it narrowly here and let
    # this resumed run republish it from scratch. This only ever runs when
    # capture-manifest.json is absent, i.e. the capture never completed.
    dataset_directory = target / "dataset"
    if dataset_directory.exists():
        shutil.rmtree(dataset_directory)

    download = fetch if fetch is not None else PanelZipClient().fetch

    progress_path = target / "capture-progress.jsonl"
    sources: list[dict[str, object]] = []
    already_done: dict[tuple[str, str, str], dict[str, Any]] = {}
    if progress_path.exists():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            sources.append(entry)
            already_done[(entry["symbol"], entry["month"], entry["kind"])] = entry

    raw_root = target / "raw"
    candles: list[PanelCandleRow] = []
    funding: list[PanelFundingRow] = []
    with progress_path.open("a", encoding="utf-8") as progress_handle:
        for symbol in symbols:
            symbol_candle_row_count = 0
            if months is not None:
                kline_months: tuple[str, ...] = months
                funding_months: tuple[str, ...] = months
                ordered_months: tuple[str, ...] = months
            else:
                kline_months = _bounded_months(
                    discover_panel_months(download, symbol=symbol, kind="klines"),
                    month_from,
                    month_to,
                )
                funding_months = _bounded_months(
                    discover_panel_months(download, symbol=symbol, kind="fundingRate"),
                    month_from,
                    month_to,
                )
                ordered_months = tuple(sorted(set(kline_months) | set(funding_months)))
            for month in ordered_months:
                for kind, url_builder, kind_months in (
                    ("klines", build_kline_zip_url, kline_months),
                    ("fundingRate", build_funding_zip_url, funding_months),
                ):
                    if month not in kind_months:
                        continue
                    key = (symbol, month, kind)
                    done = already_done.get(key)
                    if done is not None:
                        # Already recorded by a prior, interrupted run: don't
                        # refetch it (that would also mint a fresh
                        # received_time_ns, losing the original provenance),
                        # just re-derive its rows from the raw payload that
                        # run already wrote to disk.
                        if done.get("status") == "present":
                            relative = str(done["raw_relative_path"])
                            raw_bytes = (target / relative).read_bytes()
                            if hashlib.sha256(raw_bytes).hexdigest() != done.get("raw_sha256"):
                                raise PanelCaptureError(
                                    f"resumed raw payload hash mismatch: {relative}"
                                )
                            payload = PanelPayload(
                                url=str(done["url"]),
                                raw_bytes=raw_bytes,
                                received_time_ns=int(done["received_time_ns"]),
                            )
                            if kind == "klines":
                                rows = parse_kline_zip(payload, symbol=symbol)
                                candles.extend(rows)
                                symbol_candle_row_count += len(rows)
                            else:
                                funding.extend(parse_funding_zip(payload, symbol=symbol))
                        continue
                    url = url_builder(symbol, month)
                    try:
                        payload = download(url)
                    except PanelSourceAbsent:
                        absent_record: dict[str, object] = {
                            "symbol": symbol,
                            "month": month,
                            "kind": kind,
                            "url": url,
                            "status": "absent",
                        }
                        sources.append(absent_record)
                        progress_handle.write(canonical_json(absent_record).decode("utf-8"))
                        progress_handle.write("\n")
                        progress_handle.flush()
                        continue
                    _validate_panel_url(payload.url)
                    relative = f"raw/{symbol}/{kind}-{month}.zip"
                    path = target / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(payload.raw_bytes)
                    present_record: dict[str, object] = {
                        "symbol": symbol,
                        "month": month,
                        "kind": kind,
                        "url": payload.url,
                        "received_time_ns": payload.received_time_ns,
                        "raw_relative_path": relative,
                        "raw_sha256": hashlib.sha256(payload.raw_bytes).hexdigest(),
                        "status": "present",
                    }
                    sources.append(present_record)
                    progress_handle.write(canonical_json(present_record).decode("utf-8"))
                    progress_handle.write("\n")
                    progress_handle.flush()
                    if kind == "klines":
                        rows = parse_kline_zip(payload, symbol=symbol)
                        candles.extend(rows)
                        symbol_candle_row_count += len(rows)
                    else:
                        funding.extend(parse_funding_zip(payload, symbol=symbol))
            if symbol_candle_row_count == 0:
                raise PanelCaptureError(
                    f"symbol {symbol} produced no candle rows across every requested month"
                )
    if not raw_root.is_dir():
        raise PanelCaptureError("panel capture produced no raw payloads")

    dataset = publish_panel_dataset(
        tuple(candles),
        tuple(funding),
        output_directory=target / "dataset",
        raw_source_hashes=tuple(
            str(item["raw_sha256"]) for item in sources if item.get("status") == "present"
        ),
    )
    material: dict[str, object] = {
        "capture_version": "1.0.0",
        "venue": _VENUE,
        "interval": "1d",
        "symbols": list(symbols),
        "months": list(months) if months is not None else None,
        "sources": sources,
        "dataset_root_hash": dataset.root_hash,
    }
    if month_from is not None:
        material["month_from"] = month_from
    if month_to is not None:
        material["month_to"] = month_to
    capture_root_hash = content_sha256(material)
    document = dict(material)
    document["capture_root_hash"] = capture_root_hash
    manifest_path.write_bytes(canonical_json(document))
    progress_path.unlink(missing_ok=True)
    return PanelCaptureArtifact(
        capture_root=target,
        capture_manifest_path=manifest_path,
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
        if entry.get("status") == "absent":
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


def _listing_prefix(symbol: str, kind: str) -> str:
    _validate_symbol(symbol)
    if kind == "klines":
        return f"data/futures/um/monthly/klines/{symbol}/1d/"
    if kind == "fundingRate":
        return f"data/futures/um/monthly/fundingRate/{symbol}/"
    raise PanelCaptureError(f"invalid panel source kind: {kind}")


def _listing_key_pattern(symbol: str, kind: str) -> re.Pattern[str]:
    escaped = re.escape(symbol)
    if kind == "klines":
        return re.compile(
            rf"^data/futures/um/monthly/klines/{escaped}/1d/{escaped}-1d-(\d{{4}}-\d{{2}})\.zip$"
        )
    return re.compile(
        rf"^data/futures/um/monthly/fundingRate/{escaped}/"
        rf"{escaped}-fundingRate-(\d{{4}}-\d{{2}})\.zip$"
    )


def _parse_listing_months(raw: bytes, *, symbol: str, kind: str) -> tuple[str, ...]:
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as error:
        raise PanelCaptureError(f"panel month listing is unreadable: {error}") from error
    # A non-2xx S3 error can still arrive as an XML body over HTTP 200 (or
    # any other body shape entirely); a root that isn't a ListBucketResult
    # must never be read as "zero keys", so reject it outright.
    if str(root.tag).rsplit("}", 1)[-1] != "ListBucketResult":
        raise PanelCaptureError(
            f"panel month listing for {symbol} {kind} is not a ListBucketResult response"
        )
    is_truncated: str | None = None
    keys: list[str] = []
    for element in root.iter():
        tag = str(element.tag).rsplit("}", 1)[-1]
        if tag == "IsTruncated":
            is_truncated = (element.text or "").strip().lower()
        elif tag in ("NextContinuationToken", "NextMarker") and (element.text or "").strip():
            raise PanelCaptureError(
                f"panel month listing for {symbol} {kind} is paginated"
            )
        elif tag == "Key" and element.text:
            keys.append(element.text.strip())
    # Require IsTruncated to be present and exactly "false": missing, true,
    # or any other value is treated as an ambiguous, unsafe-to-trust listing.
    if is_truncated != "false":
        raise PanelCaptureError(
            f"panel month listing for {symbol} {kind} was truncated or ambiguous"
        )
    if len(keys) >= _LISTING_MAX_KEYS:
        raise PanelCaptureError(
            f"panel month listing for {symbol} {kind} reached the page limit"
        )
    # The strict pattern re-confirms the exact symbol on every key, so a
    # prefix collision (BTCUSDT vs. BTCUSDTX) can never leak a foreign month.
    pattern = _listing_key_pattern(symbol, kind)
    months = {match.group(1) for key in keys if (match := pattern.match(key)) is not None}
    return tuple(sorted(months))


def _bounded_months(
    months: tuple[str, ...], month_from: str | None, month_to: str | None
) -> tuple[str, ...]:
    return tuple(
        month
        for month in months
        if (month_from is None or month >= month_from) and (month_to is None or month <= month_to)
    )


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
