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
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, TextIO

from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.panel_dataset import (
    DAY_NS,
    PanelCandleRow,
    PanelFundingRow,
    admit_candles,
    find_missing_days,
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
_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9_]{2,32}$")
_LISTING_MAX_KEYS = 1000
_CAPTURE_VERSION = "1.0.0"
# Source "kind" recorded for a candle fetched from the daily dump to patch a
# gap in the monthly aggregates -- distinct from "klines" (the monthly dump
# itself) so the manifest keeps provenance explicit: a reader can always tell
# a monthly row from a gap-filled one.
_DAILY_FILL_KIND = "klines_daily_fill"


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
            except (OSError, http.client.IncompleteRead) as error:
                # OSError covers URLError, TimeoutError, ConnectionResetError,
                # ConnectionAbortedError, BrokenPipeError, and ssl.SSLError --
                # the whole family of connection- and read-phase transients,
                # not just the two most obvious members of it.
                # IncompleteRead is listed separately: it is an
                # http.client.HTTPException, not an OSError.
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


def build_daily_kline_zip_url(symbol: str, date: str) -> str:
    """URL of one day's kline dump -- used only to patch a gap in the monthly
    aggregate, never as the primary source for a month already covered."""
    _validate_symbol(symbol)
    _validate_date(date)
    return (
        "https://data.binance.vision/data/futures/um/daily/klines/"
        f"{symbol}/1d/{symbol}-1d-{date}.zip"
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
    discovery_mode = months is None

    # Recorded at the top of the progress file and re-checked on resume (see
    # below): a resumed run must be continuing *this exact* request, not a
    # narrowed or widened one -- otherwise sources seeded from the old run
    # (e.g. a symbol or month range the operator just dropped) would still
    # land in the manifest and raw_source_hashes without their rows ever
    # being parsed into the dataset.
    capture_parameters: dict[str, object] = {
        "capture_version": _CAPTURE_VERSION,
        "symbols": list(symbols),
        "months": list(months) if months is not None else None,
        "month_from": month_from,
        "month_to": month_to,
    }

    progress_path = target / "capture-progress.jsonl"
    sources, already_done, need_header = _load_progress(progress_path, capture_parameters)

    raw_root = target / "raw"
    candles: list[PanelCandleRow] = []
    funding: list[PanelFundingRow] = []
    discovered_months: dict[str, dict[str, list[str]]] = {}
    with progress_path.open("a", encoding="utf-8") as progress_handle:
        if need_header:
            header_record = {"capture_parameters": capture_parameters}
            progress_handle.write(canonical_json(header_record).decode("utf-8"))
            progress_handle.write("\n")
            progress_handle.flush()
        for symbol in symbols:
            symbol_candle_row_count = 0
            # Only meaningful in discovery mode (see the empty-row guard
            # below): true when the bucket does list kline months for this
            # symbol, but every one of them falls outside [month_from,
            # month_to]. That is legitimate emptiness -- a contract that had
            # not started (or had already ended) trading inside the
            # requested window, e.g. FTTUSDT first listed 2022-04 against a
            # 2022-01..2022-03 pilot window -- not a wrong symbol.
            symbol_has_no_months_inside_bounds = False
            if months is not None:
                kline_months: tuple[str, ...] = months
                funding_months: tuple[str, ...] = months
                ordered_months: tuple[str, ...] = months
            else:
                discovered_kline_months = discover_panel_months(
                    download, symbol=symbol, kind="klines"
                )
                discovered_funding_months = discover_panel_months(
                    download, symbol=symbol, kind="fundingRate"
                )
                kline_months = _bounded_months(discovered_kline_months, month_from, month_to)
                funding_months = _bounded_months(discovered_funding_months, month_from, month_to)
                discovered_months[symbol] = {
                    "klines": list(kline_months),
                    "fundingRate": list(funding_months),
                }
                ordered_months = tuple(sorted(set(kline_months) | set(funding_months)))
                symbol_has_no_months_inside_bounds = (
                    bool(discovered_kline_months) and not kline_months
                )
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
                        # In discovery mode this month came from the bucket's
                        # own listing, so a 404 on fetch means the bucket
                        # contradicted itself inside one run -- record that
                        # distinctly rather than as an ordinary absence.
                        status = "absent_after_discovery" if discovery_mode else "absent"
                        absent_record: dict[str, object] = {
                            "symbol": symbol,
                            "month": month,
                            "kind": kind,
                            "url": url,
                            "status": status,
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
            if symbol_candle_row_count == 0 and not symbol_has_no_months_inside_bounds:
                raise PanelCaptureError(
                    f"symbol {symbol} produced no candle rows across every requested month"
                )
        # The monthly aggregates can have interior holes the daily dumps do
        # not (see `_fill_gap_days`): patch them here, once, so a completed
        # capture starts clean rather than tripping the downstream capture
        # quality gate every time.
        admitted_candles, _ = admit_candles(tuple(candles))
        missing_days = find_missing_days(admitted_candles)
        candles.extend(
            _fill_gap_days(
                missing_days,
                target=target,
                download=download,
                sources=sources,
                already_done=already_done,
                progress_handle=progress_handle,
            )
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
        "capture_version": _CAPTURE_VERSION,
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
    if discovered_months:
        material["discovered_months"] = discovered_months
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


def repair_panel_capture(
    *,
    workspace_root: Path,
    source_capture_root: Path,
    output_directory: Path,
    reserve_bytes: int,
    fetch: PanelFetch | None = None,
) -> PanelCaptureArtifact:
    """Repair a completed capture's gaps into a new, immutable capture,
    without refetching a single one of its monthly sources.

    Verifies `source_capture_root`, copies every one of its raw payloads
    byte-for-byte (re-verifying each copy against its recorded hash),
    reconstructs its candle and funding rows from those copies, then runs
    the same gap-finding and daily-dump gap-filling `capture_panel` runs at
    the end of a fresh capture (`_fill_gap_days`) against the reconstructed
    rows. The source capture is only ever read, never written to, so both
    it and the new, repaired capture stay immutable. The new manifest
    records `source_capture_root_hash` -- the source capture's own root
    hash -- so the derivation is auditable, and its `sources` list carries
    both the copied sources and the newly filled ones, told apart by
    `kind`.
    """
    workspace = workspace_root.resolve()
    source_root = source_capture_root.resolve()
    target = output_directory.resolve()

    valid, verify_errors = verify_panel_capture(source_root)
    if not valid:
        raise PanelCaptureError(
            "source panel capture failed verification: " + ",".join(verify_errors)
        )
    source_manifest = _load_capture_manifest(source_root)
    source_capture_root_hash = str(source_manifest["capture_root_hash"])
    source_sources = source_manifest.get("sources")
    if not isinstance(source_sources, list):
        raise PanelCaptureError("source panel capture manifest is malformed")

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

    dataset_directory = target / "dataset"
    if dataset_directory.exists():
        shutil.rmtree(dataset_directory)

    download = fetch if fetch is not None else PanelZipClient().fetch

    # Bound the same way a fresh capture binds capture_parameters: a resumed
    # repair must be repairing *this exact* source capture, identified by
    # its own root hash, not a different one that happens to share a
    # directory.
    repair_parameters: dict[str, object] = {
        "capture_version": _CAPTURE_VERSION,
        "source_capture_root_hash": source_capture_root_hash,
    }

    progress_path = target / "capture-progress.jsonl"
    sources, already_done, need_header = _load_progress(progress_path, repair_parameters)

    raw_root = target / "raw"
    candles: list[PanelCandleRow] = []
    funding: list[PanelFundingRow] = []
    with progress_path.open("a", encoding="utf-8") as progress_handle:
        if need_header:
            header_record = {"capture_parameters": repair_parameters}
            progress_handle.write(canonical_json(header_record).decode("utf-8"))
            progress_handle.write("\n")
            progress_handle.flush()
        for entry in source_sources:
            if not isinstance(entry, dict):
                raise PanelCaptureError("source panel capture manifest is malformed")
            symbol = str(entry.get("symbol"))
            month = str(entry.get("month"))
            kind = str(entry.get("kind"))
            if kind == _DAILY_FILL_KIND and entry.get("status") != "present":
                # An earlier, unresolved gap-fill attempt from the source
                # capture -- not carried forward. The fresh gap computation
                # below rediscovers this same day from the reconstructed
                # candle rows, and this repair gets its own, current attempt
                # at it (the whole reason to run a repair later), recorded
                # once under this same (symbol, date, kind) key. Carrying the
                # stale attempt forward too would collide with that record.
                continue
            key = (symbol, month, kind)
            done = already_done.get(key)
            if done is None:
                record: dict[str, Any]
                if entry.get("status") != "present":
                    # No raw payload to copy -- carry the absence forward
                    # verbatim, exactly as the source recorded it.
                    record = {
                        "symbol": symbol,
                        "month": month,
                        "kind": kind,
                        "url": str(entry.get("url")),
                        "status": str(entry.get("status")),
                    }
                else:
                    relative = str(entry["raw_relative_path"])
                    expected_hash = str(entry["raw_sha256"])
                    source_bytes = (source_root / relative).read_bytes()
                    if hashlib.sha256(source_bytes).hexdigest() != expected_hash:
                        raise PanelCaptureError(f"source raw payload hash mismatch: {relative}")
                    copy_path = target / relative
                    copy_path.parent.mkdir(parents=True, exist_ok=True)
                    copy_path.write_bytes(source_bytes)
                    if hashlib.sha256(copy_path.read_bytes()).hexdigest() != expected_hash:
                        raise PanelCaptureError(f"copied raw payload hash mismatch: {relative}")
                    record = {
                        "symbol": symbol,
                        "month": month,
                        "kind": kind,
                        "url": str(entry["url"]),
                        "received_time_ns": int(entry["received_time_ns"]),
                        "raw_relative_path": relative,
                        "raw_sha256": expected_hash,
                        "status": "present",
                    }
                sources.append(record)
                progress_handle.write(canonical_json(record).decode("utf-8"))
                progress_handle.write("\n")
                progress_handle.flush()
                done = record
            if done.get("status") == "present":
                relative = str(done["raw_relative_path"])
                payload = PanelPayload(
                    url=str(done["url"]),
                    raw_bytes=(target / relative).read_bytes(),
                    received_time_ns=int(done["received_time_ns"]),
                )
                # A capture repaired once already can itself be the source of
                # a later repair, so a klines-shaped row can arrive under
                # either kind.
                if kind in ("klines", _DAILY_FILL_KIND):
                    candles.extend(parse_kline_zip(payload, symbol=symbol))
                elif kind == "fundingRate":
                    funding.extend(parse_funding_zip(payload, symbol=symbol))
        admitted_candles, _ = admit_candles(tuple(candles))
        missing_days = find_missing_days(admitted_candles)
        candles.extend(
            _fill_gap_days(
                missing_days,
                target=target,
                download=download,
                sources=sources,
                already_done=already_done,
                progress_handle=progress_handle,
            )
        )
    if not raw_root.is_dir():
        raise PanelCaptureError("panel capture repair produced no raw payloads")

    dataset = publish_panel_dataset(
        tuple(candles),
        tuple(funding),
        output_directory=target / "dataset",
        raw_source_hashes=tuple(
            str(item["raw_sha256"]) for item in sources if item.get("status") == "present"
        ),
    )
    material: dict[str, object] = {
        "capture_version": _CAPTURE_VERSION,
        "venue": _VENUE,
        "interval": "1d",
        "symbols": source_manifest.get("symbols"),
        "months": source_manifest.get("months"),
        "sources": sources,
        "dataset_root_hash": dataset.root_hash,
        "source_capture_root_hash": source_capture_root_hash,
    }
    month_from = source_manifest.get("month_from")
    if month_from is not None:
        material["month_from"] = month_from
    month_to = source_manifest.get("month_to")
    if month_to is not None:
        material["month_to"] = month_to
    discovered_months = source_manifest.get("discovered_months")
    if discovered_months:
        material["discovered_months"] = discovered_months
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
        if entry.get("status") != "present":
            # Any non-present status (absent, or a discovery-then-404
            # absent_after_discovery) has no raw payload to check.
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


def _load_capture_manifest(capture_root: Path) -> dict[str, object]:
    try:
        manifest = json.loads(
            (capture_root / "capture-manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise PanelCaptureError(
            f"source panel capture manifest is unreadable: {capture_root}"
        ) from error
    if not isinstance(manifest, dict):
        raise PanelCaptureError(f"source panel capture manifest is malformed: {capture_root}")
    return manifest


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


def _is_valid_progress_record(record: object) -> bool:
    """Check a parsed progress-file record has the shape its status needs.

    Every record, regardless of status, carries symbol/month/kind/url as
    strings. A "present" record additionally carries raw_relative_path and
    raw_sha256 as strings and received_time_ns as an int (not a bool --
    JSON's `true`/`false` decode to Python bools, which are technically
    `int` instances but never a valid nanosecond timestamp): those are the
    fields read back when a resumed "present" source is re-derived from the
    raw payload already on disk, and a record missing one would otherwise
    raise a raw KeyError there instead of failing closed here. Any other
    status must be one this reader itself ever writes.
    """
    if not (
        isinstance(record, dict)
        and isinstance(record.get("symbol"), str)
        and isinstance(record.get("month"), str)
        and isinstance(record.get("kind"), str)
        and isinstance(record.get("url"), str)
    ):
        return False
    if record.get("status") == "present":
        received_time_ns = record.get("received_time_ns")
        return (
            isinstance(record.get("raw_relative_path"), str)
            and isinstance(record.get("raw_sha256"), str)
            and isinstance(received_time_ns, int)
            and not isinstance(received_time_ns, bool)
        )
    return record.get("status") in ("absent", "absent_after_discovery")


def _rewrite_progress_file(
    progress_path: Path,
    *,
    need_header: bool,
    capture_parameters: dict[str, object],
    sources: list[dict[str, object]],
) -> None:
    """Rewrite the progress file to exactly the records recovered from it.

    Called once per resume, and only when the file's tail was not cleanly
    newline-terminated. This replaces the file outright rather than
    patching it in place, so any abandoned, unterminated bytes at the old
    tail are removed from disk -- not merely separated from what comes
    next by a newline, which would still leave them behind to be read back
    as an unparseable line once something is appended after them.
    """
    lines: list[bytes] = []
    if not need_header:
        lines.append(canonical_json({"capture_parameters": capture_parameters}))
    lines.extend(canonical_json(item) for item in sources)
    content = b"\n".join(lines)
    if lines:
        content += b"\n"
    progress_path.write_bytes(content)


def _load_progress(
    progress_path: Path, parameters: dict[str, object]
) -> tuple[list[dict[str, object]], dict[tuple[str, str, str], dict[str, Any]], bool]:
    """Recover `sources`/`already_done`/`need_header` from a prior run's
    progress file, or start empty if there is none.

    Shared, unmodified, by both `capture_panel` and `repair_panel_capture`:
    the recovery, corruption, and torn-write-repair rules are identical
    either way, and `parameters` is simply whatever the caller's own
    resume-binding record is (a fresh capture's `capture_parameters`, or a
    repair's own binding to its source capture).
    """
    sources: list[dict[str, object]] = []
    already_done: dict[tuple[str, str, str], dict[str, Any]] = {}
    need_header = True
    if not progress_path.exists():
        return sources, already_done, need_header
    try:
        raw_text = progress_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise PanelCaptureError(
            f"panel capture progress file is corrupt: {progress_path}"
        ) from error
    non_blank = [line for line in raw_text.splitlines() if line.strip()]
    for index, line in enumerate(non_blank):
        is_last = index == len(non_blank) - 1
        try:
            record: Any = json.loads(line)
        except json.JSONDecodeError as error:
            if is_last:
                # A hard kill can only ever tear the final line -- every
                # earlier one was flushed complete before the next write
                # began. Drop it; its payload will simply be refetched,
                # which is lossless.
                break
            raise PanelCaptureError(
                f"panel capture progress file is corrupt: {progress_path}"
            ) from error
        if index == 0:
            if not (isinstance(record, dict) and "capture_parameters" in record):
                raise PanelCaptureError(f"panel capture progress file is corrupt: {progress_path}")
            if record["capture_parameters"] != parameters:
                raise PanelCaptureError(
                    "panel capture parameters differ from the interrupted run recorded "
                    f"in {progress_path}"
                )
            need_header = False
            continue
        # A record can be syntactically valid JSON yet structurally wrong (a
        # list, a dict missing a required field, or a "present" entry
        # missing the fields only present entries carry) -- that is never
        # reachable from a torn write (truncated JSON never parses at all),
        # but it must still fail closed rather than raise a raw
        # TypeError/KeyError once indexed below or when a resumed "present"
        # source is re-read from disk.
        if not _is_valid_progress_record(record):
            raise PanelCaptureError(f"panel capture progress file is corrupt: {progress_path}")
        sources.append(record)
        already_done[(record["symbol"], record["month"], record["kind"])] = record

    if raw_text and not raw_text.endswith("\n"):
        # The file's tail is not cleanly newline-terminated: either the
        # final record was torn (dropped above) or it was complete JSON that
        # lost only its own trailing newline to the same kind of crash (the
        # JSON and its newline are two separate writes). Rewrite the file to
        # exactly the header and sources recovered above. A bare "append a
        # newline" here would leave the abandoned bytes in place; once
        # anything is appended after them they become an unparseable
        # *earlier* line on the next resume, permanently blocking the
        # capture rather than merely losing one already-lossless refetch.
        _rewrite_progress_file(
            progress_path,
            need_header=need_header,
            capture_parameters=parameters,
            sources=sources,
        )
    return sources, already_done, need_header


def _fill_gap_days(
    missing: dict[str, tuple[int, ...]],
    *,
    target: Path,
    download: PanelFetch,
    sources: list[dict[str, object]],
    already_done: dict[tuple[str, str, str], dict[str, Any]],
    progress_handle: TextIO,
) -> list[PanelCandleRow]:
    """Fetch every missing interior day from the daily kline dumps and
    return the resulting candle rows.

    Shared by both entry points: a fresh capture calls this once, right
    after its monthly fetch loop, against the gaps in what it just
    downloaded; `repair_panel_capture` calls it against the gaps in a
    capture it copied rather than fetched. Either way, `download` already
    carries the retry/failure behaviour (`PanelZipClient`, or an injected
    test double) -- this function adds only the daily-dump URL, the
    (symbol, date, kind) progress key, and the absent/present recording a
    404 here is a genuine absence (the day never traded, or Binance itself
    never published it), so it is recorded exactly the way an absent
    monthly source is recorded, and capture continues rather than aborts.
    Anything else `download` raises propagates unchanged.
    """
    filled: list[PanelCandleRow] = []
    for symbol in sorted(missing):
        for day_open_time_ns in missing[symbol]:
            date = _date_from_open_time_ns(day_open_time_ns)
            key = (symbol, date, _DAILY_FILL_KIND)
            done = already_done.get(key)
            if done is not None:
                if done.get("status") == "present":
                    relative = str(done["raw_relative_path"])
                    raw_bytes = (target / relative).read_bytes()
                    if hashlib.sha256(raw_bytes).hexdigest() != done.get("raw_sha256"):
                        raise PanelCaptureError(f"resumed raw payload hash mismatch: {relative}")
                    payload = PanelPayload(
                        url=str(done["url"]),
                        raw_bytes=raw_bytes,
                        received_time_ns=int(done["received_time_ns"]),
                    )
                    filled.append(
                        _parse_single_daily_fill_row(
                            payload, symbol=symbol, expected_open_time_ns=day_open_time_ns
                        )
                    )
                continue
            url = build_daily_kline_zip_url(symbol, date)
            try:
                payload = download(url)
            except PanelSourceAbsent:
                absent_record: dict[str, object] = {
                    "symbol": symbol,
                    "month": date,
                    "kind": _DAILY_FILL_KIND,
                    "url": url,
                    "status": "absent",
                }
                sources.append(absent_record)
                progress_handle.write(canonical_json(absent_record).decode("utf-8"))
                progress_handle.write("\n")
                progress_handle.flush()
                continue
            _validate_panel_url(payload.url)
            relative = f"raw/{symbol}/{_DAILY_FILL_KIND}-{date}.zip"
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload.raw_bytes)
            present_record: dict[str, object] = {
                "symbol": symbol,
                "month": date,
                "kind": _DAILY_FILL_KIND,
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
            filled.append(
                _parse_single_daily_fill_row(
                    payload, symbol=symbol, expected_open_time_ns=day_open_time_ns
                )
            )
    return filled


def _parse_single_daily_fill_row(
    payload: PanelPayload, *, symbol: str, expected_open_time_ns: int
) -> PanelCandleRow:
    rows = parse_kline_zip(payload, symbol=symbol)
    if len(rows) != 1 or rows[0].open_time_ns != expected_open_time_ns:
        raise PanelCaptureError(
            f"daily kline dump for {symbol} did not contain exactly the requested day "
            f"{_date_from_open_time_ns(expected_open_time_ns)}"
        )
    return rows[0]


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


def _validate_date(date: str) -> None:
    if not _DATE_PATTERN.match(date):
        raise PanelCaptureError(f"invalid date: {date}")


def _date_from_open_time_ns(open_time_ns: int) -> str:
    return datetime.fromtimestamp(open_time_ns / 1_000_000_000, tz=UTC).date().isoformat()
