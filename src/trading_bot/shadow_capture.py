"""Weekly shadow capture: a base capture extended through the last Sunday.

The strategy's data comes from Binance's monthly dumps, which appear weeks
after the fact, so a weekly shadow book cannot run on them alone. A shadow
capture is an ordinary panel capture directory -- `capture-manifest.json`,
`raw/`, `dataset/` -- whose `sources` are the union of a verified base
capture's rows, the previous week's still-uncovered tail rows, and this week's
new tail: one daily kline dump per symbol and day, and, on the perpetual
market, one funding-history REST window per symbol. Because the format and the
seal are the panel's own, `panel_capture.verify_panel_capture` and
`capture_lineage.verify_capture_superset` apply to it unchanged, and every
weekly capture is a verified superset of the base it extends.

Nothing here refetches a source the base already carries: base payloads are
hardlinked (or copied) and their manifest rows carried verbatim, so their
`raw_sha256` is by construction the base's own. Reconciliation of a daily-tail
bar against the monthly dump that later covers it is a separate step; this
builder records an empty `reconciliation` block and drops the previous week's
rows whose month the base now covers.
"""

import hashlib
import json
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date as date_type
from datetime import timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.panel_capture import (
    _CAPTURE_VERSION,
    _DAILY_FILL_KIND,
    _MONTH_PATTERN,
    _VENUE,
    PanelFetch,
    PanelPayload,
    PanelSourceAbsent,
    _validate_panel_url,
    _validate_symbol,
    build_daily_kline_zip_url,
    parse_funding_zip,
    parse_kline_zip,
    verify_panel_capture,
)
from trading_bot.panel_dataset import (
    DAY_NS,
    PanelCandleRow,
    PanelFundingRow,
    publish_panel_dataset,
)
from trading_bot.storage import StoragePolicy

# The two source kinds a shadow capture adds to the panel format. Both store
# their date (or date window) in the manifest's `month` field, exactly as
# `panel_capture`'s own daily gap fill stores its date there, so the lineage
# check keys on (kind, symbol, month) without knowing about either of them.
DAILY_TAIL_KIND = "klines_daily_tail"
FUNDING_REST_KIND = "fundingRate_rest"

_MONTHLY_KINDS = frozenset({"klines", "fundingRate"})
# Which half of `discovered_months` each source kind contributes to.
_MONTH_BUCKET = {
    "klines": "klines",
    _DAILY_FILL_KIND: "klines",
    DAILY_TAIL_KIND: "klines",
    "fundingRate": "fundingRate",
    FUNDING_REST_KIND: "fundingRate",
}
_EPOCH = date_type(1970, 1, 1)
_SUNDAY = 6  # `date.weekday()` counts from Monday
_FUNDING_REST_LIMIT = 1000
# What a lone settlement's interval is taken to be: Binance's standard eight
# hours. With one row there is no spacing to read it off, and a weekly tail
# window that holds a single settlement is the normal, not the odd, case.
_DEFAULT_FUNDING_INTERVAL_HOURS = 8
_MILLISECONDS_PER_HOUR = 3_600_000
_MILLISECONDS_PER_DAY = 86_400_000
_NANOSECONDS_PER_MILLISECOND = 1_000_000
_NANOSECONDS_PER_HOUR = 3_600_000_000_000
_SUNDAY_RETRY_SECONDS = 3600
_WORST_CASE_REQUIRED_BYTES = 500_000_000


class ShadowCaptureError(RuntimeError):
    """Raised when a weekly shadow capture fails closed."""


@dataclass(frozen=True, slots=True)
class ShadowCaptureArtifact:
    capture_root: Path
    capture_root_hash: str
    dataset_root_hash: str
    tail_through: str
    reconciliation: dict[str, object]


def build_funding_rest_url(symbol: str, *, start_time_ms: int, end_time_ms: int) -> str:
    """The funding-history endpoint for one symbol over one closed window.

    Binance answers at most `_FUNDING_REST_LIMIT` settlements per request; a
    weekly tail holds at most a few dozen, so one request always covers the
    whole span and no pagination cursor has to be trusted.
    """
    _validate_symbol(symbol)
    if start_time_ms < 0 or end_time_ms < start_time_ms:
        raise ShadowCaptureError(
            f"invalid funding window for {symbol}: {start_time_ms}..{end_time_ms}"
        )
    url = (
        f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}"
        f"&startTime={start_time_ms}&endTime={end_time_ms}&limit={_FUNDING_REST_LIMIT}"
    )
    _validate_panel_url(url)
    return url


def parse_funding_rest(
    payload: PanelPayload,
    *,
    symbol: str,
    start_time_ms: int,
    end_time_ms: int,
    venue: str = _VENUE,
) -> tuple[PanelFundingRow, ...]:
    """The settlements of a `/fapi/v1/fundingRate` answer, as panel rows.

    The endpoint's window is inclusive at both ends but Binance has been seen
    to answer with a settlement just outside it; only settlements inside
    [start, end] become rows, and the interval is read off the spacing of
    those rows alone. The monthly dumps carry an explicit
    `funding_interval_hours` column and the REST answer does not, so the
    interval is derived: each settlement takes the gap to the next one, the
    last takes the gap before it, and a lone settlement takes the venue's
    standard eight hours. A repeated `fundingTime` is a malformed answer, not
    a settlement that happened twice.
    """
    try:
        document = json.loads(payload.raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ShadowCaptureError(f"funding REST answer for {symbol} is unreadable") from error
    if not isinstance(document, list):
        raise ShadowCaptureError(f"funding REST answer for {symbol} is not a JSON array")
    rates: dict[int, Decimal] = {}
    for entry in document:
        calc_time_ms, rate = _funding_rest_settlement(entry, symbol=symbol)
        if calc_time_ms in rates:
            raise ShadowCaptureError(
                f"funding REST answer for {symbol} repeats fundingTime {calc_time_ms}"
            )
        rates[calc_time_ms] = rate
    inside = sorted(time_ms for time_ms in rates if start_time_ms <= time_ms <= end_time_ms)
    return tuple(
        PanelFundingRow(
            venue=venue,
            instrument_id=symbol,
            calc_time_ns=calc_time_ms * _NANOSECONDS_PER_MILLISECOND,
            funding_interval_hours=_funding_interval_hours(inside, index),
            rate=rates[calc_time_ms],
        )
        for index, calc_time_ms in enumerate(inside)
    )


def _funding_rest_settlement(entry: object, *, symbol: str) -> tuple[int, Decimal]:
    """One REST row's settlement time and rate, or a refusal."""
    if not isinstance(entry, dict):
        raise ShadowCaptureError(f"funding REST answer for {symbol} holds a non-object row")
    if entry.get("symbol") != symbol:
        raise ShadowCaptureError(
            f"funding REST answer for {symbol} holds a row for {entry.get('symbol')}"
        )
    calc_time_ms = entry.get("fundingTime")
    # `isinstance(True, int)` is true and a bool is never a timestamp.
    if not isinstance(calc_time_ms, int) or isinstance(calc_time_ms, bool):
        raise ShadowCaptureError(f"funding REST row for {symbol} has no integer fundingTime")
    raw_rate = entry.get("fundingRate")
    if not isinstance(raw_rate, str):
        # Binance sends the rate as a decimal string; a JSON number would have
        # gone through a binary float on the way in and is refused, not repaired.
        raise ShadowCaptureError(f"funding REST row for {symbol} has no string fundingRate")
    try:
        rate = Decimal(raw_rate)
    except InvalidOperation as error:
        raise ShadowCaptureError(
            f"funding REST row for {symbol} has an unreadable fundingRate: {raw_rate}"
        ) from error
    return calc_time_ms, rate


def _funding_interval_hours(settlements: list[int], index: int) -> int:
    if len(settlements) == 1:
        return _DEFAULT_FUNDING_INTERVAL_HOURS
    if index + 1 < len(settlements):
        return (settlements[index + 1] - settlements[index]) // _MILLISECONDS_PER_HOUR
    return (settlements[index] - settlements[index - 1]) // _MILLISECONDS_PER_HOUR


def build_shadow_capture(
    *,
    workspace_root: Path,
    base_capture_root: Path,
    output_directory: Path,
    reserve_bytes: int,
    tail_through: str,
    market: str,
    fetch: PanelFetch,
    previous_capture_root: Path | None = None,
    clock: Callable[[], int] = time.time_ns,
    sleep: Callable[[float], None] = time.sleep,
    sunday_deadline_hours: int = 24,
) -> ShadowCaptureArtifact:
    """Publish one weekly shadow capture covering the base through `tail_through`.

    `clock` and `sleep` are injected only so the Sunday deadline of spec
    section 3.3 can be exercised without waiting a real day.
    """
    tail_day = _parse_date(tail_through)
    base_root = base_capture_root.resolve()
    base_manifest = _verified_manifest(base_root, label="base")
    previous_root = (
        previous_capture_root.resolve() if previous_capture_root is not None else None
    )
    previous_manifest = (
        _verified_manifest(previous_root, label="previous") if previous_root is not None else None
    )

    base_market = str(base_manifest.get("market", "um"))
    if base_market != market:
        raise ShadowCaptureError(
            f"base capture is a {base_market} capture, not a {market} one"
        )
    # The venue string is the base's, not this module's: a spot capture's rows
    # have to stay on the spot venue or the extended capture stops being a
    # superset of what it extends.
    venue = str(base_manifest["venue"])
    symbols = tuple(_string_list(base_manifest, "symbols"))
    base_sources = _source_entries(base_manifest, label="base")

    last_base_month = _last_covered_month(base_sources)
    first_tail_day = _first_day_after_month(last_base_month)
    if tail_day < first_tail_day:
        raise ShadowCaptureError(
            f"tail_through {tail_through} does not reach past the base capture's "
            f"last month {last_base_month}"
        )

    workspace = workspace_root.resolve()
    target = output_directory.resolve()
    StoragePolicy(workspace, reserve_bytes).authorize(
        target=target,
        temporary_directory=target.parent,
        free_bytes=shutil.disk_usage(workspace).free,
        worst_case_required_bytes=_WORST_CASE_REQUIRED_BYTES,
    )
    manifest_path = target / "capture-manifest.json"
    if manifest_path.exists():
        raise ShadowCaptureError("shadow capture already exists and is immutable")
    target.mkdir(parents=True, exist_ok=True)

    sources: list[dict[str, object]] = []
    candles: list[PanelCandleRow] = []
    funding: list[PanelFundingRow] = []
    # Per symbol, every bar open time the capture already holds -- the Sunday
    # rule needs to know whether a symbol has a Saturday bar.
    open_times: dict[str, set[int]] = {}

    for entry in base_sources:
        _carry_source(
            entry,
            source_root=base_root,
            target=target,
            venue=venue,
            sources=sources,
            candles=candles,
            funding=funding,
            open_times=open_times,
        )

    carried_tail_day: dict[str, date_type] = {}
    carried_rest_day: dict[str, date_type] = {}
    if previous_manifest is not None and previous_root is not None:
        for entry in _source_entries(previous_manifest, label="previous"):
            kind = str(entry.get("kind"))
            if kind not in (DAILY_TAIL_KIND, FUNDING_REST_KIND):
                continue
            span_start, span_end = _row_span(kind, str(entry.get("month")))
            if _month_of(span_start) <= last_base_month:
                # The base's own monthly dump now covers this row's month.
                # Comparing the two is Task 2's reconciliation; until it
                # exists the monthly dump simply wins and the row is dropped.
                continue
            _carry_source(
                entry,
                source_root=previous_root,
                target=target,
                venue=venue,
                sources=sources,
                candles=candles,
                funding=funding,
                open_times=open_times,
            )
            symbol = str(entry.get("symbol"))
            reached = carried_tail_day if kind == DAILY_TAIL_KIND else carried_rest_day
            reached[symbol] = max(span_end, reached.get(symbol, span_end))

    deadline_ns = clock() + sunday_deadline_hours * _NANOSECONDS_PER_HOUR
    for symbol in symbols:
        _fetch_daily_tail(
            symbol=symbol,
            first_day=_next_day_or(carried_tail_day.get(symbol), first_tail_day),
            tail_day=tail_day,
            market=market,
            venue=venue,
            fetch=fetch,
            clock=clock,
            sleep=sleep,
            deadline_ns=deadline_ns,
            target=target,
            sources=sources,
            candles=candles,
            open_times=open_times,
        )
        if market != "um":
            # Funding exists only on the perpetual market, so a spot shadow
            # capture never reaches for the funding endpoint at all.
            continue
        _fetch_funding_window(
            symbol=symbol,
            first_day=_next_day_or(carried_rest_day.get(symbol), first_tail_day),
            tail_day=tail_day,
            venue=venue,
            fetch=fetch,
            target=target,
            sources=sources,
            funding=funding,
        )

    dataset = publish_panel_dataset(
        tuple(candles),
        tuple(funding),
        output_directory=target / "dataset",
        raw_source_hashes=tuple(
            str(item["raw_sha256"]) for item in sources if item.get("status") == "present"
        ),
    )
    reconciliation: dict[str, object] = {"compared_rows": 0, "mismatches": []}
    material: dict[str, object] = {
        "capture_version": _CAPTURE_VERSION,
        "market": market,
        "venue": venue,
        "interval": "1d",
        "symbols": list(symbols),
        "months": None,
        "sources": sources,
        "dataset_root_hash": dataset.root_hash,
        "discovered_months": _discovered_months(sources),
        "base_capture_root_hash": str(base_manifest["capture_root_hash"]),
        "previous_capture_root_hash": (
            str(previous_manifest["capture_root_hash"])
            if previous_manifest is not None
            else None
        ),
        "tail_through": tail_through,
        "reconciliation": reconciliation,
    }
    capture_root_hash = content_sha256(material)
    document = dict(material)
    document["capture_root_hash"] = capture_root_hash
    manifest_path.write_bytes(canonical_json(document))
    return ShadowCaptureArtifact(
        capture_root=target,
        capture_root_hash=capture_root_hash,
        dataset_root_hash=dataset.root_hash,
        tail_through=tail_through,
        # A copy: the manifest's own block is already sealed into
        # `capture_root_hash` and must not be reachable through the artifact.
        reconciliation=dict(reconciliation),
    )


def _fetch_daily_tail(
    *,
    symbol: str,
    first_day: date_type,
    tail_day: date_type,
    market: str,
    venue: str,
    fetch: PanelFetch,
    clock: Callable[[], int],
    sleep: Callable[[float], None],
    deadline_ns: int,
    target: Path,
    sources: list[dict[str, object]],
    candles: list[PanelCandleRow],
    open_times: dict[str, set[int]],
) -> None:
    """Fetch one symbol's daily dumps for every day of its tail.

    A 404 is an absence, recorded exactly the way `capture_panel` records one
    and then stepped over -- except on the decision Sunday itself, where spec
    section 3.3's rule applies: a symbol that traded on the Saturday must have
    a Sunday bar, so the fetch waits for the dump rather than publishing a
    capture that silently misses the week's close.
    """
    saturday_open_time_ns = _open_time_ns(tail_day - timedelta(days=1))
    for day in _plan_tail_days(first_day=first_day, tail_day=tail_day):
        date = day.isoformat()
        url = build_daily_kline_zip_url(symbol, date, market=market)
        waiting = (
            day == tail_day
            and tail_day.weekday() == _SUNDAY
            and saturday_open_time_ns in open_times.get(symbol, set())
        )
        payload = _fetch_daily_dump(
            url,
            symbol=symbol,
            waiting=waiting,
            fetch=fetch,
            clock=clock,
            sleep=sleep,
            deadline_ns=deadline_ns,
        )
        if payload is None:
            sources.append(
                {
                    "symbol": symbol,
                    "month": date,
                    "kind": DAILY_TAIL_KIND,
                    "url": url,
                    "status": "absent",
                }
            )
            continue
        _validate_panel_url(payload.url)
        rows = parse_kline_zip(payload, symbol=symbol, venue=venue)
        _require_exactly_one_day(rows, symbol=symbol, day=day)
        relative = f"raw/{symbol}/{DAILY_TAIL_KIND}-{date}.zip"
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload.raw_bytes)
        sources.append(
            {
                "symbol": symbol,
                "month": date,
                "kind": DAILY_TAIL_KIND,
                "url": payload.url,
                "received_time_ns": payload.received_time_ns,
                "raw_relative_path": relative,
                "raw_sha256": hashlib.sha256(payload.raw_bytes).hexdigest(),
                "status": "present",
            }
        )
        candles.extend(rows)
        open_times.setdefault(symbol, set()).update(row.open_time_ns for row in rows)


def _fetch_funding_window(
    *,
    symbol: str,
    first_day: date_type,
    tail_day: date_type,
    venue: str,
    fetch: PanelFetch,
    target: Path,
    sources: list[dict[str, object]],
    funding: list[PanelFundingRow],
) -> None:
    """Fetch one symbol's funding history for the whole tail in one request.

    Funding settles several times a day, so a per-day request would be a
    dozen times the traffic for the same rows; the window is recorded in the
    manifest's `month` field as `<from>_<to>` so a reader can tell exactly
    which days one stored answer speaks for.
    """
    if first_day > tail_day:
        return
    start_time_ms = _day_start_ms(first_day)
    end_time_ms = _day_end_ms(tail_day)
    url = build_funding_rest_url(symbol, start_time_ms=start_time_ms, end_time_ms=end_time_ms)
    payload = fetch(url)
    _validate_panel_url(payload.url)
    rows = parse_funding_rest(
        payload,
        symbol=symbol,
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
        venue=venue,
    )
    window = f"{first_day.isoformat()}_{tail_day.isoformat()}"
    relative = f"raw/{symbol}/{FUNDING_REST_KIND}-{window}.json"
    path = target / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload.raw_bytes)
    sources.append(
        {
            "symbol": symbol,
            "month": window,
            "kind": FUNDING_REST_KIND,
            "url": payload.url,
            "received_time_ns": payload.received_time_ns,
            "raw_relative_path": relative,
            "raw_sha256": hashlib.sha256(payload.raw_bytes).hexdigest(),
            "status": "present",
        }
    )
    funding.extend(rows)


def _fetch_daily_dump(
    url: str,
    *,
    symbol: str,
    waiting: bool,
    fetch: PanelFetch,
    clock: Callable[[], int],
    sleep: Callable[[float], None],
    deadline_ns: int,
) -> PanelPayload | None:
    """One daily dump, `None` when it is absent upstream and may stay absent.

    When `waiting` is set the dump is the decision Sunday's for a symbol that
    traded the day before: Binance publishes a day's dump the next day, so a
    miss is retried hourly and only becomes a refusal once the deadline has
    passed on `clock`.
    """
    while True:
        try:
            payload = fetch(url)
        except PanelSourceAbsent:
            payload = None
        if payload is not None:
            return payload
        if not waiting:
            return None
        if clock() >= deadline_ns:
            raise ShadowCaptureError(f"SUNDAY_DUMP_MISSING:{symbol}")
        sleep(_SUNDAY_RETRY_SECONDS)


def _carry_source(
    entry: dict[str, Any],
    *,
    source_root: Path,
    target: Path,
    venue: str,
    sources: list[dict[str, object]],
    candles: list[PanelCandleRow],
    funding: list[PanelFundingRow],
    open_times: dict[str, set[int]],
) -> None:
    """Carry one already-captured source row into the new capture.

    The row is copied into the manifest verbatim -- same url, same
    `received_time_ns`, same `raw_sha256` -- because a shadow capture claims
    to carry what its base carried, not to have fetched it again. Its rows are
    re-derived from the copied payload with the same parsers the original
    capture used.
    """
    record = dict(entry)
    sources.append(record)
    if record.get("status") != "present":
        return
    _copy_raw_payload(
        source_root=source_root,
        target=target,
        relative=str(entry["raw_relative_path"]),
        expected_sha256=str(entry["raw_sha256"]),
    )
    new_candles, new_funding = _rows_from_source(entry, capture_root=target, venue=venue)
    candles.extend(new_candles)
    funding.extend(new_funding)
    open_times.setdefault(str(entry["symbol"]), set()).update(
        row.open_time_ns for row in new_candles
    )


def _copy_raw_payload(
    *, source_root: Path, target: Path, relative: str, expected_sha256: str
) -> None:
    """Put a captured payload under the new capture without changing a byte.

    A hardlink saves a full second copy of the base every week; a filesystem
    that refuses one -- a different volume, a link-count limit, a share that
    has no links -- falls back to a real copy. Either way the result is hashed
    again, because the new manifest is about to claim this digest for it.
    """
    source = source_root / relative
    destination = target / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)
    if hashlib.sha256(destination.read_bytes()).hexdigest() != expected_sha256:
        raise ShadowCaptureError(f"copied raw payload hash mismatch: {relative}")


def _rows_from_source(
    entry: dict[str, Any], *, capture_root: Path, venue: str
) -> tuple[tuple[PanelCandleRow, ...], tuple[PanelFundingRow, ...]]:
    """Re-derive one present source row's candle and funding rows."""
    symbol = str(entry["symbol"])
    kind = str(entry["kind"])
    payload = PanelPayload(
        url=str(entry["url"]),
        raw_bytes=(capture_root / str(entry["raw_relative_path"])).read_bytes(),
        received_time_ns=int(entry["received_time_ns"]),
    )
    if kind in ("klines", _DAILY_FILL_KIND, DAILY_TAIL_KIND):
        return parse_kline_zip(payload, symbol=symbol, venue=venue), ()
    if kind == "fundingRate":
        return (), parse_funding_zip(payload, symbol=symbol, venue=venue)
    if kind == FUNDING_REST_KIND:
        start_day, end_day = _rest_window_days(str(entry["month"]))
        return (), parse_funding_rest(
            payload,
            symbol=symbol,
            start_time_ms=_day_start_ms(start_day),
            end_time_ms=_day_end_ms(end_day),
            venue=venue,
        )
    raise ShadowCaptureError(f"unknown capture source kind: {kind}")


def _require_exactly_one_day(
    rows: tuple[PanelCandleRow, ...], *, symbol: str, day: date_type
) -> None:
    """A daily dump must hold the one day it was asked for and nothing else."""
    if len(rows) != 1 or rows[0].open_time_ns != _open_time_ns(day):
        raise ShadowCaptureError(
            f"daily kline dump for {symbol} did not contain exactly {day.isoformat()}"
        )


def _discovered_months(sources: list[dict[str, object]]) -> dict[str, dict[str, list[str]]]:
    """Per symbol, the months this capture covers, klines and funding apart.

    Derived from the rows rather than copied from the base's `discovered_months`:
    a base built from an explicit month list carries none at all, and where a
    base does carry one the two agree by construction -- a discovered month is
    exactly a month that capture then fetched a source row for.
    """
    months: dict[str, dict[str, set[str]]] = {}
    for entry in sources:
        kind = str(entry.get("kind"))
        bucket = _MONTH_BUCKET.get(kind)
        if bucket is None:
            continue
        buckets = months.setdefault(
            str(entry.get("symbol")), {"klines": set(), "fundingRate": set()}
        )
        buckets[bucket].update(_months_of(kind, str(entry.get("month"))))
    return {
        symbol: {bucket: sorted(values) for bucket, values in sorted(buckets.items())}
        for symbol, buckets in sorted(months.items())
    }


def _months_of(kind: str, month_field: str) -> tuple[str, ...]:
    """Every month one source row speaks for."""
    if kind in _MONTHLY_KINDS:
        return (month_field,)
    if kind == FUNDING_REST_KIND:
        start_day, end_day = _rest_window_days(month_field)
        return _months_between(start_day, end_day)
    return (_month_of(_parse_date(month_field)),)


def _row_span(kind: str, month_field: str) -> tuple[date_type, date_type]:
    """The first and last calendar day one tail source row speaks for."""
    if kind == FUNDING_REST_KIND:
        return _rest_window_days(month_field)
    day = _parse_date(month_field)
    return day, day


def _rest_window_days(month_field: str) -> tuple[date_type, date_type]:
    start_text, separator, end_text = month_field.partition("_")
    if not separator:
        raise ShadowCaptureError(f"malformed funding REST window: {month_field}")
    return _parse_date(start_text), _parse_date(end_text)


def _last_covered_month(entries: list[dict[str, Any]]) -> str:
    """The last whole month the base capture covers from the monthly dumps."""
    months = sorted(
        str(entry.get("month"))
        for entry in entries
        if str(entry.get("kind")) in _MONTHLY_KINDS
        and _MONTH_PATTERN.match(str(entry.get("month"))) is not None
    )
    if not months:
        raise ShadowCaptureError("base capture covers no whole month to extend")
    return months[-1]


def _verified_manifest(capture_root: Path, *, label: str) -> dict[str, Any]:
    valid, reasons = verify_panel_capture(capture_root)
    if not valid:
        raise ShadowCaptureError(
            f"{label} capture failed verification: " + ",".join(reasons)
        )
    try:
        document = json.loads(
            (capture_root / "capture-manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ShadowCaptureError(
            f"{label} capture manifest is unreadable: {capture_root}"
        ) from error
    if not isinstance(document, dict):
        raise ShadowCaptureError(f"{label} capture manifest is malformed: {capture_root}")
    return document


def _source_entries(manifest: dict[str, Any], *, label: str) -> list[dict[str, Any]]:
    sources = manifest.get("sources")
    if not isinstance(sources, list):
        raise ShadowCaptureError(f"{label} capture manifest is malformed")
    entries: list[dict[str, Any]] = []
    for item in sources:
        if not isinstance(item, dict):
            raise ShadowCaptureError(f"{label} capture manifest is malformed")
        entries.append(item)
    return entries


def _string_list(manifest: dict[str, Any], key: str) -> list[str]:
    values = manifest.get(key)
    if not isinstance(values, list) or not values:
        raise ShadowCaptureError(f"base capture manifest has no {key}")
    return [str(item) for item in values]


def _parse_date(value: str) -> date_type:
    try:
        parsed = date_type.fromisoformat(value)
    except ValueError as error:
        raise ShadowCaptureError(f"invalid date: {value}") from error
    if parsed.isoformat() != value:
        # `fromisoformat` also accepts compact forms like "20200802"; the
        # manifest's dates are the dumps' own, always YYYY-MM-DD.
        raise ShadowCaptureError(f"invalid date: {value}")
    return parsed


def _plan_tail_days(*, first_day: date_type, tail_day: date_type) -> tuple[date_type, ...]:
    """Every calendar day from `first_day` through `tail_day`, inclusive."""
    if tail_day < first_day:
        return ()
    return tuple(
        first_day + timedelta(days=offset)
        for offset in range((tail_day - first_day).days + 1)
    )


def _next_day_or(reached: date_type | None, fallback: date_type) -> date_type:
    """The day after what is already covered, or where coverage has to start."""
    return fallback if reached is None else reached + timedelta(days=1)


def _first_day_after_month(month: str) -> date_type:
    year, _, number = month.partition("-")
    if int(number) == 12:
        return date_type(int(year) + 1, 1, 1)
    return date_type(int(year), int(number) + 1, 1)


def _months_between(first_day: date_type, last_day: date_type) -> tuple[str, ...]:
    months: list[str] = []
    cursor = first_day.replace(day=1)
    while cursor <= last_day:
        months.append(_month_of(cursor))
        cursor = _first_day_after_month(_month_of(cursor))
    return tuple(months)


def _month_of(day: date_type) -> str:
    return day.isoformat()[:7]


def _open_time_ns(day: date_type) -> int:
    return (day - _EPOCH).days * DAY_NS


def _day_start_ms(day: date_type) -> int:
    return (day - _EPOCH).days * _MILLISECONDS_PER_DAY


def _day_end_ms(day: date_type) -> int:
    return _day_start_ms(day) + _MILLISECONDS_PER_DAY - 1
