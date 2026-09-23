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
`raw_sha256` is by construction the base's own.

When a monthly dump finally covers a month the tail already spoke for, the
monthly dump wins and the previous week's rows for that month are dropped --
but only after every one of them has been compared against the base's own row
for the same key (spec section 3.2). Binance generates the monthly dumps from
the same data as the daily dumps and the funding endpoint, so a disagreement
is evidence of a data problem, not noise: the capture is refused with
`RECONCILIATION_MISMATCH` and a `reconciliation-refused.json` is written
beside the unpublished output for a person to read. A clean comparison is
recorded in the manifest's `reconciliation` block, inside the seal.

A real base does not cover the same month for every symbol: delisted
contracts stop years early and funding dumps lag klines. The tail therefore
starts per symbol and per source bucket -- a symbol one month behind the
furthest one is simply lagging and its tail starts a month earlier, while a
symbol further behind is named in the manifest's `stale_symbols` rather than
having years of daily dumps refetched every week. Each week plans its own
span from its own base and subtracts what the previous week already carried,
so a month that arrived since last week is fetched rather than stepped over.
"""

import hashlib
import json
import os
import shutil
import time
from collections.abc import Callable, Collection
from dataclasses import dataclass
from datetime import date as date_type
from datetime import timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, NoReturn

from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.panel_capture import (
    _CAPTURE_VERSION,
    _DAILY_FILL_KIND,
    _MONTH_PATTERN,
    _VENUE,
    PanelCaptureError,
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
# How far a measured settlement spacing may sit off a whole hour before it
# stops being a funding interval: a minute, which is far more than venue
# jitter and far less than the smallest real interval.
_FUNDING_INTERVAL_TOLERANCE_MS = 60_000
_MILLISECONDS_PER_HOUR = 3_600_000
_MILLISECONDS_PER_DAY = 86_400_000
_NANOSECONDS_PER_MILLISECOND = 1_000_000
_NANOSECONDS_PER_HOUR = 3_600_000_000_000
_SUNDAY_RETRY_SECONDS = 3600
_WORST_CASE_REQUIRED_BYTES = 500_000_000

# What reconciliation compares on a bar, in the order a refusal lists it.
# `available_time_ns` is derived from the close time and `interval_ns` is the
# same constant on both sides, while `source_payload_hash` is the digest of
# the payload the row came out of and so differs by construction: the daily
# dump and the monthly dump are two different files saying the same thing,
# which is exactly what is being checked.
_CANDLE_FIELDS = (
    "close_time_ns",
    "open",
    "high",
    "low",
    "close",
    "base_volume",
    "quote_volume",
    "trade_count",
)
# A settlement is a time and a rate. The REST answer carries no interval
# column and `parse_funding_rest` infers one from the spacing, so comparing
# `funding_interval_hours` would weigh that inference against the monthly
# dump's own measurement and call the difference a data problem.
_FUNDING_FIELDS = ("rate",)
_RECONCILIATION_REFUSAL_SUFFIX = "-reconciliation-refused.json"


class ShadowCaptureError(RuntimeError):
    """Raised when a weekly shadow capture fails closed."""


class ShadowCaptureTransportError(ShadowCaptureError):
    """Raised when a source could not be fetched at all (ruling 21).

    The one failure here that another attempt may fix. Everything else this
    module refuses with is a fact about the inputs -- a base of the wrong
    market, a dump carrying the wrong day, a tail row that disagrees with the
    monthly dump -- and a supervisor that retried those would hit the same
    refusal every hour. A dropped connection, a 5xx and a dump host that has
    not published an hour's file yet are the venue's weather, so they are
    named apart and the CLI maps this subclass, and only this subclass, to the
    exit code that asks for another attempt.
    """


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
    if start_time_ms < 0 or end_time_ms < start_time_ms:
        raise ShadowCaptureError(
            f"invalid funding window for {symbol}: {start_time_ms}..{end_time_ms}"
        )
    url = (
        f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}"
        f"&startTime={start_time_ms}&endTime={end_time_ms}&limit={_FUNDING_REST_LIMIT}"
    )
    try:
        # `panel_capture` owns both checks; its refusals are re-raised as this
        # module's own so that everything this module's surface raises is a
        # `ShadowCaptureError`.
        _validate_symbol(symbol)
        _validate_panel_url(url)
    except PanelCaptureError as error:
        raise ShadowCaptureError(f"invalid funding request for {symbol}: {error}") from error
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
    if len(document) >= _FUNDING_REST_LIMIT:
        # At the row limit the venue may have cut the window short, and a
        # silently short funding history is a wrong funding total, not a gap
        # anything downstream would notice.
        raise ShadowCaptureError(
            f"funding REST answer for {symbol} over {start_time_ms}..{end_time_ms} reached "
            f"the {_FUNDING_REST_LIMIT}-row limit and may be truncated"
        )
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
            funding_interval_hours=_funding_interval_hours(inside, index, symbol=symbol),
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


def _funding_interval_hours(settlements: list[int], index: int, *, symbol: str) -> int:
    """The measured spacing around one settlement, in whole hours.

    A funding interval is a whole number of hours by construction (eight on
    most contracts, four on some), so the measured delta is rounded to the
    nearest hour rather than floored -- a settlement a few seconds late must
    not turn an eight-hour interval into seven. Anything that is not within a
    minute of a whole positive hour is not a funding interval at all, and is
    refused rather than recorded as an approximation the accounting would
    then multiply through.
    """
    if len(settlements) == 1:
        return _DEFAULT_FUNDING_INTERVAL_HOURS
    if index + 1 < len(settlements):
        delta_ms = settlements[index + 1] - settlements[index]
    else:
        delta_ms = settlements[index] - settlements[index - 1]
    hours = (delta_ms + _MILLISECONDS_PER_HOUR // 2) // _MILLISECONDS_PER_HOUR
    if hours <= 0:
        raise ShadowCaptureError(
            f"funding settlements for {symbol} are {delta_ms} ms apart, less than an hour"
        )
    drift_ms = abs(delta_ms - hours * _MILLISECONDS_PER_HOUR)
    if drift_ms > _FUNDING_INTERVAL_TOLERANCE_MS:
        raise ShadowCaptureError(
            f"funding settlements for {symbol} are {delta_ms} ms apart, "
            f"not a whole number of hours"
        )
    return hours


def reconcile_tail_rows(
    *,
    previous_rows: tuple[PanelCandleRow | PanelFundingRow, ...],
    base_rows: tuple[PanelCandleRow | PanelFundingRow, ...],
) -> tuple[int, tuple[dict[str, object], ...]]:
    """Compare a week's tail rows against the monthly rows that now cover them.

    Returns `(compared_count, mismatches)`. A candle is matched on
    `(instrument_id, open_time_ns)` and must agree on `_CANDLE_FIELDS`; a
    settlement is matched on `(instrument_id, calc_time_ns)` and must agree on
    its rate. A previous row the base has no counterpart for is itself a
    mismatch, of kind `missing_in_base`: the monthly dump is about to replace
    the tail row, so a key only the tail has would silently disappear.

    Pure: rows in, records out. The caller decides which rows to hand it and
    what a non-empty result means.
    """
    base_candles = {
        (row.instrument_id, row.open_time_ns): row
        for row in base_rows
        if isinstance(row, PanelCandleRow)
    }
    base_funding = {
        (row.instrument_id, row.calc_time_ns): row
        for row in base_rows
        if isinstance(row, PanelFundingRow)
    }
    mismatches: list[dict[str, object]] = []
    for row in previous_rows:
        if isinstance(row, PanelCandleRow):
            counterpart = base_candles.get((row.instrument_id, row.open_time_ns))
            if counterpart is None:
                mismatches.append(_missing_in_base("candle", row.instrument_id, row.open_time_ns))
                continue
            mismatches.extend(_candle_mismatches(row, counterpart))
        else:
            settlement = base_funding.get((row.instrument_id, row.calc_time_ns))
            if settlement is None:
                mismatches.append(_missing_in_base("funding", row.instrument_id, row.calc_time_ns))
                continue
            mismatches.extend(_funding_mismatches(row, settlement))
    return len(previous_rows), tuple(mismatches)


def _candle_mismatches(
    previous: PanelCandleRow, base: PanelCandleRow
) -> tuple[dict[str, object], ...]:
    """One record per measured field of a bar the two sources disagree on."""
    return _field_mismatches(
        "candle", previous.open_time_ns, _CANDLE_FIELDS, previous=previous, base=base
    )


def _funding_mismatches(
    previous: PanelFundingRow, base: PanelFundingRow
) -> tuple[dict[str, object], ...]:
    """One record per measured field of a settlement the two sources disagree on."""
    return _field_mismatches(
        "funding", previous.calc_time_ns, _FUNDING_FIELDS, previous=previous, base=base
    )


def _field_mismatches(
    kind: str,
    key: int,
    field_names: tuple[str, ...],
    *,
    previous: PanelCandleRow | PanelFundingRow,
    base: PanelCandleRow | PanelFundingRow,
) -> tuple[dict[str, object], ...]:
    """One readable, JSON-able record per named field the two rows disagree on.

    Both values are rendered as text: a `Decimal` is not canonical JSON, and a
    person reading the refusal wants the digits a dump actually carried rather
    than a number some later reader might reformat. The comparison itself is
    on the values, so `100` and `100.00` are the same price written twice, not
    a disagreement.
    """
    records: list[dict[str, object]] = []
    for field_name in field_names:
        previous_value = getattr(previous, field_name)
        base_value = getattr(base, field_name)
        if previous_value == base_value:
            continue
        records.append(
            {
                "kind": kind,
                "instrument_id": previous.instrument_id,
                "key": key,
                "field": field_name,
                "previous": str(previous_value),
                "base": str(base_value),
            }
        )
    return tuple(records)


def _missing_in_base(row_kind: str, instrument_id: str, key: int) -> dict[str, object]:
    """A tail row the base's monthly dump has no counterpart for at all.

    `field` carries the row kind because `kind` is spent on naming the
    absence; the pair reads as "the candle at this key: present before,
    absent now".
    """
    return {
        "kind": "missing_in_base",
        "instrument_id": instrument_id,
        "key": key,
        "field": row_kind,
        "previous": "present",
        "base": "absent",
    }


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

    covered = _covered_months(base_sources)
    klines_month = _furthest_month(covered["klines"])
    if klines_month is None:
        raise ShadowCaptureError("base capture covers no whole month to extend")
    # A base with klines but no funding dump at all still gets a funding tail:
    # every symbol is then a stale one and the window starts where the klines
    # tail starts.
    funding_month = _furthest_month(covered["fundingRate"]) or klines_month
    if tail_day < _first_day_after_month(klines_month):
        raise ShadowCaptureError(
            f"tail_through {tail_through} does not reach past the base capture's "
            f"last month {klines_month}"
        )
    daily_start: dict[str, date_type] = {}
    rest_start: dict[str, date_type] = {}
    stale_symbols: dict[str, dict[str, object]] = {"klines": {}}
    if market == "um":
        stale_symbols["fundingRate"] = {}
    for symbol in symbols:
        own_klines_month = covered["klines"].get(symbol)
        daily_start[symbol], klines_stale = _bucket_tail_start(
            own_klines_month, furthest_month=klines_month
        )
        if klines_stale:
            stale_symbols["klines"][symbol] = own_klines_month
        if market != "um":
            continue
        own_funding_month = covered["fundingRate"].get(symbol)
        rest_start[symbol], funding_stale = _bucket_tail_start(
            own_funding_month, furthest_month=funding_month
        )
        if funding_stale:
            stale_symbols["fundingRate"][symbol] = own_funding_month

    # The previous week's tail, split into the rows the base's own monthly
    # dumps now cover -- which this week reconciles and then drops -- and the
    # rows that are still the tail's to carry.
    previous_tail = (
        [
            entry
            for entry in _source_entries(previous_manifest, label="previous")
            if str(entry.get("kind")) in (DAILY_TAIL_KIND, FUNDING_REST_KIND)
        ]
        if previous_manifest is not None
        else []
    )
    carried_previous, dropped_previous = _partition_previous_tail(previous_tail, covered=covered)

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

    # Reconciliation runs here and nowhere later: after the storage policy has
    # authorized the drive the refusal document is written to and after the
    # immutability check, but before a single byte goes under `target`. A
    # refused week therefore leaves no capture directory at all -- no
    # `capture-manifest.json`, no half-copied `raw/` -- so the rerun a person
    # starts once the data problem is understood is not blocked by this one.
    compared_rows = 0
    if dropped_previous and previous_root is not None and previous_manifest is not None:
        compared_rows, mismatches = _reconcile_dropped_rows(
            dropped_previous,
            previous_root=previous_root,
            base_root=base_root,
            base_sources=base_sources,
            venue=venue,
        )
        if mismatches:
            _refuse_reconciliation(
                target=target,
                base_capture_root_hash=str(base_manifest["capture_root_hash"]),
                previous_capture_root_hash=str(previous_manifest["capture_root_hash"]),
                compared_rows=compared_rows,
                mismatches=mismatches,
            )

    # Ruling 21(b): from here on the run writes under `target`, so from
    # here on it owns what it half-wrote. A run interrupted before the
    # manifest is published leaves a directory whose `raw/` already holds
    # payloads, and the rerun a person starts then refuses on the very
    # guard that keeps a published capture immutable -- for ever, with no
    # cure named. So a directory *this* run created is removed again unless
    # this run published the manifest. A directory that was already there
    # is never touched: spec section 6's "nothing is written under raw/
    # twice" is about a published capture, and the one case left names its
    # own cure in `_copy_raw_payload`. The reconciliation refusal is
    # written beside the directory, not inside it, so it survives this.
    created = not target.exists()
    published = False
    try:
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

        # Per symbol and bucket, every calendar day a carried row already speaks
        # for. A cursor ("the last day carried") cannot stand in for this: a week
        # that corrects a symbol's start backwards -- because the base now covers
        # a month it did not cover last week -- would have the older, later cursor
        # win, and the days in between would never be fetched by any week.
        carried_daily_days: dict[str, set[date_type]] = {}
        carried_rest_days: dict[str, set[date_type]] = {}
        if previous_root is not None:
            for entry in carried_previous:
                kind = str(entry.get("kind"))
                symbol = str(entry.get("symbol"))
                span_start, span_end = _row_span(kind, str(entry.get("month")))
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
                reached = carried_daily_days if kind == DAILY_TAIL_KIND else carried_rest_days
                # An `absent` carried row counts as covered: its day was asked for
                # and the venue does not have it, and flipping it to present in a
                # later week would break the lineage check against this one.
                reached.setdefault(symbol, set()).update(_span_days(span_start, span_end))

        deadline_ns = clock() + sunday_deadline_hours * _NANOSECONDS_PER_HOUR
        for symbol in symbols:
            _fetch_daily_tail(
                symbol=symbol,
                days=_uncovered_days(
                    first_day=daily_start[symbol],
                    tail_day=tail_day,
                    covered=carried_daily_days.get(symbol, frozenset()),
                ),
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
            # One request per contiguous uncovered run -- ordinarily one, or two
            # when a carried window sits inside this week's span.
            for run_first, run_last in _contiguous_runs(
                _uncovered_days(
                    first_day=rest_start[symbol],
                    tail_day=tail_day,
                    covered=carried_rest_days.get(symbol, frozenset()),
                )
            ):
                _fetch_funding_window(
                    symbol=symbol,
                    first_day=run_first,
                    last_day=run_last,
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
            "stale_symbols": stale_symbols,
            "reconciliation": _reconciliation_block(compared_rows),
        }
        capture_root_hash = content_sha256(material)
        document = dict(material)
        document["capture_root_hash"] = capture_root_hash
        manifest_path.write_bytes(canonical_json(document))
        published = True
    finally:
        if created and not published:
            shutil.rmtree(target, ignore_errors=True)
    return ShadowCaptureArtifact(
        capture_root=target,
        capture_root_hash=capture_root_hash,
        dataset_root_hash=dataset.root_hash,
        tail_through=tail_through,
        # Its own block, not the manifest's: that one is already sealed into
        # `capture_root_hash`, and a shared (or shallow-copied) `mismatches`
        # list would be reachable -- and mutable -- through the artifact.
        reconciliation=_reconciliation_block(compared_rows),
    )


def _fetch_daily_tail(
    *,
    symbol: str,
    days: tuple[date_type, ...],
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
    """Fetch one symbol's daily dumps for the planned days of its tail.

    A 404 is an absence, recorded exactly the way `capture_panel` records one
    and then stepped over -- except on the decision Sunday itself, where spec
    section 3.3's rule applies: a symbol that traded on the Saturday must have
    a Sunday bar, so the fetch waits for the dump rather than publishing a
    capture that silently misses the week's close.
    """
    saturday_open_time_ns = _open_time_ns(tail_day - timedelta(days=1))
    for day in days:
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
    last_day: date_type,
    venue: str,
    fetch: PanelFetch,
    target: Path,
    sources: list[dict[str, object]],
    funding: list[PanelFundingRow],
) -> None:
    """Fetch one symbol's funding history for one uncovered run of days.

    Funding settles several times a day, so a per-day request would be a
    dozen times the traffic for the same rows; the window is recorded in the
    manifest's `month` field as `<from>_<to>` so a reader can tell exactly
    which days one stored answer speaks for.
    """
    start_time_ms = _day_start_ms(first_day)
    end_time_ms = _day_end_ms(last_day)
    window = f"{first_day.isoformat()}_{last_day.isoformat()}"
    url = build_funding_rest_url(symbol, start_time_ms=start_time_ms, end_time_ms=end_time_ms)
    try:
        payload = fetch(url)
    except PanelCaptureError as error:
        # Including a 404: unlike a daily dump, whose absence is a normal
        # fact about a contract, the funding endpoint answers an empty array
        # for a window with no settlements. A refusal here is a real failure,
        # and it is the same retryable one the daily path raises (ruling 21).
        raise ShadowCaptureTransportError(
            f"funding history for {symbol} over {window} could not be fetched: {error}"
        ) from error
    _validate_panel_url(payload.url)
    rows = parse_funding_rest(
        payload,
        symbol=symbol,
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
        venue=venue,
    )
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
        except PanelCaptureError as error:
            # A 404 is an absence and is handled above; anything else the
            # client raises here is a transport that failed after its own
            # attempts, which is the one failure worth repeating (ruling 21).
            raise ShadowCaptureTransportError(
                f"the daily dump {url} could not be fetched: {error}"
            ) from error
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
    symbol = _present_row_text(entry, "symbol")
    _copy_raw_payload(
        source_root=source_root,
        target=target,
        relative=_present_row_text(entry, "raw_relative_path"),
        expected_sha256=_present_row_text(entry, "raw_sha256"),
    )
    new_candles, new_funding = _rows_from_source(entry, capture_root=target, venue=venue)
    candles.extend(new_candles)
    funding.extend(new_funding)
    open_times.setdefault(symbol, set()).update(row.open_time_ns for row in new_candles)


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
    if destination.exists():
        # Spec section 6: nothing is written under `raw/` twice. Overwriting
        # would hide a manifest that carries one path under two rows.
        #
        # A run unwinds a directory it created itself (ruling 21(b)), so what
        # is left here is a directory this run found already standing: a
        # capture half-written by a run that was killed before the unwinding
        # could happen, or something else's. Either way the cure is a person's
        # and the refusal names it rather than leaving a rerun to guess.
        raise ShadowCaptureError(
            f"raw payload path is already written: {relative}; this directory holds a "
            f"half-written capture from an interrupted run; delete {target} and rerun"
        )
    if not source.is_file():
        raise ShadowCaptureError(f"source raw payload is missing: {relative}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        # The source is present and the target absent, so the only thing left
        # for `os.link` to fail on is the filesystem itself -- a different
        # volume, a link-count limit, a share without links.
        try:
            shutil.copyfile(source, destination)
        except OSError as copy_error:
            raise ShadowCaptureError(
                f"raw payload could not be linked or copied: {relative}"
            ) from copy_error
    if hashlib.sha256(destination.read_bytes()).hexdigest() != expected_sha256:
        raise ShadowCaptureError(f"copied raw payload hash mismatch: {relative}")


def _rows_from_source(
    entry: dict[str, Any], *, capture_root: Path, venue: str
) -> tuple[tuple[PanelCandleRow, ...], tuple[PanelFundingRow, ...]]:
    """Re-derive one present source row's candle and funding rows."""
    symbol = _present_row_text(entry, "symbol")
    kind = _present_row_text(entry, "kind")
    payload = PanelPayload(
        url=_present_row_text(entry, "url"),
        raw_bytes=(capture_root / _present_row_text(entry, "raw_relative_path")).read_bytes(),
        received_time_ns=_present_row_received_time_ns(entry),
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


def _partition_previous_tail(
    entries: list[dict[str, Any]], *, covered: dict[str, dict[str, str]]
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], str]]]:
    """Split the previous week's tail rows into the carried and the dropped.

    A dropped row is paired with the month the base now reaches for its
    symbol and bucket, because that month is also the cutoff for what may be
    reconciled: a funding window that straddles the boundary is dropped whole
    -- its uncovered days are refetched by this week's set-difference plan --
    but only the settlements up to that month have a monthly dump to be
    compared against.
    """
    carried: list[dict[str, Any]] = []
    dropped: list[tuple[dict[str, Any], str]] = []
    for entry in entries:
        covering_month = _covering_month(entry, covered=covered)
        if covering_month is None:
            carried.append(entry)
        else:
            dropped.append((entry, covering_month))
    return carried, dropped


def _covering_month(
    entry: dict[str, Any], *, covered: dict[str, dict[str, str]]
) -> str | None:
    """The base's last monthly month for this row, when it already covers it.

    Per symbol, which is what matters when symbols cover different months:
    one contract's base may have reached August while a delisted one's stops
    years earlier.
    """
    kind = str(entry.get("kind"))
    bucket = _MONTH_BUCKET.get(kind)
    if bucket is None:
        return None
    span_start, _span_end = _row_span(kind, str(entry.get("month")))
    own_month = covered[bucket].get(str(entry.get("symbol")))
    if own_month is None or _month_of(span_start) > own_month:
        return None
    return own_month


def _reconcile_dropped_rows(
    dropped: list[tuple[dict[str, Any], str]],
    *,
    previous_root: Path,
    base_root: Path,
    base_sources: list[dict[str, Any]],
    venue: str,
) -> tuple[int, tuple[dict[str, object], ...]]:
    """Compare the rows the base's monthly dumps are about to replace.

    Both sides are parsed from their own capture's payloads -- the previous
    week's from `previous_root`, the base's from `base_root` -- and both
    captures have been verified, so the bytes behind every row match the
    digest their manifest claims. Only the base rows whose months are
    actually under comparison are parsed; the whole base is parsed once more
    a moment later by the carry, and there is no reason to do it twice for
    months nothing was dropped from.

    An `absent` dropped row is not a row: its day was asked for and the venue
    had nothing, so there is nothing to compare and nothing is lost by the
    monthly dump taking over.
    """
    previous_rows: list[PanelCandleRow | PanelFundingRow] = []
    needed: dict[tuple[str, str], set[str]] = {}
    for entry, covering_month in dropped:
        if entry.get("status") != "present":
            continue
        kind = _present_row_text(entry, "kind")
        symbol = _present_row_text(entry, "symbol")
        bucket = _MONTH_BUCKET[kind]
        candles, funding = _rows_from_source(entry, capture_root=previous_root, venue=venue)
        previous_rows.extend(candles)
        previous_rows.extend(
            row
            for row in funding
            if _month_of(_day_of_ns(row.calc_time_ns)) <= covering_month
        )
        needed.setdefault((bucket, symbol), set()).update(
            month
            for month in _months_of(kind, _present_row_text(entry, "month"))
            if month <= covering_month
        )

    base_rows: list[PanelCandleRow | PanelFundingRow] = []
    for entry in base_sources:
        if entry.get("status") != "present":
            continue
        base_kind = str(entry.get("kind"))
        base_bucket = _MONTH_BUCKET.get(base_kind)
        if base_bucket is None:
            continue
        months = needed.get((base_bucket, str(entry.get("symbol"))))
        if months is None or months.isdisjoint(_months_of(base_kind, str(entry.get("month")))):
            continue
        candles, funding = _rows_from_source(entry, capture_root=base_root, venue=venue)
        base_rows.extend(candles)
        base_rows.extend(funding)
    return reconcile_tail_rows(previous_rows=tuple(previous_rows), base_rows=tuple(base_rows))


def _refuse_reconciliation(
    *,
    target: Path,
    base_capture_root_hash: str,
    previous_capture_root_hash: str,
    compared_rows: int,
    mismatches: tuple[dict[str, object], ...],
) -> NoReturn:
    """Write the refusal document, then refuse (spec section 6).

    Beside the output directory, not inside it: the capture is not published
    and its directory must stay absent, or the rerun a person starts after
    reading this would find a manifest -- or a half-copied `raw/` -- and
    refuse for the wrong reason. The document is sealed the way the manifest
    is, so the record of *why* a week is missing cannot be edited unnoticed,
    and it is written through a temporary file so a reader never finds half
    of it.
    """
    material: dict[str, object] = {
        "base_capture_root_hash": base_capture_root_hash,
        "previous_capture_root_hash": previous_capture_root_hash,
        "compared_rows": compared_rows,
        "mismatches": list(mismatches),
    }
    document = dict(material)
    document["content_sha256"] = content_sha256(material)
    path = target.parent / f"{target.name}{_RECONCILIATION_REFUSAL_SUFFIX}"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(canonical_json(document))
    os.replace(temporary, path)
    raise ShadowCaptureError(
        f"RECONCILIATION_MISMATCH: {len(mismatches)} field(s) of the previous capture's "
        f"tail disagree with the base capture's monthly dumps over {compared_rows} "
        f"compared rows; see {path}"
    )


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


def _covered_months(entries: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    """Per bucket and symbol, the last whole month the base actually carries.

    Only `present` rows count: an `absent` month is one the venue never
    published for that symbol, so the symbol is not covered for it. The two
    buckets are kept apart because they run at different speeds -- a symbol
    can have klines for a month whose funding dump was never written, and a
    delisted symbol's last klines month is years behind the panel's furthest
    one. A single maximum over all of it would silently skip whole months for
    everything that is not at the front.
    """
    covered: dict[str, dict[str, str]] = {kind: {} for kind in _MONTHLY_KINDS}
    for entry in entries:
        kind = str(entry.get("kind"))
        if kind not in _MONTHLY_KINDS or entry.get("status") != "present":
            continue
        month = str(entry.get("month"))
        if _MONTH_PATTERN.match(month) is None:
            continue
        symbol = str(entry.get("symbol"))
        if month > covered[kind].get(symbol, ""):
            covered[kind][symbol] = month
    return covered


def _furthest_month(covered: dict[str, str]) -> str | None:
    """The furthest month any symbol reaches in one bucket."""
    return max(covered.values()) if covered else None


def _bucket_tail_start(own_month: str | None, *, furthest_month: str) -> tuple[date_type, bool]:
    """Where one symbol's tail starts in one bucket, and whether it is stale.

    Three cases, and the middle one is why this is not a single cutoff:

    - the symbol reaches the furthest month: its tail starts the day after
      that month, the ordinary case;
    - it is exactly one month behind: Binance publishes a month's dump days
      into the next month, so this is publication lag, not a gone contract.
      Its tail starts the day after *its own* last month -- at most 31 extra
      daily dumps and the same single REST window;
    - it is further behind, or has no month in this bucket at all: a delisting
      or a funding dump the venue never wrote. Refetching years of daily dumps
      for every such symbol every week is not a weekly job, so the tail starts
      at the furthest month's day and the gap is declared in the manifest's
      `stale_symbols` instead of being silently skipped.
    """
    furthest_start = _first_day_after_month(furthest_month)
    if own_month == furthest_month:
        return furthest_start, False
    if own_month is not None and _month_of(_first_day_after_month(own_month)) == furthest_month:
        return _first_day_after_month(own_month), False
    return furthest_start, True


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


def _reconciliation_block(compared_rows: int) -> dict[str, object]:
    """A fresh, unshared reconciliation block for a week that reconciled clean.

    `mismatches` is always empty here: a week with any mismatch never reaches
    the manifest, it refuses. The block is recorded all the same so a reader
    can tell a week that compared nothing from one that compared hundreds of
    rows and found them all equal.
    """
    return {"compared_rows": compared_rows, "mismatches": []}


def _present_row_name(entry: dict[str, Any]) -> str:
    return f"{entry.get('kind')}:{entry.get('symbol')}:{entry.get('month')}"


def _present_row_text(entry: dict[str, Any], key: str) -> str:
    """One string field a `present` source row must carry.

    A row a capture recorded as present but that lacks its url, path or
    digest is a malformed manifest, and it must say so rather than raise a
    bare `KeyError` out of the middle of a copy.
    """
    value = entry.get(key)
    if not isinstance(value, str) or not value:
        raise ShadowCaptureError(
            f"present source row {_present_row_name(entry)} has no {key}"
        )
    return value


def _present_row_received_time_ns(entry: dict[str, Any]) -> int:
    value = entry.get("received_time_ns")
    # `isinstance(True, int)` is true and a bool is never a timestamp.
    if not isinstance(value, int) or isinstance(value, bool):
        raise ShadowCaptureError(
            f"present source row {_present_row_name(entry)} has no received_time_ns"
        )
    return value


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


def _span_days(first_day: date_type, last_day: date_type) -> tuple[date_type, ...]:
    """Every calendar day one carried row speaks for, inclusive."""
    return _plan_tail_days(first_day=first_day, tail_day=last_day)


def _uncovered_days(
    *, first_day: date_type, tail_day: date_type, covered: Collection[date_type]
) -> tuple[date_type, ...]:
    """The days of this week's span that no carried row already speaks for.

    The span is recomputed from this week's base every week, so a symbol whose
    base coverage moved backwards-looking -- a month that arrived since last
    week -- gets the days between its new start and what was carried, which a
    forward-only cursor would have skipped for good.
    """
    return tuple(
        day
        for day in _plan_tail_days(first_day=first_day, tail_day=tail_day)
        if day not in covered
    )


def _contiguous_runs(days: tuple[date_type, ...]) -> tuple[tuple[date_type, date_type], ...]:
    """Ascending days grouped into inclusive (first, last) runs."""
    runs: list[tuple[date_type, date_type]] = []
    for day in days:
        if runs and day == runs[-1][1] + timedelta(days=1):
            runs[-1] = (runs[-1][0], day)
        else:
            runs.append((day, day))
    return tuple(runs)


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


def _day_of_ns(time_ns: int) -> date_type:
    """The UTC calendar day a nanosecond timestamp falls on."""
    return _EPOCH + timedelta(days=time_ns // DAY_NS)


def _day_start_ms(day: date_type) -> int:
    return (day - _EPOCH).days * _MILLISECONDS_PER_DAY


def _day_end_ms(day: date_type) -> int:
    return _day_start_ms(day) + _MILLISECONDS_PER_DAY - 1
