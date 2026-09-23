"""The permanent Binance measurement journal: spec, round and chained segments.

The cost journal (``binance_cost_journal``) measures what a taker pays on
fifteen pairs and stops after its declared rounds. This journal is the second
one spec section 5 asks for: it runs until the process is stopped, records the
same depth walk every 61 seconds, and every fifth round adds the three
all-symbol snapshots - every USDT perpetual's funding premium and basis, every
USDT perpetual's best bid and ask, every USDT spot pair's best bid and ask - so
that a later family reading this stream never has to have declared a symbol
list in advance.

Everything the two journals share is imported from v1 rather than copied: the
instrument model, the depth walk, the atomic publish, the storage
authorisation, the write lock, the cadence arithmetic and the back-off. v1 is
finished and immutable; nothing here changes its behaviour. What is written
here is what the day-partitioned layout needs - the segment walk, the chain
head read, the torn-file discard and the chain verification - plus the three
snapshot parsers and the round that joins them.
"""

import json
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal, NamedTuple, Self

from pydantic import ValidationError, field_validator, model_validator

from trading_bot.binance_cost_journal import (
    _BPS,
    _BPS_QUANTUM,
    _IDENTIFIER,
    _MAX_REASON_CHARACTERS,
    _SEGMENT_NAME,
    _SEGMENT_TEMPORARY_NAME,
    _SYMBOL,
    CHAIN_HEAD_NAME,
    JOURNAL_SPEC_NAME,
    SEGMENT_DIRECTORY_NAME,
    ZERO_HASH,
    BinanceCostJournalError,
    BinanceCostJournalTransportError,
    Fetcher,
    InstrumentObservation,
    JournalInstrument,
    _authorize,
    _failure_reason,
    _Frozen,
    _lock_journal,
    _observe,
    _publish,
    _quantised,
    _read_object,
    _release_journal_lock,
    _remaining_interval,
    _throttle_seconds,
    _validated_hex,
    load_journal_spec,
)
from trading_bot.canonical import content_sha256
from trading_bot.depth_adapters import (
    DepthPayloadError,
    _decimal_string,
    _integer,
    _mapping,
    _string,
)

MEASUREMENT_JOURNAL_VERSION = "binance-measurement-journal/1.0.0"
# Spec 5: every 61 seconds the depth walk, every fifth round the three
# all-symbol snapshots. The interval itself is copied from the cost journal's
# spec at creation, so the two streams keep one cadence.
SNAPSHOT_EVERY_ROUNDS = 5
PREMIUM_INDEX_URL = "https://fapi.binance.com/fapi/v1/premiumIndex"
PERP_BOOK_TICKER_URL = "https://fapi.binance.com/fapi/v1/ticker/bookTicker"
SPOT_BOOK_TICKER_URL = "https://api.binance.com/api/v3/ticker/bookTicker"
# The names a failure wears in a segment (ruling 10). The depth walk's own
# failures stay inside their observation, exactly as v1 records them.
PREMIUM_INDEX_ENDPOINT = "premiumIndex"
PERP_BOOK_TICKER_ENDPOINT = "perpBookTicker"
SPOT_BOOK_TICKER_ENDPOINT = "spotBookTicker"
# The keys every segment's `excluded` count carries, in request order.
SNAPSHOT_ENDPOINTS: tuple[str, ...] = (
    PREMIUM_INDEX_ENDPOINT,
    PERP_BOOK_TICKER_ENDPOINT,
    SPOT_BOOK_TICKER_ENDPOINT,
)
# The measurement stream is a USDT stream: the carry family quotes in USDT and
# a symbol quoted in anything else is dropped before it is parsed.
QUOTE_ASSET = "USDT"

# url -> parsed JSON array; raises on transport failure.
type ArrayFetcher = Callable[[str], Sequence[object]]

_DAY_NAME = re.compile(r"\A[0-9]{4}-[0-9]{2}-[0-9]{2}\Z")
_NANOSECONDS_A_SECOND = 1_000_000_000


class BinanceMeasurementJournalError(RuntimeError):
    """Raised when a measurement journal cannot be created, sampled or read safely."""


class BinanceMeasurementJournalSpecError(BinanceMeasurementJournalError):
    """Raised when the request does not match the journal on disk.

    The CLI maps this to its own exit code, as it does for the cost journal, so
    a supervisor stops on a mismatched or unverifiable journal instead of
    restarting a process that can only fail the same way again.
    """


class BinanceMeasurementJournalSpec(_Frozen):
    """The frozen declaration a measurement journal directory is bound to."""

    version: Literal["binance-measurement-journal/1.0.0"]
    run_id: str
    created_time_ns: int
    # The cost journal spec whose sample this stream takes over (spec 5, last
    # bullet): the two journals measure the same legs, and this hash says
    # which declaration those legs were derived under.
    cost_journal_spec_hash: str
    instruments: tuple[JournalInstrument, ...]
    notionals: tuple[Decimal, ...]
    depth_limit: int
    sample_interval_seconds: int
    snapshot_every_rounds: int
    spot_fee_bps_per_side: Decimal
    perpetual_fee_bps_per_side: Decimal
    fee_evidence_id: str

    @field_validator("cost_journal_spec_hash")
    @classmethod
    def validate_hashes(cls, value: str) -> str:
        return _validated_hex(value, field_name="cost journal spec hash")

    @field_validator("run_id", "fee_evidence_id")
    @classmethod
    def validate_identifiers(cls, value: str) -> str:
        if not _IDENTIFIER.match(value):
            raise ValueError("identifiers are non-empty and free of whitespace")
        return value

    @field_validator("created_time_ns")
    @classmethod
    def validate_times(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("times are positive nanosecond stamps")
        return value

    @field_validator("depth_limit", "sample_interval_seconds", "snapshot_every_rounds")
    @classmethod
    def validate_counts(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("depth limit, interval and snapshot cadence are positive")
        return value

    @field_validator("spot_fee_bps_per_side", "perpetual_fee_bps_per_side")
    @classmethod
    def validate_fees(cls, value: Decimal) -> Decimal:
        if value < 0:
            raise ValueError("declared fees cannot be negative")
        return value

    @field_validator("instruments")
    @classmethod
    def validate_instruments(
        cls, value: tuple[JournalInstrument, ...]
    ) -> tuple[JournalInstrument, ...]:
        if not value:
            raise ValueError("a journal samples at least one instrument")
        identifiers = [item.instrument_id for item in value]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("an instrument may appear at most once")
        return value

    @field_validator("notionals")
    @classmethod
    def validate_notionals(cls, value: tuple[Decimal, ...]) -> tuple[Decimal, ...]:
        if not value:
            raise ValueError("a journal samples at least one notional")
        if any(item <= 0 for item in value):
            raise ValueError("notionals are positive quote amounts")
        if list(value) != sorted(value) or len(set(value)) != len(value):
            raise ValueError("notionals are distinct and ascending")
        return value


class PremiumRow(_Frozen):
    """One USDT perpetual's funding premium in one snapshot.

    The basis is the mark's distance from the index in basis points, at the
    journal's declared precision of 1e-6 bps, half-even - the same arithmetic
    and the same quantum the cost journal records a perpetual's basis at, so a
    reader can put the two streams side by side.
    """

    symbol: str
    mark_price: Decimal
    index_price: Decimal
    last_funding_rate: Decimal
    next_funding_time_ms: int
    basis_bps: Decimal

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, value: str) -> str:
        if not _SYMBOL.match(value):
            raise ValueError("symbols are upper-case venue symbols")
        return value

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        if self.mark_price <= 0 or self.index_price <= 0:
            raise ValueError("a recorded premium index carries positive prices")
        if self.next_funding_time_ms <= 0:
            raise ValueError("the next funding time is a positive millisecond stamp")
        return self


class BookRow(_Frozen):
    """One USDT pair's best bid and ask in one snapshot, with its spread in bps."""

    symbol: str
    bid_price: Decimal
    bid_qty: Decimal
    ask_price: Decimal
    ask_qty: Decimal
    spread_bps: Decimal

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, value: str) -> str:
        if not _SYMBOL.match(value):
            raise ValueError("symbols are upper-case venue symbols")
        return value

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        if min(self.bid_price, self.bid_qty, self.ask_price, self.ask_qty) <= 0:
            raise ValueError("a recorded book displays a positive price and quantity a side")
        if self.bid_price >= self.ask_price:
            raise ValueError("a recorded book is not crossed")
        if self.spread_bps <= 0:
            raise ValueError("a recorded spread is positive")
        return self


class MeasurementSegment(_Frozen):
    """One round of measurements, linked to its predecessor by hash.

    ``depth`` is the cost journal's walk of the declared instruments, one
    observation apiece, carrying its own failures exactly as v1 records them.
    The three snapshot fields are ``None`` on a round that is not a snapshot
    round - the sequence decides, ``sequence % snapshot_every_rounds == 0``, so
    the first round of a journal is one - and on a snapshot round whose
    endpoint failed; ``failures`` then names the endpoint and what it answered.

    ``excluded`` counts, per endpoint, the USDT symbols that parsed and
    measured nothing - a halted pair's zero book, a delisted perpetual's zero
    index - and were left out of the rows rather than costing the whole
    endpoint (ruling 11). All three keys are present on every segment and are
    zero on a round that took no snapshot, so a reader never has to tell an
    absent count from a zero one. Symbols quoted in something other than USDT
    are out of scope and are not counted here. A snapshot field is the empty
    tuple where every USDT symbol the endpoint listed was excluded (ruling 12):
    ``None`` means the endpoint was not read, ``()`` that it was read and
    measured nothing.
    """

    version: Literal["binance-measurement-journal/1.0.0"]
    sequence: int
    spec_hash: str
    previous_segment_hash: str
    received_time_ns: int
    depth: tuple[InstrumentObservation, ...]
    premium_index: tuple[PremiumRow, ...] | None
    perp_book: tuple[BookRow, ...] | None
    spot_book: tuple[BookRow, ...] | None
    failures: tuple[str, ...]
    excluded: Mapping[str, int]
    content_hash: str

    @field_validator("spec_hash", "previous_segment_hash", "content_hash")
    @classmethod
    def validate_hashes(cls, value: str) -> str:
        return _validated_hex(value, field_name="segment hash")

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, value: int) -> int:
        if value < 0:
            raise ValueError("sequence numbers start at zero")
        return value

    @field_validator("received_time_ns")
    @classmethod
    def validate_time(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("received_time_ns must be a positive nanosecond stamp")
        return value

    @field_validator("depth")
    @classmethod
    def validate_depth(
        cls, value: tuple[InstrumentObservation, ...]
    ) -> tuple[InstrumentObservation, ...]:
        if not value:
            raise ValueError("a segment walks at least one instrument")
        identifiers = [item.instrument_id for item in value]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("an instrument is observed at most once per round")
        return value

    @field_validator("premium_index")
    @classmethod
    def validate_premium_index(
        cls, value: tuple[PremiumRow, ...] | None
    ) -> tuple[PremiumRow, ...] | None:
        return _distinct_rows(value)

    @field_validator("perp_book", "spot_book")
    @classmethod
    def validate_book(cls, value: tuple[BookRow, ...] | None) -> tuple[BookRow, ...] | None:
        return _distinct_rows(value)

    @field_validator("failures")
    @classmethod
    def validate_failures(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item or ":" not in item for item in value):
            raise ValueError("a failure names its endpoint and what it answered")
        if len(set(value)) != len(value):
            raise ValueError("an endpoint fails at most once per round")
        return value

    @field_validator("excluded")
    @classmethod
    def validate_excluded(cls, value: Mapping[str, int]) -> Mapping[str, int]:
        if set(value) != set(SNAPSHOT_ENDPOINTS):
            raise ValueError(f"a segment counts exclusions for {SNAPSHOT_ENDPOINTS}")
        if any(count < 0 for count in value.values()):
            raise ValueError("an exclusion count cannot be negative")
        return value

    @model_validator(mode="after")
    def validate_chain_link(self) -> Self:
        if self.sequence == 0 and self.previous_segment_hash != ZERO_HASH:
            raise ValueError("the first segment has no predecessor")
        if self.sequence > 0 and self.previous_segment_hash == ZERO_HASH:
            raise ValueError("a later segment names its predecessor")
        return self


class MeasurementChainHead(_Frozen):
    """The resumable tip of a measurement journal's segment chain.

    The cost journal's head in every respect but the version it carries: this
    one names the measurement journal, so the head of one journal cannot be
    read as the head of the other by accident.
    """

    version: Literal["binance-measurement-journal/1.0.0"]
    spec_hash: str
    segment_count: int
    last_sequence: int
    final_segment_hash: str
    content_hash: str

    @field_validator("spec_hash", "final_segment_hash", "content_hash")
    @classmethod
    def validate_hashes(cls, value: str) -> str:
        return _validated_hex(value, field_name="chain head hash")

    @field_validator("segment_count", "last_sequence")
    @classmethod
    def validate_counts(cls, value: int) -> int:
        if value < 0:
            raise ValueError("segment count and last sequence cannot be negative")
        return value


def _distinct_rows[RowT: PremiumRow | BookRow](
    value: tuple[RowT, ...] | None,
) -> tuple[RowT, ...] | None:
    """A recorded snapshot names every symbol it carries at most once.

    ``None`` is "this endpoint was not read this round"; the empty tuple is
    "it was read and every USDT symbol it listed was unusable" (ruling 12), and
    the segment's ``excluded`` count says how many that was.
    """
    if value is None:
        return None
    symbols = [row.symbol for row in value]
    if len(set(symbols)) != len(symbols):
        raise ValueError("a symbol is recorded at most once per snapshot")
    return value


def premium_rows(payload: Sequence[object]) -> tuple[tuple[PremiumRow, ...], int]:
    """Parse ``GET /fapi/v1/premiumIndex`` into the USDT rows and the count excluded.

    A symbol quoted in anything but USDT is dropped before its numbers are
    read, and is not counted. Every symbol that is kept must *parse* whole:
    prices and the funding rate are Binance's decimal strings, and a JSON
    number is a float that has already lost digits, so a wrong type, a missing
    field or an entry that is not an object refuses the whole payload. That is
    evidence of an API change, and half an all-symbol snapshot is a silently
    incomplete one.

    An entry that parses and measures nothing - a zero or negative index or
    mark price, a funding time that is not a stamp - is *excluded* from the
    rows and counted instead (ruling 11): a delisted perpetual can carry a zero
    index every round, and one of those must not cost the whole venue's
    snapshot.
    """
    rows: list[PremiumRow] = []
    excluded = 0
    seen: set[str] = set()
    for index, entry in enumerate(payload):
        document = _mapping_entry(entry, index=index)
        symbol = _entry_symbol(document, index=index)
        if symbol is None:
            continue
        _refuse_duplicate(symbol, seen)
        try:
            mark_price = _decimal_string(document.get("markPrice"), field_name="markPrice")
            index_price = _decimal_string(document.get("indexPrice"), field_name="indexPrice")
            funding_rate = _decimal_string(
                document.get("lastFundingRate"), field_name="lastFundingRate"
            )
            next_funding = _integer(document.get("nextFundingTime"), field_name="nextFundingTime")
        except DepthPayloadError as error:
            raise DepthPayloadError(f"malformed entry {symbol}") from error
        row = _premium_row(
            symbol,
            mark_price=mark_price,
            index_price=index_price,
            funding_rate=funding_rate,
            next_funding=next_funding,
        )
        if row is None:
            excluded += 1
            continue
        rows.append(row)
    return _kept(rows, excluded=excluded), excluded


def book_rows(payload: Sequence[object]) -> tuple[tuple[BookRow, ...], int]:
    """Parse one ``ticker/bookTicker`` array into the USDT rows and the count excluded.

    The spread is the distance between the two best prices over their mid, in
    basis points at 1e-6, half-even. What refuses the payload and what is
    merely excluded from it is the split ``premium_rows`` makes: a wrong type
    or a missing field is an API change and refuses; a book that parses and
    displays nothing to measure - a zero or negative price or quantity on
    either side, a crossed or locked book, a spread below the recorded
    precision - is excluded and counted (ruling 11). Halted spot pairs quote a
    zero book every day and may not cost the whole venue's snapshot.
    """
    rows: list[BookRow] = []
    excluded = 0
    seen: set[str] = set()
    for index, entry in enumerate(payload):
        document = _mapping_entry(entry, index=index)
        symbol = _entry_symbol(document, index=index)
        if symbol is None:
            continue
        _refuse_duplicate(symbol, seen)
        try:
            bid_price = _decimal_string(document.get("bidPrice"), field_name="bidPrice")
            bid_qty = _decimal_string(document.get("bidQty"), field_name="bidQty")
            ask_price = _decimal_string(document.get("askPrice"), field_name="askPrice")
            ask_qty = _decimal_string(document.get("askQty"), field_name="askQty")
        except DepthPayloadError as error:
            raise DepthPayloadError(f"malformed entry {symbol}") from error
        row = _book_row(
            symbol,
            bid_price=bid_price,
            bid_qty=bid_qty,
            ask_price=ask_price,
            ask_qty=ask_qty,
        )
        if row is None:
            excluded += 1
            continue
        rows.append(row)
    return _kept(rows, excluded=excluded), excluded


def _premium_row(
    symbol: str,
    *,
    mark_price: Decimal,
    index_price: Decimal,
    funding_rate: Decimal,
    next_funding: int,
) -> PremiumRow | None:
    """The row a parsed premium-index entry measures, or ``None`` for none of it."""
    if mark_price <= 0 or index_price <= 0 or next_funding <= 0:
        return None
    try:
        basis_bps = _quantised((mark_price - index_price) / index_price * _BPS, _BPS_QUANTUM)
    except DepthPayloadError:
        return None
    return PremiumRow(
        symbol=symbol,
        mark_price=mark_price,
        index_price=index_price,
        last_funding_rate=funding_rate,
        next_funding_time_ms=next_funding,
        basis_bps=basis_bps,
    )


def _book_row(
    symbol: str,
    *,
    bid_price: Decimal,
    bid_qty: Decimal,
    ask_price: Decimal,
    ask_qty: Decimal,
) -> BookRow | None:
    """The row a parsed book entry measures, or ``None`` for none of it."""
    if min(bid_price, bid_qty, ask_price, ask_qty) <= 0 or bid_price >= ask_price:
        return None
    mid = (bid_price + ask_price) / Decimal(2)
    try:
        spread_bps = _quantised((ask_price - bid_price) / mid * _BPS, _BPS_QUANTUM)
    except DepthPayloadError:
        return None
    if spread_bps <= 0:
        return None
    return BookRow(
        symbol=symbol,
        bid_price=bid_price,
        bid_qty=bid_qty,
        ask_price=ask_price,
        ask_qty=ask_qty,
        spread_bps=spread_bps,
    )


def _mapping_entry(entry: object, *, index: int) -> Mapping[str, object]:
    try:
        return _mapping(entry, field_name="entry")
    except DepthPayloadError as error:
        raise DepthPayloadError(f"malformed entry at index {index}") from error


def _entry_symbol(document: Mapping[str, object], *, index: int) -> str | None:
    """The entry's symbol, or ``None`` where the journal does not keep it.

    A symbol quoted in something other than USDT is dropped, which is the
    journal's declared scope and neither a defect nor an exclusion. A symbol
    that *is* quoted in USDT and is not an upper-case venue symbol refuses the
    payload: it is a row this parser does not understand in the set it does
    keep. The index names the entry rather than the string, so a hostile
    payload cannot write its own text into a segment.
    """
    try:
        symbol = _string(document.get("symbol"), field_name="symbol")
    except DepthPayloadError as error:
        raise DepthPayloadError(f"malformed entry at index {index}") from error
    if not symbol.endswith(QUOTE_ASSET):
        return None
    if not _SYMBOL.match(symbol):
        raise DepthPayloadError(f"malformed entry at index {index}")
    return symbol


def _refuse_duplicate(symbol: str, seen: set[str]) -> None:
    """One symbol answers once: a repeat is a payload this parser cannot read."""
    if symbol in seen:
        raise DepthPayloadError(f"duplicate entry {symbol}")
    seen.add(symbol)


def _kept[RowT](rows: list[RowT], *, excluded: int) -> tuple[RowT, ...]:
    """The parsed rows, refusing only a payload that held no USDT symbol at all.

    Binance's three all-symbol endpoints list hundreds of USDT symbols, so an
    array carrying none of them is a schema or venue anomaly and is that
    endpoint's failure for the round. A payload whose USDT symbols were all
    *excluded* is the ordinary rule at its extreme and takes the ordinary path
    (ruling 12): no rows and a count of what was left out, so a venue that
    halted everything is recorded as exactly that rather than as a parse
    failure.
    """
    if not rows and not excluded:
        raise DepthPayloadError(f"no {QUOTE_ASSET} symbols")
    return tuple(rows)


def create_measurement_journal(
    *,
    workspace_root: Path,
    journal_root: Path,
    reserve_bytes: int,
    run_id: str,
    cost_journal_root: Path,
) -> Path:
    """Write ``<journal_root>/journal-spec.json`` and an empty ``segments/``.

    The sample is not derived again: the instruments, notionals, depth limit,
    cadence and declared fees are copied from the cost journal this stream
    takes over from, and that journal's spec hash is recorded, so the two
    measure the same legs under one derivation (spec 5, last bullet). The cost
    journal is only read - v1 is finished and immutable - and its spec must
    match its own recorded hash or there is nothing to continue.

    The spec is immutable and binds the journal directory to it, so a root that
    already holds one is refused. Returns the path of the spec.
    """
    root = _authorize(workspace_root, journal_root, reserve_bytes)
    spec_path = root / JOURNAL_SPEC_NAME
    if spec_path.exists():
        raise BinanceMeasurementJournalSpecError("this journal already exists and is immutable")
    try:
        cost_spec, cost_spec_hash = load_journal_spec(cost_journal_root)
    except BinanceCostJournalError as error:
        raise BinanceMeasurementJournalSpecError(
            f"the cost journal cannot be read: {error}"
        ) from error
    instruments = [instrument.model_dump(mode="json") for instrument in cost_spec.instruments]
    document: dict[str, object] = {
        "version": MEASUREMENT_JOURNAL_VERSION,
        "run_id": run_id,
        "created_time_ns": time.time_ns(),
        "cost_journal_spec_hash": cost_spec_hash,
        "instruments": instruments,
        "notionals": [str(notional) for notional in cost_spec.notionals],
        "depth_limit": cost_spec.depth_limit,
        "sample_interval_seconds": cost_spec.sample_interval_seconds,
        "snapshot_every_rounds": SNAPSHOT_EVERY_ROUNDS,
        "spot_fee_bps_per_side": str(cost_spec.spot_fee_bps_per_side),
        "perpetual_fee_bps_per_side": str(cost_spec.perpetual_fee_bps_per_side),
        "fee_evidence_id": cost_spec.fee_evidence_id,
    }
    try:
        spec = BinanceMeasurementJournalSpec.model_validate(document)
    except ValidationError as error:
        raise BinanceMeasurementJournalSpecError(
            f"the derived measurement journal spec is invalid: {error}"
        ) from error
    material: dict[str, object] = spec.model_dump(mode="json")
    (root / SEGMENT_DIRECTORY_NAME).mkdir(parents=True, exist_ok=True)
    _publish(spec_path, {**material, "spec_hash": content_sha256(material)})
    return spec_path


def load_measurement_journal_spec(
    journal_root: Path,
) -> tuple[BinanceMeasurementJournalSpec, str]:
    """Read a measurement journal's frozen spec and return it with its recorded hash."""
    try:
        document = _read_object(journal_root / JOURNAL_SPEC_NAME, label="the journal spec")
    except BinanceCostJournalError as error:
        raise BinanceMeasurementJournalSpecError(str(error)) from error
    material = {key: value for key, value in document.items() if key != "spec_hash"}
    spec_hash = content_sha256(material)
    if document.get("spec_hash") != spec_hash:
        raise BinanceMeasurementJournalSpecError(
            "the journal spec does not match its recorded hash"
        )
    try:
        spec = BinanceMeasurementJournalSpec.model_validate(material)
    except ValidationError as error:
        raise BinanceMeasurementJournalSpecError(f"the journal spec is invalid: {error}") from error
    return spec, spec_hash


def run_measurement_journal(
    *,
    workspace_root: Path,
    journal_root: Path,
    reserve_bytes: int,
    rounds: int | None,
    fetcher: Fetcher,
    array_fetcher: ArrayFetcher,
    clock: Callable[[], int] = time.time_ns,
    sleep: Callable[[float], None] = time.sleep,
) -> MeasurementChainHead:
    """Append sampled segments to a verified journal until ``rounds`` are written.

    ``rounds=None`` is the stream spec 5 asks for: the loop samples until the
    process is stopped, and the head it would have returned is on disk after
    every round, so a kill costs at most the round it interrupted. The Startup
    launcher restarts the process and the supervisor's liveness is the newest
    segment's age.

    The journal is written by one process at a time (v1's ``run.lock``), and a
    chain head naming another spec refuses: a resumed run extends exactly the
    journal it was pointed at or none. What a restart verifies is the *tail* -
    the chain head against the newest segment, and the two newest day
    directories link by link, over a listing of the whole journal (ruling 13);
    the explicit verification command walks the whole chain from ``ZERO_HASH``.
    A permanent stream restarts routinely, and a restart that re-read every
    round since the journal began would cost more every week it stayed alive.
    Two marks of a killed process are repaired rather than refused, as in v1 -
    a half-written trailing segment is deleted and its sequence sampled again,
    and a head that was never written, was torn in half or lags a published
    segment is rebuilt over the segments once the tail has verified. Anything
    else that fails to verify is a change rather than an interruption and stops
    the run.

    It differs from ``run_journal`` in one deliberate way: a round in which
    *every* request failed is still published as a segment of failures. Spec 6
    says a round with a failing request is recorded as a partial segment and
    "never dropped"; v1 aborts such a round instead, because its receipt counts
    measured observations and a dead venue would otherwise fill its target with
    empty ones. This journal has no target and its liveness is read from the
    newest segment's age, so an outage is evidence and is recorded (ruling 9).
    The 429/418 back-off is v1's: the seconds any request in a round earned are
    waited on top of the *next* round's period.

    The declared interval is a period, not a gap (v1's ``_remaining_interval``),
    and a resumed run waits a whole one before its first round so a restart
    cannot sample faster than the spec declares.
    """
    if rounds is not None and rounds <= 0:
        raise BinanceMeasurementJournalSpecError("a run appends at least one round")
    root = _authorize(workspace_root, journal_root, reserve_bytes)
    spec, spec_hash = load_measurement_journal_spec(root)
    try:
        lock = _lock_journal(root)
    except BinanceCostJournalError as error:
        raise BinanceMeasurementJournalSpecError(str(error)) from error
    try:
        return _run_locked_measurement_journal(
            root,
            spec=spec,
            spec_hash=spec_hash,
            rounds=rounds,
            fetcher=fetcher,
            array_fetcher=array_fetcher,
            clock=clock,
            sleep=sleep,
        )
    finally:
        _release_journal_lock(lock)


def _run_locked_measurement_journal(
    root: Path,
    *,
    spec: BinanceMeasurementJournalSpec,
    spec_hash: str,
    rounds: int | None,
    fetcher: Fetcher,
    array_fetcher: ArrayFetcher,
    clock: Callable[[], int],
    sleep: Callable[[float], None],
) -> MeasurementChainHead:
    """``run_measurement_journal``'s body, under the journal's write lock."""
    head = _resumed_head(root, spec_hash=spec_hash)
    # `None` before the first round of a run that starts a journal: that one
    # round alone waits for nothing.
    cadence: Decimal | None = Decimal(spec.sample_interval_seconds) if head is not None else None
    throttle_seconds = 0
    written = 0
    while rounds is None or written < rounds:
        if cadence is not None:
            sleep(float(cadence))
            if throttle_seconds:
                sleep(float(throttle_seconds))
        segment, throttle_seconds = _sample_round(
            spec=spec,
            spec_hash=spec_hash,
            head=head,
            fetcher=fetcher,
            array_fetcher=array_fetcher,
            clock=clock,
        )
        head = _publish_segment(root, segment=segment, spec_hash=spec_hash)
        written += 1
        cadence = _remaining_interval(
            spec.sample_interval_seconds, started_ns=segment.received_time_ns, now_ns=clock()
        )
    if head is None:
        raise BinanceMeasurementJournalError("the journal is empty and no round was sampled")
    return head


def _sample_round(
    *,
    spec: BinanceMeasurementJournalSpec,
    spec_hash: str,
    head: MeasurementChainHead | None,
    fetcher: Fetcher,
    array_fetcher: ArrayFetcher,
    clock: Callable[[], int],
) -> tuple[MeasurementSegment, int]:
    """The round's segment and the seconds the venue asked to be left alone.

    The depth walk is v1's, instrument by instrument, so the slippage tiers
    stay continuous across the two journals; every fifth round adds the three
    all-symbol requests. Nothing here raises: a failed instrument is its own
    failed observation, a failed endpoint is an entry in ``failures``
    (ruling 10), a symbol that measured nothing is a count in ``excluded``
    (ruling 11), and the segment is published either way.
    """
    received_time_ns = clock()
    sequence = 0 if head is None else head.last_sequence + 1
    sampled = [
        _observe(instrument, notionals=spec.notionals, fetcher=fetcher, clock=clock)
        for instrument in spec.instruments
    ]
    depth = [observation.model_dump(mode="json") for observation, _ in sampled]
    throttle_seconds = max((seconds for _, seconds in sampled), default=0)
    premium_index: tuple[PremiumRow, ...] | None = None
    perp_book: tuple[BookRow, ...] | None = None
    spot_book: tuple[BookRow, ...] | None = None
    failures: list[str] = []
    excluded = dict.fromkeys(SNAPSHOT_ENDPOINTS, 0)
    if sequence % spec.snapshot_every_rounds == 0:
        premium = _snapshot(
            endpoint=PREMIUM_INDEX_ENDPOINT,
            url=PREMIUM_INDEX_URL,
            array_fetcher=array_fetcher,
            parse=premium_rows,
        )
        perpetual = _snapshot(
            endpoint=PERP_BOOK_TICKER_ENDPOINT,
            url=PERP_BOOK_TICKER_URL,
            array_fetcher=array_fetcher,
            parse=book_rows,
        )
        spot = _snapshot(
            endpoint=SPOT_BOOK_TICKER_ENDPOINT,
            url=SPOT_BOOK_TICKER_URL,
            array_fetcher=array_fetcher,
            parse=book_rows,
        )
        premium_index, perp_book, spot_book = premium.rows, perpetual.rows, spot.rows
        excluded = {
            PREMIUM_INDEX_ENDPOINT: premium.excluded,
            PERP_BOOK_TICKER_ENDPOINT: perpetual.excluded,
            SPOT_BOOK_TICKER_ENDPOINT: spot.excluded,
        }
        throttle_seconds = max(
            throttle_seconds, premium.throttle_seconds, perpetual.throttle_seconds,
            spot.throttle_seconds,
        )
        failures = [
            failure
            for failure in (premium.failure, perpetual.failure, spot.failure)
            if failure is not None
        ]
    material: dict[str, object] = {
        "version": MEASUREMENT_JOURNAL_VERSION,
        "sequence": sequence,
        "spec_hash": spec_hash,
        "previous_segment_hash": ZERO_HASH if head is None else head.final_segment_hash,
        "received_time_ns": received_time_ns,
        "depth": depth,
        "premium_index": _dumped(premium_index),
        "perp_book": _dumped(perp_book),
        "spot_book": _dumped(spot_book),
        "failures": failures,
        "excluded": excluded,
    }
    segment = MeasurementSegment.model_validate(
        {**material, "content_hash": content_sha256(material)}
    )
    return segment, throttle_seconds


class _Snapshot[RowT](NamedTuple):
    """What one all-symbol request came back with.

    ``rows`` is ``None`` exactly when ``failure`` is set - a snapshot is the
    whole venue or the reason there is none - and ``excluded`` is then zero,
    because a payload the parser refused was not read to the end.
    """

    rows: tuple[RowT, ...] | None
    excluded: int
    throttle_seconds: int
    failure: str | None


def _snapshot[RowT](
    *,
    endpoint: str,
    url: str,
    array_fetcher: ArrayFetcher,
    parse: Callable[[Sequence[object]], tuple[tuple[RowT, ...], int]],
) -> _Snapshot[RowT]:
    """One all-symbol request, turned into a record whether it worked or not."""
    try:
        rows, excluded = parse(array_fetcher(url))
    except Exception as error:  # one endpoint's failure is a record, not an abort
        return _Snapshot(None, 0, _throttle_seconds(error), _endpoint_failure(endpoint, error))
    return _Snapshot(rows, excluded, 0, None)


def _endpoint_failure(endpoint: str, error: Exception) -> str:
    """``<endpoint>:<status or reason>``, the name a failed request wears (ruling 10).

    A venue that answered with a status of its own is named by that status -
    ``premiumIndex:429`` - because that is what a reader of the stream needs to
    tell a throttle from a parse failure; a payload this journal refused is
    named by the refusal.
    """
    if isinstance(error, BinanceCostJournalTransportError):
        reason = str(error.status)
    elif isinstance(error, DepthPayloadError):
        reason = str(error)
    else:
        reason = _failure_reason(error)
    return f"{endpoint}:{reason}"[:_MAX_REASON_CHARACTERS]


def _dumped[RowT: PremiumRow | BookRow](rows: tuple[RowT, ...] | None) -> object:
    if rows is None:
        return None
    return [row.model_dump(mode="json") for row in rows]


def _resumed_head(root: Path, *, spec_hash: str) -> MeasurementChainHead | None:
    """The head a run appends to, once the tail on disk has been judged.

    The order matters: the head is read first, because the one sequence a crash
    can catch mid-write is the one past it; the torn file there is discarded;
    then the *tail* is verified - the head against the newest segment, and the
    two newest day directories link by link (ruling 13). The whole history is
    not re-read: this stream is permanent and a restart is routine, so a
    restart costs a directory listing and two days of segments rather than
    every round since the journal began. The listing still covers the whole
    journal, so a deleted, duplicated or foreign segment anywhere is caught by
    the head's segment count or by the layout check.

    A tail that verifies against its head is resumed. A tail that verifies
    against *itself* while its head is missing, torn or behind the segments on
    disk is a killed publish, and the head the tail implies is written over it.
    Anything else refuses.
    """
    head = _resumable_chain_head(root)
    if head is not None and head.spec_hash != spec_hash:
        raise BinanceMeasurementJournalSpecError("this journal is bound to a different spec")
    _discard_torn_trailing_segment(root, head)
    walk = _walked_tail(root, spec_hash=spec_hash)
    if walk.reasons or not walk.walked:
        raise BinanceMeasurementJournalSpecError(
            "journal verification failed: " + ",".join(walk.reasons)
        )
    reasons = _head_mismatch_reasons(head, walk.implied_head)
    if not reasons:
        return walk.implied_head
    implied = walk.implied_head
    if implied is None or (head is not None and head.segment_count >= implied.segment_count):
        raise BinanceMeasurementJournalSpecError(
            "journal verification failed: " + ",".join(reasons)
        )
    # Wider than v1's single-orphan adoption on purpose: v1 names the one file
    # a kill can leave past the head, this one compares counts, so a head that
    # was never written, one a crash tore in half and one a kill left a segment
    # behind all take the same path - and all of them are judged by a tail that
    # verified first.
    _report_repair(f"CHAIN_HEAD_REBUILT:{implied.segment_count}")
    return _publish_head(root, implied)


def _verified_chain(
    journal_root: Path, *, spec_hash: str
) -> tuple[bool, tuple[str, ...], MeasurementChainHead | None]:
    """Recompute every segment's hash, every chain link and the chain head.

    This is the whole history, which is what an explicit verification is for;
    ``_verified_tail`` is the bounded reading a restart and a status command
    can afford. Never raises: an unreadable, foreign or malformed journal comes
    back as reason codes, the way ``verify_panel_capture`` reports a broken
    capture. An empty journal - a spec and no segment yet - is valid and
    implies no head. The third member is the head the segments on disk imply.
    """
    return _judged(journal_root, _walked_segments(journal_root, spec_hash=spec_hash))


def _verified_tail(
    journal_root: Path, *, spec_hash: str
) -> tuple[bool, tuple[str, ...], MeasurementChainHead | None]:
    """Judge the journal's tip without reading its history (ruling 13).

    The same answer shape as ``_verified_chain`` over the same layout check,
    but only the head and the segments of the two newest day directories are
    opened: the window's first segment is an anchor whose predecessor is not
    resolved (unless it is sequence zero, which must name ``ZERO_HASH``), and
    every segment after it must link to the one before. A journal whose tail
    verifies can still be broken further back, which is what ``_verified_chain``
    is for.
    """
    return _judged(journal_root, _walked_tail(journal_root, spec_hash=spec_hash))


class _ChainWalk(NamedTuple):
    """What a walk of ``segments/`` found: its reasons, its head, its footing.

    ``walked`` is false where there was no chain to walk at all - no directory,
    a name this journal did not write, a segment that will not parse - and the
    chain head is then not judged against a chain nobody could read.
    """

    reasons: tuple[str, ...]
    implied_head: MeasurementChainHead | None
    walked: bool


def _judged(
    journal_root: Path, walk: _ChainWalk
) -> tuple[bool, tuple[str, ...], MeasurementChainHead | None]:
    """A walk plus the head on disk, as one verification answer."""
    reasons = walk.reasons
    if walk.walked:
        head, head_reasons = _chain_head_on_disk(journal_root)
        reasons = (*reasons, *head_reasons)
        if not head_reasons:
            reasons = (*reasons, *_head_mismatch_reasons(head, walk.implied_head))
    return not reasons, reasons, walk.implied_head


def _walked_segments(journal_root: Path, *, spec_hash: str) -> _ChainWalk:
    """Walk every segment in sequence order, checking each against its place."""
    try:
        paths = _listed_segments(journal_root)
    except _SegmentLayoutError as error:
        return _ChainWalk((error.reason,), None, False)
    if paths is None:
        return _ChainWalk(("SEGMENT_DIRECTORY_MISSING",), None, False)
    return _walked(paths, journal_root=journal_root, spec_hash=spec_hash, total=len(paths))


def _walked_tail(journal_root: Path, *, spec_hash: str) -> _ChainWalk:
    """Walk the two newest day directories, anchored on the first segment in them.

    The layout of the whole journal is still listed - names, day directories
    and duplicate sequences are cheap and catch a deletion or an intruder
    anywhere - and the segment count that listing gives is what the chain head
    is judged against; only the window's files are opened.
    """
    try:
        paths = _listed_segments(journal_root)
    except _SegmentLayoutError as error:
        return _ChainWalk((error.reason,), None, False)
    if paths is None:
        return _ChainWalk(("SEGMENT_DIRECTORY_MISSING",), None, False)
    return _walked(
        _tail_window(paths),
        journal_root=journal_root,
        spec_hash=spec_hash,
        total=len(paths),
        anchored=True,
    )


def _listed_segments(journal_root: Path) -> list[Path] | None:
    """Every segment path in sequence order, or ``None`` where there is no directory."""
    if not (journal_root / SEGMENT_DIRECTORY_NAME).is_dir():
        return None
    return _segment_paths(journal_root)


def _tail_window(paths: list[Path]) -> list[Path]:
    """The segments of the newest day directory and the one before it by name."""
    if not paths:
        return []
    days = sorted({path.parent.name for path in paths})
    newest = days.index(paths[-1].parent.name)
    window = set(days[max(newest - 1, 0) : newest + 1])
    return [path for path in paths if path.parent.name in window]


def _walked(
    paths: list[Path],
    *,
    journal_root: Path,
    spec_hash: str,
    total: int,
    anchored: bool = False,
) -> _ChainWalk:
    """Check each segment of ``paths`` against its place in the chain.

    ``anchored`` starts the walk at the first segment's own sequence and its
    own recorded predecessor instead of at zero and ``ZERO_HASH``, which is
    what a bounded walk needs: the window's first segment is where the reading
    begins, not where the journal does. A window that does begin at sequence
    zero is held to ``ZERO_HASH`` all the same, which the segment model
    enforces anyway.
    """
    reasons: list[str] = []
    expected_sequence = 0
    previous_hash = ZERO_HASH
    anchor_pending = anchored
    last: MeasurementSegment | None = None
    for path in paths:
        name = _segment_name(journal_root, path)
        try:
            document = _read_object(path, label="a journal segment")
            segment = MeasurementSegment.model_validate(document)
        except (BinanceCostJournalError, ValidationError):
            reasons.append(f"SEGMENT_UNREADABLE:{name}")
            return _ChainWalk(tuple(reasons), None, False)
        if anchor_pending:
            anchor_pending = False
            expected_sequence = segment.sequence
            if segment.sequence != 0:
                previous_hash = segment.previous_segment_hash
        reasons.extend(
            _segment_reasons(
                document,
                segment,
                name=name,
                expected_sequence=expected_sequence,
                spec_hash=spec_hash,
                previous_hash=previous_hash,
            )
        )
        expected_sequence += 1
        previous_hash = segment.content_hash
        last = segment
    implied = None if last is None else _chain_head_for(last, spec_hash=spec_hash, count=total)
    return _ChainWalk(tuple(reasons), implied, True)


def _segment_reasons(
    document: Mapping[str, object],
    segment: MeasurementSegment,
    *,
    name: str,
    expected_sequence: int,
    spec_hash: str,
    previous_hash: str,
) -> list[str]:
    """Everything one segment can be wrong about, given where it sits in the chain."""
    material = {key: value for key, value in document.items() if key != "content_hash"}
    reasons: list[str] = []
    if segment.content_hash != content_sha256(material):
        reasons.append(f"SEGMENT_HASH_MISMATCH:{name}")
    if segment.sequence != expected_sequence:
        reasons.append(f"SEGMENT_SEQUENCE_GAP:{name}")
    if segment.spec_hash != spec_hash:
        reasons.append(f"SEGMENT_SPEC_MISMATCH:{name}")
    if segment.previous_segment_hash != previous_hash:
        reasons.append(f"CHAIN_LINK_BROKEN:{name}")
    return reasons


def _head_mismatch_reasons(
    head: MeasurementChainHead | None, implied: MeasurementChainHead | None
) -> tuple[str, ...]:
    """How the head on disk differs from the head the segments imply."""
    if head is None:
        return () if implied is None else ("CHAIN_HEAD_MISSING",)
    if implied is None:
        return ("CHAIN_HEAD_UNEXPECTED",)
    fields = ("spec_hash", "segment_count", "last_sequence", "final_segment_hash")
    return tuple(
        f"CHAIN_HEAD_MISMATCH:{field}"
        for field in fields
        if getattr(head, field) != getattr(implied, field)
    )


def _segment_paths(journal_root: Path) -> list[Path]:
    """This journal's segments in sequence order, across the day directories.

    The layout is ``segments/<YYYY-MM-DD>/<sequence:010d>.json``, the day taken
    from the segment's own ``received_time_ns`` in UTC, so a journal that runs
    for years is still a directory a person can open. The order is the
    sequence's, never the name's: a clock that stepped over a day boundary
    leaves the chain's order and the directories' order disagreeing, and the
    chain's is the one that means anything.

    A ``<sequence>.json.<pid>.tmp`` is a publish that died before its rename
    and is ignored, as in v1. Any other name under ``segments/`` is refused:
    this directory holds what this journal wrote and nothing else.
    """
    directory = journal_root / SEGMENT_DIRECTORY_NAME
    if not directory.is_dir():
        return []
    found: dict[int, Path] = {}
    for day in sorted(directory.iterdir()):
        if not day.is_dir() or not _DAY_NAME.match(day.name):
            raise _SegmentLayoutError(f"SEGMENT_LAYOUT:{day.name}")
        for path in sorted(day.iterdir()):
            name = f"{day.name}/{path.name}"
            if _SEGMENT_TEMPORARY_NAME.match(path.name):
                continue
            if not path.is_file() or not _SEGMENT_NAME.match(path.name):
                raise _SegmentLayoutError(f"SEGMENT_LAYOUT:{name}")
            sequence = int(path.stem)
            if sequence in found:
                raise _SegmentLayoutError(f"SEGMENT_DUPLICATE:{name}")
            found[sequence] = path
    return [found[sequence] for sequence in sorted(found)]


class _SegmentLayoutError(BinanceMeasurementJournalSpecError):
    """A name under ``segments/`` that this journal did not write.

    It carries the verification's reason code as well as its sentence, so the
    walk can refuse a caller and report a reader with one check.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(f"the segment directory holds a name this journal did not write: {reason}")
        self.reason = reason


def _segment_name(journal_root: Path, path: Path) -> str:
    return path.relative_to(journal_root / SEGMENT_DIRECTORY_NAME).as_posix()


def _segment_day(received_time_ns: int) -> str:
    """The UTC day a stamp falls in, as the directory names it."""
    seconds = received_time_ns // _NANOSECONDS_A_SECOND
    return datetime.fromtimestamp(seconds, tz=UTC).strftime("%Y-%m-%d")


def _publish_segment(
    root: Path, *, segment: MeasurementSegment, spec_hash: str
) -> MeasurementChainHead:
    """Write the segment under its day, then the head that names it."""
    path = (
        root
        / SEGMENT_DIRECTORY_NAME
        / _segment_day(segment.received_time_ns)
        / f"{segment.sequence:010d}.json"
    )
    document: dict[str, object] = segment.model_dump(mode="json")
    _publish(path, document)
    return _publish_head(root, _chain_head_for(segment, spec_hash=spec_hash))


def _chain_head_for(
    segment: MeasurementSegment, *, spec_hash: str, count: int | None = None
) -> MeasurementChainHead:
    """The head a chain ending in ``segment`` would carry."""
    material: dict[str, object] = {
        "version": MEASUREMENT_JOURNAL_VERSION,
        "spec_hash": spec_hash,
        "segment_count": segment.sequence + 1 if count is None else count,
        "last_sequence": segment.sequence,
        "final_segment_hash": segment.content_hash,
    }
    return MeasurementChainHead.model_validate(
        {**material, "content_hash": content_sha256(material)}
    )


def _publish_head(root: Path, head: MeasurementChainHead) -> MeasurementChainHead:
    document: dict[str, object] = head.model_dump(mode="json")
    _publish(root / CHAIN_HEAD_NAME, document)
    return head


def _chain_head_on_disk(
    journal_root: Path,
) -> tuple[MeasurementChainHead | None, tuple[str, ...]]:
    """The sealed head this journal carries, or why it cannot be read."""
    path = journal_root / CHAIN_HEAD_NAME
    if not path.exists():
        return None, ()
    try:
        document = _read_object(path, label="the chain head")
    except BinanceCostJournalError:
        return None, ("CHAIN_HEAD_UNREADABLE",)
    head = _validated_chain_head(document)
    if head is None:
        return None, ("CHAIN_HEAD_UNREADABLE",)
    return head, ()


def _resumable_chain_head(journal_root: Path) -> MeasurementChainHead | None:
    """The head a run resumes from: ``None`` where there is none or it is torn.

    A head file that is not parseable JSON is a publish the last process did
    not finish - half of one head and half of another - and the segments say
    what it should have been, so the run rebuilds it. A head that *parses* and
    then fails its own recorded hash is not torn but changed, and stays fatal.
    """
    path = journal_root / CHAIN_HEAD_NAME
    if not path.exists():
        return None
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise BinanceMeasurementJournalSpecError(
            f"the chain head is unreadable: {error}"
        ) from error
    try:
        document: object = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        _report_repair(f"CHAIN_HEAD_TORN:{CHAIN_HEAD_NAME}")
        return None
    if not isinstance(document, dict):
        _report_repair(f"CHAIN_HEAD_TORN:{CHAIN_HEAD_NAME}")
        return None
    head = _validated_chain_head(document)
    if head is None:
        raise BinanceMeasurementJournalSpecError(
            "the chain head is not a sealed head of this journal"
        )
    return head


def _validated_chain_head(document: Mapping[str, object]) -> MeasurementChainHead | None:
    try:
        head = MeasurementChainHead.model_validate(document)
    except ValidationError:
        return None
    material = {key: value for key, value in document.items() if key != "content_hash"}
    if head.content_hash != content_sha256(material):
        return None
    return head


def _discard_torn_trailing_segment(
    journal_root: Path, head: MeasurementChainHead | None
) -> None:
    """Delete the half-written file a killed publish left one past the head.

    Every segment before that one was fsynced before the head that names it, so
    the one sequence a crash can catch mid-write is the next one. A file there
    that is not parseable JSON, or does not validate as a segment, is that
    crash caught in the act: it carries no measurement, it is deleted and its
    sequence is sampled again. A file there that *is* a valid segment is the
    orphan a resume adopts, and not this function's business.

    It is named more widely than v1's, which opens exactly
    ``<head.last_sequence + 1>.json``: under the day-partitioned layout that
    file's directory is not known without a listing - a round either side of
    midnight writes into a different day - so the trailing file is found by
    sequence instead, as the last of the listing, and inspected only if it sits
    past the head. Exactly one file is ever opened here.
    """
    paths = _segment_paths(journal_root)
    if not paths:
        return
    path = paths[-1]
    if head is not None and int(path.stem) <= head.last_sequence:
        return
    try:
        raw = path.read_bytes()
    except OSError:
        return
    if _parses_as_segment(raw):
        return
    path.unlink()
    _report_repair(f"SEGMENT_TORN_DISCARDED:{_segment_name(journal_root, path)}")


def _parses_as_segment(raw: bytes) -> bool:
    try:
        document: object = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(document, dict):
        return False
    try:
        MeasurementSegment.model_validate(document)
    except ValidationError:
        return False
    return True


def _report_repair(reason: str) -> None:
    """Name a repair on stderr, the one thing this module writes anywhere.

    There is no logging framework here and this journal runs unattended for as
    long as the machine does: a file this run deleted or a head it rebuilt has
    to be readable afterwards in the supervisor's transcript, or the operator
    is left comparing sequence numbers to work out what happened.
    """
    print(f"binance measurement journal repair: {reason}", file=sys.stderr)
