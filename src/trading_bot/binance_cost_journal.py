"""Models and book arithmetic for the Binance public cost journal.

The journal measures what a taker pays to cross a spot leg and a USD-M perpetual
leg on Binance's public order books at three notionals. This module holds the
frozen record shapes and the two pure functions the sampler needs: the walk of a
displayed book to a quote notional and the parse of one depth payload into an
observation. Nothing here reaches the network.
"""

import re
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from trading_bot.depth_adapters import (
    DepthPayloadError,
    _decimal_string,
    _integer,
    _level_values,
    _string,
)

JOURNAL_VERSION = "binance-cost-journal/1.0.0"
NOTIONALS: tuple[Decimal, ...] = (Decimal("500"), Decimal("5000"), Decimal("50000"))
DEPTH_LIMIT = 500
SAMPLE_INTERVAL_SECONDS = 61
TARGET_ROUNDS = 10_000
MINIMUM_OBSERVATIONS = 10_000
MINIMUM_SPAN_NS = 7 * 86_400_000_000_000
ALLOWED_HOSTS = frozenset({"api.binance.com", "fapi.binance.com"})
ZERO_HASH = "0" * 64

_BPS = Decimal(10_000)
_MARKET_HOSTS: Mapping[str, str] = {"spot": "api.binance.com", "um": "fapi.binance.com"}
_MARKET_PREFIXES: Mapping[str, str] = {"spot": "spot", "um": "perp"}
_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")
_SYMBOL = re.compile(r"\A[A-Z0-9]{4,24}\Z")
_IDENTIFIER = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9.:_-]{0,79}\Z")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _validated_hex(value: str, *, field_name: str) -> str:
    if not _HEX64.match(value):
        raise ValueError(f"{field_name} must be a lower-case 64-character SHA-256 digest")
    return value


def _validate_public_url(value: str, *, host: str, field_name: str) -> None:
    parts = urlsplit(value)
    if host not in ALLOWED_HOSTS or parts.scheme != "https" or parts.netloc != host:
        raise ValueError(f"{field_name} must be an HTTPS URL on {host}")


class JournalInstrument(_Frozen):
    """One sampled leg: a Binance market, its symbol and its public endpoints."""

    instrument_id: str
    market: Literal["spot", "um"]
    symbol: str
    pair_symbol: str
    tier: Literal[1, 2]
    depth_url: str
    premium_index_url: str | None

    @field_validator("symbol", "pair_symbol")
    @classmethod
    def validate_symbol(cls, value: str) -> str:
        if not _SYMBOL.match(value):
            raise ValueError("symbols are upper-case venue symbols")
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        expected_id = f"{_MARKET_PREFIXES[self.market]}:{self.symbol}"
        if self.instrument_id != expected_id:
            raise ValueError(f"instrument_id must be {expected_id}")
        host = _MARKET_HOSTS[self.market]
        _validate_public_url(self.depth_url, host=host, field_name="depth_url")
        if self.market == "spot":
            if self.premium_index_url is not None:
                raise ValueError("a spot leg has no premium index endpoint")
        elif self.premium_index_url is None:
            raise ValueError("a perpetual leg requires a premium index endpoint")
        else:
            _validate_public_url(self.premium_index_url, host=host, field_name="premium_index_url")
        return self


class BinanceCostJournalSpec(_Frozen):
    """The frozen declaration a journal directory is bound to."""

    version: Literal["binance-cost-journal/1.0.0"]
    run_id: str
    created_time_ns: int
    sample_decision_close_ns: int
    perpetual_capture_root_hash: str
    spot_capture_root_hash: str
    instruments: tuple[JournalInstrument, ...]
    notionals: tuple[Decimal, ...]
    depth_limit: int
    sample_interval_seconds: int
    target_rounds: int
    spot_fee_bps_per_side: Decimal
    perpetual_fee_bps_per_side: Decimal
    fee_evidence_id: str

    @field_validator("perpetual_capture_root_hash", "spot_capture_root_hash")
    @classmethod
    def validate_hashes(cls, value: str) -> str:
        return _validated_hex(value, field_name="capture root hash")

    @field_validator("run_id", "fee_evidence_id")
    @classmethod
    def validate_identifiers(cls, value: str) -> str:
        if not _IDENTIFIER.match(value):
            raise ValueError("identifiers are non-empty and free of whitespace")
        return value

    @field_validator("created_time_ns", "sample_decision_close_ns")
    @classmethod
    def validate_times(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("times are positive nanosecond stamps")
        return value

    @field_validator("depth_limit", "sample_interval_seconds", "target_rounds")
    @classmethod
    def validate_counts(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("depth limit, interval and target rounds are positive")
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


class InstrumentObservation(_Frozen):
    """One instrument's cost measurement in one round, or the reason there is none."""

    instrument_id: str
    received_time_ns: int
    ok: bool
    reason: str | None
    spread_bps: Decimal | None
    slippage_bps_per_side: dict[str, Decimal | None]
    displayed_notional_thinner_side: Decimal | None
    funding_rate: Decimal | None
    basis_bps: Decimal | None

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        if not self.instrument_id:
            raise ValueError("instrument_id must be set")
        if self.received_time_ns <= 0:
            raise ValueError("received_time_ns must be a positive nanosecond stamp")
        if not self.slippage_bps_per_side:
            raise ValueError("an observation carries one entry per sampled notional")
        if self.ok:
            return self._validate_measured()
        if not self.reason:
            raise ValueError("a failed observation carries a reason")
        if any(
            value is not None
            for value in (
                self.spread_bps,
                self.displayed_notional_thinner_side,
                self.funding_rate,
                self.basis_bps,
                *self.slippage_bps_per_side.values(),
            )
        ):
            raise ValueError("a failed observation carries no measurement")
        return self

    def _validate_measured(self) -> Self:
        if self.reason is not None:
            raise ValueError("a measured observation carries no reason")
        if self.spread_bps is None or self.displayed_notional_thinner_side is None:
            raise ValueError("a measured observation carries spread and displayed notional")
        if self.spread_bps <= 0 or self.displayed_notional_thinner_side <= 0:
            raise ValueError("spread and displayed notional are positive")
        if any(value is not None and value < 0 for value in self.slippage_bps_per_side.values()):
            raise ValueError("slippage cannot be negative on an uncrossed book")
        return self


class JournalSegment(_Frozen):
    """One round of observations, linked to its predecessor by hash."""

    version: Literal["binance-cost-journal/1.0.0"]
    sequence: int
    spec_hash: str
    previous_segment_hash: str
    received_time_ns: int
    observations: tuple[InstrumentObservation, ...]
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

    @field_validator("observations")
    @classmethod
    def validate_observations(
        cls, value: tuple[InstrumentObservation, ...]
    ) -> tuple[InstrumentObservation, ...]:
        if not value:
            raise ValueError("a segment holds at least one observation")
        identifiers = [item.instrument_id for item in value]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("an instrument is observed at most once per round")
        return value

    @model_validator(mode="after")
    def validate_chain_link(self) -> Self:
        if self.sequence == 0 and self.previous_segment_hash != ZERO_HASH:
            raise ValueError("the first segment has no predecessor")
        if self.sequence > 0 and self.previous_segment_hash == ZERO_HASH:
            raise ValueError("a later segment names its predecessor")
        return self


class ChainHead(_Frozen):
    """The resumable tip of a journal's segment chain."""

    version: Literal["binance-cost-journal/1.0.0"]
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


def walk_notional(levels: Sequence[tuple[Decimal, Decimal]], notional: Decimal) -> Decimal | None:
    """Return the VWAP paid to fill ``notional`` quote units from ``levels``.

    ``levels`` are ``(price, quantity)`` pairs ordered best first. The last level
    touched is filled partially. ``None`` means the displayed book cannot fill the
    notional. A non-positive notional or level is a caller error and raises.
    """
    if notional <= 0:
        raise ValueError("notional must be positive")
    remaining = notional
    filled_base = Decimal(0)
    for price, quantity in levels:
        if price <= 0 or quantity <= 0:
            raise ValueError("book levels must carry a positive price and quantity")
        level_quote = price * quantity
        if level_quote >= remaining:
            return notional / (filled_base + remaining / price)
        remaining -= level_quote
        filled_base += quantity
    return None


def depth_observation(
    payload: Mapping[str, object],
    *,
    instrument: JournalInstrument,
    received_time_ns: int,
    notionals: Sequence[Decimal],
    premium_index: Mapping[str, object] | None,
) -> InstrumentObservation:
    """Turn one Binance REST depth payload into an observation.

    ``payload`` carries ``lastUpdateId`` and the ``bids``/``asks`` level arrays;
    ``premium_index`` carries ``markPrice``, ``indexPrice`` and ``lastFundingRate``
    and belongs to perpetual legs only. Any payload defect - malformed, empty or
    crossed book, missing or foreign premium index - becomes ``ok=False`` with a
    reason instead of an exception; the notional keys are always present.
    """
    keys = _notional_keys(notionals)
    try:
        bids, asks = _validated_book(payload)
        funding_rate, basis_bps = _premium_values(premium_index, instrument=instrument)
    except DepthPayloadError as error:
        return _failed_observation(
            instrument_id=instrument.instrument_id,
            received_time_ns=received_time_ns,
            keys=keys,
            reason=str(error),
        )

    best_bid = bids[0][0]
    best_ask = asks[0][0]
    mid = (best_bid + best_ask) / Decimal(2)
    slippage: dict[str, Decimal | None] = {}
    for key, notional in zip(keys, notionals, strict=True):
        buy_vwap = walk_notional(asks, notional)
        sell_vwap = walk_notional(bids, notional)
        if buy_vwap is None or sell_vwap is None:
            slippage[key] = None
            continue
        slippage[key] = max(buy_vwap / mid - 1, 1 - sell_vwap / mid) * _BPS

    return InstrumentObservation(
        instrument_id=instrument.instrument_id,
        received_time_ns=received_time_ns,
        ok=True,
        reason=None,
        spread_bps=(best_ask - best_bid) / mid * _BPS,
        slippage_bps_per_side=slippage,
        displayed_notional_thinner_side=min(_displayed_notional(bids), _displayed_notional(asks)),
        funding_rate=funding_rate,
        basis_bps=basis_bps,
    )


def _notional_keys(notionals: Sequence[Decimal]) -> tuple[str, ...]:
    if not notionals:
        raise ValueError("at least one notional must be sampled")
    keys = tuple(str(notional) for notional in notionals)
    if len(set(keys)) != len(keys):
        raise ValueError("sampled notionals must be distinct")
    return keys


def _failed_observation(
    *, instrument_id: str, received_time_ns: int, keys: Sequence[str], reason: str
) -> InstrumentObservation:
    empty: dict[str, Decimal | None] = dict.fromkeys(keys)
    return InstrumentObservation(
        instrument_id=instrument_id,
        received_time_ns=received_time_ns,
        ok=False,
        reason=reason,
        spread_bps=None,
        slippage_bps_per_side=empty,
        displayed_notional_thinner_side=None,
        funding_rate=None,
        basis_bps=None,
    )


def _validated_book(
    payload: Mapping[str, object],
) -> tuple[tuple[tuple[Decimal, Decimal], ...], tuple[tuple[Decimal, Decimal], ...]]:
    _integer(payload.get("lastUpdateId"), field_name="lastUpdateId")
    bids = _validated_side(payload.get("bids"), field_name="bids", ascending=False)
    asks = _validated_side(payload.get("asks"), field_name="asks", ascending=True)
    if not bids or not asks:
        raise DepthPayloadError("empty book")
    if bids[0][0] >= asks[0][0]:
        raise DepthPayloadError("crossed book")
    return bids, asks


def _validated_side(
    value: object, *, field_name: str, ascending: bool
) -> tuple[tuple[Decimal, Decimal], ...]:
    levels = _level_values(value, field_name=field_name)
    previous: Decimal | None = None
    for price, quantity in levels:
        if price <= 0 or quantity <= 0:
            raise DepthPayloadError(f"{field_name} levels must be positive")
        if previous is not None and not (price > previous if ascending else price < previous):
            raise DepthPayloadError(f"{field_name} must be sorted best first")
        previous = price
    return levels


def _premium_values(
    premium_index: Mapping[str, object] | None, *, instrument: JournalInstrument
) -> tuple[Decimal | None, Decimal | None]:
    if instrument.premium_index_url is None:
        if premium_index is not None:
            raise DepthPayloadError("unexpected premium index")
        return None, None
    if premium_index is None:
        raise DepthPayloadError("missing premium index")
    symbol = _string(premium_index.get("symbol"), field_name="premiumIndex.symbol")
    if symbol != instrument.symbol:
        raise DepthPayloadError("premium index symbol mismatch")
    mark_price = _decimal_string(premium_index.get("markPrice"), field_name="markPrice")
    index_price = _decimal_string(premium_index.get("indexPrice"), field_name="indexPrice")
    funding_rate = _decimal_string(
        premium_index.get("lastFundingRate"), field_name="lastFundingRate"
    )
    if mark_price <= 0 or index_price <= 0:
        raise DepthPayloadError("premium index prices must be positive")
    return funding_rate, (mark_price - index_price) / index_price * _BPS


def _displayed_notional(levels: Sequence[tuple[Decimal, Decimal]]) -> Decimal:
    return sum((price * quantity for price, quantity in levels), Decimal(0))
