"""The Binance public cost journal: models, book arithmetic and the sampler.

The journal measures what a taker pays to cross a spot leg and a USD-M perpetual
leg on Binance's public order books at three notionals. This module holds the
frozen record shapes, the two pure functions the sampler needs - the walk of a
displayed book to a quote notional and the parse of one depth payload into an
observation - and the journal itself: the sample derived from the two carry
captures, the hash-chained run loop, its verification and the one fetcher that
is allowed to reach Binance.
"""

import http.client
import json
import os
import re
import shutil
import time
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from pathlib import Path
from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.carry_config import CarryUniverseRules, load_carry_family_spec
from trading_bot.carry_signals import WEEK_NS
from trading_bot.carry_universe import select_pair_universe
from trading_bot.depth_adapters import (
    DepthPayloadError,
    _decimal_string,
    _integer,
    _level_values,
    _mapping,
    _string,
)
from trading_bot.panel_capture import verify_panel_capture
from trading_bot.panel_reader import load_panel_bars
from trading_bot.panel_universe import ContractHistory, build_contract_histories
from trading_bot.storage import StoragePolicy

JOURNAL_VERSION = "binance-cost-journal/1.0.0"
NOTIONALS: tuple[Decimal, ...] = (Decimal("500"), Decimal("5000"), Decimal("50000"))
DEPTH_LIMIT = 500
SAMPLE_INTERVAL_SECONDS = 61
# 11 000 rounds at 61 s is 7.8 days: the eligibility floor below wants 10 000
# non-null observations over at least 7 days, and the margin pays for the rounds
# a venue outage or a restart costs.
TARGET_ROUNDS = 11_000
MINIMUM_OBSERVATIONS = 10_000
MINIMUM_SPAN_NS = 7 * 86_400_000_000_000
ALLOWED_HOSTS = frozenset({"api.binance.com", "fapi.binance.com"})
ZERO_HASH = "0" * 64
# Spec 2: fees are declared, never measured. The evidence id names the schedule
# the two rates were read from.
SPOT_FEE_BPS_PER_SIDE = Decimal("10")
PERPETUAL_FEE_BPS_PER_SIDE = Decimal("5")
FEE_EVIDENCE_ID = "BINANCE:fee-schedule:2026-09-11:standard-taker"
# Spec 3: the sample is ranks 1-8 of the pair universe plus eight drawn evenly
# from ranks 9-35. The journal's own tier, not the carry family's.
SAMPLE_TIER_ONE_RANKS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
SAMPLE_TIER_TWO_RANKS: tuple[int, ...] = (9, 12, 15, 19, 23, 27, 31, 35)
JOURNAL_SPEC_NAME = "journal-spec.json"
CHAIN_HEAD_NAME = "chain-head.json"
SEGMENT_DIRECTORY_NAME = "segments"

# url -> parsed JSON object; raises on transport failure.
type Fetcher = Callable[[str], Mapping[str, object]]

_BPS = Decimal(10_000)
_BPS_QUANTUM = Decimal("0.000001")
_NOTIONAL_QUANTUM = Decimal("0.01")
_MARKET_HOSTS: Mapping[str, str] = {"spot": "api.binance.com", "um": "fapi.binance.com"}
_MARKET_PREFIXES: Mapping[str, str] = {"spot": "spot", "um": "perp"}
_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")
_SYMBOL = re.compile(r"\A[A-Z0-9]{4,24}\Z")
_IDENTIFIER = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9.:_-]{0,79}\Z")
_SEGMENT_NAME = re.compile(r"\A[0-9]{10}\.json\Z")
# `<sequence>.json.<pid>.tmp` (and the bare `.tmp` an older build wrote):
# a publish that died before its replace(), which the next attempt at that
# sequence overwrites.
_SEGMENT_TEMPORARY_NAME = re.compile(r"\A[0-9]{10}\.json(\.[0-9]+)?\.tmp\Z")
_WORST_CASE_REQUIRED_BYTES = 500_000_000
_FETCH_TIMEOUT_SECONDS = 20
# A 500-level depth payload is ~30 kB; the cap only bounds a hostile response.
_MAX_RESPONSE_BYTES = 8_000_000
_MAX_REASON_CHARACTERS = 200


class BinanceCostJournalError(RuntimeError):
    """Raised when a journal cannot be created, sampled or read safely."""


class BinanceCostJournalSpecError(BinanceCostJournalError):
    """Raised when the request does not match the journal on disk.

    The CLI maps this to exit code 2 and every other failure to 1, so the
    supervisor stops on a mismatched, missing or unverifiable journal instead
    of restarting a process that can only fail the same way again.
    """


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
    # A perpetual whose premium index failed while its book was measured: the
    # observation still counts, its two funding fields are simply absent.
    premium_index_reason: str | None = None

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        if not self.instrument_id:
            raise ValueError("instrument_id must be set")
        if self.received_time_ns <= 0:
            raise ValueError("received_time_ns must be a positive nanosecond stamp")
        if not self.slippage_bps_per_side:
            raise ValueError("an observation carries one entry per sampled notional")
        if self.premium_index_reason == "":
            raise ValueError("a premium index reason is either absent or set")
        if self.ok:
            return self._validate_measured()
        if not self.reason:
            raise ValueError("a failed observation carries a reason")
        if self.premium_index_reason is not None:
            raise ValueError("a failed observation carries one reason")
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
        if self.premium_index_reason is not None and (
            self.funding_rate is not None or self.basis_bps is not None
        ):
            raise ValueError("an unusable premium index leaves no funding rate and no basis")
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
    payload: object,
    *,
    instrument: JournalInstrument,
    received_time_ns: int,
    notionals: Sequence[Decimal],
    premium_index: object,
) -> InstrumentObservation:
    """Turn one Binance REST depth payload into an observation.

    ``payload`` and ``premium_index`` are parsed JSON of unknown shape: the depth
    object carries ``lastUpdateId`` and the ``bids``/``asks`` level arrays, the
    premium index object ``markPrice``, ``indexPrice`` and ``lastFundingRate``,
    and the latter belongs to perpetual legs only (``None`` where a leg has none).
    Any defect in the book - a value that is not an object at all, a malformed,
    empty or crossed book, a book below the recorded precision - becomes
    ``ok=False`` with a reason instead of an exception; the notional keys are
    always present. A missing or unusable premium index does *not* void the
    measured book: the observation stays ``ok`` with no funding rate, no basis
    and the defect in ``premium_index_reason``.

    Every measured value is recorded at the journal's declared precision:
    1e-6 bps for the spread, the slippage and the basis, 0.01 quote units for the
    displayed notional, half-even. The funding rate is stored as received.
    """
    keys = _notional_keys(notionals)
    try:
        bids, asks = _validated_book(payload)
        funding_rate, basis_bps, premium_index_reason = _premium_values(
            premium_index, instrument=instrument
        )
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid = (best_bid + best_ask) / Decimal(2)
        spread_bps = _quantised((best_ask - best_bid) / mid * _BPS, _BPS_QUANTUM)
        displayed = _quantised(
            min(_displayed_notional(bids), _displayed_notional(asks)), _NOTIONAL_QUANTUM
        )
        if spread_bps <= 0 or displayed <= 0:
            raise DepthPayloadError("book below the recorded precision")
        slippage: dict[str, Decimal | None] = {}
        for key, notional in zip(keys, notionals, strict=True):
            buy_vwap = walk_notional(asks, notional)
            sell_vwap = walk_notional(bids, notional)
            if buy_vwap is None or sell_vwap is None:
                slippage[key] = None
                continue
            worse_side = max(buy_vwap / mid - 1, 1 - sell_vwap / mid) * _BPS
            slippage[key] = _quantised(worse_side, _BPS_QUANTUM)
    except DepthPayloadError as error:
        return _failed_observation(
            instrument_id=instrument.instrument_id,
            received_time_ns=received_time_ns,
            keys=keys,
            reason=str(error),
        )

    return InstrumentObservation(
        instrument_id=instrument.instrument_id,
        received_time_ns=received_time_ns,
        ok=True,
        reason=None,
        spread_bps=spread_bps,
        slippage_bps_per_side=slippage,
        displayed_notional_thinner_side=displayed,
        funding_rate=funding_rate,
        basis_bps=basis_bps,
        premium_index_reason=premium_index_reason,
    )


def _quantised(value: Decimal, quantum: Decimal) -> Decimal:
    """Record a measured value at the journal's declared precision."""
    try:
        return value.quantize(quantum, rounding=ROUND_HALF_EVEN)
    except InvalidOperation as error:
        raise DepthPayloadError("measurement exceeds the recorded precision") from error


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
    payload: object,
) -> tuple[tuple[tuple[Decimal, Decimal], ...], tuple[tuple[Decimal, Decimal], ...]]:
    document = _mapping(payload, field_name="depth payload")
    _integer(document.get("lastUpdateId"), field_name="lastUpdateId")
    bids = _validated_side(document.get("bids"), field_name="bids", ascending=False)
    asks = _validated_side(document.get("asks"), field_name="asks", ascending=True)
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
    premium_index: object, *, instrument: JournalInstrument
) -> tuple[Decimal | None, Decimal | None, str | None]:
    """Return ``(funding_rate, basis_bps, premium_index_reason)``.

    A perpetual's premium index is a second request beside its depth: when it
    is missing or unusable the book was still measured, so the defect is
    recorded as the third member and the observation stays ``ok``. Only a spot
    leg handed a premium index raises - that is the caller pairing an
    instrument with the wrong payload, not the venue failing.
    """
    if instrument.premium_index_url is None:
        if premium_index is not None:
            raise DepthPayloadError("unexpected premium index")
        return None, None, None
    if premium_index is None:
        return None, None, "missing premium index"
    try:
        document = _mapping(premium_index, field_name="premiumIndex")
        symbol = _string(document.get("symbol"), field_name="premiumIndex.symbol")
        if symbol != instrument.symbol:
            raise DepthPayloadError("premium index symbol mismatch")
        mark_price = _decimal_string(document.get("markPrice"), field_name="markPrice")
        index_price = _decimal_string(document.get("indexPrice"), field_name="indexPrice")
        funding_rate = _decimal_string(
            document.get("lastFundingRate"), field_name="lastFundingRate"
        )
        if mark_price <= 0 or index_price <= 0:
            raise DepthPayloadError("premium index prices must be positive")
        basis_bps = _quantised((mark_price - index_price) / index_price * _BPS, _BPS_QUANTUM)
    except DepthPayloadError as error:
        return None, None, str(error)
    return funding_rate, basis_bps, None


def _displayed_notional(levels: Sequence[tuple[Decimal, Decimal]]) -> Decimal:
    return sum((price * quantity for price, quantity in levels), Decimal(0))


def derive_sample(
    perp_capture_root: Path,
    spot_capture_root: Path,
    *,
    family_spec_path: Path,
) -> tuple[int, tuple[JournalInstrument, ...]]:
    """Derive the frozen instrument sample from the two carry captures.

    The decision is the latest one both captures support: the earlier of their
    two last daily closes, stepped back one week, because the family's
    liquidity rule needs a complete window ending at the decision and the last
    week of a capture is the one most likely to be incomplete. The universe is
    the family's own pair rule with ``minimum_pairs`` lowered to one for this
    derivation only - the sample must exist even in a thin regime, and a
    journal that measures fewer pairs is still a measurement. Ranks 1-8 become
    the journal's tier one and ranks 9, 12, 15, 19, 23, 27, 31, 35 its tier two
    (the journal's tier, not the family's); a rank the universe does not reach
    is simply absent. Both legs of every selected pair are sampled, spot first.
    """
    spec, _ = load_carry_family_spec(family_spec_path)
    perp_histories = _capture_histories(perp_capture_root, label="perpetual")
    spot_histories = _capture_histories(spot_capture_root, label="spot")
    last_supported_close_ns = min(_last_close_ns(perp_histories), _last_close_ns(spot_histories))
    decision_close_ns = last_supported_close_ns - WEEK_NS
    rules = CarryUniverseRules(**{**spec.universe.model_dump(), "minimum_pairs": 1})
    snapshot = select_pair_universe(
        perp_histories,
        spot_histories,
        pairs=spec.pairs,
        decision_close_ns=decision_close_ns,
        rules=rules,
    )
    if not snapshot.pairs:
        raise BinanceCostJournalError(
            "the pair universe is empty at the latest supported decision: "
            + ",".join(snapshot.reason_codes)
        )
    by_rank = {pair.liquidity_rank: pair for pair in snapshot.pairs}
    ranks_by_tier: tuple[tuple[Literal[1, 2], tuple[int, ...]], ...] = (
        (1, SAMPLE_TIER_ONE_RANKS),
        (2, SAMPLE_TIER_TWO_RANKS),
    )
    instruments: list[JournalInstrument] = []
    for tier, ranks in ranks_by_tier:
        for rank in ranks:
            pair = by_rank.get(rank)
            if pair is None:
                continue
            perpetual_symbol = perp_histories[pair.perpetual_contract_id].instrument_id
            spot_symbol = spot_histories[pair.spot_contract_id].instrument_id
            instruments.append(_spot_leg(spot_symbol, pair_symbol=perpetual_symbol, tier=tier))
            instruments.append(_perpetual_leg(perpetual_symbol, tier=tier))
    return decision_close_ns, tuple(instruments)


def create_journal(
    *,
    workspace_root: Path,
    journal_root: Path,
    reserve_bytes: int,
    run_id: str,
    perp_capture_root: Path,
    spot_capture_root: Path,
    family_spec_path: Path,
) -> Path:
    """Write ``<journal_root>/journal-spec.json`` and an empty ``segments/``.

    The spec is immutable and binds the journal directory to it, so a root that
    already holds one is refused: a new sample is a new journal (spec 3).
    Returns the path of the spec.
    """
    root = _authorize(workspace_root, journal_root, reserve_bytes)
    spec_path = root / JOURNAL_SPEC_NAME
    if spec_path.exists():
        raise BinanceCostJournalSpecError("this journal already exists and is immutable")
    perpetual_root = perp_capture_root.resolve()
    spot_root = spot_capture_root.resolve()
    if perpetual_root == spot_root:
        raise BinanceCostJournalSpecError("the spot capture must differ from the perpetual one")
    perpetual_hash = _capture_root_hash(perpetual_root, market="um", label="perpetual")
    spot_hash = _capture_root_hash(spot_root, market="spot", label="spot")
    decision_close_ns, instruments = derive_sample(
        perpetual_root, spot_root, family_spec_path=family_spec_path
    )
    document: dict[str, object] = {
        "version": JOURNAL_VERSION,
        "run_id": run_id,
        "created_time_ns": time.time_ns(),
        "sample_decision_close_ns": decision_close_ns,
        "perpetual_capture_root_hash": perpetual_hash,
        "spot_capture_root_hash": spot_hash,
        "instruments": [instrument.model_dump(mode="json") for instrument in instruments],
        "notionals": [str(notional) for notional in NOTIONALS],
        "depth_limit": DEPTH_LIMIT,
        "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
        "target_rounds": TARGET_ROUNDS,
        "spot_fee_bps_per_side": str(SPOT_FEE_BPS_PER_SIDE),
        "perpetual_fee_bps_per_side": str(PERPETUAL_FEE_BPS_PER_SIDE),
        "fee_evidence_id": FEE_EVIDENCE_ID,
    }
    try:
        spec = BinanceCostJournalSpec.model_validate(document)
    except ValidationError as error:
        raise BinanceCostJournalSpecError(
            f"the derived journal spec is invalid: {error}"
        ) from error
    material: dict[str, object] = spec.model_dump(mode="json")
    (root / SEGMENT_DIRECTORY_NAME).mkdir(parents=True, exist_ok=True)
    _publish(spec_path, {**material, "spec_hash": content_sha256(material)})
    return spec_path


def load_journal_spec(journal_root: Path) -> tuple[BinanceCostJournalSpec, str]:
    """Read a journal's frozen spec and return it with its recorded hash."""
    document = _read_object(journal_root / JOURNAL_SPEC_NAME, label="the journal spec")
    material = {key: value for key, value in document.items() if key != "spec_hash"}
    spec_hash = content_sha256(material)
    if document.get("spec_hash") != spec_hash:
        raise BinanceCostJournalSpecError("the journal spec does not match its recorded hash")
    try:
        spec = BinanceCostJournalSpec.model_validate(material)
    except ValidationError as error:
        raise BinanceCostJournalSpecError(f"the journal spec is invalid: {error}") from error
    return spec, spec_hash


def run_journal(
    *,
    workspace_root: Path,
    journal_root: Path,
    reserve_bytes: int,
    rounds: int,
    fetcher: Fetcher,
    clock: Callable[[], int] = time.time_ns,
    sleep: Callable[[float], None] = time.sleep,
) -> ChainHead:
    """Append up to ``rounds`` sampled segments to a verified journal.

    The whole chain is verified before anything is appended and the chain head
    must name this journal's spec, so a tampered chain or a foreign spec stops
    the run instead of extending it. The one verification failure that is
    repaired rather than refused is a segment the previous process published
    before it was killed, whose chain head never followed: that segment is
    durable and self-verifying, so the head is rebuilt over it (see
    ``_orphan_segment``) and the run continues from the next sequence.

    One instrument's failure - a fetcher exception or an unusable payload -
    becomes that observation's reason and the round is still written; a round
    in which every instrument fails aborts before writing, which is what the
    supervisor restarts. Sampling stops early once the declared
    ``target_rounds`` is on disk.

    The declared interval is waited between rounds *and* before the first round
    of a journal that already holds segments, so a restart cannot sample faster
    than the cadence the spec declares.
    """
    if rounds <= 0:
        raise BinanceCostJournalSpecError("a run appends at least one round")
    root = _authorize(workspace_root, journal_root, reserve_bytes)
    spec, spec_hash = load_journal_spec(root)
    head = _read_chain_head(root)
    if head is not None and head.spec_hash != spec_hash:
        raise BinanceCostJournalSpecError("this journal is bound to a different spec")
    resumed = head is not None
    valid, reasons = verify_journal(root)
    if not valid:
        orphan = _orphan_segment(root, head, spec_hash=spec_hash)
        if orphan is None:
            raise BinanceCostJournalSpecError("journal verification failed: " + ",".join(reasons))
        head = _publish_chain_head(root, segment=orphan, spec_hash=spec_hash)
        resumed = True
        valid, reasons = verify_journal(root)
        if not valid:
            raise BinanceCostJournalSpecError(
                "journal verification failed after adopting the trailing segment: "
                + ",".join(reasons)
            )
    written = 0
    while written < rounds and (head is None or head.segment_count < spec.target_rounds):
        if written or resumed:
            sleep(spec.sample_interval_seconds)
        segment = _sample_round(
            spec=spec, spec_hash=spec_hash, head=head, fetcher=fetcher, clock=clock
        )
        head = _publish_segment(root, segment=segment, spec_hash=spec_hash)
        written += 1
    if head is None:
        raise BinanceCostJournalError("the journal is empty and no round was sampled")
    return head


def verify_journal(journal_root: Path) -> tuple[bool, tuple[str, ...]]:
    """Recompute every segment's hash, every chain link and the chain head.

    Never raises: an unreadable or malformed journal comes back as a reason
    code, the way ``verify_panel_capture`` reports a broken capture. An empty
    journal - a spec and no segment yet - is valid.
    """
    try:
        _, spec_hash = load_journal_spec(journal_root)
    except BinanceCostJournalError:
        return False, ("JOURNAL_SPEC_UNVERIFIED",)
    directory = journal_root / SEGMENT_DIRECTORY_NAME
    if not directory.is_dir():
        return False, ("SEGMENT_DIRECTORY_MISSING",)
    paths = sorted(path for path in directory.glob("*.json") if _SEGMENT_NAME.match(path.name))
    # A `<sequence>.json.<pid>.tmp` left by a write that died mid-flight carries
    # nothing and is replaced by the next attempt at that sequence; anything
    # else under `segments/` is not this journal's and is reported.
    reasons: list[str] = [
        f"SEGMENT_UNEXPECTED_FILE:{path.name}"
        for path in sorted(directory.iterdir())
        if not _SEGMENT_NAME.match(path.name) and not _SEGMENT_TEMPORARY_NAME.match(path.name)
    ]
    previous_hash = ZERO_HASH
    for expected_sequence, path in enumerate(paths):
        try:
            document = _read_object(path, label="a journal segment")
            segment = JournalSegment.model_validate(document)
        except (BinanceCostJournalError, ValidationError):
            reasons.append(f"SEGMENT_UNREADABLE:{path.name}")
            return False, tuple(reasons)
        material = {key: value for key, value in document.items() if key != "content_hash"}
        if segment.content_hash != content_sha256(material):
            reasons.append(f"SEGMENT_HASH_MISMATCH:{path.name}")
        if segment.sequence != expected_sequence:
            reasons.append(f"SEGMENT_SEQUENCE_MISMATCH:{path.name}")
        if segment.spec_hash != spec_hash:
            reasons.append(f"SEGMENT_SPEC_MISMATCH:{path.name}")
        if segment.previous_segment_hash != previous_hash:
            reasons.append(f"SEGMENT_LINK_MISMATCH:{path.name}")
        previous_hash = segment.content_hash
    reasons.extend(
        _chain_head_reasons(
            journal_root,
            spec_hash=spec_hash,
            segment_count=len(paths),
            final_segment_hash=previous_hash,
        )
    )
    return not reasons, tuple(reasons)


def public_binance_json_fetcher(url: str) -> Mapping[str, object]:
    """GET one public Binance JSON object over HTTPS with a 20 s timeout.

    The host is checked against ``ALLOWED_HOSTS`` before any socket is opened,
    so a URL this journal did not build never reaches the network, and again on
    the URL the response came from, so a redirect cannot walk off the list. A non-200
    response, a body that is not a JSON object and any transport failure are
    refused as ``BinanceCostJournalError``, which the run loop records as that
    instrument's reason for the round.
    """
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.netloc not in ALLOWED_HOSTS:
        raise BinanceCostJournalError("journal requests are limited to Binance's public hosts")
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "hybrid-trading-research/0.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=_FETCH_TIMEOUT_SECONDS) as response:
            if response.status != 200:
                raise BinanceCostJournalError(f"public request returned {response.status}")
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
            final_url = str(response.url)
    except (OSError, http.client.HTTPException) as error:
        raise BinanceCostJournalError(f"public request failed: {error}") from error
    if urlsplit(final_url).netloc not in ALLOWED_HOSTS:
        raise BinanceCostJournalError("a redirect left Binance's public hosts")
    if len(raw) > _MAX_RESPONSE_BYTES:
        raise BinanceCostJournalError("public response exceeds the byte limit")
    try:
        document: object = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BinanceCostJournalError("public response is not valid JSON") from error
    if not isinstance(document, dict):
        raise BinanceCostJournalError("public response is not a JSON object")
    return document


def _capture_histories(capture_root: Path, *, label: str) -> dict[str, ContractHistory]:
    bars = load_panel_bars(capture_root / "dataset")
    if not bars:
        raise BinanceCostJournalError(f"the {label} capture holds no panel bars")
    return build_contract_histories(bars)


def _last_close_ns(histories: Mapping[str, ContractHistory]) -> int:
    return max(history.close_times[-1] for history in histories.values())


def _capture_root_hash(capture_root: Path, *, market: str, label: str) -> str:
    valid, errors = verify_panel_capture(capture_root)
    if not valid:
        raise BinanceCostJournalError(
            f"{label} capture verification failed: " + ",".join(errors)
        )
    manifest = _read_object(
        capture_root / "capture-manifest.json", label=f"the {label} capture manifest"
    )
    # P1.27's perpetual capture predates the `market` key, so its absence means "um".
    if str(manifest.get("market", "um")) != market:
        raise BinanceCostJournalSpecError(f"the {label} capture is not a {market} capture")
    digest = manifest.get("capture_root_hash")
    if not isinstance(digest, str):
        raise BinanceCostJournalError(f"the {label} capture manifest carries no root hash")
    return digest


def _depth_url(*, market: str, symbol: str) -> str:
    if market == "spot":
        return f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit={DEPTH_LIMIT}"
    return f"https://fapi.binance.com/fapi/v1/depth?symbol={symbol}&limit={DEPTH_LIMIT}"


def _premium_index_url(symbol: str) -> str:
    return f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}"


def _spot_leg(symbol: str, *, pair_symbol: str, tier: Literal[1, 2]) -> JournalInstrument:
    return JournalInstrument(
        instrument_id=f"spot:{symbol}",
        market="spot",
        symbol=symbol,
        pair_symbol=pair_symbol,
        tier=tier,
        depth_url=_depth_url(market="spot", symbol=symbol),
        premium_index_url=None,
    )


def _perpetual_leg(symbol: str, *, tier: Literal[1, 2]) -> JournalInstrument:
    return JournalInstrument(
        instrument_id=f"perp:{symbol}",
        market="um",
        symbol=symbol,
        pair_symbol=symbol,
        tier=tier,
        depth_url=_depth_url(market="um", symbol=symbol),
        premium_index_url=_premium_index_url(symbol),
    )


def _sample_round(
    *,
    spec: BinanceCostJournalSpec,
    spec_hash: str,
    head: ChainHead | None,
    fetcher: Fetcher,
    clock: Callable[[], int],
) -> JournalSegment:
    received_time_ns = clock()
    observations = tuple(
        _observe(instrument, notionals=spec.notionals, fetcher=fetcher, clock=clock)
        for instrument in spec.instruments
    )
    if all(not observation.ok for observation in observations):
        raise BinanceCostJournalError("every instrument failed in this round")
    material: dict[str, object] = {
        "version": JOURNAL_VERSION,
        "sequence": 0 if head is None else head.last_sequence + 1,
        "spec_hash": spec_hash,
        "previous_segment_hash": ZERO_HASH if head is None else head.final_segment_hash,
        "received_time_ns": received_time_ns,
        "observations": [observation.model_dump(mode="json") for observation in observations],
    }
    return JournalSegment.model_validate({**material, "content_hash": content_sha256(material)})


def _observe(
    instrument: JournalInstrument,
    *,
    notionals: Sequence[Decimal],
    fetcher: Fetcher,
    clock: Callable[[], int],
) -> InstrumentObservation:
    """Sample one instrument, turning every failure into its own record.

    The depth request decides whether there is an observation at all. A
    perpetual's premium index is a second request beside it: when that one
    fails the book was still measured, so the transport reason is carried in
    ``premium_index_reason`` and the observation stays ``ok``.
    """
    keys = _notional_keys(notionals)
    try:
        payload: object = fetcher(instrument.depth_url)
    except Exception as error:  # one instrument's failure is an observation, not an abort
        return _failed_observation(
            instrument_id=instrument.instrument_id,
            received_time_ns=clock(),
            keys=keys,
            reason=_failure_reason(error),
        )
    premium_index: object = None
    premium_failure: str | None = None
    if instrument.premium_index_url is not None:
        try:
            premium_index = fetcher(instrument.premium_index_url)
        except Exception as error:
            premium_failure = _failure_reason(error)
    received_time_ns = clock()
    try:
        observation = depth_observation(
            payload,
            instrument=instrument,
            received_time_ns=received_time_ns,
            notionals=notionals,
            premium_index=premium_index,
        )
    except Exception as error:  # depth_observation's contract, held to even if it breaks
        return _failed_observation(
            instrument_id=instrument.instrument_id,
            received_time_ns=received_time_ns,
            keys=keys,
            reason=_failure_reason(error),
        )
    if premium_failure is None or not observation.ok:
        return observation
    # The premium index never arrived, so `depth_observation` recorded "missing
    # premium index"; the transport's own reason says more.
    return InstrumentObservation.model_validate(
        {**observation.model_dump(mode="json"), "premium_index_reason": premium_failure}
    )


def _failure_reason(error: Exception) -> str:
    """Name the failure without letting a hostile message bloat the segment."""
    return f"{type(error).__name__}: {error}"[:_MAX_REASON_CHARACTERS]


def _publish_segment(root: Path, *, segment: JournalSegment, spec_hash: str) -> ChainHead:
    document: dict[str, object] = segment.model_dump(mode="json")
    _publish(root / SEGMENT_DIRECTORY_NAME / f"{segment.sequence:010d}.json", document)
    return _publish_chain_head(root, segment=segment, spec_hash=spec_hash)


def _publish_chain_head(root: Path, *, segment: JournalSegment, spec_hash: str) -> ChainHead:
    head = _build_chain_head(
        spec_hash=spec_hash,
        segment_count=segment.sequence + 1,
        last_sequence=segment.sequence,
        final_segment_hash=segment.content_hash,
    )
    head_document: dict[str, object] = head.model_dump(mode="json")
    _publish(root / CHAIN_HEAD_NAME, head_document)
    return head


def _orphan_segment(
    journal_root: Path, head: ChainHead | None, *, spec_hash: str
) -> JournalSegment | None:
    """The segment a killed process left beyond the chain head, or ``None``.

    A process killed between a segment's ``replace()`` and its chain head's -
    or one whose head publish failed outright - leaves segment n on disk with a
    head naming n-1, and no head at all when n is 0. The segment is durable and
    carries its own hash, so the head can be rebuilt over it instead of the
    journal dying on the next start. Adoption is deliberately narrow: exactly
    one segment file more than the head counts, and that file must be the next
    sequence, must validate, must recompute to its recorded ``content_hash``,
    must name this journal's spec and must link to the head it follows. Every
    other verification failure is corruption and stays refused.
    """
    directory = journal_root / SEGMENT_DIRECTORY_NAME
    if not directory.is_dir():
        return None
    paths = sorted(path for path in directory.glob("*.json") if _SEGMENT_NAME.match(path.name))
    counted = 0 if head is None else head.segment_count
    if len(paths) != counted + 1:
        return None
    expected_sequence = 0 if head is None else head.last_sequence + 1
    path = paths[-1]
    if path.name != f"{expected_sequence:010d}.json":
        return None
    try:
        document = _read_object(path, label="a journal segment")
        segment = JournalSegment.model_validate(document)
    except (BinanceCostJournalError, ValidationError):
        return None
    material = {key: value for key, value in document.items() if key != "content_hash"}
    if segment.content_hash != content_sha256(material):
        return None
    if segment.sequence != expected_sequence or segment.spec_hash != spec_hash:
        return None
    if segment.previous_segment_hash != (ZERO_HASH if head is None else head.final_segment_hash):
        return None
    return segment


def _build_chain_head(
    *, spec_hash: str, segment_count: int, last_sequence: int, final_segment_hash: str
) -> ChainHead:
    material: dict[str, object] = {
        "version": JOURNAL_VERSION,
        "spec_hash": spec_hash,
        "segment_count": segment_count,
        "last_sequence": last_sequence,
        "final_segment_hash": final_segment_hash,
    }
    return ChainHead.model_validate({**material, "content_hash": content_sha256(material)})


def _read_chain_head(journal_root: Path) -> ChainHead | None:
    path = journal_root / CHAIN_HEAD_NAME
    if not path.exists():
        return None
    head = _validated_chain_head(_read_object(path, label="the chain head"))
    if head is None:
        raise BinanceCostJournalSpecError("the chain head does not match its recorded hash")
    return head


def _validated_chain_head(document: Mapping[str, object]) -> ChainHead | None:
    try:
        head = ChainHead.model_validate(document)
    except ValidationError:
        return None
    material = {key: value for key, value in document.items() if key != "content_hash"}
    if head.content_hash != content_sha256(material):
        return None
    return head


def _chain_head_reasons(
    journal_root: Path, *, spec_hash: str, segment_count: int, final_segment_hash: str
) -> list[str]:
    path = journal_root / CHAIN_HEAD_NAME
    if not path.exists():
        return [] if segment_count == 0 else ["CHAIN_HEAD_MISSING"]
    if segment_count == 0:
        return ["CHAIN_HEAD_UNEXPECTED"]
    try:
        document = _read_object(path, label="the chain head")
    except BinanceCostJournalError:
        return ["CHAIN_HEAD_UNREADABLE"]
    head = _validated_chain_head(document)
    if head is None:
        return ["CHAIN_HEAD_UNREADABLE"]
    reasons: list[str] = []
    if head.spec_hash != spec_hash:
        reasons.append("CHAIN_HEAD_SPEC_MISMATCH")
    if head.segment_count != segment_count:
        reasons.append("CHAIN_HEAD_COUNT_MISMATCH")
    if head.last_sequence != segment_count - 1:
        reasons.append("CHAIN_HEAD_SEQUENCE_MISMATCH")
    if head.final_segment_hash != final_segment_hash:
        reasons.append("CHAIN_HEAD_LINK_MISMATCH")
    return reasons


def _authorize(workspace_root: Path, journal_root: Path, reserve_bytes: int) -> Path:
    workspace = workspace_root.resolve()
    root = journal_root.resolve()
    StoragePolicy(workspace, reserve_bytes).authorize(
        target=root,
        temporary_directory=root.parent,
        free_bytes=shutil.disk_usage(workspace).free,
        worst_case_required_bytes=_WORST_CASE_REQUIRED_BYTES,
    )
    return root


def _read_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BinanceCostJournalSpecError(f"{label} is unreadable: {error}") from error
    if not isinstance(document, dict):
        raise BinanceCostJournalSpecError(f"{label} must be a JSON object")
    return document


def _publish(path: Path, document: dict[str, object]) -> None:
    """Write one immutable JSON artifact through a temporary file.

    The temporary carries the writing process's pid, so a second process - one
    the supervisor should never have started - cannot half-write the artifact
    this one is publishing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(canonical_json(document))
    temporary.replace(path)
