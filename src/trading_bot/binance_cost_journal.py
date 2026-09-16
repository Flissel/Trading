"""The Binance public cost journal: models, book arithmetic and the sampler.

The journal measures what a taker pays to cross a spot leg and a USD-M perpetual
leg on Binance's public order books at three notionals. This module holds the
frozen record shapes, the two pure functions the sampler needs - the walk of a
displayed book to a quote notional and the parse of one depth payload into an
observation - and the journal itself: the sample derived from the two carry
captures, the hash-chained run loop, its verification, the one fetcher that is
allowed to reach Binance and the finalisation receipt a carry declaration may
cite once the journal has run its seven days.
"""

import http.client
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from decimal import ROUND_CEILING, ROUND_HALF_EVEN, Decimal, InvalidOperation
from email.message import Message
from pathlib import Path
from typing import Literal, Self
from urllib.parse import urlsplit

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.carry_config import CarryUniverseRules, load_carry_family_spec
from trading_bot.carry_signals import WEEK_NS
from trading_bot.carry_universe import select_pair_universe
from trading_bot.cost_evidence_rule import (
    CAPITAL_DECLARATION_RULE as CAPITAL_DECLARATION_RULE,  # re-export
)
from trading_bot.cost_evidence_rule import (
    DECLARATION_RULE as DECLARATION_RULE,  # re-export
)
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
RECEIPT_VERSION = "binance-cost-journal-receipt/1.0.0"
NOTIONALS: tuple[Decimal, ...] = (Decimal("500"), Decimal("5000"), Decimal("50000"))
DEPTH_LIMIT = 500
SAMPLE_INTERVAL_SECONDS = 61
# 11 000 rounds at 61 s is 7.8 days: the eligibility floor below wants 10 000
# non-null observations over at least 7 days, and the margin pays for the rounds
# a venue outage or a restart costs.
TARGET_ROUNDS = 11_000
MINIMUM_OBSERVATIONS = 10_000
MINIMUM_SPAN_NS = 7 * 86_400_000_000_000
# Spec 4: both floors are read at the 5 000 USDT notional, over the observations
# whose displayed book could fill it.
ELIGIBILITY_NOTIONAL = "5000"
# Spec 5, verbatim and unwrapped, lives in `cost_evidence_rule` so the receipt
# that publishes it and the declaration that cites it bind one string. A receipt
# carrying a different sentence is refused by the model below.
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
# Both legs of every declared rank: the sample the spec describes, whole.
SAMPLE_INSTRUMENT_COUNT = 2 * (len(SAMPLE_TIER_ONE_RANKS) + len(SAMPLE_TIER_TWO_RANKS))
# Spec 5's tier medians are medians: below four contributing members at a
# notional the number would be one or two instruments wearing a tier's name, so
# the receipt reports the counts and no median (fix wave W9).
TIER_MINIMUM_CONTRIBUTORS = 4
JOURNAL_SPEC_NAME = "journal-spec.json"
CHAIN_HEAD_NAME = "chain-head.json"
SEGMENT_DIRECTORY_NAME = "segments"
JOURNAL_LOCK_NAME = "run.lock"

# url -> parsed JSON object; raises on transport failure.
type Fetcher = Callable[[str], Mapping[str, object]]

_BPS = Decimal(10_000)
_BPS_QUANTUM = Decimal("0.000001")
_P50 = Decimal("0.5")
_P90 = Decimal("0.9")
_P99 = Decimal("0.99")
_QUANTILES: tuple[tuple[str, Decimal], ...] = (("p50", _P50), ("p90", _P90), ("p99", _P99))
_QUANTILE_NAMES: tuple[str, ...] = tuple(name for name, _ in _QUANTILES)
_INSTRUMENT_STATISTIC_KEYS: tuple[str, ...] = ("count", "insufficient_depth", *_QUANTILE_NAMES)
_TIER_MEDIAN_KEYS: tuple[str, ...] = ("p50_of_p50", "p50_of_p90")
_TIER_COUNT_KEYS: tuple[str, ...] = ("contributing_count", "absent_count")
_TIER_STATISTIC_KEYS: tuple[str, ...] = (*_TIER_MEDIAN_KEYS, *_TIER_COUNT_KEYS)
# The floors the receipt states so its reader never has to look them up.
_RECEIPT_ELIGIBILITY_KEYS: tuple[str, ...] = (
    "minimum_observations",
    "minimum_span_ns",
    "eligibility_notional",
    "tier_minimum_contributors",
)
# Spec 5 reports every declared tier and leg, whether or not the sample reached it.
_RECEIPT_TIERS: tuple[int, ...] = (1, 2)
_RECEIPT_MARKETS: tuple[str, ...] = ("spot", "um")
_OBSERVATION_FLOOR_CODE = "COST_OBSERVATION_FLOOR_NOT_MET"
_SPAN_FLOOR_CODE = "COST_CAPTURE_SPAN_FLOOR_NOT_MET"
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
# Binance answers a breached rate limit with 429 and a ban with 418.
_THROTTLE_STATUSES = frozenset({429, 418})
_DEFAULT_THROTTLE_SECONDS = 60
_MAX_THROTTLE_SECONDS = 300
_NANOSECONDS = Decimal(1_000_000_000)
_RATE_QUANTUM = Decimal("0.0001")
_SECOND_QUANTUM = Decimal("0.001")


class BinanceCostJournalError(RuntimeError):
    """Raised when a journal cannot be created, sampled or read safely."""


class BinanceCostJournalSpecError(BinanceCostJournalError):
    """Raised when the request does not match the journal on disk.

    The CLI maps this to exit code 2 and every other failure to 1, so the
    supervisor stops on a mismatched, missing or unverifiable journal instead
    of restarting a process that can only fail the same way again.
    """


class BinanceCostJournalTransportError(BinanceCostJournalError):
    """A public request the venue answered with a status code of its own.

    The status and the ``Retry-After`` the venue sent - when it sent one - are
    carried on the error rather than buried in its message, so the run loop can
    back off for as long as Binance asked instead of hammering a rate limit it
    has already breached (fix wave W7).
    """

    def __init__(self, status: int, retry_after: int | None = None) -> None:
        super().__init__(f"public request returned {status}")
        self.status = status
        self.retry_after = retry_after


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
    # How many levels a side the payload actually carried. A book that stops
    # short of the requested depth limit is a thin book, not a broken one, and
    # a reader of the journal cannot tell the two apart from the slippage
    # alone: `null` at a notional means the fetched depth could not fill it,
    # and these two say how much depth that was (fix wave W8).
    bid_levels: int | None = None
    ask_levels: int | None = None
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
                self.bid_levels,
                self.ask_levels,
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
        if self.bid_levels is None or self.ask_levels is None:
            raise ValueError("a measured observation counts the levels it walked")
        if self.bid_levels <= 0 or self.ask_levels <= 0:
            raise ValueError("a measured book displays at least one level a side")
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


class InstrumentStatistics(_Frozen):
    """What one sampled instrument's measured rounds came to, and whether they count.

    ``observation_count`` is how many rounds the instrument was measured in; a
    notional's ``count`` is how many of those filled it and ``insufficient_depth``
    how many did not, so the two always add up to the observation count.
    ``first_time_ns`` and ``last_time_ns`` are the stamps the span floor is read
    between - the earliest and the latest observation that filled the
    eligibility notional, taken as a minimum and a maximum so a clock that
    stepped backwards mid-run narrows the span instead of inverting it - and
    ``span_observation_count`` is how many observations that window holds, so
    the eligibility decision can be recomputed from the receipt alone (fix wave
    W10). An ineligible instrument still carries its statistics; only the tier
    medians pass it by (spec 4).
    """

    instrument_id: str
    tier: int
    market: str
    eligible: bool
    reason_codes: tuple[str, ...]
    observation_count: int
    span_observation_count: int
    first_time_ns: int | None
    last_time_ns: int | None
    spread_bps_p50: Decimal | None
    slippage: dict[str, dict[str, Decimal | int | None]]
    basis_bps_p50: Decimal | None

    @field_validator("slippage")
    @classmethod
    def validate_slippage(
        cls, value: dict[str, dict[str, Decimal | int | None]]
    ) -> dict[str, dict[str, Decimal | int | None]]:
        if not value:
            raise ValueError("statistics carry one entry per sampled notional")
        for entry in value.values():
            if set(entry) != set(_INSTRUMENT_STATISTIC_KEYS):
                raise ValueError(f"a notional's statistics are {_INSTRUMENT_STATISTIC_KEYS}")
            for name in ("count", "insufficient_depth"):
                count = entry[name]
                if not isinstance(count, int) or count < 0:
                    raise ValueError("counts are non-negative integers")
            if any(not isinstance(entry[name], Decimal | None) for name in _QUANTILE_NAMES):
                raise ValueError("a quantile is a decimal or absent")
        return value

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        if not self.instrument_id:
            raise ValueError("instrument_id must be set")
        if self.tier not in _RECEIPT_TIERS or self.market not in _RECEIPT_MARKETS:
            raise ValueError("statistics carry a declared tier and market")
        if self.eligible != (not self.reason_codes):
            raise ValueError("an instrument is eligible exactly when no floor was missed")
        if min(self.observation_count, self.span_observation_count) < 0:
            raise ValueError("the observation count cannot be negative")
        if self.span_observation_count > self.observation_count:
            raise ValueError("the eligibility window cannot hold more than was observed")
        if (self.first_time_ns is None) != (self.last_time_ns is None):
            raise ValueError("the eligibility window is either absent or complete")
        if (self.span_observation_count == 0) != (self.first_time_ns is None):
            raise ValueError("an eligibility window holds the observations it spans")
        if self.first_time_ns is not None and self.last_time_ns is not None:
            if self.first_time_ns <= 0:
                raise ValueError("the eligibility window opens at a positive stamp")
            if self.last_time_ns < self.first_time_ns:
                raise ValueError("the eligibility window cannot close before it opens")
        return self


class TierStatistics(_Frozen):
    """The medians one tier's eligible instruments reached on one market.

    ``p50_of_p50`` and ``p50_of_p90`` are the medians of the members' p50s and
    p90s at that notional, taken under the same quantile rule. A tier and market
    with no eligible member is emitted all the same, with no members and no
    numbers: the receipt says that it measured nothing there rather than leaving
    the reader to infer it from an absence.

    Eligibility is decided at 5 000 USDT, so a member can be eligible and still
    never fill 50 000: ``contributing_count`` is how many members carried a
    number at *this* notional and ``absent_count`` how many did not. Below
    ``TIER_MINIMUM_CONTRIBUTORS`` contributors the two medians are withheld -
    the median of three instruments is not a tier - and the counts say why
    (fix wave W9).
    """

    tier: int
    market: str
    instrument_count: int
    slippage: dict[str, dict[str, Decimal | int | None]]

    @field_validator("slippage")
    @classmethod
    def validate_slippage(
        cls, value: dict[str, dict[str, Decimal | int | None]]
    ) -> dict[str, dict[str, Decimal | int | None]]:
        if not value:
            raise ValueError("tier statistics carry one entry per sampled notional")
        for entry in value.values():
            if set(entry) != set(_TIER_STATISTIC_KEYS):
                raise ValueError(f"a notional's tier statistics are {_TIER_STATISTIC_KEYS}")
            for name in _TIER_COUNT_KEYS:
                count = entry[name]
                if not isinstance(count, int) or count < 0:
                    raise ValueError("contributor counts are non-negative integers")
            if any(not isinstance(entry[name], Decimal | None) for name in _TIER_MEDIAN_KEYS):
                raise ValueError("a tier median is a decimal or absent")
        return value

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        if self.tier not in _RECEIPT_TIERS or self.market not in _RECEIPT_MARKETS:
            raise ValueError("tier statistics carry a declared tier and market")
        if self.instrument_count < 0:
            raise ValueError("the instrument count cannot be negative")
        for entry in self.slippage.values():
            contributing = entry["contributing_count"]
            absent = entry["absent_count"]
            if not isinstance(contributing, int) or not isinstance(absent, int):
                raise ValueError("contributor counts are non-negative integers")
            if contributing + absent != self.instrument_count:
                raise ValueError("every member either contributes at a notional or is absent")
            if contributing < TIER_MINIMUM_CONTRIBUTORS and any(
                entry[name] is not None for name in _TIER_MEDIAN_KEYS
            ):
                raise ValueError(
                    f"a tier median needs {TIER_MINIMUM_CONTRIBUTORS} contributing members"
                )
        return self


class FinalizationReceipt(_Frozen):
    """The immutable, self-hashed reading of a finished journal.

    It binds the journal it was taken from (the spec hash, the chain head's own
    hash and the number of segments), repeats the declared fees the journal
    never measured, carries the sample's statistics and the tier medians, and
    states in ``declaration_rule`` and ``capital_declaration_rule`` the two uses
    a carry declaration may make of it - spec 5's fixed reading and spec 5.1's
    reading for a declaration with a ``capital`` block. Anything else on top of
    these numbers is not a reading of this receipt.

    ``segment_count`` is exactly the number of segments the statistics cover -
    the ones the chain head bound - and ``segments_beyond_head`` how many
    numbered segments sat past that head when the reading was taken, which a
    run appending concurrently leaves behind. A non-zero value is not a fault:
    it says this receipt reads part of a journal that has since grown.

    ``eligibility``, ``target_rounds``, ``notionals`` and
    ``sample_interval_seconds`` carry the thresholds and the cadence the
    numbers were taken under, so a reader can recompute every decision from
    the receipt without holding the build that produced it (fix wave W10).
    """

    version: Literal["binance-cost-journal-receipt/1.0.0"]
    spec_hash: str
    chain_head_hash: str
    segment_count: int
    segments_beyond_head: int
    spot_fee_bps_per_side: Decimal
    perpetual_fee_bps_per_side: Decimal
    fee_evidence_id: str
    eligibility: dict[str, int | str]
    target_rounds: int
    notionals: tuple[Decimal, ...]
    sample_interval_seconds: int
    instruments: tuple[InstrumentStatistics, ...]
    tiers: tuple[TierStatistics, ...]
    declaration_rule: str
    capital_declaration_rule: str
    content_hash: str

    @field_validator("spec_hash", "chain_head_hash", "content_hash")
    @classmethod
    def validate_hashes(cls, value: str) -> str:
        return _validated_hex(value, field_name="receipt hash")

    @field_validator("eligibility")
    @classmethod
    def validate_eligibility(cls, value: dict[str, int | str]) -> dict[str, int | str]:
        if set(value) != set(_RECEIPT_ELIGIBILITY_KEYS):
            raise ValueError(f"a receipt states the floors {_RECEIPT_ELIGIBILITY_KEYS}")
        if not isinstance(value["eligibility_notional"], str):
            raise ValueError("the eligibility notional is the key the statistics carry")
        for name in ("minimum_observations", "minimum_span_ns", "tier_minimum_contributors"):
            floor = value[name]
            if not isinstance(floor, int) or floor <= 0:
                raise ValueError("the declared floors are positive integers")
        return value

    @field_validator("notionals")
    @classmethod
    def validate_notionals(cls, value: tuple[Decimal, ...]) -> tuple[Decimal, ...]:
        if not value or any(item <= 0 for item in value):
            raise ValueError("a receipt reads at least one positive notional")
        return value

    @field_validator("fee_evidence_id")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        if not _IDENTIFIER.match(value):
            raise ValueError("identifiers are non-empty and free of whitespace")
        return value

    @field_validator("declaration_rule")
    @classmethod
    def validate_declaration_rule(cls, value: str) -> str:
        if value != DECLARATION_RULE:
            raise ValueError("a receipt carries spec section 5's declaration rule verbatim")
        return value

    @field_validator("capital_declaration_rule")
    @classmethod
    def validate_capital_declaration_rule(cls, value: str) -> str:
        if value != CAPITAL_DECLARATION_RULE:
            raise ValueError("a receipt carries spec section 5.1's declaration rule verbatim")
        return value

    @model_validator(mode="after")
    def validate_content(self) -> Self:
        if self.segment_count <= 0:
            raise ValueError("a receipt summarises at least one segment")
        if self.segments_beyond_head < 0:
            raise ValueError("the count of segments past the head cannot be negative")
        if min(self.target_rounds, self.sample_interval_seconds) <= 0:
            raise ValueError("the declared cadence and target are positive")
        if str(self.eligibility["eligibility_notional"]) not in {
            str(notional) for notional in self.notionals
        }:
            raise ValueError("the eligibility notional is one of the sampled notionals")
        if not self.instruments or not self.tiers:
            raise ValueError("a receipt lists the sample and every declared tier")
        identifiers = [item.instrument_id for item in self.instruments]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("an instrument is summarised at most once")
        if min(self.spot_fee_bps_per_side, self.perpetual_fee_bps_per_side) < 0:
            raise ValueError("declared fees cannot be negative")
        return self


class InstrumentStatus(_Frozen):
    """How often one instrument reported over a status window.

    A rate is ``None`` where the window holds no observation of the instrument
    at all - a rate over nothing is not zero.
    """

    instrument_id: str
    observation_count: int
    ok_rate: Decimal | None
    premium_index_reason_rate: Decimal | None

    @model_validator(mode="after")
    def validate_rates(self) -> Self:
        if not self.instrument_id:
            raise ValueError("instrument_id must be set")
        if self.observation_count < 0:
            raise ValueError("the observation count cannot be negative")
        for rate in (self.ok_rate, self.premium_index_reason_rate):
            if rate is not None and not (0 <= rate <= 1):
                raise ValueError("a rate lies between zero and one")
        if (self.observation_count == 0) != (self.ok_rate is None):
            raise ValueError("a rate is reported exactly when the window holds observations")
        return self


class JournalStatus(_Frozen):
    """What a running journal looks like at its tip, over its last N segments.

    The counts and the last sequence come from the chain head; the cadence, the
    per-instrument rates and the verification are read over the window only, so
    the answer costs the same on day one and on day eight (fix wave W4).
    ``verified`` is the window's verdict, not the whole chain's:
    ``verify_journal`` remains the thing that reads every segment.
    """

    segment_count: int
    last_sequence: int
    window_segments: int
    last_received_time_ns: int | None
    mean_period_seconds: Decimal | None
    instruments: tuple[InstrumentStatus, ...]
    verified: bool
    reasons: tuple[str, ...]

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        if min(self.segment_count, self.window_segments) < 0 or self.last_sequence < 0:
            raise ValueError("segment counts and sequences cannot be negative")
        if self.window_segments > self.segment_count:
            raise ValueError("a status window cannot hold more than the chain does")
        if self.last_received_time_ns is not None and self.last_received_time_ns <= 0:
            raise ValueError("received_time_ns must be a positive nanosecond stamp")
        if self.verified != (not self.reasons):
            raise ValueError("a window verifies exactly when nothing was wrong with it")
        return self


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
    ``bid_levels`` and ``ask_levels`` count the levels the payload actually
    carried, which is the depth every other number was read off.
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
        bid_levels=len(bids),
        ask_levels=len(asks),
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
        bid_levels=None,
        ask_levels=None,
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
    allow_short_sample: bool = False,
) -> Path:
    """Write ``<journal_root>/journal-spec.json`` and an empty ``segments/``.

    The spec is immutable and binds the journal directory to it, so a root that
    already holds one is refused: a new sample is a new journal (spec 3).
    Returns the path of the spec.

    A rank the universe does not reach is simply absent from the sample, which
    is right for the derivation and wrong for a journal meant to run for eight
    days: a captured universe thin enough to drop ranks would spend those days
    measuring a sample nobody declared. Unless ``allow_short_sample`` says
    otherwise, a derivation short of ``SAMPLE_INSTRUMENT_COUNT`` legs is
    refused before anything is written (fix wave W3).
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
    if len(instruments) != SAMPLE_INSTRUMENT_COUNT and not allow_short_sample:
        raise BinanceCostJournalSpecError(
            f"the captures derive {len(instruments)} of the {SAMPLE_INSTRUMENT_COUNT} declared "
            "legs; pass allow_short_sample to journal a thinner universe deliberately"
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

    The journal is written by one process at a time: an exclusive OS lock on
    ``run.lock`` is taken before anything is read, so a second sampler - a
    supervisor that started twice over a reboot - refuses instead of racing the
    first one to the same sequence (fix wave W2).

    The whole chain is verified before anything is appended and the chain head
    must name this journal's spec, so a tampered chain or a foreign spec stops
    the run instead of extending it. Three verification failures are repaired
    rather than refused, all of them the marks of a killed process rather than
    of corruption:

    - a segment the previous process published before it was killed, whose
      chain head never followed: it is durable and self-verifying, so the head
      is rebuilt over it (see ``_orphan_segment``) and the run continues from
      the next sequence. The rebuilt head is written only once the whole chain
      has verified against it, so a journal that is also broken somewhere else
      is left untouched;
    - a half-written trailing file, which is the crash itself caught in the
      act: it is deleted and its sequence sampled again (W1);
    - a chain head that was never written or torn in half, with segments on
      disk: the segments carry everything the head does, so it is rebuilt from
      them - and only if all of them verify from ``ZERO_HASH`` (W1).

    One instrument's failure - a fetcher exception or an unusable payload -
    becomes that observation's reason and the round is still written; a round
    in which every instrument fails aborts before writing, which is what the
    supervisor restarts. Sampling stops early once the declared
    ``target_rounds`` is on disk.

    The declared interval is a *period*, not a gap: after a round the loop
    waits what is left of it, so a slow round shortens the wait instead of
    sliding the cadence (W6). A resumed run waits a whole interval before its
    first round, so a restart cannot sample faster than the spec declares, and
    a venue that answered 429 or 418 buys the wait it asked for on top (W7).
    """
    if rounds <= 0:
        raise BinanceCostJournalSpecError("a run appends at least one round")
    root = _authorize(workspace_root, journal_root, reserve_bytes)
    spec, spec_hash = load_journal_spec(root)
    lock = _lock_journal(root)
    try:
        return _run_locked_journal(
            root, spec=spec, spec_hash=spec_hash, rounds=rounds,
            fetcher=fetcher, clock=clock, sleep=sleep,
        )
    finally:
        _release_journal_lock(lock)


def _run_locked_journal(
    root: Path,
    *,
    spec: BinanceCostJournalSpec,
    spec_hash: str,
    rounds: int,
    fetcher: Fetcher,
    clock: Callable[[], int],
    sleep: Callable[[float], None],
) -> ChainHead:
    """``run_journal``'s body, under the journal's write lock."""
    head = _resumable_chain_head(root)
    if head is not None and head.spec_hash != spec_hash:
        raise BinanceCostJournalSpecError("this journal is bound to a different spec")
    _discard_torn_trailing_segment(root, head)
    if head is None and _numbered_segments(root):
        # The rebuild walks and verifies the whole chain against the head it
        # publishes, so nothing below has to walk it a second time.
        head = _rebuilt_chain_head(root, spec_hash=spec_hash)
    else:
        head = _verified_or_adopted(root, head, spec_hash=spec_hash)
    resumed = head is not None
    written = 0
    throttle_seconds = 0
    # `None` before the first round of a run that starts a journal: that one
    # round alone waits for nothing.
    cadence: Decimal | None = Decimal(spec.sample_interval_seconds) if resumed else None
    while written < rounds and (head is None or head.segment_count < spec.target_rounds):
        if cadence is not None:
            sleep(float(cadence))
            if throttle_seconds:
                sleep(float(throttle_seconds))
        segment, throttle_seconds = _sample_round(
            spec=spec, spec_hash=spec_hash, head=head, fetcher=fetcher, clock=clock
        )
        head = _publish_segment(root, segment=segment, spec_hash=spec_hash)
        written += 1
        cadence = _remaining_interval(
            spec.sample_interval_seconds, started_ns=segment.received_time_ns, now_ns=clock()
        )
    if head is None:
        raise BinanceCostJournalError("the journal is empty and no round was sampled")
    return head


def _verified_or_adopted(
    root: Path, head: ChainHead | None, *, spec_hash: str
) -> ChainHead | None:
    """The head to append to once the chain has verified, adopting an orphan if it must.

    The adoption is judged before it is written: the whole chain must verify
    against the head the orphan would give it, or the journal is left exactly
    as it was found (fix round 2).
    """
    valid, reasons = verify_journal(root)
    if valid:
        return head
    orphan = _orphan_segment(root, head, spec_hash=spec_hash)
    if orphan is None:
        raise BinanceCostJournalSpecError("journal verification failed: " + ",".join(reasons))
    candidate = _chain_head_for(orphan, spec_hash=spec_hash)
    adopted, adoption_reasons = _verified_chain(root, candidate_head=candidate)
    if not adopted:
        raise BinanceCostJournalSpecError(
            "journal verification failed after adopting the trailing segment: "
            + ",".join(adoption_reasons)
        )
    return _publish_head(root, candidate)


def _remaining_interval(interval_seconds: int, *, started_ns: int, now_ns: int) -> Decimal:
    """What is left of the declared period after a round that has just ended.

    The spec declares the interval between the *starts* of consecutive rounds,
    so a round that took 40 s of a 61 s period leaves 21 s to wait and one that
    took 70 s leaves nothing (fix wave W6). A clock that stepped backwards
    under the round counts as no time at all rather than as a longer wait.
    """
    elapsed = Decimal(max(now_ns - started_ns, 0)) / _NANOSECONDS
    return max(Decimal(interval_seconds) - elapsed, Decimal(0))


def verify_journal(journal_root: Path) -> tuple[bool, tuple[str, ...]]:
    """Recompute every segment's hash, every chain link and the chain head.

    Never raises: an unreadable or malformed journal comes back as a reason
    code, the way ``verify_panel_capture`` reports a broken capture. An empty
    journal - a spec and no segment yet - is valid.
    """
    return _verified_chain(journal_root, candidate_head=None)


def _verified_chain(
    journal_root: Path, *, candidate_head: ChainHead | None, covered_segments: int | None = None
) -> tuple[bool, tuple[str, ...]]:
    """Verify the chain against ``candidate_head``, or against the head on disk.

    A candidate lets an adoption be judged before it is written: the whole
    chain is checked as it *would* stand with that head, so nothing is
    published over a journal that fails for another reason (fix round 2).

    ``covered_segments`` bounds the walk to the first N numbered segments, which
    is what a finalisation needs: a concurrent ``run_journal`` may be appending
    past the head, and those segments are neither verified nor judged against
    the head here - the receipt counts them and leaves them out (fix round 3).
    A sequence missing inside the bound is reported, not tolerated.
    """
    try:
        _, spec_hash = load_journal_spec(journal_root)
    except BinanceCostJournalError:
        return False, ("JOURNAL_SPEC_UNVERIFIED",)
    directory = journal_root / SEGMENT_DIRECTORY_NAME
    if not directory.is_dir():
        return False, ("SEGMENT_DIRECTORY_MISSING",)
    paths = _numbered_segments(journal_root)
    # A `<sequence>.json.<pid>.tmp` left by a write that died mid-flight carries
    # nothing and is replaced by the next attempt at that sequence; anything
    # else under `segments/` is not this journal's and is reported.
    reasons: list[str] = [
        f"SEGMENT_UNEXPECTED_FILE:{path.name}"
        for path in sorted(directory.iterdir())
        if not _SEGMENT_NAME.match(path.name) and not _SEGMENT_TEMPORARY_NAME.match(path.name)
    ]
    if covered_segments is not None:
        reasons.extend(
            f"SEGMENT_MISSING:{sequence:010d}.json"
            for sequence in range(len(paths), covered_segments)
        )
        paths = paths[:covered_segments]
    previous_hash = ZERO_HASH
    for expected_sequence, path in enumerate(paths):
        try:
            document = _read_object(path, label="a journal segment")
            segment = JournalSegment.model_validate(document)
        except (BinanceCostJournalError, ValidationError):
            reasons.append(f"SEGMENT_UNREADABLE:{path.name}")
            return False, tuple(reasons)
        reasons.extend(
            _segment_reasons(
                document,
                segment,
                name=path.name,
                expected_sequence=expected_sequence,
                spec_hash=spec_hash,
                previous_hash=previous_hash,
            )
        )
        previous_hash = segment.content_hash
    reasons.extend(
        _chain_head_reasons(
            journal_root,
            candidate_head=candidate_head,
            spec_hash=spec_hash,
            segment_count=len(paths),
            final_segment_hash=previous_hash,
        )
    )
    return not reasons, tuple(reasons)


def journal_status(journal_root: Path, *, last: int) -> JournalStatus:
    """Read a running journal's tip over its last ``last`` segments (fix wave W4).

    An eight-day run is watched, not waited on, and ``verify_journal`` reads
    every segment there is: by day six that is 8 000 files for one question.
    This reads the chain head, the window the head ends in, and nothing else -
    the cadence the rounds actually kept, how often each instrument reported,
    and whether that window hashes and links the way it should. The window is
    anchored on the recorded hash of the segment before it, so its first link
    is checked too; a chain broken before the window is a question for
    ``verify_journal``.

    Nothing here writes, and the caller needs no storage authorisation: reading
    a journal is not a job that can fill a disk.
    """
    if last <= 0:
        raise BinanceCostJournalSpecError("a status reads at least one segment")
    spec, spec_hash = load_journal_spec(journal_root)
    head = _read_chain_head(journal_root)
    if head is None:
        raise BinanceCostJournalSpecError("this journal holds no segment to report")
    directory = journal_root / SEGMENT_DIRECTORY_NAME
    start = max(head.segment_count - last, 0)
    anchor, reasons = _window_anchor(directory, start)
    segments: list[JournalSegment] = []
    for sequence in range(start, head.segment_count):
        name = f"{sequence:010d}.json"
        try:
            document = _read_object(directory / name, label="a journal segment")
            segment = JournalSegment.model_validate(document)
        except (BinanceCostJournalError, ValidationError):
            reasons.append(f"SEGMENT_UNREADABLE:{name}")
            break
        reasons.extend(
            _segment_reasons(
                document,
                segment,
                name=name,
                expected_sequence=sequence,
                spec_hash=spec_hash,
                # An unreadable anchor is already a reason of its own; linking
                # the window to itself keeps it from being reported twice.
                previous_hash=segment.previous_segment_hash if anchor is None else anchor,
            )
        )
        anchor = segment.content_hash
        segments.append(segment)
    reasons.extend(_window_head_reasons(head, segments, spec_hash=spec_hash))
    return JournalStatus(
        segment_count=head.segment_count,
        last_sequence=head.last_sequence,
        window_segments=len(segments),
        last_received_time_ns=segments[-1].received_time_ns if segments else None,
        mean_period_seconds=_mean_period_seconds(segments),
        instruments=tuple(
            _instrument_status(instrument, segments) for instrument in spec.instruments
        ),
        verified=not reasons,
        reasons=tuple(reasons),
    )


def _window_anchor(directory: Path, start: int) -> tuple[str | None, list[str]]:
    """The hash a bounded window links back to, and the reason it cannot be read."""
    if start == 0:
        return ZERO_HASH, []
    name = f"{start - 1:010d}.json"
    try:
        document = _read_object(directory / name, label="a journal segment")
    except BinanceCostJournalError:
        return None, [f"SEGMENT_UNREADABLE:{name}"]
    recorded = document.get("content_hash")
    if not isinstance(recorded, str):
        return None, [f"SEGMENT_UNREADABLE:{name}"]
    return recorded, []


def _window_head_reasons(
    head: ChainHead, segments: Sequence[JournalSegment], *, spec_hash: str
) -> list[str]:
    """What the chain head can be wrong about that a bounded window can see."""
    reasons: list[str] = []
    if head.spec_hash != spec_hash:
        reasons.append("CHAIN_HEAD_SPEC_MISMATCH")
    if head.last_sequence != head.segment_count - 1:
        reasons.append("CHAIN_HEAD_SEQUENCE_MISMATCH")
    if not segments:
        reasons.append("CHAIN_HEAD_LINK_UNCHECKED")
    elif head.final_segment_hash != segments[-1].content_hash:
        reasons.append("CHAIN_HEAD_LINK_MISMATCH")
    return reasons


def _mean_period_seconds(segments: Sequence[JournalSegment]) -> Decimal | None:
    """The mean time between the starts of the window's rounds, or ``None``."""
    if len(segments) < 2:
        return None
    span = Decimal(segments[-1].received_time_ns - segments[0].received_time_ns)
    return (span / _NANOSECONDS / Decimal(len(segments) - 1)).quantize(
        _SECOND_QUANTUM, rounding=ROUND_HALF_EVEN
    )


def _instrument_status(
    instrument: JournalInstrument, segments: Sequence[JournalSegment]
) -> InstrumentStatus:
    """How often one instrument reported, and how often its premium index did not."""
    observations = [
        observation
        for segment in segments
        for observation in segment.observations
        if observation.instrument_id == instrument.instrument_id
    ]
    if not observations:
        return InstrumentStatus(
            instrument_id=instrument.instrument_id,
            observation_count=0,
            ok_rate=None,
            premium_index_reason_rate=None,
        )
    total = Decimal(len(observations))
    measured = Decimal(sum(1 for item in observations if item.ok))
    degraded = Decimal(sum(1 for item in observations if item.premium_index_reason is not None))
    return InstrumentStatus(
        instrument_id=instrument.instrument_id,
        observation_count=len(observations),
        ok_rate=(measured / total).quantize(_RATE_QUANTUM, rounding=ROUND_HALF_EVEN),
        premium_index_reason_rate=(degraded / total).quantize(
            _RATE_QUANTUM, rounding=ROUND_HALF_EVEN
        ),
    )


def iso_utc_time(time_ns: int) -> str:
    """A nanosecond stamp as an ISO-8601 UTC time, to the microsecond."""
    seconds, remainder = divmod(time_ns, 1_000_000_000)
    stamp = datetime.fromtimestamp(seconds, tz=UTC).replace(microsecond=remainder // 1_000)
    return stamp.isoformat().replace("+00:00", "Z")


def finalize_journal(
    *, workspace_root: Path, journal_root: Path, output_path: Path, reserve_bytes: int
) -> Path:
    """Publish the journal's finalisation receipt and return its path.

    The whole chain is verified before a single number is computed, so a
    tampered, unreadable or foreign journal produces no receipt at all, and the
    receipt is immutable: an output that already exists is refused rather than
    replaced. Statistics are computed over the measured (``ok``) observations of
    every sampled instrument, eligible or not; only the tier medians are taken
    over the eligible ones (spec 4). A journal that never sampled the
    eligibility notional is refused - eligibility is declared at 5 000 USDT and
    cannot be read off anything else.

    The chain head is read first and everything after it is bounded by it: the
    verification walk and the measurement pass both cover exactly the segments
    ``0 .. head.segment_count - 1``, which is what ``segment_count`` reports and
    what ``chain_head_hash`` binds. A ``run_journal`` appending while the
    receipt is taken is not an error - its segments sit past the head, are left
    out of every number, and are counted in ``segments_beyond_head`` so the
    reading says how much of the journal it did not look at.

    The receipt states the floors it applied, the cadence and target the
    journal declared and the notionals it read, so every decision in it can be
    recomputed from the receipt alone; and a receipt that does not validate is
    refused as a spec error rather than escaping as a pydantic one (W10).
    """
    root = _authorize(workspace_root, journal_root, reserve_bytes)
    output = _authorize(workspace_root, output_path, reserve_bytes)
    if output.exists():
        raise BinanceCostJournalSpecError("this receipt already exists and is immutable")
    spec, spec_hash = load_journal_spec(root)
    keys = _notional_keys(spec.notionals)
    if ELIGIBILITY_NOTIONAL not in keys:
        raise BinanceCostJournalSpecError(
            f"this journal never sampled the {ELIGIBILITY_NOTIONAL} USDT eligibility notional"
        )
    head = _read_chain_head(root)
    if head is None:
        raise BinanceCostJournalSpecError("this journal holds no segment to finalise")
    valid, reasons = _verified_chain(
        root, candidate_head=None, covered_segments=head.segment_count
    )
    if not valid:
        raise BinanceCostJournalSpecError("journal verification failed: " + ",".join(reasons))
    measured = _measured_observations(root, head=head, spec_hash=spec_hash)
    beyond_head = _segments_beyond_head(root, head)
    # Every model below is built inside one guard: a receipt that does not
    # validate is this journal refusing to be read, which the supervisor stops
    # on, never a bare pydantic traceback out of a library call (fix wave W10).
    try:
        instruments = tuple(
            _instrument_statistics(
                instrument, measured.get(instrument.instrument_id, ()), keys=keys
            )
            for instrument in spec.instruments
        )
        tiers = tuple(
            _tier_statistics(instruments, tier=tier, market=market, keys=keys)
            for tier in _RECEIPT_TIERS
            for market in _RECEIPT_MARKETS
        )
        material: dict[str, object] = {
            "version": RECEIPT_VERSION,
            "spec_hash": spec_hash,
            # The head's own hash binds the spec, the count and the final segment.
            "chain_head_hash": head.content_hash,
            "segment_count": head.segment_count,
            "segments_beyond_head": beyond_head,
            "spot_fee_bps_per_side": str(spec.spot_fee_bps_per_side),
            "perpetual_fee_bps_per_side": str(spec.perpetual_fee_bps_per_side),
            "fee_evidence_id": spec.fee_evidence_id,
            "eligibility": {
                "minimum_observations": MINIMUM_OBSERVATIONS,
                "minimum_span_ns": MINIMUM_SPAN_NS,
                "eligibility_notional": ELIGIBILITY_NOTIONAL,
                "tier_minimum_contributors": TIER_MINIMUM_CONTRIBUTORS,
            },
            "target_rounds": spec.target_rounds,
            "notionals": [str(notional) for notional in spec.notionals],
            "sample_interval_seconds": spec.sample_interval_seconds,
            "instruments": [item.model_dump(mode="json") for item in instruments],
            "tiers": [item.model_dump(mode="json") for item in tiers],
            "declaration_rule": DECLARATION_RULE,
            "capital_declaration_rule": CAPITAL_DECLARATION_RULE,
        }
        receipt = FinalizationReceipt.model_validate(
            {**material, "content_hash": content_sha256(material)}
        )
    except ValidationError as error:
        raise BinanceCostJournalSpecError(
            f"the finalisation receipt is invalid: {error}"
        ) from error
    _publish(output, receipt.model_dump(mode="json"))
    return output


def public_binance_json_fetcher(url: str) -> Mapping[str, object]:
    """GET one public Binance JSON object over HTTPS with a 20 s timeout.

    The scheme and the host are checked before any socket is opened, so a URL
    this journal did not build never reaches the network, and both are checked
    again on the URL the response came from, so a redirect can walk off neither
    the allow list nor HTTPS. A non-200 response, a body that is not a JSON
    object and any transport failure are refused as
    ``BinanceCostJournalError``, which the run loop records as that
    instrument's reason for the round.

    A response that carries a status of its own - 429 for a breached rate
    limit, 418 for the ban that follows one, or any other code - is refused as
    a ``BinanceCostJournalTransportError`` carrying that status and the
    ``Retry-After`` the venue sent, so the run loop can wait as long as it was
    asked to (fix wave W7).
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
                raise BinanceCostJournalTransportError(
                    response.status, _retry_after(response.headers)
                )
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
            final_url = str(response.url)
    except urllib.error.HTTPError as error:
        # urllib raises the 4xx and 5xx responses rather than returning them.
        raise BinanceCostJournalTransportError(error.code, _retry_after(error.headers)) from error
    except (OSError, http.client.HTTPException) as error:
        raise BinanceCostJournalError(f"public request failed: {error}") from error
    final_parts = urlsplit(final_url)
    if final_parts.scheme != "https" or final_parts.netloc not in ALLOWED_HOSTS:
        raise BinanceCostJournalError("a redirect left Binance's public HTTPS hosts")
    if len(raw) > _MAX_RESPONSE_BYTES:
        raise BinanceCostJournalError("public response exceeds the byte limit")
    try:
        document: object = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BinanceCostJournalError("public response is not valid JSON") from error
    if not isinstance(document, dict):
        raise BinanceCostJournalError("public response is not a JSON object")
    return document


def _retry_after(headers: Mapping[str, str] | Message) -> int | None:
    """The ``Retry-After`` seconds a throttling response carries, if any.

    Binance sends a plain number of seconds. The HTTP-date form the RFC also
    allows is not parsed: an unreadable value is simply absent and the run
    falls back to its own minute of quiet.
    """
    raw = headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return int(str(raw).strip())
    except ValueError:
        return None


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
) -> tuple[JournalSegment, int]:
    """The round's segment and the seconds the venue asked to be left alone.

    The second member is zero unless some instrument came back 429 or 418; the
    round is written either way and the backoff is the *next* round's, so a
    throttled minute still records what every other instrument displayed.
    """
    received_time_ns = clock()
    sampled = [
        _observe(instrument, notionals=spec.notionals, fetcher=fetcher, clock=clock)
        for instrument in spec.instruments
    ]
    observations = tuple(observation for observation, _ in sampled)
    throttle_seconds = max((seconds for _, seconds in sampled), default=0)
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
    segment = JournalSegment.model_validate(
        {**material, "content_hash": content_sha256(material)}
    )
    return segment, throttle_seconds


def _observe(
    instrument: JournalInstrument,
    *,
    notionals: Sequence[Decimal],
    fetcher: Fetcher,
    clock: Callable[[], int],
) -> tuple[InstrumentObservation, int]:
    """Sample one instrument, turning every failure into its own record.

    The depth request decides whether there is an observation at all. A
    perpetual's premium index is a second request beside it: when that one
    fails the book was still measured, so the transport reason is carried in
    ``premium_index_reason`` and the observation stays ``ok``.

    The second member of the result is the backoff either request earned: the
    seconds Binance asked for when it answered 429 or 418, and zero otherwise
    (fix wave W7).
    """
    keys = _notional_keys(notionals)
    try:
        payload: object = fetcher(instrument.depth_url)
    except Exception as error:  # one instrument's failure is an observation, not an abort
        return (
            _failed_observation(
                instrument_id=instrument.instrument_id,
                received_time_ns=clock(),
                keys=keys,
                reason=_failure_reason(error),
            ),
            _throttle_seconds(error),
        )
    premium_index: object = None
    premium_failure: str | None = None
    throttle_seconds = 0
    if instrument.premium_index_url is not None:
        try:
            premium_index = fetcher(instrument.premium_index_url)
        except Exception as error:
            premium_failure = _failure_reason(error)
            throttle_seconds = _throttle_seconds(error)
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
        return (
            _failed_observation(
                instrument_id=instrument.instrument_id,
                received_time_ns=received_time_ns,
                keys=keys,
                reason=_failure_reason(error),
            ),
            throttle_seconds,
        )
    if premium_failure is None or not observation.ok:
        return observation, throttle_seconds
    # The premium index never arrived, so `depth_observation` recorded "missing
    # premium index"; the transport's own reason says more.
    degraded = InstrumentObservation.model_validate(
        {**observation.model_dump(mode="json"), "premium_index_reason": premium_failure}
    )
    return degraded, throttle_seconds


def _throttle_seconds(error: BaseException) -> int:
    """How long the venue asked to be left alone, zero when it did not ask.

    A 429 or a 418 without a ``Retry-After`` is worth a minute of quiet; one
    that names a longer wait than five minutes is honoured up to that cap,
    because a run that sleeps for an hour on a header stops being a journal
    (fix wave W7).
    """
    if not isinstance(error, BinanceCostJournalTransportError):
        return 0
    if error.status not in _THROTTLE_STATUSES:
        return 0
    requested = _DEFAULT_THROTTLE_SECONDS if error.retry_after is None else error.retry_after
    return min(max(requested, 0), _MAX_THROTTLE_SECONDS)


def _failure_reason(error: Exception) -> str:
    """Name the failure without letting a hostile message bloat the segment."""
    return f"{type(error).__name__}: {error}"[:_MAX_REASON_CHARACTERS]


def _publish_segment(root: Path, *, segment: JournalSegment, spec_hash: str) -> ChainHead:
    document: dict[str, object] = segment.model_dump(mode="json")
    _publish(root / SEGMENT_DIRECTORY_NAME / f"{segment.sequence:010d}.json", document)
    return _publish_chain_head(root, segment=segment, spec_hash=spec_hash)


def _publish_chain_head(root: Path, *, segment: JournalSegment, spec_hash: str) -> ChainHead:
    return _publish_head(root, _chain_head_for(segment, spec_hash=spec_hash))


def _chain_head_for(segment: JournalSegment, *, spec_hash: str) -> ChainHead:
    """The head a chain ending in ``segment`` would carry."""
    return _build_chain_head(
        spec_hash=spec_hash,
        segment_count=segment.sequence + 1,
        last_sequence=segment.sequence,
        final_segment_hash=segment.content_hash,
    )


def _publish_head(root: Path, head: ChainHead) -> ChainHead:
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
    paths = _numbered_segments(journal_root)
    if not paths:
        return None
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


def _numbered_segments(journal_root: Path) -> list[Path]:
    """This journal's segment files in sequence order; anything else is not one."""
    directory = journal_root / SEGMENT_DIRECTORY_NAME
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.glob("*.json") if _SEGMENT_NAME.match(path.name))


def _resumable_chain_head(journal_root: Path) -> ChainHead | None:
    """The head a run resumes from: ``None`` where there is none or it is torn.

    A head file that is not parseable JSON is a publish the last process did
    not finish - the bytes it holds are half of one head and half of another -
    and the segments say what it should have been, so the run rebuilds it (fix
    wave W1). A head that *parses* and then fails its own recorded hash is not
    torn but changed, and stays fatal.
    """
    path = journal_root / CHAIN_HEAD_NAME
    if not path.exists():
        return None
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise BinanceCostJournalSpecError(f"the chain head is unreadable: {error}") from error
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
        raise BinanceCostJournalSpecError("the chain head does not match its recorded hash")
    return head


def _discard_torn_trailing_segment(journal_root: Path, head: ChainHead | None) -> None:
    """Delete the half-written file a killed publish left one past the head (W1).

    Every segment before that one was published and fsynced before the head
    that names it, so the one sequence a crash can catch mid-write is the next
    one. A file there that is not parseable JSON, or does not validate as a
    segment, is that crash caught in the act: it carries no measurement, it is
    deleted and its sequence is sampled again. A file there that *is* a valid
    segment is either the orphan ``_orphan_segment`` adopts or a link the chain
    does not accept, and neither is this function's business.
    """
    sequence = 0 if head is None else head.last_sequence + 1
    path = journal_root / SEGMENT_DIRECTORY_NAME / f"{sequence:010d}.json"
    if not path.exists():
        return
    try:
        raw = path.read_bytes()
    except OSError:
        return
    if _parses_as_segment(raw):
        return
    path.unlink()
    _report_repair(f"SEGMENT_TORN_DISCARDED:{path.name}")


def _parses_as_segment(raw: bytes) -> bool:
    try:
        document: object = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(document, dict):
        return False
    try:
        JournalSegment.model_validate(document)
    except ValidationError:
        return False
    return True


def _rebuilt_chain_head(journal_root: Path, *, spec_hash: str) -> ChainHead:
    """Rebuild and publish the head the segments on disk imply (W1).

    A head that was never written, or one a crash tore in half, carries nothing
    the segments do not: they are walked in name order from ``ZERO_HASH`` under
    the same per-segment checks the verification uses, and the head that walk
    ends in is the head the journal had. A segment that fails any of them stops
    the rebuild - a chain that does not verify is corruption, not a crash, and
    is refused with the reason that broke it.
    """
    paths = _numbered_segments(journal_root)
    previous_hash = ZERO_HASH
    last: JournalSegment | None = None
    for expected_sequence, path in enumerate(paths):
        try:
            document = _read_object(path, label="a journal segment")
            segment = JournalSegment.model_validate(document)
        except (BinanceCostJournalError, ValidationError) as error:
            raise BinanceCostJournalSpecError(
                f"the chain head cannot be rebuilt: SEGMENT_UNREADABLE:{path.name}"
            ) from error
        reasons = _segment_reasons(
            document,
            segment,
            name=path.name,
            expected_sequence=expected_sequence,
            spec_hash=spec_hash,
            previous_hash=previous_hash,
        )
        if reasons:
            raise BinanceCostJournalSpecError(
                "the chain head cannot be rebuilt: " + ",".join(reasons)
            )
        previous_hash = segment.content_hash
        last = segment
    if last is None:
        raise BinanceCostJournalSpecError("the chain head cannot be rebuilt: no segment on disk")
    candidate = _chain_head_for(last, spec_hash=spec_hash)
    valid, chain_reasons = _verified_chain(journal_root, candidate_head=candidate)
    if not valid:
        raise BinanceCostJournalSpecError(
            "the chain head cannot be rebuilt: " + ",".join(chain_reasons)
        )
    _report_repair(f"CHAIN_HEAD_REBUILT:{candidate.segment_count}")
    return _publish_head(journal_root, candidate)


def _report_repair(reason: str) -> None:
    """Name a repair on stderr, the one thing this module writes anywhere.

    There is no logging framework here and the journal runs unattended for
    eight days under a supervisor that keeps its transcript: a file this run
    deleted or a head it rebuilt has to be readable afterwards, or the operator
    is left comparing sequence numbers to work out what happened.
    """
    print(f"binance cost journal repair: {reason}", file=sys.stderr)


def _lock_journal(journal_root: Path) -> int:
    """Take the journal's exclusive write lock, or refuse the run (fix wave W2).

    One journal directory is written by one process: two samplers resuming from
    the same head would each publish a segment at the same sequence, and the
    loser's round would be overwritten by the winner's with no trace in the
    chain. The lock is an OS byte-range lock, so it is released by process exit
    however the process ends - a `run.lock` file left behind by a reboot holds
    nothing and is simply locked again.
    """
    path = journal_root / JOURNAL_LOCK_NAME
    try:
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    except OSError as error:
        raise BinanceCostJournalSpecError(f"the journal lock is unusable: {error}") from error
    try:
        _lock_exclusive(descriptor)
    except OSError as error:
        os.close(descriptor)
        raise BinanceCostJournalSpecError(
            "journal is already being written by another process"
        ) from error
    return descriptor


def _lock_exclusive(descriptor: int) -> None:
    """Take a non-blocking exclusive lock on the first byte, or raise ``OSError``."""
    if sys.platform == "win32":
        msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
    else:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _release_journal_lock(descriptor: int) -> None:
    """Drop the write lock and close its descriptor; the file itself stays."""
    try:
        if sys.platform == "win32":
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _validated_chain_head(document: Mapping[str, object]) -> ChainHead | None:
    try:
        head = ChainHead.model_validate(document)
    except ValidationError:
        return None
    material = {key: value for key, value in document.items() if key != "content_hash"}
    if head.content_hash != content_sha256(material):
        return None
    return head


def _segment_reasons(
    document: Mapping[str, object],
    segment: JournalSegment,
    *,
    name: str,
    expected_sequence: int,
    spec_hash: str,
    previous_hash: str,
) -> list[str]:
    """Everything one segment can be wrong about, given where it sits in the chain.

    The single place the four per-segment checks live, so the verification walk
    and the finalisation's measurement pass cannot drift apart (fix round 3).
    """
    material = {key: value for key, value in document.items() if key != "content_hash"}
    reasons: list[str] = []
    if segment.content_hash != content_sha256(material):
        reasons.append(f"SEGMENT_HASH_MISMATCH:{name}")
    if segment.sequence != expected_sequence:
        reasons.append(f"SEGMENT_SEQUENCE_MISMATCH:{name}")
    if segment.spec_hash != spec_hash:
        reasons.append(f"SEGMENT_SPEC_MISMATCH:{name}")
    if segment.previous_segment_hash != previous_hash:
        reasons.append(f"SEGMENT_LINK_MISMATCH:{name}")
    return reasons


def _chain_head_reasons(
    journal_root: Path,
    *,
    candidate_head: ChainHead | None,
    spec_hash: str,
    segment_count: int,
    final_segment_hash: str,
) -> list[str]:
    head = candidate_head
    if head is None:
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
    elif segment_count == 0:
        return ["CHAIN_HEAD_UNEXPECTED"]
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


def _measured_observations(
    journal_root: Path, *, head: ChainHead, spec_hash: str
) -> dict[str, list[InstrumentObservation]]:
    """Every ``ok`` observation of the chain ``head`` covers, per instrument, in round order.

    The segments are read again rather than kept from the verification pass, and
    each one is re-checked as it is read - its own hash, its spec, its sequence
    and its link - so the numbers are computed from bytes that verified in this
    pass and from nothing else. The segments are addressed by name over
    ``0 .. head.segment_count - 1`` rather than globbed, so a ``run_journal``
    appending concurrently cannot slip a round into the statistics behind the
    receipt's ``segment_count`` (fix round 3). Any mismatch refuses.
    """
    directory = journal_root / SEGMENT_DIRECTORY_NAME
    measured: dict[str, list[InstrumentObservation]] = {}
    previous_hash = ZERO_HASH
    for sequence in range(head.segment_count):
        name = f"{sequence:010d}.json"
        document = _read_object(directory / name, label="a journal segment")
        try:
            segment = JournalSegment.model_validate(document)
        except ValidationError as error:
            raise BinanceCostJournalSpecError(f"a journal segment is invalid: {error}") from error
        reasons = _segment_reasons(
            document,
            segment,
            name=name,
            expected_sequence=sequence,
            spec_hash=spec_hash,
            previous_hash=previous_hash,
        )
        if reasons:
            raise BinanceCostJournalSpecError(
                "a journal segment changed under the measurement pass: " + ",".join(reasons)
            )
        previous_hash = segment.content_hash
        for observation in segment.observations:
            if observation.ok:
                measured.setdefault(observation.instrument_id, []).append(observation)
    if previous_hash != head.final_segment_hash:
        raise BinanceCostJournalSpecError(
            "the measured segments do not end in the chain head's final segment"
        )
    return measured


def _segments_beyond_head(journal_root: Path, head: ChainHead) -> int:
    """Numbered segments sitting past the head - a run still appending, left out."""
    directory = journal_root / SEGMENT_DIRECTORY_NAME
    paths = [path for path in directory.glob("*.json") if _SEGMENT_NAME.match(path.name)]
    return max(len(paths) - head.segment_count, 0)


def _instrument_statistics(
    instrument: JournalInstrument,
    observations: Sequence[InstrumentObservation],
    *,
    keys: Sequence[str],
) -> InstrumentStatistics:
    """Summarise one instrument's measured rounds and decide its eligibility."""
    slippage: dict[str, dict[str, Decimal | int | None]] = {}
    for key in keys:
        values = [observation.slippage_bps_per_side.get(key) for observation in observations]
        filled = [value for value in values if value is not None]
        slippage[key] = {
            "count": len(filled),
            "insufficient_depth": len(values) - len(filled),
            **{name: _quantile_or_none(filled, quantile) for name, quantile in _QUANTILES},
        }
    stamps = [
        observation.received_time_ns
        for observation in observations
        if observation.slippage_bps_per_side.get(ELIGIBILITY_NOTIONAL) is not None
    ]
    first_time_ns: int | None = None
    last_time_ns: int | None = None
    span_ns: int | None = None
    if stamps:
        # The earliest and the latest stamp, not the first and the last round:
        # a clock the host stepped backwards mid-run would otherwise read as a
        # negative span and refuse the whole receipt (fix wave W10).
        first_time_ns = min(stamps)
        last_time_ns = max(stamps)
        span_ns = last_time_ns - first_time_ns
    reason_codes: list[str] = []
    if len(stamps) < MINIMUM_OBSERVATIONS:
        reason_codes.append(_OBSERVATION_FLOOR_CODE)
    if span_ns is None or span_ns < MINIMUM_SPAN_NS:
        reason_codes.append(_SPAN_FLOOR_CODE)
    spreads = [item.spread_bps for item in observations if item.spread_bps is not None]
    bases = [item.basis_bps for item in observations if item.basis_bps is not None]
    return InstrumentStatistics(
        instrument_id=instrument.instrument_id,
        tier=instrument.tier,
        market=instrument.market,
        eligible=not reason_codes,
        reason_codes=tuple(reason_codes),
        observation_count=len(observations),
        span_observation_count=len(stamps),
        first_time_ns=first_time_ns,
        last_time_ns=last_time_ns,
        spread_bps_p50=_quantile_or_none(spreads, _P50),
        slippage=slippage,
        basis_bps_p50=_quantile_or_none(bases, _P50),
    )


def _tier_statistics(
    instruments: Sequence[InstrumentStatistics], *, tier: int, market: str, keys: Sequence[str]
) -> TierStatistics:
    """The medians of one tier's eligible instruments on one market, notional by notional.

    Eligibility is decided at 5 000 USDT, so a member that never filled 50 000
    is eligible and still contributes nothing there. Each notional reports how
    many members carried a number and how many did not, and below
    ``TIER_MINIMUM_CONTRIBUTORS`` contributors the medians are withheld rather
    than taken over a handful (fix wave W9).
    """
    members = [
        item
        for item in instruments
        if item.eligible and item.tier == tier and item.market == market
    ]
    slippage: dict[str, dict[str, Decimal | int | None]] = {}
    for key in keys:
        fifties = _member_quantiles(members, key=key, name="p50")
        nineties = _member_quantiles(members, key=key, name="p90")
        contributing = len(fifties)
        reported = contributing >= TIER_MINIMUM_CONTRIBUTORS
        slippage[key] = {
            "p50_of_p50": _quantile_or_none(fifties, _P50) if reported else None,
            "p50_of_p90": _quantile_or_none(nineties, _P50) if reported else None,
            "contributing_count": contributing,
            "absent_count": len(members) - contributing,
        }
    return TierStatistics(
        tier=tier, market=market, instrument_count=len(members), slippage=slippage
    )


def _member_quantiles(
    members: Sequence[InstrumentStatistics], *, key: str, name: str
) -> list[Decimal]:
    """One quantile of every member at one notional, skipping those that never filled it."""
    values = [item.slippage.get(key, {}).get(name) for item in members]
    return [value for value in values if isinstance(value, Decimal)]


def _quantile(values: Sequence[Decimal], quantile: Decimal) -> Decimal:
    """The OKX journal's rule: the ``ceil(n * q) - 1`` index of the ascending values."""
    ordered = sorted(values)
    index = int((Decimal(len(ordered)) * quantile).to_integral_value(rounding=ROUND_CEILING)) - 1
    return ordered[max(index, 0)]


def _quantile_or_none(values: Sequence[Decimal], quantile: Decimal) -> Decimal | None:
    return _quantile(values, quantile) if values else None


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

    The bytes are flushed and fsynced before the rename, so a machine that
    loses power mid-publish finds either the whole old artifact or the whole
    new one, never the first half of a segment (fix wave W1). The directory
    entry itself is not fsynced: Windows hands out no descriptor for a
    directory, so that half of the POSIX recipe is unavailable here and the
    rename's own durability is what the platform gives.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(canonical_json(document))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
