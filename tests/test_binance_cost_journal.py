import json
import urllib.request
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import pytest
from pydantic import ValidationError

from tests.carry_fixtures import build_captures, small_carry_config
from tests.test_panel_fold_run import DAY_MS, EPOCH_DAY_2020, MONTH_START_DAY
from trading_bot.binance_cost_journal import (
    ALLOWED_HOSTS,
    DEPTH_LIMIT,
    FEE_EVIDENCE_ID,
    JOURNAL_VERSION,
    MINIMUM_OBSERVATIONS,
    MINIMUM_SPAN_NS,
    NOTIONALS,
    PERPETUAL_FEE_BPS_PER_SIDE,
    SAMPLE_INTERVAL_SECONDS,
    SPOT_FEE_BPS_PER_SIDE,
    TARGET_ROUNDS,
    ZERO_HASH,
    BinanceCostJournalError,
    BinanceCostJournalSpec,
    BinanceCostJournalSpecError,
    ChainHead,
    InstrumentObservation,
    JournalInstrument,
    JournalSegment,
    create_journal,
    depth_observation,
    derive_sample,
    public_binance_json_fetcher,
    run_journal,
    verify_journal,
    walk_notional,
)
from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.cli import main

RECEIVED_NS = 1_757_000_123_456_789
SPOT_DEPTH_URL = "https://api.binance.com/api/v3/depth?symbol=BTCUSDT&limit=500"
PERP_DEPTH_URL = "https://fapi.binance.com/fapi/v1/depth?symbol=BTCUSDT&limit=500"
PREMIUM_URL = "https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT"

SPOT = JournalInstrument(
    instrument_id="spot:BTCUSDT",
    market="spot",
    symbol="BTCUSDT",
    pair_symbol="BTCUSDT",
    tier=1,
    depth_url=SPOT_DEPTH_URL,
    premium_index_url=None,
)
PERP = JournalInstrument(
    instrument_id="perp:BTCUSDT",
    market="um",
    symbol="BTCUSDT",
    pair_symbol="BTCUSDT",
    tier=1,
    depth_url=PERP_DEPTH_URL,
    premium_index_url=PREMIUM_URL,
)

# Three ask levels, best first: cumulative quote 300, 2 300, 11 300.
WALK_LEVELS: tuple[tuple[Decimal, Decimal], ...] = (
    (Decimal("100"), Decimal("3")),
    (Decimal("200"), Decimal("10")),
    (Decimal("225"), Decimal("40")),
)


def depth_document() -> dict[str, object]:
    """A synthetic book with mid 100: ask notional 13 500, bid notional 8 450."""
    return {
        "lastUpdateId": 8_100_200,
        "bids": [["75", "4"], ["50", "88"], ["37.5", "100"]],
        "asks": [["125", "4"], ["200", "15"], ["250", "40"]],
    }


def premium_document() -> dict[str, object]:
    return {
        "symbol": "BTCUSDT",
        "markPrice": "100.5",
        "indexPrice": "100",
        "lastFundingRate": "0.0001",
        "nextFundingTime": 1_757_001_600_000,
        "time": 1_757_000_123_456,
    }


def instrument_document() -> dict[str, object]:
    return {
        "instrument_id": "spot:BTCUSDT",
        "market": "spot",
        "symbol": "BTCUSDT",
        "pair_symbol": "BTCUSDT",
        "tier": 1,
        "depth_url": SPOT_DEPTH_URL,
        "premium_index_url": None,
    }


def spec_document() -> dict[str, object]:
    return {
        "version": JOURNAL_VERSION,
        "run_id": "binance-carry-v1",
        "created_time_ns": RECEIVED_NS,
        "sample_decision_close_ns": 1_756_000_000_000_000_000,
        "perpetual_capture_root_hash": "a" * 64,
        "spot_capture_root_hash": "b" * 64,
        "instruments": [instrument_document()],
        "notionals": [str(notional) for notional in NOTIONALS],
        "depth_limit": DEPTH_LIMIT,
        "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
        "target_rounds": TARGET_ROUNDS,
        "spot_fee_bps_per_side": "10",
        "perpetual_fee_bps_per_side": "5",
        "fee_evidence_id": "BINANCE:fee-schedule:2026-09-11:standard-taker",
    }


def observation_document() -> dict[str, object]:
    return {
        "instrument_id": "spot:BTCUSDT",
        "received_time_ns": RECEIVED_NS,
        "ok": True,
        "reason": None,
        "spread_bps": "5000",
        "slippage_bps_per_side": {"500": "3750", "5000": "10000", "50000": None},
        "displayed_notional_thinner_side": "8450",
        "funding_rate": None,
        "basis_bps": None,
    }


def segment_document() -> dict[str, object]:
    return {
        "version": JOURNAL_VERSION,
        "sequence": 0,
        "spec_hash": "c" * 64,
        "previous_segment_hash": ZERO_HASH,
        "received_time_ns": RECEIVED_NS,
        "observations": [observation_document()],
        "content_hash": "d" * 64,
    }


def chain_head_document() -> dict[str, object]:
    return {
        "version": JOURNAL_VERSION,
        "spec_hash": "c" * 64,
        "segment_count": 1,
        "last_sequence": 0,
        "final_segment_hash": "d" * 64,
        "content_hash": "e" * 64,
    }


def test_frozen_constants_match_the_declaration() -> None:
    assert list(NOTIONALS) == [Decimal("500"), Decimal("5000"), Decimal("50000")]
    assert DEPTH_LIMIT == 500
    assert SAMPLE_INTERVAL_SECONDS == 61
    assert TARGET_ROUNDS == 10_000
    assert MINIMUM_OBSERVATIONS == 10_000
    assert MINIMUM_SPAN_NS == 604_800_000_000_000
    assert sorted(ALLOWED_HOSTS) == ["api.binance.com", "fapi.binance.com"]
    assert ZERO_HASH == "0" * 64


def test_book_walk_is_hand_computable_at_every_notional() -> None:
    # 500: 3 base at 100 (300 quote) plus 1 base at 200 -> 4 base, vwap 125.
    assert walk_notional(WALK_LEVELS, Decimal("500")) == Decimal("125")
    # 5 000: 13 base for 2 300 quote plus 12 base at 225 -> 25 base, vwap 200.
    assert walk_notional(WALK_LEVELS, Decimal("5000")) == Decimal("200")
    # 50 000: the displayed book holds 11 300 quote, so it cannot fill.
    assert walk_notional(WALK_LEVELS, Decimal("50000")) is None


def test_book_walk_stops_exactly_on_a_level_boundary() -> None:
    single = ((Decimal("100"), Decimal("4")),)
    assert walk_notional(WALK_LEVELS, Decimal("300")) == Decimal("100")
    # The whole displayed book is exactly the notional.
    assert walk_notional(single, Decimal("400")) == Decimal("100")
    assert walk_notional(single, Decimal("401")) is None
    assert walk_notional((), Decimal("500")) is None


def test_book_walk_refuses_non_positive_inputs() -> None:
    with pytest.raises(ValueError, match="notional"):
        walk_notional(WALK_LEVELS, Decimal("0"))
    with pytest.raises(ValueError, match="positive"):
        walk_notional(((Decimal("100"), Decimal("0")),), Decimal("500"))


def test_observation_records_the_worse_side_at_each_notional() -> None:
    observation = depth_observation(
        depth_document(),
        instrument=SPOT,
        received_time_ns=RECEIVED_NS,
        notionals=NOTIONALS,
        premium_index=None,
    )

    assert observation.ok is True
    assert observation.reason is None
    assert observation.instrument_id == "spot:BTCUSDT"
    assert observation.received_time_ns == RECEIVED_NS
    # mid 100, spread (125 - 75) / 100 * 10 000.
    assert observation.spread_bps == Decimal("5000")
    # 500: buy vwap 125 (2 500 bps), sell vwap 62.5 (3 750 bps) -> the sell side.
    assert observation.slippage_bps_per_side["500"] == Decimal("3750")
    # 5 000: buy vwap 200 (10 000 bps), sell vwap 50 (5 000 bps) -> the buy side.
    assert observation.slippage_bps_per_side["5000"] == Decimal("10000")
    # 50 000: neither side displays enough depth.
    assert observation.slippage_bps_per_side["50000"] is None
    assert observation.displayed_notional_thinner_side == Decimal("8450")
    assert observation.funding_rate is None
    assert observation.basis_bps is None


def test_perpetual_observation_carries_funding_and_basis() -> None:
    payload = depth_document()
    payload["E"] = 1_757_000_123_456
    payload["T"] = 1_757_000_123_400

    observation = depth_observation(
        payload,
        instrument=PERP,
        received_time_ns=RECEIVED_NS,
        notionals=NOTIONALS,
        premium_index=premium_document(),
    )

    assert observation.ok is True
    assert observation.funding_rate == Decimal("0.0001")
    # (100.5 - 100) / 100 * 10 000.
    assert observation.basis_bps == Decimal("50")
    assert observation.spread_bps == Decimal("5000")


def test_crossed_and_empty_books_are_refused() -> None:
    crossed = depth_document()
    crossed["bids"] = [["126", "4"]]
    locked = depth_document()
    locked["bids"] = [["125", "4"]]
    empty = depth_document()
    empty["asks"] = []

    crossed_observation = depth_observation(
        crossed,
        instrument=SPOT,
        received_time_ns=RECEIVED_NS,
        notionals=NOTIONALS,
        premium_index=None,
    )
    locked_observation = depth_observation(
        locked,
        instrument=SPOT,
        received_time_ns=RECEIVED_NS,
        notionals=NOTIONALS,
        premium_index=None,
    )
    empty_observation = depth_observation(
        empty,
        instrument=SPOT,
        received_time_ns=RECEIVED_NS,
        notionals=NOTIONALS,
        premium_index=None,
    )

    assert crossed_observation.ok is False
    assert crossed_observation.reason == "crossed book"
    # A locked book (best bid equal to best ask) is not a book either.
    assert locked_observation.ok is False
    assert locked_observation.reason == "crossed book"
    assert empty_observation.ok is False
    assert empty_observation.reason == "empty book"


def test_failed_observation_keeps_every_notional_key_and_drops_metrics() -> None:
    observation = depth_observation(
        {"lastUpdateId": 1, "bids": [], "asks": []},
        instrument=PERP,
        received_time_ns=RECEIVED_NS,
        notionals=NOTIONALS,
        premium_index=premium_document(),
    )

    assert observation.ok is False
    assert set(observation.slippage_bps_per_side) == {"500", "5000", "50000"}
    assert all(value is None for value in observation.slippage_bps_per_side.values())
    assert observation.spread_bps is None
    assert observation.displayed_notional_thinner_side is None
    assert observation.funding_rate is None
    assert observation.basis_bps is None


@pytest.mark.parametrize(
    ("payload", "fragment"),
    [
        ({"lastUpdateId": 1, "asks": [["125", "4"]]}, "bids"),
        ({"bids": [["75", "4"]], "asks": [["125", "4"]]}, "lastUpdateId"),
        ({"lastUpdateId": 1, "bids": [[75.0, "4"]], "asks": [["125", "4"]]}, "price"),
        ({"lastUpdateId": 1, "bids": [["75"]], "asks": [["125", "4"]]}, "quantity"),
        ({"lastUpdateId": 1, "bids": [["75", "0"]], "asks": [["125", "4"]]}, "positive"),
        ({"lastUpdateId": 1, "bids": [["75", "4"]], "asks": "deep"}, "asks"),
        (
            {"lastUpdateId": 1, "bids": [["75", "4"], ["80", "4"]], "asks": [["125", "4"]]},
            "sorted",
        ),
        (
            {"lastUpdateId": 1, "bids": [["75", "4"], ["75", "4"]], "asks": [["125", "4"]]},
            "sorted",
        ),
        (
            {"lastUpdateId": 1, "bids": [["75", "4"]], "asks": [["125", "4"], ["125", "4"]]},
            "sorted",
        ),
        (
            {"lastUpdateId": 1, "bids": [["75", "4"]], "asks": [["125", "4"], ["120", "4"]]},
            "sorted",
        ),
    ],
)
def test_malformed_depth_payloads_become_failed_observations(
    payload: dict[str, object], fragment: str
) -> None:
    observation = depth_observation(
        payload,
        instrument=SPOT,
        received_time_ns=RECEIVED_NS,
        notionals=NOTIONALS,
        premium_index=None,
    )

    assert observation.ok is False
    assert observation.reason is not None
    assert fragment in observation.reason


@pytest.mark.parametrize(
    "payload",
    [["a", "b"], "depth", None, 7, 1.5, [["125", "4"]]],
)
def test_payloads_that_are_not_objects_become_failed_observations(payload: object) -> None:
    observation = depth_observation(
        payload,
        instrument=SPOT,
        received_time_ns=RECEIVED_NS,
        notionals=NOTIONALS,
        premium_index=None,
    )

    assert observation.ok is False
    assert observation.reason is not None
    assert "depth payload" in observation.reason


@pytest.mark.parametrize("premium_index", [["a", "b"], "premium", 7, 1.5])
def test_premium_indexes_that_are_not_objects_become_failed_observations(
    premium_index: object,
) -> None:
    observation = depth_observation(
        depth_document(),
        instrument=PERP,
        received_time_ns=RECEIVED_NS,
        notionals=NOTIONALS,
        premium_index=premium_index,
    )

    assert observation.ok is False
    assert observation.reason is not None
    assert "premiumIndex" in observation.reason


def test_measurements_are_quantised_to_the_declared_precision() -> None:
    # A realistic book: every measurement repeats far past the recorded precision.
    payload = {
        "lastUpdateId": 8_100_201,
        "bids": [["99.99", "12"], ["99.98", "40"]],
        "asks": [["100.01", "12"], ["100.02", "40"]],
    }
    premium = premium_document()
    premium["markPrice"] = "100.005"
    premium["indexPrice"] = "99.997"

    observation = depth_observation(
        payload,
        instrument=PERP,
        received_time_ns=RECEIVED_NS,
        notionals=NOTIONALS,
        premium_index=premium,
    )

    assert observation.ok is True
    assert str(observation.spread_bps) == "2.000000"
    # Half the spread, but the division leaves a tail 22 digits down.
    assert str(observation.slippage_bps_per_side["500"]) == "1.000000"
    assert str(observation.slippage_bps_per_side["5000"]) == "1.760042"
    assert observation.slippage_bps_per_side["50000"] is None
    # min(1 199.88 + 3 999.20, 1 200.12 + 4 000.80).
    assert str(observation.displayed_notional_thinner_side) == "5199.08"
    # (100.005 - 99.997) / 99.997 * 10 000 = 0.8000240007200216...
    assert str(observation.basis_bps) == "0.800024"
    # The funding rate is recorded as the venue sent it.
    assert str(observation.funding_rate) == "0.0001"


def test_hand_computed_values_carry_the_declared_precision() -> None:
    observation = depth_observation(
        depth_document(),
        instrument=SPOT,
        received_time_ns=RECEIVED_NS,
        notionals=NOTIONALS,
        premium_index=None,
    )

    assert str(observation.spread_bps) == "5000.000000"
    assert str(observation.slippage_bps_per_side["500"]) == "3750.000000"
    assert str(observation.slippage_bps_per_side["5000"]) == "10000.000000"
    assert str(observation.displayed_notional_thinner_side) == "8450.00"


def test_a_book_below_the_recorded_precision_is_not_a_measurement() -> None:
    payload = {
        "lastUpdateId": 1,
        "bids": [["100000000000000000000", "3"]],
        "asks": [["100000000000000000001", "3"]],
    }

    observation = depth_observation(
        payload,
        instrument=SPOT,
        received_time_ns=RECEIVED_NS,
        notionals=NOTIONALS,
        premium_index=None,
    )

    assert observation.ok is False
    assert observation.reason == "book below the recorded precision"


def test_a_book_beyond_the_recorded_precision_is_not_a_measurement() -> None:
    # A displayed notional so large that recording it to a cent overflows the context.
    payload = {
        "lastUpdateId": 1,
        "bids": [["1" + "0" * 30, "1" + "0" * 10]],
        "asks": [["2" + "0" * 30, "1" + "0" * 10]],
    }

    observation = depth_observation(
        payload,
        instrument=SPOT,
        received_time_ns=RECEIVED_NS,
        notionals=NOTIONALS,
        premium_index=None,
    )

    assert observation.ok is False
    assert observation.reason == "measurement exceeds the recorded precision"


def test_premium_index_failures_are_observations_not_exceptions() -> None:
    foreign = premium_document()
    foreign["symbol"] = "ETHUSDT"

    missing = depth_observation(
        depth_document(),
        instrument=PERP,
        received_time_ns=RECEIVED_NS,
        notionals=NOTIONALS,
        premium_index=None,
    )
    mismatched = depth_observation(
        depth_document(),
        instrument=PERP,
        received_time_ns=RECEIVED_NS,
        notionals=NOTIONALS,
        premium_index=foreign,
    )
    unexpected = depth_observation(
        depth_document(),
        instrument=SPOT,
        received_time_ns=RECEIVED_NS,
        notionals=NOTIONALS,
        premium_index=premium_document(),
    )

    assert missing.ok is False
    assert missing.reason == "missing premium index"
    assert mismatched.ok is False
    assert mismatched.reason == "premium index symbol mismatch"
    assert unexpected.ok is False
    assert unexpected.reason == "unexpected premium index"


def test_depth_observation_refuses_an_unusable_notional_set() -> None:
    with pytest.raises(ValueError, match="notional"):
        depth_observation(
            depth_document(),
            instrument=SPOT,
            received_time_ns=RECEIVED_NS,
            notionals=(),
            premium_index=None,
        )
    with pytest.raises(ValueError, match="distinct"):
        depth_observation(
            depth_document(),
            instrument=SPOT,
            received_time_ns=RECEIVED_NS,
            notionals=(Decimal("500"), Decimal("500")),
            premium_index=None,
        )


def test_instrument_binds_identity_market_and_host() -> None:
    document = instrument_document()
    with pytest.raises(ValidationError):
        JournalInstrument.model_validate({**document, "instrument_id": "spot:ETHUSDT"})
    with pytest.raises(ValidationError):
        foreign_host = "https://api.binance.us/api/v3/depth?symbol=BTCUSDT&limit=500"
        JournalInstrument.model_validate({**document, "depth_url": foreign_host})
    with pytest.raises(ValidationError):
        plain_http = "http://api.binance.com/api/v3/depth?symbol=BTCUSDT&limit=500"
        JournalInstrument.model_validate({**document, "depth_url": plain_http})
    with pytest.raises(ValidationError):
        # The spot host belongs to the spot leg only.
        JournalInstrument.model_validate({**document, "depth_url": PERP_DEPTH_URL})
    with pytest.raises(ValidationError):
        # A spot leg has no premium index.
        JournalInstrument.model_validate({**document, "premium_index_url": PREMIUM_URL})
    with pytest.raises(ValidationError):
        # A perpetual leg must have one.
        JournalInstrument.model_validate(
            {
                **document,
                "instrument_id": "perp:BTCUSDT",
                "market": "um",
                "depth_url": PERP_DEPTH_URL,
            }
        )


def test_spec_segment_and_head_round_trip() -> None:
    spec = BinanceCostJournalSpec.model_validate(spec_document())
    segment = JournalSegment.model_validate(segment_document())
    head = ChainHead.model_validate(chain_head_document())

    assert spec.version == JOURNAL_VERSION
    assert spec.notionals == NOTIONALS
    assert spec.spot_fee_bps_per_side == Decimal("10")
    assert spec.instruments[0] == SPOT
    assert segment.previous_segment_hash == ZERO_HASH
    assert segment.observations[0].slippage_bps_per_side["50000"] is None
    assert head.final_segment_hash == segment.content_hash


def test_chain_models_reject_broken_identities() -> None:
    spec = spec_document()
    with pytest.raises(ValidationError):
        BinanceCostJournalSpec.model_validate({**spec, "spot_capture_root_hash": "b" * 63})
    with pytest.raises(ValidationError):
        BinanceCostJournalSpec.model_validate({**spec, "instruments": []})
    with pytest.raises(ValidationError):
        BinanceCostJournalSpec.model_validate({**spec, "notionals": ["5000", "500"]})
    with pytest.raises(ValidationError):
        BinanceCostJournalSpec.model_validate({**spec, "target_rounds": 0})
    with pytest.raises(ValidationError):
        JournalSegment.model_validate({**segment_document(), "content_hash": "D" * 64})
    with pytest.raises(ValidationError):
        JournalSegment.model_validate({**segment_document(), "sequence": -1})
    with pytest.raises(ValidationError):
        JournalSegment.model_validate({**segment_document(), "observations": []})
    with pytest.raises(ValidationError):
        ChainHead.model_validate({**chain_head_document(), "segment_count": -1})


def test_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        JournalInstrument.model_validate({**instrument_document(), "unexpected": 1})
    with pytest.raises(ValidationError):
        BinanceCostJournalSpec.model_validate({**spec_document(), "unexpected": 1})
    with pytest.raises(ValidationError):
        InstrumentObservation.model_validate({**observation_document(), "unexpected": 1})
    with pytest.raises(ValidationError):
        JournalSegment.model_validate({**segment_document(), "unexpected": 1})
    with pytest.raises(ValidationError):
        ChainHead.model_validate({**chain_head_document(), "unexpected": 1})


def test_observation_model_binds_ok_to_its_evidence() -> None:
    document = observation_document()
    with pytest.raises(ValidationError):
        # ok without a spread.
        InstrumentObservation.model_validate({**document, "spread_bps": None})
    with pytest.raises(ValidationError):
        # ok with a reason.
        InstrumentObservation.model_validate({**document, "reason": "crossed book"})
    with pytest.raises(ValidationError):
        # not ok without a reason.
        InstrumentObservation.model_validate({**document, "ok": False})
    with pytest.raises(ValidationError):
        # not ok but still carrying metrics.
        InstrumentObservation.model_validate({**document, "ok": False, "reason": "empty book"})


# --- Task 2: sample derivation, journal creation, run loop, verification -------------

Captures = tuple[Path, Path, Path, Path]  # workspace root, perpetual, spot, family spec
# The reduced carry fixture's last daily bar closes on 2020-07-31; both captures share
# that calendar, so the latest supported decision is one week earlier.
LAST_CLOSE_NS = ((EPOCH_DAY_2020 + MONTH_START_DAY["2020-07"] + 31) * DAY_MS - 1) * 1_000_000
SAMPLE_DECISION_NS = LAST_CLOSE_NS - 604_800_000_000_000
# Twelve pairs are all the fixture has, so of the declared tier-two ranks
# (9, 12, 15, 19, 23, 27, 31, 35) only 9 and 12 exist.
SAMPLE_SYMBOLS = (*(f"C{index:02d}USDT" for index in range(8)), "C08USDT", "C11USDT")


@pytest.fixture
def captures(tmp_path: Path) -> Captures:
    perpetual, spot = build_captures(tmp_path)
    return tmp_path, perpetual, spot, small_carry_config(tmp_path)


def perpetual_instrument_document() -> dict[str, object]:
    return {
        "instrument_id": "perp:BTCUSDT",
        "market": "um",
        "symbol": "BTCUSDT",
        "pair_symbol": "BTCUSDT",
        "tier": 1,
        "depth_url": PERP_DEPTH_URL,
        "premium_index_url": PREMIUM_URL,
    }


def write_journal_directory(journal_root: Path, **overrides: object) -> str:
    """Write a two-instrument journal the way `create_journal` does; returns the spec hash."""
    document: dict[str, object] = {
        **spec_document(),
        "instruments": [instrument_document(), perpetual_instrument_document()],
        **overrides,
    }
    BinanceCostJournalSpec.model_validate(document)
    spec_hash = content_sha256(document)
    (journal_root / "segments").mkdir(parents=True)
    (journal_root / "journal-spec.json").write_bytes(
        canonical_json({**document, "spec_hash": spec_hash})
    )
    return spec_hash


def read_document(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def segment_documents(journal_root: Path) -> list[dict[str, object]]:
    return [read_document(path) for path in sorted((journal_root / "segments").glob("*.json"))]


def observations_of(document: dict[str, object]) -> list[dict[str, object]]:
    value = document["observations"]
    assert isinstance(value, list)
    return value


class FakeVenue:
    """A fetcher over the synthetic book; `dark` and `broken` are URL fragments.

    `dark` raises the way a transport failure does, `broken` answers with an
    unusable payload. Name the spot host with its scheme: "api.binance.com" is
    a substring of the perpetual host "fapi.binance.com".
    """

    def __init__(self, *, dark: tuple[str, ...] = (), broken: tuple[str, ...] = ()) -> None:
        self.urls: list[str] = []
        self.dark = dark
        self.broken = broken

    def __call__(self, url: str) -> Mapping[str, object]:
        self.urls.append(url)
        symbol = parse_qs(urlsplit(url).query)["symbol"][0]
        if any(fragment in url for fragment in self.dark):
            raise ConnectionError(f"{symbol} is unreachable")
        if any(fragment in url for fragment in self.broken):
            return {"lastUpdateId": 1, "bids": [], "asks": []}
        if "premiumIndex" in url:
            return {**premium_document(), "symbol": symbol}
        return depth_document()


class FakeClock:
    def __init__(self, start: int = RECEIVED_NS, step: int = 1_000_000) -> None:
        self.now = start
        self.step = step

    def __call__(self) -> int:
        self.now += self.step
        return self.now


class FakeSleep:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def test_derive_sample_takes_ranks_one_to_eight_and_the_declared_tier_two_ranks(
    captures: Captures,
) -> None:
    _, perpetual, spot, family_spec = captures
    decision_close_ns, instruments = derive_sample(perpetual, spot, family_spec_path=family_spec)
    assert decision_close_ns == SAMPLE_DECISION_NS
    assert decision_close_ns == 1_595_635_199_999_000_000
    assert [item.instrument_id for item in instruments] == [
        f"{prefix}:{symbol}" for symbol in SAMPLE_SYMBOLS for prefix in ("spot", "perp")
    ]
    # Ranks 1-8 are the journal's tier one, the rest its tier two.
    assert [item.tier for item in instruments] == [1] * 16 + [2] * 4
    assert all(item.pair_symbol == item.symbol for item in instruments)
    assert [item.market for item in instruments[:2]] == ["spot", "um"]
    assert instruments[0].depth_url == (
        "https://api.binance.com/api/v3/depth?symbol=C00USDT&limit=500"
    )
    assert instruments[0].premium_index_url is None
    assert instruments[1].depth_url == (
        "https://fapi.binance.com/fapi/v1/depth?symbol=C00USDT&limit=500"
    )
    assert instruments[1].premium_index_url == (
        "https://fapi.binance.com/fapi/v1/premiumIndex?symbol=C00USDT"
    )


def test_create_journal_writes_an_immutable_spec_and_an_empty_segment_directory(
    captures: Captures,
) -> None:
    root, perpetual, spot, family_spec = captures
    journal = root / "journal"
    path = create_journal(
        workspace_root=root, journal_root=journal, reserve_bytes=0, run_id="binance-carry-v1",
        perp_capture_root=perpetual, spot_capture_root=spot, family_spec_path=family_spec,
    )
    assert path == journal / "journal-spec.json"
    document = read_document(path)
    assert document.pop("spec_hash") == content_sha256(document)
    spec = BinanceCostJournalSpec.model_validate(document)
    assert spec.run_id == "binance-carry-v1"
    assert spec.created_time_ns > 0
    assert spec.sample_decision_close_ns == SAMPLE_DECISION_NS
    assert len(spec.instruments) == 20
    assert spec.notionals == NOTIONALS
    assert spec.depth_limit == DEPTH_LIMIT
    assert spec.sample_interval_seconds == SAMPLE_INTERVAL_SECONDS
    assert spec.target_rounds == TARGET_ROUNDS
    assert spec.spot_fee_bps_per_side == SPOT_FEE_BPS_PER_SIDE
    assert spec.perpetual_fee_bps_per_side == PERPETUAL_FEE_BPS_PER_SIDE
    assert spec.fee_evidence_id == FEE_EVIDENCE_ID
    perpetual_manifest = read_document(perpetual / "capture-manifest.json")
    spot_manifest = read_document(spot / "capture-manifest.json")
    assert spec.perpetual_capture_root_hash == perpetual_manifest["capture_root_hash"]
    assert spec.spot_capture_root_hash == spot_manifest["capture_root_hash"]
    assert list((journal / "segments").iterdir()) == []
    assert verify_journal(journal) == (True, ())
    with pytest.raises(BinanceCostJournalSpecError):
        create_journal(
            workspace_root=root, journal_root=journal, reserve_bytes=0,
            run_id="binance-carry-v1", perp_capture_root=perpetual, spot_capture_root=spot,
            family_spec_path=family_spec,
        )


def test_run_journal_writes_linked_segments_and_a_chain_head(tmp_path: Path) -> None:
    journal = tmp_path / "journal"
    spec_hash = write_journal_directory(journal)
    venue, clock, sleeper = FakeVenue(), FakeClock(), FakeSleep()
    head = run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=3,
        fetcher=venue, clock=clock, sleep=sleeper,
    )
    assert (head.segment_count, head.last_sequence, head.spec_hash) == (3, 2, spec_hash)
    # One spot depth, one perpetual depth and one premium index per round.
    assert len(venue.urls) == 9
    # The interval is waited between rounds, never before the first or after the last.
    assert sleeper.calls == [SAMPLE_INTERVAL_SECONDS, SAMPLE_INTERVAL_SECONDS]
    assert [path.name for path in sorted((journal / "segments").glob("*.json"))] == [
        "0000000000.json", "0000000001.json", "0000000002.json",
    ]
    previous = ZERO_HASH
    for sequence, document in enumerate(segment_documents(journal)):
        assert document["version"] == JOURNAL_VERSION
        assert document["sequence"] == sequence
        assert document["spec_hash"] == spec_hash
        assert document["previous_segment_hash"] == previous
        material = {key: value for key, value in document.items() if key != "content_hash"}
        assert document["content_hash"] == content_sha256(material)
        assert [item["ok"] for item in observations_of(document)] == [True, True]
        assert [item["instrument_id"] for item in observations_of(document)] == [
            "spot:BTCUSDT", "perp:BTCUSDT",
        ]
        previous = str(document["content_hash"])
    head_document = read_document(journal / "chain-head.json")
    assert head_document["final_segment_hash"] == previous
    assert head_document["segment_count"] == 3
    assert head == ChainHead.model_validate(head_document)
    assert verify_journal(journal) == (True, ())


def test_run_journal_resumes_from_the_chain_head(tmp_path: Path) -> None:
    journal = tmp_path / "journal"
    write_journal_directory(journal)
    first = run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=2,
        fetcher=FakeVenue(), clock=FakeClock(), sleep=FakeSleep(),
    )
    second = run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=2,
        fetcher=FakeVenue(), clock=FakeClock(start=RECEIVED_NS + 10**12), sleep=FakeSleep(),
    )
    assert (first.segment_count, first.last_sequence) == (2, 1)
    assert (second.segment_count, second.last_sequence) == (4, 3)
    documents = segment_documents(journal)
    assert len(documents) == 4
    assert documents[2]["previous_segment_hash"] == documents[1]["content_hash"]
    assert documents[3]["previous_segment_hash"] == documents[2]["content_hash"]
    assert verify_journal(journal) == (True, ())


def test_run_journal_stops_at_the_declared_target(tmp_path: Path) -> None:
    journal = tmp_path / "journal"
    write_journal_directory(journal, target_rounds=2)
    head = run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=5,
        fetcher=FakeVenue(), clock=FakeClock(), sleep=FakeSleep(),
    )
    assert head.segment_count == 2
    venue = FakeVenue()
    again = run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=5,
        fetcher=venue, clock=FakeClock(), sleep=FakeSleep(),
    )
    assert again == head
    assert venue.urls == []
    assert len(segment_documents(journal)) == 2


def test_a_failing_instrument_is_recorded_and_the_round_is_still_written(tmp_path: Path) -> None:
    journal = tmp_path / "journal"
    write_journal_directory(journal)
    head = run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=1,
        fetcher=FakeVenue(dark=("https://api.binance.com",)), clock=FakeClock(), sleep=FakeSleep(),
    )
    assert head.segment_count == 1
    spot_observation, perpetual_observation = observations_of(segment_documents(journal)[0])
    assert spot_observation["ok"] is False
    assert "ConnectionError" in str(spot_observation["reason"])
    assert spot_observation["slippage_bps_per_side"] == {"500": None, "5000": None, "50000": None}
    assert spot_observation["spread_bps"] is None
    assert perpetual_observation["ok"] is True
    assert verify_journal(journal) == (True, ())


def test_an_invalid_payload_is_recorded_as_a_failed_observation(tmp_path: Path) -> None:
    journal = tmp_path / "journal"
    write_journal_directory(journal)
    run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=1,
        fetcher=FakeVenue(broken=("https://api.binance.com",)),
        clock=FakeClock(), sleep=FakeSleep(),
    )
    spot_observation = observations_of(segment_documents(journal)[0])[0]
    assert spot_observation["ok"] is False
    assert spot_observation["reason"] == "empty book"


def test_a_round_in_which_every_instrument_fails_aborts_the_run(tmp_path: Path) -> None:
    journal = tmp_path / "journal"
    write_journal_directory(journal)
    with pytest.raises(BinanceCostJournalError):
        run_journal(
            workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=1,
            fetcher=FakeVenue(dark=("binance.com",)), clock=FakeClock(), sleep=FakeSleep(),
        )
    assert list((journal / "segments").iterdir()) == []
    assert not (journal / "chain-head.json").exists()
    assert verify_journal(journal) == (True, ())


def test_verify_journal_detects_a_tampered_segment(tmp_path: Path) -> None:
    journal = tmp_path / "journal"
    write_journal_directory(journal)
    run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=2,
        fetcher=FakeVenue(), clock=FakeClock(), sleep=FakeSleep(),
    )
    path = journal / "segments" / "0000000000.json"
    document = read_document(path)
    observations_of(document)[0]["spread_bps"] = "1"
    path.write_bytes(canonical_json(document))
    valid, reasons = verify_journal(journal)
    assert valid is False
    assert "SEGMENT_HASH_MISMATCH:0000000000.json" in reasons
    # An unverified journal is never resumed.
    with pytest.raises(BinanceCostJournalSpecError):
        run_journal(
            workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=1,
            fetcher=FakeVenue(), clock=FakeClock(), sleep=FakeSleep(),
        )


def test_verify_journal_detects_a_rewritten_chain_head(tmp_path: Path) -> None:
    journal = tmp_path / "journal"
    write_journal_directory(journal)
    run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=2,
        fetcher=FakeVenue(), clock=FakeClock(), sleep=FakeSleep(),
    )
    path = journal / "chain-head.json"
    document = read_document(path)
    material = {key: value for key, value in document.items() if key != "content_hash"}
    material["segment_count"] = 3
    path.write_bytes(canonical_json({**material, "content_hash": content_sha256(material)}))
    valid, reasons = verify_journal(journal)
    assert valid is False
    assert "CHAIN_HEAD_COUNT_MISMATCH" in reasons
    # A head whose own hash no longer covers its fields is caught too.
    path.write_bytes(canonical_json({**document, "last_sequence": 7}))
    assert verify_journal(journal) == (False, ("CHAIN_HEAD_UNREADABLE",))


def test_verify_journal_detects_a_spec_that_no_longer_matches(tmp_path: Path) -> None:
    journal = tmp_path / "journal"
    write_journal_directory(journal)
    run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=1,
        fetcher=FakeVenue(), clock=FakeClock(), sleep=FakeSleep(),
    )
    path = journal / "journal-spec.json"
    document = read_document(path)
    path.write_bytes(canonical_json({**document, "run_id": "someone-elses-journal"}))
    assert verify_journal(journal) == (False, ("JOURNAL_SPEC_UNVERIFIED",))


def test_the_public_fetcher_refuses_a_foreign_host_before_any_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(*arguments: object, **keywords: object) -> object:
        raise AssertionError("the journal fetcher must not open a connection")

    monkeypatch.setattr(urllib.request, "urlopen", explode)
    refused = (
        "https://evil.example.com/api/v3/depth?symbol=BTCUSDT&limit=500",
        "http://api.binance.com/api/v3/depth?symbol=BTCUSDT&limit=500",
        "https://api.binance.com:8443/api/v3/depth?symbol=BTCUSDT&limit=500",
        "https://user:secret@api.binance.com/api/v3/depth?symbol=BTCUSDT",
        "https://data.binance.vision/api/v3/depth?symbol=BTCUSDT",
    )
    for url in refused:
        with pytest.raises(BinanceCostJournalError):
            public_binance_json_fetcher(url)


def test_the_cli_creates_runs_and_stops_a_journal(captures: Captures) -> None:
    root, perpetual, spot, family_spec = captures
    journal = root / "journal"
    create_arguments = [
        "binance-cost-journal-create", "--workspace-root", str(root),
        "--journal", str(journal), "--run-id", "binance-carry-v1",
        "--perp-capture", str(perpetual), "--spot-capture", str(spot),
        "--family-spec", str(family_spec), "--reserve-bytes", "0",
    ]
    run_arguments = [
        "binance-cost-journal-run", "--workspace-root", str(root),
        "--journal", str(journal), "--rounds", "1", "--reserve-bytes", "0",
    ]
    assert main(create_arguments) == 0
    with patch("trading_bot.cli.public_binance_json_fetcher", side_effect=FakeVenue()):
        assert main(run_arguments) == 0
    assert (journal / "segments" / "0000000000.json").exists()
    assert verify_journal(journal) == (True, ())
    # An existing journal is immutable: the supervisor's stop code, not a retry.
    assert main(create_arguments) == 2
    # A round in which every instrument fails is retryable instead.
    dark = FakeVenue(dark=("binance.com",))
    with patch("trading_bot.cli.public_binance_json_fetcher", side_effect=dark):
        assert main(run_arguments) == 1


def test_the_cli_stops_on_a_missing_or_foreign_journal(tmp_path: Path) -> None:
    absent = [
        "binance-cost-journal-run", "--workspace-root", str(tmp_path),
        "--journal", str(tmp_path / "absent"), "--rounds", "1", "--reserve-bytes", "0",
    ]
    assert main(absent) == 2
    outside = [
        "binance-cost-journal-run", "--workspace-root", str(tmp_path),
        "--journal", str(tmp_path.parent / "outside-journal"),
        "--rounds", "1", "--reserve-bytes", "0",
    ]
    assert main(outside) == 2
    journal = tmp_path / "journal"
    write_journal_directory(journal)
    path = journal / "journal-spec.json"
    document = read_document(path)
    path.write_bytes(canonical_json({**document, "run_id": "someone-elses-journal"}))
    present = [
        "binance-cost-journal-run", "--workspace-root", str(tmp_path),
        "--journal", str(journal), "--rounds", "1", "--reserve-bytes", "0",
    ]
    assert main(present) == 2
