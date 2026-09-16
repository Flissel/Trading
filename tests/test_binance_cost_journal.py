import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from decimal import Decimal
from email.message import Message
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import pytest
from pydantic import ValidationError

from tests.carry_fixtures import build_captures, small_carry_config
from tests.test_panel_fold_run import DAY_MS, EPOCH_DAY_2020, MONTH_START_DAY
from trading_bot import binance_cost_journal as journal_module
from trading_bot.binance_cost_journal import (
    ALLOWED_HOSTS,
    CAPITAL_DECLARATION_RULE,
    DECLARATION_RULE,
    DEPTH_LIMIT,
    ELIGIBILITY_NOTIONAL,
    FEE_EVIDENCE_ID,
    JOURNAL_VERSION,
    MINIMUM_OBSERVATIONS,
    MINIMUM_SPAN_NS,
    NOTIONALS,
    PERPETUAL_FEE_BPS_PER_SIDE,
    SAMPLE_INSTRUMENT_COUNT,
    SAMPLE_INTERVAL_SECONDS,
    SPOT_FEE_BPS_PER_SIDE,
    TARGET_ROUNDS,
    TIER_MINIMUM_CONTRIBUTORS,
    ZERO_HASH,
    BinanceCostJournalError,
    BinanceCostJournalSpec,
    BinanceCostJournalSpecError,
    BinanceCostJournalTransportError,
    ChainHead,
    FinalizationReceipt,
    InstrumentObservation,
    InstrumentStatistics,
    JournalInstrument,
    JournalSegment,
    TierStatistics,
    create_journal,
    depth_observation,
    derive_sample,
    finalize_journal,
    iso_utc_time,
    journal_status,
    public_binance_json_fetcher,
    run_journal,
    verify_journal,
    walk_notional,
)
from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.cli import _journal_exit_code, main
from trading_bot.storage import StoragePolicyError, StorageReserveError

RECEIVED_NS = 1_757_000_123_456_789
# `FakeClock` advances a millisecond a call and a two-instrument round spends
# three of them - the round's own stamp, one an instrument - so the period the
# run loop waits afterwards is that much short of the declared interval (W6).
ROUND_PERIOD_REMAINDER = SAMPLE_INTERVAL_SECONDS - 0.003
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
        "bid_levels": 3,
        "ask_levels": 3,
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
    # 11 000 rounds at 61 s is 7.8 days: margin over the 10 000-observation floor.
    assert TARGET_ROUNDS == 11_000
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
def test_premium_indexes_that_are_not_objects_are_recorded_not_raised(
    premium_index: object,
) -> None:
    observation = depth_observation(
        depth_document(),
        instrument=PERP,
        received_time_ns=RECEIVED_NS,
        notionals=NOTIONALS,
        premium_index=premium_index,
    )

    assert observation.ok is True
    assert observation.premium_index_reason is not None
    assert "premiumIndex" in observation.premium_index_reason


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


def test_a_premium_index_failure_leaves_the_measured_book_standing() -> None:
    """The book was measured; only the funding fields are missing."""
    foreign = premium_document()
    foreign["symbol"] = "ETHUSDT"
    negative = premium_document()
    negative["indexPrice"] = "0"
    cases = {
        "missing premium index": None,
        "premium index symbol mismatch": foreign,
        "premium index prices must be positive": negative,
        "premiumIndex must be an object with string keys": ["a", "b"],
    }
    for reason, premium_index in cases.items():
        observation = depth_observation(
            depth_document(),
            instrument=PERP,
            received_time_ns=RECEIVED_NS,
            notionals=NOTIONALS,
            premium_index=premium_index,
        )
        assert observation.ok is True, reason
        assert observation.reason is None
        assert observation.premium_index_reason == reason
        assert observation.funding_rate is None
        assert observation.basis_bps is None
        # The measurement itself is untouched.
        assert observation.spread_bps == Decimal("5000")
        assert observation.slippage_bps_per_side["500"] == Decimal("3750")


def test_a_spot_leg_handed_a_premium_index_is_still_refused() -> None:
    """Not a venue failure but a caller pairing the wrong payload with a leg."""
    unexpected = depth_observation(
        depth_document(),
        instrument=SPOT,
        received_time_ns=RECEIVED_NS,
        notionals=NOTIONALS,
        premium_index=premium_document(),
    )

    assert unexpected.ok is False
    assert unexpected.reason == "unexpected premium index"
    assert unexpected.premium_index_reason is None


def test_a_measured_observation_binds_its_premium_index_reason() -> None:
    document = observation_document()
    with pytest.raises(ValidationError):
        # A recorded funding rate contradicts a missing premium index.
        InstrumentObservation.model_validate({
            **document, "premium_index_reason": "missing premium index",
            "funding_rate": "0.0001",
        })
    with pytest.raises(ValidationError):
        # A failed observation carries one reason, not two.
        InstrumentObservation.model_validate({
            **document, "ok": False, "reason": "empty book", "spread_bps": None,
            "displayed_notional_thinner_side": None,
            "slippage_bps_per_side": {"500": None, "5000": None, "50000": None},
            "premium_index_reason": "missing premium index",
        })
    with pytest.raises(ValidationError):
        InstrumentObservation.model_validate({**document, "premium_index_reason": ""})


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


def spot_stamps(journal_root: Path) -> list[int]:
    """The spot leg's `received_time_ns` in every segment of a journal, in round order."""
    stamps: list[int] = []
    for document in segment_documents(journal_root):
        observation = next(
            item for item in observations_of(document) if item["instrument_id"] == "spot:BTCUSDT"
        )
        stamp = observation["received_time_ns"]
        assert isinstance(stamp, int)
        stamps.append(stamp)
    return stamps


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
        allow_short_sample=True,
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
            family_spec_path=family_spec, allow_short_sample=True,
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
    # The period is waited between rounds, never before the first or after the last,
    # and each round spends part of it (W6).
    assert sleeper.calls == pytest.approx([ROUND_PERIOD_REMAINDER] * 2)
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


def test_the_cli_creates_runs_and_stops_a_journal(
    captures: Captures, capsys: pytest.CaptureFixture[str]
) -> None:
    root, perpetual, spot, family_spec = captures
    journal = root / "journal"
    create_arguments = [
        "binance-cost-journal-create", "--workspace-root", str(root),
        "--journal", str(journal), "--run-id", "binance-carry-v1",
        "--perp-capture", str(perpetual), "--spot-capture", str(spot),
        "--family-spec", str(family_spec), "--reserve-bytes", "0", "--allow-short-sample",
    ]
    run_arguments = [
        "binance-cost-journal-run", "--workspace-root", str(root),
        "--journal", str(journal), "--rounds", "1", "--reserve-bytes", "0",
    ]
    assert main(create_arguments) == 0
    reported = capsys.readouterr().out
    assert "20 instruments" in reported
    assert "2020-07-24" in reported
    # A round in which every instrument fails is retryable: exit 1, no segment.
    # It runs before the first good round so no run here resumes into the real
    # 61 s interval the CLI passes through.
    dark = FakeVenue(dark=("binance.com",))
    with patch("trading_bot.cli.public_binance_json_fetcher", side_effect=dark):
        assert main(run_arguments) == 1
    assert list((journal / "segments").iterdir()) == []
    with patch("trading_bot.cli.public_binance_json_fetcher", side_effect=FakeVenue()):
        assert main(run_arguments) == 0
    assert (journal / "segments" / "0000000000.json").exists()
    assert verify_journal(journal) == (True, ())
    # An existing journal is immutable: the supervisor's stop code, not a retry.
    assert main(create_arguments) == 2


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


# --- Fix round 1: adoption, cadence on resume, premium-index degradation ------------


def rewind_chain_head(journal_root: Path, *, segments: int) -> None:
    """Put the chain head back where a kill between the two publishes left it.

    `segments` is how many segments the head had counted when the process died:
    one fewer than the segment files on disk, and zero when the kill landed on
    the very first round, which leaves no head at all.
    """
    path = journal_root / "chain-head.json"
    documents = segment_documents(journal_root)[:segments]
    if not documents:
        path.unlink()
        return
    material: dict[str, object] = {
        "version": JOURNAL_VERSION,
        "spec_hash": documents[-1]["spec_hash"],
        "segment_count": len(documents),
        "last_sequence": len(documents) - 1,
        "final_segment_hash": documents[-1]["content_hash"],
    }
    path.write_bytes(canonical_json({**material, "content_hash": content_sha256(material)}))


def rewrite_segment(path: Path, **overrides: object) -> None:
    """Rewrite a segment with its own hash recomputed, the way a writer would."""
    document = {**read_document(path), **overrides}
    material = {key: value for key, value in document.items() if key != "content_hash"}
    path.write_bytes(canonical_json({**material, "content_hash": content_sha256(material)}))


class FakeResponse:
    """The little of `http.client.HTTPResponse` the public fetcher touches."""

    def __init__(
        self,
        *,
        url: str,
        status: int = 200,
        body: bytes = b'{"lastUpdateId": 1}',
        headers: Mapping[str, str] | None = None,
    ):
        self.url = url
        self.status = status
        self.body = body
        self.headers = dict(headers or {})

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *arguments: object) -> None:
        return None

    def read(self, amount: int) -> bytes:
        return self.body[:amount]


def urlopen_returning(response: FakeResponse) -> Callable[..., FakeResponse]:
    def fake_urlopen(request: object, timeout: int = 0) -> FakeResponse:
        return response

    return fake_urlopen


def test_run_journal_adopts_the_segment_a_kill_left_beyond_the_head(tmp_path: Path) -> None:
    journal = tmp_path / "journal"
    write_journal_directory(journal)
    run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=3,
        fetcher=FakeVenue(), clock=FakeClock(), sleep=FakeSleep(),
    )
    orphan_hash = str(segment_documents(journal)[2]["content_hash"])
    rewind_chain_head(journal, segments=2)
    assert verify_journal(journal)[0] is False
    sleeper = FakeSleep()
    head = run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=1,
        fetcher=FakeVenue(), clock=FakeClock(start=RECEIVED_NS + 10**12), sleep=sleeper,
    )
    # The trailing segment was adopted, then one more round was appended.
    assert (head.segment_count, head.last_sequence) == (4, 3)
    documents = segment_documents(journal)
    assert documents[3]["previous_segment_hash"] == orphan_hash
    assert verify_journal(journal) == (True, ())
    # A resumed run waits the whole interval before its first round.
    assert sleeper.calls == [SAMPLE_INTERVAL_SECONDS]
    assert verify_journal(journal) == (True, ())


def test_run_journal_adopts_a_first_segment_whose_head_was_never_written(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "journal"
    write_journal_directory(journal)
    run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=1,
        fetcher=FakeVenue(), clock=FakeClock(), sleep=FakeSleep(),
    )
    rewind_chain_head(journal, segments=0)
    assert not (journal / "chain-head.json").exists()
    assert verify_journal(journal) == (False, ("CHAIN_HEAD_MISSING",))
    head = run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=1,
        fetcher=FakeVenue(), clock=FakeClock(start=RECEIVED_NS + 10**12), sleep=FakeSleep(),
    )
    assert (head.segment_count, head.last_sequence) == (2, 1)
    assert verify_journal(journal) == (True, ())


@pytest.mark.parametrize(
    "overrides",
    [
        {"previous_segment_hash": "a" * 64},  # not the head it follows
        {"spec_hash": "f" * 64},  # not this journal's spec
    ],
)
def test_a_trailing_segment_that_is_not_the_head_s_successor_is_refused(
    tmp_path: Path, overrides: dict[str, object]
) -> None:
    journal = tmp_path / "journal"
    write_journal_directory(journal)
    run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=2,
        fetcher=FakeVenue(), clock=FakeClock(), sleep=FakeSleep(),
    )
    rewind_chain_head(journal, segments=1)
    rewrite_segment(journal / "segments" / "0000000001.json", **overrides)
    head_before = read_document(journal / "chain-head.json")
    with pytest.raises(BinanceCostJournalSpecError):
        run_journal(
            workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=1,
            fetcher=FakeVenue(), clock=FakeClock(), sleep=FakeSleep(),
        )
    assert len(segment_documents(journal)) == 2
    # Refused, not half-repaired: a journal this run would not touch keeps its head.
    assert read_document(journal / "chain-head.json") == head_before


def test_a_trailing_segment_whose_hash_does_not_recompute_is_refused(tmp_path: Path) -> None:
    journal = tmp_path / "journal"
    write_journal_directory(journal)
    run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=2,
        fetcher=FakeVenue(), clock=FakeClock(), sleep=FakeSleep(),
    )
    rewind_chain_head(journal, segments=1)
    path = journal / "segments" / "0000000001.json"
    document = read_document(path)
    observations_of(document)[0]["spread_bps"] = "1"  # the recorded hash is now stale
    path.write_bytes(canonical_json(document))
    head_before = read_document(journal / "chain-head.json")
    with pytest.raises(BinanceCostJournalSpecError):
        run_journal(
            workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=1,
            fetcher=FakeVenue(), clock=FakeClock(), sleep=FakeSleep(),
        )
    assert read_document(journal / "chain-head.json") == head_before


def test_a_resumed_run_waits_the_interval_before_its_first_round(tmp_path: Path) -> None:
    journal = tmp_path / "journal"
    write_journal_directory(journal)
    first = FakeSleep()
    run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=1,
        fetcher=FakeVenue(), clock=FakeClock(), sleep=first,
    )
    second = FakeSleep()
    run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=2,
        fetcher=FakeVenue(), clock=FakeClock(start=RECEIVED_NS + 10**12), sleep=second,
    )
    # Nothing on disk, nothing to wait for; a journal with segments waits first.
    assert first.calls == []
    assert second.calls == pytest.approx([SAMPLE_INTERVAL_SECONDS, ROUND_PERIOD_REMAINDER])


def test_a_premium_index_failure_does_not_cost_the_perpetual_its_round(tmp_path: Path) -> None:
    journal = tmp_path / "journal"
    write_journal_directory(journal)
    head = run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=1,
        fetcher=FakeVenue(dark=("premiumIndex",)), clock=FakeClock(), sleep=FakeSleep(),
    )
    assert head.segment_count == 1
    spot_observation, perpetual_observation = observations_of(segment_documents(journal)[0])
    assert spot_observation["ok"] is True
    assert spot_observation["premium_index_reason"] is None
    assert perpetual_observation["ok"] is True
    assert perpetual_observation["reason"] is None
    # The transport's own reason, not the parser's "missing premium index".
    assert "ConnectionError" in str(perpetual_observation["premium_index_reason"])
    assert perpetual_observation["funding_rate"] is None
    assert perpetual_observation["basis_bps"] is None
    # The book itself was measured.
    assert perpetual_observation["spread_bps"] == "5000.000000"


def test_a_depth_failure_still_costs_the_instrument_its_round(tmp_path: Path) -> None:
    journal = tmp_path / "journal"
    write_journal_directory(journal)
    run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=1,
        fetcher=FakeVenue(dark=("fapi.binance.com/fapi/v1/depth",)),
        clock=FakeClock(), sleep=FakeSleep(),
    )
    perpetual_observation = observations_of(segment_documents(journal)[0])[1]
    assert perpetual_observation["ok"] is False
    assert "ConnectionError" in str(perpetual_observation["reason"])
    assert perpetual_observation["premium_index_reason"] is None


def test_verify_journal_tolerates_a_temporary_file_and_reports_anything_else(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "journal"
    write_journal_directory(journal)
    run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=1,
        fetcher=FakeVenue(), clock=FakeClock(), sleep=FakeSleep(),
    )
    segments = journal / "segments"
    # Nothing is left behind by a publish that finished.
    assert [path.name for path in segments.iterdir()] == ["0000000000.json"]
    (segments / "0000000001.json.4242.tmp").write_bytes(b"half a segment")
    assert verify_journal(journal) == (True, ())
    (segments / "0000000001.tmp").write_bytes(b"not one of ours")
    (segments / "notes.txt").write_bytes(b"nor this")
    valid, reasons = verify_journal(journal)
    assert valid is False
    assert set(reasons) == {
        "SEGMENT_UNEXPECTED_FILE:0000000001.tmp",
        "SEGMENT_UNEXPECTED_FILE:notes.txt",
    }


def test_the_public_fetcher_refuses_a_redirect_that_leaves_the_allow_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = "https://api.binance.com/api/v3/depth?symbol=BTCUSDT&limit=500"
    monkeypatch.setattr(
        urllib.request, "urlopen", urlopen_returning(FakeResponse(url="https://evil.example.com/"))
    )
    with pytest.raises(BinanceCostJournalError, match="redirect"):
        public_binance_json_fetcher(url)
    # An allow-listed host reached over plain HTTP is a redirect off HTTPS.
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        urlopen_returning(FakeResponse(url=url.replace("https://", "http://"))),
    )
    with pytest.raises(BinanceCostJournalError, match="redirect"):
        public_binance_json_fetcher(url)
    monkeypatch.setattr(
        urllib.request, "urlopen", urlopen_returning(FakeResponse(url=url, status=500))
    )
    with pytest.raises(BinanceCostJournalError, match="500"):
        public_binance_json_fetcher(url)
    monkeypatch.setattr(
        urllib.request, "urlopen", urlopen_returning(FakeResponse(url=url, body=b"[1, 2]"))
    )
    with pytest.raises(BinanceCostJournalError, match="object"):
        public_binance_json_fetcher(url)
    monkeypatch.setattr(urllib.request, "urlopen", urlopen_returning(FakeResponse(url=url)))
    assert public_binance_json_fetcher(url) == {"lastUpdateId": 1}


def test_the_cli_retries_a_crossed_reserve_and_stops_on_every_other_refusal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A reserve is about the disk right now; every other refusal is about the request."""
    journal = tmp_path / "journal"
    write_journal_directory(journal)
    assert main([
        "binance-cost-journal-run", "--workspace-root", str(tmp_path),
        "--journal", str(journal), "--rounds", "1", "--reserve-bytes", str(10**18),
    ]) == 1
    assert "StorageReserveError" in capsys.readouterr().err
    assert _journal_exit_code(StorageReserveError("job would cross the reserve")) == 1
    assert _journal_exit_code(StoragePolicyError("target uses excluded drive: E:")) == 2
    assert _journal_exit_code(BinanceCostJournalSpecError("bound to a different spec")) == 2
    assert _journal_exit_code(BinanceCostJournalError("every instrument failed")) == 1


def test_a_publish_that_dies_before_its_replace_leaves_a_pid_named_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = tmp_path / "journal"
    write_journal_directory(journal)

    def explode(self: Path, target: object) -> None:
        raise OSError("the process died between the write and the replace")

    monkeypatch.setattr(Path, "replace", explode)
    with pytest.raises(OSError, match="died"):
        run_journal(
            workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=1,
            fetcher=FakeVenue(), clock=FakeClock(), sleep=FakeSleep(),
        )
    monkeypatch.undo()
    # Named for the process that wrote it, so a second writer cannot half-write
    # the same temporary, and tolerated by the verifier.
    assert [path.name for path in (journal / "segments").iterdir()] == [
        f"0000000000.json.{os.getpid()}.tmp"
    ]
    assert verify_journal(journal) == (True, ())
    head = run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=1,
        fetcher=FakeVenue(), clock=FakeClock(), sleep=FakeSleep(),
    )
    assert head.segment_count == 1


def test_more_than_one_trailing_segment_is_corruption_not_a_kill(tmp_path: Path) -> None:
    """A kill can strand one segment; two mean something else edited the journal."""
    journal = tmp_path / "journal"
    write_journal_directory(journal)
    run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=3,
        fetcher=FakeVenue(), clock=FakeClock(), sleep=FakeSleep(),
    )
    rewind_chain_head(journal, segments=1)
    head_before = read_document(journal / "chain-head.json")
    with pytest.raises(BinanceCostJournalSpecError):
        run_journal(
            workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=1,
            fetcher=FakeVenue(), clock=FakeClock(), sleep=FakeSleep(),
        )
    assert read_document(journal / "chain-head.json") == head_before
    assert len(segment_documents(journal)) == 3


# --- Task 3: finalisation receipt ---------------------------------------------------

SPEC_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "superpowers"
    / "specs"
    / "2026-09-12-binance-cost-journal-design.md"
)
NOTIONAL_KEYS = ("500", "5000", "50000")
FLOORS_NOT_MET = ("COST_OBSERVATION_FLOOR_NOT_MET", "COST_CAPTURE_SPAN_FLOOR_NOT_MET")
EMPTY_TIER_NOTIONAL: dict[str, object] = {
    "p50_of_p50": None, "p50_of_p90": None, "contributing_count": 0, "absent_count": 0,
}


class LadderVenue:
    """A fetcher whose book is one deep level a side, widening by a round.

    Round k of an instrument quotes bid `100 - m*k` and ask `100 + m*k` around a
    mid of 100, where m is the symbol's multiplier (1 where none is declared), so
    the spread is `200*m*k` bps and the slippage at every notional the level can
    fill is exactly half of it: `100*m*k` bps. The round counter is kept per
    depth URL, so a pair's two legs widen together. `depth` sets a symbol's level
    quantity and `thin_rounds` overrides it for one round - both leave the larger
    notionals unfillable, which is what `insufficient_depth` counts.
    """

    def __init__(
        self,
        *,
        multipliers: Mapping[str, int] | None = None,
        depth: Mapping[str, str] | None = None,
        thin_rounds: Mapping[int, str] | None = None,
    ) -> None:
        self.multipliers = dict(multipliers or {})
        self.depth = dict(depth or {})
        self.thin_rounds = dict(thin_rounds or {})
        self.rounds: dict[str, int] = {}

    def __call__(self, url: str) -> Mapping[str, object]:
        symbol = parse_qs(urlsplit(url).query)["symbol"][0]
        if "premiumIndex" in url:
            return {**premium_document(), "symbol": symbol}
        round_index = self.rounds.get(url, 0) + 1
        self.rounds[url] = round_index
        width = self.multipliers.get(symbol, 1) * round_index
        quantity = self.thin_rounds.get(round_index, self.depth.get(symbol, "1000"))
        return {
            "lastUpdateId": 1_000 + round_index,
            "bids": [[str(100 - width), quantity]],
            "asks": [[str(100 + width), quantity]],
        }


def leg_document(symbol: str, *, market: str, tier: int) -> dict[str, object]:
    if market == "spot":
        return {
            "instrument_id": f"spot:{symbol}",
            "market": "spot",
            "symbol": symbol,
            "pair_symbol": symbol,
            "tier": tier,
            "depth_url": f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=500",
            "premium_index_url": None,
        }
    return {
        "instrument_id": f"perp:{symbol}",
        "market": "um",
        "symbol": symbol,
        "pair_symbol": symbol,
        "tier": tier,
        "depth_url": f"https://fapi.binance.com/fapi/v1/depth?symbol={symbol}&limit=500",
        "premium_index_url": f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}",
    }


def lower_the_eligibility_floors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Five observations over a nanosecond stand in for 10 000 over seven days."""
    monkeypatch.setattr("trading_bot.binance_cost_journal.MINIMUM_OBSERVATIONS", 5)
    monkeypatch.setattr("trading_bot.binance_cost_journal.MINIMUM_SPAN_NS", 1)


def journal_with_rounds(
    tmp_path: Path,
    *,
    rounds: int,
    fetcher: Callable[[str], Mapping[str, object]],
    **overrides: object,
) -> Path:
    journal = tmp_path / "journal"
    write_journal_directory(journal, **overrides)
    run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=rounds,
        fetcher=fetcher, clock=FakeClock(), sleep=FakeSleep(),
    )
    return journal


def finalized_receipt(tmp_path: Path, journal: Path) -> tuple[FinalizationReceipt, Path]:
    output = finalize_journal(
        workspace_root=tmp_path,
        journal_root=journal,
        output_path=tmp_path / "receipt.json",
        reserve_bytes=0,
    )
    return FinalizationReceipt.model_validate(read_document(output)), output


def statistics_of(receipt: FinalizationReceipt, instrument_id: str) -> InstrumentStatistics:
    return next(item for item in receipt.instruments if item.instrument_id == instrument_id)


def tier_of(receipt: FinalizationReceipt, *, tier: int, market: str) -> TierStatistics:
    return next(item for item in receipt.tiers if item.tier == tier and item.market == market)


def spec_declaration_sentence() -> str:
    """The rule as spec section 5 writes it, unwrapped to one line."""
    text = SPEC_PATH.read_text(encoding="utf-8")
    start = text.index("exists):**") + len("exists):**")
    return " ".join(text[start : text.index("\n\n", start)].split())


def spec_capital_declaration_sentence() -> str:
    """The rule as spec section 5.1 writes it, unwrapped to one line."""
    text = SPEC_PATH.read_text(encoding="utf-8")
    anchor = "(declared 2026-09-16, before the receipt exists):**"
    start = text.index(anchor) + len(anchor)
    return " ".join(text[start : text.index("\n\n", start)].split())


def test_the_receipt_carries_hand_computed_quantiles_and_depth_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Eight widening rounds: round 3 fills 500 only, round 6 all but 50 000.

    The slippage of round k is 100*k bps, so the samples are 100..800 at "500",
    the same without 300 at "5000" and without 300 and 600 at "50000"; the
    quantile index is ceil(n*q) - 1 over the ascending values.
    """
    lower_the_eligibility_floors(monkeypatch)
    journal = journal_with_rounds(
        tmp_path, rounds=8, fetcher=LadderVenue(thin_rounds={3: "30", 6: "100"})
    )
    receipt, _ = finalized_receipt(tmp_path, journal)

    assert receipt.version == "binance-cost-journal-receipt/1.0.0"
    assert [item.instrument_id for item in receipt.instruments] == ["spot:BTCUSDT", "perp:BTCUSDT"]
    spot = statistics_of(receipt, "spot:BTCUSDT")
    assert (spot.tier, spot.market, spot.eligible, spot.reason_codes) == (1, "spot", True, ())
    assert spot.observation_count == 8
    assert spot.spread_bps_p50 == Decimal("800")
    assert spot.basis_bps_p50 is None
    assert spot.slippage == {
        "500": {
            "count": 8, "insufficient_depth": 0,
            "p50": Decimal("400"), "p90": Decimal("800"), "p99": Decimal("800"),
        },
        "5000": {
            "count": 7, "insufficient_depth": 1,
            "p50": Decimal("500"), "p90": Decimal("800"), "p99": Decimal("800"),
        },
        "50000": {
            "count": 6, "insufficient_depth": 2,
            "p50": Decimal("400"), "p90": Decimal("800"), "p99": Decimal("800"),
        },
    }
    # A perpetual measures the same book and carries the premium index's basis.
    perpetual = statistics_of(receipt, "perp:BTCUSDT")
    assert (perpetual.tier, perpetual.market, perpetual.eligible) == (1, "um", True)
    assert perpetual.slippage == spot.slippage
    assert perpetual.basis_bps_p50 == Decimal("50")
    # Counts stay integers through the canonical form; quantiles are strings.
    document = read_document(tmp_path / "receipt.json")
    instruments = document["instruments"]
    assert isinstance(instruments, list)
    assert instruments[0]["slippage"]["500"] == {
        "count": 8, "insufficient_depth": 0,
        "p50": "400.000000", "p90": "800.000000", "p99": "800.000000",
    }


def test_the_span_is_measured_between_the_eligibility_notional_s_observations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round 1 fills 500 only, so the 5 000 USDT window opens in round 2."""
    lower_the_eligibility_floors(monkeypatch)
    journal = journal_with_rounds(tmp_path, rounds=8, fetcher=LadderVenue(thin_rounds={1: "30"}))
    receipt, _ = finalized_receipt(tmp_path, journal)
    spot = statistics_of(receipt, "spot:BTCUSDT")
    stamps = spot_stamps(journal)
    assert spot.observation_count == 8
    assert spot.slippage["5000"]["count"] == 7
    assert (spot.first_time_ns, spot.last_time_ns) == (stamps[1], stamps[7])


def test_an_instrument_that_never_reported_carries_no_statistics(tmp_path: Path) -> None:
    journal = journal_with_rounds(
        tmp_path, rounds=3, fetcher=FakeVenue(dark=("fapi.binance.com",))
    )
    receipt, _ = finalized_receipt(tmp_path, journal)
    perpetual = statistics_of(receipt, "perp:BTCUSDT")
    assert perpetual.observation_count == 0
    assert (perpetual.first_time_ns, perpetual.last_time_ns) == (None, None)
    assert perpetual.spread_bps_p50 is None
    assert perpetual.basis_bps_p50 is None
    assert perpetual.slippage["5000"] == {
        "count": 0, "insufficient_depth": 0, "p50": None, "p90": None, "p99": None,
    }
    assert perpetual.reason_codes == FLOORS_NOT_MET


def test_a_journal_short_of_the_floors_leaves_every_tier_empty(tmp_path: Path) -> None:
    """Eight rounds are neither 10 000 observations nor seven days."""
    journal = journal_with_rounds(tmp_path, rounds=8, fetcher=LadderVenue())
    receipt, _ = finalized_receipt(tmp_path, journal)
    spot = statistics_of(receipt, "spot:BTCUSDT")
    assert spot.eligible is False
    assert spot.reason_codes == FLOORS_NOT_MET
    # An ineligible instrument still carries its own statistics (spec 4).
    assert spot.spread_bps_p50 == Decimal("800")
    assert [(item.tier, item.market) for item in receipt.tiers] == [
        (1, "spot"), (1, "um"), (2, "spot"), (2, "um"),
    ]
    for tier in receipt.tiers:
        assert tier.instrument_count == 0
        assert tier.slippage == {key: EMPTY_TIER_NOTIONAL for key in NOTIONAL_KEYS}


def test_tier_statistics_take_the_median_over_eligible_instruments_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Four tier-one spot legs at 400/800/1200/1600 bps and an ineligible tier-two leg.

    The tier-two perpetual's book holds 3 000 USDT, so it never fills the
    eligibility notional: it keeps its own statistics at 500 USDT and feeds no
    tier median. Four eligible members is exactly the contribution floor, so
    the tier one spot medians are reported (W9).
    """
    lower_the_eligibility_floors(monkeypatch)
    symbols = ("AAAUSDT", "BBBUSDT", "CCCUSDT", "EEEUSDT")
    instruments = [leg_document(symbol, market="spot", tier=1) for symbol in symbols]
    instruments.append(leg_document("DDDUSDT", market="um", tier=2))
    venue = LadderVenue(
        multipliers={"BBBUSDT": 2, "CCCUSDT": 3, "EEEUSDT": 4}, depth={"DDDUSDT": "30"}
    )
    journal = journal_with_rounds(tmp_path, rounds=8, fetcher=venue, instruments=instruments)
    receipt, _ = finalized_receipt(tmp_path, journal)

    assert [item.eligible for item in receipt.instruments] == [True] * 4 + [False]
    assert [
        statistics_of(receipt, f"spot:{symbol}").slippage["5000"]["p50"] for symbol in symbols
    ] == [Decimal("400"), Decimal("800"), Decimal("1200"), Decimal("1600")]
    tier_one_spot = tier_of(receipt, tier=1, market="spot")
    assert tier_one_spot.instrument_count == 4
    # Medians of (400, 800, 1200, 1600) and of (800, 1600, 2400, 3200), same rule.
    assert tier_one_spot.slippage["5000"] == {
        "p50_of_p50": Decimal("800"), "p50_of_p90": Decimal("1600"),
        "contributing_count": 4, "absent_count": 0,
    }
    ineligible = statistics_of(receipt, "perp:DDDUSDT")
    assert ineligible.reason_codes == FLOORS_NOT_MET
    assert ineligible.slippage["500"]["count"] == 8
    assert ineligible.slippage["5000"] == {
        "count": 0, "insufficient_depth": 8, "p50": None, "p90": None, "p99": None,
    }
    for absent in (tier_of(receipt, tier=1, market="um"), tier_of(receipt, tier=2, market="um")):
        assert absent.instrument_count == 0
        assert absent.slippage["5000"] == EMPTY_TIER_NOTIONAL


def test_the_receipt_binds_the_chain_the_fees_and_its_own_hash(tmp_path: Path) -> None:
    journal = journal_with_rounds(tmp_path, rounds=4, fetcher=LadderVenue())
    receipt, output = finalized_receipt(tmp_path, journal)
    document = read_document(output)
    material = {key: value for key, value in document.items() if key != "content_hash"}
    assert document["content_hash"] == content_sha256(material)
    assert receipt.content_hash == content_sha256(material)
    head = read_document(journal / "chain-head.json")
    assert receipt.chain_head_hash == head["content_hash"]
    assert receipt.segment_count == 4
    assert receipt.segments_beyond_head == 0
    assert receipt.spec_hash == read_document(journal / "journal-spec.json")["spec_hash"]
    assert receipt.spot_fee_bps_per_side == SPOT_FEE_BPS_PER_SIDE
    assert receipt.perpetual_fee_bps_per_side == PERPETUAL_FEE_BPS_PER_SIDE
    assert receipt.fee_evidence_id == FEE_EVIDENCE_ID


def test_the_declaration_rule_is_the_spec_s_sentence_verbatim(tmp_path: Path) -> None:
    journal = journal_with_rounds(tmp_path, rounds=1, fetcher=LadderVenue())
    receipt, _ = finalized_receipt(tmp_path, journal)
    rule = spec_declaration_sentence()
    assert rule.startswith("a v3 family sets ")
    assert rule.endswith("No other reading of the receipt is admissible for a declaration.")
    assert rule == DECLARATION_RULE
    assert receipt.declaration_rule == rule
    # No other rule may be published under this version.
    with pytest.raises(ValidationError):
        FinalizationReceipt.model_validate(
            {**read_document(tmp_path / "receipt.json"), "declaration_rule": "round down"}
        )


def test_the_capital_declaration_rule_is_the_spec_s_sentence_verbatim(tmp_path: Path) -> None:
    """Spec 5.1's rung reading is bound the same way spec 5's fixed reading is."""
    journal = journal_with_rounds(tmp_path, rounds=1, fetcher=LadderVenue())
    receipt, _ = finalized_receipt(tmp_path, journal)
    rule = spec_capital_declaration_sentence()
    assert rule.startswith("a capital-declared family sets ")
    assert rule.endswith("with its declared capital.")
    assert rule == CAPITAL_DECLARATION_RULE
    assert receipt.capital_declaration_rule == rule
    # No other rule may be published under this version.
    with pytest.raises(ValidationError):
        FinalizationReceipt.model_validate(
            {
                **read_document(tmp_path / "receipt.json"),
                "capital_declaration_rule": "round down",
            }
        )


def test_a_published_receipt_is_immutable(tmp_path: Path) -> None:
    journal = journal_with_rounds(tmp_path, rounds=2, fetcher=LadderVenue())
    _, output = finalized_receipt(tmp_path, journal)
    published = output.read_bytes()
    with pytest.raises(BinanceCostJournalSpecError):
        finalize_journal(
            workspace_root=tmp_path, journal_root=journal, output_path=output, reserve_bytes=0
        )
    assert output.read_bytes() == published


def test_finalisation_refuses_a_journal_that_does_not_verify(tmp_path: Path) -> None:
    journal = journal_with_rounds(tmp_path, rounds=3, fetcher=LadderVenue())
    rewrite_segment(journal / "segments" / "0000000001.json", received_time_ns=RECEIVED_NS + 1)
    assert verify_journal(journal)[0] is False
    with pytest.raises(BinanceCostJournalSpecError, match="verification failed"):
        finalize_journal(
            workspace_root=tmp_path,
            journal_root=journal,
            output_path=tmp_path / "receipt.json",
            reserve_bytes=0,
        )
    assert not (tmp_path / "receipt.json").exists()


def test_finalisation_refuses_an_empty_journal(tmp_path: Path) -> None:
    journal = tmp_path / "journal"
    write_journal_directory(journal)
    with pytest.raises(BinanceCostJournalSpecError, match="holds no segment"):
        finalize_journal(
            workspace_root=tmp_path,
            journal_root=journal,
            output_path=tmp_path / "receipt.json",
            reserve_bytes=0,
        )
    assert not (tmp_path / "receipt.json").exists()


def test_finalisation_refuses_a_journal_that_never_sampled_the_eligibility_notional(
    tmp_path: Path,
) -> None:
    """Eligibility is declared at 5 000 USDT, so a journal without it decides nothing."""
    journal = journal_with_rounds(tmp_path, rounds=1, fetcher=LadderVenue(), notionals=["500"])
    with pytest.raises(BinanceCostJournalSpecError, match="eligibility notional"):
        finalize_journal(
            workspace_root=tmp_path,
            journal_root=journal,
            output_path=tmp_path / "receipt.json",
            reserve_bytes=0,
        )


@pytest.mark.parametrize("rounds", [1, 3])
def test_the_cli_creates_runs_and_finalises_a_journal(
    captures: Captures,
    rounds: int,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, perpetual, spot, family_spec = captures
    journal, receipt_path = root / "journal", root / "receipt.json"
    # The spec records the cadence at creation and the CLI sleeps it for real, so
    # a one-second interval keeps the multi-round case out of the declared 61 s.
    monkeypatch.setattr("trading_bot.binance_cost_journal.SAMPLE_INTERVAL_SECONDS", 1)
    assert main([
        "binance-cost-journal-create", "--workspace-root", str(root),
        "--journal", str(journal), "--run-id", "binance-carry-v1",
        "--perp-capture", str(perpetual), "--spot-capture", str(spot),
        "--family-spec", str(family_spec), "--reserve-bytes", "0", "--allow-short-sample",
    ]) == 0
    with patch("trading_bot.cli.public_binance_json_fetcher", side_effect=FakeVenue()):
        assert main([
            "binance-cost-journal-run", "--workspace-root", str(root),
            "--journal", str(journal), "--rounds", str(rounds), "--reserve-bytes", "0",
        ]) == 0
    finalize_arguments = [
        "binance-cost-journal-finalize", "--workspace-root", str(root),
        "--journal", str(journal), "--output", str(receipt_path), "--reserve-bytes", "0",
    ]
    assert main(finalize_arguments) == 0
    document = read_document(receipt_path)
    material = {key: value for key, value in document.items() if key != "content_hash"}
    assert document["content_hash"] == content_sha256(material)
    # The operator is handed the hash a declaration cites, not only the path.
    printed = capsys.readouterr().out
    assert str(receipt_path) in printed
    assert f"receipt content hash: {document['content_hash']}" in printed
    receipt = FinalizationReceipt.model_validate(document)
    assert receipt.declaration_rule == spec_declaration_sentence()
    assert receipt.capital_declaration_rule == spec_capital_declaration_sentence()
    assert receipt.segment_count == rounds
    assert receipt.segments_beyond_head == 0
    assert len(receipt.instruments) == 20
    # A few rounds are not seven days: nothing is eligible and no tier carries a
    # number, which is the receipt saying so rather than failing.
    assert all(item.reason_codes == FLOORS_NOT_MET for item in receipt.instruments)
    assert {item.instrument_count for item in receipt.tiers} == {0}
    # The receipt is immutable: a second finalisation is the supervisor's stop code.
    assert main(finalize_arguments) == 2


def test_the_cli_refuses_a_receipt_outside_the_workspace(tmp_path: Path) -> None:
    journal = tmp_path / "journal"
    write_journal_directory(journal)
    assert main([
        "binance-cost-journal-finalize", "--workspace-root", str(tmp_path),
        "--journal", str(journal), "--output", str(tmp_path.parent / "outside-receipt.json"),
        "--reserve-bytes", "0",
    ]) == 2


# --- Fix round 2: an adoption is verified before it is written ----------------------


def edit_segment_body(path: Path, **overrides: object) -> None:
    """Change a segment's body and leave its recorded hash behind, the way tampering does."""
    path.write_bytes(canonical_json({**read_document(path), **overrides}))


def test_an_adoptable_orphan_does_not_rescue_a_chain_broken_elsewhere(tmp_path: Path) -> None:
    """The rebuilt head is written only after the whole chain verifies against it.

    Three rounds, the head rewound to two - which makes segment 2 adoptable -
    and segment 0's body edited under its recorded hash: the run must refuse and
    leave `chain-head.json` exactly as it found it.
    """
    journal = tmp_path / "journal"
    write_journal_directory(journal)
    run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=3,
        fetcher=FakeVenue(), clock=FakeClock(), sleep=FakeSleep(),
    )
    rewind_chain_head(journal, segments=2)
    edit_segment_body(journal / "segments" / "0000000000.json", received_time_ns=RECEIVED_NS + 7)
    head_before = (journal / "chain-head.json").read_bytes()
    with pytest.raises(BinanceCostJournalSpecError, match="after adopting"):
        run_journal(
            workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=1,
            fetcher=FakeVenue(), clock=FakeClock(start=RECEIVED_NS + 10**12), sleep=FakeSleep(),
        )
    assert (journal / "chain-head.json").read_bytes() == head_before
    assert len(segment_documents(journal)) == 3
    assert verify_journal(journal) == (
        False,
        ("SEGMENT_HASH_MISMATCH:0000000000.json", "CHAIN_HEAD_COUNT_MISMATCH",
         "CHAIN_HEAD_SEQUENCE_MISMATCH", "CHAIN_HEAD_LINK_MISMATCH"),
    )


# --- Fix round 3: a receipt reads exactly the chain its head covers ------------------


def test_finalisation_measures_the_head_s_range_while_a_run_appends(tmp_path: Path) -> None:
    """A segment published past the head is counted in the receipt, never measured.

    Four rounds with the head rewound to three is what a concurrent
    `binance-cost-journal-run` leaves between its two publishes: a fourth valid,
    self-verifying segment the head does not cover yet.
    """
    journal = journal_with_rounds(tmp_path, rounds=4, fetcher=LadderVenue())
    rewind_chain_head(journal, segments=3)
    receipt, _ = finalized_receipt(tmp_path, journal)
    assert len(segment_documents(journal)) == 4
    assert receipt.segment_count == 3
    assert receipt.segments_beyond_head == 1
    assert receipt.chain_head_hash == read_document(journal / "chain-head.json")["content_hash"]
    # Three rounds measured per instrument, not the four that are on disk.
    assert {item.observation_count for item in receipt.instruments} == {3}
    assert statistics_of(receipt, "spot:BTCUSDT").slippage["5000"]["count"] == 3


def test_finalisation_refuses_a_segment_that_changes_under_the_measurement_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bytes the numbers come from are re-checked, not trusted from the gate.

    The tamper is timed between the verification walk and the measurement pass -
    the window a still-running writer used to slip through - and lands inside
    the head's range, so the measurement pass has to catch it itself.
    """
    journal = journal_with_rounds(tmp_path, rounds=4, fetcher=LadderVenue())
    original = journal_module._verified_chain

    def verify_then_tamper(
        journal_root: Path,
        *,
        candidate_head: ChainHead | None,
        covered_segments: int | None = None,
    ) -> tuple[bool, tuple[str, ...]]:
        outcome = original(
            journal_root, candidate_head=candidate_head, covered_segments=covered_segments
        )
        edit_segment_body(
            journal / "segments" / "0000000001.json", received_time_ns=RECEIVED_NS + 3
        )
        return outcome

    monkeypatch.setattr(journal_module, "_verified_chain", verify_then_tamper)
    with pytest.raises(BinanceCostJournalSpecError, match="changed under the measurement pass"):
        finalize_journal(
            workspace_root=tmp_path,
            journal_root=journal,
            output_path=tmp_path / "receipt.json",
            reserve_bytes=0,
        )
    assert not (tmp_path / "receipt.json").exists()


def test_finalisation_refuses_a_gap_inside_the_head_s_range(tmp_path: Path) -> None:
    """A head that counts more segments than exist names the one that is missing."""
    journal = journal_with_rounds(tmp_path, rounds=3, fetcher=LadderVenue())
    (journal / "segments" / "0000000002.json").unlink()
    with pytest.raises(BinanceCostJournalSpecError, match=r"SEGMENT_MISSING:0000000002\.json"):
        finalize_journal(
            workspace_root=tmp_path,
            journal_root=journal,
            output_path=tmp_path / "receipt.json",
            reserve_bytes=0,
        )
    assert not (tmp_path / "receipt.json").exists()


# --- Final fix wave: durability, single instance, cadence, backoff ------------------


class ScriptClock:
    """A clock reading a written-down script, repeating its last stamp when spent.

    The run loop stamps a round with its first call and reads the clock once
    more after publishing it, so a one-instrument round spends exactly three
    calls: the round's stamp, the instrument's, and the one that measures how
    long the round took.
    """

    def __init__(self, stamps: tuple[int, ...]) -> None:
        self.stamps = stamps
        self.index = 0

    def __call__(self) -> int:
        stamp = self.stamps[min(self.index, len(self.stamps) - 1)]
        self.index += 1
        return stamp


class ThrottledVenue:
    """A fetcher whose named URL fragment answers with a venue status code."""

    def __init__(self, *, throttled: str, status: int, retry_after: int | None) -> None:
        self.throttled = throttled
        self.status = status
        self.retry_after = retry_after

    def __call__(self, url: str) -> Mapping[str, object]:
        if self.throttled in url:
            raise BinanceCostJournalTransportError(self.status, self.retry_after)
        if "premiumIndex" in url:
            symbol = parse_qs(urlsplit(url).query)["symbol"][0]
            return {**premium_document(), "symbol": symbol}
        return depth_document()


def test_a_publish_flushes_its_bytes_before_the_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rename must not publish a name whose bytes are still only in a cache (W1)."""
    synced: list[int] = []
    monkeypatch.setattr(os, "fsync", lambda descriptor: synced.append(descriptor))
    journal = tmp_path / "journal"
    write_journal_directory(journal)
    run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=1,
        fetcher=FakeVenue(), clock=FakeClock(), sleep=FakeSleep(),
    )
    # One segment and one chain head, each fsynced while its descriptor was open.
    assert len(synced) == 2
    assert verify_journal(journal) == (True, ())


def test_an_unparseable_trailing_segment_is_discarded_and_its_sequence_resampled(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A crash caught mid-write leaves no measurement, so it is not corruption (W1)."""
    journal = tmp_path / "journal"
    write_journal_directory(journal)
    run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=2,
        fetcher=FakeVenue(), clock=FakeClock(), sleep=FakeSleep(),
    )
    (journal / "segments" / "0000000002.json").write_bytes(b"{")
    assert verify_journal(journal)[0] is False

    head = run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=1,
        fetcher=FakeVenue(), clock=FakeClock(start=RECEIVED_NS + 10**12), sleep=FakeSleep(),
    )

    assert (head.segment_count, head.last_sequence) == (3, 2)
    assert "SEGMENT_TORN_DISCARDED:0000000002.json" in capsys.readouterr().err
    documents = segment_documents(journal)
    assert len(documents) == 3
    assert documents[2]["previous_segment_hash"] == documents[1]["content_hash"]
    assert verify_journal(journal) == (True, ())


def test_an_unparseable_first_segment_is_discarded_before_any_head_exists(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "journal"
    write_journal_directory(journal)
    (journal / "segments" / "0000000000.json").write_bytes(b"{")

    head = run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=1,
        fetcher=FakeVenue(), clock=FakeClock(), sleep=FakeSleep(),
    )

    assert (head.segment_count, head.last_sequence) == (1, 0)
    assert segment_documents(journal)[0]["previous_segment_hash"] == ZERO_HASH
    assert verify_journal(journal) == (True, ())


def test_a_trailing_segment_that_parses_but_does_not_link_is_still_refused(
    tmp_path: Path,
) -> None:
    """Only an unparseable trailing file is a crash; a readable one is judged (W1)."""
    journal = tmp_path / "journal"
    write_journal_directory(journal)
    run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=2,
        fetcher=FakeVenue(), clock=FakeClock(), sleep=FakeSleep(),
    )
    rewind_chain_head(journal, segments=1)
    rewrite_segment(journal / "segments" / "0000000001.json", previous_segment_hash="a" * 64)
    with pytest.raises(BinanceCostJournalSpecError):
        run_journal(
            workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=1,
            fetcher=FakeVenue(), clock=FakeClock(), sleep=FakeSleep(),
        )
    assert len(segment_documents(journal)) == 2


def test_a_torn_chain_head_is_rebuilt_from_the_segments(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The head carries nothing the segments do not, so it can be recomputed (W1)."""
    journal = tmp_path / "journal"
    write_journal_directory(journal)
    run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=3,
        fetcher=FakeVenue(), clock=FakeClock(), sleep=FakeSleep(),
    )
    published = read_document(journal / "chain-head.json")
    (journal / "chain-head.json").write_bytes(b'{"version": "binance-cos')

    head = run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=1,
        fetcher=FakeVenue(), clock=FakeClock(start=RECEIVED_NS + 10**12), sleep=FakeSleep(),
    )

    reported = capsys.readouterr().err
    assert "CHAIN_HEAD_TORN:chain-head.json" in reported
    assert "CHAIN_HEAD_REBUILT:3" in reported
    # The rebuild put the head back where it was before the run grew it by one.
    assert (head.segment_count, head.last_sequence) == (4, 3)
    assert verify_journal(journal) == (True, ())
    assert segment_documents(journal)[3]["previous_segment_hash"] == (
        published["final_segment_hash"]
    )


def test_a_missing_chain_head_over_three_segments_is_rebuilt(tmp_path: Path) -> None:
    journal = tmp_path / "journal"
    write_journal_directory(journal)
    run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=3,
        fetcher=FakeVenue(), clock=FakeClock(), sleep=FakeSleep(),
    )
    published = read_document(journal / "chain-head.json")
    (journal / "chain-head.json").unlink()
    # Three segments past no head at all is more than `_orphan_segment` adopts.
    assert verify_journal(journal) == (False, ("CHAIN_HEAD_MISSING",))

    head = run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=1,
        fetcher=FakeVenue(), clock=FakeClock(start=RECEIVED_NS + 10**12), sleep=FakeSleep(),
    )

    assert (head.segment_count, head.last_sequence) == (4, 3)
    assert segment_documents(journal)[3]["previous_segment_hash"] == (
        published["final_segment_hash"]
    )
    assert verify_journal(journal) == (True, ())


def test_a_missing_chain_head_over_a_broken_middle_segment_refuses(tmp_path: Path) -> None:
    """A chain that does not verify from zero is corruption, not a crash (W1)."""
    journal = tmp_path / "journal"
    write_journal_directory(journal)
    run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=3,
        fetcher=FakeVenue(), clock=FakeClock(), sleep=FakeSleep(),
    )
    (journal / "chain-head.json").unlink()
    edit_segment_body(journal / "segments" / "0000000001.json", received_time_ns=RECEIVED_NS + 7)

    with pytest.raises(BinanceCostJournalSpecError, match="cannot be rebuilt"):
        run_journal(
            workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=1,
            fetcher=FakeVenue(), clock=FakeClock(), sleep=FakeSleep(),
        )

    assert not (journal / "chain-head.json").exists()
    assert len(segment_documents(journal)) == 3


def test_a_chain_head_that_parses_but_fails_its_hash_is_not_rebuilt(tmp_path: Path) -> None:
    """A head that was changed, not torn, is tampering and stays fatal (W1)."""
    journal = tmp_path / "journal"
    write_journal_directory(journal)
    run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=2,
        fetcher=FakeVenue(), clock=FakeClock(), sleep=FakeSleep(),
    )
    path = journal / "chain-head.json"
    path.write_bytes(canonical_json({**read_document(path), "last_sequence": 7}))
    with pytest.raises(BinanceCostJournalSpecError, match="recorded hash"):
        run_journal(
            workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=1,
            fetcher=FakeVenue(), clock=FakeClock(), sleep=FakeSleep(),
        )


def test_a_second_writer_is_refused_while_the_lock_is_held(tmp_path: Path) -> None:
    """One journal is written by one process, whatever the supervisor started (W2)."""
    journal = tmp_path / "journal"
    write_journal_directory(journal)
    descriptor = os.open(journal / "run.lock", os.O_RDWR | os.O_CREAT)
    journal_module._lock_exclusive(descriptor)
    try:
        with pytest.raises(BinanceCostJournalSpecError, match="another process"):
            run_journal(
                workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=1,
                fetcher=FakeVenue(), clock=FakeClock(), sleep=FakeSleep(),
            )
        assert list((journal / "segments").iterdir()) == []
    finally:
        journal_module._release_journal_lock(descriptor)

    head = run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=1,
        fetcher=FakeVenue(), clock=FakeClock(), sleep=FakeSleep(),
    )
    assert head.segment_count == 1


def test_create_journal_refuses_a_sample_short_of_the_declared_legs(
    captures: Captures, capsys: pytest.CaptureFixture[str]
) -> None:
    """Twelve pairs derive 20 of the 32 declared legs: a decision, not a default (W3)."""
    root, perpetual, spot, family_spec = captures
    journal = root / "journal"
    assert SAMPLE_INSTRUMENT_COUNT == 32
    with pytest.raises(BinanceCostJournalSpecError, match="20 of the 32"):
        create_journal(
            workspace_root=root, journal_root=journal, reserve_bytes=0,
            run_id="binance-carry-v1", perp_capture_root=perpetual, spot_capture_root=spot,
            family_spec_path=family_spec,
        )
    assert not (journal / "journal-spec.json").exists()
    arguments = [
        "binance-cost-journal-create", "--workspace-root", str(root),
        "--journal", str(journal), "--run-id", "binance-carry-v1",
        "--perp-capture", str(perpetual), "--spot-capture", str(spot),
        "--family-spec", str(family_spec), "--reserve-bytes", "0",
    ]
    assert main(arguments) == 2
    assert "20 of the 32" in capsys.readouterr().err
    assert main([*arguments, "--allow-short-sample"]) == 0
    reported = capsys.readouterr().out
    assert "20 instruments" in reported
    assert "2020-07-24" in reported


def test_the_declared_interval_is_a_period_not_a_gap(tmp_path: Path) -> None:
    """A round that took 40 s of the 61 s period leaves 21 s; one that took 70 s, none."""
    journal = tmp_path / "journal"
    write_journal_directory(journal, instruments=[instrument_document()])
    second = RECEIVED_NS + 100 * 10**9
    third = second + 200 * 10**9
    clock = ScriptClock((
        RECEIVED_NS, RECEIVED_NS, RECEIVED_NS + 40 * 10**9,
        second, second, second + 70 * 10**9,
        third, third, third,
    ))
    sleeper = FakeSleep()

    run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=3,
        fetcher=FakeVenue(), clock=clock, sleep=sleeper,
    )

    assert sleeper.calls == pytest.approx([21.0, 0.0])
    assert verify_journal(journal) == (True, ())


def test_a_throttled_round_waits_the_retry_after_on_top_of_the_period(tmp_path: Path) -> None:
    """429 and 418 are the venue naming its own cadence, and the run obeys it (W7)."""
    journal = tmp_path / "journal"
    write_journal_directory(journal)
    venue = ThrottledVenue(throttled="https://api.binance.com", status=429, retry_after=12)
    sleeper = FakeSleep()

    run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=2,
        fetcher=venue, clock=FakeClock(), sleep=sleeper,
    )

    assert sleeper.calls == pytest.approx([ROUND_PERIOD_REMAINDER, 12.0])
    spot_observation, perpetual_observation = observations_of(segment_documents(journal)[0])
    assert spot_observation["ok"] is False
    assert "BinanceCostJournalTransportError: public request returned 429" in str(
        spot_observation["reason"]
    )
    # The throttled instrument costs its own round, never the other's.
    assert perpetual_observation["ok"] is True


@pytest.mark.parametrize(
    ("status", "retry_after", "expected"),
    [(429, None, 60.0), (418, 1_000, 300.0), (429, 5, 5.0), (503, 30, None)],
)
def test_the_backoff_is_defaulted_and_capped(
    tmp_path: Path, status: int, retry_after: int | None, expected: float | None
) -> None:
    journal = tmp_path / "journal"
    write_journal_directory(journal)
    sleeper = FakeSleep()
    run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=2,
        fetcher=ThrottledVenue(
            throttled="https://api.binance.com", status=status, retry_after=retry_after
        ),
        clock=FakeClock(), sleep=sleeper,
    )
    # A status that is not a throttle buys no wait at all: only the period.
    assert sleeper.calls == pytest.approx(
        [ROUND_PERIOD_REMAINDER] if expected is None else [ROUND_PERIOD_REMAINDER, expected]
    )


def test_the_public_fetcher_carries_the_status_and_the_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = "https://api.binance.com/api/v3/depth?symbol=BTCUSDT&limit=500"
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        urlopen_returning(FakeResponse(url=url, status=429, headers={"Retry-After": "17"})),
    )
    with pytest.raises(BinanceCostJournalTransportError) as refused:
        public_binance_json_fetcher(url)
    assert (refused.value.status, refused.value.retry_after) == (429, 17)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        urlopen_returning(FakeResponse(url=url, status=418, headers={"Retry-After": "soon"})),
    )
    with pytest.raises(BinanceCostJournalTransportError) as unparseable:
        public_binance_json_fetcher(url)
    # An HTTP-date `Retry-After` is not parsed: absent, and the run waits its own minute.
    assert (unparseable.value.status, unparseable.value.retry_after) == (418, None)


def test_the_public_fetcher_reads_the_status_urllib_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """urllib raises 4xx and 5xx rather than returning them, so the code is on the error."""
    url = "https://api.binance.com/api/v3/depth?symbol=BTCUSDT&limit=500"
    headers = Message()
    headers["Retry-After"] = "23"

    def raise_http_error(request: object, timeout: int = 0) -> object:
        raise urllib.error.HTTPError(url, 429, "Too Many Requests", headers, None)

    monkeypatch.setattr(urllib.request, "urlopen", raise_http_error)
    with pytest.raises(BinanceCostJournalTransportError) as refused:
        public_binance_json_fetcher(url)
    assert (refused.value.status, refused.value.retry_after) == (429, 23)


def test_an_observation_counts_the_levels_it_walked(tmp_path: Path) -> None:
    """`null` slippage says the depth could not fill it; this says how deep it was (W8)."""
    observation = depth_observation(
        depth_document(),
        instrument=SPOT,
        received_time_ns=RECEIVED_NS,
        notionals=NOTIONALS,
        premium_index=None,
    )
    assert (observation.bid_levels, observation.ask_levels) == (3, 3)
    failed = depth_observation(
        {"lastUpdateId": 1, "bids": [], "asks": []},
        instrument=SPOT,
        received_time_ns=RECEIVED_NS,
        notionals=NOTIONALS,
        premium_index=None,
    )
    assert (failed.bid_levels, failed.ask_levels) == (None, None)
    with pytest.raises(ValidationError):
        # A measured observation that counted no level is not a measurement.
        InstrumentObservation.model_validate({**observation_document(), "bid_levels": None})
    with pytest.raises(ValidationError):
        InstrumentObservation.model_validate({**observation_document(), "ask_levels": 0})

    journal = tmp_path / "journal"
    write_journal_directory(journal)
    run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=1,
        fetcher=FakeVenue(dark=("https://api.binance.com",)),
        clock=FakeClock(), sleep=FakeSleep(),
    )
    spot_observation, perpetual_observation = observations_of(segment_documents(journal)[0])
    assert (spot_observation["bid_levels"], spot_observation["ask_levels"]) == (None, None)
    assert (perpetual_observation["bid_levels"], perpetual_observation["ask_levels"]) == (3, 3)


# --- Final fix wave: the receipt's floors, its span and the status subcommand -------


def tier_ladder_instruments(*, deep: int, total: int = 8) -> list[dict[str, object]]:
    """``total`` tier-one spot legs of which ``deep`` display 50 000 USDT.

    Every leg fills the 5 000 USDT eligibility notional, so all of them are
    eligible; only the deep ones carry a number at 50 000.
    """
    return [
        leg_document(f"T{index:02d}USDT", market="spot", tier=1) for index in range(total)
    ]


def tier_ladder_depth(*, deep: int, total: int = 8) -> dict[str, str]:
    """A level quantity a symbol: 1 000 fills every notional, 100 stops at 5 000."""
    return {
        f"T{index:02d}USDT": "1000" if index < deep else "100" for index in range(total)
    }


def tier_ladder_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, deep: int
) -> FinalizationReceipt:
    lower_the_eligibility_floors(monkeypatch)
    journal = journal_with_rounds(
        tmp_path,
        rounds=8,
        fetcher=LadderVenue(depth=tier_ladder_depth(deep=deep)),
        instruments=tier_ladder_instruments(deep=deep),
    )
    receipt, _ = finalized_receipt(tmp_path, journal)
    return receipt


def test_a_tier_median_needs_four_contributing_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Eight eligible members, three of them deep enough for 50 000 (W9)."""
    receipt = tier_ladder_receipt(tmp_path, monkeypatch, deep=3)
    assert all(item.eligible for item in receipt.instruments)
    tier = tier_of(receipt, tier=1, market="spot")
    assert tier.instrument_count == 8
    # Three of eight filled 50 000: the counts are reported, the medians are not.
    assert tier.slippage["50000"] == {
        "p50_of_p50": None, "p50_of_p90": None,
        "contributing_count": 3, "absent_count": 5,
    }
    # Every member filled 5 000, so that notional carries its medians.
    assert tier.slippage["5000"]["contributing_count"] == 8
    assert tier.slippage["5000"]["absent_count"] == 0
    assert tier.slippage["5000"]["p50_of_p50"] == Decimal("400")
    assert tier.slippage["5000"]["p50_of_p90"] == Decimal("800")


def test_the_fourth_contributing_member_unlocks_the_tier_median(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert TIER_MINIMUM_CONTRIBUTORS == 4
    receipt = tier_ladder_receipt(tmp_path, monkeypatch, deep=4)
    tier = tier_of(receipt, tier=1, market="spot")
    assert tier.slippage["50000"] == {
        "p50_of_p50": Decimal("400"), "p50_of_p90": Decimal("800"),
        "contributing_count": 4, "absent_count": 4,
    }


def test_a_tier_median_below_the_floor_cannot_be_published(tmp_path: Path) -> None:
    """The model refuses the number, not only the function that would compute it."""
    journal = journal_with_rounds(tmp_path, rounds=2, fetcher=LadderVenue())
    _, output = finalized_receipt(tmp_path, journal)
    document = read_document(output)
    tiers = document["tiers"]
    assert isinstance(tiers, list)
    forged = {
        **tiers[0],
        "instrument_count": 3,
        "slippage": {
            key: {
                "p50_of_p50": "400.000000", "p50_of_p90": "800.000000",
                "contributing_count": 3, "absent_count": 0,
            }
            for key in NOTIONAL_KEYS
        },
    }
    with pytest.raises(ValidationError, match="contributing members"):
        TierStatistics.model_validate(forged)


def test_the_receipt_states_the_floors_the_cadence_and_the_notionals(tmp_path: Path) -> None:
    """A reader recomputes every decision from the receipt alone (W10)."""
    journal = journal_with_rounds(tmp_path, rounds=2, fetcher=LadderVenue())
    receipt, _ = finalized_receipt(tmp_path, journal)
    assert receipt.eligibility == {
        "minimum_observations": MINIMUM_OBSERVATIONS,
        "minimum_span_ns": MINIMUM_SPAN_NS,
        "eligibility_notional": ELIGIBILITY_NOTIONAL,
        "tier_minimum_contributors": TIER_MINIMUM_CONTRIBUTORS,
    }
    assert receipt.target_rounds == TARGET_ROUNDS
    assert receipt.notionals == NOTIONALS
    assert receipt.sample_interval_seconds == SAMPLE_INTERVAL_SECONDS
    # The published floors are integers and the notional is the statistics' key.
    document = read_document(tmp_path / "receipt.json")
    assert document["eligibility"] == {
        "minimum_observations": 10_000,
        "minimum_span_ns": 604_800_000_000_000,
        "eligibility_notional": "5000",
        "tier_minimum_contributors": 4,
    }
    assert document["notionals"] == ["500", "5000", "50000"]
    with pytest.raises(ValidationError):
        FinalizationReceipt.model_validate({**document, "eligibility": {"minimum_span_ns": 1}})
    with pytest.raises(ValidationError):
        # A floor the receipt names must be one of the notionals it read.
        FinalizationReceipt.model_validate(
            {**document, "eligibility": {**document["eligibility"], "eligibility_notional": "7"}}
        )


def test_the_span_survives_a_clock_that_stepped_backwards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A host that rewound its clock narrows the window; it does not void the receipt."""
    lower_the_eligibility_floors(monkeypatch)
    journal = tmp_path / "journal"
    write_journal_directory(journal)
    run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=8,
        fetcher=LadderVenue(), clock=FakeClock(step=-1_000_000), sleep=FakeSleep(),
    )
    receipt, _ = finalized_receipt(tmp_path, journal)

    spot = statistics_of(receipt, "spot:BTCUSDT")
    stamps = spot_stamps(journal)
    assert stamps == sorted(stamps, reverse=True)
    assert spot.span_observation_count == 8
    assert (spot.first_time_ns, spot.last_time_ns) == (min(stamps), max(stamps))
    assert spot.eligible is True


def test_the_span_observation_count_names_what_the_floors_were_read_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lower_the_eligibility_floors(monkeypatch)
    journal = journal_with_rounds(tmp_path, rounds=8, fetcher=LadderVenue(thin_rounds={1: "30"}))
    receipt, _ = finalized_receipt(tmp_path, journal)
    spot = statistics_of(receipt, "spot:BTCUSDT")
    # Round one filled 500 only, so the eligibility window holds seven of eight.
    assert (spot.observation_count, spot.span_observation_count) == (8, 7)
    assert spot.slippage["5000"]["count"] == 7
    with pytest.raises(ValidationError):
        InstrumentStatistics.model_validate(
            {**spot.model_dump(mode="json"), "span_observation_count": 9}
        )


def test_a_receipt_that_does_not_validate_is_a_refusal_not_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit 2 for the supervisor, never a pydantic error out of a library call (W10)."""
    journal = journal_with_rounds(tmp_path, rounds=2, fetcher=LadderVenue())
    monkeypatch.setattr("trading_bot.binance_cost_journal.RECEIPT_VERSION", "receipt/9.9.9")
    with pytest.raises(BinanceCostJournalSpecError, match="receipt is invalid"):
        finalize_journal(
            workspace_root=tmp_path,
            journal_root=journal,
            output_path=tmp_path / "receipt.json",
            reserve_bytes=0,
        )
    assert not (tmp_path / "receipt.json").exists()


def test_journal_status_reports_the_tip_the_cadence_and_the_rates(tmp_path: Path) -> None:
    """The window is read, not the chain: the answer costs the same on day eight (W4)."""
    journal = tmp_path / "journal"
    write_journal_directory(journal)
    run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=4,
        fetcher=FakeVenue(dark=("premiumIndex",)), clock=FakeClock(step=61_000_000_000),
        sleep=FakeSleep(),
    )

    status = journal_status(journal, last=60)

    assert (status.segment_count, status.last_sequence, status.window_segments) == (4, 3, 4)
    assert status.last_received_time_ns == segment_documents(journal)[3]["received_time_ns"]
    # The fake clock steps 61 s a call and a two-instrument round spends four.
    assert status.mean_period_seconds == Decimal("244.000")
    assert [item.instrument_id for item in status.instruments] == [
        "spot:BTCUSDT", "perp:BTCUSDT",
    ]
    spot, perpetual = status.instruments
    assert (spot.observation_count, spot.ok_rate) == (4, Decimal("1.0000"))
    assert spot.premium_index_reason_rate == Decimal("0.0000")
    # Every perpetual round measured its book and lost its premium index.
    assert (perpetual.ok_rate, perpetual.premium_index_reason_rate) == (
        Decimal("1.0000"), Decimal("1.0000"),
    )
    assert (status.verified, status.reasons) == (True, ())


def test_journal_status_reads_only_the_window_it_was_asked_for(tmp_path: Path) -> None:
    journal = tmp_path / "journal"
    write_journal_directory(journal)
    run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=5,
        fetcher=FakeVenue(dark=("https://api.binance.com",)), clock=FakeClock(),
        sleep=FakeSleep(),
    )
    # A segment before the window is broken: the bounded read does not see it,
    # and `verify_journal` still does.
    edit_segment_body(journal / "segments" / "0000000000.json", received_time_ns=RECEIVED_NS + 5)

    status = journal_status(journal, last=2)

    assert (status.segment_count, status.window_segments) == (5, 2)
    assert status.verified is True
    spot, _ = status.instruments
    assert (spot.observation_count, spot.ok_rate) == (2, Decimal("0.0000"))
    assert verify_journal(journal)[0] is False
    assert journal_status(journal, last=60).verified is False


def test_journal_status_names_what_is_wrong_inside_its_window(tmp_path: Path) -> None:
    journal = tmp_path / "journal"
    write_journal_directory(journal)
    run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=3,
        fetcher=FakeVenue(), clock=FakeClock(), sleep=FakeSleep(),
    )
    path = journal / "segments" / "0000000002.json"
    edit_segment_body(path, received_time_ns=RECEIVED_NS + 5)

    body_changed = journal_status(journal, last=2)

    assert body_changed.verified is False
    assert body_changed.reasons == ("SEGMENT_HASH_MISMATCH:0000000002.json",)
    # A tamper that recomputes the segment's own hash still breaks the head's link.
    rewrite_segment(path, received_time_ns=RECEIVED_NS + 6)

    rehashed = journal_status(journal, last=2)

    assert rehashed.verified is False
    assert rehashed.reasons == ("CHAIN_HEAD_LINK_MISMATCH",)


def test_journal_status_refuses_an_empty_or_unreadable_journal(tmp_path: Path) -> None:
    journal = tmp_path / "journal"
    write_journal_directory(journal)
    with pytest.raises(BinanceCostJournalSpecError, match="no segment"):
        journal_status(journal, last=60)
    with pytest.raises(BinanceCostJournalSpecError, match="at least one segment"):
        journal_status(journal, last=0)


def test_the_cli_reports_a_journal_s_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    journal = tmp_path / "journal"
    write_journal_directory(journal)
    run_journal(
        workspace_root=tmp_path, journal_root=journal, reserve_bytes=0, rounds=3,
        fetcher=FakeVenue(), clock=FakeClock(step=61_000_000_000), sleep=FakeSleep(),
    )
    arguments = [
        "binance-cost-journal-status", "--workspace-root", str(tmp_path),
        "--journal", str(journal), "--last", "2",
    ]

    assert main(arguments) == 0

    reported = capsys.readouterr().out
    assert "segment_count: 3" in reported
    assert "last_sequence: 2" in reported
    assert "window_segments: 2 (last 2)" in reported
    assert "mean_period_seconds: 244.000" in reported
    assert "spot:BTCUSDT: ok 1.0000 premium_index_reason 0.0000" in reported
    assert "verify: ok" in reported
    # The stamp is printed as an ISO UTC time, not a nanosecond count.
    last = segment_documents(journal)[2]["received_time_ns"]
    assert isinstance(last, int)
    assert str(last) not in reported
    assert f"last_received_time: {iso_utc_time(last)}" in reported

    # A broken window is exit 1 - something to look at, not something to stop for.
    edit_segment_body(journal / "segments" / "0000000002.json", received_time_ns=RECEIVED_NS + 5)
    assert main(arguments) == 1
    assert "verify: SEGMENT_HASH_MISMATCH:0000000002.json" in capsys.readouterr().out
    # A journal outside the workspace is the supervisor's stop code, as everywhere.
    assert main([
        "binance-cost-journal-status", "--workspace-root", str(tmp_path),
        "--journal", str(tmp_path.parent / "outside-journal"),
    ]) == 2
