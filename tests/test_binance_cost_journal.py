from decimal import Decimal

import pytest
from pydantic import ValidationError

from trading_bot.binance_cost_journal import (
    ALLOWED_HOSTS,
    DEPTH_LIMIT,
    JOURNAL_VERSION,
    MINIMUM_OBSERVATIONS,
    MINIMUM_SPAN_NS,
    NOTIONALS,
    SAMPLE_INTERVAL_SECONDS,
    TARGET_ROUNDS,
    ZERO_HASH,
    BinanceCostJournalSpec,
    ChainHead,
    InstrumentObservation,
    JournalInstrument,
    JournalSegment,
    depth_observation,
    walk_notional,
)

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
