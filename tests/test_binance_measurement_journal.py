import gc
import json
import time
import urllib.request
import weakref
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import ClassVar
from urllib.parse import parse_qs, urlsplit

import pytest

from tests.test_binance_cost_journal import (
    RECEIVED_NS,
    ROUND_PERIOD_REMAINDER,
    FakeClock,
    FakeResponse,
    FakeSleep,
    FakeVenue,
    instrument_document,
    perpetual_instrument_document,
    read_document,
    rewrite_segment,
    urlopen_returning,
    write_journal_directory,
)
from trading_bot import binance_cost_journal, binance_measurement_journal
from trading_bot.binance_cost_journal import (
    SAMPLE_INTERVAL_SECONDS,
    TIER_MINIMUM_CONTRIBUTORS,
    ZERO_HASH,
    BinanceCostJournalError,
    BinanceCostJournalTransportError,
    InstrumentObservation,
    _instrument_statistics,
    public_binance_json_array_fetcher,
    public_binance_json_fetcher,
)
from trading_bot.binance_measurement_journal import (
    MEASUREMENT_JOURNAL_VERSION,
    MEASUREMENT_SNAPSHOT_VERSION,
    PERP_BOOK_TICKER_URL,
    PREMIUM_INDEX_URL,
    SNAPSHOT_ENDPOINTS,
    SNAPSHOT_EVERY_ROUNDS,
    SPOT_BOOK_TICKER_URL,
    BinanceMeasurementJournalSpecError,
    MeasurementChainHead,
    MeasurementSegment,
    _ReadSegment,
    _segment_paths,
    _tier_floor_count,
    _verified_chain,
    _verified_tail,
    book_rows,
    create_measurement_journal,
    load_measurement_journal_spec,
    measurement_status,
    premium_rows,
    run_measurement_journal,
    snapshot_measurement_journal,
    verify_measurement_journal,
)
from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.depth_adapters import DepthPayloadError
from trading_bot.storage import StoragePolicy, StorageReserveError


def premium_payload() -> list[object]:
    """Three USDT perpetuals and one the journal does not keep."""
    return [
        {
            "symbol": "BTCUSDT",
            "markPrice": "100.5",
            "indexPrice": "100",
            "lastFundingRate": "0.0001",
            "nextFundingTime": 1_757_001_600_000,
        },
        {
            "symbol": "ETHUSDT",
            "markPrice": "2000.4",
            "indexPrice": "2000.0",
            "lastFundingRate": "-0.00005",
            "nextFundingTime": 1_757_001_600_000,
        },
        {
            "symbol": "XRPUSDT",
            "markPrice": "9.9997",
            "indexPrice": "10.0000",
            "lastFundingRate": "0",
            "nextFundingTime": 1_757_001_600_000,
        },
        {
            "symbol": "BTCUSDC",
            "markPrice": "100.5",
            "indexPrice": "100",
            "lastFundingRate": "0.0001",
            "nextFundingTime": 1_757_001_600_000,
        },
    ]


def perp_book_payload() -> list[object]:
    return [
        {
            "symbol": "BTCUSDT",
            "bidPrice": "99.99",
            "bidQty": "3",
            "askPrice": "100.01",
            "askQty": "4",
        },
        {
            "symbol": "ETHUSDT",
            "bidPrice": "1999",
            "bidQty": "1.5",
            "askPrice": "2001",
            "askQty": "2.5",
        },
        {
            "symbol": "BTCUSDC",
            "bidPrice": "99.99",
            "bidQty": "3",
            "askPrice": "100.01",
            "askQty": "4",
        },
    ]


def spot_book_payload() -> list[object]:
    return [
        {
            "symbol": "BTCUSDT",
            "bidPrice": "99.95",
            "bidQty": "2",
            "askPrice": "100.05",
            "askQty": "2",
        },
        {
            "symbol": "DOGEUSDT",
            "bidPrice": "3",
            "bidQty": "10",
            "askPrice": "4",
            "askQty": "20",
        },
        {
            "symbol": "ETHBTC",
            "bidPrice": "0.05",
            "bidQty": "10",
            "askPrice": "0.051",
            "askQty": "10",
        },
    ]


def test_the_measurement_endpoints_are_the_three_the_spec_names() -> None:
    assert PREMIUM_INDEX_URL == "https://fapi.binance.com/fapi/v1/premiumIndex"
    assert PERP_BOOK_TICKER_URL == "https://fapi.binance.com/fapi/v1/ticker/bookTicker"
    assert SPOT_BOOK_TICKER_URL == "https://api.binance.com/api/v3/ticker/bookTicker"
    assert MEASUREMENT_JOURNAL_VERSION == "binance-measurement-journal/1.0.0"
    assert SNAPSHOT_EVERY_ROUNDS == 5


def test_premium_rows_keep_the_usdt_perpetuals_and_quantise_the_basis_by_hand() -> None:
    rows, excluded = premium_rows(premium_payload())
    assert [row.symbol for row in rows] == ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
    assert excluded == 0
    # (100.5 - 100) / 100 * 10 000 = 50 bps.
    assert rows[0].basis_bps == Decimal("50.000000")
    assert rows[0].mark_price == Decimal("100.5")
    assert rows[0].index_price == Decimal("100")
    assert rows[0].last_funding_rate == Decimal("0.0001")
    assert rows[0].next_funding_time_ms == 1_757_001_600_000
    # (2000.4 - 2000.0) / 2000.0 * 10 000 = 2 bps; a negative funding rate is kept.
    assert rows[1].basis_bps == Decimal("2.000000")
    assert rows[1].last_funding_rate == Decimal("-0.00005")
    # A mark below the index is a negative basis: -0.0003 / 10 * 10 000 = -0.3 bps.
    assert rows[2].basis_bps == Decimal("-0.300000")


def test_book_rows_keep_the_usdt_pairs_and_quantise_the_spread_by_hand() -> None:
    perpetual, excluded = book_rows(perp_book_payload())
    assert [row.symbol for row in perpetual] == ["BTCUSDT", "ETHUSDT"]
    assert excluded == 0
    # mid 100, spread 0.02 -> 2 bps.
    assert perpetual[0].spread_bps == Decimal("2.000000")
    assert (perpetual[0].bid_price, perpetual[0].bid_qty) == (Decimal("99.99"), Decimal("3"))
    assert (perpetual[0].ask_price, perpetual[0].ask_qty) == (Decimal("100.01"), Decimal("4"))
    # mid 2000, spread 2 -> 10 bps.
    assert perpetual[1].spread_bps == Decimal("10.000000")
    spot, spot_excluded = book_rows(spot_book_payload())
    assert (spot_excluded, [row.symbol for row in spot]) == (0, ["BTCUSDT", "DOGEUSDT"])
    # mid 100, spread 0.1 -> 10 bps.
    assert spot[0].spread_bps == Decimal("10.000000")
    # mid 3.5, spread 1 -> 10 000 / 3.5 = 2857.142857142857... at 1e-6, half even.
    assert spot[1].spread_bps == Decimal("2857.142857")


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"markPrice": 100.5}, "malformed entry ETHUSDT"),
        ({"indexPrice": None}, "malformed entry ETHUSDT"),
        ({"lastFundingRate": 0.0001}, "malformed entry ETHUSDT"),
        ({"nextFundingTime": "1757001600000"}, "malformed entry ETHUSDT"),
        ({"symbol": 17}, "malformed entry at index 1"),
        ({"symbol": "BTCUSDT"}, "duplicate entry BTCUSDT"),
    ],
)
def test_one_bad_premium_entry_refuses_the_whole_payload(
    overrides: dict[str, object], reason: str
) -> None:
    """Fail closed: a payload the journal cannot read whole is not read at all."""
    payload = premium_payload()
    entry = payload[1]
    assert isinstance(entry, dict)
    payload[1] = {**entry, **overrides}
    with pytest.raises(DepthPayloadError, match=reason):
        premium_rows(payload)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"bidPrice": 1999.0}, "malformed entry ETHUSDT"),
        ({"askQty": ["2.5"]}, "malformed entry ETHUSDT"),
        ({"askQty": None}, "malformed entry ETHUSDT"),
    ],
)
def test_one_bad_book_entry_refuses_the_whole_payload(
    overrides: dict[str, object], reason: str
) -> None:
    payload = perp_book_payload()
    entry = payload[1]
    assert isinstance(entry, dict)
    payload[1] = {**entry, **overrides}
    with pytest.raises(DepthPayloadError, match=reason):
        book_rows(payload)


@pytest.mark.parametrize("symbol", ["ETH-USDT", "\u5e01\u5b89\u4eba\u751fUSDT"])
def test_a_usdt_symbol_the_models_cannot_name_is_excluded_not_a_failure(symbol: str) -> None:
    """Ruling 25, seen live on 2026-09-23: Binance lists a few USDT pairs with CJK
    names, and one of them must not cost the whole venue's snapshot. The entry
    is counted as excluded on every endpoint; the other symbols stay."""
    premium = premium_payload()
    entry = premium[1]
    assert isinstance(entry, dict)
    premium[1] = {**entry, "symbol": symbol}
    rows, excluded = premium_rows(premium)
    assert (excluded, [row.symbol for row in rows]) == (1, ["BTCUSDT", "XRPUSDT"])
    book = perp_book_payload()
    entry = book[1]
    assert isinstance(entry, dict)
    book[1] = {**entry, "symbol": symbol}
    rows_b, excluded_b = book_rows(book)
    assert (excluded_b, [row.symbol for row in rows_b]) == (1, ["BTCUSDT"])


def test_a_payload_without_one_usdt_symbol_is_refused() -> None:
    with pytest.raises(DepthPayloadError, match="no USDT symbols"):
        premium_rows([])
    with pytest.raises(DepthPayloadError, match="no USDT symbols"):
        book_rows([entry for entry in spot_book_payload() if "USDT" not in str(entry)])
    with pytest.raises(DepthPayloadError, match="malformed entry at index 0"):
        book_rows(["BTCUSDT"])


def test_create_measurement_journal_copies_the_cost_journal_s_sample_and_seals_a_spec(
    tmp_path: Path,
) -> None:
    cost_journal = tmp_path / "cost-journal"
    cost_spec_hash = write_journal_directory(cost_journal)
    measurement = tmp_path / "measurement"
    path = create_measurement_journal(
        workspace_root=tmp_path,
        journal_root=measurement,
        reserve_bytes=0,
        run_id="binance-measurement-v1",
        cost_journal_root=cost_journal,
    )
    assert path == measurement / "journal-spec.json"
    document = read_document(path)
    assert document.pop("spec_hash") == content_sha256(document)
    assert document["version"] == MEASUREMENT_JOURNAL_VERSION
    assert document["run_id"] == "binance-measurement-v1"
    created = document["created_time_ns"]
    assert isinstance(created, int) and created > 0
    # The v1 spec this stream continues, and the sample it takes over.
    assert document["cost_journal_spec_hash"] == cost_spec_hash
    assert document["instruments"] == [instrument_document(), perpetual_instrument_document()]
    assert document["notionals"] == ["500", "5000", "50000"]
    assert document["depth_limit"] == 500
    assert document["sample_interval_seconds"] == 61
    assert document["snapshot_every_rounds"] == SNAPSHOT_EVERY_ROUNDS
    assert document["spot_fee_bps_per_side"] == "10"
    assert document["perpetual_fee_bps_per_side"] == "5"
    assert document["fee_evidence_id"] == "BINANCE:fee-schedule:2026-09-11:standard-taker"
    assert list((measurement / "segments").iterdir()) == []


def test_a_measurement_journal_is_created_once_and_from_a_readable_cost_journal(
    tmp_path: Path,
) -> None:
    cost_journal = tmp_path / "cost-journal"
    write_journal_directory(cost_journal)
    measurement = tmp_path / "measurement"
    create_measurement_journal(
        workspace_root=tmp_path,
        journal_root=measurement,
        reserve_bytes=0,
        run_id="binance-measurement-v1",
        cost_journal_root=cost_journal,
    )
    with pytest.raises(BinanceMeasurementJournalSpecError, match="immutable"):
        create_measurement_journal(
            workspace_root=tmp_path,
            journal_root=measurement,
            reserve_bytes=0,
            run_id="binance-measurement-v1",
            cost_journal_root=cost_journal,
        )
    # A cost journal whose spec no longer matches its own hash is not a source.
    torn = tmp_path / "torn-cost-journal"
    write_journal_directory(torn)
    spec_path = torn / "journal-spec.json"
    document = read_document(spec_path)
    document["run_id"] = "someone-edited-this"
    spec_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(BinanceMeasurementJournalSpecError, match="hash"):
        create_measurement_journal(
            workspace_root=tmp_path,
            journal_root=tmp_path / "second",
            reserve_bytes=0,
            run_id="binance-measurement-v2",
            cost_journal_root=torn,
        )


def test_the_array_fetcher_refuses_a_foreign_host_before_any_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(*arguments: object, **keywords: object) -> object:
        raise AssertionError("the journal fetcher must not open a connection")

    monkeypatch.setattr(urllib.request, "urlopen", explode)
    refused = (
        "https://evil.example.com/fapi/v1/premiumIndex",
        "http://fapi.binance.com/fapi/v1/premiumIndex",
        "https://fapi.binance.com:8443/fapi/v1/premiumIndex",
        "https://data.binance.vision/fapi/v1/premiumIndex",
    )
    for url in refused:
        with pytest.raises(BinanceCostJournalError):
            public_binance_json_array_fetcher(url)


def test_the_array_fetcher_shares_the_object_fetcher_s_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two fetchers differ in the body they accept and in nothing else."""
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        urlopen_returning(FakeResponse(url="https://evil.example.com/")),
    )
    with pytest.raises(BinanceCostJournalError, match="redirect"):
        public_binance_json_array_fetcher(PREMIUM_INDEX_URL)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        urlopen_returning(FakeResponse(url=PREMIUM_INDEX_URL, status=500)),
    )
    with pytest.raises(BinanceCostJournalError, match="500"):
        public_binance_json_array_fetcher(PREMIUM_INDEX_URL)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        urlopen_returning(
            FakeResponse(url=PREMIUM_INDEX_URL, status=429, headers={"Retry-After": "17"})
        ),
    )
    with pytest.raises(BinanceCostJournalTransportError) as throttled:
        public_binance_json_array_fetcher(PREMIUM_INDEX_URL)
    assert (throttled.value.status, throttled.value.retry_after) == (429, 17)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        urlopen_returning(FakeResponse(url=PREMIUM_INDEX_URL, body=b"x" * 8_000_001)),
    )
    with pytest.raises(BinanceCostJournalError, match="byte limit"):
        public_binance_json_array_fetcher(PREMIUM_INDEX_URL)


def test_each_fetcher_refuses_the_body_shape_the_other_one_takes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        urlopen_returning(FakeResponse(url=PREMIUM_INDEX_URL, body=b'{"symbol": "BTCUSDT"}')),
    )
    with pytest.raises(BinanceCostJournalError, match="array"):
        public_binance_json_array_fetcher(PREMIUM_INDEX_URL)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        urlopen_returning(FakeResponse(url=PREMIUM_INDEX_URL, body=b'[{"symbol": "BTCUSDT"}]')),
    )
    assert public_binance_json_array_fetcher(PREMIUM_INDEX_URL) == [{"symbol": "BTCUSDT"}]
    with pytest.raises(BinanceCostJournalError, match="object"):
        public_binance_json_fetcher(PREMIUM_INDEX_URL)


class SnapshotVenue:
    """An array fetcher over the three all-symbol endpoints.

    `raises` names, per URL, an exception the venue answers with once before it
    serves its payload again - a throttled minute, not a dead endpoint -
    and `payloads` overrides what one endpoint answers.
    """

    def __init__(
        self,
        *,
        payloads: Mapping[str, list[object]] | None = None,
        raises: Mapping[str, Exception] | None = None,
        always_raises: bool = False,
    ) -> None:
        self.urls: list[str] = []
        self.payloads: dict[str, list[object]] = {
            PREMIUM_INDEX_URL: premium_payload(),
            PERP_BOOK_TICKER_URL: perp_book_payload(),
            SPOT_BOOK_TICKER_URL: spot_book_payload(),
            **(payloads or {}),
        }
        self.raises = dict(raises or {})
        self.always_raises = always_raises

    def __call__(self, url: str) -> Sequence[object]:
        self.urls.append(url)
        error = self.raises.get(url)
        if error is not None:
            if not self.always_raises:
                del self.raises[url]
            raise error
        return self.payloads[url]


def measurement_journal(tmp_path: Path) -> Path:
    """A created, empty measurement journal over the two cost-journal legs."""
    cost_journal = tmp_path / "cost-journal"
    write_journal_directory(cost_journal)
    journal = tmp_path / "measurement"
    create_measurement_journal(
        workspace_root=tmp_path,
        journal_root=journal,
        reserve_bytes=0,
        run_id="binance-measurement-v1",
        cost_journal_root=cost_journal,
    )
    return journal


def segment_paths(journal_root: Path) -> list[Path]:
    return sorted((journal_root / "segments").glob("*/*.json"))


def segment_documents(journal_root: Path) -> list[dict[str, object]]:
    return [read_document(path) for path in segment_paths(journal_root)]


def rows_of(document: dict[str, object], key: str) -> list[dict[str, object]]:
    rows = document[key]
    assert isinstance(rows, list)
    return rows


def nothing_excluded() -> dict[str, int]:
    return {"premiumIndex": 0, "perpBookTicker": 0, "spotBookTicker": 0}


def relative_names(journal_root: Path) -> list[str]:
    root = journal_root / "segments"
    return [path.relative_to(root).as_posix() for path in segment_paths(journal_root)]


def rewrite_chain_head(journal_root: Path, **overrides: object) -> None:
    """Rewrite the chain head with its own hash recomputed, the way a writer would."""
    path = journal_root / "chain-head.json"
    document = {**read_document(path), **overrides}
    material = {key: value for key, value in document.items() if key != "content_hash"}
    path.write_bytes(canonical_json({**material, "content_hash": content_sha256(material)}))


def verified(journal_root: Path) -> tuple[bool, tuple[str, ...], MeasurementChainHead | None]:
    _, spec_hash = load_measurement_journal_spec(journal_root)
    return _verified_chain(journal_root, spec_hash=spec_hash)


def test_six_rounds_chain_and_only_the_snapshot_rounds_carry_the_all_symbol_books(
    tmp_path: Path,
) -> None:
    journal = measurement_journal(tmp_path)
    venue, snapshots, sleeper = FakeVenue(), SnapshotVenue(), FakeSleep()
    head = run_measurement_journal(
        workspace_root=tmp_path,
        journal_root=journal,
        reserve_bytes=0,
        rounds=6,
        fetcher=venue,
        array_fetcher=snapshots,
        clock=FakeClock(),
        sleep=sleeper,
    )
    documents = segment_documents(journal)
    assert [document["sequence"] for document in documents] == [0, 1, 2, 3, 4, 5]
    assert (head.segment_count, head.last_sequence) == (6, 5)
    assert head.version == MEASUREMENT_JOURNAL_VERSION
    assert head.final_segment_hash == documents[5]["content_hash"]
    assert read_document(journal / "chain-head.json") == head.model_dump(mode="json")
    # The chain: every segment names its predecessor and seals itself.
    previous = ZERO_HASH
    for document in documents:
        assert document["previous_segment_hash"] == previous
        material = {key: value for key, value in document.items() if key != "content_hash"}
        assert document["content_hash"] == content_sha256(material)
        previous = str(document["content_hash"])
    # Rounds 0 and 5 are the snapshot rounds; the four between them walk depth only.
    assert [document["premium_index"] is None for document in documents] == [
        False, True, True, True, True, False
    ]
    for document in documents[1:5]:
        assert (document["perp_book"], document["spot_book"]) == (None, None)
        assert document["failures"] == []
    # Sealed on every segment, zero where the round took no snapshot (ruling 11).
    assert [document["excluded"] for document in documents] == [nothing_excluded()] * 6
    assert snapshots.urls == [
        PREMIUM_INDEX_URL, PERP_BOOK_TICKER_URL, SPOT_BOOK_TICKER_URL,
    ] * 2
    # Every round walks both declared instruments, whether it snapshots or not.
    for document in documents:
        observations = document["depth"]
        assert isinstance(observations, list)
        assert [observation["instrument_id"] for observation in observations] == [
            "spot:BTCUSDT",
            "perp:BTCUSDT",
        ]
        assert all(observation["ok"] for observation in observations)
    assert verified(journal)[:2] == (True, ())
    # A resumed run waits first; a journal that starts here samples at once.
    assert sleeper.calls == [ROUND_PERIOD_REMAINDER] * 5


def test_a_snapshot_round_records_every_usdt_symbol_and_drops_the_rest(
    tmp_path: Path,
) -> None:
    journal = measurement_journal(tmp_path)
    run_measurement_journal(
        workspace_root=tmp_path,
        journal_root=journal,
        reserve_bytes=0,
        rounds=1,
        fetcher=FakeVenue(),
        array_fetcher=SnapshotVenue(),
        clock=FakeClock(),
        sleep=FakeSleep(),
    )
    document = segment_documents(journal)[0]
    premium = document["premium_index"]
    assert premium == [
        {
            "symbol": "BTCUSDT",
            "mark_price": "100.5",
            "index_price": "100",
            "last_funding_rate": "0.0001",
            "next_funding_time_ms": 1_757_001_600_000,
            "basis_bps": "50.000000",
        },
        {
            "symbol": "ETHUSDT",
            "mark_price": "2000.4",
            "index_price": "2000.0",
            "last_funding_rate": "-0.00005",
            "next_funding_time_ms": 1_757_001_600_000,
            "basis_bps": "2.000000",
        },
        {
            "symbol": "XRPUSDT",
            "mark_price": "9.9997",
            "index_price": "10.0000",
            "last_funding_rate": "0",
            "next_funding_time_ms": 1_757_001_600_000,
            "basis_bps": "-0.300000",
        },
    ]
    assert document["perp_book"] == [
        {
            "symbol": "BTCUSDT",
            "bid_price": "99.99",
            "bid_qty": "3",
            "ask_price": "100.01",
            "ask_qty": "4",
            "spread_bps": "2.000000",
        },
        {
            "symbol": "ETHUSDT",
            "bid_price": "1999",
            "bid_qty": "1.5",
            "ask_price": "2001",
            "ask_qty": "2.5",
            "spread_bps": "10.000000",
        },
    ]
    spot = document["spot_book"]
    assert isinstance(spot, list)
    assert [row["symbol"] for row in spot] == ["BTCUSDT", "DOGEUSDT"]
    assert [row["spread_bps"] for row in spot] == ["10.000000", "2857.142857"]


def test_a_throttled_snapshot_is_a_named_failure_that_buys_the_next_round_s_wait(
    tmp_path: Path,
) -> None:
    journal = measurement_journal(tmp_path)
    snapshots = SnapshotVenue(
        raises={PREMIUM_INDEX_URL: BinanceCostJournalTransportError(429, 17)}
    )
    sleeper = FakeSleep()
    run_measurement_journal(
        workspace_root=tmp_path,
        journal_root=journal,
        reserve_bytes=0,
        rounds=2,
        fetcher=FakeVenue(),
        array_fetcher=snapshots,
        clock=FakeClock(),
        sleep=sleeper,
    )
    documents = segment_documents(journal)
    assert documents[0]["failures"] == ["premiumIndex:429"]
    assert documents[0]["premium_index"] is None
    # The two endpoints that answered are recorded, and so is the whole depth walk.
    assert documents[0]["perp_book"] is not None
    assert documents[0]["spot_book"] is not None
    observations = documents[0]["depth"]
    assert isinstance(observations, list)
    assert all(observation["ok"] for observation in observations)
    # The venue asked for 17 s; the next round waits its period and then those.
    assert sleeper.calls == [ROUND_PERIOD_REMAINDER, 17.0]
    assert documents[1]["failures"] == []
    assert verified(journal)[:2] == (True, ())


def test_a_malformed_snapshot_payload_is_a_recorded_failure_not_an_exception(
    tmp_path: Path,
) -> None:
    journal = measurement_journal(tmp_path)
    crossed = spot_book_payload()
    entry = crossed[1]
    assert isinstance(entry, dict)
    crossed[1] = {**entry, "bidPrice": 3.0}
    snapshots = SnapshotVenue(
        payloads={SPOT_BOOK_TICKER_URL: crossed},
        raises={
            PERP_BOOK_TICKER_URL: BinanceCostJournalError("public response is not a JSON array")
        },
    )
    run_measurement_journal(
        workspace_root=tmp_path,
        journal_root=journal,
        reserve_bytes=0,
        rounds=1,
        fetcher=FakeVenue(),
        array_fetcher=snapshots,
        clock=FakeClock(),
        sleep=FakeSleep(),
    )
    document = segment_documents(journal)[0]
    assert document["failures"] == [
        "perpBookTicker:BinanceCostJournalError: public response is not a JSON array",
        "spotBookTicker:malformed entry DOGEUSDT",
    ]
    assert (document["perp_book"], document["spot_book"]) == (None, None)
    assert document["premium_index"] is not None
    # A wrong type is an API change, not a halted pair: nothing was excluded.
    assert document["excluded"] == nothing_excluded()
    assert verified(journal)[:2] == (True, ())


def test_a_round_in_which_every_request_fails_is_still_published(tmp_path: Path) -> None:
    """Spec 6: a round with failures is a partial segment, never a dropped one (ruling 9)."""
    journal = measurement_journal(tmp_path)
    snapshots = SnapshotVenue(
        raises=dict.fromkeys(
            (PREMIUM_INDEX_URL, PERP_BOOK_TICKER_URL, SPOT_BOOK_TICKER_URL),
            BinanceCostJournalTransportError(418, 30),
        ),
        always_raises=True,
    )
    sleeper = FakeSleep()
    head = run_measurement_journal(
        workspace_root=tmp_path,
        journal_root=journal,
        reserve_bytes=0,
        rounds=1,
        fetcher=FakeVenue(dark=("binance.com",)),
        array_fetcher=snapshots,
        clock=FakeClock(),
        sleep=sleeper,
    )
    assert head.segment_count == 1
    document = segment_documents(journal)[0]
    assert document["failures"] == [
        "premiumIndex:418",
        "perpBookTicker:418",
        "spotBookTicker:418",
    ]
    observations = document["depth"]
    assert isinstance(observations, list)
    # Ruling 10: a depth failure stays inside its own observation.
    assert [observation["ok"] for observation in observations] == [False, False]
    assert all("ConnectionError" in str(observation["reason"]) for observation in observations)
    assert verified(journal)[:2] == (True, ())
    assert sleeper.calls == []


def test_segments_land_in_day_directories_taken_from_their_received_time(
    tmp_path: Path,
) -> None:
    journal = measurement_journal(tmp_path)
    # A second before 2026-09-23T00:00:00Z, half a second a clock call: the
    # first round is stamped on the 22nd and the second on the 23rd.
    midnight_ns = 1_790_121_600_000_000_000
    run_measurement_journal(
        workspace_root=tmp_path,
        journal_root=journal,
        reserve_bytes=0,
        rounds=2,
        fetcher=FakeVenue(),
        array_fetcher=SnapshotVenue(),
        clock=FakeClock(start=midnight_ns - 1_000_000_000, step=500_000_000),
        sleep=FakeSleep(),
    )
    assert relative_names(journal) == [
        "2026-09-22/0000000000.json",
        "2026-09-23/0000000001.json",
    ]
    documents = segment_documents(journal)
    assert documents[0]["received_time_ns"] == midnight_ns - 500_000_000
    assert documents[1]["received_time_ns"] == midnight_ns + 1_500_000_000
    assert verified(journal)[:2] == (True, ())
    assert _segment_paths(journal) == segment_paths(journal)


def test_a_resumed_run_waits_a_whole_interval_and_continues_the_chain(
    tmp_path: Path,
) -> None:
    journal = measurement_journal(tmp_path)
    run_measurement_journal(
        workspace_root=tmp_path,
        journal_root=journal,
        reserve_bytes=0,
        rounds=2,
        fetcher=FakeVenue(),
        array_fetcher=SnapshotVenue(),
        clock=FakeClock(),
        sleep=FakeSleep(),
    )
    sleeper = FakeSleep()
    head = run_measurement_journal(
        workspace_root=tmp_path,
        journal_root=journal,
        reserve_bytes=0,
        rounds=2,
        fetcher=FakeVenue(),
        array_fetcher=SnapshotVenue(),
        clock=FakeClock(start=RECEIVED_NS + 10**12),
        sleep=sleeper,
    )
    assert (head.segment_count, head.last_sequence) == (4, 3)
    assert sleeper.calls == [SAMPLE_INTERVAL_SECONDS, ROUND_PERIOD_REMAINDER]
    documents = segment_documents(journal)
    assert documents[2]["previous_segment_hash"] == documents[1]["content_hash"]
    assert verified(journal)[:2] == (True, ())


def test_an_unbounded_run_samples_until_the_process_is_stopped(tmp_path: Path) -> None:
    journal = measurement_journal(tmp_path)

    class StoppingSleep:
        def __init__(self) -> None:
            self.calls: list[float] = []

        def __call__(self, seconds: float) -> None:
            self.calls.append(seconds)
            if len(self.calls) == 3:
                raise KeyboardInterrupt

    sleeper = StoppingSleep()
    with pytest.raises(KeyboardInterrupt):
        run_measurement_journal(
            workspace_root=tmp_path,
            journal_root=journal,
            reserve_bytes=0,
            rounds=None,
            fetcher=FakeVenue(),
            array_fetcher=SnapshotVenue(),
            clock=FakeClock(),
            sleep=sleeper,
        )
    assert [document["sequence"] for document in segment_documents(journal)] == [0, 1, 2]
    assert verified(journal)[:2] == (True, ())


def test_a_run_refuses_a_spec_that_changed_after_the_journal_was_created(
    tmp_path: Path,
) -> None:
    journal = measurement_journal(tmp_path)
    run_measurement_journal(
        workspace_root=tmp_path,
        journal_root=journal,
        reserve_bytes=0,
        rounds=1,
        fetcher=FakeVenue(),
        array_fetcher=SnapshotVenue(),
        clock=FakeClock(),
        sleep=FakeSleep(),
    )
    spec_path = journal / "journal-spec.json"
    document = read_document(spec_path)
    document["run_id"] = "someone-changed-the-declaration"
    material = {key: value for key, value in document.items() if key != "spec_hash"}
    # Resealed: the spec matches its own hash and no longer matches the chain.
    spec_path.write_bytes(canonical_json({**material, "spec_hash": content_sha256(material)}))
    with pytest.raises(BinanceMeasurementJournalSpecError, match="bound to a different spec"):
        run_measurement_journal(
            workspace_root=tmp_path,
            journal_root=journal,
            reserve_bytes=0,
            rounds=1,
            fetcher=FakeVenue(),
            array_fetcher=SnapshotVenue(),
            clock=FakeClock(),
            sleep=FakeSleep(),
        )
    # Unsealed: the spec does not even match its own recorded hash.
    spec_path.write_bytes(canonical_json({**material, "spec_hash": "a" * 64}))
    with pytest.raises(BinanceMeasurementJournalSpecError, match="recorded hash"):
        run_measurement_journal(
            workspace_root=tmp_path,
            journal_root=journal,
            reserve_bytes=0,
            rounds=1,
            fetcher=FakeVenue(),
            array_fetcher=SnapshotVenue(),
            clock=FakeClock(),
            sleep=FakeSleep(),
        )


def test_a_run_of_no_rounds_is_refused(tmp_path: Path) -> None:
    journal = measurement_journal(tmp_path)
    with pytest.raises(BinanceMeasurementJournalSpecError, match="at least one round"):
        run_measurement_journal(
            workspace_root=tmp_path,
            journal_root=journal,
            reserve_bytes=0,
            rounds=0,
            fetcher=FakeVenue(),
            array_fetcher=SnapshotVenue(),
            clock=FakeClock(),
            sleep=FakeSleep(),
        )


def journal_with_rounds(tmp_path: Path, *, rounds: int) -> Path:
    journal = measurement_journal(tmp_path)
    run_measurement_journal(
        workspace_root=tmp_path,
        journal_root=journal,
        reserve_bytes=0,
        rounds=rounds,
        fetcher=FakeVenue(),
        array_fetcher=SnapshotVenue(),
        clock=FakeClock(),
        sleep=FakeSleep(),
    )
    return journal


def test_the_segment_walk_orders_by_sequence_across_days_and_refuses_a_foreign_name(
    tmp_path: Path,
) -> None:
    journal = journal_with_rounds(tmp_path, rounds=2)
    day = journal / "segments" / "2024-01-01"
    day.mkdir()
    # A day that sorts before the journal's own still follows it by sequence.
    moved = segment_paths(journal)[1].replace(day / "0000000001.json")
    assert [path.name for path in _segment_paths(journal)] == [
        "0000000000.json",
        "0000000001.json",
    ]
    assert _segment_paths(journal)[1] == moved
    # A publish that died mid-flight is ignored, not walked.
    (day / "0000000002.json.4711.tmp").write_bytes(b"{")
    (day / "0000000002.json.tmp").write_bytes(b"{")
    assert len(_segment_paths(journal)) == 2
    foreign = day / "readme.json"
    foreign.write_bytes(b"{}")
    with pytest.raises(BinanceMeasurementJournalSpecError, match=r"readme\.json"):
        _segment_paths(journal)
    assert verified(journal)[:2] == (False, ("SEGMENT_LAYOUT:2024-01-01/readme.json",))
    foreign.unlink()
    (journal / "segments" / "not-a-day").mkdir()
    with pytest.raises(BinanceMeasurementJournalSpecError, match="not-a-day"):
        _segment_paths(journal)


def test_verification_names_a_tampered_segment_a_broken_link_and_a_mismatched_head(
    tmp_path: Path,
) -> None:
    journal = journal_with_rounds(tmp_path, rounds=2)
    names = relative_names(journal)
    valid, reasons, head = verified(journal)
    assert (valid, reasons) == (True, ())
    assert head is not None and head.segment_count == 2
    # An edited segment whose recorded hash was left behind.
    path = segment_paths(journal)[1]
    unseal(path)
    assert verified(journal)[:2] == (False, (f"SEGMENT_HASH_MISMATCH:{names[1]}",))
    # A resealed segment that no longer names its predecessor.
    rewrite_segment(path, previous_segment_hash="a" * 64)
    assert verified(journal)[:2] == (
        False,
        (f"CHAIN_LINK_BROKEN:{names[1]}", "CHAIN_HEAD_MISMATCH:final_segment_hash"),
    )
    # A resealed segment that names another journal's spec.
    predecessor = read_document(segment_paths(journal)[0])["content_hash"]
    rewrite_segment(path, previous_segment_hash=predecessor, spec_hash="f" * 64)
    assert verified(journal)[:2] == (
        False,
        (f"SEGMENT_SPEC_MISMATCH:{names[1]}", "CHAIN_HEAD_MISMATCH:final_segment_hash"),
    )


def test_verification_names_a_missing_sequence_and_a_rewound_head(tmp_path: Path) -> None:
    journal = journal_with_rounds(tmp_path, rounds=3)
    names = relative_names(journal)
    segment_paths(journal)[1].unlink()
    reasons = verified(journal)[1]
    assert f"SEGMENT_SEQUENCE_GAP:{names[2]}" in reasons
    assert "CHAIN_HEAD_MISMATCH:segment_count" in reasons
    journal = journal_with_rounds(tmp_path / "second", rounds=2)
    first_hash = read_document(segment_paths(journal)[0])["content_hash"]
    rewrite_chain_head(journal, segment_count=1, last_sequence=0, final_segment_hash=first_hash)
    assert verified(journal)[:2] == (
        False,
        (
            "CHAIN_HEAD_MISMATCH:segment_count",
            "CHAIN_HEAD_MISMATCH:last_sequence",
            "CHAIN_HEAD_MISMATCH:final_segment_hash",
        ),
    )
    (journal / "chain-head.json").unlink()
    assert verified(journal)[:2] == (False, ("CHAIN_HEAD_MISSING",))
    (journal / "chain-head.json").write_bytes(b'{"version": "bina')
    assert verified(journal)[:2] == (False, ("CHAIN_HEAD_UNREADABLE",))


def test_verification_reports_an_empty_journal_and_a_missing_directory(
    tmp_path: Path,
) -> None:
    journal = measurement_journal(tmp_path)
    assert verified(journal) == (True, (), None)
    (journal / "segments").rmdir()
    assert verified(journal)[:2] == (False, ("SEGMENT_DIRECTORY_MISSING",))


def test_a_torn_trailing_segment_is_discarded_and_its_round_sampled_again(
    tmp_path: Path,
) -> None:
    journal = journal_with_rounds(tmp_path, rounds=2)
    day = segment_paths(journal)[1].parent
    torn = day / "0000000002.json"
    torn.write_bytes(canonical_json(read_document(segment_paths(journal)[1]))[:40])
    assert verified(journal)[0] is False
    head = run_measurement_journal(
        workspace_root=tmp_path,
        journal_root=journal,
        reserve_bytes=0,
        rounds=1,
        fetcher=FakeVenue(),
        array_fetcher=SnapshotVenue(),
        clock=FakeClock(start=RECEIVED_NS + 10**12),
        sleep=FakeSleep(),
    )
    assert (head.segment_count, head.last_sequence) == (3, 2)
    assert verified(journal)[:2] == (True, ())
    assert [document["sequence"] for document in segment_documents(journal)] == [0, 1, 2]


def test_a_segment_a_kill_left_beyond_the_head_is_adopted_on_resume(
    tmp_path: Path,
) -> None:
    journal = journal_with_rounds(tmp_path, rounds=2)
    orphan_hash = segment_documents(journal)[1]["content_hash"]
    first_hash = segment_documents(journal)[0]["content_hash"]
    rewrite_chain_head(journal, segment_count=1, last_sequence=0, final_segment_hash=first_hash)
    assert verified(journal)[0] is False
    head = run_measurement_journal(
        workspace_root=tmp_path,
        journal_root=journal,
        reserve_bytes=0,
        rounds=1,
        fetcher=FakeVenue(),
        array_fetcher=SnapshotVenue(),
        clock=FakeClock(start=RECEIVED_NS + 10**12),
        sleep=FakeSleep(),
    )
    assert (head.segment_count, head.last_sequence) == (3, 2)
    assert segment_documents(journal)[2]["previous_segment_hash"] == orphan_hash
    assert verified(journal)[:2] == (True, ())


def test_a_head_that_was_never_written_is_rebuilt_from_the_segments(
    tmp_path: Path,
) -> None:
    journal = journal_with_rounds(tmp_path, rounds=2)
    (journal / "chain-head.json").write_bytes(b'{"version": "bina')
    head = run_measurement_journal(
        workspace_root=tmp_path,
        journal_root=journal,
        reserve_bytes=0,
        rounds=1,
        fetcher=FakeVenue(),
        array_fetcher=SnapshotVenue(),
        clock=FakeClock(start=RECEIVED_NS + 10**12),
        sleep=FakeSleep(),
    )
    assert (head.segment_count, head.last_sequence) == (3, 2)
    assert verified(journal)[:2] == (True, ())


def test_a_resume_refuses_a_chain_that_was_changed_rather_than_interrupted(
    tmp_path: Path,
) -> None:
    """A rewound head is a kill; an edited segment and a forward head are not."""
    journal = journal_with_rounds(tmp_path, rounds=2)
    rewrite_segment(segment_paths(journal)[0], received_time_ns=RECEIVED_NS + 5)
    with pytest.raises(BinanceMeasurementJournalSpecError, match="CHAIN_LINK_BROKEN"):
        run_measurement_journal(
            workspace_root=tmp_path,
            journal_root=journal,
            reserve_bytes=0,
            rounds=1,
            fetcher=FakeVenue(),
            array_fetcher=SnapshotVenue(),
            clock=FakeClock(start=RECEIVED_NS + 10**12),
            sleep=FakeSleep(),
        )
    second = journal_with_rounds(tmp_path / "second", rounds=2)
    rewrite_chain_head(second, final_segment_hash="b" * 64)
    with pytest.raises(BinanceMeasurementJournalSpecError, match="CHAIN_HEAD_MISMATCH"):
        run_measurement_journal(
            workspace_root=tmp_path / "second",
            journal_root=second,
            reserve_bytes=0,
            rounds=1,
            fetcher=FakeVenue(),
            array_fetcher=SnapshotVenue(),
            clock=FakeClock(start=RECEIVED_NS + 10**12),
            sleep=FakeSleep(),
        )
    third = journal_with_rounds(tmp_path / "third", rounds=2)
    rewrite_chain_head(third, spec_hash="c" * 64)
    with pytest.raises(BinanceMeasurementJournalSpecError, match="bound to a different spec"):
        run_measurement_journal(
            workspace_root=tmp_path / "third",
            journal_root=third,
            reserve_bytes=0,
            rounds=1,
            fetcher=FakeVenue(),
            array_fetcher=SnapshotVenue(),
            clock=FakeClock(start=RECEIVED_NS + 10**12),
            sleep=FakeSleep(),
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"indexPrice": "0"},
        {"markPrice": "0"},
        {"markPrice": "-1"},
        {"nextFundingTime": 0},
    ],
)
def test_a_premium_entry_that_measures_nothing_is_excluded_not_refused(
    overrides: dict[str, object],
) -> None:
    """Ruling 11: a delisted perpetual's zero index costs its row, not the venue's."""
    payload = premium_payload()
    entry = payload[1]
    assert isinstance(entry, dict)
    payload[1] = {**entry, **overrides}
    rows, excluded = premium_rows(payload)
    assert [row.symbol for row in rows] == ["BTCUSDT", "XRPUSDT"]
    assert excluded == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"bidQty": "0"},
        {"askQty": "0"},
        {"bidPrice": "0"},
        {"askPrice": "0"},
        {"bidPrice": "-1999"},
        # Crossed, and locked: a zero spread is not a measurement either.
        {"bidPrice": "2001", "askPrice": "2000"},
        {"bidPrice": "2000", "askPrice": "2000"},
    ],
)
def test_a_book_entry_that_measures_nothing_is_excluded_not_refused(
    overrides: dict[str, object],
) -> None:
    """Ruling 11: a halted pair quotes a zero book every day and costs its row only."""
    payload = perp_book_payload()
    entry = payload[1]
    assert isinstance(entry, dict)
    payload[1] = {**entry, **overrides}
    rows, excluded = book_rows(payload)
    assert [row.symbol for row in rows] == ["BTCUSDT"]
    assert excluded == 1


def test_a_payload_whose_every_usdt_symbol_is_unusable_yields_no_rows_and_counts_them() -> None:
    """Ruling 12: the ordinary rule at its extreme, not a second path."""
    payload: list[object] = [
        {**entry, "bidQty": "0"} for entry in perp_book_payload() if isinstance(entry, dict)
    ]
    assert book_rows(payload) == ((), 2)
    # An array holding no USDT symbol at all is a schema anomaly, and refuses.
    with pytest.raises(DepthPayloadError, match="no USDT symbols"):
        book_rows([entry for entry in payload if "USDC" in str(entry)])


def test_unusable_symbols_are_excluded_from_a_snapshot_and_counted_in_the_segment(
    tmp_path: Path,
) -> None:
    journal = measurement_journal(tmp_path)
    premium = premium_payload()
    delisted = premium[2]
    assert isinstance(delisted, dict)
    premium[2] = {**delisted, "indexPrice": "0"}
    halted: dict[str, object] = {
        "symbol": "LUNAUSDT",
        "bidPrice": "1.5",
        "bidQty": "0",
        "askPrice": "1.6",
        "askQty": "10",
    }
    crossed: dict[str, object] = {
        "symbol": "XRPUSDT",
        "bidPrice": "2.1",
        "bidQty": "5",
        "askPrice": "2.0",
        "askQty": "5",
    }
    run_measurement_journal(
        workspace_root=tmp_path,
        journal_root=journal,
        reserve_bytes=0,
        rounds=1,
        fetcher=FakeVenue(),
        array_fetcher=SnapshotVenue(
            payloads={
                PREMIUM_INDEX_URL: premium,
                SPOT_BOOK_TICKER_URL: [*spot_book_payload(), halted, crossed],
            }
        ),
        clock=FakeClock(),
        sleep=FakeSleep(),
    )
    document = segment_documents(journal)[0]
    assert document["failures"] == []
    assert document["excluded"] == {
        "premiumIndex": 1,
        "perpBookTicker": 0,
        "spotBookTicker": 2,
    }
    assert [row["symbol"] for row in rows_of(document, "premium_index")] == [
        "BTCUSDT",
        "ETHUSDT",
    ]
    assert [row["symbol"] for row in rows_of(document, "spot_book")] == [
        "BTCUSDT",
        "DOGEUSDT",
    ]
    assert [row["symbol"] for row in rows_of(document, "perp_book")] == ["BTCUSDT", "ETHUSDT"]
    assert verified(journal)[:2] == (True, ())


def test_an_endpoint_whose_every_symbol_is_unusable_is_an_empty_snapshot(
    tmp_path: Path,
) -> None:
    """Ruling 12: rows the round could not use are counted, never a failure."""
    journal = measurement_journal(tmp_path)
    dead: list[object] = [
        {**entry, "askQty": "0"} for entry in perp_book_payload() if isinstance(entry, dict)
    ]
    run_measurement_journal(
        workspace_root=tmp_path,
        journal_root=journal,
        reserve_bytes=0,
        rounds=1,
        fetcher=FakeVenue(),
        array_fetcher=SnapshotVenue(payloads={PERP_BOOK_TICKER_URL: dead}),
        clock=FakeClock(),
        sleep=FakeSleep(),
    )
    document = segment_documents(journal)[0]
    assert document["failures"] == []
    # Read and measured nothing: the empty tuple, not the absent one.
    assert document["perp_book"] == []
    assert document["premium_index"] is not None
    assert document["excluded"] == {
        "premiumIndex": 0,
        "perpBookTicker": 2,
        "spotBookTicker": 0,
    }
    assert verified(journal)[:2] == (True, ())


DAY_NS = 86_400_000_000_000
SIX_HOURS_NS = 21_600_000_000_000
# 2026-09-23T00:00:00Z.
THREE_DAY_START_NS = 1_790_121_600_000_000_000


def three_day_journal(tmp_path: Path) -> Path:
    """A journal with one segment a day, in 2026-09-23, -24 and -25.

    A round spends four clock calls, so a six-hour step moves the round stamp a
    whole day and every round opens its own day directory.
    """
    journal = measurement_journal(tmp_path)
    run_measurement_journal(
        workspace_root=tmp_path,
        journal_root=journal,
        reserve_bytes=0,
        rounds=3,
        fetcher=FakeVenue(),
        array_fetcher=SnapshotVenue(),
        clock=FakeClock(start=THREE_DAY_START_NS, step=SIX_HOURS_NS),
        sleep=FakeSleep(),
    )
    return journal


def resume(journal: Path, *, workspace: Path) -> MeasurementChainHead:
    """One more round, stamped the day after the three-day fixture ends."""
    return run_measurement_journal(
        workspace_root=workspace,
        journal_root=journal,
        reserve_bytes=0,
        rounds=1,
        fetcher=FakeVenue(),
        array_fetcher=SnapshotVenue(),
        clock=FakeClock(start=THREE_DAY_START_NS + 3 * DAY_NS),
        sleep=FakeSleep(),
    )


def recorded_reads(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Every path opened for reading from here on, in order."""
    reads: list[Path] = []
    read_text = Path.read_text
    read_bytes = Path.read_bytes

    def record_text(path: Path, encoding: str | None = None, errors: str | None = None) -> str:
        reads.append(path)
        return read_text(path, encoding=encoding, errors=errors)

    def record_bytes(path: Path) -> bytes:
        reads.append(path)
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_text", record_text)
    monkeypatch.setattr(Path, "read_bytes", record_bytes)
    return reads


def unseal(path: Path) -> None:
    """Change a segment and leave its recorded hash behind, as an editor would.

    The stamp moves by a nanosecond rather than to the epoch, so the edit is
    the one thing under test - a segment that no longer matches its own hash -
    and not also a segment sitting under the wrong day directory.
    """
    document = read_document(path)
    stamp = document["received_time_ns"]
    assert isinstance(stamp, int)
    document["received_time_ns"] = stamp + 1
    path.write_bytes(canonical_json(document))


def test_a_resume_reads_only_the_head_and_the_two_newest_day_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ruling 13: a restart costs a listing and two days, not the whole history."""
    journal = three_day_journal(tmp_path)
    assert relative_names(journal) == [
        "2026-09-23/0000000000.json",
        "2026-09-24/0000000001.json",
        "2026-09-25/0000000002.json",
    ]
    reads = recorded_reads(monkeypatch)
    head = resume(journal, workspace=tmp_path)
    assert (head.segment_count, head.last_sequence) == (4, 3)
    segments = journal / "segments"
    opened = {
        path.relative_to(segments).as_posix() for path in reads if segments in path.parents
    }
    assert opened == {"2026-09-24/0000000001.json", "2026-09-25/0000000002.json"}
    assert journal / "chain-head.json" in reads
    assert journal / "journal-spec.json" in reads
    assert relative_names(journal)[3] == "2026-09-26/0000000003.json"


def test_a_segment_outside_the_resume_window_does_not_stop_a_restart(
    tmp_path: Path,
) -> None:
    journal = three_day_journal(tmp_path)
    _, spec_hash = load_measurement_journal_spec(journal)
    unseal(segment_paths(journal)[0])
    assert _verified_tail(journal, spec_hash=spec_hash)[:2] == (True, ())
    # The whole-chain verification Task 6 exposes still names it.
    assert _verified_chain(journal, spec_hash=spec_hash)[:2] == (
        False,
        ("SEGMENT_HASH_MISMATCH:2026-09-23/0000000000.json",),
    )
    head = resume(journal, workspace=tmp_path)
    assert (head.segment_count, head.last_sequence) == (4, 3)


def test_a_segment_inside_the_resume_window_refuses_a_restart(tmp_path: Path) -> None:
    journal = three_day_journal(tmp_path)
    _, spec_hash = load_measurement_journal_spec(journal)
    unseal(segment_paths(journal)[1])
    assert _verified_tail(journal, spec_hash=spec_hash)[:2] == (
        False,
        ("SEGMENT_HASH_MISMATCH:2026-09-24/0000000001.json",),
    )
    with pytest.raises(BinanceMeasurementJournalSpecError, match="SEGMENT_HASH_MISMATCH"):
        resume(journal, workspace=tmp_path)


def test_a_segment_deleted_outside_the_window_still_refuses_a_restart(
    tmp_path: Path,
) -> None:
    """The listing covers the whole journal even where the reading does not."""
    journal = three_day_journal(tmp_path)
    segment_paths(journal)[0].unlink()
    with pytest.raises(
        BinanceMeasurementJournalSpecError, match="CHAIN_HEAD_MISMATCH:segment_count"
    ):
        resume(journal, workspace=tmp_path)


HOUR_NS = 3_600_000_000_000
# Three runs, each two rounds, under a clock that jumps forward three days and
# is then corrected: the day directories end up out of sequence order, which is
# what a reboot with a wrong RTC followed by an NTP correction leaves behind.
CLOCK_STEPPED_DAYS = (
    THREE_DAY_START_NS - DAY_NS,  # 2026-09-22: sequences 0 and 1
    THREE_DAY_START_NS + 2 * DAY_NS,  # 2026-09-25: sequences 2 and 3
    THREE_DAY_START_NS,  # 2026-09-23: sequences 4 and 5
)
CLOCK_STEPPED_LAYOUT = [
    "2026-09-22/0000000000.json",
    "2026-09-22/0000000001.json",
    "2026-09-23/0000000004.json",
    "2026-09-23/0000000005.json",
    "2026-09-25/0000000002.json",
    "2026-09-25/0000000003.json",
]
# The last nanosecond of 2026-09-23, so a window over the two oldest day names
# holds sequences 0, 1, 4 and 5 and leaves 2 and 3 outside it.
SECOND_DAY_END_NS = THREE_DAY_START_NS + DAY_NS - 1


def clock_stepped_journal(tmp_path: Path) -> Path:
    """Six rounds whose day directories are not monotone in sequence order."""
    journal = measurement_journal(tmp_path)
    for start_ns in CLOCK_STEPPED_DAYS:
        run_measurement_journal(
            workspace_root=tmp_path,
            journal_root=journal,
            reserve_bytes=0,
            rounds=2,
            fetcher=FakeVenue(),
            array_fetcher=SnapshotVenue(),
            clock=FakeClock(start=start_ns + HOUR_NS),
            sleep=FakeSleep(),
        )
    return journal


def test_a_backwards_clock_step_across_utc_midnight_does_not_kill_the_restart(
    tmp_path: Path,
) -> None:
    """C1: the tail window is a contiguous sequence suffix, not a day-name filter.

    Filtering the sequence-ordered listing by the two newest day *names* picks
    sequences 0, 1, 4 and 5 out of this layout, which is not a run - so the
    anchored walk reports a gap and the run refuses with exit 2 on every
    restart, for ever, while the whole-chain verification says the journal is
    healthy. The suffix after the last segment outside the window is what the
    walk can actually be anchored on.
    """
    journal = clock_stepped_journal(tmp_path)
    assert relative_names(journal) == CLOCK_STEPPED_LAYOUT
    assert verify_measurement_journal(journal) == (True, ())

    head = run_measurement_journal(
        workspace_root=tmp_path,
        journal_root=journal,
        reserve_bytes=0,
        rounds=1,
        fetcher=FakeVenue(),
        array_fetcher=SnapshotVenue(),
        clock=FakeClock(start=THREE_DAY_START_NS + 2 * HOUR_NS),
        sleep=FakeSleep(),
    )
    assert (head.segment_count, head.last_sequence) == (7, 6)
    assert relative_names(journal)[4] == "2026-09-23/0000000006.json"
    assert verify_measurement_journal(journal) == (True, ())


def test_a_snapshot_of_a_clock_stepped_journal_reads_the_windows_contiguous_run(
    tmp_path: Path,
) -> None:
    """C1: the window is selected by sequence contiguity too, not by day name alone."""
    journal = clock_stepped_journal(tmp_path)
    stamps = stamps_of(journal)
    whole = read_document(
        snapshot_measurement_journal(
            workspace_root=tmp_path,
            journal_root=journal,
            output_path=tmp_path / "whole.json",
            reserve_bytes=0,
            window_start_ns=min(stamps),
            window_end_ns=max(stamps),
        )
    )
    assert (whole["first_sequence"], whole["last_sequence"], whole["rounds"]) == (0, 5, 6)
    # A window that stops before the jumped-forward rounds holds sequences 0,
    # 1, 4 and 5, which is not a run: the contiguous suffix is what it is read
    # over, and the document says so in its own sequence bounds.
    two_days = read_document(
        snapshot_measurement_journal(
            workspace_root=tmp_path,
            journal_root=journal,
            output_path=tmp_path / "two-days.json",
            reserve_bytes=0,
            window_start_ns=min(stamps),
            window_end_ns=SECOND_DAY_END_NS,
        )
    )
    assert (two_days["first_sequence"], two_days["last_sequence"], two_days["rounds"]) == (
        4,
        5,
        2,
    )


# --- Task 6: verify, status and windowed snapshots -------------------------

# `FakeClock`'s default start falls on this UTC day, so every round of a
# journal driven by it lands in one day directory.
DEFAULT_DAY = "1970-01-21"


def reseal_measurement_spec(journal_root: Path, **overrides: object) -> None:
    """Rewrite a journal's spec with its own hash recomputed, before any round."""
    path = journal_root / "journal-spec.json"
    document = {**read_document(path), **overrides}
    material = {key: value for key, value in document.items() if key != "spec_hash"}
    path.write_bytes(canonical_json({**material, "spec_hash": content_sha256(material)}))


def test_verify_measurement_journal_passes_a_chain_and_names_what_broke_it(
    tmp_path: Path,
) -> None:
    journal = journal_with_rounds(tmp_path, rounds=2)
    names = relative_names(journal)
    assert verify_measurement_journal(journal) == (True, ())
    # An edited segment whose recorded hash was left behind.
    path = segment_paths(journal)[1]
    unseal(path)
    assert verify_measurement_journal(journal) == (
        False,
        (f"SEGMENT_HASH_MISMATCH:{names[1]}",),
    )
    # A resealed segment that no longer names its predecessor.
    rewrite_segment(path, previous_segment_hash="a" * 64)
    assert verify_measurement_journal(journal) == (
        False,
        (f"CHAIN_LINK_BROKEN:{names[1]}", "CHAIN_HEAD_MISMATCH:final_segment_hash"),
    )


def test_verify_measurement_journal_names_a_head_that_lost_its_chain(
    tmp_path: Path,
) -> None:
    journal = journal_with_rounds(tmp_path, rounds=2)
    first_hash = read_document(segment_paths(journal)[0])["content_hash"]
    rewrite_chain_head(journal, segment_count=1, last_sequence=0, final_segment_hash=first_hash)
    assert verify_measurement_journal(journal) == (
        False,
        (
            "CHAIN_HEAD_MISMATCH:segment_count",
            "CHAIN_HEAD_MISMATCH:last_sequence",
            "CHAIN_HEAD_MISMATCH:final_segment_hash",
        ),
    )


def test_verify_measurement_journal_reports_an_unverifiable_spec_instead_of_raising(
    tmp_path: Path,
) -> None:
    """A journal whose declaration cannot be read is a reason code, never an exception."""
    journal = journal_with_rounds(tmp_path, rounds=1)
    (journal / "journal-spec.json").write_bytes(b'{"version": "bina')
    assert verify_measurement_journal(journal) == (False, ("JOURNAL_SPEC_UNVERIFIED",))
    (journal / "journal-spec.json").unlink()
    assert verify_measurement_journal(journal) == (False, ("JOURNAL_SPEC_UNVERIFIED",))


def test_a_segment_moved_to_another_day_directory_is_named_by_verification(
    tmp_path: Path,
) -> None:
    """The day directory is checked against the segment's own stamp (Task 5 concern 2)."""
    journal = journal_with_rounds(tmp_path, rounds=6)
    _, spec_hash = load_measurement_journal_spec(journal)
    assert relative_names(journal) == [f"{DEFAULT_DAY}/{index:010d}.json" for index in range(6)]
    assert verify_measurement_journal(journal) == (True, ())
    earlier = journal / "segments" / "1970-01-01"
    earlier.mkdir()
    segment_paths(journal)[3].replace(earlier / "0000000003.json")
    assert verify_measurement_journal(journal) == (
        False,
        ("SEGMENT_DAY_MISMATCH:1970-01-01/0000000003.json",),
    )
    # The check sits in the shared per-segment reason function, so the bounded
    # tail walk refuses a restart here too: the two newest day directories are
    # the moved segment's and the rest of the journal's.
    assert _verified_tail(journal, spec_hash=spec_hash)[:2] == (
        False,
        ("SEGMENT_DAY_MISMATCH:1970-01-01/0000000003.json",),
    )


def test_verify_since_a_day_walks_only_the_days_from_that_day_on(tmp_path: Path) -> None:
    """Ruling 24: the daily check is bounded, the weekly one is the whole chain.

    The full walk grows by 1,440 segments a day for as long as the stream
    lives, so running it every day costs more every day; `--since-day` bounds
    it to the days a daily check is actually about, anchored exactly as the
    restart's tail walk anchors.
    """
    journal = three_day_journal(tmp_path)
    names = relative_names(journal)
    unseal(segment_paths(journal)[0])

    assert verify_measurement_journal(journal) == (
        False,
        (f"SEGMENT_HASH_MISMATCH:{names[0]}",),
    )
    # That day is not opened at all, so its edit is the weekly walk's to find.
    assert verify_measurement_journal(journal, since_day="2026-09-24") == (True, ())

    unseal(segment_paths(journal)[2])
    assert verify_measurement_journal(journal, since_day="2026-09-24") == (
        False,
        (f"SEGMENT_HASH_MISMATCH:{names[2]}",),
    )


def test_verify_since_a_day_still_lists_the_whole_journal(tmp_path: Path) -> None:
    """The listing is the cheap half and covers every day, opened or not."""
    journal = three_day_journal(tmp_path)
    (journal / "segments" / "2026-09-23" / "notes.txt").write_text("x", encoding="utf-8")

    assert verify_measurement_journal(journal, since_day="2026-09-25") == (
        False,
        ("SEGMENT_LAYOUT:2026-09-23/notes.txt",),
    )


def test_verify_since_a_day_names_a_segment_deleted_behind_its_window(
    tmp_path: Path,
) -> None:
    """The count the listing gives the chain head is what catches it."""
    journal = three_day_journal(tmp_path)
    segment_paths(journal)[0].unlink()

    assert verify_measurement_journal(journal, since_day="2026-09-25") == (
        False,
        ("CHAIN_HEAD_MISMATCH:segment_count",),
    )


def test_verify_since_a_day_the_journal_never_reached_says_so(tmp_path: Path) -> None:
    """A day the stream wrote nothing on or after is the daily check's alarm,
    not a pass: there is no segment to judge the chain head against."""
    journal = three_day_journal(tmp_path)

    assert verify_measurement_journal(journal, since_day="2026-09-26") == (
        False,
        ("JOURNAL_SINCE_DAY_EMPTY:2026-09-26",),
    )


class RefusingAfterOneAuthorization(StoragePolicy):
    """A storage policy that authorizes once and then finds the reserve crossed."""

    calls: ClassVar[list[Path]] = []

    def authorize(
        self,
        *,
        target: Path,
        temporary_directory: Path | None = None,
        free_bytes: int,
        worst_case_required_bytes: int,
    ) -> Path:
        RefusingAfterOneAuthorization.calls.append(target)
        if len(RefusingAfterOneAuthorization.calls) > 1:
            raise StorageReserveError("job would cross the configured storage reserve")
        return super().authorize(
            target=target,
            temporary_directory=temporary_directory,
            free_bytes=free_bytes,
            worst_case_required_bytes=worst_case_required_bytes,
        )


def test_the_run_loop_re_authorizes_storage_at_each_utc_day_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ruling 24: the reserve is checked again on every new day of rounds.

    A permanent stream writes about 75 MB a day for as long as it is left
    running, and the reserve was checked once, when the run started. Months
    later the free space that was fine then is not, and the first the run
    would hear of it is an ENOSPC inside a publish. Re-checked once per UTC
    day, it is a named `StorageReserveError` instead - the one refusal the
    supervisor retries.
    """
    journal = measurement_journal(tmp_path)
    RefusingAfterOneAuthorization.calls = []
    monkeypatch.setattr(binance_cost_journal, "StoragePolicy", RefusingAfterOneAuthorization)

    with pytest.raises(StorageReserveError):
        run_measurement_journal(
            workspace_root=tmp_path,
            journal_root=journal,
            reserve_bytes=0,
            rounds=3,
            fetcher=FakeVenue(),
            array_fetcher=SnapshotVenue(),
            clock=FakeClock(start=THREE_DAY_START_NS, step=SIX_HOURS_NS),
            sleep=FakeSleep(),
        )

    # Once at start-up and once when the second day opened; the round that
    # would have opened it is refused before it is published.
    assert RefusingAfterOneAuthorization.calls == [journal.resolve()] * 2
    assert relative_names(journal) == ["2026-09-23/0000000000.json"]


def crossed_spot_payload() -> list[object]:
    """One measurable USDT pair and one whose book is crossed, so it is excluded."""
    return [
        {
            "symbol": "BTCUSDT",
            "bidPrice": "99.95",
            "bidQty": "2",
            "askPrice": "100.05",
            "askQty": "2",
        },
        {"symbol": "XRPUSDT", "bidPrice": "4", "bidQty": "1", "askPrice": "3", "askQty": "1"},
    ]


def test_measurement_status_reports_the_newest_age_the_failure_rate_and_the_exclusions(
    tmp_path: Path,
) -> None:
    journal = measurement_journal(tmp_path)
    snapshots = SnapshotVenue(
        payloads={SPOT_BOOK_TICKER_URL: crossed_spot_payload()},
        raises={PREMIUM_INDEX_URL: BinanceCostJournalTransportError(429, 0)},
    )
    run_measurement_journal(
        workspace_root=tmp_path,
        journal_root=journal,
        reserve_bytes=0,
        rounds=6,
        # The spot leg is dark every round, the perpetual leg is measured.
        fetcher=FakeVenue(dark=("https://api.binance.com",)),
        array_fetcher=snapshots,
        clock=FakeClock(),
        sleep=FakeSleep(),
    )
    stamps = stamps_of(journal)
    newest = stamps[-1]
    status = measurement_status(journal, last=6, clock=lambda: newest + 5_500_000_000)
    assert (status.segment_count, status.last_sequence) == (6, 5)
    assert status.newest_received_time_ns == newest
    assert status.newest_age_seconds == Decimal("5.500")
    assert (status.verify_ok, status.verify_reasons) == (True, ())
    # Rounds 0 and 5 snapshot; the premium index failed in the first of them.
    assert status.snapshot_rounds == 2
    assert status.failure_rate == {
        "premiumIndex": Decimal("0.500000"),
        "perpBookTicker": Decimal(0),
        "spotBookTicker": Decimal(0),
    }
    assert status.excluded_total == {
        "premiumIndex": 0,
        "perpBookTicker": 0,
        "spotBookTicker": 2,
    }
    # Six rounds, two legs, the spot leg failed in every one of them.
    assert status.depth_failure_rate == Decimal("0.500000")


def test_measurement_status_reads_only_the_last_segments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A status of a permanent stream costs a listing and the last rounds, nothing else."""
    journal = three_day_journal(tmp_path)
    newest = stamps_of(journal)[-1]
    reads = recorded_reads(monkeypatch)
    status = measurement_status(journal, last=2, clock=lambda: newest + 1_000_000_000)
    segments = journal / "segments"
    opened = {
        path.relative_to(segments).as_posix() for path in reads if segments in path.parents
    }
    assert opened == {"2026-09-24/0000000001.json", "2026-09-25/0000000002.json"}
    # The count is the head's; the numbers are the last two rounds'.
    assert (status.segment_count, status.last_sequence) == (3, 2)
    assert status.newest_age_seconds == Decimal("1.000")
    # Sequences 1 and 2 take no snapshot, so every rate is zero over nothing.
    assert status.snapshot_rounds == 0
    assert status.failure_rate == dict.fromkeys(SNAPSHOT_ENDPOINTS, Decimal(0))
    assert status.excluded_total == nothing_excluded()
    assert status.depth_failure_rate == Decimal(0)


def test_measurement_status_reports_a_foreign_name_instead_of_raising(
    tmp_path: Path,
) -> None:
    """A status is a report: a supervisor asked how the stream is doing.

    The strict listing refuses a name this journal did not write, which is
    right for everything that verifies; raising it out of `status` told the
    supervisor nothing about the age, the failure rate or the exclusions it
    asked after.
    """
    journal = journal_with_rounds(tmp_path, rounds=2)
    (journal / "segments" / DEFAULT_DAY / "notes.txt").write_text("x", encoding="utf-8")
    newest = stamps_of(journal)[-1]

    status = measurement_status(journal, last=2, clock=lambda: newest)

    assert (status.verify_ok, status.verify_reasons) == (
        False,
        (f"SEGMENT_LAYOUT:{DEFAULT_DAY}/notes.txt",),
    )
    assert (status.segment_count, status.last_sequence) == (2, 1)
    assert status.newest_age_seconds == Decimal("0.000")


@pytest.mark.parametrize("last", [0, -1])
def test_measurement_status_refuses_a_window_of_no_segments(tmp_path: Path, last: int) -> None:
    journal = journal_with_rounds(tmp_path, rounds=1)
    with pytest.raises(BinanceMeasurementJournalSpecError, match="at least one segment"):
        measurement_status(journal, last=last)


class ScriptedSnapshotVenue:
    """An array fetcher that answers a different payload each snapshot round.

    `payloads` holds one entry per snapshot round and per endpoint, in round
    order; `raises` names the snapshot rounds an endpoint answers with an
    exception instead. It is `SnapshotVenue` with a script rather than one
    fixed answer, which is what a window of means needs.
    """

    def __init__(
        self,
        *,
        payloads: Mapping[str, list[list[object]]],
        raises: Mapping[str, Mapping[int, Exception]] | None = None,
    ) -> None:
        self.payloads = dict(payloads)
        self.raises = {url: dict(rounds) for url, rounds in (raises or {}).items()}
        self.calls: dict[str, int] = dict.fromkeys(self.payloads, 0)
        self.urls: list[str] = []

    def __call__(self, url: str) -> Sequence[object]:
        self.urls.append(url)
        index = self.calls[url]
        self.calls[url] = index + 1
        error = self.raises.get(url, {}).get(index)
        if error is not None:
            raise error
        return self.payloads[url][index]


# Ten rounds of scripted all-symbol snapshots. Index price 100 for BTCUSDT and
# 2000 for ETHUSDT, so the basis in bps is (mark - index) / index * 10 000.
BTC_MARKS = (
    "100.1", "100.2", "100.3", "100.4", "100.5", "100.6", "100.7", "100.8", "100.9", "101",
)
BTC_FUNDING = (
    "0.0001", "0.0002", "0.0003", "0.0004", "0.0005",
    "0.0006", "0.0007", "0.0008", "0.0009", "0.0010",
)
ETH_MARKS = (
    "2000", "2000.2", "2000.4", "2000.6", "2000.8",
    "2001", "2001.2", "2001.4", "2001.6", "2001.8",
)
# Mid 100 in every round; the two prices are 0.01 * (round + 1) apart, so the
# spread in bps is the round number plus one.
BTC_PERP_BOOKS = (
    ("99.995", "100.005"), ("99.990", "100.010"), ("99.985", "100.015"),
    ("99.980", "100.020"), ("99.975", "100.025"), ("99.970", "100.030"),
    ("99.965", "100.035"), ("99.960", "100.040"), ("99.955", "100.045"),
    ("99.950", "100.050"),
)
SCRIPTED_ROUNDS = 10


def scripted_premium(index: int) -> list[object]:
    rows: list[object] = [
        {
            "symbol": "BTCUSDT",
            "markPrice": BTC_MARKS[index],
            "indexPrice": "100",
            "lastFundingRate": BTC_FUNDING[index],
            "nextFundingTime": 1_757_001_600_000,
        },
        {
            "symbol": "ETHUSDT",
            "markPrice": ETH_MARKS[index],
            "indexPrice": "2000",
            "lastFundingRate": "-0.00005",
            "nextFundingTime": 1_757_001_600_000,
        },
    ]
    if index == 2:
        # A delisted perpetual: it parses, measures nothing and is excluded.
        rows.append(
            {
                "symbol": "LUNAUSDT",
                "markPrice": "1",
                "indexPrice": "0",
                "lastFundingRate": "0",
                "nextFundingTime": 1_757_001_600_000,
            }
        )
    return rows


def scripted_perp_book(index: int) -> list[object]:
    bid, ask = BTC_PERP_BOOKS[index]
    rows: list[object] = [
        {"symbol": "BTCUSDT", "bidPrice": bid, "bidQty": "3", "askPrice": ask, "askQty": "4"}
    ]
    if index != 4:
        # ETHUSDT is quoted in the premium index every round and missing from
        # the book in round 4, so its two round counts differ.
        rows.append(
            {
                "symbol": "ETHUSDT",
                "bidPrice": "1999",
                "bidQty": "1.5",
                "askPrice": "2001",
                "askQty": "2.5",
            }
        )
    return rows


def scripted_spot_book(index: int) -> list[object]:
    rows: list[object] = [
        {
            "symbol": "BTCUSDT",
            "bidPrice": "99.95",
            "bidQty": "2",
            "askPrice": "100.05",
            "askQty": "2",
        }
    ]
    if index in (0, 1, 8, 9):
        # Quoted outside the window only, so the window's map never names it.
        rows.append(
            {
                "symbol": "DOGEUSDT",
                "bidPrice": "3",
                "bidQty": "10",
                "askPrice": "4",
                "askQty": "20",
            }
        )
    if index == 5:
        rows.append(
            {"symbol": "XRPUSDT", "bidPrice": "4", "bidQty": "1", "askPrice": "3", "askQty": "1"}
        )
    return rows


def scripted_journal(tmp_path: Path) -> Path:
    """Ten rounds, every one of them a snapshot round, with a scripted venue."""
    journal = measurement_journal(tmp_path)
    reseal_measurement_spec(journal, snapshot_every_rounds=1)
    run_measurement_journal(
        workspace_root=tmp_path,
        journal_root=journal,
        reserve_bytes=0,
        rounds=SCRIPTED_ROUNDS,
        fetcher=FakeVenue(),
        array_fetcher=ScriptedSnapshotVenue(
            payloads={
                PREMIUM_INDEX_URL: [scripted_premium(i) for i in range(SCRIPTED_ROUNDS)],
                PERP_BOOK_TICKER_URL: [scripted_perp_book(i) for i in range(SCRIPTED_ROUNDS)],
                SPOT_BOOK_TICKER_URL: [scripted_spot_book(i) for i in range(SCRIPTED_ROUNDS)],
            },
            # The spot endpoint is throttled in round 3, inside the window.
            raises={SPOT_BOOK_TICKER_URL: {3: BinanceCostJournalTransportError(429, 0)}},
        ),
        clock=FakeClock(),
        sleep=FakeSleep(),
    )
    return journal


def stamps_of(journal_root: Path) -> list[int]:
    """Every segment's `received_time_ns`, in sequence order."""
    stamps: list[int] = []
    for document in segment_documents(journal_root):
        stamp = document["received_time_ns"]
        assert isinstance(stamp, int)
        stamps.append(stamp)
    return stamps


def take_snapshot(
    tmp_path: Path, journal: Path, *, first: int, last: int, output: str = "snapshot.json"
) -> dict[str, object]:
    stamps = stamps_of(journal)
    path = snapshot_measurement_journal(
        workspace_root=tmp_path,
        journal_root=journal,
        output_path=tmp_path / output,
        reserve_bytes=0,
        window_start_ns=stamps[first],
        window_end_ns=stamps[last],
    )
    return read_document(path)


def test_a_snapshot_over_rounds_two_to_seven_equals_a_hand_computation(
    tmp_path: Path,
) -> None:
    journal = scripted_journal(tmp_path)
    document = take_snapshot(tmp_path, journal, first=2, last=7)
    stamps = stamps_of(journal)
    head = read_document(journal / "chain-head.json")
    _, spec_hash = load_measurement_journal_spec(journal)
    assert document["version"] == MEASUREMENT_SNAPSHOT_VERSION
    assert document["spec_hash"] == spec_hash
    assert document["chain_head_hash"] == head["content_hash"]
    assert (document["window_start_ns"], document["window_end_ns"]) == (stamps[2], stamps[7])
    assert (document["first_sequence"], document["last_sequence"]) == (2, 7)
    assert (document["rounds"], document["snapshot_rounds"]) == (6, 6)
    assert document["failures"] == {
        "premiumIndex": 0,
        "perpBookTicker": 0,
        "spotBookTicker": 1,
    }
    assert document["excluded"] == {
        "premiumIndex": 1,
        "perpBookTicker": 0,
        "spotBookTicker": 1,
    }
    # BTCUSDT: basis 30..80 bps over the six rounds, mean 55; funding
    # 0.0003..0.0008, mean 0.00055, last the round-7 value; spread 3..8 bps,
    # mean 5.5. ETHUSDT: basis 2..7 bps, mean 4.5; a constant 10 bps spread in
    # the five rounds its book was quoted in.
    assert document["perpetuals"] == {
        "BTCUSDT": {
            "premium_rounds": 6,
            "book_rounds": 6,
            "mean_last_funding_rate": "0.00055000",
            "last_funding_rate": "0.0008",
            "mean_basis_bps": "55.000000",
            "mean_spread_bps": "5.500000",
        },
        "ETHUSDT": {
            "premium_rounds": 6,
            "book_rounds": 5,
            "mean_last_funding_rate": "-0.00005000",
            "last_funding_rate": "-0.00005",
            "mean_basis_bps": "4.500000",
            "mean_spread_bps": "10.000000",
        },
    }
    # The spot endpoint failed in round 3, so five of the six rounds carry a
    # book; DOGEUSDT was quoted outside the window only and XRPUSDT was
    # excluded, so neither is named.
    assert document["spot"] == {
        "BTCUSDT": {"book_rounds": 5, "mean_spread_bps": "10.000000"},
    }
    material = {key: value for key, value in document.items() if key != "content_hash"}
    assert document["content_hash"] == content_sha256(material)


# A fixed creation stamp, so the journal spec - and with it every hash that
# names it - is the same on every run of the pin below.
PINNED_CREATION_NS = 1_790_000_000_000_000_000
# What rounds two to seven of `scripted_journal` seal, taken from the reading
# that held every segment of the window in memory at once. Streaming the
# window may not move a digit of it.
SCRIPTED_WINDOW_CONTENT_HASH = "607f8c37bf9ab8576018a0f566f7a60e80ad48b45c3089dab7a7535dd2d97cbe"


def test_a_windows_document_is_the_same_reading_however_it_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The document's seal is the pin: the same rounds, the same numbers, the
    same bytes, whether the window is held in memory or folded one segment at
    a time."""
    monkeypatch.setattr(time, "time_ns", lambda: PINNED_CREATION_NS)
    journal = scripted_journal(tmp_path)

    document = take_snapshot(tmp_path, journal, first=2, last=7)

    assert document["content_hash"] == SCRIPTED_WINDOW_CONTENT_HASH


def held_window_segments(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """How many of a window's segments are alive at each read, in read order.

    The parsed model is the expensive half of a segment - 2.4 MB of rows on a
    snapshot round of the real stream - so a weak reference to it is alive
    exactly while the reading is still holding that segment.
    """
    alive: list[weakref.ReferenceType[MeasurementSegment]] = []
    held: list[int] = []
    read_segment = binance_measurement_journal._read_segment

    def counting(journal_root: Path, path: Path) -> _ReadSegment:
        read = read_segment(journal_root, path)
        alive.append(weakref.ref(read.segment))
        gc.collect()
        held.append(sum(1 for reference in alive if reference() is not None))
        return read

    monkeypatch.setattr(binance_measurement_journal, "_read_segment", counting)
    return held


def test_a_snapshot_holds_at_most_two_of_the_windows_segments_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A week of the real stream is ~9 900 segments, ~2 000 of them 2.4 MB once
    parsed: a reading that kept them all would need more than 5 GB of memory to
    seal one weekly snapshot. The window is folded segment by segment instead -
    the one before the current one is still in hand, because that is what the
    chain link is checked against, and nothing older."""
    journal = scripted_journal(tmp_path)
    held = held_window_segments(monkeypatch)

    document = take_snapshot(tmp_path, journal, first=0, last=SCRIPTED_ROUNDS - 1)

    assert document["rounds"] == SCRIPTED_ROUNDS
    assert len(held) == SCRIPTED_ROUNDS
    assert max(held) <= 2


def decimal_or_none(value: object) -> str | None:
    return None if value is None else str(value)


def test_a_snapshot_s_cost_instruments_are_v1_s_arithmetic_over_the_window(
    tmp_path: Path,
) -> None:
    journal = scripted_journal(tmp_path)
    document = take_snapshot(tmp_path, journal, first=2, last=7)
    spec, _ = load_measurement_journal_spec(journal)
    keys = tuple(str(notional) for notional in spec.notionals)
    measured: dict[str, list[InstrumentObservation]] = {}
    for segment in segment_documents(journal)[2:8]:
        depth = segment["depth"]
        assert isinstance(depth, list)
        for entry in depth:
            observation = InstrumentObservation.model_validate(entry)
            if observation.ok:
                measured.setdefault(observation.instrument_id, []).append(observation)
    statistics = [
        _instrument_statistics(instrument, measured.get(instrument.instrument_id, ()), keys=keys)
        for instrument in spec.instruments
    ]
    expected_instruments = {
        instrument.instrument_id: {
            "symbol": instrument.symbol,
            "market": instrument.market,
            "tier": instrument.tier,
            "observation_count": item.observation_count,
            "slippage": {
                key: {
                    "count": item.slippage[key]["count"],
                    "p50": decimal_or_none(item.slippage[key]["p50"]),
                    "p90": decimal_or_none(item.slippage[key]["p90"]),
                }
                for key in keys
            },
        }
        for instrument, item in zip(spec.instruments, statistics, strict=True)
    }
    assert document["cost_instruments"] == expected_instruments
    # The window, not the journal: ten rounds were sampled and six are read.
    assert [item.observation_count for item in statistics] == [6, 6]
    assert [item.slippage["500"]["count"] for item in statistics] == [6, 6]
    # Six rounds is under the floor, so the two declared instruments are
    # counted and no median is taken over them (ruling 14).
    assert document["tier_floor_count"] == 100
    withheld = {
        key: {"contributing_count": 0, "p50_of_p50": None, "p50_of_p90": None} for key in keys
    }
    assert document["cost_tiers"] == {
        "1:spot": {"instrument_count": 1, "slippage": withheld},
        "1:um": {"instrument_count": 1, "slippage": withheld},
        "2:spot": {"instrument_count": 0, "slippage": withheld},
        "2:um": {"instrument_count": 0, "slippage": withheld},
    }


def at(document: Mapping[str, object], *keys: str) -> object:
    """The value a path of keys reaches in a snapshot document."""
    value: object = document
    for key in keys:
        assert isinstance(value, dict)
        value = value[key]
    return value


def test_the_window_tier_floor_is_half_the_window_and_never_below_a_hundred() -> None:
    """A week of rounds clears the hundred; an afternoon of them does not."""
    assert [_tier_floor_count(rounds) for rounds in (0, 1, 8, 200, 201, 400, 9_914)] == [
        100,
        100,
        100,
        100,
        100,
        200,
        4_957,
    ]


# Five spot legs of tier one, so a tier median has members to be taken over.
TIER_SYMBOLS = ("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT")
# The level distance from the mid, in basis points, per symbol: with one deep
# level a side the slippage per side *is* that distance, at every notional.
TIER_DISTANCES = {"AAAUSDT": 1, "BBBUSDT": 2, "CCCUSDT": 3, "DDDUSDT": 4, "EEEUSDT": 50}
TIER_ROUNDS = 8
# EEEUSDT is unreachable for the first five rounds, so it is measured three
# times in an eight-round window and stays under the floor.
TIER_DARK_ROUNDS = 5


def tier_instrument_document(symbol: str) -> dict[str, object]:
    return {
        "instrument_id": f"spot:{symbol}",
        "market": "spot",
        "symbol": symbol,
        "pair_symbol": symbol,
        "tier": 1,
        "depth_url": f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=500",
        "premium_index_url": None,
    }


class TieredVenue:
    """A depth fetcher whose book is one deep level a side, per symbol.

    One level that fills every notional costs the walk the same at all three,
    so an instrument's slippage per side is exactly its level's distance from
    the mid of 100 in basis points. The last round of the window widens every
    distance tenfold, so a member's p90 over the window is not its p50; and a
    symbol named in `dark` is unreachable for its first rounds, which is how
    an instrument stays under the window floor while still being measured.
    """

    def __init__(
        self, *, distances: Mapping[str, int], dark: Mapping[str, int], rounds: int
    ) -> None:
        self.distances = dict(distances)
        self.dark = dict(dark)
        self.rounds = rounds
        self.calls: dict[str, int] = dict.fromkeys(distances, 0)

    def __call__(self, url: str) -> Mapping[str, object]:
        symbol = parse_qs(urlsplit(url).query)["symbol"][0]
        index = self.calls[symbol]
        self.calls[symbol] = index + 1
        if index < self.dark.get(symbol, 0):
            raise ConnectionError(f"{symbol} is unreachable")
        widened = 10 if index == self.rounds - 1 else 1
        offset = Decimal(self.distances[symbol] * widened) / Decimal(100)
        return {
            "lastUpdateId": 1,
            "bids": [[str(Decimal(100) - offset), "1000"]],
            "asks": [[str(Decimal(100) + offset), "1000"]],
        }


def tiered_journal(tmp_path: Path) -> Path:
    """Eight rounds over five tier-one spot legs, one of them mostly dark."""
    cost_journal = tmp_path / "cost-journal"
    write_journal_directory(
        cost_journal, instruments=[tier_instrument_document(s) for s in TIER_SYMBOLS]
    )
    journal = tmp_path / "measurement"
    create_measurement_journal(
        workspace_root=tmp_path,
        journal_root=journal,
        reserve_bytes=0,
        run_id="binance-measurement-v1",
        cost_journal_root=cost_journal,
    )
    run_measurement_journal(
        workspace_root=tmp_path,
        journal_root=journal,
        reserve_bytes=0,
        rounds=TIER_ROUNDS,
        fetcher=TieredVenue(
            distances=TIER_DISTANCES,
            dark={"EEEUSDT": TIER_DARK_ROUNDS},
            rounds=TIER_ROUNDS,
        ),
        array_fetcher=SnapshotVenue(),
        clock=FakeClock(),
        sleep=FakeSleep(),
    )
    return journal


def test_a_snapshot_s_tier_medians_are_taken_over_the_members_that_cleared_the_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ruling 14: a window floor, not the receipt's eligibility, admits a member."""
    monkeypatch.setattr(binance_measurement_journal, "_TIER_FLOOR_MINIMUM", 4)
    journal = tiered_journal(tmp_path)
    stamps = stamps_of(journal)
    document = read_document(
        snapshot_measurement_journal(
            workspace_root=tmp_path,
            journal_root=journal,
            output_path=tmp_path / "snapshot.json",
            reserve_bytes=0,
            window_start_ns=stamps[0],
            window_end_ns=stamps[-1],
        )
    )
    assert document["rounds"] == TIER_ROUNDS
    # max(4, 8 // 2) with the minimum lowered for the fixture.
    assert document["tier_floor_count"] == 4
    # Seven rounds at the symbol's own distance and one at ten times it, so
    # the p50 is the distance and the p90 is ten times it.
    assert [
        at(document, "cost_instruments", f"spot:{symbol}", "observation_count")
        for symbol in TIER_SYMBOLS
    ] == [8, 8, 8, 8, 3]
    assert [
        at(document, "cost_instruments", f"spot:{symbol}", "slippage", "500")
        for symbol in TIER_SYMBOLS
    ] == [
        {"count": 8, "p50": "1.000000", "p90": "10.000000"},
        {"count": 8, "p50": "2.000000", "p90": "20.000000"},
        {"count": 8, "p50": "3.000000", "p90": "30.000000"},
        {"count": 8, "p50": "4.000000", "p90": "40.000000"},
        # Measured three times, so it is under the floor - and still recorded.
        {"count": 3, "p50": "50.000000", "p90": "500.000000"},
    ]
    # Four contributors, so the median is reported: `_quantile` takes the
    # ceil(4 * 0.5) - 1 = index 1 of [1, 2, 3, 4] and of [10, 20, 30, 40].
    # EEEUSDT would move both (index 2 of five values) if it were admitted.
    assert at(document, "cost_tiers", "1:spot") == {
        # Every declared leg of the tier, not only the admitted ones.
        "instrument_count": 5,
        "slippage": {
            key: {
                "contributing_count": 4,
                "p50_of_p50": "2.000000",
                "p50_of_p90": "20.000000",
            }
            for key in ("500", "5000", "50000")
        },
    }
    assert at(document, "cost_tiers", "1:um", "instrument_count") == 0
    # One member short of the floor is one member short of a median.
    monkeypatch.setattr(binance_measurement_journal, "_TIER_FLOOR_MINIMUM", 9)
    second = read_document(
        snapshot_measurement_journal(
            workspace_root=tmp_path,
            journal_root=journal,
            output_path=tmp_path / "second.json",
            reserve_bytes=0,
            window_start_ns=stamps[0],
            window_end_ns=stamps[-1],
        )
    )
    assert second["tier_floor_count"] == 9
    assert at(second, "cost_tiers", "1:spot", "slippage", "500") == {
        "contributing_count": 0,
        "p50_of_p50": None,
        "p50_of_p90": None,
    }


def test_a_tier_median_needs_four_contributors_however_many_cleared_the_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v1's withholding rule is kept: three instruments wearing a tier's name are not one."""
    assert TIER_MINIMUM_CONTRIBUTORS == 4
    monkeypatch.setattr(binance_measurement_journal, "_TIER_FLOOR_MINIMUM", 4)
    cost_journal = tmp_path / "cost-journal"
    write_journal_directory(
        cost_journal, instruments=[tier_instrument_document(s) for s in TIER_SYMBOLS[:3]]
    )
    journal = tmp_path / "measurement"
    create_measurement_journal(
        workspace_root=tmp_path,
        journal_root=journal,
        reserve_bytes=0,
        run_id="binance-measurement-v1",
        cost_journal_root=cost_journal,
    )
    run_measurement_journal(
        workspace_root=tmp_path,
        journal_root=journal,
        reserve_bytes=0,
        rounds=TIER_ROUNDS,
        fetcher=TieredVenue(distances=TIER_DISTANCES, dark={}, rounds=TIER_ROUNDS),
        array_fetcher=SnapshotVenue(),
        clock=FakeClock(),
        sleep=FakeSleep(),
    )
    stamps = stamps_of(journal)
    document = read_document(
        snapshot_measurement_journal(
            workspace_root=tmp_path,
            journal_root=journal,
            output_path=tmp_path / "snapshot.json",
            reserve_bytes=0,
            window_start_ns=stamps[0],
            window_end_ns=stamps[-1],
        )
    )
    assert at(document, "cost_tiers", "1:spot") == {
        "instrument_count": 3,
        "slippage": {
            key: {"contributing_count": 3, "p50_of_p50": None, "p50_of_p90": None}
            for key in ("500", "5000", "50000")
        },
    }


def test_a_snapshot_counts_only_the_rounds_that_took_one(tmp_path: Path) -> None:
    """The cadence the spec sealed decides, so a window can hold plain rounds."""
    journal = journal_with_rounds(tmp_path, rounds=6)
    stamps = stamps_of(journal)
    path = snapshot_measurement_journal(
        workspace_root=tmp_path,
        journal_root=journal,
        output_path=tmp_path / "snapshot.json",
        reserve_bytes=0,
        window_start_ns=stamps[0],
        window_end_ns=stamps[-1],
    )
    document = read_document(path)
    assert (document["rounds"], document["snapshot_rounds"]) == (6, 2)
    assert (document["first_sequence"], document["last_sequence"]) == (0, 5)


def test_a_snapshot_refuses_an_existing_output(tmp_path: Path) -> None:
    journal = scripted_journal(tmp_path)
    stamps = stamps_of(journal)
    output = tmp_path / "snapshot.json"
    output.write_bytes(b"{}")
    with pytest.raises(BinanceMeasurementJournalSpecError, match="already exists"):
        snapshot_measurement_journal(
            workspace_root=tmp_path,
            journal_root=journal,
            output_path=output,
            reserve_bytes=0,
            window_start_ns=stamps[0],
            window_end_ns=stamps[-1],
        )


def test_a_snapshot_refuses_an_empty_and_a_reversed_window(tmp_path: Path) -> None:
    journal = scripted_journal(tmp_path)
    stamps = stamps_of(journal)
    with pytest.raises(BinanceMeasurementJournalSpecError, match="MEASUREMENT_WINDOW_EMPTY"):
        snapshot_measurement_journal(
            workspace_root=tmp_path,
            journal_root=journal,
            output_path=tmp_path / "gap.json",
            reserve_bytes=0,
            # Between two rounds of a day this journal did write.
            window_start_ns=stamps[2] + 1,
            window_end_ns=stamps[3] - 1,
        )
    with pytest.raises(BinanceMeasurementJournalSpecError, match="MEASUREMENT_WINDOW_EMPTY"):
        snapshot_measurement_journal(
            workspace_root=tmp_path,
            journal_root=journal,
            output_path=tmp_path / "before.json",
            reserve_bytes=0,
            # A day directory this journal never opened.
            window_start_ns=1,
            window_end_ns=2,
        )
    with pytest.raises(BinanceMeasurementJournalSpecError, match="ends before it starts"):
        snapshot_measurement_journal(
            workspace_root=tmp_path,
            journal_root=journal,
            output_path=tmp_path / "reversed.json",
            reserve_bytes=0,
            window_start_ns=stamps[7],
            window_end_ns=stamps[2],
        )
    assert not (tmp_path / "gap.json").exists()
    assert not (tmp_path / "before.json").exists()
    assert not (tmp_path / "reversed.json").exists()


def test_a_snapshot_refuses_a_window_whose_segments_do_not_link(tmp_path: Path) -> None:
    journal = scripted_journal(tmp_path)
    stamps = stamps_of(journal)
    rewrite_segment(segment_paths(journal)[4], previous_segment_hash="a" * 64)
    with pytest.raises(BinanceMeasurementJournalSpecError, match="CHAIN_LINK_BROKEN"):
        snapshot_measurement_journal(
            workspace_root=tmp_path,
            journal_root=journal,
            output_path=tmp_path / "snapshot.json",
            reserve_bytes=0,
            window_start_ns=stamps[2],
            window_end_ns=stamps[7],
        )
    assert not (tmp_path / "snapshot.json").exists()


def test_a_snapshot_refuses_a_chain_head_that_does_not_recompute(tmp_path: Path) -> None:
    journal = scripted_journal(tmp_path)
    stamps = stamps_of(journal)
    head = journal / "chain-head.json"
    head.write_bytes(canonical_json({**read_document(head), "segment_count": 99}))
    with pytest.raises(BinanceMeasurementJournalSpecError, match="CHAIN_HEAD_UNREADABLE"):
        snapshot_measurement_journal(
            workspace_root=tmp_path,
            journal_root=journal,
            output_path=tmp_path / "snapshot.json",
            reserve_bytes=0,
            window_start_ns=stamps[2],
            window_end_ns=stamps[7],
        )
