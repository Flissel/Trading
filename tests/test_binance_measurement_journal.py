import json
import urllib.request
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path

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
from trading_bot.binance_cost_journal import (
    SAMPLE_INTERVAL_SECONDS,
    ZERO_HASH,
    BinanceCostJournalError,
    BinanceCostJournalTransportError,
    public_binance_json_array_fetcher,
    public_binance_json_fetcher,
)
from trading_bot.binance_measurement_journal import (
    MEASUREMENT_JOURNAL_VERSION,
    PERP_BOOK_TICKER_URL,
    PREMIUM_INDEX_URL,
    SNAPSHOT_EVERY_ROUNDS,
    SPOT_BOOK_TICKER_URL,
    BinanceMeasurementJournalSpecError,
    MeasurementChainHead,
    _segment_paths,
    _verified_chain,
    book_rows,
    create_measurement_journal,
    load_measurement_journal_spec,
    premium_rows,
    run_measurement_journal,
)
from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.depth_adapters import DepthPayloadError


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
    rows = premium_rows(premium_payload())
    assert [row.symbol for row in rows] == ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
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
    perpetual = book_rows(perp_book_payload())
    assert [row.symbol for row in perpetual] == ["BTCUSDT", "ETHUSDT"]
    # mid 100, spread 0.02 -> 2 bps.
    assert perpetual[0].spread_bps == Decimal("2.000000")
    assert (perpetual[0].bid_price, perpetual[0].bid_qty) == (Decimal("99.99"), Decimal("3"))
    assert (perpetual[0].ask_price, perpetual[0].ask_qty) == (Decimal("100.01"), Decimal("4"))
    # mid 2000, spread 2 -> 10 bps.
    assert perpetual[1].spread_bps == Decimal("10.000000")
    spot = book_rows(spot_book_payload())
    assert [row.symbol for row in spot] == ["BTCUSDT", "DOGEUSDT"]
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
        # Quoted in USDT and not a venue symbol: kept scope, unreadable row.
        ({"symbol": "ETH-USDT"}, "malformed entry at index 1"),
        ({"indexPrice": "0"}, "non-positive premium index ETHUSDT"),
        ({"markPrice": "-1"}, "non-positive premium index ETHUSDT"),
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
        ({"bidQty": "0"}, "empty book ETHUSDT"),
        ({"askPrice": "0"}, "empty book ETHUSDT"),
        ({"bidPrice": "2001", "askPrice": "2000"}, "crossed book ETHUSDT"),
        ({"bidPrice": "2000", "askPrice": "2000"}, "crossed book ETHUSDT"),
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
    document = read_document(path)
    document["received_time_ns"] = 1
    path.write_bytes(canonical_json(document))
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
