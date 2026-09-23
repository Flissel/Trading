"""Tests for the weekly shadow capture (spec sections 3.1 and 3.3).

Every capture here is built from the fixture fetches -- no network. The base
captures are the carry fixtures' own captures over `MONTHS`; the tail is the
two months registered additively in `tests.shadow_fixtures`.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from decimal import Decimal
from pathlib import Path

import pytest

from tests.carry_fixtures import build_captures, funding_rate
from tests.shadow_fixtures import (
    FIRST_TAIL_DATE,
    FIRST_TAIL_SUNDAY,
    HOUR_MS,
    SECOND_TAIL_SUNDAY,
    THIRD_TAIL_SUNDAY,
    ShadowFetch,
    daily_kline_csv,
    day_end_ms,
    day_start_ms,
    funding_rest_json,
    funding_settlement_times_ms,
)
from tests.test_panel_fold_run import MONTHS, SYMBOLS, kline_csv, zip_bytes
from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.capture_lineage import verify_capture_superset
from trading_bot.panel_capture import (
    PanelCaptureError,
    PanelPayload,
    PanelSourceAbsent,
    build_kline_zip_url,
    capture_panel,
    parse_kline_zip,
    verify_panel_capture,
)
from trading_bot.panel_dataset import DAY_NS, PanelCandleRow, PanelFundingRow
from trading_bot.panel_reader import load_funding_events, load_panel_bars
from trading_bot.shadow_capture import (
    DAILY_TAIL_KIND,
    FUNDING_REST_KIND,
    ShadowCaptureArtifact,
    ShadowCaptureError,
    ShadowCaptureTransportError,
    build_funding_rest_url,
    build_shadow_capture,
    parse_funding_rest,
    reconcile_tail_rows,
)

_SYMBOL = SYMBOLS[0]
_NANOSECONDS_PER_MILLISECOND = 1_000_000
_BUCKETS = ("klines", "fundingRate")


@pytest.fixture(autouse=True)
def _fixed_wall_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Freeze wall-clock time for every test in this module.

    `zipfile.ZipFile.writestr` stamps each archive member with
    `time.localtime(time.time())`, so two fixture zips over the same content
    are byte-identical -- and therefore carry the same `raw_sha256` -- only if
    they do not straddle that clock's tick. Freezing it removes that race from
    every comparison across two separately built captures.
    """
    monkeypatch.setattr("time.time", lambda: 1_700_000_000.0)


@dataclass
class FakeClock:
    """A clock that moves only when something sleeps on it, so the Sunday
    deadline is exercised in full without waiting a real day."""

    now_ns: int = 1_600_000_000_000_000_000
    slept: list[float] = field(default_factory=list)

    def time_ns(self) -> int:
        return self.now_ns

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now_ns += int(seconds * 1_000_000_000)


def _base_capture(
    root: Path,
    *,
    months: tuple[str, ...] = MONTHS,
    fetch: ShadowFetch | None = None,
    name: str | None = None,
) -> Path:
    """A perpetual base capture over `months`, built by the shadow fetch.

    Byte for byte the capture `tests.carry_fixtures.build_captures` builds for
    the same months: both serve the monthly URLs from the same generators.
    """
    download = fetch if fetch is not None else ShadowFetch()
    return capture_panel(
        workspace_root=root,
        output_directory=root / (name or f"base-{months[-1]}"),
        reserve_bytes=0,
        symbols=SYMBOLS,
        months=months,
        fetch=download.fetch,
    ).capture_root


def _shadow(
    root: Path,
    base: Path,
    *,
    fetch: ShadowFetch,
    tail_through: str = FIRST_TAIL_SUNDAY,
    market: str = "um",
    previous: Path | None = None,
    name: str = "shadow",
    clock: FakeClock | None = None,
) -> ShadowCaptureArtifact:
    ticker = clock if clock is not None else FakeClock()
    return build_shadow_capture(
        workspace_root=root,
        base_capture_root=base,
        output_directory=root / name,
        reserve_bytes=0,
        tail_through=tail_through,
        market=market,
        fetch=fetch.fetch,
        previous_capture_root=previous,
        clock=ticker.time_ns,
        sleep=ticker.sleep,
    )


def _manifest(capture_root: Path) -> dict[str, object]:
    document: dict[str, object] = json.loads(
        (capture_root / "capture-manifest.json").read_text(encoding="utf-8")
    )
    return document


def _rows(capture_root: Path, kind: str) -> list[dict[str, object]]:
    sources = _manifest(capture_root)["sources"]
    assert isinstance(sources, list)
    return [entry for entry in sources if entry["kind"] == kind]


def _quality(capture_root: Path) -> dict[str, dict[str, object]]:
    document = json.loads(
        (capture_root / "dataset" / "quality-report.json").read_text(encoding="utf-8")
    )
    return {str(item["instrument_id"]): item for item in document["instruments"]}


def _reseal(capture_root: Path, mutate: Callable[[dict[str, object]], None]) -> None:
    """Rewrite a capture's manifest after `mutate`, so it still verifies.

    A hand-edited manifest that no longer matches its own seal would be
    refused by the verification step long before the field under test is
    read, so the seal is recomputed over the mutated material.
    """
    material = {k: v for k, v in _manifest(capture_root).items() if k != "capture_root_hash"}
    mutate(material)
    document = dict(material)
    document["capture_root_hash"] = content_sha256(material)
    (capture_root / "capture-manifest.json").write_bytes(canonical_json(document))


def _daily_urls(fetch: ShadowFetch) -> list[str]:
    return [url for url in fetch.urls if "/daily/klines/" in url]


def test_a_shadow_capture_extends_a_verified_base_through_the_sunday(tmp_path: Path) -> None:
    base, _spot = build_captures(tmp_path)
    fetch = ShadowFetch()

    artifact = _shadow(tmp_path, base, fetch=fetch)

    assert verify_panel_capture(artifact.capture_root) == (True, ())
    assert verify_capture_superset(base, artifact.capture_root) == (True, ())
    assert artifact.capture_root == tmp_path / "shadow"
    assert artifact.tail_through == FIRST_TAIL_SUNDAY
    assert artifact.reconciliation == {"compared_rows": 0, "mismatches": []}

    manifest = _manifest(artifact.capture_root)
    assert manifest["capture_root_hash"] == artifact.capture_root_hash
    assert manifest["dataset_root_hash"] == artifact.dataset_root_hash
    assert manifest["base_capture_root_hash"] == _manifest(base)["capture_root_hash"]
    assert manifest["previous_capture_root_hash"] is None
    assert manifest["tail_through"] == FIRST_TAIL_SUNDAY
    assert manifest["reconciliation"] == {"compared_rows": 0, "mismatches": []}
    assert manifest["stale_symbols"] == {"klines": {}, "fundingRate": {}}
    assert manifest["months"] is None
    assert manifest["symbols"] == list(SYMBOLS)
    assert manifest["venue"] == "BINANCE_UM"
    assert manifest["interval"] == "1d"
    discovered = manifest["discovered_months"]
    assert isinstance(discovered, dict)
    assert discovered[_SYMBOL] == {
        "klines": [*MONTHS, "2020-08"],
        "fundingRate": [*MONTHS, "2020-08"],
    }

    # Two tail days for each of the twelve symbols, and one REST window each.
    tail_rows = _rows(artifact.capture_root, DAILY_TAIL_KIND)
    assert len(tail_rows) == 2 * len(SYMBOLS)
    assert {str(row["month"]) for row in tail_rows} == {FIRST_TAIL_DATE, FIRST_TAIL_SUNDAY}
    assert {str(row["status"]) for row in tail_rows} == {"present"}
    rest_rows = _rows(artifact.capture_root, FUNDING_REST_KIND)
    assert len(rest_rows) == len(SYMBOLS)
    assert {str(row["month"]) for row in rest_rows} == {
        f"{FIRST_TAIL_DATE}_{FIRST_TAIL_SUNDAY}"
    }

    bars = {
        (bar.instrument_id, bar.open_time_ns): bar
        for bar in load_panel_bars(artifact.capture_root / "dataset")
    }
    for date in (FIRST_TAIL_DATE, FIRST_TAIL_SUNDAY):
        bar = bars[(_SYMBOL, day_start_ms(date) * _NANOSECONDS_PER_MILLISECOND)]
        assert bar.close_time_ns == day_end_ms(date) * _NANOSECONDS_PER_MILLISECOND
        assert bar.available_time_ns == bar.close_time_ns + 1

    settlements = {
        (event.instrument_id, event.calc_time_ns): event.rate
        for event in load_funding_events(artifact.capture_root / "dataset")
    }
    tail_settlement_ns = day_start_ms(FIRST_TAIL_DATE) * _NANOSECONDS_PER_MILLISECOND
    for symbol in SYMBOLS:
        assert settlements[(symbol, tail_settlement_ns)] == Decimal(funding_rate(symbol))


def test_a_daily_tail_bar_equals_the_monthly_dumps_bar_for_the_same_date() -> None:
    monthly = parse_kline_zip(
        PanelPayload(
            url=build_kline_zip_url(_SYMBOL, "2020-08"),
            raw_bytes=zip_bytes("k.csv", kline_csv(_SYMBOL, "2020-08")),
            received_time_ns=1,
        ),
        symbol=_SYMBOL,
    )
    by_open = {row.open_time_ns: row for row in monthly}

    for date in (FIRST_TAIL_DATE, FIRST_TAIL_SUNDAY):
        daily = parse_kline_zip(
            PanelPayload(
                url=f"https://data.binance.vision/data/futures/um/daily/klines/"
                f"{_SYMBOL}/1d/{_SYMBOL}-1d-{date}.zip",
                raw_bytes=zip_bytes("k.csv", daily_kline_csv(_SYMBOL, date)),
                received_time_ns=1,
            ),
            symbol=_SYMBOL,
        )
        assert len(daily) == 1
        expected = by_open[day_start_ms(date) * _NANOSECONDS_PER_MILLISECOND]
        # Only the payload the row came out of differs -- every measured field
        # of the bar itself has to be identical, or reconciliation is a lie.
        assert replace(daily[0], source_payload_hash="") == replace(
            expected, source_payload_hash=""
        )


def test_an_absent_tail_date_is_recorded_and_the_capture_still_publishes(
    tmp_path: Path,
) -> None:
    base = _base_capture(tmp_path)
    fetch = ShadowFetch(absent_dates=frozenset({(_SYMBOL, FIRST_TAIL_DATE)}))

    artifact = _shadow(tmp_path, base, fetch=fetch)

    assert verify_panel_capture(artifact.capture_root) == (True, ())
    absent = [
        row
        for row in _rows(artifact.capture_root, DAILY_TAIL_KIND)
        if row["status"] == "absent"
    ]
    assert absent == [
        {
            "symbol": _SYMBOL,
            "month": FIRST_TAIL_DATE,
            "kind": DAILY_TAIL_KIND,
            "url": (
                f"https://data.binance.vision/data/futures/um/daily/klines/"
                f"{_SYMBOL}/1d/{_SYMBOL}-1d-{FIRST_TAIL_DATE}.zip"
            ),
            "status": "absent",
        }
    ]
    bars = {
        (bar.instrument_id, bar.open_time_ns)
        for bar in load_panel_bars(artifact.capture_root / "dataset")
    }
    assert (_SYMBOL, day_start_ms(FIRST_TAIL_DATE) * _NANOSECONDS_PER_MILLISECOND) not in bars
    assert (_SYMBOL, day_start_ms(FIRST_TAIL_SUNDAY) * _NANOSECONDS_PER_MILLISECOND) in bars


def test_a_missing_sunday_dump_waits_a_day_and_then_refuses_naming_the_symbol(
    tmp_path: Path,
) -> None:
    base = _base_capture(tmp_path)
    fetch = ShadowFetch(absent_dates=frozenset({(_SYMBOL, FIRST_TAIL_SUNDAY)}))
    clock = FakeClock()

    with pytest.raises(ShadowCaptureError) as error:
        _shadow(tmp_path, base, fetch=fetch, clock=clock)

    assert f"SUNDAY_DUMP_MISSING:{_SYMBOL}" in str(error.value)
    assert clock.slept == [3600] * 24


def test_a_late_sunday_dump_is_taken_once_it_appears(tmp_path: Path) -> None:
    base = _base_capture(tmp_path)
    fetch = ShadowFetch(appears_after_attempts={(_SYMBOL, FIRST_TAIL_SUNDAY): 3})
    clock = FakeClock()

    artifact = _shadow(tmp_path, base, fetch=fetch, clock=clock)

    assert clock.slept == [3600] * 3
    assert verify_panel_capture(artifact.capture_root) == (True, ())
    bars = {
        (bar.instrument_id, bar.open_time_ns)
        for bar in load_panel_bars(artifact.capture_root / "dataset")
    }
    assert (_SYMBOL, day_start_ms(FIRST_TAIL_SUNDAY) * _NANOSECONDS_PER_MILLISECOND) in bars


def test_a_daily_dump_carrying_another_day_refuses(tmp_path: Path) -> None:
    base = _base_capture(tmp_path)
    fetch = ShadowFetch(wrong_day_dates=frozenset({(_SYMBOL, FIRST_TAIL_DATE)}))

    with pytest.raises(ShadowCaptureError) as error:
        _shadow(tmp_path, base, fetch=fetch)

    assert FIRST_TAIL_DATE in str(error.value)


def test_a_settlement_outside_the_requested_window_is_dropped(tmp_path: Path) -> None:
    base = _base_capture(tmp_path)
    fetch = ShadowFetch(settlements_outside_window=True)

    artifact = _shadow(tmp_path, base, fetch=fetch)

    settlement_times = {
        event.calc_time_ns
        for event in load_funding_events(artifact.capture_root / "dataset")
        if event.instrument_id == _SYMBOL
    }
    window_start_ms = day_start_ms(FIRST_TAIL_DATE)
    assert max(settlement_times) == window_start_ms * _NANOSECONDS_PER_MILLISECOND
    assert (window_start_ms - 1) * _NANOSECONDS_PER_MILLISECOND not in settlement_times


def test_a_repeated_funding_time_refuses(tmp_path: Path) -> None:
    base = _base_capture(tmp_path)
    fetch = ShadowFetch(duplicate_settlement=True)

    with pytest.raises(ShadowCaptureError) as error:
        _shadow(tmp_path, base, fetch=fetch)

    assert "fundingTime" in str(error.value)


def test_an_unverifiable_base_capture_refuses(tmp_path: Path) -> None:
    base = _base_capture(tmp_path)
    (base / "raw" / _SYMBOL / "klines-2020-01.zip").write_bytes(b"not the payload")

    with pytest.raises(ShadowCaptureError) as error:
        _shadow(tmp_path, base, fetch=ShadowFetch())

    assert "base capture failed verification" in str(error.value)


def test_a_tail_that_does_not_reach_past_the_base_refuses(tmp_path: Path) -> None:
    base = _base_capture(tmp_path)

    with pytest.raises(ShadowCaptureError) as error:
        _shadow(tmp_path, base, fetch=ShadowFetch(), tail_through="2020-07-15")

    assert "2020-07-15" in str(error.value)


def test_an_existing_capture_manifest_refuses(tmp_path: Path) -> None:
    base = _base_capture(tmp_path)
    _shadow(tmp_path, base, fetch=ShadowFetch())

    with pytest.raises(ShadowCaptureError) as error:
        _shadow(tmp_path, base, fetch=ShadowFetch())

    assert "immutable" in str(error.value)


def test_a_spot_shadow_capture_never_asks_for_funding(tmp_path: Path) -> None:
    _perp, spot = build_captures(tmp_path)
    fetch = ShadowFetch()

    artifact = _shadow(tmp_path, spot, fetch=fetch, market="spot")

    assert verify_panel_capture(artifact.capture_root) == (True, ())
    assert verify_capture_superset(spot, artifact.capture_root) == (True, ())
    assert not any("fundingRate" in url for url in fetch.urls)
    assert _rows(artifact.capture_root, FUNDING_REST_KIND) == []
    assert _manifest(artifact.capture_root)["venue"] == "BINANCE_SPOT"
    assert _manifest(artifact.capture_root)["stale_symbols"] == {"klines": {}}
    assert len(_rows(artifact.capture_root, DAILY_TAIL_KIND)) == 2 * len(SYMBOLS)


def test_the_previous_weeks_tail_rows_are_carried_and_never_refetched(
    tmp_path: Path,
) -> None:
    base = _base_capture(tmp_path)
    first = _shadow(tmp_path, base, fetch=ShadowFetch(), name="week-1")
    fetch = ShadowFetch()

    second = _shadow(
        tmp_path,
        base,
        fetch=fetch,
        tail_through=SECOND_TAIL_SUNDAY,
        previous=first.capture_root,
        name="week-2",
    )

    assert verify_panel_capture(second.capture_root) == (True, ())
    assert verify_capture_superset(base, second.capture_root) == (True, ())
    assert verify_capture_superset(first.capture_root, second.capture_root) == (True, ())
    manifest = _manifest(second.capture_root)
    assert manifest["previous_capture_root_hash"] == first.capture_root_hash
    # The two carried days plus the seven the second week fetches.
    assert len(_rows(second.capture_root, DAILY_TAIL_KIND)) == 9 * len(SYMBOLS)
    assert {str(row["month"]) for row in _rows(second.capture_root, FUNDING_REST_KIND)} == {
        f"{FIRST_TAIL_DATE}_{FIRST_TAIL_SUNDAY}",
        f"2020-08-03_{SECOND_TAIL_SUNDAY}",
    }
    for date in (FIRST_TAIL_DATE, FIRST_TAIL_SUNDAY):
        assert not any(f"-1d-{date}.zip" in url for url in _daily_urls(fetch))
    assert any(f"-1d-{SECOND_TAIL_SUNDAY}.zip" in url for url in _daily_urls(fetch))


# Week 1 tails 2020-08-01 and 2020-08-02 and asks for one funding window over
# the two, which holds the single settlement of 2020-08-01: three rows per
# symbol that an advanced base then covers with its own 2020-08 monthly dumps,
# and therefore three rows per symbol that every chain below reconciles.
_DROPPED_ROWS = 3 * len(SYMBOLS)


def test_previous_tail_rows_the_base_now_covers_are_dropped(tmp_path: Path) -> None:
    base = _base_capture(tmp_path)
    first = _shadow(tmp_path, base, fetch=ShadowFetch(), name="week-1")
    advanced = _base_capture(tmp_path, months=(*MONTHS, "2020-08"))
    fetch = ShadowFetch()

    second = _shadow(
        tmp_path,
        advanced,
        fetch=fetch,
        tail_through=THIRD_TAIL_SUNDAY,
        previous=first.capture_root,
        name="week-3",
    )

    assert verify_panel_capture(second.capture_root) == (True, ())
    assert verify_capture_superset(advanced, second.capture_root) == (True, ())
    # August is the base's own month now: its daily-tail rows are gone, and so
    # is the REST window that started inside it.
    tail_months = {str(row["month"])[:7] for row in _rows(second.capture_root, DAILY_TAIL_KIND)}
    assert tail_months == {"2020-09"}
    assert {str(row["month"]) for row in _rows(second.capture_root, FUNDING_REST_KIND)} == {
        f"2020-09-01_{THIRD_TAIL_SUNDAY}"
    }
    assert not any("-1d-2020-08-" in url for url in _daily_urls(fetch))
    # September's six days for each symbol, the first of them the day after
    # the base's last month.
    assert len(_rows(second.capture_root, DAILY_TAIL_KIND)) == 6 * len(SYMBOLS)
    # Dropped, but only after being compared: two daily bars and one
    # settlement per symbol, all of them equal to the monthly dump's own row.
    assert second.reconciliation == {"compared_rows": _DROPPED_ROWS, "mismatches": []}
    assert _manifest(second.capture_root)["reconciliation"] == {
        "compared_rows": _DROPPED_ROWS,
        "mismatches": [],
    }


def _refusal(root: Path, name: str) -> dict[str, object]:
    document: dict[str, object] = json.loads(
        (root / f"{name}-reconciliation-refused.json").read_text(encoding="utf-8")
    )
    return document


def _monthly_close(symbol: str, date: str) -> str:
    """The close the monthly dump carries for `date`, as the record spells it."""
    return str(Decimal(daily_kline_csv(symbol, date).splitlines()[1].split(",")[4]))


def test_an_altered_tail_bar_refuses_and_leaves_a_readable_refusal(tmp_path: Path) -> None:
    base = _base_capture(tmp_path)
    altered = _shadow(
        tmp_path,
        base,
        fetch=ShadowFetch(altered_closes={(_SYMBOL, FIRST_TAIL_DATE): "999.5"}),
        name="week-1",
    )
    # The altered week is a perfectly sealed capture -- the daily dump it was
    # built from said 999.5 and its manifest says so too. Only the monthly
    # dump that arrives later disagrees.
    assert verify_panel_capture(altered.capture_root) == (True, ())
    advanced = _base_capture(tmp_path, months=(*MONTHS, "2020-08"))

    with pytest.raises(ShadowCaptureError) as error:
        _shadow(
            tmp_path,
            advanced,
            fetch=ShadowFetch(),
            tail_through=THIRD_TAIL_SUNDAY,
            previous=altered.capture_root,
            name="week-3",
        )

    assert str(error.value).startswith("RECONCILIATION_MISMATCH")
    # Nothing was published and nothing was half-published: a rerun after the
    # data problem is fixed must not trip over this week's leftovers.
    assert not (tmp_path / "week-3").exists()

    refusal = _refusal(tmp_path, "week-3")
    assert refusal["base_capture_root_hash"] == _manifest(advanced)["capture_root_hash"]
    assert refusal["previous_capture_root_hash"] == altered.capture_root_hash
    assert refusal["compared_rows"] == _DROPPED_ROWS
    assert refusal["mismatches"] == [
        {
            "kind": "candle",
            "instrument_id": _SYMBOL,
            "key": day_start_ms(FIRST_TAIL_DATE) * _NANOSECONDS_PER_MILLISECOND,
            # The same key a person can read: the bar's own UTC open.
            "key_utc": f"{FIRST_TAIL_DATE}T00:00:00Z",
            "field": "close",
            "previous": "999.5",
            "base": _monthly_close(_SYMBOL, FIRST_TAIL_DATE),
        }
    ]
    sealed = {key: value for key, value in refusal.items() if key != "content_sha256"}
    assert refusal["content_sha256"] == content_sha256(sealed)


def test_a_previous_settlement_the_base_never_had_is_missing_in_base(tmp_path: Path) -> None:
    base = _base_capture(tmp_path)
    # Eight hours after the fixture's weekly settlement, so the answer's
    # spacing is still a whole number of hours and week 1 builds cleanly.
    extra_ms = day_start_ms(FIRST_TAIL_DATE) + 8 * HOUR_MS
    first = _shadow(
        tmp_path,
        base,
        fetch=ShadowFetch(extra_settlements={_SYMBOL: (extra_ms,)}),
        name="week-1",
    )
    advanced = _base_capture(tmp_path, months=(*MONTHS, "2020-08"))

    with pytest.raises(ShadowCaptureError) as error:
        _shadow(
            tmp_path,
            advanced,
            fetch=ShadowFetch(),
            tail_through=THIRD_TAIL_SUNDAY,
            previous=first.capture_root,
            name="week-3",
        )

    assert str(error.value).startswith("RECONCILIATION_MISMATCH")
    assert not (tmp_path / "week-3").exists()
    refusal = _refusal(tmp_path, "week-3")
    # The extra settlement is one more row to compare than the clean chain.
    assert refusal["compared_rows"] == _DROPPED_ROWS + 1
    assert refusal["mismatches"] == [
        {
            "kind": "missing_in_base",
            "instrument_id": _SYMBOL,
            "key": extra_ms * _NANOSECONDS_PER_MILLISECOND,
            "key_utc": f"{FIRST_TAIL_DATE}T08:00:00Z",
            "field": "funding",
            "previous": "present",
            "base": "absent",
        }
    ]


_RECONCILE_CANDLE = PanelCandleRow(
    venue="BINANCE_UM",
    instrument_id=_SYMBOL,
    open_time_ns=DAY_NS,
    close_time_ns=2 * DAY_NS - 1,
    available_time_ns=2 * DAY_NS,
    interval_ns=DAY_NS,
    open=Decimal("100"),
    high=Decimal("110"),
    low=Decimal("90"),
    close=Decimal("105"),
    base_volume=Decimal("10"),
    quote_volume=Decimal("1000"),
    trade_count=7,
    source_payload_hash="from-the-daily-dump",
)
_RECONCILE_FUNDING = PanelFundingRow(
    venue="BINANCE_UM",
    instrument_id=_SYMBOL,
    calc_time_ns=DAY_NS,
    funding_interval_hours=8,
    rate=Decimal("0.0001"),
)


# The Sunday after `THIRD_TAIL_SUNDAY`, on the fixture's own calendar.
_FOURTH_TAIL_SUNDAY = "2020-09-13"
_AUGUST_DAYS = 31


def test_a_straddling_funding_window_is_reconciled_up_to_the_covered_month(
    tmp_path: Path,
) -> None:
    """A dropped REST window is compared as far as the base reaches, and no further.

    Week 1's one funding window runs from the first of August into September.
    Week 2's base has since covered August and nothing of September, so the
    window is dropped whole -- the monthly dump wins for the days it covers --
    but only its August settlements have a monthly dump to be compared
    against. The September ones are neither compared nor counted, and the days
    behind them are refetched by this week's own set-difference plan rather
    than quietly lost with the row.
    """
    base = _base_capture(tmp_path)
    first = _shadow(
        tmp_path, base, fetch=ShadowFetch(), tail_through=THIRD_TAIL_SUNDAY, name="week-1"
    )
    straddling = f"{FIRST_TAIL_DATE}_{THIRD_TAIL_SUNDAY}"
    assert {str(row["month"]) for row in _rows(first.capture_root, FUNDING_REST_KIND)} == {
        straddling
    }

    advanced = _base_capture(tmp_path, months=(*MONTHS, "2020-08"))
    fetch = ShadowFetch()
    second = _shadow(
        tmp_path,
        advanced,
        fetch=fetch,
        tail_through=_FOURTH_TAIL_SUNDAY,
        previous=first.capture_root,
        name="week-2",
    )

    assert verify_panel_capture(second.capture_root) == (True, ())
    assert verify_capture_superset(advanced, second.capture_root) == (True, ())
    # Per symbol: August's 31 daily bars and the window's August settlements.
    august = len(
        funding_settlement_times_ms(day_start_ms("2020-08-01"), day_end_ms("2020-08-31"))
    )
    assert second.reconciliation["compared_rows"] == len(SYMBOLS) * (_AUGUST_DAYS + august)
    # The whole window is gone, so September is asked for again from its first
    # day -- not from the day after the window happened to end.
    assert {str(row["month"]) for row in _rows(second.capture_root, FUNDING_REST_KIND)} == {
        f"2020-09-01_{_FOURTH_TAIL_SUNDAY}"
    }
    # The September daily bars the window straddled are carried, not refetched.
    assert _requested_dates(fetch, _SYMBOL) == {
        f"2020-09-{day:02d}" for day in range(7, 14)
    }


def test_reconcile_tail_rows_ignores_what_the_two_sources_cannot_share() -> None:
    """The fields that differ by construction are not evidence of anything.

    The payload hash is the dump's, the available time is derived from it,
    the REST interval is read off the spacing while the monthly dump has its
    own column, and `100.00` and `100` are the same price written twice.
    """
    compared, mismatches = reconcile_tail_rows(
        previous_rows=(_RECONCILE_CANDLE, _RECONCILE_FUNDING),
        base_rows=(
            replace(
                _RECONCILE_CANDLE,
                source_payload_hash="from-the-monthly-dump",
                available_time_ns=0,
                open=Decimal("100.00"),
            ),
            replace(_RECONCILE_FUNDING, funding_interval_hours=4),
        ),
    )

    assert (compared, mismatches) == (2, ())


def test_reconcile_tail_rows_names_every_field_that_differs() -> None:
    compared, mismatches = reconcile_tail_rows(
        previous_rows=(_RECONCILE_CANDLE,),
        base_rows=(replace(_RECONCILE_CANDLE, close=Decimal("106"), trade_count=8),),
    )

    assert compared == 1
    assert mismatches == (
        {
            "kind": "candle",
            "instrument_id": _SYMBOL,
            "key": DAY_NS,
            "key_utc": "1970-01-02T00:00:00Z",
            "field": "close",
            "previous": "105",
            "base": "106",
        },
        {
            "kind": "candle",
            "instrument_id": _SYMBOL,
            "key": DAY_NS,
            "key_utc": "1970-01-02T00:00:00Z",
            "field": "trade_count",
            "previous": "7",
            "base": "8",
        },
    )


def test_reconcile_tail_rows_compares_a_settlement_on_its_rate() -> None:
    compared, mismatches = reconcile_tail_rows(
        previous_rows=(_RECONCILE_FUNDING,),
        base_rows=(replace(_RECONCILE_FUNDING, rate=Decimal("0.0002")),),
    )

    assert compared == 1
    assert mismatches == (
        {
            "kind": "funding",
            "instrument_id": _SYMBOL,
            "key": DAY_NS,
            "key_utc": "1970-01-02T00:00:00Z",
            "field": "rate",
            "previous": "0.0001",
            "base": "0.0002",
        },
    )


def test_build_funding_rest_url_is_the_documented_endpoint() -> None:
    assert build_funding_rest_url(_SYMBOL, start_time_ms=1_000, end_time_ms=2_000) == (
        f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={_SYMBOL}"
        "&startTime=1000&endTime=2000&limit=1000"
    )


def _rest_payload(rows: list[dict[str, object]]) -> PanelPayload:
    return PanelPayload(
        url=build_funding_rest_url(_SYMBOL, start_time_ms=0, end_time_ms=1),
        raw_bytes=json.dumps(rows).encode("utf-8"),
        received_time_ns=1,
    )


def _settlement(time_ms: int, rate: str = "0.0001") -> dict[str, object]:
    return {
        "symbol": _SYMBOL,
        "fundingTime": time_ms,
        "fundingRate": rate,
        "markPrice": "100.0",
    }


def test_a_lone_settlement_takes_the_eight_hour_interval() -> None:
    start_ms = day_start_ms(FIRST_TAIL_DATE)
    payload = PanelPayload(
        url=build_funding_rest_url(
            _SYMBOL, start_time_ms=start_ms, end_time_ms=day_end_ms(FIRST_TAIL_SUNDAY)
        ),
        raw_bytes=funding_rest_json(_SYMBOL, start_ms, day_end_ms(FIRST_TAIL_SUNDAY)),
        received_time_ns=1,
    )

    rows = parse_funding_rest(
        payload,
        symbol=_SYMBOL,
        start_time_ms=start_ms,
        end_time_ms=day_end_ms(FIRST_TAIL_SUNDAY),
    )

    assert len(rows) == 1
    assert rows[0].calc_time_ns == start_ms * _NANOSECONDS_PER_MILLISECOND
    assert rows[0].funding_interval_hours == 8
    assert rows[0].rate == Decimal(funding_rate(_SYMBOL))
    assert rows[0].venue == "BINANCE_UM"


def test_the_funding_interval_comes_from_the_settlement_spacing() -> None:
    hour_ms = 3_600_000
    rows = parse_funding_rest(
        _rest_payload(
            [
                _settlement(0),
                _settlement(8 * hour_ms),
                _settlement(12 * hour_ms),
            ]
        ),
        symbol=_SYMBOL,
        start_time_ms=0,
        end_time_ms=24 * hour_ms,
    )

    # The last row has no successor, so it takes the delta before it.
    assert [row.funding_interval_hours for row in rows] == [8, 4, 4]


def test_settlements_outside_the_window_never_reach_a_row() -> None:
    hour_ms = 3_600_000
    rows = parse_funding_rest(
        _rest_payload(
            [_settlement(-1), _settlement(8 * hour_ms), _settlement(25 * hour_ms)]
        ),
        symbol=_SYMBOL,
        start_time_ms=0,
        end_time_ms=24 * hour_ms,
    )

    assert [row.calc_time_ns for row in rows] == [
        8 * hour_ms * _NANOSECONDS_PER_MILLISECOND
    ]
    assert rows[0].funding_interval_hours == 8


def test_a_repeated_funding_time_in_one_response_refuses() -> None:
    with pytest.raises(ShadowCaptureError) as error:
        parse_funding_rest(
            _rest_payload([_settlement(0), _settlement(0)]),
            symbol=_SYMBOL,
            start_time_ms=0,
            end_time_ms=1,
        )

    assert "fundingTime" in str(error.value)


def test_a_settlement_for_another_symbol_refuses() -> None:
    foreign = dict(_settlement(0))
    foreign["symbol"] = SYMBOLS[1]

    with pytest.raises(ShadowCaptureError) as error:
        parse_funding_rest(
            _rest_payload([foreign]), symbol=_SYMBOL, start_time_ms=0, end_time_ms=1
        )

    assert SYMBOLS[1] in str(error.value)


def test_a_response_that_is_not_an_array_refuses() -> None:
    payload = PanelPayload(
        url=build_funding_rest_url(_SYMBOL, start_time_ms=0, end_time_ms=1),
        raw_bytes=json.dumps({"code": -1121, "msg": "Invalid symbol."}).encode("utf-8"),
        received_time_ns=1,
    )

    with pytest.raises(ShadowCaptureError) as error:
        parse_funding_rest(payload, symbol=_SYMBOL, start_time_ms=0, end_time_ms=1)

    assert "array" in str(error.value)


# --- the base does not cover the same month for every symbol -----------------
#
# On the real bases most symbols reach the furthest month and the rest are
# behind by a month (publication lag) or by years (delistings, funding dumps
# the venue never wrote). A single cutoff over all of them would skip whole
# months for everything that is not at the front, so the tail is planned per
# symbol and per bucket.

_LAGGING_SYMBOL = SYMBOLS[3]
_STALE_SYMBOL = SYMBOLS[4]
_FUNDING_AHEAD_SYMBOL = SYMBOLS[1]


def test_a_symbol_one_month_behind_starts_its_tail_a_month_earlier(tmp_path: Path) -> None:
    base = _base_capture(
        tmp_path,
        fetch=ShadowFetch(absent_months=frozenset({(_LAGGING_SYMBOL, "2020-07", "klines")})),
    )

    artifact = _shadow(tmp_path, base, fetch=ShadowFetch())

    assert verify_panel_capture(artifact.capture_root) == (True, ())
    tail_rows = _rows(artifact.capture_root, DAILY_TAIL_KIND)
    lagging = [row for row in tail_rows if row["symbol"] == _LAGGING_SYMBOL]
    # All of July, which its own monthly dump never carried, plus the two
    # tail days every symbol gets.
    assert len(lagging) == 33
    assert min(str(row["month"]) for row in lagging) == "2020-07-01"
    # Nobody else moved, and a one-month lag is not staleness.
    assert len([row for row in tail_rows if row["symbol"] == _SYMBOL]) == 2
    assert _manifest(artifact.capture_root)["stale_symbols"] == {"klines": {}, "fundingRate": {}}
    # Its funding dump is not behind, so its REST window is everyone's.
    assert {
        str(row["month"])
        for row in _rows(artifact.capture_root, FUNDING_REST_KIND)
        if row["symbol"] == _LAGGING_SYMBOL
    } == {f"{FIRST_TAIL_DATE}_{FIRST_TAIL_SUNDAY}"}
    assert _quality(artifact.capture_root)[_LAGGING_SYMBOL]["missing_days"] == 0


def test_a_symbol_further_behind_is_declared_stale_and_starts_at_the_furthest_day(
    tmp_path: Path,
) -> None:
    base = _base_capture(
        tmp_path,
        fetch=ShadowFetch(
            absent_months=frozenset(
                {(_STALE_SYMBOL, "2020-06", "klines"), (_STALE_SYMBOL, "2020-07", "klines")}
            )
        ),
    )

    artifact = _shadow(tmp_path, base, fetch=ShadowFetch())

    assert verify_panel_capture(artifact.capture_root) == (True, ())
    assert _manifest(artifact.capture_root)["stale_symbols"] == {
        "klines": {_STALE_SYMBOL: "2020-05"},
        "fundingRate": {},
    }
    stale_rows = [
        row
        for row in _rows(artifact.capture_root, DAILY_TAIL_KIND)
        if row["symbol"] == _STALE_SYMBOL
    ]
    assert {str(row["month"]) for row in stale_rows} == {FIRST_TAIL_DATE, FIRST_TAIL_SUNDAY}


def test_a_funding_month_ahead_of_the_klines_months_does_not_move_the_klines_cutoff(
    tmp_path: Path,
) -> None:
    absent = {(symbol, "2020-08", "klines") for symbol in SYMBOLS} | {
        (symbol, "2020-08", "fundingRate")
        for symbol in SYMBOLS
        if symbol != _FUNDING_AHEAD_SYMBOL
    }
    base = _base_capture(
        tmp_path, months=(*MONTHS, "2020-08"), fetch=ShadowFetch(absent_months=frozenset(absent))
    )

    artifact = _shadow(tmp_path, base, fetch=ShadowFetch())

    assert verify_panel_capture(artifact.capture_root) == (True, ())
    # Every symbol's klines stop at 2020-07, so the daily tail starts on
    # 2020-08-01 for all of them -- the one 2020-08 funding dump did not drag
    # the klines cutoff forward past the tail.
    tail_rows = _rows(artifact.capture_root, DAILY_TAIL_KIND)
    assert len(tail_rows) == 2 * len(SYMBOLS)
    assert {str(row["month"]) for row in tail_rows} == {FIRST_TAIL_DATE, FIRST_TAIL_SUNDAY}
    # And in the other direction: the symbol whose funding already covers
    # 2020-08 needs no REST window at all, while the eleven a month behind
    # get theirs from the day after their own last funding month.
    rest_rows = _rows(artifact.capture_root, FUNDING_REST_KIND)
    assert {str(row["symbol"]) for row in rest_rows} == set(SYMBOLS) - {_FUNDING_AHEAD_SYMBOL}
    assert {str(row["month"]) for row in rest_rows} == {f"{FIRST_TAIL_DATE}_{FIRST_TAIL_SUNDAY}"}
    assert _manifest(artifact.capture_root)["stale_symbols"] == {"klines": {}, "fundingRate": {}}


def test_a_symbol_without_a_saturday_bar_never_waits_for_its_sunday(tmp_path: Path) -> None:
    base = _base_capture(tmp_path)
    fetch = ShadowFetch(
        absent_dates=frozenset({(_SYMBOL, FIRST_TAIL_DATE), (_SYMBOL, FIRST_TAIL_SUNDAY)})
    )
    clock = FakeClock()

    artifact = _shadow(tmp_path, base, fetch=fetch, clock=clock)

    # No Saturday bar means the symbol was not trading, so a missing Sunday
    # dump is an absence like any other and nothing waits for it.
    assert clock.slept == []
    assert verify_panel_capture(artifact.capture_root) == (True, ())
    assert {
        (str(row["month"]), str(row["status"]))
        for row in _rows(artifact.capture_root, DAILY_TAIL_KIND)
        if row["symbol"] == _SYMBOL
    } == {(FIRST_TAIL_DATE, "absent"), (FIRST_TAIL_SUNDAY, "absent")}


def test_a_funding_answer_at_the_row_limit_refuses(tmp_path: Path) -> None:
    base = _base_capture(tmp_path)

    with pytest.raises(ShadowCaptureError) as error:
        _shadow(tmp_path, base, fetch=ShadowFetch(pad_settlements_to=1000))

    assert "may be truncated" in str(error.value)


def test_an_already_written_raw_payload_path_refuses(tmp_path: Path) -> None:
    base = _base_capture(tmp_path)
    occupied = tmp_path / "shadow" / "raw" / _SYMBOL / "klines-2020-01.zip"
    occupied.parent.mkdir(parents=True)
    occupied.write_bytes(b"someone was here first")

    with pytest.raises(ShadowCaptureError) as error:
        _shadow(tmp_path, base, fetch=ShadowFetch())

    assert "already written" in str(error.value)
    # Ruling 21(c): the one case the unwinding cannot clear names its cure.
    assert "delete" in str(error.value)
    assert str(tmp_path / "shadow") in str(error.value)
    # A directory this run did not create is never removed, whatever is in it.
    assert occupied.read_bytes() == b"someone was here first"


def test_a_transport_failure_is_retryable_and_unwinds_the_half_written_capture(
    tmp_path: Path,
) -> None:
    """Ruling 21(a) and (b): a dropped transport is exit 1 and leaves nothing behind.

    A non-404 failure of a daily dump used to escape as a bare
    `PanelCaptureError` -- breaking this module's own contract and mapping to
    exit 1 by accident -- while the output directory kept whatever the run had
    copied into it, so every rerun hit the "already written" guard and the
    week could never be captured at all.
    """
    base = _base_capture(tmp_path)
    inner = ShadowFetch()
    daily_attempts: list[str] = []

    def drops_the_second_daily_dump(url: str) -> PanelPayload:
        if "/daily/klines/" in url:
            daily_attempts.append(url)
            if len(daily_attempts) == 2:
                raise PanelCaptureError("503: the dump host dropped the connection")
        return inner.fetch(url)

    with pytest.raises(ShadowCaptureTransportError) as error:
        build_shadow_capture(
            workspace_root=tmp_path,
            base_capture_root=base,
            output_directory=tmp_path / "shadow",
            reserve_bytes=0,
            tail_through=FIRST_TAIL_SUNDAY,
            market="um",
            fetch=drops_the_second_daily_dump,
            clock=FakeClock().time_ns,
        )

    assert "503" in str(error.value)
    assert isinstance(error.value.__cause__, PanelCaptureError)
    # This run created the directory, so this run takes it away again.
    assert not (tmp_path / "shadow").exists()

    artifact = _shadow(tmp_path, base, fetch=ShadowFetch())
    assert verify_panel_capture(artifact.capture_root) == (True, ())
    assert artifact.tail_through == FIRST_TAIL_SUNDAY


def test_a_funding_transport_failure_is_the_same_retryable_refusal(tmp_path: Path) -> None:
    """Both fetch paths wear one exception, so the exit code cannot differ by path."""
    base = _base_capture(tmp_path)
    inner = ShadowFetch()

    def drops_the_funding_window(url: str) -> PanelPayload:
        if "/fapi/v1/fundingRate" in url:
            raise PanelCaptureError("503: the REST host dropped the connection")
        return inner.fetch(url)

    with pytest.raises(ShadowCaptureTransportError) as error:
        build_shadow_capture(
            workspace_root=tmp_path,
            base_capture_root=base,
            output_directory=tmp_path / "shadow",
            reserve_bytes=0,
            tail_through=FIRST_TAIL_SUNDAY,
            market="um",
            fetch=drops_the_funding_window,
            clock=FakeClock().time_ns,
        )

    assert "could not be fetched" in str(error.value)
    assert not (tmp_path / "shadow").exists()


# `raw_sha256` is not in this list: dropping it makes the base itself fail
# verification, which is a different refusal and already covered.
@pytest.mark.parametrize("field", ["received_time_ns", "symbol", "url", "kind"])
def test_a_present_row_missing_a_required_field_refuses(tmp_path: Path, field: str) -> None:
    base = _base_capture(tmp_path)

    def drop_the_field(material: dict[str, object]) -> None:
        sources = material["sources"]
        assert isinstance(sources, list)
        row = next(entry for entry in sources if entry.get("status") == "present")
        del row[field]

    _reseal(base, drop_the_field)
    # The base still verifies -- none of these fields is part of what
    # `verify_panel_capture` checks -- so the refusal has to come from here.
    assert verify_panel_capture(base) == (True, ())

    with pytest.raises(ShadowCaptureError) as error:
        _shadow(tmp_path, base, fetch=ShadowFetch())

    assert f"has no {field}" in str(error.value)


def test_a_funding_request_that_fails_refuses_as_a_shadow_error(tmp_path: Path) -> None:
    base = _base_capture(tmp_path)
    inner = ShadowFetch()

    def fetch(url: str) -> PanelPayload:
        if "/fapi/v1/fundingRate" in url:
            raise PanelSourceAbsent("404: no funding history")
        return inner.fetch(url)

    with pytest.raises(ShadowCaptureError) as error:
        build_shadow_capture(
            workspace_root=tmp_path,
            base_capture_root=base,
            output_directory=tmp_path / "shadow",
            reserve_bytes=0,
            tail_through=FIRST_TAIL_SUNDAY,
            market="um",
            fetch=fetch,
            clock=FakeClock().time_ns,
        )

    assert "could not be fetched" in str(error.value)


def test_a_funding_url_for_an_impossible_symbol_refuses_as_a_shadow_error() -> None:
    with pytest.raises(ShadowCaptureError) as error:
        build_funding_rest_url("btcusdt", start_time_ms=0, end_time_ms=1)

    assert "btcusdt" in str(error.value)


def test_a_settlement_spacing_below_an_hour_refuses() -> None:
    with pytest.raises(ShadowCaptureError) as error:
        parse_funding_rest(
            _rest_payload([_settlement(0), _settlement(60_000)]),
            symbol=_SYMBOL,
            start_time_ms=0,
            end_time_ms=3_600_000,
        )

    assert "less than an hour" in str(error.value)


def test_a_settlement_spacing_off_a_whole_hour_refuses() -> None:
    hour_ms = 3_600_000
    with pytest.raises(ShadowCaptureError) as error:
        parse_funding_rest(
            _rest_payload([_settlement(0), _settlement(8 * hour_ms + 120_000)]),
            symbol=_SYMBOL,
            start_time_ms=0,
            end_time_ms=24 * hour_ms,
        )

    assert "not a whole number of hours" in str(error.value)


def test_a_settlement_a_few_seconds_early_still_measures_eight_hours() -> None:
    hour_ms = 3_600_000
    rows = parse_funding_rest(
        # Thirty seconds short of eight hours -- floor division would have
        # called this a seven-hour funding interval.
        _rest_payload([_settlement(0), _settlement(8 * hour_ms - 30_000)]),
        symbol=_SYMBOL,
        start_time_ms=0,
        end_time_ms=24 * hour_ms,
    )

    assert [row.funding_interval_hours for row in rows] == [8, 8]


# A week plans its own span from its own base and subtracts only what the
# previous week actually carried. The case that matters is a symbol whose base
# coverage catches up between two weeks: week 2's correct start is *earlier*
# than where week 1 fetched from, so a forward-only cursor would skip the days
# in between for good while `stale_symbols` stopped mentioning the symbol.

_CHAIN_SYMBOL = SYMBOLS[5]
_JULY_DATES = frozenset(f"2020-07-{day:02d}" for day in range(1, 32))


def _requested_dates(fetch: ShadowFetch, symbol: str) -> set[str]:
    """Every date whose daily dump was asked for, for one symbol."""
    return {
        url.rsplit("-1d-", 1)[-1].removesuffix(".zip")
        for url in _daily_urls(fetch)
        if f"/{symbol}/" in url
    }


def test_a_symbol_whose_base_catches_up_gets_the_months_the_stale_week_skipped(
    tmp_path: Path,
) -> None:
    # Week 1: the symbol is two months behind in both buckets, so it is stale
    # and its tail starts at the furthest month's day like everyone else's.
    behind_two_months = frozenset(
        {(_CHAIN_SYMBOL, month, kind) for month in ("2020-06", "2020-07") for kind in _BUCKETS}
    )
    first_base = _base_capture(
        tmp_path, fetch=ShadowFetch(absent_months=behind_two_months), name="base-week-1"
    )
    first = _shadow(tmp_path, first_base, fetch=ShadowFetch(), name="week-1")
    assert _manifest(first.capture_root)["stale_symbols"] == {
        "klines": {_CHAIN_SYMBOL: "2020-05"},
        "fundingRate": {_CHAIN_SYMBOL: "2020-05"},
    }

    # Week 2: 2020-06 arrived, so the symbol is only one month behind and its
    # correct start is 2020-07-01 -- a month before week 1 ever fetched.
    behind_one_month = frozenset({(_CHAIN_SYMBOL, "2020-07", kind) for kind in _BUCKETS})
    second_base = _base_capture(
        tmp_path, fetch=ShadowFetch(absent_months=behind_one_month), name="base-week-2"
    )
    fetch = ShadowFetch()
    second = _shadow(
        tmp_path,
        second_base,
        fetch=fetch,
        tail_through=SECOND_TAIL_SUNDAY,
        previous=first.capture_root,
        name="week-2",
    )

    assert verify_panel_capture(second.capture_root) == (True, ())
    assert verify_capture_superset(second_base, second.capture_root) == (True, ())
    # The whole of July -- which no week has ever fetched -- plus this week's
    # own new days, and nothing week 1 already carried.
    new_days = {"2020-08-03", "2020-08-04", "2020-08-05", "2020-08-06"}
    new_days |= {"2020-08-07", "2020-08-08", SECOND_TAIL_SUNDAY}
    assert _requested_dates(fetch, _CHAIN_SYMBOL) == set(_JULY_DATES) | new_days
    assert {FIRST_TAIL_DATE, FIRST_TAIL_SUNDAY}.isdisjoint(
        _requested_dates(fetch, _CHAIN_SYMBOL)
    )
    # Which is what the manifest now claims: the symbol is no longer stale,
    # so there had better be no gap behind that claim.
    stale = _manifest(second.capture_root)["stale_symbols"]
    assert stale == {"klines": {}, "fundingRate": {}}
    assert _quality(second.capture_root)[_CHAIN_SYMBOL]["missing_days"] == 0
    # 31 July days + 7 new ones fetched, and the 2 days week 1 carried.
    chain_rows = [
        row
        for row in _rows(second.capture_root, DAILY_TAIL_KIND)
        if row["symbol"] == _CHAIN_SYMBOL
    ]
    assert len(chain_rows) == 40

    # The funding analogue: one REST window before the carried one and one
    # after it, because the carried window sits inside this week's span.
    assert {
        str(row["month"])
        for row in _rows(second.capture_root, FUNDING_REST_KIND)
        if row["symbol"] == _CHAIN_SYMBOL
    } == {
        "2020-07-01_2020-07-31",
        f"{FIRST_TAIL_DATE}_{FIRST_TAIL_SUNDAY}",
        f"2020-08-03_{SECOND_TAIL_SUNDAY}",
    }
    # Every other symbol is unaffected: two carried days and seven new ones.
    untouched = [
        row for row in _rows(second.capture_root, DAILY_TAIL_KIND) if row["symbol"] == _SYMBOL
    ]
    assert len(untouched) == 9
