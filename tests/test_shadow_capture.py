"""Tests for the weekly shadow capture (spec sections 3.1 and 3.3).

Every capture here is built from the fixture fetches -- no network. The base
captures are the carry fixtures' own captures over `MONTHS`; the tail is the
two months registered additively in `tests.shadow_fixtures`.
"""

import json
from dataclasses import dataclass, field, replace
from decimal import Decimal
from pathlib import Path

import pytest

from tests.carry_fixtures import build_captures, funding_rate
from tests.shadow_fixtures import (
    FIRST_TAIL_DATE,
    FIRST_TAIL_SUNDAY,
    SECOND_TAIL_SUNDAY,
    THIRD_TAIL_SUNDAY,
    ShadowFetch,
    daily_kline_csv,
    day_end_ms,
    day_start_ms,
    funding_rest_json,
)
from tests.test_panel_fold_run import MONTHS, SYMBOLS, kline_csv, zip_bytes
from trading_bot.capture_lineage import verify_capture_superset
from trading_bot.panel_capture import (
    PanelPayload,
    build_kline_zip_url,
    capture_panel,
    parse_kline_zip,
    verify_panel_capture,
)
from trading_bot.panel_reader import load_funding_events, load_panel_bars
from trading_bot.shadow_capture import (
    DAILY_TAIL_KIND,
    FUNDING_REST_KIND,
    ShadowCaptureArtifact,
    ShadowCaptureError,
    build_funding_rest_url,
    build_shadow_capture,
    parse_funding_rest,
)

_SYMBOL = SYMBOLS[0]
_NANOSECONDS_PER_MILLISECOND = 1_000_000


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
    root: Path, *, months: tuple[str, ...] = MONTHS, fetch: ShadowFetch | None = None
) -> Path:
    """A perpetual base capture over `months`, built by the shadow fetch.

    Byte for byte the capture `tests.carry_fixtures.build_captures` builds for
    the same months: both serve the monthly URLs from the same generators.
    """
    download = fetch if fetch is not None else ShadowFetch()
    return capture_panel(
        workspace_root=root,
        output_directory=root / f"base-{months[-1]}",
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
