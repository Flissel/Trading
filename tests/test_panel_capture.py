import http.client
import io
import json
import re
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from email.message import Message
from pathlib import Path

import pytest

from trading_bot.panel_capture import (
    PanelCaptureError,
    PanelPayload,
    PanelSourceAbsent,
    PanelZipClient,
    build_daily_kline_zip_url,
    build_funding_zip_url,
    build_kline_zip_url,
    capture_panel,
    discover_panel_months,
    parse_funding_zip,
    parse_kline_zip,
    repair_panel_capture,
    verify_panel_capture,
)

KLINE_HEADER = (
    "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
    "taker_buy_volume,taker_buy_quote_volume,ignore"
)
DAY_MS = 86_400_000
# Deliberately contiguous (3 rows exactly, back to back) rather than aligned
# to the real length of January: these fixtures only ever cover 3 days per
# month, and a real gap between the two chunks (e.g. Feb starting at day 31)
# would now be an interior hole `capture_panel` tries to daily-fill, which
# these tests are not about and do not stub a daily-dump fetch for.
MONTH_START_DAY = {"2024-01": 0, "2024-02": 3}


class _FakeUrlopenResponse:
    """Minimal stand-in for the object urllib.request.urlopen returns."""

    def __init__(self, body: bytes, url: str) -> None:
        self._body = body
        self._url = url

    def __enter__(self) -> "_FakeUrlopenResponse":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def read(self, limit: int) -> bytes:
        return self._body

    def geturl(self) -> str:
        return self._url


def zip_bytes(name: str, text: str) -> bytes:
    buffer = io.BytesIO()
    # A bare filename makes ZipFile.writestr stamp the entry with
    # time.localtime() at two-second resolution, which makes the resulting
    # bytes -- and therefore raw_sha256 -- nondeterministic across two calls
    # that straddle a boundary. Pin a fixed date_time so identical (name,
    # text) always produces byte-identical zips.
    info = zipfile.ZipInfo(filename=name, date_time=(2024, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(info, text)
    return buffer.getvalue()


def kline_csv(days: int, *, with_header: bool = True, start_day: int = 0) -> str:
    lines = [KLINE_HEADER] if with_header else []
    for index in range(days):
        day = start_day + index
        open_ms = day * DAY_MS
        lines.append(
            f"{open_ms},100,101,99,{100 + index},10,{open_ms + DAY_MS - 1},1000,5,6,600,0"
        )
    return "\n".join(lines) + "\n"


def funding_csv() -> str:
    return "calc_time,funding_interval_hours,last_funding_rate\n0,8,0.0001\n"


def test_urls_are_pinned_to_the_public_bucket() -> None:
    assert build_kline_zip_url("BTCUSDT", "2024-01") == (
        "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1d/"
        "BTCUSDT-1d-2024-01.zip"
    )
    assert build_funding_zip_url("BTCUSDT", "2024-01") == (
        "https://data.binance.vision/data/futures/um/monthly/fundingRate/BTCUSDT/"
        "BTCUSDT-fundingRate-2024-01.zip"
    )


def test_daily_dump_url_is_pinned_to_the_public_bucket() -> None:
    assert build_daily_kline_zip_url("BTCUSDT", "2022-02-26") == (
        "https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/1d/"
        "BTCUSDT-1d-2022-02-26.zip"
    )


def test_parse_kline_zip_reads_rows_with_and_without_header() -> None:
    payload = PanelPayload(url="u", raw_bytes=zip_bytes("a.csv", kline_csv(2)), received_time_ns=1)
    rows = parse_kline_zip(payload, symbol="BTCUSDT")
    assert len(rows) == 2
    assert rows[0].instrument_id == "BTCUSDT"
    assert str(rows[1].close) == "101"
    assert rows[0].available_time_ns == rows[0].close_time_ns + 1
    bare = PanelPayload(
        url="u", raw_bytes=zip_bytes("a.csv", kline_csv(1, with_header=False)), received_time_ns=1
    )
    assert len(parse_kline_zip(bare, symbol="BTCUSDT")) == 1


def test_parse_funding_zip() -> None:
    text = "calc_time,funding_interval_hours,last_funding_rate\n0,8,-0.00006120\n"
    payload = PanelPayload(url="u", raw_bytes=zip_bytes("f.csv", text), received_time_ns=1)
    rows = parse_funding_zip(payload, symbol="BTCUSDT")
    assert len(rows) == 1
    assert str(rows[0].rate) == "-0.00006120"
    assert rows[0].funding_interval_hours == 8


def test_capture_publishes_and_verifies(tmp_path: Path) -> None:
    def fetch(url: str) -> PanelPayload:
        if "fundingRate" in url:
            text = "calc_time,funding_interval_hours,last_funding_rate\n0,8,0.0001\n"
            return PanelPayload(url=url, raw_bytes=zip_bytes("f.csv", text), received_time_ns=1)
        return PanelPayload(url=url, raw_bytes=zip_bytes("k.csv", kline_csv(3)), received_time_ns=1)

    artifact = capture_panel(
        workspace_root=tmp_path,
        output_directory=tmp_path / "capture",
        reserve_bytes=0,
        symbols=("BTCUSDT", "ETHUSDT"),
        months=("2024-01",),
        fetch=fetch,
    )
    assert artifact.capture_root_hash
    assert (artifact.capture_root / "raw" / "BTCUSDT").is_dir()
    assert verify_panel_capture(artifact.capture_root) == (True, ())


def test_capture_rejects_a_foreign_host(tmp_path: Path) -> None:
    def fetch(url: str) -> PanelPayload:
        return PanelPayload(url="https://evil.example/x.zip", raw_bytes=b"", received_time_ns=1)

    with pytest.raises(PanelCaptureError):
        capture_panel(
            workspace_root=tmp_path,
            output_directory=tmp_path / "capture",
            reserve_bytes=0,
            symbols=("BTCUSDT",),
            months=("2024-01",),
            fetch=fetch,
        )


def test_panel_zip_client_raises_source_absent_only_for_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real live-bucket bug this guards against: LUNAUSDT's 2022-05 dump fetches
    but its 2023-01 dump 404s, because the perpetual was delisted in between. That
    404 must be distinguishable from a genuine outage (e.g. a 500) so the caller can
    treat "never listed / already delisted" as a recorded fact rather than an abort."""

    def raise_404(request: urllib.request.Request, timeout: float) -> None:
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", Message(), None)

    monkeypatch.setattr(urllib.request, "urlopen", raise_404)
    sleeps: list[float] = []
    client = PanelZipClient(sleep=sleeps.append)
    with pytest.raises(PanelSourceAbsent):
        client.fetch(build_kline_zip_url("LUNAUSDT", "2023-01"))
    assert sleeps == []

    def raise_500(request: urllib.request.Request, timeout: float) -> None:
        raise urllib.error.HTTPError(request.full_url, 500, "Server Error", Message(), None)

    monkeypatch.setattr(urllib.request, "urlopen", raise_500)
    with pytest.raises(PanelCaptureError) as excinfo:
        client.fetch(build_kline_zip_url("LUNAUSDT", "2023-01"))
    assert not isinstance(excinfo.value, PanelSourceAbsent)


def test_panel_zip_client_retries_a_transient_failure_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0}

    def flaky_urlopen(request: urllib.request.Request, timeout: float) -> _FakeUrlopenResponse:
        calls["count"] += 1
        if calls["count"] == 1:
            raise urllib.error.URLError("connection reset")
        return _FakeUrlopenResponse(b"payload-bytes", request.full_url)

    monkeypatch.setattr(urllib.request, "urlopen", flaky_urlopen)
    sleeps: list[float] = []
    client = PanelZipClient(sleep=sleeps.append)
    payload = client.fetch(build_kline_zip_url("BTCUSDT", "2024-01"))
    assert payload.raw_bytes == b"payload-bytes"
    assert sleeps == [1]


def test_panel_zip_client_gives_up_after_four_transient_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def always_times_out(request: urllib.request.Request, timeout: float) -> None:
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", always_times_out)
    sleeps: list[float] = []
    client = PanelZipClient(sleep=sleeps.append, max_attempts=4)
    with pytest.raises(PanelCaptureError, match="4 attempts") as excinfo:
        client.fetch(build_kline_zip_url("BTCUSDT", "2024-01"))
    assert not isinstance(excinfo.value, PanelSourceAbsent)
    assert sleeps == [1, 2, 4]


def test_panel_zip_client_does_not_retry_a_403(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_403(request: urllib.request.Request, timeout: float) -> None:
        raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", Message(), None)

    monkeypatch.setattr(urllib.request, "urlopen", raise_403)
    sleeps: list[float] = []
    client = PanelZipClient(sleep=sleeps.append)
    with pytest.raises(PanelCaptureError) as excinfo:
        client.fetch(build_kline_zip_url("BTCUSDT", "2024-01"))
    assert not isinstance(excinfo.value, PanelSourceAbsent)
    assert sleeps == []


def test_panel_zip_client_retries_a_connection_reset_during_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0}

    class _FlakyReadResponse(_FakeUrlopenResponse):
        def read(self, limit: int) -> bytes:
            calls["count"] += 1
            if calls["count"] == 1:
                raise ConnectionResetError("reset by peer")
            return super().read(limit)

    def urlopen_stub(request: urllib.request.Request, timeout: float) -> _FlakyReadResponse:
        return _FlakyReadResponse(b"payload-bytes", request.full_url)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen_stub)
    sleeps: list[float] = []
    client = PanelZipClient(sleep=sleeps.append)
    payload = client.fetch(build_kline_zip_url("BTCUSDT", "2024-01"))
    assert payload.raw_bytes == b"payload-bytes"
    assert sleeps == [1]


def test_panel_zip_client_retries_an_incomplete_read(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    class _FlakyReadResponse(_FakeUrlopenResponse):
        def read(self, limit: int) -> bytes:
            calls["count"] += 1
            if calls["count"] == 1:
                raise http.client.IncompleteRead(b"partial")
            return super().read(limit)

    def urlopen_stub(request: urllib.request.Request, timeout: float) -> _FlakyReadResponse:
        return _FlakyReadResponse(b"payload-bytes", request.full_url)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen_stub)
    sleeps: list[float] = []
    client = PanelZipClient(sleep=sleeps.append)
    payload = client.fetch(build_kline_zip_url("BTCUSDT", "2024-01"))
    assert payload.raw_bytes == b"payload-bytes"
    assert sleeps == [1]


@pytest.mark.parametrize("error_type", [ConnectionAbortedError, BrokenPipeError])
def test_panel_zip_client_retries_the_rest_of_the_oserror_family(
    monkeypatch: pytest.MonkeyPatch, error_type: type[OSError]
) -> None:
    # ConnectionResetError and IncompleteRead were named explicitly, but the
    # retry ladder must catch the whole OSError family -- ConnectionAbortedError
    # (WinError 10053, the common Windows mid-transfer abort) and
    # BrokenPipeError are exactly as likely over ~46,000 requests.
    calls = {"count": 0}

    class _FlakyReadResponse(_FakeUrlopenResponse):
        def read(self, limit: int) -> bytes:
            calls["count"] += 1
            if calls["count"] == 1:
                raise error_type("connection dropped")
            return super().read(limit)

    def urlopen_stub(request: urllib.request.Request, timeout: float) -> _FlakyReadResponse:
        return _FlakyReadResponse(b"payload-bytes", request.full_url)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen_stub)
    sleeps: list[float] = []
    client = PanelZipClient(sleep=sleeps.append)
    payload = client.fetch(build_kline_zip_url("BTCUSDT", "2024-01"))
    assert payload.raw_bytes == b"payload-bytes"
    assert sleeps == [1]


def test_panel_zip_client_wraps_an_unexpected_error_without_retrying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def urlopen_stub(request: urllib.request.Request, timeout: float) -> None:
        raise ValueError("totally unexpected")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen_stub)
    sleeps: list[float] = []
    client = PanelZipClient(sleep=sleeps.append)
    with pytest.raises(PanelCaptureError) as excinfo:
        client.fetch(build_kline_zip_url("BTCUSDT", "2024-01"))
    assert not isinstance(excinfo.value, PanelSourceAbsent)
    assert sleeps == []


def test_panel_zip_client_retries_a_429(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def flaky_urlopen(request: urllib.request.Request, timeout: float) -> _FakeUrlopenResponse:
        calls["count"] += 1
        if calls["count"] == 1:
            raise urllib.error.HTTPError(
                request.full_url, 429, "Too Many Requests", Message(), None
            )
        return _FakeUrlopenResponse(b"payload-bytes", request.full_url)

    monkeypatch.setattr(urllib.request, "urlopen", flaky_urlopen)
    sleeps: list[float] = []
    client = PanelZipClient(sleep=sleeps.append)
    payload = client.fetch(build_kline_zip_url("BTCUSDT", "2024-01"))
    assert payload.raw_bytes == b"payload-bytes"
    assert sleeps == [1]


def test_capture_survives_and_records_a_delisted_symbols_absent_month(
    tmp_path: Path,
) -> None:
    # OLDUSDT's 2024-02 dumps (both klines and fundingRate) are 404 -- as they would
    # be for a real perpetual delisted before that month -- while 2024-01 fetches
    # normally.
    def fetch(url: str) -> PanelPayload:
        if "OLDUSDT" in url and "2024-02" in url:
            raise PanelSourceAbsent("404: contract delisted before this month")
        if "fundingRate" in url:
            return PanelPayload(
                url=url, raw_bytes=zip_bytes("f.csv", funding_csv()), received_time_ns=1
            )
        return PanelPayload(
            url=url,
            raw_bytes=zip_bytes("k.csv", kline_csv(3, start_day=MONTH_START_DAY["2024-01"])),
            received_time_ns=1,
        )

    artifact = capture_panel(
        workspace_root=tmp_path,
        output_directory=tmp_path / "capture",
        reserve_bytes=0,
        symbols=("OLDUSDT",),
        months=("2024-01", "2024-02"),
        fetch=fetch,
    )
    manifest = json.loads(artifact.capture_manifest_path.read_text(encoding="utf-8"))
    absent = [entry for entry in manifest["sources"] if entry["status"] == "absent"]
    assert {(entry["month"], entry["kind"]) for entry in absent} == {
        ("2024-02", "klines"),
        ("2024-02", "fundingRate"),
    }
    for entry in absent:
        assert entry["symbol"] == "OLDUSDT"
        assert "raw_relative_path" not in entry
        assert "raw_sha256" not in entry
    present = [entry for entry in manifest["sources"] if entry["status"] == "present"]
    assert {(entry["month"], entry["kind"]) for entry in present} == {
        ("2024-01", "klines"),
        ("2024-01", "fundingRate"),
    }
    assert all("raw_relative_path" in entry and "raw_sha256" in entry for entry in present)

    # The dataset holds only the month that actually existed.
    quality = json.loads(
        (artifact.dataset_root / "quality-report.json").read_text(encoding="utf-8")
    )
    assert quality["candle_row_count"] == 3

    assert verify_panel_capture(artifact.capture_root) == (True, ())


def test_capture_fails_closed_when_a_symbol_has_no_data_in_any_month(
    tmp_path: Path,
) -> None:
    def fetch(url: str) -> PanelPayload:
        if "GHOSTUSDT" in url:
            raise PanelSourceAbsent("404: never listed")
        if "fundingRate" in url:
            return PanelPayload(
                url=url, raw_bytes=zip_bytes("f.csv", funding_csv()), received_time_ns=1
            )
        month = "2024-01" if "2024-01" in url else "2024-02"
        return PanelPayload(
            url=url,
            raw_bytes=zip_bytes("k.csv", kline_csv(3, start_day=MONTH_START_DAY[month])),
            received_time_ns=1,
        )

    with pytest.raises(PanelCaptureError, match="GHOSTUSDT"):
        capture_panel(
            workspace_root=tmp_path,
            output_directory=tmp_path / "capture",
            reserve_bytes=0,
            symbols=("BTCUSDT", "GHOSTUSDT"),
            months=("2024-01", "2024-02"),
            fetch=fetch,
        )


def test_capture_aborts_on_a_non_absent_failure(tmp_path: Path) -> None:
    def fetch(url: str) -> PanelPayload:
        if "2024-02" in url:
            raise PanelCaptureError("panel dump request failed: simulated 500")
        if "fundingRate" in url:
            return PanelPayload(
                url=url, raw_bytes=zip_bytes("f.csv", funding_csv()), received_time_ns=1
            )
        return PanelPayload(url=url, raw_bytes=zip_bytes("k.csv", kline_csv(3)), received_time_ns=1)

    with pytest.raises(PanelCaptureError, match="simulated 500"):
        capture_panel(
            workspace_root=tmp_path,
            output_directory=tmp_path / "capture",
            reserve_bytes=0,
            symbols=("BTCUSDT",),
            months=("2024-01", "2024-02"),
            fetch=fetch,
        )


def _listing_xml(*, truncated: bool = False, keys: tuple[str, ...] = ()) -> bytes:
    contents = "".join(f"<Contents><Key>{key}</Key></Contents>" for key in keys)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        f"<IsTruncated>{'true' if truncated else 'false'}</IsTruncated>"
        f"{contents}"
        "</ListBucketResult>"
    ).encode()


def test_discover_panel_months_ignores_a_different_symbol_with_shared_prefix() -> None:
    keys = (
        "data/futures/um/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2024-01.zip",
        "data/futures/um/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2024-01.zip.CHECKSUM",
        "data/futures/um/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2024-02.zip",
        "data/futures/um/monthly/klines/BTCUSDTX/1d/BTCUSDTX-1d-2024-03.zip",
    )

    def fetch(url: str) -> PanelPayload:
        return PanelPayload(url=url, raw_bytes=_listing_xml(keys=keys), received_time_ns=1)

    months = discover_panel_months(fetch, symbol="BTCUSDT", kind="klines")
    assert months == ("2024-01", "2024-02")


def test_discover_panel_months_raises_when_the_listing_is_truncated() -> None:
    keys = ("data/futures/um/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2024-01.zip",)

    def fetch(url: str) -> PanelPayload:
        return PanelPayload(
            url=url, raw_bytes=_listing_xml(truncated=True, keys=keys), received_time_ns=1
        )

    with pytest.raises(PanelCaptureError):
        discover_panel_months(fetch, symbol="BTCUSDT", kind="klines")


def test_discover_panel_months_raises_when_is_truncated_is_missing() -> None:
    xml = (
        b'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        b"<Contents><Key>data/futures/um/monthly/klines/BTCUSDT/1d/"
        b"BTCUSDT-1d-2024-01.zip</Key></Contents>"
        b"</ListBucketResult>"
    )

    def fetch(url: str) -> PanelPayload:
        return PanelPayload(url=url, raw_bytes=xml, received_time_ns=1)

    with pytest.raises(PanelCaptureError):
        discover_panel_months(fetch, symbol="BTCUSDT", kind="klines")


def test_discover_panel_months_raises_on_a_continuation_token() -> None:
    xml = (
        b'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        b"<IsTruncated>false</IsTruncated>"
        b"<NextContinuationToken>abc</NextContinuationToken>"
        b"<Contents><Key>data/futures/um/monthly/klines/BTCUSDT/1d/"
        b"BTCUSDT-1d-2024-01.zip</Key></Contents>"
        b"</ListBucketResult>"
    )

    def fetch(url: str) -> PanelPayload:
        return PanelPayload(url=url, raw_bytes=xml, received_time_ns=1)

    with pytest.raises(PanelCaptureError):
        discover_panel_months(fetch, symbol="BTCUSDT", kind="klines")


def test_discover_panel_months_raises_on_a_non_listing_root() -> None:
    # An S3 <Error> body can be served with HTTP 200; it must never be read
    # as a complete, empty listing.
    xml = b"<Error><Code>AccessDenied</Code><Message>Access Denied</Message></Error>"

    def fetch(url: str) -> PanelPayload:
        return PanelPayload(url=url, raw_bytes=xml, received_time_ns=1)

    with pytest.raises(PanelCaptureError):
        discover_panel_months(fetch, symbol="BTCUSDT", kind="klines")


def test_discover_panel_months_raises_when_the_key_count_hits_the_page_limit() -> None:
    keys = tuple(
        "data/futures/um/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2024-01.zip" for _ in range(1000)
    )

    def fetch(url: str) -> PanelPayload:
        return PanelPayload(url=url, raw_bytes=_listing_xml(keys=keys), received_time_ns=1)

    with pytest.raises(PanelCaptureError):
        discover_panel_months(fetch, symbol="BTCUSDT", kind="klines")


def _discovery_and_zip_fetch(
    *, kline_months: tuple[str, ...], funding_months: tuple[str, ...]
) -> Callable[[str], PanelPayload]:
    def fetch(url: str) -> PanelPayload:
        if "list-type=2" in url:
            if "fundingRate" in url:
                keys = tuple(
                    f"data/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-{month}.zip"
                    for month in funding_months
                )
            else:
                keys = tuple(
                    f"data/futures/um/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-{month}.zip"
                    for month in kline_months
                )
            return PanelPayload(url=url, raw_bytes=_listing_xml(keys=keys), received_time_ns=1)
        if "fundingRate" in url:
            return PanelPayload(
                url=url, raw_bytes=zip_bytes("f.csv", funding_csv()), received_time_ns=1
            )
        return PanelPayload(url=url, raw_bytes=zip_bytes("k.csv", kline_csv(3)), received_time_ns=1)

    return fetch


def test_capture_with_months_omitted_discovers_and_fetches_only_existing_months(
    tmp_path: Path,
) -> None:
    fetch = _discovery_and_zip_fetch(kline_months=("2024-01",), funding_months=("2024-01",))

    artifact = capture_panel(
        workspace_root=tmp_path,
        output_directory=tmp_path / "capture",
        reserve_bytes=0,
        symbols=("BTCUSDT",),
        fetch=fetch,
    )
    manifest = json.loads(artifact.capture_manifest_path.read_text(encoding="utf-8"))
    assert all(entry["status"] == "present" for entry in manifest["sources"])
    assert {(entry["month"], entry["kind"]) for entry in manifest["sources"]} == {
        ("2024-01", "klines"),
        ("2024-01", "fundingRate"),
    }
    assert verify_panel_capture(artifact.capture_root) == (True, ())


def test_capture_bounds_filter_the_discovered_months(tmp_path: Path) -> None:
    fetch = _discovery_and_zip_fetch(
        kline_months=("2024-01", "2024-02", "2024-03"),
        funding_months=("2024-01", "2024-02", "2024-03"),
    )

    artifact = capture_panel(
        workspace_root=tmp_path,
        output_directory=tmp_path / "capture",
        reserve_bytes=0,
        symbols=("BTCUSDT",),
        month_from="2024-02",
        month_to="2024-02",
        fetch=fetch,
    )
    manifest = json.loads(artifact.capture_manifest_path.read_text(encoding="utf-8"))
    assert {(entry["month"], entry["kind"]) for entry in manifest["sources"]} == {
        ("2024-02", "klines"),
        ("2024-02", "fundingRate"),
    }


def test_capture_records_discovered_months_including_a_symbol_with_zero_funding_months(
    tmp_path: Path,
) -> None:
    fetch = _discovery_and_zip_fetch(kline_months=("2024-01", "2024-02"), funding_months=())

    artifact = capture_panel(
        workspace_root=tmp_path,
        output_directory=tmp_path / "capture",
        reserve_bytes=0,
        symbols=("BTCUSDT",),
        fetch=fetch,
    )
    manifest = json.loads(artifact.capture_manifest_path.read_text(encoding="utf-8"))
    assert manifest["discovered_months"]["BTCUSDT"]["klines"] == ["2024-01", "2024-02"]
    # A symbol that discovered zero funding months is now visible as such,
    # distinct from a symbol that was never asked about funding at all.
    assert manifest["discovered_months"]["BTCUSDT"]["fundingRate"] == []
    assert verify_panel_capture(artifact.capture_root) == (True, ())


def test_capture_records_a_discovery_then_404_distinctly(tmp_path: Path) -> None:
    def fetch(url: str) -> PanelPayload:
        if "list-type=2" in url:
            if "fundingRate" in url:
                keys = (
                    "data/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2024-01.zip",
                )
            else:
                keys = ("data/futures/um/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2024-01.zip",)
            return PanelPayload(url=url, raw_bytes=_listing_xml(keys=keys), received_time_ns=1)
        if "fundingRate" in url:
            # The bucket contradicts its own listing: this discovered month
            # 404s when actually fetched.
            raise PanelSourceAbsent("404: contradicts discovery")
        return PanelPayload(url=url, raw_bytes=zip_bytes("k.csv", kline_csv(3)), received_time_ns=1)

    artifact = capture_panel(
        workspace_root=tmp_path,
        output_directory=tmp_path / "capture",
        reserve_bytes=0,
        symbols=("BTCUSDT",),
        fetch=fetch,
    )
    manifest = json.loads(artifact.capture_manifest_path.read_text(encoding="utf-8"))
    funding_entry = next(e for e in manifest["sources"] if e["kind"] == "fundingRate")
    assert funding_entry["status"] == "absent_after_discovery"
    assert verify_panel_capture(artifact.capture_root) == (True, ())


def _progress_header(
    *,
    symbols: tuple[str, ...],
    months: tuple[str, ...] | None,
    month_from: str | None = None,
    month_to: str | None = None,
    market: str = "um",
) -> str:
    return json.dumps(
        {
            "capture_parameters": {
                "capture_version": "1.0.0",
                "market": market,
                "symbols": list(symbols),
                "months": list(months) if months is not None else None,
                "month_from": month_from,
                "month_to": month_to,
            }
        }
    )


def _deterministic_fetch() -> Callable[[str], PanelPayload]:
    def fetch(url: str) -> PanelPayload:
        if "fundingRate" in url:
            return PanelPayload(
                url=url, raw_bytes=zip_bytes("f.csv", funding_csv()), received_time_ns=1
            )
        month = "2024-01" if "2024-01" in url else "2024-02"
        return PanelPayload(
            url=url,
            raw_bytes=zip_bytes("k.csv", kline_csv(3, start_day=MONTH_START_DAY[month])),
            received_time_ns=1,
        )

    return fetch


def test_capture_resumes_after_an_interruption_and_matches_an_uninterrupted_run(
    tmp_path: Path,
) -> None:
    control = capture_panel(
        workspace_root=tmp_path,
        output_directory=tmp_path / "control",
        reserve_bytes=0,
        symbols=("BTCUSDT", "ETHUSDT"),
        months=("2024-01", "2024-02"),
        fetch=_deterministic_fetch(),
    )

    target = tmp_path / "resumable"
    calls = {"count": 0}
    base_fetch = _deterministic_fetch()

    def flaky_fetch(url: str) -> PanelPayload:
        calls["count"] += 1
        if calls["count"] > 3:
            raise PanelCaptureError("simulated crash")
        return base_fetch(url)

    with pytest.raises(PanelCaptureError, match="simulated crash"):
        capture_panel(
            workspace_root=tmp_path,
            output_directory=target,
            reserve_bytes=0,
            symbols=("BTCUSDT", "ETHUSDT"),
            months=("2024-01", "2024-02"),
            fetch=flaky_fetch,
        )
    assert not (target / "capture-manifest.json").exists()
    assert (target / "capture-progress.jsonl").exists()

    artifact = capture_panel(
        workspace_root=tmp_path,
        output_directory=target,
        reserve_bytes=0,
        symbols=("BTCUSDT", "ETHUSDT"),
        months=("2024-01", "2024-02"),
        fetch=_deterministic_fetch(),
    )
    assert not (target / "capture-progress.jsonl").exists()
    assert verify_panel_capture(artifact.capture_root) == (True, ())

    control_manifest = json.loads(control.capture_manifest_path.read_text(encoding="utf-8"))
    resumed_manifest = json.loads(artifact.capture_manifest_path.read_text(encoding="utf-8"))
    # The properties that actually matter: the exact, ordered source list and
    # the capture root hash it feeds into -- not just a same-elements set.
    assert control_manifest["sources"] == resumed_manifest["sources"]
    assert control_manifest["dataset_root_hash"] == resumed_manifest["dataset_root_hash"]
    assert control.capture_root_hash == artifact.capture_root_hash
    assert control_manifest["capture_root_hash"] == resumed_manifest["capture_root_hash"]


def test_capture_with_a_manifest_present_still_refuses(tmp_path: Path) -> None:
    target = tmp_path / "capture"
    capture_panel(
        workspace_root=tmp_path,
        output_directory=target,
        reserve_bytes=0,
        symbols=("BTCUSDT",),
        months=("2024-01",),
        fetch=_deterministic_fetch(),
    )
    with pytest.raises(PanelCaptureError, match="immutable"):
        capture_panel(
            workspace_root=tmp_path,
            output_directory=target,
            reserve_bytes=0,
            symbols=("BTCUSDT",),
            months=("2024-01",),
            fetch=_deterministic_fetch(),
        )


def test_capture_deletes_the_progress_file_after_success(tmp_path: Path) -> None:
    artifact = capture_panel(
        workspace_root=tmp_path,
        output_directory=tmp_path / "capture",
        reserve_bytes=0,
        symbols=("BTCUSDT",),
        months=("2024-01",),
        fetch=_deterministic_fetch(),
    )
    assert not (artifact.capture_root / "capture-progress.jsonl").exists()


def test_capture_resumes_when_a_dataset_directory_is_left_without_a_manifest(
    tmp_path: Path,
) -> None:
    target = tmp_path / "capture"
    first = capture_panel(
        workspace_root=tmp_path,
        output_directory=target,
        reserve_bytes=0,
        symbols=("BTCUSDT",),
        months=("2024-01",),
        fetch=_deterministic_fetch(),
    )
    manifest = json.loads(first.capture_manifest_path.read_text(encoding="utf-8"))
    old_dataset_hash = manifest["dataset_root_hash"]

    # Simulate a crash between publishing the dataset and writing the
    # manifest: the progress file from that run is still present (recording
    # the capture parameters, header first, and every source that was
    # fetched), but the manifest never got written.
    progress_path = target / "capture-progress.jsonl"
    with progress_path.open("w", encoding="utf-8") as handle:
        handle.write(_progress_header(symbols=("BTCUSDT",), months=("2024-01",)) + "\n")
        for record in manifest["sources"]:
            handle.write(json.dumps(record) + "\n")
    first.capture_manifest_path.unlink()
    assert (target / "dataset").exists()

    def fetch_never(url: str) -> PanelPayload:
        raise AssertionError(f"unexpected network fetch during resume: {url}")

    resumed = capture_panel(
        workspace_root=tmp_path,
        output_directory=target,
        reserve_bytes=0,
        symbols=("BTCUSDT",),
        months=("2024-01",),
        fetch=fetch_never,
    )
    assert resumed.dataset_root_hash == old_dataset_hash
    assert not progress_path.exists()
    assert verify_panel_capture(resumed.capture_root) == (True, ())


def test_capture_resume_recovers_from_a_torn_final_progress_line(tmp_path: Path) -> None:
    target = tmp_path / "resumable"
    calls = {"count": 0}
    base_fetch = _deterministic_fetch()

    def flaky_fetch(url: str) -> PanelPayload:
        calls["count"] += 1
        if calls["count"] > 2:
            raise PanelCaptureError("simulated crash")
        return base_fetch(url)

    with pytest.raises(PanelCaptureError, match="simulated crash"):
        capture_panel(
            workspace_root=tmp_path,
            output_directory=target,
            reserve_bytes=0,
            symbols=("BTCUSDT",),
            months=("2024-01", "2024-02"),
            fetch=flaky_fetch,
        )

    # Simulate a hard kill mid-write of the next record: an incomplete JSON
    # fragment appended after the header and the two flushed, complete lines.
    progress_path = target / "capture-progress.jsonl"
    with progress_path.open("a", encoding="utf-8") as handle:
        handle.write('{"symbol": "BTCUSDT", "month": "2024-02", "kind": "kli')

    control = capture_panel(
        workspace_root=tmp_path,
        output_directory=tmp_path / "control",
        reserve_bytes=0,
        symbols=("BTCUSDT",),
        months=("2024-01", "2024-02"),
        fetch=_deterministic_fetch(),
    )
    resumed = capture_panel(
        workspace_root=tmp_path,
        output_directory=target,
        reserve_bytes=0,
        symbols=("BTCUSDT",),
        months=("2024-01", "2024-02"),
        fetch=_deterministic_fetch(),
    )
    assert verify_panel_capture(resumed.capture_root) == (True, ())
    control_manifest = json.loads(control.capture_manifest_path.read_text(encoding="utf-8"))
    resumed_manifest = json.loads(resumed.capture_manifest_path.read_text(encoding="utf-8"))
    assert control_manifest["sources"] == resumed_manifest["sources"]
    assert control_manifest["capture_root_hash"] == resumed_manifest["capture_root_hash"]


def test_capture_survives_a_second_crash_right_after_a_torn_line_repair(
    tmp_path: Path,
) -> None:
    """The reviewer's exact double-crash reproduction: a torn last line is
    repaired on resume, that resume makes one more successful write, then
    dies again. Before the fix, the repair never happened -- the next write
    concatenated onto the torn bytes with no separator, and once a further
    write followed *that* splice, it became an unparseable earlier line
    that permanently blocked every future resume."""

    target = tmp_path / "resumable"
    calls_1 = {"count": 0}
    base_fetch_1 = _deterministic_fetch()

    def flaky_fetch_1(url: str) -> PanelPayload:
        calls_1["count"] += 1
        if calls_1["count"] > 2:
            raise PanelCaptureError("simulated crash 1")
        return base_fetch_1(url)

    with pytest.raises(PanelCaptureError, match="simulated crash 1"):
        capture_panel(
            workspace_root=tmp_path,
            output_directory=target,
            reserve_bytes=0,
            symbols=("BTCUSDT",),
            months=("2024-01", "2024-02"),
            fetch=flaky_fetch_1,
        )

    # Simulate a hard kill mid-write of the next record.
    progress_path = target / "capture-progress.jsonl"
    with progress_path.open("a", encoding="utf-8") as handle:
        handle.write('{"symbol": "BTCUSDT", "month": "2024-02", "kind": "kli')

    # Resume: this repairs the torn line, then makes exactly one more
    # successful write before dying again.
    calls_2 = {"count": 0}
    base_fetch_2 = _deterministic_fetch()

    def flaky_fetch_2(url: str) -> PanelPayload:
        calls_2["count"] += 1
        if calls_2["count"] > 1:
            raise PanelCaptureError("simulated crash 2")
        return base_fetch_2(url)

    with pytest.raises(PanelCaptureError, match="simulated crash 2"):
        capture_panel(
            workspace_root=tmp_path,
            output_directory=target,
            reserve_bytes=0,
            symbols=("BTCUSDT",),
            months=("2024-01", "2024-02"),
            fetch=flaky_fetch_2,
        )
    assert not (target / "capture-manifest.json").exists()

    # Every line must parse independently -- no splice of the abandoned
    # torn prefix and the record written right after it.
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        json.loads(line)

    control = capture_panel(
        workspace_root=tmp_path,
        output_directory=tmp_path / "control",
        reserve_bytes=0,
        symbols=("BTCUSDT",),
        months=("2024-01", "2024-02"),
        fetch=_deterministic_fetch(),
    )
    resumed = capture_panel(
        workspace_root=tmp_path,
        output_directory=target,
        reserve_bytes=0,
        symbols=("BTCUSDT",),
        months=("2024-01", "2024-02"),
        fetch=_deterministic_fetch(),
    )
    assert verify_panel_capture(resumed.capture_root) == (True, ())
    control_manifest = json.loads(control.capture_manifest_path.read_text(encoding="utf-8"))
    resumed_manifest = json.loads(resumed.capture_manifest_path.read_text(encoding="utf-8"))
    assert control_manifest["sources"] == resumed_manifest["sources"]
    assert control_manifest["capture_root_hash"] == resumed_manifest["capture_root_hash"]


def test_capture_resume_raises_on_a_corrupt_earlier_progress_line(tmp_path: Path) -> None:
    target = tmp_path / "resumable"
    target.mkdir(parents=True)
    progress_path = target / "capture-progress.jsonl"
    lines = [
        _progress_header(symbols=("BTCUSDT",), months=("2024-01",)),
        '{"symbol": "BTCUSDT" this is not valid json',
        json.dumps(
            {
                "symbol": "BTCUSDT",
                "month": "2024-01",
                "kind": "fundingRate",
                "url": "u",
                "received_time_ns": 1,
                "raw_relative_path": "raw/BTCUSDT/fundingRate-2024-01.zip",
                "raw_sha256": "x",
                "status": "present",
            }
        ),
    ]
    progress_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(PanelCaptureError, match=re.escape(str(progress_path))):
        capture_panel(
            workspace_root=tmp_path,
            output_directory=target,
            reserve_bytes=0,
            symbols=("BTCUSDT",),
            months=("2024-01",),
            fetch=_deterministic_fetch(),
        )


def test_capture_resume_raises_on_a_structurally_wrong_progress_line(tmp_path: Path) -> None:
    # Valid JSON, wrong shape: a list instead of an object. Never reachable
    # from a torn write (truncated JSON never parses at all), but it must
    # still fail closed instead of raising a raw TypeError when indexed.
    target = tmp_path / "resumable"
    target.mkdir(parents=True)
    progress_path = target / "capture-progress.jsonl"
    lines = [
        _progress_header(symbols=("BTCUSDT",), months=("2024-01",)),
        json.dumps(["not", "a", "record"]),
    ]
    progress_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(PanelCaptureError, match=re.escape(str(progress_path))):
        capture_panel(
            workspace_root=tmp_path,
            output_directory=target,
            reserve_bytes=0,
            symbols=("BTCUSDT",),
            months=("2024-01",),
            fetch=_deterministic_fetch(),
        )


def test_capture_resume_raises_on_a_progress_line_missing_a_required_field(
    tmp_path: Path,
) -> None:
    # Valid JSON object, missing "symbol". Must fail closed instead of
    # raising a raw KeyError when indexed.
    target = tmp_path / "resumable"
    target.mkdir(parents=True)
    progress_path = target / "capture-progress.jsonl"
    lines = [
        _progress_header(symbols=("BTCUSDT",), months=("2024-01",)),
        json.dumps({"month": "2024-01", "kind": "klines", "status": "absent", "url": "u"}),
    ]
    progress_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(PanelCaptureError, match=re.escape(str(progress_path))):
        capture_panel(
            workspace_root=tmp_path,
            output_directory=target,
            reserve_bytes=0,
            symbols=("BTCUSDT",),
            months=("2024-01",),
            fetch=_deterministic_fetch(),
        )


def test_capture_resume_raises_on_a_present_progress_line_missing_its_own_fields(
    tmp_path: Path,
) -> None:
    # The reviewer's exact record: passes a symbol/month/kind/status check,
    # but a "present" entry also needs raw_relative_path, raw_sha256,
    # received_time_ns and url -- without them, resuming a capture that
    # re-requests this exact source raises a raw KeyError where the seeded
    # record is read back from disk, not PanelCaptureError.
    target = tmp_path / "resumable"
    target.mkdir(parents=True)
    progress_path = target / "capture-progress.jsonl"
    lines = [
        _progress_header(symbols=("BTCUSDT",), months=("2024-01",)),
        json.dumps(
            {"symbol": "BTCUSDT", "month": "2024-01", "kind": "klines", "status": "present"}
        ),
    ]
    progress_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(PanelCaptureError, match=re.escape(str(progress_path))):
        capture_panel(
            workspace_root=tmp_path,
            output_directory=target,
            reserve_bytes=0,
            symbols=("BTCUSDT",),
            months=("2024-01",),
            fetch=_deterministic_fetch(),
        )


def test_capture_resume_raises_on_an_absent_progress_line_missing_url(
    tmp_path: Path,
) -> None:
    target = tmp_path / "resumable"
    target.mkdir(parents=True)
    progress_path = target / "capture-progress.jsonl"
    lines = [
        _progress_header(symbols=("BTCUSDT",), months=("2024-01",)),
        json.dumps({"symbol": "BTCUSDT", "month": "2024-01", "kind": "klines", "status": "absent"}),
    ]
    progress_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(PanelCaptureError, match=re.escape(str(progress_path))):
        capture_panel(
            workspace_root=tmp_path,
            output_directory=target,
            reserve_bytes=0,
            symbols=("BTCUSDT",),
            months=("2024-01",),
            fetch=_deterministic_fetch(),
        )


def test_capture_resume_raises_on_a_present_progress_line_with_a_wrongly_typed_field(
    tmp_path: Path,
) -> None:
    # received_time_ns as a JSON bool: technically an int subclass in
    # Python, but never a valid nanosecond timestamp.
    target = tmp_path / "resumable"
    target.mkdir(parents=True)
    progress_path = target / "capture-progress.jsonl"
    lines = [
        _progress_header(symbols=("BTCUSDT",), months=("2024-01",)),
        json.dumps(
            {
                "symbol": "BTCUSDT",
                "month": "2024-01",
                "kind": "klines",
                "status": "present",
                "url": "u",
                "received_time_ns": True,
                "raw_relative_path": "raw/BTCUSDT/klines-2024-01.zip",
                "raw_sha256": "x",
            }
        ),
    ]
    progress_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(PanelCaptureError, match=re.escape(str(progress_path))):
        capture_panel(
            workspace_root=tmp_path,
            output_directory=target,
            reserve_bytes=0,
            symbols=("BTCUSDT",),
            months=("2024-01",),
            fetch=_deterministic_fetch(),
        )


def test_capture_resume_raises_on_a_progress_line_with_an_unrecognized_status(
    tmp_path: Path,
) -> None:
    target = tmp_path / "resumable"
    target.mkdir(parents=True)
    progress_path = target / "capture-progress.jsonl"
    lines = [
        _progress_header(symbols=("BTCUSDT",), months=("2024-01",)),
        json.dumps(
            {
                "symbol": "BTCUSDT",
                "month": "2024-01",
                "kind": "klines",
                "status": "bogus",
                "url": "u",
            }
        ),
    ]
    progress_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(PanelCaptureError, match=re.escape(str(progress_path))):
        capture_panel(
            workspace_root=tmp_path,
            output_directory=target,
            reserve_bytes=0,
            symbols=("BTCUSDT",),
            months=("2024-01",),
            fetch=_deterministic_fetch(),
        )


def test_capture_resume_raises_on_non_utf8_bytes_in_the_progress_file(
    tmp_path: Path,
) -> None:
    target = tmp_path / "resumable"
    target.mkdir(parents=True)
    progress_path = target / "capture-progress.jsonl"
    header = _progress_header(symbols=("BTCUSDT",), months=("2024-01",)).encode("utf-8")
    progress_path.write_bytes(header + b"\n" + b"\xff\xfe not utf-8 \x80\x81\n")

    with pytest.raises(PanelCaptureError, match=re.escape(str(progress_path))):
        capture_panel(
            workspace_root=tmp_path,
            output_directory=target,
            reserve_bytes=0,
            symbols=("BTCUSDT",),
            months=("2024-01",),
            fetch=_deterministic_fetch(),
        )


def test_capture_resume_refuses_when_a_symbol_is_dropped(tmp_path: Path) -> None:
    target = tmp_path / "resumable"
    calls = {"count": 0}
    base_fetch = _deterministic_fetch()

    def flaky_fetch(url: str) -> PanelPayload:
        calls["count"] += 1
        if calls["count"] > 3:
            raise PanelCaptureError("simulated crash")
        return base_fetch(url)

    with pytest.raises(PanelCaptureError, match="simulated crash"):
        capture_panel(
            workspace_root=tmp_path,
            output_directory=target,
            reserve_bytes=0,
            symbols=("BTCUSDT", "ETHUSDT"),
            months=("2024-01", "2024-02"),
            fetch=flaky_fetch,
        )

    # The natural response to a failed long run -- drop the bad symbol and
    # retry -- must be refused loudly, not silently produce a manifest that
    # overstates the dataset with ETHUSDT's already-recorded sources.
    with pytest.raises(PanelCaptureError, match="differ"):
        capture_panel(
            workspace_root=tmp_path,
            output_directory=target,
            reserve_bytes=0,
            symbols=("BTCUSDT",),
            months=("2024-01", "2024-02"),
            fetch=_deterministic_fetch(),
        )


def test_capture_bounded_window_tolerates_a_symbol_not_yet_listed(tmp_path: Path) -> None:
    """FTTUSDT-style repro: the FTT perpetual first listed 2022-04, so a pilot window
    of 2022-01..2022-03 (given as bounds, months omitted) discovers a kline month for
    it that does not survive the bounds. That is legitimate emptiness -- the contract
    had not started trading inside the window -- and must not abort the whole run, as
    long as another symbol (BTCUSDT here) has real data inside the window. FTTUSDT's
    own dump URLs must never even be requested, since none of its months are in
    bounds."""

    def fetch(url: str) -> PanelPayload:
        if "list-type=2" in url:
            kind = "fundingRate" if "fundingRate" in url else "klines"
            months: tuple[str, ...]
            if "FTTUSDT" in url:
                symbol, months = "FTTUSDT", ("2022-04",)
            else:
                symbol, months = "BTCUSDT", ("2022-01", "2022-02", "2022-03")
            if kind == "klines":
                keys = tuple(
                    f"data/futures/um/monthly/klines/{symbol}/1d/{symbol}-1d-{m}.zip"
                    for m in months
                )
            else:
                keys = tuple(
                    f"data/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{m}.zip"
                    for m in months
                )
            return PanelPayload(url=url, raw_bytes=_listing_xml(keys=keys), received_time_ns=1)
        if "FTTUSDT" in url:
            raise AssertionError(f"FTTUSDT dump should never be fetched: {url}")
        if "fundingRate" in url:
            return PanelPayload(
                url=url, raw_bytes=zip_bytes("f.csv", funding_csv()), received_time_ns=1
            )
        return PanelPayload(url=url, raw_bytes=zip_bytes("k.csv", kline_csv(3)), received_time_ns=1)

    artifact = capture_panel(
        workspace_root=tmp_path,
        output_directory=tmp_path / "capture",
        reserve_bytes=0,
        symbols=("BTCUSDT", "FTTUSDT"),
        month_from="2022-01",
        month_to="2022-03",
        fetch=fetch,
    )
    manifest = json.loads(artifact.capture_manifest_path.read_text(encoding="utf-8"))
    assert manifest["discovered_months"]["FTTUSDT"]["klines"] == []
    assert all(entry["symbol"] != "FTTUSDT" for entry in manifest["sources"])
    assert verify_panel_capture(artifact.capture_root) == (True, ())


def test_capture_bounded_window_still_rejects_a_symbol_with_no_history_at_all(
    tmp_path: Path,
) -> None:
    """Distinguishes the two empty-row cases month bounds create: a symbol that
    discovered months but none survive the bounds (tolerated, above) from a symbol
    that discovered nothing at all in the bucket, at any date -- still a wrong
    symbol, and the guard must still fire even though bounds were given."""

    def fetch(url: str) -> PanelPayload:
        if "list-type=2" in url:
            if "GHOSTUSDT" in url:
                keys: tuple[str, ...] = ()
            elif "fundingRate" in url:
                keys = (
                    "data/futures/um/monthly/fundingRate/BTCUSDT/"
                    "BTCUSDT-fundingRate-2022-01.zip",
                )
            else:
                keys = ("data/futures/um/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2022-01.zip",)
            return PanelPayload(url=url, raw_bytes=_listing_xml(keys=keys), received_time_ns=1)
        if "fundingRate" in url:
            return PanelPayload(
                url=url, raw_bytes=zip_bytes("f.csv", funding_csv()), received_time_ns=1
            )
        return PanelPayload(url=url, raw_bytes=zip_bytes("k.csv", kline_csv(3)), received_time_ns=1)

    with pytest.raises(PanelCaptureError, match="GHOSTUSDT"):
        capture_panel(
            workspace_root=tmp_path,
            output_directory=tmp_path / "capture",
            reserve_bytes=0,
            symbols=("BTCUSDT", "GHOSTUSDT"),
            month_from="2022-01",
            month_to="2022-03",
            fetch=fetch,
        )


def test_capture_resume_refuses_when_the_month_window_narrows(tmp_path: Path) -> None:
    target = tmp_path / "resumable"
    calls = {"count": 0}
    base_fetch = _deterministic_fetch()

    def flaky_fetch(url: str) -> PanelPayload:
        calls["count"] += 1
        if calls["count"] > 3:
            raise PanelCaptureError("simulated crash")
        return base_fetch(url)

    with pytest.raises(PanelCaptureError, match="simulated crash"):
        capture_panel(
            workspace_root=tmp_path,
            output_directory=target,
            reserve_bytes=0,
            symbols=("BTCUSDT",),
            months=("2024-01", "2024-02"),
            fetch=flaky_fetch,
        )

    # The other natural response -- shorten the month window and retry --
    # must be refused for the same reason.
    with pytest.raises(PanelCaptureError, match="differ"):
        capture_panel(
            workspace_root=tmp_path,
            output_directory=target,
            reserve_bytes=0,
            symbols=("BTCUSDT",),
            months=("2024-01",),
            fetch=_deterministic_fetch(),
        )


# --- Gap-filling: a hole in the monthly aggregate, closed from the daily dumps. ---

_GAP_KLINE_ROWS = (0, 1, 3)  # day index 2 is missing


def _gap_monthly_csv() -> str:
    lines = [KLINE_HEADER]
    for index in _GAP_KLINE_ROWS:
        open_ms = index * DAY_MS
        lines.append(
            f"{open_ms},100,101,99,{100 + index},10,{open_ms + DAY_MS - 1},1000,5,6,600,0"
        )
    return "\n".join(lines) + "\n"


def _daily_row_csv(day_index: int) -> str:
    open_ms = day_index * DAY_MS
    return (
        KLINE_HEADER
        + "\n"
        + f"{open_ms},100,101,99,{100 + day_index},10,{open_ms + DAY_MS - 1},1000,5,6,600,0\n"
    )


def _daily_malformed_csv(day_index: int) -> str:
    # A daily dump with two rows instead of exactly one -- malformed, but not
    # a shape `parse_kline_zip` itself rejects; only the single-day check
    # inside `_fill_gap_days` catches it.
    lines = [KLINE_HEADER]
    for offset in (0, 1):
        day = day_index + offset
        open_ms = day * DAY_MS
        lines.append(
            f"{open_ms},100,101,99,{100 + day},10,{open_ms + DAY_MS - 1},1000,5,6,600,0"
        )
    return "\n".join(lines) + "\n"


def _gap_capture_fetch(
    *, daily_fetch: Callable[[str], PanelPayload]
) -> Callable[[str], PanelPayload]:
    def fetch(url: str) -> PanelPayload:
        if "/daily/klines/" in url:
            return daily_fetch(url)
        if "fundingRate" in url:
            return PanelPayload(
                url=url, raw_bytes=zip_bytes("f.csv", funding_csv()), received_time_ns=1
            )
        return PanelPayload(
            url=url, raw_bytes=zip_bytes("k.csv", _gap_monthly_csv()), received_time_ns=1
        )

    return fetch


def _daily_fill_success(url: str) -> PanelPayload:
    # Day index 2 is 1970-01-03 (epoch day 2); asserting the exact URL pins
    # down that the missing day, not some other day, is what gets requested.
    assert url == build_daily_kline_zip_url("BTCUSDT", "1970-01-03")
    return PanelPayload(
        url=url, raw_bytes=zip_bytes("d.csv", _daily_row_csv(2)), received_time_ns=2
    )


def test_capture_fills_a_monthly_gap_from_the_daily_dump(tmp_path: Path) -> None:
    artifact = capture_panel(
        workspace_root=tmp_path,
        output_directory=tmp_path / "capture",
        reserve_bytes=0,
        symbols=("BTCUSDT",),
        months=("2024-01",),
        fetch=_gap_capture_fetch(daily_fetch=_daily_fill_success),
    )
    manifest = json.loads(artifact.capture_manifest_path.read_text(encoding="utf-8"))
    fill_entries = [entry for entry in manifest["sources"] if entry["kind"] == "klines_daily_fill"]
    assert len(fill_entries) == 1
    assert fill_entries[0]["status"] == "present"
    assert fill_entries[0]["symbol"] == "BTCUSDT"
    assert fill_entries[0]["month"] == "1970-01-03"
    quality = json.loads(
        (artifact.dataset_root / "quality-report.json").read_text(encoding="utf-8")
    )
    btc = next(item for item in quality["instruments"] if item["instrument_id"] == "BTCUSDT")
    assert btc["missing_days"] == 0
    assert btc["row_count"] == 4
    assert verify_panel_capture(artifact.capture_root) == (True, ())


def test_capture_records_an_unfillable_gap_as_absent_and_continues(tmp_path: Path) -> None:
    def daily_absent(url: str) -> PanelPayload:
        raise PanelSourceAbsent("404: no daily dump for this day either")

    artifact = capture_panel(
        workspace_root=tmp_path,
        output_directory=tmp_path / "capture",
        reserve_bytes=0,
        symbols=("BTCUSDT",),
        months=("2024-01",),
        fetch=_gap_capture_fetch(daily_fetch=daily_absent),
    )
    manifest = json.loads(artifact.capture_manifest_path.read_text(encoding="utf-8"))
    fill_entries = [entry for entry in manifest["sources"] if entry["kind"] == "klines_daily_fill"]
    assert len(fill_entries) == 1
    assert fill_entries[0]["status"] == "absent"
    assert "raw_relative_path" not in fill_entries[0]
    quality = json.loads(
        (artifact.dataset_root / "quality-report.json").read_text(encoding="utf-8")
    )
    btc = next(item for item in quality["instruments"] if item["instrument_id"] == "BTCUSDT")
    assert btc["missing_days"] == 1
    assert verify_panel_capture(artifact.capture_root) == (True, ())


def test_capture_resumes_mid_gap_fill_and_matches_an_uninterrupted_run(tmp_path: Path) -> None:
    base_fetch = _gap_capture_fetch(daily_fetch=_daily_fill_success)
    control = capture_panel(
        workspace_root=tmp_path,
        output_directory=tmp_path / "control",
        reserve_bytes=0,
        symbols=("BTCUSDT",),
        months=("2024-01",),
        fetch=base_fetch,
    )

    target = tmp_path / "resumable"

    def daily_crash(url: str) -> PanelPayload:
        raise PanelCaptureError("simulated crash during gap-fill")

    with pytest.raises(PanelCaptureError, match="simulated crash during gap-fill"):
        capture_panel(
            workspace_root=tmp_path,
            output_directory=target,
            reserve_bytes=0,
            symbols=("BTCUSDT",),
            months=("2024-01",),
            fetch=_gap_capture_fetch(daily_fetch=daily_crash),
        )
    assert not (target / "capture-manifest.json").exists()

    def refetch_forbidden(url: str) -> PanelPayload:
        raise AssertionError(f"resume must not refetch an already-recorded monthly source: {url}")

    def resume_fetch(url: str) -> PanelPayload:
        if "/daily/klines/" in url:
            return _daily_fill_success(url)
        return refetch_forbidden(url)

    resumed = capture_panel(
        workspace_root=tmp_path,
        output_directory=target,
        reserve_bytes=0,
        symbols=("BTCUSDT",),
        months=("2024-01",),
        fetch=resume_fetch,
    )
    assert verify_panel_capture(resumed.capture_root) == (True, ())
    control_manifest = json.loads(control.capture_manifest_path.read_text(encoding="utf-8"))
    resumed_manifest = json.loads(resumed.capture_manifest_path.read_text(encoding="utf-8"))
    assert control_manifest["sources"] == resumed_manifest["sources"]
    assert control_manifest["capture_root_hash"] == resumed_manifest["capture_root_hash"]


# --- Repair: heal a completed capture's gaps into a new capture, without
# refetching any of its monthly sources. ---


def test_repair_refuses_when_the_source_capture_does_not_verify(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir(parents=True)
    (source / "capture-manifest.json").write_text("not json", encoding="utf-8")
    with pytest.raises(PanelCaptureError, match="failed verification"):
        repair_panel_capture(
            workspace_root=tmp_path,
            source_capture_root=source,
            output_directory=tmp_path / "repaired",
            reserve_bytes=0,
        )


def test_repair_fills_the_sources_gap_without_refetching_its_monthly_sources(
    tmp_path: Path,
) -> None:
    # The source capture never managed to fill its own gap (the daily dump
    # 404s at capture time -- a genuine absence back then); the repair is run
    # later, when the daily dump does have the day, and must fetch only that
    # one day, never any of the monthly sources already on disk.
    source = capture_panel(
        workspace_root=tmp_path,
        output_directory=tmp_path / "source",
        reserve_bytes=0,
        symbols=("BTCUSDT",),
        months=("2024-01",),
        fetch=_gap_capture_fetch(
            daily_fetch=lambda url: (_ for _ in ()).throw(
                PanelSourceAbsent("404: not yet published")
            )
        ),
    )
    source_quality = json.loads(
        (source.dataset_root / "quality-report.json").read_text(encoding="utf-8")
    )
    assert source_quality["instruments"][0]["missing_days"] == 1

    def guarded_repair_fetch(url: str) -> PanelPayload:
        if "/daily/klines/" in url:
            return _daily_fill_success(url)
        raise AssertionError(f"repair must not refetch a monthly source: {url}")

    repaired = repair_panel_capture(
        workspace_root=tmp_path,
        source_capture_root=source.capture_root,
        output_directory=tmp_path / "repaired",
        reserve_bytes=0,
        fetch=guarded_repair_fetch,
    )
    assert verify_panel_capture(repaired.capture_root) == (True, ())
    # The source stays exactly as it was: untouched and still verifying.
    assert verify_panel_capture(source.capture_root) == (True, ())

    manifest = json.loads(repaired.capture_manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_capture_root_hash"] == source.capture_root_hash
    kinds = {entry["kind"] for entry in manifest["sources"]}
    assert kinds == {"klines", "fundingRate", "klines_daily_fill"}
    fill_entries = [entry for entry in manifest["sources"] if entry["kind"] == "klines_daily_fill"]
    assert len(fill_entries) == 1
    assert fill_entries[0]["status"] == "present"

    repaired_quality = json.loads(
        (repaired.dataset_root / "quality-report.json").read_text(encoding="utf-8")
    )
    assert repaired_quality["instruments"][0]["missing_days"] == 0
    assert repaired.capture_root_hash != source.capture_root_hash


def test_repair_refuses_when_the_output_already_exists(tmp_path: Path) -> None:
    source = capture_panel(
        workspace_root=tmp_path,
        output_directory=tmp_path / "source",
        reserve_bytes=0,
        symbols=("BTCUSDT",),
        months=("2024-01",),
        fetch=_deterministic_fetch(),
    )
    existing = tmp_path / "repaired"
    capture_panel(
        workspace_root=tmp_path,
        output_directory=existing,
        reserve_bytes=0,
        symbols=("BTCUSDT",),
        months=("2024-01",),
        fetch=_deterministic_fetch(),
    )
    with pytest.raises(PanelCaptureError, match="immutable"):
        repair_panel_capture(
            workspace_root=tmp_path,
            source_capture_root=source.capture_root,
            output_directory=existing,
            reserve_bytes=0,
        )


def test_repair_resumes_after_an_interruption_and_matches_an_uninterrupted_run(
    tmp_path: Path,
) -> None:
    source = capture_panel(
        workspace_root=tmp_path,
        output_directory=tmp_path / "source",
        reserve_bytes=0,
        symbols=("BTCUSDT",),
        months=("2024-01",),
        fetch=_gap_capture_fetch(
            daily_fetch=lambda url: (_ for _ in ()).throw(
                PanelSourceAbsent("404: not yet published")
            )
        ),
    )

    control = repair_panel_capture(
        workspace_root=tmp_path,
        source_capture_root=source.capture_root,
        output_directory=tmp_path / "control",
        reserve_bytes=0,
        fetch=_daily_fill_success,
    )

    target = tmp_path / "resumable"

    def daily_crash(url: str) -> PanelPayload:
        raise PanelCaptureError("simulated crash during repair fill")

    with pytest.raises(PanelCaptureError, match="simulated crash during repair fill"):
        repair_panel_capture(
            workspace_root=tmp_path,
            source_capture_root=source.capture_root,
            output_directory=target,
            reserve_bytes=0,
            fetch=daily_crash,
        )
    assert not (target / "capture-manifest.json").exists()
    assert (target / "capture-progress.jsonl").exists()

    resumed = repair_panel_capture(
        workspace_root=tmp_path,
        source_capture_root=source.capture_root,
        output_directory=target,
        reserve_bytes=0,
        fetch=_daily_fill_success,
    )
    assert verify_panel_capture(resumed.capture_root) == (True, ())
    control_manifest = json.loads(control.capture_manifest_path.read_text(encoding="utf-8"))
    resumed_manifest = json.loads(resumed.capture_manifest_path.read_text(encoding="utf-8"))
    assert control_manifest["sources"] == resumed_manifest["sources"]
    assert control_manifest["capture_root_hash"] == resumed_manifest["capture_root_hash"]


def test_repair_resume_refuses_when_the_source_capture_differs(tmp_path: Path) -> None:
    first_source = capture_panel(
        workspace_root=tmp_path,
        output_directory=tmp_path / "source-a",
        reserve_bytes=0,
        symbols=("BTCUSDT",),
        months=("2024-01",),
        fetch=_gap_capture_fetch(
            daily_fetch=lambda url: (_ for _ in ()).throw(
                PanelSourceAbsent("404: not yet published")
            )
        ),
    )
    second_source = capture_panel(
        workspace_root=tmp_path,
        output_directory=tmp_path / "source-b",
        reserve_bytes=0,
        symbols=("ETHUSDT",),
        months=("2024-01",),
        fetch=_deterministic_fetch(),
    )

    target = tmp_path / "resumable"

    def daily_crash(url: str) -> PanelPayload:
        raise PanelCaptureError("simulated crash during repair fill")

    with pytest.raises(PanelCaptureError, match="simulated crash during repair fill"):
        repair_panel_capture(
            workspace_root=tmp_path,
            source_capture_root=first_source.capture_root,
            output_directory=target,
            reserve_bytes=0,
            fetch=daily_crash,
        )

    with pytest.raises(PanelCaptureError, match="differ"):
        repair_panel_capture(
            workspace_root=tmp_path,
            source_capture_root=second_source.capture_root,
            output_directory=target,
            reserve_bytes=0,
            fetch=_deterministic_fetch(),
        )


# --- Hardening: a malformed daily payload must not wedge resume. ---


def test_capture_recovers_after_a_malformed_daily_payload_is_rejected(tmp_path: Path) -> None:
    # A payload with more than one row is rejected by `_fill_gap_days`'
    # single-day check -- but that must happen *before* the raw bytes are
    # written to disk and the source recorded present. Otherwise a resume
    # would find the poisoned bytes already on disk and the (symbol, date,
    # kind) key already in `already_done`, and would keep re-reading and
    # re-rejecting the same bad payload forever without ever asking
    # `download` for the URL again.
    calls = {"count": 0}

    def daily_fetch(url: str) -> PanelPayload:
        calls["count"] += 1
        if calls["count"] == 1:
            return PanelPayload(
                url=url, raw_bytes=zip_bytes("d.csv", _daily_malformed_csv(2)), received_time_ns=2
            )
        return _daily_fill_success(url)

    target = tmp_path / "capture"
    with pytest.raises(PanelCaptureError, match="did not contain exactly the requested day"):
        capture_panel(
            workspace_root=tmp_path,
            output_directory=target,
            reserve_bytes=0,
            symbols=("BTCUSDT",),
            months=("2024-01",),
            fetch=_gap_capture_fetch(daily_fetch=daily_fetch),
        )
    assert calls["count"] == 1
    assert not (target / "raw" / "BTCUSDT" / "klines_daily_fill-1970-01-03.zip").exists()
    progress_text = (target / "capture-progress.jsonl").read_text(encoding="utf-8")
    assert "klines_daily_fill" not in progress_text

    resumed = capture_panel(
        workspace_root=tmp_path,
        output_directory=target,
        reserve_bytes=0,
        symbols=("BTCUSDT",),
        months=("2024-01",),
        fetch=_gap_capture_fetch(daily_fetch=daily_fetch),
    )
    # The daily URL was asked for again on resume, not skipped as
    # already-done: proof the poisoned attempt left no trace to wedge on.
    assert calls["count"] == 2
    quality = json.loads(
        (resumed.dataset_root / "quality-report.json").read_text(encoding="utf-8")
    )
    btc = next(item for item in quality["instruments"] if item["instrument_id"] == "BTCUSDT")
    assert btc["missing_days"] == 0
    assert verify_panel_capture(resumed.capture_root) == (True, ())


def test_repair_recovers_after_a_malformed_daily_payload_is_rejected(tmp_path: Path) -> None:
    source = capture_panel(
        workspace_root=tmp_path,
        output_directory=tmp_path / "source",
        reserve_bytes=0,
        symbols=("BTCUSDT",),
        months=("2024-01",),
        fetch=_gap_capture_fetch(
            daily_fetch=lambda url: (_ for _ in ()).throw(
                PanelSourceAbsent("404: not yet published")
            )
        ),
    )

    calls = {"count": 0}

    def daily_fetch(url: str) -> PanelPayload:
        calls["count"] += 1
        if calls["count"] == 1:
            return PanelPayload(
                url=url, raw_bytes=zip_bytes("d.csv", _daily_malformed_csv(2)), received_time_ns=2
            )
        return _daily_fill_success(url)

    target = tmp_path / "repaired"
    with pytest.raises(PanelCaptureError, match="did not contain exactly the requested day"):
        repair_panel_capture(
            workspace_root=tmp_path,
            source_capture_root=source.capture_root,
            output_directory=target,
            reserve_bytes=0,
            fetch=daily_fetch,
        )
    assert calls["count"] == 1
    assert not (target / "raw" / "BTCUSDT" / "klines_daily_fill-1970-01-03.zip").exists()
    progress_text = (target / "capture-progress.jsonl").read_text(encoding="utf-8")
    assert "klines_daily_fill" not in progress_text

    repaired = repair_panel_capture(
        workspace_root=tmp_path,
        source_capture_root=source.capture_root,
        output_directory=target,
        reserve_bytes=0,
        fetch=daily_fetch,
    )
    assert calls["count"] == 2
    quality = json.loads(
        (repaired.dataset_root / "quality-report.json").read_text(encoding="utf-8")
    )
    assert quality["instruments"][0]["missing_days"] == 0
    assert verify_panel_capture(repaired.capture_root) == (True, ())


# --- Hardening: repair must refuse when source and output nest. ---


def test_repair_refuses_when_the_output_is_nested_inside_the_source(tmp_path: Path) -> None:
    source = capture_panel(
        workspace_root=tmp_path,
        output_directory=tmp_path / "source",
        reserve_bytes=0,
        symbols=("BTCUSDT",),
        months=("2024-01",),
        fetch=_deterministic_fetch(),
    )
    with pytest.raises(PanelCaptureError, match="nested"):
        repair_panel_capture(
            workspace_root=tmp_path,
            source_capture_root=source.capture_root,
            output_directory=source.capture_root / "repaired-inside",
            reserve_bytes=0,
        )
    assert not (source.capture_root / "repaired-inside").exists()
    assert verify_panel_capture(source.capture_root) == (True, ())


def test_repair_refuses_when_the_source_is_nested_inside_the_output(tmp_path: Path) -> None:
    outer = tmp_path / "outer"
    source = capture_panel(
        workspace_root=tmp_path,
        output_directory=outer / "nested-source",
        reserve_bytes=0,
        symbols=("BTCUSDT",),
        months=("2024-01",),
        fetch=_deterministic_fetch(),
    )
    with pytest.raises(PanelCaptureError, match="nested"):
        repair_panel_capture(
            workspace_root=tmp_path,
            source_capture_root=source.capture_root,
            output_directory=outer,
            reserve_bytes=0,
        )
    assert not (outer / "capture-manifest.json").exists()
    assert verify_panel_capture(source.capture_root) == (True, ())


def test_spot_urls_use_the_spot_prefix() -> None:
    from trading_bot.panel_capture import build_daily_kline_zip_url, build_kline_zip_url

    assert build_kline_zip_url("BTCUSDT", "2024-01", market="spot") == (
        "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2024-01.zip"
    )
    assert build_daily_kline_zip_url("BTCUSDT", "2024-01-15", market="spot") == (
        "https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1d/BTCUSDT-1d-2024-01-15.zip"
    )
    assert build_kline_zip_url("BTCUSDT", "2024-01") == (
        "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2024-01.zip"
    )


def test_spot_capture_fetches_no_funding_and_records_its_market(tmp_path: Path) -> None:
    urls: list[str] = []

    def fetch(url: str) -> PanelPayload:
        urls.append(url)
        assert "fundingRate" not in url
        assert "/data/spot/" in url
        return PanelPayload(url=url, raw_bytes=zip_bytes("k.csv", kline_csv(3)), received_time_ns=1)

    artifact = capture_panel(
        workspace_root=tmp_path,
        output_directory=tmp_path / "spot",
        reserve_bytes=0,
        symbols=("BTCUSDT",),
        months=("2024-01",),
        fetch=fetch,
        market="spot",
    )
    manifest = json.loads(artifact.capture_manifest_path.read_text(encoding="utf-8"))
    assert manifest["market"] == "spot"
    assert manifest["venue"] == "BINANCE_SPOT"
    assert {s["kind"] for s in manifest["sources"]} == {"klines"}
    dataset_manifest = json.loads(
        (artifact.dataset_root / "dataset-manifest.json").read_text(encoding="utf-8")
    )
    assert dataset_manifest["funding_row_count"] == 0
    assert verify_panel_capture(artifact.capture_root) == (True, ())
    assert len(urls) == 1


def test_default_market_manifest_names_um(tmp_path: Path) -> None:
    def fetch(url: str) -> PanelPayload:
        if "fundingRate" in url:
            text = "calc_time,funding_interval_hours,last_funding_rate\n0,8,0.0001\n"
            return PanelPayload(url=url, raw_bytes=zip_bytes("f.csv", text), received_time_ns=1)
        return PanelPayload(url=url, raw_bytes=zip_bytes("k.csv", kline_csv(3)), received_time_ns=1)

    artifact = capture_panel(
        workspace_root=tmp_path, output_directory=tmp_path / "um", reserve_bytes=0,
        symbols=("BTCUSDT",), months=("2024-01",), fetch=fetch,
    )
    manifest = json.loads(artifact.capture_manifest_path.read_text(encoding="utf-8"))
    assert manifest["market"] == "um"
    assert manifest["venue"] == "BINANCE_UM"


def test_repair_of_a_spot_capture_stays_on_the_spot_market(tmp_path: Path) -> None:
    def fetch(url: str) -> PanelPayload:
        assert "/data/spot/" in url
        return PanelPayload(url=url, raw_bytes=zip_bytes("k.csv", kline_csv(3)), received_time_ns=1)

    source = capture_panel(
        workspace_root=tmp_path, output_directory=tmp_path / "spot", reserve_bytes=0,
        symbols=("BTCUSDT",), months=("2024-01",), fetch=fetch, market="spot",
    )
    repaired = repair_panel_capture(
        workspace_root=tmp_path, source_capture_root=source.capture_root,
        output_directory=tmp_path / "spot-repaired", reserve_bytes=0, fetch=fetch,
    )
    manifest = json.loads(repaired.capture_manifest_path.read_text(encoding="utf-8"))
    assert manifest["market"] == "spot"
    assert manifest["venue"] == "BINANCE_SPOT"
    assert verify_panel_capture(repaired.capture_root) == (True, ())


def test_parse_kline_zip_normalises_microsecond_timestamps_to_millisecond_precision() -> None:
    """Binance's spot daily dumps carry 16-digit microsecond open/close times
    from 2025-01 onward (the perpetual dumps and earlier spot dumps carry
    13-digit milliseconds). Both eras must parse to the same nanosecond
    timestamps: the open is a whole day either way, and the close is
    normalised to `open + 1d - 1ms`, so a spot close still matches the
    perpetual calendar's decision close to the nanosecond."""
    from trading_bot.panel_capture import parse_kline_zip

    header = (
        "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
        "taker_buy_volume,taker_buy_quote_volume,ignore\n"
    )
    ms_row = "1733011200000,1,1,1,1,10,1733097599999,100,1,1,1,0\n"
    us_row = "1733011200000000,1,1,1,1,10,1733097599999999,100,1,1,1,0\n"
    ms_rows = parse_kline_zip(
        PanelPayload(url="u", raw_bytes=zip_bytes("k.csv", header + ms_row), received_time_ns=1),
        symbol="BTCUSDT",
    )
    us_rows = parse_kline_zip(
        PanelPayload(url="u", raw_bytes=zip_bytes("k.csv", header + us_row), received_time_ns=1),
        symbol="BTCUSDT",
    )
    assert ms_rows[0].open_time_ns == 1733011200000 * 1_000_000
    assert ms_rows[0].close_time_ns == 1733097599999 * 1_000_000
    assert us_rows[0].open_time_ns == ms_rows[0].open_time_ns
    assert us_rows[0].close_time_ns == ms_rows[0].close_time_ns
    assert us_rows[0].available_time_ns == ms_rows[0].available_time_ns
