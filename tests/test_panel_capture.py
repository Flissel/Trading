import http.client
import io
import json
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
    build_funding_zip_url,
    build_kline_zip_url,
    capture_panel,
    discover_panel_months,
    parse_funding_zip,
    parse_kline_zip,
    verify_panel_capture,
)

KLINE_HEADER = (
    "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
    "taker_buy_volume,taker_buy_quote_volume,ignore"
)
DAY_MS = 86_400_000
MONTH_START_DAY = {"2024-01": 0, "2024-02": 31}


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
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, text)
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
    assert control_manifest["dataset_root_hash"] == resumed_manifest["dataset_root_hash"]
    assert {
        (e["symbol"], e["month"], e["kind"], e["status"]) for e in control_manifest["sources"]
    } == {
        (e["symbol"], e["month"], e["kind"], e["status"]) for e in resumed_manifest["sources"]
    }


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
    # every source that was fetched), but the manifest never got written.
    progress_path = target / "capture-progress.jsonl"
    with progress_path.open("w", encoding="utf-8") as handle:
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
