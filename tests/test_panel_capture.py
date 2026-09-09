import io
import json
import urllib.error
import urllib.request
import zipfile
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
    client = PanelZipClient()
    with pytest.raises(PanelSourceAbsent):
        client.fetch(build_kline_zip_url("LUNAUSDT", "2023-01"))

    def raise_500(request: urllib.request.Request, timeout: float) -> None:
        raise urllib.error.HTTPError(request.full_url, 500, "Server Error", Message(), None)

    monkeypatch.setattr(urllib.request, "urlopen", raise_500)
    with pytest.raises(PanelCaptureError) as excinfo:
        client.fetch(build_kline_zip_url("LUNAUSDT", "2023-01"))
    assert not isinstance(excinfo.value, PanelSourceAbsent)


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
