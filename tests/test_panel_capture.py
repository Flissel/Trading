import io
import zipfile
from pathlib import Path

import pytest

from trading_bot.panel_capture import (
    PanelCaptureError,
    PanelPayload,
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


def zip_bytes(name: str, text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, text)
    return buffer.getvalue()


def kline_csv(days: int, *, with_header: bool = True) -> str:
    lines = [KLINE_HEADER] if with_header else []
    for index in range(days):
        open_ms = index * DAY_MS
        lines.append(
            f"{open_ms},100,101,99,{100 + index},10,{open_ms + DAY_MS - 1},1000,5,6,600,0"
        )
    return "\n".join(lines) + "\n"


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
