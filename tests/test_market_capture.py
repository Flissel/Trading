import json
from pathlib import Path

import pytest

from trading_bot.cli import main
from trading_bot.market_capture import (
    CapturedPayload,
    CaptureError,
    PublicJsonClient,
    build_binance_klines_url,
    build_okx_candles_url,
    fetch_history_payloads,
    publish_candle_capture,
    publish_candle_history,
    verify_candle_capture,
)
from trading_bot.storage import StoragePolicyError


def okx_bytes() -> bytes:
    return json.dumps(
        {
            "code": "0",
            "msg": "",
            "data": [["1000", "100", "103", "99", "102", "10", "1", "1000", "1"]],
        },
        separators=(",", ":"),
    ).encode()


def binance_bytes() -> bytes:
    return json.dumps(
        [[1000, "100", "103", "99", "102", "10", 1999, "1000", 42, "6", "600", "0"]],
        separators=(",", ":"),
    ).encode()


def okx_page(*timestamps_ms: int) -> bytes:
    rows = [[str(ts), "100", "103", "99", "102", "10", "1", "1000", "1"] for ts in timestamps_ms]
    return json.dumps({"code": "0", "msg": "", "data": rows}, separators=(",", ":")).encode()


def binance_page(*open_times_ms: int) -> bytes:
    rows = [
        [ts, "100", "103", "99", "102", "10", ts + 899_999, "1000", 42, "6", "600", "0"]
        for ts in open_times_ms
    ]
    return json.dumps(rows, separators=(",", ":")).encode()


def test_official_urls_are_bounded_and_encoded() -> None:
    assert build_okx_candles_url(limit=96, bar="15m") == (
        "https://www.okx.com/api/v5/market/history-candles?instId=BTC-USDT-SWAP&bar=15m&limit=96"
    )
    assert build_binance_klines_url(limit=96, interval="15m") == (
        "https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=15m&limit=96"
    )
    with pytest.raises(CaptureError, match="limit"):
        build_okx_candles_url(limit=301, bar="15m")


def test_history_urls_encode_backward_cursors() -> None:
    assert build_okx_candles_url(limit=300, bar="15m", after_ms=123).endswith(
        "bar=15m&limit=300&after=123"
    )
    assert build_binance_klines_url(limit=1500, interval="15m", end_time_ms=456).endswith(
        "interval=15m&limit=1500&endTime=456"
    )


def test_history_fetch_pages_backward_with_bounded_requests() -> None:
    responses = {
        "OKX": [okx_page(2_700_000, 1_800_000), okx_page(900_000, 0)],
        "BINANCE": [binance_page(1_800_000, 2_700_000), binance_page(0, 900_000)],
    }
    calls: list[str] = []

    def fetch(url: str) -> CapturedPayload:
        venue = "OKX" if "okx.com" in url else "BINANCE"
        calls.append(url)
        raw = responses[venue].pop(0)
        return CapturedPayload(url, raw, 4_000_000_000_000)

    okx, binance = fetch_history_payloads(fetch, bars=4, end_time_ms=3_599_999)

    assert len(okx) == 2
    assert len(binance) == 2
    assert "after=1800000" in calls[1]
    assert "endTime=1799999" in calls[3]


def test_history_bounds_admit_protocol_window_but_reject_unbounded_request() -> None:
    def empty_fetch(url: str) -> CapturedPayload:
        raw = b'{"code":"0","data":[]}' if "okx.com" in url else b"[]"
        return CapturedPayload(url, raw, 1)

    with pytest.raises(CaptureError, match="ended"):
        fetch_history_payloads(empty_fetch, bars=49_920, end_time_ms=1)
    with pytest.raises(CaptureError, match="bounds"):
        fetch_history_payloads(empty_fetch, bars=60_001, end_time_ms=1)


def test_history_publication_preserves_each_page_and_exact_latest_rows(tmp_path: Path) -> None:
    output = tmp_path / "history"
    okx = (
        CapturedPayload("https://www.okx.com/a", okx_page(2_700_000, 1_800_000), 4_000_000_000_000),
        CapturedPayload(
            "https://www.okx.com/b", okx_page(1_800_000, 900_000, 0), 4_000_000_000_000
        ),
    )
    binance = (
        CapturedPayload(
            "https://fapi.binance.com/a",
            binance_page(1_800_000, 2_700_000),
            4_000_000_000_000,
        ),
        CapturedPayload("https://fapi.binance.com/b", binance_page(0, 900_000), 4_000_000_000_000),
    )

    artifact = publish_candle_history(okx, binance, output_directory=output, bars=4)

    assert artifact.dataset.quality.row_count == 8
    assert artifact.dataset.quality.duplicate_rows == 0
    assert (output / "raw" / "okx" / "page-0000.json").read_bytes() == okx[0].raw_bytes
    assert (output / "raw" / "binance" / "page-0001.json").read_bytes() == binance[1].raw_bytes
    assert verify_candle_capture(output).valid is True


def test_public_client_rejects_unapproved_or_insecure_urls_before_network() -> None:
    client = PublicJsonClient(max_response_bytes=1024, timeout_seconds=1)

    with pytest.raises(CaptureError, match="HTTPS"):
        client.fetch("http://www.okx.com/api/v5/market/history-candles")
    with pytest.raises(CaptureError, match="host"):
        client.fetch("https://example.com/api")


def test_capture_preserves_raw_bytes_and_publishes_dataset_atomically(tmp_path: Path) -> None:
    output = tmp_path / "capture"
    okx = CapturedPayload(
        url=build_okx_candles_url(limit=1, bar="15m"),
        raw_bytes=okx_bytes(),
        received_time_ns=10,
    )
    binance = CapturedPayload(
        url=build_binance_klines_url(limit=1, interval="15m"),
        raw_bytes=binance_bytes(),
        received_time_ns=2_000_000_000,
    )

    artifact = publish_candle_capture(okx, binance, output_directory=output)

    assert (output / "raw" / "okx.json").read_bytes() == okx.raw_bytes
    assert (output / "raw" / "binance.json").read_bytes() == binance.raw_bytes
    assert artifact.dataset.quality.row_count == 2
    assert artifact.capture_manifest_path.exists()
    assert not list(tmp_path.glob(".capture.*"))


def test_failed_capture_is_not_published(tmp_path: Path) -> None:
    output = tmp_path / "capture"
    invalid = CapturedPayload(
        url=build_okx_candles_url(limit=1, bar="15m"),
        raw_bytes=b"not-json",
        received_time_ns=10,
    )
    binance = CapturedPayload(
        url=build_binance_klines_url(limit=1, interval="15m"),
        raw_bytes=binance_bytes(),
        received_time_ns=2_000_000_000,
    )

    with pytest.raises(CaptureError, match="JSON"):
        publish_candle_capture(invalid, binance, output_directory=output)

    assert not output.exists()
    assert not list(tmp_path.glob(".capture.*"))


def test_capture_verifier_rehashes_every_published_layer(tmp_path: Path) -> None:
    output = tmp_path / "capture"
    publish_candle_capture(
        CapturedPayload(build_okx_candles_url(limit=1, bar="15m"), okx_bytes(), 10),
        CapturedPayload(
            build_binance_klines_url(limit=1, interval="15m"),
            binance_bytes(),
            2_000_000_000,
        ),
        output_directory=output,
    )

    verification = verify_candle_capture(output)

    assert verification.valid is True
    assert verification.errors == ()


def test_capture_verifier_detects_raw_payload_tampering(tmp_path: Path) -> None:
    output = tmp_path / "capture"
    publish_candle_capture(
        CapturedPayload(build_okx_candles_url(limit=1, bar="15m"), okx_bytes(), 10),
        CapturedPayload(
            build_binance_klines_url(limit=1, interval="15m"),
            binance_bytes(),
            2_000_000_000,
        ),
        output_directory=output,
    )
    (output / "raw" / "okx.json").write_bytes(b"tampered")

    verification = verify_candle_capture(output)

    assert verification.valid is False
    assert verification.errors == ("RAW_HASH_MISMATCH:OKX",)


def test_capture_cli_rejects_excluded_drive_before_network(tmp_path: Path) -> None:
    with pytest.raises(StoragePolicyError, match="excluded drive"):
        main(
            [
                "capture-public-candles",
                "--workspace-root",
                str(tmp_path),
                "--output",
                "E:/forbidden/capture",
                "--reserve-bytes",
                "0",
                "--limit",
                "1",
            ]
        )
