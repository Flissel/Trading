import json
from pathlib import Path

import pytest

from trading_bot.cli import main
from trading_bot.market_capture import (
    CapturedPayload,
    build_binance_klines_url,
    build_okx_candles_url,
    publish_candle_capture,
)
from trading_bot.research_run import (
    ResearchRunError,
    load_research_bars,
    run_capture_research,
    verify_research_report,
)


def okx_history(count: int) -> bytes:
    rows = []
    for index in reversed(range(count)):
        timestamp_ms = 1_000 + index * 900_000
        price = 100 + index
        rows.append(
            [
                str(timestamp_ms),
                str(price),
                str(price + 1),
                str(price - 1),
                str(price),
                "10",
                "1",
                "1000",
                "1",
            ]
        )
    return json.dumps({"code": "0", "msg": "", "data": rows}, separators=(",", ":")).encode()


def binance_history(count: int) -> bytes:
    rows = []
    for index in range(count):
        timestamp_ms = 1_000 + index * 900_000
        price = 99 + index
        rows.append(
            [
                timestamp_ms,
                str(price),
                str(price + 1),
                str(price - 1),
                str(price),
                "10",
                timestamp_ms + 899_999,
                "1000",
                5,
                "6",
                "600",
                "0",
            ]
        )
    return json.dumps(rows, separators=(",", ":")).encode()


def capture(tmp_path: Path, count: int = 6) -> Path:
    root = tmp_path / "capture"
    publish_candle_capture(
        CapturedPayload(build_okx_candles_url(limit=count, bar="15m"), okx_history(count), 10),
        CapturedPayload(
            build_binance_klines_url(limit=count, interval="15m"),
            binance_history(count),
            20_000_000_000_000,
        ),
        output_directory=root,
    )
    return root


def test_research_run_loads_verified_parquet_and_writes_linked_report(tmp_path: Path) -> None:
    capture_root = capture(tmp_path)
    output = tmp_path / "research.json"

    artifact = run_capture_research(
        capture_root,
        output_path=output,
        oos_fraction="0.25",
        minimum_train_samples=2,
        random_seed=7,
    )
    document = json.loads(output.read_text(encoding="utf-8"))

    assert artifact.report.total_sample_count == 4
    assert artifact.report.train_sample_count == 3
    assert artifact.report.oos_sample_count == 1
    assert artifact.report.cross_venue_coverage == 1
    assert document["capture_root_hash"] == artifact.capture_root_hash
    assert document["dataset_root_hash"] == artifact.dataset_root_hash
    assert document["status"] == "development_only"
    assert document["report_hash"] == artifact.report_hash
    assert verify_research_report(output) is True


def test_research_run_refuses_tampered_capture(tmp_path: Path) -> None:
    capture_root = capture(tmp_path)
    (capture_root / "raw" / "okx.json").write_bytes(b"tampered")

    with pytest.raises(ResearchRunError, match="verification failed"):
        run_capture_research(
            capture_root,
            output_path=tmp_path / "research.json",
            oos_fraction="0.25",
            minimum_train_samples=2,
            random_seed=7,
        )


def test_research_report_verifier_detects_changed_result(tmp_path: Path) -> None:
    output = tmp_path / "research.json"
    run_capture_research(
        capture(tmp_path),
        output_path=output,
        oos_fraction="0.25",
        minimum_train_samples=2,
        random_seed=7,
    )
    document = json.loads(output.read_text(encoding="utf-8"))
    document["oos_sample_count"] = 999
    output.write_text(json.dumps(document), encoding="utf-8")

    assert verify_research_report(output) is False


def test_research_cli_runs_only_on_existing_verified_capture(tmp_path: Path) -> None:
    capture_root = capture(tmp_path)
    output = tmp_path / "research.json"

    exit_code = main(
        [
            "research-capture",
            "--workspace-root",
            str(tmp_path),
            "--capture",
            str(capture_root),
            "--output",
            str(output),
            "--minimum-train-samples",
            "2",
            "--oos-fraction",
            "0.25",
        ]
    )

    assert exit_code == 0
    assert output.exists()


def test_research_bar_loader_never_reads_at_or_after_fold_boundary(tmp_path: Path) -> None:
    capture_root = capture(tmp_path, count=6)
    boundary_ns = 3_601_999_999_999

    bars = load_research_bars(capture_root / "dataset", available_before_ns=boundary_ns)

    assert len(bars) == 7
    assert all(bar.available_time_ns < boundary_ns for bar in bars)
