import json
from decimal import Decimal
from pathlib import Path

from tests.test_walk_forward_run import INTERVAL_NS
from trading_bot.logistic_fold_run import run_logistic_fold, verify_logistic_fold_report
from trading_bot.market_capture import (
    CapturedPayload,
    build_binance_klines_url,
    build_okx_candles_url,
    publish_candle_capture,
)
from trading_bot.registry import MetadataRegistry
from trading_bot.splits import WalkForwardConfig
from trading_bot.walk_forward_run import run_capture_walk_forward


def directional_capture(tmp_path: Path, *, count: int) -> Path:
    okx_rows: list[list[str]] = []
    binance_rows: list[list[str | int]] = []
    for index in range(count):
        timestamp_ms = 1_000 + index * 900_000
        price = 100 + (0, 2, 1)[index % 3]
        okx_rows.append(
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
        binance_rows.append(
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
    root = tmp_path / "capture"
    publish_candle_capture(
        CapturedPayload(
            build_okx_candles_url(limit=count, bar="15m"),
            json.dumps({"code": "0", "msg": "", "data": list(reversed(okx_rows))}).encode(),
            10,
        ),
        CapturedPayload(
            build_binance_klines_url(limit=count, interval="15m"),
            json.dumps(binance_rows).encode(),
            20_000_000_000_000,
        ),
        output_directory=root,
    )
    return root


def test_logistic_fold_keeps_calibration_selection_and_test_membership_separate(
    tmp_path: Path,
) -> None:
    capture_root = directional_capture(tmp_path, count=20)
    split_path = tmp_path / "walk-forward.json"
    first_decision_ns = (1_000 + 900_000) * 1_000_000 + INTERVAL_NS - 1 + 1_000_000_000
    run_capture_walk_forward(
        capture_root,
        output_path=split_path,
        config=WalkForwardConfig(
            4 * INTERVAL_NS,
            4 * INTERVAL_NS,
            4 * INTERVAL_NS,
            4 * INTERVAL_NS,
            INTERVAL_NS,
            first_decision_ns + 15 * INTERVAL_NS,
        ),
        horizon_bars=2,
    )
    output = tmp_path / "logistic-fold.json"
    registry_path = tmp_path / "metadata.sqlite3"

    artifact = run_logistic_fold(
        capture_root,
        split_manifest_path=split_path,
        output_path=output,
        registry_path=registry_path,
        fold_index=0,
        model_l2=Decimal("0.1"),
        model_iterations=50,
        model_learning_rate=Decimal("0.1"),
        calibration_l2=Decimal("0.001"),
        calibration_iterations=50,
        calibration_learning_rate=Decimal("0.1"),
        minimum_selection_trades=1,
        block_length=1,
        bootstrap_repetitions=20,
        random_seed=23,
    )
    document = json.loads(output.read_text(encoding="utf-8"))

    assert document["horizon_bars"] == 2
    assert document["train_sample_count"] == 2
    assert document["calibration_sample_count"] == 1
    assert document["selection_sample_count"] == 1
    assert document["test_sample_count"] == 2
    assert document["calibration_membership_hash"] != document["selection_membership_hash"]
    assert (
        document["calibration_end_decision_time_ns"] < document["selection_start_decision_time_ns"]
    )
    assert document["candidates"][0]["candidate_name"] == "logistic_calibrated"
    assert document["model"]["selection_minimum_trades"] == 1
    assert set(document["probability_metrics"]) == {
        "calibration_raw",
        "calibration_calibrated",
        "selection_raw",
        "selection_calibrated",
        "test_raw",
        "test_calibrated",
    }
    assert verify_logistic_fold_report(output) is True
    with MetadataRegistry(registry_path) as registry:
        registered = registry.list_experiments(artifact.family_id)
    assert len(registered) == 1
    assert registered[0].candidate_name == "logistic_calibrated"
