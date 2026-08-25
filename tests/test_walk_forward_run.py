import json
from pathlib import Path

from trading_bot.cli import main
from trading_bot.market_capture import (
    CapturedPayload,
    build_binance_klines_url,
    build_okx_candles_url,
    publish_candle_capture,
)
from trading_bot.splits import WalkForwardConfig
from trading_bot.walk_forward_run import (
    derive_walk_forward_config,
    run_capture_walk_forward,
    verify_walk_forward_manifest,
)

INTERVAL_NS = 900_000_000_000


def capture(tmp_path: Path, *, count: int) -> Path:
    okx_rows: list[list[str]] = []
    binance_rows: list[list[str | int]] = []
    for index in range(count):
        timestamp_ms = 1_000 + index * 900_000
        price = 100 + index
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


def test_walk_forward_run_writes_linked_immutable_fold_membership(tmp_path: Path) -> None:
    capture_root = capture(tmp_path, count=14)
    output = tmp_path / "walk-forward.json"
    first_decision_ns = (1_000 + 900_000) * 1_000_000 + INTERVAL_NS - 1 + 1_000_000_000
    config = WalkForwardConfig(
        train_duration_ns=4 * INTERVAL_NS,
        validation_duration_ns=2 * INTERVAL_NS,
        test_duration_ns=2 * INTERVAL_NS,
        step_ns=2 * INTERVAL_NS,
        embargo_ns=INTERVAL_NS,
        final_holdout_start_ns=first_decision_ns + 11 * INTERVAL_NS,
    )

    artifact = run_capture_walk_forward(capture_root, output_path=output, config=config)
    document = json.loads(output.read_text(encoding="utf-8"))

    assert artifact.fold_count == 1
    assert document["capture_root_hash"] == artifact.capture_root_hash
    assert document["dataset_root_hash"] == artifact.dataset_root_hash
    assert document["folds"][0]["train_ids"] == [
        "OKX:901000000000",
        "OKX:1801000000000",
        "OKX:2701000000000",
    ]
    assert document["folds"][0]["validation_ids"] == ["OKX:5401000000000"]
    assert document["folds"][0]["test_ids"] == ["OKX:8101000000000"]
    assert document["final_holdout_ids"] == ["OKX:10801000000000"]
    assert verify_walk_forward_manifest(output) is True


def test_walk_forward_verifier_detects_changed_membership(tmp_path: Path) -> None:
    output = tmp_path / "walk-forward.json"
    first_decision_ns = (1_000 + 900_000) * 1_000_000 + INTERVAL_NS - 1 + 1_000_000_000
    config = WalkForwardConfig(
        4 * INTERVAL_NS,
        2 * INTERVAL_NS,
        2 * INTERVAL_NS,
        2 * INTERVAL_NS,
        INTERVAL_NS,
        first_decision_ns + 11 * INTERVAL_NS,
    )
    run_capture_walk_forward(capture(tmp_path, count=14), output_path=output, config=config)
    document = json.loads(output.read_text(encoding="utf-8"))
    document["folds"][0]["test_ids"] = []
    output.write_text(json.dumps(document), encoding="utf-8")

    assert verify_walk_forward_manifest(output) is False


def test_walk_forward_manifest_purges_labels_for_declared_horizon(tmp_path: Path) -> None:
    capture_root = capture(tmp_path, count=20)
    output = tmp_path / "walk-forward-h2.json"
    first_decision_ns = (1_000 + 900_000) * 1_000_000 + INTERVAL_NS - 1 + 1_000_000_000

    run_capture_walk_forward(
        capture_root,
        output_path=output,
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
    document = json.loads(output.read_text(encoding="utf-8"))
    fold = document["folds"][0]

    assert document["horizon_bars"] == 2
    assert len(fold["train_ids"]) == 2
    assert len(fold["validation_ids"]) == 2
    assert len(fold["test_ids"]) == 2
    assert all(
        sample_id.endswith(":h2")
        for key in ("train_ids", "validation_ids", "test_ids")
        for sample_id in fold[key]
    )
    assert verify_walk_forward_manifest(output) is True


def test_protocol_config_places_final_holdout_at_end_of_capture(tmp_path: Path) -> None:
    capture_root = capture(tmp_path, count=14)

    config = derive_walk_forward_config(
        capture_root,
        train_duration_ns=4 * INTERVAL_NS,
        validation_duration_ns=2 * INTERVAL_NS,
        test_duration_ns=2 * INTERVAL_NS,
        step_ns=2 * INTERVAL_NS,
        embargo_ns=INTERVAL_NS,
        holdout_duration_ns=INTERVAL_NS,
    )

    last_sample_decision = 11_701_999_999_999
    assert config.final_holdout_start_ns == last_sample_decision


def test_walk_forward_cli_derives_holdout_and_publishes_manifest(tmp_path: Path) -> None:
    capture_root = capture(tmp_path, count=14)
    output = tmp_path / "cli-walk-forward.json"

    result = main(
        [
            "walk-forward-manifest",
            "--workspace-root",
            str(tmp_path),
            "--capture",
            str(capture_root),
            "--output",
            str(output),
            "--train-duration-ns",
            str(4 * INTERVAL_NS),
            "--validation-duration-ns",
            str(2 * INTERVAL_NS),
            "--test-duration-ns",
            str(2 * INTERVAL_NS),
            "--step-ns",
            str(2 * INTERVAL_NS),
            "--embargo-ns",
            str(INTERVAL_NS),
            "--holdout-duration-ns",
            str(INTERVAL_NS),
            "--horizon-bars",
            "2",
        ]
    )

    assert result == 0
    assert json.loads(output.read_text(encoding="utf-8"))["horizon_bars"] == 2
    assert verify_walk_forward_manifest(output) is True
