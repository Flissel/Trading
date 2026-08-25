import json
from decimal import Decimal
from pathlib import Path

from tests.test_walk_forward_run import INTERVAL_NS, capture
from trading_bot.linear_fold_run import run_linear_fold, verify_linear_fold_report
from trading_bot.registry import MetadataRegistry
from trading_bot.splits import WalkForwardConfig
from trading_bot.walk_forward_run import run_capture_walk_forward


def test_linear_fold_uses_separate_train_validation_and_test_membership(tmp_path: Path) -> None:
    capture_root = capture(tmp_path, count=20)
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
    output = tmp_path / "linear-fold.json"
    registry_path = tmp_path / "metadata.sqlite3"

    artifact = run_linear_fold(
        capture_root,
        split_manifest_path=split_path,
        output_path=output,
        registry_path=registry_path,
        fold_index=0,
        alpha=Decimal("0.1"),
        minimum_validation_trades=1,
        block_length=1,
        bootstrap_repetitions=20,
        random_seed=17,
    )
    document = json.loads(output.read_text(encoding="utf-8"))

    assert document["horizon_bars"] == 2
    assert document["train_sample_count"] == 2
    assert document["validation_sample_count"] == 2
    assert document["test_sample_count"] == 2
    assert document["candidates"][0]["candidate_name"] == "ridge_linear"
    assert len(document["model"]["feature_means"]) == 4
    assert document["model"]["validation_minimum_trades"] == 1
    assert document["model"]["validation_selected_trade_count"] == 2
    assert verify_linear_fold_report(output) is True
    with MetadataRegistry(registry_path) as registry:
        registered = registry.list_experiments(artifact.family_id)
    assert len(registered) == 1
    assert registered[0].candidate_name == "ridge_linear"
