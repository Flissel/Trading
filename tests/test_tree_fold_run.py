import json
from decimal import Decimal
from pathlib import Path

from tests.test_logistic_fold_run import directional_capture
from tests.test_walk_forward_run import INTERVAL_NS
from trading_bot.registry import MetadataRegistry
from trading_bot.splits import WalkForwardConfig
from trading_bot.tree_fold_run import run_tree_fold, verify_tree_fold_report
from trading_bot.walk_forward_run import run_capture_walk_forward


def test_tree_fold_uses_separate_memberships_and_records_diagnostics(tmp_path: Path) -> None:
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
    output = tmp_path / "tree-fold.json"
    registry_path = tmp_path / "metadata.sqlite3"

    artifact = run_tree_fold(
        capture_root,
        split_manifest_path=split_path,
        output_path=output,
        registry_path=registry_path,
        fold_index=0,
        estimator_count=2,
        learning_rate=Decimal("0.3"),
        minimum_leaf_samples=1,
        maximum_split_candidates=3,
        minimum_validation_trades=1,
        block_length=1,
        bootstrap_repetitions=20,
        random_seed=29,
    )
    document = json.loads(output.read_text(encoding="utf-8"))

    assert document["horizon_bars"] == 2
    assert document["train_sample_count"] == 2
    assert document["validation_sample_count"] == 2
    assert document["test_sample_count"] == 2
    assert document["candidates"][0]["candidate_name"] == "boosted_stumps"
    assert document["model"]["fitted_estimator_count"] >= 1
    assert len(document["model"]["feature_importance"]) == 4
    assert document["model"]["validation_minimum_trades"] == 1
    assert document["model"]["validation_selected_trade_count"] == 2
    assert verify_tree_fold_report(output) is True
    with MetadataRegistry(registry_path) as registry:
        registered = registry.list_experiments(artifact.family_id)
    assert len(registered) == 1
    assert registered[0].candidate_name == "boosted_stumps"
