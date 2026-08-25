import json
from pathlib import Path

from tests.test_walk_forward_run import INTERVAL_NS, capture
from trading_bot.cli import main
from trading_bot.fold_evaluation import (
    fold_reason_codes,
    run_fold_evaluation,
    verify_fold_evaluation,
)
from trading_bot.registry import MetadataRegistry
from trading_bot.splits import WalkForwardConfig
from trading_bot.walk_forward_run import run_capture_walk_forward


def test_fold_evaluation_is_registered_reproducible_and_holdout_locked(tmp_path: Path) -> None:
    capture_root = capture(tmp_path, count=14)
    split_path = tmp_path / "walk-forward.json"
    first_decision_ns = (1_000 + 900_000) * 1_000_000 + INTERVAL_NS - 1 + 1_000_000_000
    run_capture_walk_forward(
        capture_root,
        output_path=split_path,
        config=WalkForwardConfig(
            4 * INTERVAL_NS,
            2 * INTERVAL_NS,
            2 * INTERVAL_NS,
            2 * INTERVAL_NS,
            INTERVAL_NS,
            first_decision_ns + 11 * INTERVAL_NS,
        ),
    )
    output = tmp_path / "fold-evaluation.json"
    registry_path = tmp_path / "metadata.sqlite3"

    artifact = run_fold_evaluation(
        capture_root,
        split_manifest_path=split_path,
        output_path=output,
        registry_path=registry_path,
        fold_index=0,
        random_seed=17,
        block_length=1,
        bootstrap_repetitions=100,
    )
    document = json.loads(output.read_text(encoding="utf-8"))

    assert artifact.candidate_count == 4
    assert document["evaluated_sample_ids"] == ["OKX:8101000000000"]
    assert document["status"] == "development_only"
    assert "FOLD_FLOOR_NOT_MET" in document["reason_codes"]
    assert "EPISODE_FLOOR_NOT_MET" in document["reason_codes"]
    candidates = {item["candidate_name"]: item for item in document["candidates"]}
    assert candidates["no_trade"]["decision_status"] == "contextual_baseline"
    assert candidates["momentum"]["decision_status"] == "insufficient_evidence"
    assert "FOLD_FLOOR_NOT_MET" in candidates["momentum"]["decision_reason_codes"]
    assert verify_fold_evaluation(output) is True
    with MetadataRegistry(registry_path) as registry:
        registered = registry.list_experiments(artifact.family_id)
    assert tuple(item.candidate_name for item in registered) == (
        "no_trade",
        "momentum",
        "mean_reversion",
        "random",
    )
    assert all(item.result_hash == artifact.report_hash for item in registered)

    cli_output = tmp_path / "cli-fold-evaluation.json"
    assert (
        main(
            [
                "evaluate-fold",
                "--workspace-root",
                str(tmp_path),
                "--capture",
                str(capture_root),
                "--split-manifest",
                str(split_path),
                "--output",
                str(cli_output),
                "--registry",
                str(tmp_path / "cli-metadata.sqlite3"),
                "--fold-index",
                "0",
                "--random-seed",
                "17",
                "--block-length",
                "1",
                "--bootstrap-repetitions",
                "20",
            ]
        )
        == 0
    )
    assert verify_fold_evaluation(cli_output) is True


def test_fold_evaluation_verifier_detects_changed_bootstrap_result(tmp_path: Path) -> None:
    capture_root = capture(tmp_path, count=14)
    split_path = tmp_path / "walk-forward.json"
    first_decision_ns = (1_000 + 900_000) * 1_000_000 + INTERVAL_NS - 1 + 1_000_000_000
    run_capture_walk_forward(
        capture_root,
        output_path=split_path,
        config=WalkForwardConfig(
            4 * INTERVAL_NS,
            2 * INTERVAL_NS,
            2 * INTERVAL_NS,
            2 * INTERVAL_NS,
            INTERVAL_NS,
            first_decision_ns + 11 * INTERVAL_NS,
        ),
    )
    output = tmp_path / "fold-evaluation.json"
    run_fold_evaluation(
        capture_root,
        split_manifest_path=split_path,
        output_path=output,
        registry_path=tmp_path / "metadata.sqlite3",
        fold_index=0,
        random_seed=17,
        block_length=1,
        bootstrap_repetitions=20,
    )
    document = json.loads(output.read_text(encoding="utf-8"))
    document["candidates"][1]["base_bootstrap"]["lower"] = "999"
    output.write_text(json.dumps(document), encoding="utf-8")

    assert verify_fold_evaluation(output) is False


def test_fold_evaluation_uses_the_horizon_declared_by_the_split_manifest(
    tmp_path: Path,
) -> None:
    capture_root = capture(tmp_path, count=20)
    split_path = tmp_path / "walk-forward-h2.json"
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
    split_document = json.loads(split_path.read_text(encoding="utf-8"))
    output = tmp_path / "fold-evaluation-h2.json"

    run_fold_evaluation(
        capture_root,
        split_manifest_path=split_path,
        output_path=output,
        registry_path=tmp_path / "metadata-h2.sqlite3",
        fold_index=0,
        random_seed=17,
        block_length=1,
        bootstrap_repetitions=20,
    )
    document = json.loads(output.read_text(encoding="utf-8"))

    assert document["horizon_bars"] == 2
    assert document["evaluated_sample_ids"] == split_document["folds"][0]["test_ids"]
    assert all(sample_id.endswith(":h2") for sample_id in document["evaluated_sample_ids"])
    assert verify_fold_evaluation(output) is True


def test_no_trade_does_not_create_a_false_episode_floor_failure() -> None:
    assert fold_reason_codes(
        fold_count=1,
        trade_counts={
            "no_trade": 0,
            "momentum": 250,
            "mean_reversion": 250,
            "random": 210,
        },
    ) == ("FOLD_FLOOR_NOT_MET",)
