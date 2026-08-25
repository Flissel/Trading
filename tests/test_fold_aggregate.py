import json
from pathlib import Path

import pytest

from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.fold_aggregate import (
    FoldAggregateError,
    aggregate_fold_evaluations,
    verify_fold_aggregate,
)


def write_fold(path: Path, fold_index: int, base_total: str, adverse_total: str) -> None:
    material: dict[str, object] = {
        "report_version": "1.0.0",
        "status": "development_only",
        "reason_codes": [],
        "capture_root_hash": "a" * 64,
        "dataset_root_hash": "b" * 64,
        "split_manifest_hash": "c" * 64,
        "fold_index": fold_index,
        "fold_count": 3,
        "candidates": [
            {
                "candidate_name": "no_trade",
                "base": {"total_net_return": "0"},
                "adverse": {"total_net_return": "0"},
                "bh_q_value": "1",
            },
            {
                "candidate_name": "momentum",
                "base": {"total_net_return": base_total},
                "adverse": {"total_net_return": adverse_total},
                "bh_q_value": "0.5",
            },
        ],
    }
    document = dict(material)
    document["report_hash"] = content_sha256(material)
    path.write_bytes(canonical_json(document))


def test_fold_aggregate_rejects_candidate_that_is_negative_in_aggregate(tmp_path: Path) -> None:
    paths = [tmp_path / f"fold-{index}.json" for index in range(3)]
    write_fold(paths[0], 0, "1", "-1")
    write_fold(paths[1], 1, "-2", "-2")
    write_fold(paths[2], 2, "0.5", "-0.5")
    output = tmp_path / "aggregate.json"

    artifact = aggregate_fold_evaluations(tuple(paths), output_path=output)
    document = json.loads(output.read_text(encoding="utf-8"))
    momentum = next(item for item in document["candidates"] if item["candidate_name"] == "momentum")

    assert artifact.fold_count == 3
    assert momentum["positive_base_fold_count"] == 2
    assert momentum["aggregate_base_total_net_return"] == "-0.5"
    assert momentum["decision_status"] == "rejected"
    assert "AGGREGATE_BASE_NET_NON_POSITIVE" in momentum["reason_codes"]
    assert "AGGREGATE_ADVERSE_NET_NON_POSITIVE" in momentum["reason_codes"]
    assert verify_fold_aggregate(output) is True


def test_fold_aggregate_fails_closed_when_expected_fold_is_missing(tmp_path: Path) -> None:
    paths = (tmp_path / "fold-0.json", tmp_path / "fold-1.json")
    write_fold(paths[0], 0, "1", "1")
    write_fold(paths[1], 1, "1", "1")

    with pytest.raises(FoldAggregateError, match="complete"):
        aggregate_fold_evaluations(paths, output_path=tmp_path / "aggregate.json")
