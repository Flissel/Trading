import json
from pathlib import Path

from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.model_dominance import build_model_dominance_report, verify_model_dominance_report


def write_aggregate(path: Path, candidate: dict[str, object]) -> None:
    material: dict[str, object] = {
        "aggregate_version": "1.0.0",
        "status": "development_only",
        "capture_root_hash": "a" * 64,
        "dataset_root_hash": "b" * 64,
        "split_manifest_hash": "c" * 64,
        "fold_count": 3,
        "fold_report_hashes": ["d" * 64, "e" * 64, "f" * 64],
        "candidates": [candidate],
    }
    document = dict(material)
    document["report_hash"] = content_sha256(material)
    path.write_bytes(canonical_json(document))


def test_dominance_report_does_not_promote_positive_base_with_negative_adverse(
    tmp_path: Path,
) -> None:
    no_trade = tmp_path / "no-trade.json"
    challenger = tmp_path / "challenger.json"
    write_aggregate(
        no_trade,
        {
            "candidate_name": "no_trade",
            "positive_base_fold_count": 0,
            "aggregate_base_total_net_return": "0",
            "aggregate_adverse_total_net_return": "0",
            "maximum_bh_q_value": "1",
            "decision_status": "contextual_baseline",
            "reason_codes": ["NON_PROMOTABLE_CONTEXT"],
        },
    )
    write_aggregate(
        challenger,
        {
            "candidate_name": "tree",
            "positive_base_fold_count": 2,
            "aggregate_base_total_net_return": "0.2",
            "aggregate_adverse_total_net_return": "-0.1",
            "maximum_bh_q_value": "0.05",
            "decision_status": "rejected",
            "reason_codes": ["AGGREGATE_ADVERSE_NET_NON_POSITIVE"],
        },
    )
    output = tmp_path / "dominance.json"

    build_model_dominance_report((no_trade, challenger), output_path=output)
    document = json.loads(output.read_text(encoding="utf-8"))
    tree = next(item for item in document["candidates"] if item["candidate_name"] == "tree")

    assert tree["dominates_no_trade"] is False
    assert document["eligible_candidate_names"] == []
    assert document["strongest_eligible_candidate"] is None
    assert document["decision_status"] == "no_eligible_candidate"
    assert verify_model_dominance_report(output) is True
