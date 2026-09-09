import json
from decimal import Decimal
from pathlib import Path

import pytest

from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.panel_config import load_panel_family_spec
from trading_bot.panel_decision import PanelDecisionError, build_panel_decision

SPEC_PATH = Path("configs/xs-momentum-panel-v1.json")
SPEC, SPEC_HASH = load_panel_family_spec(SPEC_PATH)
MEMBERS = tuple(item.name for item in SPEC.members)
CONTROLS = tuple(item.name for item in SPEC.controls)
EPISODES_PER_FOLD = 40


def episodes(sample_prefix: str, value: str, count: int) -> list[dict[str, object]]:
    return [
        {
            "sample_id": f"{sample_prefix}:{index}",
            "net_return": value,
            "gross_return": value,
            "turnover": "1",
            "trading_cost": "0",
            "funding_cost": "0",
            "forced_close_cost": "0",
            "gross_exposure": "1",
            "net_exposure": "0",
            "forced_close_count": 0,
            "contract_net_contributions": [
                ["A:0", str(Decimal(value) / 2)],
                ["B:0", str(Decimal(value) / 2)],
            ],
        }
        for index in range(count)
    ]


def write_fold(path: Path, fold_index: int, returns: dict[str, tuple[str, str]]) -> None:
    candidates = []
    for name in MEMBERS + CONTROLS:
        base_value, adverse_value = returns.get(name, ("0", "0"))
        candidates.append(
            {
                "candidate_name": name,
                "role": "member" if name in MEMBERS else "control",
                "episode_count": EPISODES_PER_FOLD,
                "base": {
                    "total_net_return": str(
                        Decimal(base_value) * Decimal(EPISODES_PER_FOLD)
                    ),
                    "episodes": episodes(f"f{fold_index}", base_value, EPISODES_PER_FOLD),
                },
                "adverse": {
                    "total_net_return": str(
                        Decimal(adverse_value) * Decimal(EPISODES_PER_FOLD)
                    ),
                    "episodes": episodes(f"f{fold_index}", adverse_value, EPISODES_PER_FOLD),
                },
            }
        )
    material = {
        "report_version": "1.0.0",
        "status": "development_only",
        "reason_codes": [],
        "family_name": SPEC.family_name,
        "family_spec_hash": SPEC_HASH,
        "capture_root_hash": "c" * 64,
        "dataset_root_hash": "d" * 64,
        "split_manifest_hash": "e" * 64,
        "manifest_hash": "f" * 64,
        "fold_index": fold_index,
        "fold_count": 6,
        "train_sample_count": 10,
        "validation_sample_count": 10,
        "test_sample_count": EPISODES_PER_FOLD,
        "train_membership_hash": "0" * 64,
        "validation_membership_hash": "1" * 64,
        "test_membership_hash": "2" * 64,
        "random_seed": 17,
        "block_length": 4,
        "bootstrap_repetitions": 2000,
        "skipped_sample_ids": [],
        "code_hash": "3" * 64,
        "candidates": candidates,
    }
    document = dict(material)
    document["report_hash"] = content_sha256(material)
    path.write_bytes(canonical_json(document))


def build(tmp_path: Path, returns: dict[str, tuple[str, str]]) -> Path:
    paths = []
    for fold_index in range(6):
        path = tmp_path / f"fold{fold_index}.json"
        write_fold(path, fold_index, returns)
        paths.append(path)
    artifact = build_panel_decision(
        tuple(paths),
        family_spec_path=SPEC_PATH,
        output_path=tmp_path / "decision.json",
        registry_path=tmp_path / "registry.sqlite3",
    )
    return artifact.output_path


def test_a_flat_family_is_rejected_without_eligible_members(tmp_path: Path) -> None:
    output_path = build(tmp_path, {})
    document = json.loads(output_path.read_text(encoding="utf-8"))
    assert document["decision_status"] == "no_eligible_member"
    momentum = next(
        item for item in document["members"] if item["candidate_name"] == "xs_mom_4w"
    )
    assert "AGGREGATE_BASE_NET_NON_POSITIVE" in momentum["reason_codes"]


def test_negative_adverse_is_rejected(tmp_path: Path) -> None:
    output_path = build(tmp_path, {"xs_mom_4w": ("0.01", "-0.001")})
    document = json.loads(output_path.read_text(encoding="utf-8"))
    momentum = next(
        item for item in document["members"] if item["candidate_name"] == "xs_mom_4w"
    )
    assert momentum["decision_status"] == "rejected"
    assert "AGGREGATE_ADVERSE_NET_NON_POSITIVE" in momentum["reason_codes"]


def test_pooled_counts_and_hashes_are_recorded(tmp_path: Path) -> None:
    output_path = build(tmp_path, {})
    document = json.loads(output_path.read_text(encoding="utf-8"))
    assert document["pooled_episode_count"] == 6 * EPISODES_PER_FOLD
    assert document["fold_count"] == 6
    assert len(document["source_report_hashes"]) == 6
    assert document["family_spec_hash"] == SPEC_HASH


def test_duplicate_fold_indices_are_rejected(tmp_path: Path) -> None:
    write_fold(tmp_path / "a.json", 0, {})
    write_fold(tmp_path / "b.json", 0, {})
    with pytest.raises(PanelDecisionError):
        build_panel_decision(
            (tmp_path / "a.json", tmp_path / "b.json"),
            family_spec_path=SPEC_PATH,
            output_path=tmp_path / "decision.json",
            registry_path=tmp_path / "registry.sqlite3",
        )


def test_missing_fold_index_is_rejected(tmp_path: Path) -> None:
    paths = []
    for fold_index in range(6):
        path = tmp_path / f"fold{fold_index}.json"
        write_fold(path, fold_index, {})
        paths.append(path)
    incomplete = tuple(path for index, path in enumerate(paths) if index != 3)
    with pytest.raises(PanelDecisionError) as excinfo:
        build_panel_decision(
            incomplete,
            family_spec_path=SPEC_PATH,
            output_path=tmp_path / "decision.json",
            registry_path=tmp_path / "registry.sqlite3",
        )
    assert "missing fold index(es) 3" in str(excinfo.value)


def test_tampered_report_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "fold0.json"
    write_fold(path, 0, {})
    document = json.loads(path.read_text(encoding="utf-8"))
    document["fold_index"] = 1
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(PanelDecisionError):
        build_panel_decision(
            (path,),
            family_spec_path=SPEC_PATH,
            output_path=tmp_path / "decision.json",
            registry_path=tmp_path / "registry.sqlite3",
        )
