"""Evaluate one walk-forward fold of the panel family."""

import json
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.panel_accounting import EpisodeResult, evaluate_episode
from trading_bot.panel_capture import verify_panel_capture
from trading_bot.panel_config import PanelCostTable, PanelFamilySpec, load_panel_family_spec
from trading_bot.panel_reader import FundingEvent, load_funding_events, load_panel_bars
from trading_bot.panel_samples import verify_panel_manifest
from trading_bot.panel_signals import build_weight_vectors
from trading_bot.panel_universe import build_contract_histories, select_universe
from trading_bot.registry import ExperimentRecord, MetadataRegistry

_PANEL_MODULES = (
    "panel_config.py",
    "panel_dataset.py",
    "panel_capture.py",
    "panel_reader.py",
    "panel_universe.py",
    "panel_signals.py",
    "panel_accounting.py",
    "panel_samples.py",
    "panel_fold_run.py",
    "evaluation.py",
    "strategy.py",
)


class PanelFoldError(RuntimeError):
    """Raised when a panel fold cannot be evaluated or published."""


@dataclass(frozen=True, slots=True)
class PanelFoldArtifact:
    output_path: Path
    report_hash: str
    fold_index: int
    episode_count: int
    skipped_sample_count: int


def run_panel_fold(
    capture_root: Path,
    *,
    manifest_path: Path,
    family_spec_path: Path,
    output_path: Path,
    registry_path: Path,
    fold_index: int,
) -> PanelFoldArtifact:
    valid, errors = verify_panel_capture(capture_root)
    if not valid:
        raise PanelFoldError("panel capture verification failed: " + ",".join(errors))
    if not verify_panel_manifest(manifest_path):
        raise PanelFoldError("panel manifest verification failed")
    if output_path.exists():
        raise PanelFoldError("panel fold report already exists and is immutable")

    spec, family_spec_hash = load_panel_family_spec(family_spec_path)
    manifest = _load_object(manifest_path)
    if manifest.get("family_spec_hash") != family_spec_hash:
        raise PanelFoldError("family declaration does not match the manifest")
    capture_manifest = _load_object(capture_root / "capture-manifest.json")
    dataset_manifest = _load_object(capture_root / "dataset" / "dataset-manifest.json")
    if manifest.get("capture_root_hash") != capture_manifest.get("capture_root_hash"):
        raise PanelFoldError("manifest is not linked to this capture")
    if manifest.get("dataset_root_hash") != dataset_manifest.get("root_hash"):
        raise PanelFoldError("manifest is not linked to this dataset")

    folds = manifest.get("folds")
    if not isinstance(folds, list):
        raise PanelFoldError("manifest folds are malformed")
    fold = next((item for item in folds if item.get("fold_index") == fold_index), None)
    if fold is None:
        raise PanelFoldError(f"fold {fold_index} is not in the manifest")

    test_end_ns = int(fold["test_end_ns"])
    bars = load_panel_bars(capture_root / "dataset", available_before_ns=test_end_ns + 1)
    histories = build_contract_histories(bars)
    funding_by_contract: dict[str, tuple[FundingEvent, ...]] = {}
    for event in load_funding_events(capture_root / "dataset"):
        funding_by_contract.setdefault(event.contract_id, ())
        funding_by_contract[event.contract_id] += (event,)

    test_ids = [str(value) for value in fold["test_ids"]]
    decisions = sorted(int(value.split(":")[1]) for value in test_ids)
    names = [member.name for member in spec.members] + [item.name for item in spec.controls]
    scenarios: tuple[PanelCostTable, ...] = (spec.costs.base, spec.costs.adverse)
    episodes: dict[tuple[str, str], list[EpisodeResult]] = {
        (name, scenario.name): [] for name in names for scenario in scenarios
    }
    carried: dict[tuple[str, str], tuple[tuple[str, Decimal], ...]] = {
        key: () for key in episodes
    }
    skipped: list[str] = []

    for decision_close_ns in decisions:
        sample_id = f"BINANCE_UM:{decision_close_ns}:w1"
        snapshot = select_universe(
            histories, decision_close_ns=decision_close_ns, rules=spec.universe
        )
        if not snapshot.contracts:
            skipped.append(sample_id)
            for key in carried:
                carried[key] = ()
            continue
        tiers = {item.contract_id: item.tier for item in snapshot.contracts}
        vectors = build_weight_vectors(histories, snapshot, spec=spec)
        for name in names:
            for scenario in scenarios:
                key = (name, scenario.name)
                result = evaluate_episode(
                    sample_id=sample_id,
                    member=name,
                    decision_close_ns=decision_close_ns,
                    holding_days=spec.holding_days,
                    weights=vectors[name].weights,
                    previous_weights=carried[key],
                    histories=histories,
                    tiers=tiers,
                    funding_by_contract=funding_by_contract,
                    cost_table=scenario,
                )
                episodes[key].append(result)
                carried[key] = result.drifted_weights

    episode_count = len(episodes[(names[0], "base")])
    reason_codes: list[str] = []
    if skipped:
        reason_codes.append("SKIPPED_WEEK_EXIT_COST_UNCHARGED")
    if episode_count == 0:
        reason_codes.append("NO_EPISODES_IN_FOLD")

    member_names = {member.name for member in spec.members}
    candidates = [
        {
            "candidate_name": name,
            "role": "member" if name in member_names else "control",
            "episode_count": len(episodes[(name, "base")]),
            "base": _scenario_record(episodes[(name, "base")]),
            "adverse": _scenario_record(episodes[(name, "adverse")]),
        }
        for name in names
    ]

    material: dict[str, object] = {
        "report_version": "1.0.0",
        "status": "development_only",
        "reason_codes": reason_codes,
        "family_name": spec.family_name,
        "family_spec_hash": family_spec_hash,
        "capture_root_hash": str(manifest["capture_root_hash"]),
        "dataset_root_hash": str(manifest["dataset_root_hash"]),
        "split_manifest_hash": str(manifest["split_manifest_hash"]),
        "manifest_hash": str(manifest["manifest_hash"]),
        "fold_index": fold_index,
        "fold_count": len(folds),
        "train_sample_count": len(fold["train_ids"]),
        "validation_sample_count": len(fold["validation_ids"]),
        "test_sample_count": len(test_ids),
        "train_membership_hash": content_sha256(list(fold["train_ids"])),
        "validation_membership_hash": content_sha256(list(fold["validation_ids"])),
        "test_membership_hash": content_sha256(test_ids),
        "random_seed": spec.statistics.random_seed,
        "block_length": spec.statistics.block_length,
        "bootstrap_repetitions": spec.statistics.bootstrap_repetitions,
        "skipped_sample_ids": skipped,
        "code_hash": _code_hash(),
        "candidates": candidates,
    }
    report_hash = content_sha256(material)
    document = dict(material)
    document["report_hash"] = report_hash
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_bytes(canonical_json(document))
    temporary.replace(output_path)

    _register(spec, manifest, report_hash, registry_path)
    return PanelFoldArtifact(
        output_path=output_path,
        report_hash=report_hash,
        fold_index=fold_index,
        episode_count=episode_count,
        skipped_sample_count=len(skipped),
    )


def verify_panel_fold_report(path: Path) -> bool:
    try:
        document = _load_object(path)
        recorded = str(document["report_hash"])
        material = {key: value for key, value in document.items() if key != "report_hash"}
        return content_sha256(material) == recorded
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False


def _scenario_record(results: list[EpisodeResult]) -> dict[str, object]:
    return {
        "total_net_return": sum((item.net_return for item in results), Decimal(0)),
        "episodes": [
            {
                "sample_id": item.sample_id,
                "net_return": item.net_return,
                "gross_return": item.gross_return,
                "turnover": item.turnover,
                "trading_cost": item.trading_cost,
                "funding_cost": item.funding_cost,
                "forced_close_cost": item.forced_close_cost,
                "gross_exposure": item.gross_exposure,
                "net_exposure": item.net_exposure,
                "forced_close_count": item.forced_close_count,
                "contract_net_contributions": [
                    [contract_id, value] for contract_id, value in item.contract_net_contributions
                ],
            }
            for item in results
        ],
    }


def _register(
    spec: PanelFamilySpec,
    manifest: dict[str, object],
    report_hash: str,
    registry_path: Path,
) -> None:
    split_hash = str(manifest["split_manifest_hash"])
    family_id = uuid5(NAMESPACE_URL, f"{split_hash}:{spec.family_name}")
    created_at_ns = time.time_ns()
    with MetadataRegistry(registry_path) as registry:
        for member in spec.members:
            registry.register_experiment(
                ExperimentRecord(
                    experiment_id=uuid5(
                        family_id,
                        f"{member.name}:{spec.statistics.random_seed}:{report_hash}",
                    ),
                    family_id=family_id,
                    candidate_name=member.name,
                    hypothesis=spec.hypothesis,
                    split_manifest_hash=split_hash,
                    code_hash=_code_hash(),
                    random_seed=spec.statistics.random_seed,
                    outcome="completed",
                    result_hash=report_hash,
                    failure_reason=None,
                    created_at_ns=created_at_ns,
                )
            )


def _code_hash() -> str:
    root = Path(__file__).parent
    material = "".join((root / name).read_text(encoding="utf-8") for name in _PANEL_MODULES)
    return content_sha256(material)


def _load_object(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise PanelFoldError(f"{path.name} must contain a JSON object")
    return document
