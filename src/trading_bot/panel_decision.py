"""Pooled gates and the decision report for the panel family."""

import json
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.evaluation import BootstrapMeanTest, benjamini_hochberg, block_bootstrap_mean_test
from trading_bot.panel_config import PanelFamilySpec, load_panel_family_spec
from trading_bot.panel_fold_run import verify_panel_fold_report
from trading_bot.panel_statistics import (
    PanelStatisticsError,
    annualised_sharpe,
    deflated_sharpe_ratio,
    sharpe_ratio,
)
from trading_bot.registry import ArtifactRecord, MetadataRegistry


class PanelDecisionError(RuntimeError):
    """Raised when the panel decision cannot be derived or published."""


@dataclass(frozen=True, slots=True)
class PanelDecisionArtifact:
    output_path: Path
    report_hash: str
    decision_status: str
    eligible_member_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Pooled:
    base_returns: tuple[Decimal, ...]
    adverse_returns: tuple[Decimal, ...]
    base_total: Decimal
    adverse_total: Decimal
    fold_base_totals: tuple[Decimal, ...]
    contract_totals: dict[str, Decimal]
    episode_count: int


def build_panel_decision(
    fold_report_paths: tuple[Path, ...],
    *,
    family_spec_path: Path,
    output_path: Path,
    registry_path: Path,
) -> PanelDecisionArtifact:
    if not fold_report_paths:
        raise PanelDecisionError("at least one fold report is required")
    if output_path.exists():
        raise PanelDecisionError("panel decision already exists and is immutable")
    spec, family_spec_hash = load_panel_family_spec(family_spec_path)

    documents = []
    for path in fold_report_paths:
        if not verify_panel_fold_report(path):
            raise PanelDecisionError(f"fold report failed verification: {path.name}")
        documents.append(_load_object(path))
    indices = [_int_field(document, "fold_index") for document in documents]
    if len(set(indices)) != len(indices):
        raise PanelDecisionError("fold reports must have distinct fold indices")
    declared_fold_counts = {_int_field(document, "fold_count") for document in documents}
    if len(declared_fold_counts) != 1:
        raise PanelDecisionError("fold reports do not agree on the declared fold count")
    declared_fold_count = next(iter(declared_fold_counts))
    expected_indices = set(range(declared_fold_count))
    actual_indices = set(indices)
    missing_indices = sorted(expected_indices - actual_indices)
    unexpected_indices = sorted(actual_indices - expected_indices)
    if missing_indices or unexpected_indices:
        details = []
        if missing_indices:
            details.append(
                "missing fold index(es) " + ", ".join(str(index) for index in missing_indices)
            )
        if unexpected_indices:
            details.append(
                "unexpected fold index(es) "
                + ", ".join(str(index) for index in unexpected_indices)
            )
        raise PanelDecisionError(
            "fold report family is not complete: " + "; ".join(details)
        )
    linkage = {
        (
            str(document["family_spec_hash"]),
            str(document["split_manifest_hash"]),
            str(document["dataset_root_hash"]),
            str(document["capture_root_hash"]),
        )
        for document in documents
    }
    if len(linkage) != 1:
        raise PanelDecisionError("fold reports do not share one family and manifest")
    if next(iter(linkage))[0] != family_spec_hash:
        raise PanelDecisionError("fold reports were produced under a different declaration")
    documents.sort(key=lambda item: _int_field(item, "fold_index"))

    member_names = tuple(item.name for item in spec.members)
    control_names = tuple(item.name for item in spec.controls)
    pooled: dict[str, _Pooled] = {
        name: _pool(documents, name) for name in member_names + control_names
    }
    fold_count = len(documents)

    tests: dict[str, BootstrapMeanTest] = {}
    for name in member_names:
        series = pooled[name].base_returns
        if len(series) < spec.statistics.block_length:
            continue
        try:
            tests[name] = block_bootstrap_mean_test(
                series,
                block_length=spec.statistics.block_length,
                repetitions=spec.statistics.bootstrap_repetitions,
                seed=spec.statistics.random_seed,
                confidence=spec.statistics.confidence,
            )
        except ValueError as error:
            raise PanelDecisionError(f"bootstrap failed for {name}: {error}") from error
    q_values = (
        benjamini_hochberg({name: test.one_sided_p_value for name, test in tests.items()})
        if tests
        else {}
    )

    trial_sharpes: list[Decimal] = []
    for name in member_names:
        try:
            trial_sharpes.append(sharpe_ratio(pooled[name].base_returns))
        except PanelStatisticsError:
            trial_sharpes.append(Decimal(0))

    strongest_base = max(
        (pooled[name].base_total for name in ("no_trade", "random_ranks")), default=Decimal(0)
    )
    strongest_adverse = max(
        (pooled[name].adverse_total for name in ("no_trade", "random_ranks")),
        default=Decimal(0),
    )

    members: list[dict[str, object]] = []
    eligible: list[str] = []
    for name in member_names:
        record, status = _member_record(
            name,
            pooled[name],
            spec=spec,
            fold_count=fold_count,
            test=tests.get(name),
            q_value=q_values.get(name),
            trial_sharpes=tuple(trial_sharpes),
            strongest_base=strongest_base,
            strongest_adverse=strongest_adverse,
        )
        members.append(record)
        if status == "eligible_for_further_review":
            eligible.append(name)

    controls = [
        {
            "candidate_name": name,
            "episode_count": pooled[name].episode_count,
            "base_total_net_return": pooled[name].base_total,
            "adverse_total_net_return": pooled[name].adverse_total,
        }
        for name in control_names
    ]

    decision_status = "eligible_member_available" if eligible else "no_eligible_member"
    material: dict[str, object] = {
        "decision_version": "1.0.0",
        "status": "development_only",
        "family_name": spec.family_name,
        "family_spec_hash": family_spec_hash,
        "capture_root_hash": str(documents[0]["capture_root_hash"]),
        "dataset_root_hash": str(documents[0]["dataset_root_hash"]),
        "split_manifest_hash": str(documents[0]["split_manifest_hash"]),
        "fold_count": fold_count,
        "pooled_episode_count": pooled[member_names[0]].episode_count,
        "source_report_hashes": [str(document["report_hash"]) for document in documents],
        "skipped_sample_ids": sorted(
            {
                value
                for document in documents
                for value in _string_list_field(document, "skipped_sample_ids")
            }
        ),
        "block_length": spec.statistics.block_length,
        "bootstrap_repetitions": spec.statistics.bootstrap_repetitions,
        "random_seed": spec.statistics.random_seed,
        "members": members,
        "controls": controls,
        "eligible_member_names": eligible,
        "decision_status": decision_status,
    }
    report_hash = content_sha256(material)
    document = dict(material)
    document["report_hash"] = report_hash
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_bytes(canonical_json(document))
    temporary.replace(output_path)
    _register_artifact(registry_path, output_path=output_path, report_hash=report_hash)
    return PanelDecisionArtifact(
        output_path=output_path,
        report_hash=report_hash,
        decision_status=decision_status,
        eligible_member_names=tuple(eligible),
    )


def _register_artifact(registry_path: Path, *, output_path: Path, report_hash: str) -> None:
    artifact_id = uuid5(NAMESPACE_URL, f"panel_decision:{report_hash}")
    with MetadataRegistry(registry_path) as registry:
        existing = registry.get_artifact(artifact_id)
        created_at_ns = existing.created_at_ns if existing is not None else time.time_ns()
        registry.register_artifact(
            ArtifactRecord(
                artifact_id=artifact_id,
                kind="panel_decision",
                relative_path=output_path.name,
                content_hash=report_hash,
                created_at_ns=created_at_ns,
            )
        )


def _pool(documents: list[dict[str, object]], name: str) -> _Pooled:
    base: list[Decimal] = []
    adverse: list[Decimal] = []
    fold_totals: list[Decimal] = []
    contracts: dict[str, Decimal] = {}
    for document in documents:
        candidates = document["candidates"]
        if not isinstance(candidates, list):
            raise PanelDecisionError("fold report candidates are malformed")
        record = next(
            (item for item in candidates if item.get("candidate_name") == name), None
        )
        if record is None:
            raise PanelDecisionError(f"fold report is missing candidate {name}")
        fold_total = Decimal(0)
        for episode in record["base"]["episodes"]:
            value = Decimal(str(episode["net_return"]))
            base.append(value)
            fold_total += value
            for contract_id, contribution in episode["contract_net_contributions"]:
                contracts[str(contract_id)] = contracts.get(
                    str(contract_id), Decimal(0)
                ) + Decimal(str(contribution))
        fold_totals.append(fold_total)
        for episode in record["adverse"]["episodes"]:
            adverse.append(Decimal(str(episode["net_return"])))
    return _Pooled(
        base_returns=tuple(base),
        adverse_returns=tuple(adverse),
        base_total=sum(base, Decimal(0)),
        adverse_total=sum(adverse, Decimal(0)),
        fold_base_totals=tuple(fold_totals),
        contract_totals=contracts,
        episode_count=len(base),
    )


def _member_record(
    name: str,
    pooled: _Pooled,
    *,
    spec: PanelFamilySpec,
    fold_count: int,
    test: BootstrapMeanTest | None,
    q_value: Decimal | None,
    trial_sharpes: tuple[Decimal, ...],
    strongest_base: Decimal,
    strongest_adverse: Decimal,
) -> tuple[dict[str, object], str]:
    evidence: list[str] = []
    economic: list[str] = []
    if pooled.episode_count < spec.statistics.pooled_episode_floor:
        evidence.append("EPISODE_FLOOR_NOT_MET")
    if test is None:
        evidence.append("BOOTSTRAP_NOT_AVAILABLE")

    base_mean = (
        pooled.base_total / Decimal(pooled.episode_count)
        if pooled.episode_count
        else Decimal(0)
    )
    adverse_mean = (
        pooled.adverse_total / Decimal(len(pooled.adverse_returns))
        if pooled.adverse_returns
        else Decimal(0)
    )
    lower = test.interval.lower if test is not None else None
    if base_mean <= 0:
        economic.append("AGGREGATE_BASE_NET_NON_POSITIVE")
    if lower is not None and lower <= 0:
        economic.append("BASE_LOWER_BOUND_NON_POSITIVE")
    if adverse_mean < 0:
        economic.append("AGGREGATE_ADVERSE_NET_NON_POSITIVE")
    required = (
        spec.statistics.positive_fold_numerator * fold_count
        + spec.statistics.positive_fold_denominator
        - 1
    ) // spec.statistics.positive_fold_denominator
    positive_folds = sum(1 for value in pooled.fold_base_totals if value > 0)
    if positive_folds < required:
        economic.append("POSITIVE_FOLD_FRACTION_NOT_MET")
    if q_value is not None and q_value > spec.statistics.false_discovery_gate:
        economic.append("MULTIPLE_TESTING_GATE_NOT_MET")

    shares: dict[str, Decimal] = {}
    if pooled.base_total > 0:
        limit = spec.statistics.concentration_limit
        shares = {
            "largest_fold_share": max(pooled.fold_base_totals, default=Decimal(0))
            / pooled.base_total,
            "largest_contract_share": max(
                pooled.contract_totals.values(), default=Decimal(0)
            )
            / pooled.base_total,
            "largest_episode_share": max(pooled.base_returns, default=Decimal(0))
            / pooled.base_total,
        }
        if any(value > limit for value in shares.values()):
            economic.append("CONCENTRATION_LIMIT_EXCEEDED")
    if pooled.base_total <= strongest_base:
        economic.append("BASE_CONTROL_DOMINANCE_NOT_MET")
    if pooled.adverse_total <= strongest_adverse:
        economic.append("ADVERSE_CONTROL_DOMINANCE_NOT_MET")

    if evidence:
        status = "insufficient_evidence"
    elif economic:
        status = "rejected"
    else:
        status = "eligible_for_further_review"

    try:
        annual = annualised_sharpe(pooled.base_returns)
        deflated = deflated_sharpe_ratio(pooled.base_returns, trial_sharpes=trial_sharpes)
    except PanelStatisticsError:
        annual = Decimal(0)
        deflated = Decimal(0)

    record: dict[str, object] = {
        "candidate_name": name,
        "episode_count": pooled.episode_count,
        "base_total_net_return": pooled.base_total,
        "base_mean_net_return": base_mean,
        "adverse_total_net_return": pooled.adverse_total,
        "adverse_mean_net_return": adverse_mean,
        "positive_base_fold_count": positive_folds,
        "required_positive_fold_count": required,
        "base_bootstrap_lower": lower,
        "base_bootstrap_p_value": test.one_sided_p_value if test is not None else None,
        "bh_q_value": q_value,
        "annualised_sharpe": annual,
        "deflated_sharpe_ratio": deflated,
        "concentration": shares,
        "decision_status": status,
        "reason_codes": evidence + economic,
    }
    return record, status


def _load_object(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise PanelDecisionError(f"{path.name} must contain a JSON object")
    return document


def _int_field(record: dict[str, object], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise PanelDecisionError(f"field {key} must be an integer")
    return value


def _list_field(record: dict[str, object], key: str) -> list[object]:
    value = record.get(key)
    if not isinstance(value, list):
        raise PanelDecisionError(f"field {key} must be an array")
    return value


def _string_list_field(record: dict[str, object], key: str) -> list[str]:
    values = _list_field(record, key)
    if not all(isinstance(value, str) for value in values):
        raise PanelDecisionError(f"field {key} must contain strings")
    return [value for value in values if isinstance(value, str)]
