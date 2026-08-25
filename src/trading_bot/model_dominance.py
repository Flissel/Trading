"""Verified cross-model dominance report for aggregate evaluations."""

import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.fold_aggregate import verify_fold_aggregate


class ModelDominanceError(RuntimeError):
    """Raised when aggregate reports cannot be compared safely."""


@dataclass(frozen=True, slots=True)
class ModelDominanceArtifact:
    output_path: Path
    report_hash: str


def build_model_dominance_report(
    aggregate_paths: tuple[Path, ...], *, output_path: Path
) -> ModelDominanceArtifact:
    if output_path.exists():
        raise ModelDominanceError("dominance report already exists and is immutable")
    if not aggregate_paths:
        raise ModelDominanceError("dominance comparison requires aggregate reports")
    documents = tuple(_load_verified(path) for path in aggregate_paths)
    for field in ("capture_root_hash", "dataset_root_hash", "split_manifest_hash", "fold_count"):
        expected = documents[0].get(field)
        if any(document.get(field) != expected for document in documents[1:]):
            raise ModelDominanceError(f"aggregate reports disagree on {field}")
    candidates: list[tuple[dict[str, object], str]] = []
    names: set[str] = set()
    for document in documents:
        source_hash = _string_field(document, "report_hash")
        for candidate in _candidate_records(document):
            name = _string_field(candidate, "candidate_name")
            if name in names:
                raise ModelDominanceError("candidate names must be unique across aggregates")
            names.add(name)
            candidates.append((candidate, source_hash))
    no_trade = next(
        (candidate for candidate, _ in candidates if candidate["candidate_name"] == "no_trade"),
        None,
    )
    if no_trade is None:
        raise ModelDominanceError("dominance comparison requires no_trade")
    no_trade_base = _decimal_field(no_trade, "aggregate_base_total_net_return")
    no_trade_adverse = _decimal_field(no_trade, "aggregate_adverse_total_net_return")
    records = [
        _dominance_record(candidate, source_hash, no_trade_base, no_trade_adverse)
        for candidate, source_hash in candidates
    ]
    eligible = sorted(
        (
            record
            for record in records
            if record["input_decision_status"] == "eligible_for_further_review"
            and record["dominates_no_trade"] is True
        ),
        key=lambda record: (
            _decimal_field(record, "aggregate_base_total_net_return"),
            _decimal_field(record, "aggregate_adverse_total_net_return"),
        ),
        reverse=True,
    )
    eligible_names = [_string_field(record, "candidate_name") for record in eligible]
    material: dict[str, object] = {
        "dominance_version": "1.0.0",
        "status": "development_only",
        "capture_root_hash": _string_field(documents[0], "capture_root_hash"),
        "dataset_root_hash": _string_field(documents[0], "dataset_root_hash"),
        "split_manifest_hash": _string_field(documents[0], "split_manifest_hash"),
        "fold_count": _int_field(documents[0], "fold_count"),
        "source_aggregate_hashes": [
            _string_field(document, "report_hash") for document in documents
        ],
        "no_trade_candidate_name": "no_trade",
        "candidates": records,
        "eligible_candidate_names": eligible_names,
        "strongest_eligible_candidate": eligible_names[0] if eligible_names else None,
        "decision_status": "eligible_candidate_available"
        if eligible_names
        else "no_eligible_candidate",
    }
    report_hash = content_sha256(material)
    document = dict(material)
    document["report_hash"] = report_hash
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_bytes(canonical_json(document))
    temporary.replace(output_path)
    return ModelDominanceArtifact(output_path, report_hash)


def verify_model_dominance_report(path: Path) -> bool:
    try:
        document = _load_object(path)
        recorded = _string_field(document, "report_hash")
        material = dict(document)
        del material["report_hash"]
        return content_sha256(material) == recorded
    except (OSError, json.JSONDecodeError, ModelDominanceError, KeyError, TypeError, ValueError):
        return False


def _dominance_record(
    candidate: dict[str, object],
    source_hash: str,
    no_trade_base: Decimal,
    no_trade_adverse: Decimal,
) -> dict[str, object]:
    base = _decimal_field(candidate, "aggregate_base_total_net_return")
    adverse = _decimal_field(candidate, "aggregate_adverse_total_net_return")
    positive_folds = _int_field(candidate, "positive_base_fold_count")
    q_value = _decimal_field(candidate, "maximum_bh_q_value")
    input_status = _string_field(candidate, "decision_status")
    dominates = (
        base > no_trade_base
        and adverse > no_trade_adverse
        and positive_folds >= 2
        and q_value <= Decimal("0.10")
        and input_status == "eligible_for_further_review"
    )
    return {
        "candidate_name": _string_field(candidate, "candidate_name"),
        "source_aggregate_hash": source_hash,
        "aggregate_base_total_net_return": base,
        "aggregate_adverse_total_net_return": adverse,
        "base_margin_vs_no_trade": base - no_trade_base,
        "adverse_margin_vs_no_trade": adverse - no_trade_adverse,
        "positive_base_fold_count": positive_folds,
        "maximum_bh_q_value": q_value,
        "input_decision_status": input_status,
        "dominates_no_trade": dominates,
    }


def _load_verified(path: Path) -> dict[str, object]:
    if not verify_fold_aggregate(path):
        raise ModelDominanceError(f"aggregate verification failed: {path.name}")
    return _load_object(path)


def _candidate_records(document: dict[str, object]) -> tuple[dict[str, object], ...]:
    values = document.get("candidates")
    if not isinstance(values, list) or not all(isinstance(value, dict) for value in values):
        raise ModelDominanceError("candidates must be objects")
    return tuple(value for value in values if isinstance(value, dict))


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ModelDominanceError(f"{path.name} must be an object")
    return value


def _string_field(record: dict[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str):
        raise ModelDominanceError(f"field {key} must be a string")
    return value


def _int_field(record: dict[str, object], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ModelDominanceError(f"field {key} must be an integer")
    return value


def _decimal_field(record: dict[str, object], key: str) -> Decimal:
    value = record.get(key)
    if isinstance(value, Decimal):
        return value
    if not isinstance(value, str):
        raise ModelDominanceError(f"field {key} must be a decimal string")
    try:
        result = Decimal(value)
    except InvalidOperation as error:
        raise ModelDominanceError(f"field {key} must be a decimal") from error
    if not result.is_finite():
        raise ModelDominanceError(f"field {key} must be finite")
    return result
