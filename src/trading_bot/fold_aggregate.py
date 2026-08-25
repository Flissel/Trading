"""Fold-complete aggregation for registered baseline evaluations."""

import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.fold_evaluation import verify_fold_evaluation


class FoldAggregateError(RuntimeError):
    """Raised when fold reports cannot form one complete experiment family."""


@dataclass(frozen=True, slots=True)
class FoldAggregateArtifact:
    output_path: Path
    report_hash: str
    fold_count: int


def aggregate_fold_evaluations(
    fold_report_paths: tuple[Path, ...], *, output_path: Path
) -> FoldAggregateArtifact:
    if output_path.exists():
        raise FoldAggregateError("fold aggregate output already exists and is immutable")
    if not fold_report_paths:
        raise FoldAggregateError("fold report family must not be empty")
    documents = tuple(_load_verified(path) for path in fold_report_paths)
    expected_fold_count = _int_field(documents[0], "fold_count")
    indices = {_int_field(document, "fold_index") for document in documents}
    if len(documents) != expected_fold_count or indices != set(range(expected_fold_count)):
        raise FoldAggregateError("fold report family is not complete")
    linkage_fields = (
        "capture_root_hash",
        "dataset_root_hash",
        "split_manifest_hash",
    )
    for field in linkage_fields:
        expected = _string_field(documents[0], field)
        if any(_string_field(document, field) != expected for document in documents[1:]):
            raise FoldAggregateError(f"fold reports disagree on {field}")
    candidate_maps = tuple(_candidate_map(document) for document in documents)
    names = tuple(candidate_maps[0])
    if any(tuple(candidate_map) != names for candidate_map in candidate_maps[1:]):
        raise FoldAggregateError("fold reports contain different candidates")
    candidates = [
        _aggregate_candidate(
            name,
            tuple(candidate_map[name] for candidate_map in candidate_maps),
            fold_count=expected_fold_count,
        )
        for name in names
    ]
    material: dict[str, object] = {
        "aggregate_version": "1.0.0",
        "status": "development_only",
        "capture_root_hash": _string_field(documents[0], "capture_root_hash"),
        "dataset_root_hash": _string_field(documents[0], "dataset_root_hash"),
        "split_manifest_hash": _string_field(documents[0], "split_manifest_hash"),
        "fold_count": expected_fold_count,
        "fold_report_hashes": [_string_field(document, "report_hash") for document in documents],
        "candidates": candidates,
    }
    report_hash = content_sha256(material)
    document = dict(material)
    document["report_hash"] = report_hash
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_bytes(canonical_json(document))
    temporary.replace(output_path)
    return FoldAggregateArtifact(output_path, report_hash, expected_fold_count)


def verify_fold_aggregate(path: Path) -> bool:
    try:
        document = _load_object(path)
        recorded_hash = _string_field(document, "report_hash")
        material = dict(document)
        del material["report_hash"]
        return content_sha256(material) == recorded_hash
    except (OSError, json.JSONDecodeError, FoldAggregateError, KeyError, TypeError, ValueError):
        return False


def _aggregate_candidate(
    name: str, records: tuple[dict[str, object], ...], *, fold_count: int
) -> dict[str, object]:
    base_totals = tuple(_nested_decimal(record, "base", "total_net_return") for record in records)
    adverse_totals = tuple(
        _nested_decimal(record, "adverse", "total_net_return") for record in records
    )
    positive_count = sum(value > 0 for value in base_totals)
    aggregate_base = sum(base_totals, Decimal(0))
    aggregate_adverse = sum(adverse_totals, Decimal(0))
    maximum_q = max(_decimal_field(record, "bh_q_value") for record in records)
    reasons: tuple[str, ...]
    if name == "no_trade":
        status = "contextual_baseline"
        reasons = ("NON_PROMOTABLE_CONTEXT",)
    else:
        reason_list: list[str] = []
        required_positive = (2 * fold_count + 2) // 3
        if positive_count < required_positive:
            reason_list.append("POSITIVE_FOLD_FRACTION_NOT_MET")
        if aggregate_base <= 0:
            reason_list.append("AGGREGATE_BASE_NET_NON_POSITIVE")
        if aggregate_adverse <= 0:
            reason_list.append("AGGREGATE_ADVERSE_NET_NON_POSITIVE")
        if maximum_q > Decimal("0.10"):
            reason_list.append("MULTIPLE_TESTING_GATE_NOT_MET")
        reasons = tuple(reason_list)
        status = "rejected" if reasons else "eligible_for_further_review"
    return {
        "candidate_name": name,
        "positive_base_fold_count": positive_count,
        "aggregate_base_total_net_return": aggregate_base,
        "aggregate_adverse_total_net_return": aggregate_adverse,
        "maximum_bh_q_value": maximum_q,
        "decision_status": status,
        "reason_codes": list(reasons),
    }


def _load_verified(path: Path) -> dict[str, object]:
    if not verify_fold_evaluation(path):
        raise FoldAggregateError(f"fold report verification failed: {path.name}")
    return _load_object(path)


def _candidate_map(document: dict[str, object]) -> dict[str, dict[str, object]]:
    values = document.get("candidates")
    if not isinstance(values, list):
        raise FoldAggregateError("candidates must be an array")
    result: dict[str, dict[str, object]] = {}
    for value in values:
        if not isinstance(value, dict):
            raise FoldAggregateError("candidate must be an object")
        name = _string_field(value, "candidate_name")
        if name in result:
            raise FoldAggregateError("candidate names must be unique")
        result[name] = value
    return result


def _nested_decimal(record: dict[str, object], outer: str, inner: str) -> Decimal:
    value = record.get(outer)
    if not isinstance(value, dict):
        raise FoldAggregateError(f"field {outer} must be an object")
    return _decimal_field(value, inner)


def _decimal_field(record: dict[str, object], key: str) -> Decimal:
    value = record.get(key)
    if not isinstance(value, str):
        raise FoldAggregateError(f"field {key} must be a decimal string")
    try:
        number = Decimal(value)
    except InvalidOperation as error:
        raise FoldAggregateError(f"field {key} is not a decimal") from error
    if not number.is_finite():
        raise FoldAggregateError(f"field {key} must be finite")
    return number


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise FoldAggregateError(f"{path.name} must be a JSON object")
    return value


def _string_field(record: dict[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str):
        raise FoldAggregateError(f"field {key} must be a string")
    return value


def _int_field(record: dict[str, object], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise FoldAggregateError(f"field {key} must be an integer")
    return value
