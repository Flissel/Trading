"""Holdout-locked fold evaluation and append-only experiment registration."""

import hashlib
import json
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from statistics import median
from uuid import NAMESPACE_URL, UUID, uuid5

from trading_bot.bar_research import BarSample, build_bar_samples
from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.evaluation import (
    EvaluationResult,
    benjamini_hochberg,
    block_bootstrap_mean_test,
    evaluate_signals,
)
from trading_bot.horizon import manifest_horizon_bars
from trading_bot.market_capture import verify_candle_capture
from trading_bot.registry import ExperimentRecord, MetadataRegistry
from trading_bot.research_run import load_research_bars
from trading_bot.strategy import CostScenario, generate_baseline_signals
from trading_bot.walk_forward_run import verify_walk_forward_manifest


class FoldEvaluationError(RuntimeError):
    """Raised when a registered fold evaluation cannot be completed safely."""


@dataclass(frozen=True, slots=True)
class FoldEvaluationArtifact:
    output_path: Path
    family_id: UUID
    report_hash: str
    candidate_count: int


def fold_reason_codes(*, fold_count: int, trade_counts: dict[str, int]) -> tuple[str, ...]:
    reasons: list[str] = []
    if fold_count < 3:
        reasons.append("FOLD_FLOOR_NOT_MET")
    tradable_counts = (count for name, count in trade_counts.items() if name != "no_trade")
    if any(count < 200 for count in tradable_counts):
        reasons.append("EPISODE_FLOOR_NOT_MET")
    return tuple(reasons)


def run_fold_evaluation(
    capture_root: Path,
    *,
    split_manifest_path: Path,
    output_path: Path,
    registry_path: Path,
    fold_index: int,
    random_seed: int,
    block_length: int,
    bootstrap_repetitions: int,
) -> FoldEvaluationArtifact:
    if output_path.exists():
        raise FoldEvaluationError("fold evaluation output already exists and is immutable")
    capture_verification = verify_candle_capture(capture_root)
    if not capture_verification.valid or not verify_walk_forward_manifest(split_manifest_path):
        raise FoldEvaluationError("input artifact verification failed")
    capture_manifest = _load_object(capture_root / "capture-manifest.json")
    dataset_manifest = _load_object(capture_root / "dataset" / "dataset-manifest.json")
    split_manifest = _load_object(split_manifest_path)
    capture_hash = _string_field(capture_manifest, "capture_root_hash")
    dataset_hash = _string_field(dataset_manifest, "root_hash")
    if split_manifest.get("capture_root_hash") != capture_hash:
        raise FoldEvaluationError("split manifest is linked to a different capture")
    if split_manifest.get("dataset_root_hash") != dataset_hash:
        raise FoldEvaluationError("split manifest is linked to a different dataset")
    folds = _list_field(split_manifest, "folds")
    try:
        horizon_bars = manifest_horizon_bars(split_manifest)
    except ValueError as error:
        raise FoldEvaluationError(str(error)) from error
    fold = _select_fold(folds, fold_index)
    test_end_ns = _int_field(fold, "test_end_ns")
    test_ids = tuple(_string_list_field(fold, "test_ids"))
    bars = load_research_bars(capture_root / "dataset", available_before_ns=test_end_ns)
    primary = tuple(item for item in bars if item.venue == "OKX")
    reference = tuple(item for item in bars if item.venue == "BINANCE")
    samples_by_id = {
        item.sample_id: item
        for item in build_bar_samples(primary, reference, horizon_bars=horizon_bars)
    }
    try:
        samples = tuple(samples_by_id[sample_id] for sample_id in test_ids)
    except KeyError as error:
        raise FoldEvaluationError("fold sample is unavailable before the test boundary") from error
    candidates = _evaluate_candidates(
        samples,
        fold_count=len(folds),
        random_seed=random_seed,
        block_length=block_length,
        bootstrap_repetitions=bootstrap_repetitions,
    )
    reasons = fold_reason_codes(
        fold_count=len(folds),
        trade_counts={
            _string_field(item, "candidate_name"): _int_field(item, "base_trade_count")
            for item in candidates
        },
    )
    material: dict[str, object] = {
        "report_version": "1.0.0",
        "status": "development_only",
        "reason_codes": list(reasons),
        "capture_root_hash": capture_hash,
        "dataset_root_hash": dataset_hash,
        "split_manifest_hash": _string_field(split_manifest, "manifest_hash"),
        "horizon_bars": horizon_bars,
        "fold_index": fold_index,
        "fold_count": len(folds),
        "test_end_ns": test_end_ns,
        "evaluated_sample_ids": list(test_ids),
        "random_seed": random_seed,
        "block_length": block_length,
        "bootstrap_repetitions": bootstrap_repetitions,
        "bootstrap_confidence": Decimal("0.95"),
        "candidates": candidates,
    }
    report_hash = content_sha256(material)
    document = dict(material)
    document["report_hash"] = report_hash
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_bytes(canonical_json(document))
    temporary.replace(output_path)
    family_id = uuid5(NAMESPACE_URL, f"{material['split_manifest_hash']}:fold:{fold_index}")
    _register_candidates(
        registry_path,
        family_id=family_id,
        split_manifest_hash=_string_field(split_manifest, "manifest_hash"),
        report_hash=report_hash,
        random_seed=random_seed,
        candidate_names=tuple(_string_field(item, "candidate_name") for item in candidates),
    )
    return FoldEvaluationArtifact(output_path, family_id, report_hash, len(candidates))


def verify_fold_evaluation(path: Path) -> bool:
    try:
        document = _load_object(path)
        recorded_hash = _string_field(document, "report_hash")
        material = dict(document)
        del material["report_hash"]
        return content_sha256(material) == recorded_hash
    except (OSError, json.JSONDecodeError, FoldEvaluationError, KeyError, TypeError, ValueError):
        return False


def _evaluate_candidates(
    samples: tuple[BarSample, ...],
    *,
    fold_count: int,
    random_seed: int,
    block_length: int,
    bootstrap_repetitions: int,
) -> list[dict[str, object]]:
    if not samples:
        raise FoldEvaluationError("fold contains no evaluable test samples")
    observed = tuple(item.observed_return for item in samples)
    outcomes = tuple(item.forward_return for item in samples)
    spreads = tuple(Decimal("2") for _ in samples)
    signals_by_name = generate_baseline_signals(observed, random_seed=random_seed)
    base_costs = CostScenario("base", Decimal("1"), Decimal("1"), Decimal("1"), Decimal("0"))
    adverse_costs = CostScenario("adverse", Decimal("2"), Decimal("2"), Decimal("2"), Decimal("1"))
    records: list[dict[str, object]] = []
    for name, signals in signals_by_name.items():
        base = evaluate_signals(signals, outcomes, spreads, base_costs)
        adverse = evaluate_signals(signals, outcomes, spreads, adverse_costs)
        active_base = tuple(
            value for signal, value in zip(signals, base.net_returns, strict=True) if signal != 0
        )
        records.append(
            {
                "candidate_name": name,
                "base_trade_count": base.trade_count,
                "base": _evaluation_record(base),
                "adverse": _evaluation_record(adverse),
                "base_median_net_return": (
                    Decimal(median(active_base)) if active_base else Decimal(0)
                ),
                "base_bootstrap": _bootstrap_record(
                    active_base,
                    block_length=block_length,
                    repetitions=bootstrap_repetitions,
                    seed=random_seed,
                ),
            }
        )
    p_values = {
        _string_field(record, "candidate_name"): _bootstrap_p_value(record) for record in records
    }
    q_values = benjamini_hochberg(p_values)
    for record in records:
        record["bh_q_value"] = q_values[_string_field(record, "candidate_name")]
        status, reason_codes = _candidate_decision(record, fold_count=fold_count)
        record["decision_status"] = status
        record["decision_reason_codes"] = list(reason_codes)
    return records


def _candidate_decision(
    record: dict[str, object], *, fold_count: int
) -> tuple[str, tuple[str, ...]]:
    name = _string_field(record, "candidate_name")
    if name == "no_trade":
        return "contextual_baseline", ("NON_PROMOTABLE_CONTEXT",)
    evidence_reasons: list[str] = []
    if fold_count < 3:
        evidence_reasons.append("FOLD_FLOOR_NOT_MET")
    if _int_field(record, "base_trade_count") < 200:
        evidence_reasons.append("EPISODE_FLOOR_NOT_MET")
    economic_reasons: list[str] = []
    base = record.get("base")
    adverse = record.get("adverse")
    if not isinstance(base, dict) or not isinstance(adverse, dict):
        raise FoldEvaluationError("candidate evaluations must be objects")
    if _decimal_field(base, "total_net_return") <= 0:
        economic_reasons.append("BASE_NET_NON_POSITIVE")
    bootstrap = record.get("base_bootstrap")
    if not isinstance(bootstrap, dict) or _decimal_field(bootstrap, "lower") <= 0:
        economic_reasons.append("BASE_LOWER_BOUND_NON_POSITIVE")
    if _decimal_field(adverse, "total_net_return") <= 0:
        economic_reasons.append("ADVERSE_NET_NON_POSITIVE")
    reasons = tuple(evidence_reasons + economic_reasons)
    if evidence_reasons:
        return "insufficient_evidence", reasons
    if economic_reasons:
        return "rejected", reasons
    return "eligible_for_further_review", ()


def _bootstrap_record(
    values: tuple[Decimal, ...], *, block_length: int, repetitions: int, seed: int
) -> dict[str, Decimal] | None:
    if not values:
        return None
    result = block_bootstrap_mean_test(
        values,
        block_length=min(block_length, len(values)),
        repetitions=repetitions,
        seed=seed,
        confidence=Decimal("0.95"),
    )
    return {
        "lower": result.interval.lower,
        "upper": result.interval.upper,
        "one_sided_p_value": result.one_sided_p_value,
    }


def _bootstrap_p_value(record: dict[str, object]) -> Decimal:
    bootstrap = record.get("base_bootstrap")
    if bootstrap is None:
        return Decimal(1)
    if not isinstance(bootstrap, dict):
        raise FoldEvaluationError("bootstrap record must be an object")
    value = bootstrap.get("one_sided_p_value")
    if not isinstance(value, Decimal):
        raise FoldEvaluationError("bootstrap p-value must be a Decimal")
    return value


def _evaluation_record(result: EvaluationResult) -> dict[str, object]:
    return {
        "trade_count": result.trade_count,
        "total_net_return": result.total_net_return,
        "mean_net_return": result.mean_net_return,
        "win_rate": result.win_rate,
        "maximum_drawdown": result.maximum_drawdown,
    }


def _register_candidates(
    registry_path: Path,
    *,
    family_id: UUID,
    split_manifest_hash: str,
    report_hash: str,
    random_seed: int,
    candidate_names: tuple[str, ...],
) -> None:
    code_hash = _code_hash()
    now = time.time_ns()
    with MetadataRegistry(registry_path) as registry:
        for index, candidate_name in enumerate(candidate_names):
            experiment_id = uuid5(family_id, f"{candidate_name}:{random_seed}:{report_hash}")
            existing = registry.get_experiment(experiment_id)
            created_at_ns = existing.created_at_ns if existing is not None else now + index
            registry.register_experiment(
                ExperimentRecord(
                    experiment_id=experiment_id,
                    family_id=family_id,
                    candidate_name=candidate_name,
                    hypothesis=f"registered baseline: {candidate_name}",
                    split_manifest_hash=split_manifest_hash,
                    code_hash=code_hash,
                    random_seed=random_seed,
                    outcome="completed",
                    result_hash=report_hash,
                    failure_reason=None,
                    created_at_ns=created_at_ns,
                )
            )


def _code_hash() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for name in ("evaluation.py", "fold_evaluation.py", "strategy.py"):
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def _select_fold(folds: list[object], fold_index: int) -> dict[str, object]:
    for item in folds:
        if isinstance(item, dict) and item.get("fold_index") == fold_index:
            return item
    raise FoldEvaluationError("requested fold does not exist")


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise FoldEvaluationError(f"{path.name} must be a JSON object")
    return value


def _string_field(record: dict[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str):
        raise FoldEvaluationError(f"field {key} must be a string")
    return value


def _int_field(record: dict[str, object], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise FoldEvaluationError(f"field {key} must be an integer")
    return value


def _list_field(record: dict[str, object], key: str) -> list[object]:
    value = record.get(key)
    if not isinstance(value, list):
        raise FoldEvaluationError(f"field {key} must be an array")
    return value


def _decimal_field(record: dict[str, object], key: str) -> Decimal:
    value = record.get(key)
    if not isinstance(value, Decimal):
        raise FoldEvaluationError(f"field {key} must be a Decimal")
    return value


def _string_list_field(record: dict[str, object], key: str) -> list[str]:
    values = _list_field(record, key)
    if not all(isinstance(value, str) for value in values):
        raise FoldEvaluationError(f"field {key} must contain strings")
    return [value for value in values if isinstance(value, str)]
