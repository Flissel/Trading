"""Leakage-safe registered calibrated-logistic evaluation on one fold."""

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
from trading_bot.evaluation import EvaluationResult, block_bootstrap_mean_test, evaluate_signals
from trading_bot.horizon import manifest_horizon_bars
from trading_bot.logistic_baseline import (
    LogisticModel,
    PlattCalibrator,
    ProbabilityMetrics,
    fit_logistic,
    fit_platt_calibrator,
    predict_calibrated_probability,
    predict_probability,
    predict_signals,
    probability_metrics,
    select_validation_margin,
)
from trading_bot.market_capture import verify_candle_capture
from trading_bot.registry import ExperimentRecord, MetadataRegistry
from trading_bot.research_run import load_research_bars
from trading_bot.strategy import CostScenario
from trading_bot.walk_forward_run import verify_walk_forward_manifest


class LogisticFoldError(RuntimeError):
    """Raised when a logistic fold cannot be evaluated without leakage."""


@dataclass(frozen=True, slots=True)
class LogisticFoldArtifact:
    output_path: Path
    family_id: UUID
    report_hash: str


def run_logistic_fold(
    capture_root: Path,
    *,
    split_manifest_path: Path,
    output_path: Path,
    registry_path: Path,
    fold_index: int,
    model_l2: Decimal,
    model_iterations: int,
    model_learning_rate: Decimal,
    calibration_l2: Decimal,
    calibration_iterations: int,
    calibration_learning_rate: Decimal,
    minimum_selection_trades: int,
    block_length: int,
    bootstrap_repetitions: int,
    random_seed: int,
) -> LogisticFoldArtifact:
    if output_path.exists():
        raise LogisticFoldError("logistic fold output already exists and is immutable")
    capture_check = verify_candle_capture(capture_root)
    if not capture_check.valid or not verify_walk_forward_manifest(split_manifest_path):
        raise LogisticFoldError("input artifact verification failed")
    capture_manifest = _load_object(capture_root / "capture-manifest.json")
    dataset_manifest = _load_object(capture_root / "dataset" / "dataset-manifest.json")
    split = _load_object(split_manifest_path)
    capture_hash = _string_field(capture_manifest, "capture_root_hash")
    dataset_hash = _string_field(dataset_manifest, "root_hash")
    if (
        split.get("capture_root_hash") != capture_hash
        or split.get("dataset_root_hash") != dataset_hash
    ):
        raise LogisticFoldError("split linkage does not match capture")
    folds = _list_field(split, "folds")
    try:
        horizon_bars = manifest_horizon_bars(split)
    except ValueError as error:
        raise LogisticFoldError(str(error)) from error
    fold = _select_fold(folds, fold_index)
    test_end_ns = _int_field(fold, "test_end_ns")
    bars = load_research_bars(capture_root / "dataset", available_before_ns=test_end_ns)
    primary = tuple(item for item in bars if item.venue == "OKX")
    reference = tuple(item for item in bars if item.venue == "BINANCE")
    samples = {
        item.sample_id: item
        for item in build_bar_samples(primary, reference, horizon_bars=horizon_bars)
    }
    train_ids = tuple(_string_list_field(fold, "train_ids"))
    validation_ids = tuple(_string_list_field(fold, "validation_ids"))
    test_ids = tuple(_string_list_field(fold, "test_ids"))
    training = _select_samples(samples, train_ids)
    validation = _select_samples(samples, validation_ids)
    testing = _select_samples(samples, test_ids)
    calibration, selection = _split_validation(validation)
    calibration_ids = tuple(item.sample_id for item in calibration)
    selection_ids = tuple(item.sample_id for item in selection)

    model = fit_logistic(
        training,
        l2=model_l2,
        iterations=model_iterations,
        learning_rate=model_learning_rate,
    )
    calibrator = fit_platt_calibrator(
        model,
        calibration,
        l2=calibration_l2,
        iterations=calibration_iterations,
        learning_rate=calibration_learning_rate,
    )
    base_costs = CostScenario("base", Decimal("1"), Decimal("1"), Decimal("1"), Decimal("0"))
    adverse_costs = CostScenario("adverse", Decimal("2"), Decimal("2"), Decimal("2"), Decimal("1"))
    margin = select_validation_margin(
        model,
        calibrator,
        selection,
        scenario=base_costs,
        spread_bps=Decimal("2"),
        minimum_trades=minimum_selection_trades,
    )
    selection_signals = predict_signals(model, calibrator, selection, margin=margin)
    selection_result = evaluate_signals(
        selection_signals,
        tuple(item.forward_return for item in selection),
        tuple(Decimal("2") for _ in selection),
        base_costs,
    )
    signals = predict_signals(model, calibrator, testing, margin=margin)
    outcomes = tuple(item.forward_return for item in testing)
    spreads = tuple(Decimal("2") for _ in testing)
    base = evaluate_signals(signals, outcomes, spreads, base_costs)
    adverse = evaluate_signals(signals, outcomes, spreads, adverse_costs)
    active = tuple(
        value for signal, value in zip(signals, base.net_returns, strict=True) if signal != 0
    )
    bootstrap = _bootstrap(active, block_length, bootstrap_repetitions, random_seed)
    q_value = bootstrap["one_sided_p_value"] if bootstrap is not None else Decimal(1)
    decision_status, decision_reasons = _decision(
        fold_count=len(folds),
        trade_count=base.trade_count,
        base_total=base.total_net_return,
        adverse_total=adverse.total_net_return,
        lower=None if bootstrap is None else bootstrap["lower"],
    )
    candidate = {
        "candidate_name": "logistic_calibrated",
        "base_trade_count": base.trade_count,
        "base": _evaluation_record(base),
        "adverse": _evaluation_record(adverse),
        "base_median_net_return": Decimal(median(active)) if active else Decimal(0),
        "base_bootstrap": bootstrap,
        "bh_q_value": q_value,
        "decision_status": decision_status,
        "decision_reason_codes": list(decision_reasons),
    }
    split_hash = _string_field(split, "manifest_hash")
    material: dict[str, object] = {
        "report_version": "1.0.0",
        "status": "development_only",
        "reason_codes": list(decision_reasons),
        "capture_root_hash": capture_hash,
        "dataset_root_hash": dataset_hash,
        "split_manifest_hash": split_hash,
        "horizon_bars": horizon_bars,
        "fold_index": fold_index,
        "fold_count": len(folds),
        "train_sample_count": len(training),
        "validation_sample_count": len(validation),
        "calibration_sample_count": len(calibration),
        "selection_sample_count": len(selection),
        "test_sample_count": len(testing),
        "train_membership_hash": content_sha256(list(train_ids)),
        "validation_membership_hash": content_sha256(list(validation_ids)),
        "calibration_membership_hash": content_sha256(list(calibration_ids)),
        "selection_membership_hash": content_sha256(list(selection_ids)),
        "test_membership_hash": content_sha256(list(test_ids)),
        "calibration_end_decision_time_ns": calibration[-1].decision_time_ns,
        "selection_start_decision_time_ns": selection[0].decision_time_ns,
        "random_seed": random_seed,
        "block_length": block_length,
        "bootstrap_repetitions": bootstrap_repetitions,
        "model": _model_record(
            model,
            calibrator,
            margin,
            minimum_selection_trades,
            selection_result.trade_count,
        ),
        "probability_metrics": _probability_metric_records(
            model, calibrator, calibration, selection, testing
        ),
        "candidates": [candidate],
    }
    report_hash = content_sha256(material)
    document = dict(material)
    document["report_hash"] = report_hash
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_bytes(canonical_json(document))
    temporary.replace(output_path)
    family_id = uuid5(NAMESPACE_URL, f"{split_hash}:logistic_calibrated:fold:{fold_index}")
    _register(registry_path, family_id, split_hash, report_hash, random_seed)
    return LogisticFoldArtifact(output_path, family_id, report_hash)


def verify_logistic_fold_report(path: Path) -> bool:
    try:
        document = _load_object(path)
        recorded = _string_field(document, "report_hash")
        material = dict(document)
        del material["report_hash"]
        return content_sha256(material) == recorded
    except (OSError, json.JSONDecodeError, LogisticFoldError, KeyError, TypeError, ValueError):
        return False


def _split_validation(
    validation: tuple[BarSample, ...],
) -> tuple[tuple[BarSample, ...], tuple[BarSample, ...]]:
    midpoint = len(validation) // 2
    if midpoint < 1 or midpoint == len(validation):
        raise LogisticFoldError("validation requires separate calibration and selection rows")
    return validation[:midpoint], validation[midpoint:]


def _probability_metric_records(
    model: LogisticModel,
    calibrator: PlattCalibrator,
    calibration: tuple[BarSample, ...],
    selection: tuple[BarSample, ...],
    testing: tuple[BarSample, ...],
) -> dict[str, object]:
    raw_calibration = tuple(predict_probability(model, item) for item in calibration)
    calibrated_calibration = tuple(
        predict_calibrated_probability(model, calibrator, item) for item in calibration
    )
    raw_selection = tuple(predict_probability(model, item) for item in selection)
    calibrated_selection = tuple(
        predict_calibrated_probability(model, calibrator, item) for item in selection
    )
    raw_test = tuple(predict_probability(model, item) for item in testing)
    calibrated_test = tuple(
        predict_calibrated_probability(model, calibrator, item) for item in testing
    )
    return {
        "calibration_raw": _probability_record(probability_metrics(raw_calibration, calibration)),
        "calibration_calibrated": _probability_record(
            probability_metrics(calibrated_calibration, calibration)
        ),
        "selection_raw": _probability_record(probability_metrics(raw_selection, selection)),
        "selection_calibrated": _probability_record(
            probability_metrics(calibrated_selection, selection)
        ),
        "test_raw": _probability_record(probability_metrics(raw_test, testing)),
        "test_calibrated": _probability_record(probability_metrics(calibrated_test, testing)),
    }


def _probability_record(metrics: ProbabilityMetrics) -> dict[str, Decimal]:
    return {"brier_score": metrics.brier_score, "log_loss": metrics.log_loss}


def _model_record(
    model: LogisticModel,
    calibrator: PlattCalibrator,
    margin: Decimal,
    minimum_selection_trades: int,
    selected_trade_count: int,
) -> dict[str, object]:
    return {
        "kind": "logistic_calibrated",
        "feature_means": list(model.feature_means),
        "feature_scales": list(model.feature_scales),
        "coefficients": list(model.coefficients),
        "intercept": model.intercept,
        "l2": model.l2,
        "iterations": model.iterations,
        "learning_rate": model.learning_rate,
        "calibrator": {
            "kind": "platt",
            "slope": calibrator.slope,
            "intercept": calibrator.intercept,
            "l2": calibrator.l2,
            "iterations": calibrator.iterations,
            "learning_rate": calibrator.learning_rate,
        },
        "selection_margin": margin,
        "selection_minimum_trades": minimum_selection_trades,
        "selection_selected_trade_count": selected_trade_count,
    }


def _decision(
    *,
    fold_count: int,
    trade_count: int,
    base_total: Decimal,
    adverse_total: Decimal,
    lower: Decimal | None,
) -> tuple[str, tuple[str, ...]]:
    evidence: list[str] = []
    if fold_count < 3:
        evidence.append("FOLD_FLOOR_NOT_MET")
    if trade_count < 200:
        evidence.append("EPISODE_FLOOR_NOT_MET")
    economic: list[str] = []
    if base_total <= 0:
        economic.append("BASE_NET_NON_POSITIVE")
    if lower is None or lower <= 0:
        economic.append("BASE_LOWER_BOUND_NON_POSITIVE")
    if adverse_total <= 0:
        economic.append("ADVERSE_NET_NON_POSITIVE")
    reasons = tuple(evidence + economic)
    return (
        "insufficient_evidence"
        if evidence
        else "rejected"
        if economic
        else "eligible_for_further_review",
        reasons,
    )


def _bootstrap(
    values: tuple[Decimal, ...], block_length: int, repetitions: int, seed: int
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


def _evaluation_record(value: EvaluationResult) -> dict[str, object]:
    return {
        "trade_count": value.trade_count,
        "total_net_return": value.total_net_return,
        "mean_net_return": value.mean_net_return,
        "win_rate": value.win_rate,
        "maximum_drawdown": value.maximum_drawdown,
    }


def _register(path: Path, family: UUID, split_hash: str, report_hash: str, seed: int) -> None:
    code_hash = _code_hash()
    experiment_id = uuid5(family, f"logistic_calibrated:{seed}:{report_hash}")
    with MetadataRegistry(path) as registry:
        existing = registry.get_experiment(experiment_id)
        created = existing.created_at_ns if existing is not None else time.time_ns()
        registry.register_experiment(
            ExperimentRecord(
                experiment_id=experiment_id,
                family_id=family,
                candidate_name="logistic_calibrated",
                hypothesis="calibrated direction probabilities improve net next-bar decisions",
                split_manifest_hash=split_hash,
                code_hash=code_hash,
                random_seed=seed,
                outcome="completed",
                result_hash=report_hash,
                failure_reason=None,
                created_at_ns=created,
            )
        )


def _code_hash() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for name in ("logistic_baseline.py", "logistic_fold_run.py"):
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def _select_samples(mapping: dict[str, BarSample], ids: tuple[str, ...]) -> tuple[BarSample, ...]:
    try:
        return tuple(mapping[value] for value in ids)
    except KeyError as error:
        raise LogisticFoldError("fold sample is unavailable before test boundary") from error


def _select_fold(values: list[object], index: int) -> dict[str, object]:
    for value in values:
        if isinstance(value, dict) and value.get("fold_index") == index:
            return value
    raise LogisticFoldError("requested fold does not exist")


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise LogisticFoldError(f"{path.name} must be an object")
    return value


def _string_field(record: dict[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str):
        raise LogisticFoldError(f"field {key} must be a string")
    return value


def _int_field(record: dict[str, object], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise LogisticFoldError(f"field {key} must be an integer")
    return value


def _list_field(record: dict[str, object], key: str) -> list[object]:
    value = record.get(key)
    if not isinstance(value, list):
        raise LogisticFoldError(f"field {key} must be an array")
    return value


def _string_list_field(record: dict[str, object], key: str) -> list[str]:
    values = _list_field(record, key)
    if not all(isinstance(value, str) for value in values):
        raise LogisticFoldError(f"field {key} must contain strings")
    return [value for value in values if isinstance(value, str)]
