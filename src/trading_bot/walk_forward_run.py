"""Publish immutable walk-forward membership linked to a verified capture."""

import json
from dataclasses import dataclass
from pathlib import Path

from trading_bot.bar_research import build_bar_samples
from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.market_capture import verify_candle_capture
from trading_bot.research_run import load_research_bars
from trading_bot.splits import (
    SplitSample,
    WalkForwardConfig,
    WalkForwardFold,
    build_walk_forward_views,
)


class WalkForwardRunError(RuntimeError):
    """Raised when fold membership cannot be safely published."""


@dataclass(frozen=True, slots=True)
class WalkForwardArtifact:
    output_path: Path
    capture_root_hash: str
    dataset_root_hash: str
    manifest_hash: str
    fold_count: int


def derive_walk_forward_config(
    capture_root: Path,
    *,
    horizon_bars: int = 1,
    train_duration_ns: int,
    validation_duration_ns: int,
    test_duration_ns: int,
    step_ns: int,
    embargo_ns: int,
    holdout_duration_ns: int,
) -> WalkForwardConfig:
    """Place an untouched holdout at the chronological end of a verified capture."""
    verification = verify_candle_capture(capture_root)
    if not verification.valid:
        raise WalkForwardRunError("capture verification failed")
    bars = load_research_bars(capture_root / "dataset")
    primary = tuple(item for item in bars if item.venue == "OKX")
    reference = tuple(item for item in bars if item.venue == "BINANCE")
    samples = build_bar_samples(primary, reference, horizon_bars=horizon_bars)
    if len(samples) < 2 or holdout_duration_ns <= 0:
        raise WalkForwardRunError("capture cannot define a final holdout")
    sample_spacing_ns = samples[-1].decision_time_ns - samples[-2].decision_time_ns
    if holdout_duration_ns < sample_spacing_ns:
        raise WalkForwardRunError("holdout duration is shorter than one sample interval")
    holdout_start = samples[-1].decision_time_ns - holdout_duration_ns + sample_spacing_ns
    return WalkForwardConfig(
        train_duration_ns=train_duration_ns,
        validation_duration_ns=validation_duration_ns,
        test_duration_ns=test_duration_ns,
        step_ns=step_ns,
        embargo_ns=embargo_ns,
        final_holdout_start_ns=holdout_start,
    )


def run_capture_walk_forward(
    capture_root: Path,
    *,
    output_path: Path,
    config: WalkForwardConfig,
    horizon_bars: int = 1,
) -> WalkForwardArtifact:
    verification = verify_candle_capture(capture_root)
    if not verification.valid:
        raise WalkForwardRunError("capture verification failed")
    if output_path.exists():
        raise WalkForwardRunError("walk-forward manifest already exists and is immutable")
    capture_manifest = _load_object(capture_root / "capture-manifest.json")
    dataset_manifest = _load_object(capture_root / "dataset" / "dataset-manifest.json")
    capture_hash = _string_field(capture_manifest, "capture_root_hash")
    dataset_hash = _string_field(dataset_manifest, "root_hash")
    bars = load_research_bars(capture_root / "dataset")
    primary = tuple(item for item in bars if item.venue == "OKX")
    reference = tuple(item for item in bars if item.venue == "BINANCE")
    bar_samples = build_bar_samples(primary, reference, horizon_bars=horizon_bars)
    split_samples = [
        SplitSample(item.sample_id, item.decision_time_ns, item.label_available_time_ns)
        for item in bar_samples
    ]
    views = build_walk_forward_views(split_samples, config)
    if not views.folds:
        raise WalkForwardRunError("capture does not contain a complete walk-forward fold")
    material: dict[str, object] = {
        "manifest_version": "1.0.0",
        "capture_root_hash": capture_hash,
        "dataset_root_hash": dataset_hash,
        "horizon_bars": horizon_bars,
        "split_manifest_hash": views.manifest_hash,
        "config": _config_record(config),
        "folds": [_fold_record(fold) for fold in views.folds],
        "final_holdout_ids": list(views.final_holdout_ids),
    }
    manifest_hash = content_sha256(material)
    document = dict(material)
    document["manifest_hash"] = manifest_hash
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_bytes(canonical_json(document))
    temporary.replace(output_path)
    return WalkForwardArtifact(
        output_path=output_path,
        capture_root_hash=capture_hash,
        dataset_root_hash=dataset_hash,
        manifest_hash=manifest_hash,
        fold_count=len(views.folds),
    )


def verify_walk_forward_manifest(path: Path) -> bool:
    try:
        document = _load_object(path)
        recorded_hash = _string_field(document, "manifest_hash")
        material = dict(document)
        del material["manifest_hash"]
        return content_sha256(material) == recorded_hash
    except (OSError, json.JSONDecodeError, WalkForwardRunError, KeyError, TypeError, ValueError):
        return False


def _config_record(config: WalkForwardConfig) -> dict[str, int]:
    return {
        "train_duration_ns": config.train_duration_ns,
        "validation_duration_ns": config.validation_duration_ns,
        "test_duration_ns": config.test_duration_ns,
        "step_ns": config.step_ns,
        "embargo_ns": config.embargo_ns,
        "final_holdout_start_ns": config.final_holdout_start_ns,
    }


def _fold_record(fold: WalkForwardFold) -> dict[str, object]:
    return {
        "fold_index": fold.fold_index,
        "train_start_ns": fold.train_start_ns,
        "train_end_ns": fold.train_end_ns,
        "validation_start_ns": fold.validation_start_ns,
        "validation_end_ns": fold.validation_end_ns,
        "test_start_ns": fold.test_start_ns,
        "test_end_ns": fold.test_end_ns,
        "train_ids": list(fold.train_ids),
        "validation_ids": list(fold.validation_ids),
        "test_ids": list(fold.test_ids),
    }


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise WalkForwardRunError(f"{path.name} must be a JSON object")
    return value


def _string_field(record: dict[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str):
        raise WalkForwardRunError(f"manifest field {key} must be a string")
    return value
