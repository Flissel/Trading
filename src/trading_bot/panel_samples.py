"""Weekly rebalance calendar and walk-forward manifest for the panel."""

import json
from dataclasses import dataclass
from pathlib import Path

from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.panel_capture import verify_panel_capture
from trading_bot.panel_config import PanelFamilySpec, PanelFoldGeometry
from trading_bot.panel_reader import PanelBar, load_panel_bars
from trading_bot.splits import (
    SplitSample,
    WalkForwardConfig,
    WalkForwardFold,
    build_walk_forward_views,
)

DAY_NS = 86_400_000_000_000
_SUNDAY_REMAINDER = 3  # the Unix epoch began on a Thursday

# Operational data-quality threshold, not part of the frozen family
# declaration (`configs/xs-momentum-panel-v1.json`): that file is a
# pre-registration whose hash travels in every report, and a capture-quality
# gate is not a statement about the experiment family. Spec section 12 step 3.
_MAX_MISSING_DAYS_PER_INSTRUMENT = 3
# Cap on how many offending instruments are named in the failure message --
# enough to be actionable, not so many the message becomes unreadable.
_MAX_QUALITY_FAILURE_NAMES = 20


class PanelSamplesError(RuntimeError):
    """Raised when the panel rebalance calendar cannot be published."""


@dataclass(frozen=True, slots=True)
class PanelManifestArtifact:
    output_path: Path
    capture_root_hash: str
    dataset_root_hash: str
    manifest_hash: str
    fold_count: int
    pooled_test_sample_count: int


def rebalance_close_times(bars: tuple[PanelBar, ...]) -> tuple[int, ...]:
    """Return the sorted close times of Sunday bars observed anywhere in the panel."""
    times = {
        bar.close_time_ns
        for bar in bars
        if (bar.open_time_ns // DAY_NS) % 7 == _SUNDAY_REMAINDER
    }
    return tuple(sorted(times))


def build_rebalance_samples(
    bars: tuple[PanelBar, ...], *, holding_days: int
) -> tuple[SplitSample, ...]:
    if holding_days < 1:
        raise PanelSamplesError("holding period must be positive")
    return tuple(
        SplitSample(
            sample_id=f"BINANCE_UM:{close_time_ns}:w1",
            decision_time_ns=close_time_ns,
            label_end_time_ns=close_time_ns + holding_days * DAY_NS,
        )
        for close_time_ns in rebalance_close_times(bars)
    )


def derive_panel_config(
    samples: tuple[SplitSample, ...], *, folds: PanelFoldGeometry
) -> WalkForwardConfig:
    if len(samples) < 2:
        raise PanelSamplesError("panel capture cannot define a rebalance calendar")
    spacing = samples[-1].decision_time_ns - samples[-2].decision_time_ns
    if folds.holdout_duration_ns < spacing:
        raise PanelSamplesError("holdout duration is shorter than one rebalance interval")
    holdout_start = samples[-1].decision_time_ns - folds.holdout_duration_ns + spacing
    if holdout_start <= samples[0].decision_time_ns:
        raise PanelSamplesError("panel capture is too short for the declared geometry")
    return WalkForwardConfig(
        train_duration_ns=folds.train_duration_ns,
        validation_duration_ns=folds.validation_duration_ns,
        test_duration_ns=folds.test_duration_ns,
        step_ns=folds.step_ns,
        embargo_ns=folds.embargo_ns,
        final_holdout_start_ns=holdout_start,
    )


def publish_panel_walk_forward(
    capture_root: Path,
    *,
    output_path: Path,
    spec: PanelFamilySpec,
    family_spec_hash: str,
) -> PanelManifestArtifact:
    valid, errors = verify_panel_capture(capture_root)
    if not valid:
        raise PanelSamplesError("panel capture verification failed: " + ",".join(errors))
    if output_path.exists():
        raise PanelSamplesError("panel manifest already exists and is immutable")
    capture_manifest = _load_object(capture_root / "capture-manifest.json")
    dataset_manifest = _load_object(capture_root / "dataset" / "dataset-manifest.json")
    capture_hash = str(capture_manifest["capture_root_hash"])
    dataset_hash = str(dataset_manifest["root_hash"])
    _enforce_capture_quality(capture_root / "dataset" / "quality-report.json")

    bars = load_panel_bars(capture_root / "dataset")
    samples = build_rebalance_samples(bars, holding_days=spec.holding_days)
    config = derive_panel_config(samples, folds=spec.folds)
    views = build_walk_forward_views(list(samples), config)
    if not views.folds:
        raise PanelSamplesError("panel capture does not contain a complete fold")
    pooled = sum(len(fold.test_ids) for fold in views.folds)
    if pooled < spec.statistics.pooled_episode_floor:
        raise PanelSamplesError(
            f"INSUFFICIENT_EVIDENCE: pooled test samples {pooled} below "
            f"{spec.statistics.pooled_episode_floor}"
        )

    material: dict[str, object] = {
        "manifest_version": "1.0.0",
        "family_name": spec.family_name,
        "family_spec_hash": family_spec_hash,
        "capture_root_hash": capture_hash,
        "dataset_root_hash": dataset_hash,
        "holding_days": spec.holding_days,
        "split_manifest_hash": views.manifest_hash,
        "config": {
            "train_duration_ns": config.train_duration_ns,
            "validation_duration_ns": config.validation_duration_ns,
            "test_duration_ns": config.test_duration_ns,
            "step_ns": config.step_ns,
            "embargo_ns": config.embargo_ns,
            "final_holdout_start_ns": config.final_holdout_start_ns,
        },
        "folds": [_fold_record(fold) for fold in views.folds],
        "final_holdout_ids": list(views.final_holdout_ids),
        "pooled_test_sample_count": pooled,
    }
    manifest_hash = content_sha256(material)
    document = dict(material)
    document["manifest_hash"] = manifest_hash
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_bytes(canonical_json(document))
    temporary.replace(output_path)
    return PanelManifestArtifact(
        output_path=output_path,
        capture_root_hash=capture_hash,
        dataset_root_hash=dataset_hash,
        manifest_hash=manifest_hash,
        fold_count=len(views.folds),
        pooled_test_sample_count=pooled,
    )


def verify_panel_manifest(path: Path) -> bool:
    try:
        document = _load_object(path)
        recorded = str(document["manifest_hash"])
        material = {key: value for key, value in document.items() if key != "manifest_hash"}
        return content_sha256(material) == recorded
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False


def _enforce_capture_quality(quality_report_path: Path) -> None:
    """Stop the family before a manifest is built on a defective capture.

    Spec section 12 step 3: the family stops if any contract's daily series
    has more than `_MAX_MISSING_DAYS_PER_INSTRUMENT` missing days inside its
    listed span. `panel_dataset.py` already computes `missing_days` per
    instrument span-relative (a delisting does not count, only an interior
    gap does); this is the first and only reader of that field.
    """
    quality = _load_object(quality_report_path)
    instruments = quality.get("instruments")
    if not isinstance(instruments, list):
        raise PanelSamplesError("quality-report.json is malformed")
    offenders: list[tuple[str, int]] = []
    for item in instruments:
        if not isinstance(item, dict):
            raise PanelSamplesError("quality-report.json is malformed")
        missing_days = item.get("missing_days")
        instrument_id = item.get("instrument_id")
        if not isinstance(missing_days, int) or not isinstance(instrument_id, str):
            raise PanelSamplesError("quality-report.json is malformed")
        if missing_days > _MAX_MISSING_DAYS_PER_INSTRUMENT:
            offenders.append((instrument_id, missing_days))
    if not offenders:
        return
    offenders.sort(key=lambda pair: (-pair[1], pair[0]))
    shown = offenders[:_MAX_QUALITY_FAILURE_NAMES]
    detail = ", ".join(f"{name} ({count} missing days)" for name, count in shown)
    omitted = len(offenders) - len(shown)
    if omitted:
        detail += f", and {omitted} more"
    raise PanelSamplesError(
        "CAPTURE_QUALITY_FAILED: "
        f"{len(offenders)} instrument(s) exceed {_MAX_MISSING_DAYS_PER_INSTRUMENT} missing "
        f"days inside their listed span: {detail}"
    )


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
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise PanelSamplesError(f"{path.name} must contain a JSON object")
    return document
