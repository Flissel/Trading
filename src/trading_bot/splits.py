"""Immutable leakage-safe walk-forward experiment views."""

from dataclasses import dataclass
from itertools import pairwise

from trading_bot.canonical import content_sha256


@dataclass(frozen=True, slots=True)
class SplitSample:
    sample_id: str
    decision_time_ns: int
    label_end_time_ns: int

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id must not be empty")
        if self.decision_time_ns < 0 or self.label_end_time_ns <= self.decision_time_ns:
            raise ValueError("sample interval is invalid")


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    train_duration_ns: int
    validation_duration_ns: int
    test_duration_ns: int
    step_ns: int
    embargo_ns: int
    final_holdout_start_ns: int

    def __post_init__(self) -> None:
        durations = (
            self.train_duration_ns,
            self.validation_duration_ns,
            self.test_duration_ns,
            self.step_ns,
        )
        if any(value <= 0 for value in durations):
            raise ValueError("walk-forward durations and step must be positive")
        if self.embargo_ns < 0 or self.final_holdout_start_ns <= 0:
            raise ValueError("embargo and holdout boundary are invalid")


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    fold_index: int
    train_start_ns: int
    train_end_ns: int
    validation_start_ns: int
    validation_end_ns: int
    test_start_ns: int
    test_end_ns: int
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    test_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WalkForwardViews:
    folds: tuple[WalkForwardFold, ...]
    final_holdout_ids: tuple[str, ...]
    manifest_hash: str


def build_walk_forward_views(
    samples: list[SplitSample], config: WalkForwardConfig
) -> WalkForwardViews:
    _validate_samples(samples)
    holdout_ids = tuple(
        sample.sample_id
        for sample in samples
        if sample.decision_time_ns >= config.final_holdout_start_ns
    )
    folds: list[WalkForwardFold] = []
    if samples:
        anchor = samples[0].decision_time_ns
        fold_index = 0
        while True:
            train_start = anchor + fold_index * config.step_ns
            train_end = train_start + config.train_duration_ns
            validation_start = train_end + config.embargo_ns
            validation_end = validation_start + config.validation_duration_ns
            test_start = validation_end + config.embargo_ns
            test_end = test_start + config.test_duration_ns
            if test_end > config.final_holdout_start_ns:
                break
            folds.append(
                WalkForwardFold(
                    fold_index=fold_index,
                    train_start_ns=train_start,
                    train_end_ns=train_end,
                    validation_start_ns=validation_start,
                    validation_end_ns=validation_end,
                    test_start_ns=test_start,
                    test_end_ns=test_end,
                    train_ids=_eligible_ids(samples, train_start, train_end),
                    validation_ids=_eligible_ids(samples, validation_start, validation_end),
                    test_ids=_eligible_ids(samples, test_start, test_end),
                )
            )
            fold_index += 1

    manifest = {
        "config": {
            "train_duration_ns": config.train_duration_ns,
            "validation_duration_ns": config.validation_duration_ns,
            "test_duration_ns": config.test_duration_ns,
            "step_ns": config.step_ns,
            "embargo_ns": config.embargo_ns,
            "final_holdout_start_ns": config.final_holdout_start_ns,
        },
        "folds": [_fold_record(fold) for fold in folds],
        "final_holdout_ids": list(holdout_ids),
    }
    return WalkForwardViews(tuple(folds), holdout_ids, content_sha256(manifest))


def _eligible_ids(samples: list[SplitSample], start_ns: int, end_ns: int) -> tuple[str, ...]:
    return tuple(
        sample.sample_id
        for sample in samples
        if start_ns <= sample.decision_time_ns < end_ns and sample.label_end_time_ns < end_ns
    )


def _validate_samples(samples: list[SplitSample]) -> None:
    ids = [sample.sample_id for sample in samples]
    if len(set(ids)) != len(ids):
        raise ValueError("sample IDs must be unique")
    for previous, current in pairwise(samples):
        if current.decision_time_ns <= previous.decision_time_ns:
            raise ValueError("samples must be strictly chronological")


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
