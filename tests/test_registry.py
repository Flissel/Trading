from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from trading_bot.registry import (
    ArtifactRecord,
    ExperimentRecord,
    MetadataRegistry,
    RegistryConflictError,
)

ARTIFACT_ID = UUID("00000000-0000-0000-0000-000000000101")
CONTENT_HASH = "b" * 64
EXPERIMENT_ID = UUID("00000000-0000-0000-0000-000000000201")
FAMILY_ID = UUID("00000000-0000-0000-0000-000000000202")


def make_artifact(**overrides: object) -> ArtifactRecord:
    values: dict[str, object] = {
        "artifact_id": ARTIFACT_ID,
        "kind": "dataset_manifest",
        "relative_path": "artifacts/manifests/sample.json",
        "content_hash": CONTENT_HASH,
        "created_at_ns": 1_000,
    }
    values.update(overrides)
    return ArtifactRecord.model_validate(values)


def test_registry_recovers_registered_artifact_after_reopen(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    artifact = make_artifact()

    with MetadataRegistry(database_path) as registry:
        registry.register_artifact(artifact)

    with MetadataRegistry(database_path) as reopened:
        recovered = reopened.get_artifact(ARTIFACT_ID)

    assert recovered == artifact


def test_registry_does_not_overwrite_conflicting_artifact(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    original = make_artifact()
    conflicting = make_artifact(content_hash="c" * 64)

    with MetadataRegistry(database_path) as registry:
        registry.register_artifact(original)
        with pytest.raises(RegistryConflictError, match="already registered"):
            registry.register_artifact(conflicting)

        assert registry.get_artifact(ARTIFACT_ID) == original


@pytest.mark.parametrize(
    "unsafe_path",
    ["C:/outside/file.json", "../outside/file.json"],
)
def test_artifact_rejects_path_outside_registry_root(unsafe_path: str) -> None:
    with pytest.raises(ValidationError, match="relative_path"):
        make_artifact(relative_path=unsafe_path)


def make_experiment(**overrides: object) -> ExperimentRecord:
    values: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "family_id": FAMILY_ID,
        "candidate_name": "momentum",
        "hypothesis": "recent signed return persists for one bar",
        "split_manifest_hash": "c" * 64,
        "code_hash": "d" * 64,
        "random_seed": 17,
        "outcome": "completed",
        "result_hash": "e" * 64,
        "failure_reason": None,
        "created_at_ns": 2_000,
    }
    values.update(overrides)
    return ExperimentRecord.model_validate(values)


def test_registry_recovers_completed_and_failed_experiments_after_reopen(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    completed = make_experiment()
    failed = make_experiment(
        experiment_id="00000000-0000-0000-0000-000000000203",
        candidate_name="broken-candidate",
        outcome="failed",
        result_hash=None,
        failure_reason="SCHEMA_REJECTED",
        created_at_ns=2_001,
    )

    with MetadataRegistry(database_path) as registry:
        registry.register_experiment(completed)
        registry.register_experiment(failed)

    with MetadataRegistry(database_path) as reopened:
        assert reopened.get_experiment(EXPERIMENT_ID) == completed
        assert reopened.list_experiments(FAMILY_ID) == (completed, failed)


def test_registry_never_overwrites_experiment_identity(tmp_path: Path) -> None:
    with MetadataRegistry(tmp_path / "metadata.sqlite3") as registry:
        registry.register_experiment(make_experiment())

        with pytest.raises(RegistryConflictError, match="experiment"):
            registry.register_experiment(make_experiment(random_seed=99))


def test_experiment_outcome_requires_matching_result_or_failure() -> None:
    with pytest.raises(ValidationError, match="completed experiment"):
        make_experiment(result_hash=None)
    with pytest.raises(ValidationError, match="failed experiment"):
        make_experiment(outcome="failed", failure_reason=None)
