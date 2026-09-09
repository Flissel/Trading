"""SQLite-backed metadata and artifact registry."""

import sqlite3
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import TracebackType
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, StringConstraints, field_validator, model_validator

Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class RegistryConflictError(RuntimeError):
    """Raised when immutable registry identity conflicts with stored data."""


class ArtifactRecord(BaseModel):
    """Immutable pointer to a content-addressed project artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_id: UUID
    kind: str
    relative_path: str
    content_hash: Sha256Hex
    created_at_ns: int

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        posix_path = PurePosixPath(value)
        windows_path = PureWindowsPath(value)
        if posix_path.is_absolute() or windows_path.is_absolute() or ".." in windows_path.parts:
            raise ValueError("relative_path must stay inside the registry root")
        return value


class ExperimentRecord(BaseModel):
    """Append-only record for completed and failed research trials."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: UUID
    family_id: UUID
    candidate_name: str
    hypothesis: str
    split_manifest_hash: Sha256Hex
    code_hash: Sha256Hex
    random_seed: int
    outcome: Literal["completed", "failed"]
    result_hash: Sha256Hex | None
    failure_reason: str | None
    created_at_ns: int

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if self.outcome == "completed" and (
            self.result_hash is None or self.failure_reason is not None
        ):
            raise ValueError("completed experiment requires only a result hash")
        if self.outcome == "failed" and (self.result_hash is not None or not self.failure_reason):
            raise ValueError("failed experiment requires only a failure reason")
        return self


class MetadataRegistry:
    """Persist and recover local research metadata."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self._connection: sqlite3.Connection | None = None

    def __enter__(self) -> Self:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.database_path)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                created_at_ns INTEGER NOT NULL
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                family_id TEXT NOT NULL,
                candidate_name TEXT NOT NULL,
                hypothesis TEXT NOT NULL,
                split_manifest_hash TEXT NOT NULL,
                code_hash TEXT NOT NULL,
                random_seed INTEGER NOT NULL,
                outcome TEXT NOT NULL,
                result_hash TEXT,
                failure_reason TEXT,
                created_at_ns INTEGER NOT NULL
            )
            """
        )
        self._connection.commit()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def register_artifact(self, artifact: ArtifactRecord) -> None:
        connection = self._require_connection()
        existing = self.get_artifact(artifact.artifact_id)
        if existing is not None:
            if existing != artifact:
                raise RegistryConflictError(
                    f"artifact {artifact.artifact_id} is already registered with different content"
                )
            return

        with connection:
            connection.execute(
                """
                INSERT INTO artifacts (
                    artifact_id, kind, relative_path, content_hash, created_at_ns
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(artifact.artifact_id),
                    artifact.kind,
                    artifact.relative_path,
                    artifact.content_hash,
                    artifact.created_at_ns,
                ),
            )

    def get_artifact(self, artifact_id: UUID) -> ArtifactRecord | None:
        connection = self._require_connection()
        row = connection.execute(
            """
            SELECT artifact_id, kind, relative_path, content_hash, created_at_ns
            FROM artifacts
            WHERE artifact_id = ?
            """,
            (str(artifact_id),),
        ).fetchone()
        if row is None:
            return None
        return ArtifactRecord.model_validate(
            {
                "artifact_id": row[0],
                "kind": row[1],
                "relative_path": row[2],
                "content_hash": row[3],
                "created_at_ns": row[4],
            }
        )

    def register_experiment(self, experiment: ExperimentRecord) -> None:
        connection = self._require_connection()
        existing = self.get_experiment(experiment.experiment_id)
        if existing is not None:
            if existing != experiment:
                raise RegistryConflictError(
                    f"experiment {experiment.experiment_id} is already registered differently"
                )
            return
        with connection:
            connection.execute(
                """
                INSERT INTO experiments (
                    experiment_id, family_id, candidate_name, hypothesis,
                    split_manifest_hash, code_hash, random_seed, outcome,
                    result_hash, failure_reason, created_at_ns
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(experiment.experiment_id),
                    str(experiment.family_id),
                    experiment.candidate_name,
                    experiment.hypothesis,
                    experiment.split_manifest_hash,
                    experiment.code_hash,
                    experiment.random_seed,
                    experiment.outcome,
                    experiment.result_hash,
                    experiment.failure_reason,
                    experiment.created_at_ns,
                ),
            )

    def get_experiment(self, experiment_id: UUID) -> ExperimentRecord | None:
        row = (
            self._require_connection()
            .execute(
                """
            SELECT experiment_id, family_id, candidate_name, hypothesis,
                   split_manifest_hash, code_hash, random_seed, outcome,
                   result_hash, failure_reason, created_at_ns
            FROM experiments WHERE experiment_id = ?
            """,
                (str(experiment_id),),
            )
            .fetchone()
        )
        return None if row is None else _experiment_from_row(row)

    def list_experiments(self, family_id: UUID | None = None) -> tuple[ExperimentRecord, ...]:
        connection = self._require_connection()
        if family_id is None:
            rows = connection.execute(
                """
                SELECT experiment_id, family_id, candidate_name, hypothesis,
                       split_manifest_hash, code_hash, random_seed, outcome,
                       result_hash, failure_reason, created_at_ns
                FROM experiments ORDER BY candidate_name, experiment_id
                """
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT experiment_id, family_id, candidate_name, hypothesis,
                       split_manifest_hash, code_hash, random_seed, outcome,
                       result_hash, failure_reason, created_at_ns
                FROM experiments WHERE family_id = ?
                ORDER BY created_at_ns, experiment_id
                """,
                (str(family_id),),
            ).fetchall()
        return tuple(_experiment_from_row(row) for row in rows)

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("metadata registry is not open")
        return self._connection


def _experiment_from_row(row: tuple[object, ...]) -> ExperimentRecord:
    if len(row) != 11:
        raise RuntimeError("experiment registry row shape is invalid")
    return ExperimentRecord.model_validate(
        {
            "experiment_id": row[0],
            "family_id": row[1],
            "candidate_name": row[2],
            "hypothesis": row[3],
            "split_manifest_hash": row[4],
            "code_hash": row[5],
            "random_seed": row[6],
            "outcome": row[7],
            "result_hash": row[8],
            "failure_reason": row[9],
            "created_at_ns": row[10],
        }
    )
