"""Immutable dataset manifest publication."""

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, StringConstraints

from trading_bot.canonical import canonical_json
from trading_bot.registry import ArtifactRecord, MetadataRegistry

Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class ManifestPublicationError(RuntimeError):
    """Raised when a manifest cannot be published immutably."""


class DatasetManifest(BaseModel):
    """Lineage and quality identity for one derived dataset."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    dataset_id: UUID
    dataset_name: str
    dataset_version: str
    schema_hash: Sha256Hex
    content_hash: Sha256Hex
    normalizer_version: str
    parent_manifest_ids: tuple[UUID, ...]
    row_count: int
    min_available_time_ns: int
    max_available_time_ns: int
    quality_report_hash: Sha256Hex
    license_classes: tuple[str, ...]
    created_at_ns: int


@dataclass(frozen=True, slots=True)
class ManifestPublisher:
    """Publish canonical manifests inside an approved workspace."""

    workspace_root: Path

    def publish(
        self,
        *,
        manifest: DatasetManifest,
        relative_path: str,
        registry: MetadataRegistry,
    ) -> ArtifactRecord:
        workspace_root = self.workspace_root.resolve(strict=False)
        requested_path = Path(relative_path)
        target = (workspace_root / requested_path).resolve(strict=False)
        if requested_path.is_absolute() or not target.is_relative_to(workspace_root):
            raise ManifestPublicationError("manifest path is outside workspace")

        encoded = canonical_json(manifest.model_dump(mode="json"))
        artifact = ArtifactRecord(
            artifact_id=manifest.dataset_id,
            kind="dataset_manifest",
            relative_path=relative_path,
            content_hash=hashlib.sha256(encoded).hexdigest(),
            created_at_ns=manifest.created_at_ns,
        )

        registered = registry.get_artifact(manifest.dataset_id)
        if registered is not None:
            if registered != artifact:
                raise ManifestPublicationError("published manifest identity is immutable")
            if not target.exists() or target.read_bytes() != encoded:
                raise ManifestPublicationError("published manifest file is missing or changed")
            return registered

        if target.exists():
            if target.read_bytes() != encoded:
                raise ManifestPublicationError("published manifest file is immutable")
            registry.register_artifact(artifact)
            return artifact

        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary.write(encoded)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_name = temporary.name
            os.rename(temporary_name, target)
            temporary_name = None
        except OSError as error:
            raise ManifestPublicationError("manifest could not be published atomically") from error
        finally:
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)

        registry.register_artifact(artifact)
        return artifact
