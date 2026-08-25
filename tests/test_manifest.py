import hashlib
from pathlib import Path
from uuid import UUID

import pytest

from trading_bot.canonical import canonical_json
from trading_bot.manifest import DatasetManifest, ManifestPublicationError, ManifestPublisher
from trading_bot.registry import MetadataRegistry

DATASET_ID = UUID("00000000-0000-0000-0000-000000000201")


def make_manifest(**overrides: object) -> DatasetManifest:
    values: dict[str, object] = {
        "dataset_id": DATASET_ID,
        "dataset_name": "okx_trades_sample",
        "dataset_version": "1.0.0",
        "schema_hash": "a" * 64,
        "content_hash": "b" * 64,
        "normalizer_version": "0.1.0",
        "parent_manifest_ids": (),
        "row_count": 1,
        "min_available_time_ns": 1_700_000_000_000_000_000,
        "max_available_time_ns": 1_700_000_000_000_000_000,
        "quality_report_hash": "c" * 64,
        "license_classes": ("okx_public_market_data",),
        "created_at_ns": 1_800_000_000_000_000_000,
    }
    values.update(overrides)
    return DatasetManifest.model_validate(values)


def test_publishes_canonical_manifest_and_registers_real_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    database_path = workspace / "metadata.sqlite3"
    relative_path = "artifacts/manifests/okx-sample.json"
    manifest = make_manifest()
    publisher = ManifestPublisher(workspace_root=workspace)

    with MetadataRegistry(database_path) as registry:
        artifact = publisher.publish(
            manifest=manifest,
            relative_path=relative_path,
            registry=registry,
        )
        recovered = registry.get_artifact(DATASET_ID)

    written = (workspace / relative_path).read_bytes()
    assert written == canonical_json(manifest.model_dump(mode="json"))
    assert artifact.content_hash == hashlib.sha256(written).hexdigest()
    assert recovered == artifact


def test_rejects_manifest_path_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    publisher = ManifestPublisher(workspace_root=workspace)

    with (
        MetadataRegistry(workspace / "metadata.sqlite3") as registry,
        pytest.raises(ManifestPublicationError, match="outside workspace"),
    ):
        publisher.publish(
            manifest=make_manifest(),
            relative_path="../outside.json",
            registry=registry,
        )

    assert not (tmp_path / "outside.json").exists()


def test_refuses_to_overwrite_published_manifest_with_different_content(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    relative_path = "artifacts/manifests/okx-sample.json"
    publisher = ManifestPublisher(workspace_root=workspace)
    original = make_manifest()
    changed = make_manifest(row_count=2)

    with MetadataRegistry(workspace / "metadata.sqlite3") as registry:
        publisher.publish(manifest=original, relative_path=relative_path, registry=registry)
        with pytest.raises(ManifestPublicationError, match="immutable"):
            publisher.publish(manifest=changed, relative_path=relative_path, registry=registry)

    assert (workspace / relative_path).read_bytes() == canonical_json(
        original.model_dump(mode="json")
    )
