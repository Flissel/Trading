from pathlib import Path

import pytest

from trading_bot.storage import StoragePolicy, StoragePolicyError

WORKSPACE_ROOT = Path("C:/Users/User/Documents/ChatGPT/Trading")


def test_rejects_target_on_excluded_drive() -> None:
    policy = StoragePolicy(workspace_root=WORKSPACE_ROOT, reserve_bytes=1_000)

    with pytest.raises(StoragePolicyError, match="excluded drive"):
        policy.authorize(
            target=Path("E:/trading/data/sample.parquet"),
            free_bytes=10_000,
            worst_case_required_bytes=2_000,
        )


def test_rejects_drive_root_as_target() -> None:
    policy = StoragePolicy(workspace_root=WORKSPACE_ROOT, reserve_bytes=1_000)

    with pytest.raises(StoragePolicyError, match="drive root"):
        policy.authorize(
            target=Path("C:/"),
            free_bytes=10_000,
            worst_case_required_bytes=2_000,
        )


def test_rejects_job_that_would_cross_reserve() -> None:
    policy = StoragePolicy(workspace_root=WORKSPACE_ROOT, reserve_bytes=2_000)

    with pytest.raises(StoragePolicyError, match="reserve"):
        policy.authorize(
            target=WORKSPACE_ROOT / "data" / "sample.parquet",
            free_bytes=5_000,
            worst_case_required_bytes=3_001,
        )


def test_authorizes_bounded_target_with_sufficient_capacity() -> None:
    target = WORKSPACE_ROOT / "data" / "sample.parquet"
    policy = StoragePolicy(workspace_root=WORKSPACE_ROOT, reserve_bytes=2_000)

    authorized = policy.authorize(
        target=target,
        free_bytes=5_000,
        worst_case_required_bytes=3_000,
    )

    assert authorized == target


def test_rejects_target_outside_workspace() -> None:
    policy = StoragePolicy(workspace_root=WORKSPACE_ROOT, reserve_bytes=1_000)

    with pytest.raises(StoragePolicyError, match="outside workspace"):
        policy.authorize(
            target=Path("C:/Users/User/Downloads/sample.parquet"),
            free_bytes=10_000,
            worst_case_required_bytes=2_000,
        )


def test_rejects_temporary_directory_on_excluded_drive() -> None:
    policy = StoragePolicy(workspace_root=WORKSPACE_ROOT, reserve_bytes=1_000)

    with pytest.raises(StoragePolicyError, match="temporary directory"):
        policy.authorize(
            target=WORKSPACE_ROOT / "data" / "sample.parquet",
            temporary_directory=Path("E:/temp"),
            free_bytes=10_000,
            worst_case_required_bytes=2_000,
        )


def test_rejects_negative_reserve() -> None:
    with pytest.raises(StoragePolicyError, match="non-negative"):
        StoragePolicy(workspace_root=WORKSPACE_ROOT, reserve_bytes=-1)
