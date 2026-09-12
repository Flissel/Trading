"""Storage authorization boundary."""

from dataclasses import dataclass
from pathlib import Path


class StoragePolicyError(ValueError):
    """Raised when a storage operation is not authorized."""


class StorageReserveError(StoragePolicyError):
    """Raised when only the free-space reserve stands in the job's way.

    Every other refusal - an excluded drive, a target outside the workspace, a
    temporary directory on another volume - is a fact about the request that
    running it again cannot change. A crossed reserve is a fact about the disk
    at this moment: deleting something, or a job elsewhere finishing, makes the
    same request authorizable. Callers that map failures onto exit codes stop
    on the first kind and retry the second.
    """


@dataclass(frozen=True, slots=True)
class StoragePolicy:
    """Authorize bounded writes to approved local storage."""

    workspace_root: Path
    reserve_bytes: int
    excluded_drives: frozenset[str] = frozenset({"E:"})

    def __post_init__(self) -> None:
        if self.reserve_bytes < 0:
            raise StoragePolicyError("storage reserve must be non-negative")

    def authorize(
        self,
        *,
        target: Path,
        temporary_directory: Path | None = None,
        free_bytes: int,
        worst_case_required_bytes: int,
    ) -> Path:
        drive = target.drive.upper()
        excluded_drives = {item.upper() for item in self.excluded_drives}
        if drive in excluded_drives:
            raise StoragePolicyError(f"target uses excluded drive: {drive}")

        if target.anchor and target == Path(target.anchor):
            raise StoragePolicyError("target must not be a drive root")

        workspace_root = self.workspace_root.resolve(strict=False)
        resolved_target = target.resolve(strict=False)
        if not resolved_target.is_relative_to(workspace_root):
            raise StoragePolicyError("target is outside workspace")

        if temporary_directory is not None:
            temporary_drive = temporary_directory.drive.upper()
            if temporary_drive in excluded_drives or temporary_drive != drive:
                raise StoragePolicyError("temporary directory must use the authorized target drive")

        if free_bytes - worst_case_required_bytes < self.reserve_bytes:
            raise StorageReserveError("job would cross the configured storage reserve")

        return target
