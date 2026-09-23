"""The frozen weekly shadow declaration (spec sections 2 and 4.1).

A shadow week is a pure function of this declaration, the week's captures and
the Sunday it is asked about: the family it shadows and the hash that family
must have, the one release candidate whose book is recorded, the two
dominance controls the run is evaluated beside, the fixed anchor Sunday every
week's run starts from, and the phase that decides whether a week may carry
P&L at all. Nothing here computes anything -- the model and its loader are
kept apart from `shadow_book` so that reading a declaration cannot pull in the
decision loop, and so a declaration can be checked by a person, or by the CLI,
without running a week.

The anchor is a date rather than a nanosecond stamp because that is what the
spec fixes (2026-09-13, the first Sunday after the holdout window's last
exit), and because a date survives being read by a human. `family_name` is
declared beside `family_spec_hash` (ruling 15): the hash is the real gate, and
the name is what a reader -- and the fixture suite, which shadows the
unmeasured v4 family -- can check the loaded spec against without holding the
spec's bytes.
"""

import json
import re
from datetime import date
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from trading_bot.canonical import content_sha256
from trading_bot.carry_config import CONTROL_NAMES, MEMBER_NAMES_BY_FAMILY

_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")
_SUNDAY = 6  # `date.weekday()` counts from Monday


class ShadowDeclarationError(ValueError):
    """Raised when a shadow declaration cannot be read as one."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ShadowDeclaration(_Frozen):
    """What one family's weekly shadow book is declared to be."""

    version: Literal["1.0.0"]
    family_spec_path: str
    family_spec_hash: str
    family_name: str
    candidate: str
    controls: tuple[str, str] = ("no_trade", "random_pairs")
    anchor_decision_close_date: str
    phase: Literal["A", "B"]
    holdout_report_hash: str | None
    artifact_root: str
    registry_path: str

    @field_validator("family_spec_hash")
    @classmethod
    def validate_family_spec_hash(cls, value: str) -> str:
        if not _HEX64.match(value):
            raise ValueError("a declared hash is 64 lower-case hexadecimal characters")
        return value

    @field_validator("holdout_report_hash")
    @classmethod
    def validate_holdout_report_hash(cls, value: str | None) -> str | None:
        if value is not None and not _HEX64.match(value):
            raise ValueError("a declared hash is 64 lower-case hexadecimal characters")
        return value

    @field_validator("family_name")
    @classmethod
    def validate_family_name(cls, value: str) -> str:
        """The shadowed family is one the carry declarations know.

        A name no family carries could never match a loaded spec, so it is a
        typo rather than a declaration and is refused where it is written
        instead of one capture verification later.
        """
        if value not in MEMBER_NAMES_BY_FAMILY:
            raise ValueError(f"{value} is not a declared carry family")
        return value

    @field_validator("candidate")
    @classmethod
    def validate_candidate(cls, value: str) -> str:
        if not value:
            raise ValueError("the candidate is named")
        return value

    @field_validator("controls")
    @classmethod
    def validate_controls(cls, value: tuple[str, str]) -> tuple[str, str]:
        """Spec 4.1 runs the candidate beside the two dominance controls, which
        are the family's own declared controls and nothing invented here."""
        for name in value:
            if name not in CONTROL_NAMES:
                raise ValueError(f"{name} is not a declared control")
        if value[0] == value[1]:
            raise ValueError("the two controls are two different controls")
        return value

    @field_validator("family_spec_path", "artifact_root", "registry_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        """Every path a declaration names is relative to the workspace root.

        An absolute path, or one climbing out with `..`, would let a
        declaration reach outside the workspace the storage policy bounds --
        so it is refused here rather than at the first write.
        """
        if not value:
            raise ValueError("a declared path is not empty")
        posix, windows = PurePosixPath(value), PureWindowsPath(value)
        if posix.is_absolute() or windows.is_absolute() or ".." in windows.parts:
            raise ValueError("a declared path stays inside the workspace root")
        return value

    @field_validator("anchor_decision_close_date")
    @classmethod
    def validate_anchor(cls, value: str) -> str:
        """Spec 4.1's anchor is a Sunday, because every decision is one."""
        anchor = _parse_date(value)
        if anchor.weekday() != _SUNDAY:
            raise ValueError(f"the anchor {value} is not a Sunday")
        return value

    @model_validator(mode="after")
    def validate_phase(self) -> "ShadowDeclaration":
        """Spec 2: Phase B is what a confirmed holdout authorises.

        A Phase B declaration names the holdout report it stands on, by hash;
        a Phase A declaration names none, because Phase A runs while the
        holdout is still unread and a hash it could not have seen would be a
        claim about a document nobody has opened.
        """
        if self.phase == "B" and self.holdout_report_hash is None:
            raise ValueError("a Phase B declaration names the holdout report it stands on")
        if self.phase == "A" and self.holdout_report_hash is not None:
            raise ValueError("a Phase A declaration names no holdout report")
        return self


def load_shadow_declaration(path: Path) -> tuple[ShadowDeclaration, str]:
    """Load the frozen declaration and return it with its canonical hash.

    The hash is over the document as written, exactly as
    `load_carry_family_spec` hashes a family declaration, so the weekly
    artifact's `declaration_hash` binds every field of it -- the phase and the
    anchor included.
    """
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ShadowDeclarationError(f"shadow declaration is unreadable: {error}") from error
    try:
        declaration = ShadowDeclaration.model_validate(document)
    except ValidationError as error:
        raise ShadowDeclarationError(f"shadow declaration is invalid: {error}") from error
    return declaration, content_sha256(document)


def _parse_date(value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"invalid date: {value}") from error
    if parsed.isoformat() != value:
        # `fromisoformat` also accepts compact forms like "20260913"; a
        # declared Sunday is always YYYY-MM-DD.
        raise ValueError(f"invalid date: {value}")
    return parsed
