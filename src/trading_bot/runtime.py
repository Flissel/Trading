"""Runtime mode authorization boundary."""

from dataclasses import dataclass
from enum import StrEnum


class RuntimePolicyError(ValueError):
    """Raised when a runtime mode is not authorized."""


class RuntimeMode(StrEnum):
    """Supported runtime modes."""

    BACKTEST = "backtest"
    SHADOW = "shadow"
    PAPER = "paper"
    TINY_LIVE = "tiny_live"


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Fail-closed runtime configuration."""

    mode: RuntimeMode = RuntimeMode.BACKTEST

    def __post_init__(self) -> None:
        if self.mode is RuntimeMode.TINY_LIVE:
            raise RuntimePolicyError("tiny_live mode is not authorized")

    @property
    def live_execution_enabled(self) -> bool:
        """Return whether this configuration can route real-capital orders."""
        return False
