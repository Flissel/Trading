"""Frozen declaration of the trend aggregate experiment family."""

import json
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from trading_bot.canonical import content_sha256
from trading_bot.panel_config import (
    PanelCosts,
    PanelFoldGeometry,
    PanelStatistics,
    PanelUniverseRules,
    PanelWeightRules,
)

TREND_MEMBER_NAMES: tuple[str, ...] = ("ta_ts_t02", "ta_ts_t05", "ta_ts_t02_h4w", "ta_xs_q5")
TREND_CONTROL_NAMES: tuple[str, ...] = ("no_trade", "random_ranks", "passive_long_ew")
INDICATOR_NAMES: tuple[str, ...] = (
    "ma_20",
    "ma_50",
    "ma_100",
    "ma_cross_20_50",
    "ma_cross_50_100",
    "breakout_20",
    "breakout_50",
    "breakout_100",
    "roc_20",
    "roc_60",
    "macd_12_26",
    "roc_120",
)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TrendMember(_Frozen):
    name: str
    kind: Literal["time_series", "cross_sectional"]
    threshold: Decimal | None
    hold_weeks: int

    @field_validator("hold_weeks")
    @classmethod
    def validate_hold_weeks(cls, value: int) -> int:
        if value not in (1, 4):
            raise ValueError("hold_weeks must be 1 or 4")
        return value

    @model_validator(mode="after")
    def validate_threshold(self) -> "TrendMember":
        if self.kind == "time_series":
            if self.threshold is None:
                raise ValueError("time_series members require a threshold")
            if not (Decimal("0") < self.threshold <= Decimal("1")):
                raise ValueError("threshold must be greater than 0 and at most 1")
        elif self.threshold is not None:
            raise ValueError("cross_sectional members must not declare a threshold")
        return self


class TrendControl(_Frozen):
    name: str
    kind: Literal["no_trade", "random_ranks", "passive_long"]


class TrendFamilySpec(_Frozen):
    spec_version: Literal["1.0.0"]
    family_name: Literal["trend_aggregate_panel_v1"]
    hypothesis: str
    venue: Literal["BINANCE_UM"]
    holding_days: int
    indicators: tuple[str, ...]
    members: tuple[TrendMember, ...]
    controls: tuple[TrendControl, ...]
    universe: PanelUniverseRules
    weights: PanelWeightRules
    costs: PanelCosts
    folds: PanelFoldGeometry
    statistics: PanelStatistics

    @field_validator("indicators")
    @classmethod
    def validate_indicators(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != INDICATOR_NAMES:
            raise ValueError("the indicator set is frozen and ordered")
        return value

    @field_validator("members")
    @classmethod
    def validate_members(cls, value: tuple[TrendMember, ...]) -> tuple[TrendMember, ...]:
        if tuple(member.name for member in value) != TREND_MEMBER_NAMES:
            raise ValueError("the member set is frozen and ordered")
        return value

    @field_validator("controls")
    @classmethod
    def validate_controls(cls, value: tuple[TrendControl, ...]) -> tuple[TrendControl, ...]:
        if tuple(control.name for control in value) != TREND_CONTROL_NAMES:
            raise ValueError("the control set is frozen and ordered")
        return value


def load_trend_family_spec(path: Path) -> tuple[TrendFamilySpec, str]:
    """Load the frozen declaration and return it with its canonical hash."""
    document = json.loads(path.read_text(encoding="utf-8"))
    spec = TrendFamilySpec.model_validate(document)
    return spec, content_sha256(document)
