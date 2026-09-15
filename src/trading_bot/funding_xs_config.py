"""Frozen declaration of the perpetual-only funding cross-section family.

Spec 4 of `docs/superpowers/specs/2026-09-15-funding-xs-family-design.md`
declares everything but the member list, the hypothesis and one cost entry to
be `xs_momentum_panel_v1`'s, so the universe, weight, cost, fold and
statistics models are `panel_config`'s own rather than re-declared here --
the same arrangement `trend_config` has, which is what lets
`tests/test_funding_xs_config.py` compare the copied blocks value for value
against `configs/xs-momentum-panel-v1.json`.

The one declared change lives in the config file, not in the models: the
adverse table's `funding_receipt_multiplier` is P1.28's `0.75` rather than
P1.27's `0`, because this family's return *is* funding and a zero-receipt
scenario would test whether funding exists rather than whether the book
survives a worse funding regime.
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from trading_bot.canonical import content_sha256
from trading_bot.panel_config import (
    PanelCosts,
    PanelFoldGeometry,
    PanelStatistics,
    PanelUniverseRules,
    PanelWeightRules,
)

FUNDING_XS_MEMBER_NAMES: tuple[str, ...] = (
    "fx_q5_l4w_h1w",
    "fx_q5_l4w_h4w",
    "fx_q5_l1w_h4w",
    "fx_q5_l4w_h4w_exit",
)
FUNDING_XS_CONTROL_NAMES: tuple[str, ...] = ("no_trade", "random_ranks", "passive_long_ew")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class FundingXsMember(_Frozen):
    """One of spec 4.2's four trials: a lookback, a hold, and the exit rule."""

    name: str
    lookback_weeks: int
    hold_weeks: int
    exit_on_sign_flip: bool = False

    @field_validator("lookback_weeks")
    @classmethod
    def validate_lookback_weeks(cls, value: int) -> int:
        if value < 1:
            raise ValueError("lookback_weeks must be positive")
        return value

    @field_validator("hold_weeks")
    @classmethod
    def validate_hold_weeks(cls, value: int) -> int:
        # Spec 3 declares a one-week hold and a four-week cohort book; no
        # other hold has an assembly rule, so no other hold is buildable.
        if value not in (1, 4):
            raise ValueError("hold_weeks must be 1 or 4")
        return value


class FundingXsControl(_Frozen):
    name: str
    kind: Literal["no_trade", "random_ranks", "passive_long"]


class FundingXsFamilySpec(_Frozen):
    spec_version: Literal["1.0.0"]
    family_name: Literal["funding_xs_panel_v1"]
    hypothesis: str
    venue: Literal["BINANCE_UM"]
    holding_days: int
    members: tuple[FundingXsMember, ...]
    controls: tuple[FundingXsControl, ...]
    universe: PanelUniverseRules
    weights: PanelWeightRules
    costs: PanelCosts
    folds: PanelFoldGeometry
    statistics: PanelStatistics

    @field_validator("members")
    @classmethod
    def validate_members(cls, value: tuple[FundingXsMember, ...]) -> tuple[FundingXsMember, ...]:
        if tuple(member.name for member in value) != FUNDING_XS_MEMBER_NAMES:
            raise ValueError("the member set is frozen and ordered")
        return value

    @field_validator("controls")
    @classmethod
    def validate_controls(
        cls, value: tuple[FundingXsControl, ...]
    ) -> tuple[FundingXsControl, ...]:
        if tuple(control.name for control in value) != FUNDING_XS_CONTROL_NAMES:
            raise ValueError("the control set is frozen and ordered")
        return value


def load_funding_xs_family_spec(path: Path) -> tuple[FundingXsFamilySpec, str]:
    """Load the frozen declaration and return it with its canonical hash."""
    document = json.loads(path.read_text(encoding="utf-8"))
    spec = FundingXsFamilySpec.model_validate(document)
    return spec, content_sha256(document)
