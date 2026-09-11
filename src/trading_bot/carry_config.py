"""Frozen declaration of the funding carry experiment family."""

import json
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from trading_bot.canonical import content_sha256
from trading_bot.panel_config import PanelFoldGeometry, PanelStatistics

MEMBER_NAMES: tuple[str, ...] = ("carry_l1w_h4w", "carry_l4w_h4w", "carry_l4w_h13w")
CONTROL_NAMES: tuple[str, ...] = ("no_trade", "random_pairs", "all_pairs_ew")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CarryMember(_Frozen):
    name: Literal["carry_l1w_h4w", "carry_l4w_h4w", "carry_l4w_h13w"]
    lookback_weeks: int
    hold_weeks: int

    @field_validator("lookback_weeks", "hold_weeks")
    @classmethod
    def validate_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("lookback_weeks and hold_weeks must be positive")
        return value


class CarryControl(_Frozen):
    name: Literal["no_trade", "random_pairs", "all_pairs_ew"]
    kind: Literal["no_trade", "random_pairs", "all_pairs"]


class CarryPair(_Frozen):
    perpetual: str
    spot: str
    multiplier: int

    @field_validator("multiplier")
    @classmethod
    def validate_multiplier(cls, value: int) -> int:
        if value < 1:
            raise ValueError("multiplier must be at least one")
        return value


class CarryUniverseRules(_Frozen):
    minimum_history_days: int
    liquidity_window_days: int
    minimum_median_quote_volume: Decimal
    maximum_pairs: int
    minimum_pairs: int
    tier_one_rank_limit: int


class CarrySelectionRules(_Frozen):
    decile_denominator: int
    minimum_selected: int


class CarryCostTable(_Frozen):
    name: Literal["base", "adverse"]
    perpetual_fee_bps_per_side: Decimal
    spot_fee_bps_per_side: Decimal
    slippage_bps_per_side_tier_one: Decimal
    slippage_bps_per_side_tier_two: Decimal
    funding_receipt_multiplier: Decimal
    funding_payment_multiplier: Decimal
    forced_close_multiplier: Decimal


class CarryCosts(_Frozen):
    base: CarryCostTable
    adverse: CarryCostTable


class CarryFamilySpec(_Frozen):
    spec_version: Literal["1.0.0"]
    family_name: Literal["funding_carry_panel_v1"]
    hypothesis: str
    perpetual_venue: Literal["BINANCE_UM"]
    spot_venue: Literal["BINANCE_SPOT"]
    holding_days: int
    members: tuple[CarryMember, ...]
    controls: tuple[CarryControl, ...]
    pairs: tuple[CarryPair, ...]
    excluded_pairs: tuple[CarryPair, ...]
    universe: CarryUniverseRules
    selection: CarrySelectionRules
    costs: CarryCosts
    folds: PanelFoldGeometry
    statistics: PanelStatistics

    @field_validator("members")
    @classmethod
    def validate_members(cls, value: tuple[CarryMember, ...]) -> tuple[CarryMember, ...]:
        if tuple(item.name for item in value) != MEMBER_NAMES:
            raise ValueError("the member set is frozen and ordered")
        return value

    @field_validator("controls")
    @classmethod
    def validate_controls(cls, value: tuple[CarryControl, ...]) -> tuple[CarryControl, ...]:
        if tuple(item.name for item in value) != CONTROL_NAMES:
            raise ValueError("the control set is frozen and ordered")
        return value

    @field_validator("pairs")
    @classmethod
    def validate_pairs(cls, value: tuple[CarryPair, ...]) -> tuple[CarryPair, ...]:
        if any(item.multiplier != 1 for item in value):
            raise ValueError("included pairs must map one-to-one; scaled pairs are excluded")
        perpetuals = [item.perpetual for item in value]
        if len(set(perpetuals)) != len(perpetuals):
            raise ValueError("a perpetual may appear in at most one pair")
        return value

    @field_validator("excluded_pairs")
    @classmethod
    def validate_excluded(cls, value: tuple[CarryPair, ...]) -> tuple[CarryPair, ...]:
        if any(item.multiplier == 1 for item in value):
            raise ValueError("only scaled-multiplier pairs are excluded by declaration")
        return value


def load_carry_family_spec(path: Path) -> tuple[CarryFamilySpec, str]:
    """Load the frozen declaration and return it with its canonical hash."""
    document = json.loads(path.read_text(encoding="utf-8"))
    return CarryFamilySpec.model_validate(document), content_sha256(document)
