"""Frozen declaration of the cross-sectional momentum experiment family."""

import json
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from trading_bot.canonical import content_sha256

MEMBER_NAMES: tuple[str, ...] = (
    "xs_mom_1w",
    "xs_mom_4w",
    "xs_mom_12w",
    "ts_mom_4w",
    "ts_mom_12w",
    "xs_rev_1w",
)
CONTROL_NAMES: tuple[str, ...] = ("no_trade", "random_ranks", "passive_long_ew")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PanelMember(_Frozen):
    name: Literal[
        "xs_mom_1w", "xs_mom_4w", "xs_mom_12w", "ts_mom_4w", "ts_mom_12w", "xs_rev_1w"
    ]
    kind: Literal["cross_sectional", "time_series"]
    lookback_days: int
    reversed: bool

    @field_validator("lookback_days")
    @classmethod
    def validate_lookback(cls, value: int) -> int:
        if value < 1:
            raise ValueError("lookback_days must be positive")
        return value


class PanelControl(_Frozen):
    name: Literal["no_trade", "random_ranks", "passive_long_ew"]
    kind: Literal["no_trade", "random_ranks", "passive_long"]


class PanelUniverseRules(_Frozen):
    minimum_history_days: int
    liquidity_window_days: int
    minimum_median_quote_volume: Decimal
    maximum_contracts: int
    minimum_contracts: int
    tier_one_rank_limit: int


class PanelWeightRules(_Frozen):
    leg_gross: Decimal
    minimum_quintile_size: int
    volatility_window_days: int
    volatility_floor: Decimal
    time_series_cap_numerator: Decimal


class PanelCostTable(_Frozen):
    name: Literal["base", "adverse"]
    fee_bps_per_side: Decimal
    slippage_bps_per_side_tier_one: Decimal
    slippage_bps_per_side_tier_two: Decimal
    funding_receipt_multiplier: Decimal
    funding_payment_multiplier: Decimal
    forced_close_multiplier: Decimal


class PanelCosts(_Frozen):
    base: PanelCostTable
    adverse: PanelCostTable


class PanelFoldGeometry(_Frozen):
    train_duration_ns: int
    validation_duration_ns: int
    test_duration_ns: int
    step_ns: int
    embargo_ns: int
    holdout_duration_ns: int


class PanelStatistics(_Frozen):
    block_length: int
    bootstrap_repetitions: int
    random_seed: int
    confidence: Decimal
    false_discovery_gate: Decimal
    pooled_episode_floor: int
    concentration_limit: Decimal
    positive_fold_numerator: int
    positive_fold_denominator: int


class PanelFamilySpec(_Frozen):
    spec_version: Literal["1.0.0"]
    family_name: Literal["xs_momentum_panel_v1"]
    hypothesis: str
    venue: Literal["BINANCE_UM"]
    holding_days: int
    members: tuple[PanelMember, ...]
    controls: tuple[PanelControl, ...]
    universe: PanelUniverseRules
    weights: PanelWeightRules
    costs: PanelCosts
    folds: PanelFoldGeometry
    statistics: PanelStatistics

    @field_validator("members")
    @classmethod
    def validate_members(cls, value: tuple[PanelMember, ...]) -> tuple[PanelMember, ...]:
        if tuple(item.name for item in value) != MEMBER_NAMES:
            raise ValueError("the member set is frozen and ordered")
        return value

    @field_validator("controls")
    @classmethod
    def validate_controls(cls, value: tuple[PanelControl, ...]) -> tuple[PanelControl, ...]:
        if tuple(item.name for item in value) != CONTROL_NAMES:
            raise ValueError("the control set is frozen and ordered")
        return value


def load_panel_family_spec(path: Path) -> tuple[PanelFamilySpec, str]:
    """Load the frozen declaration and return it with its canonical hash."""
    document = json.loads(path.read_text(encoding="utf-8"))
    spec = PanelFamilySpec.model_validate(document)
    return spec, content_sha256(document)
