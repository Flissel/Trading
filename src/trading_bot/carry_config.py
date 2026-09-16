"""Frozen declaration of the funding carry experiment family."""

import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from trading_bot.canonical import content_sha256
from trading_bot.cost_evidence_rule import CAPITAL_DECLARATION_RULE, DECLARATION_RULE
from trading_bot.panel_config import PanelFoldGeometry, PanelStatistics

MEMBER_NAMES: tuple[str, ...] = ("carry_l1w_h4w", "carry_l4w_h4w", "carry_l4w_h13w")
MEMBER_NAMES_V2: tuple[str, ...] = (
    "carry_l4w_h26w",
    "carry_l4w_h13w_exit",
    "carry_l4w_h26w_exit",
    "carry_l4w_h26w_exit_hurdle2",
)
# v4 is a small, declared-capital book: ten equal pair slots rather than v2's
# weekly cohorts, so its four members trade hold and the exit rule the way v2
# did but carry no hurdle member (spec 2026-09-16, section 4.2).
MEMBER_NAMES_V4: tuple[str, ...] = (
    "carry_s10_l4w_h13w",
    "carry_s10_l4w_h13w_exit",
    "carry_s10_l4w_h26w",
    "carry_s10_l4w_h26w_exit",
)
# v3 is v2's book on measured execution costs: the same members, the same
# universe, the same folds - the cost tables are the only thing the Binance cost
# journal's receipt changes, so the two families stay comparable. v4_measured
# is the same relationship for the slot book. `MEASURED_COST_FAMILY` is kept
# as the v3 family's name and as `MEASURED_FAMILY_BY_BASE`'s value for the v2
# base; `MEASURED_FAMILY_BY_BASE` is the general map a new base family joins.
MEASURED_COST_FAMILY = "funding_carry_panel_v3"
MEASURED_FAMILY_BY_BASE: dict[str, str] = {
    "funding_carry_panel_v2": MEASURED_COST_FAMILY,
    "funding_carry_panel_v4": "funding_carry_panel_v4_measured",
}
MEASURED_FAMILIES: frozenset[str] = frozenset(MEASURED_FAMILY_BY_BASE.values())
MEMBER_NAMES_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "funding_carry_panel_v1": MEMBER_NAMES,
    "funding_carry_panel_v2": MEMBER_NAMES_V2,
    MEASURED_COST_FAMILY: MEMBER_NAMES_V2,
    "funding_carry_panel_v4": MEMBER_NAMES_V4,
    "funding_carry_panel_v4_measured": MEMBER_NAMES_V4,
}
CONTROL_NAMES: tuple[str, ...] = ("no_trade", "random_pairs", "all_pairs_ew")

_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CarryCapital(_Frozen):
    """The money a slot book is sized against.

    A cohort family's members speak for their own weight; a slot family's do
    not, because a slot's size is a fraction of a declared book rather than of
    whatever a weekly cohort happens to hold. `book_usdt` is the capital the
    book presumes, split evenly across `pair_slots` equal slots, so
    `per_leg_notional_usdt` -- the order size a fill actually places -- is
    fixed by the other two fields and may never be an independent number a
    declaration could drift from its own arithmetic.
    """

    book_usdt: Decimal
    pair_slots: int
    per_leg_notional_usdt: Decimal
    fee_tier: Literal["standard_taker_no_bnb"]

    @field_validator("book_usdt")
    @classmethod
    def validate_book_usdt(cls, value: Decimal) -> Decimal:
        if value <= 0:
            raise ValueError("book_usdt must be positive")
        return value

    @field_validator("pair_slots")
    @classmethod
    def validate_pair_slots(cls, value: int) -> int:
        if value < 1:
            raise ValueError("pair_slots must be at least one")
        return value

    @model_validator(mode="after")
    def validate_per_leg_notional(self) -> "CarryCapital":
        expected = self.book_usdt / (2 * self.pair_slots)
        if self.per_leg_notional_usdt != expected:
            raise ValueError("per_leg_notional_usdt must equal book_usdt / (2 * pair_slots)")
        return self


class CarryMember(_Frozen):
    name: str
    lookback_weeks: int
    hold_weeks: int
    exit_on_negative_funding: bool = False
    hurdle_multiple: Decimal | None = None
    book: Literal["cohorts", "slots"] = "cohorts"

    @field_validator("lookback_weeks", "hold_weeks")
    @classmethod
    def validate_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("lookback_weeks and hold_weeks must be positive")
        return value

    @field_validator("hurdle_multiple")
    @classmethod
    def validate_hurdle_multiple(cls, value: Decimal | None) -> Decimal | None:
        if value is not None and value <= 0:
            raise ValueError("hurdle_multiple must be positive when set")
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


class CostEvidenceReference(_Frozen):
    """The measurement a declaration's slippage tiers were read off.

    A measured family cites the Binance cost journal's finalisation receipt by
    hash, names the journal spec that receipt bound, records the two notionals
    spec section 5 fixes - the base table's and the adverse table's - and
    repeats the rule the receipt carries, so the declaration says where every
    basis point came from without its reader holding the journal.

    The rule is one of the two sentences spec sections 5 and 5.1 fixed before
    any number existed - ``cost_evidence_rule.DECLARATION_RULE`` for a
    declaration without a ``capital`` block and ``CAPITAL_DECLARATION_RULE``
    for one that carries one, each the same string the receipt is published
    under: a declaration citing any other reading is refused here, and
    ``carry_measured_costs.verify_measured_declaration`` re-derives every
    number the citation stands for from the receipt itself, including which of
    the two rules the declaration's own capital selects.
    """

    receipt_hash: str
    journal_spec_hash: str
    base_notional: Decimal
    adverse_notional: Decimal
    rule: str

    @field_validator("receipt_hash", "journal_spec_hash")
    @classmethod
    def validate_hashes(cls, value: str) -> str:
        if not _HEX64.match(value):
            raise ValueError("an evidence hash is 64 lower-case hexadecimal characters")
        return value

    @field_validator("base_notional", "adverse_notional")
    @classmethod
    def validate_notionals(cls, value: Decimal) -> Decimal:
        if value <= 0:
            raise ValueError("a cited notional is positive")
        return value

    @field_validator("rule")
    @classmethod
    def validate_rule(cls, value: str) -> str:
        if value not in (DECLARATION_RULE, CAPITAL_DECLARATION_RULE):
            raise ValueError("the cited rule is spec section 5's or 5.1's sentence, verbatim")
        return value


class CarryFamilySpec(_Frozen):
    spec_version: Literal["1.0.0"]
    family_name: Literal[
        "funding_carry_panel_v1",
        "funding_carry_panel_v2",
        "funding_carry_panel_v3",
        "funding_carry_panel_v4",
        "funding_carry_panel_v4_measured",
    ]
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
    cost_evidence: CostEvidenceReference | None = None
    capital: CarryCapital | None = None

    @model_validator(mode="after")
    def validate_cost_evidence(self) -> "CarryFamilySpec":
        """Measured costs are cited, and assumed costs may not pretend to be.

        A measured family's cost tables are a reading of a finalisation
        receipt, so the declaration carries the evidence or it is not a
        declaration; v1, v2 and v4 declare the assumed tiers and must not wear
        a citation they did not earn.
        """
        measured = self.family_name in MEASURED_FAMILIES
        if measured and self.cost_evidence is None:
            raise ValueError("a measured family cites the receipt its cost tables came from")
        if not measured and self.cost_evidence is not None:
            raise ValueError("only a measured family carries cost evidence")
        return self

    @model_validator(mode="after")
    def validate_members(self) -> "CarryFamilySpec":
        expected = MEMBER_NAMES_BY_FAMILY[self.family_name]
        if tuple(member.name for member in self.members) != expected:
            raise ValueError("the member set is frozen and ordered")
        return self

    @model_validator(mode="after")
    def validate_capital_matches_book(self) -> "CarryFamilySpec":
        """A slot book is capitalised and a cohort book is not; never a mix.

        `capital` is what gives a slot its size, so a member whose `book` is
        "slots" is meaningless without it, and a `capital` block on a family
        whose members are still cohort-sized would size money no slot spends.
        """
        books = {member.book for member in self.members}
        if self.capital is not None and books != {"slots"}:
            raise ValueError("a family with a capital block must slot every member")
        if self.capital is None and "slots" in books:
            raise ValueError("a slot member requires a family capital block")
        return self

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
