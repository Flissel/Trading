import json
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.carry_fixtures import small_carry_v4_config
from trading_bot.carry_config import (
    CONTROL_NAMES,
    MEASURED_COST_FAMILY,
    MEASURED_FAMILIES,
    MEASURED_FAMILY_BY_BASE,
    MEMBER_NAMES,
    MEMBER_NAMES_BY_FAMILY,
    MEMBER_NAMES_V4,
    CarryCapital,
    CarryFamilySpec,
    CarryMember,
    load_carry_family_spec,
)
from trading_bot.cost_evidence_rule import DECLARATION_RULE
from trading_bot.panel_config import PanelFamilySpec, load_family_spec

CONFIG = Path("configs/funding-carry-panel-v1.json")
CONFIG_V2 = Path("configs/funding-carry-panel-v2.json")
CONFIG_V4 = Path("configs/funding-carry-panel-v4.json")


def test_repository_declaration_is_the_frozen_family() -> None:
    spec, spec_hash = load_carry_family_spec(CONFIG)
    assert spec.family_name == "funding_carry_panel_v1"
    assert tuple(m.name for m in spec.members) == MEMBER_NAMES
    assert tuple(c.name for c in spec.controls) == CONTROL_NAMES
    assert [(m.lookback_weeks, m.hold_weeks) for m in spec.members] == [(1, 4), (4, 4), (4, 13)]
    assert spec.costs.base.spot_fee_bps_per_side == Decimal("10")
    assert spec.costs.adverse.funding_receipt_multiplier == Decimal("0.75")
    assert spec.selection.minimum_selected == 8
    assert len(spec.pairs) == 470
    assert len(spec.excluded_pairs) == 7
    assert all(p.multiplier == 1 for p in spec.pairs)
    assert all(p.multiplier > 1 for p in spec.excluded_pairs)
    assert len({p.perpetual for p in spec.pairs}) == len(spec.pairs)
    assert len(spec_hash) == 64


def test_member_set_is_closed_and_ordered(tmp_path: Path) -> None:
    document = json.loads(CONFIG.read_text(encoding="utf-8"))
    document["members"] = [document["members"][1], document["members"][0], document["members"][2]]
    path = tmp_path / "reordered.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_carry_family_spec(path)


def test_scaled_pair_in_the_included_list_is_rejected(tmp_path: Path) -> None:
    document = json.loads(CONFIG.read_text(encoding="utf-8"))
    document["pairs"].append({"perpetual": "1000XYZUSDT", "spot": "XYZUSDT", "multiplier": 1000})
    path = tmp_path / "scaled.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_carry_family_spec(path)


def test_dispatching_loader_returns_the_right_model() -> None:
    carry, carry_hash = load_family_spec(CONFIG)
    panel, panel_hash = load_family_spec(Path("configs/xs-momentum-panel-v1.json"))
    assert isinstance(carry, CarryFamilySpec)
    assert isinstance(panel, PanelFamilySpec)
    assert carry_hash != panel_hash


def test_dispatching_loader_rejects_an_unknown_family(tmp_path: Path) -> None:
    document = json.loads(CONFIG.read_text(encoding="utf-8"))
    document["family_name"] = "something_else_v1"
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_family_spec(path)


def test_v2_declaration_loads_the_four_members_and_their_flags() -> None:
    spec, spec_hash = load_carry_family_spec(CONFIG_V2)
    assert spec.family_name == "funding_carry_panel_v2"
    assert tuple(m.name for m in spec.members) == MEMBER_NAMES_BY_FAMILY["funding_carry_panel_v2"]
    by_name = {m.name: m for m in spec.members}
    assert by_name["carry_l4w_h26w"].exit_on_negative_funding is False
    assert by_name["carry_l4w_h26w"].hurdle_multiple is None
    assert by_name["carry_l4w_h13w_exit"].exit_on_negative_funding is True
    assert by_name["carry_l4w_h13w_exit"].hurdle_multiple is None
    assert by_name["carry_l4w_h26w_exit"].exit_on_negative_funding is True
    assert by_name["carry_l4w_h26w_exit"].hurdle_multiple is None
    assert by_name["carry_l4w_h26w_exit_hurdle2"].exit_on_negative_funding is True
    assert by_name["carry_l4w_h26w_exit_hurdle2"].hurdle_multiple == Decimal("2")
    assert len(spec_hash) == 64


def test_v1_declaration_still_loads_with_new_fields_defaulted() -> None:
    spec, _ = load_carry_family_spec(CONFIG)
    assert spec.family_name == "funding_carry_panel_v1"
    for member in spec.members:
        assert member.exit_on_negative_funding is False
        assert member.hurdle_multiple is None


def test_v2_document_with_v1_member_names_is_rejected(tmp_path: Path) -> None:
    document = json.loads(CONFIG_V2.read_text(encoding="utf-8"))
    document["members"] = json.loads(CONFIG.read_text(encoding="utf-8"))["members"]
    path = tmp_path / "v2-with-v1-members.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_carry_family_spec(path)


def test_member_with_zero_hurdle_multiple_is_rejected(tmp_path: Path) -> None:
    document = json.loads(CONFIG_V2.read_text(encoding="utf-8"))
    document["members"][-1]["hurdle_multiple"] = "0"
    path = tmp_path / "zero-hurdle.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_carry_family_spec(path)


def test_dispatching_loader_returns_carry_spec_for_both_families_with_distinct_hashes() -> None:
    v1_spec, v1_hash = load_family_spec(CONFIG)
    v2_spec, v2_hash = load_family_spec(CONFIG_V2)
    assert isinstance(v1_spec, CarryFamilySpec)
    assert isinstance(v2_spec, CarryFamilySpec)
    assert v1_hash != v2_hash


# --- v4: declared capital, slot book (spec 2026-09-16-funding-carry-v4-small-book) ---

SAMPLE_COST_EVIDENCE = {
    "receipt_hash": "a" * 64,
    "journal_spec_hash": "b" * 64,
    "base_notional": "500",
    "adverse_notional": "5000",
    "rule": DECLARATION_RULE,
}


def _capital(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "book_usdt": "10000",
        "pair_slots": 10,
        "per_leg_notional_usdt": "500",
        "fee_tier": "standard_taker_no_bnb",
    }
    return {**base, **overrides}


def test_capital_rejects_a_notional_that_is_not_book_over_twice_slots() -> None:
    with pytest.raises(ValidationError):
        CarryCapital.model_validate(_capital(per_leg_notional_usdt="999"))


def test_capital_rejects_a_non_positive_book() -> None:
    with pytest.raises(ValidationError):
        CarryCapital.model_validate(_capital(book_usdt="0", per_leg_notional_usdt="0"))


def test_capital_rejects_zero_slots() -> None:
    with pytest.raises(ValidationError):
        CarryCapital.model_validate(_capital(pair_slots=0))


def test_capital_rejects_another_fee_tier() -> None:
    with pytest.raises(ValidationError):
        CarryCapital.model_validate(_capital(fee_tier="maker"))


def test_capital_accepts_the_exact_book_over_twice_slots() -> None:
    capital = CarryCapital.model_validate(_capital())
    assert capital.per_leg_notional_usdt == Decimal("500")


def test_member_book_defaults_to_cohorts() -> None:
    member = CarryMember(name="carry_x", lookback_weeks=4, hold_weeks=13)
    assert member.book == "cohorts"


def test_capital_with_a_cohort_member_is_refused() -> None:
    document = json.loads(CONFIG_V4.read_text(encoding="utf-8"))
    document["members"][0]["book"] = "cohorts"
    with pytest.raises(ValidationError):
        CarryFamilySpec.model_validate(document)


def test_slot_member_without_capital_is_refused() -> None:
    document = json.loads(CONFIG_V2.read_text(encoding="utf-8"))
    document["members"][0]["book"] = "slots"
    with pytest.raises(ValidationError):
        CarryFamilySpec.model_validate(document)


def test_v4_base_declaration_carries_no_cost_evidence() -> None:
    spec, _ = load_carry_family_spec(CONFIG_V4)
    assert spec.family_name == "funding_carry_panel_v4"
    assert spec.cost_evidence is None
    assert spec.capital is not None


def test_v4_measured_requires_cost_evidence() -> None:
    document = json.loads(CONFIG_V4.read_text(encoding="utf-8"))
    document["family_name"] = "funding_carry_panel_v4_measured"
    with pytest.raises(ValidationError):
        CarryFamilySpec.model_validate(document)
    document["cost_evidence"] = SAMPLE_COST_EVIDENCE
    spec = CarryFamilySpec.model_validate(document)
    assert spec.family_name == "funding_carry_panel_v4_measured"
    assert spec.cost_evidence is not None


def test_member_names_by_family_binds_v4_names_to_both_v4_families() -> None:
    assert MEMBER_NAMES_BY_FAMILY["funding_carry_panel_v4"] == MEMBER_NAMES_V4
    assert MEMBER_NAMES_BY_FAMILY["funding_carry_panel_v4_measured"] == MEMBER_NAMES_V4


def test_measured_families_and_family_by_base_are_consistent() -> None:
    assert MEASURED_FAMILY_BY_BASE["funding_carry_panel_v2"] == MEASURED_COST_FAMILY
    assert MEASURED_FAMILY_BY_BASE["funding_carry_panel_v4"] == "funding_carry_panel_v4_measured"
    assert frozenset(MEASURED_FAMILY_BY_BASE.values()) == MEASURED_FAMILIES


def test_v4_config_equals_v2_except_the_declared_fields() -> None:
    v2_document = json.loads(CONFIG_V2.read_text(encoding="utf-8"))
    v4_document = json.loads(CONFIG_V4.read_text(encoding="utf-8"))
    changed = {"family_name", "hypothesis", "members"}
    assert set(v4_document) == set(v2_document) | {"capital"}
    for key in v2_document:
        if key not in changed:
            assert v4_document[key] == v2_document[key], key
    assert v4_document["family_name"] == "funding_carry_panel_v4"
    assert v4_document["hypothesis"] != v2_document["hypothesis"]
    assert v4_document["members"] != v2_document["members"]


def test_v4_config_members_are_the_four_frozen_slot_entries() -> None:
    v4_document = json.loads(CONFIG_V4.read_text(encoding="utf-8"))
    members = v4_document["members"]
    assert tuple(m["name"] for m in members) == MEMBER_NAMES_V4
    assert all(m["book"] == "slots" for m in members)
    assert [m["lookback_weeks"] for m in members] == [4, 4, 4, 4]
    assert [m["hold_weeks"] for m in members] == [13, 13, 26, 26]
    assert [m["exit_on_negative_funding"] for m in members] == [False, True, False, True]
    assert [m["hurdle_multiple"] for m in members] == [None, None, None, None]


def test_v4_config_loads_and_dispatches_to_carry_family_spec() -> None:
    spec, spec_hash = load_family_spec(CONFIG_V4)
    assert isinstance(spec, CarryFamilySpec)
    assert spec.family_name == "funding_carry_panel_v4"
    assert len(spec_hash) == 64
    assert spec.capital is not None
    assert spec.capital.pair_slots == 10
    assert spec.capital.book_usdt == Decimal("10000")
    assert spec.capital.per_leg_notional_usdt == Decimal("500")
    assert spec.capital.fee_tier == "standard_taker_no_bnb"
    assert all(member.book == "slots" for member in spec.members)


def test_v4_fixture_loads_with_four_slots_and_the_reduced_holds(tmp_path: Path) -> None:
    path = small_carry_v4_config(tmp_path)
    spec, _ = load_carry_family_spec(path)
    assert spec.family_name == "funding_carry_panel_v4"
    assert spec.capital is not None
    assert spec.capital.pair_slots == 4
    assert spec.capital.book_usdt == Decimal("10000")
    assert spec.capital.per_leg_notional_usdt == Decimal("1250")
    assert [member.hold_weeks for member in spec.members] == [1, 1, 2, 2]
    assert tuple(member.name for member in spec.members) == MEMBER_NAMES_V4
    assert [member.exit_on_negative_funding for member in spec.members] == [
        False, True, False, True,
    ]
