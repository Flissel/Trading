import json
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from trading_bot.carry_config import (
    CONTROL_NAMES,
    MEMBER_NAMES,
    MEMBER_NAMES_BY_FAMILY,
    CarryFamilySpec,
    load_carry_family_spec,
)
from trading_bot.panel_config import PanelFamilySpec, load_family_spec

CONFIG = Path("configs/funding-carry-panel-v1.json")
CONFIG_V2 = Path("configs/funding-carry-panel-v2.json")


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
