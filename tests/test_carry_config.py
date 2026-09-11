import json
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from trading_bot.carry_config import (
    CONTROL_NAMES,
    MEMBER_NAMES,
    CarryFamilySpec,
    load_carry_family_spec,
)
from trading_bot.panel_config import PanelFamilySpec, load_family_spec

CONFIG = Path("configs/funding-carry-panel-v1.json")


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
