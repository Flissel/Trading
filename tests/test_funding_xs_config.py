import json
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from trading_bot.carry_config import CarryFamilySpec
from trading_bot.funding_xs_config import (
    FUNDING_XS_CONTROL_NAMES,
    FUNDING_XS_MEMBER_NAMES,
    FundingXsFamilySpec,
    load_funding_xs_family_spec,
)
from trading_bot.panel_config import PanelFamilySpec, load_family_spec
from trading_bot.trend_config import TrendFamilySpec

CONFIG = Path("configs/funding-xs-panel-v1.json")
PANEL_CONFIG = Path("configs/xs-momentum-panel-v1.json")
CARRY_CONFIG = Path("configs/funding-carry-panel-v1.json")
TREND_CONFIG = Path("configs/trend-aggregate-panel-v1.json")

# The four frozen members of spec 4.2, as (lookback_weeks, hold_weeks, exit).
DECLARED_MEMBERS = {
    "fx_q5_l4w_h1w": (4, 1, False),
    "fx_q5_l4w_h4w": (4, 4, False),
    "fx_q5_l1w_h4w": (1, 4, False),
    "fx_q5_l4w_h4w_exit": (4, 4, True),
}


def test_repository_declaration_is_the_frozen_family() -> None:
    spec, spec_hash = load_funding_xs_family_spec(CONFIG)
    assert spec.family_name == "funding_xs_panel_v1"
    assert spec.spec_version == "1.0.0"
    assert spec.venue == "BINANCE_UM"
    assert tuple(member.name for member in spec.members) == FUNDING_XS_MEMBER_NAMES
    assert tuple(control.name for control in spec.controls) == FUNDING_XS_CONTROL_NAMES
    assert {
        member.name: (member.lookback_weeks, member.hold_weeks, member.exit_on_sign_flip)
        for member in spec.members
    } == DECLARED_MEMBERS
    control_kinds = {control.name: control.kind for control in spec.controls}
    assert control_kinds["no_trade"] == "no_trade"
    assert control_kinds["random_ranks"] == "random_ranks"
    assert control_kinds["passive_long_ew"] == "passive_long"
    assert len(spec_hash) == 64


def test_the_hypothesis_is_the_spec_sentence() -> None:
    """Spec 4.1 verbatim: one sentence, both cost scenarios, four members."""
    spec, _ = load_funding_xs_family_spec(CONFIG)
    assert spec.hypothesis.startswith(
        "A dollar-neutral book that is long the lowest-funding quintile and short the "
        "highest-funding quintile"
    )
    assert "positive expectancy under base costs and non-negative under adverse costs" in (
        spec.hypothesis
    )
    assert spec.hypothesis.endswith("after correction across four members.")


def test_the_copied_blocks_equal_p1_27_value_for_value() -> None:
    """Spec 4: everything not declared here is `xs_momentum_panel_v1`'s.

    Compared on the raw documents rather than on the parsed models, so a
    re-spelled Decimal (`"0.2"` for `"0.20"`) is caught too -- both blocks
    feed `content_sha256`, which hashes the strings as written.
    """
    funding_xs = json.loads(CONFIG.read_text(encoding="utf-8"))
    panel = json.loads(PANEL_CONFIG.read_text(encoding="utf-8"))
    for block in ("universe", "weights", "folds", "statistics"):
        assert funding_xs[block] == panel[block], block
    for field in ("holding_days", "venue", "spec_version"):
        assert funding_xs[field] == panel[field], field
    assert funding_xs["controls"] == panel["controls"]
    assert funding_xs["costs"]["base"] == panel["costs"]["base"]
    adverse = dict(panel["costs"]["adverse"])
    adverse["funding_receipt_multiplier"] = "0.75"
    assert funding_xs["costs"]["adverse"] == adverse


def test_the_adverse_funding_rule_is_p1_28s() -> None:
    """Spec 4's one declared change: receipts x0.75 (not x0), payments x2."""
    spec, _ = load_funding_xs_family_spec(CONFIG)
    panel_spec, _ = load_family_spec(PANEL_CONFIG)
    assert isinstance(panel_spec, PanelFamilySpec)
    assert spec.costs.adverse.funding_receipt_multiplier == Decimal("0.75")
    assert spec.costs.adverse.funding_payment_multiplier == Decimal("2")
    assert panel_spec.costs.adverse.funding_receipt_multiplier == Decimal("0")
    assert spec.costs.base == panel_spec.costs.base
    assert spec.costs.base.funding_receipt_multiplier == Decimal("1")
    assert spec.costs.base.funding_payment_multiplier == Decimal("1")
    assert spec.costs.adverse == panel_spec.costs.adverse.model_copy(
        update={"funding_receipt_multiplier": Decimal("0.75")}
    )
    assert spec.universe == panel_spec.universe
    assert spec.weights == panel_spec.weights
    assert spec.folds == panel_spec.folds
    assert spec.statistics == panel_spec.statistics


def test_member_set_rejects_reorder(tmp_path: Path) -> None:
    document = json.loads(CONFIG.read_text(encoding="utf-8"))
    document["members"] = [
        document["members"][1],
        document["members"][0],
        document["members"][2],
        document["members"][3],
    ]
    path = tmp_path / "reordered.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_funding_xs_family_spec(path)


def test_control_set_rejects_reorder(tmp_path: Path) -> None:
    document = json.loads(CONFIG.read_text(encoding="utf-8"))
    document["controls"] = list(reversed(document["controls"]))
    path = tmp_path / "reordered-controls.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_funding_xs_family_spec(path)


def test_a_fifth_member_is_rejected(tmp_path: Path) -> None:
    document = json.loads(CONFIG.read_text(encoding="utf-8"))
    document["members"] = [
        *document["members"],
        {
            "name": "fx_q5_l8w_h1w",
            "lookback_weeks": 8,
            "hold_weeks": 1,
            "exit_on_sign_flip": False,
        },
    ]
    path = tmp_path / "fifth-member.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_funding_xs_family_spec(path)


def test_an_unknown_field_is_rejected(tmp_path: Path) -> None:
    document = json.loads(CONFIG.read_text(encoding="utf-8"))
    document["tuning_knob"] = 3
    path = tmp_path / "extra-field.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_funding_xs_family_spec(path)


def test_a_hold_of_two_weeks_is_rejected(tmp_path: Path) -> None:
    """Spec 3 declares one- and four-week holds; nothing else is buildable."""
    document = json.loads(CONFIG.read_text(encoding="utf-8"))
    document["members"][0]["hold_weeks"] = 2
    path = tmp_path / "two-week-hold.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_funding_xs_family_spec(path)


def test_a_non_positive_lookback_is_rejected(tmp_path: Path) -> None:
    document = json.loads(CONFIG.read_text(encoding="utf-8"))
    document["members"][0]["lookback_weeks"] = 0
    path = tmp_path / "zero-lookback.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_funding_xs_family_spec(path)


def test_dispatching_loader_returns_the_funding_xs_model() -> None:
    spec, spec_hash = load_family_spec(CONFIG)
    assert isinstance(spec, FundingXsFamilySpec)
    assert spec.family_name == "funding_xs_panel_v1"
    direct, direct_hash = load_funding_xs_family_spec(CONFIG)
    assert spec == direct
    assert spec_hash == direct_hash


def test_dispatching_loader_still_returns_the_other_families() -> None:
    panel_spec, _ = load_family_spec(PANEL_CONFIG)
    carry_spec, _ = load_family_spec(CARRY_CONFIG)
    trend_spec, _ = load_family_spec(TREND_CONFIG)
    assert isinstance(panel_spec, PanelFamilySpec)
    assert isinstance(carry_spec, CarryFamilySpec)
    assert isinstance(trend_spec, TrendFamilySpec)
