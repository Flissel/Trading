import json
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from trading_bot.carry_config import CarryFamilySpec
from trading_bot.panel_config import PanelFamilySpec, load_family_spec
from trading_bot.trend_config import (
    INDICATOR_NAMES,
    TREND_CONTROL_NAMES,
    TREND_MEMBER_NAMES,
    TrendFamilySpec,
    load_trend_family_spec,
)

CONFIG = Path("configs/trend-aggregate-panel-v1.json")
PANEL_CONFIG = Path("configs/xs-momentum-panel-v1.json")
CARRY_CONFIG = Path("configs/funding-carry-panel-v1.json")


def test_repository_declaration_is_the_frozen_family() -> None:
    spec, spec_hash = load_trend_family_spec(CONFIG)
    assert spec.family_name == "trend_aggregate_panel_v1"
    assert spec.indicators == INDICATOR_NAMES
    assert tuple(m.name for m in spec.members) == TREND_MEMBER_NAMES
    assert tuple(c.name for c in spec.controls) == TREND_CONTROL_NAMES
    thresholds = {m.name: m.threshold for m in spec.members}
    assert thresholds["ta_ts_t02"] == Decimal("0.2")
    assert thresholds["ta_ts_t05"] == Decimal("0.5")
    assert thresholds["ta_ts_t02_h4w"] == Decimal("0.2")
    assert thresholds["ta_xs_q5"] is None
    hold_weeks = {m.name: m.hold_weeks for m in spec.members}
    assert hold_weeks["ta_ts_t02"] == 1
    assert hold_weeks["ta_ts_t05"] == 1
    assert hold_weeks["ta_ts_t02_h4w"] == 4
    assert hold_weeks["ta_xs_q5"] == 1
    kinds = {m.name: m.kind for m in spec.members}
    assert kinds["ta_ts_t02"] == "time_series"
    assert kinds["ta_ts_t05"] == "time_series"
    assert kinds["ta_ts_t02_h4w"] == "time_series"
    assert kinds["ta_xs_q5"] == "cross_sectional"
    control_kinds = {c.name: c.kind for c in spec.controls}
    assert control_kinds["no_trade"] == "no_trade"
    assert control_kinds["random_ranks"] == "random_ranks"
    assert control_kinds["passive_long_ew"] == "passive_long"
    assert len(spec_hash) == 64


def test_the_five_copied_blocks_equal_p1_27_byte_for_byte() -> None:
    trend_spec, _ = load_trend_family_spec(CONFIG)
    panel_spec, _ = load_family_spec(PANEL_CONFIG)
    assert isinstance(panel_spec, PanelFamilySpec)
    assert trend_spec.universe == panel_spec.universe
    assert trend_spec.weights == panel_spec.weights
    assert trend_spec.costs == panel_spec.costs
    assert trend_spec.folds == panel_spec.folds
    assert trend_spec.statistics == panel_spec.statistics
    assert trend_spec.holding_days == panel_spec.holding_days
    assert trend_spec.venue == panel_spec.venue
    assert trend_spec.spec_version == panel_spec.spec_version


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
        load_trend_family_spec(path)


def test_time_series_member_without_threshold_is_rejected(tmp_path: Path) -> None:
    document = json.loads(CONFIG.read_text(encoding="utf-8"))
    document["members"][0]["threshold"] = None
    path = tmp_path / "missing-threshold.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_trend_family_spec(path)


def test_cross_sectional_member_with_threshold_is_rejected(tmp_path: Path) -> None:
    document = json.loads(CONFIG.read_text(encoding="utf-8"))
    document["members"][3]["threshold"] = "0.2"
    path = tmp_path / "extra-threshold.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_trend_family_spec(path)


def test_dispatching_loader_returns_the_trend_model() -> None:
    spec, spec_hash = load_family_spec(CONFIG)
    assert isinstance(spec, TrendFamilySpec)
    assert spec.family_name == "trend_aggregate_panel_v1"
    assert len(spec_hash) == 64


def test_dispatching_loader_still_returns_panel_and_carry_models() -> None:
    panel_spec, _ = load_family_spec(PANEL_CONFIG)
    carry_spec, _ = load_family_spec(CARRY_CONFIG)
    assert isinstance(panel_spec, PanelFamilySpec)
    assert isinstance(carry_spec, CarryFamilySpec)
