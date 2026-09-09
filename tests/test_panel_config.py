import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from trading_bot.panel_config import (
    CONTROL_NAMES,
    MEMBER_NAMES,
    load_panel_family_spec,
)

REPOSITORY_CONFIG = Path("configs/xs-momentum-panel-v1.json")


def test_repository_config_declares_the_frozen_family() -> None:
    spec, spec_hash = load_panel_family_spec(REPOSITORY_CONFIG)
    assert spec.family_name == "xs_momentum_panel_v1"
    assert tuple(member.name for member in spec.members) == MEMBER_NAMES
    assert tuple(control.name for control in spec.controls) == CONTROL_NAMES
    assert spec.statistics.block_length == 4
    assert spec.statistics.bootstrap_repetitions == 2000
    assert spec.statistics.pooled_episode_floor == 200
    assert len(spec_hash) == 64


def test_member_set_is_closed(tmp_path: Path) -> None:
    document = json.loads(REPOSITORY_CONFIG.read_text(encoding="utf-8"))
    document["members"].append(
        {"name": "xs_mom_26w", "kind": "cross_sectional", "lookback_days": 182}
    )
    path = tmp_path / "extra-member.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_panel_family_spec(path)


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    document = json.loads(REPOSITORY_CONFIG.read_text(encoding="utf-8"))
    document["tuning_knob"] = 3
    path = tmp_path / "extra-field.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_panel_family_spec(path)
