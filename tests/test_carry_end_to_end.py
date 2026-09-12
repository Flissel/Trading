import json
from pathlib import Path
from unittest.mock import patch

from tests.carry_fixtures import perp_fetch, small_carry_config, spot_fetch
from tests.test_panel_fold_run import MONTHS, SYMBOLS
from trading_bot.cli import main

EXTRAS = {
    "funding_collected", "basis_pnl", "spot_trading_cost", "perpetual_trading_cost",
    "forced_spot_legs", "forced_perpetual_legs", "exit_rule_removals", "hurdle_rejections",
}


def test_spot_capture_manifest_carry_folds_and_decision_compose(tmp_path: Path) -> None:
    config_path = small_carry_config(tmp_path)
    capture_arguments = [
        "--workspace-root", str(tmp_path), "--reserve-bytes", "0",
        "--symbols", ",".join(SYMBOLS), "--months", ",".join(MONTHS),
    ]
    with patch("trading_bot.panel_capture.PanelZipClient.fetch", side_effect=perp_fetch):
        assert main(["panel-capture", "--output", str(tmp_path / "perp"), *capture_arguments]) == 0
    with patch("trading_bot.panel_capture.PanelZipClient.fetch", side_effect=spot_fetch):
        assert main([
            "panel-capture", "--market", "spot",
            "--output", str(tmp_path / "spot"), *capture_arguments,
        ]) == 0
    spot_manifest_path = tmp_path / "spot" / "capture-manifest.json"
    spot_manifest = json.loads(spot_manifest_path.read_text(encoding="utf-8"))
    assert spot_manifest["market"] == "spot"

    assert main([
        "panel-manifest", "--workspace-root", str(tmp_path),
        "--capture", str(tmp_path / "perp"), "--hedge-capture", str(tmp_path / "spot"),
        "--output", str(tmp_path / "manifest.json"), "--family-spec", str(config_path),
    ]) == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["hedge_capture_root_hash"] == spot_manifest["capture_root_hash"]

    reports = []
    for fold in manifest["folds"]:
        output = tmp_path / f"fold{fold['fold_index']}.json"
        assert main([
            "carry-fold", "--workspace-root", str(tmp_path),
            "--capture", str(tmp_path / "perp"), "--hedge-capture", str(tmp_path / "spot"),
            "--manifest", str(tmp_path / "manifest.json"), "--family-spec", str(config_path),
            "--output", str(output), "--registry", str(tmp_path / "registry.sqlite3"),
            "--fold-index", str(fold["fold_index"]),
        ]) == 0
        reports.append(output)

    arguments = [
        "panel-decision", "--workspace-root", str(tmp_path), "--family-spec", str(config_path),
        "--output", str(tmp_path / "decision.json"),
        "--registry", str(tmp_path / "registry.sqlite3"),
    ]
    for report in reports:
        arguments.extend(["--fold-report", str(report)])
    assert main(arguments) == 0
    decision = json.loads((tmp_path / "decision.json").read_text(encoding="utf-8"))
    assert decision["status"] == "development_only"
    assert decision["decision_status"] in {"eligible_member_available", "no_eligible_member"}
    assert decision["fold_count"] == len(reports)
    assert len(decision["members"]) == 3
    assert decision["hedge_capture_root_hash"] == manifest["hedge_capture_root_hash"]
    assert decision["hedge_dataset_root_hash"] == manifest["hedge_dataset_root_hash"]
    assert all(set(member["extras_mean"]) == EXTRAS for member in decision["members"])
