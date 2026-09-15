import json
from pathlib import Path
from unittest.mock import patch

from tests.test_funding_xs_fold_run import (
    EXTRAS_KEYS,
    funding_xs_fetch,
    small_funding_xs_config,
)
from tests.test_panel_fold_run import MONTHS, SYMBOLS
from trading_bot.cli import main
from trading_bot.funding_xs_config import FUNDING_XS_MEMBER_NAMES


def test_capture_manifest_funding_xs_folds_and_decision_compose(tmp_path: Path) -> None:
    config_path = small_funding_xs_config(tmp_path)
    with patch(
        "trading_bot.panel_capture.PanelZipClient.fetch", side_effect=funding_xs_fetch()
    ):
        assert (
            main(
                [
                    "panel-capture",
                    "--workspace-root",
                    str(tmp_path),
                    "--output",
                    str(tmp_path / "capture"),
                    "--reserve-bytes",
                    "0",
                    "--symbols",
                    ",".join(SYMBOLS),
                    "--months",
                    ",".join(MONTHS),
                ]
            )
            == 0
        )
    assert (
        main(
            [
                "panel-manifest",
                "--workspace-root",
                str(tmp_path),
                "--capture",
                str(tmp_path / "capture"),
                "--output",
                str(tmp_path / "manifest.json"),
                "--family-spec",
                str(config_path),
            ]
        )
        == 0
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["family_name"] == "funding_xs_panel_v1"

    reports = []
    for fold in manifest["folds"]:
        output = tmp_path / f"fold{fold['fold_index']}.json"
        assert (
            main(
                [
                    "funding-xs-fold",
                    "--workspace-root",
                    str(tmp_path),
                    "--capture",
                    str(tmp_path / "capture"),
                    "--manifest",
                    str(tmp_path / "manifest.json"),
                    "--family-spec",
                    str(config_path),
                    "--output",
                    str(output),
                    "--registry",
                    str(tmp_path / "registry.sqlite3"),
                    "--fold-index",
                    str(fold["fold_index"]),
                ]
            )
            == 0
        )
        reports.append(output)

    arguments = [
        "panel-decision",
        "--workspace-root",
        str(tmp_path),
        "--family-spec",
        str(config_path),
        "--output",
        str(tmp_path / "decision.json"),
        "--registry",
        str(tmp_path / "registry.sqlite3"),
    ]
    for report in reports:
        arguments.extend(["--fold-report", str(report)])
    assert main(arguments) == 0

    decision = json.loads((tmp_path / "decision.json").read_text(encoding="utf-8"))
    assert decision["status"] == "development_only"
    assert decision["family_name"] == "funding_xs_panel_v1"
    assert decision["decision_status"] in {"eligible_member_available", "no_eligible_member"}
    assert decision["fold_count"] == len(reports)
    assert [member["candidate_name"] for member in decision["members"]] == list(
        FUNDING_XS_MEMBER_NAMES
    )
    assert all(set(member["extras_mean"]) == EXTRAS_KEYS for member in decision["members"])
