"""Command-line entry point for bounded local research runs."""

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from trading_bot.backtest import BacktestCase, BacktestRunner, write_report
from trading_bot.binance_cost_journal import (
    TARGET_ROUNDS,
    BinanceCostJournalSpecError,
    JournalStatus,
    create_journal,
    finalize_journal,
    iso_utc_time,
    journal_status,
    load_journal_spec,
    public_binance_json_array_fetcher,
    public_binance_json_fetcher,
    run_journal,
)
from trading_bot.binance_measurement_journal import (
    BinanceMeasurementJournalSpecError,
    MeasurementStatus,
    create_measurement_journal,
    load_measurement_journal_spec,
    measurement_status,
    run_measurement_journal,
    snapshot_measurement_journal,
    verify_measurement_journal,
)
from trading_bot.carry_fold_run import run_carry_fold
from trading_bot.carry_holdout_run import run_carry_holdout
from trading_bot.carry_measured_costs import (
    DEFAULT_BASE_DECLARATION,
    MeasuredCostError,
    declare_measured_cost_family,
    verify_measured_declaration,
)
from trading_bot.features import MarketState
from trading_bot.fold_evaluation import run_fold_evaluation
from trading_bot.funding_xs_fold_run import run_funding_xs_fold
from trading_bot.market_capture import capture_public_candle_history, capture_public_candles
from trading_bot.panel_capture import PanelZipClient, capture_panel, repair_panel_capture
from trading_bot.panel_config import load_family_spec
from trading_bot.panel_decision import build_panel_decision
from trading_bot.panel_fold_run import run_panel_fold
from trading_bot.panel_samples import publish_panel_walk_forward
from trading_bot.research_run import run_capture_research
from trading_bot.shadow_book import ShadowBookError, run_shadow_week
from trading_bot.shadow_capture import ShadowCaptureError, build_shadow_capture
from trading_bot.shadow_config import ShadowDeclarationError
from trading_bot.storage import StoragePolicy, StoragePolicyError, StorageReserveError
from trading_bot.strategy import CostScenario
from trading_bot.trend_fold_run import run_trend_fold
from trading_bot.walk_forward_run import derive_walk_forward_config, run_capture_walk_forward


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trading-research")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo-backtest")
    demo.add_argument("--workspace-root", type=Path, default=Path.cwd())
    demo.add_argument("--output", type=Path, default=Path("artifacts/demo-report.json"))
    demo.add_argument("--reserve-bytes", type=int, default=20_000_000_000)
    capture = commands.add_parser("capture-public-candles")
    capture.add_argument("--workspace-root", type=Path, default=Path.cwd())
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--reserve-bytes", type=int, default=20_000_000_000)
    capture.add_argument("--limit", type=int, default=96)
    history = commands.add_parser("capture-history")
    history.add_argument("--workspace-root", type=Path, default=Path.cwd())
    history.add_argument("--output", type=Path, required=True)
    history.add_argument("--reserve-bytes", type=int, default=20_000_000_000)
    history.add_argument("--bars", type=int, default=2880)
    history.add_argument("--end-time-ms", type=int)
    research = commands.add_parser("research-capture")
    research.add_argument("--workspace-root", type=Path, default=Path.cwd())
    research.add_argument("--capture", type=Path, required=True)
    research.add_argument("--output", type=Path, required=True)
    research.add_argument("--oos-fraction", default="0.20")
    research.add_argument("--minimum-train-samples", type=int, default=20)
    research.add_argument("--random-seed", type=int, default=17)
    walk_forward = commands.add_parser("walk-forward-manifest")
    walk_forward.add_argument("--workspace-root", type=Path, default=Path.cwd())
    walk_forward.add_argument("--capture", type=Path, required=True)
    walk_forward.add_argument("--output", type=Path, required=True)
    walk_forward.add_argument("--train-duration-ns", type=int, required=True)
    walk_forward.add_argument("--validation-duration-ns", type=int, required=True)
    walk_forward.add_argument("--test-duration-ns", type=int, required=True)
    walk_forward.add_argument("--step-ns", type=int, required=True)
    walk_forward.add_argument("--embargo-ns", type=int, required=True)
    walk_forward.add_argument("--holdout-duration-ns", type=int, required=True)
    walk_forward.add_argument("--horizon-bars", type=int, default=1)
    fold_evaluation = commands.add_parser("evaluate-fold")
    fold_evaluation.add_argument("--workspace-root", type=Path, default=Path.cwd())
    fold_evaluation.add_argument("--capture", type=Path, required=True)
    fold_evaluation.add_argument("--split-manifest", type=Path, required=True)
    fold_evaluation.add_argument("--output", type=Path, required=True)
    fold_evaluation.add_argument("--registry", type=Path, required=True)
    fold_evaluation.add_argument("--fold-index", type=int, required=True)
    fold_evaluation.add_argument("--random-seed", type=int, default=17)
    fold_evaluation.add_argument("--block-length", type=int, default=16)
    fold_evaluation.add_argument("--bootstrap-repetitions", type=int, default=2000)
    panel_capture = commands.add_parser("panel-capture")
    panel_capture.add_argument("--workspace-root", type=Path, default=Path.cwd())
    panel_capture.add_argument("--output", type=Path, required=True)
    panel_capture.add_argument("--reserve-bytes", type=int, default=20_000_000_000)
    panel_capture.add_argument("--symbols", required=True, help="comma-separated symbol list")
    panel_capture.add_argument(
        "--months",
        default=None,
        help="comma-separated YYYY-MM list; omit to discover available months per symbol",
    )
    panel_capture.add_argument(
        "--month-from", default=None, help="earliest YYYY-MM to keep when discovering months"
    )
    panel_capture.add_argument(
        "--month-to", default=None, help="latest YYYY-MM to keep when discovering months"
    )
    panel_capture.add_argument("--market", choices=("um", "spot"), default="um")
    panel_capture_repair = commands.add_parser("panel-capture-repair")
    panel_capture_repair.add_argument("--workspace-root", type=Path, default=Path.cwd())
    panel_capture_repair.add_argument("--source-capture", type=Path, required=True)
    panel_capture_repair.add_argument("--output", type=Path, required=True)
    panel_capture_repair.add_argument("--reserve-bytes", type=int, default=20_000_000_000)
    panel_manifest = commands.add_parser("panel-manifest")
    panel_manifest.add_argument("--workspace-root", type=Path, default=Path.cwd())
    panel_manifest.add_argument("--capture", type=Path, required=True)
    panel_manifest.add_argument("--output", type=Path, required=True)
    panel_manifest.add_argument("--family-spec", type=Path, required=True)
    panel_manifest.add_argument("--hedge-capture", type=Path, default=None)
    panel_fold = commands.add_parser("panel-fold")
    panel_fold.add_argument("--workspace-root", type=Path, default=Path.cwd())
    panel_fold.add_argument("--capture", type=Path, required=True)
    panel_fold.add_argument("--manifest", type=Path, required=True)
    panel_fold.add_argument("--family-spec", type=Path, required=True)
    panel_fold.add_argument("--output", type=Path, required=True)
    panel_fold.add_argument("--registry", type=Path, required=True)
    panel_fold.add_argument("--fold-index", type=int, required=True)
    panel_decision = commands.add_parser("panel-decision")
    panel_decision.add_argument("--workspace-root", type=Path, default=Path.cwd())
    panel_decision.add_argument("--fold-report", type=Path, action="append", required=True)
    panel_decision.add_argument("--family-spec", type=Path, required=True)
    panel_decision.add_argument("--output", type=Path, required=True)
    panel_decision.add_argument("--registry", type=Path, required=True)
    carry_fold = commands.add_parser("carry-fold")
    carry_fold.add_argument("--workspace-root", type=Path, default=Path.cwd())
    carry_fold.add_argument("--capture", type=Path, required=True)
    carry_fold.add_argument("--hedge-capture", type=Path, required=True)
    carry_fold.add_argument("--manifest", type=Path, required=True)
    carry_fold.add_argument("--family-spec", type=Path, required=True)
    carry_fold.add_argument("--output", type=Path, required=True)
    carry_fold.add_argument("--registry", type=Path, required=True)
    carry_fold.add_argument("--fold-index", type=int, required=True)
    carry_holdout = commands.add_parser("carry-holdout")
    carry_holdout.add_argument("--workspace-root", type=Path, default=Path.cwd())
    carry_holdout.add_argument("--capture", type=Path, required=True)
    carry_holdout.add_argument("--hedge-capture", type=Path, required=True)
    carry_holdout.add_argument("--original-capture", type=Path, required=True)
    carry_holdout.add_argument("--original-hedge-capture", type=Path, required=True)
    carry_holdout.add_argument("--manifest", type=Path, required=True)
    carry_holdout.add_argument("--family-spec", type=Path, required=True)
    carry_holdout.add_argument("--decision", type=Path, required=True)
    carry_holdout.add_argument("--fold-report", type=Path, action="append", required=True)
    carry_holdout.add_argument("--output", type=Path, required=True)
    carry_holdout.add_argument("--registry", type=Path, required=True)
    carry_holdout.add_argument("--check-only", action="store_true")
    carry_declare = commands.add_parser("carry-declare-measured")
    carry_declare.add_argument("--workspace-root", type=Path, default=Path.cwd())
    carry_declare.add_argument("--receipt", type=Path, required=True)
    carry_declare.add_argument("--base-config", type=Path, required=True)
    carry_declare.add_argument("--output", type=Path, required=True)
    carry_verify = commands.add_parser("carry-verify-measured")
    carry_verify.add_argument("--workspace-root", type=Path, default=Path.cwd())
    carry_verify.add_argument("--receipt", type=Path, required=True)
    carry_verify.add_argument("--spec", type=Path, required=True)
    carry_verify.add_argument("--base-config", type=Path, default=None)
    trend_fold = commands.add_parser("trend-fold")
    trend_fold.add_argument("--workspace-root", type=Path, default=Path.cwd())
    trend_fold.add_argument("--capture", type=Path, required=True)
    trend_fold.add_argument("--manifest", type=Path, required=True)
    trend_fold.add_argument("--family-spec", type=Path, required=True)
    trend_fold.add_argument("--output", type=Path, required=True)
    trend_fold.add_argument("--registry", type=Path, required=True)
    trend_fold.add_argument("--fold-index", type=int, required=True)
    funding_xs_fold = commands.add_parser("funding-xs-fold")
    funding_xs_fold.add_argument("--workspace-root", type=Path, default=Path.cwd())
    funding_xs_fold.add_argument("--capture", type=Path, required=True)
    funding_xs_fold.add_argument("--manifest", type=Path, required=True)
    funding_xs_fold.add_argument("--family-spec", type=Path, required=True)
    funding_xs_fold.add_argument("--output", type=Path, required=True)
    funding_xs_fold.add_argument("--registry", type=Path, required=True)
    funding_xs_fold.add_argument("--fold-index", type=int, required=True)
    journal_create = commands.add_parser("binance-cost-journal-create")
    journal_create.add_argument("--workspace-root", type=Path, default=Path.cwd())
    journal_create.add_argument("--journal", type=Path, required=True)
    journal_create.add_argument("--run-id", required=True)
    journal_create.add_argument("--perp-capture", type=Path, required=True)
    journal_create.add_argument("--spot-capture", type=Path, required=True)
    journal_create.add_argument("--family-spec", type=Path, required=True)
    journal_create.add_argument("--reserve-bytes", type=int, default=10_000_000_000)
    journal_create.add_argument("--allow-short-sample", action="store_true")
    journal_run = commands.add_parser("binance-cost-journal-run")
    journal_run.add_argument("--workspace-root", type=Path, default=Path.cwd())
    journal_run.add_argument("--journal", type=Path, required=True)
    journal_run.add_argument("--rounds", type=int, default=TARGET_ROUNDS)
    journal_run.add_argument("--reserve-bytes", type=int, default=10_000_000_000)
    journal_finalize = commands.add_parser("binance-cost-journal-finalize")
    journal_finalize.add_argument("--workspace-root", type=Path, default=Path.cwd())
    journal_finalize.add_argument("--journal", type=Path, required=True)
    journal_finalize.add_argument("--output", type=Path, required=True)
    journal_finalize.add_argument("--reserve-bytes", type=int, default=10_000_000_000)
    journal_status_command = commands.add_parser("binance-cost-journal-status")
    journal_status_command.add_argument("--workspace-root", type=Path, default=Path.cwd())
    journal_status_command.add_argument("--journal", type=Path, required=True)
    journal_status_command.add_argument("--last", type=int, default=60)
    shadow_capture = commands.add_parser("shadow-capture")
    shadow_capture.add_argument("--workspace-root", type=Path, default=Path.cwd())
    shadow_capture.add_argument("--base-capture", type=Path, required=True)
    shadow_capture.add_argument("--output", type=Path, required=True)
    shadow_capture.add_argument(
        "--tail-through", required=True, help="the last Sunday to cover, YYYY-MM-DD"
    )
    shadow_capture.add_argument("--market", choices=("um", "spot"), required=True)
    shadow_capture.add_argument(
        "--previous-capture", type=Path, default=None, help="last week's shadow capture"
    )
    shadow_capture.add_argument("--reserve-bytes", type=int, default=20_000_000_000)
    shadow_week = commands.add_parser("shadow-week")
    shadow_week.add_argument("--workspace-root", type=Path, default=Path.cwd())
    shadow_week.add_argument("--declaration", type=Path, required=True)
    shadow_week.add_argument("--capture", type=Path, required=True)
    shadow_week.add_argument("--hedge-capture", type=Path, required=True)
    shadow_week.add_argument("--decision-sunday", required=True)
    shadow_week.add_argument("--holdout-report", type=Path, default=None)
    shadow_week.add_argument("--measurement-snapshot", type=Path, default=None)
    shadow_week.add_argument(
        "--perp-base-capture",
        type=Path,
        default=None,
        help="the base the weekly perpetual capture extends (ruling 17)",
    )
    shadow_week.add_argument("--spot-base-capture", type=Path, default=None)
    shadow_week.add_argument("--reserve-bytes", type=int, default=10_000_000_000)
    measurement_create = commands.add_parser("binance-measurement-journal-create")
    measurement_create.add_argument("--workspace-root", type=Path, default=Path.cwd())
    measurement_create.add_argument("--journal", type=Path, required=True)
    measurement_create.add_argument("--run-id", required=True)
    measurement_create.add_argument("--cost-journal", type=Path, required=True)
    measurement_create.add_argument("--reserve-bytes", type=int, default=10_000_000_000)
    measurement_run = commands.add_parser("binance-measurement-journal-run")
    measurement_run.add_argument("--workspace-root", type=Path, default=Path.cwd())
    measurement_run.add_argument("--journal", type=Path, required=True)
    measurement_run.add_argument(
        "--rounds", type=int, default=None, help="omit to sample until the process is stopped"
    )
    measurement_run.add_argument("--reserve-bytes", type=int, default=10_000_000_000)
    measurement_status_command = commands.add_parser("binance-measurement-journal-status")
    measurement_status_command.add_argument("--workspace-root", type=Path, default=Path.cwd())
    measurement_status_command.add_argument("--journal", type=Path, required=True)
    measurement_status_command.add_argument("--last", type=int, default=60)
    measurement_verify = commands.add_parser("binance-measurement-journal-verify")
    measurement_verify.add_argument("--workspace-root", type=Path, default=Path.cwd())
    measurement_verify.add_argument("--journal", type=Path, required=True)
    measurement_snapshot = commands.add_parser("binance-measurement-journal-snapshot")
    measurement_snapshot.add_argument("--workspace-root", type=Path, default=Path.cwd())
    measurement_snapshot.add_argument("--journal", type=Path, required=True)
    measurement_snapshot.add_argument("--output", type=Path, required=True)
    measurement_snapshot.add_argument(
        "--window-start", required=True, help="ISO-8601 UTC, YYYY-MM-DDTHH:MM:SSZ"
    )
    measurement_snapshot.add_argument(
        "--window-end", required=True, help="ISO-8601 UTC, YYYY-MM-DDTHH:MM:SSZ"
    )
    measurement_snapshot.add_argument("--reserve-bytes", type=int, default=10_000_000_000)
    parsed = parser.parse_args(arguments)

    if parsed.command == "demo-backtest":
        workspace = parsed.workspace_root.resolve()
        output = parsed.output
        if not output.is_absolute():
            output = workspace / output
        output = output.resolve()
        free_bytes = shutil.disk_usage(workspace).free
        StoragePolicy(workspace, parsed.reserve_bytes).authorize(
            target=output,
            temporary_directory=output.parent,
            free_bytes=free_bytes,
            worst_case_required_bytes=1_000_000,
        )
        report = _demo_runner().run(_demo_case(), baseline_name="momentum")
        write_report(output, report)
        return 0
    if parsed.command == "capture-public-candles":
        workspace = parsed.workspace_root.resolve()
        output = parsed.output
        if not output.is_absolute():
            output = workspace / output
        capture_public_candles(
            workspace_root=workspace,
            output_directory=output,
            reserve_bytes=parsed.reserve_bytes,
            limit=parsed.limit,
        )
        return 0
    if parsed.command == "capture-history":
        workspace = parsed.workspace_root.resolve()
        output = parsed.output
        if not output.is_absolute():
            output = workspace / output
        capture_public_candle_history(
            workspace_root=workspace,
            output_directory=output,
            reserve_bytes=parsed.reserve_bytes,
            bars=parsed.bars,
            end_time_ms=parsed.end_time_ms,
        )
        return 0
    if parsed.command == "research-capture":
        workspace = parsed.workspace_root.resolve()
        capture_root = parsed.capture.resolve()
        output = parsed.output.resolve()
        if not capture_root.is_relative_to(workspace) or not output.is_relative_to(workspace):
            raise ValueError("research paths must stay inside workspace")
        run_capture_research(
            capture_root,
            output_path=output,
            oos_fraction=parsed.oos_fraction,
            minimum_train_samples=parsed.minimum_train_samples,
            random_seed=parsed.random_seed,
        )
        return 0
    if parsed.command == "walk-forward-manifest":
        workspace = parsed.workspace_root.resolve()
        capture_root = parsed.capture.resolve()
        output = parsed.output.resolve()
        if not capture_root.is_relative_to(workspace) or not output.is_relative_to(workspace):
            raise ValueError("walk-forward paths must stay inside workspace")
        config = derive_walk_forward_config(
            capture_root,
            horizon_bars=parsed.horizon_bars,
            train_duration_ns=parsed.train_duration_ns,
            validation_duration_ns=parsed.validation_duration_ns,
            test_duration_ns=parsed.test_duration_ns,
            step_ns=parsed.step_ns,
            embargo_ns=parsed.embargo_ns,
            holdout_duration_ns=parsed.holdout_duration_ns,
        )
        run_capture_walk_forward(
            capture_root,
            output_path=output,
            config=config,
            horizon_bars=parsed.horizon_bars,
        )
        return 0
    if parsed.command == "evaluate-fold":
        workspace = parsed.workspace_root.resolve()
        paths = (
            parsed.capture.resolve(),
            parsed.split_manifest.resolve(),
            parsed.output.resolve(),
            parsed.registry.resolve(),
        )
        if any(not path.is_relative_to(workspace) for path in paths):
            raise ValueError("fold evaluation paths must stay inside workspace")
        run_fold_evaluation(
            paths[0],
            split_manifest_path=paths[1],
            output_path=paths[2],
            registry_path=paths[3],
            fold_index=parsed.fold_index,
            random_seed=parsed.random_seed,
            block_length=parsed.block_length,
            bootstrap_repetitions=parsed.bootstrap_repetitions,
        )
        return 0
    if parsed.command == "panel-capture":
        workspace = parsed.workspace_root.resolve()
        output = parsed.output.resolve()
        if not output.is_relative_to(workspace):
            raise ValueError("panel capture paths must stay inside workspace")
        months = (
            tuple(item for item in parsed.months.split(",") if item)
            if parsed.months is not None
            else None
        )
        capture_panel(
            workspace_root=workspace,
            output_directory=output,
            reserve_bytes=parsed.reserve_bytes,
            symbols=tuple(item for item in parsed.symbols.split(",") if item),
            months=months,
            month_from=parsed.month_from,
            month_to=parsed.month_to,
            market=parsed.market,
        )
        return 0
    if parsed.command == "panel-capture-repair":
        workspace = parsed.workspace_root.resolve()
        source_capture = parsed.source_capture.resolve()
        output = parsed.output.resolve()
        if not source_capture.is_relative_to(workspace) or not output.is_relative_to(workspace):
            raise ValueError("panel capture repair paths must stay inside workspace")
        repair_panel_capture(
            workspace_root=workspace,
            source_capture_root=source_capture,
            output_directory=output,
            reserve_bytes=parsed.reserve_bytes,
        )
        return 0
    if parsed.command == "panel-manifest":
        workspace = parsed.workspace_root.resolve()
        manifest_paths = (
            parsed.capture.resolve(),
            parsed.output.resolve(),
            parsed.family_spec.resolve(),
        )
        if any(not path.is_relative_to(workspace) for path in manifest_paths):
            raise ValueError("panel manifest paths must stay inside workspace")
        hedge_capture_root = (
            parsed.hedge_capture.resolve() if parsed.hedge_capture is not None else None
        )
        if hedge_capture_root is not None and not hedge_capture_root.is_relative_to(workspace):
            raise ValueError("panel manifest paths must stay inside workspace")
        spec, spec_hash = load_family_spec(manifest_paths[2])
        publish_panel_walk_forward(
            manifest_paths[0],
            output_path=manifest_paths[1],
            spec=spec,
            family_spec_hash=spec_hash,
            hedge_capture_root=hedge_capture_root,
        )
        return 0
    if parsed.command == "panel-fold":
        workspace = parsed.workspace_root.resolve()
        fold_paths = (
            parsed.capture.resolve(),
            parsed.manifest.resolve(),
            parsed.family_spec.resolve(),
            parsed.output.resolve(),
            parsed.registry.resolve(),
        )
        if any(not path.is_relative_to(workspace) for path in fold_paths):
            raise ValueError("panel fold paths must stay inside workspace")
        run_panel_fold(
            fold_paths[0],
            manifest_path=fold_paths[1],
            family_spec_path=fold_paths[2],
            output_path=fold_paths[3],
            registry_path=fold_paths[4],
            fold_index=parsed.fold_index,
        )
        return 0
    if parsed.command == "panel-decision":
        workspace = parsed.workspace_root.resolve()
        reports = tuple(path.resolve() for path in parsed.fold_report)
        decision_paths = (
            *reports,
            parsed.family_spec.resolve(),
            parsed.output.resolve(),
            parsed.registry.resolve(),
        )
        if any(not path.is_relative_to(workspace) for path in decision_paths):
            raise ValueError("panel decision paths must stay inside workspace")
        build_panel_decision(
            reports,
            family_spec_path=parsed.family_spec.resolve(),
            output_path=parsed.output.resolve(),
            registry_path=parsed.registry.resolve(),
        )
        return 0
    if parsed.command == "carry-fold":
        workspace = parsed.workspace_root.resolve()
        carry_paths = (
            parsed.capture.resolve(),
            parsed.hedge_capture.resolve(),
            parsed.manifest.resolve(),
            parsed.family_spec.resolve(),
            parsed.output.resolve(),
            parsed.registry.resolve(),
        )
        if any(not path.is_relative_to(workspace) for path in carry_paths):
            raise ValueError("carry fold paths must stay inside workspace")
        run_carry_fold(
            carry_paths[0],
            carry_paths[1],
            manifest_path=carry_paths[2],
            family_spec_path=carry_paths[3],
            output_path=carry_paths[4],
            registry_path=carry_paths[5],
            fold_index=parsed.fold_index,
        )
        return 0
    if parsed.command == "carry-holdout":
        return _carry_holdout(parsed)
    if parsed.command == "carry-declare-measured":
        return _carry_declare_measured(parsed)
    if parsed.command == "carry-verify-measured":
        return _carry_verify_measured(parsed)
    if parsed.command == "trend-fold":
        workspace = parsed.workspace_root.resolve()
        trend_paths = (
            parsed.capture.resolve(),
            parsed.manifest.resolve(),
            parsed.family_spec.resolve(),
            parsed.output.resolve(),
            parsed.registry.resolve(),
        )
        if any(not path.is_relative_to(workspace) for path in trend_paths):
            raise ValueError("trend fold paths must stay inside workspace")
        run_trend_fold(
            trend_paths[0],
            manifest_path=trend_paths[1],
            family_spec_path=trend_paths[2],
            output_path=trend_paths[3],
            registry_path=trend_paths[4],
            fold_index=parsed.fold_index,
        )
        return 0
    if parsed.command == "funding-xs-fold":
        workspace = parsed.workspace_root.resolve()
        funding_xs_paths = (
            parsed.capture.resolve(),
            parsed.manifest.resolve(),
            parsed.family_spec.resolve(),
            parsed.output.resolve(),
            parsed.registry.resolve(),
        )
        if any(not path.is_relative_to(workspace) for path in funding_xs_paths):
            raise ValueError("funding cross-section fold paths must stay inside workspace")
        run_funding_xs_fold(
            funding_xs_paths[0],
            manifest_path=funding_xs_paths[1],
            family_spec_path=funding_xs_paths[2],
            output_path=funding_xs_paths[3],
            registry_path=funding_xs_paths[4],
            fold_index=parsed.fold_index,
        )
        return 0
    if parsed.command == "binance-cost-journal-create":
        return _binance_cost_journal_create(parsed)
    if parsed.command == "binance-cost-journal-run":
        return _binance_cost_journal_run(parsed)
    if parsed.command == "binance-cost-journal-finalize":
        return _binance_cost_journal_finalize(parsed)
    if parsed.command == "binance-cost-journal-status":
        return _binance_cost_journal_status(parsed)
    if parsed.command == "shadow-capture":
        return _shadow_capture(parsed)
    if parsed.command == "shadow-week":
        return _shadow_week(parsed)
    if parsed.command == "binance-measurement-journal-create":
        return _binance_measurement_journal_create(parsed)
    if parsed.command == "binance-measurement-journal-run":
        return _binance_measurement_journal_run(parsed)
    if parsed.command == "binance-measurement-journal-status":
        return _binance_measurement_journal_status(parsed)
    if parsed.command == "binance-measurement-journal-verify":
        return _binance_measurement_journal_verify(parsed)
    if parsed.command == "binance-measurement-journal-snapshot":
        return _binance_measurement_journal_snapshot(parsed)
    raise AssertionError("unreachable command")


def _carry_holdout(parsed: argparse.Namespace) -> int:
    """Open one carry family's final holdout, once (protocol 16.2).

    Zero on either verdict: a failed holdout is a result the artifact records,
    not a command failure. Every refusal -- a path outside the workspace, a
    capture that is not a superset, a family whose holdout was already read --
    raises, so a supervisor sees a non-zero exit and no artifact.

    `--check-only` is the dry run: the same checks, the same refusals, one line
    naming the candidate the command derived, and no artifact. A holdout read
    cannot be retried, so readiness is confirmed with this flag rather than by
    running the read and finding out.
    """
    workspace: Path = parsed.workspace_root.resolve()
    reports = tuple(path.resolve() for path in parsed.fold_report)
    holdout_paths = (
        parsed.capture.resolve(),
        parsed.hedge_capture.resolve(),
        parsed.original_capture.resolve(),
        parsed.original_hedge_capture.resolve(),
        parsed.manifest.resolve(),
        parsed.family_spec.resolve(),
        parsed.decision.resolve(),
        parsed.output.resolve(),
        parsed.registry.resolve(),
        *reports,
    )
    if any(not path.is_relative_to(workspace) for path in holdout_paths):
        raise ValueError("carry holdout paths must stay inside workspace")
    artifact = run_carry_holdout(
        holdout_paths[0],
        holdout_paths[1],
        original_perp_capture_root=holdout_paths[2],
        original_spot_capture_root=holdout_paths[3],
        manifest_path=holdout_paths[4],
        family_spec_path=holdout_paths[5],
        decision_path=holdout_paths[6],
        fold_report_paths=reports,
        output_path=holdout_paths[7],
        registry_path=holdout_paths[8],
        check_only=parsed.check_only,
    )
    if parsed.check_only:
        print(f"carry holdout check: candidate {artifact.candidate_name} ready")
        return 0
    print(f"carry holdout written: {artifact.output_path}")
    print(f"candidate: {artifact.candidate_name}")
    print(f"verdict: {artifact.verdict}")
    return 0


def _carry_declare_measured(parsed: argparse.Namespace) -> int:
    """Write the carry declaration a finalisation receipt's measured tiers imply.

    2 is what a rerun cannot fix - a refused reading of the receipt, a base
    declaration whose family is not a key of `MEASURED_FAMILY_BY_BASE` (v2 or
    v4), an output that already exists, a path outside the workspace - and 1
    anything else.
    """
    workspace: Path = parsed.workspace_root.resolve()
    receipt: Path = parsed.receipt.resolve()
    base: Path = parsed.base_config.resolve()
    output: Path = parsed.output.resolve()
    if any(not path.is_relative_to(workspace) for path in (receipt, base, output)):
        return _journal_failure("measured declaration paths must stay inside workspace", 2)
    try:
        written, spec_hash = declare_measured_cost_family(
            receipt_path=receipt, base_declaration_path=base, output_path=output
        )
    except MeasuredCostError as error:
        return _journal_failure(f"{type(error).__name__}: {error}", 2)
    except Exception as error:  # the operator reads the code, not the traceback
        return _journal_failure(f"{type(error).__name__}: {error}", 1)
    print(f"measured carry declaration written: {written}")
    print(f"family spec hash: {spec_hash}")
    return 0


def _carry_verify_measured(parsed: argparse.Namespace) -> int:
    """Re-derive an existing measured declaration from its receipt.

    2 is a declaration that does not stand up - a receipt that does not
    recompute, a citation naming another receipt, a number that is not the
    measurement, a field that is not the base declaration's, a path outside the
    workspace - and 1 anything else.
    """
    workspace: Path = parsed.workspace_root.resolve()
    receipt: Path = parsed.receipt.resolve()
    spec: Path = parsed.spec.resolve()
    base: Path = (parsed.base_config or DEFAULT_BASE_DECLARATION).resolve()
    if any(not path.is_relative_to(workspace) for path in (receipt, spec, base)):
        return _journal_failure("measured declaration paths must stay inside workspace", 2)
    try:
        spec_hash = verify_measured_declaration(
            receipt_path=receipt, spec_path=spec, base_declaration_path=base
        )
    except MeasuredCostError as error:
        return _journal_failure(f"{type(error).__name__}: {error}", 2)
    except Exception as error:  # the operator reads the code, not the traceback
        return _journal_failure(f"{type(error).__name__}: {error}", 1)
    print(f"measured carry declaration verified: {spec}")
    print(f"family spec hash: {spec_hash}")
    return 0


def _binance_cost_journal_create(parsed: argparse.Namespace) -> int:
    workspace: Path = parsed.workspace_root.resolve()
    journal: Path = parsed.journal.resolve()
    perpetual: Path = parsed.perp_capture.resolve()
    spot: Path = parsed.spot_capture.resolve()
    family_spec: Path = parsed.family_spec.resolve()
    paths = (journal, perpetual, spot, family_spec)
    if any(not path.is_relative_to(workspace) for path in paths):
        return _journal_failure("binance cost journal paths must stay inside workspace", 2)
    try:
        create_journal(
            workspace_root=workspace,
            journal_root=journal,
            reserve_bytes=parsed.reserve_bytes,
            run_id=parsed.run_id,
            perp_capture_root=perpetual,
            spot_capture_root=spot,
            family_spec_path=family_spec,
            allow_short_sample=parsed.allow_short_sample,
        )
        spec, _ = load_journal_spec(journal)
    except Exception as error:  # the supervisor reads the code, not the traceback
        return _journal_failure(f"{type(error).__name__}: {error}", _journal_exit_code(error))
    decision = datetime.fromtimestamp(
        spec.sample_decision_close_ns // 1_000_000_000, tz=UTC
    ).strftime("%Y-%m-%d")
    print(
        f"binance cost journal created: {len(spec.instruments)} instruments "
        f"sampled at the {decision} decision"
    )
    return 0


def _binance_cost_journal_run(parsed: argparse.Namespace) -> int:
    workspace: Path = parsed.workspace_root.resolve()
    journal: Path = parsed.journal.resolve()
    if not journal.is_relative_to(workspace):
        return _journal_failure("binance cost journal paths must stay inside workspace", 2)
    try:
        run_journal(
            workspace_root=workspace,
            journal_root=journal,
            reserve_bytes=parsed.reserve_bytes,
            rounds=parsed.rounds,
            fetcher=public_binance_json_fetcher,
        )
    except Exception as error:  # the supervisor reads the code, not the traceback
        return _journal_failure(f"{type(error).__name__}: {error}", _journal_exit_code(error))
    return 0


def _binance_cost_journal_finalize(parsed: argparse.Namespace) -> int:
    workspace: Path = parsed.workspace_root.resolve()
    journal: Path = parsed.journal.resolve()
    output: Path = parsed.output.resolve()
    if any(not path.is_relative_to(workspace) for path in (journal, output)):
        return _journal_failure("binance cost journal paths must stay inside workspace", 2)
    try:
        receipt = finalize_journal(
            workspace_root=workspace,
            journal_root=journal,
            output_path=output,
            reserve_bytes=parsed.reserve_bytes,
        )
    except Exception as error:  # the supervisor reads the code, not the traceback
        return _journal_failure(f"{type(error).__name__}: {error}", _journal_exit_code(error))
    # The hash a carry declaration cites, read back off the published bytes.
    document = json.loads(receipt.read_text(encoding="utf-8"))
    print(f"binance cost journal finalised: {receipt}")
    print(f"receipt content hash: {document['content_hash']}")
    return 0


def _binance_cost_journal_status(parsed: argparse.Namespace) -> int:
    """Report a running journal's tip; 0 when its last segments verify, 1 when not."""
    workspace: Path = parsed.workspace_root.resolve()
    journal: Path = parsed.journal.resolve()
    if not journal.is_relative_to(workspace):
        return _journal_failure("binance cost journal paths must stay inside workspace", 2)
    try:
        status = journal_status(journal, last=parsed.last)
    except Exception as error:  # the supervisor reads the code, not the traceback
        return _journal_failure(f"{type(error).__name__}: {error}", _journal_exit_code(error))
    _print_journal_status(status, last=parsed.last)
    return 0 if status.verified else 1


def _print_journal_status(status: JournalStatus, *, last: int) -> None:
    print(f"segment_count: {status.segment_count}")
    print(f"last_sequence: {status.last_sequence}")
    stamp = status.last_received_time_ns
    print(f"last_received_time: {'-' if stamp is None else iso_utc_time(stamp)}")
    print(f"window_segments: {status.window_segments} (last {last})")
    period = status.mean_period_seconds
    print(f"mean_period_seconds: {'-' if period is None else period}")
    for instrument in status.instruments:
        ok_rate = "-" if instrument.ok_rate is None else str(instrument.ok_rate)
        degraded = instrument.premium_index_reason_rate
        print(
            f"{instrument.instrument_id}: ok {ok_rate} "
            f"premium_index_reason {'-' if degraded is None else degraded}"
        )
    print(f"verify: {'ok' if status.verified else ','.join(status.reasons)}")


def _shadow_capture(parsed: argparse.Namespace) -> int:
    """Publish one weekly shadow capture over a verified base (spec section 3).

    `--previous-capture` is last week's capture of the same market: its tail
    rows are carried rather than refetched, and the ones a monthly dump has
    since covered are reconciled against that dump before they are dropped.
    The first week of a chain is run without it.
    """
    workspace: Path = parsed.workspace_root.resolve()
    base: Path = parsed.base_capture.resolve()
    output: Path = parsed.output.resolve()
    previous: Path | None = _resolved(parsed.previous_capture)
    if _outside(workspace, base, output, previous):
        return _journal_failure("shadow capture paths must stay inside workspace", 2)
    try:
        artifact = build_shadow_capture(
            workspace_root=workspace,
            base_capture_root=base,
            output_directory=output,
            reserve_bytes=parsed.reserve_bytes,
            tail_through=parsed.tail_through,
            market=parsed.market,
            fetch=PanelZipClient().fetch,
            previous_capture_root=previous,
        )
    except Exception as error:  # the supervisor reads the code, not the traceback
        return _journal_failure(f"{type(error).__name__}: {error}", _shadow_exit_code(error))
    print(f"shadow capture written: {artifact.capture_root}")
    print(f"tail_through: {artifact.tail_through}")
    print(f"capture_root_hash: {artifact.capture_root_hash}")
    print(f"dataset_root_hash: {artifact.dataset_root_hash}")
    print(f"reconciled_rows: {artifact.reconciliation['compared_rows']}")
    return 0


def _shadow_week(parsed: argparse.Namespace) -> int:
    """Compute, seal and register one Sunday's shadow book (spec section 4).

    The declaration decides where the artifact and the registry go, so only
    the inputs are named here. `--perp-base-capture` and `--spot-base-capture`
    are required exactly when the weekly capture beside them records a base
    (ruling 17); a monthly capture extends nothing and is run without one.
    `--holdout-report` belongs to Phase B alone and `--measurement-snapshot`
    is the week's cited reading of the measurement stream.
    """
    workspace: Path = parsed.workspace_root.resolve()
    declaration: Path = parsed.declaration.resolve()
    perpetual: Path = parsed.capture.resolve()
    spot: Path = parsed.hedge_capture.resolve()
    holdout_report: Path | None = _resolved(parsed.holdout_report)
    snapshot: Path | None = _resolved(parsed.measurement_snapshot)
    perp_base: Path | None = _resolved(parsed.perp_base_capture)
    spot_base: Path | None = _resolved(parsed.spot_base_capture)
    if _outside(
        workspace, declaration, perpetual, spot, holdout_report, snapshot, perp_base, spot_base
    ):
        return _journal_failure("shadow week paths must stay inside workspace", 2)
    try:
        artifact = run_shadow_week(
            workspace_root=workspace,
            declaration_path=declaration,
            perp_capture_root=perpetual,
            spot_capture_root=spot,
            decision_sunday=parsed.decision_sunday,
            holdout_report_path=holdout_report,
            measurement_snapshot_path=snapshot,
            perp_base_capture_root=perp_base,
            spot_base_capture_root=spot_base,
            reserve_bytes=parsed.reserve_bytes,
        )
    except Exception as error:  # the supervisor reads the code, not the traceback
        return _journal_failure(f"{type(error).__name__}: {error}", _shadow_exit_code(error))
    print(f"shadow week written: {artifact.output_path}")
    print(f"report hash: {artifact.report_hash}")
    print(f"status: {artifact.status}")
    return 0


def _binance_measurement_journal_create(parsed: argparse.Namespace) -> int:
    """Declare the permanent measurement journal from the cost journal's sample."""
    workspace: Path = parsed.workspace_root.resolve()
    journal: Path = parsed.journal.resolve()
    cost_journal: Path = parsed.cost_journal.resolve()
    if _outside(workspace, journal, cost_journal):
        return _journal_failure("binance measurement journal paths must stay inside workspace", 2)
    try:
        create_measurement_journal(
            workspace_root=workspace,
            journal_root=journal,
            reserve_bytes=parsed.reserve_bytes,
            run_id=parsed.run_id,
            cost_journal_root=cost_journal,
        )
        spec, _ = load_measurement_journal_spec(journal)
    except Exception as error:  # the supervisor reads the code, not the traceback
        return _journal_failure(f"{type(error).__name__}: {error}", _shadow_exit_code(error))
    print(f"binance measurement journal created: {spec.run_id}")
    print(
        f"{len(spec.instruments)} instruments every {spec.sample_interval_seconds} s, "
        f"continuing cost journal spec {spec.cost_journal_spec_hash}"
    )
    return 0


def _binance_measurement_journal_run(parsed: argparse.Namespace) -> int:
    """Append rounds to the journal; without `--rounds`, until the process stops."""
    workspace: Path = parsed.workspace_root.resolve()
    journal: Path = parsed.journal.resolve()
    if _outside(workspace, journal):
        return _journal_failure("binance measurement journal paths must stay inside workspace", 2)
    try:
        run_measurement_journal(
            workspace_root=workspace,
            journal_root=journal,
            reserve_bytes=parsed.reserve_bytes,
            rounds=parsed.rounds,
            fetcher=public_binance_json_fetcher,
            array_fetcher=public_binance_json_array_fetcher,
        )
    except Exception as error:  # the supervisor reads the code, not the traceback
        return _journal_failure(f"{type(error).__name__}: {error}", _shadow_exit_code(error))
    return 0


def _binance_measurement_journal_status(parsed: argparse.Namespace) -> int:
    """Report the stream's tip; 0 when its tail verifies, 1 when it does not."""
    workspace: Path = parsed.workspace_root.resolve()
    journal: Path = parsed.journal.resolve()
    if _outside(workspace, journal):
        return _journal_failure("binance measurement journal paths must stay inside workspace", 2)
    try:
        status = measurement_status(journal, last=parsed.last)
    except Exception as error:  # the supervisor reads the code, not the traceback
        return _journal_failure(f"{type(error).__name__}: {error}", _shadow_exit_code(error))
    _print_measurement_status(status, last=parsed.last)
    return 0 if status.verify_ok else 1


def _binance_measurement_journal_verify(parsed: argparse.Namespace) -> int:
    """Walk the whole chain; 0 when it verifies, 1 when it does not.

    The daily liveness check runs this and not `status`: a restart verifies
    the tail alone (ruling 13), so corruption older than the two newest day
    directories sits unnoticed until the whole chain is walked. The walk never
    raises -- an unreadable spec is a reason like any other -- so the only
    refusal here is a path outside the workspace.
    """
    workspace: Path = parsed.workspace_root.resolve()
    journal: Path = parsed.journal.resolve()
    if _outside(workspace, journal):
        return _journal_failure("binance measurement journal paths must stay inside workspace", 2)
    verified, reasons = verify_measurement_journal(journal)
    print(f"verify: {'ok' if verified else ','.join(reasons)}")
    return 0 if verified else 1


def _binance_measurement_journal_snapshot(parsed: argparse.Namespace) -> int:
    """Seal the rounds inside a closed window into an immutable reading."""
    workspace: Path = parsed.workspace_root.resolve()
    journal: Path = parsed.journal.resolve()
    output: Path = parsed.output.resolve()
    if _outside(workspace, journal, output):
        return _journal_failure("binance measurement journal paths must stay inside workspace", 2)
    try:
        window_start_ns = _utc_stamp_ns(parsed.window_start)
        window_end_ns = _utc_stamp_ns(parsed.window_end)
    except ValueError as error:
        return _journal_failure(str(error), 2)
    try:
        written = snapshot_measurement_journal(
            workspace_root=workspace,
            journal_root=journal,
            output_path=output,
            reserve_bytes=parsed.reserve_bytes,
            window_start_ns=window_start_ns,
            window_end_ns=window_end_ns,
        )
    except Exception as error:  # the supervisor reads the code, not the traceback
        return _journal_failure(f"{type(error).__name__}: {error}", _shadow_exit_code(error))
    # The hash a weekly shadow artifact cites, read back off the published bytes.
    document = json.loads(written.read_text(encoding="utf-8"))
    print(f"binance measurement snapshot written: {written}")
    print(f"snapshot content hash: {document['content_hash']}")
    print(f"rounds: {document['rounds']}")
    return 0


def _print_measurement_status(status: MeasurementStatus, *, last: int) -> None:
    """One field a line, as `_print_journal_status` prints v1's.

    `newest_age_seconds` is the liveness figure the 2026-09-17 lesson asks
    for, and the failure rate belongs beside the exclusions: an endpoint that
    quietly stops measuring symbols raises the second long before the first.
    """
    print(f"segment_count: {status.segment_count}")
    print(f"last_sequence: {status.last_sequence}")
    print(f"newest_received_time: {iso_utc_time(status.newest_received_time_ns)}")
    print(f"newest_age_seconds: {status.newest_age_seconds}")
    print(f"window: last {last} segments")
    print(f"snapshot_rounds: {status.snapshot_rounds}")
    for endpoint, rate in status.failure_rate.items():
        print(f"failure_rate {endpoint}: {rate}")
    for endpoint, excluded in status.excluded_total.items():
        print(f"excluded_total {endpoint}: {excluded}")
    print(f"depth_failure_rate: {status.depth_failure_rate}")
    print(f"verify: {'ok' if status.verify_ok else ','.join(status.verify_reasons)}")


def _utc_stamp_ns(value: str) -> int:
    """An ISO-8601 UTC stamp `YYYY-MM-DDTHH:MM:SSZ` as nanoseconds.

    A snapshot window is written by hand, so exactly one spelling is read: a
    bare date, a local offset or anything else is refused rather than guessed
    at, because a window guessed wrong seals the wrong rounds into a document
    that a weekly shadow artifact then cites by hash.
    """
    try:
        stamp = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError:
        raise ValueError(
            "a snapshot window is an ISO-8601 UTC stamp (YYYY-MM-DDTHH:MM:SSZ), "
            f"not {value!r}"
        ) from None
    return int(stamp.timestamp()) * 1_000_000_000


def _resolved(path: Path | None) -> Path | None:
    """An optional path argument, resolved where one was given."""
    return None if path is None else path.resolve()


def _outside(workspace: Path, *paths: Path | None) -> bool:
    """True when any given path leaves the workspace; absent paths are no path."""
    return any(path is not None and not path.is_relative_to(workspace) for path in paths)


# Refusals a rerun cannot fix: a declaration that does not read, a base a
# capture does not extend, a reconciliation mismatch, a journal spec that
# changed under a running stream, an output that already exists, and a storage
# authorisation that is not about free space. `StorageReserveError` is a
# subclass of `StoragePolicyError` and is judged before this tuple is reached.
_SHADOW_REFUSALS: tuple[type[Exception], ...] = (
    ShadowCaptureError,
    ShadowBookError,
    ShadowDeclarationError,
    BinanceMeasurementJournalSpecError,
    StoragePolicyError,
)


def _shadow_exit_code(error: Exception) -> int:
    """`_journal_exit_code`'s policy over the shadow and measurement errors.

    2 stops the supervisor on what the same command would keep hitting; 1 is
    worth another attempt -- a dump the venue has not published this hour, a
    round in which every request failed, a transport that dropped, and the
    free-space reserve, which is the one thing here that changes on its own.
    """
    if isinstance(error, StorageReserveError):
        return 1
    return 2 if isinstance(error, _SHADOW_REFUSALS) else 1


def _journal_exit_code(error: Exception) -> int:
    """2 for what a restart cannot fix, 1 for what it may.

    A spec mismatch and a refused storage authorisation (an excluded drive, a
    path outside the workspace, a temporary directory on another volume) are
    conditions the same command will keep hitting, so the supervisor stops on
    them; a transport failure, a round in which every instrument failed, and a
    reserve the job would cross right now are worth another attempt - the disk
    the reserve guards is the one thing here that changes on its own.
    """
    if isinstance(error, StorageReserveError):
        return 1
    return 2 if isinstance(error, BinanceCostJournalSpecError | StoragePolicyError) else 1


def _journal_failure(message: str, code: int) -> int:
    """Report a failure on stderr and hand the supervisor its exit code."""
    print(message, file=sys.stderr)
    return code


def _demo_runner() -> BacktestRunner:
    return BacktestRunner(
        base_costs=CostScenario("base", Decimal("1"), Decimal("1"), Decimal("1"), Decimal("0")),
        adverse_costs=CostScenario(
            "adverse", Decimal("2"), Decimal("2"), Decimal("2"), Decimal("1")
        ),
        random_seed=17,
    )


def _demo_case() -> BacktestCase:
    prices = ("100", "102", "101", "104", "103")
    states = tuple(_demo_state(index, value) for index, value in enumerate(prices))
    return BacktestCase(
        case_id="synthetic-demo-v1",
        states=states,
        forward_returns=(
            Decimal("0.02"),
            Decimal("-0.01"),
            Decimal("0.03"),
            Decimal("-0.005"),
            Decimal(0),
        ),
        observed_spread_bps=(Decimal(2),) * len(states),
    )


def _demo_state(index: int, value: str) -> MarketState:
    price = Decimal(value)
    time_ns = (index + 1) * 1_000_000_000
    return MarketState(
        event_id=f"demo-{index}",
        available_time_ns=time_ns,
        decision_time_ns=time_ns,
        mid_price=price,
        best_bid=price - Decimal("0.5"),
        best_ask=price + Decimal("0.5"),
        bid_size=Decimal(2),
        ask_size=Decimal(1),
        trade_buy_quantity=Decimal(3),
        trade_sell_quantity=Decimal(1),
        funding_rate=Decimal("0.0001"),
        book_valid=True,
        feed_age_ns=1,
    )
