"""Command-line entry point for bounded local research runs."""

import argparse
import shutil
from decimal import Decimal
from pathlib import Path

from trading_bot.backtest import BacktestCase, BacktestRunner, write_report
from trading_bot.carry_fold_run import run_carry_fold
from trading_bot.features import MarketState
from trading_bot.fold_evaluation import run_fold_evaluation
from trading_bot.market_capture import capture_public_candle_history, capture_public_candles
from trading_bot.panel_capture import capture_panel, repair_panel_capture
from trading_bot.panel_config import load_family_spec
from trading_bot.panel_decision import build_panel_decision
from trading_bot.panel_fold_run import run_panel_fold
from trading_bot.panel_samples import publish_panel_walk_forward
from trading_bot.research_run import run_capture_research
from trading_bot.storage import StoragePolicy
from trading_bot.strategy import CostScenario
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
    raise AssertionError("unreachable command")


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
