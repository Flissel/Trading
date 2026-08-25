import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

from trading_bot.backtest import BacktestCase, BacktestRunner, verify_audit_chain, write_report
from trading_bot.cli import main
from trading_bot.features import MarketState
from trading_bot.strategy import CostScenario


def market_state(event_id: str, time_ns: int, mid: str) -> MarketState:
    price = Decimal(mid)
    return MarketState(
        event_id=event_id,
        available_time_ns=time_ns,
        decision_time_ns=time_ns,
        mid_price=price,
        best_bid=price - Decimal("0.5"),
        best_ask=price + Decimal("0.5"),
        bid_size=Decimal("2"),
        ask_size=Decimal("1"),
        trade_buy_quantity=Decimal("3"),
        trade_sell_quantity=Decimal("1"),
        funding_rate=Decimal("0.0001"),
        book_valid=True,
        feed_age_ns=1,
    )


def case() -> BacktestCase:
    return BacktestCase(
        case_id="fixture-v1",
        states=(
            market_state("e0", 100, "100"),
            market_state("e1", 200, "102"),
            market_state("e2", 300, "101"),
            market_state("e3", 400, "104"),
        ),
        forward_returns=(
            Decimal("0.02"),
            Decimal("-0.01"),
            Decimal("0.03"),
            Decimal("0"),
        ),
        observed_spread_bps=(Decimal("2"),) * 4,
    )


def runner() -> BacktestRunner:
    return BacktestRunner(
        base_costs=CostScenario("base", Decimal("1"), Decimal("1"), Decimal("1"), Decimal("0")),
        adverse_costs=CostScenario(
            "adverse", Decimal("2"), Decimal("2"), Decimal("2"), Decimal("1")
        ),
        random_seed=17,
    )


def test_end_to_end_backtest_traces_every_fill_to_source_and_risk_decision() -> None:
    report = runner().run(case(), baseline_name="momentum")

    assert report.mode == "backtest"
    assert report.live_execution_enabled is False
    assert report.input_count == 4
    assert report.selected_baseline == "momentum"
    assert report.traces
    assert all(trace.source_event_ids for trace in report.traces)
    assert all(trace.risk_decision.intent_id == trace.intent_id for trace in report.traces)
    assert all(trace.execution.command_id == trace.command_id for trace in report.traces)
    assert verify_audit_chain(report.audit_records) is True


def test_audit_chain_detects_injected_corruption() -> None:
    report = runner().run(case(), baseline_name="momentum")
    records = list(report.audit_records)
    records[1] = replace(records[1], record_hash="0" * 64)

    assert verify_audit_chain(tuple(records)) is False


def test_same_backtest_input_writes_identical_semantic_report(tmp_path: Path) -> None:
    first = runner().run(case(), baseline_name="momentum")
    second = runner().run(case(), baseline_name="momentum")
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    write_report(first_path, first)
    write_report(second_path, second)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert first.report_hash == second.report_hash


def test_cli_creates_bounded_backtest_report_inside_workspace(tmp_path: Path) -> None:
    output = tmp_path / "artifacts" / "demo-report.json"

    exit_code = main(
        [
            "demo-backtest",
            "--workspace-root",
            str(tmp_path),
            "--output",
            str(output),
            "--reserve-bytes",
            "0",
        ]
    )
    document = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert document["mode"] == "backtest"
    assert document["live_execution_enabled"] is False
    assert document["promotion"]["status"] == "insufficient_evidence"
    assert document["audit_chain_valid"] is True
