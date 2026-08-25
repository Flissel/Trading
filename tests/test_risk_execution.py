from decimal import Decimal
from pathlib import Path

from trading_bot.execution import (
    ExecutionAssumptions,
    ExecutionSimulator,
    ExecutionStatus,
    OrderCommand,
)
from trading_bot.risk import (
    DecisionIntent,
    RiskHealth,
    RiskPolicy,
    RiskState,
    RiskStateStore,
)


def intent(**overrides: object) -> DecisionIntent:
    values: dict[str, object] = {
        "intent_id": "intent-1",
        "instrument_id": "BTC-USDT-SWAP.OKX",
        "direction": 1,
        "requested_notional": Decimal("20"),
        "expected_net_return_adverse": Decimal("0.01"),
        "maximum_planned_loss": Decimal("0.20"),
        "expires_at_ns": 200,
        "lineage_complete": True,
        "forecast_eligible": True,
        "has_exit": True,
    }
    values.update(overrides)
    return DecisionIntent(**values)  # type: ignore[arg-type]


def healthy() -> RiskHealth:
    return RiskHealth(
        market_data_healthy=True,
        book_valid=True,
        account_synchronized=True,
        rules_valid=True,
        audit_sink_healthy=True,
        unknown_order=False,
    )


def clean_state(**overrides: object) -> RiskState:
    values: dict[str, object] = {
        "equity": Decimal("100"),
        "high_water_mark": Decimal("100"),
        "daily_loss": Decimal("0"),
        "weekly_loss": Decimal("0"),
        "cumulative_loss": Decimal("0"),
        "gross_exposure": Decimal("0"),
        "instrument_exposures": (),
        "open_order_intents": (),
        "cooldown_until_ns": None,
        "hard_kill_switch": False,
    }
    values.update(overrides)
    return RiskState(**values)  # type: ignore[arg-type]


def test_risk_engine_rejects_non_positive_adverse_ev() -> None:
    decision = RiskPolicy().evaluate(
        intent(expected_net_return_adverse=Decimal("0")),
        clean_state(),
        healthy(),
        now_ns=100,
        venue_min_notional=Decimal("5"),
        venue_notional_step=Decimal("1"),
    )

    assert decision.approved_notional == Decimal("0")
    assert decision.reason_codes == ("ADVERSE_EV_NON_POSITIVE",)


def test_risk_engine_reduces_size_for_position_and_loss_caps() -> None:
    decision = RiskPolicy().evaluate(
        intent(requested_notional=Decimal("40"), maximum_planned_loss=Decimal("0.50")),
        clean_state(),
        healthy(),
        now_ns=100,
        venue_min_notional=Decimal("5"),
        venue_notional_step=Decimal("1"),
    )

    assert decision.approved_notional == Decimal("20")
    assert decision.approved_notional <= decision.requested_notional
    assert decision.reason_codes == ("SIZE_REDUCED",)


def test_risk_engine_rejects_unsafe_venue_minimum() -> None:
    decision = RiskPolicy().evaluate(
        intent(requested_notional=Decimal("4")),
        clean_state(),
        healthy(),
        now_ns=100,
        venue_min_notional=Decimal("5"),
        venue_notional_step=Decimal("1"),
    )

    assert decision.approved_notional == Decimal("0")
    assert decision.reason_codes == ("VENUE_MINIMUM_EXCEEDS_SAFE_SIZE",)


def test_loss_limit_and_unknown_order_fail_closed() -> None:
    policy = RiskPolicy()
    loss = policy.evaluate(
        intent(),
        clean_state(daily_loss=Decimal("1")),
        healthy(),
        now_ns=100,
        venue_min_notional=Decimal("5"),
        venue_notional_step=Decimal("1"),
    )
    unknown = policy.evaluate(
        intent(),
        clean_state(),
        RiskHealth(True, True, True, True, True, True),
        now_ns=100,
        venue_min_notional=Decimal("5"),
        venue_notional_step=Decimal("1"),
    )

    assert loss.reason_codes == ("DAILY_LOSS_LIMIT",)
    assert unknown.reason_codes == ("UNKNOWN_ORDER_OUTCOME",)


def test_risk_state_store_survives_restart(tmp_path: Path) -> None:
    path = tmp_path / "risk-state.json"
    expected = clean_state(
        high_water_mark=Decimal("103.25"),
        cooldown_until_ns=900,
        hard_kill_switch=True,
    )

    RiskStateStore(path).save(expected)
    actual = RiskStateStore(path).load()

    assert actual == expected


def test_execution_simulator_is_idempotent_and_models_partial_fill() -> None:
    simulator = ExecutionSimulator(
        ExecutionAssumptions(
            latency_ns=5,
            fill_fraction=Decimal("0.5"),
            fee_bps=Decimal("2"),
            slippage_bps=Decimal("3"),
            timeout_ns=100,
        )
    )
    command = OrderCommand(
        command_id="cmd-1",
        instrument_id="BTC-USDT-SWAP.OKX",
        direction=1,
        approved_notional=Decimal("20"),
        submitted_at_ns=100,
    )

    first = simulator.execute(command, market_price=Decimal("100"), observed_at_ns=105)
    second = simulator.execute(command, market_price=Decimal("999"), observed_at_ns=999)

    assert first is second
    assert first.status is ExecutionStatus.PARTIALLY_FILLED
    assert first.filled_notional == Decimal("10.0")
    assert first.fill_price == Decimal("100.03")
    assert first.fee == Decimal("0.0020")


def test_execution_timeout_is_unknown_not_rejected() -> None:
    simulator = ExecutionSimulator(
        ExecutionAssumptions(5, Decimal("1"), Decimal("2"), Decimal("3"), 10)
    )
    result = simulator.execute(
        OrderCommand("cmd-1", "BTC-USDT-SWAP.OKX", 1, Decimal("20"), 100),
        market_price=Decimal("100"),
        observed_at_ns=111,
    )

    assert result.status is ExecutionStatus.UNKNOWN
    assert result.filled_notional is None
