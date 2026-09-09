# tests/test_panel_accounting.py
from decimal import Decimal

from trading_bot.panel_accounting import evaluate_episode
from trading_bot.panel_config import PanelCostTable
from trading_bot.panel_reader import FundingEvent
from trading_bot.panel_universe import ContractHistory

DAY_NS = 86_400_000_000_000
DECISION = 6 * DAY_NS - 1_000_000
NEXT = DECISION + 7 * DAY_NS

BASE = PanelCostTable(
    name="base",
    fee_bps_per_side=Decimal("5"),
    slippage_bps_per_side_tier_one=Decimal("5"),
    slippage_bps_per_side_tier_two=Decimal("10"),
    funding_receipt_multiplier=Decimal("1"),
    funding_payment_multiplier=Decimal("1"),
    forced_close_multiplier=Decimal("1"),
)
ADVERSE = PanelCostTable(
    name="adverse",
    fee_bps_per_side=Decimal("5"),
    slippage_bps_per_side_tier_one=Decimal("10"),
    slippage_bps_per_side_tier_two=Decimal("20"),
    funding_receipt_multiplier=Decimal("0"),
    funding_payment_multiplier=Decimal("2"),
    forced_close_multiplier=Decimal("2"),
)


def history(contract_id: str, closes: dict[int, str]) -> ContractHistory:
    parsed = {key: Decimal(value) for key, value in closes.items()}
    return ContractHistory(
        contract_id=contract_id,
        instrument_id=contract_id.split(":")[0],
        closes=parsed,
        quote_volumes={key: Decimal("1000") for key in parsed},
        close_times=tuple(sorted(parsed)),
    )


def histories(exit_b: str = "95", *, b_exit_time: int = NEXT) -> dict[str, ContractHistory]:
    return {
        "A:0": history("A:0", {DECISION: "100", NEXT: "110"}),
        "B:0": history("B:0", {DECISION: "100", b_exit_time: exit_b}),
    }


WEIGHTS = (("A:0", Decimal("0.5")), ("B:0", Decimal("-0.5")))
TIERS = {"A:0": 1, "B:0": 1}


def test_base_episode_arithmetic() -> None:
    result = evaluate_episode(
        sample_id="BINANCE_UM:1:w1",
        member="xs_mom_1w",
        decision_close_ns=DECISION,
        holding_days=7,
        weights=WEIGHTS,
        previous_weights=(),
        histories=histories(),
        tiers=TIERS,
        funding_by_contract={
            "A:0": (
                FundingEvent(
                    contract_id="A:0",
                    instrument_id="A",
                    calc_time_ns=DECISION + DAY_NS,
                    rate=Decimal("0.001"),
                ),
            )
        },
        cost_table=BASE,
    )
    assert result.gross_return == Decimal("0.075")
    assert result.turnover == Decimal("1")
    assert result.trading_cost == Decimal("0.001")
    assert result.funding_cost == Decimal("0.0005")
    assert result.forced_close_cost == Decimal("0")
    assert result.net_return == Decimal("0.0735")
    assert result.gross_exposure == Decimal("1")
    assert result.net_exposure == Decimal("0")
    assert dict(result.contract_contributions)["A:0"] == Decimal("0.05")
    net_attribution = dict(result.contract_net_contributions)
    assert net_attribution["A:0"] == Decimal("0.049")
    assert net_attribution["B:0"] == Decimal("0.0245")
    assert sum(net_attribution.values(), Decimal(0)) == result.net_return


def test_adverse_doubles_slippage_and_funding_payments() -> None:
    result = evaluate_episode(
        sample_id="BINANCE_UM:1:w1",
        member="xs_mom_1w",
        decision_close_ns=DECISION,
        holding_days=7,
        weights=WEIGHTS,
        previous_weights=(),
        histories=histories(),
        tiers=TIERS,
        funding_by_contract={
            "A:0": (
                FundingEvent(
                    contract_id="A:0",
                    instrument_id="A",
                    calc_time_ns=DECISION + DAY_NS,
                    rate=Decimal("0.001"),
                ),
            )
        },
        cost_table=ADVERSE,
    )
    assert result.trading_cost == Decimal("0.0015")
    assert result.funding_cost == Decimal("0.001")
    assert result.net_return == Decimal("0.0725")


def test_adverse_drops_funding_receipts() -> None:
    result = evaluate_episode(
        sample_id="BINANCE_UM:1:w1",
        member="xs_mom_1w",
        decision_close_ns=DECISION,
        holding_days=7,
        weights=WEIGHTS,
        previous_weights=(),
        histories=histories(),
        tiers=TIERS,
        funding_by_contract={
            "B:0": (
                FundingEvent(
                    contract_id="B:0",
                    instrument_id="B",
                    calc_time_ns=DECISION + DAY_NS,
                    rate=Decimal("0.001"),
                ),
            )
        },
        cost_table=ADVERSE,
    )
    assert result.funding_cost == Decimal("0")


def test_forced_close_uses_the_last_available_close_and_costs_one_side() -> None:
    result = evaluate_episode(
        sample_id="BINANCE_UM:1:w1",
        member="xs_mom_1w",
        decision_close_ns=DECISION,
        holding_days=7,
        weights=WEIGHTS,
        previous_weights=(),
        histories=histories("90", b_exit_time=NEXT - DAY_NS),
        tiers=TIERS,
        funding_by_contract={},
        cost_table=BASE,
    )
    assert result.forced_close_count == 1
    assert result.forced_close_cost == Decimal("0.0005")
    assert dict(result.contract_contributions)["B:0"] == Decimal("0.05")
    assert dict(result.drifted_weights)["B:0"] == Decimal("0")


def test_turnover_uses_drifted_previous_weights() -> None:
    first = evaluate_episode(
        sample_id="BINANCE_UM:1:w1",
        member="xs_mom_1w",
        decision_close_ns=DECISION,
        holding_days=7,
        weights=WEIGHTS,
        previous_weights=(),
        histories=histories(),
        tiers=TIERS,
        funding_by_contract={},
        cost_table=BASE,
    )
    second = evaluate_episode(
        sample_id="BINANCE_UM:2:w1",
        member="xs_mom_1w",
        decision_close_ns=NEXT,
        holding_days=7,
        weights=WEIGHTS,
        previous_weights=first.drifted_weights,
        histories={
            "A:0": ContractHistory(
                contract_id="A:0",
                instrument_id="A",
                closes={NEXT: Decimal("110"), NEXT + 7 * DAY_NS: Decimal("110")},
                quote_volumes={},
                close_times=(NEXT, NEXT + 7 * DAY_NS),
            ),
            "B:0": ContractHistory(
                contract_id="B:0",
                instrument_id="B",
                closes={NEXT: Decimal("95"), NEXT + 7 * DAY_NS: Decimal("95")},
                quote_volumes={},
                close_times=(NEXT, NEXT + 7 * DAY_NS),
            ),
        },
        tiers=TIERS,
        funding_by_contract={},
        cost_table=BASE,
    )
    assert second.turnover < Decimal("0.15")
    assert second.turnover > Decimal("0")
