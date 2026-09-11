# tests/test_panel_accounting.py
from decimal import Context, Decimal, Inexact

import pytest

import trading_bot.panel_accounting as panel_accounting_module
from trading_bot.panel_accounting import PanelAccountingError, evaluate_episode
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


def test_net_return_precision_headroom_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    """The "never itself needs to round" guarantee behind net_return matching
    its own contract breakdown is only real if exceeding it is loud.
    Shrinking _NET_RETURN_CONTEXT to one significant digit of precision
    (Inexact still trapped) turns this ordinary two-contract episode's own
    summation into a real rounding event, and that must now raise
    PanelAccountingError instead of silently returning a net_return that no
    longer matches its own contract_net_contributions."""
    insufficient = Context(prec=1)
    insufficient.traps[Inexact] = True
    monkeypatch.setattr(panel_accounting_module, "_NET_RETURN_CONTEXT", insufficient)

    with pytest.raises(PanelAccountingError, match="precision headroom"):
        evaluate_episode(
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


def test_exit_only_contract_uses_its_universe_tier_and_defaults_to_tier_two_when_absent() -> None:
    # C:0 is held last week (previous_weights) but has dropped out of this week's
    # decision weights entirely, while remaining in the eligible universe. Its exit
    # turnover must be charged at its own measured tier from `tiers`, not tier 2,
    # because `tiers` reflects the current eligible universe (task 9's snapshot),
    # not the member's current portfolio membership.
    exit_only_weights = (("A:0", Decimal("0.5")),)
    previous_with_exit = (("A:0", Decimal("0.5")), ("C:0", Decimal("0.3")))
    exit_only_histories = {"A:0": history("A:0", {DECISION: "100", NEXT: "110"})}

    # Case 1: C:0 is still listed in `tiers` as tier 1 (5 + 5 = 10 bps per side).
    tiered = evaluate_episode(
        sample_id="BINANCE_UM:1:w1",
        member="xs_mom_1w",
        decision_close_ns=DECISION,
        holding_days=7,
        weights=exit_only_weights,
        previous_weights=previous_with_exit,
        histories=exit_only_histories,
        tiers={"A:0": 1, "C:0": 1},
        funding_by_contract={},
        cost_table=BASE,
    )
    assert tiered.turnover == Decimal("0.3")
    # 0.3 * (5 + 5) / 10_000 = 0.0003
    assert tiered.trading_cost == Decimal("0.0003")
    assert tiered.net_return == Decimal("0.0497")
    tiered_net = dict(tiered.contract_net_contributions)
    assert tiered_net["C:0"] == Decimal("-0.0003")
    assert sum(tiered_net.values(), Decimal(0)) == tiered.net_return

    # Case 2: C:0 is absent from `tiers` (e.g. it has left the eligible universe too),
    # so it falls back to the conservative tier-2 default (5 + 10 = 15 bps per side).
    untiered = evaluate_episode(
        sample_id="BINANCE_UM:1:w1",
        member="xs_mom_1w",
        decision_close_ns=DECISION,
        holding_days=7,
        weights=exit_only_weights,
        previous_weights=previous_with_exit,
        histories=exit_only_histories,
        tiers={"A:0": 1},
        funding_by_contract={},
        cost_table=BASE,
    )
    assert untiered.turnover == Decimal("0.3")
    # 0.3 * (5 + 10) / 10_000 = 0.00045
    assert untiered.trading_cost == Decimal("0.00045")
    assert untiered.net_return == Decimal("0.04955")
    untiered_net = dict(untiered.contract_net_contributions)
    assert untiered_net["C:0"] == Decimal("-0.00045")
    assert sum(untiered_net.values(), Decimal(0)) == untiered.net_return


def test_forced_close_uses_the_later_of_two_available_bars_and_charges_full_cost() -> None:
    # B:0 has two bars strictly inside the week, at different prices, and no bar at
    # the exit time. The exit price must come from the later one (day 6: 88), not
    # the earlier one (day 3: 92) -- those give different, distinguishable returns.
    result = evaluate_episode(
        sample_id="BINANCE_UM:1:w1",
        member="xs_mom_1w",
        decision_close_ns=DECISION,
        holding_days=7,
        weights=WEIGHTS,
        previous_weights=(),
        histories={
            "A:0": history("A:0", {DECISION: "100", NEXT: "110"}),
            "B:0": history(
                "B:0",
                {
                    DECISION: "100",
                    DECISION + 3 * DAY_NS: "92",
                    DECISION + 6 * DAY_NS: "88",
                },
            ),
        },
        tiers=TIERS,
        funding_by_contract={},
        cost_table=BASE,
    )
    assert result.forced_close_count == 1
    # r_B = 88/100 - 1 = -0.12 (using the day-3 bar of 92 would give -0.08 and a
    # contribution of 0.04, not 0.06 -- this is what distinguishes "last" from "first").
    assert dict(result.contract_contributions)["B:0"] == Decimal("0.06")
    assert dict(result.drifted_weights)["B:0"] == Decimal("0")

    # Both legs turn over fully from an empty previous_weights (tier 1, 5 + 5 = 10 bps):
    # trading_cost = (0.5 + 0.5) * 10 / 10_000 = 0.001. B:0 additionally pays one forced
    # extra side: 0.5 * 10 / 10_000 * forced_close_multiplier(1) = 0.0005. The two costs
    # are additive, not exclusive: B:0 is charged its ordinary entry turnover *and* the
    # forced exit side.
    assert result.trading_cost == Decimal("0.001")
    assert result.forced_close_cost == Decimal("0.0005")
    # gross_return = 0.5*0.10 + (-0.5)*(-0.12) = 0.05 + 0.06 = 0.11
    # net_return = 0.11 - 0.001 (trading) - 0 (funding) - 0.0005 (forced) = 0.1085
    assert result.net_return == Decimal("0.1085")


def test_funding_event_at_decision_boundary_excluded_at_exit_boundary_included() -> None:
    at_decision = evaluate_episode(
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
                    calc_time_ns=DECISION,
                    rate=Decimal("0.002"),
                ),
            )
        },
        cost_table=BASE,
    )
    assert at_decision.funding_cost == Decimal("0")

    at_exit = evaluate_episode(
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
                    calc_time_ns=NEXT,
                    rate=Decimal("0.002"),
                ),
            )
        },
        cost_table=BASE,
    )
    # 0.5 * 0.002 = 0.001, positive term (payment), multiplier 1 (base)
    assert at_exit.funding_cost == Decimal("0.001")


def test_total_loss_on_a_fully_invested_vector_zeroes_every_drifted_weight() -> None:
    # A fully invested, two-leg vector (weights sum to 1) where both legs go to zero:
    # gross_return = 0.5*(-1) + 0.5*(-1) = -1, so 1 + gross_return == 0 exactly.
    result = evaluate_episode(
        sample_id="BINANCE_UM:1:w1",
        member="xs_mom_1w",
        decision_close_ns=DECISION,
        holding_days=7,
        weights=(("A:0", Decimal("0.5")), ("B:0", Decimal("0.5"))),
        previous_weights=(),
        histories={
            "A:0": history("A:0", {DECISION: "100", NEXT: "0"}),
            "B:0": history("B:0", {DECISION: "100", NEXT: "0"}),
        },
        tiers={"A:0": 1, "B:0": 1},
        funding_by_contract={},
        cost_table=BASE,
    )
    assert result.gross_return == Decimal("-1")
    # Both legs have a real bar at the exit time, so this is a total loss, not a
    # forced close -- the zero drifted weights below must come from the
    # denominator guard, not from the forced-close branch.
    assert result.forced_close_count == 0
    drifted = dict(result.drifted_weights)
    assert drifted["A:0"] == Decimal("0")
    assert drifted["B:0"] == Decimal("0")


def test_fee_overrides_apply_per_contract() -> None:
    plain = evaluate_episode(
        sample_id="s", member="m", decision_close_ns=DECISION, holding_days=7,
        weights=WEIGHTS, previous_weights=(), histories=histories(), tiers=TIERS,
        funding_by_contract={}, cost_table=BASE,
    )
    overridden = evaluate_episode(
        sample_id="s", member="m", decision_close_ns=DECISION, holding_days=7,
        weights=WEIGHTS, previous_weights=(), histories=histories(), tiers=TIERS,
        funding_by_contract={}, cost_table=BASE,
        fee_overrides={"A:0": Decimal("10")},
    )
    # A:0 turnover 0.5 at (10 + 5) bps instead of (5 + 5): +0.5 * 5 / 10000
    assert overridden.trading_cost - plain.trading_cost == Decimal("0.00025")
    plain_net = dict(plain.contract_net_contributions)
    overridden_net = dict(overridden.contract_net_contributions)
    assert overridden_net["B:0"] == plain_net["B:0"]
