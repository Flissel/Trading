# tests/test_carry_accounting.py
from decimal import Decimal

from trading_bot.carry_accounting import CarryEpisode, evaluate_carry_episode
from trading_bot.carry_config import CarryCostTable
from trading_bot.panel_reader import FundingEvent
from trading_bot.panel_universe import ContractHistory

DAY_NS = 86_400_000_000_000
DECISION = 6 * DAY_NS - 1_000_000
EXIT = DECISION + 7 * DAY_NS
BASE = CarryCostTable(
    name="base", perpetual_fee_bps_per_side=Decimal("5"), spot_fee_bps_per_side=Decimal("10"),
    slippage_bps_per_side_tier_one=Decimal("5"), slippage_bps_per_side_tier_two=Decimal("10"),
    funding_receipt_multiplier=Decimal("1"), funding_payment_multiplier=Decimal("1"),
    forced_close_multiplier=Decimal("1"),
)
ADVERSE = BASE.model_copy(update={
    "name": "adverse", "slippage_bps_per_side_tier_one": Decimal("10"),
    "slippage_bps_per_side_tier_two": Decimal("20"), "funding_receipt_multiplier": Decimal("0.75"),
    "funding_payment_multiplier": Decimal("2"), "forced_close_multiplier": Decimal("2"),
})


def history(leg: str, entry: str, exit_: str) -> ContractHistory:
    closes = {DECISION: Decimal(entry), EXIT: Decimal(exit_)}
    return ContractHistory(
        contract_id=leg, instrument_id=leg, closes=closes,
        quote_volumes={k: Decimal(1) for k in closes}, close_times=tuple(sorted(closes)),
    )


HIST = {
    "spot:A:7": history("spot:A:7", "100", "110"),
    "perp:A:0": history("perp:A:0", "100", "112"),
}
LEGS = (("perp:A:0", Decimal("-0.5")), ("spot:A:7", Decimal("0.5")))
TIERS = {"perp:A:0": 1, "spot:A:7": 1}
PAIR_OF = {"perp:A:0": "A:0", "spot:A:7": "A:0"}
FUNDING: dict[str, tuple[FundingEvent, ...]] = {
    "perp:A:0": (
        FundingEvent(
            contract_id="A:0", instrument_id="A",
            calc_time_ns=DECISION + DAY_NS, rate=Decimal("0.001"),
        ),
    )
}


def run(table: CarryCostTable) -> CarryEpisode:
    return evaluate_carry_episode(
        sample_id="s", member="carry_l1w_h4w", decision_close_ns=DECISION, holding_days=7,
        leg_weights=LEGS, previous_leg_weights=(), histories=HIST, tiers=TIERS,
        funding_by_leg=FUNDING, cost_table=table, pair_of_leg=PAIR_OF,
    )


def test_base_arithmetic_by_hand() -> None:
    episode = run(BASE)
    r = episode.result
    # basis: spot +10% on 0.5, perp +12% on -0.5 -> 0.05 - 0.06 = -0.01
    assert episode.basis_pnl == Decimal("-0.01")
    assert r.gross_return == Decimal("-0.01")
    # funding: short perp collects 0.5 * 0.001
    assert episode.funding_collected == Decimal("0.0005")
    # costs: spot 0.5 turnover at (10+5) bps = 0.00075, perp 0.5 at (5+5) = 0.0005
    assert episode.spot_trading_cost == Decimal("0.00075")
    assert episode.perpetual_trading_cost == Decimal("0.0005")
    assert r.net_return == Decimal("-0.01") + Decimal("0.0005") - Decimal("0.00125")
    # attribution is per pair and sums to net
    assert dict(r.contract_net_contributions) == {"A:0": r.net_return}
    assert r.gross_exposure == Decimal(1) and r.net_exposure == Decimal(0)


def test_adverse_haircuts_receipts_and_doubles_slippage() -> None:
    episode = run(ADVERSE)
    assert episode.funding_collected == Decimal("0.000375")
    assert episode.spot_trading_cost == Decimal("0.001")
    assert episode.perpetual_trading_cost == Decimal("0.00075")


def test_forced_leg_is_closed_and_its_partner_keeps_drifting() -> None:
    """The panel accounting force-closes the leg without an exit bar; the
    partner leg is closed by the fold runner at the next rebalance (Task 7),
    so here it still carries a drifted weight."""
    hist = dict(HIST)
    hist["perp:A:0"] = ContractHistory(
        contract_id="perp:A:0", instrument_id="perp:A:0",
        closes={DECISION: Decimal("100"), EXIT - DAY_NS: Decimal("105")},
        quote_volumes={}, close_times=(DECISION, EXIT - DAY_NS),
    )
    episode = evaluate_carry_episode(
        sample_id="s", member="carry_l1w_h4w", decision_close_ns=DECISION, holding_days=7,
        leg_weights=LEGS, previous_leg_weights=(), histories=hist, tiers=TIERS,
        funding_by_leg={}, cost_table=BASE, pair_of_leg=PAIR_OF,
    )
    r = episode.result
    assert r.forced_close_count == 1
    drifted = dict(r.drifted_weights)
    assert drifted["perp:A:0"] == Decimal(0)
    assert drifted["spot:A:7"] > 0
    # forced close of the perp leg at the last close 105: 0.5 * 10 bps * multiplier 1
    assert r.forced_close_cost == Decimal("0.0005")
    # pair attribution still sums to net
    assert dict(r.contract_net_contributions) == {"A:0": r.net_return}


def test_previous_spot_leg_exit_is_charged_the_spot_fee() -> None:
    """A spot leg present only in the previous weights is still a spot leg."""
    episode = evaluate_carry_episode(
        sample_id="s", member="carry_l1w_h4w", decision_close_ns=DECISION, holding_days=7,
        leg_weights=(), previous_leg_weights=LEGS, histories=HIST, tiers=TIERS,
        funding_by_leg={}, cost_table=BASE, pair_of_leg=PAIR_OF,
    )
    assert episode.spot_trading_cost == Decimal("0.00075")
    assert episode.perpetual_trading_cost == Decimal("0.0005")
    assert episode.result.trading_cost == Decimal("0.00125")
