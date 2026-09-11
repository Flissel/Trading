"""Two-leg carry episodes on top of the panel accounting."""

from dataclasses import dataclass
from decimal import Decimal

from trading_bot.carry_config import CarryCostTable
from trading_bot.panel_accounting import EpisodeResult, evaluate_episode
from trading_bot.panel_config import PanelCostTable
from trading_bot.panel_reader import FundingEvent
from trading_bot.panel_universe import ContractHistory

_BPS = Decimal(10_000)


@dataclass(frozen=True, slots=True)
class CarryEpisode:
    result: EpisodeResult
    funding_collected: Decimal
    basis_pnl: Decimal
    spot_trading_cost: Decimal
    perpetual_trading_cost: Decimal


def evaluate_carry_episode(
    *,
    sample_id: str,
    member: str,
    decision_close_ns: int,
    holding_days: int,
    leg_weights: tuple[tuple[str, Decimal], ...],
    previous_leg_weights: tuple[tuple[str, Decimal], ...],
    histories: dict[str, ContractHistory],
    tiers: dict[str, int],
    funding_by_leg: dict[str, tuple[FundingEvent, ...]],
    cost_table: CarryCostTable,
    pair_of_leg: dict[str, str],
) -> CarryEpisode:
    """One weekly episode of a long-spot short-perpetual book.

    Legs are ordinary contracts to the panel accounting: the perpetual leg is
    a negative weight, so a positive settlement is a receipt through the
    existing sign rule, and the spot leg carries its own fee through
    ``fee_overrides``. A leg that loses its exit bar is force-closed by the
    panel accounting; the fold runner then drops the pair from every retained
    cohort so the partner leaves at the next rebalance, because an unhedged
    leg is not the position under test. Per-leg attributions are
    re-aggregated to the pair so the concentration gate reads pairs.
    """
    panel_table = PanelCostTable(
        name=cost_table.name,
        fee_bps_per_side=cost_table.perpetual_fee_bps_per_side,
        slippage_bps_per_side_tier_one=cost_table.slippage_bps_per_side_tier_one,
        slippage_bps_per_side_tier_two=cost_table.slippage_bps_per_side_tier_two,
        funding_receipt_multiplier=cost_table.funding_receipt_multiplier,
        funding_payment_multiplier=cost_table.funding_payment_multiplier,
        forced_close_multiplier=cost_table.forced_close_multiplier,
    )
    overrides = {
        leg: cost_table.spot_fee_bps_per_side
        for leg, _ in (*leg_weights, *previous_leg_weights)
        if leg.startswith("spot:")
    }
    result = evaluate_episode(
        sample_id=sample_id,
        member=member,
        decision_close_ns=decision_close_ns,
        holding_days=holding_days,
        weights=leg_weights,
        previous_weights=previous_leg_weights,
        histories=histories,
        tiers=tiers,
        funding_by_contract=funding_by_leg,
        cost_table=panel_table,
        fee_overrides=overrides,
    )
    pair_net: dict[str, Decimal] = {}
    pair_gross: dict[str, Decimal] = {}
    for leg, value in result.contract_net_contributions:
        pair = pair_of_leg[leg]
        pair_net[pair] = pair_net.get(pair, Decimal(0)) + value
    for leg, value in result.contract_contributions:
        pair = pair_of_leg[leg]
        pair_gross[pair] = pair_gross.get(pair, Decimal(0)) + value
    aggregated = EpisodeResult(
        sample_id=result.sample_id,
        member=result.member,
        scenario=result.scenario,
        gross_return=result.gross_return,
        turnover=result.turnover,
        trading_cost=result.trading_cost,
        funding_cost=result.funding_cost,
        forced_close_cost=result.forced_close_cost,
        net_return=result.net_return,
        gross_exposure=result.gross_exposure,
        net_exposure=result.net_exposure,
        forced_close_count=result.forced_close_count,
        contract_contributions=tuple(sorted(pair_gross.items())),
        contract_net_contributions=tuple(sorted(pair_net.items())),
        drifted_weights=result.drifted_weights,
    )
    spot_cost, perp_cost = _leg_turnover_costs(leg_weights, previous_leg_weights, tiers, cost_table)
    return CarryEpisode(
        result=aggregated,
        funding_collected=-result.funding_cost,
        basis_pnl=result.gross_return,
        spot_trading_cost=spot_cost,
        perpetual_trading_cost=perp_cost,
    )


def _leg_turnover_costs(
    weights: tuple[tuple[str, Decimal], ...],
    previous: tuple[tuple[str, Decimal], ...],
    tiers: dict[str, int],
    table: CarryCostTable,
) -> tuple[Decimal, Decimal]:
    new = dict(weights)
    old = dict(previous)
    spot = Decimal(0)
    perp = Decimal(0)
    for leg in sorted(set(new) | set(old)):
        change = abs(new.get(leg, Decimal(0)) - old.get(leg, Decimal(0)))
        if change == 0:
            continue
        slippage = (
            table.slippage_bps_per_side_tier_one
            if tiers.get(leg, 2) == 1
            else table.slippage_bps_per_side_tier_two
        )
        if leg.startswith("spot:"):
            spot += change * (table.spot_fee_bps_per_side + slippage) / _BPS
        else:
            perp += change * (table.perpetual_fee_bps_per_side + slippage) / _BPS
    return spot, perp
