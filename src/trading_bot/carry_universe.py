"""Point-in-time eligibility of long-spot short-perpetual pairs."""

from dataclasses import dataclass
from decimal import Decimal

from trading_bot.carry_config import CarryPair, CarryUniverseRules
from trading_bot.panel_config import PanelUniverseRules
from trading_bot.panel_universe import ContractHistory, EligibleContract, select_universe

_UNCAPPED = 1_000_000


@dataclass(frozen=True, slots=True)
class EligiblePair:
    pair_id: str
    perpetual_contract_id: str
    spot_contract_id: str
    perpetual_tier: int
    spot_tier: int
    tier: int
    liquidity_rank: int
    median_quote_volume: Decimal


@dataclass(frozen=True, slots=True)
class PairUniverseSnapshot:
    decision_close_ns: int
    pairs: tuple[EligiblePair, ...]
    reason_codes: tuple[str, ...]


def select_pair_universe(
    perp_histories: dict[str, ContractHistory],
    spot_histories: dict[str, ContractHistory],
    *,
    pairs: tuple[CarryPair, ...],
    decision_close_ns: int,
    rules: CarryUniverseRules,
) -> PairUniverseSnapshot:
    """Apply the P1.27 leg rules to both legs, then rank and cap the pairs.

    Each leg is screened by ``select_universe`` with the rank cap lifted and
    the minimum lowered to one, so the only floor that can declare the
    universe too small is the pair-level ``minimum_pairs``. The spot tier is
    the tier the spot leg holds among all eligible spot legs, the perpetual
    tier follows the pair ranking, and a pair's cost tier is the worse of
    the two, as spec section 7.1 states.
    """
    leg_rules = PanelUniverseRules(
        minimum_history_days=rules.minimum_history_days,
        liquidity_window_days=rules.liquidity_window_days,
        minimum_median_quote_volume=rules.minimum_median_quote_volume,
        maximum_contracts=_UNCAPPED,
        minimum_contracts=1,
        tier_one_rank_limit=rules.tier_one_rank_limit,
    )
    perp_snapshot = select_universe(
        perp_histories, decision_close_ns=decision_close_ns, rules=leg_rules
    )
    spot_snapshot = select_universe(
        spot_histories, decision_close_ns=decision_close_ns, rules=leg_rules
    )
    perp_by_symbol = _by_symbol(perp_snapshot.contracts, perp_histories)
    spot_by_symbol = _by_symbol(spot_snapshot.contracts, spot_histories)

    candidates: list[tuple[Decimal, str, EligibleContract, EligibleContract]] = []
    for pair in pairs:
        perp = perp_by_symbol.get(pair.perpetual)
        spot = spot_by_symbol.get(pair.spot)
        if perp is None or spot is None:
            continue
        candidates.append((perp.median_quote_volume, perp.contract_id, perp, spot))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    if len(candidates) < rules.minimum_pairs:
        return PairUniverseSnapshot(decision_close_ns, (), ("UNIVERSE_TOO_SMALL",))

    selected = candidates[: rules.maximum_pairs]
    eligible = []
    for index, (median, _, perp, spot) in enumerate(selected):
        rank = index + 1
        perpetual_tier = 1 if rank <= rules.tier_one_rank_limit else 2
        eligible.append(
            EligiblePair(
                pair_id=perp.contract_id,
                perpetual_contract_id=perp.contract_id,
                spot_contract_id=spot.contract_id,
                perpetual_tier=perpetual_tier,
                spot_tier=spot.tier,
                tier=max(perpetual_tier, spot.tier),
                liquidity_rank=rank,
                median_quote_volume=median,
            )
        )
    return PairUniverseSnapshot(decision_close_ns, tuple(eligible), ())


def _by_symbol(
    contracts: tuple[EligibleContract, ...], histories: dict[str, ContractHistory]
) -> dict[str, EligibleContract]:
    return {histories[item.contract_id].instrument_id: item for item in contracts}
