"""Weekly portfolio accounting for the perpetual panel."""

from dataclasses import dataclass
from decimal import Context, Decimal, Inexact, localcontext

from trading_bot.panel_config import PanelCostTable
from trading_bot.panel_reader import FundingEvent
from trading_bot.panel_universe import ContractHistory

DAY_NS = 86_400_000_000_000
_BPS = Decimal(10_000)
# Precision for summing net_contributions into net_return only: generous
# headroom above the default 28 significant digits so that this one summation
# never itself needs to round. Each per-contract contribution can already
# carry close to 28-29 significant digits (from weight and return divisions
# upstream); summing the bounded number of contracts in one episode's universe
# under the default context can round, silently breaking the invariant (relied
# on by callers pooling per-contract totals across many episodes) that an
# episode's own net_contributions sum to its own net_return exactly.
#
# Single-vector books need roughly 32 digits of headroom (measured). Cohort
# books (P1.31, P1.32) can hold residual weights of about 1e-28 where a long
# in an older vector nearly cancels a short in a newer one; those residuals'
# contributions carry their own 28 digits below that, so an exact sum with a
# 1e-2 largest term needs about 60 digits. prec=120 leaves ample margin for
# both (measured on the P1.32 fold that first exceeded 50). An exact sum is
# the same at any sufficient precision, so raising the headroom changes no
# previously produced value. Inexact is trapped so that margin is an enforced
# guarantee, not an assumption: if it is ever exceeded, this raises
# immediately rather than silently rounding to a net_return that no longer
# matches its own contract breakdown.
_NET_RETURN_CONTEXT = Context(prec=120)
_NET_RETURN_CONTEXT.traps[Inexact] = True


class PanelAccountingError(RuntimeError):
    """Raised when an episode cannot be evaluated."""


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    sample_id: str
    member: str
    scenario: str
    gross_return: Decimal
    turnover: Decimal
    trading_cost: Decimal
    funding_cost: Decimal
    forced_close_cost: Decimal
    net_return: Decimal
    gross_exposure: Decimal
    net_exposure: Decimal
    forced_close_count: int
    contract_contributions: tuple[tuple[str, Decimal], ...]
    contract_net_contributions: tuple[tuple[str, Decimal], ...]
    drifted_weights: tuple[tuple[str, Decimal], ...]


def evaluate_episode(
    *,
    sample_id: str,
    member: str,
    decision_close_ns: int,
    holding_days: int,
    weights: tuple[tuple[str, Decimal], ...],
    previous_weights: tuple[tuple[str, Decimal], ...],
    histories: dict[str, ContractHistory],
    tiers: dict[str, int],
    funding_by_contract: dict[str, tuple[FundingEvent, ...]],
    cost_table: PanelCostTable,
    fee_overrides: dict[str, Decimal] | None = None,
) -> EpisodeResult:
    """Evaluate one weekly rebalance under a single cost table."""
    if holding_days < 1:
        raise PanelAccountingError("holding period must be positive")
    exit_close_ns = decision_close_ns + holding_days * DAY_NS
    weight_map = dict(weights)
    previous_map = dict(previous_weights)
    overrides = fee_overrides or {}

    returns: dict[str, Decimal] = {}
    forced: set[str] = set()
    for contract_id, _ in weights:
        history = histories.get(contract_id)
        if history is None:
            raise PanelAccountingError(f"missing history for {contract_id}")
        entry = history.closes.get(decision_close_ns)
        if entry is None or entry <= 0:
            raise PanelAccountingError(f"missing entry close for {contract_id}")
        exit_price = history.closes.get(exit_close_ns)
        if exit_price is None:
            candidates = [
                value
                for value in history.close_times
                if decision_close_ns < value < exit_close_ns
            ]
            forced.add(contract_id)
            exit_price = history.closes[candidates[-1]] if candidates else entry
        returns[contract_id] = exit_price / entry - Decimal(1)

    contributions = {
        contract_id: weight_map[contract_id] * returns[contract_id] for contract_id in weight_map
    }
    gross_return = sum(contributions.values(), Decimal(0))

    universe = sorted(set(weight_map) | set(previous_map))
    turnover = Decimal(0)
    trading_by_contract: dict[str, Decimal] = {}
    for contract_id in universe:
        change = abs(
            weight_map.get(contract_id, Decimal(0)) - previous_map.get(contract_id, Decimal(0))
        )
        if change == 0:
            continue
        turnover += change
        trading_by_contract[contract_id] = (
            change
            * _per_side_bps(cost_table, tiers.get(contract_id, 2), overrides.get(contract_id))
            / _BPS
        )
    trading_cost = sum(trading_by_contract.values(), Decimal(0))

    funding_by_id: dict[str, Decimal] = {}
    for contract_id, weight in weight_map.items():
        charged = Decimal(0)
        for event in funding_by_contract.get(contract_id, ()):
            if not decision_close_ns < event.calc_time_ns <= exit_close_ns:
                continue
            term = weight * event.rate
            if term > 0:
                charged += term * cost_table.funding_payment_multiplier
            elif term < 0:
                charged += term * cost_table.funding_receipt_multiplier
        if charged != 0:
            funding_by_id[contract_id] = charged
    funding_cost = sum(funding_by_id.values(), Decimal(0))

    forced_by_id: dict[str, Decimal] = {}
    for contract_id in sorted(forced):
        forced_by_id[contract_id] = (
            abs(weight_map[contract_id])
            * _per_side_bps(cost_table, tiers.get(contract_id, 2), overrides.get(contract_id))
            / _BPS
            * cost_table.forced_close_multiplier
        )
    forced_close_cost = sum(forced_by_id.values(), Decimal(0))

    net_contributions = {
        contract_id: contributions.get(contract_id, Decimal(0))
        - trading_by_contract.get(contract_id, Decimal(0))
        - funding_by_id.get(contract_id, Decimal(0))
        - forced_by_id.get(contract_id, Decimal(0))
        for contract_id in universe
    }
    # Derived from net_contributions rather than computed independently as
    # `gross_return - trading_cost - funding_cost - forced_close_cost`: the two
    # forms are mathematically identical, but summing four separately-aggregated
    # totals is a different order of Decimal additions than summing the same
    # money grouped per contract, and Decimal addition is not associative at its
    # default 28-significant-digit precision -- the two forms could round to
    # values differing by roughly 1e-29, silently breaking the invariant (relied
    # on by callers pooling per-contract totals across episodes) that an
    # episode's own net_contributions sum to its own net_return exactly. Summed
    # under `_NET_RETURN_CONTEXT` so this specific addition never itself needs
    # to round (see its definition) -- an exact sum is associative, so any
    # caller regrouping this same per-contract money (by episode or by
    # contract) later necessarily agrees with it to the last digit too.
    try:
        with localcontext(_NET_RETURN_CONTEXT):
            net_return = sum(net_contributions.values(), Decimal(0))
    except Inexact as error:
        raise PanelAccountingError(
            "net_return summation exceeded its precision headroom (_NET_RETURN_CONTEXT) -- "
            "net_return can no longer be guaranteed to match its own contract breakdown"
        ) from error
    denominator = Decimal(1) + gross_return
    drifted: dict[str, Decimal] = {}
    for contract_id, weight in weight_map.items():
        if contract_id in forced or denominator == 0:
            drifted[contract_id] = Decimal(0)
            continue
        drifted[contract_id] = weight * (Decimal(1) + returns[contract_id]) / denominator

    return EpisodeResult(
        sample_id=sample_id,
        member=member,
        scenario=cost_table.name,
        gross_return=gross_return,
        turnover=turnover,
        trading_cost=trading_cost,
        funding_cost=funding_cost,
        forced_close_cost=forced_close_cost,
        net_return=net_return,
        gross_exposure=sum((abs(value) for value in weight_map.values()), Decimal(0)),
        net_exposure=sum(weight_map.values(), Decimal(0)),
        forced_close_count=len(forced),
        contract_contributions=tuple(sorted(contributions.items(), key=lambda item: item[0])),
        contract_net_contributions=tuple(
            sorted(net_contributions.items(), key=lambda item: item[0])
        ),
        drifted_weights=tuple(sorted(drifted.items(), key=lambda item: item[0])),
    )


def _per_side_bps(cost_table: PanelCostTable, tier: int, fee: Decimal | None = None) -> Decimal:
    slippage = (
        cost_table.slippage_bps_per_side_tier_one
        if tier == 1
        else cost_table.slippage_bps_per_side_tier_two
    )
    return (cost_table.fee_bps_per_side if fee is None else fee) + slippage
