"""Weight vectors for the frozen panel members and controls."""

from dataclasses import dataclass
from decimal import Decimal
from itertools import pairwise
from random import Random

from trading_bot.panel_config import PanelFamilySpec, PanelWeightRules
from trading_bot.panel_universe import ContractHistory, UniverseSnapshot

DAY_NS = 86_400_000_000_000


@dataclass(frozen=True, slots=True)
class WeightVector:
    decision_close_ns: int
    member: str
    weights: tuple[tuple[str, Decimal], ...]
    reason_codes: tuple[str, ...]


def build_weight_vectors(
    histories: dict[str, ContractHistory],
    snapshot: UniverseSnapshot,
    *,
    spec: PanelFamilySpec,
) -> dict[str, WeightVector]:
    """Build every member and control weight vector for one rebalance."""
    decision = snapshot.decision_close_ns
    names = [member.name for member in spec.members] + [item.name for item in spec.controls]
    if not snapshot.contracts:
        return {name: WeightVector(decision, name, (), snapshot.reason_codes) for name in names}

    eligible = tuple(item.contract_id for item in snapshot.contracts)
    vectors: dict[str, WeightVector] = {}
    for member in spec.members:
        returns = _trailing_returns(histories, eligible, decision, member.lookback_days)
        weights: tuple[tuple[str, Decimal], ...]
        if member.kind == "cross_sectional":
            weights = _cross_sectional_weights(returns, spec.weights, reverse=member.reversed)
        else:
            weights = _time_series_weights(returns, histories, eligible, decision, spec.weights)
        vectors[member.name] = WeightVector(decision, member.name, weights, ())

    vectors["no_trade"] = WeightVector(decision, "no_trade", (), ())
    share = Decimal(1) / Decimal(len(eligible))
    vectors["passive_long_ew"] = WeightVector(
        decision,
        "passive_long_ew",
        tuple((contract_id, share) for contract_id in sorted(eligible)),
        (),
    )
    rng = Random(spec.statistics.random_seed ^ decision)
    scores = {contract_id: Decimal(str(rng.random())) for contract_id in eligible}
    vectors["random_ranks"] = WeightVector(
        decision,
        "random_ranks",
        _cross_sectional_weights(scores, spec.weights, reverse=False),
        (),
    )
    return vectors


def _trailing_returns(
    histories: dict[str, ContractHistory],
    eligible: tuple[str, ...],
    decision_close_ns: int,
    lookback_days: int,
) -> dict[str, Decimal]:
    lagged_close_ns = decision_close_ns - lookback_days * DAY_NS
    returns: dict[str, Decimal] = {}
    for contract_id in eligible:
        closes = histories[contract_id].closes
        current = closes.get(decision_close_ns)
        previous = closes.get(lagged_close_ns)
        if current is None or previous is None or previous <= 0:
            continue
        returns[contract_id] = current / previous - Decimal(1)
    return returns


def _cross_sectional_weights(
    returns: dict[str, Decimal], rules: PanelWeightRules, *, reverse: bool
) -> tuple[tuple[str, Decimal], ...]:
    ranked = sorted(returns.items(), key=lambda item: (item[1], item[0]))
    quintile = max(rules.minimum_quintile_size, len(ranked) // 5)
    if len(ranked) < 2 * quintile:
        return ()
    leg = rules.leg_gross / Decimal(quintile)
    losers = ranked[:quintile]
    winners = ranked[-quintile:]
    sign = Decimal(-1) if reverse else Decimal(1)
    weights = [(contract_id, -leg * sign) for contract_id, _ in losers]
    weights += [(contract_id, leg * sign) for contract_id, _ in winners]
    return tuple(sorted(weights, key=lambda item: item[0]))


def _time_series_weights(
    returns: dict[str, Decimal],
    histories: dict[str, ContractHistory],
    eligible: tuple[str, ...],
    decision_close_ns: int,
    rules: PanelWeightRules,
) -> tuple[tuple[str, Decimal], ...]:
    raw: dict[str, Decimal] = {}
    for contract_id in eligible:
        trailing = returns.get(contract_id)
        if trailing is None or trailing == 0:
            continue
        sigma = _annualised_volatility(
            histories[contract_id], decision_close_ns, rules.volatility_window_days
        )
        if sigma is None:
            continue
        raw[contract_id] = (Decimal(1) if trailing > 0 else Decimal(-1)) / max(
            sigma, rules.volatility_floor
        )
    if not raw:
        return ()
    cap = rules.time_series_cap_numerator / Decimal(len(raw))
    weights = _water_fill(_normalise(raw), cap)
    return tuple(sorted(weights.items(), key=lambda item: item[0]))


def _water_fill(weights: dict[str, Decimal], cap: Decimal) -> dict[str, Decimal]:
    """Clip every weight to `cap`, redistributing the rest so gross stays exactly one.

    Once an entry is clipped to `cap` it is frozen there permanently and never
    rescaled again; only entries that have never been frozen absorb the residual
    budget. That makes the frozen set grow monotonically round over round (an
    entry can newly join it, but never leave), so this terminates within
    `len(weights)` rounds, and the result satisfies both constraints exactly: no
    weight's absolute value exceeds `cap`, and the absolute values sum to exactly
    one. A frozen entry keeps its own sign.

    (An earlier version recomputed the violator set from scratch each round
    instead of accumulating it, so an already-clipped entry could sit in "others"
    on a later round and be rescaled back above the cap — the violator set
    oscillated rather than grew, and the loop could exhaust its round budget
    still over cap. Freezing cumulatively is what fixes that.)
    """
    current = dict(weights)
    frozen: set[str] = set()
    for _ in range(len(current)):
        new_violators = [
            key for key, value in current.items() if key not in frozen and abs(value) > cap
        ]
        if not new_violators:
            break
        frozen.update(new_violators)
        for key in new_violators:
            current[key] = cap if current[key] >= 0 else -cap
        frozen_gross = Decimal(len(frozen)) * cap
        if frozen_gross >= 1:
            # Every frozen contract already meets or exceeds unit gross on its
            # own; equal weights are always feasible because cap == 2/n and
            # 1/n <= 2/n.
            return {
                key: (Decimal(1) if value >= 0 else Decimal(-1)) / Decimal(len(current))
                for key, value in current.items()
            }
        others = [key for key in current if key not in frozen]
        others_gross = sum(abs(current[key]) for key in others)
        scale = (Decimal(1) - frozen_gross) / others_gross if others_gross else Decimal(0)
        for key in others:
            current[key] = current[key] * scale
    return current


def _normalise(weights: dict[str, Decimal]) -> dict[str, Decimal]:
    gross = sum((abs(value) for value in weights.values()), Decimal(0))
    if gross == 0:
        return weights
    return {contract_id: value / gross for contract_id, value in weights.items()}


def _annualised_volatility(
    history: ContractHistory, decision_close_ns: int, window_days: int
) -> Decimal | None:
    observed = [value for value in history.close_times if value <= decision_close_ns]
    window = observed[-(window_days + 1) :]
    if len(window) < window_days + 1:
        return None
    log_returns: list[Decimal] = []
    for previous, current in pairwise(window):
        earlier = history.closes[previous]
        later = history.closes[current]
        if earlier <= 0 or later <= 0:
            return None
        log_returns.append((later / earlier).ln())
    count = Decimal(len(log_returns))
    mean = sum(log_returns, Decimal(0)) / count
    variance = sum(((value - mean) ** 2 for value in log_returns), Decimal(0)) / (
        count - Decimal(1)
    )
    return variance.sqrt() * Decimal(365).sqrt()
