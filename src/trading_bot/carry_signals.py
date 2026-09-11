"""Trailing funding, cohort selection and book assembly for the carry family."""

from dataclasses import dataclass
from decimal import Decimal
from random import Random

from trading_bot.carry_config import CarrySelectionRules
from trading_bot.carry_universe import PairUniverseSnapshot
from trading_bot.panel_reader import FundingEvent

WEEK_NS = 604_800_000_000_000
NO_CARRY_COHORT = "NO_CARRY_COHORT"


def perp_leg(contract_id: str) -> str:
    return f"perp:{contract_id}"


def spot_leg(contract_id: str) -> str:
    return f"spot:{contract_id}"


@dataclass(frozen=True, slots=True)
class CohortEntry:
    pair_id: str
    perpetual_leg: str
    spot_leg: str
    tier: int


@dataclass(frozen=True, slots=True)
class Cohort:
    decision_close_ns: int
    entries: tuple[CohortEntry, ...]
    reason_codes: tuple[str, ...]


def trailing_funding(
    events: tuple[FundingEvent, ...], *, decision_close_ns: int, lookback_weeks: int
) -> Decimal | None:
    """Sum the settlements in ``(decision - L weeks, decision]``; None if there are none."""
    start = decision_close_ns - lookback_weeks * WEEK_NS
    inside = [e.rate for e in events if start < e.calc_time_ns <= decision_close_ns]
    if not inside:
        return None
    return sum(inside, Decimal(0))


def _entries(
    snapshot: PairUniverseSnapshot, pair_ids: list[str]
) -> tuple[CohortEntry, ...]:
    by_id = {pair.pair_id: pair for pair in snapshot.pairs}
    return tuple(
        CohortEntry(
            pair_id=pair_id,
            perpetual_leg=perp_leg(by_id[pair_id].perpetual_contract_id),
            spot_leg=spot_leg(by_id[pair_id].spot_contract_id),
            tier=by_id[pair_id].tier,
        )
        for pair_id in pair_ids
    )


def _decile_size(count: int, selection: CarrySelectionRules) -> int:
    return max(selection.minimum_selected, count // selection.decile_denominator)


def select_member_cohort(
    snapshot: PairUniverseSnapshot,
    *,
    trailing: dict[str, Decimal | None],
    selection: CarrySelectionRules,
) -> Cohort:
    """Top decile of the eligible pairs by trailing funding, positive funding required."""
    paying = [
        (value, pair.pair_id)
        for pair in snapshot.pairs
        for value in (trailing.get(pair.pair_id),)
        if value is not None and value > 0
    ]
    if len(paying) < selection.minimum_selected:
        return Cohort(snapshot.decision_close_ns, (), (NO_CARRY_COHORT,))
    paying.sort(key=lambda item: (-item[0], item[1]))
    size = min(len(paying), _decile_size(len(snapshot.pairs), selection))
    return Cohort(
        snapshot.decision_close_ns,
        _entries(snapshot, [pid for _, pid in paying[:size]]),
        (),
    )


def select_control_cohort(
    snapshot: PairUniverseSnapshot,
    *,
    kind: str,
    trailing: dict[str, Decimal | None],
    selection: CarrySelectionRules,
    random_seed: int,
) -> Cohort:
    if kind == "no_trade":
        return Cohort(snapshot.decision_close_ns, (), ())
    if kind == "all_pairs":
        ids = sorted(
            pair.pair_id
            for pair in snapshot.pairs
            for value in (trailing.get(pair.pair_id),)
            if value is not None and value > 0
        )
        return Cohort(snapshot.decision_close_ns, _entries(snapshot, ids), ())
    if kind == "random_pairs":
        ids = sorted(pair.pair_id for pair in snapshot.pairs)
        Random(random_seed ^ snapshot.decision_close_ns).shuffle(ids)
        size = min(len(ids), _decile_size(len(snapshot.pairs), selection))
        return Cohort(
            snapshot.decision_close_ns, _entries(snapshot, sorted(ids[:size])), ()
        )
    raise ValueError(f"unknown control kind: {kind}")


def assemble_book(
    cohorts: tuple[Cohort, ...], *, hold_weeks: int, decision_close_ns: int
) -> tuple[tuple[str, Decimal], ...]:
    """Union of the last H cohorts, each on 1/H of capital, as leg weights.

    A pair share ``c`` is expressed as spot ``+c/2`` and perpetual ``-c/2``,
    so a full book has unit gross across both legs and zero net exposure,
    which is the capital basis spec section 3.1 declares. An empty cohort
    leaves its share of capital undeployed rather than redistributing it.
    """
    if hold_weeks < 1:
        raise ValueError("hold_weeks must be positive")
    share_per_cohort = Decimal(1) / Decimal(hold_weeks)
    weights: dict[str, Decimal] = {}
    for cohort in cohorts:
        age = decision_close_ns - cohort.decision_close_ns
        if age < 0 or age >= hold_weeks * WEEK_NS or not cohort.entries:
            continue
        pair_share = share_per_cohort / Decimal(len(cohort.entries))
        for entry in cohort.entries:
            weights[entry.spot_leg] = weights.get(entry.spot_leg, Decimal(0)) + pair_share / 2
            weights[entry.perpetual_leg] = (
                weights.get(entry.perpetual_leg, Decimal(0)) - pair_share / 2
            )
    return tuple(sorted(weights.items()))
