"""Trailing funding, cohort selection and book assembly for the carry family."""

from dataclasses import dataclass
from decimal import Decimal
from random import Random

from trading_bot.carry_config import CarryCostTable, CarrySelectionRules
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
    # The entry count the cohort was formed with, when that differs from
    # ``len(entries)`` today (a pair stripped out after a forced close).
    # ``None`` means "formed with exactly len(entries)", the common case.
    formed_size: int | None = None
    # How many pairs that were otherwise rankable at formation were kept out
    # by the cost hurdle (spec 3.4). Zero for a member without a hurdle.
    hurdle_rejections: int = 0


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


def round_trip_cost_bps(cost_table: CarryCostTable, tier: int) -> Decimal:
    """The cost of entering and leaving one unit of pair capital, in bps.

    Both legs are crossed on the way in and on the way out, so the two fees
    are paid once each per unit of pair capital and the tier's slippage twice
    -- 25 bps on tier one and 35 on tier two under the v1 base table.
    """
    slippage = (
        cost_table.slippage_bps_per_side_tier_one
        if tier == 1
        else cost_table.slippage_bps_per_side_tier_two
    )
    return cost_table.spot_fee_bps_per_side + cost_table.perpetual_fee_bps_per_side + 2 * slippage


def hurdle_minimum_trailing(
    *,
    cost_table: CarryCostTable,
    tier: int,
    multiple: Decimal,
    lookback_weeks: int,
    hold_weeks: int,
) -> Decimal:
    """The smallest trailing `L`-week funding a pair may carry and still enter.

    Spec 3.3 states the hurdle as `F_L * (H / L) / 2 >= k * c_rt(tier)`: the
    trailing sum is scaled from the lookback to the hold, halved because only
    half of a pair's capital sits on the funding leg, and compared against a
    multiple of the round trip. Solved for `F_L` that is
    `2 * k * c_rt * L / H`, per unit of perpetual notional and as a fraction
    rather than bps, which is the unit funding rates come in.
    """
    return (
        Decimal(2)
        * multiple
        * round_trip_cost_bps(cost_table, tier)
        / Decimal(10_000)
        * Decimal(lookback_weeks)
        / Decimal(hold_weeks)
    )


def select_member_cohort(
    snapshot: PairUniverseSnapshot,
    *,
    trailing: dict[str, Decimal | None],
    selection: CarrySelectionRules,
    hurdle: dict[str, Decimal] | None = None,
) -> Cohort:
    """Top decile of the eligible pairs by trailing funding, positive funding required.

    `hurdle` maps a pair id to the minimum trailing funding it must carry to
    be ranked at all (spec 3.3). A pair with no entry is not hurdle-checked.
    The decile width is unchanged by the hurdle: it is still computed from
    every eligible pair, so hurdling pairs out thins the cohort rather than
    concentrating the same capital into whichever few pairs qualified.
    """
    paying = [
        (value, pair.pair_id)
        for pair in snapshot.pairs
        for value in (trailing.get(pair.pair_id),)
        if value is not None and value > 0
    ]
    rejections = 0
    if hurdle is not None:
        qualified = []
        for value, pair_id in paying:
            minimum = hurdle.get(pair_id)
            if minimum is not None and value < minimum:
                rejections += 1
                continue
            qualified.append((value, pair_id))
        paying = qualified
    if len(paying) < selection.minimum_selected:
        return Cohort(
            snapshot.decision_close_ns, (), (NO_CARRY_COHORT,), hurdle_rejections=rejections
        )
    paying.sort(key=lambda item: (-item[0], item[1]))
    size = min(len(paying), _decile_size(len(snapshot.pairs), selection))
    return Cohort(
        snapshot.decision_close_ns,
        _entries(snapshot, [pid for _, pid in paying[:size]]),
        (),
        hurdle_rejections=rejections,
    )


def exit_rule_pairs(
    cohorts: list[Cohort], *, trailing_one_week: dict[str, Decimal | None]
) -> set[str]:
    """Pair ids held in any cohort whose trailing one-week funding is None or `<= 0`.

    Spec 3.3's exit rule. A pair the caller did not measure is treated the
    same as one that had no settlement in the week: it paid nothing, so it is
    not held for another week.
    """
    return {
        entry.pair_id
        for cohort in cohorts
        for entry in cohort.entries
        for value in (trailing_one_week.get(entry.pair_id),)
        if value is None or value <= 0
    }


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
    leaves its share of capital undeployed rather than redistributing it,
    and so does a pair removed from a cohort after a forced close: its share
    stays undeployed until the cohort ages out, rather than being reinvested
    into its surviving siblings. ``Cohort.formed_size`` carries the entry
    count the cohort was assembled with, so a shrunken cohort still divides
    its capital by the count it started with, not by how many pairs remain.
    """
    if hold_weeks < 1:
        raise ValueError("hold_weeks must be positive")
    share_per_cohort = Decimal(1) / Decimal(hold_weeks)
    weights: dict[str, Decimal] = {}
    for cohort in cohorts:
        age = decision_close_ns - cohort.decision_close_ns
        if age < 0 or age >= hold_weeks * WEEK_NS or not cohort.entries:
            continue
        size = cohort.formed_size if cohort.formed_size is not None else len(cohort.entries)
        pair_share = share_per_cohort / Decimal(size)
        for entry in cohort.entries:
            weights[entry.spot_leg] = weights.get(entry.spot_leg, Decimal(0)) + pair_share / 2
            weights[entry.perpetual_leg] = (
                weights.get(entry.perpetual_leg, Decimal(0)) - pair_share / 2
            )
    return tuple(sorted(weights.items()))
