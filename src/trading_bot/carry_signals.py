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


def rank_paying_pairs(
    snapshot: PairUniverseSnapshot,
    *,
    trailing: dict[str, Decimal | None],
    selection: CarrySelectionRules,
    hurdle: dict[str, Decimal] | None = None,
) -> tuple[list[str], int]:
    """The top-decile ranking `select_member_cohort` builds its entries from.

    Positive trailing funding only, the hurdle applied exactly as spec 3.3
    states it (a pair with no hurdle entry is not checked), sorted by
    `(-funding, pair_id)` and cut to the decile width computed from every
    eligible pair -- the hurdle can thin the ranking but never widens the
    decile it is cut to. Returns `([], rejections)` when fewer than
    `minimum_selected` pairs clear the hurdle, the same "not enough to form a
    cohort" case `select_member_cohort` reports as `NO_CARRY_COHORT`. A slot
    book (spec 4.2) fills its empty slots from this same ranking rather than
    from a fresh one, so cohort and slot members agree on who is carrying
    funding this week.
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
        return [], rejections
    paying.sort(key=lambda item: (-item[0], item[1]))
    size = min(len(paying), _decile_size(len(snapshot.pairs), selection))
    return [pid for _, pid in paying[:size]], rejections


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
    every eligible pair, so the hurdle can only cap how many pairs a cohort
    holds, never widen it. A cohort that ends up with fewer pairs still
    deploys its full 1/H share across them (`assemble_book` divides by the
    entry count), so the per-pair weight is bounded only by
    `minimum_selected`, which the declaration fixes at eight. Expressed
    through `rank_paying_pairs` so the two never drift apart.
    """
    ids, rejections = rank_paying_pairs(
        snapshot, trailing=trailing, selection=selection, hurdle=hurdle
    )
    if not ids:
        return Cohort(
            snapshot.decision_close_ns, (), (NO_CARRY_COHORT,), hurdle_rejections=rejections
        )
    return Cohort(
        snapshot.decision_close_ns,
        _entries(snapshot, ids),
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


def random_pair_order(
    snapshot: PairUniverseSnapshot, *, selection: CarrySelectionRules, random_seed: int
) -> list[str]:
    """The seeded shuffle `select_control_cohort` draws `random_pairs` from.

    Every pair id, seed-shuffled and cut to the decile width, in shuffle
    order rather than sorted -- `select_control_cohort` sorts its cohort's
    entries afterward, but a slot book (spec 4.2) fills its newest empty
    slots in the order the draw names them, so it needs the unsorted order.
    """
    ids = sorted(pair.pair_id for pair in snapshot.pairs)
    Random(random_seed ^ snapshot.decision_close_ns).shuffle(ids)
    size = min(len(ids), _decile_size(len(snapshot.pairs), selection))
    return ids[:size]


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
        ids = random_pair_order(snapshot, selection=selection, random_seed=random_seed)
        return Cohort(
            snapshot.decision_close_ns, _entries(snapshot, sorted(ids)), ()
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


def fill_slots(
    ranked: list[str], *, held: set[str], skip: set[str], free: int
) -> list[str]:
    """The first `free` ranked ids that are neither already held nor skipped.

    A slot book fills its empty slots from the newest ranking (`rank_paying_
    pairs` for members, `random_pair_order` for the random control) in rank
    order, passing over a pair already holding a slot (`held`) and one this
    decision must not re-enter (`skip` -- exited this week, or otherwise put
    out of bounds). Fewer than `free` ids come back when the ranking runs out
    first; a slot the ranking has nothing left for simply stays empty.
    """
    filled: list[str] = []
    for pair_id in ranked:
        if len(filled) == free:
            break
        if pair_id in held or pair_id in skip:
            continue
        filled.append(pair_id)
    return filled


def assemble_slot_book(
    slots: tuple[Cohort, ...], *, pair_slots: int, hold_weeks: int, decision_close_ns: int
) -> tuple[tuple[str, Decimal], ...]:
    """Union of the still-held slots, each on a fixed `1/pair_slots` share.

    A slot is a `Cohort` with exactly one entry -- the pair filling it. Unlike
    `assemble_book`'s weekly cohorts, a slot's share never divides by an entry
    count: it is `1/pair_slots` of the declared book regardless of how many of
    the other slots are filled, because an empty slot leaves its share
    undeployed rather than being redistributed onto its neighbours (spec
    4.2's declared, not proportional, capital). The age test is
    `assemble_book`'s, unchanged: a slot still counts while
    `0 <= decision_close_ns - slot.decision_close_ns < hold_weeks` weeks, and
    ages out the decision it turns `hold_weeks` old. A cohort with more than
    one entry is not a slot and is refused rather than silently read as one.
    """
    if pair_slots < 1:
        raise ValueError("pair_slots must be positive")
    share = Decimal(1) / Decimal(pair_slots)
    weights: dict[str, Decimal] = {}
    for slot in slots:
        if len(slot.entries) > 1:
            raise ValueError("a slot cohort holds at most one entry")
        age = decision_close_ns - slot.decision_close_ns
        if age < 0 or age >= hold_weeks * WEEK_NS or not slot.entries:
            continue
        entry = slot.entries[0]
        weights[entry.spot_leg] = weights.get(entry.spot_leg, Decimal(0)) + share / 2
        weights[entry.perpetual_leg] = (
            weights.get(entry.perpetual_leg, Decimal(0)) - share / 2
        )
    return tuple(sorted(weights.items()))
