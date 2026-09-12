# tests/test_carry_signals.py
from decimal import Decimal
from pathlib import Path

from trading_bot.carry_config import CarrySelectionRules, load_carry_family_spec
from trading_bot.carry_signals import (
    Cohort,
    CohortEntry,
    assemble_book,
    exit_rule_pairs,
    hurdle_minimum_trailing,
    perp_leg,
    round_trip_cost_bps,
    select_control_cohort,
    select_member_cohort,
    spot_leg,
    trailing_funding,
)
from trading_bot.carry_universe import EligiblePair, PairUniverseSnapshot
from trading_bot.panel_reader import FundingEvent

WEEK_NS = 604_800_000_000_000
DECISION = 10 * WEEK_NS
SELECTION = CarrySelectionRules(decile_denominator=10, minimum_selected=2)
BASE_COSTS = load_carry_family_spec(Path("configs/funding-carry-panel-v1.json"))[0].costs.base


def event(cid: str, t: int, rate: str) -> FundingEvent:
    return FundingEvent(
        contract_id=cid,
        instrument_id=cid.split(":")[0],
        calc_time_ns=t,
        rate=Decimal(rate),
    )


def pair(i: int) -> EligiblePair:
    return EligiblePair(
        pair_id=f"P{i}:0",
        perpetual_contract_id=f"P{i}:0",
        spot_contract_id=f"P{i}:7",
        perpetual_tier=1,
        spot_tier=1,
        tier=1,
        liquidity_rank=i + 1,
        median_quote_volume=Decimal(10),
    )


def test_trailing_funding_sums_only_inside_the_window() -> None:
    events = (
        event("P0:0", DECISION - 2 * WEEK_NS, "0.5"),
        event("P0:0", DECISION - WEEK_NS + 1, "0.1"),
        event("P0:0", DECISION, "0.2"),
        event("P0:0", DECISION + 1, "9"),
    )
    assert (
        trailing_funding(events, decision_close_ns=DECISION, lookback_weeks=1)
        == Decimal("0.3")
    )
    assert trailing_funding((), decision_close_ns=DECISION, lookback_weeks=1) is None


def test_member_cohort_takes_the_top_decile_with_positive_funding() -> None:
    snapshot = PairUniverseSnapshot(DECISION, tuple(pair(i) for i in range(20)), ())
    trailing: dict[str, Decimal | None] = {
        f"P{i}:0": Decimal(i) - Decimal(5) for i in range(20)
    }  # P0..P5 <= 0
    trailing["P19:0"] = None
    cohort = select_member_cohort(snapshot, trailing=trailing, selection=SELECTION)
    assert [e.pair_id for e in cohort.entries] == ["P18:0", "P17:0"]
    assert cohort.reason_codes == ()


def test_member_cohort_is_empty_when_too_few_pay_funding() -> None:
    snapshot = PairUniverseSnapshot(DECISION, tuple(pair(i) for i in range(20)), ())
    trailing: dict[str, Decimal | None] = {
        f"P{i}:0": Decimal(-1) for i in range(20)
    }
    trailing["P3:0"] = Decimal("0.5")
    cohort = select_member_cohort(snapshot, trailing=trailing, selection=SELECTION)
    assert cohort.entries == () and cohort.reason_codes == ("NO_CARRY_COHORT",)


def test_round_trip_cost_is_both_fees_plus_two_sides_of_slippage() -> None:
    """The v1 base table: spot 10 + perpetual 5 + 2 x 5 (tier one) or 2 x 10 (tier two)."""
    assert round_trip_cost_bps(BASE_COSTS, 1) == Decimal(25)
    assert round_trip_cost_bps(BASE_COSTS, 2) == Decimal(35)


def test_hurdle_scales_the_round_trip_cost_from_the_hold_back_to_the_lookback() -> None:
    """Spec 3.3: `F_L * (H / L) / 2 >= k * c_rt`, i.e. `F_L >= 2 k c_rt L / H`."""
    tier_one = hurdle_minimum_trailing(
        cost_table=BASE_COSTS, tier=1, multiple=Decimal(2), lookback_weeks=4, hold_weeks=26
    )
    tier_two = hurdle_minimum_trailing(
        cost_table=BASE_COSTS, tier=2, multiple=Decimal(2), lookback_weeks=4, hold_weeks=26
    )
    assert tier_one == (
        Decimal(2) * Decimal(2) * Decimal(25) / Decimal(10_000) * Decimal(4) / Decimal(26)
    )
    assert tier_two == (
        Decimal(2) * Decimal(2) * Decimal(35) / Decimal(10_000) * Decimal(4) / Decimal(26)
    )
    # the declaration's worked example: 15.4 bps over four weeks on tier one,
    # 21.5 bps on tier two
    assert Decimal("0.001538") < tier_one < Decimal("0.001539")
    assert Decimal("0.002153") < tier_two < Decimal("0.002154")
    # a shorter hold has fewer weeks to earn the same round trip back
    shorter = hurdle_minimum_trailing(
        cost_table=BASE_COSTS, tier=1, multiple=Decimal(2), lookback_weeks=4, hold_weeks=13
    )
    assert shorter > tier_one
    # and a larger multiple demands proportionally more -- bounded, not equal:
    # 4/26 is not exact, so doubling the multiple and doubling the quotient
    # round differently in the last of Decimal's twenty-eight digits
    doubled = hurdle_minimum_trailing(
        cost_table=BASE_COSTS, tier=1, multiple=Decimal(4), lookback_weeks=4, hold_weeks=26
    )
    assert abs(doubled - tier_one * 2) < Decimal("1e-28")


def test_member_cohort_ranks_only_the_pairs_that_clear_their_hurdle() -> None:
    """A hurdled pair leaves the ranking but not the universe: the decile size
    still comes from every eligible pair, so a hurdle thins the book rather
    than concentrating it into whatever few pairs qualified."""
    snapshot = PairUniverseSnapshot(DECISION, tuple(pair(i) for i in range(40)), ())
    trailing: dict[str, Decimal | None] = {f"P{i}:0": Decimal(i + 1) for i in range(40)}
    hurdle = {f"P{i}:0": Decimal(36) for i in range(40)}
    cohort = select_member_cohort(
        snapshot, trailing=trailing, selection=SELECTION, hurdle=hurdle
    )
    # P35..P39 carry 36..40 and clear the bar; the decile is 40 // 10 = 4 wide,
    # not the two that `minimum_selected` would give on the five survivors
    assert [e.pair_id for e in cohort.entries] == ["P39:0", "P38:0", "P37:0", "P36:0"]
    assert cohort.hurdle_rejections == 35
    assert cohort.reason_codes == ()


def test_a_pair_without_a_hurdle_entry_is_not_hurdle_checked() -> None:
    snapshot = PairUniverseSnapshot(DECISION, tuple(pair(i) for i in range(20)), ())
    trailing: dict[str, Decimal | None] = {f"P{i}:0": Decimal(i + 1) for i in range(20)}
    hurdle = {f"P{i}:0": Decimal(100) for i in range(20)}
    del hurdle["P0:0"]
    del hurdle["P1:0"]
    cohort = select_member_cohort(
        snapshot, trailing=trailing, selection=SELECTION, hurdle=hurdle
    )
    assert [e.pair_id for e in cohort.entries] == ["P1:0", "P0:0"]
    assert cohort.hurdle_rejections == 18


def test_too_few_pairs_clear_the_hurdle_is_no_carry_cohort_not_a_skip() -> None:
    snapshot = PairUniverseSnapshot(DECISION, tuple(pair(i) for i in range(20)), ())
    trailing: dict[str, Decimal | None] = {f"P{i}:0": Decimal(i + 1) for i in range(20)}
    hurdle = {f"P{i}:0": Decimal(20) for i in range(20)}
    cohort = select_member_cohort(
        snapshot, trailing=trailing, selection=SELECTION, hurdle=hurdle
    )
    assert cohort.entries == () and cohort.reason_codes == ("NO_CARRY_COHORT",)
    assert cohort.hurdle_rejections == 19


def test_cohort_without_a_hurdle_reports_no_rejections() -> None:
    snapshot = PairUniverseSnapshot(DECISION, tuple(pair(i) for i in range(20)), ())
    trailing: dict[str, Decimal | None] = {f"P{i}:0": Decimal(i + 1) for i in range(20)}
    assert (
        select_member_cohort(snapshot, trailing=trailing, selection=SELECTION).hurdle_rejections
        == 0
    )


def test_exit_rule_takes_every_held_pair_whose_week_paid_nothing() -> None:
    """Spec 3.3: a held pair whose trailing one week is `<= 0`, or which had no
    settlement in that week at all, leaves every cohort that holds it."""

    def cohort(week: int, ids: list[str]) -> Cohort:
        return Cohort(
            week * WEEK_NS,
            tuple(
                CohortEntry(i, perp_leg(i), spot_leg(i.replace(":0", ":7")), 1) for i in ids
            ),
            (),
        )

    cohorts = [cohort(9, ["A:0", "B:0"]), cohort(10, ["B:0", "C:0", "D:0"])]
    trailing_one_week: dict[str, Decimal | None] = {
        "A:0": Decimal("0.0001"),
        "B:0": Decimal(0),
        "C:0": Decimal("-0.0001"),
        "D:0": None,
    }
    assert exit_rule_pairs(cohorts, trailing_one_week=trailing_one_week) == {
        "B:0", "C:0", "D:0"
    }
    assert exit_rule_pairs([], trailing_one_week=trailing_one_week) == set()
    # a pair the caller never measured counts as having paid nothing
    assert exit_rule_pairs(cohorts, trailing_one_week={}) == {"A:0", "B:0", "C:0", "D:0"}


def test_controls() -> None:
    snapshot = PairUniverseSnapshot(DECISION, tuple(pair(i) for i in range(20)), ())
    trailing: dict[str, Decimal | None] = {
        f"P{i}:0": Decimal(1) if i % 2 else Decimal(-1) for i in range(20)
    }
    everything = select_control_cohort(
        snapshot, kind="all_pairs", trailing=trailing, selection=SELECTION, random_seed=17
    )
    assert len(everything.entries) == 10 and all(
        int(e.pair_id[1:].split(":")[0]) % 2 for e in everything.entries
    )
    random_a = select_control_cohort(
        snapshot,
        kind="random_pairs",
        trailing=trailing,
        selection=SELECTION,
        random_seed=17,
    )
    random_b = select_control_cohort(
        snapshot,
        kind="random_pairs",
        trailing=trailing,
        selection=SELECTION,
        random_seed=17,
    )
    assert random_a.entries == random_b.entries and len(random_a.entries) == 2
    assert (
        select_control_cohort(
            snapshot,
            kind="no_trade",
            trailing=trailing,
            selection=SELECTION,
            random_seed=17,
        ).entries
        == ()
    )


def test_book_holds_the_last_h_cohorts_at_equal_capital() -> None:
    def cohort(week: int, ids: list[str]) -> Cohort:
        return Cohort(
            week * WEEK_NS,
            tuple(
                CohortEntry(i, perp_leg(i), spot_leg(i.replace(":0", ":7")), 1)
                for i in ids
            ),
            (),
        )

    cohorts = (
        cohort(6, ["Z:0"]),
        cohort(7, ["A:0", "B:0"]),
        cohort(8, ["C:0"]),
        cohort(9, ["A:0"]),
        cohort(10, ["D:0", "E:0", "F:0", "G:0"]),
    )
    book = dict(assemble_book(cohorts, hold_weeks=4, decision_close_ns=10 * WEEK_NS))
    # week 6 is out of the window: 10 - 6 = 4 >= H; weeks 7..10 are retained
    assert "perp:Z:0" not in book
    # each cohort holds 1/4; a pair in a two-pair cohort holds 1/8,
    # split +1/16 spot / -1/16 perp
    assert book["perp:B:0"] == Decimal("-0.0625")
    assert book["spot:C:7"] == Decimal("0.125") and book["perp:C:0"] == Decimal(
        "-0.125"
    )
    # A:0 is in the week-7 and week-9 cohorts: 1/16 + 1/8 = 3/16
    assert book["perp:A:0"] == Decimal("-0.1875")
    assert book["perp:D:0"] == Decimal("-0.03125")
    assert sum(abs(v) for v in book.values()) == Decimal(1)
    assert sum(book.values()) == Decimal(0)
    assert list(book) == sorted(book)


def test_book_at_thirteen_weeks_drops_the_fourteenth_cohort() -> None:
    """The longest declared hold, H = 13: the book is the last thirteen cohorts
    and the fourteenth is out, with unit gross split across 13 * 2 legs. The
    1/13 share is not exact in Decimal, so gross is bounded, not equal."""
    cohorts = tuple(
        Cohort(
            week * WEEK_NS,
            (CohortEntry(f"P{week}:0", f"perp:P{week}:0", f"spot:P{week}:7", 1),),
            (),
        )
        for week in range(14)
    )
    book = dict(assemble_book(cohorts, hold_weeks=13, decision_close_ns=13 * WEEK_NS))
    # week 0 is out of the window: 13 - 0 = 13 >= H
    assert "perp:P0:0" not in book and "spot:P0:7" not in book
    assert len(book) == 26
    share = Decimal(1) / Decimal(13) / 2
    assert all(abs(weight) == share for weight in book.values())
    assert abs(sum(abs(weight) for weight in book.values()) - Decimal(1)) < Decimal("1e-25")


def test_book_with_an_empty_cohort_deploys_less_than_unit_gross() -> None:
    cohorts = (
        Cohort(9 * WEEK_NS, (), ("NO_CARRY_COHORT",)),
        Cohort(
            10 * WEEK_NS,
            (CohortEntry("A:0", "perp:A:0", "spot:A:7", 1),),
            (),
        ),
    )
    book = dict(assemble_book(cohorts, hold_weeks=2, decision_close_ns=10 * WEEK_NS))
    assert sum(abs(v) for v in book.values()) == Decimal("0.5")


def test_a_shrunken_cohort_leaves_its_stripped_pair_undeployed() -> None:
    """A pair stripped from a cohort after a forced close is not reinvested into
    its surviving siblings: the cohort still divides its capital by the size it
    was formed with (`formed_size`), not by how many entries remain."""
    cohorts = (
        Cohort(
            10 * WEEK_NS,
            (CohortEntry("A:0", "perp:A:0", "spot:A:7", 1),),
            (),
            formed_size=2,
        ),
    )
    book = dict(assemble_book(cohorts, hold_weeks=4, decision_close_ns=10 * WEEK_NS))
    # H=4, formed with 2 entries: 1/4 / 2 = 1/8 gross for this pair, not 1/4.
    assert book["spot:A:7"] == Decimal("0.0625")
    assert book["perp:A:0"] == Decimal("-0.0625")
    assert sum(abs(v) for v in book.values()) == Decimal("0.125")
