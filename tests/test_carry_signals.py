# tests/test_carry_signals.py
from decimal import Decimal

from trading_bot.carry_config import CarrySelectionRules
from trading_bot.carry_signals import (
    Cohort,
    CohortEntry,
    assemble_book,
    perp_leg,
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
