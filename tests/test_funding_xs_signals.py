from decimal import Decimal
from pathlib import Path

from trading_bot.funding_xs_config import load_funding_xs_family_spec
from trading_bot.funding_xs_signals import (
    build_funding_xs_weight_vectors,
    funding_scores,
    sign_flip_exits,
)
from trading_bot.panel_config import load_panel_family_spec
from trading_bot.panel_reader import FundingEvent
from trading_bot.panel_signals import build_weight_vectors
from trading_bot.panel_universe import ContractHistory, EligibleContract, UniverseSnapshot

WEEK_NS = 604_800_000_000_000
DECISION = 520 * WEEK_NS
BP = Decimal("0.0001")
PANEL_SIZE = 10

SPEC, _ = load_funding_xs_family_spec(Path("configs/funding-xs-panel-v1.json"))
PANEL_SPEC, _ = load_panel_family_spec(Path("configs/xs-momentum-panel-v1.json"))

# The declared minimum quintile size is 8, which needs sixteen scored
# contracts before any book forms. The hand-computed books below run on ten,
# so both specs are reduced to a minimum of 2 -- the only change, and the same
# change on both sides, so the controls stay comparable.
REDUCED_WEIGHTS = SPEC.weights.model_copy(update={"minimum_quintile_size": 2})
REDUCED_SPEC = SPEC.model_copy(update={"weights": REDUCED_WEIGHTS})
REDUCED_PANEL_SPEC = PANEL_SPEC.model_copy(update={"weights": REDUCED_WEIGHTS})


def contract_id(index: int) -> str:
    """`F00USDT:0` .. `F09USDT:0`: contract-id order equals index order."""
    return f"F{index:02d}USDT:0"


def event(contract: str, calc_time_ns: int, rate: Decimal) -> FundingEvent:
    return FundingEvent(
        contract_id=contract,
        instrument_id=contract.split(":")[0],
        calc_time_ns=calc_time_ns,
        rate=rate,
    )


def history_for(contract: str) -> ContractHistory:
    """A one-close history: the funding book reads no prices at all.

    It exists so P1.27's builder -- which does read prices -- can run on the
    same snapshot for the control comparison below.
    """
    return ContractHistory(
        contract_id=contract,
        instrument_id=contract.split(":")[0],
        closes={DECISION: Decimal(100)},
        quote_volumes={DECISION: Decimal("10000000")},
        close_times=(DECISION,),
    )


def panel_histories(count: int = PANEL_SIZE) -> dict[str, ContractHistory]:
    return {contract_id(index): history_for(contract_id(index)) for index in range(count)}


def snapshot_for(histories: dict[str, ContractHistory]) -> UniverseSnapshot:
    return UniverseSnapshot(
        decision_close_ns=DECISION,
        contracts=tuple(
            EligibleContract(
                contract_id=contract,
                median_quote_volume=Decimal("10000000"),
                liquidity_rank=rank,
                tier=1,
            )
            for rank, contract in enumerate(sorted(histories), start=1)
        ),
        reason_codes=(),
    )


def two_settlement_funding(count: int = PANEL_SIZE) -> dict[str, tuple[FundingEvent, ...]]:
    """Contract k settles `2k` bp three weeks before the decision and `9 - k` bp at it.

    Hand-derived scores (the window is the half-open `(t - L weeks, t]`):

        L = 1 week  -> only the settlement at t:        (9 - k) bp
        L = 4 weeks -> both settlements:   2k + (9 - k) = (9 + k) bp

    So the four-week ranking rises with k (F00 lowest at 9 bp, F09 highest at
    18 bp) and the one-week ranking falls with k (F09 lowest at 0 bp, F00
    highest at 9 bp): the lookback alone reverses the book.
    """
    return {
        contract_id(index): (
            event(contract_id(index), DECISION - 3 * WEEK_NS, Decimal(2 * index) * BP),
            event(contract_id(index), DECISION, Decimal(9 - index) * BP),
        )
        for index in range(count)
    }


# --------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------


def test_funding_scores_sums_only_the_lookback_window() -> None:
    """The window is `(t - L weeks, t]`: open at the left, closed at the right.

    `AAAUSDT:0` settles
        t - 4 weeks         ->  100 bp, outside (the left end is excluded),
        t - 4 weeks + 1 ns  ->    3 bp, inside,
        t - 1 week          ->    2 bp, inside,
        t                   ->   -4 bp, inside (the right end is included),
        t + 1 ns            -> -100 bp, outside (after the decision).
    So the four-week score is 3 + 2 - 4 = 1 bp and the one-week score, whose
    window starts at t - 1 week, is -4 bp (the settlement exactly one week
    back falls on the excluded left end).
    """
    contract = "AAAUSDT:0"
    events = (
        event(contract, DECISION - 4 * WEEK_NS, Decimal(100) * BP),
        event(contract, DECISION - 4 * WEEK_NS + 1, Decimal(3) * BP),
        event(contract, DECISION - WEEK_NS, Decimal(2) * BP),
        event(contract, DECISION, Decimal(-4) * BP),
        event(contract, DECISION + 1, Decimal(-100) * BP),
    )
    scores = funding_scores({contract: events}, (contract,), DECISION, 4)
    assert scores == {contract: Decimal(1) * BP}
    assert funding_scores({contract: events}, (contract,), DECISION, 1) == {
        contract: Decimal(-4) * BP
    }


def test_funding_scores_omit_contracts_without_a_settlement_in_the_window() -> None:
    """Spec 3: a contract with no settlement in the window has no score.

    Three ways to have none -- every settlement older than the window, every
    settlement after the decision, and no settlement record at all -- and all
    three drop the contract rather than scoring it zero, which would rank it
    in the middle of the cross-section.
    """
    stale = event("STALEUSDT:0", DECISION - 5 * WEEK_NS, Decimal(7) * BP)
    future = event("LATERUSDT:0", DECISION + WEEK_NS, Decimal(7) * BP)
    scored = event("SCOREDUSDT:0", DECISION - WEEK_NS, Decimal(7) * BP)
    funding = {
        "STALEUSDT:0": (stale,),
        "LATERUSDT:0": (future,),
        "SCOREDUSDT:0": (scored,),
        "EMPTYUSDT:0": (),
    }
    eligible = ("STALEUSDT:0", "LATERUSDT:0", "SCOREDUSDT:0", "EMPTYUSDT:0", "MISSINGUSDT:0")
    assert funding_scores(funding, eligible, DECISION, 4) == {"SCOREDUSDT:0": Decimal(7) * BP}


def test_funding_scores_only_cover_the_eligible_contracts() -> None:
    funding = two_settlement_funding()
    eligible = (contract_id(0), contract_id(1))
    scores = funding_scores(funding, eligible, DECISION, 4)
    # 9 + k bp for k in (0, 1).
    assert scores == {contract_id(0): Decimal(9) * BP, contract_id(1): Decimal(10) * BP}


# --------------------------------------------------------------------------
# Weight vectors
# --------------------------------------------------------------------------


def test_the_book_is_long_the_lowest_and_short_the_highest_quintile() -> None:
    """Ten scored contracts, minimum quintile 2: quintile = max(2, 10 // 5) = 2.

    Leg gross 0.5 over two names is 0.25 each. On the four-week score
    (9 + k bp) the lowest two are F00 and F01 -- long +0.25 -- and the highest
    two are F08 and F09 -- short -0.25. Gross is 4 x 0.25 = 1 and the net is
    zero, so the book is dollar-neutral as spec 3 declares.
    """
    histories = panel_histories()
    snapshot = snapshot_for(histories)
    vectors = build_funding_xs_weight_vectors(
        histories, snapshot, two_settlement_funding(), spec=REDUCED_SPEC
    )
    expected = (
        (contract_id(0), Decimal("0.25")),
        (contract_id(1), Decimal("0.25")),
        (contract_id(8), Decimal("-0.25")),
        (contract_id(9), Decimal("-0.25")),
    )
    assert vectors["fx_q5_l4w_h1w"].weights == expected
    assert sum((weight for _, weight in expected), Decimal(0)) == Decimal(0)
    assert sum((abs(weight) for _, weight in expected), Decimal(0)) == Decimal(1)
    assert vectors["fx_q5_l4w_h1w"].decision_close_ns == DECISION
    assert vectors["fx_q5_l4w_h1w"].member == "fx_q5_l4w_h1w"
    assert vectors["fx_q5_l4w_h1w"].reason_codes == ()


def test_the_lookback_alone_reverses_the_book() -> None:
    """The one-week score falls with k where the four-week score rises, so the
    one-week member is long F08/F09 and short F00/F01 -- the mirror image of
    the four-week members' book, on the same contracts and the same snapshot."""
    histories = panel_histories()
    snapshot = snapshot_for(histories)
    vectors = build_funding_xs_weight_vectors(
        histories, snapshot, two_settlement_funding(), spec=REDUCED_SPEC
    )
    assert vectors["fx_q5_l1w_h4w"].weights == (
        (contract_id(0), Decimal("-0.25")),
        (contract_id(1), Decimal("-0.25")),
        (contract_id(8), Decimal("0.25")),
        (contract_id(9), Decimal("0.25")),
    )


def test_members_sharing_a_lookback_share_a_vector() -> None:
    """The hold and the exit rule live in the fold runner, not in the vector:
    the three four-week-lookback members must build one and the same book."""
    histories = panel_histories()
    snapshot = snapshot_for(histories)
    vectors = build_funding_xs_weight_vectors(
        histories, snapshot, two_settlement_funding(), spec=REDUCED_SPEC
    )
    four_week = ("fx_q5_l4w_h1w", "fx_q5_l4w_h4w", "fx_q5_l4w_h4w_exit")
    for name in four_week:
        assert vectors[name].weights == vectors["fx_q5_l4w_h1w"].weights, name
        assert vectors[name].member == name


def test_a_contract_without_a_score_is_not_rankable() -> None:
    """F00's settlements are dropped, leaving nine scored contracts: the
    quintile is still max(2, 9 // 5) = 2, the lowest two of the nine are now
    F01 (10 bp) and F02 (11 bp), and F00 carries no weight at all."""
    histories = panel_histories()
    snapshot = snapshot_for(histories)
    funding = dict(two_settlement_funding())
    funding[contract_id(0)] = ()
    vectors = build_funding_xs_weight_vectors(histories, snapshot, funding, spec=REDUCED_SPEC)
    assert vectors["fx_q5_l4w_h1w"].weights == (
        (contract_id(1), Decimal("0.25")),
        (contract_id(2), Decimal("0.25")),
        (contract_id(8), Decimal("-0.25")),
        (contract_id(9), Decimal("-0.25")),
    )


def test_fewer_than_two_quintiles_holds_nothing() -> None:
    """Spec 3: three scored contracts cannot fill two quintiles of two, so the
    member holds nothing -- an empty vector, not a partial book."""
    histories = panel_histories(3)
    snapshot = snapshot_for(histories)
    vectors = build_funding_xs_weight_vectors(
        histories, snapshot, two_settlement_funding(3), spec=REDUCED_SPEC
    )
    for member in REDUCED_SPEC.members:
        assert vectors[member.name].weights == (), member.name


def test_the_declared_quintile_minimum_needs_sixteen_scored_contracts() -> None:
    """Unreduced, `minimum_quintile_size` is 8: ten scored contracts fill only
    one quintile and a bit, so the declared spec holds nothing on this panel."""
    histories = panel_histories()
    snapshot = snapshot_for(histories)
    vectors = build_funding_xs_weight_vectors(
        histories, snapshot, two_settlement_funding(), spec=SPEC
    )
    for member in SPEC.members:
        assert vectors[member.name].weights == (), member.name


def test_the_controls_equal_the_panel_controls_on_the_same_snapshot() -> None:
    """The three controls are P1.27's, unchanged: `no_trade` flat,
    `passive_long_ew` equal-weight over the eligible set in contract-id order,
    and `random_ranks` drawn from `random_seed ^ decision`. The funding-xs
    config copies P1.27's `weights` and `statistics` blocks value for value,
    so the vectors must be identical objects, not merely similar ones."""
    histories = panel_histories()
    snapshot = snapshot_for(histories)
    vectors = build_funding_xs_weight_vectors(
        histories, snapshot, two_settlement_funding(), spec=REDUCED_SPEC
    )
    panel_vectors = build_weight_vectors(histories, snapshot, spec=REDUCED_PANEL_SPEC)
    for name in ("no_trade", "random_ranks", "passive_long_ew"):
        assert vectors[name] == panel_vectors[name], name
    assert vectors["no_trade"].weights == ()
    share = Decimal(1) / Decimal(PANEL_SIZE)
    assert vectors["passive_long_ew"].weights == tuple(
        (contract_id(index), share) for index in range(PANEL_SIZE)
    )
    assert len(vectors["random_ranks"].weights) == 4


def test_every_member_and_control_is_present() -> None:
    histories = panel_histories()
    snapshot = snapshot_for(histories)
    vectors = build_funding_xs_weight_vectors(
        histories, snapshot, two_settlement_funding(), spec=REDUCED_SPEC
    )
    expected = tuple(member.name for member in SPEC.members) + tuple(
        control.name for control in SPEC.controls
    )
    assert set(vectors) == set(expected)
    assert all(vector.decision_close_ns == DECISION for vector in vectors.values())
    assert all(vectors[name].member == name for name in vectors)


def test_an_empty_universe_propagates_its_reason_codes() -> None:
    snapshot = UniverseSnapshot(
        decision_close_ns=DECISION, contracts=(), reason_codes=("UNIVERSE_TOO_SMALL",)
    )
    vectors = build_funding_xs_weight_vectors({}, snapshot, {}, spec=REDUCED_SPEC)
    assert len(vectors) == len(SPEC.members) + len(SPEC.controls)
    for vector in vectors.values():
        assert vector.weights == ()
        assert vector.reason_codes == ("UNIVERSE_TOO_SMALL",)


def test_a_settlement_after_the_decision_changes_nothing() -> None:
    """Point-in-time regression: the score reads `(t - L, t]`, so a settlement
    published after the decision must not reach the book."""
    histories = panel_histories()
    snapshot = snapshot_for(histories)
    funding = two_settlement_funding()
    with_future = {
        contract: (*events, event(contract, DECISION + 1, Decimal(500) * BP))
        for contract, events in funding.items()
    }
    assert build_funding_xs_weight_vectors(
        histories, snapshot, with_future, spec=REDUCED_SPEC
    ) == build_funding_xs_weight_vectors(histories, snapshot, funding, spec=REDUCED_SPEC)


# --------------------------------------------------------------------------
# The sign-flip exit rule
# --------------------------------------------------------------------------


def test_sign_flip_exits_removes_the_legs_whose_funding_turned() -> None:
    """Spec 3's exit rule, per leg: a long is right while funding is negative
    (it is paid), a short while funding is positive. So a long whose trailing
    one-week funding is non-negative and a short whose trailing one-week
    funding is non-positive have both lost their reason to be held; an exact
    zero counts as flipped on both legs, and a contract that settled nothing
    in the week (`None`, or no entry at all) counts as flipped too -- the rule
    is fail-closed, it does not keep a position it cannot justify.
    """
    vector = (
        ("LONGOKUSDT:0", Decimal("0.25")),
        ("LONGFLIPUSDT:0", Decimal("0.25")),
        ("LONGZEROUSDT:0", Decimal("0.25")),
        ("LONGNONEUSDT:0", Decimal("0.25")),
        ("SHORTOKUSDT:0", Decimal("-0.25")),
        ("SHORTFLIPUSDT:0", Decimal("-0.25")),
        ("SHORTZEROUSDT:0", Decimal("-0.25")),
        ("MISSINGUSDT:0", Decimal("-0.25")),
    )
    trailing: dict[str, Decimal | None] = {
        "LONGOKUSDT:0": Decimal(-3) * BP,
        "LONGFLIPUSDT:0": Decimal(3) * BP,
        "LONGZEROUSDT:0": Decimal(0),
        "LONGNONEUSDT:0": None,
        "SHORTOKUSDT:0": Decimal(3) * BP,
        "SHORTFLIPUSDT:0": Decimal(-3) * BP,
        "SHORTZEROUSDT:0": Decimal(0),
    }
    assert sign_flip_exits(vector, trailing) == {
        "LONGFLIPUSDT:0",
        "LONGZEROUSDT:0",
        "LONGNONEUSDT:0",
        "SHORTFLIPUSDT:0",
        "SHORTZEROUSDT:0",
        "MISSINGUSDT:0",
    }


def test_sign_flip_exits_ignores_a_contract_that_is_not_held() -> None:
    """A zeroed weight is not a leg: an earlier exit already undeployed it, and
    re-reporting it would double-count `exit_rule_removals`."""
    vector = (("ZEROEDUSDT:0", Decimal(0)), ("HELDUSDT:0", Decimal("0.25")))
    trailing: dict[str, Decimal | None] = {"ZEROEDUSDT:0": None, "HELDUSDT:0": Decimal(-1) * BP}
    assert sign_flip_exits(vector, trailing) == set()


def test_sign_flip_exits_on_an_empty_vector_is_empty() -> None:
    assert sign_flip_exits((), {}) == set()
