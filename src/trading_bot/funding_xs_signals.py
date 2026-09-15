"""Funding scores, the funding cross-section's weight vectors, and its exit rule.

Spec 3 of `docs/superpowers/specs/2026-09-15-funding-xs-family-design.md`
scores a contract by the sum of its funding settlements over a trailing
window -- P1.28's definition, `carry_signals.trailing_funding`, reused rather
than recomputed -- and builds P1.27's cross-sectional quintile book on that
score with the ranking reversed: long the lowest-funding quintile, short the
highest. `panel_signals._cross_sectional_weights` is imported rather than
copied so the two families provably share one construction, and the three
controls are built exactly as `panel_signals.build_weight_vectors` builds
them (`tests/test_funding_xs_signals.py` pins them against its own output).
Importing a sibling module's private helper is the arrangement
`trend_signals` and `carry_accounting` already have; `panel_signals.py`
itself stays untouched, so P1.27 remains reproducible.

The book reads no prices at all: `histories` is here only because the fold
loop hands every weight builder the same three inputs, and the controls'
eligible set comes from the snapshot.
"""

from decimal import Decimal
from random import Random

from trading_bot.carry_signals import trailing_funding
from trading_bot.funding_xs_config import FundingXsFamilySpec
from trading_bot.panel_reader import FundingEvent
from trading_bot.panel_signals import WeightVector, _cross_sectional_weights
from trading_bot.panel_universe import ContractHistory, UniverseSnapshot


def funding_scores(
    funding_by_contract: dict[str, tuple[FundingEvent, ...]],
    eligible: tuple[str, ...],
    decision_close_ns: int,
    lookback_weeks: int,
) -> dict[str, Decimal]:
    """Each eligible contract's trailing funding sum over `(t - L weeks, t]`.

    A contract with no settlement in the window -- none recorded, none inside
    it, or none at all -- is absent from the result rather than carrying a
    zero: spec 3 says such a contract has no score, and a zero would rank it
    in the middle of the cross-section instead of out of it.
    """
    scores: dict[str, Decimal] = {}
    for contract_id in eligible:
        score = trailing_funding(
            funding_by_contract.get(contract_id, ()),
            decision_close_ns=decision_close_ns,
            lookback_weeks=lookback_weeks,
        )
        if score is not None:
            scores[contract_id] = score
    return scores


def build_funding_xs_weight_vectors(
    histories: dict[str, ContractHistory],
    snapshot: UniverseSnapshot,
    funding_by_contract: dict[str, tuple[FundingEvent, ...]],
    *,
    spec: FundingXsFamilySpec,
) -> dict[str, WeightVector]:
    """Build every member and control weight vector for one rebalance.

    A member's vector depends only on its lookback: the hold and the exit rule
    are applied by the fold runner, which ages these vectors into cohorts, so
    the three four-week-lookback members share one book here.
    """
    del histories  # the funding book reads no prices; see the module docstring
    decision = snapshot.decision_close_ns
    names = [member.name for member in spec.members] + [item.name for item in spec.controls]
    if not snapshot.contracts:
        return {name: WeightVector(decision, name, (), snapshot.reason_codes) for name in names}

    eligible = tuple(item.contract_id for item in snapshot.contracts)
    vectors: dict[str, WeightVector] = {}
    by_lookback: dict[int, tuple[tuple[str, Decimal], ...]] = {}
    for member in spec.members:
        weights = by_lookback.get(member.lookback_weeks)
        if weights is None:
            scores = funding_scores(
                funding_by_contract, eligible, decision, member.lookback_weeks
            )
            weights = _cross_sectional_weights(scores, spec.weights, reverse=True)
            by_lookback[member.lookback_weeks] = weights
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
    draws = {contract_id: Decimal(str(rng.random())) for contract_id in eligible}
    vectors["random_ranks"] = WeightVector(
        decision,
        "random_ranks",
        _cross_sectional_weights(draws, spec.weights, reverse=False),
        (),
    )
    return vectors


def sign_flip_exits(
    vector: tuple[tuple[str, Decimal], ...], trailing_one_week: dict[str, Decimal | None]
) -> set[str]:
    """The held contracts whose trailing one-week funding is wrong for their leg.

    Spec 3's exit rule, per leg: a long is in the lowest-funding quintile
    because it expects to be *paid* funding, so a non-negative trailing week
    flips it; a short expects to *collect*, so a non-positive week flips it.
    An exact zero flips both legs, and a contract that settled nothing in the
    week -- `None`, or no entry at all -- counts as flipped: the rule is
    fail-closed and does not keep a position it cannot justify.

    A zero weight is not a leg (an earlier decision already undeployed it) and
    is skipped, so a contract is never reported as removed twice.
    """
    return {
        contract_id
        for contract_id, weight in vector
        if weight != 0 and _flipped(weight, trailing_one_week.get(contract_id))
    }


def _flipped(weight: Decimal, funding: Decimal | None) -> bool:
    """Whether a leg's trailing week no longer justifies holding it."""
    if funding is None:
        return True
    return funding >= 0 if weight > 0 else funding <= 0
