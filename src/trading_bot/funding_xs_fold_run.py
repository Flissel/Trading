"""Evaluate one walk-forward fold of the perpetual-only funding cross-section.

The week is `vector_fold_run.run_vector_fold`'s -- P1.31's loop, which this
family shares rather than copies -- so everything about how a fold is verified,
warmed, held, skipped, priced, reported and registered is stated once, over
there. What this module declares is only what makes the family itself:

* the book is `funding_xs_signals.build_funding_xs_weight_vectors`, long the
  lowest-funding quintile and short the highest (spec 3);
* the exit rule (spec 3, member `fx_q5_l4w_h4w_exit` alone) zeroes a held leg
  whose trailing *one* week of funding no longer has the sign its leg was
  entered for -- `sign_flip_exits` decides, the loop applies it to every
  retained cohort vector before the book is assembled, and never during the
  warm-up or for a control;
* spec 4.5's four extras: `signalled_contracts` and `mean_score` at the
  decision, `exit_rule_removals` from the loop's own bookkeeping, and
  `funding_collected` -- the evaluated episode's own `-funding_cost`, which is
  why it is read from the result and merged in per scenario rather than
  computed once per decision. Under the adverse table a receipt is worth 0.75
  of itself and a payment costs double, so the two scenarios' collections
  genuinely differ.
"""

from decimal import Decimal
from pathlib import Path

from trading_bot.carry_signals import trailing_funding
from trading_bot.funding_xs_config import FundingXsFamilySpec, load_funding_xs_family_spec
from trading_bot.funding_xs_signals import (
    build_funding_xs_weight_vectors,
    funding_scores,
    sign_flip_exits,
)
from trading_bot.panel_accounting import EpisodeResult
from trading_bot.panel_fold_run import _PANEL_MODULES
from trading_bot.panel_signals import WeightVector
from trading_bot.panel_universe import UniverseSnapshot
from trading_bot.vector_fold_run import (
    BoundToFold,
    ExitRule,
    ExtrasBuilder,
    FundingByContract,
    Histories,
    Vector,
    VectorFoldArtifact,
    VectorFoldError,
    WeightBuilder,
    run_vector_fold,
)

# The panel modules this family runs on, the generic loop, and its own three.
# Built from `panel_fold_run`'s own tuple rather than a copy of it, so a change
# to any module P1.27 evaluates through necessarily moves this code hash too.
_FUNDING_XS_MODULES = (
    *_PANEL_MODULES,
    "funding_xs_config.py",
    "funding_xs_signals.py",
    "vector_fold_run.py",
    "funding_xs_fold_run.py",
)


class FundingXsFoldError(VectorFoldError):
    """Raised when a funding cross-section fold cannot be evaluated or published."""


def run_funding_xs_fold(
    capture_root: Path,
    *,
    manifest_path: Path,
    family_spec_path: Path,
    output_path: Path,
    registry_path: Path,
    fold_index: int,
) -> VectorFoldArtifact:
    spec, _ = load_funding_xs_family_spec(family_spec_path)
    return run_vector_fold(
        capture_root,
        manifest_path=manifest_path,
        family_spec_path=family_spec_path,
        output_path=output_path,
        registry_path=registry_path,
        fold_index=fold_index,
        load_spec=load_funding_xs_family_spec,
        build_vectors=_weight_builder(spec),
        exit_rule=_exit_rule(),
        exit_rule_members=frozenset(
            member.name for member in spec.members if member.exit_on_sign_flip
        ),
        extras=_extras(spec),
        modules=_FUNDING_XS_MODULES,
        error=FundingXsFoldError,
        result_extras=_funding_collected,
    )


def _weight_builder(spec: FundingXsFamilySpec) -> WeightBuilder:
    def build(
        histories: Histories, snapshot: UniverseSnapshot, funding_by_contract: FundingByContract
    ) -> dict[str, WeightVector]:
        return build_funding_xs_weight_vectors(
            histories, snapshot, funding_by_contract, spec=spec
        )

    return build


def _exit_rule() -> BoundToFold[ExitRule]:
    """Spec 3's sign-flip exit, bound to one fold's funding events.

    The trailing week is a property of the contract and the decision, not of the
    cohort asking, so it is computed once per pair and reused across the up to
    four retained vectors a member holds.
    """

    def bind(histories: Histories, funding_by_contract: FundingByContract) -> ExitRule:
        del histories  # the rule reads settlements, not prices
        cache: dict[tuple[str, int], Decimal | None] = {}

        def rule(name: str, vector: Vector, decision_close_ns: int) -> set[str]:
            del name
            trailing: dict[str, Decimal | None] = {}
            for contract_id, _ in vector:
                key = (contract_id, decision_close_ns)
                if key not in cache:
                    cache[key] = trailing_funding(
                        funding_by_contract.get(contract_id, ()),
                        decision_close_ns=decision_close_ns,
                        lookback_weeks=1,
                    )
                trailing[contract_id] = cache[key]
            return sign_flip_exits(vector, trailing)

        return rule

    return bind


def _extras(spec: FundingXsFamilySpec) -> BoundToFold[ExtrasBuilder]:
    """Spec 4.5's decision-stage extras, bound to one fold's funding events.

    `mean_score` is the mean `F_L` over the contracts that could be ranked at
    this decision, so it is a member's own number: the four members do not share
    a lookback, and a contract with no settlement in the window is not averaged
    in because it was never scored. A control ranks nothing on funding and so
    has no mean score at all; zero, as `signalled_contracts` is the honest count
    of legs it does hold.
    """
    lookbacks = {member.name: member.lookback_weeks for member in spec.members}

    def bind(histories: Histories, funding_by_contract: FundingByContract) -> ExtrasBuilder:
        del histories  # the funding book reads settlements, not prices
        cache: dict[tuple[int, int], Decimal] = {}

        def build(
            name: str, weights: Vector, snapshot: UniverseSnapshot, decision_close_ns: int
        ) -> dict[str, Decimal]:
            signalled = Decimal(sum(1 for _, weight in weights if weight != 0))
            lookback = lookbacks.get(name)
            if lookback is None:
                return {"signalled_contracts": signalled, "mean_score": Decimal(0)}
            key = (lookback, decision_close_ns)
            mean_score = cache.get(key)
            if mean_score is None:
                eligible = tuple(item.contract_id for item in snapshot.contracts)
                scores = funding_scores(
                    funding_by_contract, eligible, decision_close_ns, lookback
                )
                mean_score = (
                    sum(scores.values(), Decimal(0)) / Decimal(len(scores))
                    if scores
                    else Decimal(0)
                )
                cache[key] = mean_score
            return {"signalled_contracts": signalled, "mean_score": mean_score}

        return build

    return bind


def _funding_collected(result: EpisodeResult) -> dict[str, Decimal]:
    """Spec 4.5's `funding_collected`: the episode's funding cost, signed the
    way a book that earns funding reads it -- positive when the two legs
    collected more than they paid over the held week, under the cost table the
    episode was priced with."""
    return {"funding_collected": -result.funding_cost}
