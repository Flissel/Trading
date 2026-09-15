"""Evaluate one walk-forward fold of the trend aggregate family.

The loop itself is `vector_fold_run.run_vector_fold` -- P1.31's own loop,
moved there unchanged when the funding cross-section (P1.32) needed the same
week: the same universe, the same accounting, the same manifest linkage, the
same skipped-week reset, the same `MEMBER_HELD_NOTHING`, the same registry
rows, the same three-Sunday warm-up, and the same four-week cohort book. What
is left here is what is trend-specific: the declaration model, the weight
builder, the per-episode extras, the family's error type, and the module list
its `code_hash` speaks for.

`tests/fixtures/trend_fold1_expected.json` pins the economics of fold 1 of the
trend fixture from before that move, and `tests/test_trend_fold_run.py` asserts
them digit for digit afterwards: the generalisation moved `code_hash` and
`report_hash` (this module and `vector_fold_run.py` are both hashed) and
nothing else.
"""

from decimal import Decimal
from pathlib import Path

from trading_bot.panel_fold_run import _PANEL_MODULES
from trading_bot.panel_signals import WeightVector
from trading_bot.panel_universe import UniverseSnapshot
from trading_bot.trend_config import TrendFamilySpec, TrendMember, load_trend_family_spec
from trading_bot.trend_signals import build_trend_weight_vectors, threshold_sign, trend_scores
from trading_bot.vector_fold_run import (
    FOLD_WARMED_REASON_CODE,
    BoundToFold,
    ExtrasBuilder,
    FundingByContract,
    Histories,
    Vector,
    VectorFoldArtifact,
    VectorFoldError,
    WeightBuilder,
    run_vector_fold,
)

__all__ = [
    "FOLD_WARMED_REASON_CODE",
    "TrendFoldError",
    "run_trend_fold",
]

# The panel modules this family runs on, plus the generic loop and its own
# three. Built from `panel_fold_run`'s own tuple rather than a copy of it, so a
# change to any module P1.27 evaluates through necessarily moves the trend code
# hash too.
_TREND_MODULES = (
    *_PANEL_MODULES,
    "trend_config.py",
    "trend_signals.py",
    "vector_fold_run.py",
    "trend_fold_run.py",
)


class TrendFoldError(VectorFoldError):
    """Raised when a trend fold cannot be evaluated or published."""


def run_trend_fold(
    capture_root: Path,
    *,
    manifest_path: Path,
    family_spec_path: Path,
    output_path: Path,
    registry_path: Path,
    fold_index: int,
) -> VectorFoldArtifact:
    spec, _ = load_trend_family_spec(family_spec_path)
    return run_vector_fold(
        capture_root,
        manifest_path=manifest_path,
        family_spec_path=family_spec_path,
        output_path=output_path,
        registry_path=registry_path,
        fold_index=fold_index,
        load_spec=load_trend_family_spec,
        build_vectors=_weight_builder(spec),
        # Spec 4.2 declares no exit rule for this family: a trend member holds
        # its cohort for the declared weeks and re-forms it, nothing else.
        exit_rule=None,
        exit_rule_members=frozenset(),
        extras=_extras(spec),
        modules=_TREND_MODULES,
        error=TrendFoldError,
    )


def _weight_builder(spec: TrendFamilySpec) -> WeightBuilder:
    def build(
        histories: Histories, snapshot: UniverseSnapshot, funding_by_contract: FundingByContract
    ) -> dict[str, WeightVector]:
        del funding_by_contract  # the trend vote reads prices, not settlements
        return build_trend_weight_vectors(histories, snapshot, spec=spec)

    return build


def _extras(spec: TrendFamilySpec) -> BoundToFold[ExtrasBuilder]:
    """Spec 4.5's three per-episode extras, bound to one fold's bars.

    `mean_score` and `abstained_contracts` are read off the same indicator votes
    the week's vectors were built from, so the scores are computed once per
    decision and answered for all seven candidates -- the vote does not depend
    on which candidate is asking.
    """
    members = {member.name: member for member in spec.members}

    def bind(histories: Histories, funding_by_contract: FundingByContract) -> ExtrasBuilder:
        del funding_by_contract
        cache: dict[int, dict[str, Decimal]] = {}

        def build(
            name: str, weights: Vector, snapshot: UniverseSnapshot, decision_close_ns: int
        ) -> dict[str, Decimal]:
            eligible = tuple(item.contract_id for item in snapshot.contracts)
            scores = cache.get(decision_close_ns)
            if scores is None:
                scores = trend_scores(histories, eligible, decision_close_ns)
                cache[decision_close_ns] = scores
            mean_score = (
                sum(scores.values(), Decimal(0)) / Decimal(len(scores)) if scores else Decimal(0)
            )
            return {
                "signalled_contracts": Decimal(sum(1 for _, weight in weights if weight != 0)),
                "abstained_contracts": _abstained_contracts(
                    members.get(name), eligible, scores
                ),
                "mean_score": mean_score,
            }

        return build

    return bind


def _abstained_contracts(
    member: TrendMember | None,
    eligible: tuple[str, ...],
    scores: dict[str, Decimal],
) -> Decimal:
    """Eligible contracts whose score falls inside a time-series dead band.

    Zero for the cross-sectional member and for every control, which have no
    threshold to abstain against. A contract with no score at all is not counted:
    it did not abstain, it could not be scored.
    """
    if member is None or member.kind != "time_series":
        return Decimal(0)
    if member.threshold is None:
        raise TrendFoldError(f"time-series member {member.name} has no threshold")
    threshold = member.threshold
    return Decimal(
        sum(
            1
            for contract_id in eligible
            if contract_id in scores and threshold_sign(scores[contract_id], threshold) == 0
        )
    )
