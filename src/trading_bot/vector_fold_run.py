"""The weekly weight-vector fold loop P1.31 and P1.32 both run on.

This is `trend_fold_run.py`'s loop, lifted out of it unchanged and
parameterised by what actually differs between two families that trade a
weekly cross-sectional weight vector on the P1.27 panel: the declaration
model, the builder that turns one Sunday's universe into one weight vector
per candidate, an optional per-decision exit rule, the per-episode extras,
and the module list the code hash speaks for. Everything else -- the capture
and manifest linkage, the three-Sunday warm-up, the cohort book, the
in-window skip reset, `MEMBER_HELD_NOTHING`, P1.27's accounting, the registry
rows and the report schema -- is one implementation, so a family cannot drift
from its sibling by accident.

Three seams carry the family in:

* `build_vectors(histories, snapshot, funding_by_contract)` is asked for every
  candidate's vector at one Sunday, warm-up Sundays included;
* `exit_rule` and `extras` are *bound to the fold* first -- the loop hands them
  the histories and funding events it has just read and keeps the callables
  they return -- because a family's exit rule and its extras read the same
  evidence the book was built from, and neither the caller nor the declaration
  can hold it: the bars are loaded here, under this fold's own
  `available_before_ns`;
* `result_extras` adds the extras that exist only once an episode has been
  priced (P1.32's `funding_collected` is the evaluated episode's own
  `-funding_cost`), which are per scenario where the decision-stage extras are
  not. The two are merged per episode.

The declaration is read twice per fold: once by the family wrapper, which needs
it to bind its builder and to name its exit-rule members, and once here through
`load_spec`, whose hash is the one published and checked against the manifest.
Both read the same path in the same call, and only the hash taken here ever
reaches an artifact.
"""

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.panel_accounting import EpisodeResult, evaluate_episode
from trading_bot.panel_capture import verify_panel_capture
from trading_bot.panel_config import (
    PanelCosts,
    PanelCostTable,
    PanelStatistics,
    PanelUniverseRules,
)
from trading_bot.panel_fold_run import MEMBER_HELD_NOTHING_REASON_CODE
from trading_bot.panel_reader import FundingEvent, load_funding_events, load_panel_bars
from trading_bot.panel_samples import verify_panel_manifest
from trading_bot.panel_signals import WeightVector
from trading_bot.panel_universe import (
    ContractHistory,
    UniverseSnapshot,
    build_contract_histories,
    select_universe,
)
from trading_bot.registry import ExperimentRecord, MetadataRegistry

# Same string, and the same meaning, as `carry_fold_run.FOLD_WARMED_REASON_CODE`:
# this fold's opening book was formed over Sundays preceding its own test
# window. Declared here rather than imported so neither family running this loop
# depends on the carry module -- nothing else in them does, and their module
# lists would then no longer cover every module the code hash speaks for.
FOLD_WARMED_REASON_CODE = "FOLD_OPENING_BOOK_WARMED_FROM_PRIOR_WEEKS"

_WEEK_NS = 7 * 86_400_000_000_000

type Histories = dict[str, ContractHistory]
type FundingByContract = dict[str, tuple[FundingEvent, ...]]
type Vector = tuple[tuple[str, Decimal], ...]

type WeightBuilder = Callable[
    [Histories, UniverseSnapshot, FundingByContract], dict[str, WeightVector]
]
type ExitRule = Callable[[str, Vector, int], set[str]]
type ExtrasBuilder = Callable[[str, Vector, UniverseSnapshot, int], dict[str, Decimal]]
type ResultExtras = Callable[[EpisodeResult], dict[str, Decimal]]
# A family callable that needs this fold's own evidence: the loop reads the bars
# and the funding events, binds them once, and reuses what comes back at every
# decision of the fold.
type BoundToFold[T] = Callable[[Histories, FundingByContract], T]


class VectorFoldError(RuntimeError):
    """Raised when a weight-vector fold cannot be evaluated or published."""


@dataclass(frozen=True, slots=True)
class VectorFoldArtifact:
    output_path: Path
    report_hash: str
    fold_index: int
    episode_count: int
    skipped_sample_count: int


class VectorMember(Protocol):
    """What the loop needs of a declared member: a name and a holding period."""

    @property
    def name(self) -> str: ...

    @property
    def hold_weeks(self) -> int: ...


class VectorControl(Protocol):
    """What the loop needs of a declared control: a name."""

    @property
    def name(self) -> str: ...


class VectorFamily(Protocol):
    """The declaration surface the loop reads, whichever family declared it.

    Read-only properties rather than attributes, so a frozen pydantic model
    whose fields happen to match satisfies it structurally, without either
    family's model importing this one.
    """

    @property
    def family_name(self) -> str: ...

    @property
    def hypothesis(self) -> str: ...

    @property
    def holding_days(self) -> int: ...

    @property
    def members(self) -> tuple[VectorMember, ...]: ...

    @property
    def controls(self) -> tuple[VectorControl, ...]: ...

    @property
    def universe(self) -> PanelUniverseRules: ...

    @property
    def costs(self) -> PanelCosts: ...

    @property
    def statistics(self) -> PanelStatistics: ...


def warm_up_weeks_of(spec: VectorFamily) -> int:
    """How many Sundays before a fold's first decision are warmed.

    A fold that opened flat would ramp a four-week member's book 1/4 per week
    for three weeks, while the one-week members are fully invested from their
    first episode -- a hold-dependent haircut on the primary metric, not a
    property of the signal. So each fold forms (only) the weight vectors of the
    `max(hold_weeks) - 1` Sundays before its first test decision: that many
    vectors plus the first decision's own fill even the longest-held member's
    book exactly. Three under both families' v1, whose longest hold is four
    weeks.
    """
    return max(member.hold_weeks for member in spec.members) - 1


def assemble_cohort_book(
    vectors: Sequence[tuple[int, Vector]],
    *,
    hold_weeks: int,
    histories: Histories,
    decision_close_ns: int,
) -> Vector:
    """The book a multi-week member holds at one decision.

    The sum of `vector / hold_weeks` over every retained vector still inside the
    hold, each vector paired with the Sunday it was formed on. A vector ages out
    on the calendar -- `decision_close_ns - formed_ns >= hold_weeks * WEEK_NS`,
    `carry_signals.assemble_book`'s rule -- not on its position in the list: a
    Sunday on which the whole panel had no universe forms no vector at all, and
    counting positions would then keep the oldest vector in the book for a fifth
    week.

    Capital is committed per week, not per contract: a week whose vector is empty
    contributes nothing and its share simply stays undeployed, and a contract
    with no bar at this decision is dropped rather than having its share spread
    over the contracts that can still be traded. Dropping is also what makes the
    book tradeable at all -- `panel_accounting.evaluate_episode` refuses a weight
    on a contract with no entry close. A contract whose weeks cancel to exactly
    zero leaves the book too: a zero weight is not a position, and keeping it
    would inflate `signalled_contracts` without changing any cash flow. That is
    also how an exit rule's zeroed legs leave the book while their share of the
    cohort's capital stays undeployed.
    """
    if hold_weeks < 1:
        raise VectorFoldError("a holding period is at least one week")
    share = Decimal(hold_weeks)
    horizon = hold_weeks * _WEEK_NS
    totals: dict[str, Decimal] = {}
    for formed_ns, vector in vectors:
        if decision_close_ns - formed_ns >= horizon:
            continue
        for contract_id, weight in vector:
            if decision_close_ns not in histories[contract_id].closes:
                continue
            totals[contract_id] = totals.get(contract_id, Decimal(0)) + weight / share
    return tuple(sorted((key, value) for key, value in totals.items() if value != 0))


def run_vector_fold(
    capture_root: Path,
    *,
    manifest_path: Path,
    family_spec_path: Path,
    output_path: Path,
    registry_path: Path,
    fold_index: int,
    load_spec: Callable[[Path], tuple[VectorFamily, str]],
    build_vectors: WeightBuilder,
    exit_rule: BoundToFold[ExitRule] | None,
    exit_rule_members: frozenset[str],
    extras: BoundToFold[ExtrasBuilder],
    modules: tuple[str, ...],
    error: type[RuntimeError],
    result_extras: ResultExtras | None = None,
) -> VectorFoldArtifact:
    """Evaluate one walk-forward fold of a weekly weight-vector family."""
    valid, errors = verify_panel_capture(capture_root)
    if not valid:
        raise error("panel capture verification failed: " + ",".join(errors))
    if not verify_panel_manifest(manifest_path):
        raise error("panel manifest verification failed")
    if output_path.exists():
        raise error("fold report already exists and is immutable")

    spec, family_spec_hash = load_spec(family_spec_path)
    manifest = _load_object(manifest_path, error)
    if manifest.get("family_spec_hash") != family_spec_hash:
        raise error("family declaration does not match the manifest")
    capture_manifest = _load_object(capture_root / "capture-manifest.json", error)
    dataset_manifest = _load_object(capture_root / "dataset" / "dataset-manifest.json", error)
    if manifest.get("capture_root_hash") != capture_manifest.get("capture_root_hash"):
        raise error("manifest is not linked to this capture")
    if manifest.get("dataset_root_hash") != dataset_manifest.get("root_hash"):
        raise error("manifest is not linked to this dataset")

    folds = manifest.get("folds")
    if not isinstance(folds, list):
        raise error("manifest folds are malformed")
    fold = next((item for item in folds if item.get("fold_index") == fold_index), None)
    if fold is None:
        raise error(f"fold {fold_index} is not in the manifest")

    test_end_ns = int(fold["test_end_ns"])
    bars = load_panel_bars(capture_root / "dataset", available_before_ns=test_end_ns + 1)
    histories = build_contract_histories(bars)
    funding_by_contract: FundingByContract = {}
    for event in load_funding_events(capture_root / "dataset"):
        funding_by_contract.setdefault(event.contract_id, ())
        funding_by_contract[event.contract_id] += (event,)
    build_extras = extras(histories, funding_by_contract)
    apply_exit = None if exit_rule is None else exit_rule(histories, funding_by_contract)

    test_ids = [str(value) for value in fold["test_ids"]]
    decisions = sorted(int(value.split(":")[1]) for value in test_ids)
    members: dict[str, VectorMember] = {member.name: member for member in spec.members}
    names = [member.name for member in spec.members] + [item.name for item in spec.controls]
    scenarios: tuple[PanelCostTable, ...] = (spec.costs.base, spec.costs.adverse)
    episodes: dict[tuple[str, str], list[EpisodeResult]] = {
        (name, scenario.name): [] for name in names for scenario in scenarios
    }
    held_nothing: dict[str, list[bool]] = {name: [] for name in names}
    decision_extras: dict[str, list[dict[str, Decimal]]] = {name: [] for name in names}
    carried: dict[tuple[str, str], Vector] = {key: () for key in episodes}
    # Only a member that holds for more than one week needs its past vectors
    # kept; every other candidate's book is this week's vector.
    cohorts: dict[str, list[tuple[int, Vector]]] = {
        member.name: [] for member in spec.members if member.hold_weeks > 1
    }
    skipped: list[str] = []
    warm_up_weeks = warm_up_weeks_of(spec)

    # Warm-up: form (only) the weight vectors of the Sundays before the fold's
    # first test decision, so a multi-week member's first test episode opens on
    # a full book instead of a 1/4 stub. No episode is evaluated and nothing is
    # reported for these weeks; `carried` stays empty, so the first test episode
    # buys the whole warmed book and pays its full entry turnover inside the
    # window. Bars were loaded with `available_before_ns = test_end_ns + 1`,
    # which covers these Sundays, and `select_universe` and every weight builder
    # read only data at or before the Sunday they are asked about, so no future
    # data enters here. A warm-up Sunday whose universe is too small simply
    # contributes no vector: no position exists yet to be flattened, so there is
    # nothing to reset, and the vectors formed on the warm-up Sundays around it
    # keep ageing out on the calendar as usual. Only an in-window skipped week
    # resets, where a real position is closed (P1.27); `carry_fold_run.py`'s
    # warm-up does the same. No exit rule runs out here either: it removes a leg
    # from a position, and no position exists yet.
    warm_up_closes = (
        [decisions[0] - weeks * _WEEK_NS for weeks in range(warm_up_weeks, 0, -1)]
        if decisions
        else []
    )
    for warm_up_close_ns in warm_up_closes:
        warm_up = select_universe(
            histories, decision_close_ns=warm_up_close_ns, rules=spec.universe
        )
        if not warm_up.contracts:
            continue
        vectors = build_vectors(histories, warm_up, funding_by_contract)
        for name in cohorts:
            _retain(
                cohorts[name],
                vectors[name].weights,
                formed_ns=warm_up_close_ns,
                hold_weeks=members[name].hold_weeks,
            )

    for decision_close_ns in decisions:
        sample_id = f"BINANCE_UM:{decision_close_ns}:w1"
        snapshot = select_universe(
            histories, decision_close_ns=decision_close_ns, rules=spec.universe
        )
        if not snapshot.contracts:
            skipped.append(sample_id)
            for key in carried:
                carried[key] = ()
            for name in cohorts:
                cohorts[name] = []
            continue
        tiers = {item.contract_id: item.tier for item in snapshot.contracts}
        vectors = build_vectors(histories, snapshot, funding_by_contract)
        for name in names:
            member = members.get(name)
            rule = apply_exit if name in exit_rule_members else None
            if member is not None and member.hold_weeks > 1:
                retained = cohorts[name]
                _retain(
                    retained,
                    vectors[name].weights,
                    formed_ns=decision_close_ns,
                    hold_weeks=member.hold_weeks,
                )
                # The exit rule runs over the retained vectors -- this Sunday's
                # freshly formed one included, so a contract the rule rejects is
                # not entered either -- and before the book is assembled, so a
                # zeroed leg is out of this episode's position rather than held
                # for another week. The zero stays in the retained vectors: its
                # share of the cohort's capital is undeployed until the cohort
                # ages out, exactly as after a dropped contract.
                removed = _removals(
                    rule, name, [item[1] for item in retained], decision_close_ns
                )
                if removed:
                    cohorts[name] = retained = _zeroed(retained, removed)
                weights = assemble_cohort_book(
                    retained,
                    hold_weeks=member.hold_weeks,
                    histories=histories,
                    decision_close_ns=decision_close_ns,
                )
            else:
                vector = vectors[name].weights
                removed = _removals(rule, name, [vector], decision_close_ns)
                weights = (
                    tuple((key, value) for key, value in vector if key not in removed)
                    if removed
                    else vector
                )
            # As in P1.27: only a member's own construction can legitimately
            # produce no weights while the universe itself was not too small
            # (too few rankable contracts to form both quintiles, every contract
            # inside a dead band, or -- for a four-week member -- four empty
            # weeks behind it). The controls always have well-defined weights
            # here and `no_trade`'s emptiness is a deliberate baseline, so a
            # control is never marked.
            held_nothing[name].append(member is not None and not weights)
            entry = build_extras(name, weights, snapshot, decision_close_ns)
            if apply_exit is not None:
                # The count is the loop's own bookkeeping, not the family's, so
                # it is written here; a family that declares no exit rule has no
                # such key at all rather than a column of zeros.
                entry["exit_rule_removals"] = Decimal(len(removed))
            decision_extras[name].append(entry)
            for scenario in scenarios:
                key = (name, scenario.name)
                result = evaluate_episode(
                    sample_id=sample_id,
                    member=name,
                    decision_close_ns=decision_close_ns,
                    holding_days=spec.holding_days,
                    weights=weights,
                    previous_weights=carried[key],
                    histories=histories,
                    tiers=tiers,
                    funding_by_contract=funding_by_contract,
                    cost_table=scenario,
                )
                episodes[key].append(result)
                carried[key] = result.drifted_weights

    episode_count = len(episodes[(names[0], "base")])
    reason_codes: list[str] = []
    if decisions:
        reason_codes.append(FOLD_WARMED_REASON_CODE)
    if skipped:
        reason_codes.append("SKIPPED_WEEK_EXIT_COST_UNCHARGED")
    if episode_count == 0:
        reason_codes.append("NO_EPISODES_IN_FOLD")
    else:
        # P1.27's asymmetry, unchanged and still visible: the carried position
        # resets to empty at fold start, so the first episode pays a full entry
        # turnover, but the last episode's position is never closed out, so it
        # pays no exit. One side of unit gross per fold, uncharged, always in
        # the favourable direction.
        reason_codes.append("FOLD_FINAL_EXIT_COST_UNCHARGED")

    candidates = [
        {
            "candidate_name": name,
            "role": "member" if name in members else "control",
            "episode_count": len(episodes[(name, "base")]),
            "base": _scenario_record(
                episodes[(name, "base")],
                held_nothing[name],
                decision_extras[name],
                result_extras,
            ),
            "adverse": _scenario_record(
                episodes[(name, "adverse")],
                held_nothing[name],
                decision_extras[name],
                result_extras,
            ),
        }
        for name in names
    ]

    material: dict[str, object] = {
        "report_version": "1.0.0",
        "status": "development_only",
        "reason_codes": reason_codes,
        "family_name": spec.family_name,
        "family_spec_hash": family_spec_hash,
        "capture_root_hash": str(manifest["capture_root_hash"]),
        "dataset_root_hash": str(manifest["dataset_root_hash"]),
        "split_manifest_hash": str(manifest["split_manifest_hash"]),
        "manifest_hash": str(manifest["manifest_hash"]),
        "fold_index": fold_index,
        "fold_count": len(folds),
        "train_sample_count": len(fold["train_ids"]),
        "validation_sample_count": len(fold["validation_ids"]),
        "test_sample_count": len(test_ids),
        "train_membership_hash": content_sha256(list(fold["train_ids"])),
        "validation_membership_hash": content_sha256(list(fold["validation_ids"])),
        "test_membership_hash": content_sha256(test_ids),
        "random_seed": spec.statistics.random_seed,
        "block_length": spec.statistics.block_length,
        "bootstrap_repetitions": spec.statistics.bootstrap_repetitions,
        "skipped_sample_ids": skipped,
        "warm_up_weeks": warm_up_weeks,
        "code_hash": code_hash_of(modules),
        "candidates": candidates,
    }
    report_hash = content_sha256(material)
    document = dict(material)
    document["report_hash"] = report_hash
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_bytes(canonical_json(document))
    temporary.replace(output_path)

    _register(spec, manifest, report_hash, registry_path, modules)
    return VectorFoldArtifact(
        output_path=output_path,
        report_hash=report_hash,
        fold_index=fold_index,
        episode_count=episode_count,
        skipped_sample_count=len(skipped),
    )


def _removals(
    rule: ExitRule | None, name: str, vectors: Sequence[Vector], decision_close_ns: int
) -> set[str]:
    """The union of the rule's verdicts over the vectors a candidate holds.

    A contract flipped in any one of them leaves all of them: the rule reads a
    leg's sign, and the same contract can be a long in one cohort and a short in
    another, so asking per vector and taking the union is what makes "removed
    from every retained vector" mean what it says.
    """
    if rule is None:
        return set()
    removed: set[str] = set()
    for vector in vectors:
        removed |= rule(name, vector, decision_close_ns)
    return removed


def _zeroed(vectors: Sequence[tuple[int, Vector]], removed: set[str]) -> list[tuple[int, Vector]]:
    """Every retained vector with the removed contracts' weights set to zero.

    New tuples throughout: sibling members can share the very tuple a builder
    cached for their common lookback, and one member's exit rule must not reach
    into another member's book.
    """
    return [
        (
            formed_ns,
            tuple(
                (contract_id, Decimal(0) if contract_id in removed else weight)
                for contract_id, weight in vector
            ),
        )
        for formed_ns, vector in vectors
    ]


def _retain(
    vectors: list[tuple[int, Vector]],
    vector: Vector,
    *,
    formed_ns: int,
    hold_weeks: int,
) -> None:
    """Append this Sunday's vector and forget anything the hold has aged out.

    Age is the calendar distance from the Sunday just formed, the same rule
    `assemble_cohort_book` applies; this only keeps the retained list bounded.
    """
    vectors.append((formed_ns, vector))
    horizon = hold_weeks * _WEEK_NS
    vectors[:] = [item for item in vectors if formed_ns - item[0] < horizon]


def _scenario_record(
    results: list[EpisodeResult],
    held_nothing: list[bool],
    extras: list[dict[str, Decimal]],
    result_extras: ResultExtras | None,
) -> dict[str, object]:
    return {
        "total_net_return": sum((item.net_return for item in results), Decimal(0)),
        "episodes": [
            {
                "sample_id": item.sample_id,
                "net_return": item.net_return,
                "gross_return": item.gross_return,
                "turnover": item.turnover,
                "trading_cost": item.trading_cost,
                "funding_cost": item.funding_cost,
                "forced_close_cost": item.forced_close_cost,
                "gross_exposure": item.gross_exposure,
                "net_exposure": item.net_exposure,
                "forced_close_count": item.forced_close_count,
                "reason_codes": [MEMBER_HELD_NOTHING_REASON_CODE] if flag else [],
                "contract_net_contributions": [
                    [contract_id, value] for contract_id, value in item.contract_net_contributions
                ],
                "extras": (
                    dict(extra) if result_extras is None else {**extra, **result_extras(item)}
                ),
            }
            for item, flag, extra in zip(results, held_nothing, extras, strict=True)
        ],
    }


def _register(
    spec: VectorFamily,
    manifest: dict[str, object],
    report_hash: str,
    registry_path: Path,
    modules: tuple[str, ...],
) -> None:
    split_hash = str(manifest["split_manifest_hash"])
    family_id = uuid5(NAMESPACE_URL, f"{split_hash}:{spec.family_name}")
    created_at_ns = time.time_ns()
    with MetadataRegistry(registry_path) as registry:
        for member in spec.members:
            registry.register_experiment(
                ExperimentRecord(
                    experiment_id=uuid5(
                        family_id,
                        f"{member.name}:{spec.statistics.random_seed}:{report_hash}",
                    ),
                    family_id=family_id,
                    candidate_name=member.name,
                    hypothesis=spec.hypothesis,
                    split_manifest_hash=split_hash,
                    code_hash=code_hash_of(modules),
                    random_seed=spec.statistics.random_seed,
                    outcome="completed",
                    result_hash=report_hash,
                    failure_reason=None,
                    created_at_ns=created_at_ns,
                )
            )


def code_hash_of(modules: tuple[str, ...]) -> str:
    """The hash of the source a family's fold reports speak for."""
    root = Path(__file__).parent
    material = "".join((root / name).read_text(encoding="utf-8") for name in modules)
    return content_sha256(material)


def _load_object(path: Path, error: type[RuntimeError]) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise error(f"{path.name} must contain a JSON object")
    return document
