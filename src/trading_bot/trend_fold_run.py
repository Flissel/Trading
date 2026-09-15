"""Evaluate one walk-forward fold of the trend aggregate family.

The loop is `panel_fold_run.run_panel_fold`'s, week for week, and the report it
writes is that module's material with `warm_up_weeks` added and three per-episode
extras -- so `panel_decision.py` pools a trend fold through exactly the reader it
uses for a momentum one. Two things are new:

* the four-week member (`ta_ts_t02_h4w`), whose book at a decision is the sum of
  the last four weekly weight vectors at a quarter of capital each rather than
  this week's vector alone (spec 4.2); and
* the three-Sunday warm-up that fills that book before the fold's first episode,
  the same device `carry_fold_run.py` uses for its cohorts.

Everything else -- the universe, the accounting, the manifest linkage, the
skipped-week reset, `MEMBER_HELD_NOTHING`, the registry rows -- is P1.27's,
reused rather than restated.
"""

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.panel_accounting import EpisodeResult, evaluate_episode
from trading_bot.panel_capture import verify_panel_capture
from trading_bot.panel_config import PanelCostTable
from trading_bot.panel_fold_run import _PANEL_MODULES, MEMBER_HELD_NOTHING_REASON_CODE
from trading_bot.panel_reader import FundingEvent, load_funding_events, load_panel_bars
from trading_bot.panel_samples import verify_panel_manifest
from trading_bot.panel_universe import ContractHistory, build_contract_histories, select_universe
from trading_bot.registry import ExperimentRecord, MetadataRegistry
from trading_bot.trend_config import TrendFamilySpec, TrendMember, load_trend_family_spec
from trading_bot.trend_signals import build_trend_weight_vectors, threshold_sign, trend_scores

# Same string, and the same meaning, as `carry_fold_run.FOLD_WARMED_REASON_CODE`:
# this fold's opening book was formed over Sundays preceding its own test
# window. Declared here rather than imported so the trend line does not depend
# on the carry module -- nothing else in this family does, and `_TREND_MODULES`
# would then no longer cover every module the code hash speaks for.
FOLD_WARMED_REASON_CODE = "FOLD_OPENING_BOOK_WARMED_FROM_PRIOR_WEEKS"

_WEEK_NS = 7 * 86_400_000_000_000

# The panel modules this family runs on, plus its own three. Built from
# `panel_fold_run`'s own tuple rather than a copy of it, so a change to any
# module P1.27 evaluates through necessarily moves the trend code hash too.
_TREND_MODULES = (*_PANEL_MODULES, "trend_config.py", "trend_signals.py", "trend_fold_run.py")


class TrendFoldError(RuntimeError):
    """Raised when a trend fold cannot be evaluated or published."""


@dataclass(frozen=True, slots=True)
class TrendFoldArtifact:
    output_path: Path
    report_hash: str
    fold_index: int
    episode_count: int
    skipped_sample_count: int


def warm_up_weeks_of(spec: TrendFamilySpec) -> int:
    """How many Sundays before a fold's first decision are warmed.

    A fold that opened flat would ramp the four-week member's book 1/4 per week
    for three weeks, while the one-week members are fully invested from their
    first episode -- a hold-dependent haircut on the primary metric, not a
    property of the signal. So each fold forms (only) the weight vectors of the
    `max(hold_weeks) - 1` Sundays before its first test decision: that many
    vectors plus the first decision's own fill even the longest-held member's
    book exactly. Three under v1, whose longest hold is four weeks.
    """
    return max(member.hold_weeks for member in spec.members) - 1


def assemble_cohort_book(
    vectors: Sequence[tuple[int, tuple[tuple[str, Decimal], ...]]],
    *,
    hold_weeks: int,
    histories: dict[str, ContractHistory],
    decision_close_ns: int,
) -> tuple[tuple[str, Decimal], ...]:
    """The book a multi-week member holds at one decision (spec 4.2).

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
    would inflate `signalled_contracts` without changing any cash flow.
    """
    if hold_weeks < 1:
        raise TrendFoldError("a holding period is at least one week")
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


def run_trend_fold(
    capture_root: Path,
    *,
    manifest_path: Path,
    family_spec_path: Path,
    output_path: Path,
    registry_path: Path,
    fold_index: int,
) -> TrendFoldArtifact:
    valid, errors = verify_panel_capture(capture_root)
    if not valid:
        raise TrendFoldError("panel capture verification failed: " + ",".join(errors))
    if not verify_panel_manifest(manifest_path):
        raise TrendFoldError("panel manifest verification failed")
    if output_path.exists():
        raise TrendFoldError("trend fold report already exists and is immutable")

    spec, family_spec_hash = load_trend_family_spec(family_spec_path)
    manifest = _load_object(manifest_path)
    if manifest.get("family_spec_hash") != family_spec_hash:
        raise TrendFoldError("family declaration does not match the manifest")
    capture_manifest = _load_object(capture_root / "capture-manifest.json")
    dataset_manifest = _load_object(capture_root / "dataset" / "dataset-manifest.json")
    if manifest.get("capture_root_hash") != capture_manifest.get("capture_root_hash"):
        raise TrendFoldError("manifest is not linked to this capture")
    if manifest.get("dataset_root_hash") != dataset_manifest.get("root_hash"):
        raise TrendFoldError("manifest is not linked to this dataset")

    folds = manifest.get("folds")
    if not isinstance(folds, list):
        raise TrendFoldError("manifest folds are malformed")
    fold = next((item for item in folds if item.get("fold_index") == fold_index), None)
    if fold is None:
        raise TrendFoldError(f"fold {fold_index} is not in the manifest")

    test_end_ns = int(fold["test_end_ns"])
    bars = load_panel_bars(capture_root / "dataset", available_before_ns=test_end_ns + 1)
    histories = build_contract_histories(bars)
    funding_by_contract: dict[str, tuple[FundingEvent, ...]] = {}
    for event in load_funding_events(capture_root / "dataset"):
        funding_by_contract.setdefault(event.contract_id, ())
        funding_by_contract[event.contract_id] += (event,)

    test_ids = [str(value) for value in fold["test_ids"]]
    decisions = sorted(int(value.split(":")[1]) for value in test_ids)
    members: dict[str, TrendMember] = {member.name: member for member in spec.members}
    names = [member.name for member in spec.members] + [item.name for item in spec.controls]
    scenarios: tuple[PanelCostTable, ...] = (spec.costs.base, spec.costs.adverse)
    episodes: dict[tuple[str, str], list[EpisodeResult]] = {
        (name, scenario.name): [] for name in names for scenario in scenarios
    }
    held_nothing: dict[str, list[bool]] = {name: [] for name in names}
    extras: dict[str, list[dict[str, Decimal]]] = {name: [] for name in names}
    carried: dict[tuple[str, str], tuple[tuple[str, Decimal], ...]] = {
        key: () for key in episodes
    }
    # Only a member that holds for more than one week needs its past vectors
    # kept; every other candidate's book is this week's vector.
    cohorts: dict[str, list[tuple[int, tuple[tuple[str, Decimal], ...]]]] = {
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
    # which covers these Sundays, and `select_universe` and
    # `build_trend_weight_vectors` each read only data at or before the Sunday
    # they are asked about, so no future data enters here. A warm-up Sunday whose
    # universe is too small simply contributes no vector: no position exists yet
    # to be flattened, so there is nothing to reset, and the vectors formed on
    # the warm-up Sundays around it keep ageing out on the calendar as usual.
    # Only an in-window skipped week resets, where a real position is closed
    # (P1.27); `carry_fold_run.py`'s warm-up does the same.
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
        vectors = build_trend_weight_vectors(histories, warm_up, spec=spec)
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
        vectors = build_trend_weight_vectors(histories, snapshot, spec=spec)
        eligible = tuple(item.contract_id for item in snapshot.contracts)
        scores = trend_scores(histories, eligible, decision_close_ns)
        mean_score = (
            sum(scores.values(), Decimal(0)) / Decimal(len(scores)) if scores else Decimal(0)
        )
        for name in names:
            member = members.get(name)
            if member is not None and member.hold_weeks > 1:
                retained = cohorts[member.name]
                _retain(
                    retained,
                    vectors[name].weights,
                    formed_ns=decision_close_ns,
                    hold_weeks=member.hold_weeks,
                )
                weights = assemble_cohort_book(
                    retained,
                    hold_weeks=member.hold_weeks,
                    histories=histories,
                    decision_close_ns=decision_close_ns,
                )
            else:
                weights = vectors[name].weights
            # As in P1.27: only a member's own construction can legitimately
            # produce no weights while the universe itself was not too small
            # (every contract inside the dead band, too few rankable contracts
            # to form both quintiles, or -- for the four-week member -- four
            # empty weeks behind it). The controls always have well-defined
            # weights here and `no_trade`'s emptiness is a deliberate baseline,
            # so a control is never marked.
            held_nothing[name].append(member is not None and not weights)
            extras[name].append(
                {
                    "signalled_contracts": Decimal(
                        sum(1 for _, weight in weights if weight != 0)
                    ),
                    "abstained_contracts": _abstained_contracts(member, eligible, scores),
                    "mean_score": mean_score,
                }
            )
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
            "base": _scenario_record(episodes[(name, "base")], held_nothing[name], extras[name]),
            "adverse": _scenario_record(
                episodes[(name, "adverse")], held_nothing[name], extras[name]
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
        "code_hash": _code_hash(),
        "candidates": candidates,
    }
    report_hash = content_sha256(material)
    document = dict(material)
    document["report_hash"] = report_hash
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_bytes(canonical_json(document))
    temporary.replace(output_path)

    _register(spec, manifest, report_hash, registry_path)
    return TrendFoldArtifact(
        output_path=output_path,
        report_hash=report_hash,
        fold_index=fold_index,
        episode_count=episode_count,
        skipped_sample_count=len(skipped),
    )


def _retain(
    vectors: list[tuple[int, tuple[tuple[str, Decimal], ...]]],
    vector: tuple[tuple[str, Decimal], ...],
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


def _scenario_record(
    results: list[EpisodeResult],
    held_nothing: list[bool],
    extras: list[dict[str, Decimal]],
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
                "extras": dict(extra),
            }
            for item, flag, extra in zip(results, held_nothing, extras, strict=True)
        ],
    }


def _register(
    spec: TrendFamilySpec,
    manifest: dict[str, object],
    report_hash: str,
    registry_path: Path,
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
                    code_hash=_code_hash(),
                    random_seed=spec.statistics.random_seed,
                    outcome="completed",
                    result_hash=report_hash,
                    failure_reason=None,
                    created_at_ns=created_at_ns,
                )
            )


def _code_hash() -> str:
    root = Path(__file__).parent
    material = "".join((root / name).read_text(encoding="utf-8") for name in _TREND_MODULES)
    return content_sha256(material)


def _load_object(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TrendFoldError(f"{path.name} must contain a JSON object")
    return document
