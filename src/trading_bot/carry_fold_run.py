"""Evaluate one walk-forward fold of the funding carry family."""

import json
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.carry_accounting import CarryEpisode, evaluate_carry_episode, forced_legs
from trading_bot.carry_config import (
    CarryControl,
    CarryCostTable,
    CarryFamilySpec,
    CarryMember,
    load_carry_family_spec,
)
from trading_bot.carry_signals import (
    WEEK_NS,
    Cohort,
    assemble_book,
    exit_rule_pairs,
    hurdle_minimum_trailing,
    select_control_cohort,
    select_member_cohort,
    trailing_funding,
)
from trading_bot.carry_universe import PairUniverseSnapshot, select_pair_universe
from trading_bot.panel_capture import verify_panel_capture
from trading_bot.panel_fold_run import MEMBER_HELD_NOTHING_REASON_CODE
from trading_bot.panel_reader import FundingEvent, load_funding_events, load_panel_bars
from trading_bot.panel_samples import verify_panel_manifest
from trading_bot.panel_universe import ContractHistory, build_contract_histories
from trading_bot.registry import ExperimentRecord, MetadataRegistry

FOLD_WARMED_REASON_CODE = "FOLD_OPENING_BOOK_WARMED_FROM_PRIOR_WEEKS"

_CARRY_MODULES = (
    "panel_config.py", "panel_dataset.py", "panel_capture.py", "panel_reader.py",
    "panel_universe.py", "panel_accounting.py", "panel_samples.py",
    "carry_config.py", "carry_universe.py", "carry_signals.py", "carry_accounting.py",
    "carry_fold_run.py", "evaluation.py", "strategy.py",
)


class CarryFoldError(RuntimeError):
    """Raised when a carry fold cannot be evaluated or published."""


def control_reference(spec: CarryFamilySpec) -> CarryMember:
    """The member whose lookback and hold the controls borrow (spec 3.3).

    The shortest hold, ties broken by declaration order -- which is what
    Python's `min` does on the declared tuple. Under v1 that resolves to
    `carry_l1w_h4w`, the member the v1 runner named outright, so P1.28's
    controls are unmoved by the rule becoming generic.
    """
    return min(spec.members, key=lambda member: member.hold_weeks)


def warm_up_weeks_of(spec: CarryFamilySpec) -> int:
    """How many Sundays before a fold's first decision are warmed.

    Spec section 8.1 defines the book at t as the union of the last H
    cohorts. A fold that opened flat would instead ramp 1/H per week for H-1
    weeks, and because H differs between members that ramp is a
    member-dependent haircut on the primary metric. So each fold warms its
    cohort state over the `max(H) - 1` Sundays preceding its first test
    decision: that many prior cohorts plus the first decision's own fill even
    the longest-held member's book exactly. Twelve under v1 (longest hold
    thirteen), twenty-five under v2 (longest hold twenty-six).
    """
    return max(member.hold_weeks for member in spec.members) - 1


@dataclass(frozen=True, slots=True)
class CarryFoldArtifact:
    output_path: Path
    report_hash: str
    fold_index: int
    episode_count: int
    skipped_sample_count: int


def run_carry_fold(
    perp_capture_root: Path,
    spot_capture_root: Path,
    *,
    manifest_path: Path,
    family_spec_path: Path,
    output_path: Path,
    registry_path: Path,
    fold_index: int,
) -> CarryFoldArtifact:
    for root, label in ((perp_capture_root, "perpetual"), (spot_capture_root, "spot")):
        valid, errors = verify_panel_capture(root)
        if not valid:
            raise CarryFoldError(f"{label} capture verification failed: " + ",".join(errors))
    if not verify_panel_manifest(manifest_path):
        raise CarryFoldError("panel manifest verification failed")
    if output_path.exists():
        raise CarryFoldError("carry fold report already exists and is immutable")

    spec, family_spec_hash = load_carry_family_spec(family_spec_path)
    manifest = _load_object(manifest_path)
    if manifest.get("family_spec_hash") != family_spec_hash:
        raise CarryFoldError("family declaration does not match the manifest")
    # The two captures are not interchangeable: the perpetual leg carries the
    # funding and the spot leg is the hedge. Passing one capture twice, or a
    # second perpetual capture as the hedge, would silently evaluate a
    # perp-versus-perp book that is not the position under test.
    if spot_capture_root.resolve() == perp_capture_root.resolve():
        raise CarryFoldError("hedge capture must differ from the primary capture")
    perp_manifest = _load_object(perp_capture_root / "capture-manifest.json")
    spot_manifest = _load_object(spot_capture_root / "capture-manifest.json")
    # P1.27's capture predates the `market` key, so its absence means "um".
    if perp_manifest.get("market", "um") != "um":
        raise CarryFoldError(
            f"primary capture must be a perpetual capture, got market "
            f"{perp_manifest.get('market', 'um')}"
        )
    if spot_manifest.get("market") != "spot":
        raise CarryFoldError(
            f"hedge capture must be a spot capture, got market {spot_manifest.get('market')}"
        )
    _require_link(manifest, "capture_root_hash", perp_manifest["capture_root_hash"])
    _require_link(
        manifest, "dataset_root_hash",
        _load_object(perp_capture_root / "dataset" / "dataset-manifest.json")["root_hash"],
    )
    _require_link(manifest, "hedge_capture_root_hash", spot_manifest["capture_root_hash"])
    _require_link(
        manifest, "hedge_dataset_root_hash",
        _load_object(spot_capture_root / "dataset" / "dataset-manifest.json")["root_hash"],
    )

    folds = manifest.get("folds")
    if not isinstance(folds, list):
        raise CarryFoldError("manifest folds are malformed")
    fold = next((item for item in folds if item.get("fold_index") == fold_index), None)
    if fold is None:
        raise CarryFoldError(f"fold {fold_index} is not in the manifest")

    test_end_ns = int(fold["test_end_ns"])
    perp_bars = load_panel_bars(perp_capture_root / "dataset", available_before_ns=test_end_ns + 1)
    spot_bars = load_panel_bars(spot_capture_root / "dataset", available_before_ns=test_end_ns + 1)
    perp_histories = build_contract_histories(perp_bars)
    spot_histories = build_contract_histories(spot_bars)
    leg_histories: dict[str, ContractHistory] = {}
    for cid, history in perp_histories.items():
        leg_histories[f"perp:{cid}"] = history
    for cid, history in spot_histories.items():
        leg_histories[f"spot:{cid}"] = history
    funding_by_leg: dict[str, tuple[FundingEvent, ...]] = {}
    for event in load_funding_events(perp_capture_root / "dataset"):
        leg_key = f"perp:{event.contract_id}"
        funding_by_leg[leg_key] = (*funding_by_leg.get(leg_key, ()), event)

    test_ids = [str(value) for value in fold["test_ids"]]
    decisions = sorted(int(value.split(":")[1]) for value in test_ids)
    members: dict[str, CarryMember] = {m.name: m for m in spec.members}
    controls: dict[str, CarryControl] = {c.name: c for c in spec.controls}
    names: list[str] = [m.name for m in spec.members] + [c.name for c in spec.controls]
    scenarios: tuple[CarryCostTable, ...] = (spec.costs.base, spec.costs.adverse)
    episodes: dict[tuple[str, str], list[CarryEpisode]] = {
        (n, s.name): [] for n in names for s in scenarios
    }
    held_nothing: dict[str, list[bool]] = {n: [] for n in names}
    exit_removals: dict[str, list[int]] = {n: [] for n in names}
    hurdle_rejections: dict[str, list[int]] = {n: [] for n in names}
    cohorts: dict[str, list[Cohort]] = {n: [] for n in names}
    carried: dict[tuple[str, str], tuple[tuple[str, Decimal], ...]] = {k: () for k in episodes}
    no_carry: dict[str, list[str]] = {n: [] for n in names}
    skipped: list[str] = []
    # Fold-persistent: a leg keeps its pair after it leaves the universe, so an
    # exit-only leg can still be attributed to its pair.
    pair_of_leg: dict[str, str] = {}
    reference = control_reference(spec)
    hold_of: dict[str, int] = {
        **{m.name: m.hold_weeks for m in spec.members},
        **{c.name: reference.hold_weeks for c in spec.controls},
    }
    lookback_of: dict[str, int] = {
        **{m.name: m.lookback_weeks for m in spec.members},
        **{c.name: reference.lookback_weeks for c in spec.controls},
    }
    warm_up_weeks = warm_up_weeks_of(spec)

    # Warm-up: form (only) the cohorts of the Sundays before the fold's
    # first test decision, so the first test episode opens on a full book
    # instead of a 1/H stub. No episode is evaluated, nothing is carried and
    # nothing is reported for these weeks -- `carried` stays empty, so the
    # first test episode buys the whole warmed book and pays its full entry
    # turnover inside the window. Bars were loaded with
    # `available_before_ns = test_end_ns + 1`, which covers these Sundays, and
    # `select_pair_universe`/`trailing_funding` each read only data at or
    # before the Sunday they are asked about, so no future data enters here.
    # A warm-up Sunday whose universe is too small simply contributes no
    # cohort; there is no state yet to reset. The exit rule does not run here:
    # the warm-up only forms cohorts, and the first in-window decision applies
    # the rule to the warmed ones, which reads only data at or before it.
    warm_up_closes = (
        [decisions[0] - weeks * WEEK_NS for weeks in range(warm_up_weeks, 0, -1)]
        if decisions
        else []
    )
    for warm_up_close_ns in warm_up_closes:
        warm_up = select_pair_universe(
            perp_histories, spot_histories, pairs=spec.pairs,
            decision_close_ns=warm_up_close_ns, rules=spec.universe,
        )
        if not warm_up.pairs:
            continue
        for pair in warm_up.pairs:
            pair_of_leg.setdefault(f"perp:{pair.perpetual_contract_id}", pair.pair_id)
            pair_of_leg.setdefault(f"spot:{pair.spot_contract_id}", pair.pair_id)
        for name in names:
            cohorts[name].append(_cohort_for(
                name, warm_up, spec=spec, controls=controls, member=members.get(name),
                funding_by_leg=funding_by_leg, lookback_weeks=lookback_of[name],
            ))

    for decision_close_ns in decisions:
        sample_id = f"BINANCE_UM:{decision_close_ns}:w1"
        snapshot = select_pair_universe(
            perp_histories, spot_histories, pairs=spec.pairs,
            decision_close_ns=decision_close_ns, rules=spec.universe,
        )
        if not snapshot.pairs:
            skipped.append(sample_id)
            for episode_key in carried:
                carried[episode_key] = ()
            for name in names:
                cohorts[name] = []
            continue
        # Tiers follow P1.27: this week's universe tier, tier two for a leg that
        # is only being exited (see panel_accounting's `tiers.get(id, 2)`).
        tiers: dict[str, int] = {}
        for pair in snapshot.pairs:
            perp_key = f"perp:{pair.perpetual_contract_id}"
            spot_key = f"spot:{pair.spot_contract_id}"
            tiers[perp_key] = pair.tier
            tiers[spot_key] = pair.tier
            pair_of_leg.setdefault(perp_key, pair.pair_id)
            pair_of_leg.setdefault(spot_key, pair.pair_id)
        for name in names:
            member = members.get(name)
            cohort = _cohort_for(
                name, snapshot, spec=spec, controls=controls, member=member,
                funding_by_leg=funding_by_leg, lookback_weeks=lookback_of[name],
            )
            if "NO_CARRY_COHORT" in cohort.reason_codes:
                no_carry[name].append(sample_id)
            cohorts[name].append(cohort)
            hurdle_rejections[name].append(cohort.hurdle_rejections)
            # Exit rule (spec 3.3), for the members that declare it: a held
            # pair whose trailing one week paid nothing leaves every cohort
            # holding it, before the book is assembled, so it is exited at
            # this decision through ordinary turnover rather than held for
            # another week. Its cohort share stays undeployed until the cohort
            # ages out, exactly as after a forced close, and the pair may be
            # selected again by a later cohort once it pays again. The freshly
            # formed cohort is included: a pair that paid over the lookback
            # but not in the last week is not entered either.
            removed: set[str] = set()
            if member is not None and member.exit_on_negative_funding:
                trailing_one_week: dict[str, Decimal | None] = {
                    entry.pair_id: trailing_funding(
                        funding_by_leg.get(entry.perpetual_leg, ()),
                        decision_close_ns=decision_close_ns, lookback_weeks=1,
                    )
                    for retained in cohorts[name]
                    for entry in retained.entries
                }
                removed = exit_rule_pairs(cohorts[name], trailing_one_week=trailing_one_week)
                if removed:
                    cohorts[name] = _without_pairs(cohorts[name], removed)
            exit_removals[name].append(len(removed))
            # A pair whose leg has no bar at this decision cannot be entered or
            # held. In-window that leg was already force-closed and its pair
            # stripped when it lost its exit bar; a pair that went dark during
            # the warm-up, where no episode runs, is caught only here. Its
            # cohort share stays undeployed, exactly as after a forced close.
            untradeable = {
                entry.pair_id
                for retained in cohorts[name]
                for entry in retained.entries
                if decision_close_ns not in leg_histories[entry.perpetual_leg].closes
                or decision_close_ns not in leg_histories[entry.spot_leg].closes
            }
            if untradeable:
                cohorts[name] = _without_pairs(cohorts[name], untradeable)
            weights = assemble_book(
                tuple(cohorts[name]), hold_weeks=hold_of[name], decision_close_ns=decision_close_ns
            )
            held_nothing[name].append(name in members and not weights)
            forced_pairs: set[str] = set()
            for scenario in scenarios:
                key = (name, scenario.name)
                episode = evaluate_carry_episode(
                    sample_id=sample_id, member=name, decision_close_ns=decision_close_ns,
                    holding_days=spec.holding_days, leg_weights=weights,
                    previous_leg_weights=carried[key],
                    histories=leg_histories, tiers=tiers, funding_by_leg=funding_by_leg,
                    cost_table=scenario, pair_of_leg=pair_of_leg,
                )
                episodes[key].append(episode)
                carried[key] = episode.result.drifted_weights
                for leg in forced_legs(weights, episode.result.drifted_weights):
                    forced_pairs.add(pair_of_leg[leg])
            if forced_pairs:
                cohorts[name] = _without_pairs(cohorts[name], forced_pairs)
        # prune cohorts older than the longest hold so state stays bounded
        longest = max(hold_of.values()) * WEEK_NS
        for name in names:
            cohorts[name] = [
                c for c in cohorts[name] if decision_close_ns - c.decision_close_ns < longest
            ]

    episode_count = len(episodes[(names[0], "base")])
    reason_codes: list[str] = []
    if decisions:
        reason_codes.append(FOLD_WARMED_REASON_CODE)
    if skipped:
        reason_codes.append("SKIPPED_WEEK_EXIT_COST_UNCHARGED")
    if episode_count == 0:
        reason_codes.append("NO_EPISODES_IN_FOLD")
    else:
        reason_codes.append("FOLD_FINAL_EXIT_COST_UNCHARGED")

    candidates = [
        {
            "candidate_name": name,
            "role": "member" if name in members else "control",
            "episode_count": len(episodes[(name, "base")]),
            "no_carry_cohort_sample_ids": no_carry[name],
            "base": _scenario_record(
                episodes[(name, "base")], held_nothing[name],
                exit_removals=exit_removals[name], hurdle_rejections=hurdle_rejections[name],
            ),
            "adverse": _scenario_record(
                episodes[(name, "adverse")], held_nothing[name],
                exit_removals=exit_removals[name], hurdle_rejections=hurdle_rejections[name],
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
        "hedge_capture_root_hash": str(manifest["hedge_capture_root_hash"]),
        "hedge_dataset_root_hash": str(manifest["hedge_dataset_root_hash"]),
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
    return CarryFoldArtifact(output_path, report_hash, fold_index, episode_count, len(skipped))


def _without_pairs(cohorts: list[Cohort], pair_ids: set[str]) -> list[Cohort]:
    """Strip `pair_ids` from every cohort, keeping each cohort's size at
    formation so the removed pairs' capital stays undeployed (spec 8.1)."""
    return [
        Cohort(
            c.decision_close_ns,
            tuple(e for e in c.entries if e.pair_id not in pair_ids),
            c.reason_codes,
            formed_size=c.formed_size if c.formed_size is not None else len(c.entries),
            hurdle_rejections=c.hurdle_rejections,
        )
        for c in cohorts
    ]


def _cohort_for(
    name: str,
    snapshot: PairUniverseSnapshot,
    *,
    spec: CarryFamilySpec,
    controls: dict[str, CarryControl],
    member: CarryMember | None,
    funding_by_leg: dict[str, tuple[FundingEvent, ...]],
    lookback_weeks: int,
) -> Cohort:
    """Select one candidate's cohort for one Sunday.

    Shared by the warm-up and the in-window loop so the two cannot drift:
    a warmed cohort is formed by exactly the rule that forms an in-window
    one, the only difference being that the warm-up does not evaluate,
    carry or report anything around it.
    """
    trailing = {
        pair.pair_id: trailing_funding(
            funding_by_leg.get(f"perp:{pair.perpetual_contract_id}", ()),
            decision_close_ns=snapshot.decision_close_ns, lookback_weeks=lookback_weeks,
        )
        for pair in snapshot.pairs
    }
    control = controls.get(name)
    if control is not None:
        return select_control_cohort(
            snapshot, kind=control.kind, trailing=trailing,
            selection=spec.selection, random_seed=spec.statistics.random_seed,
        )
    if member is None:
        raise CarryFoldError(f"{name} is neither a declared member nor a declared control")
    # The hurdle is priced off the BASE table (spec 3.3): it asks whether the
    # carry is worth entering at all, which is a property of the declaration,
    # not of the scenario the same cohort is later evaluated under.
    hurdle: dict[str, Decimal] | None = None
    if member.hurdle_multiple is not None:
        hurdle = {
            pair.pair_id: hurdle_minimum_trailing(
                cost_table=spec.costs.base, tier=pair.tier,
                multiple=member.hurdle_multiple,
                lookback_weeks=lookback_weeks, hold_weeks=member.hold_weeks,
            )
            for pair in snapshot.pairs
        }
    return select_member_cohort(
        snapshot, trailing=trailing, selection=spec.selection, hurdle=hurdle
    )


def _scenario_record(
    results: list[CarryEpisode],
    held_nothing: list[bool],
    *,
    exit_removals: list[int],
    hurdle_rejections: list[int],
) -> dict[str, object]:
    return {
        "total_net_return": sum((item.result.net_return for item in results), Decimal(0)),
        "episodes": [
            {
                "sample_id": item.result.sample_id,
                "net_return": item.result.net_return,
                "gross_return": item.result.gross_return,
                "turnover": item.result.turnover,
                "trading_cost": item.result.trading_cost,
                "funding_cost": item.result.funding_cost,
                "forced_close_cost": item.result.forced_close_cost,
                "gross_exposure": item.result.gross_exposure,
                "net_exposure": item.result.net_exposure,
                "forced_close_count": item.result.forced_close_count,
                "reason_codes": [MEMBER_HELD_NOTHING_REASON_CODE] if flag else [],
                "contract_net_contributions": [
                    [cid, v] for cid, v in item.result.contract_net_contributions
                ],
                "extras": {
                    "funding_collected": item.funding_collected,
                    "basis_pnl": item.basis_pnl,
                    "spot_trading_cost": item.spot_trading_cost,
                    "perpetual_trading_cost": item.perpetual_trading_cost,
                    "forced_spot_legs": Decimal(item.forced_spot_legs),
                    "forced_perpetual_legs": Decimal(item.forced_perpetual_legs),
                    "exit_rule_removals": Decimal(exited),
                    "hurdle_rejections": Decimal(rejected),
                },
            }
            for item, flag, exited, rejected in zip(
                results, held_nothing, exit_removals, hurdle_rejections, strict=True
            )
        ],
    }


def _require_link(manifest: dict[str, object], key: str, expected: object) -> None:
    if manifest.get(key) != expected:
        raise CarryFoldError(f"manifest is not linked to this capture ({key})")


def _register(
    spec: CarryFamilySpec, manifest: dict[str, object], report_hash: str, registry_path: Path
) -> None:
    split_hash = str(manifest["split_manifest_hash"])
    family_id = uuid5(NAMESPACE_URL, f"{split_hash}:{spec.family_name}")
    created_at_ns = time.time_ns()
    with MetadataRegistry(registry_path) as registry:
        for member in spec.members:
            registry.register_experiment(ExperimentRecord(
                experiment_id=uuid5(
                    family_id, f"{member.name}:{spec.statistics.random_seed}:{report_hash}"
                ),
                family_id=family_id, candidate_name=member.name, hypothesis=spec.hypothesis,
                split_manifest_hash=split_hash, code_hash=_code_hash(),
                random_seed=spec.statistics.random_seed, outcome="completed",
                result_hash=report_hash, failure_reason=None, created_at_ns=created_at_ns,
            ))


def _code_hash() -> str:
    root = Path(__file__).parent
    material = "".join((root / name).read_text(encoding="utf-8") for name in _CARRY_MODULES)
    return content_sha256(material)


def _load_object(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise CarryFoldError(f"{path.name} must contain a JSON object")
    return document
