"""Evaluate one walk-forward fold of the funding carry family."""

import json
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.carry_accounting import CarryEpisode, evaluate_carry_episode, forced_legs
from trading_bot.carry_config import (
    CarryCapital,
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
    assemble_slot_book,
    entries_for,
    exit_rule_pairs,
    fill_slots,
    hurdle_minimum_trailing,
    random_pair_order,
    rank_paying_pairs,
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
# What a slot family reports per episode beside the eight extras every carry
# family reports (spec 4.6). A cohort family reports none of them.
_SLOT_EXTRA_KEYS = ("filled_slots", "slot_fills", "slot_releases", "no_fill")

_CARRY_MODULES = (
    "panel_config.py", "panel_dataset.py", "panel_capture.py", "panel_reader.py",
    "panel_universe.py", "panel_accounting.py", "panel_samples.py",
    "carry_config.py", "carry_universe.py", "carry_signals.py", "carry_accounting.py",
    "carry_fold_run.py", "evaluation.py", "strategy.py",
)


class CarryFoldError(RuntimeError):
    """Raised when a carry fold cannot be evaluated or published."""


def carry_module_names() -> tuple[str, ...]:
    """The module file names a carry fold report's `code_hash` covers.

    Public so the holdout read (protocol 16.2) can hash the same evaluation
    path this runner hashes plus the two modules only it uses, rather than
    keeping a second, silently divergent list of the carry stack.
    """
    return _CARRY_MODULES


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

    A slot family warms nothing (spec 3.3). Its capital is declared rather
    than accumulated: every slot is empty at a fold's first decision and is
    filled there, in full, so there is no 1/H ramp to warm away and no
    member-dependent haircut to correct. Zero, and the report says so.
    """
    if spec.capital is not None:
        return 0
    return max(member.hold_weeks for member in spec.members) - 1


@dataclass(frozen=True, slots=True)
class CarryFoldArtifact:
    output_path: Path
    report_hash: str
    fold_index: int
    episode_count: int
    skipped_sample_count: int


@dataclass(frozen=True, slots=True)
class SlotEntry:
    """One filled slot of the book a decision run ended on.

    Spec 4.1 reads the shadow book off the last decision's slot state, and
    spec 4.2 states that book per slot: the pair, the Sunday it was entered
    and how many weeks it has been held. So a slot says all three rather than
    only naming the pair and leaving a reader to rediscover its age.
    """

    pair_id: str
    perpetual_leg: str
    spot_leg: str
    tier: int
    entry_decision_close_ns: int
    weeks_held: int


@dataclass(frozen=True, slots=True)
class FinalBook:
    """The book one candidate ended a decision run on (spec 4.1).

    `leg_weights` is what the last evaluated decision actually traded, before
    that episode's forced closes; `slots` is the slot state the same decision
    ended with, forced closes stripped and the prune applied, which is what
    the next decision would start from. A cohort candidate has no slots --
    a weekly cohort is not a slot and reading it as one would invent a book --
    so `slots` is empty for it and `leg_weights` is its whole book.
    """

    decision_close_ns: int
    slots: tuple[SlotEntry, ...]
    leg_weights: tuple[tuple[str, Decimal], ...]


@dataclass(frozen=True, slots=True)
class DecisionRun:
    """What evaluating a list of decisions produced, before it is sealed.

    Exactly the parts of a fold report that come out of the decision loop
    rather than out of the manifest, so a fold report and a holdout read are
    assembled from one evaluation rather than from two copies of it.

    `final_books` is additive and defaults to empty: no fold report material
    reads it, and a caller that wants the book the run ended on -- the weekly
    shadow book, which is exactly one run of this loop (spec 4.1) -- reads it
    instead of re-deriving a book from a second copy of the rules.
    """

    candidates: list[dict[str, object]]
    skipped_sample_ids: list[str]
    episode_count: int
    reason_codes: list[str]
    warm_up_weeks: int
    final_books: dict[str, FinalBook] = field(default_factory=dict)


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
    run = evaluate_carry_decisions(
        spec,
        perp_histories=perp_histories, spot_histories=spot_histories,
        leg_histories=leg_histories, funding_by_leg=funding_by_leg,
        decisions=decisions,
    )
    material: dict[str, object] = {
        "report_version": "1.0.0",
        "status": "development_only",
        "reason_codes": run.reason_codes,
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
        "skipped_sample_ids": run.skipped_sample_ids,
        "warm_up_weeks": run.warm_up_weeks,
        "code_hash": _code_hash(),
        "candidates": run.candidates,
    }
    report_hash = content_sha256(material)
    document = dict(material)
    document["report_hash"] = report_hash
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_bytes(canonical_json(document))
    temporary.replace(output_path)
    _register(spec, manifest, report_hash, registry_path)
    return CarryFoldArtifact(
        output_path, report_hash, fold_index, run.episode_count, len(run.skipped_sample_ids)
    )


def evaluate_carry_decisions(
    spec: CarryFamilySpec,
    *,
    perp_histories: dict[str, ContractHistory],
    spot_histories: dict[str, ContractHistory],
    leg_histories: dict[str, ContractHistory],
    funding_by_leg: dict[str, tuple[FundingEvent, ...]],
    decisions: list[int],
    candidate_names: tuple[str, ...] | None = None,
) -> DecisionRun:
    """Evaluate one list of weekly decisions for a declaration's candidates.

    The fold runner's in-window loop, warm-up included: `warm_up_weeks_of(spec)`
    Sundays before `decisions[0]` form cohorts that no episode is evaluated on,
    then every decision is evaluated on the cohort or the slot book as the
    declaration says. A holdout read is the same loop over the holdout's
    decisions, so it runs through here rather than through a second copy.

    `candidate_names` restricts the evaluation to those members and controls,
    in declaration order (`None` is every candidate); a name the declaration
    does not carry is refused. Nothing else is filtered: the controls still
    borrow `control_reference(spec)`'s hold and the prune still bounds state at
    the declaration's longest hold, so a candidate's record is the same record
    whether or not its siblings were asked for.
    """
    members: dict[str, CarryMember] = {m.name: m for m in spec.members}
    controls: dict[str, CarryControl] = {c.name: c for c in spec.controls}
    names: list[str] = [m.name for m in spec.members] + [c.name for c in spec.controls]
    if candidate_names is not None:
        for name in candidate_names:
            if name not in members and name not in controls:
                raise CarryFoldError(
                    f"{name} is neither a declared member nor a declared control"
                )
        requested = set(candidate_names)
        names = [name for name in names if name in requested]
    scenarios: tuple[CarryCostTable, ...] = (spec.costs.base, spec.costs.adverse)
    episodes: dict[tuple[str, str], list[CarryEpisode]] = {
        (n, s.name): [] for n in names for s in scenarios
    }
    held_nothing: dict[str, list[bool]] = {n: [] for n in names}
    exit_removals: dict[str, list[int]] = {n: [] for n in names}
    hurdle_rejections: dict[str, list[int]] = {n: [] for n in names}
    # A slot family's slot state is the same `cohorts[name]` list, each entry a
    # one-pair cohort standing for one filled slot: `UNIVERSE_TOO_SMALL`'s
    # reset, the forced-close strip and the prune then need no second spelling,
    # and `len(cohorts[name])` is how many slots are filled.
    cohorts: dict[str, list[Cohort]] = {n: [] for n in names}
    slot_extras: dict[str, list[dict[str, Decimal]]] = {n: [] for n in names}
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
    # Spec 3.2/4.3: a declared capital block makes this a slot family, and then
    # every member and the `random_pairs` dominance control run the slot book.
    # `no_trade` holds nothing either way, and `all_pairs_ew` is P1.28's cohort
    # control unchanged -- it is context, never a gate, and is not executable
    # at the declared capital in the first place.
    capital = spec.capital
    slot_names: set[str] = (
        set()
        if capital is None
        else {name for name in names if name in members}
        | {name for name in names if name in controls and controls[name].kind == "random_pairs"}
    )

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

    # The last evaluated decision's tiers, which price the fold's uncharged
    # final exit; empty until the first decision that is not skipped.
    tiers: dict[str, int] = {}
    # Spec 4.1's book, recorded as each decision ends rather than read off the
    # live state after the loop: a trailing skipped week empties the state
    # without evaluating anything, and would otherwise label the last
    # evaluated decision's weights with a book it never held.
    last_weights: dict[str, tuple[tuple[str, Decimal], ...]] = {}
    last_slot_state: dict[str, tuple[Cohort, ...]] = {}
    last_decision_close_ns: int | None = None
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
        tiers = {}
        for pair in snapshot.pairs:
            perp_key = f"perp:{pair.perpetual_contract_id}"
            spot_key = f"spot:{pair.spot_contract_id}"
            tiers[perp_key] = pair.tier
            tiers[spot_key] = pair.tier
            pair_of_leg.setdefault(perp_key, pair.pair_id)
            pair_of_leg.setdefault(spot_key, pair.pair_id)
        for name in names:
            member = members.get(name)
            # The two books differ only in how this decision's weights are
            # arrived at. Everything downstream -- both scenarios, the forced
            # close, what is carried into next week -- is one code path.
            if capital is not None and name in slot_names:
                step = _slot_decision(
                    snapshot, cohorts[name], spec=spec, capital=capital, member=member,
                    funding_by_leg=funding_by_leg, leg_histories=leg_histories,
                    lookback_weeks=lookback_of[name], hold_weeks=hold_of[name],
                )
                cohorts[name] = step.held
                if step.no_fill:
                    no_carry[name].append(sample_id)
                hurdle_rejections[name].append(step.hurdle_rejections)
                exit_removals[name].append(step.exit_removals)
                weights = assemble_slot_book(
                    tuple(cohorts[name]), pair_slots=capital.pair_slots,
                    hold_weeks=hold_of[name], decision_close_ns=decision_close_ns,
                )
                counts = (len(cohorts[name]), step.fills, step.releases, step.no_fill)
            else:
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
                # another week. Its cohort share stays undeployed until the
                # cohort ages out, exactly as after a forced close, and the
                # pair may be selected again by a later cohort once it pays
                # again. The freshly formed cohort is included: a pair that
                # paid over the lookback but not in the last week is not
                # entered either.
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
                # A pair that went dark cannot be entered or held; its cohort
                # share stays undeployed, exactly as after a forced close.
                untradeable = _untradeable_pairs(
                    cohorts[name],
                    leg_histories=leg_histories, decision_close_ns=decision_close_ns,
                )
                if untradeable:
                    cohorts[name] = _without_pairs(cohorts[name], untradeable)
                weights = assemble_book(
                    tuple(cohorts[name]), hold_weeks=hold_of[name],
                    decision_close_ns=decision_close_ns,
                )
                # A slot family reports the slot mechanics for every candidate
                # (spec 4.6), so the two it does not run on the slot book --
                # `no_trade` and `all_pairs_ew` -- report them at zero rather
                # than reporting a different set of keys from their siblings.
                counts = (0, 0, 0, 0)
            if capital is not None:
                slot_extras[name].append({
                    key: Decimal(value)
                    for key, value in zip(_SLOT_EXTRA_KEYS, counts, strict=True)
                })
            last_weights[name] = weights
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
                cohorts[name] = _without_pairs(
                    cohorts[name], forced_pairs, drop_empty=name in slot_names
                )
        # prune cohorts older than the longest hold so state stays bounded
        longest = max(hold_of.values()) * WEEK_NS
        for name in names:
            cohorts[name] = [
                c for c in cohorts[name] if decision_close_ns - c.decision_close_ns < longest
            ]
            last_slot_state[name] = tuple(cohorts[name])
        last_decision_close_ns = decision_close_ns

    episode_count = len(episodes[(names[0], "base")]) if names else 0
    # Spec 3.3: what liquidating the fold's last book would cost, stated
    # because a slot book never warms and so leaves its whole final exit
    # outside the window. `tiers` is the last evaluated decision's.
    uncharged: dict[tuple[str, str], Decimal] = (
        {}
        if capital is None
        else {
            (name, scenario.name): _uncharged_exit_cost(
                carried[(name, scenario.name)], tiers=tiers, cost_table=scenario
            )
            for name in names
            for scenario in scenarios
        }
    )
    reason_codes: list[str] = []
    if decisions and warm_up_weeks > 0:
        reason_codes.append(FOLD_WARMED_REASON_CODE)
    if skipped:
        reason_codes.append("SKIPPED_WEEK_EXIT_COST_UNCHARGED")
    if episode_count == 0:
        reason_codes.append("NO_EPISODES_IN_FOLD")
    else:
        reason_codes.append("FOLD_FINAL_EXIT_COST_UNCHARGED")

    candidates: list[dict[str, object]] = [
        {
            "candidate_name": name,
            "role": "member" if name in members else "control",
            "episode_count": len(episodes[(name, "base")]),
            "no_carry_cohort_sample_ids": no_carry[name],
            "base": _scenario_record(
                episodes[(name, "base")], held_nothing[name],
                exit_removals=exit_removals[name], hurdle_rejections=hurdle_rejections[name],
                slot_extras=slot_extras[name] if capital is not None else None,
                uncharged_final_exit_cost=uncharged.get((name, "base")),
            ),
            "adverse": _scenario_record(
                episodes[(name, "adverse")], held_nothing[name],
                exit_removals=exit_removals[name], hurdle_rejections=hurdle_rejections[name],
                slot_extras=slot_extras[name] if capital is not None else None,
                uncharged_final_exit_cost=uncharged.get((name, "adverse")),
            ),
        }
        for name in names
    ]
    final_books: dict[str, FinalBook] = (
        {}
        if last_decision_close_ns is None
        else _final_books(
            last_slot_state,
            leg_weights=last_weights,
            decision_close_ns=last_decision_close_ns,
            slot_names=slot_names,
        )
    )
    return DecisionRun(
        candidates=candidates,
        skipped_sample_ids=skipped,
        episode_count=episode_count,
        reason_codes=reason_codes,
        warm_up_weeks=warm_up_weeks,
        final_books=final_books,
    )


def _final_books(
    slot_state: dict[str, tuple[Cohort, ...]],
    *,
    leg_weights: dict[str, tuple[tuple[str, Decimal], ...]],
    decision_close_ns: int,
    slot_names: set[str],
) -> dict[str, FinalBook]:
    """The book each evaluated candidate ended on, from the state the loop left.

    Spec 4.1 makes the last decision's slot state and leg weights the book, so
    both are read off that state rather than recomputed from a second copy of
    the rules. Only a slot family has slots; a slot is a `Cohort` holding
    exactly one pair, so anything else is a bug in the slot bookkeeping and is
    refused rather than guessed at. A candidate no decision was evaluated for
    has no entry at all, which is why this iterates the recorded weights.
    """
    books: dict[str, FinalBook] = {}
    for name, weights in leg_weights.items():
        slots: list[SlotEntry] = []
        if name in slot_names:
            for slot in slot_state[name]:
                if len(slot.entries) != 1:
                    raise CarryFoldError(
                        f"{name} ended on a slot holding {len(slot.entries)} pairs; "
                        "a slot holds exactly one"
                    )
                entry = slot.entries[0]
                slots.append(
                    SlotEntry(
                        pair_id=entry.pair_id,
                        perpetual_leg=entry.perpetual_leg,
                        spot_leg=entry.spot_leg,
                        tier=entry.tier,
                        entry_decision_close_ns=slot.decision_close_ns,
                        weeks_held=(decision_close_ns - slot.decision_close_ns) // WEEK_NS,
                    )
                )
        books[name] = FinalBook(
            decision_close_ns=decision_close_ns,
            slots=tuple(slots),
            leg_weights=weights,
        )
    return books


def _without_pairs(
    cohorts: list[Cohort], pair_ids: set[str], *, drop_empty: bool = False
) -> list[Cohort]:
    """Strip `pair_ids` from every cohort, keeping each cohort's size at
    formation so the removed pairs' capital stays undeployed (spec 8.1).

    `drop_empty` is the slot book's reading of the same removal (spec 3.2): a
    slot holds one pair, so stripping that pair does not leave a shrunken
    cohort whose share stays undeployed until it ages out -- it empties the
    slot, and an empty slot is free to be filled at the very next decision.
    """
    stripped = [
        Cohort(
            c.decision_close_ns,
            tuple(e for e in c.entries if e.pair_id not in pair_ids),
            c.reason_codes,
            formed_size=c.formed_size if c.formed_size is not None else len(c.entries),
            hurdle_rejections=c.hurdle_rejections,
        )
        for c in cohorts
    ]
    return [c for c in stripped if c.entries] if drop_empty else stripped


def _untradeable_pairs(
    cohorts: list[Cohort],
    *,
    leg_histories: dict[str, ContractHistory],
    decision_close_ns: int,
) -> set[str]:
    """Held pairs that have no close on one of their legs at this decision.

    P1.28's untradeable rule, which the slot book takes over unchanged (spec
    3.2), so the two books read it here rather than each keeping its own copy.
    A pair whose leg has no bar cannot be entered or held: there is no price to
    trade it at. In-window that leg was already force-closed and its pair
    stripped when it lost its exit bar; a pair that went dark during the
    warm-up, where no episode runs, is caught only here.
    """
    return {
        entry.pair_id
        for cohort in cohorts
        for entry in cohort.entries
        if decision_close_ns not in leg_histories[entry.perpetual_leg].closes
        or decision_close_ns not in leg_histories[entry.spot_leg].closes
    }


@dataclass(frozen=True, slots=True)
class _SlotStep:
    """One slot candidate's slots after a decision's release, exit and fill."""

    held: list[Cohort]
    releases: int
    exit_removals: int
    fills: int
    no_fill: int
    hurdle_rejections: int


def _slot_decision(
    snapshot: PairUniverseSnapshot,
    slots: list[Cohort],
    *,
    spec: CarryFamilySpec,
    capital: CarryCapital,
    member: CarryMember | None,
    funding_by_leg: dict[str, tuple[FundingEvent, ...]],
    leg_histories: dict[str, ContractHistory],
    lookback_weeks: int,
    hold_weeks: int,
) -> _SlotStep:
    """Release, empty and refill one candidate's slots for one Sunday.

    Spec 3.2's three steps in the order it declares them. A slot is a `Cohort`
    holding exactly one pair, so releasing one is dropping it from the list
    rather than shrinking it: unlike a weekly cohort, a slot's capital is not
    stranded by losing its pair, it is free capital the same decision can
    redeploy. `member` is `None` for the `random_pairs` control, which fills
    the same slots from the seeded draw rather than from the funding ranking.
    """
    decision_close_ns = snapshot.decision_close_ns
    exits_on_negative = member is not None and member.exit_on_negative_funding
    # Measured once for every pair either step 2 or step 3 will ask about: the
    # pairs already in a slot (a held pair need not still be in the universe)
    # and the pairs this week's ranking can offer.
    one_week: dict[str, Decimal | None] = {}
    if exits_on_negative:
        legs = {entry.pair_id: entry.perpetual_leg for slot in slots for entry in slot.entries}
        for pair in snapshot.pairs:
            legs.setdefault(pair.pair_id, f"perp:{pair.perpetual_contract_id}")
        one_week = {
            pair_id: trailing_funding(
                funding_by_leg.get(leg, ()),
                decision_close_ns=decision_close_ns, lookback_weeks=1,
            )
            for pair_id, leg in legs.items()
        }

    # 1. Release. A slot turns `hold_weeks` old and empties; so does one whose
    # pair lost a leg's bar at this decision and cannot be traded out of. A
    # slot force-closed in the previous episode is already gone -- the shared
    # evaluation path dropped it there, as it strips a cohort's pair.
    held_slots = [
        slot
        for slot in slots
        if decision_close_ns - slot.decision_close_ns < hold_weeks * WEEK_NS
    ]
    releases = len(slots) - len(held_slots)
    untradeable = _untradeable_pairs(
        held_slots, leg_histories=leg_histories, decision_close_ns=decision_close_ns
    )
    if untradeable:
        standing = len(held_slots)
        held_slots = _without_pairs(held_slots, untradeable, drop_empty=True)
        releases += standing - len(held_slots)

    # 2. Exit rule, per slot, for the members that declare it.
    removed: set[str] = set()
    if exits_on_negative:
        removed = exit_rule_pairs(held_slots, trailing_one_week=one_week)
        if removed:
            held_slots = _without_pairs(held_slots, removed, drop_empty=True)

    # 3. Fill the empty slots from this week's ranking, in rank order.
    held = {entry.pair_id for slot in held_slots for entry in slot.entries}
    rejections = 0
    if member is None:
        ranked = random_pair_order(
            snapshot, selection=spec.selection, random_seed=spec.statistics.random_seed
        )
    else:
        ranked, rejections = rank_paying_pairs(
            snapshot,
            trailing=_trailing_by_pair(
                snapshot, funding_by_leg=funding_by_leg, lookback_weeks=lookback_weeks
            ),
            selection=spec.selection,
            hurdle=_hurdle_by_pair(
                member, snapshot, cost_table=spec.costs.base, lookback_weeks=lookback_weeks
            ),
        )
    # `skip` says outright what this decision may not enter: what a slot
    # already holds, and -- under the exit rule -- a pair whose last week paid
    # nothing, which is not entered any more than it is held (spec 3.2).
    skip = set(held)
    if exits_on_negative:
        skip |= {
            pair_id
            for pair_id in ranked
            for value in (one_week.get(pair_id),)
            if value is None or value <= 0
        }
    new = fill_slots(
        ranked, held=held, skip=skip, free=capital.pair_slots - len(held_slots)
    )
    held_slots.extend(
        Cohort(decision_close_ns, entries_for(snapshot, [pair_id]), (), formed_size=1)
        for pair_id in new
    )
    return _SlotStep(
        held=held_slots,
        releases=releases,
        exit_removals=len(removed),
        fills=len(new),
        # Fewer than `minimum_selected` pairs paid, so the ranking is empty and
        # no slot can be filled this week -- the slot book's reading of the
        # cohort book's `NO_CARRY_COHORT`.
        no_fill=0 if ranked else 1,
        hurdle_rejections=rejections,
    )


def _uncharged_exit_cost(
    weights: tuple[tuple[str, Decimal], ...],
    *,
    tiers: dict[str, int],
    cost_table: CarryCostTable,
) -> Decimal:
    """What liquidating this book would cost, in units of capital (spec 3.3).

    A fold's last episode is never exited inside the window
    (`FOLD_FINAL_EXIT_COST_UNCHARGED`), and a slot book leaves its whole exit
    outside it, because it holds no warmed cohorts whose exits fall in-window.
    So the fold report states the cost rather than leaving it implicit: every
    drifted leg's absolute weight at one side of its own fee -- spot or
    perpetual as the leg says -- plus its tier's slippage, tier two for a leg
    this decision's universe does not name, which is exactly how the turnover
    accounting prices a side. Zero for a book that holds nothing, which is
    also the fold that ran no episode at all.
    """
    total = Decimal(0)
    for leg, weight in weights:
        slippage = (
            cost_table.slippage_bps_per_side_tier_one
            if tiers.get(leg, 2) == 1
            else cost_table.slippage_bps_per_side_tier_two
        )
        fee = (
            cost_table.spot_fee_bps_per_side
            if leg.startswith("spot:")
            else cost_table.perpetual_fee_bps_per_side
        )
        total += abs(weight) * (fee + slippage) / Decimal(10_000)
    return total


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
    trailing = _trailing_by_pair(
        snapshot, funding_by_leg=funding_by_leg, lookback_weeks=lookback_weeks
    )
    control = controls.get(name)
    if control is not None:
        return select_control_cohort(
            snapshot, kind=control.kind, trailing=trailing,
            selection=spec.selection, random_seed=spec.statistics.random_seed,
        )
    if member is None:
        raise CarryFoldError(f"{name} is neither a declared member nor a declared control")
    return select_member_cohort(
        snapshot,
        trailing=trailing,
        selection=spec.selection,
        hurdle=_hurdle_by_pair(
            member, snapshot, cost_table=spec.costs.base, lookback_weeks=lookback_weeks
        ),
    )


def _trailing_by_pair(
    snapshot: PairUniverseSnapshot,
    *,
    funding_by_leg: dict[str, tuple[FundingEvent, ...]],
    lookback_weeks: int,
) -> dict[str, Decimal | None]:
    """Each eligible pair's trailing funding over the lookback, off its
    perpetual leg -- what both books rank a Sunday's candidates by."""
    return {
        pair.pair_id: trailing_funding(
            funding_by_leg.get(f"perp:{pair.perpetual_contract_id}", ()),
            decision_close_ns=snapshot.decision_close_ns, lookback_weeks=lookback_weeks,
        )
        for pair in snapshot.pairs
    }


def _hurdle_by_pair(
    member: CarryMember,
    snapshot: PairUniverseSnapshot,
    *,
    cost_table: CarryCostTable,
    lookback_weeks: int,
) -> dict[str, Decimal] | None:
    """The minimum trailing funding each pair must carry to be ranked at all.

    `None` for a member that declares no hurdle, which is every v4 member.
    The hurdle is priced off the BASE table (spec 3.3): it asks whether the
    carry is worth entering at all, which is a property of the declaration,
    not of the scenario the same pair is later evaluated under.
    """
    if member.hurdle_multiple is None:
        return None
    return {
        pair.pair_id: hurdle_minimum_trailing(
            cost_table=cost_table, tier=pair.tier, multiple=member.hurdle_multiple,
            lookback_weeks=lookback_weeks, hold_weeks=member.hold_weeks,
        )
        for pair in snapshot.pairs
    }


def _scenario_record(
    results: list[CarryEpisode],
    held_nothing: list[bool],
    *,
    exit_removals: list[int],
    hurdle_rejections: list[int],
    slot_extras: list[dict[str, Decimal]] | None = None,
    uncharged_final_exit_cost: Decimal | None = None,
) -> dict[str, object]:
    """One candidate's episodes under one cost table.

    `slot_extras` carries the per-episode slot bookkeeping of a slot family
    and is `None` for a cohort family, whose episodes then keep exactly the
    eight extras keys they have always had; `uncharged_final_exit_cost` is
    likewise stated only where a slot book leaves one (spec 4.6).
    """
    slots = slot_extras if slot_extras is not None else [{} for _ in results]
    record: dict[str, object] = {
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
                    **extra,
                },
            }
            for item, flag, exited, rejected, extra in zip(
                results, held_nothing, exit_removals, hurdle_rejections, slots, strict=True
            )
        ],
    }
    if uncharged_final_exit_cost is not None:
        record["uncharged_final_exit_cost"] = uncharged_final_exit_cost
    return record


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
