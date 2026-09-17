"""Read a carry family's final holdout once, for one derived candidate.

Protocol section 2.3 opens the final holdout exactly once, for a named release
candidate; section 16.2 fixes how that read is executed, before any holdout is
opened. This module is that execution: it derives the candidate from the
family's decision artifact rather than taking one by hand, checks that the
extended captures it reads bars from are verified supersets of the originals
the manifest was built on, evaluates only the candidate and the two dominance
controls over the original manifest's `final_holdout_ids`, applies the fixed
confirmation criteria and seals a single-use artifact whose registry record is
keyed by the declaration alone -- so a second read of the same family refuses,
on a republished walk-forward manifest as much as on the first one.

Every refusal is a `CarryHoldoutError`: a holdout that cannot be read under
these rules is not read at all, because a failed or spoiled read cannot be
retried (a holdout is not a validation set).
"""

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.capture_lineage import verify_capture_superset
from trading_bot.carry_config import CarryFamilySpec, load_carry_family_spec
from trading_bot.carry_fold_run import (
    CarryFoldError,
    DecisionRun,
    carry_module_names,
    evaluate_carry_decisions,
)
from trading_bot.panel_capture import verify_panel_capture
from trading_bot.panel_decision import concentration_shares
from trading_bot.panel_fold_run import verify_panel_fold_report
from trading_bot.panel_reader import FundingEvent, load_funding_events, load_panel_bars
from trading_bot.panel_samples import verify_panel_manifest
from trading_bot.panel_universe import ContractHistory, build_contract_histories
from trading_bot.registry import ArtifactRecord, MetadataRegistry

DAY_NS = 86_400_000_000_000
HOLDOUT_ARTIFACT_KIND = "holdout"
# What a dry run (`check_only`) returns instead of a verdict: every input check
# held, nothing was evaluated and nothing was written, so the family's one read
# is still unspent.
HOLDOUT_CHECKED_VERDICT = "checked"
# Spec section 5.5: a holdout of 26 weekly decisions confirms nothing if a
# sixth of it never happened, so more than four UNIVERSE_TOO_SMALL skips fail
# the read rather than shrinking the window it is judged on.
MAX_SKIPPED_HOLDOUT_DECISIONS = 4
# Spec section 5.4: the fold gates' concentration rule, applied to the holdout
# alone. Fixed here rather than read from the declaration because the holdout
# criteria are frozen by the protocol before any holdout is opened.
CONCENTRATION_LIMIT = Decimal("0.5")
# Spec section 3: the two dominance controls, and nothing else. The context
# control (`all_pairs_ew`) and the other members are never evaluated, so their
# holdout stays unread and a later generation is not informed by it.
DOMINANCE_CONTROL_NAMES: tuple[str, ...] = ("no_trade", "random_pairs")
# What every fold report the decision pooled must share with the walk-forward
# manifest this read takes its calendar from: the same walk-forward, the same
# published manifest, and the same two original captures.
_FOLD_MANIFEST_LINKS: tuple[str, ...] = (
    "split_manifest_hash", "manifest_hash", "capture_root_hash", "hedge_capture_root_hash",
)
# The evaluation path the fold runner hashes, plus the three modules only a
# holdout read runs through. `panel_decision.py` is one of them because
# `concentration_shares` decides two of the eight criteria: a change to it
# moves a verdict, so it belongs in the hash the verdict is sealed under. The
# fold runner's own list stays as it is -- the v1 fold-0 pin is hashed from it.
_HOLDOUT_MODULES: tuple[str, ...] = (
    *carry_module_names(), "carry_holdout_run.py", "capture_lineage.py", "panel_decision.py",
)


class CarryHoldoutError(RuntimeError):
    """Raised when a carry holdout cannot be read or published."""


@dataclass(frozen=True, slots=True)
class CarryHoldoutArtifact:
    """What a read published, or -- on a dry run -- what it would publish.

    `report_hash` is `None` exactly when `verdict` is `HOLDOUT_CHECKED_VERDICT`:
    a dry run evaluates nothing and writes nothing, so there is no document to
    hash and `output_path` is the path the real read would write to.
    """

    output_path: Path
    report_hash: str | None
    candidate_name: str
    verdict: str


@dataclass(frozen=True, slots=True)
class _VerifiedInputs:
    """What the ordered checks proved, ready for the evaluation to use."""

    spec: CarryFamilySpec
    family_spec_hash: str
    manifest: dict[str, object]
    holdout_ids: tuple[str, ...]
    decision: dict[str, object]
    decision_report_hash: str
    fold_reports: tuple[dict[str, object], ...]
    artifact_id: UUID
    capture_hashes: dict[str, str]


def derive_candidate(
    decision: dict[str, object], fold_reports: list[dict[str, object]]
) -> tuple[str, Decimal]:
    """The one member the holdout is opened for, and the value it won on.

    Spec section 3: the candidate is derived, never chosen. Among the
    decision's eligible members it is the one with the highest adverse total
    net return *after* subtracting the pooled adverse `uncharged_final_exit_cost`
    -- the exit a slot book never paid inside any fold window, which a holdout
    read must not let a member keep -- ties broken by the decision's member
    order. A cohort family states no such cost and contributes zero.

    Refuses a decision that lists no eligible member: there is then no release
    candidate, and a holdout is opened for a candidate or not at all.
    """
    eligible = [
        _text(name, "eligible_member_names entry")
        for name in _sequence(decision.get("eligible_member_names"), "eligible_member_names")
    ]
    if not eligible:
        raise CarryHoldoutError("the decision lists no eligible member")
    members = {
        _text(member.get("candidate_name"), "decision member name"): member
        for member in _mappings(decision.get("members"), "decision members")
    }
    scored: list[tuple[str, Decimal]] = []
    for name in eligible:
        member = members.get(name)
        if member is None:
            raise CarryHoldoutError(f"the decision has no member record for {name}")
        adverse_total = _decimal(
            member.get("adverse_total_net_return"), f"{name} adverse_total_net_return"
        )
        scored.append((name, adverse_total - _pooled_uncharged_exit_cost(name, fold_reports)))
    # `max` keeps the first of equal values, which is the declared tie-break:
    # the decision's own member order.
    return max(scored, key=lambda item: item[1])


def run_carry_holdout(
    perp_capture_root: Path,
    spot_capture_root: Path,
    *,
    original_perp_capture_root: Path,
    original_spot_capture_root: Path,
    manifest_path: Path,
    family_spec_path: Path,
    decision_path: Path,
    fold_report_paths: tuple[Path, ...],
    output_path: Path,
    registry_path: Path,
    check_only: bool = False,
) -> CarryHoldoutArtifact:
    """Open one carry family's final holdout, once (protocol 16.2).

    `perp_capture_root`/`spot_capture_root` are the *extended* captures the
    bars and funding are read from; the calendar, the holdout membership and
    the family linkage all come from the original manifest and the original
    captures, which the extended ones must be verified supersets of.

    `check_only` is the dry run: every input check runs, the candidate is
    derived and the bars are loaded far enough to prove the last exit is
    priced, and then the call stops -- nothing is evaluated, nothing is written
    and no artifact is registered, so the declaration's one read is still
    unspent. It is the only way to learn whether a read would refuse without
    spending the read on finding out.
    """
    if output_path.exists():
        raise CarryHoldoutError("carry holdout report already exists and is immutable")
    verified = _verify_inputs(
        perp_capture_root,
        spot_capture_root,
        original_perp_capture_root=original_perp_capture_root,
        original_spot_capture_root=original_spot_capture_root,
        manifest_path=manifest_path,
        family_spec_path=family_spec_path,
        decision_path=decision_path,
        fold_report_paths=fold_report_paths,
        registry_path=registry_path,
    )
    spec = verified.spec
    candidate_name, _ = derive_candidate(verified.decision, list(verified.fold_reports))

    decisions = sorted(_decision_close_ns(sample_id) for sample_id in verified.holdout_ids)
    # Spec section 4: the last holdout exit is readable and nothing after it
    # is, so the extra months the extended capture adds cannot leak into the
    # read beyond the exit they exist for. A bar becomes available one
    # nanosecond after it closes (`panel_capture`), and `load_panel_bars`
    # admits `available_time_ns < available_before_ns`, so the boundary is the
    # exit bar's availability plus one: the exit itself is in, the next day is
    # out. One nanosecond lower would drop the very bar the extended capture
    # was taken for, and every last holdout episode would force-close instead
    # of exiting at its price.
    last_exit_close_ns = decisions[-1] + spec.holding_days * DAY_NS
    available_before_ns = last_exit_close_ns + 2
    perp_histories = build_contract_histories(
        load_panel_bars(perp_capture_root / "dataset", available_before_ns=available_before_ns)
    )
    spot_histories = build_contract_histories(
        load_panel_bars(spot_capture_root / "dataset", available_before_ns=available_before_ns)
    )
    leg_histories: dict[str, ContractHistory] = {}
    for cid, history in perp_histories.items():
        leg_histories[f"perp:{cid}"] = history
    for cid, history in spot_histories.items():
        leg_histories[f"spot:{cid}"] = history
    # The month check proved the calendar; this proves the bar. A capture can
    # reach the exit month and still hold no bar at the exit close itself -- a
    # dump that ends mid-month, a repair that dropped its last day -- and every
    # last holdout episode would then be force-closed at a stale price instead
    # of exiting at its own, which is the one thing the extended capture exists
    # to prevent.
    _require_exit_bar(perp_histories, close_ns=last_exit_close_ns, label="perpetual")
    _require_exit_bar(spot_histories, close_ns=last_exit_close_ns, label="spot")
    if check_only:
        return CarryHoldoutArtifact(
            output_path=output_path,
            report_hash=None,
            candidate_name=candidate_name,
            verdict=HOLDOUT_CHECKED_VERDICT,
        )

    funding_by_leg: dict[str, tuple[FundingEvent, ...]] = {}
    for event in load_funding_events(perp_capture_root / "dataset"):
        leg_key = f"perp:{event.contract_id}"
        funding_by_leg[leg_key] = (*funding_by_leg.get(leg_key, ()), event)

    try:
        run = evaluate_carry_decisions(
            spec,
            perp_histories=perp_histories, spot_histories=spot_histories,
            leg_histories=leg_histories, funding_by_leg=funding_by_leg,
            decisions=decisions,
            candidate_names=(candidate_name, *DOMINANCE_CONTROL_NAMES),
        )
    except CarryFoldError as error:
        # A holdout read refuses as a holdout read: a caller that fails closed
        # on `CarryHoldoutError` -- which is every refusal in this module --
        # would otherwise see the fold runner's own type for a declaration that
        # does not carry the derived candidate or one of the two controls.
        raise CarryHoldoutError(f"the holdout evaluation refused: {error}") from error
    confirmation = _confirmation(
        run,
        candidate_name=candidate_name,
        decision_member=_decision_member(verified.decision, candidate_name),
    )
    material: dict[str, object] = {
        "report_version": "1.0.0",
        "status": "development_only",
        "holdout": True,
        "reason_codes": run.reason_codes,
        "family_name": spec.family_name,
        "family_spec_hash": verified.family_spec_hash,
        "candidate_name": candidate_name,
        "decision_report_hash": verified.decision_report_hash,
        **verified.capture_hashes,
        "split_manifest_hash": _text(
            verified.manifest.get("split_manifest_hash"), "split_manifest_hash"
        ),
        "manifest_hash": _text(verified.manifest.get("manifest_hash"), "manifest_hash"),
        "holdout_sample_count": len(verified.holdout_ids),
        "holdout_membership_hash": content_sha256(list(verified.holdout_ids)),
        "random_seed": spec.statistics.random_seed,
        "block_length": spec.statistics.block_length,
        "bootstrap_repetitions": spec.statistics.bootstrap_repetitions,
        "skipped_sample_ids": run.skipped_sample_ids,
        "warm_up_weeks": run.warm_up_weeks,
        "code_hash": _code_hash(),
        "candidates": run.candidates,
        "confirmation": confirmation,
    }
    report_hash = _seal(material, output_path)
    try:
        _register(
            verified.artifact_id,
            output_path=output_path, report_hash=report_hash, registry_path=registry_path,
        )
    except Exception as error:
        # The report and the registry record are one artifact (spec section 6):
        # a report nothing recorded would be immutable at a path no retry could
        # reuse, while the family still reads as unopened -- the one state that
        # can neither be used nor cleared. This report was written by this call
        # alone (the pre-check proved the family had no holdout record), so
        # removing it puts the family back exactly where it was.
        output_path.unlink(missing_ok=True)
        raise CarryHoldoutError(
            f"holdout artifact could not be registered: {type(error).__name__}: {error}"
        ) from error
    return CarryHoldoutArtifact(
        output_path=output_path,
        report_hash=report_hash,
        candidate_name=candidate_name,
        verdict=_text(confirmation["verdict"], "verdict"),
    )


def _verify_inputs(
    perp_capture_root: Path,
    spot_capture_root: Path,
    *,
    original_perp_capture_root: Path,
    original_spot_capture_root: Path,
    manifest_path: Path,
    family_spec_path: Path,
    decision_path: Path,
    fold_report_paths: tuple[Path, ...],
    registry_path: Path,
) -> _VerifiedInputs:
    """Every check the read must pass before a single bar is loaded.

    In order: both extended captures verify; both original captures verify --
    they are evidence too, and everything below reads their manifests as
    authority; this declaration has no holdout artifact yet, which is checked
    before any manifest work because the key is the declaration alone (spec
    section 6); the manifest verifies and was published under this declaration;
    the originals handed in are the very captures that manifest was published
    from; each extended capture is a superset of its original (spec section 2);
    the decision's seal recomputes, it was made under this declaration and on
    this manifest, and it names exactly these fold reports, in order; every fold
    report verifies and was itself run on this manifest; and the extended
    captures reach the calendar month the holdout's last episode exits in, in
    every kind of data they carry. Nothing is written before all of them hold.
    """
    for root, label in ((perp_capture_root, "perpetual"), (spot_capture_root, "spot")):
        valid, errors = verify_panel_capture(root)
        if not valid:
            raise CarryHoldoutError(
                f"extended {label} capture verification failed: " + ",".join(errors)
            )
    # The originals are not merely a lineage baseline: their manifests state
    # the four hashes the manifest links are checked against, the rows the
    # superset check compares and the two `original_*` hashes the artifact
    # binds. An unverified original would let all three agree on bytes the
    # capture itself no longer holds.
    for root, label in (
        (original_perp_capture_root, "perpetual"), (original_spot_capture_root, "spot")
    ):
        valid, errors = verify_panel_capture(root)
        if not valid:
            raise CarryHoldoutError(
                f"original {label} capture verification failed: " + ",".join(errors)
            )

    spec, family_spec_hash = load_carry_family_spec(family_spec_path)
    # Spec section 6 under controller Ruling 9: the single-use key is the
    # *declaration* -- the family's name and the hash of its spec -- and not the
    # walk-forward manifest the read runs on. A capture repair republishes the
    # same family on a new manifest, and a key that carried the manifest hash
    # would hand that republished family a second holdout read, which is the
    # one thing a single-use artifact exists to prevent. Checked here, before
    # any manifest is read, because nothing in the key depends on one.
    artifact_id = uuid5(
        NAMESPACE_URL, f"{HOLDOUT_ARTIFACT_KIND}:{spec.family_name}:{family_spec_hash}"
    )
    with MetadataRegistry(registry_path) as registry:
        if registry.get_artifact(artifact_id) is not None:
            raise CarryHoldoutError("this family's holdout has already been read")

    if not verify_panel_manifest(manifest_path):
        raise CarryHoldoutError("panel manifest verification failed")
    manifest = _load_object(manifest_path)
    if manifest.get("family_spec_hash") != family_spec_hash:
        raise CarryHoldoutError("family declaration does not match the manifest")
    split_manifest_hash = _text(manifest.get("split_manifest_hash"), "split_manifest_hash")

    # The lineage check below only proves the extended capture contains the
    # original; it cannot tell whether that original is the capture the
    # manifest -- and so the whole family -- was built on. A capture that is
    # merely a subset of the extended one (the extended capture itself, say)
    # would pass every superset row while binding the report to a lineage the
    # decision never ran on, so the four links are required first, exactly as
    # `carry_fold_run._require_link` requires them of a fold.
    original_hashes = {
        "capture_root_hash": _capture_root_hash(original_perp_capture_root),
        "dataset_root_hash": _dataset_root_hash(original_perp_capture_root),
        "hedge_capture_root_hash": _capture_root_hash(original_spot_capture_root),
        "hedge_dataset_root_hash": _dataset_root_hash(original_spot_capture_root),
    }
    for key, expected in original_hashes.items():
        if manifest.get(key) != expected:
            raise CarryHoldoutError(f"manifest is not linked to the original capture ({key})")

    for original, extended, label in (
        (original_perp_capture_root, perp_capture_root, "perpetual"),
        (original_spot_capture_root, spot_capture_root, "spot"),
    ):
        superset, reasons = verify_capture_superset(original, extended)
        if not superset:
            raise CarryHoldoutError(
                f"extended {label} capture is not a superset of the original: "
                + ",".join(reasons)
            )

    decision = _load_object(decision_path)
    decision_report_hash = _text(decision.get("report_hash"), "decision report_hash")
    material = {key: value for key, value in decision.items() if key != "report_hash"}
    if content_sha256(material) != decision_report_hash:
        raise CarryHoldoutError("decision verification failed")
    if decision.get("family_spec_hash") != family_spec_hash:
        raise CarryHoldoutError("family declaration does not match the decision")
    # The declaration alone does not pin the walk-forward: a repair republishes
    # the same family on a new manifest, and a decision pooled under the older
    # one would otherwise pass every check here while this read takes its
    # calendar, its holdout membership and its single-use artifact id from the
    # newer one. The split manifest hash is what says which walk-forward the
    # decision was actually made on.
    if decision.get("split_manifest_hash") != split_manifest_hash:
        raise CarryHoldoutError("decision is not linked to this manifest (split_manifest_hash)")
    fold_reports = tuple(_load_object(path) for path in fold_report_paths)
    named = [
        _text(value, "source_report_hashes entry")
        for value in _sequence(decision.get("source_report_hashes"), "source_report_hashes")
    ]
    if named != [
        _text(document.get("report_hash"), "fold report_hash") for document in fold_reports
    ]:
        raise CarryHoldoutError("the decision does not name these fold reports in this order")
    for path, document in zip(fold_report_paths, fold_reports, strict=True):
        if not verify_panel_fold_report(path):
            raise CarryHoldoutError(f"fold report failed verification: {path.name}")
        # Same reasoning one level down: every fold the decision pooled must
        # have been run on this manifest and on the captures it names, or the
        # holdout would be opened for a candidate derived from other windows.
        for key in _FOLD_MANIFEST_LINKS:
            if document.get(key) != manifest.get(key):
                raise CarryHoldoutError(
                    f"fold report {path.name} is not linked to this manifest ({key})"
                )

    holdout_ids = tuple(
        _text(value, "final_holdout_ids entry")
        for value in _sequence(manifest.get("final_holdout_ids"), "final_holdout_ids")
    )
    if not holdout_ids:
        raise CarryHoldoutError("the manifest names no final holdout decision")
    last_exit_ns = (
        max(_decision_close_ns(sample_id) for sample_id in holdout_ids)
        + spec.holding_days * DAY_NS
    )
    exit_month = datetime.fromtimestamp(last_exit_ns // 1_000_000_000, tz=UTC).strftime("%Y-%m")
    extended_perp = _load_object(perp_capture_root / "capture-manifest.json")
    extended_spot = _load_object(spot_capture_root / "capture-manifest.json")
    for capture_manifest in (extended_perp, extended_spot):
        _require_coverage(capture_manifest, exit_month=exit_month)

    return _VerifiedInputs(
        spec=spec,
        family_spec_hash=family_spec_hash,
        manifest=manifest,
        holdout_ids=holdout_ids,
        decision=decision,
        decision_report_hash=decision_report_hash,
        fold_reports=fold_reports,
        artifact_id=artifact_id,
        # Spec section 2: the artifact binds both the extended captures the
        # read ran on and the originals the family was decided on, so neither
        # side of the lineage has to be reconstructed from outside the report.
        capture_hashes={
            "capture_root_hash": _text(
                extended_perp.get("capture_root_hash"), "capture_root_hash"
            ),
            "dataset_root_hash": _dataset_root_hash(perp_capture_root),
            "hedge_capture_root_hash": _text(
                extended_spot.get("capture_root_hash"), "hedge capture_root_hash"
            ),
            "hedge_dataset_root_hash": _dataset_root_hash(spot_capture_root),
            "original_capture_root_hash": original_hashes["capture_root_hash"],
            "original_hedge_capture_root_hash": original_hashes["hedge_capture_root_hash"],
        },
    )


def _require_coverage(capture_manifest: dict[str, object], *, exit_month: str) -> None:
    """Refuse a capture that does not reach the holdout's last exit month."""
    covered = _coverage_month(capture_manifest)
    if covered is None or covered < exit_month:
        raise CarryHoldoutError("extended capture does not cover the holdout's last exit")


def _require_exit_bar(
    histories: dict[str, ContractHistory], *, close_ns: int, label: str
) -> None:
    """Refuse a capture that holds no bar at the holdout's last exit close.

    One bar in one contract is enough to prove the day is there: the capture
    is a single dump per symbol and month, so a market that priced the exit
    close for any of its contracts priced it for the universe. What this
    catches is the whole day missing -- and with it every last episode's exit.
    """
    if not any(close_ns in history.closes for history in histories.values()):
        raise CarryHoldoutError(
            f"extended capture holds no bar at the holdout's last exit close ({label})"
        )


def _coverage_month(capture_manifest: dict[str, object]) -> str | None:
    """The latest month (YYYY-MM) this capture covers in *every* kind, or `None`.

    Spec section 2: the holdout's last episode exits after the original capture
    ends, so the read refuses unless the extended capture reaches that month.
    A capture built by discovery -- which every production capture and every
    repair of one is -- states `months: null` and carries the per-symbol
    `discovered_months` (`symbol -> {kind -> [months]}`) instead. There the
    month is taken per kind: the union over symbols, because one delisted
    symbol's dump stopping early says nothing about the market, but never the
    union over kinds, because the September klines and the September funding
    dump arrive separately and a capture holding only the first would price the
    last exit bar while leaving the final week's settlements unpaid. So each
    kind's latest month is taken and the earliest of those is what the capture
    covers; a kind no symbol carries a single month for is skipped (a spot
    capture states the funding kind with an empty list). The explicit `months`
    list is the fallback for a capture built without discovery, which fetches
    one month set for every kind it carries. YYYY-MM sorts chronologically as
    text, so `max` and `min` are calendar extremes.
    """
    discovered = capture_manifest.get("discovered_months")
    if isinstance(discovered, dict):
        by_kind: dict[str, set[str]] = {}
        for value in discovered.values():
            if isinstance(value, dict):
                for kind, months_of_kind in value.items():
                    if isinstance(kind, str):
                        by_kind.setdefault(kind, set()).update(_month_strings(months_of_kind))
            else:
                by_kind.setdefault("", set()).update(_month_strings(value))
        reached = [max(months) for months in by_kind.values() if months]
        return min(reached) if reached else None
    months = _month_strings(capture_manifest.get("months"))
    return max(months) if months else None


def _month_strings(value: object) -> set[str]:
    """Every YYYY-MM string in a month list, or in a mapping of them by kind."""
    if isinstance(value, list):
        return {item for item in value if isinstance(item, str)}
    if isinstance(value, dict):
        return {
            item
            for entry in value.values()
            if isinstance(entry, list)
            for item in entry
            if isinstance(item, str)
        }
    return set()


def _confirmation(
    run: DecisionRun, *, candidate_name: str, decision_member: dict[str, object]
) -> dict[str, object]:
    """Spec section 5's fixed criteria, applied to the holdout alone.

    All eight booleans must hold for `holdout_confirmed`: a positive base
    total; an adverse total that survives subtracting the exit the book never
    paid (which is also the adverse half of dominance over `no_trade`, whose
    total is zero); a base total above both dominance controls and an adverse
    total after that subtraction at or above `random_pairs`; neither
    concentration share above the limit; and at most four decisions skipped.
    No statistical test is applied -- 26 weeks confirm a sign and a magnitude,
    they do not discover -- so the mean weekly net under both scenarios, the
    positive-week fraction, the pooled forced-close count and whether the
    holdout's base mean is at or above the decision's bootstrap lower bound for
    this candidate (a one-sided comparison, not an interval) are reported
    beside the verdict, never gated.
    """
    records = {
        _text(record.get("candidate_name"), "candidate_name"): record
        for record in run.candidates
    }
    candidate = _record(records, candidate_name)
    base = _mapping(candidate.get("base"), f"{candidate_name} base")
    adverse = _mapping(candidate.get("adverse"), f"{candidate_name} adverse")
    base_total = _decimal(base.get("total_net_return"), "base total_net_return")
    adverse_total = _decimal(adverse.get("total_net_return"), "adverse total_net_return")
    # A cohort family never leaves one; a slot family leaves its whole exit
    # outside the window, and the holdout may not keep what it did not pay.
    uncharged = (
        _decimal(adverse["uncharged_final_exit_cost"], "uncharged_final_exit_cost")
        if "uncharged_final_exit_cost" in adverse
        else Decimal(0)
    )
    after_exit = adverse_total - uncharged

    base_episodes = _mappings(base.get("episodes"), "base episodes")
    base_returns = tuple(
        _decimal(episode.get("net_return"), "net_return") for episode in base_episodes
    )
    contract_totals: dict[str, Decimal] = {}
    for episode in base_episodes:
        for entry in _sequence(
            episode.get("contract_net_contributions"), "contract_net_contributions"
        ):
            pair = _sequence(entry, "contract_net_contributions entry")
            if len(pair) != 2:
                raise CarryHoldoutError("contract_net_contributions entry is malformed")
            pair_id = _text(pair[0], "pair id")
            contract_totals[pair_id] = contract_totals.get(pair_id, Decimal(0)) + _decimal(
                pair[1], "pair contribution"
            )
    shares = concentration_shares(
        base_total=base_total,
        fold_base_totals=(),
        contract_totals=contract_totals,
        base_returns=base_returns,
    )
    # Zero where the base total is not positive: a share of such a total states
    # nothing, and the first criterion has already failed there.
    episode_share = shares.get("largest_episode_share", Decimal(0))
    pair_share = shares.get("largest_contract_share", Decimal(0))

    no_trade_base = _scenario_total(records, DOMINANCE_CONTROL_NAMES[0], "base")
    random_base = _scenario_total(records, DOMINANCE_CONTROL_NAMES[1], "base")
    random_adverse = _scenario_total(records, DOMINANCE_CONTROL_NAMES[1], "adverse")
    skipped = len(run.skipped_sample_ids)
    gates: dict[str, bool] = {
        "base_total_positive": base_total > 0,
        "adverse_after_uncharged_exit_non_negative": after_exit >= 0,
        "base_dominates_no_trade": base_total > no_trade_base,
        "base_dominates_random_pairs": base_total > random_base,
        "adverse_dominates_random_pairs": after_exit >= random_adverse,
        "largest_episode_share_within_limit": episode_share <= CONCENTRATION_LIMIT,
        "largest_pair_share_within_limit": pair_share <= CONCENTRATION_LIMIT,
        "skipped_decisions_within_limit": skipped <= MAX_SKIPPED_HOLDOUT_DECISIONS,
    }

    count = len(base_episodes)
    adverse_returns = tuple(
        _decimal(episode.get("net_return"), "net_return")
        for episode in _mappings(adverse.get("episodes"), "adverse episodes")
    )
    base_mean = base_total / Decimal(count) if count else Decimal(0)
    lower = (
        _decimal(decision_member["base_bootstrap_lower"], "base_bootstrap_lower")
        if decision_member.get("base_bootstrap_lower") is not None
        else None
    )
    reported: dict[str, object] = {
        "base_mean_weekly_net_return": base_mean,
        "adverse_mean_weekly_net_return": (
            sum(adverse_returns, Decimal(0)) / Decimal(len(adverse_returns))
            if adverse_returns
            else Decimal(0)
        ),
        "positive_week_fraction": (
            Decimal(sum(1 for value in base_returns if value > 0)) / Decimal(count)
            if count
            else Decimal(0)
        ),
        # Pooled over the candidate's holdout base episodes. A verdict cannot
        # show how much of the window the book was carried out of rather than
        # exited from, and a holdout whose episodes force-closed is a holdout
        # read on data that ran out under it -- reported, never gated, because
        # the coverage checks are what refuse that case.
        "candidate_forced_close_count": sum(
            _count(episode.get("forced_close_count"), "forced_close_count")
            for episode in base_episodes
        ),
        "decision_base_mean_net_return": (
            _decimal(decision_member["base_mean_net_return"], "base_mean_net_return")
            if decision_member.get("base_mean_net_return") is not None
            else None
        ),
        "decision_base_bootstrap_lower": lower,
        "base_mean_at_or_above_decision_bootstrap_lower": (
            None if lower is None else base_mean >= lower
        ),
    }
    return {
        "verdict": "holdout_confirmed" if all(gates.values()) else "holdout_failed",
        "base_total_net_return": base_total,
        "adverse_total_net_return": adverse_total,
        "adverse_uncharged_final_exit_cost": uncharged,
        "adverse_total_after_uncharged_exit": after_exit,
        "criteria": {
            **gates,
            "largest_episode_share": episode_share,
            "largest_pair_share": pair_share,
            "skipped_decision_count": skipped,
        },
        "reported": reported,
    }


def _seal(material: dict[str, object], output_path: Path) -> str:
    """Hash the document over itself and publish it atomically."""
    report_hash = content_sha256(material)
    document = dict(material)
    document["report_hash"] = report_hash
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_bytes(canonical_json(document))
    temporary.replace(output_path)
    return report_hash


def _register(
    artifact_id: UUID, *, output_path: Path, report_hash: str, registry_path: Path
) -> None:
    """Record the single-use artifact, keyed by the declaration alone (spec 6).

    The id is `uuid5(NAMESPACE_URL, "holdout:<family name>:<family spec hash>")`,
    which carries nothing of the walk-forward manifest: a republished manifest
    for the same declaration finds the same record and refuses.
    """
    with MetadataRegistry(registry_path) as registry:
        registry.register_artifact(
            ArtifactRecord(
                artifact_id=artifact_id,
                kind=HOLDOUT_ARTIFACT_KIND,
                relative_path=output_path.name,
                content_hash=report_hash,
                created_at_ns=time.time_ns(),
            )
        )


def _pooled_uncharged_exit_cost(name: str, fold_reports: list[dict[str, object]]) -> Decimal:
    total = Decimal(0)
    for document in fold_reports:
        record = next(
            (
                item
                for item in _mappings(document.get("candidates"), "fold report candidates")
                if item.get("candidate_name") == name
            ),
            None,
        )
        if record is None:
            raise CarryHoldoutError(f"a fold report has no candidate record for {name}")
        adverse = _mapping(record.get("adverse"), f"{name} adverse")
        if "uncharged_final_exit_cost" in adverse:
            total += _decimal(
                adverse["uncharged_final_exit_cost"], "uncharged_final_exit_cost"
            )
    return total


def _decision_member(decision: dict[str, object], name: str) -> dict[str, object]:
    for member in _mappings(decision.get("members"), "decision members"):
        if member.get("candidate_name") == name:
            return member
    raise CarryHoldoutError(f"the decision has no member record for {name}")


def _scenario_total(
    records: dict[str, dict[str, object]], name: str, scenario: str
) -> Decimal:
    record = _record(records, name)
    return _decimal(
        _mapping(record.get(scenario), f"{name} {scenario}").get("total_net_return"),
        f"{name} {scenario} total_net_return",
    )


def _record(records: dict[str, dict[str, object]], name: str) -> dict[str, object]:
    record = records.get(name)
    if record is None:
        raise CarryHoldoutError(f"the holdout evaluation produced no record for {name}")
    return record


def _decision_close_ns(sample_id: str) -> int:
    """The decision close time a `VENUE:<close_ns>:w1` sample id names."""
    parts = sample_id.split(":")
    if len(parts) != 3 or not parts[1].isdigit():
        raise CarryHoldoutError(f"malformed holdout sample id: {sample_id}")
    return int(parts[1])


def _capture_root_hash(capture_root: Path) -> str:
    return _text(
        _load_object(capture_root / "capture-manifest.json").get("capture_root_hash"),
        f"{capture_root.name} capture_root_hash",
    )


def _dataset_root_hash(capture_root: Path) -> str:
    return _text(
        _load_object(capture_root / "dataset" / "dataset-manifest.json").get("root_hash"),
        f"{capture_root.name} dataset root_hash",
    )


def _code_hash() -> str:
    root = Path(__file__).parent
    material = "".join((root / name).read_text(encoding="utf-8") for name in _HOLDOUT_MODULES)
    return content_sha256(material)


def _load_object(path: Path) -> dict[str, object]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CarryHoldoutError(f"{path.name} could not be read: {error}") from error
    if not isinstance(document, dict):
        raise CarryHoldoutError(f"{path.name} must contain a JSON object")
    return document


def _mapping(value: object, what: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CarryHoldoutError(f"{what} must be an object")
    return value


def _mappings(value: object, what: str) -> list[dict[str, object]]:
    return [_mapping(item, f"{what} entry") for item in _sequence(value, what)]


def _sequence(value: object, what: str) -> list[object]:
    if not isinstance(value, list):
        raise CarryHoldoutError(f"{what} must be a list")
    return value


def _count(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CarryHoldoutError(f"{what} is not a count: {value!r}")
    return value


def _text(value: object, what: str) -> str:
    if not isinstance(value, str):
        raise CarryHoldoutError(f"{what} must be a string")
    return value


def _decimal(value: object, what: str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, str | int) and not isinstance(value, bool):
        try:
            return Decimal(value)
        except InvalidOperation as error:
            raise CarryHoldoutError(f"{what} is not a decimal: {value!r}") from error
    raise CarryHoldoutError(f"{what} is not a decimal: {value!r}")
