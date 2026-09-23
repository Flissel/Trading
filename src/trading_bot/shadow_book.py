"""The weekly shadow book: one Sunday's forecast book, sealed (spec section 4).

A shadow week is the fold runner's own decision loop, run once over every
Sunday from the declaration's anchor to the Sunday asked about, on the weekly
captures that reach that Sunday. The last decision's book is the book the
strategy *would* hold; nothing is persisted between weeks, so any week is
recomputable from data alone. The artifact around it states what it was
computed from -- both markets' capture and dataset hashes, the base each
weekly capture extends, the declaration, the code, the freshest input time --
and what it says: the book, the week's measurements for every pair the
universe ranked, the `m` labels a later abstention family will learn from,
and, in Phase B only, the previous Sunday's P&L and the running totals since
the anchor.

Phase A (spec section 2) exists so that the pipeline can run on real data
while the P1.33 holdout is unread, without any post-holdout number reaching
the go/no-go decision. A Phase A artifact is `status: development_only` and
carries no P&L block and no running totals at all -- not a redacted one, not a
summary: the keys are absent.

Spec section 4.3: no order is generated here and no venue reached. This module
imports no execution adapter, no venue client and no credential path --
`execution`, `runtime`, `binance_importer` and the journals' venue clients are
all absent from its imports, and the only things it opens are capture
directories, three JSON documents and a SQLite registry.
"""

import json
import os
import shutil
import time
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.capture_lineage import verify_capture_superset
from trading_bot.carry_config import (
    CarryCostTable,
    CarryFamilySpec,
    CarryMember,
    load_carry_family_spec,
)
from trading_bot.carry_fold_run import (
    CarryFoldError,
    DecisionInputs,
    DecisionRun,
    FinalBook,
    _trailing_by_pair,
    carry_module_names,
    evaluate_carry_decisions,
    load_decision_inputs,
)
from trading_bot.carry_signals import (
    Cohort,
    entries_for,
    exit_rule_pairs,
    round_trip_cost_bps,
    trailing_funding,
)
from trading_bot.carry_universe import PairUniverseSnapshot, select_pair_universe
from trading_bot.panel_capture import verify_panel_capture
from trading_bot.panel_reader import FundingEvent
from trading_bot.panel_samples import rebalance_close_times
from trading_bot.registry import ArtifactRecord, MetadataRegistry, RegistryConflictError
from trading_bot.shadow_capture import DAILY_TAIL_KIND, FUNDING_REST_KIND
from trading_bot.shadow_config import (
    ShadowDeclaration,
    ShadowDeclarationError,
    load_shadow_declaration,
)
from trading_bot.storage import StoragePolicy, StoragePolicyError

REPORT_VERSION = "1.0.0"
SHADOW_ARTIFACT_KIND = "shadow_week"
# The measurement snapshot document a week may cite (Ruling 1a). Named here
# rather than imported from `binance_measurement_journal` so that computing a
# book never loads the venue-facing stream module; `test_shadow_book` pins the
# two strings against each other so they cannot drift apart unnoticed.
MEASUREMENT_SNAPSHOT_VERSION = "binance-measurement-snapshot/1.0.0"

# Spec 4.2's code hash: the carry evaluation path plus the three modules only
# a shadow week runs. `carry_module_names()` is the fold runner's own list, so
# the two cannot diverge into a second, silently stale copy.
_SHADOW_MODULES = ("shadow_book.py", "shadow_capture.py", "shadow_config.py")
_STATUS_BY_PHASE = {"A": "development_only", "B": "shadow"}
_DAY_NS = 86_400_000_000_000
_NANOSECONDS_PER_MILLISECOND = 1_000_000
_SUNDAY = 6  # `date.weekday()` counts from Monday
_EPOCH = date(1970, 1, 1)
_BPS = Decimal(10_000)
# The two source kinds a weekly capture adds to its base. They are the rows
# that were fetched for *this* week, so the newest of their `received_time_ns`
# is how fresh the freshest input of the week was (ruling 2).
_TAIL_KINDS = frozenset({DAILY_TAIL_KIND, FUNDING_REST_KIND})
# One sealed document plus a registry row; nothing here writes a payload.
_WORST_CASE_REQUIRED_BYTES = 10_000_000
_PERPETUAL_MEASUREMENT_KEYS = (
    "premium_rounds",
    "book_rounds",
    "mean_last_funding_rate",
    "last_funding_rate",
    "mean_basis_bps",
    "mean_spread_bps",
)
_SPOT_MEASUREMENT_KEYS = ("book_rounds", "mean_spread_bps")
# The five fields of a cited snapshot that say which rounds it is over
# (ruling 22). Recorded in the week under a `measurement_snapshot_` prefix.
_SNAPSHOT_WINDOW_KEYS = (
    "window_start_ns",
    "window_end_ns",
    "first_sequence",
    "last_sequence",
    "rounds",
)
# The week a decision closes: the seven days ending at the Sunday close.
_WEEK_NS = 7 * _DAY_NS
_SUNDAY_NOT_IN_CAPTURE = "SHADOW_SUNDAY_NOT_IN_CAPTURE"
_NO_SUNDAY_BAR = "SHADOW_BOOK_LEG_HAS_NO_SUNDAY_BAR"
_DECISION_RUN_REFUSED = "SHADOW_DECISION_RUN_REFUSED"
_ALREADY_REGISTERED = "SHADOW_WEEK_ALREADY_REGISTERED"
_NOT_REGISTERED = "SHADOW_WEEK_NOT_REGISTERED"
_SNAPSHOT_NOT_THIS_WEEK = "SHADOW_SNAPSHOT_NOT_THIS_WEEK"


class ShadowBookError(RuntimeError):
    """Raised when a shadow week cannot be computed or published."""


class _Refusal(ShadowBookError):
    """A refusal that happened after the inputs verified, so it is recorded.

    Spec section 6: a missing week is visible, not silent. Everything raised
    from the point the inputs are known good carries the reason code the
    refusal document is written with, so the document and the exception say
    the same thing rather than two paraphrases of it.
    """

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


@dataclass(frozen=True, slots=True)
class ShadowWeekArtifact:
    output_path: Path
    report_hash: str
    status: str


@dataclass(frozen=True, slots=True)
class _Capture:
    """One verified capture of one market, with what the artifact cites of it."""

    root: Path
    capture_root_hash: str
    dataset_root_hash: str
    base_capture_root_hash: str | None
    data_available_time_ns: int


def run_shadow_week(
    *,
    workspace_root: Path,
    declaration_path: Path,
    perp_capture_root: Path,
    spot_capture_root: Path,
    decision_sunday: str,
    holdout_report_path: Path | None,
    measurement_snapshot_path: Path | None,
    perp_base_capture_root: Path | None = None,
    spot_base_capture_root: Path | None = None,
    reserve_bytes: int = 0,
) -> ShadowWeekArtifact:
    """Compute, seal and register one Sunday's shadow book.

    The order is deliberate. Everything that can be checked without reading a
    bar is checked first -- the declaration, the storage policy, whether this
    week is already published, both captures' seals and lineage, the family
    spec, the phase's holdout, the cited snapshot -- and a refusal there raises
    and writes nothing, because a week whose inputs are not trustworthy has no
    business filing a document about itself. From the moment the inputs are
    known good, every refusal is recorded beside the week that was not written
    (spec section 6) and then raised.

    `perp_base_capture_root` and `spot_base_capture_root` are required exactly
    when the matching weekly capture records a `base_capture_root_hash`
    (ruling 17): the lineage check needs the base's own manifest, which only
    the caller has. A monthly capture extends nothing and is run without one.

    A week is published and then registered, and a crash between the two
    leaves an artifact no registry row points at; `_register`'s docstring has
    the operator step that clears it.
    """
    declaration, declaration_hash = _declaration(declaration_path)
    sunday = _sunday(decision_sunday)
    decision_close_ns = _close_of(sunday)
    anchor_close_ns = _close_of(_date(declaration.anchor_decision_close_date))

    workspace = workspace_root.resolve()
    artifact_root = workspace / declaration.artifact_root
    output_path = artifact_root / declaration.family_name / f"{decision_sunday}.json"
    refusal_path = output_path.with_name(f"{decision_sunday}-refused.json")
    registry_path = workspace / declaration.registry_path
    _authorize(workspace, reserve_bytes, output_path, registry_path)
    if output_path.exists():
        raise ShadowBookError(
            f"shadow week {decision_sunday} already exists and is immutable"
        )

    perp = _capture(
        perp_capture_root, base_root=perp_base_capture_root, market="um",
        label="perpetual", decision_sunday=decision_sunday,
    )
    spot = _capture(
        spot_capture_root, base_root=spot_base_capture_root, market="spot",
        label="spot", decision_sunday=decision_sunday,
    )
    spec, member = _family(declaration, workspace)
    _verify_phase(declaration, holdout_report_path)
    snapshot_document, measurement_snapshot_hash = _measurement_snapshot(
        measurement_snapshot_path
    )

    try:
        return _publish_week(
            declaration=declaration,
            declaration_hash=declaration_hash,
            spec=spec,
            member=member,
            perp=perp,
            spot=spot,
            workspace=workspace,
            output_path=output_path,
            registry_path=registry_path,
            decision_sunday=decision_sunday,
            decision_close_ns=decision_close_ns,
            anchor_close_ns=anchor_close_ns,
            snapshot_document=snapshot_document,
            measurement_snapshot_hash=measurement_snapshot_hash,
        )
    except _Refusal as error:
        _write_refusal(
            refusal_path,
            reason=error.reason,
            detail=str(error),
            declaration=declaration,
            declaration_hash=declaration_hash,
            decision_sunday=decision_sunday,
            decision_close_ns=decision_close_ns,
        )
        raise


def _publish_week(
    *,
    declaration: ShadowDeclaration,
    declaration_hash: str,
    spec: CarryFamilySpec,
    member: CarryMember,
    perp: _Capture,
    spot: _Capture,
    workspace: Path,
    output_path: Path,
    registry_path: Path,
    decision_sunday: str,
    decision_close_ns: int,
    anchor_close_ns: int,
    snapshot_document: dict[str, Any] | None,
    measurement_snapshot_hash: str | None,
) -> ShadowWeekArtifact:
    """Everything after the inputs verified: the runs, the blocks, the seal."""
    _require_this_weeks_snapshot(snapshot_document, decision_close_ns=decision_close_ns)
    inputs, decisions = _load_inputs(
        perp.root, spot.root,
        anchor_close_ns=anchor_close_ns, decision_close_ns=decision_close_ns,
    )
    if decision_close_ns not in decisions:
        raise _Refusal(
            _SUNDAY_NOT_IN_CAPTURE,
            f"{decision_sunday} is not among the decisions from "
            f"{declaration.anchor_decision_close_date} on that these captures carry",
        )
    candidates = (declaration.candidate, *declaration.controls)
    run = _evaluate(spec, inputs, decisions, candidates)
    book = run.final_books[declaration.candidate]
    _require_sunday_bars(book, inputs=inputs, decision_close_ns=decision_close_ns)

    # Ruling 18: the book of S-1 is a second run of the same cheap loop over
    # the same loaded inputs, because `final_books` only ever holds the last
    # decision's. Only the candidate is evaluated there -- a filtered run is
    # the full run's record for the candidates it names, so its siblings would
    # change nothing about the book this reads.
    previous_close_ns = decisions[-2] if len(decisions) > 1 else None
    previous_book = (
        None
        if previous_close_ns is None
        else _evaluate(
            spec, inputs, decisions[:-1], (declaration.candidate,)
        ).final_books[declaration.candidate]
    )

    universe = select_pair_universe(
        inputs.perp_histories, inputs.spot_histories,
        pairs=spec.pairs, decision_close_ns=decision_close_ns, rules=spec.universe,
    )
    material: dict[str, object] = {
        "report_version": REPORT_VERSION,
        "status": _STATUS_BY_PHASE[declaration.phase],
        "family_name": spec.family_name,
        "family_spec_hash": declaration.family_spec_hash,
        "declaration_hash": declaration_hash,
        "candidate_name": declaration.candidate,
        "holdout_report_hash": declaration.holdout_report_hash,
        "capture_root_hash": perp.capture_root_hash,
        "dataset_root_hash": perp.dataset_root_hash,
        "base_capture_root_hash": perp.base_capture_root_hash,
        "hedge_capture_root_hash": spot.capture_root_hash,
        "hedge_dataset_root_hash": spot.dataset_root_hash,
        "hedge_base_capture_root_hash": spot.base_capture_root_hash,
        "decision_sunday": decision_sunday,
        "decision_close_ns": decision_close_ns,
        "anchor_decision_close_ns": anchor_close_ns,
        "previous_decision_close_ns": previous_close_ns,
        "data_available_time_ns": max(
            perp.data_available_time_ns, spot.data_available_time_ns
        ),
        "code_hash": _code_hash(),
        "measurement_snapshot_hash": measurement_snapshot_hash,
        # Which rounds that reading is over (ruling 22). A hash says which
        # document was cited; these say what it covers, so a reader of the
        # week never has to open the snapshot to find out.
        **_snapshot_window_block(snapshot_document),
        # Ruling 16: a Sunday the runner skipped holds nothing, and the week
        # says so outright rather than publishing a flat book that would read
        # as a deliberate decision to hold nothing.
        "skipped": _was_skipped(run, decision_close_ns),
        "reason_codes": list(run.reason_codes),
        "skipped_sample_ids": list(run.skipped_sample_ids),
        "book": _book_block(book),
        "universe_measurements": _universe_block(
            universe,
            inputs=inputs,
            snapshot_document=snapshot_document,
        ),
        "m_labels": _m_label_block(
            previous_book,
            funding_by_leg=inputs.funding_by_leg,
            decision_close_ns=decision_close_ns,
            cost_table=spec.costs.base,
            hold_weeks=member.hold_weeks,
        ),
    }
    if declaration.phase == "B":
        material["pnl"] = _pnl_block(
            run, candidate=declaration.candidate, decision_close_ns=previous_close_ns
        )
        material["running_totals"] = _running_totals_block(
            run, candidate=declaration.candidate, decisions=decisions
        )

    report_hash = _publish(output_path, material)
    _register(
        output_path,
        workspace=workspace,
        registry_path=registry_path,
        family_spec_hash=declaration.family_spec_hash,
        decision_sunday=decision_sunday,
        report_hash=report_hash,
    )
    return ShadowWeekArtifact(
        output_path=output_path,
        report_hash=report_hash,
        status=str(material["status"]),
    )


# --- the inputs ---------------------------------------------------------


def _declaration(path: Path) -> tuple[ShadowDeclaration, str]:
    try:
        return load_shadow_declaration(path)
    except ShadowDeclarationError as error:
        raise ShadowBookError(str(error)) from error


def _authorize(workspace: Path, reserve_bytes: int, *targets: Path) -> None:
    """Both writes a week makes are inside the workspace and over the reserve.

    The storage policy is the same boundary `shadow_capture` authorizes its
    capture against: the artifact and the registry are separate files and are
    each authorized, because a declaration may put the registry somewhere the
    artifact root does not cover.
    """
    policy = StoragePolicy(workspace, reserve_bytes)
    try:
        free_bytes = shutil.disk_usage(workspace).free
        for target in targets:
            policy.authorize(
                target=target,
                temporary_directory=target.parent,
                free_bytes=free_bytes,
                worst_case_required_bytes=_WORST_CASE_REQUIRED_BYTES,
            )
    except (StoragePolicyError, OSError) as error:
        raise ShadowBookError(f"shadow week is not authorized to write: {error}") from error


def _capture(
    root: Path,
    *,
    base_root: Path | None,
    market: str,
    label: str,
    decision_sunday: str,
) -> _Capture:
    """Verify one market's capture and read what the artifact cites of it.

    Spec section 6's first three refusals, in order: the capture must verify,
    it must be the market it is being used as -- the perpetual leg carries the
    funding and the spot leg is the hedge, so a swapped pair would evaluate a
    book that is not the position under test -- it must cover the Sunday, and
    it must be a verified superset of the base it records (ruling 17).
    """
    valid, reasons = verify_panel_capture(root)
    if not valid:
        raise ShadowBookError(
            f"{label} capture verification failed: " + ",".join(reasons)
        )
    manifest = _object(root / "capture-manifest.json")
    # P1.27's captures predate the `market` key, so its absence means "um".
    recorded_market = str(manifest.get("market", "um"))
    if recorded_market != market:
        raise ShadowBookError(
            f"the {label} capture is a {recorded_market} market capture, not {market}"
        )
    if not _covers(manifest, decision_sunday):
        raise ShadowBookError(
            f"the {label} capture does not cover {decision_sunday}"
        )
    recorded_base = manifest.get("base_capture_root_hash")
    base_capture_root_hash = None if recorded_base is None else str(recorded_base)
    _verify_base(root, base_root=base_root, recorded=base_capture_root_hash, label=label)
    dataset = _object(root / "dataset" / "dataset-manifest.json")
    return _Capture(
        root=root,
        capture_root_hash=_hash_of(manifest, "capture_root_hash", label=label),
        dataset_root_hash=_hash_of(dataset, "root_hash", label=f"{label} dataset"),
        base_capture_root_hash=base_capture_root_hash,
        data_available_time_ns=_data_available_time_ns(manifest, label=label),
    )


def _hash_of(manifest: dict[str, Any], key: str, *, label: str) -> str:
    """One hash a manifest is expected to carry, named where it is missing."""
    value = manifest.get(key)
    if not isinstance(value, str):
        raise ShadowBookError(f"the {label} manifest carries no {key}")
    return value


def _covers(manifest: dict[str, Any], decision_sunday: str) -> bool:
    """Whether this capture speaks for the Sunday at all.

    A weekly capture says so with `tail_through`, the last calendar day it
    covers; a monthly capture says so with the months it captured. ISO dates
    and months compare correctly as strings, which is why both are stored that
    way. A capture that says neither cannot be read as covering anything.
    """
    tail_through = manifest.get("tail_through")
    if isinstance(tail_through, str):
        return tail_through >= decision_sunday
    months = manifest.get("months")
    if isinstance(months, list):
        return decision_sunday[:7] in {str(month) for month in months}
    return False


def _verify_base(
    root: Path, *, base_root: Path | None, recorded: str | None, label: str
) -> None:
    """Ruling 17: the base a weekly capture records is checked, or refused for."""
    if recorded is None:
        if base_root is not None:
            raise ShadowBookError(
                f"the {label} capture records no base capture, so none can be "
                "checked against it"
            )
        return
    if base_root is None:
        raise ShadowBookError(
            f"the {label} capture extends a base capture, which must be given "
            "so its lineage can be verified"
        )
    base_manifest = _object(base_root / "capture-manifest.json")
    if str(base_manifest.get("capture_root_hash")) != recorded:
        raise ShadowBookError(
            f"the {label} base capture is not the one the weekly capture records"
        )
    valid, reasons = verify_capture_superset(base_root, root)
    if not valid:
        raise ShadowBookError(
            f"the {label} capture is not a superset of its base: " + ",".join(reasons)
        )


def _data_available_time_ns(manifest: dict[str, Any], *, label: str) -> int:
    """The newest `received_time_ns` this capture carries (ruling 2).

    Spec 3.3 records the time a payload was fetched per row, so the week can
    say how stale its freshest input was. A weekly capture's tail rows are the
    ones fetched for this week, so they are what is measured; a monthly
    capture has no tail, and then its own rows are all it has to speak with.
    """
    sources = manifest.get("sources")
    rows_on_disk = sources if isinstance(sources, list) else []
    present = [
        row
        for row in rows_on_disk
        if isinstance(row, dict) and row.get("status") == "present"
    ]
    tail = [entry for entry in present if entry.get("kind") in _TAIL_KINDS]
    rows = tail or present
    times = [
        value
        for entry in rows
        for value in (entry.get("received_time_ns"),)
        if isinstance(value, int) and not isinstance(value, bool)
    ]
    if len(times) != len(rows) or not times:
        raise ShadowBookError(
            f"the {label} capture has a present source row without a received_time_ns"
        )
    return max(times)


def _family(
    declaration: ShadowDeclaration, workspace: Path
) -> tuple[CarryFamilySpec, CarryMember]:
    """The declared family spec and the one member the book is computed for.

    The hash is the gate spec section 6 names; the family name is ruling 15's
    declared reading of it, so a declaration says in words which family it
    shadows and is refused if the spec it loads is another one. Its path, like
    every path a declaration names, is relative to the workspace root -- which
    for the shipped declaration is the checkout the `configs/` directory is in.
    """
    try:
        spec, family_spec_hash = load_carry_family_spec(
            workspace / declaration.family_spec_path
        )
    except (OSError, ValueError) as error:
        raise ShadowBookError(f"the family spec could not be loaded: {error}") from error
    if family_spec_hash != declaration.family_spec_hash:
        raise ShadowBookError(
            "the family spec does not hash to the one the declaration names"
        )
    if spec.family_name != declaration.family_name:
        raise ShadowBookError(
            f"the family spec declares {spec.family_name}, not {declaration.family_name}"
        )
    member = next(
        (item for item in spec.members if item.name == declaration.candidate), None
    )
    if member is None:
        raise ShadowBookError(
            f"{declaration.candidate} is not a declared member of {spec.family_name}"
        )
    declared_controls = {item.name for item in spec.controls}
    for name in declaration.controls:
        if name not in declared_controls:
            raise ShadowBookError(f"{name} is not a declared control of {spec.family_name}")
    return spec, member


def _verify_phase(
    declaration: ShadowDeclaration, holdout_report_path: Path | None
) -> None:
    """Spec section 2's gate between the two phases.

    Phase B stands on a confirmed holdout for this family and this candidate,
    and the document is re-sealed here rather than trusted: `report_hash` is
    recomputed over the rest of it, which is how every artifact in this
    repository is read. Phase A runs while the holdout is unread, so a holdout
    report handed to a Phase A week is refused rather than ignored -- ignoring
    it would leave a caller believing a week was authorised that was not.
    """
    if declaration.phase != "B":
        if holdout_report_path is not None:
            raise ShadowBookError(
                "a Phase A shadow week reads no holdout report; Phase A runs while "
                "the holdout is still unread"
            )
        return
    if holdout_report_path is None:
        raise ShadowBookError("a Phase B shadow week requires the holdout report")
    document = _object(holdout_report_path)
    material = {key: value for key, value in document.items() if key != "report_hash"}
    if content_sha256(material) != document.get("report_hash"):
        raise ShadowBookError("the holdout report's seal does not recompute")
    if document.get("report_hash") != declaration.holdout_report_hash:
        raise ShadowBookError("the holdout report is not the one the declaration names")
    if document.get("holdout") is not True:
        raise ShadowBookError("the cited report is not a holdout read")
    if document.get("family_spec_hash") != declaration.family_spec_hash:
        raise ShadowBookError("the holdout report was read for another family")
    if document.get("candidate_name") != declaration.candidate:
        raise ShadowBookError(
            f"the holdout report was opened for {document.get('candidate_name')}, "
            f"not the declared candidate {declaration.candidate}"
        )
    confirmation = document.get("confirmation")
    verdict = confirmation.get("verdict") if isinstance(confirmation, dict) else None
    if verdict != "holdout_confirmed":
        raise ShadowBookError(
            f"the holdout verdict is {verdict}, not holdout_confirmed"
        )


def _measurement_snapshot(path: Path | None) -> tuple[dict[str, Any] | None, str | None]:
    """The cited measurement snapshot, re-sealed before a value is read.

    Spec section 5 makes the snapshot a receipt, so its `content_hash` is
    recomputed over everything else exactly as the journal sealed it. A week
    that cites a document whose seal does not recompute would be citing
    something nobody can reproduce.
    """
    if path is None:
        return None, None
    document = _object(path)
    if document.get("version") != MEASUREMENT_SNAPSHOT_VERSION:
        raise ShadowBookError(
            f"the measurement snapshot is version {document.get('version')}, "
            f"not {MEASUREMENT_SNAPSHOT_VERSION}"
        )
    material = {key: value for key, value in document.items() if key != "content_hash"}
    if content_sha256(material) != document.get("content_hash"):
        raise ShadowBookError("the measurement snapshot's seal does not recompute")
    _verify_measurement_blocks(document)
    _verify_snapshot_window(document)
    return document, str(document["content_hash"])


def _verify_measurement_blocks(document: dict[str, Any]) -> None:
    """Every per-symbol reading is a string, a whole count, or absent.

    Checked here, at the input gate, rather than where each value is copied
    into a pair's block: a snapshot whose fields are the wrong shape is a bad
    input, and a bad input is refused before the week writes anything. A
    number that is neither a count nor a decimal string would be a float,
    which the canonical format refuses outright -- so it would otherwise only
    surface as a sealing error with nothing left pointing at the snapshot.
    """
    for block, keys in (
        ("perpetuals", _PERPETUAL_MEASUREMENT_KEYS),
        ("spot", _SPOT_MEASUREMENT_KEYS),
    ):
        section = document.get(block)
        if not isinstance(section, dict):
            raise ShadowBookError(f"the measurement snapshot's {block} block is malformed")
        for symbol, entry in section.items():
            if not isinstance(entry, dict):
                raise ShadowBookError(
                    f"the measurement snapshot's {block}.{symbol} is malformed"
                )
            for key in keys:
                value = entry.get(key)
                if value is None or isinstance(value, str):
                    continue
                if isinstance(value, int) and not isinstance(value, bool):
                    continue
                raise ShadowBookError(
                    f"the measurement snapshot's {block}.{symbol}.{key} is "
                    "neither a reading nor a count"
                )


def _verify_snapshot_window(document: dict[str, Any]) -> None:
    """The five fields that say which rounds a snapshot is over are whole numbers.

    Checked at the input gate beside the per-symbol shapes: they are copied
    into the sealed week, and `window_end_ns` is what binds the reading to the
    week that cites it, so a snapshot that does not carry them as counts is a
    bad input rather than a week that refuses.
    """
    for key in _SNAPSHOT_WINDOW_KEYS:
        value = document.get(key)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ShadowBookError(
                f"the measurement snapshot's {key} is not a whole number of rounds"
            )


def _require_this_weeks_snapshot(
    document: dict[str, Any] | None, *, decision_close_ns: int
) -> None:
    """Ruling 22: the cited reading is of the week this decision closes.

    The version and the seal say the document is a snapshot of this stream and
    that nobody edited it; neither says it is a reading of *this* week. Without
    this, last week's snapshot -- or one taken days after the decision -- could
    be cited by any week, and every number beside every ranked pair would be a
    measurement of a different week with nothing in the artifact saying so.

    A recorded refusal rather than a raise at the gate (spec section 6): the
    inputs are all good, the operator passed the wrong one of several
    interchangeable-looking documents, and the week that was not written says
    which one it was handed.
    """
    if document is None:
        return
    window_end_ns = document["window_end_ns"]
    if not decision_close_ns - _WEEK_NS < window_end_ns <= decision_close_ns:
        raise _Refusal(
            _SNAPSHOT_NOT_THIS_WEEK,
            f"the cited snapshot's window ends at {window_end_ns}, outside the week "
            f"ending at this decision's close {decision_close_ns}",
        )


def _snapshot_window_block(document: dict[str, Any] | None) -> dict[str, object]:
    """The cited reading's window and sequence bounds, or nulls where none is cited."""
    return {
        f"measurement_snapshot_{key}": None if document is None else document[key]
        for key in _SNAPSHOT_WINDOW_KEYS
    }


def _load_inputs(
    perp_root: Path, spot_root: Path, *, anchor_close_ns: int, decision_close_ns: int
) -> tuple[DecisionInputs, list[int]]:
    """The fold runner's own loader, plus this week's decision list.

    The loading itself is `carry_fold_run.load_decision_inputs` and nothing
    else: a week that assembled its own histories, leg keys or funding index
    could differ from the fold runner in exactly the way "shadow = fold
    mechanics" is supposed to forbid, and no comparison of two evaluations
    would notice, because both would be evaluations of the same private copy.

    `available_before_ns = S + 2` is the point-in-time boundary this caller
    owns: a bar's availability is its close plus one nanosecond, so this
    admits the Sunday's own bar and nothing after it. The decisions are the
    perpetual panel's Sunday closes restricted to `[anchor, S]` -- spec 4.1's
    "decisions before A are never evaluated", with the bars before A still
    read as history.
    """
    loaded = load_decision_inputs(
        perp_root / "dataset",
        spot_root / "dataset",
        available_before_ns=decision_close_ns + 2,
    )
    decisions = [
        close_time_ns
        for close_time_ns in rebalance_close_times(loaded.perp_bars)
        if anchor_close_ns <= close_time_ns <= decision_close_ns
    ]
    return loaded, decisions


def _evaluate(
    spec: CarryFamilySpec,
    inputs: DecisionInputs,
    decisions: list[int],
    candidate_names: tuple[str, ...],
) -> DecisionRun:
    try:
        return evaluate_carry_decisions(
            spec,
            perp_histories=inputs.perp_histories,
            spot_histories=inputs.spot_histories,
            leg_histories=inputs.leg_histories,
            funding_by_leg=inputs.funding_by_leg,
            decisions=decisions,
            candidate_names=candidate_names,
        )
    except CarryFoldError as error:
        raise _Refusal(_DECISION_RUN_REFUSED, str(error)) from error


def _require_sunday_bars(
    book: FinalBook, *, inputs: DecisionInputs, decision_close_ns: int
) -> None:
    """Spec section 6: refuse if the Sunday bar is absent for what the book holds.

    The decision loop already strips a pair whose leg has no bar at the
    decision, so this cannot fire on data the loop accepted -- which is why it
    is here: a book that could not be priced at its own Sunday would be a
    forecast of a position nobody could take, and that must be a refusal
    rather than a silently mispriced artifact.
    """
    missing = sorted(
        leg
        for leg, _ in book.leg_weights
        if decision_close_ns not in inputs.leg_histories[leg].closes
    )
    if missing:
        raise _Refusal(
            _NO_SUNDAY_BAR, "the book holds legs with no Sunday bar: " + ",".join(missing)
        )


def _was_skipped(run: DecisionRun, decision_close_ns: int) -> bool:
    """Whether the runner skipped this Sunday for want of a universe."""
    return any(
        sample_id.split(":")[1] == str(decision_close_ns)
        for sample_id in run.skipped_sample_ids
    )


# --- the blocks ---------------------------------------------------------


def _book_block(book: FinalBook) -> dict[str, object]:
    """Spec 4.2's book: per slot the pair, the Sunday it was entered and how
    long it has been held, plus the union weights per leg -- the fold runner's
    `FinalBook` as plain data and nothing re-derived."""
    return {
        "decision_close_ns": book.decision_close_ns,
        "slots": [
            {
                "pair_id": slot.pair_id,
                "perpetual_leg": slot.perpetual_leg,
                "spot_leg": slot.spot_leg,
                "tier": slot.tier,
                "entry_decision_close_ns": slot.entry_decision_close_ns,
                "weeks_held": slot.weeks_held,
            }
            for slot in book.slots
        ],
        "leg_weights": [[leg, weight] for leg, weight in book.leg_weights],
    }


def _universe_block(
    universe: PairUniverseSnapshot,
    *,
    inputs: DecisionInputs,
    snapshot_document: dict[str, Any] | None,
) -> list[dict[str, object]]:
    """Spec 4.2's measurements, for every pair the universe ranked.

    Measured by the declaration's own functions -- `_trailing_by_pair` reads a
    pair's funding off its perpetual leg exactly as the decision loop does, and
    the exit verdict is `exit_rule_pairs` applied to a cohort holding the whole
    snapshot, so the week cannot disagree with the rule it reports on. The
    pairs come in the universe's own liquidity order.
    """
    one_week = _trailing_by_pair(
        universe, funding_by_leg=inputs.funding_by_leg, lookback_weeks=1
    )
    four_weeks = _trailing_by_pair(
        universe, funding_by_leg=inputs.funding_by_leg, lookback_weeks=4
    )
    exiting = exit_rule_pairs(
        [
            Cohort(
                universe.decision_close_ns,
                entries_for(universe, [pair.pair_id for pair in universe.pairs]),
                (),
            )
        ],
        trailing_one_week=one_week,
    )
    return [
        {
            "pair_id": pair.pair_id,
            "perpetual_leg": f"perp:{pair.perpetual_contract_id}",
            "spot_leg": f"spot:{pair.spot_contract_id}",
            "tier": pair.tier,
            "trailing_funding_1w": one_week[pair.pair_id],
            "trailing_funding_4w": four_weeks[pair.pair_id],
            "exit_rule_exit": pair.pair_id in exiting,
            "measurement": _measurement_for(
                perpetual_leg=f"perp:{pair.perpetual_contract_id}",
                spot_leg=f"spot:{pair.spot_contract_id}",
                inputs=inputs,
                snapshot_document=snapshot_document,
            ),
        }
        for pair in universe.pairs
    ]


def _measurement_for(
    *,
    perpetual_leg: str,
    spot_leg: str,
    inputs: DecisionInputs,
    snapshot_document: dict[str, Any] | None,
) -> dict[str, object] | None:
    """One pair's readings from the measurement stream, or nothing cited.

    A leg's venue symbol is the panel's own `instrument_id` -- the plain
    upper-case symbol the dumps are published under, which is what the
    snapshot's `perpetuals` and `spot` maps are keyed by. A symbol the
    snapshot window never carried is absent from those maps, and absence is
    written out as nulls rather than as zeroes: nothing was measured, which is
    not the same as a measurement of nothing.
    """
    if snapshot_document is None:
        return None
    perpetual_symbol = inputs.leg_histories[perpetual_leg].instrument_id
    spot_symbol = inputs.leg_histories[spot_leg].instrument_id
    return {
        "perpetual_symbol": perpetual_symbol,
        "spot_symbol": spot_symbol,
        "perpetual": _measured(
            snapshot_document, "perpetuals", perpetual_symbol, _PERPETUAL_MEASUREMENT_KEYS
        ),
        "spot": _measured(snapshot_document, "spot", spot_symbol, _SPOT_MEASUREMENT_KEYS),
    }


def _measured(
    document: dict[str, Any], block: str, symbol: str, keys: tuple[str, ...]
) -> dict[str, object]:
    section = document.get(block)
    entry = section.get(symbol) if isinstance(section, dict) else None
    if not isinstance(entry, dict):
        return dict.fromkeys(keys, None)
    # Shapes were checked at the input gate (`_verify_measurement_blocks`).
    return {key: entry.get(key) for key in keys}


def _m_label_block(
    previous_book: FinalBook | None,
    *,
    funding_by_leg: dict[str, tuple[FundingEvent, ...]],
    decision_close_ns: int,
    cost_table: CarryCostTable,
    hold_weeks: int,
) -> list[dict[str, object]]:
    """Spec 4.2's `m`, for every slot the book held at S-1.

    The label is what the week just past paid that slot, net of what holding
    it costs: the pair's trailing one-week funding measured at S, against the
    round trip cost spread over the member's hold. It is the training signal a
    later abstention family (plan section 1) would learn from, which is why it
    is recorded weekly now, before that family exists.
    """
    if previous_book is None:
        return []
    labels: list[dict[str, object]] = []
    for slot in previous_book.slots:
        trailing = trailing_funding(
            funding_by_leg.get(slot.perpetual_leg, ()),
            decision_close_ns=decision_close_ns,
            lookback_weeks=1,
        )
        cost_band = round_trip_cost_bps(cost_table, slot.tier) / Decimal(hold_weeks) / _BPS
        labels.append(
            {
                "pair_id": slot.pair_id,
                "perpetual_leg": slot.perpetual_leg,
                "spot_leg": slot.spot_leg,
                "tier": slot.tier,
                "entry_decision_close_ns": slot.entry_decision_close_ns,
                "weeks_held": slot.weeks_held,
                "trailing_one_week_funding": trailing,
                "cost_band": cost_band,
                "m": _m_label(trailing, cost_band=cost_band),
            }
        )
    return labels


def _m_label(trailing_one_week: Decimal | None, *, cost_band: Decimal) -> int | None:
    """+1 above the cost band, -1 below zero, 0 inside it -- `None` unmeasured.

    The three-way verdict is the plan's Global Constraints, verbatim. A week
    in which the pair had no settlement at all answers none of the three
    questions: there is no value to compare, and calling that 0 would teach
    the future family that silence is the same as a carry inside the band. So
    it is recorded as absent, beside the `null` trailing funding it came from.
    """
    if trailing_one_week is None:
        return None
    if trailing_one_week > cost_band:
        return 1
    if trailing_one_week < 0:
        return -1
    return 0


def _pnl_block(
    run: DecisionRun, *, candidate: str, decision_close_ns: int | None
) -> dict[str, object] | None:
    """Spec 4.2's previous-week P&L: S-1's episode under both cost tables.

    S's own episode is the week ahead -- it exits seven days after a Sunday
    whose week has not happened -- so it is never reported. `None` in the two
    cases where there is honestly nothing to report: no previous Sunday inside
    the run at all (S is the anchor), and a previous Sunday the runner
    skipped, which evaluated no episode because its universe was too small.

    A previous Sunday that *was* evaluated and still has no episode record is
    a different thing entirely -- a bug in the runner or in this reader -- and
    is refused rather than published as an absent P&L, which would read as a
    week that simply earned nothing.
    """
    if decision_close_ns is None:
        return None
    if _was_skipped(run, decision_close_ns):
        return None
    by_scenario: dict[str, list[dict[str, object]]] = {
        scenario: [
            episode
            for episode in _episodes(run, candidate, scenario)
            if _sample_close_ns(episode) == decision_close_ns
        ]
        for scenario in ("base", "adverse")
    }
    missing = sorted(name for name, episodes in by_scenario.items() if not episodes)
    if missing:
        raise _Refusal(
            _DECISION_RUN_REFUSED,
            f"{candidate} was evaluated at {decision_close_ns} but the run carries "
            f"no {'/'.join(missing)} episode for it",
        )
    return {
        "decision_close_ns": decision_close_ns,
        "sample_id": _text(by_scenario["base"][0].get("sample_id"), "sample_id"),
        **{
            scenario: _scenario_pnl(episodes)
            for scenario, episodes in by_scenario.items()
        },
    }


def _running_totals_block(
    run: DecisionRun, *, candidate: str, decisions: list[int]
) -> dict[str, object] | None:
    """Spec 4.2's running totals: every episode of this run before S.

    Since the anchor, because the run starts there, and stopping before S for
    the same reason the P&L block does -- the last decision's episode is a
    week that has not been lived yet.
    """
    if len(decisions) < 2:
        return None
    last = decisions[-1]
    by_scenario = {
        scenario: [
            episode
            for episode in _episodes(run, candidate, scenario)
            if _sample_close_ns(episode) < last
        ]
        for scenario in ("base", "adverse")
    }
    return {
        "from_decision_close_ns": decisions[0],
        "through_decision_close_ns": decisions[-2],
        # The two scenarios evaluate the same decisions, so one count speaks
        # for both; a difference would be a bug in the runner, not a fact.
        "episode_count": len(by_scenario["base"]),
        **{
            scenario: _scenario_pnl(episodes)
            for scenario, episodes in by_scenario.items()
        },
    }


def _scenario_pnl(episodes: list[dict[str, object]]) -> dict[str, object]:
    """One cost table's money over these episodes: net, funding, cost, forced.

    Summed in episode order, which is the order the runner evaluated them in,
    so a total over one episode and a total over many are the same arithmetic.
    """
    net = Decimal(0)
    funding = Decimal(0)
    spot = Decimal(0)
    perpetual = Decimal(0)
    forced_spot = Decimal(0)
    forced_perpetual = Decimal(0)
    for episode in episodes:
        extras = _mapping(episode.get("extras"), "episode extras")
        net += _decimal(episode.get("net_return"), "net_return")
        funding += _decimal(extras.get("funding_collected"), "funding_collected")
        spot += _decimal(extras.get("spot_trading_cost"), "spot_trading_cost")
        perpetual += _decimal(
            extras.get("perpetual_trading_cost"), "perpetual_trading_cost"
        )
        forced_spot += _decimal(extras.get("forced_spot_legs"), "forced_spot_legs")
        forced_perpetual += _decimal(
            extras.get("forced_perpetual_legs"), "forced_perpetual_legs"
        )
    return {
        "net_return": net,
        "funding_collected": funding,
        "trading_cost": spot + perpetual,
        "spot_trading_cost": spot,
        "perpetual_trading_cost": perpetual,
        "forced_spot_legs": int(forced_spot),
        "forced_perpetual_legs": int(forced_perpetual),
    }


def _episodes(run: DecisionRun, candidate: str, scenario: str) -> list[dict[str, object]]:
    record = next(
        (item for item in run.candidates if item.get("candidate_name") == candidate), None
    )
    if record is None:
        raise _Refusal(
            _DECISION_RUN_REFUSED, f"the run produced no record for {candidate}"
        )
    scenario_record = _mapping(record.get(scenario), f"{candidate} {scenario}")
    return [
        _mapping(item, "episode")
        for item in _sequence(scenario_record.get("episodes"), "episodes")
    ]


def _sample_close_ns(episode: dict[str, object]) -> int:
    """The decision close a `BINANCE_UM:<close>:w1` sample id names."""
    parts = _text(episode.get("sample_id"), "sample_id").split(":")
    if len(parts) != 3 or not parts[1].isdigit():
        raise _Refusal(_DECISION_RUN_REFUSED, f"malformed sample id: {parts}")
    return int(parts[1])


# --- sealing and recording ----------------------------------------------


def _publish(output_path: Path, material: dict[str, object]) -> str:
    """Seal the week over itself and put it in place atomically."""
    report_hash = content_sha256(material)
    document = dict(material)
    document["report_hash"] = report_hash
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_bytes(canonical_json(document))
    os.replace(temporary, output_path)
    return report_hash


def _register(
    output_path: Path,
    *,
    workspace: Path,
    registry_path: Path,
    family_spec_hash: str,
    decision_sunday: str,
    report_hash: str,
) -> None:
    """Record the week, keyed by the declaration's family and its Sunday.

    The report and its registry record are one artifact: a report nothing
    recorded would be immutable at a path no retry could reuse while the week
    still reads as unpublished, so the report this call wrote is removed
    before the refusal is raised.

    That cleanup cannot run if the process itself dies between the publish and
    the registration -- a crash, a kill, the machine going down -- which
    leaves an orphan: `<artifact_root>/<family>/<S>.json` on disk with no row
    in the registry. Recovering it is an operator step, deliberately not an
    automatic one, because "delete a sealed artifact" is not a decision code
    should take on its own. The path is: open the registry, look up
    `uuid5(NAMESPACE_URL, "shadow:<family spec hash>:<S>")`, and only if there
    is no row for it, delete that JSON file and run the week again. If there
    *is* a row, the week is published and nothing needs doing -- compare the
    row's `content_hash` against the file's `report_hash` and raise it with a
    person if they differ.
    """
    artifact_id = uuid5(NAMESPACE_URL, f"shadow:{family_spec_hash}:{decision_sunday}")
    try:
        with MetadataRegistry(registry_path) as registry:
            registry.register_artifact(
                ArtifactRecord(
                    artifact_id=artifact_id,
                    kind=SHADOW_ARTIFACT_KIND,
                    relative_path=output_path.resolve().relative_to(workspace).as_posix(),
                    content_hash=report_hash,
                    created_at_ns=time.time_ns(),
                )
            )
    except RegistryConflictError as error:
        output_path.unlink(missing_ok=True)
        raise _Refusal(_ALREADY_REGISTERED, str(error)) from error
    except Exception as error:
        output_path.unlink(missing_ok=True)
        raise _Refusal(
            _NOT_REGISTERED, f"{type(error).__name__}: {error}"
        ) from error


def _write_refusal(
    refusal_path: Path,
    *,
    reason: str,
    detail: str,
    declaration: ShadowDeclaration,
    declaration_hash: str,
    decision_sunday: str,
    decision_close_ns: int,
) -> None:
    """Record the week that was not written (spec section 6).

    Unsealed and unregistered on purpose: it is a note for a person, not an
    artifact anything may cite. It is rewritten rather than refused if one is
    already there -- a week that refuses twice refuses for the newest reason,
    and refusing to record the second refusal would hide it behind the first.
    """
    document: dict[str, object] = {
        "report_version": REPORT_VERSION,
        "status": "refused",
        "reason": reason,
        "detail": detail,
        "family_name": declaration.family_name,
        "family_spec_hash": declaration.family_spec_hash,
        "declaration_hash": declaration_hash,
        "candidate_name": declaration.candidate,
        "decision_sunday": decision_sunday,
        "decision_close_ns": decision_close_ns,
        "refused_at_ns": time.time_ns(),
    }
    refusal_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = refusal_path.with_suffix(refusal_path.suffix + ".tmp")
    temporary.write_bytes(canonical_json(document))
    os.replace(temporary, refusal_path)


def _code_hash() -> str:
    root = Path(__file__).parent
    names = (*carry_module_names(), *_SHADOW_MODULES)
    material = "".join((root / name).read_text(encoding="utf-8") for name in names)
    return content_sha256(material)


# --- small readers ------------------------------------------------------


def _sunday(value: str) -> date:
    day = _date(value)
    if day.weekday() != _SUNDAY:
        raise ShadowBookError(f"{value} is not a Sunday; a decision is a Sunday close")
    return day


def _date(value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ShadowBookError(f"invalid date: {value}") from error
    if parsed.isoformat() != value:
        raise ShadowBookError(f"invalid date: {value}")
    return parsed


def _close_of(day: date) -> int:
    """The daily bar close a calendar day ends on: one millisecond to midnight,
    which is how every Binance daily kline states its close time."""
    return ((day - _EPOCH).days + 1) * _DAY_NS - _NANOSECONDS_PER_MILLISECOND


def _object(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ShadowBookError(f"{path.name} could not be read: {error}") from error
    if not isinstance(document, dict):
        raise ShadowBookError(f"{path.name} must contain a JSON object")
    return document


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise _Refusal(_DECISION_RUN_REFUSED, f"{label} is malformed")
    return value


def _sequence(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise _Refusal(_DECISION_RUN_REFUSED, f"{label} are malformed")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise _Refusal(_DECISION_RUN_REFUSED, f"{label} is malformed")
    return value


def _decimal(value: object, label: str) -> Decimal:
    if not isinstance(value, Decimal):
        raise _Refusal(_DECISION_RUN_REFUSED, f"{label} is not a decimal")
    return value
