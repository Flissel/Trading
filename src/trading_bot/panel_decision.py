"""Pooled gates and the decision report for the panel family."""

import json
import time
from dataclasses import dataclass
from decimal import Context, Decimal, Inexact, localcontext
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.carry_config import CarryFamilySpec
from trading_bot.evaluation import (
    BootstrapMeanTest,
    benjamini_hochberg,
    block_bootstrap_mean_test,
)
from trading_bot.evaluation import _maximum_drawdown as _shared_maximum_drawdown
from trading_bot.funding_xs_config import FundingXsFamilySpec
from trading_bot.panel_config import PanelFamilySpec, load_family_spec
from trading_bot.panel_fold_run import MEMBER_HELD_NOTHING_REASON_CODE, verify_panel_fold_report
from trading_bot.panel_statistics import (
    PanelStatisticsError,
    annualised_sharpe,
    deflated_sharpe_ratio,
    sharpe_ratio,
)
from trading_bot.registry import ArtifactRecord, MetadataRegistry
from trading_bot.trend_config import TrendFamilySpec

# Section 8.1's declared weight construction reads from the *full eligible*
# universe; the implementation reads from the subset of it that is actually
# rankable at that decision (has the exact lagged close for the member's
# lookback, or a full volatility window). This is a known, deliberate
# reporting departure from the spec text -- not a change to how weights are
# built -- surfaced in every decision report via `declared_deviations`.
_DECLARED_DEVIATIONS: tuple[dict[str, object], ...] = (
    {
        "id": "RANKABLE_SUBSET_NOT_FULL_UNIVERSE",
        "description": (
            "The cross-sectional quintile size and the time-series weight cap are "
            "computed from the contracts that are actually rankable at a decision, not "
            "from the full eligible universe as spec section 8.1 states, because a "
            "contract can be eligible yet lack the exact lagged close or a full "
            "volatility window."
        ),
    },
    {
        "id": "MEMBER_HELD_NOTHING_EXCLUDED_FROM_POOLED_SERIES",
        "description": (
            "Spec section 7.1 authorises excluding a week from the pooled series and "
            "the episode floor only at the universe level (UNIVERSE_TOO_SMALL), where "
            "no episode and no cost ever exist. A week a single member holds nothing "
            "is excluded from that member's own pooled series and episode count the "
            "same way, even though the episode can carry a real unwind cost (net "
            "return and turnover); that cost is never dropped -- it is rolled into "
            "the member's next retained episode in the same fold, or, if none "
            "follows, rolled backward onto that fold's last retained episode so it "
            "still enters the pooled series and not merely the total. Only a fold "
            "that retains nothing at all falls back to keeping the cost in the total "
            "alone, since there is then no episode to roll onto. Reported per member "
            "and scenario as the held-nothing episode count and the net return and "
            "turnover rolled forward."
        ),
    },
)


# Precision for the pooling summations in `_pool`/`_pool_scenario_fold` only
# (never the ambient default context): generous headroom above the default
# 28 significant digits so that summing the bounded number of already-28-29-
# digit episode and contract values pooled here never itself needs to round.
# See the comment at its point of use for why that is what actually makes
# `sum(contract_totals)` and `base_total` -- two different groupings of the
# same underlying money -- agree exactly rather than approximately.
#
# Real inputs needed roughly 32 digits of headroom for P1.27 (measured) and
# prec=50 covered P1.27 through P1.29. The trend family's four-week cohort
# book (P1.31) can leave dust weights of about 1e-28 where a long in an older
# vector nearly cancels a short in a newer one; a dust weight times a return
# carries its 28 significant digits down to about 1e-57 and needs ~60 digits
# to sum exactly. prec=120 keeps every previously exact sum exactly the same
# (an exact sum is exact at any sufficient precision -- the P1.29 decision
# re-derives to the identical report hash) and leaves twice the margin the
# worst measured input needs. Inexact stays trapped so that margin is an
# enforced guarantee, not an assumption: if it is ever exceeded, this raises
# immediately rather than silently rounding to a wrong total that nothing
# downstream would notice -- the exact failure mode this precision work
# exists to remove.
_POOLING_CONTEXT = Context(prec=120)
_POOLING_CONTEXT.traps[Inexact] = True


class PanelDecisionError(RuntimeError):
    """Raised when the panel decision cannot be derived or published."""


# `build_panel_decision` reads only `members`, `controls`, `statistics` and
# `family_name` from the loaded spec -- fields every family model declares --
# so any of them can be pooled through the same gates without this module
# knowing anything else about the carry, trend aggregate or funding
# cross-section families.
FamilySpec = PanelFamilySpec | CarryFamilySpec | TrendFamilySpec | FundingXsFamilySpec

# Control dominance (a member must beat both) is measured against the two
# "no active view" baseline controls -- no-trade and a random assignment --
# never the third, contextual benchmark control every family also declares
# (the momentum panel's passive-long-every-week, the carry family's
# all-pairs-every-week). Selecting on `kind` rather than name or position
# means this keeps meaning the same two *roles* even if a family's control
# order or names ever changed; selecting on position (`control_names[:2]`)
# was only correct by coincidence of both families' current frozen order.
_CONTEXT_CONTROL_KINDS: frozenset[str] = frozenset({"passive_long", "all_pairs"})


@dataclass(frozen=True, slots=True)
class PanelDecisionArtifact:
    output_path: Path
    report_hash: str
    decision_status: str
    eligible_member_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Pooled:
    base_returns: tuple[Decimal, ...]
    adverse_returns: tuple[Decimal, ...]
    base_total: Decimal
    adverse_total: Decimal
    fold_base_totals: tuple[Decimal, ...]
    # How many episodes this member actually retained from each fold, in
    # fold-index order -- surfaces the case the R1 backward-roll fallback
    # cannot fix on its own: a fold where the member retained zero episodes
    # (every one of that fold's decisions was MEMBER_HELD_NOTHING). That fold
    # then contributes only to `fold_base_totals`/`base_total`, never to
    # `base_returns`, `base_turnovers` or the bootstrap series they feed --
    # authorised (there is no episode within that fold to roll the cost onto)
    # but otherwise invisible, since a fold like that looks identical to a
    # healthy one everywhere else in the report.
    base_fold_retained_episode_counts: tuple[int, ...]
    adverse_fold_retained_episode_counts: tuple[int, ...]
    contract_totals: dict[str, Decimal]
    episode_count: int
    # Episode count before excluding MEMBER_HELD_NOTHING-marked episodes --
    # the same for every candidate in one decision (it reflects how many
    # decisions were not skipped for UNIVERSE_TOO_SMALL, independent of any
    # one member's own holding pattern) and used only for the cross-candidate
    # linkage check and the top-level `pooled_raw_episode_count` report field.
    # A member's own post-exclusion count is `episode_count` above; the two
    # are named apart deliberately so a reader cannot confuse the shared
    # decision calendar with the (per-member, generally smaller) count the
    # episode floor is actually checked against.
    raw_episode_count: int
    base_turnovers: tuple[Decimal, ...]
    adverse_turnovers: tuple[Decimal, ...]
    base_gross_exposures: tuple[Decimal, ...]
    adverse_gross_exposures: tuple[Decimal, ...]
    base_net_exposures: tuple[Decimal, ...]
    adverse_net_exposures: tuple[Decimal, ...]
    base_forced_close_count: int
    adverse_forced_close_count: int
    # Visibility into the roll-forward/-backward of the MEMBER_HELD_NOTHING
    # exclusion: how many episodes were marked for this candidate and
    # scenario, and the total net return and turnover those episodes
    # carried -- money and activity that never left the pooled series (each
    # was rolled into an adjacent retained episode in the same fold, forward
    # if one follows, backward onto the fold's last retained episode
    # otherwise) even though the episode itself no longer counts as its own
    # observation.
    base_held_nothing_episode_count: int
    adverse_held_nothing_episode_count: int
    base_held_nothing_net_return_rolled_forward: Decimal
    adverse_held_nothing_net_return_rolled_forward: Decimal
    base_held_nothing_turnover_rolled_forward: Decimal
    adverse_held_nothing_turnover_rolled_forward: Decimal
    # Sum, per extras key, of that key's Decimal value across this candidate's
    # retained base episodes only -- the same retained set `base_returns`
    # holds, with the same MEMBER_HELD_NOTHING episodes excluded. Populated
    # only for families whose fold reports carry an `extras` mapping per
    # episode (funding carry); empty (count 0) for the momentum panel.
    base_extras_totals: dict[str, Decimal]
    base_extras_count: int


def build_panel_decision(
    fold_report_paths: tuple[Path, ...],
    *,
    family_spec_path: Path,
    output_path: Path,
    registry_path: Path,
) -> PanelDecisionArtifact:
    if not fold_report_paths:
        raise PanelDecisionError("at least one fold report is required")
    if output_path.exists():
        raise PanelDecisionError("panel decision already exists and is immutable")
    spec, family_spec_hash = load_family_spec(family_spec_path)

    documents = []
    for path in fold_report_paths:
        if not verify_panel_fold_report(path):
            raise PanelDecisionError(f"fold report failed verification: {path.name}")
        documents.append(_load_object(path))
    indices = [_int_field(document, "fold_index") for document in documents]
    if len(set(indices)) != len(indices):
        raise PanelDecisionError("fold reports must have distinct fold indices")
    declared_fold_counts = {_int_field(document, "fold_count") for document in documents}
    if len(declared_fold_counts) != 1:
        raise PanelDecisionError("fold reports do not agree on the declared fold count")
    declared_fold_count = next(iter(declared_fold_counts))
    expected_indices = set(range(declared_fold_count))
    actual_indices = set(indices)
    missing_indices = sorted(expected_indices - actual_indices)
    unexpected_indices = sorted(actual_indices - expected_indices)
    if missing_indices or unexpected_indices:
        details = []
        if missing_indices:
            details.append(
                "missing fold index(es) " + ", ".join(str(index) for index in missing_indices)
            )
        if unexpected_indices:
            details.append(
                "unexpected fold index(es) "
                + ", ".join(str(index) for index in unexpected_indices)
            )
        raise PanelDecisionError(
            "fold report family is not complete: " + "; ".join(details)
        )
    linkage = {
        (
            str(document["family_spec_hash"]),
            str(document["split_manifest_hash"]),
            str(document["dataset_root_hash"]),
            str(document["capture_root_hash"]),
        )
        for document in documents
    }
    if len(linkage) != 1:
        raise PanelDecisionError("fold reports do not share one family and manifest")
    if next(iter(linkage))[0] != family_spec_hash:
        raise PanelDecisionError("fold reports were produced under a different declaration")
    # A family half-run under an older panel_fold_run.py (before MEMBER_HELD_NOTHING
    # marking existed, say) and half under a newer one would apply the pooled
    # exclusion to some folds only, silently, with none of the checks above ever
    # firing -- code_hash covers exactly that class of drift.
    code_hashes = {str(document["code_hash"]) for document in documents}
    if len(code_hashes) != 1:
        raise PanelDecisionError(
            "fold reports were built by different code versions (code_hash mismatch)"
        )
    # A funding-carry fold report additionally links to the hedge leg's own
    # capture and dataset: `hedge_capture_root_hash` and
    # `hedge_dataset_root_hash`. Presence must be all-or-none across the fold
    # report family (a report set where some folds carry the keys and others
    # do not is caught here too, since the set below then holds one tuple of
    # `None`s alongside the real one). The momentum panel's reports carry
    # neither key, so both members of the tuple are `None` on every document
    # and the set collapses to the single `(None, None)` entry harmlessly.
    hedge_links = {
        (document.get("hedge_capture_root_hash"), document.get("hedge_dataset_root_hash"))
        for document in documents
    }
    if len(hedge_links) != 1:
        raise PanelDecisionError("fold reports do not agree on the hedge capture linkage")
    hedge_capture_root_hash, hedge_dataset_root_hash = next(iter(hedge_links))
    # The set-agreement check above only catches disagreement *across*
    # documents; it is blind to a value every document agrees on together
    # being half-missing, e.g. every fold declaring a capture hash but none a
    # dataset hash -- one `("a"*64, None)` tuple, set size 1, no error. Only
    # a fully-present or fully-absent pair is a coherent hedge linkage.
    if (hedge_capture_root_hash is None) != (hedge_dataset_root_hash is None):
        raise PanelDecisionError("fold reports do not agree on the hedge capture linkage")
    documents.sort(key=lambda item: _int_field(item, "fold_index"))

    member_names = tuple(item.name for item in spec.members)
    control_names = tuple(item.name for item in spec.controls)
    pooled: dict[str, _Pooled] = {
        name: _pool(documents, name) for name in member_names + control_names
    }
    # This checks the *raw* per-candidate episode count, from before any
    # MEMBER_HELD_NOTHING episode is excluded below -- that raw count is what
    # every candidate must share (it reflects the fold-report family's own
    # decision calendar), whereas the post-exclusion `episode_count` now
    # legitimately differs member by member.
    raw_episode_counts = {
        pooled[name].raw_episode_count for name in member_names + control_names
    }
    if len(raw_episode_counts) != 1:
        raise PanelDecisionError("candidates do not share one raw pooled episode count")
    pooled_raw_episode_count = next(iter(raw_episode_counts))
    fold_count = len(documents)

    tests: dict[str, BootstrapMeanTest] = {}
    for name in member_names:
        series = pooled[name].base_returns
        if len(series) < spec.statistics.block_length:
            continue
        try:
            tests[name] = block_bootstrap_mean_test(
                series,
                block_length=spec.statistics.block_length,
                repetitions=spec.statistics.bootstrap_repetitions,
                seed=spec.statistics.random_seed,
                confidence=spec.statistics.confidence,
            )
        except ValueError as error:
            raise PanelDecisionError(f"bootstrap failed for {name}: {error}") from error
    q_values = (
        benjamini_hochberg({name: test.one_sided_p_value for name, test in tests.items()})
        if tests
        else {}
    )

    trial_sharpes: list[Decimal] = []
    for name in member_names:
        try:
            trial_sharpes.append(sharpe_ratio(pooled[name].base_returns))
        except PanelStatisticsError:
            trial_sharpes.append(Decimal(0))

    dominance_controls = tuple(
        item.name for item in spec.controls if item.kind not in _CONTEXT_CONTROL_KINDS
    )
    if len(dominance_controls) != 2:
        raise PanelDecisionError("family must declare exactly two dominance controls")
    strongest_base = max(
        (pooled[name].base_total for name in dominance_controls), default=Decimal(0)
    )
    strongest_adverse = max(
        (pooled[name].adverse_total for name in dominance_controls),
        default=Decimal(0),
    )

    skipped_sample_ids = sorted(
        {
            value
            for document in documents
            for value in _string_list_field(document, "skipped_sample_ids")
        }
    )
    universe_too_small_week_count = len(skipped_sample_ids)

    members: list[dict[str, object]] = []
    eligible: list[str] = []
    for name in member_names:
        record, status = _member_record(
            name,
            pooled[name],
            spec=spec,
            fold_count=fold_count,
            test=tests.get(name),
            q_value=q_values.get(name),
            trial_sharpes=tuple(trial_sharpes),
            strongest_base=strongest_base,
            strongest_adverse=strongest_adverse,
            universe_too_small_week_count=universe_too_small_week_count,
        )
        members.append(record)
        if status == "eligible_for_further_review":
            eligible.append(name)

    controls = [
        {
            "candidate_name": name,
            "pooled_retained_episode_count": pooled[name].episode_count,
            "base_total_net_return": pooled[name].base_total,
            "adverse_total_net_return": pooled[name].adverse_total,
            **_extras_mean_field(pooled[name]),
        }
        for name in control_names
    ]

    decision_status = "eligible_member_available" if eligible else "no_eligible_member"
    material: dict[str, object] = {
        "decision_version": "1.0.0",
        "status": "development_only",
        "family_name": spec.family_name,
        "family_spec_hash": family_spec_hash,
        "capture_root_hash": str(documents[0]["capture_root_hash"]),
        "dataset_root_hash": str(documents[0]["dataset_root_hash"]),
        "split_manifest_hash": str(documents[0]["split_manifest_hash"]),
        "fold_count": fold_count,
        "pooled_raw_episode_count": pooled_raw_episode_count,
        "source_report_hashes": [str(document["report_hash"]) for document in documents],
        "skipped_sample_ids": skipped_sample_ids,
        "block_length": spec.statistics.block_length,
        "bootstrap_repetitions": spec.statistics.bootstrap_repetitions,
        "random_seed": spec.statistics.random_seed,
        "declared_deviations": list(_DECLARED_DEVIATIONS),
        "members": members,
        "controls": controls,
        "eligible_member_names": eligible,
        "decision_status": decision_status,
    }
    if hedge_capture_root_hash is not None and hedge_dataset_root_hash is not None:
        material["hedge_capture_root_hash"] = str(hedge_capture_root_hash)
        material["hedge_dataset_root_hash"] = str(hedge_dataset_root_hash)
    report_hash = content_sha256(material)
    document = dict(material)
    document["report_hash"] = report_hash
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_bytes(canonical_json(document))
    temporary.replace(output_path)
    _register_artifact(registry_path, output_path=output_path, report_hash=report_hash)
    return PanelDecisionArtifact(
        output_path=output_path,
        report_hash=report_hash,
        decision_status=decision_status,
        eligible_member_names=tuple(eligible),
    )


def _register_artifact(registry_path: Path, *, output_path: Path, report_hash: str) -> None:
    artifact_id = uuid5(NAMESPACE_URL, f"panel_decision:{report_hash}")
    with MetadataRegistry(registry_path) as registry:
        existing = registry.get_artifact(artifact_id)
        created_at_ns = existing.created_at_ns if existing is not None else time.time_ns()
        registry.register_artifact(
            ArtifactRecord(
                artifact_id=artifact_id,
                kind="panel_decision",
                relative_path=output_path.name,
                content_hash=report_hash,
                created_at_ns=created_at_ns,
            )
        )


def _held_nothing(episode: object) -> bool:
    """True if the fold report marked this episode MEMBER_HELD_NOTHING.

    A member's own weight construction can return no weights (too few
    rankable contracts to form both cross-sectional quintiles, or none with a
    full volatility window) even though the week's universe was not too
    small. Such an episode holds no position and must be excluded from the
    pooled series and the episode floor the same way a UNIVERSE_TOO_SMALL
    week already is (spec section 7.1), one level down. Older-shaped episode
    dicts with no `reason_codes` key at all are simply never held-nothing.
    """
    if not isinstance(episode, dict):
        return False
    codes = episode.get("reason_codes", ())
    if not isinstance(codes, list | tuple):
        return False
    return MEMBER_HELD_NOTHING_REASON_CODE in codes


@dataclass(frozen=True, slots=True)
class _ScenarioFoldResult:
    fold_total: Decimal
    held_nothing_count: int
    held_nothing_net_return: Decimal
    held_nothing_turnover: Decimal
    forced_close_count: int
    extras_count: int


def _pool_scenario_fold(
    episodes: list[object],
    *,
    values: list[Decimal],
    turnovers: list[Decimal],
    gross_exposures: list[Decimal],
    net_exposures: list[Decimal],
    contract_totals: dict[str, Decimal] | None,
    extras_totals: dict[str, Decimal] | None = None,
) -> _ScenarioFoldResult:
    """Pool one fold's one-scenario episode list, appending each retained
    episode's (possibly cost-adjusted) values to the accumulator lists.

    A MEMBER_HELD_NOTHING episode is real accounting: if a position was
    carried into it, `evaluate_episode` charges a genuine unwind (turnover,
    a strictly negative net return). Dropping that episode outright would
    silently erase that cost while the next retained episode still pays a
    full re-entry turnover -- one side of a round trip discarded, the other
    kept, always in the favourable direction. So its net_return and turnover
    are carried forward and added onto the next retained episode in this
    same fold instead of being dropped. If a held-nothing episode is this
    fold's *last*, with no retained episode after it to roll into, it is
    instead rolled *backward* onto this fold's own last retained episode --
    round 2 review: rolling only into `fold_total` for that case left the
    cost out of `values`/`turnovers`, the exact series the bootstrap, both
    Sharpes, the drawdown, the median, the win rate, the mean turnover and
    the largest-episode share all read, reproducing the same favourable
    bias one level down. Only when the fold retains nothing at all is
    there truly no episode to roll onto; the cost then lives only in
    `fold_total` (and the `held_nothing_*` totals below), which is the one
    case today's total-only behaviour is still correct for.

    `contract_net_contributions` and `forced_close_count` are accumulated
    for *every* episode, held-nothing or not, unconditionally, before the
    held-nothing branch even runs: unlike `values`/`turnovers` they are
    plain aggregates with no per-episode attribution to preserve, so there
    is nothing to roll forward or backward -- only something that must not
    be skipped. Skipping a held-nothing episode's contract_net_contributions
    (as an earlier version did, via a `continue` that fired before this
    loop) made `sum(contract_totals)` differ from the pooled total by
    exactly the skipped amount, corrupting `largest_contract_share`.

    `fold_total` is accumulated as a single left-to-right pass over every
    episode's own net_return, independent of the roll-forward/-backward
    bookkeeping used for `values` -- Decimal addition is not associative
    at its default 28-significant-digit precision, so building the total
    by summing the (reordered, cost-adjusted) retained values instead can
    differ from a plain in-order sum by one unit in the last place.
    Accumulating it directly, in the episodes' own order, is what actually
    makes the pooled total equal the raw total exactly, to the last digit,
    rather than merely approximately.
    """
    fold_total = Decimal(0)
    held_nothing_count = 0
    held_nothing_net_return = Decimal(0)
    held_nothing_turnover = Decimal(0)
    forced_close_count = 0
    extras_count = 0
    pending_return = Decimal(0)
    pending_turnover = Decimal(0)
    retained_before = len(values)
    for episode in episodes:
        if not isinstance(episode, dict):
            raise PanelDecisionError("fold report episode is malformed")
        net_return = Decimal(str(episode["net_return"]))
        turnover = Decimal(str(episode["turnover"]))
        fold_total += net_return
        forced_close_count += int(episode["forced_close_count"])
        if contract_totals is not None:
            for contract_id, contribution in episode["contract_net_contributions"]:
                contract_totals[str(contract_id)] = contract_totals.get(
                    str(contract_id), Decimal(0)
                ) + Decimal(str(contribution))
        if _held_nothing(episode):
            held_nothing_count += 1
            held_nothing_net_return += net_return
            held_nothing_turnover += turnover
            pending_return += net_return
            pending_turnover += turnover
            continue
        values.append(net_return + pending_return)
        turnovers.append(turnover + pending_turnover)
        gross_exposures.append(Decimal(str(episode["gross_exposure"])))
        net_exposures.append(Decimal(str(episode["net_exposure"])))
        pending_return = Decimal(0)
        pending_turnover = Decimal(0)
        # `extras` is retained-episode bookkeeping (funding collected, basis
        # P&L, cost split by leg for the funding-carry family): accumulated
        # only here, in the same branch and over the same set of episodes
        # `values`/`turnovers` are, so a held-nothing episode's extras leave
        # no trace, exactly like its net_return and turnover leave no trace
        # in the pooled series (theirs is rolled forward/backward instead;
        # extras carries no such rolling, since it has no defined meaning for
        # an episode that held nothing). Absent on older-shaped episodes and
        # on every panel-family report, which is why this is `None` there.
        if extras_totals is not None:
            extras = episode.get("extras")
            if isinstance(extras, dict):
                for key, value in extras.items():
                    extras_totals[str(key)] = extras_totals.get(
                        str(key), Decimal(0)
                    ) + Decimal(str(value))
                extras_count += 1
    if (pending_return != 0 or pending_turnover != 0) and len(values) > retained_before:
        # A held-nothing run at the tail of this fold, with nothing after it to
        # roll forward into: roll it backward onto this fold's own last
        # retained episode instead. If the fold retained nothing at all
        # (`len(values) == retained_before`), there is no episode to roll onto
        # either way -- the cost stays visible only via `fold_total` and the
        # held_nothing_* totals, exactly as before.
        values[-1] += pending_return
        turnovers[-1] += pending_turnover
    return _ScenarioFoldResult(
        fold_total=fold_total,
        held_nothing_count=held_nothing_count,
        held_nothing_net_return=held_nothing_net_return,
        held_nothing_turnover=held_nothing_turnover,
        forced_close_count=forced_close_count,
        extras_count=extras_count,
    )


def _pool(documents: list[dict[str, object]], name: str) -> _Pooled:
    base: list[Decimal] = []
    adverse: list[Decimal] = []
    fold_totals: list[Decimal] = []
    base_fold_retained_counts: list[int] = []
    adverse_fold_retained_counts: list[int] = []
    contracts: dict[str, Decimal] = {}
    base_turnovers: list[Decimal] = []
    adverse_turnovers: list[Decimal] = []
    base_gross_exposures: list[Decimal] = []
    adverse_gross_exposures: list[Decimal] = []
    base_net_exposures: list[Decimal] = []
    adverse_net_exposures: list[Decimal] = []
    base_forced_close_count = 0
    adverse_forced_close_count = 0
    base_held_nothing_count = 0
    adverse_held_nothing_count = 0
    base_held_nothing_rolled = Decimal(0)
    adverse_held_nothing_rolled = Decimal(0)
    base_held_nothing_turnover_rolled = Decimal(0)
    adverse_held_nothing_turnover_rolled = Decimal(0)
    base_total = Decimal(0)
    adverse_total = Decimal(0)
    raw_episode_count = 0
    base_extras_totals: dict[str, Decimal] = {}
    base_extras_count = 0
    # Elevated precision, scoped to this pooling pass only: `base_total` is
    # accumulated episode-by-episode (in `_pool_scenario_fold`) while
    # `contract_totals` accumulates the very same underlying money grouped by
    # contract instead. Those are two different additions of the same
    # multiset of numbers, and Decimal addition is not associative at the
    # default 28-significant-digit precision -- summing the identical values
    # in a different order can round differently, so `sum(contract_totals)`
    # drifted from `base_total` by one part in roughly 1e30. Every episode
    # and contract value here already has at most ~28-29 significant digits;
    # summing at most a few hundred of them never needs more than a handful
    # of extra digits to represent the exact mathematical result, so this
    # margin makes every summation in this pass exact (no rounding at all),
    # and mathematically exact sums are associative -- any two correct
    # regroupings of the same numbers then agree to the last digit, not by
    # coincidence of accumulation order.
    try:
        with localcontext(_POOLING_CONTEXT):
            for document in documents:
                candidates = document["candidates"]
                if not isinstance(candidates, list):
                    raise PanelDecisionError("fold report candidates are malformed")
                record = next(
                    (item for item in candidates if item.get("candidate_name") == name), None
                )
                if record is None:
                    raise PanelDecisionError(f"fold report is missing candidate {name}")
                base_episodes = record["base"]["episodes"]
                adverse_episodes = record["adverse"]["episodes"]
                if not isinstance(base_episodes, list) or not isinstance(adverse_episodes, list):
                    raise PanelDecisionError("fold report episodes are malformed")
                raw_episode_count += len(base_episodes)

                base_retained_before = len(base)
                base_result = _pool_scenario_fold(
                    base_episodes,
                    values=base,
                    turnovers=base_turnovers,
                    gross_exposures=base_gross_exposures,
                    net_exposures=base_net_exposures,
                    contract_totals=contracts,
                    extras_totals=base_extras_totals,
                )
                fold_totals.append(base_result.fold_total)
                base_fold_retained_counts.append(len(base) - base_retained_before)
                base_total += base_result.fold_total
                base_held_nothing_count += base_result.held_nothing_count
                base_held_nothing_rolled += base_result.held_nothing_net_return
                base_held_nothing_turnover_rolled += base_result.held_nothing_turnover
                base_forced_close_count += base_result.forced_close_count
                base_extras_count += base_result.extras_count

                adverse_retained_before = len(adverse)
                adverse_result = _pool_scenario_fold(
                    adverse_episodes,
                    values=adverse,
                    turnovers=adverse_turnovers,
                    gross_exposures=adverse_gross_exposures,
                    net_exposures=adverse_net_exposures,
                    contract_totals=None,
                )
                adverse_fold_retained_counts.append(len(adverse) - adverse_retained_before)
                adverse_total += adverse_result.fold_total
                adverse_held_nothing_count += adverse_result.held_nothing_count
                adverse_held_nothing_rolled += adverse_result.held_nothing_net_return
                adverse_held_nothing_turnover_rolled += adverse_result.held_nothing_turnover
                adverse_forced_close_count += adverse_result.forced_close_count
    except Inexact as error:
        raise PanelDecisionError(
            "pooling summation exceeded its precision headroom (_POOLING_CONTEXT) -- "
            "the pooled total can no longer be guaranteed exact"
        ) from error
    return _Pooled(
        base_returns=tuple(base),
        adverse_returns=tuple(adverse),
        base_total=base_total,
        adverse_total=adverse_total,
        fold_base_totals=tuple(fold_totals),
        base_fold_retained_episode_counts=tuple(base_fold_retained_counts),
        adverse_fold_retained_episode_counts=tuple(adverse_fold_retained_counts),
        contract_totals=contracts,
        episode_count=len(base),
        raw_episode_count=raw_episode_count,
        base_turnovers=tuple(base_turnovers),
        adverse_turnovers=tuple(adverse_turnovers),
        base_gross_exposures=tuple(base_gross_exposures),
        adverse_gross_exposures=tuple(adverse_gross_exposures),
        base_net_exposures=tuple(base_net_exposures),
        adverse_net_exposures=tuple(adverse_net_exposures),
        base_forced_close_count=base_forced_close_count,
        adverse_forced_close_count=adverse_forced_close_count,
        base_held_nothing_episode_count=base_held_nothing_count,
        adverse_held_nothing_episode_count=adverse_held_nothing_count,
        base_held_nothing_net_return_rolled_forward=base_held_nothing_rolled,
        adverse_held_nothing_net_return_rolled_forward=adverse_held_nothing_rolled,
        base_held_nothing_turnover_rolled_forward=base_held_nothing_turnover_rolled,
        adverse_held_nothing_turnover_rolled_forward=adverse_held_nothing_turnover_rolled,
        base_extras_totals=base_extras_totals,
        base_extras_count=base_extras_count,
    )


def _folds_with_no_retained_episodes(counts: tuple[int, ...]) -> list[int]:
    """Fold indices (positional, which is also fold_index -- documents are
    sorted by fold_index before pooling) where a member retained zero
    episodes: the R1 backward-roll fallback for a fold that retains nothing
    at all, made visible rather than indistinguishable from a healthy fold.
    """
    return [index for index, count in enumerate(counts) if count == 0]


def _mean(values: tuple[Decimal, ...]) -> Decimal:
    return sum(values, Decimal(0)) / Decimal(len(values)) if values else Decimal(0)


def _median(values: tuple[Decimal, ...]) -> Decimal:
    if not values:
        return Decimal(0)
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / Decimal(2)


def _win_rate(values: tuple[Decimal, ...]) -> Decimal:
    """Wins over count, zero when there is nothing to divide by.

    Mirrors `evaluate_signals`'s win-rate definition in `evaluation.py`
    (wins among the active trades, divided by the active trade count) so the
    bar-cadence and panel research lines agree on what "win rate" means.
    """
    if not values:
        return Decimal(0)
    wins = sum(1 for value in values if value > 0)
    return Decimal(wins) / Decimal(len(values))


def _extras_mean_field(pooled: _Pooled) -> dict[str, dict[str, Decimal]]:
    """`{"extras_mean": ...}` when this candidate's fold reports carried an
    `extras` mapping, computed with ordinary Decimal division outside the
    trapped `_POOLING_CONTEXT`; `{}` (the key absent entirely) otherwise, so a
    family without extras (the momentum panel) produces byte-identical
    records to before this feature existed.
    """
    if pooled.base_extras_count == 0:
        return {}
    count = Decimal(pooled.base_extras_count)
    return {
        "extras_mean": {
            key: total / count for key, total in sorted(pooled.base_extras_totals.items())
        }
    }


def _member_record(
    name: str,
    pooled: _Pooled,
    *,
    spec: FamilySpec,
    fold_count: int,
    test: BootstrapMeanTest | None,
    q_value: Decimal | None,
    trial_sharpes: tuple[Decimal, ...],
    strongest_base: Decimal,
    strongest_adverse: Decimal,
    universe_too_small_week_count: int,
) -> tuple[dict[str, object], str]:
    evidence: list[str] = []
    economic: list[str] = []
    if pooled.episode_count < spec.statistics.pooled_episode_floor:
        evidence.append("EPISODE_FLOOR_NOT_MET")
    if test is None:
        evidence.append("BOOTSTRAP_NOT_AVAILABLE")

    base_mean = (
        pooled.base_total / Decimal(pooled.episode_count)
        if pooled.episode_count
        else Decimal(0)
    )
    adverse_mean = (
        pooled.adverse_total / Decimal(len(pooled.adverse_returns))
        if pooled.adverse_returns
        else Decimal(0)
    )
    lower = test.interval.lower if test is not None else None
    if base_mean <= 0:
        economic.append("AGGREGATE_BASE_NET_NON_POSITIVE")
    if lower is not None and lower <= 0:
        economic.append("BASE_LOWER_BOUND_NON_POSITIVE")
    if adverse_mean < 0:
        economic.append("AGGREGATE_ADVERSE_NET_NON_POSITIVE")
    required = (
        spec.statistics.positive_fold_numerator * fold_count
        + spec.statistics.positive_fold_denominator
        - 1
    ) // spec.statistics.positive_fold_denominator
    positive_folds = sum(1 for value in pooled.fold_base_totals if value > 0)
    if positive_folds < required:
        economic.append("POSITIVE_FOLD_FRACTION_NOT_MET")
    if q_value is not None and q_value > spec.statistics.false_discovery_gate:
        economic.append("MULTIPLE_TESTING_GATE_NOT_MET")

    shares: dict[str, Decimal] = {}
    if pooled.base_total > 0:
        limit = spec.statistics.concentration_limit
        shares = {
            "largest_fold_share": max(pooled.fold_base_totals, default=Decimal(0))
            / pooled.base_total,
            "largest_contract_share": max(
                pooled.contract_totals.values(), default=Decimal(0)
            )
            / pooled.base_total,
            "largest_episode_share": max(pooled.base_returns, default=Decimal(0))
            / pooled.base_total,
        }
        if any(value > limit for value in shares.values()):
            economic.append("CONCENTRATION_LIMIT_EXCEEDED")
    if pooled.base_total <= strongest_base:
        economic.append("BASE_CONTROL_DOMINANCE_NOT_MET")
    if pooled.adverse_total <= strongest_adverse:
        economic.append("ADVERSE_CONTROL_DOMINANCE_NOT_MET")

    if evidence:
        status = "insufficient_evidence"
    elif economic:
        status = "rejected"
    else:
        status = "eligible_for_further_review"

    try:
        annual = annualised_sharpe(pooled.base_returns)
        deflated = deflated_sharpe_ratio(pooled.base_returns, trial_sharpes=trial_sharpes)
    except PanelStatisticsError:
        annual = Decimal(0)
        deflated = Decimal(0)

    record: dict[str, object] = {
        "candidate_name": name,
        "pooled_retained_episode_count": pooled.episode_count,
        "base_total_net_return": pooled.base_total,
        "base_mean_net_return": base_mean,
        "base_median_net_return": _median(pooled.base_returns),
        "base_win_rate": _win_rate(pooled.base_returns),
        "base_maximum_drawdown": _shared_maximum_drawdown(list(pooled.base_returns)),
        "base_mean_turnover": _mean(pooled.base_turnovers),
        "base_mean_gross_exposure": _mean(pooled.base_gross_exposures),
        "base_mean_net_exposure": _mean(pooled.base_net_exposures),
        "base_forced_close_count": pooled.base_forced_close_count,
        "base_held_nothing_episode_count": pooled.base_held_nothing_episode_count,
        "base_held_nothing_net_return_rolled_forward": (
            pooled.base_held_nothing_net_return_rolled_forward
        ),
        "base_held_nothing_turnover_rolled_forward": (
            pooled.base_held_nothing_turnover_rolled_forward
        ),
        "base_fold_retained_episode_counts": list(pooled.base_fold_retained_episode_counts),
        "base_folds_with_no_retained_episodes": _folds_with_no_retained_episodes(
            pooled.base_fold_retained_episode_counts
        ),
        "adverse_total_net_return": pooled.adverse_total,
        "adverse_mean_net_return": adverse_mean,
        "adverse_median_net_return": _median(pooled.adverse_returns),
        "adverse_win_rate": _win_rate(pooled.adverse_returns),
        "adverse_maximum_drawdown": _shared_maximum_drawdown(list(pooled.adverse_returns)),
        "adverse_mean_turnover": _mean(pooled.adverse_turnovers),
        "adverse_mean_gross_exposure": _mean(pooled.adverse_gross_exposures),
        "adverse_mean_net_exposure": _mean(pooled.adverse_net_exposures),
        "adverse_forced_close_count": pooled.adverse_forced_close_count,
        "adverse_held_nothing_episode_count": pooled.adverse_held_nothing_episode_count,
        "adverse_held_nothing_net_return_rolled_forward": (
            pooled.adverse_held_nothing_net_return_rolled_forward
        ),
        "adverse_held_nothing_turnover_rolled_forward": (
            pooled.adverse_held_nothing_turnover_rolled_forward
        ),
        "adverse_fold_retained_episode_counts": list(
            pooled.adverse_fold_retained_episode_counts
        ),
        "adverse_folds_with_no_retained_episodes": _folds_with_no_retained_episodes(
            pooled.adverse_fold_retained_episode_counts
        ),
        "universe_too_small_week_count": universe_too_small_week_count,
        "positive_base_fold_count": positive_folds,
        "required_positive_fold_count": required,
        "base_bootstrap_lower": lower,
        "base_bootstrap_p_value": test.one_sided_p_value if test is not None else None,
        "bh_q_value": q_value,
        "annualised_sharpe": annual,
        "deflated_sharpe_ratio": deflated,
        "concentration": shares,
        "decision_status": status,
        "reason_codes": evidence + economic,
        **_extras_mean_field(pooled),
    }
    return record, status


def _load_object(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise PanelDecisionError(f"{path.name} must contain a JSON object")
    return document


def _int_field(record: dict[str, object], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise PanelDecisionError(f"field {key} must be an integer")
    return value


def _list_field(record: dict[str, object], key: str) -> list[object]:
    value = record.get(key)
    if not isinstance(value, list):
        raise PanelDecisionError(f"field {key} must be an array")
    return value


def _string_list_field(record: dict[str, object], key: str) -> list[str]:
    values = _list_field(record, key)
    if not all(isinstance(value, str) for value in values):
        raise PanelDecisionError(f"field {key} must contain strings")
    return [value for value in values if isinstance(value, str)]
