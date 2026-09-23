"""Tests for the weekly shadow book (spec sections 2, 4.1-4.3 and 6).

Every week here is computed from fixture captures -- no network, no venue
client, no key. The captures are built once for the module: they are
immutable, and a shadow week only reads them.
"""

import json
import shutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest

from tests.carry_fixtures import (
    build_captures,
    funding_rate,
    perp_fetch_with_a_liquidity_dip,
    small_carry_v4_config,
)
from tests.shadow_fixtures import (
    FIRST_TAIL_SUNDAY,
    SECOND_TAIL_SUNDAY,
    SHADOW_ANCHOR_SUNDAY,
    SHADOW_CANDIDATE,
    SHADOW_FAMILY_NAME,
    ShadowBookCaptures,
    build_shadow_book_captures,
    day_end_ms,
    funding_settlement_times_ms,
    write_shadow_declaration,
)
from tests.test_panel_fold_run import SYMBOLS
from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.carry_config import load_carry_family_spec
from trading_bot.carry_fold_run import (
    DecisionInputs,
    DecisionRun,
    FinalBook,
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
from trading_bot.carry_universe import select_pair_universe
from trading_bot.panel_samples import rebalance_close_times
from trading_bot.registry import MetadataRegistry
from trading_bot.shadow_book import (
    MEASUREMENT_SNAPSHOT_VERSION,
    SHADOW_ARTIFACT_KIND,
    ShadowBookError,
    ShadowWeekArtifact,
    _m_label,
    run_shadow_week,
)
from trading_bot.shadow_config import ShadowDeclarationError, load_shadow_declaration

_NANOSECONDS_PER_MILLISECOND = 1_000_000


def _close_ns(sunday: str) -> int:
    """The bar close a fixture Sunday decides at, off the fixture's calendar."""
    return day_end_ms(sunday) * _NANOSECONDS_PER_MILLISECOND


# --- the workspace ------------------------------------------------------


@pytest.fixture(scope="module")
def book_captures(tmp_path_factory: pytest.TempPathFactory) -> ShadowBookCaptures:
    """Both markets' base and two chained weekly captures, built once.

    `zipfile` stamps each archive member with the wall clock, so the clock is
    pinned while the captures are built -- exactly as `test_shadow_capture`
    pins it -- and the fixture zips stay byte-stable.
    """
    patch = pytest.MonkeyPatch()
    patch.setattr("time.time", lambda: 1_700_000_000.0)
    try:
        return build_shadow_book_captures(tmp_path_factory.mktemp("shadow-book"))
    finally:
        patch.undo()


@dataclass(frozen=True, slots=True)
class Harness:
    """One test's declaration, artifact root and registry inside the workspace."""

    captures: ShadowBookCaptures
    family_spec_path: Path
    family_spec_hash: str
    name: str

    @property
    def root(self) -> Path:
        return self.captures.root

    @property
    def artifact_root(self) -> Path:
        return self.root / "artifacts" / self.name

    @property
    def registry_path(self) -> Path:
        return self.artifact_root / "metadata-shadow.sqlite3"

    def output_path(self, sunday: str = SECOND_TAIL_SUNDAY) -> Path:
        return self.artifact_root / SHADOW_FAMILY_NAME / f"{sunday}.json"

    def refusal_path(self, sunday: str = SECOND_TAIL_SUNDAY) -> Path:
        return self.artifact_root / SHADOW_FAMILY_NAME / f"{sunday}-refused.json"

    def declare(
        self,
        *,
        family_spec_hash: str | None = None,
        family_name: str = SHADOW_FAMILY_NAME,
        candidate: str = SHADOW_CANDIDATE,
        anchor_decision_close_date: str = SHADOW_ANCHOR_SUNDAY,
        phase: str = "A",
        holdout_report_hash: str | None = None,
    ) -> Path:
        return write_shadow_declaration(
            self.root / f"{self.name}-declaration.json",
            family_spec_path=self.family_spec_path.name,
            family_spec_hash=family_spec_hash or self.family_spec_hash,
            artifact_root=f"artifacts/{self.name}",
            registry_path=f"artifacts/{self.name}/metadata-shadow.sqlite3",
            family_name=family_name,
            candidate=candidate,
            anchor_decision_close_date=anchor_decision_close_date,
            phase=phase,
            holdout_report_hash=holdout_report_hash,
        )


@pytest.fixture
def harness(book_captures: ShadowBookCaptures, request: pytest.FixtureRequest) -> Harness:
    spec_path = small_carry_v4_config(book_captures.root)
    _, spec_hash = load_carry_family_spec(spec_path)
    return Harness(
        captures=book_captures,
        family_spec_path=spec_path,
        family_spec_hash=spec_hash,
        name=str(request.node.name),
    )


def _run(
    harness: Harness,
    declaration: Path,
    *,
    sunday: str = SECOND_TAIL_SUNDAY,
    perp: Path | None = None,
    spot: Path | None = None,
    perp_base: Path | None = None,
    spot_base: Path | None = None,
    holdout_report_path: Path | None = None,
    measurement_snapshot_path: Path | None = None,
) -> ShadowWeekArtifact:
    captures = harness.captures
    return run_shadow_week(
        workspace_root=harness.root,
        declaration_path=declaration,
        perp_capture_root=perp if perp is not None else captures.perp_second,
        spot_capture_root=spot if spot is not None else captures.spot_second,
        decision_sunday=sunday,
        holdout_report_path=holdout_report_path,
        measurement_snapshot_path=measurement_snapshot_path,
        perp_base_capture_root=perp_base if perp_base is not None else captures.perp_base,
        spot_base_capture_root=spot_base if spot_base is not None else captures.spot_base,
    )


# --- reading the artifact -----------------------------------------------


def _document(path: Path) -> dict[str, object]:
    document: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    return document


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value


def _sequence(value: object) -> list[object]:
    assert isinstance(value, list)
    return value


def _text(value: object) -> str:
    assert isinstance(value, str)
    return value


def _decimal(value: object) -> Decimal:
    return Decimal(_text(value))


def _assert_no_key(value: object, forbidden: Iterable[str]) -> None:
    """No key anywhere under `value` is one of `forbidden` (spec 4.2)."""
    names = set(forbidden)
    if isinstance(value, dict):
        for key, item in value.items():
            assert key not in names, f"{key} leaked into a Phase A artifact"
            _assert_no_key(item, names)
    elif isinstance(value, list):
        for item in value:
            _assert_no_key(item, names)


# --- the inputs, loaded through the fold runner's own loader ------------


@dataclass(frozen=True, slots=True)
class Inputs:
    """One week's loaded inputs and the decisions a book is computed over.

    The loading is `carry_fold_run.load_decision_inputs`, the same call
    `run_shadow_week` and `run_carry_fold` make -- not a copy of it. A copy
    would make the "shadow = fold mechanics" pin compare two evaluations of
    two separately assembled input sets, which proves nothing about the
    loading at all. Only the decision list is derived here, because that is
    the one thing the declaration, not the loader, decides.
    """

    loaded: DecisionInputs
    decisions: list[int]


def _inputs(
    harness: Harness,
    *,
    sunday: str = SECOND_TAIL_SUNDAY,
    perp: Path | None = None,
    spot: Path | None = None,
    anchor: str = SHADOW_ANCHOR_SUNDAY,
) -> Inputs:
    perp_root = perp if perp is not None else harness.captures.perp_second
    spot_root = spot if spot is not None else harness.captures.spot_second
    close = _close_ns(sunday)
    loaded = load_decision_inputs(
        perp_root / "dataset", spot_root / "dataset", available_before_ns=close + 2
    )
    anchor_close = _close_ns(anchor)
    decisions = [
        time_ns
        for time_ns in rebalance_close_times(loaded.perp_bars)
        if anchor_close <= time_ns <= close
    ]
    return Inputs(loaded, decisions)


def _evaluate(
    harness: Harness,
    inputs: Inputs,
    *,
    decisions: list[int] | None = None,
    candidates: tuple[str, ...] = (SHADOW_CANDIDATE, "no_trade", "random_pairs"),
) -> DecisionRun:
    spec, _ = load_carry_family_spec(harness.family_spec_path)
    return evaluate_carry_decisions(
        spec,
        perp_histories=inputs.loaded.perp_histories,
        spot_histories=inputs.loaded.spot_histories,
        leg_histories=inputs.loaded.leg_histories,
        funding_by_leg=inputs.loaded.funding_by_leg,
        decisions=inputs.decisions if decisions is None else decisions,
        candidate_names=candidates,
    )


def _book_document(book: FinalBook) -> dict[str, object]:
    """`FinalBook` as the artifact records it, built from the dataclass alone."""
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
        "leg_weights": [[leg, str(weight)] for leg, weight in book.leg_weights],
    }


def _episodes(run: DecisionRun, scenario: str) -> list[dict[str, object]]:
    record = next(item for item in run.candidates if item["candidate_name"] == SHADOW_CANDIDATE)
    return [_mapping(item) for item in _sequence(_mapping(record[scenario])["episodes"])]


# --- Phase A ------------------------------------------------------------


def test_a_phase_a_week_publishes_the_fold_runners_book(harness: Harness) -> None:
    """Spec 4.1: the book of Sunday S is the fold runner's last decision.

    The artifact's `book` is field for field the `FinalBook` the same
    `evaluate_carry_decisions` call produces over the same loaded inputs --
    the pin that says a shadow week is fold mechanics and not a second
    implementation of them -- and the document seals itself.
    """
    declaration = harness.declare()

    artifact = _run(harness, declaration)

    assert artifact.status == "development_only"
    assert artifact.output_path == harness.output_path()
    document = _document(artifact.output_path)
    material = {key: value for key, value in document.items() if key != "report_hash"}
    assert content_sha256(material) == artifact.report_hash == document["report_hash"]
    run = _evaluate(harness, _inputs(harness))
    assert document["book"] == _book_document(run.final_books[SHADOW_CANDIDATE])
    assert document["decision_close_ns"] == _close_ns(SECOND_TAIL_SUNDAY)
    assert document["anchor_decision_close_ns"] == _close_ns(SHADOW_ANCHOR_SUNDAY)
    assert document["previous_decision_close_ns"] == _close_ns(FIRST_TAIL_SUNDAY)
    assert document["report_version"] == "1.0.0"
    assert document["status"] == "development_only"
    assert document["family_name"] == SHADOW_FAMILY_NAME
    assert document["family_spec_hash"] == harness.family_spec_hash
    assert document["candidate_name"] == SHADOW_CANDIDATE
    assert document["skipped"] is False
    assert document["reason_codes"] == run.reason_codes
    assert document["skipped_sample_ids"] == run.skipped_sample_ids
    assert document["measurement_snapshot_hash"] is None
    # Ruling 22: the window fields are the snapshot's, so a week that cites no
    # snapshot carries them as nulls rather than leaving them out.
    assert document["measurement_snapshot_window_start_ns"] is None
    assert document["measurement_snapshot_window_end_ns"] is None
    assert document["measurement_snapshot_first_sequence"] is None
    assert document["measurement_snapshot_last_sequence"] is None
    assert document["measurement_snapshot_rounds"] is None
    assert document["holdout_report_hash"] is None
    _, declaration_hash = load_shadow_declaration(declaration)
    assert document["declaration_hash"] == declaration_hash


def test_a_phase_a_week_carries_no_pnl_and_no_running_totals(harness: Harness) -> None:
    """Spec 4.2: Phase A omits the P&L block and the running totals entirely,
    not just their summary -- so no key anywhere in the document is one of
    them, and no episode's net return reaches a reader through a nested block
    either."""
    artifact = _run(harness, harness.declare())

    document = _document(artifact.output_path)

    assert "pnl" not in document
    assert "running_totals" not in document
    _assert_no_key(document, {"net_return", "pnl", "running_totals"})


def test_a_phase_a_week_registers_one_shadow_week_record(harness: Harness) -> None:
    """Spec 4.2: the week is registered, keyed by the declaration's family spec
    hash and its Sunday, with the artifact's own path and seal."""
    artifact = _run(harness, harness.declare())

    with MetadataRegistry(harness.registry_path) as registry:
        record = registry.get_artifact(
            uuid5(NAMESPACE_URL, f"shadow:{harness.family_spec_hash}:{SECOND_TAIL_SUNDAY}")
        )

    assert record is not None
    assert record.kind == SHADOW_ARTIFACT_KIND == "shadow_week"
    assert record.content_hash == artifact.report_hash
    assert record.relative_path == (
        f"artifacts/{harness.name}/{SHADOW_FAMILY_NAME}/{SECOND_TAIL_SUNDAY}.json"
    )


def test_the_captures_and_their_bases_are_named_by_hash(harness: Harness) -> None:
    """Spec 4.2: both markets' capture and dataset hashes and the base each
    weekly capture records, so a reader can re-find every byte the book read."""
    artifact = _run(harness, harness.declare())

    document = _document(artifact.output_path)

    captures = harness.captures
    for prefix, root, base in (
        ("", captures.perp_second, captures.perp_base),
        ("hedge_", captures.spot_second, captures.spot_base),
    ):
        manifest = _document(root / "capture-manifest.json")
        dataset = _document(root / "dataset" / "dataset-manifest.json")
        assert document[f"{prefix}capture_root_hash"] == manifest["capture_root_hash"]
        assert document[f"{prefix}dataset_root_hash"] == dataset["root_hash"]
        assert document[f"{prefix}base_capture_root_hash"] == (
            _document(base / "capture-manifest.json")["capture_root_hash"]
        )


def test_the_data_available_time_is_the_newest_tail_row(harness: Harness) -> None:
    """Ruling 2 and spec 3.3: a row's `received_time_ns` is when its payload
    became available, so the week states the newest one over both markets'
    tail rows -- the age of the freshest input any decision here read."""
    artifact = _run(harness, harness.declare())

    document = _document(artifact.output_path)

    newest = max(
        int(str(entry["received_time_ns"]))
        for root in (harness.captures.perp_second, harness.captures.spot_second)
        for row in _sequence(_document(root / "capture-manifest.json")["sources"])
        for entry in (_mapping(row),)
        if entry.get("status") == "present"
        and entry.get("kind") in ("klines_daily_tail", "fundingRate_rest")
    )
    assert document["data_available_time_ns"] == newest


def test_the_code_hash_covers_the_carry_stack_and_the_shadow_modules(
    harness: Harness,
) -> None:
    """Spec 4.2's code hash: the evaluation path plus the three modules only a
    shadow week runs, so a change to either moves the seal."""
    artifact = _run(harness, harness.declare())

    document = _document(artifact.output_path)

    root = Path("src/trading_bot")
    names = (*carry_module_names(), "shadow_book.py", "shadow_capture.py", "shadow_config.py")
    material = "".join((root / name).read_text(encoding="utf-8") for name in names)
    assert document["code_hash"] == content_sha256(material)


def test_the_books_slots_survive_the_last_episodes_forced_closes(
    harness: Harness,
) -> None:
    """The book is the state S traded on, not what survived S's episode.

    S's episode exits seven days after S -- a bar the capture cannot hold yet,
    because that week has not happened -- so the panel accounting force-closes
    every leg of the last decision and the fold runner strips those pairs from
    the state it would carry into the next decision. The runner records the
    book where the decision assembles its weights, before all that, so `slots`
    and `leg_weights` are one and the same book rather than a book and an
    empty list: every slot's two legs carry weight, and no leg belongs to a
    pair no slot names.
    """
    artifact = _run(harness, harness.declare())

    document = _document(artifact.output_path)
    book = _mapping(document["book"])

    slots = [_mapping(item) for item in _sequence(book["slots"])]
    assert slots
    legs = {_text(_sequence(item)[0]) for item in _sequence(book["leg_weights"])}
    assert legs == {
        leg
        for slot in slots
        for leg in (_text(slot["perpetual_leg"]), _text(slot["spot_leg"]))
    }
    assert len(legs) == 2 * len(slots)
    assert _sequence(document["m_labels"])


# --- the universe measurements ------------------------------------------


def test_the_universe_measurements_list_every_ranked_pair(harness: Harness) -> None:
    """Spec 4.2: every pair the universe ranked, not only the ones held, with
    its tier, its trailing one- and four-week funding and the exit rule's
    verdict -- all measured by the declaration's own functions."""
    artifact = _run(harness, harness.declare())

    document = _document(artifact.output_path)

    inputs = _inputs(harness)
    spec, _ = load_carry_family_spec(harness.family_spec_path)
    close = _close_ns(SECOND_TAIL_SUNDAY)
    snapshot = select_pair_universe(
        inputs.loaded.perp_histories,
        inputs.loaded.spot_histories,
        pairs=spec.pairs,
        decision_close_ns=close,
        rules=spec.universe,
    )
    one_week = {
        pair.pair_id: trailing_funding(
            inputs.loaded.funding_by_leg.get(f"perp:{pair.perpetual_contract_id}", ()),
            decision_close_ns=close,
            lookback_weeks=1,
        )
        for pair in snapshot.pairs
    }
    exiting = exit_rule_pairs(
        [Cohort(close, entries_for(snapshot, [pair.pair_id for pair in snapshot.pairs]), ())],
        trailing_one_week=one_week,
    )
    measurements = [_mapping(item) for item in _sequence(document["universe_measurements"])]
    assert [item["pair_id"] for item in measurements] == [p.pair_id for p in snapshot.pairs]
    assert len(measurements) == len(SYMBOLS)
    for entry, pair in zip(measurements, snapshot.pairs, strict=True):
        assert entry["tier"] == pair.tier
        assert entry["perpetual_leg"] == f"perp:{pair.perpetual_contract_id}"
        assert entry["spot_leg"] == f"spot:{pair.spot_contract_id}"
        assert _decimal(entry["trailing_funding_1w"]) == one_week[pair.pair_id]
        assert _decimal(entry["trailing_funding_4w"]) == trailing_funding(
            inputs.loaded.funding_by_leg.get(f"perp:{pair.perpetual_contract_id}", ()),
            decision_close_ns=close,
            lookback_weeks=4,
        )
        assert entry["exit_rule_exit"] is (pair.pair_id in exiting)
        assert entry["measurement"] is None
    # the fixture's one negative payer is the one pair the exit rule removes
    assert [item["pair_id"] for item in measurements if item["exit_rule_exit"]] == sorted(exiting)
    assert len(exiting) == 1


# --- the m labels -------------------------------------------------------


def test_the_m_labels_equal_a_hand_computation_from_the_fixture_funding(
    harness: Harness,
) -> None:
    """The plan's `m`: +1 when the week's funding beat the round trip over the
    hold, -1 when the pair paid out, 0 inside the band.

    Computed here from the fixture's own weekly settlement grid and the
    declared base table rather than from the module. Both weeks are run
    because the fixture's grid puts two settlements inside (2020-08-02 - 1
    week, 2020-08-02] and only one inside (2020-08-02, 2020-08-09]: the same
    pairs therefore clear a tier-two band of (10 + 5 + 2*10) / 2 / 10_000 in
    the first week and land inside it in the second, so both verdicts are
    produced by the real pipeline rather than only by arithmetic.
    """
    declaration = harness.declare()
    spec, _ = load_carry_family_spec(harness.family_spec_path)
    member = next(item for item in spec.members if item.name == SHADOW_CANDIDATE)
    captures = harness.captures
    seen: set[object] = set()

    for sunday, perp, spot, settlements in (
        (FIRST_TAIL_SUNDAY, captures.perp_first, captures.spot_first, 2),
        (SECOND_TAIL_SUNDAY, captures.perp_second, captures.spot_second, 1),
    ):
        artifact = _run(harness, declaration, sunday=sunday, perp=perp, spot=spot)

        document = _document(artifact.output_path)

        assert _settlements_in_the_week(sunday) == settlements
        labels = [_mapping(item) for item in _sequence(document["m_labels"])]
        inputs = _inputs(harness, sunday=sunday, perp=perp, spot=spot)
        previous = _evaluate(
            harness, inputs, decisions=inputs.decisions[:-1], candidates=(SHADOW_CANDIDATE,)
        ).final_books[SHADOW_CANDIDATE]
        assert [item["pair_id"] for item in labels] == [
            slot.pair_id for slot in previous.slots
        ]
        assert labels
        for entry, slot in zip(labels, previous.slots, strict=True):
            symbol = slot.pair_id.split(":")[0]
            expected = settlements * Decimal(funding_rate(symbol))
            band = (
                round_trip_cost_bps(spec.costs.base, slot.tier)
                / Decimal(member.hold_weeks)
                / Decimal(10_000)
            )
            assert _decimal(entry["trailing_one_week_funding"]) == expected
            assert _decimal(entry["cost_band"]) == band
            assert entry["m"] == (1 if expected > band else (-1 if expected < 0 else 0))
            assert entry["tier"] == slot.tier
            assert entry["entry_decision_close_ns"] == slot.entry_decision_close_ns
            seen.add(entry["m"])

    assert seen == {0, 1}


def _settlements_in_the_week(sunday: str) -> int:
    """How many fixture funding settlements fall in the week ending at `sunday`."""
    end_ms = day_end_ms(sunday)
    start_ms = end_ms - 7 * 86_400_000
    return len(
        [
            settlement_ms
            for settlement_ms in funding_settlement_times_ms(start_ms, end_ms)
            if settlement_ms > start_ms
        ]
    )


def test_an_m_label_is_a_three_way_verdict_and_absent_without_a_measurement() -> None:
    """The label's four cases at the band's edges. A week with no settlement
    measured nothing, and a label invented from that absence would teach the
    future family that silence is neutral, so it is `None`."""
    band = Decimal("0.00175")

    assert _m_label(Decimal("0.002"), cost_band=band) == 1
    assert _m_label(band, cost_band=band) == 0
    assert _m_label(Decimal(0), cost_band=band) == 0
    assert _m_label(Decimal("-0.0001"), cost_band=band) == -1
    assert _m_label(None, cost_band=band) is None


# --- refusals -----------------------------------------------------------


def test_a_second_week_for_the_same_sunday_refuses(harness: Harness) -> None:
    """Spec 4.2: a published week is immutable, so the second run refuses
    before it reads a byte of the captures and writes nothing at all."""
    declaration = harness.declare()
    first = _run(harness, declaration)

    with pytest.raises(ShadowBookError, match="already exists"):
        _run(harness, declaration)

    assert _document(first.output_path)["report_hash"] == first.report_hash
    assert not harness.refusal_path().exists()


def test_a_sunday_the_captures_do_not_reach_refuses_before_any_write(
    harness: Harness,
) -> None:
    """Spec 4.2: the captures must cover the Sunday. Week one's tail stops a
    week short of it, which is an input that does not verify -- so the run
    raises and records nothing, rather than filing a refusal about a week it
    was never handed the data for."""
    declaration = harness.declare()

    with pytest.raises(ShadowBookError, match="does not cover"):
        _run(
            harness,
            declaration,
            perp=harness.captures.perp_first,
            spot=harness.captures.spot_first,
        )

    assert not harness.artifact_root.exists()


def test_a_sunday_outside_the_anchor_window_records_a_refusal(harness: Harness) -> None:
    """Spec 6: a missing week is visible, not silent. The captures verify and
    cover this Sunday, but it falls before the declared anchor, so it is not
    among the decisions the book is computed over -- and the refusal is an
    unsealed artifact beside the week that was not written."""
    declaration = harness.declare(anchor_decision_close_date=FIRST_TAIL_SUNDAY)

    with pytest.raises(ShadowBookError, match="SHADOW_SUNDAY_NOT_IN_CAPTURE"):
        _run(harness, declaration, sunday=SHADOW_ANCHOR_SUNDAY)

    refusal = _document(harness.refusal_path(SHADOW_ANCHOR_SUNDAY))
    assert refusal["status"] == "refused"
    assert refusal["reason"] == "SHADOW_SUNDAY_NOT_IN_CAPTURE"
    assert refusal["decision_sunday"] == SHADOW_ANCHOR_SUNDAY
    assert "report_hash" not in refusal
    assert not harness.output_path(SHADOW_ANCHOR_SUNDAY).exists()


def test_a_candidate_the_family_does_not_declare_refuses(harness: Harness) -> None:
    """Spec 4.1: the book is one declared member's. A name the family does not
    carry is refused while the inputs are verified, so nothing is written."""
    declaration = harness.declare(candidate="carry_s10_l4w_h99w")

    with pytest.raises(ShadowBookError, match="declared member"):
        _run(harness, declaration)

    assert not harness.artifact_root.exists()


def test_a_family_spec_that_does_not_hash_to_the_declaration_refuses(
    harness: Harness,
) -> None:
    """Spec 6: the family spec hash is the gate. A declaration naming another
    hash is not this family's declaration."""
    declaration = harness.declare(family_spec_hash="0" * 64)

    with pytest.raises(ShadowBookError, match="family spec"):
        _run(harness, declaration)

    assert not harness.artifact_root.exists()


def test_a_family_name_the_declaration_does_not_name_refuses(harness: Harness) -> None:
    """Ruling 15: the declaration names the family it shadows, and the loaded
    spec must be that family."""
    declaration = harness.declare(family_name="funding_carry_panel_v4_measured")

    with pytest.raises(ShadowBookError, match="family"):
        _run(harness, declaration)

    assert not harness.artifact_root.exists()


def test_a_weekly_capture_without_its_base_refuses(harness: Harness) -> None:
    """Ruling 17 and spec 6: a weekly capture records the base it extends, so
    the book refuses to run without it rather than skipping the lineage
    check."""
    declaration = harness.declare()

    with pytest.raises(ShadowBookError, match="base capture"):
        run_shadow_week(
            workspace_root=harness.root,
            declaration_path=declaration,
            perp_capture_root=harness.captures.perp_second,
            spot_capture_root=harness.captures.spot_second,
            decision_sunday=SECOND_TAIL_SUNDAY,
            holdout_report_path=None,
            measurement_snapshot_path=None,
        )

    assert not harness.artifact_root.exists()


def test_a_base_that_is_not_the_recorded_one_refuses(harness: Harness) -> None:
    """Ruling 17: the base handed in must be the one the weekly capture
    records, so a lineage check against some other capture cannot pass."""
    declaration = harness.declare()

    with pytest.raises(ShadowBookError, match="base capture"):
        _run(harness, declaration, perp_base=harness.captures.perp_first)

    assert not harness.artifact_root.exists()


def test_a_weekly_capture_that_is_not_a_superset_of_its_base_refuses(
    harness: Harness, tmp_path: Path
) -> None:
    """Ruling 17 and spec 3.1: a weekly capture carries its base's rows byte
    for byte, so a base row the week no longer agrees with means a source the
    fold read has changed under the book.

    Both captures are copied and the base's manifest is edited so one row
    claims a different payload digest, then the weekly manifest is pointed at
    the edited base and resealed -- so the weekly capture still verifies and
    still records the base it was handed, and the only thing left to fail is
    the lineage itself.
    """
    base = tmp_path / "base"
    weekly = tmp_path / "weekly"
    shutil.copytree(harness.captures.perp_base, base)
    shutil.copytree(harness.captures.perp_second, weekly)
    row = _first_kline_row(base)
    altered = _reseal(base, lambda material: _set_row_hash(material, row, "f" * 64))
    _reseal(weekly, lambda material: material.__setitem__("base_capture_root_hash", altered))

    with pytest.raises(ShadowBookError, match="not a superset") as refusal:
        _run(harness, harness.declare(), perp=weekly, perp_base=base)

    assert f"klines:{row}: hash" in str(refusal.value)
    assert not harness.artifact_root.exists()


def test_a_stale_symbol_is_carried_into_the_week_and_beside_its_pair(
    harness: Harness, tmp_path: Path
) -> None:
    """Ruling 23: what a capture declared stale is sealed into the week that reads it.

    A non-empty `stale_symbols` says the base has no monthly dump for that
    symbol past the month it names, so the symbol's tail starts at the global
    cutoff and the days between are in no capture at all. Sealed into every
    manifest since Task 1 and read by nobody, that is a silent hole in exactly
    the pairs a reader would otherwise trust; the week now carries the whole
    declaration and repeats it beside the ranked pairs it touches.
    """
    perpetual = tmp_path / "perp"
    spot = tmp_path / "spot"
    shutil.copytree(harness.captures.perp_second, perpetual)
    shutil.copytree(harness.captures.spot_second, spot)
    stale_symbol = SYMBOLS[0]
    perp_stale = {"klines": {stale_symbol: "2020-05"}, "fundingRate": {stale_symbol: None}}
    spot_stale = {"klines": {stale_symbol: "2020-06"}}
    _reseal(perpetual, lambda material: material.__setitem__("stale_symbols", perp_stale))
    _reseal(spot, lambda material: material.__setitem__("stale_symbols", spot_stale))

    artifact = _run(harness, harness.declare(), perp=perpetual, spot=spot)

    document = _document(artifact.output_path)
    assert document["stale_symbols"] == {"perpetual": perp_stale, "hedge": spot_stale}
    measurements = [_mapping(item) for item in _sequence(document["universe_measurements"])]
    by_symbol = {_text(item["pair_id"]).split(":")[0]: item for item in measurements}
    assert by_symbol[stale_symbol]["stale"] == {
        "perpetual": {"klines": "2020-05", "fundingRate": None},
        "hedge": {"klines": "2020-06"},
    }
    # Every other pair is not stale, and says so with nothing rather than {}.
    assert by_symbol[SYMBOLS[1]]["stale"] == {"perpetual": None, "hedge": None}


def test_a_week_over_captures_with_nothing_stale_says_so(harness: Harness) -> None:
    artifact = _run(harness, harness.declare())

    document = _document(artifact.output_path)
    assert document["stale_symbols"] == {
        "perpetual": {"klines": {}, "fundingRate": {}},
        "hedge": {"klines": {}},
    }
    measurements = [_mapping(item) for item in _sequence(document["universe_measurements"])]
    assert all(item["stale"] == {"perpetual": None, "hedge": None} for item in measurements)


def _reseal(capture_root: Path, mutate: Callable[[dict[str, object]], None]) -> str:
    """Rewrite a capture manifest after `mutate`, resealed so it still verifies.

    A hand-edited manifest that no longer matched its own seal would be
    refused by the verification step long before the field under test is read,
    so the seal is recomputed over the mutated material -- the same move
    `tests.test_shadow_capture` makes.
    """
    document = _document(capture_root / "capture-manifest.json")
    material = {key: value for key, value in document.items() if key != "capture_root_hash"}
    mutate(material)
    capture_root_hash = content_sha256(material)
    (capture_root / "capture-manifest.json").write_bytes(
        canonical_json({**material, "capture_root_hash": capture_root_hash})
    )
    return capture_root_hash


def _first_kline_row(capture_root: Path) -> str:
    """`<symbol>:<month>` of the first monthly kline row a capture carries."""
    sources = _sequence(_document(capture_root / "capture-manifest.json")["sources"])
    row = next(
        entry
        for item in sources
        for entry in (_mapping(item),)
        if entry.get("kind") == "klines" and entry.get("status") == "present"
    )
    return f"{_text(row['symbol'])}:{_text(row['month'])}"


def _set_row_hash(material: dict[str, object], row: str, raw_sha256: str) -> None:
    symbol, month = row.split(":")
    for item in _sequence(material["sources"]):
        entry = _mapping(item)
        if entry.get("symbol") == symbol and entry.get("month") == month:
            entry["raw_sha256"] = raw_sha256


def test_a_spot_capture_in_the_perpetual_slot_refuses(harness: Harness) -> None:
    """The two captures are not interchangeable: the perpetual leg carries the
    funding and the spot leg is the hedge, so a swapped pair would evaluate a
    book that is not the one under test."""
    declaration = harness.declare()

    with pytest.raises(ShadowBookError, match="market"):
        _run(
            harness,
            declaration,
            perp=harness.captures.spot_second,
            perp_base=harness.captures.spot_base,
            spot=harness.captures.perp_second,
            spot_base=harness.captures.perp_base,
        )

    assert not harness.artifact_root.exists()


def test_a_registry_conflict_refuses_and_leaves_no_report(harness: Harness) -> None:
    """Spec 4.2: the report and its registry record are one artifact. A week
    already recorded under this declaration cannot be republished, and the
    report this call wrote is removed so the registry stays the authority."""
    declaration = harness.declare()
    artifact = _run(harness, declaration)
    artifact.output_path.unlink()

    with pytest.raises(ShadowBookError, match="registered"):
        _run(harness, declaration)

    assert not artifact.output_path.exists()
    assert _document(harness.refusal_path())["reason"] == "SHADOW_WEEK_ALREADY_REGISTERED"


# --- a skipped Sunday and a monthly capture -----------------------------


def test_a_skipped_sunday_publishes_an_empty_book(tmp_path: Path) -> None:
    """Ruling 16: a Sunday whose universe was too small holds nothing, so the
    week is published with an empty book and says outright that it was
    skipped -- never a refusal, and never a flat book that reads as a
    decision. The fixture's liquidity dip empties the universe at 2020-04-26
    alone."""
    perp, spot = build_captures(tmp_path, perp_fetch_function=perp_fetch_with_a_liquidity_dip)
    declaration = _monthly_declaration(tmp_path)

    artifact = run_shadow_week(
        workspace_root=tmp_path,
        declaration_path=declaration,
        perp_capture_root=perp,
        spot_capture_root=spot,
        decision_sunday="2020-04-26",
        holdout_report_path=None,
        measurement_snapshot_path=None,
    )

    document = _document(artifact.output_path)
    assert document["skipped"] is True
    assert document["book"] == {
        "decision_close_ns": _close_ns("2020-04-26"),
        "slots": [],
        "leg_weights": [],
    }
    assert document["universe_measurements"] == []
    assert f"BINANCE_UM:{_close_ns('2020-04-26')}:w1" in _sequence(
        document["skipped_sample_ids"]
    )
    assert "SKIPPED_WEEK_EXIT_COST_UNCHARGED" in _sequence(document["reason_codes"])


def test_a_monthly_capture_needs_no_base(tmp_path: Path) -> None:
    """A capture that records no base extends none, so none is asked for: the
    book runs on the monthly captures the folds themselves read."""
    perp, spot = build_captures(tmp_path)
    declaration = _monthly_declaration(tmp_path)

    artifact = run_shadow_week(
        workspace_root=tmp_path,
        declaration_path=declaration,
        perp_capture_root=perp,
        spot_capture_root=spot,
        decision_sunday="2020-07-26",
        holdout_report_path=None,
        measurement_snapshot_path=None,
    )

    document = _document(artifact.output_path)
    assert document["base_capture_root_hash"] is None
    assert document["hedge_base_capture_root_hash"] is None
    assert document["decision_close_ns"] == _close_ns("2020-07-26")


def test_phase_b_reports_no_pnl_when_the_previous_sunday_was_skipped(
    tmp_path: Path,
) -> None:
    """A skipped S-1 evaluated no episode, so there is no P&L to state -- and
    a zeroed block would read as a week that traded and earned nothing. The
    running totals still carry every episode the run did evaluate before S."""
    perp, spot = build_captures(tmp_path, perp_fetch_function=perp_fetch_with_a_liquidity_dip)
    _, spec_hash = load_carry_family_spec(small_carry_v4_config(tmp_path))
    report, report_hash = _holdout_report(
        tmp_path / "holdout.json", family_spec_hash=spec_hash
    )
    declaration = _monthly_declaration(
        tmp_path, phase="B", holdout_report_hash=report_hash
    )

    artifact = run_shadow_week(
        workspace_root=tmp_path,
        declaration_path=declaration,
        perp_capture_root=perp,
        spot_capture_root=spot,
        decision_sunday="2020-05-03",
        holdout_report_path=report,
        measurement_snapshot_path=None,
    )

    document = _document(artifact.output_path)
    skipped_close = _close_ns("2020-04-26")
    assert document["previous_decision_close_ns"] == skipped_close
    assert f"BINANCE_UM:{skipped_close}:w1" in _sequence(document["skipped_sample_ids"])
    assert document["skipped"] is False
    assert document["pnl"] is None
    totals = _mapping(document["running_totals"])
    assert totals["through_decision_close_ns"] == skipped_close
    count = totals["episode_count"]
    assert isinstance(count, int) and count > 0


def _monthly_declaration(
    root: Path, *, phase: str = "A", holdout_report_hash: str | None = None
) -> Path:
    spec_path = small_carry_v4_config(root)
    _, spec_hash = load_carry_family_spec(spec_path)
    return write_shadow_declaration(
        root / "declaration.json",
        family_spec_path=spec_path.name,
        family_spec_hash=spec_hash,
        artifact_root="artifacts",
        registry_path="artifacts/metadata-shadow.sqlite3",
        phase=phase,
        holdout_report_hash=holdout_report_hash,
    )


# --- Phase B ------------------------------------------------------------


def _holdout_report(
    path: Path,
    *,
    family_spec_hash: str,
    candidate: str = SHADOW_CANDIDATE,
    verdict: str = "holdout_confirmed",
    seal: str | None = None,
) -> tuple[Path, str]:
    """A hand-sealed holdout document with the keys Phase B reads (spec 2).

    `carry_holdout_run` writes the real one; only these keys gate a shadow
    week, so sealing them here keeps the gate under test rather than the
    holdout runner.
    """
    material: dict[str, object] = {
        "report_version": "1.0.0",
        "status": "development_only",
        "holdout": True,
        "family_name": SHADOW_FAMILY_NAME,
        "family_spec_hash": family_spec_hash,
        "candidate_name": candidate,
        "confirmation": {"verdict": verdict},
    }
    report_hash = content_sha256(material)
    path.write_bytes(canonical_json({**material, "report_hash": seal or report_hash}))
    return path, seal or report_hash


def test_phase_b_without_a_holdout_report_refuses(harness: Harness) -> None:
    """Spec 2: Phase B exists only after `holdout_confirmed`, so a Phase B week
    that is handed no holdout report has nothing to stand on."""
    declaration = harness.declare(phase="B", holdout_report_hash="a" * 64)

    with pytest.raises(ShadowBookError, match="holdout report"):
        _run(harness, declaration)

    assert not harness.artifact_root.exists()


def test_phase_b_refuses_a_holdout_that_failed(harness: Harness, tmp_path: Path) -> None:
    """Spec 2: `holdout_failed` ends the family. A shadow week may not be
    published on a holdout that did not confirm, whatever its seal says."""
    report, report_hash = _holdout_report(
        tmp_path / "holdout.json",
        family_spec_hash=harness.family_spec_hash,
        verdict="holdout_failed",
    )
    declaration = harness.declare(phase="B", holdout_report_hash=report_hash)

    with pytest.raises(ShadowBookError, match="holdout_confirmed"):
        _run(harness, declaration, holdout_report_path=report)

    assert not harness.artifact_root.exists()


def test_phase_b_refuses_a_holdout_whose_seal_does_not_recompute(
    harness: Harness, tmp_path: Path
) -> None:
    """The holdout document is a sealed artifact, so Phase B recomputes its
    `report_hash` over the rest rather than trusting the field."""
    report, report_hash = _holdout_report(
        tmp_path / "holdout.json", family_spec_hash=harness.family_spec_hash, seal="b" * 64
    )
    declaration = harness.declare(phase="B", holdout_report_hash=report_hash)

    with pytest.raises(ShadowBookError, match="seal"):
        _run(harness, declaration, holdout_report_path=report)

    assert not harness.artifact_root.exists()


def test_phase_b_refuses_a_holdout_opened_for_another_candidate(
    harness: Harness, tmp_path: Path
) -> None:
    """The confirmed candidate is the one the book is run for; a holdout
    opened for a sibling confirms nothing about this one."""
    report, report_hash = _holdout_report(
        tmp_path / "holdout.json",
        family_spec_hash=harness.family_spec_hash,
        candidate="carry_s10_l4w_h13w",
    )
    declaration = harness.declare(phase="B", holdout_report_hash=report_hash)

    with pytest.raises(ShadowBookError, match="candidate"):
        _run(harness, declaration, holdout_report_path=report)

    assert not harness.artifact_root.exists()


def test_a_phase_a_week_refuses_a_holdout_report(harness: Harness, tmp_path: Path) -> None:
    """Spec 2: Phase A runs while the holdout is unread. A holdout report handed
    to a Phase A week would either be ignored -- silently -- or read, which is
    the one thing Phase A exists not to do."""
    report, _ = _holdout_report(
        tmp_path / "holdout.json", family_spec_hash=harness.family_spec_hash
    )
    declaration = harness.declare()

    with pytest.raises(ShadowBookError, match="Phase A"):
        _run(harness, declaration, holdout_report_path=report)

    assert not harness.artifact_root.exists()


def test_phase_b_publishes_the_previous_sundays_pnl(harness: Harness, tmp_path: Path) -> None:
    """Spec 4.2: the P&L of S-1 under both cost tables -- net, funding
    collected, trading cost and forced closes -- is the episode the same run
    evaluated for that Sunday, and the running totals are its episodes since
    the anchor. S's own episode is the week ahead and is never summed."""
    report, report_hash = _holdout_report(
        tmp_path / "holdout.json", family_spec_hash=harness.family_spec_hash
    )
    declaration = harness.declare(phase="B", holdout_report_hash=report_hash)

    artifact = _run(harness, declaration, holdout_report_path=report)

    assert artifact.status == "shadow"
    document = _document(artifact.output_path)
    assert document["status"] == "shadow"
    assert document["holdout_report_hash"] == report_hash
    inputs = _inputs(harness)
    run = _evaluate(harness, inputs)
    previous_close = inputs.decisions[-2]
    pnl = _mapping(document["pnl"])
    totals = _mapping(document["running_totals"])
    assert pnl["decision_close_ns"] == previous_close
    assert totals["from_decision_close_ns"] == inputs.decisions[0]
    assert totals["through_decision_close_ns"] == previous_close
    assert totals["episode_count"] == len(inputs.decisions) - 1
    for scenario in ("base", "adverse"):
        episodes = _episodes(run, scenario)
        previous = next(
            item
            for item in episodes
            if _text(item["sample_id"]).split(":")[1] == str(previous_close)
        )
        earlier = [
            item
            for item in episodes
            if int(_text(item["sample_id"]).split(":")[1]) < inputs.decisions[-1]
        ]
        _assert_pnl(_mapping(pnl[scenario]), [previous])
        _assert_pnl(_mapping(totals[scenario]), earlier)


def test_a_week_on_the_anchor_itself_has_no_previous_sunday(
    harness: Harness, tmp_path: Path
) -> None:
    """S == A: the run holds one decision, so there is nothing before it.

    Spec 4.1 never evaluates a decision before the anchor, so the first week a
    declaration can produce has no S-1 at all: no previous close, no slot to
    label -- the labels are what the week just past paid the slots held at
    S-1, and there were none -- and in Phase B a P&L block and running totals
    that are explicitly null rather than zeroed. The two phases run on
    different Sundays so each writes its own immutable path.
    """
    phase_a = _run(harness, harness.declare(anchor_decision_close_date=SECOND_TAIL_SUNDAY))

    document = _document(phase_a.output_path)
    assert document["previous_decision_close_ns"] is None
    assert document["anchor_decision_close_ns"] == document["decision_close_ns"]
    assert document["m_labels"] == []
    assert document["skipped"] is False
    assert "pnl" not in document
    assert "running_totals" not in document

    report, report_hash = _holdout_report(
        tmp_path / "holdout.json", family_spec_hash=harness.family_spec_hash
    )
    phase_b = _run(
        harness,
        harness.declare(
            anchor_decision_close_date=FIRST_TAIL_SUNDAY,
            phase="B",
            holdout_report_hash=report_hash,
        ),
        sunday=FIRST_TAIL_SUNDAY,
        perp=harness.captures.perp_first,
        spot=harness.captures.spot_first,
        holdout_report_path=report,
    )

    document = _document(phase_b.output_path)
    assert document["previous_decision_close_ns"] is None
    assert document["m_labels"] == []
    assert document["pnl"] is None
    assert document["running_totals"] is None


def _assert_pnl(block: dict[str, object], episodes: list[dict[str, object]]) -> None:
    """The block's five money fields and two counts are sums over `episodes`."""
    extras = [_mapping(item["extras"]) for item in episodes]
    spot = _sum(extras, "spot_trading_cost")
    perpetual = _sum(extras, "perpetual_trading_cost")
    assert _decimal(block["net_return"]) == _sum(episodes, "net_return")
    assert _decimal(block["funding_collected"]) == _sum(extras, "funding_collected")
    assert _decimal(block["spot_trading_cost"]) == spot
    assert _decimal(block["perpetual_trading_cost"]) == perpetual
    assert _decimal(block["trading_cost"]) == spot + perpetual
    assert block["forced_spot_legs"] == int(_sum(extras, "forced_spot_legs"))
    assert block["forced_perpetual_legs"] == int(_sum(extras, "forced_perpetual_legs"))


def _sum(records: list[dict[str, object]], key: str) -> Decimal:
    total = Decimal(0)
    for record in records:
        value = record[key]
        assert isinstance(value, Decimal)
        total += value
    return total


# --- the measurement snapshot -------------------------------------------


_WEEK_NS = 7 * 86_400_000_000_000


def _snapshot(
    path: Path,
    *,
    perpetuals: dict[str, object],
    spot: dict[str, object],
    seal: str | None = None,
    window_end_ns: int | None = None,
    sunday: str = SECOND_TAIL_SUNDAY,
) -> tuple[Path, str]:
    """A hand-sealed measurement snapshot carrying Ruling 1a's keys.

    Its window ends at the decision close of the Sunday the fixture weeks are
    run for, because a week cites a reading of its own week or none at all
    (ruling 22).
    """
    ends_ns = _close_ns(sunday) if window_end_ns is None else window_end_ns
    material: dict[str, object] = {
        "version": MEASUREMENT_SNAPSHOT_VERSION,
        "spec_hash": "1" * 64,
        "chain_head_hash": "2" * 64,
        "window_start_ns": ends_ns - _WEEK_NS + 1,
        "window_end_ns": ends_ns,
        "first_sequence": 1,
        "last_sequence": 2,
        "rounds": 2,
        "snapshot_rounds": 1,
        "failures": {},
        "excluded": {},
        "perpetuals": perpetuals,
        "spot": spot,
        "cost_instruments": {},
        "cost_tiers": {},
        "tier_floor_count": 100,
    }
    content_hash = content_sha256(material)
    path.write_bytes(canonical_json({**material, "content_hash": seal or content_hash}))
    return path, seal or content_hash


_MEASURED_SYMBOL = SYMBOLS[0]
_PERPETUAL_MEASUREMENT: dict[str, object] = {
    "premium_rounds": 9_914,
    "book_rounds": 9_910,
    "mean_last_funding_rate": "0.00007431",
    "last_funding_rate": "0.00010000",
    "mean_basis_bps": "1.204000",
    "mean_spread_bps": "0.512000",
}
_SPOT_MEASUREMENT: dict[str, object] = {"book_rounds": 9_912, "mean_spread_bps": "0.803000"}


def test_a_measurement_snapshot_is_carried_per_pair(
    harness: Harness, tmp_path: Path
) -> None:
    """Spec 4.2: the week cites the snapshot by hash and reports the stream's
    mean premium, basis and spread beside every ranked pair. A symbol the
    window never carried is absent from the snapshot's maps, which is not a
    measurement of zero -- it is written out as nulls."""
    snapshot, content_hash = _snapshot(
        tmp_path / "snapshot.json",
        perpetuals={_MEASURED_SYMBOL: _PERPETUAL_MEASUREMENT},
        spot={_MEASURED_SYMBOL: _SPOT_MEASUREMENT},
    )

    artifact = _run(harness, harness.declare(), measurement_snapshot_path=snapshot)

    document = _document(artifact.output_path)
    assert document["measurement_snapshot_hash"] == content_hash
    measurements = [_mapping(item) for item in _sequence(document["universe_measurements"])]
    by_symbol = {_text(item["pair_id"]).split(":")[0]: item for item in measurements}
    measured = _mapping(by_symbol[_MEASURED_SYMBOL]["measurement"])
    assert measured["perpetual_symbol"] == _MEASURED_SYMBOL
    assert measured["spot_symbol"] == _MEASURED_SYMBOL
    assert measured["perpetual"] == _PERPETUAL_MEASUREMENT
    assert measured["spot"] == _SPOT_MEASUREMENT
    absent = _mapping(by_symbol[SYMBOLS[1]]["measurement"])
    assert absent["perpetual"] == dict.fromkeys(_PERPETUAL_MEASUREMENT)
    assert absent["spot"] == dict.fromkeys(_SPOT_MEASUREMENT)


def test_the_cited_snapshots_window_is_sealed_into_the_week(
    harness: Harness, tmp_path: Path
) -> None:
    """Ruling 22: the artifact records which rounds the reading it cites is over.

    A hash alone says which document was read, not what it covers, so a reader
    of the week would have to open the snapshot to find out whether the
    numbers beside every pair are this week's at all.
    """
    snapshot, content_hash = _snapshot(tmp_path / "snapshot.json", perpetuals={}, spot={})

    artifact = _run(harness, harness.declare(), measurement_snapshot_path=snapshot)

    document = _document(artifact.output_path)
    close_ns = _close_ns(SECOND_TAIL_SUNDAY)
    assert document["measurement_snapshot_hash"] == content_hash
    assert document["measurement_snapshot_window_start_ns"] == close_ns - _WEEK_NS + 1
    assert document["measurement_snapshot_window_end_ns"] == close_ns
    assert document["measurement_snapshot_first_sequence"] == 1
    assert document["measurement_snapshot_last_sequence"] == 2
    assert document["measurement_snapshot_rounds"] == 2


@pytest.mark.parametrize(
    ("label", "window_end_ns"),
    [
        # Last week's reading: its window ends exactly one week before this
        # decision, which is the first nanosecond outside this week.
        ("last week", _close_ns(SECOND_TAIL_SUNDAY) - _WEEK_NS),
        # A reading that runs past the decision it is cited for.
        ("after the decision", _close_ns(SECOND_TAIL_SUNDAY) + 1),
    ],
)
def test_a_snapshot_that_is_not_this_weeks_is_a_recorded_refusal(
    harness: Harness, tmp_path: Path, label: str, window_end_ns: int
) -> None:
    """Ruling 22: a week cites a reading of its own week, and says so if it cannot.

    Version and seal alone let any snapshot of the stream be cited by any
    week -- last week's, or one taken days after the decision -- and every
    number beside every pair would then be a measurement of a different week
    with nothing in the document saying so.
    """
    snapshot, _ = _snapshot(
        tmp_path / "snapshot.json", perpetuals={}, spot={}, window_end_ns=window_end_ns
    )

    with pytest.raises(ShadowBookError, match="SHADOW_SNAPSHOT_NOT_THIS_WEEK"):
        _run(harness, harness.declare(), measurement_snapshot_path=snapshot)

    assert not harness.output_path().exists()
    refusal = _document(harness.refusal_path())
    assert refusal["reason"] == "SHADOW_SNAPSHOT_NOT_THIS_WEEK"
    assert refusal["decision_sunday"] == SECOND_TAIL_SUNDAY
    assert str(window_end_ns) in _text(refusal["detail"])


def test_a_measurement_snapshot_whose_seal_does_not_recompute_refuses(
    harness: Harness, tmp_path: Path
) -> None:
    """The snapshot is a receipt, so the week recomputes its `content_hash`
    over the rest and refuses a document that does not seal itself."""
    snapshot, _ = _snapshot(
        tmp_path / "snapshot.json", perpetuals={}, spot={}, seal="c" * 64
    )

    with pytest.raises(ShadowBookError, match="seal"):
        _run(harness, harness.declare(), measurement_snapshot_path=snapshot)

    assert not harness.artifact_root.exists()


def test_the_measurement_snapshot_version_is_the_journals() -> None:
    """The book names the document version it reads; the journal writes it.
    Pinned so the two cannot drift apart without a test saying so."""
    from trading_bot.binance_measurement_journal import MEASUREMENT_SNAPSHOT_VERSION as written

    assert MEASUREMENT_SNAPSHOT_VERSION == written == "binance-measurement-snapshot/1.0.0"


# --- the declaration ----------------------------------------------------


def test_the_shipped_shadow_declaration_names_the_measured_v4_family() -> None:
    """`configs/shadow-carry-v4.json` is P1.24's declaration: the measured v4
    family by hash, the P1.33 release candidate, the anchor spec 4.1 fixes and
    Phase A until the holdout is read."""
    declaration, declaration_hash = load_shadow_declaration(Path("configs/shadow-carry-v4.json"))

    spec, spec_hash = load_carry_family_spec(Path(declaration.family_spec_path))
    assert declaration.family_spec_path == "configs/funding-carry-panel-v4-measured.json"
    assert declaration.family_spec_hash == spec_hash
    assert declaration.family_name == spec.family_name == "funding_carry_panel_v4_measured"
    assert declaration.candidate == "carry_s10_l4w_h26w_exit"
    assert declaration.candidate in [member.name for member in spec.members]
    assert declaration.controls == ("no_trade", "random_pairs")
    assert declaration.anchor_decision_close_date == "2026-09-13"
    assert declaration.phase == "A"
    assert declaration.holdout_report_hash is None
    assert declaration.artifact_root == "artifacts/shadow"
    assert declaration.registry_path == "artifacts/shadow/metadata-shadow.sqlite3"
    assert len(declaration_hash) == 64


def test_a_phase_b_declaration_without_a_holdout_hash_is_refused(tmp_path: Path) -> None:
    """Spec 2: Phase B is what a confirmed holdout authorises, so a Phase B
    declaration that names no holdout report is not a declaration."""
    path = write_shadow_declaration(
        tmp_path / "declaration.json",
        family_spec_path="spec.json",
        family_spec_hash="0" * 64,
        artifact_root="artifacts",
        registry_path="artifacts/registry.sqlite3",
        phase="B",
    )

    with pytest.raises(ShadowDeclarationError):
        load_shadow_declaration(path)


def test_an_anchor_that_is_not_a_sunday_is_refused(tmp_path: Path) -> None:
    """Spec 4.1: the anchor is a Sunday, because every decision is one."""
    path = write_shadow_declaration(
        tmp_path / "declaration.json",
        family_spec_path="spec.json",
        family_spec_hash="0" * 64,
        artifact_root="artifacts",
        registry_path="artifacts/registry.sqlite3",
        anchor_decision_close_date="2026-09-14",
    )

    with pytest.raises(ShadowDeclarationError):
        load_shadow_declaration(path)


def test_a_decision_day_that_is_not_a_sunday_is_refused(harness: Harness) -> None:
    """The decision is the Sunday close (spec 3.3); any other day is not a
    decision this family makes."""
    with pytest.raises(ShadowBookError, match="Sunday"):
        _run(harness, harness.declare(), sunday="2020-08-10")

    assert not harness.artifact_root.exists()
