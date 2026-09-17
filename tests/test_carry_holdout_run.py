"""Tests for the single-use final holdout read (protocol 16.2, spec sections 2-6).

The chain is the real one: two fixture capture pairs -- an *original* pair over
the seven `MONTHS` the rest of the carry fixtures use, and an *extended* pair
over those seven plus `EXTRA_MONTH`, built from exactly the same fake fetches so
every original source row reappears byte-identical -- then the reduced-geometry
walk-forward manifest and both carry folds on the originals, and a hand-sealed
decision document naming those two fold reports. The holdout's last decision
falls on the last Sunday of the original capture and its exit falls into
`EXTRA_MONTH`, which only the extended captures carry: exactly the situation
spec section 2 exists for.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest

from tests.carry_fixtures import funding_csv, small_carry_v4_config, spot_kline_csv
from tests.test_panel_fold_run import (
    MONTH_DAYS,
    MONTH_START_DAY,
    MONTHS,
    SYMBOLS,
    kline_csv,
    zip_bytes,
)
from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.capture_lineage import verify_capture_superset
from trading_bot.carry_fold_run import run_carry_fold
from trading_bot.carry_holdout_run import (
    CONCENTRATION_LIMIT,
    HOLDOUT_ARTIFACT_KIND,
    MAX_SKIPPED_HOLDOUT_DECISIONS,
    CarryHoldoutArtifact,
    CarryHoldoutError,
    _coverage_month,
    derive_candidate,
    run_carry_holdout,
)
from trading_bot.cli import main
from trading_bot.panel_capture import PanelPayload, PanelSourceAbsent, capture_panel
from trading_bot.panel_config import load_family_spec
from trading_bot.panel_samples import publish_panel_walk_forward
from trading_bot.registry import MetadataRegistry

# The eighth month the extended captures add. `kline_csv`/`funding_csv` read the
# month tables of `tests.test_panel_fold_run` by key at call time, and those
# tables stop where `MONTHS` stops; registering the eighth month here is purely
# additive -- no existing key moves and `MONTHS` itself is untouched -- so the
# extended capture is built by exactly the fetches the original one is built by.
EXTRA_MONTH = "2020-08"
EXTENDED_MONTHS = (*MONTHS, EXTRA_MONTH)
MONTH_START_DAY.setdefault(EXTRA_MONTH, MONTH_START_DAY["2020-07"] + MONTH_DAYS["2020-07"])
MONTH_DAYS.setdefault(EXTRA_MONTH, 31)

# The two v4 members the hand-sealed decision declares eligible, with adverse
# totals chosen so the *second* one wins the derivation -- a candidate that is
# derived rather than simply the first name on the list.
LOSING_MEMBER = "carry_s10_l4w_h13w"
WINNING_MEMBER = "carry_s10_l4w_h13w_exit"
DECISION_BASE_MEAN = "0.00002"
DECISION_BOOTSTRAP_LOWER = "-0.0005"


def _payload(url: str, name: str, text: str) -> PanelPayload:
    return PanelPayload(url=url, raw_bytes=zip_bytes(name, text), received_time_ns=1)


def _symbol_and_month(url: str) -> tuple[str, str]:
    symbol = next(item for item in SYMBOLS if f"/{item}/" in url or f"/{item}-" in url)
    month = next(item for item in EXTENDED_MONTHS if item in url)
    return symbol, month


def perp_fetch(url: str) -> PanelPayload:
    """`tests.carry_fixtures.perp_fetch` widened to `EXTENDED_MONTHS`."""
    if "/daily/klines/" in url:
        raise PanelSourceAbsent("404: no daily dump")
    symbol, month = _symbol_and_month(url)
    if "fundingRate" in url:
        return _payload(url, "f.csv", funding_csv(symbol, month))
    return _payload(url, "k.csv", kline_csv(symbol, month))


def spot_fetch(url: str) -> PanelPayload:
    """`tests.carry_fixtures.spot_fetch` widened to `EXTENDED_MONTHS`."""
    if "/daily/klines/" in url:
        raise PanelSourceAbsent("404: no daily dump")
    symbol, month = _symbol_and_month(url)
    return _payload(url, "k.csv", spot_kline_csv(symbol, month))


@dataclass(frozen=True, slots=True)
class Chain:
    """Everything the fixture chain published, ready for a holdout read."""

    root: Path
    original_perp: Path
    original_spot: Path
    extended_perp: Path
    extended_spot: Path
    manifest: Path
    config: Path
    fold_reports: tuple[Path, ...]
    decision: Path
    decision_report_hash: str


def _capture(root: Path, name: str, months: tuple[str, ...], market: str) -> Path:
    fetch = perp_fetch if market == "um" else spot_fetch
    return capture_panel(
        workspace_root=root, output_directory=root / name, reserve_bytes=0,
        symbols=SYMBOLS, months=months, fetch=fetch, market=market,
    ).capture_root


def _seal_decision(
    path: Path,
    *,
    manifest: dict[str, object],
    fold_report_hashes: tuple[str, ...],
    eligible: tuple[str, ...],
) -> str:
    """Write a decision document sealed exactly as `panel_decision` seals one.

    Only the fields the holdout read binds to are stated; a real decision
    carries the full pooled record, which the read never looks at.
    """
    material: dict[str, object] = {
        "decision_version": "1.0.0",
        "status": "development_only",
        "family_name": str(manifest["family_name"]),
        "family_spec_hash": str(manifest["family_spec_hash"]),
        "split_manifest_hash": str(manifest["split_manifest_hash"]),
        "source_report_hashes": list(fold_report_hashes),
        "members": [
            {
                "candidate_name": LOSING_MEMBER,
                "adverse_total_net_return": "-0.02",
                "base_mean_net_return": "0.00001",
                "base_bootstrap_lower": "-0.0009",
            },
            {
                "candidate_name": WINNING_MEMBER,
                "adverse_total_net_return": "-0.01",
                "base_mean_net_return": DECISION_BASE_MEAN,
                "base_bootstrap_lower": DECISION_BOOTSTRAP_LOWER,
            },
        ],
        "eligible_member_names": list(eligible),
        "decision_status": (
            "eligible_member_available" if eligible else "no_eligible_member"
        ),
    }
    report_hash = content_sha256(material)
    document = dict(material)
    document["report_hash"] = report_hash
    path.write_bytes(canonical_json(document))
    return report_hash


def _document(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def _dict(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value


def _dicts(value: object) -> list[dict[str, object]]:
    assert isinstance(value, list)
    return [_dict(item) for item in value]


def _strings(value: object) -> list[str]:
    assert isinstance(value, list)
    assert all(isinstance(item, str) for item in value)
    return [str(item) for item in value]


def _decimal(value: object) -> Decimal:
    assert isinstance(value, str)
    return Decimal(value)


@pytest.fixture(scope="module")
def chain(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Chain]:
    """Build both capture pairs, the manifest, both fold reports and the decision.

    Module-scoped because every test reads the same published inputs and only
    the output and the registry differ; the wall clock is pinned while the
    captures are built because `zipfile` stamps each archive member with
    `time.localtime(time.time())`, so two runs over the same (symbol, month)
    produce the identical `raw_sha256` the superset check compares only if they
    do not straddle that clock tick (`tests/test_capture_lineage.py`).
    """
    root = tmp_path_factory.mktemp("holdout")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("time.time", lambda: 1_700_000_000.0)
        original_perp = _capture(root, "original-perp", MONTHS, "um")
        original_spot = _capture(root, "original-spot", MONTHS, "spot")
        extended_perp = _capture(root, "extended-perp", EXTENDED_MONTHS, "um")
        extended_spot = _capture(root, "extended-spot", EXTENDED_MONTHS, "spot")
    config = small_carry_v4_config(root)
    spec, spec_hash = load_family_spec(config)
    manifest_path = root / "manifest.json"
    publish_panel_walk_forward(
        original_perp, output_path=manifest_path, spec=spec,
        family_spec_hash=spec_hash, hedge_capture_root=original_spot,
    )
    manifest = _document(manifest_path)
    reports: list[Path] = []
    hashes: list[str] = []
    for fold in _dicts(manifest["folds"]):
        index = fold["fold_index"]
        assert isinstance(index, int)
        artifact = run_carry_fold(
            original_perp, original_spot, manifest_path=manifest_path,
            family_spec_path=config, output_path=root / f"fold{index}.json",
            registry_path=root / "folds.sqlite3", fold_index=index,
        )
        reports.append(artifact.output_path)
        hashes.append(artifact.report_hash)
    decision = root / "decision.json"
    decision_hash = _seal_decision(
        decision, manifest=manifest, fold_report_hashes=tuple(hashes),
        eligible=(LOSING_MEMBER, WINNING_MEMBER),
    )
    yield Chain(
        root=root,
        original_perp=original_perp, original_spot=original_spot,
        extended_perp=extended_perp, extended_spot=extended_spot,
        manifest=manifest_path, config=config,
        fold_reports=tuple(reports), decision=decision,
        decision_report_hash=decision_hash,
    )


def _candidate(document: dict[str, object], name: str) -> dict[str, object]:
    return next(
        item for item in _dicts(document["candidates"]) if item["candidate_name"] == name
    )


def _case(chain: Chain, name: str) -> Path:
    directory = chain.root / name
    directory.mkdir(exist_ok=True)
    return directory


def _read(
    chain: Chain,
    directory: Path,
    *,
    perp: Path | None = None,
    spot: Path | None = None,
    original_perp: Path | None = None,
    original_spot: Path | None = None,
    decision: Path | None = None,
    output_name: str = "holdout.json",
) -> CarryHoldoutArtifact:
    return run_carry_holdout(
        perp if perp is not None else chain.extended_perp,
        spot if spot is not None else chain.extended_spot,
        original_perp_capture_root=(
            original_perp if original_perp is not None else chain.original_perp
        ),
        original_spot_capture_root=(
            original_spot if original_spot is not None else chain.original_spot
        ),
        manifest_path=chain.manifest,
        family_spec_path=chain.config,
        decision_path=decision if decision is not None else chain.decision,
        fold_report_paths=chain.fold_reports,
        output_path=directory / output_name,
        registry_path=directory / "registry.sqlite3",
    )


# --- derive_candidate -------------------------------------------------------


def _synthetic_decision(*totals: tuple[str, str]) -> dict[str, object]:
    return {
        "members": [
            {"candidate_name": name, "adverse_total_net_return": total}
            for name, total in totals
        ],
        "eligible_member_names": [name for name, _ in totals],
    }


def _synthetic_fold_report(*costs: tuple[str, str]) -> dict[str, object]:
    return {
        "candidates": [
            {"candidate_name": name, "adverse": {"uncharged_final_exit_cost": cost}}
            for name, cost in costs
        ]
    }


def test_derive_candidate_ranks_by_adverse_total_after_the_uncharged_exit() -> None:
    """Spec section 3: the adverse total *after* the exit a slot book never paid.

    `alpha` has the higher adverse total and `beta` the cheaper unpaid exit, so
    the ranking reverses once the cost is subtracted: -0.01 - 0.02 = -0.03
    against -0.02 - 0.005 = -0.025.
    """
    decision = _synthetic_decision(("alpha", "-0.01"), ("beta", "-0.02"))
    reports = [
        _synthetic_fold_report(("alpha", "0.015"), ("beta", "0.004")),
        _synthetic_fold_report(("alpha", "0.005"), ("beta", "0.001")),
    ]
    assert derive_candidate(decision, reports) == ("beta", Decimal("-0.025"))


def test_derive_candidate_breaks_a_tie_by_the_decisions_member_order() -> None:
    decision = _synthetic_decision(("alpha", "-0.01"), ("beta", "-0.01"))
    reports = [_synthetic_fold_report(("alpha", "0.002"), ("beta", "0.002"))]
    assert derive_candidate(decision, reports) == ("alpha", Decimal("-0.012"))


def test_derive_candidate_treats_a_cohort_family_as_a_zero_uncharged_cost() -> None:
    """A cohort family reports an adverse scenario with no uncharged exit cost."""
    decision = _synthetic_decision(("alpha", "-0.01"), ("beta", "-0.02"))
    reports: list[dict[str, object]] = [
        {
            "candidates": [
                {"candidate_name": "alpha", "adverse": {"total_net_return": "-0.005"}},
                {"candidate_name": "beta", "adverse": {"total_net_return": "-0.009"}},
            ]
        }
    ]
    assert derive_candidate(decision, reports) == ("alpha", Decimal("-0.01"))


def test_derive_candidate_refuses_a_decision_with_no_eligible_member() -> None:
    decision = _synthetic_decision(("alpha", "-0.01"))
    decision["eligible_member_names"] = []
    with pytest.raises(CarryHoldoutError):
        derive_candidate(decision, [_synthetic_fold_report(("alpha", "0"))])


# --- the coverage helper ----------------------------------------------------


def test_coverage_month_reads_a_discovered_capture_and_a_declared_one() -> None:
    """Spec section 2: the extended capture must reach the holdout last exit.

    A capture built from an explicit month list states `months`; one built by
    discovery (and every repair of one) states `months: null` and carries the
    per-symbol `discovered_months` instead, so the latest covered month has to
    be read off whichever of the two the manifest actually has.
    """
    assert _coverage_month({"months": ["2020-06", "2020-07"]}) == "2020-07"
    discovered: dict[str, object] = {
        "months": None,
        "discovered_months": {
            "AAAUSDT": {"klines": ["2026-08", "2026-09"], "fundingRate": ["2026-08"]},
            "BBBUSDT": {"klines": ["2026-07"], "fundingRate": ["2026-07"]},
        },
    }
    assert _coverage_month(discovered) == "2026-09"
    assert _coverage_month({"months": None}) is None


# --- the read itself --------------------------------------------------------


def test_the_holdout_report_carries_the_fold_schema_plus_the_confirmation(
    chain: Chain,
) -> None:
    directory = _case(chain, "report")
    artifact = _read(chain, directory)
    assert artifact.candidate_name == WINNING_MEMBER
    document = _document(artifact.output_path)
    assert document["report_hash"] == artifact.report_hash
    material = {key: value for key, value in document.items() if key != "report_hash"}
    assert content_sha256(material) == artifact.report_hash
    assert document["holdout"] is True
    assert document["status"] == "development_only"
    assert document["report_version"] == "1.0.0"
    assert document["candidate_name"] == WINNING_MEMBER
    assert document["decision_report_hash"] == chain.decision_report_hash
    assert document["warm_up_weeks"] == 0

    manifest = _document(chain.manifest)
    holdout_ids = _strings(manifest["final_holdout_ids"])
    assert document["holdout_sample_count"] == len(holdout_ids)
    assert document["holdout_membership_hash"] == content_sha256(holdout_ids)
    assert document["split_manifest_hash"] == manifest["split_manifest_hash"]
    assert document["manifest_hash"] == manifest["manifest_hash"]
    assert document["family_name"] == manifest["family_name"]
    assert document["family_spec_hash"] == manifest["family_spec_hash"]

    # The bars come from the extended captures; the originals are bound by hash.
    extended = _document(chain.extended_perp / "capture-manifest.json")
    extended_spot = _document(chain.extended_spot / "capture-manifest.json")
    original = _document(chain.original_perp / "capture-manifest.json")
    original_spot = _document(chain.original_spot / "capture-manifest.json")
    assert document["capture_root_hash"] == extended["capture_root_hash"]
    assert document["hedge_capture_root_hash"] == extended_spot["capture_root_hash"]
    assert document["original_capture_root_hash"] == original["capture_root_hash"]
    assert document["original_hedge_capture_root_hash"] == original_spot["capture_root_hash"]
    assert document["dataset_root_hash"] == _document(
        chain.extended_perp / "dataset" / "dataset-manifest.json"
    )["root_hash"]
    assert document["hedge_dataset_root_hash"] == _document(
        chain.extended_spot / "dataset" / "dataset-manifest.json"
    )["root_hash"]

    # Only the candidate and the two dominance controls are read (spec 3).
    names = [str(item["candidate_name"]) for item in _dicts(document["candidates"])]
    assert names == [WINNING_MEMBER, "no_trade", "random_pairs"]
    episodes = _dicts(_dict(_candidate(document, WINNING_MEMBER)["base"])["episodes"])
    assert [item["sample_id"] for item in episodes] == holdout_ids
    # What the extended capture is taken for (spec section 2): the last holdout
    # episode exits at its own price in `EXTRA_MONTH` rather than being
    # force-closed for want of an exit bar, and no bar after that exit is read.
    assert all(item["forced_close_count"] == 0 for item in episodes)

    confirmation = _dict(document["confirmation"])
    assert confirmation["verdict"] in {"holdout_confirmed", "holdout_failed"}
    criteria = _dict(confirmation["criteria"])
    assert set(criteria) == {
        "base_total_positive",
        "adverse_after_uncharged_exit_non_negative",
        "base_dominates_no_trade",
        "base_dominates_random_pairs",
        "adverse_dominates_random_pairs",
        "largest_episode_share_within_limit",
        "largest_pair_share_within_limit",
        "skipped_decisions_within_limit",
        "largest_episode_share",
        "largest_pair_share",
        "skipped_decision_count",
    }
    reported = _dict(confirmation["reported"])
    assert set(reported) == {
        "base_mean_weekly_net_return",
        "adverse_mean_weekly_net_return",
        "positive_week_fraction",
        "decision_base_mean_net_return",
        "decision_base_bootstrap_lower",
        "base_mean_at_or_above_decision_bootstrap_lower",
    }
    assert reported["decision_base_mean_net_return"] == DECISION_BASE_MEAN
    assert reported["decision_base_bootstrap_lower"] == DECISION_BOOTSTRAP_LOWER
    base_total = _decimal(_dict(_candidate(document, WINNING_MEMBER)["base"])["total_net_return"])
    assert _decimal(reported["base_mean_weekly_net_return"]) == (
        base_total / Decimal(len(holdout_ids))
    )
    positives = sum(1 for item in episodes if _decimal(item["net_return"]) > 0)
    assert _decimal(reported["positive_week_fraction"]) == (
        Decimal(positives) / Decimal(len(holdout_ids))
    )
    assert reported["base_mean_at_or_above_decision_bootstrap_lower"] is (
        base_total / Decimal(len(holdout_ids)) >= Decimal(DECISION_BOOTSTRAP_LOWER)
    )


def test_the_verdict_is_the_conjunction_of_the_reports_own_criteria(chain: Chain) -> None:
    """Spec section 5 recomputed from the published document, number by number."""
    directory = _case(chain, "verdict")
    artifact = _read(chain, directory)
    document = _document(artifact.output_path)
    confirmation = _dict(document["confirmation"])
    criteria = _dict(confirmation["criteria"])

    candidate_base = _dict(_candidate(document, WINNING_MEMBER)["base"])
    candidate_adverse = _dict(_candidate(document, WINNING_MEMBER)["adverse"])
    base_total = _decimal(candidate_base["total_net_return"])
    adverse_total = _decimal(candidate_adverse["total_net_return"])
    uncharged = _decimal(candidate_adverse["uncharged_final_exit_cost"])
    after = adverse_total - uncharged
    assert _decimal(confirmation["base_total_net_return"]) == base_total
    assert _decimal(confirmation["adverse_total_net_return"]) == adverse_total
    assert _decimal(confirmation["adverse_uncharged_final_exit_cost"]) == uncharged
    assert _decimal(confirmation["adverse_total_after_uncharged_exit"]) == after

    no_trade_base = _decimal(_dict(_candidate(document, "no_trade")["base"])["total_net_return"])
    random_base = _decimal(
        _dict(_candidate(document, "random_pairs")["base"])["total_net_return"]
    )
    random_adverse = _decimal(
        _dict(_candidate(document, "random_pairs")["adverse"])["total_net_return"]
    )
    episodes = _dicts(candidate_base["episodes"])
    pair_totals: dict[str, Decimal] = {}
    for episode in episodes:
        contributions = episode["contract_net_contributions"]
        assert isinstance(contributions, list)
        for item in contributions:
            assert isinstance(item, list)
            pair_id, value = item
            assert isinstance(pair_id, str)
            pair_totals[pair_id] = pair_totals.get(pair_id, Decimal(0)) + _decimal(value)
    returns = [_decimal(episode["net_return"]) for episode in episodes]
    shares = (
        {
            "largest_episode_share": max(returns, default=Decimal(0)) / base_total,
            "largest_pair_share": max(pair_totals.values(), default=Decimal(0)) / base_total,
        }
        if base_total > 0
        else {"largest_episode_share": Decimal(0), "largest_pair_share": Decimal(0)}
    )
    assert _decimal(criteria["largest_episode_share"]) == shares["largest_episode_share"]
    assert _decimal(criteria["largest_pair_share"]) == shares["largest_pair_share"]

    skipped = len(_strings(document["skipped_sample_ids"]))
    assert criteria["skipped_decision_count"] == skipped
    expected = {
        "base_total_positive": base_total > 0,
        "adverse_after_uncharged_exit_non_negative": after >= 0,
        "base_dominates_no_trade": base_total > no_trade_base,
        "base_dominates_random_pairs": base_total > random_base,
        "adverse_dominates_random_pairs": after >= random_adverse,
        "largest_episode_share_within_limit": (
            shares["largest_episode_share"] <= CONCENTRATION_LIMIT
        ),
        "largest_pair_share_within_limit": shares["largest_pair_share"] <= CONCENTRATION_LIMIT,
        "skipped_decisions_within_limit": skipped <= MAX_SKIPPED_HOLDOUT_DECISIONS,
    }
    assert {key: criteria[key] for key in expected} == expected
    assert confirmation["verdict"] == (
        "holdout_confirmed" if all(expected.values()) else "holdout_failed"
    )
    # The fixture holdout is a mixed verdict rather than a trivially failing
    # one: the candidate turns a positive base total and beats both dominance
    # controls, and fails on the exit its four slots never paid and on how much
    # of that thin total one week and one pair carry. A failed holdout is a
    # verdict the command returns, not a refusal.
    assert criteria["base_total_positive"] is True
    assert criteria["base_dominates_no_trade"] is True
    assert criteria["base_dominates_random_pairs"] is True
    assert criteria["adverse_after_uncharged_exit_non_negative"] is False
    assert criteria["largest_pair_share_within_limit"] is False
    assert confirmation["verdict"] == "holdout_failed"
    assert artifact.verdict == confirmation["verdict"]


def test_the_read_is_single_use_per_family(chain: Chain) -> None:
    """Spec section 6: the artifact is immutable and the family is recorded."""
    directory = _case(chain, "single-use")
    artifact = _read(chain, directory)
    manifest = _document(chain.manifest)
    family_id = uuid5(
        NAMESPACE_URL, f"{manifest['split_manifest_hash']}:{manifest['family_name']}"
    )
    with MetadataRegistry(directory / "registry.sqlite3") as registry:
        record = registry.get_artifact(uuid5(family_id, HOLDOUT_ARTIFACT_KIND))
    assert record is not None
    assert record.kind == HOLDOUT_ARTIFACT_KIND
    assert record.content_hash == artifact.report_hash
    assert record.relative_path == artifact.output_path.name

    with pytest.raises(CarryHoldoutError):
        _read(chain, directory)
    with pytest.raises(CarryHoldoutError):
        _read(chain, directory, output_name="holdout-again.json")
    assert not (directory / "holdout-again.json").exists()


def test_an_original_capture_the_manifest_was_not_built_on_refuses(chain: Chain) -> None:
    """The lineage check alone cannot catch this, so the link check runs first.

    The extended capture is a perfectly good superset base -- it is a superset
    of itself -- but it is not the capture the manifest, and so the whole
    family, was published from. Only its hash against the manifest says so.
    """
    directory = _case(chain, "link")
    with pytest.raises(CarryHoldoutError) as error:
        _read(chain, directory, original_perp=chain.extended_perp)
    assert str(error.value) == "manifest is not linked to the original capture (capture_root_hash)"
    with pytest.raises(CarryHoldoutError) as hedge_error:
        _read(chain, directory, original_spot=chain.extended_spot)
    assert str(hedge_error.value) == (
        "manifest is not linked to the original capture (hedge_capture_root_hash)"
    )
    assert not (directory / "holdout.json").exists()


def test_an_extended_capture_that_is_not_a_superset_refuses_with_the_lineage_reasons(
    chain: Chain,
) -> None:
    """The original spot capture stands in for the extended perpetual one.

    Both originals are still the manifest's own captures, so the link check
    passes and the lineage check is what refuses -- on a capture of the wrong
    market, whose venue, market and every row disagree.
    """
    directory = _case(chain, "lineage")
    with pytest.raises(CarryHoldoutError) as error:
        _read(chain, directory, perp=chain.original_spot)
    verified, reasons = verify_capture_superset(chain.original_perp, chain.original_spot)
    assert verified is False and reasons
    assert all(reason in str(error.value) for reason in reasons)


def test_an_extended_capture_that_stops_before_the_last_exit_refuses(chain: Chain) -> None:
    """A capture is a superset of itself, so only the calendar can refuse here.

    The original capture ends with `MONTHS`; the holdout last episode exits in
    `EXTRA_MONTH`, which only the extended captures carry.
    """
    directory = _case(chain, "coverage")
    with pytest.raises(CarryHoldoutError) as error:
        _read(chain, directory, perp=chain.original_perp, spot=chain.original_spot)
    assert "extended capture does not cover the holdout" in str(error.value)


def test_a_decision_that_does_not_name_these_fold_reports_refuses(chain: Chain) -> None:
    directory = _case(chain, "sources")
    decision = directory / "decision.json"
    _seal_decision(
        decision, manifest=_document(chain.manifest),
        fold_report_hashes=("0" * 64, "1" * 64),
        eligible=(LOSING_MEMBER, WINNING_MEMBER),
    )
    with pytest.raises(CarryHoldoutError):
        _read(chain, directory, decision=decision)


def test_a_decision_whose_seal_does_not_recompute_refuses(chain: Chain) -> None:
    directory = _case(chain, "seal")
    decision = directory / "decision.json"
    document = _document(chain.decision)
    document["eligible_member_names"] = [LOSING_MEMBER]
    decision.write_bytes(canonical_json(document))
    with pytest.raises(CarryHoldoutError):
        _read(chain, directory, decision=decision)


def test_a_decision_without_an_eligible_member_refuses(chain: Chain) -> None:
    directory = _case(chain, "no-eligible")
    decision = directory / "decision.json"
    _seal_decision(
        decision, manifest=_document(chain.manifest),
        fold_report_hashes=tuple(
            str(_document(path)["report_hash"]) for path in chain.fold_reports
        ),
        eligible=(),
    )
    with pytest.raises(CarryHoldoutError):
        _read(chain, directory, decision=decision)
    assert not (directory / "holdout.json").exists()


def test_the_cli_reads_the_holdout_and_keeps_every_path_in_the_workspace(
    chain: Chain,
) -> None:
    directory = _case(chain, "cli")
    arguments = [
        "carry-holdout", "--workspace-root", str(chain.root),
        "--capture", str(chain.extended_perp), "--hedge-capture", str(chain.extended_spot),
        "--original-capture", str(chain.original_perp),
        "--original-hedge-capture", str(chain.original_spot),
        "--manifest", str(chain.manifest), "--family-spec", str(chain.config),
        "--decision", str(chain.decision),
        "--output", str(directory / "holdout.json"),
        "--registry", str(directory / "registry.sqlite3"),
    ]
    for report in chain.fold_reports:
        arguments.extend(["--fold-report", str(report)])
    assert main(arguments) == 0
    document = _document(directory / "holdout.json")
    assert document["holdout"] is True
    assert document["candidate_name"] == WINNING_MEMBER

    escaped = chain.root.parent / "escaped.json"
    outside = [*arguments]
    outside[outside.index(str(directory / "holdout.json"))] = str(escaped)
    with pytest.raises(ValueError):
        main(outside)
    assert not escaped.exists()
