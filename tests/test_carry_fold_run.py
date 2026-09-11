import json
from decimal import Decimal
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest

from tests.carry_fixtures import (
    HOLE_SYMBOL,
    build_captures,
    perp_fetch_with_a_hole,
    perp_fetch_with_a_liquidity_dip,
    perp_fetch_with_a_warm_up_hole,
    perp_fetch_with_negative_funding_weeks,
    small_carry_config,
)
from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.carry_config import CONTROL_NAMES, MEMBER_NAMES
from trading_bot.carry_fold_run import CarryFoldError, run_carry_fold
from trading_bot.panel_config import load_family_spec
from trading_bot.panel_fold_run import verify_panel_fold_report
from trading_bot.panel_samples import publish_panel_walk_forward
from trading_bot.registry import MetadataRegistry

Workspace = tuple[Path, Path, Path, Path]  # root, perp capture, spot capture, config


def _publish(root: Path, perp: Path, spot: Path) -> Path:
    config_path = small_carry_config(root)
    spec, spec_hash = load_family_spec(config_path)
    publish_panel_walk_forward(
        perp, output_path=root / "manifest.json", spec=spec,
        family_spec_hash=spec_hash, hedge_capture_root=spot,
    )
    return config_path


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    perp, spot = build_captures(tmp_path)
    return tmp_path, perp, spot, _publish(tmp_path, perp, spot)


@pytest.fixture
def workspace_with_a_hole(tmp_path: Path) -> Workspace:
    perp, spot = build_captures(tmp_path, perp_fetch_function=perp_fetch_with_a_hole)
    return tmp_path, perp, spot, _publish(tmp_path, perp, spot)


@pytest.fixture
def workspace_with_a_liquidity_dip(tmp_path: Path) -> Workspace:
    perp, spot = build_captures(tmp_path, perp_fetch_function=perp_fetch_with_a_liquidity_dip)
    return tmp_path, perp, spot, _publish(tmp_path, perp, spot)


@pytest.fixture
def workspace_with_negative_funding_weeks(tmp_path: Path) -> Workspace:
    perp, spot = build_captures(
        tmp_path, perp_fetch_function=perp_fetch_with_negative_funding_weeks
    )
    return tmp_path, perp, spot, _publish(tmp_path, perp, spot)


def _run(space: Workspace, fold_index: int = 0) -> dict[str, object]:
    root, perp, spot, config_path = space
    artifact = run_carry_fold(
        perp, spot, manifest_path=root / "manifest.json", family_spec_path=config_path,
        output_path=root / f"fold{fold_index}.json", registry_path=root / "registry.sqlite3",
        fold_index=fold_index,
    )
    assert artifact.episode_count > 0
    assert verify_panel_fold_report(artifact.output_path)
    document: dict[str, object] = json.loads(artifact.output_path.read_text(encoding="utf-8"))
    return document


def _object_list(value: object) -> list[object]:
    assert isinstance(value, list)
    return value


def _object_dict(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value


def _decimal(value: object) -> Decimal:
    assert isinstance(value, str)
    return Decimal(value)


def _pairs(value: object) -> list[tuple[str, str]]:
    """Decode a `contract_net_contributions`-shaped `[[id, decimal-string], ...]` list."""
    result: list[tuple[str, str]] = []
    for item in _object_list(value):
        assert isinstance(item, list) and len(item) == 2
        contract_id, amount = item
        assert isinstance(contract_id, str)
        assert isinstance(amount, str)
        result.append((contract_id, amount))
    return result


def _candidate(document: dict[str, object], name: str) -> dict[str, object]:
    candidates = (_object_dict(item) for item in _object_list(document["candidates"]))
    return next(item for item in candidates if item["candidate_name"] == name)


def _episodes(
    document: dict[str, object], name: str, scenario: str = "base"
) -> list[dict[str, object]]:
    scenario_record = _object_dict(_candidate(document, name)[scenario])
    return [_object_dict(item) for item in _object_list(scenario_record["episodes"])]


def test_carry_fold_report_has_the_panel_schema_plus_extras(workspace: Workspace) -> None:
    document = _run(workspace)
    manifest = json.loads((workspace[0] / "manifest.json").read_text(encoding="utf-8"))
    assert document["status"] == "development_only"
    assert document["family_name"] == "funding_carry_panel_v1"
    assert document["hedge_capture_root_hash"] == manifest["hedge_capture_root_hash"]
    assert document["hedge_dataset_root_hash"] == manifest["hedge_dataset_root_hash"]
    assert document["capture_root_hash"] == manifest["capture_root_hash"]
    assert document["fold_index"] == 0 and document["fold_count"] == len(manifest["folds"])
    assert document["reason_codes"] == [
        "FOLD_OPENING_BOOK_WARMED_FROM_PRIOR_WEEKS", "FOLD_FINAL_EXIT_COST_UNCHARGED",
    ]
    assert document["warm_up_weeks"] == 12
    names = [_object_dict(item)["candidate_name"] for item in _object_list(document["candidates"])]
    assert names == list(MEMBER_NAMES + CONTROL_NAMES)
    member = _candidate(document, "carry_l1w_h4w")
    assert member["role"] == "member" and member["no_carry_cohort_sample_ids"] == []
    assert _candidate(document, "no_trade")["role"] == "control"
    episode = _episodes(document, "carry_l1w_h4w")[0]
    assert set(episode) == {
        "sample_id", "net_return", "gross_return", "turnover", "trading_cost", "funding_cost",
        "forced_close_cost", "gross_exposure", "net_exposure", "forced_close_count",
        "reason_codes", "contract_net_contributions", "extras",
    }
    extras = _object_dict(episode["extras"])
    assert set(extras) == {
        "funding_collected", "basis_pnl", "spot_trading_cost", "perpetual_trading_cost",
        "forced_spot_legs", "forced_perpetual_legs",
    }
    contributions = _pairs(episode["contract_net_contributions"])
    # per-pair attribution sums to the episode's net return exactly
    total = sum((Decimal(value) for _, value in contributions), Decimal(0))
    assert total == _decimal(episode["net_return"])
    # pair ids are perpetual contract ids, not leg ids
    assert all(not cid.startswith(("perp:", "spot:")) for cid, _ in contributions)
    # the two legs of every pair are hedged: zero net exposure
    assert _decimal(episode["net_exposure"]) == 0


def test_no_trade_control_is_flat(workspace: Workspace) -> None:
    document = _run(workspace)
    for scenario in ("base", "adverse"):
        for episode in _episodes(document, "no_trade", scenario):
            assert _decimal(episode["net_return"]) == 0
            assert _decimal(episode["gross_exposure"]) == 0
            assert episode["reason_codes"] == []


def test_book_is_fully_deployed_from_the_first_decision(workspace: Workspace) -> None:
    """Every fold opens on a book warmed from the twelve Sundays before it, so
    the first test decision is already fully deployed -- no 1/H ramp, and no
    member-dependent haircut between an H = 4 and an H = 13 member."""
    document = _run(workspace)
    exposures = [_decimal(e["gross_exposure"]) for e in _episodes(document, "carry_l1w_h4w")]
    # 1/4-based shares are exact in Decimal, so this is an equality
    assert exposures == [Decimal(1), Decimal(1), Decimal(1)]
    # 1/13-based shares are not exact, so the long-hold member is bounded
    for episode in _episodes(document, "carry_l4w_h13w"):
        assert abs(Decimal(1) - _decimal(episode["gross_exposure"])) < Decimal("1e-25")
    # the funding leg collects: every member episode has positive funding_collected
    assert all(
        _decimal(_object_dict(e["extras"])["funding_collected"]) > 0
        for e in _episodes(document, "carry_l1w_h4w")
    )
    # the adverse scenario haircuts receipts by a quarter
    base = _episodes(document, "carry_l1w_h4w", "base")
    adverse = _episodes(document, "carry_l1w_h4w", "adverse")
    for b, a in zip(base, adverse, strict=True):
        b_funding = _decimal(_object_dict(b["extras"])["funding_collected"])
        a_funding = _decimal(_object_dict(a["extras"])["funding_collected"])
        assert a_funding == b_funding * Decimal("0.75")


def test_warm_up_uses_only_data_before_each_warm_up_sunday(workspace: Workspace) -> None:
    """The warm-up forms cohorts but carries no weights into the fold: the
    first in-window episode buys the whole warmed book from flat, so its
    turnover equals its gross exposure and the full entry cost is charged
    inside the window rather than being inherited untaxed from outside it."""
    document = _run(workspace)
    first = _episodes(document, "carry_l1w_h4w")[0]
    assert _decimal(first["turnover"]) == _decimal(first["gross_exposure"]) == Decimal(1)


def test_all_pairs_control_excludes_negative_funding(workspace: Workspace) -> None:
    document = _run(workspace)
    episode = _episodes(document, "all_pairs_ew")[0]
    contributions = _pairs(episode["contract_net_contributions"])
    assert not any(cid.startswith("C11USDT:") for cid, _ in contributions)


def test_forced_leg_closes_its_partner_at_the_next_rebalance(
    workspace_with_a_hole: Workspace,
) -> None:
    document = _run(workspace_with_a_hole)
    episodes = _episodes(document, "carry_l1w_h4w")
    assert len(episodes) == 3
    # week 2 (decision 116): the perpetual leg of the hole symbol has no exit bar
    assert episodes[1]["forced_close_count"] == 1
    assert _decimal(episodes[1]["forced_close_cost"]) > 0
    # week 3 (decision 123): the pair is out of the universe and out of the book, and its
    # spot leg is charged its exit through ordinary turnover under the pair's id
    third_contributions = _pairs(episodes[2]["contract_net_contributions"])
    assert any(cid.startswith(f"{HOLE_SYMBOL}:") for cid, _ in third_contributions)
    assert episodes[2]["forced_close_count"] == 0
    assert _decimal(episodes[2]["forced_close_cost"]) == 0
    # the hole pair's only week-3 cost is its spot leg's ordinary exit turnover
    # (one side only -- the perpetual leg was already force-closed in week 2)
    hole_contribution = next(
        value for cid, value in third_contributions if cid.startswith(f"{HOLE_SYMBOL}:")
    )
    assert Decimal(hole_contribution) < 0
    # week 3 retains the cohorts of 102, 109 and 116 (the warmed 95 cohort is
    # four weeks old and has aged out). Each lost C10 to the week-2 forced
    # close but kept its formed_size of 2, so each still contributes only 1/8
    # gross (its surviving pair, C09, is not reinvested into); the new 123
    # cohort [C09,C08] contributes 1/4: 3 * 1/8 + 1/4 = 5/8.
    assert _decimal(episodes[2]["gross_exposure"]) == Decimal("0.625")
    assert _decimal(episodes[2]["turnover"]) > Decimal("0.25")


def _ordered_test_sample_ids(root: Path) -> list[str]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    fold = next(f for f in manifest["folds"] if f["fold_index"] == 0)
    return sorted((str(v) for v in fold["test_ids"]), key=lambda v: int(v.split(":")[1]))


def test_universe_too_small_week_is_skipped_and_resets_the_book(
    workspace_with_a_liquidity_dip: Workspace,
) -> None:
    document = _run(workspace_with_a_liquidity_dip)
    middle_id = _ordered_test_sample_ids(workspace_with_a_liquidity_dip[0])[1]
    middle_ns = middle_id.split(":")[1]
    skipped = document["skipped_sample_ids"]
    assert isinstance(skipped, list) and len(skipped) == 1
    skipped_id = skipped[0]
    assert isinstance(skipped_id, str) and middle_ns in skipped_id
    reason_codes = document["reason_codes"]
    assert isinstance(reason_codes, list)
    assert "SKIPPED_WEEK_EXIT_COST_UNCHARGED" in reason_codes
    assert "FOLD_OPENING_BOOK_WARMED_FROM_PRIOR_WEEKS" in reason_codes
    # the week is skipped, not evaluated: only the two untouched decisions remain
    episodes = _episodes(document, "carry_l1w_h4w")
    assert len(episodes) == 2
    exposures = [_decimal(e["gross_exposure"]) for e in episodes]
    # the first decision opens on a warmed, fully deployed book; the skip then
    # resets cohorts AND carried weights and is not re-warmed (P1.27: after a
    # skip the fold restarts flat), so the post-skip episode holds one cohort
    assert exposures == [Decimal(1), Decimal("0.25")]
    # no exit of the pre-skip book is charged (P1.27's uncharged-skip semantics)
    assert _decimal(episodes[1]["turnover"]) == Decimal("0.25")


def test_no_carry_cohort_is_not_a_skip_and_marks_held_nothing(
    workspace_with_negative_funding_weeks: Workspace,
) -> None:
    document = _run(workspace_with_negative_funding_weeks)
    root = workspace_with_negative_funding_weeks[0]
    ordered_ids = _ordered_test_sample_ids(root)
    assert document["skipped_sample_ids"] == []
    assert document["reason_codes"] == [
        "FOLD_OPENING_BOOK_WARMED_FROM_PRIOR_WEEKS", "FOLD_FINAL_EXIT_COST_UNCHARGED",
    ]

    member = _candidate(document, "carry_l1w_h4w")
    assert member["no_carry_cohort_sample_ids"] == ordered_ids[:2]
    base = _episodes(document, "carry_l1w_h4w")
    # the warm-up Sundays 88, 95 and 102 see only negative funding too, so the
    # one-week member enters the fold with no book at all
    assert base[0]["reason_codes"] == ["MEMBER_HELD_NOTHING"]
    assert _decimal(base[0]["gross_exposure"]) == 0
    assert base[1]["reason_codes"] == ["MEMBER_HELD_NOTHING"]
    assert _decimal(base[1]["gross_exposure"]) == 0
    assert base[2]["reason_codes"] == []
    assert _decimal(base[2]["gross_exposure"]) == Decimal("0.25")

    # the 4-week members still see the positive rows outside the negative
    # window -- at the warm-up Sundays as well -- so they are warmed and hold
    # something in every episode: this pins that the book isn't reset for them
    l4w_episodes = _episodes(document, "carry_l4w_h4w")
    assert all(episode["reason_codes"] == [] for episode in l4w_episodes)
    assert all(_decimal(episode["gross_exposure"]) > 0 for episode in l4w_episodes)

    # controls are never marked MEMBER_HELD_NOTHING, in either scenario
    for name in CONTROL_NAMES:
        for scenario in ("base", "adverse"):
            for episode in _episodes(document, name, scenario):
                assert episode["reason_codes"] == []


def test_members_are_registered(workspace: Workspace) -> None:
    document = _run(workspace)
    manifest = json.loads((workspace[0] / "manifest.json").read_text(encoding="utf-8"))
    family_id = uuid5(NAMESPACE_URL, f"{manifest['split_manifest_hash']}:funding_carry_panel_v1")
    with MetadataRegistry(workspace[0] / "registry.sqlite3") as registry:
        records = registry.list_experiments(family_id)
    assert sorted(record.candidate_name for record in records) == sorted(MEMBER_NAMES)
    assert {record.result_hash for record in records} == {document["report_hash"]}
    assert {record.code_hash for record in records} == {document["code_hash"]}


def test_report_is_immutable(workspace: Workspace) -> None:
    _run(workspace)
    with pytest.raises(CarryFoldError, match="immutable"):
        _run(workspace)


def _rewrite_manifest_field(path: Path, key: str, value: object) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    document[key] = value
    material = {k: v for k, v in document.items() if k != "manifest_hash"}
    document["manifest_hash"] = content_sha256(material)
    path.write_bytes(canonical_json(document))


@pytest.mark.parametrize(
    "key", ["hedge_capture_root_hash", "hedge_dataset_root_hash", "capture_root_hash"]
)
def test_linkage_to_both_captures_is_enforced(workspace: Workspace, key: str) -> None:
    _rewrite_manifest_field(workspace[0] / "manifest.json", key, "0" * 64)
    with pytest.raises(CarryFoldError, match=key):
        _run(workspace)


def test_the_hedge_capture_must_not_be_the_perpetual_capture(workspace: Workspace) -> None:
    root, perp, _spot, config_path = workspace
    with pytest.raises(CarryFoldError):
        run_carry_fold(
            perp, perp, manifest_path=root / "manifest.json", family_spec_path=config_path,
            output_path=root / "fold0.json", registry_path=root / "registry.sqlite3",
            fold_index=0,
        )


def test_family_spec_mismatch_is_rejected(workspace: Workspace) -> None:
    root, perp, spot, config_path = workspace
    document = json.loads(config_path.read_text(encoding="utf-8"))
    document["hypothesis"] += " (edited)"
    other = root / "other.json"
    other.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(CarryFoldError, match="declaration"):
        _run((root, perp, spot, other))


@pytest.fixture
def workspace_with_a_warm_up_hole(tmp_path: Path) -> Workspace:
    perp, spot = build_captures(tmp_path, perp_fetch_function=perp_fetch_with_a_warm_up_hole)
    return tmp_path, perp, spot, _publish(tmp_path, perp, spot)


def test_pair_gone_dark_during_warm_up_is_dropped_before_the_first_entry(
    workspace_with_a_warm_up_hole: Workspace,
) -> None:
    """C10USDT sits in every warmed cohort but has no bar at the first decision.
    The runner must drop it before assembling the book (no entry close exists
    to trade it), leaving its cohort shares undeployed like a forced close, and
    the fold must run through instead of raising a missing-entry-close error."""
    document = _run(workspace_with_a_warm_up_hole)
    episodes = _episodes(document, "carry_l1w_h4w")
    assert len(episodes) == 3
    first = episodes[0]
    # warmed cohorts 88, 95, 102 each formed with two pairs and stripped of C10
    # -> three eighths; the fresh 109 cohort (C10 ineligible) adds one quarter
    assert _decimal(first["gross_exposure"]) == Decimal("0.625")
    assert first["forced_close_count"] == 0
    contributions = _pairs(first["contract_net_contributions"])
    assert not any(cid.startswith(f"{HOLE_SYMBOL}:") for cid, _ in contributions)
    assert document["skipped_sample_ids"] == []
