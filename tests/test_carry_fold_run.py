import json
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest

from tests.carry_fixtures import (
    HOLE_SYMBOL,
    build_captures,
    funding_rate,
    perp_fetch_with_a_hole,
    perp_fetch_with_a_liquidity_dip,
    perp_fetch_with_a_warm_up_hole,
    perp_fetch_with_negative_funding_weeks,
    small_carry_config,
    small_carry_v2_config,
)
from tests.test_panel_fold_run import EPOCH_DAY_2020, MONTH_DAYS, MONTH_START_DAY, MONTHS, SYMBOLS
from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.carry_config import (
    CONTROL_NAMES,
    MEMBER_NAMES,
    MEMBER_NAMES_V2,
    load_carry_family_spec,
)
from trading_bot.carry_fold_run import CarryFoldError, control_reference, run_carry_fold
from trading_bot.carry_signals import hurdle_minimum_trailing
from trading_bot.panel_config import load_family_spec
from trading_bot.panel_fold_run import verify_panel_fold_report
from trading_bot.panel_samples import publish_panel_walk_forward
from trading_bot.registry import MetadataRegistry

Workspace = tuple[Path, Path, Path, Path]  # root, perp capture, spot capture, config
DAY_NS = 86_400_000_000_000


def _publish(
    root: Path,
    perp: Path,
    spot: Path,
    declaration: Callable[[Path], Path] = small_carry_config,
) -> Path:
    config_path = declaration(root)
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


V1_FOLD0_EXPECTED_PATH = Path(__file__).parent / "fixtures" / "carry_v1_fold0_expected.json"


def test_v1_fold0_economics_are_reproducible(workspace: Workspace) -> None:
    """P1.28 reproducibility: the v1 declaration's fold-0 numbers are frozen.

    `carry_fold_run.py` is inside `_CARRY_MODULES`, so any edit to the runner
    moves `code_hash` and with it `report_hash`; the report hash therefore
    cannot pin reproducibility across a runner change. The economics can:
    these values were captured from the v1 fixture at e2939b8, before the v2
    rules were written, and every later runner change must leave them exactly
    where they were, digit for digit.
    """
    document = _run(workspace)
    expected: dict[str, object] = json.loads(
        V1_FOLD0_EXPECTED_PATH.read_text(encoding="utf-8")
    )
    assert sorted(expected) == sorted(MEMBER_NAMES + CONTROL_NAMES)
    for name in expected:
        for scenario in ("base", "adverse"):
            want = _object_dict(_object_dict(expected[name])[scenario])
            record = _object_dict(_candidate(document, name)[scenario])
            assert record["total_net_return"] == want["total_net_return"], (name, scenario)
            wanted = [_object_dict(item) for item in _object_list(want["episodes"])]
            for episode, expected_episode in zip(
                _episodes(document, name, scenario), wanted, strict=True
            ):
                assert {key: episode[key] for key in expected_episode} == expected_episode


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
        "forced_spot_legs", "forced_perpetual_legs", "exit_rule_removals", "hurdle_rejections",
    }
    # v1 declares neither rule, so both new extras stay at zero throughout
    assert _decimal(extras["exit_rule_removals"]) == 0
    assert _decimal(extras["hurdle_rejections"]) == 0
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


# --- the v2 declaration: hold 26, the exit rule and the cost hurdle ----------


@pytest.fixture
def v2_workspace(tmp_path: Path) -> Workspace:
    perp, spot = build_captures(tmp_path)
    return tmp_path, perp, spot, _publish(tmp_path, perp, spot, small_carry_v2_config)


@pytest.fixture
def v2_workspace_with_negative_funding_weeks(tmp_path: Path) -> Workspace:
    perp, spot = build_captures(
        tmp_path, perp_fetch_function=perp_fetch_with_negative_funding_weeks
    )
    return tmp_path, perp, spot, _publish(tmp_path, perp, spot, small_carry_v2_config)


def _integer(value: object) -> int:
    assert isinstance(value, int)
    return value


def _extras(document: dict[str, object], name: str, key: str) -> list[Decimal]:
    return [
        _decimal(_object_dict(episode["extras"])[key])
        for episode in _episodes(document, name)
    ]


def _decision_day_offsets(space: Workspace) -> list[int]:
    """Fold 0's decisions as day offsets into the fixture's capture."""
    return [
        int(sample_id.split(":")[1]) // DAY_NS - EPOCH_DAY_2020
        for sample_id in _ordered_test_sample_ids(space[0])
    ]


def _funding_day_offsets() -> list[int]:
    """The fixture's weekly funding rows, restarting at each month's first day."""
    return sorted(
        MONTH_START_DAY[month] + offset
        for month in MONTHS
        for offset in range(0, MONTH_DAYS[month], 7)
    )


def _warmed_cohort_count(space: Workspace) -> int:
    """How many of fold 0's warm-up Sundays, plus its own first decision, can
    form a cohort at all under this fixture.

    The fixture's twelve symbols share one quote volume and run contiguously
    from day offset zero, so a Sunday's universe is non-empty exactly when it
    is late enough to carry `minimum_history_days` bars and a complete
    liquidity window, and empty -- contributing no cohort -- before that. The
    fold's first decision sits at day 109 and the v2 warm-up reaches 25 weeks
    back, to day -66, well before the capture starts.
    """
    declaration: dict[str, object] = json.loads(space[3].read_text(encoding="utf-8"))
    universe = _object_dict(declaration["universe"])
    members = [_object_dict(item) for item in _object_list(declaration["members"])]
    warm_up_weeks = max(_integer(member["hold_weeks"]) for member in members) - 1
    earliest = max(
        _integer(universe["minimum_history_days"]),
        _integer(universe["liquidity_window_days"]),
    ) - 1
    first = _decision_day_offsets(space)[0]
    days = [first - 7 * week for week in range(warm_up_weeks, 0, -1)] + [first]
    return sum(1 for day in days if day >= earliest)


def _expected_hurdle_rejections(space: Workspace, decision_day: int) -> int:
    """Pairs paying positive funding that still fall short of their tier's hurdle.

    Every fixture symbol shares one quote volume, so both legs rank in
    lexicographic symbol order and a pair's tier -- the worse of its two
    legs' -- is tier one for the first `tier_one_rank_limit` symbols and tier
    two after. Each symbol's trailing funding is its constant rate times
    however many of the fixture's weekly rows fall inside the lookback window.
    """
    spec, _ = load_carry_family_spec(space[3])
    member = next(item for item in spec.members if item.hurdle_multiple is not None)
    assert member.hurdle_multiple is not None
    window_start = decision_day - 7 * member.lookback_weeks
    rows = [day for day in _funding_day_offsets() if window_start < day <= decision_day]
    rejected = 0
    for rank, symbol in enumerate(sorted(SYMBOLS), start=1):
        trailing = Decimal(funding_rate(symbol)) * len(rows)
        if trailing <= 0:
            continue
        minimum = hurdle_minimum_trailing(
            cost_table=spec.costs.base,
            tier=1 if rank <= spec.universe.tier_one_rank_limit else 2,
            multiple=member.hurdle_multiple,
            lookback_weeks=member.lookback_weeks,
            hold_weeks=member.hold_weeks,
        )
        if trailing < minimum:
            rejected += 1
    return rejected


def test_the_controls_borrow_the_shortest_hold_member_of_each_declaration() -> None:
    """Spec 3.3: the controls run the lookback and hold of the member with the
    shortest hold, ties broken by declaration order. That generic rule has to
    reproduce v1's hard-coded `carry_l1w_h4w` exactly, or P1.28 moves."""
    v1, _ = load_carry_family_spec(Path("configs/funding-carry-panel-v1.json"))
    assert control_reference(v1).name == "carry_l1w_h4w"
    assert (control_reference(v1).lookback_weeks, control_reference(v1).hold_weeks) == (1, 4)
    # v1 declares carry_l1w_h4w and carry_l4w_h4w both at hold 4: the tie goes
    # to the first declared, which is what the v1 runner hard-coded
    assert next(m.name for m in v1.members if m.hold_weeks == 4) == "carry_l1w_h4w"
    v2, _ = load_carry_family_spec(Path("configs/funding-carry-panel-v2.json"))
    assert control_reference(v2).name == "carry_l4w_h13w_exit"
    assert (control_reference(v2).lookback_weeks, control_reference(v2).hold_weeks) == (4, 13)


def test_v2_warms_up_to_the_longest_hold_and_runs_controls_at_the_shortest(
    v2_workspace: Workspace,
) -> None:
    document = _run(v2_workspace)
    assert document["family_name"] == "funding_carry_panel_v2"
    names = [_object_dict(item)["candidate_name"] for item in _object_list(document["candidates"])]
    assert names == list(MEMBER_NAMES_V2 + CONTROL_NAMES)
    # max(hold) - 1 = 25, not v1's hard-coded twelve
    assert document["warm_up_weeks"] == 25
    warmed = _warmed_cohort_count(v2_workspace)
    # the capture cannot reach 25 Sundays back from day 109, so the warm-up
    # runs short and the opening book is partial rather than full
    assert warmed == 13
    first = _episodes(document, "carry_l4w_h26w")[0]
    expected = Decimal(warmed) / Decimal(26)
    assert abs(_decimal(first["gross_exposure"]) - expected) < Decimal("1e-25")
    # the controls hold 13 weeks, so the same thirteen warmed cohorts fill
    # their book completely where they fill the 26-week member's only halfway
    control = _episodes(document, "random_pairs")[0]
    control_expected = Decimal(min(warmed, 13)) / Decimal(13)
    assert control_expected == 1
    assert abs(_decimal(control["gross_exposure"]) - control_expected) < Decimal("1e-25")


def test_the_exit_rule_empties_a_book_whose_week_paid_nothing(
    v2_workspace_with_negative_funding_weeks: Workspace,
) -> None:
    """Every symbol's settlement in the week before decision 109 is negative,
    so an exit-rule member drops every pair it holds and opens flat, while the
    same warmed cohorts stay in the book of the member that declares no exit
    rule. The rule reads only the week that ended at the decision."""
    space = v2_workspace_with_negative_funding_weeks
    document = _run(space)
    ordered_ids = _ordered_test_sample_ids(space[0])
    assert document["skipped_sample_ids"] == []

    removals = _extras(document, "carry_l4w_h13w_exit", "exit_rule_removals")
    # decision 109: the week (102, 109] holds only the forced-negative row at
    # day 105, so every pair the warmed cohorts hold leaves
    assert removals[0] > 0
    # decision 116: that week is negative too, but 109 already took every pair
    # out of every cohort and neither 109 nor 116 can form a new one under a
    # four-week lookback that is negative throughout, so nothing is left
    assert removals[1] == 0
    assert _candidate(document, "carry_l4w_h13w_exit")["no_carry_cohort_sample_ids"] == (
        ordered_ids[:2]
    )
    # decision 123: the week (116, 123] holds the positive row at day 119
    assert removals[2] == 0

    exited = _episodes(document, "carry_l4w_h13w_exit")
    assert _decimal(exited[0]["gross_exposure"]) == 0
    assert exited[0]["reason_codes"] == ["MEMBER_HELD_NOTHING"]
    # a pair the rule removed can be selected again once it pays: the cohort
    # formed at 123 is back in the book
    assert _decimal(exited[2]["gross_exposure"]) > 0

    held = _episodes(document, "carry_l4w_h26w")
    assert _extras(document, "carry_l4w_h26w", "exit_rule_removals") == [Decimal(0)] * 3
    assert _decimal(held[0]["gross_exposure"]) > 0
    assert held[0]["reason_codes"] == []
    for name in CONTROL_NAMES:
        assert _extras(document, name, "exit_rule_removals") == [Decimal(0)] * 3


def test_the_cost_hurdle_counts_the_pairs_that_cannot_pay_for_the_round_trip(
    v2_workspace: Workspace,
) -> None:
    document = _run(v2_workspace)
    expected = [
        Decimal(_expected_hurdle_rejections(v2_workspace, day))
        for day in _decision_day_offsets(v2_workspace)
    ]
    # Four of the twelve fixture symbols pay positive funding that is still
    # short of two round trips scaled from twenty-six weeks back to four --
    # and only three at the last decision, whose four-week window catches five
    # weekly rows rather than four because the fixture's funding calendar
    # restarts at each month's first day (April's last row is day 119, May's
    # first is day 121).
    assert expected == [Decimal(4), Decimal(4), Decimal(3)]
    assert _extras(document, "carry_l4w_h26w_exit_hurdle2", "hurdle_rejections") == expected
    for name in (*MEMBER_NAMES_V2[:3], *CONTROL_NAMES):
        assert _extras(document, name, "hurdle_rejections") == [Decimal(0)] * 3
    # the hurdle thins the ranking, it does not widen the cohort: the two top
    # payers clear it, so this member's book matches the plain 26-week one
    hurdled = _episodes(document, "carry_l4w_h26w_exit_hurdle2")
    plain = _episodes(document, "carry_l4w_h26w_exit")
    assert [e["gross_exposure"] for e in hurdled] == [e["gross_exposure"] for e in plain]
