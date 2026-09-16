import json
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest

from tests.carry_fixtures import (
    EXIT_WEEK_SYMBOL,
    HOLE_SYMBOL,
    build_captures,
    funding_rate,
    perp_fetch_with_a_hole,
    perp_fetch_with_a_liquidity_dip,
    perp_fetch_with_a_warm_up_hole,
    perp_fetch_with_negative_funding_weeks,
    perp_fetch_with_one_negative_week,
    small_carry_config,
    small_carry_v2_config,
    small_carry_v4_config,
    small_carry_v4_one_slot_config,
)
from tests.test_panel_fold_run import (
    DAY_MS,
    EPOCH_DAY_2020,
    MONTH_DAYS,
    MONTH_START_DAY,
    MONTHS,
    SYMBOLS,
    kline_csv,
)
from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.carry_config import (
    CONTROL_NAMES,
    MEMBER_NAMES,
    MEMBER_NAMES_V2,
    MEMBER_NAMES_V4,
    load_carry_family_spec,
)
from trading_bot.carry_fold_run import CarryFoldError, control_reference, run_carry_fold
from trading_bot.carry_signals import hurdle_minimum_trailing, random_pair_order
from trading_bot.carry_universe import EligiblePair, PairUniverseSnapshot
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


# --- the v4 declaration: a declared-capital slot book ------------------------

COHORT_EXTRA_KEYS = {
    "funding_collected", "basis_pnl", "spot_trading_cost", "perpetual_trading_cost",
    "forced_spot_legs", "forced_perpetual_legs", "exit_rule_removals", "hurdle_rejections",
}
SLOT_EXTRA_KEYS = {"filled_slots", "slot_fills", "slot_releases", "no_fill"}
# The fixture's decile is two pairs wide -- twelve eligible pairs over a
# denominator of ten, floored at `minimum_selected` 2 -- so a slot candidate
# never fills more than two of the four declared slots, and the two it fills
# are the fixture's two top payers.
TOP_PAYING_SYMBOLS = ("C10USDT", "C09USDT")


@pytest.fixture
def v4_workspace(tmp_path: Path) -> Workspace:
    perp, spot = build_captures(tmp_path)
    return tmp_path, perp, spot, _publish(tmp_path, perp, spot, small_carry_v4_config)


@pytest.fixture
def v4_workspace_with_a_hole(tmp_path: Path) -> Workspace:
    perp, spot = build_captures(tmp_path, perp_fetch_function=perp_fetch_with_a_hole)
    return tmp_path, perp, spot, _publish(tmp_path, perp, spot, small_carry_v4_config)


@pytest.fixture
def v4_workspace_with_negative_funding_weeks(tmp_path: Path) -> Workspace:
    perp, spot = build_captures(
        tmp_path, perp_fetch_function=perp_fetch_with_negative_funding_weeks
    )
    return tmp_path, perp, spot, _publish(tmp_path, perp, spot, small_carry_v4_config)


@pytest.fixture
def v4_workspace_with_one_negative_week(tmp_path: Path) -> Workspace:
    perp, spot = build_captures(tmp_path, perp_fetch_function=perp_fetch_with_one_negative_week)
    return tmp_path, perp, spot, _publish(tmp_path, perp, spot, small_carry_v4_config)


@pytest.fixture
def v4_workspace_with_a_liquidity_dip(tmp_path: Path) -> Workspace:
    perp, spot = build_captures(tmp_path, perp_fetch_function=perp_fetch_with_a_liquidity_dip)
    return tmp_path, perp, spot, _publish(tmp_path, perp, spot, small_carry_v4_config)


@pytest.fixture
def v4_one_slot_workspace(tmp_path: Path) -> Workspace:
    """The v4 fixture pinned to one declared slot, against a ranking that is
    two pairs wide -- the case that binds `pair_slots` as a real cap rather
    than a number the fill never reaches."""
    perp, spot = build_captures(tmp_path)
    return tmp_path, perp, spot, _publish(tmp_path, perp, spot, small_carry_v4_one_slot_config)


def _held_symbols(episode: dict[str, object]) -> set[str]:
    """The symbols an episode attributed a return or a cost to."""
    return {
        contract_id.split(":")[0]
        for contract_id, _ in _pairs(episode["contract_net_contributions"])
    }


def _pair_id_suffix(document: dict[str, object]) -> str:
    """Whatever the capture appended to each symbol to make its contract id.

    Every fixture symbol starts on the same day, so they all share one suffix;
    reading it off a report keeps these tests free of the reader's
    contract-id spelling.
    """
    episode = _episodes(document, "all_pairs_ew")[0]
    return next(
        contract_id.split(":", 1)[1]
        for contract_id, _ in _pairs(episode["contract_net_contributions"])
    )


def _seeded_pair_order(space: Workspace, document: dict[str, object], index: int) -> list[str]:
    """The order `random_pairs` draws its slots in at one decision.

    Every fixture symbol is eligible at every in-window decision and they all
    share one quote volume, so the snapshot the runner builds is this one up
    to the fields the draw does not read.
    """
    spec, _ = load_carry_family_spec(space[3])
    suffix = _pair_id_suffix(document)
    pairs = tuple(
        EligiblePair(
            pair_id=f"{symbol}:{suffix}",
            perpetual_contract_id=f"{symbol}:{suffix}",
            spot_contract_id=f"{symbol}:{suffix}",
            perpetual_tier=2, spot_tier=2, tier=2,
            liquidity_rank=rank, median_quote_volume=Decimal(1),
        )
        for rank, symbol in enumerate(sorted(SYMBOLS), start=1)
    )
    decision_close_ns = int(_ordered_test_sample_ids(space[0])[index].split(":")[1])
    return random_pair_order(
        PairUniverseSnapshot(decision_close_ns, pairs, ()),
        selection=spec.selection, random_seed=spec.statistics.random_seed,
    )


def _fixture_close(symbol: str, day_offset: int) -> Decimal:
    """The perpetual close the fixture's capture carries for one day."""
    month = next(
        item for item in MONTHS
        if MONTH_START_DAY[item] <= day_offset < MONTH_START_DAY[item] + MONTH_DAYS[item]
    )
    for line in kline_csv(symbol, month).splitlines()[1:]:
        fields = line.split(",")
        if int(fields[0]) // DAY_MS - EPOCH_DAY_2020 == day_offset:
            return Decimal(fields[4])
    raise AssertionError(f"the fixture has no bar for {symbol} on day offset {day_offset}")


def _expected_uncharged(
    space: Workspace,
    document: dict[str, object],
    name: str,
    scenario: str,
    *,
    symbols: tuple[str, ...],
) -> Decimal:
    """Liquidating the last episode's drifted book, by hand.

    Each held slot is a quarter of the declared book, so its spot leg opens at
    `+1/8` and its perpetual leg at `-1/8`. The accounting drifts each leg by
    its own return over the holding week against the book's gross return
    (which the episode reports), and leaving costs one side of the leg's fee
    plus its tier's slippage. The fixture's spot closes sit exactly one unit
    under the perpetual's.
    """
    spec, _ = load_carry_family_spec(space[3])
    assert spec.capital is not None
    table = spec.costs.base if scenario == "base" else spec.costs.adverse
    tier_one = set(sorted(SYMBOLS)[: spec.universe.tier_one_rank_limit])
    last_day = _decision_day_offsets(space)[-1]
    gross_return = _decimal(_episodes(document, name, scenario)[-1]["gross_return"])
    leg_weight = Decimal(1) / Decimal(spec.capital.pair_slots) / Decimal(2)
    total = Decimal(0)
    for symbol in symbols:
        slippage = (
            table.slippage_bps_per_side_tier_one
            if symbol in tier_one
            else table.slippage_bps_per_side_tier_two
        )
        entry = _fixture_close(symbol, last_day)
        exit_close = _fixture_close(symbol, last_day + spec.holding_days)
        legs = (
            (table.spot_fee_bps_per_side, leg_weight, entry - 1, exit_close - 1),
            (table.perpetual_fee_bps_per_side, -leg_weight, entry, exit_close),
        )
        for fee, weight, leg_entry, leg_exit in legs:
            leg_return = leg_exit / leg_entry - Decimal(1)
            drifted = weight * (Decimal(1) + leg_return) / (Decimal(1) + gross_return)
            total += abs(drifted) * (fee + slippage) / Decimal(10_000)
    return total


def _scenario(document: dict[str, object], name: str, scenario: str) -> dict[str, object]:
    return _object_dict(_candidate(document, name)[scenario])


def test_a_slot_family_warms_nothing_and_fills_its_slots_at_the_first_decision(
    v4_workspace: Workspace,
) -> None:
    """Spec 3.3: a slot book has no warm-up. Every slot is empty at the fold's
    first decision and is filled there, so `warm_up_weeks` is zero, the report
    never claims a warmed opening book, and the first episode buys the whole
    book inside the window."""
    document = _run(v4_workspace)
    assert document["family_name"] == "funding_carry_panel_v4"
    names = [_object_dict(item)["candidate_name"] for item in _object_list(document["candidates"])]
    assert names == list(MEMBER_NAMES_V4 + CONTROL_NAMES)
    assert document["warm_up_weeks"] == 0
    assert document["reason_codes"] == ["FOLD_FINAL_EXIT_COST_UNCHARGED"]
    for name in MEMBER_NAMES_V4:
        first = _episodes(document, name)[0]
        extras = _object_dict(first["extras"])
        assert set(extras) == COHORT_EXTRA_KEYS | SLOT_EXTRA_KEYS
        assert _decimal(extras["filled_slots"]) == 2
        assert _decimal(extras["slot_fills"]) == 2
        assert _decimal(extras["slot_releases"]) == 0
        assert _decimal(extras["no_fill"]) == 0
        # two filled slots of four: spot +1/8 and perp -1/8 apiece, so gross is
        # the filled share of the declared book and the two legs still hedge
        assert _decimal(first["gross_exposure"]) == Decimal("0.5")
        assert _decimal(first["net_exposure"]) == 0
        assert _decimal(first["turnover"]) == _decimal(first["gross_exposure"])
        assert _held_symbols(first) == set(TOP_PAYING_SYMBOLS)


def test_pair_slots_caps_the_fill_even_when_the_ranking_is_wider(
    v4_one_slot_workspace: Workspace,
) -> None:
    """`_slot_decision` wires `free = capital.pair_slots - len(held_slots)`, but
    the fixture's ranking is only two pairs wide, so a four-slot book never
    exercises the cap. Pin the declaration to one slot instead: every member
    fills exactly that one slot, from the top of the ranking, at every one of
    fold 0's three decisions, and the seeded control does too."""
    document = _run(v4_one_slot_workspace)
    top = TOP_PAYING_SYMBOLS[0]
    for name in MEMBER_NAMES_V4:
        assert _extras(document, name, "filled_slots") == [Decimal(1)] * 3
        episodes = _episodes(document, name)
        assert len(episodes) == 3
        for episode in episodes:
            assert _decimal(episode["gross_exposure"]) == Decimal(1)
            assert _held_symbols(episode) == {top}
    assert _extras(document, "random_pairs", "filled_slots") == [Decimal(1)] * 3


def test_a_released_slot_refilled_by_the_same_pair_costs_nothing(
    v4_workspace: Workspace,
) -> None:
    """Spec 3.2: a pair released by age that still ranks is filled again in the
    same decision, its weights do not change and its hold clock restarts. The
    one-week member releases both slots at the second decision and refills them
    with the same two pairs, so its episodes are the two-week member's digit
    for digit -- only the slot bookkeeping tells the two apart."""
    document = _run(v4_workspace)
    weekly, fortnightly = MEMBER_NAMES_V4[0], MEMBER_NAMES_V4[2]
    assert _extras(document, weekly, "slot_releases") == [Decimal(0), Decimal(2), Decimal(2)]
    assert _extras(document, weekly, "slot_fills") == [Decimal(2), Decimal(2), Decimal(2)]
    # the two-week member holds its first slots through the second decision and
    # ages them out at the third
    assert _extras(document, fortnightly, "slot_releases") == [Decimal(0), Decimal(0), Decimal(2)]
    assert _extras(document, fortnightly, "slot_fills") == [Decimal(2), Decimal(0), Decimal(2)]
    for name in (weekly, fortnightly):
        assert _extras(document, name, "filled_slots") == [Decimal(2)] * 3
    for scenario in ("base", "adverse"):
        economics = [
            [
                {key: value for key, value in episode.items() if key != "extras"}
                for episode in _episodes(document, name, scenario)
            ]
            for name in (weekly, fortnightly)
        ]
        assert economics[0] == economics[1]


def test_the_exit_rule_frees_one_slot_and_the_next_ranked_pair_takes_it(
    v4_workspace_with_one_negative_week: Workspace,
) -> None:
    """Spec 3.2 step 2, per slot: the one settlement inside the week that ends
    at the second decision is negative for C10USDT alone, so the exit member
    empties that pair's slot and refills it from the next ranked paying pair it
    does not already hold, while its non-exit sibling keeps the pair and fills
    a third slot beside it."""
    document = _run(v4_workspace_with_one_negative_week)
    exiting, holding = MEMBER_NAMES_V4[3], MEMBER_NAMES_V4[2]
    assert _extras(document, exiting, "exit_rule_removals") == [
        Decimal(0), Decimal(1), Decimal(0),
    ]
    assert _extras(document, holding, "exit_rule_removals") == [Decimal(0)] * 3
    # the emptied slot is refilled in the same decision, so the exit member
    # still holds two slots where its sibling now holds three
    assert _extras(document, exiting, "slot_fills") == [Decimal(2), Decimal(1), Decimal(1)]
    assert _extras(document, exiting, "filled_slots") == [Decimal(2), Decimal(2), Decimal(2)]
    assert _extras(document, holding, "slot_fills") == [Decimal(2), Decimal(1), Decimal(1)]
    assert _extras(document, holding, "filled_slots") == [Decimal(2), Decimal(3), Decimal(2)]
    assert _decimal(_episodes(document, exiting)[1]["gross_exposure"]) == Decimal("0.5")
    assert _decimal(_episodes(document, holding)[1]["gross_exposure"]) == Decimal("0.75")
    # only the sibling still carries the dropped pair into the third decision,
    # where its slot ages out -- the exit member parted with it a week earlier
    assert _extras(document, exiting, "slot_releases") == [Decimal(0), Decimal(0), Decimal(1)]
    assert _extras(document, holding, "slot_releases") == [Decimal(0), Decimal(0), Decimal(2)]
    assert EXIT_WEEK_SYMBOL not in _held_symbols(_episodes(document, exiting)[2])
    assert EXIT_WEEK_SYMBOL in _held_symbols(_episodes(document, holding)[2])


def test_a_force_closed_pair_frees_its_slot_rather_than_ageing_out(
    v4_workspace_with_a_hole: Workspace,
) -> None:
    """A slot whose pair lost a leg is emptied by the forced close that already
    ended the episode, exactly as the cohort book strips the pair, so the slot
    is free at the next decision and is not released a second time."""
    document = _run(v4_workspace_with_a_hole)
    name = MEMBER_NAMES_V4[2]
    episodes = _episodes(document, name)
    assert len(episodes) == 3
    assert episodes[1]["forced_close_count"] == 1
    assert _decimal(episodes[1]["forced_close_cost"]) > 0
    assert _extras(document, name, "filled_slots") == [Decimal(2), Decimal(2), Decimal(2)]
    assert _extras(document, name, "slot_releases") == [Decimal(0), Decimal(0), Decimal(1)]
    assert _extras(document, name, "slot_fills") == [Decimal(2), Decimal(0), Decimal(2)]
    assert episodes[2]["forced_close_count"] == 0
    assert _decimal(episodes[2]["gross_exposure"]) == Decimal("0.5")
    # the pair is out of the universe and out of the book; its surviving spot
    # leg still leaves through ordinary turnover under the pair's id
    assert HOLE_SYMBOL in _held_symbols(episodes[2])


def test_a_week_that_pays_nothing_fills_no_slot(
    v4_workspace_with_negative_funding_weeks: Workspace,
) -> None:
    """Spec 3.2's fill step: fewer than `minimum_selected` paying pairs means
    no fill this week. The slots stay empty, the decision is not a skip, and
    the sample joins the candidate's `no_carry_cohort_sample_ids`."""
    space = v4_workspace_with_negative_funding_weeks
    document = _run(space)
    ordered_ids = _ordered_test_sample_ids(space[0])
    assert document["skipped_sample_ids"] == []
    for name in MEMBER_NAMES_V4:
        assert _extras(document, name, "no_fill") == [Decimal(1), Decimal(1), Decimal(0)]
        assert _extras(document, name, "filled_slots") == [Decimal(0), Decimal(0), Decimal(2)]
        assert _candidate(document, name)["no_carry_cohort_sample_ids"] == ordered_ids[:2]
        episodes = _episodes(document, name)
        assert [episode["reason_codes"] for episode in episodes] == [
            ["MEMBER_HELD_NOTHING"], ["MEMBER_HELD_NOTHING"], [],
        ]
        assert _decimal(episodes[2]["gross_exposure"]) == Decimal("0.5")
    # the seeded control draws from every pair rather than only the paying
    # ones, so it fills its slots in the weeks the members cannot
    assert _extras(document, "random_pairs", "no_fill") == [Decimal(0)] * 3


def test_the_controls_of_a_slot_family_keep_their_own_books(v4_workspace: Workspace) -> None:
    """Spec 4.3: `random_pairs` runs the same slot book off the seeded draw,
    `no_trade` holds nothing and `all_pairs_ew` stays on the cohort book at the
    reference hold. The slot extras are reported for all three, at zero
    wherever the mechanic does not apply."""
    space = v4_workspace
    document = _run(space)
    for scenario in ("base", "adverse"):
        for episode in _episodes(document, "no_trade", scenario):
            assert _decimal(episode["net_return"]) == 0
            assert _decimal(episode["gross_exposure"]) == 0
    for key in sorted(SLOT_EXTRA_KEYS):
        assert _extras(document, "no_trade", key) == [Decimal(0)] * 3
        assert _extras(document, "all_pairs_ew", key) == [Decimal(0)] * 3
    drawn = _seeded_pair_order(space, document, 0)
    first_random = _episodes(document, "random_pairs")[0]
    held = {contract_id for contract_id, _ in _pairs(first_random["contract_net_contributions"])}
    assert held == set(drawn[:2])
    filled = _extras(document, "random_pairs", "filled_slots")
    assert filled == [Decimal(2)] * 3
    assert all(value <= 4 for value in filled)
    # every paying pair at equal weight: eleven of the twelve pairs, each at
    # spot +1/22 and perp -1/22, so the cohort book is unit gross
    context = _episodes(document, "all_pairs_ew")[0]
    contributions = _pairs(context["contract_net_contributions"])
    assert len(contributions) == len(SYMBOLS) - 1
    assert "C11USDT" not in _held_symbols(context)
    # eleven-pair shares are not exact in Decimal, so unit gross and the hedge
    # are bounded rather than equalities
    assert abs(_decimal(context["gross_exposure"]) - Decimal(1)) < Decimal("1e-25")
    assert abs(_decimal(context["net_exposure"])) < Decimal("1e-25")


def test_uncharged_final_exit_cost_prices_the_last_book_s_liquidation(
    v4_workspace: Workspace,
) -> None:
    """Spec 3.3: a fold's final exit falls outside the window and is never
    charged, and because a slot book holds no warmed cohorts the uncharged
    share is the whole book -- so the report states it, per candidate and
    scenario, at that scenario's fees and tier slippage."""
    space = v4_workspace
    document = _run(space)
    for scenario in ("base", "adverse"):
        flat = _scenario(document, "no_trade", scenario)
        assert _decimal(flat["uncharged_final_exit_cost"]) == 0
    name = MEMBER_NAMES_V4[0]
    assert _episodes(document, name)[-1]["forced_close_count"] == 0
    assert _held_symbols(_episodes(document, name)[-1]) == set(TOP_PAYING_SYMBOLS)
    costs: dict[str, Decimal] = {}
    for scenario in ("base", "adverse"):
        expected = _expected_uncharged(
            space, document, name, scenario, symbols=TOP_PAYING_SYMBOLS
        )
        assert expected > 0
        costs[scenario] = _decimal(
            _scenario(document, name, scenario)["uncharged_final_exit_cost"]
        )
        assert abs(costs[scenario] - expected) < Decimal("1e-25")
    # the adverse table doubles tier slippage, so the same book costs more to
    # liquidate under it
    assert costs["adverse"] > costs["base"]


def test_a_cohort_family_reports_neither_slot_extras_nor_an_uncharged_exit(
    v2_workspace: Workspace,
) -> None:
    """The slot bookkeeping is a slot family's alone: v2's fold report keeps
    exactly the eight extras keys it had and states no uncharged exit."""
    document = _run(v2_workspace)
    for name in (*MEMBER_NAMES_V2, *CONTROL_NAMES):
        for scenario in ("base", "adverse"):
            assert set(_scenario(document, name, scenario)) == {"total_net_return", "episodes"}
            for episode in _episodes(document, name, scenario):
                assert set(_object_dict(episode["extras"])) == COHORT_EXTRA_KEYS


def test_a_universe_too_small_week_empties_every_slot(
    v4_workspace_with_a_liquidity_dip: Workspace,
) -> None:
    """Spec 3.3: a week whose universe is too small is skipped and empties
    every slot, as it resets every cohort under v2; the slots refill at the
    next tradeable decision and the book restarts from flat, so nothing is
    released there and the refill is charged its full entry."""
    space = v4_workspace_with_a_liquidity_dip
    document = _run(space)
    middle = _ordered_test_sample_ids(space[0])[1]
    assert document["skipped_sample_ids"] == [middle]
    for name in MEMBER_NAMES_V4:
        episodes = _episodes(document, name)
        assert len(episodes) == 2
        assert _extras(document, name, "filled_slots") == [Decimal(2), Decimal(2)]
        assert _extras(document, name, "slot_fills") == [Decimal(2), Decimal(2)]
        # the skip emptied the slots, so the refill releases nothing and no
        # exit of the pre-skip book is charged (P1.27's uncharged-skip rule)
        assert _extras(document, name, "slot_releases") == [Decimal(0), Decimal(0)]
        assert _decimal(episodes[1]["gross_exposure"]) == Decimal("0.5")
        assert _decimal(episodes[1]["turnover"]) == Decimal("0.5")
