import json
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest

from tests.test_panel_fold_run import (
    DAY_MS,
    EPOCH_DAY_2020,
    KLINE_HEADER,
    MONTH_DAYS,
    MONTH_START_DAY,
    MONTHS,
    SYMBOLS,
    kline_csv,
    small_config,
    zip_bytes,
)
from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.funding_xs_config import (
    FUNDING_XS_CONTROL_NAMES,
    FUNDING_XS_MEMBER_NAMES,
    FundingXsFamilySpec,
    load_funding_xs_family_spec,
)
from trading_bot.funding_xs_fold_run import FundingXsFoldError, run_funding_xs_fold
from trading_bot.funding_xs_signals import build_funding_xs_weight_vectors
from trading_bot.panel_capture import PanelPayload, capture_panel
from trading_bot.panel_config import load_panel_family_spec
from trading_bot.panel_fold_run import (
    MEMBER_HELD_NOTHING_REASON_CODE,
    run_panel_fold,
    verify_panel_fold_report,
)
from trading_bot.panel_reader import FundingEvent, load_funding_events, load_panel_bars
from trading_bot.panel_samples import publish_panel_walk_forward
from trading_bot.panel_universe import ContractHistory, build_contract_histories, select_universe
from trading_bot.registry import MetadataRegistry
from trading_bot.vector_fold_run import FOLD_WARMED_REASON_CODE

DAY_NS = 86_400_000_000_000
WEEK_NS = 7 * DAY_NS
BP = Decimal("0.0001")
EXTRAS_KEYS = {"signalled_contracts", "funding_collected", "exit_rule_removals", "mean_score"}

# The same fold of the same capture the trend family is exercised on, so the
# two families' fixtures can be read side by side: fold 1's three test Sundays
# and the three Sundays warmed before them.
FOLD_INDEX = 1
FOLD_1_DECISION_OFFSETS = (137, 144, 151)
FOLD_1_WARM_UP_OFFSETS = (116, 123, 130)

# Funding is what this family ranks on, so the fixture's settlements -- not its
# prices -- carry the signal: one row a week per symbol (the pattern
# `tests/carry_fixtures.py::funding_csv` uses), at a rate that rises with the
# symbol index. C00 pays 4 bp a week (its longs are paid), C11 collects 7 bp, so
# the cross-section ranks strictly by index and the extreme legs are known by
# name. The universe takes the ten most liquid of the twelve symbols.
FLIP_SYMBOL = "C09USDT"
FLIP_AFTER_OFFSET = 130
FLIPPED_RATE = Decimal("-0.5") * BP

# Dropping every symbol's quote volume across the five liquidity days ending on
# one Sunday pushes the median under the declared floor for that Sunday alone
# (`liquidity_window_days` is 5 in the reduced declaration below). Aimed at fold
# 1's last warm-up Sunday, outside the test window entirely.
_DIP_WINDOW_DAYS = 5
_WARM_UP_DIP_OFFSET = 130


def funding_rate(symbol: str) -> Decimal:
    return Decimal(int(symbol[1:3]) - 4) * BP


def funding_csv(symbol: str, month: str, *, flip: bool) -> str:
    """One settlement a week; `flip` turns C09's late weeks negative.

    The flip is what lets the sign-flip exit rule be observed at all: C09 keeps
    the four-week score that puts it in the short quintile (it collected 5 bp a
    week until day 130) while the single week the rule reads has turned
    negative, so a short is no longer justified.
    """
    text = "calc_time,funding_interval_hours,last_funding_rate\n"
    for offset in range(0, MONTH_DAYS[month], 7):
        day = EPOCH_DAY_2020 + MONTH_START_DAY[month] + offset
        rate = funding_rate(symbol)
        if flip and symbol == FLIP_SYMBOL and day - EPOCH_DAY_2020 > FLIP_AFTER_OFFSET:
            rate = FLIPPED_RATE
        text += f"{day * DAY_MS},8,{rate}\n"
    return text


def kline_csv_with_liquidity_dip(symbol: str, month: str, dip_days: frozenset[int]) -> str:
    seed = int(symbol[1:3])
    lines = [KLINE_HEADER]
    for offset in range(MONTH_DAYS[month]):
        day = EPOCH_DAY_2020 + MONTH_START_DAY[month] + offset
        open_ms = day * DAY_MS
        close = 100 + seed * 10 + (day % 11) + (seed * (day % 5)) / 4
        quote_volume = "1000000" if day in dip_days else "50000000"
        lines.append(
            f"{open_ms},{close},{close},{close},{close},10,{open_ms + DAY_MS - 1},"
            f"{quote_volume},100,50,25000000,0"
        )
    return "\n".join(lines) + "\n"


def funding_xs_fetch(
    *, flip: bool = False, dip_offset: int | None = None
) -> Callable[[str], PanelPayload]:
    """`capture_panel`'s fetch: P1.27's klines, with weekly funding rows."""
    dip_days = (
        frozenset()
        if dip_offset is None
        else frozenset(
            EPOCH_DAY_2020 + offset
            for offset in range(dip_offset - _DIP_WINDOW_DAYS + 1, dip_offset + 1)
        )
    )

    def fetch(url: str) -> PanelPayload:
        symbol = next(item for item in SYMBOLS if f"/{item}/" in url or f"/{item}-" in url)
        month = next(item for item in MONTHS if item in url)
        if "fundingRate" in url:
            return PanelPayload(
                url=url,
                raw_bytes=zip_bytes("f.csv", funding_csv(symbol, month, flip=flip)),
                received_time_ns=1,
            )
        text = (
            kline_csv(symbol, month)
            if dip_offset is None
            else kline_csv_with_liquidity_dip(symbol, month, dip_days)
        )
        return PanelPayload(url=url, raw_bytes=zip_bytes("k.csv", text), received_time_ns=1)

    return fetch


def small_funding_xs_config(tmp_path: Path) -> Path:
    """The frozen funding declaration under `small_config`'s reductions.

    Exactly the reductions `tests/test_panel_fold_run.small_config` applies to
    the momentum declaration (universe 20/5/10/10/4, quintile size 2, volatility
    window 10, the reduced fold geometry, block 2, floor 4), so this family is
    exercised over the same weeks of the same fixture capture as P1.27 and
    P1.31. Ten rankable contracts and a minimum quintile of 2 make each leg two
    contracts at 0.25.
    """
    document = json.loads(Path("configs/funding-xs-panel-v1.json").read_text(encoding="utf-8"))
    document["universe"].update(
        {
            "minimum_history_days": 20,
            "liquidity_window_days": 5,
            "maximum_contracts": 10,
            "minimum_contracts": 10,
            "tier_one_rank_limit": 4,
        }
    )
    document["weights"]["minimum_quintile_size"] = 2
    document["weights"]["volatility_window_days"] = 10
    document["folds"] = {
        "train_duration_ns": 60 * DAY_NS,
        "validation_duration_ns": 14 * DAY_NS,
        "test_duration_ns": 28 * DAY_NS,
        "step_ns": 28 * DAY_NS,
        "embargo_ns": 14 * DAY_NS,
        "holdout_duration_ns": 28 * DAY_NS,
    }
    document["statistics"].update({"block_length": 2, "pooled_episode_floor": 4})
    path = tmp_path / "small-funding-xs.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def funding_xs_workspace(
    tmp_path: Path, *, flip: bool = False, dip_offset: int | None = None
) -> tuple[Path, Path, Path]:
    capture_panel(
        workspace_root=tmp_path,
        output_directory=tmp_path / "capture",
        reserve_bytes=0,
        symbols=SYMBOLS,
        months=MONTHS,
        fetch=funding_xs_fetch(flip=flip, dip_offset=dip_offset),
    )
    config_path = small_funding_xs_config(tmp_path)
    spec, spec_hash = load_funding_xs_family_spec(config_path)
    publish_panel_walk_forward(
        tmp_path / "capture",
        output_path=tmp_path / "manifest.json",
        spec=spec,
        family_spec_hash=spec_hash,
    )
    return tmp_path, tmp_path / "capture", config_path


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    return funding_xs_workspace(tmp_path)


@pytest.fixture
def flipped_workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    return funding_xs_workspace(tmp_path, flip=True)


def _run(space: tuple[Path, Path, Path], fold_index: int = FOLD_INDEX) -> dict[str, object]:
    root, capture_root, config_path = space
    artifact = run_funding_xs_fold(
        capture_root,
        manifest_path=root / "manifest.json",
        family_spec_path=config_path,
        output_path=root / f"fold{fold_index}.json",
        registry_path=root / "registry.sqlite3",
        fold_index=fold_index,
    )
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


def _candidate(document: dict[str, object], name: str) -> dict[str, object]:
    candidates = (_object_dict(item) for item in _object_list(document["candidates"]))
    return next(item for item in candidates if item["candidate_name"] == name)


def _episodes(
    document: dict[str, object], name: str, scenario: str = "base"
) -> list[dict[str, object]]:
    record = _object_dict(_candidate(document, name)[scenario])
    return [_object_dict(item) for item in _object_list(record["episodes"])]


def _extras(episode: dict[str, object]) -> dict[str, object]:
    return _object_dict(episode["extras"])


def _held(episode: dict[str, object]) -> set[str]:
    """The contracts an episode's book touched, from its own attribution."""
    held: set[str] = set()
    for item in _object_list(episode["contract_net_contributions"]):
        pair = _object_list(item)
        assert isinstance(pair[0], str)
        held.add(pair[0])
    return held


def _fold_inputs(
    capture_root: Path, manifest_path: Path, fold_index: int
) -> tuple[dict[str, ContractHistory], dict[str, tuple[FundingEvent, ...]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fold = next(item for item in manifest["folds"] if item["fold_index"] == fold_index)
    bars = load_panel_bars(
        capture_root / "dataset", available_before_ns=int(fold["test_end_ns"]) + 1
    )
    funding: dict[str, tuple[FundingEvent, ...]] = {}
    for event in load_funding_events(capture_root / "dataset"):
        funding.setdefault(event.contract_id, ())
        funding[event.contract_id] += (event,)
    return build_contract_histories(bars), funding


def _fold_decisions(manifest_path: Path, fold_index: int) -> tuple[int, ...]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fold = next(item for item in manifest["folds"] if item["fold_index"] == fold_index)
    return tuple(sorted(int(str(value).split(":")[1]) for value in fold["test_ids"]))


def _offset_of(close_ns: int) -> int:
    """The fixture day offset of a bar close time (closes land one ms before
    midnight, so the floor division picks out the day the bar belongs to)."""
    return close_ns // DAY_NS - EPOCH_DAY_2020


def _day_offset(sample_id: object) -> int:
    assert isinstance(sample_id, str)
    return _offset_of(int(sample_id.split(":")[1]))


def _vector_at(
    histories: dict[str, ContractHistory],
    funding: dict[str, tuple[FundingEvent, ...]],
    spec: FundingXsFamilySpec,
    *,
    name: str,
    decision_close_ns: int,
) -> tuple[tuple[str, Decimal], ...]:
    """One candidate's weekly weight vector, straight from the signal module."""
    snapshot = select_universe(histories, decision_close_ns=decision_close_ns, rules=spec.universe)
    if not snapshot.contracts:
        return ()
    return build_funding_xs_weight_vectors(histories, snapshot, funding, spec=spec)[name].weights


def _expected_book(
    histories: dict[str, ContractHistory],
    funding: dict[str, tuple[FundingEvent, ...]],
    spec: FundingXsFamilySpec,
    sundays: tuple[int, ...],
    *,
    name: str,
    hold_weeks: int,
    decision_close_ns: int,
) -> tuple[tuple[str, Decimal], ...]:
    """Re-derive one decision's cohort book from the weekly weight vectors.

    Independent of the fold runner: it asks the signal module for each of the
    last `hold_weeks` Sundays' vectors and averages them at `1 / hold_weeks`,
    dropping any contract without a bar at the decision. Valid only over a
    window with no skipped Sunday, which the runner would reset, and for a
    member that declares no exit rule.
    """
    totals: dict[str, Decimal] = {}
    for sunday in sundays[-hold_weeks:]:
        for contract_id, weight in _vector_at(
            histories, funding, spec, name=name, decision_close_ns=sunday
        ):
            if decision_close_ns not in histories[contract_id].closes:
                continue
            totals[contract_id] = totals.get(contract_id, Decimal(0)) + weight / Decimal(
                hold_weeks
            )
    return tuple(sorted((key, value) for key, value in totals.items() if value != 0))


def test_fold_report_carries_every_member_control_and_the_four_extras(
    workspace: tuple[Path, Path, Path],
) -> None:
    document = _run(workspace)
    assert document["status"] == "development_only"
    assert document["family_name"] == "funding_xs_panel_v1"
    assert document["fold_index"] == FOLD_INDEX
    assert document["warm_up_weeks"] == 3
    reason_codes = _object_list(document["reason_codes"])
    assert FOLD_WARMED_REASON_CODE in reason_codes
    assert "FOLD_FINAL_EXIT_COST_UNCHARGED" in reason_codes

    candidates = [_object_dict(item) for item in _object_list(document["candidates"])]
    assert [item["candidate_name"] for item in candidates] == list(
        FUNDING_XS_MEMBER_NAMES + FUNDING_XS_CONTROL_NAMES
    )
    assert [item["role"] for item in candidates] == ["member"] * 4 + ["control"] * 3
    assert all(item["episode_count"] == len(FOLD_1_DECISION_OFFSETS) for item in candidates)

    for name in FUNDING_XS_MEMBER_NAMES + FUNDING_XS_CONTROL_NAMES:
        for scenario in ("base", "adverse"):
            episodes = _episodes(document, name, scenario)
            assert len(episodes) == len(FOLD_1_DECISION_OFFSETS)
            assert all(set(_extras(episode)) == EXTRAS_KEYS for episode in episodes)
    assert [
        _day_offset(episode["sample_id"])
        for episode in _episodes(document, "fx_q5_l4w_h1w")
    ] == list(FOLD_1_DECISION_OFFSETS)


def test_report_schema_is_the_momentum_schema_plus_warm_up_weeks_and_extras(
    workspace: tuple[Path, Path, Path],
) -> None:
    """`panel_decision.py` pools a funding fold through exactly the reader it
    uses for a momentum one, so the schema must stay P1.27's -- pinned here
    against a momentum report the real `run_panel_fold` produces over the same
    fold of the same capture, not against a copied list of key names. The only
    additions allowed are the fold-level `warm_up_weeks` and the per-episode
    `extras`."""
    root, capture_root, _ = workspace
    document = _run(workspace)

    panel_config_path = small_config(root)
    panel_spec, panel_spec_hash = load_panel_family_spec(panel_config_path)
    publish_panel_walk_forward(
        capture_root,
        output_path=root / "panel-manifest.json",
        spec=panel_spec,
        family_spec_hash=panel_spec_hash,
    )
    panel_artifact = run_panel_fold(
        capture_root,
        manifest_path=root / "panel-manifest.json",
        family_spec_path=panel_config_path,
        output_path=root / "panel-fold.json",
        registry_path=root / "panel-registry.sqlite3",
        fold_index=FOLD_INDEX,
    )
    panel: dict[str, object] = json.loads(panel_artifact.output_path.read_text(encoding="utf-8"))

    assert set(document) == set(panel) | {"warm_up_weeks"}
    assert set(_candidate(document, "fx_q5_l4w_h1w")) == set(_candidate(panel, "xs_mom_1w"))
    episode = _episodes(document, "fx_q5_l4w_h1w")[0]
    assert set(episode) == set(_episodes(panel, "xs_mom_1w")[0]) | {"extras"}
    assert set(_extras(episode)) == EXTRAS_KEYS
    # The two families' folds agree on the calendar they were cut from, which is
    # what lets `panel_decision.py` pool either through the same gates.
    assert document["test_membership_hash"] == panel["test_membership_hash"]
    assert document["split_manifest_hash"] == panel["split_manifest_hash"]


def test_no_trade_control_is_flat_and_never_marked(workspace: tuple[Path, Path, Path]) -> None:
    document = _run(workspace)
    for scenario in ("base", "adverse"):
        for episode in _episodes(document, "no_trade", scenario):
            assert episode["net_return"] == "0"
            assert episode["gross_exposure"] == "0"
            assert episode["turnover"] == "0"
            assert episode["reason_codes"] == []
            extras = _extras(episode)
            assert _decimal(extras["signalled_contracts"]) == 0
            assert _decimal(extras["funding_collected"]) == 0
            assert _decimal(extras["exit_rule_removals"]) == 0
            # A control ranks nothing on funding, so it has no mean score.
            assert _decimal(extras["mean_score"]) == 0


def test_one_week_member_holds_the_extreme_funding_legs_at_a_quarter_each(
    workspace: tuple[Path, Path, Path],
) -> None:
    """The book is spec 3's: long the lowest-funding quintile, short the
    highest, each leg half of gross. With ten rankable contracts and a minimum
    quintile of two, that is the two cheapest at +0.25 and the two richest at
    -0.25 -- here, by construction of the fixture's rates, C00/C01 long and
    C08/C09 short."""
    root, capture_root, config_path = workspace
    document = _run(workspace)
    spec, _ = load_funding_xs_family_spec(config_path)
    histories, funding = _fold_inputs(capture_root, root / "manifest.json", FOLD_INDEX)
    decisions = _fold_decisions(root / "manifest.json", FOLD_INDEX)
    assert tuple(_offset_of(value) for value in decisions) == FOLD_1_DECISION_OFFSETS

    vector = _vector_at(
        histories, funding, spec, name="fx_q5_l4w_h1w", decision_close_ns=decisions[0]
    )
    quarter = Decimal("0.25")
    assert [(contract_id.split(":")[0], weight) for contract_id, weight in vector] == [
        ("C00USDT", quarter),
        ("C01USDT", quarter),
        ("C08USDT", -quarter),
        ("C09USDT", -quarter),
    ]

    episode = _episodes(document, "fx_q5_l4w_h1w")[0]
    assert _decimal(episode["gross_exposure"]) == 1
    assert _decimal(episode["net_exposure"]) == 0
    assert _decimal(_extras(episode)["signalled_contracts"]) == 4
    assert _held(episode) == {contract_id for contract_id, _ in vector}
    # The short leg collects the rich funding it was picked for.
    assert _decimal(_extras(episode)["funding_collected"]) > 0


def test_four_week_member_holds_the_average_of_the_last_four_weekly_vectors(
    flipped_workspace: tuple[Path, Path, Path],
) -> None:
    """The four-week member's book at t is the sum of the last four weekly
    vectors at a quarter of capital each -- warm-up Sundays included -- not this
    week's vector. Pinned for every episode against an independent re-derivation
    from `build_funding_xs_weight_vectors`, on the flipped fixture, where the
    one-week lookback's vectors genuinely differ from Sunday to Sunday."""
    root, capture_root, config_path = flipped_workspace
    document = _run(flipped_workspace)
    spec, _ = load_funding_xs_family_spec(config_path)
    histories, funding = _fold_inputs(capture_root, root / "manifest.json", FOLD_INDEX)
    decisions = _fold_decisions(root / "manifest.json", FOLD_INDEX)
    warm_ups = tuple(decisions[0] - weeks * WEEK_NS for weeks in (3, 2, 1))
    sundays = warm_ups + decisions
    assert tuple(_offset_of(sunday) for sunday in sundays) == (
        FOLD_1_WARM_UP_OFFSETS + FOLD_1_DECISION_OFFSETS
    )
    weekly = [
        _vector_at(histories, funding, spec, name="fx_q5_l1w_h4w", decision_close_ns=sunday)
        for sunday in sundays
    ]
    # Sanity on the fixture: the four vectors behind the first episode are not
    # all the same book, so averaging them is a visible operation.
    assert len(set(weekly[:4])) > 1

    episodes = _episodes(document, "fx_q5_l1w_h4w")
    assert len(episodes) == len(FOLD_1_DECISION_OFFSETS)
    for index, episode in enumerate(episodes):
        decision_close_ns = decisions[index]
        expected = _expected_book(
            histories,
            funding,
            spec,
            sundays[: len(FOLD_1_WARM_UP_OFFSETS) + index + 1],
            name="fx_q5_l1w_h4w",
            hold_weeks=4,
            decision_close_ns=decision_close_ns,
        )
        assert expected  # sanity: the fixture's books are not all empty
        assert _decimal(episode["gross_exposure"]) == sum(
            (abs(weight) for _, weight in expected), Decimal(0)
        )
        assert _decimal(episode["net_exposure"]) == sum(
            (weight for _, weight in expected), Decimal(0)
        )
        assert _decimal(_extras(episode)["signalled_contracts"]) == len(expected)
        assert _held(episode) >= {contract_id for contract_id, _ in expected}


def test_exit_member_drops_a_short_whose_trailing_week_turned_non_positive(
    flipped_workspace: tuple[Path, Path, Path],
) -> None:
    """Spec 3's exit rule, the only thing separating `fx_q5_l4w_h4w_exit` from
    `fx_q5_l4w_h4w`: C09's four-week funding still puts it in the short quintile
    at the fold's first decision, but the single week the rule reads has turned
    negative, so a short is no longer justified. The exit member drops it from
    every cohort holding it -- three warmed vectors and the one just formed --
    while the member without the rule keeps it at the full quarter."""
    root, capture_root, config_path = flipped_workspace
    document = _run(flipped_workspace)
    spec, _ = load_funding_xs_family_spec(config_path)
    histories, funding = _fold_inputs(capture_root, root / "manifest.json", FOLD_INDEX)
    decisions = _fold_decisions(root / "manifest.json", FOLD_INDEX)
    flipped = next(
        contract_id
        for contract_id, _ in _vector_at(
            histories, funding, spec, name="fx_q5_l4w_h4w", decision_close_ns=decisions[0]
        )
        if contract_id.startswith(FLIP_SYMBOL)
    )

    kept = _episodes(document, "fx_q5_l4w_h4w")[0]
    exited = _episodes(document, "fx_q5_l4w_h4w_exit")[0]
    assert flipped in _held(kept)
    assert flipped not in _held(exited)
    # Nothing else changed: the removed leg's quarter of capital simply stays
    # undeployed, so gross falls by exactly the weight C09 carried.
    assert _held(kept) - _held(exited) == {flipped}
    assert _decimal(kept["gross_exposure"]) - _decimal(exited["gross_exposure"]) == Decimal(
        "0.25"
    )
    assert _decimal(_extras(kept)["signalled_contracts"]) - _decimal(
        _extras(exited)["signalled_contracts"]
    ) == Decimal(1)

    # The defining semantic, over the whole fold: the removed leg's quarter of
    # capital stays undeployed until the cohort holding it ages out, so gross
    # recovers a quarter of a vector at a time rather than snapping back.
    #
    # At offset 137 all four retained cohorts (formed on the Sundays at offsets
    # 116, 123, 130 and 137) had ranked C09 into the short quintile, so all four
    # carry it zeroed and each deploys 0.75 of a vector: (4 x 0.75) / 4 = 0.75.
    # At 144 the 116 cohort has aged out and the vector formed that day no
    # longer ranks C09 short at all -- the flipped weeks have pulled its
    # four-week funding below C07's -- so three zeroed vectors and one full one
    # give (3 x 0.75 + 1) / 4 = 0.8125. At 151 the 123 cohort has aged out too:
    # (2 x 0.75 + 2) / 4 = 0.875. Nothing is removed at either later decision,
    # because C09 is no longer held anywhere but in the zeros already taken.
    for scenario in ("base", "adverse"):
        exit_episodes = _episodes(document, "fx_q5_l4w_h4w_exit", scenario)
        assert [_decimal(episode["gross_exposure"]) for episode in exit_episodes] == [
            Decimal("0.75"),
            Decimal("0.8125"),
            Decimal("0.875"),
        ]
        assert [
            _decimal(_extras(episode)["exit_rule_removals"]) for episode in exit_episodes
        ] == [Decimal(1), Decimal(0), Decimal(0)]
        # The sibling without the rule is fully invested throughout, and no
        # other candidate ever sees a removal.
        sibling = _episodes(document, "fx_q5_l4w_h4w", scenario)
        assert [_decimal(episode["gross_exposure"]) for episode in sibling] == [Decimal(1)] * 3
        for name in ("fx_q5_l4w_h1w", "fx_q5_l4w_h4w", "fx_q5_l1w_h4w", "no_trade"):
            for episode in _episodes(document, name, scenario):
                assert _decimal(_extras(episode)["exit_rule_removals"]) == 0


def test_warm_up_sunday_without_a_universe_contributes_nothing_but_resets_nothing(
    tmp_path: Path,
) -> None:
    """A warm-up Sunday whose universe is too small forms no vector -- but there
    is no position to flatten out there, so the vectors formed on the other
    warm-up Sundays survive it and keep ageing out on the calendar. Only an
    in-window skipped week resets, where a real position is closed (P1.27)."""
    space = funding_xs_workspace(tmp_path, dip_offset=_WARM_UP_DIP_OFFSET)
    root, capture_root, config_path = space
    spec, _ = load_funding_xs_family_spec(config_path)
    histories, funding = _fold_inputs(capture_root, root / "manifest.json", FOLD_INDEX)
    decisions = _fold_decisions(root / "manifest.json", FOLD_INDEX)
    warm_ups = tuple(decisions[0] - weeks * WEEK_NS for weeks in (3, 2, 1))
    assert tuple(_offset_of(sunday) for sunday in warm_ups) == FOLD_1_WARM_UP_OFFSETS
    # Sanity on the fixture: the dip emptied the last warm-up Sunday's universe
    # and left the two before it, and the fold's own decisions, alone.
    assert not select_universe(
        histories, decision_close_ns=warm_ups[2], rules=spec.universe
    ).contracts
    for sunday in (warm_ups[0], warm_ups[1], *decisions):
        assert select_universe(
            histories, decision_close_ns=sunday, rules=spec.universe
        ).contracts

    document = _run(space)
    # The dip is outside the test window, so no decision is skipped.
    assert document["skipped_sample_ids"] == []
    assert "SKIPPED_WEEK_EXIT_COST_UNCHARGED" not in _object_list(document["reason_codes"])

    episode = _episodes(document, "fx_q5_l4w_h4w")[0]
    expected = _expected_book(
        histories,
        funding,
        spec,
        warm_ups + decisions[:1],
        name="fx_q5_l4w_h4w",
        hold_weeks=4,
        decision_close_ns=decisions[0],
    )
    assert expected  # the surviving warm-up vectors are still in the book
    assert _decimal(episode["gross_exposure"]) == sum(
        (abs(weight) for _, weight in expected), Decimal(0)
    )
    # Three of the four weeks are deployed: the dipped Sunday's quarter is not.
    assert _decimal(episode["gross_exposure"]) == Decimal("0.75")
    assert episode["reason_codes"] == []


def test_members_are_registered(workspace: tuple[Path, Path, Path]) -> None:
    root, _, _ = workspace
    _run(workspace)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    family_id = uuid5(NAMESPACE_URL, f"{manifest['split_manifest_hash']}:funding_xs_panel_v1")
    with MetadataRegistry(root / "registry.sqlite3") as registry:
        rows = registry.list_experiments(family_id)
    assert {row.candidate_name for row in rows} == set(FUNDING_XS_MEMBER_NAMES)
    assert all(row.outcome == "completed" for row in rows)


def test_report_is_immutable(workspace: tuple[Path, Path, Path]) -> None:
    _run(workspace)
    with pytest.raises(FundingXsFoldError, match="immutable"):
        _run(workspace)


def test_unknown_fold_index_is_rejected(workspace: tuple[Path, Path, Path]) -> None:
    with pytest.raises(FundingXsFoldError, match="not in the manifest"):
        _run(workspace, fold_index=9)


def test_family_spec_mismatch_is_rejected(workspace: tuple[Path, Path, Path]) -> None:
    root, capture_root, config_path = workspace
    other = json.loads(config_path.read_text(encoding="utf-8"))
    other["hypothesis"] = str(other["hypothesis"]) + " (different declaration)"
    other_path = root / "other-funding-xs.json"
    other_path.write_text(json.dumps(other), encoding="utf-8")
    with pytest.raises(FundingXsFoldError, match="does not match the manifest"):
        run_funding_xs_fold(
            capture_root,
            manifest_path=root / "manifest.json",
            family_spec_path=other_path,
            output_path=root / "fold-mismatch.json",
            registry_path=root / "registry.sqlite3",
            fold_index=FOLD_INDEX,
        )


def _rewrite_manifest_field(path: Path, key: str, value: object) -> None:
    """Tamper one manifest field and re-derive `manifest_hash`, so the file stays
    internally self-consistent and only the linkage under test is broken."""
    document = json.loads(path.read_text(encoding="utf-8"))
    document[key] = value
    material = {k: v for k, v in document.items() if k != "manifest_hash"}
    document["manifest_hash"] = content_sha256(material)
    path.write_bytes(canonical_json(document))


@pytest.mark.parametrize(
    "key,message",
    [
        ("capture_root_hash", "not linked to this capture"),
        ("dataset_root_hash", "not linked to this dataset"),
    ],
)
def test_capture_linkage_is_enforced(
    workspace: tuple[Path, Path, Path], key: str, message: str
) -> None:
    root, _, _ = workspace
    _rewrite_manifest_field(root / "manifest.json", key, "0" * 64)
    with pytest.raises(FundingXsFoldError, match=message):
        _run(workspace)


def test_member_that_cannot_form_both_quintiles_is_marked(tmp_path: Path) -> None:
    """`MEMBER_HELD_NOTHING` still means what P1.27 made it mean: the universe
    was fine, the member's own construction produced nothing. Raising the
    declared minimum quintile above half the rankable set is the funding
    family's way of reaching that state."""
    root, capture_root, config_path = funding_xs_workspace(tmp_path)
    document = json.loads(config_path.read_text(encoding="utf-8"))
    document["weights"]["minimum_quintile_size"] = 6
    wide_path = root / "wide-quintile.json"
    wide_path.write_text(json.dumps(document), encoding="utf-8")
    spec, spec_hash = load_funding_xs_family_spec(wide_path)
    publish_panel_walk_forward(
        capture_root,
        output_path=root / "wide-manifest.json",
        spec=spec,
        family_spec_hash=spec_hash,
    )
    artifact = run_funding_xs_fold(
        capture_root,
        manifest_path=root / "wide-manifest.json",
        family_spec_path=wide_path,
        output_path=root / "wide-fold.json",
        registry_path=root / "registry.sqlite3",
        fold_index=FOLD_INDEX,
    )
    report: dict[str, object] = json.loads(artifact.output_path.read_text(encoding="utf-8"))
    for name in FUNDING_XS_MEMBER_NAMES:
        episode = _episodes(report, name)[0]
        assert episode["reason_codes"] == [MEMBER_HELD_NOTHING_REASON_CODE]
        assert episode["gross_exposure"] == "0"
    # The controls are never marked: `no_trade`'s emptiness is a baseline.
    assert _episodes(report, "no_trade")[0]["reason_codes"] == []
