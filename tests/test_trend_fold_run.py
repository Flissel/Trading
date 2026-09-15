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
    fetch,
    small_config,
    zip_bytes,
)
from trading_bot import vector_fold_run as vector_fold_run_module
from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.panel_capture import PanelPayload, capture_panel
from trading_bot.panel_config import load_panel_family_spec
from trading_bot.panel_fold_run import (
    MEMBER_HELD_NOTHING_REASON_CODE,
    run_panel_fold,
    verify_panel_fold_report,
)
from trading_bot.panel_reader import load_panel_bars
from trading_bot.panel_samples import publish_panel_walk_forward
from trading_bot.panel_universe import ContractHistory, build_contract_histories, select_universe
from trading_bot.registry import MetadataRegistry
from trading_bot.trend_config import (
    TREND_CONTROL_NAMES,
    TREND_MEMBER_NAMES,
    TrendFamilySpec,
    load_trend_family_spec,
)
from trading_bot.trend_fold_run import FOLD_WARMED_REASON_CODE, TrendFoldError, run_trend_fold
from trading_bot.trend_signals import build_trend_weight_vectors
from trading_bot.vector_fold_run import assemble_cohort_book

DAY_NS = 86_400_000_000_000
WEEK_NS = 7 * DAY_NS
EXTRAS_KEYS = {"signalled_contracts", "abstained_contracts", "mean_score"}

# Fold 1, not fold 0, carries the interesting weeks. `roc_120` needs 121 closes,
# so no contract of this fixture has a trend score before day offset 120, and
# fold 0's three test Sundays (offsets 109, 116 and 123) put two of them before
# that line -- every member holds nothing there, which exercises nothing. Fold
# 1's Sundays (137, 144, 151) and the three Sundays warmed before them (116,
# 123, 130) straddle the line instead: one empty warm-up week, two filled ones,
# and a first decision whose scores fall inside both dead bands. Every offset
# here was confirmed empirically against this fixture.
FOLD_INDEX = 1
FOLD_1_DECISION_OFFSETS = (137, 144, 151)
FOLD_1_WARM_UP_OFFSETS = (116, 123, 130)
# Offset 137 scores 1/12 on every contract -- inside the 0.2 and the 0.5 dead
# band alike, so both one-week time-series members hold nothing there while the
# four-week member still carries its warmed cohorts.
DEAD_BAND_EPISODE_INDEX = 0

# Dropping every symbol's quote volume across the five liquidity days ending on
# one Sunday pushes the median under the declared floor for that Sunday alone, so
# its universe comes back empty while the weeks on either side are untouched --
# `liquidity_window_days` is 5 in the reduced declaration below. Two placements
# are used: fold 1's middle decision, an in-window skipped week; and fold 1's
# last warm-up Sunday, which lies outside the test window entirely.
_DIP_WINDOW_DAYS = 5
_IN_WINDOW_DIP_OFFSET = 144
_WARM_UP_DIP_OFFSET = 130


def _dip_days(target_offset: int) -> frozenset[int]:
    return frozenset(
        EPOCH_DAY_2020 + offset
        for offset in range(target_offset - _DIP_WINDOW_DAYS + 1, target_offset + 1)
    )


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


def fetch_with_liquidity_dip(target_offset: int) -> Callable[[str], PanelPayload]:
    """A `capture_panel` fetch whose klines lose their liquidity at `target_offset`."""
    dip_days = _dip_days(target_offset)

    def fetch_dipped(url: str) -> PanelPayload:
        symbol = next(item for item in SYMBOLS if f"/{item}/" in url or f"/{item}-" in url)
        month = next(item for item in MONTHS if item in url)
        if "fundingRate" in url:
            day = EPOCH_DAY_2020 + MONTH_START_DAY[month]
            text = "calc_time,funding_interval_hours,last_funding_rate\n"
            text += f"{day * DAY_MS},8,0.0001\n"
            return PanelPayload(url=url, raw_bytes=zip_bytes("f.csv", text), received_time_ns=1)
        return PanelPayload(
            url=url,
            raw_bytes=zip_bytes("k.csv", kline_csv_with_liquidity_dip(symbol, month, dip_days)),
            received_time_ns=1,
        )

    return fetch_dipped


def dipped_workspace(tmp_path: Path, target_offset: int) -> tuple[Path, Path, Path]:
    """`workspace`, captured through a fetch that empties one Sunday's universe."""
    capture_panel(
        workspace_root=tmp_path,
        output_directory=tmp_path / "capture",
        reserve_bytes=0,
        symbols=SYMBOLS,
        months=MONTHS,
        fetch=fetch_with_liquidity_dip(target_offset),
    )
    config_path = small_trend_config(tmp_path)
    spec, spec_hash = load_trend_family_spec(config_path)
    publish_panel_walk_forward(
        tmp_path / "capture",
        output_path=tmp_path / "manifest.json",
        spec=spec,
        family_spec_hash=spec_hash,
    )
    return tmp_path, tmp_path / "capture", config_path


def small_trend_config(tmp_path: Path) -> Path:
    """The frozen trend declaration under `small_config`'s reductions.

    Exactly the reductions `tests/test_panel_fold_run.small_config` applies to
    the momentum declaration (universe 20/5/10/10/4, quintile size 2,
    volatility window 10, the reduced fold geometry, block 2, floor 4), so both
    families are exercised over the same weeks of the same fixture capture.
    """
    document = json.loads(
        Path("configs/trend-aggregate-panel-v1.json").read_text(encoding="utf-8")
    )
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
    path = tmp_path / "small-trend.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    capture_panel(
        workspace_root=tmp_path,
        output_directory=tmp_path / "capture",
        reserve_bytes=0,
        symbols=SYMBOLS,
        months=MONTHS,
        fetch=fetch,
    )
    config_path = small_trend_config(tmp_path)
    spec, spec_hash = load_trend_family_spec(config_path)
    publish_panel_walk_forward(
        tmp_path / "capture",
        output_path=tmp_path / "manifest.json",
        spec=spec,
        family_spec_hash=spec_hash,
    )
    return tmp_path, tmp_path / "capture", config_path


def _run(space: tuple[Path, Path, Path], fold_index: int = FOLD_INDEX) -> dict[str, object]:
    root, capture_root, config_path = space
    artifact = run_trend_fold(
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


def _fold_histories(
    capture_root: Path, manifest_path: Path, fold_index: int
) -> dict[str, ContractHistory]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fold = next(item for item in manifest["folds"] if item["fold_index"] == fold_index)
    bars = load_panel_bars(
        capture_root / "dataset", available_before_ns=int(fold["test_end_ns"]) + 1
    )
    return build_contract_histories(bars)


def _offset_of(close_ns: int) -> int:
    """The fixture day offset of a bar close time (closes land one ms before
    midnight, so the floor division picks out the day the bar belongs to)."""
    return close_ns // DAY_NS - EPOCH_DAY_2020


def _day_offset(sample_id: object) -> int:
    assert isinstance(sample_id, str)
    return _offset_of(int(sample_id.split(":")[1]))


def _fold_decisions(manifest_path: Path, fold_index: int) -> tuple[int, ...]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fold = next(item for item in manifest["folds"] if item["fold_index"] == fold_index)
    return tuple(sorted(int(str(value).split(":")[1]) for value in fold["test_ids"]))


def _expected_book(
    histories: dict[str, ContractHistory],
    spec: TrendFamilySpec,
    sundays: tuple[int, ...],
    *,
    name: str,
    hold_weeks: int,
    decision_close_ns: int,
) -> tuple[tuple[str, Decimal], ...]:
    """Re-derive one decision's four-week book from the weekly weight vectors.

    Independent of the fold runner: it asks `build_trend_weight_vectors` for each
    of the last `hold_weeks` Sundays' vectors and averages them at `1 /
    hold_weeks`, dropping any contract without a bar at the decision. Valid only
    over a window with no skipped Sunday, which the runner would reset.
    """
    totals: dict[str, Decimal] = {}
    for sunday in sundays[-hold_weeks:]:
        snapshot = select_universe(histories, decision_close_ns=sunday, rules=spec.universe)
        if not snapshot.contracts:
            continue
        vector = build_trend_weight_vectors(histories, snapshot, spec=spec)[name].weights
        for contract_id, weight in vector:
            if decision_close_ns not in histories[contract_id].closes:
                continue
            share = weight / Decimal(hold_weeks)
            totals[contract_id] = totals.get(contract_id, Decimal(0)) + share
    return tuple(sorted((key, value) for key, value in totals.items() if value != 0))


FOLD_1_EXPECTED_PATH = Path(__file__).parent / "fixtures" / "trend_fold1_expected.json"


def test_fold1_economics_are_reproducible(workspace: tuple[Path, Path, Path]) -> None:
    """P1.31 reproducibility: this fixture's fold-1 numbers are frozen.

    `trend_fold_run.py` is inside `_TREND_MODULES`, so any edit to the runner
    moves `code_hash` and with it `report_hash`; the report hash therefore
    cannot pin reproducibility across a runner change. The economics can:
    every candidate's total and every episode's exposure, turnover, net return
    and per-contract attribution were captured from this fixture at ea777b3,
    before the fold loop was generalised into `vector_fold_run.py`, and every
    later runner change must leave them exactly where they were, digit for
    digit.
    """
    document = _run(workspace)
    expected: dict[str, object] = json.loads(
        FOLD_1_EXPECTED_PATH.read_text(encoding="utf-8")
    )
    assert sorted(expected) == sorted(TREND_MEMBER_NAMES + TREND_CONTROL_NAMES)
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


def test_fold_report_carries_every_member_control_and_the_three_extras(
    workspace: tuple[Path, Path, Path],
) -> None:
    document = _run(workspace)
    assert document["status"] == "development_only"
    assert document["family_name"] == "trend_aggregate_panel_v1"
    assert document["fold_index"] == FOLD_INDEX
    assert document["warm_up_weeks"] == 3
    reason_codes = _object_list(document["reason_codes"])
    assert FOLD_WARMED_REASON_CODE in reason_codes
    assert "FOLD_FINAL_EXIT_COST_UNCHARGED" in reason_codes

    candidates = [_object_dict(item) for item in _object_list(document["candidates"])]
    assert [item["candidate_name"] for item in candidates] == [
        "ta_ts_t02",
        "ta_ts_t05",
        "ta_ts_t02_h4w",
        "ta_xs_q5",
        "no_trade",
        "random_ranks",
        "passive_long_ew",
    ]
    assert [item["role"] for item in candidates] == ["member"] * 4 + ["control"] * 3
    assert all(item["episode_count"] == len(FOLD_1_DECISION_OFFSETS) for item in candidates)

    for scenario in ("base", "adverse"):
        episodes = _episodes(document, "ta_ts_t02", scenario)
        assert len(episodes) == len(FOLD_1_DECISION_OFFSETS)
        for episode in episodes:
            assert set(episode) == {
                "sample_id",
                "net_return",
                "gross_return",
                "turnover",
                "trading_cost",
                "funding_cost",
                "forced_close_cost",
                "gross_exposure",
                "net_exposure",
                "forced_close_count",
                "reason_codes",
                "contract_net_contributions",
                "extras",
            }
            assert set(_extras(episode)) == EXTRAS_KEYS
    assert [
        _day_offset(episode["sample_id"]) for episode in _episodes(document, "ta_ts_t02")
    ] == list(FOLD_1_DECISION_OFFSETS)


def test_report_schema_is_the_momentum_schema_plus_warm_up_weeks_and_extras(
    workspace: tuple[Path, Path, Path],
) -> None:
    """`panel_decision.py` pools a trend fold through exactly the reader it uses
    for a momentum one, so the schema must stay P1.27's -- pinned here against a
    momentum report the real `run_panel_fold` produces over the same fold of the
    same capture, not against a copied list of key names. The only additions
    allowed are the fold-level `warm_up_weeks` and the per-episode `extras`."""
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
    panel: dict[str, object] = json.loads(
        panel_artifact.output_path.read_text(encoding="utf-8")
    )

    assert set(document) == set(panel) | {"warm_up_weeks"}
    assert set(_candidate(document, "ta_ts_t02")) == set(_candidate(panel, "xs_mom_1w"))
    trend_episode = _episodes(document, "ta_ts_t02")[0]
    panel_episode = _episodes(panel, "xs_mom_1w")[0]
    assert set(trend_episode) == set(panel_episode) | {"extras"}
    assert set(_extras(trend_episode)) == EXTRAS_KEYS
    # The two families' folds agree on the calendar they were cut from, which is
    # what lets `panel_decision.py` pool either through the same gates.
    assert document["test_membership_hash"] == panel["test_membership_hash"]
    assert document["split_manifest_hash"] == panel["split_manifest_hash"]


def test_no_trade_control_is_flat_and_never_marked(workspace: tuple[Path, Path, Path]) -> None:
    document = _run(workspace)
    for scenario in ("base", "adverse"):
        episodes = _episodes(document, "no_trade", scenario)
        assert all(episode["net_return"] == "0" for episode in episodes)
        assert all(episode["gross_exposure"] == "0" for episode in episodes)
        assert all(episode["turnover"] == "0" for episode in episodes)
        assert all(episode["reason_codes"] == [] for episode in episodes)
        assert all(_extras(episode)["signalled_contracts"] == "0" for episode in episodes)
        assert all(_extras(episode)["abstained_contracts"] == "0" for episode in episodes)


def test_four_week_member_holds_the_average_of_the_last_four_weekly_vectors(
    workspace: tuple[Path, Path, Path],
) -> None:
    """The four-week member's book at t is the sum of the last four weekly
    vectors at a quarter of capital each -- warm-up Sundays included -- not this
    week's vector. Pinned for every episode of the fold against an independent
    re-derivation from `build_trend_weight_vectors`."""
    root, capture_root, config_path = workspace
    document = _run(workspace)
    spec, _ = load_trend_family_spec(config_path)
    histories = _fold_histories(capture_root, root / "manifest.json", FOLD_INDEX)
    decisions = _fold_decisions(root / "manifest.json", FOLD_INDEX)
    warm_ups = tuple(decisions[0] - weeks * WEEK_NS for weeks in (3, 2, 1))
    sundays = warm_ups + decisions
    # Sanity on the fixture: these are the Sundays the constants above describe.
    assert tuple(_offset_of(sunday) for sunday in sundays) == (
        FOLD_1_WARM_UP_OFFSETS + FOLD_1_DECISION_OFFSETS
    )
    episodes = _episodes(document, "ta_ts_t02_h4w")
    assert len(episodes) == len(FOLD_1_DECISION_OFFSETS)

    for index, episode in enumerate(episodes):
        sample_id = episode["sample_id"]
        assert isinstance(sample_id, str)
        decision_close_ns = int(sample_id.split(":")[1])
        expected = _expected_book(
            histories,
            spec,
            sundays[: len(FOLD_1_WARM_UP_OFFSETS) + index + 1],
            name="ta_ts_t02_h4w",
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
        assert _extras(episode)["signalled_contracts"] == str(Decimal(len(expected)))


def test_warm_up_fills_the_opening_book_a_cold_fold_would_ramp_into(
    workspace: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the three warmed Sundays the four-week member would open every
    fold on a 1/4 stub and ramp for three weeks -- a hold-dependent haircut on
    the primary metric that the one-week members never pay. With them the book
    is already at its full level at the fold's first episode."""
    warmed_gross = [
        _decimal(episode["gross_exposure"])
        for episode in _episodes(_run(workspace), "ta_ts_t02_h4w")
    ]

    monkeypatch.setattr(vector_fold_run_module, "warm_up_weeks_of", lambda spec: 0)
    root, capture_root, config_path = workspace
    artifact = run_trend_fold(
        capture_root,
        manifest_path=root / "manifest.json",
        family_spec_path=config_path,
        output_path=root / "fold-cold.json",
        registry_path=root / "registry.sqlite3",
        fold_index=FOLD_INDEX,
    )
    document: dict[str, object] = json.loads(artifact.output_path.read_text(encoding="utf-8"))
    cold_gross = [
        _decimal(episode["gross_exposure"])
        for episode in _episodes(document, "ta_ts_t02_h4w")
    ]
    assert document["warm_up_weeks"] == 0
    # The cold fold's first week holds nothing at all: its own vector abstains
    # inside the dead band and there is no warmed cohort behind it.
    assert cold_gross[0] == 0
    assert all(
        without < warm for without, warm in zip(cold_gross, warmed_gross, strict=True)
    )


def test_dead_band_silences_the_one_week_members_but_not_the_warmed_cohort(
    workspace: tuple[Path, Path, Path],
) -> None:
    """At fold 1's first decision every contract scores 1/12, inside both dead
    bands: the one-week time-series members hold nothing and are marked, while
    the four-week member still carries the two filled warm-up vectors."""
    document = _run(workspace)
    for name in ("ta_ts_t02", "ta_ts_t05"):
        for scenario in ("base", "adverse"):
            episode = _episodes(document, name, scenario)[DEAD_BAND_EPISODE_INDEX]
            assert episode["reason_codes"] == [MEMBER_HELD_NOTHING_REASON_CODE]
            assert episode["gross_exposure"] == "0"
            extras = _extras(episode)
            assert extras["signalled_contracts"] == "0"
            # All ten eligible contracts have a score, all ten are in the band.
            assert extras["abstained_contracts"] == "10"
            assert abs(_decimal(extras["mean_score"]) - Decimal(1) / Decimal(12)) < Decimal(
                "1e-25"
            )

    cohort = _episodes(document, "ta_ts_t02_h4w")[DEAD_BAND_EPISODE_INDEX]
    assert cohort["reason_codes"] == []
    assert _decimal(cohort["gross_exposure"]) > 0
    # Its own vector abstained too, so the book it holds is purely warmed.
    assert _extras(cohort)["abstained_contracts"] == "10"

    # A cross-sectional member has no dead band, so it never abstains.
    quintiles = _episodes(document, "ta_xs_q5")[DEAD_BAND_EPISODE_INDEX]
    assert _extras(quintiles)["abstained_contracts"] == "0"
    assert _decimal(_extras(quintiles)["signalled_contracts"]) > 0


def test_skipped_week_resets_the_four_week_cohort_vectors(tmp_path: Path) -> None:
    """A skipped week flattens the book (P1.27 semantics), so the retained weekly
    vectors go with it: the week after the skip opens on its own vector alone, a
    quarter of capital, rather than resuming the pre-skip cohorts."""
    space = dipped_workspace(tmp_path, _IN_WINDOW_DIP_OFFSET)
    config_path = space[2]
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    fold = next(item for item in manifest["folds"] if item["fold_index"] == FOLD_INDEX)
    test_ids = [str(value) for value in fold["test_ids"]]
    assert len(test_ids) == 3  # sanity: two live weeks bracket the dipped one

    artifact = run_trend_fold(
        tmp_path / "capture",
        manifest_path=tmp_path / "manifest.json",
        family_spec_path=config_path,
        output_path=tmp_path / "fold.json",
        registry_path=tmp_path / "registry.sqlite3",
        fold_index=FOLD_INDEX,
    )
    document: dict[str, object] = json.loads(artifact.output_path.read_text(encoding="utf-8"))
    assert document["skipped_sample_ids"] == [test_ids[1]]
    assert "SKIPPED_WEEK_EXIT_COST_UNCHARGED" in _object_list(document["reason_codes"])
    assert artifact.skipped_sample_count == 1

    cohort = _episodes(document, "ta_ts_t02_h4w")
    weekly = _episodes(document, "ta_ts_t02")
    assert [episode["sample_id"] for episode in cohort] == [test_ids[0], test_ids[2]]
    before = _decimal(cohort[0]["gross_exposure"])
    after = _decimal(cohort[1]["gross_exposure"])
    # Before the skip the book carries the warmed vectors; after it, only the
    # resumed week's own vector at a quarter of capital.
    resumed_vector_gross = _decimal(weekly[1]["gross_exposure"])
    assert abs(after - resumed_vector_gross / Decimal(4)) < Decimal("1e-25")
    assert 0 < after < before
    # The carried position was reset too, so the resumed week pays full entry.
    assert _decimal(cohort[1]["turnover"]) == after


def test_empty_warm_up_sunday_contributes_no_vector_but_resets_nothing(tmp_path: Path) -> None:
    """The counterpart to the test above, and the line between the two rules.

    A warm-up Sunday whose universe is too small forms no vector -- but there is
    no position to flatten out there, so the vectors formed on the other warm-up
    Sundays survive it and keep ageing out on the calendar. Only an in-window
    skipped week resets, where a real position is actually closed (P1.27).

    The dip is aimed at fold 1's *last* warm-up Sunday (offset 130), the one
    placement that separates the two rules on this fixture: the earlier warm-up
    Sunday at 116 is below the score's 121-day span and forms an empty vector
    anyway, and the fold's own first decision abstains inside the dead band, so
    the whole opening book is the single vector formed at offset 123. Resetting
    on the dipped Sunday would discard it and open the fold flat.
    """
    space = dipped_workspace(tmp_path, _WARM_UP_DIP_OFFSET)
    root, capture_root, config_path = space
    spec, _ = load_trend_family_spec(config_path)
    histories = _fold_histories(capture_root, root / "manifest.json", FOLD_INDEX)
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

    episode = _episodes(document, "ta_ts_t02_h4w")[0]
    expected = _expected_book(
        histories,
        spec,
        warm_ups + decisions[:1],
        name="ta_ts_t02_h4w",
        hold_weeks=4,
        decision_close_ns=decisions[0],
    )
    assert expected  # the surviving warm-up vector is still in the book
    assert _decimal(episode["gross_exposure"]) == sum(
        (abs(weight) for _, weight in expected), Decimal(0)
    )
    assert _decimal(episode["net_exposure"]) == sum(
        (weight for _, weight in expected), Decimal(0)
    )
    assert _decimal(episode["gross_exposure"]) > 0
    assert episode["reason_codes"] == []


def test_members_are_registered(workspace: tuple[Path, Path, Path]) -> None:
    root, _, _ = workspace
    _run(workspace)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    family_id = uuid5(
        NAMESPACE_URL, f"{manifest['split_manifest_hash']}:trend_aggregate_panel_v1"
    )
    with MetadataRegistry(root / "registry.sqlite3") as registry:
        rows = registry.list_experiments(family_id)
    assert {row.candidate_name for row in rows} == {
        "ta_ts_t02",
        "ta_ts_t05",
        "ta_ts_t02_h4w",
        "ta_xs_q5",
    }
    assert all(row.outcome == "completed" for row in rows)


def test_report_is_immutable(workspace: tuple[Path, Path, Path]) -> None:
    _run(workspace)
    with pytest.raises(TrendFoldError, match="immutable"):
        _run(workspace)


def test_unknown_fold_index_is_rejected(workspace: tuple[Path, Path, Path]) -> None:
    with pytest.raises(TrendFoldError, match="not in the manifest"):
        _run(workspace, fold_index=9)


def test_family_spec_mismatch_is_rejected(workspace: tuple[Path, Path, Path]) -> None:
    root, capture_root, config_path = workspace
    other = json.loads(config_path.read_text(encoding="utf-8"))
    other["hypothesis"] = str(other["hypothesis"]) + " (different declaration)"
    other_path = root / "other-trend.json"
    other_path.write_text(json.dumps(other), encoding="utf-8")
    with pytest.raises(TrendFoldError, match="does not match the manifest"):
        run_trend_fold(
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
    with pytest.raises(TrendFoldError, match=message):
        _run(workspace)


def _history(contract_id: str, close_times: tuple[int, ...]) -> ContractHistory:
    return ContractHistory(
        contract_id=contract_id,
        instrument_id=contract_id,
        closes={close_time: Decimal(100) for close_time in close_times},
        quote_volumes={close_time: Decimal(1) for close_time in close_times},
        close_times=close_times,
    )


def test_cohort_book_drops_a_contract_without_a_bar_and_undeploys_its_capital() -> None:
    """A contract that cannot be traded at t leaves the book, and the share of
    capital its cohorts held stays undeployed rather than being spread over the
    survivors -- gross falls to one half, it is not renormalised back to one."""
    decision = 4 * WEEK_NS
    histories = {
        "A": _history("A", tuple(week * WEEK_NS for week in range(5))),
        "B": _history("B", tuple(week * WEEK_NS for week in range(4))),
    }
    vector = (("A", Decimal("0.5")), ("B", Decimal("0.5")))
    book = assemble_cohort_book(
        [(week * WEEK_NS, vector) for week in (1, 2, 3, 4)],
        hold_weeks=4,
        histories=histories,
        decision_close_ns=decision,
    )
    assert book == (("A", Decimal("0.5")),)


def test_cohort_book_deploys_one_quarter_per_retained_vector() -> None:
    decision = 4 * WEEK_NS
    histories = {"A": _history("A", tuple(week * WEEK_NS for week in range(5)))}
    vector = (("A", Decimal(1)),)
    assert assemble_cohort_book(
        [(4 * WEEK_NS, vector)], hold_weeks=4, histories=histories, decision_close_ns=decision
    ) == (("A", Decimal("0.25")),)
    assert assemble_cohort_book(
        [(3 * WEEK_NS, vector), (4 * WEEK_NS, vector)],
        hold_weeks=4,
        histories=histories,
        decision_close_ns=decision,
    ) == (("A", Decimal("0.5")),)
    # An empty vector -- a week whose member signalled nothing -- contributes
    # nothing, and its quarter of capital stays undeployed.
    assert assemble_cohort_book(
        [(week * WEEK_NS, vector if week % 2 else ()) for week in (1, 2, 3, 4)],
        hold_weeks=4,
        histories=histories,
        decision_close_ns=decision,
    ) == (("A", Decimal("0.5")),)


def test_cohort_book_ages_a_vector_out_by_the_calendar_not_by_list_position() -> None:
    """A Sunday on which the whole panel had no universe forms no vector at all.
    Counting list positions would then keep the oldest vector in the book for a
    fifth week; ageing on the calendar (`carry_signals.assemble_book`'s rule)
    retires it on time, and the gap simply stays undeployed."""
    decision = 4 * WEEK_NS
    histories = {"A": _history("A", tuple(week * WEEK_NS for week in range(5)))}
    vector = (("A", Decimal(1)),)
    # Four retained vectors, but the one formed four weeks ago has aged out and
    # the Sunday three weeks ago formed none, so three quarters are deployed.
    book = assemble_cohort_book(
        [(week * WEEK_NS, vector) for week in (0, 2, 3, 4)],
        hold_weeks=4,
        histories=histories,
        decision_close_ns=decision,
    )
    assert book == (("A", Decimal("0.75")),)


def test_cohort_book_drops_a_contract_whose_cohorts_cancel_exactly() -> None:
    decision = 4 * WEEK_NS
    histories = {"A": _history("A", tuple(week * WEEK_NS for week in range(5)))}
    book = assemble_cohort_book(
        [(3 * WEEK_NS, (("A", Decimal(1)),)), (4 * WEEK_NS, (("A", Decimal(-1)),))],
        hold_weeks=4,
        histories=histories,
        decision_close_ns=decision,
    )
    assert book == ()
