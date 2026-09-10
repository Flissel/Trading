import io
import json
import zipfile
from decimal import Decimal
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest

from trading_bot import panel_fold_run as panel_fold_run_module
from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.panel_capture import PanelPayload, capture_panel
from trading_bot.panel_config import PanelFamilySpec, load_panel_family_spec
from trading_bot.panel_decision import _pool
from trading_bot.panel_fold_run import (
    MEMBER_HELD_NOTHING_REASON_CODE,
    PanelFoldError,
    run_panel_fold,
    verify_panel_fold_report,
)
from trading_bot.panel_reader import load_panel_bars
from trading_bot.panel_samples import publish_panel_walk_forward
from trading_bot.panel_signals import WeightVector, build_weight_vectors
from trading_bot.panel_universe import (
    ContractHistory,
    UniverseSnapshot,
    build_contract_histories,
    select_universe,
)
from trading_bot.registry import MetadataRegistry

DAY_MS = 86_400_000
MONTHS = ("2020-01", "2020-02", "2020-03", "2020-04", "2020-05", "2020-06", "2020-07")
SYMBOLS = tuple(f"C{index:02d}USDT" for index in range(12))
KLINE_HEADER = (
    "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
    "taker_buy_volume,taker_buy_quote_volume,ignore"
)
MONTH_START_DAY = {
    "2020-01": 0,
    "2020-02": 31,
    "2020-03": 60,
    "2020-04": 91,
    "2020-05": 121,
    "2020-06": 152,
    "2020-07": 182,
}
MONTH_DAYS = {
    "2020-01": 31,
    "2020-02": 29,
    "2020-03": 31,
    "2020-04": 30,
    "2020-05": 31,
    "2020-06": 30,
    "2020-07": 31,
}
EPOCH_DAY_2020 = 18_262  # 2020-01-01 in days since the Unix epoch


def zip_bytes(name: str, text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, text)
    return buffer.getvalue()


def kline_csv(symbol: str, month: str) -> str:
    seed = int(symbol[1:3])
    lines = [KLINE_HEADER]
    for offset in range(MONTH_DAYS[month]):
        day = EPOCH_DAY_2020 + MONTH_START_DAY[month] + offset
        open_ms = day * DAY_MS
        close = 100 + seed * 10 + (day % 11) + (seed * (day % 5)) / 4
        lines.append(
            f"{open_ms},{close},{close},{close},{close},10,{open_ms + DAY_MS - 1},"
            f"50000000,100,50,25000000,0"
        )
    return "\n".join(lines) + "\n"


def fetch(url: str) -> PanelPayload:
    symbol = next(item for item in SYMBOLS if f"/{item}/" in url or f"/{item}-" in url)
    month = next(item for item in MONTHS if item in url)
    if "fundingRate" in url:
        day = EPOCH_DAY_2020 + MONTH_START_DAY[month]
        text = "calc_time,funding_interval_hours,last_funding_rate\n"
        text += f"{day * DAY_MS},8,0.0001\n"
        return PanelPayload(url=url, raw_bytes=zip_bytes("f.csv", text), received_time_ns=1)
    return PanelPayload(
        url=url, raw_bytes=zip_bytes("k.csv", kline_csv(symbol, month)), received_time_ns=1
    )


# Under `small_config`'s fold geometry, fold 0's test window holds exactly three weekly
# Sundays at day offsets 109, 116, and 123 (confirmed empirically against this fixture).
# Dropping every symbol's quote volume across the 5-day liquidity window ending on day 116
# (`liquidity_window_days` is 5 below) pushes every contract's median volume under the base
# config's unmodified `minimum_median_quote_volume` floor for that one decision only, so the
# middle week's universe comes back empty while the weeks before and after are untouched.
_LIQUIDITY_DIP_TARGET_OFFSET = 116
_LIQUIDITY_DIP_WINDOW_DAYS = 5
_LIQUIDITY_DIP_DAYS = frozenset(
    EPOCH_DAY_2020 + offset
    for offset in range(
        _LIQUIDITY_DIP_TARGET_OFFSET - _LIQUIDITY_DIP_WINDOW_DAYS + 1,
        _LIQUIDITY_DIP_TARGET_OFFSET + 1,
    )
)


def kline_csv_with_liquidity_dip(symbol: str, month: str) -> str:
    seed = int(symbol[1:3])
    lines = [KLINE_HEADER]
    for offset in range(MONTH_DAYS[month]):
        day = EPOCH_DAY_2020 + MONTH_START_DAY[month] + offset
        open_ms = day * DAY_MS
        close = 100 + seed * 10 + (day % 11) + (seed * (day % 5)) / 4
        quote_volume = "1000000" if day in _LIQUIDITY_DIP_DAYS else "50000000"
        lines.append(
            f"{open_ms},{close},{close},{close},{close},10,{open_ms + DAY_MS - 1},"
            f"{quote_volume},100,50,25000000,0"
        )
    return "\n".join(lines) + "\n"


def fetch_with_liquidity_dip(url: str) -> PanelPayload:
    symbol = next(item for item in SYMBOLS if f"/{item}/" in url or f"/{item}-" in url)
    month = next(item for item in MONTHS if item in url)
    if "fundingRate" in url:
        day = EPOCH_DAY_2020 + MONTH_START_DAY[month]
        text = "calc_time,funding_interval_hours,last_funding_rate\n"
        text += f"{day * DAY_MS},8,0.0001\n"
        return PanelPayload(url=url, raw_bytes=zip_bytes("f.csv", text), received_time_ns=1)
    return PanelPayload(
        url=url,
        raw_bytes=zip_bytes("k.csv", kline_csv_with_liquidity_dip(symbol, month)),
        received_time_ns=1,
    )


def small_config(tmp_path: Path) -> Path:
    document = json.loads(Path("configs/xs-momentum-panel-v1.json").read_text(encoding="utf-8"))
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
    day_ns = 86_400_000_000_000
    document["folds"] = {
        "train_duration_ns": 60 * day_ns,
        "validation_duration_ns": 14 * day_ns,
        "test_duration_ns": 28 * day_ns,
        "step_ns": 28 * day_ns,
        "embargo_ns": 14 * day_ns,
        "holdout_duration_ns": 28 * day_ns,
    }
    document["statistics"].update({"block_length": 2, "pooled_episode_floor": 4})
    path = tmp_path / "small-panel.json"
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
    config_path = small_config(tmp_path)
    spec, spec_hash = load_panel_family_spec(config_path)
    publish_panel_walk_forward(
        tmp_path / "capture",
        output_path=tmp_path / "manifest.json",
        spec=spec,
        family_spec_hash=spec_hash,
    )
    return tmp_path, tmp_path / "capture", config_path


def test_fold_report_carries_every_member_and_its_episodes(
    workspace: tuple[Path, Path, Path],
) -> None:
    root, capture_root, config_path = workspace
    artifact = run_panel_fold(
        capture_root,
        manifest_path=root / "manifest.json",
        family_spec_path=config_path,
        output_path=root / "fold0.json",
        registry_path=root / "registry.sqlite3",
        fold_index=0,
    )
    document = json.loads(artifact.output_path.read_text(encoding="utf-8"))
    assert document["status"] == "development_only"
    assert document["fold_index"] == 0
    names = [item["candidate_name"] for item in document["candidates"]]
    assert names == [
        "xs_mom_1w",
        "xs_mom_4w",
        "xs_mom_12w",
        "ts_mom_4w",
        "ts_mom_12w",
        "xs_rev_1w",
        "no_trade",
        "random_ranks",
        "passive_long_ew",
    ]
    momentum = next(
        item for item in document["candidates"] if item["candidate_name"] == "xs_mom_1w"
    )
    assert momentum["role"] == "member"
    assert momentum["episode_count"] == artifact.episode_count > 0
    assert len(momentum["base"]["episodes"]) == artifact.episode_count
    assert len(momentum["adverse"]["episodes"]) == artifact.episode_count
    episode = momentum["base"]["episodes"][0]
    assert set(episode) >= {
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
        "contract_net_contributions",
    }
    assert verify_panel_fold_report(artifact.output_path)


def test_no_trade_control_has_zero_returns(workspace: tuple[Path, Path, Path]) -> None:
    root, capture_root, config_path = workspace
    run_panel_fold(
        capture_root,
        manifest_path=root / "manifest.json",
        family_spec_path=config_path,
        output_path=root / "fold0.json",
        registry_path=root / "registry.sqlite3",
        fold_index=0,
    )
    document = json.loads((root / "fold0.json").read_text(encoding="utf-8"))
    no_trade = next(
        item for item in document["candidates"] if item["candidate_name"] == "no_trade"
    )
    assert all(episode["net_return"] == "0" for episode in no_trade["base"]["episodes"])


def test_members_are_registered(workspace: tuple[Path, Path, Path]) -> None:
    root, capture_root, config_path = workspace
    run_panel_fold(
        capture_root,
        manifest_path=root / "manifest.json",
        family_spec_path=config_path,
        output_path=root / "fold0.json",
        registry_path=root / "registry.sqlite3",
        fold_index=0,
    )
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    family_id = uuid5(NAMESPACE_URL, f"{manifest['split_manifest_hash']}:xs_momentum_panel_v1")
    with MetadataRegistry(root / "registry.sqlite3") as registry:
        rows = registry.list_experiments(family_id)
    assert {row.candidate_name for row in rows} == {
        "xs_mom_1w",
        "xs_mom_4w",
        "xs_mom_12w",
        "ts_mom_4w",
        "ts_mom_12w",
        "xs_rev_1w",
    }
    assert all(row.outcome == "completed" for row in rows)
    assert all(row.family_id == family_id for row in rows)


def test_report_is_immutable(workspace: tuple[Path, Path, Path]) -> None:
    root, capture_root, config_path = workspace
    for _ in range(1):
        run_panel_fold(
            capture_root,
            manifest_path=root / "manifest.json",
            family_spec_path=config_path,
            output_path=root / "fold0.json",
            registry_path=root / "registry.sqlite3",
            fold_index=0,
        )
    with pytest.raises(PanelFoldError):
        run_panel_fold(
            capture_root,
            manifest_path=root / "manifest.json",
            family_spec_path=config_path,
            output_path=root / "fold0.json",
            registry_path=root / "registry.sqlite3",
            fold_index=0,
        )


def test_unknown_fold_index_is_rejected(workspace: tuple[Path, Path, Path]) -> None:
    root, capture_root, config_path = workspace
    with pytest.raises(PanelFoldError):
        run_panel_fold(
            capture_root,
            manifest_path=root / "manifest.json",
            family_spec_path=config_path,
            output_path=root / "fold9.json",
            registry_path=root / "registry.sqlite3",
            fold_index=9,
        )


def test_skipped_week_resets_position_and_records_reason_code(tmp_path: Path) -> None:
    capture_panel(
        workspace_root=tmp_path,
        output_directory=tmp_path / "capture",
        reserve_bytes=0,
        symbols=SYMBOLS,
        months=MONTHS,
        fetch=fetch_with_liquidity_dip,
    )
    config_path = small_config(tmp_path)
    spec, spec_hash = load_panel_family_spec(config_path)
    publish_panel_walk_forward(
        tmp_path / "capture",
        output_path=tmp_path / "manifest.json",
        spec=spec,
        family_spec_hash=spec_hash,
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    fold0 = next(item for item in manifest["folds"] if item["fold_index"] == 0)
    test_ids = fold0["test_ids"]
    # Sanity on the fixture itself: two normal weeks bracket the one the dip hits.
    assert len(test_ids) == 3
    skipped_sample_id = test_ids[1]
    surviving_sample_ids = [test_ids[0], test_ids[2]]

    artifact = run_panel_fold(
        tmp_path / "capture",
        manifest_path=tmp_path / "manifest.json",
        family_spec_path=config_path,
        output_path=tmp_path / "fold0.json",
        registry_path=tmp_path / "registry.sqlite3",
        fold_index=0,
    )
    document = json.loads(artifact.output_path.read_text(encoding="utf-8"))

    # Consequence 1: the skipped sample id is recorded under `skipped_sample_ids`.
    assert document["skipped_sample_ids"] == [skipped_sample_id]

    # Consequence 2: the reason code is added to the report.
    assert "SKIPPED_WEEK_EXIT_COST_UNCHARGED" in document["reason_codes"]

    passive = next(
        item for item in document["candidates"] if item["candidate_name"] == "passive_long_ew"
    )
    episodes = passive["base"]["episodes"]

    # Consequence 3: episode count is short of the test-sample count by exactly the number
    # of skipped weeks, and the surviving episodes are the two weeks around the gap.
    assert document["test_sample_count"] - passive["episode_count"] == len(
        document["skipped_sample_ids"]
    )
    assert [item["sample_id"] for item in episodes] == surviving_sample_ids

    # Consequence 4: the episode right after the skip is charged full entry turnover —
    # turnover equals gross exposure, which only holds if the running position going into
    # that week was flat rather than the drifted weights carried across the gap.
    resumed = episodes[1]
    assert resumed["turnover"] == resumed["gross_exposure"]
    assert Decimal(resumed["gross_exposure"]) > 0


def test_family_spec_mismatch_is_rejected(workspace: tuple[Path, Path, Path]) -> None:
    root, capture_root, config_path = workspace
    other_document = json.loads(config_path.read_text(encoding="utf-8"))
    other_document["hypothesis"] = other_document["hypothesis"] + " (different declaration)"
    other_config_path = root / "other-panel.json"
    other_config_path.write_text(json.dumps(other_document), encoding="utf-8")

    with pytest.raises(PanelFoldError):
        run_panel_fold(
            capture_root,
            manifest_path=root / "manifest.json",
            family_spec_path=other_config_path,
            output_path=root / "fold0-mismatch.json",
            registry_path=root / "registry.sqlite3",
            fold_index=0,
        )


def _rewrite_manifest_field(path: Path, key: str, value: object) -> None:
    """Tamper one field of a published manifest and re-derive `manifest_hash` so the
    file stays internally self-consistent (passes `verify_panel_manifest`) while
    carrying a value that no longer matches its source-of-truth elsewhere -- isolating
    whichever downstream linkage check is under test."""
    document = json.loads(path.read_text(encoding="utf-8"))
    document[key] = value
    material = {k: v for k, v in document.items() if k != "manifest_hash"}
    document["manifest_hash"] = content_sha256(material)
    path.write_bytes(canonical_json(document))


def test_capture_root_hash_linkage_is_enforced(workspace: tuple[Path, Path, Path]) -> None:
    root, capture_root, config_path = workspace
    _rewrite_manifest_field(root / "manifest.json", "capture_root_hash", "0" * 64)
    with pytest.raises(PanelFoldError, match="not linked to this capture"):
        run_panel_fold(
            capture_root,
            manifest_path=root / "manifest.json",
            family_spec_path=config_path,
            output_path=root / "fold0.json",
            registry_path=root / "registry.sqlite3",
            fold_index=0,
        )


def test_dataset_root_hash_linkage_is_enforced(workspace: tuple[Path, Path, Path]) -> None:
    root, capture_root, config_path = workspace
    _rewrite_manifest_field(root / "manifest.json", "dataset_root_hash", "0" * 64)
    with pytest.raises(PanelFoldError, match="not linked to this dataset"):
        run_panel_fold(
            capture_root,
            manifest_path=root / "manifest.json",
            family_spec_path=config_path,
            output_path=root / "fold0.json",
            registry_path=root / "registry.sqlite3",
            fold_index=0,
        )


def test_uncharged_final_exit_reason_code_is_recorded(
    workspace: tuple[Path, Path, Path],
) -> None:
    """The fold's carried position resets to empty at fold start (so the first
    episode pays a full entry turnover) but the last episode's position is never
    closed out (so it pays no exit) -- one side of unit gross, uncharged, always in
    the favourable direction. Pre-registered accounting semantics must not change to
    start charging it, but the omission must be visible in the report."""
    root, capture_root, config_path = workspace
    artifact = run_panel_fold(
        capture_root,
        manifest_path=root / "manifest.json",
        family_spec_path=config_path,
        output_path=root / "fold0.json",
        registry_path=root / "registry.sqlite3",
        fold_index=0,
    )
    document = json.loads(artifact.output_path.read_text(encoding="utf-8"))
    assert artifact.episode_count > 0
    assert "FOLD_FINAL_EXIT_COST_UNCHARGED" in document["reason_codes"]


def test_universe_and_weights_are_point_in_time_across_the_fold(
    workspace: tuple[Path, Path, Path],
) -> None:
    """Regression test for the point-in-time guarantee `run_panel_fold` relies on:
    for every test decision of fold 0, `select_universe` and `build_weight_vectors`
    must return identical results whether given the fold-wide histories `run_panel_
    fold` actually loads, or histories truncated to bars closing at or before that
    one decision. The truncated histories are built by filtering the already-loaded
    bars (never by re-reading), matching how `run_panel_fold` builds its own
    histories once per fold rather than once per decision."""
    root, capture_root, config_path = workspace
    spec, _ = load_panel_family_spec(config_path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    fold0 = next(item for item in manifest["folds"] if item["fold_index"] == 0)
    test_end_ns = int(fold0["test_end_ns"])
    bars = load_panel_bars(capture_root / "dataset", available_before_ns=test_end_ns + 1)
    fold_wide_histories = build_contract_histories(bars)
    decisions = sorted(int(value.split(":")[1]) for value in fold0["test_ids"])
    assert decisions  # sanity: the fixture fold actually has test decisions

    for decision_close_ns in decisions:
        truncated_bars = tuple(bar for bar in bars if bar.close_time_ns <= decision_close_ns)
        truncated_histories = build_contract_histories(truncated_bars)

        full_snapshot = select_universe(
            fold_wide_histories, decision_close_ns=decision_close_ns, rules=spec.universe
        )
        truncated_snapshot = select_universe(
            truncated_histories, decision_close_ns=decision_close_ns, rules=spec.universe
        )
        assert truncated_snapshot == full_snapshot

        full_vectors = build_weight_vectors(fold_wide_histories, full_snapshot, spec=spec)
        truncated_vectors = build_weight_vectors(
            truncated_histories, truncated_snapshot, spec=spec
        )
        assert truncated_vectors == full_vectors


def test_fold_wide_bars_are_bounded_exactly_at_the_test_end_boundary(
    workspace: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pins the exact availability boundary `run_panel_fold` uses to load its
    fold-wide bars. Widening `test_end_ns + 1` by even a day would extend the fold's
    information set past its test boundary -- into the embargo gap or beyond -- with
    no visible symptom other than this pin breaking."""
    root, capture_root, config_path = workspace
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    fold0 = next(item for item in manifest["folds"] if item["fold_index"] == 0)
    test_end_ns = int(fold0["test_end_ns"])

    original_load_panel_bars = load_panel_bars
    captured_boundaries: list[int | None] = []

    def spy(dataset_root: Path, *, available_before_ns: int | None = None) -> object:
        captured_boundaries.append(available_before_ns)
        return original_load_panel_bars(dataset_root, available_before_ns=available_before_ns)

    monkeypatch.setattr(panel_fold_run_module, "load_panel_bars", spy)
    run_panel_fold(
        capture_root,
        manifest_path=root / "manifest.json",
        family_spec_path=config_path,
        output_path=root / "fold0.json",
        registry_path=root / "registry.sqlite3",
        fold_index=0,
    )
    assert captured_boundaries == [test_end_ns + 1]


def test_member_held_nothing_is_marked_and_controls_are_never_marked(
    workspace: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A member's own construction can legitimately return no weights (too few
    rankable contracts to form both cross-sectional quintiles, or none with a full
    volatility window) even though the week's universe was not too small. Such an
    episode must be visibly marked MEMBER_HELD_NOTHING, not recorded as an ordinary,
    unremarkable zero-return episode -- and a control whose weights are also empty
    by design (`no_trade`, always) must never be marked, since its emptiness is a
    deliberate baseline, not a data-insufficiency artifact.

    Round 1 review: forcing the emptiness at the fold's *first* decision (as this
    test originally did) is the one index where the bug this guards against is
    invisible -- the carried position is empty there regardless, so the forced
    episode's net_return and turnover are exactly zero and dropping it loses
    nothing. Decision 2 of this fold's 3 carries a real position in from decision
    1, so forcing it there charges a genuine unwind: a strictly negative net
    return and non-zero turnover. That is the case that actually exercises
    `panel_decision.py`'s cost-conservation fix (see
    test_panel_decision.test_member_held_nothing_episode_is_excluded_but_its_cost_is_rolled_forward
    for the pooling side); this test pins that no money vanishes from the pooled
    total the real fold runner produces."""
    root, capture_root, config_path = workspace
    original_build_weight_vectors = build_weight_vectors
    calls = {"count": 0}

    def flaky_vectors(
        histories: dict[str, ContractHistory],
        snapshot: UniverseSnapshot,
        *,
        spec: PanelFamilySpec,
    ) -> dict[str, WeightVector]:
        vectors = original_build_weight_vectors(histories, snapshot, spec=spec)
        calls["count"] += 1
        if calls["count"] == 2:
            # Force xs_mom_1w to have produced no weights on the fold's *second*
            # decision -- a position is already carried into it from the first,
            # so the enforced emptiness charges a real unwind, not a free zero --
            # without touching the universe snapshot itself, which stays
            # non-empty.
            existing = vectors["xs_mom_1w"]
            forced = dict(vectors)
            forced["xs_mom_1w"] = WeightVector(
                existing.decision_close_ns, existing.member, (), existing.reason_codes
            )
            return forced
        return vectors

    monkeypatch.setattr(panel_fold_run_module, "build_weight_vectors", flaky_vectors)

    artifact = run_panel_fold(
        capture_root,
        manifest_path=root / "manifest.json",
        family_spec_path=config_path,
        output_path=root / "fold0.json",
        registry_path=root / "registry.sqlite3",
        fold_index=0,
    )
    document: dict[str, object] = json.loads(artifact.output_path.read_text(encoding="utf-8"))
    candidates = document["candidates"]
    assert isinstance(candidates, list)
    momentum = next(
        item for item in candidates if item["candidate_name"] == "xs_mom_1w"
    )
    # Sanity on the fixture: exactly 3 decisions in this fold, so index 1 (the
    # second) is the one forced above, and it is neither the first nor the last.
    assert calls["count"] == 3
    assert len(momentum["base"]["episodes"]) == 3

    for scenario in ("base", "adverse"):
        episodes = momentum[scenario]["episodes"]
        reason_codes = [episode["reason_codes"] for episode in episodes]
        assert reason_codes == [[], [MEMBER_HELD_NOTHING_REASON_CODE], []]
        # The forced episode carries a real position into an empty target: a
        # genuine unwind, not the free zero a first-decision force would produce.
        assert Decimal(episodes[1]["net_return"]) < 0
        assert Decimal(episodes[1]["turnover"]) > 0

    no_trade = next(item for item in candidates if item["candidate_name"] == "no_trade")
    assert all(episode["reason_codes"] == [] for episode in no_trade["base"]["episodes"])
    assert all(episode["reason_codes"] == [] for episode in no_trade["adverse"]["episodes"])

    # The reviewer's exact reproduction: reproduce the raw (ground-truth) pooled
    # total by summing every episode including the held-nothing one, and confirm
    # panel_decision's pooling -- which now excludes that episode from the
    # observation series but rolls its net_return into the next retained episode
    # -- lands on that same raw total, to the last digit, in both scenarios.
    pooled = _pool([document], "xs_mom_1w")
    raw_base_total = sum(
        (Decimal(episode["net_return"]) for episode in momentum["base"]["episodes"]), Decimal(0)
    )
    raw_adverse_total = sum(
        (Decimal(episode["net_return"]) for episode in momentum["adverse"]["episodes"]),
        Decimal(0),
    )
    assert pooled.base_total == raw_base_total
    assert pooled.adverse_total == raw_adverse_total
    assert pooled.episode_count == len(momentum["base"]["episodes"]) - 1

