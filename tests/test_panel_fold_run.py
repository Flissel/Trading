import io
import json
import zipfile
from decimal import Decimal
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest

from trading_bot.panel_capture import PanelPayload, capture_panel
from trading_bot.panel_config import load_panel_family_spec
from trading_bot.panel_fold_run import PanelFoldError, run_panel_fold, verify_panel_fold_report
from trading_bot.panel_samples import publish_panel_walk_forward
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
