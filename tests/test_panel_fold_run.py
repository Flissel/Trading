import io
import json
import zipfile
from pathlib import Path

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
    with MetadataRegistry(root / "registry.sqlite3") as registry:
        rows = registry.list_experiments()
    assert {row.candidate_name for row in rows} == {
        "xs_mom_1w",
        "xs_mom_4w",
        "xs_mom_12w",
        "ts_mom_4w",
        "ts_mom_12w",
        "xs_rev_1w",
    }
    assert all(row.outcome == "completed" for row in rows)


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
