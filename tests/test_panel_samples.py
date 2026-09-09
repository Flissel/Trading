import io
import json
import zipfile
from decimal import Decimal
from itertools import pairwise
from pathlib import Path

import pytest

from trading_bot.panel_capture import PanelPayload, capture_panel
from trading_bot.panel_config import PanelFoldGeometry, load_panel_family_spec
from trading_bot.panel_reader import PanelBar
from trading_bot.panel_samples import (
    PanelSamplesError,
    build_rebalance_samples,
    derive_panel_config,
    publish_panel_walk_forward,
    rebalance_close_times,
)
from trading_bot.splits import build_walk_forward_views

DAY_NS = 86_400_000_000_000
DAY_MS = 86_400_000
SPEC, _ = load_panel_family_spec(Path("configs/xs-momentum-panel-v1.json"))


def bar(day_index: int) -> PanelBar:
    open_time_ns = day_index * DAY_NS
    close_time_ns = open_time_ns + DAY_NS - 1_000_000
    return PanelBar(
        contract_id="BTCUSDT:0",
        instrument_id="BTCUSDT",
        open_time_ns=open_time_ns,
        close_time_ns=close_time_ns,
        available_time_ns=close_time_ns + 1,
        close=Decimal("100"),
        quote_volume=Decimal("1000"),
    )


def test_rebalance_dates_are_sundays() -> None:
    bars = tuple(bar(index) for index in range(21))
    times = rebalance_close_times(bars)
    assert [value // DAY_NS for value in times] == [3, 10, 17]


def test_samples_are_chronological_and_carry_the_holding_label() -> None:
    bars = tuple(bar(index) for index in range(21))
    samples = build_rebalance_samples(bars, holding_days=7)
    assert samples[0].sample_id == f"BINANCE_UM:{4 * DAY_NS - 1_000_000}:w1"
    assert samples[0].label_end_time_ns - samples[0].decision_time_ns == 7 * DAY_NS
    assert all(
        later.decision_time_ns > earlier.decision_time_ns
        for earlier, later in pairwise(samples)
    )


def test_config_places_the_holdout_at_the_end() -> None:
    bars = tuple(bar(index) for index in range(400))
    samples = build_rebalance_samples(bars, holding_days=7)
    geometry = PanelFoldGeometry(
        train_duration_ns=100 * DAY_NS,
        validation_duration_ns=20 * DAY_NS,
        test_duration_ns=40 * DAY_NS,
        step_ns=40 * DAY_NS,
        embargo_ns=14 * DAY_NS,
        holdout_duration_ns=40 * DAY_NS,
    )
    config = derive_panel_config(samples, folds=geometry)
    assert config.final_holdout_start_ns < samples[-1].decision_time_ns
    assert config.final_holdout_start_ns > samples[0].decision_time_ns


def test_holdout_is_disjoint_from_every_fold_partition() -> None:
    bars = tuple(bar(index) for index in range(400))
    samples = build_rebalance_samples(bars, holding_days=7)
    geometry = PanelFoldGeometry(
        train_duration_ns=100 * DAY_NS,
        validation_duration_ns=20 * DAY_NS,
        test_duration_ns=40 * DAY_NS,
        step_ns=40 * DAY_NS,
        embargo_ns=14 * DAY_NS,
        holdout_duration_ns=40 * DAY_NS,
    )
    config = derive_panel_config(samples, folds=geometry)
    views = build_walk_forward_views(list(samples), config)
    holdout_ids = set(views.final_holdout_ids)
    fold_ids = {
        sample_id
        for fold in views.folds
        for sample_id in (*fold.train_ids, *fold.validation_ids, *fold.test_ids)
    }
    # A non-empty holdout guards against the disjointness check passing vacuously.
    assert holdout_ids
    assert holdout_ids.isdisjoint(fold_ids)


def test_short_history_is_rejected() -> None:
    samples = build_rebalance_samples(tuple(bar(index) for index in range(10)), holding_days=7)
    with pytest.raises(PanelSamplesError):
        derive_panel_config(samples, folds=SPEC.folds)


_FLOOR_TEST_MONTHS = ("2020-01", "2020-02", "2020-03")
_FLOOR_TEST_MONTH_START_DAY = {"2020-01": 0, "2020-02": 31, "2020-03": 60}
_FLOOR_TEST_MONTH_DAYS = {"2020-01": 31, "2020-02": 29, "2020-03": 31}
_FLOOR_TEST_KLINE_HEADER = (
    "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
    "taker_buy_volume,taker_buy_quote_volume,ignore"
)


def _zip_bytes(name: str, text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, text)
    return buffer.getvalue()


def _floor_test_kline_csv(month: str) -> str:
    lines = [_FLOOR_TEST_KLINE_HEADER]
    for offset in range(_FLOOR_TEST_MONTH_DAYS[month]):
        day = _FLOOR_TEST_MONTH_START_DAY[month] + offset
        open_ms = day * DAY_MS
        lines.append(
            f"{open_ms},100,101,99,100,10,{open_ms + DAY_MS - 1},50000000,100,50,25000000,0"
        )
    return "\n".join(lines) + "\n"


def _floor_test_fetch(url: str) -> PanelPayload:
    month = next(item for item in _FLOOR_TEST_MONTHS if item in url)
    if "fundingRate" in url:
        text = "calc_time,funding_interval_hours,last_funding_rate\n0,8,0.0001\n"
        return PanelPayload(url=url, raw_bytes=_zip_bytes("f.csv", text), received_time_ns=1)
    return PanelPayload(
        url=url, raw_bytes=_zip_bytes("k.csv", _floor_test_kline_csv(month)), received_time_ns=1
    )


def test_pooled_sample_floor_is_enforced(tmp_path: Path) -> None:
    # Three months of daily data (91 days, ~13 weekly Sundays) under a tiny fold
    # geometry produces a real, complete fold -- so the earlier "no complete fold"
    # guard does not fire -- but that fold pools only 2 test samples, far under an
    # inflated `pooled_episode_floor` of 1000. This is the INSUFFICIENT_EVIDENCE
    # guard in `publish_panel_walk_forward`, confirmed by mutation to be undetected
    # by the rest of the suite.
    capture_panel(
        workspace_root=tmp_path,
        output_directory=tmp_path / "capture",
        reserve_bytes=0,
        symbols=("BTCUSDT",),
        months=_FLOOR_TEST_MONTHS,
        fetch=_floor_test_fetch,
    )
    document = json.loads(Path("configs/xs-momentum-panel-v1.json").read_text(encoding="utf-8"))
    document["folds"] = {
        "train_duration_ns": 20 * DAY_NS,
        "validation_duration_ns": 7 * DAY_NS,
        "test_duration_ns": 14 * DAY_NS,
        "step_ns": 14 * DAY_NS,
        "embargo_ns": 7 * DAY_NS,
        "holdout_duration_ns": 14 * DAY_NS,
    }
    document["statistics"]["pooled_episode_floor"] = 1000
    config_path = tmp_path / "tiny-panel.json"
    config_path.write_text(json.dumps(document), encoding="utf-8")
    spec, spec_hash = load_panel_family_spec(config_path)

    with pytest.raises(PanelSamplesError, match="INSUFFICIENT_EVIDENCE"):
        publish_panel_walk_forward(
            tmp_path / "capture",
            output_path=tmp_path / "manifest.json",
            spec=spec,
            family_spec_hash=spec_hash,
        )
