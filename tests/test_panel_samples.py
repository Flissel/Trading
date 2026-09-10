import io
import json
import zipfile
from collections.abc import Callable
from datetime import date, timedelta
from decimal import Decimal
from itertools import pairwise
from pathlib import Path

import pytest

from trading_bot.panel_capture import PanelPayload, PanelSourceAbsent, capture_panel
from trading_bot.panel_config import PanelFoldGeometry, load_panel_family_spec
from trading_bot.panel_reader import PanelBar
from trading_bot.panel_samples import (
    _MAX_MISSING_DAYS_PER_INSTRUMENT,
    PanelSamplesError,
    _enforce_capture_quality,
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


_QUALITY_TEST_MONTH = "2020-01"
_QUALITY_TEST_MONTH_DAYS = 31


def _quality_gap_kline_csv(skip: frozenset[int]) -> str:
    lines = [_FLOOR_TEST_KLINE_HEADER]
    for offset in range(_QUALITY_TEST_MONTH_DAYS):
        if offset in skip:
            continue
        open_ms = offset * DAY_MS
        lines.append(
            f"{open_ms},100,101,99,100,10,{open_ms + DAY_MS - 1},50000000,100,50,25000000,0"
        )
    return "\n".join(lines) + "\n"


def _quality_gap_fetch(skip: frozenset[int]) -> Callable[[str], PanelPayload]:
    def fetch(url: str) -> PanelPayload:
        if "/daily/klines/" in url:
            # The daily dumps are missing these days too (a genuine absence,
            # not merely a hole in the monthly aggregate): `capture_panel`
            # still records the attempt as `klines_daily_fill` / "absent",
            # which is exactly what the quality gate now needs to tell this
            # apart from a capture defect it could have prevented.
            raise PanelSourceAbsent("404: no daily dump for this day either")
        if "fundingRate" in url:
            text = "calc_time,funding_interval_hours,last_funding_rate\n0,8,0.0001\n"
            return PanelPayload(url=url, raw_bytes=_zip_bytes("f.csv", text), received_time_ns=1)
        return PanelPayload(
            url=url,
            raw_bytes=_zip_bytes("k.csv", _quality_gap_kline_csv(skip)),
            received_time_ns=1,
        )

    return fetch


def test_capture_quality_gate_forgives_days_the_daily_dump_also_lacks(tmp_path: Path) -> None:
    """BTCUSDT's January dump is missing five interior days (offsets 10-14), and
    the daily dump 404s on every one of them too (see `_quality_gap_fetch`) --
    `capture_panel` records each as `klines_daily_fill` / "absent". These are
    proven absent at the source, not a defect this capture could have
    prevented (the real-world case is ICPUSDT, 2022-09-22 through 2022-09-26,
    absent from both Binance dumps), so the quality gate must not stop the
    family over them, unlike before this fix."""
    capture_panel(
        workspace_root=tmp_path,
        output_directory=tmp_path / "capture",
        reserve_bytes=0,
        symbols=("BTCUSDT",),
        months=(_QUALITY_TEST_MONTH,),
        fetch=_quality_gap_fetch(frozenset({10, 11, 12, 13, 14})),
    )
    with pytest.raises(PanelSamplesError) as excinfo:
        publish_panel_walk_forward(
            tmp_path / "capture",
            output_path=tmp_path / "manifest.json",
            spec=SPEC,
            family_spec_hash="a" * 64,
        )
    # One month of one symbol still fails downstream (too little data for a
    # real fold geometry), the same way `test_capture_quality_stop_does_not_
    # misfire_at_the_threshold` does -- the point here is exclusively that it
    # is not CAPTURE_QUALITY_FAILED that stops it.
    assert "CAPTURE_QUALITY_FAILED" not in str(excinfo.value)


def test_capture_quality_stop_does_not_misfire_at_the_threshold(tmp_path: Path) -> None:
    # Exactly 3 missing days (offsets 10-12) is at, not over, the threshold -- the
    # quality gate must not fire. The run still fails downstream (one month of one
    # symbol cannot satisfy the real fold geometry), but not for CAPTURE_QUALITY_FAILED.
    capture_panel(
        workspace_root=tmp_path,
        output_directory=tmp_path / "capture",
        reserve_bytes=0,
        symbols=("BTCUSDT",),
        months=(_QUALITY_TEST_MONTH,),
        fetch=_quality_gap_fetch(frozenset({10, 11, 12})),
    )
    with pytest.raises(PanelSamplesError) as excinfo:
        publish_panel_walk_forward(
            tmp_path / "capture",
            output_path=tmp_path / "manifest.json",
            spec=SPEC,
            family_spec_hash="a" * 64,
        )
    assert "CAPTURE_QUALITY_FAILED" not in str(excinfo.value)


def test_manifest_records_the_capture_quality_threshold(tmp_path: Path) -> None:
    # The threshold a capture was accepted under belongs in the artifact itself,
    # not only in a CAPTURE_QUALITY_FAILED message nobody sees once a manifest
    # does get built.
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
    document["statistics"]["pooled_episode_floor"] = 1
    config_path = tmp_path / "tiny-panel.json"
    config_path.write_text(json.dumps(document), encoding="utf-8")
    spec, spec_hash = load_panel_family_spec(config_path)

    artifact = publish_panel_walk_forward(
        tmp_path / "capture",
        output_path=tmp_path / "manifest.json",
        spec=spec,
        family_spec_hash=spec_hash,
    )
    manifest = json.loads(artifact.output_path.read_text(encoding="utf-8"))
    assert (
        manifest["capture_quality_max_missing_days_per_instrument"]
        == _MAX_MISSING_DAYS_PER_INSTRUMENT
    )
    # No gaps in this capture -- the block is always present, even when empty.
    assert manifest["absent_at_source_days"] == {}


_GAP_DAY_OFFSETS = frozenset({40, 41, 42, 43, 44})  # global day offsets, inside February


def _floor_test_kline_csv_with_gap(month: str, skip: frozenset[int]) -> str:
    lines = [_FLOOR_TEST_KLINE_HEADER]
    for offset in range(_FLOOR_TEST_MONTH_DAYS[month]):
        day = _FLOOR_TEST_MONTH_START_DAY[month] + offset
        if day in skip:
            continue
        open_ms = day * DAY_MS
        lines.append(
            f"{open_ms},100,101,99,100,10,{open_ms + DAY_MS - 1},50000000,100,50,25000000,0"
        )
    return "\n".join(lines) + "\n"


def _floor_test_fetch_with_forgiven_gap(url: str) -> PanelPayload:
    if "/daily/klines/" in url:
        # Attempted and proven absent at source too -- the ICPUSDT shape.
        raise PanelSourceAbsent("404: no daily dump for this day either")
    month = next(item for item in _FLOOR_TEST_MONTHS if item in url)
    if "fundingRate" in url:
        text = "calc_time,funding_interval_hours,last_funding_rate\n0,8,0.0001\n"
        return PanelPayload(url=url, raw_bytes=_zip_bytes("f.csv", text), received_time_ns=1)
    return PanelPayload(
        url=url,
        raw_bytes=_zip_bytes("k.csv", _floor_test_kline_csv_with_gap(month, _GAP_DAY_OFFSETS)),
        received_time_ns=1,
    )


def test_manifest_names_the_absent_at_source_days_it_forgave(tmp_path: Path) -> None:
    """The gate not stopping the family over a proven-absent gap (previous test)
    is only half the requirement: the limitation must travel in the immutable
    walk-forward manifest itself, per instrument, naming the exact days --
    not merely live in a log line nobody keeps."""
    capture_panel(
        workspace_root=tmp_path,
        output_directory=tmp_path / "capture",
        reserve_bytes=0,
        symbols=("BTCUSDT",),
        months=_FLOOR_TEST_MONTHS,
        fetch=_floor_test_fetch_with_forgiven_gap,
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
    document["statistics"]["pooled_episode_floor"] = 1
    config_path = tmp_path / "tiny-panel.json"
    config_path.write_text(json.dumps(document), encoding="utf-8")
    spec, spec_hash = load_panel_family_spec(config_path)

    artifact = publish_panel_walk_forward(
        tmp_path / "capture",
        output_path=tmp_path / "manifest.json",
        spec=spec,
        family_spec_hash=spec_hash,
    )
    manifest = json.loads(artifact.output_path.read_text(encoding="utf-8"))
    expected_dates = [
        (date(1970, 1, 1) + timedelta(days=day)).isoformat()
        for day in sorted(_GAP_DAY_OFFSETS)
    ]
    assert manifest["absent_at_source_days"] == {"BTCUSDT": expected_dates}


def _write_json(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def _daily_fill_sources(symbol: str, dates: tuple[str, ...], *, status: str) -> dict[str, object]:
    return {
        "sources": [
            {
                "symbol": symbol,
                "month": day,
                "kind": "klines_daily_fill",
                "url": f"https://data.binance.vision/x/{symbol}-1d-{day}.zip",
                "status": status,
            }
            for day in dates
        ]
    }


def test_enforce_capture_quality_forgives_days_proven_absent_at_source(tmp_path: Path) -> None:
    quality_report = tmp_path / "quality-report.json"
    _write_json(
        quality_report,
        {"instruments": [{"instrument_id": "ICPUSDT", "missing_days": 5, "row_count": 1}]},
    )
    icpusdt_gap = (
        "2022-09-22",
        "2022-09-23",
        "2022-09-24",
        "2022-09-25",
        "2022-09-26",
    )
    capture_manifest = _daily_fill_sources("ICPUSDT", icpusdt_gap, status="absent")
    absent = _enforce_capture_quality(quality_report, capture_manifest)
    assert absent == {"ICPUSDT": icpusdt_gap}


def test_enforce_capture_quality_still_trips_on_an_unattempted_gap(tmp_path: Path) -> None:
    """A missing day the capture never attempted to patch from the daily dump --
    or attempted and failed for a reason other than an absence there too -- is
    still a capture defect. Forgiving unattempted gaps would let a genuinely
    defective capture pass just because it also happens to be missing days
    Binance never published."""
    quality_report = tmp_path / "quality-report.json"
    _write_json(
        quality_report,
        {"instruments": [{"instrument_id": "BTCUSDT", "missing_days": 5, "row_count": 1}]},
    )
    capture_manifest: dict[str, object] = {"sources": []}
    with pytest.raises(PanelSamplesError, match="CAPTURE_QUALITY_FAILED") as excinfo:
        _enforce_capture_quality(quality_report, capture_manifest)
    assert "BTCUSDT" in str(excinfo.value)
    assert "5 missing days" in str(excinfo.value)


def test_enforce_capture_quality_ignores_an_attempt_still_marked_present(tmp_path: Path) -> None:
    """A `klines_daily_fill` entry recorded `status: present` means the gap was
    filled, not left absent -- it must not be double-counted as a proven
    absence on top of whatever `missing_days` already reflects."""
    quality_report = tmp_path / "quality-report.json"
    _write_json(
        quality_report,
        {"instruments": [{"instrument_id": "BTCUSDT", "missing_days": 5, "row_count": 1}]},
    )
    capture_manifest = _daily_fill_sources(
        "BTCUSDT", ("2020-01-11", "2020-01-12", "2020-01-13", "2020-01-14", "2020-01-15"),
        status="present",
    )
    with pytest.raises(PanelSamplesError, match="CAPTURE_QUALITY_FAILED") as excinfo:
        _enforce_capture_quality(quality_report, capture_manifest)
    assert "5 missing days" in str(excinfo.value)


def test_enforce_capture_quality_only_credits_proven_absent_days(tmp_path: Path) -> None:
    """Partial credit: one of five missing days is proven absent at source, the
    other four are not -- four unexplained missing days is over the threshold,
    so the gate must still fire, on the net count rather than the raw one."""
    quality_report = tmp_path / "quality-report.json"
    _write_json(
        quality_report,
        {"instruments": [{"instrument_id": "BTCUSDT", "missing_days": 5, "row_count": 1}]},
    )
    capture_manifest = _daily_fill_sources("BTCUSDT", ("2020-01-11",), status="absent")
    with pytest.raises(PanelSamplesError, match="CAPTURE_QUALITY_FAILED") as excinfo:
        _enforce_capture_quality(quality_report, capture_manifest)
    assert "4 missing days" in str(excinfo.value)


def test_enforce_capture_quality_rejects_more_proven_absent_than_missing(tmp_path: Path) -> None:
    """Defensive guard: the capture-manifest and quality-report describe the same
    capture, so more proven-absent days than missing days for an instrument is
    an inconsistency to fail closed on, not a case to silently clamp at zero."""
    quality_report = tmp_path / "quality-report.json"
    _write_json(
        quality_report,
        {"instruments": [{"instrument_id": "BTCUSDT", "missing_days": 1, "row_count": 1}]},
    )
    capture_manifest = _daily_fill_sources(
        "BTCUSDT", ("2020-01-11", "2020-01-12"), status="absent"
    )
    with pytest.raises(PanelSamplesError, match="inconsistent"):
        _enforce_capture_quality(quality_report, capture_manifest)
