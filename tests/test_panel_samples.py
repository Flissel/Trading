import io
import json
import zipfile
from collections.abc import Callable
from datetime import date, timedelta
from decimal import Decimal
from itertools import pairwise
from pathlib import Path

import pytest

import trading_bot.panel_samples as panel_samples_module
from tests.carry_fixtures import (
    build_captures,
    perp_fetch,
    small_carry_config,
    spot_fetch_with_a_hole,
)
from tests.test_panel_fold_run import MONTHS, SYMBOLS
from trading_bot.canonical import content_sha256
from trading_bot.panel_capture import PanelPayload, PanelSourceAbsent, capture_panel
from trading_bot.panel_config import PanelFoldGeometry, load_family_spec, load_panel_family_spec
from trading_bot.panel_reader import PanelBar
from trading_bot.panel_samples import (
    _MAX_MISSING_DAYS_PER_INSTRUMENT,
    PanelSamplesError,
    _enforce_capture_quality,
    build_rebalance_samples,
    derive_panel_config,
    publish_panel_walk_forward,
    rebalance_close_times,
    verify_panel_manifest,
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
    prevented, so the quality gate must not stop the family over them, unlike
    before this fix. (The real-world instance is ICPUSDT: a dormant contract
    -- flat at 6.44 USDT with `quote_volume` exactly zero for 103 of the 104
    days before the gap -- whose final five days, 2022-09-22 through
    2022-09-26, are absent from both Binance dumps; it is not delisted and
    resumes trading on 2022-09-27. It is outside the eligible universe at
    every decision whose volatility window straddles the gap regardless of
    this gate, by the liquidity floor or the rank cutoff, see
    `panel_universe.py`.)"""
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
        # Attempted and proven absent at source too -- the shape of ICPUSDT's
        # final five days (2022-09-22 through 2022-09-26). Not modelled here:
        # ICPUSDT was dormant (flat price, quote_volume exactly zero) for the
        # 103 days before that point, which is what actually excludes it from
        # the universe there -- it is not delisted and resumes on 2022-09-27.
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


def test_manifest_rejects_a_capture_claiming_days_that_were_never_missing(
    tmp_path: Path,
) -> None:
    """Reproduces the exact attack that broke a count-only subtraction: rewrite
    a real capture's proven-absent dates to ones that were never actually
    missing, then recompute `capture_root_hash` over the tampered content so
    the file still verifies internally -- `content_sha256` proves only that a
    file agrees with its own declared hash, not that its claims agree with
    the dataset. Before the per-day cross-check, this tampered manifest
    published, naming fabricated days as BTCUSDT's limitation; it must now
    fail closed instead."""
    capture_panel(
        workspace_root=tmp_path,
        output_directory=tmp_path / "capture",
        reserve_bytes=0,
        symbols=("BTCUSDT",),
        months=_FLOOR_TEST_MONTHS,
        fetch=_floor_test_fetch_with_forgiven_gap,
    )
    manifest_path = tmp_path / "capture" / "capture-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fill_entries = [
        entry
        for entry in manifest["sources"]
        if entry.get("kind") == "klines_daily_fill" and entry.get("status") == "absent"
    ]
    assert fill_entries  # sanity: the fixture actually produced absent entries
    for index, entry in enumerate(fill_entries):
        entry["month"] = f"1999-01-{index + 1:02d}"  # never missing in this dataset
    material = {key: value for key, value in manifest.items() if key != "capture_root_hash"}
    manifest["capture_root_hash"] = content_sha256(material)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

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

    with pytest.raises(PanelSamplesError, match="inconsistent with the published dataset"):
        publish_panel_walk_forward(
            tmp_path / "capture",
            output_path=tmp_path / "manifest.json",
            spec=spec,
            family_spec_hash=spec_hash,
        )
    assert not (tmp_path / "manifest.json").exists()


def _write_json(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def _bars_for_dataset(
    symbol: str, *, first: str, last: str, missing: frozenset[str]
) -> tuple[PanelBar, ...]:
    """One `PanelBar` per calendar day for `symbol` from `first` through `last`
    (inclusive, ISO dates), skipping every date in `missing` -- exactly the
    shape `panel_samples._missing_day_strings` needs to recompute the
    instrument's own real interior missing days, so a unit test can construct
    a capture-manifest claim and check it against a dataset that either
    supports or contradicts it."""
    epoch = date(1970, 1, 1)
    start = date.fromisoformat(first)
    end = date.fromisoformat(last)
    bars: list[PanelBar] = []
    current = start
    while current <= end:
        if current.isoformat() not in missing:
            open_time_ns = (current - epoch).days * DAY_NS
            close_time_ns = open_time_ns + DAY_NS - 1_000_000
            bars.append(
                PanelBar(
                    contract_id=f"{symbol}:0",
                    instrument_id=symbol,
                    open_time_ns=open_time_ns,
                    close_time_ns=close_time_ns,
                    available_time_ns=close_time_ns + 1,
                    close=Decimal("100"),
                    quote_volume=Decimal("1000"),
                )
            )
        current += timedelta(days=1)
    return tuple(bars)


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


_ICPUSDT_GAP = (
    "2022-09-22",
    "2022-09-23",
    "2022-09-24",
    "2022-09-25",
    "2022-09-26",
)
# ICPUSDT trades again shortly after the gap (2022-09-27 through 2022-10-05
# here): the gap must be genuinely interior -- bars on both sides -- or
# `_missing_day_strings` would read it as a trailing censor (a delisting) and
# count it as zero missing days, the same way `find_missing_days` documents.
_ICPUSDT_BARS = _bars_for_dataset(
    "ICPUSDT", first="2022-09-01", last="2022-10-05", missing=frozenset(_ICPUSDT_GAP)
)

_BTCUSDT_FIVE_DAY_GAP = frozenset(
    {"2020-01-11", "2020-01-12", "2020-01-13", "2020-01-14", "2020-01-15"}
)
_BTCUSDT_BARS = _bars_for_dataset(
    "BTCUSDT", first="2020-01-01", last="2020-01-31", missing=_BTCUSDT_FIVE_DAY_GAP
)


def test_enforce_capture_quality_forgives_days_proven_absent_at_source(tmp_path: Path) -> None:
    quality_report = tmp_path / "quality-report.json"
    _write_json(
        quality_report,
        {"instruments": [{"instrument_id": "ICPUSDT", "missing_days": 5, "row_count": 1}]},
    )
    capture_manifest = _daily_fill_sources("ICPUSDT", _ICPUSDT_GAP, status="absent")
    absent = _enforce_capture_quality(quality_report, capture_manifest, _ICPUSDT_BARS)
    assert absent == {"ICPUSDT": _ICPUSDT_GAP}


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
        _enforce_capture_quality(quality_report, capture_manifest, _BTCUSDT_BARS)
    assert "BTCUSDT" in str(excinfo.value)
    assert "5 unexplained missing days" in str(excinfo.value)


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
        "BTCUSDT", tuple(sorted(_BTCUSDT_FIVE_DAY_GAP)), status="present"
    )
    with pytest.raises(PanelSamplesError, match="CAPTURE_QUALITY_FAILED") as excinfo:
        _enforce_capture_quality(quality_report, capture_manifest, _BTCUSDT_BARS)
    assert "5 unexplained missing days" in str(excinfo.value)


def test_enforce_capture_quality_ignores_an_attempt_recorded_absent_after_discovery(
    tmp_path: Path,
) -> None:
    """`absent_after_discovery` means the bucket's own listing contradicted
    itself inside one run -- the opposite of proof the day is genuinely
    absent -- so it must not count, even though it is a status other than
    `present`. Only the literal `absent` status this codebase's daily-fill
    path ever writes counts as proof."""
    quality_report = tmp_path / "quality-report.json"
    _write_json(
        quality_report,
        {"instruments": [{"instrument_id": "BTCUSDT", "missing_days": 5, "row_count": 1}]},
    )
    capture_manifest = _daily_fill_sources(
        "BTCUSDT", tuple(sorted(_BTCUSDT_FIVE_DAY_GAP)), status="absent_after_discovery"
    )
    with pytest.raises(PanelSamplesError, match="CAPTURE_QUALITY_FAILED") as excinfo:
        _enforce_capture_quality(quality_report, capture_manifest, _BTCUSDT_BARS)
    assert "5 unexplained missing days" in str(excinfo.value)


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
        _enforce_capture_quality(quality_report, capture_manifest, _BTCUSDT_BARS)
    assert "4 unexplained missing days" in str(excinfo.value)


def test_enforce_capture_quality_rejects_more_proven_absent_than_missing(tmp_path: Path) -> None:
    """Defensive guard: the capture-manifest and quality-report describe the same
    capture, so more proven-absent days than missing days for an instrument is
    an inconsistency to fail closed on, not a case to silently clamp at zero.
    Both claimed days are genuinely missing in `bars` (2020-01-11 and
    2020-01-12 are inside `_BTCUSDT_FIVE_DAY_GAP`) -- the inconsistency here
    is deliberately only the report's stale `missing_days` count, isolating
    this check from the day-existence check below."""
    quality_report = tmp_path / "quality-report.json"
    _write_json(
        quality_report,
        {"instruments": [{"instrument_id": "BTCUSDT", "missing_days": 1, "row_count": 1}]},
    )
    capture_manifest = _daily_fill_sources(
        "BTCUSDT", ("2020-01-11", "2020-01-12"), status="absent"
    )
    with pytest.raises(PanelSamplesError, match="inconsistent"):
        _enforce_capture_quality(quality_report, capture_manifest, _BTCUSDT_BARS)


def test_enforce_capture_quality_rejects_a_day_that_was_never_actually_missing(
    tmp_path: Path,
) -> None:
    """The subtraction in `_enforce_capture_quality` is by count; this is the
    check that keeps it honest by day. A capture-manifest entry can name any
    date at all and still re-hash cleanly (the hash only proves internal
    self-consistency, not agreement with the dataset) -- so a claimed absence
    that is not actually one of the instrument's own interior missing days
    (recomputed here from `bars`, which do not have a gap at all) must fail
    closed rather than be trusted by count alone."""
    quality_report = tmp_path / "quality-report.json"
    _write_json(
        quality_report,
        {"instruments": [{"instrument_id": "BTCUSDT", "missing_days": 5, "row_count": 1}]},
    )
    capture_manifest = _daily_fill_sources("BTCUSDT", ("1999-01-01",), status="absent")
    gapless_bars = _bars_for_dataset(
        "BTCUSDT", first="2020-01-01", last="2020-01-31", missing=frozenset()
    )
    with pytest.raises(PanelSamplesError, match="inconsistent with the published dataset"):
        _enforce_capture_quality(quality_report, capture_manifest, gapless_bars)


def test_manifest_binds_a_hedge_capture(tmp_path: Path) -> None:
    perp, spot = build_captures(tmp_path)
    spec, spec_hash = load_family_spec(small_carry_config(tmp_path))
    artifact = publish_panel_walk_forward(
        perp, output_path=tmp_path / "m.json", spec=spec,
        family_spec_hash=spec_hash, hedge_capture_root=spot,
    )
    manifest = json.loads(artifact.output_path.read_text(encoding="utf-8"))
    spot_manifest = json.loads((spot / "capture-manifest.json").read_text(encoding="utf-8"))
    spot_dataset = json.loads(
        (spot / "dataset" / "dataset-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["hedge_capture_root_hash"] == spot_manifest["capture_root_hash"]
    assert manifest["hedge_dataset_root_hash"] == spot_dataset["root_hash"]
    assert manifest["hedge_absent_at_source_days"] == {}
    assert manifest["family_name"] == "funding_carry_panel_v1"
    assert manifest["hedge_capture_root_hash"] != manifest["capture_root_hash"]
    assert verify_panel_manifest(artifact.output_path)


def test_manifest_without_a_hedge_has_no_hedge_keys(tmp_path: Path) -> None:
    perp, _ = build_captures(tmp_path)
    spec, spec_hash = load_family_spec(small_carry_config(tmp_path))
    artifact = publish_panel_walk_forward(
        perp, output_path=tmp_path / "m.json", spec=spec, family_spec_hash=spec_hash,
    )
    manifest = json.loads(artifact.output_path.read_text(encoding="utf-8"))
    assert not any(key.startswith("hedge_") for key in manifest)


def test_hedge_capture_names_its_absent_at_source_days(tmp_path: Path) -> None:
    perp, spot = build_captures(tmp_path, spot_fetch_function=spot_fetch_with_a_hole)
    spec, spec_hash = load_family_spec(small_carry_config(tmp_path))
    artifact = publish_panel_walk_forward(
        perp, output_path=tmp_path / "m.json", spec=spec,
        family_spec_hash=spec_hash, hedge_capture_root=spot,
    )
    manifest = json.loads(artifact.output_path.read_text(encoding="utf-8"))
    assert set(manifest["hedge_absent_at_source_days"]) == {"C00USDT"}
    assert len(manifest["hedge_absent_at_source_days"]["C00USDT"]) == 3
    assert manifest["absent_at_source_days"] == {}


def test_hedge_capture_must_not_be_the_primary_capture(tmp_path: Path) -> None:
    """One capture passed twice would bind a manifest to a perp-versus-perp
    book, which is not the position the family declares."""
    perp, _ = build_captures(tmp_path)
    spec, spec_hash = load_family_spec(small_carry_config(tmp_path))
    with pytest.raises(PanelSamplesError, match="differ"):
        publish_panel_walk_forward(
            perp, output_path=tmp_path / "m.json", spec=spec,
            family_spec_hash=spec_hash, hedge_capture_root=perp,
        )
    assert not (tmp_path / "m.json").exists()


def test_hedge_capture_must_be_a_spot_capture(tmp_path: Path) -> None:
    """Two distinct captures are not enough: the hedge leg is the spot leg, so
    a second perpetual capture must be rejected on its declared market."""
    perp, _ = build_captures(tmp_path)
    second_perp = capture_panel(
        workspace_root=tmp_path, output_directory=tmp_path / "perp2", reserve_bytes=0,
        symbols=SYMBOLS, months=MONTHS, fetch=perp_fetch,
    ).capture_root
    spec, spec_hash = load_family_spec(small_carry_config(tmp_path))
    with pytest.raises(PanelSamplesError, match="spot"):
        publish_panel_walk_forward(
            perp, output_path=tmp_path / "m.json", spec=spec,
            family_spec_hash=spec_hash, hedge_capture_root=second_perp,
        )
    assert not (tmp_path / "m.json").exists()


def test_hedge_capture_goes_through_the_same_quality_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    perp, spot = build_captures(tmp_path)
    spec, spec_hash = load_family_spec(small_carry_config(tmp_path))
    real_gate = panel_samples_module._enforce_capture_quality
    seen: list[Path] = []

    def gate(path: Path, manifest: dict[str, object], bars: object) -> dict[str, tuple[str, ...]]:
        seen.append(path)
        if path.is_relative_to(spot):
            raise PanelSamplesError("CAPTURE_QUALITY_FAILED: hedge capture")
        return real_gate(path, manifest, bars)  # type: ignore[arg-type]

    monkeypatch.setattr(panel_samples_module, "_enforce_capture_quality", gate)
    with pytest.raises(PanelSamplesError, match="CAPTURE_QUALITY_FAILED"):
        publish_panel_walk_forward(
            perp, output_path=tmp_path / "m.json", spec=spec,
            family_spec_hash=spec_hash, hedge_capture_root=spot,
        )
    assert seen == [
        perp / "dataset" / "quality-report.json",
        spot / "dataset" / "quality-report.json",
    ]
    assert not (tmp_path / "m.json").exists()
