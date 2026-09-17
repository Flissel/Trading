"""Tests for `verify_capture_superset` against real fixture captures.

`MONTHS` and `SYMBOLS` come from `tests.test_panel_fold_run`; the fake fetches
and `build_captures` come from `tests.carry_fixtures`. Every capture here is
built with `capture_panel` (either directly, for a narrowed month tuple, or
through `build_captures`, for the paired perp/spot captures) -- no network.
"""

from pathlib import Path

import pytest

from tests.carry_fixtures import build_captures, perp_fetch, perp_fetch_with_a_liquidity_dip
from tests.test_panel_fold_run import MONTHS, SYMBOLS
from trading_bot.capture_lineage import verify_capture_superset
from trading_bot.panel_capture import capture_panel

# `MONTH_DAYS`/`MONTH_START_DAY` in `tests.test_panel_fold_run` only cover the
# seven `MONTHS` entries, so a narrowed six-month tuple (rather than an
# out-of-range eighth month) is what actually captures cleanly against the
# fake fetches; it exercises the exact same "more months captured" lineage.
_FEWER_MONTHS = MONTHS[:-1]


@pytest.fixture(autouse=True)
def _fixed_wall_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Freeze wall-clock time for every test in this module.

    `zipfile.ZipFile.writestr` stamps each archive member with
    `time.localtime(time.time())`, so two `capture_panel` runs over the same
    (symbol, month) produce byte-identical zips -- and therefore the same
    `raw_sha256` -- only if they do not straddle that clock's tick. Freezing
    it removes that race so a comparison across two separately built
    captures is judging the fixture's content, not scheduling luck.
    """
    monkeypatch.setattr("time.time", lambda: 1_700_000_000.0)


def _perp_capture(root: Path, *, months: tuple[str, ...]) -> Path:
    return capture_panel(
        workspace_root=root,
        output_directory=root / "-".join(months),
        reserve_bytes=0,
        symbols=SYMBOLS,
        months=months,
        fetch=perp_fetch,
    ).capture_root


def test_more_months_is_a_superset_of_fewer_months(tmp_path: Path) -> None:
    fewer = _perp_capture(tmp_path, months=_FEWER_MONTHS)
    more = _perp_capture(tmp_path, months=MONTHS)
    assert verify_capture_superset(fewer, more) == (True, ())


def test_fewer_months_is_not_a_superset_of_more_months(tmp_path: Path) -> None:
    fewer = _perp_capture(tmp_path, months=_FEWER_MONTHS)
    more = _perp_capture(tmp_path, months=MONTHS)
    verified, reasons = verify_capture_superset(more, fewer)
    assert verified is False
    # The dropped month (2020-07) drops 12 symbols x 2 source kinds (klines,
    # fundingRate) = 24 missing rows -- one more than the 20-reason cap.
    assert len(reasons) == 20
    expected = tuple(
        sorted(
            f"{kind}:{symbol}:2020-07: missing"
            for symbol in SYMBOLS
            for kind in ("klines", "fundingRate")
        )
    )[:20]
    assert reasons == expected


def test_a_liquidity_dip_changes_only_the_dipped_months_klines_hash(tmp_path: Path) -> None:
    # `capture_panel` requires its workspace root to already exist (it checks
    # free disk space there), so each side's root is made explicitly rather
    # than reusing `tmp_path` itself for both.
    original_workspace = tmp_path / "original"
    original_workspace.mkdir()
    dipped_workspace = tmp_path / "dipped"
    dipped_workspace.mkdir()
    original_root, _ = build_captures(original_workspace, perp_fetch_function=perp_fetch)
    dipped_root, _ = build_captures(
        dipped_workspace, perp_fetch_function=perp_fetch_with_a_liquidity_dip
    )
    verified, reasons = verify_capture_superset(original_root, dipped_root)
    assert verified is False
    expected = tuple(sorted(f"klines:{symbol}:2020-04: hash" for symbol in SYMBOLS))
    assert reasons == expected


def test_a_spot_capture_is_not_a_superset_of_a_perp_capture(tmp_path: Path) -> None:
    perp_root, spot_root = build_captures(tmp_path)
    verified, reasons = verify_capture_superset(perp_root, spot_root)
    assert verified is False
    # The perp venue is BINANCE_UM and the spot venue is BINANCE_SPOT, so the
    # venue header disagrees too; both header reasons come before any row
    # reason, in the fixed venue/interval/market order.
    assert reasons[0] == "venue: mismatch"
    assert "market: mismatch" in reasons[:2]


def test_a_missing_manifest_is_a_single_reason(tmp_path: Path) -> None:
    original_root, _ = build_captures(tmp_path)
    missing_root = tmp_path / "does-not-exist"
    assert verify_capture_superset(original_root, missing_root) == (
        False,
        ("extended: manifest missing or unreadable",),
    )
    assert verify_capture_superset(missing_root, original_root) == (
        False,
        ("original: manifest missing or unreadable",),
    )
