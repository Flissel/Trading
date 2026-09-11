"""Shared fixtures for the funding carry tests.

Two small captures over the twelve `SYMBOLS` and seven `MONTHS` of the
P1.27 fold-run fixture: a perpetual capture with weekly funding rows whose
rate rises with the symbol's seed (C11USDT pays negative funding), and a spot
capture whose closes sit one unit below the perpetual so the basis is
non-zero. `capture_panel` fills interior gaps from the daily dumps, so every
fake fetch answers a `/daily/klines/` request with `PanelSourceAbsent`.
"""

import json
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

from tests.test_panel_fold_run import (
    DAY_MS,
    EPOCH_DAY_2020,
    MONTH_DAYS,
    MONTH_START_DAY,
    MONTHS,
    SYMBOLS,
    kline_csv,
    zip_bytes,
)
from trading_bot.panel_capture import PanelPayload, PanelSourceAbsent, capture_panel

NEGATIVE_FUNDING_SYMBOL = "C11USDT"
# Under `small_carry_config`, fold 0's test window holds the Sundays at day
# offsets 109, 116 and 123 (the P1.27 fixture confirmed this empirically).
# C10USDT pays the highest funding, so every member holds it; its perpetual
# lacks the three days 121..123, so the position entered on 116 has no exit
# bar on 123 and is force-closed, and the pair is ineligible on 123.
HOLE_SYMBOL = "C10USDT"
HOLE_DAY_OFFSETS = frozenset({121, 122, 123})


def funding_rate(symbol: str) -> str:
    seed = int(symbol[1:3])
    if symbol == NEGATIVE_FUNDING_SYMBOL:
        return "-0.0001"
    return str(Decimal("0.0001") * (seed + 1))


def funding_csv(symbol: str, month: str) -> str:
    text = "calc_time,funding_interval_hours,last_funding_rate\n"
    for offset in range(0, MONTH_DAYS[month], 7):
        day = EPOCH_DAY_2020 + MONTH_START_DAY[month] + offset
        text += f"{day * DAY_MS},8,{funding_rate(symbol)}\n"
    return text


def spot_kline_csv(symbol: str, month: str) -> str:
    """The perpetual's bars with every price one unit lower."""
    lines = kline_csv(symbol, month).splitlines()
    out = [lines[0]]
    for line in lines[1:]:
        fields = line.split(",")
        close = str(Decimal(fields[4]) - 1)
        fields[1] = fields[2] = fields[3] = fields[4] = close
        out.append(",".join(fields))
    return "\n".join(out) + "\n"


def _symbol_and_month(url: str) -> tuple[str, str]:
    symbol = next(item for item in SYMBOLS if f"/{item}/" in url or f"/{item}-" in url)
    month = next(item for item in MONTHS if item in url)
    return symbol, month


def _payload(url: str, name: str, text: str) -> PanelPayload:
    return PanelPayload(url=url, raw_bytes=zip_bytes(name, text), received_time_ns=1)


def perp_fetch(url: str) -> PanelPayload:
    if "/daily/klines/" in url:
        raise PanelSourceAbsent("404: no daily dump")
    symbol, month = _symbol_and_month(url)
    if "fundingRate" in url:
        return _payload(url, "f.csv", funding_csv(symbol, month))
    return _payload(url, "k.csv", kline_csv(symbol, month))


def perp_fetch_with_a_hole(url: str) -> PanelPayload:
    if "/daily/klines/" in url or "fundingRate" in url:
        return perp_fetch(url)
    symbol, month = _symbol_and_month(url)
    if symbol != HOLE_SYMBOL:
        return perp_fetch(url)
    lines = kline_csv(symbol, month).splitlines()
    kept = [lines[0]] + [
        line
        for line in lines[1:]
        if int(line.split(",")[0]) // DAY_MS - EPOCH_DAY_2020 not in HOLE_DAY_OFFSETS
    ]
    return _payload(url, "k.csv", "\n".join(kept) + "\n")


def spot_fetch(url: str) -> PanelPayload:
    assert "fundingRate" not in url, "the spot market has no funding"
    assert "/data/spot/" in url
    if "/daily/klines/" in url:
        raise PanelSourceAbsent("404: no daily dump")
    symbol, month = _symbol_and_month(url)
    return _payload(url, "k.csv", spot_kline_csv(symbol, month))


def spot_fetch_with_a_hole(url: str) -> PanelPayload:
    """C00USDT's spot bars lack day offsets 70..72, proven absent at source."""
    if "/daily/klines/" in url:
        raise PanelSourceAbsent("404: no daily dump")
    symbol, month = _symbol_and_month(url)
    if symbol != "C00USDT" or month != "2020-03":
        return spot_fetch(url)
    lines = spot_kline_csv(symbol, month).splitlines()
    kept = [lines[0]] + [
        line
        for line in lines[1:]
        if int(line.split(",")[0]) // DAY_MS - EPOCH_DAY_2020 not in {70, 71, 72}
    ]
    return _payload(url, "k.csv", "\n".join(kept) + "\n")


def build_captures(
    root: Path,
    *,
    perp_fetch_function: Callable[[str], PanelPayload] = perp_fetch,
    spot_fetch_function: Callable[[str], PanelPayload] = spot_fetch,
) -> tuple[Path, Path]:
    """Capture both markets under `root`; returns (perp_root, spot_root)."""
    perp = capture_panel(
        workspace_root=root, output_directory=root / "perp", reserve_bytes=0,
        symbols=SYMBOLS, months=MONTHS, fetch=perp_fetch_function,
    )
    spot = capture_panel(
        workspace_root=root, output_directory=root / "spot", reserve_bytes=0,
        symbols=SYMBOLS, months=MONTHS, fetch=spot_fetch_function, market="spot",
    )
    return perp.capture_root, spot.capture_root


def small_carry_config(tmp_path: Path) -> Path:
    """The frozen carry declaration with the P1.27 test fold geometry and a
    twelve-symbol pair list, so a seven-month fixture yields real folds."""
    document = json.loads(Path("configs/funding-carry-panel-v1.json").read_text(encoding="utf-8"))
    document["pairs"] = [{"perpetual": s, "spot": s, "multiplier": 1} for s in SYMBOLS]
    document["excluded_pairs"] = []
    document["universe"].update(
        {
            "minimum_history_days": 20,
            "liquidity_window_days": 5,
            # 12, not 10: every fixture symbol shares one quote volume, so a
            # cap below the symbol count would drop the top-funding symbols by
            # the lexicographic tie-break at every decision
            "maximum_pairs": 12,
            "minimum_pairs": 6,
            "tier_one_rank_limit": 4,
        }
    )
    document["selection"]["minimum_selected"] = 2
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
    path = tmp_path / "small-carry.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path
