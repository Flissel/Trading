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
from typing import cast

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
# The liquidity window ending on the middle decision (day 116) is the five days
# 112..116 (`liquidity_window_days` is 5 under `small_carry_config`). Dropping
# every symbol's quote volume there below the base config's unmodified
# `minimum_median_quote_volume` (5,000,000) floor pushes that one decision's
# universe empty while the weeks before and after are untouched.
LIQUIDITY_DIP_DAY_OFFSETS = frozenset({112, 113, 114, 115, 116})
# The weekly funding rows inside `carry_l1w_h4w`'s one-week lookback windows
# for the three warm-up Sundays 88, 95 and 102 and the first two decisions 109
# and 116 -- (81, 88] holds only day 88, (88, 95] only day 91, (95, 102] only
# day 98, (102, 109] only day 105, (109, 116] only day 112 -- forced negative
# for every symbol, so the one-week member sees no positive-funding pair at
# any of them and enters the fold with no book at all. Decision 123's window
# (116, 123] still holds the positive row at day 119, and the 4-week members
# still see March's positive rows at 67, 74 and 81.
# Funding rows restart at each month's first day, so March's are 60, 67, 74,
# 81, 88 -- day 88, not 84, is the row inside the Sunday-88 window.
NEGATIVE_FUNDING_DAY_OFFSETS = frozenset({88, 91, 98, 105, 112})


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


def funding_csv_with_negative_weeks(symbol: str, month: str) -> str:
    """Every symbol's funding, with the rows at `NEGATIVE_FUNDING_DAY_OFFSETS` forced negative."""
    text = "calc_time,funding_interval_hours,last_funding_rate\n"
    for offset in range(0, MONTH_DAYS[month], 7):
        day = EPOCH_DAY_2020 + MONTH_START_DAY[month] + offset
        day_offset = day - EPOCH_DAY_2020
        rate = "-0.0001" if day_offset in NEGATIVE_FUNDING_DAY_OFFSETS else funding_rate(symbol)
        text += f"{day * DAY_MS},8,{rate}\n"
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


def kline_csv_with_liquidity_dip(symbol: str, month: str) -> str:
    """The perpetual's bars with `LIQUIDITY_DIP_DAY_OFFSETS`'s quote volume dropped."""
    lines = kline_csv(symbol, month).splitlines()
    out = [lines[0]]
    for line in lines[1:]:
        fields = line.split(",")
        day_offset = int(fields[0]) // DAY_MS - EPOCH_DAY_2020
        if day_offset in LIQUIDITY_DIP_DAY_OFFSETS:
            fields[7] = "1000000"
        out.append(",".join(fields))
    return "\n".join(out) + "\n"


def perp_fetch_with_a_liquidity_dip(url: str) -> PanelPayload:
    if "/daily/klines/" in url:
        raise PanelSourceAbsent("404: no daily dump")
    symbol, month = _symbol_and_month(url)
    if "fundingRate" in url:
        return _payload(url, "f.csv", funding_csv(symbol, month))
    return _payload(url, "k.csv", kline_csv_with_liquidity_dip(symbol, month))


def perp_fetch_with_negative_funding_weeks(url: str) -> PanelPayload:
    if "/daily/klines/" in url:
        raise PanelSourceAbsent("404: no daily dump")
    symbol, month = _symbol_and_month(url)
    if "fundingRate" in url:
        return _payload(url, "f.csv", funding_csv_with_negative_weeks(symbol, month))
    return _payload(url, "k.csv", kline_csv(symbol, month))


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


def _reduced(declaration: Path) -> dict[str, object]:
    """A shipped carry declaration cut down to the P1.27 test fold geometry and
    a twelve-symbol pair list, so a seven-month fixture yields real folds.

    Everything the two families share is reduced identically, so a v1 and a v2
    run over the same capture differ only by what their declarations declare.
    """
    document = json.loads(declaration.read_text(encoding="utf-8"))
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
    reduced: dict[str, object] = document
    return reduced


def small_carry_config(tmp_path: Path) -> Path:
    """The frozen v1 carry declaration under the reduced fixture geometry."""
    path = tmp_path / "small-carry.json"
    document = _reduced(Path("configs/funding-carry-panel-v1.json"))
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def small_carry_v2_config(tmp_path: Path) -> Path:
    """The frozen v2 carry declaration -- hold 26, the exit rule and the cost
    hurdle -- under exactly the reductions `small_carry_config` applies."""
    path = tmp_path / "small-carry-v2.json"
    document = _reduced(Path("configs/funding-carry-panel-v2.json"))
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def small_carry_v4_config(tmp_path: Path) -> Path:
    """The frozen v4 carry declaration -- four slots on a 10,000 USDT book,
    holds shrunk to 1 and 2 weeks so fold 0's three Sundays exercise age-out --
    under exactly the reductions `small_carry_config` applies."""
    path = tmp_path / "small-carry-v4.json"
    document = _reduced(Path("configs/funding-carry-panel-v4.json"))
    document["capital"] = {
        "book_usdt": "10000",
        "pair_slots": 4,
        "per_leg_notional_usdt": "1250",
        "fee_tier": "standard_taker_no_bnb",
    }
    hold_weeks = (1, 1, 2, 2)
    members = cast("list[dict[str, object]]", document["members"])
    document["members"] = [
        {**member, "hold_weeks": weeks}
        for member, weeks in zip(members, hold_weeks, strict=True)
    ]
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def small_carry_v4_one_slot_config(tmp_path: Path) -> Path:
    """The v4 declaration pinned to a single pair slot, so `pair_slots` binds
    the fill even though the fixture's ranking is two pairs wide -- otherwise
    identical to `small_carry_v4_config`."""
    path = tmp_path / "small-carry-v4-one-slot.json"
    document = _reduced(Path("configs/funding-carry-panel-v4.json"))
    document["capital"] = {
        "book_usdt": "10000",
        "pair_slots": 1,
        "per_leg_notional_usdt": "5000",
        "fee_tier": "standard_taker_no_bnb",
    }
    hold_weeks = (1, 1, 2, 2)
    members = cast("list[dict[str, object]]", document["members"])
    document["members"] = [
        {**member, "hold_weeks": weeks}
        for member, weeks in zip(members, hold_weeks, strict=True)
    ]
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


# C10USDT's perpetual is dark from day offset 103 through 109: it enters the
# warm-up cohorts formed on Sundays 88, 95 and 102 and has no bar at fold 0's
# first decision (day 109), the case a forced close cannot catch because no
# episode runs during the warm-up. The seven days are absent from the daily
# dumps too, so the quality gate treats them as proven absent at source.
WARM_UP_HOLE_DAY_OFFSETS = frozenset(range(103, 110))


def perp_fetch_with_a_warm_up_hole(url: str) -> PanelPayload:
    if "/daily/klines/" in url or "fundingRate" in url:
        return perp_fetch(url)
    symbol, month = _symbol_and_month(url)
    if symbol != HOLE_SYMBOL:
        return perp_fetch(url)
    lines = kline_csv(symbol, month).splitlines()
    kept = [lines[0]] + [
        line
        for line in lines[1:]
        if int(line.split(",")[0]) // DAY_MS - EPOCH_DAY_2020 not in WARM_UP_HOLE_DAY_OFFSETS
    ]
    return _payload(url, "k.csv", "\n".join(kept) + "\n")


# The week `(109, 116]` -- the one that ends at fold 0's second decision --
# holds exactly one weekly settlement, April's row at day offset 112. Forcing
# only `EXIT_WEEK_SYMBOL`'s copy of that one row negative leaves its four-week
# trailing funding positive (three normal rows against one negative one), so
# the pair stays rankable and stays held, while the exit rule's one-week window
# sees it pay nothing. That is the case a slot book's exit step exists for: one
# held slot empties and refills from the ranking while its siblings' slots do
# not move. C10USDT is the fixture's top payer, so it is held by every member
# that fills a slot at all.
EXIT_WEEK_SYMBOL = "C10USDT"
EXIT_WEEK_DAY_OFFSET = 112


def funding_csv_with_one_negative_week(symbol: str, month: str) -> str:
    """Every symbol's funding, with `EXIT_WEEK_SYMBOL`'s one row at
    `EXIT_WEEK_DAY_OFFSET` forced negative and nothing else touched."""
    text = "calc_time,funding_interval_hours,last_funding_rate\n"
    for offset in range(0, MONTH_DAYS[month], 7):
        day = EPOCH_DAY_2020 + MONTH_START_DAY[month] + offset
        negative = (
            symbol == EXIT_WEEK_SYMBOL and day - EPOCH_DAY_2020 == EXIT_WEEK_DAY_OFFSET
        )
        text += f"{day * DAY_MS},8,{'-0.0001' if negative else funding_rate(symbol)}\n"
    return text


def perp_fetch_with_one_negative_week(url: str) -> PanelPayload:
    if "/daily/klines/" in url:
        raise PanelSourceAbsent("404: no daily dump")
    symbol, month = _symbol_and_month(url)
    if "fundingRate" in url:
        return _payload(url, "f.csv", funding_csv_with_one_negative_week(symbol, month))
    return _payload(url, "k.csv", kline_csv(symbol, month))
