"""Fixtures for the weekly shadow capture: daily dumps and REST funding.

Everything here is derived from the monthly generators the rest of the panel
fixtures use (`kline_csv`, `spot_kline_csv`, `funding_csv`), so a daily dump
carries bar for bar what the monthly dump for the same date carries and a REST
funding answer carries exactly the settlements the monthly funding dump
carries. That is the whole point of the shadow capture's reconciliation
(spec section 3.2): the two sources must agree, so the fixture must not be
able to disagree by accident.

`MONTHS` stops at 2020-07; the tail needs the two months after it. They are
registered additively with `setdefault`, exactly as `tests.test_carry_holdout_run`
registers its eighth month -- no existing key moves and `MONTHS` itself is
untouched, so every capture built from `MONTHS` is built by exactly the
fetches it was built by before.
"""

import json
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

from tests.carry_fixtures import _payload, funding_csv, funding_rate, spot_kline_csv
from tests.test_panel_fold_run import (
    DAY_MS,
    EPOCH_DAY_2020,
    MONTH_DAYS,
    MONTH_START_DAY,
    MONTHS,
    SYMBOLS,
    kline_csv,
)
from trading_bot.panel_capture import PanelPayload, PanelSourceAbsent, capture_panel
from trading_bot.shadow_capture import build_shadow_capture

MONTH_START_DAY.setdefault("2020-08", MONTH_START_DAY["2020-07"] + MONTH_DAYS["2020-07"])
MONTH_DAYS.setdefault("2020-08", 31)
MONTH_START_DAY.setdefault("2020-09", MONTH_START_DAY["2020-08"] + MONTH_DAYS["2020-08"])
MONTH_DAYS.setdefault("2020-09", 30)

TAIL_MONTHS = ("2020-08", "2020-09")
ALL_MONTHS = (*MONTHS, *TAIL_MONTHS)

# The fixture calendar runs on the same lattice the panel does: day offset 0 is
# 2020-01-01, a Wednesday, and `panel_samples` calls a day a Sunday when its
# epoch day number leaves remainder 3 modulo 7. 2020-08-01 is a Saturday and
# 2020-08-02 the first Sunday after the base capture's last month.
HOUR_MS = 3_600_000
FIRST_TAIL_DATE = "2020-08-01"
FIRST_TAIL_SUNDAY = "2020-08-02"
SECOND_TAIL_SUNDAY = "2020-08-09"
# The first Sunday after 2020-08, for a base capture that has advanced a month.
THIRD_TAIL_SUNDAY = "2020-09-06"


def epoch_day(date: str) -> int:
    """Days since the Unix epoch for an ISO date, read off the fixture's own
    month tables rather than the real calendar, so a fixture bar and a fixture
    date can never drift apart."""
    return EPOCH_DAY_2020 + MONTH_START_DAY[date[:7]] + int(date[8:10]) - 1


def day_start_ms(date: str) -> int:
    return epoch_day(date) * DAY_MS


def day_end_ms(date: str) -> int:
    return (epoch_day(date) + 1) * DAY_MS - 1


def _single_day(text: str, date: str) -> str:
    """The one CSV row of `text` that opens on `date`, with `text`'s header."""
    open_ms = day_start_ms(date)
    lines = text.splitlines()
    body = [line for line in lines[1:] if line.split(",")[0] == str(open_ms)]
    if len(body) != 1:
        raise AssertionError(f"the monthly fixture has no single bar for {date}")
    return "\n".join([lines[0], *body]) + "\n"


def daily_kline_csv(symbol: str, date: str) -> str:
    """One perpetual day, cut out of the month `kline_csv` generates."""
    return _single_day(kline_csv(symbol, date[:7]), date)


def daily_spot_kline_csv(symbol: str, date: str) -> str:
    """One spot day, cut out of the month `spot_kline_csv` generates."""
    return _single_day(spot_kline_csv(symbol, date[:7]), date)


def funding_settlement_times_ms(start_ms: int, end_ms: int) -> tuple[int, ...]:
    """`funding_csv`'s weekly settlement grid restricted to [start, end].

    `funding_csv` restarts the grid at each month's first day, so the times are
    generated month by month exactly as it generates them.
    """
    times: list[int] = []
    for month in ALL_MONTHS:
        for offset in range(0, MONTH_DAYS[month], 7):
            settlement_ms = (EPOCH_DAY_2020 + MONTH_START_DAY[month] + offset) * DAY_MS
            if start_ms <= settlement_ms <= end_ms:
                times.append(settlement_ms)
    return tuple(sorted(times))


def funding_rest_rows(symbol: str, start_ms: int, end_ms: int) -> list[dict[str, object]]:
    """Binance's `/fapi/v1/fundingRate` rows for the window."""
    return [
        {
            "symbol": symbol,
            "fundingTime": settlement_ms,
            "fundingRate": funding_rate(symbol),
            "markPrice": "100.00000000",
        }
        for settlement_ms in funding_settlement_times_ms(start_ms, end_ms)
    ]


def funding_rest_json(symbol: str, start_ms: int, end_ms: int) -> bytes:
    """The REST answer body for the window, byte for byte as it is stored."""
    return json.dumps(funding_rest_rows(symbol, start_ms, end_ms)).encode("utf-8")


@dataclass
class ShadowFetch:
    """One fake fetch over all three shadow sources: the monthly dumps, the
    daily dumps and the funding REST endpoint.

    Every request is recorded in `urls`, so a test can assert what was *not*
    asked for -- that a spot capture never reaches for funding, or that a
    carried tail date is never refetched. The hooks are the failures the
    builder has to survive: a monthly dump a symbol never got (the lagging
    and delisted contracts a real base is full of), a date that is absent for
    good, a date that only appears after some hours (the Sunday wait), a
    dump that carries the wrong day, a response that repeats a settlement,
    a response that carries settlements outside the window it was asked for,
    and a response long enough to have been truncated.

    Two hooks exist for reconciliation (spec section 3.2) alone. Because the
    daily dump and the monthly dump are generated from the same numbers here,
    a capture can only disagree with the monthly dump it is later reconciled
    against if the *daily* source it was built from said something else:
    `altered_closes` changes one served day's close, and `extra_settlements`
    adds a settlement the monthly funding dump does not carry. The capture is
    then built from the altered payload and verifies against its own seal --
    which is the situation reconciliation exists to catch.
    """

    absent_months: frozenset[tuple[str, str, str]] = frozenset()
    absent_dates: frozenset[tuple[str, str]] = frozenset()
    appears_after_attempts: dict[tuple[str, str], int] = field(default_factory=dict)
    wrong_day_dates: frozenset[tuple[str, str]] = frozenset()
    duplicate_settlement: bool = False
    settlements_outside_window: bool = False
    pad_settlements_to: int = 0
    altered_closes: dict[tuple[str, str], str] = field(default_factory=dict)
    extra_settlements: dict[str, tuple[int, ...]] = field(default_factory=dict)
    urls: list[str] = field(default_factory=list)
    attempts: dict[tuple[str, str], int] = field(default_factory=dict)

    def fetch(self, url: str) -> PanelPayload:
        self.urls.append(url)
        if "/fapi/v1/fundingRate" in url:
            return self._funding_rest(url)
        if "/daily/klines/" in url:
            return self._daily_kline(url)
        return self._monthly(url)

    def _funding_rest(self, url: str) -> PanelPayload:
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        symbol = query["symbol"][0]
        start_ms = int(query["startTime"][0])
        end_ms = int(query["endTime"][0])
        rows = funding_rest_rows(symbol, start_ms, end_ms)
        rate = funding_rate(symbol)
        while len(rows) < self.pad_settlements_to:
            # Past the window's end, so the padding cannot be mistaken for a
            # settlement: it is there to make the answer long, nothing else.
            rows.append(
                {
                    "symbol": symbol,
                    "fundingTime": end_ms + len(rows) * HOUR_MS + HOUR_MS,
                    "fundingRate": rate,
                    "markPrice": "100.00000000",
                }
            )
        if self.duplicate_settlement and rows:
            rows.append(dict(rows[0]))
        if self.settlements_outside_window:
            for outside_ms in (start_ms - 1, end_ms + 1):
                rows.append(
                    {
                        "symbol": symbol,
                        "fundingTime": outside_ms,
                        "fundingRate": rate,
                        "markPrice": "100.00000000",
                    }
                )
        for extra_ms in self.extra_settlements.get(symbol, ()):
            rows.append(
                {
                    "symbol": symbol,
                    "fundingTime": extra_ms,
                    "fundingRate": rate,
                    "markPrice": "100.00000000",
                }
            )
        return PanelPayload(
            url=url, raw_bytes=json.dumps(rows).encode("utf-8"), received_time_ns=1
        )

    def _daily_kline(self, url: str) -> PanelPayload:
        symbol, _, remainder = url.rsplit("/", 1)[-1].partition("-1d-")
        date = remainder.removesuffix(".zip")
        key = (symbol, date)
        self.attempts[key] = self.attempts.get(key, 0) + 1
        if key in self.absent_dates:
            raise PanelSourceAbsent(f"404: no daily dump for {symbol} {date}")
        needed = self.appears_after_attempts.get(key)
        if needed is not None and self.attempts[key] <= needed:
            raise PanelSourceAbsent(f"404: the daily dump for {symbol} {date} is not published")
        served = _next_date(date) if key in self.wrong_day_dates else date
        text = (
            daily_spot_kline_csv(symbol, served)
            if "/data/spot/" in url
            else daily_kline_csv(symbol, served)
        )
        altered_close = self.altered_closes.get(key)
        if altered_close is not None:
            text = _with_close(text, altered_close)
        return _payload(url, "k.csv", text)

    def _monthly(self, url: str) -> PanelPayload:
        symbol = next(item for item in SYMBOLS if f"/{item}/" in url or f"/{item}-" in url)
        month = next(item for item in ALL_MONTHS if item in url)
        kind = "fundingRate" if "fundingRate" in url else "klines"
        if (symbol, month, kind) in self.absent_months:
            raise PanelSourceAbsent(f"404: no {kind} dump for {symbol} {month}")
        if "fundingRate" in url:
            return _payload(url, "f.csv", funding_csv(symbol, month))
        if "/data/spot/" in url:
            return _payload(url, "k.csv", spot_kline_csv(symbol, month))
        return _payload(url, "k.csv", kline_csv(symbol, month))


def _with_close(text: str, close: str) -> str:
    """A one-day kline CSV with its close column replaced and nothing else.

    Only the close moves, so the bar disagrees with the monthly dump in
    exactly one field and the refusal document has exactly one record to
    name -- which is what makes the reconciliation test's expectation an
    equality rather than a search.
    """
    header, row = text.splitlines()[:2]
    fields = row.split(",")
    fields[4] = close
    return "\n".join([header, ",".join(fields)]) + "\n"


def _next_date(date: str) -> str:
    """The fixture day after `date`, on the fixture's own month tables."""
    month = date[:7]
    day = int(date[8:10])
    if day < MONTH_DAYS[month]:
        return f"{month}-{day + 1:02d}"
    following = ALL_MONTHS[ALL_MONTHS.index(month) + 1]
    return f"{following}-01"


# --- the weekly shadow book ---------------------------------------------

# The first Sunday the fixture chain can decide on. `select_pair_universe`
# needs `minimum_history_days` daily bars at or before the decision -- 20
# under the reduced v4 declaration -- and the fixture panel opens on
# 2020-01-01, so the Sundays at day offsets 4, 11 and 18 carry 5, 12 and 19
# bars and come back `UNIVERSE_TOO_SMALL`; the fourth Sunday, 2020-01-26 at
# offset 25, carries 26 bars and is the first one a book can be decided on.
SHADOW_ANCHOR_SUNDAY = "2020-01-26"
# The wall clock a fixture shadow capture is built under: the Sunday
# deadline never fires here, because every fixture dump is published
# the moment it is asked for.
_SHADOW_CLOCK_NS = 1_600_000_000_000_000_000
SHADOW_FAMILY_NAME = "funding_carry_panel_v4"
SHADOW_CANDIDATE = "carry_s10_l4w_h26w_exit"


@dataclass(frozen=True, slots=True)
class ShadowBookCaptures:
    """One workspace's captures for a shadow week: both markets, two weeks.

    `run_shadow_week` reads a perpetual capture and a spot capture, and
    Ruling 17 makes the base each weekly capture records required, so a
    fixture week is four directories. The second week is chained onto the
    first with `previous_capture_root`, which is the shape the real Monday
    job runs in.
    """

    root: Path
    perp_base: Path
    spot_base: Path
    perp_first: Path
    spot_first: Path
    perp_second: Path
    spot_second: Path


def build_shadow_book_captures(root: Path) -> ShadowBookCaptures:
    """Base captures over `MONTHS` and two chained weekly captures per market.

    Both markets are built from the same `ShadowFetch`, so the spot capture
    never reaches for funding (`market="spot"`) and the perpetual capture
    carries the REST funding windows the book's trailing measures read.
    """
    perp_base = capture_panel(
        workspace_root=root, output_directory=root / "perp-base", reserve_bytes=0,
        symbols=SYMBOLS, months=MONTHS, fetch=ShadowFetch().fetch,
    ).capture_root
    spot_base = capture_panel(
        workspace_root=root, output_directory=root / "spot-base", reserve_bytes=0,
        symbols=SYMBOLS, months=MONTHS, fetch=ShadowFetch().fetch, market="spot",
    ).capture_root
    weeks: dict[str, Path] = {}
    for market, base in (("um", perp_base), ("spot", spot_base)):
        previous: Path | None = None
        for label, sunday in (("first", FIRST_TAIL_SUNDAY), ("second", SECOND_TAIL_SUNDAY)):
            name = f"{'perp' if market == 'um' else 'spot'}-{label}"
            artifact = build_shadow_capture(
                workspace_root=root,
                base_capture_root=base,
                output_directory=root / name,
                reserve_bytes=0,
                tail_through=sunday,
                market=market,
                fetch=ShadowFetch().fetch,
                previous_capture_root=previous,
                clock=lambda: _SHADOW_CLOCK_NS,
                sleep=lambda _seconds: None,
            )
            weeks[name] = artifact.capture_root
            previous = artifact.capture_root
    return ShadowBookCaptures(
        root=root,
        perp_base=perp_base,
        spot_base=spot_base,
        perp_first=weeks["perp-first"],
        spot_first=weeks["spot-first"],
        perp_second=weeks["perp-second"],
        spot_second=weeks["spot-second"],
    )


def write_shadow_declaration(
    path: Path,
    *,
    family_spec_path: str,
    family_spec_hash: str,
    artifact_root: str,
    registry_path: str,
    family_name: str = SHADOW_FAMILY_NAME,
    candidate: str = SHADOW_CANDIDATE,
    controls: tuple[str, str] = ("no_trade", "random_pairs"),
    anchor_decision_close_date: str = SHADOW_ANCHOR_SUNDAY,
    phase: str = "A",
    holdout_report_hash: str | None = None,
    version: str = "1.0.0",
) -> Path:
    """Write one shadow declaration, defaulting to the fixture family's Phase A."""
    document = {
        "version": version,
        "family_spec_path": family_spec_path,
        "family_spec_hash": family_spec_hash,
        "family_name": family_name,
        "candidate": candidate,
        "controls": list(controls),
        "anchor_decision_close_date": anchor_decision_close_date,
        "phase": phase,
        "holdout_report_hash": holdout_report_hash,
        "artifact_root": artifact_root,
        "registry_path": registry_path,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path
