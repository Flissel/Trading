from decimal import Decimal

from trading_bot.panel_config import PanelUniverseRules
from trading_bot.panel_reader import PanelBar
from trading_bot.panel_universe import build_contract_histories, select_universe

DAY_NS = 86_400_000_000_000
RULES = PanelUniverseRules(
    minimum_history_days=5,
    liquidity_window_days=3,
    minimum_median_quote_volume=Decimal("100"),
    maximum_contracts=3,
    minimum_contracts=2,
    tier_one_rank_limit=1,
)


def bars(symbol: str, *, days: int, volume: str, first_day: int = 0) -> list[PanelBar]:
    output: list[PanelBar] = []
    for index in range(first_day, first_day + days):
        open_time_ns = index * DAY_NS
        close_time_ns = open_time_ns + DAY_NS - 1_000_000
        output.append(
            PanelBar(
                contract_id=f"{symbol}:{first_day * DAY_NS}",
                instrument_id=symbol,
                open_time_ns=open_time_ns,
                close_time_ns=close_time_ns,
                available_time_ns=close_time_ns + 1,
                close=Decimal(100 + index),
                quote_volume=Decimal(volume),
            )
        )
    return output


def test_ranking_capping_and_tiers() -> None:
    rows = (
        bars("AAAUSDT", days=8, volume="900")
        + bars("BBBUSDT", days=8, volume="800")
        + bars("CCCUSDT", days=8, volume="700")
        + bars("DDDUSDT", days=8, volume="600")
    )
    histories = build_contract_histories(tuple(rows))
    snapshot = select_universe(histories, decision_close_ns=7 * DAY_NS - 1_000_000, rules=RULES)
    assert [item.contract_id for item in snapshot.contracts] == [
        "AAAUSDT:0",
        "BBBUSDT:0",
        "CCCUSDT:0",
    ]
    assert snapshot.contracts[0].tier == 1
    assert snapshot.contracts[1].tier == 2
    assert snapshot.reason_codes == ()


def test_short_history_and_low_liquidity_are_excluded() -> None:
    rows = (
        bars("AAAUSDT", days=8, volume="900")
        + bars("BBBUSDT", days=8, volume="800")
        + bars("SHORTUSDT", days=3, volume="900", first_day=5)
        + bars("THINUSDT", days=8, volume="10")
    )
    histories = build_contract_histories(tuple(rows))
    snapshot = select_universe(histories, decision_close_ns=7 * DAY_NS - 1_000_000, rules=RULES)
    assert [item.contract_id for item in snapshot.contracts] == ["AAAUSDT:0", "BBBUSDT:0"]


def test_missing_decision_day_excludes_the_contract() -> None:
    rows = bars("AAAUSDT", days=8, volume="900") + bars("BBBUSDT", days=6, volume="800")
    histories = build_contract_histories(tuple(rows))
    snapshot = select_universe(histories, decision_close_ns=7 * DAY_NS - 1_000_000, rules=RULES)
    assert snapshot.contracts == ()
    assert snapshot.reason_codes == ("UNIVERSE_TOO_SMALL",)


def test_too_small_universe_is_reported() -> None:
    histories = build_contract_histories(tuple(bars("AAAUSDT", days=8, volume="900")))
    snapshot = select_universe(histories, decision_close_ns=7 * DAY_NS - 1_000_000, rules=RULES)
    assert snapshot.contracts == ()
    assert snapshot.reason_codes == ("UNIVERSE_TOO_SMALL",)


def daily_volume_bars(symbol: str, volumes: list[str], *, first_day: int = 0) -> list[PanelBar]:
    """Build one bar per day with an explicit, independently chosen volume per day."""
    output: list[PanelBar] = []
    for offset, volume in enumerate(volumes):
        index = first_day + offset
        open_time_ns = index * DAY_NS
        close_time_ns = open_time_ns + DAY_NS - 1_000_000
        output.append(
            PanelBar(
                contract_id=f"{symbol}:{first_day * DAY_NS}",
                instrument_id=symbol,
                open_time_ns=open_time_ns,
                close_time_ns=close_time_ns,
                available_time_ns=close_time_ns + 1,
                close=Decimal(100 + index),
                quote_volume=Decimal(volume),
            )
        )
    return output


def test_even_liquidity_window_averages_the_two_middle_volumes() -> None:
    # Production's real config (configs/xs-momentum-panel-v1.json) sets
    # liquidity_window_days=30, which is even, so the even-length branch of the
    # median must be exercised here rather than left to the odd-window RULES fixture.
    rules = PanelUniverseRules(
        minimum_history_days=4,
        liquidity_window_days=4,
        minimum_median_quote_volume=Decimal("250"),
        maximum_contracts=3,
        minimum_contracts=1,
        tier_one_rank_limit=1,
    )
    # Trailing four volumes sort to [100, 100, 300, 300]; the correct median averages
    # the two middle values to 200, which falls below the 250 floor and is excluded.
    # A broken even-length median that returns one middle value outright (300) would
    # clear the floor and wrongly admit the contract, changing the outcome below.
    rows = daily_volume_bars("AAAUSDT", ["100", "300", "100", "300"])
    histories = build_contract_histories(tuple(rows))
    snapshot = select_universe(histories, decision_close_ns=4 * DAY_NS - 1_000_000, rules=rules)
    assert snapshot.contracts == ()
    assert snapshot.reason_codes == ("UNIVERSE_TOO_SMALL",)


def _close_time_ns(day_index: int) -> int:
    return (day_index + 1) * DAY_NS - 1_000_000


def bars_on_days(symbol: str, day_volumes: dict[int, str]) -> list[PanelBar]:
    """One bar per (day index, volume) pair, at whatever days are given -- unlike
    `bars`/`daily_volume_bars`, the day indices need not be consecutive, so
    callers can construct a series with an interior hole."""
    output: list[PanelBar] = []
    for index, volume in sorted(day_volumes.items()):
        open_time_ns = index * DAY_NS
        close_time_ns = _close_time_ns(index)
        output.append(
            PanelBar(
                contract_id=f"{symbol}:0",
                instrument_id=symbol,
                open_time_ns=open_time_ns,
                close_time_ns=close_time_ns,
                available_time_ns=close_time_ns + 1,
                close=Decimal(100 + index),
                quote_volume=Decimal(volume),
            )
        )
    return output


def test_liquidity_median_requires_a_complete_calendar_window() -> None:
    """Spec 7.1 item 3's "trailing 30-day median quote_volume" is a calendar
    span ending at the decision, and that window must be complete -- the same
    requirement as `_annualised_volatility`'s (panel_signals.py), and for a
    sharper reason here: a subset median is not merely a smaller sample, it
    is a biased one, because days go missing non-randomly -- concentrated on
    halted, dormant, and delisting-adjacent contracts, exactly where the
    missing days are the low-volume ones -- and the liquidity floor is a
    one-sided gate. Only two of a ten-calendar-day window's days have any
    bar at all, both carrying an enormous volume (50,000,000); the other
    eight are an interior hole. A pre-fix, merely calendar-bounded window
    would compute a median from only those two present values and admit the
    contract at 50,000,000 -- the reviewer's own measured failure mode on
    the real capture, reproduced here from first principles. Requiring
    completeness excludes the contract instead, regardless of how high the
    partial median would be."""
    day_volumes = {16: "50000000", 17: "50000000"}
    rules = PanelUniverseRules(
        minimum_history_days=2,
        liquidity_window_days=10,
        minimum_median_quote_volume=Decimal("5000000"),
        maximum_contracts=1,
        minimum_contracts=1,
        tier_one_rank_limit=1,
    )
    histories = build_contract_histories(tuple(bars_on_days("AAAUSDT", day_volumes)))
    decision = _close_time_ns(17)
    snapshot = select_universe(histories, decision_close_ns=decision, rules=rules)
    assert snapshot.contracts == ()
    assert snapshot.reason_codes == ("UNIVERSE_TOO_SMALL",)


def test_liquidity_median_admits_a_genuinely_complete_calendar_window() -> None:
    """Sanity counterpart to the completeness test above: with every day of a
    ten-calendar-day window present and above the floor, the contract must
    still be admitted -- the fix requires completeness, it does not turn the
    liquidity filter into an unconditional exclusion."""
    day_volumes = {day: "10000000" for day in range(8, 18)}
    rules = PanelUniverseRules(
        minimum_history_days=10,
        liquidity_window_days=10,
        minimum_median_quote_volume=Decimal("5000000"),
        maximum_contracts=1,
        minimum_contracts=1,
        tier_one_rank_limit=1,
    )
    histories = build_contract_histories(tuple(bars_on_days("AAAUSDT", day_volumes)))
    decision = _close_time_ns(17)
    snapshot = select_universe(histories, decision_close_ns=decision, rules=rules)
    assert [item.contract_id for item in snapshot.contracts] == ["AAAUSDT:0"]
    assert snapshot.contracts[0].median_quote_volume == Decimal("10000000")


def test_equal_medians_break_ties_by_contract_id() -> None:
    # All three contracts carry the same volume, so their medians tie exactly; the
    # rows are added out of alphabetical order so the assertion cannot pass merely
    # by echoing input order back unchanged.
    rows = (
        bars("CCCUSDT", days=8, volume="900")
        + bars("AAAUSDT", days=8, volume="900")
        + bars("BBBUSDT", days=8, volume="900")
    )
    histories = build_contract_histories(tuple(rows))
    snapshot = select_universe(histories, decision_close_ns=7 * DAY_NS - 1_000_000, rules=RULES)
    assert [item.contract_id for item in snapshot.contracts] == [
        "AAAUSDT:0",
        "BBBUSDT:0",
        "CCCUSDT:0",
    ]
