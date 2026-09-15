from decimal import Decimal
from pathlib import Path

import pytest

from trading_bot.panel_config import load_panel_family_spec
from trading_bot.panel_reader import PanelBar
from trading_bot.panel_signals import (
    _cross_sectional_weights,
    build_weight_vectors,
)
from trading_bot.panel_signals import (
    _time_series_weights as panel_time_series_weights,
)
from trading_bot.panel_universe import (
    ContractHistory,
    UniverseSnapshot,
    build_contract_histories,
    select_universe,
)
from trading_bot.trend_config import INDICATOR_NAMES, load_trend_family_spec
from trading_bot.trend_signals import (
    _time_series_weights as trend_time_series_weights,
)
from trading_bot.trend_signals import (
    build_trend_weight_vectors,
    closes_before,
    complete_span,
    ema,
    indicator_votes,
    sma,
    threshold_sign,
    trend_score,
    trend_scores,
)

DAY_NS = 86_400_000_000_000
DAYS = 130
DECISION = DAYS * DAY_NS - 1_000_000
TOLERANCE = Decimal("1E-20")

SPEC, _ = load_trend_family_spec(Path("configs/trend-aggregate-panel-v1.json"))
PANEL_SPEC, _ = load_panel_family_spec(Path("configs/xs-momentum-panel-v1.json"))
ELIGIBLE_COUNT = SPEC.universe.minimum_contracts


# --------------------------------------------------------------------------
# Close-series fixtures. Every series is daily and ends exactly at `DECISION`,
# so a series of `n` closes fills the n-day calendar span ending at the
# decision: the 91-bar history rule, the complete 30-day liquidity window and
# the complete 30-day volatility window all hold for the 130-day series.
# --------------------------------------------------------------------------


def flat_closes() -> tuple[Decimal, ...]:
    """A perfectly constant series: every indicator's difference is an exact zero."""
    return tuple(Decimal(100) for _ in range(DAYS))


def rising_closes(days: int = DAYS) -> tuple[Decimal, ...]:
    """A strictly increasing arithmetic series 100, 101, ... (step 1)."""
    return tuple(Decimal(100 + day) for day in range(days))


def rise_then_flat_closes(turn: int) -> tuple[Decimal, ...]:
    """Rises 100, 101, ... up to day `turn`, then holds `100 + turn` to the end."""
    return tuple(Decimal(100 + min(day, turn)) for day in range(DAYS))


def decline_then_rally_closes() -> tuple[Decimal, ...]:
    """400 falling by 3 a day to 40 on day 120, then rallying by 6 a day to 94.

    A long decline with a short sharp rally on the end: the fixture the
    whole-series MACD is pinned with below.
    """
    return tuple(
        Decimal(400) - Decimal(3) * day if day <= 120 else Decimal(40) + Decimal(6) * (day - 120)
        for day in range(DAYS)
    )


def step_closes(scale: int) -> tuple[Decimal, ...]:
    """`100 * 2 ** (scale * (day // 6))`: a staircase that steps once every six days.

    Non-decreasing for a positive `scale` and non-increasing for a negative one,
    never constant, and every value is an exact Decimal (a power of two times
    100, at most 2**84 * 100, which is 28 significant digits). Every daily log
    return is `scale * ln 2` on a step day and 0 otherwise, so the realised
    volatility of `step_closes(4)` is exactly four times that of
    `step_closes(1)` and `step_closes(-1)`'s equals `step_closes(1)`'s -- the
    property the inverse-volatility weights below are hand-computed from.
    """
    return tuple(Decimal(100) * Decimal(2) ** (scale * (day // 6)) for day in range(DAYS))


def close_times_for(count: int) -> tuple[int, ...]:
    """`count` consecutive daily close times, the last of them `DECISION`."""
    return tuple(DECISION - (count - 1 - index) * DAY_NS for index in range(count))


def history_for(
    symbol: str, closes: tuple[Decimal, ...], *, holes: tuple[int, ...] = ()
) -> ContractHistory:
    """One contract's daily history ending at `DECISION`.

    `holes` names day offsets counted back from the decision (0 is the decision
    day itself); those closes are deleted, which is how a calendar hole is
    punched into an otherwise contiguous series.
    """
    dropped = {DECISION - offset * DAY_NS for offset in holes}
    kept = {
        close_time: close
        for close_time, close in zip(close_times_for(len(closes)), closes, strict=True)
        if close_time not in dropped
    }
    return ContractHistory(
        contract_id=f"{symbol}:0",
        instrument_id=symbol,
        closes=kept,
        quote_volumes={},
        close_times=tuple(sorted(kept)),
    )


def votes_for(closes: tuple[Decimal, ...], *, holes: tuple[int, ...] = ()) -> dict[str, int]:
    return indicator_votes(history_for("VOTEUSDT", closes, holes=holes), DECISION)


def score_for(closes: tuple[Decimal, ...], *, holes: tuple[int, ...] = ()) -> Decimal | None:
    return trend_score(history_for("VOTEUSDT", closes, holes=holes), DECISION)


def bars_for(symbol: str, closes: tuple[Decimal, ...]) -> list[PanelBar]:
    rows: list[PanelBar] = []
    for close_time_ns, close in zip(close_times_for(len(closes)), closes, strict=True):
        rows.append(
            PanelBar(
                contract_id=f"{symbol}:0",
                instrument_id=symbol,
                open_time_ns=close_time_ns - DAY_NS + 1_000_000,
                close_time_ns=close_time_ns,
                available_time_ns=close_time_ns + 1,
                close=close,
                quote_volume=Decimal("10000000"),
            )
        )
    return rows


def build_panel(signalled: dict[str, tuple[Decimal, ...]]) -> dict[str, ContractHistory]:
    """`signalled` plus enough flat `F###USDT` fillers to reach the eligible minimum.

    The fillers are perfectly flat, so their trend score is exactly 0: they are
    eligible (they carry the full history and the liquidity floor) but no
    time-series member ever holds them, which keeps the hand-computed weights
    below over the named contracts only. `F` sorts before every signalled
    symbol used here (`M`, `S`), which pins the quintile membership.
    """
    rows: list[PanelBar] = []
    for index in range(ELIGIBLE_COUNT - len(signalled)):
        rows.extend(bars_for(f"F{index:03d}USDT", flat_closes()))
    for symbol, closes in signalled.items():
        rows.extend(bars_for(symbol, closes))
    return build_contract_histories(tuple(rows))


def snapshot_for(histories: dict[str, ContractHistory]) -> UniverseSnapshot:
    return select_universe(histories, decision_close_ns=DECISION, rules=SPEC.universe)


# --------------------------------------------------------------------------
# Moving averages
# --------------------------------------------------------------------------


def test_sma_averages_the_last_n_closes() -> None:
    # (3 + 4 + 5) / 3 = 4, and the leading 1, 2 are outside the window.
    closes = tuple(Decimal(value) for value in (1, 2, 3, 4, 5))
    assert sma(closes, 3) == Decimal(4)
    # (1 + 2 + 3 + 4 + 5) / 5 = 3.
    assert sma(closes, 5) == Decimal(3)


def test_sma_rejects_a_window_longer_than_the_series() -> None:
    closes = tuple(Decimal(value) for value in (1, 2, 3))
    with pytest.raises(ValueError, match="needs 4 closes"):
        sma(closes, 4)


def test_ema_matches_a_hand_rolled_recursion() -> None:
    """`alpha = 2 / (n + 1)`, seeded with the first close, applied to the whole series.

    Both windows below give an alpha that is exact in decimal, so the whole
    recursion is exact hand arithmetic rather than a re-run of the code.

    n = 3 -> alpha = 2/4 = 0.5 over (100, 110, 90, 130, 120):
        e0 = 100
        e1 = 0.5*110 + 0.5*100  = 55 + 50     = 105
        e2 = 0.5* 90 + 0.5*105  = 45 + 52.5   = 97.5
        e3 = 0.5*130 + 0.5*97.5 = 65 + 48.75  = 113.75
        e4 = 0.5*120 + 0.5*113.75 = 60 + 56.875 = 116.875

    n = 4 -> alpha = 2/5 = 0.4 over (100, 200, 100, 200, 100):
        e0 = 100
        e1 = 0.4*200 + 0.6*100   = 80 + 60    = 140
        e2 = 0.4*100 + 0.6*140   = 40 + 84    = 124
        e3 = 0.4*200 + 0.6*124   = 80 + 74.4  = 154.4
        e4 = 0.4*100 + 0.6*154.4 = 40 + 92.64 = 132.64
    """
    first = tuple(Decimal(value) for value in (100, 110, 90, 130, 120))
    assert ema(first, 3) == Decimal("116.875")
    second = tuple(Decimal(value) for value in (100, 200, 100, 200, 100))
    assert ema(second, 4) == Decimal("132.64")


def test_ema_of_a_single_close_is_that_close() -> None:
    assert ema((Decimal("7.5"),), 12) == Decimal("7.5")


# --------------------------------------------------------------------------
# Indicator votes
# --------------------------------------------------------------------------


def test_a_monotone_rise_votes_long_on_every_indicator() -> None:
    """130 closes 100, 101, ..., 229. Every vote is +1, by hand:

    - `ma_n`: SMA over the last n closes is the mean of an ascending run whose
      largest member is P_t = 229, so P_t - SMA_n > 0. (SMA_20 = 219.5,
      SMA_50 = 204.5, SMA_100 = 179.5.)
    - `ma_cross_a_b` (a < b): the b-window contains every a-window close plus
      strictly smaller older ones, so SMA_a > SMA_b (219.5 > 204.5 > 179.5).
    - `breakout_n`: the series is strictly increasing, so P_t is the maximum of
      every trailing window and never its minimum -- +1 by construction.
    - `roc_n`: P_t = 229 > P_{t-20} = 209 > P_{t-60} = 169 > P_{t-120} = 109.
    - `macd_12_26`: an EMA is a weighted mean of the series whose weights shift
      toward the recent end as alpha grows, so on a non-decreasing,
      non-constant series the faster EMA_12 is strictly above EMA_26.
    """
    assert votes_for(rising_closes()) == dict.fromkeys(INDICATOR_NAMES, 1)
    assert score_for(rising_closes()) == Decimal(1)


def test_a_flat_series_abstains_on_every_indicator() -> None:
    """A constant series: every SMA, every EMA and every lagged close equals
    P_t, so all nine difference-based votes are an exact zero, and each
    breakout window's high and low are both P_t -- the tie the spec resolves
    to 0 rather than to +1."""
    assert votes_for(flat_closes()) == dict.fromkeys(INDICATOR_NAMES, 0)
    assert score_for(flat_closes()) == Decimal(0)


def test_a_twenty_day_flat_tail_ties_the_twenty_day_breakout() -> None:
    """Closes rise 100..209 through day 109 and then hold 209 for the last 20 days.

    P_t = 209. By hand, window by window:
    - SMA_20 = mean of twenty 209s = 209          -> ma_20 = sign(0)  = 0
    - SMA_50 = (sum 180..209 + 20*209)/50
             = (5835 + 4180)/50 = 200.3           -> ma_50 = +1
    - SMA_100 = (sum 130..209 + 20*209)/100
             = (13560 + 4180)/100 = 177.4         -> ma_100 = +1
    - ma_cross_20_50  = sign(209 - 200.3)         = +1
    - ma_cross_50_100 = sign(200.3 - 177.4)       = +1
    - breakout_20: the last 20 closes are all 209, so P_t is both the high and
      the low of that window -> the tie resolves to 0
    - breakout_50: high 209 = P_t, low 180       -> +1
    - breakout_100: high 209 = P_t, low 130      -> +1
    - roc_20:  P_{t-20} = close[109] = 209        -> sign(0) = 0
    - roc_60:  P_{t-60} = close[69]  = 169        -> +1
    - roc_120: P_{t-120} = close[9]  = 109        -> +1
    - macd_12_26: non-decreasing and non-constant -> +1
    Sum = 9 over twelve votes -> score 9/12 = 0.75.
    """
    closes = rise_then_flat_closes(109)
    assert closes[-1] == Decimal(209)
    assert votes_for(closes) == {
        "ma_20": 0,
        "ma_50": 1,
        "ma_100": 1,
        "ma_cross_20_50": 1,
        "ma_cross_50_100": 1,
        "breakout_20": 0,
        "breakout_50": 1,
        "breakout_100": 1,
        "roc_20": 0,
        "roc_60": 1,
        "macd_12_26": 1,
        "roc_120": 1,
    }
    assert score_for(closes) == Decimal("0.75")


def test_a_sixty_day_flat_tail_scores_five_twelfths() -> None:
    """Closes rise 100..169 through day 69 and then hold 169 for the last 60 days.

    P_t = 169. By hand:
    - SMA_20 = SMA_50 = 169 (both windows lie inside the flat tail) -> ma_20,
      ma_50 and ma_cross_20_50 are all sign(0) = 0
    - SMA_100 = (sum 130..169 + 60*169)/100 = (5980 + 10140)/100 = 161.2
      -> ma_100 = +1 and ma_cross_50_100 = sign(169 - 161.2) = +1
    - breakout_20 and breakout_50: the window is constant at 169, so P_t is
      both high and low -> 0 each; breakout_100's low is 130 -> +1
    - roc_20 (close[109] = 169) and roc_60 (close[69] = 169) are sign(0) = 0;
      roc_120 (close[9] = 109) is +1
    - macd_12_26: non-decreasing and non-constant -> +1
    Sum = 5 over twelve votes -> score 5/12, which sits inside [0.2, 0.5) and
    so separates the two time-series thresholds.
    """
    closes = rise_then_flat_closes(69)
    assert closes[-1] == Decimal(169)
    assert votes_for(closes) == {
        "ma_20": 0,
        "ma_50": 0,
        "ma_100": 1,
        "ma_cross_20_50": 0,
        "ma_cross_50_100": 1,
        "breakout_20": 0,
        "breakout_50": 0,
        "breakout_100": 1,
        "roc_20": 0,
        "roc_60": 0,
        "macd_12_26": 1,
        "roc_120": 1,
    }
    score = score_for(closes)
    assert score == Decimal(5) / Decimal(12)
    assert score is not None and Decimal("0.2") <= score < Decimal("0.5")


def test_a_staircase_fall_votes_short_on_every_indicator() -> None:
    """`step_closes(-1)` is non-increasing and non-constant, the mirror of the
    monotone rise: every SMA and every lagged close sits strictly above P_t,
    P_t is each window's low and never its high, and the faster EMA is strictly
    below the slower one."""
    assert votes_for(step_closes(-1)) == dict.fromkeys(INDICATOR_NAMES, -1)
    assert score_for(step_closes(-1)) == Decimal(-1)


def test_the_macd_runs_over_the_whole_series_not_the_last_twenty_six_closes() -> None:
    """A long decline with a short sharp rally, where the two readings disagree.

    `decline_then_rally_closes()` falls 400 -> 40 by 3 a day through day 120 and
    then rallies by 6 a day to 94. The last 26 closes run 88, 85, ... 40, then
    46 ... 94, so a MACD computed over only that window is seeded at 88, and a
    26-day EMA sheds just (1 - 2/27)**25 = 15% of its seed in 25 steps: it lands
    at 69.80, *below* the fast EMA's 71.75, and would vote +1.

    Run over the whole series -- what spec 3 asks for, "seeded with the first
    close of the contract's history" -- the slow EMA still carries the 400 ->
    88 decline above that window and sits at 75.28, above the fast EMA's 72.01
    (the fast EMA barely moves between the two readings: its own seed decays by
    (1 - 2/13)**25, under 2%). So the whole-series vote is -1.

    The sign, not the value, is what is pinned here, and the two readings give
    opposite signs -- so truncating the EMA to any recent window fails this.
    """
    closes = decline_then_rally_closes()
    assert closes[120] == Decimal(40)
    assert closes[-1] == Decimal(94)
    assert closes[-26] == Decimal(88)
    assert votes_for(closes)["macd_12_26"] == -1
    # The truncated reading this guards against, spelled out independently.
    truncated = closes[-26:]
    assert ema(truncated, 12) - ema(truncated, 26) > 0
    whole = closes_before(history_for("VOTEUSDT", closes), DECISION)
    assert ema(whole, 12) - ema(whole, 26) < 0


# --------------------------------------------------------------------------
# Calendar spans (P1.27 protocol 16.1: a window is a calendar span ending at
# the decision, not a count of whatever closes happen to be on hand)
# --------------------------------------------------------------------------

MINIMUM_SPAN_DAYS = {
    "ma_20": 20,
    "ma_50": 50,
    "ma_100": 100,
    "ma_cross_20_50": 50,
    "ma_cross_50_100": 100,
    "breakout_20": 20,
    "breakout_50": 50,
    "breakout_100": 100,
    "roc_20": 21,
    "roc_60": 61,
    "macd_12_26": 121,
    "roc_120": 121,
}


def test_complete_span_returns_the_closes_only_when_the_span_has_no_hole() -> None:
    """The span is the `days` consecutive daily closes ending at the decision.
    A hole anywhere inside it -- or no close at the decision at all -- makes it
    empty rather than reaching further back to make up the count."""
    history = history_for("SPANUSDT", rising_closes())
    assert complete_span(history, DECISION, 3) == (Decimal(227), Decimal(228), Decimal(229))
    assert len(complete_span(history, DECISION, DAYS)) == DAYS
    # A 131-day span reaches one day past the start of the history.
    assert complete_span(history, DECISION, DAYS + 1) == ()
    holed = history_for("SPANUSDT", rising_closes(), holes=(2,))
    assert complete_span(holed, DECISION, 2) == (Decimal(228), Decimal(229))
    assert complete_span(holed, DECISION, 3) == ()
    assert complete_span(history_for("SPANUSDT", rising_closes(), holes=(0,)), DECISION, 1) == ()


def test_each_indicator_appears_exactly_at_its_minimum_span() -> None:
    """An indicator whose calendar span reaches past the contract's history is
    absent, not zero: a crossover needs the longer of its two SMAs, a rate of
    change needs the lagged day as well (n + 1 days), and the MACD needs the
    whole span the score rests on because its EMAs must not step across a hole
    inside it."""
    assert set(MINIMUM_SPAN_DAYS) == set(INDICATOR_NAMES)
    series = rising_closes(max(MINIMUM_SPAN_DAYS.values()))
    for name, minimum in MINIMUM_SPAN_DAYS.items():
        assert name not in votes_for(series[-(minimum - 1) :]), name
        assert name in votes_for(series[-minimum:]), name


def test_an_interior_hole_drops_every_indicator_whose_span_covers_it() -> None:
    """One day deleted 69 days before the decision, out of an otherwise
    contiguous 130-day series.

    The spans that reach past day 69 -- `ma_100` and `breakout_100` (100 days),
    `ma_cross_50_100` (its slow leg is 100 days), `roc_120` (121 days, the
    lagged day included) and `macd_12_26` (the score's own 121-day span) --
    are no longer calendar-complete, so those five votes are absent. The 20-,
    50- and 60-day spans all end before day 69 and are untouched. With five
    votes missing there is no score, so the contract is not rankable at this
    decision -- the same fail-closed outcome `_annualised_volatility` and
    `select_universe` already produce for a hole in their own windows.
    """
    votes = votes_for(rising_closes(), holes=(69,))
    assert set(votes) == {
        "ma_20",
        "ma_50",
        "ma_cross_20_50",
        "breakout_20",
        "breakout_50",
        "roc_20",
        "roc_60",
    }
    assert score_for(rising_closes(), holes=(69,)) is None


def test_a_hole_outside_every_span_changes_nothing() -> None:
    """The longest span the score rests on is 121 days, so a day deleted 125
    days before the decision lies outside every one of them: every vote is
    computable and unchanged, and so is the score."""
    assert votes_for(rising_closes(), holes=(125,)) == votes_for(rising_closes())
    assert score_for(rising_closes(), holes=(125,)) == Decimal(1)


def test_a_missing_decision_close_leaves_no_votes() -> None:
    """Every span ends at the decision, so a contract with no bar at the
    decision has no computable indicator at all."""
    assert votes_for(rising_closes(), holes=(0,)) == {}
    assert score_for(rising_closes(), holes=(0,)) is None


def test_votes_come_back_in_the_declared_indicator_order() -> None:
    assert tuple(votes_for(rising_closes())) == INDICATOR_NAMES


def test_trend_score_is_none_below_twelve_votes() -> None:
    """A 120-day history leaves the two 121-day spans (`roc_120` and
    `macd_12_26`) short, so ten votes are computable and there is no score."""
    votes = votes_for(rising_closes(120))
    assert len(votes) == 10
    assert "roc_120" not in votes
    assert "macd_12_26" not in votes
    assert score_for(rising_closes(120)) is None
    assert score_for(rising_closes(121)) is not None
    assert score_for(()) is None


# --------------------------------------------------------------------------
# Point-in-time close selection and the threshold rule
# --------------------------------------------------------------------------


def test_closes_before_stops_at_the_decision() -> None:
    closes = {day * DAY_NS: Decimal(100 + day) for day in range(5)}
    history = ContractHistory(
        contract_id="CUTUSDT:0",
        instrument_id="CUTUSDT",
        closes=closes,
        quote_volumes={},
        close_times=tuple(sorted(closes)),
    )
    assert closes_before(history, 2 * DAY_NS) == (Decimal(100), Decimal(101), Decimal(102))
    assert closes_before(history, 2 * DAY_NS - 1) == (Decimal(100), Decimal(101))
    assert closes_before(history, -1) == ()


def test_threshold_sign_is_inclusive_at_both_edges() -> None:
    threshold = Decimal("0.2")
    assert threshold_sign(Decimal("0.2"), threshold) == 1
    assert threshold_sign(Decimal("0.200001"), threshold) == 1
    assert threshold_sign(Decimal("0.199999"), threshold) == 0
    assert threshold_sign(Decimal(0), threshold) == 0
    assert threshold_sign(Decimal("-0.199999"), threshold) == 0
    assert threshold_sign(Decimal("-0.2"), threshold) == -1
    assert threshold_sign(Decimal(-1), threshold) == -1


def test_trend_scores_skips_a_contract_without_twelve_votes() -> None:
    histories = {
        history.contract_id: history
        for history in (
            history_for("LONGUSDT", rising_closes()),
            history_for("HOLEUSDT", rising_closes(), holes=(69,)),
            history_for("SHORTUSDT", rising_closes(120)),
        )
    }
    scores = trend_scores(histories, tuple(histories), DECISION)
    assert scores == {"LONGUSDT:0": Decimal(1)}


# --------------------------------------------------------------------------
# Weight vectors
# --------------------------------------------------------------------------


def test_time_series_weights_are_inverse_volatility_and_unit_gross() -> None:
    """Two signalled contracts whose sigmas stand in an exact 4:1 ratio.

    `step_closes(4)`'s daily log returns are exactly four times
    `step_closes(1)`'s (both step on the same days, by ln 16 = 4 ln 2 against
    ln 2), so its realised sigma is exactly four times as large, and both sit
    far above the 0.20 volatility floor (about 20.1 and 5.0 annualised), so
    neither is clipped by it. The raw weights are 1/sigma and 1/(4 sigma);
    normalised to unit gross that is 0.8 and 0.2. The cap is
    `time_series_cap_numerator / n` = 2/2 = 1, so no water-filling applies.
    The remaining 38 contracts are flat (score 0), so they are not signalled.
    """
    histories = build_panel({"S000USDT": step_closes(1), "S001USDT": step_closes(4)})
    snapshot = snapshot_for(histories)
    assert len(snapshot.contracts) == ELIGIBLE_COUNT
    vectors = build_trend_weight_vectors(histories, snapshot, spec=SPEC)
    weights = dict(vectors["ta_ts_t02"].weights)
    assert set(weights) == {"S000USDT:0", "S001USDT:0"}
    assert abs(weights["S000USDT:0"] - Decimal("0.8")) < TOLERANCE
    assert abs(weights["S001USDT:0"] - Decimal("0.2")) < TOLERANCE
    gross = sum((abs(value) for value in weights.values()), Decimal(0))
    assert abs(gross - Decimal(1)) < TOLERANCE
    assert weights["S000USDT:0"] > weights["S001USDT:0"]


def test_time_series_weights_water_fill_at_the_cap() -> None:
    """One low-volatility contract against four whose sigma is exactly four times it.

    Raw weights 1/sigma and 4 x 1/(4 sigma); normalised that is 0.5 for the
    quiet contract and 0.125 for each of the other four. With five signalled
    contracts the cap is 2/5 = 0.4, so the quiet one violates it: it is clipped
    to exactly 0.4 and frozen, leaving 1 - 0.4 = 0.6 of gross for the other
    four, whose own gross is 4 x 0.125 = 0.5 -- so they are scaled by
    0.6/0.5 = 1.2 to 0.15 each. 0.4 + 4 x 0.15 = 1.
    """
    histories = build_panel(
        {
            "S000USDT": step_closes(1),
            "S001USDT": step_closes(4),
            "S002USDT": step_closes(4),
            "S003USDT": step_closes(4),
            "S004USDT": step_closes(4),
        }
    )
    snapshot = snapshot_for(histories)
    vectors = build_trend_weight_vectors(histories, snapshot, spec=SPEC)
    weights = dict(vectors["ta_ts_t02"].weights)
    assert len(weights) == 5
    assert weights["S000USDT:0"] == Decimal("0.4")
    for index in range(1, 5):
        assert abs(weights[f"S00{index}USDT:0"] - Decimal("0.15")) < TOLERANCE
    gross = sum((abs(value) for value in weights.values()), Decimal(0))
    assert abs(gross - Decimal(1)) < TOLERANCE


def test_a_falling_contract_is_shorted() -> None:
    """`step_closes(-1)` scores -1 and `step_closes(1)` scores +1; their log
    returns are negatives of one another, so their sigmas are identical and the
    inverse-volatility weights are a balanced +0.5 / -0.5 pair."""
    histories = build_panel({"S000USDT": step_closes(1), "S001USDT": step_closes(-1)})
    snapshot = snapshot_for(histories)
    vectors = build_trend_weight_vectors(histories, snapshot, spec=SPEC)
    weights = dict(vectors["ta_ts_t02"].weights)
    assert abs(weights["S000USDT:0"] - Decimal("0.5")) < TOLERANCE
    assert abs(weights["S001USDT:0"] + Decimal("0.5")) < TOLERANCE


def test_the_trend_time_series_weights_are_the_panel_construction() -> None:
    """P1.27 drift guard: fed one signed set, `trend_signals._time_series_weights`
    and `panel_signals._time_series_weights` must return the identical vector.

    The two differ only in where the sign comes from -- a thresholded score
    here, a trailing return there -- so scores of +/-1 against trailing returns
    of +/-0.1 with the same signs (and the two families' byte-identical weight
    rules) have to produce the same inverse-volatility, water-filled book.
    """
    histories = build_panel(
        {
            "S000USDT": step_closes(1),
            "S001USDT": step_closes(4),
            "S002USDT": step_closes(-1),
        }
    )
    snapshot = snapshot_for(histories)
    eligible = tuple(item.contract_id for item in snapshot.contracts)
    # `F000USDT` is one of the flat fillers: its realised sigma is exactly 0, so
    # the volatility floor binds for it and not for the three staircases (sigma
    # 5.0 and 20.1) -- both branches of `max(sigma, floor)` are exercised.
    signs = {
        "F000USDT:0": Decimal(1),
        "S000USDT:0": Decimal(1),
        "S001USDT:0": Decimal(1),
        "S002USDT:0": Decimal(-1),
    }
    returns = {contract: sign / Decimal(10) for contract, sign in signs.items()}
    ours = trend_time_series_weights(
        signs, histories, eligible, DECISION, SPEC.weights, Decimal("0.2")
    )
    theirs = panel_time_series_weights(returns, histories, eligible, DECISION, PANEL_SPEC.weights)
    assert ours == theirs
    assert len(ours) == 4


def test_the_higher_threshold_silences_a_mid_score_contract() -> None:
    """`M000USDT` scores 5/12 and `S000USDT` scores 1. `ta_ts_t02` (threshold
    0.2) holds both; `ta_ts_t05` (threshold 0.5) holds only the strong one,
    which then carries the whole unit gross on its own (the cap is 2/1 = 2)."""
    histories = build_panel({"S000USDT": step_closes(1), "M000USDT": rise_then_flat_closes(69)})
    snapshot = snapshot_for(histories)
    vectors = build_trend_weight_vectors(histories, snapshot, spec=SPEC)
    assert {contract for contract, _ in vectors["ta_ts_t02"].weights} == {
        "M000USDT:0",
        "S000USDT:0",
    }
    assert vectors["ta_ts_t05"].weights == (("S000USDT:0", Decimal(1)),)


def test_a_score_exactly_at_the_threshold_is_long() -> None:
    """The declared thresholds (0.2, 0.5) are unreachable exactly by a mean of
    twelve votes, so the boundary is pinned with a threshold of 1 against a
    contract scoring exactly 1: `>=` must hold it, while the 5/12 contract
    stays flat."""
    members = tuple(
        member.model_copy(update={"threshold": Decimal(1)})
        if member.name == "ta_ts_t02"
        else member
        for member in SPEC.members
    )
    spec = SPEC.model_copy(update={"members": members})
    histories = build_panel({"S000USDT": step_closes(1), "M000USDT": rise_then_flat_closes(69)})
    snapshot = snapshot_for(histories)
    vectors = build_trend_weight_vectors(histories, snapshot, spec=spec)
    assert vectors["ta_ts_t02"].weights == (("S000USDT:0", Decimal(1)),)


def test_a_flat_panel_leaves_every_time_series_member_empty() -> None:
    """Every contract scores exactly 0, which is inside both dead bands, so no
    time-series member holds anything -- the empty weight vector Task 3 reports
    as `MEMBER_HELD_NOTHING`."""
    histories = build_panel({})
    snapshot = snapshot_for(histories)
    vectors = build_trend_weight_vectors(histories, snapshot, spec=SPEC)
    for name in ("ta_ts_t02", "ta_ts_t05", "ta_ts_t02_h4w"):
        assert vectors[name].weights == ()


def test_the_four_week_member_matches_the_one_week_member_at_a_decision() -> None:
    """`ta_ts_t02_h4w` differs from `ta_ts_t02` only in how the fold runner
    holds it (Task 3); the vector built at a single decision is the same."""
    histories = build_panel({"S000USDT": step_closes(1), "S001USDT": step_closes(4)})
    snapshot = snapshot_for(histories)
    vectors = build_trend_weight_vectors(histories, snapshot, spec=SPEC)
    assert vectors["ta_ts_t02_h4w"].weights == vectors["ta_ts_t02"].weights


def test_the_cross_sectional_member_is_the_panel_quintile_of_the_scores() -> None:
    """Five staircase contracts score 1 and the 35 flat fillers score 0. With
    40 rankable contracts the quintile is max(8, 40 // 5) = 8 and each leg
    carries `leg_gross` 0.5, so every position is 0.5 / 8 = 0.0625. The panel
    helper ranks on (score, contract_id), so the long leg is the five `M`
    contracts plus the three highest-sorting flats, and the short leg is the
    eight lowest-sorting flats."""
    histories = build_panel({f"M{index:03d}USDT": step_closes(1) for index in range(5)})
    snapshot = snapshot_for(histories)
    vectors = build_trend_weight_vectors(histories, snapshot, spec=SPEC)
    weights = dict(vectors["ta_xs_q5"].weights)
    longs = {contract for contract, value in weights.items() if value > 0}
    shorts = {contract for contract, value in weights.items() if value < 0}
    assert longs == {f"M{index:03d}USDT:0" for index in range(5)} | {
        "F032USDT:0",
        "F033USDT:0",
        "F034USDT:0",
    }
    assert shorts == {f"F{index:03d}USDT:0" for index in range(8)}
    assert all(abs(value) == Decimal("0.0625") for value in weights.values())
    expected_scores = {
        contract.contract_id: (Decimal(1) if contract.contract_id.startswith("M") else Decimal(0))
        for contract in snapshot.contracts
    }
    assert vectors["ta_xs_q5"].weights == _cross_sectional_weights(
        expected_scores, SPEC.weights, reverse=False
    )
    assert trend_scores(histories, tuple(expected_scores), DECISION) == expected_scores


def test_the_controls_equal_the_panel_controls_on_the_same_snapshot() -> None:
    """The three controls are P1.27's, unchanged: `no_trade` flat,
    `passive_long_ew` equal-weight over the eligible set in contract-id order,
    and `random_ranks` drawn from `random_seed ^ decision`. The trend config
    copies P1.27's `weights` and `statistics` blocks byte for byte, so the
    vectors must be identical objects, not merely similar ones."""
    histories = build_panel({"S000USDT": step_closes(1), "S001USDT": step_closes(4)})
    snapshot = snapshot_for(histories)
    trend_vectors = build_trend_weight_vectors(histories, snapshot, spec=SPEC)
    panel_vectors = build_weight_vectors(histories, snapshot, spec=PANEL_SPEC)
    for name in ("no_trade", "passive_long_ew", "random_ranks"):
        assert trend_vectors[name] == panel_vectors[name], name
    assert trend_vectors["no_trade"].weights == ()
    assert len(trend_vectors["passive_long_ew"].weights) == ELIGIBLE_COUNT


def test_every_member_and_control_is_present() -> None:
    histories = build_panel({"S000USDT": step_closes(1), "S001USDT": step_closes(4)})
    snapshot = snapshot_for(histories)
    vectors = build_trend_weight_vectors(histories, snapshot, spec=SPEC)
    expected = tuple(member.name for member in SPEC.members) + tuple(
        control.name for control in SPEC.controls
    )
    assert set(vectors) == set(expected)
    assert all(vector.decision_close_ns == DECISION for vector in vectors.values())
    assert all(vectors[name].member == name for name in vectors)


def test_an_empty_universe_propagates_its_reason_codes() -> None:
    """Too few eligible contracts: every member and control comes back empty
    carrying the snapshot's reason codes, exactly as the panel builder does."""
    histories = build_contract_histories(tuple(bars_for("S000USDT", step_closes(1))))
    snapshot = snapshot_for(histories)
    assert snapshot.contracts == ()
    vectors = build_trend_weight_vectors(histories, snapshot, spec=SPEC)
    assert len(vectors) == len(SPEC.members) + len(SPEC.controls)
    for vector in vectors.values():
        assert vector.weights == ()
        assert vector.reason_codes == ("UNIVERSE_TOO_SMALL",)


def test_weight_vectors_ignore_a_bar_that_closes_after_the_decision() -> None:
    """Point-in-time regression: a bar closing one day after the decision must
    change nothing -- not the scores (whose calendar spans end at the decision)
    and not the volatility window."""
    histories = build_panel({"S000USDT": step_closes(1), "S001USDT": step_closes(4)})
    earlier_decision = DECISION - DAY_NS
    snapshot = select_universe(histories, decision_close_ns=earlier_decision, rules=SPEC.universe)
    with_future = build_trend_weight_vectors(histories, snapshot, spec=SPEC)

    truncated = {
        contract_id: ContractHistory(
            contract_id=history.contract_id,
            instrument_id=history.instrument_id,
            closes={
                time: close for time, close in history.closes.items() if time <= earlier_decision
            },
            quote_volumes={
                time: volume
                for time, volume in history.quote_volumes.items()
                if time <= earlier_decision
            },
            close_times=tuple(time for time in history.close_times if time <= earlier_decision),
        )
        for contract_id, history in histories.items()
    }
    without_future = build_trend_weight_vectors(
        truncated,
        select_universe(truncated, decision_close_ns=earlier_decision, rules=SPEC.universe),
        spec=SPEC,
    )
    assert with_future == without_future
