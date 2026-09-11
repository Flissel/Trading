from decimal import Decimal

from trading_bot.carry_config import CarryPair, CarryUniverseRules
from trading_bot.carry_universe import select_pair_universe
from trading_bot.panel_reader import PanelBar
from trading_bot.panel_universe import build_contract_histories

DAY_NS = 86_400_000_000_000
RULES = CarryUniverseRules(
    minimum_history_days=5, liquidity_window_days=3, minimum_median_quote_volume=Decimal("100"),
    maximum_pairs=3, minimum_pairs=2, tier_one_rank_limit=1,
)
DECISION = 7 * DAY_NS - 1_000_000


def bars(symbol: str, *, days: int, volume: str, first_day: int = 0) -> list[PanelBar]:
    out = []
    for index in range(first_day, first_day + days):
        open_time_ns = index * DAY_NS
        close_time_ns = open_time_ns + DAY_NS - 1_000_000
        out.append(PanelBar(
            contract_id=f"{symbol}:{first_day * DAY_NS}", instrument_id=symbol,
            open_time_ns=open_time_ns, close_time_ns=close_time_ns,
            available_time_ns=close_time_ns + 1, close=Decimal(100 + index),
            quote_volume=Decimal(volume),
        ))
    return out


PAIRS = tuple(
    CarryPair(perpetual=s, spot=s, multiplier=1)
    for s in ("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT")
)


def test_pairs_need_both_legs_and_are_ranked_by_the_perpetual() -> None:
    perp = build_contract_histories(tuple(
        bars("AAAUSDT", days=8, volume="900") + bars("BBBUSDT", days=8, volume="800")
        + bars("CCCUSDT", days=8, volume="700") + bars("DDDUSDT", days=8, volume="600")))
    spot = build_contract_histories(tuple(
        bars("AAAUSDT", days=8, volume="500") + bars("BBBUSDT", days=8, volume="900")
        + bars("CCCUSDT", days=8, volume="10")))  # CCC is illiquid, DDD has no spot
    snapshot = select_pair_universe(
        perp, spot, pairs=PAIRS, decision_close_ns=DECISION, rules=RULES
    )
    assert [p.pair_id for p in snapshot.pairs] == ["AAAUSDT:0", "BBBUSDT:0"]
    assert (
        snapshot.pairs[0].perpetual_tier == 1
        and snapshot.pairs[1].perpetual_tier == 2
    )
    # spot ranks among spot legs: BBB (900) -> tier 1, AAA (500) -> tier 2
    assert snapshot.pairs[0].spot_tier == 2 and snapshot.pairs[1].spot_tier == 1
    # the pair tier is the worse of the two legs
    assert [p.tier for p in snapshot.pairs] == [2, 2]
    assert snapshot.reason_codes == ()


def test_too_few_pairs_is_universe_too_small() -> None:
    perp = build_contract_histories(tuple(bars("AAAUSDT", days=8, volume="900")))
    spot = build_contract_histories(tuple(bars("AAAUSDT", days=8, volume="900")))
    snapshot = select_pair_universe(
        perp, spot, pairs=PAIRS, decision_close_ns=DECISION, rules=RULES
    )
    assert (
        snapshot.pairs == () and snapshot.reason_codes == ("UNIVERSE_TOO_SMALL",)
    )


def test_a_hole_in_either_leg_excludes_the_pair() -> None:
    perp = build_contract_histories(tuple(
        bars("AAAUSDT", days=8, volume="900") + bars("BBBUSDT", days=8, volume="800")
    ))
    spot_rows = (
        bars("AAAUSDT", days=8, volume="900") + bars("BBBUSDT", days=8, volume="800")
    )
    spot_rows = [
        b
        for b in spot_rows
        if not (b.instrument_id == "BBBUSDT" and b.open_time_ns == 5 * DAY_NS)
    ]
    spot = build_contract_histories(tuple(spot_rows))
    snapshot = select_pair_universe(
        perp,
        spot,
        pairs=PAIRS,
        decision_close_ns=DECISION,
        rules=CarryUniverseRules(
            **{**RULES.model_dump(), "minimum_pairs": 1}
        ),
    )
    assert [p.pair_id for p in snapshot.pairs] == ["AAAUSDT:0"]


def test_maximum_pairs_truncates_after_ranking() -> None:
    symbols = [f"S{i}USDT" for i in range(6)]
    pairs = tuple(CarryPair(perpetual=s, spot=s, multiplier=1) for s in symbols)
    perp = build_contract_histories(
        tuple(
            b
            for i, s in enumerate(symbols)
            for b in bars(s, days=8, volume=str(900 - i))
        )
    )
    spot = build_contract_histories(
        tuple(b for s in symbols for b in bars(s, days=8, volume="900"))
    )
    snapshot = select_pair_universe(
        perp, spot, pairs=pairs, decision_close_ns=DECISION, rules=RULES
    )
    assert len(snapshot.pairs) == 3
    assert [p.liquidity_rank for p in snapshot.pairs] == [1, 2, 3]
