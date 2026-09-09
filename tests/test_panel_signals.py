from decimal import Decimal
from pathlib import Path

from trading_bot.panel_config import load_panel_family_spec
from trading_bot.panel_reader import PanelBar
from trading_bot.panel_signals import build_weight_vectors
from trading_bot.panel_universe import build_contract_histories, select_universe

DAY_NS = 86_400_000_000_000
SPEC, _ = load_panel_family_spec(Path("configs/xs-momentum-panel-v1.json"))


def series(symbol: str, closes: list[str]) -> list[PanelBar]:
    output: list[PanelBar] = []
    for index, close in enumerate(closes):
        open_time_ns = index * DAY_NS
        close_time_ns = open_time_ns + DAY_NS - 1_000_000
        output.append(
            PanelBar(
                contract_id=f"{symbol}:0",
                instrument_id=symbol,
                open_time_ns=open_time_ns,
                close_time_ns=close_time_ns,
                available_time_ns=close_time_ns + 1,
                close=Decimal(close),
                quote_volume=Decimal("10000000"),
            )
        )
    return output


def panel(count: int, days: int = 100) -> tuple[PanelBar, ...]:
    rows: list[PanelBar] = []
    for index in range(count):
        symbol = f"C{index:03d}USDT"
        drift = Decimal(index) / Decimal(1000)
        closes = [str(Decimal(100) * (Decimal(1) + drift) ** day) for day in range(days)]
        rows.extend(series(symbol, closes))
    return tuple(rows)


def test_cross_sectional_legs_are_balanced_and_sorted() -> None:
    bars = panel(50)
    histories = build_contract_histories(bars)
    decision = 99 * DAY_NS - 1_000_000
    snapshot = select_universe(histories, decision_close_ns=decision, rules=SPEC.universe)
    vectors = build_weight_vectors(histories, snapshot, spec=SPEC)
    momentum = vectors["xs_mom_4w"]
    positive = [item for item in momentum.weights if item[1] > 0]
    negative = [item for item in momentum.weights if item[1] < 0]
    assert len(positive) == len(negative) == 10
    assert sum((item[1] for item in positive), Decimal(0)) == Decimal("0.5")
    assert sum((item[1] for item in negative), Decimal(0)) == Decimal("-0.5")
    winners = {item[0] for item in positive}
    assert "C049USDT:0" in winners
    assert "C000USDT:0" not in winners


def test_reversal_is_the_mirror_of_momentum() -> None:
    bars = panel(50)
    histories = build_contract_histories(bars)
    decision = 99 * DAY_NS - 1_000_000
    snapshot = select_universe(histories, decision_close_ns=decision, rules=SPEC.universe)
    vectors = build_weight_vectors(histories, snapshot, spec=SPEC)
    momentum = dict(vectors["xs_mom_1w"].weights)
    reversal = dict(vectors["xs_rev_1w"].weights)
    assert set(momentum) == set(reversal)
    assert all(reversal[key] == -value for key, value in momentum.items())


def test_time_series_weights_are_unit_gross_and_capped() -> None:
    bars = panel(50)
    histories = build_contract_histories(bars)
    decision = 99 * DAY_NS - 1_000_000
    snapshot = select_universe(histories, decision_close_ns=decision, rules=SPEC.universe)
    vectors = build_weight_vectors(histories, snapshot, spec=SPEC)
    weights = vectors["ts_mom_12w"].weights
    gross = sum((abs(value) for _, value in weights), Decimal(0))
    assert abs(gross - Decimal(1)) < Decimal("0.0000000001")
    cap = Decimal(2) / Decimal(len(weights))
    assert all(abs(value) <= cap + Decimal("0.0000000001") for _, value in weights)


def test_controls_are_present_and_shaped() -> None:
    bars = panel(50)
    histories = build_contract_histories(bars)
    decision = 99 * DAY_NS - 1_000_000
    snapshot = select_universe(histories, decision_close_ns=decision, rules=SPEC.universe)
    vectors = build_weight_vectors(histories, snapshot, spec=SPEC)
    assert vectors["no_trade"].weights == ()
    passive = vectors["passive_long_ew"].weights
    assert len(passive) == len(snapshot.contracts)
    assert all(value > 0 for _, value in passive)
    random_ranks = vectors["random_ranks"].weights
    assert sum((abs(value) for _, value in random_ranks), Decimal(0)) == Decimal(1)


def test_empty_universe_propagates_reason_codes() -> None:
    bars = panel(3)
    histories = build_contract_histories(bars)
    decision = 99 * DAY_NS - 1_000_000
    snapshot = select_universe(histories, decision_close_ns=decision, rules=SPEC.universe)
    vectors = build_weight_vectors(histories, snapshot, spec=SPEC)
    assert vectors["xs_mom_1w"].weights == ()
    assert vectors["xs_mom_1w"].reason_codes == ("UNIVERSE_TOO_SMALL",)
