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


def series_with_volume(symbol: str, closes: list[str], volume: str) -> list[PanelBar]:
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
                quote_volume=Decimal(volume),
            )
        )
    return output


def panel_with_varied_volume(count: int, days: int = 100) -> tuple[PanelBar, ...]:
    """Same trend structure as `panel`, but liquidity strictly increases with the
    contract index, so liquidity-rank order (descending volume) is the exact
    reverse of ascending contract-id order rather than coinciding with it."""
    rows: list[PanelBar] = []
    for index in range(count):
        symbol = f"C{index:03d}USDT"
        drift = Decimal(index) / Decimal(1000)
        closes = [str(Decimal(100) * (Decimal(1) + drift) ** day) for day in range(days)]
        volume = str(5_000_000 + (index + 1) * 100_000)
        rows.extend(series_with_volume(symbol, closes, volume))
    return tuple(rows)


def oscillating_series(symbol: str, days: int, drift: Decimal, amplitude: Decimal) -> list[str]:
    """A strong shared trend with a day-parity square-wave wobble layered on top,
    so realised volatility is driven by `amplitude` rather than by the trend."""
    closes: list[str] = []
    for day in range(days):
        trend = Decimal(100) * (Decimal(1) + drift) ** day
        wobble = amplitude if day % 2 == 0 else -amplitude
        closes.append(str(trend * (Decimal(1) + wobble)))
    return closes


def sparse_time_series_panel(days: int = 100) -> tuple[PanelBar, ...]:
    """40 eligible contracts (the config's `minimum_contracts`), of which only 5
    carry a non-zero trailing return and so survive into the time-series raw set:
    35 are perfectly flat (zero trailing return, excluded by design) and 5 trend
    together but at wildly different amplitudes. One of those 5 is near-flat
    (amplitude 0.006, sigma barely above the volatility floor, so its inverse-vol
    raw weight dominates); the other 4 share one much larger amplitude (0.45, far
    higher realised volatility, far smaller raw weight). Normalised to unit gross
    over only 5 names, the dominant contract's share is far above `cap = 2/5`,
    so capping is not just reached but reached hard — this is what actually
    exercises the water-filling redistribution rather than approaching the cap
    asymptotically the way a wide, evenly-spread universe would."""
    rows: list[PanelBar] = []
    for index in range(35):
        symbol = f"F{index:03d}USDT"
        closes = [str(Decimal(100)) for _ in range(days)]
        rows.extend(series(symbol, closes))
    drift = Decimal("0.01")
    amplitudes = [
        Decimal("0.006"),
        Decimal("0.45"),
        Decimal("0.45"),
        Decimal("0.45"),
        Decimal("0.45"),
    ]
    for index, amplitude in enumerate(amplitudes):
        symbol = f"M{index:03d}USDT"
        closes = oscillating_series(symbol, days, drift, amplitude)
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
    bars = sparse_time_series_panel()
    histories = build_contract_histories(bars)
    decision = 99 * DAY_NS - 1_000_000
    snapshot = select_universe(histories, decision_close_ns=decision, rules=SPEC.universe)
    assert len(snapshot.contracts) == 40
    vectors = build_weight_vectors(histories, snapshot, spec=SPEC)
    weights = vectors["ts_mom_12w"].weights
    # Only the 5 trending contracts carry a non-zero trailing return; the 35 flat
    # ones are excluded by design, so the cap is computed from 5, not 40.
    assert len(weights) == 5
    gross = sum((abs(value) for _, value in weights), Decimal(0))
    assert abs(gross - Decimal(1)) < Decimal("0.0000000001")
    cap = Decimal(2) / Decimal(len(weights))
    assert all(abs(value) <= cap + Decimal("0.0000000001") for _, value in weights)
    # The near-flat contract's raw (inverse-vol) weight is a huge outlier before
    # capping; at least one weight must land exactly on the cap, proving the
    # water-filling redistribution actually ran rather than trivially passing an
    # unreached bound.
    assert any(abs(abs(value) - cap) < Decimal("0.0000000001") for _, value in weights)


def test_controls_are_present_and_shaped() -> None:
    # Liquidity strictly increases with contract index here, so liquidity-rank
    # order (the order `eligible` is derived from) is the exact reverse of
    # ascending contract-id order. A `passive_long_ew` built without its own
    # sort would come back in liquidity-rank order and fail the assertion below.
    bars = panel_with_varied_volume(50)
    histories = build_contract_histories(bars)
    decision = 99 * DAY_NS - 1_000_000
    snapshot = select_universe(histories, decision_close_ns=decision, rules=SPEC.universe)
    vectors = build_weight_vectors(histories, snapshot, spec=SPEC)
    assert vectors["no_trade"].weights == ()
    passive = vectors["passive_long_ew"].weights
    assert len(passive) == len(snapshot.contracts)
    assert all(value > 0 for _, value in passive)
    passive_ids = [item[0] for item in passive]
    assert passive_ids == sorted(passive_ids)
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
