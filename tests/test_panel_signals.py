import random
from decimal import Decimal
from pathlib import Path

from trading_bot.panel_config import load_panel_family_spec
from trading_bot.panel_reader import PanelBar
from trading_bot.panel_signals import (
    _annualised_volatility,
    _normalise,
    _water_fill,
    build_weight_vectors,
)
from trading_bot.panel_universe import ContractHistory, build_contract_histories, select_universe

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


def noisy_series(days: int, drift: Decimal, amplitude: Decimal, seed: int) -> list[str]:
    """A trend with genuinely aperiodic, deterministic day-to-day noise -- unlike
    `oscillating_series`'s day-parity square wave, whose period-2 symmetry makes its
    realised variance invariant to a one-day window shift (any 31-day window still
    contains exactly the same alternating +amplitude/-amplitude pattern), which would
    make a volatility-window test built on it pass even when the window boundary
    leaks a day it should not see."""
    generator = random.Random(seed)
    closes: list[str] = []
    for day in range(days):
        trend = Decimal(100) * (Decimal(1) + drift) ** day
        wobble = Decimal(str(generator.uniform(-1, 1))) * amplitude
        closes.append(str(trend * (Decimal(1) + wobble)))
    return closes


def volatility_window_sensitive_panel(days: int = 100) -> tuple[PanelBar, ...]:
    """40 eligible contracts (the config's `minimum_contracts`): 35 perfectly flat
    (zero trailing return, excluded by design, same as `sparse_time_series_panel`)
    and 5 carrying genuinely aperiodic noise, so that `_annualised_volatility`'s
    30-day window actually produces a different sigma when shifted by one day."""
    rows: list[PanelBar] = []
    for index in range(35):
        symbol = f"F{index:03d}USDT"
        closes = [str(Decimal(100)) for _ in range(days)]
        rows.extend(series(symbol, closes))
    for index in range(5):
        symbol = f"V{index:03d}USDT"
        closes = noisy_series(days, Decimal("0.01"), Decimal("0.05"), seed=1000 + index)
        rows.extend(series(symbol, closes))
    return tuple(rows)


def water_fill_cases() -> list[dict[str, Decimal]]:
    """A deterministic battery of raw weight vectors for the capping routine,
    covering single-outlier, staggered, and many-simultaneous-violator shapes
    across varying contract counts, each duplicated with alternating signs
    layered on. No randomness: every value is a fixed arithmetic formula, so
    the case set is identical on every run."""
    cases: list[dict[str, Decimal]] = []

    # A single dominant outlier among otherwise-equal unit weights.
    for n in (1, 2, 3, 5, 8, 13, 21, 34):
        raw = {f"O{i:02d}": Decimal(1) for i in range(n)}
        raw["O00"] = Decimal(50)
        cases.append(raw)

    # Strictly decaying magnitudes (each successive round freezes exactly one
    # more entry, forcing repeated re-scaling of whatever remains unfrozen).
    for n in (3, 5, 8, 13, 21):
        raw = {f"S{i:02d}": Decimal(2) ** (n - i) for i in range(n)}
        cases.append(raw)

    # More than one violator at once: a quarter of the contracts share one
    # large magnitude, the rest one small magnitude. (A 50/50 split can never
    # produce a violator here: cap = 2/n means half the population sitting
    # exactly at the cap already exhausts the entire unit gross, so the large
    # half's per-entry share always lands strictly below cap regardless of the
    # magnitude ratio — confirmed numerically before picking the 1-in-4 split.)
    for n in (4, 8, 12, 20, 40):
        large_count = n // 4
        raw = {f"M{i:02d}": (Decimal(100) if i < large_count else Decimal(1)) for i in range(n)}
        cases.append(raw)

    # Every contract already equal (no violator should ever appear).
    for n in (1, 4, 9, 16):
        raw = {f"E{i:02d}": Decimal(1) for i in range(n)}
        cases.append(raw)

    # Every pattern above again, with alternating signs layered on.
    for raw in list(cases):
        signed = {
            key: (value if index % 2 == 0 else -value)
            for index, (key, value) in enumerate(raw.items())
        }
        cases.append(signed)

    return cases


def test_water_fill_satisfies_cap_gross_and_sign_invariants() -> None:
    """Property test for the capping routine in isolation, independent of any
    price-series fixture: for every case in `water_fill_cases()`, water-filling
    the normalised raw weights to `cap = 2/n` must always produce a result whose
    absolute values sum to exactly one, none of which exceeds the cap, and each
    of which keeps the sign of its raw input."""
    for raw in water_fill_cases():
        cap = Decimal(2) / Decimal(len(raw))
        positive_inputs = {key for key, value in raw.items() if value >= 0}
        result = _water_fill(_normalise(raw), cap)
        gross = sum((abs(value) for value in result.values()), Decimal(0))
        assert abs(gross - Decimal(1)) < Decimal("0.0000000001"), raw
        assert all(abs(value) <= cap + Decimal("0.0000000001") for value in result.values()), raw
        assert all((value >= 0) == (key in positive_inputs) for key, value in result.items()), raw


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


def test_weight_vectors_are_unaffected_by_a_bar_one_day_after_the_decision() -> None:
    """Point-in-time regression test: nothing computed as of `decision` may depend on
    whether a bar closing after it happens to be present in the history. Uses
    `volatility_window_sensitive_panel()` rather than the smooth-compounding `panel()`
    fixture, or the day-parity `sparse_time_series_panel()`, deliberately: both of
    those have realised variance that is invariant to a one-day window shift (zero
    variance throughout, respectively perfect period-2 symmetry), so neither can
    exercise `_annualised_volatility`'s window boundary at all -- the aperiodic noise
    here gives genuine day-to-day variance, so a shifted volatility window actually
    changes sigma. The fixture carries 100 days per contract (index 0..99); the
    decision below lands on day 98's close, so day 99's bar closes exactly one day
    later. Comparing the full fixture against the same fixture with that one trailing
    bar removed isolates exactly the leakage this guards against: `_trailing_returns`
    reading `decision_close_ns + DAY_NS` instead of `decision_close_ns`, or
    `_annualised_volatility`'s window filter admitting a bar one day past the
    boundary."""
    decision = 99 * DAY_NS - 1_000_000
    with_future_bar = volatility_window_sensitive_panel()
    without_future_bar = tuple(bar for bar in with_future_bar if bar.close_time_ns <= decision)
    # Sanity: the fixture actually has a bar strictly after `decision` to drop.
    assert any(bar.close_time_ns > decision for bar in with_future_bar)
    assert all(bar.close_time_ns <= decision for bar in without_future_bar)

    histories_with = build_contract_histories(with_future_bar)
    histories_without = build_contract_histories(without_future_bar)
    snapshot_with = select_universe(histories_with, decision_close_ns=decision, rules=SPEC.universe)
    snapshot_without = select_universe(
        histories_without, decision_close_ns=decision, rules=SPEC.universe
    )
    assert snapshot_with == snapshot_without

    vectors_with = build_weight_vectors(histories_with, snapshot_with, spec=SPEC)
    vectors_without = build_weight_vectors(histories_without, snapshot_without, spec=SPEC)
    assert vectors_with == vectors_without


def test_annualised_volatility_requires_a_complete_calendar_window() -> None:
    """Spec 8.1's `sigma_30d` is "the trailing 30-day standard deviation of daily
    log returns" -- a calendar span, not a count of whatever bars happen to be on
    hand. Before this fix, `_annualised_volatility` took the last `window_days + 1`
    *observations* regardless of the calendar span they covered: with day 3 of
    0..6 missing, six observations (0, 1, 2, 4, 5, 6) still satisfy a `window_days
    = 5` count, so the old code would reach back to day 0 and treat the two-day
    log return spanning the hole as an ordinary one-day move -- computing a number
    instead of recognising the window is incomplete. The fix must return `None`."""
    decision = 6 * DAY_NS
    closes = {day * DAY_NS: Decimal(100) + Decimal(day) for day in (0, 1, 2, 4, 5, 6)}
    history = ContractHistory(
        contract_id="GAPUSDT:0",
        instrument_id="GAPUSDT",
        closes=closes,
        quote_volumes={},
        close_times=tuple(sorted(closes)),
    )
    assert _annualised_volatility(history, decision, window_days=5) is None


def test_annualised_volatility_accepts_a_genuinely_complete_calendar_window() -> None:
    """Sanity counterpart to the hole test above: with every day of the same span
    present, the calendar-aware check must still produce a value rather than
    turning into an unconditional `None`."""
    decision = 6 * DAY_NS
    closes = {day * DAY_NS: Decimal(100) + Decimal(day) for day in range(7)}
    history = ContractHistory(
        contract_id="FULLUSDT:0",
        instrument_id="FULLUSDT",
        closes=closes,
        quote_volumes={},
        close_times=tuple(sorted(closes)),
    )
    assert _annualised_volatility(history, decision, window_days=5) is not None


def test_time_series_weights_exclude_a_contract_with_a_hole_in_its_volatility_window() -> None:
    """End-to-end regression through `build_weight_vectors`, not just the helper in
    isolation: a contract that is otherwise eligible and trending, but is missing
    one bar inside the 30-day span ending at the decision, must drop out of every
    time-series member at that decision. This exercises the mechanics of the
    real-world instance that motivated the fix -- a calendar hole the daily
    dumps never fill either -- reproduced from first principles rather than
    from a fixture file. It does not reproduce that instance's own price
    action: ICPUSDT was a dead contract by the time of its hole (flat price,
    zero `quote_volume` for the preceding 104 days), which is a separate,
    additional reason it never reaches this code path at all -- see
    `panel_universe.py`'s liquidity filter."""
    decision = 99 * DAY_NS - 1_000_000
    bars = volatility_window_sensitive_panel()
    gap_index = 90  # well inside the 30-day window ending at day 98 (indices 68..98)
    gap_close_time_ns = (gap_index + 1) * DAY_NS - 1_000_000
    bars_with_gap = tuple(
        item
        for item in bars
        if not (item.instrument_id == "V000USDT" and item.close_time_ns == gap_close_time_ns)
    )
    assert len(bars_with_gap) == len(bars) - 1

    histories = build_contract_histories(bars_with_gap)
    snapshot = select_universe(histories, decision_close_ns=decision, rules=SPEC.universe)
    assert len(snapshot.contracts) == 40
    assert "V000USDT:0" in {item.contract_id for item in snapshot.contracts}

    vectors = build_weight_vectors(histories, snapshot, spec=SPEC)
    for member_name in ("ts_mom_4w", "ts_mom_12w"):
        weight_ids = {contract_id for contract_id, _ in vectors[member_name].weights}
        assert "V000USDT:0" not in weight_ids
        assert any(f"V{index:03d}USDT:0" in weight_ids for index in range(1, 5))
