from decimal import Decimal
from itertools import pairwise
from pathlib import Path

import pytest

from trading_bot.panel_config import PanelFoldGeometry, load_panel_family_spec
from trading_bot.panel_reader import PanelBar
from trading_bot.panel_samples import (
    PanelSamplesError,
    build_rebalance_samples,
    derive_panel_config,
    rebalance_close_times,
)

DAY_NS = 86_400_000_000_000
SPEC, _ = load_panel_family_spec(Path("configs/xs-momentum-panel-v1.json"))


def bar(day_index: int) -> PanelBar:
    open_time_ns = day_index * DAY_NS
    close_time_ns = open_time_ns + DAY_NS - 1_000_000
    return PanelBar(
        contract_id="BTCUSDT:0",
        instrument_id="BTCUSDT",
        open_time_ns=open_time_ns,
        close_time_ns=close_time_ns,
        available_time_ns=close_time_ns + 1,
        close=Decimal("100"),
        quote_volume=Decimal("1000"),
    )


def test_rebalance_dates_are_sundays() -> None:
    bars = tuple(bar(index) for index in range(21))
    times = rebalance_close_times(bars)
    assert [value // DAY_NS for value in times] == [3, 10, 17]


def test_samples_are_chronological_and_carry_the_holding_label() -> None:
    bars = tuple(bar(index) for index in range(21))
    samples = build_rebalance_samples(bars, holding_days=7)
    assert samples[0].sample_id == f"BINANCE_UM:{4 * DAY_NS - 1_000_000}:w1"
    assert samples[0].label_end_time_ns - samples[0].decision_time_ns == 7 * DAY_NS
    assert all(
        later.decision_time_ns > earlier.decision_time_ns
        for earlier, later in pairwise(samples)
    )


def test_config_places_the_holdout_at_the_end() -> None:
    bars = tuple(bar(index) for index in range(400))
    samples = build_rebalance_samples(bars, holding_days=7)
    geometry = PanelFoldGeometry(
        train_duration_ns=100 * DAY_NS,
        validation_duration_ns=20 * DAY_NS,
        test_duration_ns=40 * DAY_NS,
        step_ns=40 * DAY_NS,
        embargo_ns=14 * DAY_NS,
        holdout_duration_ns=40 * DAY_NS,
    )
    config = derive_panel_config(samples, folds=geometry)
    assert config.final_holdout_start_ns < samples[-1].decision_time_ns
    assert config.final_holdout_start_ns > samples[0].decision_time_ns


def test_short_history_is_rejected() -> None:
    samples = build_rebalance_samples(tuple(bar(index) for index in range(10)), holding_days=7)
    with pytest.raises(PanelSamplesError):
        derive_panel_config(samples, folds=SPEC.folds)
