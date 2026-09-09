from decimal import Decimal
from pathlib import Path

import pytest

from trading_bot.panel_dataset import PanelCandleRow, PanelFundingRow, publish_panel_dataset
from trading_bot.panel_reader import PanelReaderError, load_funding_events, load_panel_bars

DAY_NS = 86_400_000_000_000


def row(symbol: str, index: int, *, volume: str = "1000") -> PanelCandleRow:
    open_time_ns = index * DAY_NS
    return PanelCandleRow(
        venue="BINANCE_UM",
        instrument_id=symbol,
        open_time_ns=open_time_ns,
        close_time_ns=open_time_ns + DAY_NS - 1_000_000,
        available_time_ns=open_time_ns + DAY_NS,
        interval_ns=DAY_NS,
        open=Decimal("100"),
        high=Decimal("101"),
        low=Decimal("99"),
        close=Decimal(100 + index),
        base_volume=Decimal("10"),
        quote_volume=Decimal(volume),
        trade_count=5,
        source_payload_hash="b" * 64,
    )


def dataset(tmp_path: Path) -> Path:
    artifact = publish_panel_dataset(
        tuple(row("BTCUSDT", index) for index in range(3))
        + tuple(row("ETHUSDT", index) for index in range(1, 3)),
        (
            PanelFundingRow(
                venue="BINANCE_UM",
                instrument_id="BTCUSDT",
                calc_time_ns=DAY_NS,
                funding_interval_hours=8,
                rate=Decimal("0.0001"),
            ),
        ),
        output_directory=tmp_path / "dataset",
        raw_source_hashes=("a" * 64,),
    )
    return artifact.dataset_root


def test_contract_id_is_bound_to_the_first_bar(tmp_path: Path) -> None:
    bars = load_panel_bars(dataset(tmp_path))
    btc = [bar for bar in bars if bar.instrument_id == "BTCUSDT"]
    eth = [bar for bar in bars if bar.instrument_id == "ETHUSDT"]
    assert {bar.contract_id for bar in btc} == {"BTCUSDT:0"}
    assert {bar.contract_id for bar in eth} == {f"ETHUSDT:{DAY_NS}"}
    assert [bar.open_time_ns for bar in btc] == [0, DAY_NS, 2 * DAY_NS]


def test_availability_boundary_is_strict(tmp_path: Path) -> None:
    root = dataset(tmp_path)
    included = load_panel_bars(root, available_before_ns=2 * DAY_NS + 1)
    assert max(bar.open_time_ns for bar in included) == DAY_NS
    excluded = load_panel_bars(root, available_before_ns=2 * DAY_NS)
    assert max(bar.open_time_ns for bar in excluded) == 0


def test_funding_events_carry_contract_ids(tmp_path: Path) -> None:
    events = load_funding_events(dataset(tmp_path))
    assert len(events) == 1
    assert events[0].contract_id == "BTCUSDT:0"
    assert events[0].rate == Decimal("0.0001")


def test_invalid_boundary_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(PanelReaderError):
        load_panel_bars(dataset(tmp_path), available_before_ns=0)
