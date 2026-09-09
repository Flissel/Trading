"""DuckDB reads over the published perpetual panel."""

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import duckdb


class PanelReaderError(RuntimeError):
    """Raised when the panel dataset cannot be read safely."""


@dataclass(frozen=True, slots=True)
class PanelBar:
    contract_id: str
    instrument_id: str
    open_time_ns: int
    close_time_ns: int
    available_time_ns: int
    close: Decimal
    quote_volume: Decimal


@dataclass(frozen=True, slots=True)
class FundingEvent:
    contract_id: str
    instrument_id: str
    calc_time_ns: int
    rate: Decimal


def load_panel_bars(
    dataset_root: Path, *, available_before_ns: int | None = None
) -> tuple[PanelBar, ...]:
    if available_before_ns is not None and available_before_ns <= 0:
        raise PanelReaderError("panel availability boundary must be positive")
    glob = (dataset_root / "dataset=daily_candles" / "**" / "*.parquet").as_posix()
    query = """
        SELECT instrument_id, open_time_ns, close_time_ns, available_time_ns,
               close, quote_volume
        FROM read_parquet(?)
    """
    parameters: list[object] = [glob]
    if available_before_ns is not None:
        query += " WHERE available_time_ns < ?"
        parameters.append(available_before_ns)
    query += " ORDER BY instrument_id, open_time_ns"
    rows = duckdb.sql(query, params=parameters).fetchall()
    first_open = _first_open_times(dataset_root)
    bars: list[PanelBar] = []
    for record in rows:
        if len(record) != 6:
            raise PanelReaderError("unexpected panel candle schema")
        instrument_id = str(record[0])
        bars.append(
            PanelBar(
                contract_id=f"{instrument_id}:{first_open[instrument_id]}",
                instrument_id=instrument_id,
                open_time_ns=int(record[1]),
                close_time_ns=int(record[2]),
                available_time_ns=int(record[3]),
                close=Decimal(str(record[4])),
                quote_volume=Decimal(str(record[5])),
            )
        )
    return tuple(bars)


def load_funding_events(dataset_root: Path) -> tuple[FundingEvent, ...]:
    glob = (dataset_root / "dataset=funding" / "**" / "*.parquet").as_posix()
    try:
        rows = duckdb.sql(
            "SELECT instrument_id, calc_time_ns, rate FROM read_parquet(?) "
            "ORDER BY instrument_id, calc_time_ns",
            params=[glob],
        ).fetchall()
    except duckdb.IOException:
        return ()
    first_open = _first_open_times(dataset_root)
    events: list[FundingEvent] = []
    for record in rows:
        if len(record) != 3:
            raise PanelReaderError("unexpected funding event schema")
        instrument_id = str(record[0])
        if instrument_id not in first_open:
            continue
        events.append(
            FundingEvent(
                contract_id=f"{instrument_id}:{first_open[instrument_id]}",
                instrument_id=instrument_id,
                calc_time_ns=int(record[1]),
                rate=Decimal(str(record[2])),
            )
        )
    return tuple(events)


def _first_open_times(dataset_root: Path) -> dict[str, int]:
    glob = (dataset_root / "dataset=daily_candles" / "**" / "*.parquet").as_posix()
    rows = duckdb.sql(
        "SELECT instrument_id, MIN(open_time_ns) FROM read_parquet(?) GROUP BY instrument_id",
        params=[glob],
    ).fetchall()
    result = {}
    for record in rows:
        if len(record) != 2:
            raise PanelReaderError("unexpected first open times schema")
        result[str(record[0])] = int(record[1])
    return result
