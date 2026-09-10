"""Point-in-time eligibility and liquidity tiering for the perpetual panel."""

from dataclasses import dataclass
from decimal import Decimal

from trading_bot.panel_config import PanelUniverseRules
from trading_bot.panel_reader import PanelBar

DAY_NS = 86_400_000_000_000


@dataclass(frozen=True, slots=True)
class ContractHistory:
    contract_id: str
    instrument_id: str
    closes: dict[int, Decimal]
    quote_volumes: dict[int, Decimal]
    close_times: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class EligibleContract:
    contract_id: str
    median_quote_volume: Decimal
    liquidity_rank: int
    tier: int


@dataclass(frozen=True, slots=True)
class UniverseSnapshot:
    decision_close_ns: int
    contracts: tuple[EligibleContract, ...]
    reason_codes: tuple[str, ...]


def build_contract_histories(bars: tuple[PanelBar, ...]) -> dict[str, ContractHistory]:
    """Index bars by contract and close time for point-in-time lookups."""
    closes: dict[str, dict[int, Decimal]] = {}
    volumes: dict[str, dict[int, Decimal]] = {}
    instruments: dict[str, str] = {}
    for bar in bars:
        closes.setdefault(bar.contract_id, {})[bar.close_time_ns] = bar.close
        volumes.setdefault(bar.contract_id, {})[bar.close_time_ns] = bar.quote_volume
        instruments[bar.contract_id] = bar.instrument_id
    return {
        contract_id: ContractHistory(
            contract_id=contract_id,
            instrument_id=instruments[contract_id],
            closes=closes[contract_id],
            quote_volumes=volumes[contract_id],
            close_times=tuple(sorted(closes[contract_id])),
        )
        for contract_id in sorted(closes)
    }


def select_universe(
    histories: dict[str, ContractHistory],
    *,
    decision_close_ns: int,
    rules: PanelUniverseRules,
) -> UniverseSnapshot:
    scored: list[tuple[Decimal, str]] = []
    for contract_id in sorted(histories):
        history = histories[contract_id]
        observed = [value for value in history.close_times if value <= decision_close_ns]
        # Spec 7.1 item 1's "at least 91 daily bars" is explicitly a bar
        # count, not a calendar span -- unlike the liquidity window below,
        # this stays a plain observation count.
        if len(observed) < rules.minimum_history_days:
            continue
        if not observed or observed[-1] != decision_close_ns:
            continue
        # Spec 7.1 item 3's "trailing 30-day median" is a calendar span ending
        # at the decision, not the last `liquidity_window_days` *observations*
        # regardless of span. Bounding by calendar day (rather than slicing
        # the last N observed values) means a hole inside the window simply
        # yields fewer values for the median -- it never reaches outside the
        # window to replace them, the way an observation-count slice would on
        # a series with a gap. That reach-outside behaviour, not the median
        # having fewer points, was the bug: a five-day hole let this window
        # span 35 calendar days instead of 30 and pull in volume the window
        # was never meant to see.
        window_start_ns = decision_close_ns - (rules.liquidity_window_days - 1) * DAY_NS
        window = [value for value in observed if value >= window_start_ns]
        median = _median(tuple(history.quote_volumes[value] for value in window))
        if median < rules.minimum_median_quote_volume:
            continue
        scored.append((median, contract_id))

    scored.sort(key=lambda item: (-item[0], item[1]))
    if len(scored) < rules.minimum_contracts:
        return UniverseSnapshot(decision_close_ns, (), ("UNIVERSE_TOO_SMALL",))
    selected = scored[: rules.maximum_contracts]
    contracts = tuple(
        EligibleContract(
            contract_id=contract_id,
            median_quote_volume=median,
            liquidity_rank=index + 1,
            tier=1 if index + 1 <= rules.tier_one_rank_limit else 2,
        )
        for index, (median, contract_id) in enumerate(selected)
    )
    return UniverseSnapshot(decision_close_ns, contracts, ())


def _median(values: tuple[Decimal, ...]) -> Decimal:
    if not values:
        return Decimal(0)
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / Decimal(2)
