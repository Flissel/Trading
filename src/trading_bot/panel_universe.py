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
        # regardless of span, and the window must be complete -- every day
        # present, no hole -- mirroring `_annualised_volatility`
        # (panel_signals.py). A median computed over whichever days survive a
        # hole is not a safe substitute for the missing ones: days go missing
        # non-randomly, concentrated on halted, dormant, and
        # delisting-adjacent contracts, which is exactly where the removed
        # days are the low-volume ones -- so a partial median is biased
        # upward on precisely the contracts a one-sided liquidity floor
        # exists to exclude. A five-day hole could once let this window reach
        # back to span 35 calendar days for a full count of observations;
        # requiring completeness instead means the contract simply drops out
        # of the eligible universe at this decision, the same way an
        # incomplete volatility window drops a contract from the time-series
        # members.
        expected_close_times = tuple(
            decision_close_ns - day_offset * DAY_NS
            for day_offset in range(rules.liquidity_window_days - 1, -1, -1)
        )
        if any(value not in history.quote_volumes for value in expected_close_times):
            continue
        median = _median(tuple(history.quote_volumes[value] for value in expected_close_times))
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
