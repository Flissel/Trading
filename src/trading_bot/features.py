"""Point-in-time-safe market feature computation."""

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class MarketState:
    event_id: str
    available_time_ns: int
    decision_time_ns: int
    mid_price: Decimal
    best_bid: Decimal
    best_ask: Decimal
    bid_size: Decimal
    ask_size: Decimal
    trade_buy_quantity: Decimal
    trade_sell_quantity: Decimal
    funding_rate: Decimal | None
    book_valid: bool
    feed_age_ns: int

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must not be empty")
        if self.available_time_ns < 0 or self.decision_time_ns < 0:
            raise ValueError("timestamps must be non-negative")
        if self.feed_age_ns < 0:
            raise ValueError("feed_age_ns must be non-negative")
        for name in ("mid_price", "best_bid", "best_ask"):
            value = getattr(self, name)
            if not isinstance(value, Decimal) or not value.is_finite() or value <= 0:
                raise ValueError(f"{name} must be a positive finite Decimal")
        for name in ("bid_size", "ask_size", "trade_buy_quantity", "trade_sell_quantity"):
            value = getattr(self, name)
            if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
                raise ValueError(f"{name} must be a non-negative finite Decimal")
        if self.funding_rate is not None and (
            not isinstance(self.funding_rate, Decimal) or not self.funding_rate.is_finite()
        ):
            raise ValueError("funding_rate must be a finite Decimal or None")


@dataclass(frozen=True, slots=True)
class FeatureRow:
    decision_time_ns: int
    available_time_ns: int
    input_event_ids: tuple[str, ...]
    simple_return: Decimal | None
    realized_volatility: Decimal | None
    spread_bps: Decimal | None
    book_imbalance: Decimal | None
    trade_flow_imbalance: Decimal | None
    funding_rate: Decimal | None
    feed_age_ns: int
    feed_healthy: bool


class PointInTimeFeatureEngine:
    """Compute deterministic features without reading future observations."""

    def __init__(self, *, lookback: int, max_feed_age_ns: int) -> None:
        if lookback < 1:
            raise ValueError("lookback must be positive")
        if max_feed_age_ns < 0:
            raise ValueError("max_feed_age_ns must be non-negative")
        self._lookback = lookback
        self._max_feed_age_ns = max_feed_age_ns

    def transform(self, states: list[MarketState]) -> list[FeatureRow]:
        rows: list[FeatureRow] = []
        returns: list[Decimal] = []
        previous: MarketState | None = None

        for index, state in enumerate(states):
            if state.available_time_ns > state.decision_time_ns:
                raise ValueError(f"event {state.event_id} is available after decision")
            if previous is not None and state.decision_time_ns <= previous.decision_time_ns:
                raise ValueError("decision times must be strictly increasing")

            simple_return = None
            if previous is not None:
                simple_return = state.mid_price / previous.mid_price - Decimal(1)
                returns.append(simple_return)
            window_returns = returns[-self._lookback :]
            volatility = _root_mean_square(window_returns)
            healthy = state.book_valid and state.feed_age_ns <= self._max_feed_age_ns
            spread_bps = None
            book_imbalance = None
            if healthy:
                spread_bps = (state.best_ask - state.best_bid) / state.mid_price * Decimal(10_000)
                book_imbalance = _imbalance(state.bid_size, state.ask_size)

            start = max(0, index - self._lookback + 1)
            lineage = tuple(item.event_id for item in states[start : index + 1])
            rows.append(
                FeatureRow(
                    decision_time_ns=state.decision_time_ns,
                    available_time_ns=max(
                        item.available_time_ns for item in states[start : index + 1]
                    ),
                    input_event_ids=lineage,
                    simple_return=simple_return,
                    realized_volatility=volatility,
                    spread_bps=spread_bps,
                    book_imbalance=book_imbalance,
                    trade_flow_imbalance=_imbalance(
                        state.trade_buy_quantity, state.trade_sell_quantity
                    ),
                    funding_rate=state.funding_rate,
                    feed_age_ns=state.feed_age_ns,
                    feed_healthy=healthy,
                )
            )
            previous = state
        return rows


def _imbalance(positive: Decimal, negative: Decimal) -> Decimal | None:
    total = positive + negative
    if total == 0:
        return None
    return (positive - negative) / total


def _root_mean_square(values: list[Decimal]) -> Decimal | None:
    if not values:
        return None
    mean_square = sum((value * value for value in values), Decimal(0)) / Decimal(len(values))
    return mean_square.sqrt()
