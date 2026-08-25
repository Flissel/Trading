"""Deterministic baseline signals and explicit execution-cost scenarios."""

from dataclasses import dataclass
from decimal import Decimal
from random import Random


@dataclass(frozen=True, slots=True)
class CostScenario:
    name: str
    fee_bps_per_side: Decimal
    spread_multiplier: Decimal
    slippage_bps_per_side: Decimal
    funding_bps: Decimal

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("cost scenario name must not be empty")
        values = (
            self.fee_bps_per_side,
            self.spread_multiplier,
            self.slippage_bps_per_side,
            self.funding_bps,
        )
        if any(not isinstance(value, Decimal) or not value.is_finite() for value in values):
            raise ValueError("cost parameters must be finite Decimals")
        if any(value < 0 for value in values):
            raise ValueError("cost parameters must be non-negative")

    def round_trip_cost_return(self, observed_spread_bps: Decimal) -> Decimal:
        if observed_spread_bps < 0:
            raise ValueError("observed_spread_bps must be non-negative")
        total_bps = (
            self.fee_bps_per_side * 2
            + observed_spread_bps * self.spread_multiplier
            + self.slippage_bps_per_side * 2
            + self.funding_bps
        )
        return total_bps / Decimal(10_000)


def generate_baseline_signals(
    observed_returns: tuple[Decimal, ...], *, random_seed: int
) -> dict[str, tuple[int, ...]]:
    """Generate signals using only the contemporaneously observed return."""
    momentum = tuple(_sign(value) for value in observed_returns)
    random = Random(random_seed)
    return {
        "no_trade": tuple(0 for _ in observed_returns),
        "momentum": momentum,
        "mean_reversion": tuple(-signal for signal in momentum),
        "random": tuple(random.choice((-1, 0, 1)) for _ in observed_returns),
    }


def _sign(value: Decimal) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0
