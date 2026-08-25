"""Deterministic event-driven execution simulation with unknown outcomes."""

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum


class ExecutionStatus(StrEnum):
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ExecutionAssumptions:
    latency_ns: int
    fill_fraction: Decimal
    fee_bps: Decimal
    slippage_bps: Decimal
    timeout_ns: int

    def __post_init__(self) -> None:
        if self.latency_ns < 0 or self.timeout_ns < 0:
            raise ValueError("latency and timeout must be non-negative")
        if not Decimal(0) <= self.fill_fraction <= Decimal(1):
            raise ValueError("fill_fraction must be between zero and one")
        if self.fee_bps < 0 or self.slippage_bps < 0:
            raise ValueError("fees and slippage must be non-negative")


@dataclass(frozen=True, slots=True)
class OrderCommand:
    command_id: str
    instrument_id: str
    direction: int
    approved_notional: Decimal
    submitted_at_ns: int

    def __post_init__(self) -> None:
        if not self.command_id or not self.instrument_id:
            raise ValueError("order identity must not be empty")
        if self.direction not in (-1, 1):
            raise ValueError("direction must be -1 or 1")
        if self.approved_notional <= 0:
            raise ValueError("approved_notional must be positive")


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    command_id: str
    status: ExecutionStatus
    filled_notional: Decimal | None
    fill_price: Decimal | None
    fee: Decimal | None
    completed_at_ns: int | None


class ExecutionSimulator:
    def __init__(self, assumptions: ExecutionAssumptions) -> None:
        self._assumptions = assumptions
        self._results: dict[str, tuple[OrderCommand, ExecutionResult]] = {}

    def execute(
        self, command: OrderCommand, *, market_price: Decimal, observed_at_ns: int
    ) -> ExecutionResult:
        existing = self._results.get(command.command_id)
        if existing is not None:
            if existing[0] != command:
                raise ValueError("command ID was reused with different content")
            return existing[1]
        if observed_at_ns - command.submitted_at_ns > self._assumptions.timeout_ns:
            result = ExecutionResult(
                command.command_id, ExecutionStatus.UNKNOWN, None, None, None, None
            )
        else:
            if market_price <= 0:
                raise ValueError("market_price must be positive")
            filled = command.approved_notional * self._assumptions.fill_fraction
            slippage = self._assumptions.slippage_bps / Decimal(10_000)
            fill_price = market_price * (Decimal(1) + Decimal(command.direction) * slippage)
            status = (
                ExecutionStatus.FILLED
                if self._assumptions.fill_fraction == 1
                else ExecutionStatus.PARTIALLY_FILLED
            )
            result = ExecutionResult(
                command_id=command.command_id,
                status=status,
                filled_notional=filled,
                fill_price=fill_price,
                fee=filled * self._assumptions.fee_bps / Decimal(10_000),
                completed_at_ns=command.submitted_at_ns + self._assumptions.latency_ns,
            )
        self._results[command.command_id] = (command, result)
        return result
