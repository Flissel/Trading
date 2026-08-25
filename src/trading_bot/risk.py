"""Pure fail-closed risk policy and persistent policy state."""

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from trading_bot.canonical import canonical_json


@dataclass(frozen=True, slots=True)
class DecisionIntent:
    intent_id: str
    instrument_id: str
    direction: int
    requested_notional: Decimal
    expected_net_return_adverse: Decimal
    maximum_planned_loss: Decimal
    expires_at_ns: int
    lineage_complete: bool
    forecast_eligible: bool
    has_exit: bool

    def __post_init__(self) -> None:
        if not self.intent_id or not self.instrument_id:
            raise ValueError("intent identity must not be empty")
        if self.direction not in (-1, 1):
            raise ValueError("direction must be -1 or 1")
        for name in (
            "requested_notional",
            "expected_net_return_adverse",
            "maximum_planned_loss",
        ):
            value = getattr(self, name)
            if not isinstance(value, Decimal) or not value.is_finite():
                raise ValueError(f"{name} must be a finite Decimal")
        if self.requested_notional <= 0 or self.maximum_planned_loss <= 0:
            raise ValueError("requested notional and planned loss must be positive")


@dataclass(frozen=True, slots=True)
class RiskHealth:
    market_data_healthy: bool
    book_valid: bool
    account_synchronized: bool
    rules_valid: bool
    audit_sink_healthy: bool
    unknown_order: bool


@dataclass(frozen=True, slots=True)
class RiskState:
    equity: Decimal
    high_water_mark: Decimal
    daily_loss: Decimal
    weekly_loss: Decimal
    cumulative_loss: Decimal
    gross_exposure: Decimal
    instrument_exposures: tuple[tuple[str, Decimal], ...]
    open_order_intents: tuple[str, ...]
    cooldown_until_ns: int | None
    hard_kill_switch: bool

    def __post_init__(self) -> None:
        values = (
            self.equity,
            self.high_water_mark,
            self.daily_loss,
            self.weekly_loss,
            self.cumulative_loss,
            self.gross_exposure,
        )
        if any(not isinstance(value, Decimal) or not value.is_finite() for value in values):
            raise ValueError("risk-state amounts must be finite Decimals")
        if any(value < 0 for value in values):
            raise ValueError("risk-state amounts must be non-negative")


@dataclass(frozen=True, slots=True)
class RiskDecision:
    intent_id: str
    requested_notional: Decimal
    approved_notional: Decimal
    reason_codes: tuple[str, ...]


class RiskPolicy:
    """Apply the frozen EUR-100 paper limits without increasing exposure."""

    maximum_gross_exposure = Decimal("50")
    maximum_instrument_exposure = Decimal("25")
    maximum_planned_loss = Decimal("0.25")
    maximum_daily_loss = Decimal("1")
    maximum_weekly_loss = Decimal("2.5")
    maximum_drawdown = Decimal("5")
    hard_cumulative_loss = Decimal("10")
    maximum_positions = 2

    def evaluate(
        self,
        intent: DecisionIntent,
        state: RiskState,
        health: RiskHealth,
        *,
        now_ns: int,
        venue_min_notional: Decimal,
        venue_notional_step: Decimal,
    ) -> RiskDecision:
        rejection = self._entry_rejection(intent, state, health, now_ns)
        if rejection is not None:
            return self._reject(intent, rejection)
        if venue_min_notional <= 0 or venue_notional_step <= 0:
            return self._reject(intent, "INVALID_VENUE_RULES")

        exposures = dict(state.instrument_exposures)
        current_instrument = abs(exposures.get(intent.instrument_id, Decimal(0)))
        if (
            current_instrument == 0
            and len([value for value in exposures.values() if value != 0]) >= 2
        ):
            return self._reject(intent, "POSITION_COUNT_LIMIT")
        loss_limited = (
            intent.requested_notional * self.maximum_planned_loss / intent.maximum_planned_loss
        )
        safe = min(
            intent.requested_notional,
            self.maximum_instrument_exposure - current_instrument,
            self.maximum_gross_exposure - state.gross_exposure,
            loss_limited,
        )
        safe = max(Decimal(0), safe)
        rounded = (safe // venue_notional_step) * venue_notional_step
        if rounded < venue_min_notional:
            return self._reject(intent, "VENUE_MINIMUM_EXCEEDS_SAFE_SIZE")
        reasons = ("SIZE_REDUCED",) if rounded < intent.requested_notional else ()
        return RiskDecision(intent.intent_id, intent.requested_notional, rounded, reasons)

    def _entry_rejection(
        self, intent: DecisionIntent, state: RiskState, health: RiskHealth, now_ns: int
    ) -> str | None:
        if state.hard_kill_switch:
            return "HARD_KILL_SWITCH"
        if state.cumulative_loss >= self.hard_cumulative_loss:
            return "CUMULATIVE_LOSS_LIMIT"
        if state.daily_loss >= self.maximum_daily_loss:
            return "DAILY_LOSS_LIMIT"
        if state.weekly_loss >= self.maximum_weekly_loss:
            return "WEEKLY_LOSS_LIMIT"
        if state.high_water_mark - state.equity >= self.maximum_drawdown:
            return "DRAWDOWN_LIMIT"
        if health.unknown_order:
            return "UNKNOWN_ORDER_OUTCOME"
        if not health.audit_sink_healthy:
            return "AUDIT_SINK_UNHEALTHY"
        if not health.account_synchronized:
            return "ACCOUNT_NOT_SYNCHRONIZED"
        if not health.market_data_healthy or not health.book_valid:
            return "MARKET_DATA_UNHEALTHY"
        if not health.rules_valid:
            return "INVALID_VENUE_RULES"
        if state.cooldown_until_ns is not None and now_ns < state.cooldown_until_ns:
            return "COOLDOWN_ACTIVE"
        if now_ns > intent.expires_at_ns:
            return "INTENT_EXPIRED"
        if not intent.lineage_complete:
            return "LINEAGE_INCOMPLETE"
        if not intent.forecast_eligible:
            return "FORECAST_INELIGIBLE"
        if not intent.has_exit:
            return "EXIT_UNDEFINED"
        if intent.expected_net_return_adverse <= 0:
            return "ADVERSE_EV_NON_POSITIVE"
        if intent.instrument_id in state.open_order_intents:
            return "OPEN_ORDER_INTENT_LIMIT"
        return None

    @staticmethod
    def _reject(intent: DecisionIntent, reason: str) -> RiskDecision:
        return RiskDecision(intent.intent_id, intent.requested_notional, Decimal(0), (reason,))


class RiskStateStore:
    """Atomically persist risk state so restart cannot reset kill switches."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def save(self, state: RiskState) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(self._path.suffix + ".tmp")
        temporary.write_bytes(canonical_json(_state_record(state)))
        temporary.replace(self._path)

    def load(self) -> RiskState:
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("risk state must be a JSON object")
        return RiskState(
            equity=Decimal(_record_string(raw, "equity")),
            high_water_mark=Decimal(_record_string(raw, "high_water_mark")),
            daily_loss=Decimal(_record_string(raw, "daily_loss")),
            weekly_loss=Decimal(_record_string(raw, "weekly_loss")),
            cumulative_loss=Decimal(_record_string(raw, "cumulative_loss")),
            gross_exposure=Decimal(_record_string(raw, "gross_exposure")),
            instrument_exposures=_record_exposures(raw),
            open_order_intents=_record_strings(raw, "open_order_intents"),
            cooldown_until_ns=_record_optional_int(raw, "cooldown_until_ns"),
            hard_kill_switch=_record_bool(raw, "hard_kill_switch"),
        )


def _state_record(state: RiskState) -> dict[str, object]:
    return {
        "equity": state.equity,
        "high_water_mark": state.high_water_mark,
        "daily_loss": state.daily_loss,
        "weekly_loss": state.weekly_loss,
        "cumulative_loss": state.cumulative_loss,
        "gross_exposure": state.gross_exposure,
        "instrument_exposures": [list(item) for item in state.instrument_exposures],
        "open_order_intents": list(state.open_order_intents),
        "cooldown_until_ns": state.cooldown_until_ns,
        "hard_kill_switch": state.hard_kill_switch,
    }


def _record_string(record: dict[object, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str):
        raise ValueError(f"risk-state {key} must be a string")
    return value


def _record_strings(record: dict[object, object], key: str) -> tuple[str, ...]:
    value = record.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"risk-state {key} must be a string array")
    return tuple(value)


def _record_exposures(record: dict[object, object]) -> tuple[tuple[str, Decimal], ...]:
    value = record.get("instrument_exposures")
    if not isinstance(value, list):
        raise ValueError("risk-state instrument_exposures must be an array")
    result: list[tuple[str, Decimal]] = []
    for item in value:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], str)
        ):
            raise ValueError("risk-state exposure entry is invalid")
        result.append((item[0], Decimal(item[1])))
    return tuple(result)


def _record_optional_int(record: dict[object, object], key: str) -> int | None:
    value = record.get(key)
    if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
        raise ValueError(f"risk-state {key} must be an integer or null")
    return value


def _record_bool(record: dict[object, object], key: str) -> bool:
    value = record.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"risk-state {key} must be a boolean")
    return value
