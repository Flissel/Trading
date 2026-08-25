"""Auditable end-to-end research backtest without a live execution route."""

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.evaluation import (
    EvaluationResult,
    PromotionDecision,
    assess_shadow_promotion,
    block_bootstrap_mean_interval,
    evaluate_signals,
)
from trading_bot.execution import (
    ExecutionAssumptions,
    ExecutionResult,
    ExecutionSimulator,
    OrderCommand,
)
from trading_bot.features import FeatureRow, MarketState, PointInTimeFeatureEngine
from trading_bot.risk import DecisionIntent, RiskDecision, RiskHealth, RiskPolicy, RiskState
from trading_bot.strategy import CostScenario, generate_baseline_signals


@dataclass(frozen=True, slots=True)
class BacktestCase:
    case_id: str
    states: tuple[MarketState, ...]
    forward_returns: tuple[Decimal, ...]
    observed_spread_bps: tuple[Decimal, ...]

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must not be empty")
        if not (len(self.states) == len(self.forward_returns) == len(self.observed_spread_bps)):
            raise ValueError("backtest inputs must have equal length")
        if not self.states:
            raise ValueError("backtest case must not be empty")


@dataclass(frozen=True, slots=True)
class ExecutionTrace:
    source_event_ids: tuple[str, ...]
    intent_id: str
    command_id: str
    risk_decision: RiskDecision
    execution: ExecutionResult


@dataclass(frozen=True, slots=True)
class AuditRecord:
    index: int
    event_type: str
    entity_id: str
    previous_hash: str | None
    record_hash: str


@dataclass(frozen=True, slots=True)
class BacktestReport:
    mode: str
    live_execution_enabled: bool
    case_id: str
    input_count: int
    selected_baseline: str
    base_result: EvaluationResult
    adverse_result: EvaluationResult
    promotion: PromotionDecision
    traces: tuple[ExecutionTrace, ...]
    audit_records: tuple[AuditRecord, ...]
    audit_chain_valid: bool
    report_hash: str


class BacktestRunner:
    def __init__(
        self,
        *,
        base_costs: CostScenario,
        adverse_costs: CostScenario,
        random_seed: int,
    ) -> None:
        self._base_costs = base_costs
        self._adverse_costs = adverse_costs
        self._random_seed = random_seed

    def run(self, case: BacktestCase, *, baseline_name: str) -> BacktestReport:
        features = PointInTimeFeatureEngine(lookback=3, max_feed_age_ns=5_000_000_000).transform(
            list(case.states)
        )
        observed_returns = tuple(
            row.simple_return if row.simple_return is not None else Decimal(0) for row in features
        )
        baselines = generate_baseline_signals(observed_returns, random_seed=self._random_seed)
        if baseline_name not in baselines:
            raise ValueError(f"unknown baseline: {baseline_name}")
        signals = baselines[baseline_name]
        base = evaluate_signals(
            signals, case.forward_returns, case.observed_spread_bps, self._base_costs
        )
        adverse = evaluate_signals(
            signals, case.forward_returns, case.observed_spread_bps, self._adverse_costs
        )
        active_base = tuple(
            value for signal, value in zip(signals, base.net_returns, strict=True) if signal != 0
        )
        lower_bound = Decimal(0)
        if active_base:
            lower_bound = block_bootstrap_mean_interval(
                active_base,
                block_length=min(2, len(active_base)),
                repetitions=200,
                seed=self._random_seed,
                confidence=Decimal("0.95"),
            ).lower
        promotion = assess_shadow_promotion(
            base=base,
            adverse=adverse,
            base_lower_confidence_bound=lower_bound,
            minimum_episodes=200,
        )

        audit: list[AuditRecord] = []
        _append_audit(audit, "backtest_started", case.case_id)
        traces = self._simulate_decisions(case, features, signals, audit)
        _append_audit(audit, "backtest_completed", case.case_id)
        audit_records = tuple(audit)
        material = _report_record(
            case.case_id,
            len(case.states),
            baseline_name,
            base,
            adverse,
            promotion,
            traces,
            audit_records,
        )
        return BacktestReport(
            mode="backtest",
            live_execution_enabled=False,
            case_id=case.case_id,
            input_count=len(case.states),
            selected_baseline=baseline_name,
            base_result=base,
            adverse_result=adverse,
            promotion=promotion,
            traces=traces,
            audit_records=audit_records,
            audit_chain_valid=verify_audit_chain(audit_records),
            report_hash=content_sha256(material),
        )

    def _simulate_decisions(
        self,
        case: BacktestCase,
        features: list[FeatureRow],
        signals: tuple[int, ...],
        audit: list[AuditRecord],
    ) -> tuple[ExecutionTrace, ...]:
        risk = RiskPolicy()
        state = _clean_risk_state()
        health = RiskHealth(True, True, True, True, True, False)
        simulator = ExecutionSimulator(
            ExecutionAssumptions(
                latency_ns=1,
                fill_fraction=Decimal(1),
                fee_bps=self._base_costs.fee_bps_per_side,
                slippage_bps=self._base_costs.slippage_bps_per_side,
                timeout_ns=100,
            )
        )
        traces: list[ExecutionTrace] = []
        for index, (row, signal, spread) in enumerate(
            zip(features, signals, case.observed_spread_bps, strict=True)
        ):
            if signal == 0:
                continue
            intent_id = f"{case.case_id}:intent:{index}"
            expected = abs(
                row.simple_return or Decimal(0)
            ) - self._adverse_costs.round_trip_cost_return(spread)
            intent = DecisionIntent(
                intent_id=intent_id,
                instrument_id="BTC-USDT-SWAP.RESEARCH",
                direction=signal,
                requested_notional=Decimal(10),
                expected_net_return_adverse=expected,
                maximum_planned_loss=Decimal("0.20"),
                expires_at_ns=row.decision_time_ns + 10,
                lineage_complete=bool(row.input_event_ids),
                forecast_eligible=True,
                has_exit=True,
            )
            decision = risk.evaluate(
                intent,
                state,
                health,
                now_ns=row.decision_time_ns,
                venue_min_notional=Decimal(5),
                venue_notional_step=Decimal(1),
            )
            _append_audit(audit, "risk_decision", intent_id)
            if decision.approved_notional == 0:
                continue
            command_id = f"{case.case_id}:command:{index}"
            command = OrderCommand(
                command_id,
                intent.instrument_id,
                signal,
                decision.approved_notional,
                row.decision_time_ns,
            )
            execution = simulator.execute(
                command,
                market_price=case.states[index].mid_price,
                observed_at_ns=row.decision_time_ns + 1,
            )
            _append_audit(audit, "execution_result", command_id)
            traces.append(
                ExecutionTrace(
                    source_event_ids=row.input_event_ids,
                    intent_id=intent_id,
                    command_id=command_id,
                    risk_decision=decision,
                    execution=execution,
                )
            )
        return tuple(traces)


def verify_audit_chain(records: tuple[AuditRecord, ...]) -> bool:
    previous: str | None = None
    for index, record in enumerate(records):
        if record.index != index or record.previous_hash != previous:
            return False
        expected = _audit_hash(index, record.event_type, record.entity_id, previous)
        if record.record_hash != expected:
            return False
        previous = record.record_hash
    return True


def write_report(path: Path, report: BacktestReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json(_full_report_record(report)))
    temporary.replace(path)


def _append_audit(records: list[AuditRecord], event_type: str, entity_id: str) -> None:
    previous = records[-1].record_hash if records else None
    index = len(records)
    records.append(
        AuditRecord(
            index=index,
            event_type=event_type,
            entity_id=entity_id,
            previous_hash=previous,
            record_hash=_audit_hash(index, event_type, entity_id, previous),
        )
    )


def _audit_hash(index: int, event_type: str, entity_id: str, previous: str | None) -> str:
    return content_sha256(
        {
            "index": index,
            "event_type": event_type,
            "entity_id": entity_id,
            "previous_hash": previous,
        }
    )


def _clean_risk_state() -> RiskState:
    return RiskState(
        equity=Decimal(100),
        high_water_mark=Decimal(100),
        daily_loss=Decimal(0),
        weekly_loss=Decimal(0),
        cumulative_loss=Decimal(0),
        gross_exposure=Decimal(0),
        instrument_exposures=(),
        open_order_intents=(),
        cooldown_until_ns=None,
        hard_kill_switch=False,
    )


def _evaluation_record(result: EvaluationResult) -> dict[str, object]:
    return {
        "scenario_name": result.scenario_name,
        "net_returns": list(result.net_returns),
        "trade_count": result.trade_count,
        "total_net_return": result.total_net_return,
        "mean_net_return": result.mean_net_return,
        "win_rate": result.win_rate,
        "maximum_drawdown": result.maximum_drawdown,
    }


def _promotion_record(decision: PromotionDecision) -> dict[str, object]:
    return {"status": decision.status.value, "reason_codes": list(decision.reason_codes)}


def _trace_record(trace: ExecutionTrace) -> dict[str, object]:
    return {
        "source_event_ids": list(trace.source_event_ids),
        "intent_id": trace.intent_id,
        "command_id": trace.command_id,
        "approved_notional": trace.risk_decision.approved_notional,
        "risk_reason_codes": list(trace.risk_decision.reason_codes),
        "execution_status": trace.execution.status.value,
        "filled_notional": trace.execution.filled_notional,
        "fill_price": trace.execution.fill_price,
        "fee": trace.execution.fee,
    }


def _audit_record(record: AuditRecord) -> dict[str, object]:
    return {
        "index": record.index,
        "event_type": record.event_type,
        "entity_id": record.entity_id,
        "previous_hash": record.previous_hash,
        "record_hash": record.record_hash,
    }


def _report_record(
    case_id: str,
    input_count: int,
    baseline_name: str,
    base: EvaluationResult,
    adverse: EvaluationResult,
    promotion: PromotionDecision,
    traces: tuple[ExecutionTrace, ...],
    audit: tuple[AuditRecord, ...],
) -> dict[str, object]:
    return {
        "mode": "backtest",
        "live_execution_enabled": False,
        "case_id": case_id,
        "input_count": input_count,
        "selected_baseline": baseline_name,
        "base_result": _evaluation_record(base),
        "adverse_result": _evaluation_record(adverse),
        "promotion": _promotion_record(promotion),
        "traces": [_trace_record(trace) for trace in traces],
        "audit_records": [_audit_record(record) for record in audit],
        "audit_chain_valid": verify_audit_chain(audit),
    }


def _full_report_record(report: BacktestReport) -> dict[str, object]:
    record = _report_record(
        report.case_id,
        report.input_count,
        report.selected_baseline,
        report.base_result,
        report.adverse_result,
        report.promotion,
        report.traces,
        report.audit_records,
    )
    record["report_hash"] = report.report_hash
    return record
