"""Run development-only baselines over a verified candle capture."""

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import duckdb

from trading_bot.bar_research import (
    BaselineScenarioEvaluation,
    DevelopmentResearchReport,
    ResearchBar,
    build_bar_samples,
    evaluate_development_samples,
)
from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.market_capture import verify_candle_capture
from trading_bot.strategy import CostScenario


class ResearchRunError(RuntimeError):
    """Raised when a research run cannot safely consume its capture."""


@dataclass(frozen=True, slots=True)
class ResearchRunArtifact:
    output_path: Path
    capture_root_hash: str
    dataset_root_hash: str
    report_hash: str
    report: DevelopmentResearchReport


def run_capture_research(
    capture_root: Path,
    *,
    output_path: Path,
    oos_fraction: str,
    minimum_train_samples: int,
    random_seed: int,
) -> ResearchRunArtifact:
    verification = verify_candle_capture(capture_root)
    if not verification.valid:
        raise ResearchRunError("capture verification failed: " + ",".join(verification.errors))
    if output_path.exists():
        raise ResearchRunError("research report already exists and is immutable")
    capture_manifest = _load_object(capture_root / "capture-manifest.json")
    dataset_manifest = _load_object(capture_root / "dataset" / "dataset-manifest.json")
    capture_hash = _string_field(capture_manifest, "capture_root_hash")
    dataset_hash = _string_field(dataset_manifest, "root_hash")
    bars = load_research_bars(capture_root / "dataset")
    primary = tuple(item for item in bars if item.venue == "OKX")
    reference = tuple(item for item in bars if item.venue == "BINANCE")
    samples = build_bar_samples(primary, reference)
    report = evaluate_development_samples(
        samples,
        oos_fraction=Decimal(oos_fraction),
        minimum_train_samples=minimum_train_samples,
        random_seed=random_seed,
        base_costs=CostScenario("base", Decimal("1"), Decimal("1"), Decimal("1"), Decimal("0")),
        adverse_costs=CostScenario(
            "adverse", Decimal("2"), Decimal("2"), Decimal("2"), Decimal("1")
        ),
        assumed_spread_bps=Decimal("2"),
    )
    material = _report_record(report, capture_hash, dataset_hash, random_seed)
    report_hash = content_sha256(material)
    document = dict(material)
    document["report_hash"] = report_hash
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_bytes(canonical_json(document))
    temporary.replace(output_path)
    return ResearchRunArtifact(
        output_path=output_path,
        capture_root_hash=capture_hash,
        dataset_root_hash=dataset_hash,
        report_hash=report_hash,
        report=report,
    )


def verify_research_report(path: Path) -> bool:
    """Verify the semantic report hash independently of JSON formatting."""
    try:
        document = _load_object(path)
        recorded_hash = _string_field(document, "report_hash")
        material = dict(document)
        del material["report_hash"]
        return content_sha256(material) == recorded_hash
    except (OSError, json.JSONDecodeError, ResearchRunError, KeyError, ValueError, TypeError):
        return False


def load_research_bars(
    dataset_root: Path, *, available_before_ns: int | None = None
) -> tuple[ResearchBar, ...]:
    parquet_glob = (dataset_root / "**" / "*.parquet").as_posix()
    if available_before_ns is not None and available_before_ns <= 0:
        raise ResearchRunError("bar availability boundary must be positive")
    query = """
        SELECT source_payload_hash, venue, open_time_ns, available_time_ns,
               close, high, low, base_volume
        FROM read_parquet(?)
    """
    parameters: list[object] = [parquet_glob]
    if available_before_ns is not None:
        query += " WHERE available_time_ns < ?"
        parameters.append(available_before_ns)
    query += " ORDER BY venue, open_time_ns"
    rows = duckdb.sql(query, params=parameters).fetchall()
    bars: list[ResearchBar] = []
    for row in rows:
        if len(row) != 8:
            raise ResearchRunError("unexpected candle dataset schema")
        bars.append(
            ResearchBar(
                source_id=_row_string(row[0], "source_payload_hash"),
                venue=_row_string(row[1], "venue"),
                open_time_ns=_row_int(row[2], "open_time_ns"),
                available_time_ns=_row_int(row[3], "available_time_ns"),
                close=Decimal(_row_string(row[4], "close")),
                high=Decimal(_row_string(row[5], "high")),
                low=Decimal(_row_string(row[6], "low")),
                base_volume=Decimal(_row_string(row[7], "base_volume")),
            )
        )
    return tuple(bars)


def _report_record(
    report: DevelopmentResearchReport,
    capture_hash: str,
    dataset_hash: str,
    random_seed: int,
) -> dict[str, object]:
    return {
        "report_version": "1.0.0",
        "status": report.status.value,
        "reason_codes": list(report.reason_codes),
        "capture_root_hash": capture_hash,
        "dataset_root_hash": dataset_hash,
        "random_seed": random_seed,
        "total_sample_count": report.total_sample_count,
        "train_sample_count": report.train_sample_count,
        "oos_sample_count": report.oos_sample_count,
        "oos_start_time_ns": report.oos_start_time_ns,
        "cross_venue_coverage": report.cross_venue_coverage,
        "baselines": [_baseline_record(item) for item in report.baselines],
    }


def _baseline_record(item: BaselineScenarioEvaluation) -> dict[str, object]:
    return {
        "baseline_name": item.baseline_name,
        "base": _evaluation_record(item.base),
        "adverse": _evaluation_record(item.adverse),
    }


def _evaluation_record(value: object) -> dict[str, object]:
    from trading_bot.evaluation import EvaluationResult

    if not isinstance(value, EvaluationResult):
        raise TypeError("expected EvaluationResult")
    return {
        "scenario_name": value.scenario_name,
        "trade_count": value.trade_count,
        "total_net_return": value.total_net_return,
        "mean_net_return": value.mean_net_return,
        "win_rate": value.win_rate,
        "maximum_drawdown": value.maximum_drawdown,
        "net_returns": list(value.net_returns),
    }


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ResearchRunError(f"{path.name} must be a JSON object")
    return value


def _string_field(record: dict[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str):
        raise ResearchRunError(f"manifest field {key} must be a string")
    return value


def _row_string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ResearchRunError(f"dataset field {field} must be a string")
    return value


def _row_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ResearchRunError(f"dataset field {field} must be an integer")
    return value
