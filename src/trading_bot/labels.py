"""Future-outcome labels kept independent from feature generation."""

from dataclasses import dataclass, replace
from decimal import Decimal
from itertools import pairwise


@dataclass(frozen=True, slots=True)
class PricePoint:
    event_id: str
    event_time_ns: int
    available_time_ns: int
    price: Decimal

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must not be empty")
        if self.event_time_ns < 0 or self.available_time_ns < self.event_time_ns:
            raise ValueError("price-point timestamps are invalid")
        if not isinstance(self.price, Decimal) or not self.price.is_finite() or self.price <= 0:
            raise ValueError("price must be a positive finite Decimal")


@dataclass(frozen=True, slots=True)
class ReturnLabel:
    source_event_id: str
    outcome_event_id: str
    label_start_time_ns: int
    label_end_time_ns: int
    label_available_time_ns: int
    gross_return: Decimal
    net_return: Decimal
    direction: int
    overlaps_next: bool


def build_return_labels(
    points: list[PricePoint], *, horizon_ns: int, round_trip_cost_bps: Decimal
) -> list[ReturnLabel]:
    if horizon_ns <= 0:
        raise ValueError("horizon_ns must be positive")
    if round_trip_cost_bps < 0:
        raise ValueError("round_trip_cost_bps must be non-negative")
    _validate_chronology(points)
    cost_return = round_trip_cost_bps / Decimal(10_000)
    labels: list[ReturnLabel] = []

    for index, point in enumerate(points):
        target_time = point.event_time_ns + horizon_ns
        outcome = next(
            (
                candidate
                for candidate in points[index + 1 :]
                if candidate.event_time_ns >= target_time
            ),
            None,
        )
        if outcome is None:
            break
        gross_return = outcome.price / point.price - Decimal(1)
        direction = 1 if gross_return > cost_return else -1 if gross_return < -cost_return else 0
        labels.append(
            ReturnLabel(
                source_event_id=point.event_id,
                outcome_event_id=outcome.event_id,
                label_start_time_ns=point.event_time_ns,
                label_end_time_ns=outcome.event_time_ns,
                label_available_time_ns=outcome.available_time_ns,
                gross_return=gross_return,
                net_return=gross_return - cost_return,
                direction=direction,
                overlaps_next=False,
            )
        )

    for index in range(len(labels) - 1):
        if labels[index + 1].label_start_time_ns < labels[index].label_end_time_ns:
            labels[index] = replace(labels[index], overlaps_next=True)
    return labels


def _validate_chronology(points: list[PricePoint]) -> None:
    for previous, current in pairwise(points):
        if current.event_time_ns <= previous.event_time_ns:
            raise ValueError("price-point event times must be strictly increasing")
