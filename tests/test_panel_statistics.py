from decimal import Decimal

import pytest

from trading_bot.panel_statistics import (
    PanelStatisticsError,
    annualised_sharpe,
    deflated_sharpe_ratio,
    sharpe_ratio,
)


def alternating(mean: str, deviation: str, count: int) -> tuple[Decimal, ...]:
    high = Decimal(mean) + Decimal(deviation)
    low = Decimal(mean) - Decimal(deviation)
    return tuple(high if index % 2 == 0 else low for index in range(count))


def test_sharpe_of_a_two_point_series() -> None:
    series = alternating("0.01", "0.02", 40)
    ratio = sharpe_ratio(series)
    assert abs(ratio - Decimal("0.5")) < Decimal("0.02")
    annual = annualised_sharpe(series)
    assert abs(annual - ratio * Decimal(52).sqrt()) < Decimal("0.000001")


def test_zero_variance_is_rejected() -> None:
    with pytest.raises(PanelStatisticsError):
        sharpe_ratio((Decimal("0.01"),) * 10)


def test_short_series_is_rejected() -> None:
    with pytest.raises(PanelStatisticsError):
        sharpe_ratio((Decimal("0.01"),))


def test_deflated_sharpe_is_a_probability_and_falls_with_trial_dispersion() -> None:
    series = alternating("0.01", "0.02", 200)
    observed = sharpe_ratio(series)
    tight = deflated_sharpe_ratio(
        series, trial_sharpes=(observed, observed - Decimal("0.01"), observed + Decimal("0.01"))
    )
    wide = deflated_sharpe_ratio(
        series, trial_sharpes=(observed, observed - Decimal("2"), observed + Decimal("2"))
    )
    assert Decimal(0) <= wide < tight <= Decimal(1)


def test_deflated_sharpe_needs_at_least_two_trials() -> None:
    series = alternating("0.01", "0.02", 50)
    with pytest.raises(PanelStatisticsError):
        deflated_sharpe_ratio(series, trial_sharpes=(Decimal("0.5"),))
