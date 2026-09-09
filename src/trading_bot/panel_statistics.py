"""Sharpe ratios and the multiple-testing-aware deflated Sharpe ratio."""

import math
from decimal import Decimal
from statistics import NormalDist

_EULER_MASCHERONI = 0.5772156649015329


class PanelStatisticsError(ValueError):
    """Raised when a statistic is not defined for the given series."""


def sharpe_ratio(net_returns: tuple[Decimal, ...]) -> Decimal:
    """Return the per-period Sharpe ratio with a zero risk-free rate."""
    mean, deviation = _mean_and_deviation(net_returns)
    return mean / deviation


def annualised_sharpe(net_returns: tuple[Decimal, ...], *, periods_per_year: int = 52) -> Decimal:
    """Return the Sharpe ratio scaled to an annual horizon by sqrt(periods_per_year)."""
    if periods_per_year < 1:
        raise PanelStatisticsError("periods per year must be positive")
    return sharpe_ratio(net_returns) * Decimal(periods_per_year).sqrt()


def deflated_sharpe_ratio(
    net_returns: tuple[Decimal, ...], *, trial_sharpes: tuple[Decimal, ...]
) -> Decimal:
    """Return the probability that the observed Sharpe exceeds the trial maximum.

    Bailey and Lopez de Prado (2014). Floats are used only inside this function
    because the normal distribution has no exact Decimal form; the result is
    reported evidence and never a gate input.
    """
    if len(trial_sharpes) < 2:
        raise PanelStatisticsError("the deflated Sharpe ratio needs at least two trials")
    observations = len(net_returns)
    if observations < 3:
        raise PanelStatisticsError("the deflated Sharpe ratio needs at least three observations")
    _, _ = _mean_and_deviation(net_returns)

    values = [float(value) for value in net_returns]
    mean = sum(values) / observations
    variance = sum((value - mean) ** 2 for value in values) / (observations - 1)
    deviation = math.sqrt(variance)
    skewness = sum(((value - mean) / deviation) ** 3 for value in values) / observations
    kurtosis = sum(((value - mean) / deviation) ** 4 for value in values) / observations
    observed = mean / deviation

    trials = [float(value) for value in trial_sharpes]
    trial_mean = sum(trials) / len(trials)
    trial_variance = sum((value - trial_mean) ** 2 for value in trials) / (len(trials) - 1)
    trial_deviation = math.sqrt(trial_variance)

    normal = NormalDist()
    count = len(trials)
    expected_maximum = trial_deviation * (
        (1.0 - _EULER_MASCHERONI) * normal.inv_cdf(1.0 - 1.0 / count)
        + _EULER_MASCHERONI * normal.inv_cdf(1.0 - 1.0 / (count * math.e))
    )
    denominator = 1.0 - skewness * observed + (kurtosis - 1.0) / 4.0 * observed**2
    if denominator <= 0.0:
        raise PanelStatisticsError("the deflated Sharpe denominator is not positive")
    statistic = (
        (observed - expected_maximum) * math.sqrt(observations - 1) / math.sqrt(denominator)
    )
    return Decimal(str(normal.cdf(statistic)))


def _mean_and_deviation(net_returns: tuple[Decimal, ...]) -> tuple[Decimal, Decimal]:
    count = len(net_returns)
    if count < 2:
        raise PanelStatisticsError("at least two observations are required")
    mean = sum(net_returns, Decimal(0)) / Decimal(count)
    variance = sum(((value - mean) ** 2 for value in net_returns), Decimal(0)) / Decimal(count - 1)
    if variance <= 0:
        raise PanelStatisticsError("a zero-variance series has no Sharpe ratio")
    return mean, variance.sqrt()
