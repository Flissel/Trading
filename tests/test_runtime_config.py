import pytest

from trading_bot.runtime import RuntimeConfig, RuntimeMode, RuntimePolicyError


def test_defaults_to_backtest_mode() -> None:
    config = RuntimeConfig()

    assert config.mode is RuntimeMode.BACKTEST


def test_rejects_tiny_live_mode() -> None:
    with pytest.raises(RuntimePolicyError, match="not authorized"):
        RuntimeConfig(mode=RuntimeMode.TINY_LIVE)


def test_allows_paper_mode_without_enabling_live() -> None:
    config = RuntimeConfig(mode=RuntimeMode.PAPER)

    assert config.mode is RuntimeMode.PAPER
    assert config.live_execution_enabled is False
