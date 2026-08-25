from decimal import Decimal

import pytest

from trading_bot.candles import CandlePayloadError, parse_binance_klines, parse_okx_candles


def test_okx_candles_are_strict_and_sorted_chronologically() -> None:
    payload: dict[str, object] = {
        "code": "0",
        "msg": "",
        "data": [
            ["2000", "102", "105", "101", "104", "20", "2", "2000", "1"],
            ["1000", "100", "103", "99", "102", "10", "1", "1000", "1"],
        ],
    }

    candles = parse_okx_candles(
        payload,
        instrument_id="BTC-USDT-SWAP.OKX",
        interval_ns=1_000_000_000,
        availability_lag_ns=5,
    )

    assert [item.open_time_ns for item in candles] == [1_000_000_000, 2_000_000_000]
    assert candles[0].open == Decimal("100")
    assert candles[0].close == Decimal("102")
    assert candles[0].available_time_ns == 2_000_000_004
    assert candles[0].confirmed is True
    assert candles[0].venue == "OKX"


def test_binance_klines_preserve_close_time_and_trade_count() -> None:
    payload: list[object] = [
        [
            1000,
            "100",
            "103",
            "99",
            "102",
            "10",
            1999,
            "1000",
            42,
            "6",
            "600",
            "0",
        ]
    ]

    candle = parse_binance_klines(
        payload,
        instrument_id="BTC-USDT-SWAP.BINANCE",
        interval_ns=1_000_000_000,
        availability_lag_ns=7,
    )[0]

    assert candle.open_time_ns == 1_000_000_000
    assert candle.close_time_ns == 1_999_000_000
    assert candle.available_time_ns == 1_999_000_007
    assert candle.trade_count == 42
    assert candle.quote_volume == Decimal("1000")
    assert candle.venue == "BINANCE"


def test_binance_kline_is_unconfirmed_until_close_time_is_observable() -> None:
    payload: list[object] = [
        [1000, "100", "103", "99", "102", "10", 1999, "1000", 42, "6", "600", "0"]
    ]

    before_close = parse_binance_klines(
        payload,
        instrument_id="BTC-USDT-SWAP.BINANCE",
        interval_ns=1_000_000_000,
        availability_lag_ns=7,
        confirmed_through_ns=1_998_999_999,
    )[0]
    at_close = parse_binance_klines(
        payload,
        instrument_id="BTC-USDT-SWAP.BINANCE",
        interval_ns=1_000_000_000,
        availability_lag_ns=7,
        confirmed_through_ns=1_999_000_000,
    )[0]

    assert before_close.confirmed is False
    assert at_close.confirmed is True


def test_candle_parser_rejects_binary_float_and_impossible_ohlc() -> None:
    float_payload: dict[str, object] = {
        "code": "0",
        "msg": "",
        "data": [["1000", 100.0, "103", "99", "102", "10", "1", "1000", "1"]],
    }
    impossible_payload: list[object] = [
        [1000, "100", "101", "99", "102", "10", 1999, "1000", 1, "6", "600", "0"]
    ]

    with pytest.raises(CandlePayloadError, match="open must be a string"):
        parse_okx_candles(
            float_payload,
            instrument_id="BTC-USDT-SWAP.OKX",
            interval_ns=1_000_000_000,
            availability_lag_ns=0,
        )
    with pytest.raises(CandlePayloadError, match="OHLC range"):
        parse_binance_klines(
            impossible_payload,
            instrument_id="BTC-USDT-SWAP.BINANCE",
            interval_ns=1_000_000_000,
            availability_lag_ns=0,
        )


def test_okx_api_error_fails_closed() -> None:
    with pytest.raises(CandlePayloadError, match="OKX API error 50011"):
        parse_okx_candles(
            {"code": "50011", "msg": "rate limit", "data": []},
            instrument_id="BTC-USDT-SWAP.OKX",
            interval_ns=900_000_000_000,
            availability_lag_ns=0,
        )
