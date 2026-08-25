import hashlib
from decimal import Decimal

import pytest

from trading_bot.canonical import CanonicalizationError, canonical_json, content_sha256


def test_serializes_decimal_without_binary_float_loss() -> None:
    payload = {
        "venue": "OKX",
        "price": Decimal("101.25"),
        "quantity": Decimal("0.500"),
    }

    encoded = canonical_json(payload)

    assert encoded == b'{"price":"101.25","quantity":"0.500","venue":"OKX"}'


def test_hash_is_stable_when_mapping_order_changes() -> None:
    first = {"venue": "OKX", "price": Decimal("101.25")}
    second = {"price": Decimal("101.25"), "venue": "OKX"}
    expected = hashlib.sha256(b'{"price":"101.25","venue":"OKX"}').hexdigest()

    assert content_sha256(first) == expected
    assert content_sha256(second) == expected


def test_rejects_binary_float_at_any_nesting_depth() -> None:
    payload = {"book": {"bids": [[Decimal("101.25"), 0.5]]}}

    with pytest.raises(CanonicalizationError, match="binary float"):
        canonical_json(payload)


def test_rejects_non_string_mapping_keys() -> None:
    payload = {1: "not canonical"}

    with pytest.raises(CanonicalizationError, match="string keys"):
        canonical_json(payload)
