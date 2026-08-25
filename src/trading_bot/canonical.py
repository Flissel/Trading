"""Canonical serialization and hashing for audit records."""

import hashlib
import json
from decimal import Decimal

type CanonicalScalar = str | int | bool | Decimal | None
type CanonicalValue = CanonicalScalar | list[CanonicalValue] | dict[str, CanonicalValue]


class CanonicalizationError(ValueError):
    """Raised when a value cannot enter the canonical audit format."""


def canonical_json(value: object) -> bytes:
    """Serialize a supported value into deterministic UTF-8 JSON."""
    normalized = _normalize(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def content_sha256(value: object) -> str:
    """Return the hexadecimal SHA-256 hash of a canonical value."""
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _normalize(value: object) -> CanonicalValue:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, float):
        raise CanonicalizationError("binary float values are not canonical")
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise CanonicalizationError("canonical mappings require string keys")
        return {key: _normalize(item) for key, item in value.items()}
    raise CanonicalizationError(f"unsupported canonical value: {type(value).__name__}")
