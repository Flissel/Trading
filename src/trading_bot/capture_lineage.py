"""Verify that an extended panel capture is a superset of an earlier one.

A holdout read for a panel family must run on a capture that opens the
holdout window while still covering everything the original, holdout-free
capture covered -- otherwise a source row the original panel read could
silently vanish, or change under it, once the read moves to the extended
capture. This module checks exactly that lineage between two captures'
`capture-manifest.json` files, independent of the rest of the capture
machinery, so a holdout read can gate on it without importing anything else.
"""

import json
from pathlib import Path

_MANIFEST_NAME = "capture-manifest.json"
_MAX_REASONS = 20


def verify_capture_superset(
    original_root: Path, extended_root: Path
) -> tuple[bool, tuple[str, ...]]:
    """True when the extended capture manifest contains every `sources` row of the original --
    same (kind, symbol, month) with identical `raw_sha256` and `status` -- and both manifests
    agree on `venue`, `interval` and `market` (absent market means "um"). Reasons name the
    first 20 mismatches as "<kind>:<symbol>:<month>: missing|hash|status". Reads
    `<root>/capture-manifest.json` only; a missing or malformed manifest is a single reason."""
    original = _read_manifest(original_root)
    if original is None:
        return False, ("original: manifest missing or unreadable",)
    extended = _read_manifest(extended_root)
    if extended is None:
        return False, ("extended: manifest missing or unreadable",)

    reasons: list[str] = []
    header_fields = (
        ("venue", original.get("venue"), extended.get("venue")),
        ("interval", original.get("interval"), extended.get("interval")),
        ("market", original.get("market", "um"), extended.get("market", "um")),
    )
    for name, original_value, extended_value in header_fields:
        if original_value != extended_value:
            reasons.append(f"{name}: mismatch")

    original_rows = _row_index(original)
    extended_rows = _row_index(extended)
    row_reasons: list[str] = []
    for key in sorted(original_rows):
        kind, symbol, month = key
        if key not in extended_rows:
            row_reasons.append(f"{kind}:{symbol}:{month}: missing")
            continue
        original_hash, original_status = original_rows[key]
        extended_hash, extended_status = extended_rows[key]
        if original_hash != extended_hash:
            row_reasons.append(f"{kind}:{symbol}:{month}: hash")
        elif original_status != extended_status:
            row_reasons.append(f"{kind}:{symbol}:{month}: status")
    reasons.extend(row_reasons[:_MAX_REASONS])
    return not reasons, tuple(reasons)


def _read_manifest(root: Path) -> dict[str, object] | None:
    try:
        document = json.loads((root / _MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(document, dict):
        return None
    return document


def _row_index(manifest: dict[str, object]) -> dict[tuple[str, str, str], tuple[object, object]]:
    """Every `sources` row keyed by `(kind, symbol, month)`, holding its
    `raw_sha256` and `status` -- the two fields lineage can differ on."""
    sources = manifest.get("sources")
    index: dict[tuple[str, str, str], tuple[object, object]] = {}
    if not isinstance(sources, list):
        return index
    for entry in sources:
        if not isinstance(entry, dict):
            continue
        kind, symbol, month = entry.get("kind"), entry.get("symbol"), entry.get("month")
        if not (isinstance(kind, str) and isinstance(symbol, str) and isinstance(month, str)):
            continue
        index[(kind, symbol, month)] = (entry.get("raw_sha256"), entry.get("status"))
    return index
