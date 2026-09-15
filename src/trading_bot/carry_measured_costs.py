"""Declare a carry family from the Binance cost journal's finalisation receipt.

Section 5 of ``docs/superpowers/specs/2026-09-12-binance-cost-journal-design.md``
fixed, before any number existed, the only admissible reading of a receipt: a
v3 family sets ``slippage_bps_per_side_tier_<t>`` in its base table to
``tier_p50_of_p50`` of the *worse leg* at 5 000 USDT and in its adverse table to
``tier_p50_of_p90`` at 50 000 USDT, each rounded up to the next whole basis
point, and cites the receipt hash.

This module performs that reading and nothing else. Everything the receipt does
not speak to is carried over from the v2 declaration unchanged, so the only
difference between the assumed family and the measured one is the two cost
tables and the evidence they came from - which is what makes the two
comparable. The receipt is verified against its own hash first: a declaration
cites a receipt by hash, so bytes that do not recompute to it are not that
receipt.
"""

import json
from decimal import ROUND_CEILING, Decimal
from pathlib import Path

from pydantic import ValidationError

from trading_bot.binance_cost_journal import FinalizationReceipt
from trading_bot.canonical import CanonicalizationError, content_sha256
from trading_bot.carry_config import MEASURED_COST_FAMILY, CarryFamilySpec
from trading_bot.panel_config import load_family_spec

BASE_FAMILY_NAME = "funding_carry_panel_v2"
# Spec 5's two notionals, as the receipt keys them.
BASE_NOTIONAL = "5000"
ADVERSE_NOTIONAL = "50000"
BASE_QUANTILE = "p50_of_p50"
ADVERSE_QUANTILE = "p50_of_p90"
# The worse leg is the worse of the two markets the journal samples.
DECLARED_MARKETS: tuple[str, ...] = ("spot", "um")
DECLARED_TIERS: tuple[int, ...] = (1, 2)
TIER_FIELDS: dict[int, str] = {
    1: "slippage_bps_per_side_tier_one",
    2: "slippage_bps_per_side_tier_two",
}
# The one sentence the measured family adds to the hypothesis it inherits.
HYPOTHESIS_SENTENCE = (
    " Execution costs are the slippage tiers measured by the Binance cost journal "
    "(receipt {receipt_hash}) under the rule fixed in its design before any number existed."
)


class MeasuredCostError(RuntimeError):
    """Raised when a receipt cannot be read the one way the spec allows."""


def measured_slippage_tiers(
    receipt: dict[str, object],
) -> tuple[dict[int, Decimal], dict[int, Decimal]]:
    """The base and adverse slippage tiers spec 5 reads off one receipt.

    Returns ``(base_by_tier, adverse_by_tier)`` in whole basis points: for each
    declared tier the base value is the worse of the two legs' ``p50_of_p50``
    at 5 000 USDT and the adverse value the worse of their ``p50_of_p90`` at
    50 000 USDT, each rounded up. A receipt that withholds a median the rule
    needs - a tier whose contributors stayed below the receipt's floor - or
    that carries no row for a declared tier and market refuses: the declaration
    is the measurement or it does not exist.
    """
    parsed = _validated_receipt(receipt)
    base = {
        tier: _worse_leg(parsed, tier=tier, notional=BASE_NOTIONAL, name=BASE_QUANTILE)
        for tier in DECLARED_TIERS
    }
    adverse = {
        tier: _worse_leg(parsed, tier=tier, notional=ADVERSE_NOTIONAL, name=ADVERSE_QUANTILE)
        for tier in DECLARED_TIERS
    }
    return base, adverse


def declare_measured_cost_family(
    *, receipt_path: Path, base_declaration_path: Path, output_path: Path
) -> tuple[Path, str]:
    """Write the measured carry declaration and return it with its spec hash.

    The receipt is verified against its own content hash, the base declaration
    must be ``funding_carry_panel_v2``, and an output that already exists is
    refused rather than replaced: a declaration is as immutable as the receipt
    it cites. The document is validated as a ``CarryFamilySpec`` before a byte
    is written, so a refusal leaves nothing behind, and the returned hash is
    read back off the published bytes.
    """
    receipt = _read_object(receipt_path, label="the finalisation receipt")
    _verify_receipt_hash(receipt)
    base_tiers, adverse_tiers = measured_slippage_tiers(receipt)
    declaration = _read_object(base_declaration_path, label="the base declaration")
    if declaration.get("family_name") != BASE_FAMILY_NAME:
        raise MeasuredCostError(
            f"a measured declaration is derived from {BASE_FAMILY_NAME}, not "
            f"{declaration.get('family_name')!r}"
        )
    if output_path.exists():
        raise MeasuredCostError("this declaration already exists and is immutable")
    document = _measured_document(
        declaration, receipt=receipt, base=base_tiers, adverse=adverse_tiers
    )
    try:
        CarryFamilySpec.model_validate(document)
    except ValidationError as error:
        raise MeasuredCostError(f"the measured declaration is invalid: {error}") from error
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _, spec_hash = load_family_spec(output_path)
    return output_path, spec_hash


def _validated_receipt(receipt: dict[str, object]) -> FinalizationReceipt:
    """The receipt's own model, which binds its declaration rule verbatim."""
    try:
        return FinalizationReceipt.model_validate(receipt)
    except ValidationError as error:
        raise MeasuredCostError(f"this is not a finalisation receipt: {error}") from error


def _worse_leg(
    receipt: FinalizationReceipt, *, tier: int, notional: str, name: str
) -> Decimal:
    """The worse of the two legs' medians at one notional, in whole basis points."""
    values: list[Decimal] = []
    for market in DECLARED_MARKETS:
        rows = [item for item in receipt.tiers if item.tier == tier and item.market == market]
        if len(rows) != 1:
            raise MeasuredCostError(
                f"the receipt must carry exactly one tier {tier} {market} row"
            )
        entry = rows[0].slippage.get(notional)
        if entry is None:
            raise MeasuredCostError(
                f"the receipt's tier {tier} {market} row never read {notional} USDT"
            )
        value = entry.get(name)
        if not isinstance(value, Decimal):
            raise MeasuredCostError(
                f"the receipt withholds {name} for tier {tier} {market} at {notional} USDT"
            )
        values.append(value)
    return _whole_bps(max(values))


def _whole_bps(value: Decimal) -> Decimal:
    """Round a measured value up to the next whole basis point (spec 5)."""
    # Through `int` so the declaration writes "3" and never "3E+0": the
    # rounding leaves a Decimal whose exponent depends on what it was handed.
    return Decimal(int(value.to_integral_value(rounding=ROUND_CEILING)))


def _verify_receipt_hash(receipt: dict[str, object]) -> None:
    """A receipt is cited by hash, so its bytes must recompute to that hash."""
    material = {key: value for key, value in receipt.items() if key != "content_hash"}
    try:
        recomputed = content_sha256(material)
    except CanonicalizationError as error:
        raise MeasuredCostError(f"the receipt is not canonical: {error}") from error
    if receipt.get("content_hash") != recomputed:
        raise MeasuredCostError("the receipt's content hash does not recompute")


def _measured_document(
    declaration: dict[str, object],
    *,
    receipt: dict[str, object],
    base: dict[int, Decimal],
    adverse: dict[int, Decimal],
) -> dict[str, object]:
    """The v2 document with the family name, the cost tables and the evidence changed."""
    hypothesis = declaration.get("hypothesis")
    if not isinstance(hypothesis, str):
        raise MeasuredCostError("the base declaration carries no hypothesis")
    costs = declaration.get("costs")
    if not isinstance(costs, dict):
        raise MeasuredCostError("the base declaration carries no cost tables")
    receipt_hash = str(receipt["content_hash"])
    return {
        **declaration,
        "family_name": MEASURED_COST_FAMILY,
        "hypothesis": hypothesis + HYPOTHESIS_SENTENCE.format(receipt_hash=receipt_hash),
        "costs": {
            "base": _measured_table(costs.get("base"), tiers=base, name="base"),
            "adverse": _measured_table(costs.get("adverse"), tiers=adverse, name="adverse"),
        },
        "cost_evidence": {
            "receipt_hash": receipt_hash,
            "journal_spec_hash": str(receipt["spec_hash"]),
            "base_notional": BASE_NOTIONAL,
            "adverse_notional": ADVERSE_NOTIONAL,
            "rule": str(receipt["declaration_rule"]),
        },
    }


def _measured_table(table: object, *, tiers: dict[int, Decimal], name: str) -> dict[str, object]:
    """One cost table with its two slippage tiers replaced by measured values."""
    if not isinstance(table, dict):
        raise MeasuredCostError(f"the base declaration carries no {name} cost table")
    measured = {TIER_FIELDS[tier]: str(value) for tier, value in tiers.items()}
    return {**table, **measured}


def _read_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MeasuredCostError(f"{label} is unreadable: {error}") from error
    if not isinstance(document, dict):
        raise MeasuredCostError(f"{label} must be a JSON object")
    return document
