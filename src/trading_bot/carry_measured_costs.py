"""Declare a carry family from the Binance cost journal's finalisation receipt.

Section 5 of ``docs/superpowers/specs/2026-09-12-binance-cost-journal-design.md``
fixed, before any number existed, the reading of a receipt for a declaration
without a ``capital`` block: a v3 family sets ``slippage_bps_per_side_tier_<t>``
in its base table to ``tier_p50_of_p50`` of the *worse leg* at 5 000 USDT and in
its adverse table to ``tier_p50_of_p90`` at 50 000 USDT, each rounded up to the
next whole basis point, and cites the receipt hash. Section 5.1 fixed the
second reading, for a declaration whose base carries a ``capital`` block - a
small, declared-capital book: the base table reads the smallest notional the
receipt measured that is at least the book's per-leg order notional and the
adverse table ten times that notional. ``declaration_notionals`` is the one
place that picks between the two.

This module performs those readings and nothing else. Everything the receipt
does not speak to is carried over from the base declaration unchanged, so the
only difference between the assumed family and the measured one is the two
cost tables and the evidence they came from - which is what makes the two
comparable. A base declaration must be a key of ``MEASURED_FAMILY_BY_BASE``,
and it is that base's own ``capital`` - not the caller's - that decides which
reading applies. The receipt is verified against its own hash first: a
declaration cites a receipt by hash, so bytes that do not recompute to it are
not that receipt.

``verify_measured_declaration`` runs the same reading against an existing
declaration instead of writing one, so a reviewer can re-derive every number in
it from the receipt and the base declaration rather than take the file's word.
"""

import json
import os
from decimal import ROUND_CEILING, Decimal
from pathlib import Path

from pydantic import ValidationError

from trading_bot.binance_cost_journal import FinalizationReceipt
from trading_bot.canonical import CanonicalizationError, content_sha256
from trading_bot.carry_config import (
    MEASURED_FAMILY_BY_BASE,
    CarryCapital,
    CarryCostTable,
    CarryFamilySpec,
)
from trading_bot.panel_config import load_family_spec

DEFAULT_BASE_DECLARATION = Path("configs/funding-carry-panel-v2.json")
# Spec 5's two notionals, as the receipt keys them - the reading a base without
# a `capital` block uses.
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
# The cost-table fields the receipt writes; every other one is the base's.
MEASURED_FIELDS = frozenset(TIER_FIELDS.values())
# The fields a measured declaration is allowed to differ from its base in.
DECLARED_FIELDS = frozenset({"family_name", "hypothesis", "costs"})
# The one sentence the measured family adds to the hypothesis it inherits.
HYPOTHESIS_SENTENCE = (
    " Execution costs are the slippage tiers measured by the Binance cost journal "
    "(receipt {receipt_hash}) under the rule fixed in its design before any number existed."
)


class MeasuredCostError(RuntimeError):
    """Raised when a receipt cannot be read the one way the spec allows."""


def declaration_notionals(
    receipt: FinalizationReceipt, *, capital: CarryCapital | None
) -> tuple[str, str]:
    """The base and adverse notionals a declaration reads, as the receipt's own keys.

    Without a ``capital`` block this is spec 5's fixed pair, ``("5000",
    "50000")``. With one, spec 5.1's ``ladder(N)`` is the smallest of the
    receipt's own notionals that is at least the book's per-leg order notional
    ``N``, and the adverse notional is ten times it. Both must be notionals the
    receipt actually measured - Decimal-equal to one of ``receipt.notionals`` -
    or the declaration refuses rather than reading a rung nobody sampled.

    The two returned strings are always ``str()`` of the actual elements of
    ``receipt.notionals``, never of a freshly computed ``Decimal`` - a receipt
    entry that is numerically ten times the base but written with a different
    exponent (``Decimal("50000.0")`` rather than ``Decimal("50000")``) keys the
    tier statistics under its own string, and a computed ``"50000"`` would miss
    that key even though the membership check above it would pass.
    """
    if capital is None:
        return BASE_NOTIONAL, ADVERSE_NOTIONAL
    order_notional = capital.per_leg_notional_usdt
    ladder = sorted(value for value in receipt.notionals if value >= order_notional)
    if not ladder:
        raise MeasuredCostError(
            "the receipt's notional ladder has nothing at or above the declared "
            f"{order_notional} USDT per-leg order"
        )
    base_notional = ladder[0]
    adverse_target = base_notional * 10
    adverse_notional = next(
        (value for value in receipt.notionals if value == adverse_target), None
    )
    if adverse_notional is None:
        raise MeasuredCostError(
            f"the receipt's notional ladder has no {adverse_target} USDT rung, ten times "
            f"the {base_notional} USDT the declared capital selects"
        )
    return str(base_notional), str(adverse_notional)


def measured_slippage_tiers(
    receipt: dict[str, object], *, capital: CarryCapital | None = None
) -> tuple[dict[int, Decimal], dict[int, Decimal]]:
    """The base and adverse slippage tiers spec 5 or 5.1 reads off one receipt.

    Returns ``(base_by_tier, adverse_by_tier)`` in whole basis points: for each
    declared tier the base value is the worse of the two legs' ``p50_of_p50``
    at ``declaration_notionals``'s base notional and the adverse value the
    worse of their ``p50_of_p90`` at its adverse notional, each rounded up.
    ``capital`` selects which of the two readings applies; ``None`` is spec 5's
    fixed 5 000 / 50 000 USDT pair. A receipt that withholds a median the rule
    needs - a tier whose contributors stayed below the receipt's floor - or
    that carries no row for a declared tier and market refuses: the declaration
    is the measurement or it does not exist.
    """
    parsed = _validated_receipt(receipt)
    base_notional, adverse_notional = declaration_notionals(parsed, capital=capital)
    base = {
        tier: _worse_leg(parsed, tier=tier, notional=base_notional, name=BASE_QUANTILE)
        for tier in DECLARED_TIERS
    }
    adverse = {
        tier: _worse_leg(parsed, tier=tier, notional=adverse_notional, name=ADVERSE_QUANTILE)
        for tier in DECLARED_TIERS
    }
    return base, adverse


def declare_measured_cost_family(
    *, receipt_path: Path, base_declaration_path: Path, output_path: Path
) -> tuple[Path, str]:
    """Write the measured carry declaration and return it with its spec hash.

    The receipt is verified against its own content hash and the base
    declaration's family must be a key of ``MEASURED_FAMILY_BY_BASE``; the
    output's family name is the value that key maps to, and the base's own
    ``capital`` decides which of spec 5's or 5.1's readings prices it. The
    document is validated as a ``CarryFamilySpec`` before a byte is written, so
    a refusal leaves nothing behind, and the returned hash is read back off the
    published bytes.

    The output is created exclusively (``open(..., "x")``): an existing
    declaration is refused rather than replaced, and two writers racing for the
    same path cannot both believe they published it - a declaration is as
    immutable as the receipt it cites. A process killed mid-write therefore
    leaves a partial file at the output path, which will not parse as a
    declaration; the operator deletes it and runs the command again.
    """
    receipt = _read_object(receipt_path, label="the finalisation receipt")
    _verify_receipt_hash(receipt)
    declaration = _read_object(base_declaration_path, label="the base declaration")
    measured_family_name = _verify_base_family(declaration)
    base_spec = _validated_base(declaration)
    base_tiers, adverse_tiers = measured_slippage_tiers(receipt, capital=base_spec.capital)
    document = _measured_document(
        declaration,
        receipt=receipt,
        base=base_tiers,
        adverse=adverse_tiers,
        family_name=measured_family_name,
        capital=base_spec.capital,
    )
    try:
        CarryFamilySpec.model_validate(document)
    except ValidationError as error:
        raise MeasuredCostError(f"the measured declaration is invalid: {error}") from error
    _publish_exclusively(output_path, document)
    _, spec_hash = load_family_spec(output_path)
    return output_path, spec_hash


def verify_measured_declaration(
    *,
    receipt_path: Path,
    spec_path: Path,
    base_declaration_path: Path = DEFAULT_BASE_DECLARATION,
) -> str:
    """Re-derive a measured declaration from its receipt; return its spec hash.

    The declaration is not trusted to describe itself: the receipt is verified
    against its own hash, the declaration's family must be its base's mapped
    name, the citation must name that receipt, its journal spec, the notionals
    and the rule the base's ``capital`` selects, the four slippage values must
    be exactly what that reading of the receipt produces today, and every
    other field - top level and inside both cost tables - must still be the
    base declaration's, with the hypothesis its sentence plus the one
    citation. Any mismatch refuses; nothing is written.
    """
    receipt = _read_object(receipt_path, label="the finalisation receipt")
    _verify_receipt_hash(receipt)
    base = _read_object(base_declaration_path, label="the base declaration")
    measured_family_name = _verify_base_family(base)
    base_spec = _validated_base(base)
    base_tiers, adverse_tiers = measured_slippage_tiers(receipt, capital=base_spec.capital)
    document = _read_object(spec_path, label="the measured declaration")
    spec = _validated_declaration(document, measured_family_name=measured_family_name)
    _verify_evidence(spec, receipt=receipt, capital=base_spec.capital)
    _verify_hypothesis(document, base=base, receipt=receipt)
    _verify_carried_fields(document, base=base)
    _verify_cost_tables(document, base=base)
    _verify_measured_tiers(spec, base=base_tiers, adverse=adverse_tiers)
    return _canonical_hash(document, label="the measured declaration")


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
    if receipt.get("content_hash") != _canonical_hash(material, label="the receipt"):
        raise MeasuredCostError("the receipt's content hash does not recompute")


def _canonical_hash(document: dict[str, object], *, label: str) -> str:
    try:
        return content_sha256(document)
    except CanonicalizationError as error:
        raise MeasuredCostError(f"{label} is not canonical: {error}") from error


def _verify_base_family(declaration: dict[str, object]) -> str:
    """The base family must be a key of ``MEASURED_FAMILY_BY_BASE``; returns its mapped name."""
    name = declaration.get("family_name")
    if not isinstance(name, str) or name not in MEASURED_FAMILY_BY_BASE:
        bases = ", ".join(sorted(MEASURED_FAMILY_BY_BASE))
        raise MeasuredCostError(
            f"a measured declaration is derived from one of {bases}, not {name!r}"
        )
    return MEASURED_FAMILY_BY_BASE[name]


def _validated_base(declaration: dict[str, object]) -> CarryFamilySpec:
    """The base declaration's own model, whose ``capital`` selects the reading."""
    try:
        return CarryFamilySpec.model_validate(declaration)
    except ValidationError as error:
        raise MeasuredCostError(f"the base declaration does not validate: {error}") from error


def _validated_declaration(
    document: dict[str, object], *, measured_family_name: str
) -> CarryFamilySpec:
    """The declaration's own model, which binds the rule it cites verbatim."""
    try:
        spec = CarryFamilySpec.model_validate(document)
    except ValidationError as error:
        raise MeasuredCostError(f"the declaration does not validate: {error}") from error
    if spec.family_name != measured_family_name:
        raise MeasuredCostError(f"this is not a {measured_family_name} declaration")
    return spec


def _verify_evidence(
    spec: CarryFamilySpec, *, receipt: dict[str, object], capital: CarryCapital | None
) -> None:
    """The citation must name this receipt, its journal spec, the rule and the two notionals."""
    evidence = spec.cost_evidence
    if evidence is None:  # the model refuses this already; the guard keeps it local
        raise MeasuredCostError("the declaration cites no cost evidence")
    if evidence.receipt_hash != receipt.get("content_hash"):
        raise MeasuredCostError("the declaration cites another receipt")
    if evidence.journal_spec_hash != receipt.get("spec_hash"):
        raise MeasuredCostError("the declaration cites another journal spec")
    parsed = _validated_receipt(receipt)
    base_notional, adverse_notional = declaration_notionals(parsed, capital=capital)
    cited = (evidence.base_notional, evidence.adverse_notional)
    if cited != (Decimal(base_notional), Decimal(adverse_notional)):
        raise MeasuredCostError("the declaration cites notionals the rule does not fix")
    rule_field = "capital_declaration_rule" if capital is not None else "declaration_rule"
    if evidence.rule != receipt.get(rule_field):
        raise MeasuredCostError("the declaration cites a rule its capital does not select")


def _verify_hypothesis(
    document: dict[str, object], *, base: dict[str, object], receipt: dict[str, object]
) -> None:
    hypothesis = base.get("hypothesis")
    if not isinstance(hypothesis, str):
        raise MeasuredCostError("the base declaration carries no hypothesis")
    sentence = HYPOTHESIS_SENTENCE.format(receipt_hash=str(receipt.get("content_hash")))
    if document.get("hypothesis") != hypothesis + sentence:
        raise MeasuredCostError("the hypothesis is not the base's plus its one citation")


def _verify_carried_fields(document: dict[str, object], *, base: dict[str, object]) -> None:
    """Everything the receipt does not speak to is still the base declaration's."""
    if set(document) != set(base) | {"cost_evidence"}:
        raise MeasuredCostError("the declaration's fields are not the base declaration's")
    for key in base:
        if key not in DECLARED_FIELDS and document[key] != base[key]:
            raise MeasuredCostError(f"the declaration's {key} is not the base declaration's")


def _verify_cost_tables(document: dict[str, object], *, base: dict[str, object]) -> None:
    """Both cost tables differ from the base's in the two measured fields alone."""
    costs = document.get("costs")
    before = base.get("costs")
    if not isinstance(costs, dict) or not isinstance(before, dict):
        raise MeasuredCostError("a declaration carries a base and an adverse cost table")
    for name in ("base", "adverse"):
        table, original = costs.get(name), before.get(name)
        if not isinstance(table, dict) or not isinstance(original, dict):
            raise MeasuredCostError(f"a declaration carries a {name} cost table")
        if set(table) != set(original):
            raise MeasuredCostError(f"the {name} table's fields are not the base table's")
        for key in original:
            if key not in MEASURED_FIELDS and table[key] != original[key]:
                raise MeasuredCostError(
                    f"the {name} table's {key} is not the base declaration's"
                )


def _verify_measured_tiers(
    spec: CarryFamilySpec, *, base: dict[int, Decimal], adverse: dict[int, Decimal]
) -> None:
    """The four slippage values are what spec 5's reading produces today."""
    for name, table, measured in (
        ("base", spec.costs.base, base),
        ("adverse", spec.costs.adverse, adverse),
    ):
        declared = _declared_tiers(table)
        for tier, value in measured.items():
            if declared[tier] != value:
                raise MeasuredCostError(
                    f"the {name} table's {TIER_FIELDS[tier]} is {declared[tier]}, "
                    f"not the measured {value}"
                )


def _declared_tiers(table: CarryCostTable) -> dict[int, Decimal]:
    return {
        1: table.slippage_bps_per_side_tier_one,
        2: table.slippage_bps_per_side_tier_two,
    }


def _publish_exclusively(output_path: Path, document: dict[str, object]) -> None:
    """Create the declaration, or refuse: the path is claimed by the first writer."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise MeasuredCostError("this declaration already exists and is immutable") from error
    except OSError as error:
        raise MeasuredCostError(f"the declaration could not be written: {error}") from error


def _measured_document(
    declaration: dict[str, object],
    *,
    receipt: dict[str, object],
    base: dict[int, Decimal],
    adverse: dict[int, Decimal],
    family_name: str,
    capital: CarryCapital | None,
) -> dict[str, object]:
    """The base document with the family name, the cost tables and the evidence changed."""
    hypothesis = declaration.get("hypothesis")
    if not isinstance(hypothesis, str):
        raise MeasuredCostError("the base declaration carries no hypothesis")
    costs = declaration.get("costs")
    if not isinstance(costs, dict):
        raise MeasuredCostError("the base declaration carries no cost tables")
    receipt_hash = str(receipt["content_hash"])
    parsed = _validated_receipt(receipt)
    base_notional, adverse_notional = declaration_notionals(parsed, capital=capital)
    rule_field = "capital_declaration_rule" if capital is not None else "declaration_rule"
    return {
        **declaration,
        "family_name": family_name,
        "hypothesis": hypothesis + HYPOTHESIS_SENTENCE.format(receipt_hash=receipt_hash),
        "costs": {
            "base": _measured_table(costs.get("base"), tiers=base, name="base"),
            "adverse": _measured_table(costs.get("adverse"), tiers=adverse, name="adverse"),
        },
        "cost_evidence": {
            "receipt_hash": receipt_hash,
            "journal_spec_hash": str(receipt["spec_hash"]),
            "base_notional": base_notional,
            "adverse_notional": adverse_notional,
            "rule": str(receipt[rule_field]),
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
