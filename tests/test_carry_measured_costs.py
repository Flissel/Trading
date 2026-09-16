"""The carry family declared from the Binance cost journal's receipt (spec 5)."""

import json
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.carry_fixtures import small_carry_v4_config
from tests.test_binance_cost_journal import (
    LadderVenue,
    finalized_receipt,
    journal_with_rounds,
    leg_document,
    lower_the_eligibility_floors,
    read_document,
)
from trading_bot.binance_cost_journal import (
    CAPITAL_DECLARATION_RULE,
    DECLARATION_RULE,
    FinalizationReceipt,
)
from trading_bot.carry_config import (
    MEMBER_NAMES_BY_FAMILY,
    CarryCapital,
    CarryFamilySpec,
    load_carry_family_spec,
)
from trading_bot.carry_measured_costs import (
    MeasuredCostError,
    declaration_notionals,
    declare_measured_cost_family,
    measured_slippage_tiers,
    verify_measured_declaration,
)
from trading_bot.cli import main
from trading_bot.cost_evidence_rule import DECLARATION_RULE as LEAF_RULE
from trading_bot.panel_config import load_family_spec

CONFIG_V1 = Path("configs/funding-carry-panel-v1.json")
CONFIG_V2 = Path("configs/funding-carry-panel-v2.json")
CONFIG_V4 = Path("configs/funding-carry-panel-v4.json")
ROUNDS = 8
LEGS_PER_TIER_AND_MARKET = 4
# Four legs a tier and market, so every tier median clears the receipt's
# four-contributor floor. The multiplier is the leg's width: at round k the
# venue quotes `100 * multiplier * k` bps of slippage a side, so a leg's p50
# over eight rounds is `400 * multiplier` and its p90 `800 * multiplier`.
# Tier one's worse leg is the perpetual and tier two's is the spot, so the rule
# picks a market per tier rather than once for the whole receipt.
LEG_PLAN: tuple[tuple[str, int, int], ...] = (
    ("spot", 1, 1),
    ("um", 1, 2),
    ("spot", 2, 3),
    ("um", 2, 1),
)
# The sentence the declaration appends to the v2 hypothesis, written out here
# rather than imported: the wording is part of what this test pins down.
HYPOTHESIS_SENTENCE = (
    " Execution costs are the slippage tiers measured by the Binance cost journal "
    "(receipt {receipt_hash}) under the rule fixed in its design before any number existed."
)


def leg_symbol(*, market: str, tier: int, index: int) -> str:
    return f"{'S' if market == 'spot' else 'P'}{tier}{index}USDT"


def sample_instruments() -> list[dict[str, object]]:
    return [
        leg_document(leg_symbol(market=market, tier=tier, index=index), market=market, tier=tier)
        for market, tier, _ in LEG_PLAN
        for index in range(LEGS_PER_TIER_AND_MARKET)
    ]


def sample_multipliers() -> dict[str, int]:
    return {
        leg_symbol(market=market, tier=tier, index=index): multiplier
        for market, tier, multiplier in LEG_PLAN
        for index in range(LEGS_PER_TIER_AND_MARKET)
    }


def measured_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, thin: tuple[str, ...] = ()
) -> dict[str, object]:
    """A genuine receipt over sixteen legs; a `thin` leg never fills 50 000 USDT."""
    lower_the_eligibility_floors(monkeypatch)
    journal = journal_with_rounds(
        tmp_path,
        rounds=ROUNDS,
        fetcher=LadderVenue(
            multipliers=sample_multipliers(), depth=dict.fromkeys(thin, "100")
        ),
        instruments=sample_instruments(),
    )
    _, output = finalized_receipt(tmp_path, journal)
    return read_document(output)


def declared_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, object], Path, str]:
    """The receipt, the v3 document it declares and that document's spec hash."""
    receipt = measured_receipt(tmp_path, monkeypatch)
    output, spec_hash = declare_measured_cost_family(
        receipt_path=tmp_path / "receipt.json",
        base_declaration_path=CONFIG_V2,
        output_path=tmp_path / "funding-carry-panel-v3.json",
    )
    return receipt, output, spec_hash


def declared_v4_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, object], Path, str]:
    """The receipt, the v4-measured document it declares and that document's spec hash."""
    receipt = measured_receipt(tmp_path, monkeypatch)
    output, spec_hash = declare_measured_cost_family(
        receipt_path=tmp_path / "receipt.json",
        base_declaration_path=CONFIG_V4,
        output_path=tmp_path / "funding-carry-panel-v4-measured.json",
    )
    return receipt, output, spec_hash


def slot_capital(*, per_leg: str, slots: int = 10) -> CarryCapital:
    """A capital block sized so `per_leg` and `slots` are its own arithmetic."""
    return CarryCapital.model_validate(
        {
            "book_usdt": str(Decimal(per_leg) * 2 * slots),
            "pair_slots": slots,
            "per_leg_notional_usdt": per_leg,
            "fee_tier": "standard_taker_no_bnb",
        }
    )


def write_json(path: Path, document: dict[str, object]) -> Path:
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def edited_declaration(
    output: Path, path: Path, **overrides: object
) -> Path:
    """The published declaration with top-level fields replaced, written elsewhere."""
    return write_json(path, {**read_document(output), **overrides})


def edited_evidence(output: Path, path: Path, **overrides: object) -> Path:
    document = read_document(output)
    evidence = document["cost_evidence"]
    assert isinstance(evidence, dict)
    return edited_declaration(output, path, cost_evidence={**evidence, **overrides})


def edited_cost_table(output: Path, path: Path, *, name: str, **overrides: object) -> Path:
    document = read_document(output)
    costs = document["costs"]
    assert isinstance(costs, dict)
    tables = {**costs, name: {**costs[name], **overrides}}
    return edited_declaration(output, path, costs=tables)


def tier_row(document: dict[str, object], *, tier: int, market: str) -> dict[str, object]:
    tiers = document["tiers"]
    assert isinstance(tiers, list)
    row = next(item for item in tiers if item["tier"] == tier and item["market"] == market)
    assert isinstance(row, dict)
    return row


def notional_row(
    document: dict[str, object], *, tier: int, market: str, notional: str
) -> dict[str, object]:
    slippage = tier_row(document, tier=tier, market=market)["slippage"]
    assert isinstance(slippage, dict)
    entry = slippage[notional]
    assert isinstance(entry, dict)
    return entry


def tier_median(
    document: dict[str, object], *, tier: int, market: str, notional: str, name: str
) -> object:
    return notional_row(document, tier=tier, market=market, notional=notional)[name]


def with_medians(
    document: dict[str, object], medians: Mapping[tuple[int, str, str, str], str]
) -> dict[str, object]:
    """A copy of the receipt whose named tier medians read the given values."""
    copied = json.loads(json.dumps(document))
    assert isinstance(copied, dict)
    for (tier, market, notional, name), value in medians.items():
        notional_row(copied, tier=tier, market=market, notional=notional)[name] = value
    return copied


def test_the_worse_leg_of_each_tier_sets_the_measured_costs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tier one is the perpetual's number, tier two the spot's - the worse of the two."""
    document = measured_receipt(tmp_path, monkeypatch)
    base, adverse = measured_slippage_tiers(document)
    assert base == {1: Decimal("800"), 2: Decimal("1200")}
    assert adverse == {1: Decimal("1600"), 2: Decimal("2400")}
    # The better leg of each tier measured less and is not what was declared.
    assert tier_median(
        document, tier=1, market="spot", notional="5000", name="p50_of_p50"
    ) == "400.000000"
    assert tier_median(
        document, tier=2, market="um", notional="50000", name="p50_of_p90"
    ) == "800.000000"


def test_the_base_table_reads_five_thousand_and_the_adverse_fifty_thousand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the two declared notionals and the two declared quantiles are read."""
    document = measured_receipt(tmp_path, monkeypatch)
    forged = with_medians(
        document,
        {
            (1, "um", "500", "p50_of_p50"): "9999.000000",
            (1, "um", "5000", "p50_of_p90"): "9999.000000",
            (1, "um", "50000", "p50_of_p50"): "9999.000000",
        },
    )
    base, adverse = measured_slippage_tiers(forged)
    assert (base[1], adverse[1]) == (Decimal("800"), Decimal("1600"))


def test_a_measured_tier_rounds_up_to_the_next_whole_basis_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fraction of a basis point costs a whole one; an exact integer stays."""
    document = measured_receipt(tmp_path, monkeypatch)
    forged = with_medians(
        document,
        {
            (1, "spot", "5000", "p50_of_p50"): "2.000001",
            (1, "um", "5000", "p50_of_p50"): "0.500000",
            (2, "spot", "5000", "p50_of_p50"): "2.000000",
            (2, "um", "5000", "p50_of_p50"): "1.999999",
            (1, "spot", "50000", "p50_of_p90"): "9.999999",
            (1, "um", "50000", "p50_of_p90"): "0.000001",
            (2, "spot", "50000", "p50_of_p90"): "3.000000",
            (2, "um", "50000", "p50_of_p90"): "0.000001",
        },
    )
    base, adverse = measured_slippage_tiers(forged)
    assert base == {1: Decimal("3"), 2: Decimal("2")}
    assert adverse == {1: Decimal("10"), 2: Decimal("3")}
    # Whole basis points are written plainly, never in exponent form.
    assert [str(value) for value in base.values()] == ["3", "2"]


def test_a_tier_median_the_receipt_withheld_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One thin tier-one spot leg leaves three contributors at 50 000: no median."""
    thin = (leg_symbol(market="spot", tier=1, index=0),)
    document = measured_receipt(tmp_path, monkeypatch, thin=thin)
    row = notional_row(document, tier=1, market="spot", notional="50000")
    assert row["contributing_count"] == 3
    assert row["p50_of_p90"] is None
    with pytest.raises(MeasuredCostError, match="tier 1 spot"):
        measured_slippage_tiers(document)


def test_a_missing_tier_row_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = measured_receipt(tmp_path, monkeypatch)
    rows = document["tiers"]
    assert isinstance(rows, list)
    kept = [item for item in rows if not (item["tier"] == 2 and item["market"] == "um")]
    with pytest.raises(MeasuredCostError, match="tier 2 um"):
        measured_slippage_tiers({**document, "tiers": kept})


def test_a_document_that_is_not_a_receipt_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A foreign declaration rule is not a receipt this rule may be read off."""
    document = measured_receipt(tmp_path, monkeypatch)
    with pytest.raises(MeasuredCostError, match="not a finalisation receipt"):
        measured_slippage_tiers({**document, "declaration_rule": "round down"})


def test_declaration_notionals_without_capital_reads_five_and_fifty_thousand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = measured_receipt(tmp_path, monkeypatch)
    receipt = FinalizationReceipt.model_validate(document)
    assert declaration_notionals(receipt, capital=None) == ("5000", "50000")


def test_declaration_notionals_with_capital_reads_the_ladder_rung_at_or_above_the_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """500 per leg picks the 500/5 000 rung; 1 250 per leg is above 500, so it
    lands on the same 5 000/50 000 rung a declaration without capital reads."""
    document = measured_receipt(tmp_path, monkeypatch)
    receipt = FinalizationReceipt.model_validate(document)
    assert declaration_notionals(receipt, capital=slot_capital(per_leg="500")) == (
        "500", "5000",
    )
    assert declaration_notionals(
        receipt, capital=slot_capital(per_leg="1250", slots=4)
    ) == ("5000", "50000")


def test_declaration_notionals_refuses_an_order_above_the_ladder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = measured_receipt(tmp_path, monkeypatch)
    receipt = FinalizationReceipt.model_validate(document)
    with pytest.raises(MeasuredCostError, match="60000"):
        declaration_notionals(receipt, capital=slot_capital(per_leg="60000", slots=1))


def test_declaration_notionals_refuses_a_ladder_without_the_tenfold_rung(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """5 000 is on the ladder but ten times it, 50 000, is not: refuse rather
    than read a rung nobody measured."""
    document = measured_receipt(tmp_path, monkeypatch)
    truncated = FinalizationReceipt.model_validate({**document, "notionals": ["500", "5000"]})
    with pytest.raises(MeasuredCostError, match="50000"):
        declaration_notionals(truncated, capital=slot_capital(per_leg="5000", slots=1))


def test_declaration_notionals_resolves_a_rescaled_tenfold_rung_to_its_own_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The receipt may record its top rung as "50000.0": numerically ten times
    the 5 000 base, but a different string than a freshly computed "50000".

    The tier statistics are keyed by the receipt's own notional strings
    (`_notional_keys`), so the adverse notional `declaration_notionals` returns
    must be resolved from the receipt's own elements, not recomputed - a
    recomputed "50000" would pass the numeric-equality membership check yet
    miss the real "50000.0" key, and `_worse_leg` would refuse it as never
    measured even though the receipt did measure it.
    """
    lower_the_eligibility_floors(monkeypatch)
    journal = journal_with_rounds(
        tmp_path,
        rounds=ROUNDS,
        fetcher=LadderVenue(multipliers=sample_multipliers()),
        instruments=sample_instruments(),
        notionals=["500", "5000", "50000.0"],
    )
    _, output = finalized_receipt(tmp_path, journal)
    document = read_document(output)
    row = tier_row(document, tier=1, market="um")
    slippage = row["slippage"]
    assert isinstance(slippage, dict)
    assert set(slippage) == {"500", "5000", "50000.0"}

    receipt = FinalizationReceipt.model_validate(document)
    capital = slot_capital(per_leg="5000", slots=1)
    assert declaration_notionals(receipt, capital=capital) == ("5000", "50000.0")
    # The pre-fix code returned a computed "50000", which this receipt never
    # keys, and `_worse_leg` would have refused it as never measured.
    base, adverse = measured_slippage_tiers(document, capital=capital)
    assert base == {1: Decimal("800"), 2: Decimal("1200")}
    assert adverse == {1: Decimal("1600"), 2: Decimal("2400")}


def test_the_declared_family_carries_the_measured_tiers_and_its_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, output, spec_hash = declared_family(tmp_path, monkeypatch)
    spec, loaded_hash = load_family_spec(output)
    assert isinstance(spec, CarryFamilySpec)
    assert (spec_hash, len(spec_hash)) == (loaded_hash, 64)
    assert spec.family_name == "funding_carry_panel_v3"
    assert tuple(item.name for item in spec.members) == MEMBER_NAMES_BY_FAMILY[
        "funding_carry_panel_v2"
    ]
    assert spec.costs.base.slippage_bps_per_side_tier_one == Decimal("800")
    assert spec.costs.base.slippage_bps_per_side_tier_two == Decimal("1200")
    assert spec.costs.adverse.slippage_bps_per_side_tier_one == Decimal("1600")
    assert spec.costs.adverse.slippage_bps_per_side_tier_two == Decimal("2400")
    evidence = spec.cost_evidence
    assert evidence is not None
    assert evidence.receipt_hash == receipt["content_hash"]
    assert evidence.journal_spec_hash == receipt["spec_hash"]
    assert evidence.base_notional == Decimal("5000")
    assert evidence.adverse_notional == Decimal("50000")
    assert evidence.rule == DECLARATION_RULE


def test_the_declaration_changes_only_the_costs_the_name_and_the_hypothesis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Everything the receipt does not speak to is the v2 declaration's own value."""
    receipt, output, _ = declared_family(tmp_path, monkeypatch)
    written = read_document(output)
    base = read_document(CONFIG_V2)
    assert set(written) == set(base) | {"cost_evidence"}
    declared = {"family_name", "hypothesis", "costs", "cost_evidence"}
    assert all(written[key] == base[key] for key in base if key not in declared)
    hypothesis = base["hypothesis"]
    assert isinstance(hypothesis, str)
    assert written["hypothesis"] == hypothesis + HYPOTHESIS_SENTENCE.format(
        receipt_hash=receipt["content_hash"]
    )
    costs = written["costs"]
    original = base["costs"]
    assert isinstance(costs, dict)
    assert isinstance(original, dict)
    measured = {"slippage_bps_per_side_tier_one", "slippage_bps_per_side_tier_two"}
    for name in ("base", "adverse"):
        table, before = costs[name], original[name]
        assert set(table) == set(before)
        assert all(table[key] == before[key] for key in before if key not in measured)
    assert costs["base"]["slippage_bps_per_side_tier_one"] == "800"
    assert costs["base"]["slippage_bps_per_side_tier_two"] == "1200"
    assert costs["adverse"]["slippage_bps_per_side_tier_one"] == "1600"
    assert costs["adverse"]["slippage_bps_per_side_tier_two"] == "2400"


def test_a_receipt_whose_hash_does_not_recompute_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = measured_receipt(tmp_path, monkeypatch)
    forged = with_medians(document, {(1, "um", "5000", "p50_of_p50"): "1.000000"})
    path = tmp_path / "forged-receipt.json"
    path.write_text(json.dumps(forged), encoding="utf-8")
    output = tmp_path / "forged-v3.json"
    with pytest.raises(MeasuredCostError, match="content hash"):
        declare_measured_cost_family(
            receipt_path=path, base_declaration_path=CONFIG_V2, output_path=output
        )
    assert not output.exists()


def test_a_receipt_that_is_not_canonical_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A binary float never entered the canonical form the hash is taken over."""
    document = measured_receipt(tmp_path, monkeypatch)
    path = tmp_path / "float-receipt.json"
    path.write_text(json.dumps({**document, "segments_beyond_head": 0.5}), encoding="utf-8")
    with pytest.raises(MeasuredCostError, match="canonical"):
        declare_measured_cost_family(
            receipt_path=path,
            base_declaration_path=CONFIG_V2,
            output_path=tmp_path / "float-v3.json",
        )


def test_an_unreadable_receipt_is_refused(tmp_path: Path) -> None:
    with pytest.raises(MeasuredCostError, match="unreadable"):
        declare_measured_cost_family(
            receipt_path=tmp_path / "absent.json",
            base_declaration_path=CONFIG_V2,
            output_path=tmp_path / "absent-v3.json",
        )


def test_a_v1_base_declaration_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The measured family is v2's book on measured costs, not v1's."""
    measured_receipt(tmp_path, monkeypatch)
    output = tmp_path / "from-v1.json"
    with pytest.raises(MeasuredCostError, match="funding_carry_panel_v2"):
        declare_measured_cost_family(
            receipt_path=tmp_path / "receipt.json",
            base_declaration_path=CONFIG_V1,
            output_path=output,
        )
    assert not output.exists()


def test_a_published_declaration_is_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The output path is claimed exclusively; a second writer is refused, not merged."""
    receipt, output, spec_hash = declared_family(tmp_path, monkeypatch)
    published = output.read_bytes()
    # The published bytes round-trip: a whole document, flushed and fsynced.
    assert read_document(output)["cost_evidence"] == {
        "receipt_hash": receipt["content_hash"],
        "journal_spec_hash": receipt["spec_hash"],
        "base_notional": "5000",
        "adverse_notional": "50000",
        "rule": DECLARATION_RULE,
    }
    assert load_family_spec(output)[1] == spec_hash
    with pytest.raises(MeasuredCostError, match="already exists"):
        declare_measured_cost_family(
            receipt_path=tmp_path / "receipt.json",
            base_declaration_path=CONFIG_V2,
            output_path=output,
        )
    assert output.read_bytes() == published


def test_the_measured_family_holds_the_v2_member_set() -> None:
    assert MEMBER_NAMES_BY_FAMILY["funding_carry_panel_v3"] == MEMBER_NAMES_BY_FAMILY[
        "funding_carry_panel_v2"
    ]


def test_the_assumed_families_carry_no_cost_evidence() -> None:
    for config in (CONFIG_V1, CONFIG_V2):
        spec, _ = load_carry_family_spec(config)
        assert spec.cost_evidence is None


def test_a_measured_family_without_its_evidence_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, output, _ = declared_family(tmp_path, monkeypatch)
    document = read_document(output)
    del document["cost_evidence"]
    with pytest.raises(ValidationError, match="cites the receipt"):
        CarryFamilySpec.model_validate(document)


def test_an_assumed_family_carrying_evidence_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v1 and v2 declare assumed tiers; they may not wear a citation."""
    _, output, _ = declared_family(tmp_path, monkeypatch)
    declared = read_document(output)
    document = {**read_document(CONFIG_V2), "cost_evidence": declared["cost_evidence"]}
    with pytest.raises(ValidationError, match="only a measured family"):
        CarryFamilySpec.model_validate(document)


def test_the_cited_rule_is_the_one_the_receipt_is_published_under() -> None:
    """One string, in a leaf module both the journal and the declaration bind."""
    assert LEAF_RULE == DECLARATION_RULE


def test_a_declaration_citing_another_rule_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, output, _ = declared_family(tmp_path, monkeypatch)
    document = read_document(output)
    evidence = document["cost_evidence"]
    assert isinstance(evidence, dict)
    forged = {**document, "cost_evidence": {**evidence, "rule": "round down"}}
    with pytest.raises(ValidationError, match="verbatim"):
        CarryFamilySpec.model_validate(forged)


def test_a_genuine_declaration_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, output, spec_hash = declared_family(tmp_path, monkeypatch)
    verified = verify_measured_declaration(
        receipt_path=tmp_path / "receipt.json",
        spec_path=output,
        base_declaration_path=CONFIG_V2,
    )
    assert verified == spec_hash


def test_verification_refuses_an_edited_slippage_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cheaper tier than the receipt measured is not this receipt's declaration."""
    _, output, _ = declared_family(tmp_path, monkeypatch)
    edited = edited_cost_table(
        output, tmp_path / "cheaper-v3.json", name="base",
        slippage_bps_per_side_tier_one="799",
    )
    with pytest.raises(MeasuredCostError, match="slippage_bps_per_side_tier_one"):
        verify_measured_declaration(
            receipt_path=tmp_path / "receipt.json",
            spec_path=edited,
            base_declaration_path=CONFIG_V2,
        )


def test_verification_refuses_an_edited_receipt_citation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, output, _ = declared_family(tmp_path, monkeypatch)
    edited = edited_evidence(output, tmp_path / "other-receipt-v3.json", receipt_hash="0" * 64)
    with pytest.raises(MeasuredCostError, match="cites another receipt"):
        verify_measured_declaration(
            receipt_path=tmp_path / "receipt.json",
            spec_path=edited,
            base_declaration_path=CONFIG_V2,
        )


def test_verification_refuses_an_edited_journal_spec_citation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, output, _ = declared_family(tmp_path, monkeypatch)
    edited = edited_evidence(
        output, tmp_path / "other-journal-v3.json", journal_spec_hash="1" * 64
    )
    with pytest.raises(MeasuredCostError, match="another journal spec"):
        verify_measured_declaration(
            receipt_path=tmp_path / "receipt.json",
            spec_path=edited,
            base_declaration_path=CONFIG_V2,
        )


def test_verification_refuses_a_cited_notional_the_rule_does_not_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, output, _ = declared_family(tmp_path, monkeypatch)
    edited = edited_evidence(output, tmp_path / "other-notional-v3.json", base_notional="500")
    with pytest.raises(MeasuredCostError, match="notionals the rule does not fix"):
        verify_measured_declaration(
            receipt_path=tmp_path / "receipt.json",
            spec_path=edited,
            base_declaration_path=CONFIG_V2,
        )


def test_verification_refuses_a_wrong_rule_string(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, output, _ = declared_family(tmp_path, monkeypatch)
    edited = edited_evidence(output, tmp_path / "wrong-rule-v3.json", rule="round down")
    with pytest.raises(MeasuredCostError, match="does not validate"):
        verify_measured_declaration(
            receipt_path=tmp_path / "receipt.json",
            spec_path=edited,
            base_declaration_path=CONFIG_V2,
        )


def test_verification_refuses_an_edited_member_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The book is v2's: a member the receipt never spoke to may not move."""
    _, output, _ = declared_family(tmp_path, monkeypatch)
    document = read_document(output)
    members = document["members"]
    assert isinstance(members, list)
    moved = [{**members[0], "hold_weeks": 12}, *members[1:]]
    edited = edited_declaration(output, tmp_path / "moved-member-v3.json", members=moved)
    with pytest.raises(MeasuredCostError, match="members is not the base declaration's"):
        verify_measured_declaration(
            receipt_path=tmp_path / "receipt.json",
            spec_path=edited,
            base_declaration_path=CONFIG_V2,
        )


def test_verification_refuses_an_edited_hypothesis_and_an_edited_fee(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, output, _ = declared_family(tmp_path, monkeypatch)
    quieter = edited_declaration(
        output, tmp_path / "quiet-v3.json", hypothesis="Costs were measured."
    )
    with pytest.raises(MeasuredCostError, match="one citation"):
        verify_measured_declaration(
            receipt_path=tmp_path / "receipt.json",
            spec_path=quieter,
            base_declaration_path=CONFIG_V2,
        )
    cheaper = edited_cost_table(
        output, tmp_path / "cheap-fee-v3.json", name="adverse", spot_fee_bps_per_side="1"
    )
    with pytest.raises(MeasuredCostError, match="spot_fee_bps_per_side"):
        verify_measured_declaration(
            receipt_path=tmp_path / "receipt.json",
            spec_path=cheaper,
            base_declaration_path=CONFIG_V2,
        )


def test_verification_refuses_an_assumed_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    measured_receipt(tmp_path, monkeypatch)
    with pytest.raises(MeasuredCostError, match="funding_carry_panel_v3 declaration"):
        verify_measured_declaration(
            receipt_path=tmp_path / "receipt.json",
            spec_path=CONFIG_V2,
            base_declaration_path=CONFIG_V2,
        )


def test_the_cli_declares_the_family_and_prints_the_path_and_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    measured_receipt(tmp_path, monkeypatch)
    base = tmp_path / "funding-carry-panel-v2.json"
    base.write_bytes(CONFIG_V2.read_bytes())
    output = tmp_path / "funding-carry-panel-v3.json"
    arguments = [
        "carry-declare-measured", "--workspace-root", str(tmp_path),
        "--receipt", str(tmp_path / "receipt.json"), "--base-config", str(base),
        "--output", str(output),
    ]
    assert main(arguments) == 0
    spec, spec_hash = load_family_spec(output)
    assert isinstance(spec, CarryFamilySpec)
    assert spec.family_name == "funding_carry_panel_v3"
    printed = capsys.readouterr().out
    assert str(output) in printed
    assert f"family spec hash: {spec_hash}" in printed
    # The declaration is immutable: a second run is the operator's stop code.
    assert main(arguments) == 2


def test_the_cli_refuses_a_path_outside_the_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    measured_receipt(tmp_path, monkeypatch)
    assert main([
        "carry-declare-measured", "--workspace-root", str(tmp_path),
        "--receipt", str(tmp_path / "receipt.json"), "--base-config", str(CONFIG_V2.resolve()),
        "--output", str(tmp_path / "outside-v3.json"),
    ]) == 2


def test_the_cli_verifies_a_declaration_and_stops_on_an_edited_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _, output, spec_hash = declared_family(tmp_path, monkeypatch)
    base = tmp_path / "funding-carry-panel-v2.json"
    base.write_bytes(CONFIG_V2.read_bytes())
    arguments = [
        "carry-verify-measured", "--workspace-root", str(tmp_path),
        "--receipt", str(tmp_path / "receipt.json"), "--spec", str(output),
        "--base-config", str(base),
    ]
    assert main(arguments) == 0
    printed = capsys.readouterr().out
    assert str(output) in printed
    assert f"family spec hash: {spec_hash}" in printed
    edited = edited_cost_table(
        output, tmp_path / "cheaper-v3.json", name="adverse",
        slippage_bps_per_side_tier_two="1",
    )
    assert main([
        "carry-verify-measured", "--workspace-root", str(tmp_path),
        "--receipt", str(tmp_path / "receipt.json"), "--spec", str(edited),
        "--base-config", str(base),
    ]) == 2
    # A spec outside the workspace never reaches the verification.
    assert main([
        "carry-verify-measured", "--workspace-root", str(tmp_path),
        "--receipt", str(tmp_path / "receipt.json"), "--spec", str(CONFIG_V2.resolve()),
        "--base-config", str(base),
    ]) == 2


# --- Task 2: capital-declared reading (spec 5.1) --------------------------------------


def test_the_declared_v4_family_carries_the_measured_tiers_and_its_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The v4 base reads 500 (base) and 5 000 (adverse): ladder(500) and ten times it."""
    receipt, output, spec_hash = declared_v4_family(tmp_path, monkeypatch)
    spec, loaded_hash = load_family_spec(output)
    assert isinstance(spec, CarryFamilySpec)
    assert (spec_hash, len(spec_hash)) == (loaded_hash, 64)
    assert spec.family_name == "funding_carry_panel_v4_measured"
    assert tuple(item.name for item in spec.members) == MEMBER_NAMES_BY_FAMILY[
        "funding_carry_panel_v4"
    ]
    assert spec.costs.base.slippage_bps_per_side_tier_one == Decimal("800")
    assert spec.costs.base.slippage_bps_per_side_tier_two == Decimal("1200")
    assert spec.costs.adverse.slippage_bps_per_side_tier_one == Decimal("1600")
    assert spec.costs.adverse.slippage_bps_per_side_tier_two == Decimal("2400")
    v4_document = json.loads(CONFIG_V4.read_text(encoding="utf-8"))
    assert read_document(output)["capital"] == v4_document["capital"]
    assert spec.capital is not None
    assert spec.capital.per_leg_notional_usdt == Decimal("500")
    evidence = spec.cost_evidence
    assert evidence is not None
    assert evidence.receipt_hash == receipt["content_hash"]
    assert evidence.journal_spec_hash == receipt["spec_hash"]
    assert evidence.base_notional == Decimal("500")
    assert evidence.adverse_notional == Decimal("5000")
    assert evidence.rule == CAPITAL_DECLARATION_RULE


def test_a_genuine_v4_declaration_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, output, spec_hash = declared_v4_family(tmp_path, monkeypatch)
    verified = verify_measured_declaration(
        receipt_path=tmp_path / "receipt.json",
        spec_path=output,
        base_declaration_path=CONFIG_V4,
    )
    assert verified == spec_hash


def test_v4_verification_refuses_an_edited_slippage_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, output, _ = declared_v4_family(tmp_path, monkeypatch)
    edited = edited_cost_table(
        output, tmp_path / "cheaper-v4.json", name="base",
        slippage_bps_per_side_tier_one="799",
    )
    with pytest.raises(MeasuredCostError, match="slippage_bps_per_side_tier_one"):
        verify_measured_declaration(
            receipt_path=tmp_path / "receipt.json",
            spec_path=edited,
            base_declaration_path=CONFIG_V4,
        )


def test_v4_verification_refuses_a_cited_notional_the_rule_does_not_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, output, _ = declared_v4_family(tmp_path, monkeypatch)
    edited = edited_evidence(output, tmp_path / "other-notional-v4.json", base_notional="5000")
    with pytest.raises(MeasuredCostError, match="notionals the rule does not fix"):
        verify_measured_declaration(
            receipt_path=tmp_path / "receipt.json",
            spec_path=edited,
            base_declaration_path=CONFIG_V4,
        )


def test_v4_verification_refuses_the_v3_rule_text_in_a_v4_citation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The v3 sentence is a valid rule in general, but not the one this capital selects."""
    _, output, _ = declared_v4_family(tmp_path, monkeypatch)
    edited = edited_evidence(output, tmp_path / "v3-rule-v4.json", rule=DECLARATION_RULE)
    with pytest.raises(MeasuredCostError, match="capital does not select"):
        verify_measured_declaration(
            receipt_path=tmp_path / "receipt.json",
            spec_path=edited,
            base_declaration_path=CONFIG_V4,
        )


def test_a_v4_measured_declaration_whose_base_was_v2_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A measured declaration's family name must be its own base's mapped name."""
    _, output, _ = declared_v4_family(tmp_path, monkeypatch)
    with pytest.raises(MeasuredCostError, match="funding_carry_panel_v3 declaration"):
        verify_measured_declaration(
            receipt_path=tmp_path / "receipt.json",
            spec_path=output,
            base_declaration_path=CONFIG_V2,
        )


def test_the_cli_round_trips_the_v4_fixture_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """1 250 per leg lands on the same 5 000/50 000 rung `declared_family` exercises."""
    measured_receipt(tmp_path, monkeypatch)
    base = small_carry_v4_config(tmp_path)
    output = tmp_path / "funding-carry-panel-v4-measured.json"
    declare_arguments = [
        "carry-declare-measured", "--workspace-root", str(tmp_path),
        "--receipt", str(tmp_path / "receipt.json"), "--base-config", str(base),
        "--output", str(output),
    ]
    assert main(declare_arguments) == 0
    spec, spec_hash = load_family_spec(output)
    assert isinstance(spec, CarryFamilySpec)
    assert spec.family_name == "funding_carry_panel_v4_measured"
    printed = capsys.readouterr().out
    assert str(output) in printed
    assert f"family spec hash: {spec_hash}" in printed
    verify_arguments = [
        "carry-verify-measured", "--workspace-root", str(tmp_path),
        "--receipt", str(tmp_path / "receipt.json"), "--spec", str(output),
        "--base-config", str(base),
    ]
    assert main(verify_arguments) == 0
    printed = capsys.readouterr().out
    assert str(output) in printed
    assert f"family spec hash: {spec_hash}" in printed
