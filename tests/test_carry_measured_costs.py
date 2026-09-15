"""The carry family declared from the Binance cost journal's receipt (spec 5)."""

import json
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.test_binance_cost_journal import (
    LadderVenue,
    finalized_receipt,
    journal_with_rounds,
    leg_document,
    lower_the_eligibility_floors,
    read_document,
)
from trading_bot.binance_cost_journal import DECLARATION_RULE
from trading_bot.carry_config import (
    MEMBER_NAMES_BY_FAMILY,
    CarryFamilySpec,
    load_carry_family_spec,
)
from trading_bot.carry_measured_costs import (
    MeasuredCostError,
    declare_measured_cost_family,
    measured_slippage_tiers,
)
from trading_bot.cli import main
from trading_bot.panel_config import load_family_spec

CONFIG_V1 = Path("configs/funding-carry-panel-v1.json")
CONFIG_V2 = Path("configs/funding-carry-panel-v2.json")
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
    _, output, _ = declared_family(tmp_path, monkeypatch)
    published = output.read_bytes()
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
