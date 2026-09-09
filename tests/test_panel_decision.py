import json
import random
from decimal import Decimal
from pathlib import Path

import pytest

from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.panel_config import load_panel_family_spec
from trading_bot.panel_decision import PanelDecisionError, build_panel_decision

SPEC_PATH = Path("configs/xs-momentum-panel-v1.json")
SPEC, SPEC_HASH = load_panel_family_spec(SPEC_PATH)
MEMBERS = tuple(item.name for item in SPEC.members)
CONTROLS = tuple(item.name for item in SPEC.controls)
EPISODES_PER_FOLD = 40


def episodes(sample_prefix: str, value: str, count: int) -> list[dict[str, object]]:
    return [
        {
            "sample_id": f"{sample_prefix}:{index}",
            "net_return": value,
            "gross_return": value,
            "turnover": "1",
            "trading_cost": "0",
            "funding_cost": "0",
            "forced_close_cost": "0",
            "gross_exposure": "1",
            "net_exposure": "0",
            "forced_close_count": 0,
            "contract_net_contributions": [
                ["A:0", str(Decimal(value) / 2)],
                ["B:0", str(Decimal(value) / 2)],
            ],
        }
        for index in range(count)
    ]


def gaussian_episode_values(
    count: int, *, target_mean: float, spread: float, seed: int
) -> list[str]:
    """A deterministic, genuinely noisy per-episode series (not a uniform or spike value)
    with its sample mean pinned exactly to `target_mean`, used to build a member whose
    evidence is real but statistically weak -- as opposed to the strongly one-sided profiles
    every other fixture in this file uses. `random.Random(seed).gauss` is stable across
    Python 3 versions, so this is exactly reproducible."""
    generator = random.Random(seed)
    raw = [generator.gauss(target_mean, spread) for _ in range(count)]
    shift = target_mean - (sum(raw) / count)
    return [str(round(value + shift, 6)) for value in raw]


def three_positive_three_negative_folds(high: str, low: str) -> list[str]:
    """240 episode values: the first three folds (120 episodes) at `high`, the last three
    at `low` -- a fold-uniform split used only to hand a member a chosen, tunable bootstrap
    p-value (via the high/low contrast) without caring about its own economic gates."""
    return ([high] * (EPISODES_PER_FOLD * 3)) + ([low] * (EPISODES_PER_FOLD * 3))


def episodes_with_values(sample_prefix: str, values: list[str]) -> list[dict[str, object]]:
    """Like `episodes`, but each episode carries its own value from `values` instead of one
    value repeated uniformly `count` times."""
    return [
        {
            "sample_id": f"{sample_prefix}:{index}",
            "net_return": value,
            "gross_return": value,
            "turnover": "1",
            "trading_cost": "0",
            "funding_cost": "0",
            "forced_close_cost": "0",
            "gross_exposure": "1",
            "net_exposure": "0",
            "forced_close_count": 0,
            "contract_net_contributions": [
                ["A:0", str(Decimal(value) / 2)],
                ["B:0", str(Decimal(value) / 2)],
            ],
        }
        for index, value in enumerate(values)
    ]


def write_fold(
    path: Path,
    fold_index: int,
    returns: dict[str, tuple[str, str]],
    *,
    episodes_per_fold: int = EPISODES_PER_FOLD,
) -> None:
    candidates = []
    for name in MEMBERS + CONTROLS:
        base_value, adverse_value = returns.get(name, ("0", "0"))
        candidates.append(
            {
                "candidate_name": name,
                "role": "member" if name in MEMBERS else "control",
                "episode_count": episodes_per_fold,
                "base": {
                    "total_net_return": str(
                        Decimal(base_value) * Decimal(episodes_per_fold)
                    ),
                    "episodes": episodes(f"f{fold_index}", base_value, episodes_per_fold),
                },
                "adverse": {
                    "total_net_return": str(
                        Decimal(adverse_value) * Decimal(episodes_per_fold)
                    ),
                    "episodes": episodes(f"f{fold_index}", adverse_value, episodes_per_fold),
                },
            }
        )
    material = {
        "report_version": "1.0.0",
        "status": "development_only",
        "reason_codes": [],
        "family_name": SPEC.family_name,
        "family_spec_hash": SPEC_HASH,
        "capture_root_hash": "c" * 64,
        "dataset_root_hash": "d" * 64,
        "split_manifest_hash": "e" * 64,
        "manifest_hash": "f" * 64,
        "fold_index": fold_index,
        "fold_count": 6,
        "train_sample_count": 10,
        "validation_sample_count": 10,
        "test_sample_count": episodes_per_fold,
        "train_membership_hash": "0" * 64,
        "validation_membership_hash": "1" * 64,
        "test_membership_hash": "2" * 64,
        "random_seed": 17,
        "block_length": 4,
        "bootstrap_repetitions": 2000,
        "skipped_sample_ids": [],
        "code_hash": "3" * 64,
        "candidates": candidates,
    }
    document = dict(material)
    document["report_hash"] = content_sha256(material)
    path.write_bytes(canonical_json(document))


def write_fold_with_episodes(
    path: Path,
    fold_index: int,
    base_episode_values: dict[str, list[str]],
    adverse_episode_values: dict[str, list[str]],
) -> None:
    """Like `write_fold`, but each candidate's episodes within this one fold can each carry
    their own value, instead of one value uniform across the whole fold. Any candidate not
    present in the mapping falls back to an all-zero fold, matching `write_fold`'s default."""
    candidates = []
    for name in MEMBERS + CONTROLS:
        base_values = base_episode_values.get(name, ["0"] * EPISODES_PER_FOLD)
        adverse_values = adverse_episode_values.get(name, ["0"] * EPISODES_PER_FOLD)
        candidates.append(
            {
                "candidate_name": name,
                "role": "member" if name in MEMBERS else "control",
                "episode_count": len(base_values),
                "base": {
                    "total_net_return": str(
                        sum((Decimal(value) for value in base_values), Decimal(0))
                    ),
                    "episodes": episodes_with_values(f"f{fold_index}", base_values),
                },
                "adverse": {
                    "total_net_return": str(
                        sum((Decimal(value) for value in adverse_values), Decimal(0))
                    ),
                    "episodes": episodes_with_values(f"f{fold_index}", adverse_values),
                },
            }
        )
    material = {
        "report_version": "1.0.0",
        "status": "development_only",
        "reason_codes": [],
        "family_name": SPEC.family_name,
        "family_spec_hash": SPEC_HASH,
        "capture_root_hash": "c" * 64,
        "dataset_root_hash": "d" * 64,
        "split_manifest_hash": "e" * 64,
        "manifest_hash": "f" * 64,
        "fold_index": fold_index,
        "fold_count": 6,
        "train_sample_count": 10,
        "validation_sample_count": 10,
        "test_sample_count": EPISODES_PER_FOLD,
        "train_membership_hash": "0" * 64,
        "validation_membership_hash": "1" * 64,
        "test_membership_hash": "2" * 64,
        "random_seed": 17,
        "block_length": 4,
        "bootstrap_repetitions": 2000,
        "skipped_sample_ids": [],
        "code_hash": "3" * 64,
        "candidates": candidates,
    }
    document = dict(material)
    document["report_hash"] = content_sha256(material)
    path.write_bytes(canonical_json(document))


def build(
    tmp_path: Path,
    returns: dict[str, tuple[str, str]],
    *,
    episodes_per_fold: int = EPISODES_PER_FOLD,
) -> Path:
    paths = []
    for fold_index in range(6):
        path = tmp_path / f"fold{fold_index}.json"
        write_fold(path, fold_index, returns, episodes_per_fold=episodes_per_fold)
        paths.append(path)
    artifact = build_panel_decision(
        tuple(paths),
        family_spec_path=SPEC_PATH,
        output_path=tmp_path / "decision.json",
        registry_path=tmp_path / "registry.sqlite3",
    )
    return artifact.output_path


def build_with_fold_returns(
    tmp_path: Path, per_fold_returns: dict[int, dict[str, tuple[str, str]]]
) -> Path:
    """Like `build`, but each fold index can declare its own per-candidate (base, adverse)
    pair, instead of the same pair applying uniformly to every fold. A fold index absent
    from `per_fold_returns` gets an all-zero fold for every candidate."""
    paths = []
    for fold_index in range(6):
        path = tmp_path / f"fold{fold_index}.json"
        write_fold(path, fold_index, per_fold_returns.get(fold_index, {}))
        paths.append(path)
    artifact = build_panel_decision(
        tuple(paths),
        family_spec_path=SPEC_PATH,
        output_path=tmp_path / "decision.json",
        registry_path=tmp_path / "registry.sqlite3",
    )
    return artifact.output_path


def build_with_fold_episodes(
    tmp_path: Path,
    per_fold_base_values: dict[int, dict[str, list[str]]],
    per_fold_adverse_values: dict[int, dict[str, list[str]]],
) -> Path:
    """Like `build_with_fold_returns`, but with per-episode (not just per-fold-uniform)
    control, for fixtures that need specific episodes to carry specific values (e.g. a
    single concentrated spike) rather than a single value repeated across the fold."""
    paths = []
    for fold_index in range(6):
        path = tmp_path / f"fold{fold_index}.json"
        write_fold_with_episodes(
            path,
            fold_index,
            per_fold_base_values.get(fold_index, {}),
            per_fold_adverse_values.get(fold_index, {}),
        )
        paths.append(path)
    artifact = build_panel_decision(
        tuple(paths),
        family_spec_path=SPEC_PATH,
        output_path=tmp_path / "decision.json",
        registry_path=tmp_path / "registry.sqlite3",
    )
    return artifact.output_path


def test_a_flat_family_is_rejected_without_eligible_members(tmp_path: Path) -> None:
    output_path = build(tmp_path, {})
    document = json.loads(output_path.read_text(encoding="utf-8"))
    assert document["decision_status"] == "no_eligible_member"
    momentum = next(
        item for item in document["members"] if item["candidate_name"] == "xs_mom_4w"
    )
    assert "AGGREGATE_BASE_NET_NON_POSITIVE" in momentum["reason_codes"]


def test_negative_adverse_is_rejected(tmp_path: Path) -> None:
    output_path = build(tmp_path, {"xs_mom_4w": ("0.01", "-0.001")})
    document = json.loads(output_path.read_text(encoding="utf-8"))
    momentum = next(
        item for item in document["members"] if item["candidate_name"] == "xs_mom_4w"
    )
    assert momentum["decision_status"] == "rejected"
    assert "AGGREGATE_ADVERSE_NET_NON_POSITIVE" in momentum["reason_codes"]


def test_episode_floor_is_rejected_in_isolation(tmp_path: Path) -> None:
    # Every candidate gets the same reduced episode count (30/fold * 6 = 180 < the 200
    # floor), so the floor check fails while every other check is still fed a clean,
    # otherwise-passing profile: uniform positive base, uniform positive adverse, all six
    # folds individually positive, tiny bootstrap p-value against five flat (zero) peers,
    # no concentration (uniform value spreads evenly across folds/contracts/episodes), and
    # totals above the (default zero) controls.
    output_path = build(tmp_path, {"xs_mom_4w": ("0.01", "0.001")}, episodes_per_fold=30)
    document = json.loads(output_path.read_text(encoding="utf-8"))
    momentum = next(
        item for item in document["members"] if item["candidate_name"] == "xs_mom_4w"
    )
    assert momentum["decision_status"] == "insufficient_evidence"
    assert momentum["reason_codes"] == ["EPISODE_FLOOR_NOT_MET"]


def test_positive_fold_fraction_is_rejected_in_isolation(tmp_path: Path) -> None:
    # Three folds strongly positive (0.05/episode) and three mildly negative (-0.01/
    # episode) keep the pooled base mean clearly positive (0.02) and the bootstrap lower
    # bound clearly positive, but only 3 of 6 folds are individually positive against a
    # requirement of ceil(2/3 * 6) = 4 -- failing only the fold-fraction check. The largest
    # single fold holds 50/120 = 41.7% of pooled base PnL, under the 50% concentration
    # limit, and the uniform positive adverse keeps both the adverse and dominance checks
    # clean.
    output_path = build_with_fold_returns(
        tmp_path,
        {
            0: {"xs_mom_4w": ("0.05", "0.001")},
            1: {"xs_mom_4w": ("0.05", "0.001")},
            2: {"xs_mom_4w": ("0.05", "0.001")},
            3: {"xs_mom_4w": ("-0.01", "0.001")},
            4: {"xs_mom_4w": ("-0.01", "0.001")},
            5: {"xs_mom_4w": ("-0.01", "0.001")},
        },
    )
    document = json.loads(output_path.read_text(encoding="utf-8"))
    momentum = next(
        item for item in document["members"] if item["candidate_name"] == "xs_mom_4w"
    )
    assert momentum["decision_status"] == "rejected"
    assert momentum["reason_codes"] == ["POSITIVE_FOLD_FRACTION_NOT_MET"]


def test_multiple_testing_gate_is_rejected_in_isolation(tmp_path: Path) -> None:
    # xs_mom_4w gets a genuinely noisy (not uniform or spiked) series with mean=0.004 and
    # spread=0.027 (seed 41): a positive base mean, a positive bootstrap lower bound
    # (0.00029), every fold individually positive, and no concentration (the noise spreads
    # unevenly across folds by chance, but the largest single fold still holds only 30% of
    # the pooled base total) -- every check passes except one. Its bootstrap one-sided
    # p-value is ~0.021: real, and (by construction) the single smallest -- most
    # significant-looking -- of the six raw p-values, since the other five members are each
    # given a deliberately weaker three-folds-up/three-folds-down split (several of them
    # fail their own bootstrap and concentration checks, which is irrelevant here, since
    # only their raw p-value feeds the shared correction). Benjamini-Hochberg's correction
    # for a family of six means even the *smallest* p-value only survives the 0.10 gate if
    # it is below roughly 0.10/6 = 0.0167; xs_mom_4w's ~0.021 narrowly misses that bar even
    # at the best possible rank, so BH reports q=0.1199, over the gate. No weaker peer is
    # "borrowing" against xs_mom_4w here -- being the strongest of six is still not strong
    # enough once six simultaneous trials are corrected for.
    probed_base = gaussian_episode_values(240, target_mean=0.004, spread=0.027, seed=41)
    probed_adverse = ["0.0004"] * 240
    peer_low_values = {
        "xs_mom_1w": "-0.00065",
        "xs_mom_12w": "-0.00075",
        "ts_mom_4w": "-0.00082",
        "ts_mom_12w": "-0.00088",
        "xs_rev_1w": "-0.00095",
    }
    peer_series = {
        name: three_positive_three_negative_folds("0.001", low)
        for name, low in peer_low_values.items()
    }
    per_fold_base: dict[int, dict[str, list[str]]] = {index: {} for index in range(6)}
    per_fold_adverse: dict[int, dict[str, list[str]]] = {index: {} for index in range(6)}
    for fold_index in range(6):
        window = slice(fold_index * EPISODES_PER_FOLD, (fold_index + 1) * EPISODES_PER_FOLD)
        per_fold_base[fold_index]["xs_mom_4w"] = probed_base[window]
        per_fold_adverse[fold_index]["xs_mom_4w"] = probed_adverse[window]
        for name, series in peer_series.items():
            per_fold_base[fold_index][name] = series[window]
    output_path = build_with_fold_episodes(tmp_path, per_fold_base, per_fold_adverse)
    document = json.loads(output_path.read_text(encoding="utf-8"))
    momentum = next(
        item for item in document["members"] if item["candidate_name"] == "xs_mom_4w"
    )
    assert momentum["decision_status"] == "rejected"
    assert momentum["reason_codes"] == ["MULTIPLE_TESTING_GATE_NOT_MET"]


def test_concentration_limit_is_rejected_in_isolation(tmp_path: Path) -> None:
    # One fold (100) dwarfs the other five (10, 10, 10, 10, -20), so it alone holds
    # 100/120 = 83.3% of the pooled base total -- over the 50% limit -- while five of six
    # folds are still individually positive (clearing the fold-fraction floor of 4), the
    # pooled mean (0.5) and bootstrap lower bound are clearly positive, and the uniform
    # positive adverse keeps every other check clean.
    output_path = build_with_fold_returns(
        tmp_path,
        {
            0: {"xs_mom_4w": ("2.5", "0.001")},
            1: {"xs_mom_4w": ("0.25", "0.001")},
            2: {"xs_mom_4w": ("0.25", "0.001")},
            3: {"xs_mom_4w": ("0.25", "0.001")},
            4: {"xs_mom_4w": ("0.25", "0.001")},
            5: {"xs_mom_4w": ("-0.5", "0.001")},
        },
    )
    document = json.loads(output_path.read_text(encoding="utf-8"))
    momentum = next(
        item for item in document["members"] if item["candidate_name"] == "xs_mom_4w"
    )
    assert momentum["decision_status"] == "rejected"
    assert momentum["reason_codes"] == ["CONCENTRATION_LIMIT_EXCEEDED"]


def test_base_control_dominance_is_rejected_in_isolation(tmp_path: Path) -> None:
    # xs_mom_4w gets the same clean uniform profile as the success path (positive mean,
    # positive lower bound, positive folds, tiny p, no concentration), but `no_trade` is
    # given a stronger base return (0.02 vs 0.01/episode) than the member -- so the member
    # fails only the base dominance check. `no_trade`'s adverse stays at the default zero,
    # so it does not threaten the member's adverse dominance.
    output_path = build(
        tmp_path, {"xs_mom_4w": ("0.01", "0.001"), "no_trade": ("0.02", "0.0")}
    )
    document = json.loads(output_path.read_text(encoding="utf-8"))
    momentum = next(
        item for item in document["members"] if item["candidate_name"] == "xs_mom_4w"
    )
    assert momentum["decision_status"] == "rejected"
    assert momentum["reason_codes"] == ["BASE_CONTROL_DOMINANCE_NOT_MET"]


def test_adverse_control_dominance_is_rejected_in_isolation(tmp_path: Path) -> None:
    # xs_mom_4w again gets the clean uniform profile, but `random_ranks` is given a
    # stronger adverse return (0.01 vs the member's 0.001/episode) while keeping its base
    # return at the default zero (well below the member's base total) -- so the member
    # fails only the adverse dominance check.
    output_path = build(
        tmp_path, {"xs_mom_4w": ("0.01", "0.001"), "random_ranks": ("0", "0.01")}
    )
    document = json.loads(output_path.read_text(encoding="utf-8"))
    momentum = next(
        item for item in document["members"] if item["candidate_name"] == "xs_mom_4w"
    )
    assert momentum["decision_status"] == "rejected"
    assert momentum["reason_codes"] == ["ADVERSE_CONTROL_DOMINANCE_NOT_MET"]


def test_member_clearing_every_gate_is_eligible(tmp_path: Path) -> None:
    # The true-positive path: a genuinely positive pooled base mean (0.01/episode) with a
    # positive bootstrap lower bound, a non-negative adverse mean (0.001/episode), every
    # fold individually positive (6 of 6, clearing the ceil(2/3 * 6) = 4 requirement), a
    # tiny bootstrap p-value against five flat (zero) peers so its Benjamini-Hochberg q
    # stays far under the 0.10 gate, an even 50/40/tiny spread across contracts/folds/
    # episodes well under the 50% concentration limit, and totals strictly above both
    # (default zero) controls on both scenarios.
    output_path = build(tmp_path, {"xs_mom_4w": ("0.01", "0.001")})
    document = json.loads(output_path.read_text(encoding="utf-8"))
    momentum = next(
        item for item in document["members"] if item["candidate_name"] == "xs_mom_4w"
    )
    assert momentum["decision_status"] == "eligible_for_further_review"
    assert momentum["reason_codes"] == []
    assert document["decision_status"] == "eligible_member_available"
    assert "xs_mom_4w" in document["eligible_member_names"]


def test_pooled_counts_and_hashes_are_recorded(tmp_path: Path) -> None:
    output_path = build(tmp_path, {})
    document = json.loads(output_path.read_text(encoding="utf-8"))
    assert document["pooled_episode_count"] == 6 * EPISODES_PER_FOLD
    assert document["fold_count"] == 6
    assert len(document["source_report_hashes"]) == 6
    assert document["family_spec_hash"] == SPEC_HASH


def test_duplicate_fold_indices_are_rejected(tmp_path: Path) -> None:
    write_fold(tmp_path / "a.json", 0, {})
    write_fold(tmp_path / "b.json", 0, {})
    with pytest.raises(PanelDecisionError):
        build_panel_decision(
            (tmp_path / "a.json", tmp_path / "b.json"),
            family_spec_path=SPEC_PATH,
            output_path=tmp_path / "decision.json",
            registry_path=tmp_path / "registry.sqlite3",
        )


def test_missing_fold_index_is_rejected(tmp_path: Path) -> None:
    paths = []
    for fold_index in range(6):
        path = tmp_path / f"fold{fold_index}.json"
        write_fold(path, fold_index, {})
        paths.append(path)
    incomplete = tuple(path for index, path in enumerate(paths) if index != 3)
    with pytest.raises(PanelDecisionError) as excinfo:
        build_panel_decision(
            incomplete,
            family_spec_path=SPEC_PATH,
            output_path=tmp_path / "decision.json",
            registry_path=tmp_path / "registry.sqlite3",
        )
    assert "missing fold index(es) 3" in str(excinfo.value)


def test_pooled_episode_count_mismatch_is_rejected(tmp_path: Path) -> None:
    # Only xs_mom_4w is given a shorter per-fold episode list (30 instead of the
    # default EPISODES_PER_FOLD=40); every other candidate falls back to the default
    # 40/fold. That makes xs_mom_4w's pooled episode count (180) disagree with every
    # other candidate's (240), which the shared-episode-count invariant must reject
    # before any economic gate is evaluated.
    per_fold_base = {index: {"xs_mom_4w": ["0.01"] * 30} for index in range(6)}
    per_fold_adverse = {index: {"xs_mom_4w": ["0.001"] * 30} for index in range(6)}
    with pytest.raises(PanelDecisionError, match="pooled episode count"):
        build_with_fold_episodes(tmp_path, per_fold_base, per_fold_adverse)


def test_tampered_report_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "fold0.json"
    write_fold(path, 0, {})
    document = json.loads(path.read_text(encoding="utf-8"))
    document["fold_index"] = 1
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(PanelDecisionError):
        build_panel_decision(
            (path,),
            family_spec_path=SPEC_PATH,
            output_path=tmp_path / "decision.json",
            registry_path=tmp_path / "registry.sqlite3",
        )
