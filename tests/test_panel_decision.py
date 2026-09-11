import json
import random
from collections.abc import Callable
from decimal import Context, Decimal, Inexact
from pathlib import Path

import pytest

import trading_bot.panel_decision as panel_decision_module
from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.carry_config import CONTROL_NAMES as CARRY_CONTROLS
from trading_bot.carry_config import MEMBER_NAMES as CARRY_MEMBERS
from trading_bot.carry_config import load_carry_family_spec
from trading_bot.evaluation import _maximum_drawdown, evaluate_signals
from trading_bot.panel_config import load_panel_family_spec
from trading_bot.panel_decision import PanelDecisionError, _pool, _win_rate, build_panel_decision
from trading_bot.panel_fold_run import MEMBER_HELD_NOTHING_REASON_CODE
from trading_bot.strategy import CostScenario

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


def test_context_control_does_not_enter_the_dominance_baseline(tmp_path: Path) -> None:
    # `passive_long_ew` is the third, contextual benchmark control (kind
    # "passive_long"), not one of the two "no active view" dominance
    # controls (no_trade, random_ranks). Giving it a dominant base and
    # adverse return alongside the same clean, otherwise-eligible member
    # profile `test_member_clearing_every_gate_is_eligible` uses must leave
    # the member's eligibility untouched -- a regression back to selecting
    # dominance controls by position (e.g. `control_names[:3]`, or any
    # selection that includes the context control) would instead reject it
    # via BASE_CONTROL_DOMINANCE_NOT_MET / ADVERSE_CONTROL_DOMINANCE_NOT_MET.
    output_path = build(
        tmp_path,
        {"xs_mom_4w": ("0.01", "0.001"), "passive_long_ew": ("0.05", "0.05")},
    )
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
    assert document["pooled_raw_episode_count"] == 6 * EPISODES_PER_FOLD
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


def write_fold_with_held_nothing(
    path: Path,
    fold_index: int,
    returns: dict[str, tuple[str, str]],
    *,
    held_nothing_candidate: str,
    held_nothing_index: int = 0,
) -> None:
    """Like `write_fold`, but one episode of `held_nothing_candidate`, in both
    scenarios, is marked the way `panel_fold_run.py` marks a week where a member's
    own construction produced no weights while the universe was not too small."""
    candidates = []
    for name in MEMBERS + CONTROLS:
        base_value, adverse_value = returns.get(name, ("0", "0"))
        base_episodes = episodes(f"f{fold_index}", base_value, EPISODES_PER_FOLD)
        adverse_episodes = episodes(f"f{fold_index}", adverse_value, EPISODES_PER_FOLD)
        if name == held_nothing_candidate:
            base_episodes[held_nothing_index]["reason_codes"] = [MEMBER_HELD_NOTHING_REASON_CODE]
            adverse_episodes[held_nothing_index]["reason_codes"] = [
                MEMBER_HELD_NOTHING_REASON_CODE
            ]
        candidates.append(
            {
                "candidate_name": name,
                "role": "member" if name in MEMBERS else "control",
                "episode_count": EPISODES_PER_FOLD,
                "base": {
                    "total_net_return": str(
                        Decimal(base_value) * Decimal(EPISODES_PER_FOLD)
                    ),
                    "episodes": base_episodes,
                },
                "adverse": {
                    "total_net_return": str(
                        Decimal(adverse_value) * Decimal(EPISODES_PER_FOLD)
                    ),
                    "episodes": adverse_episodes,
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


def test_member_held_nothing_episode_is_excluded_but_its_cost_is_rolled_forward(
    tmp_path: Path,
) -> None:
    # Fold 0 marks xs_mom_4w's episode 5 of 40 (not the first, not the last, so a
    # retained episode exists both before and after it) MEMBER_HELD_NOTHING in both
    # scenarios; every other fold and every other candidate is a plain uniform
    # profile. Round 1 review: excluding the marked episode by simply dropping it
    # silently erased its real net_return from the pooled total, an always-
    # favourable bias, since a real re-entry episode right after it was still kept.
    # After the fix, the marked episode still stops counting as an observation, but
    # its net_return and turnover are rolled into the next retained episode in the
    # same fold, so the pooled total equals the raw total exactly -- no money
    # disappears -- and the roll-forward is itself visible in the new
    # held_nothing_episode_count / net_return_rolled_forward fields.
    paths = []
    for fold_index in range(6):
        path = tmp_path / f"fold{fold_index}.json"
        if fold_index == 0:
            write_fold_with_held_nothing(
                path,
                fold_index,
                {"xs_mom_4w": ("0.01", "0.001")},
                held_nothing_candidate="xs_mom_4w",
                held_nothing_index=5,
            )
        else:
            write_fold(path, fold_index, {"xs_mom_4w": ("0.01", "0.001")})
        paths.append(path)
    artifact = build_panel_decision(
        tuple(paths),
        family_spec_path=SPEC_PATH,
        output_path=tmp_path / "decision.json",
        registry_path=tmp_path / "registry.sqlite3",
    )
    document = json.loads(artifact.output_path.read_text(encoding="utf-8"))
    momentum = next(
        item for item in document["members"] if item["candidate_name"] == "xs_mom_4w"
    )
    # One fewer counted observation than the raw episode count...
    assert momentum["pooled_retained_episode_count"] == 6 * EPISODES_PER_FOLD - 1
    # ...but the marked episode's money is still fully present in both pooled
    # totals -- equal to the *full* raw total, not short by the excluded episode.
    assert momentum["base_total_net_return"] == str(Decimal("0.01") * 6 * EPISODES_PER_FOLD)
    assert momentum["adverse_total_net_return"] == str(Decimal("0.001") * 6 * EPISODES_PER_FOLD)
    assert momentum["base_held_nothing_episode_count"] == 1
    assert momentum["adverse_held_nothing_episode_count"] == 1
    assert momentum["base_held_nothing_net_return_rolled_forward"] == "0.01"
    assert momentum["adverse_held_nothing_net_return_rolled_forward"] == "0.001"
    # Turnover is rolled forward the same way: 240 raw episodes each charge "1" of
    # turnover, so the true total is 240 even though only 239 remain as their own
    # observation -- the reported mean turnover must reflect that real total, not
    # merely divide the 239 individually-recorded values.
    assert Decimal(momentum["base_mean_turnover"]) == Decimal(240) / Decimal(239)
    # The top-level pooled count reports the shared raw decision calendar, not any
    # one member's post-exclusion count.
    assert document["pooled_raw_episode_count"] == 6 * EPISODES_PER_FOLD
    # An untouched candidate's own pooled count is unaffected.
    control = next(item for item in document["controls"] if item["candidate_name"] == "no_trade")
    assert control["pooled_retained_episode_count"] == 6 * EPISODES_PER_FOLD


def _alternating(count: int, *, low: str, high: str) -> list[str]:
    half = count // 2
    return [low] * half + [high] * (count - half)


def test_member_record_reports_the_spec_8_4_fields(tmp_path: Path) -> None:
    # xs_mom_4w gets the same 40-value alternating base pattern in every one of the
    # six folds (20 episodes at "-0.01", 20 at "0.02") and a uniformly positive
    # adverse return; every other candidate stays at the default all-zero profile.
    # Every quantity asserted below is independently hand-derived (or, for maximum
    # drawdown, cross-checked against the exact function `evaluation.py` uses) from
    # that same fixture, so this pins the wiring of every section 8.4 field this
    # item adds, not just their presence.
    base_pattern = _alternating(EPISODES_PER_FOLD, low="-0.01", high="0.02")
    adverse_pattern = ["0.001"] * EPISODES_PER_FOLD
    per_fold_base = {index: {"xs_mom_4w": base_pattern} for index in range(6)}
    per_fold_adverse = {index: {"xs_mom_4w": adverse_pattern} for index in range(6)}
    output_path = build_with_fold_episodes(tmp_path, per_fold_base, per_fold_adverse)
    document = json.loads(output_path.read_text(encoding="utf-8"))
    momentum = next(
        item for item in document["members"] if item["candidate_name"] == "xs_mom_4w"
    )

    assert momentum["pooled_retained_episode_count"] == 6 * EPISODES_PER_FOLD
    # 120 episodes at -0.01 and 120 at 0.02: the two middle (sorted) values straddle
    # the boundary between the two groups, so the median is their average.
    assert Decimal(momentum["base_median_net_return"]) == (Decimal("-0.01") + Decimal("0.02")) / 2
    assert Decimal(momentum["base_win_rate"]) == Decimal("120") / Decimal("240")
    assert momentum["base_mean_turnover"] == "1"
    assert momentum["base_mean_gross_exposure"] == "1"
    assert momentum["base_mean_net_exposure"] == "0"
    assert momentum["base_forced_close_count"] == 0
    pooled_base_series = [Decimal(value) for value in base_pattern] * 6
    assert Decimal(momentum["base_maximum_drawdown"]) == _maximum_drawdown(pooled_base_series)

    # The adverse series is uniformly positive: every episode is a "win", the
    # median equals the constant, and there is never a drawdown.
    assert momentum["adverse_median_net_return"] == "0.001"
    assert momentum["adverse_win_rate"] == "1"
    assert momentum["adverse_maximum_drawdown"] == "0"
    assert momentum["adverse_mean_turnover"] == "1"
    assert momentum["adverse_forced_close_count"] == 0

    # No week in this fixture is UNIVERSE_TOO_SMALL.
    assert momentum["universe_too_small_week_count"] == 0


def test_universe_too_small_week_count_reflects_skipped_samples(tmp_path: Path) -> None:
    paths = []
    for fold_index in range(6):
        path = tmp_path / f"fold{fold_index}.json"
        write_fold(path, fold_index, {})
        if fold_index == 0:
            document = json.loads(path.read_text(encoding="utf-8"))
            document["skipped_sample_ids"] = ["BINANCE_UM:1:w1", "BINANCE_UM:2:w1"]
            document["reason_codes"] = ["SKIPPED_WEEK_EXIT_COST_UNCHARGED"]
            material = {k: v for k, v in document.items() if k != "report_hash"}
            document["report_hash"] = content_sha256(material)
            path.write_bytes(canonical_json(document))
        paths.append(path)
    artifact = build_panel_decision(
        tuple(paths),
        family_spec_path=SPEC_PATH,
        output_path=tmp_path / "decision.json",
        registry_path=tmp_path / "registry.sqlite3",
    )
    document = json.loads(artifact.output_path.read_text(encoding="utf-8"))
    assert document["skipped_sample_ids"] == ["BINANCE_UM:1:w1", "BINANCE_UM:2:w1"]
    momentum = next(
        item for item in document["members"] if item["candidate_name"] == "xs_mom_4w"
    )
    assert momentum["universe_too_small_week_count"] == 2


def test_declared_deviations_are_reported(tmp_path: Path) -> None:
    output_path = build(tmp_path, {})
    document = json.loads(output_path.read_text(encoding="utf-8"))
    deviations = document["declared_deviations"]
    assert len(deviations) >= 2
    ids = {item["id"] for item in deviations}
    assert "RANKABLE_SUBSET_NOT_FULL_UNIVERSE" in ids
    assert "MEMBER_HELD_NOTHING_EXCLUDED_FROM_POOLED_SERIES" in ids
    for item in deviations:
        assert item["description"]


def test_code_hash_mismatch_across_fold_reports_is_rejected(tmp_path: Path) -> None:
    # A family half-run under one version of panel_fold_run.py and half under
    # another (e.g. before vs. after MEMBER_HELD_NOTHING marking existed) would
    # apply the pooled exclusion to some folds only, silently, with no other
    # check catching it -- the code_hash linkage check must refuse it outright.
    paths = []
    for fold_index in range(6):
        path = tmp_path / f"fold{fold_index}.json"
        write_fold(path, fold_index, {})
        paths.append(path)
    document = json.loads(paths[3].read_text(encoding="utf-8"))
    document["code_hash"] = "9" * 64
    material = {key: value for key, value in document.items() if key != "report_hash"}
    document["report_hash"] = content_sha256(material)
    paths[3].write_bytes(canonical_json(document))

    with pytest.raises(PanelDecisionError, match="code_hash mismatch"):
        build_panel_decision(
            tuple(paths),
            family_spec_path=SPEC_PATH,
            output_path=tmp_path / "decision.json",
            registry_path=tmp_path / "registry.sqlite3",
        )


def test_win_rate_matches_evaluate_signals_across_series() -> None:
    """Pins agreement between panel_decision._win_rate and evaluation.py's own
    inline win-rate computation inside evaluate_signals (there was nothing
    importable, since evaluate_signals computes it inline), including the tie
    convention -- a value of exactly zero is never a win in either -- so the
    bar-cadence and panel research lines cannot silently drift apart on what
    "win rate" means. A zero-cost, always-active scenario makes
    evaluate_signals's net_returns equal the input series exactly."""
    scenario = CostScenario(
        name="zero_cost",
        fee_bps_per_side=Decimal(0),
        spread_multiplier=Decimal(0),
        slippage_bps_per_side=Decimal(0),
        funding_bps=Decimal(0),
    )
    series_cases: list[tuple[Decimal, ...]] = [
        (),
        (Decimal("0.01"),),
        (Decimal("-0.01"), Decimal("0.02"), Decimal("0.03")),
        (Decimal("0"), Decimal("0"), Decimal("0.01")),
        (Decimal("-1"), Decimal("-2"), Decimal("3"), Decimal("4"), Decimal("-5")),
    ]
    for series in series_cases:
        evaluation = evaluate_signals(
            tuple(1 for _ in series),
            series,
            tuple(Decimal(0) for _ in series),
            scenario,
        )
        assert _win_rate(series) == evaluation.win_rate


def test_pooling_precision_headroom_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    """The "never itself needs to round" guarantee behind exact pooled totals
    is only real if exceeding it is loud. Shrinking _POOLING_CONTEXT to one
    significant digit of precision (Inexact still trapped) turns the very
    first addition of two two-digit values into a real rounding event, and
    that must now raise PanelDecisionError instead of silently returning a
    wrong total."""
    insufficient = Context(prec=1)
    insufficient.traps[Inexact] = True
    monkeypatch.setattr(panel_decision_module, "_POOLING_CONTEXT", insufficient)

    document: dict[str, object] = {
        "candidates": [
            {
                "candidate_name": "xs_mom_4w",
                "base": {"episodes": episodes_with_values("f0", ["0.12", "0.34"])},
                "adverse": {"episodes": episodes_with_values("f0", ["0", "0"])},
            }
        ]
    }
    with pytest.raises(PanelDecisionError, match="precision headroom"):
        _pool([document], "xs_mom_4w")


def test_folds_where_a_member_retained_nothing_are_reported(tmp_path: Path) -> None:
    # Fold 2 marks every one of xs_mom_4w's 40 episodes MEMBER_HELD_NOTHING in
    # both scenarios -- the fold-retains-nothing fallback R1's backward roll
    # cannot fix, since there is no retained episode in that fold to roll onto.
    # That fold must still be visible: a per-fold retained-episode-count list
    # and a distinct list naming which fold(s) retained zero.
    all_held_nothing = episodes_with_values("f2", ["0.01"] * EPISODES_PER_FOLD)
    for episode in all_held_nothing:
        episode["reason_codes"] = [MEMBER_HELD_NOTHING_REASON_CODE]
    paths = []
    for fold_index in range(6):
        path = tmp_path / f"fold{fold_index}.json"
        if fold_index == 2:
            write_fold_with_episodes(
                path,
                fold_index,
                {},
                {},
            )
            document = json.loads(path.read_text(encoding="utf-8"))
            for candidate in document["candidates"]:
                if candidate["candidate_name"] == "xs_mom_4w":
                    candidate["base"]["episodes"] = all_held_nothing
                    candidate["adverse"]["episodes"] = all_held_nothing
            material = {key: value for key, value in document.items() if key != "report_hash"}
            document["report_hash"] = content_sha256(material)
            path.write_bytes(canonical_json(document))
        else:
            write_fold(path, fold_index, {"xs_mom_4w": ("0.01", "0.001")})
        paths.append(path)

    artifact = build_panel_decision(
        tuple(paths),
        family_spec_path=SPEC_PATH,
        output_path=tmp_path / "decision.json",
        registry_path=tmp_path / "registry.sqlite3",
    )
    document = json.loads(artifact.output_path.read_text(encoding="utf-8"))
    momentum = next(
        item for item in document["members"] if item["candidate_name"] == "xs_mom_4w"
    )
    assert momentum["base_fold_retained_episode_counts"] == [
        EPISODES_PER_FOLD,
        EPISODES_PER_FOLD,
        0,
        EPISODES_PER_FOLD,
        EPISODES_PER_FOLD,
        EPISODES_PER_FOLD,
    ]
    assert momentum["base_folds_with_no_retained_episodes"] == [2]
    assert momentum["adverse_folds_with_no_retained_episodes"] == [2]
    # An untouched candidate never retains nothing anywhere.
    other = next(
        item for item in document["members"] if item["candidate_name"] == "xs_mom_1w"
    )
    assert other["base_folds_with_no_retained_episodes"] == []


CARRY_SPEC_PATH = Path("configs/funding-carry-panel-v1.json")
CARRY_SPEC, CARRY_SPEC_HASH = load_carry_family_spec(CARRY_SPEC_PATH)
EXTRAS = {
    "funding_collected": "0.001",
    "basis_pnl": "-0.0005",
    "spot_trading_cost": "0.0002",
    "perpetual_trading_cost": "0.0001",
}


def _rewrite_report(path: Path, mutate: Callable[[dict[str, object]], None]) -> None:
    """Mutate a fold report in place and re-derive its report_hash."""
    document = json.loads(path.read_text(encoding="utf-8"))
    mutate(document)
    material = {key: value for key, value in document.items() if key != "report_hash"}
    document["report_hash"] = content_sha256(material)
    path.write_bytes(canonical_json(document))


def _six_folds(tmp_path: Path) -> list[Path]:
    paths = []
    for fold_index in range(6):
        path = tmp_path / f"fold{fold_index}.json"
        write_fold(path, fold_index, {})
        paths.append(path)
    return paths


def _decide(tmp_path: Path, paths: list[Path], spec_path: Path = SPEC_PATH) -> dict[str, object]:
    artifact = build_panel_decision(
        tuple(paths),
        family_spec_path=spec_path,
        output_path=tmp_path / "decision.json",
        registry_path=tmp_path / "registry.sqlite3",
    )
    document: dict[str, object] = json.loads(artifact.output_path.read_text(encoding="utf-8"))
    return document


def _add_hedge(capture_hash: str) -> Callable[[dict[str, object]], None]:
    def mutate(document: dict[str, object]) -> None:
        document["hedge_capture_root_hash"] = capture_hash
        document["hedge_dataset_root_hash"] = "c" * 64

    return mutate


def _member_record(decision: dict[str, object], name: str) -> dict[str, object]:
    members = decision["members"]
    assert isinstance(members, list)
    return next(item for item in members if item["candidate_name"] == name)


def test_hedge_linkage_must_agree_across_folds(tmp_path: Path) -> None:
    paths = _six_folds(tmp_path)
    for index, path in enumerate(paths):
        _rewrite_report(path, _add_hedge(("a" if index < 5 else "b") * 64))
    with pytest.raises(PanelDecisionError, match="hedge"):
        _decide(tmp_path, paths)


def test_hedge_linkage_must_be_present_on_every_fold_or_none(tmp_path: Path) -> None:
    paths = _six_folds(tmp_path)
    _rewrite_report(paths[2], _add_hedge("a" * 64))
    with pytest.raises(PanelDecisionError, match="hedge"):
        _decide(tmp_path, paths)


def test_hedge_linkage_with_one_key_missing_is_rejected(tmp_path: Path) -> None:
    # Every fold agrees on a hedge_capture_root_hash but none carries a
    # hedge_dataset_root_hash at all -- the cross-document set check alone
    # cannot see this (it agrees, at size 1, on the single half-populated
    # tuple), so the half-present pair must be caught by a separate check.
    paths = _six_folds(tmp_path)

    def add_capture_hash_only(document: dict[str, object]) -> None:
        document["hedge_capture_root_hash"] = "a" * 64

    for path in paths:
        _rewrite_report(path, add_capture_hash_only)
    with pytest.raises(PanelDecisionError, match="hedge"):
        _decide(tmp_path, paths)


def test_hedge_linkage_is_copied_when_it_agrees(tmp_path: Path) -> None:
    paths = _six_folds(tmp_path)
    for path in paths:
        _rewrite_report(path, _add_hedge("a" * 64))
    decision = _decide(tmp_path, paths)
    assert decision["hedge_capture_root_hash"] == "a" * 64
    assert decision["hedge_dataset_root_hash"] == "c" * 64


def test_panel_reports_without_hedge_or_extras_are_unchanged(tmp_path: Path) -> None:
    decision = _decide(tmp_path, _six_folds(tmp_path))
    assert not any(key.startswith("hedge_") for key in decision)
    assert "extras_mean" not in _member_record(decision, MEMBERS[0])


def test_extras_are_averaged_over_retained_base_episodes(tmp_path: Path) -> None:
    paths = _six_folds(tmp_path)
    member = MEMBERS[0]

    def add_extras(document: dict[str, object]) -> None:
        candidates = document["candidates"]
        assert isinstance(candidates, list)
        for candidate in candidates:
            if candidate["candidate_name"] != member:
                continue
            for scenario in ("base", "adverse"):
                for index, episode in enumerate(candidate[scenario]["episodes"]):
                    episode["extras"] = dict(EXTRAS)
                    if index == 0:
                        # a held-nothing episode is excluded from the mean, so its
                        # absurd extras value must leave no trace
                        episode["reason_codes"] = [MEMBER_HELD_NOTHING_REASON_CODE]
                        episode["extras"]["funding_collected"] = "9"

    for path in paths:
        _rewrite_report(path, add_extras)
    decision = _decide(tmp_path, paths)
    record = _member_record(decision, member)
    assert record["extras_mean"] == EXTRAS
    assert "extras_mean" not in _member_record(decision, MEMBERS[1])


def write_carry_fold(path: Path, fold_index: int) -> None:
    candidates: list[dict[str, object]] = []
    for name in CARRY_MEMBERS + CARRY_CONTROLS:
        value = "0.001" if name in CARRY_MEMBERS else "0"
        rows = episodes(f"f{fold_index}", value, EPISODES_PER_FOLD)
        for row in rows:
            row["extras"] = dict(EXTRAS)
        candidates.append(
            {
                "candidate_name": name,
                "role": "member" if name in CARRY_MEMBERS else "control",
                "episode_count": EPISODES_PER_FOLD,
                "no_carry_cohort_sample_ids": [],
                "base": {
                    "total_net_return": str(Decimal(value) * EPISODES_PER_FOLD),
                    "episodes": rows,
                },
                "adverse": {
                    "total_net_return": str(Decimal(value) * EPISODES_PER_FOLD),
                    "episodes": [dict(row) for row in rows],
                },
            }
        )
    material = {
        "report_version": "1.0.0",
        "status": "development_only",
        "reason_codes": [],
        "family_name": CARRY_SPEC.family_name,
        "family_spec_hash": CARRY_SPEC_HASH,
        "capture_root_hash": "c" * 64,
        "dataset_root_hash": "d" * 64,
        "hedge_capture_root_hash": "a" * 64,
        "hedge_dataset_root_hash": "b" * 64,
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


def test_carry_family_is_pooled_through_the_same_gates(tmp_path: Path) -> None:
    paths = []
    for fold_index in range(6):
        path = tmp_path / f"carry{fold_index}.json"
        write_carry_fold(path, fold_index)
        paths.append(path)
    decision = _decide(tmp_path, paths, CARRY_SPEC_PATH)
    assert decision["family_name"] == "funding_carry_panel_v1"
    assert decision["family_spec_hash"] == CARRY_SPEC_HASH
    assert decision["decision_status"] in {"eligible_member_available", "no_eligible_member"}
    members = decision["members"]
    assert isinstance(members, list)
    assert [item["candidate_name"] for item in members] == list(CARRY_MEMBERS)
    assert all(item["extras_mean"] == EXTRAS for item in members)
    assert decision["hedge_capture_root_hash"] == "a" * 64


def test_carry_reports_are_rejected_against_the_panel_declaration(tmp_path: Path) -> None:
    paths = []
    for fold_index in range(6):
        path = tmp_path / f"carry{fold_index}.json"
        write_carry_fold(path, fold_index)
        paths.append(path)
    with pytest.raises(PanelDecisionError):
        _decide(tmp_path, paths, SPEC_PATH)
