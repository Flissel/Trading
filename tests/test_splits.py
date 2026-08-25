from trading_bot.splits import SplitSample, WalkForwardConfig, build_walk_forward_views


def samples() -> list[SplitSample]:
    return [
        SplitSample(
            sample_id=f"s{decision}",
            decision_time_ns=decision,
            label_end_time_ns=decision + 15,
        )
        for decision in range(0, 121, 10)
    ]


def test_walk_forward_builder_applies_purge_and_embargo() -> None:
    views = build_walk_forward_views(
        samples(),
        WalkForwardConfig(
            train_duration_ns=30,
            validation_duration_ns=20,
            test_duration_ns=20,
            step_ns=20,
            embargo_ns=10,
            final_holdout_start_ns=100,
        ),
    )

    assert len(views.folds) == 1
    fold = views.folds[0]
    assert fold.train_ids == ("s0", "s10")
    assert fold.validation_ids == ("s40",)
    assert fold.test_ids == ("s70",)
    assert fold.train_end_ns == 30
    assert fold.validation_start_ns == 40
    assert fold.test_start_ns == 70


def test_final_holdout_is_separate_from_all_walk_forward_folds() -> None:
    views = build_walk_forward_views(
        samples(),
        WalkForwardConfig(30, 20, 20, 20, 10, 100),
    )

    fold_ids = {
        sample_id
        for fold in views.folds
        for sample_id in fold.train_ids + fold.validation_ids + fold.test_ids
    }
    assert views.final_holdout_ids == ("s100", "s110", "s120")
    assert fold_ids.isdisjoint(views.final_holdout_ids)


def test_walk_forward_manifest_is_deterministic_and_input_sensitive() -> None:
    config = WalkForwardConfig(30, 20, 20, 20, 10, 100)

    first = build_walk_forward_views(samples(), config)
    second = build_walk_forward_views(samples(), config)
    changed = build_walk_forward_views(samples()[:-1], config)

    assert first.manifest_hash == second.manifest_hash
    assert first.manifest_hash != changed.manifest_hash


def test_builder_rejects_duplicate_or_non_chronological_samples() -> None:
    config = WalkForwardConfig(30, 20, 20, 20, 10, 100)
    duplicate = [
        SplitSample("same", 0, 5),
        SplitSample("same", 10, 15),
    ]

    try:
        build_walk_forward_views(duplicate, config)
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate sample IDs must fail closed")
