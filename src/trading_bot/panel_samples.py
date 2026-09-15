"""Weekly rebalance calendar and walk-forward manifest for the panel."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.carry_config import CarryFamilySpec
from trading_bot.funding_xs_config import FundingXsFamilySpec
from trading_bot.panel_capture import verify_panel_capture
from trading_bot.panel_config import PanelFamilySpec, PanelFoldGeometry
from trading_bot.panel_reader import PanelBar, load_panel_bars
from trading_bot.splits import (
    SplitSample,
    WalkForwardConfig,
    WalkForwardFold,
    build_walk_forward_views,
)
from trading_bot.trend_config import TrendFamilySpec

DAY_NS = 86_400_000_000_000
_SUNDAY_REMAINDER = 3  # the Unix epoch began on a Thursday

# Operational data-quality threshold, not part of the frozen family
# declaration (`configs/xs-momentum-panel-v1.json`): that file is a
# pre-registration whose hash travels in every report, and a capture-quality
# gate is not a statement about the experiment family. Spec section 12 step 3.
_MAX_MISSING_DAYS_PER_INSTRUMENT = 3
# Cap on how many offending instruments are named in the failure message --
# enough to be actionable, not so many the message becomes unreadable.
_MAX_QUALITY_FAILURE_NAMES = 20


class PanelSamplesError(RuntimeError):
    """Raised when the panel rebalance calendar cannot be published."""


@dataclass(frozen=True, slots=True)
class PanelManifestArtifact:
    output_path: Path
    capture_root_hash: str
    dataset_root_hash: str
    manifest_hash: str
    fold_count: int
    pooled_test_sample_count: int


def rebalance_close_times(bars: tuple[PanelBar, ...]) -> tuple[int, ...]:
    """Return the sorted close times of Sunday bars observed anywhere in the panel."""
    times = {
        bar.close_time_ns
        for bar in bars
        if (bar.open_time_ns // DAY_NS) % 7 == _SUNDAY_REMAINDER
    }
    return tuple(sorted(times))


def build_rebalance_samples(
    bars: tuple[PanelBar, ...], *, holding_days: int
) -> tuple[SplitSample, ...]:
    if holding_days < 1:
        raise PanelSamplesError("holding period must be positive")
    return tuple(
        SplitSample(
            sample_id=f"BINANCE_UM:{close_time_ns}:w1",
            decision_time_ns=close_time_ns,
            label_end_time_ns=close_time_ns + holding_days * DAY_NS,
        )
        for close_time_ns in rebalance_close_times(bars)
    )


def derive_panel_config(
    samples: tuple[SplitSample, ...], *, folds: PanelFoldGeometry
) -> WalkForwardConfig:
    if len(samples) < 2:
        raise PanelSamplesError("panel capture cannot define a rebalance calendar")
    spacing = samples[-1].decision_time_ns - samples[-2].decision_time_ns
    if folds.holdout_duration_ns < spacing:
        raise PanelSamplesError("holdout duration is shorter than one rebalance interval")
    holdout_start = samples[-1].decision_time_ns - folds.holdout_duration_ns + spacing
    if holdout_start <= samples[0].decision_time_ns:
        raise PanelSamplesError("panel capture is too short for the declared geometry")
    return WalkForwardConfig(
        train_duration_ns=folds.train_duration_ns,
        validation_duration_ns=folds.validation_duration_ns,
        test_duration_ns=folds.test_duration_ns,
        step_ns=folds.step_ns,
        embargo_ns=folds.embargo_ns,
        final_holdout_start_ns=holdout_start,
    )


def publish_panel_walk_forward(
    capture_root: Path,
    *,
    output_path: Path,
    spec: PanelFamilySpec | CarryFamilySpec | TrendFamilySpec | FundingXsFamilySpec,
    family_spec_hash: str,
    hedge_capture_root: Path | None = None,
) -> PanelManifestArtifact:
    valid, errors = verify_panel_capture(capture_root)
    if not valid:
        raise PanelSamplesError("panel capture verification failed: " + ",".join(errors))
    if output_path.exists():
        raise PanelSamplesError("panel manifest already exists and is immutable")
    capture_manifest = _load_object(capture_root / "capture-manifest.json")
    dataset_manifest = _load_object(capture_root / "dataset" / "dataset-manifest.json")
    capture_hash = str(capture_manifest["capture_root_hash"])
    dataset_hash = str(dataset_manifest["root_hash"])
    # Bars load first so the quality gate can cross-check every day the
    # capture manifest claims is absent at source against this instrument's
    # own, independently recomputed missing days -- see
    # `_verify_absent_at_source_days`.
    bars = load_panel_bars(capture_root / "dataset")
    absent_at_source_days = _enforce_capture_quality(
        capture_root / "dataset" / "quality-report.json", capture_manifest, bars
    )

    samples = build_rebalance_samples(bars, holding_days=spec.holding_days)
    config = derive_panel_config(samples, folds=spec.folds)
    views = build_walk_forward_views(list(samples), config)
    if not views.folds:
        raise PanelSamplesError("panel capture does not contain a complete fold")
    pooled = sum(len(fold.test_ids) for fold in views.folds)
    if pooled < spec.statistics.pooled_episode_floor:
        raise PanelSamplesError(
            f"INSUFFICIENT_EVIDENCE: pooled test samples {pooled} below "
            f"{spec.statistics.pooled_episode_floor}"
        )

    hedge: dict[str, object] = {}
    if hedge_capture_root is not None:
        # The hedge leg of a funding-carry pair is the spot leg, and it is a
        # different capture from the perpetual one. Neither is inferable from
        # the manifest the two are bound into afterwards, so both are checked
        # here: a second perpetual capture, or the same capture passed twice,
        # would bind a manifest to a book that is not the position under test.
        if hedge_capture_root.resolve() == capture_root.resolve():
            raise PanelSamplesError("hedge capture must differ from the primary capture")
        # P1.27's capture predates the `market` key, so its absence means "um".
        if capture_manifest.get("market", "um") != "um":
            raise PanelSamplesError(
                f"primary capture must be a perpetual capture, got market "
                f"{capture_manifest.get('market', 'um')}"
            )
        hedge_valid, hedge_errors = verify_panel_capture(hedge_capture_root)
        if not hedge_valid:
            raise PanelSamplesError(
                "hedge capture verification failed: " + ",".join(hedge_errors)
            )
        hedge_manifest = _load_object(hedge_capture_root / "capture-manifest.json")
        if hedge_manifest.get("market") != "spot":
            raise PanelSamplesError(
                f"hedge capture must be a spot capture, got market "
                f"{hedge_manifest.get('market')}"
            )
        hedge_dataset = _load_object(hedge_capture_root / "dataset" / "dataset-manifest.json")
        hedge_bars = load_panel_bars(hedge_capture_root / "dataset")
        hedge_absent = _enforce_capture_quality(
            hedge_capture_root / "dataset" / "quality-report.json", hedge_manifest, hedge_bars
        )
        hedge = {
            "hedge_capture_root_hash": str(hedge_manifest["capture_root_hash"]),
            "hedge_dataset_root_hash": str(hedge_dataset["root_hash"]),
            "hedge_absent_at_source_days": {
                instrument_id: list(days) for instrument_id, days in sorted(hedge_absent.items())
            },
        }

    material: dict[str, object] = {
        "manifest_version": "1.0.0",
        "family_name": spec.family_name,
        "family_spec_hash": family_spec_hash,
        "capture_root_hash": capture_hash,
        "dataset_root_hash": dataset_hash,
        "holding_days": spec.holding_days,
        "split_manifest_hash": views.manifest_hash,
        "config": {
            "train_duration_ns": config.train_duration_ns,
            "validation_duration_ns": config.validation_duration_ns,
            "test_duration_ns": config.test_duration_ns,
            "step_ns": config.step_ns,
            "embargo_ns": config.embargo_ns,
            "final_holdout_start_ns": config.final_holdout_start_ns,
        },
        "folds": [_fold_record(fold) for fold in views.folds],
        "final_holdout_ids": list(views.final_holdout_ids),
        "pooled_test_sample_count": pooled,
        "capture_quality_max_missing_days_per_instrument": _MAX_MISSING_DAYS_PER_INSTRUMENT,
        "absent_at_source_days": {
            instrument_id: list(days)
            for instrument_id, days in sorted(absent_at_source_days.items())
        },
        **hedge,
    }
    manifest_hash = content_sha256(material)
    document = dict(material)
    document["manifest_hash"] = manifest_hash
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_bytes(canonical_json(document))
    temporary.replace(output_path)
    return PanelManifestArtifact(
        output_path=output_path,
        capture_root_hash=capture_hash,
        dataset_root_hash=dataset_hash,
        manifest_hash=manifest_hash,
        fold_count=len(views.folds),
        pooled_test_sample_count=pooled,
    )


def verify_panel_manifest(path: Path) -> bool:
    try:
        document = _load_object(path)
        recorded = str(document["manifest_hash"])
        material = {key: value for key, value in document.items() if key != "manifest_hash"}
        return content_sha256(material) == recorded
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False


def _enforce_capture_quality(
    quality_report_path: Path,
    capture_manifest: dict[str, object],
    bars: tuple[PanelBar, ...],
) -> dict[str, tuple[str, ...]]:
    """Stop the family before a manifest is built on a defective capture.

    Spec section 12 step 3: the family stops if any contract's daily series
    has more than `_MAX_MISSING_DAYS_PER_INSTRUMENT` missing days inside its
    listed span. `panel_dataset.py` already computes `missing_days` per
    instrument span-relative (a delisting does not count, only an interior
    gap does); this is the first and only reader of that field.

    Not every missing day is a capture defect, though: a day the capture
    itself attempted to patch from Binance's daily dump and recorded as
    absent there too (`_absent_at_source_days`) is a hole in the published
    data, not something this capture could have prevented -- it is
    subtracted from `missing_days` before the threshold is applied, leaving
    `unexplained_missing_days`. A day never attempted this way, or attempted
    and missing for any other reason, is not in that mapping and so still
    counts fully. The mapping is returned so the caller can record exactly
    what was subtracted in the walk-forward manifest, rather than letting
    the exemption live only here.

    Every claimed absence is cross-checked (`_verify_absent_at_source_days`)
    against this instrument's own missing days, recomputed independently
    from `bars`: the subtraction above is by count, and a count alone cannot
    tell a genuine absence from a capture-manifest entry naming a day that
    was never actually missing. A false claim of this kind is worse than a
    wrong gate decision -- it would let an immutable artifact assert
    something about the data that isn't true.
    """
    quality = _load_object(quality_report_path)
    instruments = quality.get("instruments")
    if not isinstance(instruments, list):
        raise PanelSamplesError("quality-report.json is malformed")
    absent_at_source = _absent_at_source_days(capture_manifest)
    _verify_absent_at_source_days(absent_at_source, bars)
    offenders: list[tuple[str, int]] = []
    for item in instruments:
        if not isinstance(item, dict):
            raise PanelSamplesError("quality-report.json is malformed")
        missing_days = item.get("missing_days")
        instrument_id = item.get("instrument_id")
        if not isinstance(missing_days, int) or not isinstance(instrument_id, str):
            raise PanelSamplesError("quality-report.json is malformed")
        proven_absent = len(absent_at_source.get(instrument_id, ()))
        if proven_absent > missing_days:
            raise PanelSamplesError(
                "quality-report.json is inconsistent with capture-manifest.json: "
                f"{instrument_id} has more days proven absent at source than missing days"
            )
        unexplained_missing_days = missing_days - proven_absent
        if unexplained_missing_days > _MAX_MISSING_DAYS_PER_INSTRUMENT:
            offenders.append((instrument_id, unexplained_missing_days))
    if not offenders:
        return absent_at_source
    offenders.sort(key=lambda pair: (-pair[1], pair[0]))
    shown = offenders[:_MAX_QUALITY_FAILURE_NAMES]
    detail = ", ".join(f"{name} ({count} unexplained missing days)" for name, count in shown)
    omitted = len(offenders) - len(shown)
    if omitted:
        detail += f", and {omitted} more"
    raise PanelSamplesError(
        "CAPTURE_QUALITY_FAILED: "
        f"{len(offenders)} instrument(s) exceed {_MAX_MISSING_DAYS_PER_INSTRUMENT} "
        f"unexplained missing days inside their listed span: {detail}"
    )


# Must match `trading_bot.panel_capture._DAILY_FILL_KIND`: the "kind" a
# capture-manifest source entry records for a day fetched from Binance's daily
# dump to patch a gap in the monthly aggregates. Re-declared here (not
# imported) because that name is a private module constant of panel_capture.
_DAILY_FILL_SOURCE_KIND = "klines_daily_fill"
# The one status `panel_capture._fill_gap_days` ever writes for a day the
# daily dump itself answered absent. Checked by explicit inclusion, not by
# excluding "present": `absent_after_discovery` is a real status this
# codebase writes elsewhere (a monthly source the bucket's own listing
# claimed exists, that then 404s on fetch -- the bucket contradicting
# itself, not proof of an absence), and excluding only "present" would have
# silently accepted it, and any future status, as proof too.
_DAILY_FILL_ABSENT_STATUS = "absent"


def _absent_at_source_days(capture_manifest: dict[str, object]) -> dict[str, tuple[str, ...]]:
    """Per instrument, the days the capture attempted from Binance's daily dump
    and recorded as absent there too -- proof the day is missing from the
    published data itself, not a defect this capture could have avoided.

    Only a `klines_daily_fill` source entry with `status: "absent"` counts. A
    day never attempted this way -- or attempted and recorded under any
    other status -- is simply absent from the returned mapping, so it is not
    exempted anywhere: this must not become a way for an unattempted,
    genuinely defective gap to pass the quality gate.
    """
    sources = capture_manifest.get("sources")
    if not isinstance(sources, list):
        raise PanelSamplesError("capture-manifest.json is malformed")
    absent: dict[str, set[str]] = {}
    for entry in sources:
        if not isinstance(entry, dict):
            raise PanelSamplesError("capture-manifest.json is malformed")
        if (
            entry.get("kind") != _DAILY_FILL_SOURCE_KIND
            or entry.get("status") != _DAILY_FILL_ABSENT_STATUS
        ):
            continue
        symbol = entry.get("symbol")
        # The daily-fill source record reuses the "month" field for the exact
        # date it fetched (see `panel_capture._fill_gap_days`), not a month.
        date = entry.get("month")
        if not isinstance(symbol, str) or not isinstance(date, str):
            raise PanelSamplesError("capture-manifest.json is malformed")
        absent.setdefault(symbol, set()).add(date)
    return {symbol: tuple(sorted(dates)) for symbol, dates in absent.items()}


def _verify_absent_at_source_days(
    absent_at_source: dict[str, tuple[str, ...]], bars: tuple[PanelBar, ...]
) -> None:
    """Fail closed if the capture manifest claims a day absent at source that
    the published dataset does not actually show as missing.

    `_enforce_capture_quality` discounts by count, which a count alone
    cannot make honest: a capture-manifest entry can name any date at all,
    and its `capture_root_hash` re-hashes cleanly over that fabrication as
    readily as over the truth, since the hash proves only internal
    self-consistency, not agreement with the dataset it describes. This
    recomputes each instrument's real interior missing days directly from
    the published bars -- already hash-verified against dataset-manifest.json
    by `verify_panel_capture` -- and requires every claimed day to be one of
    them.
    """
    actual_missing_days = _missing_day_strings(bars)
    for instrument_id, claimed_days in sorted(absent_at_source.items()):
        real_missing = actual_missing_days.get(instrument_id, frozenset())
        bogus = sorted(day for day in claimed_days if day not in real_missing)
        if bogus:
            raise PanelSamplesError(
                "capture-manifest.json is inconsistent with the published dataset: "
                f"{instrument_id} claims day(s) absent at source that are not among its "
                f"missing days: {', '.join(bogus)}"
            )


def _missing_day_strings(bars: tuple[PanelBar, ...]) -> dict[str, frozenset[str]]:
    """Per instrument, the calendar dates missing inside its own first-to-last
    observed day.

    The same span-relative interior gap `panel_dataset.find_missing_days`
    computes at capture time (a delisting or a late listing contributes
    nothing, only an interior hole does), recomputed here from the published
    `PanelBar`s rather than the `PanelCandleRow`s that function expects, so
    `_verify_absent_at_source_days` can check a capture-manifest claim
    against what the dataset actually shows missing.
    """
    grouped: dict[str, list[int]] = {}
    for bar in bars:
        grouped.setdefault(bar.instrument_id, []).append(bar.open_time_ns)
    result: dict[str, frozenset[str]] = {}
    for instrument_id, times in grouped.items():
        times.sort()
        present = set(times)
        first = times[0]
        last = times[-1]
        missing = (day for day in range(first, last + DAY_NS, DAY_NS) if day not in present)
        result[instrument_id] = frozenset(_date_from_open_time_ns(day) for day in missing)
    return result


def _date_from_open_time_ns(open_time_ns: int) -> str:
    return datetime.fromtimestamp(open_time_ns / 1_000_000_000, tz=UTC).date().isoformat()


def _fold_record(fold: WalkForwardFold) -> dict[str, object]:
    return {
        "fold_index": fold.fold_index,
        "train_start_ns": fold.train_start_ns,
        "train_end_ns": fold.train_end_ns,
        "validation_start_ns": fold.validation_start_ns,
        "validation_end_ns": fold.validation_end_ns,
        "test_start_ns": fold.test_start_ns,
        "test_end_ns": fold.test_end_ns,
        "train_ids": list(fold.train_ids),
        "validation_ids": list(fold.validation_ids),
        "test_ids": list(fold.test_ids),
    }


def _load_object(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise PanelSamplesError(f"{path.name} must contain a JSON object")
    return document
