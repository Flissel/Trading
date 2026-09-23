"""Tests for the shadow and measurement-journal commands (spec sections 2, 5 and 8).

Every command here runs through `main` on fixture data: the weekly captures
come from `tests.shadow_fixtures`' fake fetch, the measurement rounds from the
cost journal's fake venue. No network, no venue client, no key -- the fetchers
the handlers would reach for are replaced before the command is invoked, and
the refusal tests prove the path bound is checked before one is built at all.
"""

import json
from decimal import Decimal
from pathlib import Path

import pytest

from tests.carry_fixtures import small_carry_v4_config
from tests.shadow_fixtures import (
    FIRST_TAIL_SUNDAY,
    SECOND_TAIL_SUNDAY,
    SHADOW_FAMILY_NAME,
    ShadowBookCaptures,
    ShadowFetch,
    build_shadow_book_captures,
    write_shadow_declaration,
)
from tests.test_binance_cost_journal import (
    FakeClock,
    FakeSleep,
    FakeVenue,
    rewrite_segment,
    write_journal_directory,
)
from tests.test_binance_measurement_journal import (
    SnapshotVenue,
    measurement_journal,
    segment_paths,
)
from trading_bot import cli
from trading_bot.binance_cost_journal import iso_utc_time
from trading_bot.binance_measurement_journal import (
    run_measurement_journal,
    verify_measurement_journal,
)
from trading_bot.carry_config import load_carry_family_spec
from trading_bot.cli import main
from trading_bot.panel_capture import verify_panel_capture

_RUN_ID = "binance-measurement-v2"


def _document(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def _no_key_anywhere(value: object, forbidden: str) -> bool:
    """True when `forbidden` is not a key anywhere under `value` (spec 4.2)."""
    if isinstance(value, dict):
        return all(
            key != forbidden and _no_key_anywhere(item, forbidden)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return all(_no_key_anywhere(item, forbidden) for item in value)
    return True


# --- the weekly chain ---------------------------------------------------


@pytest.fixture(scope="module")
def book_captures(tmp_path_factory: pytest.TempPathFactory) -> ShadowBookCaptures:
    """Both markets' base and two chained weekly captures, built once.

    The wall clock is pinned while they are built, exactly as
    `tests.test_shadow_book` pins it: `zipfile` stamps each archive member with
    it, so a fixture capture is byte-stable only if it does not straddle a tick.
    """
    patch = pytest.MonkeyPatch()
    patch.setattr("time.time", lambda: 1_700_000_000.0)
    try:
        return build_shadow_book_captures(tmp_path_factory.mktemp("cli-shadow"))
    finally:
        patch.undo()


def test_the_cli_builds_the_monday_chain_of_weekly_shadow_captures(
    book_captures: ShadowBookCaptures,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The Monday job's two captures, and the next week chained onto the first."""
    root = book_captures.root
    monkeypatch.setattr(cli, "PanelZipClient", lambda: ShadowFetch())
    perpetual = root / "cli-perp-first"
    assert (
        main(
            [
                "shadow-capture", "--workspace-root", str(root),
                "--base-capture", str(book_captures.perp_base),
                "--output", str(perpetual), "--tail-through", FIRST_TAIL_SUNDAY,
                "--market", "um", "--reserve-bytes", "0",
            ]
        )
        == 0
    )
    reported = capsys.readouterr().out
    assert str(perpetual) in reported
    assert FIRST_TAIL_SUNDAY in reported
    assert verify_panel_capture(perpetual) == (True, ())

    spot = root / "cli-spot-first"
    assert (
        main(
            [
                "shadow-capture", "--workspace-root", str(root),
                "--base-capture", str(book_captures.spot_base),
                "--output", str(spot), "--tail-through", FIRST_TAIL_SUNDAY,
                "--market", "spot", "--reserve-bytes", "0",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert verify_panel_capture(spot) == (True, ())

    # Next week: the same base, this week's capture as the previous one.
    following = root / "cli-perp-second"
    assert (
        main(
            [
                "shadow-capture", "--workspace-root", str(root),
                "--base-capture", str(book_captures.perp_base),
                "--output", str(following), "--tail-through", SECOND_TAIL_SUNDAY,
                "--previous-capture", str(perpetual), "--market", "um",
                "--reserve-bytes", "0",
            ]
        )
        == 0
    )
    assert SECOND_TAIL_SUNDAY in capsys.readouterr().out
    assert verify_panel_capture(following) == (True, ())
    manifest = _document(following / "capture-manifest.json")
    assert manifest["tail_through"] == SECOND_TAIL_SUNDAY


def test_the_cli_refuses_a_shadow_capture_outside_the_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode() -> object:
        raise AssertionError("a refused path must not build a client")

    monkeypatch.setattr(cli, "PanelZipClient", explode)
    assert (
        main(
            [
                "shadow-capture", "--workspace-root", str(tmp_path),
                "--base-capture", str(tmp_path / "base"),
                "--output", str(tmp_path.parent / "outside-capture"),
                "--tail-through", FIRST_TAIL_SUNDAY, "--market", "um",
                "--reserve-bytes", "0",
            ]
        )
        == 2
    )


def test_the_cli_refuses_a_shadow_capture_it_cannot_build(
    book_captures: ShadowBookCaptures, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A base of the other market is a refusal a rerun cannot fix: exit 2."""
    root = book_captures.root
    monkeypatch.setattr(cli, "PanelZipClient", lambda: ShadowFetch())
    assert (
        main(
            [
                "shadow-capture", "--workspace-root", str(root),
                "--base-capture", str(book_captures.spot_base),
                "--output", str(root / "cli-wrong-market"),
                "--tail-through", FIRST_TAIL_SUNDAY, "--market", "um",
                "--reserve-bytes", "0",
            ]
        )
        == 2
    )


# --- the weekly book ----------------------------------------------------


def _declaration(root: Path, name: str, *, phase: str = "A") -> Path:
    spec_path = small_carry_v4_config(root)
    _, spec_hash = load_carry_family_spec(spec_path)
    return write_shadow_declaration(
        root / f"{name}-declaration.json",
        family_spec_path=spec_path.name,
        family_spec_hash=spec_hash,
        artifact_root=f"artifacts/{name}",
        registry_path=f"artifacts/{name}/metadata-shadow.sqlite3",
        phase=phase,
    )


def test_the_cli_writes_a_phase_a_shadow_week_without_a_pnl_block(
    book_captures: ShadowBookCaptures, capsys: pytest.CaptureFixture[str]
) -> None:
    root = book_captures.root
    declaration = _declaration(root, "cli-week")
    assert (
        main(
            [
                "shadow-week", "--workspace-root", str(root),
                "--declaration", str(declaration),
                "--capture", str(book_captures.perp_second),
                "--hedge-capture", str(book_captures.spot_second),
                "--perp-base-capture", str(book_captures.perp_base),
                "--spot-base-capture", str(book_captures.spot_base),
                "--decision-sunday", SECOND_TAIL_SUNDAY, "--reserve-bytes", "0",
            ]
        )
        == 0
    )
    output = root / "artifacts" / "cli-week" / SHADOW_FAMILY_NAME / f"{SECOND_TAIL_SUNDAY}.json"
    document = _document(output)
    assert document["status"] == "development_only"
    assert _no_key_anywhere(document, "pnl")
    assert _no_key_anywhere(document, "running_totals")
    reported = capsys.readouterr().out
    assert str(output) in reported
    assert str(document["report_hash"]) in reported
    assert "development_only" in reported


def test_the_cli_refuses_a_shadow_week_outside_the_workspace(tmp_path: Path) -> None:
    assert (
        main(
            [
                "shadow-week", "--workspace-root", str(tmp_path),
                "--declaration", str(tmp_path / "declaration.json"),
                "--capture", str(tmp_path.parent / "outside-perp"),
                "--hedge-capture", str(tmp_path / "spot"),
                "--decision-sunday", SECOND_TAIL_SUNDAY, "--reserve-bytes", "0",
            ]
        )
        == 2
    )


def test_the_cli_refuses_a_shadow_week_it_cannot_publish(
    book_captures: ShadowBookCaptures,
) -> None:
    """A Phase A week handed a holdout report is refused: exit 2, nothing written."""
    root = book_captures.root
    declaration = _declaration(root, "cli-week-refused")
    report = root / "cli-week-refused-holdout.json"
    report.write_text(json.dumps({"holdout": True}), encoding="utf-8")
    assert (
        main(
            [
                "shadow-week", "--workspace-root", str(root),
                "--declaration", str(declaration),
                "--capture", str(book_captures.perp_second),
                "--hedge-capture", str(book_captures.spot_second),
                "--perp-base-capture", str(book_captures.perp_base),
                "--spot-base-capture", str(book_captures.spot_base),
                "--decision-sunday", SECOND_TAIL_SUNDAY,
                "--holdout-report", str(report), "--reserve-bytes", "0",
            ]
        )
        == 2
    )
    assert not (root / "artifacts" / "cli-week-refused").exists()


# --- the measurement stream ---------------------------------------------


def test_the_cli_creates_and_runs_a_measurement_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cost_journal = tmp_path / "cost-journal"
    write_journal_directory(cost_journal)
    journal = tmp_path / "measurement"
    create = [
        "binance-measurement-journal-create", "--workspace-root", str(tmp_path),
        "--journal", str(journal), "--run-id", _RUN_ID,
        "--cost-journal", str(cost_journal), "--reserve-bytes", "0",
    ]
    assert main(create) == 0
    reported = capsys.readouterr().out
    assert "2 instruments" in reported
    assert _RUN_ID in reported

    monkeypatch.setattr(cli, "public_binance_json_fetcher", FakeVenue())
    monkeypatch.setattr(cli, "public_binance_json_array_fetcher", SnapshotVenue())
    assert (
        main(
            [
                "binance-measurement-journal-run", "--workspace-root", str(tmp_path),
                "--journal", str(journal), "--rounds", "1", "--reserve-bytes", "0",
            ]
        )
        == 0
    )
    assert len(segment_paths(journal)) == 1
    assert verify_measurement_journal(journal) == (True, ())
    # An existing journal is immutable: the supervisor's stop code, not a retry.
    assert main(create) == 2


@pytest.fixture
def running_journal(tmp_path: Path) -> Path:
    """Six sampled rounds -- two of them snapshot rounds -- on the fake venue."""
    journal = measurement_journal(tmp_path)
    run_measurement_journal(
        workspace_root=tmp_path,
        journal_root=journal,
        reserve_bytes=0,
        rounds=6,
        fetcher=FakeVenue(),
        array_fetcher=SnapshotVenue(),
        clock=FakeClock(),
        sleep=FakeSleep(),
    )
    return journal


def test_the_cli_reports_a_running_measurement_journal(
    tmp_path: Path, running_journal: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "binance-measurement-journal-status", "--workspace-root", str(tmp_path),
                "--journal", str(running_journal), "--last", "6",
            ]
        )
        == 0
    )
    reported = capsys.readouterr().out.splitlines()
    assert "segment_count: 6" in reported
    assert "last_sequence: 5" in reported
    assert "window: last 6 segments" in reported
    assert "snapshot_rounds: 2" in reported
    assert "depth_failure_rate: 0.000000" in reported
    assert "failure_rate premiumIndex: 0.000000" in reported
    assert "excluded_total premiumIndex: 0" in reported
    assert "verify: ok" in reported
    newest = _document(segment_paths(running_journal)[-1])["received_time_ns"]
    assert isinstance(newest, int)
    assert f"newest_received_time: {iso_utc_time(newest)}" in reported
    # The liveness figure: the wall clock less the newest segment's own stamp.
    age = next(line for line in reported if line.startswith("newest_age_seconds: "))
    assert Decimal(age.removeprefix("newest_age_seconds: ")) > 0


def test_the_cli_verifies_the_whole_chain_and_names_what_broke_it(
    tmp_path: Path, running_journal: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    verify = [
        "binance-measurement-journal-verify", "--workspace-root", str(tmp_path),
        "--journal", str(running_journal),
    ]
    assert main(verify) == 0
    assert capsys.readouterr().out == "verify: ok\n"
    # A rewritten segment in the middle is older than the tail a restart reads,
    # so this command is the one that finds it.
    rewrite_segment(segment_paths(running_journal)[2], sequence=99)
    assert main(verify) == 1
    assert "verify: " in capsys.readouterr().out


def test_the_cli_seals_a_window_of_the_measurement_journal(
    tmp_path: Path, running_journal: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    day = segment_paths(running_journal)[0].parent.name
    output = tmp_path / f"{_RUN_ID}-{day}.json"
    snapshot = [
        "binance-measurement-journal-snapshot", "--workspace-root", str(tmp_path),
        "--journal", str(running_journal), "--output", str(output),
        "--window-start", f"{day}T00:00:00Z", "--window-end", f"{day}T23:59:59Z",
        "--reserve-bytes", "0",
    ]
    assert main(snapshot) == 0
    document = _document(output)
    assert document["rounds"] == 6
    assert (document["first_sequence"], document["last_sequence"]) == (0, 5)
    reported = capsys.readouterr().out
    assert str(output) in reported
    assert str(document["content_hash"]) in reported
    # A snapshot is a receipt: an existing one is refused, not replaced.
    assert main(snapshot) == 2


@pytest.mark.parametrize("stamp", ["2025-09-04", "2025-09-04T00:00:00+02:00", "yesterday"])
def test_the_cli_refuses_a_snapshot_window_that_is_not_an_iso_utc_stamp(
    tmp_path: Path, running_journal: Path, stamp: str
) -> None:
    assert (
        main(
            [
                "binance-measurement-journal-snapshot", "--workspace-root", str(tmp_path),
                "--journal", str(running_journal), "--output", str(tmp_path / "snapshot.json"),
                "--window-start", stamp, "--window-end", "2025-09-05T00:00:00Z",
                "--reserve-bytes", "0",
            ]
        )
        == 2
    )
    assert not (tmp_path / "snapshot.json").exists()


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param(
            [
                "binance-measurement-journal-create", "--journal", "{outside}",
                "--run-id", _RUN_ID, "--cost-journal", "{inside}", "--reserve-bytes", "0",
            ],
            id="create-journal",
        ),
        pytest.param(
            [
                "binance-measurement-journal-create", "--journal", "{inside}",
                "--run-id", _RUN_ID, "--cost-journal", "{outside}", "--reserve-bytes", "0",
            ],
            id="create-cost-journal",
        ),
        pytest.param(
            [
                "binance-measurement-journal-run", "--journal", "{outside}",
                "--rounds", "1", "--reserve-bytes", "0",
            ],
            id="run",
        ),
        pytest.param(
            ["binance-measurement-journal-status", "--journal", "{outside}"], id="status"
        ),
        pytest.param(
            ["binance-measurement-journal-verify", "--journal", "{outside}"], id="verify"
        ),
        pytest.param(
            [
                "binance-measurement-journal-snapshot", "--journal", "{inside}",
                "--output", "{outside}", "--window-start", "2025-09-04T00:00:00Z",
                "--window-end", "2025-09-05T00:00:00Z", "--reserve-bytes", "0",
            ],
            id="snapshot",
        ),
    ],
)
def test_the_cli_refuses_a_measurement_journal_path_outside_the_workspace(
    tmp_path: Path, arguments: list[str]
) -> None:
    filled = [
        item.format(inside=str(tmp_path / "inside"), outside=str(tmp_path.parent / "outside"))
        for item in arguments
    ]
    assert main([*filled, "--workspace-root", str(tmp_path)]) == 2


def test_the_cli_stops_on_a_measurement_journal_that_is_not_there(tmp_path: Path) -> None:
    """A journal that does not exist is a stop code, not a retry."""
    absent = str(tmp_path / "absent")
    assert (
        main(
            [
                "binance-measurement-journal-run", "--workspace-root", str(tmp_path),
                "--journal", absent, "--rounds", "1", "--reserve-bytes", "0",
            ]
        )
        == 2
    )
    assert (
        main(
            [
                "binance-measurement-journal-status", "--workspace-root", str(tmp_path),
                "--journal", absent,
            ]
        )
        == 2
    )
    assert (
        main(
            [
                "binance-measurement-journal-verify", "--workspace-root", str(tmp_path),
                "--journal", absent,
            ]
        )
        == 1
    )
