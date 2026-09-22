# Holdout Capture Extension Plan (operations, no code)

> Runs in October 2026, after Binance publishes the September 2026 monthly dumps. Every step
> before the last one spends nothing; the last one is the user's explicit decision and is not
> part of this plan.

**Goal:** Extend both panel captures through September 2026 as verified supersets of the
originals so that `carry-holdout --check-only` reports the P1.33 release candidate ready.

**Spec:** `docs/superpowers/specs/2026-09-22-holdout-capture-extension-design.md`; the read
itself is governed by `docs/superpowers/specs/2026-09-17-holdout-read-design.md` and
`PHASE_0_EVALUATION_PROTOCOL.md` section 16.2.

## Global Constraints

- Workspace root `C:\Users\User\Documents\ChatGPT\Trading`; `TEMP`/`TMP` on C:; the
  originals `data/captures/2026-09-10-binance-um-usdt-perps-1d-repaired` and
  `data/captures/2026-09-11-binance-spot-usdt-1d-repaired` are never modified or moved.
- Symbol lists come from the original manifests' `symbols` fields (860 / 470), nothing
  else. Discovery from the originals' earliest month, bounded with `--month-to 2026-09`.
- No `--force`, no edits to any manifest, no change to any config or threshold. A lineage
  or coverage refusal is diagnosed and reported, not overridden.

## Task 1: Wait for the September dumps

- [ ] Check both monthly dumps exist (HTTP 200), e.g. for `BTCUSDT`:
  `https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2026-09.zip`,
  `https://data.binance.vision/data/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2026-09.zip`,
  `https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2026-09.zip`.
  Until all three exist, stop here (the daily-check cron re-tries). Expected: early October.

## Task 2: Derive the symbol lists from the original manifests

- [ ] Write `C:\Users\User\.trading-jobs\perp-symbols-2026-10.txt` and
  `spot-symbols-2026-10.txt` as comma-separated lists from
  `data/captures/2026-09-10-binance-um-usdt-perps-1d-repaired/capture-manifest.json`
  (`symbols`, 860 entries) and the spot original (470 entries). Verify the counts.
- [ ] Read the originals' earliest discovered month per market (minimum over
  `discovered_months`) to use as `--month-from`.

## Task 3: Capture and repair the perpetual market (discovery through 2026-09)

- [ ] Supervisor script `C:\Users\User\.trading-jobs\panel-capture-2026-10.ps1` in the
  style of the September one (log, retry loop, TEMP on C:), running:
  `uv run trading-research panel-capture --workspace-root <ws> --output data/captures/<date>-binance-um-usdt-perps-1d --symbols <perp list> --month-from <earliest> --month-to 2026-09`
  then `uv run trading-research panel-capture-repair --workspace-root <ws> --source-capture data/captures/<date>-binance-um-usdt-perps-1d --output data/captures/<date>-binance-um-usdt-perps-1d-repaired`.
  The September run took about 7.5 hours; run detached, check the log.
- [ ] Confirm `discovered_months` in the repaired manifest reaches `2026-09` for both
  `klines` and `fundingRate` (max over symbols per kind), and that
  `verify_capture_superset(original, extended)` is `True` with no reasons. A `hash` or
  `status` reason names a row Binance changed; report it and stop.

## Task 4: Capture and repair the spot market (same procedure)

- [ ] `panel-capture --market spot --reserve-bytes 10000000000 --output data/captures/<date>-binance-spot-usdt-1d --symbols <spot list> --month-from <earliest> --month-to 2026-09`
  (the September spot run needed a retry after a `MemoryError` in `find_missing_days`; the
  supervisor's retry loop covers it), then the repair, then the same two checks.

## Task 5: Dry run

- [ ] `uv run trading-research carry-holdout --check-only --workspace-root <ws> --capture <perp-extended-repaired> --hedge-capture <spot-extended-repaired> --original-capture data/captures/2026-09-10-binance-um-usdt-perps-1d-repaired --original-hedge-capture data/captures/2026-09-11-binance-spot-usdt-1d-repaired --manifest artifacts/carry/carry-v4-walk-forward-usdt-pairs-1d-w1-v1.json --family-spec configs/funding-carry-panel-v4-measured.json --decision artifacts/carry/funding-carry-v4-decision-v1.json --fold-report artifacts/carry/funding-carry-v4-fold0-v1.json … --fold-report artifacts/carry/funding-carry-v4-fold8-v1.json --output artifacts/carry/funding-carry-v4-holdout-v1.json --registry artifacts/carry/metadata-funding-carry-v4.sqlite3`
  Expected: exit 0 and `carry holdout check: candidate carry_s10_l4w_h26w_exit ready`.
  Any refusal: report the message and the extended manifests' per-kind months; do not
  retry with other inputs.
- [ ] Report readiness to the user with the four capture root hashes. **Stop.** The read
  (the same command without `--check-only`) runs only on the user's explicit go, once.

## Automation (added 2026-09-22, on the user's go to prepare the read)

Tasks 1 to 5 are implemented by the detached watcher
`C:/Users/User/.trading-jobs/holdout-capture-extension.ps1` (log
`holdout-capture-extension.log`, Startup launcher `holdout-capture-extension.cmd`, helpers
`holdout-extension-symbols.py` and `holdout-extension-lineage.py` beside it). It polls the
three September dump URLs every six hours, then runs the captures (resumable, retried),
the repairs, the superset checks and the dry run, and ends with `READY`, `REFUSED` or
`STOP` in the log. It never runs the read itself. Started 2026-09-22 21:53 local.

## After the plan

Only on that go: run the read, write `P1_33_HOLDOUT_<date>.md` with the verdict, the
criteria, the reported values and the bound hashes, commit, update memory.
`holdout_confirmed` → shadow phase design and the paper-episode protocol revision (user
decisions); `holdout_failed` → the v4 generation is closed.
