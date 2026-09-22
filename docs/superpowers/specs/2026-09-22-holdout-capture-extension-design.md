# Holdout Capture Extension Design (P1.33 → holdout read)

Status: Decided autonomously on 2026-09-22 under the standing goal, on the day
`P1_33_DECISION_2026-09-22.md` named `carry_s10_l4w_h26w_exit` as the release candidate.
Fixes how the two panel captures are extended so that the single-use holdout read
(`carry-holdout`, protocol section 16.2, `docs/superpowers/specs/2026-09-17-holdout-read-design.md`)
can run. It authorises no holdout read: the read itself is the user's explicit decision.
Everything here is preparation that spends nothing.

## 1. Why an extension is needed

The holdout's 26 decisions run 2026-03-08 to 2026-08-30 and its last episode exits on
2026-09-06. The captures the family was evaluated on end with the August 2026 monthly
dumps (`discovered_months` latest `2026-08` in both), so the last episode's exit bar and
its final week of funding settlements are missing. The holdout runner refuses on exactly
that: it requires a bar at the exit close in both markets and the exit month reached by
every data kind (klines and funding for the perpetual capture, klines for spot).

Binance publishes monthly dumps for September 2026 in early October. Daily kline dumps for
2026-09-01 to 09-06 already exist, but funding has no daily dump, and the coverage rule is
per data kind, so the read waits for the September funding monthly dump.

## 2. Rules the extension must satisfy (all already enforced by `carry-holdout`)

1. **Discovery run, not a repair of a pre-September source.** `panel-capture` without
   `--months` discovers the months available per symbol; the manifest records them as
   `discovered_months`. `panel-capture-repair` copies its source's `discovered_months`
   verbatim, so repairing the 2026-09-10/11 captures would carry `2026-08` forward and the
   coverage check would refuse (safe direction, but wasted). The extension is a fresh
   discovery capture per market, then its own repair.
2. **Same symbol lists as the originals.** The original manifests are the authority: 860
   perpetual symbols and 470 spot symbols in their `symbols` fields. The scratch files the
   September captures were launched from are session-local and not to be trusted; the
   lists are read from the manifests.
3. **Verified superset of the originals.** Every `sources` row of an original manifest
   (kind, symbol, month) must appear in the extended manifest with identical `raw_sha256`
   and `status`, and the headers (venue, interval, market) must agree
   (`capture_lineage.verify_capture_superset`). Binance's monthly dumps are stable bytes,
   so a fresh download reproduces the originals' hashes; the two known ways this fails are
   a republished monthly dump and a daily-fill row whose status changed because Binance
   backfilled a day. Both are refusals to diagnose, never to override.
4. **Same start month.** The extension discovers from the same earliest month as the
   originals (no `--month-from` narrower or wider than theirs); an earlier start would
   shift contract identities and history requirements relative to the family's own
   capture. `--month-to 2026-09` bounds the run so a later October dump cannot slip in.
5. **The originals stay on disk untouched.** `carry-holdout` binds the manifest to the
   original captures (root and dataset hashes) before it checks lineage; the originals are
   the lineage base and the read cannot run without them.
6. **Storage.** Each capture is written under the default 20 GB reserve; C: had 135 GB
   free on 2026-09-22.

## 3. Readiness is confirmed by a dry run, never by the read

`carry-holdout --check-only` runs every input check — both extended captures verify, the
originals verify, the manifest links to the originals, lineage, the decision's and fold
reports' seals and links, the registry's single-use record, the exit month per data kind,
and a bar at the exit close — derives the candidate, and stops. Nothing is evaluated,
written or registered. The extension is "ready" when the dry run prints
`carry holdout check: candidate carry_s10_l4w_h26w_exit ready`. A refusal names the check
that failed; the coverage refusal does not name the data kind, so the diagnosis is the
extended manifest's `discovered_months` per kind.

## 4. What happens after readiness

Nothing, until the user says so. The read is `carry-holdout` without `--check-only`, once,
writing `artifacts/carry/funding-carry-v4-holdout-v1.json` and the registry record; then
`P1_33_HOLDOUT_<date>.md`. `holdout_confirmed` authorises shadow forecasts only;
`holdout_failed` closes the v4 generation. Neither outcome moves a threshold.

## 5. Artifacts

- `data/captures/<date>-binance-um-usdt-perps-1d` and `-repaired` (perpetual, discovery
  through 2026-09)
- `data/captures/<date>-binance-spot-usdt-1d` and `-repaired` (spot, discovery through
  2026-09)
- Supervisor scripts beside the September ones in `C:\Users\User\.trading-jobs\`, symbol
  lists derived from the original manifests into that directory (not the scratchpad)
- No new code, no new declaration, no change to any frozen rule
