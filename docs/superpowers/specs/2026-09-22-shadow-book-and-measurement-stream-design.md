# Shadow Book and Measurement Stream Design (P1.24 for the small-book carry)

Status: Decided with the user on 2026-09-22 ("1" for the shadow infrastructure, "ok" for
the permanent measurement stream, no directional betting). Designed while the P1.33 holdout
is locked; nothing here reads it, evaluates it, or claims anything from it. Authority:
`PHASE_0_EVALUATION_PROTOCOL.md` section 13 ("historical promotion authorizes shadow
forecasts only") and backlog P1.24. No orders, no credentials, no execution path.

## 1. Purpose and the two components

P1.33 named `carry_s10_l4w_h26w_exit` as the release candidate on nine locked folds. If the
holdout confirms it, the protocol allows shadow forecasts: the book the strategy *would*
hold each week, computed from current data, with the P&L it *would* have earned, recorded
without any trade. The strategy's data today comes from Binance's monthly dumps, which
appear weeks after the fact, so a weekly shadow cannot run on them alone. Two things are
built:

1. **Weekly shadow captures and the shadow book.** Extend the panel captures every week
   through the last Sunday with daily kline dumps and REST funding history, verified against
   the monthly dumps once those appear; compute the book from those captures with the
   existing fold runner; seal one artifact per week.
2. **A permanent measurement stream.** A second, unbounded Binance journal that records,
   every 61 seconds, the current funding premium and basis of every perpetual, the best
   bid and ask of every perpetual and every spot pair, and the order-book depth of the
   fifteen cost pairs the first journal measures. It is the live time series a later CTM
   family (abstention/regime over this book, inputs = measurements, output m ∈ {+, −, 0})
   would consume, and the continuous cost comparison the paper phase requires.

Directional price prediction is out of scope by the user's decision; the measurement
stream records prices only as basis and spread, never as a forecast target.

## 2. Phasing (what may run before the holdout is read)

- **Phase A, now:** both components may run on real data, but the shadow book is computed
  on real data only for a pipeline check whose artifacts are `status: development_only`
  and whose weekly P&L is *not* summarised or reported, so that no post-holdout number
  reaches the go/no-go decision on the holdout read. Concretely, Phase A produces weekly
  captures, verifies them, runs the measurement stream, and runs the book computation on
  the fixture suite only.
- **Phase B, after `holdout_confirmed`:** weekly shadow artifacts with the book and the
  shadow P&L, reported to the user each week. `holdout_failed` ends the family; the
  captures and the measurement stream continue for whatever family comes next.

## 3. Weekly shadow captures

### 3.1 What a weekly capture is

A capture directory in the existing format (`capture-manifest.json`, `raw/`, `dataset/`),
immutable once published, verified by `verify_panel_capture` unchanged. Its `sources`
rows are the union of:

- every row of a **base capture** (the latest repaired monthly capture — today the
  2026-09-10/11 originals, from October the holdout-extension captures), payloads copied
  byte for byte, hardlinked when the filesystem allows, so `raw_sha256` is unchanged;
- **daily kline rows** (new kind `klines_daily_tail`) for every day after the base's last
  month through the target Sunday, one dump per symbol and day from
  `data.binance.vision/.../daily/klines/<symbol>/1d/<symbol>-1d-<date>.zip`, fetched only
  for days the base does not cover;
- **REST funding rows** (new kind `fundingRate_rest`) for the perpetual market, one request
  per symbol covering the same span (`GET /fapi/v1/fundingRate?symbol=&startTime=&endTime=&limit=1000`),
  the raw JSON response stored and hashed like a dump.

Rows of the same (kind, symbol, month) in the base are never refetched, so every weekly
capture is a verified superset of the base under `capture_lineage.verify_capture_superset`,
and of the previous week's capture. The manifest adds `base_capture_root_hash` and
`tail_through` (the last calendar day covered).

### 3.2 Reconciliation against the monthly dumps

When a monthly dump appears for a month the tail covered, the next weekly capture takes
that month from the dump (a new base) and compares: every daily-tail bar and every REST
funding settlement of that month must equal the dump's row for the same key
(open time, close, volume, quote volume; settlement time, rate). A mismatch is recorded in
the manifest under `reconciliation` with the rows involved and refuses the weekly capture
until a person has read it. Binance's dumps are generated from the same data as the daily
dumps and the REST history, so a mismatch is evidence of a data problem, not noise.

### 3.3 Cadence and deadlines

The decision is the Sunday close (23:59:59.999 UTC). Binance publishes a day's dump the
next day; the weekly capture runs Monday 06:00 UTC and requires the Sunday bar to be
present for every symbol that has a bar the day before; a symbol whose Sunday dump is
missing while its Saturday dump exists makes the capture wait (retry hourly, up to 24 h),
then refuse. Funding via REST is available immediately after each settlement. The capture
records `data_available_time_ns` per row (the time the payload was fetched), so a later
reader can tell how stale any decision's inputs were — the protocol's "stale or missing
data produces no trade" rule for the paper phase starts here.

## 4. The shadow book

### 4.1 Computation

The book of Sunday S is a pure function of the declaration, the weekly capture through S,
and a fixed anchor Sunday A: `evaluate_carry_decisions` (the fold runner's loop, unchanged)
is run over every Sunday from A to S with `candidate_names` = the release candidate and
the two dominance controls; the last decision's slot state and leg weights are the book.
There is no persisted slot state and nothing to corrupt: any week's book is recomputable
from data. The run is cheap (one member, a few dozen Sundays).

`A` is fixed in the shadow declaration as the first Sunday after the holdout window's last
exit, 2026-09-13; decisions before A are never evaluated. Bars before A are read only as
history (universe selection needs 91 days, the signal 4 weeks), exactly as a fold reads
the bars before its first decision.

### 4.2 The weekly artifact

`artifacts/shadow/<family>/<S>.json`, sealed with `report_hash`, immutable, registered in
`artifacts/shadow/metadata-shadow.sqlite3` (kind `shadow_week`, id `uuid5(family, S)`):

- `status` (`development_only` in Phase A, `shadow` in Phase B), the family spec hash, the
  weekly capture root hashes (both markets), `base_capture_root_hash`, the anchor, the
  code hash, `data_available_time_ns` of the latest input;
- the **book**: per slot the pair, entry Sunday, weeks held, leg weights; the union weights
  per leg;
- the **measurements per pair** for the week, for every pair the universe ranked (not only
  the ten held): trailing one- and four-week funding, the exit-rule verdict, the
  measurement stream's mean premium, basis and spread over the week (section 5), the
  cost tier;
- the **previous week's shadow P&L**: the episode the runner evaluated for S−1 with the
  declaration's base and adverse tables — net return, funding collected, trading cost,
  forced closes — and the running totals since A;
- the `m` **labels** the future CTM family needs: per held slot, whether the next week's
  measured funding net of measured cost was positive, negative or inside the cost band.

Phase A artifacts omit the P&L block and the running totals entirely, not just their
summary.

### 4.3 What a shadow week is not

No order is generated, no venue key exists, no execution adapter is imported (P1.24's
acceptance: shadow mode has no execution credentials or live client factory). The artifact
is a forecast of a book, recorded before the week it applies to.

## 5. The permanent measurement stream

A second journal, `binance-measurement-journal/1.0.0`, beside the cost journal, with its
own spec, run id, hash-chained segments, `verify`, `status` and a `snapshot` command in
place of `finalize`:

- **Every 61 seconds**, one round of four public requests: `GET /fapi/v1/premiumIndex`
  (all perpetuals; current funding rate, next funding time, mark and index price → basis;
  weight 10), `GET /fapi/v1/ticker/bookTicker` (all perpetuals; best bid/ask → spread;
  weight 5), `GET /api/v3/ticker/bookTicker` (all spot pairs; weight 4), and the depth
  walk of the fifteen cost pairs at 500 / 5,000 / 50,000 USDT exactly as the cost journal
  does (continuity of the slippage tiers; the same per-request weights and back-off rules).
  A round is one segment; the whole all-symbol responses are stored, so no symbol list
  has to be re-declared when the book changes.
- **No target rounds.** The journal runs until stopped; the Startup launcher restarts it;
  the supervisor's liveness is the newest segment's age, as learned on 2026-09-17.
- **`snapshot`** seals the chain head at a chosen time into an immutable receipt-like
  document with, per pair, the week's mean and last premium, basis and spread, and the
  fifteen pairs' tier medians for the week. The weekly shadow artifact cites the snapshot
  hash of its week. A snapshot never changes a declared cost table: the family's tiers stay
  the receipt's; the stream is context and the paper phase's cost monitor.
- The cost journal v1 finishes its 11,000 rounds and stays immutable; v2 starts beside it
  and takes over the fifteen pairs' depth measurement.

## 6. Failure handling (fail closed, as everywhere here)

- Weekly capture: any fetch that is not a clean 200 with a parseable payload is retried
  with the venue's `Retry-After`; a symbol still missing after the deadline refuses the
  capture; a reconciliation mismatch refuses; the storage policy's reserve applies; nothing
  is written under `raw/` twice.
- Shadow book: refuses if the weekly capture does not verify, if it is not a superset of
  the base, if the Sunday bar is absent for the pairs the book holds, or if the family spec
  hash differs from P1.33's measured declaration. A refusal is a recorded artifact with
  `status: refused` and the reason, so a missing week is visible, not silent.
- Measurement stream: 429/418 back-off as in the cost journal; a round with any request
  failing is recorded as a partial segment with the failure, never dropped; the daily
  liveness check relaunches a dead supervisor.

## 7. Testing

Fixture captures (the carry fixtures) plus fixture daily dumps and a fake REST funding
payload: the weekly capture is a verified superset of its base; a changed payload refuses;
reconciliation against a fixture monthly dump passes and a deliberately altered bar refuses;
the shadow book of the fixture family equals the fold runner's last decision on the same
data (the pin that proves "shadow = fold mechanics"); Phase A artifacts carry no P&L
fields; the measurement journal's rounds verify and a snapshot's per-pair values equal a
hand computation over the segments; no network in tests; `mypy --strict` and `ruff` clean.

## 8. Modules

- `shadow_capture.py`: weekly capture builder (base + daily tail + REST funding,
  reconciliation, `tail_through`), reusing `panel_capture`'s fetchers, dataset publisher
  and `verify_panel_capture`.
- `shadow_book.py`: anchor run over `evaluate_carry_decisions`, artifact assembly, sealing,
  registry, Phase A/B switch.
- `binance_measurement_journal.py`: spec, round, segment chain, `snapshot`; shares the
  depth walk and back-off with `binance_cost_journal.py`.
- `cli.py`: `shadow-capture`, `shadow-week`, `binance-measurement-journal-create|run|status|snapshot`.
- `configs/shadow-carry-v4.json`: the shadow declaration (family spec hash, anchor,
  phase, artifact roots).
- Operations (outside the repo): a Monday 06:00 UTC supervisor for capture + week, a
  Startup launcher for the measurement journal, the daily liveness check.

## 9. What this does not decide

The holdout read (the user's go), the paper-phase revision of protocol section 13 for
weekly books, the tiny-live cap and venue, and the CTM family over this book — each is a
later, separate decision. This design only makes the shadow phase startable the day after a
confirmed holdout and starts recording the measurements a CTM family would need.
