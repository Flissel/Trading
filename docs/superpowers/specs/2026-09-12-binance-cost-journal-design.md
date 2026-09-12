# Binance Cost Journal Design

Status: Decided autonomously on 2026-09-12 under the standing goal "make it better, decide
autonomously until a reliable solution exists". `research_only`. Measurement
infrastructure; no live, paper, or shadow authority follows from it.

## 1. Purpose

P1.28 showed that the funding carry's fate is decided by execution cost: the long-hold
member cleared every statistical gate and failed only on the declared adverse slippage
(doubled from 5/10 bps per side) plus the receipt haircut. Those slippage tiers are
assumptions inherited from P1.27, not measurements. The OKX public cost journal that is
running measures one instrument (BTC-USDT-SWAP) on another venue at a 100 USDT notional
— it cannot stand in for Binance spot and USD-M perpetual pairs at carry-sized clips.

This journal measures, on Binance's public order books, what a taker pays to cross a
spot leg and a perpetual leg at three notionals, for a fixed sample of liquid pairs,
every minute, for at least seven days, hash-chained and resumable like the OKX journal.
Its finalisation receipt is the only admissible source of *measured* slippage tiers for a
later carry declaration (`funding_carry_panel_v3`). The journal itself decides nothing.

## 2. What it measures

For each sampled instrument and each round:

- `spread_bps` = `(best_ask − best_bid) / mid × 10 000`.
- `slippage_bps_per_side[notional]` for notional ∈ {500, 5 000, 50 000} USDT: walk the
  displayed book from the best level until `notional` is filled; cost per side =
  `(vwap − mid) / mid × 10 000` for a buy (asks) and `(mid − vwap) / mid × 10 000` for a
  sell (bids); record the **worse of the two sides**. If the displayed book cannot fill
  the notional within the fetched depth, the value is recorded as `null` and counted as
  `insufficient_depth` — that is an observation about executability, not a gap.
- `displayed_notional_thinner_side`: total displayed notional within the fetched depth on
  the thinner side, in USDT (for the record; no gate).
- For perpetuals additionally: `funding_rate` (as received) and `basis_bps` derived from
  `premiumIndex` (`(mark − index) / index × 10 000` at the sample instant).
- Measured bps values are quantised to 1e-6 bps and the displayed notional to 0.01 USDT at
  observation time; this is the journal's declared measurement precision.

Fees are not measured: the receipt carries the declared taker fees (spot 10 bps, USD-M
5 bps, standard tier, no BNB discount, verified 2026-09-11) with an evidence id.

## 3. Sample

Instruments are fixed at journal creation from the latest decision the two P1.28
captures support (2026-08-24): the pairs whose spot leg and perpetual leg both clear the
family's liquidity rule on that date, ranked by perpetual median quote volume. Sixteen
pairs: ranks 1–8 (tier one) and eight drawn evenly from ranks 9–35 (tier two; the pairs at
ranks 9, 12, 15, 19, 23, 27, 31, 35 by that ranking). Both legs of each pair are sampled,
32 instruments per round. The list is written into the journal spec file at creation and
never changes; a new sample is a new journal.

Endpoints (public, unauthenticated, HTTPS only):

- Spot depth: `https://api.binance.com/api/v3/depth?symbol=<S>&limit=500` (weight 25)
- USD-M depth: `https://fapi.binance.com/fapi/v1/depth?symbol=<S>&limit=500` (weight 10)
- USD-M premium index: `https://fapi.binance.com/fapi/v1/premiumIndex?symbol=<S>` (weight 1)

One round is 16 spot depth + 16 perpetual depth + 16 premium-index requests ≈ 576 weight
per minute against limits of 6 000 (spot) and 2 400 (USD-M) per minute. Hosts are
allow-listed; any other host is refused.

## 4. Cadence, eligibility, chain

- Sample interval 61 seconds between rounds (as the OKX journal); target 10 000 rounds.
- One segment per round holding every instrument's observation; segments are numbered,
  carry `previous_segment_hash`, `spec_hash`, `received_time_ns` and `content_hash`; a
  chain head records the last sequence and final hash. Same shapes and verification as
  the OKX journal's V1, versioned `binance-cost-journal/1.0.0`.
- A request failure (transport, non-200, malformed) for one instrument records that
  instrument's observation as `null` with a reason string; the round is still written.
  A failure of every instrument in a round aborts the process (the supervisor restarts).
- Resumable: on start the chain head is read and verified; the next sequence continues.
  The journal directory is bound to its spec hash; a different spec refuses to resume.
- Eligibility, per instrument: ≥ 10 000 non-null observations at the 5 000 USDT notional
  and ≥ 7 days between first and last non-null observation. The receipt lists eligible
  and ineligible instruments; ineligible ones carry no tier statistics.

## 5. Finalisation receipt

Per eligible instrument and notional: count, `null` count, p50, p90 and p99 of
`slippage_bps_per_side`, p50 of `spread_bps`; for perpetuals p50 of the basis.
Per tier (one, two) and leg (spot, perpetual) at each notional: the median of the
instruments' p50s (`tier_p50_of_p50`) and the median of their p90s (`tier_p50_of_p90`).
The receipt carries the sample list, the fee evidence, the spec hash, the chain head hash,
the observation count and its own hash; it is immutable.

**How a carry declaration may use it (declared here, before any number exists):** a v3
family sets `slippage_bps_per_side_tier_<t>` in its base table to `tier_p50_of_p50` of the
*worse leg* at 5 000 USDT and in its adverse table to `tier_p50_of_p90` at 50 000 USDT,
rounded **up** to the next whole basis point, and cites the receipt hash. No other
reading of the receipt is admissible for a declaration.

## 6. Modules

- `src/trading_bot/binance_cost_journal.py`: spec model (`BinanceCostJournalSpec`),
  observation and segment models, `create_journal(...)`, `run_journal(...)` (resumable
  loop with injected fetcher), `verify_journal(root)`, `finalize_journal(root, output)`;
  the book walk reuses the level parsing of `depth_adapters.BinanceDepthAdapter` where it
  fits and otherwise validates levels the way `cost_capture._validated_levels` does.
- `cli.py`: `binance-cost-journal-create` (writes the spec from the two captures),
  `binance-cost-journal-run`, `binance-cost-journal-finalize`.
- Supervisor `C:\Users\User\.trading-jobs\binance-cost-journal.ps1` plus a Startup `.cmd`,
  mirroring the OKX journal's; journal root `data/cost-journals/binance-carry-v1`.

## 7. Non-goals

No order placement, no authenticated endpoints, no fee measurement, no OKX changes, no
change to any carry family. The journal does not read the carry captures after creation.
