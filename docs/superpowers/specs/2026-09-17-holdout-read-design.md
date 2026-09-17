# Final Holdout Read Design (panel and carry families)

Status: Decided autonomously on 2026-09-17 under the standing goal "make it better, decide
autonomously until a reliable solution exists", after the user accepted the phase plan
("ok") on 2026-09-17. Fixes, before any holdout is opened, how the final holdout of a panel
family is read, for whom, and what counts as confirmation. Adds section 16.2 to
`PHASE_0_EVALUATION_PROTOCOL.md` (prospective; it binds every family whose holdout is
still unopened, which on 2026-09-17 is every family). Opening a holdout stays a separate,
explicit user decision; this design only makes that step executable and single-use.

## 1. Purpose

Protocol section 2.3: the final holdout is opened once for a named release candidate; a
failed holdout does not become a validation set; any post-holdout change is a new research
generation. Until now no command could read a panel holdout, no criterion said what a
holdout confirmation is, and nothing prevented a second read. P1.33's provisional run
(2026-09-17) made a first candidate likely, so these three gaps are closed now, before the
official P1.33 decision exists.

## 2. The holdout window and its data

The panel manifest places the holdout at the chronological end of the capture:
`final_holdout_start_ns = last decision − holdout_duration + spacing`, so its 26 weekly
decisions run 2026-03-08 to 2026-08-30 and the last episode exits 2026-09-06. The
captures of 2026-09-10/11 end with the August monthly dumps; the last episode needs the
September daily bars of both markets and the September funding settlements, which Binance
publishes as a monthly dump in early October.

Rule: the holdout is read with the **original manifest's** `final_holdout_ids` and
calendar — never a re-derived one, which an extended capture would shift — and with
**extended captures** (perpetual and spot) that are verified supersets of the originals:
every `sources` row of an original capture manifest (kind, symbol, month) must appear in
the extended capture's manifest with the identical `raw_sha256` and status. A row that is
absent or differs refuses the read. The holdout artifact binds both the original and the
extended capture hashes.

## 3. Who is read

The holdout is opened for exactly one member. The candidate is not chosen by hand: it is
the eligible member of the family's decision artifact with the highest adverse total net
return **after** subtracting its pooled `uncharged_final_exit_cost` under the adverse
scenario (slot families; zero for cohort families), ties broken by declaration order. The
command derives the candidate itself and refuses if the decision lists no eligible member.

Only the candidate and the two dominance controls (`no_trade`, `random_pairs`) are
evaluated on the holdout. The other members and the context control are **not** evaluated:
their holdout is never read, so a later generation is not informed by it.

## 4. What is evaluated

Exactly the family's fold mechanics on the holdout decisions: the same spec (for P1.33 the
measured declaration the decision was made under), the same accounting, both cost
scenarios, warm-up per the family's rule (0 for a slot family; `max(H) − 1` Sundays before
the first holdout decision for a cohort family, which lie inside the last fold's span and
read only data at or before themselves), bars loaded with `available_before_ns` = last
holdout exit + 1. The report has the fold-report schema plus `holdout: true`, the candidate
name, the decision report hash, the original manifest hash, the four capture hashes and
the confirmation verdict.

## 5. Confirmation criteria (fixed here, before any read)

On the candidate's holdout episodes, all of the following:

1. base total net return > 0;
2. adverse total net return − adverse `uncharged_final_exit_cost` ≥ 0;
3. base total > `no_trade` (0) and > `random_pairs` base total; adverse total after the
   subtraction ≥ `random_pairs` adverse total and ≥ 0;
4. largest single episode's share of the base total ≤ 0.5 and largest pair share ≤ 0.5
   (the fold gates' concentration rule, on the holdout alone);
5. no `UNIVERSE_TOO_SMALL` skip may remove more than 4 of the 26 decisions.

No statistical test is applied: 26 weeks confirm a sign and a magnitude, they do not
discover. Reported, never gated: the holdout's mean weekly net under both scenarios, the
fraction of positive weeks, and whether the holdout base mean lies inside the folds'
bootstrap interval of the pooled mean. Verdict `holdout_confirmed` or `holdout_failed`.

## 6. Single use

The output path is immutable and the registry records an artifact of kind `holdout` whose
id is derived from the family id alone; a second read of the same family refuses on either.
A failed holdout ends the family's generation (protocol 2.3): no re-read, no re-declaration
of the same family; a successor family needs a later untouched period.

## 7. What a confirmation authorises

Protocol section 13 unchanged: shadow forecasts only. Paper needs the real-time
infrastructure of section 13 and the prospective definition of paper episodes for weekly
books, which is a separate protocol revision the user decides.

## 8. Modules

- `capture_lineage.py`: `verify_capture_superset(original_root, extended_root) -> tuple[bool, tuple[str, ...]]` over the two capture manifests' `sources`.
- `carry_fold_run.py`: the in-window loop extracted into a function that takes explicit
  decision ids, the candidate filter and the warm-up count, used by `run_carry_fold`
  (unchanged output — the v1 fold-0 pin and the v2/v4 tests guard it) and by the holdout.
- `carry_holdout_run.py`: `run_carry_holdout(perp_root, spot_root, *, original_perp_root,
  original_spot_root, manifest_path, family_spec_path, decision_path, fold_report_paths,
  output_path, registry_path)`: derives the candidate, checks lineage, evaluates, applies
  section 5, writes the sealed report, registers the single-use artifact.
- `cli.py`: `carry-holdout`.
- `PHASE_0_EVALUATION_PROTOCOL.md`: section 16.2.

## 9. Artifacts

`artifacts/carry/funding-carry-v4-holdout-v1.json`, registry
`artifacts/carry/metadata-funding-carry-v4.sqlite3` (artifact kind `holdout`), record
`P1_33_HOLDOUT_<date>.md`; extended captures under `data/captures/<date>-binance-um-usdt-perps-1d-repaired` and `…-spot-usdt-1d-repaired` (October).

## 10. Decision rule

`holdout_confirmed` names the release candidate for the shadow phase; the next step is the
weekly shadow job and the paper-phase protocol revision. `holdout_failed` closes the v4
generation; the carry line continues only with a new family evaluated on a later untouched
period. No threshold moves either way.
