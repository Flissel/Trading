# Hybrid Trading Research Stack

Lokales, fail-closed Research- und Backtest-System für BTC/ETH-Perpetuals. Der
aktuelle Stand ist absichtlich **kein Live-Trading-Bot** und enthält keine
Order-Route, API-Schlüssel oder Profitabilitätsbehauptung.

## Enthalten

- strikte kanonische Markt-Events und deterministische Hashes;
- lokale Storage-Grenzen mit Ausschluss von `E:` und konfigurierbarer Reserve;
- OKX- und Binance-Importer sowie venue-spezifische L2-Depth-Adapter;
- fail-closed Orderbuch-Rekonstruktion mit Sequenzprüfung;
- point-in-time Features und getrennte Zukunftslabels;
- Walk-forward-Splits mit Purge, Embargo und isoliertem Holdout;
- No-trade-, Random-, Momentum- und Mean-Reversion-Baselines;
- Base-/Adverse-Kosten, Block-Bootstrap und konservative Promotion;
- deterministische Risk-Policy für EUR 100 Referenzkapital;
- idempotenter Ausführungssimulator mit Partial-Fill und Unknown-Outcome;
- End-to-End-Backtest mit Fill-Lineage, Audit-Hash-Kette und JSON-Artefakt.

## Installation und Prüfung

```powershell
uv sync
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src tests
```

## Lokaler Demo-Backtest

Die Demo verwendet ausschließlich synthetische Fixtures und hält standardmäßig
20 GB freien Platz auf `C:` zurück:

```powershell
uv run trading-research demo-backtest `
  --workspace-root . `
  --output artifacts/demo-report.json `
  --reserve-bytes 20000000000
```

Das JSON enthält Base- und Adverse-Ergebnisse, Promotion-Status, Risk-/Fill-
Lineage, eine verifizierte Audit-Kette und einen semantischen Report-Hash.

## Begrenzter öffentlicher Marktdaten-Capture

Der Capture lädt ohne API-Schlüssel höchstens die angegebene Zahl historischer
15-Minuten-Candles von OKX und Binance USD-M. Er speichert die empfangenen
Rohbytes unverändert, normalisiert bestätigte Candles und veröffentlicht erst
nach erfolgreicher Prüfung ein partitioniertes Parquet-Dataset:

```powershell
uv run trading-research capture-public-candles `
  --workspace-root . `
  --output data/captures/btc-15m-sample `
  --reserve-bytes 20000000000 `
  --limit 96
```

Ein bestehender Capture wird niemals überschrieben. Netzwerk-, Schema-,
Speicher- oder Qualitätsfehler hinterlassen keinen veröffentlichten
Teildatensatz. Die Parquet-Partitionen können direkt mit DuckDB abgefragt
werden.

Für längere, weiterhin strikt begrenzte Historien paginiert `capture-history`
rückwärts. Jede HTTP-Seite wird separat unverändert gespeichert und im
Capture-Manifest gehasht. Standardmäßig werden 2.880 abgeschlossene
15-Minuten-Candles je Venue erfasst (ungefähr 30 Tage):

```powershell
uv run trading-research capture-history `
  --workspace-root . `
  --output data/captures/btc-15m-30d `
  --reserve-bytes 20000000000 `
  --bars 2880
```

Der automatisch eingefrorene Endzeitpunkt liegt unmittelbar vor der aktuell
laufenden Kerze. Binance-Candles gelten nur dann als bestätigt, wenn ihre
Schlusszeit beim Empfang bereits beobachtbar war.

## Development-OOS-Research auf einem Capture

Ein vollständig hash-verifizierter Capture kann durch die kausale Bar-Feature-
und Baseline-Pipeline laufen. Der letzte chronologische Anteil wird als OOS
verwendet; der Status bleibt bei kurzen Bootstrap-Captures technisch auf
`development_only`:

```powershell
uv run trading-research research-capture `
  --workspace-root . `
  --capture data/captures/btc-15m-sample `
  --output artifacts/research-btc-15m.json `
  --minimum-train-samples 20 `
  --oos-fraction 0.20 `
  --random-seed 17
```

Der Report vergleicht No-trade, Momentum, Mean-Reversion und eine reproduzierbare
Zufallsbaseline unter denselben Base- und Adverse-Kosten. Positive Ergebnisse
auf kurzen Captures sind Diagnosewerte und keine Promotionsevidenz.

## Unveränderliches Walk-forward-Manifest

Ein ausreichend langer, vollständig verifizierter Capture kann in feste
Train-/Validation-/Test-Mitgliedschaften und einen unangetasteten finalen
Holdout materialisiert werden. Das folgende Profil verwendet 365/90/30 Tage,
30 Tage Schrittweite, 4 Stunden 15 Minuten Embargo und 30 Tage Holdout:

```powershell
uv run trading-research walk-forward-manifest `
  --workspace-root . `
  --capture data/captures/btc-15m-520d `
  --output artifacts/walk-forward-btc-15m-520d.json `
  --train-duration-ns 31536000000000000 `
  --validation-duration-ns 7776000000000000 `
  --test-duration-ns 2592000000000000 `
  --step-ns 2592000000000000 `
  --embargo-ns 15300000000000 `
  --holdout-duration-ns 2592000000000000
```

Samples mit Labels über einer Partitionsgrenze werden entfernt. Der Holdout
liegt am chronologischen Ende des Captures und seine IDs werden ausschließlich
im Manifest veröffentlicht; der Evaluator öffnet ihn noch nicht.

## Registrierte Fold-Evaluation

`evaluate-fold` liest Parquet bereits auf Datenbankebene nur bis zur jeweiligen
Testgrenze. Es vergleicht alle Baselines unter identischen Kosten, berechnet
Mittelwert und Median der aktiven Episoden, einen nullzentrierten
Moving-Block-Bootstrap und Benjamini-Hochberg-korrigierte q-Werte:

```powershell
uv run trading-research evaluate-fold `
  --workspace-root . `
  --capture data/captures/btc-15m-520d `
  --split-manifest artifacts/walk-forward-btc-15m-520d.json `
  --output artifacts/fold-evaluation-btc-15m-fold0.json `
  --registry artifacts/research-metadata.sqlite3 `
  --fold-index 0 `
  --random-seed 17 `
  --block-length 16 `
  --bootstrap-repetitions 2000
```

Jeder Baseline-Versuch wird append-only in SQLite registriert. Fehlgeschlagene
Versuche besitzen einen Grundcode statt eines Resultathashes. Ein Report mit
zu wenigen Folds oder Episoden bleibt `development_only`; unregistrierte
Reports können keine Promotion autorisieren.

Mehrere Fold-Reports werden nur dann aggregiert, wenn alle erwarteten Indizes
vorhanden sind und Capture-, Dataset- sowie Split-Hashes übereinstimmen. Der
Aggregate-Gate prüft mindestens zwei Drittel positive Base-Folds, positive
Base- und Adverse-Gesamtergebnisse sowie den schlechtesten BH-q-Wert. Ein
fehlender Fold oder eine abweichende Experimentfamilie stoppt fail-closed.

## Kleine lineare Challenger-Baseline

Die Ridge-Baseline verwendet vier Marktfeatures und benötigt keine externe
ML-Laufzeit. Mittelwerte, Skalen und Koeffizienten werden ausschließlich auf
dem Trainingsfold gelernt. Eine begrenzte Menge fester Prognosequantile wird
nur auf Validation nach Nettokosten ausgewählt. Ein Threshold ist nur zulässig,
wenn er die konfigurierte Mindestanzahl an Validierungstrades erreicht; dadurch
kann ein einzelner Glückstreffer nicht gegen eine breiter abgestützte Schwelle
gewinnen. Test und finaler Holdout beeinflussen weder Fit noch Schwelle.
Modellparameter, Auswahl-Tradezahl und Membership-Hashes werden im Fold-Report
festgehalten.

## Kalibrierter Logistic-Challenger

Der Logistic-Challenger prognostiziert die Wahrscheinlichkeit einer positiven
nächsten Bar-Rendite mit denselben vier train-only standardisierten Features.
Die Validation wird chronologisch in zwei disjunkte Teile zerlegt: Auf der früheren
Hälfte wird ausschließlich der Platt-Kalibrator gefittet, die spätere Hälfte wählt eine
kostenbereinigte Long/Short/Hold-Margin mit Mindest-Tradezahl. Der Test-Fold
bleibt von beiden Schritten ausgeschlossen. Raw- und kalibrierter Brier Score
sowie Log-Loss werden auf Calibration, Selection und Test soweit anwendbar
getrennt ausgewiesen; positive In-Sample-Kalibrierung gilt nicht als OOS-Beleg.

## Kleiner Tree-Challenger und Modell-Dominanz

Der Tree-Challenger besteht aus höchstens 16 deterministischen Regression-
Stumps. Splitkandidaten, Residuen, Blattwerte und Gain-Importance entstehen
ausschließlich aus dem Trainingsfold. Mindestblattgröße und eine begrenzte
Anzahl train-only Quantilschwellen halten die Modellkapazität klein. Validation
wählt weiterhin nur den kostenbereinigten Long/Short/Hold-Threshold mit
Mindest-Tradezahl. Feature Importance bleibt eine Diagnose und kann keine
Promotion begründen.

Ein separater, hash-verifizierter Dominanzreport bindet die Aggregate der
einfachen Regeln, Ridge-, Logistic- und Tree-Challenger an dieselben Capture-,
Dataset- und Split-Hashes. Ein Kandidat dominiert No-trade nur bei besserer
Base- und Adverse-Metrik, ausreichender Fold-Stabilität, bestandenem q-Gate und
bereits bestehender Eignung zur weiteren Prüfung. Der numerisch beste Verlierer
wird dadurch nicht als Edge ausgewiesen.

## Horizon-sichere 1h- und 4h-Auswertung

`walk-forward-manifest` akzeptiert `--horizon-bars`. Bei 15-Minuten-Bars stehen
`4` für eine Stunde und `16` für vier Stunden. Der Horizont wird in Sample-IDs,
Label-Verfügbarkeit, Manifesten und Fold-Reports gebunden; Fold-Runner
rekonstruieren ihre Samples ausschließlich mit dem im Manifest deklarierten
Horizont. Alte Manifeste ohne das Feld bleiben als Ein-Bar-Manifeste lesbar.

Die reale 580-Tage-Auswertung vom 25. August 2026 ergab auf beiden Horizonten
`no_eligible_candidate`. Der 4h-Tree war unter Basiskosten positiv, scheiterte
aber am adversen Kostenfall und am Multiple-Testing-Gate. P1.15 ist daher nicht
freigegeben. Der vollständige Gate-Nachweis steht in
`P1_15_DECISION_2026-08-25.md`; der finale Holdout blieb gesperrt.

## Panel-Forschung (P1.27)

Neben der 15-Minuten-BTC-Linie gibt es eine zweite, davon unabhängige Linie:
ein Wochen-Rebalancing auf einem Panel von Binance-USD-M-USDT-Perpetuals aus
den öffentlichen Tagesdumps. Ein Sample ist ein Rebalance-Termin, nicht ein
Coin-Tag; die Kontrakte innerhalb einer Woche sind keine unabhängigen Samples
(Protokoll Abschnitt 16). Vier Befehle bilden die Kette:

```powershell
uv run trading-research panel-capture --output data/captures/<datum>-binance-um-usdt-perps-1d --symbols <liste> --months <liste>
uv run trading-research panel-manifest --capture <capture> --output artifacts/panel-walk-forward-v1.json --family-spec configs/xs-momentum-panel-v1.json
uv run trading-research panel-fold --capture <capture> --manifest <manifest> --family-spec configs/xs-momentum-panel-v1.json --output artifacts/panel/fold0.json --registry artifacts/panel/metadata-xs-momentum-v1.sqlite3 --fold-index 0
uv run trading-research panel-decision --fold-report artifacts/panel/fold0.json --family-spec configs/xs-momentum-panel-v1.json --output artifacts/panel/decision-v1.json --registry artifacts/panel/metadata-xs-momentum-v1.sqlite3
```

Die Familie ist vor dem ersten Datenabruf eingefroren: sechs Mitglieder, drei
Kontrollen, keine Parametersuche, alles in `configs/xs-momentum-panel-v1.json`,
dessen SHA-256 in jedem Report als `family_spec_hash` steht.

### Funding-Carry (P1.28)

Zweite Panel-Familie: long Spot, short Perpetual, ausgewählt nach dem zuletzt gezahlten
Funding, gehalten über überlappende Wochen-Kohorten. Das Spot-Bein kommt über
`panel-capture --market spot`, das Manifest bindet beide Captures, der Fold-Runner
schreibt das P1.27-Reportschema, und `panel-decision` bleibt dieselbe Instanz.

```powershell
uv run trading-research panel-capture --market spot --output data/captures/<datum>-binance-spot-usdt-1d --symbols <liste>
uv run trading-research panel-manifest --capture <perp> --hedge-capture <spot> --output artifacts/carry/carry-walk-forward-<datum>-usdt-pairs-1d-w1-v1.json --family-spec configs/funding-carry-panel-v1.json
uv run trading-research carry-fold --capture <perp> --hedge-capture <spot> --manifest <manifest> --family-spec configs/funding-carry-panel-v1.json --output artifacts/carry/funding-carry-<datum>-fold0-v1.json --registry artifacts/carry/metadata-funding-carry-v1.sqlite3 --fold-index 0
uv run trading-research panel-decision --fold-report artifacts/carry/funding-carry-<datum>-fold0-v1.json --family-spec configs/funding-carry-panel-v1.json --output artifacts/carry/funding-carry-<datum>-decision-v1.json --registry artifacts/carry/metadata-funding-carry-v1.sqlite3
```

Zweite Version der Familie (P1.29): `configs/funding-carry-panel-v2.json` deklariert vier
Mitglieder, die drei Regeln einzeln und gestapelt testen — `carry_l4w_h26w` (nur Hold 26
Wochen), `carry_l4w_h13w_exit` (Hold 13 plus Exit, sobald das nachlaufende Ein-Wochen-Funding
eines gehaltenen Paars nicht mehr positiv ist), `carry_l4w_h26w_exit` (beides) und
`carry_l4w_h26w_exit_hurdle2` (zusätzlich Aufnahme nur, wenn das Funding das Doppelte der
Round-Trip-Kosten deckt). Manifest, Folds, Entscheidung und Registry heißen
`artifacts/carry/carry-v2-walk-forward-<datum>-usdt-pairs-1d-w1-v1.json`,
`artifacts/carry/funding-carry-v2-<datum>-fold<i>-v1.json`,
`artifacts/carry/funding-carry-v2-<datum>-decision-v1.json` und
`artifacts/carry/metadata-funding-carry-v2.sqlite3`. Ergebnis: `P1_29_DECISION_2026-09-12.md`.

### Binance-Kostenjournal (P1.30)

Misst auf Binance-Orderbüchern (Spot und USD-M), was ein Taker beim Überqueren beider Beine
bei 500 / 5 000 / 50 000 USDT zahlt — alle 61 s, hash-verkettet, resumefähig, für eine bei
Erstellung eingefrorene Stichprobe von Carry-Paaren. Die Finalisierungsquittung ist die
einzige zulässige Quelle *gemessener* Slippage-Tiers für `funding_carry_panel_v3`.

```powershell
uv run trading-research binance-cost-journal-create --journal data/cost-journals/binance-carry-v1 --run-id binance-carry-public-cost-v1 --perp-capture <perp> --spot-capture <spot> --family-spec configs/funding-carry-panel-v1.json
uv run trading-research binance-cost-journal-run --journal data/cost-journals/binance-carry-v1
uv run trading-research binance-cost-journal-status --journal data/cost-journals/binance-carry-v1 --last 60
uv run trading-research binance-cost-journal-finalize --journal data/cost-journals/binance-carry-v1 --output artifacts/cost/binance-carry-v1-receipt.json
```

`run` hält eine exklusive Sperre und wird von einem Supervisor neu gestartet (Exit 1);
Exit 2 bedeutet gestoppt. `status` ist nur lesend und der einzige sichere Blick auf ein
laufendes Journal.

Aus der Quittung wird `funding_carry_panel_v3` erklärt — Basis-Tier = `tier_p50_of_p50`
des schlechteren Beins bei 5 000 USDT, Adverse-Tier = `tier_p50_of_p90` bei 50 000 USDT,
je auf ganze Basispunkte aufgerundet (Spec 5). Alles andere ist die v2-Erklärung
unverändert; die Quittung wird gegen ihren eigenen Hash geprüft und zitiert:

```powershell
uv run trading-research carry-declare-measured --receipt artifacts/cost/binance-carry-v1-receipt.json --base-config configs/funding-carry-panel-v2.json --output configs/funding-carry-panel-v3.json
uv run trading-research carry-verify-measured --receipt artifacts/cost/binance-carry-v1-receipt.json --spec configs/funding-carry-panel-v3.json
```

Exit 2 heißt: so nicht erklärbar (Quittung ohne Median, falsche Basis-Erklärung,
Ausgabe existiert bereits). Die Ausgabe wird exklusiv angelegt (`open(..., "x")`) und ist
unveränderlich wie die Quittung — bricht der Prozess mitten im Schreiben ab, bleibt eine
unvollständige Datei liegen, die nicht parst: löschen und Befehl wiederholen.
`carry-verify-measured` leitet jede Zahl der Erklärung neu aus Quittung und v2-Erklärung
ab (Exit 2 bei jeder Abweichung) und druckt den Spec-Hash.

### Trend-Aggregat (P1.31)

Dritte Panel-Familie und die erste gerichtete: ein Trend-Score als Mittel aus zwölf
eingefrorenen Indikatoren über 20 bis 120 Tage Tagesschluss (gleitende Durchschnitte und
ihre Kreuzungen, Ausbrüche, Rate-of-Change, MACD), deklariert in
`configs/trend-aggregate-panel-v1.json`. Vier Mitglieder: `ta_ts_t02` und `ta_ts_t05`
(long ab Score 0.2 bzw. 0.5, short spiegelbildlich, eine Woche gehalten),
`ta_ts_t02_h4w` (Schwelle 0.2, vier Wochen in überlappenden Kohorten zu je einem Viertel
des Kapitals) und `ta_xs_q5` (oberstes gegen unterstes Quintil). Universum, Gewichte,
Kosten, Folds und Statistik sind wertgleich zu P1.27; Manifest und Entscheidung sind
`panel-manifest` und `panel-decision` unverändert, neu ist nur `trend-fold`.

```powershell
uv run trading-research panel-manifest --capture <perp> --output artifacts/trend/trend-walk-forward-<datum>-usdt-perps-1d-w1-v1.json --family-spec configs/trend-aggregate-panel-v1.json
uv run trading-research trend-fold --capture <perp> --manifest <manifest> --family-spec configs/trend-aggregate-panel-v1.json --output artifacts/trend/trend-aggregate-<datum>-fold0-v1.json --registry artifacts/trend/metadata-trend-aggregate-v1.sqlite3 --fold-index 0
uv run trading-research panel-decision --fold-report artifacts/trend/trend-aggregate-<datum>-fold0-v1.json --family-spec configs/trend-aggregate-panel-v1.json --output artifacts/trend/trend-aggregate-<datum>-decision-v1.json --registry artifacts/trend/metadata-trend-aggregate-v1.sqlite3
```

Jeder Fold wärmt die drei Sonntage vor seiner ersten Entscheidung auf (nur Gewichtsvektoren,
keine Episoden), damit das Buch des Vier-Wochen-Mitglieds voll eröffnet statt drei Wochen
lang hochzulaufen; der Report hält das als `warm_up_weeks` und
`FOLD_OPENING_BOOK_WARMED_FROM_PRIOR_WEEKS` fest. Pro Episode kommen drei Extras dazu:
`signalled_contracts`, `abstained_contracts` und `mean_score`.

### Funding-Querschnitt (P1.32)

Vierte Panel-Familie und die erste ohne Spot-Bein: ein dollarneutrales Buch, long im
Quintil mit der niedrigsten und short im Quintil mit der höchsten Funding-Rate desselben
Universums, deklariert in `configs/funding-xs-panel-v1.json`. Rangiert wird auf der Summe
der Funding-Settlements der letzten ein oder vier Wochen (P1.28s Definition); vier
Mitglieder: `fx_q5_l4w_h1w` (vier Wochen Rückblick, eine Woche gehalten),
`fx_q5_l4w_h4w` und `fx_q5_l1w_h4w` (vier Wochen in überlappenden Kohorten zu je einem
Viertel des Kapitals) und `fx_q5_l4w_h4w_exit` (wie das zweite, aber mit Ausstiegsregel:
ein gehaltenes Bein, dessen letzte Woche das falsche Vorzeichen zeigt, wird in jedem
gehaltenen Vektor auf null gesetzt, sein Kapitalanteil bleibt bis zum Auslaufen der
Kohorte unangelegt). Universum, Gewichte, Folds und Statistik sind wertgleich zu P1.27,
die Kosten bis auf eine deklarierte Änderung: im Adverse-Szenario zählen
Funding-Einnahmen zu 0.75 und Zahlungen doppelt (P1.28s Regel statt P1.27s Nullreceipt),
weil der Ertrag dieser Familie das Funding selbst ist. Manifest und Entscheidung sind
`panel-manifest` und `panel-decision` unverändert, neu ist nur `funding-xs-fold`.

```powershell
uv run trading-research panel-manifest --capture <perp> --output artifacts/fxs/fxs-walk-forward-<datum>-usdt-perps-1d-w1-v1.json --family-spec configs/funding-xs-panel-v1.json
uv run trading-research funding-xs-fold --capture <perp> --manifest <manifest> --family-spec configs/funding-xs-panel-v1.json --output artifacts/fxs/funding-xs-<datum>-fold0-v1.json --registry artifacts/fxs/metadata-funding-xs-v1.sqlite3 --fold-index 0
uv run trading-research panel-decision --fold-report artifacts/fxs/funding-xs-<datum>-fold0-v1.json --family-spec configs/funding-xs-panel-v1.json --output artifacts/fxs/funding-xs-<datum>-decision-v1.json --registry artifacts/fxs/metadata-funding-xs-v1.sqlite3
```

Die Fold-Schleife ist dieselbe wie bei P1.31 — sie liegt seit dieser Familie in
`vector_fold_run.py`, `trend-fold` und `funding-xs-fold` sind zwei dünne Hüllen darum,
und P1.31s Fold-1-Zahlen sind gegen `tests/fixtures/trend_fold1_expected.json`
festgenagelt. Aufwärmen und `warm_up_weeks` wie oben; pro Episode kommen vier Extras
dazu: `signalled_contracts`, `funding_collected` (das Gegenteil der Funding-Kosten der
Episode), `exit_rule_removals` und `mean_score`.

### Funding-Carry v4, kleines Buch (P1.33)

Fünfte Panel-Familie und die erste mit einer erklärten Kapitalgrenze: derselbe
Spot/Perpetual-Carry wie P1.28/P1.29, aber auf das Buch erklärt, das tatsächlich gefahren
würde — unter 10 000 USDT, in Orders von höchstens 500 USDT pro Bein statt der rund
zweihundert gleichzeitigen Paarpositionen von v2. `configs/funding-carry-panel-v4.json`
erklärt dafür einen `capital`-Block (`book_usdt` 10 000, `pair_slots` 10,
`per_leg_notional_usdt` 500, `fee_tier` Standard-Taker ohne BNB); die gemessene Form
`funding_carry_panel_v4_measured` liest die Journal-Quittung nach der neuen Regel in
Abschnitt 5.1 der Journal-Spec: Basis-Tier `tier_p50_of_p50` des schlechteren Beins bei
500 USDT, Adverse-Tier `tier_p50_of_p90` bei 5 000 USDT, je auf ganze Basispunkte
aufgerundet, `carry-verify-measured` leitet beide Werte aus Quittung und Basis-Erklärung
neu ab.

Vier Mitglieder — `carry_s10_l4w_h13w`, `carry_s10_l4w_h13w_exit`, `carry_s10_l4w_h26w`
und `carry_s10_l4w_h26w_exit` — testen 13 und 26 Wochen Haltedauer mit und ohne
Ausstiegsregel; drei Kontrollen begleiten sie: `no_trade`, `random_pairs` (dasselbe
Slot-Buch, in der zufälligen Reihenfolge von P1.28s Kontrolle statt nach Funding befüllt)
und, nur als Kontext, P1.28s `all_pairs_ew`, das beim erklärten Kapital nicht handelbar
wäre.

Statt Wochen-Kohorten führt die Familie ein Slot-Buch: zehn feste Plätze, die frei werden,
sobald ihr Paar `H` Wochen gehalten wurde oder ein Bein seinen Balken verliert (die beiden
Exit-Mitglieder räumen zusätzlich, sobald das nachlaufende Ein-Wochen-Funding eines
gehaltenen Paars nicht mehr positiv ist), und die aus dem obersten Dezil zahlender Paare
neu befüllt werden — gehaltene Paare und, bei den Exit-Mitgliedern, zuletzt nicht
zahlende Paare übersprungen; ein im selben Schritt frei- und wieder befülltes Paar behält
seine Gewichte und löst keinen Turnover aus. Es gibt kein Warm-up: jeder Fold startet mit
leerem Buch (`warm_up_weeks` 0).

Pro Episode kommen vier neue Extras hinzu — `filled_slots`, `slot_fills`,
`slot_releases` und `no_fill` — zu den acht, die die Carry-Familien schon melden; pro
Kandidat und Szenario meldet jeder Fold zusätzlich `uncharged_final_exit_cost`, weil das
Slot-Buch ohne aufgewärmte Kohorten am Fold-Ende einen größeren ungeladenen Ausstieg
zurücklässt als v2.

```powershell
uv run trading-research carry-declare-measured --receipt artifacts/cost/binance-carry-v1-receipt.json --base-config configs/funding-carry-panel-v4.json --output configs/funding-carry-panel-v4-measured.json
uv run trading-research carry-verify-measured --receipt artifacts/cost/binance-carry-v1-receipt.json --spec configs/funding-carry-panel-v4-measured.json --base-config configs/funding-carry-panel-v4.json
uv run trading-research panel-manifest --capture <perp> --hedge-capture <spot> --output artifacts/carry/carry-v4-walk-forward-usdt-pairs-1d-w1-v1.json --family-spec configs/funding-carry-panel-v4-measured.json
uv run trading-research carry-fold --capture <perp> --hedge-capture <spot> --manifest <manifest> --family-spec configs/funding-carry-panel-v4-measured.json --output artifacts/carry/funding-carry-v4-fold0-v1.json --registry artifacts/carry/metadata-funding-carry-v4.sqlite3 --fold-index 0
uv run trading-research panel-decision --fold-report artifacts/carry/funding-carry-v4-fold0-v1.json --family-spec configs/funding-carry-panel-v4-measured.json --output artifacts/carry/funding-carry-v4-decision-v1.json --registry artifacts/carry/metadata-funding-carry-v4.sqlite3
```

Die Kette läuft neunmal (`--fold-index 0` bis `8`) und mündet in eine `panel-decision`
über alle Folds; Manifest, Folds, Entscheidung und Registry heißen
`artifacts/carry/carry-v4-walk-forward-usdt-pairs-1d-w1-v1.json`,
`artifacts/carry/funding-carry-v4-fold<i>-v1.json`,
`artifacts/carry/funding-carry-v4-decision-v1.json` und
`artifacts/carry/metadata-funding-carry-v4.sqlite3`. Kein Fold ist gelaufen; Ergebnis
folgt als `P1_33_DECISION_<date>.md`.

### Holdout-Read (P1.34)

Der eine Befehl, der das finale Holdout einer **Carry**-Familie liest —
Protokoll Abschnitt 2.3: einmal, für einen benannten Kandidaten, bevor eine Bestätigung
irgendetwas autorisiert. Bis jetzt gab es kein Kommando, das ein Holdout lesen konnte,
kein Kriterium, das eine Bestätigung war, und nichts, was einen zweiten Lesevorgang
verhindert hätte; Protokoll-Abschnitt 16.2 (2026-09-17) und
`docs/superpowers/specs/2026-09-17-holdout-read-design.md` legen das jetzt fest, bevor
irgendein Holdout geöffnet wird. Abschnitt 16.2 bindet jede Familie mit ungeöffnetem
Holdout, ausführbar ist bisher aber nur der Carry-Weg: einen Panel-Holdout-Pfad gibt es
noch nicht, er wäre ein eigenes Kommando auf derselben Regel.

Herkunftsregel: gelesen wird mit den `final_holdout_ids` und dem Kalender des
**ursprünglichen** Walk-forward-Manifests, nie mit neu abgeleiteten — eine erweiterte
Aufnahme würde sie verschieben. Die Bars kommen aus **erweiterten** Captures (Perpetual
und Spot), die als geprüfte Obermengen der ursprünglichen Captures vorliegen müssen: jede
`sources`-Zeile der Original-Capture-Manifeste mit identischem `raw_sha256` und Status.
Vier Links binden das Manifest zuerst an die Original-Captures — `capture_root_hash`,
`dataset_root_hash`, `hedge_capture_root_hash`, `hedge_dataset_root_hash` —, sonst könnte
eine beliebige, bloß in der erweiterten Capture enthaltene Aufnahme die Obermengenprüfung
bestehen, ohne die Aufnahme zu sein, auf der die Familie tatsächlich entschieden wurde.
Verifiziert werden alle vier Captures, die erweiterten wie die originalen: die
Original-Manifeste sind die Autorität für genau diese Links, für die Obermengenzeilen und
für die zwei `original_*`-Hashes, die das Artefakt bindet.

Abdeckung ist eine eigene Abweisung, und sie hat zwei Stufen. Erstens der Kalender: die
erweiterten Captures müssen den Exit-Monat der letzten Holdout-Episode **in jeder
Datenart** erreichen, die sie führen — der Monat wird je Art genommen (Vereinigung über
Symbole, nie über Arten), denn Klines und Funding-Settlements werden getrennt
veröffentlicht. Für P1.33 heißt das: die September-Klines **beider** Märkte und die
September-Funding-Dumps (die es nur im Perpetual-Markt gibt). Eine Aufnahme mit
September-Klines, aber August-Funding würde den letzten Exit bepreisen und die
Settlements der letzten Woche stillschweigend verschlucken. Zweitens der Balken selbst:
nach dem Laden muss zum Schlusszeitpunkt des letzten Exits in beiden Märkten tatsächlich
ein Bar vorliegen, sonst würde jede letzte Episode zwangsgeschlossen statt zu ihrem
eigenen Preis auszusteigen — genau das, wofür die erweiterte Aufnahme existiert.

Der Kandidat wird nicht gewählt, sondern abgeleitet: das entscheidungsfähige Mitglied mit
dem höchsten Adverse-Gesamtertrag nach Abzug seines gepoolten adversen
`uncharged_final_exit_cost` (0 bei Kohorten-Familien), bei Gleichstand nach
Deklarationsreihenfolge. Gelesen werden nur der Kandidat und die zwei
Dominanz-Kontrollen `no_trade` und `random_pairs`; die übrigen Mitglieder und die
Kontext-Kontrolle bleiben ungelesen, damit eine spätere Generation von ihrem Holdout nicht
informiert ist.

Bestätigt wird auf den Holdout-Episoden des Kandidaten, wenn zutrifft: Basis-Gesamtertrag
> 0; Adverse-Gesamtertrag nach dem Abzug ≥ 0; Dominanz über `no_trade` und `random_pairs`
in beiden Szenarien; größter Episoden- und größter Paaranteil ≤ 0.5; höchstens 4 der 26
Entscheidungen als `UNIVERSE_TOO_SMALL` übersprungen. Kein statistischer Test — 26 Wochen
bestätigen ein Vorzeichen und eine Größenordnung, sie entdecken nichts —; gemeldet, aber
nie gegatet, werden der mittlere wöchentliche Nettoertrag unter beiden Szenarien, der
Anteil positiver Wochen, die gepoolte Zahl der Zwangsschließungen des Kandidaten und ob
der Holdout-Basis-Mittelwert **auf oder über** der Bootstrap-Untergrenze liegt, die die
Entscheidung für genau diesen Kandidaten ausweist — einseitig, kein Intervall: ein
Holdout, das besser ausfällt als die Folds, ist kein Befund gegen den Kandidaten.
Verdikt `holdout_confirmed` oder `holdout_failed`;
der Befehl endet in beiden Fällen mit Exit 0, denn ein gescheitertes Holdout ist ein
Ergebnis, das der Report festhält, kein Kommandofehler.

Einmalgebrauch: die Ausgabedatei ist unveränderlich, und die Registry hält ein Artefakt
vom Typ `holdout`, dessen Id allein aus der **Deklaration** abgeleitet ist
(`uuid5(NAMESPACE_URL, "holdout:<Familienname>:<family_spec_hash>")`) — nichts vom
Walk-forward-Manifest steckt darin, damit eine Reparatur, die dieselbe Familie auf einem
neuen Manifest neu veröffentlicht, ihr keinen zweiten Lesevorgang verschafft. Ein zweiter
Lesevorgang derselben Deklaration scheitert an der schon vorhandenen Datei oder am schon
vorhandenen Registry-Eintrag. Scheitert nur die Registrierung, nachdem der Report
geschrieben wurde, löscht der Befehl den gerade geschriebenen Report wieder, damit ein
Wiederholungsversuch sauber neu starten kann, statt die Familie in einem Zustand
zurückzulassen, der weder als gelesen noch als ungelesen behandelbar wäre.

```powershell
uv run trading-research carry-holdout --capture <perp-erweitert> --hedge-capture <spot-erweitert> --original-capture <perp-original> --original-hedge-capture <spot-original> --manifest artifacts/carry/carry-v4-walk-forward-usdt-pairs-1d-w1-v1.json --family-spec configs/funding-carry-panel-v4-measured.json --decision artifacts/carry/funding-carry-v4-decision-v1.json --fold-report artifacts/carry/funding-carry-v4-fold0-v1.json --fold-report artifacts/carry/funding-carry-v4-fold1-v1.json --fold-report artifacts/carry/funding-carry-v4-fold2-v1.json --fold-report artifacts/carry/funding-carry-v4-fold3-v1.json --fold-report artifacts/carry/funding-carry-v4-fold4-v1.json --fold-report artifacts/carry/funding-carry-v4-fold5-v1.json --fold-report artifacts/carry/funding-carry-v4-fold6-v1.json --fold-report artifacts/carry/funding-carry-v4-fold7-v1.json --fold-report artifacts/carry/funding-carry-v4-fold8-v1.json --output artifacts/carry/funding-carry-v4-holdout-v1.json --registry artifacts/carry/metadata-funding-carry-v4.sqlite3
```

Ob ein Lesevorgang durchginge, wird mit `--check-only` geklärt, nicht mit dem Lesevorgang
selbst: derselbe Befehl mit diesem Schalter führt jede Eingabeprüfung aus — beide
Capture-Paare, Herkunft, Links, Siegel, die Kandidatenableitung und beide
Abdeckungsprüfungen einschließlich des Balkens am letzten Exit —, gibt eine Zeile mit dem
abgeleiteten Kandidaten und `ready` aus und endet mit 0. Er wertet nichts aus, schreibt
nichts und registriert nichts; die eine Lesung der Familie bleibt unverbraucht. Jede
Abweisung bleibt eine Abweisung. Ein Holdout-Read ist nicht wiederholbar — deshalb wird
die Bereitschaft so festgestellt und nicht dadurch, dass man ihn probiert.

```powershell
uv run trading-research carry-holdout --check-only ...  # gleiche Argumente wie oben
```

Es ist noch kein Holdout gelesen worden; das Öffnen bleibt eine eigene, ausdrückliche
Entscheidung des Nutzers.

### Shadow-Buch und Messstrom (P1.24)

Zwei Dinge, die von hier an wöchentlich bzw. dauerhaft laufen, beide ohne jede
Handelsanbindung: wöchentliche **Shadow-Captures** mit dem daraus gerechneten
**Shadow-Buch** und ein **permanenter Messstrom** neben dem abgeschlossenen Kostenjournal
v1. Entworfen in
`docs/superpowers/specs/2026-09-22-shadow-book-and-measurement-stream-design.md`.

Die Strategie liest Binances Monats-Dumps, die erst Wochen später erscheinen — ein
Wochenbuch kann darauf allein nicht laufen. Eine Shadow-Capture ist deshalb eine
gewöhnliche Panel-Capture, deren `sources` die Vereinigung aus einer geprüften
Basis-Capture, dem noch nicht abgedeckten Rest der Vorwoche und dem neuen Schwanz sind: je
ein Tages-Kline-Dump pro Symbol und Tag und, im Perpetual-Markt, ein
Funding-History-Fenster über REST. Format und Siegel sind die des Panels, jede
Wochen-Capture ist also eine geprüfte Obermenge ihrer Basis. Deckt ein Monats-Dump später
einen Monat ab, für den der Schwanz schon Zeilen führt, gewinnt der Monats-Dump — aber erst,
nachdem jede dieser Zeilen gegen die Zeile desselben Schlüssels in der Basis verglichen
wurde; eine einzige Abweichung weist die Capture ab (`RECONCILIATION_MISMATCH`).

Die Montagskette (06:00 UTC), je Markt einmal von der zuletzt reparierten Basis, mit der
Capture der Vorwoche als `--previous-capture` (die erste Woche einer Kette läuft ohne):

```powershell
uv run trading-research shadow-capture --base-capture <perp-basis> --output data/shadow/perp-<S> --tail-through <S> --market um --previous-capture data/shadow/perp-<S-1>
uv run trading-research shadow-capture --base-capture <spot-basis> --output data/shadow/spot-<S> --tail-through <S> --market spot --previous-capture data/shadow/spot-<S-1>
uv run trading-research shadow-week --declaration configs/shadow-carry-v4.json --capture data/shadow/perp-<S> --hedge-capture data/shadow/spot-<S> --perp-base-capture <perp-basis> --spot-base-capture <spot-basis> --decision-sunday <S> --measurement-snapshot artifacts/cost/binance-measurement-v2-<S>.json
```

Bricht eine Capture ab, bevor ihr Manifest geschrieben ist, räumt sie das selbst angelegte
Ausgabeverzeichnis wieder weg — Exit 1 heißt hier „Transport weg, gleich noch einmal“ —, und
steht doch eines da, weil ein Kill vor dem Aufräumen kam, nennt die Abweisung den Weg:
Verzeichnis löschen und neu starten.

Nach jeder Capture meldet `shadow-capture` `stale_symbols: klines <n>, fundingRate <n>`, und
das Wochenartefakt trägt dieselbe Erklärung weiter — ganz unter `stale_symbols` und noch
einmal neben jedem gerankten Paar, dessen Bein betroffen ist. Ein Symbol steht dort, wenn die
Basis für es keinen Monats-Dump jenseits des genannten Monats hat (`null`: gar keinen). Sein
Schwanz beginnt dann nicht am Tag nach seinem eigenen letzten Monat, sondern am globalen
Stichtag, und die Tage dazwischen stehen in keiner Capture. Das ist kein Fehler des Laufs,
sondern eine Lücke in der Basis: bei Delistings bleibt sie, sonst heilt sie eine frischere
oder reparierte Basis. Eine steigende Zahl heißt, dass die Basis zurückfällt.

`--perp-base-capture` und `--spot-base-capture` sind genau dann nötig, wenn die
Wochen-Capture daneben eine Basis nennt (Ruling 17) — die Herkunftsprüfung braucht deren
Manifest, das nur der Aufrufer hat. Das Buch des Sonntags `S` ist eine reine Funktion aus
Deklaration, Wochen-Capture und dem festen Anker-Sonntag **2026-09-13**:
`evaluate_carry_decisions` — dieselbe Schleife wie im Fold-Runner, unverändert — läuft über
jeden Sonntag vom Anker bis `S`, und der letzte Entscheid ist das Buch. Es gibt keinen
gespeicherten Slot-Zustand und nichts, was kaputtgehen könnte: jede Woche ist aus Daten neu
berechenbar. Wohin Artefakt und Registry gehen, sagt die Deklaration, nicht die
Kommandozeile. Exit 2 heißt: so nicht berechenbar (Deklaration, Siegel, Herkunft,
fehlender Sonntagsbalken, Woche schon veröffentlicht), Exit 1 heißt: später noch einmal.

Phase A und Phase B (Spec Abschnitt 2): In **Phase A** trägt jedes Wochenartefakt
`status: development_only` und **keine P&L** — weder den Block noch die laufenden Summen,
damit keine nach dem Holdout entstandene Zahl in die Go/No-go-Entscheidung über den
Holdout-Read gerät. **Phase B** beginnt erst, wenn `P1_33_HOLDOUT_<Datum>.md` `holdout_confirmed`
festhält und der `report_hash` dieses Reports in `configs/shadow-carry-v4.json` steht; erst
dann wird `--holdout-report` übergeben, und das Wochenkommando siegelt den Report neu,
bevor es eine Zahl daraus liest. Ein Holdout-Report an eine Phase-A-Woche wird abgewiesen,
nicht ignoriert.

Der Messstrom ist ein zweites Journal neben dem Kostenjournal v1, mit eigener Spec, eigener
Run-Id und hash-verketteten Segmenten. Alle 61 s eine Runde: der Tiefenlauf der Kostenpaare
aus v1 bei 500 / 5 000 / 50 000 USDT in jeder Runde, dazu in jeder fünften die drei
All-Symbol-Endpunkte — Funding-Prämie und Basis aller Perpetuals, bestes Bid/Ask aller
Perpetuals und aller Spot-Paare. Keine Zielrundenzahl: der Strom läuft, bis er gestoppt
wird.

```powershell
uv run trading-research binance-measurement-journal-create --journal data/measurement-journals/binance-measurement-v2 --run-id binance-measurement-v2 --cost-journal data/cost-journals/binance-carry-v1
uv run trading-research binance-measurement-journal-run --journal data/measurement-journals/binance-measurement-v2
uv run trading-research binance-measurement-journal-status --journal data/measurement-journals/binance-measurement-v2 --last 60
uv run trading-research binance-measurement-journal-verify --journal data/measurement-journals/binance-measurement-v2
uv run trading-research binance-measurement-journal-snapshot --journal data/measurement-journals/binance-measurement-v2 --output artifacts/cost/binance-measurement-v2-<S>.json --window-start <S-6>T00:00:00Z --window-end <S>T23:59:59Z
```

`create` übernimmt Stichprobe, Notionals, Tiefenlimit, Takt und erklärte Gebühren
unverändert aus dem Kostenjournal v1 und hält dessen Spec-Hash fest — beide Journale messen
dieselben Beine unter einer Ableitung. `run` hält wie v1 eine exklusive Sperre, wird vom
Startup-Launcher neu gestartet (Exit 1) und prüft beim Resume nur den Schwanz; Exit 2
heißt gestoppt. Ohne `--rounds` läuft es, bis der Prozess beendet wird. `status` ist nur
lesend und liefert das Lebenszeichen `newest_age_seconds` neben Fehlerrate und
ausgeschlossenen Symbolen je Endpunkt. `verify` prüft die **ganze** Kette ab `ZERO_HASH` —
der tägliche Liveness-Check ruft ihn auf, weil ein Neustart nur den Schwanz prüft — und
endet mit 0 oder 1. `snapshot` versiegelt ein beidseitig geschlossenes Fenster
(`--window-start`/`--window-end` als ISO-8601-UTC-Stempel `YYYY-MM-DDTHH:MM:SSZ`, alles
andere wird abgewiesen) zu einer unveränderlichen Quittung, die das Wochenartefakt per Hash
zitiert. Ein Snapshot ändert **keine** erklärte Kostentabelle: die Tiers der Familie
bleiben die der v1-Quittung, der Messstrom ist der Kostenmonitor der Papierphase daneben.

Nirgends in diesem Pfad existiert eine Order, ein API-Key oder ein Ausführungsadapter.

## Harte Grenzen

- `tiny_live` wird von der Runtime-Konfiguration abgewiesen.
- Historische Ergebnisse autorisieren höchstens eine spätere Shadow-Phase.
- `INSUFFICIENT_EVIDENCE` ist weder bestanden noch durchgefallen.
- CTM, ML, RL, Tweets und Multi-Agent-Evidence werden erst als Challenger
  zugelassen, nachdem einfache Baselines auf echten, eingefrorenen OOS-Daten
  bestehen.
- Acht Wochen und mindestens 200 unabhängige Paper-Episoden lassen sich nicht
  durch einen Backtest ersetzen.

Die eingefrorenen Anforderungen stehen in den `PHASE_0_*.md`-Dokumenten; der
weitere Forschungsbacklog steht in `PHASE_1_BACKLOG.md`.
