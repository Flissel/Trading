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
```

Exit 2 heißt: so nicht erklärbar (Quittung ohne Median, falsche Basis-Erklärung,
Ausgabe existiert bereits). Die Ausgabe ist unveränderlich wie die Quittung.

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
