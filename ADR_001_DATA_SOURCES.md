# ADR 001: Initial Data Sources and Rights Boundary

Status: Accepted for Phase 1
Date: 2026-08-24

## Context

The research system needs market, microstructure, news, social, and on-chain evidence. The initial account basis is EUR 100, local storage is constrained, and third-party data rights differ materially. Data that is technically accessible is not automatically authorized for model training, redistribution, or durable retention.

## Decision

### Market data

- Primary live and historical venue: OKX EEA BTC and ETH perpetuals.
- Cross-venue reference: Binance USD-M public BTC and ETH perpetual feeds.
- Official OKX historical downloads are used before any paid vendor subscription.
- Both raw forward collection and historical backfill are retained as separate provenance classes.

### Paid historical data

- Phase 1 paid-data budget: EUR 0.
- Tardis.dev is a conditional fallback, not an approved subscription.
- A purchase requires a demonstrated missing-data problem, sample validation, current quotation, storage-capacity check, and explicit user approval.

### News

- Start with official exchange, protocol, project, regulator, and company announcements.
- Retain source URL, publication and receipt timestamps, content hash, extracted event, and model provenance.
- Full article retention requires source-specific permission; otherwise store only the minimal permitted evidence and derived structured record.

### On-chain

- Start with Coin Metrics Community data, which is accessible without an API key.
- Do not claim exchange inflows or outflows from unlabeled network metrics.
- Paid address-label and exchange-flow providers are challengers only after the free baseline has measurable incremental value.
- Self-hosted full or archive nodes are deferred because current storage and operations are insufficient.

### X / Twitter

- Use only the official X API; scraping and browser automation are not permitted.
- The current X Developer materials prohibit using X data to train AI/ML models.
- Therefore, X-derived data is excluded from Phase 1 model training.
- Develop the evidence schema and agent logic with synthetic or expressly reusable fixtures.
- Any future live inference use requires a written policy basis or permission, commercial-use review, official credentials, prepaid credits, disabled auto-recharge, a hard spend cap, compliance handling, and explicit approval.
- Raw X content must not be copied into the Git repository, model artifacts, or distributable datasets.

## Trust and retention model

```text
Third-party source
  -> source-specific rights check
  -> timestamped raw ingest, only when permitted
  -> immutable content hash and source ID
  -> structured evidence extraction
  -> retention/deletion policy
  -> feature admission gate
```

Every source adapter must declare:

- allowed use: inference, training, display, redistribution;
- credential and cost class;
- raw-content retention period;
- deletion/compliance mechanism;
- event, publication, and receipt timestamps;
- license or terms version and review date.

Unknown rights fail closed: the data may be inspected manually but is not admitted to training, automated decisions, or durable datasets.

## Consequences

Advantages:

- Phase 1 remains low cost and reproducible.
- The project avoids building its predictive core on a source it cannot legally retain or train on.
- Social-agent interfaces can be developed without prematurely purchasing API credits.
- Market-data baselines remain separable from alternative-data claims.

Trade-offs:

- Phase 1 will not test real X alpha.
- Free on-chain data will not provide reliable labeled exchange flows.
- Source-rights reviews add work before new data can enter training.

## Revisit conditions

Revisit this ADR when any of the following occurs:

- X provides written permission or terms clearly authorize the intended training or automated-decision use;
- a market-data-only model passes its promotion gate and alternative data becomes the next measured hypothesis;
- paid on-chain or news data can be evaluated through a bounded trial;
- the project changes from personal research to a commercial product;
- a source changes its license, pricing, API, or compliance requirements.

## Evidence checked

- X API pricing: https://docs.x.com/x-api/getting-started/pricing
- X Developer Policy: https://docs.x.com/developer-terms/policy
- X Developer Agreement: https://docs.x.com/developer-terms/agreement
- X compliance streams: https://docs.x.com/x-api/compliance/streams/introduction
- Coin Metrics Community data: https://gitbook-docs.coinmetrics.io/packages/coin-metrics-community-data
- Coin Metrics API conventions: https://gitbook-docs.coinmetrics.io/access-our-data/api
- OKX historical market data: https://www.okx.com/en-us/historical-data
- Tardis.dev pricing: https://tardis.dev/
