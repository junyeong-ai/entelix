# entelix observability — operator reference

This directory ships the Grafana dashboard JSON used by the entelix
team in production. Import it directly into Grafana 11+ (the
schema-version is 39); panels query a Tempo / Loki tracing pipeline
fed by `tracing-opentelemetry` exporting the events `OtelLayer`
emits.

## What the dashboard surfaces

Per-tenant rollups of every cost-bearing operation, partitioned by
the OTel attributes the SDK stamps on each event:

| Panel | Source attribute | Notes |
|---|---|---|
| Model spend (per tenant, per model) | `gen_ai.usage.cost` | Stack by `gen_ai.response.model` |
| Tool spend (per tenant, per tool) | `gen_ai.tool.cost` | Excludes free tools (no row → omitted) |
| Embedding spend (per tenant, per model) | `gen_ai.embedding.cost` | Aggregated per call |
| Tokens in / tokens out | `gen_ai.usage.input_tokens` / `output_tokens` | Per-tenant lines |
| Tool error rate | `gen_ai.tool.error` count / `gen_ai.tool.end` count | Heatmap by `gen_ai.tool.name` |
| Graph recursion depth distribution | `entelix.graph.depth` | Histogram, alert when p99 > 0.8 × `recursion_limit` |
| MCP pool pressure | (operator emits via `tracing` from `McpManager`) | Gauge per `(tenant, server)` |
| Cancellation rate | `entelix.cancelled = true` count | Per-tenant lines |

## Required pipeline

One-call wire-up via `entelix-otel` (cargo feature `otlp`). The
returned handle owns the tracer provider; hold it for the lifetime
of the process — it flushes pending spans on drop.

```rust
use entelix_otel::init::{OtlpConfig, init_otlp};

// Pulls endpoint from `OTEL_EXPORTER_OTLP_ENDPOINT` and service
// name from `OTEL_SERVICE_NAME`; falls back to localhost defaults.
let _otel = init_otlp(&OtlpConfig::from_env())?;
```

The OTel collector then forwards traces to Tempo (or any compatible
backend); Grafana's Tempo data source surfaces them as spans the
dashboard queries via the `traceql` panel queries embedded in the
dashboard JSON.

## Importing

1. Open Grafana → **Dashboards** → **New** → **Import**
2. Paste `entelix-grafana-dashboard.json`
3. Set the Tempo data-source UID to your environment's value
4. Save — the panels self-populate as traffic arrives

## Tuning per environment

The dashboard's queries assume the default attribute names the SDK
emits. If your deployment renames anything (e.g. you wrap
`OtelLayer` to add a `team` label), edit the corresponding panel
queries — every query lives in the JSON's `panels[].targets[].query`
field for greppable diffs.
