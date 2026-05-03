# entelix-otel

OpenTelemetry GenAI semconv coverage. Tracing-driven design — the layer never touches the OpenTelemetry SDK directly; it emits `tracing::event!` events with `gen_ai.*` keys, and a `tracing-opentelemetry` subscriber bridges to OTLP.

## Surface

- **`semconv`** — `gen_ai.*` attribute name constants. Tracks the OpenTelemetry GenAI semconv (specific snapshot pinned in `crates/entelix-otel/src/semconv.rs` module-doc).
- **`OtelLayer`** — `tower::Layer<S>` middleware. Same struct wraps both `Service<ModelInvocation>` (model side) and `Service<ToolInvocation>` (tool dispatch). Compose via `ChatModel::layer(OtelLayer::new("anthropic"))` and `ToolRegistry::layer(...)`.
- **`GenAiMetrics`** — pre-built `opentelemetry::metrics` instrument handles (token-usage histogram, operation-duration histogram). Bucket layout matches the semconv recommendation; teams should not reinvent.
- **`init`** (cargo feature `otlp`) — convenience helpers wiring `opentelemetry-otlp` into a `tracing` subscriber. Optional — teams with their own bootstrap skip this.

## Crate-local rules

- **Cost emission is `Ok`-branch only** (invariant 12). `OtelLayer` must record `gen_ai.usage.cost` / `gen_ai.tool.cost` / `gen_ai.embedding.cost` only after the inner `Service::call` returns `Ok`. Tests assert the error branch produces no cost attribute. Mirror coverage in `entelix-policy::CostMeter` and `entelix-memory::MeteredEmbedder`.
- **Attribute names come from `semconv` constants.** No string literals like `"gen_ai.system"` inline in the layer code — always `semconv::GEN_AI_SYSTEM`. Drift in attribute names is the most common silent bug; the constant module is the single source of truth.
- **Tracing-only emit, not direct OTel SDK.** `tracing::event!(target: "gen_ai", ...)` is the seam. Adding `opentelemetry::Context::current()` calls into the layer breaks the optional-SDK contract.
- New attribute (semconv revision bump): add to `semconv::` first, then thread through emission. Bump the snapshot reference in the module doc.

## Forbidden

- Cost attributes on the error branch.
- Hard-coded attribute name strings outside the `semconv` module.
- Direct OTel SDK calls in the layer (use `tracing::event!`).

## References

- ADR-0009 — OpenTelemetry GenAI semconv adoption.
- F4 mitigation — transactional cost emission.
- OpenTelemetry GenAI semconv: <https://github.com/open-telemetry/semantic-conventions/tree/main/docs/gen-ai>
