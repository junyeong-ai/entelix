# ADR 0082 — `RunBudget` downstream surface integrity

**Status**: Accepted
**Date**: 2026-05-05
**Decision**: The frozen `UsageSnapshot` captured by `Agent::run_inner` (ADR-0081) rides through the `entelix.agent.run` span as five typed attributes — `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.usage.total_tokens`, `entelix.usage.requests`, `entelix.usage.tool_calls`. Fields are declared as `tracing::field::Empty` placeholders on span open and populated via `Span::record` at the snapshot freeze point — runs without a budget keep the fields empty so the `tracing-opentelemetry` bridge omits them from the exported span. `DefaultRetryClassifier::should_retry` adds an explicit `Error::UsageLimitExceeded { .. } => RetryDecision::STOP` arm even though the wildcard would catch it; the explicit match defends against future variants accidentally landing budget breach on the retry path.

## Context

ADR-0080 introduced `RunBudget` (5-axis usage cap shared by all dispatches in one run via `Arc<RunBudgetState>`) and ADR-0081 added the `AgentRunResult<S>` envelope that carries the frozen `UsageSnapshot` at terminal. A B-4 / B-5 surface audit (`/loop continue B-4 RunBudget work`) flagged five integrity gaps:

1. **OTel emit not wired** — the `entelix.agent.run` root span carried only `gen_ai.agent.name`, `entelix.run_id`, `entelix.tenant_id`, `entelix.thread_id`. Per-run usage was reachable through `AgentRunResult.usage` but invisible to OTel-driven dashboards (Grafana, Honeycomb, Datadog) that filter / aggregate on span attributes.
2. **Sub-agent rollup verified** — the `Arc<RunBudgetState>` shared through `Extensions::clone` is correct end-to-end (`crates/entelix-agents/src/subagent.rs:371` clones the parent ctx; `Extensions: Arc<HashMap<TypeId, Arc<dyn Any>>>` rides through unchanged). No code change needed.
3. **`Error::UsageLimitExceeded` retry-classifier** behaviour was implicit — the wildcard `_ => STOP` arm caught budget breach correctly, but no explicit arm declared the intent.
4. **`AuditSink::record_usage_limit_exceeded`** absent — invariant 18 (managed-agent lifecycle auditable) records sub-agent invocations, supervisor handoffs, resumes, and memory recall but has no record verb for budget breach. Compliance / billing audits that need a permanent log of which run hit which limit would have to scrape `tracing` events instead.
5. **`UsageSnapshot::total_tokens()` helper unused** in-repo — it is a published affordance for callers, not a bug.

This ADR addresses (1) and (3). Hole (4) — extending `AuditSink` and `GraphEvent` with a `UsageLimitExceeded` audit verb — is a larger surface change with downstream `entelix-session` projection impact and ships under a separate ADR.

## Decision

### Five typed span attributes, declared `Empty`, populated via `Span::record`

The natural OTel GenAI semconv field names are `gen_ai.usage.input_tokens` and `gen_ai.usage.output_tokens` (already published in the semconv registry that `entelix-otel::semconv` tracks). `total_tokens` is included as a derived convenience so dashboards do not have to compute the sum at query time. The `entelix.usage.*` namespace covers the two request-level counters (`requests`, `tool_calls`) that have no GenAI semconv equivalent — those axes are entelix-specific.

```rust
fn run_span(&self, run_id: &str, ctx: &ExecutionContext) -> tracing::Span {
    tracing::info_span!(
        target: "gen_ai",
        "entelix.agent.run",
        gen_ai.agent.name = %self.name,
        entelix.run_id = %run_id,
        entelix.tenant_id = %ctx.tenant_id(),
        entelix.thread_id = ctx.thread_id(),
        gen_ai.usage.input_tokens = tracing::field::Empty,
        gen_ai.usage.output_tokens = tracing::field::Empty,
        gen_ai.usage.total_tokens = tracing::field::Empty,
        entelix.usage.requests = tracing::field::Empty,
        entelix.usage.tool_calls = tracing::field::Empty,
    )
}
```

Population happens at the `AgentRunResult::usage` freeze point in `run_inner` — between `runnable.invoke` returning and `on_complete` observers firing (the frozen-pre-observer attribution rule from ADR-0081):

```rust
let usage = ctx.run_budget().map(|budget| budget.snapshot());
if let Some(snapshot) = usage {
    let span = tracing::Span::current();
    span.record("gen_ai.usage.input_tokens", snapshot.input_tokens);
    span.record("gen_ai.usage.output_tokens", snapshot.output_tokens);
    span.record("gen_ai.usage.total_tokens", snapshot.total_tokens());
    span.record("entelix.usage.requests", snapshot.requests);
    span.record("entelix.usage.tool_calls", snapshot.tool_calls);
}
```

`Span::current()` inside the `.instrument(self.run_span(...))` future is the agent-run root span.

### Why per-axis fields, not a single JSON blob

A single `entelix.usage = "{...}"` string field would carry the same data with one `record` call. Rejected because:

- OTel backends (Tempo / Honeycomb / Datadog) filter and aggregate on typed numeric attributes — `WHERE gen_ai.usage.input_tokens > 1000` is a one-line query against typed fields, but a JSON payload requires the backend's custom JSON-extraction syntax (different per vendor).
- The OTel GenAI semconv defines `gen_ai.usage.input_tokens` and `gen_ai.usage.output_tokens` as numeric attributes by name. A blob would break semconv compatibility.
- Span attribute count is not a constraining resource at this scale — five fields land within every export-format envelope without bloat.

### Why declare `Empty` rather than conditional `info_span!`

Two alternatives:

- **Conditional span construction** — branch on `ctx.run_budget()` at span open, add the fields when present. Forces two divergent `info_span!` macro invocations whose only difference is the field list; spreads the attribute-name source of truth across two sites.
- **Always emit, default to zero** — open the span with `gen_ai.usage.input_tokens = 0_u64`. Misleading: a run without a budget is not "consumed zero tokens", it is "not budgeted". Dashboards that filter on `> 0` would silently drop budgeted-zero runs alongside non-budgeted runs.

The `Empty` placeholder + conditional `record` pattern keeps the attribute-name list in one place (the `info_span!` macro) and lets the bridge omit absent fields from the export. This is the same `Empty` + `record` pattern `tracing` documents for late-bound attributes elsewhere.

### Explicit `UsageLimitExceeded` arm in `DefaultRetryClassifier`

```rust
Error::UsageLimitExceeded { .. } => RetryDecision::STOP,
```

Behavioural equivalence with the prior wildcard: today, both arms produce the identical `RetryDecision::STOP`. The explicit arm pays for itself in two scenarios:

1. **Future variant insertion** — a new `Error::SomethingTransient` variant added between `Provider` and the wildcard could accidentally land on a retry path if a maintainer extends the retry-set without re-auditing budget breach. The explicit arm pins budget breach at STOP regardless.
2. **Reader comprehension** — a production caller reading the classifier sees `UsageLimitExceeded` mentioned by name and immediately understands the contract. The wildcard alone left the reader to derive the answer.

This is the same defensive-explicit-match pattern invariant 15 enforces for codec match arms (no wildcard absorbing unknown vendor signals). Retry classification is not a wire-protocol decode, but the same principle applies — match every variant explicitly when the fall-through has a different default than the typed answer would specify.

The retry-class explicit arm also includes a doc comment noting that re-issuing after budget breach is doubly wrong: not only does the call produce the same `UsageLimitExceeded`, the pre-call CAS in `RunBudget::check_pre_request` increments the counter before the cap check, so each retry attempt burns one more counter slot. (The CAS was deliberate ADR-0080 design — race-free under tokio work-stealing — and surfaces here as a small consequence operators should know about.)

## Consequences

- **`entelix-otel` is not modified.** The OTel emit is on the agent-run span itself, which `tracing-opentelemetry` exports as part of its standard span-bridging — no `OtelLayer` change required. Operators that wire `entelix-otel` see the new attributes automatically.
- **Public-API baseline unchanged.** Span attributes are not a `cargo-public-api`-tracked surface. The retry classifier's explicit arm is internal-fn-only.
- **`tracing-opentelemetry` bridge requirement** — the bridge must be at a version that respects `tracing::field::Empty` as omit-from-export (every published version does; documented for posterity).
- **Breaking change in the audit area is deferred.** `AuditSink::record_usage_limit_exceeded` (audit-trail emit on budget breach) lands in a separate ADR alongside the `GraphEvent::UsageLimitExceeded` projection — it is a larger surface change that affects `entelix-session` and the `SessionAuditSink` adapter.
- **Regression test** — `crates/entelix-agents/tests/agent_run_budget_span.rs` installs a `tracing::Layer` that captures `Span::record` calls and asserts the five fields populate when a budget is attached and stay omitted when one is not. The test pins both the field-name set and the `Empty`-omission contract.
- **Test for retry classifier** — `default_classifier_does_not_retry_usage_limit_exceeded` in `crates/entelix-core/src/transports/retry.rs` pins the explicit arm so a future refactor that drops it gets caught at CI.

## References

- ADR-0080 — `RunBudget` 5-axis (the snapshot's source).
- ADR-0081 — `AgentRunResult<S>` envelope (the freeze point this attribute mirror co-locates with).
- OTel GenAI semconv — `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens` field names.
- Invariant 15 — explicit-match-over-wildcard discipline (analogous reasoning here, not a direct enforcement).
- Audit-channel extension — deferred to a follow-up ADR.
