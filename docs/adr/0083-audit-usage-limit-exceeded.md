# ADR 0083 — `AuditSink::record_usage_limit_exceeded` audit verb

**Status**: Accepted
**Date**: 2026-05-05
**Decision**: `AuditSink` gains a fifth typed `record_*` verb — `record_usage_limit_exceeded(axis: &str, limit: u64, observed: u64)`. `GraphEvent` gains a matching `UsageLimitExceeded { axis, limit, observed, timestamp }` variant. `SessionAuditSink` maps the verb onto a `SessionLog::append` of the new variant. The single emit point lives at `Agent::execute_inner`'s `Err` branch — the `Error::UsageLimitExceeded` shape already propagates from the dispatch site (`ChatModel::complete_full` / `complete_typed` / `stream_deltas` and `ToolRegistry::dispatch`), so the agent boundary is the natural funnel for the audit signal without re-emitting per dispatch path. The `axis` parameter is `&str` (lower-snake-case `requests` / `input_tokens` / `output_tokens` / `total_tokens` / `tool_calls`) rather than `UsageLimitAxis` directly so `entelix-tools` and `entelix-graph` emit sites stay free of the `entelix-core::run_budget` import — same pattern the existing four verbs use for `tier` / `agent_id` / `class` strings.

## Context

ADR-0080 introduced `RunBudget` (5-axis usage cap) and ADR-0081 added the `AgentRunResult<S>` envelope that carries the frozen `UsageSnapshot`. ADR-0082 wired the snapshot through the `entelix.agent.run` OTel span. A B-4 / B-5 surface audit (`/loop continue B-4 RunBudget work`) flagged six gaps; four of them landed (OTel emit, retry classifier explicit arm, sub-agent rollup verified working, snapshot helper coverage). Two were deferred — this ADR addresses the audit-channel one; the `CHANGELOG.md` infrastructure ships separately.

`AuditSink` (ADR-0037, invariant 18) carried four `record_*` verbs for the managed-agent lifecycle:

- `record_sub_agent_invoked` — sub-agent dispatch from a parent run
- `record_agent_handoff` — supervisor recipe handoffs
- `record_resumed` — wake from a checkpoint
- `record_memory_recall` — long-term memory retrievals

These four cover orchestration / lifecycle. `RunBudget` introduced a new operational signal — budget breach — that compliance and billing audits need a permanent log of. Today it surfaces only as an `Error::UsageLimitExceeded` propagation; OTel spans capture it as a span-error (already through the `Failed` sink event), but the durable per-tenant per-run audit record was missing. Per invariant 18:

> Sub-agent dispatch, supervisor handoff, resume-from-checkpoint, and long-term memory recall emit through the typed `entelix_core::AuditSink` channel that operators wire onto `ExecutionContext::with_audit_sink`.

The invariant lists only those four lifecycle events because they were the audit set ADR-0037 closed. Budget breach does not fit any existing verb — it is neither lifecycle (a run is ending, not transitioning) nor recall (it is a refusal, not a retrieval). A new typed verb keeps the channel's shape (typed methods over a single `emit(GraphEvent)` blob) while expanding the set.

## Decision

### Fifth verb — `record_usage_limit_exceeded`

```rust
pub trait AuditSink: Send + Sync + 'static {
    fn record_sub_agent_invoked(&self, agent_id: &str, sub_thread_id: &str);
    fn record_agent_handoff(&self, from: Option<&str>, to: &str);
    fn record_resumed(&self, from_checkpoint: &str);
    fn record_memory_recall(&self, tier: &str, namespace_key: &str, hits: usize);
    fn record_usage_limit_exceeded(&self, axis: &str, limit: u64, observed: u64);
}
```

`axis: &str` rather than `UsageLimitAxis` for the same reason `tier: &str` is on `record_memory_recall` — `entelix-tools` / `entelix-graph` emit sites depend only on `entelix-core` and the four existing `record_*` methods carry strings, not domain enums. `UsageLimitAxis::Display` is the canonical source for the rendered string (`"requests"` / `"input_tokens"` / etc.); the agent-side emit calls `axis.to_string()` so the wire format ties to one place.

`limit: u64` and `observed: u64` carry the raw counter values. Wider than the canonical axis units (`request_limit: u32`, `tool_calls_limit: u32`) so future axes that exceed `u32` (a tenant's total monthly tokens, etc.) ride the same shape — `u64` matches `UsageSnapshot::total_tokens()`'s return type.

### `GraphEvent::UsageLimitExceeded` variant

```rust
GraphEvent::UsageLimitExceeded {
    axis: String,
    limit: u64,
    observed: u64,
    timestamp: DateTime<Utc>,
}
```

`#[non_exhaustive]` on `GraphEvent` keeps the variant addition non-breaking. The `timestamp` field aligns with the rest of the variant set so `GraphEvent::timestamp()` continues to work uniformly.

### Emit point — `Agent::execute_inner` `Err` branch

```rust
Err(err) => {
    if let Error::UsageLimitExceeded { axis, limit, observed } = &err
        && let Some(handle) = ctx.audit_sink()
    {
        handle.as_sink().record_usage_limit_exceeded(
            &axis.to_string(),
            *limit,
            *observed,
        );
    }
    let _ = self.sink.send(AgentEvent::Failed { ... }).await;
    Err(err)
}
```

The dispatch site (`ChatModel::complete_full` / `complete_typed` / `stream_deltas`, `ToolRegistry::dispatch`) is where the budget actually breaches, but emit at the dispatch site would mean four sites emitting independently and the audit channel never seeing the breach when the dispatch surface is wrapped (e.g. by a retry layer that catches and rewrites). The agent boundary is the single funnel — every `RunBudget`-aware dispatch eventually returns through `Agent::execute_inner`, so the typed `Err(Error::UsageLimitExceeded)` lands here exactly once per run.

The emit is ordered *before* the `Failed` sink event so the `GraphEvent::UsageLimitExceeded` audit row precedes any operator-visible `Failed` propagation. Operators querying the audit log for budget breaches read `UsageLimitExceeded` directly without correlating through `Failed.error.contains("UsageLimitExceeded")` string-matching.

### Why not at `RunBudget::check_pre_request` / `observe_usage`

Two arguments against:

- `RunBudget` lives in `entelix-core` and has no `ExecutionContext` accessor at the breach site (the methods take `&self`, not ctx). Threading ctx through every check / observe call is a wider change than the audit signal warrants.
- The four existing verbs all emit at the agent / orchestration boundary — `record_sub_agent_invoked` fires on `Subagent::execute`, not on every dispatch the sub-agent issues. Keeping breach emit at the agent boundary matches the existing pattern.

### Why not extend `Error` variant

`AgentEvent::Failed { run_id, error: String }` already carries the `Display` rendering, but `error: String` is operator-channel free-text, not a typed audit row. Querying "all runs that breached `tool_calls`" via string-matching `Failed.error` is brittle. The typed `GraphEvent::UsageLimitExceeded { axis, limit, observed }` row carries the structured fields directly.

## Consequences

- **`AuditSink` trait surface change** — adding a method to a `pub trait` is a breaking change. Per invariant 14 (no shim) the operator-side `impl AuditSink` for any custom sink must implement the new method at the same time. `SessionAuditSink` lands its impl in this commit; downstream operator impls (Datadog forwarder, Splunk, etc.) need a one-line addition.
- **`GraphEvent` `#[non_exhaustive]`** — the variant addition is non-breaking for consumer `match` arms (they already have a fallback). `GraphEvent::timestamp()` extends the existing match arm.
- **Public-API baselines refrozen** — `entelix-core` (AuditSink trait gains a method), `entelix-session` (GraphEvent gains a variant + SessionAuditSink impl).
- **No emit on dispatch path mid-stream** — a streaming run that breaches mid-flight resolves the completion future to `Err(Error::UsageLimitExceeded)`; `Agent::run_inner` propagates that through `runnable.invoke`, and the agent-boundary emit fires on the way out. No dispatch-site emit, no double-record.
- **Ordering with `AgentEvent::Failed`** — `record_usage_limit_exceeded` runs before the `Failed` sink event. Operators wiring a single sink that listens on both channels see the audit emit first; the ordering is documented so dashboards relying on it stay correct.

## Test surface

`crates/entelix-agents/tests/agent_audit_usage_limit.rs` (new) installs a custom `AuditSink` that captures `record_usage_limit_exceeded` calls. Exercises:

- `RunBudget::unlimited().with_request_limit(1)` + a runnable that internally calls `RunBudget::check_pre_request` twice → second call returns `Error::UsageLimitExceeded { axis: Requests, limit: 1, observed: 1 }`. `Agent::execute` propagates `Err`; the captured `AuditSink` records `axis="requests", limit=1, observed=1`.
- The `Failed` sink event still fires with the operator-facing error string (existing contract).
- Run without an `AuditSink` attached: `Err` propagates, no `record_*` call (zero-overhead opt-out).

## References

- ADR-0037 — `AuditSink` typed channel + `SessionAuditSink` adapter (the four prior verbs).
- ADR-0080 — `RunBudget` 5-axis (the breach signal's source).
- ADR-0081 — `AgentRunResult<S>` envelope (the success-path artifact pair).
- ADR-0082 — `entelix.agent.run` span attributes (the OTel telemetry side).
- Invariant 18 — managed-agent lifecycle is auditable (the trait-surface contract this verb extends).
