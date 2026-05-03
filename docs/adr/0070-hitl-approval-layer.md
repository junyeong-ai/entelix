# ADR 0070 — HITL approval `tower::Layer` wiring + `AgentEvent` variants

**Status**: Accepted
**Date**: 2026-05-02
**Decision**: `Approver::decide` is invoked from a `tower::Layer<S>` over `Service<ToolInvocation>`. `ApprovalLayer` is auto-attached by `ReActAgentBuilder::build` whenever an `Approver` is configured. Approve / reject decisions emit `AgentEvent::ToolCallApproved` / `ToolCallDenied` through a type-erased `ToolApprovalEventSink` adapter that the agent runtime stamps into `ExecutionContext` extensions, so the layer reaches the typed `AgentEventSink<S>` without taking the generic `S` as a constructor arg.

## Context

Earlier slices added `ToolDispatchScope` (ambient task-locals around tool dispatch) and `RunOverrides` (per-call ChatModel + graph knobs). The audit flagged a third HITL gap — `AgentEvent::ToolCallApproved/Denied` variants. While auditing the existing approver wiring, a deeper finding surfaced:

- `Approver::decide` trait method exists.
- `AgentBuilder::with_approver`, `ReActAgentBuilder::with_approver`, `Subagent::with_approver` all wire it onto the agent.
- `ExecutionMode::Supervised` + a build-time validation rejects supervised mode without an approver.
- `Agent::approver()` accessor exists.
- **But no production code path ever calls `approver.decide(...)`.** `grep -rn '\.decide(' crates/ | grep -v approver.rs | grep -v test` returns nothing.

The HITL approval mechanism was documented infrastructure with no actual gating logic. Adding `ToolCallApproved/Denied` variants without a firing path would have produced dead enum variants — a violation of the "흔적 무" mandate.

This ADR closes both gaps in one slice: wires the missing dispatch + adds the missing event variants.

## Decision

### `ApprovalLayer<S>` Tower middleware

Approval gating ships as a `tower::Layer<S>` over `Service<ToolInvocation>`, mirroring the design slice 117 used for `ScopedToolLayer`:

```rust
pub struct ApprovalLayer { approver: Arc<dyn Approver> }

impl<S> Layer<S> for ApprovalLayer {
    type Service = ApprovalService<S>;
    fn layer(&self, inner: S) -> Self::Service { ... }
}
```

`ApprovalService::call` runs `approver.decide(&request, &ctx)` before passing the invocation through. On `Approve`, the inner service runs. On `Reject{reason}`, the inner service is skipped and `Error::InvalidRequest` is returned with the reason in the message.

### Type-erasure of the agent's typed sink

`AgentEventSink<S>` is generic over the agent's state type. `ApprovalLayer` lives below the agent (one layer per registry, but agents may share registries with heterogeneous `S`). Taking the sink as a constructor arg would tie the layer to one `S` and prevent registry sharing.

Solution: a type-erased `ToolApprovalEventSink` trait + `ToolApprovalEventSinkHandle` that operators (and the agent runtime) wrap an `AgentEventSink<S>` into:

```rust
#[async_trait]
pub trait ToolApprovalEventSink: Send + Sync + 'static {
    async fn record_approved(&self, run_id: &str, tool_use_id: &str, tool: &str);
    async fn record_denied(&self, run_id: &str, tool_use_id: &str, tool: &str, reason: &str);
}

pub struct ToolApprovalEventSinkHandle { sink: Arc<dyn ToolApprovalEventSink> }

impl ToolApprovalEventSinkHandle {
    pub fn for_agent_sink<S>(sink: Arc<dyn AgentEventSink<S>>) -> Self
    where S: Clone + Send + Sync + 'static { ... }
}
```

The handle rides as an `ExecutionContext` extension. The layer reads it via `ctx.extension::<ToolApprovalEventSinkHandle>()` and emits when present. Operators with custom direct-observability (OTel, audit-log) implement `ToolApprovalEventSink` directly and wrap with `ToolApprovalEventSinkHandle::new(...)`.

### Auto-attachment in `Agent::execute` and `ReActAgentBuilder::build`

Two attachment points:

1. **Layer attachment** (registry → tools): `ReActAgentBuilder::build` wraps the supplied `tools` registry with `ApprovalLayer::new(approver)` whenever an approver is configured. Operators get HITL by calling `with_approver` — no extra wiring step.

2. **Sink-handle attachment** (agent → request ctx): `Agent::execute` unconditionally adds `ToolApprovalEventSinkHandle::for_agent_sink(self.sink.clone())` to the request context's extensions before dispatching the inner runnable. The cost is one `Arc` clone + one Extensions slot insert per execute. Agents without an `ApprovalLayer` pay this cost but never observe the handle.

### Why `AwaitExternal` is rejected (not yet supported)

The third `ApprovalDecision` variant is `AwaitExternal` — out-of-band review (web UI / Slack). Properly wiring it requires graph-level `interrupt` integration: the layer would need to raise an `Error::Interrupted { payload }` and the resume path would need to thread the operator's eventual decision back through. That's a separate ADR-sized change.

Today the layer surfaces `AwaitExternal` as a typed rejection with an explicit message ("out-of-band review is not yet wired to graph interrupt"). The dispatch fails fast rather than blocking indefinitely. A follow-up slice replaces this with the real interrupt integration.

### Why two new `AgentEvent` variants (not one)

`ToolCallApproved` and `ToolCallDenied` carry asymmetric payloads — denial requires a `reason`, approval doesn't. A single `ToolCallDecision { decision: ApprovalDecision, ... }` variant would force consumers to match on `decision` to distinguish — same code, more awkward at the call site. Two variants give the dashboard / metrics consumer a clean filter (`if let AgentEvent::ToolCallDenied { .. }`).

Both variants live on `AgentEvent<S>` (alongside `ToolStart` / `ToolComplete` / `ToolError`) with `to_graph_event(...)` returning `None` — approval markers are runtime-only, not audit-projecting (the audit log already records the actual `ToolCall` / `ToolResult` pair, or skips it on denial).

## Consequences

- New public types in `entelix-agents`: `ApprovalLayer`, `ApprovalService<S>`, `ToolApprovalEventSink` trait, `ToolApprovalEventSinkHandle`. All re-exported through the facade.
- New `AgentEvent` variants `ToolCallApproved` + `ToolCallDenied`. Both `to_graph_event(...)` → `None`.
- `Agent::execute` unconditionally attaches `ToolApprovalEventSinkHandle` to the request ctx. Cost: one Arc clone + one Extensions entry per execute.
- `ReActAgentBuilder::build` auto-wraps the registry with `ApprovalLayer` when an approver is configured. Operators get HITL "for free" with `with_approver(approver)`.
- Sub-agent narrowed registries (`restricted_to`) inherit the layer factory by Arc (ADR-0035) — `ApprovalLayer` fires for sub-agent dispatches automatically.
- Four regression tests in `approval_layer::tests`: approve dispatches inner, reject short-circuits with typed error, sink records both decisions, layer runs without sink attached.
- `Subagent::with_approver` is **not** auto-wired into the inner registry today (Subagent's approver storage exists but the layer attachment lives on `ReActAgentBuilder`). Subagent HITL ships as a follow-up alignment slice if demand surfaces.

## References

- ADR-0011 — `Tool` trait does not extend `Runnable`; `ApprovalLayer` lives at the dispatch boundary.
- ADR-0035 — sub-agent layer-factory inheritance.
- ADR-0068 — `ToolDispatchScope`; `ApprovalLayer` mirrors that design (Tower middleware + ctx extension for per-request data).
- `crates/entelix-agents/src/agent/approval_layer.rs` — implementation + tests.
- `crates/entelix-agents/src/agent/event.rs` — `ToolCallApproved` / `ToolCallDenied` variants.
- `crates/entelix-agents/src/agent/mod.rs::Agent::execute` — ctx-extension attachment.
- `crates/entelix-agents/src/react_agent.rs::ReActAgentBuilder::build` — layer auto-attachment.
