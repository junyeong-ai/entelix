# ADR 0072 — `Command::ApproveTool` typed resume primitive

**Status**: Accepted (supersedes the surface portion of ADR-0071)
**Date**: 2026-05-03
**Decision**: The `AwaitExternal` pause-and-resume flow surfaces as a typed resume primitive `entelix_graph::Command::ApproveTool { tool_use_id, decision }` rather than an operator-attached `ApprovalDecisionOverrides` HashMap on `ExecutionContext`. `ApprovalDecision` moves to `entelix-core` so both the graph runtime (`Command`) and the agent runtime (`Approver`) reference one canonical type without violating the workspace DAG. The internal carrier (`PendingApprovalDecisions`) remains as the implementation mechanism but is no longer the operator-facing API.

## Context

ADR-0071 wired the `AwaitExternal` decision into a real pause-and-resume flow: `ApprovalLayer` raises `Error::Interrupted`, the graph dispatch loop persists a checkpoint, and resume re-enters the same dispatch with the operator's eventual decision. The resume mechanism documented in ADR-0071 used `ApprovalDecisionOverrides` — an operator-constructed `HashMap<String, ApprovalDecision>` attached to `ExecutionContext` extensions before the resume call.

The ADR-0071 §"Why not extend the `Command` enum instead" rejected the typed-Command alternative as out-of-scope:

> A `Command::ApproveTool { tool_use_id, decision }` resume primitive would auto-attach the override during `CompiledGraph::resume_with`. That's cleaner at the API boundary but requires extending the `Command` enum and threading the resume payload through the graph's resume path. The ctx-extension approach reaches feature-parity with one new type and zero changes to graph dispatch semantics.

The 8th-pass strategic-scope audit (Phase 12 close) re-evaluated this and found:

- `ApprovalDecisionOverrides` HashMap is over-design for the 90% case (single-decision resume).
- `INTERRUPT_KIND_APPROVAL_PENDING` magic-string discriminator is a cross-layer coupling shape that doesn't scale (each future layer needs its own magic string + collision check).
- The ctx-extension pattern shifts surface complexity onto the operator.

The ctx-extension *mechanism* is correct (the layer reads from ctx, not from a global). The *operator surface* should be typed, not HashMap-shaped.

## Decision

### `Command::ApproveTool` is the operator-facing resume primitive

`entelix_graph::Command<S>` grows a new variant:

```rust
pub enum Command<S> {
    Resume,
    Update(S),
    GoTo(String),
    ApproveTool {
        tool_use_id: String,
        decision: ApprovalDecision,
    },
}
```

`CompiledGraph::resume_with(Command::ApproveTool { ... }, &ctx)` constructs the internal `PendingApprovalDecisions` carrier and attaches it to ctx before re-dispatching from the checkpoint. Operators never construct the carrier themselves.

```rust
// Operator code on resume:
let final_state = compiled_graph
    .resume_with(
        Command::ApproveTool {
            tool_use_id: pending_id,
            decision: ApprovalDecision::Approve,
        },
        &ctx,
    )
    .await?;
```

### `ApprovalDecision` moves to `entelix-core`

`Command<S>` lives in `entelix-graph`; `ApprovalDecision` previously lived in `entelix-agents`. To let `Command::ApproveTool` carry an `ApprovalDecision` without `entelix-graph` depending on `entelix-agents` (which would invert the workspace DAG), `ApprovalDecision` and the related `INTERRUPT_KIND_APPROVAL_PENDING` discriminator and internal `PendingApprovalDecisions` carrier all move to `entelix-core::approval`.

`entelix-agents::approver` re-exports `ApprovalDecision` so existing import paths keep working at the source level (operators using `entelix_agents::ApprovalDecision` are unaffected; operators using `entelix::ApprovalDecision` get the same type). The `Approver` trait stays in `entelix-agents` (it's an agent-level abstraction; the decision type is core-level since it crosses the agent/graph boundary).

### `ApprovalDecisionOverrides` is removed entirely

Operators no longer need to construct decision carriers — `Command::ApproveTool` does it for them. The `HashMap<String, ApprovalDecision>` shape was justified only by "what if the operator needs to decide multiple pending tool calls in one resume" — but that's a degenerate case (operators decide pendings as they arrive; multi-decision resume is rare enough to deserve a slower API). When it arises, operators chain multiple `Command::ApproveTool` calls or attach `PendingApprovalDecisions` directly via the lower-level (doc-hidden) path.

`PendingApprovalDecisions` itself remains `pub`-visible because both `entelix-graph` (writer) and `entelix-agents` (reader) need to reference the same type for the ctx-extension contract — and operators dispatching through the raw `ToolRegistry` (no graph, no checkpointer; e.g. tests, custom embedded loops) legitimately construct it directly to attach decisions on the request `ExecutionContext`. The type's rustdoc explicitly distinguishes the two attachment paths: typed (`Command::ApproveTool`, recommended for graph-driven agents) vs direct (advanced, registry-only). `Command::ApproveTool` is the higher-level convenience.

### `INTERRUPT_KIND_APPROVAL_PENDING` becomes the typed payload discriminator

The magic-string discriminator is preserved (operators inspecting the `Error::Interrupted::payload` JSON do match on the `kind` field) but moves to `entelix-core::approval` and is re-exported as `entelix::INTERRUPT_KIND_APPROVAL_PENDING` (no facade alias needed — the constant lives at the right namespace).

The previously-shipped `entelix::APPROVAL_PENDING_INTERRUPT_KIND` alias (slice 121) is removed; the canonical name is `INTERRUPT_KIND_APPROVAL_PENDING` (the convention `INTERRUPT_KIND_*` reads better as a discriminator namespace).

### `ApprovalDecision::AwaitExternal` is rejected at resume

If an operator passes `Command::ApproveTool { decision: AwaitExternal, ... }`, `CompiledGraph::resume_with` rejects with `Error::InvalidRequest` ("AwaitExternal is not a valid resume decision — pausing again on resume defeats the purpose"). The decision variant is `Approve` or `Reject{reason}`; AwaitExternal is purely a pause signal returned by `Approver::decide`, not a resume signal.

### Idempotency requirements move to the `Approver` trait rustdoc

ADR-0071 documented operator idempotency requirements ("`Tool::execute` must be idempotent on `tool_use_id`; multi-tool-call resumes re-fire all tool calls") in an ADR section. The 8th-pass audit found this contract was unenforceable and surfaced too low — operators implementing `Approver` for the first time would not see it.

This ADR moves the requirement to the `Approver` trait's own rustdoc as a `## Idempotency` section. Discoverable from `cargo doc`, surfaces during code review of operator-implemented Approvers.

## Consequences

- New variant `Command::ApproveTool { tool_use_id: String, decision: ApprovalDecision }`.
- `ApprovalDecision` now lives in `entelix-core::approval` (re-exported from `entelix-agents` and `entelix`).
- `INTERRUPT_KIND_APPROVAL_PENDING` constant moves to `entelix-core::approval`.
- `PendingApprovalDecisions` (was `ApprovalDecisionOverrides` in `entelix-agents`) lives in `entelix-core::approval`, `pub`-visible. The typed `Command::ApproveTool` is the recommended operator-facing path; the direct ctx-attachment of `PendingApprovalDecisions` is the explicit lower-level path for non-graph dispatch (registry-only loops, tests).
- `entelix::ApprovalDecisionOverrides` removed from facade.
- `entelix::APPROVAL_PENDING_INTERRUPT_KIND` alias removed; canonical name is `entelix::INTERRUPT_KIND_APPROVAL_PENDING`.
- 4 regression tests in `approval_layer::tests` updated to use the new type names; behaviour unchanged.
- Example `18_tool_approval.rs` rewritten to demonstrate the typed `Command::ApproveTool` path. The example also uses a generically-named tool (`process_payment` rather than the previous vertical-flavoured `delete_production_database`) so it reads as a scaffolding pattern for general agentic SDK consumers.

## References

- ADR-0070 — `ApprovalLayer` Tower middleware (unchanged).
- ADR-0071 — `AwaitExternal` pause-and-resume *mechanism* (preserved); the surface portion (`ApprovalDecisionOverrides` + magic-string constant naming) is superseded by this ADR.
- ADR-0028 — `interrupt_before` / `interrupt_after` (the underlying graph-pause infrastructure this builds on).
- `crates/entelix-core/src/approval.rs` — canonical `ApprovalDecision` + `PendingApprovalDecisions` + `INTERRUPT_KIND_APPROVAL_PENDING`.
- `crates/entelix-graph/src/command.rs` — `Command::ApproveTool` variant.
- `crates/entelix-graph/src/compiled.rs::dispatch_from_checkpoint` — resume-side handling of `Command::ApproveTool`.
- `crates/entelix-agents/src/agent/approval_layer.rs` — reads `PendingApprovalDecisions` from ctx.
- `crates/entelix/examples/18_tool_approval.rs` — operator example demonstrating both the typed `Command::ApproveTool` path and the lower-level direct-ctx-attachment path.
