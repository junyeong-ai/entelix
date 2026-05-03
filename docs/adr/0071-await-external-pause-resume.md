# ADR 0071 — `AwaitExternal` pause-and-resume via graph interrupt + decision overrides

**Status**: Surface portion superseded by [ADR-0072](./0072-approval-resume-via-command.md). The pause-and-checkpoint *mechanism* described here remains current; the operator-facing resume surface (`ApprovalDecisionOverrides` HashMap + magic-string discriminator naming) was replaced by the typed `Command::ApproveTool` resume primitive on 2026-05-03.
**Date**: 2026-05-02
**Decision**: When `Approver::decide` returns `ApprovalDecision::AwaitExternal`, `ApprovalLayer` raises `Error::Interrupted { payload }` with a structured `kind = "approval_pending"` payload identifying the pending dispatch. The graph dispatch loop persists a checkpoint with pre-node state and surfaces the typed error to the caller. Resume attaches the operator's eventual decision to `ExecutionContext` via `ApprovalDecisionOverrides` keyed by `tool_use_id`; the layer's override-lookup runs before the approver and short-circuits without re-asking.

## Context

ADR-0070 wired the `ApprovalLayer` Tower middleware that gates tool dispatch through `Approver::decide`. Two of three `ApprovalDecision` variants were live: `Approve` ran the inner service; `Reject` short-circuited with a typed 4xx. The third — `AwaitExternal` — was documented as a follow-up:

> **`AwaitExternal` decisions.** Today the layer treats `AwaitExternal` as a rejection with a placeholder reason — the full wait-and-resume flow ships in a follow-up slice (it needs graph-level `interrupt` integration, out of scope here).

The motivating use case is **out-of-band human review**: an agent dispatches a high-stakes tool call, the approver returns `AwaitExternal`, the run pauses and releases inflight resources, an operator reviews via a separate UI/Slack/email channel, and the resume restarts the dispatch with the operator's eventual decision. Pre-slice-123 the agent had no way to express this — `AwaitExternal` produced a typed error and the agent run terminated, losing the pending dispatch context.

`CompiledGraph::execute_loop_inner` already had infrastructure for this shape: any node returning `Err(Error::Interrupted { payload })` triggers a checkpoint write with the pre-node state, and the typed error bubbles to the caller. The pause is observable, the state is durable, and resume re-enters the same node from the checkpoint. What was missing was (a) the layer raising `Interrupted` instead of a typed error, (b) the ReAct tool node propagating `Interrupted` instead of catching as `is_error`, and (c) a mechanism for resume to deliver the operator's eventual decision to the same dispatch path so the approver isn't re-asked indefinitely.

## Decision

Three coordinated changes wire the full pause-and-resume flow:

### 1. `ApprovalLayer` raises `Error::Interrupted` on `AwaitExternal`

```rust
ApprovalDecision::AwaitExternal => {
    Err(Error::Interrupted {
        payload: json!({
            "kind": INTERRUPT_KIND_APPROVAL_PENDING,
            "run_id": run_id,
            "tool_use_id": tool_use_id,
            "tool": tool_name,
            "input": input,
        }),
    })
}
```

The payload's `kind` discriminator (`"approval_pending"`, exposed as `INTERRUPT_KIND_APPROVAL_PENDING`) lets resume code recognise approval-driven pauses without parsing the full payload. The remaining fields (`tool_use_id`, `tool`, `input`) carry enough context for the operator's review UI to render the pending dispatch.

### 2. ReAct tool node propagates `Interrupted` instead of catching

Pre-slice-123 the tool node caught every error and converted to `ContentPart::ToolResult { is_error: true }`. That swallowed the interrupt and the agent treated the pause as a "tool failure" — the model would synthesise a reply about the failure and the loop continued.

Slice 123 special-cases `Error::Interrupted`:

```rust
match tools.dispatch(id, name, input.clone(), &ctx).await {
    Ok(value) => (ToolResultContent::Json(value), false),
    Err(Error::Interrupted { payload }) => return Err(Error::Interrupted { payload }),
    Err(e) => (ToolResultContent::Text(e.render_for_llm()), true),
}
```

The early-return abandons any successful tool calls earlier in the same `tool_use` block — the resume re-fires the entire tool node from the checkpoint. This pins the responsibility on `Tool::execute` impls and the operator's `Approver` to be **idempotent on `tool_use_id`**: a tool call that already ran successfully should produce the same result on re-fire, and the approver should return the same decision (or use `ApprovalDecisionOverrides` to skip approval entirely on re-entry).

### 3. `ApprovalDecisionOverrides` ctx extension delivers the operator's decision

```rust
pub struct ApprovalDecisionOverrides {
    by_tool_use_id: HashMap<String, ApprovalDecision>,
}
```

The layer's `call` method runs the override lookup **before** the approver:

```rust
let override_decision = invocation
    .ctx
    .extension::<ApprovalDecisionOverrides>()
    .and_then(|o| o.lookup(&invocation.tool_use_id).cloned());
let decision = match override_decision {
    Some(d) => d,
    None => approver.decide(...).await?,
};
```

The operator attaches overrides on the resume call:

```rust
let overrides = ApprovalDecisionOverrides::new()
    .with_decision("tu-1", ApprovalDecision::Approve);
let resume_ctx = ctx.add_extension(overrides);
agent.execute(state, &resume_ctx).await?;
```

Resumes that re-invoke the dispatch with `tool_use_id == "tu-1"` skip the approver and use the supplied decision. Other dispatches (different `tool_use_id`) fall through to the approver as normal — the override is per-id, not global.

### Why not extend the `Command` enum instead

A `Command::ApproveTool { tool_use_id, decision }` resume primitive would auto-attach the override during `CompiledGraph::resume_with`. That's cleaner at the API boundary but requires extending the `Command` enum and threading the resume payload through the graph's resume path. The ctx-extension approach reaches feature-parity with one new type and zero changes to graph dispatch semantics, and operators that build custom resume flows (e.g. a job-queue worker that drives many parallel resumes) keep direct control over the override attachment. A future slice can layer a `Command::ApproveTool` convenience on top without breaking the underlying mechanism.

### Why `tool_use_id` keying (not `(run_id, tool_use_id)`)

`tool_use_id` is unique within the conversation by IR contract — generated by the model alongside each `ContentPart::ToolUse` and threaded through `ToolResult` as the correlation key. Across runs the model produces fresh ids, so collisions are scoped to the run that produced them. Adding `run_id` to the key would not increase precision and would force operators to thread the run id through their review UI.

### Idempotency requirements

The pause-and-resume flow re-enters the tool node from a pre-node checkpoint. Concretely:

- **Approver impls**: must return the same decision for the same `tool_use_id` across calls. Implementations that call out to a stateful review queue should look up by id first; the `ApprovalDecisionOverrides` shortcut handles the in-memory case.
- **Tool impls**: should be idempotent on the same `(tool_use_id, input)` pair. Tools that already honour the `Idempotency-Key` ctx field (vendor dedupe) are automatically safe.
- **Single-tool-call resumes**: when the model emitted a single `ToolUse` block, the resume re-fires only that one call — no idempotency risk beyond the call itself.
- **Multi-tool-call resumes**: when the model emitted multiple `ToolUse` blocks and one returned `Interrupted`, the resume re-fires *all* of them. Idempotency on each is the operator's contract.

## Consequences

- New `entelix_agents::ApprovalDecisionOverrides` type — operator-facing knob for the resume path. Re-exported through the facade.
- New `entelix_agents::INTERRUPT_KIND_APPROVAL_PENDING` constant — discriminator in the `Error::Interrupted::payload`. Re-exported through the facade as `APPROVAL_PENDING_INTERRUPT_KIND` (alias avoids collision with future interrupt-kind constants from other layers).
- `ApprovalLayer::call` raises `Error::Interrupted` on `AwaitExternal` (was `Error::InvalidRequest`).
- ReAct tool node propagates `Error::Interrupted` (was caught as `is_error`).
- 4 new regression tests in `approval_layer::tests`: payload shape, override short-circuits approver, override propagates reject, override scoped to matching `tool_use_id`.
- Operators with `AwaitExternal`-returning approvers must now build a resume path; pre-slice-123 their AwaitExternal calls erroneously terminated runs with a typed error.

## References

- ADR-0028 — `interrupt_before` / `interrupt_after` deadlock avoidance on resume; the same checkpoint-and-bubble mechanism this ADR builds on.
- ADR-0029 — `AgentEvent` lifecycle; pause does not emit a terminal `Complete` / `Failed`, the `Started` / next-`Started` pair correlates the run.
- ADR-0070 — `ApprovalLayer` and the two non-`AwaitExternal` decisions; this ADR closes the third.
- `crates/entelix-agents/src/agent/approval_layer.rs` — implementation + tests.
- `crates/entelix-agents/src/react_agent.rs::tool_node` — `Interrupted` propagation.
- `crates/entelix-graph/src/compiled.rs::execute_loop_inner` — checkpoint-and-bubble infrastructure (existing, unchanged).
