# ADR 0081 — `AgentRunResult<S>` envelope at terminal

**Status**: Accepted
**Date**: 2026-05-05
**Decision**: `Agent::execute` and `Agent::execute_with` return `Result<AgentRunResult<S>>` instead of `Result<S>`. The envelope carries the terminal state, the per-run correlation id (already minted by `scoped_run_context`), and a frozen `UsageSnapshot` of the `RunBudget` counters captured *between* the inner runnable returning and `on_complete` observers firing. `AgentEvent::Complete` carries the same `usage: Option<UsageSnapshot>` so the streaming surface (`Agent::execute_stream`) and the one-shot surface observe one identical terminal artifact. The `Runnable<S, S>` impl on `Agent<S>` unwraps the envelope's `state` so composing agents inside larger graphs see the existing `S → S` shape — callers that need the per-run usage / id go through `Agent::execute` directly.

## Context

ADR-0080 added `RunBudget` — a 5-axis pre/post enforcement cap shared by all dispatches in one run via `Arc<RunBudgetState>`. Operators who wire a budget want to read out the final counter state per run: "this run consumed 3 of the 5 model-call budget, 1.2k of the 4k token budget". Three surfaces compete:

1. **Sink-only** — emit `AgentEvent::Complete { state, run_id }` with usage attached, force the caller to wire a `CaptureSink` to read it. Fails the `Auto`-mode `execute()` shape that does not own a sink loop, and re-introduces stateful telemetry coupling that the agent runtime exists to avoid.
2. **`RunBudget` accessor on the caller** — caller clones the `Arc<RunBudgetState>` before / after `execute`, computes the delta. Works, but the delta is not the snapshot the agent would naturally observe; observer dispatches (memory consolidation, summary writes) that happen post-invoke can mutate the same shared budget, so the caller's "after" snapshot reflects observer overhead the run itself did not incur.
3. **Envelope return type** — agent returns `AgentRunResult<S>` carrying the snapshot frozen at the agent-run boundary. Caller reads `result.usage` directly; observers cannot perturb the value because the freeze happens before they fire.

Fork 1 industry parity: pydantic-ai 1.90's `agent.run(...) -> AgentRunResult` already carries `usage: RunUsage` alongside `output` — one envelope per run, frozen artifact at terminal. LangGraph returns the typed state directly (no envelope) but its `runnable.with_config(...).invoke(...)` provides `metadata` through a side-channel that callers thread through `RunnableConfig`. OpenAI Agents SDK's `Runner.run_sync` returns a `RunResult` carrying both the final output and a usage tally. The envelope shape is the converging industry pattern.

## Decision

### `AgentRunResult<S>` lives in `entelix-agents`

Not in `entelix-core`. `AgentRunResult<S>` is the agent layer's terminal artifact — graph-level dispatch (`CompiledGraph::invoke`) keeps the bare `Result<S>` shape because graphs are domain-shaped state machines and the envelope concept is agent-runtime-scoped. Mixing the two would force every `Runnable<I, O>` implementor to thread an envelope they do not own.

Crate-level position: `crates/entelix-agents/src/agent/result.rs`. Re-exported through `entelix-agents` lib.rs and the `entelix` facade as `AgentRunResult`.

### Three fields, `#[non_exhaustive]`

```rust
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct AgentRunResult<S> {
    pub state: S,
    pub run_id: String,
    pub usage: Option<UsageSnapshot>,
}
```

- `state` — final value the inner runnable produced. Public field for ergonomic destructuring.
- `run_id` — UUID v7 by default (or the value the caller pre-stamped via `ExecutionContext::with_run_id`). Echoes the same id every `AgentEvent` carries, so callers correlating logs / sinks / direct returns work from one identifier.
- `usage` — `None` when no `RunBudget` was attached to the context (operators that don't cap usage pay zero overhead); `Some(UsageSnapshot)` otherwise.

`#[non_exhaustive]` keeps the shape forward-compatible — adding a `cost_usd: Option<CostSnapshot>` field post-1.0 to mirror `entelix-policy::CostMeter` is non-breaking.

### Snapshot frozen *between* runnable return and observer dispatch

```rust
async fn run_inner(&self, input: S, run_id: String, ctx: &ExecutionContext)
    -> Result<AgentRunResult<S>>
{
    for observer in &self.observers {
        observer.pre_turn(&input, ctx).await?;
    }
    let state = self.runnable.invoke(input, ctx).await?;
    let usage = ctx.run_budget().map(|budget| budget.snapshot());  // ← frozen here
    for observer in &self.observers {
        observer.on_complete(&state, ctx).await?;
    }
    Ok(AgentRunResult::new(state, run_id, usage))
}
```

The freeze position is load-bearing. Observers may issue downstream `ChatModel` calls (vector store writes, summary persistence) — those calls land on the same `RunBudget` through the shared `Arc<RunBudgetState>` and would inflate any post-observer snapshot beyond what the agent run itself consumed. Freezing pre-observer reflects exactly the agent-run cost: the observer's own dispatches show up on the *next* run's snapshot if the observer state persists, which is the correct attribution.

A unit test (`execute_envelope_carries_frozen_usage_snapshot_when_budget_is_attached` in `crates/entelix-agents/src/agent/mod.rs`) regression-locks the freeze — it pre-stamps a counter, runs the agent, mutates the budget after the call returns, and asserts the snapshot is unchanged. The test exists specifically to catch a future refactor that moves `snapshot()` to read from `Arc<RunBudgetState>` lazily.

### Streaming surface mirrors the envelope

`AgentEvent::Complete` gains a `usage: Option<UsageSnapshot>` field. `Agent::execute_stream`'s `book_end_stream` emits the same snapshot the one-shot path returns:

```rust
match outcome {
    Ok(result) => {
        let complete = AgentEvent::Complete {
            run_id: result.run_id,
            state: result.state,
            usage: result.usage,
        };
        ...
    }
    ...
}
```

A single freeze point feeds both surfaces — operators that wire telemetry through the sink see the same artifact direct callers receive. The `to_graph_event` projection on `AgentEvent::Complete` continues to return `None` (audit log records `ToolCall` / `ToolResult` pairs, not lifecycle markers).

### `Runnable<S, S>` unwraps the envelope

```rust
impl<S> Runnable<S, S> for Agent<S> where S: ... {
    async fn invoke(&self, input: S, ctx: &ExecutionContext) -> Result<S> {
        self.execute(input, ctx).await.map(AgentRunResult::into_state)
    }
}
```

Composing `Agent<S>` inside a parent `StateGraph<ParentState>` keeps the `S → S` shape that the rest of the graph contract expects. The envelope is an agent-runtime concept; the graph dispatch layer does not see it.

### `pub(crate)` constructor

`AgentRunResult::new` is `pub(crate)` to `entelix-agents` — third parties cannot synthesize a result that did not come from a real run. Public field access supports destructuring (`let AgentRunResult { state, usage, .. } = ...`); `into_state(self) -> S` is the explicit unwrap helper for the common composition pattern.

## Why not these alternatives

- **Add `usage()` accessor to `Agent`** — accessor reads through the live `Arc<RunBudgetState>`, which observers can mutate. Same bug as alternative 2 above.
- **Embed snapshot in `ExecutionContext` extension on completion** — extensions are clone-on-write; the parent caller's `ctx` would not see the mutation made inside the agent run. Sub-agents already share `Arc`-backed state; layering a write-through extension just for the snapshot is more machinery for less ergonomic access.
- **Return `(S, RunSummary)` tuple** — wider blast radius, no `#[non_exhaustive]` evolution path, and tuples can't carry doc on each field. Envelope is strictly better.
- **Mirror pydantic-ai's `output` field name instead of `state`** — pydantic-ai's runtime is single-message-out-shaped; entelix's typed state is the existing public surface (`StateGraph<S>` / `Runnable<S, S>`). Using `output` would create a synonym for `state` only on this one type.

## Consequences

- **Breaking change** — `Agent::execute`'s public signature changes from `Result<S>` to `Result<AgentRunResult<S>>`. Per invariant 14 (no shim), callers update at the same time the type ships. `Subagent::into_tool` consumers, `team_from_supervisor`, and the `Runnable<S, S>` impl all unwrap via `into_state()` so the existing graph composition stays intact.
- **Public-API baselines** — `entelix-agents` and `entelix` baselines refrozen.
- **CHANGELOG** — landed under "Changed" for the upcoming 1.0.0-rc.2 cut. The prior `Result<S>` shape never shipped on a published 1.0; the breaking-change marker is for clarity, not contract violation.
- **Future axes** — the `#[non_exhaustive]` envelope means adding `cost_usd: Option<CostSnapshot>` (when `entelix-policy::CostMeter` exposes its own snapshot type) is non-breaking. The same path applies to any future per-run tally the operator community asks for.

## References

- ADR-0080 — `RunBudget` 5-axis (the snapshot's source).
- pydantic-ai 1.90 `AgentRunResult` — Fork 1 industry parity reference.
- OpenAI Agents SDK `Runner.run_sync` `RunResult` — second confirming reference.
- Invariant 14 (no shim) — drives the breaking-change-without-deprecation choice.
