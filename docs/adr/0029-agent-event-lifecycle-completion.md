# ADR-0029 — `AgentEvent` lifecycle completion

* **Status**: Accepted
* **Date**: 2026-04-27
* **Drivers**: Phase 8D
* **Refines**: ADR-0024 §"Agent runtime" (where `AgentEvent` was
  introduced as a `Started` / `Complete` book-end pair).

## Context

`AgentEvent<S>` shipped in Phase 7 with two variants — `Started` and
`Complete(S)`. Two operational gaps emerged in audit:

1. **No terminal signal on error.** When `Agent::execute` or
   `book_end_stream` propagates an error from the inner runnable,
   sinks observe `Started` and then silence. Caller-facing streams
   yield an `Err` (Rust-typed), but sinks — by design observability-
   facing — see only `AgentEvent` values, not `Result`s. SSE
   consumers and audit-log subscribers wait forever for the matching
   `Complete` that never comes.
2. **No invocation-level visibility.** Tool dispatch happens inside
   the agent's inner graph. Operators wanting per-tool spans, cost
   accounting, or partial-failure recovery have no event stream for
   it. The event surface is *turn-level only*.

Adjacent observation problem: an agent runtime can in principle
service multiple concurrent runs through one `AgentEventSink`
(`BroadcastSink` is the obvious case). Today's events carry only the
agent name — there is no per-run identifier to correlate `Started`
with its matching `Complete`.

## Decision

Extend `AgentEvent<S>` to carry the missing terminal + invocation-
level variants, propagate a `run_id` through every event, and
introduce a `ToolEventLayer` that emits invocation events on the
tool-service path.

### `AgentEvent<S>` shape

```rust
#[non_exhaustive]
pub enum AgentEvent<S> {
    /// Run opened.
    Started     { run_id: String, agent: String },

    /// One tool dispatch began.
    ToolStart   { run_id: String, tool_use_id: String, tool: String, input: Value },
    /// One tool dispatch finished successfully.
    ToolComplete{ run_id: String, tool_use_id: String, tool: String, duration_ms: u64 },
    /// One tool dispatch failed.
    ToolError   { run_id: String, tool_use_id: String, tool: String, error: String, duration_ms: u64 },

    /// Run terminated with the inner runnable's error.
    Failed      { run_id: String, error: String },
    /// Run terminated successfully.
    Complete    { run_id: String, state: S },
}
```

The contract: **every run emits `Started{run_id}` and exactly one
of `Complete{run_id, ...}` or `Failed{run_id, ...}`** with the same
`run_id`. Sink consumers can keyed-correlate; the agent itself
guarantees the pair, regardless of error path.

### `run_id` lifecycle

A `run_id` is **per execute / execute_stream call**. Generated on
the agent's entry boundary (`UUID v7` for time-ordering), carried in
every emitted event.

`ExecutionContext` gains an optional `run_id` field. The agent reads
it on entry and inherits the caller-supplied id when present;
otherwise it generates a fresh one and writes it back into a cloned
context that the inner runnable receives.

This serves two purposes:
- **Caller pre-allocates** — recipes that compose agents into a
  larger graph can pass the parent's run identifier so all events
  share a tree.
- **Telemetry correlation** — the `run_id` flows alongside
  `tenant_id` / `thread_id` into OTel attributes (`entelix.run_id`).

### `Failed` event

On any error path returning from the inner runnable (or from
pre-turn observers), the agent emits `Failed{run_id, error}` with
`error = err.to_string()` before propagating the typed error. The
sink is best-effort — if the sink itself errors (e.g., dropped
receiver), the agent swallows the secondary error and surfaces the
original.

The caller-facing stream still yields the typed `Err(err)` so
matching on `Error::Cancelled` / `Error::Provider { status, .. }` /
etc. continues to work. The `Failed` variant is the *sink-side*
terminal — mirrors `Complete` for symmetry.

### `ToolEventLayer` (operator-explicit wiring)

```rust
pub struct ToolEventLayer<S> { sink: Arc<dyn AgentEventSink<S>> }
```

Wraps any `Service<ToolInvocation>` and emits `ToolStart` /
`ToolComplete` / `ToolError` to the configured sink. The `run_id`
is read from `ExecutionContext::run_id`; absent context means the
layer falls through with no event (the layer is a no-op when the
agent runtime hasn't initialised run-id propagation).

**Wiring is operator-explicit, not auto-magic.** Recipes register
the layer themselves:

```rust
let registry = ToolRegistry::new()
    .layer(ToolEventLayer::new(sink.clone()))
    .register(my_tool)?;
```

Auto-wiring on `AgentBuilder::sink(...)` was rejected (ADR-0027
§"alternatives" deferred-pattern parity): magic wiring obscures the
cost model and makes debugging "why did this layer fire?" harder
than necessary. The cost is one explicit line in each recipe; the
benefit is unambiguous control flow.

### Why the per-tool variants live on the *same enum* as turn-level

Two alternatives considered:

1. **Two enums** — `AgentEvent` (turn-level) + `ToolEvent`
   (invocation-level), with separate sink traits.
2. **Single enum** — both event classes as variants of `AgentEvent`,
   one sink type fans out everything.

Option 2 wins because the typical SSE consumer wants *one* stream
per run with all relevant signals interleaved in time order. Two
streams would force a join in every consumer. The flat enum
trade-off is one extra level of `match` discrimination in
observability code — cheaper than maintaining two correlated
streams.

## Consequences

- `entelix-agents::AgentEvent<S>` gains four variants; the previous
  `Started { agent }` shape becomes `Started { run_id, agent }` and
  `Complete(S)` becomes `Complete { run_id, state }`. Tests / docs
  inside the workspace are updated atomically (no compat shims).
- `entelix-core::ExecutionContext` gains a `run_id: Option<String>`
  field with `with_run_id` / `run_id()` accessors.
- `entelix-agents::agent::tool_event_layer` is new (exports
  `ToolEventLayer<S>` + `ToolEventService<S>`).
- `entelix-otel::semconv` gains an `ENTELIX_RUN_ID` constant
  (`entelix.run_id`) for span attribute use; OTel layer reads
  `ctx.run_id()` and attaches when present.
- Public-api baselines refrozen for `entelix-core`,
  `entelix-agents`, `entelix-otel`, `entelix`.
- `AgentEvent<S>` carries an audit-projection method
  `to_graph_event(&self, timestamp) -> Option<GraphEvent>`
  establishing the runtime-side superset relationship to
  `entelix_session::GraphEvent` (RC-1 second half). The lifecycle
  variants (`Started`, `Complete`, `Failed`) project to `None` —
  they are runtime-only by design — while the tool variants
  project onto `GraphEvent::ToolCall` / `GraphEvent::ToolResult`
  with the runtime-only metric fields (`run_id`, `tool_version`,
  `duration_ms`) dropped at the boundary.

## Alternatives considered

1. **Use `AgentEvent::Started.agent` as the correlation key.**
   Rejected — multiple concurrent runs of the same agent collapse
   into one logical stream. `run_id` is the only correctly-scoped
   key.
2. **Auto-wire `ToolEventLayer` when `AgentBuilder::sink` is set.**
   Rejected — see "Wiring is operator-explicit" above.
3. **Emit raw `tower::Service` events.** Rejected — couples sinks
   to the middleware shape; the `AgentEvent` indirection lets the
   tool path migrate (e.g., to a different invocation envelope) in
   the future without touching consumers.
4. **Carry the typed `Error` value in `Failed`.** Rejected for the
   sink-facing variant — `Error` is `!Serialize` and `!Clone` for
   nested causes; `String` is the lossless-enough projection for
   observability. Caller-facing path retains the typed `Err`.

## References

- ADR-0024 — Agent SDK direction (initial `AgentEvent` shape)
- F3 — cancellation propagation (informs the run-id-on-context shape)
- RC-1 (audit SSoT) — `AgentEvent::to_graph_event` projection closes the second half (8B closed the first half by extending `GraphEvent` variants).
