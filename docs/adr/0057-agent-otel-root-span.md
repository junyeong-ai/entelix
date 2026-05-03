# ADR 0057 — `entelix.agent.run` root span for `Agent::execute` / `execute_stream`

**Status**: Accepted
**Date**: 2026-05-01
**Decision**: Phase 8 of the post-7-차원-audit roadmap (fifth sub-slice — Phase 8 close)

## Context

ADR-0009 (OTel adoption), ADR-0055 (cache token telemetry),
and ADR-0056 (tool I/O capture mode) shipped the
**model side** and **tool side** of the GenAI semconv
surface. `OtelLayer` wraps the model service and the tool
service independently; each emits its own
`gen_ai.request` / `gen_ai.response` / `gen_ai.tool.start` /
`gen_ai.tool.end` spans.

What was missing: an **agent-level root span**. When an
operator opened a trace UI, the model + tool spans for one
agent run appeared as siblings — there was no parent linking
them. For an OperatorReact loop with 4 layers (planner →
tool → planner → finish), the trace UI showed 5 disconnected
spans without explicit ordering or grouping.

The roadmap §S9 (Phase 8) called this out as the fifth and
final OTel item: **agent OTel span + tool I/O event body**
+ cache token telemetry + sampling/elicitation. This slice
lands the agent span half — closes Phase 8.

## Decision

Open a `tracing::info_span!("entelix.agent.run", ...)` at the
entry of `Agent::execute` and `Agent::execute_stream`.
Instrument the inner `run_inner(...)` future with this span
via `tracing::Instrument::instrument`. The
`tracing-opentelemetry` bridge sees the span as the active
parent for any spans / events emitted during `run_inner`,
which is when the model + tool services run.

```rust
let outcome = self
    .run_inner(input, ctx)
    .instrument(self.run_span(&run_id, ctx))
    .await;
```

### Span shape

- **Name**: `entelix.agent.run` — namespace-prefixed so
  operators filtering on `name LIKE 'entelix.%'` find every
  entelix-emitted span.
- **Target**: `gen_ai` — same target as model + tool layer
  events for consistent dashboard filter (`target = "gen_ai"`).
- **Fields**:
  - `gen_ai.agent.name` — the configured agent name (`"react"`,
    `"supervisor"`, etc.).
  - `entelix.run_id` — UUID v7 stamped per-execute (or
    inherited from `ctx.run_id()` when caller pre-allocated).
  - `entelix.tenant_id` — from `ctx.tenant_id()`.
  - `entelix.thread_id` — from `ctx.thread_id()` if set.

### Why not include `Started` / `Failed` / `Complete` in the span

The book-end events go to the agent's `AgentEventSink`, not
to tracing. They're a separate observability channel
(downstream consumers wire SSE / WebSockets / channel
receivers to them). Putting them inside the span would
either double-emit (sink + tracing event) or move sink data
into tracing — wrong shape.

Keeping the span scoped to `run_inner` means the sink emits
happen *before* (Started) and *after* (Complete / Failed)
the span. The span captures only the model / tool work;
the sink captures the lifecycle. Each channel's data is
where operators expect it.

### Why a method, not a free function

`run_span(&run_id, ctx)` reads field values off `&self.name`
and `ctx`. A free function would force the caller to pass
`&self.name` explicitly — unnecessary indirection. The
private method keeps the call sites compact (`execute` and
`execute_stream` each call it once).

### Why same span for both `execute` and `execute_stream`

Both methods drive the same inner runnable through
`run_inner`. Operators picking one or the other should see
identical trace shapes. Different span shapes would
fragment the dashboard query story (`name = "entelix.agent.run.execute"`
vs `name = "entelix.agent.run.stream"`).

### Why no opt-out

Tracing is opt-in by subscriber: the cost of an unsubscribed
`info_span!` is ~10ns (atomic load + branch). Operators
without OTel don't see any cost; operators with OTel see
correct trace nesting. No knob is needed.

### Tests

Three regression tests in `agent_otel_span.rs`:

- `execute_opens_entelix_agent_run_span_around_inner_runnable`
  — uses a custom `Runnable` that snapshots
  `tracing::Span::current()` metadata inside `invoke`. Asserts
  the span name is `"entelix.agent.run"` and target is
  `"gen_ai"`.
- `execute_stream_opens_entelix_agent_run_span_around_inner_runnable`
  — same pattern through the streaming surface. Confirms the
  span shape is identical across both entry points.
- `no_span_is_active_outside_execute` — sanity check: the
  span must be scoped to `run_inner`, not entered globally.

The tests install a `tracing-subscriber::fmt::Layer` once
per process (via `Once`) — without an installed subscriber,
`Span::current()` returns a disabled span with no metadata.
`tracing-subscriber` becomes a dev-dep on `entelix-agents`.

## Consequences

✅ Operators using a `tracing-opentelemetry` bridge see one
parent span per agent run with model + tool spans nested as
children. The trace UI shows agent → planner → tool → planner
→ finish as a tree, not as 5 disconnected siblings.
✅ Span fields (`gen_ai.agent.name`, `entelix.run_id`,
`entelix.tenant_id`, `entelix.thread_id`) match the rest of
the OTel surface — dashboards joining on `entelix.run_id`
work uniformly across the agent + layer stack.
✅ `execute` and `execute_stream` produce identical span
shapes — operator dashboards stay surface-agnostic.
✅ Sink emissions (`Started` / `Complete` / `Failed`) stay
outside the span — they're a separate observability channel
and don't pollute the OTel trace.
✅ Cost ~10ns per call without subscriber installed — no
opt-out needed.
❌ Adds `tracing-subscriber` as a dev-dep on entelix-agents.
Workspace already imports it (entelix-otel uses it), so the
build graph picks up cached artifacts.
❌ Operators wiring their own root span (e.g., HTTP request
span around the agent invocation) get one extra layer in the
trace tree. The agent span sits between the HTTP span and
the model span — clearer hierarchy, slightly more depth.

## Alternatives considered

1. **Manual span enter/exit**:
   ```rust
   let _enter = self.run_span(...).enter();
   self.run_inner(...).await
   ```
   The `_enter` guard is dropped at the `.await` point —
   the span won't survive the suspension. Tracing's
   `Instrument` trait exists to fix exactly this. Rejected.
2. **Span per `Runnable::invoke`** (lower-level
   instrumentation) — opens the span at every runnable in
   the chain, producing N nested spans per agent run.
   Trace UI noise; misses the agent-level context that
   matters most. Rejected.
3. **Separate `entelix.agent.run.execute` and `.stream`
   span names** — fragments dashboard queries. The shape is
   identical at the OTel level, the surface is identical to
   the operator. Same name. Rejected.
4. **`OtelLayer` wraps `Agent` directly** (mirrors
   model-side wrap) — makes the agent runtime depend on
   `entelix-otel`. The tracing-only approach keeps the
   dependency direction clean: `entelix-agents` emits
   `tracing` events / spans; `entelix-otel`'s subscriber
   picks them up. No new dep edge.
5. **Include sink events as span events** — would double-emit
   to both sink and tracing, OR force the sink data into
   tracing. Each channel has its own audience; keep them
   separate. Rejected.

## Operator usage patterns

**Default development setup** (with `tracing-subscriber`
fmt layer): the agent span shows up in console as
`entelix.agent.run{...}` with field values; child events
(model `gen_ai.response`, tool `gen_ai.tool.end`) nest
under it via tracing's indentation.

**Production with OTel** (with `tracing-opentelemetry`
bridge): the `entelix.agent.run` span becomes an OTLP root
span in the trace backend (Datadog, Honeycomb, Tempo, etc.).
Operator filters on `name = "entelix.agent.run"` to find
agent runs; clicks one to see the full tree of model + tool
work.

**Trace correlation across services**: operators wrapping
`agent.execute(...)` inside an HTTP request span (e.g., from
`tower-http::trace`) see `http.request → entelix.agent.run →
gen_ai.tool.* / gen_ai.response`. The full request chain is
linear in the trace UI.

**Filtering by tenant for multi-tenant deployments**:
`entelix.tenant_id` is on the agent span; child spans
inherit the agent's trace ID. Filter on
`entelix.tenant_id = "acme"` at the root to scope a query
to one tenant.

## References

- ADR-0009 — OTel GenAI semconv adoption (parent).
- ADR-0055 — cache token telemetry (sibling Phase 8 slice).
- ADR-0056 — tool I/O capture mode (sibling Phase 8 slice).
- 7-차원 roadmap §S9 — Phase 8 (MCP + OTel completion),
  fifth sub-slice closing the phase.
- `crates/entelix-agents/src/agent/mod.rs` — `Agent::execute`
  + `book_end_stream` instrument the inner future with
  `Self::run_span`.
- `crates/entelix-agents/tests/agent_otel_span.rs` — 3
  regression tests.
- `crates/entelix-agents/Cargo.toml` —
  `tracing-subscriber` added as dev-dep.
