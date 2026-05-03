# ADR 0037 — `AuditSink` channel + managed-agent event variants

**Status**: Accepted
**Date**: 2026-05-01
**Decision**: Phase 4 of the post-7-차원-audit roadmap

## Context

Invariant #1 names `SessionGraph::events: Vec<GraphEvent>` as the
single first-class audit record — every other piece of state is
derived. The 7-차원 audit's H7 / F-rec items showed three
managed-agent lifecycle moments that never reached that record:

- **Sub-agent dispatch** — `Subagent::execute` opened a child run
  and called the parent's tools. The parent session log only saw
  the model's `ToolUse` parts; the fact that an agent boundary was
  crossed (with what name, into which `thread_id`) was implicit.
- **Supervisor handoff** — the supervisor recipe routed between
  named child agents (`SupervisorDecision::Agent("research")` →
  `SupervisorDecision::Agent("writer")`). The decisions disappeared
  into `tracing` spans and never landed in the durable event log.
- **Resume from checkpoint** — `CompiledGraph::resume_with` lifted
  state from a `Checkpointer` snapshot and re-entered the loop.
  Operators replaying the event log saw the post-resume entries
  but no marker that this was a resumption — the audit trail
  looked indistinguishable from a fresh run.
- **Long-term memory recall** — `query_semantic_memory` and
  `list_entity_facts` returned 0..N hits to the model. The hits
  themselves landed in the conversation transcript, but the
  retrieval *act* (which tier, which namespace, how many hits)
  never showed up in the audit channel — operators reviewing why
  an agent reached a conclusion couldn't see what it had been
  told to look at.

The shared symptom: managed-agent shape was being respected at the
type level but not at the audit level. A re-run reading only the
event log would not be able to reconstruct the lifecycle.

A second concern shaped the design: any tool / graph / recipe that
wants to *emit* one of these events should not depend on
`entelix-session`. `entelix-tools` and `entelix-graph` are
load-bearing crates and pulling in `SessionLog` (an async batch
`append(events)` API) at every emit site would invert the
dependency DAG (`entelix-core` is the root — invariant restated in
`crates/entelix-core/CLAUDE.md`).

## Decision

Land four pieces in one slice.

### 4 new `GraphEvent` variants

```rust
// crates/entelix-session/src/event.rs
pub enum GraphEvent {
    /* existing variants ... */

    /// Parent run dispatched a sub-agent into `sub_thread_id`.
    SubAgentInvoked {
        timestamp: DateTime<Utc>,
        agent_id: String,
        sub_thread_id: String,
    },

    /// Supervisor handed control between named agents.
    /// `from = None` on the first turn.
    AgentHandoff {
        timestamp: DateTime<Utc>,
        from: Option<String>,
        to: String,
    },

    /// Run resumed from a prior checkpoint.
    /// `from_checkpoint` empty when resuming from a fresh state.
    Resumed {
        timestamp: DateTime<Utc>,
        from_checkpoint: String,
    },

    /// A long-term memory tier returned `hits` records for
    /// `namespace_key`. The model-facing hits stay in the
    /// conversation transcript — only the *retrieval act* is
    /// audited, never the retrieved corpus (would balloon the log).
    MemoryRecall {
        timestamp: DateTime<Utc>,
        tier: String,
        namespace_key: String,
        hits: usize,
    },
}
```

`timestamp()` arm coverage extended; codecs / persistence backends
that already pattern-match every variant get a deterministic
compile-time error rather than a silent skip.

### `AuditSink` trait pinned in `entelix-core`

```rust
// crates/entelix-core/src/audit.rs
pub trait AuditSink: Send + Sync + 'static {
    fn record_sub_agent_invoked(&self, agent_id: &str, sub_thread_id: &str);
    fn record_agent_handoff(&self, from: Option<&str>, to: &str);
    fn record_resumed(&self, from_checkpoint: &str);
    fn record_memory_recall(&self, tier: &str, namespace_key: &str, hits: usize);
}

#[derive(Clone)]
pub struct AuditSinkHandle(Arc<dyn AuditSink>);
```

Three forces shaped the surface:

1. **Typed `record_*` methods, not `emit(GraphEvent)`** —
   `entelix-tools` and `entelix-graph` cannot see `GraphEvent`
   without depending on `entelix-session`, so the trait offers
   typed verbs that the adapter (`SessionAuditSink`) translates.
2. **`&self` synchronous methods** — emit sites sit inside hot
   dispatch loops (`Tool::execute`, `Subagent::execute`). Any
   `.await` ceremony at the emit site would push the audit
   channel into the same backpressure path as the agent loop;
   instead, the persistent impl spawns a detached task. Audit is
   fire-and-forget by contract: an audit-sink failure must never
   block the agent.
3. **Newtype handle for `Extensions`** — `AuditSinkHandle` wraps
   `Arc<dyn AuditSink>` so the `ExecutionContext::extension::<T>()`
   `TypeId` lookup is unambiguous (no `Arc<dyn ...>` confusion).

### `ExecutionContext::with_audit_sink` + `audit_sink`

```rust
// crates/entelix-core/src/context.rs
impl ExecutionContext {
    pub fn with_audit_sink(self, handle: AuditSinkHandle) -> Self { ... }
    pub fn audit_sink(&self) -> Option<Arc<AuditSinkHandle>> { ... }
}
```

Stored in the existing `Extensions` slot — single-tenancy per
`TypeId`, copy-on-write semantics, sub-agents inherit through
`ExecutionContext::child` automatically. Recipes that don't wire a
sink see no change in behaviour: the absent extension makes every
emit site a no-op via `ctx.audit_sink()` returning `None`.

### `SessionAuditSink` adapter in `entelix-session`

```rust
// crates/entelix-session/src/audit_sink.rs
pub struct SessionAuditSink {
    log: Arc<dyn SessionLog>,
    key: ThreadKey,
}
```

Maps each `record_*` call onto a fire-and-forget
`tokio::spawn(async move { log.append(&key, &[event]).await })`.
Persistence failures land in `tracing::warn!` rather than bubbling
back — by design the audit channel is a one-way pipe. Operators
who need transactional audit semantics can wrap their own
`AuditSink` impl that batches synchronously inside their service
boundary.

### Producer wiring across the four event types

| Event | Producer | Site |
|---|---|---|
| `MemoryRecall` | `QuerySemanticMemoryTool::execute` / `ListEntityFactsTool::execute` | `crates/entelix-tools/src/memory/mod.rs` — emit after the backend returns; `tier ∈ {"semantic","entity"}`, `namespace_key = ns.render()`, `hits = docs.len()`. `save_to_*` / `set_entity_fact` are mutations audited via the model's `ToolUse` part. |
| `SubAgentInvoked` | `SubagentTool::execute` | `crates/entelix-agents/src/subagent.rs` — emit before invoking the inner agent with a fresh UUID v7 `sub_thread_id`; the child context the inner agent sees inherits that `thread_id` (`ctx.clone().with_thread_id(sub_thread_id)`) so its persistence + audit records line up with the id the parent emitted. |
| `AgentHandoff` | `build_supervisor_graph` router node | `crates/entelix-agents/src/supervisor.rs` — emit inside the supervisor lambda when `SupervisorDecision::Agent(name)` is selected. `from = state.last_speaker.as_deref()` (None on the first turn), `to = name`. `Finish` produces no handoff entry. |
| `Resumed` | `CompiledGraph::dispatch_from_checkpoint` | `crates/entelix-graph/src/compiled.rs` — emit at the top of the dispatch routine before applying the `Command`. Both `resume` and `resume_from` route through this path, so the emit is single-origin. `from_checkpoint = checkpoint.id.to_hyphenated_string()`. |

Every producer guards on `if let Some(handle) = ctx.audit_sink()`
so the absent-sink path is a single typed lookup with no further
overhead. None of the producers `.await` inside the emit — the
trait is sync `&self` by design.

## Consequences

✅ Operators replaying a `SessionLog` can reconstruct: which
sub-agents ran (by name + thread), which handoffs happened, when
resumes occurred, and which memory tiers the model queried.
Invariant #1 ("session is event SSoT") covers the managed-agent
boundaries it was supposed to.
✅ Recipes that do not wire a sink — every existing test, every
single-agent recipe — see *zero* behaviour change. The trait is
opt-in by `ctx.audit_sink()` returning `None`.
✅ `entelix-tools` / `entelix-graph` emit through the typed
`AuditSink` surface without picking up an `entelix-session`
dependency. The DAG root (`entelix-core`) stays the only
shared edge.
✅ Audit-sink failures cannot cascade into agent failures —
spawned task swallows + warns. A vendor that loses durability
(network blip on Postgres) does not propagate to a user-visible
`Tool::execute` error.
✅ MemoryRecall captures the *act* (tier, namespace, hits) without
the corpus — the conversation transcript already holds the
retrieved content and storing it twice would balloon the log
without adding observability.
❌ `AuditSink` impls that genuinely need transactional emit must
write their own batching layer; the default adapter is
fire-and-forget. Rejected: making the trait `async` for everyone,
which would force `.await` at every emit site (not worth the
ceremony for the 95% one-way-pipe case).
❌ Sub-agent dispatch now allocates a fresh UUID v7 per
invocation. The cost is negligible (UUID v7 is ns-scale on modern
CPUs) and the alternative — letting parent + child share a
`thread_id` — would conflate two distinct persistence scopes in
`Checkpointer` + `SessionLog`, defeating the audit boundary the
event records. Operators that need pre-allocated child ids can
override before invoking by `ctx.clone().with_thread_id(my_id)`
and reading the same id off the emitted event.

## Alternatives considered

1. **Add `emit(GraphEvent)` to the trait** — pulls the
   `entelix-session::GraphEvent` enum into every emit site's
   crate. `entelix-tools` would have to depend on
   `entelix-session`, inverting the DAG. Rejected for that reason
   alone; the typed `record_*` shape also reads better at the call
   site (`record_memory_recall("semantic", ns, n)` vs. `emit(
   GraphEvent::MemoryRecall { timestamp: now(), tier:
   "semantic".into(), .. })`).
2. **Make `AuditSink::record_*` async** — removes the spawn from
   the adapter but pushes `.await` into every emit site, including
   the synchronous reducer/observer paths that have no async
   surface today. Rejected.
3. **Carry `Arc<dyn SessionLog>` directly in `ExecutionContext`** —
   does the same job for the four current emits but locks in
   `entelix-session` as a transitive dep of every emit site, and
   forces every alternative durable channel (Kafka, S3-batched,
   in-memory test sink) to implement the full `SessionLog` trait.
   Rejected; the typed `AuditSink` is the narrower waist.
4. **Stamp emits inside `Tool::execute` via a layer** — `tower::
   Layer<S>` wrapping `ToolInvocation` could emit a generic
   `tool_called` event without touching tool code. Works for
   tool-call audit but doesn't cover sub-agent / handoff / resume,
   which happen outside the tool dispatch path. Rejected as the
   primary mechanism; a layer that emits a tool-call audit can
   still be added later as a sibling concern.

## References

- ADR-0034 — heuristic policy externalisation (sibling slice that
  closed `Error::Provider` + supervisor decision).
- ADR-0035 — sub-agent layer-stack inheritance (the same
  `ExecutionContext::child` machinery that propagates the audit
  sink to children).
- 7-차원 audit fork report `audit-managed-shape-gaps` H7 / F-rec.
- `crates/entelix-core/src/audit.rs` — module-level rationale.
- `crates/entelix-session/src/audit_sink.rs` — adapter contract.
