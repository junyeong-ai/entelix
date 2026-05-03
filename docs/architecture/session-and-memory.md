# Session & Memory — three-tier state model

## Why three tiers, not one

LangGraph collapses thread state + audit trail into one Checkpointer. Anthropic separates Session (events) from working state. LangChain (legacy) had Memory as a fourth concept. Mixing these creates bugs that take a release cycle to undo.

entelix makes the partition explicit:

| Tier | Concept | Lifetime | Owner crate | Storage table |
|---|---|---|---|---|
| **1. StateGraph state** | working memory the graph mutates | per-thread, working | `entelix-graph` | `checkpoints` |
| **2. SessionGraph events** | append-only audit log | per-thread, durable | `entelix-session` | `session_events` |
| **3. Memory Store** | cross-thread persistent knowledge | cross-thread, durable | `entelix-memory` | `memory_items` |

These three tiers are **orthogonal**. Conflating them in code is invariant 3 violation (CLAUDE.md).

## Tier 1 — StateGraph state

Owned by `entelix-graph`. Detail in `docs/architecture/state-graph.md`. Summary:

- Typed `S` per graph (user-defined struct).
- Mutated by node functions via `StateUpdate<S>`.
- Reducer determines per-field merge.
- Snapshotted to `Checkpointer` per node (default).
- Used to **resume in-flight executions** without replaying from event 0.

## Tier 2 — SessionGraph events

Owned by `entelix-session`.

```rust
pub struct SessionGraph {
    pub thread_id: ThreadId,
    pub tenant_id: TenantId,
    pub events: Vec<GraphEvent>,
    pub archived_watermark: Option<EventSeq>,
}

#[non_exhaustive]
pub enum GraphEvent {
    UserMessage { content: Vec<ContentPart>, at: DateTime<Utc> },
    AssistantStart { node: NodeName },
    AssistantDelta { text: String, at: DateTime<Utc> },
    AssistantComplete { content: Vec<ContentPart>, usage: Usage, at: DateTime<Utc> },
    ToolCall { id: String, name: String, input: serde_json::Value, at: DateTime<Utc> },
    ToolResult { tool_call_id: String, output: serde_json::Value, is_error: bool, at: DateTime<Utc> },
    SystemMessage { content: String, at: DateTime<Utc> },
    BranchCreate { branch: BranchName, parent: Option<EventSeq> },
    Checkpoint { checkpoint_id: CheckpointId, at_node: NodeName },
    Interrupt { reason: String, payload: serde_json::Value },
    Resume { command: ResumeCommand },
    HookFired { name: String, phase: HookPhase },
    Error { message: String, recoverable: bool },
}
```

### Why an event log when we have Checkpointer?

- **Audit & compliance** — events are immutable; checkpoints are derived. Auditors require event log.
- **Replay independence** — you can rebuild any state shape from events. Useful when state schema evolves.
- **Tool result visibility** — checkpoints carry final state, but events show *every* intermediate tool call.
- **Time travel beyond checkpoint** — checkpoints are sparse; events are dense.

### Archival watermark

Long-lived sessions accumulate events. The `archived_watermark` enables soft-deletion of old events past a configurable threshold:

```rust
session.archive_before(EventSeq::from_offset(now() - Duration::days(30)));
```

After archival, only events after the watermark are loaded by default. Pre-watermark events stay in storage for compliance but are not pulled into memory unless explicitly requested via `load_archived_range`.


### Fork

```rust
let child_session = parent_session.fork(at_event_seq, "alternative_branch")?;
```

Fork creates a new session whose events list starts as a copy of parent's events up to `at_event_seq`. Sub-agent isolation is built on this primitive.

## Tier 3 — Memory Store (cross-thread)

Owned by `entelix-memory`.

```rust
pub trait Store<V>: Send + Sync
where V: Serialize + DeserializeOwned + Send + Sync {
    async fn put(&self, ns: &Namespace, key: &str, value: V) -> Result<(), StoreError>;
    async fn get(&self, ns: &Namespace, key: &str) -> Result<Option<V>, StoreError>;
    async fn list(&self, ns: &Namespace, prefix: Option<&str>, limit: usize) -> Result<Vec<(String, V)>, StoreError>;
    async fn search(&self, ns: &Namespace, query: SearchQuery) -> Result<Vec<(String, V)>, StoreError>;
    async fn delete(&self, ns: &Namespace, key: &str) -> Result<(), StoreError>;
}

pub struct Namespace {
    pub tenant_id: TenantId,                          // F2 — mandatory
    pub scope: Vec<Cow<'static, str>>,                // hierarchical path
}

impl Namespace {
    /// The ONLY constructor. tenant_id mandatory by API design.
    pub fn new(tenant_id: TenantId, scope: impl IntoIterator<Item = impl Into<Cow<'static, str>>>) -> Self;
}
```

### Multi-tenant safety (F2 mitigation, invariant 11)

There is no `Namespace::default()`. There is no `Namespace::without_tenant()`. Cross-tenant data leak via Memory Store is **structurally impossible**.

### Memory patterns (built-in in `entelix-memory`)

Five first-class patterns over the `Store<V>` trait:

| Pattern | Backed by | Needs Embedder? |
|---|---|---|
| `BufferMemory` | `Store<Vec<Message>>` — sliding window | no |
| `SummaryMemory` | `Store<String>` — auto-summary via LLM call | no (uses ChatModel) |
| `EntityMemory` | `Store<HashMap<String, EntityFacts>>` — extracted facts | no (uses ChatModel) |
| `SemanticMemory<E, V>` | `VectorStore + Embedder` user plug-ins | yes (user-supplied) |
| `EpisodicMemory<V>` | `Store<Vec<Episode<V>>>` — time-ordered episode log with `range` / `recent` / `since` queries (ADR-0038) | no |

`GraphMemory<N, E>` is a separate trait (typed nodes + timestamped
edges, BFS traversal + shortest-path) — not memory tier composition
over `Store<V>`. Built-ins: `InMemoryGraphMemory` (reference);
companion `entelix-graphmemory-pg` provides `PgGraphMemory<N, E>`
with `WITH RECURSIVE` BFS fast path and `INSERT … SELECT FROM
UNNEST(…)` bulk insert.

### Embedder / Retriever / VectorStore / GraphMemory traits

```rust
pub trait Embedder: Send + Sync + 'static {
    async fn embed(&self, ctx: &ExecutionContext, texts: &[String]) -> Result<Vec<Embedding>>;
    fn dimension(&self) -> usize;
}

pub trait VectorStore: Send + Sync + 'static {
    async fn add(&self, ctx: &ExecutionContext, ns: &Namespace, items: Vec<VectorItem>) -> Result<()>;
    async fn search_filtered(&self, ctx: &ExecutionContext, ns: &Namespace, query: Embedding,
                             k: usize, filter: VectorFilter) -> Result<Vec<VectorMatch>>;
    async fn delete(&self, ctx: &ExecutionContext, ns: &Namespace, ids: Vec<String>) -> Result<()>;
}

pub trait Retriever: Send + Sync + 'static {
    async fn retrieve(&self, query: &str, ctx: &ExecutionContext) -> Result<Vec<Document>>;
}

pub trait GraphMemory<N, E>: Send + Sync + 'static
where
    N: Clone + Send + Sync + 'static,
    E: Clone + Send + Sync + 'static,
{
    // Write
    async fn add_node(&self, ctx: &ExecutionContext, ns: &Namespace, node: N) -> Result<NodeId>;
    async fn add_edge(&self, ctx: &ExecutionContext, ns: &Namespace,
                      from: &NodeId, to: &NodeId, edge: E, ts: DateTime<Utc>) -> Result<EdgeId>;
    async fn add_edges_batch(&self, ctx: &ExecutionContext, ns: &Namespace,
                             edges: Vec<(NodeId, NodeId, E, DateTime<Utc>)>) -> Result<Vec<EdgeId>>;
    async fn delete_edge(&self, ctx: &ExecutionContext, ns: &Namespace, edge_id: &EdgeId) -> Result<()>;
    async fn delete_node(&self, ctx: &ExecutionContext, ns: &Namespace, node_id: &NodeId) -> Result<usize>;
    // Read single
    async fn node(&self, ctx: &ExecutionContext, ns: &Namespace, id: &NodeId) -> Result<Option<N>>;
    async fn edge(&self, ctx: &ExecutionContext, ns: &Namespace, id: &EdgeId) -> Result<Option<GraphHop<E>>>;
    // Read many
    async fn neighbors(&self, ctx: &ExecutionContext, ns: &Namespace,
                       node: &NodeId, direction: Direction) -> Result<Vec<(EdgeId, NodeId, E)>>;
    async fn traverse(&self, ctx: &ExecutionContext, ns: &Namespace,
                      start: &NodeId, direction: Direction, max_depth: usize) -> Result<Vec<GraphHop<E>>>;
    async fn find_path(&self, ctx: &ExecutionContext, ns: &Namespace,
                       from: &NodeId, to: &NodeId, direction: Direction, max_depth: usize)
                       -> Result<Option<Vec<GraphHop<E>>>>;
    async fn temporal_filter(&self, ctx: &ExecutionContext, ns: &Namespace,
                             from: DateTime<Utc>, to: DateTime<Utc>) -> Result<Vec<GraphHop<E>>>;
    // Operator metric / cleanup (default impls)
    async fn node_count(&self, _ctx: &ExecutionContext, _ns: &Namespace) -> Result<usize> { Ok(0) }
    async fn edge_count(&self, _ctx: &ExecutionContext, _ns: &Namespace) -> Result<usize> { Ok(0) }
    async fn prune_older_than(&self, _ctx: &ExecutionContext, _ns: &Namespace, _ttl: Duration) -> Result<usize> { Ok(0) }
}
```

The `GraphMemory` trait is intentionally lean — operator-side
admin paths (paginated enumeration, orphan-prune) live as
**inherent** methods on backend types like `PgGraphMemory<N, E>`,
not on the trait, so trait-erased dispatch (`Arc<dyn GraphMemory<N, E>>`)
sees only the surface agents call (ADR-0065).

User plugs concrete impls. 1.0 ships:
- `entelix-memory-openai` — OpenAI Embeddings
- `entelix-memory-pgvector` — Postgres + pgvector with row-level security
- `entelix-memory-qdrant` — Qdrant gRPC
- `entelix-graphmemory-pg` — Postgres GraphMemory with row-level security

(See ADR-0008 — companion crate pattern.)

## Audit channel (Tier 4 — managed-agent lifecycle, invariant 18)

`entelix::AuditSink` is the typed channel for managed-agent
lifecycle events:

```rust
pub trait AuditSink: Send + Sync + 'static {
    fn record_sub_agent_invoked(&self, agent_id: &str, sub_thread_id: &str);
    fn record_agent_handoff(&self, from: Option<&str>, to: &str);
    fn record_resumed(&self, from_checkpoint: &str);
    fn record_memory_recall(&self, tier: &str, namespace_key: &str, hits: usize);
}
```

Methods are `&self` (sync) so emit sites in hot dispatch loops
stay free of `.await` ceremony. Failures land in `tracing::warn!`
and never propagate back — the audit channel is one-way by
contract (ADR-0037).

`entelix-session::SessionAuditSink` is the canonical adapter that
maps each `record_*` call onto `SessionLog::append` of the
corresponding `GraphEvent` variant
(`SubAgentInvoked` / `AgentHandoff` / `Resumed` / `MemoryRecall`).
Replays of the session reconstruct the managed-agent lifecycle
without re-running the dispatch path.

Operators wire the sink onto `ExecutionContext::with_audit_sink`;
recipes that don't wire one incur zero overhead (`ctx.audit_sink()`
returns `None` → emit sites become no-ops).

## Storage layout (`entelix-persistence` Postgres example)

```sql
-- Tier 1
CREATE TABLE checkpoints (
    id UUID PRIMARY KEY,
    thread_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    parent_id UUID,
    state JSONB NOT NULL,
    at_node TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    INDEX (thread_id, created_at DESC),
    INDEX (tenant_id)
);

-- Tier 2
CREATE TABLE session_events (
    thread_id UUID NOT NULL,
    seq BIGSERIAL,
    tenant_id UUID NOT NULL,
    event JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (thread_id, seq),
    INDEX (tenant_id, created_at)
);

CREATE TABLE session_archive_watermarks (
    thread_id UUID PRIMARY KEY,
    watermark_seq BIGINT NOT NULL,
    archived_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Tier 3
CREATE TABLE memory_items (
    tenant_id UUID NOT NULL,
    scope TEXT NOT NULL,                  -- joined namespace path
    key TEXT NOT NULL,
    value JSONB NOT NULL,
    embedding VECTOR,                     -- pgvector, optional
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, scope, key),
    INDEX (tenant_id, scope)
);
```

All three tables include `tenant_id` and have indexes scoped by
it. Postgres row-level security (RLS) is **mandatory** in every
backend that persists tenant-scoped data: each table gets `ENABLE`
+ `FORCE ROW LEVEL SECURITY` + a `tenant_isolation` policy that
consults `current_setting('entelix.tenant_id', true)`. Backends
call `set_tenant_session(tx, ns.tenant_id())` once per request
inside a tenant-stamped tx envelope (ADR-0041) — defense-in-depth
on top of the application-level `Namespace` scoping.

The same RLS pattern applies to the companion vector / graph
backends: `PgVectorStore` (ADR-0044), `PgGraphMemory` (ADR-0043).

## Distributed lock

When a thread receives concurrent requests (multi-pod scenario), only one harness should mutate at a time:

```rust
persistence.with_session_lock(thread_id, async {
    // load → mutate → save under lock
    Ok::<_, Error>(())
}).await?;
```

Backends:
- `PostgresPersistence` — `pg_advisory_xact_lock(hash(tenant_id, thread_id))`
- `RedisPersistence` — `SET NX PX` with Lua release script
- `MemoryPersistence` — `tokio::sync::Mutex` (test only)

## Lock ordering (F11 mitigation, CLAUDE.md)

```
tenant > session > checkpoint > memory > tool_registry > orchestrator
```

Always acquire in this order. Never await a user future while holding any of these.

## Cross-references

- CLAUDE.md invariants 1, 2, 3, 7, 11, 13, 18
- ADR-0007 — Memory abstractions in 1.0
- ADR-0008 — companion crate pattern (concrete `Embedder` /
  `VectorStore` / `GraphMemory` impls live in sibling crates)
- ADR-0017 — `tenant_id` strengthening (`Namespace::new`
  panics on empty)
- ADR-0037 — `AuditSink` trait + 4 `record_*` verbs +
  `SessionAuditSink` (invariant 18)
- ADR-0038 — `EpisodicMemory<V>` (5th memory pattern)
- ADR-0039 — `Namespace::parse` (audit-key reversibility)
- ADR-0041 — Postgres row-level security (mandatory)
- ADR-0042 / 0043 — `PgGraphMemory` + RLS
- ADR-0044 — `PgVectorStore` row-level security
- ADR-0058 — `WITH RECURSIVE` BFS fast path
- ADR-0063 — `add_edges_batch` UNNEST bulk insert
- ADR-0065 — `GraphMemory` trait surface shrink (admin paths
  → backend inherent)
