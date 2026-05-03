# ADR 0038 — `EpisodicMemory<V>` — fifth memory pattern

**Status**: Accepted
**Date**: 2026-05-01
**Decision**: Phase 7 of the post-7-차원-audit roadmap (first slice)

## Context

`entelix-memory` shipped four `LangChain`-style memory patterns over
the `Store<V>` trait:

| Pattern | Question it answers |
|---|---|
| `BufferMemory` | "what messages have we exchanged so far?" |
| `SummaryMemory` | "what's the running summary of this thread?" |
| `EntityMemory` | "what is the current fact about entity X?" |
| `SemanticMemory<E, V>` | "what stored content resembles this query?" |

None of those answers a *temporal* question. Sessions that need to
revisit "what happened in this thread between Tuesday and
Friday?", "what were the last five things this agent did?", or
"replay the decision log from yesterday's incident" had no
1급 surface — operators were free to roll their own append-only
list under a `Store<Vec<Foo>>`, but every roll-your-own variant
re-derived the same shape (id + timestamp + payload) and the same
operations (append, range, prune) without sharing tests, semantics,
or audit story.

The 7-차원 audit's Phase 7 roadmap entry called this out as the
first sub-slice of memory expansion. The roadmap also mentioned
`Namespace::parse`, real `list_namespaces`, `entelix-graphmemory-pg`,
Postgres RLS, and per-field `Annotated<T,R>`. Each of those is its
own slice; episodic-memory is the cleanest atomic wedge — no new
crate, no new schema, no proc-macro design space.

## Decision

Add `EpisodicMemory<V>` to `entelix-memory`. Surface is parallel
to `EntityMemory` so adoption matches what operators already know:
single-key vector under one `Store<Vec<Episode<V>>>` per namespace,
atomic per-thread, every existing `Store` backend works unchanged.

### Public surface

```rust
// crates/entelix-memory/src/episodic.rs
pub struct EpisodeId(uuid::Uuid);  // UUID v7 — time-ordered
pub struct Episode<V> {
    pub id: EpisodeId,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub payload: V,
}
pub struct EpisodicMemory<V>
where V: Clone + Serialize + DeserializeOwned + Send + Sync + 'static;

impl<V> EpisodicMemory<V> {
    pub fn new(store: Arc<dyn Store<Vec<Episode<V>>>>, namespace: Namespace) -> Self;
    pub fn namespace(&self) -> &Namespace;

    pub async fn append(&self, ctx, payload: V) -> Result<EpisodeId>;
    pub async fn append_at(&self, ctx, payload: V, ts: DateTime<Utc>) -> Result<EpisodeId>;
    pub async fn append_record(&self, ctx, episode: Episode<V>) -> Result<()>;

    pub async fn all(&self, ctx) -> Result<Vec<Episode<V>>>;
    pub async fn recent(&self, ctx, n: usize) -> Result<Vec<Episode<V>>>;
    pub async fn range(&self, ctx, start, end: DateTime<Utc>) -> Result<Vec<Episode<V>>>;
    pub async fn since(&self, ctx, start: DateTime<Utc>) -> Result<Vec<Episode<V>>>;

    pub async fn count(&self, ctx) -> Result<usize>;
    pub async fn prune_older_than(&self, ctx, ttl: Duration) -> Result<usize>;
    pub async fn clear(&self, ctx) -> Result<()>;
}
```

### Three design choices worth pinning

1. **UUID v7 for `EpisodeId`.** Time-ordered ids let downstream
   audit / external systems sort by id without an extra sequence
   column, and align with `CheckpointId::to_hyphenated_string` so
   id surfaces read uniformly across the SDK. `EpisodeId::new()`
   on every append; `EpisodeId::from_uuid` for backends decoding
   stored rows.
2. **Single-key `Vec<Episode<V>>` per namespace.** Mirrors
   `EntityMemory`'s atomic single-key shape — read-modify-write is
   atomic per-thread on every `Store` backend without per-row
   schema work. Companion crates that need per-row indexing for
   very long histories (10⁵+ episodes per namespace) ship a
   dedicated backend without changing this surface. The trade-off
   is explicit in the module docs.
3. **`append_at` binary-inserts to preserve order.** Backfill from
   external ledgers / replay flows can mint episodes with
   historical timestamps; the store stays sorted so `range` /
   `since` keep their `partition_point`-shaped O(log n) cost. The
   `append` (now-timestamped) fast path skips the search since
   `Utc::now() ≥` the stored tail.

### `range` semantics

`[start, end]` inclusive on both ends. `start > end` returns
`Ok(vec![])` rather than erroring — the question "what happened
between two timestamps?" is well-defined even when the answer is
empty, and the alternative (returning `Err(InvalidRequest)`) would
force every caller to add a guard before a check that's
information-cheap to express in the result.

## Consequences

✅ Operators with temporal memory needs (conversation episodes,
task-completion records, decision logs, incident replay) no
longer hand-roll `Store<Vec<Foo>>`. The episode shape, ordering
invariant, range semantics, and TTL-prune behaviour are shared
through one tested implementation.
✅ Every existing `Store<V>` backend (`InMemoryStore`, Postgres,
Redis) gains episodic-memory support transparently — no schema
migration needed.
✅ `EpisodeId` reuses the UUID v7 + hyphenated-string convention
already established by `CheckpointId`, keeping audit-channel id
surfaces uniform.
✅ The five memory patterns now answer the four `LangChain`
parity questions plus the temporal axis Anthropic-style managed
agents need.
❌ Single-key vector shape caps practical history at the
read-modify-write throughput the underlying `Store` can sustain.
A namespace with 10⁵+ episodes will see proportional latency on
every append. Documented in the module-level docs; companion
crates ship per-row backends when the trade-off isn't acceptable.
❌ `Episode<V>` cannot derive `Eq` because `V` is operator-shaped.
Tests compare on `payload` / `id` rather than on `Episode` directly,
which is the natural shape anyway (two episodes with the same
payload but different ids should not be considered equal at the
record level).

## Alternatives considered

1. **Per-row storage shape (every episode is its own `Store` key)**
   — scales to long histories but breaks atomic prune (the
   `EntityMemory::prune_older_than` invariant), needs backend
   support for prefix scans not all `Store<V>` impls have, and
   forces every `Store` backend to grow a "keys-by-prefix"
   surface. Rejected as the default; companion crates can ship it
   when the size trade-off matters.
2. **`SortedSet<Episode<V>>` with custom `Ord`** — would need
   `V: Ord` which the operator can't always satisfy, and Rust's
   `BTreeSet` doesn't support range-by-key-projection cleanly.
   Rejected; `Vec` + `partition_point` reads as plainly without
   the bound.
3. **Make `EpisodeId` a `String` for backend friendliness** —
   `CheckpointId` is `Uuid`, and operators reading both surfaces
   benefit from one convention. Rejected; the wire-friendly
   rendering is `to_hyphenated_string()` for both.
4. **Allow `start > end` to error in `range`** — see "range
   semantics" above; well-defined empty result beats forcing
   every caller to guard.
5. **`Annotated<Vec<Episode<V>>, Append<Episode<V>>>` instead of
   a dedicated facade** — Annotated is a state-graph reducer
   helper, not a memory pattern. They sit at different layers:
   reducer composes nodes inside one super-step, memory composes
   threads across runs. Rejected; the audience is different.

## References

- ADR-0007 — memory trait surface (parent).
- ADR-0008 — companion crate pattern (where per-row backends
  would land).
- 7-차원 roadmap §S8 — Phase 7 first sub-slice.
- `crates/entelix-memory/src/episodic.rs` — surface + tests.
