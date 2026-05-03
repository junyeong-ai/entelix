# ADR 0047 — `GraphMemory::list_nodes` — paginated enumeration primitive

**Status**: Superseded by ADR-0065 (trait-method portion moved to `PgGraphMemory` inherent — `GraphMemory` trait stays at the agent-facing surface)
**Date**: 2026-05-01
**Decision**: Phase 7 of the post-7-차원-audit roadmap (tenth sub-slice)

## Context

After ADR-0046 added `delete_edge` + `delete_node`, the
`GraphMemory` CRUD surface was complete *except* for one
missing primitive: enumeration. Operators had no way to walk
the node set under a namespace through the trait. The orphan-
sweep pattern documented in ADR-0046 §"Operator usage patterns"
read:

```rust
// Find orphans: nodes with no remaining edges. Operator-side
// because the trait doesn't expose `list_nodes` (yet).
// For each candidate orphan:
graph.delete_node(&ctx, &ns, &orphan_id).await?;
```

The "(yet)" was the gap. Operators that wanted orphan cleanup
had to either:

1. Track every minted `NodeId` in their own external store —
   defeating the SDK's "we hold the structural identity for
   you" contract.
2. Drop down to backend SQL (`SELECT id FROM graph_nodes
   WHERE namespace_key = …`) — bypassing the tenant anchor and
   breaking invariant #11.

Both paths failed. Enumeration is the missing link.

The pattern is well-established for sibling memory shapes:

- `Store::list(ns, prefix) -> Vec<String>` — keys, no values.
- `EpisodicMemory::all(ns) -> Vec<Episode<V>>` — full payload
  (because episodes are typically read in bulk).
- `EntityMemory::all_records(ns) -> HashMap<...>` — full record.

`GraphMemory` was the odd one out. Closing the gap.

## Decision

Add `list_nodes` to the `GraphMemory` trait with a default
empty-vec impl so existing implementors compile unchanged:

```rust
async fn list_nodes(
    &self,
    _ctx: &ExecutionContext,
    _ns: &Namespace,
    _limit: usize,
    _offset: usize,
) -> Result<Vec<NodeId>> {
    Ok(Vec::new())
}
```

### Ids-only return shape

Returns `Vec<NodeId>`, not `Vec<(NodeId, N)>`. Two reasons:

1. **Matches `Store::list`**, the closest sibling primitive.
   Operators build a uniform mental model across memory shapes
   without remembering which one is which.
2. **Cheaper hot path.** The orphan-sweep use case wants ids
   only — fetching JSONB payloads for every node when the
   operator just wants to compute `neighbors().is_empty()` is
   wasteful. Audit / inspection use cases call `node(id)` per
   id, accepting the N+1 cost where they actually need the
   payload.

The trait doc makes the trade-off explicit:

> Ids-only by design — matches [`crate::Store::list`].
> Operators that need the payload too call [`Self::node`] per
> id, accepting the N+1 cost for the inspection use case.

A future `list_node_records(ns, limit, offset) -> Vec<(NodeId, N)>`
addition (separate slice) could serve the inspection case
without an N+1 hit if there's demand.

### `(limit, offset)` pagination, not cursor-based

`limit` + `offset` is the conventional shape for SQL-backed
backends and matches `Store::list` siblings. Cursor-based
pagination (next-token) is more efficient at scale but requires
each backend to mint stable cursors — an over-design for the
current crate set. Operators with very large graphs page through
in chunks (`offset += limit` each iteration); the
`(namespace_key, id)` PK gives `OFFSET` an indexed scan.

### Sort order: `NodeId` ascending

`UUID v7` sorts by mint time, so the natural ascending order
doubles as creation-time chronology for ids minted via
`NodeId::new`. Operators that import external ids
(`NodeId::from_string`) get lexicographic order — predictable,
not incidentally tied to insertion order.

`InMemoryGraphMemory` already stores nodes in `BTreeMap<NodeId, N>`,
so iteration yields ascending order without a sort step.
`PgGraphMemory` uses `ORDER BY id ASC` (the composite PK
`(namespace_key, id)` is already indexed for this).

### `InMemoryGraphMemory` impl

Read-side lock on the per-namespace table, then
`guard.nodes.keys().skip(offset).take(limit).cloned().collect()`.
Read lock is shared, so concurrent readers don't serialise.

### `PgGraphMemory` impl

```sql
SELECT id FROM graph_nodes
WHERE namespace_key = $1
ORDER BY id ASC
LIMIT $2 OFFSET $3
```

Wrapped in the standard `pool.begin → set_tenant_session →
query → commit` envelope (ADR-0043 RLS-aware tenant tx). The
composite PK gives an indexed scan; OFFSET is O(offset+limit)
in the worst case (Postgres must skip rows), so operators
paging deep into very large namespaces should consider a
keyset-pagination alternative — out of scope for this slice
but a potential future optimization.

### Tests

- `InMemoryGraphMemory` 3 unit tests: pagination in id order,
  namespace isolation, empty namespace.
- `PgGraphMemory` 1 testcontainers test: pagination + cross-
  namespace isolation against real Postgres.

## Consequences

✅ `GraphMemory` enumeration surface is complete. Orphan sweep,
audit walk, batch processing all build on top without
per-operator workarounds.
✅ Ids-only matches `Store::list`. Mental model uniform across
memory shapes. Cheaper hot path for the common case.
✅ The `(namespace_key, id)` PK index serves both `node()`
single-id lookup and `list_nodes()` ranged scan — no schema
change, no new index.
✅ Default empty-vec impl shields existing user-implemented
backends (Neo4j, ArangoDB, etc.) from breaking. Backends that
own a node index pick up the behaviour by overriding.
❌ Trait surface grew by one method. Public-API baselines
refreshed for `entelix-memory` + `entelix-graphmemory-pg`.
❌ `OFFSET` pagination cost grows with depth — `O(offset+limit)`
in Postgres. Operators paging tens of thousands of nodes deep
should switch to keyset pagination (out of scope).
❌ Ids-only means inspection-style consumers pay an N+1 cost
when they want payloads. Documented; a `list_node_records`
addition is reserved for a future slice.

## Alternatives considered

1. **`list_nodes -> Vec<(NodeId, N)>`** — hot-path waste for
   the orphan-sweep use case. Inspection consumers call
   `node(id)` per id. Rejected for the default; reserved as a
   future `list_node_records` companion.
2. **Cursor-based pagination** — over-design for current crate
   set. Mints stable cursors per backend, complicates the
   trait. Rejected.
3. **`list_nodes` returning `Iterator` (lazy / streaming)** —
   async iterators in Rust are still finicky (`Stream` trait,
   ownership over the lock guard). The `Vec` shape is what
   `Store::list` returns; consistency wins.
4. **Add `list_edges(ns, limit, offset)` for symmetry** —
   `temporal_filter(ns, t0, t_max)` already enumerates edges.
   Adding a second enumeration primitive would duplicate
   surface for a marginal benefit. Out of scope.
5. **`list_nodes` with `Option<NodeIdPrefix>` filter** —
   premature; orphan sweep doesn't need it, and operators who
   want id-prefix filtering can post-filter on the client. The
   trait stays minimal.

## Operator usage patterns

**Orphan sweep** (the slice's primary motivation):
```rust
let mut offset = 0;
let page_size = 1000;
let mut total_orphans = 0;
loop {
    let nodes = graph.list_nodes(&ctx, &ns, page_size, offset).await?;
    if nodes.is_empty() {
        break;
    }
    for node_id in &nodes {
        if graph.neighbors(&ctx, &ns, node_id, Direction::Both)
            .await?.is_empty()
        {
            graph.delete_node(&ctx, &ns, node_id).await?;
            total_orphans += 1;
        }
    }
    offset += page_size;
}
tracing::info!(orphans = total_orphans, "sweep complete");
```

**Audit walk** (needs payloads):
```rust
let nodes = graph.list_nodes(&ctx, &ns, 100, 0).await?;
for id in nodes {
    if let Some(payload) = graph.node(&ctx, &ns, &id).await? {
        // inspect / log / export
    }
}
```

**Bulk migration** (export every node):
```rust
let mut offset = 0;
loop {
    let page = graph.list_nodes(&ctx, &ns, 500, offset).await?;
    if page.is_empty() { break; }
    for id in &page {
        if let Some(payload) = graph.node(&ctx, &ns, id).await? {
            export(id, payload).await?;
        }
    }
    offset += page.len();
}
```

## References

- ADR-0007 — `GraphMemory` trait surface (parent).
- ADR-0042 — `entelix-graphmemory-pg` companion crate.
- ADR-0043 — RLS pattern (tenant-tx envelope reused here).
- ADR-0045 — `prune_older_than` (the slice that pointed at
  orphan cleanup as future work).
- ADR-0046 — `delete_edge` + `delete_node` (the previous slice
  that explicitly noted `list_nodes` as the missing piece).
- 7-차원 roadmap §S8 — Phase 7 tenth sub-slice.
- `crates/entelix-memory/src/graph.rs` — trait method + default
  + `InMemoryGraphMemory` impl + 3 unit tests.
- `crates/entelix-graphmemory-pg/src/store.rs` — `PgGraphMemory`
  impl with tenant-tx envelope.
- `crates/entelix-graphmemory-pg/tests/postgres_e2e.rs` — 1
  testcontainers regression.
