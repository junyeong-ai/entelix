# ADR 0046 — `GraphMemory::delete_edge` + `delete_node` (cascading) — CRUD completion

**Status**: Accepted
**Date**: 2026-05-01
**Decision**: Phase 7 of the post-7-차원-audit roadmap (ninth sub-slice)

## Context

`GraphMemory<N, E>` shipped with `add_node` / `add_edge` and
read methods, plus the recently-added `prune_older_than`
(ADR-0045). It had no single-id delete. Operators that needed
to remove a specific node or edge had to either:

1. Drop down to backend-specific SQL (`DELETE FROM graph_edges
   WHERE id = …` against Postgres directly), bypassing the
   trait's tenant anchor and breaking invariant 11's "every
   read/write rides a tenant anchor" contract.
2. Reach for `prune_older_than` and accept its time-based
   semantic — wrong tool for "I just want this one wrong fact
   gone".
3. Wait for orphan-node cleanup support promised in ADR-0045's
   "future slice" note — and meanwhile not be able to act on
   the orphans the prune sweep created.

CRUD on a graph isn't complete without single-id delete. This
slice closes the gap.

## Decision

Add two trait methods with default impls (`Ok(())` / `Ok(0)`)
so existing `GraphMemory` implementors compile unchanged:

```rust
async fn delete_edge(
    &self,
    _ctx: &ExecutionContext,
    _ns: &Namespace,
    _edge_id: &EdgeId,
) -> Result<()> {
    Ok(())
}

async fn delete_node(
    &self,
    _ctx: &ExecutionContext,
    _ns: &Namespace,
    _node_id: &NodeId,
) -> Result<usize> {
    Ok(0)
}
```

### `delete_edge` — idempotent, single edge

Removes one edge by id. Idempotent: deleting an absent edge
succeeds silently with `Ok(())`. The trait doc makes the
idempotency explicit so backends don't surface "edge not found"
as an error (which would force every caller into a `match` for
the no-op case).

### `delete_node` — cascading, returns removed-edge count

Removes one node by id and **cascades to every edge incident on
that node** (both outgoing and incoming). Returns the count of
removed edges so callers can log or expose cleanup metrics; `0`
when the node had no edges or was absent (idempotent on absent).

The cascade decision matters. Three options were on the table:

1. **Cascade** — drop incident edges automatically.
2. **`RESTRICT` shape** — refuse if incident edges exist; force
   operator to delete edges first.
3. **Dangling** — drop only the node, leave edges pointing at
   the now-absent endpoint.

Rejected:

- **Dangling** breaks the invariant "every edge endpoint
  resolves to a node id" that traversal relies on. A `find_path`
  result whose hop's `to` doesn't resolve via `node()` would
  silently corrupt callers. Out.
- **`RESTRICT`** forces every operator into a manual edge-delete
  loop. The two-step pattern (`for e in neighbors(...)
  delete_edge`, then `delete_node`) is mechanical and never
  what the operator wanted to think about. Out.

Cascade is the default. Operators that genuinely want
`RESTRICT` semantics check `neighbors(...).is_empty()` before
calling `delete_node` — explicit, no surprise.

### Why returning the removed-edge count matters

`delete_node` cascading happens server-side; the operator can't
predict how many edges will go. Returning the count lets
audit/log paths attribute "X edges cleaned up by this delete"
without a follow-up query. Mirrors `prune_older_than`'s
`-> Result<usize>` shape.

### `InMemoryGraphMemory` impl

Both methods take the per-namespace write lock once, mutate
in place, and clean up adjacency lists (`out_adj`, `in_adj`)
to prevent dangling edge ids in the lookup tables.

`delete_node` snapshots incident edges into a `HashSet<EdgeId>`
*before* the deletion loop. The HashSet dedups self-loops —
without it, a self-loop `a → a` appears in both `out_adj[a]`
and `in_adj[a]` and would be counted twice.

### `PgGraphMemory` impl

`delete_edge`: single `DELETE FROM graph_edges WHERE
namespace_key = $1 AND id = $2` wrapped in the
`pool.begin → set_tenant_session → query → commit` envelope
(ADR-0043 RLS-aware tenant tx).

`delete_node`: two `DELETE` statements in **one** transaction:

```sql
-- 1. Drop incident edges (returns the cascade count)
DELETE FROM graph_edges
WHERE namespace_key = $1 AND (from_node = $2 OR to_node = $2);

-- 2. Drop the node itself
DELETE FROM graph_nodes
WHERE namespace_key = $1 AND id = $2;
```

The transaction is essential — a concurrent reader must never
see a half-applied state (an edge whose endpoint node is gone,
or a node whose edges are still present). The covering indexes
on `(namespace_key, from_node)` / `(namespace_key, to_node)`
keep the cascade scan O(log n + k) where k is the cascade size.

### Tests

- `InMemoryGraphMemory` 4 unit tests: idempotent edge delete +
  adjacency dedup; cascade with surviving siblings; self-loop
  dedup count; absent node returns 0.
- `PgGraphMemory` 2 testcontainers tests mirror the in-memory
  cases against real Postgres.

## Consequences

✅ `GraphMemory` CRUD is complete: add / read / single-id
delete / TTL prune. Operators can build any composite cleanup
pattern (orphan sweep, age-bounded retention, manual fact
removal) on top of these primitives without bypassing the SDK.
✅ Cascade default makes the operator's "delete this node"
intent match what happens — no dangling edge surprises.
✅ Returning removed-edge count from `delete_node` gives audit/
log paths the cleanup attribution they need without a
follow-up query.
✅ Both impls ride the existing tenant-tx envelope (Postgres
RLS) and adjacency-lookup discipline (in-memory) — no
special-case path, defense-in-depth uniform.
✅ Default impls (`Ok(())` / `Ok(0)`) shield existing
user-implemented backends (Neo4j, ArangoDB, etc.) from
breaking. Backends that override pick up the behaviour;
others stay no-op.
❌ Trait surface grew by two methods. Public-API baselines
refreshed for `entelix-memory` + `entelix-graphmemory-pg`.
❌ Cascade is the default; operators that genuinely want
`RESTRICT` semantics (refuse delete when incident edges exist)
check `neighbors(...).is_empty()` first. Document the pattern
in the trait doc string.
❌ `PgGraphMemory::delete_node` does two round-trips inside
its tx (one DELETE for edges, one for the node). A single
`DELETE FROM graph_nodes ... RETURNING id` plus a separate
edge sweep would be the same number of round-trips; a CTE
chaining both into one statement could be a future
optimization slice.

## Alternatives considered

1. **`RESTRICT` semantics by default** — forces a manual
   edge-delete loop on every caller. Wrong default for the
   common case ("delete this node and clean up after it");
   advanced callers check `neighbors(...).is_empty()` first.
   Rejected.
2. **Cascade returns `Vec<EdgeId>` instead of `usize`** —
   heavier surface, useful only for audit replays that should
   be logging the delete intent themselves. Count matches
   sibling shape (`prune_older_than`).
3. **Soft-delete (mark as deleted, garbage-collect later)** —
   adds a `deleted_at` column, schema migration, every read has
   to filter. Heavyweight; reserved for backends that genuinely
   need replay semantics. Out of scope.
4. **One `delete(ctx, ns, GraphTarget)` enum-shaped method**
   — `GraphTarget::{Edge(EdgeId), Node(NodeId)}`. Saves one
   trait method but loses the typed return distinction
   (`Result<()>` vs `Result<usize>`). Two methods read more
   plainly at the call site.

## Operator usage patterns

**Manual fact removal** (single edge):
```rust
graph.delete_edge(&ctx, &ns, &wrong_edge_id).await?;
```

**Drop a node and clean up** (cascade):
```rust
let cleaned = graph.delete_node(&ctx, &ns, &node_id).await?;
tracing::info!(removed_edges = cleaned, "node deleted");
```

**Orphan-node sweep after `prune_older_than`** (manual loop):
```rust
let removed_edges = graph.prune_older_than(&ctx, &ns, ttl).await?;
// Find orphans: nodes with no remaining edges. Operator-side
// because the trait doesn't expose `list_nodes` (yet).
// For each candidate orphan:
graph.delete_node(&ctx, &ns, &orphan_id).await?;
```

**`RESTRICT`-style "fail if edges exist"**:
```rust
if !graph.neighbors(&ctx, &ns, &node_id, Direction::Both)
    .await?.is_empty() {
    return Err(MyError::NodeHasEdges(node_id.clone()));
}
graph.delete_node(&ctx, &ns, &node_id).await?;
```

## References

- ADR-0007 — `GraphMemory` trait surface (parent).
- ADR-0042 — `entelix-graphmemory-pg` companion crate.
- ADR-0043 — RLS pattern (the tenant-tx envelope this slice
  reuses for `PgGraphMemory`).
- ADR-0045 — `prune_older_than` (the slice this builds on,
  mentioned operator orphan cleanup as future work).
- 7-차원 roadmap §S8 — Phase 7 ninth sub-slice.
- `crates/entelix-memory/src/graph.rs` — trait methods + default
  impls + `InMemoryGraphMemory` impl + 4 unit tests.
- `crates/entelix-graphmemory-pg/src/store.rs` — `PgGraphMemory`
  impl with single-tx cascade.
- `crates/entelix-graphmemory-pg/tests/postgres_e2e.rs` —
  2 testcontainers regressions.
