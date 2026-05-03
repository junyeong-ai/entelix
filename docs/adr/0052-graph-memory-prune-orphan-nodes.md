# ADR 0052 — `GraphMemory::prune_orphan_nodes` — second-phase TTL cleanup

**Status**: Superseded by ADR-0065 (trait-method portion moved to `PgGraphMemory` inherent — `GraphMemory` trait stays at the agent-facing surface)
**Date**: 2026-05-01
**Decision**: Phase 7 of the post-7-차원-audit roadmap (fifteenth sub-slice)

## Context

ADR-0045 introduced `prune_older_than` as **edge-only by
design** — nodes have no timestamp on the trait surface, so a
TTL sweep can't reason about them directly. The doc explicitly
left orphan-node cleanup to the operator:

> Edge-only by design — nodes left orphaned by edge removal
> stay in place until the operator drops them explicitly via
> a future operation.

ADR-0046 + ADR-0047 (delete primitives + list_nodes) gave
operators the building blocks for orphan cleanup, but
composing them was N+1:

```rust
let mut offset = 0;
let mut total_orphans = 0;
loop {
    let nodes = graph.list_nodes(&ctx, &ns, 1000, offset).await?;
    if nodes.is_empty() { break; }
    for node_id in &nodes {
        if graph.neighbors(&ctx, &ns, node_id, Direction::Both)
            .await?.is_empty()
        {
            graph.delete_node(&ctx, &ns, node_id).await?;
            total_orphans += 1;
        }
    }
    offset += 1000;
}
```

That's `O(N + M)` round-trips for `N` nodes and `M` orphans.
For Postgres backends, a single SQL anti-join would be `O(1)`
round-trips with index-only scan cost. The pattern was
documented but the optimisation wasn't shipped.

This slice closes that gap.

## Decision

Add `prune_orphan_nodes(ctx, ns) -> Result<usize>` to the
`GraphMemory` trait with a default `Ok(0)` impl. Drops every
node in `ns` that has zero incident edges (both directions);
returns the removed-node count.

```rust
async fn prune_orphan_nodes(
    &self,
    _ctx: &ExecutionContext,
    _ns: &Namespace,
) -> Result<usize> {
    Ok(0)
}
```

### Why default `Ok(0)`, not the composition

The naïve composition (`list_nodes` → `neighbors` per id →
`delete_node`) is `O(N)` round-trips. Putting it as the
default impl would silently encourage that cost on backends
without an override — surprising operators who reach for the
trait expecting a single-roundtrip primitive.

`Ok(0)` is the safe fallback. Backends that implement the
optimised single-query path (Postgres anti-join, in-memory
adjacency check) override; backends without support return
`0`, signalling to the operator that they should compose the
loop themselves explicitly if they need orphan cleanup.

The doc string makes this trade-off explicit:

> The naïve composition (`list_nodes` → `neighbors` per id →
> `delete_node`) is O(N) round-trips and would surprise
> operators reaching for the trait — backends override with
> a server-side single-query implementation.

### `InMemoryGraphMemory` impl

Single write-lock acquisition. Snapshot orphan ids by walking
`guard.nodes.keys()` and checking each against `out_adj` /
`in_adj` (an orphan has neither entry, or both empty). Then
delete each — `BTreeMap` `O(log n)` per delete.

```rust
let orphans: Vec<NodeId> = guard.nodes.keys()
    .filter(|id| {
        guard.out_adj.get(id).is_none_or(Vec::is_empty)
        && guard.in_adj.get(id).is_none_or(Vec::is_empty)
    })
    .cloned()
    .collect();
```

The snapshot-then-mutate pattern keeps the borrow checker
happy (immutable iter while building the orphans list, then
mutable deletes after).

### `PgGraphMemory` impl

Single SQL DELETE with anti-join:

```sql
DELETE FROM graph_nodes
WHERE namespace_key = $1
  AND id NOT IN (
      SELECT from_node FROM graph_edges WHERE namespace_key = $1
      UNION
      SELECT to_node FROM graph_edges WHERE namespace_key = $1
  )
```

Wrapped in the standard `pool.begin → set_tenant_session →
query → commit` envelope (ADR-0043 RLS-aware tenant tx).

The `(namespace_key, from_node)` and `(namespace_key, to_node)`
covering indexes installed by the bootstrap (ADR-0042) make
the inner SELECTs index-only scans. The UNION dedups
endpoints (a node connected by N edges still appears once);
the outer NOT IN filters orphans.

`rows_affected` clamps to `usize` with `usize::MAX` saturation
— same defensive pattern as `prune_older_than`.

### Tests

- 3 `InMemoryGraphMemory` unit tests:
  - Drops zero-edge nodes, leaves connected ones intact.
  - **Two-phase prune**: `prune_older_than` first leaves both
    endpoints orphaned, `prune_orphan_nodes` cleans them up.
  - Empty namespace → `Ok(0)`.
- 1 `PgGraphMemory` testcontainers test mirrors the
  zero-edge-node case against real Postgres.

## Consequences

✅ The two-phase TTL prune (`prune_older_than` →
`prune_orphan_nodes`) is now a clean two-call pattern with
single-round-trip cost on each call. No N+1 workaround.
✅ Operators wanting the orphan cleanup explicitly write the
two calls in sequence; operators that don't want it skip the
second call. Cascade behaviour stays opt-in (ADR-0045 design
preserved).
✅ Postgres impl uses anti-join over covering indexes —
production-ready cost profile.
✅ Default `Ok(0)` shields existing user-implemented backends
from breaking. Backends that own adjacency tracking pick up
the behaviour by overriding.
❌ Trait surface grew by another method. After this slice
GraphMemory has 17 methods. The doc structure groups them
clearly (add / read / count / enumerate / delete / prune); the
surface is comprehensive but well-organised.
❌ Default `Ok(0)` silently lies for backends without an
override — operators see "0 orphans removed" and may
incorrectly believe the namespace had no orphans. The doc
warns this; backends are expected to override. The pattern
matches `node_count` / `edge_count` (ADR-0051) where the
default impl returns 0 for the same reason.

## Alternatives considered

1. **Default impl that composes `list_nodes` + `neighbors` +
   `delete_node`** — gives correct behaviour everywhere but
   silently incurs `O(N)` round-trips on backends without
   override. Surprising. The current `Ok(0)` default is honest:
   "your backend doesn't optimise this; compose explicitly if
   you need it".
2. **Cascade prune that drops orphans inside
   `prune_older_than`** — couples two-axis cleanup (time-based
   edge sweep + structural orphan check) into one method.
   ADR-0045 explicitly rejected this for surprise-data-loss
   reasons (active inserts racing the sweep). Two methods
   stay separate.
3. **`prune_orphan_nodes(ctx, ns, max: usize) -> Result<Vec<NodeId>>`
   returning the deleted ids** — more information, but the
   common use case is "tell me how many" for logging. Sibling
   `prune_older_than` returns `Result<usize>`; consistency
   wins.
4. **Postgres impl using `LEFT JOIN ... WHERE edges.id IS NULL`
   instead of `NOT IN (UNION)`** — equivalent semantics, may
   plan slightly differently. Both indexed scans on the
   covering indexes; the `NOT IN (UNION)` form reads more
   plainly as "drop nodes not appearing as either endpoint".
   Either works; sticking with the more explicit form.

## Operator usage patterns

**Two-phase TTL prune** (the canonical sweep loop):
```rust
let edges_removed = graph.prune_older_than(&ctx, &ns, ttl).await?;
let nodes_removed = graph.prune_orphan_nodes(&ctx, &ns).await?;
tracing::info!(
    edges_removed,
    nodes_removed,
    "two-phase prune complete",
);
```

**Orphan-only sweep** (without TTL):
```rust
// Some operator pipeline produces structural updates that
// might leave nodes disconnected. Periodic orphan cleanup
// without time-based eviction.
let cleaned = graph.prune_orphan_nodes(&ctx, &ns).await?;
if cleaned > 0 {
    metrics::counter!("graph.orphans.swept").increment(cleaned as u64);
}
```

**Maintenance loop fast-fail** (combine with counts):
```rust
if graph.node_count(&ctx, &ns).await? == 0 {
    continue; // empty namespace, nothing to prune
}
let cleaned = graph.prune_orphan_nodes(&ctx, &ns).await?;
// log or metric
```

## References

- ADR-0007 — `GraphMemory` trait surface (parent).
- ADR-0042 — `entelix-graphmemory-pg` companion crate.
- ADR-0043 — RLS pattern (tenant-tx envelope reused here).
- ADR-0045 — `prune_older_than` (the slice that explicitly
  reserved this orphan-cleanup follow-up).
- ADR-0046 — `delete_edge` / `delete_node` (the building
  blocks operators previously composed manually).
- ADR-0047 — `list_nodes` (the enumeration primitive that
  enabled the manual composition).
- 7-차원 roadmap §S8 — Phase 7 fifteenth sub-slice.
- `crates/entelix-memory/src/graph.rs` — trait method + default
  + `InMemoryGraphMemory` impl + 3 unit tests.
- `crates/entelix-graphmemory-pg/src/store.rs` — `PgGraphMemory`
  impl with single-SQL anti-join.
- `crates/entelix-graphmemory-pg/tests/postgres_e2e.rs` — 1
  testcontainers regression.
