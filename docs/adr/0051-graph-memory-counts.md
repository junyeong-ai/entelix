# ADR 0051 — `GraphMemory::node_count` + `edge_count` — operator metric primitives

**Status**: Accepted
**Date**: 2026-05-01
**Decision**: Phase 7 of the post-7-차원-audit roadmap (fourteenth sub-slice)

## Context

After ADR-0050 closed the read-CRUD asymmetry, GraphMemory's
surface had every per-row primitive but lacked **size
metrics**. Operators making size-based decisions —
"should I paginate or stream?", "is this namespace empty
enough to skip the maintenance loop?", "what's the dashboard
count for this tenant's knowledge graph?" — had to either:

1. Loop over `list_nodes` / `list_edges` paginated, accumulating
   the count — N round-trips for N nodes.
2. Drop down to backend SQL (`SELECT COUNT(*)`), bypassing the
   tenant anchor and breaking invariant 11.

Both wrong tools. Counts are a different shape from enumeration
— cheap operator metrics that backends serve with one indexed
aggregate query.

The pattern is established for sibling memory traits
(`Store::list` is for keys, `Store::evict_expired` is for
deletes — but no sibling has `Store::count` either; the
omission is shared). For a graph backend, where operators
genuinely need size visibility for paging strategy decisions,
the omission was real.

## Decision

Add two trait methods with default `Ok(0)` so existing
implementors compile unchanged:

```rust
async fn node_count(&self, _ctx: &ExecutionContext, _ns: &Namespace) -> Result<usize> {
    Ok(0)
}

async fn edge_count(&self, _ctx: &ExecutionContext, _ns: &Namespace) -> Result<usize> {
    Ok(0)
}
```

### Why `usize`, not `Option<usize>`

Returning `Option<usize>` would force every caller to handle
"backend doesn't support counts" as a branch. The default impl
returns `Ok(0)` — operators querying a backend without count
support get a falsy result they can guard on (`count == 0`),
which collapses to the empty-namespace path that's also valid
for actually-empty namespaces.

The alternative of returning a sentinel like `usize::MAX` is
worse — it lies about size and silently breaks size-based
decisions.

### Two methods, not one

`node_count` + `edge_count` separately, not a single
`count_summary -> (usize, usize)`. Three reasons:

1. **One use case at a time**. Most callers want either nodes
   or edges, not both — paying for the second query is waste.
2. **Indexed-scan symmetry**. Each table has its own
   `(namespace_key, _)` PK; one COUNT per table is the natural
   shape.
3. **Sibling pattern**. The enumeration methods come in pairs
   (`list_nodes` / `list_edges`) — counts pair the same way.

### `InMemoryGraphMemory` impl

Read-side lock; `guard.nodes.len()` / `guard.edges.len()`.
O(1) on the `BTreeMap` (length is tracked).

### `PgGraphMemory` impl

```sql
SELECT COUNT(*) FROM graph_nodes WHERE namespace_key = $1
```

Wrapped in the standard `pool.begin → set_tenant_session →
query → commit` envelope (ADR-0043 RLS-aware tenant tx). The
composite PK `(namespace_key, id)` makes this an
index-only scan — `COUNT(*)` over a covering index without
touching the heap.

The `i64 → usize` conversion uses `try_from` with `usize::MAX`
saturation — same defensive pattern as `prune_older_than`'s
`rows_affected` clamp.

### Tests

- 2 `InMemoryGraphMemory` unit tests: insert tracking
  (empty → +nodes → +edges), namespace isolation.
- 1 `PgGraphMemory` testcontainers test: insert tracking
  against real Postgres + cross-namespace isolation.

## Consequences

✅ Operators paging through large namespaces decide
upfront whether to stream or fully enumerate. No N+1 size
discovery.
✅ Dashboard / monitoring surfaces have first-class access to
graph size via the trait — no bypass-the-SDK required.
✅ Empty-namespace fast-fail patterns (skip maintenance loops
when count == 0) become trivial.
✅ Postgres impl uses index-only scan — `COUNT(*)` over the
composite PK doesn't read the heap.
✅ Default `Ok(0)` impl shields existing user-implemented
backends from breaking. Backends that own size tracking pick
up the behaviour by overriding.
❌ Trait surface grew by another two methods. After this slice
GraphMemory has 16 methods. The doc structure groups them
clearly (add / read / count / enumerate / delete); the surface
is comprehensive but not bloated.
❌ Default `Ok(0)` lies for backends that don't override —
operators using a no-count backend see "0" and may incorrectly
believe the namespace is empty. The doc string explicitly
warns this and recommends backends override when the
underlying store can serve the query cheaply.

## Alternatives considered

1. **`count_summary -> (usize, usize)` single method** —
   forces both queries even when caller wants one. Wrong cost
   shape. Rejected.
2. **`Option<usize>` to distinguish "unsupported" vs "empty"** —
   adds caller branch for a marginal benefit. The
   `count == 0` collapsed path serves both correctly for the
   empty-skip use case. Rejected.
3. **`stats() -> GraphStats { nodes, edges, ... }` extensible
   struct** — over-design until there's a third metric. Two
   methods stay simple; struct can land later if more metrics
   appear (e.g., `oldest_edge_ts`, `unique_payload_kinds`).
4. **Accept the N+1 paginated-count workaround as the
   canonical pattern** — wrong cost profile, encourages
   bypassing the SDK. Rejected.

## Operator usage patterns

**Paging strategy decision**:
```rust
let n = graph.node_count(&ctx, &ns).await?;
if n == 0 {
    return Ok(()); // empty namespace, nothing to do
}
let page_size = if n > 10_000 { 100 } else { n };
// page through with the chosen size
```

**Dashboard surface**:
```rust
let (nodes, edges) = tokio::join!(
    graph.node_count(&ctx, &ns),
    graph.edge_count(&ctx, &ns),
);
println!("namespace {}: {} nodes / {} edges", ns.render(), nodes?, edges?);
```

**Maintenance loop fast-fail**:
```rust
for ns in tenant_namespaces {
    if graph.edge_count(&ctx, &ns).await? == 0 {
        continue; // no edges → no prune work
    }
    let removed = graph.prune_older_than(&ctx, &ns, ttl).await?;
    tracing::info!(ns = %ns.render(), removed, "prune sweep");
}
```

## References

- ADR-0007 — `GraphMemory` trait surface (parent).
- ADR-0042 — `entelix-graphmemory-pg` companion crate.
- ADR-0043 — RLS pattern (tenant-tx envelope reused here).
- ADR-0047 — `list_nodes` (the slice that established the
  paginated-enumeration pattern this complements).
- ADR-0050 — `edge` single-id lookup (the prior slice that
  closed read-CRUD asymmetry).
- 7-차원 roadmap §S8 — Phase 7 fourteenth sub-slice.
- `crates/entelix-memory/src/graph.rs` — trait methods + default
  + `InMemoryGraphMemory` impls + 2 unit tests.
- `crates/entelix-graphmemory-pg/src/store.rs` — `PgGraphMemory`
  impls with tenant-tx envelope.
- `crates/entelix-graphmemory-pg/tests/postgres_e2e.rs` — 1
  testcontainers regression.
