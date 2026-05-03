# ADR 0050 — `GraphMemory::edge` — single-id lookup completing read CRUD

**Status**: Accepted
**Date**: 2026-05-01
**Decision**: Phase 7 of the post-7-차원-audit roadmap (thirteenth sub-slice)

## Context

After ADR-0049 added paginated edge enumeration (`list_edges`
+ `list_edge_records`), the GraphMemory CRUD surface had one
remaining asymmetric gap: **`node(id)` existed but `edge(id)`
did not**. ADR-0049's "Operator usage patterns" section
explicitly noted this:

> No `edge(id)` accessor on the trait yet — operators who
> need single-edge lookup compose via `temporal_filter` on a
> tight window or `list_edge_records` over a known page.
>
> Pattern reserved for a future `edge(id)` slice.

This is that slice. Single-id lookup is the most basic read
primitive — operators that have an edge id from upstream
(audit log, error message, external system) shouldn't need to
page-scan or time-filter to find it. Closing the gap.

## Decision

Add `edge(ctx, ns, edge_id) -> Result<Option<GraphHop<E>>>`
to the `GraphMemory` trait with a default `Ok(None)` impl.

```rust
async fn edge(
    &self,
    _ctx: &ExecutionContext,
    _ns: &Namespace,
    _edge_id: &EdgeId,
) -> Result<Option<GraphHop<E>>> {
    Ok(None)
}
```

### Why `Option<GraphHop<E>>`, not `Option<E>`

`node(id)` returns `Option<N>` because the node's structural
body *is* the payload — there's nothing else to return.
`edge(id)` is different: edges have a payload AND endpoints
(`from` / `to`) AND a `timestamp`. Operators almost always
need at least one of those auxiliary fields:

- **Audit context**: "what did this edge connect, and when?"
- **Freshness check**: "is this fact stale?"
- **Neighbour navigation**: "from this edge's `to`, what's
  next?"
- **Validation**: "do this edge's endpoints still exist?"

Returning `Option<E>` (payload only) would force every caller
into a follow-up query for the structural body — N+1 in
disguise. Returning `Option<GraphHop<E>>` gives the full body
in one round-trip.

The asymmetry with `node`'s return shape (`Option<N>` vs
`Option<GraphHop<E>>`) is intentional, not an oversight —
it reflects a real difference in what each lookup needs to
serve. The trait doc is explicit:

> Asymmetric with [`Self::node`] (which returns `Option<N>`
> because nodes have no separate structural body) — the
> shape difference is intentional, not an oversight.

### Why `GraphHop<E>` reuse

Same reasoning as ADR-0049: every other edge-yielding method
returns `GraphHop<E>` (or close — `neighbors` returns
`(EdgeId, NodeId, E)` which is a hop with neighbour-only). New
tuple shapes would surprise callers. One struct, used
everywhere.

### `InMemoryGraphMemory` impl

```rust
table.read().edges.get(edge_id).map(|e| GraphHop { ... })
```

Single read-lock, single map lookup. O(log n) on the
`BTreeMap`.

### `PgGraphMemory` impl

```sql
SELECT from_node, to_node, payload, ts FROM graph_edges
WHERE namespace_key = $1 AND id = $2
```

Wrapped in the standard `pool.begin → set_tenant_session →
query → commit` envelope (ADR-0043 RLS-aware tenant tx). The
composite PK `(namespace_key, id)` makes this an indexed
lookup — same shape as `node(id)`.

### Tests

- 2 `InMemoryGraphMemory` unit tests: full-hop lookup with
  field-by-field assertion + absent-id returns None;
  cross-namespace isolation.
- 1 `PgGraphMemory` testcontainers test: full-hop lookup
  against real Postgres + cross-namespace isolation.

## Consequences

✅ GraphMemory **read CRUD is complete and symmetric**:
`node(id)` + `edge(id)` for single-id, `neighbors(id, dir)` +
`traverse(start, dir, depth)` + `find_path(from, to, dir,
depth)` + `temporal_filter(from, to)` for relational reads,
`list_nodes` + `list_node_records` + `list_edges` +
`list_edge_records` for enumeration.
✅ Operators with an edge id from upstream (audit, error,
external) get the structural body in one round-trip — no
page-scan or time-filter workaround.
✅ `GraphHop<E>` reused — uniform shape across every
edge-yielding surface in the trait.
✅ Default `Ok(None)` impl shields existing user-implemented
backends (Neo4j, ArangoDB, etc.) from breaking. Backends that
own an edge index pick up the behaviour by overriding.
❌ Trait surface grew by another method. After this slice
GraphMemory has 14 methods. The doc structure groups them
clearly (add / read / traverse / delete / enumerate); the
final shape is comprehensive but not bloated.
❌ Asymmetric return shape (`Option<N>` for nodes vs
`Option<GraphHop<E>>` for edges) — operators learning the API
have to remember the distinction. The trait doc calls it out
explicitly so the surprise is bounded.

## Alternatives considered

1. **`edge(id) -> Option<E>` (payload-only, symmetric with
   `node`)** — forces N+1 for the common case (operators
   always also need `from`/`to`/`timestamp`). Rejected — see
   "Why `Option<GraphHop<E>>`" above.
2. **`edge(id) -> Option<(NodeId, NodeId, E, DateTime<Utc>)>`
   tuple instead of struct** — three return shapes for edges
   already exist (`(EdgeId, NodeId, E)` from `neighbors`,
   `Vec<(EdgeId, NodeId, NodeId, E)>` from temporal_filter
   pre-GraphHop, `GraphHop<E>` from traverse). Adding a
   fourth would compound the surprise. Reusing `GraphHop<E>`
   matches the slice family.
3. **Skip `edge(id)`, document `list_edge_records` workaround
   as the canonical path** — the workaround is N round-trips
   on average to find an edge by id (page-scan). Wrong cost
   profile for a basic primitive.
4. **`edge(id) -> Result<Option<Edge<E>>>` with a new `Edge`
   struct distinct from `GraphHop`** — `Edge` and `GraphHop`
   would carry the same fields with different names. Pure
   bikeshedding cost. Reuse wins.

## Operator usage patterns

**Audit-log follow-up** (operator has edge id from a log
entry, wants context):
```rust
if let Some(hop) = graph.edge(&ctx, &ns, &edge_id_from_log).await? {
    tracing::info!(
        from = %hop.from, to = %hop.to,
        when = %hop.timestamp,
        "edge context for audit",
    );
}
```

**Freshness check** (operator has edge id, wants timestamp):
```rust
let stale_cutoff = Utc::now() - Duration::days(30);
if let Some(hop) = graph.edge(&ctx, &ns, &id).await? {
    if hop.timestamp < stale_cutoff {
        graph.delete_edge(&ctx, &ns, &id).await?;
    }
}
```

**Hybrid sweep enrichment** (the pattern ADR-0049 reserved for
this slice):
```rust
let mut offset = 0;
loop {
    let ids = graph.list_edges(&ctx, &ns, 1000, offset).await?;
    if ids.is_empty() { break; }
    for id in &ids {
        if needs_inspection(id) {
            // Selective deep look — single-id lookup, no scan.
            if let Some(hop) = graph.edge(&ctx, &ns, id).await? {
                inspect(&hop).await?;
            }
        }
    }
    offset += ids.len();
}
```

## References

- ADR-0007 — `GraphMemory` trait surface (parent).
- ADR-0042 — `entelix-graphmemory-pg` companion crate.
- ADR-0043 — RLS pattern (tenant-tx envelope reused here).
- ADR-0049 — `list_edges` / `list_edge_records` (the slice
  that explicitly reserved this `edge(id)` slice).
- 7-차원 roadmap §S8 — Phase 7 thirteenth sub-slice.
- `crates/entelix-memory/src/graph.rs` — trait method + default
  + `InMemoryGraphMemory` impl + 2 unit tests + Surface doc
  update.
- `crates/entelix-graphmemory-pg/src/store.rs` — `PgGraphMemory`
  impl with tenant-tx envelope.
- `crates/entelix-graphmemory-pg/tests/postgres_e2e.rs` — 1
  testcontainers regression.
