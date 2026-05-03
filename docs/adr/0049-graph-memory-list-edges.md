# ADR 0049 — `GraphMemory::list_edges` + `list_edge_records` — symmetric edge enumeration

**Status**: Superseded by ADR-0065 (trait-method portion moved to `PgGraphMemory` inherent — `GraphMemory` trait stays at the agent-facing surface)
**Date**: 2026-05-01
**Decision**: Phase 7 of the post-7-차원-audit roadmap (twelfth sub-slice)

## Context

ADR-0047 + ADR-0048 added paginated node enumeration
(`list_nodes` + `list_node_records`). Edges were the asymmetric
gap: `temporal_filter(ns, from, to)` enumerated edges but only
within a time window, and there was no general "list every
edge in this namespace, paginated" primitive.

Operators wanting bulk edge inspection — schema validation,
export, or migration — had two unappealing paths:

1. Call `temporal_filter(ns, MIN_DATE, MAX_DATE)` with no
   pagination, materializing every edge in memory at once.
2. Loop over `list_nodes` and call `neighbors` per node —
   N+1 round-trips, plus the dedup hassle of edges incident
   on multiple visited nodes.

Both bad. Pagination + ids-only/full-record duality is the
established pattern for `list_nodes` / `list_node_records`;
edges deserve the same.

## Decision

Add two trait methods symmetric with the node enumeration pair.
Both default to `Ok(vec![])` so existing implementors compile
unchanged.

```rust
/// Ids-only — cheapest per-row cost. Use when the loop body
/// doesn't read payload/from/to/timestamp.
async fn list_edges(
    &self,
    _ctx: &ExecutionContext,
    _ns: &Namespace,
    _limit: usize,
    _offset: usize,
) -> Result<Vec<EdgeId>>;

/// Full record (`GraphHop<E>`) — single round-trip versus
/// the N+1 cost of `list_edges` followed by per-id lookups.
async fn list_edge_records(
    &self,
    _ctx: &ExecutionContext,
    _ns: &Namespace,
    _limit: usize,
    _offset: usize,
) -> Result<Vec<GraphHop<E>>>;
```

### Why `GraphHop<E>` for the records shape

Every edge-yielding method in the trait already returns
`GraphHop<E>` (or close): `temporal_filter` returns
`Vec<GraphHop<E>>`, `traverse` / `find_path` return
`Vec<GraphHop<E>>`, `neighbors` returns
`Vec<(EdgeId, NodeId, E)>` (a hop with neighbour-only).

Reusing `GraphHop<E>` means inspection consumers iterate
uniformly across all edge surfaces without unwrapping different
tuple shapes. The struct already carries every field operators
need (`edge_id`, `from`, `to`, `edge`, `timestamp`).

### Why two methods, again

Same rationale as ADR-0048's `list_nodes` + `list_node_records`
split: hot-path callers don't pay the JSON-decode tax for
payloads they don't need. For edges the orphan-sweep parallel
is weaker (edges with no resolvable endpoint shouldn't exist
under correct cascade behaviour from ADR-0046), but
`list_edges` is still useful for:

- Bulk-id migration to another backend (no payload needed).
- "How many edges do we have?" cheap existence check (count
  by paging without payload decode).
- Per-id audit-log emission (operator already has the id
  from upstream, wants to confirm presence cheaply).

The cost of having both methods is one extra trait surface
entry. The benefit is consistent shape across `list_nodes` /
`list_node_records` / `list_edges` / `list_edge_records` —
operators learn one pagination idiom, apply it everywhere.

### When to use vs `temporal_filter`

`temporal_filter(ns, from, to)` is the right call when the
operator has a time window in mind ("edges learned last week").
It returns *every* hit in the window — no pagination.

`list_edges` / `list_edge_records` is the right call for
time-agnostic enumeration with pagination ("walk every edge in
this namespace 1000 at a time"). The trait docs cross-link to
make the choice explicit.

### `InMemoryGraphMemory` impl

Read-side lock; `BTreeMap<EdgeId, StoredEdge<E>>` iteration is
already ascending. `list_edges` collects keys; `list_edge_records`
constructs `GraphHop<E>` by cloning the stored edge fields.

### `PgGraphMemory` impl

`list_edges`:
```sql
SELECT id FROM graph_edges
WHERE namespace_key = $1
ORDER BY id ASC
LIMIT $2 OFFSET $3
```

`list_edge_records`:
```sql
SELECT id, from_node, to_node, payload, ts FROM graph_edges
WHERE namespace_key = $1
ORDER BY id ASC
LIMIT $2 OFFSET $3
```

Both wrap in the standard `pool.begin → set_tenant_session →
query → commit` envelope (ADR-0043 RLS-aware tenant tx). The
`(namespace_key, id)` composite PK serves both the WHERE anchor
and the ORDER BY.

### Tests

- 4 `InMemoryGraphMemory` unit tests: pagination in id order,
  full-hop record shape (verifies `from`/`to`/`edge`/`timestamp`
  set), namespace isolation, empty namespace.
- 1 `PgGraphMemory` testcontainers test: list_edges +
  list_edge_records round-trip + cross-namespace isolation.

## Consequences

✅ Edge enumeration surface is symmetric with node enumeration.
Operators learn one pagination idiom for nodes / edges
together.
✅ `GraphHop<E>` reused everywhere it makes sense — uniform
shape across `neighbors` / `traverse` / `find_path` /
`temporal_filter` / `list_edge_records`. No new tuple shape to
remember.
✅ Hot-path bulk-id callers stay cheap via `list_edges` (no
JSONB decode).
✅ Inspection callers get full hops in one round-trip via
`list_edge_records`.
✅ Default empty-vec impls shield existing user-implemented
backends (Neo4j, ArangoDB, etc.) from breaking.
❌ Trait surface grew by another two methods. After this slice
GraphMemory has 13 methods. The doc structure groups them
clearly (add / read / traverse / delete / enumerate); we'll
revisit if the surface starts feeling cluttered.
❌ `OFFSET` pagination cost grows with depth — same caveat as
ADR-0047. Operators paging tens of thousands of edges deep
should switch to keyset pagination (out of scope).

## Alternatives considered

1. **Only `list_edge_records`, drop `list_edges`** — orphan-
   sweep parallel for edges is weak (no edge-orphan concept
   under correct cascade). But the bulk-id-migration use case
   is real, and asymmetry with the node pair would surprise
   operators learning the API. Both shipped for consistency.
2. **`list_edges -> Vec<(EdgeId, NodeId, NodeId)>` (id +
   endpoints, no payload)** — three return shapes (`EdgeId`
   alone, endpoints+id, full hop) doesn't match a real use
   case and complicates the surface. Two-method shape keeps
   the dichotomy clean (id-only vs full).
3. **Generalize via `list_edges_filtered(ns, EdgeFilter)`** —
   `EdgeFilter::All` / `EdgeFilter::TimeRange(from, to)` /
   `EdgeFilter::FromNode(id)` / etc. Larger surface area
   without enabling anything `temporal_filter` + `neighbors`
   + `list_edges*` don't already do. Future-extensible enum
   can land later if real use cases appear.
4. **Drop `temporal_filter` since `list_edges*` covers
   enumeration** — different shape (`temporal_filter` is
   time-windowed without pagination, `list_edges*` is
   time-agnostic with pagination). Both serve distinct
   patterns. Out.

## Operator usage patterns

**Bulk export of every edge** (full payload, paginated):
```rust
let mut offset = 0;
loop {
    let page = graph.list_edge_records(&ctx, &ns, 500, offset).await?;
    if page.is_empty() { break; }
    for hop in &page {
        export(&hop.edge_id, &hop.from, &hop.to, &hop.edge, hop.timestamp).await?;
    }
    offset += page.len();
}
```

**Edge-id migration** (cheap, no payload):
```rust
let mut offset = 0;
loop {
    let ids = graph.list_edges(&ctx, &ns, 1000, offset).await?;
    if ids.is_empty() { break; }
    write_id_mapping(&ids).await?;
    offset += ids.len();
}
```

**Hybrid sweep** (paginated existence check + selective deep
inspection):
```rust
let mut offset = 0;
loop {
    let ids = graph.list_edges(&ctx, &ns, 1000, offset).await?;
    if ids.is_empty() { break; }
    for id in &ids {
        if needs_inspection(id) {
            // Selective deep look — N+1 acceptable since N is small.
            // (No `edge(id)` accessor on the trait yet — operators
            // who need single-edge lookup compose via temporal_filter
            // on a tight window or list_edge_records over a known page.)
            //
            // Pattern reserved for a future `edge(id)` slice.
        }
    }
    offset += ids.len();
}
```

## References

- ADR-0007 — `GraphMemory` trait surface (parent).
- ADR-0042 — `entelix-graphmemory-pg` companion crate.
- ADR-0043 — RLS pattern (tenant-tx envelope reused here).
- ADR-0047 — `list_nodes` (the slice that established the
  ids-only pagination shape).
- ADR-0048 — `list_node_records` (the slice that established
  the dual-method pattern this slice mirrors).
- 7-차원 roadmap §S8 — Phase 7 twelfth sub-slice.
- `crates/entelix-memory/src/graph.rs` — trait methods + default
  + `InMemoryGraphMemory` impls + 4 unit tests.
- `crates/entelix-graphmemory-pg/src/store.rs` — `PgGraphMemory`
  impls with tenant-tx envelope.
- `crates/entelix-graphmemory-pg/tests/postgres_e2e.rs` — 1
  testcontainers regression.
