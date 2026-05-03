# ADR 0048 — `GraphMemory::list_node_records` — paginated `(NodeId, N)` enumeration

**Status**: Superseded by ADR-0065 (trait-method portion moved to `PgGraphMemory` inherent — `GraphMemory` trait stays at the agent-facing surface)
**Date**: 2026-05-01
**Decision**: Phase 7 of the post-7-차원-audit roadmap (eleventh sub-slice)

## Context

ADR-0047 added `list_nodes` returning `Vec<NodeId>` (ids only)
to match `Store::list`'s sibling shape and to keep the orphan-
sweep hot path cheap (no payload decode). The doc explicitly
called out the trade-off:

> Operators that need the payload too call [`Self::node`] per
> id, accepting the N+1 cost for the inspection use case.

The N+1 was acceptable as a starting point but isn't ideal for
real inspection workloads — bulk export, audit walks, and
structural inspection all want the payload alongside the id.

`EpisodicMemory::all` and `EntityMemory::all_records` both
return `(id, payload)` shapes (as `Vec<Episode<V>>` and
`HashMap<String, EntityRecord>` respectively) — the precedent
across the family is "if you need payloads, the trait gives
them in one round-trip". `GraphMemory` was outdoorlier than it
needed to be.

## Decision

Add `list_node_records` to `GraphMemory` returning paginated
`Vec<(NodeId, N)>` with the same order and pagination contract
as `list_nodes`. Default impl returns the empty vector so
existing implementors compile unchanged.

```rust
async fn list_node_records(
    &self,
    _ctx: &ExecutionContext,
    _ns: &Namespace,
    _limit: usize,
    _offset: usize,
) -> Result<Vec<(NodeId, N)>> {
    Ok(Vec::new())
}
```

### Why two methods, not one

`list_nodes` and `list_node_records` coexist for one reason:
**the orphan-sweep hot path doesn't want payload decode cost**.
Sweeping a 100k-node namespace, decoding 100k JSON payloads
when the only fact the loop reads is "this node has no
neighbours" is wasteful — both bandwidth (over-the-wire bytes)
and CPU (serde_json deserialize per row).

A single `list_node_records` would force every caller to pay
the payload cost. A single `list_nodes` would force inspection
callers into the N+1 pattern. Both methods, both honest.

The doc on each method points at the other:

- `list_nodes`: "Inspection-style callers that *do* want
  payloads use `list_node_records` — same pagination, single
  round-trip."
- `list_node_records`: "For orphan-sweep and traversal-seeding
  paths that only need the id, prefer `list_nodes` to skip the
  payload-decode cost."

### Same pagination semantics as `list_nodes`

`(limit, offset)`, `NodeId` ascending. Mirrors the sibling so
operators page through both with identical loop shape.

### `InMemoryGraphMemory` impl

```rust
guard.nodes.iter()
    .skip(offset).take(limit)
    .map(|(id, payload)| (id.clone(), payload.clone()))
    .collect()
```

Read-side lock; clones both id and payload — `N: Clone` is
required by the trait so the cost is the operator's choice of
payload shape.

### `PgGraphMemory` impl

```sql
SELECT id, payload FROM graph_nodes
WHERE namespace_key = $1
ORDER BY id ASC
LIMIT $2 OFFSET $3
```

Wrapped in the standard `pool.begin → set_tenant_session →
query → commit` envelope (ADR-0043 RLS-aware tenant tx).
Decodes JSONB payload into `N` via `serde_json::from_value`;
errors propagate through the result.

The composite PK `(namespace_key, id)` covers both the WHERE
anchor and the ORDER BY, so the query plan stays an indexed
scan.

### Tests

- 3 `InMemoryGraphMemory` unit tests: pagination payload-equal,
  namespace isolation, empty namespace.
- 1 `PgGraphMemory` testcontainers test: round-trip in real
  Postgres + cross-namespace isolation.

## Consequences

✅ Inspection-style consumers (audit walk, bulk export,
structural diff) get payloads in a single round-trip — no N+1.
✅ Orphan-sweep + traversal-seeding paths keep the
`list_nodes` cheap shape that ADR-0047 established. Both costs
are explicit in the trait surface.
✅ `EpisodicMemory::all` / `EntityMemory::all_records` family
shape extended — payloads-with-ids is the consistent
inspection idiom across the memory crates.
✅ Default empty-vec impl shields existing user-implemented
backends (Neo4j, ArangoDB, etc.) from breaking. Backends that
own a node index pick up the behaviour by overriding.
❌ Trait surface grew by another method — two pagination
methods on `GraphMemory` (`list_nodes` + `list_node_records`).
The naming distinguishes them clearly, and the docs cross-link
so callers find the right one.
❌ JSONB decode cost on `list_node_records` is real for large
namespaces. Operators paging through 100k+ rows with payloads
should consider keyset pagination + parallel decode (out of
scope).

## Alternatives considered

1. **Single `list_nodes(ns, limit, offset, with_payload: bool)`
   method** — runtime branching, leaks an enum / option through
   the trait. The two-method shape reads more clearly at the
   call site and lets the type system distinguish the return
   shape (`Vec<NodeId>` vs `Vec<(NodeId, N)>`).
2. **`list_nodes(ns, limit, offset) -> Vec<(NodeId, Option<N>)>`
   with payload always `Some` from this slice** — collapses
   the return shape but defers the decode question to the
   caller. Worst of both worlds.
3. **`list_node_records` returns iterator (lazy decode)** —
   the existing `list_nodes` is `Vec`-shaped; consistency
   wins. Streaming pagination is a future slice if demand
   appears.
4. **Make `list_nodes` deprecated and just keep
   `list_node_records`** — violates invariant 14 (no
   backwards-compat shims) and forces every orphan-sweep
   caller into the payload-decode tax. Rejected; both methods
   are honest, both stay.

## Operator usage patterns

**Audit walk** (now one round-trip per page):
```rust
let mut offset = 0;
loop {
    let page = graph.list_node_records(&ctx, &ns, 100, offset).await?;
    if page.is_empty() { break; }
    for (id, payload) in &page {
        audit_log(id, payload).await?;
    }
    offset += page.len();
}
```

**Bulk export**:
```rust
let mut offset = 0;
loop {
    let page = graph.list_node_records(&ctx, &ns, 500, offset).await?;
    if page.is_empty() { break; }
    for (id, payload) in &page {
        export(id, payload).await?;
    }
    offset += page.len();
}
```

**Structural inspection** (e.g., schema validation):
```rust
let records = graph.list_node_records(&ctx, &ns, 1000, 0).await?;
for (id, payload) in records {
    if !schema.is_valid(&payload) {
        tracing::warn!(node = %id, "schema mismatch");
    }
}
```

**Hybrid sweep** (orphan check uses `list_nodes`; payload-using
follow-up uses `list_node_records`):
```rust
let candidates = graph.list_nodes(&ctx, &ns, 1000, 0).await?;
let mut orphans = Vec::new();
for id in candidates {
    if graph.neighbors(&ctx, &ns, &id, Direction::Both)
        .await?.is_empty()
    {
        orphans.push(id);
    }
}
// Now fetch payloads only for the orphans we actually plan to act on.
for id in orphans {
    if let Some(payload) = graph.node(&ctx, &ns, &id).await? {
        if archive_before_delete(&payload) {
            archive(&id, &payload).await?;
        }
    }
    graph.delete_node(&ctx, &ns, &id).await?;
}
```

## References

- ADR-0007 — `GraphMemory` trait surface (parent).
- ADR-0042 — `entelix-graphmemory-pg` companion crate.
- ADR-0043 — RLS pattern (tenant-tx envelope reused here).
- ADR-0047 — `list_nodes` (the slice this builds on; explicitly
  documented the N+1 cost as the gap this slice closes).
- 7-차원 roadmap §S8 — Phase 7 eleventh sub-slice.
- `crates/entelix-memory/src/graph.rs` — trait method + default
  + `InMemoryGraphMemory` impl + 3 unit tests.
- `crates/entelix-graphmemory-pg/src/store.rs` — `PgGraphMemory`
  impl with tenant-tx envelope.
- `crates/entelix-graphmemory-pg/tests/postgres_e2e.rs` — 1
  testcontainers regression.
