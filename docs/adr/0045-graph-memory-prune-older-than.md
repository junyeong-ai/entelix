# ADR 0045 — `GraphMemory::prune_older_than` — TTL maintenance for graph backends

**Status**: Accepted
**Date**: 2026-05-01
**Decision**: Phase 7 of the post-7-차원-audit roadmap (eighth sub-slice)

## Context

`GraphMemory<N, E>` records typed nodes plus typed timestamped
edges (ADR-0007). Long-running deployments accumulate edges that
the operator may want to prune by age — a knowledge-graph agent
that learned facts months ago may need to drop relationships
that have not been re-confirmed within a TTL.

The trait shipped without a TTL-sweep primitive. Operators
needing one had two paths:

1. Roll their own `DELETE FROM graph_edges WHERE …` against the
   backing Postgres directly, bypassing the SDK and breaking
   invariant 11's "every read/write rides a tenant anchor"
   contract.
2. Use `temporal_filter` to read every old edge into application
   memory and call no-op (no `delete_edge` exists) — a workaround
   that gathers data but cannot act on it.

Both paths failed. The trait needed a first-class sweep method.

The pattern already existed for sibling memory shapes:
[`EntityMemory::prune_older_than`] for entity facts and
[`EpisodicMemory::prune_older_than`] for episodic logs (both
take `Duration`, return removed count). `GraphMemory` was the
odd one out.

## Decision

Add `prune_older_than(ctx, ns, ttl: Duration) -> Result<usize>`
to the `GraphMemory` trait with a default `Ok(0)` impl. Drops
edges whose `timestamp` is older than `Utc::now() - ttl` in the
namespace; returns the count of removed edges so callers can
log or expose pruning metrics.

```rust
async fn prune_older_than(
    &self,
    _ctx: &ExecutionContext,
    _ns: &Namespace,
    _ttl: std::time::Duration,
) -> Result<usize> {
    Ok(0)
}
```

### Edge-only, not node-cascading

Edges have a timestamp; nodes don't. A TTL sweep cannot reason
about node age directly. Two options were on the table:

1. **Edge-only sweep**: drop only edges, leave nodes — including
   ones that become orphans — in place. Operators clean up
   orphans separately if they want.
2. **Cascade sweep**: drop edges, then drop nodes that have no
   remaining edges.

Decision: option 1. Three reasons:

- **Mirrors siblings**: `EntityMemory::prune_older_than` and
  `EpisodicMemory::prune_older_than` operate on a single
  timestamp axis without cascading semantics. Same shape across
  the family.
- **Avoids surprise**: an operator running prune on an active
  graph might lose a node that was actively being inserted
  (orphan-by-race). Edge-only sweep makes this impossible.
- **Cheap to layer**: operators wanting orphan cleanup write a
  follow-up query — the building blocks already exist
  (`neighbors` to detect orphans, manual deletion via direct
  SQL or a future `delete_node` extension). Forcing cascade
  into the trait would impose the cost on operators that don't
  want it.

The doc string makes this explicit: "Edge-only by design — nodes
left orphaned by edge removal stay in place until the operator
drops them explicitly."

### `InMemoryGraphMemory` impl

Walks the per-namespace edge map under the write lock,
collects stale `EdgeId`s, removes each from `edges` map and
both adjacency lists (`out_adj`, `in_adj`). Returns the count.
The `chrono::Duration::from_std(ttl).unwrap_or(Duration::MAX)`
pattern matches the saturation handling
`EntityMemory::prune_older_than` already uses for pathological
TTLs.

### `PgGraphMemory` impl

Single `DELETE FROM graph_edges WHERE namespace_key = $1 AND
ts < $2` wrapped in the standard `pool.begin →
set_tenant_session → query → commit` envelope (ADR-0043 RLS
pattern). Returns `result.rows_affected()` clamped to `usize`.
The `(namespace_key, ts)` covering index installed by the
ADR-0042 bootstrap makes this an O(log n + k) scan.

### Two regression tests for `InMemoryGraphMemory`

1. `prune_older_than_drops_stale_edges_only` — inserts one old
   + one fresh edge, verifies removed count, verifies both
   nodes survive (edge-only contract), verifies adjacency lists
   are deduplicated (no dangling references).
2. `prune_older_than_on_empty_namespace_is_noop` — prune on a
   namespace with no edges returns 0 with no error.

### One testcontainers test for `PgGraphMemory`

`prune_older_than_drops_stale_edges_at_db_layer` mirrors the
in-memory test against a real Postgres container. Verifies
edge-only contract + adjacency consistency (via `neighbors`
query post-prune).

## Consequences

✅ `GraphMemory` family now has a uniform TTL-sweep surface,
matching `EntityMemory` and `EpisodicMemory`. Long-running
deployments can bound graph growth without bypassing the SDK.
✅ The default `Ok(0)` impl means existing backends (e.g. an
operator's custom `GraphMemory`) compile unchanged. Backends
that implement it pick up the new behaviour.
✅ Edge-only design means operators control orphan-node policy
explicitly — no surprise data loss from active inserts racing a
sweep.
✅ Postgres impl uses the existing `(namespace_key, ts)` covering
index, no schema change. Sweep cost stays sub-linear in total
edge count.
✅ Postgres impl rides the same RLS-aware tenant-tx envelope as
every other write site (ADR-0043) — no special-case path,
defense-in-depth uniform.
❌ Trait surface grew by one method. The default impl shields
existing implementors, but the public-API baseline diff includes
the new signature. Refreshed for `entelix-memory` and
`entelix-graphmemory-pg`.
❌ Orphan-node cleanup is the operator's responsibility. A
`delete_node` trait method or a `prune_orphan_nodes` companion
sweep is reserved for a future slice when there's a clear
operator demand signal. The current shape stays minimal.

## Alternatives considered

1. **Cascade sweep (drop edges + orphan nodes)** — rejected;
   see "edge-only" rationale above. Surprise data loss + breaks
   sibling-pattern symmetry.
2. **`prune_older_than(cutoff: DateTime<Utc>)` instead of
   `Duration`** — caller passes the absolute cutoff. More
   flexible but breaks the existing `Duration`-shaped sibling
   APIs. Operators that need absolute cutoffs pass
   `Utc::now() - cutoff` themselves; the asymmetry isn't worth
   it.
3. **Return `Vec<GraphHop<E>>` of removed edges instead of
   count** — heavier surface, useful only for audit replays
   that should already be using `temporal_filter` + manual
   delete. Count matches sibling shape (`EntityMemory`,
   `EpisodicMemory`).
4. **Add `delete_edge(ctx, ns, edge_id)` first** — would let
   operators build their own prune by calling `temporal_filter`
   + `delete_edge` per hit. Two round-trips per edge in
   non-Postgres backends; one round-trip pruning is the right
   default. `delete_edge` is its own slice (not blocked by
   prune).

## References

- ADR-0007 — `GraphMemory` trait surface (parent).
- ADR-0042 — `entelix-graphmemory-pg` companion crate (where
  the Postgres impl lives).
- ADR-0043 — RLS pattern (the tenant-tx envelope this slice
  reuses).
- 7-차원 roadmap §S8 — Phase 7 eighth sub-slice.
- `crates/entelix-memory/src/graph.rs` — trait method + default
  impl + `InMemoryGraphMemory` impl + 2 unit tests.
- `crates/entelix-graphmemory-pg/src/store.rs` — `PgGraphMemory`
  impl.
- `crates/entelix-graphmemory-pg/tests/postgres_e2e.rs` —
  testcontainers regression.
