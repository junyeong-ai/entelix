# ADR 0058 — `WITH RECURSIVE` BFS fast path for `PgGraphMemory`

**Status**: Accepted
**Date**: 2026-05-01
**Decision**: Phase 9 of the post-7-차원-audit roadmap (first sub-slice — perf hardening of the graphmemory-pg companion)

## Context

ADR-0042 shipped the `PgGraphMemory<N, E>` companion crate with
BFS traversal and shortest-path implemented client-side: each
`max_depth = D` `traverse` or `find_path` call issued **D** sequential
neighbour queries, expanding one BFS layer per round-trip. ADR-0042's
"Traversal model" doc explicitly flagged this as deferred work:

> Future work: a `WITH RECURSIVE`-based fast path that folds the
> BFS into one round-trip; the trait surface stays unchanged.

For `max_depth = 5` on a Postgres deployment with 5ms RTT, that
was ~25ms of round-trip latency on top of the actual server-side
work. Wider knowledge-graph traversals (`max_depth = 10`) doubled
it. The workload is precisely what `WITH RECURSIVE` was designed
for — graph BFS expressed as a self-referential CTE that the
planner expands server-side.

This slice closes that deferred item.

## Decision

Replace the layer-by-layer Rust BFS in `traverse` and `find_path`
with single-round-trip `WITH RECURSIVE` queries. The trait
surface stays unchanged; only the implementation moves.

### `traverse` shape

```sql
WITH RECURSIVE walk(edge_id, edge_from, edge_to, payload, ts,
                    frontier, depth, visited) AS (
    SELECT NULL, NULL, NULL, NULL, NULL,
           $start::TEXT, 0, ARRAY[$start::TEXT]
    UNION ALL
    SELECT e.id, e.from_node, e.to_node, e.payload, e.ts,
           <next_node_expr>, w.depth + 1,
           w.visited || (<next_node_expr>)
    FROM walk w
    JOIN graph_edges e ON e.namespace_key = $ns AND <join_predicate>
    WHERE w.depth < $max_depth
      AND NOT ((<next_node_expr>) = ANY(w.visited))
),
ranked AS (
    SELECT *, ROW_NUMBER() OVER (
        PARTITION BY frontier ORDER BY depth ASC, edge_id ASC
    ) AS rn
    FROM walk WHERE depth > 0
)
SELECT edge_id, edge_from, edge_to, payload, ts
FROM ranked WHERE rn = 1
ORDER BY depth ASC, edge_id ASC;
```

The per-row `visited` array prevents path-level revisits (which
would cause infinite recursion on cyclic graphs); the `ROW_NUMBER()`
deduplication keeps exactly one row per reachable destination —
the shortest-depth arrival, matching the original Rust BFS's
"first arrival wins" dedupe semantic.

### `find_path` shape

```sql
WITH RECURSIVE walk(frontier, depth, visited, edge_path) AS (
    SELECT $from::TEXT, 0, ARRAY[$from::TEXT], ARRAY[]::TEXT[]
    UNION ALL
    SELECT <next_node_expr>, w.depth + 1,
           w.visited || (<next_node_expr>),
           w.edge_path || e.id
    FROM walk w
    JOIN graph_edges e ON e.namespace_key = $ns AND <join_predicate>
    WHERE w.depth < $max_depth AND w.frontier <> $to
      AND NOT ((<next_node_expr>) = ANY(w.visited))
),
shortest AS (
    SELECT edge_path FROM walk
    WHERE frontier = $to AND depth > 0
    ORDER BY depth ASC LIMIT 1
),
unrolled AS (
    SELECT u.eid, u.ord
    FROM shortest s, unnest(s.edge_path) WITH ORDINALITY AS u(eid, ord)
)
SELECT e.id, e.from_node, e.to_node, e.payload, e.ts
FROM unrolled u
JOIN graph_edges e ON e.namespace_key = $ns AND e.id = u.eid
ORDER BY u.ord ASC;
```

Each in-flight path carries its own `edge_path` array. The outer
query selects the shortest path that reached the target, unrolls
the array `WITH ORDINALITY` to preserve hop order, and rejoins
`graph_edges` to materialise the full hop bodies (`from_node`,
`to_node`, `payload`, `ts`). Empty result → `Ok(None)`. The
`from == to` early return stays in client code (it's a 1-line
shortcut for the depth-0 case the SQL deliberately filters out
with `WHERE depth > 0`).

### Direction parameterisation

A `DirectionSql` struct supplies SQL fragments for `Outgoing`,
`Incoming`, and `Both`, derived once per call by `direction_sql`.
The recursive path uses `e.from_node = w.frontier` (Outgoing) /
`e.to_node = w.frontier` (Incoming) / `(e.from_node = w.frontier
OR e.to_node = w.frontier)` (Both). The `Both` next-node
expression is the standard `CASE WHEN ... END` shape that picks
the endpoint that *isn't* the frontier. The `flat_*` variants
mirror the same logic against `$2` for the one-shot
`fetch_neighbours` projection used by `neighbors()`.

`Direction` is `#[non_exhaustive]`; an unknown variant surfaces
as `Error::invalid_request` (invariant 15 — no silent
approximation onto an existing arm).

### Why a per-row `visited` array, not a process-level visited table

A temporary table or a `RECURSIVE` query without per-row visited
state would be wrong for two reasons. First, a temp table needs
a separate `CREATE TEMP TABLE` round-trip — defeating the
single-RT goal. Second, a `WITH RECURSIVE` without cycle
prevention runs unboundedly when the graph has any cycle, even
when `max_depth` would terminate it; the planner cannot prove
finiteness from the depth filter alone. The per-row `visited`
array makes the recursion proof-of-termination explicit: every
recursive row has a strictly larger `visited` array than its
parent and we cannot revisit, so the depth is strictly bounded
by the namespace's reachable node count.

The cost is `O(D)` array storage per intermediate row and `O(D)`
linear scans per `= ANY(visited)` check, where `D ≤ max_depth`.
For the `max_depth ≤ 10` regime this is dominated by the JSONB
payload column reads, not the array work.

### Why edge-path accumulation in `find_path`, not parent tracking

A parent-tracking variant (each row records `parent_node`, then
the outer query reconstructs the path with another recursive CTE)
would split the work into two recursive expansions. The
edge-path-array approach folds reconstruction into the same pass:
the outer `unnest WITH ORDINALITY` plus a single `JOIN` against
`graph_edges` is one extra non-recursive scan. Lower constant
factor, simpler query.

### Tenant-tx envelope preserved

`set_tenant_session(tx, ns.tenant_id())` is still called once per
request inside a single transaction wrapping the recursive query
— invariant 11's RLS contract (ADR-0043) holds unchanged. The
single-RT win is on top of the existing single-tx envelope.

### Tests

Five new docker-ignored regressions in `postgres_e2e.rs`:

- `traverse_terminates_on_cycle` — `a → b → c → a` cycle with
  `max_depth = 10` returns exactly `b` and `c` (the seed `a` is
  excluded), proving the `visited`-array cycle guard works.
- `find_path_picks_shortest_among_multiple` — two paths from `a`
  to `d` (one 2-hop, one 3-hop); the `ORDER BY depth ASC LIMIT 1`
  in the `shortest` CTE picks the 2-hop one.
- `traverse_max_depth_zero_returns_empty` — `max_depth = 0`
  returns `[]`; verifies the `WHERE depth > 0` filter on the
  ranked projection matches the original Rust BFS's
  `while ... && depth < max_depth` semantic.
- `traverse_direction_both_handles_cycles` — exercises the
  `(from_node OR to_node)` + `CASE WHEN ... END` next-node shape
  on a cyclic graph; reachability + termination both hold.
- (Existing tests from ADR-0042 — `traverse_bfs_respects_max_depth`,
  `find_path_returns_shortest_or_none` — continue to pass with
  the new SQL, anchoring the original contract.)

## Consequences

✅ `traverse` and `find_path` cost one round-trip regardless of
`max_depth`; for `max_depth = 5` on a 5ms-RTT deployment, latency
drops from ~25ms to ~5ms (5× win), bigger for deeper traversals.
✅ Server-side BFS lets the Postgres planner use the existing
`(namespace_key, from_node)` / `(namespace_key, to_node)` covering
indexes for every recursive expansion — no per-layer round-trip
overhead for index lookup.
✅ Trait surface unchanged. Operators using `GraphMemory<N, E>`
through the trait (which is most of the codebase) see the speedup
transparently.
✅ Cycle prevention is now explicit in the SQL (per-row `visited`
array) rather than implicit in client-side state — a Postgres
planner cost-estimate failure cannot cause runaway recursion.
✅ The `flat_*` direction fragments share `direction_sql`, so the
one-shot `fetch_neighbours` projection used by `neighbors()`
benefits from the same single-source dispatch — drift between
the recursive and one-shot SQL shapes is structurally impossible.
❌ Adds query-shape complexity. Operators reading the SQL for the
first time pay one-time learning cost on `WITH RECURSIVE` +
`unnest WITH ORDINALITY`; both are documented Postgres features
since 8.4 / 9.4 respectively.
❌ Per-row `visited` array is `O(D)` memory per intermediate row.
For pathological `max_depth` values combined with a fanout-heavy
graph, the working set could grow large. Mitigation: `max_depth`
is operator-supplied and capped by the `i32::try_from(max_depth)
.unwrap_or(i32::MAX)` shape; the operator owns the bound.

## Alternatives considered

1. **Keep layer-by-layer Rust BFS** (status quo) — `D` round-trips
   per call. The known-bad shape ADR-0042 deferred. Rejected.
2. **PL/pgSQL stored procedure** for BFS / shortest-path —
   server-side recursion in procedural form. Splits schema-as-code
   between two surfaces (the migration's `CREATE TABLE` and a
   `CREATE OR REPLACE FUNCTION` block) and adds version-drift risk
   between the procedure body and the Rust code that calls it.
   `WITH RECURSIVE` in inline SQL gives the same single-RT win
   without the second source. Rejected.
3. **Materialised view of transitive closure** — pre-compute every
   reachable `(start, dest, depth, path)` tuple. `O(N²)` storage
   in the dense case; refresh cost dominates write throughput.
   Wrong tool for an OLTP knowledge-graph workload. Rejected.
4. **Batched layer query** (one `UNION ALL` of `D` neighbour
   queries pinned to a single connection) — folds round-trips
   into one but requires knowing `D` upfront and constructing a
   variable-arity SQL string. The recursive CTE approach is more
   uniform: one query shape, depth-agnostic. Rejected.
5. **Apache AGE / `pg_graphql` extension** — provides Cypher-style
   graph queries on Postgres but requires installing an extension
   that not every operator can opt into (managed-Postgres providers
   restrict extension installation). The native `WITH RECURSIVE`
   shape works on any vanilla Postgres ≥ 8.4. Rejected.

## References

- ADR-0042 — `PgGraphMemory` companion crate (parent — explicitly
  deferred this work).
- ADR-0043 — graphmemory-pg RLS (the tenant tx envelope this slice
  preserves).
- ADR-0046 — GraphMemory delete CRUD (sibling perf-shaped work).
- ADR-0052 — `prune_orphan_nodes` single-SQL anti-join (similar
  fold-into-one-query pattern).
- 7-차원 roadmap §S10 — Phase 9 (companion-perf hardening), first
  sub-slice.
- `crates/entelix-graphmemory-pg/src/store.rs` — `traverse_recursive`
  + `find_path_recursive` + `direction_sql` helpers.
- `crates/entelix-graphmemory-pg/tests/postgres_e2e.rs` — 4 new
  regression tests covering cycle, multi-path, depth-0, and
  Direction::Both shapes.
