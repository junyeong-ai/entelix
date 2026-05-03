# ADR 0042 — `entelix-graphmemory-pg` companion crate

**Status**: Accepted
**Date**: 2026-05-01
**Decision**: Phase 7 of the post-7-차원-audit roadmap (fifth sub-slice)

## Context

`entelix_memory::GraphMemory<N, E>` is the trait for relationship-
aware long-term memory: typed nodes, typed timestamped edges,
BFS traversal, shortest-path search, temporal-window filter
(ADR-0007 §"GraphMemory"). Until this slice the only impl was
`InMemoryGraphMemory` — a `BTreeMap` adjacency-list reference
that loses every edge on process restart.

The 7-차원 audit's Phase 7 roadmap entry called for a production
backend so operators don't have to choose between using the
GraphMemory abstraction and surviving a restart. Choices were:

- **Neo4j** — purpose-built graph DB. Operationally heavyweight
  for the typical operator who already has Postgres. Out of
  scope for this slice; trait-only support continues.
- **Neptune / TigerGraph** — same story.
- **Postgres with adjacency tables** — the operator's existing
  database, no new operational surface, no extension dependency,
  honours the established companion-crate pattern (ADR-0008).

The companion-crate pattern is the established shape:
`entelix-memory-pgvector` (ADR-0008), `entelix-memory-qdrant`,
`entelix-memory-openai` all sit alongside `entelix-memory` and
plug a single trait without changing the trait surface from
outside.

## Decision

Add `entelix-graphmemory-pg` as the workspace's 18th crate. Two
adjacency tables (`graph_nodes` + `graph_edges`), composite
`(namespace_key, id)` primary keys, JSONB payload columns,
covering indexes on `(namespace_key, from_node)` /
`(namespace_key, to_node)` / `(namespace_key, ts)`. Builder
mirroring the pgvector pattern (`with_connection_string` /
`with_pool` / `with_auto_migrate(false)` for IaC-managed
schemas).

### Schema shape

```sql
CREATE TABLE graph_nodes (
    namespace_key TEXT NOT NULL,
    id TEXT NOT NULL,
    payload JSONB NOT NULL,
    PRIMARY KEY (namespace_key, id)
);

CREATE TABLE graph_edges (
    namespace_key TEXT NOT NULL,
    id TEXT NOT NULL,
    from_node TEXT NOT NULL,
    to_node TEXT NOT NULL,
    payload JSONB NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (namespace_key, id)
);

CREATE INDEX graph_edges_from_idx ON graph_edges (namespace_key, from_node);
CREATE INDEX graph_edges_to_idx   ON graph_edges (namespace_key, to_node);
CREATE INDEX graph_edges_ts_idx   ON graph_edges (namespace_key, ts);
```

Three deliberate choices:

1. **JSONB payload columns** — `N` and `E` are operator-shaped
   generics (the trait says `Clone + Send + Sync + 'static`).
   The backend adds `Serialize + DeserializeOwned` via its own
   impl bounds and stores everything as JSONB. Single schema
   serves every payload shape; operators that want stronger
   typing per node label run schemas externally with
   `with_auto_migrate(false)`.
2. **`(namespace_key, …)` composite PKs** — invariant 11 / F2:
   tenant boundary is structural, not advisory. The composite
   PK doubles as the B-tree index that every query's
   `WHERE namespace_key = $1` anchor relies on, so namespace
   isolation costs zero extra index maintenance.
3. **Three covering indexes on edges** —
   `(namespace_key, from_node)` and `(namespace_key, to_node)`
   keep `neighbors()` queries O(log n) regardless of total edge
   count; `(namespace_key, ts)` supports `temporal_filter()`'s
   range scan cheaply. The composite PK is the fourth implicit
   index covering single-edge lookup by id.

### Traversal model

`traverse` and `find_path` perform BFS layer-by-layer in the
Rust client. For `max_depth = D` that's at most `D` round-trips
per call. A `WITH RECURSIVE`-based fast path that folds the BFS
into one round-trip is reserved for a follow-up slice — the
trait surface stays unchanged whichever model the impl uses.

`find_path` uses BFS with parent tracking: walks layers
breadth-first, stops as soon as the target is reached, then
reconstructs the path by tracing parents from `to` backwards
through the visited-hop list. Same-node case yields `Some(vec![])`
(empty path), unreachable yields `None`.

### Direction enum forward compatibility

`entelix_memory::Direction` is `#[non_exhaustive]`. The match
on direction in `fetch_neighbours` covers `Outgoing` /
`Incoming` / `Both` and explicitly errors with
`Error::invalid_request` on any future variant — no silent
approximation with one of the existing arms. Aligns with
invariant #15 (no silent fallback).

### Deferred: Postgres RLS

ADR-0041 added row-level security to the
`entelix-persistence` tables. The same treatment makes sense
for the two graph tables here, but is deferred:

1. Companion crates ship together with the trait crate — the
   established pattern (`entelix-memory-pgvector`,
   `entelix-memory-qdrant`) does not enable RLS, and adding it
   to one new companion would introduce drift across the family.
2. The scope of "RLS for every backend table" is its own slice
   that touches all four crates uniformly.

The doc string of `PgGraphMemory` calls this out as future work
so operators know to wire it externally if they need it before
the unified slice lands.

## Consequences

✅ Operators using `GraphMemory` for production workloads now
have a 1급 backend that survives process restart, without
introducing Neo4j as a new operational dependency.
✅ Schema is self-bootstrapping (`auto_migrate=true` default)
for fast spin-up; IaC-managed deployments opt out via
`with_auto_migrate(false)` — same pattern as
`entelix-memory-pgvector`.
✅ Multi-tenancy is structural — composite `(namespace_key, id)`
PKs make cross-tenant queries impossible at the schema layer
(F2 mitigation), and the PK doubles as the index every query's
namespace anchor uses.
✅ Per-call payload codec is JSON; operators that want
strongly-typed per-label schemas run their own DDL with
`with_auto_migrate(false)`.
✅ Workspace grows from 17 → 18 crates; facade re-exports
under `graphmemory-pg` feature; `full` aggregator gains the new
flag.
❌ BFS issues `D` round-trips per call (where `D` is
`max_depth`). For very deep graphs this is a real latency cost.
A `WITH RECURSIVE` fast path is reserved for a follow-up slice
that keeps the trait surface unchanged.
❌ JSONB payload columns can't enforce per-label schema — every
operator pays the JSON encode/decode cost on every read/write.
Strongly-typed schemas remain available via the schema-as-code
escape hatch.
❌ RLS is not yet wired for these tables — the operator must
add it externally if defense-in-depth at the DB layer is
required. Tracked as a follow-up slice that unifies RLS across
all companion-crate tables (not just `entelix-persistence`).

## Alternatives considered

1. **Single-table edge-list with `node_payload` denormalised
   into `graph_edges`** — fewer joins for traversal queries
   but breaks cleanly when nodes have no edges (orphan nodes
   wouldn't be queryable). Rejected; two-table normalised form
   is the standard adjacency-list shape.
2. **`uuid` typed columns instead of `TEXT`** — `NodeId` /
   `EdgeId` happen to be UUID v7 today but the trait specifies
   them as opaque `String`-backed. Storing as `TEXT` honours
   the trait's intent and lets backends mint non-UUID ids
   later (e.g. operator-supplied externally-generated ids).
   Rejected the type narrowing.
3. **Per-namespace tables** (one `graph_nodes_<ns>` table per
   namespace) — moves namespace boundary from a `WHERE` clause
   to schema isolation. Cleaner per-tenant DBA story but
   explodes the schema for any tenant count over a few dozen.
   Rejected for the standard companion shape.
4. **`pg_graph` extension** — Postgres has experimental graph
   extensions. Most are alpha-quality or carry significant
   operational risk. Rejected; vanilla Postgres is the
   conservative choice.

## Operator deployment notes

- New crate ships as a workspace member; depend via the facade
  feature flag `graphmemory-pg` or directly on
  `entelix-graphmemory-pg`.
- `auto_migrate=true` is on by default. Toggle off when the
  schema is owned externally (DBA, IaC) — the migration is
  three `CREATE TABLE` and three `CREATE INDEX` statements,
  all `IF NOT EXISTS`.
- Custom table names supported via `with_nodes_table` /
  `with_edges_table` for operators that namespace by table name.
- For RLS defense-in-depth, layer your own policy externally
  using the `namespace_key` column (a follow-up slice will
  unify this across companion crates).

## References

- ADR-0007 — `GraphMemory` trait surface (parent).
- ADR-0008 — companion crate pattern.
- ADR-0017 — `Namespace` + tenant_id mandatory.
- ADR-0040 — `list_namespaces` round-trip via `Namespace::parse`
  (analogous structural-correctness slice).
- ADR-0041 — `entelix-persistence` RLS (the pattern this crate
  defers integrating).
- 7-차원 roadmap §S8 — Phase 7 fifth sub-slice.
- `crates/entelix-graphmemory-pg/src/store.rs` — surface +
  trait impl.
- `crates/entelix-graphmemory-pg/tests/postgres_e2e.rs` —
  testcontainers regression (6 cases).
