# ADR 0043 — `entelix-graphmemory-pg` row-level security

**Status**: Accepted
**Date**: 2026-05-01
**Decision**: Phase 7 of the post-7-차원-audit roadmap (sixth sub-slice)

## Context

ADR-0042 shipped `entelix-graphmemory-pg` — a Postgres-backed
`GraphMemory<N, E>` companion crate — and explicitly deferred row-
level security:

> ### Deferred: Postgres RLS
>
> ADR-0041 added row-level security to the `entelix-persistence`
> tables. The same treatment makes sense for the two graph
> tables here, but is deferred […]

The "deferred" rationale was real but narrow: the unified RLS
slice across all Postgres-backed companion crates (pgvector +
graphmemory-pg) was its own scope. Splitting it per-companion
is the right cadence — each companion crate evolves its own
schema, and ADR-0041's machinery only commits to its own
tables.

This slice closes the deferred half for graphmemory-pg,
mirroring ADR-0041's design: `tenant_id` column populated on
INSERT, `ENABLE` + `FORCE ROW LEVEL SECURITY` + a single
`tenant_isolation` policy, every query site wrapped in a
transaction that stamps `entelix.tenant_id` via `set_config`.

The brand-new schema makes the change cheap — operators that
booted graphmemory-pg in the prior slice's state simply re-run
the bootstrap; the migration is idempotent (`IF NOT EXISTS` on
tables, `DROP POLICY IF EXISTS` before each policy create).

## Decision

### Schema additions

`graph_nodes` and `graph_edges` gain a `tenant_id TEXT NOT NULL`
column populated from `Namespace::tenant_id()` on every INSERT.
The denormalised column avoids parsing `namespace_key` (escaped
`tenant:scope[:scope...]`) at policy-evaluation time on every
row.

Bootstrap installs `ENABLE` + `FORCE ROW LEVEL SECURITY` on both
tables, plus a single `tenant_isolation` policy spanning USING
+ WITH CHECK on
`tenant_id = current_setting('entelix.tenant_id', true)`. Same
fail-closed semantic as ADR-0041 — unset variable surfaces as
`NULL`, which the comparison treats as false (invariant #15).

```sql
ALTER TABLE graph_nodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE graph_nodes FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation ON graph_nodes;
CREATE POLICY tenant_isolation ON graph_nodes
    USING (tenant_id = current_setting('entelix.tenant_id', true))
    WITH CHECK (tenant_id = current_setting('entelix.tenant_id', true));
-- repeat for graph_edges
```

`DROP POLICY IF EXISTS` before `CREATE POLICY` keeps the
bootstrap idempotent; without it a re-run would error on
already-existing policy. The `entelix-persistence` migration
runs once via `sqlx::migrate!`, so it doesn't need this — but
graphmemory-pg uses the runtime-DDL bootstrap pattern, so
re-runs are part of the contract.

### `set_tenant_session` helper (per-companion)

The `entelix-persistence` `set_tenant_session` helper is
`pub(super)` and not exported. Companion crates each ship their
own: a 6-line function calling `SELECT set_config('entelix.tenant_id', $1, true)`.
Replicating beats centralising for two reasons:

1. **No suitable shared crate.** Putting it in `entelix-memory`
   would force a `sqlx` dependency on the trait crate (rejected
   by ADR-0008's "trait surface stays sqlx-free" stance).
   Inventing a new `entelix-pg-utils` crate for one helper is
   over-engineered.
2. **No drift risk.** The SQL is one literal; behaviour is
   trivially testable. Companion crates that follow this pattern
   (future RLS for pgvector, qdrant) replicate the same line.

### Query-site wrapping

All five query methods (`add_node`, `add_edge`, `node`,
`neighbors`, `temporal_filter`) wrap in
`pool.begin() → set_tenant_session → query → commit`.

`bfs_layers` (the BFS helper used by `traverse` and
`find_path`) opens **one** transaction for the entire BFS,
stamps the tenant once, and reuses the same `&mut Transaction`
across every layer's `fetch_neighbours` call. This keeps the
RLS overhead at one BEGIN + one COMMIT regardless of `max_depth`
— the per-layer round-trips that the BFS already pays don't get
multiplied by an additional tx setup each.

### Why `bfs_layers` doesn't break the lock-ordering invariant

The BFS holds a Postgres transaction across multiple round-
trips. The CLAUDE.md lock-ordering rule says "Never hold any
lock across `.await` on a user-supplied future" — but the BFS
holds only the *connection's* transaction state, not a Rust
lock. The connection itself is borrowed from the pool, which is
a separate concern from the lock ordering rule (which targets
`parking_lot::Mutex` / `RwLock`-style guards).

`fetch_neighbours` was refactored to accept a generic
`Executor<'_, Database = Postgres>` so it can run against either
`&PgPool` (impossible now since all callers wrap in tx) or
`&mut Transaction<'_, Postgres>`. Single signature, two callers
through the same code path.

### Testcontainers regression — `tests/postgres_rls.rs`

Mirror of `entelix-persistence/tests/postgres_rls.rs`:

1. Boot postgres container as `postgres` (SUPERUSER), run
   bootstrap (creates schema + RLS policies).
2. `CREATE ROLE graph_app WITH LOGIN PASSWORD 'apppwd' NOSUPERUSER NOBYPASSRLS;`
3. `GRANT SELECT, INSERT, UPDATE, DELETE ON graph_nodes, graph_edges TO graph_app;`
4. Open second pool through `graph_app`, build a second
   `PgGraphMemory` with `with_auto_migrate(false)` (the app role
   lacks DDL privileges).
5. Verify SDK reads/writes work through the app pool (the
   helper wires the variable correctly).
6. Verify raw reads with no `entelix.tenant_id` set return 0
   rows (forgotten SET LOCAL → fail-closed).
7. Verify INSERT with mismatched `tenant_id` trips `WITH CHECK`.
8. Verify the same gate applies to `graph_edges`, not just
   `graph_nodes`.
9. Verify `count_in_tx` with the correct vs. wrong tenant
   returns 1 vs. 0 rows.

Existing `tests/postgres_e2e.rs` runs as the SUPERUSER `postgres`
role which bypasses RLS, so it continues to pass without
modification — exercises the application-layer correctness which
RLS sits on top of.

## Consequences

✅ Defense in depth for invariant #11 on the graph backend.
Forgotten SET LOCAL, direct DBA access without the variable,
and any future query site that omits the namespace anchor all
fail closed at the database layer.
✅ The bootstrap is idempotent — operators that already
deployed graphmemory-pg from ADR-0042 just re-run the bootstrap
and pick up RLS automatically. Existing rows have `NULL` in
`tenant_id` though, which would fail every `WITH CHECK` going
forward; the migration documents that operators with existing
data must backfill (single `UPDATE` from rendered `namespace_key`).
The brand-new shipping window for ADR-0042 means this is
approximately zero operators in practice.
✅ Companion-crate RLS pattern is now established. Future
slices for `entelix-memory-pgvector` and `entelix-memory-qdrant`
follow the same shape (per-crate `set_tenant_session`, FORCE
RLS, single-policy USING + WITH CHECK).
✅ `bfs_layers` keeps the BFS at D round-trips for traversal
plus 2 round-trips total (BEGIN + COMMIT) for the entire BFS
— no per-layer tx setup tax.
❌ `tenant_id` is denormalised — populated from
`Namespace::tenant_id()` on every INSERT. A code bug that bound
the wrong tenant to the column (vs. what `set_config` sees)
would still trip `WITH CHECK` rather than silently leak. The
WITH CHECK is the saving grace; the denormalisation cost is
~36 bytes per row.
❌ `evict_expired`-style cross-tenant operations don't exist on
`GraphMemory` (the trait has no sweeper method), so the
`BYPASSRLS` role pattern from ADR-0041 doesn't apply here. If
operators add a future cross-tenant maintenance method, it'll
need the same role pattern.
❌ `bfs_layers` holds a transaction across multiple round-trips.
Connection sticks with the BFS for its duration; high-cardinality
deep traversals can elevate pool pressure. Mitigated by the
client-side BFS being explicit about layer count.

## Alternatives considered

1. **Function-based RLS policy that parses `namespace_key`** —
   `current_setting('entelix.tenant_id', true) = split_part(namespace_key, ':', 1)`
   would avoid the denormalised `tenant_id` column. Rejected:
   the policy must call a function on every row eval, which the
   planner can't fold into the index scan; the denormalised
   column lets the policy use a simple equality the planner
   pushes into the WHERE chain.
2. **Move `set_tenant_session` to `entelix-memory`** — would
   force `entelix-memory` to depend on `sqlx`, reversing
   ADR-0008's "trait crate stays vendor-free" stance. Rejected.
3. **One `tenant_isolation` policy per table action (SELECT,
   INSERT, UPDATE, DELETE)** — finer-grained but the four
   actions all gate on the same predicate. The single-policy
   form means schema changes update one policy, not four.
   Rejected for the simpler shape.
4. **Bootstrap the policy only on freshly-created tables (skip
   on existing)** — would let operators with pre-RLS deployments
   keep the unprotected behaviour. Rejected; aligning the
   schema fully is the point. Operators who don't want RLS run
   `with_auto_migrate(false)` and own the schema externally.

## Operator deployment notes

- Operators upgrading from ADR-0042's graphmemory-pg ship: the
  bootstrap re-runs idempotently and adds RLS on the next
  `build()`. **Existing rows must be backfilled** with a
  `tenant_id` value derived from `namespace_key` (parse with
  `Namespace::parse(rendered).tenant_id()`). One-shot:
  `UPDATE graph_nodes SET tenant_id = split_part(namespace_key, ':', 1) WHERE tenant_id = '';`
  (assuming no `:` in tenant_id segments, which the escape
  contract guarantees for non-escaped tenants).
- Same `BYPASSRLS` role pattern as ADR-0041 for any future
  cross-tenant maintenance ops.
- Direct DBA access: `SELECT set_config('entelix.tenant_id', '<tenant>', true);`
  at the start of an interactive session.
- For deployments that prefer not to use RLS at all, run
  `with_auto_migrate(false)` and own a no-RLS schema externally.
  The SDK still wraps every query in `set_config` calls — those
  are harmless on a schema without the policy.

## References

- ADR-0007 — `GraphMemory` trait surface (parent).
- ADR-0008 — companion crate pattern (where this lives).
- ADR-0017 — `Namespace` + tenant_id mandatory.
- ADR-0032 — invariant #15 (no silent fallback) — RLS variant
  fail-close behaviour aligns with the same principle.
- ADR-0041 — `entelix-persistence` RLS (the design this slice
  mirrors per-companion).
- ADR-0042 — graphmemory-pg companion crate (the deferred half
  this slice closes).
- 7-차원 roadmap §S8 — Phase 7 sixth sub-slice.
- Migration: `crates/entelix-graphmemory-pg/src/migration.rs`
  (`enable_rls` helper).
- Helper: `crates/entelix-graphmemory-pg/src/tenant.rs`
  (`set_tenant_session`).
- Regression: `crates/entelix-graphmemory-pg/tests/postgres_rls.rs`
  (4 testcontainers cases).
