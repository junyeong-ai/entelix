# ADR 0041 — Postgres row-level security as defense-in-depth for invariant #11

**Status**: Accepted
**Date**: 2026-05-01
**Decision**: Phase 7 of the post-7-차원-audit roadmap (fourth sub-slice)

## Context

Invariant #11 declares cross-tenant data leak structurally
impossible by API design. The application layer enforces this
through:

- `Namespace::tenant_id` mandatory at the type level (no
  zero-arg constructor — F2 mitigation).
- `ThreadKey::from_ctx(ctx)` deriving `(tenant_id, thread_id)`
  from `ExecutionContext`'s mandatory `tenant_id` field
  (ADR-0017).
- Every Postgres query carries `WHERE tenant_id = $1` against
  the namespace / thread key.

The application path is correct as written, but the boundary is
purely advisory at the database layer. Three failure modes
remain:

1. **Forgotten WHERE clause in a future query** — a new query
   site could be added that queries the table without the
   tenant predicate. The CI gate doesn't (and can't easily)
   verify every SQL string.
2. **Direct DBA / external tooling access** — operations teams,
   reporting tools, ad-hoc scripts that connect to the same
   database can read across tenants. The boundary is enforced by
   convention, not by the database.
3. **SQL injection or string-formatting mistakes** — a future
   refactor that constructs SQL with mistakenly-formatted tenant
   filtering would silently leak. RLS at the database layer
   short-circuits this regardless of what the SQL string looks
   like.

The 7-차원 audit's Phase 7 roadmap entry called for Postgres
RLS as the production-recommended hardening for invariant #11.

## Decision

Add a new migration that enables row-level security on every
tenant-scoped table (`memory_items`, `session_events`,
`checkpoints`) with a single `tenant_isolation` policy that
gates both reads (`USING`) and writes (`WITH CHECK`) on
`current_setting('entelix.tenant_id', true)`. The SDK stamps
this variable per transaction before issuing any tenant-scoped
query.

### Migration `20260501000001_row_level_security.sql`

```sql
ALTER TABLE memory_items ENABLE ROW LEVEL SECURITY;
ALTER TABLE memory_items FORCE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation ON memory_items
    USING (tenant_id = current_setting('entelix.tenant_id', true))
    WITH CHECK (tenant_id = current_setting('entelix.tenant_id', true));

-- repeat for session_events, checkpoints
```

Three deliberate choices:

1. **`FORCE ROW LEVEL SECURITY`** — without `FORCE`, the table
   owner is exempt from the policy. Application database roles
   are typically also the table owner (the role that ran
   migrations), so without `FORCE` the policy would not apply
   to the SDK's normal pool. With `FORCE`, the only escape
   hatches are `SUPERUSER` (which bypasses every policy) and
   `BYPASSRLS` (per-role attribute reserved for maintenance
   roles).
2. **Single `tenant_isolation` policy spanning USING +
   WITH CHECK** — the read-side guard alone would silently let
   wrong-tenant writes succeed; the write-side guard alone would
   let wrong-tenant reads succeed. Both clauses are necessary.
3. **`current_setting(name, true)`** — the second `true`
   argument is `missing_ok`. When the variable is unset,
   `current_setting` returns `NULL`. `tenant_id = NULL` is
   `NULL` (unknown), which the policy treats as `false`. This
   means application code that *forgets* to stamp the variable
   produces empty result sets and `WITH CHECK` violations —
   loud failure, not silent cross-tenant leak. Aligns with
   invariant #15 (no silent fallback).

### `set_tenant_session` helper + per-query transaction wrapper

```rust
// crates/entelix-persistence/src/postgres/tenant.rs
pub(super) async fn set_tenant_session<'e, E>(executor: E, tenant_id: &str) -> Result<()>
where E: Executor<'e, Database = Postgres>
{
    sqlx::query("SELECT set_config('entelix.tenant_id', $1, true)")
        .bind(tenant_id).execute(executor).await?;
    Ok(())
}
```

Each tenant-scoped query method wraps in:

```rust
let mut tx = self.pool.begin().await?;
set_tenant_session(&mut *tx, ns.tenant_id()).await?;
sqlx::query("SELECT … FROM memory_items WHERE tenant_id = $1 …")
    .bind(ns.tenant_id())
    .execute(&mut *tx).await?;
tx.commit().await?;
```

`set_config(name, value, true)` mirrors `SET LOCAL` semantics —
the variable is scoped to the enclosing transaction. Connections
returning to the pool carry no leftover state (verified by the
testcontainers regression).

13 query sites refactored across `store.rs` (5),
`session_log.rs` (4 — including the existing multi-query
`append` transaction which gets `set_tenant_session` at the top
without an extra round-trip), and `checkpointer.rs` (4 —
`update_state` delegates to `put`).

### `Store::evict_expired` exception

`evict_expired` is the one query site without a tenant scope
— it sweeps every expired row across all tenants by design.
Under the policy installed here, it returns 0 rows when run by
a role *subject* to RLS (the variable is unset, no row matches).
Operators run TTL sweepers from a separate database role
configured with the `BYPASSRLS` attribute, scheduled outside
the per-request application path. The query body itself does
not change — the deployment-level role configuration determines
whether it actually deletes anything.

### Testcontainers regression

`tests/postgres_rls.rs` creates a second `NOSUPERUSER
NOBYPASSRLS` role inside the container, grants it CRUD on the
three tables, opens a second pool as that role, and verifies:

- SDK reads/writes work through the RLS-enforced role (proves
  the helper wires the variable correctly).
- Raw queries with no `entelix.tenant_id` set return 0 rows
  (proves the policy fail-closes on forgotten SET LOCAL).
- Raw queries with the wrong tenant return 0 rows (proves the
  policy filters per row).
- Raw queries with the correct tenant return the row (proves
  the policy doesn't over-restrict).
- INSERT with a row whose `tenant_id` differs from the session
  variable trips a `WITH CHECK` violation (proves the write
  side of the gate).
- Same gate applies to `session_events` and `checkpoints` (not
  just `memory_items`).

Default `postgres` role used by other testcontainers tests is
`SUPERUSER`, which bypasses RLS entirely. The existing test
suite (`postgres_integration.rs`,
`postgres_namespace_collision.rs`) runs as that role and
continues to pass without modification — they exercise the
application-layer correctness which RLS sits on top of.

## Consequences

✅ Defense in depth for invariant #11. Future query sites that
omit the tenant predicate, direct DBA access without the
session variable, and string-formatting mistakes all fail
closed at the database layer instead of leaking.
✅ No performance regression for normal app use — tenant-scoped
queries already had a `WHERE tenant_id = $1` clause; RLS adds
the same predicate as a row filter the planner can fold with
the existing index. The extra cost is one additional
round-trip per query (BEGIN + SET LOCAL + COMMIT vs. direct
pool execution).
✅ External tooling that needs to query the database directly
must either set `entelix.tenant_id` per query or run as a role
with `BYPASSRLS`. This explicit separation surfaces the
tenant-boundary contract instead of leaving it implicit.
✅ The SDK's existing testcontainers tests continue to pass —
no migration of existing tests required, RLS is additive at
both schema and code layers.
❌ One additional round-trip per single-query method. Pool
acquire + BEGIN + SET LOCAL + query + COMMIT. For tight loops
the latency adds up. Operators with extreme throughput
requirements can layer a connection-pinning wrapper on top
later (out of scope for this slice).
❌ `evict_expired` no longer works for the SDK's normal role
under RLS. Operators must run sweepers as a `BYPASSRLS` role.
Documented in the `Store::evict_expired` doc and the migration
file's preamble.
❌ Deployments running third-party tooling against the same
database silently broke until they apply the same SET LOCAL
discipline. The migration's preamble calls this out explicitly.
The alternative — leaving the policy off — would defeat the
purpose. Trade-off accepted as the right default for the
production-grade SDK.

## Alternatives considered

1. **Connection pinning per request (set the variable once
   per checkout, not per query)** — eliminates the per-query
   round-trip. But pool semantics complicate it: the SDK shares
   pool connections across requests, and `SET` (not `SET LOCAL`)
   leaks the variable between checkouts. The clean variant
   requires per-tenant connection partitioning, substantially
   more pool engineering. Deferred until throughput becomes a
   real bottleneck.
2. **Skip `FORCE ROW LEVEL SECURITY`** — works for non-owner
   roles. But typical deployments have the SDK role own the
   tables (it ran the migrations). Without `FORCE`, the
   protection wouldn't apply to the SDK's own pool. Rejected.
3. **Two policies (separate USING and WITH CHECK)** — same
   semantic as a single policy spanning both, but easier to
   accidentally drop one half during a future refactor. The
   single-policy form means both halves move together.
   Rejected.
4. **Opt-in via `PostgresPersistenceBuilder::with_rls(true)`**
   — adds dual code paths and contradicts the "no backwards-
   compatibility shims" invariant (#14). The migration's
   additive nature handles the deployment compatibility story:
   operators that haven't applied the migration aren't subject
   to RLS, and the SDK's `set_config` calls are harmless on a
   schema that has no policy. Rejected.

## Operator deployment notes

- The migration is additive; existing data is untouched.
- Roles subject to RLS must hold `SELECT, INSERT, UPDATE,
  DELETE` privileges on the three tables (these were already
  required to use the SDK).
- Maintenance roles that need cross-tenant access (TTL
  sweepers, audit / reporting tools) must hold the `BYPASSRLS`
  attribute or be `SUPERUSER`.
- `Store::evict_expired` is the only SDK method that needs
  cross-tenant access. Operators schedule it from a dedicated
  maintenance pool configured with a `BYPASSRLS` role.
- Direct DBA access for incident response: `SELECT
  set_config('entelix.tenant_id', '<tenant>', true);` at the
  start of an interactive session, then queries scope to that
  tenant for the rest of the transaction.

## References

- ADR-0017 — `tenant_id` mandatory + `Namespace` shape.
- ADR-0007 — `Store<V>` trait surface.
- ADR-0032 — invariant #15 (no silent fallback) — RLS variant
  fail-close behaviour aligns with the same principle at the
  persistence boundary.
- ADR-0040 — `list_namespaces` structural round-trip — the
  prior slice this depends on indirectly (parsing rendered keys
  back to typed `Namespace` for admin-tool consumers).
- 7-차원 roadmap §S8 — Phase 7 fourth sub-slice.
- Migration: `crates/entelix-persistence/migrations/20260501000001_row_level_security.sql`.
- Helper: `crates/entelix-persistence/src/postgres/tenant.rs`.
- Regression: `crates/entelix-persistence/tests/postgres_rls.rs`.
