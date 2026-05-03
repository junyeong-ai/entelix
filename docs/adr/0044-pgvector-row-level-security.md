# ADR 0044 — `entelix-memory-pgvector` row-level security

**Status**: Accepted
**Date**: 2026-05-01
**Decision**: Phase 7 of the post-7-차원-audit roadmap (seventh sub-slice)

## Context

ADR-0042 + ADR-0043 walked the companion-crate RLS pattern
through `entelix-graphmemory-pg`. ADR-0042 introduced the
crate; ADR-0043 added RLS as defense in depth for invariant #11.
The pattern was deliberately framed as the template for other
Postgres-backed companion crates.

`entelix-memory-pgvector` is the second such companion. It
shipped without RLS (single-table `entelix_vectors`, composite
`(namespace_key, doc_id)` PK, GIN + HNSW indexes — ADR-0008's
trait + companion shape). The application-layer
`WHERE namespace_key = $1` anchor is already on every query, so
the boundary is correct as written — but the same three failure
modes ADR-0041 / ADR-0043 enumerated apply:

1. Forgotten `WHERE namespace_key = $1` in a future query site.
2. Direct DBA / external tooling access without the anchor.
3. SQL injection / refactor mistakes that bypass the anchor.

This slice closes that gap by mirroring ADR-0043's design
exactly — `tenant_id TEXT NOT NULL` column populated from
`Namespace::tenant_id()`, `ENABLE` + `FORCE ROW LEVEL SECURITY`,
single `tenant_isolation` policy spanning USING + WITH CHECK on
`current_setting('entelix.tenant_id', true)`, every query site
wrapped in a `pool.begin → set_tenant_session → query → commit`
transaction. With this slice, every Postgres-backed companion in
the workspace has the same RLS surface — pattern complete.

## Decision

### Schema additions (migration `bootstrap`)

`entelix_vectors` gains `tenant_id TEXT NOT NULL` populated on
INSERT from `Namespace::tenant_id()`. The denormalised column
lets the policy filter rows with a simple equality the planner
folds into the index scan, instead of a function call against
`namespace_key` per row (ADR-0043 §"Why denormalised").

`enable_rls(table)` mirrors ADR-0043's helper — `ENABLE` +
`FORCE` + `DROP POLICY IF EXISTS` + `CREATE POLICY tenant_isolation`
spanning USING + WITH CHECK. `DROP POLICY IF EXISTS` keeps the
runtime-DDL bootstrap idempotent across re-runs (the
`with_auto_migrate(false)` path skips both schema and policy
changes).

### `set_tenant_session` helper (per-companion, again)

Replicated identically from ADR-0043's design — 6-line function
calling `SELECT set_config('entelix.tenant_id', $1, true)`. The
rationale stays:

- `entelix-memory` is sqlx-free (ADR-0008).
- Inventing a new utility crate for one helper is over-engineered.
- The SQL is one literal; behaviour drift between companions is
  not a real risk.

### Query-site wrapping (6 sites)

`add` / `batch_add` / `search_filtered` / `delete` / `count` /
`list` all wrap in `pool.begin() → set_tenant_session → query →
commit`. `update` delegates to `add`; `search` delegates to
`search_filtered`. The `batch_add` path captures `ns.tenant_id()`
once before pushing values, then binds it per row inside the
`QueryBuilder::push_values` callback — matches the existing
shape where `ns_key.clone()` is bound per row.

### Cancellation check ordering

Each method's `if ctx.is_cancelled() { return Err(Cancelled) }`
guard stays at the top of the function, before the `pool.begin()`.
Cancellation should be cheap on the cancellation-fired path, and
opening a transaction just to cancel it is wasteful — keeping the
guard above the BEGIN preserves the cheap fast path.

### Testcontainers regression — `tests/pgvector_rls.rs`

Mirror of `entelix-graphmemory-pg/tests/postgres_rls.rs`, adapted
for the `pgvector/pgvector:pg17` image (the existing e2e suite's
container choice). Three cases:

1. SDK reads/writes succeed through the NOSUPERUSER role; raw
   reads with no `entelix.tenant_id` set return 0 rows
   (forgotten SET LOCAL → fail-closed).
2. Raw INSERT with mismatched `tenant_id` (set tenant-A in
   session, INSERT row with tenant-B) trips `WITH CHECK`
   violation.
3. `count_in_tx` with the correct vs. wrong tenant returns 1
   vs. 0 rows.

Existing `tests/pgvector_e2e.rs` runs as the SUPERUSER `postgres`
role which bypasses RLS, so it continues to pass without
modification — exercises the application-layer correctness which
RLS sits on top of.

## Consequences

✅ Defense in depth for invariant #11 on the vector backend.
The companion-crate RLS pattern is now consistent across both
Postgres-backed companions (`entelix-graphmemory-pg`,
`entelix-memory-pgvector`). Any future Postgres companion follows
the same template.
✅ Forgotten SET LOCAL, direct DBA access without the variable,
and any future query site that omits the namespace anchor all
fail closed at the database layer.
✅ The bootstrap is idempotent; re-running picks up RLS for
deployments that exist on the prior schema. Pre-existing rows
with `NULL` tenant_id need a one-shot backfill (same SQL as
ADR-0043 §"Operator deployment notes").
✅ Existing `tests/pgvector_e2e.rs` runs unchanged — RLS is
additive at both schema and code layers.
❌ One additional round-trip per single-query method (BEGIN +
SET LOCAL + COMMIT). Same trade-off ADR-0041 + ADR-0043
documented; same future-optimization path (connection-pinning
wrapper).
❌ Pre-existing rows must be backfilled with `tenant_id`
extracted from `namespace_key`. The brand-new shipping window
for ADR-0042 / ADR-0043 minimises this; pgvector has been live
longer (since ADR-0008's first companion-crate slice), so
operators with existing rows must run the backfill before the
re-run of `bootstrap` activates RLS.
❌ `qdrant` companion remains without an equivalent gate —
qdrant is a separate vector DB with no RLS concept; isolation
there relies on the `must`-anchor `entelix_namespace_key` payload
field (ADR-0008's qdrant slice). The "companion-family RLS
unification" thus covers only Postgres-backed companions; qdrant
is structurally different.

## Alternatives considered

1. **Reuse `entelix-graphmemory-pg::tenant::set_tenant_session`** —
   would force a cross-companion dependency. Rejected; both
   crates ship as siblings under `entelix-memory`, neither
   should depend on the other.
2. **Promote `set_tenant_session` to `entelix-core`** — adds
   `sqlx` to the DAG root. Hard rejected; invariant 9-ish
   "core stays minimal" + ADR-0008's "core has no vendor deps".
3. **Skip the cancellation-fast-path check, wrap everything
   uniformly inside the tx** — symmetrical but slower for
   callers that pre-cancel. Rejected; the existing fast path
   has a real reason and stays.
4. **Use `ALTER TABLE … ADD COLUMN tenant_id TEXT NOT NULL
   DEFAULT ''` to ease backfill** — `DEFAULT ''` would let
   existing rows insert with empty `tenant_id` matching nothing,
   silently hiding them under RLS. Rejected; explicit backfill
   is the safer story (operator gets a clear failure if they
   skip it).

## Operator deployment notes

- Operators upgrading from the prior pgvector ship: re-run
  `bootstrap` (auto on next `build()` with `auto_migrate=true`).
  **Backfill existing rows** before the next live query:

  ```sql
  UPDATE entelix_vectors
  SET tenant_id = split_part(namespace_key, ':', 1)
  WHERE tenant_id = '';
  ```

  (Assumes no `:` characters in tenant segments — the
  `Namespace::render` escape contract guarantees this for any
  tenant_id minted through `Namespace::new`.)
- Same `BYPASSRLS` role pattern as ADR-0041 for any future
  cross-tenant maintenance ops on this table.
- Direct DBA access:
  `SELECT set_config('entelix.tenant_id', '<tenant>', true);`
  at the start of an interactive session.
- For deployments that prefer not to use RLS, run
  `with_auto_migrate(false)` and own a no-RLS schema externally.
  The SDK still wraps every query in `set_config` calls — those
  are harmless on a schema without the policy.

## References

- ADR-0007 — `VectorStore` trait surface (parent).
- ADR-0008 — companion crate pattern.
- ADR-0017 — `Namespace` + tenant_id mandatory.
- ADR-0032 — invariant #15 (no silent fallback).
- ADR-0041 — `entelix-persistence` RLS (the original pattern).
- ADR-0043 — `entelix-graphmemory-pg` RLS (the per-companion
  template this slice mirrors).
- 7-차원 roadmap §S8 — Phase 7 seventh sub-slice.
- Migration: `crates/entelix-memory-pgvector/src/migration.rs`
  (`enable_rls` helper).
- Helper: `crates/entelix-memory-pgvector/src/tenant.rs`
  (`set_tenant_session`).
- Regression: `crates/entelix-memory-pgvector/tests/pgvector_rls.rs`
  (3 testcontainers cases).
