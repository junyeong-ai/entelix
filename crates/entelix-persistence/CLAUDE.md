# entelix-persistence

Postgres + Redis backends for `Checkpointer` (entelix-graph) + `Store` (entelix-memory) + `SessionLog` (entelix-session) + `DistributedLock` (this crate). Aggregate facade `*Persistence` carries a wired pool plus the trait impls.

## Surface

- **`PostgresPersistence` + `PostgresPersistenceBuilder`** (feature `postgres`) — `with_connection_string(url)` / `connect()` returns the wired aggregate. Exposes `PostgresCheckpointer` / `PostgresStore` / `PostgresSessionLog` / `PostgresLock` siblings as separate handles.
- **`RedisPersistence` + `RedisPersistenceBuilder`** (feature `redis`) — same shape, Redis-backed. `RedisCheckpointer` / `RedisStore` / `RedisSessionLog` / `RedisLock`.
- **`DistributedLock` trait** + `LockGuard` + `with_session_lock(persistence, tenant, thread, &fn)` — distributed advisory lock keyed on `(tenant_id, thread_id)`. `PostgresLock` uses `pg_advisory_xact_lock(hash(tenant, thread))`; `RedisLock` uses `SET NX PX` with a Lua release script.
- **`AdvisoryKey`** — `for_session(tenant, thread)` derives a 64-bit advisory key + a Redis-compatible hex string. `from_strings(parts)` for non-session locks. NUL-separator + xxh3 64-bit hash.
- **`SessionSchemaVersion`** — `validate(self) -> PersistenceResult<()>` checks the persisted version against the build's `[MIN_SUPPORTED_VERSION, CURRENT_VERSION]` window. Loud on mismatch — never silently downgrade.
- **`PersistenceError` + `PersistenceResult`** — typed errors (`Sqlx`, `Redis`, `Backend`, `Config`, `Serde`, `SchemaVersionMismatch`).

## Crate-local rules

- **Row-level security mandatory on every Postgres backend** (ADR-0041). Each table has a `tenant_id` column + `FORCE ROW LEVEL SECURITY` + `tenant_isolation` policy. Every query runs inside a transaction wrapped by `set_tenant_session(tenant_id)` so the policy actually fires. Removing the wrap is an instant-reject review comment.
- **Distributed lock is mandatory for cross-pod safety** (CLAUDE.md §"Lock ordering"). `with_session_lock` is the only sanctioned mutation path when the same `thread_id` may be racing on multiple pods. Skipping it because "this deployment is single-pod" is a F8 reintroduction.
- **Backend isolation tests at the persistence layer** (invariant 13). `tests/postgres_namespace_collision.rs` + `tests/redis_isolation.rs` exercise multi-tenant collision under the real backend (testcontainers). In-memory mock tests do not satisfy this invariant.
- **Schema version on every persisted shape** — `SessionSchemaVersion` is stamped on every event row. A missing version is a hard error, not a "default to current" silent migration (invariant 15 — no silent fallback).
- **`PostgresPersistence::builder` returns `Result<Self>` from `connect()`, not from `build()`** — connection failure is the failure mode, builder validation is infallible.

## Forbidden

- A query path that bypasses `set_tenant_session` — RLS won't fire, cross-tenant data leak.
- A lock release path that drops the lock outside the transaction (Postgres) or without the Lua release script (Redis) — race-window for double-acquire.
- Caching `tenant_id` on the connection beyond the transaction scope — connection pool reuse would leak the previous request's tenant context.
- An `[entelix-persistence]` direct dependency from any sub-crate other than the facade — every other crate pulls the relevant trait through `entelix-graph` / `entelix-memory` / `entelix-session`.

## References

- ADR-0041 — Postgres row-level security (`set_tenant_session` + `tenant_isolation` policy).
- ADR-0044 — pgvector RLS extension (companion-family uniformity).
- ADR-0064 — 1.0 release charter (`postgres` / `redis` are facade features that pass-through to this crate's features).
- F8 mitigation — distributed lock for cross-pod thread-id mutation.
