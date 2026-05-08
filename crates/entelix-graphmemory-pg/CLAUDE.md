# entelix-graphmemory-pg

Companion crate. Concrete `GraphMemory<N, E>` impl backed by Postgres (sqlx) — production graph-memory tier with row-level security, single-SQL BFS via `WITH RECURSIVE`, and `INSERT … SELECT FROM UNNEST` bulk insert.

## Surface

- **`PgGraphMemory<N, E>`** + **`PgGraphMemoryBuilder`** — `with_pool(pool)` / `build() -> Result<Self>`. Two-table schema (`graph_nodes` + `graph_edges`), composite `(namespace_key, id)` PK, JSONB payload columns.
- **Inherent admin methods** (NOT on the `GraphMemory` trait) — `list_nodes` / `list_node_records` / `list_edges` / `list_edge_records` / `prune_orphan_nodes`. Operator-time admin paths live on the backend type, not the agent-facing trait surface.
- **`PgGraphMemoryError`** — typed error wrapping `sqlx::Error`, `serde_json::Error`, plus `Malformed` / `Config`.

## Crate-local rules

- **Row-level security mandatory** — `tenant_id` column + `FORCE ROW LEVEL SECURITY` + `tenant_isolation` policy. Every query runs inside a transaction wrapped by `set_tenant_session(tenant_id)`. Mirrors the `entelix-persistence` RLS pattern.
- **`traverse` + `find_path` are single-SQL** — `WITH RECURSIVE` BFS with per-row `visited` array for cycle prevention. D round-trips → 1 round-trip regardless of depth.
- **`add_edges_batch` is single-SQL** — `INSERT … SELECT FROM UNNEST(.)` binds per-column arrays. 10k edges on 5ms-RTT Postgres: 50s → 5ms (3 orders of magnitude reduction).
- **Backend isolation tests at the persistence layer** (invariant 13) — testcontainers-driven cross-tenant collision suite.
- **`N: Serialize + DeserializeOwned, E: Serialize + DeserializeOwned`** — node/edge payloads are JSONB-serialized. The trait bound is enforced at the inherent admin methods (per-method `where` clause, not on the struct).

## Forbidden

- A query path that bypasses `set_tenant_session` — RLS won't fire.
- A method on the `GraphMemory` trait surface that's actually backend-specific (admin / pagination). Trait surface stays at the agent-facing CRUD.
