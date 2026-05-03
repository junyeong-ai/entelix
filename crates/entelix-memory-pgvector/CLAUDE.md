# entelix-memory-pgvector

Companion crate. Concrete `VectorStore` impl backed by Postgres + pgvector — production vector store with row-level security, SQL filter projection, and operator-tunable distance metric + index kind.

## Surface

- **`PgVectorStore`** + **`PgVectorStoreBuilder`** — `with_pool(pool)` / `with_dimension(d)` / `with_distance_metric(m)` / `with_index_kind(k)` / `build() -> Result<Self>`.
- **`DistanceMetric`** (re-exported as `PgVectorDistanceMetric` from facade) — `Cosine` / `L2` / `InnerProduct` (pgvector operators `<=>`, `<->`, `<#>`).
- **`IndexKind`** (re-exported as `PgVectorIndexKind`) — `Hnsw { m, ef_construction }` / `IvfFlat { lists }` / `None`. Operator picks based on dataset size + write/read ratio.
- **`PgVectorStoreError`** — typed error including `FilterProjection` for SQL filter parse failures.

## Crate-local rules

- **Row-level security mandatory** (ADR-0044) — `tenant_id` column + `enable_rls` migration helper + `set_tenant_session(tenant_id)` per-call wrap. Mirrors `entelix-persistence` and `entelix-graphmemory-pg` RLS pattern (companion-family uniformity).
- **Filter projection preserves namespace anchor** — operator-supplied `VectorFilter` is composed AS A WHERE clause on top of the namespace constraint, never replaces it. The namespace is the security boundary, not a hint (entelix-memory CLAUDE.md rule).
- **Distance metric default = `Cosine`** — most embedding models (OpenAI, Cohere) ship normalized vectors where cosine === inner product but cosine is what every downstream tool assumes.
- **Backend isolation tests at the persistence layer** (invariant 13) — testcontainers-driven cross-tenant search test.

## Forbidden

- A query path that bypasses `set_tenant_session` — RLS won't fire, cross-tenant vector retrieval.
- A filter projection that drops the namespace anchor "for performance" — security boundary violation.

## References

- ADR-0044 — pgvector RLS extension (companion-family RLS uniformity).
- ADR-0008 — companion crate pattern.
