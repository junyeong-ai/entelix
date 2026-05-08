# entelix-memory

Tier-3 cross-thread persistent knowledge. Trait surface + zero-dependency reference impls. Concrete vendor-backed impls ship as companion crates.

## Surface

- **`Namespace`** — `Namespace::new(tenant_id)` (mandatory non-empty `tenant_id`, runtime-asserted) + `with_scope(segment)` builder for nested scopes. No constructor exists that omits `tenant_id` (invariant 11).
- **`Store<V>` trait** — KV with TTL (`PutOptions { expires_at }`). Methods take `(&self, ctx, namespace, …)` (ctx-first per naming taxonomy).
- **`VectorStore` trait** — vector + metadata storage. `add` / `search_filtered` keyed on `Namespace`. Reference: `InMemoryVectorStore` (brute-force cosine, namespace-isolated).
- **`Embedder` trait** + `MeteredEmbedder<E>` — `Arc<Self>` constraint (F10 — pool-shared, no per-call client construction). `MeteredEmbedder` records `gen_ai.embedding.cost` only on `Ok` (invariant 12).
- **`Retriever` trait** + `Reranker` trait + `MmrReranker` — diversity-aware retrieval composition.
- **Memory patterns** — `BufferMemory`, `SummaryMemory`, `EntityMemory`, `SemanticMemory<E, V>`, `EpisodicMemory<V>`, `ConsolidatingBufferMemory` (LangChain-style facades over `Store<V>`).
- **`GraphMemory<N, E>` trait** + `InMemoryGraphMemory<N, E>` — typed-node + timestamped-edge knowledge graph (BFS traversal, shortest-path). Postgres-backed `PgGraphMemory` companion folds traversal into a single `WITH RECURSIVE` round-trip.

## Companion crates (1.0)

| Crate | Backend | Use case |
|---|---|---|
| `entelix-memory-openai` | OpenAI Embeddings API | text-embedding-3-{small,large} |
| `entelix-memory-qdrant` | qdrant gRPC | production vector store, single-collection multi-tenancy |
| `entelix-memory-pgvector` | Postgres pgvector | production vector store, SQL filter projection |
| `entelix-graphmemory-pg` | Postgres + WITH RECURSIVE | production `GraphMemory` with FORCE RLS + UNNEST bulk insert |

Companion-crate pattern: live in `entelix-memory-<vendor>`, depend on `entelix-memory` for the trait, never modify the trait surface from outside.

## Crate-local rules

- **`Namespace` always carries `tenant_id`.** Convenience constructors that drop it are bugs. The render path includes the tenant prefix unconditionally.
- **Backend impls must run namespace-collision tests at the persistence layer** (testcontainers or equivalent) — invariant 13. Reference suite: `tests/namespace_collision.rs`. Backend-specific suites: `entelix-persistence/tests/postgres_namespace_collision.rs`, etc.
- New `Embedder` impl: constrain to `Arc<Self>`. Per-call client construction is an instant-reject review comment.
- Retrieval cost: emit only inside `Ok` (mirror `MeteredEmbedder` shape).

## Forbidden

- A `Namespace` constructor without `tenant_id`.
- Vector store impl whose filter projection drops the namespace anchor "for performance" — namespace is the security boundary, not a hint.
- New companion concrete impls in `entelix-memory` itself — they go in `entelix-memory-<vendor>`.

