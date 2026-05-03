# ADR 0008 — Memory: trait surface in `entelix-memory`, concrete impls as siblings

**Status**: Accepted
**Date**: 2026-04-26

## Context

ADR-0007 brings Memory abstractions into 1.0. The natural next question: where do concrete `OpenAIEmbedder`, `VoyageEmbedder`, `QdrantVectorStore`, `LanceDbVectorStore` etc. live?

Trade-offs:

**Pro folding concrete impls into `entelix-memory`**:
- Single dep
- Lower friction for new users — `entelix-memory = "..."` brings RAG online

**Con folding concrete impls into `entelix-memory`**:
- Each provider adds dependencies (`reqwest` calls, auth flows, version churn) to every user
- Vector DB clients (`qdrant-client`, `lancedb`) have their own ecosystems and release cadences
- 1.0 API freeze becomes harder — every concrete impl is a stability commitment
- Surface bloat that users who don't need it still pay for in compile time
- Embedder/VectorStore impls overlap heavily with `swiftide` (ADR-0009)

## Decision

`entelix-memory` is **trait-and-policy oriented**. It defines:

- `Embedder`, `Retriever`, `VectorStore`, `Store`, `Reranker` traits
- `Document`, `Embedding`, `EmbeddingUsage`, `RetrievalQuery`, `Namespace` data types
- `SemanticMemory<E: Embedder, R: Retriever>` generic struct
- `BufferMemory`, `SummaryMemory`, `EntityMemory`, `GraphMemory`, `ConsolidatingBufferMemory` policies

It also ships **one** concrete implementation per backend boundary that does **not** require an external SDK / network dependency:

- `InMemoryStore` — in-process `Store<V>` (no I/O)
- `InMemoryGraphMemory` — in-process `GraphMemory`
- `InMemoryVectorStore` — brute-force cosine `VectorStore` (single linear scan, no ANN)
- `MmrReranker` — pure-math `Reranker`

These zero-dep impls let tests, dev loops, and small-corpus production deployments wire the full `SemanticMemory` pipeline without pulling a vector DB or embedding-API client. The `*` prefix is `InMemory*` — naming makes the no-external-dep guarantee visible at the call site.

**Concrete impls that ride a vendor SDK or HTTP API live in their own crates**, named `entelix-{role}-{vendor}` per ADR-0010:

- `entelix-memory-openai` — `OpenAiEmbedder` (text-embedding-3-{small,large}, requires `reqwest` + bearer auth)
- `entelix-memory-qdrant` — `QdrantVectorStore` (thin wrapper over `qdrant-client`; single-collection multi-tenancy with payload-anchor)
- `entelix-memory-pgvector` — `PgVectorStore` (sqlx + `pgvector` extension; single-table multi-tenancy with `(namespace_key, doc_id)` composite PK; auto-migrate opt-out for IaC-managed schemas)
- (future) `entelix-memory-voyage` — `VoyageEmbedder`
- (future) `entelix-memory-cohere` — `CohereEmbedder`
- (future) `entelix-memory-lancedb` — `LanceDbVectorStore` (thin wrapper over `lancedb`)

Companion crates are **never** required by the entelix facade — strictly opt-in via feature flags (`embedders-openai`, etc.) for users who want plug-and-play.

## Trait shapes (frozen in 1.0)

```rust
pub trait Embedder: Send + Sync + 'static {
    fn dimension(&self) -> usize;
    async fn embed(&self, text: &str, ctx: &ExecutionContext) -> Result<Embedding>;
    async fn embed_batch(&self, texts: &[String], ctx: &ExecutionContext) -> Result<Vec<Embedding>>;
}

pub trait VectorStore: Send + Sync + 'static {
    fn dimension(&self) -> usize;
    async fn add(&self, ctx: &ExecutionContext, ns: &Namespace, doc: Document, vector: Vec<f32>) -> Result<()>;
    async fn search(&self, ctx: &ExecutionContext, ns: &Namespace, query: &[f32], top_k: usize) -> Result<Vec<Document>>;
    async fn delete(&self, ctx: &ExecutionContext, ns: &Namespace, doc_id: &str) -> Result<()>;
    // … filters, update, count, list — see entelix-memory::traits.
}

pub trait Retriever: Send + Sync + 'static {
    async fn retrieve(&self, query: RetrievalQuery, ctx: &ExecutionContext) -> Result<Vec<Document>>;
}
```

Signatures are committed in 1.0. Changes are major-version events.

## Consequences

✅ 1.0 surface stays bounded — easier to freeze and audit per crate.
✅ Users have full freedom — plug `swiftide`, plug self-built, plug `qdrant-client` directly.
✅ No transitive deps on vector DB or embedding-model SDKs in `entelix-memory` itself.
✅ Companion crates can iterate independently — `entelix-memory-openai` releasing v1.1 does not bump `entelix-memory`.
✅ In-process impls (`InMemoryVectorStore`, `InMemoryStore`) close the "missing batteries" gap for tests and small-corpus deployments without dragging in vendor SDKs.
❌ Operators wanting `OpenAiEmbedder` add a second dep (`entelix-memory-openai`) — accepted tradeoff for keeping the trait crate lean.

## References

- ADR-0007 — Memory in 1.0 (this ADR's parent decision)
- ADR-0009 — swiftide non-dependency (related rationale)
- ADR-0010 — naming taxonomy (`entelix-{role}-{vendor}` companion pattern)
