# ADR 0007 — Memory abstractions in 1.0

**Status**: Accepted
**Date**: 2026-04-26
**Decision**: D3 (사용자 재확인 후 포함)

## Context

Initial plan deferred Memory to 1.1. Re-analysis revealed this was wrong:

1. LangChain 1.0 deprecated its built-in Memory and **moved memory to LangGraph**.
2. LangGraph's published 4 core features include *"comprehensive memory: short-term working memory + cross-session long-term memory."*
3. Without long-term memory, no chatbot/CS agent/SaaS assistant works correctly across user sessions — users must re-establish context every time.

Anthropic's managed-agents shape covers session-level (per-thread) state via the event log. **Cross-session state is a separate concern** not covered by Session/Harness/Hand. LangGraph's `BaseStore` abstraction is the standard pattern.

## Three-tier state model (PLAN.md §5)

entelix recognizes **three orthogonal state tiers**:

| Tier | Lifetime | Crate | LG equivalent |
|---|---|---|---|
| StateGraph state | per-thread, working | `entelix-graph` | StateGraph + Checkpointer |
| SessionGraph events | per-thread, durable audit | `entelix-session` | (LG uses LangSmith externally) |
| Memory Store | cross-thread, persistent knowledge | `entelix-memory` | BaseStore |

Without all three, the SDK is incomplete.

## Decision

Add `entelix-memory` crate to the 1.0 milestone with the following surface:

### Trait surface
- `Store<V>` — namespace-keyed `get`/`put`/`list`/`search` for value type `V`
- `Embedder` — `embed(text, ctx) → Embedding` + `embed_batch(texts, ctx) → Vec<Embedding>` + `dimension() → usize`
- `Retriever` — `retrieve(query: RetrievalQuery, ctx) → Vec<Document>`
- `VectorStore` — namespace-scoped `add`/`search`/`delete`/`update`/`count`/`list` + `dimension`
- `Reranker` — re-order candidates by relevance
- `Document` — id + content + metadata + score
- `Embedding` + `EmbeddingUsage` — vector + token-accounting metadata

### Concrete implementations in 1.0 (zero external dependency)

These ride alongside the trait surface in `entelix-memory` itself — naming makes the no-external-dep guarantee visible:

- `InMemoryStore<V>` — in-process default (HashMap-backed)
- `InMemoryVectorStore` — brute-force cosine `VectorStore` (linear scan, every `VectorFilter` variant native, namespace-isolated)
- `InMemoryGraphMemory` — in-process `GraphMemory`
- `MmrReranker` — pure-math `Reranker` (Maximal Marginal Relevance)
- `BufferMemory` — sliding window of last N turns
- `SummaryMemory` — auto-summarize old turns via LLM call
- `EntityMemory` — extracted facts via LLM call
- `ConsolidatingBufferMemory` + `ConsolidationPolicy` — policy-driven buffer→summary transition
- `SemanticMemory<E: Embedder, R: Retriever>` — generic over plug-ins

### Concrete vendor-SDK impls — companion crates (ADR-0008)

Concrete impls that ride a vendor SDK or HTTP API live in `entelix-{role}-{vendor}` companion crates so the trait crate carries no transitive cost:

- `entelix-memory-openai` — `OpenAiEmbedder` (text-embedding-3-{small,large}, native dimension reduction via `dimensions` API parameter, batch endpoint coalescing, F10 amortization)
- (future) `entelix-memory-voyage`, `entelix-memory-cohere`, `entelix-memory-qdrant`, `entelix-memory-lancedb`

### Persistence backends
- `PostgresStore<V>` — in `entelix-persistence`
- `RedisStore<V>` — in `entelix-persistence`

### Multi-tenant safety (F2 mitigation)

`Namespace` requires `tenant_id`:

```rust
pub struct Namespace {
    pub tenant_id: TenantId,
    pub scope: Vec<Cow<'static, str>>,
}

impl Namespace {
    pub fn new(tenant_id: TenantId, scope: impl IntoIterator<Item = impl Into<Cow<'static, str>>>) -> Self { ... }
}
```

There is no constructor without `tenant_id`. Cross-tenant data leak is structurally impossible — verified end-to-end against Postgres + Redis (`postgres_namespace_collision.rs` + `redis_isolation.rs`, Invariant 13).

## Consequences

✅ Honest "LG-class" claim — Memory present, including cross-session.
✅ Three tiers cleanly separated; no conflation.
✅ Multi-tenant safety enforced at API boundary, not by convention.
✅ Zero-dep concrete impls (`InMemoryVectorStore`, `InMemoryStore`, …) close the "missing batteries" gap for tests and small-corpus deployments without dragging in vendor SDKs.
✅ Vendor-SDK impls live in companion crates — operators opt in per backend.
❌ Operators wanting `OpenAiEmbedder` add a second dep (`entelix-memory-openai`) — accepted tradeoff for keeping the trait crate lean.
❌ One additional crate (`entelix-memory`) — already accepted in ADR-0006.

## Alternatives considered

1. **Skip Memory in 1.0, defer to 1.1** — original plan. Rejected as analysis showed it's a 1.0-blocking gap.
2. **Fold Memory into `entelix-graph`** — conflates per-thread with cross-thread state; violates invariant 3.
3. **Ship every concrete Embedder in `entelix-memory` itself** — rejected (ADR-0008): trait crate stays vendor-SDK-free, companion crates own each vendor.

## References

- LangGraph overview — comprehensive memory as core feature
- LangChain 1.0 release notes — Memory deprecated, moved to LG
- PLAN.md §5 — three-tier state model
- CLAUDE.md invariants 3, 11, 13
- ADR-0008 — trait surface in entelix-memory, concrete impls as siblings
