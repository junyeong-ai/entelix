# ADR 0009 — swiftide is not a dependency

**Status**: Accepted
**Date**: 2026-04-26
**Decision**: D4

## Context

`swiftide` (`bosun-ai/swiftide`, 0.32.1, ~690★, 2025-11-15) is a Rust LLM ingestion + indexing + agents library. Its strengths overlap with the surface entelix is intentionally NOT building:

- Document loaders (file, URL, Slack, Notion, ...)
- Text splitters (chunkers)
- Embedder adapters (OpenAI, Voyage, Cohere)
- Vector store integrations (Qdrant, LanceDB)
- A small agent module

Question: should `entelix-memory` (or any entelix crate) take a hard dependency on `swiftide`?

## Decision

**No.** entelix has zero dependencies on `swiftide`.

Rationale:

1. **Layering mismatch**: entelix is an agent runtime + control flow + provider abstraction. swiftide is a RAG ingestion pipeline. Different layers; should be composable, not fused.
2. **Dependency weight**: pulling swiftide transitively brings `qdrant-client`, `tokenizers`, and other heavy crates into anyone using `entelix-memory` — even users who don't want RAG.
3. **User freedom**: forcing swiftide forces a particular RAG stack. Users wanting LanceDB-only, or self-hosted ES, or Pinecone, would be punished.
4. **Release cadence coupling**: swiftide is 0.x; coupling entelix's 1.0 release to swiftide's stability is risky.
5. **Trait surface is sufficient**: `Embedder`, `Retriever`, `VectorStore` traits (ADR-0008) let users plug swiftide if they want, without entelix having to know about swiftide.

## Recommended user pattern

A user who wants RAG-style semantic memory:

```rust
// User writes a small adapter (≤30 lines)
struct SwiftideEmbedderAdapter(swiftide::integrations::openai::OpenAI);

impl entelix_memory::Embedder for SwiftideEmbedderAdapter {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Embedding>, EmbedderError> {
        // delegate to swiftide
    }
    fn dimension(&self) -> usize { 1536 }
}

// Then plug into SemanticMemory:
let memory = SemanticMemory::new(SwiftideEmbedderAdapter(...), my_retriever);
```

The adapter is the user's code, in their crate. entelix never imports swiftide.

## 1.1 consideration: official `entelix-swiftide` crate?

**Possibly yes**, as a *separate companion crate* outside the entelix workspace OR in a sibling workspace. This would be:
- An **opt-in** crate, never pulled by `entelix` facade
- Maintained independently — its releases tied to swiftide's, not entelix's
- Useful as a "batteries-included" path for swiftide-aligned users

This is a 1.1+ decision. Not committed yet.

## Same logic for other RAG ecosystems

- `qdrant-client` — same pattern: trait + user adapter (or 1.1 companion `entelix-qdrant`)
- `lancedb` — same
- `langchain-rust` interop — explicitly **not** considered (different design philosophy)

## Consequences

✅ entelix surface stays focused on agent runtime.
✅ swiftide users keep their stack, adopt entelix incrementally.
✅ Non-RAG users pay nothing for RAG functionality.
✅ entelix 1.0 release is not blocked by swiftide stability.
❌ Users who want plug-and-play RAG out of the box will see an "adapter step."
❌ If 1.1 ships `entelix-swiftide`, we own a small ongoing maintenance burden.

## References

- ADR-0007 — Memory in 1.0
- ADR-0008 — Embedder/VectorStore/Retriever trait only
- swiftide README — `https://github.com/bosun-ai/swiftide`
