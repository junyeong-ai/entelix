# ADR 0001 — Workspace structure: 18 crates, not one

**Status**: Accepted
**Date**: 2026-04-26

## Context

A single-crate-with-many-feature-flags layout, scaled, accumulates:
- hundreds of .rs files,
- 100K+ LoC,
- 700+ public types,
- a `cargo public-api` diff long enough to take minutes to read.

Single-crate + many features creates two failure modes:
1. **Cognitive load** — users opening rustdoc see all 711 types, even when their feature set uses only ~50.
2. **Compile-time amplification** — every feature combination is a unique compilation unit; CI matrix explodes.

## Decision

entelix is a **Cargo workspace with 18 crates**. Each crate represents a deployment surface:

```
entelix                — umbrella facade (re-exports behind feature flags)

# DAG root
entelix-core           — IR, codec, transport, agent, tool trait, hooks, auth

# composition + control flow
entelix-runnable       — Runnable trait, LCEL pipe, Sequence/Parallel/Router/Lambda
entelix-prompt         — PromptTemplate, ChatPromptTemplate, FewShot, ExampleSelector
entelix-graph          — StateGraph, reducer, conditional edges, Send, Checkpointer trait

# state / memory
entelix-session        — SessionGraph (event log), fork, archival watermark
entelix-memory         — Store/Embedder/Retriever/VectorStore traits + InMemoryVectorStore + Buffer/Summary/Entity
entelix-memory-openai  — concrete OpenAI Embedder (text-embedding-3-{small,large})
entelix-memory-qdrant  — concrete VectorStore (qdrant-client gRPC; multi-tenant via payload anchor)
entelix-memory-pgvector— concrete VectorStore (sqlx + pgvector; multi-tenant via composite-PK anchor)
entelix-persistence    — Postgres + Redis impls (Checkpointer + Store + SessionLog), distributed lock

# action surface
entelix-tools          — http_fetch, search adapter trait, calculator
entelix-mcp            — MCP client (JSON-RPC 2.0 over HTTP, per-tenant pool)
entelix-cloud          — Bedrock + Vertex + Foundry transports

# operational
entelix-policy         — multi-tenant, rate-limit, PII, cost meter, quota
entelix-otel           — OpenTelemetry GenAI semconv
entelix-server         — axum HTTP server, 5-mode SSE streaming

# recipes
entelix-agents         — ReAct, Supervisor, Hierarchical, Chat pre-built recipes
```

**Constraints**:
- `entelix-core` depends on no other entelix crate (DAG root).
- Cross-crate dependencies form a strict DAG (cycles forbidden, enforced by Cargo).
- The umbrella `entelix` crate re-exports each member behind a feature flag of the same name.
- Vendor-specific concrete impls follow the `entelix-{role}-{detail}` naming pattern (e.g. `entelix-memory-openai`); the role crate stays trait-and-policy oriented while concrete impls live in their own siblings (ADR-0008).

## Consequences

✅ Users add only the crates they need; rustdoc surface is bounded by their dependencies.
✅ Each crate has independent semver. `entelix-core 1.0` can ship while `entelix-server` is 0.7.
✅ External integrators (e.g., restate-sdk wrapper) depend on `entelix-core` + `entelix-graph` directly, skipping axum/policy/otel.
✅ CI parallelizes per-crate test matrix.
✅ Vendor-specific dependencies (`reqwest` for OpenAI embedder) are isolated to companion crates — the trait crate carries no transitive cost for users who plug their own backend.
❌ Workspace versioning requires care — bumping `entelix-core` forces all dependents to bump.
❌ More boilerplate (18 Cargo.toml, 18 lib.rs) — accepted tradeoff.

## Alternatives considered

1. **Single crate, ~6 features** (compromise) — still mixes deployment surfaces in one rustdoc namespace. Rejected.
2. **Two crates: core + extras** — too coarse; `entelix-extras` would be a 600-type dumping ground. Rejected.
3. **Per-codec crates (rig pattern: rig-anthropic, rig-openai, etc.)** — codecs are small (~500 LoC each) and share the IR; splitting adds release overhead without ergonomic gain. The five codecs live together in `entelix-core::codecs`. Per-vendor splitting is reserved for surfaces where the vendor SDK pulls a heavy dep tree (`entelix-memory-openai`, `entelix-cloud`'s Bedrock/Vertex/Foundry).

## References

- tonic, axum, swiftide-core/swiftide-agents — workspace examples
- rig (`rig-core`, `rig-bedrock`, `rig-cohere` …) — partial precedent
