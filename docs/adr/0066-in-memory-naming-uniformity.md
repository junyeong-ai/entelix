# ADR 0066 — `InMemory*` prefix for in-memory backends

**Status**: Accepted
**Date**: 2026-05-02
**Decision**: Naming-taxonomy supplement — every in-memory implementation of a workspace trait carries the `InMemory*` prefix.

## Context

Five workspace traits ship an in-memory default impl:

| Crate | Trait | In-memory impl |
|---|---|---|
| `entelix-memory` | `Store<V>` | `InMemoryStore<V>` |
| `entelix-memory` | `VectorStore` | `InMemoryVectorStore` |
| `entelix-memory` | `GraphMemory<N, E>` | `InMemoryGraphMemory<N, E>` |
| `entelix-graph` | `Checkpointer<S>` | `InMemoryCheckpointer<S>` |
| `entelix-session` | `SessionLog` | `InMemorySessionLog` |

A naming convention is needed so future in-memory backends pick the right prefix without a per-PR debate. Two ambiguities pin the choice:

1. **"Memory" overloads with the memory tier itself.** `entelix-memory` ships `BufferMemory`, `EntityMemory`, `SemanticMemory`, `EpisodicMemory`, `SummaryMemory`, `GraphMemory` — higher-level *memory patterns* over a `Store<V>`. A `Memory*` prefix on backends would read "the memory tier's store" rather than "an in-process backend".
2. **Operator migration.** Python `langchain_community` uses `InMemoryVectorStore` / `InMemoryRetriever` / `InMemoryDocumentLoader`. Operators porting from Python recognise the `InMemory*` prefix instantly.

## Decision

Every in-process default impl of a workspace trait — present or future — uses the `InMemory*` prefix. The prefix is reserved: types using `InMemory*` MUST be in-process / non-persistent backends of a public trait.

`docs/adr/0010-naming-taxonomy.md` references this ADR; `.claude/rules/naming.md` mirrors the rule for CI / reviewer enforcement.

### Why `InMemory*` over alternatives

- **`Mem*` (e.g. `MemStore`)** — too terse, loses the "in-process / non-persistent" signal. Rejected.
- **`Local*` (e.g. `LocalStore`)** — confusable with "deployment-local" (vs. cloud) rather than "in-process". `In*` is the standard English idiom for "in memory". Rejected.
- **No prefix (`Store` itself is the in-memory impl)** — collides with the trait of the same name; impossible for `Store`, `Checkpointer`, `SessionLog`, `VectorStore`, `GraphMemory`. Rejected.

`InMemory*` matches the Python LangChain / LangGraph naming convention and the Java Collections Framework's `InMemoryDataSource` shape — both bodies of prior art.

## Consequences

- Five in-memory backends share one prefix; reviewers reject `MemFoo` / `LocalFoo` on the same trait shape.
- "Memory" reserved for the memory tier. Type names disambiguate "memory tier" from "in-memory backend" by design.
- Aligns with Python LangChain / LangGraph naming — the migration guide reads naturally.

## References

- ADR-0007 — memory in 1.0 (`InMemoryStore` ships here).
- ADR-0010 — naming taxonomy (this ADR supplements the type-suffix table with the `InMemory*` prefix rule).
- `crates/entelix-memory/src/store.rs` — `InMemoryStore<V>`.
- `crates/entelix-graph/src/in_memory_checkpointer.rs` — `InMemoryCheckpointer<S>`.
- `crates/entelix-session/src/log.rs` — `InMemorySessionLog`.
