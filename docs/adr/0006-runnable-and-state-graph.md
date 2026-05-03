# ADR 0006 — Runnable trait + StateGraph as 1.0 spine

**Status**: Accepted
**Date**: 2026-04-26
**Decision**: D1 (확장)

## Context

Initial entelix plan covered Anthropic *managed-agent* shape (Session/Harness/Hand) as the architectural spine, with `Agent`, `Tool`, `Codec`, `Transport` as the primary public surface.

Deep analysis vs LangChain (1.0) and LangGraph (1.0) revealed a structural gap: **composition primitive** (LC's Runnable + LCEL pipe) and **control-flow graph** (LG's StateGraph) were both missing. Without these, entelix would be an "agent runtime with provider abstraction" — useful, but not "LangChain/LangGraph class."

The four CRITICAL gaps identified:
1. No `Runnable<I, O>` trait → no LCEL composition
2. No `StateGraph<S>` → no typed control flow
3. No conditional edges / Send fan-out → no routing/parallel agent patterns
4. No Checkpointer with time travel → no LangGraph-equivalent durability semantics

## Decision

Add `Runnable` and `StateGraph` to the **1.0 spine**, not 1.1.

- **`entelix-runnable`** (new crate) — `Runnable<I, O>` trait + `.pipe()` + `RunnableSequence/Parallel/Router/Lambda/Passthrough`
- **`entelix-graph`** (new crate) — `StateGraph<S>` + `Reducer<T>` + conditional edges + `Send` fan-out + `Checkpointer` trait + `interrupt()` + subgraphs
- **`entelix-prompt`** (new crate) — `PromptTemplate`, `ChatPromptTemplate`, `MessagesPlaceholder`, `FewShot`
- **`entelix-agents`** (new crate) — `create_react_agent`, `create_supervisor_agent`, `create_hierarchical_agent`, `create_chat_agent`

This raises the workspace from 10 → 15 crates. ADR-0001 already accommodated the multi-crate approach; this expands the count.

## Invariant additions (CLAUDE.md)

Two new invariants codify the contract:

> **11. Runnable is the composition contract** — Anything composable implements `Runnable<I, O>`. Codecs, prompts, parsers, tools (via adapter), agents — all. `.pipe()` is the universal connector.
>
> **12. StateGraph is the control-flow contract** — Multi-step / conditional / cyclic flows are defined as `StateGraph<S>`. Ad-hoc loops in user code are a smell.

## Consequences

✅ Honest claim: "LangChain/LangGraph-class Rust SDK" — verifiable via 1:1 parity matrix in PLAN.md §7.
✅ Migration path from Python LG users — every LG concept maps.
✅ Stronger types than LG: typed state schema, typed reducer, compile-time edge validation.
✅ Composability beyond agent context — users build LCEL chains for non-agent use too.
❌ Crate count grows (10 → 15) — managed via clear DAG and feature flags.
❌ More API surface to design correctly upfront — requires Phase 1+2 discipline.
❌ Decision points like "Tool extends Runnable?" must be made early (ADR-0011).

## Alternatives considered

1. **Defer Runnable + StateGraph to 1.1** — would not earn the LC/LG-class claim until 2027. Rejected — bad GTM.
2. **Build Runnable but skip StateGraph** — composition without control flow is half-baked; users would still need LangGraph. Rejected.
3. **Build StateGraph without Runnable** — no LCEL composition; users would write boilerplate. Rejected.
4. **Reuse `petgraph` for StateGraph storage** — petgraph is generic graph, not state-machine; wrap not adopt. Decision: use petgraph internally for traversal, public API stays opinionated.

## Implementation order

Phase 1: `entelix-runnable` + `entelix-prompt` + `Runnable<I, O>` working end-to-end.
Phase 2: `entelix-graph` + `Checkpointer` + interrupts + memory.
Phase 3: `entelix-agents` recipes.

This is the critical path. PLAN.md §8 expanded accordingly.

## Amendment 2026-04-30 — State drop semantics

`Checkpointer<S>` impls take ownership of values during `put` /
`update_state` and may drop them — eagerly during eviction, lazily on
backing-buffer reallocation, or at process shutdown — **inside their
internal mutexes**. `InMemoryCheckpointer::put` is the canonical
example: `Vec::push` may reallocate and free the old buffer (and the
`Checkpoint<S>` values it held) while the mutex is held.

Contract: `S` and any field reachable from `S` must satisfy
**`Drop` does not block** — no `block_on`, no synchronous IO, no
lock acquisition that could contend with the dispatch loop. Drop must
return promptly; if the work is unavoidable, the impl spawns a
detached task or pushes to a non-blocking sink and returns.

This is a one-line addition to the existing `S: Clone + Send + Sync +
'static` bound — the bound stays compile-checked, and the drop
discipline is doc-enforced on the trait surface
(`entelix-graph::Checkpointer`).

## References

- LangGraph overview (langchain-ai.github.io): "low-level orchestration framework for long-running stateful agents"
- LangChain LCEL: `Runnable.invoke / batch / stream / astream` + `|` operator
- ADR-0001 — workspace structure
- CLAUDE.md invariants 11–12
