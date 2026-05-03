# ADR 0005 — Adopt Anthropic's managed-agent shape as the v0.1 spine

**Status**: Accepted
**Date**: 2026-04-26

## Context

Anthropic published [Managed Agents](https://www.anthropic.com/engineering/managed-agents) in 2026, articulating a three-component architecture:

- **Session** — append-only event log, externally stored
- **Harness** — stateless brain (Claude + loop)
- **Sandbox** — hand, single `execute(name, input) → string` interface

Anthropic's premise: model improvement happens fast; harness assumptions become obsolete fast. A library locked to today's harness shape becomes 1.0 debt within a release cycle.

LangGraph (Python) achieved comparable shape via StateGraph + checkpointers, but the patterns are conventions, not invariants.

## Decision

entelix's v0.1 architecture is **structurally** Anthropic's managed-agent shape. This is not an aspiration — it is a hard invariant enforced by:

1. **Crate boundaries** — Session = `entelix-graph` + `entelix-persistence`, Harness = `entelix-core::Agent`, Hand = `entelix-core::Tool` + `entelix-mcp`
2. **Trait design** — `Tool::execute(input, ctx) → output`, no other Tool method exists
3. **Type design** — `Agent` holds no `Persistence` field; it accepts `&dyn Persistence` per call. Stateless by construction.
4. **Lifecycle rules** — see `CLAUDE.md` invariants 1-3

If a future feature blurs these boundaries (e.g., harness caching session state, tool inputs receiving credentials), reject it — even if it would be ergonomically convenient.

## Consequences

✅ Crash recovery is free — `Agent::wake(session_id)` is a 10-line implementation, not a feature.
✅ Multi-pod scale-out is free — any pod can serve any session.
✅ Hot-swap LLM mid-session is structurally possible — IR is provider-neutral.
✅ Sub-agent (brain-passes-hand) is structurally one line — `parent.spawn_subagent()`.
✅ Aligns with the most authoritative source on agent architecture in 2026.
❌ Forces eager ergonomic decisions — no caching shortcuts during prototype.
❌ Reading/writing event logs adds latency (~1ms per event with Postgres) — accepted.
❌ Some users want simpler in-memory-only agents — `MemoryPersistence` provides this without breaking shape.

## Concrete invariants

These are CI-enforceable (grep gates + unit tests):

| Invariant | Test |
|---|---|
| `Agent` struct holds no `Persistence` | `cargo expand --crate entelix-core agent` + grep |
| `Tool::execute` signature is exactly `(input, ctx) → output` | trait def lint |
| `ExecutionContext` does not embed `CredentialProvider` | type assertion in `entelix-core/tests/` |
| `SessionGraph::events` is `Vec<GraphEvent>`, not behind `Cell`/`RefCell` | type assertion |
| Persistence backends mutate only inside `with_session_lock` | grep gate |

These run in `scripts/check-managed-shape.sh`, blocking CI if they fail.

## Trade-offs Anthropic flagged that we accept

1. **Container provisioning cost** — Anthropic moved sandboxing out of the harness; entelix similarly delegates to deployment. Cold-start for tools that need a fresh container is the deployer's problem, not ours.
2. **Event log size** — long sessions accumulate events. `entelix-graph::archived_watermark` enables soft-delete + on-demand replay (similar to git-style packfiles). Phase 5 work.
3. **Latency from external session** — Postgres + lock = ~1-3ms per event. For real-time agents, `RedisPersistence` brings this to ~0.5ms. For low-volume agents, `MemoryPersistence` is free (no durability).

## References

- Anthropic, *Managed Agents* (engineering blog)
- LangGraph StateGraph + Checkpointer documentation
- ADR 0001 — workspace structure (the crate boundaries enforce this shape)
- CLAUDE.md invariants 1-3
