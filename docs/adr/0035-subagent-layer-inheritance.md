# ADR 0035 — Sub-agent layer-stack inheritance

**Status**: Accepted
**Date**: 2026-04-30
**Decision**: Phase 4 of the post-7-차원-audit roadmap

## Context

Anthropic's managed-agent pattern is structured around three roles —
*Session* (the event log), *Harness* (the brain — `Agent` +
`Codec`/`Transport`), and *Hand* (the `Tool` trait). The
"brain passes hand" rule says a sub-agent must dispatch through the
*same* tool surface the parent uses, narrowed to whatever subset the
operator deems appropriate.

The 7-차원 audit (2026-04-30) surfaced D1: `Subagent::into_react_agent`
materialised a fresh `ToolRegistry::new()` from a borrowed
`Vec<Arc<dyn Tool>>` and re-`register`-ed each tool. The freshly
built registry had no layer stack — the parent's `PolicyLayer` (PII
redaction, quota), `OtelLayer` (`gen_ai.tool.*` events, cost), and
any retry middleware silently disappeared at the sub-agent boundary.
Operators saw redacted outputs from the parent and unredacted outputs
from the sub-agent dispatching the same tools, with no warning.

The defect was structural — the type signature
`from_whitelist(model, parent_tools: &[Arc<dyn Tool>], ...)` admitted
no other implementation. As long as `Subagent` carried a raw
`Vec<Arc<dyn Tool>>`, the layer stack had nowhere to ride.

## Decision

`Subagent` takes the parent `ToolRegistry` directly and narrows it
through one of two view methods on the registry. The raw
`Vec<Arc<dyn Tool>>` surface is removed.

### Surface

`ToolRegistry` gains two view constructors. The factory `Arc` rides
over by clone — no per-dispatch cost.

```rust
impl ToolRegistry {
    /// Predicate-based view. Inherits the parent's layer stack.
    pub fn filter<F>(&self, predicate: F) -> Self
    where
        F: Fn(&dyn Tool) -> bool;

    /// Name-allowlist view. Returns Error::Config when any name is absent.
    pub fn restricted_to(&self, allowed: &[&str]) -> Result<Self>;
}
```

`Subagent` constructors take `&ToolRegistry` and route through the
view methods. The struct's tool field becomes `tool_registry:
ToolRegistry` — the narrowed view itself, layer-stack-shared.

```rust
pub struct Subagent<M> { /* ... */ tool_registry: ToolRegistry, /* ... */ }

impl<M> Subagent<M> {
    pub fn from_whitelist(model: M, parent: &ToolRegistry, allowed: &[&str]) -> Result<Self>;
    pub fn from_filter<F>(model: M, parent: &ToolRegistry, predicate: F) -> Self;
}
```

`into_react_agent` no longer constructs a registry — it consumes the
already-narrowed `tool_registry` and (when `with_skills` was called)
appends the three LLM-facing skill tools through
`entelix_tools::skills::install`, which is `Self -> Self` and
preserves the layer factory.

### Enforcement

Three layers, each independent:

1. **Type-level**: the constructor signatures admit only
   `&ToolRegistry`. Building a sub-agent from a raw `Vec<Arc<dyn
   Tool>>` is no longer expressible.
2. **Static gate**: `scripts/check-managed-shape.sh` greps
   `crates/entelix-agents/src/subagent.rs` for `ToolRegistry::new(`
   in non-doc lines and rejects any match. Doc lines (`///`, `//!`)
   are excluded so prose can name the symbol.
3. **Dynamic regression**: `tests/subagent_layer_inheritance.rs`
   attaches a counting `Layer` at the parent registry, dispatches
   one call from the parent and one from the sub-agent's narrowed
   view, and asserts the counter advances on both.

## Consequences

✅ The brain↔hand boundary preserves every layer. Operators no longer
write per-sub-agent layer wiring; the parent's stack rides over.
✅ The `Vec<Arc<dyn Tool>>` "raw bag" surface disappears from the
sub-agent path; downstream code touches `ToolRegistry` directly,
which is the canonical container.
✅ `ToolRegistry::filter` / `restricted_to` are useful beyond
sub-agents — the supervisor recipe can narrow per-handoff, the
chat recipe can hide an admin tool from a particular tenant, etc.
❌ Callers updating from the old shape pass `&parent_registry`
instead of `&parent_tools`. One-line change at every call site.
❌ The static gate forbids `ToolRegistry::new()` inside
`subagent.rs`. New construction patterns inside the file require an
ADR amendment plus a gate update — the friction is the point.

## Alternatives considered

1. **Have `Subagent` rebuild the registry but copy the parent's
   factory**: leaks the boxed factory through `Subagent`'s public
   API, which couples the recipe layer to `entelix-core` internals.
   Rejected.
2. **Keep `Vec<Arc<dyn Tool>>` and add a separate
   `Vec<Arc<dyn Layer<...>>>`**: doubles the surface area and still
   admits a fresh registry path. Rejected.
3. **Inject the parent registry via a dyn trait**: solves nothing
   that the concrete `&ToolRegistry` path doesn't, at the cost of an
   indirection layer. Rejected.

## References

- 7-차원 audit fork report: `audit-tools-obs-server` D1.
- ADR-0005 — Anthropic managed-agent shape.
- ADR-0017 — `tenant_id` strengthening (the same "narrow at the
  boundary, never reconstruct" principle for tenant scope).
- ADR-0032 — invariant #15 silent-fallback prohibition (sibling
  contract: silent loss at boundaries is structurally forbidden).
