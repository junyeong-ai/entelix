# ADR 0084 — `AgentContext<D>` typed-deps carrier

**Status**: Accepted
**Date**: 2026-05-06
**Decision**: A new request-scope carrier `AgentContext<D = ()>` separates infra context (`ExecutionContext` — cancellation, tenant scope, deadline, audit sink, extensions) from operator-side typed deps (`D` — database pool, HTTP client, tenant config, ...). Layers and the `tower::Service` spine consume `ExecutionContext` only via `ctx.core()`; the `Tool` and validator surfaces consume `AgentContext<D>` and reach `D` via `ctx.deps()`. `D` defaults to `()` so deps-less agents pay no type-system tax. This is the foundation for slices 100–114 (Tool trait migration, ToolRegistry generics, Agent generic) and the structural fix for v3 plan Flaw #1 (layer ecosystem generic explosion).

## Context

The v3 Phase A plan (slice 99–114) introduces typed deps — pydantic-ai's `Agent[Deps, Out]` shape, where every tool callback receives a typed handle to operator-side state (DB pool, HTTP client, tenant config). The first-cut design proposed parameterizing `ExecutionContext<D>`, but self-audit (Flaw #1) caught a structural problem:

- Invariant 4 requires sub-agents to share the parent's `tower::Layer` factory by `Arc`. The factory is generic over the service's request type.
- If `ExecutionContext` carries `D`, then `ToolInvocation` carries `D`, then every layer (`PolicyLayer`, `OtelLayer`, `RetryService`, `ApprovalLayer`, `ToolEventLayer`) is generic over `D`.
- Layers do not need `D` — they read tenant id, run id, idempotency key, run budget, audit sink. Forcing `D` through the layer ecosystem is generic explosion with zero semantic gain.

The fix splits the carrier:

- **`ExecutionContext`** stays the D-free infra carrier. Layers / tower spine / codecs / transports never see `D`.
- **`AgentContext<D>`** is a new wrapper exposing `core()` for layer-side access and `deps()` for tool-side access.

This matches pydantic-ai's actual shape (`RunContext[Deps]` = a wrapper, not the agent run's only context) and aligns with how Rust generics scale: keep generic boundaries narrow, parameterize only the surfaces that need it.

## Decision

### Carrier types

```rust
pub struct AgentContext<D = ()> {
    core: ExecutionContext,
    deps: D,
}
```

- `D` defaults to `()` — every existing deps-less callsite stays `AgentContext<()>` with no annotation.
- Constructor: `AgentContext::new(core, deps)`.
- Accessors: `core()`, `core_mut()`, `deps()`, `deps_mut()`. No `get_*` per ADR-0010.
- Decompose: `into_parts() -> (ExecutionContext, D)`.
- Combinator: `map_deps<E>(self, f: FnOnce(D) -> E) -> AgentContext<E>` for sub-agent deps narrowing.

### Forwarders for ergonomics

`AgentContext<D>` exposes the high-traffic `ExecutionContext` accessors directly so tool bodies write `ctx.tenant_id()` / `ctx.cancellation()` / `ctx.is_cancelled()` without a `.core()` hop. Forwarded surface:

- `cancellation()`, `deadline()`, `thread_id()`, `tenant_id()`, `run_id()`, `idempotency_key()`, `is_cancelled()`
- `extensions()`, `extension::<T>()`, `run_budget()`, `audit_sink()`

Builder methods delegate to `core` and return `Self` for chaining: `with_deadline`, `with_thread_id`, `with_tenant_id`, `with_run_id`, `with_idempotency_key`, `with_run_budget`, `with_audit_sink`, `add_extension`. The retry middleware reaches `ctx.core_mut().ensure_idempotency_key(...)` for the one mutable-borrow case.

### Trait implementations

- `Default for AgentContext<()>` — fresh `ExecutionContext::default()` + unit deps.
- `From<ExecutionContext> for AgentContext<()>` — single-line wrap of an existing context.
- `Clone for AgentContext<D> where D: Clone` — shallow: `ExecutionContext` clones cheaply (Arc refcounts), `D` clones with operator semantics.
- `Debug for AgentContext<D> where D: Debug` — formats core + deps for diagnostics.
- `child(&self) -> Self where D: Clone` — child cancellation token + cloned deps. Sub-agent dispatch path.

### What stays unchanged in slice 99

- `ExecutionContext` itself — unchanged. All existing consumers continue.
- `Tool::execute(input, ctx: &ExecutionContext)` — unchanged. Slice 100 changes the trait signature to take `AgentContext<D>`.
- `ToolRegistry` — unchanged. Slice 103 makes it generic on `D`.
- `Agent<S>` — unchanged. Slice 104 makes it `Agent<S, D = ()>`.
- All layers (`PolicyLayer`, `OtelLayer`, `RetryService`, `ApprovalLayer`, `ToolEventLayer`) — unchanged forever. They consume `ExecutionContext` directly via `ctx.core()` once the upstream sites switch to `AgentContext`.

### What slice 99 wires

- `crates/entelix-core/src/agent_context.rs` — module + 18 unit tests.
- `crates/entelix-core/src/lib.rs` — module declaration + `pub use AgentContext`.
- `crates/entelix/src/lib.rs` — facade flat re-export + prelude include.
- `docs/adr/0067-agent-context-typed-deps.md` — this document.

The type is foundation for the next 14 slices. It does not yet thread through any production codepath; the v3 plan calls slice 100–114 the integration sequence. invariant 14 (no shims) is honored — there is no old surface being kept; the type is new and additive.

## Why typed `D`, not dynamic extension

`ExecutionContext::extensions` already provides a dynamic typed slot via `add_extension::<T>(value)` + `extension::<T>()`. Why introduce a static `D` parameter when the dynamic path exists?

1. **Compile-time presence guarantee**. A tool that needs a `Database` handle should not silently receive `None` from `ctx.extension::<Database>()` because someone forgot to wire it. With `D = AppDeps { db: Database, ... }`, the type system enforces that every dispatch site has supplied the deps.
2. **Test ergonomics**. `agent.with_deps(test_deps)` swaps the entire dependency graph atomically. Dynamic extensions require per-key replacement — a maintenance burden that scales poorly.
3. **Refactor safety**. Renaming a deps field is a `cargo check` away from updating every call site. With dynamic extensions, renames are a string-search exercise.
4. **Invariant 10 alignment**. Credentials structurally cannot reach `D` because `CredentialProvider` is a sealed contract on `Transport`, not a deps surface. Operators who try to stash a token in `D` get caught by the same review that catches it in `Extensions` today — but with `D`, the boundary is explicit at the type level.

Dynamic extensions remain available for *cross-cutting* state (run id, audit sink, run budget) where the framework owns the lifecycle. Operator state goes in `D`.

## Why `D` defaults to `()`, not `Box<dyn Any>`

Defaulting to `()` makes deps-less code zero-overhead and idiomatic — `AgentContext::default()` returns `AgentContext<()>` with no annotations. A `Box<dyn Any>` default would force every callsite to either accept the dynamic dispatch tax or specify a phantom type. `()` is the canonical "no information" type in Rust and matches how `Result<T, ()>` and `Vec<()>` document "I'm not using this slot."

## Naming

- `AgentContext` — `*Context` suffix per ADR-0010 (request-scope state).
- `core()` / `deps()` — accessors, no `get_` prefix.
- `with_*` builder methods — verb-prefix exception per ADR-0010 §"Builder verb-prefix exception" (configuration setter on a single target).
- `add_extension` — collection insert, matches `ExecutionContext::add_extension`.
- `child()` — derives a scoped child, matches `ExecutionContext::child()`.
- `into_parts()` / `map_deps()` — value-level combinators, no `get_` prefix.

## Consequences

- **Slice 100** changes the `Tool` trait signature: `async fn execute(&self, input, ctx: &AgentContext<D>) -> Result<ToolOutcome<O>>`. Built-in tools migrate mechanically — the slice 99 forwarders mean most tool bodies do not change beyond the parameter type.
- **Slice 103** makes `ToolRegistry<D>` generic. The `tower::Service<ToolInvocation>` shape — and therefore every layer — stays D-free; the leaf service (`InnerToolService<D>`) holds an `Arc<dyn Tool<D>>` plus a clone of `D`, materializes `AgentContext<D>` from the invocation's `ExecutionContext` plus the held deps, and dispatches to `Tool::execute`. Generic explosion is contained to one type (the leaf), not the layer ecosystem.
- **Slice 104** makes `Agent<S, D = ()>` generic. `AgentBuilder::with_deps(d: D)` is the single entry point.
- **Slice 106** (`OutputValidator<O, D>`) and slice 110 (Subagent) inherit `D` through the `AgentContext` carrier.
- **Future slice** when `pre_model_hook` / `post_model_hook` (slice 123) lands, they receive `&AgentContext<D>` for symmetry with tools.

## Alternatives considered

- **`ExecutionContext<D>`** (parameterize the existing carrier). Rejected — generic explosion through the layer ecosystem (Flaw #1).
- **`AgentContext` as a trait, `RunContext<D>` as the impl** (mirror pydantic-ai's `RunContext[Deps]` literal). Rejected — Rust traits with associated types are higher-friction than a struct + generic. The struct shape composes with `Tool<D>` and `OutputValidator<O, D>` cleanly without an extra trait.
- **Hide `D` behind a typed extension key**. Rejected — invariant 10 alignment and refactor safety arguments above. Dynamic extensions stay for cross-cutting framework state, not operator state.

## References

- ADR-0010 — naming taxonomy (`*Context` suffix, accessor naming, ctx parameter ordering, builder verb-prefix exception).
- ADR-0017 — `tenant_id` mandatory + `Extensions` slot.
- ADR-0064 — 1.0 release charter (Phase A foundation).
- Invariant 4 — Hand contract (Tool::execute single method).
- Invariant 10 — tokens never reach Tool input.
- v3 plan — `project_entelix_2026_05_06_redesign_v3.md` (Flaw #1, slice 99 scope).
