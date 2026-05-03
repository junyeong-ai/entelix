# ADR 0068 — `ToolDispatchScope` for ambient request-scope state

**Status**: Accepted
**Date**: 2026-05-02
**Decision**: Operators wire a `ToolDispatchScope` impl behind a `ScopedToolLayer` `tower::Layer<S>` to wrap every `Tool::execute` future with ambient request-scope state (tokio task-locals, RLS `SET LOCAL` settings, tracing scopes) the SDK cannot supply through `ExecutionContext` directly. The hook fires uniformly for parent-agent dispatch, sub-agent narrowed dispatch, and recipe-driven direct dispatch — anywhere `ToolRegistry::dispatch` flows through.

## Context

`ExecutionContext` carries typed request-scope data — `cancellation`, `deadline`, `thread_id`, `tenant_id`, `extensions` — that every `Runnable`, `Tool`, and codec consumes by value. Some tool implementations need ambient state that lives in a *thread-local* or *tokio task-local* rather than a typed field on the context, and they need that state active *while their future polls* (not just when the synchronous setup runs).

The driving example is Postgres row-level security: the tool's query path reads `current_setting('entelix.tenant_id', true)`, which is `SET LOCAL`-scoped to the current transaction. The SDK must enter that scope before the tool's future starts polling and restore the prior scope when the future resolves. Other examples:

- W3C trace-context propagation through `tracing::Span::in_scope`.
- A workspace-id task-local that the operator's repository tools read for filesystem layout.
- A "system bypass" task-local that disables RLS during operator-initiated maintenance jobs.

Pre-1.0 the SDK had no such hook. Operators porting workloads from frameworks that ship one had no clean equivalent — they had to monkey-patch `Tool::execute` impls or wrap every tool individually. The audit identified this as the single CRITICAL gap blocking ambient-task-local porting.

## Decision

Add a `ToolDispatchScope` trait and a `ScopedToolLayer` `tower::Layer<S>` to `entelix-core::tools`. Operators implement the trait with their wrap logic and attach the layer to a `ToolRegistry`:

```rust
let registry = ToolRegistry::new()
    .layer(ScopedToolLayer::new(MyWorkspaceScope))
    .register(Arc::new(my_tool))?;
```

The trait is object-safe (no generic on output type — pinned to `Result<serde_json::Value>` since that's `Tool::execute`'s shape) and takes `ctx` by value (`ExecutionContext` clone is cheap — `Arc<str>` for `tenant_id`, refcounted handles for extensions). Implementations read whatever fields they need from ctx and seed task-locals via `tokio::task_local!::scope`.

```rust
pub trait ToolDispatchScope: Send + Sync + 'static {
    fn wrap(
        &self,
        ctx: ExecutionContext,
        fut: BoxFuture<'static, Result<Value>>,
    ) -> BoxFuture<'static, Result<Value>>;
}
```

### Why a Tower Layer (not a `Tool::wrap` trait method)

A trait method on `Tool` would require every tool author to pick up the wrap responsibility — fragile, easy to miss. A Tower Layer slots into the registry's existing layer factory (slice 95/96 + ADR-0011) so the wrap is **infrastructural**, attached once per registry, and inherited automatically by every `Tool::execute` dispatched through that registry. Sub-agents that narrow the parent registry through `ToolRegistry::restricted_to` / `filter` (ADR-0035) share the layer factory by `Arc` — the wrap fires for every sub-agent dispatch with no extra wiring.

### Why object-safe via `BoxFuture<'static, Result<Value>>`

A generic `wrap<T>(...)` would let the trait wrap futures of arbitrary output type but would lose object-safety, blocking `Arc<dyn ToolDispatchScope>`. Pinning the output type to `Result<Value>` matches `Tool::execute`'s actual shape and keeps `dyn` storage usable. Operators that need multi-shape wrapping can implement the trait twice with different concrete impls (or compose two `ScopedToolLayer`s in the registry's stack — Tower layers compose).

### Why ctx is moved (not borrowed)

The `wrap` method moves `ctx` rather than borrowing because `ExecutionContext::clone` is intentionally cheap (refcounted fields) and an owned `ctx` removes lifetime gymnastics from the trait signature. Implementations that need only one field copy that field out and discard the ctx; implementations that need the full context keep it.

### Why no auto-wiring on `Agent::execute`

Recipes that build agents already attach observability layers (`OtelLayer`) and policy layers (`PolicyLayer`) via the same `ToolRegistry::layer` mechanism. `ScopedToolLayer` is one more layer in that stack — operators choose its position relative to the others (typically innermost, so the scope is active even during PII redaction; sometimes outermost, so OTel observes the wrap timing). Auto-wiring would remove that operator control and would also impose a hidden cost on operators who don't need scoping.

### What this does NOT cover

- **`ChatModel` / model-invocation futures.** Vendor LLM API calls don't need RLS task-locals; the wrap is intentionally restricted to tool dispatch. If a future use case emerges (e.g. observability that needs ambient context across the full request), a sibling `ModelDispatchScope` ships in a separate ADR.
- **`Runnable::invoke` futures other than tools.** A graph node that's not a tool dispatches its inner runnable directly, not through the registry's layer stack. Operators that need scope around graph nodes wrap the node's runnable explicitly via `RunnableExt::pipe` with a wrap-runnable adapter (no SDK primitive for that today; demand-driven).

## Consequences

- New trait `entelix_core::tools::ToolDispatchScope` and Layer `entelix_core::tools::ScopedToolLayer` (also reachable through the facade as `entelix::tools::*`).
- New `ScopedToolService<S>` is the produced `Service<ToolInvocation>` — public so operators that build their own dispatch path (rare) can compose it manually.
- Sub-agent dispatch automatically inherits attached scopes through `Arc`-shared layer factory (ADR-0035) — no extra wiring.
- Lock-ordering, cancellation, deadline, and audit-sink semantics unchanged — `ScopedToolService::call` defers to the inner service for `poll_ready`, so backpressure flows through.
- Three regression tests in `tools::scope::tests`: per-dispatch wrap counter, multi-dispatch counter, narrowed-view inheritance.

## References

- ADR-0011 — `Tool` trait (does not extend `Runnable`); `ScopedToolLayer` lives at the dispatch boundary, not on the trait.
- ADR-0035 — sub-agent layer-factory inheritance (`restricted_to` / `filter` share the parent's `Arc<LayerFactory>` so attached scopes propagate).
- `crates/entelix-core/src/tools/scope.rs` — implementation + tests.
