# ADR 0085 — `Tool<D>` typed-deps trait migration

**Status**: Accepted
**Date**: 2026-05-06
**Decision**: The `Tool` trait gains a single type parameter `D` defaulting to `()`, and its `execute` method takes `&AgentContext<D>` instead of `&ExecutionContext`. The tower service spine (`ToolInvocation`, every layer, `ToolRegistry`) keeps consuming `ExecutionContext` — typed deps live on the leaf tool surface only, not on layer-side machinery. Existing deps-less tools migrate mechanically: `impl Tool` stays valid (default `()`) and the body switches `&ExecutionContext` to `&AgentContext<()>`. ADR-0085 closes the v3 plan slice 100a; slice 103 will widen `ToolRegistry<D>` to thread typed deps end-to-end.

## Context

Slice 99 (ADR-0084) introduced `AgentContext<D>` as the carrier that separates infra context from operator-side typed deps. With the carrier in place, the next step is to flow `D` into the only surface that benefits from it: the `Tool` trait. pydantic-ai's `@agent.tool` decorator and Vercel AI SDK 5's `tool({ execute(args, options) })` shape both demonstrate that typed-deps access is the most-used tool ergonomic feature in modern agent SDKs — and both pay a runtime-typing tax that Rust's compile-time generics can avoid entirely.

## Decision

### Tool trait shape

```rust
#[async_trait]
pub trait Tool<D = ()>: Send + Sync + 'static
where
    D: Send + Sync + 'static,
{
    fn metadata(&self) -> &ToolMetadata;

    async fn execute(
        &self,
        input: serde_json::Value,
        ctx: &AgentContext<D>,
    ) -> Result<serde_json::Value>;
}
```

`D = ()` default keeps `impl Tool for X` (without an explicit type parameter) syntactically valid and semantically identical to `impl Tool<()> for X`. Operator-typed tools declare `impl Tool<MyDeps> for X` and reach for `ctx.deps()` inside the body; deps-less tools see `ctx.deps() == &()` for free.

### Layer ecosystem stays D-free

`ToolInvocation` continues to carry `ctx: ExecutionContext`. The leaf service (`InnerToolService`) wraps that as `AgentContext::<()>::from(invocation.ctx)` immediately before dispatching to `Tool::execute`. Every layer (`PolicyLayer`, `OtelLayer`, `RetryService`, `ApprovalLayer`, `ToolEventLayer`, `ScopedToolLayer`) continues operating on `ExecutionContext` exclusively — they never see `D`, so no generic explosion ripples through the layer stack.

### Adapter integration

- `SchemaToolAdapter`'s `Tool<()>` impl wraps `SchemaTool::execute(input, ctx: &AgentContext<()>)`.
- `SchemaTool` itself takes `&AgentContext<()>` — typed-input tool authors get the same forwarder ergonomics as raw `Tool` authors.
- `ToolToRunnableAdapter::invoke` (which still receives `&ExecutionContext` because `Runnable` is layer-side / D-free) wraps the context as `AgentContext::<()>::from(ctx.clone())` before dispatching to the inner `Tool`.
- `McpToolAdapter::execute` extracts `ctx.core()` to forward to `McpManager::call_tool`, which is an HTTP-side facade and does not accept typed deps.
- `SubagentTool::execute` builds `child_ctx` from `ctx.core().clone()` so the inner `Runnable<ReActState, ReActState>` (D-free) sees the same shape it always has.

### Backend pass-through pattern

Tool bodies that delegate to memory backends (`SemanticMemory`, `EntityMemory`, `BufferMemory`, …), sandbox backends (`Sandbox` trait), or skill backends (`Skill::load`, `SkillResource::read`) call `ctx.core()` to obtain the `&ExecutionContext` those D-free traits expect. The forwarder accessors on `AgentContext<D>` (`ctx.cancellation()`, `ctx.tenant_id()`, `ctx.audit_sink()`, …) cover the in-body lookups that do not cross a backend boundary.

### Test ergonomics

- Direct `tool.execute(input, &ctx)` call sites construct `AgentContext::default()` (deps-less) or `AgentContext::<()>::from(execution_ctx.clone())` when an existing `ExecutionContext` needs reuse for `ToolRegistry::dispatch` in the same test.
- `ToolRegistry::dispatch(name, input, &ctx)` continues to take `&ExecutionContext` — registry surface is D-free.

## Consequences

- 19 first-party Tool impls migrated mechanically (`HttpFetchTool`, `CalculatorTool`, `SearchTool`, `SchemaToolAdapter`, `SandboxedShellTool` / `Read` / `Write` / `List` / `Code`, `MockSandbox` test impls, `SkillRegistry` tools, `MemoryGetTool` / `MemoryPutTool` / `MemorySearchTool` / `MemoryDeleteTool` / `MemoryUpdateTool` / `EntityFactSetTool` / `EntityFactGetTool` / `EntityListTool` / `EntityForgetTool`, `McpToolAdapter`, `SubagentTool`, `ApprovalLayer` test `EchoTool`, `ToolToRunnableAdapter` test `DoubleTool`, `recipes` test `MockTool` / `RecursionTool` / `LoopingTool`).
- 1 example (`13_mcp_tools.rs`) migrated to wrap its `ExecutionContext` into `AgentContext` at the dispatch boundary.
- Public-API baselines refreshed: `entelix-core`, `entelix-tools`, `entelix-mcp`, `entelix-agents`.
- No layers, no transports, no codecs touched.
- Slice 103 will widen `ToolRegistry` to `ToolRegistry<D>`, hold typed deps in `InnerToolService<D>`, and reach `Subagent::restrict_to(...)` builder — at that point operator-typed deps thread end-to-end.

## Why D defaults to `()`

The same reasoning as `AgentContext<D = ()>` (ADR-0084 §"Why `D` defaults to `()`, not `Box<dyn Any>`"). `()` is Rust's canonical "no information" type; deps-less code stays zero-overhead and idiomatic. A `Box<dyn Any>` default would force every callsite to either accept dynamic dispatch tax or specify a phantom type.

## Alternatives considered

- **Keep `Tool` D-free, attach typed deps via `Extensions`**. Rejected — defeats the type-safety purpose of slice 99. Operators that forget to attach deps would silently receive `None` from `ctx.extension::<MyDeps>()`. Compile-time presence guarantees are the entire reason for slice 99.
- **Two traits — `Tool` (D-free) and `TypedTool<D>`**. Rejected — bifurcates the ecosystem. Every adapter, every registry method, every example ends up duplicated. Single trait with a default type parameter avoids this.
- **`Tool` extends `Runnable<Value, Value>`** (replace adapter pattern). Rejected per ADR-0011 — `Tool::metadata` is the descriptor surface, `Runnable` is the composition contract; conflating the two has been rejected before for good reasons (descriptor overhead in pure-composition pipelines).

## References

- ADR-0011 — `Tool` / `Runnable` adapter boundary (`ToolToRunnableAdapter`).
- ADR-0084 — `AgentContext<D>` typed-deps carrier.
- Invariant 4 — Hand contract (`Tool::execute` single method).
- Invariant 10 — operator-side handles never reach Tool input dynamically.
- v3 plan slice 100a — Tool trait migration.
