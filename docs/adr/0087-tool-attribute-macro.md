# ADR 0087 — `#[tool]` attribute macro for typed-input tool authoring

**Status**: Accepted
**Date**: 2026-05-06
**Decision**: A new proc-macro crate `entelix-tool-derive` (22nd workspace member) hosts the `#[tool]` attribute macro. Applied to an `async fn` whose signature is `async fn name(ctx: &AgentContext<()>, ...args) -> Result<O>` (ctx optional), the macro generates an `<Name>Input` struct (`Deserialize + JsonSchema`), a unit struct `<Name>` (snake_case → PascalCase), and an `entelix_tools::SchemaTool` impl that deserialises, dispatches, and returns the typed `O`. The original function stays callable; the tool struct is the agent-side surface. The macro is re-exported as `entelix_tools::tool` and `entelix::tool` for ergonomic single-import.

## Context

Slice 100 settled the `Tool<D=()>` trait shape. Manual implementation requires ~30 lines of boilerplate per tool — `Input` struct + `JsonSchema` derive + `SchemaTool` impl + `description` accessor + dispatcher body that unpacks and forwards. pydantic-ai's `@agent.tool` decorator and Vercel AI SDK 5's `tool({ description, parameters, execute })` builder demonstrate that typed-input tool authoring is the most-used ergonomic feature in modern agent SDKs; both pay a runtime-typing tax that Rust's compile-time generics let us avoid entirely.

The 22nd-crate boundary (proc-macros must live in their own crate) is canonical Rust hygiene; `entelix-graph-derive` (slice 51) already established the precedent.

## Decision

### Macro form: function-attribute, not derive

```rust
use entelix::AgentContext;
use entelix::tool;

#[tool]
/// Calculate compound interest.
async fn compound_interest(
    _ctx: &AgentContext<()>,
    principal: f64,
    rate: f64,
    years: u32,
) -> Result<f64> {
    Ok(principal * (1.0 + rate).powi(years as i32))
}
```

Generates:

- `pub struct CompoundInterestInput { principal: f64, rate: f64, years: u32 }` (with `Deserialize + JsonSchema`).
- `pub struct CompoundInterest;` — unit struct.
- `impl SchemaTool for CompoundInterest` with `Input = CompoundInterestInput`, `Output = f64`, `NAME = "compound_interest"`, and an `execute` body that calls back into the original function.

The original `compound_interest` fn stays in scope and remains callable — the macro **adds** items to the surrounding module rather than replacing the function. Operators register the tool via `CompoundInterest.into_adapter()`.

### Why function-attribute over `#[derive(Tool)]`

`#[derive(Tool)]` would attach to a struct that needs a separate `execute` method definition somewhere — no win over manual `SchemaTool` impl. Function-attribute on an async fn lets the macro derive **the entire surface** from one source of truth (the fn signature + body + doc comment), which is the actual win.

Slice 51's `#[derive(StateMerge)]` is on a struct because the source of truth there *is* a struct (the state shape). Different inputs, different macro forms.

### Ctx detection by type, not by name

The first parameter is treated as the optional ctx slot when its type is a reference to `AgentContext<…>`, regardless of whether the binding is `ctx`, `_ctx`, or `_`. Matching by name would force operators to fight clippy's `unused_variable` lint that prefers `_ctx` over `ctx` for unused parameters.

### `SchemaTool`, not `Tool`, is the generation target

The macro emits `SchemaTool` (typed input/output) rather than `Tool` (`Value` in/out). `entelix_tools::SchemaToolAdapter` then bridges to `Tool<()>` via the existing `into_adapter()` path. This keeps the macro thin (no JSON deserialisation logic in generated code; `SchemaTool::execute`'s adapter handles it) and reuses the typed-input infrastructure that ADR-0011 specified.

### Re-export through `entelix-tools`

Operators import either `use entelix_tools::tool;` or `use entelix::tool;`. The proc-macro crate (`entelix-tool-derive`) is an implementation detail — listing it in `Cargo.toml` is not required because `entelix-tools` already depends on it transitively.

## What the macro does NOT support (1.0)

- **Generic / lifetime-parameterised functions** — rejected at parse time with a diagnostic that points operators to manual `SchemaTool` impl.
- **Custom tool name overrides** — operators rename the function. Adding `#[tool(name = "…")]` is straightforward but premature without a validating use case.
- **`Effect` / `RetryHint` / `version` annotations** — same reason; manual `SchemaTool` impl covers the long tail until 1.x adds attribute arguments.
- **Typed `D` (non-`()`) deps** — the macro hard-codes `Tool<()>` because slice 103 (`ToolRegistry<D>` generic) hasn't shipped. Once `D` propagates through registry / agent surfaces, the macro extends to detect `&AgentContext<MyDeps>` at the ctx slot and emit `SchemaTool<MyDeps>` impl.

## Why these limitations are acceptable

The macro covers the common case (typed input, `()` deps, no custom metadata) cleanly. The long tail goes through manual `SchemaTool` impl, which already exists and is documented. Adding attribute arguments speculatively without seeing real-world usage patterns risks bikeshedding the syntax of features nobody actually needs.

## Consequences

- 22nd workspace crate. Proc-macro crate hygiene matches slice 51's `entelix-graph-derive` precedent — separate crate, narrow surface, dev-deps for tests only.
- 6 regression tests in `crates/entelix-tool-derive/tests/basic.rs` exercise the canonical paths (with-ctx, without-ctx, metadata generation, schema generation, malformed-input rejection).
- `entelix-tools` gains a single re-export (`pub use entelix_tool_derive::tool`); `entelix` facade re-exports the same.
- 4 public-API baselines refreshed (entelix-tool-derive new + entelix-tools / entelix-mcp / entelix-agents from slice 100 carry-over).
- Slice 102 (built-in tool migration to `#[tool]`) becomes mostly mechanical — most existing tools fit the canonical signature shape.

## References

- ADR-0011 — `Tool` / `SchemaTool` adapter boundary; `#[tool]` is the typed-input authoring entry point.
- ADR-0084 — `AgentContext<D>` typed-deps carrier; ctx detection by type lives here.
- ADR-0085 — `Tool<D>` trait migration; sets up `D = ()` default the macro currently targets.
- `crates/entelix-graph-derive/CLAUDE.md` — sibling proc-macro crate using the same authoring conventions.
- v3 plan slice 101 / D-1 — typed-input ergonomics.
