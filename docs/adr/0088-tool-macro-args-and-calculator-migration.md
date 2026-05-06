# ADR 0088 — `#[tool]` macro arguments + Calculator migration

**Status**: Accepted
**Date**: 2026-05-06
**Decision**: The `#[tool]` attribute macro (slice 101 / ADR-0087) gains optional metadata arguments — `name`, `effect`, `idempotent`, `version`, `retry_hint` — that emit the matching `SchemaTool` accessor overrides on the generated impl. `SchemaTool` gains an `idempotent()` accessor (default `false`) so the adapter can propagate the flag onto `ToolMetadata`. `CalculatorTool` (the only stateless built-in) migrates from a manual `Tool` impl to the `#[tool]` form, demonstrating the macro's coverage and shrinking the file from a hand-rolled struct + builder + `Tool` impl to a single free async fn. `pub mod calculator` becomes public so `#[tool]`-generated `pub` items re-export cleanly through `entelix_tools::*` and `entelix::*`.

## Context

Slice 101 shipped `#[tool]` covering the deps-less stateless tool path. Built-in tool audit (slice 102 prep) showed 1 of the 19 first-party tools fits that profile cleanly:

| Tool | State | Migrates to `#[tool]`? |
|---|---|---|
| `CalculatorTool` | metadata only | ✅ |
| `HttpFetchTool` | `reqwest::Client + HostAllowlist + config` | ❌ — needs `Tool<Deps>` (slice 103+) |
| `SearchTool` | `Arc<dyn SearchProvider>` | ❌ |
| `Sandboxed*Tool` (5 tools) | `Arc<dyn Sandbox> + policy` | ❌ |
| `*SkillTool` (3 tools) | `SkillRegistry` | ❌ |
| Memory tools (~9 tools) | `Arc<dyn ...Backend>, namespace` | ❌ |
| `McpToolAdapter`, `SubagentTool` | dynamic / typed-state | ❌ |

The macro-generated unit struct shape doesn't fit state-rich tools. Once `ToolRegistry<D>` (slice 103) and `Tool<D>` typed-deps thread end-to-end, those tools migrate by moving struct fields into `D` — but that's a follow-on slice, not slice 102.

Calculator's manual impl had three pieces of metadata fidelity the v1 macro couldn't preserve: a custom `name` differing from the fn ident, `effect = ToolEffect::ReadOnly`, and `with_idempotent(true)`. Slice 102 closes those gaps.

## Decision

### Macro arguments

```rust
#[tool(
    name = "calculator",            // override fn-derived name
    effect = "ReadOnly",            // ToolEffect variant ident
    idempotent,                     // flag (no value)
    version = "1.2.3",              // version label
    retry_hint = "idempotent_transport",  // RetryHint constructor ident
)]
/// Description goes here.
async fn calc(...) -> Result<f64> { ... }
```

Each argument emits the matching accessor on the generated `SchemaTool` impl:

- `name = "..."` overrides `const NAME: &'static str`.
- `effect = "Variant"` emits `fn effect() -> ToolEffect { ToolEffect::Variant }`.
- `idempotent` emits `fn idempotent() -> bool { true }`.
- `version = "..."` emits `fn version() -> Option<&str> { Some("...") }`.
- `retry_hint = "ctor"` emits `fn retry_hint() -> Option<RetryHint> { Some(RetryHint::ctor()) }`.

Absent arguments leave the `SchemaTool` default in place. Unknown arguments are rejected at parse time with a diagnostic listing the supported set.

### `SchemaTool::idempotent()`

```rust
pub trait SchemaTool: Send + Sync + 'static {
    // ... existing ...
    fn idempotent(&self) -> bool { false }
}
```

`SchemaToolAdapter::new` now calls `metadata.with_idempotent(inner.idempotent())` so the flag propagates into `ToolMetadata` for OTel / audit / retry-policy consumers. Defaulting `false` matches `ToolMetadata`'s default.

### `CalculatorTool` → `Calculator`

```rust
// Before (slice 100): manual Tool impl, struct holding ToolMetadata, ~90 lines.
pub struct CalculatorTool { metadata: ToolMetadata }
impl CalculatorTool { fn new() -> Self { ... } }
#[async_trait] impl Tool for CalculatorTool { ... }

// After (slice 102): single free async fn + #[tool] macro.
#[tool(effect = "ReadOnly", idempotent)]
/// Evaluate an arithmetic expression. Supports `+ - * / ^`, unary minus, and parentheses;
/// no variables or named functions. Returns the `f64` result.
pub async fn calculator(_ctx: &AgentContext<()>, expression: String) -> Result<CalculatorOutput> {
    let result = evaluate(&expression).map_err(ToolError::Calculator)?;
    Ok(CalculatorOutput { expression, result })
}
```

The macro emits `pub struct Calculator;`, `pub struct CalculatorInput { expression: String }` (with `Deserialize + JsonSchema`), and the `SchemaTool` impl. Operators register via `Calculator.into_adapter()` — same surface shape as the old `CalculatorTool::new()` builder, single line shorter at the call site.

Renames at the public boundary:
- `entelix_tools::CalculatorTool` → `entelix_tools::Calculator`
- New typed surface: `entelix_tools::CalculatorInput`, `entelix_tools::CalculatorOutput`
- `entelix::CalculatorTool` → `entelix::Calculator`

### `extern crate self as entelix_tools`

`entelix-tools` uses its own `#[tool]` proc-macro internally (Calculator). Generated paths use `::entelix_tools::SchemaTool` for caller-side hygiene. Inside the crate that resolves only when `entelix_tools` resolves to itself; the canonical Rust pattern is `extern crate self as entelix_tools;` at the crate root. Same trick `serde` uses internally for its derive macros.

### `pub mod calculator`

The macro emits `pub` items inheriting the source fn's visibility. `pub` items inside a private `mod` cannot be re-exported through `pub use`. Promoting `mod calculator → pub mod calculator` keeps the canonical re-exports working without rewriting the macro to override visibility.

## Why other built-ins do not migrate now

State-rich tools (`HttpFetchTool` / sandboxed / memory / skills) hold runtime handles that don't fit the unit-struct shape `#[tool]` generates. Forcing them through the macro now would either:

1. Generate empty unit structs that delegate to top-level functions reading lazy `OnceLock` / `static` state — ergonomically worse than the current explicit struct + constructor.
2. Require the macro to grow a `with_state` / `from_fields` mode — speculative complexity without seeing how `Tool<D>` (slice 103) lands.

The right migration shape: stateful tools become typed-deps tools whose state lives in the agent's `D` carrier. That migration is slice 104+ work after `ToolRegistry<D>` ships.

## Consequences

- 1 built-in migrated, 18 stay manual until slice 103+ enables typed deps.
- Macro covers the long tail of operator-authored stateless tools (most agent-author code falls in this shape).
- Public surface change: `CalculatorTool` removed, `Calculator` + `CalculatorInput` + `CalculatorOutput` added. Re-export aliases NOT preserved (invariant 14 — no shim). Operators upgrade by renaming.
- 8 tests in `crates/entelix-tool-derive/tests/basic.rs` (1 new for metadata args).
- 11 tests in `crates/entelix-tools/src/calculator.rs::tests` (1 new for metadata-args propagation through SchemaToolAdapter).
- Public-API baselines refreshed (entelix-tools, entelix-tool-derive, entelix-agents, entelix-mcp from slice 100 carry-over).

## References

- ADR-0085 — `Tool<D>` trait migration; sets up `D = ()` default that `#[tool]` targets.
- ADR-0087 — `#[tool]` attribute macro contract; this ADR extends it with metadata args.
- ADR-0011 — `Tool` / `SchemaTool` adapter boundary; `idempotent()` accessor lives on the typed-input surface.
- v3 plan slice 102 — built-in migration scope clarified.
