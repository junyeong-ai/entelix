# ADR 0093 — Sub-agent identity is set at builder time, not at `into_tool`

**Status**: Accepted
**Date**: 2026-05-06
**Decision**: `Subagent::builder` accepts `name` and `description` as required arguments, the builder threads them into the resulting `Subagent`, and `Subagent::name()` / `Subagent::description()` / `Subagent::metadata()` accessors expose the identity *before* the conversion-to-`Tool` boundary. `Subagent::into_tool()` becomes argument-less — it consumes the stored identity rather than re-asking for it. A new `SubagentMetadata` struct (`name`, `description`, `tool_count`, `tool_names: Vec<String>`) bundles the inspection surface for parent-side system-prompt enrichment and registry-style listings.

## Context

ADR-0089 unified the sub-agent construction surface into `SubagentBuilder` but left identity (`name`, `description`) on `into_tool` — the conversion that wraps the sub-agent as a `Tool`. That meant:

- A built `Subagent` was nameless. `Subagent::tool_count()` / `Subagent::tool_names()` worked, but the operator-facing identity was unset until `into_tool("name", "desc")` materialised the `SubagentTool`.
- Operators who wanted to list available sub-agents in a parent agent's system prompt — branchforge / pydantic-ai parity — had to either consume the `Subagent` (drop it) or maintain a parallel registry.
- `into_tool` carried metadata that conceptually belonged to the sub-agent definition, not the tool conversion.

Slice 111 closes the gap: identity flows through the builder, `into_tool` is argument-free, and `metadata()` lets parents inspect sub-agents without consuming them.

## Decision

### `Subagent::builder` signature

```rust
pub fn builder(
    model: M,
    parent_registry: &ToolRegistry,
    name: impl Into<String>,
    description: impl Into<String>,
) -> SubagentBuilder<'_, M>
```

`name` and `description` are required. They thread through `SubagentBuilder` → `Subagent` → `SubagentTool` (when `into_tool` is called) without further argument passes.

### `Subagent` accessors

```rust
impl<M> Subagent<M> {
    pub fn name(&self) -> &str;
    pub fn description(&self) -> &str;
    pub fn metadata(&self) -> SubagentMetadata;
    // Existing: tool_count, tool_names, tool_registry, skills, ...
}
```

`metadata()` returns a fresh snapshot — `Vec<String>` of tool names is owned, so the snapshot outlives a borrow of the sub-agent. Operators that want minimal allocation cost reach for the individual accessors directly.

### `SubagentMetadata`

```rust
#[derive(Clone, Debug)]
pub struct SubagentMetadata {
    pub name: String,
    pub description: String,
    pub tool_count: usize,
    pub tool_names: Vec<String>,
}
```

Fields are `pub` (operators construct test fixtures, persist to JSON, etc.) but the struct is conceptually a snapshot — entelix never reads it back as configuration. A future addition (effect, tags, version) extends the struct under `#[non_exhaustive]` if needed; for 1.0 the four fields are the load-bearing surface.

### `Subagent::into_tool()`

```rust
pub fn into_tool(self) -> Result<SubagentTool>
```

Argument-less. Reads `self.name` / `self.description`, builds the inner ReAct agent, wraps as `SubagentTool` with metadata derived from the stored identity. Operators that want to override at conversion time call `with_effect` etc. on the returned tool — those exist already.

## Migration cost

13 call sites across 6 test files migrated mechanically. Original shape:
```rust
let sub = Subagent::builder(model, &parent).restrict_to(&[]).build()?;
let tool = sub.into_tool("name", "desc")?;
```
Becomes:
```rust
let sub = Subagent::builder(model, &parent, "name", "desc")
    .restrict_to(&[])
    .build()?;
let tool = sub.into_tool()?;
```

Tests that didn't call `into_tool` get stub identity (`"test_subagent"`, `"test description"`) — the builder requires identity even when the test never looks at it.

## Consequences

- `SubagentBuilder` requires identity at construction. Tests / operators that built nameless sub-agents must supply stub strings — small mechanical cost, structural win.
- `SubagentMetadata` ships as a public type. Three new public symbols (`SubagentBuilder`, `SubagentMetadata`, plus `Subagent::name()` / `description()` / `metadata()` accessors).
- `entelix-agents` and `entelix` facade public-API baselines refreshed.
- 13 test sites migrated; all 7 test binaries pass.

## References

- ADR-0089 — `SubagentBuilder` consolidation; this ADR closes the identity-on-into_tool gap left there.
- `crates/entelix-agents/CLAUDE.md` — managed-agent shape (sub-agent name flows to `SubagentTool` metadata, parent-side LLM dispatches it like any other tool).
- branchforge `SubagentIndex` — inspiration for metadata-first sub-agent listings.
- v3 plan slice 111.
