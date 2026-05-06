# ADR 0097 — `McpToolAdapter::with_prefix` for custom namespacing

**Status**: Accepted
**Date**: 2026-05-06
**Decision**: Add `McpToolAdapter::with_prefix(prefix)` so operators can override the default `mcp:{server}:{tool}` namespace with `{prefix}:{server}:{tool}`. Empty `prefix` is silently rejected (programmer-error guard, not a configuration choice). Mutually exclusive with `with_unqualified_name`; the *last* call wins.

## Context

Slice 64 / 76 era introduced `mcp:{server}:{tool}` as the default LLM-facing tool name from `McpToolAdapter`, with `with_unqualified_name` as the bare-`{tool}` opt-out for single-server deployments. The two-mode shape covered:

- "default safe" — multi-server collision-resistant naming.
- "single-server bare" — operators whose model is already prompt-trained on the unqualified name.

It missed one shape: operators integrating entelix-managed agents into a wider tool catalogue that already namespaces by team or surface (`platform:files:read`, `external:slack:post`, `internal:github:fetch_pr`). They'd want MCP-side tools to fit *that* scheme, not the entelix-internal `mcp:` prefix. Without the override, they had to construct the adapter, manually rewrite `metadata.name` after the fact (private field — not possible), or build a wrapper `Tool` impl that forwards everything (high boilerplate).

## Decision

```rust
impl McpToolAdapter {
    /// Override the namespace prefix used in the LLM-facing tool name.
    /// Default is `mcp:{server}:{tool}`; calling with_prefix("myapp")
    /// produces `myapp:{server}:{tool}` instead.
    #[must_use]
    pub fn with_prefix(self, prefix: impl Into<String>) -> Self;
}
```

### Empty prefix is silently rejected

`with_prefix("")` would render as `:{server}:{tool}` — the leading colon is not a useful artefact. We could fail-loud (panic / `Result`), but the builder is `#[must_use] -> Self` so a `Result` return is a chain-breaker. Silent no-op is the right shape: the adapter retains its current namespace, and the empty-string call is treated as a programming error caught by the test suite (the regression below pins it).

### Mutually exclusive with `with_unqualified_name` — last call wins

Operators that want both branches build two adapters from the same definition; the chained-builder shape "last call wins" is consistent with `with_*` semantics elsewhere in the SDK. Both orderings are pinned by regression test.

## Test coverage

Extended `crates/entelix-mcp/tests/http_e2e.rs` namespace test with five new asserts:

1. `with_prefix("platform")` produces `platform:mock:echo`.
2. `with_prefix("")` keeps the default — empty rejected silently.
3. `with_unqualified_name() + with_prefix("platform")` produces `platform:mock:echo` (last call wins).
4. `with_prefix("platform") + with_unqualified_name()` produces `echo` (last call wins).
5. Original default and `with_unqualified_name`-only paths still pass.

## Public-API impact

`entelix-mcp` baseline grows by 1 line (`with_prefix` method). Nothing else changes.

## Consequences

- The full namespace knob set is now: default (`mcp:{s}:{t}`), unqualified bare, custom prefix. Three explicit, named modes covering the operator use cases without leaking the implementation detail (private `metadata.name` field) into `pub` surface.
- The `mcp:` constant `MCP_NAME_PREFIX` stays as the default; `qualified_name(server, tool)` continues to return the `mcp:`-prefixed form (operators looking up adapters by name in a registry don't change behaviour).
- Operators integrating with prefix-rigid tool catalogues (the use case that motivated this slice) get a one-line solution.

## References

- v3 plan slice 115 (Phase B-1 — MCP tool prefix).
- ADR-0017 — `tenant_id` strengthening (the partner namespace concern; `with_prefix` is operator-side, tenant_id is system-side).
