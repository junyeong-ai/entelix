# ADR 0033 — LLM / operator channel separation (invariant #16)

**Status**: Accepted
**Date**: 2026-05-01
**Decision**: Phase 2 of the post-7-차원-audit roadmap

## Context

The 7-차원 LLM-context economy audit (R1–R7) found seven sites where
operator-only content was leaking into the model's view:

- **R1** `SchemaToolAdapter::execute` — `std::any::type_name::<T::Input>()`
  surfaced internal Rust paths (`entelix_tools::http_fetch::FetchInput`)
  in deserialization errors the model sees.
- **R2** `SchemaToolAdapter::new` — `schemars::schema_for!` output
  shipped to every codec verbatim, including `$schema`, `title`,
  `$defs`, `$ref`, and integer width hints (`format: "int64"`).
  30–120 tokens per tool per request × every turn.
- **R3** `HttpFetchTool` — every response header (`set-cookie`,
  `cf-ray`, `x-amz-request-id`, `via`, …) reached the model.
  Often hundreds of tokens of vendor chrome the model cannot use.
- **R4** `QuerySemanticMemoryTool` — raw `score: f32` (cosine
  distance) plus the entire `metadata` `Value` (including backend
  internals like `namespace_key`, `embedding_hash`).
- **R5** `ListEntityFactsTool` — every entity carried two RFC3339
  timestamps (`created_at`, `last_seen`), 32 bytes each.
- **R6** `react_agent` tool dispatch — `format!("error: {e}")`
  echoed the operator-facing `Display` (vendor status, `hint:`,
  source chain) into the model's tool-result content.
- **R7** `agent::event::AgentEvent::ToolError → GraphEvent::ToolResult`
  audit projection — same R6 leak rode through SessionGraph and
  reappeared in the model's view on replay/resume.

Each site individually was small. Aggregated, they cost hundreds of
tokens per turn, exposed prompt-injection surface (`hint:` text the
operator wrote becomes content the model treats as instructions),
and burned model attention with `provider returned 503` framing the
model cannot act on.

The defect was structural — every site used `Display` /
`serde_json::to_value` to serve both operators (logs, OTel, sinks)
and the model (tool-result content). One channel, two consumers,
opposing requirements.

## Decision

Add invariant **#16 — LLM / operator channel separation** to
CLAUDE.md and enforce it through three SDK-wide pieces:

1. **`LlmFacingError` trait** — `render_for_llm() -> String`. Default
   impl on `entelix_core::Error` returns short, model-actionable
   messages. Vendor status, `hint:`, source chains, RFC3339
   timestamps, and internal type identifiers are deliberately omitted.
2. **`LlmFacingSchema::strip(&Value) -> Value`** — JSON-Schema
   sanitiser that walks the schema tree, resolves `$ref`/`$defs`
   indirection inline, drops envelope keys (`$schema`, `title`),
   and discards integer width hints (`format: "int64"`). Only
   vendor-honored keys (`type`, `properties`, `required`, `items`,
   `enum`, …) survive.
3. **Built-in tool exposure knobs** — `HttpFetchToolBuilder::with_exposed_response_headers`,
   `MemoryToolConfig::expose_metadata_fields` /
   `with_entity_temporal_signals`. All default to "off"; operators
   opt specific fields in.

> **16. LLM / operator channel separation** — `Display` / source
> chains / vendor status / `hint:` / internal type identifiers
> never reach the model. Tool errors flow to the model through
> `LlmFacingError::render_for_llm`; tool input schemas through
> `LlmFacingSchema::strip`; tool outputs through default-deny
> exposure knobs (HTTP headers, memory metadata, temporal signals).
> Operator channels (event sinks, OTel, logs) carry the full
> diagnostic.

### Concrete contracts

| Site | Old behaviour | New behaviour |
|---|---|---|
| `Error → tool_result` (`react_agent`) | `format!("error: {e}")` echoed `Display` | `e.render_for_llm()` — short, actionable |
| `AgentEvent::ToolError` audit projection | `error: String` field used for both audit and replay | two fields: `error` (operator) + `error_for_llm` (audit-replay → model). Projection picks `error_for_llm` |
| `SchemaToolAdapter::new` schema | schemars output verbatim | `LlmFacingSchema::strip(schema)` |
| `SchemaToolAdapter::execute` deserialize error | `"input failed to deserialize as `entelix_tools::…::FooInput`: {e}"` | `"tool '{name}': input did not match schema: {e}"` |
| `HttpFetchTool` response headers | every header in output | empty map by default; `with_exposed_response_headers([...])` opts in |
| `QuerySemanticMemoryTool` results | `{content, metadata, score}` | `{rank, content}`; `metadata` only when `expose_metadata_fields` allowlists keys |
| `ListEntityFactsTool` results | `{entity, fact, created_at, last_seen}` (RFC3339) | `{entity, fact}`; integer day-counts when `with_entity_temporal_signals(true)` |

### Enforcement

`crates/entelix-tools/tests/llm_context_economy.rs` regression-checks
all of the above end-to-end:

- `Error::render_for_llm` rejects `provider returned`, vendor status
  codes, `hint:`, and source chain text.
- `LlmFacingSchema::strip` drops `$schema`/`$defs`/`title`/`$ref`
  recursively across nested user-named properties.
- Every built-in tool's wire schema (semantic memory query, entity
  facts, skill tools) is asserted free of schemars envelope keys.
- `query_semantic_memory` default output carries `rank` not `score`
  and no metadata; allowlist mode filters backend internals.
- `list_entity_facts` default output is RFC3339-free; opted-in
  temporal signals are integer day counts.

The audit-projection test in `entelix-agents/src/agent/event.rs`
asserts `GraphEvent::ToolResult.content` carries the
`error_for_llm` field and never echoes operator-facing patterns.

## Consequences

✅ Per-turn token cost down 200–700 tokens for typical agent
configurations (six tools, a few memory queries).
✅ Operator hints are no longer instructions in the model's view —
prompt-injection surface narrowed.
✅ Tool errors read like tool errors, not vendor stack traces — the
model has actionable text, the operator has full diagnostics on the
sink/OTel channels.
✅ Adding a new built-in tool surfaces R3-class regressions
immediately — the gate test asserts schemas and outputs at the
public registry boundary.
❌ Operators upgrading to the new memory tool config must opt in any
metadata fields the model previously read. One-line change at the
`MemoryToolConfig::new()` site.
❌ The default response headers vanish from `HttpFetchTool` output;
operators that branched on `content-type` add
`with_exposed_response_headers(["content-type"])`.

## Alternatives considered

1. **Strip only at codec time** — schemars envelope would still
   round-trip through `ToolMetadata`, defeating audit and consuming
   memory. Rejected.
2. **One `LlmFacing` trait covering errors and values together** —
   forces every value type to implement the trait. The split keeps
   `Error` the only mandatory implementor; tool outputs use
   default-deny knobs that are simpler to audit. Rejected.
3. **Per-tool `LlmFacing` overrides instead of allowlists** — moves
   the policy decision to library authors who don't know operator
   intent. Rejected; allowlists keep the decision with the operator.

## References

- 7-차원 audit fork report `audit-llm-context-economy` R1–R7.
- ADR-0027 — Skills progressive disclosure (T1/T2/T3) — same
  principle of default-deny exposure surfaces.
- ADR-0032 — invariant #15 (silent fallback prohibition) — sibling
  contract: operator-channel signal must remain visible to operators.
