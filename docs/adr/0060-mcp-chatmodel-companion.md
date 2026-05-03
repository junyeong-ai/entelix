# ADR 0060 — `entelix-mcp-chatmodel` companion: ChatModel-backed `SamplingProvider`

**Status**: Accepted
**Date**: 2026-05-01
**Decision**: Phase 9 of the post-7-차원-audit roadmap (third sub-slice — operator-side completion of the MCP sampling channel)

## Context

ADR-0054 shipped the [`SamplingProvider`](../../crates/entelix-mcp/src/sampling.rs)
trait + the dispatcher arm in `entelix-mcp` that routes incoming
`sampling/createMessage` server-initiated requests onto a wired
provider. The trait stayed minimal by design — the in-tree
[`StaticSamplingProvider`] is a deterministic test stub, and the
real "wire to an LLM" path was deferred to operator-side code with
this comment in `sampling.rs`:

> ## Why a trait, not a `ChatModel` adapter shipped here
>
> A "wire `ChatModel` directly" adapter would force this crate to
> depend on the chat-model surface. Instead the trait stays
> minimal and operators write a 20-line wrapper that converts MCP
> messages → `entelix_core::ir::Message` → `ChatModel::invoke` →
> MCP response.

That decision kept `entelix-mcp` lean and was right at the trait
level. But the "20-line wrapper every operator writes themselves"
turned into ~250 lines of careful IR conversion + per-request
override plumbing + collapse-back that should not be reinvented
per deployment. ADR-0008's companion-crate pattern is exactly the
release valve: ship the bridge as a sibling crate that depends on
both `entelix-core` (for `ChatModel<C, T>`) and `entelix-mcp` (for
the provider trait), without forcing either dependency on users
who don't need the bridge.

## Decision

Add `entelix-mcp-chatmodel` (20th workspace member). It exports a
single public type:

```rust
pub struct ChatModelSamplingProvider<C: Codec + 'static, T: Transport + 'static> {
    chat: ChatModel<C, T>,
}
```

with `ChatModelSamplingProvider::new(chat)` and a `Clone` impl
(cheap — `ChatModel` is `Arc`-backed).

The struct implements [`SamplingProvider`] by:

1. Translating each `SamplingMessage.role` into IR `Role` (only
   `user` and `assistant` accepted — anything else is a
   protocol-spec violation surfaced as `McpError::Config`).
2. Translating each `SamplingContent` into a `ContentPart`:
   `Text` → `ContentPart::Text`; `Image`/`Audio` → respective
   variants with `MediaSource::Base64`. Future MCP variants land
   as a typed `tracing::warn!` + empty-text surrogate so the
   dispatch chain stays panic-free.
3. Cloning the wrapped `ChatModel<C, T>` per request and applying
   per-request overrides — `system_prompt` →
   `ChatModel::with_system`, `temperature` (f64 cast to f32) →
   `with_temperature`, `max_tokens` → `with_max_tokens`,
   `stop_sequences` → `with_stop_sequences`. The wrapped instance
   stays unchanged across concurrent dispatches.
4. Dispatching through `ChatModel::complete_full` — the full
   layer stack (`PolicyLayer`, `OtelLayer`, retry middleware) runs.
5. Collapsing the resulting `ModelResponse` back into a
   `SamplingResponse`: `model` is echoed verbatim, `stop_reason`
   maps via the canonical `endTurn` / `maxTokens` / `stopSequence`
   wire tokens (`StopReason::Other { raw }` passes through; future
   non-exhaustive variants warn + collapse to `endTurn`), and
   `content` picks the first emittable `Text` / `Image` / `Audio`
   block — multi-block responses (typical with
   `Thinking` + `Text`) drop the auxiliary blocks with a
   `tracing::warn!`.

### Why not a separate `ChatModel<C, T>` re-export

The companion's value is exactly the conversion logic, not a
re-export. Operators already have a `ChatModel<C, T>` from
`entelix-core`; they pass it into the companion. The crate is
purely a bridge.

### Why per-request `clone` rather than a long-lived configured `ChatModel`

A single sampling provider instance serves every request the MCP
server initiates. Per-request overrides (`system_prompt`,
`temperature`, …) are request-local — the wrapped `ChatModel`
must stay unchanged. `ChatModel::clone()` is cheap because
codec + transport + factory are all `Arc`-backed; only the
`ChatModelConfig` struct copies (a handful of `Option`s and small
`Vec`s).

### Why advisory MCP fields (`model_preferences`, `include_context`) are not enforced

The MCP spec describes both as hints, not contracts:

> The client decides how to honour this (the spec is intentionally
> vague — it's a hint, not a contract). [from
> `IncludeContext` doc on the trait]

Honouring `model_preferences.hints` would require the adapter to
reach into the `ChatModel<C, T>`'s `model` field and mutate it,
which (a) breaks the type-level model-codec compatibility (a
random hint string isn't guaranteed to work with the wired codec)
and (b) hides operator routing inside library code. The adapter
emits both fields via `tracing::debug!` so operator dashboards see
the request shape; operators with bespoke routing implement
`SamplingProvider` directly using this crate as a reference.

### Error mapping to JSON-RPC error codes

The MCP dispatcher converts `McpError::JsonRpc { code, message }`
to the wire `error` slot. The adapter maps:

- `entelix_core::Error::Provider` → `code: -32603` (internal error)
- `entelix_core::Error::InvalidRequest` / `Config` / `Auth` →
  `code: -32602` (invalid params)
- Everything else → `-32603`

`Cancelled` and `DeadlineExceeded` from the chat-model layer also
collapse to `-32603` — there's no MCP code for "request timeout"
distinct from internal error, so the message string carries the
discriminator.

### Why facade gates `mcp-chatmodel` behind both `mcp` and the new flag

The companion needs `entelix-mcp` types — that's a hard
dependency. Treating `mcp-chatmodel` as an extension of the
`mcp` feature (`mcp-chatmodel = ["mcp", "dep:entelix-mcp-chatmodel"]`)
keeps the facade Cargo flag taxonomy consistent with the
`vectorstores-pgvector` / `embedders-openai` pattern: a base
capability + opt-in companion behind their own flag.

### Tests

- 6 unit tests (`provider::tests`):
  - `translate_role` accepts `user` / `assistant`, rejects `system` / `tool` / empty.
  - `sampling_text_to_content_part` round-trips text.
  - `sampling_image_to_base64_image_part` round-trips image with mime preservation.
  - `stop_reason_canonical_mappings` covers every IR `StopReason` variant.
  - `first_emittable_content_skips_thinking_blocks`.
  - `first_emittable_content_empty_when_no_emittable`.
- 3 integration tests (`tests/sample_through_chat_model.rs`)
  drive a stub `Codec` + stub `Transport` end-to-end:
  - `sampling_request_translates_to_model_request_and_back` —
    captures the encode-time IR via the stub, verifies role
    sequence + per-request overrides + the response collapse.
  - `invalid_role_returns_error_to_dispatcher` — protocol-spec
    role surfaces as `McpError`.
  - `static_provider_still_satisfies_trait_object_use_path` —
    `Box<dyn SamplingProvider>` accepts both the in-tree
    `StaticSamplingProvider` and this adapter without API drift.

## Consequences

✅ Operators wire MCP sampling end-to-end in 5 lines:
```rust
let chat = ChatModel::new(codec, transport, "claude-3-5-sonnet");
let provider = Arc::new(ChatModelSamplingProvider::new(chat));
let server = McpServerConfig::http(url).with_sampling_provider(provider);
```
✅ ADR-0054's deferred operator-side completion now lands without
reopening the `entelix-mcp` crate. The trait surface is unchanged.
✅ The full `ChatModel` layer stack runs on every sampling
dispatch — operators get `PolicyLayer` redaction, `OtelLayer`
spans + cost emission, retry middleware uniformly across both
agent calls and MCP-initiated sampling. One observability surface,
two entry points.
✅ Wrapping is `Arc`-backed end-to-end — the per-request
`ChatModel::clone()` is cheap and the wrapped instance stays
shareable across concurrent dispatches.
✅ Multimodal supported — `Image` / `Audio` content survives the
round-trip with the original mime type.
✅ Workspace grows to 21 crates; new feature flag
`mcp-chatmodel` follows the `vectorstores-*` / `embedders-*`
companion taxonomy on the facade.
❌ The adapter currently doesn't honour
`model_preferences.hints` (would need cross-codec model-name
adaptation). Operators with multi-model routing implement
`SamplingProvider` directly. Documented in the ADR + crate doc.
❌ `Refusal` / `ToolUse` IR stop reasons collapse to `endTurn` on
the wire — MCP's stringly-typed surface has no closer match.
Operators wanting sharper semantics implement `SamplingProvider`
directly.
❌ Public-API baseline grew (1 new crate baseline,
`entelix-mcp-chatmodel.txt` 18 lines).

## Alternatives considered

1. **Extend `entelix-mcp` itself with the adapter** — drops the
   companion-crate pattern (ADR-0008) and adds a `ChatModel`
   dependency to the lean MCP crate. Rejected.
2. **Generic `LlmAdapter` trait inside `entelix-mcp` that
   `ChatModel` implements** — invents a new abstraction layer
   over `ChatModel` for one use-case. Single companion crate is
   more direct. Rejected.
3. **Honour `model_preferences.hints` by mutating the chat
   model's `model` field** — breaks the codec / model
   compatibility invariant (a hint string isn't guaranteed to
   match the wired codec) and hides routing inside library code
   that operator dashboards can't inspect. Rejected.
4. **Synchronously block on a `model_preferences` callback the
   operator wires** — pushes the routing complexity onto the
   operator while pretending the adapter handles it.
   Asymmetrically worse than just letting the operator implement
   `SamplingProvider` directly. Rejected.
5. **Surface vendor-specific stop reasons as a structured enum** —
   the MCP wire format is stringly-typed; mapping IR `StopReason`
   into a new enum and serialising back to strings is extra
   surface for no net gain. Pass-through via `Other { raw }` is
   the right shape. Rejected.

## Operator usage patterns

**Default deployment** (one model wired through the agent's chat
model also serves sampling):

```rust
use std::sync::Arc;
use entelix_core::ChatModel;
use entelix_mcp::McpServerConfig;
use entelix_mcp_chatmodel::ChatModelSamplingProvider;

let chat = ChatModel::new(codec, transport, "claude-3-5-sonnet")
    .with_max_tokens(1024)
    .layer(otel_layer)
    .layer(policy_layer);
let provider = Arc::new(ChatModelSamplingProvider::new(chat));

let server = McpServerConfig::http("https://server.example/mcp")
    .with_sampling_provider(provider);
```

**Multi-model deployment** (different model for sampling vs
agent), implemented as separate `ChatModel<C, T>` instances:

```rust
let agent_chat = ChatModel::new(...);
let sampling_chat = ChatModel::new(...);  // smaller / cheaper
let provider = Arc::new(ChatModelSamplingProvider::new(sampling_chat));
```

**Bespoke routing** (operator owns `model_preferences` honouring):
implement `SamplingProvider` directly using `provider.rs` as a
reference; the conversion helpers are crate-internal but the
shape is small enough to reproduce.

## References

- ADR-0008 — companion crate pattern (parent — concrete impls
  live in sibling crates).
- ADR-0054 — MCP sampling/createMessage trait surface (parent —
  this slice closes its operator-side).
- `crates/entelix-mcp/src/sampling.rs` — `SamplingProvider` trait
  + `StaticSamplingProvider` (in-tree test stub).
- `crates/entelix-core/src/chat.rs` — `ChatModel<C, T>` and the
  per-request `clone()` cost story.
- `crates/entelix-mcp-chatmodel/` — the new 20th workspace
  member.
- `crates/entelix/Cargo.toml` — `mcp-chatmodel` feature flag.
