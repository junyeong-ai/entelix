# ADR 0054 — MCP `sampling/createMessage` server-initiated channel

**Status**: Accepted
**Date**: 2026-05-01
**Decision**: Phase 8 of the post-7-차원-audit roadmap (second sub-slice)

## Context

ADR-0053 shipped MCP elicitation and explicitly deferred
sampling to its own slice — sampling has more surface area
(model preferences, system prompts, sampling parameters,
multimodal content), so it warranted dedicated design space.

This slice closes the deferred half. After it lands, every
MCP 1.5 server-initiated channel has a `*Provider` trait and
a dispatcher arm in `HttpMcpClient`:

- ✅ `roots/list` (10E, 2026-04-29)
- ✅ `elicitation/create` (ADR-0053)
- ✅ `sampling/createMessage` (this ADR)

## What is sampling?

MCP servers can ask the client to run an LLM completion on
their behalf. Typical uses:

- A server orchestrating tool dispatch wants the agent's LLM
  to choose the next tool.
- A server doing structured extraction wants natural-language
  reasoning capability it doesn't own.
- A server building a multi-step plan wants the agent's
  reasoning model in the loop.

The server provides:
- Conversation prefix (`messages` — user/assistant turns,
  text/image/audio content blocks).
- Optional `modelPreferences` (hint names, cost/speed/intelligence
  priorities in `[0, 1]`).
- Optional `systemPrompt`.
- Optional `includeContext` hint (`none` / `thisServer` /
  `allServers`).
- Optional `temperature`, `maxTokens`, `stopSequences`,
  vendor-opaque `metadata`.

Returns:
- `model` (which model the provider used — for audit / cost).
- `stopReason` (string per spec: `endTurn` / `stopSequence` /
  `maxTokens`).
- `role` (`assistant` typically).
- `content` (text / image / audio block).

## Decision

Add a `SamplingProvider` trait + request / response shapes in
a new `sampling.rs` module. Wire it into the dispatcher with
a new arm; advertise the capability iff a provider is wired.
Mirror the elicitation slice exactly so the dispatcher pattern
stays uniform.

### Public surface

```rust
pub struct SamplingMessage { pub role: String, pub content: SamplingContent }

#[non_exhaustive]
pub enum SamplingContent {
    Text { text: String },
    Image { data: String, mime_type: String },
    Audio { data: String, mime_type: String },
}

pub struct ModelPreferences {
    pub hints: Vec<ModelHint>,
    pub cost_priority: Option<f64>,
    pub speed_priority: Option<f64>,
    pub intelligence_priority: Option<f64>,
}

#[non_exhaustive]
pub enum IncludeContext { None, ThisServer, AllServers }

pub struct SamplingRequest {
    pub messages: Vec<SamplingMessage>,
    pub model_preferences: Option<ModelPreferences>,
    pub system_prompt: Option<String>,
    pub include_context: Option<IncludeContext>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    pub stop_sequences: Vec<String>,
    pub metadata: Option<serde_json::Value>,
}

pub struct SamplingResponse {
    pub model: String,
    pub stop_reason: String,
    pub role: String,
    pub content: SamplingContent,
}

#[async_trait]
pub trait SamplingProvider: Send + Sync + 'static + Debug {
    async fn sample(&self, request: SamplingRequest) -> McpResult<SamplingResponse>;
}

pub struct StaticSamplingProvider { /* new(response) / text(model, text) */ }
```

### Why no `ChatModel` adapter shipped here

The "wire `ChatModel` directly" adapter would force this
crate to depend on the chat-model surface. Concrete production
providers wrap a `ChatModel` themselves — convert MCP messages
→ `entelix_core::ir::Message` → `ChatModel::invoke` → MCP
response — and that conversion is deployment-specific:

- Which model gets used (the operator decides, possibly
  weighing the server's `modelPreferences`).
- Which prompt envelope (system prompt placement, persona
  injection).
- Which IR translation (text-only vs multimodal pass-through).

Forcing one shape via a built-in adapter would surprise every
operator with a different opinion. The trait surface stays
minimal so the conversion choices stay operator-side.

### Why stringly-typed `role` and `stop_reason`

The MCP spec doesn't enumerate these — `role` is open-ended
("user", "assistant", and theoretically more), `stop_reason`
is `"endTurn"` / `"stopSequence"` / `"maxTokens"` per spec but
servers may emit vendor-specific reasons too. A typed enum
with `Other(String)` would buy little — operators that care
inspect the string anyway. Pass through verbatim.

### Why three-variant `SamplingContent` (NOT `Vec<ContentBlock>`)

The MCP `sampling/createMessage` request and response carry a
single content block per message, not a list. The IR shape
(`Vec<ContentPart>`) is for assistants that interleave text
with tool calls — sampling has no tool-call concept. Single
block matches the wire shape.

### `#[non_exhaustive]` on enums

`SamplingContent` and `IncludeContext` are both
`#[non_exhaustive]` so future MCP spec revisions (additional
content types, more context modes) can land without breaking
caller pattern matches. Caught by
`scripts/check-surface-hygiene.sh`.

### `model_preferences` is optional all the way down

`ModelPreferences` itself is `Option`, and every field inside
is also `Option`. The provider may ignore the structure
entirely (single-model deployment), partially honour it (cost
priority but not speed), or fully honour it. The trait makes
no claim about how the provider weighs the priorities — that's
the whole point of putting them on the wire as hints.

### Tests

- 6 unit tests in `sampling.rs`: text content wire shape,
  image content wire shape, full request deserialization with
  optional fields, minimal request deserialization, response
  serialization with `stopReason` camel case, static text
  provider behaviour.
- 3 wiremock e2e tests in `streamable_sampling_e2e.rs`:
  - Server-initiated `sampling/createMessage` dispatches to
    static provider; response carries spec-shaped fields.
  - No provider wired → JSON-RPC `-32601` Method not found.
  - Capability advertised iff provider wired.

## Consequences

✅ MCP 1.5 server-initiated channel coverage is complete:
roots / elicitation / sampling all dispatch through their
own `*Provider` per the established pattern.
✅ The dispatcher in `HttpMcpClient::handle_server_request`
stays one-arm-per-method — the next MCP spec revision adding
a server-initiated channel follows the same template.
✅ Stringly-typed `role` / `stop_reason` / model identifiers
let operators pass vendor-specific reasons through without
SDK churn.
✅ `#[non_exhaustive]` on `SamplingContent` and
`IncludeContext` future-proofs against MCP spec evolution.
✅ Default impls (no method on `McpClient` itself) shield
existing custom-client implementors from breaking.
❌ Trait surface gained 9 new public types
(`SamplingProvider`, `SamplingRequest`, `SamplingResponse`,
`SamplingMessage`, `SamplingContent`, `ModelPreferences`,
`ModelHint`, `IncludeContext`, `StaticSamplingProvider`).
The doc structure groups them clearly.
❌ Producing a real LLM-backed provider is operator work —
the SDK ships only the trait + a static stub. A
`ChatModelSamplingProvider` companion (in a future slice if
demanded) could wrap the conversion behind a one-line builder.

## Alternatives considered

1. **Ship a `ChatModelSamplingProvider` adapter in this
   slice** — would couple `entelix-mcp` to a `ChatModel` trait
   surface and force one specific message-conversion shape on
   every operator. Better as a future companion if demand
   materialises.
2. **Typed enum for `stop_reason`** — `enum
   StopReason { EndTurn, StopSequence, MaxTokens, Other(String)
   }`. Forces every caller into a `match` for a value they
   often just log. The spec uses strings; pass through.
3. **Single rich `Content` enum unifying text/image/audio
   with parts** — would let one message carry interleaved
   text + image. The MCP spec restricts to one content block
   per message, so the shape would mislead.
4. **Required `max_tokens`** — the spec doesn't mandate it,
   but operators wrapping Anthropic vendors that demand it
   reject the request when missing. Keeping `max_tokens` as
   `Option` honours the wire shape; provider implementations
   handle the validation.
5. **Auto-derive `Serialize` on `SamplingResponse`** —
   straight derive works because the field-rename annotations
   (`#[serde(rename = "stopReason")]`) handle the camel-case
   cases. Custom impl wasn't needed (unlike
   `ElicitationResponse` which needed externally-tagged
   action serialization).

## Operator usage patterns

**Static stub for development**:
```rust
let provider = Arc::new(StaticSamplingProvider::text(
    "claude-3-sonnet",
    "stub response for local dev",
));
let config = McpServerConfig::http("server", url)?
    .with_sampling_provider(provider);
```

**Custom provider wrapping a ChatModel** (operator-side):
```rust
#[derive(Debug)]
struct ChatModelProvider {
    model: Arc<dyn ChatModel>,
    default_model_name: String,
}

#[async_trait]
impl SamplingProvider for ChatModelProvider {
    async fn sample(&self, req: SamplingRequest) -> McpResult<SamplingResponse> {
        // Convert MCP messages → IR
        let ir_messages = req.messages.iter().map(convert_message).collect();
        // Apply system prompt, sampling params...
        let ir_response = self.model.invoke(ir_messages, &ctx).await?;
        // Convert IR → MCP response
        Ok(SamplingResponse {
            model: self.default_model_name.clone(),
            stop_reason: convert_stop_reason(&ir_response),
            role: "assistant".into(),
            content: convert_content(&ir_response),
        })
    }
}
```

**Auto-decline (policy: server may not use my LLM)** — operator
simply doesn't wire a provider; the dispatcher returns
`-32601` Method not found, server respects the missing
capability.

## References

- ADR-0004 — native JSON-RPC client (parent decision tree).
- ADR-0011 amendment — Roots slice (the dispatcher pattern
  this slice extends, again).
- ADR-0053 — Elicitation slice (the immediate precedent
  this slice mirrors).
- 7-차원 roadmap §S9 — Phase 8 (MCP completion), second
  sub-slice.
- MCP 2024-11-05 spec §"Sampling" — wire-format source.
- `crates/entelix-mcp/src/sampling.rs` — trait + types +
  static provider + 6 unit tests.
- `crates/entelix-mcp/src/protocol.rs` — `SamplingCapability`
  shape + `ClientCapabilities::sampling` field.
- `crates/entelix-mcp/src/server_config.rs` —
  `with_sampling_provider` + accessor.
- `crates/entelix-mcp/src/client.rs` — capability
  advertisement + dispatcher arm.
- `crates/entelix-mcp/tests/streamable_sampling_e2e.rs` —
  3 wiremock e2e regressions.
