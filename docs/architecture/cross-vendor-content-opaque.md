# Cross-vendor opaque round-trip carrier (`ProviderEchoSnapshot`)

## Problem

Vendors emit per-turn **opaque round-trip tokens** the harness must echo verbatim on the next request. Each vendor names and shapes the token differently, but the contract is identical: *opaque to the harness, mandatory on the wire, lost = next turn rejected or degraded*.

Verified against 2026-05 vendor docs:

| Vendor | Token | Wire location | Level | Echo failure mode |
|---|---|---|---|---|
| Anthropic Messages | `signature` | `content[].signature` (`type: "thinking"`) | Part | Tool round → HTTP 400; non-tool round → silent thinking-disabled |
| Anthropic Messages | `redacted_thinking.data` | `content[].data` (`type: "redacted_thinking"`) | Part | Same as `signature` (full block round-trip) |
| Anthropic on Bedrock Converse | `reasoningText.signature` + `redactedContent` | `content[].reasoningContent.*` | Part | Hash-validated; any tampering throws |
| Amazon Nova 2 (Bedrock Converse) | `signature` (same shape as Anthropic on Bedrock) | `content[].reasoningContent.reasoningText.signature` | Part | Required for extended-thinking continuity |
| Anthropic on Vertex AI | `signature` (wire-shape identical to first-party) | `content[].signature` | Part | Same as first-party Anthropic |
| Gemini (AI Studio + Vertex) 3.x | `thought_signature` (snake_case strict on Vertex) | `Part.thought_signature` on `text` / `functionCall` / thinking parts | Part | First `functionCall` of step missing → HTTP 400 `INVALID_ARGUMENT`; trailing `text` missing → recommended only |
| OpenAI Responses | `encrypted_content` | `output[].encrypted_content` (`type: "reasoning"`, opt-in via `include`) | Part | `store: false` + dropped → silent CoT loss |
| OpenAI Responses | `previous_response_id` | request root | **Response** | Optional alternative to manual echo; 30-day TTL when `store: true` |
| OpenAI Responses | item `id` (`rs_…` / `fc_…` / `msg_…`) | per-item | Part | Required when echoing prior output items in stateless mode |
| OpenAI Responses | `code_interpreter_call.container_id` | per-item | Part | Required for cross-call container reuse |
| xAI Grok 4.x | `reasoning.encrypted_content` | OpenAI-Responses-shaped | Part | Same as OpenAI Responses |
| Azure OpenAI Responses | `encrypted_content` (**API-key-scoped**) | per-item | Part | Switching API key invalidates the blob |
| DeepSeek V4 (Thinking Mode) | `reasoning_content` (text) | assistant message field | Part | Empty string echo required; omission → HTTP 400 |

Out of scope vendors (no server-issued opaque round-trip token in 2026-05 docs): Cohere v2 (full message echo only), Mistral first-party (text-only `[THINK]` traces), Bedrock Llama / Mistral / Cohere (tool-call ids only), Vertex partner-MaaS Llama / Mistral, Azure OpenAI Chat Completions, OpenAI Chat Completions.

The IR currently encodes one of these tokens (`Thinking::signature: Option<String>`) as a vendor-named field, which:

1. Forces the IR to grow a new field every time a vendor adopts the pattern.
2. Couples downstream consumers to a vendor-specific concept they should ignore.
3. Cannot represent vendors whose token attaches to non-thinking parts (Gemini's `text` and `functionCall`), response root (OpenAI `previous_response_id`), or carries multi-field payloads (Anthropic `redacted_thinking.data` has no associated text).

The Gemini 3.x rollout exposes the cost in production: multi-turn tool dispatches against any reasoning-tier Gemini model fail with `function call ... is missing a thought_signature`.

## Promotion criterion (invariant 22)

Five distinct native vendors carry the concept (Anthropic, Amazon Nova, OpenAI Responses, xAI Grok, DeepSeek V4 Thinking Mode); promoting to cross-vendor IR is mandated.

## Design

One vendor-keyed opaque carrier rides on every IR site that may transport a round-trip token.

```rust
/// Vendor-issued opaque data the harness must echo verbatim on the
/// next turn. Codecs decode their own vendor's blob into this carrier
/// and read it back verbatim on the encode side. The harness never
/// inspects the payload — it forwards whichever blob lands at decode
/// time, untouched. The type is open-constructable (no sealed
/// constructor bottleneck) so external codec crates can stamp their
/// own provider key — a prerequisite for invariant 22's "new vendor =
/// one codec impl, zero IR change" promise. Codec autonomy is
/// enforced by convention plus the `cross_vendor_*_isolation_*`
/// regression suite (`tests/provider_echo_round_trip.rs`), not by
/// type-level visibility.
///
/// Cross-vendor: a transcript may carry entries for multiple vendors
/// after a transport switch; each codec only emits its own entries on
/// the wire and silently leaves the other vendor's blob alone in IR.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProviderEchoSnapshot {
    /// Provider key matching `Codec::name`
    /// (`"anthropic-messages"`, `"gemini"`, `"openai-responses"`,
    /// `"bedrock-converse"`, `"vertex-anthropic"`, `"vertex-gemini"`).
    /// Codecs select their own blobs by this key.
    pub provider: Cow<'static, str>,
    /// Vendor-defined opaque payload. Typically a JSON object whose
    /// keys are the vendor's wire field names (`signature`,
    /// `thought_signature`, `encrypted_content`, `data`,
    /// `redacted_content`, `container_id`, `id`). The harness never
    /// reads or rewrites these keys — only the originating codec
    /// does.
    pub payload: serde_json::Value,
}
```

### IR surface — three carriers, one shape

The carrier rides on **part**, **response**, and **request** levels — each maps to one of the three echo scopes seen in the wild.

```rust
// Part-level — every ContentPart variant gains:
provider_echoes: Vec<ProviderEchoSnapshot>
// Anthropic signature, Anthropic redacted_thinking.data, Gemini
// thought_signature, OpenAI encrypted_content, OpenAI item ids,
// Bedrock reasoningText.signature + redactedContent, Nova 2 signature,
// xAI encrypted_content, code_interpreter container_id.

// Response-level — ModelResponse gains:
provider_echoes: Vec<ProviderEchoSnapshot>
// OpenAI Responses Response.id (so the next ModelRequest can chain
// via previous_response_id), OpenAI Responses compaction-item return.

// Request-level — ModelRequest gains:
continued_from: Vec<ProviderEchoSnapshot>
// OpenAI previous_response_id (the chain pointer the harness sends),
// future vendors that adopt the same pattern.
```

`Vec<…>` (not `Option<…>`) on every site so a transcript that has crossed transports retains all vendors' blobs simultaneously (cross-vendor harmlessness — the IR is an audit-faithful record, not a single-vendor projection).

`#[serde(default, skip_serializing_if = "Vec::is_empty")]` everywhere so the absence cost is zero in serialised IR.

### Streaming surface — codec pre-wraps

Vendor-shape tokens never reach `StreamDelta`. The codec that decoded the wire constructs the `ProviderEchoSnapshot` and yields it on the delta:

```rust
pub enum StreamDelta {
    // …
    ThinkingDelta {
        text: String,
        provider_echoes: Vec<ProviderEchoSnapshot>,
    },
    // text/tool deltas may also carry provider_echoes for vendors
    // (Gemini) whose signature attaches to text/functionCall parts.
}
```

`StreamAggregator` stays codec-agnostic — it concatenates `text`, extends `provider_echoes`, and emits `ContentPart::*` with the merged carrier on close.

### Audit surface — same shape

```rust
GraphEvent::ThinkingDelta {
    text: String,
    provider_echoes: Vec<ProviderEchoSnapshot>,
    timestamp: DateTime<Utc>,
}
```

`SessionAuditSink` round-trips the carrier via the existing `record_*` channel; replay-from-checkpoint hands the model exactly the bytes it issued.

### Codec autonomy invariant

A codec **only reads and writes entries whose `provider` matches its own `Codec::name()`**. This is type-checked indirectly via codec impl review and asserted by `tests/codec_isolation.rs`:

- Cross-vendor passthrough: a transcript with entries for `"anthropic-messages"` *and* `"gemini"` sent to either codec → only the matching entries reach the wire; the other vendor's blob survives in IR for downstream consumers.
- No accidental promotion: codecs never inject defaults, never coerce a missing entry into a fabricated value (invariant 15 — silent fallback).

**Provider key is the wire-shape ID, not the codec ID.** The key identifies the *shape of the opaque blob on the wire*; two codecs that emit the identical wire shape share one key. This decouples opaque-token semantics from transport-layer differences (URL routing, body markers, header conventions):

- `"anthropic-messages"` — first-party Anthropic + Vertex AI Anthropic (wire shape identical, only the routing differs).
- `"bedrock-converse"` — Anthropic Claude + Amazon Nova 2 on Bedrock (identical `reasoningContent.signature` shape across both model families; codec branches internally on model-family-specific behaviour).
- `"gemini"` — AI Studio Gemini + Vertex AI Gemini (identical `Part.thought_signature` shape).
- `"openai-responses"`, `"openai-chat"` — single codec, single shape.

Codec composition follows: `VertexAnthropicCodec` composes `AnthropicMessagesCodec` and inherits its provider key transparently; `VertexGeminiCodec` composes `GeminiCodec` likewise. Operators switching transports within the same wire shape get automatic signature transfer (which is correct — Anthropic explicitly documents cross-platform compatibility of signatures). Switching *across* wire shapes (first-party Anthropic ↔ Bedrock Converse) keeps both vendors' blobs in IR for audit, and each codec only emits its own key on the wire.

### Wire mapping — verified examples

**Anthropic Messages — `signature` round-trip** (assistant `thinking` block):

```json
// Wire (decode):
{"type": "thinking", "thinking": "let me reason…", "signature": "WaUjzkypQ2…"}

// IR ContentPart::Thinking:
{
    "type": "thinking",
    "text": "let me reason…",
    "provider_echoes": [{
        "provider": "anthropic-messages",
        "payload": {"signature": "WaUjzkypQ2…"}
    }]
}

// Wire (encode next turn):
{"type": "thinking", "thinking": "let me reason…", "signature": "WaUjzkypQ2…"}
```

**Anthropic Messages — `redacted_thinking` round-trip** (Claude 3.7-only block; absent on Claude 4.x):

```json
// Wire (decode):
{"type": "redacted_thinking", "data": "EmwKAhgBEgy3va3pzix/…"}

// IR — new ContentPart variant `RedactedThinking`:
{
    "type": "redacted_thinking",
    "provider_echoes": [{
        "provider": "anthropic-messages",
        "payload": {"data": "EmwKAhgBEgy3va3pzix/…"}
    }]
}
```

The `data` field carries no harness-readable text; the new IR variant `ContentPart::RedactedThinking` exists *only* to round-trip the carrier (and to satisfy invariant 6 — the prior silent-drop is replaced by a typed channel).

**Bedrock Converse — Anthropic Claude or Nova 2 reasoning** (single codec, single provider key):

```json
// Wire (decode, Anthropic OR Nova on Bedrock):
{
    "reasoningContent": {
        "reasoningText": {"text": "…", "signature": "…"},
        "redactedContent": "<base64>"
    }
}

// IR ContentPart::Thinking:
{
    "type": "thinking",
    "text": "…",
    "provider_echoes": [{
        "provider": "bedrock-converse",
        "payload": {"signature": "…", "redacted_content": "<base64>"}
    }]
}
```

The codec transparently handles both Anthropic-on-Bedrock and Nova-on-Bedrock under the single `"bedrock-converse"` key. Model-family branching (Anthropic vs Nova feature differences) lives inside `BedrockConverseCodec`, never on the IR.

**Gemini — `thought_signature` round-trip** on `functionCall` part (Vertex strict snake_case):

```json
// Wire (decode):
{"functionCall": {"name": "get_weather", "args": {…}}, "thought_signature": "EhsM…"}

// IR ContentPart::ToolUse:
{
    "type": "tool_use",
    "id": "<derived>",
    "name": "get_weather",
    "input": {…},
    "provider_echoes": [{
        "provider": "gemini",
        "payload": {"thought_signature": "EhsM…"}
    }]
}

// Wire (encode next turn — snake_case mandatory on Vertex, AI Studio
// also accepts):
{"functionCall": {"name": "get_weather", "args": {…}}, "thought_signature": "EhsM…"}
```

Gemini's signature attaches to `text`, `functionCall`, and thinking parts — the codec must populate `provider_echoes` on every `ContentPart` variant the wire carries it on (verified: 400 `INVALID_ARGUMENT` if the first `functionCall` of a step is missing the signature on the next request).

**OpenAI Responses — `encrypted_content` (part) + `previous_response_id` (response/request)**:

```json
// Wire response (decode, opt-in via include: ["reasoning.encrypted_content"]):
{
    "id": "resp_abc…",
    "output": [
        {"id": "rs_def…", "type": "reasoning", "summary": [], "encrypted_content": "gAAAAABoISQ24…"}
    ]
}

// IR ModelResponse:
{
    "content": [
        {
            "type": "thinking",
            "text": "",
            "provider_echoes": [{
                "provider": "openai-responses",
                "payload": {"id": "rs_def…", "encrypted_content": "gAAAAABoISQ24…"}
            }]
        }
    ],
    "provider_echoes": [{
        "provider": "openai-responses",
        "payload": {"response_id": "resp_abc…"}
    }]
}

// IR next ModelRequest (chain via previous_response_id):
{
    "messages": [/* new user turn only */],
    "continued_from": [{
        "provider": "openai-responses",
        "payload": {"previous_response_id": "resp_abc…"}
    }]
}
```

When the operator runs in stateless mode (`store: false`), the codec emits no `previous_response_id` chain pointer; instead it echoes the prior `output[]` items by reading their `provider_echoes` payloads (which carry the per-item `id`).

### Properties

- **IR is vendor-neutral.** Adding vendor X with its own opaque token requires zero IR changes — the codec serialises into / out of `provider_echoes` with `provider: "x"`.
- **No backwards-compatibility shims.** `Thinking::signature` is removed in the same PR (invariant 14). `StreamDelta::ThinkingDelta::signature` and `GraphEvent::ThinkingDelta::signature` are removed in the same PR. No deprecated alias, no migration helper. Persisted sessions older than the cutover are not preserved (entelix is pre-1.0).
- **Default-deny exposure.** `provider_echoes` is `Serialize + Deserialize` for IR persistence (session log, checkpointer) but never reaches the model — codecs serialise it onto the wire, never into the visible content channel.
- **Cross-codec passthrough.** A transcript carrying entries for multiple vendors round-trips through any codec without affecting wire bytes; foreign entries survive in IR.
- **One provider key per codec** (matches `Codec::name()` 1:1). Codecs hosting multiple model families (Bedrock Converse → Anthropic + Nova) share one key and branch internally.
- **`Vec`, not `Option`.** A transcript that survived a vendor switch retains both vendors' blobs simultaneously — the IR is audit-faithful, not single-vendor.

### `redacted_thinking` IR variant

The current IR has no representation for Anthropic's `redacted_thinking` block; codecs silently drop it on decode (invariant 6 violation). This PR introduces a typed variant so the round-trip carrier is the only payload it needs:

```rust
pub enum ContentPart {
    // …existing variants…

    /// A reasoning block the safety system flagged for redaction.
    /// Carries no harness-readable text — the entire block is an
    /// opaque round-trip artifact preserved in `provider_echoes`.
    /// Emitted by Claude 3.7; Claude 4.x and later do not produce
    /// this variant. Codecs that don't recognise it on encode emit
    /// `LossyEncode`.
    RedactedThinking {
        provider_echoes: Vec<ProviderEchoSnapshot>,
    },
}
```

## Out of scope (explicit)

- Compatibility shims for `Thinking::signature`, `StreamDelta::ThinkingDelta::signature`, `GraphEvent::ThinkingDelta::signature` — all removed in the same PR.
- An operator-facing API to inspect or rewrite `provider_echoes` payloads. The contract is opaque-to-harness; if a future use case requires inspection (e.g. cost attribution by reasoning bytes), it lands as a new typed surface, not by reaching into the JSON payload.
- Per-credential "scope fingerprint" on the carrier (Azure / xAI `encrypted_content` is API-key-scoped). Documented as an operator concern; deferred until a real load-balancing failure mode lands. Operators currently work around by binding a transport to a single credential.
- Response-level carrier on `StreamDelta::Stop` (some vendors emit response-id only at stream close). Streaming `previous_response_id` chaining works because the response-level carrier rides on the aggregator's terminal `ModelResponse`; streaming `Stop` itself stays vendor-neutral.

## Phased work plan

Each phase is a self-contained green-build commit on `feat/cross-vendor-content-opaque`. The branch lands on `main` as a single squash commit when phase 8 passes.

### Phase 0 — RFC lock (this document)

This document is the deliverable. Decisions locked: carrier name `ProviderEchoSnapshot`, field name `provider_echoes`, three-tier IR surface (part / response / request), codec pre-wrap streaming, single Bedrock provider key, response-level included in same PR, Gemini snake_case fix bundled, `redacted_thinking` variant added.

### Phase 1 — IR types

- New module `crates/entelix-core/src/ir/provider_echo.rs` defining `ProviderEchoSnapshot` plus helpers (`new`, `for_provider`, `payload_field`).
- `crates/entelix-core/src/ir/content.rs`:
  - All 11 existing `ContentPart` variants gain `provider_echoes: Vec<ProviderEchoSnapshot>` with `#[serde(default, skip_serializing_if = "Vec::is_empty")]`.
  - New 12th variant `ContentPart::RedactedThinking { provider_echoes }`.
  - `Thinking::signature` deleted.
  - Helper constructors default `provider_echoes: Vec::new()`. Add `with_provider_echo(self, ProviderEchoSnapshot) -> Self` for codec construction.
- `crates/entelix-core/src/ir/response.rs` — `ModelResponse` gains `provider_echoes: Vec<ProviderEchoSnapshot>`.
- `crates/entelix-core/src/ir/request.rs` — `ModelRequest` gains `continued_from: Vec<ProviderEchoSnapshot>`.
- `ir/mod.rs` re-exports `ProviderEchoSnapshot`.

### Phase 2 — Stream + audit IR

- `crates/entelix-core/src/stream.rs`:
  - `StreamDelta::ThinkingDelta::signature` deleted; replaced with `provider_echoes: Vec<ProviderEchoSnapshot>`.
  - `PendingThinking::signature` deleted; replaced with `provider_echoes: Vec<ProviderEchoSnapshot>`.
  - `StreamAggregator::flush_thinking` extends accumulator instead of overwriting.
  - Text and tool deltas may carry `provider_echoes` (Gemini attaches to non-thinking parts) — extend `StreamDelta::TextDelta` and `StreamDelta::ToolUseStart` accordingly.
- `crates/entelix-session/src/event.rs` — `GraphEvent::ThinkingDelta::signature` deleted; replaced with `provider_echoes`.

### Phase 3 — Anthropic family codecs (Anthropic Messages + Bedrock Converse + Vertex Anthropic)

- `crates/entelix-core/src/codecs/anthropic.rs`:
  - Decode `signature` on `thinking` blocks → `provider_echoes` with `provider: "anthropic-messages"`, payload `{ "signature": "<value>" }`.
  - Decode `redacted_thinking` blocks → new `ContentPart::RedactedThinking` with `provider_echoes` payload `{ "data": "<value>" }`.
  - Encode reads matching entry; missing → no `signature` on wire (matches behaviour for fresh thinking blocks).
  - Streaming `signature_delta` event pre-wraps into `provider_echoes` before yielding `StreamDelta::ThinkingDelta`.
- `crates/entelix-core/src/codecs/bedrock_converse.rs`:
  - Decode `reasoningContent.reasoningText.signature` and `reasoningContent.redactedContent` → `provider_echoes` with `provider: "bedrock-converse"`, payload `{ "signature": "<value>", "redacted_content": "<base64>" }` (single carrier per part).
  - Encode reads matching entry; codec branches internally on model family (Anthropic vs Nova 2) for any feature differences.
- `crates/entelix-core/src/codecs/vertex_anthropic.rs`:
  - Inherits Anthropic Messages behaviour. Provider key: `"vertex-anthropic"` (matches `Codec::name`).
- Codec unit tests updated — `Thinking { signature: … }` references rewritten.

### Phase 4 — Gemini family codecs

- `crates/entelix-core/src/codecs/gemini.rs`:
  - **Wire field name fix**: `thoughtSignature` (camelCase) → `thought_signature` (snake_case) on encode and decode. Vertex strictly rejects camelCase; AI Studio accepts both. Snake_case is the safe contract.
  - Decode every `parts[]` entry that carries `thought_signature` (regardless of inner shape — `text`, `thinking`, `functionCall`) → `provider_echoes` with `provider: "gemini"`, payload `{ "thought_signature": "<value>" }`.
  - Encode reads matching entry on every part type the wire format permits; encoder writes `thought_signature` field on the same wire object.
  - Decode `functionCall.id` → `provider_echoes` payload key `"function_call_id"` (Gemini's vendor-issued correlation id; required for parallel-call ordering invariants on Gemini 3.x).
  - Streaming: signatures may arrive on parts with empty text; aggregator carrier-merge handles it.
- `crates/entelix-core/src/codecs/vertex_gemini.rs`:
  - Inherits Gemini behaviour through composition. Provider key: `"vertex-gemini"`.

### Phase 5 — OpenAI Responses

- `crates/entelix-core/src/codecs/openai_responses.rs`:
  - Decode `output[].encrypted_content` (when present) → part-level `provider_echoes` on the corresponding reasoning content with `provider: "openai-responses"`, payload `{ "id": "<rs_…>", "encrypted_content": "<value>" }`.
  - Decode `Response.id` → response-level `ModelResponse.provider_echoes` with `provider: "openai-responses"`, payload `{ "response_id": "<resp_…>" }`.
  - Encode reads `ModelRequest.continued_from` for `previous_response_id`; writes request root `previous_response_id` field.
  - In stateless mode (`store: false`), the encoder echoes prior `output[]` items by reading their `provider_echoes` per-item `id`.
  - Built-in tool item ids (`web_search_call.id`, `file_search_call.id`, `code_interpreter_call.id`, `code_interpreter_call.container_id`, `computer_call.id`) → `provider_echoes` on the corresponding `ContentPart` with payload key `"id"` / `"container_id"`.
- `crates/entelix-core/src/codecs/openai_chat.rs`:
  - No round-trip token in Chat Completions. Chat codec drops any incoming `provider_echoes` with `provider: "openai-responses"` on encode (foreign entries) and emits no `LossyEncode` (no information lost — Chat Completions has no carrier slot).

### Phase 6 — Mechanical fix on every consumer

- `cargo build --workspace --all-features` lists every site. Three buckets:
  - **Constructors** (~160 sites): add `provider_echoes: Vec::new()` (or chain `with_provider_echo(...)` from codec construction).
  - **Pattern matches with full field binding** (~250 sites): add `provider_echoes: _` (or `..` where the binding has no consumer).
  - **Forward-required sites** (highest risk): codec encoders, `StreamAggregator::flush_*`, compaction (`entelix-session/src/compaction.rs`, `entelix-agents/src/compaction.rs`), summarizer (`entelix-agents/src/summarizer.rs`), MCP bridge (`entelix-mcp/src/chatmodel.rs`), audit serialisation (`entelix-session/src/event.rs`). These must explicitly preserve `provider_echoes` across reconstruction or consciously drop with rationale.
- Crates touched (recursive): `entelix-core`, `entelix-tools`, `entelix-agents`, `entelix-session`, `entelix-mcp`, `entelix-graph`, `entelix-runnable`, `entelix-policy`, `entelix-otel`, `entelix-prompt`, `entelix-cloud`, `entelix-server`, `entelix-persistence`, `entelix-memory`, `entelix-auth-claude-code`.

### Phase 7 — Test sweep

- `crates/entelix-core/tests/codec_robustness_proptest.rs` — proptest generator regenerated with new `ContentPart` shape (12 variants, `provider_echoes` field).
- `crates/entelix-core/tests/lossy_warning_completeness.rs` — `Thinking::signature` row removed; `RedactedThinking` row added (no LossyEncode for codecs that handle it; LossyEncode for Gemini / OpenAI which lack the concept).
- `crates/entelix-core/tests/codec_consistency_matrix.rs` — matrix entry per (codec, ContentPart variant, provider_echoes preservation) tuple.
- New `crates/entelix-core/tests/provider_echo_round_trip.rs`:
  - Anthropic Messages: encode → decode → re-encode preserves `signature` on `Thinking`.
  - Anthropic Messages: same for `RedactedThinking`.
  - Bedrock Converse: signature + redactedContent round-trip (Anthropic-on-Bedrock and Nova-on-Bedrock test fixtures both pass under the single `bedrock-converse` provider key).
  - Vertex Anthropic: signature round-trip.
  - Gemini: `thought_signature` round-trip on `Text`, `ToolUse`, `Thinking` parts; assert wire field is **snake_case** on encode.
  - Vertex Gemini: same as Gemini.
  - OpenAI Responses: `encrypted_content` part-level round-trip; `previous_response_id` request → response chain.
  - Cross-vendor isolation: a `Thinking` part carrying both `anthropic-messages` and `gemini` entries — encoded through Anthropic codec emits only `signature`; symmetric for Gemini; original Gemini entry survives in IR.
- `crates/entelix-core/tests/streaming_thinking.rs` — assertions migrated from `signature.as_deref()` to `provider_echoes` lookup.
- `cargo xtask invariants` clean (no-shims, silent-fallback, naming, managed-shape, surface-hygiene).

### Phase 8 — Live validation

- `crates/entelix-cloud/tests/live_vertex_gemini_tool_round_trip.rs` re-run against `gemini-3.1-pro-preview` — multi-turn tool dispatch survives. **This unblocks the ontosyx production blocker.**
- `crates/entelix-core/tests/live_anthropic.rs` (manual run) — extended-thinking signature round-trips one extra turn without regression.
- `crates/entelix-cloud/tests/live_bedrock.rs` (manual run) — Anthropic-on-Bedrock thinking round-trip; Nova 2 thinking round-trip if credentials available.
- `crates/entelix-core/tests/live_openai_responses.rs` (manual run) — `encrypted_content` stateless chain (`store: false`) survives one extra turn.

## Verification gates (squash merge prerequisite)

- [ ] `cargo build --workspace --all-features` green
- [ ] `cargo clippy --workspace --all-features --all-targets -- -D warnings` green
- [ ] `cargo test --workspace --all-features` green
- [ ] `cargo xtask invariants` green
- [ ] `crates/entelix-core/tests/provider_echo_round_trip.rs` covers every codec wire-shape — Anthropic Messages (signature + redacted_thinking), Bedrock Converse (single codec hosting Anthropic + Nova 2 under one shape; one fixture exercises both), Vertex Anthropic (composition delegate), Gemini (snake_case wire, Text + ToolUse + Thinking parts, legacy camelCase decode tolerance), Vertex Gemini (composition delegate), OpenAI Responses (3-tier — part `encrypted_content` + response `Response.id` + request `previous_response_id`), OpenAI Chat (foreign-echo silent drop) — plus cross-vendor isolation in both directions and IR-preserves-foreign-blob through decode
- [ ] `live_vertex_gemini_tool_round_trip.rs` passes against a Gemini 3.x reasoning model
- [ ] `live_anthropic.rs` passes (where credentials available)
- [ ] `live_bedrock.rs` passes (where credentials available)
- [ ] `live_openai_responses.rs` passes (where credentials available)
- [ ] `grep` audit: `Thinking { … signature: …` returns 0 hits in the workspace
- [ ] `grep` audit: `signature: Option<String>` returns 0 hits inside `crates/entelix-core/src/ir/`, `crates/entelix-core/src/stream.rs`, `crates/entelix-session/src/event.rs`
- [ ] `grep` audit: `thoughtSignature` (camelCase) returns 0 hits in `crates/entelix-core/src/codecs/gemini.rs` and `vertex_gemini.rs`

## Why this design ages well

- **Schema-stable IR.** Vendors continue to invent multi-turn-context tokens; the IR doesn't grow with each one. New vendor X = one new codec impl, zero IR / consumer changes.
- **Codec autonomy.** Each codec owns its vendor's wire shape end-to-end; cross-codec passthrough of foreign entries is a structural property, not a runtime branch.
- **Auditability.** `provider_echoes` is `Serialize`, so session logs preserve the same opaque tokens for replay. Resume-from-checkpoint hands the model exactly the bytes it issued.
- **Tampering safety preserved.** Anthropic's signature semantics (server-side hash validation) survive verbatim; the IR just stops naming the field.
- **Three-tier coverage.** Part / response / request carriers handle every vendor surface seen in 2026-05; promoting any future vendor's response-id or request-chain pointer is a codec-internal change.
- **No silent fallback.** Codecs that lack a vendor's concept emit `LossyEncode` (invariant 15) instead of fabricating defaults. Codec autonomy guarantees foreign blobs never get mistakenly emitted on the wrong wire.
