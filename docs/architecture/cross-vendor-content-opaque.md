# Cross-vendor `ContentPart` opaque round-trip carrier

## Problem

Vendors emit per-part **opaque round-trip tokens** that the harness must echo verbatim on the next turn for the model to recover its prior state. Each vendor names and shapes the token differently, but they all share the same contract: *opaque to the harness, mandatory on the wire, lost = next turn rejected or degraded*.

| Vendor | Token | Attached to | Echo contract |
|---|---|---|---|
| Anthropic Messages | `signature` | `thinking` block | tampering-resistant replay; missing on echo → block silently dropped |
| Gemini (AI Studio + Vertex) 3.x | `thoughtSignature` | any reasoning-bearing part (`functionCall`, `text`, `thinking`) | thinking-context preservation; missing on echo → `INVALID_ARGUMENT` |
| OpenAI Responses | `previous_response_id` | response root, conversation continuity | next request references prior id; missing on echo → fresh conversation |

Every additional vendor (Mistral, Cohere, …) is expected to follow the same pattern with its own name. The IR currently encodes one of these (`Thinking::signature`) as a *vendor-named field*, which both:

1. Forces the IR to grow a new field every time a vendor adopts the pattern.
2. Couples downstream consumers (codecs that don't know what `signature` means in this position) to a vendor-specific concept they should ignore.

The Gemini 3.x rollout exposed the cost in production: multi-turn tool dispatches against any reasoning-tier Gemini model fail with `function call ... is missing a thought_signature` because the IR had nowhere to store the token between turns.

## Design

Replace the per-vendor field with one **vendor-keyed opaque carrier** common to every `ContentPart` variant:

```rust
/// Vendor-keyed opaque round-trip token. Codecs decode their own
/// vendor's blob into this carrier and read it back verbatim on
/// the encode side. The harness never inspects the payload — it
/// forwards whichever blob lands at decode time, untouched.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct OpaqueProviderState {
    /// Provider id matching `Codec::name` (`"anthropic-messages"`,
    /// `"gemini"`, `"openai-responses"`, …). Codecs select their
    /// own blobs by this key.
    pub provider: Cow<'static, str>,
    /// Vendor-defined opaque payload. Typically a JSON object
    /// shaped only by the codec that produced it.
    pub payload: serde_json::Value,
}
```

Every `ContentPart` variant carries:

```rust
/// Vendor-keyed round-trip tokens this part must echo back on
/// the next turn (Gemini 3.x `thoughtSignature`, Anthropic
/// `signature`, …). Defaults to empty. Codecs only read /
/// write entries matching their own `Codec::name`.
#[serde(default, skip_serializing_if = "Vec::is_empty")]
pub provider_state: Vec<OpaqueProviderState>,
```

### Properties

- **IR is vendor-neutral.** New vendor X with its own opaque token requires *zero IR changes* — the codec serialises into / out of `provider_state` with `provider: "x"`.
- **Anthropic `Thinking::signature` is removed.** The Anthropic Messages codec round-trips the same value through `provider_state` with `provider: "anthropic-messages"`. Existing IR-level vendor coupling deleted in the same PR (no migration shim, no deprecated alias).
- **Codec isolation.** A codec that doesn't recognise a blob (different vendor, no match) leaves it untouched on the IR but never emits it on the wire. The blob survives one round-trip then gets garbage collected when the conversation moves past that part.
- **Default-deny exposure.** `provider_state` is `Serialize + Deserialize` for IR persistence (session log, checkpointer) but never reaches the model — codecs serialise it onto the wire, never into the visible content channel.

### Wire mapping examples

**Anthropic `signature` round-trip** (assistant `thinking` block):

```json
// Wire (from API):
{"type": "thinking", "thinking": "let me reason…", "signature": "abcd…"}

// Decoded ContentPart:
ContentPart::Thinking {
    text: "let me reason…",
    cache_control: None,
    provider_state: vec![OpaqueProviderState {
        provider: "anthropic-messages".into(),
        payload: json!({ "signature": "abcd…" }),
    }],
}

// Re-encoded on next turn (Anthropic codec):
{"type": "thinking", "thinking": "let me reason…", "signature": "abcd…"}
```

**Gemini `thoughtSignature` round-trip** (assistant `functionCall` part):

```json
// Wire (from API):
{"functionCall": {"name": "get_weather", "args": {…}}, "thoughtSignature": "EhsM…"}

// Decoded ContentPart:
ContentPart::ToolUse {
    id: "<derived>",
    name: "get_weather",
    input: json!({…}),
    provider_state: vec![OpaqueProviderState {
        provider: "gemini".into(),
        payload: json!({ "thought_signature": "EhsM…" }),
    }],
}

// Re-encoded on next turn (Gemini codec):
{"functionCall": {"name": "get_weather", "args": {…}}, "thoughtSignature": "EhsM…"}
```

**Cross-vendor harmlessness** — a transcript containing parts with both Anthropic and Gemini `provider_state` entries can be sent to either codec; each codec only emits its own entries on the wire and silently leaves the other vendor's blob alone in IR.

## Out of scope (explicit)

- `ContentPart` re-shaping into `struct ContentPart { kind: ContentKind, … }` — the enum-of-structs surface stays. The carrier rides on each variant.
- Compatibility shims for `Thinking::signature` — removed in the same PR. No deprecated alias, no migration helper. Persisted sessions older than the cutover are not preserved (entelix is pre-1.0).

## Phased work plan

Each phase is a self-contained green-build commit on `feat/cross-vendor-content-opaque`. The branch lands on `main` as a single squash commit when phase 7 passes.

### Phase 1 — `OpaqueProviderState` + IR refactor
- New module `crates/entelix-core/src/ir/provider_state.rs` defining `OpaqueProviderState` with helper accessors (`new`, `for_provider`, `payload_field`).
- `crates/entelix-core/src/ir/content.rs`:
  - 11 `ContentPart` variants gain `provider_state: Vec<OpaqueProviderState>` with `#[serde(default, skip_serializing_if = "Vec::is_empty")]`.
  - `Thinking::signature` field deleted.
  - Helper constructors (`text`, `image`, `audio`, `video`, `document`, `thinking`, `citation`, `tool_use`, `image_output`, `audio_output`, `tool_result`) default `provider_state: Vec::new()`. Add a builder method `with_provider_state(self, OpaqueProviderState) -> Self` for codec construction.
- `ir/mod.rs` re-exports `OpaqueProviderState`.

### Phase 2 — Mechanical fix on every consumer
- Pattern matches: `cargo build --workspace --all-features` lists every site. Each `ContentPart::Foo { … }` pattern adds `provider_state: _` (or `..`) where the value isn't read; constructors add `provider_state: Vec::new()`.
- Crates touched (recursive): `entelix-core`, `entelix-tools`, `entelix-agents`, `entelix-session`, `entelix-mcp`, `entelix-graph`, `entelix-runnable`, `entelix-policy`, `entelix-otel`, `entelix-prompt`, `entelix-cloud`, `entelix-server`, `entelix-persistence`, `entelix-memory`, `entelix-auth-claude-code`.

### Phase 3 — Anthropic codec migration
- `crates/entelix-core/src/codecs/anthropic.rs` decoder: wire `signature` on a `thinking` block lands in `provider_state` with `provider: "anthropic-messages"`, payload `{ "signature": "<value>" }`.
- Encoder: when emitting an assistant `thinking` block, look up the same provider entry and write it back to the wire `signature` field. Missing entry → no `signature` on the wire (matches current behaviour for fresh thinking blocks).
- Anthropic codec unit tests update — `Thinking::signature` references rewritten.

### Phase 4 — Gemini codec implementation
- `crates/entelix-core/src/codecs/gemini.rs`:
  - Decode every `parts[]` entry that carries `thoughtSignature` (regardless of which inner shape — `text`, `thinking`, `functionCall`) and attach a `provider_state` entry with `provider: "gemini"`, payload `{ "thought_signature": "<value>" }`.
  - Encode assistant turns: when re-emitting any part, look up the `gemini` entry and write `thoughtSignature` onto the same wire object.
- `crates/entelix-core/src/codecs/vertex_gemini.rs` inherits the behaviour through composition (already wraps `GeminiCodec`).

### Phase 5 — Other codecs (silent passthrough)
- OpenAI Chat / OpenAI Responses / Bedrock Converse / VertexAnthropic: no encoder / decoder changes — unrelated `provider_state` entries flow through IR untouched and never reach the wire.
- Add to `tests/codec_isolation.rs` (new): a transcript carrying entries for `provider: "x"` round-trips through every codec without affecting wire bytes.

### Phase 6 — Test sweep
- `tests/codec_robustness_proptest.rs` regenerates with new `ContentPart` shape.
- `tests/lossy_warning_completeness.rs` audited — `Thinking::signature` row removed.
- `tests/compat_matrix.rs` audited.
- New `crates/entelix-core/tests/provider_state_round_trip.rs`:
  - Anthropic Messages: encode → decode → re-encode preserves `signature`.
  - Gemini: same for `thoughtSignature` on `text`, `thinking`, `functionCall` parts.
  - Cross-vendor: a `Thinking` part carrying both `anthropic-messages` and `gemini` entries, encoded through Anthropic codec, emits only the `signature` (Gemini entry survives in IR but never on wire); the symmetric case for Gemini codec.
- `cargo xtask invariants` clean.

### Phase 7 — Live validation
- `crates/entelix-cloud/tests/live_vertex_gemini_tool_round_trip.rs` re-runs against `gemini-3.1-pro-preview` and passes — multi-turn tool dispatch survives.
- `crates/entelix-core/tests/live_anthropic.rs` (manual run with `ANTHROPIC_API_KEY`) — extended-thinking signature round-trips one extra turn without the assertion regressing.

## Verification gates (squash merge prerequisite)

- [ ] `cargo build --workspace --all-features` green
- [ ] `cargo clippy --workspace --all-features --all-targets -- -D warnings` green
- [ ] `cargo test --workspace --all-features` green
- [ ] `cargo xtask invariants` green
- [ ] `crates/entelix-core/tests/provider_state_round_trip.rs` covers Anthropic + Gemini + cross-vendor isolation
- [ ] `live_vertex_gemini_tool_round_trip.rs` passes against a Gemini 3.x reasoning model
- [ ] `live_anthropic.rs` passes (where credentials available)
- [ ] `grep` audit: `Thinking { … signature: …` returns 0 hits in the workspace (the field is fully removed)

## Why this design ages well

- **Schema-stable IR** — vendors continue to invent multi-turn-context tokens (Gemini 3.x, OpenAI Responses prior-id, anything next year). The IR doesn't grow with each one.
- **Codec autonomy** — adding a new vendor is one new codec impl; nothing else in the workspace changes.
- **Auditability** — `provider_state` is `Serialize`, so session logs (`SessionLog::append(GraphEvent::*)`) preserve the same opaque tokens for replay. Resume-from-checkpoint after server restart hands the model exactly the bytes it issued.
- **Tampering safety preserved** — Anthropic's signature semantics (tampering detection) survive verbatim; the IR just stops naming the field.
