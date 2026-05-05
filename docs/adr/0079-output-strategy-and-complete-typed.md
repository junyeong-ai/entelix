# ADR 0079 — `OutputStrategy` enum + `ChatModel::complete_typed<O>`

**Status**: Accepted
**Date**: 2026-05-05
**Decision**: `OutputStrategy { Auto, Native, Tool, Prompted }` is the cross-vendor IR enum that selects the dispatch shape for structured output. `Auto` resolves at codec-construction time per `Codec::auto_output_strategy(model)`; per-request resolution would let one logical request resolve to different shapes across replays, breaking the SessionGraph event log's deterministic-replay guarantee. `ChatModel::complete_typed<O: JsonSchema + DeserializeOwned + Send + 'static>` derives the schema, builds the `ResponseFormat`, dispatches through the existing `tower::Service` spine, and parses the typed `O` from either a forced-tool `tool_use` block (Anthropic / Bedrock-Anthropic default) or a text content part (OpenAI / Gemini default). `Prompted` is shipped as a typed `OutputStrategy` variant but rejected at encode time with `Error::invalid_request` — it lands in 1.1.

## Context

Industry consensus (Fork 3 audit synthesis on `project_entelix_2026_05_05_master_plan.md`) converges on three dispatch shapes for structured output: vendor-native channel (OpenAI `text.format = json_schema`, Gemini `responseJsonSchema`, Anthropic `output_config.format`), forced single tool call (every vendor's tool-call surface predates native structured output and accepts arbitrary JSON schemas), and prompted (schema injected into the system prompt + best-effort parse). Mature SDKs (LangChain 1.0 `ProviderStrategy`/`ToolStrategy`, pydantic-ai 1.90 `NativeOutput`/`ToolOutput`/`PromptedOutput`/`TextOutput`, BAML SAP, Vercel AI SDK 5 `generateObject`, Instructor's mode flag) all expose a runtime knob plus an automatic picker.

Up to 1.0-RC.1 entelix exposed `ResponseFormat { json_schema: JsonSchemaSpec, strict: bool }` with a single dispatch shape per codec — Anthropic emitted `output_config` (the newer surface, no strict toggle, less mature than tool calls); OpenAI Chat / OpenAI Responses / Gemini emitted their respective native shapes; Bedrock Anthropic-passthrough emitted Anthropic's. Operators routing the same `ResponseFormat` across vendors got vendor-shape-specific behaviour (Anthropic structurally less precise than OpenAI's strict-mode validation); operators wanting forced-tool dispatch had to build the `tools` and `tool_choice` themselves.

Per Fork 4's vendor wire-format research:

- **Anthropic** native `output_config.format` ships without a `strict` toggle and is newer than the tool-call surface. The forced-tool surface is more mature, parity across Anthropic versions, and accepts arbitrary JSON schemas without strict-mode constraints.
- **OpenAI Responses** `text.format = { type: "json_schema", strict: true, schema }` is industry baseline.
- **OpenAI Chat Completions** `response_format = { type: "json_schema", json_schema: { ..., strict: true } }` is the established mature surface.
- **Gemini 2.5+** `generationConfig.responseJsonSchema` always strict-validates.
- **Bedrock Converse** for Anthropic family routes Anthropic's shapes through `additionalModelRequestFields` passthrough; for non-Anthropic Bedrock models (Nova, Mistral, Llama on Converse) there is no canonical structured-output channel today.

## Decision

### `OutputStrategy` enum

`entelix-core/src/ir/structured.rs`:

```rust
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum OutputStrategy {
    #[default]
    Auto,
    Native,
    Tool,
    Prompted,
}
```

Four variants — `Auto / Native / Tool / Prompted` — matching the industry vocabulary. `#[non_exhaustive]` keeps room for post-1.0 variants (e.g. a `Cohere` literal-mode if a vendor surfaces something genuinely new).

`ResponseFormat::strategy: OutputStrategy` is the new field; default is `Auto`. `ResponseFormat::with_strategy(OutputStrategy::Tool)` is the builder override.

### `Codec::auto_output_strategy(model: &str) -> OutputStrategy`

The trait method codec authors override to pick their preferred dispatch shape per model. Default returns `Native` — most codecs ship vendor-native structured output as their most mature surface. Codecs whose native channel is newer / less mature override to `Tool`:

- `AnthropicMessagesCodec::auto_output_strategy` returns `Tool` (native `output_config` ships without strict toggle).
- `BedrockConverseCodec::auto_output_strategy` returns `Tool` for `anthropic.claude-…` models (parity with direct Anthropic) and `Native` otherwise.
- `OpenAiChatCodec`, `OpenAiResponsesCodec`, `GeminiCodec` use the default `Native`.

The trait method is reachable for callers who want to know which strategy `Auto` will resolve to before building the request (e.g. `OtelLayer` could surface `entelix.output_strategy = "tool"` on the request span).

### `Auto` resolves at codec-construction time, not per-request

A per-request resolution path — "this call's `Auto` re-resolves on each replay through whatever codec is wired" — would break ADR-0001 §"events SSoT" (the SessionGraph event log carries `GraphEvent::ToolResult` content for replay; if `Auto` resolved to `Native` on the first call but `Tool` on a replay, the model's reply would land in different content parts and downstream `parse_typed_response` would extract from the wrong slot). Resolution happens once, when the codec encodes the request — same call site every retry, same call site every replay-from-checkpoint.

### Per-codec encode helpers

Each codec carries its own `encode_*_structured_output` private helper that resolves `Auto` against the codec's preference, then emits either the native shape or the forced-tool shape. The Anthropic forced-tool shape is one tool prepended to the operator's `tools` array with `tool_choice: { type: "tool", name, disable_parallel_tool_use: true }`. OpenAI Chat / OpenAI Responses match the same shape with their respective wire formats. Gemini wraps in `tools[0].functionDeclarations` with `toolConfig.functionCallingConfig.mode = "ANY"` + `allowedFunctionNames: [name]`. Bedrock-Anthropic routes through `additionalModelRequestFields` to avoid colliding with Bedrock's own `toolConfig` top-level surface.

`Prompted` is universally rejected at encode time with `Error::invalid_request("OutputStrategy::Prompted is deferred to entelix 1.1; …")`. The 1.1 implementation lands as a `PromptedOutputParser` wrapping the dispatch — operator-supplied schema gets injected into the system prompt, and a post-hoc parser extracts the JSON. Production use cases for `Prompted` are narrow (vendors lacking both Native and Tool support — none today), so the deferral does not block 1.0.

### `ChatModel::complete_typed<O>`

```rust
pub async fn complete_typed<O>(
    &self,
    messages: Vec<Message>,
    ctx: &ExecutionContext,
) -> Result<O>
where
    O: schemars::JsonSchema + serde::de::DeserializeOwned + Send + 'static,
```

The method derives the JSON Schema for `O` via `schemars::schema_for!`, builds a strict `ResponseFormat` (defaulting `OutputStrategy::Auto`), attaches it to the request, dispatches through the same `tower::Service` spine `complete_full` uses, and parses the response.

`parse_typed_response<O>` tries the forced-tool shape first (extracts `ContentPart::ToolUse::input` and parses as `O`) then falls through to the native shape (extracts `ContentPart::Text::text` and parses as `O`). The order matches reality: codecs that prefer `Tool` (Anthropic, Bedrock-Anthropic) emit `ToolUse` blocks; codecs that prefer `Native` (OpenAI, Gemini) emit `Text` blocks containing the JSON document; the helper handles both transparently.

`O: JsonSchema + DeserializeOwned + Send + 'static`. `Sync` is intentionally not required (the trait bound on `complete_full`'s callers does not require `Sync` either). `Serialize` is not required — `O` is the *output* type, never serialised back to the model.

The schema name embedded in the request (`JsonSchemaSpec::name`) is the rightmost path segment of `std::any::type_name::<O>()` — `entelix_core::ir::request::ModelRequest` becomes `ModelRequest`. Vendors that surface the name in observability ship a readable string; the legacy form was the full module path which broke the OpenAI 64-character `name` length limit on long crate names.

### `JsonSchema` adds `schemars` to `entelix-core`

`schemars` was previously only a dependency of `entelix-tools` (for `SchemaTool`-derived input schemas). `complete_typed<O>` requires it in `entelix-core` so the `ChatModel` surface can derive at call time. `schemars` is feature-flag-free and pulls minimal transitive dependencies (`dyn-clone`, `serde_json` — already in the tree); no clippy-pedantic regression.

## Consequences

**Positive**:

- Operators write `let result: User = chat_model.complete_typed(messages, &ctx).await?;` and the SDK picks the right dispatch shape per vendor without operator branching.
- Anthropic users get the forced-tool surface by default — more mature than `output_config`, no `LossyEncode` warnings on `strict=true` (Anthropic native ships without the toggle).
- OpenAI / Gemini users get the native surface by default — strictest validation, fewest tokens spent on tool-call ceremony.
- Operators that need the override path build `ResponseFormat::strict(spec).with_strategy(OutputStrategy::Tool)` and route through `complete_full` — the per-codec encode helpers honour the explicit choice.
- `Prompted` is shipped as a typed enum variant from day one (no breaking change when 1.1 implements it), but encode-time rejection means operators see the gap clearly instead of a silent wire failure.
- `Auto`'s codec-construction-time resolution preserves replay determinism — the SessionGraph event log's `GraphEvent::ToolResult` content lands in the same slot on every replay.

**Negative**:

- Each codec's structured-output encode site grew from a few lines (single-shape inline emit) to a 50–80-line helper with a `match strategy` block. The verbosity pays for itself in per-strategy LossyEncode precision (operators see "Anthropic Tool-strategy structured output is always schema-validated; strict=false was approximated", not a generic message).
- `entelix-core` gains `schemars` as a transitive dependency. The cost is one extra crate in `cargo metadata` resolution; the benefit is `complete_typed<O>` without a feature flag.
- One additional public type (`OutputStrategy`), one new field on `ResponseFormat` (`strategy`), one new method on `ChatModel` (`complete_typed`). Per ADR-0064 §"Public-API baseline contract" the typed-strengthening drift is authorised at the 1.0 RC contract boundary.

**Migration outcome (one-shot, no shim)**: Existing `ResponseFormat::strict(spec)` and `ResponseFormat::best_effort(spec)` constructors continue to work — they default `strategy` to `Auto`. Codec encode sites now route through the per-codec helpers; the previous unconditional-emit blocks are deleted, not deprecated. There is no `#[deprecated]`, no `pub use`, no shim — invariant 14 forbids them.

## References

- CLAUDE.md invariant 5 — provider IR before wire format (ResponseFormat is the IR, codecs translate)
- CLAUDE.md invariant 6 — lossy encoding emits warnings (per-strategy LossyEncode emit sites)
- CLAUDE.md invariant 14 — no backwards-compatibility shims
- ADR-0001 — events SSoT (Auto's codec-construction-time resolution preserves replay determinism)
- ADR-0024 — structured output IR (`ResponseFormat` introduction; this ADR extends with strategy)
- ADR-0031 — vendor structured-output channels (per-codec wire shape catalogue)
- ADR-0064 — 1.0 release charter (post-RC typed-strengthening authorised)
- ADR-0078 — `ReasoningEffort` cross-vendor IR (parallel pattern: cross-vendor IR field with per-codec wire mapping + LossyEncode emit)
- Master plan May-05 §"B-1 + B-2" — Fork 3 industry survey + Fork 4 vendor wire-format research
- `crates/entelix-core/src/ir/structured.rs` — `OutputStrategy` enum + `ResponseFormat::strategy` field
- `crates/entelix-core/src/codecs/codec.rs` — `Codec::auto_output_strategy` trait method
- `crates/entelix-core/src/codecs/anthropic.rs` `encode_anthropic_structured_output` — Tool-default mapping
- `crates/entelix-core/src/codecs/openai_chat.rs` `encode_openai_chat_structured_output` — Native-default mapping
- `crates/entelix-core/src/codecs/openai_responses.rs` `encode_openai_responses_structured_output` — Native-default mapping
- `crates/entelix-core/src/codecs/gemini.rs` `encode_gemini_structured_output` — Native-default mapping
- `crates/entelix-core/src/codecs/bedrock_converse.rs` `encode_bedrock_structured_output` — Anthropic-family Tool / non-Anthropic LossyEncode
- `crates/entelix-core/src/chat.rs` `ChatModel::complete_typed` + `parse_typed_response` — typed-output dispatch
