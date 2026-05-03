# ADR 0036 — IR provider flagship features

**Status**: Accepted
**Date**: 2026-05-01
**Decision**: Phase 5 of the post-7-차원-audit roadmap

## Context

The 7-차원 provider/IR audit (D2 + F7 + O3 + O4) listed four
vendor-flagship surfaces the IR could not represent without
operators bypassing the typed channel:

- **D2** — Anthropic *extended thinking*. Round-tripping `Thinking`
  content blocks worked, but there was no way to *enable* the
  feature on the request side.
- **F7** — OpenAI *reasoning effort* + summary on the Responses API.
  Same shape: o-series flagship knob with no IR slot.
- **O3** — `ToolKind` only carried `Function`, `WebSearch`,
  `Computer`. Anthropic's `text_editor`, `bash`, `code_execution`,
  `mcp` connector, `memory` and OpenAI Responses' `file_search`,
  `code_interpreter`, `image_generation` had no representation.
- **O4** — Multimodal output. The IR carried `Image`, `Audio`,
  `Video`, `Document` as inputs; assistant-produced media
  (gpt-image-1, OpenAI tts, Gemini image generation) had no slot.
- **F1** (Bedrock subset) — `extract_rate_limit` was unimplemented
  for Bedrock; `x-amzn-bedrock-*` headers reached
  `RateLimitSnapshot` as `None`.

The pattern across all four was the same: an IR that should have
been the SSoT was leaving a flagship feature behind, forcing
operators down the typed-extension escape hatch.

## Decision

Land four IR additions plus the Bedrock rate-limit gap-fill in one
slice. All variants are `#[non_exhaustive]`; all new field types
are public so downstream operators depend on them at the type
system level instead of stringly-typed JSON.

### `ThinkingConfig` (Anthropic)

`AnthropicExt::thinking: Option<ThinkingConfig>` carries
`{budget_tokens: u32}`. `AnthropicMessagesCodec` emits the
documented `thinking: {type:"enabled", budget_tokens: N}` request
field. `BedrockConverseCodec` rides it through
`additionalModelRequestFields.thinking` for Anthropic-on-Bedrock
deployments and emits field-precise `LossyEncode` warnings for the
two Anthropic-only ext fields it cannot honour
(`disable_parallel_tool_use`, `user_id`).

### `ReasoningConfig` (OpenAI Responses)

`OpenAiResponsesExt::reasoning: Option<ReasoningConfig>` carries
`{effort, summary?}`. `ReasoningEffort` covers
`Minimal`/`Low`/`Medium`/`High`; `ReasoningSummary` covers
`Auto`/`Concise`/`Detailed`. The codec emits the snake-case wire
form OpenAI documents.

### `ToolKind` expansion

Eight new variants:

- Anthropic-native: `TextEditor`, `Bash`, `CodeExecution`,
  `McpConnector { name, server_url, authorization_token }`,
  `Memory`.
- OpenAI Responses-native: `FileSearch { vector_store_ids }`,
  `CodeInterpreter`, `ImageGeneration`.

Each codec fans out: vendor-native variants encode to the
documented wire shape, foreign variants emit
`ModelWarning::LossyEncode { field: "tools[N]", detail }` with a
diagnostic that tells the operator which codec ships the feature.
The `tests/lossy_warning_completeness.rs` matrix grew nine rows
(one per new variant) so every codec × variant combination is
asserted as either native or `LossyEncode`.

### Multimodal output `ContentPart` variants

`ContentPart::ImageOutput { source: MediaSource }` and
`ContentPart::AudioOutput { source: MediaSource, transcript:
Option<String> }`. Output blocks carry no `cache_control` (the
model produces them fresh per turn). Codecs that receive these on
encode (multi-turn replay where assistant output is folded back
into history) drop the block with `LossyEncode` — no vendor input
shape consumes assistant-produced media. Decode-side mapping for
gpt-image-1, OpenAI tts, and Gemini inline image parts lands in a
follow-up slice; the IR slots are introduced now so the decode
work is non-breaking.

### Bedrock `extract_rate_limit`

Captures every `x-amzn-bedrock-*` header into
`RateLimitSnapshot::raw` plus `Retry-After` when the gateway sets
it. AWS does not standardise typed quota headers today; the raw
map keeps cost dashboards honest without inventing structure the
vendor has not committed to.

## Consequences

✅ Anthropic extended thinking, OpenAI reasoning effort, and
vendor-side built-ins are first-class in the IR. Operators stop
shipping raw JSON through ext escape hatches.
✅ Multimodal output round-trips cleanly through the IR — once
codec decoders adopt the new variants in the follow-up slice the
agent layer sees structured assistant media instead of opaque
`Text` blocks.
✅ Bedrock cost / latency telemetry now flows into
`RateLimitSnapshot::raw`, unblocking dashboards.
✅ `lossy_warning_completeness.rs` matrix is exhaustive for every
shipping `ToolKind` variant.
❌ `ContentPart` and `ToolKind` `match` arms in operator code now
need to handle the new variants (mitigated by `#[non_exhaustive]`
which forces a wildcard).
❌ Codecs grew `encode_tools` warnings parameter (Anthropic /
OpenAI Responses / Bedrock) for the `LossyEncode` path; downstream
codec implementors copying the pattern need to follow.

## Alternatives considered

1. **Leave flagship features in `provider_extensions` as raw
   `Value`**. Invariant 5 would silently weaken — every `Value`
   patch is one place the IR stops being the SSoT. Rejected.
2. **Expose `ToolKind::VendorRaw { vendor, payload }` instead of
   typed variants**. Lower ceremony but defeats schema validation
   and surfaces vendor naming in operator code. Rejected.
3. **Defer multimodal output to a later slice**. Keeping round-trip
   lossy meant operator code parsing `Text { text: "[image:
   data:image/png;base64,...]" }`. The typed slot costs nothing to
   ship now and unblocks decoders.

## References

- 7-차원 audit fork report `audit-provider-ir` D2 / F7 / O3 / O4 / F1.
- ADR-0026 — IR modernisation (parent of this slice).
- ADR-0031 — `CacheControl` IR expansion (shape precedent for
  per-vendor ext config types).
- ADR-0033 — invariant #16 (`LlmFacingSchema::strip` keeps the
  expanded tool-spec wire small even when several built-ins are
  advertised at once).
