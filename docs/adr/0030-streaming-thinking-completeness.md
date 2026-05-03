# ADR-0030 — Streaming thinking completeness

* **Status**: Accepted
* **Date**: 2026-04-27
* **Drivers**: Phase 9A
* **Refines**: ADR-0026 (IR modernization — added `ContentPart::Thinking`)

## Context

Phase 8A added `ContentPart::Thinking { text, signature }` to the IR
and routed every codec's *non-streaming* `decode` path through it.
The streaming path was never updated:

- `StreamDelta` had no variant for thinking deltas, so codecs that
  detect a thinking block on the wire have no type-level place to
  emit it. Anthropic SSE `content_block_start type="thinking"` falls
  through the `BlockKind` enum (`Text` / `ToolUse` only); Gemini
  `part.thought == true` is ignored in the stream parser; OpenAI
  Chat / Responses' reasoning deltas are also dropped silently.
- `Anthropic decode_usage` hard-codes `reasoning_tokens: 0` even
  when the vendor reports thinking-token accounting on
  `message_start`.
- `Bedrock Converse decode_output` never handles
  `reasoningContent` blocks — encode emits them, decode silently
  drops, so a Claude-on-Bedrock thinking response cannot
  round-trip.

The non-streaming path *does* honour `Thinking`; the asymmetry
breaks the LLM-facing token discipline (the streaming consumer sees
a different shape of conversation than the non-streaming consumer).

## Decision

### `StreamDelta::ThinkingDelta`

```rust
pub enum StreamDelta {
    …
    /// Append text to the in-progress thinking block. Consecutive
    /// `ThinkingDelta`s fold into a single `ContentPart::Thinking`
    /// in the output.
    ThinkingDelta {
        text: String,
        /// Vendor signature for redaction-resistant replay
        /// (Anthropic supplies a discrete `signature_delta` event
        /// or a single `signature` on the block-start event).
        signature: Option<String>,
    },
    …
}
```

`StreamAggregator::push` opens a fresh `Thinking` block on the
first `ThinkingDelta` and folds subsequent ones in (mirroring how
`TextDelta` builds `Text`). When a `ToolUseStart` or `TextDelta`
arrives mid-thinking, the open thinking block closes — preserving
intra-turn order.

### Codec stream path coverage

| Codec | Wire signal for thinking | Action |
|---|---|---|
| Anthropic | `content_block_start type: "thinking"`, `delta.type: "thinking_delta"`, `delta.type: "signature_delta"` | New `BlockKind::Thinking`; emit `ThinkingDelta { text, .. }` on `thinking_delta`; emit `ThinkingDelta { text: "", signature: Some(..) }` on `signature_delta` |
| OpenAI Chat | (no streaming reasoning delta on this API as of 2026-01) | Document as "no native streaming thinking" — non-streaming responses still decode `reasoning_content` if present |
| OpenAI Responses | `response.reasoning.delta` / `response.reasoning_summary.delta` | Map to `ThinkingDelta` |
| Gemini | `parts[].thought == true` with non-empty `text` (regular streaming-content payload), `parts[].thoughtSignature` | Branch on `thought` *before* the text/functionCall branches |
| Bedrock Converse (decode-side) | `reasoningContent.reasoningText` block in the response | Decode into `ContentPart::Thinking` (closes the encode/decode asymmetry) |

The Bedrock Converse non-streaming decode path is updated even
though Bedrock's stream framing lives in `entelix-cloud` (per
`bedrock_converse.rs` module doc) — `decode_output` is shared by
the buffered fallback, and a real-time Bedrock SSE wiring (Phase
1.x companion) inherits the corrected non-streaming logic.

### Anthropic `reasoning_tokens` extraction

`message_start.usage` and `message_delta.usage` may carry thinking
accounting (vendor-published as
`output_tokens` partitioned with a thinking subtotal — Anthropic
exposes both raw `output_tokens` and per-content-block accounting
in the response usage block). The current decoder reads only
`input_tokens` / `output_tokens` / cache fields; it now also reads
the thinking-token field and routes it to `Usage::reasoning_tokens`.

### Round-trip assertion

A new test in `entelix-core/tests/streaming_thinking.rs` exercises
each codec's stream path with a synthetic SSE/JSON corpus
containing a thinking block, asserting that the
`StreamAggregator::finalize` output carries
`ContentPart::Thinking { text, signature }` — closing the round-
trip gap that currently exists only on non-streaming paths.

## Consequences

- `entelix-core::stream::StreamDelta` gains one variant. The enum
  is `#[non_exhaustive]`-equivalent in practice (every consumer
  matches the enum directly; Phase 9A updates them atomically —
  no compat shim).
- `StreamAggregator::push` gains a `ThinkingDelta` arm. The
  pending-block scratch space tracks "open text", "open tool", or
  "open thinking" — at most one open at a time.
- Public-api baseline refrozen for `entelix-core` (variant
  addition).

## Alternatives considered

1. **Reuse `TextDelta` for thinking** — rejected. Stream consumers
   can't tell thinking from regular text without a discriminator;
   defeats the IR distinction added in Phase 8A.
2. **Separate `ThinkingStream` parallel to `StreamDelta`** —
   rejected. The single-stream model is the SSE consumer's natural
   shape (interleaved order).
3. **Skip Anthropic `signature_delta` events** — rejected. The
   signature is required for redaction-resistant replay on
   follow-up turns; dropping it on streaming would force callers
   to use non-streaming for any conversation that re-presents
   thinking.

## References

- ADR-0026 §"`Thinking` reasoning blocks"
- Anthropic Messages streaming reference: `content_block_start`,
  `content_block_delta`, `signature_delta` event kinds
- Gemini streaming `parts[].thought` shape
