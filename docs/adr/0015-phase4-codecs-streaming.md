# ADR 0015 — Phase 4 codec subset + streaming trait surface

**Status**: Accepted
**Date**: 2026-04-26
**Closes**: Phase 4 codec subset (5 codecs) + token-level streaming
plumbing through `Codec::decode_stream` and `Transport::send_streaming`.
**Outstanding**: Persistent backends (`PostgresCheckpointer`,
`RedisStore`, `with_session_lock`) — needs DB infra; cloud transports
(`BedrockTransport`, `VertexTransport`, `FoundryTransport`) — needs
cloud auth; MCP (`McpManager`) — needs an MCP server.

## Context

Phase 3 closed 2026-04-26 with the composition spine (Runnable +
graph + agents) functionally complete and `StreamMode::Messages`
falling back to a single `Value` chunk because the underlying codec
+ transport had no streaming surface. Phase 4 was scoped (PLAN.md §8)
across persistence, cloud transports, MCP, and the multi-codec
matrix. **The subset closed in this ADR is the codec + streaming
half** — the half that is exercisable end-to-end without external
infrastructure (DBs, cloud IAM, MCP servers). The infra-bound work
moves to Phase 4-bis (separate retro).

## What landed

### Slice 1 — `Codec` / `Transport` streaming trait surface

Two trait extensions, both with graceful default impls so existing
codec/transport implementors keep compiling:

- `Codec::encode_streaming(&request) -> Result<EncodedRequest>` —
  default delegates to `encode` and marks the request streaming.
- `Codec::decode_stream(bytes_in, warnings_in) -> BoxDeltaStream` —
  default buffers every chunk, runs `decode` once, and emits a
  semantically equivalent `StreamDelta` sequence built by a private
  `deltas_from_response` helper.
- `Transport::send_streaming(request, ctx) -> TransportStream` —
  default calls `send` and wraps the buffered body in a single-chunk
  stream.
- New types: `BoxByteStream<'a>`, `BoxDeltaStream<'a>` (codec module
  level), `TransportStream { status, headers, body }` (transport
  module).
- `EncodedRequest { streaming: bool, ... }` flag + `into_streaming`
  builder.

`ChatModel::stream_deltas(messages, ctx)` orchestrates the three:
encode_streaming → send_streaming → decode_stream → typed delta
stream. The `Runnable<Vec<Message>, Message>` impl on `ChatModel`
overrides `Runnable::stream` so Phase 3's `StreamMode::Messages`
emits real `RunnableStreamChunk::Message(StreamDelta)` items with
`StreamAggregator`-driven aggregation.

### Slice 2 — Anthropic SSE streaming + `DirectTransport::send_streaming`

- `AnthropicMessagesCodec::encode_streaming` adds `stream: true` to
  the body and `Accept: text/event-stream` to headers.
- `AnthropicMessagesCodec::decode_stream` walks the SSE event sequence:
  `message_start` → `Start { id, model }`,
  `content_block_start` (text/tool_use) → opens block,
  `content_block_delta` (text_delta / input_json_delta) → emits
  `TextDelta` / `ToolUseInputDelta`, `content_block_stop` → emits
  `ToolUseStop` for tool blocks (text closure handled by
  `StreamAggregator::flush_text`), `message_delta` → cumulative
  `Usage` + `stop_reason` cache, `message_stop` → emits `Stop`,
  `error` → surfaces as `Error::Provider`. CRLF and LF frame
  delimiters both supported.
- `DirectTransport::send_streaming` consumes `reqwest`'s
  `bytes_stream()` and forwards to a `TransportStream`. Cancellation
  via `tokio::select!` on `ExecutionContext::cancellation`. Non-2xx
  drains the buffered body so the caller sees the provider's error
  text.
- End-to-end test through `wiremock`: SSE bytes → `DirectTransport`
  → `AnthropicMessagesCodec` → `StreamAggregator` → `ModelResponse`.

### Slices 3-6 — four additional codecs (full encode/decode + streaming for 3 of them)

| Codec | Path | Streaming | LossyEncode coverage |
|---|---|---|---|
| `OpenAiChatCodec` | `/v1/chat/completions` | SSE `data: {...}\n\n` + `data: [DONE]\n\n` | tool_result.is_error stringified into content; provider_options |
| `GeminiCodec` | `:generateContent` / `:streamGenerateContent?alt=sse` | SSE chunked GenerateContentResponse | tool_use_id dropped (gemini doesn't roundtrip), placeholder name on functionResponse, image URL emitted as fileData with image/* mime |
| `OpenAiResponsesCodec` | `/v1/responses` | rich SSE events (`response.created`, `response.output_text.delta`, `response.function_call_arguments.delta`, `response.completed`, `response.error`) | function_call_output has no error flag |
| `BedrockConverseCodec` | `/model/{id}/converse` (+ `/converse-stream`) | binary `vnd.amazon.eventstream` — **deferred** to `entelix-cloud` (decode_stream uses default fallback) | image URLs not accepted (base64 only) |

All five share a common parser shape (LF/CRLF SSE frame splitter +
`data:` line accumulator) but **do not** lift it into a common helper
module. Each codec owns its parser inline — the per-codec event names
and state machines diverge enough that a shared helper would either
be one giant if-tree or a leaky abstraction. The repetition is
intentional and limited to ~15 lines per codec.

### Slice 7 — closure

- `examples/09_multi_codec.rs` — same `ModelRequest` encoded by all
  five codecs side-by-side, demonstrating IR neutrality.
- `examples/10_streaming.rs` — synthetic Anthropic SSE byte stream
  pumped through `decode_stream` → `StreamAggregator`, no LLM
  dependency.
- 5 codec capability tables surfaced via `Codec::capabilities`.

### Test surface

| Crate | New tests |
|---|---|
| `entelix-core::codec_streaming_fallback` | +3 |
| `entelix-core::anthropic_streaming` | +6 |
| `entelix-core::direct_transport_streaming` | +3 |
| `entelix-core::openai_chat_codec` | +12 |
| `entelix-core::gemini_codec` | +11 |
| `entelix-core::openai_responses_codec` | +10 |
| `entelix-core::bedrock_converse_codec` | +9 |

54 new mock-tested behaviours; ~291 total at workspace level.

### Gates — all green

- `cargo fmt --all -- --check`
- `cargo build --workspace --all-features --examples`
- `cargo clippy --workspace --all-features --all-targets -- -D warnings`
- `cargo test --workspace --all-features`
- `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps`
- `scripts/check-{no-fs,managed-shape,naming}.sh`

## Design decisions surfaced this phase

### 1. `Codec::decode_stream` returns `BoxDeltaStream`, not `&mut StreamAggregator`

Two shapes were considered:

a. **Push model** — `Codec::feed_chunk(&self, bytes, &mut StreamAggregator) -> Result<()>`.
b. **Pull model** — `Codec::decode_stream(bytes_stream) -> BoxDeltaStream`.

We chose (b). Reasons:
- The codec needs internal parser state (tool block open/closed, last
  stop_reason, accumulated buffer). Putting that state in the codec's
  trait method means `&mut self` and per-call lifecycle, breaking the
  "stateless codec" invariant. With (b) the state lives in the
  generator's local closure scope — codec stays `&self`.
- Symmetry with `decode`: `decode` is bytes → IR; `decode_stream` is
  byte-stream → IR-stream. The shape is parallel.
- Composability: a `BoxDeltaStream` plugs directly into the same
  `StreamAggregator` users already know from non-streaming use.

### 2. Trait-level default impls preserve old codecs without changing them

Adding two new trait methods to `Codec` and one to `Transport` would
normally break every implementor. Default impls at the trait level
prevent that:

- `encode_streaming` defaults to `encode + into_streaming` — adequate
  for codecs whose streaming body shape only differs by a flag.
- `decode_stream` defaults to "buffer everything, run `decode`,
  reify into `StreamDelta`s" — preserves the `StreamMode::Messages`
  contract even when the codec has no real SSE parser.
- `send_streaming` defaults to "call `send`, wrap into single-chunk
  stream" — works for any blocking transport.

This choice meant the `BedrockConverseCodec` ships with no
custom `decode_stream` impl at all and the workspace stays green —
binary event-stream parsing arrives later in `entelix-cloud`
without touching this crate.

### 3. Per-codec parser inlining over a shared SSE helper module

OpenAI Chat / OpenAI Responses / Gemini / Anthropic all use SSE
frames separated by `\n\n` with `data:` payload lines, but the
**event taxonomy diverges**: Anthropic has typed events
(`message_start`, `content_block_delta`, …); OpenAI Chat just sends
typed JSON in untyped `data:` lines; OpenAI Responses sends typed
events with rich IDs; Gemini sends untyped chunked
`GenerateContentResponse`s. Trying to abstract those into a single
"SSE codec" trait either:

- creates a half-abstraction that reads "if anthropic then X else if
  openai then Y" inside the helper, or
- forces every codec to expose its own taxonomy through a vtable,
  which is just a less-direct version of the inline approach.

We chose to keep the SSE byte-frame splitter (`find_double_newline`,
`parse_sse_data`) duplicated as ~10-line helpers per codec module.
Total duplication: ~50 lines across five codecs. The decision is
revisited in Phase 5 if a sixth SSE codec or a binary streaming
codec lands and the duplication crosses 100 lines.

### 4. Bedrock binary event-stream defers to `entelix-cloud`

`vnd.amazon.eventstream` is a binary AWS framing protocol with
12-byte headers, CRC32 checksums, and length prefixes. Implementing
it requires either `aws-smithy-eventstream` (an AWS SDK crate
external to entelix's own dependency graph as of Phase 4) or a
hand-rolled parser. Both belong alongside SigV4 signing, which
itself needs `aws-sigv4`. Both arrive together in `entelix-cloud`'s
`BedrockTransport`. Until then, `BedrockConverseCodec::decode_stream`
falls back to the trait default — buffer the whole body, decode once.
The stop-gap behaviour is acceptable because a Bedrock
**non-streaming** call still works end-to-end through this codec.

### 5. `ChatModel::stream_deltas` (inherent) vs `Runnable::stream` (trait)

These had a name collision once the streaming impls landed in
`entelix-runnable`. We renamed the inherent method to
`stream_deltas` rather than disambiguating with UFCS. Reasons:

- The inherent return type is `BoxDeltaStream<StreamDelta>` —
  meaningfully different from `Runnable::stream`'s
  `BoxStream<RunnableStreamChunk<Message>>`. Different name = honest.
- `stream_deltas` is the codec-direct surface; `stream` is the
  composition surface (with mode selection, aggregation,
  finalization). Keeping them named differently surfaces the
  layering at the call site.

## What was deferred and why

| Deferred | To | Rationale |
|---|---|---|
| `BedrockTransport`, `VertexTransport`, `FoundryTransport` | Phase 4-bis (`entelix-cloud`) | Cloud-IAM signing + OAuth refresh; needs SigV4 / GCP service-account / AAD. |
| Bedrock binary event-stream `decode_stream` | Phase 4-bis | Lives with BedrockTransport. |
| `PostgresCheckpointer`, `RedisCheckpointer`, `Postgres/RedisStore`, `with_session_lock` | Phase 4-bis (`entelix-persistence`) | Needs DB schema + advisory-lock decisions. |
| `McpManager` (`entelix-mcp`) | Phase 4-bis | Native JSON-RPC client + tenant-keyed pool design (ADR-0004 — rmcp dependency rejected). |
| Live multi-codec verification | HANDOFF §9 trigger | Cost-bearing, user-approval gated. |

## Phase 4-bis entry plan (next)

The cloud + persistence + MCP work is grouped together because all
three share infrastructure dependencies (auth refresh, connection
pooling, schema migrations). Likely first slice: SigV4 signing in
`entelix-cloud::BedrockTransport` plus the binary event-stream
parser. That unblocks live Bedrock verification (closes the AWS arm
of the codec×transport matrix) without committing to GCP/Azure
auth flows yet.

## References

- ADR-0006 — Runnable + StateGraph spine decision
- PLAN.md §8 Phase 4 — original scope (codec subset is the largest
  closed slice; persistence / cloud / MCP move to Phase 4-bis)
- HANDOFF.md §5e — pre-Phase-4 task list (codec subset consumed,
  rest reorganised)
- HANDOFF.md §9 — live-API approval trigger
