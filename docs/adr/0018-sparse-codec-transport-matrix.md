# ADR 0018 — Sparse codec×transport compatibility matrix

**Status**: Accepted
**Date**: 2026-04-27
**Context**: Phase 4-bis Slice C — `VertexTransport` + `FoundryTransport`
landed alongside `BedrockTransport` (Slice B), bringing the cloud
transport count to four. Five `Codec` impls now exist
(`AnthropicMessages`, `OpenAiChat`, `OpenAiResponses`, `Gemini`,
`BedrockConverse`). The naïve cross-product is 5×4 = 20.

## Decision

Of the 20 cells, **only 10 are valid pairings**. The other 10 don't
exist in the wild — vendors don't ship those models on those routes
— and the SDK refuses to advertise them as supported.

| Codec ↓ \ Transport →     | Direct | Bedrock | Vertex | Foundry |
|---------------------------|:------:|:-------:|:------:|:-------:|
| `AnthropicMessagesCodec`  |   ✓    |   ✓     |   ✓    |    ✓    |
| `OpenAiChatCodec`         |   ✓    |   ·     |   ·    |    ✓    |
| `OpenAiResponsesCodec`    |   ✓    |   ·     |   ·    |    ·    |
| `GeminiCodec`             |   ✓    |   ·     |   ✓    |    ·    |
| `BedrockConverseCodec`    |   ·    |   ✓     |   ·    |    ·    |

10 cells (✓), 10 absent (·). Rationale per cell:

- **`AnthropicMessages × {Direct, Bedrock, Vertex, Foundry}`** —
  Anthropic ships Claude on its own API, on AWS Bedrock, on Google
  Vertex AI, and on Azure AI Foundry. All four pairings accept the
  same `/messages` shape; only auth + base URL differ. The codec is
  oblivious.
- **`OpenAiChat × {Direct, Foundry}`** — `chat/completions` is the
  GA shape on Direct; Azure Foundry exposes the same body via the
  `deployments/{name}/chat/completions` path with API-key or AAD
  auth. Bedrock and Vertex don't ship OpenAI Chat models.
- **`OpenAiResponses × Direct`** — `/v1/responses` is OpenAI's newer
  agentic-shape endpoint. Foundry hadn't surfaced a Responses-shaped
  endpoint as of 2026-04. Bedrock and Vertex never will.
- **`Gemini × {Direct, Vertex}`** — Google ships Gemini on the
  generative-language API (Direct, with API key) and on Vertex AI
  (with OAuth via `gcp_auth`). Anthropic and Azure don't ship
  Gemini.
- **`BedrockConverse × Bedrock`** — `Converse` is AWS's vendor-
  neutral inference API that wraps Anthropic / Cohere / Meta / Mistral
  / DeepSeek models behind a single shape. It only exists on
  Bedrock; running it through any other transport is meaningless.

## Type-level enforcement

`ChatModel<C, T>` is generic over both type parameters with no
`PhantomData` cell-level constraints. The matrix is enforced by:

1. **`Codec::capabilities()` honesty** — `OpenAiChatCodec` claims it
   doesn't support `vendor: gemini` etc., so attempting to encode an
   IR `ModelRequest` for a model the codec doesn't recognise still
   succeeds (the codec is permissive about model strings) but the
   *runtime* call hits the wrong endpoint and fails.
2. **`crates/entelix-cloud/tests/compat_matrix.rs`** — one `#[test]`
   per ✓ cell that calls `ChatModel::new(codec, transport, model)`
   and asserts the construction compiles. Pairs the matrix says are
   absent are intentionally not tested; if someone adds them by
   mistake the test reads odd at review and the comment table is
   stale.
3. **`examples/12_compat_matrix.rs`** — runtime walk that prints
   each ✓ cell's encoded wire bytes side-by-side, so a reader can
   verify the matrix at a glance.

We deliberately do **not** add a sealed marker trait `pub trait
ValidPairing<C, T> {}` with manual `impl` per cell. Reasons:

- Adds a third source of truth (the table, the tests, the marker
  impls) — three sources drift; two don't.
- The "phantom invalid" pairings *do* compile today (the type system
  has no reason to reject them) and that's fine — they just won't
  reach a real endpoint at runtime. Type-level rejection would be
  pure ceremony with zero runtime safety improvement.
- New cells (a future Bedrock-Cohere codec etc.) need no marker
  bookkeeping; just add a `#[test]` and a docs row.

## Adding a new cell

Process when a new (codec, transport) pair becomes real:

1. Add a row or column to the table above.
2. Add a `#[test]` in `crates/entelix-cloud/tests/compat_matrix.rs`.
3. Add a `print_pair(...)` line in `examples/12_compat_matrix.rs`.
4. Note it in the next phase retrospective ADR.

## Removing a cell (deprecation)

If a vendor sunsets a route (e.g. Bedrock drops a codec), invariant
12 (no compat shims) applies — delete the test, the example line,
and the docs row in the same PR. Never leave a `· (was ✓)` marker.

## References

- ADR-0015 — Phase 4 codec subset retro (5 codecs landing)
- `crates/entelix-cloud/tests/compat_matrix.rs` — type-level proof
- `crates/entelix/examples/12_compat_matrix.rs` — runtime walk
  matrix (every codec assumed every transport; intentionally
  tightened for entelix)
