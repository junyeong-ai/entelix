# Provider Capabilities Matrix

Per-codec coverage of every cross-provider IR field, vendor-built-in tool, and reasoning / multimodal surface. Use this matrix to preflight a `(model, codec)` pairing before the first call — `LossyEncode` warnings surface at runtime when an operator-set IR field has no native wire equivalent on the chosen codec; this table makes the same fact discoverable at design time.

The IR-promotion criterion (CLAUDE.md invariant 22) admits a vendor knob into `ModelRequest` only when **two or more first-party codecs** carry the concept natively. Vendor-only knobs live on per-vendor `*Ext` (e.g. `OpenAiResponsesExt::reasoning_summary`).

## Sampling parameters

| IR field | Anthropic | OpenAI Chat | OpenAI Responses | Gemini | Bedrock Converse |
|---|---|---|---|---|---|
| `temperature` | ✅ native | ✅ native | ✅ native | ✅ `generationConfig.temperature` | ✅ `inferenceConfig.temperature` |
| `top_p` | ✅ native | ✅ native | ✅ native | ✅ `generationConfig.topP` | ✅ `inferenceConfig.topP` |
| `top_k` | ✅ native | ⚠️ LossyEncode | ⚠️ LossyEncode | ✅ `generationConfig.topK` | ✅ on-Anthropic via `additionalModelRequestFields.top_k`; ⚠️ LossyEncode for non-Anthropic models |
| `max_tokens` | ✅ native (mandatory) | ✅ native | ✅ native | ✅ `generationConfig.maxOutputTokens` | ✅ `inferenceConfig.maxTokens` |
| `stop_sequences` | ✅ native | ✅ `stop` | ✅ `stop` | ✅ `generationConfig.stopSequences` | ✅ `inferenceConfig.stopSequences` |
| `parallel_tool_calls` | ✅ inverted via `tool_choice.disable_parallel_tool_use` | ✅ native | ✅ native | ⚠️ LossyEncode | ✅ on-Anthropic via inverted polarity; ⚠️ LossyEncode otherwise |

Per-call overrides flow through `RequestOverrides` (attached to `ExecutionContext`) for every field above. See `crates/entelix-core/src/overrides.rs` for the full builder surface.

## Structured output

| Surface | Anthropic | OpenAI Chat | OpenAI Responses | Gemini | Bedrock Converse |
|---|---|---|---|---|---|
| `OutputStrategy::Native` | — | ✅ `response_format` strict JSON Schema | ✅ `text.format` strict JSON Schema | ✅ `responseJsonSchema` | ✅ on-Anthropic via `additionalModelRequestFields.output_config` |
| `OutputStrategy::Tool` | ✅ forced-tool | ⚠️ degrades to Native | ⚠️ degrades to Native | — | ✅ on-Anthropic |
| `OutputStrategy::Auto` | resolves to `Tool` | resolves to `Native` | resolves to `Native` | resolves to `Native` | resolves to `Native` (or `Tool` per model) |
| `OutputStrategy::Prompted` | deferred to 1.1 | deferred to 1.1 | deferred to 1.1 | deferred to 1.1 | deferred to 1.1 |

`complete_typed::<O>` and the `with_structured_output::<O>()` Runnable adapter route through `auto_output_strategy(model)` so operators rarely need to pick a strategy explicitly.

## Reasoning / thinking

| Surface | Anthropic | OpenAI Chat | OpenAI Responses | Gemini | Bedrock Converse |
|---|---|---|---|---|---|
| `reasoning_effort: ReasoningEffort` | ✅ adaptive vs enabled per model family | ⚠️ no reasoning channel — LossyEncode | ✅ `reasoning.effort` | ✅ `thinkingBudget` (2.5) / `thinkingLevel` (3.0) | ✅ on-Anthropic via thinking passthrough |
| Thinking blocks (`ContentPart::Thinking`) | ✅ native + `signature` | ⚠️ LossyEncode | ✅ Responses API surfaces reasoning items | ✅ Gemini thinking | ✅ on-Anthropic |
| `OpenAiResponsesExt::reasoning_summary` | — | — | ✅ vendor-only | — | — |

Reasoning text never reaches the LLM channel by default — `RenderedForLlm<T>` (invariant 16) gates explicit forwarding. Operator dashboards see the full reasoning content via `AuditSink` and OTel spans.

## Prompt caching

| Surface | Anthropic | OpenAI Chat | OpenAI Responses | Gemini | Bedrock Converse |
|---|---|---|---|---|---|
| `cache_control` per `ContentPart` / `ToolSpec` / `SystemBlock` | ✅ native (`type: "ephemeral"`, TTL `5m` / `1h`) | ⚠️ LossyEncode | ⚠️ LossyEncode | ⚠️ LossyEncode | ✅ on-Anthropic |
| `ModelRequest.cache_key` (auto-cache routing) | ⚠️ LossyEncode | ✅ `prompt_cache_key` | ✅ `prompt_cache_key` | ⚠️ LossyEncode | ⚠️ LossyEncode |
| `ModelRequest.cached_content` (server-side cache) | ⚠️ LossyEncode | ⚠️ LossyEncode | ⚠️ LossyEncode | ✅ `cachedContent` | ⚠️ LossyEncode |
| Cache token telemetry | ✅ `cache_creation_input_tokens` + `cached_input_tokens` | ✅ `cached_input_tokens` (auto-cache) | ✅ `cached_input_tokens` | ✅ Gemini cache hit metric | ✅ on-Anthropic |

OTel emits `gen_ai.usage.cached_input_tokens` and `gen_ai.usage.cache_creation_input_tokens` per the GenAI semconv 0.32 schema (see `entelix-otel`).

## Vendor built-in tools

| `ToolKind` variant | Anthropic | OpenAI Chat | OpenAI Responses | Gemini | Bedrock Converse |
|---|---|---|---|---|---|
| `Function` (operator-defined) | ✅ | ✅ | ✅ | ✅ | ✅ |
| `WebSearch` | ✅ native | ⚠️ LossyEncode | ✅ native | ✅ `googleSearch` grounding | ⚠️ LossyEncode |
| `Computer` | ✅ native | ⚠️ LossyEncode | ✅ native (computer use) | ⚠️ LossyEncode | ⚠️ LossyEncode |
| `TextEditor` | ✅ native | ⚠️ LossyEncode | ⚠️ LossyEncode | ⚠️ LossyEncode | ⚠️ LossyEncode |
| `Bash` | ✅ native | ⚠️ LossyEncode | ⚠️ LossyEncode | ⚠️ LossyEncode | ⚠️ LossyEncode |
| `CodeExecution` | ✅ native | ⚠️ LossyEncode | ⚠️ LossyEncode | ✅ `codeExecution` | ⚠️ LossyEncode |
| `FileSearch` | ⚠️ LossyEncode | ⚠️ LossyEncode | ✅ native | ⚠️ LossyEncode | ⚠️ LossyEncode |
| `CodeInterpreter` | ⚠️ LossyEncode | ⚠️ LossyEncode | ✅ native | ⚠️ LossyEncode | ⚠️ LossyEncode |
| `ImageGeneration` | ⚠️ LossyEncode | ⚠️ LossyEncode | ✅ `image_generation` | ⚠️ LossyEncode | ⚠️ LossyEncode |
| `McpConnector` | ✅ native | ⚠️ LossyEncode | ⚠️ LossyEncode | ⚠️ LossyEncode | ⚠️ LossyEncode |
| `Memory` | ✅ native | ⚠️ LossyEncode | ⚠️ LossyEncode | ⚠️ LossyEncode | ⚠️ LossyEncode |

`Codec::capabilities(model: &str) -> Capabilities` carries the typed flags (`web_search`, `computer_use`, `code_execution`, `multimodal_image`, `multimodal_document`, `multimodal_audio`, `thinking`, `prompt_cache`) so operators preflight at runtime when a model is dynamically selected.

## Multimodal

| `ContentPart` variant | Anthropic | OpenAI Chat | OpenAI Responses | Gemini | Bedrock Converse |
|---|---|---|---|---|---|
| `Image` (input) — Base64 / Url / FileId | ✅ all three sources | ✅ all three sources | ✅ all three sources | ✅ Base64 + Url; FileId via cache | ✅ on-Anthropic |
| `Document` (input) — PDF / docx | ✅ native; FileId via Files API | ⚠️ LossyEncode | ✅ FileId via Files API | ✅ inline | ✅ on-Anthropic |
| `Audio` (input) | ⚠️ LossyEncode | ✅ native (gpt-4o-audio) | ✅ native | ✅ inline | ⚠️ LossyEncode |
| `Video` (input) | ⚠️ LossyEncode | ⚠️ LossyEncode | ⚠️ LossyEncode | ✅ inline | ⚠️ LossyEncode |
| `ImageOutput` | ⚠️ LossyEncode | ⚠️ LossyEncode | ✅ via `image_generation` | ✅ inline | ⚠️ LossyEncode |
| `AudioOutput` (text + transcript) | ⚠️ LossyEncode | ⚠️ partial | ✅ partial | ✅ partial | ⚠️ LossyEncode |

`MediaSource` encodes provenance — `Base64 { media_type, data }` for inline, `Url { url, media_type }` for vendor-fetched, `FileId { id, media_type }` for vendor-pre-uploaded references. See `examples/24_file_id_attachments.rs` for the upload workflow.

## Streaming

| Delta | Anthropic | OpenAI Chat | OpenAI Responses | Gemini | Bedrock Converse |
|---|---|---|---|---|---|
| `StreamDelta::TextDelta` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `StreamDelta::ToolUseStart` / `ToolUseInputDelta` (partial JSON) / `ToolUseStop` | ✅ | ✅ | ✅ | ✅ | ✅ via SigV4 event-stream |
| `StreamDelta::ThinkingDelta { text, signature }` | ✅ | — | ✅ on `o1`/`o3`/`o4` | ✅ Gemini thinking | ✅ on-Anthropic |
| `StreamDelta::Stop { stop_reason, usage }` (terminal) | ✅ | ✅ | ✅ | ✅ | ✅ |
| `StreamAggregator` reconstruction | uniform across all codecs |

Cost emission (`gen_ai.usage.cost`) fires on the terminal `StreamDelta::Stop` after `StreamAggregator::finalize()` produces the final `ModelResponse`. A stream that errors mid-flight surfaces the error and never charges (invariant 12).

## Trait production-impl status

Operator-extension traits ship with a documented "production impl status" so expectations stay honest:

| Trait | Status | First-party impl(s) | Companion crate(s) |
|---|---|---|---|
| `Codec` | ✅ shipped | `AnthropicMessagesCodec`, `OpenAiChatCodec`, `OpenAiResponsesCodec`, `GeminiCodec`, `BedrockConverseCodec` | — |
| `Transport` | ✅ shipped | `DirectTransport` | `entelix-cloud` (Bedrock SigV4, Vertex GCP, Azure Foundry) |
| `Tool<D>` | ✅ shipped | `Calculator`, `HttpFetchTool`, `SchemaToolAdapter`, sandboxed-* | — |
| `Embedder` | ✅ shipped | `MeteredEmbedder<E>` (cost-tracking decorator) | `entelix-memory-openai` |
| `VectorStore` | ✅ shipped | `InMemoryVectorStore` | `entelix-memory-pgvector`, `entelix-memory-qdrant` |
| `GraphMemory<N, E>` | ✅ shipped | `InMemoryGraphMemory` | `entelix-graphmemory-pg` |
| `Checkpointer<S>` | ✅ shipped | `InMemoryCheckpointer` | `entelix-persistence` (Postgres + Redis) |
| `SessionLog` | ✅ shipped | `InMemorySessionLog` | `entelix-persistence` (Postgres + Redis) |
| `Approver` | ✅ shipped | `AlwaysApprove`, `ChannelApprover` | — |
| `AuditSink` | ✅ shipped | `SessionAuditSink` (in `entelix-session`) | — |
| `Sandbox` | ⚠️ **BYO** | `MockSandbox` (in-memory, for tests) | companion crate (e2b, modal, …) when ecosystem demand lines up |
| `SearchProvider` | ⚠️ **BYO** | — | companion crate (Tavily, Brave, …) when ecosystem demand lines up |
| `RootsProvider` / `ElicitationProvider` / `SamplingProvider` | ✅ shipped | `Static*` impls | `entelix-mcp` `chatmodel-sampling` feature (closes the `Sampling` operator side) |
| `RateLimiter` | ✅ shipped | `TokenBucketLimiter` | — |
| `PiiRedactor` | ✅ shipped | `RegexRedactor` | — |
| `DistributedLock` | ✅ shipped | — | `entelix-persistence` (Postgres advisory lock, Redis lock) |
| `Clock` | ✅ shipped | `SystemClock` | — |

**BYO traits** — operators implementing `Sandbox` and `SearchProvider` themselves wire whatever backend (e2b, Tavily, Brave) their deployment requires. No first-party companion ships today — a placeholder companion would violate invariant 14 (no production-shaped fakes). Companions land per-vendor when a stable backend choice consolidates in the ecosystem.
