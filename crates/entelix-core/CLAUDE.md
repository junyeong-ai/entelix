# entelix-core

DAG root of the workspace. Depends on **no** other entelix crate. Adding any `entelix-*` to `Cargo.toml` here breaks the workspace — wire the dependency the other way around.

## Surface

- **IR** (`ir::`) — `ModelRequest`, `ModelResponse`, `Message`, `ContentPart`, `ToolUse`, `ToolResult`, `StopReason`, `ModelWarning`, `ProviderExtensions`. The provider-neutral payload every codec encodes from / decodes to.
- **Codec** (`codecs::`) — `Codec` trait + impls per vendor wire format (`AnthropicMessagesCodec`, `OpenAiChatCodec`, `OpenAiResponsesCodec`, `GeminiCodec`, `BedrockConverseCodec`). Stateless, totalable: `encode` is total, `decode` is panic-free on arbitrary bytes (verified by `tests/codec_robustness_proptest.rs`).
- **Transport** (`transports::`) — `Transport` trait + `DirectTransport`. Stateful (auth + connection pool). Cloud-routed transports (Bedrock/Vertex/Foundry) live in `entelix-cloud`.
- **Tool** (`tools::`) — `Tool<D = ()>` trait, `ToolRegistry<D = ()>`, `ToolMetadata`, `ToolEffect`, `RetryHint`. Single `execute(input, ctx: &AgentContext<D>) -> Result<Value>` method (invariant 4). `D` is operator-supplied typed deps (default `()` for the deps-less zero-cost path); the layer ecosystem (`PolicyLayer`, `OtelLayer`, …) consumes the D-free `ToolInvocation` so existing layers compile unchanged regardless of `D` (invariant 19, ADR-0085 + ADR-0089). No back-channels.
- **Auth** (`auth::`) — `CredentialProvider` + `ApiKeyProvider`, `BearerProvider`, `Cached*`, `Chained*`. Tokens never leave this module via context (invariant 10).
- **Service spine** (`service::`) — `ModelInvocation` / `ToolInvocation` `tower::Service` types. Cross-cutting concerns plug in as `tower::Layer<S>`.
- **`ExecutionContext`** (`context::`) — request-scope carrier. Fields: `cancellation`, `deadline`, `thread_id`, `tenant_id` (mandatory, defaults to `DEFAULT_TENANT_ID`), `run_id`, `idempotency_key`, `run_budget`, `audit_sink`, `extensions`. Builder methods: `with_deadline` / `with_thread_id` / `with_tenant_id` / `with_run_id` / `with_idempotency_key` / `with_run_budget` / `with_audit_sink` (single-slot setters), `add_extension` (collection insert per naming taxonomy).
- **`AgentContext<D>`** (`agent_context::`) — typed-Deps carrier. Wraps `ExecutionContext` plus an operator-supplied typed handle `D` (defaults to `()`). Tool dispatch threads `&AgentContext<D>` to leaf execution; the `D`-free `ToolInvocation` rides through every layer so the ecosystem stays generic-free (invariant 19, ADR-0084).
- **`Extensions`** (`extensions::`) — type-keyed `Arc<HashMap<TypeId, …>>`, copy-on-write. Mirrors `http::Extensions`. **Never stash credentials here** — invariant 10 alignment.
- **Chat** (`chat::`) — `ChatModel` + `ChatModelConfig`. `complete` (text), `complete_full` (full `ModelResponse`), `complete_typed::<O>` (typed structured output via `LlmFacingSchema::strip` + `serde_json` deserialization), `complete_typed_validated::<O>(.., validator)` (the typed path with an `OutputValidator<O>` semantic check). `with_validation_retries(n)` enables the unified retry loop that catches both schema-mismatch and validator failures through `Error::ModelRetry` (invariant 20, ADR-0090 + ADR-0091).
- **`OutputValidator<O>`** (`output_validator::`) — post-decode semantic validator. `Fn(&O) -> Result<()>` blanket impl makes the common case a closure; validators returning `Err(Error::ModelRetry { hint, .. })` route through the chat-model retry loop with the hint reflected to the model as a corrective `User` message.

## Crate-local rules

- New codec / transport / tool: implement the trait, add the `*Codec` / `*Transport` / `*Tool` suffix, do not add a method to the trait — extend metadata struct or add a sibling trait instead.
- New IR field: bump `ProviderExtensions` for vendor-specific knobs; only add to `ModelRequest`/`Response` itself if every codec carries it natively.
- Lossy codec encoding: emit `ModelWarning::LossyEncode { codec, dropped_field }`. Silent drop fails review (invariant 6).
- `tests/codec_robustness_proptest.rs` is load-bearing — run after any codec change.

## Forbidden

- `std::fs`, `std::process`, `tokio::fs`, `tokio::process`, `landlock`, `seatbelt` (invariant 9).
- Embedding `CredentialProvider` in `ExecutionContext` (invariant 10).
- Returning `anyhow::Error` from public API.
- Trait methods returning vendor-shaped JSON (invariant 5).

## References

- ADR-0011 — `Tool`/`Runnable` adapter boundary, `SchemaTool` typed-I/O.
- ADR-0017 — `tenant_id` mandatory + `Extensions` slot.
- ADR-0018 — codec × transport sparse matrix (which pairs are valid).
