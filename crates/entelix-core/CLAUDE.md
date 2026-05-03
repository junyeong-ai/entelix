# entelix-core

DAG root of the workspace. Depends on **no** other entelix crate. Adding any `entelix-*` to `Cargo.toml` here breaks the workspace — wire the dependency the other way around.

## Surface

- **IR** (`ir::`) — `ModelRequest`, `ModelResponse`, `Message`, `ContentPart`, `ToolUse`, `ToolResult`, `StopReason`, `ModelWarning`, `ProviderExtensions`. The provider-neutral payload every codec encodes from / decodes to.
- **Codec** (`codecs::`) — `Codec` trait + impls per vendor wire format (`AnthropicMessagesCodec`, `OpenAiChatCodec`, `OpenAiResponsesCodec`, `GeminiCodec`, `BedrockConverseCodec`). Stateless, totalable: `encode` is total, `decode` is panic-free on arbitrary bytes (verified by `tests/codec_robustness_proptest.rs`).
- **Transport** (`transports::`) — `Transport` trait + `DirectTransport`. Stateful (auth + connection pool). Cloud-routed transports (Bedrock/Vertex/Foundry) live in `entelix-cloud`.
- **Tool** (`tools::`) — `Tool` trait, `ToolRegistry`, `ToolMetadata`, `ToolEffect`, `RetryHint`. Single `execute(input, ctx) → output` method (invariant 4). No back-channels.
- **Auth** (`auth::`) — `CredentialProvider` + `ApiKeyProvider`, `BearerProvider`, `Cached*`, `Chained*`. Tokens never leave this module via context (invariant 10).
- **Service spine** (`service::`) — `ModelInvocation` / `ToolInvocation` `tower::Service` types. Cross-cutting concerns plug in as `tower::Layer<S>`.
- **`ExecutionContext`** (`context::`) — request-scope carrier. Fields: `cancellation`, `deadline`, `thread_id`, `tenant_id` (mandatory, defaults to `DEFAULT_TENANT_ID`), `run_id`, `extensions`. Builder methods: `with_deadline` / `with_thread_id` / `with_tenant_id` / `with_run_id` (single-slot setters), `add_extension` (collection insert per naming taxonomy).
- **`Extensions`** (`extensions::`) — type-keyed `Arc<HashMap<TypeId, …>>`, copy-on-write. Mirrors `http::Extensions`. **Never stash credentials here** — invariant 10 alignment.

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
