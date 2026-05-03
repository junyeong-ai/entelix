# entelix-cloud

Cloud-routed `Transport` impls. Provider IR is unchanged across cloud routes — only credential resolution, request signing, and transport-specific framing live here.

## Surface

| Module | Feature flag | Auth | Native codec |
|---|---|---|---|
| `bedrock::BedrockTransport` | `aws` | SigV4 default chain (env, profile, IMDS) + bearer fallback | `BedrockConverseCodec` |
| `vertex::VertexTransport` | `gcp` | `gcp_auth` (ADC, service-account JSON, GKE) | `GeminiCodec` (Vertex-flavoured) |
| `foundry::FoundryTransport` | `azure` | API-key + AAD (`azure_identity`) | `OpenAiResponsesCodec` (Azure-flavoured) |

## Crate-local rules

- **Codec × transport pairing is sparse, not free.** Per ADR-0018, only the listed pairings are valid. A `BedrockTransport` cannot run an `AnthropicMessagesCodec` directly — Anthropic on Bedrock uses Converse. Live integration tests must pair the matched codec; the live-API smoke crate file names encode the intended pair (`live_bedrock.rs` uses `BedrockConverseCodec`).
- **Signing happens in the transport, not the codec.** SigV4 / AAD / OAuth refresh is `Transport::send` territory. The codec sees neutral IR and produces neutral wire bytes; the transport stamps auth headers.
- **Credential refresh is async-cached.** Long-lived processes hit refresh on the timer the credential type carries (e.g. SigV4 chain caches via `aws-config`). New auth chain → wrap with `Cached*Provider` so concurrent requests share one refresh.
- **AWS event-stream framing is transport-internal.** The `:event-type` headers, `:exception-type` mapping, and CRC validation never leak into IR.

## Forbidden

- Codec × transport pairings outside the ADR-0018 sparse matrix.
- Hand-rolled SigV4 / AAD when the official SDK helpers exist (`aws-sigv4` for SigV4, `azure_identity` for AAD).
- Embedding raw API keys in transport struct fields — credentials always flow through `Arc<dyn CredentialProvider>` (invariant 10).

## References

- ADR-0018 — codec × transport sparse matrix.
