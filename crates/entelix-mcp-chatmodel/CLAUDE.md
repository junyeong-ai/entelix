# entelix-mcp-chatmodel

Companion crate (vendor-neutral). Bridges the MCP `sampling/createMessage` server-initiated channel onto a `ChatModel<C, T>` so the same model the agent uses for its own reasoning serves the MCP server's sampling requests.

## Surface

- **`ChatModelSamplingProvider<C, T>`** — wraps `Arc<ChatModel<C, T>>` and impls `entelix_mcp::SamplingProvider`. Translates `SamplingRequest` → IR `ModelRequest` → dispatch through the chat model's full layer stack → IR `ModelResponse` → `SamplingResponse`. Per-request `system_prompt` / `temperature` / `max_tokens` / `stop_sequences` override the underlying model's defaults for that one dispatch via a per-call clone.

## Crate-local rules

- **Sampling shares the agent's layer stack** — `PolicyLayer`, `OtelLayer`, retry middleware, cost meter all apply transparently to sampling dispatches because the wrapper holds an `Arc<ChatModel>`. Operators get cost roll-up + tenant policy + observability for free.
- **Multi-block responses lose to the wire format** — MCP sampling's `SamplingContent` is single-block. When the model emits `Thinking + Text + Image`, the adapter takes the FIRST text/image/audio part and drops the rest with a `tracing::warn!`. The forward path emits `LossyEncode` at the conversion boundary.
- **`model_preferences` and `include_context` are advisory** — surfaced via `tracing::debug!` and otherwise passed through unchanged. Operators with bespoke model-routing requirements implement `SamplingProvider` directly.
- **Per-request override is non-mutating** — overrides happen on a `ChatModel::clone()` (Arc-cheap), so concurrent dispatches see independent override stacks.

## Forbidden

- Mutating the wrapped `ChatModel` between calls — overrides flow through the per-call clone.
- Dropping `model_preferences` silently — at least the `tracing::debug!` must fire so operators can audit which advisory hints arrived.


- `crates/entelix/examples/17_mcp_sampling_provider.rs` — integration demo.
