# entelix-mcp

Native Model Context Protocol client. JSON-RPC 2.0 over MCP streamable-http.

## Surface

- **`McpManager`** — register servers, dispatch `tools/call`, `resources/{list,read}`, `prompts/{list,get}`, `completion/complete`, `notifications/roots/list_changed`. Per-tenant connection pool keyed by `(TenantId, ServerName)` (invariant 11).
- **`McpServerConfig`** — HTTP-only by design. `with_roots_provider` / `with_sampling_provider` / `with_elicitation_provider` opt the server into the corresponding server-initiated channel.
- **`McpClient` trait** + `HttpMcpClient` — production transport. `Mcp-Session-Id` sticky session + background SSE listener for server-initiated requests. Tests inject deterministic mocks.
- **`McpToolAdapter`** — implements `entelix_core::tools::Tool` so MCP-published tools plug into agents.
- **Server-initiated channel providers** — `RootsProvider` (`roots/list`, `StaticRootsProvider` reference impl + `McpRoot`), `SamplingProvider` (`sampling/createMessage`, ), `ElicitationProvider` (`elicitation/create`, ). One provider per server, per channel.
- **`ChatModelSamplingProvider<C, T>`** (feature `chatmodel-sampling`) — canonical adapter that bridges `SamplingProvider` onto `entelix_core::ChatModel<C, T>`. Wraps `Arc<ChatModel>` and translates each `SamplingRequest` → IR `ModelRequest` → dispatches through the chat model's full layer stack (`PolicyLayer`, `OtelLayer`, retries) → IR `ModelResponse` → `SamplingResponse`. Per-request `system_prompt` / `temperature` / `max_tokens` / `stop_sequences` override the underlying defaults via a per-call clone (Arc-cheap, concurrent dispatches see independent overrides).

## Crate-local rules

- **Pool key is `(tenant_id, server_name)` — never widen to `server_name` alone.** A "default tenant" deployment keys with `DEFAULT_TENANT_ID`; that is still a tenant. Cross-tenant credential sharing is a structural bug.
- **HTTP only.** Stdio MCP servers require `std::process` which invariant 9 forbids first-party. Wrap stdio externally → expose HTTP endpoint.
- Server-initiated dispatcher: per-method `*Provider` trait pattern (Roots / Sampling / Elicitation), not new methods on `McpClient`.
- Stateless servers (no `Mcp-Session-Id` echo): listener task spawn is skipped automatically; server-initiated providers stay dormant.
- SSE response read loop must consume to EOF before returning the matched response — server may interleave additional server-initiated requests on the same stream.
- **`ChatModelSamplingProvider` shares the agent's layer stack** — `PolicyLayer`, `OtelLayer`, retry middleware, cost meter all apply transparently to sampling dispatches because the wrapper holds an `Arc<ChatModel>`. Operators get cost roll-up + tenant policy + observability for free.
- **`ChatModelSamplingProvider` multi-block responses lose to the wire format** — MCP sampling's `SamplingContent` is single-block. When the model emits `Thinking + Text + Image`, the adapter takes the FIRST text/image/audio part and drops the rest with a `tracing::warn!`. The forward path emits `LossyEncode` at the conversion boundary.
- **`ChatModelSamplingProvider` per-request overrides ride a per-call clone** — overrides happen on `ChatModel::clone()` (Arc-cheap), so concurrent dispatches see independent override stacks; the wrapped model is never mutated between calls.

## Forbidden

- `tokio::process` / `std::process` for stdio MCP servers.
- Per-server-only pool key (drops invariant 11).
- Hard-coded protocol version: bump `protocol::PROTOCOL_VERSION` in lockstep with the spec revision; do not fall back silently.
- Mutating the `ChatModel` wrapped by `ChatModelSamplingProvider` between calls — overrides flow through the per-call clone.
