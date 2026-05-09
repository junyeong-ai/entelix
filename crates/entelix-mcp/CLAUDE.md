# entelix-mcp

Multi-tenant **operator surface** over the official Model Context Protocol Rust SDK (`rmcp`). rmcp owns the wire layer — JSON-RPC framing, streamable-http session lifecycle, spec keep-up. entelix-mcp owns everything that is *operator policy*: per-tenant connection pool, frame/dispatch caps, capability-gated server-initiated providers, request decorators, `LlmRenderable` funnel at the dispatch boundary.

## Surface

- **`McpManager`** — register servers, dispatch `tools/call`, `resources/{list,read}`, `prompts/{list,get}`, `completion/complete`, `notifications/roots/list_changed`. Per-tenant connection pool keyed by `(TenantId, ServerName)` (invariant 11).
- **`McpServerConfig`** — HTTP-only by design. `with_roots_provider` / `with_sampling_provider` / `with_elicitation_provider` opt the server into the corresponding server-initiated channel.
- **`McpClient` trait** + `HttpMcpClient` — public seam. `HttpMcpClient` is implemented over `rmcp::ServiceExt::serve` + `StreamableHttpClientTransport`; the trait survives so third-party impls (test mocks, alternate transports) plug into `McpManager` without touching rmcp.
- **`McpToolAdapter`** — implements `entelix_core::tools::Tool` so MCP-published tools plug into agents.
- **Server-initiated channel providers** — `RootsProvider` (`roots/list`, `StaticRootsProvider` reference impl + `McpRoot`), `SamplingProvider` (`sampling/createMessage`), `ElicitationProvider` (`elicitation/create`). One provider per server, per channel. Bridged to rmcp via an internal `ClientHandler` adapter that overrides the matching method when (and only when) the operator has wired a provider.

## Architecture split

| Concern | Owner | Notes |
|---|---|---|
| Wire framing, JSON-RPC, streamable-http session | **rmcp** | spec keep-up tracked upstream |
| Protocol version constant | **rmcp** | re-exported as `protocol::PROTOCOL_VERSION` |
| `tools/call`, `resources/*`, `prompts/*`, `completion/complete`, `logging/setLevel` | **rmcp** | reached via `Peer<RoleClient>` deref |
| Server-initiated channel routing (`create_message` / `list_roots` / `create_elicitation`) | **rmcp `ClientHandler`** | bridged to entelix `*Provider` traits |
| Per-tenant connection pool | **entelix-mcp** | `(tenant_id, server_name)` key |
| Frame-size cap, dispatch-concurrency cap, idle TTL | **entelix-mcp** | `tower::Layer` over the rmcp transport |
| Per-request trace-context headers (W3C) | **entelix-mcp** | reqwest middleware (rmcp's `custom_headers` is connection-wide) |
| Cancellation bridge | **entelix-mcp** | forwards `ctx.cancellation()` to `RunningService::cancellation_token` + per-call `tokio::select!` |
| `LlmRenderable` funnel on response path | **entelix-mcp** | invariant 16 |

## Crate-local rules

- **Pool key is `(tenant_id, server_name)` — never widen to `server_name` alone.** A "default tenant" deployment keys with `DEFAULT_TENANT_ID`; that is still a tenant. Cross-tenant credential sharing is a structural bug.
- **HTTP only.** Stdio MCP servers require `std::process` which invariant 9 forbids first-party. The rmcp dep is feature-scrubbed: `default-features = false, features = ["client", "transport-streamable-http-client-reqwest"]` — no `transport-io`, no `transport-child-process`. Wrap stdio externally → expose HTTP endpoint.
- **`ClientHandler::create_elicitation` MUST be overridden whenever an `ElicitationProvider` is wired** — rmcp's default implementation auto-declines, which would silently NACK every elicitation request and is invisible at the operator surface.
- Server-initiated dispatcher: per-method `*Provider` trait pattern (Roots / Sampling / Elicitation), not new methods on `McpClient`.
- Stateless servers (no `Mcp-Session-Id` echo): listener task spawn is skipped automatically; server-initiated providers stay dormant.
- Frame-flood + dispatch-flood caps live in our `tower::Layer` stack, not in rmcp — rmcp's 1.6 hardening (origin validation, init timeout) is server-side only.

## Forbidden

- `tokio::process` / `std::process` for stdio MCP servers; `transport-io` or `transport-child-process` rmcp features.
- Per-server-only pool key (drops invariant 11).
- Hard-coded protocol version: read `rmcp::model::ProtocolVersion::LATEST`; do not fall back silently.
- Routing server-initiated content directly to the model — every payload that crosses the boundary into `Message` flows through `LlmRenderable` (invariant 16).
