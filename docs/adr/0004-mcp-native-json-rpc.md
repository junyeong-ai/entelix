# ADR 0004 — MCP integration: native JSON-RPC over streamable-http

**Status**: Accepted
**Date**: 2026-04-26

## Context

The Model Context Protocol (MCP) is the dominant standard for tool/resource exposure to LLM agents in 2026. Per the Linux Foundation's 2026 announcement (governance moved to the Agentic AI Foundation), MCP is no longer Anthropic-specific.

Two Rust SDK options were considered:
- **rmcp** (`modelcontextprotocol/rust-sdk`) — official, 1.5.0 released 2026-04-16. Bundles client + server + several transports + helpers in one crate.
- **Native impl** — write the JSON-RPC 2.0 client directly on top of `reqwest`. Smaller surface, no upstream version coupling.

Initial intent (D2, Phase 0) was to depend on `rmcp`. Phase 4-bis prototyping revealed that entelix's actual MCP needs are narrow: HTTP-only client transport, per-tenant connection pool, lazy provisioning, F9-mitigated key derivation. rmcp's bundled abstractions (server scaffolding, stdio, multiple SSE flavours, helper macros) are inert weight for an HTTP-only client. Worse, rmcp's transport extension points couple us to its trait shapes — entelix's `RequestDecorator` + `ExecutionContext::tenant_id` flow does not slot cleanly into rmcp's connection setup.

## Decision

`entelix-mcp` is a **native JSON-RPC 2.0 client** built directly on `reqwest`, speaking MCP's **streamable-http** transport. No `rmcp` dependency. stdio is intentionally out of scope (invariant 9 forbids process spawn).

Streamable-http — the MCP 1.5 successor to stdio — multiplexes both directions of the conversation onto a single endpoint:

- **Client → Server requests** ride POST. The response is either a single JSON envelope (stateless servers) or an SSE stream (streamable servers — every event is one JSON-RPC message, matched to the request by `id`).
- **Server → Client requests** arrive on a long-lived `GET /` SSE the client opens once after `initialize`. The dispatcher routes by `method`:
    - `roots/list` → operator-supplied `RootsProvider`
    - any other method → JSON-RPC `-32601 Method not found`.
- **Notifications** (either direction) are POSTs whose body has no `id`. The client emits `notifications/initialized` and `notifications/roots/list_changed`; the server may emit any notifications via the SSE listener.

Sticky session: when the server returns an `Mcp-Session-Id` header on `initialize`, every later client request echoes it. Servers that omit the header signal stateless mode — the client respects that, skips the listener, and `RootsProvider` (if configured) stays dormant. This auto-fallback keeps legacy stateless servers working without the operator opting into a different transport.

**Surface**:

- `McpClient` trait — `initialize` + `call_tool` + `list_resources` + `read_resource` + `list_prompts` + `prompt` + `complete` + `notify_roots_changed` + `state`
- `HttpMcpClient` — production impl over `reqwest` (streamable-http with SSE response support, bearer auth, optional `RequestDecorator`, sticky `Mcp-Session-Id`, background SSE listener with cancellation-token-driven shutdown on drop)
- `McpManager` + `McpManagerBuilder` — per-`(tenant_id, server_name)` connection pool (F9 mitigation)
- `McpClientState` — 11-state forward-only FSM
- `McpRoot` + `RootsProvider` (trait) + `StaticRootsProvider` — server-initiated `roots/list` channel; `McpServerConfig::with_roots_provider` wires per-server providers
- Protocol coverage: MCP 2025-03-26 revision — Tools (`tools/list`, `tools/call`), Resources (`resources/list`, `resources/read`), Prompts (`prompts/list`, `prompts/get`), Completion (`completion/complete`), Roots (`roots/list` server-initiated + `notifications/roots/list_changed` client-initiated).

**Rationale**:

- Surface bounded — 1.0 freeze covers exactly what entelix wires through, not rmcp's union of features.
- No upstream coupling — rmcp's release cadence does not gate entelix's.
- Clean fit with entelix primitives — `ExecutionContext::tenant_id`, `RequestDecorator`, `CredentialProvider`, `McpClientState` FSM all integrate without adapter friction.
- HTTP-only matches entelix's web-service persona — stdio MCP servers are wrapped externally if needed.

**Cargo.toml** (`entelix-mcp`):

```toml
[dependencies]
entelix-core = { workspace = true }
async-trait = { workspace = true }
dashmap = { workspace = true }
futures = { workspace = true }
http = { workspace = true }
parking_lot = { workspace = true }
reqwest = { workspace = true }
secrecy = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
thiserror = { workspace = true }
tokio = { workspace = true }
tokio-util = { workspace = true }   # CancellationToken for the SSE listener
tracing = { workspace = true }
# No rmcp — native JSON-RPC over reqwest + streamable-http SSE.
```

## Consequences

✅ 1.0 surface stays bounded — every public type traces to a real entelix wiring need.
✅ Operator-supplied `RequestDecorator` rides every outbound request (traceparent injection, custom auth headers).
✅ Per-`(tenant, server)` pool enforces tenant isolation at the connection level (F9).
✅ Lazy provisioning — connections open on the first dispatch, not at builder time. Critical for serverless deployments where most requests don't touch every configured MCP server.
✅ MCP 1.5 spec coverage at ~100% of the client surface (Tools + Resources + Prompts + Completion + Roots). Server-initiated `roots/list` rides the streamable-http SSE channel; sampling and elicitation extend the same dispatcher pattern when those land.
❌ Server-side MCP (entelix exposing tools to other MCP clients) is not in scope. If/when needed, a separate `entelix-mcp-server` companion would land.
❌ stdio MCP servers must be wrapped externally (HTTP proxy) — accepted tradeoff, invariant 9 forbids in-process spawn.

## Lifecycle

`entelix-mcp::McpManager` owns the lifecycle:
- **Lazy connect** — first dispatch through `McpManager::call_tool / list_resources / …` opens the connection for the `(tenant, server)` pair.
- **Idle TTL** — `prune_idle` and `prune_idle_per_config` evict idle entries. Operators run these at the cadence of their housekeeping loop.
- **Per-tenant isolation** — `(tenant_id, server_name)` pool key (F9 mitigation; invariant 11).
- **FSM-tracked** — `McpClientState` advances forward-only through 11 states; `Failed` is terminal.
- **Background SSE listener** — `HttpMcpClient` spawns a `tokio` task on `initialize` that reads the long-lived `GET /` SSE stream. The task is owned by a `CancellationToken` + `JoinHandle`; `Drop` cancels and aborts so no detached state survives. Stateless servers (no `Mcp-Session-Id`) skip the listener entirely.

## Roots dispatcher

Server-initiated `roots/list` requests arrive through the SSE listener (or — for streamable POST responses — alongside the matching response on the same SSE). The dispatcher routes by method:

- `roots/list` → `McpServerConfig::roots_provider()` → operator-supplied `RootsProvider::list_roots()` → JSON-RPC response POSTed back to the same endpoint with the original server-side `id`.
- A configured provider that returns `Err` surfaces as JSON-RPC `-32603 Internal error`.
- An absent provider surfaces as `-32601 Method not found` — operators see explicitly that the capability advertisement and the wiring drifted.

`notifications/roots/list_changed` is the inverse direction: client → server, fire-and-forget. Operators trigger it via `McpManager::notify_roots_changed(ctx, server)` whenever the source of truth backing their `RootsProvider` mutates (workspace switched, sandbox re-rooted, etc.).

Future server-initiated methods (sampling, elicitation) extend the same dispatcher pattern with their own `*Provider` traits — by-method dispatch keeps each spec surface type-checked at the boundary instead of erasing through a generic `Value` handler.

## Naming convention

`McpToolAdapter` exposes MCP-published tools to the agent's `ToolRegistry` as:

```
mcp:<server_name>:<tool_name>
```

Example: `mcp:github:list_issues`. The colon separator follows entelix's general namespacing convention (mirrors `Namespace::render`). `with_unqualified_name` is available for single-server deployments where the qualifier is redundant.

## References

- MCP spec (modelcontextprotocol.io), 2025-03-26 revision (streamable-http transport, Roots)
- Linux Foundation Agentic AI Foundation announcement (2026)
- ADR-0010 — naming taxonomy (`*Client` / `*Manager` / `*Adapter` / `*Provider` suffixes)
- ADR-0017 — `ExecutionContext::tenant_id` (F9 mitigation)
- HANDOFF §2.5 — Phase 4-bis Slice D rationale ("native JSON-RPC, no rmcp")
