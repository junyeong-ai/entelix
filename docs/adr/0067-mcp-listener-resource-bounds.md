# ADR 0067 — MCP listener resource bounds

**Status**: Accepted
**Date**: 2026-05-02
**Decision**: `HttpMcpClient`'s SSE listener and POST-response decoder enforce two operator-tunable resource bounds — a per-frame byte cap and an in-flight server-initiated dispatch concurrency cap — both with deterministic failure semantics that prefer to drop traffic over pinning client memory.

## Context

The streamable-HTTP MCP transport (MCP 2025-03-26) lets the server initiate three classes of traffic over a single client-opened SSE stream:

- `roots/list` — solicit the client's filesystem roots.
- `elicitation/create` — solicit operator confirmation through a typed channel.
- `sampling/createMessage` — request the client run a chat completion on its own model.

Pre-1.0 the client did this safely *enough*: each inbound server request was demuxed off the SSE byte stream (`Vec<u8>` accumulator), parsed into a `JsonRpcServerRequest`, and dispatched on a fresh `tokio::spawn`. There were two unbounded growth surfaces:

1. **Frame buffer** — `consume_sse_response` (POST-response SSE) and `spawn_listener` (background SSE) both did `buf.extend_from_slice(&chunk)` until they found a `\n\n` terminator. The MCP spec puts no upper bound on a single SSE frame; an adversarial server can stream bytes forever without the terminator and pin client heap.
2. **Dispatch fan-out** — every parsed `JsonRpcServerRequest` was unconditionally `tokio::spawn`ed. An adversarial or malfunctioning server can flood server-initiated requests faster than the client completes them, and each pending dispatch holds a future in the executor heap. With no cap, peak concurrent dispatches grows with the server's send rate — another open OOM vector.

Both vectors share a property: legitimate MCP traffic does not approach either limit. JSON-RPC frames are typically kilobytes; in-flight server-initiated dispatches per client are typically zero or one. So the right bound is one that's invisible to legitimate use and decisive when crossed.

## Decision

`McpServerConfig` carries two new operator knobs, both with sensible defaults:

| Knob | Default | Override |
|---|---|---|
| `max_frame_bytes` | `DEFAULT_MAX_FRAME_BYTES = 1 << 20` (1 MiB) | `with_max_frame_bytes(n)` |
| `listener_concurrency` | `DEFAULT_LISTENER_CONCURRENCY = 32` | `with_listener_concurrency(n)` |

**Frame cap enforcement** — both `consume_sse_response` and the background listener check `buf.len() > cap` after every `bytes_stream().next()`. Crossing the threshold is treated as malicious or broken and the connection is closed:

- `consume_sse_response` returns `McpError::ResourceBounded { kind: ResourceBoundKind::FrameSize, message }`. The variant is semantically distinct from `McpError::MalformedResponse` — "vendor sent garbage" (malformed) vs. "vendor exceeded an SDK-shipped DoS guard" (resource-bounded). Operator alerting can split on the variant. The `From<McpError> for Error` mapping lands both on `Error::Provider::Network` (502 at the HTTP boundary) but the resource-bounded path carries a hint pointing at `with_max_frame_bytes` for legitimate large-payload operators.
- The background listener `tracing::warn!`s and returns from the `tokio::spawn` task — the listener `JoinHandle` resolves, and the next `ensure_listener` call respawns. No unbounded retry; a hostile server reconnecting still hits the cap on the new connection.

**Concurrency cap enforcement** — `HttpMcpClient` holds an `Arc<Semaphore>` sized at `config.listener_concurrency()`. Every server-initiated dispatch (POST-response SSE *and* background SSE) routes through `spawn_bounded_dispatch`, which `try_acquire_owned`s a permit. On saturation:

- The request is **dropped** (not queued) with a `tracing::warn!` carrying the method name.
- The drop is one-way — the server is expected to retry on its own cadence (every documented MCP server-initiated method is idempotent or has retry semantics).

The permit is moved into the spawned future and released when the dispatch completes (success, error, or cancellation), so completion always restores capacity.

### Why drop instead of queue

A queue (e.g. `mpsc::channel(N)`) would create a second open OOM vector — the queue itself can grow under flood. `try_acquire_owned + drop` keeps the cap as a hard ceiling on outstanding work rather than a soft buffer.

### Why `Semaphore` instead of `JoinSet`

`tokio::task::JoinSet` would also bound concurrency, but only by *waiting* on the producer. Our producer is the SSE listener loop — back-pressuring it would stall the connection rather than drop the request. A `Semaphore` lets the producer continue serving the byte stream while excess server-initiated dispatches are silently dropped at the gate.

### Why per-client (not per-server-method)

A method-aware quota (e.g. "max 10 in-flight `roots/list`") encodes too much policy at the transport. Operators that want per-method shaping can wire a `Layer` over `McpToolAdapter` or implement their own provider that internally rate-limits. The transport-level cap is a backstop, not a policy.

### Why these defaults

- **1 MiB max frame** — JSON-RPC frames typically run 100B–10KB. 1 MiB has 100×–10000× headroom for legitimate traffic and is small enough that even a 1024-fold flood (1 GiB attacker payload) gets caught before the OS OOM-kills the client. Operators sending large structured outputs through `sampling/createMessage` raise the cap explicitly.
- **32 concurrent dispatches** — chosen as a guard, not a measured ceiling. The SDK does not ship with field telemetry from a published MCP-server fleet, so the bound is set conservatively-high: an order of magnitude above the realistic single-digit concurrency for typical `roots/list` / `elicitation/create` patterns, low enough that 32 outstanding dispatches × the per-dispatch task-future size still fits comfortably in the executor's working set even on small VMs. Operators who run sampling-heavy synthesis workloads where the server fans out batched `sampling/createMessage` calls should raise the cap and (separately) attach a `tower::Layer` over the MCP-tool dispatch path for per-method rate shaping. If field telemetry from production deployments later surfaces a different realistic ceiling, this default revisits in a follow-up ADR.

## Consequences

- New `pub const` constants `DEFAULT_MAX_FRAME_BYTES` and `DEFAULT_LISTENER_CONCURRENCY` in `entelix_mcp::server_config`, re-exported by `entelix_mcp` and the `entelix` facade with the `MCP_*` prefix.
- New builder methods `McpServerConfig::with_max_frame_bytes(n)` and `with_listener_concurrency(n)`, both `const`, both verb-prefix-conformant per ADR-0010.
- New module-private helper `spawn_bounded_dispatch` collapses the two prior `tokio::spawn(handle_server_request)` call sites into one bounded path. Both POST-response SSE (`consume_sse_response → spawn_handle_server_request`) and background SSE (`spawn_listener` inline spawn) now share the cap.
- Two regression suites:
  - `crates/entelix-mcp/src/client.rs::tests` (inline `#[cfg(test)]`) — semaphore saturation drop and permit-return-after-completion.
  - `crates/entelix-mcp/tests/hostile_server_caps.rs` — wiremock-served SSE that exceeds the configured cap; companion happy-path test that the cap doesn't false-trip on legitimate frames.
- Operator-facing failure mode is decisive: oversize frame → `McpError::MalformedResponse` with a diagnostic; dispatch flood → `tracing::warn!` per dropped request. No silent buffer growth, no crash.

## References

- ADR-0004 — native MCP client (no `rmcp`), MCP 1.5 streamable-http (the listener architecture this ADR hardens).
- ADR-0017 — `tenant_id` mandatory + `Extensions` slot (the `(tenant_id, server_name)` pool key that scopes a client instance).
- ADR-0034 — heuristic externalisation. The two caps are vendor-policy defaults but exposed on the typed `McpServerConfig` surface, not buried in dispatch hot paths — operators override declaratively.
- `crates/entelix-mcp/src/server_config.rs` — the cap constants and `with_*` setters.
- `crates/entelix-mcp/src/client.rs` — `consume_sse_response` (POST-response cap), `spawn_listener` (background cap), `spawn_bounded_dispatch` (concurrency gate).
- `crates/entelix-mcp/tests/hostile_server_caps.rs` — frame-cap regression suite.
