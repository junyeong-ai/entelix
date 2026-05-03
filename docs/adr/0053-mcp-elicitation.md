# ADR 0053 — MCP `elicitation/create` server-initiated channel

**Status**: Accepted
**Date**: 2026-05-01
**Decision**: Phase 8 of the post-7-차원-audit roadmap (first sub-slice)

## Context

ADR-0004 (the rmcp-rejection / native-JSON-RPC ADR) noted
sampling and elicitation as future MCP work, deferring them
behind the Roots slice (ADR-0011 amendment / 10E). The Roots
slice (2026-04-29) shipped:

- A `RootsProvider` trait answering `roots/list` from the
  client side.
- Server-initiated request dispatcher in `HttpMcpClient` with
  a single match arm on `request.method`.
- `ClientCapabilities::roots` advertised iff a provider is
  wired.
- Capability propagation through `McpServerConfig::with_roots_provider`.

The dispatcher pattern was deliberately one-arm-per-method so
sampling and elicitation could land cleanly when their time
came. The CLAUDE.md for `entelix-mcp` makes this explicit:

> Server-initiated dispatcher: per-method `*Provider` trait
> pattern (Roots today; sampling / elicitation will add their
> own providers, not new methods on `McpClient`).

This slice is the elicitation half. Sampling will follow as a
separate slice — its `*Provider` shape is more complex
(model preferences, system prompt, sampling parameters) and
warrants its own design space.

## What is elicitation?

MCP 2025-03 §"Elicitation" lets a server ask the client (the
agent harness) for typed input mid-session. Typical uses:

- **Missing config**: server needs an API key the operator
  hasn't surfaced yet.
- **Confirmation**: server wants explicit consent before a
  destructive action.
- **Disambiguation**: server can't pick between options; asks
  the operator (or auto-policy) to choose.

The server sends `elicitation/create` carrying a
human-readable `message` and a JSON Schema describing the
expected response shape. The client returns one of three
explicit actions:

- **accept** — operator approved, payload conforms to schema.
- **decline** — operator explicitly refused; server SHOULD
  respect and not retry.
- **cancel** — operator dismissed without answering; server
  MAY retry under different conditions.

## Decision

Add an `ElicitationProvider` trait + `ElicitationRequest` /
`ElicitationResponse` types in a new `elicitation.rs` module.
Wire it into the dispatcher with a new arm; advertise the
capability iff a provider is configured. Mirror the Roots
slice exactly so the dispatcher pattern stays uniform.

### Public surface

```rust
pub struct ElicitationRequest {
    pub message: String,
    pub requested_schema: serde_json::Value,
}

#[non_exhaustive]
pub enum ElicitationResponse {
    Accept(serde_json::Value),
    Decline,
    Cancel,
}

#[async_trait]
pub trait ElicitationProvider: Send + Sync + 'static + Debug {
    async fn elicit(&self, request: ElicitationRequest) -> McpResult<ElicitationResponse>;
}

pub struct StaticElicitationProvider { /* accept(value) / decline() / cancel() */ }
```

### Why an `Action` enum, not `Result<Option<Value>>`

The spec distinguishes three outcomes — accept, decline,
cancel — each with operator-meaningful semantics:

- accept → server has the data it needed
- decline → server should NOT retry (operator policy)
- cancel → server MAY retry (operator dismissed)

Collapsing decline + cancel into `Ok(None)` would lose the
"should retry?" signal. `Result<Option<Value>>` is the wrong
shape for a tri-state response.

The trait returns `Result<ElicitationResponse>` — `Result` for
provider failures (transport / cache lookup / etc.), the enum
for the spec's intentional outcome cases.

### Wire serialization

`ElicitationResponse` carries a custom `Serialize` impl
because the spec mandates a specific JSON shape:

- `Accept(content)` → `{"action": "accept", "content": <content>}`
- `Decline` → `{"action": "decline"}` (no `content` field)
- `Cancel` → `{"action": "cancel"}`

The default derive would produce internally-tagged or
externally-tagged shapes that don't match. Custom impl keeps
the wire bytes exact.

### Why a trait, not a closure

Real elicitation handlers reach across the agent's execution
environment — CLI prompt, UI form, stored cache, declining
policy, audit-log emit. Each wants its own state. A closure
shape (`Fn(...)`) would force every operator into
`Arc<Mutex>`-wrapping by hand. The trait's `Debug` requirement
also helps `McpServerConfig`'s `Debug` derivation surface
provider presence in operator logs.

### `#[non_exhaustive]` on the response enum

Future MCP spec revisions may add more action variants (e.g.,
"defer to a later turn"). Marking the enum non-exhaustive lets
us accept new variants without breaking caller pattern matches
(`match` requires `_ =>` arm). Caught by
`scripts/check-surface-hygiene.sh`.

### No `ExecutionContext` parameter

Mirrors the Roots design (ADR-0004 amendment). Server-initiated
requests arrive on a background SSE listener, not in the middle
of a client-driven call. Threading an `ExecutionContext`
through the listener would force the listener to invent one
(which request's context?) — that choice has no honest answer.
The signature stays context-free.

### Capability advertisement

`ClientCapabilities` gains an `elicitation: Option<ElicitationCapability>`
field. The capability struct is empty (`{}`) — the spec defines
no sub-fields, presence alone signals support. Advertised iff
the operator wired an `ElicitationProvider` via
`McpServerConfig::with_elicitation_provider`.

Servers respect the advertisement: an MCP server that doesn't
see `elicitation` in the client's capabilities shouldn't issue
`elicitation/create`. If a non-conforming server does, the
dispatcher returns JSON-RPC `-32601` "Method not found".

### Tests

- 6 unit tests in `elicitation.rs`: wire-shape serialization
  for accept / decline / cancel; deserialization of the
  request shape; static provider behaviour for accept /
  decline / cancel.
- 4 wiremock e2e tests in `streamable_elicitation_e2e.rs`:
  - Server-initiated `elicitation/create` dispatches to a
    static accept provider; response carries `{action: accept,
    content: ...}`.
  - Decline provider serializes action only (no `content`).
  - No provider wired → JSON-RPC `-32601` Method not found.
  - Capability advertised iff provider wired.

## Consequences

✅ MCP 1.5 surface coverage extends from Tools / Resources /
Prompts / Completion / Roots to also include Elicitation.
Sampling remains the last server-initiated channel.
✅ The dispatcher pattern stays uniform — one arm per method,
each arm gates on a `*Provider` from `McpServerConfig`. New
server-initiated methods follow the same template.
✅ Default impls (no `notify_elicitation_changed` trait method
needed because elicitation is per-event, not list-style)
shield existing `McpClient` implementors (mocks, custom
impls) from breaking.
✅ Three-action enum preserves the spec's "decline vs cancel"
distinction the operator's policy may care about.
✅ `#[non_exhaustive]` on `ElicitationResponse` keeps future
MCP spec evolution non-breaking for callers.
❌ Custom `Serialize` impl on `ElicitationResponse` — operators
deriving their own shape would conflict. Doc explicitly notes
the wire bytes are spec-mandated.
❌ Trait surface grew. The `*Provider` per-method pattern is
the explicit choice — adding a method to `McpClient` instead
would couple every implementor (including mocks) to every new
server-initiated method.

## Alternatives considered

1. **Single trait method on `McpClient` instead of a
   `*Provider`** — every mock / custom client would have to
   implement (or default-no-op) every new server-initiated
   method as the spec evolves. The `*Provider` pattern keeps
   `McpClient` stable and lets each method's handler be
   wired independently. Rejected; CLAUDE.md called this out
   explicitly.
2. **`Result<Option<Value>>` instead of three-variant enum**
   — collapses decline + cancel into `Ok(None)`, losing the
   server-side retry signal. Rejected.
3. **Auto-derive `Serialize` with a tag** — produces shapes
   like `{"Accept": {...}}` (externally tagged) or
   `{"variant": "Accept", "content": {...}}` (internally
   tagged). Neither matches the spec's `{action, content?}`.
   Custom impl is the right tool.
4. **Validate response content against `requested_schema`
   inside the trait** — the SDK doesn't bundle a JSON Schema
   engine; operators choose `jsonschema` / `boon` / etc. via
   their own wiring. The trait surface keeps validation
   operator-side; doc string recommends it for auto-respond
   providers.
5. **Ship sampling in the same slice** — sampling has more
   surface area (model preferences, system prompts, sampling
   parameters, usage tracking). Sliced separately so the
   design space gets dedicated attention. Elicitation is the
   smaller, cleaner half.

## Operator usage patterns

**Auto-confirm a known prompt**:
```rust
let provider = Arc::new(StaticElicitationProvider::accept(
    json!({"confirmed": true})
));
let config = McpServerConfig::http("server", url)?
    .with_elicitation_provider(provider);
```

**Auto-decline by policy**:
```rust
let provider = Arc::new(StaticElicitationProvider::decline());
let config = McpServerConfig::http("server", url)?
    .with_elicitation_provider(provider);
```

**Custom provider** (CLI prompt, UI form, etc.):
```rust
#[derive(Debug)]
struct CliElicitProvider { /* state */ }

#[async_trait]
impl ElicitationProvider for CliElicitProvider {
    async fn elicit(&self, req: ElicitationRequest) -> McpResult<ElicitationResponse> {
        println!("MCP server asks: {}", req.message);
        // gather input from stdin / UI / ...
        Ok(ElicitationResponse::Accept(json!({...})))
    }
}
```

## References

- ADR-0004 — native JSON-RPC client (parent decision tree).
- ADR-0011 amendment — Roots slice (the dispatcher pattern
  this slice extends).
- 7-차원 roadmap §S9 — Phase 8 (MCP + OTel completion).
- MCP 2025-03 spec §"Elicitation" — the wire-format source.
- `crates/entelix-mcp/src/elicitation.rs` — trait + types +
  static provider + 6 unit tests.
- `crates/entelix-mcp/src/protocol.rs` —
  `ElicitationCapability` shape + `JsonRpcServerRequest::params`
  no longer `#[expect(dead_code)]`.
- `crates/entelix-mcp/src/server_config.rs` —
  `with_elicitation_provider` + accessor.
- `crates/entelix-mcp/src/client.rs` — capability
  advertisement + dispatcher arm.
- `crates/entelix-mcp/tests/streamable_elicitation_e2e.rs` —
  4 wiremock e2e regressions.
