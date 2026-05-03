# ADR 0017 — `ExecutionContext::tenant_id` is mandatory + type-keyed `Extensions` slot

**Status**: Accepted
**Date**: 2026-04-26 (tenant_id) · 2026-04-30 (Extensions slot)
**Strengthens**: Invariant 11 (Multi-tenant Namespace mandatory) by
extending tenant scoping from the `entelix_memory::Namespace` boundary
to the request-scope `ExecutionContext` carried through every
`Runnable::invoke`, `Tool::execute`, hook, and codec call.

## Context

Phase 1 closed with `ExecutionContext { cancellation, deadline,
thread_id }`. Multi-tenant scoping lived only at the storage boundary
via `Namespace::new(tenant_id, ...)`. Phase 4-bis adds:

- `entelix_persistence::PostgresCheckpointer` whose row keys must
  include tenant scope (otherwise cross-tenant graph state can leak
  via shared thread_ids).
- `entelix_mcp::McpManager` whose connection pool must be keyed by
  `(TenantId, ServerName)` (F9 mitigation — same MCP server,
  different tenant credentials).
- `entelix_cloud::*Transport` whose request hooks may need tenant
  context for billing / cost attribution.

These three subsystems all need tenant scope at the request layer, not
at the storage layer. Plumbing tenant via per-call arguments would
fork the API surface (every persistence / MCP / transport call would
grow a `tenant_id: &str` parameter) and silently break composition
when one layer forgets to propagate it.

## Decision

`ExecutionContext` gains a mandatory `tenant_id: String` field with a
build-time default of `"default"` (exposed as
`entelix_core::DEFAULT_TENANT_ID`). Single-tenant deployments are
unaffected; multi-tenant operators set the field per request via
`ExecutionContext::with_tenant_id(...)` and every downstream
subsystem reads `ctx.tenant_id()` for scoping.

```rust
let ctx = ExecutionContext::new()
    .with_tenant_id("acme-corp")
    .with_thread_id("conv-42");
agent.invoke(input, &ctx).await?;
```

`with_cancellation(token)` (which builds a context from an externally
provided token) deliberately resets tenant to the default — the
parent's tenant should never be silently inherited across an explicit
cancellation-token handoff. Callers that intend to share tenant scope
chain `.with_tenant_id(parent.tenant_id())` explicitly.

## Why this is invariant 11 strengthening

Invariant 11 originally read "Multi-tenant Namespace mandatory —
`entelix_memory::Namespace` requires a `tenant_id`. Cross-tenant data
leak is structurally impossible by API design." That guarantee held
at the storage boundary but not at the request boundary: a hook,
tool, or codec receiving an `ExecutionContext` had no tenant
information to act on.

After this ADR the invariant reads (effectively): **tenant scope is
present in every request-scope context, not just at the storage
boundary.** Subsystems that key on tenant (persistence, MCP pool,
cost meter) read directly from `ExecutionContext::tenant_id()`
instead of forcing callers to plumb a separate `&str`.

## Why a string + default rather than a typed `TenantId` newtype

Considered but rejected: `pub struct TenantId(String);`. Reasons:

- A `TenantId(String)` newtype would force every existing call site
  that constructs `ExecutionContext` to import the newtype and wrap
  the string. The migration cost is real and the type-safety win is
  marginal — `tenant_id: String` is already named so confusion with
  other strings is unlikely.
- The default-tenant case (`"default"`) is a string. Wrapping it in
  a newtype just to unwrap it everywhere it's compared is friction.
- Phase 5 may introduce a richer `TenantContext` (with quota /
  cost-meter handles); at that point a typed wrapper makes sense as
  a separate field. The string `tenant_id` then becomes
  `TenantContext::id()`. Extension is forward-compatible.

## Consequences

- All `ExecutionContext` constructors set `tenant_id` to
  `DEFAULT_TENANT_ID`. Existing call sites need no changes.
- `entelix_memory::Namespace::new(tenant_id, ...)` still owns
  tenant scope at the storage boundary. The two scopes are linked
  but not equated — code that translates a request into a storage
  call typically does
  `Namespace::new(ctx.tenant_id(), "session_log").with_scope(...)`.
- F9 mitigation (`McpManager::pool` keyed by
  `(TenantId, ServerName)`) becomes compile-time enforceable
  because the `&ExecutionContext` argument always carries a
  tenant.

## Addendum (2026-04-30) — Type-keyed `Extensions` slot

`ExecutionContext` first-class fields cover what *the SDK* reasons about
(`tenant_id`, `cancellation`, `deadline`, `run_id`, `thread_id`).
Operators routinely need to thread *their own* request-scope state — a
workspace handle, a per-run cache, a tenant-specific rate-limiter, a
custom telemetry tag — through the same context the rest of the SDK
already carries. Without a typed slot, operators reach for
`thread_local!` (broken under tokio task migration) or bespoke
parameter plumbing (forks every Runnable / Tool signature).

`ExecutionContext` now carries a typed-keyed `Extensions` (in
`entelix_core::extensions`), mirroring the well-trodden
`http::Extensions` / `axum::Extensions` / `tower::Service` request-
extensions pattern. Entries are stored as `Arc<dyn Any + Send + Sync>`
keyed by `TypeId`, one entry per type.

```rust
let ctx = ExecutionContext::new()
    .with_tenant_id("acme-corp")
    .add_extension(Workspace::open("./repo"));

// Inside a tool / hook / codec:
if let Some(ws) = ctx.extension::<Workspace>() {
    // Arc<Workspace> — caller can hold across awaits without
    // pinning the ExecutionContext alive.
}
```

### Design properties

- **Copy-on-write semantics** — `add_extension` returns a *new*
  `ExecutionContext` with a fresh `Arc<Extensions>`; the original
  context is unchanged. Combinators that fan out to parallel branches
  see consistent, non-mutating state. Cloning is cheap (Arc bump).
- **Send + Sync by construction** — entries are constrained
  `T: Send + Sync + 'static`; multi-tenant deployments can clone the
  context across parallel branches without lock juggling.
- **`get<T>() -> Option<Arc<T>>`** — the returned `Arc<T>` is
  independent of the carrier; safe to hold across awaits or after the
  owning `ExecutionContext` is dropped.

### Invariant 10 alignment (no tokens in tools)

`Extensions` is **not** a credentials channel. Operators who stash a
`CredentialProvider` handle here would surface the credential value to
every `Tool::execute` site that touches the context — a structural
violation. The trait surface for this slot deliberately does not
advertise an "auth" affordance; credentials live exclusively in
transports per Invariant 10. The module's rustdoc carries this
constraint as load-bearing prose so operators reading the type
discover it before reaching for it.

### Why a type-keyed `Any` map rather than typed fields

Considered: `pub struct UserExtensions { ... }` with nominal fields the
operator subclasses. Rejected because:

- Field-named extensions force every `entelix-core` revision to know
  the operator's slots — back to bespoke parameter plumbing.
- The `http::Extensions` shape is the de-facto industry standard for
  request-scope cross-cutting state in Rust. Reusing the shape lets
  operators import their existing intuition.
- `TypeId`-keyed lookup is `O(1)` and zero-allocation on the read path.
  Insert is `O(N)` (clone the underlying `HashMap`) but happens once at
  context construction, not per-tool-call.

## References

- Invariant 11 (CLAUDE.md)
- Invariant 10 (CLAUDE.md) — the Extensions slot deliberately
  excludes credentials
- F2 / F9 (PLAN.md logical flaw register)
  the persistence and MCP slices)
- `http::Extensions` (the prior art the addendum mirrors)
