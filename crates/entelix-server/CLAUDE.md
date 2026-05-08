# entelix-server

Axum integration. `AgentRouterBuilder` produces an `axum::Router` exposing the canonical agent HTTP surface (`POST /v1/threads/{thread_id}/runs`, `GET /v1/threads/{thread_id}/stream`, `POST /v1/threads/{thread_id}/wake`, `GET /v1/health`) over any `Runnable<S, S>`.

## Surface

- **`AgentRouterBuilder<S>`** — `new(runnable)` / `with_checkpointer(cp)` / `with_tenant_mode(mode)` / `with_tenant_header(name)` / `build() -> BuildResult<axum::Router>`. The router carries SSE streaming on `/stream` for every `StreamMode` the underlying `Runnable` exposes (Values / Updates / Messages / Debug / Events).
- **`TenantMode`** — `#[non_exhaustive]` enum: `Default` (single-tenant — every request runs under `DEFAULT_TENANT_ID`) or `RequiredHeader(HeaderName)` (multi-tenant strict — missing/empty header → `400`). `with_tenant_mode(mode)` is the canonical setter; `with_tenant_header(name)` is the convenience wrapper that constructs `RequiredHeader`.
- **`DEFAULT_TENANT_HEADER`** — `"x-tenant-id"`. Default name passed to `with_tenant_header`.
- **`BuildError` + `BuildResult`** — startup-only failures returned by `AgentRouterBuilder::build`. Today: `InvalidTenantHeader { name }` for header bytes that don't parse as an HTTP header name. Never traverses an HTTP response — operator sees it at process startup.
- **`ServerError` + `ServerResult`** — `#[non_exhaustive]` enum returned by request handlers. Variants: `Core(entelix_core::Error)`, `BadRequest(String)`, `NotFound(String)`, `MissingTenantHeader { header }`. Maps to HTTP status via `IntoResponse` — including `Cancelled → 499`, `DeadlineExceeded → 504`, `Interrupted → 202`.

## Crate-local rules

- **Tenant routing matches the configured `TenantMode`** — `Default` keeps `ExecutionContext::tenant_id()` at `DEFAULT_TENANT_ID`; `RequiredHeader(name)` extracts from the named header and rejects missing / non-UTF-8 / whitespace-only with a typed 4xx (invariant 11). There is no silent fall-through from strict to default.
- **`Error::Interrupted` maps to `202 Accepted`** — graph node requested HITL; the response body carries the `payload` for the operator to drive the resume call. Per.
- **No request-time agent registration** — the runnable is wired at builder time. Replacing it post-build requires a fresh `AgentRouterBuilder` instance. Avoids the F11 race window.
- **SSE backpressure honours `tower_http::limit::ResponseBodyLimitLayer`** — operators wire it explicitly per deployment policy. Server crate doesn't pick a default cap.
- **No filesystem / shell** (invariant 9) — server is a thin axum integration, never exposes its own sandbox.
- **Builder-error vs handler-error split is load-bearing** — `BuildError` flows to the operator at startup (no `IntoResponse`); `ServerError` flows to the HTTP caller (JSON envelope). Adding a startup-only failure mode? extend `BuildError`. Adding a request-time failure? extend `ServerError`. Don't conflate the two.

## Forbidden

- A handler that pulls credentials out of `ExecutionContext` (invariant 10). Auth middleware lives ahead of the agent dispatch; tokens are never embedded in `ctx`.
- A `*Service` type in this crate that does NOT impl `tower::Service` (per  / `scripts/check-naming.sh`).
- Caching agent state on the router struct — invariant 1 (session is event SSoT) requires every request reload from `SessionLog`.
- Adding a new variant to `BuildError` that the request handlers can also reach (it would re-collapse the audience-channel split). If both surfaces need it, the variant belongs on `ServerError`.

## References

- `docs/architecture/managed-agents.md` — Session/Harness/Hand decoupling on the HTTP boundary.
