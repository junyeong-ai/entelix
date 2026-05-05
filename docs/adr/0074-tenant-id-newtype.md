# ADR 0074 — `TenantId` newtype + validating constructors

**Status**: Accepted
**Date**: 2026-05-05
**Decision**: Cross-tenant data leak becomes structurally impossible — every tenant-bearing field on `ExecutionContext`, `ThreadKey`, `Namespace`, `NamespacePrefix`, and `Checkpoint` carries an `entelix_core::TenantId` newtype whose only constructors validate non-emptiness, whose `serde::Deserialize` impl runs the same validator, and whose `Namespace::parse` / persistence-layer row hydration paths surface empty tenant input as `Error::InvalidRequest` rather than building a tenantless instance.

## Context

Invariant 11 (CLAUDE.md §"Architecture invariants — Security") demands cross-tenant data leak be structurally impossible. Up to and including 1.0-RC.1 the enforcement was a layered runtime guard:

- `Namespace::new("")` and `NamespacePrefix::new("")` carried a runtime `assert!` (slice 64 mitigation) so a hand-crafted empty-tenant `Namespace` panicked at construction.
- `ThreadKey::new("", _)` carried the same assertion.
- `ExecutionContext::with_tenant_id(impl Into<String>)` accepted any string, including the empty one — the boundary lived elsewhere.

Three holes survived the runtime regime:

1. **`Namespace::parse(":scope")` silent empty-tenant** — the audit-channel inverse of `render` split a leading colon into an empty first segment, then constructed `Namespace { tenant_id: "", scope: ["scope"] }` without re-validating. `assert!` in `new` did not fire, because `parse` did not call `new`. Rendered keys persisted under one tenant, replayed under "no tenant", silently mis-routed at the row-level filter.
2. **`serde::Deserialize` on tenantless wire payload** — `Namespace`'s derived `Deserialize` (and the same on `ThreadKey` / `Checkpoint`) hydrated an empty `tenant_id: String` field straight from JSON. A wire payload with `{"tenant_id": "", "scope": [...]}` produced a tenantless instance whose downstream rendering then collapsed every tenant onto the same key prefix.
3. **Persistence-layer row hydration** — `PostgresCheckpointer::CheckpointRow.tenant_id: String` and `RedisCheckpointer::unwrap_envelope` decoded the column / JSON field directly into `String`. A row written under one tenant_id but later mutated to empty (e.g. by a misconfigured admin script) surfaced as a tenantless `Checkpoint` whose subsequent RLS-filter comparison ran against `''` and silently widened the result set.

`assert!` is the wrong tool for these holes — every one of them is at a *deserialisation* boundary the assertion never observes. The invariant was correct; the enforcement layer was one tier too high.

The 6-fork audit (2026-05-05) cross-confirmed three independent observations:

- pydantic-ai 1.90, LangGraph 1.0 GA, OpenAI Agents SDK, Claude Agent SDK, Vercel AI SDK 5, rig — **none** ship `TenantId` as a typed primitive. LangGraph leaks tenant via thread_id convention. pydantic-ai leaves it to application code. Multi-tenant typed enforcement at the SDK level is industry-leading, not industry-standard.
- The OTel GenAI semconv 0.32 attribute registry has no `gen_ai.tenant_id`. The shape is invented at the SDK layer.
- Anthropic managed-agents shape and Postgres RLS pattern both presume a tenant exists at every sub-agent and row boundary; "tenant by convention" is the standard enforcement gap.

The audit also clarified that runtime `assert!` on a constructor leaves the *deserialisation* boundary unguarded by construction. Newtype + validating `try_from` + delegating `serde::Deserialize` is the standard fix; Rust idiom does not allow `From<&str>` for a partial type, so the validator is reachable through `TryFrom` (and through `Deserialize`'s blanket "deserialize String, then try_from").

## Decision

### `TenantId` newtype is the single tenant-bearing surface

`entelix-core/src/tenant_id.rs` introduces:

```rust
#[derive(Clone, Eq, Hash, PartialEq)]
pub struct TenantId(Arc<str>);
```

- `Arc<str>` backing — cloning is an atomic refcount bump, not an allocation. Same hot-path optimisation `ExecutionContext` previously held inline.
- `Default::default()` returns a process-shared `TenantId` for `DEFAULT_TENANT_ID` (`"default"`) via `OnceLock`. Single-tenant deployments allocate zero strings.
- `TryFrom<&str>` and `TryFrom<String>` reject empty input with `Error::invalid_request("tenant_id must be non-empty (invariant 11)")` — the **only** validating path that production-untrusted input takes.
- `serde::Deserialize` deserialises `String`, then routes through `TryFrom`. A tenantless wire payload (`""`) surfaces through `serde::de::Error::custom` carrying the same message — wire-payload empty input cannot construct a `TenantId`.
- `serde::Serialize` is transparent (emits the underlying string). Round-trip property: `serde_json::from_str(&serde_json::to_string(&t))? == t`.
- `TenantId::new(impl AsRef<str>) -> Self` panics on empty — programmer-error grade for test fixtures and migration tooling that have already validated the inputs. Production paths take a `TenantId` argument and inherit the validation from whoever built it.
- `Display`, `AsRef<str>`, `Borrow<str>` for read-side ergonomics. The `Borrow<str>` impl lets a `HashMap<TenantId, V>` be looked up by `&str` (used by the `CostMeter` ledger to avoid an extra Arc bump per spend lookup).

### Five carriers carry the typed field, not `String`

| Carrier | Crate | Field |
|---|---|---|
| `ExecutionContext::tenant_id` | `entelix-core` | `TenantId` |
| `ThreadKey::tenant_id` | `entelix-core` | `TenantId` |
| `Namespace::tenant_id` | `entelix-memory` | `TenantId` |
| `NamespacePrefix::tenant_id` | `entelix-memory` | `TenantId` |
| `Checkpoint::tenant_id` | `entelix-graph` | `TenantId` |

Every constructor that previously accepted `impl Into<String>` now accepts `TenantId` directly. Callers either own one already (extracted from `ExecutionContext::tenant_id()`, which is the canonical production path) or build one via `TenantId::new` (test) / `TenantId::try_from` (production input).

### `Namespace::parse` validates the tenant segment

The first segment of a rendered key is constructed through `TenantId::try_from(segments.remove(0))?`. A leading-colon input (`":scope"`) — which would previously have produced `Namespace { tenant_id: "", … }` — surfaces as `Error::InvalidRequest`. The audit-channel use case (`GraphEvent::MemoryRecall::namespace_key` round-trip) gains structural defence against operator-replay bugs that previously could have routed events under "no tenant".

### Persistence-layer row hydration runs the validator

`PostgresCheckpointer::CheckpointRow::try_into_checkpoint` and `RedisCheckpointer::unwrap_envelope` route the persisted `tenant_id` string through `TenantId::try_from`. Empty values surface as `PersistenceError::Backend("invalid persisted tenant_id: …")` rather than constructing a tenantless `Checkpoint`. The contract: *no path from the database to a `Checkpoint` instance bypasses the validator.*

### Multi-tenant boundary functions accept the typed parameter

Six functions whose contract was "tenant scope" previously took `&str`:

| Function | Crate | New signature |
|---|---|---|
| `PolicyRegistry::policy_for` | `entelix-policy` | `&TenantId` |
| `CostMeter::charge` / `spent_by` / `drain` | `entelix-policy` | `&TenantId` |
| `QuotaLimiter::check_pre_request` | `entelix-policy` | `&TenantId` |
| `McpManager::client_for` (internal) | `entelix-mcp` | `&entelix_core::TenantId` |
| `set_tenant_session` (3 backends) | `entelix-persistence`, `entelix-graphmemory-pg`, `entelix-memory-pgvector` | `&TenantId` |
| `ExecutionContext::with_tenant_id` | `entelix-core` | `TenantId` |

`set_tenant_session` is the load-bearing one — it is the helper that arms the Postgres RLS policy via `set_config('entelix.tenant_id', $1, true)`. The previous `&str` signature meant any caller could arm the policy with the empty string and silently see *every* row (RLS treats `''` as "unknown tenant"). The typed signature makes that bug structurally impossible.

### `with_tenant_id` is type-strict, not fallible-builder

The decision is to keep `ExecutionContext::with_tenant_id(TenantId) -> Self` (infallible builder, consume self, return self) rather than splitting into `try_with_tenant_id(impl AsRef<str>) -> Result<Self>`. Rationale:

- `with_*` is the configuration-setter naming taxonomy slot (ADR-0010 §"Builder verb-prefix exception"); fallible variants are `try_*`. Mixing the two on the same axis would produce two parallel surfaces, exactly the LangGraph anti-pattern (`create_react_agent` deprecated alongside `create_agent`) that invariant 14 forbids.
- Production untrusted input has *one* fallible boundary: `TenantId::try_from`. Once across that boundary, downstream code has a typed value and the rest of the chain stays infallible. This matches Rust idiom — `Url::parse` is fallible, every method on the resulting `Url` is infallible.
- The verbose test-side ergonomics (`with_tenant_id(TenantId::new("acme"))`) are a feature, not a bug — they make the type-system boundary visible at every call site, which is exactly what invariant 11 wants surface readers to see.
- An `impl Into<TenantId>` overload would require a `From<&str> for TenantId`, which is unsound — `From` advertises infallibility but the validator panics on empty input. Rust idiom forbids panicking `From` impls.

## Consequences

**Positive**:
- Three previously-silent holes (`parse`, serde, row hydration) are now typed `Error::InvalidRequest` paths. Empty-tenant input cannot construct an instance.
- The CostMeter ledger and policy registry can use `TenantId::Borrow<str>` for `&str` lookup against an existing `HashMap<TenantId, V>` — no extra allocation on the hot path.
- `Arc<str>` backing means cloning a `TenantId` (done implicitly per tool dispatch and per sub-agent context fan-out) is a refcount bump, identical to the previous `Arc<str>` field optimisation on `ExecutionContext`.
- Every `set_tenant_session` call site is now structurally guaranteed to arm the Postgres RLS policy with a non-empty tenant — RLS-bypass via empty session variable becomes impossible.
- `Namespace::parse` round-trip gains a structural defence: `parse(render(ns)) == Ok(ns)` for every well-formed namespace, and *only* for well-formed ones. The audit-channel replay path is now safe.
- Public-API surface gains one new type (`TenantId`); five carrier types' accessor signature shifts from `tenant_id() -> &str` to `tenant_id() -> &TenantId`. This is a deliberate breaking change at the 1.0 RC contract boundary (ADR-0064 release charter authorises such changes pre-GA).

**Negative**:
- Test fixtures that previously wrote `Namespace::new("acme")` now write `Namespace::new(TenantId::new("acme"))`. The verbosity is intentional (see Decision §"`with_tenant_id` is type-strict") — a global rewrite landed in the same commit per invariant 14.
- One additional public type (`TenantId`) in `entelix-core`'s surface. Surface line count drift recorded in `docs/public-api/entelix-core.txt`; ADR-0064 §"Public-API baseline contract" covers this kind of typed-strengthening drift.
- Cross-tenant maintenance operations (TTL sweepers running under a `BYPASSRLS` role) now have to materialise a `TenantId` to call `set_tenant_session` on the per-tenant queries they issue. The pattern is documented in `entelix-persistence/src/postgres/tenant.rs` rustdoc; the friction is acceptable given invariant 11.

**Migration outcome (one-shot, no shim)**: All five carrier types, six boundary-function signatures, and ~35 test/example call sites land in the same commit. There is no `pub use OldName as NewName`, no `#[deprecated]`, no transition period — invariant 14 forbids it. The previous `Namespace::new(impl Into<String>)` runtime-`assert!` surface is deleted, not deprecated.

## References

- CLAUDE.md invariant 11 — multi-tenant Namespace mandatory (strengthened text)
- CLAUDE.md invariant 14 — no backwards-compatibility shims
- ADR-0017 — `tenant_id` mandatory on `ExecutionContext` + `Extensions` slot
- ADR-0040 — `list_namespaces` parse round-trip contract (validator now closes parse-side hole)
- ADR-0041 — Postgres row-level security (`set_tenant_session` is the policy-arming helper that now takes `&TenantId`)
- ADR-0043 — graphmemory-pg RLS extension
- ADR-0044 — pgvector RLS extension
- ADR-0064 — 1.0 release charter (post-RC typed-strengthening authorised)
- F2 mitigation — original namespace tenant_id mandatory introduction
- `crates/entelix-core/src/tenant_id.rs` — newtype implementation + serde + tests
- `crates/entelix-memory/src/namespace.rs` — `parse()` validating reconstruction + regression tests
- `crates/entelix-persistence/src/postgres/checkpointer.rs` — `CheckpointRow::try_into_checkpoint` validating hydration
- `crates/entelix-persistence/src/redis/checkpointer.rs` — `unwrap_envelope` validating hydration
