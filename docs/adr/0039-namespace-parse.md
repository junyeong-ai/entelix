# ADR 0039 — `Namespace::parse` — audit-key reversibility

**Status**: Accepted
**Date**: 2026-05-01
**Decision**: Phase 7 of the post-7-차원-audit roadmap (second sub-slice)

## Context

`Namespace::render() -> String` flattens a `Namespace` to a
`:`-separated key with `\:` / `\\` escapes so backends that take a
single string per row (Postgres `memory_items.namespace`, Redis
keys, audit-log fields) can store the typed scope without
reinventing collision-free encoding. The escape rules guarantee
two distinct namespaces never collide.

`render` was one-way. Audit-channel consumers (introduced in
ADR-0037 — invariant #18) and operator dashboards reading
`GraphEvent::MemoryRecall::namespace_key` had to either:

1. Re-derive the typed scope by ad-hoc string splitting on `:`,
   which silently corrupted segments that legitimately contained
   `\:` (the escape was invisible to a naive `split(':')` reader),
   or
2. Treat the rendered key as opaque and lose the tenant boundary
   downstream — defeating the whole point of having a structured
   identifier in the first place.

Both paths exist in the wild for similar SDKs. Both are quietly
wrong: case (1) produces a *different* tenant_id under hand-edited
keys, case (2) makes operator dashboards unable to filter by
tenant. The fix is to ship the inverse on the SDK surface so
consumers never reinvent it.

## Decision

Add `Namespace::parse(rendered: &str) -> Result<Namespace>` to
`entelix-memory`. Same module as `render`, mirror semantics.

```rust
impl Namespace {
    pub fn parse(rendered: &str) -> Result<Self> { /* ... */ }
}
```

Returns `entelix_core::Result<Namespace>` so the public-API
convention (no `anyhow`, no `*Error` leaking from public) holds.
Failures surface as `Error::InvalidRequest`:

- `Namespace::parse("acme\\")` → trailing backslash (incomplete escape).
- `Namespace::parse("acme\\x")` → unknown escape `\x`.

Round-trip property — `Namespace::parse(&ns.render()) == Ok(ns)`
for every `Namespace` — is enforced by 4 unit tests covering:

1. Simple one-tenant / one-tenant-with-scope / multi-scope cases.
2. Empty segments at every position (empty tenant, empty trailing
   scope, only-scope-no-tenant).
3. Segments containing `:` (round-trips through `\:`).
4. Segments containing `\` (round-trips through `\\`, including
   the maximally-escaped `\\\\\\:` corner).

Plus 4 negative / direct cases covering the failure modes and
the happy-path field extraction.

### Why not surface a typed `*ParseError`

The naming taxonomy allows module-internal typed errors
(`CodecError`, `GraphError`, `McpError`, `PersistenceError`) but
public surfaces project to `entelix_core::Error`. Parse failures
are user-facing — operators feed audit-log keys, hand-edited
config, or external-system identifiers into `parse`. Surfacing
`Error::InvalidRequest` keeps the consumer story uniform with
`Error::invalid_request` sites everywhere else. A typed
`NamespaceParseError` would carry more shape but force every
consumer into a `match` against module-private variants — wrong
trade-off for a function with two distinct failure modes.

### Empty-segment policy

`Namespace::new("")` and `Namespace::with_scope("")` are valid
constructions today (no input validation on segment content), so
the round-trip needs to preserve them: `parse("")` → tenant_id="",
scope=[]; `parse("a:")` → tenant_id="a", scope=[""]. The parse
function does not impose stricter validation than the constructors
— if `Namespace::new("")` can produce a namespace, parse must
accept its rendering.

## Consequences

✅ Audit-channel consumers (`SessionLog` readers, operator
dashboards, replay tools) recover the typed `Namespace` from
`GraphEvent::MemoryRecall::namespace_key` with one call. No more
hand-rolled `:`-split / `\`-unescape parsers in downstream code.
✅ Round-trip property is testable in CI — any future change to
`render` that breaks reversibility fails the
`parse_round_trips_*` test family immediately.
✅ Hand-crafted invalid keys (typos, mid-pipeline corruption)
surface as `Error::InvalidRequest` rather than as silent
mis-parses into the wrong tenant. Defense in depth for invariant
#11.
❌ `parse` does not validate segment content beyond escape
correctness — an empty tenant_id is parseable because empty
tenant_id is constructable. The policy "tenant_id must be
non-empty" lives outside the parse layer (currently enforced only
by convention; could be added later as a separate check without
changing parse semantics).

## Alternatives considered

1. **Module-internal `NamespaceParseError`** — see "Why not
   surface a typed `*ParseError`" above. Forces consumers into
   variant matching for a 2-failure-mode function. Rejected.
2. **`TryFrom<&str> for Namespace`** — works for the inverse
   direction, but the `try_from` form drops the discoverability
   advantage of `Namespace::parse` (which sits next to `render`
   in the same impl block). The taxonomy reserves `try_xxx` for
   fallible variants of an existing infallible operation; here
   there is no infallible peer.
3. **Lenient parse (silent unknown-escape passthrough)** — would
   turn `"acme\\x"` into `Namespace::new("acmex")` or similar.
   Reintroduces the silent-misparse failure mode the typed parse
   was meant to close. Rejected; matches invariant #15 (no silent
   fallback) at the parse boundary.
4. **Validate non-empty tenant_id at parse time** — useful but
   asymmetric with `Namespace::new("")` which currently allows
   it. Either both layers reject, or both accept — sliced as a
   separate change.

## References

- ADR-0017 — `tenant_id` mandatory + `Namespace` shape.
- ADR-0032 — invariant #15 (no silent fallback).
- ADR-0037 — invariant #18 (audit channel consumers, primary
  motivation).
- `crates/entelix-memory/src/namespace.rs` — `parse` + 8 tests.
