# ADR 0040 — `list_namespaces` structural round-trip via `Namespace::parse`

**Status**: Accepted
**Date**: 2026-05-01
**Decision**: Phase 7 of the post-7-차원-audit roadmap (third sub-slice)

## Context

`Store::list_namespaces(&prefix) -> Vec<Namespace>` was specified
to "list every Namespace under prefix that holds at least one
entry" (trait docstring). Two backends implemented it before
ADR-0039:

- `InMemoryStore::list_namespaces` enumerated rendered keys, then
  for every match returned a *synthetic* `Namespace` shaped like
  the prefix. The actual stored namespace's deeper scope (e.g.
  `agent-a/conv-1`) was lost — every match returned an identical
  prefix-shape clone.
- `PostgresStore::list_namespaces` had the same shape: SELECT
  DISTINCT namespace, then map each row to a prefix-shape clone.

Both backends carried a `// We can't rebuild …` comment
acknowledging the workaround was driven by the absence of
`Namespace::parse` (the inverse of `render`). The trait contract
was effectively unimplementable until parse landed.

The visible symptom was subtle but real: an admin tool calling
`list_namespaces(prefix=tenant/agent-a)` to enumerate every
conversation under `agent-a` would see N copies of
`tenant/agent-a` (one per stored conversation) instead of
`[tenant/agent-a/conv-1, tenant/agent-a/conv-2, ...]`. The count
was right; the *identity* of each namespace was wrong.

ADR-0039 shipped `Namespace::parse(&str) -> Result<Namespace>`.
That made the workaround unnecessary.

## Decision

Replace the synthetic-clone pattern in both backends with
`Namespace::parse(&rendered_key)` per row. The trait contract
holds as written: each returned `Namespace` is structurally equal
to the value originally passed to `Store::put`, recovered via
the round-trip render → store → list → parse.

```rust
// crates/entelix-memory/src/store.rs (InMemoryStore)
seen.into_iter().map(|key| Namespace::parse(&key)).collect()

// crates/entelix-persistence/src/postgres/store.rs (PostgresStore)
rows.into_iter()
    .map(|(rendered,)| Namespace::parse(&rendered))
    .collect()
```

The `?` propagation through `Result<Vec<_>>` means a corrupt row
(unknown escape, trailing backslash) surfaces as
`Error::InvalidRequest` rather than silently producing a wrong
`Namespace`. Defense in depth: hand-edited / corrupted rows
become loud errors at the admin-tool boundary, never silent
mis-attributions.

### Why this needs `Namespace::parse`, not a backend-side parser

A backend-side parser would either:

1. Re-implement the escape rules in every backend (duplicate
   logic, drift risk — invariant #14 forbids the duplication
   shape).
2. Skip the round-trip and return the un-parsed string in a new
   field (`raw_key: String`), pushing parsing onto every consumer
   (the same anti-pattern ADR-0039's "Why not surface a typed
   `*ParseError`" rejected for the public surface).

Surfacing `parse` once on `Namespace` and consuming it from every
backend keeps the escape contract single-source.

### Test additions

- `InMemoryStore`: existing `list_namespaces_finds_subscopes_under_prefix`
  test extended to assert that the returned `Namespace`s are
  structurally equal to the originally stored values (`ns_a` and
  `ns_b`), not just count-equal.
- `InMemoryStore`: new `list_namespaces_recovers_escaped_segments`
  test exercises `:` inside a scope segment (`k8s:pod:foo`) — the
  most error-prone failure mode of a hand-rolled `:`-split parser.
- `PostgresStore` integration: new
  `list_namespaces_returns_typed_scopes_round_tripped_through_render`
  testcontainers test stores three distinct namespaces under one
  prefix (including a `:`-bearing one) and asserts the returned
  set matches the originals structurally.

## Consequences

✅ Admin tools that walk per-tenant storage trees see the correct
namespace identity per row. The `// We can't rebuild …`
workaround comments are gone — the trait contract is honoured as
written.
✅ Round-trip property — `parse(render(ns)) == Ok(ns)` —
becomes a load-bearing test for both backends. Future changes to
`render` that break reversibility fail at the
`list_namespaces_*` boundary, not at a downstream consumer.
✅ Corrupted rows surface as typed errors at the admin-tool
boundary, not as silent wrong-tenant attributions. Aligns with
invariant #15 (no silent fallback) and invariant #11 (tenant
isolation).
✅ `PostgresStore::list_namespaces` no longer constructs the
prefix-shape `Namespace` per row — small allocation savings, but
the bigger win is structural correctness.
❌ A backend that stores partially-corrupted namespace strings
will now bubble parse errors to the admin tool. The prior
behaviour silently returned wrong-but-plausible namespaces; the
new behaviour returns an error. Operators with such corruption
have to fix the underlying rows or skip parse failures explicitly
in their consumer code (the surface gives them the typed error
to act on).

## Alternatives considered

1. **Add a `raw_namespace` field to the result type** — pushes
   parsing onto every consumer, invites the same hand-rolled
   `:`-split bugs `Namespace::parse` was meant to close. Rejected.
2. **Keep the synthetic-clone behaviour as a `_compat` mode** —
   contradicts invariant #14 (no backwards-compatibility shims)
   and ADR-0039's "no silent misparse" stance. Rejected.
3. **Parse lazily inside an iterator returned by
   `list_namespaces`** — defers errors past the function
   boundary, complicates the trait shape. Eager parse
   matches the `Vec<Namespace>` return type the trait already
   commits to. Rejected.

## References

- ADR-0007 — `Store<V>` trait surface (parent).
- ADR-0017 — `Namespace` mandatory tenant_id.
- ADR-0039 — `Namespace::parse` (the prerequisite this slice
  consumes immediately).
- `crates/entelix-memory/src/store.rs:259-289` — fixed
  `InMemoryStore::list_namespaces`.
- `crates/entelix-persistence/src/postgres/store.rs:138-181` —
  fixed `PostgresStore::list_namespaces`.
- `crates/entelix-persistence/tests/postgres_integration.rs:118-148`
  — testcontainers structural round-trip.
