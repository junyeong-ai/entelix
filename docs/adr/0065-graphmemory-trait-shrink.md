# ADR 0065 — `GraphMemory` trait surface shrink (operator-side admin moves to backend inherent methods)

**Status**: Accepted (supersedes ADRs 0047, 0048, 0049, 0052 trait-method portions)
**Date**: 2026-05-02
**Decision**: Phase 9 final cleanup before 1.0 RC — trait-vs-operator-API split

## Context

ADRs 0047-0052 grew `GraphMemory<N, E>` from 7 to 17 methods over
slices 39-44 to round out CRUD / enumeration / cleanup. The
additions made sense in isolation: `list_nodes` for orphan-sweep
seeding, `list_node_records` for one-round-trip inspection,
`list_edges` / `list_edge_records` mirrors, `prune_orphan_nodes`
as the two-phase-prune second step.

Three problems surfaced when the slice-by-slice pattern was
audited (1.0 RC release prep, slice 67):

1. **Silent no-op risk (invariant 15 conflict)**. Five of the
   added methods (`list_nodes` / `list_node_records` /
   `list_edges` / `list_edge_records` / `prune_orphan_nodes`)
   shipped with `Ok(Vec::new())` / `Ok(0)` default impls.
   A new backend that didn't override them would silently return
   empty / zero. Operators couldn't distinguish "backend doesn't
   support enumeration" from "namespace really is empty" — a
   silent-fallback shape the rest of the codebase rejects (ADR-0032).

2. **Wrong audience**. The added methods serve *operator-side*
   workflows — backup, migration, admin-time orphan sweep, bulk
   export. The agent itself never calls them: agent loops use
   `add_node`, `add_edge`, `node`, `edge`, `neighbors`,
   `traverse`, `find_path`. Putting admin paths on the trait
   forced every backend (current and future) to think about
   them, even when the deployment never reaches for that path.

3. **YAGNI on `*_records` variants**. `list_nodes` (ids only) +
   `list_node_records` (ids + payload) is 2× method count for a
   minor perf gain (skip JSON decode on the hot path). Operators
   that genuinely need the records can call `list_nodes` then
   `node()` per id — the round-trip cost matters only for
   operator-side bulk export, where the operator already holds
   a concrete `PgGraphMemory<N, E>` and can call backend-specific
   bulk paths directly.

The instruction "*sdk를 활용하는 측에서 작업을 하는게 더 효과적이면
trait만 제공해야돼*" makes the answer clear: the trait stays lean,
backends keep their bulk paths as inherent methods.

## Decision

Remove five methods from the `GraphMemory<N, E>` trait surface:

- `list_nodes(ctx, ns, limit, offset) -> Result<Vec<NodeId>>`
- `list_node_records(ctx, ns, limit, offset) -> Result<Vec<(NodeId, N)>>`
- `list_edges(ctx, ns, limit, offset) -> Result<Vec<EdgeId>>`
- `list_edge_records(ctx, ns, limit, offset) -> Result<Vec<GraphHop<E>>>`
- `prune_orphan_nodes(ctx, ns) -> Result<usize>`

`PgGraphMemory<N, E>` keeps these methods as **inherent** (not
trait) on the backend type. The signature drops the `ctx`
parameter — operator-side admin paths don't need an
`ExecutionContext`; the tenant scope lives in `&Namespace` and
the backend handles its own RLS via `set_tenant_session`. Callers
that hold a concrete `PgGraphMemory<N, E>` reach for these
directly:

```rust
let pg: PgGraphMemory<MyNode, MyEdge> = ...;
let ids = pg.list_nodes(&ns, 100, 0).await?;       // inherent
let records = pg.list_node_records(&ns, 100, 0).await?;
let removed = pg.prune_orphan_nodes(&ns).await?;
```

Trait-erased call sites (`Arc<dyn GraphMemory<N, E>>`) lose
access — by design. Trait-erased usage is the agent path; admin
paths require the concrete backend type.

`InMemoryGraphMemory` does **not** ship inherent versions of
these methods. The reference impl is for tests / single-binary
agents; deployments needing admin-time enumeration use
`PgGraphMemory`. Operators on the in-memory backend who genuinely
need to walk every node compose `node()` lookups themselves over
their own id tracking (the in-memory store is a `HashMap` —
operators with this need typically already track their ids
elsewhere).

### Methods that stay in the trait (12 — required)

- `add_node`, `add_edge`, `add_edges_batch` — write
- `node`, `edge` — single-id read
- `neighbors` — read adjacency
- `traverse`, `find_path` — BFS
- `temporal_filter` — time-window query
- `delete_edge`, `delete_node` — write CRUD (now required, no
  `Ok(())` / `Ok(0)` default — backend must explicitly support)

### Methods that stay in the trait with default impl (3 — optional)

- `node_count`, `edge_count` — operator metric (default `Ok(0)`,
  backend overrides for cheap impl). Surface kept because *agents
  legitimately query these* for size-based decisions (paginate
  vs stream, fast-fail empty-namespace check).
- `prune_older_than` — TTL sweep (default `Ok(0)`, backend
  overrides). Operator-side but called from agent contexts when
  the agent owns its own memory cleanup policy.

### Why `delete_edge` / `delete_node` lose their default impls

They were `Ok(())` / `Ok(0)` defaults — same silent-no-op
hazard as the removed list methods. Backend that doesn't
override silently succeeds without doing anything, leaving stale
data. Agents legitimately call these (entity unlearn,
relationship retraction); a silent no-op there is a worse bug
than at the admin-side enumeration path. Marking required forces
every backend to confront the write story explicitly.

## Consequences

✅ Trait surface drops from 17 to 14 methods (12 required + 2
optional with explicit `Ok(0)` default for cheap operator
metrics). Every method now has an unambiguous "what happens if
the backend doesn't override" story.
✅ Silent-no-op risk eliminated. New backend implementing
`GraphMemory<N, E>` can't quietly return empty results from
admin paths the operator was relying on.
✅ Backend-specific perf paths (single-SQL anti-join for
orphan prune, single-RT enumerate via `LIMIT/OFFSET`) preserved
on `PgGraphMemory` as inherent methods. Operators get the same
performance via a slightly different call shape
(`graph.list_nodes(&ns, ...)` instead of
`graph.list_nodes(&ctx, &ns, ...)`).
✅ Trait-erased agent code (`Arc<dyn GraphMemory<N, E>>`) sees
only the surface that *agents actually call*. Fewer methods to
mock in tests; fewer to misunderstand.
✅ The `ctx` parameter drop on inherent admin methods reflects
their semantic — they're operator-time, not request-time.
Removing the parameter from a no-op slot is a small clarity win.
❌ Public-API baseline drift on `entelix-memory` (5 trait
methods removed, 2 default impls removed),
`entelix-graphmemory-pg` (5 trait impl methods → 5 inherent
methods with simpler signature), `entelix` (facade re-export
follows). Refrozen.
❌ `InMemoryGraphMemory` users that called `list_nodes` etc.
through the trait (existing code through 0.x cycle) now have to
compose their own enumeration. Rare in practice — invariant
"agents call only the lean trait surface" was the dominant
pattern even before this slice.

## Alternatives considered

1. **Keep all 17 trait methods, change default impls to
   `Err(...)`** — would surface "this backend doesn't support
   enumeration" but at runtime, not compile time. Defeats the
   trait's contract: trait methods should be callable, not
   sometimes-callable. Rejected.
2. **Keep trait methods, drop only the `*_records` variants** —
   reduces bloat (4 methods → 2) but doesn't address the
   wrong-audience problem (agents still see admin paths) or
   the silent-no-op risk. Rejected as a half-measure.
3. **Move methods to a separate `GraphMemoryAdmin` trait that
   `PgGraphMemory` impls** — preserves trait-erased dispatch
   for admin paths. But `Arc<dyn GraphMemoryAdmin>` admin
   dispatch over heterogeneous backends is a use-case nobody
   asked for; the operator usually knows which backend they
   wired. Inherent-method shape is simpler. Rejected.
4. **Free function `pub fn prune_orphan_nodes<G: GraphMemory<N, E>>(...)`
   in entelix-memory** — generic over backend, naïve
   composition (`list_nodes` → `neighbors` per id →
   `delete_node`). N+1 round-trips on the Pg backend; defeats
   the single-SQL path. Rejected.
5. **Ship `entelix-graphmemory-admin` companion crate with the
   admin surface** — over-engineering for what's a 5-method
   inherent block on the existing backend type. The PG backend
   already lives in its own crate; admin methods belong with
   it. Rejected.

## Migration

Operators who previously called the trait methods through
`Arc<dyn GraphMemory<N, E>>`:

```rust
// Before — trait method
let nodes = graph_dyn.list_nodes(&ctx, &ns, 100, 0).await?;

// After — call requires concrete backend type
let pg: &PgGraphMemory<MyNode, MyEdge> = ...;
let nodes = pg.list_nodes(&ns, 100, 0).await?;  // ctx dropped
```

For `InMemoryGraphMemory` users, the recommended pattern is
operator-side id tracking in their application state — the
in-memory backend exists primarily for tests where total state
is small enough to enumerate via test fixtures.

## References

- ADR-0042 — `PgGraphMemory` companion crate (parent — bulk
  paths now live here as inherent methods).
- ADR-0046 — `delete_edge` / `delete_node` cascade (parent —
  these methods stay in the trait, default impls dropped).
- ADRs 0047, 0048, 0049, 0052 — list / records / orphan-prune
  methods (this ADR supersedes the trait-method portion of
  each; the SQL strategies they documented now live on
  `PgGraphMemory` inherent methods unchanged).
- ADR-0032 — invariant 15 (no silent fallback) — the doctrine
  this slice extends to trait method default impls.
- `crates/entelix-memory/src/graph.rs` — `GraphMemory<N, E>`
  trait shrunk to 14 methods; `InMemoryGraphMemory` impls 14.
- `crates/entelix-graphmemory-pg/src/store.rs` — trait impl
  shrunk to 14, inherent block grew to 5 admin methods
  (`list_nodes`, `list_node_records`, `list_edges`,
  `list_edge_records`, `prune_orphan_nodes`) without `ctx`
  parameter.
- `crates/entelix-graphmemory-pg/tests/postgres_e2e.rs` —
  e2e tests updated to call inherent method shape (`graph.list_nodes(&ns, ...)`).
