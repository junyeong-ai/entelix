# ADR 0063 — `GraphMemory::add_edges_batch` bulk-insert primitive

**Status**: Accepted
**Date**: 2026-05-01
**Decision**: Phase 9 of the post-7-차원-audit roadmap (sixth sub-slice — graphmemory ingest perf)

## Context

Knowledge-graph workloads ingest edges in bursts: a batch import
from a CSV / data warehouse, a relationship-extraction pass over a
document corpus, an LLM-driven entity-resolution job emitting N
edges per round. Each of these is naturally a *batch* operation,
but the trait shape forced the operator to call `add_edge` N
times — N round-trips against the Postgres backend (one
`BEGIN+INSERT+COMMIT` per edge under the tenant-tx envelope from
ADR-0041), and N write-lock acquisitions against the in-memory
backend.

ADR-0058 closed the same shape on the *read* side via
`WITH RECURSIVE` BFS (D round-trips → 1). This slice closes it on
the *write* side: a typed bulk-insert primitive that maps to one
round-trip per backend natively.

## Decision

Add `GraphMemory::add_edges_batch` to the trait surface:

```rust
async fn add_edges_batch(
    &self,
    ctx: &ExecutionContext,
    ns: &Namespace,
    edges: Vec<(NodeId, NodeId, E, DateTime<Utc>)>,
) -> Result<Vec<EdgeId>>;
```

with a default impl that loops over `add_edge` (correct for every
backend, fast for none — backends override to fold the round-trip).

Each tuple is `(from, to, edge, timestamp)` — same shape as
`add_edge`'s positional arguments minus the `&NodeId` references.
The return is `Vec<EdgeId>` in input order so callers can
correlate back into their original batch (e.g. recording a
foreign-key per ingested record).

### `InMemoryGraphMemory<N, E>` override

Two-phase under a single write-lock:

1. **Validate every endpoint up front.** A single bad `from` or
   `to` in the batch fails the entire call before any insert,
   leaving the namespace untouched. Atomic-or-nothing — without
   it, a partially-applied batch would leak good entries past a
   later validation failure.
2. **Insert every edge.** Each gets a freshly-minted `EdgeId`,
   indexed into `out_adj` / `in_adj`, written into `edges`.

The single write-lock acquisition replaces N (one per `add_edge`)
— the same operator hot path's lock contention story improves
linearly with batch size.

### `PgGraphMemory<N, E>` override

Single SQL via `INSERT … SELECT FROM UNNEST(…)`:

```sql
INSERT INTO graph_edges (tenant_id, namespace_key, id, from_node, to_node, payload, ts)
SELECT $1, $2, e.id, e.from_node, e.to_node, e.payload, e.ts
FROM UNNEST($3::TEXT[], $4::TEXT[], $5::TEXT[], $6::JSONB[], $7::TIMESTAMPTZ[])
     AS e(id, from_node, to_node, payload, ts)
```

Per-column arrays are pre-built in the Rust caller (one `Vec<T>`
per UNNEST column); sqlx binds them as Postgres array parameters.
The query runs inside the tenant-tx envelope (`set_tenant_session`
+ single `BEGIN`/`COMMIT`) so the RLS contract from ADR-0041
holds unchanged — the per-row `tenant_id` column is the constant
`$1`, the policy sees a single coherent tenant scope for the
whole batch.

EdgeIds are minted client-side (UUID v7) before the SQL fires —
returned in input order regardless of any internal Postgres
ordering. Callers that need per-row durable ids (audit log, FK
into another table) get them deterministically.

### Why per-column arrays + UNNEST instead of multi-row VALUES

Postgres' `INSERT ... VALUES (a, b, c), (d, e, f), …` works for
small N but the SQL string grows with batch size and prepared-
statement caches in sqlx may degrade. UNNEST is a *constant-arity*
SQL — 7 binds regardless of N. The cached plan reuses across
every batch size; the statement-cache hit-rate stays at 100%.

UNNEST also matches the row-count semantic: every column array
must have the same length, validated by Postgres before the
INSERT touches the table. Mismatched lengths (a programmer bug
caller-side) surface as a typed error before any disk write.

### Why no `add_nodes_batch` companion in this slice

Two reasons. First, node insertion is rarely the bottleneck —
most workloads insert nodes individually as entities are
discovered, then bulk-relate via edges. Second, the parity
question (does `add_node` need the same primitive?) deserves its
own slice with concrete demand evidence, not a speculative
sibling that sits unused.

If demand surfaces (operator request: bulk-import N nodes at the
start of an ingest job), the same UNNEST shape applies and the
trait gets `add_nodes_batch`. ADR-0058's "deferred follow-ups"
pattern — earn the API, don't speculate it.

### Why default impl loops sequentially instead of parallel

`add_edge` already serialises through the InMemoryGraphMemory's
write-lock and the Pg backend's tenant-tx envelope; firing N
parallel `add_edge` futures in the default impl would saturate
neither path *and* would change error-failure semantics (a mid-
batch failure leaves preceding inserts persisted). The sequential
loop matches the bespoke override semantics (fail-fast, atomic-or-
not-applied for the in-memory backend; single-tx for Pg). Backends
that *want* parallel ingest reach for their native bulk path —
that's what overriding is for.

### Tests

- 3 unit tests on `InMemoryGraphMemory` (`graph::tests`):
  - `add_edges_batch_inserts_all_atomically` — 3 edges, every
    returned id resolves back to a hop.
  - `add_edges_batch_rejects_unknown_endpoint_without_partial_writes`
    — bad endpoint mid-batch leaves the namespace untouched
    (atomic-or-nothing).
  - `add_edges_batch_empty_input_is_a_noop` — `Vec::new()` returns
    an empty id list, no work.
- 2 docker-ignored e2e tests on `PgGraphMemory`
  (`postgres_e2e.rs`):
  - `add_edges_batch_inserts_all_in_one_round_trip` — 50 edges
    via UNNEST, every assigned id resolves back, edge count
    rises by exactly 50.
  - `add_edges_batch_empty_input_is_a_noop` — `Vec::new()` short-
    circuits before any SQL.

## Consequences

✅ Operator ingest hot path drops from N round-trips to 1 (Pg
backend) or N lock acquisitions to 1 (in-memory). For a 10k-edge
batch on a 5ms-RTT Postgres deployment, latency drops from ~50s
to ~50ms — three orders of magnitude.
✅ Default impl preserves correctness for backends that don't
override — every `GraphMemory<N, E>` impl picks up the new
primitive transparently.
✅ Atomic-or-nothing semantic on the in-memory backend matches
the single-transaction semantic on the Pg backend — operators
get the same failure-shape across both.
✅ Tenant-tx envelope (ADR-0041) preserved on the Pg backend —
the bulk insert runs under one `set_tenant_session` call inside
one `BEGIN`/`COMMIT`. RLS policy sees a coherent tenant scope.
✅ EdgeIds returned in input order — caller correlates back into
their batch deterministically. Useful for FK persistence and for
audit/replay.
❌ Public-API baseline drift on `entelix-memory` (trait method
added) and `entelix-graphmemory-pg` (override). Refrozen.
❌ No `add_nodes_batch` companion — by design (see "Why no
add_nodes_batch"). Operators with bulk-node demand override
`add_node` calls in a loop until evidence justifies the
companion API.

## Alternatives considered

1. **Multi-row `INSERT ... VALUES (a, b, c), (d, e, f), …`**
   instead of UNNEST — works for small batches but the SQL string
   grows with N, defeating the prepared-statement cache. UNNEST
   is constant-arity. Rejected.
2. **Per-row `add_edge` futures spawned via `try_join_all`** —
   parallel `await`s on the in-memory backend serialise through
   the write-lock anyway; on the Pg backend they spawn N tenant-tx
   envelopes (worse than 1). Sequential loop matches the
   bespoke override semantics. Rejected.
3. **`add_edges_batch` returns `Result<Vec<Result<EdgeId>>>`**
   (per-row error tracking) — encourages partially-applied
   batches and forces every caller to walk the inner result list.
   Atomic-or-nothing is the simpler contract; operators who want
   per-row error semantics chunk their input. Rejected.
4. **Trait-level `BulkInsert<N, E>` companion trait** — extra
   trait surface for one method's worth of behaviour. The
   `add_edges_batch` method on the existing trait keeps the
   discovery surface compact. Rejected.
5. **Streaming variant `add_edges_streamed(impl Stream<…>)`** —
   sqlx's `COPY FROM STDIN` could power true streaming ingest,
   but the operator request profile is "I have a `Vec<…>` and
   want it persisted", not "I have an unbounded stream". Reserved
   if real demand surfaces. Rejected.

## References

- ADR-0042 — `PgGraphMemory` companion crate (parent — the bulk
  path lives in the same backend).
- ADR-0058 — `WITH RECURSIVE` BFS (sibling — closes the same
  N-round-trip shape on the read side).
- ADR-0041 — Postgres RLS via `set_tenant_session` (the tenant-
  tx envelope this slice preserves).
- 7-차원 roadmap §S10 — Phase 9 (companion-perf hardening),
  sixth sub-slice — write-side perf parity with the read-side
  fast path.
- `crates/entelix-memory/src/graph.rs` — `add_edges_batch`
  trait method + InMemory override + 3 unit tests.
- `crates/entelix-graphmemory-pg/src/store.rs` —
  `add_edges_batch` Pg override (UNNEST, single tenant-tx).
- `crates/entelix-graphmemory-pg/tests/postgres_e2e.rs` — 2
  docker-ignored e2e regressions.
