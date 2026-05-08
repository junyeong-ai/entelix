# entelix-memory-qdrant

Companion crate. Concrete `VectorStore` impl backed by qdrant gRPC — production vector store with single-collection multi-tenancy via payload-filter on `namespace_key`.

## Surface

- **`QdrantVectorStore`** + **`QdrantVectorStoreBuilder`** — `with_url(.)` / `with_api_key(.)` / `with_collection(name)` / `with_dimension(d)` / `with_distance_metric(m)` / `with_timeout(duration)` / `build() -> Result<Self>`. Pool-shared via `Arc<qdrant_client::Qdrant>`.
- **`DistanceMetric`** (re-exported as `QdrantDistanceMetric` from facade) — `Cosine` / `Dot` / `Euclid` (qdrant native enum).
- **`QdrantStoreError`** — typed error wrapping `qdrant_client::QdrantError` plus `Malformed` / `Config` / `FilterProjection`.

## Crate-local rules

- **Single-collection multi-tenancy** — every write tags the point with `namespace_key` (rendered `tenant_id:scope.` from `Namespace::render`). Every search composes `must`-match on `namespace_key` BEFORE any user-supplied filter. The namespace is the security boundary, not a hint.
- **`with_timeout` is per-call**, not per-collection — qdrant's gRPC client allows per-RPC timeout override; the builder default applies when caller doesn't override via `ctx.deadline()`.
- **Cancellation polled at every gRPC entry** — every gRPC call site in `store.rs` (search / upsert / delete / count / …) checks `ctx.is_cancelled()` before sending.
- **Filter projection preserves the namespace anchor** — operator-supplied `VectorFilter` becomes additional `must` clauses; the `namespace_key` filter never moves to `should` or drops.

## Forbidden

- A search/upsert path that omits the `namespace_key` filter — cross-tenant vector retrieval, security boundary break.
- Per-call client construction — defeats the connection pool (same rule the `Embedder` pool follows).


- `entelix-memory` CLAUDE.md — namespace as security boundary, mandatory `tenant_id` (invariant 11).
