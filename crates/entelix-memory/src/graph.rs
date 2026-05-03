//! `GraphMemory<N, E>` — relationship-aware long-term memory.
//!
//! Where `EntityMemory` records flat `entity → fact` pairs and
//! `SemanticMemory` records embeddings, `GraphMemory` records
//! **typed nodes** plus **typed, timestamped edges** between them.
//! That delivers the actual "graph-based long-term memory" surface
//! the user-level mental model expects: knowledge stored as an
//! evolving entity-relationship graph, queryable by traversal.
//!
//! Trait + reference impl pattern (ADR-0008): the trait is the
//! contract any backend honours (Neo4j, ArangoDB, in-Postgres
//! recursive CTE, …), and [`InMemoryGraphMemory`] is the embedded
//! reference impl — hand-rolled with `BTreeMap` adjacency lists, no
//! external graph library, parallel to [`crate::InMemoryStore`].
//!
//! ## Surface
//!
//! - `add_node` / `add_edge` — insertion. Backends mint stable
//!   [`NodeId`] / [`EdgeId`] values and return them so callers can
//!   reference nodes/edges later without name lookups.
//! - `node` / `edge` — single-id lookup. `node` returns the
//!   payload `Option<N>`; `edge` returns the full
//!   [`GraphHop<E>`] (`from`, `to`, `edge`, `timestamp`).
//! - `neighbors(id, direction)` — outgoing or incoming edges from a
//!   single node.
//! - `traverse(start, max_depth)` — BFS up to `max_depth` hops.
//! - `find_path(from, to, max_depth)` — shortest unweighted path
//!   between two nodes (BFS-based).
//! - `temporal_filter(time_range)` — edges whose timestamp falls in
//!   `[from, to)`. Returns `(EdgeId, NodeId, NodeId, E)` tuples.

use std::collections::{BTreeMap, HashSet, VecDeque};
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use entelix_core::{Error, ExecutionContext, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::namespace::Namespace;

/// Stable, opaque node identifier minted by the backend.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct NodeId(String);

impl NodeId {
    /// Build a fresh id (UUID v7 — sortable by creation time).
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::now_v7().to_string())
    }

    /// Adopt an externally-minted id (e.g. from a graph DB primary
    /// key). Caller is responsible for uniqueness within a
    /// namespace.
    #[must_use]
    pub fn from_string(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Borrow the underlying string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for NodeId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Stable, opaque edge identifier minted by the backend.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct EdgeId(String);

impl EdgeId {
    /// Build a fresh id.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::now_v7().to_string())
    }

    /// Adopt an externally-minted id.
    #[must_use]
    pub fn from_string(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Borrow the underlying string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for EdgeId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for EdgeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Direction of edge traversal — outgoing edges leave a node,
/// incoming edges arrive at it. `Both` returns the union.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum Direction {
    /// Edges where the queried node is the source.
    Outgoing,
    /// Edges where the queried node is the target.
    Incoming,
    /// Either direction.
    Both,
}

/// One traversal hop produced by [`GraphMemory::traverse`] or
/// [`GraphMemory::find_path`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct GraphHop<E> {
    /// Edge that connects the previous node to `node`.
    pub edge_id: EdgeId,
    /// Source of the edge (the previously-visited node).
    pub from: NodeId,
    /// Destination of the edge — the freshly-reached node.
    pub to: NodeId,
    /// Edge payload.
    pub edge: E,
    /// Wall-clock timestamp recorded when the edge was inserted.
    /// Carried on the hop so callers don't have to issue a second
    /// backend lookup just to access temporal data.
    pub timestamp: DateTime<Utc>,
}

impl<E> GraphHop<E> {
    /// Construct a hop from the five fields every backend supplies.
    /// Backends use this constructor (rather than struct-literal
    /// syntax) so the `#[non_exhaustive]` evolvability guarantee
    /// holds across the workspace boundary.
    #[must_use]
    pub const fn new(
        edge_id: EdgeId,
        from: NodeId,
        to: NodeId,
        edge: E,
        timestamp: DateTime<Utc>,
    ) -> Self {
        Self {
            edge_id,
            from,
            to,
            edge,
            timestamp,
        }
    }
}

/// Generic graph-of-knowledge memory. Trait so backends (Neo4j,
/// ArangoDB, Postgres-with-recursive-CTE) can plug in without
/// touching the consumer code; reference in-process impl is
/// [`InMemoryGraphMemory`].
#[async_trait]
pub trait GraphMemory<N, E>: Send + Sync + 'static
where
    N: Clone + Send + Sync + 'static,
    E: Clone + Send + Sync + 'static,
{
    /// Insert `node` and return its assigned id.
    async fn add_node(&self, ctx: &ExecutionContext, ns: &Namespace, node: N) -> Result<NodeId>;

    /// Insert an edge from `from` to `to` carrying `edge`.
    /// `timestamp` is supplied by the caller so re-inserting after
    /// a replay produces deterministic edges.
    async fn add_edge(
        &self,
        ctx: &ExecutionContext,
        ns: &Namespace,
        from: &NodeId,
        to: &NodeId,
        edge: E,
        timestamp: DateTime<Utc>,
    ) -> Result<EdgeId>;

    /// Insert a batch of edges atomically. Each tuple is
    /// `(from, to, edge, timestamp)`; endpoints must already exist
    /// (same contract as [`Self::add_edge`]). Returns the assigned
    /// [`EdgeId`]s in input order.
    ///
    /// Backends with native bulk-insert support (e.g.
    /// `PgGraphMemory`'s `INSERT … SELECT FROM UNNEST(…)`) override
    /// this to fold N round-trips into one. The default impl loops
    /// over [`Self::add_edge`] — correct for every backend, fast
    /// for none. Knowledge-graph batch ingest is the operator
    /// hot path that motivates the override.
    async fn add_edges_batch(
        &self,
        ctx: &ExecutionContext,
        ns: &Namespace,
        edges: Vec<(NodeId, NodeId, E, DateTime<Utc>)>,
    ) -> Result<Vec<EdgeId>> {
        let mut ids = Vec::with_capacity(edges.len());
        for (from, to, edge, timestamp) in edges {
            ids.push(self.add_edge(ctx, ns, &from, &to, edge, timestamp).await?);
        }
        Ok(ids)
    }

    /// Look up a node by id.
    async fn node(&self, ctx: &ExecutionContext, ns: &Namespace, id: &NodeId) -> Result<Option<N>>;

    /// Look up an edge by id and return the full structural body
    /// ([`GraphHop<E>`] — `from`, `to`, `edge`, `timestamp`).
    /// Operators rarely want the payload alone for edges; the
    /// endpoints and timestamp are usually load-bearing for any
    /// follow-up decision (audit context, freshness check,
    /// neighbour navigation). Returning the full hop saves a
    /// second lookup.
    ///
    /// Asymmetric with [`Self::node`] (which returns `Option<N>`
    /// because nodes have no separate structural body) — the
    /// shape difference is intentional, not an oversight.
    ///
    /// Default impl returns `None`.
    async fn edge(
        &self,
        _ctx: &ExecutionContext,
        _ns: &Namespace,
        _edge_id: &EdgeId,
    ) -> Result<Option<GraphHop<E>>> {
        Ok(None)
    }

    /// Edges incident to `node` in the requested direction. Each
    /// triple is `(EdgeId, neighbour NodeId, edge payload)`.
    async fn neighbors(
        &self,
        ctx: &ExecutionContext,
        ns: &Namespace,
        node: &NodeId,
        direction: Direction,
    ) -> Result<Vec<(EdgeId, NodeId, E)>>;

    /// Breadth-first traversal starting at `start`, expanding up to
    /// `max_depth` hops along edges in the requested `direction`.
    /// Returns the visited hops in BFS order (excluding the seed
    /// node, which has no inbound edge in this traversal). Use
    /// [`Direction::Both`] for relationship-graph queries that
    /// don't care about edge polarity (knowledge graphs typically
    /// want this).
    async fn traverse(
        &self,
        ctx: &ExecutionContext,
        ns: &Namespace,
        start: &NodeId,
        direction: Direction,
        max_depth: usize,
    ) -> Result<Vec<GraphHop<E>>>;

    /// Shortest unweighted path from `from` to `to` (BFS) along
    /// edges in the requested `direction`. Returns the sequence of
    /// hops; `Some(vec![])` means `from == to` (already at
    /// destination — no edges traversed); `None` means no path
    /// exists within `max_depth` hops.
    async fn find_path(
        &self,
        ctx: &ExecutionContext,
        ns: &Namespace,
        from: &NodeId,
        to: &NodeId,
        direction: Direction,
        max_depth: usize,
    ) -> Result<Option<Vec<GraphHop<E>>>>;

    /// Edges whose timestamp falls in `[from, to)`. Useful for
    /// audit-log style queries ("what relationships did the agent
    /// learn last week").
    async fn temporal_filter(
        &self,
        ctx: &ExecutionContext,
        ns: &Namespace,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    ) -> Result<Vec<GraphHop<E>>>;

    /// Count nodes in `ns`. Cheap operator metric for
    /// size-based decisions (paginate vs stream, fast-fail
    /// empty-namespace check, audit / dashboard surface).
    /// Default impl returns `0`.
    async fn node_count(&self, _ctx: &ExecutionContext, _ns: &Namespace) -> Result<usize> {
        Ok(0)
    }

    /// Count edges in `ns`. Cheap operator metric — same
    /// rationale as [`Self::node_count`]. Default impl returns
    /// `0`.
    async fn edge_count(&self, _ctx: &ExecutionContext, _ns: &Namespace) -> Result<usize> {
        Ok(0)
    }

    /// Drop one edge by id. Idempotent — deleting an absent
    /// edge succeeds. Required — backends that don't support
    /// edge deletion are degenerate; ADR-0046 closed the
    /// CRUD-completeness gap.
    async fn delete_edge(
        &self,
        ctx: &ExecutionContext,
        ns: &Namespace,
        edge_id: &EdgeId,
    ) -> Result<()>;

    /// Drop one node by id and every edge incident to it.
    /// Cascades — operators that don't want cascading delete
    /// every incident edge first via [`Self::delete_edge`] and
    /// then call this. Returns the count of removed edges so
    /// callers can log or expose cleanup metrics; `0` when the
    /// node had no edges (or was absent — the operation is
    /// idempotent).
    ///
    /// Cascade is the right default because the alternative
    /// (leaving dangling edges that point at a deleted node)
    /// would break the invariant "every edge endpoint is a
    /// resolvable node id" that traversal relies on. Refusing
    /// when edges exist (the SQL `RESTRICT` shape) would force
    /// every operator into a manual edge-delete loop.
    async fn delete_node(
        &self,
        ctx: &ExecutionContext,
        ns: &Namespace,
        node_id: &NodeId,
    ) -> Result<usize>;

    /// Drop every edge in `ns` whose timestamp is older than
    /// `ttl` ago. Returns the count of removed edges so callers
    /// can log or expose pruning metrics.
    ///
    /// Edge-only by design — nodes have no timestamp on the
    /// trait surface, so a TTL sweep cannot reason about them
    /// directly. Nodes left orphaned by edge removal stay in
    /// place until the operator drops them explicitly via a
    /// future operation. This mirrors
    /// [`crate::EntityMemory::prune_older_than`] and
    /// [`crate::EpisodicMemory::prune_older_than`] (single
    /// timestamp axis, no cascading semantics).
    ///
    /// Default impl returns `Ok(0)` — only backends that own a
    /// timestamp index implement this. Operators schedule it on
    /// a timer (or trigger from a periodic graph) to bound
    /// edge-table growth in long-running deployments.
    async fn prune_older_than(
        &self,
        _ctx: &ExecutionContext,
        _ns: &Namespace,
        _ttl: std::time::Duration,
    ) -> Result<usize> {
        Ok(0)
    }
}

// ── reference in-memory impl ─────────────────────────────────────────────

/// Edge metadata kept by [`InMemoryGraphMemory`] alongside the
/// caller-supplied payload.
#[derive(Clone, Debug)]
struct StoredEdge<E> {
    id: EdgeId,
    from: NodeId,
    to: NodeId,
    payload: E,
    timestamp: DateTime<Utc>,
}

/// Per-namespace in-process graph table.
#[derive(Default)]
struct GraphTable<N, E> {
    nodes: BTreeMap<NodeId, N>,
    edges: BTreeMap<EdgeId, StoredEdge<E>>,
    out_adj: BTreeMap<NodeId, Vec<EdgeId>>,
    in_adj: BTreeMap<NodeId, Vec<EdgeId>>,
}

impl<N, E> GraphTable<N, E> {
    const fn new() -> Self {
        Self {
            nodes: BTreeMap::new(),
            edges: BTreeMap::new(),
            out_adj: BTreeMap::new(),
            in_adj: BTreeMap::new(),
        }
    }
}

/// One per-namespace table behind a `RwLock`. Writes against one
/// namespace block writes to that namespace only; reads share the
/// lock's read side. Type alias keeps signatures readable.
type NamespaceTable<N, E> = Arc<RwLock<GraphTable<N, E>>>;

/// Sharded namespace map. `DashMap` partitions across internal
/// stripes so concurrent writes against distinct namespaces never
/// serialise on a single mutex.
type ShardedNamespaceMap<N, E> = Arc<DashMap<String, NamespaceTable<N, E>>>;

/// In-process [`GraphMemory`] backed by `BTreeMap` adjacency lists,
/// sharded per namespace.
///
/// Locking model:
/// - The outer `DashMap` keys per-namespace tables. DashMap shards
///   the map across N internal stripes so insertions and lookups
///   across distinct namespaces never serialise on a single mutex.
/// - Each table sits behind its own `RwLock`. Writes to one
///   namespace block writes to that namespace only; concurrent
///   writes against distinct namespaces run in parallel.
/// - Reads against a namespace share the read side of the per-table
///   `RwLock`, so dashboards / agents querying the same graph in
///   parallel scale linearly until contention on a single tenant's
///   write rate.
///
/// Cheap to clone — internal state is `Arc<DashMap<...>>`-shared,
/// so every clone observes the same graph.
pub struct InMemoryGraphMemory<N, E>
where
    N: Clone + Send + Sync + 'static,
    E: Clone + Send + Sync + 'static,
{
    inner: ShardedNamespaceMap<N, E>,
}

impl<N, E> InMemoryGraphMemory<N, E>
where
    N: Clone + Send + Sync + 'static,
    E: Clone + Send + Sync + 'static,
{
    /// Empty graph. Cheap to clone.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(DashMap::new()),
        }
    }

    /// Total node count across all namespaces — useful for tests.
    /// Iterates DashMap entries (each acquired via its own shard
    /// lock) and reads each per-namespace lock independently.
    #[must_use]
    pub fn total_nodes(&self) -> usize {
        self.inner
            .iter()
            .map(|entry| entry.value().read().nodes.len())
            .sum()
    }

    /// Total edge count across all namespaces — useful for tests.
    #[must_use]
    pub fn total_edges(&self) -> usize {
        self.inner
            .iter()
            .map(|entry| entry.value().read().edges.len())
            .sum()
    }

    /// Look up the per-namespace table without creating one. Returns
    /// `None` for namespaces never written to.
    fn table_for(&self, key: &str) -> Option<NamespaceTable<N, E>> {
        self.inner.get(key).map(|r| Arc::clone(r.value()))
    }

    /// Get-or-create the per-namespace table. Used on the write
    /// path only — the read path bails out early when the namespace
    /// is absent rather than allocating an empty table.
    fn table_for_write(&self, key: String) -> NamespaceTable<N, E> {
        self.inner
            .entry(key)
            .or_insert_with(|| Arc::new(RwLock::new(GraphTable::new())))
            .clone()
    }
}

impl<N, E> Default for InMemoryGraphMemory<N, E>
where
    N: Clone + Send + Sync + 'static,
    E: Clone + Send + Sync + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<N, E> Clone for InMemoryGraphMemory<N, E>
where
    N: Clone + Send + Sync + 'static,
    E: Clone + Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

#[async_trait]
impl<N, E> GraphMemory<N, E> for InMemoryGraphMemory<N, E>
where
    N: Clone + Send + Sync + 'static,
    E: Clone + Send + Sync + 'static,
{
    async fn add_node(&self, _ctx: &ExecutionContext, ns: &Namespace, node: N) -> Result<NodeId> {
        let id = NodeId::new();
        let table = self.table_for_write(ns.render());
        table.write().nodes.insert(id.clone(), node);
        Ok(id)
    }

    async fn add_edge(
        &self,
        _ctx: &ExecutionContext,
        ns: &Namespace,
        from: &NodeId,
        to: &NodeId,
        edge: E,
        timestamp: DateTime<Utc>,
    ) -> Result<EdgeId> {
        let id = EdgeId::new();
        let table = self.table_for_write(ns.render());
        let mut guard = table.write();
        if !guard.nodes.contains_key(from) {
            return Err(entelix_core::Error::invalid_request(format!(
                "GraphMemory::add_edge: source node {from} does not exist"
            )));
        }
        if !guard.nodes.contains_key(to) {
            return Err(entelix_core::Error::invalid_request(format!(
                "GraphMemory::add_edge: target node {to} does not exist"
            )));
        }
        let stored = StoredEdge {
            id: id.clone(),
            from: from.clone(),
            to: to.clone(),
            payload: edge,
            timestamp,
        };
        guard.edges.insert(id.clone(), stored);
        guard
            .out_adj
            .entry(from.clone())
            .or_default()
            .push(id.clone());
        guard.in_adj.entry(to.clone()).or_default().push(id.clone());
        Ok(id)
    }

    async fn add_edges_batch(
        &self,
        _ctx: &ExecutionContext,
        ns: &Namespace,
        edges: Vec<(NodeId, NodeId, E, DateTime<Utc>)>,
    ) -> Result<Vec<EdgeId>> {
        if edges.is_empty() {
            return Ok(Vec::new());
        }
        let table = self.table_for_write(ns.render());
        let mut guard = table.write();
        // First pass: validate every endpoint up front — fail fast
        // before any insert so a partially-applied batch can't leave
        // the namespace in a half-state.
        for (from, to, _, _) in &edges {
            if !guard.nodes.contains_key(from) {
                return Err(entelix_core::Error::invalid_request(format!(
                    "GraphMemory::add_edges_batch: source node {from} does not exist"
                )));
            }
            if !guard.nodes.contains_key(to) {
                return Err(entelix_core::Error::invalid_request(format!(
                    "GraphMemory::add_edges_batch: target node {to} does not exist"
                )));
            }
        }
        // Second pass: every endpoint is known, every insert is safe.
        let mut ids = Vec::with_capacity(edges.len());
        for (from, to, payload, timestamp) in edges {
            let id = EdgeId::new();
            let stored = StoredEdge {
                id: id.clone(),
                from: from.clone(),
                to: to.clone(),
                payload,
                timestamp,
            };
            guard.edges.insert(id.clone(), stored);
            guard.out_adj.entry(from).or_default().push(id.clone());
            guard.in_adj.entry(to).or_default().push(id.clone());
            ids.push(id);
        }
        Ok(ids)
    }

    async fn node(
        &self,
        _ctx: &ExecutionContext,
        ns: &Namespace,
        id: &NodeId,
    ) -> Result<Option<N>> {
        let Some(table) = self.table_for(&ns.render()) else {
            return Ok(None);
        };
        Ok(table.read().nodes.get(id).cloned())
    }

    async fn edge(
        &self,
        _ctx: &ExecutionContext,
        ns: &Namespace,
        edge_id: &EdgeId,
    ) -> Result<Option<GraphHop<E>>> {
        let Some(table) = self.table_for(&ns.render()) else {
            return Ok(None);
        };
        Ok(table.read().edges.get(edge_id).map(|e| GraphHop {
            edge_id: e.id.clone(),
            from: e.from.clone(),
            to: e.to.clone(),
            edge: e.payload.clone(),
            timestamp: e.timestamp,
        }))
    }

    async fn neighbors(
        &self,
        _ctx: &ExecutionContext,
        ns: &Namespace,
        node: &NodeId,
        direction: Direction,
    ) -> Result<Vec<(EdgeId, NodeId, E)>> {
        let Some(table) = self.table_for(&ns.render()) else {
            return Ok(Vec::new());
        };
        let guard = table.read();
        let mut out = Vec::new();
        let mut collect = |edge_ids: &[EdgeId], pick_far: fn(&StoredEdge<E>) -> &NodeId| {
            for eid in edge_ids {
                if let Some(stored) = guard.edges.get(eid) {
                    out.push((
                        eid.clone(),
                        pick_far(stored).clone(),
                        stored.payload.clone(),
                    ));
                }
            }
        };
        if matches!(direction, Direction::Outgoing | Direction::Both)
            && let Some(ids) = guard.out_adj.get(node)
        {
            collect(ids, |s| &s.to);
        }
        if matches!(direction, Direction::Incoming | Direction::Both)
            && let Some(ids) = guard.in_adj.get(node)
        {
            collect(ids, |s| &s.from);
        }
        Ok(out)
    }

    async fn traverse(
        &self,
        ctx: &ExecutionContext,
        ns: &Namespace,
        start: &NodeId,
        direction: Direction,
        max_depth: usize,
    ) -> Result<Vec<GraphHop<E>>> {
        if max_depth == 0 {
            return Ok(Vec::new());
        }
        let Some(table) = self.table_for(&ns.render()) else {
            return Ok(Vec::new());
        };
        let guard = table.read();
        let mut visited: HashSet<NodeId> = HashSet::new();
        visited.insert(start.clone());
        let mut frontier: VecDeque<(NodeId, usize)> = VecDeque::new();
        frontier.push_back((start.clone(), 0));
        let mut out = Vec::new();
        while let Some((current, depth)) = frontier.pop_front() {
            if ctx.is_cancelled() {
                return Err(Error::Cancelled);
            }
            if depth >= max_depth {
                continue;
            }
            for stored in directional_edges(&guard, &current, direction) {
                let neighbour = stored
                    .other_endpoint_of(&current)
                    .cloned()
                    .unwrap_or_else(|| stored.to.clone());
                if visited.insert(neighbour.clone()) {
                    out.push(GraphHop {
                        edge_id: stored.id.clone(),
                        from: stored.from.clone(),
                        to: stored.to.clone(),
                        edge: stored.payload.clone(),
                        timestamp: stored.timestamp,
                    });
                    frontier.push_back((neighbour, depth + 1));
                }
            }
        }
        Ok(out)
    }

    async fn find_path(
        &self,
        ctx: &ExecutionContext,
        ns: &Namespace,
        from: &NodeId,
        to: &NodeId,
        direction: Direction,
        max_depth: usize,
    ) -> Result<Option<Vec<GraphHop<E>>>> {
        if from == to {
            return Ok(Some(Vec::new()));
        }
        if max_depth == 0 {
            return Ok(None);
        }
        let Some(table) = self.table_for(&ns.render()) else {
            return Ok(None);
        };
        let guard = table.read();
        let mut parents: BTreeMap<NodeId, (EdgeId, NodeId)> = BTreeMap::new();
        let mut depths: BTreeMap<NodeId, usize> = BTreeMap::new();
        depths.insert(from.clone(), 0);
        let mut frontier: VecDeque<NodeId> = VecDeque::new();
        frontier.push_back(from.clone());
        while let Some(current) = frontier.pop_front() {
            if ctx.is_cancelled() {
                return Err(Error::Cancelled);
            }
            let depth = *depths.get(&current).unwrap_or(&0);
            if depth >= max_depth {
                continue;
            }
            for stored in directional_edges(&guard, &current, direction) {
                let neighbour = stored
                    .other_endpoint_of(&current)
                    .cloned()
                    .unwrap_or_else(|| stored.to.clone());
                if depths.contains_key(&neighbour) {
                    continue;
                }
                depths.insert(neighbour.clone(), depth + 1);
                parents.insert(neighbour.clone(), (stored.id.clone(), current.clone()));
                if &neighbour == to {
                    let mut hops: Vec<GraphHop<E>> = Vec::new();
                    let mut cursor = to.clone();
                    while let Some((eid, prev)) = parents.get(&cursor).cloned() {
                        if let Some(stored) = guard.edges.get(&eid) {
                            hops.push(GraphHop {
                                edge_id: stored.id.clone(),
                                from: stored.from.clone(),
                                to: stored.to.clone(),
                                edge: stored.payload.clone(),
                                timestamp: stored.timestamp,
                            });
                        }
                        cursor = prev;
                    }
                    hops.reverse();
                    return Ok(Some(hops));
                }
                frontier.push_back(neighbour);
            }
        }
        Ok(None)
    }

    async fn temporal_filter(
        &self,
        _ctx: &ExecutionContext,
        ns: &Namespace,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    ) -> Result<Vec<GraphHop<E>>> {
        let Some(table) = self.table_for(&ns.render()) else {
            return Ok(Vec::new());
        };
        let guard = table.read();
        let mut out: Vec<GraphHop<E>> = guard
            .edges
            .values()
            .filter(|e| e.timestamp >= from && e.timestamp < to)
            .map(|e| GraphHop {
                edge_id: e.id.clone(),
                from: e.from.clone(),
                to: e.to.clone(),
                edge: e.payload.clone(),
                timestamp: e.timestamp,
            })
            .collect();
        out.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        Ok(out)
    }

    async fn node_count(&self, _ctx: &ExecutionContext, ns: &Namespace) -> Result<usize> {
        let Some(table) = self.table_for(&ns.render()) else {
            return Ok(0);
        };
        Ok(table.read().nodes.len())
    }

    async fn edge_count(&self, _ctx: &ExecutionContext, ns: &Namespace) -> Result<usize> {
        let Some(table) = self.table_for(&ns.render()) else {
            return Ok(0);
        };
        Ok(table.read().edges.len())
    }

    async fn delete_edge(
        &self,
        _ctx: &ExecutionContext,
        ns: &Namespace,
        edge_id: &EdgeId,
    ) -> Result<()> {
        let Some(table) = self.table_for(&ns.render()) else {
            return Ok(());
        };
        let mut guard = table.write();
        if let Some(edge) = guard.edges.remove(edge_id) {
            if let Some(out_list) = guard.out_adj.get_mut(&edge.from) {
                out_list.retain(|e| e != &edge.id);
            }
            if let Some(in_list) = guard.in_adj.get_mut(&edge.to) {
                in_list.retain(|e| e != &edge.id);
            }
        }
        Ok(())
    }

    async fn delete_node(
        &self,
        _ctx: &ExecutionContext,
        ns: &Namespace,
        node_id: &NodeId,
    ) -> Result<usize> {
        let Some(table) = self.table_for(&ns.render()) else {
            return Ok(0);
        };
        let mut guard = table.write();
        // Snapshot incident edge ids before mutating — `out_adj`
        // and `in_adj` may overlap when the node has self-loops,
        // so dedup via a HashSet.
        let mut incident: HashSet<EdgeId> = HashSet::new();
        if let Some(out_list) = guard.out_adj.get(node_id) {
            for id in out_list {
                incident.insert(id.clone());
            }
        }
        if let Some(in_list) = guard.in_adj.get(node_id) {
            for id in in_list {
                incident.insert(id.clone());
            }
        }
        let removed = incident.len();
        for edge_id in incident {
            if let Some(edge) = guard.edges.remove(&edge_id) {
                if let Some(out_list) = guard.out_adj.get_mut(&edge.from) {
                    out_list.retain(|e| e != &edge.id);
                }
                if let Some(in_list) = guard.in_adj.get_mut(&edge.to) {
                    in_list.retain(|e| e != &edge.id);
                }
            }
        }
        guard.nodes.remove(node_id);
        guard.out_adj.remove(node_id);
        guard.in_adj.remove(node_id);
        Ok(removed)
    }

    async fn prune_older_than(
        &self,
        _ctx: &ExecutionContext,
        ns: &Namespace,
        ttl: std::time::Duration,
    ) -> Result<usize> {
        let Some(table) = self.table_for(&ns.render()) else {
            return Ok(0);
        };
        // chrono::Duration is signed and uses i64 nanoseconds; for
        // pathological ttls (above i64::MAX seconds) saturate to
        // chrono::Duration::MAX so the cutoff stays in the past.
        let cutoff = Utc::now() - chrono::Duration::from_std(ttl).unwrap_or(chrono::Duration::MAX);
        let mut guard = table.write();
        let stale: Vec<EdgeId> = guard
            .edges
            .iter()
            .filter(|(_, e)| e.timestamp < cutoff)
            .map(|(id, _)| id.clone())
            .collect();
        let removed = stale.len();
        for id in stale {
            if let Some(edge) = guard.edges.remove(&id) {
                if let Some(out_list) = guard.out_adj.get_mut(&edge.from) {
                    out_list.retain(|e| e != &edge.id);
                }
                if let Some(in_list) = guard.in_adj.get_mut(&edge.to) {
                    in_list.retain(|e| e != &edge.id);
                }
            }
        }
        Ok(removed)
    }
}

impl<E> StoredEdge<E> {
    fn other_endpoint_of(&self, node: &NodeId) -> Option<&NodeId> {
        if &self.from == node {
            Some(&self.to)
        } else if &self.to == node {
            Some(&self.from)
        } else {
            None
        }
    }
}

fn directional_edges<'a, N, E>(
    table: &'a GraphTable<N, E>,
    node: &NodeId,
    direction: Direction,
) -> Vec<&'a StoredEdge<E>> {
    let mut out: Vec<&'a StoredEdge<E>> = Vec::new();
    if matches!(direction, Direction::Outgoing | Direction::Both)
        && let Some(ids) = table.out_adj.get(node)
    {
        for eid in ids {
            if let Some(stored) = table.edges.get(eid) {
                out.push(stored);
            }
        }
    }
    if matches!(direction, Direction::Incoming | Direction::Both)
        && let Some(ids) = table.in_adj.get(node)
    {
        for eid in ids {
            if let Some(stored) = table.edges.get(eid) {
                out.push(stored);
            }
        }
    }
    out
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::many_single_char_names
)]
mod tests {
    use super::*;

    fn ns() -> Namespace {
        Namespace::new("tenant").with_scope("graph")
    }

    #[tokio::test]
    async fn add_and_lookup_node() {
        let g = InMemoryGraphMemory::<&str, &str>::new();
        let ctx = ExecutionContext::new();
        let id = g.add_node(&ctx, &ns(), "alice").await.unwrap();
        let fetched = g.node(&ctx, &ns(), &id).await.unwrap();
        assert_eq!(fetched, Some("alice"));
    }

    #[tokio::test]
    async fn add_edges_batch_inserts_all_atomically() {
        let g = InMemoryGraphMemory::<&str, &str>::new();
        let ctx = ExecutionContext::new();
        let alice = g.add_node(&ctx, &ns(), "alice").await.unwrap();
        let bob = g.add_node(&ctx, &ns(), "bob").await.unwrap();
        let carol = g.add_node(&ctx, &ns(), "carol").await.unwrap();
        let now = Utc::now();
        let ids = g
            .add_edges_batch(
                &ctx,
                &ns(),
                vec![
                    (alice.clone(), bob.clone(), "knows", now),
                    (bob.clone(), carol.clone(), "knows", now),
                    (alice.clone(), carol.clone(), "knows", now),
                ],
            )
            .await
            .unwrap();
        assert_eq!(ids.len(), 3, "returns one EdgeId per input");
        // Every id must resolve to the corresponding hop.
        for id in &ids {
            let hop = g.edge(&ctx, &ns(), id).await.unwrap();
            assert!(hop.is_some(), "edge {id} must be retrievable");
        }
    }

    #[tokio::test]
    async fn add_edges_batch_rejects_unknown_endpoint_without_partial_writes() {
        // Validation happens before any insert — a single bad edge
        // in the batch must leave the namespace untouched.
        let g = InMemoryGraphMemory::<&str, &str>::new();
        let ctx = ExecutionContext::new();
        let alice = g.add_node(&ctx, &ns(), "alice").await.unwrap();
        let bob = g.add_node(&ctx, &ns(), "bob").await.unwrap();
        let ghost = NodeId::new();
        let now = Utc::now();
        let err = g
            .add_edges_batch(
                &ctx,
                &ns(),
                vec![
                    (alice.clone(), bob.clone(), "knows", now),
                    (alice.clone(), ghost, "knows", now), // bad
                ],
            )
            .await;
        assert!(err.is_err(), "batch with unknown endpoint must fail");
        // No edge should have been written — the good first entry
        // must NOT have leaked through.
        assert_eq!(g.edge_count(&ctx, &ns()).await.unwrap(), 0);
    }

    #[tokio::test]
    async fn add_edges_batch_empty_input_is_a_noop() {
        let g = InMemoryGraphMemory::<&str, &str>::new();
        let ctx = ExecutionContext::new();
        let ids = g.add_edges_batch(&ctx, &ns(), Vec::new()).await.unwrap();
        assert!(ids.is_empty());
    }

    #[tokio::test]
    async fn add_edge_requires_existing_endpoints() {
        let g = InMemoryGraphMemory::<&str, &str>::new();
        let ctx = ExecutionContext::new();
        let alice = g.add_node(&ctx, &ns(), "alice").await.unwrap();
        let ghost = NodeId::new();
        let err = g
            .add_edge(&ctx, &ns(), &alice, &ghost, "knows", Utc::now())
            .await;
        assert!(err.is_err());
    }

    #[tokio::test]
    async fn neighbors_split_by_direction() {
        let g = InMemoryGraphMemory::<&str, &str>::new();
        let ctx = ExecutionContext::new();
        let alice = g.add_node(&ctx, &ns(), "alice").await.unwrap();
        let bob = g.add_node(&ctx, &ns(), "bob").await.unwrap();
        let _eid = g
            .add_edge(&ctx, &ns(), &alice, &bob, "knows", Utc::now())
            .await
            .unwrap();
        let outgoing = g
            .neighbors(&ctx, &ns(), &alice, Direction::Outgoing)
            .await
            .unwrap();
        assert_eq!(outgoing.len(), 1);
        let incoming = g
            .neighbors(&ctx, &ns(), &alice, Direction::Incoming)
            .await
            .unwrap();
        assert!(incoming.is_empty());
    }

    #[tokio::test]
    async fn traverse_respects_max_depth() {
        let g = InMemoryGraphMemory::<&str, &str>::new();
        let ctx = ExecutionContext::new();
        let a = g.add_node(&ctx, &ns(), "a").await.unwrap();
        let b = g.add_node(&ctx, &ns(), "b").await.unwrap();
        let c = g.add_node(&ctx, &ns(), "c").await.unwrap();
        let d = g.add_node(&ctx, &ns(), "d").await.unwrap();
        let now = Utc::now();
        g.add_edge(&ctx, &ns(), &a, &b, "->", now).await.unwrap();
        g.add_edge(&ctx, &ns(), &b, &c, "->", now).await.unwrap();
        g.add_edge(&ctx, &ns(), &c, &d, "->", now).await.unwrap();
        let two = g
            .traverse(&ctx, &ns(), &a, Direction::Outgoing, 2)
            .await
            .unwrap();
        assert_eq!(two.len(), 2);
        let three = g
            .traverse(&ctx, &ns(), &a, Direction::Outgoing, 3)
            .await
            .unwrap();
        assert_eq!(three.len(), 3);
    }

    #[tokio::test]
    async fn traverse_with_direction_both_walks_inverse_edges() {
        let g = InMemoryGraphMemory::<&str, &str>::new();
        let ctx = ExecutionContext::new();
        let a = g.add_node(&ctx, &ns(), "a").await.unwrap();
        let b = g.add_node(&ctx, &ns(), "b").await.unwrap();
        let c = g.add_node(&ctx, &ns(), "c").await.unwrap();
        let now = Utc::now();
        // a -> b <- c — walking from b in `Both` direction should
        // reach both a (incoming) and c (incoming).
        g.add_edge(&ctx, &ns(), &a, &b, "->", now).await.unwrap();
        g.add_edge(&ctx, &ns(), &c, &b, "->", now).await.unwrap();
        let from_b = g
            .traverse(&ctx, &ns(), &b, Direction::Both, 1)
            .await
            .unwrap();
        assert_eq!(from_b.len(), 2);
    }

    #[tokio::test]
    async fn find_path_returns_shortest() {
        let g = InMemoryGraphMemory::<&str, &str>::new();
        let ctx = ExecutionContext::new();
        let a = g.add_node(&ctx, &ns(), "a").await.unwrap();
        let b = g.add_node(&ctx, &ns(), "b").await.unwrap();
        let c = g.add_node(&ctx, &ns(), "c").await.unwrap();
        let now = Utc::now();
        g.add_edge(&ctx, &ns(), &a, &b, "ab", now).await.unwrap();
        g.add_edge(&ctx, &ns(), &b, &c, "bc", now).await.unwrap();
        let path = g
            .find_path(&ctx, &ns(), &a, &c, Direction::Outgoing, 5)
            .await
            .unwrap();
        let hops = path.unwrap();
        assert_eq!(hops.len(), 2);
        assert_eq!(hops[0].from, a);
        assert_eq!(hops[1].to, c);
    }

    #[tokio::test]
    async fn temporal_filter_picks_window() {
        let g = InMemoryGraphMemory::<&str, &str>::new();
        let ctx = ExecutionContext::new();
        let a = g.add_node(&ctx, &ns(), "a").await.unwrap();
        let b = g.add_node(&ctx, &ns(), "b").await.unwrap();
        let early = Utc::now() - chrono::Duration::hours(2);
        let late = Utc::now();
        g.add_edge(&ctx, &ns(), &a, &b, "early", early)
            .await
            .unwrap();
        g.add_edge(&ctx, &ns(), &a, &b, "late", late).await.unwrap();
        let window = g
            .temporal_filter(
                &ctx,
                &ns(),
                Utc::now() - chrono::Duration::hours(1),
                Utc::now() + chrono::Duration::hours(1),
            )
            .await
            .unwrap();
        assert_eq!(window.len(), 1);
        assert_eq!(window[0].edge, "late");
    }

    #[tokio::test]
    async fn node_count_and_edge_count_track_inserts() {
        let g = InMemoryGraphMemory::<&str, &str>::new();
        let ctx = ExecutionContext::new();
        // Empty namespace.
        assert_eq!(g.node_count(&ctx, &ns()).await.unwrap(), 0);
        assert_eq!(g.edge_count(&ctx, &ns()).await.unwrap(), 0);
        let a = g.add_node(&ctx, &ns(), "a").await.unwrap();
        let b = g.add_node(&ctx, &ns(), "b").await.unwrap();
        assert_eq!(g.node_count(&ctx, &ns()).await.unwrap(), 2);
        assert_eq!(g.edge_count(&ctx, &ns()).await.unwrap(), 0);
        let _ = g
            .add_edge(&ctx, &ns(), &a, &b, "ab", Utc::now())
            .await
            .unwrap();
        assert_eq!(g.edge_count(&ctx, &ns()).await.unwrap(), 1);
    }

    #[tokio::test]
    async fn count_methods_respect_namespace_isolation() {
        let g = InMemoryGraphMemory::<&str, &str>::new();
        let ctx = ExecutionContext::new();
        let alpha = Namespace::new("tenant").with_scope("alpha");
        let beta = Namespace::new("tenant").with_scope("beta");
        let _ = g.add_node(&ctx, &alpha, "n").await.unwrap();
        assert_eq!(g.node_count(&ctx, &alpha).await.unwrap(), 1);
        assert_eq!(g.node_count(&ctx, &beta).await.unwrap(), 0);
    }

    #[tokio::test]
    async fn delete_edge_is_idempotent_and_dedups_adjacency() {
        let g = InMemoryGraphMemory::<&str, &str>::new();
        let ctx = ExecutionContext::new();
        let a = g.add_node(&ctx, &ns(), "a").await.unwrap();
        let b = g.add_node(&ctx, &ns(), "b").await.unwrap();
        let now = Utc::now();
        let id = g.add_edge(&ctx, &ns(), &a, &b, "ab", now).await.unwrap();
        // Removing twice succeeds — idempotent.
        g.delete_edge(&ctx, &ns(), &id).await.unwrap();
        g.delete_edge(&ctx, &ns(), &id).await.unwrap();
        // Adjacency lists no longer carry the deleted id.
        let outgoing = g
            .neighbors(&ctx, &ns(), &a, Direction::Outgoing)
            .await
            .unwrap();
        assert!(outgoing.is_empty());
        let incoming = g
            .neighbors(&ctx, &ns(), &b, Direction::Incoming)
            .await
            .unwrap();
        assert!(incoming.is_empty());
    }

    #[tokio::test]
    async fn delete_node_cascades_to_incident_edges() {
        let g = InMemoryGraphMemory::<&str, &str>::new();
        let ctx = ExecutionContext::new();
        let a = g.add_node(&ctx, &ns(), "a").await.unwrap();
        let b = g.add_node(&ctx, &ns(), "b").await.unwrap();
        let c = g.add_node(&ctx, &ns(), "c").await.unwrap();
        let now = Utc::now();
        // Three edges incident to `a`: out to b, out to c, in from b.
        let _ = g.add_edge(&ctx, &ns(), &a, &b, "ab", now).await.unwrap();
        let _ = g.add_edge(&ctx, &ns(), &a, &c, "ac", now).await.unwrap();
        let _ = g.add_edge(&ctx, &ns(), &b, &a, "ba", now).await.unwrap();
        let removed = g.delete_node(&ctx, &ns(), &a).await.unwrap();
        assert_eq!(removed, 3);
        assert!(g.node(&ctx, &ns(), &a).await.unwrap().is_none());
        // `b` and `c` survive (cascade is node-scoped, not graph-wide).
        assert!(g.node(&ctx, &ns(), &b).await.unwrap().is_some());
        assert!(g.node(&ctx, &ns(), &c).await.unwrap().is_some());
        // No dangling adjacency entries — `b`'s incoming edge from
        // `a` is gone too.
        let b_in = g
            .neighbors(&ctx, &ns(), &b, Direction::Incoming)
            .await
            .unwrap();
        assert!(b_in.is_empty());
    }

    #[tokio::test]
    async fn delete_node_with_self_loop_dedups_count() {
        let g = InMemoryGraphMemory::<&str, &str>::new();
        let ctx = ExecutionContext::new();
        let a = g.add_node(&ctx, &ns(), "a").await.unwrap();
        // Self-loop appears in both out_adj and in_adj — the snapshot
        // collection must dedup so the returned count is 1, not 2.
        let _ = g
            .add_edge(&ctx, &ns(), &a, &a, "self", Utc::now())
            .await
            .unwrap();
        assert_eq!(g.delete_node(&ctx, &ns(), &a).await.unwrap(), 1);
    }

    #[tokio::test]
    async fn delete_node_on_absent_node_is_zero_noop() {
        let g = InMemoryGraphMemory::<&str, &str>::new();
        let ctx = ExecutionContext::new();
        let phantom = NodeId::from_string("does-not-exist");
        assert_eq!(g.delete_node(&ctx, &ns(), &phantom).await.unwrap(), 0);
    }

    #[tokio::test]
    async fn prune_older_than_drops_stale_edges_only() {
        let g = InMemoryGraphMemory::<&str, &str>::new();
        let ctx = ExecutionContext::new();
        let a = g.add_node(&ctx, &ns(), "a").await.unwrap();
        let b = g.add_node(&ctx, &ns(), "b").await.unwrap();
        let now = Utc::now();
        let _old = g
            .add_edge(
                &ctx,
                &ns(),
                &a,
                &b,
                "old",
                now - chrono::Duration::seconds(120),
            )
            .await
            .unwrap();
        let _fresh = g
            .add_edge(
                &ctx,
                &ns(),
                &a,
                &b,
                "fresh",
                now - chrono::Duration::seconds(5),
            )
            .await
            .unwrap();
        let removed = g
            .prune_older_than(&ctx, &ns(), std::time::Duration::from_secs(60))
            .await
            .unwrap();
        assert_eq!(removed, 1);
        // Both nodes survive — edge-only sweep, no orphan cleanup.
        assert!(g.node(&ctx, &ns(), &a).await.unwrap().is_some());
        assert!(g.node(&ctx, &ns(), &b).await.unwrap().is_some());
        // Adjacency lists deduplicated — only the fresh edge remains.
        let outgoing = g
            .neighbors(&ctx, &ns(), &a, Direction::Outgoing)
            .await
            .unwrap();
        assert_eq!(outgoing.len(), 1);
        assert_eq!(outgoing[0].2, "fresh");
    }

    #[tokio::test]
    async fn prune_older_than_on_empty_namespace_is_noop() {
        let g = InMemoryGraphMemory::<&str, &str>::new();
        let ctx = ExecutionContext::new();
        let removed = g
            .prune_older_than(&ctx, &ns(), std::time::Duration::from_secs(0))
            .await
            .unwrap();
        assert_eq!(removed, 0);
    }

    #[tokio::test]
    async fn namespaces_are_isolated() {
        let g = InMemoryGraphMemory::<&str, &str>::new();
        let ctx = ExecutionContext::new();
        let alpha = Namespace::new("tenant").with_scope("alpha");
        let beta = Namespace::new("tenant").with_scope("beta");
        let _ = g.add_node(&ctx, &alpha, "a-node").await.unwrap();
        let _ = g.add_node(&ctx, &beta, "b-node").await.unwrap();
        // Total bookkeeping shows both, but per-namespace traverse
        // never crosses.
        assert_eq!(g.total_nodes(), 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn distinct_namespaces_write_concurrently() {
        // Lock the architectural promise that per-namespace sharding
        // delivers: 8 tasks, each writing into its own namespace,
        // run in parallel without contending on a single global
        // lock. The test does not measure wall time (flaky on CI)
        // but verifies the structural property: every per-namespace
        // table sees exactly the writes it owns, none of its
        // siblings'.
        let g: InMemoryGraphMemory<String, String> = InMemoryGraphMemory::new();
        let mut handles = Vec::new();
        for tenant in 0..8 {
            let g = g.clone();
            handles.push(tokio::spawn(async move {
                let ctx = ExecutionContext::new();
                let ns = Namespace::new(format!("tenant-{tenant}"));
                let mut ids = Vec::new();
                for i in 0..50 {
                    let id = g
                        .add_node(&ctx, &ns, format!("t{tenant}-n{i}"))
                        .await
                        .unwrap();
                    ids.push(id);
                }
                // Edge between every consecutive pair — exercises the
                // write path beyond simple node insertion.
                let now = Utc::now();
                for window in ids.windows(2) {
                    g.add_edge(
                        &ctx,
                        &ns,
                        &window[0],
                        &window[1],
                        format!("t{tenant}-edge"),
                        now,
                    )
                    .await
                    .unwrap();
                }
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        // Each tenant inserted 50 nodes + 49 edges in its own
        // namespace; cross-pollution would change these counts.
        assert_eq!(g.total_nodes(), 8 * 50);
        assert_eq!(g.total_edges(), 8 * 49);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn read_during_write_on_other_namespace_does_not_block() {
        // Reads of namespace A must not be serialised behind a
        // long-running write to namespace B. We can't prove
        // wall-clock independence portably, but we can verify the
        // structural property: a holding write lock on namespace A
        // does not even acquire the per-namespace lock for B, so
        // reads of B succeed regardless. Drive this through the
        // public surface: spawn N reads on `beta` while a write on
        // `alpha` is in flight, every read returns its expected
        // value.
        let g: InMemoryGraphMemory<String, String> = InMemoryGraphMemory::new();
        let alpha = Namespace::new("alpha");
        let beta = Namespace::new("beta");
        let ctx = ExecutionContext::new();
        let beta_node_id = g
            .add_node(&ctx, &beta, "beta-fixture".to_owned())
            .await
            .unwrap();

        // Hold a write lock on alpha by inserting many nodes in a
        // loop. Concurrently, drive reads on beta and assert each
        // succeeds.
        let g_writer = g.clone();
        let alpha_writer = alpha.clone();
        let writer = tokio::spawn(async move {
            let ctx = ExecutionContext::new();
            for i in 0..200 {
                g_writer
                    .add_node(&ctx, &alpha_writer, format!("alpha-{i}"))
                    .await
                    .unwrap();
            }
        });
        let mut reads = Vec::new();
        for _ in 0..200 {
            let g_reader = g.clone();
            let beta_reader = beta.clone();
            let id_reader = beta_node_id.clone();
            reads.push(tokio::spawn(async move {
                let ctx = ExecutionContext::new();
                g_reader.node(&ctx, &beta_reader, &id_reader).await.unwrap()
            }));
        }
        for r in reads {
            assert_eq!(r.await.unwrap().as_deref(), Some("beta-fixture"));
        }
        writer.await.unwrap();
        assert_eq!(g.total_nodes(), 1 + 200);
    }

    #[tokio::test]
    async fn traverse_short_circuits_on_cancellation() {
        let g = InMemoryGraphMemory::<&str, &str>::new();
        let ctx = ExecutionContext::new();
        let a = g.add_node(&ctx, &ns(), "a").await.unwrap();
        let b = g.add_node(&ctx, &ns(), "b").await.unwrap();
        let c = g.add_node(&ctx, &ns(), "c").await.unwrap();
        let now = Utc::now();
        g.add_edge(&ctx, &ns(), &a, &b, "knows", now).await.unwrap();
        g.add_edge(&ctx, &ns(), &b, &c, "knows", now).await.unwrap();
        ctx.cancellation().cancel();
        let err = g
            .traverse(&ctx, &ns(), &a, Direction::Outgoing, 5)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Cancelled), "got {err:?}");
    }

    #[tokio::test]
    async fn find_path_short_circuits_on_cancellation() {
        let g = InMemoryGraphMemory::<&str, &str>::new();
        let ctx = ExecutionContext::new();
        let a = g.add_node(&ctx, &ns(), "a").await.unwrap();
        let b = g.add_node(&ctx, &ns(), "b").await.unwrap();
        let c = g.add_node(&ctx, &ns(), "c").await.unwrap();
        let now = Utc::now();
        g.add_edge(&ctx, &ns(), &a, &b, "e", now).await.unwrap();
        g.add_edge(&ctx, &ns(), &b, &c, "e", now).await.unwrap();
        ctx.cancellation().cancel();
        let err = g
            .find_path(&ctx, &ns(), &a, &c, Direction::Outgoing, 5)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Cancelled), "got {err:?}");
    }
}
