//! `PgGraphMemory<N, E>` — `entelix_memory::GraphMemory<N, E>`
//! over Postgres + JSONB payload columns.
//!
//! ## Storage shape
//!
//! Two tables:
//!
//! - **nodes** (`graph_nodes` by default): `(namespace_key, id,
//!   payload)` with composite PK `(namespace_key, id)`.
//! - **edges** (`graph_edges` by default): `(namespace_key, id,
//!   from_node, to_node, payload, ts)` with composite PK
//!   `(namespace_key, id)`, plus covering indexes on
//!   `(namespace_key, from_node)` / `(namespace_key, to_node)` /
//!   `(namespace_key, ts)`.
//!
//! Every read / write rides a `WHERE namespace_key = $1` anchor —
//! invariant 11 / F2 demands structural tenant isolation, and the
//! composite PK doubles as the B-tree index that anchor relies
//! on.
//!
//! ## Traversal model
//!
//! `traverse` and `find_path` issue a single `WITH RECURSIVE` query
//! per call — Postgres expands the BFS server-side, returning the
//! visited hops (or the reconstructed shortest path) in one
//! round-trip regardless of `max_depth`. The recursive CTE carries
//! a per-row `visited` array for cycle prevention, and `find_path`
//! additionally accumulates an `edge_path` array that the outer
//! query unrolls and rejoins to the edges table to reconstruct the
//! full hop sequence in BFS order.
//!
//! ## Schema-as-code escape hatch
//!
//! Operators that own the schema externally (DBA-managed, IaC,
//! migration pipeline) call [`PgGraphMemoryBuilder::with_auto_migrate`]
//! with `false` — the builder skips table / index creation,
//! trusting the operator to have stamped them.

use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use entelix_core::{ExecutionContext, Result};
use entelix_memory::{Direction, EdgeId, GraphHop, GraphMemory, Namespace, NodeId};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use sqlx::postgres::{PgPool, PgPoolOptions};

use crate::error::{PgGraphMemoryError, PgGraphMemoryResult};
use crate::migration::bootstrap;
use crate::tenant::set_tenant_session;

const DEFAULT_NODES_TABLE: &str = "graph_nodes";
const DEFAULT_EDGES_TABLE: &str = "graph_edges";

/// Postgres-backed [`GraphMemory<N, E>`].
///
/// Cheap to clone — internal state is an `Arc<PgPool>` plus two
/// owned table-name strings.
pub struct PgGraphMemory<N, E> {
    pool: Arc<PgPool>,
    nodes_table: Arc<str>,
    edges_table: Arc<str>,
    _phantom: PhantomData<fn() -> (N, E)>,
}

impl<N, E> Clone for PgGraphMemory<N, E> {
    fn clone(&self) -> Self {
        Self {
            pool: Arc::clone(&self.pool),
            nodes_table: Arc::clone(&self.nodes_table),
            edges_table: Arc::clone(&self.edges_table),
            _phantom: PhantomData,
        }
    }
}

impl<N, E> std::fmt::Debug for PgGraphMemory<N, E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PgGraphMemory")
            .field("nodes_table", &self.nodes_table)
            .field("edges_table", &self.edges_table)
            .finish_non_exhaustive()
    }
}

impl<N, E> PgGraphMemory<N, E> {
    /// Start a fluent builder. `connection_string` is the only
    /// required field; everything else has a sensible default.
    pub fn builder() -> PgGraphMemoryBuilder<N, E> {
        PgGraphMemoryBuilder::new()
    }

    // ── Backend-specific admin / migration surface ────────────────────
    //
    // The methods below are *not* part of the `GraphMemory` trait — they
    // are operator-side enumeration / cleanup paths the SDK delegates to
    // the backend type because (a) the trait would otherwise grow a
    // surface every backend must re-implement (silent no-op risk per
    // invariant 15), and (b) these are admin/migration concerns the
    // *operator* runs, not the *agent*. Callers that hold a concrete
    // `PgGraphMemory<N, E>` reach for these directly; trait-erased
    // call sites do not see them.

    /// Paginated node-id enumeration, ascending by id (UUID v7
    /// mint-time order). Operator-side admin / migration path.
    pub async fn list_nodes(
        &self,
        ns: &Namespace,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<NodeId>> {
        let sql = format!(
            "SELECT id FROM {} \
             WHERE namespace_key = $1 \
             ORDER BY id ASC \
             LIMIT $2 OFFSET $3",
            self.nodes_table
        );
        let limit_i64 = i64::try_from(limit).unwrap_or(i64::MAX);
        let offset_i64 = i64::try_from(offset).unwrap_or(i64::MAX);
        let mut tx = self.pool.begin().await.map_err(into_core_sqlx)?;
        set_tenant_session(&mut *tx, ns.tenant_id()).await?;
        let rows: Vec<(String,)> = sqlx::query_as(&sql)
            .bind(ns.render())
            .bind(limit_i64)
            .bind(offset_i64)
            .fetch_all(&mut *tx)
            .await
            .map_err(into_core_sqlx)?;
        tx.commit().await.map_err(into_core_sqlx)?;
        Ok(rows
            .into_iter()
            .map(|(id,)| NodeId::from_string(id))
            .collect())
    }

    /// Paginated `(NodeId, N)` enumeration — single round-trip
    /// versus `list_nodes` + per-id `node()`. Operator-side
    /// bulk-export path.
    pub async fn list_node_records(
        &self,
        ns: &Namespace,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<(NodeId, N)>>
    where
        N: DeserializeOwned,
    {
        let sql = format!(
            "SELECT id, payload FROM {} \
             WHERE namespace_key = $1 \
             ORDER BY id ASC \
             LIMIT $2 OFFSET $3",
            self.nodes_table
        );
        let limit_i64 = i64::try_from(limit).unwrap_or(i64::MAX);
        let offset_i64 = i64::try_from(offset).unwrap_or(i64::MAX);
        let mut tx = self.pool.begin().await.map_err(into_core_sqlx)?;
        set_tenant_session(&mut *tx, ns.tenant_id()).await?;
        let rows: Vec<(String, Value)> = sqlx::query_as(&sql)
            .bind(ns.render())
            .bind(limit_i64)
            .bind(offset_i64)
            .fetch_all(&mut *tx)
            .await
            .map_err(into_core_sqlx)?;
        tx.commit().await.map_err(into_core_sqlx)?;
        rows.into_iter()
            .map(|(id, payload)| {
                let node: N = serde_json::from_value(payload).map_err(into_core_codec)?;
                Ok((NodeId::from_string(id), node))
            })
            .collect()
    }

    /// Paginated edge-id enumeration. Operator-side migration path.
    pub async fn list_edges(
        &self,
        ns: &Namespace,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<EdgeId>> {
        let sql = format!(
            "SELECT id FROM {} \
             WHERE namespace_key = $1 \
             ORDER BY id ASC \
             LIMIT $2 OFFSET $3",
            self.edges_table
        );
        let limit_i64 = i64::try_from(limit).unwrap_or(i64::MAX);
        let offset_i64 = i64::try_from(offset).unwrap_or(i64::MAX);
        let mut tx = self.pool.begin().await.map_err(into_core_sqlx)?;
        set_tenant_session(&mut *tx, ns.tenant_id()).await?;
        let rows: Vec<(String,)> = sqlx::query_as(&sql)
            .bind(ns.render())
            .bind(limit_i64)
            .bind(offset_i64)
            .fetch_all(&mut *tx)
            .await
            .map_err(into_core_sqlx)?;
        tx.commit().await.map_err(into_core_sqlx)?;
        Ok(rows
            .into_iter()
            .map(|(id,)| EdgeId::from_string(id))
            .collect())
    }

    /// Paginated `GraphHop<E>` enumeration — full structural body
    /// in one round-trip. Operator-side bulk-export path.
    pub async fn list_edge_records(
        &self,
        ns: &Namespace,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<GraphHop<E>>>
    where
        E: DeserializeOwned,
    {
        let sql = format!(
            "SELECT id, from_node, to_node, payload, ts FROM {} \
             WHERE namespace_key = $1 \
             ORDER BY id ASC \
             LIMIT $2 OFFSET $3",
            self.edges_table
        );
        let limit_i64 = i64::try_from(limit).unwrap_or(i64::MAX);
        let offset_i64 = i64::try_from(offset).unwrap_or(i64::MAX);
        let mut tx = self.pool.begin().await.map_err(into_core_sqlx)?;
        set_tenant_session(&mut *tx, ns.tenant_id()).await?;
        let rows: Vec<(String, String, String, Value, DateTime<Utc>)> = sqlx::query_as(&sql)
            .bind(ns.render())
            .bind(limit_i64)
            .bind(offset_i64)
            .fetch_all(&mut *tx)
            .await
            .map_err(into_core_sqlx)?;
        tx.commit().await.map_err(into_core_sqlx)?;
        rows.into_iter()
            .map(|(id, fr, to_n, payload, ts)| {
                let edge: E = serde_json::from_value(payload).map_err(into_core_codec)?;
                Ok(GraphHop::new(
                    EdgeId::from_string(id),
                    NodeId::from_string(fr),
                    NodeId::from_string(to_n),
                    edge,
                    ts,
                ))
            })
            .collect()
    }

    /// Drop every node with no incident edge — single SQL anti-join
    /// against the edges table. Two-phase prune companion to
    /// `prune_older_than`: the edge sweep leaves orphans that this
    /// call cleans up. Operator-side admin path.
    pub async fn prune_orphan_nodes(&self, ns: &Namespace) -> Result<usize> {
        let sql = format!(
            "DELETE FROM {nodes} \
             WHERE namespace_key = $1 \
               AND id NOT IN ( \
                   SELECT from_node FROM {edges} WHERE namespace_key = $1 \
                   UNION \
                   SELECT to_node FROM {edges} WHERE namespace_key = $1 \
               )",
            nodes = self.nodes_table,
            edges = self.edges_table
        );
        let mut tx = self.pool.begin().await.map_err(into_core_sqlx)?;
        set_tenant_session(&mut *tx, ns.tenant_id()).await?;
        let result = sqlx::query(&sql)
            .bind(ns.render())
            .execute(&mut *tx)
            .await
            .map_err(into_core_sqlx)?;
        tx.commit().await.map_err(into_core_sqlx)?;
        Ok(usize::try_from(result.rows_affected()).unwrap_or(usize::MAX))
    }
}

/// Fluent builder for [`PgGraphMemory`]. Use
/// [`PgGraphMemory::builder`].
#[must_use]
pub struct PgGraphMemoryBuilder<N, E> {
    url: Option<String>,
    pool: Option<Arc<PgPool>>,
    nodes_table: String,
    edges_table: String,
    auto_migrate: bool,
    _phantom: PhantomData<fn() -> (N, E)>,
}

impl<N, E> Default for PgGraphMemoryBuilder<N, E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N, E> PgGraphMemoryBuilder<N, E> {
    /// Empty builder.
    pub fn new() -> Self {
        Self {
            url: None,
            pool: None,
            nodes_table: DEFAULT_NODES_TABLE.to_owned(),
            edges_table: DEFAULT_EDGES_TABLE.to_owned(),
            auto_migrate: true,
            _phantom: PhantomData,
        }
    }

    /// Postgres connection string. Mutually exclusive with
    /// [`Self::with_pool`] — the builder rejects construction if
    /// both are set.
    pub fn with_connection_string(mut self, url: impl Into<String>) -> Self {
        self.url = Some(url.into());
        self
    }

    /// Reuse an existing pool — useful when the operator already
    /// manages a `PgPool` for other persistence layers and wants
    /// `PgGraphMemory` to share it.
    pub fn with_pool(mut self, pool: Arc<PgPool>) -> Self {
        self.pool = Some(pool);
        self
    }

    /// Override the nodes table name (default `graph_nodes`).
    pub fn with_nodes_table(mut self, name: impl Into<String>) -> Self {
        self.nodes_table = name.into();
        self
    }

    /// Override the edges table name (default `graph_edges`).
    pub fn with_edges_table(mut self, name: impl Into<String>) -> Self {
        self.edges_table = name.into();
        self
    }

    /// Toggle the idempotent schema bootstrap. Default `true`;
    /// set to `false` when the schema is owned externally.
    pub const fn with_auto_migrate(mut self, on: bool) -> Self {
        self.auto_migrate = on;
        self
    }

    /// Open the pool (if needed), run the migration (if enabled),
    /// and return the configured backend.
    pub async fn build(self) -> PgGraphMemoryResult<PgGraphMemory<N, E>> {
        let pool = match (self.pool, self.url) {
            (Some(_), Some(_)) => {
                return Err(PgGraphMemoryError::Config(
                    "set with_pool() OR with_connection_string(), not both".into(),
                ));
            }
            (Some(pool), None) => pool,
            (None, Some(url)) => Arc::new(PgPoolOptions::new().connect(&url).await?),
            (None, None) => {
                return Err(PgGraphMemoryError::Config(
                    "with_pool() or with_connection_string() is required".into(),
                ));
            }
        };
        if self.auto_migrate {
            bootstrap(&pool, &self.nodes_table, &self.edges_table).await?;
        }
        Ok(PgGraphMemory {
            pool,
            nodes_table: Arc::from(self.nodes_table),
            edges_table: Arc::from(self.edges_table),
            _phantom: PhantomData,
        })
    }
}

#[async_trait]
impl<N, E> GraphMemory<N, E> for PgGraphMemory<N, E>
where
    N: Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
    E: Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
{
    async fn add_node(&self, _ctx: &ExecutionContext, ns: &Namespace, node: N) -> Result<NodeId> {
        let id = NodeId::new();
        let payload = serde_json::to_value(&node).map_err(into_core_codec)?;
        let sql = format!(
            "INSERT INTO {} (tenant_id, namespace_key, id, payload) \
             VALUES ($1, $2, $3, $4)",
            self.nodes_table
        );
        let mut tx = self.pool.begin().await.map_err(into_core_sqlx)?;
        set_tenant_session(&mut *tx, ns.tenant_id()).await?;
        sqlx::query(&sql)
            .bind(ns.tenant_id().as_str())
            .bind(ns.render())
            .bind(id.as_str())
            .bind(&payload)
            .execute(&mut *tx)
            .await
            .map_err(into_core_sqlx)?;
        tx.commit().await.map_err(into_core_sqlx)?;
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
        let payload = serde_json::to_value(&edge).map_err(into_core_codec)?;
        let sql = format!(
            "INSERT INTO {} (tenant_id, namespace_key, id, from_node, to_node, payload, ts) \
             VALUES ($1, $2, $3, $4, $5, $6, $7)",
            self.edges_table
        );
        let mut tx = self.pool.begin().await.map_err(into_core_sqlx)?;
        set_tenant_session(&mut *tx, ns.tenant_id()).await?;
        sqlx::query(&sql)
            .bind(ns.tenant_id().as_str())
            .bind(ns.render())
            .bind(id.as_str())
            .bind(from.as_str())
            .bind(to.as_str())
            .bind(&payload)
            .bind(timestamp)
            .execute(&mut *tx)
            .await
            .map_err(into_core_sqlx)?;
        tx.commit().await.map_err(into_core_sqlx)?;
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
        // Pre-allocate per-column arrays. Postgres' UNNEST takes one
        // array per column and zips them row-wise — N edges become
        // one INSERT … SELECT FROM UNNEST(…), one round-trip
        // regardless of N.
        let count = edges.len();
        let mut ids: Vec<EdgeId> = Vec::with_capacity(count);
        let mut id_strings: Vec<String> = Vec::with_capacity(count);
        let mut from_strings: Vec<String> = Vec::with_capacity(count);
        let mut to_strings: Vec<String> = Vec::with_capacity(count);
        let mut payloads: Vec<Value> = Vec::with_capacity(count);
        let mut timestamps: Vec<DateTime<Utc>> = Vec::with_capacity(count);
        for (from, to, payload, ts) in edges {
            let id = EdgeId::new();
            id_strings.push(id.as_str().to_owned());
            from_strings.push(from.as_str().to_owned());
            to_strings.push(to.as_str().to_owned());
            payloads.push(serde_json::to_value(&payload).map_err(into_core_codec)?);
            timestamps.push(ts);
            ids.push(id);
        }
        let sql = format!(
            "INSERT INTO {} (tenant_id, namespace_key, id, from_node, to_node, payload, ts) \
             SELECT $1, $2, e.id, e.from_node, e.to_node, e.payload, e.ts \
             FROM UNNEST($3::TEXT[], $4::TEXT[], $5::TEXT[], $6::JSONB[], $7::TIMESTAMPTZ[]) \
                  AS e(id, from_node, to_node, payload, ts)",
            self.edges_table
        );
        let mut tx = self.pool.begin().await.map_err(into_core_sqlx)?;
        set_tenant_session(&mut *tx, ns.tenant_id()).await?;
        sqlx::query(&sql)
            .bind(ns.tenant_id().as_str())
            .bind(ns.render())
            .bind(&id_strings)
            .bind(&from_strings)
            .bind(&to_strings)
            .bind(&payloads)
            .bind(&timestamps)
            .execute(&mut *tx)
            .await
            .map_err(into_core_sqlx)?;
        tx.commit().await.map_err(into_core_sqlx)?;
        Ok(ids)
    }

    async fn node(
        &self,
        _ctx: &ExecutionContext,
        ns: &Namespace,
        id: &NodeId,
    ) -> Result<Option<N>> {
        let sql = format!(
            "SELECT payload FROM {} WHERE namespace_key = $1 AND id = $2",
            self.nodes_table
        );
        let mut tx = self.pool.begin().await.map_err(into_core_sqlx)?;
        set_tenant_session(&mut *tx, ns.tenant_id()).await?;
        let row: Option<(Value,)> = sqlx::query_as(&sql)
            .bind(ns.render())
            .bind(id.as_str())
            .fetch_optional(&mut *tx)
            .await
            .map_err(into_core_sqlx)?;
        tx.commit().await.map_err(into_core_sqlx)?;
        row.map(|(p,)| serde_json::from_value(p).map_err(into_core_codec))
            .transpose()
    }

    async fn edge(
        &self,
        _ctx: &ExecutionContext,
        ns: &Namespace,
        edge_id: &EdgeId,
    ) -> Result<Option<GraphHop<E>>> {
        let sql = format!(
            "SELECT from_node, to_node, payload, ts FROM {} \
             WHERE namespace_key = $1 AND id = $2",
            self.edges_table
        );
        let mut tx = self.pool.begin().await.map_err(into_core_sqlx)?;
        set_tenant_session(&mut *tx, ns.tenant_id()).await?;
        let row: Option<(String, String, Value, DateTime<Utc>)> = sqlx::query_as(&sql)
            .bind(ns.render())
            .bind(edge_id.as_str())
            .fetch_optional(&mut *tx)
            .await
            .map_err(into_core_sqlx)?;
        tx.commit().await.map_err(into_core_sqlx)?;
        row.map(|(fr, to_n, payload, ts)| {
            let edge: E = serde_json::from_value(payload).map_err(into_core_codec)?;
            Ok(GraphHop::new(
                edge_id.clone(),
                NodeId::from_string(fr),
                NodeId::from_string(to_n),
                edge,
                ts,
            ))
        })
        .transpose()
    }

    async fn neighbors(
        &self,
        _ctx: &ExecutionContext,
        ns: &Namespace,
        node: &NodeId,
        direction: Direction,
    ) -> Result<Vec<(EdgeId, NodeId, E)>> {
        let mut tx = self.pool.begin().await.map_err(into_core_sqlx)?;
        set_tenant_session(&mut *tx, ns.tenant_id()).await?;
        let rows = fetch_neighbours(&mut *tx, &self.edges_table, ns, node, direction).await?;
        tx.commit().await.map_err(into_core_sqlx)?;
        rows.into_iter()
            .map(|row| {
                let payload: E = serde_json::from_value(row.payload).map_err(into_core_codec)?;
                Ok((row.id, row.neighbour, payload))
            })
            .collect()
    }

    async fn traverse(
        &self,
        _ctx: &ExecutionContext,
        ns: &Namespace,
        start: &NodeId,
        direction: Direction,
        max_depth: usize,
    ) -> Result<Vec<GraphHop<E>>> {
        traverse_recursive(self, ns, start, direction, max_depth).await
    }

    async fn find_path(
        &self,
        _ctx: &ExecutionContext,
        ns: &Namespace,
        from: &NodeId,
        to: &NodeId,
        direction: Direction,
        max_depth: usize,
    ) -> Result<Option<Vec<GraphHop<E>>>> {
        if from == to {
            return Ok(Some(Vec::new()));
        }
        find_path_recursive(self, ns, from, to, direction, max_depth).await
    }

    async fn temporal_filter(
        &self,
        _ctx: &ExecutionContext,
        ns: &Namespace,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    ) -> Result<Vec<GraphHop<E>>> {
        let sql = format!(
            "SELECT id, from_node, to_node, payload, ts \
             FROM {} \
             WHERE namespace_key = $1 AND ts >= $2 AND ts < $3 \
             ORDER BY ts ASC",
            self.edges_table
        );
        let mut tx = self.pool.begin().await.map_err(into_core_sqlx)?;
        set_tenant_session(&mut *tx, ns.tenant_id()).await?;
        let rows: Vec<(String, String, String, Value, DateTime<Utc>)> = sqlx::query_as(&sql)
            .bind(ns.render())
            .bind(from)
            .bind(to)
            .fetch_all(&mut *tx)
            .await
            .map_err(into_core_sqlx)?;
        tx.commit().await.map_err(into_core_sqlx)?;
        rows.into_iter()
            .map(|(id, fr, to_n, payload, ts)| {
                let edge: E = serde_json::from_value(payload).map_err(into_core_codec)?;
                Ok(GraphHop::new(
                    EdgeId::from_string(id),
                    NodeId::from_string(fr),
                    NodeId::from_string(to_n),
                    edge,
                    ts,
                ))
            })
            .collect()
    }

    async fn node_count(&self, _ctx: &ExecutionContext, ns: &Namespace) -> Result<usize> {
        let sql = format!(
            "SELECT COUNT(*) FROM {} WHERE namespace_key = $1",
            self.nodes_table
        );
        let mut tx = self.pool.begin().await.map_err(into_core_sqlx)?;
        set_tenant_session(&mut *tx, ns.tenant_id()).await?;
        let row: (i64,) = sqlx::query_as(&sql)
            .bind(ns.render())
            .fetch_one(&mut *tx)
            .await
            .map_err(into_core_sqlx)?;
        tx.commit().await.map_err(into_core_sqlx)?;
        Ok(usize::try_from(row.0.max(0)).unwrap_or(usize::MAX))
    }

    async fn edge_count(&self, _ctx: &ExecutionContext, ns: &Namespace) -> Result<usize> {
        let sql = format!(
            "SELECT COUNT(*) FROM {} WHERE namespace_key = $1",
            self.edges_table
        );
        let mut tx = self.pool.begin().await.map_err(into_core_sqlx)?;
        set_tenant_session(&mut *tx, ns.tenant_id()).await?;
        let row: (i64,) = sqlx::query_as(&sql)
            .bind(ns.render())
            .fetch_one(&mut *tx)
            .await
            .map_err(into_core_sqlx)?;
        tx.commit().await.map_err(into_core_sqlx)?;
        Ok(usize::try_from(row.0.max(0)).unwrap_or(usize::MAX))
    }

    async fn delete_edge(
        &self,
        _ctx: &ExecutionContext,
        ns: &Namespace,
        edge_id: &EdgeId,
    ) -> Result<()> {
        let sql = format!(
            "DELETE FROM {} WHERE namespace_key = $1 AND id = $2",
            self.edges_table
        );
        let mut tx = self.pool.begin().await.map_err(into_core_sqlx)?;
        set_tenant_session(&mut *tx, ns.tenant_id()).await?;
        sqlx::query(&sql)
            .bind(ns.render())
            .bind(edge_id.as_str())
            .execute(&mut *tx)
            .await
            .map_err(into_core_sqlx)?;
        tx.commit().await.map_err(into_core_sqlx)?;
        Ok(())
    }

    async fn delete_node(
        &self,
        _ctx: &ExecutionContext,
        ns: &Namespace,
        node_id: &NodeId,
    ) -> Result<usize> {
        // Cascade — drop incident edges first, then the node
        // itself, in one tenant-stamped transaction so a
        // concurrent reader never sees a half-applied state
        // (an edge whose endpoint node is gone, or vice versa).
        let edges_sql = format!(
            "DELETE FROM {} \
             WHERE namespace_key = $1 AND (from_node = $2 OR to_node = $2)",
            self.edges_table
        );
        let nodes_sql = format!(
            "DELETE FROM {} WHERE namespace_key = $1 AND id = $2",
            self.nodes_table
        );
        let mut tx = self.pool.begin().await.map_err(into_core_sqlx)?;
        set_tenant_session(&mut *tx, ns.tenant_id()).await?;
        let edge_result = sqlx::query(&edges_sql)
            .bind(ns.render())
            .bind(node_id.as_str())
            .execute(&mut *tx)
            .await
            .map_err(into_core_sqlx)?;
        sqlx::query(&nodes_sql)
            .bind(ns.render())
            .bind(node_id.as_str())
            .execute(&mut *tx)
            .await
            .map_err(into_core_sqlx)?;
        tx.commit().await.map_err(into_core_sqlx)?;
        Ok(usize::try_from(edge_result.rows_affected()).unwrap_or(usize::MAX))
    }

    async fn prune_older_than(
        &self,
        _ctx: &ExecutionContext,
        ns: &Namespace,
        ttl: std::time::Duration,
    ) -> Result<usize> {
        // chrono::Duration is signed and uses i64 nanoseconds; for
        // pathological ttls (above i64::MAX seconds) saturate to
        // chrono::Duration::MAX so the cutoff stays in the past.
        let cutoff = Utc::now() - chrono::Duration::from_std(ttl).unwrap_or(chrono::Duration::MAX);
        let sql = format!(
            "DELETE FROM {} WHERE namespace_key = $1 AND ts < $2",
            self.edges_table
        );
        let mut tx = self.pool.begin().await.map_err(into_core_sqlx)?;
        set_tenant_session(&mut *tx, ns.tenant_id()).await?;
        let result = sqlx::query(&sql)
            .bind(ns.render())
            .bind(cutoff)
            .execute(&mut *tx)
            .await
            .map_err(into_core_sqlx)?;
        tx.commit().await.map_err(into_core_sqlx)?;
        Ok(usize::try_from(result.rows_affected()).unwrap_or(usize::MAX))
    }
}

/// One row decoded from the `neighbours` projection. `neighbour`
/// is whichever endpoint *isn't* the queried node (for
/// `Direction::Both`, we project both — see [`fetch_neighbours`]).
struct NeighbourRow {
    id: EdgeId,
    neighbour: NodeId,
    payload: Value,
}

async fn fetch_neighbours<'e, E>(
    executor: E,
    edges_table: &str,
    ns: &Namespace,
    node: &NodeId,
    direction: Direction,
) -> Result<Vec<NeighbourRow>>
where
    E: sqlx::Executor<'e, Database = sqlx::Postgres>,
{
    let dir = direction_sql(direction)?;
    let sql = format!(
        "SELECT id, {next_node} AS neighbour, payload \
         FROM {edges_table} \
         WHERE namespace_key = $1 AND {join_pred}",
        next_node = dir.flat_next_node,
        join_pred = dir.flat_join_predicate,
    );
    let rows: Vec<(String, String, Value)> = sqlx::query_as(&sql)
        .bind(ns.render())
        .bind(node.as_str())
        .fetch_all(executor)
        .await
        .map_err(into_core_sqlx)?;
    Ok(rows
        .into_iter()
        .map(|(id, neighbour, payload)| NeighbourRow {
            id: EdgeId::from_string(id),
            neighbour: NodeId::from_string(neighbour),
            payload,
        })
        .collect())
}

/// SQL fragments parameterised on [`Direction`]. `recursive_*`
/// variants reference `w.frontier` (the current row of the
/// recursive CTE); `flat_*` variants reference the bound parameter
/// `$2` (the seed node) for one-shot neighbour lookups.
struct DirectionSql {
    recursive_join_predicate: &'static str,
    recursive_next_node: &'static str,
    flat_join_predicate: &'static str,
    flat_next_node: &'static str,
}

/// `Direction` is `#[non_exhaustive]`. A future variant added
/// upstream surfaces as a typed encode-time rejection so the
/// backend never silently approximates an unknown traversal
/// semantic with one of the existing arms (invariant #15).
fn direction_sql(direction: Direction) -> Result<DirectionSql> {
    match direction {
        Direction::Outgoing => Ok(DirectionSql {
            recursive_join_predicate: "e.from_node = w.frontier",
            recursive_next_node: "e.to_node",
            flat_join_predicate: "from_node = $2",
            flat_next_node: "to_node",
        }),
        Direction::Incoming => Ok(DirectionSql {
            recursive_join_predicate: "e.to_node = w.frontier",
            recursive_next_node: "e.from_node",
            flat_join_predicate: "to_node = $2",
            flat_next_node: "from_node",
        }),
        Direction::Both => Ok(DirectionSql {
            recursive_join_predicate: "(e.from_node = w.frontier OR e.to_node = w.frontier)",
            recursive_next_node: "CASE WHEN e.from_node = w.frontier THEN e.to_node ELSE e.from_node END",
            flat_join_predicate: "(from_node = $2 OR to_node = $2)",
            flat_next_node: "CASE WHEN from_node = $2 THEN to_node ELSE from_node END",
        }),
        other => Err(entelix_core::error::Error::invalid_request(format!(
            "PgGraphMemory: unsupported Direction variant {other:?}"
        ))),
    }
}

/// Single-round-trip BFS via `WITH RECURSIVE`. Postgres expands
/// the frontier server-side, carrying a `visited` array per row
/// for cycle prevention. The outer query keeps the first arrival
/// at each destination node (BFS shortest-depth wins), preserving
/// the dedupe semantics of the original layer-by-layer Rust BFS.
async fn traverse_recursive<N, E>(
    g: &PgGraphMemory<N, E>,
    ns: &Namespace,
    start: &NodeId,
    direction: Direction,
    max_depth: usize,
) -> Result<Vec<GraphHop<E>>>
where
    N: Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
    E: Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
{
    let dir = direction_sql(direction)?;
    let max_depth_i32 = i32::try_from(max_depth).unwrap_or(i32::MAX);
    let sql = format!(
        "WITH RECURSIVE walk(edge_id, edge_from, edge_to, payload, ts, frontier, depth, visited) AS ( \
            SELECT NULL::TEXT, NULL::TEXT, NULL::TEXT, NULL::JSONB, NULL::TIMESTAMPTZ, \
                   $2::TEXT, 0, ARRAY[$2::TEXT] \
            UNION ALL \
            SELECT e.id, e.from_node, e.to_node, e.payload, e.ts, \
                   {next_node}, w.depth + 1, w.visited || ({next_node}) \
            FROM walk w \
            JOIN {edges_table} e ON e.namespace_key = $1 AND {join_pred} \
            WHERE w.depth < $3 AND NOT (({next_node}) = ANY(w.visited)) \
        ), \
        ranked AS ( \
            SELECT edge_id, edge_from, edge_to, payload, ts, depth, \
                   ROW_NUMBER() OVER ( \
                       PARTITION BY frontier ORDER BY depth ASC, edge_id ASC \
                   ) AS rn \
            FROM walk WHERE depth > 0 \
        ) \
        SELECT edge_id, edge_from, edge_to, payload, ts \
        FROM ranked WHERE rn = 1 \
        ORDER BY depth ASC, edge_id ASC",
        edges_table = g.edges_table,
        join_pred = dir.recursive_join_predicate,
        next_node = dir.recursive_next_node,
    );
    let mut tx = g.pool.begin().await.map_err(into_core_sqlx)?;
    set_tenant_session(&mut *tx, ns.tenant_id()).await?;
    let rows: Vec<(String, String, String, Value, DateTime<Utc>)> = sqlx::query_as(&sql)
        .bind(ns.render())
        .bind(start.as_str())
        .bind(max_depth_i32)
        .fetch_all(&mut *tx)
        .await
        .map_err(into_core_sqlx)?;
    tx.commit().await.map_err(into_core_sqlx)?;
    rows.into_iter()
        .map(|(eid, fr, to_n, payload, ts)| {
            let edge: E = serde_json::from_value(payload).map_err(into_core_codec)?;
            Ok(GraphHop::new(
                EdgeId::from_string(eid),
                NodeId::from_string(fr),
                NodeId::from_string(to_n),
                edge,
                ts,
            ))
        })
        .collect()
}

/// Single-round-trip shortest-path via `WITH RECURSIVE`. The
/// recursive CTE accumulates an `edge_path` array per row; the
/// outer query picks the shortest path that reached `to` and
/// rejoins the unrolled edge ids back to the edges table for
/// payload + endpoint reconstruction. Returns `Ok(None)` when no
/// path within `max_depth` exists; caller handles `from == to` as
/// `Ok(Some(Vec::new()))` ahead of this call.
async fn find_path_recursive<N, E>(
    g: &PgGraphMemory<N, E>,
    ns: &Namespace,
    from: &NodeId,
    to: &NodeId,
    direction: Direction,
    max_depth: usize,
) -> Result<Option<Vec<GraphHop<E>>>>
where
    N: Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
    E: Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
{
    let dir = direction_sql(direction)?;
    let max_depth_i32 = i32::try_from(max_depth).unwrap_or(i32::MAX);
    let sql = format!(
        "WITH RECURSIVE walk(frontier, depth, visited, edge_path) AS ( \
            SELECT $2::TEXT, 0, ARRAY[$2::TEXT]::TEXT[], ARRAY[]::TEXT[] \
            UNION ALL \
            SELECT {next_node}, w.depth + 1, \
                   w.visited || ({next_node}), w.edge_path || e.id \
            FROM walk w \
            JOIN {edges_table} e ON e.namespace_key = $1 AND {join_pred} \
            WHERE w.depth < $4 AND w.frontier <> $3 \
              AND NOT (({next_node}) = ANY(w.visited)) \
        ), \
        shortest AS ( \
            SELECT edge_path FROM walk \
            WHERE frontier = $3 AND depth > 0 \
            ORDER BY depth ASC LIMIT 1 \
        ), \
        unrolled AS ( \
            SELECT u.eid, u.ord \
            FROM shortest s, unnest(s.edge_path) WITH ORDINALITY AS u(eid, ord) \
        ) \
        SELECT e.id, e.from_node, e.to_node, e.payload, e.ts \
        FROM unrolled u \
        JOIN {edges_table} e ON e.namespace_key = $1 AND e.id = u.eid \
        ORDER BY u.ord ASC",
        edges_table = g.edges_table,
        join_pred = dir.recursive_join_predicate,
        next_node = dir.recursive_next_node,
    );
    let mut tx = g.pool.begin().await.map_err(into_core_sqlx)?;
    set_tenant_session(&mut *tx, ns.tenant_id()).await?;
    let rows: Vec<(String, String, String, Value, DateTime<Utc>)> = sqlx::query_as(&sql)
        .bind(ns.render())
        .bind(from.as_str())
        .bind(to.as_str())
        .bind(max_depth_i32)
        .fetch_all(&mut *tx)
        .await
        .map_err(into_core_sqlx)?;
    tx.commit().await.map_err(into_core_sqlx)?;
    if rows.is_empty() {
        return Ok(None);
    }
    let hops: Vec<GraphHop<E>> = rows
        .into_iter()
        .map(|(eid, fr, to_n, payload, ts)| {
            let edge: E = serde_json::from_value(payload).map_err(into_core_codec)?;
            Ok(GraphHop::new(
                EdgeId::from_string(eid),
                NodeId::from_string(fr),
                NodeId::from_string(to_n),
                edge,
                ts,
            ))
        })
        .collect::<Result<_>>()?;
    Ok(Some(hops))
}

fn into_core_sqlx(e: sqlx::Error) -> entelix_core::error::Error {
    PgGraphMemoryError::from(e).into()
}

fn into_core_codec(e: serde_json::Error) -> entelix_core::error::Error {
    PgGraphMemoryError::from(e).into()
}
