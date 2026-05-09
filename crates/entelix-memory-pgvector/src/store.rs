//! `PgVectorStore` — concrete `VectorStore` over Postgres + pgvector.
//!
//! Single-table design: `(namespace_key, doc_id)` composite primary
//! key + `embedding VECTOR(N)` + `metadata JSONB`. The composite PK
//! doubles as the namespace anchor index, so every read / write /
//! count / list rides a B-tree probe before the vector / GIN index
//! ever sees a row. Cross-tenant data leak is structurally
//! impossible — the namespace anchor is mandatory in every query.

use std::sync::Arc;

use async_trait::async_trait;
use pgvector::Vector;
use serde_json::Value;
use sqlx::{PgPool, Postgres, QueryBuilder, Row};
use uuid::Uuid;

use entelix_core::context::ExecutionContext;
use entelix_core::error::{Error, Result};
use entelix_memory::{Document, Namespace, VectorFilter, VectorStore};

use crate::error::{PgVectorStoreError, PgVectorStoreResult};
use crate::filter::append_where;
use crate::migration;
use crate::tenant::set_tenant_session;

/// Distance metric used for vector similarity. Mirrors pgvector's
/// own taxonomy 1:1 — operators familiar with `<=>` / `<->` /
/// `<#>` pick the metric they would have picked there.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
#[non_exhaustive]
pub enum DistanceMetric {
    /// Cosine similarity (`<=>` operator). The right default for
    /// normalized embeddings (`text-embedding-3-*`, etc.).
    #[default]
    Cosine,
    /// Euclidean / L2 distance (`<->` operator).
    L2,
    /// Inner product (`<#>` operator). Note: pgvector's `<#>`
    /// returns the *negative* inner product so smaller is "more
    /// similar"; the store inverts it on read so caller-facing
    /// scores stay "higher = better".
    InnerProduct,
}

/// ANN index kind. HNSW is the production default; IVFFlat is
/// selected when build time matters more than query latency.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
#[non_exhaustive]
pub enum IndexKind {
    /// Hierarchical Navigable Small World — pgvector's HNSW. Best
    /// recall / throughput trade-off for ≤10M vectors per
    /// namespace.
    #[default]
    Hnsw,
    /// IVF-Flat — fast build, lower memory at the cost of recall.
    /// Operators must `SET ivfflat.probes = N` per session for
    /// query-time recall tuning.
    IvfFlat,
}

/// Concrete [`VectorStore`] backed by Postgres + pgvector.
///
/// Cloning is cheap — the pool is `Arc`-shared internally.
#[derive(Clone)]
pub struct PgVectorStore {
    pool: PgPool,
    table: Arc<str>,
    dimension: usize,
    distance: DistanceMetric,
}

impl std::fmt::Debug for PgVectorStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PgVectorStore")
            .field("table", &self.table)
            .field("dimension", &self.dimension)
            .field("distance", &self.distance)
            .finish_non_exhaustive()
    }
}

impl PgVectorStore {
    /// Begin building a [`PgVectorStore`].
    pub fn builder(dimension: usize) -> PgVectorStoreBuilder {
        PgVectorStoreBuilder::new(dimension)
    }

    fn distance_op(&self) -> &'static str {
        match self.distance {
            DistanceMetric::Cosine => "<=>",
            DistanceMetric::L2 => "<->",
            DistanceMetric::InnerProduct => "<#>",
        }
    }

    /// Convert pgvector's distance into a "higher = better"
    /// similarity score in `[0.0, 1.0]` for cosine / L2 (best
    /// effort) and the negated inner product for ip metric.
    /// Comparable only within a single query result set.
    fn distance_to_score(&self, distance: f64) -> f32 {
        let s = match self.distance {
            DistanceMetric::Cosine => 1.0 - distance,
            DistanceMetric::L2 => 1.0 / (1.0 + distance),
            // pgvector's `<#>` returns negative inner product;
            // `-distance` recovers the operator-facing similarity.
            DistanceMetric::InnerProduct => -distance,
        };
        s as f32
    }
}

/// Builder for [`PgVectorStore`].
#[must_use]
pub struct PgVectorStoreBuilder {
    table: String,
    dimension: usize,
    distance: DistanceMetric,
    index_kind: IndexKind,
    auto_migrate: bool,
    connection_string: Option<String>,
    pool: Option<PgPool>,
    max_connections: u32,
}

impl PgVectorStoreBuilder {
    fn new(dimension: usize) -> Self {
        Self {
            table: "entelix_vectors".into(),
            dimension,
            distance: DistanceMetric::default(),
            index_kind: IndexKind::default(),
            auto_migrate: true,
            connection_string: None,
            pool: None,
            max_connections: 10,
        }
    }

    /// Override the table name. Defaults to `entelix_vectors`.
    /// Must satisfy SQL-identifier rules
    /// (`[a-zA-Z_][a-zA-Z0-9_]{0,62}`).
    pub fn with_table(mut self, table: impl Into<String>) -> Self {
        self.table = table.into();
        self
    }

    /// Override the distance metric. Defaults to
    /// [`DistanceMetric::Cosine`].
    pub const fn with_distance(mut self, distance: DistanceMetric) -> Self {
        self.distance = distance;
        self
    }

    /// Override the ANN index kind. Defaults to
    /// [`IndexKind::Hnsw`].
    pub const fn with_index_kind(mut self, kind: IndexKind) -> Self {
        self.index_kind = kind;
        self
    }

    /// Disable the automatic schema bootstrap.
    ///
    /// Pass `false` when the table + extension + indexes are
    /// provisioned externally (DBA-managed, IaC, migration
    /// pipeline) and the store should consume an existing schema.
    /// Defaults to `true`.
    pub const fn with_auto_migrate(mut self, auto: bool) -> Self {
        self.auto_migrate = auto;
        self
    }

    /// Connect with a libpq-style connection string. Mutually
    /// exclusive with [`Self::with_pool`].
    pub fn with_connection_string(mut self, url: impl Into<String>) -> Self {
        self.connection_string = Some(url.into());
        self
    }

    /// Reuse an existing `PgPool`. Mutually exclusive with
    /// [`Self::with_connection_string`].
    pub fn with_pool(mut self, pool: PgPool) -> Self {
        self.pool = Some(pool);
        self
    }

    /// Override the pool's `max_connections` (when the builder
    /// constructs the pool). Ignored when [`Self::with_pool`]
    /// supplies a pre-built pool.
    pub const fn with_max_connections(mut self, max: u32) -> Self {
        self.max_connections = max;
        self
    }

    /// Finalize the builder. Connects (or adopts the supplied
    /// pool) and runs the schema bootstrap when
    /// `auto_migrate=true`.
    pub async fn build(self) -> PgVectorStoreResult<PgVectorStore> {
        let pool = match (self.pool, self.connection_string) {
            (Some(p), None) => p,
            (None, Some(url)) => {
                sqlx::postgres::PgPoolOptions::new()
                    .max_connections(self.max_connections)
                    .connect(&url)
                    .await?
            }
            (None, None) => {
                return Err(PgVectorStoreError::Config(
                    "either with_pool or with_connection_string is required".into(),
                ));
            }
            (Some(_), Some(_)) => {
                return Err(PgVectorStoreError::Config(
                    "with_pool and with_connection_string are mutually exclusive".into(),
                ));
            }
        };

        if self.auto_migrate {
            migration::bootstrap(
                &pool,
                &self.table,
                self.dimension,
                self.distance,
                self.index_kind,
            )
            .await?;
        }

        Ok(PgVectorStore {
            pool,
            table: self.table.into(),
            dimension: self.dimension,
            distance: self.distance,
        })
    }
}

#[async_trait]
impl VectorStore for PgVectorStore {
    fn dimension(&self) -> usize {
        self.dimension
    }

    async fn add(
        &self,
        ctx: &ExecutionContext,
        ns: &Namespace,
        document: Document,
        vector: Vec<f32>,
    ) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(Error::Cancelled);
        }
        if vector.len() != self.dimension {
            return Err(Error::invalid_request(format!(
                "PgVectorStore: vector dimension {} does not match \
                 index dimension {}",
                vector.len(),
                self.dimension
            )));
        }
        let ns_key = ns.render();
        let doc_id = document
            .doc_id
            .clone()
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let metadata = if document.metadata.is_null() {
            Value::Object(serde_json::Map::new())
        } else {
            document.metadata
        };
        let stmt = format!(
            "INSERT INTO {table} (tenant_id, namespace_key, doc_id, content, metadata, embedding) \
             VALUES ($1, $2, $3, $4, $5, $6) \
             ON CONFLICT (namespace_key, doc_id) DO UPDATE SET \
                 content = EXCLUDED.content, \
                 metadata = EXCLUDED.metadata, \
                 embedding = EXCLUDED.embedding",
            table = self.table
        );
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| Error::from(PgVectorStoreError::from(e)))?;
        set_tenant_session(&mut *tx, ns.tenant_id()).await?;
        sqlx::query(&stmt)
            .bind(ns.tenant_id().as_str())
            .bind(ns_key)
            .bind(doc_id)
            .bind(document.content)
            .bind(sqlx::types::Json(metadata))
            .bind(Vector::from(vector))
            .execute(&mut *tx)
            .await
            .map_err(|e| Error::from(PgVectorStoreError::from(e)))?;
        tx.commit()
            .await
            .map_err(|e| Error::from(PgVectorStoreError::from(e)))?;
        Ok(())
    }

    async fn add_batch(
        &self,
        ctx: &ExecutionContext,
        ns: &Namespace,
        items: Vec<(Document, Vec<f32>)>,
    ) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(Error::Cancelled);
        }
        if items.is_empty() {
            return Ok(());
        }
        let ns_key = ns.render();
        for (_, vector) in &items {
            if vector.len() != self.dimension {
                return Err(Error::invalid_request(format!(
                    "PgVectorStore: vector dimension {} does not match \
                     index dimension {}",
                    vector.len(),
                    self.dimension
                )));
            }
        }
        // Bulk insert via QueryBuilder::push_values — single round-trip.
        let tenant_id = ns.tenant_id().as_str().to_owned();
        let mut qb: QueryBuilder<'_, Postgres> = QueryBuilder::new(format!(
            "INSERT INTO {table} \
             (tenant_id, namespace_key, doc_id, content, metadata, embedding) ",
            table = self.table
        ));
        qb.push_values(items, |mut b, (mut document, vector)| {
            let doc_id = document
                .doc_id
                .take()
                .unwrap_or_else(|| Uuid::new_v4().to_string());
            let metadata = if document.metadata.is_null() {
                Value::Object(serde_json::Map::new())
            } else {
                document.metadata
            };
            b.push_bind(tenant_id.clone())
                .push_bind(ns_key.clone())
                .push_bind(doc_id)
                .push_bind(document.content)
                .push_bind(sqlx::types::Json(metadata))
                .push_bind(Vector::from(vector));
        });
        qb.push(
            " ON CONFLICT (namespace_key, doc_id) DO UPDATE SET \
                 content = EXCLUDED.content, \
                 metadata = EXCLUDED.metadata, \
                 embedding = EXCLUDED.embedding",
        );
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| Error::from(PgVectorStoreError::from(e)))?;
        set_tenant_session(&mut *tx, ns.tenant_id()).await?;
        qb.build()
            .execute(&mut *tx)
            .await
            .map_err(|e| Error::from(PgVectorStoreError::from(e)))?;
        tx.commit()
            .await
            .map_err(|e| Error::from(PgVectorStoreError::from(e)))?;
        Ok(())
    }

    async fn search(
        &self,
        ctx: &ExecutionContext,
        ns: &Namespace,
        query_vector: &[f32],
        top_k: usize,
    ) -> Result<Vec<Document>> {
        self.search_filtered(ctx, ns, query_vector, top_k, &VectorFilter::All)
            .await
    }

    async fn search_filtered(
        &self,
        ctx: &ExecutionContext,
        ns: &Namespace,
        query_vector: &[f32],
        top_k: usize,
        filter: &VectorFilter,
    ) -> Result<Vec<Document>> {
        if ctx.is_cancelled() {
            return Err(Error::Cancelled);
        }
        if query_vector.len() != self.dimension {
            return Err(Error::invalid_request(format!(
                "PgVectorStore: query dimension {} does not match \
                 index dimension {}",
                query_vector.len(),
                self.dimension
            )));
        }
        let ns_key = ns.render();

        // Postgres lets `ORDER BY <alias>` reference the SELECT
        // alias directly, so the query vector binds exactly once
        // — emitted into the SELECT distance expression and
        // reused by the ORDER BY through the `distance` alias.
        let mut qb: QueryBuilder<'_, Postgres> = QueryBuilder::new(format!(
            "SELECT doc_id, content, metadata, embedding {op} ",
            op = self.distance_op(),
        ));
        qb.push_bind(Vector::from(query_vector.to_vec()));
        qb.push(format!(" AS distance FROM {table}", table = self.table));
        append_where(&mut qb, &ns_key, Some(filter)).map_err(Error::from)?;
        qb.push(" ORDER BY distance LIMIT ");
        qb.push_bind(top_k as i64);

        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| Error::from(PgVectorStoreError::from(e)))?;
        set_tenant_session(&mut *tx, ns.tenant_id()).await?;
        let rows = qb
            .build()
            .fetch_all(&mut *tx)
            .await
            .map_err(|e| Error::from(PgVectorStoreError::from(e)))?;
        tx.commit()
            .await
            .map_err(|e| Error::from(PgVectorStoreError::from(e)))?;
        rows.into_iter()
            .map(|row| self.row_to_document(&row, true))
            .collect()
    }

    async fn delete(&self, ctx: &ExecutionContext, ns: &Namespace, doc_id: &str) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(Error::Cancelled);
        }
        let stmt = format!(
            "DELETE FROM {table} WHERE namespace_key = $1 AND doc_id = $2",
            table = self.table
        );
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| Error::from(PgVectorStoreError::from(e)))?;
        set_tenant_session(&mut *tx, ns.tenant_id()).await?;
        sqlx::query(&stmt)
            .bind(ns.render())
            .bind(doc_id.to_owned())
            .execute(&mut *tx)
            .await
            .map_err(|e| Error::from(PgVectorStoreError::from(e)))?;
        tx.commit()
            .await
            .map_err(|e| Error::from(PgVectorStoreError::from(e)))?;
        Ok(())
    }

    async fn update(
        &self,
        ctx: &ExecutionContext,
        ns: &Namespace,
        doc_id: &str,
        document: Document,
        vector: Vec<f32>,
    ) -> Result<()> {
        // `INSERT … ON CONFLICT … DO UPDATE` is atomic per-row, so
        // we override the trait's non-atomic delete-then-add
        // default via the same code path as `add`.
        let stored = Document {
            doc_id: Some(doc_id.to_owned()),
            ..document
        };
        self.add(ctx, ns, stored, vector).await
    }

    async fn count(
        &self,
        ctx: &ExecutionContext,
        ns: &Namespace,
        filter: Option<&VectorFilter>,
    ) -> Result<usize> {
        if ctx.is_cancelled() {
            return Err(Error::Cancelled);
        }
        let ns_key = ns.render();
        let mut qb: QueryBuilder<'_, Postgres> =
            QueryBuilder::new(format!("SELECT COUNT(*) FROM {table}", table = self.table));
        append_where(&mut qb, &ns_key, filter).map_err(Error::from)?;
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| Error::from(PgVectorStoreError::from(e)))?;
        set_tenant_session(&mut *tx, ns.tenant_id()).await?;
        let row = qb
            .build()
            .fetch_one(&mut *tx)
            .await
            .map_err(|e| Error::from(PgVectorStoreError::from(e)))?;
        tx.commit()
            .await
            .map_err(|e| Error::from(PgVectorStoreError::from(e)))?;
        let count: i64 = row.try_get::<i64, _>(0).map_err(|e| {
            Error::from(PgVectorStoreError::Malformed(format!(
                "COUNT(*) row missing expected column: {e}"
            )))
        })?;
        Ok(count.max(0) as usize)
    }

    async fn list(
        &self,
        ctx: &ExecutionContext,
        ns: &Namespace,
        filter: Option<&VectorFilter>,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<Document>> {
        if ctx.is_cancelled() {
            return Err(Error::Cancelled);
        }
        let ns_key = ns.render();
        let mut qb: QueryBuilder<'_, Postgres> = QueryBuilder::new(format!(
            "SELECT doc_id, content, metadata FROM {table}",
            table = self.table
        ));
        append_where(&mut qb, &ns_key, filter).map_err(Error::from)?;
        // Stable iteration order — `(namespace_key, doc_id)` is
        // the PK so the ordering is deterministic across calls.
        qb.push(" ORDER BY doc_id");
        qb.push(" LIMIT ");
        qb.push_bind(limit as i64);
        qb.push(" OFFSET ");
        qb.push_bind(offset as i64);
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| Error::from(PgVectorStoreError::from(e)))?;
        set_tenant_session(&mut *tx, ns.tenant_id()).await?;
        let rows = qb
            .build()
            .fetch_all(&mut *tx)
            .await
            .map_err(|e| Error::from(PgVectorStoreError::from(e)))?;
        tx.commit()
            .await
            .map_err(|e| Error::from(PgVectorStoreError::from(e)))?;
        rows.into_iter()
            .map(|row| self.row_to_document(&row, false))
            .collect()
    }
}

impl PgVectorStore {
    fn row_to_document(
        &self,
        row: &sqlx::postgres::PgRow,
        with_distance: bool,
    ) -> Result<Document> {
        let doc_id: String = row.try_get("doc_id").map_err(|e| {
            Error::from(PgVectorStoreError::Malformed(format!(
                "row missing doc_id: {e}"
            )))
        })?;
        let content: String = row.try_get("content").map_err(|e| {
            Error::from(PgVectorStoreError::Malformed(format!(
                "row missing content: {e}"
            )))
        })?;
        let metadata: sqlx::types::Json<Value> = row.try_get("metadata").map_err(|e| {
            Error::from(PgVectorStoreError::Malformed(format!(
                "row missing metadata: {e}"
            )))
        })?;
        let score = if with_distance {
            let distance: f64 = row.try_get("distance").map_err(|e| {
                Error::from(PgVectorStoreError::Malformed(format!(
                    "row missing distance: {e}"
                )))
            })?;
            Some(self.distance_to_score(distance))
        } else {
            None
        };
        Ok(Document {
            doc_id: Some(doc_id),
            content,
            metadata: metadata.0,
            score,
        })
    }
}
