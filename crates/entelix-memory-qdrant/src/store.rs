//! `QdrantVectorStore` — concrete `VectorStore` over qdrant 1.5+.
//!
//! Single-collection, multi-tenant via payload filter (qdrant
//! official multi-tenancy guidance). Every read / write / count /
//! list rides a `must`-clause anchored on
//! [`crate::filter::NAMESPACE_KEY`] so cross-tenant data leak is
//! structurally impossible (Invariant 11 / F2).
//!
//! `PointId` derivation — qdrant's internal id is a deterministic
//! `Uuid::new_v5(NAMESPACE_OID, "{namespace}:{doc_id}")` so two
//! tenants writing the same operator-facing `doc_id` are stored as
//! distinct points without coordination. The original `doc_id`
//! survives in the payload under [`crate::filter::DOC_ID_KEY`] so
//! operators round-trip through the operator-facing id.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use qdrant_client::Payload;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    CountPointsBuilder, CreateCollectionBuilder, DeletePointsBuilder, Distance, FieldType,
    PointStruct, PointsIdsList, ScrollPointsBuilder, SearchPointsBuilder, UpsertPointsBuilder,
    VectorParamsBuilder, points_selector::PointsSelectorOneOf,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use uuid::Uuid;

use entelix_core::context::ExecutionContext;
use entelix_core::error::{Error, Result};
use entelix_memory::{Document, Namespace, VectorFilter, VectorStore};

use crate::error::{QdrantStoreError, QdrantStoreResult};
use crate::filter::{self, CONTENT_KEY, DOC_ID_KEY, METADATA_KEY, NAMESPACE_KEY};

/// Distance metric used for vector similarity. Mirrors qdrant's
/// own taxonomy 1:1 — operators familiar with qdrant pick the
/// metric they would have picked there.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
#[non_exhaustive]
pub enum DistanceMetric {
    /// Cosine similarity — the right default for normalized
    /// embeddings (`text-embedding-3-*`, etc.).
    #[default]
    Cosine,
    /// Dot product (unnormalized).
    Dot,
    /// Euclidean (L2) distance.
    Euclidean,
}

impl From<DistanceMetric> for Distance {
    fn from(m: DistanceMetric) -> Self {
        match m {
            DistanceMetric::Cosine => Self::Cosine,
            DistanceMetric::Dot => Self::Dot,
            DistanceMetric::Euclidean => Self::Euclid,
        }
    }
}

/// Concrete [`VectorStore`] backed by a single qdrant collection.
///
/// Cloning is cheap — every internal state is behind an `Arc`.
#[derive(Clone)]
pub struct QdrantVectorStore {
    client: Arc<Qdrant>,
    collection: Arc<str>,
    dimension: usize,
}

impl std::fmt::Debug for QdrantVectorStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QdrantVectorStore")
            .field("collection", &self.collection)
            .field("dimension", &self.dimension)
            .finish_non_exhaustive()
    }
}

impl QdrantVectorStore {
    /// Begin building a [`QdrantVectorStore`].
    pub fn builder(collection: impl Into<String>, dimension: usize) -> QdrantVectorStoreBuilder {
        QdrantVectorStoreBuilder::new(collection, dimension)
    }

    /// Derive the qdrant `PointId` from `(namespace_key, doc_id)`
    /// via SHA-256 truncated to a 16-byte UUID. Two tenants writing
    /// the same operator-facing `doc_id` are stored as distinct
    /// points without coordination.
    fn point_id(namespace_key: &str, doc_id: &str) -> qdrant_client::qdrant::PointId {
        let mut hasher = Sha256::new();
        hasher.update(namespace_key.as_bytes());
        hasher.update(b":");
        hasher.update(doc_id.as_bytes());
        let digest = hasher.finalize();
        let mut bytes = [0u8; 16];
        bytes.copy_from_slice(&digest[..16]);
        Uuid::from_bytes(bytes).to_string().into()
    }

    fn build_payload(namespace_key: &str, doc_id: &str, document: &Document) -> Payload {
        let mut map = serde_json::Map::with_capacity(4);
        map.insert(
            NAMESPACE_KEY.into(),
            Value::String(namespace_key.to_owned()),
        );
        map.insert(DOC_ID_KEY.into(), Value::String(doc_id.to_owned()));
        map.insert(CONTENT_KEY.into(), Value::String(document.content.clone()));
        // `metadata` is a free-form `serde_json::Value`. Stamp it
        // verbatim — operator-facing filter expressions reference
        // `metadata.<key>` paths which qdrant resolves through
        // structured JSON values.
        map.insert(METADATA_KEY.into(), document.metadata.clone());
        Payload::try_from(Value::Object(map))
            .expect("payload is a JSON object — Payload::try_from infallible on Object")
    }

    fn point_to_document(point: qdrant_client::qdrant::ScoredPoint) -> Document {
        let (doc_id, content, metadata) = extract_payload_fields(&point.payload);
        Document {
            doc_id,
            content,
            metadata,
            score: Some(point.score),
        }
    }

    fn retrieved_to_document(point: qdrant_client::qdrant::RetrievedPoint) -> Document {
        let (doc_id, content, metadata) = extract_payload_fields(&point.payload);
        Document {
            doc_id,
            content,
            metadata,
            score: None,
        }
    }
}

fn extract_payload_fields(
    payload: &HashMap<String, qdrant_client::qdrant::Value>,
) -> (Option<String>, String, Value) {
    let doc_id = payload
        .get(DOC_ID_KEY)
        .and_then(|v| v.as_str().map(ToOwned::to_owned));
    let content = payload
        .get(CONTENT_KEY)
        .and_then(|v| v.as_str().map(ToOwned::to_owned))
        .unwrap_or_default();
    let metadata = payload
        .get(METADATA_KEY)
        .map_or(Value::Null, qdrant_value_to_json);
    (doc_id, content, metadata)
}

/// Convert a qdrant payload value back to a generic
/// `serde_json::Value`. Sufficient for round-tripping the
/// metadata that `build_payload` originally serialized — qdrant
/// stores numbers as either int or double on the wire and we
/// recover the closest JSON shape on read.
fn qdrant_value_to_json(v: &qdrant_client::qdrant::Value) -> Value {
    match &v.kind {
        Some(qdrant_client::qdrant::value::Kind::NullValue(_)) | None => Value::Null,
        Some(qdrant_client::qdrant::value::Kind::DoubleValue(d)) => {
            serde_json::Number::from_f64(*d).map_or(Value::Null, Value::Number)
        }
        Some(qdrant_client::qdrant::value::Kind::IntegerValue(i)) => Value::Number((*i).into()),
        Some(qdrant_client::qdrant::value::Kind::StringValue(s)) => Value::String(s.clone()),
        Some(qdrant_client::qdrant::value::Kind::BoolValue(b)) => Value::Bool(*b),
        Some(qdrant_client::qdrant::value::Kind::ListValue(list)) => {
            Value::Array(list.values.iter().map(qdrant_value_to_json).collect())
        }
        Some(qdrant_client::qdrant::value::Kind::StructValue(s)) => Value::Object(
            s.fields
                .iter()
                .map(|(k, v)| (k.clone(), qdrant_value_to_json(v)))
                .collect(),
        ),
    }
}

/// Builder for [`QdrantVectorStore`].
#[must_use]
pub struct QdrantVectorStoreBuilder {
    collection: String,
    dimension: usize,
    distance: DistanceMetric,
    url: String,
    api_key: Option<String>,
    timeout: Option<std::time::Duration>,
    skip_create_collection: bool,
    on_disk: Option<bool>,
}

impl QdrantVectorStoreBuilder {
    fn new(collection: impl Into<String>, dimension: usize) -> Self {
        Self {
            collection: collection.into(),
            dimension,
            distance: DistanceMetric::default(),
            url: "http://localhost:6334".into(),
            api_key: None,
            timeout: None,
            skip_create_collection: false,
            on_disk: None,
        }
    }

    /// Set the qdrant gRPC endpoint. Defaults to
    /// `http://localhost:6334`.
    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = url.into();
        self
    }

    /// Attach an API key for qdrant Cloud / restricted deployments.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Override the per-call timeout. qdrant's own default is
    /// applied when unset.
    pub const fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Override the distance metric. Defaults to
    /// [`DistanceMetric::Cosine`].
    pub const fn with_distance(mut self, distance: DistanceMetric) -> Self {
        self.distance = distance;
        self
    }

    /// Skip auto-creation of the collection at build time. Use this
    /// when the collection is provisioned outside the application
    /// (schema-as-code, infra-as-code) and the application should
    /// only consume an existing collection.
    pub const fn with_existing_collection(mut self) -> Self {
        self.skip_create_collection = true;
        self
    }

    /// Force on-disk vector storage (RAM saver for large indices).
    /// Default lets qdrant pick.
    pub const fn with_on_disk(mut self, on_disk: bool) -> Self {
        self.on_disk = Some(on_disk);
        self
    }

    /// Finalize the builder. Connects to qdrant, creates the
    /// collection (unless [`Self::with_existing_collection`] was
    /// called), and provisions the namespace + doc_id payload
    /// indexes that the namespace anchor relies on.
    pub async fn build(self) -> QdrantStoreResult<QdrantVectorStore> {
        let mut config = qdrant_client::config::QdrantConfig::from_url(&self.url);
        if let Some(api_key) = self.api_key {
            config.api_key = Some(api_key);
        }
        if let Some(timeout) = self.timeout {
            config.timeout = timeout;
        }
        let client = Qdrant::new(config)?;

        if !self.skip_create_collection {
            // Idempotent — qdrant returns Ok if the collection
            // already exists with matching shape.
            let exists = client
                .collection_exists(&self.collection)
                .await
                .unwrap_or(false);
            if !exists {
                let mut vector_params =
                    VectorParamsBuilder::new(self.dimension as u64, Distance::from(self.distance));
                if let Some(on_disk) = self.on_disk {
                    vector_params = vector_params.on_disk(on_disk);
                }
                client
                    .create_collection(
                        CreateCollectionBuilder::new(&self.collection)
                            .vectors_config(vector_params),
                    )
                    .await?;

                // Payload indexes — namespace anchor + doc_id lookup
                // ride this on every query, so indexing them is
                // mandatory rather than nice-to-have.
                let _ = client
                    .create_field_index(
                        qdrant_client::qdrant::CreateFieldIndexCollectionBuilder::new(
                            &self.collection,
                            NAMESPACE_KEY,
                            FieldType::Keyword,
                        ),
                    )
                    .await?;
                let _ = client
                    .create_field_index(
                        qdrant_client::qdrant::CreateFieldIndexCollectionBuilder::new(
                            &self.collection,
                            DOC_ID_KEY,
                            FieldType::Keyword,
                        ),
                    )
                    .await?;
            }
        }

        Ok(QdrantVectorStore {
            client: Arc::new(client),
            collection: self.collection.into(),
            dimension: self.dimension,
        })
    }
}

#[async_trait]
impl VectorStore for QdrantVectorStore {
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
                "QdrantVectorStore: vector dimension {} does not match \
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
        let stored_doc = Document {
            doc_id: Some(doc_id.clone()),
            ..document
        };
        let payload = Self::build_payload(&ns_key, &doc_id, &stored_doc);
        let point = PointStruct::new(Self::point_id(&ns_key, &doc_id), vector, payload);
        self.client
            .upsert_points(UpsertPointsBuilder::new(&*self.collection, vec![point]).wait(true))
            .await
            .map_err(|e| Error::from(QdrantStoreError::from(e)))?;
        Ok(())
    }

    async fn batch_add(
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
        let mut points = Vec::with_capacity(items.len());
        for (mut document, vector) in items {
            if vector.len() != self.dimension {
                return Err(Error::invalid_request(format!(
                    "QdrantVectorStore: vector dimension {} does not match \
                     index dimension {}",
                    vector.len(),
                    self.dimension
                )));
            }
            let doc_id = document
                .doc_id
                .clone()
                .unwrap_or_else(|| Uuid::new_v4().to_string());
            document.doc_id = Some(doc_id.clone());
            let payload = Self::build_payload(&ns_key, &doc_id, &document);
            points.push(PointStruct::new(
                Self::point_id(&ns_key, &doc_id),
                vector,
                payload,
            ));
        }
        self.client
            .upsert_points(UpsertPointsBuilder::new(&*self.collection, points).wait(true))
            .await
            .map_err(|e| Error::from(QdrantStoreError::from(e)))?;
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
                "QdrantVectorStore: query dimension {} does not match \
                 index dimension {}",
                query_vector.len(),
                self.dimension
            )));
        }
        let ns_key = ns.render();
        let projected = filter::project(Some(filter), &ns_key).map_err(Error::from)?;

        let resp = self
            .client
            .search_points(
                SearchPointsBuilder::new(&*self.collection, query_vector.to_vec(), top_k as u64)
                    .filter(projected)
                    .with_payload(true),
            )
            .await
            .map_err(|e| Error::from(QdrantStoreError::from(e)))?;
        Ok(resp
            .result
            .into_iter()
            .map(Self::point_to_document)
            .collect())
    }

    async fn delete(&self, ctx: &ExecutionContext, ns: &Namespace, doc_id: &str) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(Error::Cancelled);
        }
        let ns_key = ns.render();
        let pid = Self::point_id(&ns_key, doc_id);
        self.client
            .delete_points(
                DeletePointsBuilder::new(&*self.collection)
                    .points(PointsSelectorOneOf::Points(PointsIdsList {
                        ids: vec![pid],
                    }))
                    .wait(true),
            )
            .await
            .map_err(|e| Error::from(QdrantStoreError::from(e)))?;
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
        // qdrant `upsert_points` is atomic per-id: an existing point
        // is replaced in a single request. We override the trait's
        // default delete-then-add (non-atomic) accordingly.
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
        let projected = filter::project(filter, &ns_key).map_err(Error::from)?;
        let resp = self
            .client
            .count(
                CountPointsBuilder::new(&*self.collection)
                    .filter(projected)
                    .exact(true),
            )
            .await
            .map_err(|e| Error::from(QdrantStoreError::from(e)))?;
        Ok(resp.result.map(|r| r.count as usize).unwrap_or(0))
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
        let projected = filter::project(filter, &ns_key).map_err(Error::from)?;
        // qdrant `scroll` is cursor-based — true pagination needs
        // the prior page's `next_page_offset`. Operators using the
        // trait's `(limit, offset)` shape get a best-effort
        // emulation: we read `offset + limit` and discard the
        // first `offset`. This is accurate but cost-linear in
        // `offset`, so the docstring on the trait calls it out.
        let resp = self
            .client
            .scroll(
                ScrollPointsBuilder::new(&*self.collection)
                    .filter(projected)
                    .limit((limit + offset) as u32)
                    .with_payload(true),
            )
            .await
            .map_err(|e| Error::from(QdrantStoreError::from(e)))?;
        Ok(resp
            .result
            .into_iter()
            .skip(offset)
            .take(limit)
            .map(Self::retrieved_to_document)
            .collect())
    }
}
