//! [`InMemoryVectorStore`] — concrete brute-force [`VectorStore`].
//!
//! In-process, namespace-scoped, cosine-similarity ranking over a
//! linear scan. Designed for two real workloads:
//!
//! - **Tests / dev loops** — deterministic, zero I/O, no
//!   external dependencies. Round-trips through the same
//!   [`VectorStore`] surface a production backend (qdrant /
//!   lancedb / pgvector) implements, so `SemanticMemory<E, V>`
//!   wired to this store exercises the full pipeline end-to-end.
//! - **Small-corpus production** — under ~10K documents per
//!   namespace, brute-force scan with `simd`-friendly dot product
//!   beats the operational complexity of a vector DB. Hot path is a
//!   single read-locked `Vec` walk; writes briefly take the write
//!   lock.
//!
//! Above ~10K docs/namespace, swap in a companion `VectorStore` with
//! an ANN index (HNSW / IVF). The trait surface is identical, so the
//! swap is a one-line replacement at the [`SemanticMemory`]
//! construction site.
//!
//! Filters: `search_filtered`, `count`, `list` all honour the
//! [`VectorFilter`] taxonomy in full — no `LossyEncode` cases. The
//! linear scan evaluates the predicate per row before the dot
//! product, so filter selectivity directly reduces work.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use entelix_core::{Error, ExecutionContext, Result};
use parking_lot::RwLock;
use uuid::Uuid;

use crate::namespace::Namespace;
use crate::traits::{Document, VectorFilter, VectorStore};

/// In-process [`VectorStore`] backed by a per-namespace `Vec<Slot>`.
///
/// Cloning is cheap — internal state lives behind `Arc<RwLock<...>>`
/// so multiple `SemanticMemory` instances can share one store.
pub struct InMemoryVectorStore {
    dimension: usize,
    inner: Arc<RwLock<HashMap<String, Vec<Slot>>>>,
}

#[derive(Clone, Debug)]
struct Slot {
    doc_id: String,
    document: Document,
    vector: Vec<f32>,
    /// Pre-computed `‖vector‖` so cosine similarity reduces to a
    /// single dot product per candidate.
    norm: f32,
}

impl InMemoryVectorStore {
    /// Build an empty store fixed to `dimension`. Inserts whose
    /// `vector.len()` differs surface
    /// [`Error::InvalidRequest`] at `add` time.
    #[must_use]
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            inner: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Total slot count across every namespace. Useful for tests.
    #[must_use]
    pub fn total_slots(&self) -> usize {
        let guard = self.inner.read();
        guard.values().map(Vec::len).sum()
    }
}

impl Clone for InMemoryVectorStore {
    fn clone(&self) -> Self {
        Self {
            dimension: self.dimension,
            inner: Arc::clone(&self.inner),
        }
    }
}

impl std::fmt::Debug for InMemoryVectorStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let guard = self.inner.read();
        f.debug_struct("InMemoryVectorStore")
            .field("dimension", &self.dimension)
            .field("namespaces", &guard.len())
            .field("total_slots", &guard.values().map(Vec::len).sum::<usize>())
            .finish()
    }
}

/// Cosine similarity between two equal-length vectors. Pre-computed
/// `lhs_norm` / `rhs_norm` (the L2 norm of each vector) skip the
/// `sqrt` per candidate during search.
fn cosine_similarity(lhs: &[f32], lhs_norm: f32, rhs: &[f32], rhs_norm: f32) -> f32 {
    if lhs_norm == 0.0 || rhs_norm == 0.0 {
        return 0.0;
    }
    let dot: f32 = lhs.iter().zip(rhs.iter()).map(|(a, b)| a * b).sum();
    dot / (lhs_norm * rhs_norm)
}

fn vector_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[async_trait]
impl VectorStore for InMemoryVectorStore {
    fn dimension(&self) -> usize {
        self.dimension
    }

    async fn add(
        &self,
        _ctx: &ExecutionContext,
        ns: &Namespace,
        document: Document,
        vector: Vec<f32>,
    ) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(Error::invalid_request(format!(
                "InMemoryVectorStore: vector dimension {} does not match index dimension {}",
                vector.len(),
                self.dimension
            )));
        }
        let norm = vector_norm(&vector);
        // Backends typically mint a stable id at insertion time; we
        // honour an operator-supplied `doc_id` and otherwise mint a
        // fresh UUIDv4. Either way the slot carries a non-empty id
        // so subsequent update/delete calls can address it.
        let doc_id = document
            .doc_id
            .clone()
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let stored_doc = Document {
            doc_id: Some(doc_id.clone()),
            ..document
        };
        let mut guard = self.inner.write();
        guard.entry(ns.render()).or_default().push(Slot {
            doc_id,
            document: stored_doc,
            vector,
            norm,
        });
        Ok(())
    }

    async fn search(
        &self,
        _ctx: &ExecutionContext,
        ns: &Namespace,
        query_vector: &[f32],
        top_k: usize,
    ) -> Result<Vec<Document>> {
        if query_vector.len() != self.dimension {
            return Err(Error::invalid_request(format!(
                "InMemoryVectorStore: query dimension {} does not match index dimension {}",
                query_vector.len(),
                self.dimension
            )));
        }
        let q_norm = vector_norm(query_vector);
        let key = ns.render();
        let scored: Vec<(f32, Document)> = {
            let guard = self.inner.read();
            let Some(slots) = guard.get(&key) else {
                return Ok(Vec::new());
            };
            let mut scored: Vec<(f32, Document)> = slots
                .iter()
                .map(|s| {
                    let score = cosine_similarity(query_vector, q_norm, &s.vector, s.norm);
                    let mut doc = s.document.clone();
                    doc.score = Some(score);
                    (score, doc)
                })
                .collect();
            scored.sort_by(|a, b| b.0.total_cmp(&a.0));
            scored.truncate(top_k);
            scored
        };
        Ok(scored.into_iter().map(|(_, d)| d).collect())
    }

    async fn delete(&self, _ctx: &ExecutionContext, ns: &Namespace, doc_id: &str) -> Result<()> {
        let key = ns.render();
        let mut guard = self.inner.write();
        if let Some(slots) = guard.get_mut(&key) {
            slots.retain(|s| s.doc_id != doc_id);
        }
        Ok(())
    }

    async fn update(
        &self,
        _ctx: &ExecutionContext,
        ns: &Namespace,
        doc_id: &str,
        document: Document,
        vector: Vec<f32>,
    ) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(Error::invalid_request(format!(
                "InMemoryVectorStore: vector dimension {} does not match index dimension {}",
                vector.len(),
                self.dimension
            )));
        }
        let norm = vector_norm(&vector);
        let stored_doc = Document {
            doc_id: Some(doc_id.to_owned()),
            ..document
        };
        let mut guard = self.inner.write();
        let slots = guard.entry(ns.render()).or_default();
        if let Some(slot) = slots.iter_mut().find(|s| s.doc_id == doc_id) {
            slot.document = stored_doc;
            slot.vector = vector;
            slot.norm = norm;
        } else {
            return Err(Error::invalid_request(format!(
                "InMemoryVectorStore::update: doc_id '{doc_id}' not found"
            )));
        }
        Ok(())
    }

    async fn search_filtered(
        &self,
        _ctx: &ExecutionContext,
        ns: &Namespace,
        query_vector: &[f32],
        top_k: usize,
        filter: &VectorFilter,
    ) -> Result<Vec<Document>> {
        if query_vector.len() != self.dimension {
            return Err(Error::invalid_request(format!(
                "InMemoryVectorStore: query dimension {} does not match index dimension {}",
                query_vector.len(),
                self.dimension
            )));
        }
        let q_norm = vector_norm(query_vector);
        let key = ns.render();
        let scored: Vec<(f32, Document)> = {
            let guard = self.inner.read();
            let Some(slots) = guard.get(&key) else {
                return Ok(Vec::new());
            };
            let mut scored: Vec<(f32, Document)> = slots
                .iter()
                .filter(|s| evaluate_filter(filter, &s.document.metadata))
                .map(|s| {
                    let score = cosine_similarity(query_vector, q_norm, &s.vector, s.norm);
                    let mut doc = s.document.clone();
                    doc.score = Some(score);
                    (score, doc)
                })
                .collect();
            scored.sort_by(|a, b| b.0.total_cmp(&a.0));
            scored.truncate(top_k);
            scored
        };
        Ok(scored.into_iter().map(|(_, d)| d).collect())
    }

    async fn count(
        &self,
        _ctx: &ExecutionContext,
        ns: &Namespace,
        filter: Option<&VectorFilter>,
    ) -> Result<usize> {
        let key = ns.render();
        let guard = self.inner.read();
        let count = guard.get(&key).map_or(0, |slots| match filter {
            None => slots.len(),
            Some(f) => slots
                .iter()
                .filter(|s| evaluate_filter(f, &s.document.metadata))
                .count(),
        });
        Ok(count)
    }

    async fn list(
        &self,
        _ctx: &ExecutionContext,
        ns: &Namespace,
        filter: Option<&VectorFilter>,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<Document>> {
        let key = ns.render();
        let guard = self.inner.read();
        let Some(slots) = guard.get(&key) else {
            return Ok(Vec::new());
        };
        let out = slots
            .iter()
            .filter(|s| match filter {
                None => true,
                Some(f) => evaluate_filter(f, &s.document.metadata),
            })
            .skip(offset)
            .take(limit)
            .map(|s| s.document.clone())
            .collect();
        Ok(out)
    }
}

/// Evaluate a [`VectorFilter`] against a document's metadata blob.
/// Returns `true` when the predicate matches. The implementation
/// matches the wire-level semantics every backend agrees on:
/// - missing metadata key → predicate false (except [`VectorFilter::Not`]).
/// - non-numeric operands on numeric variants → false.
/// - equality is JSON-value equality (not relaxed coercion).
fn evaluate_filter(filter: &VectorFilter, metadata: &serde_json::Value) -> bool {
    match filter {
        VectorFilter::All => true,
        VectorFilter::Eq { key, value } => lookup(metadata, key).is_some_and(|v| v == value),
        VectorFilter::Lt { key, value } => {
            compare_numeric(metadata, key, value, std::cmp::Ordering::Less, false)
        }
        VectorFilter::Lte { key, value } => {
            compare_numeric(metadata, key, value, std::cmp::Ordering::Less, true)
        }
        VectorFilter::Gt { key, value } => {
            compare_numeric(metadata, key, value, std::cmp::Ordering::Greater, false)
        }
        VectorFilter::Gte { key, value } => {
            compare_numeric(metadata, key, value, std::cmp::Ordering::Greater, true)
        }
        VectorFilter::Range { key, min, max } => {
            compare_numeric(metadata, key, min, std::cmp::Ordering::Greater, true)
                && compare_numeric(metadata, key, max, std::cmp::Ordering::Less, true)
        }
        VectorFilter::In { key, values } => {
            lookup(metadata, key).is_some_and(|v| values.contains(v))
        }
        VectorFilter::Exists { key } => lookup(metadata, key).is_some(),
        VectorFilter::And(children) => children.iter().all(|c| evaluate_filter(c, metadata)),
        VectorFilter::Or(children) => children.iter().any(|c| evaluate_filter(c, metadata)),
        VectorFilter::Not(child) => !evaluate_filter(child, metadata),
    }
}

/// Look up a dotted metadata path. Each segment indexes one level
/// of the JSON tree; non-object intermediates short-circuit to
/// `None`.
fn lookup<'a>(value: &'a serde_json::Value, key: &str) -> Option<&'a serde_json::Value> {
    let mut cursor = value;
    for segment in key.split('.') {
        cursor = cursor.as_object()?.get(segment)?;
    }
    Some(cursor)
}

/// Numeric comparison helper. `direction` is the side of the ordering
/// we want (Less means "lhs < rhs"); `inclusive` flips strict
/// inequality to include equality. Returns false on non-numeric
/// operands rather than coerce — a metadata field that stores `"42"`
/// as a string is a schema bug; surface it by failing the predicate.
fn compare_numeric(
    metadata: &serde_json::Value,
    key: &str,
    rhs: &serde_json::Value,
    direction: std::cmp::Ordering,
    inclusive: bool,
) -> bool {
    let Some(lhs) = lookup(metadata, key).and_then(serde_json::Value::as_f64) else {
        return false;
    };
    let Some(rhs) = rhs.as_f64() else {
        return false;
    };
    let cmp = lhs.partial_cmp(&rhs).unwrap_or(std::cmp::Ordering::Equal);
    if cmp == std::cmp::Ordering::Equal {
        return inclusive;
    }
    cmp == direction
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use serde_json::json;

    fn ns() -> Namespace {
        Namespace::new("acme").with_scope("agent-a")
    }

    fn ctx() -> ExecutionContext {
        ExecutionContext::new()
    }

    fn doc(id: &str, content: &str, metadata: serde_json::Value) -> Document {
        Document::new(content)
            .with_doc_id(id)
            .with_metadata(metadata)
    }

    #[tokio::test]
    async fn add_then_search_returns_top_k_by_similarity() {
        let store = InMemoryVectorStore::new(3);
        let n = ns();
        store
            .add(
                &ctx(),
                &n,
                doc("a", "alpha", json!({})),
                vec![1.0, 0.0, 0.0],
            )
            .await
            .unwrap();
        store
            .add(&ctx(), &n, doc("b", "beta", json!({})), vec![0.0, 1.0, 0.0])
            .await
            .unwrap();
        store
            .add(
                &ctx(),
                &n,
                doc("c", "gamma", json!({})),
                vec![0.9, 0.1, 0.0],
            )
            .await
            .unwrap();
        let hits = store.search(&ctx(), &n, &[1.0, 0.0, 0.0], 2).await.unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].doc_id.as_deref(), Some("a"));
        assert_eq!(hits[1].doc_id.as_deref(), Some("c"));
        // Score is cosine — exact match → 1.0.
        assert!((hits[0].score.unwrap() - 1.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn search_returns_empty_for_unknown_namespace() {
        let store = InMemoryVectorStore::new(2);
        let hits = store.search(&ctx(), &ns(), &[1.0, 0.0], 5).await.unwrap();
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn dimension_mismatch_is_invalid_request() {
        let store = InMemoryVectorStore::new(3);
        let err = store
            .add(&ctx(), &ns(), doc("a", "x", json!({})), vec![1.0, 0.0])
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("dimension"));
    }

    #[tokio::test]
    async fn delete_then_search_omits_deleted_doc() {
        let store = InMemoryVectorStore::new(2);
        store
            .add(&ctx(), &ns(), doc("a", "x", json!({})), vec![1.0, 0.0])
            .await
            .unwrap();
        store.delete(&ctx(), &ns(), "a").await.unwrap();
        let hits = store.search(&ctx(), &ns(), &[1.0, 0.0], 5).await.unwrap();
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn update_replaces_vector_atomically() {
        let store = InMemoryVectorStore::new(2);
        store
            .add(&ctx(), &ns(), doc("a", "v1", json!({})), vec![1.0, 0.0])
            .await
            .unwrap();
        store
            .update(
                &ctx(),
                &ns(),
                "a",
                doc("a", "v2", json!({"version": 2})),
                vec![0.0, 1.0],
            )
            .await
            .unwrap();
        let hits = store.search(&ctx(), &ns(), &[0.0, 1.0], 1).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].content, "v2");
        assert_eq!(hits[0].metadata["version"], 2);
    }

    #[tokio::test]
    async fn update_unknown_doc_returns_invalid_request() {
        let store = InMemoryVectorStore::new(2);
        let err = store
            .update(
                &ctx(),
                &ns(),
                "ghost",
                doc("ghost", "x", json!({})),
                vec![1.0, 0.0],
            )
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("not found"));
    }

    #[tokio::test]
    async fn search_filtered_honours_eq_filter() {
        let store = InMemoryVectorStore::new(2);
        store
            .add(
                &ctx(),
                &ns(),
                doc("a", "x", json!({"category": "A"})),
                vec![1.0, 0.0],
            )
            .await
            .unwrap();
        store
            .add(
                &ctx(),
                &ns(),
                doc("b", "y", json!({"category": "B"})),
                vec![1.0, 0.0],
            )
            .await
            .unwrap();
        let filter = VectorFilter::Eq {
            key: "category".into(),
            value: json!("A"),
        };
        let hits = store
            .search_filtered(&ctx(), &ns(), &[1.0, 0.0], 5, &filter)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id.as_deref(), Some("a"));
    }

    #[tokio::test]
    async fn search_filtered_honours_range_and_negation() {
        let store = InMemoryVectorStore::new(2);
        for (id, score) in [("a", 5.0), ("b", 12.0), ("c", 25.0), ("d", 50.0)] {
            store
                .add(
                    &ctx(),
                    &ns(),
                    doc(id, "x", json!({"score": score})),
                    vec![1.0, 0.0],
                )
                .await
                .unwrap();
        }
        let in_range = VectorFilter::Range {
            key: "score".into(),
            min: json!(10.0),
            max: json!(30.0),
        };
        let hits = store
            .search_filtered(&ctx(), &ns(), &[1.0, 0.0], 10, &in_range)
            .await
            .unwrap();
        assert_eq!(hits.len(), 2);
        let ids: Vec<&str> = hits.iter().filter_map(|d| d.doc_id.as_deref()).collect();
        assert!(ids.contains(&"b"));
        assert!(ids.contains(&"c"));

        let outside = VectorFilter::Not(Box::new(in_range));
        let hits = store
            .search_filtered(&ctx(), &ns(), &[1.0, 0.0], 10, &outside)
            .await
            .unwrap();
        assert_eq!(hits.len(), 2);
    }

    #[tokio::test]
    async fn count_with_filter_returns_matching_subset() {
        let store = InMemoryVectorStore::new(2);
        for (id, cat) in [("a", "X"), ("b", "Y"), ("c", "X")] {
            store
                .add(
                    &ctx(),
                    &ns(),
                    doc(id, "x", json!({"cat": cat})),
                    vec![1.0, 0.0],
                )
                .await
                .unwrap();
        }
        assert_eq!(store.count(&ctx(), &ns(), None).await.unwrap(), 3);
        let only_x = VectorFilter::Eq {
            key: "cat".into(),
            value: json!("X"),
        };
        assert_eq!(store.count(&ctx(), &ns(), Some(&only_x)).await.unwrap(), 2);
    }

    #[tokio::test]
    async fn list_paginates() {
        let store = InMemoryVectorStore::new(2);
        for i in 0..5 {
            store
                .add(
                    &ctx(),
                    &ns(),
                    doc(&format!("d{i}"), "x", json!({})),
                    vec![1.0, 0.0],
                )
                .await
                .unwrap();
        }
        let page = store.list(&ctx(), &ns(), None, 2, 1).await.unwrap();
        assert_eq!(page.len(), 2);
    }

    #[tokio::test]
    async fn batch_add_default_loops_through_add() {
        let store = InMemoryVectorStore::new(2);
        let items = vec![
            (doc("a", "x", json!({})), vec![1.0, 0.0]),
            (doc("b", "y", json!({})), vec![0.0, 1.0]),
        ];
        store.batch_add(&ctx(), &ns(), items).await.unwrap();
        assert_eq!(store.total_slots(), 2);
    }

    #[tokio::test]
    async fn namespaces_are_isolated() {
        let store = InMemoryVectorStore::new(2);
        let ns_a = Namespace::new("acme").with_scope("agent-a");
        let ns_b = Namespace::new("acme").with_scope("agent-b");
        store
            .add(&ctx(), &ns_a, doc("a", "x", json!({})), vec![1.0, 0.0])
            .await
            .unwrap();
        let hits_a = store.search(&ctx(), &ns_a, &[1.0, 0.0], 5).await.unwrap();
        let hits_b = store.search(&ctx(), &ns_b, &[1.0, 0.0], 5).await.unwrap();
        assert_eq!(hits_a.len(), 1);
        assert_eq!(hits_b.len(), 0);
    }
}
