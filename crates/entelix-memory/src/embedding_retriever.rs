//! `EmbeddingRetriever` — adapter that wires a [`VectorStore`] +
//! [`Embedder`] into a [`Retriever`].
//!
//! The two pieces every dense-retrieval pipeline needs sit on
//! opposite sides of the [`Retriever`] surface — the embedder turns
//! query text into a vector, the vector store does the
//! nearest-neighbour search. This adapter wires them together with
//! a fixed [`Namespace`] (the multi-tenant boundary, invariant 11)
//! and exposes the canonical [`Retriever::retrieve`] shape so the
//! result drops directly into [`SemanticMemory`](crate::SemanticMemory),
//! `entelix-rag` recipes, or any custom retrieval-aware agent.
//!
//! ## Filter + score handling
//!
//! - [`RetrievalQuery::filter`] routes through
//!   [`VectorStore::search_filtered`] when set; backends without
//!   filter support surface their own
//!   [`Error::Config`](entelix_core::Error::Config) as the trait
//!   contract dictates. Filter-less queries route through
//!   [`VectorStore::search`].
//! - [`RetrievalQuery::min_score`] is applied as a post-filter on
//!   the returned hits; backend-side score floors are not portable
//!   across dot-product, cosine, and L2 distance backends, so the
//!   adapter trims locally rather than translating the floor.
//! - [`RetrievalQuery::top_k`] flows directly into the backend
//!   call. The min-score post-filter then trims further; a query
//!   that requests `top_k = 10` with `min_score = 0.5` may return
//!   fewer than 10 hits when scores fall below the floor.

use std::sync::Arc;

use async_trait::async_trait;
use entelix_core::{ExecutionContext, Result};

use crate::namespace::Namespace;
use crate::traits::{Document, Embedder, RetrievalQuery, Retriever, VectorStore};

/// Adapter that combines an [`Embedder`] and a [`VectorStore`]
/// (scoped to one [`Namespace`]) into a [`Retriever`].
///
/// Cloning is cheap — both the embedder and the store sit behind
/// `Arc`, so multiple retrievers can share one connection pool /
/// embedding client.
pub struct EmbeddingRetriever<E, V> {
    embedder: Arc<E>,
    store: Arc<V>,
    namespace: Namespace,
}

impl<E, V> Clone for EmbeddingRetriever<E, V> {
    fn clone(&self) -> Self {
        Self {
            embedder: Arc::clone(&self.embedder),
            store: Arc::clone(&self.store),
            namespace: self.namespace.clone(),
        }
    }
}

impl<E, V> std::fmt::Debug for EmbeddingRetriever<E, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingRetriever")
            .field("namespace", &self.namespace)
            .finish_non_exhaustive()
    }
}

impl<E, V> EmbeddingRetriever<E, V>
where
    E: Embedder,
    V: VectorStore,
{
    /// Build a retriever that runs every query against `store`
    /// scoped to `namespace`, using `embedder` to turn query text
    /// into the search vector.
    #[must_use]
    pub const fn new(embedder: Arc<E>, store: Arc<V>, namespace: Namespace) -> Self {
        Self {
            embedder,
            store,
            namespace,
        }
    }

    /// Borrow the wired embedder.
    #[must_use]
    pub const fn embedder(&self) -> &Arc<E> {
        &self.embedder
    }

    /// Borrow the wired vector store.
    #[must_use]
    pub const fn store(&self) -> &Arc<V> {
        &self.store
    }

    /// Borrow the configured namespace.
    #[must_use]
    pub const fn namespace(&self) -> &Namespace {
        &self.namespace
    }
}

#[async_trait]
impl<E, V> Retriever for EmbeddingRetriever<E, V>
where
    E: Embedder + 'static,
    V: VectorStore + 'static,
{
    async fn retrieve(
        &self,
        query: RetrievalQuery,
        ctx: &ExecutionContext,
    ) -> Result<Vec<Document>> {
        let embedding = self.embedder.embed(&query.text, ctx).await?;
        let mut hits = match query.filter.as_ref() {
            Some(filter) => {
                self.store
                    .search_filtered(ctx, &self.namespace, &embedding.vector, query.top_k, filter)
                    .await?
            }
            None => {
                self.store
                    .search(ctx, &self.namespace, &embedding.vector, query.top_k)
                    .await?
            }
        };
        if let Some(floor) = query.min_score {
            hits.retain(|doc| doc.score.is_some_and(|s| s >= floor));
        }
        Ok(hits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::in_memory_vector_store::InMemoryVectorStore;
    use crate::traits::{Embedding, VectorFilter};
    use entelix_core::TenantId;
    use std::sync::Arc;

    /// Tiny BoW embedder over a fixed vocabulary — every recognised
    /// word increments one basis component, the vector is L2-
    /// normalised. Stable, no IO, deterministic.
    struct BowEmbedder {
        vocab: std::collections::HashMap<String, usize>,
        dimension: usize,
    }

    impl BowEmbedder {
        fn new(words: &[&str]) -> Self {
            let dimension = words.len();
            let vocab = words
                .iter()
                .enumerate()
                .map(|(i, w)| ((*w).to_owned(), i))
                .collect();
            Self { vocab, dimension }
        }
    }

    #[async_trait]
    impl Embedder for BowEmbedder {
        fn dimension(&self) -> usize {
            self.dimension
        }
        async fn embed(&self, text: &str, _ctx: &ExecutionContext) -> Result<Embedding> {
            let mut v = vec![0.0_f32; self.dimension];
            for word in text.to_lowercase().split_whitespace() {
                if let Some(&idx) = self.vocab.get(word)
                    && let Some(slot) = v.get_mut(idx)
                {
                    *slot += 1.0;
                }
            }
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in &mut v {
                    *x /= norm;
                }
            }
            Ok(Embedding::new(v))
        }
    }

    fn ns(tenant: &str) -> Namespace {
        Namespace::new(TenantId::new(tenant))
    }

    async fn seed_store(
        embedder: &Arc<BowEmbedder>,
        store: &Arc<InMemoryVectorStore>,
        namespace: &Namespace,
        docs: &[(&str, &str)],
    ) -> Result<()> {
        let ctx = ExecutionContext::new();
        let mut items = Vec::new();
        for (id, content) in docs {
            let emb = embedder.embed(content, &ctx).await?;
            let doc = Document::new(*content).with_doc_id((*id).to_owned());
            items.push((doc, emb.vector));
        }
        store.add_batch(&ctx, namespace, items).await
    }

    #[tokio::test]
    async fn retrieves_top_k_for_query() -> Result<()> {
        let embedder = Arc::new(BowEmbedder::new(&[
            "rust", "agent", "tokio", "async", "memory", "graph",
        ]));
        let store = Arc::new(InMemoryVectorStore::new(embedder.dimension()));
        let namespace = ns("acme");
        seed_store(
            &embedder,
            &store,
            &namespace,
            &[
                ("a", "rust agent tokio"),
                ("b", "graph memory"),
                ("c", "async rust"),
            ],
        )
        .await?;

        let retriever =
            EmbeddingRetriever::new(Arc::clone(&embedder), Arc::clone(&store), namespace.clone());
        let ctx = ExecutionContext::new();
        let hits = retriever
            .retrieve(RetrievalQuery::new("rust agent", 2), &ctx)
            .await?;
        assert_eq!(hits.len(), 2);
        // The doc with both "rust" + "agent" must rank first.
        assert_eq!(hits.first().and_then(|h| h.doc_id.as_deref()), Some("a"));
        Ok(())
    }

    #[tokio::test]
    async fn min_score_post_filters_below_floor() -> Result<()> {
        let embedder = Arc::new(BowEmbedder::new(&["alpha", "bravo", "charlie"]));
        let store = Arc::new(InMemoryVectorStore::new(embedder.dimension()));
        let namespace = ns("acme");
        seed_store(
            &embedder,
            &store,
            &namespace,
            &[("a", "alpha bravo"), ("b", "alpha"), ("c", "charlie")],
        )
        .await?;

        let retriever =
            EmbeddingRetriever::new(Arc::clone(&embedder), Arc::clone(&store), namespace.clone());
        let ctx = ExecutionContext::new();
        // Floor of 0.99 — only the exact match (cosine 1.0) survives.
        let hits = retriever
            .retrieve(
                RetrievalQuery::new("alpha bravo", 5).with_min_score(0.99),
                &ctx,
            )
            .await?;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits.first().and_then(|h| h.doc_id.as_deref()), Some("a"));
        Ok(())
    }

    #[tokio::test]
    async fn filter_routes_through_search_filtered() -> Result<()> {
        // InMemoryVectorStore implements search_filtered; verifying
        // the adapter takes the filtered branch when query.filter is
        // set.
        let embedder = Arc::new(BowEmbedder::new(&["alpha", "bravo"]));
        let store = Arc::new(InMemoryVectorStore::new(embedder.dimension()));
        let namespace = ns("acme");
        let ctx = ExecutionContext::new();
        let docs = [
            ("a", "alpha bravo", serde_json::json!({"kind": "code"})),
            ("b", "alpha", serde_json::json!({"kind": "doc"})),
        ];
        let mut items = Vec::new();
        for (id, content, meta) in &docs {
            let emb = embedder.embed(content, &ctx).await?;
            let doc = Document::new(*content)
                .with_doc_id((*id).to_owned())
                .with_metadata(meta.clone());
            items.push((doc, emb.vector));
        }
        store.add_batch(&ctx, &namespace, items).await?;

        let retriever =
            EmbeddingRetriever::new(Arc::clone(&embedder), Arc::clone(&store), namespace.clone());
        let hits = retriever
            .retrieve(
                RetrievalQuery::new("alpha", 5).with_filter(VectorFilter::Eq {
                    key: "kind".to_owned(),
                    value: serde_json::json!("doc"),
                }),
                &ctx,
            )
            .await?;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits.first().and_then(|h| h.doc_id.as_deref()), Some("b"));
        Ok(())
    }

    #[tokio::test]
    async fn namespace_isolation_blocks_cross_tenant_reads() -> Result<()> {
        let embedder = Arc::new(BowEmbedder::new(&["alpha", "bravo", "charlie"]));
        let store = Arc::new(InMemoryVectorStore::new(embedder.dimension()));
        let alice = ns("alice");
        let bob = ns("bob");
        seed_store(
            &embedder,
            &store,
            &alice,
            &[("alice-doc", "alpha bravo charlie")],
        )
        .await?;
        // Bob's namespace stays empty.

        let bob_retriever = EmbeddingRetriever::new(Arc::clone(&embedder), Arc::clone(&store), bob);
        let ctx = ExecutionContext::new();
        let hits = bob_retriever
            .retrieve(RetrievalQuery::new("alpha bravo charlie", 10), &ctx)
            .await?;
        assert!(
            hits.is_empty(),
            "Bob must not observe Alice's documents: {hits:?}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn clone_shares_embedder_and_store() {
        let embedder = Arc::new(BowEmbedder::new(&["x"]));
        let store = Arc::new(InMemoryVectorStore::new(1));
        let namespace = ns("acme");
        let original =
            EmbeddingRetriever::new(Arc::clone(&embedder), Arc::clone(&store), namespace.clone());
        let cloned = original.clone();
        assert!(Arc::ptr_eq(original.embedder(), cloned.embedder()));
        assert!(Arc::ptr_eq(original.store(), cloned.store()));
        assert_eq!(cloned.namespace(), &namespace);
    }
}
