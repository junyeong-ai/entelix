//! `SemanticMemory<E, V>` — generic composition of `Embedder` +
//! `VectorStore` scoped to one `Namespace`. Plus
//! [`SemanticMemoryBackend`], the object-safe consumer trait.
//!
//! ## Three-tier layering
//!
//! 1. **Primitives** — [`crate::Embedder`] + [`crate::VectorStore`]
//!    are operator-implemented backend traits. The `VectorStore`
//!    takes `Namespace` as a per-call parameter so a single store
//!    instance backs many tenants. The `Embedder` is independent
//!    and pool-shared via `Arc<Self>`.
//! 2. **Bundle** — [`SemanticMemory<E, V>`] glues `Arc<E>` +
//!    `Arc<V>` + a fixed `Namespace` into one surface. Generic over
//!    the concrete embedder / vector-store types so static dispatch
//!    is preserved on hot paths.
//! 3. **Consumer trait** — [`SemanticMemoryBackend`] is the object-
//!    safe view tools and orchestration code consume as
//!    `Arc<dyn SemanticMemoryBackend>`. The bound `Namespace` is
//!    baked in via [`SemanticMemoryBackend::namespace`]; consumers
//!    don't pass one. Implemented automatically for every
//!    `SemanticMemory<E, V>`.
//!
//! Operators add a backend by implementing `VectorStore` (and
//! optionally `Embedder` for non-OpenAI vendors); they never need
//! to implement `SemanticMemoryBackend` directly — wrapping in
//! `SemanticMemory::new` produces the trait-object view for free.

use std::sync::Arc;

use async_trait::async_trait;
use entelix_core::{Error, ExecutionContext, Result};

use crate::namespace::Namespace;
use crate::traits::{Document, Embedder, RerankedDocument, Reranker, VectorFilter, VectorStore};

/// Object-safe consumer trait — tier 3 of the semantic-memory
/// layering documented at the module level. Consumers (tools,
/// orchestration code, recipes) take
/// `Arc<dyn SemanticMemoryBackend>` to operate on a namespace-scoped
/// embed-and-search surface without parameterising over the
/// concrete embedder / vector-store types.
///
/// **Operators do not implement this trait directly.** Implement
/// [`crate::VectorStore`] (and optionally [`crate::Embedder`]),
/// then wrap in [`SemanticMemory::new`] — the
/// `impl SemanticMemoryBackend for SemanticMemory<E, V>` blanket
/// produces the trait-object view automatically.
///
/// The trait mirrors the full [`SemanticMemory`] surface (search,
/// add, delete, update, add_batch, search_filtered, plus a
/// rerank-aware variant via `&dyn Reranker`) so consumers do not
/// need to downcast to the concrete generic type to access mutating
/// or rerank operations.
#[async_trait]
pub trait SemanticMemoryBackend: Send + Sync + 'static {
    /// Borrow the bound [`Namespace`]. Tools and orchestration code
    /// that route queries by tenant or scope read this to validate
    /// the backend is wired to the expected slice without downcasting
    /// to the concrete generic type.
    fn namespace(&self) -> &Namespace;

    /// Vector dimension the backend embeds and indexes at. Lets
    /// schedulers verify a query embedder matches before issuing a
    /// search, and lets dashboards report index width per tenant.
    fn dimension(&self) -> usize;

    /// Embed `query` and return the top `top_k` matches.
    async fn search(
        &self,
        ctx: &ExecutionContext,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<Document>>;

    /// Embed `query`, fetch `candidates`, push down `filter` if the
    /// backend supports it; otherwise the underlying `VectorStore`
    /// returns `Error::Config`.
    async fn search_filtered(
        &self,
        ctx: &ExecutionContext,
        query: &str,
        top_k: usize,
        filter: &VectorFilter,
    ) -> Result<Vec<Document>>;

    /// Two-stage retrieval: over-fetch `candidates` then rerank
    /// down to `top_k`. The reranker is supplied as a trait object
    /// so the backend trait stays object-safe (the concrete
    /// [`SemanticMemory::search_with_rerank`] also accepts
    /// monomorphic `R: Reranker` for users who prefer static
    /// dispatch). Returns [`RerankedDocument`]s so callers can
    /// inspect the reranker's score alongside the retrieval score.
    async fn search_with_rerank_dyn(
        &self,
        ctx: &ExecutionContext,
        query: &str,
        top_k: usize,
        candidates: usize,
        reranker: &dyn Reranker,
    ) -> Result<Vec<RerankedDocument>>;

    /// Embed `document.content` and add the document to the index.
    async fn add(&self, ctx: &ExecutionContext, document: Document) -> Result<()>;

    /// Add many documents at once. Default implementations defer to
    /// the embedder's batch path then to the vector store's batch
    /// path so backends that support either can amortise round-trips.
    async fn add_batch(&self, ctx: &ExecutionContext, documents: Vec<Document>) -> Result<()>;

    /// Delete a previously-indexed document by its backend id.
    async fn delete(&self, ctx: &ExecutionContext, doc_id: &str) -> Result<()>;

    /// Replace an existing document's vector and metadata atomically
    /// when the backend supports it; otherwise non-atomic via
    /// delete + add.
    async fn update(&self, ctx: &ExecutionContext, doc_id: &str, document: Document) -> Result<()>;

    /// Count documents in the bound namespace, optionally narrowed
    /// by a metadata filter. Pass-through to
    /// [`VectorStore::count`] — backends without count support
    /// surface `Error::Config`.
    async fn count(&self, ctx: &ExecutionContext, filter: Option<&VectorFilter>) -> Result<usize>;

    /// Enumerate documents in the bound namespace. Pass-through to
    /// [`VectorStore::list`] — backends without enumeration
    /// support surface `Error::Config`.
    async fn list(
        &self,
        ctx: &ExecutionContext,
        filter: Option<&VectorFilter>,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<Document>>;
}

/// `Embedder + VectorStore + Namespace` bundle.
///
/// The embedder produces vectors at `add` and `search` time; the vector
/// store holds them. Both must agree on `dimension()` — checked at
/// construction.
pub struct SemanticMemory<E, V>
where
    E: Embedder,
    V: VectorStore,
{
    embedder: Arc<E>,
    vector_store: Arc<V>,
    namespace: Namespace,
}

impl<E, V> SemanticMemory<E, V>
where
    E: Embedder,
    V: VectorStore,
{
    /// Construct from owned components, validating dimension parity.
    ///
    /// Returns `Error::Config` if the embedder and vector store report
    /// different dimensions.
    pub fn new(embedder: Arc<E>, vector_store: Arc<V>, namespace: Namespace) -> Result<Self> {
        let e_dim = embedder.dimension();
        let v_dim = vector_store.dimension();
        if e_dim != v_dim {
            return Err(Error::config(format!(
                "SemanticMemory: embedder dimension ({e_dim}) does not match vector-store \
                 dimension ({v_dim})"
            )));
        }
        Ok(Self {
            embedder,
            vector_store,
            namespace,
        })
    }

    /// Borrow the bound namespace.
    pub const fn namespace(&self) -> &Namespace {
        &self.namespace
    }

    /// Embed `document.content` and add it to the vector store.
    /// The embedder's usage metadata (when surfaced) is dropped here —
    /// callers that need to charge cost meters per-embed should use
    /// the embedder directly and then call
    /// [`VectorStore::add`](crate::VectorStore::add).
    pub async fn add(&self, ctx: &ExecutionContext, document: Document) -> Result<()> {
        let embedding = self.embedder.embed(&document.content, ctx).await?;
        self.vector_store
            .add(ctx, &self.namespace, document, embedding.vector)
            .await
    }

    /// Add many documents at once — uses `Embedder::embed_batch` to
    /// amortise embedder calls then `VectorStore::add_batch` to
    /// amortise index writes.
    ///
    /// Returns [`Error::Config`] if the embedder produces a vector
    /// count that doesn't match the input documents — silent
    /// truncation via `zip` would drop documents without surfacing
    /// the embedder bug.
    pub async fn add_batch(&self, ctx: &ExecutionContext, documents: Vec<Document>) -> Result<()> {
        if documents.is_empty() {
            return Ok(());
        }
        let texts: Vec<String> = documents.iter().map(|d| d.content.clone()).collect();
        let embeddings = self.embedder.embed_batch(&texts, ctx).await?;
        if embeddings.len() != texts.len() {
            return Err(Error::config(format!(
                "SemanticMemory::add_batch: embedder returned {} vectors for {} documents",
                embeddings.len(),
                texts.len()
            )));
        }
        let items: Vec<(Document, Vec<f32>)> = documents
            .into_iter()
            .zip(embeddings)
            .map(|(doc, embedding)| (doc, embedding.vector))
            .collect();
        self.vector_store
            .add_batch(ctx, &self.namespace, items)
            .await
    }

    /// Delete a previously-indexed document by id.
    pub async fn delete(&self, ctx: &ExecutionContext, doc_id: &str) -> Result<()> {
        self.vector_store.delete(ctx, &self.namespace, doc_id).await
    }

    /// Update a previously-indexed document. Re-embeds the
    /// document's content via the embedder and asks the vector
    /// store to swap vector + metadata under the same id.
    pub async fn update(
        &self,
        ctx: &ExecutionContext,
        doc_id: &str,
        document: Document,
    ) -> Result<()> {
        let embedding = self.embedder.embed(&document.content, ctx).await?;
        self.vector_store
            .update(ctx, &self.namespace, doc_id, document, embedding.vector)
            .await
    }

    /// Embed `query` and search the vector store for the top `top_k`
    /// matches.
    pub async fn search(
        &self,
        ctx: &ExecutionContext,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<Document>> {
        let embedding = self.embedder.embed(query, ctx).await?;
        self.vector_store
            .search(ctx, &self.namespace, &embedding.vector, top_k)
            .await
    }

    /// Embed `query` and search with a metadata filter. Backends
    /// without filter support return `Error::Config` per the
    /// `VectorStore::search_filtered` contract.
    pub async fn search_filtered(
        &self,
        ctx: &ExecutionContext,
        query: &str,
        top_k: usize,
        filter: &VectorFilter,
    ) -> Result<Vec<Document>> {
        let embedding = self.embedder.embed(query, ctx).await?;
        self.vector_store
            .search_filtered(ctx, &self.namespace, &embedding.vector, top_k, filter)
            .await
    }

    /// Two-stage retrieval: over-fetch `candidates` from the vector
    /// store, then rerank down to `top_k` via the supplied
    /// [`Reranker`]. The over-fetch factor is the operator's lever
    /// for trading recall against rerank latency — passing
    /// `candidates == top_k` makes the reranker no-op-shaped, while
    /// `candidates >> top_k` exposes more candidates to the
    /// reranker's scoring. Returns [`RerankedDocument`]s so callers
    /// retain both the retrieval and rerank scores for explainability.
    pub async fn search_with_rerank<R: Reranker>(
        &self,
        ctx: &ExecutionContext,
        query: &str,
        top_k: usize,
        candidates: usize,
        reranker: &R,
    ) -> Result<Vec<RerankedDocument>> {
        let pool = self.search(ctx, query, candidates.max(top_k)).await?;
        reranker.rerank(query, pool, top_k, ctx).await
    }

    /// Count documents in the bound namespace. Pass-through to
    /// [`VectorStore::count`] — backends without count support
    /// surface `Error::Config`.
    pub async fn count(
        &self,
        ctx: &ExecutionContext,
        filter: Option<&VectorFilter>,
    ) -> Result<usize> {
        self.vector_store.count(ctx, &self.namespace, filter).await
    }

    /// Enumerate documents in the bound namespace. Pass-through to
    /// [`VectorStore::list`] — backends without enumeration
    /// support surface `Error::Config`.
    pub async fn list(
        &self,
        ctx: &ExecutionContext,
        filter: Option<&VectorFilter>,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<Document>> {
        self.vector_store
            .list(ctx, &self.namespace, filter, limit, offset)
            .await
    }
}

#[async_trait]
impl<E, V> SemanticMemoryBackend for SemanticMemory<E, V>
where
    E: Embedder,
    V: VectorStore,
{
    fn namespace(&self) -> &Namespace {
        &self.namespace
    }

    fn dimension(&self) -> usize {
        self.embedder.dimension()
    }

    async fn search(
        &self,
        ctx: &ExecutionContext,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<Document>> {
        Self::search(self, ctx, query, top_k).await
    }

    async fn search_filtered(
        &self,
        ctx: &ExecutionContext,
        query: &str,
        top_k: usize,
        filter: &VectorFilter,
    ) -> Result<Vec<Document>> {
        Self::search_filtered(self, ctx, query, top_k, filter).await
    }

    async fn add(&self, ctx: &ExecutionContext, document: Document) -> Result<()> {
        Self::add(self, ctx, document).await
    }

    async fn add_batch(&self, ctx: &ExecutionContext, documents: Vec<Document>) -> Result<()> {
        Self::add_batch(self, ctx, documents).await
    }

    async fn delete(&self, ctx: &ExecutionContext, doc_id: &str) -> Result<()> {
        Self::delete(self, ctx, doc_id).await
    }

    async fn update(&self, ctx: &ExecutionContext, doc_id: &str, document: Document) -> Result<()> {
        Self::update(self, ctx, doc_id, document).await
    }

    async fn search_with_rerank_dyn(
        &self,
        ctx: &ExecutionContext,
        query: &str,
        top_k: usize,
        candidates: usize,
        reranker: &dyn Reranker,
    ) -> Result<Vec<RerankedDocument>> {
        let pool = self.search(ctx, query, candidates.max(top_k)).await?;
        reranker.rerank(query, pool, top_k, ctx).await
    }

    async fn count(&self, ctx: &ExecutionContext, filter: Option<&VectorFilter>) -> Result<usize> {
        Self::count(self, ctx, filter).await
    }

    async fn list(
        &self,
        ctx: &ExecutionContext,
        filter: Option<&VectorFilter>,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<Document>> {
        Self::list(self, ctx, filter, limit, offset).await
    }
}
