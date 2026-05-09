//! `Embedder`, `Retriever`, `VectorStore` traits + `Document`.
//!
//! Per, entelix 1.0 ships only the traits. Concrete impls
//! (`OpenAI`/Voyage/Cohere embedders, qdrant/lancedb vector stores,
//! BM25 retrievers) land in 1.1 companion crates.

use async_trait::async_trait;
use entelix_core::{ExecutionContext, Result};
use serde::{Deserialize, Serialize};

use crate::namespace::Namespace;

/// One retrieved document with optional similarity score.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Document {
    /// Backend-assigned stable id, when the originating
    /// [`VectorStore`] mints one. Pass back to
    /// [`VectorStore::update`] / [`VectorStore::delete`] to mutate
    /// or remove this document. `None` for embedded usages where
    /// stable identity isn't tracked.
    #[serde(default)]
    pub doc_id: Option<DocumentId>,
    /// Body text. Implementations may store any UTF-8 payload.
    pub content: String,
    /// Free-form metadata — stored alongside the document for filtering
    /// / display. Use a JSON object by convention.
    #[serde(default)]
    pub metadata: serde_json::Value,
    /// Similarity score from the retriever, if available. Higher = better
    /// match. Comparable only within a single query's result set.
    #[serde(default)]
    pub score: Option<f32>,
}

impl Document {
    /// Convenience: build a document with empty metadata and no score.
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            doc_id: None,
            content: content.into(),
            metadata: serde_json::Value::Null,
            score: None,
        }
    }

    /// Builder-style metadata setter.
    #[must_use]
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }

    /// Builder-style doc_id setter — backends use this when
    /// surfacing previously-indexed documents so callers can mutate
    /// or delete via the same id.
    #[must_use]
    pub fn with_doc_id(mut self, id: impl Into<DocumentId>) -> Self {
        self.doc_id = Some(id.into());
        self
    }
}

/// Token-accounting metadata an [`Embedder`] reports alongside the
/// computed vector. Mirrors the chat-model `Usage` shape so cost
/// meters can charge embedding calls with the same machinery they
/// use for completions.
///
/// Marked `#[non_exhaustive]` so future fields (cached_tokens, total
/// for cost, latency_ms) are forward-compatible.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct EmbeddingUsage {
    /// Tokens the embedder counted as input. `0` when the backend
    /// does not surface a count (some local impls do not).
    pub input_tokens: u32,
}

impl EmbeddingUsage {
    /// Build a usage record with the supplied input-token count.
    #[must_use]
    pub const fn new(input_tokens: u32) -> Self {
        Self { input_tokens }
    }
}

/// One embedded text's vector plus optional usage metadata.
///
/// `usage` is `None` when the backend does not surface token
/// accounting (in-process stub embedders, hash-based encoders).
/// Cost-aware backends (`OpenAI`, Voyage, Cohere) return `Some` so
/// downstream meters can charge per call.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Embedding {
    /// The encoded vector. Length always equals
    /// [`Embedder::dimension`].
    pub vector: Vec<f32>,
    /// Token-accounting metadata when the backend reports it.
    #[serde(default)]
    pub usage: Option<EmbeddingUsage>,
}

impl Embedding {
    /// Build an embedding with no usage metadata. Stub backends and
    /// tests use this when they have no token count to report.
    #[must_use]
    pub const fn new(vector: Vec<f32>) -> Self {
        Self {
            vector,
            usage: None,
        }
    }

    /// Builder-style usage attachment.
    #[must_use]
    pub const fn with_usage(mut self, usage: EmbeddingUsage) -> Self {
        self.usage = Some(usage);
        self
    }
}

/// Text → vector encoder.
///
/// Implementations are typically backed by a remote API
/// (`OpenAI` / Voyage / Cohere). Per F10, instances are wrapped in `Arc`
/// at the call boundary; do not create a new client per call.
///
/// **Override `embed_batch`** when the underlying API supports
/// batch inference. The provided default loops sequentially via
/// `embed`, which is correct but allocates one HTTP call per
/// document — avoid in production.
#[async_trait]
pub trait Embedder: Send + Sync + 'static {
    /// Output vector dimension. Used by `VectorStore` impls to validate
    /// inserts against the configured index dimension.
    fn dimension(&self) -> usize;

    /// Embed one input string. Returns the vector plus optional
    /// usage metadata so downstream cost meters can charge the
    /// call without a second round-trip.
    async fn embed(&self, text: &str, ctx: &ExecutionContext) -> Result<Embedding>;

    /// Batch embed. Default impl runs sequentially via `embed`,
    /// polling [`ExecutionContext::is_cancelled`] between iterations
    /// so a long batch bails out within one `embed` round-trip of
    /// cancellation rather than draining the full pool.
    /// Implementations that support a true batch endpoint
    /// **should** override — sequential calls amplify network
    /// latency by `N`.
    async fn embed_batch(
        &self,
        texts: &[String],
        ctx: &ExecutionContext,
    ) -> Result<Vec<Embedding>> {
        let mut out = Vec::with_capacity(texts.len());
        for text in texts {
            if ctx.is_cancelled() {
                return Err(entelix_core::Error::Cancelled);
            }
            out.push(self.embed(text, ctx).await?);
        }
        Ok(out)
    }
}

/// Declarative description of one retrieval call.
///
/// Carries the text query plus optional knobs: a metadata filter, a
/// minimum score floor, and a top-k cap. Future hybrid-search
/// dimensions (per-field boosts, dense+keyword fusion, reranker
/// hints) ride on `#[non_exhaustive]` without breaking call sites.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct RetrievalQuery {
    /// Free-text query. Backends interpret it however they like —
    /// dense vector embed-and-search, BM25 token match, hybrid fusion.
    pub text: String,
    /// Maximum number of hits to return. Backends truncate at this
    /// cap; exceeding it is a backend bug.
    pub top_k: usize,
    /// Minimum similarity score required to include a hit. `None`
    /// surfaces every hit regardless of score; useful when score
    /// semantics differ across backends. Comparable only within a
    /// single query's result set.
    pub min_score: Option<f32>,
    /// Metadata predicate. `None` is identity (no filter). Backends
    /// that cannot honour the supplied [`VectorFilter`] variant fall
    /// back to filterless retrieval and emit a `LossyEncode`-style
    /// warning at the operator layer.
    pub filter: Option<VectorFilter>,
}

impl RetrievalQuery {
    /// Build a query with sensible defaults — no filter, no
    /// floor, top-k cap supplied.
    #[must_use]
    pub fn new(text: impl Into<String>, top_k: usize) -> Self {
        Self {
            text: text.into(),
            top_k,
            min_score: None,
            filter: None,
        }
    }

    /// Attach a metadata filter. Builder-style.
    #[must_use]
    pub fn with_filter(mut self, filter: VectorFilter) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Attach a minimum-score floor. Builder-style.
    #[must_use]
    pub const fn with_min_score(mut self, min_score: f32) -> Self {
        self.min_score = Some(min_score);
        self
    }
}

/// Returns documents ranked by relevance to a [`RetrievalQuery`].
///
/// Backed by anything from a BM25 index to a remote search API; the
/// signature is intentionally backend-agnostic. Implementations
/// honour the query's `top_k`, `min_score`, and `filter` to the
/// extent the underlying backend supports them.
#[async_trait]
pub trait Retriever: Send + Sync + 'static {
    /// Look up matches for `query`.
    async fn retrieve(
        &self,
        query: RetrievalQuery,
        ctx: &ExecutionContext,
    ) -> Result<Vec<Document>>;
}

/// Stable identifier for an indexed document. Backends mint these
/// at insertion time; passing the same id to [`VectorStore::update`]
/// or [`VectorStore::delete`] is the canonical way to mutate or
/// remove a previously-indexed document.
pub type DocumentId = String;

/// Predicate against [`Document::metadata`] used by
/// [`VectorStore::search_filtered`]. Backends translate this into
/// their native filter language (`pgvector` `WHERE`, qdrant `Filter`,
/// lancedb `where`); backends that cannot honour a given variant
/// fall back to filterless search and emit a `LossyEncode`-style
/// warning at the operator layer.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum VectorFilter {
    /// Always passes — useful as identity inside `And`/`Or` chains.
    All,
    /// `metadata.<key> == value`.
    Eq {
        /// Metadata key path (dotted notation).
        key: String,
        /// Comparison value.
        value: serde_json::Value,
    },
    /// `metadata.<key> < value`. Numeric semantics; non-numeric
    /// operands produce an empty result set.
    Lt {
        /// Metadata key path.
        key: String,
        /// Comparison value (numeric).
        value: serde_json::Value,
    },
    /// `metadata.<key> <= value`. Inclusive variant of [`Self::Lt`].
    Lte {
        /// Metadata key path.
        key: String,
        /// Comparison value (numeric).
        value: serde_json::Value,
    },
    /// `metadata.<key> > value`.
    Gt {
        /// Metadata key path.
        key: String,
        /// Comparison value (numeric).
        value: serde_json::Value,
    },
    /// `metadata.<key> >= value`. Inclusive variant of [`Self::Gt`].
    Gte {
        /// Metadata key path.
        key: String,
        /// Comparison value (numeric).
        value: serde_json::Value,
    },
    /// `min <= metadata.<key> <= max`. Closed interval — backends
    /// that natively support range queries can push this down as a
    /// single index probe instead of decomposing into `And(Gte, Lte)`.
    Range {
        /// Metadata key path.
        key: String,
        /// Lower bound (inclusive). Numeric.
        min: serde_json::Value,
        /// Upper bound (inclusive). Numeric.
        max: serde_json::Value,
    },
    /// `metadata.<key>` is one of `values`. Empty `values` matches
    /// no documents — equivalent to a no-op filter that can be used
    /// to short-circuit zero-result queries without consulting the
    /// index.
    In {
        /// Metadata key path.
        key: String,
        /// Allowed values.
        values: Vec<serde_json::Value>,
    },
    /// `metadata.<key>` is present (any value, including `null`).
    /// Distinguishes "field unset" from "field set to null".
    Exists {
        /// Metadata key path.
        key: String,
    },
    /// All children must match.
    And(Vec<Self>),
    /// At least one child must match.
    Or(Vec<Self>),
    /// Negate the inner filter.
    Not(Box<Self>),
}

/// Vector index keyed by [`Namespace`]. Backed by qdrant, lancedb,
/// pgvector, etc. in companion crates.
///
/// **Layering** — this is **tier 1 (primitive)** of the
/// semantic-memory three-tier architecture. Operators implement
/// `VectorStore` once per backend; the bundle
/// [`crate::SemanticMemory<E, V>`] (tier 2) and the consumer trait
/// [`crate::SemanticMemoryBackend`] (tier 3) compose it into the
/// agent-facing surface automatically. Take `Namespace` as a
/// per-call parameter so a single store instance serves many
/// tenants.
///
/// Every async method accepts an [`ExecutionContext`] so backends
/// can honour caller-side cancellation and deadlines (CLAUDE.md
/// §"Cancellation"). The `delete` / `update` / `add_batch` /
/// `search_filtered` methods have default impls so simple backends
/// only need `add` and `search` — production backends override
/// every method for efficiency and correctness.
///
/// **Atomicity**: the default `update` impl is non-atomic
/// (delete-then-add): concurrent `search` calls observe a momentary
/// gap. Backends that support transactional updates **must**
/// override.
#[async_trait]
pub trait VectorStore: Send + Sync + 'static {
    /// Vector dimension this index expects.
    fn dimension(&self) -> usize;

    /// Add a document with its pre-computed vector to the index.
    /// Implementations validate `vector.len() == self.dimension()`.
    async fn add(
        &self,
        ctx: &ExecutionContext,
        ns: &Namespace,
        document: Document,
        vector: Vec<f32>,
    ) -> Result<()>;

    /// Search for the top `top_k` nearest documents to `query_vector`.
    async fn search(
        &self,
        ctx: &ExecutionContext,
        ns: &Namespace,
        query_vector: &[f32],
        top_k: usize,
    ) -> Result<Vec<Document>>;

    /// Delete a document by its backend-assigned id. Default impl
    /// returns `Error::Config` — backends without a stable id space
    /// must override or document the lifecycle.
    async fn delete(&self, _ctx: &ExecutionContext, _ns: &Namespace, _doc_id: &str) -> Result<()> {
        Err(entelix_core::Error::config(
            "VectorStore::delete is not supported by this backend",
        ))
    }

    /// Replace an existing document's vector and metadata. Default
    /// impl chains `delete` + `add` (non-atomic — concurrent
    /// searches observe a gap); backends with atomic-update support
    /// **must** override.
    async fn update(
        &self,
        ctx: &ExecutionContext,
        ns: &Namespace,
        doc_id: &str,
        document: Document,
        vector: Vec<f32>,
    ) -> Result<()> {
        self.delete(ctx, ns, doc_id).await?;
        self.add(ctx, ns, document, vector).await
    }

    /// Insert many documents at once. Default impl loops over `add`,
    /// polling [`ExecutionContext::is_cancelled`] between iterations
    /// so a cancelled caller releases the index lock within one
    /// `add` round-trip instead of completing the full batch.
    /// Backends that support a native batch endpoint **should**
    /// override — sequential calls amplify network latency by `N`.
    async fn add_batch(
        &self,
        ctx: &ExecutionContext,
        ns: &Namespace,
        items: Vec<(Document, Vec<f32>)>,
    ) -> Result<()> {
        for (doc, vec) in items {
            if ctx.is_cancelled() {
                return Err(entelix_core::Error::Cancelled);
            }
            self.add(ctx, ns, doc, vec).await?;
        }
        Ok(())
    }

    /// Top-K nearest matches with a metadata filter pushed down to
    /// the index. Default impl returns [`entelix_core::Error::Config`] —
    /// silently dropping the filter would return wrong results, so
    /// the trait makes the backend's lack of filter support
    /// explicit. Backends with filter support **must** override.
    async fn search_filtered(
        &self,
        _ctx: &ExecutionContext,
        _ns: &Namespace,
        _query_vector: &[f32],
        _top_k: usize,
        _filter: &VectorFilter,
    ) -> Result<Vec<Document>> {
        Err(entelix_core::Error::config(
            "VectorStore::search_filtered is not supported by this backend; \
             override the trait method to push filters down to the index",
        ))
    }

    /// Count documents in the namespace, optionally narrowed by a
    /// filter. Used by dashboards reporting per-tenant index sizes
    /// and by memory-budget enforcement (skip indexing when the
    /// namespace is at its cap).
    ///
    /// Default impl returns [`entelix_core::Error::Config`] —
    /// counting requires either a backend-native COUNT or a full
    /// scan, both of which are operator-visible cost decisions
    /// that should not be silently approximated.
    async fn count(
        &self,
        _ctx: &ExecutionContext,
        _ns: &Namespace,
        _filter: Option<&VectorFilter>,
    ) -> Result<usize> {
        Err(entelix_core::Error::config(
            "VectorStore::count is not supported by this backend; \
             override the trait method to surface index cardinality",
        ))
    }

    /// Enumerate documents in the namespace, optionally narrowed by
    /// a filter. `limit` caps the page size; `offset` is the page
    /// start (cursor-style pagination semantics depend on the
    /// backend). Returned documents may omit their vectors — the
    /// method is for inspection / pagination, not for retrieval.
    ///
    /// Default impl returns [`entelix_core::Error::Config`] —
    /// listing requires a stable iteration order that not every
    /// vector backend exposes (e.g. ANN indices give no useful
    /// ordering across calls).
    async fn list(
        &self,
        _ctx: &ExecutionContext,
        _ns: &Namespace,
        _filter: Option<&VectorFilter>,
        _limit: usize,
        _offset: usize,
    ) -> Result<Vec<Document>> {
        Err(entelix_core::Error::config(
            "VectorStore::list is not supported by this backend; \
             override the trait method to enumerate documents in the index",
        ))
    }
}

/// One reranked document paired with the score the [`Reranker`] assigned.
///
/// Two scores ride together: the inner `Document::score` carries the
/// original retrieval score from the [`VectorStore`] (or `None` if the
/// store did not surface one), and `rerank_score` carries the reranker's
/// own score. Keeping them distinct preserves explainability — UIs and
/// dashboards can show "the embedding ranked this 0.82, the cross-encoder
/// ranked it 0.41 → moved from rank 1 to rank 7" without ambiguity, and
/// downstream filters can threshold on whichever score the deployment
/// trusts.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RerankedDocument {
    /// The reranked document. Its `score` field still holds the
    /// retrieval score from the originating [`VectorStore`] — the
    /// reranker does not mutate it.
    pub document: Document,
    /// The reranker's score for this document at this query.
    /// Comparable only within a single rerank call. Higher = better
    /// per the reranker's scoring model.
    pub rerank_score: f32,
}

impl RerankedDocument {
    /// Build a reranked document from a candidate and a fresh score.
    pub const fn new(document: Document, rerank_score: f32) -> Self {
        Self {
            document,
            rerank_score,
        }
    }
}

/// Re-rank a candidate document set against the originating query.
///
/// Used after a `VectorStore::search` to apply MMR / cross-encoder
/// scoring / time-decay before returning results to the caller. The
/// default identity impl ([`IdentityReranker`]) preserves the
/// retrieval order and copies each candidate's retrieval score into
/// `rerank_score` so the return type is uniform regardless of which
/// reranker is wired up; production deployments substitute with
/// cross-encoder or MMR implementations from companion crates.
#[async_trait]
pub trait Reranker: Send + Sync + 'static {
    /// Re-order (and optionally trim) the candidate list, attaching
    /// a reranker-specific score to each survivor. The returned
    /// `Vec` MUST contain only documents from the input candidates
    /// — rerankers cannot fabricate new content.
    async fn rerank(
        &self,
        query: &str,
        candidates: Vec<Document>,
        top_k: usize,
        ctx: &ExecutionContext,
    ) -> Result<Vec<RerankedDocument>>;
}

/// No-op [`Reranker`]: returns the first `top_k` candidates in the
/// order the underlying `VectorStore` produced them, copying the
/// retrieval score into [`RerankedDocument::rerank_score`] so
/// downstream consumers see a uniform shape.
#[derive(Clone, Copy, Debug, Default)]
pub struct IdentityReranker;

#[async_trait]
impl Reranker for IdentityReranker {
    async fn rerank(
        &self,
        _query: &str,
        mut candidates: Vec<Document>,
        top_k: usize,
        _ctx: &ExecutionContext,
    ) -> Result<Vec<RerankedDocument>> {
        candidates.truncate(top_k);
        Ok(candidates
            .into_iter()
            .map(|doc| {
                let score = doc.score.unwrap_or(0.0);
                RerankedDocument::new(doc, score)
            })
            .collect())
    }
}
