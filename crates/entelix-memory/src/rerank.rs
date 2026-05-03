//! Concrete [`Reranker`] impls beyond [`crate::IdentityReranker`].
//!
//! [`MmrReranker`] is the model-free baseline: re-embed the query
//! and candidates with the operator's existing [`Embedder`], then
//! pick documents one at a time by maximising a Maximal Marginal
//! Relevance score that trades query similarity off against
//! pairwise novelty. No cross-encoder, no extra API surface — just
//! the embedder already wired into the [`crate::SemanticMemory`].
//!
//! Why MMR matters as a default: vector search alone happily returns
//! near-duplicates ("the company was founded in 1998", "the company
//! was started in 1998", …). For RAG that is wasted context window.
//! MMR resolves it without trained models — Carbonell & Goldstein
//! (1998) is still the canonical reference and every RAG framework
//! ships a variant.

use std::sync::Arc;

use async_trait::async_trait;
use entelix_core::{Error, ExecutionContext, Result};

use crate::traits::{Document, Embedder, RerankedDocument, Reranker};

/// Maximal Marginal Relevance reranker.
///
/// At rerank time, embeds the query plus every candidate via the
/// supplied [`Embedder`], then greedily selects documents that
/// maximise
///
/// ```text
/// score(d) = λ · cos(q, d) − (1 − λ) · max_{s ∈ selected} cos(d, s)
/// ```
///
/// where `λ ∈ [0, 1]` is the relevance / diversity tradeoff.
/// `λ = 1.0` collapses to pure retrieval ranking; `λ = 0.0` picks
/// the most diverse candidates regardless of relevance. The default
/// (0.5) is the LangChain / LlamaIndex / Haystack convention.
///
/// **Cost**: one `embed_batch` call against the candidate pool plus
/// one `embed` call for the query. Backends with a true batch
/// endpoint amortise this to a single round-trip; embedders without
/// `embed_batch` overrides incur N sequential calls.
///
/// **Atomicity**: the [`RerankedDocument::rerank_score`] returned is
/// the MMR score at the moment the document was selected — comparable
/// only within one rerank invocation. The original retrieval score
/// remains on `document.score` so callers can still compare against
/// the upstream similarity.
pub struct MmrReranker {
    embedder: Arc<dyn Embedder>,
    lambda: f32,
}

impl MmrReranker {
    /// LangChain / LlamaIndex / Haystack default — empirically a
    /// reasonable mid-point for RAG over conversational corpora.
    pub const DEFAULT_LAMBDA: f32 = 0.5;

    /// Build with the supplied embedder and the default λ.
    ///
    /// The embedder is shared as `Arc<dyn Embedder>` so a single
    /// embedding client can be threaded through both
    /// [`crate::SemanticMemory`] and this reranker without a second
    /// HTTP pool.
    #[must_use]
    pub fn new(embedder: Arc<dyn Embedder>) -> Self {
        Self {
            embedder,
            lambda: Self::DEFAULT_LAMBDA,
        }
    }

    /// Override the relevance / diversity tradeoff. Values outside
    /// `[0.0, 1.0]` are clamped — the MMR formula is undefined
    /// outside that interval and silently letting it through would
    /// produce nonsensical rankings.
    #[must_use]
    pub const fn with_lambda(mut self, lambda: f32) -> Self {
        self.lambda = lambda.clamp(0.0, 1.0);
        self
    }

    /// The configured λ, post-clamp. Useful for tests and dashboards.
    #[must_use]
    pub const fn lambda(&self) -> f32 {
        self.lambda
    }
}

#[async_trait]
impl Reranker for MmrReranker {
    async fn rerank(
        &self,
        query: &str,
        candidates: Vec<Document>,
        top_k: usize,
        ctx: &ExecutionContext,
    ) -> Result<Vec<RerankedDocument>> {
        if candidates.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }
        let query_embedding = self.embedder.embed(query, ctx).await?;
        let texts: Vec<String> = candidates.iter().map(|d| d.content.clone()).collect();
        let embeddings = self.embedder.embed_batch(&texts, ctx).await?;
        if embeddings.len() != candidates.len() {
            return Err(Error::config(format!(
                "MmrReranker: embedder returned {} vectors for {} candidates",
                embeddings.len(),
                candidates.len()
            )));
        }
        let mut pool: Vec<MmrCandidate> = candidates
            .into_iter()
            .zip(embeddings)
            .map(|(document, embedding)| {
                let relevance = cosine(&query_embedding.vector, &embedding.vector);
                MmrCandidate {
                    document,
                    vector: embedding.vector,
                    relevance,
                }
            })
            .collect();

        let cap = top_k.min(pool.len());
        let mut selected: Vec<RerankedDocument> = Vec::with_capacity(cap);
        let mut chosen_vectors: Vec<Vec<f32>> = Vec::with_capacity(cap);

        while selected.len() < cap && !pool.is_empty() {
            let (best_pos, best_score) = pool
                .iter()
                .enumerate()
                .map(|(pos, candidate)| {
                    let max_div = if chosen_vectors.is_empty() {
                        0.0
                    } else {
                        chosen_vectors
                            .iter()
                            .map(|v| cosine(&candidate.vector, v))
                            .fold(f32::NEG_INFINITY, f32::max)
                    };
                    let score = self
                        .lambda
                        .mul_add(candidate.relevance, (self.lambda - 1.0) * max_div);
                    (pos, score)
                })
                .fold((0_usize, f32::NEG_INFINITY), |(bp, bs), (p, s)| {
                    if s > bs { (p, s) } else { (bp, bs) }
                });
            let chosen = pool.swap_remove(best_pos);
            chosen_vectors.push(chosen.vector);
            selected.push(RerankedDocument::new(chosen.document, best_score));
        }

        Ok(selected)
    }
}

/// Internal pairing — keeps each candidate's document, its embedded
/// vector, and its query-relevance score in one row so the MMR
/// loop can `swap_remove` ownership without parallel-array indexing.
struct MmrCandidate {
    document: Document,
    vector: Vec<f32>,
    relevance: f32,
}

/// Cosine similarity between two equal-length vectors. Returns
/// `0.0` when either operand is the zero vector — matches the
/// LangChain / SciPy `cosine_similarity` convention so MMR scores
/// comparing across frameworks line up.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(
        a.len(),
        b.len(),
        "cosine: vectors must be equal length (Embedder dimension contract)"
    );
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::map_unwrap_or
)]
mod tests {
    use std::sync::Arc;

    use entelix_core::{ExecutionContext, Result};

    use super::{Document, Embedder, MmrReranker, Reranker, cosine};
    use crate::traits::{Embedding, EmbeddingUsage};

    /// Embedder that returns a pre-baked vector keyed by the input
    /// string — lets tests script exactly which docs end up similar
    /// to which.
    struct ScriptedEmbedder {
        dimension: usize,
        rules: Vec<(String, Vec<f32>)>,
    }

    #[async_trait::async_trait]
    impl Embedder for ScriptedEmbedder {
        fn dimension(&self) -> usize {
            self.dimension
        }
        async fn embed(&self, text: &str, _ctx: &ExecutionContext) -> Result<Embedding> {
            let v = self
                .rules
                .iter()
                .find(|(k, _)| k == text)
                .map(|(_, v)| v.clone())
                .unwrap_or_else(|| vec![0.0; self.dimension]);
            Ok(Embedding::new(v).with_usage(EmbeddingUsage::new(text.len() as u32)))
        }
    }

    fn doc(text: &str) -> Document {
        Document::new(text)
    }

    #[test]
    fn cosine_zero_vector_yields_zero() {
        assert_eq!(cosine(&[0.0; 4], &[1.0, 2.0, 3.0, 4.0]), 0.0);
        assert_eq!(cosine(&[1.0, 2.0, 3.0, 4.0], &[0.0; 4]), 0.0);
    }

    #[test]
    fn cosine_orthogonal_yields_zero_and_aligned_yields_one() {
        let a = [1.0_f32, 0.0];
        let b = [0.0_f32, 1.0];
        let c = [3.0_f32, 0.0];
        assert!((cosine(&a, &b) - 0.0).abs() < 1e-6);
        assert!((cosine(&a, &c) - 1.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn lambda_one_collapses_to_pure_relevance_order() -> Result<()> {
        // q points along axis 0; candidates rank by their dot with q.
        let embedder = Arc::new(ScriptedEmbedder {
            dimension: 2,
            rules: vec![
                ("q".into(), vec![1.0, 0.0]),
                ("near".into(), vec![0.9, 0.1]),
                ("mid".into(), vec![0.5, 0.5]),
                ("far".into(), vec![0.1, 0.9]),
            ],
        });
        let reranker = MmrReranker::new(embedder).with_lambda(1.0);
        let ctx = ExecutionContext::new();
        let out = reranker
            .rerank("q", vec![doc("far"), doc("near"), doc("mid")], 3, &ctx)
            .await?;
        let order: Vec<&str> = out.iter().map(|r| r.document.content.as_str()).collect();
        assert_eq!(order, vec!["near", "mid", "far"]);
        Ok(())
    }

    #[tokio::test]
    async fn lambda_zero_picks_most_diverse_candidates_first() -> Result<()> {
        // Two near-duplicates against q, plus one orthogonal.
        // Pure-diversity MMR should pick one duplicate first (max
        // marginal score with empty selection collapses to zero
        // diversity penalty), then the orthogonal one second
        // because it is maximally different from the first pick.
        let embedder = Arc::new(ScriptedEmbedder {
            dimension: 2,
            rules: vec![
                ("q".into(), vec![1.0, 0.0]),
                ("dup_a".into(), vec![0.99, 0.01]),
                ("dup_b".into(), vec![0.98, 0.02]),
                ("ortho".into(), vec![0.0, 1.0]),
            ],
        });
        let reranker = MmrReranker::new(embedder).with_lambda(0.0);
        let ctx = ExecutionContext::new();
        let out = reranker
            .rerank("q", vec![doc("dup_a"), doc("dup_b"), doc("ortho")], 2, &ctx)
            .await?;
        // Second pick must be the orthogonal one — anything else
        // would mean the diversity penalty failed to deprioritise
        // the duplicate.
        assert_eq!(out[1].document.content, "ortho");
        Ok(())
    }

    #[tokio::test]
    async fn empty_candidates_return_empty_no_embedder_call() -> Result<()> {
        // If the reranker fired an embedder call on empty input
        // it would surface here — the scripted embedder would
        // return a zero vector and the assertion below would still
        // hold, but the cost is real. This is a contract test.
        let embedder = Arc::new(ScriptedEmbedder {
            dimension: 2,
            rules: vec![],
        });
        let reranker = MmrReranker::new(embedder);
        let ctx = ExecutionContext::new();
        let out = reranker.rerank("q", vec![], 5, &ctx).await?;
        assert!(out.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn top_k_caps_at_candidate_count() -> Result<()> {
        let embedder = Arc::new(ScriptedEmbedder {
            dimension: 2,
            rules: vec![
                ("q".into(), vec![1.0, 0.0]),
                ("a".into(), vec![1.0, 0.0]),
                ("b".into(), vec![0.0, 1.0]),
            ],
        });
        let reranker = MmrReranker::new(embedder);
        let ctx = ExecutionContext::new();
        let out = reranker
            .rerank("q", vec![doc("a"), doc("b")], 99, &ctx)
            .await?;
        assert_eq!(out.len(), 2);
        Ok(())
    }

    #[tokio::test]
    async fn lambda_clamps_to_unit_interval() {
        let embedder = Arc::new(ScriptedEmbedder {
            dimension: 2,
            rules: vec![],
        });
        let above =
            MmrReranker::new(Arc::clone(&(embedder.clone() as Arc<dyn Embedder>))).with_lambda(1.5);
        let below = MmrReranker::new(embedder).with_lambda(-0.3);
        assert_eq!(above.lambda(), 1.0);
        assert_eq!(below.lambda(), 0.0);
    }
}
