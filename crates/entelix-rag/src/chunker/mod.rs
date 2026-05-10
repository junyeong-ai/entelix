//! `Chunker` — async transform over a chunk sequence.
//!
//! Where [`TextSplitter`](crate::TextSplitter) is a pure-algorithm
//! slicer, `Chunker` is an async transform that may issue model
//! calls. The canonical use cases:
//!
//! - **Anthropic Contextual Retrieval (2024-09)** — for each
//!   chunk, generate a 50-100 token contextual prefix grounded in
//!   the parent document, prepend it to the chunk text. Increases
//!   retrieval accuracy ~30%; requires one LLM call per chunk.
//! - **HyDE (Hypothetical Document Embeddings)** — replace the
//!   chunk text with a generated answer-shaped paraphrase before
//!   embedding. Improves retrieval over short queries.
//! - **Chunk metadata enrichment** — extract entities / topics /
//!   sentiment from the chunk and stamp them onto
//!   [`Document::metadata`](crate::Document::metadata) for
//!   downstream filtering.
//!
//! Chunkers run *after* splitting and *before* embedding. The
//! transformed sequence may be **shorter than the input** — a
//! chunker may drop chunks that fail enrichment (see
//! [`ContextualChunker`]'s [`FailurePolicy::Skip`]) or that fail a
//! filter pass. Order is preserved: when chunk N survives, it
//! appears in the output before any surviving chunk M > N. A
//! chunker that wants to fan one chunk out into several (rare)
//! belongs on a fresh splitter, not this surface.

mod contextual;

pub use contextual::{
    CONTEXTUAL_CHUNKER_DEFAULT_INSTRUCTION, ContextualChunker, ContextualChunkerBuilder,
    FailurePolicy,
};

use async_trait::async_trait;
use entelix_core::{ExecutionContext, Result};

use crate::document::Document;

/// Async transform applied to a sequence of chunks after a
/// [`TextSplitter`](crate::TextSplitter) ran. Implementations may
/// issue LLM calls, embedding lookups, or external metadata
/// enrichment; the [`ExecutionContext`] supplies cancellation,
/// deadline, and any [`entelix_core::RunBudget`] caps the parent
/// pipeline configured.
///
/// Stamps the chunker's identity onto every transformed chunk's
/// [`Lineage::chunker_chain`](crate::Lineage::chunker_chain) so
/// the audit trail records the order of transforms a leaf
/// underwent.
#[async_trait]
pub trait Chunker: Send + Sync {
    /// Stable chunker identifier — appended to every transformed
    /// chunk's `lineage.chunker_chain`. `"contextual"`,
    /// `"hyde"`, `"entity-tag"`, etc.
    fn name(&self) -> &'static str;

    /// Transform every chunk in `chunks`. Returning the input
    /// vector unchanged is a valid no-op; the transformed
    /// sequence may also be shorter (a filter chunker / a chunker
    /// running [`FailurePolicy::Skip`] over a partial-failure
    /// batch). Order MUST be preserved among surviving chunks —
    /// the ingestion pipeline relies on positional alignment with
    /// the embedder's output.
    ///
    /// The pipeline calls `process` once per chunker per ingestion
    /// run; chunkers that batch their LLM calls over the whole
    /// vector pay the round-trip cost once, not per chunk.
    async fn process(&self, chunks: Vec<Document>, ctx: &ExecutionContext)
    -> Result<Vec<Document>>;
}
