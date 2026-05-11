//! # entelix-rag
//!
//! Algorithmic primitives for retrieval-augmented generation
//! pipelines — `Document` with provenance + lineage, plus the
//! `DocumentLoader` / `TextSplitter` / `Chunker` trait surface
//! every RAG path composes around.
//!
//! ## Position
//!
//! 2026-era agentic RAG (Contextual Retrieval, Self-RAG, CRAG,
//! Adaptive-RAG) is no longer a side pipeline — it's the agent's
//! baseline working memory. This crate ships the *algorithmic*
//! primitives (splitters, chunkers, ingestion composition) that
//! every consumer reaches for. Concrete source connectors (S3,
//! Notion, Confluence, GDrive, …) live in companion crates so the
//! core surface stays small and dependency-light.
//!
//! ## Surface
//!
//! - [`Document`] — RAG-shaped document with [`Source`] (where it
//!   came from), [`Lineage`] (split / chunk ancestry), and
//!   [`entelix_memory::Namespace`] (multi-tenant boundary). The
//!   retrieval-side [`entelix_memory::Document`] (with similarity
//!   score) is a *result* shape; this is the *ingestion* shape.
//! - [`DocumentLoader`] — async source-side trait. Streams to keep
//!   ingestion memory-bounded over arbitrarily large corpora.
//! - [`TextSplitter`] — sync algorithmic primitive. Slices a
//!   `Document` into smaller `Document`s preserving `Lineage`.
//! - [`Chunker`] — async transform over a chunk sequence. LLM-call
//!   capable (Anthropic Contextual Retrieval, HyDE, query
//!   decomposition).
//!
//! ## What lives in companion crates
//!
//! - **Source connectors** — `entelix-rag-s3`, `entelix-rag-notion`,
//!   `entelix-rag-confluence`, `entelix-rag-fs` (filesystem-backed,
//!   invariant 9 exemption).
//! - **Vendor-accurate tokenizers** — `entelix-tokenizer-tiktoken`,
//!   `entelix-tokenizer-hf`, locale-aware companions
//!   (Korean / Japanese morphology). The
//!   [`entelix_core::TokenCounter`] trait is the integration
//!   surface; this crate's [`TokenCountSplitter`] is generic over
//!   any `C: TokenCounter + ?Sized + 'static` (default
//!   `dyn TokenCounter`) so concrete `Arc<TiktokenCounter>` and
//!   type-erased `Arc<dyn TokenCounter>` plug in interchangeably.
//!   Vendor accuracy is a counter swap, not a splitter rewrite.
//!
//! ## Why algorithmic primitives only
//!
//! The LangChain ecosystem's mistake was bundling 100+ source
//! connectors into the core surface — version churn became
//! unmanageable. entelix-rag's coreis explicitly small (4 traits +
//! `Document` + provenance types) so vendor-specific loaders ship
//! independently and never gate the core's release cadence. The
//! algorithmic primitives (splitters, chunkers, ingestion
//! composition) ARE universal so they live here.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(html_root_url = "https://docs.rs/entelix-rag/0.4.1")]
#![deny(missing_docs)]
#![allow(
    clippy::doc_markdown,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions,
    clippy::too_long_first_doc_paragraph,
    // Tests use unwrap/expect liberally; the splitter modules call
    // `Regex::new(...).expect(...)` on a compile-time-constant
    // pattern (the round-trip test pins regex correctness).
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::unwrap_used
)]

mod chunker;
mod corrective;
mod document;
mod loader;
mod pipeline;
mod splitter;

pub use chunker::{
    CONTEXTUAL_CHUNKER_DEFAULT_INSTRUCTION, Chunker, ContextualChunker, ContextualChunkerBuilder,
    FailurePolicy,
};
pub use corrective::{
    CORRECTIVE_RAG_AGENT_NAME, CorrectiveRagState, CragConfig, DEFAULT_GENERATOR_SYSTEM_PROMPT,
    DEFAULT_GRADER_INSTRUCTION, DEFAULT_MAX_REWRITE_ATTEMPTS, DEFAULT_MIN_CORRECT_FRACTION,
    DEFAULT_RETRIEVAL_TOP_K, DEFAULT_REWRITER_INSTRUCTION, GradeVerdict, LlmQueryRewriter,
    LlmQueryRewriterBuilder, LlmRetrievalGrader, LlmRetrievalGraderBuilder, QueryRewriter,
    RetrievalGrader, build_corrective_rag_graph, create_corrective_rag_agent,
};
pub use document::{Document, DocumentId, Lineage, Source};
pub use loader::{DocumentLoader, DocumentStream};
pub use pipeline::{
    IngestError, IngestReport, IngestionPipeline, IngestionPipelineBuilder, PROVENANCE_METADATA_KEY,
};
pub use splitter::{
    DEFAULT_CHUNK_OVERLAP_CHARS, DEFAULT_CHUNK_OVERLAP_TOKENS, DEFAULT_CHUNK_SIZE_CHARS,
    DEFAULT_CHUNK_SIZE_TOKENS, DEFAULT_MARKDOWN_HEADING_LEVELS, DEFAULT_RECURSIVE_SEPARATORS,
    MarkdownStructureSplitter, RecursiveCharacterSplitter, TextSplitter, TokenCountSplitter,
};
