//! # entelix-memory
//!
//! Tier-3 cross-thread persistent knowledge. Defines `Store<V>`, `Namespace`
//! (mandatory `tenant_id` — invariant 11 / F2 mitigation), the `Embedder` /
//! `Retriever` / `VectorStore` traits, and the five `LangChain`-style
//! patterns: `BufferMemory`, `SummaryMemory`, `EntityMemory`,
//! `SemanticMemory<E, V>`, `EpisodicMemory<V>`. Concrete `Embedder` /
//! `VectorStore` / `Retriever` impls ship in companion crates
//!.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(html_root_url = "https://docs.rs/entelix-memory/0.5.3")]
#![deny(missing_docs)]
// Doc-prose lints fire on legitimate proper nouns (LangChain,
// Neo4j, ArangoDB, BTreeMap) and on long opening paragraphs that
// explain trait intent; we accept the trade-off so docs read as
// natural prose rather than artificially split sentences.
#![allow(
    clippy::doc_markdown,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::option_if_let_else,
    clippy::significant_drop_tightening,
    clippy::too_long_first_doc_paragraph
)]

mod buffer;
mod consolidating;
mod consolidation;
mod embedding_retriever;
mod entity;
mod episodic;
mod graph;
mod in_memory_vector_store;
mod metered;
mod namespace;
mod rerank;
mod semantic;
mod store;
mod summary;
mod traits;
mod vector;

pub use buffer::{BufferMemory, PolicyExtras};
pub use consolidating::{ConsolidatingBufferMemory, Summarizer};
pub use consolidation::{
    ConsolidationContext, ConsolidationPolicy, NeverConsolidate, OnMessageCount, OnTokenBudget,
};
pub use embedding_retriever::EmbeddingRetriever;
pub use entity::{EntityMemory, EntityRecord};
pub use episodic::{Episode, EpisodeId, EpisodicMemory};
pub use graph::{Direction, EdgeId, GraphHop, GraphMemory, InMemoryGraphMemory, NodeId};
pub use in_memory_vector_store::InMemoryVectorStore;
pub use metered::{CostCalculatorAdapter, EmbeddingCostCalculator, MeteredEmbedder};
pub use namespace::{Namespace, NamespacePrefix};
pub use rerank::MmrReranker;
pub use semantic::{SemanticMemory, SemanticMemoryBackend};
pub use store::{InMemoryStore, PutOptions, Store};
pub use summary::SummaryMemory;
pub use traits::{
    Document, DocumentId, Embedder, Embedding, EmbeddingUsage, IdentityReranker, RerankedDocument,
    Reranker, RetrievalQuery, Retriever, VectorFilter, VectorStore,
};
pub use vector::{first_non_finite_vector_value, validate_vector_shape};
