//! Corrective Retrieval (CRAG, Yan et al. 2024) primitives.
//!
//! Splits the retrieval-quality decision and the
//! query-correction step out of any specific recipe so operators
//! plug their own grading + rewrite policy under a stable trait
//! surface. The graph topology that wires these primitives —
//! retrieve → grade → (rewrite ↔ retry / generate) — lives in a
//! separate recipe slice; this module ships the *primitives* the
//! recipe composes, plus reference LLM-driven impls every
//! deployment can start from.
//!
//! ## Why traits, not a single monolithic recipe
//!
//! Grader and rewriter prompts are inherently corpus-shaped — a
//! technical-docs corpus needs different relevance criteria than a
//! customer-support transcript corpus needs different rewrite
//! style than a code-search index. Hard-coding either inside a
//! recipe locks operators out of the per-corpus tuning the CRAG
//! literature emphasises is decisive. The trait split also lets
//! deployments swap LLM-driven graders for keyword / heuristic
//! variants (cheaper, sometimes more accurate on narrow
//! corpora).
//!
//! ## Surface
//!
//! - [`GradeVerdict`] — three-way relevance verdict per the CRAG
//!   paper (`Correct` / `Ambiguous` / `Incorrect`).
//! - [`RetrievalGrader`] — async trait. Given query + retrieved
//!   document, returns a [`GradeVerdict`].
//! - [`QueryRewriter`] — async trait. Given the original query +
//!   the prior failed attempts, returns a corrected query string.
//! - [`LlmRetrievalGrader<M>`] — reference grader that asks any
//!   `Runnable<Vec<Message>, Message>` model to classify
//!   relevance.
//! - [`LlmQueryRewriter<M>`] — reference rewriter that asks the
//!   model for a corrected query string.

mod grader;
mod recipe;
mod rewriter;

pub use grader::{
    DEFAULT_GRADER_INSTRUCTION, GradeVerdict, LlmRetrievalGrader, LlmRetrievalGraderBuilder,
    RetrievalGrader,
};
pub use recipe::{
    CORRECTIVE_RAG_AGENT_NAME, CorrectiveRagState, CragConfig, DEFAULT_GENERATOR_SYSTEM_PROMPT,
    DEFAULT_MAX_REWRITE_ATTEMPTS, DEFAULT_MIN_CORRECT_FRACTION, DEFAULT_RETRIEVAL_TOP_K,
    build_corrective_rag_graph, create_corrective_rag_agent,
};
pub use rewriter::{
    DEFAULT_REWRITER_INSTRUCTION, LlmQueryRewriter, LlmQueryRewriterBuilder, QueryRewriter,
};
