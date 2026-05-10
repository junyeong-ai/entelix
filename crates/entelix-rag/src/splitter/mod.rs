//! `TextSplitter` — sync algorithmic primitive.
//!
//! Slices one [`Document`] into a sequence of smaller `Document`s
//! while preserving every chunk's [`Lineage`](crate::Lineage)
//! reference back to the parent. The output sequence is owned —
//! the input is consumed (or borrowed; both shapes ship via
//! [`split`](Self::split)'s `&Document` signature so callers can
//! re-use the parent if the splitter is non-destructive).
//!
//! Sync by contract — splitters are pure algorithms over text
//! (boundary detection, token-budget bucketing, markdown structure
//! parsing). Mirrors the [`entelix_core::TokenCounter`] discipline:
//! low-level primitives stay pure so they compose with locks and
//! hot paths. Splitters that depend on async resources (LLM-call
//! based segmentation) belong on [`Chunker`](crate::Chunker), not
//! `TextSplitter`.

mod common;
mod markdown;
mod recursive;
mod token_count;

pub use markdown::{DEFAULT_MARKDOWN_HEADING_LEVELS, MarkdownStructureSplitter};
pub use recursive::{
    DEFAULT_CHUNK_OVERLAP_CHARS, DEFAULT_CHUNK_SIZE_CHARS, DEFAULT_RECURSIVE_SEPARATORS,
    RecursiveCharacterSplitter,
};
pub use token_count::{
    DEFAULT_CHUNK_OVERLAP_TOKENS, DEFAULT_CHUNK_SIZE_TOKENS, TokenCountSplitter,
};

use crate::document::Document;

/// Pure-algorithm slice of a [`Document`] into smaller documents.
///
/// Implementations preserve every produced chunk's
/// [`Lineage`](crate::Lineage) so audit / replay flows can walk
/// from any leaf back to the original load. The input
/// `Document::source` and `Document::namespace` flow through
/// every chunk unchanged via [`Document::child`].
pub trait TextSplitter: Send + Sync {
    /// Stable splitter identifier — surfaces on every produced
    /// chunk's [`Lineage::splitter`](crate::Lineage::splitter)
    /// field. `"recursive-character"`, `"markdown-structure"`,
    /// `"token-count"`, etc.
    fn name(&self) -> &'static str;

    /// Slice `document` and return the resulting chunks. Returning
    /// a single-element vec equal to the input is a valid no-op
    /// (e.g. when the document already fits the target size); an
    /// empty vec is also valid (e.g. content is whitespace-only).
    ///
    /// Implementations MUST stamp [`Lineage`](crate::Lineage) onto
    /// every produced chunk — typically via the
    /// [`Document::child`] helper, which copies source / metadata /
    /// namespace over from the parent.
    fn split(&self, document: &Document) -> Vec<Document>;
}
