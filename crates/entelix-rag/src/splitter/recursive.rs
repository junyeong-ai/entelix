//! `RecursiveCharacterSplitter` — character-budget splitter that
//! prefers semantic boundaries (paragraph → line → word → char) and
//! recurses into smaller boundaries only when the larger ones would
//! produce oversized chunks.
//!
//! The canonical text-splitter shape — LangChain's
//! `RecursiveCharacterTextSplitter` covers the same algorithm, and
//! every RAG pipeline that doesn't need vendor-token-accurate
//! splitting reaches for this default. For token-accurate budgeting,
//! reach for [`TokenCountSplitter`](super::TokenCountSplitter)
//! instead — both share the same recursive-merge core under
//! [`splitter::common`](super::common) and differ only in their size
//! metric.
//!
//! ## Algorithm
//!
//! 1. Walk separators in priority order (default: `["\n\n", "\n",
//!    " ", ""]`). The empty-string separator is the always-fits
//!    fallback that splits on every character.
//! 2. At each separator: split the input. Long segments (over
//!    `chunk_size`) recurse into the next separator; short segments
//!    are accumulated greedily into chunks up to `chunk_size`.
//! 3. Once a chunk is full, the next chunk is seeded with the
//!    trailing `chunk_overlap` characters of the previous chunk so
//!    semantically-adjacent content shares context across the
//!    boundary.
//!
//! Boundary handling is *char-aware*, not byte-aware — chunk size
//! and overlap count Unicode scalar values, not UTF-8 bytes. A
//! Korean / Japanese / Devanagari corpus does not silently lose
//! the last grapheme to a mid-byte split.

use crate::document::{Document, Lineage};
use crate::splitter::TextSplitter;
use crate::splitter::common::{merge_with_overlap_metric, recurse_with_metric};

/// Default chunk size in characters. ~1000 chars maps to roughly
/// 200-300 tokens for English under `cl100k_base`, comfortably under
/// every shipping vendor's per-message ceiling.
pub const DEFAULT_CHUNK_SIZE_CHARS: usize = 1000;

/// Default overlap between consecutive chunks. ~10% of
/// [`DEFAULT_CHUNK_SIZE_CHARS`] preserves enough trailing context
/// for retrieval grounding without bloating the index.
pub const DEFAULT_CHUNK_OVERLAP_CHARS: usize = 100;

/// Default separator priority list. Paragraph break → line break →
/// word boundary → character. The empty-string fallback guarantees
/// termination even on pathological input (one giant unbroken
/// token).
pub const DEFAULT_RECURSIVE_SEPARATORS: &[&str] = &["\n\n", "\n", " ", ""];

/// Stable identifier surfaced on every produced chunk's
/// [`Lineage::splitter`](crate::Lineage::splitter) field.
const SPLITTER_NAME: &str = "recursive-character";

/// Recursive character-budget splitter.
///
/// Construct via [`Self::new`] for the default 1000-char / 100-char
/// shape, or [`Self::with_chunk_size`] / [`Self::with_chunk_overlap`]
/// / [`Self::with_separators`] for tuning. Cloning is cheap — the
/// separator list is held by `Arc` so multiple pipelines can share
/// one configured splitter.
#[derive(Clone, Debug)]
pub struct RecursiveCharacterSplitter {
    chunk_size: usize,
    chunk_overlap: usize,
    separators: std::sync::Arc<[String]>,
}

impl RecursiveCharacterSplitter {
    /// Build with the default chunk size + overlap +
    /// separator priority. The 99% case for English / Latin-script
    /// corpora.
    #[must_use]
    pub fn new() -> Self {
        Self {
            chunk_size: DEFAULT_CHUNK_SIZE_CHARS,
            chunk_overlap: DEFAULT_CHUNK_OVERLAP_CHARS,
            separators: DEFAULT_RECURSIVE_SEPARATORS
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
        }
    }

    /// Override the target chunk size in characters. Chunks larger
    /// than this are recursed; chunks smaller are accumulated
    /// greedily.
    #[must_use]
    pub const fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Override the overlap (in characters) between consecutive
    /// chunks. Must be strictly less than [`Self::chunk_size`] —
    /// equal-or-greater overlap would loop indefinitely. Values at
    /// or above the chunk size silently clamp to `chunk_size - 1`
    /// at split time.
    #[must_use]
    pub const fn with_chunk_overlap(mut self, chunk_overlap: usize) -> Self {
        self.chunk_overlap = chunk_overlap;
        self
    }

    /// Override the separator priority list. Pipelines splitting
    /// LaTeX, Python source, or other domain-specific corpora ship
    /// alternative priority lists that bias toward
    /// language-meaningful boundaries.
    #[must_use]
    pub fn with_separators<I, S>(mut self, separators: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.separators = separators.into_iter().map(Into::into).collect();
        self
    }

    /// Effective chunk size in characters.
    #[must_use]
    pub const fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Effective chunk overlap in characters.
    #[must_use]
    pub const fn chunk_overlap(&self) -> usize {
        self.chunk_overlap
    }
}

impl Default for RecursiveCharacterSplitter {
    fn default() -> Self {
        Self::new()
    }
}

impl TextSplitter for RecursiveCharacterSplitter {
    fn name(&self) -> &'static str {
        SPLITTER_NAME
    }

    fn split(&self, document: &Document) -> Vec<Document> {
        let chunk_size = self.chunk_size.max(1);
        let chunk_overlap = self.chunk_overlap.min(chunk_size.saturating_sub(1));

        let measure = |text: &str| char_count(text);
        let take_tail = |text: &str, n: usize| take_tail_chars(text, n);
        let fallback = |text: &str, n: usize| char_chunks(text, n);

        let segments = recurse_with_metric(
            &document.content,
            &self.separators,
            chunk_size,
            &measure,
            &fallback,
        );
        let texts =
            merge_with_overlap_metric(segments, chunk_size, chunk_overlap, &measure, &take_tail);
        let total = texts.len();
        if total == 0 {
            return Vec::new();
        }
        #[allow(clippy::cast_possible_truncation)]
        let total_u32 = total.min(u32::MAX as usize) as u32;
        texts
            .into_iter()
            .enumerate()
            .map(|(idx, text)| {
                #[allow(clippy::cast_possible_truncation)]
                let idx_u32 = idx.min(u32::MAX as usize) as u32;
                let lineage =
                    Lineage::from_split(document.id.clone(), idx_u32, total_u32, SPLITTER_NAME);
                document.child(text, lineage)
            })
            .collect()
    }
}

/// Char-aware fixed-size chunker — used as the always-fits
/// fallback when no separator matches.
fn char_chunks(text: &str, chunk_size: usize) -> Vec<String> {
    if chunk_size == 0 || text.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut current = String::new();
    let mut count = 0usize;
    for ch in text.chars() {
        current.push(ch);
        count += 1;
        if count == chunk_size {
            out.push(std::mem::take(&mut current));
            count = 0;
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

/// Take the last `n` characters (Unicode scalar values, not bytes)
/// of `text`. Returns the whole text when `n` exceeds its char
/// count.
fn take_tail_chars(text: &str, n: usize) -> String {
    let total = char_count(text);
    if n >= total {
        return text.to_owned();
    }
    let skip = total - n;
    text.chars().skip(skip).collect()
}

#[inline]
fn char_count(text: &str) -> usize {
    text.chars().count()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::document::Source;
    use entelix_memory::Namespace;

    fn ns() -> Namespace {
        Namespace::new(entelix_core::TenantId::new("acme"))
    }

    fn doc(content: &str) -> Document {
        Document::root("doc", content, Source::now("test://", "test"), ns())
    }

    #[test]
    fn empty_input_produces_no_chunks() {
        let chunks = RecursiveCharacterSplitter::new().split(&doc(""));
        assert!(chunks.is_empty());
    }

    #[test]
    fn small_input_produces_single_chunk_with_lineage() {
        let chunks = RecursiveCharacterSplitter::new().split(&doc("short text"));
        assert_eq!(chunks.len(), 1);
        let lineage = chunks[0].lineage.as_ref().unwrap();
        assert_eq!(lineage.chunk_index, 0);
        assert_eq!(lineage.total_chunks, 1);
        assert_eq!(lineage.splitter, "recursive-character");
        assert_eq!(lineage.parent_id.as_str(), "doc");
    }

    #[test]
    fn paragraph_split_prefers_double_newline_boundary() {
        let text = "alpha paragraph.\n\nbeta paragraph.\n\ngamma paragraph.";
        let splitter = RecursiveCharacterSplitter::new()
            .with_chunk_size(20)
            .with_chunk_overlap(0);
        let chunks = splitter.split(&doc(text));
        assert_eq!(chunks.len(), 3);
        assert!(chunks[0].content.contains("alpha"));
        assert!(chunks[1].content.contains("beta"));
        assert!(chunks[2].content.contains("gamma"));
    }

    #[test]
    fn oversized_paragraph_recurses_to_word_boundary() {
        let text = "alpha bravo charlie delta echo foxtrot golf hotel";
        let splitter = RecursiveCharacterSplitter::new()
            .with_chunk_size(20)
            .with_chunk_overlap(0);
        let chunks = splitter.split(&doc(text));
        assert!(chunks.len() > 1, "must descend past paragraph break");
        for chunk in &chunks {
            assert!(
                char_count(&chunk.content) <= 20,
                "chunk size cap honoured: {} chars",
                char_count(&chunk.content)
            );
        }
    }

    #[test]
    fn overlap_seeds_tail_into_next_chunk() {
        let text = "0123456789 abcdefghij KLMNOPQRST uvwxyz0123";
        let splitter = RecursiveCharacterSplitter::new()
            .with_chunk_size(15)
            .with_chunk_overlap(5);
        let chunks = splitter.split(&doc(text));
        assert!(chunks.len() >= 2);
        for window in chunks.windows(2) {
            let prev_tail = take_tail_chars(&window[0].content, 5);
            assert!(
                window[1].content.starts_with(&prev_tail),
                "next chunk must begin with previous tail: prev_tail={prev_tail:?}, next={:?}",
                window[1].content
            );
        }
    }

    #[test]
    fn char_aware_split_respects_unicode_boundary() {
        let text = "안녕하세요반갑습니다";
        let splitter = RecursiveCharacterSplitter::new()
            .with_chunk_size(4)
            .with_chunk_overlap(0)
            .with_separators(["", ""]);
        let chunks = splitter.split(&doc(text));
        for chunk in &chunks {
            let chars: String = chunk.content.chars().collect();
            assert_eq!(chars, chunk.content);
            assert!(char_count(&chunk.content) <= 4);
        }
        let joined: String = chunks.iter().map(|c| c.content.as_str()).collect();
        assert_eq!(joined, text);
    }

    #[test]
    fn lineage_total_chunks_matches_emitted_count() {
        let text = "para one.\n\npara two.\n\npara three.";
        let chunks = RecursiveCharacterSplitter::new()
            .with_chunk_size(15)
            .with_chunk_overlap(0)
            .split(&doc(text));
        let total = chunks.len();
        for (idx, chunk) in chunks.iter().enumerate() {
            let lineage = chunk.lineage.as_ref().unwrap();
            #[allow(clippy::cast_possible_truncation)]
            let idx_u32 = idx as u32;
            #[allow(clippy::cast_possible_truncation)]
            let total_u32 = total as u32;
            assert_eq!(lineage.chunk_index, idx_u32);
            assert_eq!(lineage.total_chunks, total_u32);
        }
    }

    #[test]
    fn child_id_carries_chunk_index_suffix() {
        let chunks = RecursiveCharacterSplitter::new()
            .with_chunk_size(5)
            .with_chunk_overlap(0)
            .split(&doc("alpha beta gamma delta"));
        for (idx, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.id.as_str(), format!("doc:{idx}"));
        }
    }

    #[test]
    fn overlap_clamped_below_chunk_size_terminates() {
        let splitter = RecursiveCharacterSplitter::new()
            .with_chunk_size(5)
            .with_chunk_overlap(100);
        let chunks = splitter.split(&doc("0123456789 abcdefghij"));
        assert!(
            !chunks.is_empty() && chunks.len() < 1000,
            "split terminated with bounded chunk count, got {}",
            chunks.len()
        );
    }
}
