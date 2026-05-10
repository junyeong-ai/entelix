//! `TokenCountSplitter` — token-budget splitter built on top of an
//! operator-supplied [`TokenCounter`].
//!
//! Shares the recursive-merge core with
//! [`RecursiveCharacterSplitter`](super::RecursiveCharacterSplitter)
//! via [`splitter::common`](super::common); the two differ only in
//! the size metric. Operators reach for this splitter when chunks
//! must fit a vendor-token-count budget — typical embedding models
//! cap context at 512 / 8192 tokens, and char-count approximation
//! mis-estimates by 20-50% on multilingual or code-heavy corpora.
//!
//! ## Algorithm
//!
//! Same separator-priority recursion as the char splitter — the
//! only differences are:
//!
//! 1. **Size metric** — [`TokenCounter::count`] replaces character
//!    counting. Multilingual and CJK corpora get vendor-accurate
//!    chunk sizing instead of a heuristic that under-counts CJK
//!    tokens (one Korean syllable typically tokenises to 2-3 BPE
//!    tokens, not the ~3 chars a char-based budget would assume).
//! 2. **Tail extraction** — overlap seeding bisects on suffix length
//!    until the largest suffix fitting `chunk_overlap` tokens is
//!    found. Bisection cost is `O(log N)` `count` calls per chunk
//!    seal, amortised across the chunk's tokens.
//! 3. **Soft-cap discipline** — `chunk_size` is honoured strictly
//!    when `chunk_overlap = 0`. With overlap engaged, the splitter
//!    follows the same soft-cap contract as
//!    [`RecursiveCharacterSplitter`](super::RecursiveCharacterSplitter):
//!    the overlap-seeded prefix plus the next segment can briefly
//!    land above `chunk_size` before the next split point, and BPE
//!    seam effects can additionally shift the concatenated count by
//!    a token or two at chunk boundaries. Operators wanting a strict
//!    cap configure `chunk_overlap = 0`.
//!
//! ## Pairing with a `TokenCounter`
//!
//! Wire any [`TokenCounter`] impl — the
//! [`entelix_core::ByteCountTokenCounter`] zero-dep default, the
//! [`TiktokenCounter`](https://docs.rs/entelix-tokenizer-tiktoken)
//! companion for OpenAI BPE accuracy, or any other vendor /
//! locale-specific counter shipping over the same trait.

use std::sync::Arc;

use entelix_core::TokenCounter;

use crate::document::{Document, Lineage};
use crate::splitter::TextSplitter;
use crate::splitter::common::{merge_with_overlap_metric, recurse_with_metric};
use crate::splitter::recursive::DEFAULT_RECURSIVE_SEPARATORS;

/// Default chunk size in tokens. `512` matches the typical embedding
/// context window (`text-embedding-3-small` and `-large` both cap at
/// 8191 tokens; chunking under 512 leaves headroom for query +
/// instruction tokens at retrieval time).
pub const DEFAULT_CHUNK_SIZE_TOKENS: usize = 512;

/// Default overlap between consecutive chunks in tokens. ~12.5% of
/// [`DEFAULT_CHUNK_SIZE_TOKENS`] preserves enough trailing context
/// for retrieval grounding without bloating the index.
pub const DEFAULT_CHUNK_OVERLAP_TOKENS: usize = 64;

/// Stable identifier surfaced on every produced chunk's
/// [`Lineage::splitter`](crate::Lineage::splitter) field.
const SPLITTER_NAME: &str = "token-count";

/// Recursive token-budget splitter.
///
/// Construct via [`Self::new`] with any `Arc<C>` where
/// `C: TokenCounter + ?Sized + 'static`. Both concrete
/// (`Arc<TiktokenCounter>`) and type-erased
/// (`Arc<dyn TokenCounter>`) inputs are accepted — the splitter
/// monomorphises per concrete counter for inlined hot-path
/// dispatch, or falls through to dyn dispatch when the operator
/// passes an erased Arc. Chain [`Self::with_chunk_size`] /
/// [`Self::with_chunk_overlap`] / [`Self::with_separators`] for
/// tuning. Cloning is cheap — the counter sits behind an [`Arc`]
/// and the separator list is held by `Arc<[String]>`.
#[derive(Clone)]
pub struct TokenCountSplitter<C: TokenCounter + ?Sized + 'static = dyn TokenCounter> {
    counter: Arc<C>,
    chunk_size: usize,
    chunk_overlap: usize,
    separators: Arc<[String]>,
}

impl<C: TokenCounter + ?Sized + 'static> std::fmt::Debug for TokenCountSplitter<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenCountSplitter")
            .field("counter", &self.counter.encoding_name())
            .field("chunk_size", &self.chunk_size)
            .field("chunk_overlap", &self.chunk_overlap)
            .field("separators", &self.separators)
            .finish()
    }
}

impl<C: TokenCounter + ?Sized + 'static> TokenCountSplitter<C> {
    /// Build with the supplied [`TokenCounter`] and the default
    /// 512-token / 64-token shape.
    #[must_use]
    pub fn new(counter: Arc<C>) -> Self {
        Self {
            counter,
            chunk_size: DEFAULT_CHUNK_SIZE_TOKENS,
            chunk_overlap: DEFAULT_CHUNK_OVERLAP_TOKENS,
            separators: DEFAULT_RECURSIVE_SEPARATORS
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
        }
    }

    /// Override the target chunk size in tokens.
    #[must_use]
    pub const fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Override the overlap (in tokens) between consecutive chunks.
    /// Values at or above the chunk size silently clamp to
    /// `chunk_size - 1` at split time so the recursion terminates.
    #[must_use]
    pub const fn with_chunk_overlap(mut self, chunk_overlap: usize) -> Self {
        self.chunk_overlap = chunk_overlap;
        self
    }

    /// Override the separator priority list. Defaults to
    /// `["\n\n", "\n", " ", ""]` — paragraph → line → word → unit
    /// fallback. Pipelines splitting source code or LaTeX ship
    /// alternative priorities.
    #[must_use]
    pub fn with_separators<I, S>(mut self, separators: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.separators = separators.into_iter().map(Into::into).collect();
        self
    }

    /// Effective chunk size in tokens.
    #[must_use]
    pub const fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Effective chunk overlap in tokens.
    #[must_use]
    pub const fn chunk_overlap(&self) -> usize {
        self.chunk_overlap
    }

    /// Borrow the wired token counter — surfaces
    /// [`TokenCounter::encoding_name`] for OTel attribute emission
    /// and operator diagnostics.
    #[must_use]
    pub const fn counter(&self) -> &Arc<C> {
        &self.counter
    }
}

impl<C: TokenCounter + ?Sized + 'static> TextSplitter for TokenCountSplitter<C> {
    fn name(&self) -> &'static str {
        SPLITTER_NAME
    }

    fn split(&self, document: &Document) -> Vec<Document> {
        let chunk_size = self.chunk_size.max(1);
        let chunk_overlap = self.chunk_overlap.min(chunk_size.saturating_sub(1));

        let counter = Arc::clone(&self.counter);
        let measure = move |text: &str| count_tokens(&*counter, text);
        let counter_for_tail = Arc::clone(&self.counter);
        let take_tail = move |text: &str, n: usize| take_tail_tokens(&*counter_for_tail, text, n);
        let counter_for_fallback = Arc::clone(&self.counter);
        let fallback = move |text: &str, n: usize| token_chunks(&*counter_for_fallback, text, n);

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

fn count_tokens<C: TokenCounter + ?Sized>(counter: &C, text: &str) -> usize {
    usize::try_from(counter.count(text)).unwrap_or(usize::MAX)
}

/// Token-aware tail extraction: bisect on suffix char-length to find
/// the largest suffix whose token count is `<= target`. Cost is
/// `O(log L)` token counts where `L` is the input char-length.
fn take_tail_tokens<C: TokenCounter + ?Sized>(counter: &C, text: &str, target: usize) -> String {
    if text.is_empty() || target == 0 {
        return String::new();
    }
    let total = count_tokens(counter, text);
    if target >= total {
        return text.to_owned();
    }
    let chars: Vec<char> = text.chars().collect();
    let total_chars = chars.len();
    let mut lo: usize = 0;
    let mut hi: usize = total_chars;
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        let suffix_start = total_chars.saturating_sub(mid);
        let suffix: String = chars.iter().skip(suffix_start).collect();
        if count_tokens(counter, &suffix) <= target {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let suffix_start = total_chars.saturating_sub(lo);
    chars.iter().skip(suffix_start).collect()
}

/// Always-fits token-budget fallback. Walks chars greedily,
/// flushing a chunk every time the next char would push the chunk's
/// token count over `chunk_size`. Cost is `O(C)` token counts where
/// `C` is the input char-length — acceptable because this is the
/// terminator for the recursion and runs only on segments that
/// every separator already failed to split below the cap.
fn token_chunks<C: TokenCounter + ?Sized>(
    counter: &C,
    text: &str,
    chunk_size: usize,
) -> Vec<String> {
    if chunk_size == 0 || text.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if count_tokens(counter, &current) > chunk_size {
            // Roll back the last char into the next chunk.
            current.pop();
            if !current.is_empty() {
                out.push(std::mem::take(&mut current));
            }
            current.push(ch);
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::document::Source;
    use entelix_core::ByteCountTokenCounter;
    use entelix_memory::Namespace;

    fn ns() -> Namespace {
        Namespace::new(entelix_core::TenantId::new("acme"))
    }

    fn doc(content: &str) -> Document {
        Document::root("doc", content, Source::now("test://", "test"), ns())
    }

    fn byte_counter() -> Arc<dyn TokenCounter> {
        Arc::new(ByteCountTokenCounter::new())
    }

    #[test]
    fn empty_input_produces_no_chunks() {
        let chunks = TokenCountSplitter::new(byte_counter()).split(&doc(""));
        assert!(chunks.is_empty());
    }

    #[test]
    fn small_input_produces_single_chunk_with_lineage() {
        let chunks = TokenCountSplitter::new(byte_counter()).split(&doc("short"));
        assert_eq!(chunks.len(), 1);
        let lineage = chunks[0].lineage.as_ref().unwrap();
        assert_eq!(lineage.chunk_index, 0);
        assert_eq!(lineage.total_chunks, 1);
        assert_eq!(lineage.splitter, "token-count");
        assert_eq!(lineage.parent_id.as_str(), "doc");
    }

    #[test]
    fn paragraph_split_prefers_double_newline_boundary() {
        // ByteCountTokenCounter counts as div_ceil(bytes / 4). Each
        // 16-byte paragraph is 4 tokens; chunk_size=5 fits one but
        // not two.
        let text = "alpha paragraph\n\nbeta paragraph\n\ngamma paragraph";
        let splitter = TokenCountSplitter::new(byte_counter())
            .with_chunk_size(5)
            .with_chunk_overlap(0);
        let chunks = splitter.split(&doc(text));
        assert_eq!(chunks.len(), 3);
        assert!(chunks[0].content.contains("alpha"));
        assert!(chunks[1].content.contains("beta"));
        assert!(chunks[2].content.contains("gamma"));
    }

    #[test]
    fn cap_enforced_on_every_chunk() {
        let splitter = TokenCountSplitter::new(byte_counter())
            .with_chunk_size(8)
            .with_chunk_overlap(0);
        let text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november";
        let chunks = splitter.split(&doc(text));
        assert!(chunks.len() > 1);
        for chunk in &chunks {
            let count = byte_counter().count(&chunk.content);
            assert!(
                count <= 8,
                "chunk over cap: {} tokens, content={:?}",
                count,
                chunk.content
            );
        }
    }

    #[test]
    fn overlap_seeds_tail_into_next_chunk() {
        let text = "0123456789 abcdefghij KLMNOPQRST uvwxyz0123";
        let splitter = TokenCountSplitter::new(byte_counter())
            .with_chunk_size(5)
            .with_chunk_overlap(1);
        let chunks = splitter.split(&doc(text));
        assert!(chunks.len() >= 2);
        for window in chunks.windows(2) {
            let tail = take_tail_tokens(&byte_counter(), &window[0].content, 1);
            // Empty tail (when chunk smaller than 1 token boundary)
            // is the trivial case; only non-empty tails carry a
            // semantic claim.
            if !tail.is_empty() {
                assert!(
                    window[1].content.starts_with(&tail),
                    "next chunk must begin with previous tail: tail={tail:?}, next={:?}",
                    window[1].content
                );
            }
        }
    }

    #[test]
    fn unicode_input_split_preserves_grapheme_boundary() {
        // Korean text: byte-counter hits the cap fast (each syllable
        // is 3 UTF-8 bytes ~ 1 token). Verify the splitter never
        // breaks mid-grapheme — Rust's String type guarantees valid
        // UTF-8 so any panic here would surface as `chars()` decode
        // failure.
        let text = "안녕하세요반갑습니다오늘은좋은날이에요";
        let splitter = TokenCountSplitter::new(byte_counter())
            .with_chunk_size(2)
            .with_chunk_overlap(0)
            .with_separators(["", ""]);
        let chunks = splitter.split(&doc(text));
        for chunk in &chunks {
            let chars: String = chunk.content.chars().collect();
            assert_eq!(
                chars, chunk.content,
                "chunk must be valid UTF-8 with no mid-grapheme cut"
            );
        }
        let joined: String = chunks.iter().map(|c| c.content.as_str()).collect();
        assert_eq!(joined, text, "round-trip must reproduce input");
    }

    #[test]
    fn child_id_carries_chunk_index_suffix() {
        let chunks = TokenCountSplitter::new(byte_counter())
            .with_chunk_size(2)
            .with_chunk_overlap(0)
            .split(&doc("alpha beta gamma delta"));
        for (idx, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.id.as_str(), format!("doc:{idx}"));
        }
    }

    #[test]
    fn lineage_total_chunks_matches_emitted_count() {
        let text = "para one.\n\npara two.\n\npara three.";
        let chunks = TokenCountSplitter::new(byte_counter())
            .with_chunk_size(4)
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
    fn overlap_clamped_below_chunk_size_terminates() {
        let splitter = TokenCountSplitter::new(byte_counter())
            .with_chunk_size(3)
            .with_chunk_overlap(100);
        let chunks = splitter.split(&doc("0123456789 abcdefghij KLMNOP uvwxyz"));
        assert!(
            !chunks.is_empty() && chunks.len() < 1000,
            "split terminated with bounded chunk count, got {}",
            chunks.len()
        );
    }

    #[test]
    fn counter_accessor_exposes_encoding_name() {
        let splitter = TokenCountSplitter::new(byte_counter());
        assert_eq!(splitter.counter().encoding_name(), "byte-count-naive");
    }

    #[test]
    fn debug_lists_encoding_not_arc_pointer() {
        let splitter = TokenCountSplitter::new(byte_counter());
        let debug = format!("{splitter:?}");
        assert!(debug.contains("byte-count-naive"));
        assert!(debug.contains("chunk_size"));
    }

    #[test]
    fn take_tail_tokens_handles_empty_and_oversize_target() {
        let counter = byte_counter();
        assert_eq!(take_tail_tokens(&counter, "", 5), "");
        assert_eq!(take_tail_tokens(&counter, "abc", 0), "");
        assert_eq!(take_tail_tokens(&counter, "abc", 1000), "abc");
    }

    #[test]
    fn take_tail_tokens_returns_largest_fitting_suffix() {
        let counter = byte_counter();
        // ByteCountTokenCounter: 4 bytes per token (rounds up).
        // "abcdefgh" = 8 bytes = 2 tokens. Asking for 1-token tail
        // should return the trailing 4-byte slice.
        let tail = take_tail_tokens(&counter, "abcdefgh", 1);
        assert_eq!(counter.count(&tail), 1);
        assert!("abcdefgh".ends_with(&tail));
    }
}
