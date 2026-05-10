//! Metric-agnostic split helpers shared by every recursive splitter.
//!
//! The recursive-splitter algorithm — separator-priority walk →
//! greedy merge with trailing-tail overlap → empty-separator fallback
//! — is identical regardless of whether the size budget is denominated
//! in characters, tokens, or bytes. Each splitter (e.g.
//! [`RecursiveCharacterSplitter`](super::RecursiveCharacterSplitter),
//! [`TokenCountSplitter`](super::TokenCountSplitter)) injects three
//! closures that close over its size metric:
//!
//! - `measure(text) -> usize` — how big is `text`.
//! - `take_tail(text, n) -> String` — the suffix of `text` whose
//!   measured size is the largest value `<= n`.
//! - `fallback(text, n) -> Vec<String>` — the always-fits split when
//!   no separator boundary applies (typically per-unit chunking).
//!
//! Splitters wrap these in their own structs so the public surface
//! stays one-splitter-per-metric — operators reach for the splitter
//! whose name matches the budgeting unit they care about.
//!
//! ## `chunk_size` is a soft cap
//!
//! Overlap seeding deliberately allows the chunk that follows a flush
//! to exceed `chunk_size` by up to `chunk_overlap` units before the
//! next split point — the overlap tail is *part* of the new chunk, so
//! a freshly-seeded chunk plus the next segment can briefly land
//! above the cap. Token-budget metrics additionally tolerate a small
//! BPE seam-effect drift (concatenation can shift the count by a
//! token at the chunk boundary). Operators wanting strict cap
//! adherence configure `chunk_overlap = 0`; the merge step then
//! produces chunks strictly `<= chunk_size` for additive metrics.

/// Recursively walk separators until every produced segment fits
/// `chunk_size` under the supplied `measure`. The last separator
/// (typically `""`) routes into `fallback`, which the splitter
/// guarantees always produces fitting pieces.
pub(super) fn recurse_with_metric(
    text: &str,
    separators: &[String],
    chunk_size: usize,
    measure: &dyn Fn(&str) -> usize,
    fallback: &dyn Fn(&str, usize) -> Vec<String>,
) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    if measure(text) <= chunk_size {
        return vec![text.to_owned()];
    }
    let (sep, rest) = separators
        .iter()
        .enumerate()
        .find(|(_, sep)| sep.is_empty() || text.contains(sep.as_str()))
        .map_or((None, separators), |(idx, sep)| {
            let rest = separators.get(idx + 1..).unwrap_or(&[]);
            (Some(sep.as_str()), rest)
        });

    let pieces: Vec<String> = match sep {
        Some("") | None => fallback(text, chunk_size),
        Some(sep) => split_keeping_separator(text, sep),
    };

    let mut out = Vec::with_capacity(pieces.len());
    for piece in pieces {
        if measure(&piece) <= chunk_size {
            if !piece.is_empty() {
                out.push(piece);
            }
        } else {
            out.extend(recurse_with_metric(
                &piece, rest, chunk_size, measure, fallback,
            ));
        }
    }
    out
}

/// Greedy accumulator — fold short segments into chunks up to
/// `chunk_size`, with a `chunk_overlap`-sized trailing tail seeded
/// at the start of each subsequent chunk.
///
/// Size accumulation is additive (`current_size += seg_size`) for the
/// hot path. The cap is soft — a freshly-seeded chunk plus its first
/// segment can briefly land above `chunk_size` before the next split
/// point. Operators needing strict adherence configure
/// `chunk_overlap = 0`.
pub(super) fn merge_with_overlap_metric(
    segments: Vec<String>,
    chunk_size: usize,
    chunk_overlap: usize,
    measure: &dyn Fn(&str) -> usize,
    take_tail: &dyn Fn(&str, usize) -> String,
) -> Vec<String> {
    if segments.is_empty() {
        return Vec::new();
    }
    let mut chunks: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut current_size: usize = 0;

    for segment in segments {
        let seg_size = measure(&segment);
        if current_size + seg_size > chunk_size && current_size > 0 {
            chunks.push(std::mem::take(&mut current));
            if chunk_overlap > 0
                && let Some(last) = chunks.last()
            {
                let tail = take_tail(last, chunk_overlap);
                current.push_str(&tail);
                current_size = measure(&current);
            } else {
                current_size = 0;
            }
        }
        current.push_str(&segment);
        current_size += seg_size;
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

/// Split `text` on `separator` while keeping the separator at the
/// end of every preceding piece. `"a\n\nb\n\nc"` with sep `"\n\n"`
/// yields `["a\n\n", "b\n\n", "c"]` so paragraph breaks survive the
/// merge step.
pub(super) fn split_keeping_separator(text: &str, separator: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut last_end = 0usize;
    for (idx, mat) in text.match_indices(separator) {
        let chunk_end = idx + mat.len();
        out.push(text[last_end..chunk_end].to_owned());
        last_end = chunk_end;
    }
    if last_end < text.len() {
        out.push(text[last_end..].to_owned());
    }
    out
}
