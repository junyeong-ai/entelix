//! `MarkdownStructureSplitter` — heading-aware splitter that
//! preserves the document's logical sectioning.
//!
//! Splits at ATX headings (`#`, `##`, `###`, …) so each chunk
//! corresponds to a meaningful section of the source document. The
//! heading line stays attached to its body (an "orphan heading"
//! chunk would lose retrieval context), and nested sections under a
//! parent heading flow into the same chunk until the next
//! same-or-higher heading appears.
//!
//! Operators tune the split granularity through the *heading
//! levels* configuration: with `[1, 2]` only `#` and `##` start a
//! new chunk, deeper headings stay inline. The default
//! `[1, 2, 3]` covers the typical "split at major sections, keep
//! sub-sections inline" shape.
//!
//! ## Algorithm
//!
//! 1. Walk the input line by line. Lines matching the ATX heading
//!    regex `^(#{1,6})\s+...` whose level appears in
//!    [`Self::heading_levels`] open a new section.
//! 2. Buffer lines under the current heading until the next
//!    matching heading (or end of input).
//! 3. Emit one [`Document`] per accumulated section, preserving
//!    [`Lineage`].
//!
//! Setext headings (`===` / `---` underlines) and code-fenced
//! `#`-lines (inside `\`\`\`` blocks) are intentionally NOT split
//! on — the regex anchors at line start and ignores fence-aware
//! parsing in service of zero-dependency simplicity. Documents
//! relying on setext or `#`-comments inside code blocks need the
//! recursive splitter or a future markdown-fenced companion.

use std::sync::OnceLock;

use regex::Regex;

use crate::document::{Document, Lineage};
use crate::splitter::TextSplitter;

/// Default ATX heading levels that open a new chunk. `[1, 2, 3]`
/// splits at `#`, `##`, `###`; deeper sub-headings (`####+`) stay
/// inline.
pub const DEFAULT_MARKDOWN_HEADING_LEVELS: &[u8] = &[1, 2, 3];

/// Stable identifier surfaced on every produced chunk's
/// [`Lineage::splitter`](crate::Lineage::splitter) field.
const SPLITTER_NAME: &str = "markdown-structure";

/// ATX heading regex — `^` anchors at line start, `#{1,6}` matches
/// 1-6 hashes, followed by required whitespace and the heading
/// text. Compiled once via `OnceLock` so repeated splits don't pay
/// regex compilation per call.
fn heading_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        // Safe `unwrap` — pattern is a compile-time constant whose
        // validity is verified by the regex round-trip test below.
        Regex::new(r"^(#{1,6})\s+\S").expect("heading regex compiles")
    })
}

/// Heading-aware markdown splitter.
///
/// Construct via [`Self::new`] for the default `[1, 2, 3]`
/// configuration, or [`Self::with_heading_levels`] to widen / narrow
/// the split granularity.
#[derive(Clone, Debug)]
pub struct MarkdownStructureSplitter {
    heading_levels: std::sync::Arc<[u8]>,
}

impl MarkdownStructureSplitter {
    /// Build with the default `[1, 2, 3]` heading-levels
    /// configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            heading_levels: DEFAULT_MARKDOWN_HEADING_LEVELS.into(),
        }
    }

    /// Override which ATX heading levels open a new chunk. Levels
    /// outside `1..=6` are silently ignored at split time. Order
    /// is irrelevant — duplicates are tolerated.
    #[must_use]
    pub fn with_heading_levels<I>(mut self, levels: I) -> Self
    where
        I: IntoIterator<Item = u8>,
    {
        self.heading_levels = levels.into_iter().filter(|l| (1..=6).contains(l)).collect();
        self
    }

    /// Borrow the configured heading levels.
    #[must_use]
    pub fn heading_levels(&self) -> &[u8] {
        &self.heading_levels
    }

    /// Whether `level` matches the configured split set.
    fn matches_level(&self, level: u8) -> bool {
        self.heading_levels.contains(&level)
    }
}

impl Default for MarkdownStructureSplitter {
    fn default() -> Self {
        Self::new()
    }
}

impl TextSplitter for MarkdownStructureSplitter {
    fn name(&self) -> &'static str {
        SPLITTER_NAME
    }

    fn split(&self, document: &Document) -> Vec<Document> {
        let sections = collect_sections(self, &document.content);
        let total = sections.len();
        if total == 0 {
            return Vec::new();
        }
        #[allow(clippy::cast_possible_truncation)]
        let total_u32 = total.min(u32::MAX as usize) as u32;
        sections
            .into_iter()
            .enumerate()
            .map(|(idx, content)| {
                #[allow(clippy::cast_possible_truncation)]
                let idx_u32 = idx.min(u32::MAX as usize) as u32;
                let lineage =
                    Lineage::from_split(document.id.clone(), idx_u32, total_u32, SPLITTER_NAME);
                document.child(content, lineage)
            })
            .collect()
    }
}

/// Walk lines and accumulate sections. The opening heading line
/// (if any) stays attached to the section body so retrieval hits
/// land on a heading-anchored payload.
fn collect_sections(splitter: &MarkdownStructureSplitter, text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    let mut sections: Vec<String> = Vec::new();
    let mut current = String::new();
    for line in text.split_inclusive('\n') {
        if let Some(level) = matching_heading_level(splitter, line) {
            // A section break — emit accumulated content and start a
            // new section seeded with this heading line.
            if !current.is_empty() {
                sections.push(std::mem::take(&mut current));
            }
            current.push_str(line);
            // `level` is the heading depth we matched on; bound by
            // the regex (1..=6) so the `_` discard is safe.
            let _ = level;
        } else {
            current.push_str(line);
        }
    }
    if !current.is_empty() {
        sections.push(current);
    }
    sections
}

/// Return the heading level of `line` when it matches a configured
/// split level, `None` otherwise. Lines without a heading match —
/// or with a level outside the configured set — return `None`.
fn matching_heading_level(splitter: &MarkdownStructureSplitter, line: &str) -> Option<u8> {
    let captures = heading_regex().captures(line.trim_end_matches('\n'))?;
    // Group 1 is the run of `#` characters. Length is bounded by the
    // regex repetition cap (`{1,6}`) so cast to u8 is exact.
    #[allow(clippy::cast_possible_truncation)]
    let level = captures.get(1)?.as_str().len() as u8;
    splitter.matches_level(level).then_some(level)
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
        let chunks = MarkdownStructureSplitter::new().split(&doc(""));
        assert!(chunks.is_empty());
    }

    #[test]
    fn no_headings_keeps_input_as_single_chunk() {
        let text = "Just a paragraph.\n\nAnother paragraph.\n";
        let chunks = MarkdownStructureSplitter::new().split(&doc(text));
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, text);
    }

    #[test]
    fn h1_h2_split_at_default_levels() {
        let text = "# Introduction\nIntro body.\n\n## Overview\nOverview body.\n\n## Details\nDetails body.\n";
        let chunks = MarkdownStructureSplitter::new().split(&doc(text));
        assert_eq!(chunks.len(), 3);
        assert!(chunks[0].content.starts_with("# Introduction"));
        assert!(chunks[1].content.starts_with("## Overview"));
        assert!(chunks[2].content.starts_with("## Details"));
    }

    #[test]
    fn heading_attached_to_body_not_orphaned() {
        // The first chunk's body must include both the heading line
        // and the paragraph below — an orphan heading chunk would
        // lose retrieval context.
        let text = "# Title\nbody line one.\nbody line two.\n";
        let chunks = MarkdownStructureSplitter::new().split(&doc(text));
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].content.contains("# Title"));
        assert!(chunks[0].content.contains("body line one"));
        assert!(chunks[0].content.contains("body line two"));
    }

    #[test]
    fn deeper_headings_stay_inline_under_default_config() {
        // Default config splits at 1..=3; `####` stays attached to
        // its parent section.
        let text = "## Section\nintro.\n\n#### Sub-detail\ndetail body.\n";
        let chunks = MarkdownStructureSplitter::new().split(&doc(text));
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].content.contains("#### Sub-detail"));
    }

    #[test]
    fn narrowed_levels_skip_h2_split() {
        let text = "# A\nbody A.\n\n## B\nbody B.\n";
        // Only split on H1 — H2 stays inline under its parent.
        let chunks = MarkdownStructureSplitter::new()
            .with_heading_levels([1])
            .split(&doc(text));
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].content.contains("# A"));
        assert!(chunks[0].content.contains("## B"));
    }

    #[test]
    fn lineage_carries_chunk_metadata() {
        let text = "# A\nbody.\n# B\nbody.\n";
        let chunks = MarkdownStructureSplitter::new().split(&doc(text));
        assert_eq!(chunks.len(), 2);
        for (idx, chunk) in chunks.iter().enumerate() {
            let lineage = chunk.lineage.as_ref().unwrap();
            #[allow(clippy::cast_possible_truncation)]
            let idx_u32 = idx as u32;
            assert_eq!(lineage.chunk_index, idx_u32);
            assert_eq!(lineage.total_chunks, 2);
            assert_eq!(lineage.splitter, "markdown-structure");
            assert_eq!(lineage.parent_id.as_str(), "doc");
        }
    }

    #[test]
    fn level_clamp_silently_ignores_invalid_levels() {
        // Levels outside `1..=6` (regex max) are dropped at config
        // time — `0` and `7` here disappear, leaving only `2`.
        let splitter = MarkdownStructureSplitter::new().with_heading_levels([0, 2, 7]);
        assert_eq!(splitter.heading_levels(), &[2]);
    }

    #[test]
    fn rejoined_chunks_reproduce_the_input() {
        // Chunks concatenate back to the original — splitter is
        // lossless. Critical for downstream consumers that need
        // round-trip equality (e.g. replay, audit reconstruction).
        let text = "# A\nbody A.\n\n## B\nbody B.\n\n### C\nbody C.\nfinal.\n";
        let chunks = MarkdownStructureSplitter::new().split(&doc(text));
        let joined: String = chunks.iter().map(|c| c.content.as_str()).collect();
        assert_eq!(joined, text);
    }

    #[test]
    fn child_id_carries_chunk_index_suffix() {
        let text = "# A\nbody.\n# B\nbody.\n";
        let chunks = MarkdownStructureSplitter::new().split(&doc(text));
        for (idx, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.id.as_str(), format!("doc:{idx}"));
        }
    }

    #[test]
    fn heading_regex_round_trips_levels_1_through_6() {
        // Compile-time guarantee that the regex matches every ATX
        // heading depth — this anchors the `expect("heading regex
        // compiles")` claim and validates the level extraction.
        let cases = [
            ("# h1", 1),
            ("## h2", 2),
            ("### h3", 3),
            ("#### h4", 4),
            ("##### h5", 5),
            ("###### h6", 6),
        ];
        for (line, expected_level) in cases {
            let captures = heading_regex().captures(line).unwrap();
            #[allow(clippy::cast_possible_truncation)]
            let level = captures.get(1).unwrap().as_str().len() as u8;
            assert_eq!(level, expected_level);
        }
        // 7 hashes does NOT match — markdown spec caps at 6.
        assert!(heading_regex().captures("####### too deep").is_none());
    }
}
