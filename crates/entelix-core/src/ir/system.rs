//! `SystemPrompt` — ordered system-prompt blocks with optional
//! per-block cache control.
//!
//! The IR represents the system prompt as a sequence of
//! [`SystemBlock`] values. The most common case (a single text block,
//! no cache directive) is constructible directly via
//! `SystemPrompt::from("text")`; the multi-block / cached form
//! arrives via [`SystemPrompt::with_block`] or
//! [`SystemBlock::cached`].
//!
//! Codecs route blocks to vendor-canonical channels:
//!
//! - **Anthropic Messages / Bedrock Converse (Claude)** — emit
//!   `system: [{type: "text", text, cache_control: {...}}]` per
//!   block when `cache_control` is set.
//! - **OpenAI Chat / Responses / Gemini** — concatenate block text
//!   into a single instruction string; emit
//!   [`crate::ir::ModelWarning::LossyEncode`] when any block
//!   carries a `cache_control` the codec cannot represent
//!   natively.

use serde::{Deserialize, Serialize};

use crate::ir::cache::CacheControl;

/// One block of the system prompt — an ordered text payload with an
/// optional cache directive.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct SystemBlock {
    /// Block text. Codecs may reject blocks that exceed
    /// vendor-imposed length limits.
    pub text: String,
    /// Optional per-block cache directive. `None` = pass through
    /// uncached.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

impl SystemBlock {
    /// Plain text block with no cache directive.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            cache_control: None,
        }
    }

    /// Text block with the supplied cache directive.
    #[must_use]
    pub fn cached(text: impl Into<String>, cache: CacheControl) -> Self {
        Self {
            text: text.into(),
            cache_control: Some(cache),
        }
    }

    /// Borrow the block text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.text
    }
}

/// Ordered sequence of [`SystemBlock`]s. An empty `SystemPrompt`
/// represents "no system prompt" — codecs treat it as if the field
/// were absent.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SystemPrompt {
    blocks: Vec<SystemBlock>,
}

impl SystemPrompt {
    /// Empty prompt — semantically equivalent to "no system prompt
    /// configured."
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Single-block prompt from a text string.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            blocks: vec![SystemBlock::text(text)],
        }
    }

    /// Single-block prompt with the supplied cache directive.
    #[must_use]
    pub fn cached(text: impl Into<String>, cache: CacheControl) -> Self {
        Self {
            blocks: vec![SystemBlock::cached(text, cache)],
        }
    }

    /// Whether the prompt contains zero blocks.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Number of blocks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Borrow the blocks in order.
    #[must_use]
    pub fn blocks(&self) -> &[SystemBlock] {
        &self.blocks
    }

    /// Append a block. Returns `self` for builder-style chaining.
    #[must_use]
    pub fn with_block(mut self, block: SystemBlock) -> Self {
        self.blocks.push(block);
        self
    }

    /// Append a block — `&mut self` mutator for callers that
    /// already own the prompt by value.
    pub fn push(&mut self, block: SystemBlock) {
        self.blocks.push(block);
    }

    /// Whether any block carries a cache directive — used by codecs
    /// without native cache support to decide whether to emit a
    /// `LossyEncode` warning.
    #[must_use]
    pub fn any_cached(&self) -> bool {
        self.blocks.iter().any(|b| b.cache_control.is_some())
    }

    /// Iterator over mutable block references — used by PII
    /// redactors to scrub text in-place without re-allocating the
    /// outer prompt.
    pub fn blocks_mut(&mut self) -> impl Iterator<Item = &mut SystemBlock> {
        self.blocks.iter_mut()
    }

    /// Concatenate every block's text with `\n\n` separators —
    /// the canonical "flatten to single string" rendering codecs
    /// without per-block channels rely on.
    #[must_use]
    pub fn concat_text(&self) -> String {
        self.blocks
            .iter()
            .map(|b| b.text.as_str())
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

impl From<&str> for SystemPrompt {
    fn from(s: &str) -> Self {
        Self::text(s)
    }
}

impl From<String> for SystemPrompt {
    fn from(s: String) -> Self {
        Self::text(s)
    }
}

impl From<SystemBlock> for SystemPrompt {
    fn from(block: SystemBlock) -> Self {
        Self {
            blocks: vec![block],
        }
    }
}

impl From<Vec<SystemBlock>> for SystemPrompt {
    fn from(blocks: Vec<SystemBlock>) -> Self {
        Self { blocks }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn empty_prompt_has_no_blocks_and_renders_to_empty_string() {
        let p = SystemPrompt::empty();
        assert!(p.is_empty());
        assert_eq!(p.len(), 0);
        assert_eq!(p.concat_text(), "");
        assert!(!p.any_cached());
    }

    #[test]
    fn from_str_produces_single_uncached_block() {
        let p: SystemPrompt = "Be terse.".into();
        assert_eq!(p.len(), 1);
        assert_eq!(p.blocks()[0].text, "Be terse.");
        assert!(p.blocks()[0].cache_control.is_none());
    }

    #[test]
    fn cached_constructor_attaches_cache_control() {
        let p = SystemPrompt::cached("stable instructions", CacheControl::one_hour());
        assert!(p.any_cached());
        assert_eq!(
            p.blocks()[0].cache_control.unwrap().ttl,
            crate::ir::cache::CacheTtl::OneHour
        );
    }

    #[test]
    fn concat_text_joins_blocks_with_double_newline() {
        let p = SystemPrompt::default()
            .with_block(SystemBlock::text("first"))
            .with_block(SystemBlock::text("second"));
        assert_eq!(p.concat_text(), "first\n\nsecond");
    }

    #[test]
    fn blocks_mut_lets_redactor_walk_in_place() {
        let mut p = SystemPrompt::default()
            .with_block(SystemBlock::text("alpha"))
            .with_block(SystemBlock::text("beta"));
        for block in p.blocks_mut() {
            block.text = block.text.to_uppercase();
        }
        assert_eq!(p.concat_text(), "ALPHA\n\nBETA");
    }

    #[test]
    fn round_trips_via_serde_when_cached() {
        let p = SystemPrompt::cached("x", CacheControl::five_minutes());
        let json = serde_json::to_string(&p).unwrap();
        let back: SystemPrompt = serde_json::from_str(&json).unwrap();
        assert_eq!(p, back);
    }
}
