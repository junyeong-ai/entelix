//! `SystemPrompt` — ordered system-prompt blocks with optional
//! per-block cache control.
//!
//! The IR represents the system prompt as a refcounted slice of
//! [`SystemBlock`] values. The most common case (a single text
//! block, no cache directive) is constructible directly via
//! `SystemPrompt::from("text")` or [`SystemPrompt::text`]; the
//! multi-block / cached form arrives via [`SystemBlock::cached`] +
//! `Vec<SystemBlock>` collected and converted via [`From<Vec<…>>`].
//!
//! Storage is `Arc<[SystemBlock]>` so per-dispatch cloning of the
//! enclosing [`crate::ir::ModelRequest`] is an atomic refcount bump
//! rather than a deep walk of every block's text. Codecs read
//! through the [`Self::blocks`] borrow — every `&prompt.blocks()[i]`
//! site continues to see `&[SystemBlock]` unchanged.
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

use std::sync::Arc;

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
///
/// Storage is `Arc<[SystemBlock]>` so cloning a `SystemPrompt`
/// (and the [`crate::ir::ModelRequest`] that holds it) is an atomic
/// refcount bump. The hot-path cost of stamping the same prompt on
/// every model call is O(1) regardless of block count or block
/// length.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SystemPrompt {
    blocks: Arc<[SystemBlock]>,
}

impl Default for SystemPrompt {
    fn default() -> Self {
        Self::empty()
    }
}

impl SystemPrompt {
    /// Empty prompt — semantically equivalent to "no system prompt
    /// configured."
    #[must_use]
    pub fn empty() -> Self {
        Self {
            blocks: Arc::from([]),
        }
    }

    /// Single-block prompt from a text string.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            blocks: Arc::from([SystemBlock::text(text)]),
        }
    }

    /// Single-block prompt with the supplied cache directive.
    #[must_use]
    pub fn cached(text: impl Into<String>, cache: CacheControl) -> Self {
        Self {
            blocks: Arc::from([SystemBlock::cached(text, cache)]),
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

    /// Whether any block carries a cache directive — used by codecs
    /// without native cache support to decide whether to emit a
    /// `LossyEncode` warning.
    #[must_use]
    pub fn any_cached(&self) -> bool {
        self.blocks.iter().any(|b| b.cache_control.is_some())
    }

    /// Map every block through `f`, returning a fresh prompt whose
    /// shared storage is rebuilt from the transformed sequence. The
    /// canonical PII-redaction surface — redactors clone each
    /// block, scrub the text in place, and assemble a fresh
    /// `Arc<[SystemBlock]>` once. The original `Arc` is never
    /// mutated; callers retaining a clone of the source prompt see
    /// it untouched.
    ///
    /// (`Arc::try_unwrap` fast-pathing the sole-owner case isn't
    /// available because Rust's stdlib does not implement
    /// `try_unwrap` for `Arc<[T]>` — slice DSTs aren't `Sized`.)
    #[must_use]
    pub fn map_blocks<F>(&self, mut f: F) -> Self
    where
        F: FnMut(&mut SystemBlock),
    {
        let blocks: Vec<SystemBlock> = self
            .blocks
            .iter()
            .map(|b| {
                let mut clone = b.clone();
                f(&mut clone);
                clone
            })
            .collect();
        Self {
            blocks: Arc::from(blocks),
        }
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
            blocks: Arc::from([block]),
        }
    }
}

impl From<Vec<SystemBlock>> for SystemPrompt {
    fn from(blocks: Vec<SystemBlock>) -> Self {
        Self {
            blocks: Arc::from(blocks),
        }
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
        let p = SystemPrompt::from(vec![
            SystemBlock::text("first"),
            SystemBlock::text("second"),
        ]);
        assert_eq!(p.concat_text(), "first\n\nsecond");
    }

    #[test]
    fn map_blocks_lets_redactor_rebuild_with_transformed_text() {
        let p = SystemPrompt::from(vec![SystemBlock::text("alpha"), SystemBlock::text("beta")]);
        let upper = p.map_blocks(|block| {
            block.text = block.text.to_uppercase();
        });
        assert_eq!(upper.concat_text(), "ALPHA\n\nBETA");
        // Original prompt untouched — `Arc<[SystemBlock]>` is shared
        // immutably, so map_blocks returns a fresh prompt.
        assert_eq!(p.concat_text(), "alpha\n\nbeta");
    }

    #[test]
    fn clone_is_atomic_refcount_bump() {
        // `Arc::ptr_eq` confirms clones share the same allocation —
        // the design property that retires the per-dispatch deep
        // walk over every block's text on `ModelRequest` clone.
        let p = SystemPrompt::cached("long stable preamble".repeat(100), CacheControl::one_hour());
        let cloned = p.clone();
        assert!(Arc::ptr_eq(&p.blocks, &cloned.blocks));
    }

    #[test]
    fn round_trips_via_serde_when_cached() {
        let p = SystemPrompt::cached("x", CacheControl::five_minutes());
        let json = serde_json::to_string(&p).unwrap();
        let back: SystemPrompt = serde_json::from_str(&json).unwrap();
        assert_eq!(p, back);
    }
}
