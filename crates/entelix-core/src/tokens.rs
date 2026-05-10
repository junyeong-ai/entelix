//! `TokenCounter` — operator-supplied token-count surface.
//!
//! Tokens are the unit budget caps and chunk boundaries are denominated
//! in. The vendor's wire-level usage report (the `Usage` block on a
//! `ModelResponse`) gives the *post-flight* count. `TokenCounter` is
//! the *pre-flight* counterpart — what operators reach for when:
//!
//! - **`RunBudget` pre-flight check** — refuse a request whose
//!   estimated input would already exceed the configured input-token
//!   cap, before paying the round-trip cost.
//! - **RAG chunking** — `entelix-rag::TokenCountSplitter` slices a
//!   document so each chunk lands under the model's per-message
//!   ceiling.
//! - **Content-economy budgeting** — system prompt + tools + history
//!   sum estimation when assembling a request.
//!
//! Vendor-accurate counters live in companion crates —
//! [`entelix-tokenizer-tiktoken`](https://docs.rs/entelix-tokenizer-tiktoken)
//! for the OpenAI BPE family (`cl100k_base`, `o200k_base`,
//! `p50k_base`, `r50k_base`) and
//! [`entelix-tokenizer-hf`](https://docs.rs/entelix-tokenizer-hf)
//! for HuggingFace tokenizer.json sources (Llama, Qwen, Mistral,
//! DeepSeek, Gemma, Phi, …). Future companions cover locale-aware
//! morphological accuracy for Korean and Japanese.
//! [`ByteCountTokenCounter`] ships in core as a zero-dependency
//! conservative default — accurate enough for development scaffolding,
//! never for production budgeting on non-English content.

use std::sync::Arc;

use crate::ir::Message;

/// Counts tokens for budget enforcement, splitter sizing, and
/// content-economy estimation.
///
/// Synchronous by contract — counters that need IO (remote tokenizer
/// service, lazy file-backed model) should pre-load eagerly at
/// construction or wrap the slow path in
/// `tokio::task::spawn_blocking` at the *call* site rather than
/// hiding `.await` inside the trait. Mirrors the
/// [`crate::time::Clock`] discipline: low-level primitives stay
/// pure so they compose with locks and hot paths.
pub trait TokenCounter: Send + Sync + std::fmt::Debug {
    /// Count the tokens in `text` under this counter's encoding.
    fn count(&self, text: &str) -> u64;

    /// Sum the token count across every text-bearing content part
    /// of a message slice. The default impl walks
    /// [`crate::ir::ContentPart::Text`] parts; non-text parts (image,
    /// tool-use, tool-result blocks) are vendor-specific in their
    /// token cost — counters that need an exact tally for those
    /// shapes override this method.
    fn count_messages(&self, msgs: &[Message]) -> u64 {
        msgs.iter()
            .flat_map(|m| m.content.iter())
            .filter_map(|part| match part {
                crate::ir::ContentPart::Text { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .map(|t| self.count(t))
            .sum()
    }

    /// Vendor-published encoding name (`"cl100k_base"`,
    /// `"o200k_base"`, `"claude"`, `"gemini-tokenizer"`, …) — surfaced
    /// on OTel `gen_ai.tokenizer.name` and operator diagnostics.
    fn encoding_name(&self) -> &'static str;
}

impl<T: TokenCounter + ?Sized> TokenCounter for Arc<T> {
    fn count(&self, text: &str) -> u64 {
        (**self).count(text)
    }
    fn count_messages(&self, msgs: &[Message]) -> u64 {
        (**self).count_messages(msgs)
    }
    fn encoding_name(&self) -> &'static str {
        (**self).encoding_name()
    }
}

/// Zero-dependency conservative counter — `bytes.div_ceil(4)`.
///
/// Approximates English at the ~4-bytes-per-token rule of thumb that
/// tiktoken's `cl100k_base` is built around. **Systematically
/// inaccurate** for CJK, Devanagari, Arabic, and other scripts whose
/// UTF-8 byte cost diverges from typical token boundaries — operators
/// shipping multilingual workloads inject a vendor-accurate counter
/// (`entelix-tokenizer-tiktoken`, `entelix-tokenizer-hf`, locale-aware
/// companions) at `ChatModel::with_token_counter(...)` time.
///
/// The bias direction is deliberate: `div_ceil` rounds up, so the
/// estimate skews *over* the real count on average. Pre-flight
/// `RunBudget` checks built on top remain conservative — a
/// near-budget call is more likely refused than admitted, which is
/// the correct error direction for budget enforcement.
#[derive(Clone, Copy, Debug, Default)]
pub struct ByteCountTokenCounter;

impl ByteCountTokenCounter {
    /// Construct the counter. Stateless — every call to [`Self::new`]
    /// returns the same logical instance.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl TokenCounter for ByteCountTokenCounter {
    fn count(&self, text: &str) -> u64 {
        // `usize::div_ceil` only stabilised in 1.73 — the workspace
        // pins 1.95 so the direct call is fine. `u64::from` over an
        // `as` cast keeps the lossless-conversion lint happy.
        u64::from(u32::try_from(text.len().div_ceil(4)).unwrap_or(u32::MAX))
    }

    fn encoding_name(&self) -> &'static str {
        "byte-count-naive"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ContentPart, Role};

    #[test]
    fn byte_count_rounds_up() {
        let c = ByteCountTokenCounter::new();
        assert_eq!(c.count(""), 0, "empty string is zero");
        assert_eq!(c.count("a"), 1, "one byte rounds up to one token");
        assert_eq!(c.count("abcd"), 1, "exactly four bytes is one token");
        assert_eq!(c.count("abcde"), 2, "five bytes rounds up to two");
        assert_eq!(c.count("abcdefgh"), 2, "exactly eight bytes is two");
    }

    #[test]
    fn byte_count_handles_multibyte_utf8_at_byte_granularity() {
        // Korean "안녕" — 2 chars, 6 UTF-8 bytes → 2 tokens by the
        // div_ceil(4) heuristic. Real cl100k_base would tokenise
        // these as 2-3 tokens; the naive counter is documented as
        // approximate, not exact.
        let c = ByteCountTokenCounter::new();
        assert_eq!(c.count("안녕"), 2);
    }

    #[test]
    fn count_messages_sums_text_parts_only() {
        let counter = ByteCountTokenCounter::new();
        let msg = Message::new(
            Role::User,
            vec![
                ContentPart::text("hello world!"), // 12 bytes → 3 tokens
                ContentPart::text("xyz"),          // 3 bytes  → 1 token
            ],
        );
        assert_eq!(counter.count_messages(std::slice::from_ref(&msg)), 4);
    }

    #[test]
    fn count_messages_default_impl_skips_non_text_parts() {
        // Tool-use blocks etc. carry vendor-specific tokens; the
        // default counter walks Text parts only. A counter that
        // wants exact counting for tool-use shapes overrides
        // count_messages.
        let counter = ByteCountTokenCounter::new();
        let msg = Message::new(
            Role::Assistant,
            vec![
                ContentPart::text("hi"), // 2 bytes → 1 token
                ContentPart::ToolUse {
                    id: "call_1".into(),
                    name: "tool".into(),
                    input: serde_json::json!({}),
                    provider_echoes: Vec::new(),
                },
            ],
        );
        assert_eq!(counter.count_messages(std::slice::from_ref(&msg)), 1);
    }

    #[test]
    fn encoding_name_surfaces_for_otel_attribute() {
        assert_eq!(
            ByteCountTokenCounter::new().encoding_name(),
            "byte-count-naive"
        );
    }

    #[test]
    fn arc_blanket_impl_forwards() {
        let c: Arc<dyn TokenCounter> = Arc::new(ByteCountTokenCounter::new());
        assert_eq!(c.count("abcd"), 1);
        assert_eq!(c.encoding_name(), "byte-count-naive");
    }
}
