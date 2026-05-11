//! # entelix-tokenizer-tiktoken
//!
//! Vendor-accurate [`TokenCounter`] for OpenAI's BPE tokenizer family —
//! `cl100k_base`, `o200k_base`, `p50k_base`, `r50k_base`. Wraps
//! [`tiktoken-rs`](https://crates.io/crates/tiktoken-rs) with eager BPE
//! preload at construction so the per-call `count` stays synchronous
//! per the [`TokenCounter`] contract.
//!
//! ## Encoding to model mapping
//!
//! - [`TiktokenEncoding::Cl100kBase`] — GPT-3.5-turbo, GPT-4,
//!   GPT-4-turbo, text-embedding-3-*.
//! - [`TiktokenEncoding::O200kBase`] — GPT-4o, GPT-4o-mini, o1, o3,
//!   o3-mini, o4.
//! - [`TiktokenEncoding::P50kBase`] — GPT-3 davinci, codex.
//! - [`TiktokenEncoding::R50kBase`] — GPT-3 ada / babbage / curie,
//!   GPT-2.
//!
//! The mapping is left to operators by design — OpenAI changes it over
//! time, and accidentally pinning a stale mapping silently miscounts
//! without surfacing a build failure. Pick the encoding for your
//! target model and the wrapper preloads the matching BPE tables.
//!
//! ## Why eager preload
//!
//! The [`TokenCounter`] trait is intentionally synchronous — counters
//! get called from inside hot dispatch paths (pre-flight `RunBudget`
//! checks, splitter sizing) where awaiting on a lazy table-load
//! introduces unbounded latency. `TiktokenCounter` therefore loads the
//! BPE tables eagerly inside [`TiktokenCounter::for_encoding`] and
//! caches them behind an [`Arc`]. Cloning a `TiktokenCounter` is
//! cheap; loading a fresh one re-parses the embedded tables so prefer
//! `clone` for fan-out.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(html_root_url = "https://docs.rs/entelix-tokenizer-tiktoken/0.4.2")]
#![deny(missing_docs)]
#![allow(
    // Vendor-name proper nouns (`OpenAI`, `OTel`, `BPE`, `GPT-4o`)
    // appear throughout the docs; backtick-quoting every occurrence
    // hurts readability without adding signal.
    clippy::doc_markdown
)]

use std::fmt;
use std::sync::Arc;

use entelix_core::TokenCounter;
use thiserror::Error;
use tiktoken_rs::CoreBPE;

/// OpenAI BPE encoding family. Pick the variant matching the target
/// model — see the [crate-level docs](crate#encoding-to-model-mapping)
/// for the model-to-encoding table.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum TiktokenEncoding {
    /// `cl100k_base` — GPT-3.5-turbo, GPT-4, GPT-4-turbo, the
    /// `text-embedding-3-*` family.
    Cl100kBase,
    /// `o200k_base` — GPT-4o, GPT-4o-mini, o1, o3, o3-mini, o4.
    O200kBase,
    /// `p50k_base` — GPT-3 davinci, codex.
    P50kBase,
    /// `r50k_base` — GPT-3 ada / babbage / curie + the original GPT-2
    /// tokenizer.
    R50kBase,
}

impl TiktokenEncoding {
    /// Canonical encoding name as published by OpenAI's tiktoken
    /// reference implementation. Surfaces on
    /// [`TokenCounter::encoding_name`] and the OTel
    /// `gen_ai.tokenizer.name` attribute.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Cl100kBase => "cl100k_base",
            Self::O200kBase => "o200k_base",
            Self::P50kBase => "p50k_base",
            Self::R50kBase => "r50k_base",
        }
    }
}

/// Errors raised when constructing a [`TiktokenCounter`].
///
/// [`tiktoken-rs`](https://crates.io/crates/tiktoken-rs) returns
/// `Box<dyn Error>` from its loader functions; this type strips the
/// upstream chain to a `String` so the error stays
/// `Send + Sync + 'static` for ergonomic cross-thread propagation
/// (downstream operators map this onto `entelix_core::Error::config`).
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TiktokenError {
    /// Loading the BPE tables for the requested encoding failed.
    /// In practice the upstream loaders only fail if the embedded
    /// merge / vocab tables fail to parse, which would indicate an
    /// upstream packaging bug. Variant shape mirrors
    /// `HfTokenizerError::Load` for cross-companion consistency.
    #[error("tiktoken BPE load failed for {encoding_name}: {message}")]
    Load {
        /// Canonical encoding name the load was attempted for
        /// (e.g. `"cl100k_base"`).
        encoding_name: &'static str,
        /// Upstream `tiktoken-rs` error message (chain stripped).
        message: String,
    },
}

/// [`TokenCounter`] impl backed by [`tiktoken-rs`](https://crates.io/crates/tiktoken-rs).
///
/// Cloning is cheap — the BPE tables sit behind an [`Arc`] so every
/// clone shares one preloaded instance. Construct once at app boot,
/// share across `ChatModelConfig` instances and ingestion pipelines.
#[derive(Clone)]
pub struct TiktokenCounter {
    bpe: Arc<CoreBPE>,
    encoding: TiktokenEncoding,
}

impl fmt::Debug for TiktokenCounter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TiktokenCounter")
            .field("encoding", &self.encoding)
            .finish_non_exhaustive()
    }
}

impl TiktokenCounter {
    /// Construct a counter for `encoding`, preloading the BPE tables
    /// eagerly. The returned counter is `Clone` and ready for hot-path
    /// dispatch.
    pub fn for_encoding(encoding: TiktokenEncoding) -> Result<Self, TiktokenError> {
        let bpe = match encoding {
            TiktokenEncoding::Cl100kBase => tiktoken_rs::cl100k_base(),
            TiktokenEncoding::O200kBase => tiktoken_rs::o200k_base(),
            TiktokenEncoding::P50kBase => tiktoken_rs::p50k_base(),
            TiktokenEncoding::R50kBase => tiktoken_rs::r50k_base(),
        }
        .map_err(|e| TiktokenError::Load {
            encoding_name: encoding.name(),
            message: e.to_string(),
        })?;
        Ok(Self {
            bpe: Arc::new(bpe),
            encoding,
        })
    }

    /// Inspect the configured encoding.
    #[must_use]
    pub const fn encoding(&self) -> TiktokenEncoding {
        self.encoding
    }
}

impl TokenCounter for TiktokenCounter {
    fn count(&self, text: &str) -> u64 {
        // `encode_ordinary` skips special-token handling — the right
        // shape for content-economy budgeting since system / chat
        // priming overhead is vendor-and-version-specific (operators
        // wanting an exact chat-message tally override
        // `count_messages` on a wrapper counter).
        u64::try_from(self.bpe.encode_ordinary(text).len()).unwrap_or(u64::MAX)
    }

    fn encoding_name(&self) -> &'static str {
        self.encoding.name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use entelix_core::ir::{ContentPart, Message, Role};

    type TestResult = Result<(), TiktokenError>;

    #[test]
    fn each_encoding_loads_successfully() -> TestResult {
        for encoding in [
            TiktokenEncoding::Cl100kBase,
            TiktokenEncoding::O200kBase,
            TiktokenEncoding::P50kBase,
            TiktokenEncoding::R50kBase,
        ] {
            let counter = TiktokenCounter::for_encoding(encoding)?;
            assert_eq!(counter.encoding(), encoding);
            assert_eq!(counter.encoding_name(), encoding.name());
        }
        Ok(())
    }

    #[test]
    fn empty_string_counts_zero() -> TestResult {
        let counter = TiktokenCounter::for_encoding(TiktokenEncoding::Cl100kBase)?;
        assert_eq!(counter.count(""), 0);
        Ok(())
    }

    #[test]
    fn cl100k_base_counts_match_known_tiktoken_values() -> TestResult {
        // Reference values verified against the upstream Python
        // `tiktoken` library (`enc.encode_ordinary(text)`):
        //   "Hello world"      → [9906, 1917]                    = 2
        //   "tiktoken is great!" → [83, 1609, 5963, 374, 2294, 0] = 6
        // Hard-pinning here is a regression gate against
        // `tiktoken-rs` upstream encoding drift.
        let counter = TiktokenCounter::for_encoding(TiktokenEncoding::Cl100kBase)?;
        assert_eq!(counter.count("Hello world"), 2);
        assert_eq!(counter.count("tiktoken is great!"), 6);
        Ok(())
    }

    #[test]
    fn o200k_base_handles_multibyte_utf8() -> TestResult {
        // CJK characters: tokenisation differs vs cl100k_base
        // because o200k_base extends the vocabulary. Just verify the
        // count is non-zero and bounded — exact value is encoding-
        // version-specific so a strict pin would brittle-test.
        let counter = TiktokenCounter::for_encoding(TiktokenEncoding::O200kBase)?;
        let count = counter.count("안녕 세계");
        assert!(count > 0, "non-empty CJK text must count above zero");
        assert!(
            count < 20,
            "five-grapheme CJK should not bloat past 20 tokens"
        );
        Ok(())
    }

    #[test]
    fn longer_text_produces_more_tokens() -> TestResult {
        let counter = TiktokenCounter::for_encoding(TiktokenEncoding::Cl100kBase)?;
        let short = counter.count("hello");
        let long = counter.count("hello world this is a longer sentence with more tokens");
        assert!(long > short, "monotonicity: longer input → more tokens");
        Ok(())
    }

    #[test]
    fn count_messages_default_walks_text_parts() -> TestResult {
        let counter = TiktokenCounter::for_encoding(TiktokenEncoding::Cl100kBase)?;
        let msg = Message::new(
            Role::User,
            vec![
                ContentPart::text("Hello world"),        // 2 tokens (verified above)
                ContentPart::text("tiktoken is great!"), // 6 tokens
            ],
        );
        assert_eq!(counter.count_messages(std::slice::from_ref(&msg)), 8);
        Ok(())
    }

    #[test]
    fn count_messages_skips_non_text_parts() -> TestResult {
        let counter = TiktokenCounter::for_encoding(TiktokenEncoding::Cl100kBase)?;
        let msg = Message::new(
            Role::Assistant,
            vec![
                ContentPart::text("Hello world"), // 2 tokens
                ContentPart::ToolUse {
                    id: "call_1".into(),
                    name: "search".into(),
                    input: serde_json::json!({"q": "rust"}),
                    provider_echoes: Vec::new(),
                },
            ],
        );
        assert_eq!(counter.count_messages(std::slice::from_ref(&msg)), 2);
        Ok(())
    }

    #[test]
    fn arc_dyn_dispatch_forwards_through_blanket_impl() -> TestResult {
        let counter: Arc<dyn TokenCounter> =
            Arc::new(TiktokenCounter::for_encoding(TiktokenEncoding::Cl100kBase)?);
        assert_eq!(counter.count("Hello world"), 2);
        assert_eq!(counter.encoding_name(), "cl100k_base");
        Ok(())
    }

    #[test]
    fn clone_shares_bpe_and_keeps_encoding() -> TestResult {
        let original = TiktokenCounter::for_encoding(TiktokenEncoding::O200kBase)?;
        let cloned = original.clone();
        assert_eq!(cloned.encoding(), TiktokenEncoding::O200kBase);
        assert_eq!(cloned.count("hello"), original.count("hello"));
        // Both clones share the same Arc — pointer equality verifies
        // clone is cheap (shared parsed BPE table).
        assert!(Arc::ptr_eq(&original.bpe, &cloned.bpe));
        Ok(())
    }

    #[test]
    fn debug_includes_encoding_not_bpe_table() -> TestResult {
        let counter = TiktokenCounter::for_encoding(TiktokenEncoding::Cl100kBase)?;
        let debug = format!("{counter:?}");
        assert!(debug.contains("Cl100kBase"));
        assert!(
            !debug.contains("CoreBPE"),
            "Debug must not dump the BPE tables: {debug}"
        );
        Ok(())
    }

    #[test]
    fn encoding_name_round_trips() {
        assert_eq!(TiktokenEncoding::Cl100kBase.name(), "cl100k_base");
        assert_eq!(TiktokenEncoding::O200kBase.name(), "o200k_base");
        assert_eq!(TiktokenEncoding::P50kBase.name(), "p50k_base");
        assert_eq!(TiktokenEncoding::R50kBase.name(), "r50k_base");
    }
}
