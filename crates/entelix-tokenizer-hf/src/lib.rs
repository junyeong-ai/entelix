//! # entelix-tokenizer-hf
//!
//! Vendor-accurate [`TokenCounter`] wrapping the
//! [HuggingFace `tokenizers`](https://crates.io/crates/tokenizers)
//! crate. Construct from any `tokenizer.json` byte payload — the
//! single canonical entry point for Llama 3, Qwen 2.5, Mistral,
//! DeepSeek, Gemma, Phi, and every other open-weight model whose
//! tokenizer is published in the HF format.
//!
//! ## Why bytes-only construction
//!
//! [`HfTokenCounter::from_bytes`] is the only constructor — there is
//! no `from_file` or `from_pretrained` shortcut. Two reasons:
//!
//! - **Invariant 9 alignment** — entelix first-party crates do not
//!   import `std::fs`. Operators read tokenizer files in their own
//!   application code (or at compile time via `include_bytes!`) and
//!   pass the byte payload in.
//! - **No silent network IO** —
//!   `tokenizers::Tokenizer::from_pretrained` does HTTP downloads
//!   and disk caching as a side effect. Wrappers that need hub
//!   integration ship as separate companion crates; this crate stays
//!   pure.
//!
//! ## Encoding name
//!
//! [`TokenCounter::encoding_name`] returns `&'static str`, but the
//! HF tokenizer format does not embed a canonical name. Operators
//! supply a name at construction; the wrapper leaks it once into a
//! `&'static str` so the trait method can return it directly. One
//! allocation per [`HfTokenCounter::from_bytes`] call — the canonical
//! "construct once at app boot, share an `Arc` everywhere"
//! pattern keeps the leak cost a single `String` per process.
//!
//! ## Encode-failure semantics
//!
//! `tokenizers::Tokenizer::encode` is fallible — a misconfigured
//! tokenizer JSON or a post-processor that rejects the input
//! surfaces as `Err`. [`TokenCounter::count`] returns `u64::MAX` on
//! such failures so `RunBudget` pre-flight checks fail closed
//! (refuses the call rather than silently under-counting).
//! `tracing::warn!` records the underlying error.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(html_root_url = "https://docs.rs/entelix-tokenizer-hf/0.5.2")]
#![deny(missing_docs)]
#![allow(
    // Vendor-name proper nouns (`HuggingFace`, `Llama`, `Qwen`,
    // `OpenAI`, `BPE`) appear throughout the docs; backtick-quoting
    // every occurrence hurts readability without adding signal.
    clippy::doc_markdown
)]

use std::fmt;
use std::sync::Arc;

use entelix_core::TokenCounter;
use thiserror::Error;
use tokenizers::Tokenizer;

/// Errors raised when constructing an [`HfTokenCounter`].
///
/// The underlying `tokenizers` crate error chain is stripped to a
/// `String` so the variant stays `Send + Sync + 'static` for
/// ergonomic cross-thread propagation (operators map this onto
/// `entelix_core::Error::config`). Variant shape mirrors
/// `TiktokenError` for cross-companion consistency.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum HfTokenizerError {
    /// Loading the tokenizer JSON failed — invalid format,
    /// unsupported tokenizer type, or schema-version mismatch.
    #[error("HuggingFace tokenizer load failed for {encoding_name}: {message}")]
    Load {
        /// Operator-supplied encoding name the load was attempted
        /// for. Captured here so the error trail names which
        /// counter construction failed when an app boot wires
        /// multiple HF tokenizers.
        encoding_name: String,
        /// Upstream `tokenizers` error message (chain stripped).
        message: String,
    },
}

/// [`TokenCounter`] backed by a HuggingFace [`Tokenizer`].
///
/// Cloning is cheap — the tokenizer sits behind an [`Arc`] so every
/// clone shares one parsed instance. Construct once at app boot,
/// share across `ChatModelConfig` and ingestion pipelines.
#[derive(Clone)]
pub struct HfTokenCounter {
    tokenizer: Arc<Tokenizer>,
    encoding_name: &'static str,
}

impl fmt::Debug for HfTokenCounter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HfTokenCounter")
            .field("encoding_name", &self.encoding_name)
            .finish_non_exhaustive()
    }
}

impl HfTokenCounter {
    /// Construct a counter from a `tokenizer.json` byte payload.
    /// The supplied `encoding_name` surfaces on
    /// [`TokenCounter::encoding_name`] and the OTel
    /// `gen_ai.tokenizer.name` attribute — pick a stable identifier
    /// for the model whose tokenizer the bytes encode (`"llama-3"`,
    /// `"qwen-2.5"`, `"mistral"`, …).
    ///
    /// The encoding name is leaked once into a `&'static str` —
    /// see the [crate-level docs](crate#encoding-name) for the
    /// rationale.
    pub fn from_bytes(
        bytes: &[u8],
        encoding_name: impl Into<String>,
    ) -> Result<Self, HfTokenizerError> {
        let encoding_name = encoding_name.into();
        let tokenizer = Tokenizer::from_bytes(bytes).map_err(|e| HfTokenizerError::Load {
            encoding_name: encoding_name.clone(),
            message: e.to_string(),
        })?;
        let encoding_name: &'static str = Box::leak(encoding_name.into_boxed_str());
        Ok(Self {
            tokenizer: Arc::new(tokenizer),
            encoding_name,
        })
    }

    /// Inspect the configured encoding name.
    #[must_use]
    pub const fn encoding(&self) -> &'static str {
        self.encoding_name
    }
}

impl TokenCounter for HfTokenCounter {
    fn count(&self, text: &str) -> u64 {
        match self.tokenizer.encode(text, false) {
            Ok(encoding) => u64::try_from(encoding.len()).unwrap_or(u64::MAX),
            Err(error) => {
                tracing::warn!(
                    tokenizer = %self.encoding_name,
                    error = %error,
                    "HfTokenCounter::count encode failed; returning u64::MAX (conservative)",
                );
                u64::MAX
            }
        }
    }

    fn encoding_name(&self) -> &'static str {
        self.encoding_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use entelix_core::ir::{ContentPart, Message, Role};

    /// Hand-crafted minimal HuggingFace `tokenizer.json` — a
    /// `WordLevel` model with a 4-word vocab plus `[UNK]` and the
    /// `Whitespace` pre-tokenizer. Mirrors the canonical HF schema
    /// so the wrapper's `from_bytes` path is exercised against the
    /// exact byte shape an operator would `include_bytes!` from a
    /// downloaded model.
    const TINY_TOKENIZER_JSON: &str = r#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": { "type": "Whitespace" },
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": {
                "[UNK]": 0,
                "hello": 1,
                "world": 2,
                "foo": 3,
                "bar": 4
            },
            "unk_token": "[UNK]"
        }
    }"#;

    type TestResult = Result<(), HfTokenizerError>;

    fn counter() -> Result<HfTokenCounter, HfTokenizerError> {
        HfTokenCounter::from_bytes(TINY_TOKENIZER_JSON.as_bytes(), "tiny-wordlevel")
    }

    #[test]
    fn from_bytes_accepts_valid_tokenizer_json() -> TestResult {
        let counter = counter()?;
        assert_eq!(counter.encoding(), "tiny-wordlevel");
        assert_eq!(counter.encoding_name(), "tiny-wordlevel");
        Ok(())
    }

    #[test]
    fn from_bytes_rejects_garbage_input() {
        let result = HfTokenCounter::from_bytes(b"this is not json", "any");
        assert!(matches!(result, Err(HfTokenizerError::Load { .. })));
    }

    #[test]
    fn from_bytes_rejects_empty_input() {
        let result = HfTokenCounter::from_bytes(b"", "any");
        assert!(matches!(result, Err(HfTokenizerError::Load { .. })));
    }

    #[test]
    fn load_error_captures_encoding_name() {
        let result = HfTokenCounter::from_bytes(b"garbage", "my-bad-tokenizer");
        match result {
            Err(HfTokenizerError::Load {
                encoding_name,
                message,
            }) => {
                assert_eq!(encoding_name, "my-bad-tokenizer");
                assert!(!message.is_empty(), "upstream message must propagate");
            }
            other => panic!("expected Load error, got {other:?}"),
        }
    }

    #[test]
    fn count_known_inputs_match_vocab_size() -> TestResult {
        let counter = counter()?;
        assert_eq!(counter.count(""), 0);
        assert_eq!(counter.count("hello"), 1);
        assert_eq!(counter.count("hello world"), 2);
        assert_eq!(counter.count("hello world foo bar"), 4);
        Ok(())
    }

    #[test]
    fn unknown_words_count_as_unk_tokens() -> TestResult {
        // The vocab has [UNK] → 0; every out-of-vocab whitespace-
        // separated word becomes one [UNK] token.
        let counter = counter()?;
        assert_eq!(counter.count("xyz abc"), 2);
        assert_eq!(counter.count("hello xyz world abc"), 4);
        Ok(())
    }

    #[test]
    fn count_messages_default_walks_text_parts() -> TestResult {
        let counter = counter()?;
        let msg = Message::new(
            Role::User,
            vec![
                ContentPart::text("hello world"), // 2 tokens
                ContentPart::text("foo bar"),     // 2 tokens
            ],
        );
        assert_eq!(counter.count_messages(std::slice::from_ref(&msg)), 4);
        Ok(())
    }

    #[test]
    fn count_messages_skips_non_text_parts() -> TestResult {
        let counter = counter()?;
        let msg = Message::new(
            Role::Assistant,
            vec![
                ContentPart::text("hello world"), // 2 tokens
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
        let counter: Arc<dyn TokenCounter> = Arc::new(counter()?);
        assert_eq!(counter.count("hello world"), 2);
        assert_eq!(counter.encoding_name(), "tiny-wordlevel");
        Ok(())
    }

    #[test]
    fn clone_shares_tokenizer_and_keeps_encoding_name() -> TestResult {
        let original = counter()?;
        let cloned = original.clone();
        assert_eq!(cloned.encoding(), "tiny-wordlevel");
        assert_eq!(cloned.count("hello"), original.count("hello"));
        assert!(Arc::ptr_eq(&original.tokenizer, &cloned.tokenizer));
        Ok(())
    }

    #[test]
    fn debug_includes_encoding_not_tokenizer_table() -> TestResult {
        let counter = counter()?;
        let debug = format!("{counter:?}");
        assert!(debug.contains("tiny-wordlevel"));
        assert!(
            !debug.contains("Tokenizer ") && !debug.contains("vocab"),
            "Debug must not dump the parsed tokenizer: {debug}"
        );
        Ok(())
    }

    #[test]
    fn encoding_name_outlives_counter_drop() -> TestResult {
        // Box::leak guarantees `&'static` lifetime — the name stays
        // valid even after the counter that produced it is dropped.
        let leaked: &'static str = {
            let counter = HfTokenCounter::from_bytes(TINY_TOKENIZER_JSON.as_bytes(), "scoped")?;
            counter.encoding_name()
        };
        assert_eq!(leaked, "scoped");
        Ok(())
    }
}
