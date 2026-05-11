//! Default [`TokenCounterRegistry`] population.
//!
//! Operators with a single tokenizer dependency typically construct
//! their counter at boot and pass it to one [`ChatModel`]. Operators
//! gatewaying multiple model handles (one process fronting several
//! `(provider, model)` pairs — production routing layer, ingestion
//! pipeline serving many tenants) instead share a single
//! [`TokenCounterRegistry`] across every dispatch.
//!
//! This module ships [`default_token_counter_registry`] — a
//! pre-populated registry that maps every known `OpenAI` BPE family
//! (gated on the `tokenizer-tiktoken` feature) to the matching
//! [`TiktokenCounter`] encoding. `HuggingFace` tokenizers ship as
//! bytes the operator supplies at build time (no embedded
//! `tokenizer.json` in the SDK by design — invariant 9 + no silent
//! network IO), so `HuggingFace` entries are operator-added through
//! [`TokenCounterRegistry::register`] after the call.
//!
//! Anthropic models route to the registry's fallback
//! ([`ByteCountTokenCounter`] unless replaced) — Anthropic does not
//! publish a public BPE, and `ByteCountTokenCounter`'s deliberate
//! over-count bias keeps pre-flight [`RunBudget`] checks fail-closed.
//!
//! [`ChatModel`]: entelix_core::ChatModel
//! [`ByteCountTokenCounter`]: entelix_core::ByteCountTokenCounter
//! [`TiktokenCounter`]: entelix_tokenizer_tiktoken::TiktokenCounter
//! [`RunBudget`]: entelix_core::RunBudget

#[cfg(feature = "tokenizer-tiktoken")]
use std::sync::Arc;

#[cfg(feature = "tokenizer-tiktoken")]
use entelix_core::Error;
use entelix_core::{Result, TokenCounterRegistry};

/// Build a [`TokenCounterRegistry`] pre-populated with every counter
/// the enabled tokenizer features can supply.
///
/// Current mappings:
///
/// - **`OpenAI`** (`tokenizer-tiktoken` feature):
///   - `gpt-5`, `gpt-4o`, `chatgpt-4o`, `gpt-4.1`, `gpt-4.5`,
///     `o1`, `o3`, `o4` → `o200k_base`
///   - `gpt-4`, `gpt-3.5`, `text-embedding-3` → `cl100k_base`
///
/// `gpt-4.1` and `gpt-4.5` carry explicit `o200k_base` entries with
/// dotted prefixes — longest-prefix matching ensures they win over
/// the `gpt-4` → `cl100k_base` catchall (`OpenAI` re-routed those
/// families to `o200k_base` on release, and prefix-matching against
/// `gpt-4` alone would silently miscount).
///
/// Operators register their own `HuggingFace` counters (Gemini, Llama,
/// Qwen, Mistral, …) by calling [`TokenCounterRegistry::register`]
/// on the returned value with bytes loaded in application code.
///
/// Returns [`Error::Config`] on the (vanishingly rare) failure to
/// load embedded `tiktoken-rs` BPE tables — the failure mode is an
/// upstream packaging bug, not a runtime concern.
pub fn default_token_counter_registry() -> Result<TokenCounterRegistry> {
    let registry = TokenCounterRegistry::new();
    #[cfg(feature = "tokenizer-tiktoken")]
    let registry = {
        use entelix_tokenizer_tiktoken::{TiktokenCounter, TiktokenEncoding};

        let o200k: Arc<dyn entelix_core::TokenCounter> = Arc::new(
            TiktokenCounter::for_encoding(TiktokenEncoding::O200kBase)
                .map_err(|e| Error::config(format!("default registry: load o200k_base: {e}")))?,
        );
        let cl100k: Arc<dyn entelix_core::TokenCounter> = Arc::new(
            TiktokenCounter::for_encoding(TiktokenEncoding::Cl100kBase)
                .map_err(|e| Error::config(format!("default registry: load cl100k_base: {e}")))?,
        );

        registry
            // o200k_base family — newest OpenAI tokenizer. `gpt-4.1`
            // and `gpt-4.5` need explicit entries because the
            // `gpt-4` cl100k catchall would otherwise prefix-match
            // them silently.
            .register("openai", "gpt-5", Arc::clone(&o200k))
            .register("openai", "gpt-4o", Arc::clone(&o200k))
            .register("openai", "chatgpt-4o", Arc::clone(&o200k))
            .register("openai", "gpt-4.1", Arc::clone(&o200k))
            .register("openai", "gpt-4.5", Arc::clone(&o200k))
            .register("openai", "o1", Arc::clone(&o200k))
            .register("openai", "o3", Arc::clone(&o200k))
            .register("openai", "o4", Arc::clone(&o200k))
            // cl100k_base family — GPT-4 / GPT-3.5 / text-embedding-3.
            .register("openai", "gpt-4", Arc::clone(&cl100k))
            .register("openai", "gpt-3.5", Arc::clone(&cl100k))
            .register("openai", "text-embedding-3", cl100k)
    };
    Ok(registry)
}

#[cfg(test)]
#[cfg(feature = "tokenizer-tiktoken")]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn default_registry_routes_openai_models() {
        let reg = default_token_counter_registry().expect("load embedded tables");

        // o200k_base — matched entries.
        for (model, expected) in [
            ("gpt-5", "o200k_base"),
            ("gpt-4o-mini", "o200k_base"),
            ("chatgpt-4o-latest", "o200k_base"),
            ("o3-mini", "o200k_base"),
            ("o4-mini", "o200k_base"),
            // gpt-4.1 / gpt-4.5 route to o200k via explicit entries —
            // a regression that drops them would silently misroute to
            // cl100k_base via the `gpt-4` catchall.
            ("gpt-4.1", "o200k_base"),
            ("gpt-4.1-mini", "o200k_base"),
            ("gpt-4.5-preview", "o200k_base"),
        ] {
            let r = reg.resolve("openai", model);
            assert!(r.is_match(), "{model} should be a registered entry");
            assert_eq!(r.counter().encoding_name(), expected, "{model}");
        }

        // cl100k_base — gpt-4 catchall + gpt-3.5 + text-embedding-3.
        for (model, expected) in [
            ("gpt-4-turbo", "cl100k_base"),
            ("gpt-3.5-turbo", "cl100k_base"),
            ("text-embedding-3-small", "cl100k_base"),
        ] {
            let r = reg.resolve("openai", model);
            assert!(r.is_match(), "{model} should be a registered entry");
            assert_eq!(r.counter().encoding_name(), expected, "{model}");
        }

        // Unknown model — fallback surfaces explicitly.
        let unknown = reg.resolve("openai", "davinci-002");
        assert!(
            unknown.is_fallback(),
            "unknown model must surface as fallback"
        );
        assert_eq!(unknown.counter().encoding_name(), "byte-count-naive");
    }

    #[test]
    fn default_registry_routes_anthropic_through_fallback() {
        let reg = default_token_counter_registry().expect("load embedded tables");
        // No Anthropic entry — ByteCount fallback (conservative).
        let r = reg.resolve("anthropic", "claude-sonnet-4-5");
        assert!(r.is_fallback());
        assert_eq!(r.counter().encoding_name(), "byte-count-naive");
    }
}
