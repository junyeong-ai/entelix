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

/// Routing table from `(provider, model)` pairs to a vendor-accurate
/// [`TokenCounter`].
///
/// Operators registering one counter per `(provider, model_prefix)`
/// pair drive the [`Self::resolve`] lookup; gateways (one process
/// fronting many model handles) read every chat-side dispatch
/// through one shared registry. The facade ships
/// `default_token_counter_registry()` factory (feature-gated on
/// `tokenizer-tiktoken` / `tokenizer-hf`) with the OpenAI BPE
/// pre-populated; operators extend it with their HuggingFace
/// `tokenizer.json` bytes.
///
/// ## Matching algorithm
///
/// Entries match when the provider name is **exactly equal** and the
/// model name **starts with** the registered prefix. Among all
/// matching entries, the one with the longest prefix wins — so
/// registering both `"gpt-4"` and `"gpt-4o"` routes
/// `"gpt-4o-mini"` to the `"gpt-4o"` entry without depending on
/// registration order. Ties on prefix length resolve to the
/// last-registered entry (operator-overridable). Misses fall through
/// to the registry's fallback counter.
///
/// ## Why prefix matching
///
/// Vendors version models with stable family prefixes (`gpt-4o-*`,
/// `claude-sonnet-*`, `gemini-1.5-*`). Exact-name matching forces
/// the operator to update the registry on every minor model release;
/// prefix matching absorbs new patch revisions silently into the
/// same tokenizer mapping the family uses. Regex was considered and
/// rejected — too expressive, and the typical mistake is *missing* a
/// model, which prefix matching handles by falling through to the
/// fallback rather than misrouting silently.
pub struct TokenCounterRegistry {
    entries: Vec<RegistryEntry>,
    fallback: Arc<dyn TokenCounter>,
}

struct RegistryEntry {
    provider: &'static str,
    model_prefix: &'static str,
    counter: Arc<dyn TokenCounter>,
}

impl TokenCounterRegistry {
    /// Construct an empty registry with [`ByteCountTokenCounter`] as
    /// the fallback. Add entries with [`Self::register`]; replace
    /// the fallback with [`Self::with_default`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            fallback: Arc::new(ByteCountTokenCounter::new()),
        }
    }

    /// Replace the fallback counter used when no entry matches.
    /// Default is [`ByteCountTokenCounter`] (conservative — biased
    /// to over-count so pre-flight budget checks fail closed).
    #[must_use]
    pub fn with_default(mut self, fallback: Arc<dyn TokenCounter>) -> Self {
        self.fallback = fallback;
        self
    }

    /// Append a `(provider, model_prefix) → counter` entry. Multiple
    /// entries for the same provider partition the model space by
    /// prefix; the longest-prefix match wins at lookup time.
    /// Ties on prefix length resolve to the last-registered entry,
    /// so a later `register` call overrides an earlier one for the
    /// same `(provider, model_prefix)` pair.
    #[must_use]
    pub fn register(
        mut self,
        provider: &'static str,
        model_prefix: &'static str,
        counter: Arc<dyn TokenCounter>,
    ) -> Self {
        self.entries.push(RegistryEntry {
            provider,
            model_prefix,
            counter,
        });
        self
    }

    /// Resolve `(provider, model)` to a counter. The returned
    /// [`Resolution`] makes match vs. fallback visible at the type
    /// level so callers branch on the miss without inspecting
    /// `encoding_name()` after the fact (invariant 15 — silent
    /// fallback is not an option). See [`Resolution`] for the
    /// pattern-match shape and the deliberate "accept fallback"
    /// idiom.
    #[must_use]
    pub fn resolve(&self, provider: &str, model: &str) -> Resolution {
        let mut best: Option<&RegistryEntry> = None;
        for entry in &self.entries {
            if entry.provider != provider {
                continue;
            }
            if !model.starts_with(entry.model_prefix) {
                continue;
            }
            match best {
                Some(prev) if prev.model_prefix.len() > entry.model_prefix.len() => {}
                _ => best = Some(entry),
            }
        }
        match best {
            Some(entry) => Resolution::Matched(Arc::clone(&entry.counter)),
            None => Resolution::Fallback(Arc::clone(&self.fallback)),
        }
    }

    /// Number of registered `(provider, model_prefix)` entries.
    /// Excludes the fallback. Operators wiring a `tracing::info!` on
    /// boot read this to confirm the table is the expected size.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the registry has no registered entries (the fallback
    /// is always present).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for TokenCounterRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Outcome of [`TokenCounterRegistry::resolve`] — surfaces whether a
/// registered entry actually matched `(provider, model)` or the
/// registry's fallback counter was returned because no entry matched.
///
/// Making fallback visible at the type level lets production cost
/// estimation paths log / alert on unknown models without inspecting
/// `encoding_name()` after the fact (invariant 15 — silent fallback is
/// not an option). Callers consume the resolution via pattern match —
/// there is no "give me the counter regardless" shortcut, because
/// such a shortcut would re-enable silent absorption of the fallback.
///
/// ```ignore
/// let counter = match registry.resolve("openai", "gpt-9") {
///     Resolution::Matched(c) => c,
///     Resolution::Fallback(c) => {
///         tracing::warn!("unknown model — using fallback counter");
///         c
///     }
/// };
/// ```
///
/// Operators that intentionally accept the fallback bind both arms to
/// the same name (`Resolution::Matched(c) | Resolution::Fallback(c) => c`).
/// The pattern is short, and the call site signals the deliberate
/// choice in source.
///
/// ## Future variants
///
/// `Resolution` is `#[non_exhaustive]`. The two predicates
/// [`Self::is_match`] and [`Self::is_fallback`] each return `true`
/// only for their own variant — `!is_match()` and `is_fallback()`
/// are **not** equivalent under future variant additions. Operators
/// that want to alert on the no-match branch should match on
/// `Self::Fallback` explicitly.
#[derive(Clone)]
#[non_exhaustive]
pub enum Resolution {
    /// A registered `(provider, model_prefix)` entry matched.
    Matched(Arc<dyn TokenCounter>),
    /// No entry matched; the registry's fallback counter is returned.
    Fallback(Arc<dyn TokenCounter>),
}

impl Resolution {
    /// Borrow the resolved counter regardless of which branch matched.
    /// Operators that need the owned `Arc` consume the enum through a
    /// pattern match.
    #[must_use]
    pub fn counter(&self) -> &Arc<dyn TokenCounter> {
        match self {
            Self::Matched(c) | Self::Fallback(c) => c,
        }
    }

    /// `true` when a registered entry matched.
    #[must_use]
    pub const fn is_match(&self) -> bool {
        matches!(self, Self::Matched(_))
    }

    /// `true` when no entry matched and the fallback was returned.
    /// Distinct from `!is_match()` — `Resolution` is `non_exhaustive`,
    /// so a future variant could make both predicates return `false`.
    #[must_use]
    pub const fn is_fallback(&self) -> bool {
        matches!(self, Self::Fallback(_))
    }
}

impl std::fmt::Debug for Resolution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (kind, encoding) = match self {
            Self::Matched(c) => ("Matched", c.encoding_name()),
            Self::Fallback(c) => ("Fallback", c.encoding_name()),
        };
        f.debug_struct(kind).field("encoding", &encoding).finish()
    }
}

impl std::fmt::Debug for TokenCounterRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let entries: Vec<(&'static str, &'static str, &'static str)> = self
            .entries
            .iter()
            .map(|e| (e.provider, e.model_prefix, e.counter.encoding_name()))
            .collect();
        f.debug_struct("TokenCounterRegistry")
            .field("entries", &entries)
            .field("fallback", &self.fallback.encoding_name())
            .finish()
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

    #[derive(Debug)]
    struct LabelledCounter(&'static str, u64);
    impl TokenCounter for LabelledCounter {
        fn count(&self, _text: &str) -> u64 {
            self.1
        }
        fn encoding_name(&self) -> &'static str {
            self.0
        }
    }

    fn labelled(name: &'static str, fixed: u64) -> Arc<dyn TokenCounter> {
        Arc::new(LabelledCounter(name, fixed))
    }

    #[test]
    fn registry_returns_fallback_when_empty() {
        let reg = TokenCounterRegistry::new();
        let resolution = reg.resolve("openai", "gpt-5");
        assert!(
            resolution.is_fallback(),
            "empty registry should fall through"
        );
        assert_eq!(resolution.counter().encoding_name(), "byte-count-naive");
    }

    #[test]
    fn registry_resolves_exact_provider_and_prefix() {
        let reg = TokenCounterRegistry::new().register("openai", "gpt-4o", labelled("o200k", 1));
        let resolution = reg.resolve("openai", "gpt-4o-mini");
        assert!(resolution.is_match(), "registered prefix should match");
        assert_eq!(resolution.counter().encoding_name(), "o200k");
    }

    #[test]
    fn registry_ignores_wrong_provider() {
        let reg =
            TokenCounterRegistry::new().register("anthropic", "claude", labelled("anthropic", 2));
        let resolution = reg.resolve("openai", "claude-clone");
        // Provider mismatch — fall through to fallback.
        assert!(resolution.is_fallback());
        assert_eq!(resolution.counter().encoding_name(), "byte-count-naive");
    }

    #[test]
    fn registry_longest_prefix_wins_regardless_of_registration_order() {
        // Register "gpt-4" first, then "gpt-4o" — longest-prefix wins
        // on "gpt-4o-mini" regardless of order.
        let reg = TokenCounterRegistry::new()
            .register("openai", "gpt-4", labelled("cl100k", 1))
            .register("openai", "gpt-4o", labelled("o200k", 1));
        assert_eq!(
            reg.resolve("openai", "gpt-4o-mini")
                .counter()
                .encoding_name(),
            "o200k"
        );

        // Reverse registration order — same outcome.
        let reg = TokenCounterRegistry::new()
            .register("openai", "gpt-4o", labelled("o200k", 1))
            .register("openai", "gpt-4", labelled("cl100k", 1));
        assert_eq!(
            reg.resolve("openai", "gpt-4o-mini")
                .counter()
                .encoding_name(),
            "o200k"
        );
    }

    #[test]
    fn registry_falls_through_to_fallback_on_non_matching_model() {
        let reg = TokenCounterRegistry::new().register("openai", "gpt-4o", labelled("o200k", 1));
        // Same provider, prefix doesn't match — fall through.
        let resolution = reg.resolve("openai", "davinci");
        assert!(resolution.is_fallback());
        assert_eq!(resolution.counter().encoding_name(), "byte-count-naive");
    }

    #[test]
    fn registry_last_wins_on_tie() {
        let reg = TokenCounterRegistry::new()
            .register("openai", "gpt-4", labelled("first", 1))
            .register("openai", "gpt-4", labelled("second", 1));
        assert_eq!(
            reg.resolve("openai", "gpt-4-turbo")
                .counter()
                .encoding_name(),
            "second"
        );
    }

    #[test]
    fn registry_with_default_replaces_fallback() {
        let reg = TokenCounterRegistry::new().with_default(labelled("custom-fb", 0));
        let resolution = reg.resolve("any", "x");
        assert!(resolution.is_fallback());
        assert_eq!(resolution.counter().encoding_name(), "custom-fb");
    }

    #[test]
    fn registry_len_excludes_fallback() {
        let reg = TokenCounterRegistry::new()
            .register("openai", "gpt-4", labelled("a", 1))
            .register("openai", "gpt-4o", labelled("b", 1));
        assert_eq!(reg.len(), 2);
        assert!(!reg.is_empty());
    }

    #[test]
    fn resolution_pattern_match_yields_counter() {
        let reg = TokenCounterRegistry::new().register("openai", "gpt-4o", labelled("o200k", 1));
        let counter = match reg.resolve("openai", "gpt-4o") {
            Resolution::Matched(c) | Resolution::Fallback(c) => c,
        };
        assert_eq!(counter.encoding_name(), "o200k");
    }

    #[test]
    fn resolution_match_and_fallback_are_distinguishable() {
        let reg = TokenCounterRegistry::new().register("openai", "gpt-4o", labelled("o200k", 1));
        let matched = reg.resolve("openai", "gpt-4o-mini");
        let fallback = reg.resolve("openai", "davinci-002");
        assert!(matched.is_match() && !matched.is_fallback());
        assert!(fallback.is_fallback() && !fallback.is_match());
    }
}
