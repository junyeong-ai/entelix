//! `Usage` — token and safety accounting reported by the provider.

use serde::{Deserialize, Serialize};

use crate::ir::safety::SafetyRating;

/// Per-call accounting from the vendor.
///
/// Token fields are `u32` with default `0` — every shipping codec
/// populates them, and `0` is the natural "no cache hit" /
/// "no reasoning" value. Whether the field is *meaningful* for the
/// (codec, model) pair is governed by [`Capabilities`](crate::ir::Capabilities)
/// flags, not by an option/null distinction here.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Usage {
    /// Tokens consumed from the prompt this call.
    pub input_tokens: u32,
    /// Tokens produced as output (assistant content).
    pub output_tokens: u32,
    /// Tokens served from the prompt cache (typically discounted).
    #[serde(default)]
    pub cached_input_tokens: u32,
    /// Tokens written to the prompt cache (typically billed at a premium).
    #[serde(default)]
    pub cache_creation_input_tokens: u32,
    /// Tokens spent on internal reasoning (Anthropic thinking, OpenAI
    /// o-series reasoning, Gemini thinking budget).
    #[serde(default)]
    pub reasoning_tokens: u32,
    /// Per-category safety scores reported by the vendor (Gemini today;
    /// empty otherwise).
    #[serde(default)]
    pub safety_ratings: Vec<SafetyRating>,
}

impl Usage {
    /// Construct a `Usage` from the two universally-populated token
    /// counts; cache, reasoning, and safety fields stay at their
    /// defaults (`0` / `Vec::new()`). Use the `with_*` setters to
    /// override the rest.
    #[must_use]
    pub fn new(input_tokens: u32, output_tokens: u32) -> Self {
        Self {
            input_tokens,
            output_tokens,
            ..Self::default()
        }
    }

    /// Override `cached_input_tokens` (prompt-cache reads).
    #[must_use]
    pub const fn with_cached_input_tokens(mut self, tokens: u32) -> Self {
        self.cached_input_tokens = tokens;
        self
    }

    /// Override `cache_creation_input_tokens` (prompt-cache writes).
    #[must_use]
    pub const fn with_cache_creation_input_tokens(mut self, tokens: u32) -> Self {
        self.cache_creation_input_tokens = tokens;
        self
    }

    /// Override `reasoning_tokens` (Anthropic thinking, OpenAI o-series
    /// reasoning, Gemini thinking budget).
    #[must_use]
    pub const fn with_reasoning_tokens(mut self, tokens: u32) -> Self {
        self.reasoning_tokens = tokens;
        self
    }

    /// Attach the vendor-reported per-category safety ratings.
    #[must_use]
    pub fn with_safety_ratings(mut self, ratings: Vec<SafetyRating>) -> Self {
        self.safety_ratings = ratings;
        self
    }

    /// Billable input tokens — fresh prompt input plus cache writes (which
    /// vendors typically charge at a premium). Cache *reads* are excluded
    /// because vendors discount them, often heavily.
    #[must_use]
    pub const fn billable_input(&self) -> u32 {
        self.input_tokens
            .saturating_add(self.cache_creation_input_tokens)
    }

    /// Sum of input + output tokens (rough cost proxy when no per-bucket
    /// pricing is configured).
    #[must_use]
    pub const fn total(&self) -> u32 {
        self.input_tokens.saturating_add(self.output_tokens)
    }
}
