//! `ProviderExtensions` — typed escape hatch for vendor-specific
//! request knobs that don't generalise to a cross-provider IR field.
//!
//! ## Why typed, not `serde_json::Value`
//!
//! Letting operators drop arbitrary JSON into a request body would
//! erode the IR honesty contract: codecs could no longer reason
//! about what crosses the wire, `LossyEncode` warnings would lose
//! meaning, and a typo in a vendor-specific key would surface as a
//! provider error rather than a Rust compile error. Each vendor
//! gets a `*Ext` struct with concrete fields; codecs read their
//! own ext when encoding and ignore others (with a `LossyEncode`
//! warning when an inactive vendor's ext is present — the operator
//! intended a knob the wire format cannot honour).
//!
//! ## Forward compatibility
//!
//! Every ext struct is `#[non_exhaustive]`. Adding a new field
//! (e.g. a vendor ships a new beta knob) is non-breaking — operators
//! that ignore the field keep working; operators that opt in get
//! type-checked access via the builder.

use serde::{Deserialize, Serialize};

/// Per-provider typed extensions. Defaults to `None` for every
/// vendor, which corresponds to "no overrides — codec uses its
/// vendor default for any field operators didn't set on the IR
/// proper".
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ProviderExtensions {
    /// Anthropic Messages API knobs. Honoured by
    /// `AnthropicMessagesCodec`; emits `LossyEncode` if set on
    /// requests routed to non-Anthropic codecs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anthropic: Option<AnthropicExt>,
    /// OpenAI Chat Completions knobs. Honoured by `OpenAiChatCodec`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub openai_chat: Option<OpenAiChatExt>,
    /// OpenAI Responses API knobs. Honoured by
    /// `OpenAiResponsesCodec`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub openai_responses: Option<OpenAiResponsesExt>,
    /// Gemini Generative Language API knobs. Honoured by
    /// `GeminiCodec`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gemini: Option<GeminiExt>,
    /// Bedrock Converse API knobs. Honoured by `BedrockConverseCodec`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bedrock: Option<BedrockExt>,
}

impl ProviderExtensions {
    /// Builder-style attach for the Anthropic ext.
    #[must_use]
    pub fn with_anthropic(mut self, ext: AnthropicExt) -> Self {
        self.anthropic = Some(ext);
        self
    }

    /// Builder-style attach for the OpenAI Chat ext.
    #[must_use]
    pub fn with_openai_chat(mut self, ext: OpenAiChatExt) -> Self {
        self.openai_chat = Some(ext);
        self
    }

    /// Builder-style attach for the OpenAI Responses ext.
    #[must_use]
    pub fn with_openai_responses(mut self, ext: OpenAiResponsesExt) -> Self {
        self.openai_responses = Some(ext);
        self
    }

    /// Builder-style attach for the Gemini ext.
    #[must_use]
    pub fn with_gemini(mut self, ext: GeminiExt) -> Self {
        self.gemini = Some(ext);
        self
    }

    /// Builder-style attach for the Bedrock ext.
    #[must_use]
    pub fn with_bedrock(mut self, ext: BedrockExt) -> Self {
        self.bedrock = Some(ext);
        self
    }

    /// True when no vendor ext is set — codecs short-circuit the
    /// LossyEncode-warning sweep on this hot path.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.anthropic.is_none()
            && self.openai_chat.is_none()
            && self.openai_responses.is_none()
            && self.gemini.is_none()
            && self.bedrock.is_none()
    }
}

/// Anthropic-specific request knobs.
///
/// Each field maps 1:1 to an Anthropic Messages API field that has
/// no cross-provider equivalent. Setting one of these on a request
/// routed to a non-Anthropic codec emits a `LossyEncode` warning.
///
/// Construct via [`AnthropicExt::default()`] and the `with_*`
/// chain — direct struct-literal construction is closed off by
/// `#[non_exhaustive]` so future field additions are non-breaking.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AnthropicExt {
    /// `disable_parallel_tool_use` — when `true`, Anthropic emits
    /// at most one `tool_use` block per turn even when the model
    /// would otherwise parallelise. Useful for tools that mutate
    /// shared state and cannot run concurrently.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disable_parallel_tool_use: Option<bool>,
    /// `metadata.user_id` — Anthropic stamps this on the request
    /// for end-user attribution in their abuse-monitoring stack.
    /// Operator pseudonymous id, not a PII identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    /// Extended-thinking budget — when set, Anthropic runs its
    /// chain-of-thought before the user-facing reply and the
    /// response carries `Thinking` content blocks alongside `Text`.
    /// Anthropic-on-Bedrock honours the same shape via Bedrock's
    /// `additionalModelRequestFields` passthrough; other vendors
    /// emit `LossyEncode`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
}

impl AnthropicExt {
    /// Set `disable_parallel_tool_use`. Builder-style.
    #[must_use]
    pub const fn with_disable_parallel_tool_use(mut self, disable: bool) -> Self {
        self.disable_parallel_tool_use = Some(disable);
        self
    }

    /// Set the abuse-monitoring `user_id`.
    #[must_use]
    pub fn with_user_id(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    /// Enable extended thinking with the supplied token budget.
    /// Anthropic recommends `budget_tokens >= 1024` for meaningful
    /// reasoning; the codec leaves the value untouched and lets the
    /// vendor reject obviously-invalid sizes (≤0 / > model cap).
    #[must_use]
    pub const fn with_thinking_budget(mut self, budget_tokens: u32) -> Self {
        self.thinking = Some(ThinkingConfig { budget_tokens });
        self
    }
}

/// Extended-thinking configuration for Anthropic-family models.
///
/// Wire shape: `thinking: { type: "enabled", budget_tokens: N }`
/// at the Messages API request root. The `type` field is constant
/// today; if Anthropic ever ships a `type: "disabled"` toggle the
/// IR adopts an enum variant rather than overloading this struct.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ThinkingConfig {
    /// Maximum tokens Anthropic may spend reasoning before the
    /// user-facing reply. Counted against the response's
    /// `output_tokens` budget, so callers size `max_tokens`
    /// accordingly.
    pub budget_tokens: u32,
}

/// OpenAI Chat Completions API knobs.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct OpenAiChatExt {
    /// Deterministic-generation seed. Same seed + same request →
    /// same output (best-effort; OpenAI documents this as not
    /// strictly guaranteed across model versions).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    /// `user` — abuse-monitoring identifier. Stable per end-user
    /// pseudonym; not PII.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

impl OpenAiChatExt {
    /// Set the deterministic-generation seed.
    #[must_use]
    pub const fn with_seed(mut self, seed: i64) -> Self {
        self.seed = Some(seed);
        self
    }
    /// Set the abuse-monitoring user identifier.
    #[must_use]
    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }
}

/// OpenAI Responses API knobs.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct OpenAiResponsesExt {
    /// Deterministic-generation seed (same semantics as
    /// [`OpenAiChatExt::seed`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    /// `user` — abuse-monitoring identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// Reasoning configuration for o-series models. When set, the
    /// codec emits `reasoning: { effort, summary? }` at the
    /// Responses API request root. Non-o-series models silently
    /// ignore the field server-side; the IR carries it so operators
    /// can route the same request across model tiers without
    /// branching.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,
}

impl OpenAiResponsesExt {
    /// Set the deterministic-generation seed.
    #[must_use]
    pub const fn with_seed(mut self, seed: i64) -> Self {
        self.seed = Some(seed);
        self
    }
    /// Set the abuse-monitoring user identifier.
    #[must_use]
    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }
    /// Configure reasoning effort (and optional summary verbosity).
    #[must_use]
    pub const fn with_reasoning(mut self, reasoning: ReasoningConfig) -> Self {
        self.reasoning = Some(reasoning);
        self
    }
}

/// Reasoning-effort knob for OpenAI o-series models.
///
/// Wire shape: `reasoning: {effort: "<level>", summary?: "<mode>"}`
/// at the Responses API request root.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ReasoningConfig {
    /// How hard the model should reason before replying.
    pub effort: ReasoningEffort,
    /// Whether (and how verbosely) to surface a reasoning summary
    /// alongside the reply. `None` = vendor default (omit field).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<ReasoningSummary>,
}

impl ReasoningConfig {
    /// Build with just an effort level (no summary).
    #[must_use]
    pub const fn new(effort: ReasoningEffort) -> Self {
        Self {
            effort,
            summary: None,
        }
    }
    /// Attach a summary verbosity preference.
    #[must_use]
    pub const fn with_summary(mut self, summary: ReasoningSummary) -> Self {
        self.summary = Some(summary);
        self
    }
}

/// Reasoning-effort levels OpenAI o-series accept on the Responses API.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ReasoningEffort {
    /// Minimal — fastest, least tokens, suitable for routing /
    /// classification.
    Minimal,
    /// Low — light reasoning.
    Low,
    /// Medium — vendor default for most o-series models.
    #[default]
    Medium,
    /// High — deepest reasoning, highest cost / latency.
    High,
}

/// Verbosity of the reasoning summary surfaced alongside the reply.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ReasoningSummary {
    /// Vendor picks the verbosity (`auto`).
    Auto,
    /// Concise summary — a few short sentences.
    Concise,
    /// Detailed summary — full chain breakdown.
    Detailed,
}

/// Gemini Generative Language API knobs.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct GeminiExt {
    /// `safetySettings` — per-category harm thresholds. Vendor
    /// canonical names ride through (`HARM_CATEGORY_HATE_SPEECH`,
    /// `HARM_CATEGORY_HARASSMENT`, …). Operators that need broader
    /// coverage than [`crate::ir::SafetyCategory`] use this ext.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub safety_settings: Vec<GeminiSafetyOverride>,
    /// `candidateCount` — number of independent completions to
    /// generate. Defaults to vendor default (1) when unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_count: Option<u32>,
}

impl GeminiExt {
    /// Append a safety-category override.
    #[must_use]
    pub fn with_safety_override(mut self, category: &str, threshold: &str) -> Self {
        self.safety_settings.push(GeminiSafetyOverride {
            category: category.to_owned(),
            threshold: threshold.to_owned(),
        });
        self
    }
    /// Set the `candidateCount`.
    #[must_use]
    pub const fn with_candidate_count(mut self, n: u32) -> Self {
        self.candidate_count = Some(n);
        self
    }
}

/// One Gemini safety-category override. Vendor names ride through
/// verbatim — see Gemini API docs for the full list.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeminiSafetyOverride {
    /// Category name (e.g. `HARM_CATEGORY_HATE_SPEECH`).
    pub category: String,
    /// Threshold name (e.g. `BLOCK_LOW_AND_ABOVE`).
    pub threshold: String,
}

/// Bedrock Converse API knobs.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct BedrockExt {
    /// Bedrock guardrail to enforce on the request. Carries the
    /// guardrail identifier and version Bedrock issued at
    /// console-create time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guardrail: Option<BedrockGuardrail>,
    /// Performance-tier hint (`standard` / `optimized`). Bedrock
    /// uses this to route to a faster pool for latency-sensitive
    /// workflows where a fraction of model quality is acceptable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub performance_config_tier: Option<String>,
}

impl BedrockExt {
    /// Attach a guardrail.
    #[must_use]
    pub fn with_guardrail(mut self, guardrail: BedrockGuardrail) -> Self {
        self.guardrail = Some(guardrail);
        self
    }
    /// Set the performance-tier hint.
    #[must_use]
    pub fn with_performance_config_tier(mut self, tier: impl Into<String>) -> Self {
        self.performance_config_tier = Some(tier.into());
        self
    }
}

/// Bedrock guardrail reference.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BedrockGuardrail {
    /// Guardrail identifier (UUID-like string AWS issued).
    pub identifier: String,
    /// Guardrail version (e.g. `"DRAFT"`, `"1"`).
    pub version: String,
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn default_is_empty() {
        let ext = ProviderExtensions::default();
        assert!(ext.is_empty());
    }

    #[test]
    fn builder_chain_attaches_each_vendor_ext() {
        let ext = ProviderExtensions::default()
            .with_anthropic(AnthropicExt {
                disable_parallel_tool_use: Some(true),
                ..Default::default()
            })
            .with_openai_chat(OpenAiChatExt {
                seed: Some(42),
                ..Default::default()
            })
            .with_gemini(GeminiExt {
                candidate_count: Some(2),
                ..Default::default()
            })
            .with_bedrock(BedrockExt {
                guardrail: Some(BedrockGuardrail {
                    identifier: "abc-123".into(),
                    version: "1".into(),
                }),
                ..Default::default()
            });
        assert!(!ext.is_empty());
        assert_eq!(
            ext.anthropic.as_ref().unwrap().disable_parallel_tool_use,
            Some(true)
        );
        assert_eq!(ext.openai_chat.as_ref().unwrap().seed, Some(42));
        assert_eq!(ext.gemini.as_ref().unwrap().candidate_count, Some(2));
        assert_eq!(
            ext.bedrock
                .as_ref()
                .unwrap()
                .guardrail
                .as_ref()
                .unwrap()
                .identifier,
            "abc-123"
        );
    }

    #[test]
    fn provider_extensions_serde_round_trip() {
        let ext = ProviderExtensions::default()
            .with_anthropic(
                AnthropicExt::default()
                    .with_disable_parallel_tool_use(true)
                    .with_user_id("op-9")
                    .with_thinking_budget(2048),
            )
            .with_gemini(GeminiExt {
                safety_settings: vec![GeminiSafetyOverride {
                    category: "HARM_CATEGORY_HATE_SPEECH".into(),
                    threshold: "BLOCK_LOW_AND_ABOVE".into(),
                }],
                candidate_count: None,
            });
        let s = serde_json::to_string(&ext).unwrap();
        let back: ProviderExtensions = serde_json::from_str(&s).unwrap();
        assert_eq!(ext, back);
    }

    #[test]
    fn empty_serialization_omits_inactive_vendor_keys() {
        // `skip_serializing_if = "Option::is_none"` keeps the wire
        // small for the common "no extensions" case.
        let ext = ProviderExtensions::default();
        let s = serde_json::to_string(&ext).unwrap();
        assert_eq!(s, "{}");
    }
}
