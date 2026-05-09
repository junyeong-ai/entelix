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
    /// `anthropic-beta` HTTP header values — comma-joined and sent
    /// as a single header so beta capabilities (extended thinking,
    /// computer-use updates, prompt-caching variants, …) gate at
    /// the transport layer per Anthropic's documented opt-in.
    /// Anthropic-specific (no other vendor exposes a comparable
    /// capability-gating header).
    /// Empty vec means no beta header is sent.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub betas: Vec<String>,
}

impl AnthropicExt {
    /// Replace the `anthropic-beta` opt-in list. Each element rides
    /// as one comma-separated value in the single `anthropic-beta`
    /// header the Anthropic Messages API documents.
    #[must_use]
    pub fn with_betas<I, S>(mut self, betas: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.betas = betas.into_iter().map(Into::into).collect();
        self
    }
}

/// OpenAI Chat Completions API knobs.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct OpenAiChatExt {
    /// `prompt_cache_key` — routing key into OpenAI's auto-cache
    /// bucket. Related requests sharing this key land in the same
    /// cache shard for higher hit rate. OpenAI-specific (Anthropic
    /// per-block `cache_control`, Gemini `cachedContent`, and
    /// Bedrock `cachePoint` are different mechanisms with different
    /// shapes). Mirrored on [`OpenAiResponsesExt::cache_key`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_key: Option<String>,
    /// `service_tier` — cost / latency routing for the request.
    /// `Flex` halves cost in exchange for higher latency, `Priority`
    /// reserves dedicated capacity for SLA-bound workflows, `Scale`
    /// targets sustained high-throughput tenants. OpenAI-specific
    /// (no other vendor exposes a comparable per-request routing
    /// knob). Mirrored on [`OpenAiResponsesExt::service_tier`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,
}

impl OpenAiChatExt {
    /// Set the prompt-cache routing key.
    #[must_use]
    pub fn with_cache_key(mut self, key: impl Into<String>) -> Self {
        self.cache_key = Some(key.into());
        self
    }
    /// Set the service-tier routing.
    #[must_use]
    pub const fn with_service_tier(mut self, tier: ServiceTier) -> Self {
        self.service_tier = Some(tier);
        self
    }
}

/// OpenAI Responses API knobs.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct OpenAiResponsesExt {
    /// `prompt_cache_key` — routing key into OpenAI's auto-cache
    /// bucket (same semantics as [`OpenAiChatExt::cache_key`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_key: Option<String>,
    /// `service_tier` — cost / latency routing (same semantics as
    /// [`OpenAiChatExt::service_tier`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,
    /// Reasoning summary verbosity for o-series models. When set,
    /// the codec emits `reasoning.summary: "<mode>"` at the
    /// Responses API request root, paired with the cross-vendor
    /// [`crate::ir::ModelRequest::reasoning_effort`] field. The
    /// summary knob is OpenAI-specific (Anthropic / Gemini /
    /// Bedrock have no equivalent) so it stays on this extension
    /// rather than the IR root.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_summary: Option<ReasoningSummary>,
}

impl OpenAiResponsesExt {
    /// Set the prompt-cache routing key.
    #[must_use]
    pub fn with_cache_key(mut self, key: impl Into<String>) -> Self {
        self.cache_key = Some(key.into());
        self
    }
    /// Set the service-tier routing.
    #[must_use]
    pub const fn with_service_tier(mut self, tier: ServiceTier) -> Self {
        self.service_tier = Some(tier);
        self
    }
    /// Set the reasoning summary verbosity. Pair with
    /// [`crate::ir::ModelRequest::reasoning_effort`] (the
    /// cross-vendor effort knob) — OpenAI Responses emits both as
    /// `reasoning: { effort, summary }`.
    #[must_use]
    pub const fn with_reasoning_summary(mut self, summary: ReasoningSummary) -> Self {
        self.reasoning_summary = Some(summary);
        self
    }
}

/// OpenAI service-tier routing — cost / latency knob shared by both
/// Chat Completions and Responses APIs. OpenAI-specific (no other
/// vendor exposes a comparable per-request routing channel).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ServiceTier {
    /// Vendor picks the tier (default behaviour when the field is
    /// omitted; supplying `Auto` makes the choice explicit).
    Auto,
    /// Standard processing tier — vendor's default capacity pool.
    Default,
    /// Cheaper async tier with relaxed latency SLO (~50% cost cut).
    Flex,
    /// Reserved-capacity tier for latency-bound enterprise workflows.
    Priority,
    /// High-throughput sustained-traffic tier.
    Scale,
}

/// Verbosity of the reasoning summary surfaced alongside the reply
/// — OpenAI Responses-only (Anthropic / Gemini / Bedrock have no
/// equivalent knob). Cross-vendor effort levels live on
/// [`crate::ir::ReasoningEffort`].
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
    /// Gemini-specific (Anthropic / OpenAI / Bedrock expose
    /// content-moderation through different surfaces — Anthropic
    /// has no per-request override, OpenAI surfaces it as
    /// `moderations` API responses, Bedrock surfaces it as
    /// `BedrockExt::guardrail` with vendor-issued identifiers).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub safety_settings: Vec<GeminiSafetyOverride>,
    /// `candidateCount` — number of independent completions to
    /// generate. Defaults to vendor default (1) when unset.
    /// Gemini-specific (Anthropic / OpenAI / Bedrock require N
    /// separate requests to obtain N completions; only Gemini
    /// batches them into one call).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_count: Option<u32>,
    /// `cachedContent` — server-side cached-content reference. The
    /// value is a vendor-minted resource name (e.g.
    /// `cachedContents/<id>`) typically returned by a prior
    /// `cachedContents.create` API call; the wire codec emits it
    /// verbatim. Gemini-specific (Anthropic per-block
    /// `cache_control`, OpenAI `prompt_cache_key`, and Bedrock
    /// `cachePoint` are different mechanisms with different shapes).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_content: Option<String>,
    /// `url_context` — opt into Gemini's built-in URL-fetch tool.
    /// When enabled the codec emits `tools[].url_context: {}` so
    /// the model can autonomously fetch and ground on URLs found
    /// in the prompt (up to 20 per request, vendor-side cap).
    /// Gemini-specific (no other vendor exposes a comparable
    /// in-context URL-fetch primitive).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url_context: Option<UrlContext>,
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
    /// Set the server-side cached-content reference.
    #[must_use]
    pub fn with_cached_content(mut self, name: impl Into<String>) -> Self {
        self.cached_content = Some(name.into());
        self
    }
    /// Enable Gemini's built-in `url_context` tool for the request.
    #[must_use]
    pub const fn with_url_context(mut self) -> Self {
        self.url_context = Some(UrlContext::ENABLED);
        self
    }
}

/// Toggle marker for Gemini's built-in `url_context` tool. Constructed
/// via [`UrlContext::ENABLED`] (or `Default`) — the wire shape is a
/// parameterless `{}` object, so no runtime fields exist today; the
/// struct is `#[non_exhaustive]` so a future Gemini release that
/// surfaces options (max URLs, content filters, …) lands additively.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct UrlContext;

impl UrlContext {
    /// Single canonical instance — the tool is parameterless on the
    /// wire today, so every enable site shares this value.
    pub const ENABLED: Self = Self;
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
    /// console-create time. Bedrock-specific (Anthropic /
    /// OpenAI / Gemini have no equivalent operator-defined
    /// per-request safety policy reference; safety is either
    /// vendor-managed or surfaced as
    /// [`GeminiExt::safety_settings`] inline overrides).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guardrail: Option<BedrockGuardrail>,
    /// Performance-tier hint (`standard` / `optimized`). Bedrock
    /// uses this to route to a faster pool for latency-sensitive
    /// workflows where a fraction of model quality is acceptable.
    /// Bedrock-specific (Anthropic / OpenAI / Gemini do not
    /// expose a per-request latency-vs-quality routing knob).
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
            .with_anthropic(AnthropicExt::default().with_betas(["thinking-2025"]))
            .with_openai_chat(OpenAiChatExt::default().with_cache_key("user-42"))
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
            ext.anthropic.as_ref().unwrap().betas,
            vec!["thinking-2025".to_owned()]
        );
        assert_eq!(
            ext.openai_chat.as_ref().unwrap().cache_key.as_deref(),
            Some("user-42")
        );
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
            .with_anthropic(AnthropicExt::default().with_betas(["computer-use-2025"]))
            .with_gemini(GeminiExt {
                safety_settings: vec![GeminiSafetyOverride {
                    category: "HARM_CATEGORY_HATE_SPEECH".into(),
                    threshold: "BLOCK_LOW_AND_ABOVE".into(),
                }],
                candidate_count: None,
                cached_content: None,
                url_context: None,
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
