//! `SamplingProvider` + request/response shapes — client-side
//! answer to the server-initiated `sampling/createMessage`
//! request (MCP 2024-11-05 §"Sampling").
//!
//! Sampling lets an MCP server ask the client to run an LLM
//! completion on its behalf — typically when the server needs
//! reasoning capability it doesn't own (e.g., a server
//! orchestrating tool dispatch wants the agent's LLM to choose
//! the next tool). The server provides the conversation
//! prefix, optional sampling parameters, and gets back a
//! finalized assistant message.
//!
//! ## Why a trait, not a `ChatModel` adapter shipped here
//!
//! A "wire `ChatModel` directly" adapter would force this
//! crate to depend on the chat-model surface. Instead the
//! trait stays minimal and operators write a 20-line wrapper
//! that converts MCP messages → `entelix_core::ir::Message` →
//! `ChatModel::invoke` → MCP response. The conversion is
//! deployment-specific (which model, which prompt envelope,
//! which IR translation) and doesn't generalise cleanly into
//! the trait surface.
//!
//! ## No `ExecutionContext` parameter
//!
//! Mirrors [`crate::RootsProvider`] and
//! [`crate::ElicitationProvider`]: server-initiated requests
//! arrive on a background SSE listener, not in the middle of
//! a client-driven call. No honest context to thread.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::McpResult;

/// One conversation message in a sampling request.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SamplingMessage {
    /// Speaker role — either `user` or `assistant` per MCP
    /// spec. The variants are stringly-typed at the wire level
    /// because MCP doesn't enumerate them — the agent is free
    /// to pass through whatever role the server requested.
    pub role: String,
    /// Message body. Text is by far the common case; image /
    /// audio variants exist for multimodal servers.
    pub content: SamplingContent,
}

/// Body of one [`SamplingMessage`]. Tagged by `type` field on
/// the wire — matches the MCP spec's content-block shape.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
#[non_exhaustive]
pub enum SamplingContent {
    /// Plain text content.
    Text {
        /// UTF-8 text body.
        text: String,
    },
    /// Image content (base64-encoded data + MIME type).
    Image {
        /// Base64-encoded image bytes.
        data: String,
        /// MIME type (e.g., `image/png`, `image/jpeg`).
        #[serde(rename = "mimeType")]
        mime_type: String,
    },
    /// Audio content (base64-encoded data + MIME type).
    Audio {
        /// Base64-encoded audio bytes.
        data: String,
        /// MIME type (e.g., `audio/wav`).
        #[serde(rename = "mimeType")]
        mime_type: String,
    },
}

/// Operator hints + priorities the server passes to bias
/// model selection. All fields optional — the provider
/// chooses how to honour them (or ignores them entirely).
#[derive(Clone, Debug, Default, PartialEq, Deserialize, Serialize)]
pub struct ModelPreferences {
    /// Suggested model names, best-fit-first. Provider may
    /// match against any of them; spec encourages substring
    /// matching (e.g., hint `"claude-3-sonnet"` matches
    /// `"claude-3-sonnet-20240229"`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hints: Vec<ModelHint>,
    /// Cost-vs-quality preference in `[0.0, 1.0]`. Higher =
    /// prefer cheaper models.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "costPriority"
    )]
    pub cost_priority: Option<f64>,
    /// Speed-vs-quality preference in `[0.0, 1.0]`. Higher =
    /// prefer faster models.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "speedPriority"
    )]
    pub speed_priority: Option<f64>,
    /// Intelligence-vs-cost preference in `[0.0, 1.0]`. Higher
    /// = prefer more capable models.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "intelligencePriority"
    )]
    pub intelligence_priority: Option<f64>,
}

/// One model hint (a name string the server suggests).
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct ModelHint {
    /// Hint string (e.g., `"claude-3-sonnet"`).
    pub name: String,
}

/// How much surrounding context the server wants the client
/// to include in the sampling call. The client decides how to
/// honour this (the spec is intentionally vague — it's a
/// hint, not a contract).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub enum IncludeContext {
    /// No context outside the supplied messages.
    #[default]
    None,
    /// Include context from this MCP server only.
    ThisServer,
    /// Include context from every MCP server the client knows.
    AllServers,
}

/// Sampling request as it arrives from the server. Mirrors
/// the spec's `sampling/createMessage` params block.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct SamplingRequest {
    /// Conversation prefix the server wants completed.
    pub messages: Vec<SamplingMessage>,
    /// Optional model selection hints + priorities.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "modelPreferences"
    )]
    pub model_preferences: Option<ModelPreferences>,
    /// Optional system prompt to prepend.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "systemPrompt"
    )]
    pub system_prompt: Option<String>,
    /// Whether to include surrounding context.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "includeContext"
    )]
    pub include_context: Option<IncludeContext>,
    /// Sampling temperature (vendor-defined range — typically
    /// `[0.0, 1.0]` or `[0.0, 2.0]`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Token cap for the completion. Servers SHOULD set this;
    /// providers that pass through to a vendor demanding the
    /// field (Anthropic) reject the request when missing.
    #[serde(default, skip_serializing_if = "Option::is_none", rename = "maxTokens")]
    pub max_tokens: Option<u32>,
    /// Stop sequences (model halts as soon as one is generated).
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        rename = "stopSequences"
    )]
    pub stop_sequences: Vec<String>,
    /// Vendor-opaque metadata the server attached; passed
    /// through to the provider verbatim.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

/// Sampling response the client sends back. Mirrors the spec's
/// `sampling/createMessage` result block.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SamplingResponse {
    /// Model identifier the provider used (e.g.,
    /// `"claude-3-sonnet-20240229"`). Surfaced for the
    /// server's audit / cost accounting.
    pub model: String,
    /// Why generation stopped. MCP spec uses string tokens
    /// (`endTurn`, `stopSequence`, `maxTokens`); the field
    /// stays stringly-typed to mirror the wire shape.
    #[serde(rename = "stopReason")]
    pub stop_reason: String,
    /// Role of the produced message — typically `assistant`.
    pub role: String,
    /// Generated content. Same shape as request messages.
    pub content: SamplingContent,
}

/// Async source-of-truth for sampling completions. Mirrors
/// the `*Provider` taxonomy (ADR-0010) — async, single-purpose,
/// replaceable.
///
/// Operators wire one provider per server through
/// [`crate::McpServerConfig::with_sampling_provider`]. Most
/// production providers wrap a `ChatModel` from
/// `entelix_core` — convert MCP messages to IR, dispatch, map
/// the response back. The trait stays minimal so the
/// conversion choices stay operator-side.
#[async_trait]
pub trait SamplingProvider: Send + Sync + 'static + std::fmt::Debug {
    /// Resolve one server-initiated sampling request — call
    /// the underlying LLM (or stub it) and return the result.
    async fn sample(&self, request: SamplingRequest) -> McpResult<SamplingResponse>;
}

/// In-memory [`SamplingProvider`] returning a fixed response.
///
/// Useful for tests and for deployments that want a
/// deterministic stub (e.g., during local development before
/// a real LLM is wired).
#[derive(Clone, Debug)]
pub struct StaticSamplingProvider {
    response: SamplingResponse,
}

impl StaticSamplingProvider {
    /// Wrap a fixed response.
    #[must_use]
    pub const fn new(response: SamplingResponse) -> Self {
        Self { response }
    }

    /// Convenience: text-only response with `endTurn` stop reason.
    #[must_use]
    pub fn text(model: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            response: SamplingResponse {
                model: model.into(),
                stop_reason: "endTurn".into(),
                role: "assistant".into(),
                content: SamplingContent::Text { text: text.into() },
            },
        }
    }
}

#[async_trait]
impl SamplingProvider for StaticSamplingProvider {
    async fn sample(&self, _request: SamplingRequest) -> McpResult<SamplingResponse> {
        Ok(self.response.clone())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn text_content_serializes_with_type_tag() {
        let c = SamplingContent::Text {
            text: "hello".into(),
        };
        let s = serde_json::to_value(&c).unwrap();
        assert_eq!(s, json!({"type": "text", "text": "hello"}));
    }

    #[test]
    fn image_content_serializes_with_mime_type() {
        let c = SamplingContent::Image {
            data: "AAAA".into(),
            mime_type: "image/png".into(),
        };
        let s = serde_json::to_value(&c).unwrap();
        assert_eq!(
            s,
            json!({"type": "image", "data": "AAAA", "mimeType": "image/png"})
        );
    }

    #[test]
    fn request_deserializes_from_wire_shape_with_optional_fields() {
        let raw = json!({
            "messages": [
                {"role": "user", "content": {"type": "text", "text": "hi"}}
            ],
            "modelPreferences": {
                "hints": [{"name": "claude-3-sonnet"}],
                "intelligencePriority": 0.9
            },
            "systemPrompt": "be concise",
            "includeContext": "thisServer",
            "temperature": 0.7,
            "maxTokens": 256
        });
        let parsed: SamplingRequest = serde_json::from_value(raw).unwrap();
        assert_eq!(parsed.messages.len(), 1);
        assert_eq!(
            parsed.messages[0].content,
            SamplingContent::Text { text: "hi".into() }
        );
        let prefs = parsed.model_preferences.as_ref().unwrap();
        assert_eq!(prefs.hints[0].name, "claude-3-sonnet");
        assert_eq!(prefs.intelligence_priority, Some(0.9));
        assert_eq!(parsed.system_prompt.as_deref(), Some("be concise"));
        assert_eq!(parsed.include_context, Some(IncludeContext::ThisServer));
        assert_eq!(parsed.max_tokens, Some(256));
    }

    #[test]
    fn request_deserializes_minimal_messages_only() {
        let raw = json!({
            "messages": [{"role": "user", "content": {"type": "text", "text": "x"}}]
        });
        let parsed: SamplingRequest = serde_json::from_value(raw).unwrap();
        assert!(parsed.model_preferences.is_none());
        assert!(parsed.system_prompt.is_none());
        assert!(parsed.include_context.is_none());
        assert!(parsed.temperature.is_none());
        assert!(parsed.max_tokens.is_none());
        assert!(parsed.stop_sequences.is_empty());
    }

    #[test]
    fn response_serializes_with_stop_reason_camel_case() {
        let r = SamplingResponse {
            model: "claude-3".into(),
            stop_reason: "endTurn".into(),
            role: "assistant".into(),
            content: SamplingContent::Text {
                text: "done".into(),
            },
        };
        let s = serde_json::to_value(&r).unwrap();
        assert_eq!(s["model"], "claude-3");
        assert_eq!(s["stopReason"], "endTurn");
        assert_eq!(s["content"]["type"], "text");
    }

    #[tokio::test]
    async fn static_text_provider_returns_configured_response() {
        let provider = StaticSamplingProvider::text("claude-3", "ack");
        let req = SamplingRequest {
            messages: vec![],
            model_preferences: None,
            system_prompt: None,
            include_context: None,
            temperature: None,
            max_tokens: None,
            stop_sequences: vec![],
            metadata: None,
        };
        let resp = provider.sample(req).await.unwrap();
        assert_eq!(resp.model, "claude-3");
        assert_eq!(resp.stop_reason, "endTurn");
        assert_eq!(resp.role, "assistant");
        assert_eq!(resp.content, SamplingContent::Text { text: "ack".into() });
    }
}
