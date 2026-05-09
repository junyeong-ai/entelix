//! `ChatModelSamplingProvider<C, T>` — bridges
//! [`crate::SamplingProvider`] onto an
//! [`entelix_core::ChatModel<C, T>`]. Feature-gated behind
//! `chatmodel-sampling` so MCP consumers that wire their own
//! `SamplingProvider` against a non-`ChatModel` model don't pay the
//! extra compile cost.

use async_trait::async_trait;
use entelix_core::ChatModel;
use entelix_core::codecs::Codec;
use entelix_core::context::ExecutionContext;
use entelix_core::ir::{ContentPart, MediaSource, Message, ModelResponse, Role, StopReason};
use entelix_core::transports::Transport;

use crate::{
    McpError, McpResult, SamplingContent, SamplingMessage, SamplingProvider, SamplingRequest,
    SamplingResponse,
};

/// Stop-reason wire tokens MCP carries on the
/// [`SamplingResponse::stop_reason`] field. Mirrors the spec's
/// camel-case shape; `Other { raw }` from the IR passes through
/// the raw string verbatim so vendor-specific stop reasons survive
/// without information loss.
const STOP_END_TURN: &str = "endTurn";
const STOP_MAX_TOKENS: &str = "maxTokens";
const STOP_SEQUENCE: &str = "stopSequence";

/// Adapter from [`SamplingProvider`] to
/// [`entelix_core::ChatModel<C, T>`]. Cheap to clone — the
/// underlying chat model is `Arc`-backed.
///
/// Per-request overrides supplied on the
/// [`SamplingRequest`](crate::SamplingRequest)
/// (`system_prompt`, `temperature`, `max_tokens`,
/// `stop_sequences`) are applied to a per-call clone of the chat
/// model so the wrapped instance is unchanged across concurrent
/// dispatches. `model_preferences` and `include_context` are
/// advisory in MCP and are surfaced via `tracing::debug!` rather
/// than enforced — operators with bespoke routing implement
/// `SamplingProvider` directly and use this crate as a reference.
#[derive(Debug)]
pub struct ChatModelSamplingProvider<C: Codec + 'static, T: Transport + 'static> {
    chat: ChatModel<C, T>,
}

impl<C: Codec + 'static, T: Transport + 'static> Clone for ChatModelSamplingProvider<C, T> {
    fn clone(&self) -> Self {
        Self {
            chat: self.chat.clone(),
        }
    }
}

impl<C: Codec + 'static, T: Transport + 'static> ChatModelSamplingProvider<C, T> {
    /// Wrap a configured [`ChatModel<C, T>`].
    #[must_use]
    pub const fn new(chat: ChatModel<C, T>) -> Self {
        Self { chat }
    }

    /// Borrow the wrapped [`ChatModel<C, T>`].
    #[must_use]
    pub const fn chat_model(&self) -> &ChatModel<C, T> {
        &self.chat
    }
}

#[async_trait]
impl<C, T> SamplingProvider for ChatModelSamplingProvider<C, T>
where
    C: Codec + std::fmt::Debug + 'static,
    T: Transport + std::fmt::Debug + 'static,
{
    async fn sample(&self, request: SamplingRequest) -> McpResult<SamplingResponse> {
        let messages = request
            .messages
            .iter()
            .map(sampling_to_message)
            .collect::<McpResult<Vec<_>>>()?;

        // Per-request clone — Arc-backed, cheap. The wrapped
        // ChatModel stays unchanged across concurrent dispatches
        // even when MCP servers send different overrides.
        let mut chat = self.chat.clone();
        if let Some(prompt) = request.system_prompt.as_deref() {
            chat = chat.with_system(prompt);
        }
        if let Some(temp) = request.temperature {
            // Codec accepts f32; SamplingRequest carries f64
            // because that's the spec wire type. Loss across the
            // f64 → f32 boundary is bounded and accepted (vendor
            // temperature ranges are 0.0–2.0).
            #[allow(clippy::cast_possible_truncation)]
            let cast = temp as f32;
            chat = chat.with_temperature(cast);
        }
        if let Some(cap) = request.max_tokens {
            chat = chat.with_max_tokens(cap);
        }
        if !request.stop_sequences.is_empty() {
            chat = chat.with_stop_sequences(request.stop_sequences.clone());
        }

        // Advisory MCP fields — surfaced for operator dashboards
        // but not enforced (the spec is intentionally vague).
        if let Some(prefs) = request.model_preferences.as_ref() {
            tracing::debug!(
                target: "entelix_mcp::chatmodel",
                hints = ?prefs.hints,
                cost_priority = ?prefs.cost_priority,
                speed_priority = ?prefs.speed_priority,
                intelligence_priority = ?prefs.intelligence_priority,
                "MCP sampling: model_preferences advisory hints (not honoured by default adapter)"
            );
        }
        if let Some(ctx_kind) = request.include_context {
            tracing::debug!(
                target: "entelix_mcp::chatmodel",
                include_context = ?ctx_kind,
                "MCP sampling: includeContext advisory (not honoured by default adapter)"
            );
        }

        let ctx = ExecutionContext::new();
        let response = chat
            .complete_full(messages, &ctx)
            .await
            .map_err(|err| map_chat_error(&err))?;
        Ok(model_response_to_sampling(&response))
    }
}

/// Wire `SamplingMessage.role` into IR `Role`. MCP spec restricts
/// `role` to `user` | `assistant`; anything else is a protocol
/// violation that the server SHOULD NOT have sent — surface a
/// JSON-RPC `-32602` invalid-params equivalent rather than
/// silently coercing.
fn translate_role(raw: &str) -> McpResult<Role> {
    match raw {
        "user" => Ok(Role::User),
        "assistant" => Ok(Role::Assistant),
        other => Err(McpError::Config(format!(
            "MCP sampling message role '{other}' is not 'user' or 'assistant' — \
             protocol-spec violation"
        ))),
    }
}

fn sampling_to_message(msg: &SamplingMessage) -> McpResult<Message> {
    let role = translate_role(&msg.role)?;
    let content = vec![sampling_content_to_part(&msg.content)];
    Ok(Message::new(role, content))
}

fn sampling_content_to_part(c: &SamplingContent) -> ContentPart {
    match c {
        SamplingContent::Text { text } => ContentPart::Text {
            text: text.clone(),
            cache_control: None,
        },
        SamplingContent::Image { data, mime_type } => ContentPart::Image {
            source: MediaSource::base64(mime_type.clone(), data.clone()),
            cache_control: None,
        },
        SamplingContent::Audio { data, mime_type } => ContentPart::Audio {
            source: MediaSource::base64(mime_type.clone(), data.clone()),
            cache_control: None,
        },
    }
}

fn model_response_to_sampling(response: &ModelResponse) -> SamplingResponse {
    let stop_reason = stop_reason_to_wire(&response.stop_reason);
    let content = first_emittable_content(&response.content);
    SamplingResponse {
        model: response.model.clone(),
        stop_reason,
        role: "assistant".into(),
        content,
    }
}

/// MCP `stop_reason` is stringly-typed on the wire. Map the IR
/// canonical variants to the spec tokens; `Other { raw }`
/// passes through verbatim so vendor-specific reasons survive.
/// `ToolUse` and `Refusal` have no MCP equivalent (sampling
/// channel has no tools, and the spec doesn't model refusals);
/// they collapse to `endTurn` — operators wanting sharper
/// semantics implement `SamplingProvider` directly.
fn stop_reason_to_wire(reason: &StopReason) -> String {
    match reason {
        StopReason::MaxTokens => STOP_MAX_TOKENS.into(),
        StopReason::StopSequence { .. } => STOP_SEQUENCE.into(),
        StopReason::Other { raw } => raw.clone(),
        // `EndTurn`, `ToolUse`, `Refusal` all collapse to `endTurn`
        // — sampling has no tools and the spec doesn't model
        // refusals, so the natural-stop token is the closest neutral.
        StopReason::EndTurn | StopReason::ToolUse | StopReason::Refusal { .. } => {
            STOP_END_TURN.into()
        }
        // `StopReason` is `#[non_exhaustive]`; future variants
        // collapse to endTurn until an explicit mapping is added.
        // Surfaces via tracing so operators notice.
        other => {
            tracing::warn!(
                target: "entelix_mcp::chatmodel",
                stop_reason = ?other,
                "MCP sampling: unmapped IR StopReason variant — collapsed to endTurn"
            );
            STOP_END_TURN.into()
        }
    }
}

/// MCP wire format admits exactly one [`SamplingContent`] per
/// response. Walk the IR `ContentPart` list and pick the first
/// emittable shape (`Text` / `Image` / `Audio`); skip `Thinking`
/// blocks (operator-internal scaffolding) and other variants.
/// When nothing emittable is found, return an empty `Text` —
/// MCP requires the field, and an empty body is the least
/// surprising fill.
fn first_emittable_content(parts: &[ContentPart]) -> SamplingContent {
    for part in parts {
        if let Some(content) = part_to_sampling_content(part) {
            // If we dropped any auxiliary blocks (Thinking, etc.),
            // surface the loss in tracing.
            if parts.len() > 1 {
                let dropped: Vec<&'static str> = parts
                    .iter()
                    .filter_map(|p| {
                        if std::ptr::eq(p, part) {
                            None
                        } else {
                            Some(part_kind(p))
                        }
                    })
                    .collect();
                if !dropped.is_empty() {
                    tracing::warn!(
                        target: "entelix_mcp::chatmodel",
                        dropped_kinds = ?dropped,
                        "MCP sampling: response carried multiple content parts; \
                         only the first emittable one survives the wire shape"
                    );
                }
            }
            return content;
        }
    }
    SamplingContent::Text {
        text: String::new(),
    }
}

fn part_to_sampling_content(part: &ContentPart) -> Option<SamplingContent> {
    match part {
        ContentPart::Text { text, .. } => Some(SamplingContent::Text { text: text.clone() }),
        ContentPart::Image { source, .. } => media_to_sampling(source, true),
        ContentPart::Audio { source, .. } => media_to_sampling(source, false),
        _ => None,
    }
}

fn media_to_sampling(source: &MediaSource, is_image: bool) -> Option<SamplingContent> {
    let MediaSource::Base64 { media_type, data } = source else {
        // MCP sampling carries inlined base64 only — Url and
        // FileId references can't be expressed on the wire and
        // would force a fetch the adapter doesn't perform.
        tracing::warn!(
            target: "entelix_mcp::chatmodel",
            ?source,
            "MCP sampling: non-base64 media source dropped from response"
        );
        return None;
    };
    let (data, mime_type) = (data.clone(), media_type.clone());
    Some(if is_image {
        SamplingContent::Image { data, mime_type }
    } else {
        SamplingContent::Audio { data, mime_type }
    })
}

const fn part_kind(part: &ContentPart) -> &'static str {
    match part {
        ContentPart::Text { .. } => "text",
        ContentPart::Image { .. } => "image",
        ContentPart::Audio { .. } => "audio",
        ContentPart::Video { .. } => "video",
        ContentPart::Document { .. } => "document",
        ContentPart::Thinking { .. } => "thinking",
        ContentPart::Citation { .. } => "citation",
        ContentPart::ToolUse { .. } => "tool_use",
        ContentPart::ToolResult { .. } => "tool_result",
        _ => "unknown",
    }
}

/// `entelix_core::Error` → `McpError`. Map provider failures onto
/// the JSON-RPC error shape MCP servers expect (the dispatcher in
/// `entelix-mcp` translates `McpError::JsonRpc { code, message }`
/// into the wire `error.code` slot).
///
/// The wire `error.message` flows to the MCP client (often an LLM)
/// through the JSON-RPC envelope; per invariant 16 the message is
/// composed only from the `entelix_core::Error::for_llm()` rendering
/// (operator-only diagnostics — source chains, vendor status codes,
/// raw provider messages — never enter this channel). Operator
/// observability continues through `tracing::error!`.
fn map_chat_error(err: &entelix_core::Error) -> McpError {
    use entelix_core::{Error, LlmRenderable};
    let code = match err {
        Error::InvalidRequest(_) | Error::Config(_) | Error::Auth(_) => -32602,
        _ => -32603,
    };
    tracing::error!(
        target: "entelix_mcp::chatmodel",
        error = %err,
        "MCP sampling chat-model failure",
    );
    McpError::JsonRpc {
        code,
        message: format!("MCP sampling: {}", err.for_llm().into_inner()),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn translate_role_user_assistant_only() {
        assert!(matches!(translate_role("user"), Ok(Role::User)));
        assert!(matches!(translate_role("assistant"), Ok(Role::Assistant)));
        assert!(translate_role("system").is_err());
        assert!(translate_role("tool").is_err());
        assert!(translate_role("").is_err());
    }

    #[test]
    fn sampling_text_to_content_part() {
        let c = sampling_content_to_part(&SamplingContent::Text { text: "hi".into() });
        assert!(matches!(c, ContentPart::Text { ref text, .. } if text == "hi"));
    }

    #[test]
    fn sampling_image_to_base64_image_part() {
        let c = sampling_content_to_part(&SamplingContent::Image {
            data: "AAA".into(),
            mime_type: "image/png".into(),
        });
        let ContentPart::Image { source, .. } = c else {
            panic!("expected Image");
        };
        let MediaSource::Base64 { media_type, data } = source else {
            panic!("expected Base64");
        };
        assert_eq!(media_type, "image/png");
        assert_eq!(data, "AAA");
    }

    #[test]
    fn stop_reason_canonical_mappings() {
        assert_eq!(stop_reason_to_wire(&StopReason::EndTurn), "endTurn");
        assert_eq!(stop_reason_to_wire(&StopReason::MaxTokens), "maxTokens");
        assert_eq!(
            stop_reason_to_wire(&StopReason::StopSequence {
                sequence: "STOP".into()
            }),
            "stopSequence"
        );
        // ToolUse / Refusal collapse to endTurn (no MCP equivalent).
        assert_eq!(stop_reason_to_wire(&StopReason::ToolUse), "endTurn");
        // Other passes through verbatim.
        assert_eq!(
            stop_reason_to_wire(&StopReason::Other {
                raw: "vendor-specific-token".into()
            }),
            "vendor-specific-token"
        );
    }

    #[test]
    fn first_emittable_content_skips_thinking_blocks() {
        let parts = vec![
            ContentPart::Thinking {
                text: "internal".into(),
                signature: None,
                cache_control: None,
            },
            ContentPart::Text {
                text: "user-facing".into(),
                cache_control: None,
            },
        ];
        let c = first_emittable_content(&parts);
        assert!(matches!(c, SamplingContent::Text { ref text } if text == "user-facing"));
    }

    #[test]
    fn first_emittable_content_empty_when_no_emittable() {
        let parts = vec![ContentPart::Thinking {
            text: "internal".into(),
            signature: None,
            cache_control: None,
        }];
        let c = first_emittable_content(&parts);
        assert!(matches!(c, SamplingContent::Text { ref text } if text.is_empty()));
    }

    /// Pin invariant 16 on the sampling-error wire path: the
    /// `error.message` carried back to the MCP client (often an LLM)
    /// composes only from the LLM-safe rendering of `Error`.
    /// Operator-only diagnostics — vendor messages, source chains,
    /// underlying auth detail — never surface.
    #[test]
    fn map_chat_error_strips_provider_message() {
        let err = entelix_core::Error::provider_http(
            503,
            "vendor returned reasoning-token bucket overflow",
        );
        let mapped = map_chat_error(&err);
        let McpError::JsonRpc { code, message } = mapped else {
            panic!("expected JsonRpc variant");
        };
        assert_eq!(code, -32603);
        assert_eq!(message, "MCP sampling: upstream model error");
        assert!(!message.contains("503"));
        assert!(!message.contains("reasoning-token"));
        assert!(!message.contains("vendor"));
    }

    #[test]
    fn map_chat_error_strips_auth_detail() {
        let err = entelix_core::Error::Auth(entelix_core::auth::AuthError::missing_from(
            "ANTHROPIC_API_KEY environment variable absent — operator install hint",
        ));
        let mapped = map_chat_error(&err);
        let McpError::JsonRpc { code, message } = mapped else {
            panic!("expected JsonRpc variant");
        };
        assert_eq!(code, -32602);
        assert_eq!(message, "MCP sampling: authentication failed");
        assert!(!message.contains("ANTHROPIC_API_KEY"));
        assert!(!message.contains("operator install hint"));
    }

    #[test]
    fn map_chat_error_strips_invalid_request_inner() {
        let err = entelix_core::Error::invalid_request("internal: tenant_id=t-private-001 missing");
        let mapped = map_chat_error(&err);
        let McpError::JsonRpc { code, message } = mapped else {
            panic!("expected JsonRpc variant");
        };
        assert_eq!(code, -32602);
        // InvalidRequest renders verbatim — caller is expected to
        // pass already-LLM-safe messages, but the rendering still
        // routes through `LlmRenderable` and never adds vendor
        // framing.
        assert!(message.starts_with("MCP sampling: invalid input:"));
        assert!(!message.contains("provider"));
        assert!(!message.contains("vendor"));
    }
}
