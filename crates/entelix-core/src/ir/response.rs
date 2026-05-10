//! `ModelResponse` — the provider-neutral reply shape.

use serde::{Deserialize, Serialize};

use crate::ir::content::ContentPart;
use crate::ir::provider_echo::ProviderEchoSnapshot;
use crate::ir::usage::Usage;
use crate::ir::warning::ModelWarning;
use crate::rate_limit::RateLimitSnapshot;

impl ModelResponse {
    /// Borrow the first text block, if any. Convenient when the
    /// model is expected to reply with a single text answer (the
    /// 5-line-agent path) — saves the manual `match
    /// &response.content[0] { ContentPart::Text { text, .. } => …,
    /// _ => panic!() }` dance at every call site.
    ///
    /// Returns `None` when the response has no text block (e.g. the
    /// reply is purely a `ToolUse` block).
    #[must_use]
    pub fn first_text(&self) -> Option<&str> {
        self.content.iter().find_map(|part| match part {
            ContentPart::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
    }

    /// Concatenate every text block in order. Useful when the model
    /// emits multiple `Text` blocks interleaved with `Thinking` or
    /// `ToolUse` blocks and the caller only wants the user-visible
    /// answer string.
    #[must_use]
    pub fn full_text(&self) -> String {
        let mut out = String::new();
        for part in &self.content {
            if let ContentPart::Text { text, .. } = part {
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(text);
            }
        }
        out
    }

    /// Borrow every `ToolUse` block in declaration order. Empty
    /// when the response carried no tool calls. Used by ReAct-style
    /// agents to drive the next dispatch round.
    #[must_use]
    pub fn tool_uses(&self) -> Vec<ToolUseRef<'_>> {
        self.content
            .iter()
            .filter_map(|part| match part {
                ContentPart::ToolUse {
                    id, name, input, ..
                } => Some(ToolUseRef { id, name, input }),
                _ => None,
            })
            .collect()
    }

    /// True iff the response has at least one `ToolUse` block —
    /// hot path for agent loops that branch on "did the model ask
    /// to call a tool?".
    #[must_use]
    pub fn has_tool_uses(&self) -> bool {
        self.content
            .iter()
            .any(|part| matches!(part, ContentPart::ToolUse { .. }))
    }
}

/// Borrowed view of a [`ContentPart::ToolUse`] block — returned by
/// [`ModelResponse::tool_uses`] so callers can iterate without
/// pattern-matching every entry. Owned data still lives on the
/// `ModelResponse`; this is a zero-copy projection.
#[derive(Clone, Copy, Debug)]
pub struct ToolUseRef<'a> {
    /// Stable id matched by the corresponding `ToolResult` reply.
    pub id: &'a str,
    /// Tool name the model invoked.
    pub name: &'a str,
    /// JSON arguments the model produced for the tool.
    pub input: &'a serde_json::Value,
}

/// One reply from a model invocation, after decoding.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelResponse {
    /// Vendor-assigned response ID (used for tracing and replay).
    pub id: String,
    /// Echo of the model that produced this response — useful when the codec
    /// resolved an alias (e.g. `claude-opus` → `claude-opus-4-7-20260415`).
    pub model: String,
    /// Why the model stopped producing tokens.
    pub stop_reason: StopReason,
    /// Returned content blocks (text, tool calls, etc.).
    pub content: Vec<ContentPart>,
    /// Token / cache accounting from the vendor.
    pub usage: Usage,
    /// Provider rate-limit state at response time, when the codec could
    /// extract it from response headers (`Codec::extract_rate_limit`).
    #[serde(default)]
    pub rate_limit: Option<RateLimitSnapshot>,
    /// Codec-emitted warnings (lossy encoding, unknown stop reasons, etc.).
    /// Always non-fatal; consumers may surface them in observability.
    #[serde(default)]
    pub warnings: Vec<ModelWarning>,
    /// Vendor-keyed opaque round-trip tokens that ride at the
    /// response root (rather than on a single content part) —
    /// OpenAI Responses `Response.id` is the canonical example,
    /// captured here so the *next* `ModelRequest::continued_from`
    /// can chain via `previous_response_id`. Codecs only populate
    /// entries matching their own `Codec::name`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub provider_echoes: Vec<ProviderEchoSnapshot>,
}

/// Why a refusal happened, when the model halts on a non-success
/// signal that codecs map to [`StopReason::Refusal`]. Vendors expose
/// distinct flavours — safety filters, copyright/recitation guards,
/// guardrail interventions, vendor-side failures — and the IR keeps
/// them separate so observability can report the right thing instead
/// of collapsing every refusal-shaped stop into a single bucket.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum RefusalReason {
    /// Vendor safety / content filter blocked the response
    /// (Anthropic `refusal`, OpenAI `content_filter`, Gemini `SAFETY`).
    Safety,
    /// Vendor recitation / copyright guard blocked the response
    /// (Gemini `RECITATION`).
    Recitation,
    /// Vendor guardrail intervened (Bedrock `guardrail_intervened` /
    /// `content_filtered`).
    Guardrail,
    /// Vendor declared the request failed for a non-safety reason
    /// (OpenAI Responses `status: "failed"`). Distinct from
    /// `Safety` because the cause is server-side rather than a
    /// content-policy decision.
    ProviderFailure,
    /// Vendor signalled a refusal but did not classify it. The raw
    /// vendor token is preserved so dashboards can group by it.
    Other {
        /// Raw vendor refusal string.
        raw: String,
    },
}

/// Reason the model halted generation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum StopReason {
    /// Natural end of turn.
    EndTurn,
    /// Hit `max_tokens` cap.
    MaxTokens,
    /// Matched one of `stop_sequences`.
    StopSequence {
        /// The matched stop string.
        sequence: String,
    },
    /// Model emitted a tool call and is waiting for the result.
    ToolUse,
    /// Model refused (safety / recitation / guardrail / provider
    /// failure). The codec classifies the flavour into
    /// [`RefusalReason`] so observability can split by cause.
    Refusal {
        /// Why the refusal happened.
        reason: RefusalReason,
    },
    /// Vendor returned a stop reason we don't model yet. Codec emits a
    /// `ModelWarning::UnknownStopReason` alongside.
    Other {
        /// Raw vendor reason string.
        raw: String,
    },
}
