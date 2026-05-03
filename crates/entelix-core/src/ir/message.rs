//! `Message` and `Role` — the conversational unit shared by every codec.

use serde::{Deserialize, Serialize};

use crate::ir::content::{ContentPart, ToolResultContent};

/// Conversational role assigned to a `Message`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Role {
    /// Free-form user input.
    User,
    /// Model-produced reply (or partial reply during streaming).
    Assistant,
    /// System / instruction message. Some providers carry this out-of-band
    /// (Anthropic `system` field); codecs handle the placement.
    System,
    /// A tool result message authored by the harness on behalf of a tool.
    Tool,
}

/// A single turn in the conversation.
///
/// `content` is always a list of [`ContentPart`]s. Codecs that accept a string
/// shorthand are responsible for collapsing a single `ContentPart::Text` to a
/// bare string at encode time.
///
/// `Message` is an open data carrier: codec/runnable internals
/// pattern-match exhaustively against the IR, so the type stays
/// constructable via struct-literal syntax. New IR signals land as
/// additional `ContentPart` variants (which `ContentPart`'s
/// `#[non_exhaustive]` covers) or as new helpers on `Message`.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// Who authored this message.
    pub role: Role,
    /// One or more content parts. Empty content is permitted (some providers
    /// emit empty assistant messages alongside tool calls).
    pub content: Vec<ContentPart>,
}

impl Message {
    /// Construct a message with a typed role + content list. Use the
    /// role-specific helpers (`user` / `assistant` / `system` / `tool_*`)
    /// for the common single-text-part cases; reach for `new` when
    /// assembling multi-part content (multimodal, tool-use blocks, etc.).
    #[must_use]
    pub fn new(role: Role, content: Vec<ContentPart>) -> Self {
        Self { role, content }
    }

    /// Convenience: `user` message with a single text part.
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentPart::text(text)],
        }
    }

    /// Convenience: `assistant` message with a single text part.
    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![ContentPart::text(text)],
        }
    }

    /// Convenience: `system` message with a single text part.
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: vec![ContentPart::text(text)],
        }
    }

    /// Convenience: `tool` message wrapping a tool's reply to a
    /// prior [`ContentPart::ToolUse`]. Mirrors LangChain's
    /// `ToolMessage(content=…, tool_call_id=…, name=…)` shape so
    /// the RAG / agent loop reads as a one-line append after each
    /// tool call instead of hand-constructing a `Message { role:
    /// Role::Tool, content: vec![ContentPart::ToolResult { … }] }`.
    ///
    /// Both `tool_use_id` and `name` are required: Anthropic /
    /// OpenAI / Bedrock correlate by id, but Gemini's
    /// `functionResponse` keys by `name`. Carrying both keeps
    /// the IR provider-neutral so a single agent harness works
    /// across all four codecs without per-vendor adaptation.
    ///
    /// `output` accepts any string-like — for structured payloads,
    /// use [`Self::tool_result_json`] and the codec will emit native
    /// JSON (or stringify with a `LossyEncode` warning if the
    /// provider lacks structured tool-result support).
    pub fn tool_result(
        tool_use_id: impl Into<String>,
        name: impl Into<String>,
        output: impl Into<String>,
    ) -> Self {
        Self {
            role: Role::Tool,
            content: vec![ContentPart::ToolResult {
                tool_use_id: tool_use_id.into(),
                name: name.into(),
                content: ToolResultContent::Text(output.into()),
                is_error: false,
                cache_control: None,
            }],
        }
    }

    /// Same as [`Self::tool_result`] but carries a structured JSON
    /// payload. Use when the tool returns objects/arrays the model
    /// should reason over without re-parsing a stringified blob.
    pub fn tool_result_json(
        tool_use_id: impl Into<String>,
        name: impl Into<String>,
        output: serde_json::Value,
    ) -> Self {
        Self {
            role: Role::Tool,
            content: vec![ContentPart::ToolResult {
                tool_use_id: tool_use_id.into(),
                name: name.into(),
                content: ToolResultContent::Json(output),
                is_error: false,
                cache_control: None,
            }],
        }
    }

    /// Same as [`Self::tool_result`] but flagged as an error reply.
    /// Anthropic and Bedrock surface the `is_error` flag natively;
    /// other codecs prefix the text or emit a `LossyEncode` warning.
    pub fn tool_error(
        tool_use_id: impl Into<String>,
        name: impl Into<String>,
        output: impl Into<String>,
    ) -> Self {
        Self {
            role: Role::Tool,
            content: vec![ContentPart::ToolResult {
                tool_use_id: tool_use_id.into(),
                name: name.into(),
                content: ToolResultContent::Text(output.into()),
                is_error: true,
                cache_control: None,
            }],
        }
    }
}
