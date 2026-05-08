//! `ModelRequest` — the provider-neutral request shape (invariant 4).
//!
//! Every model call must pass through this type before reaching a `Codec`.

use serde::{Deserialize, Serialize};

use crate::ir::message::Message;
use crate::ir::provider_extensions::ProviderExtensions;
use crate::ir::reasoning::ReasoningEffort;
use crate::ir::structured::ResponseFormat;
use crate::ir::system::SystemPrompt;
use crate::ir::tool_spec::{ToolChoice, ToolSpec};

/// One model invocation, before encoding to vendor wire format.
///
/// Built by users (or higher-level recipes) and handed to `Codec::encode`.
/// Codecs produce vendor-shaped JSON; the IR is the canonical surface and
/// never carries vendor-specific fields directly.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelRequest {
    /// Vendor model identifier (e.g. `claude-opus-4-7`, `gpt-4.1`).
    pub model: String,
    /// Conversation up to this turn. Must contain at least one user message
    /// for most providers; codecs reject empty lists at encode time.
    pub messages: Vec<Message>,
    /// Ordered system-prompt blocks. Empty = "no system prompt"
    /// (codecs treat as if the field were absent). Per-block
    /// [`crate::ir::CacheControl`] is honored natively by codecs
    /// that support it (Anthropic, Bedrock Converse for Claude);
    /// other codecs concatenate block text and emit
    /// `LossyEncode` warnings when any block is cached.
    #[serde(default)]
    pub system: SystemPrompt,
    /// Hard cap on output tokens. `None` = vendor default.
    pub max_tokens: Option<u32>,
    /// Sampling temperature `[0.0, 2.0]`. Codecs clamp to vendor range.
    pub temperature: Option<f32>,
    /// Nucleus sampling parameter.
    pub top_p: Option<f32>,
    /// Sequences that, when produced, halt generation.
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    /// Tools advertised to the model. Empty = no tool calls permitted.
    #[serde(default)]
    pub tools: Vec<ToolSpec>,
    /// Constraint on tool selection. Defaults to [`ToolChoice::Auto`].
    #[serde(default)]
    pub tool_choice: ToolChoice,
    /// Optional structured-output constraint. Codecs route to
    /// vendor-canonical channels (Anthropic `output_config.format`,
    /// OpenAI `response_format` / `text.format`, Gemini
    /// `responseJsonSchema`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    /// Prompt-cache routing key. When set, codecs that
    /// expose a `prompt_cache_key`-style field (OpenAI Chat,
    /// OpenAI Responses) emit it so the vendor's auto-cache routes
    /// related requests into the same bucket. Other codecs emit
    /// `LossyEncode`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_key: Option<String>,
    /// Server-side cached-content reference. When set,
    /// codecs that accept it (Gemini's `cachedContent` request
    /// field) emit it; the value is a vendor-minted resource name
    /// (e.g. `cachedContents/<id>`) typically returned by a prior
    /// `cachedContents` API call. Other codecs emit `LossyEncode`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_content: Option<String>,
    /// Cross-vendor reasoning-effort knob. When `Some`, codecs
    /// translate onto their native wire shape per the mapping in
    /// [`ReasoningEffort`]'s module doc — `Off`/`Minimal`/`Low`/
    /// `Medium`/`High`/`Auto` snap to vendor buckets, lossy
    /// approximations emit `ModelWarning::LossyEncode`, and
    /// `VendorSpecific(s)` passes through the literal vendor wire
    /// value. `None` ⇒ vendor default (codec emits no thinking /
    /// reasoning field).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Per-vendor typed knobs that don't generalise to a
    /// cross-provider IR field — e.g. Anthropic
    /// `disable_parallel_tool_use`, Gemini `safetySettings`,
    /// Bedrock guardrails. Codecs read their own ext when encoding
    /// and emit `ModelWarning::ProviderExtensionIgnored` when
    /// another vendor's ext is present (the operator intended a
    /// knob this wire format cannot honour).
    #[serde(default, skip_serializing_if = "ProviderExtensions::is_empty")]
    pub provider_extensions: ProviderExtensions,
}
