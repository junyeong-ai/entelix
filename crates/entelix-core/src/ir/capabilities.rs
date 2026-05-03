//! `Capabilities` — what features a (codec, model) pair supports.
//!
//! Codecs expose this via `Codec::capabilities(model)` so the harness
//! can preflight a request and emit a
//! [`ModelWarning::UnsupportedCapability`](crate::ir::ModelWarning)
//! (or hard-reject) when the IR references a feature the target can't
//! deliver.

use serde::{Deserialize, Serialize};

/// The capability surface a codec advertises for a given model.
///
/// One flag per IR feature that admits "supported / not supported"
/// resolution. Codec preflight reads these to issue `LossyEncode` /
/// `UnsupportedCapability` warnings before the wire encode rather than
/// silently dropping.
#[allow(clippy::struct_excessive_bools)] // capability flags are intentionally orthogonal
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Capabilities {
    /// Codec can deliver streaming responses.
    pub streaming: bool,
    /// Model accepts `ToolSpec`s and emits `ToolUse` content.
    pub tools: bool,

    /// Model accepts `ContentPart::Image`.
    pub multimodal_image: bool,
    /// Model accepts `ContentPart::Audio`.
    pub multimodal_audio: bool,
    /// Model accepts `ContentPart::Video`.
    pub multimodal_video: bool,
    /// Model accepts `ContentPart::Document`.
    pub multimodal_document: bool,

    /// Model honours a system / instruction prompt.
    pub system_prompt: bool,
    /// Model can be steered to JSON-Schema-validated output.
    pub structured_output: bool,
    /// Codec implements prompt caching for this model.
    pub prompt_caching: bool,

    /// Model produces reasoning / extended-thinking blocks.
    pub thinking: bool,
    /// Model produces grounded citations.
    pub citations: bool,
    /// `ToolKind::WebSearch` is natively supported by codec.
    pub web_search: bool,
    /// `ToolKind::Computer` is natively supported by codec.
    pub computer_use: bool,

    /// Maximum context window in tokens.
    pub max_context_tokens: u32,
}

impl Default for Capabilities {
    /// Conservative defaults — every flag off, context window 0. Codecs
    /// override per model so absent fields error rather than silently
    /// drop.
    fn default() -> Self {
        Self {
            streaming: false,
            tools: false,
            multimodal_image: false,
            multimodal_audio: false,
            multimodal_video: false,
            multimodal_document: false,
            system_prompt: false,
            structured_output: false,
            prompt_caching: false,
            thinking: false,
            citations: false,
            web_search: false,
            computer_use: false,
            max_context_tokens: 0,
        }
    }
}
