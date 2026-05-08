//! `ContentPart` — a single block within a [`Message`](crate::ir::Message).
//!
//! Provider-neutral. Vendors that lack support for a variant cause the
//! codec to emit a [`ModelWarning::LossyEncode`](crate::ir::ModelWarning)
//! rather than failing silently (invariant 6).

use serde::{Deserialize, Serialize};

use crate::ir::cache::CacheControl;
use crate::ir::source::{CitationSource, MediaSource};

/// One block of content inside a [`Message`](crate::ir::Message).
///
/// The enum is `#[non_exhaustive]` so future variants don't break user
/// `match` arms. New modalities or capability blocks land here as
/// additional variants — codecs reach 100% IR coverage by either
/// emitting native wire shape or a `LossyEncode` warning for each.
///
/// Every input-side variant carries an
/// `Option<CacheControl>` field () — operators mark a
/// block as cached and codecs that support per-block caching
/// (Anthropic, Bedrock-on-Anthropic) emit the directive natively.
/// Other codecs emit `LossyEncode`. The `ToolUse` variant — the
/// assistant's outbound call — does not carry caching: the model
/// emits it, there is nothing to cache.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// Plain UTF-8 text — the primary medium.
    Text {
        /// The text payload.
        text: String,
        /// Per-block cache directive.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },

    /// An image input.
    Image {
        /// Where the image bytes live.
        source: MediaSource,
        /// Per-block cache directive.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },

    /// An audio input.
    Audio {
        /// Where the audio bytes live.
        source: MediaSource,
        /// Per-block cache directive.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },

    /// A video input.
    Video {
        /// Where the video bytes live.
        source: MediaSource,
        /// Per-block cache directive.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },

    /// A document input (PDF, plain-text file, etc.).
    Document {
        /// Where the document bytes live.
        source: MediaSource,
        /// Display name shown to the model (e.g. `"contract.pdf"`).
        /// Optional — codecs that require a name supply a stable
        /// derivation when absent.
        name: Option<String>,
        /// Per-block cache directive.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },

    /// A reasoning / extended-thinking block produced by the assistant
    /// before the user-facing reply.
    ///
    /// Surfaced as its own variant (rather than mixed with `Text`) so
    /// recipes can show / hide / cache reasoning independently. Order
    /// relative to `Text` parts is preserved — vendors that rely on
    /// chain-of-thought integrity (Anthropic thinking) require the
    /// original block order on follow-up turns.
    Thinking {
        /// The reasoning text.
        text: String,
        /// Vendor signature for redaction-resistant replay (Anthropic
        /// supplies; others leave `None`).
        signature: Option<String>,
        /// Per-block cache directive.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },

    /// A grounded citation produced by the assistant — the `snippet`
    /// is the verbatim cited text; `source` describes provenance.
    Citation {
        /// The cited substring as it appears in the assistant's reply.
        snippet: String,
        /// Where the snippet came from.
        source: CitationSource,
        /// Per-block cache directive.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },

    /// A tool call emitted by the assistant. The harness dispatches it.
    /// Tool calls are model output — they do not carry a cache
    /// directive (the model emits each call afresh).
    ToolUse {
        /// Stable ID matched against the corresponding `ToolResult`.
        id: String,
        /// Tool name to dispatch (must exist in the active `ToolRegistry`).
        name: String,
        /// JSON arguments for the tool — must validate against the
        /// tool's `input_schema` before dispatch.
        input: serde_json::Value,
    },

    /// An image the assistant produced (vendor-managed image
    /// generation: OpenAI `image_generation_call.result`, Gemini
    /// `inline_data` parts on the model output, …). Distinct from
    /// [`Self::Image`] which is a user-supplied input.
    ///
    /// Output blocks have no cache directive — they are produced
    /// fresh per turn.
    ImageOutput {
        /// Where the produced image bytes live. Most vendors return
        /// inline base64 ([`MediaSource::Base64`]); some return a
        /// hosted URL.
        source: MediaSource,
    },

    /// Audio the assistant produced (text-to-speech reply). Distinct
    /// from [`Self::Audio`] which is a user-supplied input.
    AudioOutput {
        /// Where the produced audio bytes live.
        source: MediaSource,
        /// Optional textual transcript the vendor returned alongside
        /// the audio. Surfaced separately so callers can route
        /// transcript text through the operator's logging channel
        /// without re-decoding the audio.
        transcript: Option<String>,
    },

    /// The harness's reply to a previous `ToolUse` call.
    ///
    /// Both `tool_use_id` and `name` are carried because providers
    /// disagree on which one keys correlation: Anthropic / OpenAI /
    /// Bedrock use the id (`tool_use_id` / `tool_call_id` /
    /// `toolUseId`), while Gemini's `functionResponse` keys by
    /// `name`. Carrying both keeps the IR provider-neutral —
    /// codecs use whichever their wire format requires without
    /// needing the agent harness to know.
    ToolResult {
        /// The originating `ToolUse::id`.
        tool_use_id: String,
        /// The originating `ToolUse::name`. Required for Gemini's
        /// `functionResponse` wire shape; ignored by codecs that
        /// correlate purely by id.
        name: String,
        /// Result payload — either a string or structured data.
        content: ToolResultContent,
        /// True if the tool reported a failure.
        #[serde(default)]
        is_error: bool,
        /// Per-block cache directive. Tool result blocks
        /// often carry the heaviest payloads; caching them across
        /// turns is the canonical RAG-cache pattern.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
}

impl ContentPart {
    /// Build a text part from anything string-like.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text {
            text: text.into(),
            cache_control: None,
        }
    }

    /// Build an image part from a media source.
    #[must_use]
    pub fn image(source: MediaSource) -> Self {
        Self::Image {
            source,
            cache_control: None,
        }
    }

    /// Build an audio part from a media source.
    #[must_use]
    pub fn audio(source: MediaSource) -> Self {
        Self::Audio {
            source,
            cache_control: None,
        }
    }

    /// Build a video part from a media source.
    #[must_use]
    pub fn video(source: MediaSource) -> Self {
        Self::Video {
            source,
            cache_control: None,
        }
    }

    /// Build a document part from a media source.
    #[must_use]
    pub fn document(source: MediaSource, name: Option<String>) -> Self {
        Self::Document {
            source,
            name,
            cache_control: None,
        }
    }

    /// Build a thinking part with no signature.
    #[must_use]
    pub fn thinking(text: impl Into<String>) -> Self {
        Self::Thinking {
            text: text.into(),
            signature: None,
            cache_control: None,
        }
    }

    /// Borrow the cache directive on this part, when any is set.
    /// Returns `None` for assistant-output variants (`ToolUse`,
    /// `ImageOutput`, `AudioOutput`) — they are produced fresh per
    /// turn and have nothing to cache.
    #[must_use]
    pub const fn cache_control(&self) -> Option<&CacheControl> {
        match self {
            Self::Text { cache_control, .. }
            | Self::Image { cache_control, .. }
            | Self::Audio { cache_control, .. }
            | Self::Video { cache_control, .. }
            | Self::Document { cache_control, .. }
            | Self::Thinking { cache_control, .. }
            | Self::Citation { cache_control, .. }
            | Self::ToolResult { cache_control, .. } => cache_control.as_ref(),
            Self::ToolUse { .. } | Self::ImageOutput { .. } | Self::AudioOutput { .. } => None,
        }
    }

    /// Attach (or clear) a cache directive on this part. Returns the
    /// value back so callers can chain. No-op on `ToolUse` (model
    /// output never carries caching) — the directive is silently
    /// dropped.
    #[must_use]
    pub fn with_cache_control(self, cache: CacheControl) -> Self {
        match self {
            Self::Text { text, .. } => Self::Text {
                text,
                cache_control: Some(cache),
            },
            Self::Image { source, .. } => Self::Image {
                source,
                cache_control: Some(cache),
            },
            Self::Audio { source, .. } => Self::Audio {
                source,
                cache_control: Some(cache),
            },
            Self::Video { source, .. } => Self::Video {
                source,
                cache_control: Some(cache),
            },
            Self::Document { source, name, .. } => Self::Document {
                source,
                name,
                cache_control: Some(cache),
            },
            Self::Thinking {
                text, signature, ..
            } => Self::Thinking {
                text,
                signature,
                cache_control: Some(cache),
            },
            Self::Citation {
                snippet, source, ..
            } => Self::Citation {
                snippet,
                source,
                cache_control: Some(cache),
            },
            Self::ToolResult {
                tool_use_id,
                name,
                content,
                is_error,
                ..
            } => Self::ToolResult {
                tool_use_id,
                name,
                content,
                is_error,
                cache_control: Some(cache),
            },
            other
            @ (Self::ToolUse { .. } | Self::ImageOutput { .. } | Self::AudioOutput { .. }) => other,
        }
    }
}

/// Payload of a [`ContentPart::ToolResult`].
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
#[non_exhaustive]
pub enum ToolResultContent {
    /// Plain text result — the most common shape.
    Text(String),
    /// Structured JSON result. Codecs that lack structured-result support
    /// stringify and emit a `LossyEncode` warning.
    Json(serde_json::Value),
}
