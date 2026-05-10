//! `ContentPart` — a single block within a [`Message`](crate::ir::Message).
//!
//! Provider-neutral. Vendors that lack support for a variant cause the
//! codec to emit a [`ModelWarning::LossyEncode`](crate::ir::ModelWarning)
//! rather than failing silently (invariant 6).

use serde::{Deserialize, Serialize};

use crate::ir::cache::CacheControl;
use crate::ir::provider_echo::ProviderEchoSnapshot;
use crate::ir::source::{CitationSource, MediaSource};

/// One block of content inside a [`Message`](crate::ir::Message).
///
/// The enum is `#[non_exhaustive]` so future variants don't break user
/// `match` arms. New modalities or capability blocks land here as
/// additional variants — codecs reach 100% IR coverage by either
/// emitting native wire shape or a `LossyEncode` warning for each.
///
/// Every input-side variant carries an
/// `Option<CacheControl>` field — operators mark a
/// block as cached and codecs that support per-block caching
/// (Anthropic, Bedrock-on-Anthropic) emit the directive natively.
/// Other codecs emit `LossyEncode`. The `ToolUse` variant — the
/// assistant's outbound call — does not carry caching: the model
/// emits it, there is nothing to cache.
///
/// Every variant also carries
/// `provider_echoes: Vec<ProviderEchoSnapshot>` — vendor-keyed opaque
/// round-trip tokens this part must echo back on the next turn
/// (Gemini 3.x `thought_signature`, Anthropic `signature`, OpenAI
/// Responses `encrypted_content`, …). Defaults to empty. Codecs
/// only read / write entries matching their own `Codec::name`. See
/// [`ProviderEchoSnapshot`] for the cross-vendor design.
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
        /// Vendor opaque round-trip tokens (Gemini emits
        /// `thought_signature` on `text` parts in reasoning turns).
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        provider_echoes: Vec<ProviderEchoSnapshot>,
    },

    /// An image input.
    Image {
        /// Where the image bytes live.
        source: MediaSource,
        /// Per-block cache directive.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
        /// Vendor opaque round-trip tokens.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        provider_echoes: Vec<ProviderEchoSnapshot>,
    },

    /// An audio input.
    Audio {
        /// Where the audio bytes live.
        source: MediaSource,
        /// Per-block cache directive.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
        /// Vendor opaque round-trip tokens.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        provider_echoes: Vec<ProviderEchoSnapshot>,
    },

    /// A video input.
    Video {
        /// Where the video bytes live.
        source: MediaSource,
        /// Per-block cache directive.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
        /// Vendor opaque round-trip tokens.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        provider_echoes: Vec<ProviderEchoSnapshot>,
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
        /// Vendor opaque round-trip tokens.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        provider_echoes: Vec<ProviderEchoSnapshot>,
    },

    /// A reasoning / extended-thinking block produced by the assistant
    /// before the user-facing reply.
    ///
    /// Surfaced as its own variant (rather than mixed with `Text`) so
    /// recipes can show / hide / cache reasoning independently. Order
    /// relative to `Text` parts is preserved — vendors that rely on
    /// chain-of-thought integrity (Anthropic thinking, Gemini 3.x
    /// `thought_signature`) require the original block order on
    /// follow-up turns.
    ///
    /// Vendor opaque tokens (Anthropic `signature`, Gemini
    /// `thought_signature`, OpenAI Responses reasoning-item
    /// `encrypted_content`) ride on `provider_echoes`.
    Thinking {
        /// The reasoning text.
        text: String,
        /// Per-block cache directive.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
        /// Vendor opaque round-trip tokens.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        provider_echoes: Vec<ProviderEchoSnapshot>,
    },

    /// A reasoning block the safety system flagged for redaction.
    /// Carries no harness-readable text — the entire block is an
    /// opaque round-trip artifact preserved in `provider_echoes`.
    ///
    /// Emitted by Anthropic Claude 3.7 Sonnet only; Claude 4.x and
    /// later do not produce this variant. Codecs that don't recognise
    /// it on encode emit `LossyEncode` (invariant 6 — the prior
    /// silent-drop is replaced by a typed channel).
    RedactedThinking {
        /// Vendor opaque round-trip tokens. Anthropic emits
        /// `{ "data": "<base64>" }` here under provider key
        /// `"anthropic-messages"`.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        provider_echoes: Vec<ProviderEchoSnapshot>,
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
        /// Vendor opaque round-trip tokens.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        provider_echoes: Vec<ProviderEchoSnapshot>,
    },

    /// A tool call emitted by the assistant. The harness dispatches it.
    /// Tool calls are model output — they do not carry a cache
    /// directive (the model emits each call afresh).
    ///
    /// Vendor opaque tokens (Gemini 3.x `thought_signature` on
    /// `functionCall` parts, OpenAI Responses `function_call.id`)
    /// ride on `provider_echoes`. Missing the Gemini token on the
    /// first `functionCall` of a step yields HTTP 400 on the next
    /// turn — codecs MUST round-trip it.
    ToolUse {
        /// Stable ID matched against the corresponding `ToolResult`.
        id: String,
        /// Tool name to dispatch (must exist in the active `ToolRegistry`).
        name: String,
        /// JSON arguments for the tool — must validate against the
        /// tool's `input_schema` before dispatch.
        input: serde_json::Value,
        /// Vendor opaque round-trip tokens.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        provider_echoes: Vec<ProviderEchoSnapshot>,
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
        /// Vendor opaque round-trip tokens.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        provider_echoes: Vec<ProviderEchoSnapshot>,
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
        /// Vendor opaque round-trip tokens.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        provider_echoes: Vec<ProviderEchoSnapshot>,
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
        /// Vendor opaque round-trip tokens.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        provider_echoes: Vec<ProviderEchoSnapshot>,
    },
}

impl ContentPart {
    /// Build a text part from anything string-like.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text {
            text: text.into(),
            cache_control: None,
            provider_echoes: Vec::new(),
        }
    }

    /// Build an image part from a media source.
    #[must_use]
    pub fn image(source: MediaSource) -> Self {
        Self::Image {
            source,
            cache_control: None,
            provider_echoes: Vec::new(),
        }
    }

    /// Build an audio part from a media source.
    #[must_use]
    pub fn audio(source: MediaSource) -> Self {
        Self::Audio {
            source,
            cache_control: None,
            provider_echoes: Vec::new(),
        }
    }

    /// Build a video part from a media source.
    #[must_use]
    pub fn video(source: MediaSource) -> Self {
        Self::Video {
            source,
            cache_control: None,
            provider_echoes: Vec::new(),
        }
    }

    /// Build a document part from a media source.
    #[must_use]
    pub fn document(source: MediaSource, name: Option<String>) -> Self {
        Self::Document {
            source,
            name,
            cache_control: None,
            provider_echoes: Vec::new(),
        }
    }

    /// Build a thinking part with no opaque tokens. Codecs attach
    /// vendor opaque tokens via [`Self::with_provider_echo`].
    #[must_use]
    pub fn thinking(text: impl Into<String>) -> Self {
        Self::Thinking {
            text: text.into(),
            cache_control: None,
            provider_echoes: Vec::new(),
        }
    }

    /// Build a redacted-thinking part. Anthropic Claude 3.7 Sonnet
    /// emits these for safety-flagged reasoning; the codec attaches
    /// the opaque `data` payload via [`Self::with_provider_echo`].
    #[must_use]
    pub fn redacted_thinking() -> Self {
        Self::RedactedThinking {
            provider_echoes: Vec::new(),
        }
    }

    /// Borrow the cache directive on this part, when any is set.
    /// Returns `None` for assistant-output variants (`ToolUse`,
    /// `ImageOutput`, `AudioOutput`, `RedactedThinking`) — they are
    /// produced fresh per turn and have nothing to cache.
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
            Self::ToolUse { .. }
            | Self::ImageOutput { .. }
            | Self::AudioOutput { .. }
            | Self::RedactedThinking { .. } => None,
        }
    }

    /// Borrow this part's vendor opaque round-trip tokens. Codecs use
    /// this to recover their own blob (matched by `Codec::name`) on
    /// the encode side.
    #[must_use]
    pub fn provider_echoes(&self) -> &[ProviderEchoSnapshot] {
        match self {
            Self::Text {
                provider_echoes, ..
            }
            | Self::Image {
                provider_echoes, ..
            }
            | Self::Audio {
                provider_echoes, ..
            }
            | Self::Video {
                provider_echoes, ..
            }
            | Self::Document {
                provider_echoes, ..
            }
            | Self::Thinking {
                provider_echoes, ..
            }
            | Self::RedactedThinking { provider_echoes }
            | Self::Citation {
                provider_echoes, ..
            }
            | Self::ToolUse {
                provider_echoes, ..
            }
            | Self::ImageOutput {
                provider_echoes, ..
            }
            | Self::AudioOutput {
                provider_echoes, ..
            }
            | Self::ToolResult {
                provider_echoes, ..
            } => provider_echoes,
        }
    }

    /// Attach (or clear) a cache directive on this part. Returns the
    /// value back so callers can chain. No-op on `ToolUse` /
    /// `ImageOutput` / `AudioOutput` / `RedactedThinking` (model
    /// output never carries caching) — the directive is silently
    /// dropped.
    #[must_use]
    pub fn with_cache_control(self, cache: CacheControl) -> Self {
        match self {
            Self::Text {
                text,
                provider_echoes,
                ..
            } => Self::Text {
                text,
                cache_control: Some(cache),
                provider_echoes,
            },
            Self::Image {
                source,
                provider_echoes,
                ..
            } => Self::Image {
                source,
                cache_control: Some(cache),
                provider_echoes,
            },
            Self::Audio {
                source,
                provider_echoes,
                ..
            } => Self::Audio {
                source,
                cache_control: Some(cache),
                provider_echoes,
            },
            Self::Video {
                source,
                provider_echoes,
                ..
            } => Self::Video {
                source,
                cache_control: Some(cache),
                provider_echoes,
            },
            Self::Document {
                source,
                name,
                provider_echoes,
                ..
            } => Self::Document {
                source,
                name,
                cache_control: Some(cache),
                provider_echoes,
            },
            Self::Thinking {
                text,
                provider_echoes,
                ..
            } => Self::Thinking {
                text,
                cache_control: Some(cache),
                provider_echoes,
            },
            Self::Citation {
                snippet,
                source,
                provider_echoes,
                ..
            } => Self::Citation {
                snippet,
                source,
                cache_control: Some(cache),
                provider_echoes,
            },
            Self::ToolResult {
                tool_use_id,
                name,
                content,
                is_error,
                provider_echoes,
                ..
            } => Self::ToolResult {
                tool_use_id,
                name,
                content,
                is_error,
                cache_control: Some(cache),
                provider_echoes,
            },
            other @ (Self::ToolUse { .. }
            | Self::ImageOutput { .. }
            | Self::AudioOutput { .. }
            | Self::RedactedThinking { .. }) => other,
        }
    }

    /// Append a vendor opaque round-trip token to this part. Codecs
    /// call this on the decode side after extracting the wire-shape
    /// signature / encrypted_content / data field. Returns the value
    /// back so callers can chain.
    #[must_use]
    pub fn with_provider_echo(mut self, echo: ProviderEchoSnapshot) -> Self {
        match &mut self {
            Self::Text {
                provider_echoes, ..
            }
            | Self::Image {
                provider_echoes, ..
            }
            | Self::Audio {
                provider_echoes, ..
            }
            | Self::Video {
                provider_echoes, ..
            }
            | Self::Document {
                provider_echoes, ..
            }
            | Self::Thinking {
                provider_echoes, ..
            }
            | Self::RedactedThinking { provider_echoes }
            | Self::Citation {
                provider_echoes, ..
            }
            | Self::ToolUse {
                provider_echoes, ..
            }
            | Self::ImageOutput {
                provider_echoes, ..
            }
            | Self::AudioOutput {
                provider_echoes, ..
            }
            | Self::ToolResult {
                provider_echoes, ..
            } => provider_echoes.push(echo),
        }
        self
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
