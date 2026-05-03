//! `MediaSource` and `CitationSource` — provenance for multimodal and
//! citation [`ContentPart`](crate::ir::ContentPart) variants.
//!
//! `MediaSource` is shared across every modality (`Image / Audio / Video /
//! Document`) so codecs can route on a single shape regardless of media
//! type. `CitationSource` is the lean common subset every vendor returns
//! for grounding outputs — vendor-specific positioning stays out of IR
//! per the 2-codec rule (ADR-0024 §5).

use serde::{Deserialize, Serialize};

/// How a media payload is delivered to the model.
///
/// Every codec routes on the variant alone, so adding a new modality
/// ([`ContentPart::Audio`](crate::ir::ContentPart::Audio),
/// [`ContentPart::Video`](crate::ir::ContentPart::Video) etc.) reuses the
/// same source shape — no parallel `*Source` enums.
///
/// `media_type` is required on [`MediaSource::Base64`] (no other context
/// to infer from) and optional on [`MediaSource::Url`] /
/// [`MediaSource::FileId`] where URL extension or vendor metadata
/// typically supplies it.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum MediaSource {
    /// Hosted at an `https://` URL the vendor can fetch directly.
    Url {
        /// HTTPS URL.
        url: String,
        /// IANA media type (e.g. `image/png`, `audio/mpeg`). Optional —
        /// the URL extension or HTTP `Content-Type` header normally
        /// suffices for the vendor.
        media_type: Option<String>,
    },
    /// Inlined as base64 bytes with a media type tag.
    Base64 {
        /// IANA media type — required.
        media_type: String,
        /// Base64-encoded payload bytes.
        data: String,
    },
    /// Already uploaded via the vendor's Files API.
    ///
    /// Covers OpenAI Files (`file-…` IDs), Gemini File API (`files/…` IDs),
    /// Anthropic file inputs.
    FileId {
        /// Vendor-assigned file identifier.
        id: String,
        /// IANA media type — optional (vendor metadata typically supplies).
        media_type: Option<String>,
    },
}

impl MediaSource {
    /// Convenience constructor for a base64-inlined source.
    #[must_use]
    pub fn base64(media_type: impl Into<String>, data: impl Into<String>) -> Self {
        Self::Base64 {
            media_type: media_type.into(),
            data: data.into(),
        }
    }

    /// Convenience constructor for a URL source.
    #[must_use]
    pub fn url(url: impl Into<String>) -> Self {
        Self::Url {
            url: url.into(),
            media_type: None,
        }
    }

    /// Convenience constructor for a vendor file-id source.
    #[must_use]
    pub fn file_id(id: impl Into<String>) -> Self {
        Self::FileId {
            id: id.into(),
            media_type: None,
        }
    }

    /// Borrow the media type if known.
    #[must_use]
    pub fn media_type(&self) -> Option<&str> {
        match self {
            Self::Url { media_type, .. } | Self::FileId { media_type, .. } => media_type.as_deref(),
            Self::Base64 { media_type, .. } => Some(media_type),
        }
    }
}

/// Provenance for a [`ContentPart::Citation`](crate::ir::ContentPart::Citation).
///
/// Two-variant lean union covering the common subset every vendor returns
/// for grounding outputs. Vendor-specific positioning (start/end byte
/// offsets, chunk indices) is *not* modeled — codecs that receive offsets
/// emit a [`ModelWarning::LossyEncode`](crate::ir::ModelWarning::LossyEncode).
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum CitationSource {
    /// External web reference — covers Anthropic web-search citations,
    /// OpenAI URL annotations, Gemini grounding-chunk URIs.
    Url {
        /// Cited URL.
        url: String,
        /// Document or page title (vendor-supplied if known).
        title: Option<String>,
    },
    /// Reference to a [`ContentPart::Document`](crate::ir::ContentPart::Document)
    /// supplied earlier in the request.
    Document {
        /// Index into the request's `Document` content blocks (0-based,
        /// counted across the whole conversation).
        document_index: u32,
        /// Document title (vendor-supplied if known).
        title: Option<String>,
    },
}
