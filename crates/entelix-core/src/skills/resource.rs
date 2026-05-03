//! `SkillResource` trait + `SkillResourceContent` — on-demand T3
//! resources accessed via the third-tier tool.

use async_trait::async_trait;

use crate::context::ExecutionContext;
use crate::error::Result;

/// A lazy resource attached to a [`LoadedSkill`](crate::skills::LoadedSkill).
///
/// The trait is async and `Send + Sync` so backends can range freely
/// (in-memory, sandbox-internal file, HTTP fetch, MCP `resources/read`).
/// The bytes are read only when the model invokes the resource-read
/// tool — registries hold `Arc<dyn SkillResource>` handles, never the
/// resolved content.
#[async_trait]
pub trait SkillResource: Send + Sync + std::fmt::Debug {
    /// Read the resource. Implementations are expected to be cheap on
    /// repeated reads (cache internally if appropriate); the runtime
    /// makes no caching assumption.
    async fn read(&self, ctx: &ExecutionContext) -> Result<SkillResourceContent>;
}

/// Resolved resource payload.
///
/// Text resources are returned in full to the LLM; binary resources
/// surface as metadata only at the tool-result boundary
/// ([`crate::skills`] §"Three built-in tools" in ADR-0027). Embedding
/// raw bytes in conversation context would consume tens of thousands
/// of tokens for medium-sized assets — the metadata-only LLM-facing
/// shape preserves the cost discipline progressive disclosure exists
/// to enforce.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum SkillResourceContent {
    /// UTF-8 text payload — the standard shape for instructions,
    /// reference docs, schema files, examples.
    Text(String),
    /// Binary payload (image, PDF, archive). The host application
    /// handles uploads / vendor-side ingestion; the LLM sees only
    /// metadata.
    Binary {
        /// IANA media type.
        mime_type: String,
        /// Raw bytes.
        bytes: Vec<u8>,
    },
}

impl SkillResourceContent {
    /// Borrow the text body, or `None` if the content is binary.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(t) => Some(t),
            Self::Binary { .. } => None,
        }
    }

    /// Whether the payload is binary.
    #[must_use]
    pub const fn is_binary(&self) -> bool {
        matches!(self, Self::Binary { .. })
    }

    /// Byte length of the payload (UTF-8 byte count for text).
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::Text(t) => t.len(),
            Self::Binary { bytes, .. } => bytes.len(),
        }
    }
}
