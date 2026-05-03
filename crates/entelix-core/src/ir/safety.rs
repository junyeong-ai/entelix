//! `SafetyRating` — category-level moderation scores returned by vendors
//! that expose per-response safety reporting (Gemini today; OpenAI /
//! Anthropic surface refusals via `StopReason::Refusal`).

use serde::{Deserialize, Serialize};

/// One category × level pair carried on
/// [`Usage::safety_ratings`](crate::ir::Usage).
///
/// Codecs leave the list empty when the vendor doesn't report safety per
/// response. Refusal stop reasons stay on `StopReason::Refusal` regardless
/// — `safety_ratings` is for the dimensional, non-blocking score shape.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SafetyRating {
    /// Moderation category.
    pub category: SafetyCategory,
    /// Score level on the four-bucket scale common to vendor APIs.
    pub level: SafetyLevel,
}

/// Moderation category. Vendors that emit categories outside this set
/// land in [`SafetyCategory::Other`] preserving the raw vendor name.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(tag = "kind", content = "raw", rename_all = "snake_case")]
pub enum SafetyCategory {
    /// Harassment / personal attack content.
    Harassment,
    /// Hate-speech content.
    HateSpeech,
    /// Sexually explicit content.
    SexuallyExplicit,
    /// Dangerous-content (self-harm, violence, illicit acts).
    DangerousContent,
    /// Vendor-specific category — `raw` is the vendor's own name.
    Other(String),
}

/// Four-bucket score scale common to Gemini-family safety APIs.
///
/// Codecs that surface a numeric score from a vendor map onto these
/// buckets via the standard thresholds (`< 0.25` → `Negligible`,
/// `< 0.5` → `Low`, `< 0.75` → `Medium`, otherwise `High`); the original
/// number, if needed for analytics, surfaces via observability spans, not
/// the IR.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum SafetyLevel {
    /// Below the lowest concern threshold.
    Negligible,
    /// Low concern.
    Low,
    /// Medium concern.
    Medium,
    /// High concern (typically the threshold above which vendors block).
    High,
}
