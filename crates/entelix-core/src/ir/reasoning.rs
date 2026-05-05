//! Reasoning effort — cross-vendor IR for "how hard should the model
//! reason before replying".
//!
//! The four 2026-Q2 reasoning-capable model families (Anthropic
//! extended-thinking, OpenAI o-series, Gemini 2.5/3 thinking,
//! Bedrock Converse Anthropic-passthrough) expose this knob in
//! incompatible wire shapes:
//!
//! - **Anthropic** — `thinking: { type: "enabled" | "adaptive" |
//!   "disabled", budget_tokens: N }`. Opus 4.7 is adaptive-only —
//!   manual `budget_tokens` is rejected. Sonnet 4.6 / 4.5 / Haiku
//!   accept either `enabled` with explicit budget or `adaptive`.
//! - **OpenAI Responses** — `reasoning: { effort: "none" |
//!   "minimal" | "low" | "medium" | "high" | "xhigh" }`. Discrete
//!   bucket, no token budget.
//! - **Gemini 2.5** — `thinkingConfig: { thinkingBudget: N | -1 |
//!   0 }`. Token budget; `-1` is "auto", `0` disables (Flash only;
//!   Pro cannot disable).
//! - **Gemini 3** — `thinkingConfig: { thinkingLevel: "minimal" |
//!   "low" | "medium" | "high" }`. Discrete bucket only.
//! - **Bedrock Converse (Anthropic family)** — Anthropic passthrough
//!   via `additionalModelRequestFields`. Same constraints as
//!   Anthropic direct.
//!
//! [`ReasoningEffort`] is the cross-vendor enum every codec
//! translates onto its native wire shape. Translation is **lossy
//! by vendor design**, not by SDK design — the per-codec mapping
//! below documents exactly which variants fall through cleanly,
//! which approximate, and which emit `ModelWarning::LossyEncode`.
//!
//! ## Per-vendor mapping (encode-time)
//!
//! | `ReasoningEffort` | Anthropic                                    | OpenAI Responses        | Gemini 2.5 (Flash)        | Gemini 2.5 (Pro)          | Gemini 3                  |
//! |-------------------|----------------------------------------------|-------------------------|---------------------------|---------------------------|---------------------------|
//! | `Off`             | `{type:"disabled"}`                          | `effort:"none"`         | `thinkingBudget:0`        | LossyEncode → `512`       | LossyEncode → `"minimal"` |
//! | `Minimal`         | `{type:"adaptive", effort:"low"}` ¹          | `effort:"minimal"`      | `thinkingBudget:512` ²    | `thinkingBudget:512`      | `thinkingLevel:"minimal"` |
//! | `Low`             | `{type:"enabled", budget_tokens:1024}`       | `effort:"low"`          | `thinkingBudget:1024`     | `thinkingBudget:1024`     | `thinkingLevel:"low"`     |
//! | `Medium`          | `{type:"enabled", budget_tokens:4096}`       | `effort:"medium"`       | `thinkingBudget:8192`     | `thinkingBudget:8192`     | `thinkingLevel:"medium"`  |
//! | `High`            | `{type:"enabled", budget_tokens:16384}`      | `effort:"high"`         | `thinkingBudget:24576`    | `thinkingBudget:24576`    | `thinkingLevel:"high"`    |
//! | `Auto`            | `{type:"adaptive"}` (4.6+)                   | LossyEncode → `medium`  | `thinkingBudget:-1`       | `thinkingBudget:-1`       | LossyEncode → `"high"`    |
//! | `VendorSpecific`  | passthrough — see [`Self::VendorSpecific`]   | passthrough             | passthrough               | passthrough               | passthrough               |
//!
//! ¹ Anthropic Opus 4.7 emits `{type:"adaptive", effort:"low"}`
//!   (manual budget is rejected). Sonnet 4.6 / 4.5 / Haiku emit
//!   `{type:"adaptive", effort:"low"}` for parity with Opus 4.7.
//!
//! ² Gemini 2.5's `thinkingBudget` is a continuous integer; `512`
//!   is the closest discrete approximation to "minimal" and emits
//!   `LossyEncode` so operators see the snap.
//!
//! Every coercion site emits a typed
//! [`crate::ir::ModelWarning::LossyEncode`] so operators see the
//! information loss in observability — invariant 6.

use serde::{Deserialize, Serialize};

/// How hard the model should reason before producing the
/// user-facing reply, abstracted across vendors.
///
/// The variants are deliberately ordinal-but-not-numeric:
/// `Minimal < Low < Medium < High` is meaningful, but they are
/// discrete buckets each codec snaps to its native shape, not
/// token counts. Operators that need the exact Anthropic
/// `budget_tokens` value or the exact Gemini `thinkingBudget`
/// integer reach for [`Self::VendorSpecific`].
///
/// The default is [`Self::Medium`] — matches every vendor's
/// "reasonable balance" tier.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ReasoningEffort {
    /// Disable reasoning entirely. The model produces a direct
    /// reply without an internal reasoning pass.
    ///
    /// Wire mapping: Anthropic `{type:"disabled"}`, OpenAI
    /// `effort:"none"`, Gemini 2.5 Flash `thinkingBudget:0`.
    /// Gemini 2.5 Pro cannot disable thinking — codec emits
    /// `LossyEncode` and snaps to `Minimal`. Gemini 3 has no
    /// `Off` bucket — codec emits `LossyEncode` and snaps to
    /// `Minimal`.
    Off,
    /// The lightest reasoning bucket. Suitable for routing /
    /// classification / fast lookups.
    ///
    /// Wire mapping: Anthropic `{type:"adaptive", effort:"low"}`
    /// (LossyEncode — closest adaptive bucket), OpenAI
    /// `effort:"minimal"`, Gemini 2.5 `thinkingBudget:512`
    /// (LossyEncode on Pro — discrete approximation), Gemini 3
    /// `thinkingLevel:"minimal"`.
    Minimal,
    /// Light reasoning — multi-step but bounded.
    ///
    /// Wire mapping: Anthropic
    /// `{type:"enabled", budget_tokens:1024}`, OpenAI
    /// `effort:"low"`, Gemini 2.5 `thinkingBudget:1024`, Gemini 3
    /// `thinkingLevel:"low"`.
    Low,
    /// Vendor default — balanced reasoning for most production
    /// workflows.
    ///
    /// Wire mapping: Anthropic
    /// `{type:"enabled", budget_tokens:4096}`, OpenAI
    /// `effort:"medium"`, Gemini 2.5 `thinkingBudget:8192`,
    /// Gemini 3 `thinkingLevel:"medium"`.
    #[default]
    Medium,
    /// Deep reasoning — highest cost / latency, suitable for
    /// math / planning / multi-hop research.
    ///
    /// Wire mapping: Anthropic
    /// `{type:"enabled", budget_tokens:16384}`, OpenAI
    /// `effort:"high"`, Gemini 2.5 `thinkingBudget:24576`,
    /// Gemini 3 `thinkingLevel:"high"`.
    High,
    /// Let the vendor decide the reasoning budget per request
    /// based on prompt complexity.
    ///
    /// Wire mapping: Anthropic `{type:"adaptive"}` (4.6+ only —
    /// pre-4.6 models LossyEncode → `Medium`), Gemini 2.5
    /// `thinkingBudget:-1`. OpenAI Responses has no auto bucket
    /// — codec LossyEncode → `Medium`. Gemini 3 has no auto
    /// bucket — codec LossyEncode → `High`.
    Auto,
    /// Pass a vendor-specific raw value through to the wire
    /// shape, bypassing the cross-vendor mapping. Operators
    /// reach for this when they need the exact Anthropic
    /// `budget_tokens: 9000`, the exact Gemini `thinkingBudget:
    /// 16384`, or OpenAI's `effort:"xhigh"` (which has no
    /// cross-vendor analogue).
    ///
    /// The string is the vendor's literal wire representation:
    ///
    /// - Anthropic: numeric string parsed as `budget_tokens`
    ///   (e.g. `"9000"`) with `{type:"enabled"}`. Non-numeric
    ///   strings emit `LossyEncode` and fall through to
    ///   `Medium`. Opus 4.7 rejects this variant entirely
    ///   (model is adaptive-only).
    /// - OpenAI Responses: literal `effort` value (e.g.
    ///   `"xhigh"`).
    /// - Gemini 2.5: numeric string parsed as `thinkingBudget`
    ///   (e.g. `"6000"`).
    /// - Gemini 3: literal `thinkingLevel` value.
    /// - Bedrock Converse: Anthropic semantics.
    ///
    /// Codecs that do not understand the literal value emit
    /// `LossyEncode` and fall through to the vendor's nearest
    /// supported bucket.
    VendorSpecific(String),
}

impl ReasoningEffort {
    /// True when this is one of the discrete cross-vendor buckets
    /// (`Off` / `Minimal` / `Low` / `Medium` / `High` / `Auto`)
    /// — i.e. NOT [`Self::VendorSpecific`]. Codec encode paths
    /// branch on this to decide between the canonical mapping
    /// and the vendor passthrough.
    #[must_use]
    pub const fn is_canonical(&self) -> bool {
        !matches!(self, Self::VendorSpecific(_))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn default_is_medium() {
        assert_eq!(ReasoningEffort::default(), ReasoningEffort::Medium);
    }

    #[test]
    fn is_canonical_branches() {
        assert!(ReasoningEffort::Off.is_canonical());
        assert!(ReasoningEffort::Auto.is_canonical());
        assert!(!ReasoningEffort::VendorSpecific("xhigh".into()).is_canonical());
    }

    #[test]
    fn serde_round_trip_canonical() {
        for variant in [
            ReasoningEffort::Off,
            ReasoningEffort::Minimal,
            ReasoningEffort::Low,
            ReasoningEffort::Medium,
            ReasoningEffort::High,
            ReasoningEffort::Auto,
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            let back: ReasoningEffort = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, back);
        }
    }

    #[test]
    fn serde_round_trip_vendor_specific() {
        let v = ReasoningEffort::VendorSpecific("xhigh".into());
        let json = serde_json::to_string(&v).unwrap();
        // serde renames the variant to `vendor_specific` (snake_case).
        assert!(json.contains("vendor_specific"));
        assert!(json.contains("xhigh"));
        let back: ReasoningEffort = serde_json::from_str(&json).unwrap();
        assert_eq!(v, back);
    }
}
