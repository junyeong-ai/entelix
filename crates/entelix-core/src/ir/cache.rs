//! Prompt-cache control — vendor-agnostic IR knob.
//!
//! Per: a vendor knob enters IR only when 2+ shipping
//! codecs natively support it. Cache control qualifies — Anthropic
//! Messages and Bedrock Converse (for Claude models) both expose it
//! natively. Codecs that don't support it emit
//! [`crate::ir::ModelWarning::LossyEncode`] — silent
//! drop is forbidden.
//!
//! ## TTL choices
//!
//! `CacheTtl` is `#[non_exhaustive]` and currently exposes the two
//! TTLs Anthropic publishes (5-minute default + 1-hour premium).
//! New TTLs are added by extending the enum without breaking callers.
//!
//! ## Wire format (Anthropic)
//!
//! Anthropic's `cache_control` block is always
//! `{"type": "ephemeral"}` — `type` never carries the TTL string.
//! The TTL rides in a sibling `ttl` field, omitted for the 5-minute
//! default and present for premium tiers (`"1h"`). Codecs use
//! [`CacheTtl::wire_ttl_field`] to render the optional sibling.

use serde::{Deserialize, Serialize};

/// Time-to-live tier for a cached prompt block.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum CacheTtl {
    /// Five-minute default TTL — what Anthropic emits when
    /// `cache_control: {type: "ephemeral"}` carries no `ttl` field.
    /// Use this for hot-path caches (most agents).
    #[default]
    FiveMinutes,
    /// One-hour TTL (Anthropic premium-cache tier — wire `ttl: "1h"`).
    OneHour,
}

impl CacheTtl {
    /// Wire string for the optional `ttl` sibling field — `None`
    /// when the TTL is the vendor default (5 minutes), `Some("1h")`
    /// for premium. The `type` field is always `"ephemeral"` per
    /// Anthropic's contract; only `ttl` varies.
    #[must_use]
    pub const fn wire_ttl_field(self) -> Option<&'static str> {
        match self {
            Self::FiveMinutes => None,
            Self::OneHour => Some("1h"),
        }
    }
}

/// Cache directive attached to a [`SystemBlock`](crate::ir::SystemBlock).
///
/// Currently a single field (`ttl`); kept as a struct rather than a
/// bare `CacheTtl` so codec-relevant knobs (`type`, breakpoint
/// markers) can be added later without breaking the IR shape.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CacheControl {
    /// Cache lifetime tier.
    pub ttl: CacheTtl,
}

impl CacheControl {
    /// Five-minute TTL — Anthropic's default cache tier. The most
    /// common choice; renders as a bare `cache_control:
    /// {type: "ephemeral"}` (no `ttl` field).
    #[must_use]
    pub const fn five_minutes() -> Self {
        Self {
            ttl: CacheTtl::FiveMinutes,
        }
    }

    /// One-hour TTL — Anthropic premium-cache tier. Renders as
    /// `cache_control: {type: "ephemeral", ttl: "1h"}`.
    #[must_use]
    pub const fn one_hour() -> Self {
        Self {
            ttl: CacheTtl::OneHour,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn wire_ttl_field_is_stable() {
        assert_eq!(CacheTtl::FiveMinutes.wire_ttl_field(), None);
        assert_eq!(CacheTtl::OneHour.wire_ttl_field(), Some("1h"));
    }

    #[test]
    fn const_constructors_match_variants() {
        assert_eq!(CacheControl::five_minutes().ttl, CacheTtl::FiveMinutes);
        assert_eq!(CacheControl::one_hour().ttl, CacheTtl::OneHour);
    }

    #[test]
    fn cache_control_round_trips_via_serde() {
        let cc = CacheControl::one_hour();
        let json = serde_json::to_string(&cc).unwrap();
        let back: CacheControl = serde_json::from_str(&json).unwrap();
        assert_eq!(cc, back);
    }
}
