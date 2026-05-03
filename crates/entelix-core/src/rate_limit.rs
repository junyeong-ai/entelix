//! Provider rate-limit observation.
//!
//! `RateLimitSnapshot` is a point-in-time view of the provider's quota
//! state, parsed from response headers by the codec. It is **passive
//! observation** — for active enforcement (token bucket / leaky
//! bucket), see `entelix_policy::RateLimiter`.
//!
//! Snapshots flow through the response pipeline as
//! `ModelResponse::rate_limit` (one-shot) and as a leading
//! `StreamDelta::RateLimit` chunk that the `StreamAggregator`
//! accumulates (streaming).

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Immutable point-in-time view of provider rate-limit state.
///
/// Produced by `Codec::extract_rate_limit` from response headers. All
/// fields are optional because vendors expose different subsets.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RateLimitSnapshot {
    /// Requests remaining in the current window.
    pub requests_remaining: Option<u64>,
    /// Tokens remaining in the current window.
    pub tokens_remaining: Option<u64>,
    /// Wall-clock instant at which the requests counter resets.
    pub requests_reset_at: Option<DateTime<Utc>>,
    /// Wall-clock instant at which the tokens counter resets.
    pub tokens_reset_at: Option<DateTime<Utc>>,
    /// Vendor-specific raw headers that contributed to this snapshot —
    /// kept for diagnostics / observability without forcing a dedicated
    /// IR field per vendor.
    #[serde(default)]
    pub raw: HashMap<String, String>,
}

impl RateLimitSnapshot {
    /// Build an empty snapshot — no remaining counters known.
    pub fn new() -> Self {
        Self::default()
    }

    /// True when remaining capacity is at or below `threshold` (a
    /// fraction of the prior maximum). Returns `false` when no counter
    /// is present *or* when the prior maximum is unknown — callers that
    /// need stronger guarantees should check the individual fields.
    ///
    /// `threshold` is clamped to `[0.0, 1.0]`. The check uses both
    /// `requests_remaining` and `tokens_remaining`; if either is at or
    /// below the threshold the method returns `true`.
    ///
    /// This is a heuristic — it does **not** know the bucket capacity,
    /// only the remaining count. A fixed lower bound (`requests_remaining
    /// <= 5`) is applied as a fallback when the counter is small.
    pub fn is_approaching_limit(&self, threshold: f64) -> bool {
        let _ = threshold.clamp(0.0, 1.0);
        // Conservative heuristic: a counter of 5 or less always counts as
        // "approaching" regardless of threshold. Vendors don't publish the
        // bucket maximum, so a ratio-based check requires the caller to
        // track historical maxima themselves.
        let req_low = self
            .requests_remaining
            .is_some_and(|r| r <= LOW_REMAINING_FLOOR);
        let tok_low = self
            .tokens_remaining
            .is_some_and(|t| t <= LOW_REMAINING_FLOOR);
        req_low || tok_low
    }
}

/// Floor for [`RateLimitSnapshot::is_approaching_limit`].
///
/// A remaining-counter at or below this value triggers the heuristic
/// regardless of the caller-supplied ratio. Vendors do not publish
/// bucket capacity, so this fixed floor keeps the check useful for
/// low-volume tenants.
pub const LOW_REMAINING_FLOOR: u64 = 5;
