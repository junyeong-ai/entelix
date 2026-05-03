//! [`ToolEffect`] — declarative side-effect classification + retry hints.
//!
//! These metadata fields ride alongside `Tool::name` / `Tool::description`
//! and let the runtime reason about a tool *before* it is dispatched:
//!
//! - `ToolEffect` partitions the tool surface into `ReadOnly` /
//!   `Mutating` / `Destructive` so [`Approver`](crate) defaults
//!   (and optionally `PolicyLayer` guardrails) can require human
//!   confirmation for irreversible operations.
//! - [`RetryHint`] tells [`tower::retry::RetryLayer`] (and similar
//!   middleware) how many times a tool may be retried, and over what
//!   spread. Defaults are conservative — the runtime never retries
//!   a tool that does not opt in.
//! - [`Tool::idempotent`] is the cheap binary version of
//!   `RetryHint::is_some()` for callers that only care about
//!   "safe to retry on transport hiccup".
//!
//! All fields surface as `gen_ai.tool.*` OTel attributes so dashboards
//! can partition spend / latency / error rate by side-effect class.

use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Side-effect classification of a tool — surfaces both to the
/// runtime (Approver defaults, retry policy) and to the LLM
/// (rendered in the tool description so the model can reason about
/// safety on its own).
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ToolEffect {
    /// Pure read — no side effects on external state. Safe to call
    /// in parallel, retry on transport hiccups, cache aggressively.
    /// Examples: search, fetch, calculator.
    #[default]
    ReadOnly,
    /// Changes external state but the change is recoverable
    /// (overwrite, undo, idempotent overwrite). Retry-safe with a
    /// reasonable backoff. Examples: write a file, set a config
    /// value, update a record.
    Mutating,
    /// Irreversible — once it runs the operator cannot undo. Default
    /// `Approver` policy MAY require human confirmation. Retry is
    /// off by default. Examples: send an email, post a payment,
    /// delete a row, run an `rm -rf`.
    Destructive,
}

impl ToolEffect {
    /// Stable wire string used by OTel attributes
    /// (`gen_ai.tool.effect`) and event-log records.
    #[must_use]
    pub const fn as_wire(self) -> &'static str {
        match self {
            Self::ReadOnly => "read_only",
            Self::Mutating => "mutating",
            Self::Destructive => "destructive",
        }
    }
}

impl std::fmt::Display for ToolEffect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_wire())
    }
}

/// How a `Tool` opts into runtime retry. `None` (the default) means
/// the tool is *not* retried by middleware.
///
/// Tools that expose `RetryHint` should be idempotent for the
/// `attempts` count they declare — middleware will treat the hint
/// as authoritative.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RetryHint {
    /// Maximum number of attempts (including the first call).
    /// Must be `>= 1`.
    pub max_attempts: u32,
    /// Initial backoff between attempts. Middleware applies
    /// exponential growth + jitter on top of this baseline.
    pub initial_backoff: Duration,
}

impl RetryHint {
    /// Conservative default for a transport-bound idempotent tool —
    /// 3 attempts, 200 ms initial backoff.
    #[must_use]
    pub const fn idempotent_transport() -> Self {
        Self {
            max_attempts: 3,
            initial_backoff: Duration::from_millis(200),
        }
    }

    /// Construct a hint with custom values. Panics on
    /// `max_attempts == 0` because a zero retry budget is a config
    /// bug, not a runtime condition the middleware should silently
    /// paper over.
    #[must_use]
    pub const fn new(max_attempts: u32, initial_backoff: Duration) -> Self {
        assert!(max_attempts >= 1, "RetryHint::max_attempts must be >= 1");
        Self {
            max_attempts,
            initial_backoff,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn effect_default_is_read_only() {
        assert_eq!(ToolEffect::default(), ToolEffect::ReadOnly);
    }

    #[test]
    fn effect_wire_strings_are_stable() {
        assert_eq!(ToolEffect::ReadOnly.as_wire(), "read_only");
        assert_eq!(ToolEffect::Mutating.as_wire(), "mutating");
        assert_eq!(ToolEffect::Destructive.as_wire(), "destructive");
    }

    #[test]
    fn effect_serde_round_trip() {
        let s = serde_json::to_string(&ToolEffect::Destructive).unwrap();
        assert_eq!(s, "\"destructive\"");
        let back: ToolEffect = serde_json::from_str(&s).unwrap();
        assert_eq!(back, ToolEffect::Destructive);
    }

    #[test]
    fn retry_hint_const_ctor_baseline() {
        let h = RetryHint::idempotent_transport();
        assert_eq!(h.max_attempts, 3);
        assert_eq!(h.initial_backoff, Duration::from_millis(200));
    }

    #[test]
    #[should_panic(expected = "RetryHint::max_attempts must be >= 1")]
    fn retry_hint_zero_attempts_panics() {
        let _ = RetryHint::new(0, Duration::from_millis(100));
    }
}
