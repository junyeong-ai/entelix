//! `ToolErrorKind` — tool-dispatch failure category derived from
//! [`crate::Error`] for observability and retry classification.
//!
//! Tool authors return `Result<Value, Error>` from `Tool::execute`;
//! the runtime classifies the error variant into one of these
//! seven categories so observability sinks (`AgentEvent::ToolError`),
//! retry middleware (`RetryToolLayer`), and recovery sinks all
//! reach the same cross-tool taxonomy.
//!
//! Mirrors [`crate::ProviderErrorKind`] in shape (typed enum
//! categorising failures) but operates at a higher level — provider
//! kinds describe transport mechanisms, tool kinds describe the
//! semantic outcome the operator (or the model) actually cares about.

use crate::error::Error;

/// Cross-tool failure category.
///
/// Derive from [`Error`] via [`Self::classify`]. Used for retry
/// middleware (`RetryToolLayer` retries [`Self::Transient`] /
/// [`Self::RateLimit`]), observability sinks (operators surface the
/// category in dashboards), and downstream recovery routing
/// (different categories trigger different operator responses —
/// page on `Auth`, alert on `Quota`, ignore `Validation` noise).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum ToolErrorKind {
    /// Network blip, transient 5xx, generic transport failure —
    /// safe to retry.
    Transient,
    /// Vendor signalled rate limiting (429 with `Retry-After` hint).
    /// Retryable after the configured cooldown.
    RateLimit,
    /// Vendor signalled exhausted quota / billing cap. Retry will
    /// not succeed until the quota resets or operator intervenes —
    /// surface to ops, do not retry automatically.
    Quota,
    /// Credential rejected (401 / 403 / [`Error::Auth`]). Retry will
    /// not succeed until credentials are rotated.
    Auth,
    /// Permanent vendor failure (4xx other than auth/rate/quota,
    /// 405, 410, 422 …). The same call will fail again.
    Permanent,
    /// Caller-side input rejected ([`Error::InvalidRequest`],
    /// [`Error::Serde`]) — the operator's payload does not match
    /// the tool contract. Retry is meaningless without changing the
    /// payload.
    Validation,
    /// Tool-internal bug or misconfiguration ([`Error::Config`], or
    /// any unclassified shape). Surface to ops; retry is not
    /// meaningful.
    Internal,
}

impl ToolErrorKind {
    /// Derive the category from an [`Error`].
    ///
    /// The mapping is intentionally exhaustive over the variants
    /// [`Error`] surfaces today — the `_` catch-all routes to
    /// [`Self::Internal`] so future variants stay observable until
    /// classified explicitly. Operational variants
    /// ([`Error::Cancelled`], [`Error::DeadlineExceeded`],
    /// [`Error::Interrupted`], [`Error::ModelRetry`]) flow through
    /// `Internal` because they are agent-runtime control signals,
    /// not tool failures — call sites that observe them should not
    /// reach this classifier in the first place.
    #[must_use]
    pub fn classify(error: &Error) -> Self {
        use crate::error::ProviderErrorKind;
        match error {
            Error::Provider {
                kind: ProviderErrorKind::Network | ProviderErrorKind::Tls | ProviderErrorKind::Dns,
                ..
            } => Self::Transient,
            Error::Provider {
                kind: ProviderErrorKind::Http(429),
                retry_after,
                ..
            } => {
                // Vendor distinguishes 429-with-Retry-After (transient
                // back-pressure) from 429-without (often quota
                // exhaustion). The hint presence is the cue.
                if retry_after.is_some() {
                    Self::RateLimit
                } else {
                    Self::Quota
                }
            }
            Error::Provider {
                kind: ProviderErrorKind::Http(status),
                ..
            } => {
                if *status == 401 || *status == 403 {
                    Self::Auth
                } else if (500..600).contains(status) || *status == 408 || *status == 425 {
                    Self::Transient
                } else {
                    Self::Permanent
                }
            }
            Error::Auth(_) => Self::Auth,
            Error::UsageLimitExceeded(_) => Self::Quota,
            Error::InvalidRequest(_) | Error::Serde(_) => Self::Validation,
            // Operational variants (Cancelled, DeadlineExceeded,
            // Interrupted, ModelRetry) and any future shape route
            // here together with Config — none of them are tool
            // failures the operator can act on per-category.
            _ => Self::Internal,
        }
    }

    /// Whether the runtime should attempt the tool call again.
    ///
    /// `Transient` and `RateLimit` are retryable; everything else
    /// is a surface-and-stop signal. `RetryToolLayer` consults this
    /// via the underlying `RetryClassifier` (which can be
    /// overridden per deployment) — operators that want different
    /// retry policy install a custom classifier rather than mutating
    /// this method.
    #[must_use]
    pub const fn is_retryable(self) -> bool {
        matches!(self, Self::Transient | Self::RateLimit)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn provider_network_classifies_as_transient() {
        let err = Error::provider_network("connect refused");
        assert_eq!(ToolErrorKind::classify(&err), ToolErrorKind::Transient);
        assert!(ToolErrorKind::classify(&err).is_retryable());
    }

    #[test]
    fn provider_dns_classifies_as_transient() {
        let err = Error::provider_dns("no such host");
        assert_eq!(ToolErrorKind::classify(&err), ToolErrorKind::Transient);
    }

    #[test]
    fn provider_5xx_classifies_as_transient() {
        let err = Error::provider_http(503, "down");
        assert_eq!(ToolErrorKind::classify(&err), ToolErrorKind::Transient);
        let err = Error::provider_http(502, "bad gateway");
        assert_eq!(ToolErrorKind::classify(&err), ToolErrorKind::Transient);
    }

    #[test]
    fn http_408_and_425_classify_as_transient() {
        // 408 Request Timeout, 425 Too Early — both retryable per
        // spec semantics.
        let err = Error::provider_http(408, "timeout");
        assert_eq!(ToolErrorKind::classify(&err), ToolErrorKind::Transient);
        let err = Error::provider_http(425, "too early");
        assert_eq!(ToolErrorKind::classify(&err), ToolErrorKind::Transient);
    }

    #[test]
    fn http_429_with_retry_after_classifies_as_rate_limit() {
        let err = Error::provider_http(429, "slow down").with_retry_after(Duration::from_secs(5));
        assert_eq!(ToolErrorKind::classify(&err), ToolErrorKind::RateLimit);
        assert!(ToolErrorKind::classify(&err).is_retryable());
    }

    #[test]
    fn http_429_without_retry_after_classifies_as_quota() {
        // Vendor signalling quota exhaustion typically omits
        // `Retry-After` because the cooldown is a billing cycle,
        // not a request window.
        let err = Error::provider_http(429, "monthly cap reached");
        assert_eq!(ToolErrorKind::classify(&err), ToolErrorKind::Quota);
        assert!(!ToolErrorKind::classify(&err).is_retryable());
    }

    #[test]
    fn http_401_403_classify_as_auth() {
        let err = Error::provider_http(401, "unauthorized");
        assert_eq!(ToolErrorKind::classify(&err), ToolErrorKind::Auth);
        let err = Error::provider_http(403, "forbidden");
        assert_eq!(ToolErrorKind::classify(&err), ToolErrorKind::Auth);
        assert!(!ToolErrorKind::classify(&err).is_retryable());
    }

    #[test]
    fn http_4xx_other_classifies_as_permanent() {
        let err = Error::provider_http(404, "not found");
        assert_eq!(ToolErrorKind::classify(&err), ToolErrorKind::Permanent);
        let err = Error::provider_http(422, "unprocessable");
        assert_eq!(ToolErrorKind::classify(&err), ToolErrorKind::Permanent);
        assert!(!ToolErrorKind::classify(&err).is_retryable());
    }

    #[test]
    fn invalid_request_and_serde_classify_as_validation() {
        let err = Error::invalid_request("bad input");
        assert_eq!(ToolErrorKind::classify(&err), ToolErrorKind::Validation);
        let serde_err: serde_json::Error = serde_json::from_str::<i32>("not-a-number").unwrap_err();
        let err: Error = serde_err.into();
        assert_eq!(ToolErrorKind::classify(&err), ToolErrorKind::Validation);
    }

    #[test]
    fn config_classifies_as_internal() {
        let err = Error::config("misconfigured");
        assert_eq!(ToolErrorKind::classify(&err), ToolErrorKind::Internal);
    }

    #[test]
    fn usage_limit_exceeded_classifies_as_quota() {
        use crate::run_budget::UsageLimitBreach;
        let err = Error::UsageLimitExceeded(UsageLimitBreach::Requests {
            limit: 10,
            observed: 11,
        });
        assert_eq!(ToolErrorKind::classify(&err), ToolErrorKind::Quota);
    }
}
