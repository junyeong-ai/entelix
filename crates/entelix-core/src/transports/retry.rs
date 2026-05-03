//! `RetryLayer` + `RetryService` — `tower::Layer<S>` middleware that
//! retries transient errors with exponential backoff and jitter.
//!
//! Wraps any `Service` whose `Error` is [`crate::error::Error`].
//! Composes uniformly on the `Service<ModelInvocation>` and
//! `Service<ToolInvocation>` paths; the same layer handles both
//! because the retry decision is error-shape-driven, not invocation-
//! shape-driven.
//!
//! ## Cancellation
//!
//! Every iteration head checks
//! [`ExecutionContext::cancellation`](crate::context::ExecutionContext)
//! and returns `Error::Cancelled` immediately if signalled — operator
//! intent always wins over the retry policy.
//!
//! ## Layering with other middleware
//!
//! Place `RetryLayer` *outside* observability middleware
//! (`OtelLayer`) so each retry attempt produces its own span — the
//! span tree shows `n` attempts, not `1` opaque envelope. Place it
//! *inside* policy gates (`PolicyLayer`) so the policy decision
//! fires once per logical call, not per attempt.
//!
//! ```text
//!   PolicyLayer  ↘
//!     RetryLayer  ↘
//!       OtelLayer  ↘
//!         <inner Service>
//! ```

use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use futures::future::BoxFuture;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use tower::{Layer, Service, ServiceExt};

use crate::backoff::{DEFAULT_MAX_ATTEMPTS, ExponentialBackoff};
use crate::error::{Error, Result};

/// Parse a `Retry-After` HTTP header value into a [`Duration`].
///
/// Per RFC 7231, the value is either an integer number of seconds
/// or an HTTP-date. Vendor APIs in this corner of the ecosystem
/// (Anthropic, OpenAI, Bedrock) all use the integer form. Returns
/// `None` for missing / malformed / zero values; `Some(Duration)`
/// is the vendor's authoritative cooldown that retry classifiers
/// honour ahead of self-jitter (invariant #17).
#[must_use]
pub fn parse_retry_after(header: Option<&http::HeaderValue>) -> Option<Duration> {
    let header = header?.to_str().ok()?;
    let secs: u64 = header.trim().parse().ok()?;
    if secs == 0 {
        return None;
    }
    Some(Duration::from_secs(secs))
}

/// Trait that classifies whether an error justifies another attempt
/// and — when the vendor supplies one — the cooldown to wait before
/// the next try.
///
/// The same trait drives `RetryLayer` *and*
/// `RunnableExt::with_fallbacks`: in both cases the question is
/// "is this error transient or permanent?". Reusing one trait keeps
/// users out of policy-selection paralysis and avoids parallel
/// taxonomies.
pub trait RetryClassifier: Send + Sync + std::fmt::Debug {
    /// Whether to attempt again, plus the optional vendor-supplied
    /// cooldown. `attempt` starts at `0` for the first failed call
    /// (i.e., before the first *retry*); the next failure passes
    /// `attempt = 1`, and so on. Implementations can gate by
    /// attempt count, error variant, kind, or any combination.
    fn should_retry(&self, error: &Error, attempt: u32) -> RetryDecision;
}

/// Decision returned by [`RetryClassifier::should_retry`]. The
/// `after` field carries the vendor's `Retry-After` hint when
/// present — `RetryService` honours it ahead of its own
/// exponential-backoff plan, capping at the configured maximum so a
/// malicious vendor cannot pin the loop forever.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetryDecision {
    /// Whether to attempt again.
    pub retry: bool,
    /// Vendor `Retry-After` hint when present. `None` means the
    /// classifier defers to the configured backoff.
    pub after: Option<Duration>,
}

impl RetryDecision {
    /// Convenience: do not retry.
    pub const STOP: Self = Self {
        retry: false,
        after: None,
    };

    /// Convenience: retry with the configured backoff.
    pub const RETRY: Self = Self {
        retry: true,
        after: None,
    };

    /// Convenience: retry after the supplied vendor cooldown.
    #[must_use]
    pub const fn retry_after(after: Duration) -> Self {
        Self {
            retry: true,
            after: Some(after),
        }
    }
}

/// Standard classifier — retries on transient HTTP / transport
/// classes:
///
/// - `Provider { status: 0, .. }` — transport / DNS / connect failure.
/// - `Provider { status: 408 | 425 | 429 | 500..=599, .. }` —
///   Request Timeout / Too Early / Too Many Requests / 5xx server.
///
/// Permanent failures (`InvalidRequest`, `Config`, `Cancelled`,
/// `DeadlineExceeded`, `Interrupted`, `Serde`, and 4xx other than
/// 408 / 425 / 429) are not retried.
#[derive(Clone, Copy, Debug, Default)]
pub struct DefaultRetryClassifier;

impl RetryClassifier for DefaultRetryClassifier {
    fn should_retry(&self, error: &Error, _attempt: u32) -> RetryDecision {
        match error {
            Error::Provider {
                kind, retry_after, ..
            } if is_transient_kind(*kind) => match retry_after {
                Some(after) => RetryDecision::retry_after(*after),
                None => RetryDecision::RETRY,
            },
            // Everything else is deterministic, caller-intent, or HITL —
            // retrying produces the same outcome (or violates intent).
            _ => RetryDecision::STOP,
        }
    }
}

const fn is_transient_kind(kind: crate::error::ProviderErrorKind) -> bool {
    use crate::error::ProviderErrorKind;
    match kind {
        // Network / TLS / DNS failures are transient by default —
        // operator routes, peer connections, name resolution all
        // recover.
        ProviderErrorKind::Network | ProviderErrorKind::Tls | ProviderErrorKind::Dns => true,
        // HTTP-class transience is the documented retry set:
        // 408 Request Timeout, 425 Too Early, 429 Too Many Requests,
        // 5xx server errors. 4xx other than these are caller-side
        // bugs and re-issuing produces the same response.
        ProviderErrorKind::Http(status) => matches!(status, 408 | 425 | 429 | 500..=599),
    }
}

/// Retry policy: how many attempts, how to space them, what to retry.
#[derive(Clone, Debug)]
pub struct RetryPolicy {
    /// Maximum total attempts (including the first call). `1`
    /// disables retry — the first failure surfaces unchanged.
    max_attempts: u32,
    /// Backoff sequence between attempts.
    backoff: ExponentialBackoff,
    /// Error → "retry?" decision.
    classifier: Arc<dyn RetryClassifier>,
}

impl RetryPolicy {
    /// Build with explicit components.
    #[must_use]
    pub fn new(
        max_attempts: u32,
        backoff: ExponentialBackoff,
        classifier: Arc<dyn RetryClassifier>,
    ) -> Self {
        Self {
            max_attempts,
            backoff,
            classifier,
        }
    }

    /// Default policy: [`DEFAULT_MAX_ATTEMPTS`] attempts,
    /// `100 ms` → `5 s` backoff with 30% jitter,
    /// [`DefaultRetryClassifier`].
    #[must_use]
    pub fn standard() -> Self {
        Self::new(
            DEFAULT_MAX_ATTEMPTS,
            ExponentialBackoff::new(Duration::from_millis(100), Duration::from_secs(5)),
            Arc::new(DefaultRetryClassifier),
        )
    }

    /// Override the attempt cap.
    #[must_use]
    pub const fn with_max_attempts(mut self, n: u32) -> Self {
        self.max_attempts = n;
        self
    }

    /// Override the backoff sequence.
    #[must_use]
    pub const fn with_backoff(mut self, backoff: ExponentialBackoff) -> Self {
        self.backoff = backoff;
        self
    }

    /// Override the classifier.
    #[must_use]
    pub fn with_classifier(mut self, classifier: Arc<dyn RetryClassifier>) -> Self {
        self.classifier = classifier;
        self
    }

    /// Borrow the configured attempt cap.
    #[must_use]
    pub const fn max_attempts(&self) -> u32 {
        self.max_attempts
    }

    /// Borrow the configured backoff.
    #[must_use]
    pub const fn backoff(&self) -> ExponentialBackoff {
        self.backoff
    }

    /// Borrow the configured classifier.
    #[must_use]
    pub fn classifier(&self) -> &Arc<dyn RetryClassifier> {
        &self.classifier
    }
}

/// Layer that adds retry semantics to any `Service<Req>` whose
/// `Error` is [`Error`].
///
/// The wrapped service must be `Clone` (each retry attempt clones a
/// fresh handle to call). `tower`'s convention is that `Clone`
/// returns a cheap reference-counted view, not a deep copy — every
/// `*Layer` in entelix follows that contract.
#[derive(Clone, Debug)]
pub struct RetryLayer {
    policy: RetryPolicy,
}

impl RetryLayer {
    /// Build with a retry policy.
    #[must_use]
    pub const fn new(policy: RetryPolicy) -> Self {
        Self { policy }
    }
}

impl<S> Layer<S> for RetryLayer {
    type Service = RetryService<S>;
    fn layer(&self, inner: S) -> Self::Service {
        RetryService {
            inner,
            policy: self.policy.clone(),
        }
    }
}

/// `Service` produced by [`RetryLayer`]. Generic over the request
/// type so a single layer drives both `ModelInvocation` and
/// `ToolInvocation` paths.
#[derive(Clone, Debug)]
pub struct RetryService<S> {
    inner: S,
    policy: RetryPolicy,
}

impl<S, Req, Resp> Service<Req> for RetryService<S>
where
    S: Service<Req, Response = Resp, Error = Error> + Clone + Send + 'static,
    S::Future: Send + 'static,
    Req: Retryable + Send + 'static,
    Resp: Send + 'static,
{
    type Response = Resp;
    type Error = Error;
    type Future = BoxFuture<'static, Result<Resp>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<()>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, request: Req) -> Self::Future {
        let inner = self.inner.clone();
        let policy = self.policy.clone();
        Box::pin(async move { run_with_retry(inner, request, policy).await })
    }
}

/// Marker that a request type can be cloned for retries.
///
/// `RetryService` clones the request once per attempt (the inner
/// service is consumed by each call). For `ModelInvocation` and
/// `ToolInvocation` the clone is cheap (`Arc`-backed `ModelRequest`
/// and JSON-`Value` body).
pub trait Retryable: Clone {
    /// Borrow the [`ExecutionContext`](crate::context::ExecutionContext)
    /// the retry loop checks for cancellation between attempts.
    fn ctx(&self) -> &crate::context::ExecutionContext;

    /// Mutable handle so [`RetryService`] can stamp an idempotency
    /// key on first entry — every clone of the request that follows
    /// shares the stamped key, so vendor-side dedupe sees one
    /// logical call across N attempts.
    fn ctx_mut(&mut self) -> &mut crate::context::ExecutionContext;
}

impl Retryable for crate::service::ModelInvocation {
    fn ctx(&self) -> &crate::context::ExecutionContext {
        &self.ctx
    }
    fn ctx_mut(&mut self) -> &mut crate::context::ExecutionContext {
        &mut self.ctx
    }
}

impl Retryable for crate::service::ToolInvocation {
    fn ctx(&self) -> &crate::context::ExecutionContext {
        &self.ctx
    }
    fn ctx_mut(&mut self) -> &mut crate::context::ExecutionContext {
        &mut self.ctx
    }
}

async fn run_with_retry<S, Req, Resp>(
    mut inner: S,
    mut request: Req,
    policy: RetryPolicy,
) -> Result<Resp>
where
    S: Service<Req, Response = Resp, Error = Error> + Clone + Send,
    S::Future: Send,
    Req: Retryable + Send,
{
    // Per-call RNG seeded from a system-time component so two
    // concurrent calls never share a jitter sequence. The RNG lives
    // for the lifetime of the call only.
    let seed = seed_from_time();
    let mut rng = SmallRng::seed_from_u64(seed);

    // Stamp an idempotency key on the request the first time we see
    // it. Every subsequent clone (one per retry attempt) inherits
    // the key, so vendor-side dedupe treats N attempts as one
    // logical call (invariant #17 — no double-charge when a client
    // timeout races a server-side success).
    request
        .ctx_mut()
        .ensure_idempotency_key(|| uuid::Uuid::new_v4().to_string());

    let max_attempts = policy.max_attempts.max(1);
    let mut attempt: u32 = 0;
    loop {
        // Deadline + cancellation check at the head of every attempt.
        // A caller with `with_timeout(200ms)` and exponential backoff
        // would otherwise sleep through the deadline and surface
        // `Provider` instead of the more specific `DeadlineExceeded`,
        // making it impossible to distinguish "upstream is sick"
        // from "we ran out of time".
        let ctx_token = request.ctx().cancellation();
        if ctx_token.is_cancelled() {
            return Err(Error::Cancelled);
        }
        if let Some(deadline) = request.ctx().deadline()
            && tokio::time::Instant::now() >= deadline
        {
            return Err(Error::DeadlineExceeded);
        }

        let cloned = request.clone();
        let result = inner.ready().await?.call(cloned).await;

        match result {
            Ok(resp) => return Ok(resp),
            Err(err) => {
                attempt = attempt.saturating_add(1);
                let exhausted = attempt >= max_attempts;
                let decision = policy.classifier.should_retry(&err, attempt - 1);
                if exhausted || !decision.retry {
                    return Err(err);
                }
                // Vendor `Retry-After` beats self-jitter (invariant
                // #17 — vendor authoritative signal wins). Cap at the
                // configured backoff cap so a malicious or stuck
                // vendor cannot pin the loop.
                let backoff_delay = policy.backoff.delay_for_attempt(attempt - 1, &mut rng);
                let delay = match decision.after {
                    Some(hint) => hint.min(policy.backoff.max()),
                    None => backoff_delay,
                };
                // If the caller's deadline lands inside this backoff
                // window, cap the sleep to the remaining budget and
                // surface `DeadlineExceeded` rather than waking up
                // past the deadline only to retry-then-deadline.
                let effective_delay = if let Some(deadline) = request.ctx().deadline() {
                    let now = tokio::time::Instant::now();
                    let remaining = deadline.saturating_duration_since(now);
                    if remaining.is_zero() {
                        return Err(Error::DeadlineExceeded);
                    }
                    delay.min(remaining)
                } else {
                    delay
                };
                let deadline_for_select = request.ctx().deadline();
                tokio::select! {
                    () = tokio::time::sleep(effective_delay) => {
                        // If we capped the sleep at the deadline,
                        // bail out instead of looping for one more
                        // doomed attempt.
                        if let Some(deadline) = deadline_for_select
                            && tokio::time::Instant::now() >= deadline
                        {
                            return Err(Error::DeadlineExceeded);
                        }
                    }
                    () = ctx_token.cancelled() => return Err(Error::Cancelled),
                }
            }
        }
    }
}

/// Seed the per-call RNG from the system clock's nanosecond
/// component XORed with a process-local counter so two calls
/// arriving in the same nanosecond still get distinct sequences.
fn seed_from_time() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    // u128 nanoseconds wraps once every ~584 years at u64; truncating
    // is fine for jitter — we only need uncorrelated low-order bits.
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| {
            let n = d.as_nanos();
            #[allow(clippy::cast_possible_truncation)]
            {
                n as u64
            }
        })
        .unwrap_or(0);
    let bump = COUNTER.fetch_add(1, Ordering::Relaxed);
    nanos ^ bump
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn default_classifier_retries_transient_http_status_codes() {
        let c = DefaultRetryClassifier;
        for status in [408_u16, 425, 429, 500, 502, 503, 504, 599] {
            let err = Error::provider_http(status, "x");
            assert!(c.should_retry(&err, 0).retry, "status {status} must retry");
        }
    }

    #[test]
    fn default_classifier_retries_transport_class_failures() {
        let c = DefaultRetryClassifier;
        assert!(
            c.should_retry(&Error::provider_network("connect refused"), 0)
                .retry
        );
        assert!(
            c.should_retry(&Error::provider_tls("handshake failed"), 0)
                .retry
        );
        assert!(
            c.should_retry(&Error::provider_dns("no such host"), 0)
                .retry
        );
    }

    #[test]
    fn default_classifier_does_not_retry_permanent_status_codes() {
        let c = DefaultRetryClassifier;
        for status in [400_u16, 401, 403, 404, 410, 422] {
            let err = Error::provider_http(status, "x");
            assert!(
                !c.should_retry(&err, 0).retry,
                "status {status} must NOT retry"
            );
        }
    }

    #[test]
    fn default_classifier_does_not_retry_caller_intent_or_programming_errors() {
        let c = DefaultRetryClassifier;
        assert!(!c.should_retry(&Error::Cancelled, 0).retry);
        assert!(!c.should_retry(&Error::DeadlineExceeded, 0).retry);
        assert!(!c.should_retry(&Error::invalid_request("nope"), 0).retry);
        assert!(!c.should_retry(&Error::config("bad"), 0).retry);
    }

    #[test]
    fn default_classifier_propagates_vendor_retry_after_hint() {
        let c = DefaultRetryClassifier;
        let err = Error::provider_http(429, "rate limited")
            .with_retry_after(Duration::from_secs(7));
        let decision = c.should_retry(&err, 0);
        assert!(decision.retry);
        assert_eq!(decision.after, Some(Duration::from_secs(7)));
    }

    #[test]
    fn ensure_idempotency_key_stamps_once_and_subsequent_calls_observe_the_same_value() {
        // RetryService relies on this contract — the first call
        // sets a fresh UUID; later calls in the same logical
        // call (one per attempt) see the same value rather than
        // generating new ones.
        use crate::context::ExecutionContext;
        let mut ctx = ExecutionContext::new();
        assert!(ctx.idempotency_key().is_none());
        let mut counter = 0u32;
        let first = ctx
            .ensure_idempotency_key(|| {
                counter += 1;
                "first-uuid".to_owned()
            })
            .to_owned();
        let second = ctx
            .ensure_idempotency_key(|| {
                counter += 1;
                "second-uuid".to_owned()
            })
            .to_owned();
        assert_eq!(first, "first-uuid");
        assert_eq!(second, "first-uuid", "stamp must be stable across calls");
        assert_eq!(counter, 1, "generator must run exactly once");
        // Cloning the ctx propagates the same key — every retry of
        // a logical call shares the stamp.
        let cloned = ctx.clone();
        assert_eq!(cloned.idempotency_key(), Some("first-uuid"));
    }

    #[test]
    fn default_classifier_does_not_attach_retry_after_when_vendor_does_not_supply_one() {
        let c = DefaultRetryClassifier;
        let err = Error::provider_http(503, "down");
        let decision = c.should_retry(&err, 0);
        assert!(decision.retry);
        assert!(decision.after.is_none());
    }

    #[test]
    fn retry_policy_standard_uses_default_max_attempts() {
        let p = RetryPolicy::standard();
        assert_eq!(p.max_attempts(), DEFAULT_MAX_ATTEMPTS);
    }

    #[test]
    fn retry_policy_overrides_compose() {
        let p = RetryPolicy::standard()
            .with_max_attempts(2)
            .with_backoff(ExponentialBackoff::new(
                Duration::from_millis(1),
                Duration::from_millis(10),
            ));
        assert_eq!(p.max_attempts(), 2);
        assert_eq!(p.backoff().base(), Duration::from_millis(1));
    }
}
