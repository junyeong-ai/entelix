//! `RetryToolLayer` — `tower::Layer<Service<ToolInvocation>>` that
//! turns the metadata-level [`RetryHint`](crate::tools::RetryHint)
//! contract into runtime retry behaviour.
//!
//! Tool authors annotate intent on
//! [`ToolMetadata::retry_hint`](crate::tools::ToolMetadata) when a
//! tool is *idempotent* and *transport-bound* (HTTP fetch, RPC call,
//! search adapter). Without this layer the metadata is documentation;
//! with it the runtime honours the hint — re-invoking on transient
//! failures up to the configured budget.
//!
//! ## Contract
//!
//! - **No hint, no retry.** Tools that have not opted in
//!   (`retry_hint == None`) pass through the layer unchanged. The
//!   default for non-idempotent tools is fail-fast, regardless of
//!   error category.
//! - **Hint present + retryable error → retry.** The layer reads
//!   `metadata.retry_hint.max_attempts` for the cap, applies the
//!   layer's `RetryClassifier` (default: matches transient errors
//!   per [`ToolErrorKind::is_retryable`](crate::ToolErrorKind)),
//!   and waits `hint.initial_backoff * 2^attempt` (jittered, capped
//!   at the layer's max-backoff) between attempts. Vendor
//!   `Retry-After` hints (`RetryDecision::after`) override the
//!   computed delay when present.
//! - **Cancellation-aware sleep.** Backoff sleeps respect
//!   [`ExecutionContext::cancellation`](crate::ExecutionContext) —
//!   a cancellation during backoff returns
//!   [`Error::Cancelled`](crate::Error::Cancelled) immediately, no
//!   final attempt.
//!
//! ## Composition order
//!
//! Wire `RetryToolLayer` *innermost* (closest to the leaf service)
//! so observability layers (`OtelLayer`, `ToolEventLayer`) emit one
//! event per retry attempt rather than one event for the entire
//! retry envelope. Mirrors the pattern transport-side
//! `RetryService` / `OtelLayer` use for model invocations.
//!
//! ```ignore
//! use entelix_core::ToolRegistry;
//! use entelix_core::tools::RetryToolLayer;
//!
//! let registry = ToolRegistry::new()
//!     .layer(RetryToolLayer::new())          // innermost
//!     .layer(my_observability_layer)         // outermost
//!     .register(my_tool)?;
//! # Ok::<(), entelix_core::Error>(())
//! ```

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures::future::BoxFuture;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use serde_json::Value;
use tower::{Layer, Service, ServiceExt};

use crate::backoff::ExponentialBackoff;
use crate::error::{Error, Result};
use crate::service::ToolInvocation;
use crate::transports::{DefaultRetryClassifier, RetryClassifier};

/// Default upper bound on the per-attempt backoff. Caps the geometric
/// growth of `hint.initial_backoff * 2^attempt` so a misconfigured
/// hint cannot pin the loop indefinitely.
pub const DEFAULT_MAX_BACKOFF: Duration = Duration::from_secs(30);

/// `tower::Layer` that retries tool dispatches per the wrapped tool's
/// [`RetryHint`](crate::tools::RetryHint) metadata.
///
/// Cloning is cheap — internal state is `Arc`-backed.
#[derive(Clone)]
pub struct RetryToolLayer {
    classifier: Arc<dyn RetryClassifier>,
    max_backoff: Duration,
}

impl RetryToolLayer {
    /// Build with the default classifier ([`DefaultRetryClassifier`])
    /// and [`DEFAULT_MAX_BACKOFF`] cap.
    #[must_use]
    pub fn new() -> Self {
        Self {
            classifier: Arc::new(DefaultRetryClassifier),
            max_backoff: DEFAULT_MAX_BACKOFF,
        }
    }

    /// Replace the [`RetryClassifier`] consulted on each failure.
    /// Operators with custom retry policy (e.g. retry only on
    /// `Transient`, ignore `RateLimit`) install their own
    /// classifier here.
    #[must_use]
    pub fn with_classifier(mut self, classifier: Arc<dyn RetryClassifier>) -> Self {
        self.classifier = classifier;
        self
    }

    /// Override the per-attempt backoff cap. The geometric growth of
    /// `hint.initial_backoff * 2^attempt` is clamped to this value
    /// before jitter is applied.
    #[must_use]
    pub const fn with_max_backoff(mut self, max: Duration) -> Self {
        self.max_backoff = max;
        self
    }
}

impl Default for RetryToolLayer {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for RetryToolLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetryToolLayer")
            .field("max_backoff", &self.max_backoff)
            .finish_non_exhaustive()
    }
}

impl<S> Layer<S> for RetryToolLayer
where
    S: Service<ToolInvocation, Response = Value, Error = Error> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Service = RetryToolService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        RetryToolService {
            inner,
            classifier: Arc::clone(&self.classifier),
            max_backoff: self.max_backoff,
        }
    }
}

/// `Service<ToolInvocation>` produced by [`RetryToolLayer`].
#[derive(Clone)]
pub struct RetryToolService<Inner> {
    inner: Inner,
    classifier: Arc<dyn RetryClassifier>,
    max_backoff: Duration,
}

impl<Inner> std::fmt::Debug for RetryToolService<Inner> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetryToolService")
            .field("max_backoff", &self.max_backoff)
            .finish_non_exhaustive()
    }
}

impl<Inner> Service<ToolInvocation> for RetryToolService<Inner>
where
    Inner: Service<ToolInvocation, Response = Value, Error = Error> + Clone + Send + 'static,
    Inner::Future: Send + 'static,
{
    type Response = Value;
    type Error = Error;
    type Future = BoxFuture<'static, Result<Value>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<()>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, invocation: ToolInvocation) -> Self::Future {
        let mut inner = self.inner.clone();
        let classifier = Arc::clone(&self.classifier);
        let max_backoff = self.max_backoff;

        Box::pin(async move {
            let hint = invocation.metadata.retry_hint;
            // Tools without an explicit hint pass through unchanged
            // — the metadata-level fail-fast contract for
            // non-idempotent tools.
            let Some(hint) = hint else {
                return inner.ready().await?.call(invocation).await;
            };

            let max_attempts = hint.max_attempts.max(1);
            // Per-tool baseline + layer-level cap → fresh backoff
            // strategy. Each invocation gets its own tuned schedule
            // tied to that tool's hint.
            let backoff = ExponentialBackoff::new(hint.initial_backoff, max_backoff);
            let mut rng = SmallRng::seed_from_u64(seed_from_time());
            let mut attempt: u32 = 0;

            loop {
                let ctx_token = invocation.ctx.cancellation();
                if ctx_token.is_cancelled() {
                    return Err(Error::Cancelled);
                }

                let cloned = invocation.clone();
                let result = inner.ready().await?.call(cloned).await;

                match result {
                    Ok(value) => return Ok(value),
                    Err(err) => {
                        attempt = attempt.saturating_add(1);
                        let exhausted = attempt >= max_attempts;
                        let decision = classifier.should_retry(&err, attempt - 1);
                        if exhausted || !decision.retry {
                            return Err(err);
                        }
                        let computed = backoff.delay_for_attempt(attempt - 1, &mut rng);
                        // Vendor hints win over self-jitter (mirrors
                        // model-side `RetryService`, invariant 17).
                        let delay = decision
                            .after
                            .map_or(computed, |hint| hint.min(max_backoff));

                        tokio::select! {
                            () = tokio::time::sleep(delay) => {}
                            () = ctx_token.cancelled() => return Err(Error::Cancelled),
                        }
                    }
                }
            }
        })
    }
}

/// Seed a per-call RNG from system clock nanoseconds XOR a process-
/// local counter — uncorrelated jitter even when two calls collide
/// in the same tick.
fn seed_from_time() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    // silent-fallback-ok: jitter seed only — `now() < UNIX_EPOCH`
    // cannot happen on a sane clock, and the per-process atomic
    // counter XORed below already breaks ties so a 0 nanos
    // contribution still yields uncorrelated low-order bits.
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| {
        let n = d.as_nanos();
        #[allow(clippy::cast_possible_truncation)]
        {
            n as u64
        }
    });
    let bump = COUNTER.fetch_add(1, Ordering::Relaxed);
    nanos ^ bump
}
