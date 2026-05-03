//! `Retrying<R>` — `Runnable<I, O>` adapter that retries the inner
//! runnable using a [`RetryPolicy`].
//!
//! Mirrors the semantics of
//! [`RetryLayer`](entelix_core::transports::RetryLayer) at the
//! `Runnable` layer rather than the `tower::Service` layer — pick
//! whichever boundary matches the unit you want to retry.
//! `with_retry` on a `Runnable` is the right choice for a whole
//! chain (`prompt.pipe(model).pipe(parser)`); a `RetryLayer` is the
//! right choice for a single model-or-tool service call.

use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use rand::SeedableRng;
use rand::rngs::SmallRng;

use entelix_core::ExecutionContext;
use entelix_core::error::{Error, Result};
use entelix_core::transports::RetryPolicy;

use crate::runnable::Runnable;

/// `Runnable<I, O>` adapter applying a [`RetryPolicy`] to the inner
/// runnable on every invocation.
pub struct Retrying<R, I, O>
where
    R: Runnable<I, O> + 'static,
    I: Clone + Send + 'static,
    O: Send + 'static,
{
    inner: Arc<R>,
    policy: RetryPolicy,
    _io: PhantomData<fn(I) -> O>,
}

impl<R, I, O> Retrying<R, I, O>
where
    R: Runnable<I, O> + 'static,
    I: Clone + Send + 'static,
    O: Send + 'static,
{
    /// Build with the inner runnable and a retry policy.
    pub fn new(inner: R, policy: RetryPolicy) -> Self {
        Self {
            inner: Arc::new(inner),
            policy,
            _io: PhantomData,
        }
    }
}

impl<R, I, O> std::fmt::Debug for Retrying<R, I, O>
where
    R: Runnable<I, O> + 'static,
    I: Clone + Send + 'static,
    O: Send + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Retrying")
            .field("max_attempts", &self.policy.max_attempts())
            .finish_non_exhaustive()
    }
}

/// Per-call RNG seed — system clock nanos `XOR`-mixed with a
/// process-local counter so two concurrent calls get distinct
/// jitter sequences.
fn seed_from_time() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    // u128 nanoseconds wraps once every ~584 years at u64; truncation
    // is fine — we only need uncorrelated low-order bits for jitter.
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

#[async_trait]
impl<R, I, O> Runnable<I, O> for Retrying<R, I, O>
where
    R: Runnable<I, O> + 'static,
    I: Clone + Send + 'static,
    O: Send + 'static,
{
    async fn invoke(&self, input: I, ctx: &ExecutionContext) -> Result<O> {
        let max_attempts = self.policy.max_attempts().max(1);
        let mut rng = SmallRng::seed_from_u64(seed_from_time());
        let mut attempt: u32 = 0;
        loop {
            if ctx.is_cancelled() {
                return Err(Error::Cancelled);
            }
            let cloned = input.clone();
            match self.inner.invoke(cloned, ctx).await {
                Ok(value) => return Ok(value),
                Err(err) => {
                    attempt = attempt.saturating_add(1);
                    let exhausted = attempt >= max_attempts;
                    let decision = self.policy.classifier().should_retry(&err, attempt - 1);
                    if exhausted || !decision.retry {
                        return Err(err);
                    }
                    let backoff_delay = self
                        .policy
                        .backoff()
                        .delay_for_attempt(attempt - 1, &mut rng);
                    let delay = decision
                        .after
                        .map_or(backoff_delay, |hint| hint.min(self.policy.backoff().max()));
                    let token = ctx.cancellation();
                    tokio::select! {
                        () = tokio::time::sleep(delay) => {}
                        () = token.cancelled() => return Err(Error::Cancelled),
                    }
                }
            }
        }
    }
}
